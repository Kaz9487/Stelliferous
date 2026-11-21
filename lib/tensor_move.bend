# Moving cells. A task holds the pieces it reads of a source and writes its
# target pieces by runs: a run is a line of target cells that read source
# cells one stride apart, cut where it leaves a source piece, so every copy
# reads one array. The target cells a part does not reach keep their values.
import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./storage_copy.bend as Copy
import ./traversal_partition.bend as Partition
import ./tensor_view.bend as Views
import ./tensor_pieces.bend as Pieces

# ---- The pieces a task holds ----

# Pieces in ascending order of their first cell.
type Stack<-Element: Data> is Type:
  Bottom{}
  Held{buffer: Storage.Buffer<Element>,first: U32,count: U32,below: Stack<Element>}

# The current piece on top of ahead, the pieces before it on behind, nearest
# first.
type Holding<-Element: Data> is Type:
  Holding{behind: Stack<Element>,ahead: Stack<Element>}

def stacked(~Element: Data,tree: Pieces.Pieces(~Element),rest: Stack<Element>) -> Stack<Element>:
  match tree:
    case Partition.Leaf{buffer,first,count,details}: Held{buffer,first,count,rest}
    case Partition.Fork{left,right}: stacked(~Element,left,stacked(~Element,right,rest))

# The pieces of a tree, the first current.
def held(~Element: Data,tree: Pieces.Pieces(~Element)) -> Holding<Element>:
  Holding{Bottom{},stacked(~Element,tree,Bottom{})}

# Forward past pieces that end at or before index, while a piece follows.
def advanced(~Element: Data,below: Stack<Element>,step: Bool,+index: U32,buffer: Storage.Buffer<Element>,+first: U32,+count: U32,behind: Stack<Element>) -> Holding<Element>:
  match below step:
    case Held{next,+next_first,+next_count,rest} True{}:
      advanced(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind})
    case below step: Holding{behind,Held{buffer,first,count,below}}

# Back over pieces that begin after index, while a piece precedes.
def backed(~Element: Data,behind: Stack<Element>,step: Bool,+index: U32,ahead: Stack<Element>) -> Holding<Element>:
  match behind step:
    case Held{buffer,+first,+count,rest} True{}: backed(~Element,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead})
    case behind step: Holding{behind,ahead}

def located(~Element: Data,before: Bool,+index: U32,behind: Stack<Element>,buffer: Storage.Buffer<Element>,+first: U32,+count: U32,below: Stack<Element>) -> Holding<Element>:
  match before:
    case True{}: backed(~Element,behind,True{},index,Held{buffer,first,count,below})
    case False{}: advanced(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind)

# The holding with the piece of cell index current, when it holds one.
def resolve(~Element: Data,holding: Holding<Element>,+index: U32) -> Holding<Element>:
  match holding:
    case Holding{behind,Held{buffer,+first,+count,below}}: located(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below)
    case Holding{behind,Bottom{}}: backed(~Element,behind,True{},index,Bottom{})

# ---- Runs ----

# Cells of a run of count cells from offset, stride apart (backward when
# back), that lie in the piece from first of span cells. The reach counts the
# cells of the piece past offset in the direction of the run.
def inside(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32) -> U32:
  +holds = U32.is_le(first,offset) && U32.is_lt(offset,(first + span : U32))
  +reach = Bool.pick(U32,back,(offset - first : U32),(first + span - offset - 1 : U32))
  Bool.pick(U32,holds,Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,(reach / U32.max(stride,1) + 1 : U32))),0)

def directed(+back: Bool,+stride: U32) -> U32:
  Bool.pick(U32,back,(0 - stride : U32),stride)

# The axes of a run: a contiguous run is copied from its offset alone.
def run_axes(contiguous: Bool,+cells: U32,+stride: U32,+back: Bool) -> List<&2,Views.Axis>:
  match contiguous:
    case True{}: []
    case False{}: [Views.Axis{cells,directed(back,stride)}]

def copied_run(~Element: Data,+cells: U32,+local: U32,+stride: U32,+back: Bool,+position: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  +contiguous = U32.is_eq(stride,1) && Bool.not(back)
  Copy.view(~Element,contiguous,cells,Views.View{run_axes(contiguous,cells,stride,back),local},position,input,output)

# A run after a step: the cells it has left, from source cell offset into
# target from position on, and whether none are left.
type Stepped<-Element: Data> is Type:
  Stepped{holding: Holding<Element>,target: Storage.Buffer<Element>,count: U32,offset: U32,position: U32,done: Bool}

def stepped(~Element: Data,holding: Holding<Element>,target: Storage.Buffer<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+cells: U32) -> Stepped<Element>:
  +left = (count - cells : U32)
  Stepped{holding,target,left,(offset + cells * directed(back,stride) : U32),(position + cells : U32),U32.is_zero(left)}

def stepped_in(~Element: Data,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+cells: U32,behind: Stack<Element>,+first: U32,+span: U32,below: Stack<Element>,
  pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Stepped<Element>:
  (input,output) = pair
  stepped(~Element,Holding{behind,Held{input,first,span,below}},output,count,offset,stride,back,position,cells)

# The part of a run that the current piece holds, copied.
def step(~Element: Data,holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,target: Storage.Buffer<Element>) -> Stepped<Element>:
  match holding:
    case Holding{behind,Held{buffer,+first,+span,below}}:
      +cells = inside(count,offset,stride,back,first,span)
      stepped_in(~Element,count,offset,stride,back,position,cells,behind,first,span,below,
        copied_run(~Element,cells,(offset - first : U32),stride,back,position,buffer,target))
    case Holding{behind,Bottom{}}: stepped(~Element,Holding{behind,Bottom{}},target,count,offset,stride,back,position,0)

# The rest of a run after a step. A run stops early only where no held piece
# holds its cell.
def run_on(~Element: Data,fuel: Nat,+stride: U32,+back: Bool,last: Stepped<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  match fuel last:
    case 0n Stepped{holding,target,count,offset,position,done}: (holding,target)
    case 1n+rest Stepped{holding,target,count,offset,position,True{}}: (holding,target)
    case 1n+rest Stepped{holding,target,+count,+offset,+position,False{}}:
      run_on(~Element,rest,stride,back,step(~Element,resolve(~Element,holding,offset),count,offset,stride,back,position,target))

# count cells from source cell offset, stride apart, into target from
# position on.
def run(~Element: Data,fuel: Nat,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,holding: Holding<Element>,target: Storage.Buffer<Element>) ->
  Holding<Element> & Storage.Buffer<Element>:
  run_on(~Element,fuel,stride,back,Stepped{holding,target,count,offset,position,U32.is_zero(count)})

# ---- Parts ----

# One axis of a part: its extent, the source stride (backward when back) and
# the target stride.
type Axis is Data:
  Axis{extent: U32,source: U32,back: Bool,target: U32}

# A box of target cells and the source cells it reads: every axis with its
# strides, the first source cell in the source tensor and the first target
# cell in the target piece. The last axis is the run.
type Part is Data:
  Part{axes: List<&2,Axis>,source: U32,target: U32}

def same_direction(+left: Bool,+right: Bool) -> Bool:
  (left && right) || (Bool.not(left) && Bool.not(right))

def chains(+outer: Axis,+inner: Axis) -> Bool:
  Axis{+extent,+source,+back,+target} = outer
  Axis{+inner_extent,+inner_source,+inner_back,+inner_target} = inner
  U32.is_eq(target,(inner_target * inner_extent : U32)) && U32.is_eq(source,(inner_source * inner_extent : U32)) && same_direction(back,inner_back)

def merged(+outer: Axis,+inner: Axis) -> Axis:
  Axis{+extent,+source,+back,+target} = outer
  Axis{+inner_extent,+inner_source,+inner_back,+inner_target} = inner
  Axis{(extent * inner_extent : U32),inner_source,inner_back,inner_target}

def joined_axis(chain: Bool,+outer: Axis,+inner: Axis,rest: List<&2,Axis>) -> List<&2,Axis>:
  match chain:
    case True{}: merged(outer,inner) <> rest
    case False{}: outer <> inner <> rest

def joined_coalesced(+axis: Axis,rest: List<&2,Axis>) -> List<&2,Axis>:
  match rest:
    case Nil{}: [axis]
    case Con{+inner,tail}: joined_axis(chains(axis,inner),axis,inner,tail)

def kept_coalesced(single: Bool,+axis: Axis,rest: List<&2,Axis>) -> List<&2,Axis>:
  match single:
    case True{}: rest
    case False{}: joined_coalesced(axis,rest)

# Adjacent axes whose strides chain on both sides, merged into one. An axis
# of one index steps nowhere and is left out.
def coalesced(axes: List<&2,Axis>) -> List<&2,Axis>:
  match axes:
    case Nil{}: []
    case Con{Axis{+extent,+source,+back,+target},rest}: kept_coalesced(U32.is_eq(extent,1),Axis{extent,source,back,target},coalesced(rest))

# The source and target offsets of row index of the axes before the run.
type Offsets is Data:
  Offsets{source: U32,target: U32}

def rows_of(axes: List<&2,Axis>) -> U32:
  match axes:
    case Nil{}: 1
    case Con{Axis{+extent,source,back,target},rest}: (extent * rows_of(rest) : U32)

def row_offsets(axes: List<&2,Axis>,+row: U32,+source: U32,+target: U32) -> Offsets:
  match axes:
    case Nil{}: Offsets{source,target}
    case Con{Axis{+extent,+stride,+back,+target_stride},+rest}:
      +inner = U32.max(rows_of(rest),1)
      +index = (row / inner % U32.max(extent,1) : U32)
      row_offsets(rest,(row % inner : U32),(source + index * directed(back,stride) : U32),(target + index * target_stride : U32))

# The axes before the run: the last of them steps in the inner loop, the ones
# before it give the offsets of every block of its rows.
type Nest is Data:
  Nest{outer: List<&2,Axis>,inner: Axis}

def nest_prepended(+axis: Axis,nest: Nest) -> Nest:
  Nest{+outer,+inner} = nest
  Nest{axis <> outer,inner}

def nest_of(axes: List<&2,Axis>) -> Nest:
  match axes:
    case Nil{}: Nest{[],Axis{1,0,False{},0}}
    case Con{+axis,Nil{}}: Nest{[],axis}
    case Con{+axis,+rest}: nest_prepended(axis,nest_of(rest))

def row_run(~Element: Data,+run_axis: Axis,+source: U32,+position: U32,holding: Holding<Element>,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  Axis{+extent,+stride,+back,target_stride} = run_axis
  run(~Element,U32.to_nat(extent),extent,source,stride,back,position,holding,target)

# The rows of one block, each a step of the inner axis after the last.
def inner_rows(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+run_axis: Axis,+source: U32,+position: U32,pair: Holding<Element> & Storage.Buffer<Element>) ->
  Holding<Element> & Storage.Buffer<Element>:
  match fuel:
    case 0n: pair
    case 1n+rest:
      (holding,target) = pair
      inner_rows(~Element,rest,step,inner_target,run_axis,(source + step : U32),(position + inner_target : U32),row_run(~Element,run_axis,source,position,holding,target))

def block_rows(~Element: Data,+inner: Axis,+run_axis: Axis,offsets: Offsets,pair: Holding<Element> & Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  Axis{+extent,+stride,+back,+target_stride} = inner
  Offsets{+source,+position} = offsets
  inner_rows(~Element,U32.to_nat(extent),directed(back,stride),target_stride,run_axis,source,position,pair)

# Blocks from row on of the axes before the inner one.
def rows(~Element: Data,fuel: Nat,+row: U32,+outer: List<&2,Axis>,+inner: Axis,+run_axis: Axis,+source: U32,+target_offset: U32,pair: Holding<Element> & Storage.Buffer<Element>) ->
  Holding<Element> & Storage.Buffer<Element>:
  match fuel:
    case 0n: pair
    case 1n+rest: rows(~Element,rest,(row + 1 : U32),outer,inner,run_axis,source,target_offset,block_rows(~Element,inner,run_axis,row_offsets(outer,row,source,target_offset),pair))

def nested_rows(~Element: Data,+nest: Nest,+run_axis: Axis,+source: U32,+target_offset: U32,pair: Holding<Element> & Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  Nest{+outer,+inner} = nest
  rows(~Element,U32.to_nat(rows_of(outer)),0,outer,inner,run_axis,source,target_offset,pair)

type Split is Data:
  Split{outer: List<&2,Axis>,run: Axis}

def split_prepended(+axis: Axis,split: Split) -> Split:
  Split{+outer,+run} = split
  Split{axis <> outer,run}

# The last axis is the run when it steps one target cell. Otherwise every axis
# steps rows and each run is one cell.
def split_dense(dense: Bool,+axis: Axis) -> Split:
  match dense:
    case True{}: Split{[],axis}
    case False{}: Split{[axis],Axis{1,0,False{},1}}

def split_single(+axis: Axis) -> Split:
  Axis{+extent,+source,+back,+target} = axis
  split_dense(U32.is_eq(target,1),Axis{extent,source,back,target})

def split_last(axes: List<&2,Axis>) -> Split:
  match axes:
    case Nil{}: Split{[],Axis{1,0,False{},1}}
    case Con{+axis,Nil{}}: split_single(axis)
    case Con{+axis,+rest}: split_prepended(axis,split_last(rest))

# Whether one piece holds every source cell of the axes before the run and
# the run from offset on: each forward axis, from the last cell the axes
# before it reach, is one step of a run that takes all its cells.
def run_fits(+run_axis: Axis,+offset: U32,+first: U32,+span: U32) -> Bool:
  Axis{+extent,+stride,+back,target} = run_axis
  Bool.not(back) && U32.is_le(extent,inside(extent,offset,stride,False{},first,span))

def fits_from(axes: List<&2,Axis>,+run_axis: Axis,+offset: U32,+first: U32,+span: U32) -> Bool:
  match axes:
    case Nil{}: run_fits(run_axis,offset,first,span)
    case Con{Axis{+extent,+stride,+back,target},rest}:
      run_fits(Axis{extent,stride,back,target},offset,first,span) &&
        fits_from(rest,run_axis,(offset + (U32.max(extent,1) - 1) * stride : U32),first,span)

def piece_run(~Element: Data,+run_axis: Axis,+local: U32,+position: U32,input: Storage.Buffer<Element>,target: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Axis{+extent,+stride,back,target_stride} = run_axis
  copied_run(~Element,extent,local,stride,False{},position,input,target)

# The rows of one block of a part that one piece holds, each a step of the
# inner axis after the last, local cells from the first of the piece.
def piece_inner(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+run_axis: Axis,+local: U32,+position: U32,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match fuel:
    case 0n: pair
    case 1n+rest:
      (input,target) = pair
      piece_inner(~Element,rest,step,inner_target,run_axis,(local + step : U32),(position + inner_target : U32),piece_run(~Element,run_axis,local,position,input,target))

def piece_block(~Element: Data,+inner: Axis,+run_axis: Axis,offsets: Offsets,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Axis{+extent,+stride,+back,+target_stride} = inner
  Offsets{+local,+position} = offsets
  piece_inner(~Element,U32.to_nat(extent),directed(back,stride),target_stride,run_axis,local,position,pair)

# Blocks from row on of a part that one piece holds.
def piece_rows(~Element: Data,fuel: Nat,+row: U32,+outer: List<&2,Axis>,+inner: Axis,+run_axis: Axis,+local: U32,+target_offset: U32,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match fuel:
    case 0n: pair
    case 1n+rest: piece_rows(~Element,rest,(row + 1 : U32),outer,inner,run_axis,local,target_offset,piece_block(~Element,inner,run_axis,row_offsets(outer,row,local,target_offset),pair))

def piece_nested(~Element: Data,+nest: Nest,+run_axis: Axis,+local: U32,+target_offset: U32,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Nest{+outer,+inner} = nest
  piece_rows(~Element,U32.to_nat(rows_of(outer)),0,outer,inner,run_axis,local,target_offset,pair)

def piece_returned(~Element: Data,behind: Stack<Element>,+first: U32,+span: U32,below: Stack<Element>,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Holding<Element> & Storage.Buffer<Element>:
  (input,target) = pair
  (Holding{behind,Held{input,first,span,below}},target)

def piece_part(~Element: Data,inside: Bool,+nest: Nest,+run_axis: Axis,+source: U32,+target_offset: U32,behind: Stack<Element>,buffer: Storage.Buffer<Element>,+first: U32,+span: U32,
  below: Stack<Element>,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  match inside:
    case True{}: piece_returned(~Element,behind,first,span,below,piece_nested(~Element,nest,run_axis,(source - first : U32),target_offset,(buffer,target)))
    case False{}: nested_rows(~Element,nest,run_axis,source,target_offset,(Holding{behind,Held{buffer,first,span,below}},target))

def current_part(~Element: Data,+outer: List<&2,Axis>,+run_axis: Axis,+source: U32,+target_offset: U32,holding: Holding<Element>,target: Storage.Buffer<Element>) ->
  Holding<Element> & Storage.Buffer<Element>:
  match holding:
    case Holding{behind,Held{buffer,+first,+span,below}}:
      piece_part(~Element,fits_from(outer,run_axis,source,first,span),nest_of(outer),run_axis,source,target_offset,behind,buffer,first,span,below,target)
    case Holding{behind,Bottom{}}: nested_rows(~Element,nest_of(outer),run_axis,source,target_offset,(Holding{behind,Bottom{}},target))

def split_part(~Element: Data,split: Split,+source: U32,+target_offset: U32,holding: Holding<Element>,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  Split{+outer,+run_axis} = split
  current_part(~Element,outer,run_axis,source,target_offset,resolve(~Element,holding,source),target)

# The target box of a part written from the held source pieces.
def part(~Element: Data,+own: Part,holding: Holding<Element>,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  Part{+axes,+source,+target_offset} = own
  split_part(~Element,split_last(coalesced(axes)),source,target_offset,holding,target)

def parts_on(~Element: Data,listed: List<&2,Part>,pair: Holding<Element> & Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  match listed:
    case Nil{}: pair
    case Con{+own,rest}:
      (holding,target) = pair
      parts_on(~Element,rest,part(~Element,own,holding,target))

def parts(~Element: Data,listed: List<&2,Part>,holding: Holding<Element>,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  parts_on(~Element,listed,(holding,target))

# ---- Cells ----

def held_read(~Element: Data,behind: Stack<Element>,+first: U32,+span: U32,below: Stack<Element>,pair: Storage.Buffer<Element> & Element) -> Holding<Element> & Element:
  (buffer,value) = pair
  (Holding{behind,Held{buffer,first,span,below}},value)

def read_held(~Element: Data,held: Bool,+index: U32,+fallback: Element,behind: Stack<Element>,buffer: Storage.Buffer<Element>,+first: U32,+span: U32,below: Stack<Element>) ->
  Holding<Element> & Element:
  match held:
    case True{}: held_read(~Element,behind,first,span,below,Storage.read(~Element,buffer,(index - first : U32)))
    case False{}: (Holding{behind,Held{buffer,first,span,below}},fallback)

def read_current(~Element: Data,holding: Holding<Element>,+index: U32,+fallback: Element) -> Holding<Element> & Element:
  match holding:
    case Holding{behind,Held{buffer,+first,+span,below}}:
      read_held(~Element,U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)),index,fallback,behind,buffer,first,span,below)
    case Holding{behind,Bottom{}}: (Holding{behind,Bottom{}},fallback)

# The source cell index, or fallback where no held piece holds it.
def read(~Element: Data,holding: Holding<Element>,+index: U32,+fallback: Element) -> Holding<Element> & Element:
  read_current(~Element,resolve(~Element,holding,index),index,fallback)

# ---- Bands ----

# A buffer kept between bands, with room for capacity cells: a band
# assembled from several held pieces is copied into it.
type Spare<-Element: Data> is Type:
  Spare{buffer: Storage.Buffer<Element>,capacity: U32}

# A spare with no room yet.
def spare(~Element: Data,+fill: Element) -> Spare<Element>:
  Spare{Allocation.allocate(~Element,0n,fill,0n),0}

# A band lent from a held piece leaves a hole to put it back in and keeps the
# spare; a band assembled into the spare leaves the holding whole.
type Lent<-Element: Data> is Type:
  Hole{behind: Stack<Element>,below: Stack<Element>,first: U32,count: U32,spare: Spare<Element>}
  Assembled{holding: Holding<Element>,capacity: U32}

# The cells [first, first + count) of the held pieces in one buffer, from
# offset on.
type Band<-Element: Data> is Type:
  Band{lent: Lent<Element>,buffer: Storage.Buffer<Element>,offset: U32}

# The cells [first, first + count) of the held pieces into target from its
# first cell on.
def copied_into(~Element: Data,holding: Holding<Element>,+first: U32,+count: U32,target: Storage.Buffer<Element>) -> Holding<Element> & Storage.Buffer<Element>:
  run(~Element,U32.to_nat(count),count,first,1,False{},0,holding,target)

# A copy of the cells [first, first + count) of the held pieces.
def copied(~Element: Data,holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) -> Holding<Element> & Storage.Buffer<Element>:
  copied_into(~Element,holding,first,count,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))

def assembled_band(~Element: Data,+capacity: U32,pair: Holding<Element> & Storage.Buffer<Element>) -> Band<Element>:
  (holding,buffer) = pair
  Band{Assembled{holding,capacity},buffer,0}

# The spare when it has room for count cells, otherwise a new buffer of count.
def roomy(~Element: Data,enough: Bool,+count: U32,+fill: Element,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  match enough:
    case True{}: buffer
    case False{}: Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))

def assembled(~Element: Data,+first: U32,+count: U32,+fill: Element,holding: Holding<Element>,spare: Spare<Element>) -> Band<Element>:
  Spare{buffer,+capacity} = spare
  assembled_band(~Element,U32.max(capacity,count),copied_into(~Element,holding,first,count,roomy(~Element,U32.is_le(count,capacity),count,fill,buffer)))

def band_when(~Element: Data,within: Bool,+first: U32,+count: U32,+fill: Element,behind: Stack<Element>,buffer: Storage.Buffer<Element>,+start: U32,+span: U32,below: Stack<Element>,
  spare: Spare<Element>) -> Band<Element>:
  match within:
    case True{}: Band{Hole{behind,below,start,span,spare},buffer,(first - start : U32)}
    case False{}: assembled(~Element,first,count,fill,Holding{behind,Held{buffer,start,span,below}},spare)

def band_current(~Element: Data,+first: U32,+count: U32,+fill: Element,holding: Holding<Element>,spare: Spare<Element>) -> Band<Element>:
  match holding:
    case Holding{behind,Held{buffer,+start,+span,below}}:
      band_when(~Element,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),first,count,fill,behind,buffer,start,span,below,spare)
    case Holding{behind,Bottom{}}: assembled(~Element,first,count,fill,Holding{behind,Bottom{}},spare)

# The band of cells [first, first + count): the held piece that holds all of
# it, lent, or the spare holding a copy of the parts the pieces hold.
def band(~Element: Data,holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,spare: Spare<Element>) -> Band<Element>:
  band_current(~Element,first,count,fill,resolve(~Element,holding,first),spare)

# The holding after a band, a lent piece back in its hole, and the spare.
def returned(~Element: Data,lent: Lent<Element>,buffer: Storage.Buffer<Element>) -> Holding<Element> & Spare<Element>:
  match lent:
    case Hole{behind,below,+first,+count,spare}: (Holding{behind,Held{buffer,first,count,below}},spare)
    case Assembled{holding,+capacity}: (holding,Spare{buffer,capacity})

# The holding after a band whose spare is not kept.
def returned_holding(~Element: Data,lent: Lent<Element>,buffer: Storage.Buffer<Element>) -> Holding<Element>:
  Pair.fst(Holding<Element>,Spare<Element>,returned(~Element,lent,buffer))

# The held pieces as a tree, in order.
def unstacked(~Element: Data,stack: Stack<Element>,tree: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  match stack:
    case Bottom{}: tree
    case Held{buffer,+first,+count,below}: unstacked(~Element,below,Partition.Fork{tree,Partition.Leaf{buffer,first,count,Unit{}}})

# The pieces of behind, nearest first, onto ahead: in the other order.
def onto_ahead(~Element: Data,behind: Stack<Element>,ahead: Stack<Element>) -> Stack<Element>:
  match behind:
    case Bottom{}: ahead
    case Held{buffer,+first,+count,below}: onto_ahead(~Element,below,Held{buffer,first,count,ahead})

def first_leaf(~Element: Data,stack: Stack<Element>,+fill: Element) -> Pieces.Pieces(~Element) & Stack<Element>:
  match stack:
    case Bottom{}: (Partition.Leaf{Allocation.allocate(~Element,0n,fill,0n),0,0,Unit{}},Bottom{})
    case Held{buffer,+first,+count,below}: (Partition.Leaf{buffer,first,count,Unit{}},below)

def released_stack(~Element: Data,pair: Pieces.Pieces(~Element) & Stack<Element>) -> Pieces.Pieces(~Element):
  (tree,rest) = pair
  unstacked(~Element,rest,tree)

# The held pieces back as a tree in ascending order.
def released(~Element: Data,holding: Holding<Element>,+fill: Element) -> Pieces.Pieces(~Element):
  Holding{behind,ahead} = holding
  released_stack(~Element,first_leaf(~Element,onto_ahead(~Element,behind,ahead),fill))
