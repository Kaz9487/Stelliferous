# One packer for every window. A tile holds eight consecutive output
# positions; its entries follow the reduction order: channel, kernel row,
# kernel column. A call first transposes its input cells into planes, one
# for each column phase and channel: plane row r holds the input row of the
# window's first output row times the stride plus r, its columns every
# stride-th input column from the phase on, +0 outside the input. The rows
# lie end to end, each a whole number of tiles wide, so the positions of one
# output row are consecutive plane values for every tap. A plane is kept as
# pairs of a tile and the next one, sixteen words behind one index. A packed
# tile is one pass for each output row its positions touch: the pass reads a
# pair shifted by a lane count fixed for the pass, the first pass stores the
# tile and a later one moves its lanes into it.
import Base
import ./traversal_loop.bend as Loop
import ./kernel_packing_samples.bend as PackingSamples
import ./kernel_tile.bend as Tile
import ./math_fp32.bend as FP32

# ---- Reading eight lanes of an input row ----

# A sample of an input element: the value itself, or one lane of a cell.
def identity(value: F32) -> F32:
  value

def field_of(~Element: Data,~field: Element -> F32,read: Array<Element> & Element) -> Array<Element> & F32:
  (input,element) = read
  (input,field(element))

def masked_field(~Element: Data,inside: Bool,pair: Array<Element> & F32) -> Array<Element> & F32:
  (input_storage,value) = pair
  (input_storage,PackingSamples.masked(value,inside))

# The element of a channel at a pixel lies step elements a pixel from the
# element of the channel at pixel zero.
def pixel(~Element: Data,~field: Element -> F32,input_storage: Array<Element>,+input_row: U32,+input_column: U32,+channel_base: U32,+input_height: U32,+input_width: U32,+step: U32) ->
  Array<Element> & F32:
  masked_field(~Element,U32.is_lt(input_row,input_height) && U32.is_lt(input_column,input_width),
    field_of(~Element,~field,Array.get(Element,input_storage,(channel_base + (input_row * input_width + input_column) * step : U32))))

type PixelBlock is Data:
  PixelBlock{row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,step: U32}

def read_block_pixel(~Element: Data,~field: Element -> F32,input: Array<Element>,context: PixelBlock,index: Nat) -> Array<Element> & F32:
  PixelBlock{row,column,channel_base,height,width,+stride,step} = context
  pixel(~Element,~field,input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width,step)

# Eight lanes of one input row, stride columns apart.
def block(~Element: Data,~field: Element -> F32,input: Array<Element>,row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,next_row: U32,offset: U32,output_width: U32,
  kernel_column: U32,padding: U32,step: U32) -> Array<Element> & Tile.Tile8:
  +row = row
  +column = column
  +channel_base = channel_base
  +height = height
  +width = width
  +step = step
  Tile.read_tail(~Array<Element>,~PixelBlock,~read_block_pixel(~Element,~field),pixel(~Element,~field,input,row,column,channel_base,height,width,step),
    PixelBlock{row,column,channel_base,height,width,stride,step})

# ---- The window ----

# The input cells of a call from base on, height rows of width pixels,
# groups cells a pixel; output position q of an output output_width wide
# reads input row (q / output_width) * stride + kernel row + row_origin and
# input column (q % output_width) * stride + kernel column + column_origin.
type Stencil is Data:
  Stencil{base: U32,height: U32,width: U32,tap_rows: U32,tap_columns: U32,stride: U32,row_origin: U32,column_origin: U32,output_width: U32,inner: U32,groups: U32}

def square(height: U32,width: U32,+kernel: U32,stride: U32,+padding: U32,output_width: U32,inner: U32,groups: U32) -> Stencil:
  Stencil{0,height,width,kernel,kernel,stride,(0 - padding : U32),(0 - padding : U32),output_width,inner,groups}

def channels_of(window: Stencil) -> U32:
  Stencil{base,height,width,+tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  (inner / (tap_rows * tap_columns) : U32)

# ---- Planes ----

# The planes of output positions [start, start + count): their first output
# row, the plane rows, the width of a row in values (the output width and the
# column taps past it, whole tiles) and the elements of a plane, one for each
# tile of its rows.
type Planes is Data:
  Planes{first_output_row: U32,rows: U32,width: U32,elements: U32}

def planes_of(window: Stencil,+start: U32,+count: U32) -> Planes:
  Stencil{base,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +first = (start / output_width : U32)
  +rows = (((start + count - 1) / output_width - first) * stride + tap_rows : U32)
  +row_width = ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8 : U32)
  Planes{first,rows,row_width,(rows * row_width / 8 : U32)}

# The planes of a call on count positions from start; a call on no positions
# has none.
def call_planes_when(empty: Bool,+window: Stencil,+start: U32,+count: U32) -> Planes:
  match empty:
    case True{}: Planes{0,0,0,0}
    case False{}: planes_of(window,start,count)

def call_planes(+window: Stencil,+start: U32,+count: U32) -> Planes:
  call_planes_when(U32.is_zero(count),window,start,count)

# The pairs of all planes: one plane for each column phase and channel.
def plane_pairs(+window: Stencil,planes: Planes) -> U32:
  Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  Planes{first_output_row,rows,row_width,+elements} = planes
  (stride * channels_of(window) * elements : U32)

# Room for the planes of any count consecutive positions.
def plane_bound(+window: Stencil,+count: U32) -> U32:
  Stencil{base,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +rows = (((count + output_width - 1) / output_width) * stride + tap_rows : U32)
  (stride * channels_of(window) * (rows * ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8) / 8) : U32)

# ---- The transpose ----

# One plane: the first step reads tile 0, step j after it reads tile j and
# writes the pair of tiles j - 1 and j at element j - 1. The channels of an
# element lie side by side, so the elements of a plane are channels apart.
type Rowing is Type:
  Rowing{input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,previous: Tile.Tile8,column: U32,write: U32,row: U32,in_row: U32,row_tiles: U32,first_column: U32,
    channel_base: U32,height: U32,width: U32,stride: U32,groups: U32,channels: U32}

def row_moved(input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,tile: Tile.Tile8,+column: U32,write: U32,+row: U32,+in_row: U32,+row_tiles: U32,+first_column: U32,
  channel_base: U32,height: U32,width: U32,+stride: U32,groups: U32,channels: U32) -> Rowing:
  +wrap = U32.is_eq((in_row + 1 : U32),row_tiles)
  Rowing{input,pairs,tile,Bool.pick(U32,wrap,first_column,(column + 8 * stride : U32)),write,Bool.pick(U32,wrap,(row + 1 : U32),row),Bool.pick(U32,wrap,0,(in_row + 1 : U32)),
    row_tiles,first_column,channel_base,height,width,stride,groups,channels}

def row_kept(input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,previous: Tile.Tile8,+tile: Tile.Tile8,+column: U32,+write: U32,+row: U32,+in_row: U32,+row_tiles: U32,
  +first_column: U32,+channel_base: U32,+height: U32,+width: U32,+stride: U32,+groups: U32,+channels: U32) -> Rowing:
  row_moved(input,Array.set(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile}),tile,column,(write + channels : U32),row,in_row,row_tiles,first_column,channel_base,height,
    width,stride,groups,channels)

def row_read(pairs: Array<Tile.Tile16>,previous: Tile.Tile8,+column: U32,+write: U32,+row: U32,+in_row: U32,+row_tiles: U32,+first_column: U32,+channel_base: U32,
  +height: U32,+width: U32,+stride: U32,+groups: U32,+channels: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Rowing:
  (input,tile) = read
  row_kept(input,pairs,previous,tile,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels)

def row_step(~lane: Tile.Tile8 -> F32,state: Rowing) -> Rowing:
  Rowing{input,pairs,previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  row_read(pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,
    block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups))

def row_first_read(pairs: Array<Tile.Tile16>,previous: Tile.Tile8,+column: U32,+write: U32,+row: U32,+in_row: U32,+row_tiles: U32,+first_column: U32,+channel_base: U32,
  +height: U32,+width: U32,+stride: U32,+groups: U32,+channels: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Rowing:
  (input,tile) = read
  row_moved(input,pairs,tile,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels)

def row_first(~lane: Tile.Tile8 -> F32,state: Rowing) -> Rowing:
  Rowing{input,pairs,previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = state
  row_first_read(pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,
    block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups))

def row_run(~lane: Tile.Tile8 -> F32,count: Nat,state: Rowing) -> Rowing:
  Loop.run(~Rowing,~row_step(~lane),count,row_first(~lane,state))

# A channel reads lane channel % 8 of the cells of its group; the lane is
# chosen once for the plane.
def row_low(low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match low first:
    case True{} True{}: row_run(~Tile.lane0,count,state)
    case True{} False{}: row_run(~Tile.lane1,count,state)
    case False{} True{}: row_run(~Tile.lane2,count,state)
    case False{} False{}: row_run(~Tile.lane3,count,state)

def row_high(low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match low first:
    case True{} True{}: row_run(~Tile.lane4,count,state)
    case True{} False{}: row_run(~Tile.lane5,count,state)
    case False{} True{}: row_run(~Tile.lane6,count,state)
    case False{} False{}: row_run(~Tile.lane7,count,state)

def row_lane(half: Bool,low: Bool,first: Bool,count: Nat,state: Rowing) -> Rowing:
  match half:
    case True{}: row_low(low,first,count,state)
    case False{}: row_high(low,first,count,state)

# Plane phase * channels + channel, in turn.
type Transposing is Type:
  Transposing{input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,plane: U32,window: Stencil,planes: Planes}

def plane_done(+plane: U32,+window: Stencil,+planes: Planes,done: Rowing) -> Transposing:
  Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = done
  Transposing{input,pairs,(plane + 1 : U32),window,planes}

def plane_step(state: Transposing) -> Transposing:
  Transposing{input,pairs,+plane,+window,+planes} = state
  Stencil{+base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,output_width,inner,+groups} = window
  Planes{+first_output_row,rows,+row_width,+elements} = planes
  +channels = channels_of(window)
  +channel = (plane % channels : U32)
  +phase = (plane / channels : U32)
  +lane = (channel % 8 : U32)
  plane_done(plane,window,planes,row_lane(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),
    Rowing{input,pairs,Tile.filled(0.0),(phase + column_origin : U32),(phase * elements * channels + channel : U32),(first_output_row * stride + row_origin : U32),0,
      (row_width / 8 : U32),(phase + column_origin : U32),(base + channel / 8 : U32),height,width,stride,groups,channels}))

def transposed(state: Transposing) -> Array<Tile.Tile8> & Array<Tile.Tile16>:
  Transposing{input,pairs,plane,window,planes} = state
  (input,pairs)

# The planes of a call: the input comes back unchanged.
def transpose(+window: Stencil,+planes: Planes,input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>) -> Array<Tile.Tile8> & Array<Tile.Tile16>:
  Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  transposed(Loop.run(~Transposing,~plane_step,U32.to_nat((stride * channels_of(window) : U32)),Transposing{input,pairs,0,window,planes}))

# ---- Shifts ----

# Lanes shift to shift + 7 of a pair.
def shifted0(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  a

def shifted1(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a1,a2,a3,a4,a5,a6,a7,b0}

def shifted2(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a2,a3,a4,a5,a6,a7,b0,b1}

def shifted3(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a3,a4,a5,a6,a7,b0,b1,b2}

def shifted4(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a4,a5,a6,a7,b0,b1,b2,b3}

def shifted5(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a5,a6,a7,b0,b1,b2,b3,b4}

def shifted6(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a6,a7,b0,b1,b2,b3,b4,b5}

def shifted7(pair: Tile.Tile16) -> Tile.Tile8:
  Tile.Tiles{a,b} = pair
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile.Tile{a7,b0,b1,b2,b3,b4,b5,b6}

# Lanes from lane on take the pass's tile. A word select, so both tiles are
# read whole and every bit is kept.
def lane_mask(+lane: U32,+index: U32) -> U32:
  Bool.pick(U32,U32.is_le(lane,index),4294967295,0)

def merged(+lane: U32,kept: Tile.Tile8,tile: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{x0,x1,x2,x3,x4,x5,x6,x7} = kept
  Tile.Tile{y0,y1,y2,y3,y4,y5,y6,y7} = tile
  Tile.Tile{FP32.selected(lane_mask(lane,0),y0,x0),FP32.selected(lane_mask(lane,1),y1,x1),FP32.selected(lane_mask(lane,2),y2,x2),FP32.selected(lane_mask(lane,3),y3,x3),
    FP32.selected(lane_mask(lane,4),y4,x4),FP32.selected(lane_mask(lane,5),y5,x5),FP32.selected(lane_mask(lane,6),y6,x6),FP32.selected(lane_mask(lane,7),y7,x7)}

# ---- A pass: the positions of a unit in one output row, for one kernel column ----

# Entries (channel, kernel row) of one kernel column, the kernel row
# innermost. The read steps one plane row a kernel row; a new channel starts
# one past the previous channel's start. The first pass stores its tiles, a
# later one merges its lanes from lane on into them.
type Passing is Type:
  Passing{pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>,start: U32,read: U32,kernel_row: U32,write: U32,lane: U32,tap_rows: U32,tap_columns: U32,row_step: U32}

def advanced(pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>,+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,
  +row_step: U32) -> Passing:
  +wrap = U32.is_eq((kernel_row + 1 : U32),tap_rows)
  Passing{pairs,tiles,Bool.pick(U32,wrap,(start + 1 : U32),start),Bool.pick(U32,wrap,(start + 1 : U32),(read + row_step : U32)),Bool.pick(U32,wrap,0,(kernel_row + 1 : U32)),
    Bool.pick(U32,wrap,(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),(write + tap_columns : U32)),lane,tap_rows,tap_columns,row_step}

def first_stored(~shift: Tile.Tile16 -> Tile.Tile8,tiles: Array<Tile.Tile8>,+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,
  +row_step: U32,pair_read: Array<Tile.Tile16> & Tile.Tile16) -> Passing:
  (pairs,pair) = pair_read
  advanced(pairs,Array.set(Tile.Tile8,tiles,write,shift(pair)),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step)

def first_step(~shift: Tile.Tile16 -> Tile.Tile8,state: Passing) -> Passing:
  Passing{pairs,tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  first_stored(~shift,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,Array.get(Tile.Tile16,pairs,read))

def merge_stored(pairs: Array<Tile.Tile16>,+tile: Tile.Tile8,+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,
  +row_step: U32,old: Array<Tile.Tile8> & Tile.Tile8) -> Passing:
  (tiles,kept) = old
  advanced(pairs,Array.set(Tile.Tile8,tiles,write,merged(lane,kept,tile)),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step)

def merge_read(~shift: Tile.Tile16 -> Tile.Tile8,tiles: Array<Tile.Tile8>,+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,
  +row_step: U32,pair_read: Array<Tile.Tile16> & Tile.Tile16) -> Passing:
  (pairs,pair) = pair_read
  merge_stored(pairs,shift(pair),start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,Array.get(Tile.Tile8,tiles,write))

def merge_step(~shift: Tile.Tile16 -> Tile.Tile8,state: Passing) -> Passing:
  Passing{pairs,tiles,+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = state
  merge_read(~shift,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,Array.get(Tile.Tile16,pairs,read))

def pass_run(~shift: Tile.Tile16 -> Tile.Tile8,first: Bool,count: Nat,state: Passing) -> Passing:
  match first:
    case True{}: Loop.run(~Passing,~first_step(~shift),count,state)
    case False{}: Loop.run(~Passing,~merge_step(~shift),count,state)

def pass_low(low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match low odd:
    case True{} False{}: pass_run(~shifted0,first,count,state)
    case True{} True{}: pass_run(~shifted1,first,count,state)
    case False{} False{}: pass_run(~shifted2,first,count,state)
    case False{} True{}: pass_run(~shifted3,first,count,state)

def pass_high(low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match low odd:
    case True{} False{}: pass_run(~shifted4,first,count,state)
    case True{} True{}: pass_run(~shifted5,first,count,state)
    case False{} False{}: pass_run(~shifted6,first,count,state)
    case False{} True{}: pass_run(~shifted7,first,count,state)

def pass_shift(half: Bool,low: Bool,odd: Bool,first: Bool,count: Nat,state: Passing) -> Passing:
  match half:
    case True{}: pass_low(low,odd,first,count,state)
    case False{}: pass_high(low,odd,first,count,state)

type Arrays is Type:
  Arrays{pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>}

def passed(state: Passing) -> Arrays:
  Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  Arrays{pairs,tiles}

# ---- Units ----

# The passes of a unit for one kernel column: pass m starts at lane, its
# lane 0 at plane offset base. The next output row lies stride plane rows
# further, its first position one output row on.
type Passes is Type:
  Passes{arrays: Arrays,pass: U32,lane: U32,base: U32,second_lane: U32,kernel_column: U32,unit: U32,window: Stencil,planes: Planes}

def pass_step(state: Passes) -> Passes:
  Passes{arrays,+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = state
  Arrays{pairs,tiles} = arrays
  Stencil{base_cell,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,+inner,groups} = window
  Planes{first_output_row,rows,+row_width,+elements} = planes
  +channels = channels_of(window)
  +shift = (base % 8 : U32)
  +read = (((kernel_column % stride) * elements + base / 8) * channels : U32)
  Passes{passed(pass_shift(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),U32.to_nat((channels * tap_rows : U32)),
      Passing{pairs,tiles,read,read,0,(unit * inner + kernel_column : U32),lane,tap_rows,tap_columns,(row_width / 8 * channels : U32)})),
    (pass + 1 : U32),Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32)),(base + stride * row_width - output_width : U32),second_lane,kernel_column,unit,
    window,planes}

def passes_done(state: Passes) -> Arrays:
  Passes{arrays,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  arrays

# The unit at position: output row oy, column ox. Lane output_width - ox
# starts its second output row, every output_width lanes another.
def kernel_column_step(+kernel_column: U32,+window: Stencil,+planes: Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  Planes{+first_output_row,rows,+row_width,elements} = planes
  +column = (position % output_width : U32)
  +second_lane = (output_width - column : U32)
  +count = Bool.pick(U32,U32.is_le(8,second_lane),1,(1 + (8 - second_lane + output_width - 1) / output_width : U32))
  passes_done(Loop.run(~Passes,~pass_step,U32.to_nat(count),
    Passes{arrays,0,0,((position / output_width - first_output_row) * stride * row_width + column + kernel_column / stride : U32),second_lane,kernel_column,unit,window,planes}))

def kernel_columns(remaining: Nat,+kernel_column: U32,+window: Stencil,+planes: Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  match remaining:
    case 0n: arrays
    case 1n+rest: kernel_columns(rest,(kernel_column + 1 : U32),window,planes,position,unit,kernel_column_step(kernel_column,window,planes,position,unit,arrays))

def units(remaining: Nat,+window: Stencil,+planes: Planes,+position: U32,+unit: U32,arrays: Arrays) -> Arrays:
  match remaining:
    case 0n: arrays
    case 1n+rest:
      Stencil{base,height,width,tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
      units(rest,window,planes,(position + 8 : U32),(unit + 1 : U32),kernel_columns(U32.to_nat(tap_columns),0,window,planes,position,unit,arrays))

def packed(arrays: Arrays) -> Array<Tile.Tile16> & Array<Tile.Tile8>:
  Arrays{pairs,tiles} = arrays
  (pairs,tiles)

# count entries, inner per tile, of output positions from start, read from the
# planes of a range that holds them; tiles are written from zero.
def pack(+window: Stencil,+planes: Planes,count: Nat,start: U32,pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>) -> Array<Tile.Tile16> & Array<Tile.Tile8>:
  Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  packed(units(Nat.div(count,U32.to_nat(inner)),window,planes,start,0,Arrays{pairs,tiles}))
