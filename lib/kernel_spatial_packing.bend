# One packer for every output width. A tile holds eight consecutive output
# positions; its entries follow the reduction order: channel, kernel row,
# kernel column. Only a new tile divides: it finds the output row and column
# of its first position. A tile inside one or two output rows runs three
# nested loops whose counters give the input row and column of lane 0, and
# reads the lanes of each row at one stride. A tile that spans more rows (an
# output narrower than eight) locates every lane from its position, entry by
# entry.
import Base
import ./traversal_loop.bend as Loop
import ./kernel_packing_samples.bend as PackingSamples
import ./kernel_tile.bend as Tile

def masked(inside: Bool, pair: Array<F32> & F32) -> Array<F32> & F32:
  (input_storage, value) = pair
  (input_storage, PackingSamples.masked(value, inside))

def pixel(input_storage: Array<F32>, +input_row: U32, +input_column: U32, +channel_base: U32, +input_height: U32, +input_width: U32) -> Array<F32> & F32:
  masked(U32.is_lt(input_row, input_height) && U32.is_lt(input_column, input_width), Array.get(F32, input_storage, (channel_base + input_row * input_width + input_column : U32)))

type PixelBlock is Data:
  PixelBlock{row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32}

def read_block_pixel(input: Array<F32>,context: PixelBlock,index: Nat) -> Array<F32> & F32:
  PixelBlock{row,column,channel_base,height,width,+stride} = context
  pixel(input,row,(column + U32.from_nat(index) * stride : U32),channel_base,height,width)

# Eight lanes of one output row.
def block(input: Array<F32>,row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,next_row: U32,offset: U32,output_width: U32,kernel_column: U32,padding: U32) -> Array<F32> & Tile.Tile8:
  +row = row
  +column = column
  +channel_base = channel_base
  +height = height
  +width = width
  Tile.read_tail(~Array<F32>,~PixelBlock,~read_block_pixel,pixel(input,row,column,channel_base,height,width),PixelBlock{row,column,channel_base,height,width,stride})

# Lane 0 is output column `offset`. A lane past the last output column lies
# in the next output row, whose input row is `next_row`.
type WrapBlock is Data:
  WrapBlock{row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,next_row: U32,offset: U32,output_width: U32,kernel_column: U32,padding: U32}

def wrap_pixel(same: Bool,input: Array<F32>,row: U32,column: U32,channel_base: U32,height: U32,width: U32,+stride: U32,next_row: U32,offset: U32,output_width: U32,
  kernel_column: U32,padding: U32,+lane: U32) -> Array<F32> & F32:
  match same:
    case True{}: pixel(input,row,(column + lane * stride : U32),channel_base,height,width)
    case False{}: pixel(input,next_row,((offset + lane - output_width) * stride + kernel_column - padding : U32),channel_base,height,width)

def wrap_lane(input: Array<F32>,row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,next_row: U32,+offset: U32,+output_width: U32,
  kernel_column: U32,padding: U32,+lane: U32) -> Array<F32> & F32:
  wrap_pixel(U32.is_lt((offset + lane : U32),output_width),input,row,column,channel_base,height,width,stride,next_row,offset,output_width,kernel_column,padding,lane)

def read_wrap_pixel(input: Array<F32>,context: WrapBlock,index: Nat) -> Array<F32> & F32:
  WrapBlock{row,column,channel_base,height,width,stride,next_row,offset,output_width,kernel_column,padding} = context
  wrap_lane(input,row,column,channel_base,height,width,stride,next_row,offset,output_width,kernel_column,padding,U32.from_nat(index))

def wrap_block(input: Array<F32>,row: U32,column: U32,channel_base: U32,height: U32,width: U32,stride: U32,next_row: U32,offset: U32,output_width: U32,kernel_column: U32,padding: U32) -> Array<F32> & Tile.Tile8:
  +row = row
  +column = column
  +channel_base = channel_base
  +height = height
  +width = width
  Tile.read_tail(~Array<F32>,~WrapBlock,~read_wrap_pixel,pixel(input,row,column,channel_base,height,width),
    WrapBlock{row,column,channel_base,height,width,stride,next_row,offset,output_width,kernel_column,padding})

def Block() -> Type:
  Array<F32> -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> U32 -> Array<F32> & Tile.Tile8

# The entries of a tile whose lane 0 is output column `offset` of output row
# `row` of an output `output_width` wide. block reads the eight lanes of an entry.
type Stencil is Data:
  Stencil{base: U32,height: U32,width: U32,tap_rows: U32,tap_columns: U32,stride: U32,row_origin: U32,column_origin: U32,output_width: U32,inner: U32}

def square(height: U32,width: U32,+kernel: U32,stride: U32,+padding: U32,output_width: U32,inner: U32) -> Stencil:
  Stencil{0,height,width,kernel,kernel,stride,(0 - padding : U32),(0 - padding : U32),output_width,inner}

type RowState<ownership,-InputStorage: Kind(ownership),-OutputStorage: Kind(ownership)> is Kind(ownership):
  RowTile{input: InputStorage,output: OutputStorage,index: U32,channel: U32,kernel_row: U32,kernel_column: U32,row: U32,offset: U32,
    window: Stencil,tap_rows: Nat,tap_columns: Nat}

def RowTiles() -> Type:
  RowState<&1,Array<F32>,Array<Tile.Tile8>>

def store_block(output: Array<Tile.Tile8>,+index: U32,channel: U32,kernel_row: U32,kernel_column: U32,row: U32,offset: U32,window: Stencil,
  tap_rows: Nat,tap_columns: Nat,pair: Array<F32> & Tile.Tile8) -> RowTiles():
  (input,value) = pair
  RowTile{input,Array.set(Tile.Tile8,output,index,value),(index + 1 : U32),channel,kernel_row,(kernel_column + 1 : U32),row,offset,window,tap_rows,tap_columns}

def column_step(~block: Block(),state: RowTiles()) -> RowTiles():
  RowTile{input,output,index,+channel,+kernel_row,+kernel_column,+row,+offset,+window,tap_rows,tap_columns} = state
  Stencil{+base,+height,+width,tap_height,tap_width,+stride,+row_origin,+column_origin,+output_width,inner} = window
  store_block(output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns,
    block(input,(row * stride + kernel_row + row_origin : U32),(offset * stride + kernel_column + column_origin : U32),(base + channel * height * width : U32),height,width,stride,
      ((row + 1) * stride + kernel_row + row_origin : U32),offset,output_width,kernel_column,(0 - column_origin : U32)))

def next_kernel_row(state: RowTiles()) -> RowTiles():
  RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns} = state
  RowTile{input,output,index,channel,(kernel_row + 1 : U32),0,row,offset,window,tap_rows,tap_columns}

def kernel_row_step(~block: Block(),state: RowTiles()) -> RowTiles():
  RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,+tap_columns} = state
  next_kernel_row(Loop.run(~RowTiles(),~column_step(~block),tap_columns,RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns}))

def next_channel(state: RowTiles()) -> RowTiles():
  RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns} = state
  RowTile{input,output,index,(channel + 1 : U32),0,0,row,offset,window,tap_rows,tap_columns}

def channel_step(~block: Block(),state: RowTiles()) -> RowTiles():
  RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,+tap_rows,tap_columns} = state
  next_channel(Loop.run(~RowTiles(),~kernel_row_step(~block),tap_rows,RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns}))

# The entries of a tile that crosses an output row, by reduction index.
type CrossingState<ownership,-InputStorage: Kind(ownership),-OutputStorage: Kind(ownership)> is Kind(ownership):
  Crossing{input: InputStorage,output: OutputStorage,index: U32,reduction: U32,position: U32,window: Stencil}

def Crossings() -> Type:
  CrossingState<&1,Array<F32>,Array<Tile.Tile8>>

def store_crossing(output: Array<Tile.Tile8>,+index: U32,reduction: U32,position: U32,window: Stencil,
  pair: Array<F32> & Tile.Tile8) -> Crossings():
  (input,value) = pair
  Crossing{input,Array.set(Tile.Tile8,output,index,value),(index + 1 : U32),(reduction + 1 : U32),position,window}

def crossing_step(state: Crossings()) -> Crossings():
  Crossing{input,output,index,+reduction,+position,+window} = state
  Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,inner} = window
  store_crossing(output,index,reduction,position,window,
    Tile.read_tail(~Array<F32>,~PackingSamples.WindowPixelBlock,~PackingSamples.read_window_pixel,
      PackingSamples.window_pixel(input,position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width),
      PackingSamples.window_pixel_block(position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width)))

# The tiles from output position `position`, eight positions each.
type TileState<ownership,-InputStorage: Kind(ownership),-OutputStorage: Kind(ownership)> is Kind(ownership):
  Tiles{input: InputStorage,output: OutputStorage,index: U32,position: U32,window: Stencil,channels: Nat,tap_rows: Nat,tap_columns: Nat,inner: Nat}

def Tiled() -> Type:
  TileState<&1,Array<F32>,Array<Tile.Tile8>>

def after_row_tile(position: U32,channels: Nat,inner: Nat,state: RowTiles()) -> Tiled():
  RowTile{input,output,index,channel,kernel_row,kernel_column,row,offset,window,tap_rows,tap_columns} = state
  Tiles{input,output,index,(position + 8 : U32),window,channels,tap_rows,tap_columns,inner}

def after_crossing(channels: Nat,tap_rows: Nat,tap_columns: Nat,inner: Nat,state: Crossings()) -> Tiled():
  Crossing{input,output,index,reduction,position,window} = state
  Tiles{input,output,index,(position + 8 : U32),window,channels,tap_rows,tap_columns,inner}

# A tile inside one output row, one that ends in the next row, and any other.
def tile(within: Bool,wraps: Bool,input: Array<F32>,output: Array<Tile.Tile8>,index: U32,+position: U32,+window: Stencil,
  +channels: Nat,tap_rows: Nat,tap_columns: Nat,+inner: Nat) -> Tiled():
  match within wraps:
    case True{} wraps:
      Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+output_width,inner_word} = window
      after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~block),channels,
        RowTile{input,output,index,0,0,0,(position / output_width : U32),(position % output_width : U32),window,tap_rows,tap_columns}))
    case False{} True{}:
      Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+output_width,inner_word} = window
      after_row_tile(position,channels,inner,Loop.run(~RowTiles(),~channel_step(~wrap_block),channels,
        RowTile{input,output,index,0,0,0,(position / output_width : U32),(position % output_width : U32),window,tap_rows,tap_columns}))
    case False{} False{}:
      after_crossing(channels,tap_rows,tap_columns,inner,Loop.run(~Crossings(),~crossing_step,inner,Crossing{input,output,index,0,position,window}))

# Eight positions from output column c stay in one output row when c + 8 is at
# most the output width, and in two when it is at most twice the width.
def tile_step(state: Tiled()) -> Tiled():
  Tiles{input,output,index,+position,+window,channels,tap_rows,tap_columns,inner} = state
  Stencil{base,height,width,tap_height,tap_width,stride,row_origin,column_origin,+output_width,inner_word} = window
  tile(U32.is_le((position % output_width + 8 : U32),output_width),U32.is_le((position % output_width + 8 : U32),(output_width + output_width : U32)),
    input,output,index,position,window,channels,tap_rows,tap_columns,inner)

def finish(state: Tiled()) -> Array<F32> & Array<Tile.Tile8>:
  Tiles{input,output,index,position,window,channels,tap_rows,tap_columns,inner} = state
  (input,output)

# count entries, inner per tile, from output position start.
def pack_window(+window: Stencil,count: Nat,start: U32,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Stencil{base,height,width,+tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,+inner} = window
  finish(Loop.run(~Tiled(),~tile_step,Nat.div(count,U32.to_nat(inner)),
    Tiles{input,output,0,start,window,U32.to_nat((inner / (tap_rows * tap_columns) : U32)),U32.to_nat(tap_rows),U32.to_nat(tap_columns),U32.to_nat(inner)}))

def pack(count: Nat,start: U32,input: Array<F32>,output: Array<Tile.Tile8>,height: U32,width: U32,kernel: U32,stride: U32,+padding: U32,output_width: U32,
  inner: U32) -> Array<F32> & Array<Tile.Tile8>:
  pack_window(square(height,width,kernel,stride,padding,output_width,inner),count,start,input,output)

# One packing call shared by raw convolution and certified Buffer entry points.
type Call is Data:
  Call{count: Nat,start: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32,inner: U32}

def run(call: Call,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Call{count,start,h,w,k,s,p,ow,inner} = call
  pack(count,start,input,output,h,w,k,s,p,ow,inner)
