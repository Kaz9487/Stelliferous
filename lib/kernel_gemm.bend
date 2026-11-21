# One GEMM kernel: a block of up to eight rows at eight lanes. Each cell folds
# k in order from +0, then finishes once. Which tiles each step reads, how
# the rows are written and which blocks run are the caller's parameters.
import Base
import ./kernel_tile.bend as Tile
import ./math_choice.bend as Choice
import ./traversal_loop.bend as Iteration
import ./traversal_array.bend as Traversal

# Eight row accumulators of eight lanes, one flat record.
type Accumulators is Data:
  Accumulators{r0: Tile.Tile8,r1: Tile.Tile8,r2: Tile.Tile8,r3: Tile.Tile8,r4: Tile.Tile8,r5: Tile.Tile8,r6: Tile.Tile8,r7: Tile.Tile8}

# Rows r < rows of a block are stored from base + r * row_stride: cell
# (r, l) at l * lane_stride further for l < lanes (lane writer), or the row as
# one tile whose lanes at or above lanes are +0 (tile writer).
type Store is Data:
  Store{base: U32,row_stride: U32,lane_stride: U32,rows: U32,lanes: U32}

# One block: depth steps, each reading the lane and row tiles its cursor
# points to. Row r (lane writer) or lane l (tile writer) takes the bias at
# bias_start + r or bias_start + l.
type Job<-Cursor: Data> is Data:
  Job{bias_start: U32,depth: Nat,cursor: Cursor,store: Store}

# Row tiles hold one value for each of the eight rows of a block, lane tiles
# one for each of its eight lanes. The output holds elements of the writer:
# F32 cells or Tile8 rows.
type Operands<-E: Data,-Bias: Type> is Type:
  Operands{row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,bias: Bias,output: Array<E>}

def row_tiles_of(~E: Data,~Bias: Type,operands: Operands<E,Bias>) -> Array<Tile.Tile8>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  row_tiles

def lane_tiles_of(~E: Data,~Bias: Type,operands: Operands<E,Bias>) -> Array<Tile.Tile8>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  lane_tiles

def bias_of(~E: Data,~Bias: Type,operands: Operands<E,Bias>) -> Bias:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  bias

def output_of(~E: Data,~Bias: Type,operands: Operands<E,Bias>) -> Array<E>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  output

# The k loop state: every owner, the cursor, the job fields and the
# accumulators are fields of one record; a nested owner record or a longer
# helper chain makes every k step a real call.
type Loop<-Cursor: Data,-E: Data,-Bias: Type> is Type:
  Loop{row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,bias: Bias,output: Array<E>,
    cursor: Cursor,bias_start: U32,store: Store,accumulators: Accumulators}

def row0(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a0

def row1(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a1

def row2(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a2

def row3(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a3

def row4(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a4

def row5(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a5

def row6(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a6

def row7(accumulators: Accumulators) -> Tile.Tile8:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  a7

def zero() -> Accumulators:
  +z = Tile.filled(0.0)
  Accumulators{z,z,z,z,z,z,z,z}

# A step reads the lane tile and the row tile at these addresses; the cursor
# is the next step's.
type Step<-Cursor: Data> is Data:
  Step{lane_address: U32,row_address: U32,cursor: Cursor}

# A reader gives the address of one operand at the current step and the
# cursor of the next. Direct: consecutive tiles from the first.
def direct(cursor: U32) -> U32 & U32:
  +cursor = cursor
  (cursor,(cursor + 1 : U32))

# A grid of entries: +1 along a row, row_jump after the last column of a row,
# output_jump after the last row.
type Entries is Data:
  Entries{address: U32,column: U32,row: U32,columns: U32,rows: U32,row_jump: U32,output_jump: U32}

# The entries after one: along the row, at the start of the next row, or
# after the last row. Only the next entries are built (rule F14).
def entry_next(along: Bool,down: Bool,+address: U32,+column: U32,+row: U32,+columns: U32,+rows: U32,+row_jump: U32,+output_jump: U32) -> Entries:
  match along down:
    case True{} down: Entries{(address + 1 : U32),(column + 1 : U32),row,columns,rows,row_jump,output_jump}
    case False{} True{}: Entries{(address + row_jump : U32),0,(row + 1 : U32),columns,rows,row_jump,output_jump}
    case False{} False{}: Entries{(address + output_jump : U32),0,0,columns,rows,row_jump,output_jump}

def entry(cursor: Entries) -> U32 & Entries:
  Entries{+address,+column,+row,+columns,+rows,+row_jump,+output_jump} = cursor
  (address,entry_next(U32.is_lt((column + 1 : U32),columns),U32.is_lt((row + 1 : U32),rows),address,column,row,columns,rows,row_jump,output_jump))

# One reader for each operand.
type Both<-Lane: Data,-Row: Data> is Data:
  Both{lane: Lane,row: Row}

def stepped(~Lane: Data,~Row: Data,lane: U32 & Lane,row: U32 & Row) -> Step<Both<Lane,Row>>:
  (lane_address,lane_cursor) = lane
  (row_address,row_cursor) = row
  Step{lane_address,row_address,Both{lane_cursor,row_cursor}}

def both(~Lane: Data,~Row: Data,~lane: Lane -> U32 & Lane,~row: Row -> U32 & Row,cursor: Both<Lane,Row>) -> Step<Both<Lane,Row>>:
  Both{first,second} = cursor
  stepped(~Lane,~Row,lane(first),row(second))

def Direct() -> Data:
  Both<U32,U32>

# Both operands consecutive.
def directs(cursor: Direct()) -> Step<Direct()>:
  both(~U32,~U32,~direct,~direct,cursor)

def unbiased(bias: Array<F32>,row: U32) -> Array<F32> & F32:
  (bias,0.0)

def plus(value: F32,bias: F32) -> F32:
  (value + bias : F32)

def kept(value: F32,bias: F32) -> F32:
  value

# Every row takes the step; rows at or above store.rows are never stored.
# The product of a lane value and a row value: lane first or row first, as
# the specification of the contraction multiplies them.
def lane_times(lane: F32,row: F32) -> F32:
  (lane * row : F32)

def row_times(lane: F32,row: F32) -> F32:
  (row * lane : F32)

def update(~times: F32 -> F32 -> F32,accumulators: Accumulators,+lane_tile: Tile.Tile8,row_tile: Tile.Tile8) -> Accumulators:
  Accumulators{a0,a1,a2,a3,a4,a5,a6,a7} = accumulators
  Tile.Tile{r0,r1,r2,r3,r4,r5,r6,r7} = row_tile
  Accumulators{Tile.update(~times,a0,lane_tile,r0),Tile.update(~times,a1,lane_tile,r1),Tile.update(~times,a2,lane_tile,r2),Tile.update(~times,a3,lane_tile,r3),
    Tile.update(~times,a4,lane_tile,r4),Tile.update(~times,a5,lane_tile,r5),Tile.update(~times,a6,lane_tile,r6),Tile.update(~times,a7,lane_tile,r7)}

def with_row_tile(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,lane_tiles: Array<Tile.Tile8>,bias: Bias,output: Array<E>,cursor: Cursor,+row: U32,
  store: Store,accumulators: Accumulators,lane_tile: Tile.Tile8,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop<Cursor,E,Bias>:
  (row_tiles,row_tile) = read
  Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,update(~times,accumulators,lane_tile,row_tile)}

def with_lane_tile(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,row_tiles: Array<Tile.Tile8>,bias: Bias,output: Array<E>,+row_address: U32,cursor: Cursor,+row: U32,
  store: Store,accumulators: Accumulators,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop<Cursor,E,Bias>:
  (lane_tiles,lane_tile) = read
  with_row_tile(~Cursor,~E,~Bias,~times,lane_tiles,bias,output,cursor,row,store,accumulators,lane_tile,Array.get(Tile.Tile8,row_tiles,row_address))

def with_cursor(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,bias: Bias,output: Array<E>,+row: U32,
  store: Store,accumulators: Accumulators,read: Step<Cursor>) -> Loop<Cursor,E,Bias>:
  Step{lane_address,row_address,cursor} = read
  with_lane_tile(~Cursor,~E,~Bias,~times,row_tiles,bias,output,row_address,cursor,row,store,accumulators,Array.get(Tile.Tile8,lane_tiles,lane_address))

# One ordered k step: the lane tile, then the row tile, then the products.
def step(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Step<Cursor>,~times: F32 -> F32 -> F32,state: Loop<Cursor,E,Bias>) -> Loop<Cursor,E,Bias>:
  Loop{row_tiles,lane_tiles,bias,output,cursor,+row,store,accumulators} = state
  with_cursor(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,row,store,accumulators,advance(cursor))

# Every cell is read and written back, with its own value only when inside:
# straight-line code in which each accumulator has one consumer (rule F13).
def kept_or(~E: Data,inside: Bool,+offset: U32,value: E,read: Array<E> & E) -> Array<E>:
  (output,stored) = read
  Array.set(E,output,offset,Choice.chosen(~E,inside,value,stored))

def put(~E: Data,inside: Bool,output: Array<E>,+offset: U32,value: E) -> Array<E>:
  kept_or(~E,inside,offset,value,Array.get(E,output,offset))

# A writer stores the finished rows of a block and gives both owners back.
type Written<-E: Data,-Bias: Type> is Type:
  Written{bias: Bias,output: Array<E>}

# Lane l of a block's biases is the bias at start + l; the eight are read
# once per block.
def bias_lane(~fetch: Array<F32> -> U32 -> Array<F32> & F32,bias: Array<F32>,start: U32,lane: Nat) -> Array<F32> & F32:
  fetch(bias,(start + U32.from_nat(lane) : U32))

def bias_lanes(~fetch: Array<F32> -> U32 -> Array<F32> & F32,bias: Array<F32>,+start: U32) -> Array<F32> & Tile.Tile8:
  Tile.read_tail(~Array<F32>,~U32,~bias_lane(~fetch),fetch(bias,start),start)

# A row of the tile writer: lane l closes with the bias of lane l, lanes at or
# above count are +0.
def tile_finished(~close: F32 -> F32 -> F32,+count: U32,+biases: Tile.Tile8,+values: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{Choice.chosen(~F32,U32.is_lt(0,count),close(Tile.lane0(values),Tile.lane0(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(1,count),close(Tile.lane1(values),Tile.lane1(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(2,count),close(Tile.lane2(values),Tile.lane2(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(3,count),close(Tile.lane3(values),Tile.lane3(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(4,count),close(Tile.lane4(values),Tile.lane4(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(5,count),close(Tile.lane5(values),Tile.lane5(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(6,count),close(Tile.lane6(values),Tile.lane6(biases)),0.0),
    Choice.chosen(~F32,U32.is_lt(7,count),close(Tile.lane7(values),Tile.lane7(biases)),0.0)}

def tiles_finished(~Bias: Type,~close: F32 -> F32 -> F32,+base: U32,+row_stride: U32,+rows: U32,+count: U32,+accumulators: Accumulators,output: Array<Tile.Tile8>,
  read: Bias & Tile.Tile8) -> Written<Tile.Tile8,Bias>:
  (bias,+biases) = read
  output = put(~Tile.Tile8,U32.is_lt(0,rows),output,base,tile_finished(~close,count,biases,row0(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(1,rows),output,(base + row_stride : U32),tile_finished(~close,count,biases,row1(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(2,rows),output,(base + 2 * row_stride : U32),tile_finished(~close,count,biases,row2(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(3,rows),output,(base + 3 * row_stride : U32),tile_finished(~close,count,biases,row3(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(4,rows),output,(base + 4 * row_stride : U32),tile_finished(~close,count,biases,row4(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(5,rows),output,(base + 5 * row_stride : U32),tile_finished(~close,count,biases,row5(accumulators)))
  output = put(~Tile.Tile8,U32.is_lt(6,rows),output,(base + 6 * row_stride : U32),tile_finished(~close,count,biases,row6(accumulators)))
  Written{bias,put(~Tile.Tile8,U32.is_lt(7,rows),output,(base + 7 * row_stride : U32),tile_finished(~close,count,biases,row7(accumulators)))}

# The tile writer: each row below store.rows is one tile.
def write_tiles(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~close: F32 -> F32 -> F32,+bias_start: U32,+store: Store,+accumulators: Accumulators,
  written: Written<Tile.Tile8,Array<F32>>) -> Written<Tile.Tile8,Array<F32>>:
  Store{+base,+row_stride,lane_stride,+rows,+count} = store
  Written{bias,output} = written
  tiles_finished(~Array<F32>,~close,base,row_stride,rows,count,accumulators,output,bias_lanes(~fetch,bias,bias_start))

def operands_written(~E: Data,row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,written: Written<E,Array<F32>>) -> Operands<E,Array<F32>>:
  Written{bias,output} = written
  Operands{row_tiles,lane_tiles,bias,output}

# One straight-line finish after the k loop: each accumulator has one
# consumer (rule F13).
def finish(~Cursor: Data,~E: Data,~write: @+bias_start: U32 -> @+store: Store -> @+accumulators: Accumulators -> Written<E,Array<F32>> -> Written<E,Array<F32>>,state: Loop<Cursor,E,Array<F32>>) -> Operands<E,Array<F32>>:
  Loop{row_tiles,lane_tiles,bias,output,cursor,+row,+store,+accumulators} = state
  operands_written(~E,row_tiles,lane_tiles,write(row,store,accumulators,Written{bias,output}))

def looped(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Step<Cursor>,~times: F32 -> F32 -> F32,+job: Job<Cursor>,start: Operands<E,Bias> & Accumulators) -> Loop<Cursor,E,Bias>:
  Job{+row,depth,cursor,+store} = job
  (operands,accumulators) = start
  Operands{row_tiles,lane_tiles,bias,output} = operands
  Iteration.run(~Loop<Cursor,E,Bias>,~step(~Cursor,~E,~Bias,~advance,~times),depth,Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,accumulators})

# One block of eight rows: the k loop from +0, then the finish.
def block(~Cursor: Data,~E: Data,~advance: Cursor -> Step<Cursor>,~times: F32 -> F32 -> F32,~write: @+bias_start: U32 -> @+store: Store -> @+accumulators: Accumulators -> Written<E,Array<F32>> -> Written<E,Array<F32>>,+job: Job<Cursor>,operands: Operands<E,Array<F32>>) -> Operands<E,Array<F32>>:
  finish(~Cursor,~E,~write,looped(~Cursor,~E,~Array<F32>,~advance,~times,job,(operands,zero())))

def job_step(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Job<Cursor>,~advance: Cursor -> Step<Cursor>,~times: F32 -> F32 -> F32,~write: @+bias_start: U32 -> @+store: Store -> @+accumulators: Accumulators -> Written<E,Array<F32>> -> Written<E,Array<F32>>,index: U32,plan: Plan,operands: Operands<E,Array<F32>>) -> Operands<E,Array<F32>>:
  block(~Cursor,~E,~advance,~times,~write,job_at(plan,index),operands)

def held(~State: Type,state: State) -> State:
  state

# Two call sites retain this non-flat worklist segment (rule F12), so the
# register-allocated MAC loop stays separate from its caller.
def segment(~State: Type,~Plan: Data,~update: U32 -> Plan -> State -> State,depth: Nat,count: Nat,plan: Plan,state: State) -> State:
  match depth:
    case 0n: Traversal.indexed_fold(~State,~Plan,~update,count,0,plan,state)
    case 1n+rest: held(~State,segment(~State,~Plan,~update,rest,count,plan,state))
# Jobs 0 to count - 1 of the plan, in order.
def run(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Job<Cursor>,~advance: Cursor -> Step<Cursor>,~times: F32 -> F32 -> F32,~write: @+bias_start: U32 -> @+store: Store -> @+accumulators: Accumulators -> Written<E,Array<F32>> -> Written<E,Array<F32>>,count: Nat,plan: Plan,operands: Operands<E,Array<F32>>) -> Operands<E,Array<F32>>:
  match count:
    case 0n: segment(~Operands<E,Array<F32>>,~Plan,~job_step(~Plan,~Cursor,~E,~job_at,~advance,~times,~write),0n,0n,plan,operands)
    case 1n+rest: segment(~Operands<E,Array<F32>>,~Plan,~job_step(~Plan,~Cursor,~E,~job_at,~advance,~times,~write),0n,1n+rest,plan,operands)

# The blocks of eight rows that cover rows rows.
def blocks(rows: U32) -> U32:
  ((rows + 7) / 8 : U32)

# A product the tile writer stores: positions x channels cells, positions a
# multiple of eight. Row r of a block is a position, its tile the channels
# 8g to 8g + 7, stored at (start + position) * groups + g, the groups of a
# position side by side. The row
# tiles are [positions / 8, inner] tiles of eight positions, the lane tiles
# [channels / 8, inner] tiles of eight channels.
type TileMatrix is Data:
  TileMatrix{channels: U32,positions: U32,inner: U32,start: U32,output_positions: U32}

def tile_jobs(+matrix: TileMatrix) -> U32:
  TileMatrix{+channels,+positions,inner,start,output_positions} = matrix
  (positions / 8 * blocks(channels) : U32)

# One tile is 32 bytes; 8192 tiles are 256 KB.
def block_tiles() -> U32:
  8192

# Weights of a block or less stay in cache while each eight positions pass over
# them. Larger ones pass once, each channel group in cache over the positions.
def streamed(+groups: U32,+inner: U32) -> Bool:
  U32.is_lt(block_tiles(),(groups * inner : U32))

def job_group(once: Bool,+index: U32,+groups: U32,+units: U32) -> U32:
  match once:
    case True{}: (index / units : U32)
    case False{}: (index % groups : U32)

def job_unit(once: Bool,+index: U32,+groups: U32,+units: U32) -> U32:
  match once:
    case True{}: (index % units : U32)
    case False{}: (index / groups : U32)

def tile_major(matrix: TileMatrix,index: U32) -> Job<Direct()>:
  TileMatrix{+channels,+positions,+inner,+start,+output_positions} = matrix
  +index = index
  +groups = blocks(channels)
  +units = (positions / 8 : U32)
  +once = streamed(groups,inner)
  +group = job_group(once,index,groups,units)
  +channel = (group * 8 : U32)
  +unit = job_unit(once,index,groups,units)
  +position = (start + unit * 8 : U32)
  Job{channel,U32.to_nat(inner),Both{(group * inner : U32),(unit * inner : U32)},
    Store{(position * groups + group : U32),groups,1,U32.min(8,(output_positions - position : U32)),U32.min(8,(channels - channel : U32))}}

# A cell bias is owned beside the shared product loop. Unit carries no
# scalar owner; the epilogue reads one cell for eight output channels.
type CellOperands is Type:
  CellOperands{bias: Array<Tile.Tile8>,product: Operands<Tile.Tile8,Unit>}

def cell_bias(state: CellOperands) -> Array<Tile.Tile8>:
  CellOperands{bias,product} = state
  bias

def cell_product(state: CellOperands) -> Operands<Tile.Tile8,Unit>:
  CellOperands{bias,product} = state
  product

def cell_output(state: CellOperands) -> Array<Tile.Tile8>:
  output_of(~Tile.Tile8,~Unit,cell_product(state))

def cell_stored(row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,bias: Array<Tile.Tile8>,written: Written<Tile.Tile8,Unit>) -> CellOperands:
  Written{unused_bias,output} = written
  CellOperands{bias,Operands{row_tiles,lane_tiles,unused_bias,output}}

def cell_read(row_tiles: Array<Tile.Tile8>,lane_tiles: Array<Tile.Tile8>,unused_bias: Unit,output: Array<Tile.Tile8>,
  +store: Store,+accumulators: Accumulators,read: Array<Tile.Tile8> & Tile.Tile8) -> CellOperands:
  Store{base,row_stride,lane_stride,rows,count} = store
  (bias,biases) = read
  cell_stored(row_tiles,lane_tiles,bias,tiles_finished(~Unit,~plus,base,row_stride,rows,count,accumulators,output,(unused_bias,biases)))

def cell_finish(bias: Array<Tile.Tile8>,state: Loop<Direct(),Tile.Tile8,Unit>) -> CellOperands:
  Loop{row_tiles,lane_tiles,unused_bias,output,cursor,+start,store,accumulators} = state
  cell_read(row_tiles,lane_tiles,unused_bias,output,store,accumulators,Array.get(Tile.Tile8,bias,(start / 8 : U32)))

def cell_job(index: U32,matrix: TileMatrix,state: CellOperands) -> CellOperands:
  CellOperands{bias,product} = state
  cell_finish(bias,looped(~Direct(),~Tile.Tile8,~Unit,~directs,~row_times,tile_major(matrix,index),(product,zero())))

def cell_run(count: Nat,matrix: TileMatrix,state: CellOperands) -> CellOperands:
  match count:
    case 0n: segment(~CellOperands,~TileMatrix,~cell_job,0n,0n,matrix,state)
    case 1n+rest: segment(~CellOperands,~TileMatrix,~cell_job,0n,1n+rest,matrix,state)

# ---- Rows read from cells, sixteen channels a block ----

# A pointwise product reads the rows of its blocks from the input cells and
# multiplies them by pairs of lane tiles, sixteen channels stored as one
# element. Row r of a block is cell cell + r * stride, whose lanes are eight
# consecutive k, the cells of a row consecutive groups of k: step
# k = 16h + 8o + j reads lane j of cell 2h + o of each row and pair
# address + k.
type Points is Data:
  Points{address: U32,cell: U32,stride: U32}

# Four rows of sixteen channels: four positions are as many rows as clang
# still inlines into the k loop at sixteen k a step.
type Quad is Data:
  Quad{r0: Tile.Tile16,r1: Tile.Tile16,r2: Tile.Tile16,r3: Tile.Tile16}

def quad_zero() -> Quad:
  +zero = Tile.filled_pair(0.0)
  Quad{zero,zero,zero,zero}

def quad0(accumulators: Quad) -> Tile.Tile16:
  Quad{a0,a1,a2,a3} = accumulators
  a0

def quad1(accumulators: Quad) -> Tile.Tile16:
  Quad{a0,a1,a2,a3} = accumulators
  a1

def quad2(accumulators: Quad) -> Tile.Tile16:
  Quad{a0,a1,a2,a3} = accumulators
  a2

def quad3(accumulators: Quad) -> Tile.Tile16:
  Quad{a0,a1,a2,a3} = accumulators
  a3

type Row4 is Data:
  Row4{x0: F32,x1: F32,x2: F32,x3: F32}

type PointLoop is Type:
  PointLoop{cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,cursor: Points,accumulators: Quad}

# Lane `lane` of the four row cells; point_row_k holds the lanes of rows
# below k and reads row k.
def point_row_3(~lane: Tile.Tile8 -> F32,x0: F32,x1: F32,x2: F32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Row4:
  (cells,cell) = read
  (cells,Row4{x0,x1,x2,lane(cell)})

def point_row_2(~lane: Tile.Tile8 -> F32,+cell: U32,+stride: U32,x0: F32,x1: F32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Row4:
  (cells,value) = read
  point_row_3(~lane,x0,x1,lane(value),Array.get(Tile.Tile8,cells,(cell + 3 * stride : U32)))

def point_row_1(~lane: Tile.Tile8 -> F32,+cell: U32,+stride: U32,x0: F32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Row4:
  (cells,value) = read
  point_row_2(~lane,cell,stride,x0,lane(value),Array.get(Tile.Tile8,cells,(cell + 2 * stride : U32)))

def point_row_0(~lane: Tile.Tile8 -> F32,+cell: U32,+stride: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Row4:
  (cells,value) = read
  point_row_1(~lane,cell,stride,lane(value),Array.get(Tile.Tile8,cells,(cell + 1 * stride : U32)))

def point_rows(~lane: Tile.Tile8 -> F32,cells: Array<Tile.Tile8>,+cell: U32,+stride: U32) -> Array<Tile.Tile8> & Row4:
  point_row_0(~lane,cell,stride,Array.get(Tile.Tile8,cells,cell))

# One ordered k step: the pair of lane tiles, the four row lanes, then the
# products.
def point_updated(~times: F32 -> F32 -> F32,accumulators: Quad,+weight: Tile.Tile16,values: Row4) -> Quad:
  Quad{a0,a1,a2,a3} = accumulators
  Row4{x0,x1,x2,x3} = values
  Quad{Tile.update_pair(~times,a0,weight,x0),Tile.update_pair(~times,a1,weight,x1),Tile.update_pair(~times,a2,weight,x2),Tile.update_pair(~times,a3,weight,x3)}

def point_products(~times: F32 -> F32 -> F32,pairs: Array<Tile.Tile16>,cursor: Points,accumulators: Quad,weight: Tile.Tile16,read: Array<Tile.Tile8> & Row4) -> PointLoop:
  (cells,values) = read
  PointLoop{cells,pairs,cursor,point_updated(~times,accumulators,weight,values)}

def point_weight(~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,cells: Array<Tile.Tile8>,+cell: U32,+stride: U32,cursor: Points,accumulators: Quad,
  read: Array<Tile.Tile16> & Tile.Tile16) -> PointLoop:
  (pairs,weight) = read
  point_products(~times,pairs,cursor,accumulators,weight,point_rows(~lane,cells,cell,stride))

# The cell of a step: the first or the second of the two a group reads.
def first_cell(cell: U32) -> U32:
  cell

def second_cell(cell: U32) -> U32:
  (cell + 1 : U32)

# at and lane tell the steps of a group apart, so each specialization has
# one call site and clang inlines it (rule F7).
def point_step(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,state: PointLoop) -> PointLoop:
  PointLoop{cells,pairs,cursor,accumulators} = state
  Points{+address,+cell,+stride} = cursor
  point_weight(~lane,~times,cells,at(cell),stride,Points{(address + 1 : U32),cell,stride},accumulators,Array.get(Tile.Tile16,pairs,address))

def next_cells(state: PointLoop) -> PointLoop:
  PointLoop{cells,pairs,cursor,accumulators} = state
  Points{address,cell,stride} = cursor
  PointLoop{cells,pairs,Points{address,(cell + 2 : U32),stride},accumulators}

# The sixteen k of two cells of each row, then the next two cells.
def point_group(~times: F32 -> F32 -> F32,state: PointLoop) -> PointLoop:
  s = point_step(~first_cell,~Tile.lane0,~times,state)
  s = point_step(~first_cell,~Tile.lane1,~times,s)
  s = point_step(~first_cell,~Tile.lane2,~times,s)
  s = point_step(~first_cell,~Tile.lane3,~times,s)
  s = point_step(~first_cell,~Tile.lane4,~times,s)
  s = point_step(~first_cell,~Tile.lane5,~times,s)
  s = point_step(~first_cell,~Tile.lane6,~times,s)
  s = point_step(~first_cell,~Tile.lane7,~times,s)
  s = point_step(~second_cell,~Tile.lane0,~times,s)
  s = point_step(~second_cell,~Tile.lane1,~times,s)
  s = point_step(~second_cell,~Tile.lane2,~times,s)
  s = point_step(~second_cell,~Tile.lane3,~times,s)
  s = point_step(~second_cell,~Tile.lane4,~times,s)
  s = point_step(~second_cell,~Tile.lane5,~times,s)
  s = point_step(~second_cell,~Tile.lane6,~times,s)
  next_cells(point_step(~second_cell,~Tile.lane7,~times,s))

# A product whose rows are read from cells: the rows of position p from cell
# cell + p * stride on; channels, positions and inner as product, inner a
# multiple of sixteen in the whole-inner job. A job
# makes four positions by the sixteen channels of a pair of groups; the
# upper group of the last pair may be missing.
type PointMatrix is Data:
  PointMatrix{product: TileMatrix,cell: U32,stride: U32}

def pairs_of(+channels: U32) -> U32:
  ((blocks(channels) + 1) / 2 : U32)

def point_jobs(+matrix: TileMatrix) -> U32:
  TileMatrix{+channels,+positions,inner,start,output_positions} = matrix
  ((positions + 3) / 4 * pairs_of(channels) : U32)

# The pair and the four-position unit of a job, in the order tile_major
# takes its groups and units.
type Place is Data:
  Place{pair: U32,unit: U32}

def point_place(+matrix: PointMatrix,+index: U32) -> Place:
  PointMatrix{TileMatrix{+channels,+positions,+inner,start,output_positions},cell,stride} = matrix
  +pairs = pairs_of(channels)
  +units = ((positions + 3) / 4 : U32)
  +once = streamed(blocks(channels),inner)
  Place{job_group(once,index,pairs,units),job_unit(once,index,pairs,units)}

# The pairs of the weights are [pairs, inner]; the staged output holds the
# pairs of a position side by side.
type PointOperands is Type:
  PointOperands{bias: Array<Tile.Tile8>,cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,staged: Array<Tile.Tile16>}

def point_bias(state: PointOperands) -> Array<Tile.Tile8>:
  PointOperands{bias,cells,pairs,staged} = state
  bias

def point_cells(state: PointOperands) -> Array<Tile.Tile8>:
  PointOperands{bias,cells,pairs,staged} = state
  cells

def point_pairs(state: PointOperands) -> Array<Tile.Tile16>:
  PointOperands{bias,cells,pairs,staged} = state
  pairs

def point_staged(state: PointOperands) -> Array<Tile.Tile16>:
  PointOperands{bias,cells,pairs,staged} = state
  staged

def point_start(cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,+matrix: PointMatrix,+place: Place) -> PointLoop:
  PointMatrix{TileMatrix{channels,positions,+inner,start,output_positions},+cell,+stride} = matrix
  Place{+pair,+unit} = place
  PointLoop{cells,pairs,Points{(pair * inner : U32),(cell + unit * 4 * stride : U32),stride},quad_zero()}

# The finish: row r of the block is position start + 4 * unit + r, stored
# when it lies below output_positions, plus the biases of the pair's groups.
def point_stored_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,+biases: Tile.Tile16,+accumulators: Quad) -> Array<Tile.Tile16>:
  staged = put(~Tile.Tile16,U32.is_lt(0,rows),staged,base,add(quad0(accumulators),biases))
  staged = put(~Tile.Tile16,U32.is_lt(1,rows),staged,(base + stride : U32),add(quad1(accumulators),biases))
  staged = put(~Tile.Tile16,U32.is_lt(2,rows),staged,(base + 2 * stride : U32),add(quad2(accumulators),biases))
  put(~Tile.Tile16,U32.is_lt(3,rows),staged,(base + 3 * stride : U32),add(quad3(accumulators),biases))

def upper_biased_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,lower: Tile.Tile8,+accumulators: Quad,
  read: Array<Tile.Tile8> & Tile.Tile8) -> PointOperands:
  (bias,upper) = read
  PointOperands{bias,cells,pairs,point_stored_with(~add,staged,base,stride,rows,Tile.Tiles{lower,upper},accumulators)}

def lower_biased_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,+above: U32,+accumulators: Quad,
  read: Array<Tile.Tile8> & Tile.Tile8) -> PointOperands:
  (bias,lower) = read
  upper_biased_with(~add,cells,pairs,staged,base,stride,rows,lower,accumulators,Array.get(Tile.Tile8,bias,above))

def point_finish_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,bias: Array<Tile.Tile8>,staged: Array<Tile.Tile16>,+matrix: PointMatrix,+place: Place,state: PointLoop) -> PointOperands:
  PointMatrix{TileMatrix{+channels,positions,inner,+start,+output_positions},cell,stride} = matrix
  Place{+pair,+unit} = place
  PointLoop{cells,pairs,cursor,+accumulators} = state
  +groups = blocks(channels)
  +pairs_count = pairs_of(channels)
  +position = (start + unit * 4 : U32)
  lower_biased_with(~add,cells,pairs,staged,(position * pairs_count + pair : U32),pairs_count,U32.min(4,(output_positions - position : U32)),
    U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)),accumulators,Array.get(Tile.Tile8,bias,(pair * 2 : U32)))

def point_stored(staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,+biases: Tile.Tile16,+accumulators: Quad) -> Array<Tile.Tile16>:
  point_stored_with(~Tile.plus_pair,staged,base,stride,rows,biases,accumulators)

def upper_biased(cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,lower: Tile.Tile8,+accumulators: Quad,
  read: Array<Tile.Tile8> & Tile.Tile8) -> PointOperands:
  upper_biased_with(~Tile.plus_pair,cells,pairs,staged,base,stride,rows,lower,accumulators,read)

def lower_biased(cells: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,+above: U32,+accumulators: Quad,
  read: Array<Tile.Tile8> & Tile.Tile8) -> PointOperands:
  lower_biased_with(~Tile.plus_pair,cells,pairs,staged,base,stride,rows,above,accumulators,read)

def point_finish(bias: Array<Tile.Tile8>,staged: Array<Tile.Tile16>,+matrix: PointMatrix,+place: Place,state: PointLoop) -> PointOperands:
  point_finish_with(~Tile.plus_pair,bias,staged,matrix,place,state)

# Keep the bias stores at one native call site per kernel (F7, F13).
def panel_bias(value: Tile.Tile16,bias: Tile.Tile16) -> Tile.Tile16:
  Tile.plus_pair(value,bias)

def point_groups_of(+matrix: PointMatrix) -> Nat:
  PointMatrix{TileMatrix{channels,positions,+inner,start,output_positions},cell,stride} = matrix
  U32.to_nat((inner / 16 : U32))

def point_remainder(+matrix: PointMatrix) -> U32:
  PointMatrix{TileMatrix{channels,positions,+inner,start,output_positions},cell,stride} = matrix
  (inner % 16 : U32)

# A distinct specialization keeps the wide hot steps single-call-site when
# the suffix and the full inner loop are linked together.
def panel_times(lane: F32,row: F32) -> F32:
  row_times(lane,row)

def tail_times(lane: F32,row: F32) -> F32:
  row_times(lane,row)

# The short suffix reads only the remaining inner indices. Padding the
# operands would add products beyond the inner extent.
def tail_step(~at: U32 -> U32,lane: Nat,state: PointLoop) -> PointLoop:
  match lane:
    case 0n: point_step(~at,~Tile.lane0,~tail_times,state)
    case 1n: point_step(~at,~Tile.lane1,~tail_times,state)
    case 2n: point_step(~at,~Tile.lane2,~tail_times,state)
    case 3n: point_step(~at,~Tile.lane3,~tail_times,state)
    case 4n: point_step(~at,~Tile.lane4,~tail_times,state)
    case 5n: point_step(~at,~Tile.lane5,~tail_times,state)
    case 6n: point_step(~at,~Tile.lane6,~tail_times,state)
    case 7n+rest: point_step(~at,~Tile.lane7,~tail_times,state)

def tail_cell(~at: U32 -> U32,count: Nat,+lane: U32,state: PointLoop) -> PointLoop:
  match count:
    case 0n: state
    case 1n+rest: tail_cell(~at,rest,(lane + 1 : U32),tail_step(~at,U32.to_nat(lane),state))

def finish_inner(+matrix: PointMatrix,state: PointLoop) -> PointLoop:
  +tail = point_remainder(matrix)
  tail_cell(~second_cell,U32.to_nat((tail - U32.min(tail,8) : U32)),0,tail_cell(~first_cell,U32.to_nat(U32.min(tail,8)),0,state))

def whole_inner(+matrix: PointMatrix,state: PointLoop) -> PointLoop:
  state

# A job: the groups of sixteen k from +0, then the finish.
def point_job_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,~times: F32 -> F32 -> F32,~tail: @+matrix: PointMatrix -> PointLoop -> PointLoop,index: U32,matrix: PointMatrix,state: PointOperands) -> PointOperands:
  PointOperands{bias,cells,pairs,staged} = state
  +matrix = matrix
  +place = point_place(matrix,index)
  point_finish_with(~add,bias,staged,matrix,place,tail(matrix,Iteration.run(~PointLoop,~point_group(~times),point_groups_of(matrix),point_start(cells,pairs,matrix,place))))

def point_job(index: U32,matrix: PointMatrix,state: PointOperands) -> PointOperands:
  PointOperands{bias,cells,pairs,staged} = state
  +matrix = matrix
  +place = point_place(matrix,index)
  point_finish(bias,staged,matrix,place,Iteration.run(~PointLoop,~point_group(~row_times),point_groups_of(matrix),point_start(cells,pairs,matrix,place)))

# Jobs 0 to count - 1 of the product.
def point_run_indexed(~job: U32 -> PointMatrix -> PointOperands -> PointOperands,count: Nat,matrix: PointMatrix,state: PointOperands) -> PointOperands:
  match count:
    case 0n: state
    case 1n: job(0,matrix,state)
    case 2n+rest: segment(~PointOperands,~PointMatrix,~job,0n,2n+rest,matrix,state)

def point_run(count: Nat,matrix: PointMatrix,state: PointOperands) -> PointOperands:
  point_run_indexed(~point_job,count,matrix,state)

# The pair of lane tiles at index pair * inner + k: tiles 2 * pair and
# 2 * pair + 1 of k, the lower tile again for a missing upper group.
type PairPlan is Data:
  PairPlan{groups: U32,inner: U32}

def pair_upper(lower: Tile.Tile8,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile16:
  (tiles,upper) = read
  (tiles,Tile.Tiles{lower,upper})

def pair_lower(+above: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile16:
  (tiles,lower) = read
  pair_upper(lower,Array.get(Tile.Tile8,tiles,above))

def pair_read(tiles: Array<Tile.Tile8>,plan: PairPlan,index: U32) -> Array<Tile.Tile8> & Tile.Tile16:
  PairPlan{+groups,+inner} = plan
  +index = index
  +pair = (index / inner : U32)
  +k = (index % inner : U32)
  pair_lower((U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)) * inner + k : U32),Array.get(Tile.Tile8,tiles,(pair * 2 * inner + k : U32)))

def pair_placed(plan: PairPlan,index: U32) -> U32:
  index

# A panel already has the output lanes together. Pair those tiles directly
# from the inner-major panel, starting at base in its owning buffer.
type PanelPlan is Data:
  PanelPlan{groups: U32,inner: U32,base: U32}

def panel_pair_read(tiles: Array<Tile.Tile8>,plan: PanelPlan,index: U32) -> Array<Tile.Tile8> & Tile.Tile16:
  PanelPlan{+groups,+inner,+base} = plan
  +index = index
  +lower = (2 * (index / inner) : U32)
  +row = (base + (index % inner) * groups : U32)
  pair_lower((row + U32.min((lower + 1 : U32),(groups - 1 : U32)) : U32),Array.get(Tile.Tile8,tiles,(row + lower : U32)))

def panel_pair_placed(plan: PanelPlan,index: U32) -> U32:
  index

def replaced(~E: Data,old: E,value: E) -> E:
  value

# The tiles of one group from source on, each into one half of the pair at
# target + index.
type TilePlan is Data:
  TilePlan{source: U32,target: U32}

def tile_read(tiles: Array<Tile.Tile8>,plan: TilePlan,index: U32) -> Array<Tile.Tile8> & Tile.Tile8:
  TilePlan{+source,target} = plan
  Array.get(Tile.Tile8,tiles,(source + index : U32))

def tile_placed(plan: TilePlan,index: U32) -> U32:
  TilePlan{source,+target} = plan
  (target + index : U32)

def lower_half(pair: Tile.Tile16,tile: Tile.Tile8) -> Tile.Tile16:
  Tile.Tiles{lower,upper} = pair
  Tile.Tiles{tile,upper}

def upper_half(pair: Tile.Tile16,tile: Tile.Tile8) -> Tile.Tile16:
  Tile.Tiles{lower,upper} = pair
  Tile.Tiles{lower,tile}

# Cell index of the cells of the staged pairs: the half of pair
# (index / groups) * pairs + group / 2 that holds the group, mapped.
type StagedPlan<-Context: Data> is Data:
  StagedPlan{groups: U32,pairs: U32,context: Context}

def staged_half(value: Tile.Tile16,upper: Bool) -> Tile.Tile8:
  Tile.Tiles{lower,high} = value
  match upper:
    case True{}: high
    case False{}: lower

def staged_mapped(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,context: Context,upper: Bool,read: Array<Tile.Tile16> & Tile.Tile16) -> Array<Tile.Tile16> & Tile.Tile8:
  (staged,value) = read
  (staged,map(context,staged_half(value,upper)))

def staged_read(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,staged: Array<Tile.Tile16>,plan: StagedPlan<Context>,index: U32) -> Array<Tile.Tile16> & Tile.Tile8:
  StagedPlan{+groups,+pairs,context} = plan
  +index = index
  +group = (index % groups : U32)
  staged_mapped(~Context,~map,context,U32.is_eq((group % 2 : U32),1),Array.get(Tile.Tile16,staged,((index / groups) * pairs + group / 2 : U32)))

def staged_placed(~Context: Data,plan: StagedPlan<Context>,index: U32) -> U32:
  index

# A contiguous fragment of the target cell order. In the group-major order
# a group covers all positions before the next group begins.
type StagedSlice is Data:
  StagedSlice{groups: U32,positions: U32,first: U32,group_major: Bool}

def staged_slice_index(+slice: StagedSlice,+index: U32) -> U32:
  StagedSlice{+groups,+positions,+first,+group_major} = slice
  +cell = (first + index : U32)
  Bool.pick(U32,group_major,((cell % positions) * groups + cell / positions : U32),cell)

def unmapped_tile(context: Unit,tile: Tile.Tile8) -> Tile.Tile8:
  tile

def staged_slice_read(staged: Array<Tile.Tile16>,slice: StagedSlice,index: U32) -> Array<Tile.Tile16> & Tile.Tile8:
  StagedSlice{+groups,+positions,+first,+group_major} = slice
  staged_read(~Unit,~unmapped_tile,staged,StagedPlan{groups,((groups + 1) / 2 : U32),Unit{}},staged_slice_index(StagedSlice{groups,positions,first,group_major},index))

def staged_slice_destination(slice: StagedSlice,index: U32) -> U32:
  index

def point_count(+matrix: PointMatrix) -> U32:
  PointMatrix{product,cell,stride} = matrix
  point_jobs(product)
