# The cells of a logical tensor in the canonical order, and the values back.
# A logical shape reads around its lane axis as before indices, lanes values
# along the axis and after indices after it. A cell holds eight lanes of the
# axis: the cell of outer index o = b * after + r and group g lies at
# o * groups + g with the groups innermost and at g * outer + o with the
# groups outermost, and its lane l is the value at
# (b * lanes + 8 * g + l) * after + r, +0 past lanes. The source array is
# only read.
import Base
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./tensor_grid.bend as Grid
import ./traversal_blocks.bend as Blocks
import ./traversal_array.bend as Traversal

# The axes of a logical shape around its lane axis: outer indices of the
# other axes, after of them after the lane axis, lanes values and groups
# cells along it, and where the groups lie among the cell axes.
type Cells is Data:
  Cells{outer: U32,after: U32,lanes: U32,groups: U32,position: Grid.Position}

# One lane read: its channel, where that channel holds the position, the
# plane size and the true channel count.
type Spot is Data:
  Spot{channel: U32,index: U32,plane: U32,lanes: U32}

def next(spot: Spot) -> Spot:
  Spot{+channel,+index,+plane,lanes} = spot
  Spot{(channel + 1 : U32),(index + plane : U32),plane,lanes}

def read_spot(input: Array<F32>,spot: Spot) -> Array<F32> & F32:
  Spot{+channel,index,plane,+lanes} = spot
  Storage.read_when(~F32,0.0,U32.is_lt(channel,lanes),input,index)

def lane8(+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,h) = read
  (input,Tile.Tile{a,b,c,d,e,f,g,h})

def lane7(+spot: Spot,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,g) = read
  lane8(a,b,c,d,e,f,g,read_spot(input,next(spot)))

def lane6(+spot: Spot,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,f) = read
  lane7(next(spot),a,b,c,d,e,f,read_spot(input,next(spot)))

def lane5(+spot: Spot,+a: F32,+b: F32,+c: F32,+d: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,e) = read
  lane6(next(spot),a,b,c,d,e,read_spot(input,next(spot)))

def lane4(+spot: Spot,+a: F32,+b: F32,+c: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,d) = read
  lane5(next(spot),a,b,c,d,read_spot(input,next(spot)))

def lane3(+spot: Spot,+a: F32,+b: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,c) = read
  lane4(next(spot),a,b,c,read_spot(input,next(spot)))

def lane2(+spot: Spot,+a: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,b) = read
  lane3(next(spot),a,b,read_spot(input,next(spot)))

def lane1(+spot: Spot,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,a) = read
  lane2(next(spot),a,read_spot(input,next(spot)))

# With the groups innermost the cells of one outer index are a run of groups
# cells: cell g of the run of outer index o reads from lane 8 * g of o.
type Tiling is Data:
  Tiling{after: U32,lanes: U32,groups: U32,first: U32}

type Run is Data:
  Run{first: Spot,start: U32}

# Block block of a piece whose first outer index is first.
def group_run(context: Tiling,block: U32,start: U32) -> Run:
  Tiling{+after,+lanes,groups,+first} = context
  +index = (first + block : U32)
  +rest = (index % U32.max(after,1) : U32)
  Run{Spot{0,((index / U32.max(after,1)) * lanes * after + rest : U32),after,lanes},start}

def run_spot(run: Run,position: U32) -> Spot:
  Run{Spot{channel,index,+plane,lanes},start} = run
  +group = (position - start : U32)
  Spot{(channel + group * 8 : U32),(index + group * 8 * plane : U32),plane,lanes}

def cell_read(input: Array<F32>,run: Run,position: U32) -> Array<F32> & Tile.Tile8:
  +spot = run_spot(run,position)
  lane1(spot,read_spot(input,spot))

# The cells of blocks outer indices, groups cells each.
def tiled(+blocks: U32,context: Tiling,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Tiling{+after,+lanes,+groups,+first} = context
  Blocks.run(~Array<F32>,~Tile.Tile8,~Tiling,~Run,~group_run,~cell_read,~Blocks.in_place(~Run),U32.to_nat(blocks),groups,Tiling{after,lanes,groups,first},input,output)

# With the groups outermost every cell finds its group and outer index.
type Outer is Data:
  Outer{outer: U32,after: U32,lanes: U32,first: U32}

def outer_spot(context: Outer,position: U32) -> Spot:
  Outer{+outer,+after,+lanes,+first} = context
  +cell = (first + position : U32)
  +group = (cell / U32.max(outer,1) : U32)
  +index = (cell % U32.max(outer,1) : U32)
  Spot{(group * 8 : U32),(((index / U32.max(after,1)) * lanes + group * 8) * after + index % U32.max(after,1) : U32),after,lanes}

def outer_read(input: Array<F32>,context: Outer,position: U32) -> Array<F32> & Tile.Tile8:
  +spot = outer_spot(context,position)
  lane1(spot,read_spot(input,spot))

# count cells of the groups outermost, from cell first on.
def tiled_outermost(+count: U32,context: Outer,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Traversal.finish(~Array<F32>,~Tile.Tile8,~Outer,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Outer,~outer_read,~Blocks.in_place(~Outer),U32.to_nat(count),Traversal.State{input,output,0,context}))

# A lane selection is resolved before its tile is read.
type Field is Data:
  Field0{}
  Field1{}
  Field2{}
  Field3{}
  Field4{}
  Field5{}
  Field6{}
  Field7{}

def field_of(lane: U32) -> Field:
  match lane:
    case 0: Field0{}
    case 1: Field1{}
    case 2: Field2{}
    case 3: Field3{}
    case 4: Field4{}
    case 5: Field5{}
    case 6: Field6{}
    case _: Field7{}

def field_value(field: Field,tile: Tile.Tile8) -> F32:
  match field:
    case Field0{}: Tile.lane0(tile)
    case Field1{}: Tile.lane1(tile)
    case Field2{}: Tile.lane2(tile)
    case Field3{}: Tile.lane3(tile)
    case Field4{}: Tile.lane4(tile)
    case Field5{}: Tile.lane5(tile)
    case Field6{}: Tile.lane6(tile)
    case Field7{}: Tile.lane7(tile)

# Lane lane of a tile, for lane below 8.
def lane_of(tile: Tile.Tile8,+lane: U32) -> F32:
  field_value(field_of(lane),tile)

def lane_read(+lane: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & F32:
  (input,tile) = read
  (input,lane_of(tile,lane))

# The values in logical order: the block of before index b and lane c reads,
# for its after index r, lane c % 8 of the cell of outer index b * after + r
# and group c / 8.
type Pick is Data:
  Pick{lane: U32,base: U32,step: U32,start: U32}

def picked(position: Grid.Position,+outer: U32,+after: U32,+groups: U32,+before: U32,+channel: U32,+start: U32) -> Pick:
  match position:
    case Grid.Innermost{}: Pick{(channel % 8 : U32),(before * after * groups + channel / 8 : U32),groups,start}
    case Grid.Outermost{}: Pick{(channel % 8 : U32),(channel / 8 * outer + before * after : U32),1,start}

def channel_pick(context: Cells,block: U32,start: U32) -> Pick:
  Cells{+outer,+after,+lanes,+groups,position} = context
  +block = block
  picked(position,outer,after,groups,(block / U32.max(lanes,1) : U32),(block % U32.max(lanes,1) : U32),start)

def value_read(input: Array<Tile.Tile8>,pick: Pick,position: U32) -> Array<Tile.Tile8> & F32:
  Pick{+lane,base,step,start} = pick
  lane_read(lane,Array.get(Tile.Tile8,input,(base + (position - start) * step : U32)))

# The values of every before index and lane, a block of after values each.
def untiled(+blocks: U32,context: Cells,input: Array<Tile.Tile8>,output: Array<F32>) -> Array<Tile.Tile8> & Array<F32>:
  Cells{+outer,+after,+lanes,+groups,+position} = context
  Blocks.run(~Array<Tile.Tile8>,~F32,~Cells,~Pick,~channel_pick,~value_read,~Blocks.in_place(~Pick),U32.to_nat(blocks),after,Cells{outer,after,lanes,groups,position},input,output)

# The lanes of a cell from its first lane on: those below the true count lanes
# keep their value, the others become +0.
type Tail is Data:
  Tail{first: U32,lanes: U32}

def tail_lane(+first: U32,+lanes: U32,+lane: U32,value: F32) -> F32:
  Bool.pick(F32,U32.is_lt((first + lane : U32),lanes),value,0.0)

def tail(context: Tail,tile: Tile.Tile8) -> Tile.Tile8:
  Tail{+first,+lanes} = context
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  Tile.Tile{tail_lane(first,lanes,0,a),tail_lane(first,lanes,1,b),tail_lane(first,lanes,2,c),tail_lane(first,lanes,3,d),
    tail_lane(first,lanes,4,e),tail_lane(first,lanes,5,f),tail_lane(first,lanes,6,g),tail_lane(first,lanes,7,h)}
