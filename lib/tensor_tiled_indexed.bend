# Indexed maps of the cells of a tiled tensor: each lane receives the
# row-major index of its value in the logical shape. Lane l of the cell of
# outer index o = b * after + r and group g is value
# (b * lanes + 8 * g + l) * after + r, so the lanes of a cell step by after.
import Base
import ./kernel_tile.bend as Tile
import ./tensor_grid.bend as Grid
import ./traversal_array.bend as Traversal
import ./traversal_blocks.bend as TraversalBlocks

# The axes of the logical shape around the lane axis, where the groups lie,
# and the first cell of the piece.
type Indexing is Data:
  Indexing{outer: U32,after: U32,lanes: U32,groups: U32,position: Grid.Position,first: U32}

def index_of(+outer: U32,+group: U32,+after: U32,+lanes: U32) -> U32:
  +inner = U32.max(after,1)
  (((outer / inner) * lanes + group * 8) * after + outer % inner : U32)

def index_at(position: Grid.Position,+cell: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32) -> U32:
  match position:
    case Grid.Innermost{}: index_of((cell / U32.max(groups,1) : U32),(cell % U32.max(groups,1) : U32),after,lanes)
    case Grid.Outermost{}: index_of((cell % U32.max(outer,1) : U32),(cell / U32.max(outer,1) : U32),after,lanes)

def lane_index(context: Indexing,+position: U32) -> U32:
  Indexing{+outer,+after,+lanes,+groups,place,+first} = context
  index_at(place,(first + position : U32),outer,after,lanes,groups)

def lane_step(context: Indexing) -> U32:
  Indexing{outer,after,lanes,groups,place,first} = context
  after

def lanes(~operation: U32 -> F32 -> F32,+index: U32,+step: U32,tile: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  Tile.Tile{operation(index,a),operation((index + step : U32),b),operation((index + 2 * step : U32),c),operation((index + 3 * step : U32),d),
    operation((index + 4 * step : U32),e),operation((index + 5 * step : U32),f),operation((index + 6 * step : U32),g),operation((index + 7 * step : U32),h)}

def cell(~operation: U32 -> F32 -> F32,+context: Indexing,+position: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile8:
  (input,tile) = read
  (input,lanes(~operation,lane_index(context,position),lane_step(context),tile))

def read(~operation: U32 -> F32 -> F32,input: Array<Tile.Tile8>,context: Indexing,position: U32) -> Array<Tile.Tile8> & Tile.Tile8:
  +position = position
  cell(~operation,context,position,Array.get(Tile.Tile8,input,position))

type Call is Data:
  Call{count: Nat,context: Indexing}

def run(~operation: U32 -> F32 -> F32,call: Call,input: Array<Tile.Tile8>,output: Array<Tile.Tile8>) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  Call{count,context} = call
  Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Indexing,
    Traversal.run(~Array<Tile.Tile8>,~Tile.Tile8,~Indexing,~read(~operation),~TraversalBlocks.in_place(~Indexing),count,Traversal.State{input,output,0,context}))
