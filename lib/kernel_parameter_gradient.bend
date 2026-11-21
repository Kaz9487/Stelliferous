# Bias gradient traversal over ordered spatial groups. The output gradient is
# cells whose lanes hold eight outputs; the result is cells of eight outputs,
# +0 past the outputs. Every lane of a cell is summed in the same order, so
# the cells are summed whole.
import Base
import ./kernel_tile.bend as Tile
import ./traversal_grouped.bend as Grouped
import ./traversal_loop.bend as Loop
import ./storage_buffer.bend as Storage
import ./tensor_move.bend as Move

type Work is Data:
  Work{images: Nat,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,rows: Nat,columns: Nat,group: Nat}

type Context is Data:
  Context{work: Work,image: Nat,element: Nat}

def groups(outputs: Nat) -> Nat:
  Nat.div(Nat.add(outputs,7n),8n)

# The cell of output element of image at a position: the groups of a position side by side.
def bias_address(context: Context,position: Nat) -> U32:
  Context{Work{n,c,+o,h,w,k,s,p,+rows,+columns,g},+image,+element} = context
  U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,Nat.mul(rows,columns)),position),groups(o)),Nat.div(element,8n)))

# ---- The sum of a group ----

# The positions of a range lie step cells apart in a band.
type Folding is Type:
  Folding{band: Storage.Buffer<Tile.Tile8>,address: U32,step: U32,total: Tile.Tile8}

def fold_added(+address: U32,+step: U32,total: Tile.Tile8,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Folding:
  (band,value) = read
  Folding{band,(address + step : U32),step,Tile.plus(total,value)}

def fold_step(state: Folding) -> Folding:
  Folding{band,+address,step,total} = state
  fold_added(address,step,total,Storage.read(~Tile.Tile8,band,address))

def folded(lent: Move.Lent<Tile.Tile8>,state: Folding) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Folding{band,address,step,total} = state
  (Move.returned_holding(~Tile.Tile8,lent,band),total)

def cell_step(context: Context) -> U32:
  Context{Work{n,c,o,h,w,k,s,p,rows,columns,g},image,element} = context
  U32.from_nat(groups(o))

def fold_banded(+step: U32,count: Nat,banded: Move.Band<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Move.Band{lent,band,+offset} = banded
  folded(lent,Loop.run(~Folding,~fold_step,count,Folding{band,offset,step,Tile.filled(0.0)}))

# The cells of a range of positions, ordered from +0, read from the band of
# held cells that holds them.
def fold(+context: Context,range: Grouped.Range,reader: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Grouped.Range{first,+count} = range
  +step = cell_step(context)
  +cells = U32.from_nat(count)
  fold_banded(step,count,Move.band(~Tile.Tile8,reader,bias_address(context,first),Bool.pick(U32,U32.is_zero(cells),0,((cells - 1) * step + 1 : U32)),Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0))))

def prepended(value: Tile.Tile8,pair: Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>) -> Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>:
  (reader,values) = pair
  (reader,value <> values)

def group_tail(pair: Move.Holding<Tile.Tile8> & Tile.Tile8,tail: Move.Holding<Tile.Tile8> -> Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>:
  (reader,value) = pair
  prepended(value,tail(reader))

def fold_ranges(+context: Context,ranges: List<&2,Grouped.Range>,reader: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>:
  match ranges:
    case []: (reader,[])
    case range <> rest: group_tail(fold(context,range,reader),reader => fold_ranges(context,rest,reader))

# ---- Trees of cells ----

def tile_zero(context: Unit) -> Tile.Tile8:
  Tile.filled(0.0)

def tile_added(context: Unit,left: Tile.Tile8,right: Tile.Tile8) -> Tile.Tile8:
  Tile.plus(left,right)

def tree(fuel: Nat,values: List<&2,Tile.Tile8>) -> Tile.Tile8:
  Grouped.levels(~&2,~Tile.Tile8,~Unit,~tile_zero,~tile_added,fuel,Unit{},values)

def reduced(fuel: Nat,pair: Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  (reader,values) = pair
  (reader,tree(fuel,values))

# ---- The cells of the outputs ----

# The outputs of a block of eight and the groups they are summed in.
type Block is Data:
  Block{work: Work,block: Nat,ranges: List<&2,Grouped.Range>}

def image_values(remaining: Nat,+image: Nat,+element: Nat,+work: Work,+ranges: List<&2,Grouped.Range>,reader: Move.Holding<Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & List<&2,Tile.Tile8>:
  match remaining:
    case 0n: (reader,[])
    case 1n+rest:
      group_tail(reduced(List.length(&2,Grouped.Range,ranges),fold_ranges(Context{work,image,element},ranges,reader)),
        reader => image_values(rest,1n+image,element,work,ranges,reader))

# Output 8 * block + lane of a lane, +0 past the outputs.
def lane_kept(+outputs: Nat,+block: Nat,+lane: Nat,value: F32) -> F32:
  Bool.pick(F32,Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),outputs),value,0.0)

def kept(+outputs: Nat,+block: Nat,tile: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  Tile.Tile{lane_kept(outputs,block,0n,a),lane_kept(outputs,block,1n,b),lane_kept(outputs,block,2n,c),lane_kept(outputs,block,3n,d),
    lane_kept(outputs,block,4n,e),lane_kept(outputs,block,5n,f),lane_kept(outputs,block,6n,g),lane_kept(outputs,block,7n,h)}

def cell_kept(+outputs: Nat,+block: Nat,pair: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  (reader,tile) = pair
  (reader,kept(outputs,block,tile))

def cell(+context: Block,reader: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Block{+work,+block,+ranges} = context
  Work{+n,c,+o,h,w,k,s,p,rows,columns,g} = work
  cell_kept(o,block,reduced(n,image_values(n,0n,Nat.mul(block,8n),work,ranges,reader)))

# One cell written at its position, then the rest of the cells.
def stored(+position: U32,result: Storage.Buffer<Tile.Tile8>,pair: Move.Holding<Tile.Tile8> & Tile.Tile8,
  tail: Move.Holding<Tile.Tile8> -> Storage.Buffer<Tile.Tile8> -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (reader,value) = pair
  tail(reader,Storage.write(~Tile.Tile8,position,value,result))

def written(remaining: Nat,+index: Nat,+position: U32,+work: Work,+ranges: List<&2,Grouped.Range>,
  reader: Move.Holding<Tile.Tile8>,result: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match remaining:
    case 0n: (reader,result)
    case 1n+rest:
      stored(position,result,cell(Block{work,index,ranges},reader),
        reader => result => written(rest,1n+index,(position + 1 : U32),work,ranges,reader,result))

# The cells of d are read from the pieces that hold them; every cell of the
# outputs is written into result.
def bias(+work: Work,d: Move.Holding<Tile.Tile8>,result: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  written(groups(o),0n,0,work,Grouped.ranges(rows,columns,g),d,result)
