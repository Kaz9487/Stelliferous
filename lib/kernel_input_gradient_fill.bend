# The tiles of a chunk of the output gradient: for each tile row and group of
# the chunk, every output channel and every column shift, zero outside. The
# input gradient reads its phase entries from them.
import Base
import ./kernel_tile.bend as Tile
import ./kernel_spatial_packing.bend as Spatial
import ./traversal_loop.bend as Iteration
import ./traversal_array.bend as Traversal

type Fill is Type:
  Fill{input: Array<Tile.Tile8>,tiles: Array<Tile.Tile8>,index: U32,row: U32,column: U32,first: U32,base: U32,groups: U32,height: U32,width: U32,shifts: Nat,channel: U32}

def filled(tiles: Array<Tile.Tile8>,+index: U32,row: U32,+column: U32,first: U32,base: U32,groups: U32,height: U32,width: U32,shifts: Nat,channel: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> Fill:
  (input,tile) = read
  Fill{input,Array.set(Tile.Tile8,tiles,index,tile),(index + 1 : U32),row,(column + 1 : U32),first,base,groups,height,width,shifts,channel}

def fill_shift(~field: Tile.Tile8 -> F32,state: Fill) -> Fill:
  Fill{input,tiles,index,+row,+column,first,+base,+groups,+height,+width,shifts,+channel} = state
  filled(tiles,index,row,column,first,base,groups,height,width,shifts,channel,
    Spatial.block(~Tile.Tile8,~field,input,row,column,(base + channel / 8 : U32),height,width,1,0,0,0,0,0,groups))

def next_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,+first,base,groups,height,width,shifts,channel} = state
  Fill{input,tiles,index,row,first,first,base,groups,height,width,shifts,(channel + 1 : U32)}

def fill_lane(~field: Tile.Tile8 -> F32,state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,+shifts,channel} = state
  next_channel(Iteration.run(~Fill,~fill_shift(~field),shifts,Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel}))

def fill_selected(half: Bool,low: Bool,first: Bool,state: Fill) -> Fill:
  match half low first:
    case True{} True{} True{}: fill_lane(~Tile.lane0,state)
    case True{} True{} False{}: fill_lane(~Tile.lane1,state)
    case True{} False{} True{}: fill_lane(~Tile.lane2,state)
    case True{} False{} False{}: fill_lane(~Tile.lane3,state)
    case False{} True{} True{}: fill_lane(~Tile.lane4,state)
    case False{} True{} False{}: fill_lane(~Tile.lane5,state)
    case False{} False{} True{}: fill_lane(~Tile.lane6,state)
    case False{} False{} False{}: fill_lane(~Tile.lane7,state)

def fill_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,+channel} = state
  +lane = (channel % 8 : U32)
  fill_selected(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel})

def fill_finished(state: Fill) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  Fill{input,tiles,index,row,column,first,base,groups,height,width,shifts,channel} = state
  (input,tiles)

# Tile rows from tile row low and groups [group, group + span) of a tiling of
# outputs channels and shifts column shifts, chunk_groups groups to a tile
# row, read from rows [first, first + rows) of a source of width columns. Tile
# row t is gradient row t - kernel; shift i starts at gradient column
# 8 * group + i + column_low - kernel.
type Filling is Data:
  Filling{shifts: U32,outputs: U32,column_low: U32,kernel: U32,width: U32,chunk_groups: U32,
    low: U32,base: U32,groups: U32,first: U32,rows: U32,group: U32,span: U32}

# One tile row and group: every output channel, every shift.
def fill_group(index: U32,context: Filling,owners: Array<Tile.Tile8> & Array<Tile.Tile8>) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  Filling{+shifts,+outputs,+column_low,+kernel,+width,+chunk_groups,+low,+base,+groups,+first,+source_rows,+group,+span} = context
  (input,tiles) = owners
  +index = index
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  fill_finished(Iteration.run(~Fill,~fill_channel,U32.to_nat(outputs),
    Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
      base,groups,source_rows,width,U32.to_nat(shifts),0}))

# count groups of tile rows, from group 0.
type Call is Data:
  Call{count: Nat,filling: Filling}

def run(call: Call,input: Array<Tile.Tile8>,tiles: Array<Tile.Tile8>) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  Call{count,filling} = call
  Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Filling,~fill_group,count,0,filling,(input,tiles))
