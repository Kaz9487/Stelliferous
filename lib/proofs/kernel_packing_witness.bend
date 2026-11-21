# The packer hands back what it reads and keeps the witness of what it
# writes: the transpose reads cells and writes pairs, the pack reads pairs
# and writes tiles, both at the depth they started with.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./kernel_packing_read_witness.bend as PackingReadWitness

# ---- The transpose ----

type RowingCursor is Data:
  RowingCursor{column: U32,write: U32,row: U32,in_row: U32,row_tiles: U32,first_column: U32,channel_base: U32,height: U32,width: U32,stride: U32,groups: U32,channels: U32}

def restore_rowing(input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,previous: Tile.Tile8,cursor: RowingCursor) -> Packing.Rowing:
  RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = cursor
  Packing.Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}

type RowingEvidence<-input: Array<Tile.Tile8>,-depth: Nat,-state: Packing.Rowing> is Data:
  RowingEvidence{-pairs: Array<Tile.Tile16>,-previous: Tile.Tile8,cursor: RowingCursor,equation: {state == restore_rowing(input,pairs,previous,cursor) : Packing.Rowing},
    balanced: Witness.Witness(Tile.Tile16,depth,pairs)}

def kept_rowing(pairs: Array<Tile.Tile16>,previous: Tile.Tile8,cursor: RowingCursor,result: Array<Tile.Tile8> & Tile.Tile8) -> Packing.Rowing:
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  Packing.row_read(pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,result)

def store_rowing(+depth: Nat,-input: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-previous: Tile.Tile8,cursor: RowingCursor,-result: Array<Tile.Tile8> & Tile.Tile8,
  balanced: Witness.Witness(Tile.Tile16,depth,pairs),sample: TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,result>) ->
  RowingEvidence<input,depth,kept_rowing(pairs,previous,cursor,result)>:
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  TraversalWitness.Sample{value,equation} = sample
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) :
    RowingEvidence<input,depth,kept_rowing(pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels},_)>
  +wrap = U32.is_eq((in_row + 1 : U32),row_tiles)
  RowingEvidence{Array.set(Tile.Tile16,pairs,write,Tile.Tiles{previous,value}),value,
    RowingCursor{Bool.pick(U32,wrap,first_column,(column + 8 * stride : U32)),(write + channels : U32),Bool.pick(U32,wrap,(row + 1 : U32),row),Bool.pick(U32,wrap,0,(in_row + 1 : U32)),
      row_tiles,first_column,channel_base,height,width,stride,groups,channels},
    {==},Witness.write(Tile.Tile16,depth,pairs,write,Tile.Tiles{previous,value},balanced)}

def row_step(~lane: Tile.Tile8 -> F32,+depth: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_step(~lane,state)>:
  RowingEvidence{pairs,previous,cursor,equation,balanced} = evidence
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  %Equal.sym(Packing.Rowing,state,
    restore_rowing(input,pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}),equation) :
    RowingEvidence<input,depth,Packing.row_step(~lane,_)>
  store_rowing(depth,input,pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels},
    Packing.block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups),balanced,
    PackingReadWitness.row_block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups,input_evidence))

def row_loop(~lane: Tile.Tile8 -> F32,+depth: Nat,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Loop.run(~Packing.Rowing,~Packing.row_step(~lane),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: row_loop(~lane,depth,rest,input,Packing.row_step(~lane,state),input_evidence,row_step(~lane,depth,input,state,input_evidence,evidence))

def first_rowing(pairs: Array<Tile.Tile16>,previous: Tile.Tile8,cursor: RowingCursor,result: Array<Tile.Tile8> & Tile.Tile8) -> Packing.Rowing:
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  Packing.row_first_read(pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels,result)

def first_read(-depth: Nat,-input: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-previous: Tile.Tile8,cursor: RowingCursor,-result: Array<Tile.Tile8> & Tile.Tile8,
  balanced: Witness.Witness(Tile.Tile16,depth,pairs),sample: TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,result>) ->
  RowingEvidence<input,depth,first_rowing(pairs,previous,cursor,result)>:
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  TraversalWitness.Sample{value,equation} = sample
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) :
    RowingEvidence<input,depth,first_rowing(pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels},_)>
  +wrap = U32.is_eq((in_row + 1 : U32),row_tiles)
  RowingEvidence{pairs,value,
    RowingCursor{Bool.pick(U32,wrap,first_column,(column + 8 * stride : U32)),write,Bool.pick(U32,wrap,(row + 1 : U32),row),Bool.pick(U32,wrap,0,(in_row + 1 : U32)),
      row_tiles,first_column,channel_base,height,width,stride,groups,channels},
    {==},balanced}

def row_first(~lane: Tile.Tile8 -> F32,+depth: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_first(~lane,state)>:
  RowingEvidence{pairs,previous,cursor,equation,balanced} = evidence
  RowingCursor{+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = cursor
  %Equal.sym(Packing.Rowing,state,
    restore_rowing(input,pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}),equation) :
    RowingEvidence<input,depth,Packing.row_first(~lane,_)>
  first_read(depth,input,pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels},
    Packing.block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups),balanced,
    PackingReadWitness.row_block(~Tile.Tile8,~lane,input,row,column,channel_base,height,width,stride,0,0,0,0,0,groups,input_evidence))

def row_run(~lane: Tile.Tile8 -> F32,+depth: Nat,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_run(~lane,count,state)>:
  row_loop(~lane,depth,count,input,Packing.row_first(~lane,state),input_evidence,row_first(~lane,depth,input,state,input_evidence,evidence))

def row_low(+depth: Nat,low: Bool,first: Bool,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_low(low,first,count,state)>:
  match low first:
    case True{} True{}: row_run(~Tile.lane0,depth,count,input,state,input_evidence,evidence)
    case True{} False{}: row_run(~Tile.lane1,depth,count,input,state,input_evidence,evidence)
    case False{} True{}: row_run(~Tile.lane2,depth,count,input,state,input_evidence,evidence)
    case False{} False{}: row_run(~Tile.lane3,depth,count,input,state,input_evidence,evidence)

def row_high(+depth: Nat,low: Bool,first: Bool,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_high(low,first,count,state)>:
  match low first:
    case True{} True{}: row_run(~Tile.lane4,depth,count,input,state,input_evidence,evidence)
    case True{} False{}: row_run(~Tile.lane5,depth,count,input,state,input_evidence,evidence)
    case False{} True{}: row_run(~Tile.lane6,depth,count,input,state,input_evidence,evidence)
    case False{} False{}: row_run(~Tile.lane7,depth,count,input,state,input_evidence,evidence)

def row_lane(+depth: Nat,half: Bool,low: Bool,first: Bool,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Rowing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: RowingEvidence<input,depth,state>) -> RowingEvidence<input,depth,Packing.row_lane(half,low,first,count,state)>:
  match half:
    case True{}: row_low(depth,low,first,count,input,state,input_evidence,evidence)
    case False{}: row_high(depth,low,first,count,input,state,input_evidence,evidence)

type TransposingCursor is Data:
  TransposingCursor{plane: U32,window: Packing.Stencil,planes: Packing.Planes}

def restore_transposing(input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,cursor: TransposingCursor) -> Packing.Transposing:
  TransposingCursor{plane,window,planes} = cursor
  Packing.Transposing{input,pairs,plane,window,planes}

type TransposingEvidence<-input: Array<Tile.Tile8>,-depth: Nat,-state: Packing.Transposing> is Data:
  TransposingEvidence{-pairs: Array<Tile.Tile16>,cursor: TransposingCursor,equation: {state == restore_transposing(input,pairs,cursor) : Packing.Transposing},
    balanced: Witness.Witness(Tile.Tile16,depth,pairs)}

def plane_done(-depth: Nat,+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,-input: Array<Tile.Tile8>,-state: Packing.Rowing,evidence: RowingEvidence<input,depth,state>) ->
  TransposingEvidence<input,depth,Packing.plane_done(plane,window,planes,state)>:
  RowingEvidence{pairs,previous,cursor,equation,balanced} = evidence
  RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = cursor
  %Equal.sym(Packing.Rowing,state,
    restore_rowing(input,pairs,previous,RowingCursor{column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}),equation) :
    TransposingEvidence<input,depth,Packing.plane_done(plane,window,planes,_)>
  TransposingEvidence{pairs,TransposingCursor{(plane + 1 : U32),window,planes},{==},balanced}

def plane_step(+depth: Nat,-input: Array<Tile.Tile8>,-state: Packing.Transposing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: TransposingEvidence<input,depth,state>) -> TransposingEvidence<input,depth,Packing.plane_step(state)>:
  TransposingEvidence{pairs,cursor,equation,balanced} = evidence
  TransposingCursor{+plane,+window,+planes} = cursor
  Packing.Stencil{+base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,output_width,inner,+groups} = window
  Packing.Planes{+first_output_row,rows,+row_width,+elements} = planes
  %Equal.sym(Packing.Transposing,state,restore_transposing(input,pairs,TransposingCursor{plane,window,planes}),equation) :
    TransposingEvidence<input,depth,Packing.plane_step(_)>
  +channels = Packing.channels_of(window)
  +channel = (plane % channels : U32)
  +phase = (plane / channels : U32)
  +lane = (channel % 8 : U32)
  +cursor = {RowingCursor{(phase + column_origin : U32),(phase * elements * channels + channel : U32),(first_output_row * stride + row_origin : U32),0,(row_width / 8 : U32),
    (phase + column_origin : U32),(base + channel / 8 : U32),height,width,stride,groups,channels} : RowingCursor}
  -start = restore_rowing(input,pairs,Tile.filled(0.0),cursor)
  plane_done(depth,plane,window,planes,input,
    Packing.row_lane(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),start),
    row_lane(depth,U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),U32.to_nat(elements),input,start,input_evidence,
      RowingEvidence{pairs,Tile.filled(0.0),cursor,{==},balanced}))

def plane_run(+depth: Nat,count: Nat,-input: Array<Tile.Tile8>,-state: Packing.Transposing,+input_evidence: PackingReadWitness.family(~Tile.Tile8,input),
  evidence: TransposingEvidence<input,depth,state>) -> TransposingEvidence<input,depth,Loop.run(~Packing.Transposing,~Packing.plane_step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: plane_run(depth,rest,input,Packing.plane_step(state),input_evidence,plane_step(depth,input,state,input_evidence,evidence))

def transposed(-depth: Nat,-input: Array<Tile.Tile8>,-state: Packing.Transposing,evidence: TransposingEvidence<input,depth,state>) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile16,input,depth,Packing.transposed(state)>:
  TransposingEvidence{pairs,cursor,equation,balanced} = evidence
  TransposingCursor{plane,window,planes} = cursor
  %Equal.sym(Packing.Transposing,state,restore_transposing(input,pairs,TransposingCursor{plane,window,planes}),equation) :
    TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile16,input,depth,Packing.transposed(_)>
  TraversalWitness.ResultEvidence{pairs,{==},balanced}

def transpose(+depth: Nat,+window: Packing.Stencil,+planes: Packing.Planes,-input: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,
  +input_evidence: PackingReadWitness.family(~Tile.Tile8,input),balanced: Witness.Witness(Tile.Tile16,depth,pairs)) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile16,input,depth,Packing.transpose(window,planes,input,pairs)>:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  +count = U32.to_nat((stride * Packing.channels_of(window) : U32))
  +cursor = {TransposingCursor{0,window,planes} : TransposingCursor}
  transposed(depth,input,Loop.run(~Packing.Transposing,~Packing.plane_step,count,restore_transposing(input,pairs,cursor)),
    plane_run(depth,count,input,restore_transposing(input,pairs,cursor),input_evidence,TransposingEvidence{pairs,cursor,{==},balanced}))

# ---- The pack ----

type PassingCursor is Data:
  PassingCursor{start: U32,read: U32,kernel_row: U32,write: U32,lane: U32,tap_rows: U32,tap_columns: U32,row_step: U32}

def restore_passing(pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>,cursor: PassingCursor) -> Packing.Passing:
  PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = cursor
  Packing.Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step}

type PassingEvidence<-pairs: Array<Tile.Tile16>,-depth: Nat,-state: Packing.Passing> is Data:
  PassingEvidence{-tiles: Array<Tile.Tile8>,cursor: PassingCursor,equation: {state == restore_passing(pairs,tiles,cursor) : Packing.Passing},
    balanced: Witness.Witness(Tile.Tile8,depth,tiles)}

# The cursor of Packing.advanced.
def advanced(+start: U32,+read: U32,+kernel_row: U32,+write: U32,+lane: U32,+tap_rows: U32,+tap_columns: U32,+row_step: U32) -> PassingCursor:
  +wrap = U32.is_eq((kernel_row + 1 : U32),tap_rows)
  PassingCursor{Bool.pick(U32,wrap,(start + 1 : U32),start),Bool.pick(U32,wrap,(start + 1 : U32),(read + row_step : U32)),Bool.pick(U32,wrap,0,(kernel_row + 1 : U32)),
    Bool.pick(U32,wrap,(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),(write + tap_columns : U32)),lane,tap_rows,tap_columns,row_step}

def first_kept(~shift: Tile.Tile16 -> Tile.Tile8,tiles: Array<Tile.Tile8>,cursor: PassingCursor,result: Array<Tile.Tile16> & Tile.Tile16) -> Packing.Passing:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Packing.first_stored(~shift,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,result)

def first_store(~shift: Tile.Tile16 -> Tile.Tile8,+depth: Nat,-pairs: Array<Tile.Tile16>,-tiles: Array<Tile.Tile8>,cursor: PassingCursor,-result: Array<Tile.Tile16> & Tile.Tile16,
  balanced: Witness.Witness(Tile.Tile8,depth,tiles),observed: Witness.Read<Tile.Tile16,pairs,result>) -> PassingEvidence<pairs,depth,first_kept(~shift,tiles,cursor,result)>:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,result,(pairs,value),equation) :
    PassingEvidence<pairs,depth,first_kept(~shift,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},_)>
  PassingEvidence{Array.set(Tile.Tile8,tiles,write,shift(value)),advanced(start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step),{==},
    Witness.write(Tile.Tile8,depth,tiles,write,shift(value),balanced)}

def first_step(~shift: Tile.Tile16 -> Tile.Tile8,+depth: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),
  evidence: PassingEvidence<pairs,depth,state>) -> PassingEvidence<pairs,depth,Packing.first_step(~shift,state)>:
  PassingEvidence{tiles,cursor,equation,balanced} = evidence
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  %Equal.sym(Packing.Passing,state,restore_passing(pairs,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step}),equation) :
    PassingEvidence<pairs,depth,Packing.first_step(~shift,_)>
  first_store(~shift,depth,pairs,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},Array.get(Tile.Tile16,pairs,read),balanced,
    Witness.observed(Tile.Tile16,pairs,read,pairs_evidence))

def merge_kept(pairs: Array<Tile.Tile16>,+tile: Tile.Tile8,cursor: PassingCursor,result: Array<Tile.Tile8> & Tile.Tile8) -> Packing.Passing:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Packing.merge_stored(pairs,tile,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,result)

def merge_store(+depth: Nat,-pairs: Array<Tile.Tile16>,-tile: Tile.Tile8,-tiles: Array<Tile.Tile8>,cursor: PassingCursor,-result: Array<Tile.Tile8> & Tile.Tile8,
  balanced: Witness.Witness(Tile.Tile8,depth,tiles),observed: Witness.Read<Tile.Tile8,tiles,result>) -> PassingEvidence<pairs,depth,merge_kept(pairs,tile,cursor,result)>:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Witness.Read{kept,equation} = observed
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(tiles,kept),equation) :
    PassingEvidence<pairs,depth,merge_kept(pairs,tile,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},_)>
  PassingEvidence{Array.set(Tile.Tile8,tiles,write,Packing.merged(lane,kept,tile)),advanced(start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step),{==},
    Witness.write(Tile.Tile8,depth,tiles,write,Packing.merged(lane,kept,tile),balanced)}

def merge_read_kept(~shift: Tile.Tile16 -> Tile.Tile8,tiles: Array<Tile.Tile8>,cursor: PassingCursor,result: Array<Tile.Tile16> & Tile.Tile16) -> Packing.Passing:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Packing.merge_read(~shift,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step,result)

def merge_read(~shift: Tile.Tile16 -> Tile.Tile8,+depth: Nat,-pairs: Array<Tile.Tile16>,-tiles: Array<Tile.Tile8>,cursor: PassingCursor,-result: Array<Tile.Tile16> & Tile.Tile16,
  +balanced: Witness.Witness(Tile.Tile8,depth,tiles),observed: Witness.Read<Tile.Tile16,pairs,result>) -> PassingEvidence<pairs,depth,merge_read_kept(~shift,tiles,cursor,result)>:
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  Witness.Read{value,equation} = observed
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,result,(pairs,value),equation) :
    PassingEvidence<pairs,depth,merge_read_kept(~shift,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},_)>
  merge_store(depth,pairs,shift(value),tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},Array.get(Tile.Tile8,tiles,write),balanced,
    Witness.read(Tile.Tile8,depth,tiles,write,balanced))

def merge_step(~shift: Tile.Tile16 -> Tile.Tile8,+depth: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),
  evidence: PassingEvidence<pairs,depth,state>) -> PassingEvidence<pairs,depth,Packing.merge_step(~shift,state)>:
  PassingEvidence{tiles,cursor,equation,+balanced} = evidence
  PassingCursor{+start,+read,+kernel_row,+write,+lane,+tap_rows,+tap_columns,+row_step} = cursor
  %Equal.sym(Packing.Passing,state,restore_passing(pairs,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step}),equation) :
    PassingEvidence<pairs,depth,Packing.merge_step(~shift,_)>
  merge_read(~shift,depth,pairs,tiles,PassingCursor{start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step},Array.get(Tile.Tile16,pairs,read),balanced,
    Witness.observed(Tile.Tile16,pairs,read,pairs_evidence))

def PassingPreserves(~step: Packing.Passing -> Packing.Passing) -> Type:
  @+depth: Nat -> @-pairs: Array<Tile.Tile16> -> @-state: Packing.Passing -> @+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs) ->
    PassingEvidence<pairs,depth,state> -> PassingEvidence<pairs,depth,step(state)>

def passing_loop(~step: Packing.Passing -> Packing.Passing,~preserves: PassingPreserves(~step),+depth: Nat,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: PassingEvidence<pairs,depth,state>) ->
  PassingEvidence<pairs,depth,Loop.run(~Packing.Passing,~step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: passing_loop(~step,~preserves,depth,rest,pairs,step(state),pairs_evidence,preserves(depth,pairs,state,pairs_evidence,evidence))

def pass_run(~shift: Tile.Tile16 -> Tile.Tile8,+depth: Nat,first: Bool,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: PassingEvidence<pairs,depth,state>) ->
  PassingEvidence<pairs,depth,Packing.pass_run(~shift,first,count,state)>:
  match first:
    case True{}: passing_loop(~Packing.first_step(~shift),~first_step(~shift),depth,count,pairs,state,pairs_evidence,evidence)
    case False{}: passing_loop(~Packing.merge_step(~shift),~merge_step(~shift),depth,count,pairs,state,pairs_evidence,evidence)

def pass_low(+depth: Nat,low: Bool,odd: Bool,first: Bool,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: PassingEvidence<pairs,depth,state>) ->
  PassingEvidence<pairs,depth,Packing.pass_low(low,odd,first,count,state)>:
  match low odd:
    case True{} False{}: pass_run(~Packing.shifted0,depth,first,count,pairs,state,pairs_evidence,evidence)
    case True{} True{}: pass_run(~Packing.shifted1,depth,first,count,pairs,state,pairs_evidence,evidence)
    case False{} False{}: pass_run(~Packing.shifted2,depth,first,count,pairs,state,pairs_evidence,evidence)
    case False{} True{}: pass_run(~Packing.shifted3,depth,first,count,pairs,state,pairs_evidence,evidence)

def pass_high(+depth: Nat,low: Bool,odd: Bool,first: Bool,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: PassingEvidence<pairs,depth,state>) ->
  PassingEvidence<pairs,depth,Packing.pass_high(low,odd,first,count,state)>:
  match low odd:
    case True{} False{}: pass_run(~Packing.shifted4,depth,first,count,pairs,state,pairs_evidence,evidence)
    case True{} True{}: pass_run(~Packing.shifted5,depth,first,count,pairs,state,pairs_evidence,evidence)
    case False{} False{}: pass_run(~Packing.shifted6,depth,first,count,pairs,state,pairs_evidence,evidence)
    case False{} True{}: pass_run(~Packing.shifted7,depth,first,count,pairs,state,pairs_evidence,evidence)

def pass_shift(+depth: Nat,half: Bool,low: Bool,odd: Bool,first: Bool,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passing,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: PassingEvidence<pairs,depth,state>) ->
  PassingEvidence<pairs,depth,Packing.pass_shift(half,low,odd,first,count,state)>:
  match half:
    case True{}: pass_low(depth,low,odd,first,count,pairs,state,pairs_evidence,evidence)
    case False{}: pass_high(depth,low,odd,first,count,pairs,state,pairs_evidence,evidence)

type PassesCursor is Data:
  PassesCursor{pass: U32,lane: U32,base: U32,second_lane: U32,kernel_column: U32,unit: U32,window: Packing.Stencil,planes: Packing.Planes}

def restore_passes(pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>,cursor: PassesCursor) -> Packing.Passes:
  PassesCursor{pass,lane,base,second_lane,kernel_column,unit,window,planes} = cursor
  Packing.Passes{Packing.Arrays{pairs,tiles},pass,lane,base,second_lane,kernel_column,unit,window,planes}

type PassesEvidence<-pairs: Array<Tile.Tile16>,-depth: Nat,-state: Packing.Passes> is Data:
  PassesEvidence{-tiles: Array<Tile.Tile8>,cursor: PassesCursor,equation: {state == restore_passes(pairs,tiles,cursor) : Packing.Passes},
    balanced: Witness.Witness(Tile.Tile8,depth,tiles)}

def pass_done(-depth: Nat,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,
  -pairs: Array<Tile.Tile16>,-state: Packing.Passing,evidence: PassingEvidence<pairs,depth,state>) ->
  PassesEvidence<pairs,depth,Packing.Passes{Packing.passed(state),pass,lane,base,second_lane,kernel_column,unit,window,planes}>:
  PassingEvidence{tiles,cursor,equation,balanced} = evidence
  PassingCursor{start,read,kernel_row,write,passing_lane,tap_rows,tap_columns,row_step} = cursor
  %Equal.sym(Packing.Passing,state,restore_passing(pairs,tiles,PassingCursor{start,read,kernel_row,write,passing_lane,tap_rows,tap_columns,row_step}),equation) :
    PassesEvidence<pairs,depth,Packing.Passes{Packing.passed(_),pass,lane,base,second_lane,kernel_column,unit,window,planes}>
  PassesEvidence{tiles,PassesCursor{pass,lane,base,second_lane,kernel_column,unit,window,planes},{==},balanced}

def pass_step(+depth: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passes,+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),
  evidence: PassesEvidence<pairs,depth,state>) -> PassesEvidence<pairs,depth,Packing.pass_step(state)>:
  PassesEvidence{tiles,cursor,equation,balanced} = evidence
  PassesCursor{+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = cursor
  Packing.Stencil{base_cell,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,+inner,groups} = window
  Packing.Planes{first_output_row,rows,+row_width,+elements} = planes
  %Equal.sym(Packing.Passes,state,restore_passes(pairs,tiles,PassesCursor{pass,lane,base,second_lane,kernel_column,unit,window,planes}),equation) :
    PassesEvidence<pairs,depth,Packing.pass_step(_)>
  +channels = Packing.channels_of(window)
  +shift = (base % 8 : U32)
  +read = (((kernel_column % stride) * elements + base / 8) * channels : U32)
  -start = restore_passing(pairs,tiles,PassingCursor{read,read,0,(unit * inner + kernel_column : U32),lane,tap_rows,tap_columns,(row_width / 8 * channels : U32)})
  pass_done(depth,(pass + 1 : U32),Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32)),(base + stride * row_width - output_width : U32),second_lane,
    kernel_column,unit,window,planes,pairs,
    Packing.pass_shift(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),U32.to_nat((channels * tap_rows : U32)),start),
    pass_shift(depth,U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),U32.to_nat((channels * tap_rows : U32)),pairs,start,
      pairs_evidence,PassingEvidence{tiles,PassingCursor{read,read,0,(unit * inner + kernel_column : U32),lane,tap_rows,tap_columns,(row_width / 8 * channels : U32)},{==},balanced}))

def pass_loop(+depth: Nat,count: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passes,+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),
  evidence: PassesEvidence<pairs,depth,state>) -> PassesEvidence<pairs,depth,Loop.run(~Packing.Passes,~Packing.pass_step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: pass_loop(depth,rest,pairs,Packing.pass_step(state),pairs_evidence,pass_step(depth,pairs,state,pairs_evidence,evidence))

# The arrays between units: the pairs read and the tiles written.
type ArraysEvidence<-pairs: Array<Tile.Tile16>,-depth: Nat,-arrays: Packing.Arrays> is Data:
  ArraysEvidence{-tiles: Array<Tile.Tile8>,equation: {arrays == Packing.Arrays{pairs,tiles} : Packing.Arrays},balanced: Witness.Witness(Tile.Tile8,depth,tiles)}

def passes_done(-depth: Nat,-pairs: Array<Tile.Tile16>,-state: Packing.Passes,evidence: PassesEvidence<pairs,depth,state>) ->
  ArraysEvidence<pairs,depth,Packing.passes_done(state)>:
  PassesEvidence{tiles,cursor,equation,balanced} = evidence
  PassesCursor{pass,lane,base,second_lane,kernel_column,unit,window,planes} = cursor
  %Equal.sym(Packing.Passes,state,restore_passes(pairs,tiles,PassesCursor{pass,lane,base,second_lane,kernel_column,unit,window,planes}),equation) :
    ArraysEvidence<pairs,depth,Packing.passes_done(_)>
  ArraysEvidence{tiles,{==},balanced}

def kernel_column_step(+depth: Nat,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,-pairs: Array<Tile.Tile16>,-arrays: Packing.Arrays,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: ArraysEvidence<pairs,depth,arrays>) ->
  ArraysEvidence<pairs,depth,Packing.kernel_column_step(kernel_column,window,planes,position,unit,arrays)>:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  Packing.Planes{+first_output_row,rows,+row_width,elements} = planes
  ArraysEvidence{tiles,equation,balanced} = evidence
  %Equal.sym(Packing.Arrays,arrays,Packing.Arrays{pairs,tiles},equation) : ArraysEvidence<pairs,depth,Packing.kernel_column_step(kernel_column,window,planes,position,unit,_)>
  +column = (position % output_width : U32)
  +second_lane = (output_width - column : U32)
  +count = U32.to_nat(Bool.pick(U32,U32.is_le(8,second_lane),1,(1 + (8 - second_lane + output_width - 1) / output_width : U32)))
  +cursor = {PassesCursor{0,0,((position / output_width - first_output_row) * stride * row_width + column + kernel_column / stride : U32),second_lane,kernel_column,unit,window,
    planes} : PassesCursor}
  passes_done(depth,pairs,Loop.run(~Packing.Passes,~Packing.pass_step,count,restore_passes(pairs,tiles,cursor)),
    pass_loop(depth,count,pairs,restore_passes(pairs,tiles,cursor),pairs_evidence,PassesEvidence{tiles,cursor,{==},balanced}))

def kernel_columns(+depth: Nat,remaining: Nat,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,-pairs: Array<Tile.Tile16>,
  -arrays: Packing.Arrays,+pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: ArraysEvidence<pairs,depth,arrays>) ->
  ArraysEvidence<pairs,depth,Packing.kernel_columns(remaining,kernel_column,window,planes,position,unit,arrays)>:
  match remaining:
    case 0n: evidence
    case 1n+rest:
      kernel_columns(depth,rest,(kernel_column + 1 : U32),window,planes,position,unit,pairs,Packing.kernel_column_step(kernel_column,window,planes,position,unit,arrays),
        pairs_evidence,kernel_column_step(depth,kernel_column,window,planes,position,unit,pairs,arrays,pairs_evidence,evidence))

def units(+depth: Nat,remaining: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,-pairs: Array<Tile.Tile16>,-arrays: Packing.Arrays,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),evidence: ArraysEvidence<pairs,depth,arrays>) ->
  ArraysEvidence<pairs,depth,Packing.units(remaining,window,planes,position,unit,arrays)>:
  match remaining:
    case 0n: evidence
    case 1n+rest:
      Packing.Stencil{base,height,width,tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
      units(depth,rest,window,planes,(position + 8 : U32),(unit + 1 : U32),pairs,
        Packing.kernel_columns(U32.to_nat(tap_columns),0,window,planes,position,unit,arrays),pairs_evidence,
        kernel_columns(depth,U32.to_nat(tap_columns),0,window,planes,position,unit,pairs,arrays,pairs_evidence,evidence))

def packed(-depth: Nat,-pairs: Array<Tile.Tile16>,-arrays: Packing.Arrays,evidence: ArraysEvidence<pairs,depth,arrays>) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile16>,Tile.Tile8,pairs,depth,Packing.packed(arrays)>:
  ArraysEvidence{tiles,equation,balanced} = evidence
  %Equal.sym(Packing.Arrays,arrays,Packing.Arrays{pairs,tiles},equation) : TraversalWitness.ResultEvidence<Array<Tile.Tile16>,Tile.Tile8,pairs,depth,Packing.packed(_)>
  TraversalWitness.ResultEvidence{tiles,{==},balanced}

def pack(+depth: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,-pairs: Array<Tile.Tile16>,-tiles: Array<Tile.Tile8>,
  +pairs_evidence: PackingReadWitness.family(~Tile.Tile16,pairs),balanced: Witness.Witness(Tile.Tile8,depth,tiles)) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile16>,Tile.Tile8,pairs,depth,Packing.pack(window,planes,count,start,pairs,tiles)>:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  +unit_count = Nat.div(count,U32.to_nat(inner))
  packed(depth,pairs,Packing.units(unit_count,window,planes,start,0,Packing.Arrays{pairs,tiles}),
    units(depth,unit_count,window,planes,start,0,pairs,Packing.Arrays{pairs,tiles},pairs_evidence,ArraysEvidence{tiles,{==},balanced}))
