# Shape evidence follows the real readers and writers: a run gives back every
# owner it read and an output of the same balance. Only the final flat
# certificates are live; witness traversals occur beneath erased rewrites.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../math_choice.bend as Choice
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./traversal_witness.bend as TraversalWitness
import ./kernel_tile_witness.bend as TileWitness

# The parts of the step a reader returned.
type Stepped<-Cursor: Data,-read: Gemm.Step<Cursor>> is Data:
  Stepped{lane_address: U32,row_address: U32,next: Cursor,equation: {read == Gemm.Step{lane_address,row_address,next} : Gemm.Step<Cursor>}}

def stepped(-Cursor: Data,read: Gemm.Step<Cursor>) -> Stepped<Cursor,read>:
  Gemm.Step{lane_address,row_address,next} = read
  Stepped{lane_address,row_address,next,{==}}

# A bias is read and the bias owner comes back.
def FetchRead(~fetch: Array<F32> -> U32 -> Array<F32> & F32) -> Type:
  @-bias: Array<F32> -> @+row: U32 -> Witness.ArrayWitness<F32,bias> -> Witness.Read<F32,bias,fetch(bias,row)>

def unbiased_read(-bias: Array<F32>,+row: U32,evidence: Witness.ArrayWitness<F32,bias>) -> Witness.Read<F32,bias,Gemm.unbiased(bias,row)>:
  Witness.Read{0.0,{==}}

type Readers<-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>> is Data:
  Readers{row_tiles: Witness.ArrayWitness<Tile.Tile8,row_tiles>,lane_tiles: Witness.ArrayWitness<Tile.Tile8,lane_tiles>,bias: Witness.ArrayWitness<F32,bias>}

def bias_reader(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,readers: Readers<row_tiles,lane_tiles,bias>) -> Witness.ArrayWitness<F32,bias>:
  Readers{row_evidence,lane_evidence,bias_evidence} = readers
  bias_evidence

type TileReaders<-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>> is Data:
  TileReaders{row_tiles: Witness.ArrayWitness<Tile.Tile8,row_tiles>,lane_tiles: Witness.ArrayWitness<Tile.Tile8,lane_tiles>}

def tile_readers(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,readers: Readers<row_tiles,lane_tiles,bias>) -> TileReaders<row_tiles,lane_tiles>:
  Readers{row_evidence,lane_evidence,bias_evidence} = readers
  TileReaders{row_evidence,lane_evidence}

# The k loop returns every owner it received.
type Looped<-Cursor: Data,-E: Data,-Bias: Type,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,-state: Gemm.Loop<Cursor,E,Bias>> is Data:
  Looped{cursor: Cursor,row: U32,store: Gemm.Store,-accumulators: Gemm.Accumulators,
    equation: {state == Gemm.Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,accumulators} : Gemm.Loop<Cursor,E,Bias>}}

def after_row_tile(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,
  +cursor: Cursor,+row: U32,+store: Gemm.Store,-accumulators: Gemm.Accumulators,-samples: Tile.Tile8,+row_address: U32,
  read: Witness.Read<Tile.Tile8,row_tiles,Array.get(Tile.Tile8,row_tiles,row_address)>) ->
  Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_row_tile(~Cursor,~E,~Bias,~times,lane_tiles,bias,output,cursor,row,store,accumulators,samples,
    Array.get(Tile.Tile8,row_tiles,row_address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,row_tiles,row_address),(row_tiles,value),equation) :
    Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_row_tile(~Cursor,~E,~Bias,~times,lane_tiles,bias,output,cursor,row,store,accumulators,samples,_)>
  Looped{cursor,row,store,Gemm.update(~times,accumulators,samples,value),{==}}

def after_lane_tile(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,
  +row_address: U32,+cursor: Cursor,+row: U32,+store: Gemm.Store,-accumulators: Gemm.Accumulators,+address: U32,
  +row_evidence: Witness.ArrayWitness<Tile.Tile8,row_tiles>,read: Witness.Read<Tile.Tile8,lane_tiles,Array.get(Tile.Tile8,lane_tiles,address)>) ->
  Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_lane_tile(~Cursor,~E,~Bias,~times,row_tiles,bias,output,row_address,cursor,row,store,accumulators,Array.get(Tile.Tile8,lane_tiles,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,lane_tiles,address),(lane_tiles,value),equation) :
    Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_lane_tile(~Cursor,~E,~Bias,~times,row_tiles,bias,output,row_address,cursor,row,store,accumulators,_)>
  after_row_tile(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,cursor,row,store,accumulators,value,row_address,Witness.observed(Tile.Tile8,row_tiles,row_address,row_evidence))

def after_cursor(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,
  +row: U32,+store: Gemm.Store,-accumulators: Gemm.Accumulators,-read: Gemm.Step<Cursor>,
  +row_evidence: Witness.ArrayWitness<Tile.Tile8,row_tiles>,+lane_evidence: Witness.ArrayWitness<Tile.Tile8,lane_tiles>,parts: Stepped<Cursor,read>) ->
  Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_cursor(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,row,store,accumulators,read)>:
  Stepped{+lane_address,+row_address,+next,equation} = parts
  %Equal.sym(Gemm.Step<Cursor>,read,Gemm.Step{lane_address,row_address,next},equation) :
    Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.with_cursor(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,row,store,accumulators,_)>
  after_lane_tile(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,row_address,next,row,store,accumulators,lane_address,row_evidence,
    Witness.observed(Tile.Tile8,lane_tiles,lane_address,lane_evidence))

def step(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  -row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,-state: Gemm.Loop<Cursor,E,Bias>,
  +readers: TileReaders<row_tiles,lane_tiles>,evidence: Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,state>) ->
  Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.step(~Cursor,~E,~Bias,~advance,~times,state)>:
  TileReaders{+row_evidence,+lane_evidence} = readers
  Looped{+cursor,+row,+store,accumulators,equation} = evidence
  %Equal.sym(Gemm.Loop<Cursor,E,Bias>,state,Gemm.Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,accumulators},equation) :
    Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Gemm.step(~Cursor,~E,~Bias,~advance,~times,_)>
  after_cursor(~Cursor,~E,~Bias,~times,row_tiles,lane_tiles,bias,output,row,store,accumulators,advance(cursor),row_evidence,lane_evidence,
    stepped(Cursor,advance(cursor)))

def looped(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,count: Nat,
  -row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Bias,-output: Array<E>,-state: Gemm.Loop<Cursor,E,Bias>,
  +readers: TileReaders<row_tiles,lane_tiles>,evidence: Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,state>) ->
  Looped<Cursor,E,Bias,row_tiles,lane_tiles,bias,output,Loop.run(~Gemm.Loop<Cursor,E,Bias>,~Gemm.step(~Cursor,~E,~Bias,~advance,~times),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      looped(~Cursor,~E,~Bias,~advance,~times,rest,row_tiles,lane_tiles,bias,output,Gemm.step(~Cursor,~E,~Bias,~advance,~times,state),readers,
        step(~Cursor,~E,~Bias,~advance,~times,row_tiles,lane_tiles,bias,output,state,readers,evidence))

# The stores keep the bias owner and the output balance.
type Written<-E: Data,-Bias: Type,-bias: Bias,-depth: Nat,-written: Gemm.Written<E,Bias>> is Data:
  Written{-output: Array<E>,equation: {written == Gemm.Written{bias,output} : Gemm.Written<E,Bias>},balanced: Witness.Witness(E,depth,output)}

# A writer gives back the bias owner and an output of the same balance.
def Keeps(~E: Data,~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>) -> Type:
  @+depth: Nat -> @+bias_start: U32 -> @+store: Gemm.Store -> @-accumulators: Gemm.Accumulators -> @-bias: Array<F32> -> @-output: Array<E> ->
    @+bias_evidence: Witness.ArrayWitness<F32,bias> -> Witness.Witness(E,depth,output) ->
    Written<E,Array<F32>,bias,depth,write(bias_start,store,accumulators,Gemm.Written{bias,output})>

# ---- Puts ----

def put_read(~E: Data,+depth: Nat,inside: Bool,-output: Array<E>,+offset: U32,-value: E,+evidence: Witness.Witness(E,depth,output),
  read: Witness.Read<E,output,Array.get(E,output,offset)>) -> Witness.Witness(E,depth,Gemm.put(~E,inside,output,offset,value)):
  Witness.Read{stored,equation} = read
  %Equal.sym(Array<E> & E,Array.get(E,output,offset),(output,stored),equation) : Witness.Witness(E,depth,Gemm.kept_or(~E,inside,offset,value,_))
  Witness.write(E,depth,output,offset,Choice.chosen(~E,inside,value,stored),evidence)

def put(~E: Data,+depth: Nat,inside: Bool,-output: Array<E>,+offset: U32,-value: E,+evidence: Witness.Witness(E,depth,output)) ->
  Witness.Witness(E,depth,Gemm.put(~E,inside,output,offset,value)):
  put_read(~E,depth,inside,output,offset,value,evidence,Witness.read(E,depth,output,offset,evidence))

# ---- The lane writer ----

# ---- The tile writer ----

def sampled(-bias: Array<F32>,-result: Array<F32> & F32,read: Witness.Read<F32,bias,result>) -> TraversalWitness.Sample<Array<F32>,F32,bias,result>:
  Witness.Read{value,equation} = read
  TraversalWitness.Sample{value,equation}

def bias_family(-bias: Array<F32>) -> Data:
  Witness.ArrayWitness<F32,bias>

def bias_lane(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~fetch_read: FetchRead(~fetch),-bias: Array<F32>,start: U32,lane: Nat,evidence: bias_family(bias)) ->
  TraversalWitness.Sample<Array<F32>,F32,bias,Gemm.bias_lane(~fetch,bias,start,lane)>:
  sampled(bias,fetch(bias,(start + U32.from_nat(lane) : U32)),fetch_read(bias,(start + U32.from_nat(lane) : U32),evidence))

def bias_lanes(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~fetch_read: FetchRead(~fetch),-bias: Array<F32>,+start: U32,+evidence: Witness.ArrayWitness<F32,bias>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,bias,Gemm.bias_lanes(~fetch,bias,start)>:
  TileWitness.tail(~Array<F32>,~U32,~bias_family,~Gemm.bias_lane(~fetch),~bias_lane(~fetch,~fetch_read),bias,start,fetch(bias,start),evidence,
    sampled(bias,fetch(bias,start),fetch_read(bias,start,evidence)))

def tiles_finished(~Bias: Type,~close: F32 -> F32 -> F32,+depth: Nat,+base: U32,+row_stride: U32,+rows: U32,+count: U32,-accumulators: Gemm.Accumulators,
  -output: Array<Tile.Tile8>,-bias: Bias,-result: Bias & Tile.Tile8,sample: TraversalWitness.Sample<Bias,Tile.Tile8,bias,result>,
  evidence: Witness.Witness(Tile.Tile8,depth,output)) ->
  Written<Tile.Tile8,Bias,bias,depth,Gemm.tiles_finished(~Bias,~close,base,row_stride,rows,count,accumulators,output,result)>:
  TraversalWitness.Sample{biases,equation} = sample
  %Equal.sym(Bias & Tile.Tile8,result,(bias,biases),equation) :
    Written<Tile.Tile8,Bias,bias,depth,Gemm.tiles_finished(~Bias,~close,base,row_stride,rows,count,accumulators,output,_)>
  -o1 = Gemm.put(~Tile.Tile8,U32.is_lt(0,rows),output,base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)))
  -o2 = Gemm.put(~Tile.Tile8,U32.is_lt(1,rows),o1,(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators)))
  -o3 = Gemm.put(~Tile.Tile8,U32.is_lt(2,rows),o2,(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators)))
  -o4 = Gemm.put(~Tile.Tile8,U32.is_lt(3,rows),o3,(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators)))
  -o5 = Gemm.put(~Tile.Tile8,U32.is_lt(4,rows),o4,(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators)))
  -o6 = Gemm.put(~Tile.Tile8,U32.is_lt(5,rows),o5,(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators)))
  -o7 = Gemm.put(~Tile.Tile8,U32.is_lt(6,rows),o6,(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators)))
  evidence = put(~Tile.Tile8,depth,U32.is_lt(0,rows),output,base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(1,rows),o1,(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(2,rows),o2,(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(3,rows),o3,(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(4,rows),o4,(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(5,rows),o5,(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators)),evidence)
  evidence = put(~Tile.Tile8,depth,U32.is_lt(6,rows),o6,(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators)),evidence)
  Written{Gemm.put(~Tile.Tile8,U32.is_lt(7,rows),o7,(base + 7 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row7(accumulators))),{==},
    put(~Tile.Tile8,depth,U32.is_lt(7,rows),o7,(base + 7 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row7(accumulators)),evidence)}

def write_tiles(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~fetch_read: FetchRead(~fetch),~close: F32 -> F32 -> F32,+depth: Nat,
  +bias_start: U32,+store: Gemm.Store,-accumulators: Gemm.Accumulators,-bias: Array<F32>,-output: Array<Tile.Tile8>,+bias_evidence: Witness.ArrayWitness<F32,bias>,
  balanced: Witness.Witness(Tile.Tile8,depth,output)) ->
  Written<Tile.Tile8,Array<F32>,bias,depth,Gemm.write_tiles(~fetch,~close,bias_start,store,accumulators,Gemm.Written{bias,output})>:
  Gemm.Store{+base,+row_stride,+lane_stride,+rows,+count} = store
  tiles_finished(~Array<F32>,~close,depth,base,row_stride,rows,count,accumulators,output,bias,Gemm.bias_lanes(~fetch,bias,bias_start),
    bias_lanes(~fetch,~fetch_read,bias,bias_start,bias_evidence),balanced)

# ---- A run ----

# A run gives back the three owners it only reads; the output keeps its balance.
type Kept<-E: Data,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-operands: Gemm.Operands<E,Array<F32>>> is Data:
  Kept{-output: Array<E>,equation: {operands == Gemm.Operands{row_tiles,lane_tiles,bias,output} : Gemm.Operands<E,Array<F32>>},balanced: Witness.Witness(E,depth,output)}

def operands_written(~E: Data,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,+depth: Nat,-written: Gemm.Written<E,Array<F32>>,
  evidence: Written<E,Array<F32>,bias,depth,written>) -> Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.operands_written(~E,row_tiles,lane_tiles,written)>:
  Written{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Written<E,Array<F32>>,written,Gemm.Written{bias,output},equation) : Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.operands_written(~E,row_tiles,lane_tiles,_)>
  Kept{output,{==},balanced}

def finished(~Cursor: Data,~E: Data,~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,
  ~keeps: Keeps(~E,~write),+depth: Nat,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<E>,-state: Gemm.Loop<Cursor,E,Array<F32>>,
  +bias_evidence: Witness.ArrayWitness<F32,bias>,balanced: Witness.Witness(E,depth,output),evidence: Looped<Cursor,E,Array<F32>,row_tiles,lane_tiles,bias,output,state>) ->
  Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.finish(~Cursor,~E,~write,state)>:
  Looped{cursor,+row,+store,accumulators,equation} = evidence
  %Equal.sym(Gemm.Loop<Cursor,E,Array<F32>>,state,Gemm.Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,accumulators},equation) :
    Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.finish(~Cursor,~E,~write,_)>
  operands_written(~E,row_tiles,lane_tiles,bias,depth,write(row,store,accumulators,Gemm.Written{bias,output}),
    keeps(depth,row,store,accumulators,bias,output,bias_evidence,balanced))

# One block: the k loop from its start, then the finish.
def block(~Cursor: Data,~E: Data,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: Keeps(~E,~write),
  +depth: Nat,+job: Gemm.Job<Cursor>,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-operands: Gemm.Operands<E,Array<F32>>,
  +readers: Readers<row_tiles,lane_tiles,bias>,evidence: Kept<E,row_tiles,lane_tiles,bias,depth,operands>) ->
  Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.block(~Cursor,~E,~advance,~times,~write,job,operands)>:
  Gemm.Job{+row,+count,+cursor,+store} = job
  Kept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Operands<E,Array<F32>>,operands,Gemm.Operands{row_tiles,lane_tiles,bias,output},equation) :
    Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.block(~Cursor,~E,~advance,~times,~write,Gemm.Job{row,count,cursor,store},_)>
  -start = {Gemm.Loop{row_tiles,lane_tiles,bias,output,cursor,row,store,Gemm.zero()} : Gemm.Loop<Cursor,E,Array<F32>>}
  finished(~Cursor,~E,~write,~keeps,depth,row_tiles,lane_tiles,bias,output,Loop.run(~Gemm.Loop<Cursor,E,Array<F32>>,~Gemm.step(~Cursor,~E,~Array<F32>,~advance,~times),count,start),
    bias_reader(row_tiles,lane_tiles,bias,readers),balanced,looped(~Cursor,~E,~Array<F32>,~advance,~times,count,row_tiles,lane_tiles,bias,output,start,tile_readers(row_tiles,lane_tiles,bias,readers),Looped{cursor,row,store,Gemm.zero(),{==}}))

def jobs(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: Keeps(~E,~write),
  +depth: Nat,count: Nat,+index: U32,+plan: Plan,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-operands: Gemm.Operands<E,Array<F32>>,
  +readers: Readers<row_tiles,lane_tiles,bias>,evidence: Kept<E,row_tiles,lane_tiles,bias,depth,operands>) ->
  Kept<E,row_tiles,lane_tiles,bias,depth,Traversal.indexed_fold(~Gemm.Operands<E,Array<F32>>,~Plan,~Gemm.job_step(~Plan,~Cursor,~E,~job_at,~advance,~times,~write),
    count,index,plan,operands)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      jobs(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,rest,(index + 1 : U32),plan,row_tiles,lane_tiles,bias,
        Gemm.job_step(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,index,plan,operands),readers,
        block(~Cursor,~E,~advance,~times,~write,~keeps,depth,job_at(plan,index),row_tiles,lane_tiles,bias,operands,readers,evidence))

# Every job of a plan from +0.
def run(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: Keeps(~E,~write),
  +depth: Nat,count: Nat,+plan: Plan,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<E>,
  +readers: Readers<row_tiles,lane_tiles,bias>,balanced: Witness.Witness(E,depth,output)) ->
  Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,count,plan,Gemm.Operands{row_tiles,lane_tiles,bias,output})>:
  match count:
    case 0n:
      jobs(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,0n,0,plan,row_tiles,lane_tiles,bias,Gemm.Operands{row_tiles,lane_tiles,bias,output},
        readers,Kept{output,{==},balanced})
    case 1n+rest:
      jobs(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,1n+rest,0,plan,row_tiles,lane_tiles,bias,Gemm.Operands{row_tiles,lane_tiles,bias,output},
        readers,Kept{output,{==},balanced})

# The live certificates of a run: the read owners keep theirs, the output's
# balance carries over.
type Certified<-E: Data,-operands: Gemm.Operands<E,Array<F32>>> is Data:
  Certified{row_tiles: Certificate.Certificate<Tile.Tile8,Gemm.row_tiles_of(~E,~Array<F32>,operands)>,lane_tiles: Certificate.Certificate<Tile.Tile8,Gemm.lane_tiles_of(~E,~Array<F32>,operands)>,
    bias: Certificate.Certificate<F32,Gemm.bias_of(~E,~Array<F32>,operands)>,output: Certificate.Certificate<E,Gemm.output_of(~E,~Array<F32>,operands)>}

# The operands of a run are the owners it read and its output.
def kept_operands(~E: Data,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-depth: Nat,-operands: Gemm.Operands<E,Array<F32>>,
  evidence: Kept<E,row_tiles,lane_tiles,bias,depth,operands>) -> {Gemm.Operands{row_tiles,lane_tiles,bias,Gemm.output_of(~E,~Array<F32>,operands)} == operands : Gemm.Operands<E,Array<F32>>}:
  Kept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Operands<E,Array<F32>>,operands,Gemm.Operands{row_tiles,lane_tiles,bias,output},equation) :
    {Gemm.Operands{row_tiles,lane_tiles,bias,Gemm.output_of(~E,~Array<F32>,_)} == _ : Gemm.Operands<E,Array<F32>>}
  {==}

# The output keeps the depth of the certificate it came with.
def kept_balance(~E: Data,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,+depth: Nat,-operands: Gemm.Operands<E,Array<F32>>,
  evidence: Kept<E,row_tiles,lane_tiles,bias,depth,operands>) ->
  {Tree.pack(E,Certificate.normalized(E,depth,Tree.reflect(E,Gemm.output_of(~E,~Array<F32>,operands)))) == Gemm.output_of(~E,~Array<F32>,operands) : Array<E>}:
  Kept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Operands<E,Array<F32>>,operands,Gemm.Operands{row_tiles,lane_tiles,bias,output},equation) :
    {Tree.pack(E,Certificate.normalized(E,depth,Tree.reflect(E,Gemm.output_of(~E,~Array<F32>,_)))) == Gemm.output_of(~E,~Array<F32>,_) : Array<E>}
  Certificate.restored(E,depth,output,balanced)

def readers(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,
  row_certificate: Certificate.Certificate<Tile.Tile8,row_tiles>,lane_certificate: Certificate.Certificate<Tile.Tile8,lane_tiles>,
  bias_certificate: Certificate.Certificate<F32,bias>) -> Readers<row_tiles,lane_tiles,bias>:
  Readers{Certificate.witness(Tile.Tile8,row_tiles,row_certificate),Certificate.witness(Tile.Tile8,lane_tiles,lane_certificate),
    Certificate.witness(F32,bias,bias_certificate)}

# Every job of a plan from the certificates of its owners.
def kept(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: Keeps(~E,~write),
  +depth: Nat,count: Nat,+plan: Plan,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<E>,
  row_certificate: Certificate.Certificate<Tile.Tile8,row_tiles>,lane_certificate: Certificate.Certificate<Tile.Tile8,lane_tiles>,
  bias_certificate: Certificate.Certificate<F32,bias>,balanced: {output == Tree.pack(E,Certificate.normalized(E,depth,Tree.reflect(E,output))) : Array<E>}) ->
  Kept<E,row_tiles,lane_tiles,bias,depth,Gemm.run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,count,plan,Gemm.Operands{row_tiles,lane_tiles,bias,output})>:
  run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,count,plan,row_tiles,lane_tiles,bias,output,
    readers(row_tiles,lane_tiles,bias,row_certificate,lane_certificate,bias_certificate),Certificate.witness_at(E,depth,output,balanced))

# Only the certificates are live: the jobs are walked beneath erased rewrites.
def certificates(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: Keeps(~E,~write),
  +count: Nat,+plan: Plan,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<F32>,-output: Array<E>,
  +row_certificate: Certificate.Certificate<Tile.Tile8,row_tiles>,+lane_certificate: Certificate.Certificate<Tile.Tile8,lane_tiles>,
  +bias_certificate: Certificate.Certificate<F32,bias>,output_certificate: Certificate.Certificate<E,output>) ->
  Certified<E,Gemm.run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,count,plan,Gemm.Operands{row_tiles,lane_tiles,bias,output})>:
  Certificate.Certificate{+depth,+balanced} = output_certificate
  -operands = Gemm.run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,count,plan,Gemm.Operands{row_tiles,lane_tiles,bias,output})
  Certified{
    %kept_operands(~E,row_tiles,lane_tiles,bias,depth,operands,
      kept(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,count,plan,row_tiles,lane_tiles,bias,output,row_certificate,lane_certificate,bias_certificate,balanced)) :
      Certificate.Certificate<Tile.Tile8,Gemm.row_tiles_of(~E,~Array<F32>,_)>
    row_certificate,
    %kept_operands(~E,row_tiles,lane_tiles,bias,depth,operands,
      kept(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,count,plan,row_tiles,lane_tiles,bias,output,row_certificate,lane_certificate,bias_certificate,balanced)) :
      Certificate.Certificate<Tile.Tile8,Gemm.lane_tiles_of(~E,~Array<F32>,_)>
    lane_certificate,
    %kept_operands(~E,row_tiles,lane_tiles,bias,depth,operands,
      kept(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,count,plan,row_tiles,lane_tiles,bias,output,row_certificate,lane_certificate,bias_certificate,balanced)) :
      Certificate.Certificate<F32,Gemm.bias_of(~E,~Array<F32>,_)>
    bias_certificate,
    Certificate.Certificate{depth,
      %kept_balance(~E,row_tiles,lane_tiles,bias,depth,operands,
        kept(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,depth,count,plan,row_tiles,lane_tiles,bias,output,row_certificate,lane_certificate,bias_certificate,balanced)) :
        {_ == Tree.pack(E,Certificate.normalized(E,depth,Tree.reflect(E,Gemm.output_of(~E,~Array<F32>,operands)))) : Array<E>}
      {==}}}

# Cell biases use the shared product loop and tile stores. All read owners
# return unchanged; the output retains its allocation depth.
type CellKept<-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,-depth: Nat,-state: Gemm.CellOperands> is Data:
  CellKept{-output: Array<Tile.Tile8>,equation: {state == Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}} : Gemm.CellOperands},
    balanced: Witness.Witness(Tile.Tile8,depth,output)}

def cell_stored(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,-written: Gemm.Written<Tile.Tile8,Unit>,
  evidence: Written<Tile.Tile8,Unit,unused_bias,depth,written>) -> CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_stored(row_tiles,lane_tiles,bias,written)>:
  Written{output,equation,balanced} = evidence
  %Equal.sym(Gemm.Written<Tile.Tile8,Unit>,written,Gemm.Written{unused_bias,output},equation) : CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_stored(row_tiles,lane_tiles,bias,_)>
  CellKept{output,{==},balanced}

def cell_read(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,-output: Array<Tile.Tile8>,
  +depth: Nat,+store: Gemm.Store,-accumulators: Gemm.Accumulators,+address: U32,
  sample: Witness.Read<Tile.Tile8,bias,Array.get(Tile.Tile8,bias,address)>,balanced: Witness.Witness(Tile.Tile8,depth,output)) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_read(row_tiles,lane_tiles,unused_bias,output,store,accumulators,Array.get(Tile.Tile8,bias,address))>:
  Gemm.Store{+base,+row_stride,lane_stride,+rows,+count} = store
  Witness.Read{biases,equation} = sample
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,bias,address),(bias,biases),equation) :
    CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_read(row_tiles,lane_tiles,unused_bias,output,store,accumulators,_)>
  cell_stored(row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.tiles_finished(~Unit,~Gemm.plus,base,row_stride,rows,count,accumulators,output,(unused_bias,biases)),
    tiles_finished(~Unit,~Gemm.plus,depth,base,row_stride,rows,count,accumulators,output,unused_bias,(unused_bias,biases),TraversalWitness.Sample{biases,{==}},balanced))

def cell_finished(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,-output: Array<Tile.Tile8>,+depth: Nat,
  -state: Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,balanced: Witness.Witness(Tile.Tile8,depth,output),
  evidence: Looped<Gemm.Direct(),Tile.Tile8,Unit,row_tiles,lane_tiles,unused_bias,output,state>) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_finish(bias,state)>:
  Looped{+cursor,+row,+store,accumulators,equation} = evidence
  %Equal.sym(Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>,state,Gemm.Loop{row_tiles,lane_tiles,unused_bias,output,cursor,row,store,accumulators},equation) :
    CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_finish(bias,_)>
  cell_read(row_tiles,lane_tiles,bias,unused_bias,output,depth,store,accumulators,(row / 8 : U32),Witness.observed(Tile.Tile8,bias,(row / 8 : U32),bias_evidence),balanced)

def cell_block(+job: Gemm.Job<Gemm.Direct()>,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,
  -state: Gemm.CellOperands,+readers: TileReaders<row_tiles,lane_tiles>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,
  evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,
    Gemm.cell_finish(bias,Gemm.looped(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times,job,
      (Gemm.Operands{row_tiles,lane_tiles,unused_bias,Gemm.cell_output(state)},Gemm.zero())))>:
  Gemm.Job{+row,+count,+cursor,+store} = job
  CellKept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.CellOperands,state,Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}},equation) :
    CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_finish(bias,Gemm.looped(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times,job,
      (Gemm.Operands{row_tiles,lane_tiles,unused_bias,Gemm.cell_output(_)},Gemm.zero())))>
  -start = {Gemm.Loop{row_tiles,lane_tiles,unused_bias,output,cursor,row,store,Gemm.zero()} : Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>}
  cell_finished(row_tiles,lane_tiles,bias,unused_bias,output,depth,
    Loop.run(~Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>,~Gemm.step(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times),count,start),bias_evidence,balanced,
    looped(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times,count,row_tiles,lane_tiles,unused_bias,output,start,readers,Looped{cursor,row,store,Gemm.zero(),{==}}))

def cell_operands(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,-depth: Nat,-state: Gemm.CellOperands,
  evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  {Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,Gemm.cell_output(state)}} == state : Gemm.CellOperands}:
  CellKept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.CellOperands,state,Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}},equation) :
    {Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,Gemm.cell_output(_)}} == _ : Gemm.CellOperands}
  {==}

def cell_job(+index: U32,+matrix: Gemm.TileMatrix,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,
  -state: Gemm.CellOperands,+readers: TileReaders<row_tiles,lane_tiles>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,
  +evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_job(index,matrix,state)>:
  %cell_operands(row_tiles,lane_tiles,bias,unused_bias,depth,state,evidence) : CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_job(index,matrix,_)>
  cell_block(Gemm.tile_major(matrix,index),row_tiles,lane_tiles,bias,unused_bias,depth,state,readers,bias_evidence,evidence)

def cell_jobs(count: Nat,+index: U32,+matrix: Gemm.TileMatrix,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,
  -state: Gemm.CellOperands,+readers: TileReaders<row_tiles,lane_tiles>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,
  evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Traversal.indexed_fold(~Gemm.CellOperands,~Gemm.TileMatrix,~Gemm.cell_job,count,index,matrix,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      cell_jobs(rest,(index + 1 : U32),matrix,row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_job(index,matrix,state),readers,bias_evidence,
        cell_job(index,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,state,readers,bias_evidence,evidence))

type CellCertified<-state: Gemm.CellOperands> is Data:
  CellCertified{bias: Certificate.Certificate<Tile.Tile8,Gemm.cell_bias(state)>,
    rows: Certificate.Certificate<Tile.Tile8,Gemm.row_tiles_of(~Tile.Tile8,~Unit,Gemm.cell_product(state))>,
    lanes: Certificate.Certificate<Tile.Tile8,Gemm.lane_tiles_of(~Tile.Tile8,~Unit,Gemm.cell_product(state))>,
    output: Certificate.Certificate<Tile.Tile8,Gemm.cell_output(state)>}

def cell_balance(-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,-state: Gemm.CellOperands,
  evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  {Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,Gemm.cell_output(state)))) == Gemm.cell_output(state) : Array<Tile.Tile8>}:
  CellKept{output,equation,balanced} = evidence
  %Equal.sym(Gemm.CellOperands,state,Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}},equation) :
    {Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,Gemm.cell_output(_)))) == Gemm.cell_output(_) : Array<Tile.Tile8>}
  Certificate.restored(Tile.Tile8,depth,output,balanced)

def cell_run_same(count: Nat,+matrix: Gemm.TileMatrix,-state: Gemm.CellOperands) ->
  {Gemm.cell_run(count,matrix,state) == Traversal.indexed_fold(~Gemm.CellOperands,~Gemm.TileMatrix,~Gemm.cell_job,count,0,matrix,state) : Gemm.CellOperands}:
  match count:
    case 0n: {==}
    case 1n+rest: {==}

def cell_run_kept(+count: Nat,+matrix: Gemm.TileMatrix,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,+depth: Nat,
  -state: Gemm.CellOperands,+readers: TileReaders<row_tiles,lane_tiles>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,
  evidence: CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,state>) ->
  CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,Gemm.cell_run(count,matrix,state)>:
  %Equal.sym(Gemm.CellOperands,Gemm.cell_run(count,matrix,state),Traversal.indexed_fold(~Gemm.CellOperands,~Gemm.TileMatrix,~Gemm.cell_job,count,0,matrix,state),cell_run_same(count,matrix,state)) :
    CellKept<row_tiles,lane_tiles,bias,unused_bias,depth,_>
  cell_jobs(count,0,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,state,readers,bias_evidence,evidence)

def cell_certificates(+count: Nat,+matrix: Gemm.TileMatrix,-row_tiles: Array<Tile.Tile8>,-lane_tiles: Array<Tile.Tile8>,-bias: Array<Tile.Tile8>,-unused_bias: Unit,-output: Array<Tile.Tile8>,
  +row_certificate: Certificate.Certificate<Tile.Tile8,row_tiles>,+lane_certificate: Certificate.Certificate<Tile.Tile8,lane_tiles>,
  +bias_certificate: Certificate.Certificate<Tile.Tile8,bias>,output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  CellCertified<Gemm.cell_run(count,matrix,Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}})>:
  Certificate.Certificate{+depth,+balanced} = output_certificate
  -initial = {Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}} : Gemm.CellOperands}
  -state = Gemm.cell_run(count,matrix,initial)
  CellCertified{
    %cell_operands(row_tiles,lane_tiles,bias,unused_bias,depth,state,
      cell_run_kept(count,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,initial,
        TileReaders{Certificate.witness(Tile.Tile8,row_tiles,row_certificate),Certificate.witness(Tile.Tile8,lane_tiles,lane_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        CellKept{output,{==},Certificate.witness_at(Tile.Tile8,depth,output,balanced)})) :
      Certificate.Certificate<Tile.Tile8,Gemm.cell_bias(_)> bias_certificate,
    %cell_operands(row_tiles,lane_tiles,bias,unused_bias,depth,state,
      cell_run_kept(count,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,initial,
        TileReaders{Certificate.witness(Tile.Tile8,row_tiles,row_certificate),Certificate.witness(Tile.Tile8,lane_tiles,lane_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        CellKept{output,{==},Certificate.witness_at(Tile.Tile8,depth,output,balanced)})) :
      Certificate.Certificate<Tile.Tile8,Gemm.row_tiles_of(~Tile.Tile8,~Unit,Gemm.cell_product(_))> row_certificate,
    %cell_operands(row_tiles,lane_tiles,bias,unused_bias,depth,state,
      cell_run_kept(count,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,initial,
        TileReaders{Certificate.witness(Tile.Tile8,row_tiles,row_certificate),Certificate.witness(Tile.Tile8,lane_tiles,lane_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        CellKept{output,{==},Certificate.witness_at(Tile.Tile8,depth,output,balanced)})) :
      Certificate.Certificate<Tile.Tile8,Gemm.lane_tiles_of(~Tile.Tile8,~Unit,Gemm.cell_product(_))> lane_certificate,
    Certificate.Certificate{depth,
      %cell_balance(row_tiles,lane_tiles,bias,unused_bias,depth,state,
        cell_run_kept(count,matrix,row_tiles,lane_tiles,bias,unused_bias,depth,initial,
          TileReaders{Certificate.witness(Tile.Tile8,row_tiles,row_certificate),Certificate.witness(Tile.Tile8,lane_tiles,lane_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
          CellKept{output,{==},Certificate.witness_at(Tile.Tile8,depth,output,balanced)})) :
        {_ == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,Gemm.cell_output(state)))) : Array<Tile.Tile8>} {==}}}

# ---- Rows read from cells, sixteen channels a block ----

# The input cells and the pairs are only read; the k loop gives them back.
type PointReaders<-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>> is Data:
  PointReaders{cells: Witness.ArrayWitness<Tile.Tile8,cells>,pairs: Witness.ArrayWitness<Tile.Tile16,pairs>}

type PointLooped<-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop> is Data:
  PointLooped{cursor: Gemm.Points,-accumulators: Gemm.Quad,equation: {state == Gemm.PointLoop{cells,pairs,cursor,accumulators} : Gemm.PointLoop}}

# The four row lanes come back with the cells they were read from.
type RowsRead<-cells: Array<Tile.Tile8>,-read: Array<Tile.Tile8> & Gemm.Row4> is Data:
  RowsRead{-values: Gemm.Row4,equation: {read == (cells,values) : Array<Tile.Tile8> & Gemm.Row4}}

def point_row_3(~lane: Tile.Tile8 -> F32,-cells: Array<Tile.Tile8>,-x0: F32,-x1: F32,-x2: F32,+address: U32,
  read: Witness.Read<Tile.Tile8,cells,Array.get(Tile.Tile8,cells,address)>) -> RowsRead<cells,Gemm.point_row_3(~lane,x0,x1,x2,Array.get(Tile.Tile8,cells,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cells,address),(cells,value),equation) : RowsRead<cells,Gemm.point_row_3(~lane,x0,x1,x2,_)>
  RowsRead{Gemm.Row4{x0,x1,x2,lane(value)},{==}}

def point_row_2(~lane: Tile.Tile8 -> F32,-cells: Array<Tile.Tile8>,+cell: U32,+stride: U32,-x0: F32,-x1: F32,+address: U32,+evidence: Witness.ArrayWitness<Tile.Tile8,cells>,
  read: Witness.Read<Tile.Tile8,cells,Array.get(Tile.Tile8,cells,address)>) -> RowsRead<cells,Gemm.point_row_2(~lane,cell,stride,x0,x1,Array.get(Tile.Tile8,cells,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cells,address),(cells,value),equation) : RowsRead<cells,Gemm.point_row_2(~lane,cell,stride,x0,x1,_)>
  point_row_3(~lane,cells,x0,x1,lane(value),(cell + 3 * stride : U32),Witness.observed(Tile.Tile8,cells,(cell + 3 * stride : U32),evidence))

def point_row_1(~lane: Tile.Tile8 -> F32,-cells: Array<Tile.Tile8>,+cell: U32,+stride: U32,-x0: F32,+address: U32,+evidence: Witness.ArrayWitness<Tile.Tile8,cells>,
  read: Witness.Read<Tile.Tile8,cells,Array.get(Tile.Tile8,cells,address)>) -> RowsRead<cells,Gemm.point_row_1(~lane,cell,stride,x0,Array.get(Tile.Tile8,cells,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cells,address),(cells,value),equation) : RowsRead<cells,Gemm.point_row_1(~lane,cell,stride,x0,_)>
  point_row_2(~lane,cells,cell,stride,x0,lane(value),(cell + 2 * stride : U32),evidence,Witness.observed(Tile.Tile8,cells,(cell + 2 * stride : U32),evidence))

def point_row_0(~lane: Tile.Tile8 -> F32,-cells: Array<Tile.Tile8>,+cell: U32,+stride: U32,+evidence: Witness.ArrayWitness<Tile.Tile8,cells>,
  read: Witness.Read<Tile.Tile8,cells,Array.get(Tile.Tile8,cells,cell)>) -> RowsRead<cells,Gemm.point_row_0(~lane,cell,stride,Array.get(Tile.Tile8,cells,cell))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cells,cell),(cells,value),equation) : RowsRead<cells,Gemm.point_row_0(~lane,cell,stride,_)>
  point_row_1(~lane,cells,cell,stride,lane(value),(cell + 1 * stride : U32),evidence,Witness.observed(Tile.Tile8,cells,(cell + 1 * stride : U32),evidence))

def point_products(~times: F32 -> F32 -> F32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+cursor: Gemm.Points,-accumulators: Gemm.Quad,-weight: Tile.Tile16,
  -read: Array<Tile.Tile8> & Gemm.Row4,rows: RowsRead<cells,read>) -> PointLooped<cells,pairs,Gemm.point_products(~times,pairs,cursor,accumulators,weight,read)>:
  RowsRead{values,equation} = rows
  %Equal.sym(Array<Tile.Tile8> & Gemm.Row4,read,(cells,values),equation) : PointLooped<cells,pairs,Gemm.point_products(~times,pairs,cursor,accumulators,weight,_)>
  PointLooped{cursor,Gemm.point_updated(~times,accumulators,weight,values),{==}}

def point_weight(~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+cell: U32,+stride: U32,+cursor: Gemm.Points,
  -accumulators: Gemm.Quad,+address: U32,+cell_evidence: Witness.ArrayWitness<Tile.Tile8,cells>,read: Witness.Read<Tile.Tile16,pairs,Array.get(Tile.Tile16,pairs,address)>) ->
  PointLooped<cells,pairs,Gemm.point_weight(~lane,~times,cells,cell,stride,cursor,accumulators,Array.get(Tile.Tile16,pairs,address))>:
  Witness.Read{weight,equation} = read
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,pairs,address),(pairs,weight),equation) :
    PointLooped<cells,pairs,Gemm.point_weight(~lane,~times,cells,cell,stride,cursor,accumulators,_)>
  point_products(~times,cells,pairs,cursor,accumulators,weight,Gemm.point_rows(~lane,cells,cell,stride),
    point_row_0(~lane,cells,cell,stride,cell_evidence,Witness.observed(Tile.Tile8,cells,cell,cell_evidence)))

def point_at(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+cursor: Gemm.Points,
  -accumulators: Gemm.Quad,+readers: PointReaders<cells,pairs>) ->
  PointLooped<cells,pairs,Gemm.point_step(~at,~lane,~times,Gemm.PointLoop{cells,pairs,cursor,accumulators})>:
  Gemm.Points{+address,+cell,+stride} = cursor
  PointReaders{+cell_evidence,+pair_evidence} = readers
  point_weight(~lane,~times,cells,pairs,at(cell),stride,Gemm.Points{(address + 1 : U32),cell,stride},accumulators,address,cell_evidence,
    Witness.observed(Tile.Tile16,pairs,address,pair_evidence))

def point_step(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,
  +readers: PointReaders<cells,pairs>,evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.point_step(~at,~lane,~times,state)>:
  PointLooped{cursor,accumulators,equation} = evidence
  %Equal.sym(Gemm.PointLoop,state,Gemm.PointLoop{cells,pairs,cursor,accumulators},equation) : PointLooped<cells,pairs,Gemm.point_step(~at,~lane,~times,_)>
  point_at(~at,~lane,~times,cells,pairs,cursor,accumulators,readers)

def next_at(-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,cursor: Gemm.Points,-accumulators: Gemm.Quad) ->
  PointLooped<cells,pairs,Gemm.next_cells(Gemm.PointLoop{cells,pairs,cursor,accumulators})>:
  Gemm.Points{address,cell,stride} = cursor
  PointLooped{Gemm.Points{address,(cell + 2 : U32),stride},accumulators,{==}}

def next_cells(-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,evidence: PointLooped<cells,pairs,state>) ->
  PointLooped<cells,pairs,Gemm.next_cells(state)>:
  PointLooped{cursor,accumulators,equation} = evidence
  %Equal.sym(Gemm.PointLoop,state,Gemm.PointLoop{cells,pairs,cursor,accumulators},equation) : PointLooped<cells,pairs,Gemm.next_cells(_)>
  next_at(cells,pairs,cursor,accumulators)

def point_group(~times: F32 -> F32 -> F32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.point_group(~times,state)>:
  -s0 = Gemm.point_step(~Gemm.first_cell,~Tile.lane0,~times,state)
  evidence = point_step(~Gemm.first_cell,~Tile.lane0,~times,cells,pairs,state,readers,evidence)
  -s1 = Gemm.point_step(~Gemm.first_cell,~Tile.lane1,~times,s0)
  evidence = point_step(~Gemm.first_cell,~Tile.lane1,~times,cells,pairs,s0,readers,evidence)
  -s2 = Gemm.point_step(~Gemm.first_cell,~Tile.lane2,~times,s1)
  evidence = point_step(~Gemm.first_cell,~Tile.lane2,~times,cells,pairs,s1,readers,evidence)
  -s3 = Gemm.point_step(~Gemm.first_cell,~Tile.lane3,~times,s2)
  evidence = point_step(~Gemm.first_cell,~Tile.lane3,~times,cells,pairs,s2,readers,evidence)
  -s4 = Gemm.point_step(~Gemm.first_cell,~Tile.lane4,~times,s3)
  evidence = point_step(~Gemm.first_cell,~Tile.lane4,~times,cells,pairs,s3,readers,evidence)
  -s5 = Gemm.point_step(~Gemm.first_cell,~Tile.lane5,~times,s4)
  evidence = point_step(~Gemm.first_cell,~Tile.lane5,~times,cells,pairs,s4,readers,evidence)
  -s6 = Gemm.point_step(~Gemm.first_cell,~Tile.lane6,~times,s5)
  evidence = point_step(~Gemm.first_cell,~Tile.lane6,~times,cells,pairs,s5,readers,evidence)
  -s7 = Gemm.point_step(~Gemm.first_cell,~Tile.lane7,~times,s6)
  evidence = point_step(~Gemm.first_cell,~Tile.lane7,~times,cells,pairs,s6,readers,evidence)
  -s8 = Gemm.point_step(~Gemm.second_cell,~Tile.lane0,~times,s7)
  evidence = point_step(~Gemm.second_cell,~Tile.lane0,~times,cells,pairs,s7,readers,evidence)
  -s9 = Gemm.point_step(~Gemm.second_cell,~Tile.lane1,~times,s8)
  evidence = point_step(~Gemm.second_cell,~Tile.lane1,~times,cells,pairs,s8,readers,evidence)
  -s10 = Gemm.point_step(~Gemm.second_cell,~Tile.lane2,~times,s9)
  evidence = point_step(~Gemm.second_cell,~Tile.lane2,~times,cells,pairs,s9,readers,evidence)
  -s11 = Gemm.point_step(~Gemm.second_cell,~Tile.lane3,~times,s10)
  evidence = point_step(~Gemm.second_cell,~Tile.lane3,~times,cells,pairs,s10,readers,evidence)
  -s12 = Gemm.point_step(~Gemm.second_cell,~Tile.lane4,~times,s11)
  evidence = point_step(~Gemm.second_cell,~Tile.lane4,~times,cells,pairs,s11,readers,evidence)
  -s13 = Gemm.point_step(~Gemm.second_cell,~Tile.lane5,~times,s12)
  evidence = point_step(~Gemm.second_cell,~Tile.lane5,~times,cells,pairs,s12,readers,evidence)
  -s14 = Gemm.point_step(~Gemm.second_cell,~Tile.lane6,~times,s13)
  evidence = point_step(~Gemm.second_cell,~Tile.lane6,~times,cells,pairs,s13,readers,evidence)
  -s15 = Gemm.point_step(~Gemm.second_cell,~Tile.lane7,~times,s14)
  evidence = point_step(~Gemm.second_cell,~Tile.lane7,~times,cells,pairs,s14,readers,evidence)
  next_cells(cells,pairs,s15,evidence)

def point_groups(~times: F32 -> F32 -> F32,count: Nat,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Loop.run(~Gemm.PointLoop,~Gemm.point_group(~times),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      point_groups(~times,rest,cells,pairs,Gemm.point_group(~times,state),readers,point_group(~times,cells,pairs,state,readers,evidence))

def point_start(-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+matrix: Gemm.PointMatrix,+place: Gemm.Place) ->
  PointLooped<cells,pairs,Gemm.point_start(cells,pairs,matrix,place)>:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,output_positions},+cell,+stride} = matrix
  Gemm.Place{+pair,+unit} = place
  PointLooped{Gemm.Points{(pair * inner : U32),(cell + unit * 4 * stride : U32),stride},Gemm.quad_zero(),{==}}

# The bias, the cells and the pairs come back; the staged output keeps its
# allocation depth.
type PointKept<-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-depth: Nat,-state: Gemm.PointOperands> is Data:
  PointKept{-staged: Array<Tile.Tile16>,equation: {state == Gemm.PointOperands{bias,cells,pairs,staged} : Gemm.PointOperands},
    balanced: Witness.Witness(Tile.Tile16,depth,staged)}

def point_stored_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,+depth: Nat,-staged: Array<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,-biases: Tile.Tile16,-accumulators: Gemm.Quad,
  evidence: Witness.Witness(Tile.Tile16,depth,staged)) -> Witness.Witness(Tile.Tile16,depth,Gemm.point_stored_with(~add,staged,base,stride,rows,biases,accumulators)):
  -o1 = Gemm.put(~Tile.Tile16,U32.is_lt(0,rows),staged,base,add(Gemm.quad0(accumulators),biases))
  -o2 = Gemm.put(~Tile.Tile16,U32.is_lt(1,rows),o1,(base + stride : U32),add(Gemm.quad1(accumulators),biases))
  -o3 = Gemm.put(~Tile.Tile16,U32.is_lt(2,rows),o2,(base + 2 * stride : U32),add(Gemm.quad2(accumulators),biases))
  evidence = put(~Tile.Tile16,depth,U32.is_lt(0,rows),staged,base,add(Gemm.quad0(accumulators),biases),evidence)
  evidence = put(~Tile.Tile16,depth,U32.is_lt(1,rows),o1,(base + stride : U32),add(Gemm.quad1(accumulators),biases),evidence)
  evidence = put(~Tile.Tile16,depth,U32.is_lt(2,rows),o2,(base + 2 * stride : U32),add(Gemm.quad2(accumulators),biases),evidence)
  put(~Tile.Tile16,depth,U32.is_lt(3,rows),o3,(base + 3 * stride : U32),add(Gemm.quad3(accumulators),biases),evidence)

def upper_biased_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-staged: Array<Tile.Tile16>,+depth: Nat,+base: U32,+stride: U32,+rows: U32,
  -lower: Tile.Tile8,-accumulators: Gemm.Quad,+address: U32,read: Witness.Read<Tile.Tile8,bias,Array.get(Tile.Tile8,bias,address)>,
  balanced: Witness.Witness(Tile.Tile16,depth,staged)) ->
  PointKept<bias,cells,pairs,depth,Gemm.upper_biased_with(~add,cells,pairs,staged,base,stride,rows,lower,accumulators,Array.get(Tile.Tile8,bias,address))>:
  Witness.Read{upper,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,bias,address),(bias,upper),equation) :
    PointKept<bias,cells,pairs,depth,Gemm.upper_biased_with(~add,cells,pairs,staged,base,stride,rows,lower,accumulators,_)>
  PointKept{Gemm.point_stored_with(~add,staged,base,stride,rows,Tile.Tiles{lower,upper},accumulators),{==},
    point_stored_with(~add,depth,staged,base,stride,rows,Tile.Tiles{lower,upper},accumulators,balanced)}

def lower_biased_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-staged: Array<Tile.Tile16>,+depth: Nat,+base: U32,+stride: U32,+rows: U32,
  +above: U32,-accumulators: Gemm.Quad,+address: U32,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,read: Witness.Read<Tile.Tile8,bias,Array.get(Tile.Tile8,bias,address)>,
  balanced: Witness.Witness(Tile.Tile16,depth,staged)) ->
  PointKept<bias,cells,pairs,depth,Gemm.lower_biased_with(~add,cells,pairs,staged,base,stride,rows,above,accumulators,Array.get(Tile.Tile8,bias,address))>:
  Witness.Read{lower,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,bias,address),(bias,lower),equation) :
    PointKept<bias,cells,pairs,depth,Gemm.lower_biased_with(~add,cells,pairs,staged,base,stride,rows,above,accumulators,_)>
  upper_biased_with(~add,bias,cells,pairs,staged,depth,base,stride,rows,lower,accumulators,above,Witness.observed(Tile.Tile8,bias,above,bias_evidence),balanced)

def point_finish_with(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-staged: Array<Tile.Tile16>,+depth: Nat,+matrix: Gemm.PointMatrix,
  +place: Gemm.Place,-state: Gemm.PointLoop,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,balanced: Witness.Witness(Tile.Tile16,depth,staged),
  evidence: PointLooped<cells,pairs,state>) -> PointKept<bias,cells,pairs,depth,Gemm.point_finish_with(~add,bias,staged,matrix,place,state)>:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,positions,inner,+start,+output_positions},cell,stride} = matrix
  Gemm.Place{+pair,+unit} = place
  PointLooped{cursor,accumulators,equation} = evidence
  %Equal.sym(Gemm.PointLoop,state,Gemm.PointLoop{cells,pairs,cursor,accumulators},equation) : PointKept<bias,cells,pairs,depth,Gemm.point_finish_with(~add,bias,staged,matrix,place,_)>
  +groups = Gemm.blocks(channels)
  +pairs_count = Gemm.pairs_of(channels)
  +position = (start + unit * 4 : U32)
  lower_biased_with(~add,bias,cells,pairs,staged,depth,(position * pairs_count + pair : U32),pairs_count,U32.min(4,(output_positions - position : U32)),
    U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)),accumulators,(pair * 2 : U32),bias_evidence,Witness.observed(Tile.Tile8,bias,(pair * 2 : U32),bias_evidence),balanced)

def PointTail(~tail: @+matrix: Gemm.PointMatrix -> Gemm.PointLoop -> Gemm.PointLoop) -> Type:
  @+matrix: Gemm.PointMatrix -> @-cells: Array<Tile.Tile8> -> @-pairs: Array<Tile.Tile16> -> @-state: Gemm.PointLoop ->
  @+readers: PointReaders<cells,pairs> -> PointLooped<cells,pairs,state> -> PointLooped<cells,pairs,tail(matrix,state)>

def whole_inner(+matrix: Gemm.PointMatrix,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.whole_inner(matrix,state)>:
  evidence

def tail_step(~at: U32 -> U32,lane: Nat,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.tail_step(~at,lane,state)>:
  match lane:
    case 0n: point_step(~at,~Tile.lane0,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 1n: point_step(~at,~Tile.lane1,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 2n: point_step(~at,~Tile.lane2,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 3n: point_step(~at,~Tile.lane3,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 4n: point_step(~at,~Tile.lane4,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 5n: point_step(~at,~Tile.lane5,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 6n: point_step(~at,~Tile.lane6,~Gemm.tail_times,cells,pairs,state,readers,evidence)
    case 7n+rest: point_step(~at,~Tile.lane7,~Gemm.tail_times,cells,pairs,state,readers,evidence)

def tail_cell(~at: U32 -> U32,count: Nat,+lane: U32,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.tail_cell(~at,count,lane,state)>:
  match count:
    case 0n: evidence
    case 1n+rest: tail_cell(~at,rest,(lane + 1 : U32),cells,pairs,Gemm.tail_step(~at,U32.to_nat(lane),state),readers,
      tail_step(~at,U32.to_nat(lane),cells,pairs,state,readers,evidence))

def finish_inner(+matrix: Gemm.PointMatrix,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-state: Gemm.PointLoop,+readers: PointReaders<cells,pairs>,
  evidence: PointLooped<cells,pairs,state>) -> PointLooped<cells,pairs,Gemm.finish_inner(matrix,state)>:
  +tail = Gemm.point_remainder(matrix)
  +lower = U32.min(tail,8)
  tail_cell(~Gemm.second_cell,U32.to_nat((tail - lower : U32)),0,cells,pairs,Gemm.tail_cell(~Gemm.first_cell,U32.to_nat(lower),0,state),readers,
    tail_cell(~Gemm.first_cell,U32.to_nat(lower),0,cells,pairs,state,readers,evidence))

def point_job(~add: Tile.Tile16 -> Tile.Tile16 -> Tile.Tile16,~times: F32 -> F32 -> F32,~tail: @+matrix: Gemm.PointMatrix -> Gemm.PointLoop -> Gemm.PointLoop,~tail_keeps: PointTail(~tail),+index: U32,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+depth: Nat,-state: Gemm.PointOperands,
  +readers: PointReaders<cells,pairs>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,evidence: PointKept<bias,cells,pairs,depth,state>) ->
  PointKept<bias,cells,pairs,depth,Gemm.point_job_with(~add,~times,~tail,index,matrix,state)>:
  PointKept{staged,equation,balanced} = evidence
  %Equal.sym(Gemm.PointOperands,state,Gemm.PointOperands{bias,cells,pairs,staged},equation) : PointKept<bias,cells,pairs,depth,Gemm.point_job_with(~add,~times,~tail,index,matrix,_)>
  +place = Gemm.point_place(matrix,index)
  -start = Gemm.point_start(cells,pairs,matrix,place)
  -grouped = Loop.run(~Gemm.PointLoop,~Gemm.point_group(~times),Gemm.point_groups_of(matrix),start)
  point_finish_with(~add,bias,cells,pairs,staged,depth,matrix,place,tail(matrix,grouped),bias_evidence,balanced,
    tail_keeps(matrix,cells,pairs,grouped,readers,point_groups(~times,Gemm.point_groups_of(matrix),cells,pairs,start,readers,point_start(cells,pairs,matrix,place))))

def PointJob(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands) -> Type:
  @+index: U32 -> @+matrix: Gemm.PointMatrix -> @-bias: Array<Tile.Tile8> -> @-cells: Array<Tile.Tile8> -> @-pairs: Array<Tile.Tile16> -> @+depth: Nat -> @-state: Gemm.PointOperands ->
  @+readers: PointReaders<cells,pairs> -> @+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias> ->
  PointKept<bias,cells,pairs,depth,state> -> PointKept<bias,cells,pairs,depth,job(index,matrix,state)>

def point_job_whole(+index: U32,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+depth: Nat,-state: Gemm.PointOperands,
  +readers: PointReaders<cells,pairs>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,evidence: PointKept<bias,cells,pairs,depth,state>) ->
  PointKept<bias,cells,pairs,depth,Gemm.point_job(index,matrix,state)>:
  PointKept{staged,equation,balanced} = evidence
  %Equal.sym(Gemm.PointOperands,state,Gemm.PointOperands{bias,cells,pairs,staged},equation) : PointKept<bias,cells,pairs,depth,Gemm.point_job(index,matrix,_)>
  point_job(~Tile.plus_pair,~Gemm.row_times,~Gemm.whole_inner,~whole_inner,index,matrix,bias,cells,pairs,depth,Gemm.PointOperands{bias,cells,pairs,staged},readers,bias_evidence,
    PointKept{staged,{==},balanced})

def point_jobs(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,~keeps: PointJob(~job),count: Nat,+index: U32,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+depth: Nat,
  -state: Gemm.PointOperands,+readers: PointReaders<cells,pairs>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,evidence: PointKept<bias,cells,pairs,depth,state>) ->
  PointKept<bias,cells,pairs,depth,Traversal.indexed_fold(~Gemm.PointOperands,~Gemm.PointMatrix,~job,count,index,matrix,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      point_jobs(~job,~keeps,rest,(index + 1 : U32),matrix,bias,cells,pairs,depth,job(index,matrix,state),readers,bias_evidence,
        keeps(index,matrix,bias,cells,pairs,depth,state,readers,bias_evidence,evidence))

def point_run_same(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,count: Nat,+matrix: Gemm.PointMatrix,-state: Gemm.PointOperands) ->
  {Gemm.point_run_indexed(~job,count,matrix,state) == Traversal.indexed_fold(~Gemm.PointOperands,~Gemm.PointMatrix,~job,count,0,matrix,state) : Gemm.PointOperands}:
  match count:
    case 0n: {==}
    case 1n: {==}
    case 2n+rest: {==}

def point_run_kept(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,~keeps: PointJob(~job),+count: Nat,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+depth: Nat,
  -state: Gemm.PointOperands,+readers: PointReaders<cells,pairs>,+bias_evidence: Witness.ArrayWitness<Tile.Tile8,bias>,evidence: PointKept<bias,cells,pairs,depth,state>) ->
  PointKept<bias,cells,pairs,depth,Gemm.point_run_indexed(~job,count,matrix,state)>:
  %Equal.sym(Gemm.PointOperands,Gemm.point_run_indexed(~job,count,matrix,state),Traversal.indexed_fold(~Gemm.PointOperands,~Gemm.PointMatrix,~job,count,0,matrix,state),
    point_run_same(~job,count,matrix,state)) : PointKept<bias,cells,pairs,depth,_>
  point_jobs(~job,~keeps,count,0,matrix,bias,cells,pairs,depth,state,readers,bias_evidence,evidence)

# The operands of a run are the owners it read and its staged output.
def point_operands(-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-depth: Nat,-state: Gemm.PointOperands,
  evidence: PointKept<bias,cells,pairs,depth,state>) -> {Gemm.PointOperands{bias,cells,pairs,Gemm.point_staged(state)} == state : Gemm.PointOperands}:
  PointKept{staged,equation,balanced} = evidence
  %Equal.sym(Gemm.PointOperands,state,Gemm.PointOperands{bias,cells,pairs,staged},equation) :
    {Gemm.PointOperands{bias,cells,pairs,Gemm.point_staged(_)} == _ : Gemm.PointOperands}
  {==}

def point_balance(-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,+depth: Nat,-state: Gemm.PointOperands,
  evidence: PointKept<bias,cells,pairs,depth,state>) ->
  {Tree.pack(Tile.Tile16,Certificate.normalized(Tile.Tile16,depth,Tree.reflect(Tile.Tile16,Gemm.point_staged(state)))) == Gemm.point_staged(state) : Array<Tile.Tile16>}:
  PointKept{staged,equation,balanced} = evidence
  %Equal.sym(Gemm.PointOperands,state,Gemm.PointOperands{bias,cells,pairs,staged},equation) :
    {Tree.pack(Tile.Tile16,Certificate.normalized(Tile.Tile16,depth,Tree.reflect(Tile.Tile16,Gemm.point_staged(_)))) == Gemm.point_staged(_) : Array<Tile.Tile16>}
  Certificate.restored(Tile.Tile16,depth,staged,balanced)

type PointCertified<-state: Gemm.PointOperands> is Data:
  PointCertified{bias: Certificate.Certificate<Tile.Tile8,Gemm.point_bias(state)>,cells: Certificate.Certificate<Tile.Tile8,Gemm.point_cells(state)>,
    pairs: Certificate.Certificate<Tile.Tile16,Gemm.point_pairs(state)>,staged: Certificate.Certificate<Tile.Tile16,Gemm.point_staged(state)>}

# Only the certificates are live: the jobs are walked beneath erased rewrites.
def point_certificates_with(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,~keeps: PointJob(~job),+count: Nat,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-staged: Array<Tile.Tile16>,
  +bias_certificate: Certificate.Certificate<Tile.Tile8,bias>,+cell_certificate: Certificate.Certificate<Tile.Tile8,cells>,
  +pair_certificate: Certificate.Certificate<Tile.Tile16,pairs>,staged_certificate: Certificate.Certificate<Tile.Tile16,staged>) ->
  PointCertified<Gemm.point_run_indexed(~job,count,matrix,Gemm.PointOperands{bias,cells,pairs,staged})>:
  Certificate.Certificate{+depth,+balanced} = staged_certificate
  -initial = {Gemm.PointOperands{bias,cells,pairs,staged} : Gemm.PointOperands}
  -state = Gemm.point_run_indexed(~job,count,matrix,initial)
  PointCertified{
    %point_operands(bias,cells,pairs,depth,state,
      point_run_kept(~job,~keeps,count,matrix,bias,cells,pairs,depth,initial,
        PointReaders{Certificate.witness(Tile.Tile8,cells,cell_certificate),Certificate.witness(Tile.Tile16,pairs,pair_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        PointKept{staged,{==},Certificate.witness_at(Tile.Tile16,depth,staged,balanced)})) : Certificate.Certificate<Tile.Tile8,Gemm.point_bias(_)> bias_certificate,
    %point_operands(bias,cells,pairs,depth,state,
      point_run_kept(~job,~keeps,count,matrix,bias,cells,pairs,depth,initial,
        PointReaders{Certificate.witness(Tile.Tile8,cells,cell_certificate),Certificate.witness(Tile.Tile16,pairs,pair_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        PointKept{staged,{==},Certificate.witness_at(Tile.Tile16,depth,staged,balanced)})) : Certificate.Certificate<Tile.Tile8,Gemm.point_cells(_)> cell_certificate,
    %point_operands(bias,cells,pairs,depth,state,
      point_run_kept(~job,~keeps,count,matrix,bias,cells,pairs,depth,initial,
        PointReaders{Certificate.witness(Tile.Tile8,cells,cell_certificate),Certificate.witness(Tile.Tile16,pairs,pair_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        PointKept{staged,{==},Certificate.witness_at(Tile.Tile16,depth,staged,balanced)})) : Certificate.Certificate<Tile.Tile16,Gemm.point_pairs(_)> pair_certificate,
    Certificate.Certificate{depth,
      %point_balance(bias,cells,pairs,depth,state,
      point_run_kept(~job,~keeps,count,matrix,bias,cells,pairs,depth,initial,
        PointReaders{Certificate.witness(Tile.Tile8,cells,cell_certificate),Certificate.witness(Tile.Tile16,pairs,pair_certificate)},Certificate.witness(Tile.Tile8,bias,bias_certificate),
        PointKept{staged,{==},Certificate.witness_at(Tile.Tile16,depth,staged,balanced)})) :
        {_ == Tree.pack(Tile.Tile16,Certificate.normalized(Tile.Tile16,depth,Tree.reflect(Tile.Tile16,Gemm.point_staged(state)))) : Array<Tile.Tile16>} {==}}}

def point_certificates(+count: Nat,+matrix: Gemm.PointMatrix,-bias: Array<Tile.Tile8>,-cells: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,-staged: Array<Tile.Tile16>,
  +bias_certificate: Certificate.Certificate<Tile.Tile8,bias>,+cell_certificate: Certificate.Certificate<Tile.Tile8,cells>,
  +pair_certificate: Certificate.Certificate<Tile.Tile16,pairs>,staged_certificate: Certificate.Certificate<Tile.Tile16,staged>) ->
  PointCertified<Gemm.point_run(count,matrix,Gemm.PointOperands{bias,cells,pairs,staged})>:
  point_certificates_with(~Gemm.point_job,~point_job_whole,count,matrix,bias,cells,pairs,staged,bias_certificate,cell_certificate,pair_certificate,staged_certificate)

# ---- Pairs of lane tiles and the staged cells ----

def pair_upper(-tiles: Array<Tile.Tile8>,-lower: Tile.Tile8,+address: U32,read: Witness.Read<Tile.Tile8,tiles,Array.get(Tile.Tile8,tiles,address)>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.pair_upper(lower,Array.get(Tile.Tile8,tiles,address))>:
  Witness.Read{upper,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,tiles,address),(tiles,upper),equation) :
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.pair_upper(lower,_)>
  TraversalWitness.Sample{Tile.Tiles{lower,upper},{==}}

def pair_lower(-tiles: Array<Tile.Tile8>,+above: U32,+address: U32,+evidence: Witness.ArrayWitness<Tile.Tile8,tiles>,
  read: Witness.Read<Tile.Tile8,tiles,Array.get(Tile.Tile8,tiles,address)>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.pair_lower(above,Array.get(Tile.Tile8,tiles,address))>:
  Witness.Read{lower,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,tiles,address),(tiles,lower),equation) :
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.pair_lower(above,_)>
  pair_upper(tiles,lower,above,Witness.observed(Tile.Tile8,tiles,above,evidence))

def pair_read(-tiles: Array<Tile.Tile8>,plan: Gemm.PairPlan,index: U32,evidence: Witness.ArrayWitness<Tile.Tile8,tiles>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.pair_read(tiles,plan,index)>:
  Gemm.PairPlan{+groups,+inner} = plan
  +index = index
  +evidence = evidence
  +pair = (index / inner : U32)
  +k = (index % inner : U32)
  pair_lower(tiles,(U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)) * inner + k : U32),(pair * 2 * inner + k : U32),evidence,
    Witness.observed(Tile.Tile8,tiles,(pair * 2 * inner + k : U32),evidence))

def tile_sample(-tiles: Array<Tile.Tile8>,+address: U32,read: Witness.Read<Tile.Tile8,tiles,Array.get(Tile.Tile8,tiles,address)>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,tiles,Array.get(Tile.Tile8,tiles,address)>:
  Witness.Read{tile,equation} = read
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,tiles,address),(tiles,tile),equation) :
    TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,tiles,_>
  TraversalWitness.Sample{tile,{==}}

def tile_read(-tiles: Array<Tile.Tile8>,plan: Gemm.TilePlan,index: U32,evidence: Witness.ArrayWitness<Tile.Tile8,tiles>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,tiles,Gemm.tile_read(tiles,plan,index)>:
  Gemm.TilePlan{+source,target} = plan
  +address = (source + index : U32)
  tile_sample(tiles,address,Witness.observed(Tile.Tile8,tiles,address,evidence))

def panel_pair_read(-tiles: Array<Tile.Tile8>,plan: Gemm.PanelPlan,index: U32,evidence: Witness.ArrayWitness<Tile.Tile8,tiles>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile16,tiles,Gemm.panel_pair_read(tiles,plan,index)>:
  Gemm.PanelPlan{+groups,+inner,+base} = plan
  +index = index
  +evidence = evidence
  +lower = (2 * (index / inner) : U32)
  +row = (base + (index % inner) * groups : U32)
  +address = (row + lower : U32)
  pair_lower(tiles,(row + U32.min((lower + 1 : U32),(groups - 1 : U32)) : U32),address,evidence,Witness.observed(Tile.Tile8,tiles,address,evidence))

def staged_mapped(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,-staged: Array<Tile.Tile16>,context: Context,upper: Bool,+address: U32,
  read: Witness.Read<Tile.Tile16,staged,Array.get(Tile.Tile16,staged,address)>) ->
  TraversalWitness.Sample<Array<Tile.Tile16>,Tile.Tile8,staged,Gemm.staged_mapped(~Context,~map,context,upper,Array.get(Tile.Tile16,staged,address))>:
  Witness.Read{value,equation} = read
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,staged,address),(staged,value),equation) :
    TraversalWitness.Sample<Array<Tile.Tile16>,Tile.Tile8,staged,Gemm.staged_mapped(~Context,~map,context,upper,_)>
  TraversalWitness.Sample{map(context,Gemm.staged_half(value,upper)),{==}}

def staged_read(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,-staged: Array<Tile.Tile16>,plan: Gemm.StagedPlan<Context>,index: U32,
  evidence: Witness.ArrayWitness<Tile.Tile16,staged>) ->
  TraversalWitness.Sample<Array<Tile.Tile16>,Tile.Tile8,staged,Gemm.staged_read(~Context,~map,staged,plan,index)>:
  Gemm.StagedPlan{+groups,+pairs,context} = plan
  +index = index
  +group = (index % groups : U32)
  staged_mapped(~Context,~map,staged,context,U32.is_eq((group % 2 : U32),1),((index / groups) * pairs + group / 2 : U32),
    Witness.observed(Tile.Tile16,staged,((index / groups) * pairs + group / 2 : U32),evidence))

def staged_slice_read(-staged: Array<Tile.Tile16>,slice: Gemm.StagedSlice,index: U32,evidence: Witness.ArrayWitness<Tile.Tile16,staged>) ->
  TraversalWitness.Sample<Array<Tile.Tile16>,Tile.Tile8,staged,Gemm.staged_slice_read(staged,slice,index)>:
  Gemm.StagedSlice{+groups,+positions,+first,+group_major} = slice
  staged_read(~Unit,~Gemm.unmapped_tile,staged,Gemm.StagedPlan{groups,((groups + 1) / 2 : U32),Unit{}},Gemm.staged_slice_index(Gemm.StagedSlice{groups,positions,first,group_major},index),evidence)
