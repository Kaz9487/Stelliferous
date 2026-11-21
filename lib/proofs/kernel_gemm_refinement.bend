# The native GEMM over packed trees is a model whose output is a program of
# write lists: each block appends the stores of its cells. Values are read
# from the model accumulators (kernel_gemm_spec.bend); addresses are separate.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../math_choice.bend as Choice
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./kernel_tile_refinement.bend as TileLaws
import ./write_program.bend as Writes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./nat_order.bend as NaturalOrder

type Operands<-E: Data> is Data:
  Operands{row_tiles: Tree.Tree<Tile.Tile8>,lane_tiles: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Writes.Program<E>}

def represent(~E: Data,operands: Operands<E>) -> Gemm.Operands<E,Array<F32>>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  Gemm.Operands{Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),Tree.pack(F32,bias),Tree.pack(E,Writes.evaluate(E,output))}

def operand_output(~E: Data,operands: Operands<E>) -> Writes.Program<E>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  output

def output_tree(~E: Data,operands: Operands<E>) -> Tree.Tree<E>:
  Writes.evaluate(E,operand_output(~E,operands))

# The product order of a contraction: the lane value first or the row value
# first, as its specification multiplies them.
type Order is Data:
  LaneFirst{}
  RowFirst{}

def ordered_times(+order: Order,lane: F32,row: F32) -> F32:
  match order:
    case LaneFirst{}: Gemm.lane_times(lane,row)
    case RowFirst{}: Gemm.row_times(lane,row)

def ordered_update(+order: Order,accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,row_tile: Tile.Tile8) -> Gemm.Accumulators:
  match order:
    case LaneFirst{}: Gemm.update(~Gemm.lane_times,accumulators,lane_tile,row_tile)
    case RowFirst{}: Gemm.update(~Gemm.row_times,accumulators,lane_tile,row_tile)

# A native product multiplies in the order.
def Ordered(~times: F32 -> F32 -> F32,+order: Order) -> Type:
  @+accumulators: Gemm.Accumulators -> @+lane_tile: Tile.Tile8 -> @+row_tile: Tile.Tile8 ->
    {Gemm.update(~times,accumulators,lane_tile,row_tile) == ordered_update(order,accumulators,lane_tile,row_tile) : Gemm.Accumulators}

def lane_first(+accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,+row_tile: Tile.Tile8) ->
  {Gemm.update(~Gemm.lane_times,accumulators,lane_tile,row_tile) == ordered_update(LaneFirst{},accumulators,lane_tile,row_tile) : Gemm.Accumulators}:
  {==}

def row_first(+accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,+row_tile: Tile.Tile8) ->
  {Gemm.update(~Gemm.row_times,accumulators,lane_tile,row_tile) == ordered_update(RowFirst{},accumulators,lane_tile,row_tile) : Gemm.Accumulators}:
  {==}

# The k loop of a block: the reader's cursor and the accumulators. Each step
# reads the lane tile and the row tile the cursor points to.
type Reduced<-Cursor: Data> is Data:
  Reduced{cursor: Cursor,accumulators: Gemm.Accumulators}

def reduced_accumulators(~Cursor: Data,state: Reduced<Cursor>) -> Gemm.Accumulators:
  Reduced{cursor,accumulators} = state
  accumulators

def reduced_with(~Cursor: Data,~order: Order,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,+accumulators: Gemm.Accumulators,
  read: Gemm.Step<Cursor>) -> Reduced<Cursor>:
  Gemm.Step{lane_address,row_address,cursor} = read
  Reduced{cursor,ordered_update(order,accumulators,Tree.read(Tile.Tile8,lane_tiles,lane_address),Tree.read(Tile.Tile8,row_tiles,row_address))}

def reduced_step(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: Order,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  state: Reduced<Cursor>) -> Reduced<Cursor>:
  Reduced{+cursor,+accumulators} = state
  reduced_with(~Cursor,~order,row_tiles,lane_tiles,accumulators,advance(cursor))

def reduce(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: Order,count: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  state: Reduced<Cursor>) -> Reduced<Cursor>:
  match count:
    case 0n: state
    case 1n+rest: reduce(~Cursor,~advance,~order,rest,row_tiles,lane_tiles,reduced_step(~Cursor,~advance,~order,row_tiles,lane_tiles,state))

def operand_row_tiles(~E: Data,operands: Operands<E>) -> Tree.Tree<Tile.Tile8>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  row_tiles

def operand_lane_tiles(~E: Data,operands: Operands<E>) -> Tree.Tree<Tile.Tile8>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  lane_tiles

def represent_owned(~E: Data,~Bias: Type,bias: Bias,operands: Operands<E>) -> Gemm.Operands<E,Bias>:
  Operands{row_tiles,lane_tiles,model_bias,output} = operands
  Gemm.Operands{Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),bias,Tree.pack(E,Writes.evaluate(E,output))}

def represent_loop(~Cursor: Data,~E: Data,~Bias: Type,bias: Bias,+operands: Operands<E>,+row: U32,+store: Gemm.Store,state: Reduced<Cursor>) -> Gemm.Loop<Cursor,E,Bias>:
  Operands{row_tiles,lane_tiles,model_bias,output} = operands
  Reduced{cursor,accumulators} = state
  Gemm.Loop{Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),bias,Tree.pack(E,Writes.evaluate(E,output)),cursor,row,store,accumulators}

def cursor_matches(~Cursor: Data,~E: Data,~Bias: Type,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  -bias: Bias,+operands: Operands<E>,+row: U32,+store: Gemm.Store,+accumulators: Gemm.Accumulators,read: Gemm.Step<Cursor>) ->
  {Gemm.with_cursor(~Cursor,~E,~Bias,~times,Tree.pack(Tile.Tile8,operand_row_tiles(~E,operands)),Tree.pack(Tile.Tile8,operand_lane_tiles(~E,operands)),bias,Tree.pack(E,output_tree(~E,operands)),row,store,accumulators,read) ==
    represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,reduced_with(~Cursor,~order,operand_row_tiles(~E,operands),operand_lane_tiles(~E,operands),accumulators,read)) : Gemm.Loop<Cursor,E,Bias>}:
  Operands{+row_tiles,+lane_tiles,model_bias,+output} = operands
  Gemm.Step{+lane_address,+row_address,+cursor} = read
  -row_array = Tree.pack(Tile.Tile8,row_tiles)
  -lane_array = Tree.pack(Tile.Tile8,lane_tiles)
  -output_array = Tree.pack(E,Writes.evaluate(E,output))
  -result = represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,reduced_with(~Cursor,~order,row_tiles,lane_tiles,accumulators,Gemm.Step{lane_address,row_address,cursor}))
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,lane_array,lane_address),(lane_array,Tree.read(Tile.Tile8,lane_tiles,lane_address)),StorageLaws.read(Tile.Tile8,lane_tiles,lane_address)) :
    {Gemm.with_lane_tile(~Cursor,~E,~Bias,~times,row_array,bias,output_array,row_address,cursor,row,store,accumulators,_) == result : Gemm.Loop<Cursor,E,Bias>}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,row_array,row_address),(row_array,Tree.read(Tile.Tile8,row_tiles,row_address)),StorageLaws.read(Tile.Tile8,row_tiles,row_address)) :
    {Gemm.with_row_tile(~Cursor,~E,~Bias,~times,lane_array,bias,output_array,cursor,row,store,accumulators,Tree.read(Tile.Tile8,lane_tiles,lane_address),_) == result : Gemm.Loop<Cursor,E,Bias>}
  Equal.cong(Gemm.Accumulators,Gemm.Loop<Cursor,E,Bias>,next => Gemm.Loop{row_array,lane_array,bias,output_array,cursor,row,store,next},
    Gemm.update(~times,accumulators,Tree.read(Tile.Tile8,lane_tiles,lane_address),Tree.read(Tile.Tile8,row_tiles,row_address)),
    ordered_update(order,accumulators,Tree.read(Tile.Tile8,lane_tiles,lane_address),Tree.read(Tile.Tile8,row_tiles,row_address)),
    ordered(accumulators,Tree.read(Tile.Tile8,lane_tiles,lane_address),Tree.read(Tile.Tile8,row_tiles,row_address)))

def step_matches(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  -bias: Bias,+operands: Operands<E>,+row: U32,+store: Gemm.Store,+state: Reduced<Cursor>) ->
  {Gemm.step(~Cursor,~E,~Bias,~advance,~times,represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,state)) ==
    represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,reduced_step(~Cursor,~advance,~order,operand_row_tiles(~E,operands),operand_lane_tiles(~E,operands),state)) : Gemm.Loop<Cursor,E,Bias>}:
  Operands{+row_tiles,+lane_tiles,model_bias,+output} = operands
  Reduced{+cursor,+accumulators} = state
  cursor_matches(~Cursor,~E,~Bias,~times,~order,~ordered,bias,operands,row,store,accumulators,advance(cursor))

def loop_matches(~Cursor: Data,~E: Data,~Bias: Type,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  count: Nat,-bias: Bias,+operands: Operands<E>,+row: U32,+store: Gemm.Store,+state: Reduced<Cursor>) ->
  {Loop.run(~Gemm.Loop<Cursor,E,Bias>,~Gemm.step(~Cursor,~E,~Bias,~advance,~times),count,represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,state)) ==
    represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,reduce(~Cursor,~advance,~order,count,operand_row_tiles(~E,operands),operand_lane_tiles(~E,operands),state)) : Gemm.Loop<Cursor,E,Bias>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +next = reduced_step(~Cursor,~advance,~order,operand_row_tiles(~E,operands),operand_lane_tiles(~E,operands),state)
      %Equal.sym(Gemm.Loop<Cursor,E,Bias>,Gemm.step(~Cursor,~E,~Bias,~advance,~times,represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,state)),
        represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,next),step_matches(~Cursor,~E,~Bias,~advance,~times,~order,~ordered,bias,operands,row,store,state)) :
        {Loop.run(~Gemm.Loop<Cursor,E,Bias>,~Gemm.step(~Cursor,~E,~Bias,~advance,~times),rest,_) ==
          represent_loop(~Cursor,~E,~Bias,bias,operands,row,store,reduce(~Cursor,~advance,~order,1n+rest,operand_row_tiles(~E,operands),operand_lane_tiles(~E,operands),state)) : Gemm.Loop<Cursor,E,Bias>}
      loop_matches(~Cursor,~E,~Bias,~advance,~times,~order,~ordered,rest,bias,operands,row,store,next)

# ---- The writers ----

# The two writers of a block. The lane writer stores each lane of a row as one
# cell; the tile writer stores each row as one tile whose lanes at or above
# the store's lanes are +0.
type Writer is Data:
  Lanes{}
  Tiles{}

def Element(writer: Writer) -> Data:
  match writer:
    case Lanes{}: F32
    case Tiles{}: Tile.Tile8

# The cells of a row.
def columns(+writer: Writer,+store: Gemm.Store) -> U32:
  match writer:
    case Lanes{}:
      Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
      lanes
    case Tiles{}: 1

# A word plus offset, written as the stores write the first offsets.
def shifted(+base: U32,offset: Nat) -> U32:
  match offset:
    case 0n: base
    case 1n+rest: (base + U32.from_nat(1n+rest) : U32)

def shifted_exact(+base: U32,+offset: Nat) -> {shifted(base,offset) == (base + U32.from_nat(offset) : U32) : U32}:
  match offset:
    case 0n: Equal.sym(U32,(base + 0 : U32),base,Arithmetic.add_zero(base))
    case 1n+rest: {==}

# A finished tile: lane l below lanes takes bias first + l, the others +0.
def finished_tile(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+first: U32,+lanes: U32,+bias: Tree.Tree<F32>,+values: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{Choice.chosen(~F32,U32.is_lt(0,lanes),close(Tile.lane0(values),value(bias,first)),0.0),
    Choice.chosen(~F32,U32.is_lt(1,lanes),close(Tile.lane1(values),value(bias,(first + 1 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(2,lanes),close(Tile.lane2(values),value(bias,(first + 2 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(3,lanes),close(Tile.lane3(values),value(bias,(first + 3 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(4,lanes),close(Tile.lane4(values),value(bias,(first + 4 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(5,lanes),close(Tile.lane5(values),value(bias,(first + 5 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(6,lanes),close(Tile.lane6(values),value(bias,(first + 6 : U32))),0.0),
    Choice.chosen(~F32,U32.is_lt(7,lanes),close(Tile.lane7(values),value(bias,(first + 7 : U32))),0.0)}

# Lane l of a finished tile below its lanes closes lane l of the values with
# the bias of first + l.
def finished_lane(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+first: U32,+lanes: U32,+bias: Tree.Tree<F32>,+values: Tile.Tile8,+lane: Nat,
  +inside: {U32.is_lt(U32.from_nat(lane),lanes) == True{} : Bool},+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,finished_tile(~value,~close,first,lanes,bias,values)) == close(TileLaws.select_lane(lane,values),value(bias,shifted(first,lane))) : F32}:
  match lane:
    case 0n:
      %Equal.sym(Bool,U32.is_lt(0,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane0(values),value(bias,first)),0.0) == close(Tile.lane0(values),value(bias,first)) : F32}
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          %Equal.sym(Bool,U32.is_lt(1,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane1(values),value(bias,(first + 1 : U32))),0.0) == close(Tile.lane1(values),value(bias,(first + 1 : U32))) : F32}
          {==}
        case 1n+ +b:
          match b:
            case 0n:
              %Equal.sym(Bool,U32.is_lt(2,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane2(values),value(bias,(first + 2 : U32))),0.0) == close(Tile.lane2(values),value(bias,(first + 2 : U32))) : F32}
              {==}
            case 1n+ +c:
              match c:
                case 0n:
                  %Equal.sym(Bool,U32.is_lt(3,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane3(values),value(bias,(first + 3 : U32))),0.0) == close(Tile.lane3(values),value(bias,(first + 3 : U32))) : F32}
                  {==}
                case 1n+ +d:
                  match d:
                    case 0n:
                      %Equal.sym(Bool,U32.is_lt(4,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane4(values),value(bias,(first + 4 : U32))),0.0) == close(Tile.lane4(values),value(bias,(first + 4 : U32))) : F32}
                      {==}
                    case 1n+ +e:
                      match e:
                        case 0n:
                          %Equal.sym(Bool,U32.is_lt(5,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane5(values),value(bias,(first + 5 : U32))),0.0) == close(Tile.lane5(values),value(bias,(first + 5 : U32))) : F32}
                          {==}
                        case 1n+ +f:
                          match f:
                            case 0n:
                              %Equal.sym(Bool,U32.is_lt(6,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane6(values),value(bias,(first + 6 : U32))),0.0) == close(Tile.lane6(values),value(bias,(first + 6 : U32))) : F32}
                              {==}
                            case 1n+ +g:
                              match g:
                                case 0n:
                                  %Equal.sym(Bool,U32.is_lt(7,lanes),True{},inside) : {Choice.chosen(~F32,_,close(Tile.lane7(values),value(bias,(first + 7 : U32))),0.0) == close(Tile.lane7(values),value(bias,(first + 7 : U32))) : F32}
                                  {==}
                                case 1n+ +h:
                                  Empty.absurd({TileLaws.select_lane(8n+h,finished_tile(~value,~close,first,lanes,bias,values)) == close(TileLaws.select_lane(8n+h,values),value(bias,shifted(first,8n+h))) : F32},
                                    WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

def picked_close(~close: F32 -> F32 -> F32,valid: Bool,+left: F32,+right: F32,+bias: F32,equal: @valid_true: {valid == True{} : Bool} -> {left == right : F32}) ->
  {Choice.chosen(~F32,valid,close(left,bias),0.0) == Choice.chosen(~F32,valid,close(right,bias),0.0) : F32}:
  match valid:
    case True{}: Equal.cong(F32,F32,value => close(value,bias),left,right,equal({==}))
    case False{}: {==}

# Two finished tiles agree when their values agree on every lane below lanes.
def finished_agree(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+first: U32,+lanes: U32,+bias: Tree.Tree<F32>,+left: Tile.Tile8,+right: Tile.Tile8,
  e0: @inside: {U32.is_lt(0,lanes) == True{} : Bool} -> {Tile.lane0(left) == Tile.lane0(right) : F32},e1: @inside: {U32.is_lt(1,lanes) == True{} : Bool} -> {Tile.lane1(left) == Tile.lane1(right) : F32},e2: @inside: {U32.is_lt(2,lanes) == True{} : Bool} -> {Tile.lane2(left) == Tile.lane2(right) : F32},e3: @inside: {U32.is_lt(3,lanes) == True{} : Bool} -> {Tile.lane3(left) == Tile.lane3(right) : F32},e4: @inside: {U32.is_lt(4,lanes) == True{} : Bool} -> {Tile.lane4(left) == Tile.lane4(right) : F32},e5: @inside: {U32.is_lt(5,lanes) == True{} : Bool} -> {Tile.lane5(left) == Tile.lane5(right) : F32},e6: @inside: {U32.is_lt(6,lanes) == True{} : Bool} -> {Tile.lane6(left) == Tile.lane6(right) : F32},e7: @inside: {U32.is_lt(7,lanes) == True{} : Bool} -> {Tile.lane7(left) == Tile.lane7(right) : F32}) ->
  {finished_tile(~value,~close,first,lanes,bias,left) == finished_tile(~value,~close,first,lanes,bias,right) : Tile.Tile8}:
  TileLaws.tiles_equal(finished_tile(~value,~close,first,lanes,bias,left),finished_tile(~value,~close,first,lanes,bias,right),
    picked_close(~close,U32.is_lt(0,lanes),Tile.lane0(left),Tile.lane0(right),value(bias,first),e0),
    picked_close(~close,U32.is_lt(1,lanes),Tile.lane1(left),Tile.lane1(right),value(bias,(first + 1 : U32)),e1),
    picked_close(~close,U32.is_lt(2,lanes),Tile.lane2(left),Tile.lane2(right),value(bias,(first + 2 : U32)),e2),
    picked_close(~close,U32.is_lt(3,lanes),Tile.lane3(left),Tile.lane3(right),value(bias,(first + 3 : U32)),e3),
    picked_close(~close,U32.is_lt(4,lanes),Tile.lane4(left),Tile.lane4(right),value(bias,(first + 4 : U32)),e4),
    picked_close(~close,U32.is_lt(5,lanes),Tile.lane5(left),Tile.lane5(right),value(bias,(first + 5 : U32)),e5),
    picked_close(~close,U32.is_lt(6,lanes),Tile.lane6(left),Tile.lane6(right),value(bias,(first + 6 : U32)),e6),
    picked_close(~close,U32.is_lt(7,lanes),Tile.lane7(left),Tile.lane7(right),value(bias,(first + 7 : U32)),e7))

# Cell (row, column) of a block from the lanes of its row: lane l of a row
# takes bias first + row (lane writer) or first + l (tile writer).
def finished(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: Writer,+first: U32,+store: Gemm.Store,+bias: Tree.Tree<F32>,
  +row: Nat,+column: Nat,+values: Tile.Tile8) -> Element(writer):
  match writer:
    case Lanes{}: close(TileLaws.select_lane(column,values),value(bias,shifted(first,row)))
    case Tiles{}:
      Gemm.Store{base,row_stride,lane_stride,rows,+lanes} = store
      finished_tile(~value,~close,first,lanes,bias,values)

# A cell outside the store writes nothing.
def put(~E: Data,inside: Bool,+offset: U32,+value: E) -> List<&2,Writes.Assignment<E>>:
  match inside:
    case False{}: []
    case True{}: [Writes.Put{offset,value}]

# The lanes of a row below count, from base step apart.
def lane_writes(~close: F32 -> F32 -> F32,+valid: Bool,+base: U32,+step: U32,+count: U32,+bias: F32,+values: Tile.Tile8) -> List<&2,Writes.Assignment<F32>>:
  w = put(~F32,Bool.and(valid,U32.is_lt(0,count)),base,close(Tile.lane0(values),bias))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(1,count)),(base + step : U32),close(Tile.lane1(values),bias)))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(2,count)),(base + 2 * step : U32),close(Tile.lane2(values),bias)))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(3,count)),(base + 3 * step : U32),close(Tile.lane3(values),bias)))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(4,count)),(base + 4 * step : U32),close(Tile.lane4(values),bias)))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(5,count)),(base + 5 * step : U32),close(Tile.lane5(values),bias)))
  w = List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(6,count)),(base + 6 * step : U32),close(Tile.lane6(values),bias)))
  List.append(&2,Writes.Assignment<F32>,w,put(~F32,Bool.and(valid,U32.is_lt(7,count)),(base + 7 * step : U32),close(Tile.lane7(values),bias)))

# The stores of a row when valid: its lanes, or its tile at base.
def row_writes(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: Writer,+first: U32,+store: Gemm.Store,+bias: Tree.Tree<F32>,
  +row: Nat,+valid: Bool,+base: U32,+values: Tile.Tile8) -> List<&2,Writes.Assignment<Element(writer)>>:
  match writer:
    case Lanes{}:
      Gemm.Store{origin,row_stride,+step,rows,+count} = store
      lane_writes(~close,valid,base,step,count,value(bias,shifted(first,row)),values)
    case Tiles{}: put(~Tile.Tile8,valid,base,finished(~value,~close,Tiles{},first,store,bias,row,0n,values))

# A row that is not stored writes nothing.
def row_unwritten(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: Writer,+first: U32,+store: Gemm.Store,+bias: Tree.Tree<F32>,
  +row: Nat,+base: U32,+values: Tile.Tile8) -> {row_writes(~value,~close,writer,first,store,bias,row,False{},base,values) == Nil{} : List<&2,Writes.Assignment<Element(writer)>>}:
  match writer:
    case Lanes{}:
      Gemm.Store{+origin,+row_stride,+step,+rows,+count} = store
      {==}
    case Tiles{}: {==}

# The stores of a block as one write list: every row below store.rows, in
# order.
def block_writes(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: Writer,+bias_start: U32,+store: Gemm.Store,+accumulators: Gemm.Accumulators,
  +bias: Tree.Tree<F32>) -> List<&2,Writes.Assignment<Element(writer)>>:
  Gemm.Store{+base,+row_stride,lane_stride,+rows,lanes} = store
  w = row_writes(~value,~close,writer,bias_start,store,bias,0n,U32.is_lt(0,rows),base,Gemm.row0(accumulators))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,1n,U32.is_lt(1,rows),(base + row_stride : U32),Gemm.row1(accumulators)))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,2n,U32.is_lt(2,rows),(base + 2 * row_stride : U32),Gemm.row2(accumulators)))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,3n,U32.is_lt(3,rows),(base + 3 * row_stride : U32),Gemm.row3(accumulators)))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,4n,U32.is_lt(4,rows),(base + 4 * row_stride : U32),Gemm.row4(accumulators)))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,5n,U32.is_lt(5,rows),(base + 5 * row_stride : U32),Gemm.row5(accumulators)))
  w = List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,6n,U32.is_lt(6,rows),(base + 6 * row_stride : U32),Gemm.row6(accumulators)))
  List.append(&2,Writes.Assignment<Element(writer)>,w,row_writes(~value,~close,writer,bias_start,store,bias,7n,U32.is_lt(7,rows),(base + 7 * row_stride : U32),Gemm.row7(accumulators)))

def stored_written(~E: Data,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<E>>,output: Array<E>) -> Gemm.Written<E,Array<F32>>:
  Gemm.Written{Tree.pack(F32,bias),Writes.native(E,writes,output)}

# A native writer applies the block writes of its model writer.
def Stored(~writer: Writer,~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32) -> Type:
  @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> @+bias: Tree.Tree<F32> -> @+tree: Tree.Tree<Element(writer)> ->
    {write(bias_start,store,accumulators,Gemm.Written{Tree.pack(F32,bias),Tree.pack(Element(writer),tree)}) ==
      stored_written(~Element(writer),bias,block_writes(~value,~close,writer,bias_start,store,accumulators,bias),Tree.pack(Element(writer),tree)) : Gemm.Written<Element(writer),Array<F32>>}

# ---- Puts ----

def put_kept(~E: Data,+inside: Bool,+tree: Tree.Tree<E>,+offset: U32,+value: E) ->
  {Gemm.kept_or(~E,inside,offset,value,(Tree.pack(E,tree),Tree.read(E,tree,offset))) == Writes.native(E,put(~E,inside,offset,value),Tree.pack(E,tree)) : Array<E>}:
  match inside:
    case True{}: {==}
    case False{}: StorageLaws.set_read(E,tree,offset)

def put_packed(~E: Data,+inside: Bool,+tree: Tree.Tree<E>,+offset: U32,+value: E) ->
  {Gemm.put(~E,inside,Tree.pack(E,tree),offset,value) == Writes.native(E,put(~E,inside,offset,value),Tree.pack(E,tree)) : Array<E>}:
  %Equal.sym(Array<E> & E,Array.get(E,Tree.pack(E,tree),offset),(Tree.pack(E,tree),Tree.read(E,tree,offset)),StorageLaws.read(E,tree,offset)) :
    {Gemm.kept_or(~E,inside,offset,value,_) == Writes.native(E,put(~E,inside,offset,value),Tree.pack(E,tree)) : Array<E>}
  put_kept(~E,inside,tree,offset,value)

# A put after earlier writes appends its own.
def put_appended(~E: Data,+inside: Bool,+tree: Tree.Tree<E>,+previous: List<&2,Writes.Assignment<E>>,+offset: U32,+value: E) ->
  {Gemm.put(~E,inside,Writes.native(E,previous,Tree.pack(E,tree)),offset,value) ==
    Writes.native(E,List.append(&2,Writes.Assignment<E>,previous,put(~E,inside,offset,value)),Tree.pack(E,tree)) : Array<E>}:
  +own = put(~E,inside,offset,value)
  +written = Writes.pure(E,previous,tree)
  %Equal.sym(Array<E>,Writes.native(E,List.append(&2,Writes.Assignment<E>,previous,own),Tree.pack(E,tree)),
    Writes.native(E,own,Writes.native(E,previous,Tree.pack(E,tree))),Writes.concatenated_native(E,previous,own,Tree.pack(E,tree))) :
    {Gemm.put(~E,inside,Writes.native(E,previous,Tree.pack(E,tree)),offset,value) == _ : Array<E>}
  %Equal.sym(Array<E>,Writes.native(E,previous,Tree.pack(E,tree)),Tree.pack(E,written),Writes.refinement(E,previous,tree)) :
    {Gemm.put(~E,inside,_,offset,value) == Writes.native(E,own,_) : Array<E>}
  put_packed(~E,inside,written,offset,value)

def put_next(~E: Data,+inside: Bool,-native: Array<E>,+tree: Tree.Tree<E>,+previous: List<&2,Writes.Assignment<E>>,+offset: U32,+value: E,
  equation: {native == Writes.native(E,previous,Tree.pack(E,tree)) : Array<E>}) ->
  {Gemm.put(~E,inside,native,offset,value) ==
    Writes.native(E,List.append(&2,Writes.Assignment<E>,previous,put(~E,inside,offset,value)),Tree.pack(E,tree)) : Array<E>}:
  %Equal.sym(Array<E>,native,Writes.native(E,previous,Tree.pack(E,tree)),equation) :
    {Gemm.put(~E,inside,_,offset,value) == Writes.native(E,List.append(&2,Writes.Assignment<E>,previous,put(~E,inside,offset,value)),Tree.pack(E,tree)) : Array<E>}
  put_appended(~E,inside,tree,previous,offset,value)

# ---- The lane writer ----

# A row's bias read gives the bias owner back with a model value.
def Fetched(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~value: Tree.Tree<F32> -> U32 -> F32) -> Type:
  @+bias: Tree.Tree<F32> -> @+row: U32 -> {fetch(Tree.pack(F32,bias),row) == (Tree.pack(F32,bias),value(bias,row)) : Array<F32> & F32}

def biased_value(bias: Tree.Tree<F32>,row: U32) -> F32:
  Tree.read(F32,bias,row)

def unbiased_value(bias: Tree.Tree<F32>,row: U32) -> F32:
  0.0

def unbiased_fetched(+bias: Tree.Tree<F32>,+row: U32) -> {Gemm.unbiased(Tree.pack(F32,bias),row) == (Tree.pack(F32,bias),unbiased_value(bias,row)) : Array<F32> & F32}:
  {==}

# ---- The tile writer ----

def pack_bias(bias: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,bias)

def bias_sample(~value: Tree.Tree<F32> -> U32 -> F32,bias: Tree.Tree<F32>,start: U32,lane: Nat) -> F32:
  value(bias,(start + U32.from_nat(lane) : U32))

def bias_values(~value: Tree.Tree<F32> -> U32 -> F32,+bias: Tree.Tree<F32>,+start: U32) -> Tile.Tile8:
  Tile.Tile{value(bias,start),value(bias,(start + 1 : U32)),value(bias,(start + 2 : U32)),value(bias,(start + 3 : U32)),value(bias,(start + 4 : U32)),value(bias,(start + 5 : U32)),value(bias,(start + 6 : U32)),value(bias,(start + 7 : U32))}

def bias_lane_fetched(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~value: Tree.Tree<F32> -> U32 -> F32,~fetched: Fetched(~fetch,~value),
  +bias: Tree.Tree<F32>,+start: U32,+lane: Nat) ->
  {Gemm.bias_lane(~fetch,pack_bias(bias),start,lane) == (pack_bias(bias),bias_sample(~value,bias,start,lane)) : Array<F32> & F32}:
  fetched(bias,(start + U32.from_nat(lane) : U32))

# The biases of a block are read with the model values of their lanes.
def biases_fetched(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~value: Tree.Tree<F32> -> U32 -> F32,~fetched: Fetched(~fetch,~value),
  +bias: Tree.Tree<F32>,+start: U32) ->
  {Gemm.bias_lanes(~fetch,Tree.pack(F32,bias),start) == (Tree.pack(F32,bias),bias_values(~value,bias,start)) : Array<F32> & Tile.Tile8}:
  %Equal.sym(Array<F32> & F32,fetch(Tree.pack(F32,bias),start),(Tree.pack(F32,bias),value(bias,start)),fetched(bias,start)) :
    {Tile.read_tail(~Array<F32>,~U32,~Gemm.bias_lane(~fetch),_,start) == (Tree.pack(F32,bias),bias_values(~value,bias,start)) : Array<F32> & Tile.Tile8}
  TileLaws.tail_block_refines(~Array<F32>,~Tree.Tree<F32>,~U32,~pack_bias,~Gemm.bias_lane(~fetch),~bias_sample(~value),
    ~bias_lane_fetched(~fetch,~value,~fetched),bias,start,value(bias,start))

# The rows of the tile writer with the biases of its lanes.
def tile_rows(~close: F32 -> F32 -> F32,+base: U32,+row_stride: U32,+rows: U32,+count: U32,+biases: Tile.Tile8,+accumulators: Gemm.Accumulators) ->
  List<&2,Writes.Assignment<Tile.Tile8>>:
  w = put(~Tile.Tile8,U32.is_lt(0,rows),base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(1,rows),(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators))))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(2,rows),(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators))))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(3,rows),(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators))))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(4,rows),(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators))))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(5,rows),(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators))))
  w = List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(6,rows),(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators))))
  List.append(&2,Writes.Assignment<Tile.Tile8>,w,put(~Tile.Tile8,U32.is_lt(7,rows),(base + 7 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row7(accumulators))))

def tiles_native(~Bias: Type,~close: F32 -> F32 -> F32,-bias: Bias,+tree: Tree.Tree<Tile.Tile8>,+base: U32,+row_stride: U32,+rows: U32,+count: U32,
  +biases: Tile.Tile8,+accumulators: Gemm.Accumulators) ->
  {Gemm.tiles_finished(~Bias,~close,base,row_stride,rows,count,accumulators,Tree.pack(Tile.Tile8,tree),(bias,biases)) ==
    Gemm.Written{bias,Writes.native(Tile.Tile8,tile_rows(~close,base,row_stride,rows,count,biases,accumulators),Tree.pack(Tile.Tile8,tree))} : Gemm.Written<Tile.Tile8,Bias>}:
  +p0 = put(~Tile.Tile8,U32.is_lt(0,rows),base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)))
  +p1 = put(~Tile.Tile8,U32.is_lt(1,rows),(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators)))
  +p2 = put(~Tile.Tile8,U32.is_lt(2,rows),(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators)))
  +p3 = put(~Tile.Tile8,U32.is_lt(3,rows),(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators)))
  +p4 = put(~Tile.Tile8,U32.is_lt(4,rows),(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators)))
  +p5 = put(~Tile.Tile8,U32.is_lt(5,rows),(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators)))
  +p6 = put(~Tile.Tile8,U32.is_lt(6,rows),(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators)))
  +w1 = List.append(&2,Writes.Assignment<Tile.Tile8>,p0,p1)
  +w2 = List.append(&2,Writes.Assignment<Tile.Tile8>,w1,p2)
  +w3 = List.append(&2,Writes.Assignment<Tile.Tile8>,w2,p3)
  +w4 = List.append(&2,Writes.Assignment<Tile.Tile8>,w3,p4)
  +w5 = List.append(&2,Writes.Assignment<Tile.Tile8>,w4,p5)
  +w6 = List.append(&2,Writes.Assignment<Tile.Tile8>,w5,p6)
  -o0 = Gemm.put(~Tile.Tile8,U32.is_lt(0,rows),Tree.pack(Tile.Tile8,tree),base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)))
  -o1 = Gemm.put(~Tile.Tile8,U32.is_lt(1,rows),o0,(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators)))
  -o2 = Gemm.put(~Tile.Tile8,U32.is_lt(2,rows),o1,(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators)))
  -o3 = Gemm.put(~Tile.Tile8,U32.is_lt(3,rows),o2,(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators)))
  -o4 = Gemm.put(~Tile.Tile8,U32.is_lt(4,rows),o3,(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators)))
  -o5 = Gemm.put(~Tile.Tile8,U32.is_lt(5,rows),o4,(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators)))
  -o6 = Gemm.put(~Tile.Tile8,U32.is_lt(6,rows),o5,(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators)))
  +e0 = put_packed(~Tile.Tile8,U32.is_lt(0,rows),tree,base,Gemm.tile_finished(~close,count,biases,Gemm.row0(accumulators)))
  +e1 = put_next(~Tile.Tile8,U32.is_lt(1,rows),o0,tree,p0,(base + row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row1(accumulators)),e0)
  +e2 = put_next(~Tile.Tile8,U32.is_lt(2,rows),o1,tree,w1,(base + 2 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row2(accumulators)),e1)
  +e3 = put_next(~Tile.Tile8,U32.is_lt(3,rows),o2,tree,w2,(base + 3 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row3(accumulators)),e2)
  +e4 = put_next(~Tile.Tile8,U32.is_lt(4,rows),o3,tree,w3,(base + 4 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row4(accumulators)),e3)
  +e5 = put_next(~Tile.Tile8,U32.is_lt(5,rows),o4,tree,w4,(base + 5 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row5(accumulators)),e4)
  +e6 = put_next(~Tile.Tile8,U32.is_lt(6,rows),o5,tree,w5,(base + 6 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row6(accumulators)),e5)
  Equal.cong(Array<Tile.Tile8>,Gemm.Written<Tile.Tile8,Bias>,result => Gemm.Written{bias,result},
    Gemm.put(~Tile.Tile8,U32.is_lt(7,rows),o6,(base + 7 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row7(accumulators))),
    Writes.native(Tile.Tile8,tile_rows(~close,base,row_stride,rows,count,biases,accumulators),Tree.pack(Tile.Tile8,tree)),
    put_next(~Tile.Tile8,U32.is_lt(7,rows),o6,tree,w6,(base + 7 * row_stride : U32),Gemm.tile_finished(~close,count,biases,Gemm.row7(accumulators)),e6))

# The tile writer applies the block writes of the tile model.
def tiles_stored(~fetch: Array<F32> -> U32 -> Array<F32> & F32,~value: Tree.Tree<F32> -> U32 -> F32,~fetched: Fetched(~fetch,~value),~close: F32 -> F32 -> F32,
  +bias_start: U32,+store: Gemm.Store,+accumulators: Gemm.Accumulators,+bias: Tree.Tree<F32>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Gemm.write_tiles(~fetch,~close,bias_start,store,accumulators,Gemm.Written{Tree.pack(F32,bias),Tree.pack(Tile.Tile8,tree)}) ==
    stored_written(~Tile.Tile8,bias,block_writes(~value,~close,Tiles{},bias_start,store,accumulators,bias),Tree.pack(Tile.Tile8,tree)) : Gemm.Written<Tile.Tile8,Array<F32>>}:
  Gemm.Store{+base,+row_stride,+lane_stride,+rows,+count} = store
  +biases = bias_values(~value,bias,bias_start)
  %Equal.sym(Array<F32> & Tile.Tile8,Gemm.bias_lanes(~fetch,Tree.pack(F32,bias),bias_start),(Tree.pack(F32,bias),biases),biases_fetched(~fetch,~value,~fetched,bias,bias_start)) :
    {Gemm.tiles_finished(~Array<F32>,~close,base,row_stride,rows,count,accumulators,Tree.pack(Tile.Tile8,tree),_) ==
      stored_written(~Tile.Tile8,bias,block_writes(~value,~close,Tiles{},bias_start,Gemm.Store{base,row_stride,lane_stride,rows,count},accumulators,bias),
        Tree.pack(Tile.Tile8,tree)) : Gemm.Written<Tile.Tile8,Array<F32>>}
  tiles_native(~Array<F32>,~close,Tree.pack(F32,bias),tree,base,row_stride,rows,count,biases,accumulators)

# ---- A block and a run ----

def operand_bias(~E: Data,operands: Operands<E>) -> Tree.Tree<F32>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  bias

def appended(~E: Data,operands: Operands<E>,+writes: List<&2,Writes.Assignment<E>>) -> Operands<E>:
  Operands{row_tiles,lane_tiles,bias,output} = operands
  Operands{row_tiles,lane_tiles,bias,Writes.Append{writes,output}}

def finish_matches(~Cursor: Data,~writer: Writer,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: Stored(~writer,~write,~value,~close),
  +operands: Operands<Element(writer)>,+row: U32,+store: Gemm.Store,+state: Reduced<Cursor>) ->
  {Gemm.finish(~Cursor,~Element(writer),~write,represent_loop(~Cursor,~Element(writer),~Array<F32>,Tree.pack(F32,operand_bias(~Element(writer),operands)),operands,row,store,state)) ==
    represent(~Element(writer),appended(~Element(writer),operands,
      block_writes(~value,~close,writer,row,store,reduced_accumulators(~Cursor,state),operand_bias(~Element(writer),operands)))) : Gemm.Operands<Element(writer),Array<F32>>}:
  Operands{+row_tiles,+lane_tiles,+bias,+output} = operands
  Reduced{+cursor,+accumulators} = state
  +tree = Writes.evaluate(Element(writer),output)
  +writes = block_writes(~value,~close,writer,row,store,accumulators,bias)
  %Equal.sym(Gemm.Written<Element(writer),Array<F32>>,write(row,store,accumulators,Gemm.Written{Tree.pack(F32,bias),Tree.pack(Element(writer),tree)}),
    stored_written(~Element(writer),bias,writes,Tree.pack(Element(writer),tree)),stored(row,store,accumulators,bias,tree)) :
    {Gemm.operands_written(~Element(writer),Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),_) ==
      represent(~Element(writer),appended(~Element(writer),Operands{row_tiles,lane_tiles,bias,output},writes)) : Gemm.Operands<Element(writer),Array<F32>>}
  %Equal.sym(Array<Element(writer)>,Writes.native(Element(writer),writes,Tree.pack(Element(writer),tree)),Tree.pack(Element(writer),Writes.pure(Element(writer),writes,tree)),
    Writes.refinement(Element(writer),writes,tree)) :
    {Gemm.Operands{Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),Tree.pack(F32,bias),_} ==
      represent(~Element(writer),appended(~Element(writer),Operands{row_tiles,lane_tiles,bias,output},writes)) : Gemm.Operands<Element(writer),Array<F32>>}
  {==}

# One block: the k loop from +0, then the stores.
def block(~Cursor: Data,~writer: Writer,~advance: Cursor -> Gemm.Step<Cursor>,~order: Order,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+job: Gemm.Job<Cursor>,+operands: Operands<Element(writer)>) -> Operands<Element(writer)>:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  +state = reduce(~Cursor,~advance,~order,depth,operand_row_tiles(~Element(writer),operands),operand_lane_tiles(~Element(writer),operands),
    Reduced{cursor,Gemm.zero()})
  appended(~Element(writer),operands,block_writes(~value,~close,writer,row,store,reduced_accumulators(~Cursor,state),operand_bias(~Element(writer),operands)))

def looped_matches(~Cursor: Data,~writer: Writer,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: Stored(~writer,~write,~value,~close),
  +row: U32,+depth: Nat,+cursor: Cursor,+store: Gemm.Store,+operands: Operands<Element(writer)>,+initial: Gemm.Accumulators) ->
  {Gemm.finish(~Cursor,~Element(writer),~write,Gemm.looped(~Cursor,~Element(writer),~Array<F32>,~advance,~times,Gemm.Job{row,depth,cursor,store},(represent(~Element(writer),operands),initial))) ==
    represent(~Element(writer),appended(~Element(writer),operands,block_writes(~value,~close,writer,row,store,
      reduced_accumulators(~Cursor,reduce(~Cursor,~advance,~order,depth,operand_row_tiles(~Element(writer),operands),operand_lane_tiles(~Element(writer),operands),Reduced{cursor,initial})),
      operand_bias(~Element(writer),operands)))) : Gemm.Operands<Element(writer),Array<F32>>}:
  Operands{+row_tiles,+lane_tiles,+bias,+output} = operands
  +ops = {Operands{row_tiles,lane_tiles,bias,output} : Operands<Element(writer)>}
  +start = {Reduced{cursor,initial} : Reduced<Cursor>}
  -loop = represent_loop(~Cursor,~Element(writer),~Array<F32>,Tree.pack(F32,bias),ops,row,store,start)
  +done = reduce(~Cursor,~advance,~order,depth,row_tiles,lane_tiles,start)
  -result = represent(~Element(writer),appended(~Element(writer),ops,block_writes(~value,~close,writer,row,store,reduced_accumulators(~Cursor,done),bias)))
  %Equal.sym(Gemm.Loop<Cursor,Element(writer),Array<F32>>,Loop.run(~Gemm.Loop<Cursor,Element(writer),Array<F32>>,~Gemm.step(~Cursor,~Element(writer),~Array<F32>,~advance,~times),depth,loop),
    represent_loop(~Cursor,~Element(writer),~Array<F32>,Tree.pack(F32,bias),ops,row,store,done),loop_matches(~Cursor,~Element(writer),~Array<F32>,~advance,~times,~order,~ordered,depth,Tree.pack(F32,bias),ops,row,store,start)) :
    {Gemm.finish(~Cursor,~Element(writer),~write,_) == result : Gemm.Operands<Element(writer),Array<F32>>}
  finish_matches(~Cursor,~writer,~write,~value,~close,~stored,ops,row,store,done)

def block_matches(~Cursor: Data,~writer: Writer,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: Stored(~writer,~write,~value,~close),
  +job: Gemm.Job<Cursor>,+operands: Operands<Element(writer)>) ->
  {Gemm.block(~Cursor,~Element(writer),~advance,~times,~write,job,represent(~Element(writer),operands)) ==
    represent(~Element(writer),block(~Cursor,~writer,~advance,~order,~value,~close,job,operands)) : Gemm.Operands<Element(writer),Array<F32>>}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  looped_matches(~Cursor,~writer,~advance,~times,~order,~ordered,~write,~value,~close,~stored,row,depth,cursor,store,operands,Gemm.zero())

def job_step(~Plan: Data,~Cursor: Data,~writer: Writer,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: Order,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  index: U32,plan: Plan,operands: Operands<Element(writer)>) -> Operands<Element(writer)>:
  block(~Cursor,~writer,~advance,~order,~value,~close,job_at(plan,index),operands)

def job_step_matches(~Plan: Data,~Cursor: Data,~writer: Writer,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: Stored(~writer,~write,~value,~close),+index: U32,+plan: Plan,+operands: Operands<Element(writer)>) ->
  {Gemm.job_step(~Plan,~Cursor,~Element(writer),~job_at,~advance,~times,~write,index,plan,represent(~Element(writer),operands)) ==
    represent(~Element(writer),job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close,index,plan,operands)) : Gemm.Operands<Element(writer),Array<F32>>}:
  block_matches(~Cursor,~writer,~advance,~times,~order,~ordered,~write,~value,~close,~stored,job_at(plan,index),operands)

# Jobs 0 to count - 1 of the plan, each appending its block's writes.
def run(~Plan: Data,~Cursor: Data,~writer: Writer,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: Order,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  count: Nat,plan: Plan,operands: Operands<Element(writer)>) -> Operands<Element(writer)>:
  Traversal.indexed_fold(~Operands<Element(writer)>,~Plan,~job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),count,0,plan,operands)

def run_matches(~Plan: Data,~Cursor: Data,~writer: Writer,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: Order,~ordered: Ordered(~times,order),
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<Element(writer),Array<F32>> -> Gemm.Written<Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: Stored(~writer,~write,~value,~close),+count: Nat,+plan: Plan,+operands: Operands<Element(writer)>) ->
  {Gemm.run(~Plan,~Cursor,~Element(writer),~job_at,~advance,~times,~write,count,plan,represent(~Element(writer),operands)) ==
    represent(~Element(writer),run(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close,count,plan,operands)) : Gemm.Operands<Element(writer),Array<F32>>}:
  match count:
    case 0n:
      ArrayLaws.indexed_fold_refines_model(~Gemm.Operands<Element(writer),Array<F32>>,~Operands<Element(writer)>,~Plan,
        ~Gemm.job_step(~Plan,~Cursor,~Element(writer),~job_at,~advance,~times,~write),~job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),
        ~represent(~Element(writer)),~job_step_matches(~Plan,~Cursor,~writer,~job_at,~advance,~times,~order,~ordered,~write,~value,~close,~stored),0n,0,plan,operands)
    case 1n+ +rest:
      ArrayLaws.indexed_fold_refines_model(~Gemm.Operands<Element(writer),Array<F32>>,~Operands<Element(writer)>,~Plan,
        ~Gemm.job_step(~Plan,~Cursor,~Element(writer),~job_at,~advance,~times,~write),~job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),
        ~represent(~Element(writer)),~job_step_matches(~Plan,~Cursor,~writer,~job_at,~advance,~times,~order,~ordered,~write,~value,~close,~stored),1n+rest,0,plan,operands)

# Cell-bias jobs share the reduction and the tile write program.
type CellOperands is Data:
  CellOperands{bias: Tree.Tree<Tile.Tile8>,product: Operands<Tile.Tile8>}

def cell_represent(state: CellOperands) -> Gemm.CellOperands:
  CellOperands{bias,product} = state
  Gemm.CellOperands{Tree.pack(Tile.Tile8,bias),represent_owned(~Tile.Tile8,~Unit,Unit{},product)}

def cell_rows(+bias: Tree.Tree<Tile.Tile8>,+row: U32,store: Gemm.Store,+accumulators: Gemm.Accumulators) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Gemm.Store{base,row_stride,lane_stride,rows,count} = store
  tile_rows(~Gemm.plus,base,row_stride,rows,count,Tree.read(Tile.Tile8,bias,(row / 8 : U32)),accumulators)

def cell_appended(state: CellOperands,writes: List<&2,Writes.Assignment<Tile.Tile8>>) -> CellOperands:
  CellOperands{bias,product} = state
  CellOperands{bias,appended(~Tile.Tile8,product,writes)}

def cell_finish_matches(+bias: Tree.Tree<Tile.Tile8>,+product: Operands<Tile.Tile8>,+row: U32,+store: Gemm.Store,+state: Reduced<Gemm.Direct()>) ->
  {Gemm.cell_finish(Tree.pack(Tile.Tile8,bias),represent_loop(~Gemm.Direct(),~Tile.Tile8,~Unit,Unit{},product,row,store,state)) ==
    cell_represent(cell_appended(CellOperands{bias,product},cell_rows(bias,row,store,reduced_accumulators(~Gemm.Direct(),state)))) : Gemm.CellOperands}:
  Operands{+row_tiles,+lane_tiles,+unused_bias,+output} = product
  Gemm.Store{+base,+row_stride,lane_stride,+rows,+count} = store
  Reduced{cursor,+accumulators} = state
  +tree = Writes.evaluate(Tile.Tile8,output)
  +biases = Tree.read(Tile.Tile8,bias,(row / 8 : U32))
  +writes = tile_rows(~Gemm.plus,base,row_stride,rows,count,biases,accumulators)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,bias),(row / 8 : U32)),(Tree.pack(Tile.Tile8,bias),biases),StorageLaws.read(Tile.Tile8,bias,(row / 8 : U32))) :
    {Gemm.cell_read(Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),Unit{},Tree.pack(Tile.Tile8,tree),store,accumulators,_) ==
      cell_represent(cell_appended(CellOperands{bias,product},writes)) : Gemm.CellOperands}
  %Equal.sym(Gemm.Written<Tile.Tile8,Unit>,Gemm.tiles_finished(~Unit,~Gemm.plus,base,row_stride,rows,count,accumulators,Tree.pack(Tile.Tile8,tree),(Unit{},biases)),
    Gemm.Written{Unit{},Writes.native(Tile.Tile8,writes,Tree.pack(Tile.Tile8,tree))},tiles_native(~Unit,~Gemm.plus,Unit{},tree,base,row_stride,rows,count,biases,accumulators)) :
    {Gemm.cell_stored(Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),Tree.pack(Tile.Tile8,bias),_) ==
      cell_represent(cell_appended(CellOperands{bias,product},writes)) : Gemm.CellOperands}
  %Equal.sym(Array<Tile.Tile8>,Writes.native(Tile.Tile8,writes,Tree.pack(Tile.Tile8,tree)),Tree.pack(Tile.Tile8,Writes.pure(Tile.Tile8,writes,tree)),Writes.refinement(Tile.Tile8,writes,tree)) :
    {Gemm.CellOperands{Tree.pack(Tile.Tile8,bias),Gemm.Operands{Tree.pack(Tile.Tile8,row_tiles),Tree.pack(Tile.Tile8,lane_tiles),Unit{},_}} ==
      cell_represent(cell_appended(CellOperands{bias,product},writes)) : Gemm.CellOperands}
  {==}

def cell_block(+job: Gemm.Job<Gemm.Direct()>,+state: CellOperands) -> CellOperands:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  CellOperands{+bias,+product} = state
  +reduced = reduce(~Gemm.Direct(),~Gemm.directs,~RowFirst{},depth,operand_row_tiles(~Tile.Tile8,product),operand_lane_tiles(~Tile.Tile8,product),Reduced{cursor,Gemm.zero()})
  cell_appended(state,cell_rows(bias,row,store,reduced_accumulators(~Gemm.Direct(),reduced)))

def cell_block_matches(+job: Gemm.Job<Gemm.Direct()>,+bias: Tree.Tree<Tile.Tile8>,+product: Operands<Tile.Tile8>) ->
  {Gemm.cell_finish(Tree.pack(Tile.Tile8,bias),Gemm.looped(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times,job,(represent_owned(~Tile.Tile8,~Unit,Unit{},product),Gemm.zero()))) ==
    cell_represent(cell_block(job,CellOperands{bias,product})) : Gemm.CellOperands}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Operands{row_tiles,lane_tiles,unused_bias,output} = product
  +initial = {Reduced{cursor,Gemm.zero()} : Reduced<Gemm.Direct()>}
  +done = reduce(~Gemm.Direct(),~Gemm.directs,~RowFirst{},depth,operand_row_tiles(~Tile.Tile8,product),operand_lane_tiles(~Tile.Tile8,product),initial)
  %Equal.sym(Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>,Loop.run(~Gemm.Loop<Gemm.Direct(),Tile.Tile8,Unit>,~Gemm.step(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times),depth,represent_loop(~Gemm.Direct(),~Tile.Tile8,~Unit,Unit{},product,row,store,initial)),
    represent_loop(~Gemm.Direct(),~Tile.Tile8,~Unit,Unit{},product,row,store,done),loop_matches(~Gemm.Direct(),~Tile.Tile8,~Unit,~Gemm.directs,~Gemm.row_times,~RowFirst{},~row_first,depth,Unit{},product,row,store,initial)) :
    {Gemm.cell_finish(Tree.pack(Tile.Tile8,bias),_) == cell_represent(cell_block(job,CellOperands{bias,product})) : Gemm.CellOperands}
  cell_finish_matches(bias,product,row,store,done)

def cell_job(index: U32,matrix: Gemm.TileMatrix,state: CellOperands) -> CellOperands:
  cell_block(Gemm.tile_major(matrix,index),state)

def cell_job_matches(+index: U32,+matrix: Gemm.TileMatrix,+state: CellOperands) ->
  {Gemm.cell_job(index,matrix,cell_represent(state)) == cell_represent(cell_job(index,matrix,state)) : Gemm.CellOperands}:
  CellOperands{+bias,+product} = state
  cell_block_matches(Gemm.tile_major(matrix,index),bias,product)

def cell_run(count: Nat,matrix: Gemm.TileMatrix,state: CellOperands) -> CellOperands:
  Traversal.indexed_fold(~CellOperands,~Gemm.TileMatrix,~cell_job,count,0,matrix,state)

def cell_run_matches(+count: Nat,+matrix: Gemm.TileMatrix,+state: CellOperands) ->
  {Gemm.cell_run(count,matrix,cell_represent(state)) == cell_represent(cell_run(count,matrix,state)) : Gemm.CellOperands}:
  %Equal.sym(Gemm.CellOperands,Gemm.cell_run(count,matrix,cell_represent(state)),Traversal.indexed_fold(~Gemm.CellOperands,~Gemm.TileMatrix,~Gemm.cell_job,count,0,matrix,cell_represent(state)),
    GemmCertificate.cell_run_same(count,matrix,cell_represent(state))) : {_ == cell_represent(cell_run(count,matrix,state)) : Gemm.CellOperands}
  ArrayLaws.indexed_fold_refines_model(~Gemm.CellOperands,~CellOperands,~Gemm.TileMatrix,~Gemm.cell_job,~cell_job,~cell_represent,~cell_job_matches,count,0,matrix,state)

# The eight scalar observations of a bias cell, used only by the write model.
def bias_tree(biases: Tile.Tile8) -> Tree.Tree<F32>:
  Tile.Tile{a,b,c,d,e,f,g,h} = biases
  Tree.Node{Tree.Node{Tree.Node{Tree.Leaf{a},Tree.Leaf{b}},Tree.Node{Tree.Leaf{c},Tree.Leaf{d}}},
    Tree.Node{Tree.Node{Tree.Leaf{e},Tree.Leaf{f}},Tree.Node{Tree.Leaf{g},Tree.Leaf{h}}}}

def cell_rows_block(+biases: Tile.Tile8,+store: Gemm.Store,+accumulators: Gemm.Accumulators) ->
  {cell_rows(Tree.Leaf{biases},0,store,accumulators) == block_writes(~biased_value,~Gemm.plus,Tiles{},0,store,accumulators,bias_tree(biases)) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  Tile.Tile{a,b,c,d,e,f,g,h} = biases
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  {==}

def bias_tree_lane(+biases: Tile.Tile8,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,bias_tree(biases),shifted(0,lane)) == TileLaws.select_lane(lane,biases) : F32}:
  Tile.Tile{a,b,c,d,e,f,g,h} = biases
  match lane:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+rest: Empty.absurd({Tree.read(F32,bias_tree(biases),shifted(0,8n+rest)) == TileLaws.select_lane(8n+rest,biases) : F32},NaturalOrder.below_zero(rest,inside))
