# A strided block store is one write list: row r < rows, lane i < count
# writes its accumulator lane plus +0.0 at the row's target + i * step, rows
# and lanes in store order. The list is the writes of the block's view.
import Base
import ../kernel_strided_store.bend as Stores
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ./write_program.bend as Writes
import ./kernel_output_refinement.bend as Lanes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./kernel_tile_refinement.bend as Tiles
import ./kernel_accumulator.bend as Accumulator
import ../tensor_view.bend as Views
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./binary_radix.bend as Radix

# The first two cases use exactly the expressions in Stores.row/block.
def address(+target: U32,+stride: U32,index: Nat) -> U32:
  match index:
    case 0n: target
    case 1n: (target + stride : U32)
    case 2n+rest: (target + U32.from_nat(2n+rest) * stride : U32)

type Row is Data:
  Row{target: U32,step: U32,count: U32,values: Tile.Tile8}

def row_target(context: Row) -> U32:
  Row{target,step,count,values} = context
  target
def row_step(context: Row) -> U32:
  Row{target,step,count,values} = context
  step
def row_count(context: Row) -> U32:
  Row{target,step,count,values} = context
  count
def row_value(context: Row,index: Nat) -> F32:
  Row{target,step,count,values} = context
  (Tiles.select_lane(index,values) + 0.0 : F32)
def row_address(context: Row,index: Nat) -> U32:
  Row{+target,+step,count,values} = context
  address(target,step,index)

def row_values(context: Row) -> Tile.Tile8:
  Row{target,step,count,values} = context
  values

type Block is Data:
  Block{store: Stores.Store,values: Rows.Rows<Tile.Tile8>}

def store(context: Block) -> Stores.Store:
  Block{store,values} = context
  store
def values(context: Block) -> Rows.Rows<Tile.Tile8>:
  Block{store,values} = context
  values
def store_target(store: Stores.Store) -> U32:
  Stores.Store{target,step,plane,rows,count} = store
  target
def store_step(store: Stores.Store) -> U32:
  Stores.Store{target,step,plane,rows,count} = store
  step
def store_plane(store: Stores.Store) -> U32:
  Stores.Store{target,step,plane,rows,count} = store
  plane
def store_rows(store: Stores.Store) -> U32:
  Stores.Store{target,step,plane,rows,count} = store
  rows
def store_count(store: Stores.Store) -> U32:
  Stores.Store{target,step,plane,rows,count} = store
  count
def rows(+context: Block) -> U32:
  store_rows(store(context))

def row_context(+context: Block,+index: Nat) -> Row:
  +store = store(context)
  Row{address(store_target(store),store_plane(store),index),store_step(store),store_count(store),Accumulator.select_row(~Tile.Tile8,index,values(context))}

def encoded_address(+target: U32,+stride: U32,index: Nat) -> {address(target,stride,index) == (target + U32.from_nat(index) * stride : U32) : U32}:
  match index:
    case 0n:
      Equal.sym(U32,(target + 0 * stride : U32),target,
        Equal.trans(U32,(target + 0 * stride : U32),(target + 0 : U32),target,Equal.cong(U32,U32,value => (target + value : U32),(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)),Arithmetic.add_zero(target)))
    case 1n:
      +one = Equal.trans(U32,(1 * stride : U32),(0 * stride + stride : U32),stride,Arithmetic.multiply_successor(0,stride),
        Equal.trans(U32,(0 * stride + stride : U32),(0 + stride : U32),stride,Equal.cong(U32,U32,value => (value + stride : U32),(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)),Arithmetic.add_zero_left(stride)))
      Equal.sym(U32,(target + 1 * stride : U32),(target + stride : U32),Equal.cong(U32,U32,value => (target + value : U32),(1 * stride : U32),stride,one))
    case 2n+rest: {==}

# A cell of a block: the address it writes and the value it stores.

type BlockQuery is Data:
  BlockQuery{context: Block,row: Nat,lane: Nat}
def block_context(query: BlockQuery) -> Block:
  BlockQuery{context,row,lane} = query
  context
def block_selected_row(query: BlockQuery) -> Nat:
  BlockQuery{context,row,lane} = query
  row
def block_selected_lane(query: BlockQuery) -> Nat:
  BlockQuery{context,row,lane} = query
  lane

# The address of a cell depends on its store only.
def cell_address(+store: Stores.Store,+row: Nat,lane: Nat) -> U32:
  address(address(store_target(store),store_plane(store),row),store_step(store),lane)
def block_address(query: BlockQuery) -> U32:
  +query = query
  row_address(row_context(block_context(query),block_selected_row(query)),block_selected_lane(query))
def block_value(query: BlockQuery) -> F32:
  +query = query
  row_value(row_context(block_context(query),block_selected_row(query)),block_selected_lane(query))

def row_writes(valid: Bool,+row: Row) -> List<&2,Writes.Assignment<F32>>:
  match valid:
    case False{}: []
    case True{}:
      +count = row_count(row)
      written = Lanes.lane_writes(U32.is_lt(0,count),row_address(row,0n),row_value(row,0n))
      written = Lanes.append_lane(U32.is_lt(1,count),written,row_address(row,1n),row_value(row,1n))
      written = Lanes.append_lane(U32.is_lt(2,count),written,row_address(row,2n),row_value(row,2n))
      written = Lanes.append_lane(U32.is_lt(3,count),written,row_address(row,3n),row_value(row,3n))
      written = Lanes.append_lane(U32.is_lt(4,count),written,row_address(row,4n),row_value(row,4n))
      written = Lanes.append_lane(U32.is_lt(5,count),written,row_address(row,5n),row_value(row,5n))
      written = Lanes.append_lane(U32.is_lt(6,count),written,row_address(row,6n),row_value(row,6n))
      Lanes.append_lane(U32.is_lt(7,count),written,row_address(row,7n),row_value(row,7n))

def row_native(valid: Bool,-output: Array<F32>,+row: Row) ->
  {Stores.row(valid,output,row_target(row),row_step(row),row_count(row),row_values(row)) == Writes.native(F32,row_writes(valid,row),output) : Array<F32>}:
  match valid:
    case False{}: {==}
    case True{}:
      Row{+target,+step,+count,+accumulator} = row
      +written0 = Lanes.lane_writes(U32.is_lt(0,count),target,(Tile.lane0(accumulator) + 0.0 : F32))
      +written1 = Lanes.append_lane(U32.is_lt(1,count),written0,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32))
      +written2 = Lanes.append_lane(U32.is_lt(2,count),written1,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32))
      +written3 = Lanes.append_lane(U32.is_lt(3,count),written2,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32))
      +written4 = Lanes.append_lane(U32.is_lt(4,count),written3,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32))
      +written5 = Lanes.append_lane(U32.is_lt(5,count),written4,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32))
      +written6 = Lanes.append_lane(U32.is_lt(6,count),written5,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32))
      -stored0 = Gemm.store_lane(U32.is_lt(0,count),output,target,(Tile.lane0(accumulator) + 0.0 : F32))
      -stored1 = Gemm.store_lane(U32.is_lt(1,count),stored0,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32))
      -stored2 = Gemm.store_lane(U32.is_lt(2,count),stored1,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32))
      -stored3 = Gemm.store_lane(U32.is_lt(3,count),stored2,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32))
      -stored4 = Gemm.store_lane(U32.is_lt(4,count),stored3,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32))
      -stored5 = Gemm.store_lane(U32.is_lt(5,count),stored4,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32))
      -stored6 = Gemm.store_lane(U32.is_lt(6,count),stored5,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32))
      Lanes.next_lane(U32.is_lt(7,count),stored6,written6,output,(target + 7 * step : U32),(Tile.lane7(accumulator) + 0.0 : F32),
        Lanes.next_lane(U32.is_lt(6,count),stored5,written5,output,(target + 6 * step : U32),(Tile.lane6(accumulator) + 0.0 : F32),
          Lanes.next_lane(U32.is_lt(5,count),stored4,written4,output,(target + 5 * step : U32),(Tile.lane5(accumulator) + 0.0 : F32),
            Lanes.next_lane(U32.is_lt(4,count),stored3,written3,output,(target + 4 * step : U32),(Tile.lane4(accumulator) + 0.0 : F32),
              Lanes.next_lane(U32.is_lt(3,count),stored2,written2,output,(target + 3 * step : U32),(Tile.lane3(accumulator) + 0.0 : F32),
                Lanes.next_lane(U32.is_lt(2,count),stored1,written1,output,(target + 2 * step : U32),(Tile.lane2(accumulator) + 0.0 : F32),
                  Lanes.next_lane(U32.is_lt(1,count),stored0,written0,output,(target + step : U32),(Tile.lane1(accumulator) + 0.0 : F32),
                    Lanes.native_lane(U32.is_lt(0,count),output,target,(Tile.lane0(accumulator) + 0.0 : F32)))))))))

def block_writes(+block: Block) -> List<&2,Writes.Assignment<F32>>:
  +rows = rows(block)
  List.append(&2,Writes.Assignment<F32>,
    List.append(&2,Writes.Assignment<F32>,
      List.append(&2,Writes.Assignment<F32>,
        row_writes(U32.is_lt(0,rows),row_context(block,0n)),
        row_writes(U32.is_lt(1,rows),row_context(block,1n))),
      row_writes(U32.is_lt(2,rows),row_context(block,2n))),
    row_writes(U32.is_lt(3,rows),row_context(block,3n)))

def block_native(+block: Block,-output: Array<F32>) ->
  {Stores.block(store(block),values(block),output) == Writes.native(F32,block_writes(block),output) : Array<F32>}:
  Block{Stores.Store{+target,+step,+plane,+rows,+count},+accumulators} = block
  +first = row_writes(U32.is_lt(0,rows),row_context(block,0n))
  +second = row_writes(U32.is_lt(1,rows),row_context(block,1n))
  +third = row_writes(U32.is_lt(2,rows),row_context(block,2n))
  +fourth = row_writes(U32.is_lt(3,rows),row_context(block,3n))
  +two = List.append(&2,Writes.Assignment<F32>,first,second)
  +three = List.append(&2,Writes.Assignment<F32>,two,third)
  -stored1 = Stores.row(U32.is_lt(0,rows),output,target,step,count,Rows.first(~Tile.Tile8,accumulators))
  -stored2 = Stores.row(U32.is_lt(1,rows),stored1,(target + plane : U32),step,count,Rows.second(~Tile.Tile8,accumulators))
  -stored3 = Stores.row(U32.is_lt(2,rows),stored2,(target + 2 * plane : U32),step,count,Rows.third(~Tile.Tile8,accumulators))
  %Equal.sym(Array<F32>,Writes.native(F32,List.append(&2,Writes.Assignment<F32>,three,fourth),output),Writes.native(F32,fourth,Writes.native(F32,three,output)),
    Writes.concatenated_native(F32,three,fourth,output)) : {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == _ : Array<F32>}
  %Equal.sym(Array<F32>,Writes.native(F32,three,output),Writes.native(F32,third,Writes.native(F32,two,output)),
    Writes.concatenated_native(F32,two,third,output)) : {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == Writes.native(F32,fourth,_) : Array<F32>}
  %Equal.sym(Array<F32>,Writes.native(F32,two,output),Writes.native(F32,second,Writes.native(F32,first,output)),
    Writes.concatenated_native(F32,first,second,output)) : {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == Writes.native(F32,fourth,Writes.native(F32,third,_)) : Array<F32>}
  %row_native(U32.is_lt(0,rows),output,row_context(block,0n)) :
    {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == Writes.native(F32,fourth,Writes.native(F32,third,Writes.native(F32,second,_))) : Array<F32>}
  %row_native(U32.is_lt(1,rows),stored1,row_context(block,1n)) :
    {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == Writes.native(F32,fourth,Writes.native(F32,third,_)) : Array<F32>}
  %row_native(U32.is_lt(2,rows),stored2,row_context(block,2n)) :
    {Stores.row(U32.is_lt(3,rows),stored3,(target + 3 * plane : U32),step,count,Rows.fourth(~Tile.Tile8,accumulators)) == Writes.native(F32,fourth,_) : Array<F32>}
  row_native(U32.is_lt(3,rows),stored3,row_context(block,3n))

# The lanes of a row and the rows of a block, each written when stored.
def lane_part(+row: Row,+index: Nat) -> List<&2,Writes.Assignment<F32>>:
  Lanes.lane_writes(U32.is_lt(U32.from_nat(index),row_count(row)),row_address(row,index),row_value(row,index))
def row_part(+block: Block,+index: Nat) -> List<&2,Writes.Assignment<F32>>:
  row_writes(U32.is_lt(U32.from_nat(index),rows(block)),row_context(block,index))

# A block store writes through a view: the stored rows, at most four, by the
# stored lanes, at most eight. Element k of the view is lane k mod lanes of
# accumulator row k / lanes, plus +0.0.
def store_view(+store: Stores.Store) -> Views.View:
  Views.View{[Views.Axis{U32.min(4,store_rows(store)),store_plane(store)},Views.Axis{U32.min(8,store_count(store)),store_step(store)}],store_target(store)}
def stored_rows(+store: Stores.Store) -> Nat:
  U32.to_nat(U32.min(4,store_rows(store)))
def stored_lanes(+store: Stores.Store) -> Nat:
  U32.to_nat(U32.min(8,store_count(store)))
def tile_value(+block: Block,+position: Nat) -> F32:
  +lanes = stored_lanes(store(block))
  (Tiles.select_lane(Nat.mod(position,lanes),Accumulator.select_row(~Tile.Tile8,Nat.div(position,lanes),values(block))) + 0.0 : F32)

# A cell of a stored row writes the view address of its position.
def cell_view_address(+block: Block,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,stored_rows(store(block))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,stored_lanes(store(block))) == True{} : Bool}) ->
  {row_address(row_context(block,row),lane) == Copy.address(store_view(store(block)),Nat.add(Nat.mul(row,stored_lanes(store(block))),lane)) : U32}:
  +store = store(block)
  +target = store_target(store)
  +plane = store_plane(store)
  +step = store_step(store)
  +start = address(target,plane,row)
  Equal.trans(U32,address(start,step,lane),(start + U32.from_nat(lane) * step : U32),Copy.address(store_view(store),Nat.add(Nat.mul(row,stored_lanes(store)),lane)),
    encoded_address(start,step,lane),
    Equal.trans(U32,(start + U32.from_nat(lane) * step : U32),(target + U32.from_nat(row) * plane + U32.from_nat(lane) * step : U32),
      Copy.address(store_view(store),Nat.add(Nat.mul(row,stored_lanes(store)),lane)),
      Equal.cong(U32,U32,word => (word + U32.from_nat(lane) * step : U32),start,(target + U32.from_nat(row) * plane : U32),encoded_address(target,plane,row)),
      Equal.sym(U32,Copy.address(store_view(store),Nat.add(Nat.mul(row,stored_lanes(store)),lane)),(target + U32.from_nat(row) * plane + U32.from_nat(lane) * step : U32),
        ViewWrites.plane_address(target,U32.min(4,store_rows(store)),plane,row,U32.min(8,store_count(store)),step,lane,row_inside,lane_inside))))

# A cell of a stored row stores the view element of its position.
def cell_view_value(+block: Block,+row: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,stored_lanes(store(block))) == True{} : Bool}) ->
  {row_value(row_context(block,row),lane) == tile_value(block,Nat.add(Nat.mul(row,stored_lanes(store(block))),lane)) : F32}:
  +lanes = stored_lanes(store(block))
  +position = Nat.add(Nat.mul(row,lanes),lane)
  +positive = Order.positive_bound(lane,lanes,lane_inside)
  %Equal.sym(Nat,Nat.div(position,lanes),row,Radix.quotient(row,lane,lanes,positive,lane_inside)) :
    {row_value(row_context(block,row),lane) == (Tiles.select_lane(Nat.mod(position,lanes),Accumulator.select_row(~Tile.Tile8,_,values(block))) + 0.0 : F32) : F32}
  %Equal.sym(Nat,Nat.mod(position,lanes),lane,Radix.remainder(row,lane,lanes,positive,lane_inside)) :
    {row_value(row_context(block,row),lane) == (Tiles.select_lane(_,Accumulator.select_row(~Tile.Tile8,row,values(block))) + 0.0 : F32) : F32}
  {==}

# A stored row of a block.
type Line is Data:
  Line{block: Block,row: Nat,inside: {Nat.is_lt(row,stored_rows(store(block))) == True{} : Bool}}
def line_block(+line: Line) -> Block:
  Line{block,row,inside} = line
  block
def line_view(+line: Line) -> Views.View:
  store_view(store(line_block(line)))
def line_base(+line: Line) -> Nat:
  Line{block,row,inside} = line
  Nat.mul(row,stored_lanes(store(block)))
def line_lanes(+line: Line) -> Nat:
  stored_lanes(store(line_block(line)))
def single(+line: Line) -> Nat:
  1n
def eight(+line: Line) -> Nat:
  8n
def line_lane(+line: Line,+index: Nat) -> List<&2,Writes.Assignment<F32>>:
  Line{block,row,inside} = line
  lane_part(row_context(block,row),index)

def line_present(+line: Line,+index: Nat,+below: {Nat.is_lt(index,line_lanes(line)) == True{} : Bool}) ->
  {line_lane(line,index) == ViewWrites.writes_from(~F32,~Block,~tile_value,single(line),Nat.add(line_base(line),Nat.mul(index,single(line))),line_view(line),line_block(line)) :
    List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+inside} = line
  +context = row_context(block,row)
  +position = Nat.add(Nat.mul(row,stored_lanes(store(block))),index)
  +mask = ViewWrites.clipped_inside(index,8,store_count(store(block)),{==},below)
  %Equal.sym(Nat,Nat.mul(index,1n),index,Order.times_one(index)) :
    {lane_part(context,index) == ViewWrites.writes_from(~F32,~Block,~tile_value,1n,Nat.add(Nat.mul(row,stored_lanes(store(block))),_),store_view(store(block)),block) :
      List<&2,Writes.Assignment<F32>>}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),store_count(store(block))),True{},mask) :
    {Lanes.lane_writes(_,row_address(context,index),row_value(context,index)) == [Writes.Put{Copy.address(store_view(store(block)),position),tile_value(block,position)}] :
      List<&2,Writes.Assignment<F32>>}
  %cell_view_address(block,row,index,inside,below) :
    {[Writes.Put{row_address(context,index),row_value(context,index)}] == [Writes.Put{_,tile_value(block,position)}] : List<&2,Writes.Assignment<F32>>}
  %cell_view_value(block,row,index,below) :
    {[Writes.Put{row_address(context,index),row_value(context,index)}] == [Writes.Put{row_address(context,index),_}] : List<&2,Writes.Assignment<F32>>}
  {==}

def line_absent(+line: Line,+extra: Nat,+below: {Nat.is_lt(Nat.add(line_lanes(line),extra),eight(line)) == True{} : Bool}) ->
  {line_lane(line,Nat.add(line_lanes(line),extra)) == Nil{} : List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+inside} = line
  +context = row_context(block,row)
  +lanes = stored_lanes(store(block))
  +index = Nat.add(lanes,extra)
  +mask = ViewWrites.clipped_outside(index,8,store_count(store(block)),{==},below,Order.not_below(lanes,index,Order.le_add(lanes,extra)))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),store_count(store(block))),False{},mask) :
    {Lanes.lane_writes(_,row_address(context,index),row_value(context,index)) == Nil{} : List<&2,Writes.Assignment<F32>>}
  {==}

# A stored row is the run of the view's writes at the row's positions.
def line_writes(+block: Block,+row: Nat,+inside: {Nat.is_lt(row,stored_rows(store(block))) == True{} : Bool}) ->
  {row_writes(True{},row_context(block,row)) ==
    ViewWrites.writes_from(~F32,~Block,~tile_value,stored_lanes(store(block)),Nat.mul(row,stored_lanes(store(block))),store_view(store(block)),block) : List<&2,Writes.Assignment<F32>>}:
  +line = {Line{block,row,inside} : Line}
  +lanes = stored_lanes(store(block))
  %Order.times_one(lanes) :
    {row_writes(True{},row_context(block,row)) ==
      ViewWrites.writes_from(~F32,~Block,~tile_value,_,Nat.mul(row,lanes),store_view(store(block)),block) : List<&2,Writes.Assignment<F32>>}
  ViewWrites.masked(~F32,~Block,~tile_value,~Line,~line_lane,~line_view,~line_block,~line_base,~single,~line_lanes,~eight,~line_present,~line_absent,line,
    ViewWrites.limited(8,store_count(store(block))))

def same_block(+block: Block) -> Block:
  block
def block_view(+block: Block) -> Views.View:
  store_view(store(block))
def start(+block: Block) -> Nat:
  0n
def block_lanes(+block: Block) -> Nat:
  stored_lanes(store(block))
def block_rows(+block: Block) -> Nat:
  stored_rows(store(block))
def four(+block: Block) -> Nat:
  4n

def row_present(+block: Block,+index: Nat,+below: {Nat.is_lt(index,block_rows(block)) == True{} : Bool}) ->
  {row_part(block,index) == ViewWrites.writes_from(~F32,~Block,~tile_value,block_lanes(block),Nat.add(start(block),Nat.mul(index,block_lanes(block))),block_view(block),same_block(block)) :
    List<&2,Writes.Assignment<F32>>}:
  +mask = ViewWrites.clipped_inside(index,4,store_rows(store(block)),{==},below)
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),store_rows(store(block))),True{},mask) :
    {row_writes(_,row_context(block,index)) ==
      ViewWrites.writes_from(~F32,~Block,~tile_value,stored_lanes(store(block)),Nat.mul(index,stored_lanes(store(block))),store_view(store(block)),block) : List<&2,Writes.Assignment<F32>>}
  line_writes(block,index,below)

def row_absent(+block: Block,+extra: Nat,+below: {Nat.is_lt(Nat.add(block_rows(block),extra),four(block)) == True{} : Bool}) ->
  {row_part(block,Nat.add(block_rows(block),extra)) == Nil{} : List<&2,Writes.Assignment<F32>>}:
  +rows = stored_rows(store(block))
  +index = Nat.add(rows,extra)
  +mask = ViewWrites.clipped_outside(index,4,store_rows(store(block)),{==},below,Order.not_below(rows,index,Order.le_add(rows,extra)))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),store_rows(store(block))),False{},mask) :
    {row_writes(_,row_context(block,index)) == Nil{} : List<&2,Writes.Assignment<F32>>}
  {==}

# A block store is the writes of its view.
def view_writes(+block: Block) ->
  {block_writes(block) == ViewWrites.writes(~F32,~Block,~tile_value,Nat.mul(stored_rows(store(block)),stored_lanes(store(block))),store_view(store(block)),block) :
    List<&2,Writes.Assignment<F32>>}:
  ViewWrites.masked(~F32,~Block,~tile_value,~Block,~row_part,~block_view,~same_block,~start,~block_lanes,~block_rows,~four,~row_present,~row_absent,block,
    ViewWrites.limited(4,store_rows(store(block))))
