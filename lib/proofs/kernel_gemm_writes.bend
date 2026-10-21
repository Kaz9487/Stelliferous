# The whole GEMM storage model is one ordered write list. Readback proofs use
# that list; they do not re-derive the job traversal or the store sequence.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_reduction.bend as Reduction
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./kernel_output_refinement.bend as KernelOutputLaws
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_reduction_model.bend as ReductionModel
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./write_program_readback.bend as ProgramRead
import ./kernel_tile_refinement.bend as TileLaws
import ./word_encoding.bend as Words
import ./nat_division.bend as Division
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./list_storage_readback.bend as ListRead
import ./kernel_accumulator.bend as Accumulator
import ./traversal_lanes_refinement.bend as Lanes
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./word_arithmetic.bend as WordArithmetic
import ../tensor_view.bend as Views
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./binary_radix.bend as Radix

def row_writes(~biased: Bool,valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  match valid:
    case False{}: []
    case True{}: KernelOutputLaws.tail_assignments(biased,U32.min(8,(columns - position : U32)),(index * columns + position : U32),Tree.read(F32,bias,index),accumulator)

# Rows are appended in store order: row r writes after rows 0..r-1.
def context_writes(~biased: Bool,context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  List.append(&2,Writes.Assignment<F32>,
    List.append(&2,Writes.Assignment<F32>,
      List.append(&2,Writes.Assignment<F32>,
        row_writes(~biased,U32.is_lt(index,rows),index,position,columns,Rows.first(~Tile.Tile8,accumulators),bias),
        row_writes(~biased,U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,Rows.second(~Tile.Tile8,accumulators),bias)),
      row_writes(~biased,U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,Rows.third(~Tile.Tile8,accumulators),bias)),
    row_writes(~biased,U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,Rows.fourth(~Tile.Tile8,accumulators),bias))

# A job store writes through a view: its stored rows, at most four, by its
# stored lanes, at most eight, one after another in a row. Element k of the
# view is lane k mod lanes of accumulator row k / lanes, with that row's bias
# when biased.
type Block is Data:
  Block{context: Gemm.Context,accumulators: Rows.Rows<Tile.Tile8>,bias: Tree.Tree<F32>}
def block_context(+block: Block) -> Gemm.Context:
  Block{context,accumulators,bias} = block
  context
def block_accumulators(+block: Block) -> Rows.Rows<Tile.Tile8>:
  Block{context,accumulators,bias} = block
  accumulators
def block_bias(+block: Block) -> Tree.Tree<F32>:
  Block{context,accumulators,bias} = block
  bias
def context_row(+context: Gemm.Context) -> U32:
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  row
def context_position(+context: Gemm.Context) -> U32:
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  position
def context_rows(+context: Gemm.Context) -> U32:
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  rows
def context_columns(+context: Gemm.Context) -> U32:
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  columns

# Word base plus offset, written as the store writes the first offsets.
def shifted(+base: U32,offset: Nat) -> U32:
  match offset:
    case 0n: base
    case 1n+rest: (base + U32.from_nat(1n+rest) : U32)
def shifted_exact(+base: U32,+offset: Nat) -> {shifted(base,offset) == (base + U32.from_nat(offset) : U32) : U32}:
  match offset:
    case 0n: Equal.sym(U32,(base + 0 : U32),base,Arithmetic.add_zero(base))
    case 1n+rest: {==}

def store_rows(+context: Gemm.Context) -> U32:
  U32.min(4,(context_rows(context) - context_row(context) : U32))
def store_lanes(+context: Gemm.Context) -> U32:
  U32.min(8,(context_columns(context) - context_position(context) : U32))
def store_target(+context: Gemm.Context) -> U32:
  (context_row(context) * context_columns(context) + context_position(context) : U32)
def store_view(+context: Gemm.Context) -> Views.View:
  Views.View{[Views.Axis{store_rows(context),context_columns(context)},Views.Axis{store_lanes(context),1}],store_target(context)}
def stored_rows(+context: Gemm.Context) -> Nat:
  U32.to_nat(store_rows(context))
def stored_lanes(+context: Gemm.Context) -> Nat:
  U32.to_nat(store_lanes(context))

# The first address a row of the job stores.
def row_offset(+context: Gemm.Context,+row: Nat) -> U32:
  (shifted(context_row(context),row) * context_columns(context) + context_position(context) : U32)
def cell_value(~biased: Bool,+block: Block,+row: Nat,+lane: Nat) -> F32:
  +values = TileLaws.select_lane(lane,Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)))
  Bool.pick(F32,biased,(values + Tree.read(F32,block_bias(block),shifted(context_row(block_context(block)),row)) : F32),values)
def tile_value(~biased: Bool,+block: Block,+position: Nat) -> F32:
  +lanes = stored_lanes(block_context(block))
  cell_value(~biased,block,Nat.div(position,lanes),Nat.mod(position,lanes))

# The first address of a row is the view target plus the row's plane.
def row_offset_exact(+context: Gemm.Context,+row: Nat) ->
  {row_offset(context,row) == (store_target(context) + U32.from_nat(row) * context_columns(context) : U32) : U32}:
  +first = context_row(context)
  +columns = context_columns(context)
  +position = context_position(context)
  +word = U32.from_nat(row)
  +product = (first * columns : U32)
  +plane = (word * columns : U32)
  %Equal.sym(U32,shifted(first,row),(first + word : U32),shifted_exact(first,row)) : {(_ * columns + position : U32) == (product + position + plane : U32) : U32}
  %Equal.sym(U32,((first + word : U32) * columns : U32),(product + plane : U32),Arithmetic.multiply_additive(first,word,columns)) :
    {(_ + position : U32) == (product + position + plane : U32) : U32}
  Equal.trans(U32,(product + plane + position : U32),(product + (plane + position : U32) : U32),(product + position + plane : U32),
    Arithmetic.add_associative(product,plane,position),
    Equal.trans(U32,(product + (plane + position : U32) : U32),(product + (position + plane : U32) : U32),(product + position + plane : U32),
      Equal.cong(U32,U32,tail => (product + tail : U32),(plane + position : U32),(position + plane : U32),Arithmetic.add_commutative(plane,position)),
      Equal.sym(U32,(product + position + plane : U32),(product + (position + plane : U32) : U32),Arithmetic.add_associative(product,position,plane))))

# A stored cell writes the view address and element of its position.
def cell_address(+context: Gemm.Context,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,stored_rows(context)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,stored_lanes(context)) == True{} : Bool}) ->
  {shifted(row_offset(context,row),lane) == Copy.address(store_view(context),Nat.add(Nat.mul(row,stored_lanes(context)),lane)) : U32}:
  +target = store_target(context)
  +columns = context_columns(context)
  +offset = row_offset(context,row)
  +lane_word = U32.from_nat(lane)
  +row_word = U32.from_nat(row)
  Equal.trans(U32,shifted(offset,lane),(offset + lane_word : U32),Copy.address(store_view(context),Nat.add(Nat.mul(row,stored_lanes(context)),lane)),
    shifted_exact(offset,lane),
    Equal.trans(U32,(offset + lane_word : U32),(target + row_word * columns + lane_word * 1 : U32),Copy.address(store_view(context),Nat.add(Nat.mul(row,stored_lanes(context)),lane)),
      %Equal.sym(U32,offset,(target + row_word * columns : U32),row_offset_exact(context,row)) :
        {(_ + lane_word : U32) == (target + row_word * columns + lane_word * 1 : U32) : U32}
      Equal.cong(U32,U32,word => (target + row_word * columns + word : U32),lane_word,(lane_word * 1 : U32),Equal.sym(U32,(lane_word * 1 : U32),lane_word,Arithmetic.multiply_one(lane_word))),
      Equal.sym(U32,Copy.address(store_view(context),Nat.add(Nat.mul(row,stored_lanes(context)),lane)),(target + row_word * columns + lane_word * 1 : U32),
        ViewWrites.plane_address(target,store_rows(context),columns,row,store_lanes(context),1,lane,row_inside,lane_inside))))
def cell_element(~biased: Bool,+block: Block,+row: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,stored_lanes(block_context(block))) == True{} : Bool}) ->
  {cell_value(~biased,block,row,lane) == tile_value(~biased,block,Nat.add(Nat.mul(row,stored_lanes(block_context(block))),lane)) : F32}:
  +lanes = stored_lanes(block_context(block))
  +position = Nat.add(Nat.mul(row,lanes),lane)
  +positive = Order.positive_bound(lane,lanes,lane_inside)
  %Equal.sym(Nat,Nat.div(position,lanes),row,Radix.quotient(row,lane,lanes,positive,lane_inside)) :
    {cell_value(~biased,block,row,lane) == cell_value(~biased,block,_,Nat.mod(position,lanes)) : F32}
  %Equal.sym(Nat,Nat.mod(position,lanes),lane,Radix.remainder(row,lane,lanes,positive,lane_inside)) :
    {cell_value(~biased,block,row,lane) == cell_value(~biased,block,row,_) : F32}
  {==}

# A stored row of a job.
type Line is Data:
  Line{block: Block,row: Nat,inside: {Nat.is_lt(row,stored_rows(block_context(block))) == True{} : Bool}}
def line_block(+line: Line) -> Block:
  Line{block,row,inside} = line
  block
def line_view(+line: Line) -> Views.View:
  store_view(block_context(line_block(line)))
def line_base(+line: Line) -> Nat:
  Line{block,row,inside} = line
  Nat.mul(row,stored_lanes(block_context(block)))
def line_lanes(+line: Line) -> Nat:
  stored_lanes(block_context(line_block(line)))
def single(+line: Line) -> Nat:
  1n
def eight(+line: Line) -> Nat:
  8n
def line_lane(~biased: Bool,+line: Line,+lane: Nat) -> List<&2,Writes.Assignment<F32>>:
  Line{block,row,inside} = line
  +context = block_context(block)
  KernelOutputLaws.lane_writes(U32.is_lt(U32.from_nat(lane),store_lanes(context)),shifted(row_offset(context,row),lane),cell_value(~biased,block,row,lane))

def line_present(~biased: Bool,+line: Line,+lane: Nat,+below: {Nat.is_lt(lane,line_lanes(line)) == True{} : Bool}) ->
  {line_lane(~biased,line,lane) ==
    ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),single(line),Nat.add(line_base(line),Nat.mul(lane,single(line))),line_view(line),line_block(line)) :
    List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+inside} = line
  +context = block_context(block)
  +position = Nat.add(Nat.mul(row,stored_lanes(context)),lane)
  +small = ViewWrites.clipped_small(lane,8,(context_columns(context) - context_position(context) : U32),below)
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(lane),store_lanes(context)),Nat.is_lt(lane,stored_lanes(context)),True{},
    ViewWrites.word_compared(lane,8,store_lanes(context),{==},Bounds.strict_le(lane,8n,small)),below)
  %Equal.sym(Nat,Nat.mul(lane,1n),lane,Order.times_one(lane)) :
    {line_lane(~biased,Line{block,row,inside},lane) ==
      ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),1n,Nat.add(Nat.mul(row,stored_lanes(context)),_),store_view(context),block) : List<&2,Writes.Assignment<F32>>}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),store_lanes(context)),True{},mask) :
    {KernelOutputLaws.lane_writes(_,shifted(row_offset(context,row),lane),cell_value(~biased,block,row,lane)) ==
      [Writes.Put{Copy.address(store_view(context),position),tile_value(~biased,block,position)}] : List<&2,Writes.Assignment<F32>>}
  %cell_address(context,row,lane,inside,below) :
    {[Writes.Put{shifted(row_offset(context,row),lane),cell_value(~biased,block,row,lane)}] == [Writes.Put{_,tile_value(~biased,block,position)}] : List<&2,Writes.Assignment<F32>>}
  %cell_element(~biased,block,row,lane,below) :
    {[Writes.Put{shifted(row_offset(context,row),lane),cell_value(~biased,block,row,lane)}] == [Writes.Put{shifted(row_offset(context,row),lane),_}] : List<&2,Writes.Assignment<F32>>}
  {==}

def line_absent(~biased: Bool,+line: Line,+extra: Nat,+below: {Nat.is_lt(Nat.add(line_lanes(line),extra),eight(line)) == True{} : Bool}) ->
  {line_lane(~biased,line,Nat.add(line_lanes(line),extra)) == Nil{} : List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+inside} = line
  +context = block_context(block)
  +lanes = stored_lanes(context)
  +lane = Nat.add(lanes,extra)
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(lane),store_lanes(context)),Nat.is_lt(lane,lanes),False{},
    ViewWrites.word_compared(lane,8,store_lanes(context),{==},Bounds.strict_le(lane,8n,below)),Order.not_below(lanes,lane,Order.le_add(lanes,extra)))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),store_lanes(context)),False{},mask) :
    {KernelOutputLaws.lane_writes(_,shifted(row_offset(context,row),lane),cell_value(~biased,block,row,lane)) == Nil{} : List<&2,Writes.Assignment<F32>>}
  {==}

# A stored row is the run of the view's writes at the row's positions.
def line_writes(~biased: Bool,+block: Block,+row: Nat,+inside: {Nat.is_lt(row,stored_rows(block_context(block))) == True{} : Bool}) ->
  {row_writes(~biased,True{},shifted(context_row(block_context(block)),row),context_position(block_context(block)),context_columns(block_context(block)),
      Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)),block_bias(block)) ==
    ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),stored_lanes(block_context(block)),Nat.mul(row,stored_lanes(block_context(block))),store_view(block_context(block)),block) :
    List<&2,Writes.Assignment<F32>>}:
  +line = {Line{block,row,inside} : Line}
  +context = block_context(block)
  +lanes = stored_lanes(context)
  %Order.times_one(lanes) :
    {row_writes(~biased,True{},shifted(context_row(context),row),context_position(context),context_columns(context),Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)),block_bias(block)) ==
      ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),_,Nat.mul(row,lanes),store_view(context),block) : List<&2,Writes.Assignment<F32>>}
  ViewWrites.masked(~F32,~Block,~tile_value(~biased),~Line,~line_lane(~biased),~line_view,~line_block,~line_base,~single,~line_lanes,~eight,~line_present(~biased),~line_absent(~biased),line,
    ViewWrites.limited(8,(context_columns(context) - context_position(context) : U32)))

# A job store whose first row lies within its rows, below a word capacity
# with room for four rows.
type Placed is Data:
  Placed{block: Block,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    ordered: {Nat.is_le(U32.to_nat(context_row(block_context(block))),U32.to_nat(context_rows(block_context(block)))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(context_rows(block_context(block))),3n),U32.to_nat(capacity)) == True{} : Bool}}
def placed_block(+placed: Placed) -> Block:
  Placed{block,capacity,capacity_ok,ordered,room} = placed
  block
def placed_view(+placed: Placed) -> Views.View:
  store_view(block_context(placed_block(placed)))
def start(+placed: Placed) -> Nat:
  0n
def placed_lanes(+placed: Placed) -> Nat:
  stored_lanes(block_context(placed_block(placed)))
def placed_rows(+placed: Placed) -> Nat:
  stored_rows(block_context(placed_block(placed)))
def four(+placed: Placed) -> Nat:
  4n
def row_part(~biased: Bool,+placed: Placed,+row: Nat) -> List<&2,Writes.Assignment<F32>>:
  +block = placed_block(placed)
  +context = block_context(block)
  +first = shifted(context_row(context),row)
  row_writes(~biased,U32.is_lt(first,context_rows(context)),first,context_position(context),context_columns(context),
    Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)),block_bias(block))

# The row mask of a placed job: row r is stored exactly when it is below the
# rows left after the first.
def row_mask(+placed: Placed,+row: Nat,+small: {Nat.is_lt(row,4n) == True{} : Bool}) ->
  {U32.is_lt(shifted(context_row(block_context(placed_block(placed))),row),context_rows(block_context(placed_block(placed)))) ==
    Nat.is_lt(row,U32.to_nat((context_rows(block_context(placed_block(placed))) - context_row(block_context(placed_block(placed))) : U32))) : Bool}:
  Placed{+block,+capacity,+capacity_ok,+ordered,+room} = placed
  +context = block_context(block)
  +first = context_row(context)
  +rows = context_rows(context)
  +total = U32.to_nat(rows)
  +three = {Order.strict_follow(row,4n,small) : {Nat.is_le(row,3n) == True{} : Bool}}
  +row_room = Order.transitive(Nat.add(U32.to_nat(first),row),Nat.add(total,3n),U32.to_nat(capacity),
    Order.transitive(Nat.add(U32.to_nat(first),row),Nat.add(total,row),Nat.add(total,3n),Order.add_le(U32.to_nat(first),total,row,ordered),
      Equal.trans(Bool,Nat.is_le(Nat.add(total,row),Nat.add(total,3n)),Nat.is_le(row,3n),True{},Order.le_offset(total,row,3n),three)),
    room)
  %Equal.sym(U32,shifted(first,row),(first + U32.from_nat(row) : U32),shifted_exact(first,row)) :
    {U32.is_lt(_,rows) == Nat.is_lt(row,U32.to_nat((rows - first : U32))) : Bool}
  ViewWrites.offset_compared(first,row,rows,capacity,capacity_ok,ordered,row_room,
    Order.transitive(total,Nat.add(total,3n),U32.to_nat(capacity),Order.le_add(total,3n),room))

def row_present(~biased: Bool,+placed: Placed,+row: Nat,+below: {Nat.is_lt(row,placed_rows(placed)) == True{} : Bool}) ->
  {row_part(~biased,placed,row) ==
    ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),placed_lanes(placed),Nat.add(start(placed),Nat.mul(row,placed_lanes(placed))),placed_view(placed),placed_block(placed)) :
    List<&2,Writes.Assignment<F32>>}:
  +block = placed_block(placed)
  +context = block_context(block)
  +first = shifted(context_row(context),row)
  +left = (context_rows(context) - context_row(context) : U32)
  +small = ViewWrites.clipped_small(row,4,left,below)
  +mask = Equal.trans(Bool,U32.is_lt(first,context_rows(context)),Nat.is_lt(row,U32.to_nat(left)),True{},row_mask(placed,row,small),ViewWrites.clipped_below(row,4,left,below))
  %Equal.sym(Bool,U32.is_lt(first,context_rows(context)),True{},mask) :
    {row_writes(~biased,_,first,context_position(context),context_columns(context),Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)),block_bias(block)) ==
      ViewWrites.writes_from(~F32,~Block,~tile_value(~biased),stored_lanes(context),Nat.mul(row,stored_lanes(context)),store_view(context),block) : List<&2,Writes.Assignment<F32>>}
  line_writes(~biased,block,row,below)

def row_absent(~biased: Bool,+placed: Placed,+extra: Nat,+below: {Nat.is_lt(Nat.add(placed_rows(placed),extra),four(placed)) == True{} : Bool}) ->
  {row_part(~biased,placed,Nat.add(placed_rows(placed),extra)) == Nil{} : List<&2,Writes.Assignment<F32>>}:
  +block = placed_block(placed)
  +context = block_context(block)
  +used = placed_rows(placed)
  +row = Nat.add(used,extra)
  +first = shifted(context_row(context),row)
  +left = (context_rows(context) - context_row(context) : U32)
  +mask = Equal.trans(Bool,U32.is_lt(first,context_rows(context)),Nat.is_lt(row,U32.to_nat(left)),False{},row_mask(placed,row,below),
    ViewWrites.clipped_beyond(row,4,left,below,Order.not_below(used,row,Order.le_add(used,extra))))
  %Equal.sym(Bool,U32.is_lt(first,context_rows(context)),False{},mask) :
    {row_writes(~biased,_,first,context_position(context),context_columns(context),Accumulator.select_row(~Tile.Tile8,row,block_accumulators(block)),block_bias(block)) ==
      Nil{} : List<&2,Writes.Assignment<F32>>}
  {==}

# A job store is the writes of its view.
def view_writes(~biased: Bool,+placed: Placed) ->
  {context_writes(~biased,block_context(placed_block(placed)),block_accumulators(placed_block(placed)),block_bias(placed_block(placed))) ==
    ViewWrites.writes(~F32,~Block,~tile_value(~biased),Nat.mul(placed_rows(placed),placed_lanes(placed)),placed_view(placed),placed_block(placed)) :
    List<&2,Writes.Assignment<F32>>}:
  Placed{Block{Gemm.Context{+row,+position,+rows,+columns,+input_base,+weight_base},+accumulators,+bias},+capacity,+capacity_ok,+ordered,+room} = placed
  ViewWrites.masked(~F32,~Block,~tile_value(~biased),~Placed,~row_part(~biased),~placed_view,~placed_block,~start,~placed_lanes,~placed_rows,~four,~row_present(~biased),~row_absent(~biased),
    Placed{Block{Gemm.Context{row,position,rows,columns,input_base,weight_base},accumulators,bias},capacity,capacity_ok,ordered,room},
    ViewWrites.limited(4,(rows - row : U32)))

def owner_bias(owners: KernelOutputLaws.Owners) -> Tree.Tree<F32>:
  KernelOutputLaws.Owners{bias,output} = owners
  bias

def owner_output(owners: KernelOutputLaws.Owners) -> Tree.Tree<F32>:
  KernelOutputLaws.Owners{bias,output} = owners
  Writes.evaluate(F32,output)

def row_bias(~biased: Bool,+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners) ->
  {owner_bias(KernelOutputLaws.row(biased,valid,index,position,columns,accumulator,owners)) == owner_bias(owners) : Tree.Tree<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} KernelOutputLaws.Owners{bias,output}: {==}

def row_output(~biased: Bool,+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners) ->
  {owner_output(KernelOutputLaws.row(biased,valid,index,position,columns,accumulator,owners)) ==
    Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)) : Tree.Tree<F32>}:
  match valid owners:
    case False{} owners: {==}
    case True{} KernelOutputLaws.Owners{bias,output}: {==}

def Observed(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>) -> Data:
  Shape.Both<{owner_bias(owners) == bias : Tree.Tree<F32>},{owner_output(owners) == Writes.pure(F32,writes,tree) : Tree.Tree<F32>}>

def appended_output(~biased: Bool,+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners,
  +bias: Tree.Tree<F32>,+previous: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  same_bias: {owner_bias(owners) == bias : Tree.Tree<F32>},same_output: {owner_output(owners) == Writes.pure(F32,previous,tree) : Tree.Tree<F32>}) ->
  {Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)) ==
    Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}:
  %Equal.sym(Tree.Tree<F32>,owner_bias(owners),bias,same_bias) :
    {Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,_),owner_output(owners)) ==
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,owner_output(owners),Writes.pure(F32,previous,tree),same_output) :
    {Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,bias),_) ==
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree) : Tree.Tree<F32>}
  Equal.sym(Tree.Tree<F32>,
    Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree),
    Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,bias),Writes.pure(F32,previous,tree)),
    Writes.concatenated_writes(F32,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias),tree))

def next_row(~biased: Bool,+valid: Bool,+index: U32,+position: U32,+columns: U32,+accumulator: Tile.Tile8,+owners: KernelOutputLaws.Owners,
  +bias: Tree.Tree<F32>,+previous: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,observed: Observed(owners,bias,previous,tree)) ->
  Observed(KernelOutputLaws.row(biased,valid,index,position,columns,accumulator,owners),bias,
    List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree):
  Shape.Both{+same_bias,+same_output} = observed
  Shape.Both{
    Equal.trans(Tree.Tree<F32>,owner_bias(KernelOutputLaws.row(biased,valid,index,position,columns,accumulator,owners)),owner_bias(owners),bias,
      row_bias(~biased,valid,index,position,columns,accumulator,owners),same_bias),
    Equal.trans(Tree.Tree<F32>,owner_output(KernelOutputLaws.row(biased,valid,index,position,columns,accumulator,owners)),
      Writes.pure(F32,row_writes(~biased,valid,index,position,columns,accumulator,owner_bias(owners)),owner_output(owners)),
      Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,previous,row_writes(~biased,valid,index,position,columns,accumulator,bias)),tree),
      row_output(~biased,valid,index,position,columns,accumulator,owners),
      appended_output(~biased,valid,index,position,columns,accumulator,owners,bias,previous,tree,same_bias,same_output))}

def write_observed(~biased: Bool,+context: Gemm.Context,+accumulators: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>) ->
  Observed(KernelOutputLaws.write(biased,context,accumulators,bias,output),bias,context_writes(~biased,context,accumulators,bias),Writes.evaluate(F32,output)):
  Gemm.Context{+index,+position,+rows,+columns,input_base,weight_base} = context
  +tree = Writes.evaluate(F32,output)
  +a = Rows.first(~Tile.Tile8,accumulators)
  +b = Rows.second(~Tile.Tile8,accumulators)
  +c = Rows.third(~Tile.Tile8,accumulators)
  +d = Rows.fourth(~Tile.Tile8,accumulators)
  +w0 = row_writes(~biased,U32.is_lt(index,rows),index,position,columns,a,bias)
  +w1 = row_writes(~biased,U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,bias)
  +w2 = row_writes(~biased,U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,bias)
  +initial = {KernelOutputLaws.Owners{bias,output} : KernelOutputLaws.Owners}
  +first = KernelOutputLaws.row(biased,U32.is_lt(index,rows),index,position,columns,a,initial)
  +second = KernelOutputLaws.row(biased,U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,first)
  +third = KernelOutputLaws.row(biased,U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,second)
  +two = List.append(&2,Writes.Assignment<F32>,w0,w1)
  +three = List.append(&2,Writes.Assignment<F32>,two,w2)
  +start = {Shape.Both{{==},{==}} : Observed(initial,bias,[],tree)}
  +after_first = next_row(~biased,U32.is_lt(index,rows),index,position,columns,a,initial,bias,[],tree,start)
  next_row(~biased,U32.is_lt((index + 3 : U32),rows),(index + 3 : U32),position,columns,d,third,bias,three,tree,
    next_row(~biased,U32.is_lt((index + 2 : U32),rows),(index + 2 : U32),position,columns,c,second,bias,two,tree,
      next_row(~biased,U32.is_lt((index + 1 : U32),rows),(index + 1 : U32),position,columns,b,first,bias,w0,tree,after_first)))

def observed_bias(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  observed: Observed(owners,bias,writes,tree)) -> {owner_bias(owners) == bias : Tree.Tree<F32>}:
  Shape.Both{same_bias,same_output} = observed
  same_bias

def observed_output(+owners: KernelOutputLaws.Owners,+bias: Tree.Tree<F32>,+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,
  observed: Observed(owners,bias,writes,tree)) -> {owner_output(owners) == Writes.pure(F32,writes,tree) : Tree.Tree<F32>}:
  Shape.Both{same_bias,same_output} = observed
  same_output

def state_input(state: ReductionModel.State()) -> Tree.Tree<Tile.Tile8>:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  input

def state_weights(state: ReductionModel.State()) -> Tree.Tree<Tile.Tile8>:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  weights

def source_input(source: ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>) -> Tree.Tree<Tile.Tile8>:
  ReductionLaws.SampleSource{input,weights,context} = source
  input

def source_weights(source: ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>) -> Tree.Tree<Tile.Tile8>:
  ReductionLaws.SampleSource{input,weights,context} = source
  weights

def state_source(state: ReductionModel.State()) -> ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>:
  ReductionLaws.source(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<Tile.Tile8>,state)

def state_source_input(+state: ReductionModel.State()) -> {state_input(state) == source_input(state_source(state)) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  {==}

def state_source_weights(+state: ReductionModel.State()) -> {state_weights(state) == source_weights(state_source(state)) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{input,weights,input_index,weight_index,base,accumulators} = state
  {==}

def selected_source(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+input_index: U32,+weight_index: U32,+base: U32) ->
  {state_source(ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_index,weight_index,base))) == ReductionLaws.SampleSource{input,weights,base} :
    ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>}:
  match upper:
    case False{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~False{}),~ReductionModel.update,count,ReductionModel.initial(input,weights,input_index,weight_index,base))
    case True{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~True{}),~ReductionModel.update,count,ReductionModel.initial(input,weights,input_index,weight_index,base))

def selected_with_source(upper: Bool,+count: Nat,+state: ReductionModel.State()) ->
  {state_source(ReductionModel.selected(upper,count,state)) == state_source(state) : ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>}:
  match upper:
    case False{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~False{}),~ReductionModel.update,count,state)
    case True{}:
      ReductionLaws.loop_preserves_source(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
        ~ReductionModel.input_tile,~ReductionModel.weights_half(~True{}),~ReductionModel.update,count,state)

def reduce_input(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+context: Gemm.Context) ->
  {state_input(GemmLaws.reduce(upper,count,input,weights,context)) == input : Tree.Tree<Tile.Tile8>}:
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  +state = ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))
  Equal.trans(Tree.Tree<Tile.Tile8>,state_input(state),source_input(state_source(state)),input,state_source_input(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source => source_input(source),state_source(state),
      ReductionLaws.SampleSource{input,weights,weight_base},selected_source(upper,count,input,weights,input_base,0,weight_base)))

def reduce_weights(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+context: Gemm.Context) ->
  {state_weights(GemmLaws.reduce(upper,count,input,weights,context)) == weights : Tree.Tree<Tile.Tile8>}:
  Gemm.Context{row,position,rows,columns,+input_base,+weight_base} = context
  +state = ReductionModel.selected(upper,count,ReductionModel.initial(input,weights,input_base,0,weight_base))
  Equal.trans(Tree.Tree<Tile.Tile8>,state_weights(state),source_weights(state_source(state)),weights,state_source_weights(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source => source_weights(source),state_source(state),
      ReductionLaws.SampleSource{input,weights,weight_base},selected_source(upper,count,input,weights,input_base,0,weight_base)))

def buffers_input(buffers: GemmLaws.Buffers) -> Tree.Tree<Tile.Tile8>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  input

def buffers_weights(buffers: GemmLaws.Buffers) -> Tree.Tree<Tile.Tile8>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  weights

def buffers_bias(buffers: GemmLaws.Buffers) -> Tree.Tree<F32>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  bias

def buffers_program(buffers: GemmLaws.Buffers) -> Writes.Program<F32>:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  output

def buffers_eta(+buffers: GemmLaws.Buffers) ->
  {buffers == GemmLaws.Buffers{buffers_input(buffers),buffers_weights(buffers),buffers_bias(buffers),buffers_program(buffers)} : GemmLaws.Buffers}:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  {==}

def job_writes(~biased: Bool,job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  Gemm.Job{+context,upper,count} = job
  context_writes(~biased,context,
    Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
      GemmLaws.reduce(upper,count,input,weights,context)),bias)

def written_input(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_input(GemmLaws.after_write(input,weights,owners)) == input : Tree.Tree<Tile.Tile8>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_weights(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_weights(GemmLaws.after_write(input,weights,owners)) == weights : Tree.Tree<Tile.Tile8>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_bias(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {buffers_bias(GemmLaws.after_write(input,weights,owners)) == owner_bias(owners) : Tree.Tree<F32>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def written_output(+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+owners: KernelOutputLaws.Owners) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.after_write(input,weights,owners))) == owner_output(owners) : Tree.Tree<F32>}:
  KernelOutputLaws.Owners{bias,output} = owners
  {==}

def reduced_input(~biased: Bool,+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_input(GemmLaws.finish(~biased,bias,output,context,state)) == state_input(state) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  written_input(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))

def reduced_weights(~biased: Bool,+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_weights(GemmLaws.finish(~biased,bias,output,context,state)) == state_weights(state) : Tree.Tree<Tile.Tile8>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  written_weights(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))

def reduced_bias(~biased: Bool,+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {buffers_bias(GemmLaws.finish(~biased,bias,output,context,state)) == bias : Tree.Tree<F32>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  Equal.trans(Tree.Tree<F32>,buffers_bias(GemmLaws.after_write(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output))),
    owner_bias(KernelOutputLaws.write(biased,context,accumulators,bias,output)),bias,
    written_bias(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output)),
    observed_bias(KernelOutputLaws.write(biased,context,accumulators,bias,output),bias,context_writes(~biased,context,accumulators,bias),Writes.evaluate(F32,output),
      write_observed(~biased,context,accumulators,bias,output)))

def reduced_output(~biased: Bool,+context: Gemm.Context,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+state: ReductionModel.State()) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.finish(~biased,bias,output,context,state))) ==
    Writes.pure(F32,context_writes(~biased,context,
      Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,state),bias),Writes.evaluate(F32,output)) : Tree.Tree<F32>}:
  Reduction.State{+input,+weights,input_index,weight_index,base,+accumulators} = state
  Equal.trans(Tree.Tree<F32>,Writes.evaluate(F32,buffers_program(GemmLaws.after_write(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output)))),
    owner_output(KernelOutputLaws.write(biased,context,accumulators,bias,output)),
    Writes.pure(F32,context_writes(~biased,context,accumulators,bias),Writes.evaluate(F32,output)),
    written_output(input,weights,KernelOutputLaws.write(biased,context,accumulators,bias,output)),
    observed_output(KernelOutputLaws.write(biased,context,accumulators,bias,output),bias,context_writes(~biased,context,accumulators,bias),Writes.evaluate(F32,output),
      write_observed(~biased,context,accumulators,bias,output)))

# One job leaves every source owner unchanged and appends exactly its writes.
def executed(~biased: Bool,+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {GemmLaws.execute(~biased,job,GemmLaws.Buffers{input,weights,bias,program}) ==
    GemmLaws.Buffers{input,weights,bias,buffers_program(GemmLaws.execute(~biased,job,GemmLaws.Buffers{input,weights,bias,program}))} : GemmLaws.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  +state = GemmLaws.reduce(upper,count,input,weights,context)
  +result = GemmLaws.finish(~biased,bias,program,context,state)
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_input(result),state_input(state),input,
    reduced_input(~biased,context,bias,program,state),reduce_input(upper,count,input,weights,context)) :
    {result == GemmLaws.Buffers{_,weights,bias,buffers_program(result)} : GemmLaws.Buffers}
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_weights(result),state_weights(state),weights,
    reduced_weights(~biased,context,bias,program,state),reduce_weights(upper,count,input,weights,context)) :
    {result == GemmLaws.Buffers{buffers_input(result),_,bias,buffers_program(result)} : GemmLaws.Buffers}
  %reduced_bias(~biased,context,bias,program,state) :
    {result == GemmLaws.Buffers{buffers_input(result),buffers_weights(result),_,buffers_program(result)} : GemmLaws.Buffers}
  buffers_eta(result)

def executed_output(~biased: Bool,+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.execute(~biased,job,GemmLaws.Buffers{input,weights,bias,program}))) ==
    Writes.pure(F32,job_writes(~biased,job,input,weights,bias),Writes.evaluate(F32,program)) : Tree.Tree<F32>}:
  Gemm.Job{+context,+upper,+count} = job
  reduced_output(~biased,context,bias,program,GemmLaws.reduce(upper,count,input,weights,context))

def reduce_with(job: Gemm.Job,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,initial: Rows.Rows<Tile.Tile8>) -> ReductionModel.State():
  Gemm.Job{Gemm.Context{row,position,rows,columns,input_base,weight_base},upper,count} = job
  ReductionModel.selected(upper,count,Reduction.State{input,weights,input_base,0,weight_base,initial})

def job_context(job: Gemm.Job) -> Gemm.Context:
  Gemm.Job{context,upper,count} = job
  context

def job_writes_with(~biased: Bool,+job: Gemm.Job,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,initial: Rows.Rows<Tile.Tile8>) -> List<&2,Writes.Assignment<F32>>:
  context_writes(~biased,job_context(job),Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,reduce_with(job,input,weights,initial)),bias)

def job_weight_base(job: Gemm.Job) -> U32:
  Gemm.Job{Gemm.Context{row,position,rows,columns,input_base,weight_base},upper,count} = job
  weight_base

def reduce_with_source(+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {state_source(reduce_with(job,input,weights,initial)) == ReductionLaws.SampleSource{input,weights,job_weight_base(job)} : ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>}:
  Gemm.Job{Gemm.Context{row,position,rows,columns,input_base,+weight_base},+upper,+count} = job
  selected_with_source(upper,count,Reduction.State{input,weights,input_base,0,weight_base,initial})

def reduce_with_input(+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {state_input(reduce_with(job,input,weights,initial)) == input : Tree.Tree<Tile.Tile8>}:
  +state = reduce_with(job,input,weights,initial)
  Equal.trans(Tree.Tree<Tile.Tile8>,state_input(state),source_input(state_source(state)),input,state_source_input(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source_input,state_source(state),ReductionLaws.SampleSource{input,weights,job_weight_base(job)},reduce_with_source(job,input,weights,initial)))

def reduce_with_weights(+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+initial: Rows.Rows<Tile.Tile8>) ->
  {state_weights(reduce_with(job,input,weights,initial)) == weights : Tree.Tree<Tile.Tile8>}:
  +state = reduce_with(job,input,weights,initial)
  Equal.trans(Tree.Tree<Tile.Tile8>,state_weights(state),source_weights(state_source(state)),weights,state_source_weights(state),
    Equal.cong(ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>,Tree.Tree<Tile.Tile8>,source_weights,state_source(state),ReductionLaws.SampleSource{input,weights,job_weight_base(job)},reduce_with_source(job,input,weights,initial)))

def executed_with(~biased: Bool,+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>,+initial: Rows.Rows<Tile.Tile8>) ->
  {GemmLaws.execute_with(~biased,job,GemmLaws.Buffers{input,weights,bias,program},initial) ==
    GemmLaws.Buffers{input,weights,bias,buffers_program(GemmLaws.execute_with(~biased,job,GemmLaws.Buffers{input,weights,bias,program},initial))} : GemmLaws.Buffers}:
  Gemm.Job{+context,+upper,+count} = job
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  +state = reduce_with(job,input,weights,initial)
  +result = GemmLaws.finish(~biased,bias,program,context,state)
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_input(result),state_input(state),input,reduced_input(~biased,context,bias,program,state),reduce_with_input(job,input,weights,initial)) :
    {result == GemmLaws.Buffers{_,weights,bias,buffers_program(result)} : GemmLaws.Buffers}
  %Equal.trans(Tree.Tree<Tile.Tile8>,buffers_weights(result),state_weights(state),weights,reduced_weights(~biased,context,bias,program,state),reduce_with_weights(job,input,weights,initial)) :
    {result == GemmLaws.Buffers{buffers_input(result),_,bias,buffers_program(result)} : GemmLaws.Buffers}
  %reduced_bias(~biased,context,bias,program,state) :
    {result == GemmLaws.Buffers{buffers_input(result),buffers_weights(result),_,buffers_program(result)} : GemmLaws.Buffers}
  buffers_eta(result)

def executed_with_output(~biased: Bool,+job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+program: Writes.Program<F32>,+initial: Rows.Rows<Tile.Tile8>) ->
  {Writes.evaluate(F32,buffers_program(GemmLaws.execute_with(~biased,job,GemmLaws.Buffers{input,weights,bias,program},initial))) ==
    Writes.pure(F32,job_writes_with(~biased,job,input,weights,bias,initial),Writes.evaluate(F32,program)) : Tree.Tree<F32>}:
  Gemm.Job{+context,+upper,+count} = job
  Gemm.Context{row,position,rows,columns,input_base,weight_base} = context
  reduced_output(~biased,context,bias,program,reduce_with(job,input,weights,initial))

def half_context(upper: Bool) -> Gemm.Context:
  Gemm.Context{Bool.pick(U32,upper,4,0),0,8,8,0,0}

def half_writes(+upper: Bool,values: Rows.Rows<Tile.Tile8>,bias: Tree.Tree<F32>) -> List<&2,Writes.Assignment<F32>>:
  context_writes(~False{},half_context(upper),values,bias)

def lane_room(+lane: Nat,bound: {Nat.is_le(lane,7n) == True{} : Bool}) -> {Nat.is_le(Nat.add(56n,lane),63n) == True{} : Bool}:
  %Order.commutative(lane,56n) : {Nat.is_le(_,63n) == True{} : Bool}
  Order.add_le(lane,7n,56n,bound)

def lane_inside(+offset: U32,+lane: U32,+base_bound: {Nat.is_le(U32.to_nat(offset),56n) == True{} : Bool},+lane_bound: {Nat.is_le(U32.to_nat(lane),7n) == True{} : Bool}) ->
  {U32.is_lt((offset + lane : U32),64) == True{} : Bool}:
  +first = Order.add_le(U32.to_nat(offset),56n,U32.to_nat(lane),base_bound)
  +second = lane_room(U32.to_nat(lane),lane_bound)
  +room = Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(lane)),Nat.add(56n,U32.to_nat(lane)),63n,first,second)
  %Equal.sym(Bool,U32.is_lt((offset + lane : U32),64),Nat.is_lt(U32.to_nat((offset + lane : U32)),64n),Comparison.lt((offset + lane : U32),64)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((offset + lane : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(lane)),Arithmetic.add(offset,lane,64,{==},Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(lane)),63n,64n,room,{==}))) : {Nat.is_lt(_,64n) == True{} : Bool}
  Order.le_lt_trans(Nat.add(U32.to_nat(offset),U32.to_nat(lane)),63n,64n,room,{==})

def full_row_inside(+offset: U32,+values: Tile.Tile8,+room: {Nat.is_le(U32.to_nat(offset),56n) == True{} : Bool}) -> ProgramRead.Inside(~F32,64,KernelOutputLaws.tail_assignments(False{},8,offset,0.0,values)):
  Shape.Both{
    %Equal.sym(Bool,U32.is_lt(offset,64),Nat.is_lt(U32.to_nat(offset),64n),Comparison.lt(offset,64)) : {_ == True{} : Bool}
    Order.le_lt_trans(U32.to_nat(offset),56n,64n,room,{==}),
    Shape.Both{lane_inside(offset,1,room,{==}),Shape.Both{lane_inside(offset,2,room,{==}),Shape.Both{lane_inside(offset,3,room,{==}),Shape.Both{lane_inside(offset,4,room,{==}),Shape.Both{lane_inside(offset,5,room,{==}),Shape.Both{lane_inside(offset,6,room,{==}),Shape.Both{lane_inside(offset,7,room,{==}),Unit{}}}}}}}}}

def four_writes(+offsets: Rows.Rows<U32>,values: Rows.Rows<Tile.Tile8>) -> List<&2,Writes.Assignment<F32>>:
  Rows.Rows{pa,pb,pc,pd} = offsets
  Rows.Rows{a,b,c,d} = values
  List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,KernelOutputLaws.tail_assignments(False{},8,pa,0.0,a),KernelOutputLaws.tail_assignments(False{},8,pb,0.0,b)),KernelOutputLaws.tail_assignments(False{},8,pc,0.0,c)),KernelOutputLaws.tail_assignments(False{},8,pd,0.0,d))

def four_inside(+offsets: Rows.Rows<U32>,+values: Rows.Rows<Tile.Tile8>,
  a_room: {Nat.is_le(U32.to_nat(Rows.first(~U32,offsets)),56n) == True{} : Bool},
  b_room: {Nat.is_le(U32.to_nat(Rows.second(~U32,offsets)),56n) == True{} : Bool},
  c_room: {Nat.is_le(U32.to_nat(Rows.third(~U32,offsets)),56n) == True{} : Bool},
  d_room: {Nat.is_le(U32.to_nat(Rows.fourth(~U32,offsets)),56n) == True{} : Bool}) -> ProgramRead.Inside(~F32,64,four_writes(offsets,values)):
  Rows.Rows{+pa,+pb,+pc,+pd} = offsets
  Rows.Rows{+a,+b,+c,+d} = values
  +wa = KernelOutputLaws.tail_assignments(False{},8,pa,0.0,a)
  +wb = KernelOutputLaws.tail_assignments(False{},8,pb,0.0,b)
  +wc = KernelOutputLaws.tail_assignments(False{},8,pc,0.0,c)
  +wd = KernelOutputLaws.tail_assignments(False{},8,pd,0.0,d)
  ProgramRead.inside_append(~F32,64,List.append(&2,Writes.Assignment<F32>,List.append(&2,Writes.Assignment<F32>,wa,wb),wc),wd,
    ProgramRead.inside_append(~F32,64,List.append(&2,Writes.Assignment<F32>,wa,wb),wc,
      ProgramRead.inside_append(~F32,64,wa,wb,full_row_inside(pa,a,a_room),full_row_inside(pb,b,b_room)),full_row_inside(pc,c,c_room)),full_row_inside(pd,d,d_room))

def half_offsets(upper: Bool) -> Rows.Rows<U32>:
  match upper:
    case False{}: Rows.Rows{0,8,16,24}
    case True{}: Rows.Rows{32,40,48,56}

def half_shape(upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {half_writes(upper,values,bias) == four_writes(half_offsets(upper),values) : List<&2,Writes.Assignment<F32>>}:
  match upper values:
    case False{} Rows.Rows{a,b,c,d}: {==}
    case True{} Rows.Rows{a,b,c,d}: {==}

def half_inside(upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>) -> ProgramRead.Inside(~F32,64,half_writes(upper,values,bias)):
  match upper:
    case False{}:
      %Equal.sym(List<&2,Writes.Assignment<F32>>,half_writes(False{},values,bias),four_writes(half_offsets(False{}),values),half_shape(False{},values,bias)) : ProgramRead.Inside(~F32,64,_)
      four_inside(Rows.Rows{0,8,16,24},values,{==},{==},{==},{==})
    case True{}:
      %Equal.sym(List<&2,Writes.Assignment<F32>>,half_writes(True{},values,bias),four_writes(half_offsets(True{}),values),half_shape(True{},values,bias)) : ProgramRead.Inside(~F32,64,_)
      four_inside(Rows.Rows{32,40,48,56},values,{==},{==},{==},{==})

def half_readback(+upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+query: U32,
  +balanced: Shape.Balanced(F32,6n,output),inside: {U32.is_lt(query,64) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,half_writes(upper,values,bias),output),query) ==
    ProgramRead.selected(~F32,half_writes(upper,values,bias),query,Tree.read(F32,output,query)) : F32}:
  ProgramRead.program(~F32,half_writes(upper,values,bias),6n,output,query,balanced,{==},
    %Equal.sym(U32,Tree.capacity(F32,output),64,Shape.capacity(F32,6n,output,balanced)) : {U32.is_lt(query,_) == True{} : Bool} inside,
    %Equal.sym(U32,Tree.capacity(F32,output),64,Shape.capacity(F32,6n,output,balanced)) : ProgramRead.Inside(~F32,_,half_writes(upper,values,bias)) half_inside(upper,values,bias))

def row_number(row: Accumulator.Row) -> U32:
  match row:
    case Accumulator.R0{}: 0
    case Accumulator.R1{}: 1
    case Accumulator.R2{}: 2
    case Accumulator.R3{}: 3

def half_address(+upper: Bool,row: Accumulator.Row,lane: Nat) -> U32:
  (Bool.pick(U32,upper,32,0) + row_number(row) * 8 + U32.from_nat(lane) : U32)

# These are the finite hardware rows and lanes, independently of problem size.
def half_selected(upper: Bool,row: Accumulator.Row,+lane: Nat,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+previous: F32,within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {ProgramRead.selected(~F32,half_writes(upper,values,bias),half_address(upper,row,lane),previous) == TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) : F32}:
  match upper row lane values:
    case False{} Accumulator.R0{} 0n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 1n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 2n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 3n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 4n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 5n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 6n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R0{} 7n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 0n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 1n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 2n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 3n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 4n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 5n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 6n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R1{} 7n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 0n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 1n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 2n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 3n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 4n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 5n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 6n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R2{} 7n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 0n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 1n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 2n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 3n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 4n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 5n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 6n Rows.Rows{a,b,c,d}: {==}
    case False{} Accumulator.R3{} 7n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 0n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 1n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 2n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 3n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 4n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 5n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 6n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R0{} 7n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 0n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 1n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 2n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 3n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 4n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 5n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 6n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R1{} 7n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 0n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 1n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 2n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 3n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 4n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 5n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 6n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R2{} 7n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 0n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 1n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 2n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 3n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 4n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 5n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 6n Rows.Rows{a,b,c,d}: {==}
    case True{} Accumulator.R3{} 7n Rows.Rows{a,b,c,d}: {==}
    case upper row 8n+ +rest values:
      Empty.absurd({ProgramRead.selected(~F32,half_writes(upper,values,bias),half_address(upper,row,lane),previous) == TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) : F32},
        Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),within)))

def rows_values(values: Rows.Rows<Tile.Tile8>) -> List<&2,F32>:
  Rows.Rows{a,b,c,d} = values
  List.append(&2,F32,Lanes.tile_values(a),List.append(&2,F32,Lanes.tile_values(b),List.append(&2,F32,Lanes.tile_values(c),Lanes.tile_values(d))))

def rows_length(values: Rows.Rows<Tile.Tile8>) -> {List.length(&2,F32,rows_values(values)) == 32n : Nat}:
  Rows.Rows{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7}} = values
  {==}

def half_sequence(upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {half_writes(upper,values,bias) == Writes.sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0)) : List<&2,Writes.Assignment<F32>>}:
  match upper values:
    case False{} Rows.Rows{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7}}: {==}
    case True{} Rows.Rows{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7}}: {==}

def half_written_list(+upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+tree: Tree.Tree<F32>) ->
  {Writes.pure(F32,half_writes(upper,values,bias),tree) == Tree.write_sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0),tree) : Tree.Tree<F32>}:
  Equal.trans(Tree.Tree<F32>,Writes.pure(F32,half_writes(upper,values,bias),tree),Writes.pure(F32,Writes.sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0)),tree),Tree.write_sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0),tree),
    Equal.cong(List<&2,Writes.Assignment<F32>>,Tree.Tree<F32>,writes => Writes.pure(F32,writes,tree),half_writes(upper,values,bias),Writes.sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0)),half_sequence(upper,values,bias)),Writes.sequence_refines(~F32,rows_values(values),Bool.pick(U32,upper,32,0),tree))

def half_room(upper: Bool,+values: Rows.Rows<Tile.Tile8>) ->
  {Nat.is_le(Nat.add(Bool.pick(Nat,upper,32n,0n),List.length(&2,F32,rows_values(values))),64n) == True{} : Bool}:
  match upper:
    case False{}:
      %Equal.sym(Nat,List.length(&2,F32,rows_values(values)),32n,rows_length(values)) : {Nat.is_le(Nat.add(0n,_),64n) == True{} : Bool}
      {==}
    case True{}:
      %Equal.sym(Nat,List.length(&2,F32,rows_values(values)),32n,rows_length(values)) : {Nat.is_le(Nat.add(32n,_),64n) == True{} : Bool}
      {==}

def rows_list_preserved(+offset: Nat,+values: Rows.Rows<Tile.Tile8>,+output: Tree.Tree<F32>,+query: Nat,
  +balanced: Shape.Balanced(F32,6n,output),inside: {Nat.is_lt(query,64n) == True{} : Bool},
  room: {Nat.is_le(Nat.add(offset,List.length(&2,F32,rows_values(values))),64n) == True{} : Bool},
  separate: Interval.Outside<offset,32n,query>) ->
  {Tree.read(F32,Tree.write_sequence(~F32,rows_values(values),U32.from_nat(offset),output),U32.from_nat(query)) == Tree.read(F32,output,U32.from_nat(query)) : F32}:
  ListRead.outside(~F32,rows_values(values),offset,query,6n,output,balanced,{==},
    %Equal.sym(U32,Tree.capacity(F32,output),64,Shape.capacity(F32,6n,output,balanced)) : {Nat.is_le(Nat.add(offset,List.length(&2,F32,rows_values(values))),U32.to_nat(_)) == True{} : Bool} room,
    %Equal.sym(U32,Tree.capacity(F32,output),64,Shape.capacity(F32,6n,output,balanced)) : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool} inside,
    %Equal.sym(Nat,List.length(&2,F32,rows_values(values)),32n,rows_length(values)) : Interval.Outside<offset,_,query> separate)

def half_list_preserved(upper: Bool,+values: Rows.Rows<Tile.Tile8>,+output: Tree.Tree<F32>,+query: Nat,
  +balanced: Shape.Balanced(F32,6n,output),inside: {Nat.is_lt(query,64n) == True{} : Bool},
  separate: Interval.Outside<Bool.pick(Nat,upper,32n,0n),32n,query>) ->
  {Tree.read(F32,Tree.write_sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0),output),U32.from_nat(query)) == Tree.read(F32,output,U32.from_nat(query)) : F32}:
  match upper:
    case False{}: rows_list_preserved(0n,values,output,query,balanced,inside,half_room(False{},values),separate)
    case True{}: rows_list_preserved(32n,values,output,query,balanced,inside,half_room(True{},values),separate)

def half_preserved(+upper: Bool,+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+query: Nat,
  +balanced: Shape.Balanced(F32,6n,output),inside: {Nat.is_lt(query,64n) == True{} : Bool},
  separate: Interval.Outside<Bool.pick(Nat,upper,32n,0n),32n,query>) ->
  {Tree.read(F32,Writes.pure(F32,half_writes(upper,values,bias),output),U32.from_nat(query)) == Tree.read(F32,output,U32.from_nat(query)) : F32}:
  Equal.trans(F32,Tree.read(F32,Writes.pure(F32,half_writes(upper,values,bias),output),U32.from_nat(query)),
    Tree.read(F32,Tree.write_sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0),output),U32.from_nat(query)),Tree.read(F32,output,U32.from_nat(query)),
    Equal.cong(Tree.Tree<F32>,F32,tree => Tree.read(F32,tree,U32.from_nat(query)),Writes.pure(F32,half_writes(upper,values,bias),output),Tree.write_sequence(~F32,rows_values(values),Bool.pick(U32,upper,32,0),output),half_written_list(upper,values,bias,output)),
    half_list_preserved(upper,values,output,query,balanced,inside,separate))

def half_base(+upper: Bool,row: Accumulator.Row) -> U32:
  (Bool.pick(U32,upper,32,0) + row_number(row) * 8 : U32)

def half_base_bound(upper: Bool,row: Accumulator.Row) -> {Nat.is_le(U32.to_nat(half_base(upper,row)),56n) == True{} : Bool}:
  match upper row:
    case False{} Accumulator.R0{}: {==}
    case False{} Accumulator.R1{}: {==}
    case False{} Accumulator.R2{}: {==}
    case False{} Accumulator.R3{}: {==}
    case True{} Accumulator.R0{}: {==}
    case True{} Accumulator.R1{}: {==}
    case True{} Accumulator.R2{}: {==}
    case True{} Accumulator.R3{}: {==}

def half_address_inside(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.is_lt(half_address(upper,row,lane),64) == True{} : Bool}:
  lane_inside(half_base(upper,row),U32.from_nat(lane),half_base_bound(upper,row),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,Bounds.roundtrip(lane,8,{==},Bounds.strict_le(lane,8n,within))) : {Nat.is_le(_,7n) == True{} : Bool}
    Division.lt_succ_le(lane,7n,within))

def half_job(+upper: Bool,count: Nat) -> Gemm.Job:
  Gemm.Job{half_context(upper),upper,count}

def half_accumulators(+upper: Bool,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,initial: Rows.Rows<Tile.Tile8>) -> Rows.Rows<Tile.Tile8>:
  Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,reduce_with(half_job(upper,count),input,weights,initial))

def half_execute(+upper: Bool,count: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Writes.Program<F32>,initial: Rows.Rows<Tile.Tile8>) -> GemmLaws.Buffers:
  GemmLaws.execute_with(~False{},half_job(upper,count),GemmLaws.Buffers{input,weights,bias,output},initial)

def result_program(+buffers: GemmLaws.Buffers) -> {GemmLaws.result(buffers) == Writes.evaluate(F32,buffers_program(buffers)) : Tree.Tree<F32>}:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  {==}

def half_execute_output(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+initial: Rows.Rows<Tile.Tile8>) ->
  {GemmLaws.result(half_execute(upper,count,input,weights,bias,output,initial)) ==
    Writes.pure(F32,half_writes(upper,half_accumulators(upper,count,input,weights,initial),bias),Writes.evaluate(F32,output)) : Tree.Tree<F32>}:
  +result = half_execute(upper,count,input,weights,bias,output,initial)
  Equal.trans(Tree.Tree<F32>,GemmLaws.result(result),Writes.evaluate(F32,buffers_program(result)),Writes.pure(F32,half_writes(upper,half_accumulators(upper,count,input,weights,initial),bias),Writes.evaluate(F32,output)),result_program(result),
    executed_with_output(~False{},half_job(upper,count),input,weights,bias,output,initial))

def half_execute_readback(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+initial: Rows.Rows<Tile.Tile8>,
  +balanced: Shape.Balanced(F32,6n,Writes.evaluate(F32,output)),+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.result(half_execute(upper,count,input,weights,bias,output,initial)),half_address(upper,row,lane)) ==
    TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,half_accumulators(upper,count,input,weights,initial))) : F32}:
  +values = half_accumulators(upper,count,input,weights,initial)
  +tree = Writes.evaluate(F32,output)
  +query = half_address(upper,row,lane)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.result(half_execute(upper,count,input,weights,bias,output,initial)),Writes.pure(F32,half_writes(upper,values,bias),tree),half_execute_output(upper,count,input,weights,bias,output,initial)) :
    {Tree.read(F32,_,query) == TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) : F32}
  Equal.trans(F32,Tree.read(F32,Writes.pure(F32,half_writes(upper,values,bias),tree),query),ProgramRead.selected(~F32,half_writes(upper,values,bias),query,Tree.read(F32,tree,query)),TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),
    half_readback(upper,values,bias,tree,query,balanced,half_address_inside(upper,row,lane,within)),half_selected(upper,row,lane,values,bias,Tree.read(F32,tree,query),within))

def half_execute_balance(+upper: Bool,+count: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<F32>,+initial: Rows.Rows<Tile.Tile8>,
  balanced: Shape.Balanced(F32,6n,Writes.evaluate(F32,output))) -> Shape.Balanced(F32,6n,GemmLaws.result(half_execute(upper,count,input,weights,bias,output,initial))):
  +values = half_accumulators(upper,count,input,weights,initial)
  %Equal.sym(Tree.Tree<F32>,GemmLaws.result(half_execute(upper,count,input,weights,bias,output,initial)),Writes.pure(F32,half_writes(upper,values,bias),Writes.evaluate(F32,output)),half_execute_output(upper,count,input,weights,bias,output,initial)) : Shape.Balanced(F32,6n,_)
  Writes.writes_preserve_balance(F32,half_writes(upper,values,bias),Writes.evaluate(F32,output),6n,balanced)

def rows_equal(+left: Rows.Rows<Tile.Tile8>,+right: Rows.Rows<Tile.Tile8>,
  e0: {Rows.first(~Tile.Tile8,left) == Rows.first(~Tile.Tile8,right) : Tile.Tile8},
  e1: {Rows.second(~Tile.Tile8,left) == Rows.second(~Tile.Tile8,right) : Tile.Tile8},
  e2: {Rows.third(~Tile.Tile8,left) == Rows.third(~Tile.Tile8,right) : Tile.Tile8},
  e3: {Rows.fourth(~Tile.Tile8,left) == Rows.fourth(~Tile.Tile8,right) : Tile.Tile8}) -> {left == right : Rows.Rows<Tile.Tile8>}:
  Rows.Rows{+a,+b,+c,+d} = left
  Rows.Rows{+p,+q,+r,+s} = right
  %e0 : {Rows.Rows{a,b,c,d} == Rows.Rows{_,q,r,s} : Rows.Rows<Tile.Tile8>}
  %e1 : {Rows.Rows{a,b,c,d} == Rows.Rows{a,_,r,s} : Rows.Rows<Tile.Tile8>}
  %e2 : {Rows.Rows{a,b,c,d} == Rows.Rows{a,b,_,s} : Rows.Rows<Tile.Tile8>}
  %e3 : {Rows.Rows{a,b,c,d} == Rows.Rows{a,b,c,_} : Rows.Rows<Tile.Tile8>}
  {==}

def upper_seed_preserved(+values: Rows.Rows<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+balanced: Shape.Balanced(F32,6n,output)) ->
  {GemmLaws.seed(Writes.pure(F32,half_writes(False{},values,bias),output),32,8) == GemmLaws.seed(output,32,8) : Rows.Rows<Tile.Tile8>}:
  +written = Writes.pure(F32,half_writes(False{},values,bias),output)
  rows_equal(GemmLaws.seed(written,32,8),GemmLaws.seed(output,32,8),
    Lanes.identity_cell_equal(written,output,32,
      half_preserved(False{},values,bias,output,32n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,33n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,34n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,35n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,36n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,37n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,38n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,39n,balanced,{==},Interval.After{{==}})),
    Lanes.identity_cell_equal(written,output,40,
      half_preserved(False{},values,bias,output,40n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,41n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,42n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,43n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,44n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,45n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,46n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,47n,balanced,{==},Interval.After{{==}})),
    Lanes.identity_cell_equal(written,output,48,
      half_preserved(False{},values,bias,output,48n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,49n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,50n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,51n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,52n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,53n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,54n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,55n,balanced,{==},Interval.After{{==}})),
    Lanes.identity_cell_equal(written,output,56,
      half_preserved(False{},values,bias,output,56n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,57n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,58n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,59n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,60n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,61n,balanced,{==},Interval.After{{==}}),
      half_preserved(False{},values,bias,output,62n,balanced,{==},Interval.After{{==}}),half_preserved(False{},values,bias,output,63n,balanced,{==},Interval.After{{==}})))

def half_seed_at(upper: Bool,row: Accumulator.Row,+lane: Nat,+output: Tree.Tree<F32>,within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmLaws.seed(output,Bool.pick(U32,upper,32,0),8))) == Tree.read(F32,output,half_address(upper,row,lane)) : F32}:
  match upper row:
    case False{} Accumulator.R0{}: Lanes.identity_cell_indexed(lane,output,0,within)
    case False{} Accumulator.R1{}: Lanes.identity_cell_indexed(lane,output,8,within)
    case False{} Accumulator.R2{}: Lanes.identity_cell_indexed(lane,output,16,within)
    case False{} Accumulator.R3{}: Lanes.identity_cell_indexed(lane,output,24,within)
    case True{} Accumulator.R0{}: Lanes.identity_cell_indexed(lane,output,32,within)
    case True{} Accumulator.R1{}: Lanes.identity_cell_indexed(lane,output,40,within)
    case True{} Accumulator.R2{}: Lanes.identity_cell_indexed(lane,output,48,within)
    case True{} Accumulator.R3{}: Lanes.identity_cell_indexed(lane,output,56,within)

def encoded_lane_address(+base: U32,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {(base + U32.from_nat(lane) : U32) == U32.from_nat(Nat.add(U32.to_nat(base),lane)) : U32}:
  %Arithmetic.native_roundtrip(base) : {(_ + U32.from_nat(lane) : U32) == U32.from_nat(Nat.add(U32.to_nat(base),lane)) : U32}
  WordArithmetic.native_add(U32.to_nat(base),lane)

def lane_below(+base: U32,+maximum: Nat,+capacity: U32,+lane: Nat,base_bound: {Nat.is_le(U32.to_nat(base),maximum) == True{} : Bool},+small: {Nat.is_lt(lane,8n) == True{} : Bool},room: {Nat.is_le(Nat.add(maximum,8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt((base + U32.from_nat(lane) : U32),capacity) == True{} : Bool}:
  +first = Order.add_le(U32.to_nat(base),maximum,lane,base_bound)
  +second = Order.add_lt(lane,8n,maximum,small)
  +below = Order.le_lt_trans(Nat.add(U32.to_nat(base),lane),Nat.add(maximum,lane),Nat.add(maximum,8n),first,
    %Order.commutative(lane,maximum) : {Nat.is_lt(_,Nat.add(maximum,8n)) == True{} : Bool}
    %Order.commutative(8n,maximum) : {Nat.is_lt(Nat.add(lane,maximum),_) == True{} : Bool} second)
  %Equal.sym(U32,(base + U32.from_nat(lane) : U32),U32.from_nat(Nat.add(U32.to_nat(base),lane)),encoded_lane_address(base,lane,small)) : {U32.is_lt(_,capacity) == True{} : Bool}
  ListRead.inside(Nat.add(U32.to_nat(base),lane),capacity,Bounds.nat_trans(Nat.add(U32.to_nat(base),lane),Nat.add(maximum,8n),U32.to_nat(capacity),below,room))

def lower_base_bound(row: Accumulator.Row) -> {Nat.is_le(U32.to_nat(half_base(False{},row)),24n) == True{} : Bool}:
  match row:
    case Accumulator.R0{}: {==}
    case Accumulator.R1{}: {==}
    case Accumulator.R2{}: {==}
    case Accumulator.R3{}: {==}

def lower_address_inside(+row: Accumulator.Row,+lane: Nat,small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.is_lt(half_address(False{},row,lane),32) == True{} : Bool}:
  lane_below(half_base(False{},row),24n,32,lane,lower_base_bound(row),small,{==})

def all_writes(~biased: Bool,count: Nat,+first: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  match count:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<F32>,job_writes(~biased,Gemm.job(layout,first),input,weights,bias),
        all_writes(~biased,rest,(first + 1 : U32),layout,input,weights,bias))

# Source owners and the evaluated output are observed together so one induction
# supplies both the reusable-buffer invariant and the ordered write list.
type Visible is Data:
  Visible{input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>}

def visible(buffers: GemmLaws.Buffers) -> Visible:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  Visible{input,weights,bias,Writes.evaluate(F32,output)}

def visible_output(state: Visible) -> Tree.Tree<F32>:
  Visible{input,weights,bias,output} = state
  output

def visible_result(+buffers: GemmLaws.Buffers) -> {visible_output(visible(buffers)) == GemmLaws.result(buffers) : Tree.Tree<F32>}:
  GemmLaws.Buffers{input,weights,bias,output} = buffers
  {==}

def fold_visible(~biased: Bool,+count: Nat,+first: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {visible(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),count,first,layout,GemmLaws.Buffers{input,weights,bias,program})) ==
    Visible{input,weights,bias,Writes.pure(F32,all_writes(~biased,count,first,layout,input,weights,bias),Writes.evaluate(F32,program))} : Visible}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +job = Gemm.job(layout,first)
      +next = buffers_program(GemmLaws.execute(~biased,job,GemmLaws.Buffers{input,weights,bias,program}))
      +here = job_writes(~biased,job,input,weights,bias)
      +later = all_writes(~biased,rest,(first + 1 : U32),layout,input,weights,bias)
      %Equal.sym(GemmLaws.Buffers,GemmLaws.execute(~biased,job,GemmLaws.Buffers{input,weights,bias,program}),GemmLaws.Buffers{input,weights,bias,next},
        executed(~biased,job,input,weights,bias,program)) :
        {visible(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),rest,(first + 1 : U32),layout,_)) ==
          Visible{input,weights,bias,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),Writes.evaluate(F32,program))} : Visible}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),Writes.evaluate(F32,program)),
        Writes.pure(F32,later,Writes.pure(F32,here,Writes.evaluate(F32,program))),Writes.concatenated_writes(F32,here,later,Writes.evaluate(F32,program))) :
        {visible(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),rest,(first + 1 : U32),layout,GemmLaws.Buffers{input,weights,bias,next})) == Visible{input,weights,bias,_} : Visible}
      %executed_output(~biased,job,input,weights,bias,program) :
        {visible(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),rest,(first + 1 : U32),layout,GemmLaws.Buffers{input,weights,bias,next})) ==
          Visible{input,weights,bias,Writes.pure(F32,later,_)} : Visible}
      fold_visible(~biased,rest,(first + 1 : U32),layout,input,weights,bias,next)


def fold_output(~biased: Bool,+count: Nat,+first: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<F32>,+program: Writes.Program<F32>) ->
  {GemmLaws.result(Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),count,first,layout,GemmLaws.Buffers{input,weights,bias,program})) ==
    Writes.pure(F32,all_writes(~biased,count,first,layout,input,weights,bias),Writes.evaluate(F32,program)) : Tree.Tree<F32>}:
  +state = Traversal.indexed_fold(~GemmLaws.Buffers,~Gemm.Layout,~GemmLaws.step(~biased),count,first,layout,GemmLaws.Buffers{input,weights,bias,program})
  %visible_result(state) : {_ == Writes.pure(F32,all_writes(~biased,count,first,layout,input,weights,bias),Writes.evaluate(F32,program)) : Tree.Tree<F32>}
  Equal.cong(Visible,Tree.Tree<F32>,state => visible_output(state),visible(state),
    Visible{input,weights,bias,Writes.pure(F32,all_writes(~biased,count,first,layout,input,weights,bias),Writes.evaluate(F32,program))},
    fold_visible(~biased,count,first,layout,input,weights,bias,program))

def job_count(+rows: U32,+columns: U32) -> Nat:
  U32.to_nat(Gemm.job_count(rows,columns))

def run_at_writes(~biased: Bool,+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +output: Tree.Tree<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) ->
  {GemmLaws.run_at(~biased,rows,input,weights,bias,output,columns,inner,start,output_columns) ==
    Writes.pure(F32,all_writes(~biased,job_count(rows,columns),0,Gemm.Layout{rows,columns,inner,start,output_columns},input,weights,bias),output) : Tree.Tree<F32>}:
  fold_output(~biased,job_count(rows,columns),0,Gemm.Layout{rows,columns,inner,start,output_columns},input,weights,bias,Writes.Initial{output})

def run_writes(~biased: Bool,+rows: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+columns: U32,+inner: U32) ->
  {GemmLaws.run(~biased,rows,input,weights,bias,output,columns,inner) == Writes.pure(F32,all_writes(~biased,job_count(rows,columns),0,Gemm.Layout{rows,columns,inner,0,columns},input,weights,bias),output) : Tree.Tree<F32>}:
  run_at_writes(~biased,rows,input,weights,bias,output,columns,inner,0,columns)
