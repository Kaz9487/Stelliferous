# A block's stores are the writes of a view: its stored rows (below the
# store's rows) by the writer's columns, row by row. Element k of the view is
# column k mod columns of row k / columns.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_spec.bend as GemmSpec
import ./traversal_reduction_refinement.bend as ReductionLaws
import ../tensor_view.bend as Views
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./binary_radix.bend as Radix
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition

type Block is Data:
  Block{row: U32,store: Gemm.Store,accumulators: Gemm.Accumulators,bias: Tree.Tree<F32>}

def block_row(+block: Block) -> U32:
  Block{row,store,accumulators,bias} = block
  row

def block_store(+block: Block) -> Gemm.Store:
  Block{row,store,accumulators,bias} = block
  store

def block_accumulators(+block: Block) -> Gemm.Accumulators:
  Block{row,store,accumulators,bias} = block
  accumulators

def block_bias(+block: Block) -> Tree.Tree<F32>:
  Block{row,store,accumulators,bias} = block
  bias

def store_base(+store: Gemm.Store) -> U32:
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  base

def store_row_stride(+store: Gemm.Store) -> U32:
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  row_stride

def store_lane_stride(+store: Gemm.Store) -> U32:
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  lane_stride

def store_rows(+store: Gemm.Store) -> U32:
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  rows

def store_lanes(+store: Gemm.Store) -> U32:
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  lanes

def scaled(+base: U32,offset: Nat,+stride: U32) -> U32:
  match offset:
    case 0n: base
    case 1n+rest:
      match rest:
        case 0n: (base + stride : U32)
        case 1n+more: (base + U32.from_nat(2n+more) * stride : U32)

def scaled_exact(+base: U32,+offset: Nat,+stride: U32) -> {scaled(base,offset,stride) == (base + U32.from_nat(offset) * stride : U32) : U32}:
  match offset:
    case 0n:
      %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {base == (base + _ : U32) : U32}
      Equal.sym(U32,(base + 0 : U32),base,Arithmetic.add_zero(base))
    case 1n+ +rest:
      match rest:
        case 0n:
          Equal.cong(U32,U32,word => (base + word : U32),stride,(1 * stride : U32),
            Equal.trans(U32,stride,(stride * 1 : U32),(1 * stride : U32),Equal.sym(U32,(stride * 1 : U32),stride,Arithmetic.multiply_one(stride)),
              Arithmetic.multiply_commutative(stride,1)))
        case 1n+more: {==}

# The view of a block: stored rows by the writer's columns from the store's
# base.
def stored_rows(+block: Block) -> U32:
  store_rows(block_store(block))

def rows_of(+block: Block) -> Nat:
  U32.to_nat(stored_rows(block))

def stored_columns(+writer: GemmLaws.Writer,+block: Block) -> U32:
  GemmLaws.columns(writer,block_store(block))

def columns_of(+writer: GemmLaws.Writer,+block: Block) -> Nat:
  U32.to_nat(stored_columns(writer,block))

def block_view(+writer: GemmLaws.Writer,+block: Block) -> Views.View:
  +store = block_store(block)
  Views.View{[Views.Axis{stored_rows(block),store_row_stride(store)},Views.Axis{stored_columns(writer,block),store_lane_stride(store)}],store_base(store)}

def cell_value(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: GemmLaws.Writer,+block: Block,+row: Nat,+column: Nat) -> GemmLaws.Element(writer):
  GemmLaws.finished(~value,~close,writer,block_row(block),block_store(block),block_bias(block),row,column,GemmSpec.select_row(row,block_accumulators(block)))

def element(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+block: Block,+position: Nat) -> GemmLaws.Element(writer):
  +columns = columns_of(writer,block)
  cell_value(~value,~close,writer,block,Nat.div(position,columns),Nat.mod(position,columns))

# The address the store writes for a cell is the view's address of it.
def cell_address(+writer: GemmLaws.Writer,+block: Block,+row: Nat,+column: Nat,+row_inside: {Nat.is_lt(row,rows_of(block)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,columns_of(writer,block)) == True{} : Bool}) ->
  {scaled(scaled(store_base(block_store(block)),row,store_row_stride(block_store(block))),column,store_lane_stride(block_store(block))) ==
    Copy.address(block_view(writer,block),Nat.add(Nat.mul(row,columns_of(writer,block)),column)) : U32}:
  +store = block_store(block)
  +base = store_base(store)
  +plane = store_row_stride(store)
  +step = store_lane_stride(store)
  +first = scaled(base,row,plane)
  +row_word = U32.from_nat(row)
  +column_word = U32.from_nat(column)
  %Equal.sym(U32,scaled(first,column,step),(first + column_word * step : U32),scaled_exact(first,column,step)) :
    {_ == Copy.address(block_view(writer,block),Nat.add(Nat.mul(row,columns_of(writer,block)),column)) : U32}
  %Equal.sym(U32,first,(base + row_word * plane : U32),scaled_exact(base,row,plane)) :
    {(_ + column_word * step : U32) == Copy.address(block_view(writer,block),Nat.add(Nat.mul(row,columns_of(writer,block)),column)) : U32}
  Equal.sym(U32,Copy.address(block_view(writer,block),Nat.add(Nat.mul(row,columns_of(writer,block)),column)),(base + row_word * plane + column_word * step : U32),
    ViewWrites.plane_address(base,stored_rows(block),plane,row,stored_columns(writer,block),step,column,row_inside,column_inside))

def cell_element(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+block: Block,+row: Nat,+column: Nat,
  +column_inside: {Nat.is_lt(column,columns_of(writer,block)) == True{} : Bool}) ->
  {cell_value(~value,~close,writer,block,row,column) == element(~value,~close,~writer,block,Nat.add(Nat.mul(row,columns_of(writer,block)),column)) : GemmLaws.Element(writer)}:
  +columns = columns_of(writer,block)
  +position = Nat.add(Nat.mul(row,columns),column)
  +positive = Order.positive_bound(column,columns,column_inside)
  %Equal.sym(Nat,Nat.div(position,columns),row,Radix.quotient(row,column,columns,positive,column_inside)) :
    {cell_value(~value,~close,writer,block,row,column) == cell_value(~value,~close,writer,block,_,Nat.mod(position,columns)) : GemmLaws.Element(writer)}
  %Equal.sym(Nat,Nat.mod(position,columns),column,Radix.remainder(row,column,columns,positive,column_inside)) :
    {cell_value(~value,~close,writer,block,row,column) == cell_value(~value,~close,writer,block,row,_) : GemmLaws.Element(writer)}
  {==}

# The cells of a row from a column on, at the addresses the store writes.
def cells(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: GemmLaws.Writer,+block: Block,+row: Nat,count: Nat,+column: Nat) ->
  List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  match count:
    case 0n: []
    case 1n+rest:
      +store = block_store(block)
      Writes.Put{scaled(scaled(store_base(store),row,store_row_stride(store)),column,store_lane_stride(store)),cell_value(~value,~close,writer,block,row,column)} <>
        cells(~value,~close,writer,block,row,rest,1n+column)

# The cells of a stored row are the writes of the view from the row's first
# element.
def cells_viewed(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+block: Block,+row: Nat,count: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,rows_of(block)) == True{} : Bool},+room: {Nat.add(column,count) == columns_of(writer,block) : Nat}) ->
  {cells(~value,~close,writer,block,row,count,column) ==
    ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),count,Nat.add(Nat.mul(row,columns_of(writer,block)),column),
      block_view(writer,block),block) : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +store = block_store(block)
      +line = Nat.mul(row,columns_of(writer,block))
      +address = scaled(scaled(store_base(store),row,store_row_stride(store)),column,store_lane_stride(store))
      +own = cell_value(~value,~close,writer,block,row,column)
      +inside = Order.below_complement(column,rest,columns_of(writer,block),room)
      +later = Equal.trans(Nat,1n+Nat.add(column,rest),Nat.add(column,1n+rest),columns_of(writer,block),
        Equal.sym(Nat,Nat.add(column,1n+rest),1n+Nat.add(column,rest),Addition.successor_right(column,rest)),room)
      %cell_address(writer,block,row,column,row_inside,inside) :
        {Writes.Put{address,own} <> cells(~value,~close,writer,block,row,rest,1n+column) ==
          Writes.Put{_,element(~value,~close,~writer,block,Nat.add(line,column))} <>
            ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),rest,1n+Nat.add(line,column),block_view(writer,block),block) :
          List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
      %cell_element(~value,~close,~writer,block,row,column,inside) :
        {Writes.Put{address,own} <> cells(~value,~close,writer,block,row,rest,1n+column) ==
          Writes.Put{address,_} <>
            ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),rest,1n+Nat.add(line,column),block_view(writer,block),block) :
          List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
      %Addition.successor_right(line,column) :
        {Writes.Put{address,own} <> cells(~value,~close,writer,block,row,rest,1n+column) ==
          Writes.Put{address,own} <> ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),rest,_,block_view(writer,block),block) :
          List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
      Equal.cong(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,
        tail => Writes.Put{address,own} <> tail,cells(~value,~close,writer,block,row,rest,1n+column),
        ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),rest,Nat.add(line,1n+column),block_view(writer,block),block),
        cells_viewed(~value,~close,~writer,block,row,rest,1n+column,row_inside,later))

# ---- The lanes of a row ----

# A stored row of the lane writer: its lanes below the store's lanes are the
# view's writes.
type Line is Data:
  Line{block: Block,row: Nat,lanes_small: {Nat.is_le(columns_of(GemmLaws.Lanes{},block),8n) == True{} : Bool},inside: {Nat.is_lt(row,rows_of(block)) == True{} : Bool}}

def line_block(+line: Line) -> Block:
  Line{block,row,lanes_small,inside} = line
  block

def line_view(+line: Line) -> Views.View:
  block_view(GemmLaws.Lanes{},line_block(line))

def line_base(+line: Line) -> Nat:
  Line{block,row,lanes_small,inside} = line
  Nat.mul(row,columns_of(GemmLaws.Lanes{},block))

def line_lanes(+line: Line) -> Nat:
  columns_of(GemmLaws.Lanes{},line_block(line))

def single(~Context: Data,+context: Context) -> Nat:
  1n

def eight(~Context: Data,+context: Context) -> Nat:
  8n

def line_lane(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+line: Line,+lane: Nat) -> List<&2,Writes.Assignment<F32>>:
  Line{block,row,lanes_small,inside} = line
  +store = block_store(block)
  GemmLaws.put(~F32,U32.is_lt(U32.from_nat(lane),stored_columns(GemmLaws.Lanes{},block)),
    scaled(scaled(store_base(store),row,store_row_stride(store)),lane,store_lane_stride(store)),cell_value(~value,~close,GemmLaws.Lanes{},block,row,lane))

def line_present(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+line: Line,+lane: Nat,+below: {Nat.is_lt(lane,line_lanes(line)) == True{} : Bool}) ->
  {line_lane(~value,~close,line,lane) ==
    ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),single(~Line,line),Nat.add(line_base(line),Nat.mul(lane,single(~Line,line))),
      line_view(line),line_block(line)) : List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+lanes_small,+inside} = line
  +store = block_store(block)
  +lanes = columns_of(GemmLaws.Lanes{},block)
  +position = Nat.add(Nat.mul(row,lanes),lane)
  +address = scaled(scaled(store_base(store),row,store_row_stride(store)),lane,store_lane_stride(store))
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(lane),stored_columns(GemmLaws.Lanes{},block)),Nat.is_lt(lane,lanes),True{},
    ViewWrites.word_compared(lane,8,stored_columns(GemmLaws.Lanes{},block),{==},Order.transitive(lane,lanes,8n,Bounds.strict_le(lane,lanes,below),lanes_small)),below)
  %Equal.sym(Nat,Nat.mul(lane,1n),lane,Order.times_one(lane)) :
    {line_lane(~value,~close,Line{block,row,lanes_small,inside},lane) ==
      ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),1n,Nat.add(Nat.mul(row,lanes),_),block_view(GemmLaws.Lanes{},block),block) :
      List<&2,Writes.Assignment<F32>>}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),stored_columns(GemmLaws.Lanes{},block)),True{},mask) :
    {GemmLaws.put(~F32,_,address,cell_value(~value,~close,GemmLaws.Lanes{},block,row,lane)) ==
      [Writes.Put{Copy.address(block_view(GemmLaws.Lanes{},block),position),element(~value,~close,~GemmLaws.Lanes{},block,position)}] : List<&2,Writes.Assignment<F32>>}
  %cell_address(GemmLaws.Lanes{},block,row,lane,inside,below) :
    {[Writes.Put{address,cell_value(~value,~close,GemmLaws.Lanes{},block,row,lane)}] ==
      [Writes.Put{_,element(~value,~close,~GemmLaws.Lanes{},block,position)}] : List<&2,Writes.Assignment<F32>>}
  %cell_element(~value,~close,~GemmLaws.Lanes{},block,row,lane,below) :
    {[Writes.Put{address,cell_value(~value,~close,GemmLaws.Lanes{},block,row,lane)}] == [Writes.Put{address,_}] : List<&2,Writes.Assignment<F32>>}
  {==}

def line_absent(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+line: Line,+extra: Nat,
  +below: {Nat.is_lt(Nat.add(line_lanes(line),extra),eight(~Line,line)) == True{} : Bool}) ->
  {line_lane(~value,~close,line,Nat.add(line_lanes(line),extra)) == Nil{} : List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+lanes_small,+inside} = line
  +store = block_store(block)
  +lanes = columns_of(GemmLaws.Lanes{},block)
  +lane = Nat.add(lanes,extra)
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(lane),stored_columns(GemmLaws.Lanes{},block)),Nat.is_lt(lane,lanes),False{},
    ViewWrites.word_compared(lane,8,stored_columns(GemmLaws.Lanes{},block),{==},Bounds.strict_le(lane,8n,below)),Order.not_below(lanes,lane,Order.le_add(lanes,extra)))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),stored_columns(GemmLaws.Lanes{},block)),False{},mask) :
    {GemmLaws.put(~F32,_,scaled(scaled(store_base(store),row,store_row_stride(store)),lane,store_lane_stride(store)),
        cell_value(~value,~close,GemmLaws.Lanes{},block,row,lane)) == Nil{} : List<&2,Writes.Assignment<F32>>}
  {==}

def line_row(+line: Line) -> Nat:
  Line{block,row,lanes_small,inside} = line
  row

def line_writes(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+line: Line) ->
  {GemmLaws.lane_writes(~close,True{},scaled(store_base(block_store(line_block(line))),line_row(line),store_row_stride(block_store(line_block(line)))),
      store_lane_stride(block_store(line_block(line))),stored_columns(GemmLaws.Lanes{},line_block(line)),
      value(block_bias(line_block(line)),GemmLaws.shifted(block_row(line_block(line)),line_row(line))),GemmSpec.select_row(line_row(line),block_accumulators(line_block(line)))) ==
    ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),line_lanes(line),line_base(line),line_view(line),line_block(line)) :
    List<&2,Writes.Assignment<F32>>}:
  Line{+block,+row,+lanes_small,+inside} = line
  +lanes = columns_of(GemmLaws.Lanes{},block)
  %Order.times_one(lanes) :
    {ViewWrites.parts(~F32,~Line,~line_lane(~value,~close),Line{block,row,lanes_small,inside},8n) ==
      ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),_,Nat.mul(row,lanes),block_view(GemmLaws.Lanes{},block),block) :
      List<&2,Writes.Assignment<F32>>}
  ViewWrites.masked(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),~Line,~line_lane(~value,~close),~line_view,~line_block,~line_base,~single(~Line),~line_lanes,~eight(~Line),
    ~line_present(~value,~close),~line_absent(~value,~close),Line{block,row,lanes_small,inside},lanes_small)

# A stored row of the lane writer writes its cells.
def lane_cells(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+block: Block,+row: Nat,+inside: {Nat.is_lt(row,rows_of(block)) == True{} : Bool},
  +lanes_small: {Nat.is_le(columns_of(GemmLaws.Lanes{},block),8n) == True{} : Bool}) ->
  {GemmLaws.lane_writes(~close,True{},scaled(store_base(block_store(block)),row,store_row_stride(block_store(block))),store_lane_stride(block_store(block)),
      stored_columns(GemmLaws.Lanes{},block),value(block_bias(block),GemmLaws.shifted(block_row(block),row)),GemmSpec.select_row(row,block_accumulators(block))) ==
    cells(~value,~close,GemmLaws.Lanes{},block,row,columns_of(GemmLaws.Lanes{},block),0n) : List<&2,Writes.Assignment<F32>>}:
  +lanes = columns_of(GemmLaws.Lanes{},block)
  +viewed = ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),lanes,Nat.mul(row,lanes),block_view(GemmLaws.Lanes{},block),block)
  Equal.trans(List<&2,Writes.Assignment<F32>>,
    GemmLaws.lane_writes(~close,True{},scaled(store_base(block_store(block)),row,store_row_stride(block_store(block))),store_lane_stride(block_store(block)),
      stored_columns(GemmLaws.Lanes{},block),value(block_bias(block),GemmLaws.shifted(block_row(block),row)),GemmSpec.select_row(row,block_accumulators(block))),
    viewed,cells(~value,~close,GemmLaws.Lanes{},block,row,lanes,0n),
    line_writes(~value,~close,Line{block,row,lanes_small,inside}),
    Equal.sym(List<&2,Writes.Assignment<F32>>,cells(~value,~close,GemmLaws.Lanes{},block,row,lanes,0n),viewed,
      %Addition.zero_right(Nat.mul(row,lanes)) :
        {cells(~value,~close,GemmLaws.Lanes{},block,row,lanes,0n) ==
          ViewWrites.writes_from(~F32,~Block,~element(~value,~close,~GemmLaws.Lanes{}),lanes,_,block_view(GemmLaws.Lanes{},block),block) : List<&2,Writes.Assignment<F32>>}
      cells_viewed(~value,~close,~GemmLaws.Lanes{},block,row,lanes,0n,inside,{==})))

# A stored row writes its cells: its lanes, or its one tile.
def row_cells(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: GemmLaws.Writer,+block: Block,+row: Nat,
  +inside: {Nat.is_lt(row,rows_of(block)) == True{} : Bool},+columns_small: {Nat.is_le(columns_of(writer,block),8n) == True{} : Bool}) ->
  {GemmLaws.row_writes(~value,~close,writer,block_row(block),block_store(block),block_bias(block),row,True{},
      scaled(store_base(block_store(block)),row,store_row_stride(block_store(block))),GemmSpec.select_row(row,block_accumulators(block))) ==
    cells(~value,~close,writer,block,row,columns_of(writer,block),0n) : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  match writer:
    case GemmLaws.Lanes{}:
      Block{+first,Gemm.Store{+base,+row_stride,+lane_stride,+rows,+lanes},+accumulators,+bias} = block
      lane_cells(~value,~close,Block{first,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias},row,inside,columns_small)
    case GemmLaws.Tiles{}: {==}

# ---- The rows of a block ----

# A block whose store has at most eight rows and columns.
type Placed<-writer: GemmLaws.Writer> is Data:
  Placed{block: Block,rows_small: {Nat.is_le(rows_of(block),8n) == True{} : Bool},columns_small: {Nat.is_le(columns_of(writer,block),8n) == True{} : Bool}}

def placed_block(~writer: GemmLaws.Writer,+placed: Placed<writer>) -> Block:
  Placed{block,rows_small,columns_small} = placed
  block

def placed_view(~writer: GemmLaws.Writer,+placed: Placed<writer>) -> Views.View:
  block_view(writer,placed_block(~writer,placed))

def start(~writer: GemmLaws.Writer,+placed: Placed<writer>) -> Nat:
  0n

def placed_columns(~writer: GemmLaws.Writer,+placed: Placed<writer>) -> Nat:
  columns_of(writer,placed_block(~writer,placed))

def placed_rows(~writer: GemmLaws.Writer,+placed: Placed<writer>) -> Nat:
  rows_of(placed_block(~writer,placed))

def row_part(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+placed: Placed<writer>,+row: Nat) ->
  List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  Placed{block,rows_small,columns_small} = placed
  +store = block_store(block)
  GemmLaws.row_writes(~value,~close,writer,block_row(block),store,block_bias(block),row,U32.is_lt(U32.from_nat(row),store_rows(store)),
    scaled(store_base(store),row,store_row_stride(store)),GemmSpec.select_row(row,block_accumulators(block)))

def row_present(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+placed: Placed<writer>,+row: Nat,
  +below: {Nat.is_lt(row,placed_rows(~writer,placed)) == True{} : Bool}) ->
  {row_part(~value,~close,~writer,placed,row) ==
    ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),placed_columns(~writer,placed),
      Nat.add(start(~writer,placed),Nat.mul(row,placed_columns(~writer,placed))),placed_view(~writer,placed),placed_block(~writer,placed)) :
    List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  Placed{+block,+rows_small,+columns_small} = placed
  +store = block_store(block)
  +columns = columns_of(writer,block)
  +small = Order.transitive(row,rows_of(block),8n,Bounds.strict_le(row,rows_of(block),below),rows_small)
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(row),store_rows(store)),Nat.is_lt(row,rows_of(block)),True{},
    ViewWrites.word_compared(row,8,store_rows(store),{==},small),below)
  +written = GemmLaws.row_writes(~value,~close,writer,block_row(block),store,block_bias(block),row,True{},scaled(store_base(store),row,store_row_stride(store)),
    GemmSpec.select_row(row,block_accumulators(block)))
  +viewed = ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),columns,Nat.mul(row,columns),block_view(writer,block),block)
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(row),store_rows(store)),True{},mask) :
    {GemmLaws.row_writes(~value,~close,writer,block_row(block),store,block_bias(block),row,_,scaled(store_base(store),row,store_row_stride(store)),
        GemmSpec.select_row(row,block_accumulators(block))) == viewed : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
  Equal.trans(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,written,cells(~value,~close,writer,block,row,columns,0n),viewed,
    row_cells(~value,~close,writer,block,row,below,columns_small),
    %Addition.zero_right(Nat.mul(row,columns)) :
      {cells(~value,~close,writer,block,row,columns,0n) ==
        ViewWrites.writes_from(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),columns,_,block_view(writer,block),block) :
        List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
    cells_viewed(~value,~close,~writer,block,row,columns,0n,below,{==}))

def row_absent(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+placed: Placed<writer>,+extra: Nat,
  +below: {Nat.is_lt(Nat.add(placed_rows(~writer,placed),extra),eight(~Placed<writer>,placed)) == True{} : Bool}) ->
  {row_part(~value,~close,~writer,placed,Nat.add(placed_rows(~writer,placed),extra)) == Nil{} : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  Placed{+block,+rows_small,+columns_small} = placed
  +store = block_store(block)
  +used = rows_of(block)
  +row = Nat.add(used,extra)
  +mask = Equal.trans(Bool,U32.is_lt(U32.from_nat(row),store_rows(store)),Nat.is_lt(row,used),False{},
    ViewWrites.word_compared(row,8,store_rows(store),{==},Bounds.strict_le(row,8n,below)),Order.not_below(used,row,Order.le_add(used,extra)))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(row),store_rows(store)),False{},mask) :
    {GemmLaws.row_writes(~value,~close,writer,block_row(block),store,block_bias(block),row,_,scaled(store_base(store),row,store_row_stride(store)),
        GemmSpec.select_row(row,block_accumulators(block))) == Nil{} : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}
  GemmLaws.row_unwritten(~value,~close,writer,block_row(block),store,block_bias(block),row,scaled(store_base(store),row,store_row_stride(store)),
    GemmSpec.select_row(row,block_accumulators(block)))

# The stores of a block are the writes of its view.
def view_writes(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~writer: GemmLaws.Writer,+row: U32,+base: U32,+row_stride: U32,+lane_stride: U32,
  +rows: U32,+lanes: U32,+accumulators: Gemm.Accumulators,+bias: Tree.Tree<F32>,+rows_small: {Nat.is_le(U32.to_nat(rows),8n) == True{} : Bool},
  +columns_small: {Nat.is_le(columns_of(writer,Block{row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias}),8n) == True{} : Bool}) ->
  {GemmLaws.block_writes(~value,~close,writer,row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias) ==
    ViewWrites.writes(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),
      Nat.mul(U32.to_nat(rows),columns_of(writer,Block{row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias})),
      block_view(writer,Block{row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias}),Block{row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias}) :
    List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  +placed = {Placed{Block{row,Gemm.Store{base,row_stride,lane_stride,rows,lanes},accumulators,bias},rows_small,columns_small} : Placed<writer>}
  ViewWrites.masked(~GemmLaws.Element(writer),~Block,~element(~value,~close,~writer),~Placed<writer>,~row_part(~value,~close,~writer),~placed_view(~writer),
    ~placed_block(~writer),~start(~writer),~placed_columns(~writer),~placed_rows(~writer),~eight(~Placed<writer>),~row_present(~value,~close,~writer),
    ~row_absent(~value,~close,~writer),placed,rows_small)

# ---- What a caller asks of a cell ----

# Cell (row, column) as a caller states it: the finish of the ordered dots of
# its row from +0.
def formed(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~order: GemmLaws.Order,+writer: GemmLaws.Writer,+first: U32,+store: Gemm.Store,+bias: Tree.Tree<F32>,
  +taps: List<&2,ReductionLaws.AddressPair>,+source: GemmSpec.Source,+row: Nat,+column: Nat) -> GemmLaws.Element(writer):
  match writer:
    case GemmLaws.Lanes{}: close(GemmSpec.dot(order,taps,row,column,source,0.0),value(bias,(first + U32.from_nat(row) : U32)))
    case GemmLaws.Tiles{}: GemmLaws.finished(~value,~close,GemmLaws.Tiles{},first,store,bias,row,column,GemmSpec.dot_row(order,taps,row,source))

# A row equal to its dots finishes as the caller states it.
def formed_finished(~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~order: GemmLaws.Order,+writer: GemmLaws.Writer,+first: U32,+store: Gemm.Store,+bias: Tree.Tree<F32>,
  +taps: List<&2,ReductionLaws.AddressPair>,+source: GemmSpec.Source,+row: Nat,+column: Nat,+values: Tile.Tile8,
  +column_small: {Nat.is_lt(column,8n) == True{} : Bool},+same: {values == GemmSpec.dot_row(order,taps,row,source) : Tile.Tile8}) ->
  {formed(~value,~close,~order,writer,first,store,bias,taps,source,row,column) == GemmLaws.finished(~value,~close,writer,first,store,bias,row,column,values) :
    GemmLaws.Element(writer)}:
  match writer:
    case GemmLaws.Lanes{}:
      +formed_value = formed(~value,~close,~order,GemmLaws.Lanes{},first,store,bias,taps,source,row,column)
      %Equal.sym(Tile.Tile8,values,GemmSpec.dot_row(order,taps,row,source),same) :
        {formed_value == close(TileLaws.select_lane(column,_),value(bias,GemmLaws.shifted(first,row))) : F32}
      %Equal.sym(F32,TileLaws.select_lane(column,GemmSpec.dot_row(order,taps,row,source)),GemmSpec.dot(order,taps,row,column,source,0.0),
        GemmSpec.dot_lane(order,taps,row,column,source,column_small)) :
        {formed_value == close(_,value(bias,GemmLaws.shifted(first,row))) : F32}
      %Equal.sym(U32,GemmLaws.shifted(first,row),(first + U32.from_nat(row) : U32),GemmLaws.shifted_exact(first,row)) :
        {formed_value == close(GemmSpec.dot(order,taps,row,column,source,0.0),value(bias,_)) : F32}
      {==}
    case GemmLaws.Tiles{}:
      %Equal.sym(Tile.Tile8,values,GemmSpec.dot_row(order,taps,row,source),same) :
        {formed(~value,~close,~order,GemmLaws.Tiles{},first,store,bias,taps,source,row,column) ==
          GemmLaws.finished(~value,~close,GemmLaws.Tiles{},first,store,bias,row,column,_) : Tile.Tile8}
      {==}
