# A run is the writes of its jobs in order. Cell (row, column) of a job
# is written at the store's base + row * row stride + column * lane stride
# with the writer's finish of the ordered dots of its row from +0. A caller
# proves what its cells mean and that the cell it asks for is a cell of some
# job; the run's readback follows from these alone.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../tensor_view.bend as Views
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./write_program.bend as Writes
import ./write_program_fold.bend as Fold
import ./write_program_every.bend as Every
import ./write_program_readback.bend as WriteReadback
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_spec.bend as GemmSpec
import ./kernel_gemm_writes.bend as GemmWrites
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ../traversal_array.bend as Traversal

# ---- The writes of a run ----

# The trees a run reads.
type Readers is Data:
  Readers{row_tiles: Tree.Tree<Tile.Tile8>,lane_tiles: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>}

# The writes of a job: the block's stores.
def job_writes(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers) -> List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  GemmLaws.block_writes(~value,~close,writer,row,store,
    GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,depth,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()})),bias)

# Job index of a plan.
def step_writes(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,index: U32,plan: Plan,readers: Readers) -> List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),readers)

def kept(~Plan: Data,index: U32,plan: Plan,readers: Readers) -> Readers:
  readers

# Jobs first, first + 1, ... (count of them).
def all_writes(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,count: Nat,+first: U32,+plan: Plan,+readers: Readers) -> List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  Fold.all(~GemmLaws.Element(writer),~Plan,~Readers,~step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),~kept(~Plan),count,first,plan,readers)

def readers_of(~E: Data,operands: GemmLaws.Operands<E>) -> Readers:
  GemmLaws.Operands{row_tiles,lane_tiles,bias,output} = operands
  Readers{row_tiles,lane_tiles,bias}

# A job keeps the trees it reads and appends its writes.
def block_appends(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Writes.Program<GemmLaws.Element(writer)>) ->
  {GemmLaws.block(~Cursor,~writer,~advance,~order,~value,~close,job,GemmLaws.Operands{row_tiles,lane_tiles,bias,output}) ==
    GemmLaws.Operands{row_tiles,lane_tiles,bias,Writes.Append{job_writes(~Cursor,~advance,~order,~writer,~value,~close,job,Readers{row_tiles,lane_tiles,bias}),output}} :
    GemmLaws.Operands<GemmLaws.Element(writer)>}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  {==}

def readers_step(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+index: U32,+plan: Plan,+operands: GemmLaws.Operands<GemmLaws.Element(writer)>) ->
  {readers_of(~GemmLaws.Element(writer),GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close,
      index,plan,operands)) ==
    kept(~Plan,index,plan,readers_of(~GemmLaws.Element(writer),operands)) : Readers}:
  GemmLaws.Operands{+row_tiles,+lane_tiles,+bias,+output} = operands
  %Equal.sym(GemmLaws.Operands<GemmLaws.Element(writer)>,
    GemmLaws.block(~Cursor,~writer,~advance,~order,~value,~close,job_at(plan,index),GemmLaws.Operands{row_tiles,lane_tiles,bias,output}),
    GemmLaws.Operands{row_tiles,lane_tiles,bias,Writes.Append{job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),Readers{row_tiles,lane_tiles,bias}),output}},
    block_appends(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),row_tiles,lane_tiles,bias,output)) :
    {readers_of(~GemmLaws.Element(writer),_) == Readers{row_tiles,lane_tiles,bias} : Readers}
  {==}

def output_step(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+index: U32,+plan: Plan,+operands: GemmLaws.Operands<GemmLaws.Element(writer)>) ->
  {GemmLaws.output_tree(~GemmLaws.Element(writer),GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,
      ~value,~close,index,plan,operands)) ==
    Writes.pure(GemmLaws.Element(writer),step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,index,plan,readers_of(~GemmLaws.Element(writer),operands)),
      GemmLaws.output_tree(~GemmLaws.Element(writer),operands)) : Tree.Tree<GemmLaws.Element(writer)>}:
  GemmLaws.Operands{+row_tiles,+lane_tiles,+bias,+output} = operands
  %Equal.sym(GemmLaws.Operands<GemmLaws.Element(writer)>,
    GemmLaws.block(~Cursor,~writer,~advance,~order,~value,~close,job_at(plan,index),GemmLaws.Operands{row_tiles,lane_tiles,bias,output}),
    GemmLaws.Operands{row_tiles,lane_tiles,bias,Writes.Append{job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),Readers{row_tiles,lane_tiles,bias}),output}},
    block_appends(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),row_tiles,lane_tiles,bias,output)) :
    {GemmLaws.output_tree(~GemmLaws.Element(writer),_) ==
      Writes.pure(GemmLaws.Element(writer),job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,index),Readers{row_tiles,lane_tiles,bias}),
        Writes.evaluate(GemmLaws.Element(writer),output)) : Tree.Tree<GemmLaws.Element(writer)>}
  {==}

# A run from job first applies the writes of its jobs in order.
def run_writes(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+count: Nat,+first: U32,+plan: Plan,+operands: GemmLaws.Operands<GemmLaws.Element(writer)>) ->
  {GemmLaws.output_tree(~GemmLaws.Element(writer),Traversal.indexed_fold(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,
      ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),count,first,plan,operands)) ==
    Writes.pure(GemmLaws.Element(writer),all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,first,plan,readers_of(~GemmLaws.Element(writer),operands)),
      GemmLaws.output_tree(~GemmLaws.Element(writer),operands)) : Tree.Tree<GemmLaws.Element(writer)>}:
  Fold.output(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,~GemmLaws.Element(writer),~Readers,
    ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),
    ~readers_of(~GemmLaws.Element(writer)),~GemmLaws.output_tree(~GemmLaws.Element(writer)),~step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),~kept(~Plan),
    ~readers_step(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),~output_step(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),count,first,plan,operands)

# A run keeps the trees it reads.
def run_keeps(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+count: Nat,+first: U32,+plan: Plan,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<F32>,+output: Writes.Program<GemmLaws.Element(writer)>) ->
  {Traversal.indexed_fold(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,
      ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),count,first,plan,
      GemmLaws.Operands{row_tiles,lane_tiles,bias,output}) ==
    GemmLaws.Operands{row_tiles,lane_tiles,bias,GemmLaws.operand_output(~GemmLaws.Element(writer),Traversal.indexed_fold(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,
      ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),count,first,plan,
      GemmLaws.Operands{row_tiles,lane_tiles,bias,output}))} : GemmLaws.Operands<GemmLaws.Element(writer)>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +appended = {Writes.Append{job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,first),Readers{row_tiles,lane_tiles,bias}),output} : Writes.Program<GemmLaws.Element(writer)>}
      +next = {GemmLaws.Operands{row_tiles,lane_tiles,bias,appended} : GemmLaws.Operands<GemmLaws.Element(writer)>}
      %Equal.sym(GemmLaws.Operands<GemmLaws.Element(writer)>,
        GemmLaws.block(~Cursor,~writer,~advance,~order,~value,~close,job_at(plan,first),GemmLaws.Operands{row_tiles,lane_tiles,bias,output}),
        next,block_appends(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan,first),row_tiles,lane_tiles,bias,output)) :
        {Traversal.indexed_fold(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,
            ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),rest,(first + 1 : U32),plan,_) ==
          GemmLaws.Operands{row_tiles,lane_tiles,bias,GemmLaws.operand_output(~GemmLaws.Element(writer),Traversal.indexed_fold(~GemmLaws.Operands<GemmLaws.Element(writer)>,~Plan,
            ~GemmLaws.job_step(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close),rest,(first + 1 : U32),plan,_))} :
          GemmLaws.Operands<GemmLaws.Element(writer)>}
      run_keeps(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,rest,(first + 1 : U32),plan,row_tiles,lane_tiles,bias,appended)

def represented_output(~E: Data,+operands: GemmLaws.Operands<E>) ->
  {Gemm.output_of(~E,~Array<F32>,GemmLaws.represent(~E,operands)) == Tree.pack(E,GemmLaws.output_tree(~E,operands)) : Array<E>}:
  GemmLaws.Operands{+row_tiles,+lane_tiles,+bias,+output} = operands
  {==}

# The output of a native run over packed trees is the packed tree after
# the writes of its jobs.
def native_output(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: GemmLaws.Order,~ordered: GemmLaws.Ordered(~times,order),~writer: GemmLaws.Writer,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<GemmLaws.Element(writer),Array<F32>> -> Gemm.Written<GemmLaws.Element(writer),Array<F32>>,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~stored: GemmLaws.Stored(~writer,~write,~value,~close),
  +count: Nat,+plan: Plan,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+tree: Tree.Tree<GemmLaws.Element(writer)>) ->
  {Gemm.output_of(~GemmLaws.Element(writer),~Array<F32>,Gemm.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,count,plan,
      GemmLaws.represent(~GemmLaws.Element(writer),GemmLaws.Operands{row_tiles,lane_tiles,bias,Writes.Initial{tree}}))) ==
    Tree.pack(GemmLaws.Element(writer),Writes.pure(GemmLaws.Element(writer),all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,Readers{row_tiles,lane_tiles,bias}),
      tree)) : Array<GemmLaws.Element(writer)>}:
  +operands = {GemmLaws.Operands{row_tiles,lane_tiles,bias,Writes.Initial{tree}} : GemmLaws.Operands<GemmLaws.Element(writer)>}
  +model = GemmLaws.run(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close,count,plan,operands)
  +written = Writes.pure(GemmLaws.Element(writer),all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,Readers{row_tiles,lane_tiles,bias}),tree)
  %Equal.sym(Gemm.Operands<GemmLaws.Element(writer),Array<F32>>,
    Gemm.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,count,plan,
      GemmLaws.represent(~GemmLaws.Element(writer),operands)),
    GemmLaws.represent(~GemmLaws.Element(writer),model),
    GemmLaws.run_matches(~Plan,~Cursor,~writer,~job_at,~advance,~times,~order,~ordered,
      ~write,~value,~close,~stored,count,plan,operands)) :
    {Gemm.output_of(~GemmLaws.Element(writer),~Array<F32>,_) == Tree.pack(GemmLaws.Element(writer),written) : Array<GemmLaws.Element(writer)>}
  %Equal.sym(Array<GemmLaws.Element(writer)>,Gemm.output_of(~GemmLaws.Element(writer),~Array<F32>,GemmLaws.represent(~GemmLaws.Element(writer),model)),
    Tree.pack(GemmLaws.Element(writer),GemmLaws.output_tree(~GemmLaws.Element(writer),model)),represented_output(~GemmLaws.Element(writer),model)) :
    {_ == Tree.pack(GemmLaws.Element(writer),written) : Array<GemmLaws.Element(writer)>}
  Equal.cong(Tree.Tree<GemmLaws.Element(writer)>,Array<GemmLaws.Element(writer)>,result => Tree.pack(GemmLaws.Element(writer),result),
    GemmLaws.output_tree(~GemmLaws.Element(writer),model),written,run_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,operands))

# ---- The cells of a job ----

def job_store(~Cursor: Data,+job: Gemm.Job<Cursor>) -> Gemm.Store:
  Gemm.Job{row,depth,cursor,store} = job
  store

def job_rows(~Cursor: Data,+job: Gemm.Job<Cursor>) -> Nat:
  U32.to_nat(GemmWrites.store_rows(job_store(~Cursor,job)))

def job_columns(~Cursor: Data,+writer: GemmLaws.Writer,+job: Gemm.Job<Cursor>) -> Nat:
  U32.to_nat(GemmLaws.columns(writer,job_store(~Cursor,job)))

def cell_address(~Cursor: Data,+job: Gemm.Job<Cursor>,+row: Nat,+column: Nat) -> U32:
  Gemm.Job{first,depth,cursor,Gemm.Store{+base,+row_stride,+lane_stride,rows,lanes}} = job
  (base + U32.from_nat(row) * row_stride + U32.from_nat(column) * lane_stride : U32)

def cell_value(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,+writer: GemmLaws.Writer,
  +job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+column: Nat) -> GemmLaws.Element(writer):
  Gemm.Job{+first,+depth,+cursor,+store} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  GemmWrites.formed(~value,~close,~order,writer,first,store,bias,GemmSpec.taps(~Cursor,~advance,depth,cursor),GemmSpec.Source{row_tiles,lane_tiles},row,column)

# The only cell of a tile row is cell zero.
def only_cell(+column: Nat,+inside: {Nat.is_lt(column,1n) == True{} : Bool}) -> {column == 0n : Nat}:
  match column:
    case 0n: {==}
    case 1n+ +rest: Empty.absurd({1n+rest == 0n : Nat},Order.below_zero(rest,inside))

# The closed dot of a row and lane of a job.
def lane_value(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+lane: Nat) -> F32:
  Gemm.Job{+first,+depth,+cursor,store} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  close(GemmSpec.dot(order,GemmSpec.taps(~Cursor,~advance,depth,cursor),row,lane,GemmSpec.Source{row_tiles,lane_tiles},0.0),value(bias,GemmLaws.shifted(first,lane)))

# A stored lane of the tile of a row is the closed dot of the row and lane.
def tile_lane(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {U32.is_lt(U32.from_nat(lane),GemmWrites.store_lanes(job_store(~Cursor,job))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,cell_value(~Cursor,~advance,~order,~value,~close,GemmLaws.Tiles{},job,readers,row,0n)) ==
    lane_value(~Cursor,~advance,~order,~value,~close,job,readers,row,lane) : F32}:
  Gemm.Job{+first,+depth,+cursor,Gemm.Store{+base,+row_stride,+lane_stride,+rows,+lanes}} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  +taps = GemmSpec.taps(~Cursor,~advance,depth,cursor)
  +source = {GemmSpec.Source{row_tiles,lane_tiles} : GemmSpec.Source}
  +values = GemmSpec.dot_row(order,taps,row,source)
  +bias_value = value(bias,GemmLaws.shifted(first,lane))
  Equal.trans(F32,TileLaws.select_lane(lane,GemmLaws.finished_tile(~value,~close,first,lanes,bias,values)),close(TileLaws.select_lane(lane,values),bias_value),
    close(GemmSpec.dot(order,taps,row,lane,source,0.0),bias_value),
    GemmLaws.finished_lane(~value,~close,first,lanes,bias,values,lane,inside,small),
    Equal.cong(F32,F32,dot => close(dot,bias_value),TileLaws.select_lane(lane,values),GemmSpec.dot(order,taps,row,lane,source,0.0),GemmSpec.dot_lane(order,taps,row,lane,source,small)))

# The block a job stores.
def job_block(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+job: Gemm.Job<Cursor>,+readers: Readers) -> GemmWrites.Block:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  GemmWrites.Block{row,store,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,depth,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()})),bias}

def block_store_is(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+job: Gemm.Job<Cursor>,+readers: Readers) ->
  {GemmWrites.block_store(job_block(~Cursor,~advance,~order,job,readers)) == job_store(~Cursor,job) : Gemm.Store}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  {==}

def stored_columns(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+writer: GemmLaws.Writer,+job: Gemm.Job<Cursor>,+readers: Readers) ->
  {GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers)) == job_columns(~Cursor,writer,job) : Nat}:
  Equal.cong(Gemm.Store,Nat,store => U32.to_nat(GemmLaws.columns(writer,store)),GemmWrites.block_store(job_block(~Cursor,~advance,~order,job,readers)),job_store(~Cursor,job),
    block_store_is(~Cursor,~advance,~order,job,readers))

def stored_rows(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+job: Gemm.Job<Cursor>,+readers: Readers) ->
  {GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,readers)) == job_rows(~Cursor,job) : Nat}:
  Equal.cong(Gemm.Store,Nat,store => U32.to_nat(GemmWrites.store_rows(store)),GemmWrites.block_store(job_block(~Cursor,~advance,~order,job,readers)),job_store(~Cursor,job),
    block_store_is(~Cursor,~advance,~order,job,readers))

# The address and the value of a stored cell, as the view writes them.
def stored_address(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+writer: GemmLaws.Writer,+job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,readers))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers))) == True{} : Bool}) ->
  {cell_address(~Cursor,job,row,column) ==
    Copy.address(GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,job,readers)),
      Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers))),column)) : U32}:
  Gemm.Job{+first,+depth,+cursor,Gemm.Store{+base,+row_stride,+lane_stride,+rows,+lanes}} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  +block = job_block(~Cursor,~advance,~order,Gemm.Job{first,depth,cursor,Gemm.Store{base,row_stride,lane_stride,rows,lanes}},Readers{row_tiles,lane_tiles,bias})
  +row_word = U32.from_nat(row)
  +column_word = U32.from_nat(column)
  %GemmWrites.cell_address(writer,block,row,column,row_inside,column_inside) : {(base + row_word * row_stride + column_word * lane_stride : U32) == _ : U32}
  %Equal.sym(U32,GemmWrites.scaled(GemmWrites.scaled(base,row,row_stride),column,lane_stride),(GemmWrites.scaled(base,row,row_stride) + column_word * lane_stride : U32),
    GemmWrites.scaled_exact(GemmWrites.scaled(base,row,row_stride),column,lane_stride)) :
    {(base + row_word * row_stride + column_word * lane_stride : U32) == _ : U32}
  %Equal.sym(U32,GemmWrites.scaled(base,row,row_stride),(base + row_word * row_stride : U32),GemmWrites.scaled_exact(base,row,row_stride)) :
    {(base + row_word * row_stride + column_word * lane_stride : U32) == (_ + column_word * lane_stride : U32) : U32}
  {==}

def stored_value(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+column: Nat,
  +rows_small: {Nat.is_le(job_rows(~Cursor,job),8n) == True{} : Bool},+columns_small: {Nat.is_le(job_columns(~Cursor,writer,job),8n) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,readers))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers))) == True{} : Bool}) ->
  {cell_value(~Cursor,~advance,~order,~value,~close,writer,job,readers,row,column) ==
    GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,job,readers),
      Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers))),column)) : GemmLaws.Element(writer)}:
  Gemm.Job{+first,+depth,+cursor,Gemm.Store{+base,+row_stride,+lane_stride,+rows,+lanes}} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  +store = {Gemm.Store{base,row_stride,lane_stride,rows,lanes} : Gemm.Store}
  +block = job_block(~Cursor,~advance,~order,Gemm.Job{first,depth,cursor,store},Readers{row_tiles,lane_tiles,bias})
  +taps = GemmSpec.taps(~Cursor,~advance,depth,cursor)
  +source = {GemmSpec.Source{row_tiles,lane_tiles} : GemmSpec.Source}
  +small = Bounds.nat_trans(row,U32.to_nat(rows),8n,row_inside,rows_small)
  +column_small = Bounds.nat_trans(column,GemmWrites.columns_of(writer,block),8n,column_inside,columns_small)
  +values = GemmSpec.select_row(row,GemmWrites.block_accumulators(block))
  Equal.trans(GemmLaws.Element(writer),cell_value(~Cursor,~advance,~order,~value,~close,writer,Gemm.Job{first,depth,cursor,store},Readers{row_tiles,lane_tiles,bias},row,column),
    GemmWrites.cell_value(~value,~close,writer,block,row,column),
    GemmWrites.element(~value,~close,~writer,block,Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,block)),column)),
    GemmWrites.formed_finished(~value,~close,~order,writer,first,store,bias,taps,source,row,column,values,column_small,
      GemmSpec.reduced_row(~Cursor,~advance,~order,depth,row_tiles,lane_tiles,cursor,row,small)),
    GemmWrites.cell_element(~value,~close,~writer,block,row,column,column_inside))

# The stores of a job are the writes of its view.
def job_view(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers,+rows_small: {Nat.is_le(job_rows(~Cursor,job),8n) == True{} : Bool},
  +columns_small: {Nat.is_le(job_columns(~Cursor,writer,job),8n) == True{} : Bool}) ->
  {job_writes(~Cursor,~advance,~order,~writer,~value,~close,job,readers) ==
    ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
      Nat.mul(GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,readers)),GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,readers))),
      GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,job,readers)),job_block(~Cursor,~advance,~order,job,readers)) : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  Gemm.Job{+row,+depth,+cursor,Gemm.Store{+base,+row_stride,+lane_stride,+rows,+lanes}} = job
  Readers{+row_tiles,+lane_tiles,+bias} = readers
  GemmWrites.view_writes(~value,~close,~writer,row,base,row_stride,lane_stride,rows,lanes,
    GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,depth,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()})),bias,rows_small,columns_small)

# ---- Every cell holds what the caller says of it ----

def plane_view(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,
  ~readers: @+environment: Environment -> Readers,+environment: Environment) -> Views.View:
  GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,job(environment),readers(environment)))

def plane_block(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,
  ~readers: @+environment: Environment -> Readers,+environment: Environment) -> GemmWrites.Block:
  job_block(~Cursor,~advance,~order,job(environment),readers(environment))

def plane_rows(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,
  ~readers: @+environment: Environment -> Readers,+environment: Environment) -> Nat:
  GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job(environment),readers(environment)))

def plane_columns(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,
  ~readers: @+environment: Environment -> Readers,+environment: Environment) -> Nat:
  GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job(environment),readers(environment)))

def plane_cell(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,~Holds: @+target: Target -> U32 -> GemmLaws.Element(writer) -> Data,
  ~job: @+environment: Environment -> Gemm.Job<Cursor>,~readers: @+environment: Environment -> Readers,
  ~rows_small: @+environment: Environment -> {Nat.is_le(job_rows(~Cursor,job(environment)),8n) == True{} : Bool},
  ~columns_small: @+environment: Environment -> {Nat.is_le(job_columns(~Cursor,writer,job(environment)),8n) == True{} : Bool},
  ~cell: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job(environment))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job(environment))) == True{} : Bool} ->
    Holds(target(environment),cell_address(~Cursor,job(environment),row,column),cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),readers(environment),row,column)),
  +environment: Environment,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,plane_rows(~Cursor,~advance,~order,~Environment,~job,~readers,environment)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment)) == True{} : Bool}) ->
  Holds(target(environment),Copy.address(plane_view(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment),
      Nat.add(Nat.mul(row,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment)),column)),
    GemmWrites.element(~value,~close,~writer,plane_block(~Cursor,~advance,~order,~Environment,~job,~readers,environment),
      Nat.add(Nat.mul(row,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment)),column))):
  +current = job(environment)
  +sources = readers(environment)
  +row_below = {%stored_rows(~Cursor,~advance,~order,current,sources) : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(row,job_rows(~Cursor,current)) == True{} : Bool}}
  +column_below = {%stored_columns(~Cursor,~advance,~order,writer,current,sources) : {Nat.is_lt(column,_) == True{} : Bool} column_inside :
    {Nat.is_lt(column,job_columns(~Cursor,writer,current)) == True{} : Bool}}
  %stored_address(~Cursor,~advance,~order,writer,current,sources,row,column,row_inside,column_inside) :
    Holds(target(environment),_,GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,sources),
      Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,sources))),column)))
  %stored_value(~Cursor,~advance,~order,~writer,~value,~close,current,sources,row,column,rows_small(environment),columns_small(environment),row_inside,column_inside) :
    Holds(target(environment),cell_address(~Cursor,current,row,column),_)
  cell(environment,row,column,row_below,column_below)

# Every write of a job holds what the caller says of its cells.
def job_every(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,~Holds: @+target: Target -> U32 -> GemmLaws.Element(writer) -> Data,
  ~job: @+environment: Environment -> Gemm.Job<Cursor>,~readers: @+environment: Environment -> Readers,
  ~rows_small: @+environment: Environment -> {Nat.is_le(job_rows(~Cursor,job(environment)),8n) == True{} : Bool},
  ~columns_small: @+environment: Environment -> {Nat.is_le(job_columns(~Cursor,writer,job(environment)),8n) == True{} : Bool},
  ~cell: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job(environment))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job(environment))) == True{} : Bool} ->
    Holds(target(environment),cell_address(~Cursor,job(environment),row,column),cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),readers(environment),row,column)),
  +environment: Environment) ->
  Every.Every(~GemmLaws.Element(writer),~Target,~Holds,target(environment),job_writes(~Cursor,~advance,~order,~writer,~value,~close,job(environment),readers(environment))):
  %Equal.sym(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,job_writes(~Cursor,~advance,~order,~writer,~value,~close,job(environment),readers(environment)),
    ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
      Nat.mul(plane_rows(~Cursor,~advance,~order,~Environment,~job,~readers,environment),plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment)),
      plane_view(~Cursor,~advance,~order,~writer,~Environment,~job,~readers,environment),plane_block(~Cursor,~advance,~order,~Environment,~job,~readers,environment)),
    job_view(~Cursor,~advance,~order,~writer,~value,~close,job(environment),readers(environment),rows_small(environment),columns_small(environment))) :
    Every.Every(~GemmLaws.Element(writer),~Target,~Holds,target(environment),_)
  ViewWrites.every_plane(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),~Environment,~Target,~target,~Holds,
    ~plane_view(~Cursor,~advance,~order,~writer,~Environment,~job,~readers),~plane_block(~Cursor,~advance,~order,~Environment,~job,~readers),
    ~plane_rows(~Cursor,~advance,~order,~Environment,~job,~readers),~plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~readers),
    ~plane_cell(~Cursor,~advance,~order,~writer,~value,~close,~Environment,~Target,~target,~Holds,~job,~readers,~rows_small,~columns_small,~cell),environment)

# ---- A cell of a job is written ----

# The writes of a job contain the address of each of its cells.
def job_contains(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+readers: Readers,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,job_rows(~Cursor,job)) == True{} : Bool},+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job)) == True{} : Bool},
  +rows_small: {Nat.is_le(job_rows(~Cursor,job),8n) == True{} : Bool},+columns_small: {Nat.is_le(job_columns(~Cursor,writer,job),8n) == True{} : Bool}) ->
  {WriteReadback.contains(~GemmLaws.Element(writer),cell_address(~Cursor,job,row,column),job_writes(~Cursor,~advance,~order,~writer,~value,~close,job,readers)) == True{} : Bool}:
  +block = job_block(~Cursor,~advance,~order,job,readers)
  +row_stored = {
    %Equal.sym(Nat,GemmWrites.rows_of(block),job_rows(~Cursor,job),stored_rows(~Cursor,~advance,~order,job,readers)) : {Nat.is_lt(row,_) == True{} : Bool}
    row_inside : {Nat.is_lt(row,GemmWrites.rows_of(block)) == True{} : Bool}}
  +column_stored = {
    %Equal.sym(Nat,GemmWrites.columns_of(writer,block),job_columns(~Cursor,writer,job),stored_columns(~Cursor,~advance,~order,writer,job,readers)) : {Nat.is_lt(column,_) == True{} : Bool}
    column_inside : {Nat.is_lt(column,GemmWrites.columns_of(writer,block)) == True{} : Bool}}
  +written = ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
    Nat.mul(GemmWrites.rows_of(block),GemmWrites.columns_of(writer,block)),GemmWrites.block_view(writer,block),block)
  %Equal.sym(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,job_writes(~Cursor,~advance,~order,~writer,~value,~close,job,readers),written,
    job_view(~Cursor,~advance,~order,~writer,~value,~close,job,readers,rows_small,columns_small)) :
    {WriteReadback.contains(~GemmLaws.Element(writer),cell_address(~Cursor,job,row,column),_) == True{} : Bool}
  %Equal.sym(U32,cell_address(~Cursor,job,row,column),
    Copy.address(GemmWrites.block_view(writer,block),Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,block)),column)),
    stored_address(~Cursor,~advance,~order,writer,job,readers,row,column,row_stored,column_stored)) :
    {WriteReadback.contains(~GemmLaws.Element(writer),_,written) == True{} : Bool}
  ViewWrites.covered_plane(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),GemmWrites.rows_of(block),GemmWrites.columns_of(writer,block),
    row,column,GemmWrites.block_view(writer,block),block,row_stored,column_stored)

# ---- Two readers whose cells agree store the same writes ----

# The store of a block does not depend on the trees it reads.
def same_layout(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  +job: Gemm.Job<Cursor>,+left: Readers,+right: Readers) ->
  {ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
      Nat.mul(GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,right)),GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,right))),
      GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,job,right)),job_block(~Cursor,~advance,~order,job,right)) ==
    ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
      Nat.mul(GemmWrites.rows_of(job_block(~Cursor,~advance,~order,job,left)),GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,job,left))),
      GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,job,left)),job_block(~Cursor,~advance,~order,job,right)) : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Readers{+left_weights,+left_tiles,+left_bias} = left
  Readers{+right_weights,+right_tiles,+right_bias} = right
  {==}

def agree_cell(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,~left: @+environment: Environment -> Readers,~right: @+environment: Environment -> Readers,
  ~rows_small: @+environment: Environment -> {Nat.is_le(job_rows(~Cursor,job(environment)),8n) == True{} : Bool},
  ~columns_small: @+environment: Environment -> {Nat.is_le(job_columns(~Cursor,writer,job(environment)),8n) == True{} : Bool},
  ~same: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job(environment))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job(environment))) == True{} : Bool} ->
    {cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),left(environment),row,column) ==
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),right(environment),row,column) : GemmLaws.Element(writer)},
  +environment: Environment,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,plane_rows(~Cursor,~advance,~order,~Environment,~job,~left,environment)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~left,environment)) == True{} : Bool}) ->
  {GemmWrites.element(~value,~close,~writer,plane_block(~Cursor,~advance,~order,~Environment,~job,~left,environment),
      Nat.add(Nat.mul(row,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~left,environment)),column)) ==
    GemmWrites.element(~value,~close,~writer,plane_block(~Cursor,~advance,~order,~Environment,~job,~right,environment),
      Nat.add(Nat.mul(row,plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~left,environment)),column)) : GemmLaws.Element(writer)}:
  +current = job(environment)
  +first = left(environment)
  +second = right(environment)
  +bounded = {%stored_rows(~Cursor,~advance,~order,current,first) : {Nat.is_lt(row,_) == True{} : Bool} row_inside :
    {Nat.is_lt(row,job_rows(~Cursor,current)) == True{} : Bool}}
  +column_bounded = {%stored_columns(~Cursor,~advance,~order,writer,current,first) : {Nat.is_lt(column,_) == True{} : Bool} column_inside :
    {Nat.is_lt(column,job_columns(~Cursor,writer,current)) == True{} : Bool}}
  +second_row = {%Equal.sym(Nat,GemmWrites.rows_of(job_block(~Cursor,~advance,~order,current,second)),job_rows(~Cursor,current),
      stored_rows(~Cursor,~advance,~order,current,second)) : {Nat.is_lt(row,_) == True{} : Bool} bounded :
    {Nat.is_lt(row,GemmWrites.rows_of(job_block(~Cursor,~advance,~order,current,second))) == True{} : Bool}}
  +second_column = {%Equal.sym(Nat,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second)),job_columns(~Cursor,writer,current),
      stored_columns(~Cursor,~advance,~order,writer,current,second)) : {Nat.is_lt(column,_) == True{} : Bool} column_bounded :
    {Nat.is_lt(column,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second))) == True{} : Bool}}
  +columns_equal = Equal.trans(Nat,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,first)),job_columns(~Cursor,writer,current),
    GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second)),stored_columns(~Cursor,~advance,~order,writer,current,first),
    Equal.sym(Nat,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second)),job_columns(~Cursor,writer,current),stored_columns(~Cursor,~advance,~order,writer,current,second)))
  %Equal.sym(Nat,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,first)),GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second)),columns_equal) :
    {GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,first),
        Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,first))),column)) ==
      GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,second),Nat.add(Nat.mul(row,_),column)) : GemmLaws.Element(writer)}
  Equal.trans(GemmLaws.Element(writer),
    GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,first),
      Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,first))),column)),
    cell_value(~Cursor,~advance,~order,~value,~close,writer,current,first,row,column),
    GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,second),
      Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second))),column)),
    Equal.sym(GemmLaws.Element(writer),cell_value(~Cursor,~advance,~order,~value,~close,writer,current,first,row,column),
      GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,first),
        Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,first))),column)),
      stored_value(~Cursor,~advance,~order,~writer,~value,~close,current,first,row,column,rows_small(environment),columns_small(environment),row_inside,column_inside)),
    Equal.trans(GemmLaws.Element(writer),cell_value(~Cursor,~advance,~order,~value,~close,writer,current,first,row,column),
      cell_value(~Cursor,~advance,~order,~value,~close,writer,current,second,row,column),
      GemmWrites.element(~value,~close,~writer,job_block(~Cursor,~advance,~order,current,second),
        Nat.add(Nat.mul(row,GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second))),column)),
      same(environment,row,column,bounded,column_bounded),
      stored_value(~Cursor,~advance,~order,~writer,~value,~close,current,second,row,column,rows_small(environment),columns_small(environment),second_row,second_column)))

# A job stores the same writes from two readers whose cells agree.
def job_agree(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~Environment: Data,~job: @+environment: Environment -> Gemm.Job<Cursor>,~left: @+environment: Environment -> Readers,~right: @+environment: Environment -> Readers,
  ~rows_small: @+environment: Environment -> {Nat.is_le(job_rows(~Cursor,job(environment)),8n) == True{} : Bool},
  ~columns_small: @+environment: Environment -> {Nat.is_le(job_columns(~Cursor,writer,job(environment)),8n) == True{} : Bool},
  ~same: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job(environment))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job(environment))) == True{} : Bool} ->
    {cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),left(environment),row,column) ==
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job(environment),right(environment),row,column) : GemmLaws.Element(writer)},
  +environment: Environment) ->
  {job_writes(~Cursor,~advance,~order,~writer,~value,~close,job(environment),left(environment)) ==
    job_writes(~Cursor,~advance,~order,~writer,~value,~close,job(environment),right(environment)) : List<&2,Writes.Assignment<GemmLaws.Element(writer)>>}:
  +current = job(environment)
  +first = left(environment)
  +second = right(environment)
  +count = Nat.mul(plane_rows(~Cursor,~advance,~order,~Environment,~job,~left,environment),plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~left,environment))
  +view = plane_view(~Cursor,~advance,~order,~writer,~Environment,~job,~left,environment)
  +left_view = ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),count,view,job_block(~Cursor,~advance,~order,current,first))
  +right_view = ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),count,view,job_block(~Cursor,~advance,~order,current,second))
  Equal.trans(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,job_writes(~Cursor,~advance,~order,~writer,~value,~close,current,first),left_view,
    job_writes(~Cursor,~advance,~order,~writer,~value,~close,current,second),
    job_view(~Cursor,~advance,~order,~writer,~value,~close,current,first,rows_small(environment),columns_small(environment)),
    Equal.trans(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,left_view,right_view,job_writes(~Cursor,~advance,~order,~writer,~value,~close,current,second),
      ViewWrites.agreeing_plane(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),~Environment,
        ~plane_view(~Cursor,~advance,~order,~writer,~Environment,~job,~left),~plane_block(~Cursor,~advance,~order,~Environment,~job,~left),~plane_block(~Cursor,~advance,~order,~Environment,~job,~right),
        ~plane_rows(~Cursor,~advance,~order,~Environment,~job,~left),~plane_columns(~Cursor,~advance,~order,~writer,~Environment,~job,~left),
        ~agree_cell(~Cursor,~advance,~order,~writer,~value,~close,~Environment,~job,~left,~right,~rows_small,~columns_small,~same),environment),
      Equal.sym(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,job_writes(~Cursor,~advance,~order,~writer,~value,~close,current,second),right_view,
        Equal.trans(List<&2,Writes.Assignment<GemmLaws.Element(writer)>>,job_writes(~Cursor,~advance,~order,~writer,~value,~close,current,second),
          ViewWrites.writes(~GemmLaws.Element(writer),~GemmWrites.Block,~GemmWrites.element(~value,~close,~writer),
            Nat.mul(GemmWrites.rows_of(job_block(~Cursor,~advance,~order,current,second)),GemmWrites.columns_of(writer,job_block(~Cursor,~advance,~order,current,second))),
            GemmWrites.block_view(writer,job_block(~Cursor,~advance,~order,current,second)),job_block(~Cursor,~advance,~order,current,second)),right_view,
          job_view(~Cursor,~advance,~order,~writer,~value,~close,current,second,rows_small(environment),columns_small(environment)),
          same_layout(~Cursor,~advance,~order,~writer,~value,~close,current,first,second)))))

# ---- Readback by cell keys ----

# A caller names the cells of a run by the keys of a layout: the layout
# gives the run's plan, readers and job count, and for each valid key the
# address and the value it asks for. When every cell of every job is a
# valid key, every valid key lies below a bound, two valid keys at one
# address ask for one value, and every valid key is a cell of some job, the
# run reads back the value of each valid key.

# A write that stores the value of a valid key at the key's address.
type Keyed<-E: Data,-Key: Data,-Valid: Key -> Data,-word: Key -> U32,-wanted: Key -> E,-address: U32,-stored: E> is Data:
  Keyed{key: Key,valid: Valid(key),address_exact: {address == word(key) : U32},value_exact: {stored == wanted(key) : E}}

def Located(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> E,+layout: Layout,address: U32,stored: E) -> Data:
  Keyed<E,Key,(key => Valid(layout,key)),(key => word(layout,key)),(key => wanted(layout,key)),address,stored>

# The cell of a job that stores a key.
type Found<-Cursor: Data,-writer: GemmLaws.Writer,-job: Nat -> Gemm.Job<Cursor>,-count: Nat,-address: U32> is Data:
  Found{index: Nat,row: Nat,column: Nat,inside: {Nat.is_lt(index,count) == True{} : Bool},
    row_inside: {Nat.is_lt(row,job_rows(~Cursor,job(index))) == True{} : Bool},column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job(index))) == True{} : Bool},
    address_exact: {cell_address(~Cursor,job(index),row,column) == address : U32}}

def Locate(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~writer: GemmLaws.Writer,~Layout: Data,~plan: @+layout: Layout -> Plan,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~word: @+layout: Layout -> @+key: Key -> U32,+layout: Layout,+key: Key) -> Data:
  Found<Cursor,writer,(index => job_at(plan(layout),U32.from_nat(index))),jobs(layout),word(layout,key)>

def found_index(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~writer: GemmLaws.Writer,~Layout: Data,~plan: @+layout: Layout -> Plan,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~word: @+layout: Layout -> @+key: Key -> U32,+layout: Layout,+key: Key,
  +located: Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key)) -> Nat:
  Found{index,row,column,inside,row_inside,column_inside,address_exact} = located
  index

def found_inside(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~writer: GemmLaws.Writer,~Layout: Data,~plan: @+layout: Layout -> Plan,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~word: @+layout: Layout -> @+key: Key -> U32,+layout: Layout,+key: Key,
  +located: Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key)) ->
  {Nat.is_lt(found_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key,located),jobs(layout)) == True{} : Bool}:
  Found{+index,+row,+column,+inside,+row_inside,+column_inside,+address_exact} = located
  inside

# The writes of a layout's run.
def layout_writes(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,+layout: Layout) -> List<&2,Writes.Assignment<GemmLaws.Element(writer)>>:
  all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,jobs(layout),0,plan(layout),readers(layout))

# ---- Every write is a valid key ----

# Job index of a layout.
type Placement<-Layout: Data,-jobs: Layout -> Nat> is Data:
  Placement{layout: Layout,index: Nat,inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool}}

def Placed(~Layout: Data,~jobs: @+layout: Layout -> Nat) -> Data:
  Placement<Layout,(layout => jobs(layout))>

def placement_layout(~Layout: Data,~jobs: @+layout: Layout -> Nat,+placement: Placed(~Layout,~jobs)) -> Layout:
  Placement{layout,index,inside} = placement
  layout

def placement_job(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~Layout: Data,~plan: @+layout: Layout -> Plan,~jobs: @+layout: Layout -> Nat,
  +placement: Placed(~Layout,~jobs)) -> Gemm.Job<Cursor>:
  Placement{+layout,+index,inside} = placement
  job_at(plan(layout),U32.from_nat(index))

def placement_readers(~Layout: Data,~readers: @+layout: Layout -> Readers,~jobs: @+layout: Layout -> Nat,+placement: Placed(~Layout,~jobs)) -> Readers:
  Placement{+layout,index,inside} = placement
  readers(layout)

def placement_rows(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~Layout: Data,~plan: @+layout: Layout -> Plan,~jobs: @+layout: Layout -> Nat,
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  +placement: Placed(~Layout,~jobs)) -> {Nat.is_le(job_rows(~Cursor,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement)),8n) == True{} : Bool}:
  Placement{+layout,+index,+inside} = placement
  rows_small(layout,index,inside)

def placement_columns(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~writer: GemmLaws.Writer,~Layout: Data,~plan: @+layout: Layout -> Plan,
  ~jobs: @+layout: Layout -> Nat,
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  +placement: Placed(~Layout,~jobs)) -> {Nat.is_le(job_columns(~Cursor,writer,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement)),8n) == True{} : Bool}:
  Placement{+layout,+index,+inside} = placement
  columns_small(layout,index,inside)

def placement_cell(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> GemmLaws.Element(writer),
  ~cell: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} -> @+row: Nat -> @+column: Nat ->
    @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,layout,cell_address(~Cursor,job_at(plan(layout),U32.from_nat(index)),row,column),
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job_at(plan(layout),U32.from_nat(index)),readers(layout),row,column)),
  +placement: Placed(~Layout,~jobs),+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,job_rows(~Cursor,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement))) == True{} : Bool}) ->
  Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,placement_layout(~Layout,~jobs,placement),
    cell_address(~Cursor,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement),row,column),
    cell_value(~Cursor,~advance,~order,~value,~close,writer,placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,placement),placement_readers(~Layout,~readers,~jobs,placement),
      row,column)):
  Placement{+layout,+index,+inside} = placement
  cell(layout,index,inside,row,column,row_inside,column_inside)

# The jobs from a position run the layout's plan over its readers.
def Remaining(~Plan: Data,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,~jobs: @+layout: Layout -> Nat,
  +layout: Layout,rest: Nat,position: Nat,context: Plan,current: Readers) -> Data:
  Gemm.Both<{Nat.add(position,rest) == jobs(layout) : Nat},Gemm.Both<{context == plan(layout) : Plan},{current == readers(layout) : Readers}>>

def target_layout(~Layout: Data,+layout: Layout) -> Layout:
  layout

def every_step(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> GemmLaws.Element(writer),
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~cell: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} -> @+row: Nat -> @+column: Nat ->
    @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,layout,cell_address(~Cursor,job_at(plan(layout),U32.from_nat(index)),row,column),
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job_at(plan(layout),U32.from_nat(index)),readers(layout),row,column)),
  +layout: Layout,+rest: Nat,+position: Nat,+context: Plan,+current: Readers,left: Remaining(~Plan,~Layout,~plan,~readers,~jobs,layout,1n+rest,position,context,current)) ->
  Every.Every(~GemmLaws.Element(writer),~Layout,~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),target_layout(~Layout,layout),
    step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,U32.from_nat(position),context,current)):
  Gemm.Both{room,same} = left
  Gemm.Both{same_context,same_readers} = same
  %Equal.sym(Plan,context,plan(layout),same_context) :
    Every.Every(~GemmLaws.Element(writer),~Layout,~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),layout,
      job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(_,U32.from_nat(position)),current))
  %Equal.sym(Readers,current,readers(layout),same_readers) :
    Every.Every(~GemmLaws.Element(writer),~Layout,~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),layout,
      job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan(layout),U32.from_nat(position)),_))
  job_every(~Cursor,~advance,~order,~writer,~value,~close,~Placed(~Layout,~jobs),~Layout,~placement_layout(~Layout,~jobs),
    ~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),~placement_job(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs),~placement_readers(~Layout,~readers,~jobs),
    ~placement_rows(~Plan,~Cursor,~job_at,~Layout,~plan,~jobs,~rows_small),~placement_columns(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~columns_small),
    ~placement_cell(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~Valid,~word,~wanted,~cell),
    Placement{layout,position,Order.below_complement(position,rest,jobs(layout),room)})

def remaining_kept(~Plan: Data,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,~jobs: @+layout: Layout -> Nat,
  +layout: Layout,+rest: Nat,+position: Nat,+context: Plan,+current: Readers,left: Remaining(~Plan,~Layout,~plan,~readers,~jobs,layout,1n+rest,position,context,current)) ->
  Remaining(~Plan,~Layout,~plan,~readers,~jobs,layout,rest,1n+position,context,kept(~Plan,U32.from_nat(position),context,current)):
  Gemm.Both{room,same} = left
  Gemm.Both{Equal.trans(Nat,1n+Nat.add(position,rest),Nat.add(position,1n+rest),jobs(layout),
    Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)),room),same}

# Every write of a layout's run is a valid key.
def all_every(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> GemmLaws.Element(writer),
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~cell: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} -> @+row: Nat -> @+column: Nat ->
    @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,layout,cell_address(~Cursor,job_at(plan(layout),U32.from_nat(index)),row,column),
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job_at(plan(layout),U32.from_nat(index)),readers(layout),row,column)),
  +layout: Layout) ->
  Every.Every(~GemmLaws.Element(writer),~Layout,~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),layout,
    layout_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,layout)):
  Fold.every(~GemmLaws.Element(writer),~Plan,~Readers,~step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),~kept(~Plan),~Layout,~Layout,~target_layout(~Layout),
    ~Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted),~Remaining(~Plan,~Layout,~plan,~readers,~jobs),
    ~every_step(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~Valid,~word,~wanted,~rows_small,~columns_small,~cell),
    ~remaining_kept(~Plan,~Layout,~plan,~readers,~jobs),layout,jobs(layout),0n,plan(layout),readers(layout),Gemm.Both{{==},Gemm.Both{{==},{==}}})

# ---- Every write lies inside the output ----

type Output<-Layout: Data,-bound: Layout -> Nat> is Data:
  Output{layout: Layout,capacity: U32,room: {Nat.is_le(bound(layout),U32.to_nat(capacity)) == True{} : Bool}}

def Bounded(~Layout: Data,~bound: @+layout: Layout -> Nat) -> Data:
  Output<Layout,(layout => bound(layout))>

def output_layout(~Layout: Data,~bound: @+layout: Layout -> Nat,+output: Bounded(~Layout,~bound)) -> Layout:
  Output{layout,capacity,room} = output
  layout

def output_capacity(~Layout: Data,~bound: @+layout: Layout -> Nat,+output: Bounded(~Layout,~bound)) -> U32:
  Output{layout,capacity,room} = output
  capacity

def located_inside(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> E,~bound: @+layout: Layout -> Nat,
  ~below: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> {Nat.is_lt(U32.to_nat(word(layout,key)),bound(layout)) == True{} : Bool},
  +output: Bounded(~Layout,~bound),+address: U32,+stored: E,holds: Located(~E,~Layout,~Key,~Valid,~word,~wanted,output_layout(~Layout,~bound,output),address,stored)) ->
  {U32.is_lt(address,output_capacity(~Layout,~bound,output)) == True{} : Bool}:
  Output{+layout,+capacity,+room} = output
  Keyed{+key,+valid,+address_exact,+value_exact} = holds
  +cell = word(layout,key)
  %Equal.sym(U32,address,cell,address_exact) : {U32.is_lt(_,capacity) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(cell,capacity),Nat.is_lt(U32.to_nat(cell),U32.to_nat(capacity)),Comparison.lt(cell,capacity)) : {_ == True{} : Bool}
  Bounds.nat_trans(U32.to_nat(cell),bound(layout),U32.to_nat(capacity),below(layout,key,valid),room)

# ---- A write at a key's address stores the key's value ----

type Query<-Layout: Data,-Key: Data,-Valid: Layout -> Key -> Data> is Data:
  Query{layout: Layout,key: Key,valid: Valid(layout,key)}

def Asked(~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data) -> Data:
  Query<Layout,Key,(layout => key => Valid(layout,key))>

def query_layout(~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,+query: Asked(~Layout,~Key,~Valid)) -> Layout:
  Query{layout,key,valid} = query
  layout

def query_word(~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  +query: Asked(~Layout,~Key,~Valid)) -> U32:
  Query{+layout,+key,valid} = query
  word(layout,key)

def query_wanted(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~wanted: @+layout: Layout -> @+key: Key -> E,
  +query: Asked(~Layout,~Key,~Valid)) -> E:
  Query{+layout,+key,valid} = query
  wanted(layout,key)

def keyed_matches(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> E,
  ~same: @+layout: Layout -> @+key: Key -> @+other: Key -> @+valid: Valid(layout,key) -> @+other_valid: Valid(layout,other) ->
    @+equal: {word(layout,key) == word(layout,other) : U32} -> {wanted(layout,key) == wanted(layout,other) : E},
  +layout: Layout,+key: Key,+valid: Valid(layout,key),+other: Key,+other_valid: Valid(layout,other),+address: U32,+stored: E,
  +address_exact: {address == word(layout,other) : U32},+value_exact: {stored == wanted(layout,other) : E},equal: Bool,
  compared: {U32.is_eq(address,word(layout,key)) == equal : Bool}) -> Boolean.When(equal,{stored == wanted(layout,key) : E}):
  match equal:
    case False{}: Unit{}
    case True{}:
      +words = Equal.trans(U32,word(layout,other),address,word(layout,key),Equal.sym(U32,address,word(layout,other),address_exact),
        Arithmetic.equal_of_is_eq(address,word(layout,key),compared))
      Equal.trans(E,stored,wanted(layout,other),wanted(layout,key),value_exact,same(layout,other,key,other_valid,valid,words))

def located_matches(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> E,
  ~same: @+layout: Layout -> @+key: Key -> @+other: Key -> @+valid: Valid(layout,key) -> @+other_valid: Valid(layout,other) ->
    @+equal: {word(layout,key) == word(layout,other) : U32} -> {wanted(layout,key) == wanted(layout,other) : E},
  +query: Asked(~Layout,~Key,~Valid),+address: U32,+stored: E,holds: Located(~E,~Layout,~Key,~Valid,~word,~wanted,query_layout(~Layout,~Key,~Valid,query),address,stored)) ->
  Boolean.When(U32.is_eq(address,query_word(~Layout,~Key,~Valid,~word,query)),{stored == query_wanted(~E,~Layout,~Key,~Valid,~wanted,query) : E}):
  Query{+layout,+key,+valid} = query
  Keyed{+other,+other_valid,+address_exact,+value_exact} = holds
  keyed_matches(~E,~Layout,~Key,~Valid,~word,~wanted,~same,layout,key,valid,other,other_valid,address,stored,address_exact,value_exact,U32.is_eq(address,word(layout,key)),{==})

# ---- The writes contain each valid key's address ----

def query_index(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~writer: GemmLaws.Writer,~Layout: Data,~plan: @+layout: Layout -> Plan,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~found: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key),
  +query: Asked(~Layout,~Key,~Valid)) -> Nat:
  Query{+layout,+key,+valid} = query
  found_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key,found(layout,key,valid))

def Scheduled(~Plan: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  +query: Asked(~Layout,~Key,~Valid),rest: Nat,position: Nat,context: Plan,current: Readers) -> Data:
  Gemm.Both<{context == plan(query_layout(~Layout,~Key,~Valid,query)) : Plan},{current == readers(query_layout(~Layout,~Key,~Valid,query)) : Readers}>

def scheduled_kept(~Plan: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  +query: Asked(~Layout,~Key,~Valid),+rest: Nat,+position: Nat,+context: Plan,+current: Readers,
  schedule: Scheduled(~Plan,~Layout,~Key,~Valid,~plan,~readers,query,1n+rest,position,context,current)) ->
  Scheduled(~Plan,~Layout,~Key,~Valid,~plan,~readers,query,rest,1n+position,context,kept(~Plan,U32.from_nat(position),context,current)):
  schedule

def found_contains(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  +layout: Layout,+key: Key,+located: Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key)) ->
  {WriteReadback.contains(~GemmLaws.Element(writer),word(layout,key),job_writes(~Cursor,~advance,~order,~writer,~value,~close,
    job_at(plan(layout),U32.from_nat(found_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key,located))),readers(layout))) == True{} : Bool}:
  Found{+index,+row,+column,+inside,+row_inside,+column_inside,+address_exact} = located
  +job = job_at(plan(layout),U32.from_nat(index))
  %address_exact : {WriteReadback.contains(~GemmLaws.Element(writer),_,job_writes(~Cursor,~advance,~order,~writer,~value,~close,job,readers(layout))) == True{} : Bool}
  job_contains(~Cursor,~advance,~order,~writer,~value,~close,job,readers(layout),row,column,row_inside,column_inside,rows_small(layout,index,inside),columns_small(layout,index,inside))

def scheduled_found(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~found: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key),
  +query: Asked(~Layout,~Key,~Valid),+rest: Nat,+context: Plan,+current: Readers,
  schedule: Scheduled(~Plan,~Layout,~Key,~Valid,~plan,~readers,query,1n+rest,
    query_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~Valid,~word,~found,query),context,current)) ->
  {WriteReadback.contains(~GemmLaws.Element(writer),query_word(~Layout,~Key,~Valid,~word,query),step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,
    U32.from_nat(query_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~Valid,~word,~found,query)),context,current)) == True{} : Bool}:
  Query{+layout,+key,+valid} = query
  Gemm.Both{same_context,same_readers} = schedule
  +index = found_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key,found(layout,key,valid))
  %Equal.sym(Plan,context,plan(layout),same_context) :
    {WriteReadback.contains(~GemmLaws.Element(writer),word(layout,key),job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(_,U32.from_nat(index)),current)) == True{} : Bool}
  %Equal.sym(Readers,current,readers(layout),same_readers) :
    {WriteReadback.contains(~GemmLaws.Element(writer),word(layout,key),job_writes(~Cursor,~advance,~order,~writer,~value,~close,job_at(plan(layout),U32.from_nat(index)),_)) == True{} :
      Bool}
  found_contains(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~word,~rows_small,~columns_small,layout,key,found(layout,key,valid))

def all_contains(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~found: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key),
  +query: Asked(~Layout,~Key,~Valid)) ->
  {WriteReadback.contains(~GemmLaws.Element(writer),query_word(~Layout,~Key,~Valid,~word,query),
    layout_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,query_layout(~Layout,~Key,~Valid,query))) == True{} : Bool}:
  Query{+layout,+key,+valid} = query
  +whole = {Query{layout,key,valid} : Asked(~Layout,~Key,~Valid)}
  Fold.contains_at(~GemmLaws.Element(writer),~Plan,~Readers,~step_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close),~kept(~Plan),~Asked(~Layout,~Key,~Valid),
    ~query_word(~Layout,~Key,~Valid,~word),~query_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~Valid,~word,~found),
    ~Scheduled(~Plan,~Layout,~Key,~Valid,~plan,~readers),~scheduled_kept(~Plan,~Layout,~Key,~Valid,~plan,~readers),
    ~scheduled_found(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~Valid,~word,~rows_small,~columns_small,~found),
    whole,jobs(layout),query_index(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~Valid,~word,~found,whole),0n,plan(layout),readers(layout),Gemm.Both{{==},{==}},
    found_inside(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key,found(layout,key,valid)),{==})

# ---- Readback ----

# Located writes with a covered key read back that key's value.
def located_readback(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,
  ~word: @+layout: Layout -> @+key: Key -> U32,~wanted: @+layout: Layout -> @+key: Key -> E,~bound: @+layout: Layout -> Nat,
  ~below: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> {Nat.is_lt(U32.to_nat(word(layout,key)),bound(layout)) == True{} : Bool},
  ~same: @+layout: Layout -> @+key: Key -> @+other: Key -> @+valid: Valid(layout,key) -> @+other_valid: Valid(layout,other) ->
    @+equal: {word(layout,key) == word(layout,other) : U32} -> {wanted(layout,key) == wanted(layout,other) : E},
  +layout: Layout,+key: Key,+valid: Valid(layout,key),+depth: Nat,+tree: Tree.Tree<E>,+balanced: Shape.Balanced(E,depth,tree),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(bound(layout),U32.to_nat(Tree.capacity(E,tree))) == True{} : Bool},
  +writes: List<&2,Writes.Assignment<E>>,+holds: Every.Every(~E,~Layout,~Located(~E,~Layout,~Key,~Valid,~word,~wanted),layout,writes),
  contains: {WriteReadback.contains(~E,word(layout,key),writes) == True{} : Bool}) ->
  {Tree.read(E,Writes.pure(E,writes,tree),word(layout,key)) == wanted(layout,key) : E}:
  +output = {Output{layout,Tree.capacity(E,tree),room} : Bounded(~Layout,~bound)}
  +query = {Query{layout,key,valid} : Asked(~Layout,~Key,~Valid)}
  WriteReadback.covered(~E,writes,depth,tree,word(layout,key),wanted(layout,key),balanced,depth_ok,
    located_inside(~E,~Layout,~Key,~Valid,~word,~wanted,~bound,~below,output,word(layout,key),wanted(layout,key),Keyed{key,valid,{==},{==}}),
    Every.inside(~E,~Layout,~Located(~E,~Layout,~Key,~Valid,~word,~wanted),~Bounded(~Layout,~bound),~output_layout(~Layout,~bound),
      ~output_capacity(~Layout,~bound),~located_inside(~E,~Layout,~Key,~Valid,~word,~wanted,~bound,~below),output,writes,holds),
    Every.matching(~E,~Layout,~Located(~E,~Layout,~Key,~Valid,~word,~wanted),~Asked(~Layout,~Key,~Valid),
      ~query_layout(~Layout,~Key,~Valid),~query_word(~Layout,~Key,~Valid,~word),~query_wanted(~E,~Layout,~Key,~Valid,~wanted),
      ~located_matches(~E,~Layout,~Key,~Valid,~word,~wanted,~same),query,writes,holds),contains)

# A run from an output tree reads back the value of every valid key.
def readback(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,~writer: GemmLaws.Writer,
  ~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,~Layout: Data,~plan: @+layout: Layout -> Plan,~readers: @+layout: Layout -> Readers,
  ~jobs: @+layout: Layout -> Nat,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> GemmLaws.Element(writer),
  ~rows_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~columns_small: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} ->
    {Nat.is_le(job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index))),8n) == True{} : Bool},
  ~cell: @+layout: Layout -> @+index: Nat -> @+inside: {Nat.is_lt(index,jobs(layout)) == True{} : Bool} -> @+row: Nat -> @+column: Nat ->
    @+row_inside: {Nat.is_lt(row,job_rows(~Cursor,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,job_columns(~Cursor,writer,job_at(plan(layout),U32.from_nat(index)))) == True{} : Bool} ->
    Located(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,layout,cell_address(~Cursor,job_at(plan(layout),U32.from_nat(index)),row,column),
      cell_value(~Cursor,~advance,~order,~value,~close,writer,job_at(plan(layout),U32.from_nat(index)),readers(layout),row,column)),
  ~bound: @+layout: Layout -> Nat,
  ~below: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> {Nat.is_lt(U32.to_nat(word(layout,key)),bound(layout)) == True{} : Bool},
  ~same: @+layout: Layout -> @+key: Key -> @+other: Key -> @+valid: Valid(layout,key) -> @+other_valid: Valid(layout,other) ->
    @+equal: {word(layout,key) == word(layout,other) : U32} -> {wanted(layout,key) == wanted(layout,other) : GemmLaws.Element(writer)},
  ~found: @+layout: Layout -> @+key: Key -> @+valid: Valid(layout,key) -> Locate(~Plan,~Cursor,~job_at,~writer,~Layout,~plan,~jobs,~Key,~word,layout,key),
  +layout: Layout,+key: Key,+valid: Valid(layout,key),+depth: Nat,+tree: Tree.Tree<GemmLaws.Element(writer)>,+balanced: Shape.Balanced(GemmLaws.Element(writer),depth,tree),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(bound(layout),U32.to_nat(Tree.capacity(GemmLaws.Element(writer),tree))) == True{} : Bool}) ->
  {Tree.read(GemmLaws.Element(writer),Writes.pure(GemmLaws.Element(writer),layout_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,layout),
      tree),word(layout,key)) == wanted(layout,key) : GemmLaws.Element(writer)}:
  +writes = layout_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,layout)
  +holds = all_every(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~Valid,~word,~wanted,~rows_small,~columns_small,~cell,layout)
  located_readback(~GemmLaws.Element(writer),~Layout,~Key,~Valid,~word,~wanted,~bound,~below,~same,layout,key,valid,depth,tree,balanced,depth_ok,room,writes,holds,
    all_contains(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,~Layout,~plan,~readers,~jobs,~Key,~Valid,~word,~rows_small,~columns_small,~found,Query{layout,key,valid}))

# A write of a valid key is never at an address the caller keeps.
def located_apart(~E: Data,~Layout: Data,~Key: Data,~Valid: @+layout: Layout -> @+key: Key -> Data,~word: @+layout: Layout -> @+key: Key -> U32,
  ~wanted: @+layout: Layout -> @+key: Key -> E,~Kept: Data,~kept_layout: @+kept: Kept -> Layout,~kept_word: @+kept: Kept -> U32,
  ~apart: @+kept: Kept -> @+key: Key -> @+valid: Valid(kept_layout(kept),key) -> {U32.is_eq(word(kept_layout(kept),key),kept_word(kept)) == False{} : Bool},
  +kept: Kept,+address: U32,+stored: E,holds: Located(~E,~Layout,~Key,~Valid,~word,~wanted,kept_layout(kept),address,stored)) ->
  {U32.is_eq(address,kept_word(kept)) == False{} : Bool}:
  Keyed{+key,+valid,+address_exact,+value_exact} = holds
  %Equal.sym(U32,address,word(kept_layout(kept),key),address_exact) : {U32.is_eq(_,kept_word(kept)) == False{} : Bool}
  apart(kept,key,valid)

type CellReaders is Data:
  CellReaders{row_tiles: Tree.Tree<Tile.Tile8>,lane_tiles: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>,unused_bias: Tree.Tree<F32>}

def cell_readers(state: GemmLaws.CellOperands) -> CellReaders:
  GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,unused_bias,output}} = state
  CellReaders{rows,lanes,bias,unused_bias}

def cell_program(state: GemmLaws.CellOperands) -> Writes.Program<Tile.Tile8>:
  GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,unused_bias,output}} = state
  output

def cell_output(state: GemmLaws.CellOperands) -> Tree.Tree<Tile.Tile8>:
  Writes.evaluate(Tile.Tile8,cell_program(state))

def cell_operands(readers: CellReaders,output: Writes.Program<Tile.Tile8>) -> GemmLaws.CellOperands:
  CellReaders{rows,lanes,bias,unused_bias} = readers
  GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,unused_bias,output}}

def cell_job_writes(+job: Gemm.Job<Gemm.Direct()>,+readers: CellReaders) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  CellReaders{+rows,+lanes,+bias,unused_bias} = readers
  GemmLaws.cell_rows(bias,row,store,GemmLaws.reduced_accumulators(~Gemm.Direct(),GemmLaws.reduce(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},depth,rows,lanes,GemmLaws.Reduced{cursor,Gemm.zero()})))

def cell_step_writes(index: U32,matrix: Gemm.TileMatrix,readers: CellReaders) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  cell_job_writes(Gemm.tile_major(matrix,index),readers)

def cell_kept(index: U32,matrix: Gemm.TileMatrix,readers: CellReaders) -> CellReaders:
  readers

def cell_all_writes(count: Nat,+first: U32,+matrix: Gemm.TileMatrix,+readers: CellReaders) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Fold.all(~Tile.Tile8,~Gemm.TileMatrix,~CellReaders,~cell_step_writes,~cell_kept,count,first,matrix,readers)

def cell_block_appends(+job: Gemm.Job<Gemm.Direct()>,+readers: CellReaders,+output: Writes.Program<Tile.Tile8>) ->
  {GemmLaws.cell_block(job,cell_operands(readers,output)) == cell_operands(readers,Writes.Append{cell_job_writes(job,readers),output}) : GemmLaws.CellOperands}:
  Gemm.Job{row,depth,cursor,store} = job
  CellReaders{rows,lanes,bias,unused_bias} = readers
  {==}

def cell_readers_step(+index: U32,+matrix: Gemm.TileMatrix,+state: GemmLaws.CellOperands) ->
  {cell_readers(GemmLaws.cell_job(index,matrix,state)) == cell_kept(index,matrix,cell_readers(state)) : CellReaders}:
  GemmLaws.CellOperands{+bias,GemmLaws.Operands{+rows,+lanes,+unused_bias,+output}} = state
  +readers = {CellReaders{rows,lanes,bias,unused_bias} : CellReaders}
  %Equal.sym(GemmLaws.CellOperands,GemmLaws.cell_block(Gemm.tile_major(matrix,index),state),cell_operands(readers,Writes.Append{cell_step_writes(index,matrix,readers),output}),
    cell_block_appends(Gemm.tile_major(matrix,index),readers,output)) : {cell_readers(_) == readers : CellReaders}
  {==}

def cell_output_step(+index: U32,+matrix: Gemm.TileMatrix,+state: GemmLaws.CellOperands) ->
  {cell_output(GemmLaws.cell_job(index,matrix,state)) == Writes.pure(Tile.Tile8,cell_step_writes(index,matrix,cell_readers(state)),cell_output(state)) : Tree.Tree<Tile.Tile8>}:
  GemmLaws.CellOperands{+bias,GemmLaws.Operands{+rows,+lanes,+unused_bias,+output}} = state
  +readers = {CellReaders{rows,lanes,bias,unused_bias} : CellReaders}
  %Equal.sym(GemmLaws.CellOperands,GemmLaws.cell_block(Gemm.tile_major(matrix,index),state),cell_operands(readers,Writes.Append{cell_step_writes(index,matrix,readers),output}),
    cell_block_appends(Gemm.tile_major(matrix,index),readers,output)) :
    {cell_output(_) == Writes.pure(Tile.Tile8,cell_step_writes(index,matrix,readers),Writes.evaluate(Tile.Tile8,output)) : Tree.Tree<Tile.Tile8>}
  {==}

def cell_run_writes(+count: Nat,+first: U32,+matrix: Gemm.TileMatrix,+state: GemmLaws.CellOperands) ->
  {cell_output(Traversal.indexed_fold(~GemmLaws.CellOperands,~Gemm.TileMatrix,~GemmLaws.cell_job,count,first,matrix,state)) ==
    Writes.pure(Tile.Tile8,cell_all_writes(count,first,matrix,cell_readers(state)),cell_output(state)) : Tree.Tree<Tile.Tile8>}:
  Fold.output(~GemmLaws.CellOperands,~Gemm.TileMatrix,~Tile.Tile8,~CellReaders,~GemmLaws.cell_job,~cell_readers,~cell_output,~cell_step_writes,~cell_kept,
    ~cell_readers_step,~cell_output_step,count,first,matrix,state)

def cell_run_keeps(count: Nat,+first: U32,+matrix: Gemm.TileMatrix,+readers: CellReaders,+output: Writes.Program<Tile.Tile8>) ->
  {Traversal.indexed_fold(~GemmLaws.CellOperands,~Gemm.TileMatrix,~GemmLaws.cell_job,count,first,matrix,cell_operands(readers,output)) ==
    cell_operands(readers,cell_program(Traversal.indexed_fold(~GemmLaws.CellOperands,~Gemm.TileMatrix,~GemmLaws.cell_job,count,first,matrix,cell_operands(readers,output)))) : GemmLaws.CellOperands}:
  match count:
    case 0n:
      CellReaders{rows,lanes,bias,unused_bias} = readers
      {==}
    case 1n+ +rest:
      +next = {Writes.Append{cell_step_writes(first,matrix,readers),output} : Writes.Program<Tile.Tile8>}
      %Equal.sym(GemmLaws.CellOperands,GemmLaws.cell_block(Gemm.tile_major(matrix,first),cell_operands(readers,output)),cell_operands(readers,next),
        cell_block_appends(Gemm.tile_major(matrix,first),readers,output)) :
        {Traversal.indexed_fold(~GemmLaws.CellOperands,~Gemm.TileMatrix,~GemmLaws.cell_job,rest,(first + 1 : U32),matrix,_) ==
          cell_operands(readers,cell_program(Traversal.indexed_fold(~GemmLaws.CellOperands,~Gemm.TileMatrix,~GemmLaws.cell_job,rest,(first + 1 : U32),matrix,_))) : GemmLaws.CellOperands}
      cell_run_keeps(rest,(first + 1 : U32),matrix,readers,next)

# A cell epilogue is the scalar write model of its eight local bias values.
def cell_virtual_job(job: Gemm.Job<Gemm.Direct()>) -> Gemm.Job<Gemm.Direct()>:
  Gemm.Job{row,depth,cursor,store} = job
  Gemm.Job{0,depth,cursor,store}

def cell_virtual_readers(job: Gemm.Job<Gemm.Direct()>,readers: CellReaders) -> Readers:
  Gemm.Job{row,depth,cursor,store} = job
  CellReaders{rows,lanes,bias,unused_bias} = readers
  Readers{rows,lanes,GemmLaws.bias_tree(Tree.read(Tile.Tile8,bias,(row / 8 : U32)))}

def cell_job_as_virtual(+job: Gemm.Job<Gemm.Direct()>,+readers: CellReaders) ->
  {cell_job_writes(job,readers) == job_writes(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.biased_value,~Gemm.plus,
    cell_virtual_job(job),cell_virtual_readers(job,readers)) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  Gemm.Job{+row,+depth,+cursor,+store} = job
  Gemm.Store{base,row_stride,lane_stride,stored_rows,stored_lanes} = store
  CellReaders{+rows,+lanes,+bias,unused_bias} = readers
  GemmLaws.cell_rows_block(Tree.read(Tile.Tile8,bias,(row / 8 : U32)),store,
    GemmLaws.reduced_accumulators(~Gemm.Direct(),GemmLaws.reduce(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},depth,rows,lanes,GemmLaws.Reduced{cursor,Gemm.zero()})))

def cell_job_value(+job: Gemm.Job<Gemm.Direct()>,+readers: CellReaders,+row: Nat,+column: Nat) -> Tile.Tile8:
  cell_value(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.biased_value,~Gemm.plus,GemmLaws.Tiles{},cell_virtual_job(job),cell_virtual_readers(job,readers),row,column)

def cell_job_contains(+job: Gemm.Job<Gemm.Direct()>,+readers: CellReaders,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,job_rows(~Gemm.Direct(),job)) == True{} : Bool},+column_inside: {Nat.is_lt(column,1n) == True{} : Bool},
  +rows_small: {Nat.is_le(job_rows(~Gemm.Direct(),job),8n) == True{} : Bool}) ->
  {WriteReadback.contains(~Tile.Tile8,cell_address(~Gemm.Direct(),job,row,column),cell_job_writes(job,readers)) == True{} : Bool}:
  Gemm.Job{first,depth,cursor,store} = job
  Gemm.Store{base,row_stride,lane_stride,rows,lanes} = store
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile8>>,cell_job_writes(job,readers),job_writes(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.biased_value,~Gemm.plus,cell_virtual_job(job),cell_virtual_readers(job,readers)),cell_job_as_virtual(job,readers)) :
    {WriteReadback.contains(~Tile.Tile8,cell_address(~Gemm.Direct(),job,row,column),_) == True{} : Bool}
  job_contains(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.biased_value,~Gemm.plus,cell_virtual_job(job),cell_virtual_readers(job,readers),row,column,row_inside,column_inside,rows_small,{==})
