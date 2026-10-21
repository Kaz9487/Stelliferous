# One actual job writes its ordered accumulator through the shared strided store.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_strided_store.bend as Stores
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_loop.bend as Iteration
import ./kernel_accumulator.bend as Accumulator
import ./kernel_coordinate_refinement.bend as Coordinates
import ./kernel_gemm_readback.bend as Gemm
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_reduction_model.bend as Accumulators
import ./kernel_strided_store_writes.bend as Writes
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_tiled_reduction.bend as Reduction
import ./input_gradient_tiled_reduction_indices.bend as Indices
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./storage_buffer_model.bend as Buffer
import ./input_gradient_phase_domain.bend as Domain
import ./nat_order.bend as Order

def row_number(row: Accumulator.Row) -> Nat:
  Coordinates.row_number(row)
def row_matches(+row: Accumulator.Row) -> {Accumulator.row_of(row_number(row)) == row : Accumulator.Row}:
  match row:
    case Accumulator.R0{}: {==}
    case Accumulator.R1{}: {==}
    case Accumulator.R2{}: {==}
    case Accumulator.R3{}: {==}
def number_matches(+row: Nat,small: {Nat.is_lt(row,4n) == True{} : Bool}) -> {row_number(Accumulator.row_of(row)) == row : Nat}:
  match row:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n+ +rest: Empty.absurd({row_number(Accumulator.row_of(4n+rest)) == 4n+rest : Nat},Order.below_zero(rest,small))
def row_small(+row: Accumulator.Row) -> {Nat.is_lt(row_number(row),4n) == True{} : Bool}:
  match row:
    case Accumulator.R0{}: {==}
    case Accumulator.R1{}: {==}
    case Accumulator.R2{}: {==}
    case Accumulator.R3{}: {==}
def count_matches(+seed: Chunks.Reduction) -> {NativeJobs.job_count(Runtime.plan(seed)) == Indices.count(Chunks.phase(Chunks.selection(seed))) : Nat}:
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def execute_selected(+upper: Bool,+count: Nat,+state: Reduction.Loop) ->
  {NativeJobs.execute(upper,count,state) == Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count,state) : Reduction.Loop}:
  match upper:
    case False{}: {==}
    case True{}: {==}
def final(~upper: Bool,+seed: Chunks.Reduction,+weights: Buffer.Model<F32>) -> Reduction.Loop:
  Iteration.run(~Reduction.Loop,~Reduction.step(~upper),Indices.count(Chunks.phase(Chunks.selection(seed))),Runtime.initial(seed,weights))
def block(~upper: Bool,+seed: Chunks.Reduction,+weights: Buffer.Model<F32>) -> Writes.Block:
  Writes.Block{NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Reduction.accumulators(final(~upper,seed,weights))}
def query(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+weights: Buffer.Model<F32>) -> Writes.BlockQuery:
  Writes.BlockQuery{block(~upper,seed,weights),row_number(row),lane}
def block_value_matches(+store: Stores.Store,+values: Rows.Rows<Tile.Tile8>,+row: Accumulator.Row,+lane: Nat) ->
  {Writes.block_value(Writes.BlockQuery{Writes.Block{store,values},row_number(row),lane}) == (Accumulators.accumulator_at(Accumulators.RowLane{row,TileLaws.lane_of(lane)},values) + 0.0 : F32) : F32}:
  Stores.Store{target,step,plane,rows,count} = store
  %Equal.sym(Tile.Tile8,Accumulator.select_row(~Tile.Tile8,row_number(row),values),Accumulator.row(~Tile.Tile8,row,values),
    Equal.trans(Tile.Tile8,Accumulator.select_row(~Tile.Tile8,row_number(row),values),Accumulator.row(~Tile.Tile8,Accumulator.row_of(row_number(row)),values),Accumulator.row(~Tile.Tile8,row,values),Accumulator.selection_matches(~Tile.Tile8,row_number(row),values),Equal.cong(Accumulator.Row,Tile.Tile8,r => Accumulator.row(~Tile.Tile8,r,values),Accumulator.row_of(row_number(row)),row,row_matches(row)))) :
    {(TileLaws.select_lane(lane,_) + 0.0 : F32) == (Tile.element_at(TileLaws.lane_of(lane),Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32) : F32}
  Equal.cong(F32,F32,value => (value + 0.0 : F32),TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),Tile.element_at(TileLaws.lane_of(lane),Accumulator.row(~Tile.Tile8,row,values)),TileLaws.selected_lane(lane,Accumulator.row(~Tile.Tile8,row,values)))
def value_matches(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+seed: Chunks.Reduction,+weights: Buffer.Model<F32>) ->
  {Writes.block_value(query(~upper,row,lane,seed,weights)) == (Accumulators.accumulator_at(Accumulators.RowLane{row,TileLaws.lane_of(lane)},Reduction.accumulators(final(~upper,seed,weights))) + 0.0 : F32) : F32}:
  block_value_matches(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed)),Reduction.accumulators(final(~upper,seed,weights)),row,lane)

# The actual half selector is fixed by the four-channel job ordinal.
def upper(+seed: Chunks.Reduction) -> Bool:
  Nat.is_eq(Nat.mod(Chunks.unit(seed),2n),1n)
def upper_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> {NativeJobs.job_upper(Runtime.job_index(seed),Runtime.plan(seed)) == upper(seed) : Bool}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  +work = Chunks.work(selected)
  +active = Jobs.active_phase(Chunks.phase(selected),enabled)
  Gemm.parity_exact(Runtime.plan_unit(work,wb,ob,line,group,unit),unit,Jobs.unit_decode(work,active,line,group,unit,ji))

# A seed's job loop is the ordered reduction of its phase products.
def final_loop_matches(~selected: Bool,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+weights: Buffer.Model<F32>,+half: {upper(seed) == selected : Bool}) ->
  {NativeJobs.final_loop(Runtime.job_index(seed),Runtime.plan(seed),Runtime.readers(seed,weights)) == final(~selected,seed,weights) : Reduction.Loop}:
  +index = Runtime.job_index(seed)
  +plan = Runtime.plan(seed)
  +initial = Runtime.initial(seed,weights)
  +count = Indices.count(Chunks.phase(Chunks.selection(seed)))
  %Equal.sym(Bool,NativeJobs.job_upper(index,plan),selected,Equal.trans(Bool,NativeJobs.job_upper(index,plan),upper(seed),selected,upper_matches(seed,valid),half)) :
    {NativeJobs.execute(_,NativeJobs.job_count(plan),initial) == final(~selected,seed,weights) : Reduction.Loop}
  %Equal.sym(Nat,NativeJobs.job_count(plan),count,count_matches(seed)) :
    {NativeJobs.execute(selected,_,initial) == final(~selected,seed,weights) : Reduction.Loop}
  execute_selected(selected,count,initial)

# The rows and lanes a job stores.
def row_count(+seed: Chunks.Reduction) -> Nat:
  U32.to_nat(Writes.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))
def lane_count(+seed: Chunks.Reduction) -> Nat:
  U32.to_nat(Writes.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))
