# The leaf writes the requested point of an enabled phase: the point's chunk
# among the leaf's chunks, its phase among the chunk's phases, its job among
# the phase's jobs and its row in the job.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ../kernel_gemm.bend as Gemm
import ./write_program_readback.bend as Readback
import ./write_program_fold.bend as Fold
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_gemm_refinement.bend as GemmLaws
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_producer.bend as Producer
import ./input_gradient_tiled_producer_cells.bend as Cells
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_chunks.bend as Partition
import ./input_gradient_tiled_output_layout.bend as Output
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_tiled_store_addresses.bend as Addresses
import ./input_gradient_tiled_store_columns.bend as Columns
import ./input_gradient_tiled_phase_index.bend as Index
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_leaf_writes.bend as LeafWrites
import ./input_gradient_tiled_sweep_selected.bend as Selected

# The tile of the requested point in the output region of its leaf, and its
# address.
def requested(+problem: Producer.Problem) -> Output.Point<Output.region(Producer.query(problem))>:
  +point = Producer.point(problem)
  +query = Producer.query(problem)
  +geometry = Chunks.geometry(query)
  +batch = GeometryChunks.grid_batch(Chunks.grid(query))
  +inside = Producer.point_inside(problem)
  Output.Point{Nat.div(Domain.channel(point),8n),Domain.row(point),Domain.column(point),
    Partition.chunks_inside(Domain.channel(point),Domain.channels(geometry),8n,{==},Domain.channel_inside(geometry,batch,point,inside)),
    Producer.row_lower(problem),Producer.row_upper(problem),Domain.column_inside(geometry,batch,point,inside)}
def address(+problem: Producer.Problem) -> U32:
  Output.native_address(Output.region(Producer.query(problem)),requested(problem))

# A requested point whose phase is enabled, in a leaf that fits its machine.
type Active is Data:
  Active{problem: Producer.Problem,
    enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
    fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}}
def problem(active: Active) -> Producer.Problem:
  Active{problem,enabled,fits} = active
  problem
def active_address(+active: Active) -> U32:
  address(problem(active))

# The requested point's job: its producer with the packing base the phase
# fold supplies.
type Job is Data:
  Job{active: Active,weight_base: U32}
def job_active(job: Job) -> Active:
  Job{active,weight_base} = job
  active
def weight_base(job: Job) -> U32:
  Job{active,weight_base} = job
  weight_base
def job_address(+job: Job) -> U32:
  active_address(job_active(job))
def seed(+job: Job) -> Chunks.Reduction:
  Producer.reduction(Producer.with_base(problem(job_active(job)),weight_base(job)))
def job_work(+job: Job) -> Execution.Work:
  +selected = Chunks.selection(seed(job))
  +query = Chunks.query(selected)
  Execution.work(Jobs.context(Chunks.phase(selected)),Chunks.chunk(query),Chunks.weight_base(query))
def job_target(+job: Job) -> Nat:
  +seed = seed(job)
  Jobs.job_position(Chunks.work(Chunks.selection(seed)),Chunks.line(seed),Chunks.local_group(seed),Chunks.unit(seed))

# The requested point's job writes its address: the row of the point's
# position holds it.
def job_covered(+job: Job,+readers: GemmJobs.Readers) ->
  {Readback.contains(~Tile.Tile8,job_address(job),GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,Runtime.job(seed(job)),readers)) == True{} : Bool}:
  Job{+active,+wb} = job
  Active{+problem,+enabled,+fits} = active
  Producer.Problem{+query,+point,+inside,+lower,+upper,+contained} = problem
  Chunks.Query{+grid,+first,+target,+source,+row,+group,+wb0} = query
  GeometryChunks.Grid{+call,+e,+count,+range,+rp,+rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+padding,+gh,+gw} = g
  Domain.Point{+image,+channel,+point_row,+point_column} = point
  Domain.Inside{+ii,+ci,+yi,+xi} = inside
  +changed = Producer.with_base(problem,wb)
  +seed = Producer.reduction(changed)
  +valid = Producer.valid(changed,enabled,fits)
  +located = Runtime.job(seed)
  +row = Producer.group_row(changed)
  +row_inside = Cells.row_inside(changed,enabled,fits)
  +selected = Chunks.query(Chunks.selection(seed))
  +cell = Equal.trans(U32,GemmJobs.cell_address(~Kernel.Cursor(),located,row,0n),
    Output.native_address(selected,Cells.output_point(changed,enabled,fits)),Output.native_address(selected,Cells.requested(changed)),
    Addresses.actual_address(row,seed,valid,row_inside),
    Cells.address_exact(changed,enabled,fits))
  %cell : {Readback.contains(~Tile.Tile8,_,GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,located,readers)) == True{} : Bool}
  GemmJobs.job_contains(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,located,readers,row,0n,row_inside,{==},
    Columns.rows_small(seed,valid),{==})

# The requested point's job is among the jobs of its phase.
def job_inside(+job: Job) -> {Nat.is_lt(job_target(job),Execution.work_count(job_work(job))) == True{} : Bool}:
  Job{+active,+wb} = job
  Active{+problem,+enabled,fits} = active
  Producer.Problem{+query,+point,+inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,source,row,group,wb0} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,padding,gh,gw} = g
  Domain.Point{image,channel,point_row,point_column} = point
  Domain.Inside{ii,ci,yi,xi} = inside
  +changed = Producer.with_base(problem,wb)
  +work = Producer.work(changed)
  %Equal.sym(Nat,Execution.work_count(job_work(job)),Jobs.natural_count(work),
    Jobs.native_count_exact(work,Jobs.active_phase(Producer.phase(changed),enabled),wb)) :
    {Nat.is_lt(job_target(job),_) == True{} : Bool}
  Jobs.job_position_inside(work,Producer.line(changed),Producer.group(changed),Producer.unit(changed),Producer.job_inside(changed,enabled))

# The jobs of the requested point's phase write its address: the fold keeps
# the phase's plan.
def JobPlan(+job: Job,rest: Nat,position: Nat,plan: Kernel.Plan,readers: GemmJobs.Readers) -> Data:
  {plan == Runtime.plan(seed(job)) : Kernel.Plan}
def plan_kept(+job: Job,+rest: Nat,+position: Nat,+plan: Kernel.Plan,+readers: GemmJobs.Readers,kept: JobPlan(job,1n+rest,position,plan,readers)) ->
  JobPlan(job,rest,1n+position,plan,GemmJobs.kept(~Kernel.Plan,U32.from_nat(position),plan,readers)):
  kept
def job_found(+job: Job,+rest: Nat,+plan: Kernel.Plan,+readers: GemmJobs.Readers,same: JobPlan(job,1n+rest,job_target(job),plan,readers)) ->
  {Readback.contains(~Tile.Tile8,job_address(job),
    GemmJobs.step_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,U32.from_nat(job_target(job)),plan,readers)) == True{} : Bool}:
  %Equal.sym(Kernel.Plan,plan,Runtime.plan(seed(job)),same) :
    {Readback.contains(~Tile.Tile8,job_address(job),
      GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,Kernel.job(_,U32.from_nat(job_target(job))),readers)) == True{} : Bool}
  job_covered(job,readers)
def jobs_covered(+job: Job,+readers: GemmJobs.Readers) ->
  {Readback.contains(~Tile.Tile8,job_address(job),Execution.jobs_writes(Execution.work_count(job_work(job)),Runtime.plan(seed(job)),readers)) == True{} : Bool}:
  Fold.contains_at(~Tile.Tile8,~Kernel.Plan,~GemmJobs.Readers,~GemmJobs.step_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus),
    ~GemmJobs.kept(~Kernel.Plan),~Job,~job_address,~job_target,~JobPlan,~plan_kept,~job_found,
    job,Execution.work_count(job_work(job)),job_target(job),0n,Runtime.plan(seed(job)),readers,{==},job_inside(job),{==})

# The requested point's chunk, and its phase among the chunk's phases.
def producer_chunk(+active: Active) -> Kernel.Chunk:
  +problem = problem(active)
  +query = Producer.query(problem)
  GeometryChunks.selected_chunk(Chunks.grid(query),Chunks.first(query),Chunks.target(query),Producer.relative_row(problem),Producer.column_group(problem))
def phase_target(+active: Active) -> Nat:
  Index.ordinal(Producer.phase(problem(active)))
def phase_inside(+active: Active) -> {Nat.is_lt(phase_target(active),Kernel.window_jobs(LeafWrites.chunk_geometry(producer_chunk(active)))) == True{} : Bool}:
  Active{+problem,enabled,fits} = active
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,source,row,group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Index.inside(Producer.phase(problem))

# The requested point's phase writes its address, whatever base it starts
# from.
def bases_covered(+active: Active,+readers: GemmJobs.Readers,+wb: U32) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phase_writes(U32.from_nat(phase_target(active)),producer_chunk(active),LeafWrites.Bases{readers,wb})) == True{} : Bool}:
  Active{+problem,+enabled,+fits} = active
  Producer.Problem{+query,+point,+inside,+lower,+upper,+contained} = problem
  Chunks.Query{+grid,+first,+target,+source,+row,+group,+wb0} = query
  GeometryChunks.Grid{+call,+e,+count,+range,+rp,+rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+padding,+gh,+gw} = g
  Domain.Point{+image,+channel,+point_row,+point_column} = point
  Domain.Inside{+ii,+ci,+yi,+xi} = inside
  +phase = Producer.phase(problem)
  +context = Jobs.context(phase)
  +chunk = producer_chunk(active)
  %Equal.sym(Kernel.Phase,Kernel.phase_of(Jobs.geometry(phase),U32.from_nat(Index.ordinal(phase))),context,Index.phase_matches(phase)) :
    {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.selected_writes(U32.is_eq(Kernel.phase_inner(_),0),_,chunk,LeafWrites.Bases{readers,wb})) == True{} : Bool}
  %Equal.sym(Bool,U32.is_eq(Kernel.phase_inner(context),0),False{},enabled) :
    {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.selected_writes(_,context,chunk,LeafWrites.Bases{readers,wb})) == True{} : Bool}
  jobs_covered(Job{active,wb},readers)
def phase_covered(+active: Active,+bases: LeafWrites.Bases) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phase_writes(U32.from_nat(phase_target(active)),producer_chunk(active),bases)) == True{} : Bool}:
  LeafWrites.Bases{+readers,+wb} = bases
  bases_covered(active,readers,wb)

# The phases of the requested point's chunk write its address: the fold
# keeps the chunk.
def ChunkFixed(+active: Active,rest: Nat,position: Nat,chunk: Kernel.Chunk,bases: LeafWrites.Bases) -> Data:
  {chunk == producer_chunk(active) : Kernel.Chunk}
def chunk_kept(+active: Active,+rest: Nat,+position: Nat,+chunk: Kernel.Chunk,+bases: LeafWrites.Bases,kept: ChunkFixed(active,1n+rest,position,chunk,bases)) ->
  ChunkFixed(active,rest,1n+position,chunk,LeafWrites.phase_bases(U32.from_nat(position),chunk,bases)):
  kept
def phase_found(+active: Active,+rest: Nat,+chunk: Kernel.Chunk,+bases: LeafWrites.Bases,same: ChunkFixed(active,1n+rest,phase_target(active),chunk,bases)) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phase_writes(U32.from_nat(phase_target(active)),chunk,bases)) == True{} : Bool}:
  %Equal.sym(Kernel.Chunk,chunk,producer_chunk(active),same) :
    {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phase_writes(U32.from_nat(phase_target(active)),_,bases)) == True{} : Bool}
  phase_covered(active,bases)
def phases_covered(+active: Active,+bases: LeafWrites.Bases) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phases_writes(Kernel.window_jobs(LeafWrites.chunk_geometry(producer_chunk(active))),0,producer_chunk(active),bases)) == True{} : Bool}:
  +count = Kernel.window_jobs(LeafWrites.chunk_geometry(producer_chunk(active)))
  Fold.contains_at(~Tile.Tile8,~Kernel.Chunk,~LeafWrites.Bases,~LeafWrites.phase_writes,~LeafWrites.phase_bases,~Active,~active_address,~phase_target,~ChunkFixed,~chunk_kept,~phase_found,
    active,count,phase_target(active),0n,producer_chunk(active),bases,{==},phase_inside(active),{==})

# The requested point's chunk among the leaf's chunks.
def leaf_context(+active: Active) -> Kernel.LeafContext:
  Selected.native_context(Producer.query(problem(active)))
def chunk_target(+active: Active) -> Nat:
  +problem = problem(active)
  Partition.producing_chunk(GeometryChunks.query(Chunks.grid(Producer.query(problem)),Producer.relative_row(problem),Producer.column_group(problem)))
def chunk_inside(+active: Active) -> {Nat.is_lt(chunk_target(active),Execution.chunk_count(leaf_context(active))) == True{} : Bool}:
  Active{+problem,+enabled,fits} = active
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,source,row,group,wb} = query
  %Equal.sym(Nat,Execution.chunk_count(Selected.native_context(query)),Partition.chunk_count(GeometryChunks.query(grid,0n,0n)),Selected.count(query)) :
    {Nat.is_lt(chunk_target(active),_) == True{} : Bool}
  Partition.producer_inside(GeometryChunks.query(grid,Producer.relative_row(problem),Producer.column_group(problem)),Producer.chunk_inside(problem,enabled))

# The requested point's chunk writes its address, whatever inputs it reads.
def chunk_covered(+active: Active,+inputs: LeafWrites.Inputs) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.chunk_writes(U32.from_nat(chunk_target(active)),leaf_context(active),inputs)) == True{} : Bool}:
  LeafWrites.Inputs{+source,+weights} = inputs
  +problem = problem(active)
  +query = Producer.query(problem)
  +grid = Chunks.grid(query)
  +row = Producer.relative_row(problem)
  +group = Producer.column_group(problem)
  +shape = Execution.chunk_shape(GeometryChunks.native_producer(grid,row,group),leaf_context(active),source)
  +bases = {LeafWrites.Bases{LeafWrites.chunk_readers(shape,source,weights),0} : LeafWrites.Bases}
  %Equal.sym(Kernel.Chunk,Execution.shape_chunk(shape),producer_chunk(active),
    GeometryChunks.selected_chunk_matches(grid,Chunks.first(query),Chunks.target(query),row,group,source)) :
    {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.phases_writes(Kernel.window_jobs(LeafWrites.chunk_geometry(_)),0,_,bases)) == True{} : Bool}
  phases_covered(active,bases)

# The chunks of the leaf write the requested address: the fold keeps the
# leaf context.
def ContextFixed(+active: Active,rest: Nat,position: Nat,context: Kernel.LeafContext,inputs: LeafWrites.Inputs) -> Data:
  {context == leaf_context(active) : Kernel.LeafContext}
def context_kept(+active: Active,+rest: Nat,+position: Nat,+context: Kernel.LeafContext,+inputs: LeafWrites.Inputs,kept: ContextFixed(active,1n+rest,position,context,inputs)) ->
  ContextFixed(active,rest,1n+position,context,LeafWrites.inputs_kept(U32.from_nat(position),context,inputs)):
  kept
def chunk_found(+active: Active,+rest: Nat,+context: Kernel.LeafContext,+inputs: LeafWrites.Inputs,same: ContextFixed(active,1n+rest,chunk_target(active),context,inputs)) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.chunk_writes(U32.from_nat(chunk_target(active)),context,inputs)) == True{} : Bool}:
  %Equal.sym(Kernel.LeafContext,context,leaf_context(active),same) :
    {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.chunk_writes(U32.from_nat(chunk_target(active)),_,inputs)) == True{} : Bool}
  chunk_covered(active,inputs)

# The leaf writes the requested address of an enabled phase.
def leaf_covered(+active: Active,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>) ->
  {Readback.contains(~Tile.Tile8,active_address(active),LeafWrites.leaf_writes(leaf_context(active),source,weights)) == True{} : Bool}:
  +context = leaf_context(active)
  Fold.contains_at(~Tile.Tile8,~Kernel.LeafContext,~LeafWrites.Inputs,~LeafWrites.chunk_writes,~LeafWrites.inputs_kept,~Active,~active_address,~chunk_target,~ContextFixed,~context_kept,~chunk_found,
    active,Execution.chunk_count(context),chunk_target(active),0n,context,LeafWrites.Inputs{source,weights},{==},chunk_inside(active),{==})
