# Every write of a leaf stores the specified value at its own output point.
# A write is checked by itself: its point in the leaf region, the address of
# that point, and the specified value there.
import Base
import ../../lib/kernel_strided_store.bend as Stores
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_reduction_model.bend as Accumulators
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/kernel_strided_store_writes.bend as StoreModel
import ../../lib/proofs/view_writes.bend as ViewWrites
import ../../lib/tensor_view.bend as Views
import ../../lib/proofs/tensor_copy_values.bend as Copy
import ../../lib/proofs/input_gradient_tiled_reduction.bend as Reduction
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_job_store.bend as Store
import ../../lib/proofs/input_gradient_tiled_jobs.bend as NativeJobs
import ../../lib/proofs/input_gradient_tiled_store_geometry.bend as Channels
import ../../lib/proofs/input_gradient_tiled_store_columns.bend as Columns
import ../../lib/proofs/input_gradient_tiled_store_addresses.bend as Addresses
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Patches
import ./input_gradient_tiled_dot.bend as Dot
import ./input_gradient_tiled_products.bend as Products
import ./input_gradient_order.bend as Order
import ../input_gradient_specification.bend as Strided
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_tiled_source_witness.bend as Witnesses
import ./input_gradient_sources.bend as Sources
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/input_gradient_tiled_chunks.bend as Partition
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_band_extents.bend as Bands
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_reduction_indices.bend as Indices
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/row_major_coordinates.bend as Coordinates
import ../../lib/proofs/nat_order.bend as NatOrder
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program_fold.bend as Fold

type Witness<-query: Chunks.Query,-problem: Strided.Problem,-address: U32,-value: F32> is Data:
  Witness{point: Output.Point<query>,
    at: {Output.native_address(query,point) == address : U32},
    correct: {value == Strided.value(problem,Output.channel(query,point),Output.row(query,point),Output.column(query,point)) : F32}}

# What every write of a leaf satisfies: its point lies in the output region
# and its value is the problem's value there.
type Target is Data:
  Target{region: Chunks.Query,problem: Strided.Problem}
def target_region(target: Target) -> Chunks.Query:
  Target{region,problem} = target
  region
def target_problem(target: Target) -> Strided.Problem:
  Target{region,problem} = target
  problem
def Written(+target: Target,address: U32,value: F32) -> Data:
  Witness<target_region(target),target_problem(target),address,value>

# The store of a point's job and the cell of the point in it.
def store(+upper: Bool,+point: Dot.Point<upper>) -> Stores.Store:
  NativeJobs.job_store(Runtime.job_index(Dot.seed(upper,point)),Runtime.plan(Dot.seed(upper,point)))
def cell(~upper: Bool,+point: Dot.Point<upper>) -> StoreModel.BlockQuery:
  Store.query(~upper,Dot.row(upper,point),Dot.lane(upper,point),Dot.seed(upper,point),Dot.weights(upper,point))

# The packed channel of a cell is its channel in the output.
def channel_matches(+upper: Bool,+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row: Accumulator.Row,+half: {Store.upper(seed) == upper : Bool}) ->
  {Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(upper,row)) == Nat.add(Channels.first(Channels.channel_job(seed,valid)),Store.row_number(row)) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(upper,row)),Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(Store.upper(seed),row)),
    Nat.add(Channels.first(Channels.channel_job(seed,valid)),Store.row_number(row)),
    Equal.cong(Bool,Nat,selected => Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(selected,row)),upper,Store.upper(seed),Equal.sym(Bool,Store.upper(seed),upper,half)),
    Channels.channel_key(Channels.channel_job(seed,valid),row))

# A stored cell writes the ordered accumulator of its point plus +0.0.
def cell_written(~upper: Bool,+frame: Dot.Whole<upper>,
  +row_inside: {Nat.is_lt(Store.row_number(Dot.row(upper,Dot.whole_point(upper,frame))),U32.to_nat(StoreModel.store_rows(store(upper,Dot.whole_point(upper,frame))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(Dot.lane(upper,Dot.whole_point(upper,frame)),U32.to_nat(StoreModel.store_count(store(upper,Dot.whole_point(upper,frame))))) == True{} : Bool},
  +half: {Store.upper(Dot.seed(upper,Dot.whole_point(upper,frame))) == upper : Bool}) ->
  Witness<Output.region(Chunks.query(Chunks.selection(Dot.seed(upper,Dot.whole_point(upper,frame))))),Order.flat_problem(Dot.flat(upper,Dot.whole_point(upper,frame))),
    StoreModel.block_address(cell(~upper,Dot.whole_point(upper,frame))),StoreModel.block_value(cell(~upper,Dot.whole_point(upper,frame)))>:
  Dot.Whole{+point,source} = frame
  Dot.Point{+seed,+valid,bases,+image,+gradient,+weights,+row,+lane,lane_small,column_inside,channel_inside,range} = point
  +query = Chunks.query(Chunks.selection(seed))
  +located = Output.job_point(seed,valid,row,lane,row_inside,lane_inside)
  +flat = Products.flat(upper,row,lane,seed,image,gradient,weights)
  +problem = Order.flat_problem(flat)
  +accumulated = (Accumulators.accumulator_at(Accumulators.RowLane{row,TileLaws.lane_of(lane)},Reduction.accumulators(Store.final(~upper,seed,weights))) + 0.0 : F32)
  +specified = Strided.value(problem,Order.flat_channel(flat),Order.flat_y(flat),Order.flat_x(flat))
  Witness{Output.region_point(query,located),
    Equal.sym(U32,StoreModel.block_address(cell(~upper,point)),Output.native_address(query,located),Addresses.actual_address(row,lane,seed,valid,row_inside,lane_inside)),
    Equal.trans(F32,StoreModel.block_value(cell(~upper,point)),accumulated,Strided.value(problem,Output.channel(query,located),Output.row(query,located),Output.column(query,located)),
      Store.value_matches(~upper,row,lane,seed,weights),
      Equal.trans(F32,accumulated,specified,Strided.value(problem,Output.channel(query,located),Output.row(query,located),Output.column(query,located)),
        Dot.whole_mac_value(~upper,frame),
        Equal.cong(Nat,F32,channel => Strided.value(problem,channel,Order.flat_y(flat),Order.flat_x(flat)),Order.flat_channel(flat),Output.channel(query,located),
          channel_matches(upper,seed,valid,row,half))))}

# Facts about a chunk query that every job of the chunk needs.
def ChunkInside(+query: Chunks.Query) -> Data:
  Partition.Inside<GeometryChunks.query(Chunks.grid(query),Chunks.row(query),Chunks.group(query))>
def Contained(+query: Chunks.Query) -> Data:
  {Nat.is_le(Nat.add(U32.to_nat(Chunks.first(query)),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query)))),Domain.height(Chunks.geometry(query))) == True{} : Bool}
def Fits(+query: Chunks.Query) -> Data:
  {Kernel.fits(Chunks.geometry(query),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(query)))) == True{} : Bool}

# The band of gradient rows a query's leaf reads, and the source rows that
# hold it: for one phase, and for every phase of the geometry.
def band(+query: Chunks.Query,+contained: Contained(query)) -> Bands.Query:
  +grid = Chunks.grid(query)
  Bands.caller_query(GeometryChunks.grid_call(grid),GeometryChunks.grid_envelope(grid),
    Chunks.first(query),GeometryChunks.grid_count(grid),contained)
def SelectionSource(+selection: Chunks.Selection,+contained: Contained(Chunks.query(selection)),+image: U32,+gradient: Buffer.Model<F32>) -> Data:
  +phase = Jobs.context(Chunks.phase(selection))
  +query = Chunks.query(selection)
  Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,Packing.input_base(phase,image),Packing.gradient_step(phase),Fill.outputs(Chunks.filling(query))>
def ChunkSource(+query: Chunks.Query,+contained: Contained(query),+image: U32,+gradient: Buffer.Model<F32>) -> Data:
  +geometry = Chunks.geometry(query)
  Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,Packing.image_base(geometry,image),Packing.plane_size(geometry),Fill.outputs(Chunks.filling(query))>
def selection_source(+selection: Chunks.Selection,+contained: Contained(Chunks.query(selection)),+image: U32,+gradient: Buffer.Model<F32>,
  +source: ChunkSource(Chunks.query(selection),contained,image,gradient)) -> SelectionSource(selection,contained,image,gradient):
  +query = Chunks.query(selection)
  +geometry = Chunks.geometry(query)
  +y = U32.from_nat(Chunks.y(selection))
  +x = U32.from_nat(Chunks.x(selection))
  +phase = Kernel.phase(geometry,y,x)
  %Equal.sym(U32,Packing.input_base(phase,image),Packing.image_base(geometry,image),Packing.input_base_matches(geometry,y,x,image)) :
    Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,_,Packing.gradient_step(phase),Fill.outputs(Chunks.filling(query))>
  %Equal.sym(U32,Packing.gradient_step(phase),Packing.plane_size(geometry),Packing.gradient_step_matches(geometry,y,x)) :
    Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,Packing.image_base(geometry,image),_,Fill.outputs(Chunks.filling(query))>
  source

# A selected phase of a chunk and what all its jobs share: the chunk and
# phase facts, the packed weight and table bases, what the cells read and
# the source rows.
type Phase is Data:
  Phase{selection: Chunks.Selection,
    chunk_inside: ChunkInside(Chunks.query(selection)),contained: Contained(Chunks.query(selection)),fits: Fits(Chunks.query(selection)),
    enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Chunks.phase(selection))),0) == False{} : Bool},
    bases: Runtime.Bases<selection>,
    image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,
    source: SelectionSource(selection,contained,image,gradient)}
def selection(phase: Phase) -> Chunks.Selection:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  selection
def chunk_inside(phase: Phase) -> ChunkInside(Chunks.query(selection(phase))):
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  chunk_inside
def contained(phase: Phase) -> Contained(Chunks.query(selection(phase))):
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  contained
def fits(phase: Phase) -> Fits(Chunks.query(selection(phase))):
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  fits
def enabled(phase: Phase) -> {U32.is_eq(Kernel.phase_inner(Jobs.context(Chunks.phase(selection(phase)))),0) == False{} : Bool}:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  enabled
def phase_bases(phase: Phase) -> Runtime.Bases<selection(phase)>:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  bases
def image(phase: Phase) -> U32:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  image
def gradient(phase: Phase) -> Buffer.Model<F32>:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  gradient
def weights(phase: Phase) -> Buffer.Model<F32>:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  weights
def source(phase: Phase) -> SelectionSource(selection(phase),contained(phase),image(phase),gradient(phase)):
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  source

def phase_query(+phase: Phase) -> Chunks.Query:
  Chunks.query(selection(phase))
def phase_problem(+phase: Phase) -> Strided.Problem:
  Sources.problem(Jobs.geometry(Chunks.phase(selection(phase))),Sources.models_reader(gradient(phase),weights(phase)),U32.to_nat(image(phase)))
def phase_target(+phase: Phase) -> Target:
  Target{Output.region(phase_query(phase)),phase_problem(phase)}
def phase_work(+phase: Phase) -> Execution.Work:
  +query = phase_query(phase)
  Execution.work(Jobs.context(Chunks.phase(selection(phase))),Chunks.chunk(query),Chunks.weight_base(query),Chunks.offset_base(query))
def phase_readers(+phase: Phase) -> NativeJobs.Readers:
  +packed = Weights.full_state(Chunks.phase(selection(phase)),weights(phase))
  +query = phase_query(phase)
  NativeJobs.Readers{Fill.tree(Execution.fill_chunk(Chunks.shape(query),Chunks.source(query))),Packed.weight_tree(packed),Packed.offset_tree(packed)}
def job_count(+phase: Phase) -> Nat:
  Jobs.natural_count(Chunks.work(selection(phase)))

# Job ordinal of a phase: its line, column group and unit, and the first
# product of the phase.
def job_seed(+phase: Phase,+ordinal: Nat) -> Chunks.Reduction:
  +selected = selection(phase)
  +job = Jobs.job_coordinates(Chunks.work(selected),ordinal)
  +tap = Indices.values(Chunks.phase(selected),0n)
  Chunks.Reduction{selected,Coordinates.channel(job),Coordinates.row(job),Coordinates.column(job),Coordinates.channel(tap),Coordinates.row(tap),Coordinates.column(tap)}
def job_valid(+phase: Phase,+ordinal: Nat,+inside: {Nat.is_lt(ordinal,job_count(phase)) == True{} : Bool}) -> Chunks.Valid<job_seed(phase,ordinal)>:
  +selected = selection(phase)
  +phase_selected = Chunks.phase(selected)
  +outputs = Domain.outputs(Jobs.geometry(phase_selected))
  +rows = Indices.rows(phase_selected)
  +columns = Indices.columns(phase_selected)
  +first = Indices.values(phase_selected,0n)
  +taps = Indices.bounded(phase_selected,enabled(phase),0n,PhaseBounds.phase_inner_positive(Jobs.context(phase_selected),enabled(phase)))
  Chunks.Valid{chunk_inside(phase),contained(phase),fits(phase),enabled(phase),Jobs.job_coordinates_inside(Chunks.work(selected),ordinal,inside),
    Coordinates.channel_below(outputs,rows,columns,first,taps),Coordinates.row_below(outputs,rows,columns,first,taps),Coordinates.column_below(outputs,rows,columns,first,taps)}

# A job of a phase, and the half of the packed weights its rows take.
type Job<-upper: Bool> is Data:
  Job{phase: Phase,ordinal: Nat,inside: {Nat.is_lt(ordinal,job_count(phase)) == True{} : Bool},half: {Store.upper(job_seed(phase,ordinal)) == upper : Bool}}
def job_phase(+upper: Bool,job: Job<upper>) -> Phase:
  Job{phase,ordinal,inside,half} = job
  phase
def ordinal(+upper: Bool,job: Job<upper>) -> Nat:
  Job{phase,ordinal,inside,half} = job
  ordinal
def ordinal_inside(+upper: Bool,job: Job<upper>) -> {Nat.is_lt(ordinal(upper,job),job_count(job_phase(upper,job))) == True{} : Bool}:
  Job{phase,ordinal,inside,half} = job
  inside
def half(+upper: Bool,job: Job<upper>) -> {Store.upper(job_seed(job_phase(upper,job),ordinal(upper,job))) == upper : Bool}:
  Job{phase,ordinal,inside,half} = job
  half

def seed(+upper: Bool,+job: Job<upper>) -> Chunks.Reduction:
  job_seed(job_phase(upper,job),ordinal(upper,job))
def valid(+upper: Bool,+job: Job<upper>) -> Chunks.Valid<seed(upper,job)>:
  job_valid(job_phase(upper,job),ordinal(upper,job),ordinal_inside(upper,job))
def bases(+upper: Bool,+job: Job<upper>) -> Runtime.Bases<Chunks.selection(seed(upper,job))>:
  phase_bases(job_phase(upper,job))
def job_store(+upper: Bool,+job: Job<upper>) -> Stores.Store:
  NativeJobs.job_store(Runtime.job_index(seed(upper,job)),Runtime.plan(seed(upper,job)))
def job_block(~upper: Bool,+job: Job<upper>) -> StoreModel.Block:
  Store.block(~upper,seed(upper,job),weights(job_phase(upper,job)))

# The cell of a job at one stored row and lane.
def frame(~upper: Bool,+job: Job<upper>,+row: Accumulator.Row,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +row_inside: {Nat.is_lt(Store.row_number(row),U32.to_nat(StoreModel.store_rows(job_store(upper,job)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,U32.to_nat(StoreModel.store_count(job_store(upper,job)))) == True{} : Bool}) -> Dot.Whole<upper>:
  +shared = job_phase(upper,job)
  +seed = seed(upper,job)
  +valid = valid(upper,job)
  +gradient = gradient(shared)
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed)))
  +query = Chunks.query(Chunks.selection(seed))
  Dot.Whole{Dot.Point{seed,valid,bases(upper,job),image(shared),gradient,weights(shared),row,lane,lane_small,
      Columns.packed_column_inside(seed,valid,lane,lane_inside),
      %half(upper,job) : {Nat.is_lt(Nat.add(U32.to_nat((U32.from_nat(Nat.div(Chunks.unit(seed),2n)) * 8 : U32)),Patches.patch_key(_,row)),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool}
        Channels.packed_channel_inside(seed,valid,row,row_inside),
      Witnesses.range(Chunks.leaf_band(seed,valid),Chunks.source(query),gradient,Packing.input_base(phase,image(shared)),Packing.gradient_step(phase),Fill.outputs(Chunks.filling(query)),source(shared))},
    source(shared)}

# What a job's store writes: its target, its view and the cells of the view.
def job_target(~upper: Bool,+job: Job<upper>) -> Target:
  phase_target(job_phase(upper,job))
def job_view(~upper: Bool,+job: Job<upper>) -> Views.View:
  StoreModel.store_view(StoreModel.store(job_block(~upper,job)))
def job_rows(~upper: Bool,+job: Job<upper>) -> Nat:
  StoreModel.stored_rows(StoreModel.store(job_block(~upper,job)))
def job_lanes(~upper: Bool,+job: Job<upper>) -> Nat:
  StoreModel.stored_lanes(StoreModel.store(job_block(~upper,job)))

# A stored cell of a job writes its point's value.
def job_cell(~upper: Bool,+job: Job<upper>,+row: Nat,+lane: Nat,
  +row_inside: {Nat.is_lt(row,StoreModel.stored_rows(StoreModel.store(job_block(~upper,job)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,StoreModel.stored_lanes(StoreModel.store(job_block(~upper,job)))) == True{} : Bool}) ->
  Written(phase_target(job_phase(upper,job)),
    Copy.address(StoreModel.store_view(StoreModel.store(job_block(~upper,job))),Nat.add(Nat.mul(row,StoreModel.stored_lanes(StoreModel.store(job_block(~upper,job)))),lane)),
    StoreModel.tile_value(job_block(~upper,job),Nat.add(Nat.mul(row,StoreModel.stored_lanes(StoreModel.store(job_block(~upper,job)))),lane))):
  +block = job_block(~upper,job)
  +store = job_store(upper,job)
  +position = Nat.add(Nat.mul(row,StoreModel.stored_lanes(store)),lane)
  +target = phase_target(job_phase(upper,job))
  +key = Accumulator.row_of(row)
  +named = Store.number_matches(row,ViewWrites.clipped_small(row,4,StoreModel.store_rows(store),row_inside))
  +lane_small = ViewWrites.clipped_small(lane,8,StoreModel.store_count(store),lane_inside)
  +lane_stored = ViewWrites.clipped_below(lane,8,StoreModel.store_count(store),lane_inside)
  +row_stored = {
    %Equal.sym(Nat,Store.row_number(key),row,named) : {Nat.is_lt(_,U32.to_nat(StoreModel.store_rows(store))) == True{} : Bool}
    ViewWrites.clipped_below(row,4,StoreModel.store_rows(store),row_inside) : {Nat.is_lt(Store.row_number(key),U32.to_nat(StoreModel.store_rows(store))) == True{} : Bool}}
  %StoreModel.cell_view_address(block,row,lane,row_inside,lane_inside) : Written(target,_,StoreModel.tile_value(block,position))
  %StoreModel.cell_view_value(block,row,lane,lane_inside) : Written(target,StoreModel.row_address(StoreModel.row_context(block,row),lane),_)
  %named : Written(target,StoreModel.row_address(StoreModel.row_context(block,_),lane),StoreModel.row_value(StoreModel.row_context(block,_),lane))
  cell_written(~upper,frame(~upper,job,key,lane,lane_small,row_stored,lane_stored),row_stored,lane_stored,half(upper,job))

# Every write of a job is a stored cell's.
def job_written(~upper: Bool,+job: Job<upper>) ->
  Every.Every(~F32,~Target,~Written,phase_target(job_phase(upper,job)),
    NativeJobs.job_writes(Runtime.job_index(seed(upper,job)),Runtime.plan(seed(upper,job)),Runtime.readers(seed(upper,job),weights(job_phase(upper,job))))):
  +shared = job_phase(upper,job)
  +seed = seed(upper,job)
  +index = Runtime.job_index(seed)
  +plan = Runtime.plan(seed)
  %Equal.sym(Reduction.Loop,NativeJobs.final_loop(index,plan,Runtime.readers(seed,weights(shared))),Store.final(~upper,seed,weights(shared)),
    Store.final_loop_matches(~upper,seed,valid(upper,job),weights(shared),half(upper,job))) :
    Every.Every(~F32,~Target,~Written,phase_target(job_phase(upper,job)),StoreModel.block_writes(StoreModel.Block{NativeJobs.job_store(index,plan),Reduction.accumulators(_)}))
  +block = job_block(~upper,job)
  +store = StoreModel.store(block)
  %Equal.sym(List<&2,Writes.Assignment<F32>>,StoreModel.block_writes(block),
    ViewWrites.writes(~F32,~StoreModel.Block,~StoreModel.tile_value,Nat.mul(StoreModel.stored_rows(store),StoreModel.stored_lanes(store)),StoreModel.store_view(store),block),
    StoreModel.view_writes(block)) :
    Every.Every(~F32,~Target,~Written,phase_target(shared),_)
  ViewWrites.every_plane(~F32,~StoreModel.Block,~StoreModel.tile_value,~Job<upper>,~Target,~job_target(~upper),~Written,~job_view(~upper),~job_block(~upper),
    ~job_rows(~upper),~job_lanes(~upper),~job_cell(~upper),job)

# The job of a phase at an ordinal, taken by the half its unit selects.
def selected_job(+phase: Phase,+ordinal: Nat,+inside: {Nat.is_lt(ordinal,job_count(phase)) == True{} : Bool},
  selected: Bool,+half: {Store.upper(job_seed(phase,ordinal)) == selected : Bool}) ->
  Every.Every(~F32,~Target,~Written,phase_target(phase),
    NativeJobs.job_writes(Runtime.job_index(job_seed(phase,ordinal)),Runtime.plan(job_seed(phase,ordinal)),Runtime.readers(job_seed(phase,ordinal),weights(phase)))):
  match selected:
    case True{}: job_written(~True{},Job{phase,ordinal,inside,half})
    case False{}: job_written(~False{},Job{phase,ordinal,inside,half})

# What the fold over the jobs of a phase keeps: the jobs left fit in the
# phase, and the plan and the readers are the phase's.
def JobsLeft(+phase: Phase,rest: Nat,position: Nat,plan: Kernel.Plan,readers: NativeJobs.Readers) -> Data:
  Shape.Both<{Nat.is_le(Nat.add(position,rest),job_count(phase)) == True{} : Bool},
    Shape.Both<{plan == Execution.work_plan(phase_work(phase)) : Kernel.Plan},{readers == phase_readers(phase) : NativeJobs.Readers}>>
def job_step(+phase: Phase,+rest: Nat,+position: Nat,+plan: Kernel.Plan,+readers: NativeJobs.Readers,left: JobsLeft(phase,1n+rest,position,plan,readers)) ->
  Every.Every(~F32,~Target,~Written,phase_target(phase),NativeJobs.job_writes(U32.from_nat(position),plan,readers)):
  Shape.Both{room,same} = left
  Shape.Both{same_plan,same_readers} = same
  +inside = NatOrder.room_index(position,rest,job_count(phase),room)
  +work_plan = Execution.work_plan(phase_work(phase))
  %Equal.sym(Kernel.Plan,plan,work_plan,same_plan) :
    Every.Every(~F32,~Target,~Written,phase_target(phase),NativeJobs.job_writes(U32.from_nat(position),_,readers))
  %Equal.sym(NativeJobs.Readers,readers,phase_readers(phase),same_readers) :
    Every.Every(~F32,~Target,~Written,phase_target(phase),NativeJobs.job_writes(U32.from_nat(position),work_plan,_))
  %Jobs.encoded_coordinates_exact(Chunks.work(selection(phase)),position,inside) :
    Every.Every(~F32,~Target,~Written,phase_target(phase),NativeJobs.job_writes(_,work_plan,phase_readers(phase)))
  selected_job(phase,position,inside,Store.upper(job_seed(phase,position)),{==})
def jobs_left_kept(+phase: Phase,+rest: Nat,+position: Nat,+plan: Kernel.Plan,+readers: NativeJobs.Readers,left: JobsLeft(phase,1n+rest,position,plan,readers)) ->
  JobsLeft(phase,rest,1n+position,plan,NativeJobs.readers_kept(U32.from_nat(position),plan,readers)):
  Shape.Both{room,same} = left
  Shape.Both{NatOrder.advance_room(position,rest,job_count(phase),room),same}

# The native job count of a phase is its number of jobs.
def jobs_fit(+phase: Phase) -> {Nat.is_le(Execution.work_count(phase_work(phase)),job_count(phase)) == True{} : Bool}:
  +selected = selection(phase)
  +query = phase_query(phase)
  +total = job_count(phase)
  %Equal.sym(Nat,Execution.work_count(phase_work(phase)),total,
    Jobs.native_count_exact(Chunks.work(selected),Jobs.active_phase(Chunks.phase(selected),enabled(phase)),Chunks.weight_base(query),Chunks.offset_base(query))) :
    {Nat.is_le(_,total) == True{} : Bool}
  NatOrder.reflexive(total)

# Every write of the jobs of a phase stores the specified value at its own
# output point.
def phase_written(+phase: Phase) ->
  Every.Every(~F32,~Target,~Written,phase_target(phase),
    NativeJobs.jobs_writes(Execution.work_count(phase_work(phase)),0,Execution.work_plan(phase_work(phase)),phase_readers(phase))):
  +count = Execution.work_count(phase_work(phase))
  +room = jobs_fit(phase)
  Fold.every(~F32,~Kernel.Plan,~NativeJobs.Readers,~NativeJobs.job_writes,~NativeJobs.readers_kept,~Phase,~Target,~phase_target,~Written,~JobsLeft,~job_step,~jobs_left_kept,
    phase,count,0n,Execution.work_plan(phase_work(phase)),phase_readers(phase),Shape.Both{room,Shape.Both{{==},{==}}})
