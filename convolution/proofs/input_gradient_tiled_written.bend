# Every write of a leaf stores the specified values at its own output tile.
# A write is checked by itself: its tile point in the leaf region, the address
# of that point, and the specified value in each lane that holds a channel.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/kernel_gemm_jobs.bend as GemmJobs
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/view_writes.bend as ViewWrites
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_store_geometry.bend as Channels
import ../../lib/proofs/input_gradient_tiled_store_columns.bend as Columns
import ../../lib/proofs/input_gradient_tiled_store_addresses.bend as Addresses
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ./input_gradient_tiled_dot.bend as Dot
import ./input_gradient_tiled_products.bend as Products
import ./input_gradient_order.bend as Order
import ../input_gradient_specification.bend as Strided
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_tiled_source_witness.bend as Witnesses
import ./input_gradient_cell_sources.bend as Sources
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
import ../../lib/proofs/write_program_fold.bend as Fold

# The channel of lane l of a tile point.
def lane_channel(+query: Chunks.Query,+point: Output.Point<query>,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(Output.group(query,point),8n),lane)

# Lane l of a written tile holds the problem's value at its channel, when the
# lane holds a channel.
def LaneValue(+query: Chunks.Query,+problem: Strided.Problem,+point: Output.Point<query>,+value: Tile.Tile8,+lane: Nat) -> Data:
  Boolean.When(Nat.is_lt(lane_channel(query,point,lane),Domain.channels(Chunks.geometry(query))),
    {TileLaws.select_lane(lane,value) == Strided.value(problem,lane_channel(query,point,lane),Output.row(query,point),Output.column(query,point)) : F32})

type Witness<-query: Chunks.Query,-problem: Strided.Problem,-address: U32,-value: Tile.Tile8> is Data:
  Witness{point: Output.Point<query>,
    at: {Output.native_address(query,point) == address : U32},
    lane0: LaneValue(query,problem,point,value,0n),lane1: LaneValue(query,problem,point,value,1n),
    lane2: LaneValue(query,problem,point,value,2n),lane3: LaneValue(query,problem,point,value,3n),
    lane4: LaneValue(query,problem,point,value,4n),lane5: LaneValue(query,problem,point,value,5n),
    lane6: LaneValue(query,problem,point,value,6n),lane7: LaneValue(query,problem,point,value,7n)}
def witness_point(+query: Chunks.Query,+problem: Strided.Problem,+address: U32,+value: Tile.Tile8,witness: Witness<query,problem,address,value>) -> Output.Point<query>:
  Witness{point,at,l0,l1,l2,l3,l4,l5,l6,l7} = witness
  point
# What a witness states of one lane.
def witness_lane(+query: Chunks.Query,+problem: Strided.Problem,+address: U32,+value: Tile.Tile8,+witness: Witness<query,problem,address,value>,
  +lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> LaneValue(query,problem,witness_point(query,problem,address,value,witness),value,lane):
  Witness{+point,+at,+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} = witness
  match lane:
    case 0n: l0
    case 1n+ +a:
      match a:
        case 0n: l1
        case 1n+ +b:
          match b:
            case 0n: l2
            case 1n+ +c:
              match c:
                case 0n: l3
                case 1n+ +d:
                  match d:
                    case 0n: l4
                    case 1n+ +e:
                      match e:
                        case 0n: l5
                        case 1n+ +f:
                          match f:
                            case 0n: l6
                            case 1n+ +g:
                              match g:
                                case 0n: l7
                                case 1n+ +rest: Empty.absurd(LaneValue(query,problem,point,value,8n+rest),NatOrder.below_zero(rest,small))

# What every write of a leaf satisfies: its tile point lies in the output
# region and its lanes hold the problem's values there.
type Target is Data:
  Target{region: Chunks.Query,problem: Strided.Problem}
def target_region(target: Target) -> Chunks.Query:
  Target{region,problem} = target
  region
def target_problem(target: Target) -> Strided.Problem:
  Target{region,problem} = target
  problem
def Written(+target: Target,address: U32,value: Tile.Tile8) -> Data:
  Witness<target_region(target),target_problem(target),address,value>

# The channel of lane l of a job is its unit's first channel plus l.
def channel_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+lane: Nat) ->
  {Nat.add(U32.to_nat(Products.channel_base(seed)),lane) == Nat.add(Channels.first(Channels.channel_job(seed,valid)),lane) : Nat}:
  Equal.cong(Nat,Nat,first => Nat.add(first,lane),U32.to_nat(Products.channel_base(seed)),Channels.first(Channels.channel_job(seed,valid)),
    Channels.first_word_exact(Channels.channel_job(seed,valid)))

# Facts about a chunk query that every job of the chunk needs.
def ChunkInside(+query: Chunks.Query) -> Data:
  Partition.Inside<GeometryChunks.query(Chunks.grid(query),Chunks.row(query),Chunks.group(query))>
def Contained(+query: Chunks.Query) -> Data:
  {Nat.is_le(Nat.add(U32.to_nat(Chunks.first(query)),U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query)))),Domain.height(Chunks.geometry(query))) == True{} : Bool}
def Fits(+query: Chunks.Query) -> Data:
  {Kernel.fits(Chunks.geometry(query),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(query)))) == True{} : Bool}

# The band of gradient rows a query's leaf reads, and the gradient it reads:
# the source rows that hold the band, of an image of the batch, for one phase
# and for every phase of the geometry.
def band(+query: Chunks.Query,+contained: Contained(query)) -> Bands.Query:
  +grid = Chunks.grid(query)
  Bands.caller_query(GeometryChunks.grid_call(grid),GeometryChunks.grid_envelope(grid),
    Chunks.first(query),GeometryChunks.grid_count(grid),contained)
def SelectionSource(+selection: Chunks.Selection,+contained: Contained(Chunks.query(selection)),+image: U32,+gradient: Buffer.Model<Tile.Tile8>) -> Data:
  +phase = Jobs.context(Chunks.phase(selection))
  +query = Chunks.query(selection)
  Gemm.Both<Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,Packing.cells_base(phase,image),Packing.output_groups(Chunks.geometry(query)),Fill.outputs(Chunks.filling(query))>,
    Products.ImageInside(query,image)>
def ChunkSource(+query: Chunks.Query,+contained: Contained(query),+image: U32,+gradient: Buffer.Model<Tile.Tile8>) -> Data:
  +geometry = Chunks.geometry(query)
  Gemm.Both<Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,Packing.cells_image_base(geometry,image),Packing.output_groups(geometry),Fill.outputs(Chunks.filling(query))>,
    Products.ImageInside(query,image)>
def selection_source(+selection: Chunks.Selection,+contained: Contained(Chunks.query(selection)),+image: U32,+gradient: Buffer.Model<Tile.Tile8>,
  +source: ChunkSource(Chunks.query(selection),contained,image,gradient)) -> SelectionSource(selection,contained,image,gradient):
  +query = Chunks.query(selection)
  +geometry = Chunks.geometry(query)
  +y = U32.from_nat(Chunks.y(selection))
  +x = U32.from_nat(Chunks.x(selection))
  +phase = Kernel.phase(geometry,y,x)
  %Equal.sym(U32,Packing.cells_base(phase,image),Packing.cells_image_base(geometry,image),Packing.cells_base_matches(geometry,y,x,image)) :
    Gemm.Both<Witnesses.Witness<band(query,contained),Chunks.source(query),gradient,_,Packing.output_groups(geometry),Fill.outputs(Chunks.filling(query))>,Products.ImageInside(query,image)>
  source

# A selected phase of a chunk and what all its jobs share: the chunk and
# phase facts, the packed weight and table bases, what the cells read and
# the source rows.
type Phase is Data:
  Phase{selection: Chunks.Selection,
    chunk_inside: ChunkInside(Chunks.query(selection)),contained: Contained(Chunks.query(selection)),fits: Fits(Chunks.query(selection)),
    enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Chunks.phase(selection))),0) == False{} : Bool},
    bases: Runtime.Bases<selection>,
    image: U32,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,
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
def gradient(phase: Phase) -> Buffer.Model<Tile.Tile8>:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  gradient
def weights(phase: Phase) -> Buffer.Model<Tile.Tile8>:
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  weights
def source(phase: Phase) -> SelectionSource(selection(phase),contained(phase),image(phase),gradient(phase)):
  Phase{selection,chunk_inside,contained,fits,enabled,bases,image,gradient,weights,source} = phase
  source

def phase_query(+phase: Phase) -> Chunks.Query:
  Chunks.query(selection(phase))
def phase_problem(+phase: Phase) -> Strided.Problem:
  Sources.problem(Jobs.geometry(Chunks.phase(selection(phase))),Sources.models_reader(Jobs.geometry(Chunks.phase(selection(phase))),gradient(phase),weights(phase)),U32.to_nat(image(phase)))
def phase_target(+phase: Phase) -> Target:
  Target{Output.region(phase_query(phase)),phase_problem(phase)}
def phase_work(+phase: Phase) -> Execution.Work:
  +query = phase_query(phase)
  Execution.work(Jobs.context(Chunks.phase(selection(phase))),Chunks.chunk(query),Chunks.weight_base(query))
def phase_readers(+phase: Phase) -> GemmJobs.Readers:
  +query = phase_query(phase)
  LeafWrites.chunk_readers(Chunks.shape(query),Chunks.source(query),Packed.weight_tree(Weights.full_state(Chunks.phase(selection(phase)),weights(phase))))
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

# A job of a phase.
type Job is Data:
  Job{phase: Phase,ordinal: Nat,inside: {Nat.is_lt(ordinal,job_count(phase)) == True{} : Bool}}
def job_phase(job: Job) -> Phase:
  Job{phase,ordinal,inside} = job
  phase
def ordinal(job: Job) -> Nat:
  Job{phase,ordinal,inside} = job
  ordinal
def ordinal_inside(job: Job) -> {Nat.is_lt(ordinal(job),job_count(job_phase(job))) == True{} : Bool}:
  Job{phase,ordinal,inside} = job
  inside

def seed(+job: Job) -> Chunks.Reduction:
  job_seed(job_phase(job),ordinal(job))
def valid(+job: Job) -> Chunks.Valid<seed(job)>:
  job_valid(job_phase(job),ordinal(job),ordinal_inside(job))
def gemm_job(+job: Job) -> Gemm.Job<Kernel.Cursor()>:
  Runtime.job(seed(job))
def job_readers(+job: Job) -> GemmJobs.Readers:
  phase_readers(job_phase(job))
def job_target(+job: Job) -> Target:
  phase_target(job_phase(job))
def channel_job(+job: Job) -> Channels.Channels:
  Channels.channel_job(seed(job),valid(job))
def column_job(+job: Job) -> Columns.Columns:
  Columns.column_job(seed(job),valid(job))

# A job stores min(8, columns - 8 g) rows, one position each, with the lanes
# of its unit's channels.
def row_counted(+job: Job,+row: Nat,+inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),gemm_job(job))) == True{} : Bool}) ->
  {Nat.is_lt(row,U32.to_nat(Columns.count(column_job(job)))) == True{} : Bool}:
  %Columns.rows_matches(seed(job),valid(job)) : {Nat.is_lt(row,U32.to_nat(_)) == True{} : Bool}
  inside
def rows_small(+job: Job) -> {Nat.is_le(GemmJobs.job_rows(~Kernel.Cursor(),gemm_job(job)),8n) == True{} : Bool}:
  Columns.rows_small(seed(job),valid(job))
def columns_small(+job: Job) -> {Nat.is_le(GemmJobs.job_columns(~Kernel.Cursor(),GemmLaws.Tiles{},gemm_job(job)),8n) == True{} : Bool}:
  {==}

# Lane l of the tile of row r holds the specified value at its channel and
# the row's position, when the lane holds a channel.
def lane_written(present: Bool,+job: Job,+row: Nat,+row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),gemm_job(job))) == True{} : Bool},
  +lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +same: {Nat.is_lt(Nat.add(Nat.mul(Chunks.unit(seed(job)),8n),lane),Domain.channels(Chunks.geometry(Chunks.query(Chunks.selection(seed(job)))))) == present : Bool}) ->
  Boolean.When(present,{TileLaws.select_lane(lane,GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},gemm_job(job),job_readers(job),row,0n)) ==
    Strided.value(phase_problem(job_phase(job)),Nat.add(Nat.mul(Chunks.unit(seed(job)),8n),lane),Jobs.physical_row(Chunks.work(Chunks.selection(seed(job))),Chunks.line(seed(job))),
      Columns.physical_column(seed(job),row)) : F32}):
  match present:
    case False{}: Unit{}
    case True{}:
      +shared = job_phase(job)
      +seed = seed(job)
      +valid = valid(job)
      +gradient = gradient(shared)
      +context = Jobs.context(Chunks.phase(Chunks.selection(seed)))
      +query = Chunks.query(Chunks.selection(seed))
      +channels = channel_job(job)
      +columns = column_job(job)
      +row_small = ViewWrites.clipped_small(row,8,(Columns.count_word(columns) - Columns.first_word(columns) : U32),row_counted(job,row,row_inside))
      +channel_inside = {
        %Equal.sym(Nat,U32.to_nat(Products.channel_base(seed)),Channels.first(channels),Channels.first_word_exact(channels)) :
          {Nat.is_lt(Nat.add(_,lane),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool}
        same : {Nat.is_lt(Nat.add(U32.to_nat(Products.channel_base(seed)),lane),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool}}
      +point = {Dot.Point{seed,valid,phase_bases(shared),image(shared),gradient,weights(shared),lane,row,small,row_small,
        Columns.packed_column_inside(seed,valid,row,row_inside),channel_inside,
        Witnesses.range(Chunks.leaf_band(seed,valid),Chunks.source(query),gradient,Packing.cells_base(context,image(shared)),Packing.output_groups(Chunks.geometry(query)),
          Fill.outputs(Chunks.filling(query)),
          Shape.first(Witnesses.Witness<Chunks.leaf_band(seed,valid),Chunks.source(query),gradient,Packing.cells_base(context,image(shared)),Packing.output_groups(Chunks.geometry(query)),Fill.outputs(Chunks.filling(query))>,
            Products.ImageInside(query,image(shared)),source(shared)))} : Dot.Point}
      +flat = Products.flat(lane,row,seed,image(shared),gradient,weights(shared))
      +problem = Order.flat_problem(flat)
      Equal.trans(F32,TileLaws.select_lane(lane,GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},Runtime.job(seed),job_readers(job),row,0n)),
        Strided.value(problem,Order.flat_channel(flat),Order.flat_y(flat),Order.flat_x(flat)),
        Strided.value(problem,Nat.add(Nat.mul(Chunks.unit(seed),8n),lane),Order.flat_y(flat),Order.flat_x(flat)),
        Dot.cell_value(Dot.Whole{point,source(shared)},Buffer.tree(~F32,Execution.bias())),
        Equal.cong(Nat,F32,channel => Strided.value(problem,channel,Order.flat_y(flat),Order.flat_x(flat)),Order.flat_channel(flat),Nat.add(Nat.mul(Chunks.unit(seed),8n),lane),
          channel_matches(seed,valid,lane)))

# The tile of a stored row of a job: its point, its address and its lanes.
def job_cell(+job: Job,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),gemm_job(job))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,GemmJobs.job_columns(~Kernel.Cursor(),GemmLaws.Tiles{},gemm_job(job))) == True{} : Bool}) ->
  Written(job_target(job),GemmJobs.cell_address(~Kernel.Cursor(),gemm_job(job),row,column),
    GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},gemm_job(job),job_readers(job),row,column)):
  +seed = seed(job)
  +valid = valid(job)
  +query = Chunks.query(Chunks.selection(seed))
  +located = Output.job_point(seed,valid,row,row_inside)
  +channels = Domain.channels(Chunks.geometry(query))
  +first = Nat.mul(Chunks.unit(seed),8n)
  %Equal.sym(Nat,column,0n,GemmJobs.only_cell(column,column_inside)) :
    Written(job_target(job),GemmJobs.cell_address(~Kernel.Cursor(),gemm_job(job),row,_),
      GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},gemm_job(job),job_readers(job),row,_))
  Witness{Output.region_point(query,located),
    Equal.sym(U32,GemmJobs.cell_address(~Kernel.Cursor(),Runtime.job(seed),row,0n),Output.native_address(query,located),Addresses.actual_address(row,seed,valid,row_inside)),
    lane_written(Nat.is_lt(Nat.add(first,0n),channels),job,row,row_inside,0n,{==},{==}),lane_written(Nat.is_lt(Nat.add(first,1n),channels),job,row,row_inside,1n,{==},{==}),
    lane_written(Nat.is_lt(Nat.add(first,2n),channels),job,row,row_inside,2n,{==},{==}),lane_written(Nat.is_lt(Nat.add(first,3n),channels),job,row,row_inside,3n,{==},{==}),
    lane_written(Nat.is_lt(Nat.add(first,4n),channels),job,row,row_inside,4n,{==},{==}),lane_written(Nat.is_lt(Nat.add(first,5n),channels),job,row,row_inside,5n,{==},{==}),
    lane_written(Nat.is_lt(Nat.add(first,6n),channels),job,row,row_inside,6n,{==},{==}),lane_written(Nat.is_lt(Nat.add(first,7n),channels),job,row,row_inside,7n,{==},{==})}

# Every write of a job is a stored row's tile.
def job_written(+job: Job) ->
  Every.Every(~Tile.Tile8,~Target,~Written,job_target(job),
    GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,gemm_job(job),job_readers(job))):
  GemmJobs.job_every(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,~Job,~Target,~job_target,~Written,~gemm_job,~job_readers,
    ~rows_small,~columns_small,~job_cell,job)

# What the fold over the jobs of a phase keeps: the jobs left fit in the
# phase, and the plan and the readers are the phase's.
def JobsLeft(+phase: Phase,rest: Nat,position: Nat,plan: Kernel.Plan,readers: GemmJobs.Readers) -> Data:
  Gemm.Both<{Nat.is_le(Nat.add(position,rest),job_count(phase)) == True{} : Bool},
    Gemm.Both<{plan == Execution.work_plan(phase_work(phase)) : Kernel.Plan},{readers == phase_readers(phase) : GemmJobs.Readers}>>
def job_step(+phase: Phase,+rest: Nat,+position: Nat,+plan: Kernel.Plan,+readers: GemmJobs.Readers,left: JobsLeft(phase,1n+rest,position,plan,readers)) ->
  Every.Every(~Tile.Tile8,~Target,~Written,phase_target(phase),
    GemmJobs.step_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,U32.from_nat(position),plan,readers)):
  Gemm.Both{room,same} = left
  Gemm.Both{same_plan,same_readers} = same
  +inside = NatOrder.room_index(position,rest,job_count(phase),room)
  +work_plan = Execution.work_plan(phase_work(phase))
  %Equal.sym(Kernel.Plan,plan,work_plan,same_plan) :
    Every.Every(~Tile.Tile8,~Target,~Written,phase_target(phase),
      GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,Kernel.job(_,U32.from_nat(position)),readers))
  %Equal.sym(GemmJobs.Readers,readers,phase_readers(phase),same_readers) :
    Every.Every(~Tile.Tile8,~Target,~Written,phase_target(phase),
      GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,Kernel.job(work_plan,U32.from_nat(position)),_))
  %Jobs.encoded_coordinates_exact(Chunks.work(selection(phase)),position,inside) :
    Every.Every(~Tile.Tile8,~Target,~Written,phase_target(phase),
      GemmJobs.job_writes(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,Kernel.job(work_plan,_),phase_readers(phase)))
  job_written(Job{phase,position,inside})
def jobs_left_kept(+phase: Phase,+rest: Nat,+position: Nat,+plan: Kernel.Plan,+readers: GemmJobs.Readers,left: JobsLeft(phase,1n+rest,position,plan,readers)) ->
  JobsLeft(phase,rest,1n+position,plan,GemmJobs.kept(~Kernel.Plan,U32.from_nat(position),plan,readers)):
  Gemm.Both{room,same} = left
  Gemm.Both{NatOrder.advance_room(position,rest,job_count(phase),room),same}

# The native job count of a phase is its number of jobs.
def jobs_fit(+phase: Phase) -> {Nat.is_le(Execution.work_count(phase_work(phase)),job_count(phase)) == True{} : Bool}:
  +selected = selection(phase)
  +query = phase_query(phase)
  +total = job_count(phase)
  %Equal.sym(Nat,Execution.work_count(phase_work(phase)),total,
    Jobs.native_count_exact(Chunks.work(selected),Jobs.active_phase(Chunks.phase(selected),enabled(phase)),Chunks.weight_base(query))) :
    {Nat.is_le(_,total) == True{} : Bool}
  NatOrder.reflexive(total)

# Every write of the jobs of a phase stores the specified value at its own
# output point.
def phase_written(+phase: Phase) ->
  Every.Every(~Tile.Tile8,~Target,~Written,phase_target(phase),
    Execution.jobs_writes(Execution.work_count(phase_work(phase)),Execution.work_plan(phase_work(phase)),phase_readers(phase))):
  +count = Execution.work_count(phase_work(phase))
  +room = jobs_fit(phase)
  Fold.every(~Tile.Tile8,~Kernel.Plan,~GemmJobs.Readers,~GemmJobs.step_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus),
    ~GemmJobs.kept(~Kernel.Plan),~Phase,~Target,~phase_target,~Written,~JobsLeft,~job_step,~jobs_left_kept,
    phase,count,0n,Execution.work_plan(phase_work(phase)),phase_readers(phase),Gemm.Both{room,Gemm.Both{{==},{==}}})
