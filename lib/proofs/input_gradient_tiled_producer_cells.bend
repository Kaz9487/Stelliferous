# The located job's clipped row and lane denote the requested input element.
import Base
import ./input_gradient_tiled_output_layout.bend as Output
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_tiled_job_reads.bend as Reads
import ./storage_shape.bend as Shape
import ./input_gradient_phase_point.bend as Point
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./input_gradient_tiled_value_reads.bend as Input
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_producer.bend as Producer
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_store.bend as Store
import ./input_gradient_tiled_store_geometry.bend as Channels
import ./input_gradient_tiled_store_columns.bend as Columns
import ./kernel_strided_store_writes.bend as StoreModel
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./kernel_accumulator.bend as Accumulator
import ./kernel_gemm_readback.bend as Gemm
import ./nat_order.bend as Order
import ./nat_extrema.bend as Extrema
import ./nat_division_order.bend as Division

def row(+problem: Producer.Problem) -> Accumulator.Row:
  Accumulator.row_of(Producer.row_in_unit(problem))
def row_exact(+problem: Producer.Problem) -> {Store.row_number(row(problem)) == Producer.row_in_unit(problem) : Nat}:
  Gemm.row_number_of(Producer.row_in_unit(problem),Producer.row_small(problem))
def channel_split(+problem: Producer.Problem) -> {Domain.channel(Producer.point(problem)) == Nat.add(Nat.mul(Producer.unit(problem),4n),Producer.row_in_unit(problem)) : Nat}:
  Division.quotient_exact(Domain.channel(Producer.point(problem)),4n,{==})
def row_inside(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Nat.is_lt(Store.row_number(row(problem)),Store.row_count(Producer.reduction(problem))) == True{} : Bool}:
  Producer.Problem{+query,+point,+inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.Inside{ii,+ci,yi,xi} = inside
  +seed = Producer.reduction(problem)
  +valid = Producer.valid(problem,enabled,fits)
  +job = Channels.channel_job(seed,valid)
  +number = Producer.row_in_unit(problem)
  +total = Domain.channels(g)
  +start = Nat.mul(Producer.unit(problem),4n)
  %Equal.sym(Nat,Store.row_number(row(problem)),number,row_exact(problem)) : {Nat.is_lt(_,Store.row_count(seed)) == True{} : Bool}
  %Equal.sym(U32,StoreModel.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))),Channels.count(job),Channels.rows_matches(seed,valid)) : {Nat.is_lt(number,U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Channels.count(job)),Nat.min(4n,Nat.sub(total,start)),Channels.count_exact(job)) : {Nat.is_lt(number,_) == True{} : Bool}
  Extrema.minimum_strict_lower(4n,Nat.sub(total,start),number,Producer.row_small(problem),
    %Equal.sym(Bool,Nat.is_lt(number,Nat.sub(total,start)),Nat.is_lt(Nat.add(start,number),total),Order.subtraction_lt(total,start,number)) : {_ == True{} : Bool}
    %channel_split(problem) : {Nat.is_lt(_,total) == True{} : Bool}
    ci)
def first_column(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Columns.first(Columns.column_job(Producer.reduction(problem),Producer.valid(problem,enabled,fits))) == Nat.mul(Producer.column_group(problem),8n) : Nat}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +first = U32.to_nat(Jobs.chunk_group(Jobs.chunk(Producer.work(problem))))
  Equal.cong(Nat,Nat,value => Nat.mul(value,8n),Nat.add(first,Producer.group(problem)),Producer.column_group(problem),Order.restore_shift(first,Producer.column_group(problem),Producer.group_lower(problem,enabled)))
def column_split(+problem: Producer.Problem) -> {Producer.column_position(problem) == Nat.add(Nat.mul(Producer.column_group(problem),8n),Producer.lane(problem)) : Nat}:
  Division.quotient_exact(Producer.column_position(problem),8n,{==})
def lane_inside(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Nat.is_lt(Producer.lane(problem),Store.lane_count(Producer.reduction(problem))) == True{} : Bool}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +seed = Producer.reduction(problem)
  +valid = Producer.valid(problem,enabled,fits)
  +job = Columns.column_job(seed,valid)
  +number = Producer.lane(problem)
  +total = U32.to_nat(Columns.count_word(job))
  +start = Nat.mul(Producer.column_group(problem),8n)
  %Equal.sym(U32,StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))),Columns.count(job),Columns.counts_matches(seed,valid)) : {Nat.is_lt(number,U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Columns.count(job)),Nat.min(8n,Nat.sub(total,Columns.first(job))),Columns.count_exact(job)) : {Nat.is_lt(number,_) == True{} : Bool}
  %Equal.sym(Nat,Columns.first(job),start,first_column(problem,enabled,fits)) : {Nat.is_lt(number,Nat.min(8n,Nat.sub(total,_))) == True{} : Bool}
  Extrema.minimum_strict_lower(8n,Nat.sub(total,start),number,Producer.lane_small(problem),
    %Equal.sym(Bool,Nat.is_lt(number,Nat.sub(total,start)),Nat.is_lt(Nat.add(start,number),total),Order.subtraction_lt(total,start,number)) : {_ == True{} : Bool}
    %column_split(problem) : {Nat.is_lt(_,total) == True{} : Bool}
    Producer.column_axis_inside(problem))
def native_row_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool}) ->
  {U32.to_nat(Jobs.row_native(Producer.work(problem),Producer.line(problem))) == Producer.row_position(problem) : Nat}:
  Producer.Problem{+q0,location,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,wb,ob} = q0
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +work = Producer.work(problem)
  +position = Producer.row_position(problem)
  +first = U32.to_nat(Jobs.start(work))
  +lower = Shape.first({Nat.is_le(first,position) == True{} : Bool},{Nat.is_lt(position,U32.to_nat(Jobs.end(work))) == True{} : Bool},Producer.row_indices(problem,enabled))
  Equal.trans(Nat,U32.to_nat(Jobs.row_native(work,Producer.line(problem))),Nat.add(first,Producer.line(problem)),position,Jobs.row_native_exact(work,Jobs.active_phase(Jobs.phase(work),enabled),Producer.line(problem),Producer.line_inside(problem,enabled)),Order.restore_shift(first,position,lower))
def physical_row_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool}) ->
  {Jobs.physical_row(Producer.work(problem),Producer.line(problem)) == Domain.row(Producer.point(problem)) : Nat}:
  Producer.Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,+s,p,gh,gw} = g
  +first = Point.first(Domain.row(point),s)
  +position = Jobs.row_native(Producer.work(problem),Producer.line(problem))
  +pure_first = Producer.phase_y(problem)
  Equal.trans(Nat,Jobs.physical_row(Producer.work(problem),Producer.line(problem)),Nat.add(pure_first,Nat.mul(Producer.row_position(problem),U32.to_nat(s))),Domain.row(point),
    Equal.trans(Nat,Jobs.physical_row(Producer.work(problem),Producer.line(problem)),Nat.add(pure_first,Nat.mul(U32.to_nat(position),U32.to_nat(s))),Nat.add(pure_first,Nat.mul(Producer.row_position(problem),U32.to_nat(s))),
      Equal.cong(U32,Nat,value => Nat.add(U32.to_nat(value),Nat.mul(U32.to_nat(position),U32.to_nat(s))),U32.from_nat(pure_first),first,Arithmetic.native_roundtrip(first)),
      Equal.cong(Nat,Nat,value => Nat.add(pure_first,Nat.mul(value,U32.to_nat(s))),U32.to_nat(position),Producer.row_position(problem),native_row_exact(problem,enabled))),Producer.row_pixel_from(problem,Producer.axes(problem)))
def input_column_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {U32.to_nat(Input.input_column(Producer.reduction(problem))) == Nat.mul(Producer.column_group(problem),8n) : Nat}:
  +seed = Producer.reduction(problem)
  +valid = Producer.valid(problem,enabled,fits)
  +job = Columns.column_job(seed,valid)
  Equal.trans(Nat,U32.to_nat(Input.input_column(seed)),U32.to_nat(Columns.first_word(job)),Nat.mul(Producer.column_group(problem),8n),Equal.cong(U32,Nat,U32.to_nat,Input.input_column(seed),Columns.first_word(job),Columns.input_column_matches(seed,valid)),
    Equal.trans(Nat,U32.to_nat(Columns.first_word(job)),Columns.first(job),Nat.mul(Producer.column_group(problem),8n),Columns.first_word_exact(job),first_column(problem,enabled,fits)))
def column_position_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {U32.to_nat((Input.input_column(Producer.reduction(problem)) + U32.from_nat(Producer.lane(problem)) : U32)) == Producer.column_position(problem) : Nat}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +seed = Producer.reduction(problem)
  +valid = Producer.valid(problem,enabled,fits)
  +input = Input.input_column(seed)
  +lane = Producer.lane(problem)
  +phase = Chunks.phase(Chunks.selection(seed))
  +first = U32.from_nat(Jobs.phase_x(phase))
  +position = Producer.column_position(problem)
  Equal.trans(Nat,U32.to_nat((input + U32.from_nat(lane) : U32)),Nat.add(U32.to_nat(input),lane),position,PhaseBounds.lane_position_exact(call,first,input,lane,Reads.phase_x_inside(phase),Columns.packed_column_inside(seed,valid,lane,lane_inside(problem,enabled,fits))),
    Equal.trans(Nat,Nat.add(U32.to_nat(input),lane),Nat.add(Nat.mul(Producer.column_group(problem),8n),lane),position,Equal.cong(Nat,Nat,value => Nat.add(value,lane),U32.to_nat(input),Nat.mul(Producer.column_group(problem),8n),input_column_exact(problem,enabled,fits)),Equal.sym(Nat,position,Nat.add(Nat.mul(Producer.column_group(problem),8n),lane),column_split(problem))))
def physical_column_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Columns.physical_column(Producer.reduction(problem),Producer.lane(problem)) == Domain.column(Producer.point(problem)) : Nat}:
  Producer.Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,+s,p,gh,gw} = g
  +seed = Producer.reduction(problem)
  +first = Point.first(Domain.column(point),s)
  +position = (Input.input_column(seed) + U32.from_nat(Producer.lane(problem)) : U32)
  +pure_first = Producer.phase_x(problem)
  Equal.trans(Nat,Columns.physical_column(seed,Producer.lane(problem)),Nat.add(pure_first,Nat.mul(Producer.column_position(problem),U32.to_nat(s))),Domain.column(point),
    Equal.trans(Nat,Columns.physical_column(seed,Producer.lane(problem)),Nat.add(pure_first,Nat.mul(U32.to_nat(position),U32.to_nat(s))),Nat.add(pure_first,Nat.mul(Producer.column_position(problem),U32.to_nat(s))),
      Equal.cong(U32,Nat,value => Nat.add(U32.to_nat(value),Nat.mul(U32.to_nat(position),U32.to_nat(s))),U32.from_nat(pure_first),first,Arithmetic.native_roundtrip(first)),
      Equal.cong(Nat,Nat,value => Nat.add(pure_first,Nat.mul(value,U32.to_nat(s))),U32.to_nat(position),Producer.column_position(problem),column_position_exact(problem,enabled,fits))),Producer.column_pixel_from(problem,Producer.axes(problem)))
def output_point(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) -> Output.Point<Chunks.query(Chunks.selection(Producer.reduction(problem)))>:
  Output.job_point(Producer.reduction(problem),Producer.valid(problem,enabled,fits),row(problem),Producer.lane(problem),row_inside(problem,enabled,fits),lane_inside(problem,enabled,fits))
def channel_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Output.channel(Chunks.query(Chunks.selection(Producer.reduction(problem))),output_point(problem,enabled,fits)) == Domain.channel(Producer.point(problem)) : Nat}:
  Producer.Problem{+query,+point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +unit = Producer.unit(problem)
  +number = Producer.row_in_unit(problem)
  Equal.trans(Nat,Nat.add(Nat.mul(unit,4n),Store.row_number(row(problem))),Nat.add(Nat.mul(unit,4n),number),Domain.channel(point),Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(unit,4n),value),Store.row_number(row(problem)),number,row_exact(problem)),Equal.sym(Nat,Domain.channel(point),Nat.add(Nat.mul(unit,4n),number),channel_split(problem)))
def requested(+problem: Producer.Problem) -> Output.Point<Chunks.query(Chunks.selection(Producer.reduction(problem)))>:
  Producer.Problem{+query,+point,+inside,+lower,+upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.Inside{ii,ci,yi,xi} = inside
  Output.Point{Domain.channel(point),Domain.row(point),Domain.column(point),ci,lower,upper,xi}
def row_matches(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Output.row(Chunks.query(Chunks.selection(Producer.reduction(problem))),output_point(problem,enabled,fits)) == Domain.row(Producer.point(problem)) : Nat}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  physical_row_exact(problem,enabled)
def column_matches(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Output.column(Chunks.query(Chunks.selection(Producer.reduction(problem))),output_point(problem,enabled,fits)) == Domain.column(Producer.point(problem)) : Nat}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  physical_column_exact(problem,enabled,fits)
def address_exact(+problem: Producer.Problem,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem))),0) == False{} : Bool},
  +fits: {Kernel.fits(Chunks.geometry(Producer.query(problem)),Machine.capacity(GeometryChunks.grid_limits(Chunks.grid(Producer.query(problem))))) == True{} : Bool}) ->
  {Output.native_address(Chunks.query(Chunks.selection(Producer.reduction(problem))),output_point(problem,enabled,fits)) == Output.native_address(Chunks.query(Chunks.selection(Producer.reduction(problem))),requested(problem)) : U32}:
  Producer.Problem{+query,point,inside,lower,upper,contained} = problem
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.Inside{ii,ci,yi,xi} = inside
  +region = Chunks.query(Chunks.selection(Producer.reduction(problem)))
  +actual = output_point(problem,enabled,fits)
  +wanted = requested(problem)
  Equal.cong(Nat,U32,U32.from_nat,Output.address(region,actual),Output.address(region,wanted),Output.address_congruence(region,actual,wanted,channel_exact(problem,enabled,fits),row_matches(problem,enabled,fits),column_matches(problem,enabled,fits)))
