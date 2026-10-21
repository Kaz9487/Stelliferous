# Column groups and the final eight-lane tail stay inside the phase width.
import Base
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./kernel_strided_store_writes.bend as StoreModel
import ./storage_shape.bend as Shape
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_tiled_value_reads.bend as Input
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_geometry.bend as PhaseGeometry
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits

type Columns is Data:
  Columns{work: Jobs.Work,active: Jobs.Active<Jobs.phase(work)>,group: Nat,inside: {Nat.is_lt(group,U32.to_nat(Jobs.groups(work))) == True{} : Bool}}
def work(job: Columns) -> Jobs.Work:
  Columns{work,active,group,inside} = job
  work
def group(job: Columns) -> Nat:
  Columns{work,active,group,inside} = job
  group
def group_position(+job: Columns) -> Nat:
  Nat.add(U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(job)))),group(job))
def first(+job: Columns) -> Nat:
  Nat.mul(group_position(job),8n)
def first_word(+job: Columns) -> U32:
  ((Jobs.chunk_group(Jobs.chunk(work(job))) + U32.from_nat(group(job))) * 8 : U32)
def count_word(+job: Columns) -> U32:
  Jobs.column_count(Jobs.phase(work(job)))
def count(+job: Columns) -> U32:
  U32.min(8,(count_word(job) - first_word(job) : U32))
def group_inside(+job: Columns) -> {Nat.is_lt(group_position(job),U32.to_nat(Jobs.native_groups(Jobs.phase(work(job))))) == True{} : Bool}:
  Columns{+work,active,+group,inside} = job
  Jobs.group_span_location(Jobs.chunk_group(Jobs.chunk(work)),Jobs.native_groups(Jobs.phase(work)),Lookup.layout_groups(Lookup.chunk_layout(Jobs.chunk(work))),group,inside)
def phase_groups_exact(+job: Columns) -> {U32.to_nat(Jobs.native_groups(Jobs.phase(work(job)))) == Ceiling.ceiling(U32.to_nat(count_word(job)),8n) : Nat}:
  Columns{+work,+active,group,inside} = job
  Jobs.Work{+phase,chunk} = work
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Jobs.phase_groups_exact(Jobs.columns(phase),x,xi,Jobs.column_active(phase,active))
def first_inside(+job: Columns) -> {Nat.is_lt(first(job),U32.to_nat(count_word(job))) == True{} : Bool}:
  %Ceiling.upper_comparison(U32.to_nat(count_word(job)),8n,group_position(job),{==}) : {_ == True{} : Bool}
  %phase_groups_exact(job) : {Nat.is_lt(group_position(job),_) == True{} : Bool}
  group_inside(job)
def count_room(+job: Columns) -> {Nat.is_le(U32.to_nat(count_word(job)),U32.to_nat(Machine.capacity(Jobs.limits(Jobs.phase(work(job)))))) == True{} : Bool}:
  Columns{+work,+active,group,inside} = job
  Jobs.Work{+phase,chunk} = work
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Jobs.phase_count_room(Jobs.columns(phase),Jobs.phase_x(phase),xi,Jobs.column_active(phase,active))
def group_position_room(+job: Columns) -> {Nat.is_le(group_position(job),U32.to_nat(Machine.capacity(Jobs.limits(Jobs.phase(work(job)))))) == True{} : Bool}:
  Order.transitive(group_position(job),first(job),U32.to_nat(Machine.capacity(Jobs.limits(Jobs.phase(work(job))))),RowMajor.product_covers(group_position(job),8n,{==}),Order.transitive(first(job),U32.to_nat(count_word(job)),U32.to_nat(Machine.capacity(Jobs.limits(Jobs.phase(work(job))))),Bounds.strict_le(first(job),U32.to_nat(count_word(job)),first_inside(job)),count_room(job)))
def group_word_exact(+job: Columns) -> {U32.to_nat(U32.from_nat(group(job))) == group(job) : Nat}:
  +cap = Machine.capacity(Jobs.limits(Jobs.phase(work(job))))
  Bounds.roundtrip(group(job),cap,Limits.capacity_fits(Jobs.limits(Jobs.phase(work(job)))),Order.transitive(group(job),group_position(job),U32.to_nat(cap),Order.drop_left_le(U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(job)))),group(job),group_position(job),Order.reflexive(group_position(job))),group_position_room(job)))
def group_sum_exact(+job: Columns) -> {U32.to_nat((Jobs.chunk_group(Jobs.chunk(work(job))) + U32.from_nat(group(job)) : U32)) == group_position(job) : Nat}:
  +cap = Machine.capacity(Jobs.limits(Jobs.phase(work(job))))
  %group_word_exact(job) : {U32.to_nat((Jobs.chunk_group(Jobs.chunk(work(job))) + U32.from_nat(group(job)) : U32)) == Nat.add(U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(job)))),_) : Nat}
  Arithmetic.add(Jobs.chunk_group(Jobs.chunk(work(job))),U32.from_nat(group(job)),cap,Limits.capacity_fits(Jobs.limits(Jobs.phase(work(job)))),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(group(job))),group(job),group_word_exact(job)) : {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_group(Jobs.chunk(work(job)))),_),U32.to_nat(cap)) == True{} : Bool}
    group_position_room(job))
def first_word_exact(+job: Columns) -> {U32.to_nat(first_word(job)) == first(job) : Nat}:
  +cap = Machine.capacity(Jobs.limits(Jobs.phase(work(job))))
  %group_sum_exact(job) : {U32.to_nat(first_word(job)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply((Jobs.chunk_group(Jobs.chunk(work(job))) + U32.from_nat(group(job)) : U32),8,cap,Limits.capacity_fits(Jobs.limits(Jobs.phase(work(job)))),
    %Equal.sym(Nat,U32.to_nat((Jobs.chunk_group(Jobs.chunk(work(job))) + U32.from_nat(group(job)) : U32)),group_position(job),group_sum_exact(job)) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(first(job),U32.to_nat(count_word(job)),U32.to_nat(cap),Bounds.strict_le(first(job),U32.to_nat(count_word(job)),first_inside(job)),count_room(job)))
def remaining_exact(+job: Columns) -> {U32.to_nat((count_word(job) - first_word(job) : U32)) == Nat.sub(U32.to_nat(count_word(job)),first(job)) : Nat}:
  +cap = Machine.capacity(Jobs.limits(Jobs.phase(work(job))))
  %first_word_exact(job) : {U32.to_nat((count_word(job) - first_word(job) : U32)) == Nat.sub(U32.to_nat(count_word(job)),_) : Nat}
  Arithmetic.subtract(count_word(job),first_word(job),cap,Limits.capacity_fits(Jobs.limits(Jobs.phase(work(job)))),
    %Equal.sym(Nat,U32.to_nat(first_word(job)),first(job),first_word_exact(job)) : {Nat.is_le(_,U32.to_nat(count_word(job))) == True{} : Bool} Bounds.strict_le(first(job),U32.to_nat(count_word(job)),first_inside(job)),
    Order.transitive(Nat.sub(U32.to_nat(count_word(job)),U32.to_nat(first_word(job))),U32.to_nat(count_word(job)),U32.to_nat(cap),Order.sub_le(U32.to_nat(count_word(job)),U32.to_nat(first_word(job))),count_room(job)))
def count_exact(+job: Columns) -> {U32.to_nat(count(job)) == Nat.min(8n,Nat.sub(U32.to_nat(count_word(job)),first(job))) : Nat}:
  Equal.trans(Nat,U32.to_nat(count(job)),Nat.min(8n,U32.to_nat((count_word(job) - first_word(job) : U32))),Nat.min(8n,Nat.sub(U32.to_nat(count_word(job)),first(job))),NativeExtrema.minimum(8,(count_word(job) - first_word(job) : U32)),Equal.cong(Nat,Nat,value => Nat.min(8n,value),U32.to_nat((count_word(job) - first_word(job) : U32)),Nat.sub(U32.to_nat(count_word(job)),first(job)),remaining_exact(job)))
def lane_inside(+job: Columns,+lane: Nat,+inside: {Nat.is_lt(lane,U32.to_nat(count(job))) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first(job),lane),U32.to_nat(count_word(job))) == True{} : Bool}:
  %Order.subtraction_lt(U32.to_nat(count_word(job)),first(job),lane) : {_ == True{} : Bool}
  Bounds.nat_trans(lane,U32.to_nat(count(job)),Nat.sub(U32.to_nat(count_word(job)),first(job)),inside,
    %Equal.sym(Nat,U32.to_nat(count(job)),Nat.min(8n,Nat.sub(U32.to_nat(count_word(job)),first(job))),count_exact(job)) : {Nat.is_le(_,Nat.sub(U32.to_nat(count_word(job)),first(job))) == True{} : Bool}
    Extrema.minimum_right(8n,Nat.sub(U32.to_nat(count_word(job)),first(job))))
def distributed_first(+job: Columns) -> {((Jobs.chunk_group(Jobs.chunk(work(job))) * 8) + (U32.from_nat(group(job)) * 8) : U32) == first_word(job) : U32}:
  Equal.sym(U32,first_word(job),((Jobs.chunk_group(Jobs.chunk(work(job))) * 8) + (U32.from_nat(group(job)) * 8) : U32),Arithmetic.multiply_additive(Jobs.chunk_group(Jobs.chunk(work(job))),U32.from_nat(group(job)),8))
def column_job(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> Columns:
  Chunks.Reduction{+selected,line,+local_group,unit,out,tr,tc} = seed
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  Columns{Chunks.work(selected),Jobs.active_phase(Chunks.phase(selected),enabled),local_group,gi}
def input_column_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> {Input.input_column(seed) == first_word(column_job(seed,valid)) : U32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +work = Chunks.work(selected)
  +sample = Chunks.sample(seed)
  +relative = Runtime.tap(seed)
  +same = Shape.first({Lookup.read_group(sample) == U32.from_nat(local_group) : U32},{Lookup.read_line(sample) == U32.from_nat(line) : U32},Reads.read_job_counters(work,Jobs.active_phase(Chunks.phase(selected),enabled),wb,ob,line,local_group,unit,U32.from_nat(relative),ji))
  Equal.cong(U32,U32,value => ((Jobs.chunk_group(Chunks.chunk(query)) + value) * 8 : U32),Lookup.read_group(sample),U32.from_nat(local_group),same)
def counts_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> {StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))) == count(column_job(seed,valid)) : U32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +work = Chunks.work(selected)
  +phase = Chunks.phase(selected)
  +job = column_job(seed,valid)
  +base = Jobs.chunk_group(Chunks.chunk(query))
  +local = U32.from_nat(local_group)
  +total = count_word(job)
  +sample = Chunks.sample(seed)
  +same = Shape.first({Lookup.read_group(sample) == local : U32},{Lookup.read_line(sample) == U32.from_nat(line) : U32},Reads.read_job_counters(work,Jobs.active_phase(phase,enabled),wb,ob,line,local_group,unit,U32.from_nat(Runtime.tap(seed)),ji))
  %Equal.sym(U32,Lookup.read_group(sample),local,same) : {U32.min(8,((total - base * 8 : U32) - _ * 8 : U32)) == count(job) : U32}
  Equal.cong(U32,U32,value => U32.min(8,value),((total - base * 8 : U32) - local * 8 : U32),(total - first_word(job) : U32),
    Equal.trans(U32,((total - base * 8 : U32) - local * 8 : U32),(total - ((base * 8) + (local * 8) : U32) : U32),(total - first_word(job) : U32),Arithmetic.subtract_sum(total,(base * 8 : U32),(local * 8 : U32)),Equal.cong(U32,U32,value => (total - value : U32),((base * 8) + (local * 8) : U32),first_word(job),distributed_first(job))))
def count_phase_matches(+job: Columns) -> {count_word(job) == PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Jobs.phase(work(job))))) : U32}:
  Columns{+work,active,group,inside} = job
  Jobs.Work{+phase,chunk} = work
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def packed_column_inside(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+lane: Nat,
  +inside: {Nat.is_lt(lane,U32.to_nat(StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(seed)),lane),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(seed))))))) == True{} : Bool}:
  Chunks.Reduction{+selected,line,local_group,unit,out,tr,tc} = seed
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +job = column_job(seed,valid)
  %count_phase_matches(job) : {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(seed)),lane),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(U32,Input.input_column(seed),first_word(job),input_column_matches(seed,valid)) : {Nat.is_lt(Nat.add(U32.to_nat(_),lane),U32.to_nat(count_word(job))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(first_word(job)),first(job),first_word_exact(job)) : {Nat.is_lt(Nat.add(_,lane),U32.to_nat(count_word(job))) == True{} : Bool}
  lane_inside(job,lane,%counts_matches(seed,valid) : {Nat.is_lt(lane,U32.to_nat(_)) == True{} : Bool} inside)
def physical_column(+seed: Chunks.Reduction,+lane: Nat) -> Nat:
  PhaseBounds.pixel(U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))),Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))),(Input.input_column(seed) + U32.from_nat(lane) : U32))
def pixel_inside(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: U32,+offset: Nat,+gradient: U32,+capacity: U32,
  facts: PhaseBounds.AxisFacts<first,size,kernel,stride,padding,position,offset,gradient,capacity>) -> {Nat.is_lt(PhaseBounds.pixel(first,stride,position),U32.to_nat(size)) == True{} : Bool}:
  PhaseBounds.AxisFacts{residue,+point,pixel_radix,tap_inside,tap_radix,counter,source_room} = facts
  point
def physical_column_inside(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+lane: Nat,
  +inside: {Nat.is_lt(lane,U32.to_nat(StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) ->
  {Nat.is_lt(physical_column(seed,lane),Domain.width(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool}:
  Chunks.Reduction{+selected,line,local_group,unit,out,tr,+tc} = seed
  Chunks.Selection{+query,y,+x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,+w,+k,+s,+p,gh,+gw} = g
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,+coli} = valid
  +first = U32.from_nat(x)
  +position = Input.input_column(seed)
  pixel_inside(first,w,k,s,p,(position + U32.from_nat(lane) : U32),tc,gw,Machine.coordinate(l),PhaseBounds.column_lane_facts(call,first,position,lane,tc,Reads.phase_x_inside(Chunks.phase(selected)),packed_column_inside(seed,valid,lane,inside),coli))
def physical_word(+seed: Chunks.Reduction,+lane: Nat) -> U32:
  (U32.from_nat(Jobs.phase_x(Chunks.phase(Chunks.selection(seed)))) + (Input.input_column(seed) + U32.from_nat(lane) : U32) * Domain.stride_word(Jobs.geometry(Chunks.phase(Chunks.selection(seed)))) : U32)
def physical_word_exact(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+lane: Nat,
  +inside: {Nat.is_lt(lane,U32.to_nat(StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) ->
  {U32.to_nat(physical_word(seed,lane)) == physical_column(seed,lane) : Nat}:
  Chunks.Reduction{+selected,line,local_group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,+x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,+w,+k,+s,+p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,ji,oi,ri,coli} = valid
  +first = U32.from_nat(x)
  +position = Input.input_column(seed)
  +point = (position + U32.from_nat(lane) : U32)
  +first_inside = Reads.phase_x_inside(Chunks.phase(selected))
  +packed_inside = packed_column_inside(seed,valid,lane,inside)
  +position_exact = PhaseBounds.lane_position_exact(call,first,position,lane,first_inside,packed_inside)
  PhaseGeometry.position_exact(first,w,k,s,p,point,Machine.coordinate(l),Limits.coordinate_fits(l),PhaseBounds.width_coordinate_room(call),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),first_inside,
    %Equal.sym(Nat,U32.to_nat(point),Nat.add(U32.to_nat(position),lane),position_exact) : {Nat.is_lt(_,U32.to_nat(PhaseGeometry.axis_count(Kernel.axis(first,w,k,s,p)))) == True{} : Bool} packed_inside)
