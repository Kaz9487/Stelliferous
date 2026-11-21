# Unit u of a phase job stores the lanes of channels 8 * u .. 8 * u + min(8, c - 8 * u).
import Base
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_phase_domain.bend as Domain
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_gemm_writes.bend as GemmWrites
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded
import ./u32_comparison.bend as Comparison
import ./word_multiplication.bend as WordMultiplication

type Channels is Data:
  Channels{phase: Jobs.Phase,unit: Nat,inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool}}
def phase(job: Channels) -> Jobs.Phase:
  Channels{phase,unit,inside} = job
  phase
def unit(job: Channels) -> Nat:
  Channels{phase,unit,inside} = job
  unit
def first(+job: Channels) -> Nat:
  Nat.mul(unit(job),8n)
def count(+job: Channels) -> U32:
  U32.min(8,(Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 8 : U32))
def first_inside(+job: Channels) -> {Nat.is_lt(first(job),Domain.channels(Jobs.geometry(phase(job)))) == True{} : Bool}:
  Channels{+phase,+unit,+inside} = job
  Jobs.Phase{+call,+e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  %Ceiling.upper_comparison(Domain.channels(g),8n,unit,{==}) : {_ == True{} : Bool}
  %Jobs.units_exact(call,e) : {Nat.is_lt(unit,_) == True{} : Bool}
  inside
def unit_exact(+job: Channels) -> {U32.to_nat(U32.from_nat(unit(job))) == unit(job) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  Bounds.roundtrip(unit,cap,Limits.capacity_fits(l),Order.transitive(unit,Domain.channels(g),U32.to_nat(cap),Order.transitive(unit,first(job),Domain.channels(g),RowMajor.product_covers(unit,8n,{==}),Bounds.strict_le(first(job),Domain.channels(g),first_inside(job))),Domain.channels_capacity_room(call)))
def first_word_exact(+job: Channels) -> {U32.to_nat((U32.from_nat(unit(job)) * 8 : U32)) == first(job) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  %unit_exact(job) : {U32.to_nat((U32.from_nat(unit) * 8 : U32)) == Nat.mul(_,8n) : Nat}
  WordMultiplication.bounded_multiply(U32.from_nat(unit),8,cap,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(unit)),unit,unit_exact(job)) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(first(job),Domain.channels(g),U32.to_nat(cap),Bounds.strict_le(first(job),Domain.channels(g),first_inside(job)),Domain.channels_capacity_room(call)))
def remaining_exact(+job: Channels) -> {U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 8 : U32)) == Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job)) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  %first_word_exact(job) : {U32.to_nat((Domain.channel_word(g) - U32.from_nat(unit) * 8 : U32)) == Nat.sub(Domain.channels(g),_) : Nat}
  Arithmetic.subtract(Domain.channel_word(g),(U32.from_nat(unit) * 8 : U32),cap,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(unit) * 8 : U32)),first(job),first_word_exact(job)) : {Nat.is_le(_,Domain.channels(g)) == True{} : Bool} Bounds.strict_le(first(job),Domain.channels(g),first_inside(job)),
    Order.transitive(Nat.sub(Domain.channels(g),U32.to_nat((U32.from_nat(unit) * 8 : U32))),Domain.channels(g),U32.to_nat(cap),Order.sub_le(Domain.channels(g),U32.to_nat((U32.from_nat(unit) * 8 : U32))),Domain.channels_capacity_room(call)))
def count_exact(+job: Channels) -> {U32.to_nat(count(job)) == Nat.min(8n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))) : Nat}:
  Equal.trans(Nat,U32.to_nat(count(job)),Nat.min(8n,U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 8 : U32))),Nat.min(8n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))),NativeExtrema.minimum(8,(Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 8 : U32)),
    Equal.cong(Nat,Nat,value => Nat.min(8n,value),U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 8 : U32)),Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job)),remaining_exact(job)))
def channel_job(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> Channels:
  +work = Chunks.work(Chunks.selection(seed))
  Channels{Chunks.phase(Chunks.selection(seed)),Chunks.unit(seed),
    Jobs.unit_inside(work,Chunks.line(seed),Chunks.local_group(seed),Chunks.unit(seed),Chunks.job_inside(seed,valid))}
def lanes_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) ->
  {GemmWrites.store_lanes(GemmJobs.job_store(~Kernel.Cursor(),Runtime.job(seed))) == count(channel_job(seed,valid)) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,source,chunk_row,chunk_group,+wb} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  +work = Chunks.work(selected)
  +native_unit = (Runtime.job_index(seed) % Jobs.native_units(work) : U32)
  %Equal.sym(U32,native_unit,U32.from_nat(unit),Encoded.encoding(native_unit,unit,Jobs.unit_decode(work,Jobs.active_phase(Chunks.phase(selected),enabled),line,group,unit,ji))) :
    {U32.min(8,(c - _ * 8 : U32)) == count(channel_job(seed,valid)) : U32}
  {==}
# The unit of a job is a group of the channels.
def unit_below(+job: Channels) -> {Nat.is_lt(unit(job),Ceiling.ceiling(Domain.channels(Jobs.geometry(phase(job))),8n)) == True{} : Bool}:
  Channels{+phase,+unit,+inside} = job
  Jobs.Phase{+call,+e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  %Jobs.units_exact(call,e) : {Nat.is_lt(unit,_) == True{} : Bool}
  inside
# A lane of a job's unit that is a channel is a lane the job stores.
def lane_counted(+job: Channels,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(first(job),lane),Domain.channels(Jobs.geometry(phase(job)))) == True{} : Bool}) -> {Nat.is_lt(lane,U32.to_nat(count(job))) == True{} : Bool}:
  +total = Domain.channels(Jobs.geometry(phase(job)))
  +start = first(job)
  %Equal.sym(Nat,U32.to_nat(count(job)),Nat.min(8n,Nat.sub(total,start)),count_exact(job)) : {Nat.is_lt(lane,_) == True{} : Bool}
  Extrema.minimum_strict_lower(8n,Nat.sub(total,start),lane,small,
    %Equal.sym(Bool,Nat.is_lt(lane,Nat.sub(total,start)),Nat.is_lt(Nat.add(start,lane),total),Order.subtraction_lt(total,start,lane)) : {_ == True{} : Bool}
    inside)
def lane_stored(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(first(channel_job(seed,valid)),lane),Domain.channels(Jobs.geometry(phase(channel_job(seed,valid))))) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(lane),GemmWrites.store_lanes(GemmJobs.job_store(~Kernel.Cursor(),Runtime.job(seed)))) == True{} : Bool}:
  +job = channel_job(seed,valid)
  +lane_exact = Bounds.roundtrip(lane,8,{==},Bounds.strict_le(lane,8n,small))
  %Equal.sym(U32,GemmWrites.store_lanes(GemmJobs.job_store(~Kernel.Cursor(),Runtime.job(seed))),count(job),lanes_matches(seed,valid)) :
    {U32.is_lt(U32.from_nat(lane),_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),count(job)),Nat.is_lt(U32.to_nat(U32.from_nat(lane)),U32.to_nat(count(job))),Comparison.lt(U32.from_nat(lane),count(job))) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_lt(_,U32.to_nat(count(job))) == True{} : Bool}
  lane_counted(job,lane,small,inside)
