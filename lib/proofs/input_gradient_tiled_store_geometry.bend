# Four-channel store tails and packed eight-channel weight blocks agree.
import Base
import ./kernel_strided_store_writes.bend as StoreModel
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_weight_reads.bend as Weights
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./input_gradient_phase_domain.bend as Domain
import ./kernel_accumulator.bend as Accumulator
import ./kernel_coordinate_refinement.bend as Coordinates
import ./kernel_weight_gradient_packing.bend as Patches
import ./kernel_gemm_readback.bend as Gemm
import ./input_gradient_tiled_job_store.bend as Store
import ../math_sequence.bend as Sequence
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as Division
import ./machine_limit_bounds.bend as Limits

type Channels is Data:
  Channels{phase: Jobs.Phase,unit: Nat,inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool}}
def phase(job: Channels) -> Jobs.Phase:
  Channels{phase,unit,inside} = job
  phase
def unit(job: Channels) -> Nat:
  Channels{phase,unit,inside} = job
  unit
def first(+job: Channels) -> Nat:
  Nat.mul(unit(job),4n)
def count(+job: Channels) -> U32:
  U32.min(4,(Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 4 : U32))
def first_inside(+job: Channels) -> {Nat.is_lt(first(job),Domain.channels(Jobs.geometry(phase(job)))) == True{} : Bool}:
  Channels{+phase,+unit,+inside} = job
  Jobs.Phase{+call,+e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  %Ceiling.upper_comparison(Domain.channels(g),4n,unit,{==}) : {_ == True{} : Bool}
  %Jobs.units_exact(call,e) : {Nat.is_lt(unit,_) == True{} : Bool}
  inside
def unit_exact(+job: Channels) -> {U32.to_nat(U32.from_nat(unit(job))) == unit(job) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  Bounds.roundtrip(unit,cap,Limits.capacity_fits(l),Order.transitive(unit,Domain.channels(g),U32.to_nat(cap),Order.transitive(unit,first(job),Domain.channels(g),RowMajor.product_covers(unit,4n,{==}),Bounds.strict_le(first(job),Domain.channels(g),first_inside(job))),Domain.channels_capacity_room(call)))
def first_word_exact(+job: Channels) -> {U32.to_nat((U32.from_nat(unit(job)) * 4 : U32)) == first(job) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  %unit_exact(job) : {U32.to_nat((U32.from_nat(unit) * 4 : U32)) == Nat.mul(_,4n) : Nat}
  Arithmetic.multiply(U32.from_nat(unit),4,cap,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(unit)),unit,unit_exact(job)) : {Nat.is_le(Nat.mul(_,4n),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(first(job),Domain.channels(g),U32.to_nat(cap),Bounds.strict_le(first(job),Domain.channels(g),first_inside(job)),Domain.channels_capacity_room(call)))
def remaining_exact(+job: Channels) -> {U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 4 : U32)) == Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job)) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  %first_word_exact(job) : {U32.to_nat((Domain.channel_word(g) - U32.from_nat(unit) * 4 : U32)) == Nat.sub(Domain.channels(g),_) : Nat}
  Arithmetic.subtract(Domain.channel_word(g),(U32.from_nat(unit) * 4 : U32),cap,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(unit) * 4 : U32)),first(job),first_word_exact(job)) : {Nat.is_le(_,Domain.channels(g)) == True{} : Bool} Bounds.strict_le(first(job),Domain.channels(g),first_inside(job)),
    Order.transitive(Nat.sub(Domain.channels(g),U32.to_nat((U32.from_nat(unit) * 4 : U32))),Domain.channels(g),U32.to_nat(cap),Order.sub_le(Domain.channels(g),U32.to_nat((U32.from_nat(unit) * 4 : U32))),Domain.channels_capacity_room(call)))
def count_exact(+job: Channels) -> {U32.to_nat(count(job)) == Nat.min(4n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))) : Nat}:
  Equal.trans(Nat,U32.to_nat(count(job)),Nat.min(4n,U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 4 : U32))),Nat.min(4n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))),NativeExtrema.minimum(4,(Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 4 : U32)),
    Equal.cong(Nat,Nat,value => Nat.min(4n,value),U32.to_nat((Domain.channel_word(Jobs.geometry(phase(job))) - U32.from_nat(unit(job)) * 4 : U32)),Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job)),remaining_exact(job)))
def row_inside(+job: Channels,+row: Nat,+inside: {Nat.is_lt(row,U32.to_nat(count(job))) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first(job),row),Domain.channels(Jobs.geometry(phase(job)))) == True{} : Bool}:
  %Order.subtraction_lt(Domain.channels(Jobs.geometry(phase(job))),first(job),row) : {_ == True{} : Bool}
  Bounds.nat_trans(row,U32.to_nat(count(job)),Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job)),inside,
    %Equal.sym(Nat,U32.to_nat(count(job)),Nat.min(4n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))),count_exact(job)) : {Nat.is_le(_,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))) == True{} : Bool}
    Extrema.minimum_right(4n,Nat.sub(Domain.channels(Jobs.geometry(phase(job))),first(job))))
def block_first_inside(+job: Channels) -> {Nat.is_lt(Nat.mul(Nat.div(unit(job),2n),8n),Domain.channels(Jobs.geometry(phase(job)))) == True{} : Bool}:
  Channels{+phase,+unit,+inside} = job
  %Ceiling.upper_comparison(Domain.channels(Jobs.geometry(phase)),8n,Nat.div(unit,2n),{==}) : {_ == True{} : Bool}
  Weights.block_inside(phase,unit,inside)
def block_word_exact(+job: Channels) -> {U32.to_nat((U32.from_nat(Nat.div(unit(job),2n)) * 8 : U32)) == Nat.mul(Nat.div(unit(job),2n),8n) : Nat}:
  Channels{+phase,+unit,inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  +block = Nat.div(unit,2n)
  +product = Nat.mul(block,8n)
  +room = Order.transitive(product,Domain.channels(g),U32.to_nat(cap),Bounds.strict_le(product,Domain.channels(g),block_first_inside(job)),Domain.channels_capacity_room(call))
  +block_room = Order.transitive(block,product,U32.to_nat(cap),RowMajor.product_covers(block,8n,{==}),room)
  +exact = Bounds.roundtrip(block,cap,Limits.capacity_fits(l),block_room)
  %exact : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(U32.from_nat(block),8,cap,Limits.capacity_fits(l),%Equal.sym(Nat,U32.to_nat(U32.from_nat(block)),block,exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(cap)) == True{} : Bool} room)
def patch_parity(+unit: Nat,+row: Accumulator.Row) -> {Patches.patch_key(Nat.is_eq(Nat.mod(unit,2n),1n),row) == Nat.add(Nat.mul(Nat.mod(unit,2n),4n),Store.row_number(row)) : Nat}:
  +parity = Nat.mod(unit,2n)
  +key = Nat.add(Nat.mul(parity,4n),Store.row_number(row))
  +small = Division.remainder_small(unit,2n,{==})
  +lane_same = Gemm.parity_lane(Nat.is_eq(parity,1n),parity,Store.row_number(row),{==},small,Store.row_small(row))
  Equal.trans(Nat,Patches.patch_key(Nat.is_eq(parity,1n),row),Sequence.index_value(8n,Coordinates.weight_lane(Nat.is_eq(parity,1n),row)),key,
    Equal.sym(Nat,Sequence.index_value(8n,Coordinates.weight_lane(Nat.is_eq(parity,1n),row)),Patches.patch_key(Nat.is_eq(parity,1n),row),Coordinates.weight_coordinate(Nat.is_eq(parity,1n),row)),
    Equal.trans(Nat,Sequence.index_value(8n,Coordinates.weight_lane(Nat.is_eq(parity,1n),row)),Sequence.index_value(8n,TileLaws.lane_of(key)),key,
      Equal.cong(Sequence.Index(8n),Nat,index => Sequence.index_value(8n,index),Coordinates.weight_lane(Nat.is_eq(parity,1n),row),TileLaws.lane_of(key),
        %Store.row_matches(row) : {Coordinates.weight_lane(Nat.is_eq(parity,1n),_) == TileLaws.lane_of(key) : Sequence.Index(8n)} lane_same),
      Gemm.lane_value_of(key,Gemm.half_row_inside(parity,Store.row_number(row),small,Store.row_small(row)))))
def channel_key(+job: Channels,+row: Accumulator.Row) ->
  {Nat.add(U32.to_nat((U32.from_nat(Nat.div(unit(job),2n)) * 8 : U32)),Patches.patch_key(Nat.is_eq(Nat.mod(unit(job),2n),1n),row)) == Nat.add(first(job),Store.row_number(row)) : Nat}:
  +unit = unit(job)
  +block = Nat.div(unit,2n)
  +parity = Nat.mod(unit,2n)
  +position = Nat.add(first(job),Store.row_number(row))
  %Equal.sym(Nat,U32.to_nat((U32.from_nat(block) * 8 : U32)),Nat.mul(block,8n),block_word_exact(job)) : {Nat.add(_,Patches.patch_key(Nat.is_eq(parity,1n),row)) == position : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(block,8n),Patches.patch_key(Nat.is_eq(parity,1n),row)),Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),Store.row_number(row))),position,
    Equal.cong(Nat,Nat,lane => Nat.add(Nat.mul(block,8n),lane),Patches.patch_key(Nat.is_eq(parity,1n),row),Nat.add(Nat.mul(parity,4n),Store.row_number(row)),patch_parity(unit,row)),
    Equal.sym(Nat,position,Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),Store.row_number(row))),Gemm.eight_split(position,unit,Store.row_number(row),block,parity,{==},Division.quotient_exact(unit,2n,{==}))))
def channel_job(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) -> Channels:
  +work = Chunks.work(Chunks.selection(seed))
  Channels{Chunks.phase(Chunks.selection(seed)),Chunks.unit(seed),
    Jobs.unit_inside(work,Chunks.line(seed),Chunks.local_group(seed),Chunks.unit(seed),Chunks.job_inside(seed,valid))}
def rows_matches(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>) ->
  {StoreModel.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))) == count(channel_job(seed,valid)) : U32}:
  Chunks.Reduction{+selected,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +work = Chunks.work(selected)
  %Equal.sym(U32,Runtime.plan_unit(work,wb,ob,line,group,unit),U32.from_nat(unit),Runtime.plan_unit_exact(work,Jobs.active_phase(Chunks.phase(selected),enabled),wb,ob,line,group,unit,ji)) : {U32.min(4,(c - _ * 4 : U32)) == count(channel_job(seed,valid)) : U32}
  {==}
def packed_channel_inside(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row: Accumulator.Row,
  +inside: {Nat.is_lt(Store.row_number(row),U32.to_nat(StoreModel.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat((U32.from_nat(Nat.div(Chunks.unit(seed),2n)) * 8 : U32)),Patches.patch_key(Store.upper(seed),row)),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool}:
  Chunks.Reduction{+selected,line,group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,leaf_first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,contained,fits,enabled,+ji,oi,ri,coli} = valid
  Jobs.Inside{li,gi,ui} = ji
  +job = channel_job(seed,valid)
  %Equal.sym(Nat,Nat.add(U32.to_nat((U32.from_nat(Nat.div(unit,2n)) * 8 : U32)),Patches.patch_key(Store.upper(seed),row)),Nat.add(first(job),Store.row_number(row)),channel_key(job,row)) : {Nat.is_lt(_,Domain.channels(g)) == True{} : Bool}
  row_inside(job,Store.row_number(row),%rows_matches(seed,valid) : {Nat.is_lt(Store.row_number(row),U32.to_nat(_)) == True{} : Bool} inside)
def channel_word(+job: Channels,+row: Accumulator.Row) -> U32:
  (U32.from_nat(unit(job)) * 4 + U32.from_nat(Store.row_number(row)) : U32)
def channel_word_exact(+job: Channels,+row: Accumulator.Row,+inside: {Nat.is_lt(Store.row_number(row),U32.to_nat(count(job))) == True{} : Bool}) ->
  {U32.to_nat(channel_word(job,row)) == Nat.add(first(job),Store.row_number(row)) : Nat}:
  Channels{+phase,+unit,unit_inside} = job
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +cap = Machine.capacity(l)
  +number = Store.row_number(row)
  +row_exact = Bounds.roundtrip(number,4,{==},Bounds.strict_le(number,4n,Store.row_small(row)))
  +room = Order.transitive(Nat.add(first(job),number),Domain.channels(g),U32.to_nat(cap),Bounds.strict_le(Nat.add(first(job),number),Domain.channels(g),row_inside(job,number,inside)),Domain.channels_capacity_room(call))
  %first_word_exact(job) : {U32.to_nat(channel_word(job,row)) == Nat.add(_,number) : Nat}
  %row_exact : {U32.to_nat(channel_word(job,row)) == Nat.add(U32.to_nat((U32.from_nat(unit) * 4 : U32)),_) : Nat}
  Arithmetic.add((U32.from_nat(unit) * 4 : U32),U32.from_nat(number),cap,Limits.capacity_fits(l),
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(unit) * 4 : U32)),first(job),first_word_exact(job)) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(number))),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,row_exact) : {Nat.is_le(Nat.add(first(job),_),U32.to_nat(cap)) == True{} : Bool}
    room)
