# Actual job read pointers coincide with the complete phase-packing addresses.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_phase_index.bend as Index
import ./input_gradient_tiled_pack_values.bend as Packed
import ./input_gradient_tiled_packing_bounds.bend as PackingBounds
import ./input_gradient_tiled_weight_reads.bend as Weights
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ./input_gradient_tiled_reduction.bend as Reduction
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ../kernel_tile.bend as Tile
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_tiled_weights.bend as PackedWeights
import ./input_gradient_tiled_offsets.bend as Offsets
import ./nat_ceiling.bend as Ceiling
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./encoded_coordinates.bend as Encoded
import ./u32_division.bend as Division
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./word_arithmetic.bend as WordArithmetic
import ./input_gradient_tiled_reduction.bend as Natural

def weight_bounds(+phase: Jobs.Phase) -> PackingBounds.Query:
  Packed.reflag(True{},Index.packing_bounds(phase))
def offset_bounds(+phase: Jobs.Phase) -> PackingBounds.Query:
  Packed.reflag(False{},Index.packing_bounds(phase))
def weight_base(+phase: Jobs.Phase) -> U32:
  PackingBounds.prefix_word(weight_bounds(phase),Index.ordinal(phase))
def offset_base(+phase: Jobs.Phase) -> U32:
  PackingBounds.prefix_word(offset_bounds(phase),Index.ordinal(phase))
# The packing bases of a selection are the prefix sums at its phase.
type Bases<-selection: Chunks.Selection> is Data:
  Bases{weights: {Chunks.weight_base(Chunks.query(selection)) == weight_base(Chunks.phase(selection)) : U32},
    offsets: {Chunks.offset_base(Chunks.query(selection)) == offset_base(Chunks.phase(selection)) : U32}}
def context_weight(context: Kernel.Context) -> U32:
  Kernel.Context{source,weight} = context
  weight
def plan_unit(+work: Jobs.Work,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat) -> U32:
  (Jobs.job_index(work,line,group,unit) % Jobs.plan_units(Execution.work_plan(Execution.work(Jobs.context(Jobs.phase(work)),Jobs.chunk(work),wb,ob))) : U32)
def plan_unit_exact(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+inside: Jobs.Inside<work,line,group,unit>) ->
  {plan_unit(work,wb,ob,line,group,unit) == U32.from_nat(unit) : U32}:
  Jobs.Work{Jobs.Phase{+call,e,y,x,yi,xi},+chunk} = work
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr0} = layout
  Encoded.encoding(plan_unit(work,wb,ob,line,group,unit),unit,Jobs.unit_decode(work,active,line,group,unit,inside))
def unit_half_exact(+phase: Jobs.Phase,+unit: Nat,+inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool}) ->
  {(U32.from_nat(unit) / 2 : U32) == U32.from_nat(Nat.div(unit,2n)) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  +cap = Machine.capacity(l)
  +room = Jobs.phase_units_room(phase)
  +unit_room = Order.transitive(U32.to_nat(Jobs.units(g)),Domain.channels(g),U32.to_nat(cap),room,Domain.channels_capacity_room(call))
  +exact = Bounds.roundtrip(unit,cap,Limits.capacity_fits(l),Bounds.strict_le(unit,U32.to_nat(cap),Bounds.nat_trans(unit,U32.to_nat(Jobs.units(g)),U32.to_nat(cap),inside,unit_room)))
  Encoded.encoding((U32.from_nat(unit) / 2 : U32),Nat.div(unit,2n),
    %exact : {U32.to_nat((U32.from_nat(unit) / 2 : U32)) == Nat.div(_,2n) : Nat} Division.divide(U32.from_nat(unit),2,Machine.half_word(l),Limits.half_word_fits(l),
      Order.transitive(4n,7n,U32.to_nat(Machine.half_word(l)),{==},Order.transitive(7n,U32.to_nat(cap),U32.to_nat(Machine.half_word(l)),Bounds.as_nat_le(7,cap,Limits.seven_within_capacity(l)),Bounds.as_nat_le(cap,Machine.half_word(l),Limits.capacity_within_half(l)))),{==}))

def weight_pointer_matches(+work: Jobs.Work,+active: Jobs.Active<Jobs.phase(work)>,+wb: U32,+ob: U32,+line: Nat,+group: Nat,+unit: Nat,+tap: Nat,+inside: Jobs.Inside<work,line,group,unit>,+readers: NativeJobs.Readers) ->
  {(context_weight(Reduction.native_context(NativeJobs.initial_loop(Jobs.job_index(work,line,group,unit),Execution.work_plan(Execution.work(Jobs.context(Jobs.phase(work)),Jobs.chunk(work),wb,ob)),readers))) + (ob + U32.from_nat(tap)) : U32) ==
    (wb + U32.from_nat(Nat.add(Nat.mul(Nat.div(unit,2n),U32.to_nat(Kernel.phase_inner(Jobs.context(Jobs.phase(work))))),tap)) : U32) : U32}:
  Jobs.Work{+phase,+chunk} = work
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,chunk_group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr0} = layout
  Jobs.Inside{li,gi,+ui} = inside
  NativeJobs.Readers{tiles,weights,offsets} = readers
  +uw = plan_unit(work,wb,ob,line,group,unit)
  +inner = Kernel.phase_inner(Jobs.context(phase))
  +block = Nat.div(unit,2n)
  +scale = (uw / 2 * inner : U32)
  +start = (wb - ob : U32)
  +q = U32.from_nat(tap)
  +unit_equal = plan_unit_exact(work,active,wb,ob,line,group,unit,inside)
  +half_equal = Equal.trans(U32,(uw / 2 : U32),(U32.from_nat(unit) / 2 : U32),U32.from_nat(block),Equal.cong(U32,U32,value => (value / 2 : U32),uw,U32.from_nat(unit),unit_equal),unit_half_exact(phase,unit,ui))
  +scale_equal = Equal.cong(U32,U32,value => (value * inner : U32),(uw / 2 : U32),U32.from_nat(block),half_equal)
  +tail = Equal.trans(U32,(scale + (ob + q) : U32),((scale + ob) + q : U32),(ob + (scale + q) : U32),Equal.sym(U32,((scale + ob) + q : U32),(scale + (ob + q) : U32),Arithmetic.add_associative(scale,ob,q)),
    Equal.trans(U32,((scale + ob) + q : U32),((ob + scale) + q : U32),(ob + (scale + q) : U32),Equal.cong(U32,U32,value => (value + q : U32),(scale + ob : U32),(ob + scale : U32),Arithmetic.add_commutative(scale,ob)),Arithmetic.add_associative(ob,scale,q)))
  +restored = Equal.trans(U32,((start + scale) + (ob + q) : U32),(start + (scale + (ob + q)) : U32),(wb + (scale + q) : U32),Arithmetic.add_associative(start,scale,(ob + q : U32)),
    Equal.trans(U32,(start + (scale + (ob + q)) : U32),(start + (ob + (scale + q)) : U32),(wb + (scale + q) : U32),Equal.cong(U32,U32,value => (start + value : U32),(scale + (ob + q) : U32),(ob + (scale + q) : U32),tail),
      Equal.trans(U32,(start + (ob + (scale + q)) : U32),((start + ob) + (scale + q) : U32),(wb + (scale + q) : U32),Equal.sym(U32,((start + ob) + (scale + q) : U32),(start + (ob + (scale + q)) : U32),Arithmetic.add_associative(start,ob,(scale + q : U32))),Equal.cong(U32,U32,value => (value + (scale + q) : U32),(start + ob : U32),wb,Subtraction.add_subtract(wb,ob)))))
  Equal.trans(U32,((start + scale) + (ob + q) : U32),(wb + (scale + q) : U32),(wb + U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(inner)),tap)) : U32),restored,
    Equal.cong(U32,U32,value => (wb + value : U32),(scale + q : U32),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(inner)),tap)),
      Equal.trans(U32,(scale + q : U32),(U32.from_nat(block) * inner + q : U32),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(inner)),tap)),Equal.cong(U32,U32,value => (value + q : U32),scale,(U32.from_nat(block) * inner : U32),scale_equal),
        %Arithmetic.native_roundtrip(inner) : {(U32.from_nat(block) * _ + q : U32) == U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(inner)),tap)) : U32} Encoded.encoded_madd(block,U32.to_nat(inner),tap))))

def packed_weight_read(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> Tile.Tile8:
  +phase = Chunks.phase(Chunks.selection(reduction))
  +q = Weights.packed_query(phase,original,Nat.div(Chunks.unit(reduction),2n),Packing.phase_index(Jobs.context(phase),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction)))
  Weights.weight(Packed.read(q,Packed.complete(q,Packed.initial(q))))
def plan(+reduction: Chunks.Reduction) -> Kernel.Plan:
  +selected = Chunks.selection(reduction)
  +q = Chunks.query(selected)
  Execution.work_plan(Execution.work(Jobs.context(Chunks.phase(selected)),Chunks.chunk(q),Chunks.weight_base(q),Chunks.offset_base(q)))
def job_index(+reduction: Chunks.Reduction) -> U32:
  Jobs.job_index(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction),Chunks.local_group(reduction),Chunks.unit(reduction))
def tap(+reduction: Chunks.Reduction) -> Nat:
  Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))
def readers(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> NativeJobs.Readers:
  +packed = Weights.full_state(Chunks.phase(Chunks.selection(reduction)),original)
  NativeJobs.Readers{Fill.tree(Chunks.filled(reduction)),Packed.weight_tree(packed),Packed.offset_tree(packed)}
def initial(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> Reduction.Loop:
  NativeJobs.initial_loop(job_index(reduction),plan(reduction),readers(reduction,original))
def weight_read(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Packed.weight_tree(Weights.full_state(Chunks.phase(Chunks.selection(reduction)),original)),
    (context_weight(Reduction.native_context(initial(reduction,original))) + (Chunks.offset_base(Chunks.query(Chunks.selection(reduction))) + U32.from_nat(tap(reduction))) : U32))
def weight_read_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+bases: Bases<Chunks.selection(reduction)>,+original: Buffer.Model<F32>) ->
  {weight_read(reduction,original) == packed_weight_read(reduction,original) : Tile.Tile8}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,oc,tr,tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,+enabled,+ji,oi,ri,coli} = valid
  Bases{+wb_equal,ob_equal} = bases
  +phase = Chunks.phase(selected)
  +pq = Weights.packed_query(phase,original,Nat.div(unit,2n),tap(reduction))
  +full = Weights.full_state(phase,original)
  +complete = Packed.complete(pq,Packed.initial(pq))
  +tree = Packed.weight_tree(full)
  +address = (context_weight(Reduction.native_context(initial(reduction,original))) + (ob + U32.from_nat(tap(reduction))) : U32)
  +slot = U32.from_nat(Nat.add(Nat.mul(Nat.div(unit,2n),U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap(reduction)))
  +address_equal = Equal.trans(U32,address,(wb + slot : U32),Packed.weight_address(pq),weight_pointer_matches(Chunks.work(selected),Jobs.active_phase(phase,enabled),wb,ob,line,local_group,unit,tap(reduction),ji,readers(reduction,original)),Equal.cong(U32,U32,base => (base + slot : U32),wb,weight_base(phase),wb_equal))
  Equal.trans(Tile.Tile8,weight_read(reduction,original),Tree.read(Tile.Tile8,tree,Packed.weight_address(pq)),packed_weight_read(reduction,original),Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,tree),address,Packed.weight_address(pq),address_equal),
    Equal.cong(PackedWeights.Packed,Tile.Tile8,state => Tree.read(Tile.Tile8,Packed.weight_tree(state),Packed.weight_address(pq)),full,complete,Equal.sym(PackedWeights.Packed,complete,full,Weights.full_state_matches(phase,original,Nat.div(unit,2n),tap(reduction)))))

def offset_read(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> U32:
  Tree.read(U32,Packed.offset_tree(Weights.full_state(Chunks.phase(Chunks.selection(reduction)),original)),(Chunks.offset_base(Chunks.query(Chunks.selection(reduction))) + U32.from_nat(tap(reduction)) : U32))
def offset_read_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+bases: Bases<Chunks.selection(reduction)>,+original: Buffer.Model<F32>) ->
  {offset_read(reduction,original) == Offsets.value(PackedWeights.phase_table(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Kernel.tiling(Jobs.geometry(Chunks.phase(Chunks.selection(reduction)))),0),U32.from_nat(tap(reduction))) : U32}:
  Chunks.Reduction{+selected,line,local_group,+unit,+oc,+tr,+tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,+ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,+ji,+oi,+ri,+coli} = valid
  Jobs.Inside{li,gi,+ui} = ji
  Bases{wb_equal,+ob_equal} = bases
  +phase = Chunks.phase(selected)
  +block = Nat.div(unit,2n)
  +pq = Weights.packed_query(phase,original,block,tap(reduction))
  +full = Weights.full_state(phase,original)
  +complete = Packed.complete(pq,Packed.initial(pq))
  +tree = Packed.offset_tree(full)
  +block_inside = {%Equal.sym(Nat,Weights.block_count(phase),Ceiling.ceiling(Domain.channels(g),8n),Weights.block_count_exact(phase)) : {Nat.is_lt(block,_) == True{} : Bool} Weights.block_inside(phase,unit,ui) : {Nat.is_lt(block,Weights.block_count(phase)) == True{} : Bool}}
  +tap_inside = Weights.tap_index_inside(phase,oc,tr,tc,oi,ri,coli)
  +address_equal = Equal.cong(U32,U32,base => (base + U32.from_nat(tap(reduction)) : U32),ob,offset_base(phase),ob_equal)
  +wanted = Offsets.value(PackedWeights.phase_table(Jobs.context(phase),Kernel.tiling(g),0),U32.from_nat(tap(reduction)))
  Equal.trans(U32,offset_read(reduction,original),Tree.read(U32,tree,Packed.offset_address(pq)),wanted,
    Equal.cong(U32,U32,Tree.read(U32,tree),(ob + U32.from_nat(tap(reduction)) : U32),Packed.offset_address(pq),address_equal),
    Equal.trans(U32,Tree.read(U32,tree,Packed.offset_address(pq)),Weights.offset(Packed.read(pq,complete)),wanted,
      Equal.cong(PackedWeights.Packed,U32,state => Tree.read(U32,Packed.offset_tree(state),Packed.offset_address(pq)),full,complete,Equal.sym(PackedWeights.Packed,complete,full,Weights.full_state_matches(phase,original,block,tap(reduction)))),
      Weights.packed_offset_readback(phase,original,block,tap(reduction),block_inside,tap_inside)))

def input_read(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),(Lookup.context_source(Reduction.native_context(initial(reduction,original))) + offset_read(reduction,original) : U32))
def input_read_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+bases: Bases<Chunks.selection(reduction)>,+original: Buffer.Model<F32>) ->
  {input_read(reduction,original) == Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))) : Tile.Tile8}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,oc,tr,tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,+wb,+ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +phase = Jobs.context(Chunks.phase(selected))
  +job_readers = readers(reduction,original)
  +source = Lookup.context_source(Reduction.native_context(initial(reduction,original)))
  +wanted_offset = Offsets.value(PackedWeights.phase_table(phase,Kernel.tiling(g),0),U32.from_nat(tap(reduction)))
  +at_base = Offsets.value(PackedWeights.phase_table(phase,Lookup.chunk_layout(Chunks.chunk(q)),ob),U32.from_nat(tap(reduction)))
  +table_same = {{==} : {wanted_offset == at_base : U32}}
  +address_equal = Equal.trans(U32,(source + offset_read(reduction,original) : U32),(source + wanted_offset : U32),Lookup.read_address(Chunks.sample(reduction)),Equal.cong(U32,U32,value => (source + value : U32),offset_read(reduction,original),wanted_offset,offset_read_matches(reduction,valid,bases,original)),
    Equal.trans(U32,(source + wanted_offset : U32),(source + at_base : U32),Lookup.read_address(Chunks.sample(reduction)),Equal.cong(U32,U32,value => (source + value : U32),wanted_offset,at_base,table_same),Lookup.source_table_matches(phase,Chunks.chunk(q),wb,ob,job_index(reduction),U32.from_nat(tap(reduction)),job_readers)))
  Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction))),(source + offset_read(reduction,original) : U32),Lookup.read_address(Chunks.sample(reduction)),address_equal)

# The natural address tape observes those same MAC operands at every ordinal.
def initial_index(+reduction: Chunks.Reduction,+original: Buffer.Model<F32>) ->
  {Natural.loop_index(initial(reduction,original)) == Chunks.offset_base(Chunks.query(Chunks.selection(reduction))) : U32}:
  Chunks.Reduction{+selected,line,group,unit,out,row,column} = reduction
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def natural_address(+base: U32,+index: Nat) -> {U32.from_nat(Nat.add(U32.to_nat(base),index)) == (base + U32.from_nat(index) : U32) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(base),index)),(U32.from_nat(U32.to_nat(base)) + U32.from_nat(index) : U32),(base + U32.from_nat(index) : U32),
    Equal.sym(U32,(U32.from_nat(U32.to_nat(base)) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(base),index)),WordArithmetic.native_add(U32.to_nat(base),index)),
    Equal.cong(U32,U32,value => (value + U32.from_nat(index) : U32),U32.from_nat(U32.to_nat(base)),base,Arithmetic.native_roundtrip(base)))
