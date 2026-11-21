# A phase job reads the weights of its unit and the tiles of its grid: step t
# of the job is weight slot unit * inner + t of the packed phase and entry t
# of the grid.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_input_gradient.bend as Kernel
import ../kernel_gemm.bend as Gemm
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_phase_index.bend as Index
import ./input_gradient_tiled_pack_values.bend as Packed
import ./input_gradient_tiled_packing_bounds.bend as PackingBounds
import ./input_gradient_tiled_weight_reads.bend as Weights
import ./input_gradient_tiled_entry_reads.bend as Lookup
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./kernel_gemm_spec.bend as GemmSpec
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer

# The packed weights of a phase start at the prefix sum of the phases before.
def weight_base(+phase: Jobs.Phase) -> U32:
  PackingBounds.prefix_word(Index.packing_bounds(phase),Index.ordinal(phase))
type Bases<-selection: Chunks.Selection> is Data:
  Bases{weights: {Chunks.weight_base(Chunks.query(selection)) == weight_base(Chunks.phase(selection)) : U32}}

# ---- The taps of a job ----

def job_cursor(+job: Gemm.Job<Kernel.Cursor()>) -> Kernel.Cursor():
  Gemm.Job{first,depth,cursor,store} = job
  cursor

def plan_weight(+plan: Kernel.Plan,+index: U32) -> U32:
  Kernel.Plan{+units,groups,+inner,source_base,row_step,group_step,target_base,target_row,step,plane,channels,columns,+weight_base,
    tap_columns,tap_rows,row_jump,output_jump} = plan
  +index = index
  (weight_base + index % units * inner : U32)

def taps_key(+grid: GemmSpec.Grid,+weight: U32) -> Gemm.Both<Nat,GemmSpec.Grid>:
  Gemm.Both{U32.to_nat(weight),grid}

def keyed(count: Nat,+key: Gemm.Both<Nat,GemmSpec.Grid>) -> List<&2,ReductionLaws.AddressPair>:
  Ordered.map_keys(~Nat,~ReductionLaws.AddressPair,~Gemm.Both<Nat,GemmSpec.Grid>,~GemmSpec.both_key(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read),
    Ranges.range(count,0n),key)

def advanced(+first: U32,+second: Gemm.Entries) ->
  {Kernel.entries(Gemm.Both{first,second}) == Gemm.Step{Pair.fst(U32,U32,Gemm.direct(first)),Pair.fst(U32,Gemm.Entries,Gemm.entry(second)),
    Gemm.Both{Pair.snd(U32,U32,Gemm.direct(first)),Pair.snd(U32,Gemm.Entries,Gemm.entry(second))}} : Gemm.Step<Kernel.Cursor()>}:
  Gemm.Entries{+address,+column,+row,+columns,+rows,+row_jump,+output_jump} = second
  {==}

# The first entry of a grid from position 0 is its base.
def grid_start(+source: U32,+row_step: U32,+columns: U32,+rows: U32,+output_jump: U32,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == GemmSpec.grid_entry(GemmSpec.Grid{source,U32.to_nat(row_step),columns,rows,output_jump,0n},0n) : Gemm.Entries}:
  +grid = {GemmSpec.Grid{source,U32.to_nat(row_step),columns,rows,output_jump,0n} : GemmSpec.Grid}
  %Equal.sym(Nat & Nat,Nat.divmod(0n,U32.to_nat(columns)),(0n,0n),Division.small(0n,U32.to_nat(columns),columns_positive)) :
    {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == GemmSpec.grid_cell(grid,_) : Gemm.Entries}
  %Equal.sym(Nat & Nat,Nat.divmod(0n,U32.to_nat(rows)),(0n,0n),Division.small(0n,U32.to_nat(rows),rows_positive)) :
    {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == GemmSpec.grid_placed(grid,_,0n) : Gemm.Entries}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(row_step)),row_step,Arithmetic.native_roundtrip(row_step)) :

    {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == Gemm.Entries{(source + 0 : U32),0,0,columns,rows,(_ - columns + 1 : U32),output_jump} : Gemm.Entries}
  %Equal.sym(U32,(source + 0 : U32),source,Arithmetic.add_zero(source)) :

    {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == Gemm.Entries{_,0,0,columns,rows,(row_step - columns + 1 : U32),output_jump} : Gemm.Entries}
  %Equal.sym(U32,((row_step - columns : U32) + 1 : U32),(row_step - (columns - 1 : U32) : U32),Arithmetic.subtract_difference(row_step,columns,1)) :

    {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} == Gemm.Entries{source,0,0,columns,rows,_,output_jump} : Gemm.Entries}
  {==}

# A direct weight reader and a grid reader step together: step t reads
# weight + t and entry t of the grid.
def grid_taps(+source: U32,+row_step: U32,+columns: U32,+rows: U32,+output_jump: U32,+weight: U32,+count: Nat,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  {GemmSpec.taps(~Kernel.Cursor(),~Kernel.entries,count,Gemm.Both{weight,Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump}}) ==
    keyed(count,taps_key(GemmSpec.Grid{source,U32.to_nat(row_step),columns,rows,output_jump,0n},weight)) : List<&2,ReductionLaws.AddressPair>}:
  +entries = {Gemm.Entries{source,0,0,columns,rows,(row_step - (columns - 1) : U32),output_jump} : Gemm.Entries}
  +grid = {GemmSpec.Grid{source,U32.to_nat(row_step),columns,rows,output_jump,0n} : GemmSpec.Grid}
  +base = U32.to_nat(weight)
  +tiles = Ordered.map_keys(~Nat,~U32,~GemmSpec.Grid,~GemmSpec.grid_read,Ranges.range(count,0n),grid)
  +weights = Ordered.map_keys(~Nat,~U32,~Nat,~GemmSpec.offset_word,Ranges.range(count,0n),base)
  +wanted = keyed(count,taps_key(grid,weight))
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,GemmSpec.taps(~Kernel.Cursor(),~Kernel.entries,count,Gemm.Both{weight,entries}),
      GemmSpec.paired(GemmSpec.reads(~U32,~Gemm.direct,count,weight),GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,entries)),
      GemmSpec.both_taps(~U32,~Gemm.Entries,~Gemm.direct,~Gemm.entry,~Kernel.entries,~advanced,count,weight,entries)) :
    {_ == wanted : List<&2,ReductionLaws.AddressPair>}
  %Equal.sym(Gemm.Entries,entries,GemmSpec.grid_entry(grid,0n),grid_start(source,row_step,columns,rows,output_jump,columns_positive,rows_positive)) :
    {GemmSpec.paired(GemmSpec.reads(~U32,~Gemm.direct,count,weight),GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,_)) == wanted : List<&2,ReductionLaws.AddressPair>}
  %Equal.sym(List<&2,U32>,GemmSpec.reads(~Gemm.Entries,~Gemm.entry,count,GemmSpec.grid_entry(grid,0n)),tiles,
      GemmSpec.entry_reads(count,grid,0n,columns_positive,rows_positive)) :
    {GemmSpec.paired(GemmSpec.reads(~U32,~Gemm.direct,count,weight),_) == wanted : List<&2,ReductionLaws.AddressPair>}
  +start = Equal.trans(U32,weight,U32.from_nat(base),U32.from_nat(Nat.add(base,0n)),
    Equal.sym(U32,U32.from_nat(base),weight,Arithmetic.native_roundtrip(weight)),
    Equal.cong(Nat,U32,U32.from_nat,base,Nat.add(base,0n),Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base))))
  +direct = Equal.trans(List<&2,U32>,GemmSpec.reads(~U32,~Gemm.direct,count,weight),GemmSpec.reads(~U32,~Gemm.direct,count,U32.from_nat(Nat.add(base,0n))),weights,
    Equal.cong(U32,List<&2,U32>,first => GemmSpec.reads(~U32,~Gemm.direct,count,first),weight,U32.from_nat(Nat.add(base,0n)),start),
    GemmSpec.direct_reads(count,base,0n))
  %Equal.sym(List<&2,U32>,GemmSpec.reads(~U32,~Gemm.direct,count,weight),weights,direct) :
    {GemmSpec.paired(_,tiles) == wanted : List<&2,ReductionLaws.AddressPair>}
  GemmSpec.paired_keys(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read,Ranges.range(count,0n),base,grid)

# The taps of job index of a phase chunk are the weights from its unit's slot
# and entries of its grid.
def phase_job(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32) -> Gemm.Job<Kernel.Cursor()>:
  Kernel.job(Execution.work_plan(Execution.work(phase,chunk,wb)),index)
def phase_key(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32) -> Gemm.Both<Nat,GemmSpec.Grid>:
  +plan = Execution.work_plan(Execution.work(phase,chunk,wb))
  taps_key(Lookup.job_grid(plan,index),plan_weight(plan,index))
def phase_taps(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(Packing.phase_columns(phase))) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(Packing.phase_rows(phase))) == True{} : Bool}) ->
  {GemmSpec.taps(~Kernel.Cursor(),~Kernel.entries,U32.to_nat(Kernel.phase_inner(phase)),job_cursor(phase_job(phase,chunk,wb,index))) ==
    keyed(U32.to_nat(Kernel.phase_inner(phase)),phase_key(phase,chunk,wb,index)) : List<&2,ReductionLaws.AddressPair>}:
  Kernel.Phase{Kernel.Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Kernel.Axis{+yf,+yc,yt,+yr,+ys},Kernel.Axis{+xf,+xc,xt,+xr,+xs},+inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},+first,count,+target},+chunk_row,+rows,+low,+chunk_group} = chunk
  Kernel.Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+cg,cr} = layout
  +units = ((c + 7) / 8 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(chunk_group,phase_groups),U32.min(cg,(phase_groups - chunk_group : U32)),0)
  +block = (outputs * shifts + 1 : U32)
  +step = (cg * block : U32)
  +start = Bool.pick(U32,U32.is_lt(yf,chunk_row),((chunk_row - yf + s - 1) / s : U32),0)
  +source = ((start + ys + k - low) * step + xs + k - column_low : U32)
  +base = (source + index / units / span * step + index / units % span * block : U32)
  +jump = (shifts - (yr - 1) * step - (xr - 1) : U32)
  grid_taps(base,step,xr,yr,jump,(wb + index % units * inner : U32),U32.to_nat(inner),columns_positive,rows_positive)

# ---- The steps of a job ----

# Step t of a grid tap list reads entry (output, row, column) when t is the
# row-major index of that entry.
def entry_step(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+index: U32,+output: Nat,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(phase))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(phase))) == True{} : Bool}) ->
  {GemmSpec.grid_read(Lookup.job_grid(Execution.work_plan(Execution.work(phase,chunk,wb)),index),Packing.phase_index(phase,output,row,column)) ==
    Lookup.read_address(Lookup.phase_read(phase,chunk,wb,index,output,row,column)) : U32}:
  Kernel.Phase{Kernel.Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Kernel.Axis{+yf,+yc,yt,+yr,+ys},Kernel.Axis{+xf,+xc,xt,+xr,+xs},+inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},+first,count,+target},+chunk_row,+rows,+low,+chunk_group} = chunk
  Kernel.Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+cg,cr} = layout
  +units = ((c + 7) / 8 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(chunk_group,phase_groups),U32.min(cg,(phase_groups - chunk_group : U32)),0)
  +block = (outputs * shifts + 1 : U32)
  +step = (cg * block : U32)
  +start = Bool.pick(U32,U32.is_lt(yf,chunk_row),((chunk_row - yf + s - 1) / s : U32),0)
  +source = ((start + ys + k - low) * step + xs + k - column_low : U32)
  +base = (source + index / units / span * step + index / units % span * block : U32)
  +jump = (shifts - (yr - 1) * step - (xr - 1) : U32)
  +columns_positive = Order.positive_bound(column,U32.to_nat(xr),column_inside)
  +rows_positive = Order.positive_bound(row,U32.to_nat(yr),row_inside)
  Equal.trans(U32,GemmSpec.grid_read(GemmSpec.Grid{base,U32.to_nat(step),xr,yr,jump,0n},RowMajor.index(output,U32.to_nat(yr),U32.to_nat(xr),row,column)),
    GemmSpec.grid_at_word(GemmSpec.Grid{base,U32.to_nat(step),xr,yr,jump,0n},output,row,column),
    Lookup.read_address(Lookup.phase_read(phase,chunk,wb,index,output,row,column)),
    GemmSpec.grid_read_index(base,U32.to_nat(step),xr,yr,jump,output,row,column,row_inside,column_inside),
    Lookup.entry_word(phase,chunk,wb,index,output,row,column,columns_positive,rows_positive))

# Step t of the weights reads weight + t.
def weight_step(+weight: U32,+step: Nat) -> {GemmSpec.offset_word(U32.to_nat(weight),step) == (weight + U32.from_nat(step) : U32) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(weight),step)),(U32.from_nat(U32.to_nat(weight)) + U32.from_nat(step) : U32),(weight + U32.from_nat(step) : U32),
    GemmSpec.encoded_sum(U32.to_nat(weight),step),
    Equal.cong(U32,U32,value => (value + U32.from_nat(step) : U32),U32.from_nat(U32.to_nat(weight)),weight,Arithmetic.native_roundtrip(weight)))

# ---- The job of a reduction ----

def plan(+reduction: Chunks.Reduction) -> Kernel.Plan:
  +selected = Chunks.selection(reduction)
  +q = Chunks.query(selected)
  Execution.work_plan(Execution.work(Jobs.context(Chunks.phase(selected)),Chunks.chunk(q),Chunks.weight_base(q)))
def job_index(+reduction: Chunks.Reduction) -> U32:
  Jobs.job_index(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction),Chunks.local_group(reduction),Chunks.unit(reduction))
def job(+reduction: Chunks.Reduction) -> Gemm.Job<Kernel.Cursor()>:
  Kernel.job(plan(reduction),job_index(reduction))
def key(+reduction: Chunks.Reduction) -> Gemm.Both<Nat,GemmSpec.Grid>:
  +selected = Chunks.selection(reduction)
  phase_key(Jobs.context(Chunks.phase(selected)),Chunks.chunk(Chunks.query(selected)),Chunks.weight_base(Chunks.query(selected)),job_index(reduction))
def count(+reduction: Chunks.Reduction) -> Nat:
  U32.to_nat(Kernel.phase_inner(Jobs.context(Chunks.phase(Chunks.selection(reduction)))))
def tap(+reduction: Chunks.Reduction) -> Nat:
  Packing.phase_index(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))

# The weight slot a step reads is its slot in the packed phase.
def weight_slot(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+bases: Bases<Chunks.selection(reduction)>,+original: Buffer.Model<Tile.Tile8>,+step: Nat) ->
  {(plan_weight(plan(reduction),job_index(reduction)) + U32.from_nat(step) : U32) ==
    Packed.address(Weights.packed_query(Chunks.phase(Chunks.selection(reduction)),original,Chunks.unit(reduction),step)) : U32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,oc,tr,tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,+wb} = q
  GeometryChunks.Grid{+call,e,leaf_count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,+enabled,+ji,oi,ri,coli} = valid
  Bases{+wb_equal} = bases
  +work = Chunks.work(selected)
  +inner = Kernel.phase_inner(Jobs.context(Chunks.phase(selected)))
  +native_unit = (job_index(reduction) % Jobs.native_units(work) : U32)
  +t = U32.from_nat(step)
  +unit_equal = Encoded.encoding(native_unit,unit,Jobs.unit_decode(work,Jobs.active_phase(Chunks.phase(selected),enabled),line,local_group,unit,ji))
  +slot = Nat.add(Nat.mul(unit,U32.to_nat(inner)),step)
  +scaled = Equal.trans(U32,((wb + native_unit * inner : U32) + t : U32),(wb + (native_unit * inner + t : U32) : U32),(wb + U32.from_nat(slot) : U32),
    Arithmetic.add_associative(wb,(native_unit * inner : U32),t),
    Equal.cong(U32,U32,value => (wb + value : U32),(native_unit * inner + t : U32),U32.from_nat(slot),
      Equal.trans(U32,(native_unit * inner + t : U32),(U32.from_nat(unit) * inner + t : U32),U32.from_nat(slot),
        Equal.cong(U32,U32,value => (value * inner + t : U32),native_unit,U32.from_nat(unit),unit_equal),
        %Arithmetic.native_roundtrip(inner) : {(U32.from_nat(unit) * _ + t : U32) == U32.from_nat(slot) : U32} Encoded.encoded_madd(unit,U32.to_nat(inner),step))))
  Equal.trans(U32,((wb + native_unit * inner : U32) + t : U32),(wb + U32.from_nat(slot) : U32),(weight_base(Chunks.phase(selected)) + U32.from_nat(slot) : U32),scaled,
    Equal.cong(U32,U32,value => (value + U32.from_nat(slot) : U32),wb,weight_base(Chunks.phase(selected)),wb_equal))
