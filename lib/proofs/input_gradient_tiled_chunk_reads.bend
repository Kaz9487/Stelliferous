# Actual valid chunk jobs read the slots produced by the allocated tile fill.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_refinement.bend as Filling
import ./storage_tree_model.bend as Tree
import ./input_gradient_tiled_chunk_geometry.bend as Chunks
import ./input_gradient_tiled_chunk_allocation.bend as Allocation
import ./input_gradient_tiled_chunks.bend as Partition
import ./input_gradient_tiled_bands.bend as PureBands
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_geometry.bend as PhaseGeometry
import ./input_gradient_tiled_columns.bend as Columns
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_band_geometry.bend as BandSamples
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order

type Query is Data:
  Query{grid: Chunks.Grid,first: U32,target: U32,plane: U32,source: Execution.Source,row: Nat,group: Nat,weight_base: U32,offset_base: U32}
def grid(query: Query) -> Chunks.Grid:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  grid
def first(query: Query) -> U32:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  first
def target(query: Query) -> U32:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  target
def plane(query: Query) -> U32:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  plane
def source(query: Query) -> Execution.Source:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  source
def row(query: Query) -> Nat:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  row
def group(query: Query) -> Nat:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  group
def weight_base(query: Query) -> U32:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  wb
def offset_base(query: Query) -> U32:
  Query{grid,first,target,plane,source,row,group,wb,ob} = query
  ob
def chunk(+query: Query) -> Kernel.Chunk:
  Chunks.selected_chunk(grid(query),first(query),target(query),plane(query),row(query),group(query))
def filling(+query: Query) -> Kernel.Filling:
  Chunks.selected_filling(grid(query),first(query),row(query),group(query),source(query))
def geometry(+query: Query) -> Kernel.Geometry:
  Chunks.grid_geometry(grid(query))

type Selection is Data:
  Selection{query: Query,y: Nat,x: Nat,
    y_inside: {Nat.is_lt(y,U32.to_nat(U32.min(Domain.stride_word(geometry(query)),Domain.height_word(geometry(query))))) == True{} : Bool},
    x_inside: {Nat.is_lt(x,U32.to_nat(U32.min(Domain.stride_word(geometry(query)),Domain.width_word(geometry(query))))) == True{} : Bool}}
def query(selection: Selection) -> Query:
  Selection{query,y,x,yi,xi} = selection
  query
def y(selection: Selection) -> Nat:
  Selection{query,y,x,yi,xi} = selection
  y
def x(selection: Selection) -> Nat:
  Selection{query,y,x,yi,xi} = selection
  x
def y_inside(+selection: Selection) -> {Nat.is_lt(y(selection),U32.to_nat(U32.min(Domain.stride_word(geometry(query(selection))),Domain.height_word(geometry(query(selection)))))) == True{} : Bool}:
  Selection{query,y,x,yi,xi} = selection
  yi
def x_inside(+selection: Selection) -> {Nat.is_lt(x(selection),U32.to_nat(U32.min(Domain.stride_word(geometry(query(selection))),Domain.width_word(geometry(query(selection)))))) == True{} : Bool}:
  Selection{query,y,x,yi,xi} = selection
  xi
def phase(+selection: Selection) -> Jobs.Phase:
  +grid = grid(query(selection))
  Jobs.Phase{Chunks.grid_call(grid),Chunks.grid_envelope(grid),
    y(selection),x(selection),y_inside(selection),x_inside(selection)}
def work(+selection: Selection) -> Jobs.Work:
  Jobs.Work{phase(selection),chunk(query(selection))}

type Reduction is Data:
  Reduction{selection: Selection,line: Nat,group: Nat,unit: Nat,output: Nat,tap_row: Nat,tap_column: Nat}
def selection(reduction: Reduction) -> Selection:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  selection
def line(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  line
def local_group(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  group
def unit(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  unit
def output(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  output
def tap_row(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  row
def tap_column(reduction: Reduction) -> Nat:
  Reduction{selection,line,group,unit,output,row,column} = reduction
  column
def sample(+reduction: Reduction) -> Lookup.Read:
  +selected = selection(reduction)
  +q = query(selected)
  Reads.read(work(selected),weight_base(q),offset_base(q),line(reduction),local_group(reduction),unit(reduction),U32.from_nat(Packing.phase_index(Jobs.context(phase(selected)),output(reduction),tap_row(reduction),tap_column(reduction))))
def position(+reduction: Reduction) -> Nat:
  +s = sample(reduction)
  Nat.add(Nat.mul(U32.to_nat(Lookup.read_row_total(s)),U32.to_nat(Fill.span(filling(query(selection(reduction)))))),U32.to_nat(Lookup.read_group(s)))
def slot(+reduction: Reduction) -> Fill.Query:
  +q = query(selection(reduction))
  +s = sample(reduction)
  Allocation.slot_query(grid(q),first(q),row(q),group(q),source(q),position(reduction),U32.to_nat(Lookup.read_output(s)),U32.to_nat(Lookup.read_shift_total(s)))

type Valid<-reduction: Reduction> is Data:
  Valid{chunk_inside: Partition.Inside<Chunks.query(grid(query(selection(reduction))),row(query(selection(reduction))),group(query(selection(reduction))))>,
    contained: {Nat.is_le(Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction)))))),Domain.height(geometry(query(selection(reduction))))) == True{} : Bool},
    fits: {Kernel.fits(geometry(query(selection(reduction))),Machine.capacity(Chunks.grid_limits(grid(query(selection(reduction)))))) == True{} : Bool},
    enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(phase(selection(reduction)))),0) == False{} : Bool},
    job_inside: Jobs.Inside<work(selection(reduction)),line(reduction),local_group(reduction),unit(reduction)>,
    output_inside: {Nat.is_lt(output(reduction),Domain.outputs(geometry(query(selection(reduction))))) == True{} : Bool},
    row_inside: {Nat.is_lt(tap_row(reduction),U32.to_nat(Packing.phase_rows(Jobs.context(phase(selection(reduction)))))) == True{} : Bool},
    column_inside: {Nat.is_lt(tap_column(reduction),U32.to_nat(Packing.phase_columns(Jobs.context(phase(selection(reduction)))))) == True{} : Bool}}
def leaf_room(+reduction: Reduction,valid: Valid<reduction>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction)))))),Domain.height(geometry(query(selection(reduction))))) == True{} : Bool}:
  Valid{chunk_inside,contained,fits,enabled,job_inside,output_inside,row_inside,column_inside} = valid
  contained
def job_inside(+reduction: Reduction,valid: Valid<reduction>) -> Jobs.Inside<work(selection(reduction)),line(reduction),local_group(reduction),unit(reduction)>:
  Valid{chunk_inside,contained,fits,enabled,job_inside,output_inside,row_inside,column_inside} = valid
  job_inside

def layout_matches(+selected: Selection) ->
  {Lookup.chunk_layout(Jobs.chunk(work(selected))) == Kernel.tiling(Jobs.geometry(phase(selected))) : Kernel.Tiling}:
  Selection{Query{+grid,first,target,plane,source,row,group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{call,e,count,contained,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  {==}
def low_matches(+selected: Selection) ->
  {Reads.chunk_low(Jobs.chunk(work(selected))) == (Kernel.band_low(Jobs.geometry(phase(selected)),Jobs.chunk_row(Jobs.chunk(work(selected))),(Jobs.chunk_row(Jobs.chunk(work(selected))) + Jobs.chunk_rows(Jobs.chunk(work(selected))) : U32)) : U32) : U32}:
  Selection{Query{+grid,first,target,plane,source,row,group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{call,e,count,contained,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  {==}
def contained(+selected: Selection,
  +inside: Partition.Inside<Chunks.query(grid(query(selected)),row(query(selected)),group(query(selected)))>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first(query(selected))),U32.to_nat(Chunks.grid_count(grid(query(selected))))),Domain.height(geometry(query(selected)))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Jobs.chunk_row(Jobs.chunk(work(selected)))),U32.to_nat(Jobs.chunk_rows(Jobs.chunk(work(selected))))),Domain.height(Jobs.geometry(phase(selected)))) == True{} : Bool}:
  Selection{Query{+grid,+first,target,plane,source,+row,+group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{call,e,count,contained,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Chunks.selected_band_contains(grid,first,row,group,inside,room)
def filling_span_matches(+selected: Selection) -> {Fill.span(filling(query(selected))) == Reads.filled_span(work(selected)) : U32}:
  Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  {==}
def slot_rows_matches(+reduction: Reduction) ->
  {Fill.rows(slot(reduction)) == U32.to_nat(Kernel.band_rows(Jobs.geometry(phase(selection(reduction))),Jobs.chunk_row(Jobs.chunk(work(selection(reduction)))),Jobs.chunk_rows(Jobs.chunk(work(selection(reduction)))))) : Nat}:
  Reduction{Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tap_row,tap_column} = reduction
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  {==}
def filling_outputs_matches(+selected: Selection) -> {U32.to_nat(Fill.outputs(filling(query(selected)))) == Domain.outputs(Jobs.geometry(phase(selected))) : Nat}:
  Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  {==}
def filling_shifts_matches(+selected: Selection) -> {Fill.shifts(filling(query(selected))) == Columns.layout_shifts(Kernel.tiling(Jobs.geometry(phase(selected)))) : U32}:
  Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi} = selected
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  {==}

def row_inside_slot(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(U32.to_nat(Lookup.read_row_total(sample(reduction))),Fill.rows(slot(reduction))) == True{} : Bool}:
  Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,+wb,+ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{+ci,+room,fits,+enabled,+ji,+oi,+ri,+coli} = valid
  %Equal.sym(Nat,Fill.rows(slot(reduction)),U32.to_nat(Kernel.band_rows(g,Jobs.chunk_row(chunk(q)),Jobs.chunk_rows(chunk(q)))),slot_rows_matches(reduction)) : {Nat.is_lt(U32.to_nat(Lookup.read_row_total(sample(reduction))),_) == True{} : Bool}
  Reads.read_row_inside(work(selected),enabled,wb,ob,line,local_group,unit,output,tr,tc,ji,oi,ri,coli,contained(selected,ci,room),low_matches(selected))
def group_inside_slot(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(U32.to_nat(Lookup.read_group(sample(reduction))),U32.to_nat(Fill.span(filling(query(selection(reduction)))))) == True{} : Bool}:
  Reduction{+selected,+line,+local_group,+unit,output,tr,tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{grid,first,target,plane,source,row,group,+wb,+ob} = q
  Valid{ci,room,fits,+enabled,+ji,oi,ri,coli} = valid
  +relative = U32.from_nat(Packing.phase_index(Jobs.context(phase(selected)),output,tr,tc))
  %Equal.sym(U32,Fill.span(filling(q)),Reads.filled_span(work(selected)),filling_span_matches(selected)) : {Nat.is_lt(U32.to_nat(Lookup.read_group(sample(reduction))),U32.to_nat(_)) == True{} : Bool}
  Reads.read_group_inside(work(selected),Jobs.active_phase(phase(selected),enabled),wb,ob,line,local_group,unit,relative,ji,layout_matches(selected))
def slot_count_matches(+reduction: Reduction) -> {Fill.count(slot(reduction)) == Nat.mul(Fill.rows(slot(reduction)),U32.to_nat(Fill.span(filling(query(selection(reduction)))))) : Nat}:
  Reduction{Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tap_row,tap_column} = reduction
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  {==}
def position_inside_slot(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(position(reduction),Fill.count(slot(reduction))) == True{} : Bool}:
  %Equal.sym(Nat,Fill.count(slot(reduction)),Nat.mul(Fill.rows(slot(reduction)),U32.to_nat(Fill.span(filling(query(selection(reduction)))))),slot_count_matches(reduction)) : {Nat.is_lt(position(reduction),_) == True{} : Bool}
  RowMajor.radix(U32.to_nat(Lookup.read_row_total(sample(reduction))),Fill.rows(slot(reduction)),U32.to_nat(Lookup.read_group(sample(reduction))),U32.to_nat(Fill.span(filling(query(selection(reduction))))),row_inside_slot(reduction,valid),group_inside_slot(reduction,valid))
def output_inside_slot(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(U32.to_nat(Lookup.read_output(sample(reduction))),U32.to_nat(Fill.outputs(filling(query(selection(reduction)))))) == True{} : Bool}:
  Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,+wb,+ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{ci,room,fits,+enabled,ji,+oi,+ri,+coli} = valid
  %Equal.sym(Nat,U32.to_nat(Fill.outputs(filling(q))),Domain.outputs(g),filling_outputs_matches(selected)) : {Nat.is_lt(U32.to_nat(Lookup.read_output(sample(reduction))),_) == True{} : Bool}
  Reads.read_output_inside(work(selected),enabled,wb,ob,line,local_group,unit,output,tr,tc,oi,ri,coli)
def shift_inside_slot(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(U32.to_nat(Lookup.read_shift_total(sample(reduction))),U32.to_nat(Fill.shifts(filling(query(selection(reduction)))))) == True{} : Bool}:
  Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,+wb,+ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{ci,room,fits,+enabled,ji,+oi,+ri,+coli} = valid
  %Equal.sym(U32,Fill.shifts(filling(q)),Columns.layout_shifts(Kernel.tiling(g)),filling_shifts_matches(selected)) : {Nat.is_lt(U32.to_nat(Lookup.read_shift_total(sample(reduction))),U32.to_nat(_)) == True{} : Bool}
  Reads.read_shift_inside(work(selected),enabled,wb,ob,line,local_group,unit,output,tr,tc,oi,ri,coli,layout_matches(selected))

def slot_frame(+reduction: Reduction,+valid: Valid<reduction>) -> Fill.Frame<slot(reduction),Fill.initial(slot(reduction))>:
  Reduction{+selected,line,local_group,unit,output,tr,tc} = reduction
  Selection{+q,y,+x,yi,+xi} = selected
  Query{+grid,+first,target,plane,+source,+row,+group,wb,ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{+ci,+room,+fits,+enabled,ji,oi,ri,coli} = valid
  Allocation.slot_frame(grid,first,row,group,source,position(reduction),U32.to_nat(Lookup.read_output(sample(reduction))),U32.to_nat(Lookup.read_shift_total(sample(reduction))),ci,room,fits,
    position_inside_slot(reduction,valid),output_inside_slot(reduction,valid),shift_inside_slot(reduction,valid))

def lookup_fields(+reduction: Reduction,+valid: Valid<reduction>) -> Lookup.Fields<Lookup.query_for_read(slot(reduction),sample(reduction))>:
  Reduction{Selection{Query{+grid,first,target,plane,+source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tap_row,tap_column} = reduction
  Chunks.Grid{+call,e,count,contained,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  Lookup.Fields{{==},{==},{==},group_inside_slot(reduction,valid)}
def lookup_layout(+reduction: Reduction) -> Lookup.Layout<slot(reduction),sample(reduction)>:
  Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,+first,+target,+plane,+source,+row,+group,+wb,+ob} = q
  Lookup.slot_layout(grid,first,target,plane,row,group,source,position(reduction),U32.to_nat(Lookup.read_output(sample(reduction))),U32.to_nat(Lookup.read_shift_total(sample(reduction))),Jobs.context(phase(selected)),wb,ob,
    Jobs.job_index(work(selected),line,local_group,unit),U32.from_nat(Packing.phase_index(Jobs.context(phase(selected)),output,tr,tc)))
def shape(+query: Query) -> Execution.ChunkShape:
  +grid = grid(query)
  Execution.chunk_shape(Chunks.native_producer(grid,row(query),group(query)),
    Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first(query),Chunks.grid_count(grid),target(query),plane(query)},source(query))
def filled(+reduction: Reduction) -> Filling.Owners:
  +q = query(selection(reduction))
  Execution.fill_chunk(shape(q),source(q))
def filled_matches(+reduction: Reduction,+valid: Valid<reduction>) ->
  {filled(reduction) == Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(slot(reduction)),0,Fill.filling(slot(reduction)),Fill.initial(slot(reduction))) : Filling.Owners}:
  Reduction{+selected,line,local_group,unit,output,tr,tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,+first,+target,+plane,+source,+row,+group,wb,ob} = q
  +s = sample(reduction)
  Allocation.fill_matches_slot(grid,first,target,plane,row,group,source,position(reduction),U32.to_nat(Lookup.read_output(s)),U32.to_nat(Lookup.read_shift_total(s)),Fill.frame_facts(slot(reduction),Fill.initial(slot(reduction)),slot_frame(reduction,valid)))
def readback(+reduction: Reduction,+valid: Valid<reduction>) ->
  {Tree.read(Tile.Tile8,Fill.tree(filled(reduction)),Lookup.read_address(sample(reduction))) == Fill.wanted(slot(reduction)) : Tile.Tile8}:
  +fq = slot(reduction)
  +s = sample(reduction)
  +lq = Lookup.query_for_read(fq,s)
  +model = Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(fq),0,Fill.filling(fq),Fill.initial(fq))
  +address_same = Equal.trans(U32,Lookup.read_address(s),Lookup.read_canonical(s),Lookup.address(lq),Lookup.read_matches_canonical(s),Lookup.canonical_matches_query(fq,s,lookup_layout(reduction)))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(filled(reduction)),Lookup.read_address(s)),Tree.read(Tile.Tile8,Fill.tree(model),Lookup.read_address(s)),Fill.wanted(fq),
    Equal.cong(Filling.Owners,Tile.Tile8,owners => Tree.read(Tile.Tile8,Fill.tree(owners),Lookup.read_address(s)),filled(reduction),model,filled_matches(reduction,valid)),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(model),Lookup.read_address(s)),Tree.read(Tile.Tile8,Fill.tree(model),Lookup.address(lq)),Fill.wanted(fq),
      Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,Fill.tree(model)),Lookup.read_address(s),Lookup.address(lq),address_same),Lookup.readback(lq,lookup_fields(reduction,valid),Fill.initial(fq),slot_frame(reduction,valid))))

def row_in_leaf_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(Jobs.chunk_row(chunk(query(selection(reduction))))),Jobs.physical_row(work(selection(reduction)),line(reduction))) == True{} : Bool},
    {Nat.is_lt(Jobs.physical_row(work(selection(reduction)),line(reduction)),U32.to_nat((Jobs.chunk_row(chunk(query(selection(reduction)))) + Jobs.chunk_rows(chunk(query(selection(reduction)))) : U32))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(first(query(selection(reduction)))),Jobs.physical_row(work(selection(reduction)),line(reduction))) == True{} : Bool},
    {Nat.is_lt(Jobs.physical_row(work(selection(reduction)),line(reduction)),Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction))))))) == True{} : Bool}>:
  Reduction{+selected,+line,local_group,unit,output,tr,tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,+first,target,plane,source,+row,+group,wb,ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{+ci,+room,fits,enabled,ji,oi,ri,coli} = valid
  Shape.Both{+lo,+hi} = bounds
  +physical = Jobs.physical_row(work(selected),line)
  +start = Jobs.chunk_row(chunk(q))
  +rows = Jobs.chunk_rows(chunk(q))
  Shape.Both{Order.transitive(U32.to_nat(first),U32.to_nat(start),physical,Chunks.selected_band_lower(grid,first,row,group,ci,room),lo),
    Bounds.nat_trans(physical,Nat.add(U32.to_nat(start),U32.to_nat(rows)),Nat.add(U32.to_nat(first),U32.to_nat(Chunks.grid_count(grid))),
      %Jobs.chunk_end_exact(work(selected),contained(selected,ci,room)) : {Nat.is_lt(physical,_) == True{} : Bool} hi,Chunks.selected_band_end(grid,first,row,group,ci,room))}
def row_in_leaf(+reduction: Reduction,+valid: Valid<reduction>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(first(query(selection(reduction)))),Jobs.physical_row(work(selection(reduction)),line(reduction))) == True{} : Bool},
    {Nat.is_lt(Jobs.physical_row(work(selection(reduction)),line(reduction)),Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction))))))) == True{} : Bool}>:
  Reduction{+selected,+line,+local_group,+unit,output,tr,tc} = reduction
  Valid{+ci,+room,fits,+enabled,+ji,oi,ri,coli} = valid
  row_in_leaf_from(reduction,valid,Jobs.physical_row_bounds(work(selected),Jobs.active_phase(phase(selected),enabled),line,contained(selected,ci,room),Jobs.line_inside(work(selected),line,local_group,unit,ji)))

def leaf_band(+reduction: Reduction,+valid: Valid<reduction>) -> Bands.Query:
  +query = query(selection(reduction))
  +grid = grid(query)
  Bands.caller_query(Chunks.grid_call(grid),Chunks.grid_envelope(grid),
    first(query),Chunks.grid_count(grid),leaf_room(reduction,valid))
def leaf_end_exact(+reduction: Reduction,+valid: Valid<reduction>) ->
  {U32.to_nat((first(query(selection(reduction))) + Chunks.grid_count(grid(query(selection(reduction)))) : U32)) == Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction)))))) : Nat}:
  Reduction{Selection{Query{+grid,+first,target,plane,source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Grid{+call,e,+count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Valid{ci,+room,fits,enabled,ji,oi,ri,coli} = valid
  Arithmetic.add(first,count,Machine.dimension(l),Limits.dimension_fits(l),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(g),U32.to_nat(Machine.dimension(l)),room,Domain.height_room(call)))
def leaf_indices(+reduction: Reduction) -> BandSamples.Indices:
  +selected = selection(reduction)
  +q = query(selected)
  BandSamples.indices(Jobs.row_axis(phase(selected)),Domain.stride_word(geometry(q)),first(q),(first(q) + Chunks.grid_count(grid(q)) : U32))
def leaf_indices_bounds_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(first(query(selection(reduction)))),Jobs.physical_row(work(selection(reduction)),line(reduction))) == True{} : Bool},
    {Nat.is_lt(Jobs.physical_row(work(selection(reduction)),line(reduction)),Nat.add(U32.to_nat(first(query(selection(reduction)))),U32.to_nat(Chunks.grid_count(grid(query(selection(reduction))))))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(BandSamples.low(leaf_indices(reduction))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(BandSamples.high(leaf_indices(reduction)))) == True{} : Bool}>:
  Reduction{+selected,+line,+local_group,+unit,output,tr,tc} = reduction
  Selection{+q,+y,x,yi,xi} = selected
  Query{+grid,+first,target,plane,source,row,group,wb,ob} = q
  Chunks.Grid{+call,e,+count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,+s,+p,gh,gw} = g
  Valid{+ci,+room,fits,+enabled,+ji,oi,ri,coli} = valid
  Shape.Both{+lo,+hi} = bounds
  +band = leaf_band(reduction,valid)
  +active = Jobs.active_phase(phase(selected),enabled)
  +position = Jobs.row_native(work(selected),line)
  BandSamples.member_indices(Kernel.axis(U32.from_nat(y),h,k,s,p),s,first,(first + count : U32),l,Bands.first_room(band),Bands.last_room(band),Domain.stride_room(call),Domain.stride_positive(call),U32.to_nat(position),
    Jobs.row_selection_inside(work(selected),active,line,contained(selected,ci,room),Jobs.line_inside(work(selected),line,local_group,unit,ji)),lo,
    %Equal.sym(Nat,U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),leaf_end_exact(reduction,valid)) : {Nat.is_lt(Jobs.physical_row(work(selected),line),_) == True{} : Bool} hi)
def leaf_indices_bounds(+reduction: Reduction,+valid: Valid<reduction>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(BandSamples.low(leaf_indices(reduction))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(BandSamples.high(leaf_indices(reduction)))) == True{} : Bool}>:
  leaf_indices_bounds_from(reduction,valid,row_in_leaf(reduction,valid))

def leaf_phase_y_inside(+reduction: Reduction,+valid: Valid<reduction>) -> {Nat.is_lt(Jobs.phase_y(phase(selection(reduction))),Bands.count(leaf_band(reduction,valid))) == True{} : Bool}:
  Reduction{Selection{Query{+grid,first,target,plane,source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  yi

def leaf_used_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(BandSamples.low(leaf_indices(reduction))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(BandSamples.high(leaf_indices(reduction)))) == True{} : Bool}>) ->
  {Bands.used(leaf_band(reduction,valid),U32.from_nat(Jobs.phase_y(phase(selection(reduction))))) == True{} : Bool}:
  Reduction{+selected,+line,local_group,unit,output,tr,tc} = reduction
  Selection{+q,+y,x,yi,xi} = selected
  Query{+grid,+first,target,plane,source,row,group,wb,ob} = q
  Chunks.Grid{+call,e,+count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,+h,w,+k,+s,+p,gh,gw} = g
  Valid{ci,room,fits,+enabled,ji,oi,ri,coli} = valid
  +axis = Kernel.axis(U32.from_nat(y),h,k,s,p)
  %BandSamples.used_matches(axis,k,s,first,(first + count : U32)) : {_ == True{} : Bool}
  BandSamples.member_used_from(axis,s,first,(first + count : U32),U32.to_nat(Jobs.row_native(work(selected),line)),Jobs.row_active(phase(selected),Jobs.active_phase(phase(selected),enabled)),bounds)
def leaf_used(+reduction: Reduction,+valid: Valid<reduction>) -> {Bands.used(leaf_band(reduction,valid),U32.from_nat(Jobs.phase_y(phase(selection(reduction))))) == True{} : Bool}:
  leaf_used_from(reduction,valid,leaf_indices_bounds(reduction,valid))

def leaf_selection(+reduction: Reduction,+valid: Valid<reduction>) -> BandSamples.Selection:
  Bands.selection(leaf_band(reduction,valid),Jobs.phase_y(phase(selection(reduction))),leaf_phase_y_inside(reduction,valid),leaf_used(reduction,valid))
def leaf_selection_indices(+reduction: Reduction,+valid: Valid<reduction>) -> {BandSamples.selection_indices(leaf_selection(reduction,valid)) == leaf_indices(reduction) : BandSamples.Indices}:
  Reduction{Selection{Query{+grid,first,target,plane,source,row,group,wb,ob},y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  {==}
def leaf_position_bounds_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(BandSamples.low(leaf_indices(reduction))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(BandSamples.high(leaf_indices(reduction)))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(PureBands.low(BandSamples.selection_model(leaf_selection(reduction,valid))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),PureBands.high(BandSamples.selection_model(leaf_selection(reduction,valid)))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  +same = leaf_selection_indices(reduction,valid)
  BandSamples.selection_bounds(leaf_selection(reduction,valid),Jobs.row_native(work(selection(reduction)),line(reduction)),
    %Equal.sym(BandSamples.Indices,BandSamples.selection_indices(leaf_selection(reduction,valid)),leaf_indices(reduction),same) : {Nat.is_le(U32.to_nat(BandSamples.low(_)),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool} lo,
    %Equal.sym(BandSamples.Indices,BandSamples.selection_indices(leaf_selection(reduction,valid)),leaf_indices(reduction),same) : {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(BandSamples.high(_))) == True{} : Bool} hi)
def leaf_position_bounds(+reduction: Reduction,+valid: Valid<reduction>) ->
  Shape.Both<{Nat.is_le(PureBands.low(BandSamples.selection_model(leaf_selection(reduction,valid))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),PureBands.high(BandSamples.selection_model(leaf_selection(reduction,valid)))) == True{} : Bool}>:
  leaf_position_bounds_from(reduction,valid,leaf_indices_bounds(reduction,valid))

def leaf_position_inside(+reduction: Reduction,+valid: Valid<reduction>) ->
  {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),U32.to_nat(PhaseGeometry.axis_count(BandSamples.selection_axis(leaf_selection(reduction,valid))))) == True{} : Bool}:
  Reduction{+selected,+line,+local_group,+unit,output,tr,tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,wb,ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{+ci,+room,fits,+enabled,+ji,oi,ri,coli} = valid
  Jobs.row_selection_inside(work(selected),Jobs.active_phase(phase(selected),enabled),line,contained(selected,ci,room),Jobs.line_inside(work(selected),line,local_group,unit,ji))
def leaf_coordinate(+reduction: Reduction,+valid: Valid<reduction>) -> U32:
  BandSamples.native_tapped(leaf_selection(reduction,valid),Jobs.row_native(work(selection(reduction)),line(reduction)),U32.from_nat(tap_row(reduction)))
def leaf_tapped_bounds_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{Nat.is_le(PureBands.low(BandSamples.selection_model(leaf_selection(reduction,valid))),U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(Jobs.row_native(work(selection(reduction)),line(reduction))),PureBands.high(BandSamples.selection_model(leaf_selection(reduction,valid)))) == True{} : Bool}>) ->
  Shape.Both<{U32.is_le(Bands.low_fold(leaf_band(reduction,valid)),leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(leaf_coordinate(reduction,valid),Bands.high_fold(leaf_band(reduction,valid))) == True{} : Bool}>:
  Reduction{+selected,line,local_group,unit,output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,wb,ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{ci,room,fits,+enabled,ji,oi,+ri,+coli} = valid
  Shape.Both{+lo,+hi} = bounds
  +exact = Shape.first({U32.to_nat(U32.from_nat(tr)) == tr : Nat},{U32.to_nat(U32.from_nat(tc)) == tc : Nat},Reads.tap_words_exact(phase(selected),enabled,tr,tc,ri,coli))
  Bands.tapped_native_bounds(leaf_band(reduction,valid),Jobs.phase_y(phase(selected)),leaf_phase_y_inside(reduction,valid),leaf_used(reduction,valid),Jobs.row_native(work(selected),line),U32.from_nat(tr),leaf_position_inside(reduction,valid),lo,hi,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(tr)),tr,exact) : {Nat.is_lt(_,BandSamples.selection_taps(leaf_selection(reduction,valid))) == True{} : Bool} ri)
def leaf_tapped_bounds(+reduction: Reduction,+valid: Valid<reduction>) ->
  Shape.Both<{U32.is_le(Bands.low_fold(leaf_band(reduction,valid)),leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(leaf_coordinate(reduction,valid),Bands.high_fold(leaf_band(reduction,valid))) == True{} : Bool}>:
  leaf_tapped_bounds_from(reduction,valid,leaf_position_bounds(reduction,valid))

def row_relative_to_leaf_coordinate(+reduction: Reduction,+valid: Valid<reduction>) ->
  {Lookup.read_row_total(sample(reduction)) == (leaf_coordinate(reduction,valid) - Kernel.band_low(geometry(query(selection(reduction))),Jobs.chunk_row(chunk(query(selection(reduction)))),(Jobs.chunk_row(chunk(query(selection(reduction)))) + Jobs.chunk_rows(chunk(query(selection(reduction)))) : U32)) : U32) : U32}:
  Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Selection{+q,y,x,yi,xi} = selected
  Query{+grid,first,target,plane,source,row,group,+wb,+ob} = q
  Chunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Valid{+ci,+room,fits,+enabled,+ji,+oi,+ri,+coli} = valid
  +chunk_contained = contained(selected,ci,room)
  +band = Jobs.work_band(work(selected),chunk_contained)
  +coordinate = leaf_coordinate(reduction,valid)
  +low = Kernel.band_low(g,Jobs.chunk_row(chunk(q)),(Jobs.chunk_row(chunk(q)) + Jobs.chunk_rows(chunk(q)) : U32))
  Equal.trans(U32,Lookup.read_row_total(sample(reduction)),(coordinate - Bands.low_fold(band) : U32),(coordinate - low : U32),
    Reads.read_row_relative(work(selected),enabled,wb,ob,line,local_group,unit,output,tr,tc,ji,oi,ri,coli,chunk_contained,low_matches(selected)),
    Equal.cong(U32,U32,lower => (coordinate - lower : U32),Bands.low_fold(band),low,Equal.sym(U32,low,Bands.low_fold(band),Bands.actual_low_matches(band,Jobs.work_band_rows(work(selected),chunk_contained)))))

def leaf_halo_bounds_from(+reduction: Reduction,+valid: Valid<reduction>,
  bounds: Shape.Both<{U32.is_le(Bands.low_fold(leaf_band(reduction,valid)),leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(leaf_coordinate(reduction,valid),Bands.high_fold(leaf_band(reduction,valid))) == True{} : Bool}>) ->
  Shape.Both<{U32.is_le(Kernel.band_low(Bands.geometry(leaf_band(reduction,valid)),Bands.first(leaf_band(reduction,valid)),Bands.last(leaf_band(reduction,valid))),leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(leaf_coordinate(reduction,valid),Kernel.band_high(Bands.geometry(leaf_band(reduction,valid)),Bands.first(leaf_band(reduction,valid)),Bands.last(leaf_band(reduction,valid)))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  +band = leaf_band(reduction,valid)
  Shape.Both{
    %Equal.sym(U32,Kernel.band_low(Bands.geometry(band),Bands.first(band),Bands.last(band)),Bands.low_fold(band),Bands.actual_low_matches(band,{==})) : {U32.is_le(_,leaf_coordinate(reduction,valid)) == True{} : Bool} lo,
    %Equal.sym(U32,Kernel.band_high(Bands.geometry(band),Bands.first(band),Bands.last(band)),Bands.high_fold(band),Bands.actual_high_matches(band,{==})) : {U32.is_lt(leaf_coordinate(reduction,valid),_) == True{} : Bool} hi}
def leaf_halo_bounds(+reduction: Reduction,+valid: Valid<reduction>) ->
  Shape.Both<{U32.is_le(Kernel.band_low(Bands.geometry(leaf_band(reduction,valid)),Bands.first(leaf_band(reduction,valid)),Bands.last(leaf_band(reduction,valid))),leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(leaf_coordinate(reduction,valid),Kernel.band_high(Bands.geometry(leaf_band(reduction,valid)),Bands.first(leaf_band(reduction,valid)),Bands.last(leaf_band(reduction,valid)))) == True{} : Bool}>:
  leaf_halo_bounds_from(reduction,valid,leaf_tapped_bounds(reduction,valid))
