# The filled slot's lane is the ordinary retained-gradient pixel at that address.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Spatial
import ../kernel_packing_samples.bend as Masks
import ../math_sequence.bend as Sequence
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_tile_refinement.bend as TileLaws
import ./indexed_sequence_laws.bend as SequenceLaws
import ./storage_tree_model.bend as Tree
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_lookup.bend as Lookup
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_job_reads.bend as Reads
import ./input_gradient_tiled_columns.bend as Columns
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_phase_domain.bend as Domain
import ./word_subtraction.bend as Subtraction
import ./input_gradient_tiled_source_rows.bend as SourceRows
import ./storage_buffer_model.bend as Buffer
import ./u32_comparison.bend as Comparison
import ./word_multiplication.bend as Multiplication
import ./nat_order.bend as Order
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./binary_radix.bend as Radix
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition

def pure_lane(+input: Tree.Tree<F32>,+context: Spatial.PixelBlock,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Samples.pure_vector(input,context)) == Samples.block_sample(input,context,Sequence.index_value(8n,lane)) : F32}:
  %Addition.zero_right(Sequence.index_value(8n,lane)) : {Tile.element_at(lane,Samples.pure_vector(input,context)) == Samples.block_sample(input,context,_) : F32}
  SequenceLaws.get_tabulated(~F32,~TileLaws.SampleContext<Tree.Tree<F32>,Spatial.PixelBlock>,~TileLaws.sample_context(~Tree.Tree<F32>,~Spatial.PixelBlock,~Samples.block_sample),8n,lane,0n,TileLaws.SampleContext{input,context})
def position_exact(+query: Fill.Query,+facts: Fill.Facts<query>) -> {U32.to_nat(U32.from_nat(Fill.position(query))) == Fill.position(query) : Nat}:
  Fill.Facts{+capacity_ok,span_positive,span_ordered,double_span,room,+position_inside,output_inside,shift_inside} = facts
  Bounds.roundtrip(Fill.position(query),Fill.capacity(query),capacity_ok,Bounds.strict_le(Fill.position(query),U32.to_nat(Fill.capacity(query)),Bounds.nat_trans(Fill.position(query),Fill.count(query),U32.to_nat(Fill.capacity(query)),position_inside,Fill.count_room(query,facts))))
def quotient_exact(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>) ->
  {(U32.from_nat(Fill.position(Lookup.fill(query))) / Lookup.span(query) : U32) == Lookup.row(query) : U32}:
  Lookup.Fields{+located,output,shift,+inside} = fields
  Fill.Facts{+capacity_ok,+positive,ordered,+double_span,room,position_inside,output_inside,shift_inside} = facts
  +fq = Lookup.fill(query)
  +rw = Lookup.row(query)
  +gw = Lookup.group(query)
  +width = Lookup.span(query)
  +quotient = (U32.from_nat(Fill.position(fq)) / width : U32)
  +encoded = Equal.trans(Nat,U32.to_nat(U32.from_nat(Fill.position(fq))),Fill.position(fq),Nat.add(Nat.mul(U32.to_nat(rw),U32.to_nat(width)),U32.to_nat(gw)),position_exact(fq,facts),located)
  +exact = Radix.quotient_word(U32.from_nat(Fill.position(fq)),width,U32.to_nat(rw),U32.to_nat(gw),U32.to_nat(width),Fill.capacity(fq),capacity_ok,encoded,{==},positive,inside,double_span)
  Equal.trans(U32,quotient,U32.from_nat(U32.to_nat(rw)),rw,Encoded.encoding(quotient,U32.to_nat(rw),exact),Arithmetic.native_roundtrip(rw))
def remainder_exact(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>) ->
  {(U32.from_nat(Fill.position(Lookup.fill(query))) % Lookup.span(query) : U32) == Lookup.group(query) : U32}:
  Lookup.Fields{+located,output,shift,+inside} = fields
  Fill.Facts{+capacity_ok,+positive,ordered,+double_span,room,position_inside,output_inside,shift_inside} = facts
  +fq = Lookup.fill(query)
  +rw = Lookup.row(query)
  +gw = Lookup.group(query)
  +width = Lookup.span(query)
  +remainder = (U32.from_nat(Fill.position(fq)) % width : U32)
  +encoded = Equal.trans(Nat,U32.to_nat(U32.from_nat(Fill.position(fq))),Fill.position(fq),Nat.add(Nat.mul(U32.to_nat(rw),U32.to_nat(width)),U32.to_nat(gw)),position_exact(fq,facts),located)
  +exact = Radix.remainder_word(U32.from_nat(Fill.position(fq)),width,U32.to_nat(rw),U32.to_nat(gw),U32.to_nat(width),Fill.capacity(fq),capacity_ok,encoded,{==},positive,inside,double_span)
  Equal.trans(U32,remainder,U32.from_nat(U32.to_nat(gw)),gw,Encoded.encoding(remainder,U32.to_nat(gw),exact),Arithmetic.native_roundtrip(gw))

def block(context: Kernel.Filling,+row: U32,+group: U32,+output: U32,+shift: U32) -> Spatial.PixelBlock:
  Kernel.Filling{Kernel.Tiling{groups,shifts,outputs,+column_low,+kernel,height,+width,cg,cr},+low,+base,+plane,+first,+rows,+first_group,span} = context
  Spatial.PixelBlock{(row + low - kernel - first : U32),((group + first_group) * 8 + column_low - kernel + shift : U32),(base + output * plane : U32),rows,width,1}
def query_block(+query: Lookup.Query) -> Spatial.PixelBlock:
  block(Fill.filling(Lookup.fill(query)),Lookup.row(query),Lookup.group(query),Lookup.output(query),Lookup.shift(query))
def wanted_block(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>) ->
  {Fill.wanted(Lookup.fill(query)) == Samples.pure_vector(Fill.original(Lookup.fill(query)),query_block(query)) : Tile.Tile8}:
  Lookup.Query{+fq,+row,+group,+output,+shift} = query
  Fill.Query{+context,+input,rows,+position,+channel,+selected_shift,depth,capacity} = fq
  Kernel.Filling{+layout,low,base,plane,first,source_rows,first_group,+span} = context
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  Lookup.Fields{located,+oe,+se,inside} = fields
  +qw = (U32.from_nat(position) / span : U32)
  +gw = (U32.from_nat(position) % span : U32)
  +ow = U32.from_nat(channel)
  +sw = U32.from_nat(selected_shift)
  +out_exact = Equal.trans(U32,ow,U32.from_nat(U32.to_nat(output)),output,Equal.cong(Nat,U32,U32.from_nat,channel,U32.to_nat(output),oe),Arithmetic.native_roundtrip(output))
  +shift_exact = Equal.trans(U32,sw,U32.from_nat(U32.to_nat(shift)),shift,Equal.cong(Nat,U32,U32.from_nat,selected_shift,U32.to_nat(shift),se),Arithmetic.native_roundtrip(shift))
  %Equal.sym(U32,qw,row,quotient_exact(query,fields,facts)) : {Samples.pure_vector(input,block(context,_,gw,ow,sw)) == Samples.pure_vector(input,query_block(query)) : Tile.Tile8}
  %Equal.sym(U32,gw,group,remainder_exact(query,fields,facts)) : {Samples.pure_vector(input,block(context,row,_,ow,sw)) == Samples.pure_vector(input,query_block(query)) : Tile.Tile8}
  %Equal.sym(U32,ow,output,out_exact) : {Samples.pure_vector(input,block(context,row,group,_,sw)) == Samples.pure_vector(input,query_block(query)) : Tile.Tile8}
  %Equal.sym(U32,sw,shift,shift_exact) : {Samples.pure_vector(input,block(context,row,group,output,_)) == Samples.pure_vector(input,query_block(query)) : Tile.Tile8}
  {==}
def wanted_lane(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Fill.wanted(Lookup.fill(query))) == Samples.block_sample(Fill.original(Lookup.fill(query)),query_block(query),Sequence.index_value(8n,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,Fill.wanted(Lookup.fill(query))),Tile.element_at(lane,Samples.pure_vector(Fill.original(Lookup.fill(query)),query_block(query))),Samples.block_sample(Fill.original(Lookup.fill(query)),query_block(query),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),Fill.wanted(Lookup.fill(query)),Samples.pure_vector(Fill.original(Lookup.fill(query)),query_block(query)),wanted_block(query,fields,facts)),pure_lane(Fill.original(Lookup.fill(query)),query_block(query),lane))
def filled_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) ==
    Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)) : F32}:
  +query = Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),Tile.element_at(lane,Fill.wanted(Chunks.slot(reduction))),Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(query),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))),Fill.wanted(Chunks.slot(reduction)),Chunks.readback(reduction,valid)),
    wanted_lane(query,Chunks.lookup_fields(reduction,valid),Fill.frame_facts(Chunks.slot(reduction),Fill.initial(Chunks.slot(reduction)),Chunks.slot_frame(reduction,valid)),lane))

def block_row(value: Spatial.PixelBlock) -> U32:
  Spatial.PixelBlock{row,column,base,height,width,stride} = value
  row
def block_column(value: Spatial.PixelBlock) -> U32:
  Spatial.PixelBlock{row,column,base,height,width,stride} = value
  column
def row_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) ->
  {block_row(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))) ==
    (Chunks.leaf_coordinate(reduction,valid) - Domain.kernel_word(Chunks.geometry(Chunks.query(Chunks.selection(reduction)))) - Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))) : U32) : U32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,+first,target,plane,+source,+row,+group,wb,ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,+source_first,source_rows} = source
  +coordinate = Chunks.leaf_coordinate(reduction,valid)
  +rw = Lookup.read_row_total(Chunks.sample(reduction))
  +start = GeometryChunks.selected_start(grid,first,row,group)
  +rows = GeometryChunks.selected_rows(grid,row,group)
  +low = Kernel.band_low(g,start,(start + rows : U32))
  +restore_coordinate = Equal.trans(U32,(rw + low : U32),((coordinate - low) + low : U32),coordinate,
    Equal.cong(U32,U32,value => (value + low : U32),rw,(coordinate - low : U32),Chunks.row_relative_to_leaf_coordinate(reduction,valid)),Subtraction.add_subtract(coordinate,low))
  Equal.cong(U32,U32,value => (value - k - source_first : U32),(rw + low : U32),coordinate,restore_coordinate)

def image(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,original: Buffer.Model<F32>,base: U32,plane: U32) -> SourceRows.Slice:
  SourceRows.query_slice(Chunks.leaf_band(reduction,valid),Chunks.source(Chunks.query(Chunks.selection(reduction))),original,base,plane)
def global_row(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) -> U32:
  (Chunks.leaf_coordinate(reduction,valid) - Domain.kernel_word(Chunks.geometry(Chunks.query(Chunks.selection(reduction)))) : U32)
def column_at(+reduction: Chunks.Reduction,+lane: Sequence.Index(8n)) -> U32:
  (block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))) + U32.from_nat(Sequence.index_value(8n,lane)) * 1 : U32)
def local_pixel_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+lane: Sequence.Index(8n)) ->
  {Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)) ==
    SourceRows.local_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,first,target,leaf_plane,+source,row,group,wb,ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,+gw} = g
  Execution.Source{+data,+source_base,+source_plane,+source_first,+source_rows} = source
  Chunks.Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  +query = Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))
  +column = column_at(reduction,lane)
  +output = Lookup.read_output(Chunks.sample(reduction))
  Equal.cong(U32,F32,row => Samples.value(data,row,column,(source_base + output * source_plane : U32),source_rows,gw),block_row(query_block(query)),(global_row(reduction,valid) - source_first : U32),row_matches(reduction,valid))

def global_pixel_matches_from(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base,plane),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>,
  halo: Shape.Both<{U32.is_le(Kernel.band_low(Bands.geometry(Chunks.leaf_band(reduction,valid)),Bands.first(Chunks.leaf_band(reduction,valid)),Bands.last(Chunks.leaf_band(reduction,valid))),Chunks.leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(Chunks.leaf_coordinate(reduction,valid),Kernel.band_high(Bands.geometry(Chunks.leaf_band(reduction,valid)),Bands.first(Chunks.leaf_band(reduction,valid)),Bands.last(Chunks.leaf_band(reduction,valid)))) == True{} : Bool}>) ->
  {Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)) ==
    SourceRows.global_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,first,target,leaf_plane,+source,row,group,wb,ob} = q
  GeometryChunks.Grid{+call,e,count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  Shape.Both{+lo,+hi} = halo
  +slice = image(reduction,valid,original,base,plane)
  +oc = Lookup.read_output(Chunks.sample(reduction))
  +row = global_row(reduction,valid)
  +column = column_at(reduction,lane)
  +outputs = Fill.outputs(Chunks.filling(q))
  +oc_inside = {%Equal.sym(Bool,U32.is_lt(oc,outputs),Nat.is_lt(U32.to_nat(oc),U32.to_nat(outputs)),Comparison.lt(oc,outputs)) : {_ == True{} : Bool} Chunks.output_inside_slot(reduction,valid) : {U32.is_lt(oc,outputs) == True{} : Bool}}
  Equal.trans(F32,Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)),SourceRows.local_pixel(slice,oc,row,column),SourceRows.global_pixel(slice,oc,row,column),local_pixel_matches(reduction,valid,original,base,plane,lane),
    SourceRows.halo_pixel(Chunks.leaf_band(reduction,valid),source,original,base,plane,outputs,oc,Chunks.leaf_coordinate(reduction,valid),column,range,covers,stored,oc_inside,lo,hi))
def global_pixel_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base,plane),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)) ==
    SourceRows.global_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  global_pixel_matches_from(reduction,valid,original,base,plane,lane,range,covers,stored,Chunks.leaf_halo_bounds(reduction,valid))

def filled_global_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+base: U32,+plane: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base,plane),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) ==
    SourceRows.global_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),Samples.block_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Sequence.index_value(8n,lane)),SourceRows.global_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)),
    filled_lane(reduction,valid,lane),global_pixel_matches(reduction,valid,original,base,plane,lane,range,covers,stored))

def row_source_phase_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) ->
  {global_row(reduction,valid) == (Packing.input_row(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction))) + U32.from_nat(Chunks.tap_row(reduction)) : U32) : U32}:
  Chunks.Reduction{+selected,+line,local_group,unit,output,+tr,tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +position = Jobs.row_native(Chunks.work(selected),line)
  +ys = Packing.axis_source(Packing.phase_y(phase))
  +tap = U32.from_nat(tr)
  +origin = (position + ys : U32)
  +grouped = Equal.cong(U32,U32,value => (value + tap : U32),(position + (ys + k) : U32),(origin + k : U32),Equal.sym(U32,(origin + k : U32),(position + (ys + k) : U32),Arithmetic.add_associative(position,ys,k)))
  +removed = Equal.trans(U32,(((origin + k) + tap) - k : U32),(((origin + k) - k) + tap : U32),(origin + tap : U32),Equal.sym(U32,(((origin + k) - k) + tap : U32),(((origin + k) + tap) - k : U32),Arithmetic.add_after_subtract((origin + k : U32),k,tap)),
    Equal.cong(U32,U32,value => (value + tap : U32),((origin + k) - k : U32),origin,Subtraction.subtract_add(origin,k)))
  Equal.trans(U32,global_row(reduction,valid),(((origin + k) + tap) - k : U32),(Packing.input_row(phase,position) + tap : U32),
    Equal.cong(U32,U32,value => (value - k : U32),Chunks.leaf_coordinate(reduction,valid),((origin + k) + tap : U32),grouped),
    Equal.trans(U32,(((origin + k) + tap) - k : U32),(origin + tap : U32),(Packing.input_row(phase,position) + tap : U32),removed,Equal.cong(U32,U32,value => (value + tap : U32),origin,Packing.input_row(phase,position),Arithmetic.add_commutative(position,ys))))

def column_source_phase_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) ->
  {block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))) ==
    (Packing.input_column(Jobs.context(Chunks.phase(Chunks.selection(reduction))),((Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(reduction)))) + Lookup.read_group(Chunks.sample(reduction))) * 8 : U32)) + U32.from_nat(Chunks.tap_column(reduction)) : U32) : U32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,+source,row,group,+wb,+ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  Chunks.Valid{ci,room,fits,+enabled,ji,+oi,+ri,+coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +sample = Chunks.sample(reduction)
  +group_word = Lookup.read_group(sample)
  +first_group = GeometryChunks.group_offset(grid,row,group)
  +packed_column = ((group_word + first_group) * 8 : U32)
  +position = ((first_group + group_word) * 8 : U32)
  +low = Columns.layout_column_low(Kernel.tiling(g))
  +xs = Packing.axis_source(Packing.phase_x(phase))
  +shift = (xs + k - low : U32)
  +tap = U32.from_nat(tc)
  +prefix = (packed_column + low - k : U32)
  +tap_equal = Reads.read_tap_column_from(Chunks.work(selected),wb,ob,line,local_group,unit,output,tr,tc,Reads.taps(Chunks.phase(selected),enabled,output,tr,tc,oi,ri,coli))
  +shift_grouped = Equal.trans(U32,Lookup.read_shift_total(sample),(tap + shift : U32),(shift + tap : U32),Equal.cong(U32,U32,value => (value + shift : U32),Lookup.read_tap_column(sample),tap,tap_equal),Arithmetic.add_commutative(tap,shift))
  +restore_low = Equal.trans(U32,(low + shift : U32),(shift + low : U32),(xs + k : U32),Arithmetic.add_commutative(low,shift),Subtraction.add_subtract((xs + k : U32),low))
  +restore_origin = Equal.trans(U32,((packed_column + low) + shift : U32),(packed_column + (low + shift) : U32),((packed_column + xs) + k : U32),Arithmetic.add_associative(packed_column,low,shift),
    Equal.trans(U32,(packed_column + (low + shift) : U32),(packed_column + (xs + k) : U32),((packed_column + xs) + k : U32),Equal.cong(U32,U32,value => (packed_column + value : U32),(low + shift : U32),(xs + k : U32),restore_low),Equal.sym(U32,((packed_column + xs) + k : U32),(packed_column + (xs + k) : U32),Arithmetic.add_associative(packed_column,xs,k))))
  +prefix_restored = Equal.trans(U32,(prefix + shift : U32),(((packed_column + low) + shift) - k : U32),(packed_column + xs : U32),Arithmetic.add_after_subtract((packed_column + low : U32),k,shift),
    Equal.trans(U32,(((packed_column + low) + shift) - k : U32),(((packed_column + xs) + k) - k : U32),(packed_column + xs : U32),Equal.cong(U32,U32,value => (value - k : U32),((packed_column + low) + shift : U32),((packed_column + xs) + k : U32),restore_origin),Subtraction.subtract_add((packed_column + xs : U32),k)))
  +group_equal = Equal.cong(U32,U32,value => (value * 8 : U32),(group_word + first_group : U32),(first_group + group_word : U32),Arithmetic.add_commutative(group_word,first_group))
  +source_equal = Equal.trans(U32,(packed_column + xs : U32),(xs + packed_column : U32),Packing.input_column(phase,position),Arithmetic.add_commutative(packed_column,xs),Equal.cong(U32,U32,value => (xs + value : U32),packed_column,position,group_equal))
  Equal.trans(U32,block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),sample))),(prefix + (shift + tap) : U32),(Packing.input_column(phase,position) + tap : U32),Equal.cong(U32,U32,value => (prefix + value : U32),Lookup.read_shift_total(sample),(shift + tap : U32),shift_grouped),
    Equal.trans(U32,(prefix + (shift + tap) : U32),((prefix + shift) + tap : U32),(Packing.input_column(phase,position) + tap : U32),Equal.sym(U32,((prefix + shift) + tap : U32),(prefix + (shift + tap) : U32),Arithmetic.add_associative(prefix,shift,tap)),
      Equal.cong(U32,U32,value => (value + tap : U32),(prefix + shift : U32),Packing.input_column(phase,position),Equal.trans(U32,(prefix + shift : U32),(packed_column + xs : U32),Packing.input_column(phase,position),prefix_restored,source_equal))))

def input_column(+reduction: Chunks.Reduction) -> U32:
  ((Jobs.chunk_group(Chunks.chunk(Chunks.query(Chunks.selection(reduction)))) + Lookup.read_group(Chunks.sample(reduction))) * 8 : U32)
def phase_sample(+reduction: Chunks.Reduction,+image: U32) -> Packing.Sample:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(reduction)))
  Packing.Sample{(Packing.input_base(phase,image) + U32.from_nat(Nat.mul(Chunks.output(reduction),U32.to_nat(Packing.gradient_step(phase)))) : U32),
    (Packing.input_row(phase,Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction))) + U32.from_nat(Chunks.tap_row(reduction)) : U32),
    (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(Chunks.tap_column(reduction)) : U32),Packing.gradient_rows(phase),Packing.gradient_columns(phase)}
def sample_raw(+input: Buffer.Model<F32>,context: Packing.Sample,+index: Nat) -> F32:
  Packing.Sample{base,row,column,height,width} = context
  Samples.value(Buffer.tree(~F32,input),row,(column + U32.from_nat(index) : U32),base,height,width)
def sample_mask(context: Packing.Sample,+index: Nat) -> Bool:
  Packing.Sample{base,row,column,height,width} = context
  U32.is_lt(row,height) && U32.is_lt((column + U32.from_nat(index) : U32),width)
def sample_choice_case(+input: Buffer.Model<F32>,+context: Packing.Sample,+index: Nat,enabled: Bool,+same: {sample_mask(context,index) == enabled : Bool}) ->
  {sample_raw(input,context,index) == Packing.sample_value(input,context,index) : F32}:
  Packing.Sample{+base,+row,+column,+height,+width} = context
  match enabled:
    case False{}:
      %Equal.sym(Bool,sample_mask(context,index),False{},same) :
        {Masks.masked(Tree.read(F32,Buffer.tree(~F32,input),(base + row * width + (column + U32.from_nat(index)) : U32)),_) == Bool.pick(F32,_,Tree.read(F32,Buffer.tree(~F32,input),(base + row * width + (column + U32.from_nat(index)) : U32)),0.0) : F32} {==}
    case True{}:
      %Equal.sym(Bool,sample_mask(context,index),True{},same) :
        {Masks.masked(Tree.read(F32,Buffer.tree(~F32,input),(base + row * width + (column + U32.from_nat(index)) : U32)),_) == Bool.pick(F32,_,Tree.read(F32,Buffer.tree(~F32,input),(base + row * width + (column + U32.from_nat(index)) : U32)),0.0) : F32} {==}
def sample_choice(+input: Buffer.Model<F32>,+context: Packing.Sample,+index: Nat) -> {sample_raw(input,context,index) == Packing.sample_value(input,context,index) : F32}:
  sample_choice_case(input,context,index,sample_mask(context,index),{==})
def source_pixel_phase_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+image_id: U32,+lane: Sequence.Index(8n)) ->
  {SourceRows.global_pixel(image(reduction,valid,original,Packing.input_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id),Packing.gradient_step(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) ==
    Packing.sample_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)) : F32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,row,group,+wb,+ob} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = g
  Chunks.Valid{ci,room,fits,+enabled,ji,+oi,+ri,+coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +step = Packing.gradient_step(phase)
  +base = Packing.input_base(phase,image_id)
  +out_word = Lookup.read_output(Chunks.sample(reduction))
  +out_equal = Reads.read_output_from(Chunks.work(selected),wb,ob,line,local_group,unit,output,tr,tc,Reads.taps(Chunks.phase(selected),enabled,output,tr,tc,oi,ri,coli))
  +out_product = Equal.trans(U32,(out_word * step : U32),(U32.from_nat(output) * step : U32),U32.from_nat(Nat.mul(output,U32.to_nat(step))),Equal.cong(U32,U32,value => (value * step : U32),out_word,U32.from_nat(output),out_equal),
    Equal.trans(U32,(U32.from_nat(output) * step : U32),(U32.from_nat(output) * U32.from_nat(U32.to_nat(step)) : U32),U32.from_nat(Nat.mul(output,U32.to_nat(step))),Equal.cong(U32,U32,value => (U32.from_nat(output) * value : U32),step,U32.from_nat(U32.to_nat(step)),Equal.sym(U32,U32.from_nat(U32.to_nat(step)),step,Arithmetic.native_roundtrip(step))),Multiplication.native_encoded_multiply(output,U32.to_nat(step))))
  +source_base = (base + out_word * step : U32)
  +wanted_base = (base + U32.from_nat(Nat.mul(output,U32.to_nat(step))) : U32)
  +base_equal = Equal.cong(U32,U32,value => (base + value : U32),(out_word * step : U32),U32.from_nat(Nat.mul(output,U32.to_nat(step))),out_product)
  +ln = Sequence.index_value(8n,lane)
  +lw = U32.from_nat(ln)
  +lane_unit = Equal.trans(U32,(lw * 1 : U32),U32.from_nat(Nat.mul(ln,1n)),lw,Multiplication.native_encoded_multiply(ln,1n),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(ln,1n),ln,Order.times_one(ln)))
  +wanted_row = (Packing.input_row(phase,Jobs.row_native(Chunks.work(selected),line)) + U32.from_nat(tr) : U32)
  +wanted_column = (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) + lw : U32)
  +column_equal = Equal.trans(U32,column_at(reduction,lane),((Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc)) + lw * 1 : U32),wanted_column,
    Equal.cong(U32,U32,value => (value + lw * 1 : U32),block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))),(Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) : U32),column_source_phase_matches(reduction,valid)),
    Equal.cong(U32,U32,value => (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) + value : U32),(lw * 1 : U32),lw,lane_unit))
  %Equal.sym(U32,global_row(reduction,valid),wanted_row,row_source_phase_matches(reduction,valid)) : {Samples.value(Buffer.tree(~F32,original),_,column_at(reduction,lane),source_base,gh,gw) == Packing.sample_value(original,phase_sample(reduction,image_id),ln) : F32}
  %Equal.sym(U32,column_at(reduction,lane),wanted_column,column_equal) : {Samples.value(Buffer.tree(~F32,original),wanted_row,_,source_base,gh,gw) == Packing.sample_value(original,phase_sample(reduction,image_id),ln) : F32}
  %Equal.sym(U32,source_base,wanted_base,base_equal) : {Samples.value(Buffer.tree(~F32,original),wanted_row,wanted_column,_,gh,gw) == Packing.sample_value(original,phase_sample(reduction,image_id),ln) : F32}
  sample_choice(original,phase_sample(reduction,image_id),ln)

def phase_input_tile(+reduction: Chunks.Reduction,+image_id: U32,original: Buffer.Model<F32>) -> Tile.Tile8:
  Packing.input_tile(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id,Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction)),input_column(reduction),original,Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))
def phase_tile_matches(+reduction: Chunks.Reduction,+image_id: U32,+original: Buffer.Model<F32>) -> {phase_input_tile(reduction,image_id,original) == Packing.sample_tile(original,phase_sample(reduction,image_id)) : Tile.Tile8}:
  {==}
def phase_tile_lane(+reduction: Chunks.Reduction,+image_id: U32,+original: Buffer.Model<F32>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,phase_input_tile(reduction,image_id,original)) == Packing.sample_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),Tile.element_at(lane,Packing.sample_tile(original,phase_sample(reduction,image_id))),Packing.sample_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),phase_input_tile(reduction,image_id,original),Packing.sample_tile(original,phase_sample(reduction,image_id)),phase_tile_matches(reduction,image_id,original)),Packing.tile_value_index(~Packing.Sample,~Packing.sample_value,original,phase_sample(reduction,image_id),lane))
def input_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<F32>,+image_id: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,Packing.input_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id),Packing.gradient_step(Jobs.context(Chunks.phase(Chunks.selection(reduction))))),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) == Tile.element_at(lane,phase_input_tile(reduction,image_id,original)) : F32}:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(reduction)))
  +base = Packing.input_base(phase,image_id)
  +plane = Packing.gradient_step(phase)
  +wanted = SourceRows.global_pixel(image(reduction,valid,original,base,plane),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane))
  +sample = Packing.sample_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane))
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Fill.tree(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),wanted,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),filled_global_lane(reduction,valid,original,base,plane,lane,range,covers,stored),
    Equal.trans(F32,wanted,sample,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),source_pixel_phase_matches(reduction,valid,original,image_id,lane),Equal.sym(F32,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),sample,phase_tile_lane(reduction,image_id,original,lane))))
