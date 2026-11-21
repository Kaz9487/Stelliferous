# The filled slot's lane is the ordinary retained-gradient pixel at that address.
import Base
import ../tensor_tiled.bend as Tiled
import ./input_gradient_tiled_tiles.bend as Tiles
import ./kernel_channel_lanes.bend as CellFields
import ./tensor_tiled_values.bend as TiledValues
import ../tensor_tiled_cells.bend as Cells
import ./u32_division.bend as NativeDivision
import ../kernel_input_gradient.bend as Kernel
import ../kernel_input_gradient_fill.bend as FillKernel

import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Spatial
import ../kernel_packing_samples.bend as Masks
import ../math_sequence.bend as Sequence
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_tile_refinement.bend as TileLaws
import ./indexed_sequence_laws.bend as SequenceLaws
import ./storage_tree_model.bend as Tree
import ./input_gradient_tiled_fill.bend as Fill
import ./kernel_input_gradient_fill_refinement.bend as Filling
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_entry_reads.bend as Lookup
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
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./binary_radix.bend as Radix
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

def pure_lane(~field: Tile.Tile8 -> F32,+input: Tree.Tree<Tile.Tile8>,+context: Spatial.PixelBlock,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Samples.pure_vector(~Tile.Tile8,~field,input,context)) == Samples.block_sample(~Tile.Tile8,~field,input,context,Sequence.index_value(8n,lane)) : F32}:
  %Addition.zero_right(Sequence.index_value(8n,lane)) : {Tile.element_at(lane,Samples.pure_vector(~Tile.Tile8,~field,input,context)) == Samples.block_sample(~Tile.Tile8,~field,input,context,_) : F32}
  SequenceLaws.get_tabulated(~F32,~TileLaws.SampleContext<Tree.Tree<Tile.Tile8>,Spatial.PixelBlock>,~TileLaws.sample_context(~Tree.Tree<Tile.Tile8>,~Spatial.PixelBlock,~Samples.block_sample(~Tile.Tile8,~field)),8n,lane,0n,TileLaws.SampleContext{input,context})

def selection(output: U32) -> Tiles.Selection:
  +lane = (output % 8 : U32)
  Tiles.Selection{U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32))}

def vector(input: Tree.Tree<Tile.Tile8>,context: Spatial.PixelBlock,output: U32) -> Tile.Tile8:
  Tiles.selected_vector(selection(output),input,context)

# A pixel of a channel whose cells lie step cells apart.
def cell_pixel(input: Tree.Tree<Tile.Tile8>,+row: U32,+column: U32,base: U32,height: U32,+width: U32,step: U32,output: U32) -> F32:
  Masks.masked(Cells.lane_of(Tree.read(Tile.Tile8,input,(base + (row * width + column) * step : U32)),(output % 8 : U32)),U32.is_lt(row,height) && U32.is_lt(column,width))

def cell_sample(input: Tree.Tree<Tile.Tile8>,context: Spatial.PixelBlock,output: U32,index: Nat) -> F32:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = context
  cell_pixel(input,row,(column + U32.from_nat(index) * stride : U32),base,height,width,step,output)

def sample_address(context: Spatial.PixelBlock,index: Nat) -> U32:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = context
  (base + (row * width + (column + U32.from_nat(index) * stride)) * step : U32)

def block_mask(context: Spatial.PixelBlock,index: Nat) -> Bool:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = context
  U32.is_lt(row,height) && U32.is_lt((column + U32.from_nat(index) * stride : U32),width)

def field_lane(~field: Tile.Tile8 -> F32,+input: Tree.Tree<Tile.Tile8>,+context: Spatial.PixelBlock,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Samples.pure_vector(~Tile.Tile8,~field,input,context)) ==
    Masks.masked(field(Tree.read(Tile.Tile8,input,sample_address(context,Sequence.index_value(8n,lane)))),block_mask(context,Sequence.index_value(8n,lane))) : F32}:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = context
  pure_lane(~field,input,context,lane)

def selected_lane(+half: Bool,+low: Bool,+first: Bool,+input: Tree.Tree<Tile.Tile8>,+context: Spatial.PixelBlock,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Tiles.selected_vector(Tiles.Selection{half,low,first},input,context)) ==
    Masks.masked(CellFields.lane_dispatch(half,low,first,Tree.read(Tile.Tile8,input,sample_address(context,Sequence.index_value(8n,lane)))),block_mask(context,Sequence.index_value(8n,lane))) : F32}:
  match half low first:
    case True{} True{} True{}: field_lane(~Tile.lane0,input,context,lane)
    case True{} True{} False{}: field_lane(~Tile.lane1,input,context,lane)
    case True{} False{} True{}: field_lane(~Tile.lane2,input,context,lane)
    case True{} False{} False{}: field_lane(~Tile.lane3,input,context,lane)
    case False{} True{} True{}: field_lane(~Tile.lane4,input,context,lane)
    case False{} True{} False{}: field_lane(~Tile.lane5,input,context,lane)
    case False{} False{} True{}: field_lane(~Tile.lane6,input,context,lane)
    case False{} False{} False{}: field_lane(~Tile.lane7,input,context,lane)

def dispatched_lane(+output: U32,+tile: Tile.Tile8) ->
  {CellFields.lane_at(CellFields.bits(output),tile) == Cells.lane_of(tile,(output % 8 : U32)) : F32}:
  +word = (output % 8 : U32)
  +number = U32.to_nat(word)
  +exact = NativeDivision.modulo(output,8,16,{==},{==},{==})
  +small = {%Equal.sym(Nat,number,Nat.mod(U32.to_nat(output),8n),exact) : {Nat.is_lt(_,8n) == True{} : Bool} Radix.remainder_inside(output,8n,{==}) : {Nat.is_lt(number,8n) == True{} : Bool}}
  Equal.trans(F32,CellFields.lane_at(CellFields.bits(output),tile),TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),tile),Cells.lane_of(tile,word),CellFields.lane_bits(output,tile),
    %exact : {TileLaws.select_lane(_,tile) == Cells.lane_of(tile,word) : F32}
    %Arithmetic.native_roundtrip(word) : {TileLaws.select_lane(number,tile) == Cells.lane_of(tile,_) : F32}
    Equal.sym(F32,Cells.lane_of(tile,U32.from_nat(number)),TileLaws.select_lane(number,tile),TiledValues.lane_selected(tile,number,small)))

def vector_lane(+input: Tree.Tree<Tile.Tile8>,+context: Spatial.PixelBlock,+output: U32,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,vector(input,context,output)) == cell_sample(input,context,output,Sequence.index_value(8n,lane)) : F32}:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = context
  +index = Sequence.index_value(8n,lane)
  +word = (output % 8 : U32)
  +tile = Tree.read(Tile.Tile8,input,sample_address(context,index))
  +mask = block_mask(context,index)
  Equal.trans(F32,Tile.element_at(lane,vector(input,context,output)),Masks.masked(CellFields.lane_at(CellFields.bits(output),tile),mask),cell_sample(input,context,output,index),
    selected_lane(U32.is_lt(word,4),U32.is_lt((word % 4 : U32),2),U32.is_zero((word % 2 : U32)),input,context,lane),
    Equal.cong(F32,F32,value => Masks.masked(value,mask),CellFields.lane_at(CellFields.bits(output),tile),Cells.lane_of(tile,word),dispatched_lane(output,tile)))

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

def block(context: FillKernel.Filling,+row: U32,+group: U32,+output: U32,+shift: U32) -> Spatial.PixelBlock:
  FillKernel.Filling{shifts,outputs,+column_low,+kernel,+width,cg,+low,+base,+groups,+first,+rows,+first_group,span} = context

  Spatial.PixelBlock{(row + low - kernel - first : U32),((group + first_group) * 8 + column_low - kernel + shift : U32),(base + output / 8 : U32),rows,width,1,groups}
def query_block(+query: Lookup.Query) -> Spatial.PixelBlock:
  block(Fill.filling(Lookup.fill(query)),Lookup.row(query),Lookup.group(query),Lookup.output(query),Lookup.shift(query))
def wanted_block(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>) ->
  {Fill.wanted(Lookup.fill(query)) == vector(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query)) : Tile.Tile8}:
  Lookup.Query{+fq,+row,+group,+output,+shift} = query
  Fill.Query{+context,+input,rows,+position,+channel,+selected_shift,depth,capacity} = fq
  FillKernel.Filling{shifts,outputs,column_low,kernel,width,cg,low,base,groups,first,source_rows,first_group,+span} = context

  Lookup.Fields{located,+oe,+se,inside} = fields
  +qw = (U32.from_nat(position) / span : U32)
  +gw = (U32.from_nat(position) % span : U32)
  +ow = U32.from_nat(channel)
  +sw = U32.from_nat(selected_shift)
  +out_exact = Equal.trans(U32,ow,U32.from_nat(U32.to_nat(output)),output,Equal.cong(Nat,U32,U32.from_nat,channel,U32.to_nat(output),oe),Arithmetic.native_roundtrip(output))
  +shift_exact = Equal.trans(U32,sw,U32.from_nat(U32.to_nat(shift)),shift,Equal.cong(Nat,U32,U32.from_nat,selected_shift,U32.to_nat(shift),se),Arithmetic.native_roundtrip(shift))
  %Equal.sym(U32,qw,row,quotient_exact(query,fields,facts)) : {vector(input,block(context,_,gw,ow,sw),ow) == vector(input,query_block(query),output) : Tile.Tile8}
  %Equal.sym(U32,gw,group,remainder_exact(query,fields,facts)) : {vector(input,block(context,row,_,ow,sw),ow) == vector(input,query_block(query),output) : Tile.Tile8}
  %Equal.sym(U32,ow,output,out_exact) : {vector(input,block(context,row,group,_,sw),_) == vector(input,query_block(query),output) : Tile.Tile8}
  %Equal.sym(U32,sw,shift,shift_exact) : {vector(input,block(context,row,group,output,_),output) == vector(input,query_block(query),output) : Tile.Tile8}
  {==}
def wanted_lane(+query: Lookup.Query,+fields: Lookup.Fields<query>,+facts: Fill.Facts<Lookup.fill(query)>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Fill.wanted(Lookup.fill(query))) == cell_sample(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query),Sequence.index_value(8n,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,Fill.wanted(Lookup.fill(query))),Tile.element_at(lane,vector(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query))),cell_sample(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),Fill.wanted(Lookup.fill(query)),vector(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query)),wanted_block(query,fields,facts)),vector_lane(Fill.original(Lookup.fill(query)),query_block(query),Lookup.output(query),lane))
def filled_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) ==
    cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)) : F32}:
  +query = Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),Tile.element_at(lane,Fill.wanted(Chunks.slot(reduction))),cell_sample(Fill.original(Chunks.slot(reduction)),query_block(query),Lookup.output(query),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction))),Fill.wanted(Chunks.slot(reduction)),Chunks.readback(reduction,valid)),
    wanted_lane(query,Chunks.lookup_fields(reduction,valid),Fill.frame_facts(Chunks.slot(reduction),Fill.initial(Chunks.slot(reduction)),Chunks.slot_frame(reduction,valid)),lane))

def block_row(value: Spatial.PixelBlock) -> U32:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = value
  row
def block_column(value: Spatial.PixelBlock) -> U32:
  Spatial.PixelBlock{row,column,base,height,width,stride,step} = value
  column
def row_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) ->
  {block_row(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))) ==
    (Chunks.leaf_coordinate(reduction,valid) - Domain.kernel_word(Chunks.geometry(Chunks.query(Chunks.selection(reduction)))) - Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))) : U32) : U32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,+first,target,+source,+row,+group,wb} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Execution.Source{data,base,+source_first,source_rows} = source
  +coordinate = Chunks.leaf_coordinate(reduction,valid)
  +rw = Lookup.read_row_total(Chunks.sample(reduction))
  +start = GeometryChunks.selected_start(grid,first,row,group)
  +rows = GeometryChunks.selected_rows(grid,row,group)
  +low = Kernel.band_low(g,start,(start + rows : U32))
  +restore_coordinate = Equal.trans(U32,(rw + low : U32),((coordinate - low) + low : U32),coordinate,
    Equal.cong(U32,U32,value => (value + low : U32),rw,(coordinate - low : U32),Chunks.row_relative_to_leaf_coordinate(reduction,valid)),Subtraction.add_subtract(coordinate,low))
  Equal.cong(U32,U32,value => (value - k - source_first : U32),(rw + low : U32),coordinate,restore_coordinate)

# The gradient image a reduction reads, the groups of its outputs to a pixel.
def image(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,original: Buffer.Model<Tile.Tile8>,base: U32) -> SourceRows.Slice:
  SourceRows.query_slice(Chunks.leaf_band(reduction,valid),Chunks.source(Chunks.query(Chunks.selection(reduction))),original,base,
    Packing.output_groups(Chunks.geometry(Chunks.query(Chunks.selection(reduction)))))
def global_row(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) -> U32:
  (Chunks.leaf_coordinate(reduction,valid) - Domain.kernel_word(Chunks.geometry(Chunks.query(Chunks.selection(reduction)))) : U32)
def column_at(+reduction: Chunks.Reduction,+lane: Sequence.Index(8n)) -> U32:
  (block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))) + U32.from_nat(Sequence.index_value(8n,lane)) * 1 : U32)
def local_pixel_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+base: U32,+lane: Sequence.Index(8n)) ->
  {cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)) ==
    SourceRows.local_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,first,target,+source,row,group,wb} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,+gw} = g
  Execution.Source{+data,+source_base,+source_first,+source_rows} = source
  Chunks.Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  +query = Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))
  +column = column_at(reduction,lane)
  +output = Lookup.read_output(Chunks.sample(reduction))
  Equal.cong(U32,F32,row => cell_pixel(data,row,column,(source_base + output / 8 : U32),source_rows,gw,Tiled.groups(o),output),block_row(query_block(query)),(global_row(reduction,valid) - source_first : U32),row_matches(reduction,valid))

def global_pixel_matches_from(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+base: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>,
  halo: KernelGemm.Both<{U32.is_le(Kernel.band_low(Bands.geometry(Chunks.leaf_band(reduction,valid)),Bands.first(Chunks.leaf_band(reduction,valid)),Bands.last(Chunks.leaf_band(reduction,valid))),Chunks.leaf_coordinate(reduction,valid)) == True{} : Bool},
    {U32.is_lt(Chunks.leaf_coordinate(reduction,valid),Kernel.band_high(Bands.geometry(Chunks.leaf_band(reduction,valid)),Bands.first(Chunks.leaf_band(reduction,valid)),Bands.last(Chunks.leaf_band(reduction,valid)))) == True{} : Bool}>) ->
  {cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)) ==
    SourceRows.global_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Chunks.Reduction{Chunks.Selection{+q,y,x,yi,xi},line,local_group,unit,output,tr,tc} = reduction
  Chunks.Query{+grid,first,target,+source,row,group,wb} = q
  GeometryChunks.Grid{+call,e,count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{ci,room,fits,enabled,ji,oi,ri,coli} = valid
  KernelGemm.Both{+lo,+hi} = halo
  +slice = image(reduction,valid,original,base)
  +oc = Lookup.read_output(Chunks.sample(reduction))
  +row = global_row(reduction,valid)
  +column = column_at(reduction,lane)
  +outputs = Fill.outputs(Chunks.filling(q))
  +oc_inside = {%Equal.sym(Bool,U32.is_lt(oc,outputs),Nat.is_lt(U32.to_nat(oc),U32.to_nat(outputs)),Comparison.lt(oc,outputs)) : {_ == True{} : Bool} Chunks.output_inside_slot(reduction,valid) : {U32.is_lt(oc,outputs) == True{} : Bool}}
  Equal.trans(F32,cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)),SourceRows.local_pixel(slice,oc,row,column),SourceRows.global_pixel(slice,oc,row,column),local_pixel_matches(reduction,valid,original,base,lane),
    SourceRows.halo_pixel(Chunks.leaf_band(reduction,valid),source,original,base,Tiled.groups(o),outputs,oc,Chunks.leaf_coordinate(reduction,valid),column,range,covers,stored,oc_inside,lo,hi))
def global_pixel_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+base: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)) ==
    SourceRows.global_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  global_pixel_matches_from(reduction,valid,original,base,lane,range,covers,stored,Chunks.leaf_halo_bounds(reduction,valid))

def filled_global_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+base: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,base),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) ==
    SourceRows.global_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),cell_sample(Fill.original(Chunks.slot(reduction)),query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction))),Lookup.read_output(Chunks.sample(reduction)),Sequence.index_value(8n,lane)),SourceRows.global_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)),
    filled_lane(reduction,valid,lane),global_pixel_matches(reduction,valid,original,base,lane,range,covers,stored))

def row_source_phase_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>) ->
  {global_row(reduction,valid) == (Packing.input_row(Jobs.context(Chunks.phase(Chunks.selection(reduction))),Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction))) + U32.from_nat(Chunks.tap_row(reduction)) : U32) : U32}:
  Chunks.Reduction{+selected,+line,local_group,unit,output,+tr,tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,row,group,wb} = q
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
  Chunks.Query{+grid,first,target,+source,row,group,+wb} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Execution.Source{data,base,source_first,source_rows} = source
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
  +tap_equal = Reads.read_tap_column_from(Chunks.work(selected),wb,line,local_group,unit,output,tr,tc)
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
def phase_sample(+reduction: Chunks.Reduction,+image: U32) -> Packing.CellsSample:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(reduction)))
  Packing.CellsSample{(Packing.cells_base(phase,image) + U32.from_nat(Chunks.output(reduction)) / 8 : U32),
    (Packing.input_row(phase,Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction))) + U32.from_nat(Chunks.tap_row(reduction)) : U32),
    (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(Chunks.tap_column(reduction)) : U32),Packing.gradient_rows(phase),Packing.gradient_columns(phase),U32.from_nat(Chunks.output(reduction)),Packing.gradient_groups(phase)}
def sample_raw(+input: Buffer.Model<Tile.Tile8>,context: Packing.CellsSample,+index: Nat) -> F32:
  Packing.CellsSample{base,row,column,height,width,channel,groups} = context
  cell_pixel(Buffer.tree(~Tile.Tile8,input),row,(column + U32.from_nat(index) : U32),base,height,width,groups,channel)

def sample_mask(context: Packing.CellsSample,+index: Nat) -> Bool:
  Packing.CellsSample{base,row,column,height,width,channel,groups} = context
  U32.is_lt(row,height) && U32.is_lt((column + U32.from_nat(index) : U32),width)

def sample_choice_case(+input: Buffer.Model<Tile.Tile8>,+context: Packing.CellsSample,+index: Nat,enabled: Bool,+same: {sample_mask(context,index) == enabled : Bool}) ->
  {sample_raw(input,context,index) == Packing.cells_value(input,context,index) : F32}:
  Packing.CellsSample{+base,+row,+column,+height,+width,+channel,+groups} = context
  match enabled:
    case False{}:
      %Equal.sym(Bool,sample_mask(context,index),False{},same) :
        {Masks.masked(Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + (row * width + (column + U32.from_nat(index))) * groups : U32)),(channel % 8 : U32)),_) == Bool.pick(F32,_,Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + (row * width + (column + U32.from_nat(index))) * groups : U32)),(channel % 8 : U32)),0.0) : F32} {==}
    case True{}:
      %Equal.sym(Bool,sample_mask(context,index),True{},same) :
        {Masks.masked(Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + (row * width + (column + U32.from_nat(index))) * groups : U32)),(channel % 8 : U32)),_) == Bool.pick(F32,_,Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + (row * width + (column + U32.from_nat(index))) * groups : U32)),(channel % 8 : U32)),0.0) : F32} {==}

def sample_choice(+input: Buffer.Model<Tile.Tile8>,+context: Packing.CellsSample,+index: Nat) -> {sample_raw(input,context,index) == Packing.cells_value(input,context,index) : F32}:
  sample_choice_case(input,context,index,sample_mask(context,index),{==})
def source_pixel_phase_matches(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+image_id: U32,+lane: Sequence.Index(8n)) ->
  {SourceRows.global_pixel(image(reduction,valid,original,Packing.cells_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id)),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane)) ==
    Packing.cells_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)) : F32}:
  Chunks.Reduction{+selected,+line,+local_group,+unit,+output,+tr,+tc} = reduction
  Chunks.Selection{+q,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,row,group,+wb} = q
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = g
  Chunks.Valid{ci,room,fits,+enabled,ji,+oi,+ri,+coli} = valid
  +phase = Jobs.context(Chunks.phase(selected))
  +step = Packing.gradient_groups(phase)
  +base = Packing.cells_base(phase,image_id)
  +out_word = Lookup.read_output(Chunks.sample(reduction))
  +out_equal = Reads.read_output_from(Chunks.work(selected),wb,line,local_group,unit,output,tr,tc)
  +ln = Sequence.index_value(8n,lane)
  +lw = U32.from_nat(ln)
  +lane_unit = Equal.trans(U32,(lw * 1 : U32),U32.from_nat(Nat.mul(ln,1n)),lw,Multiplication.native_encoded_multiply(ln,1n),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(ln,1n),ln,Order.times_one(ln)))
  +wanted_row = (Packing.input_row(phase,Jobs.row_native(Chunks.work(selected),line)) + U32.from_nat(tr) : U32)
  +wanted_column = (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) + lw : U32)
  +column_equal = Equal.trans(U32,column_at(reduction,lane),((Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc)) + lw * 1 : U32),wanted_column,
    Equal.cong(U32,U32,value => (value + lw * 1 : U32),block_column(query_block(Lookup.query_for_read(Chunks.slot(reduction),Chunks.sample(reduction)))),(Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) : U32),column_source_phase_matches(reduction,valid)),
    Equal.cong(U32,U32,value => (Packing.input_column(phase,input_column(reduction)) + U32.from_nat(tc) + value : U32),(lw * 1 : U32),lw,lane_unit))
  %Equal.sym(U32,global_row(reduction,valid),wanted_row,row_source_phase_matches(reduction,valid)) : {cell_pixel(Buffer.tree(~Tile.Tile8,original),_,column_at(reduction,lane),(base + out_word / 8 : U32),gh,gw,step,out_word) == Packing.cells_value(original,phase_sample(reduction,image_id),ln) : F32}
  %Equal.sym(U32,column_at(reduction,lane),wanted_column,column_equal) : {cell_pixel(Buffer.tree(~Tile.Tile8,original),wanted_row,_,(base + out_word / 8 : U32),gh,gw,step,out_word) == Packing.cells_value(original,phase_sample(reduction,image_id),ln) : F32}
  %Equal.sym(U32,out_word,U32.from_nat(output),out_equal) : {cell_pixel(Buffer.tree(~Tile.Tile8,original),wanted_row,wanted_column,(base + _ / 8 : U32),gh,gw,step,_) == Packing.cells_value(original,phase_sample(reduction,image_id),ln) : F32}
  sample_choice(original,phase_sample(reduction,image_id),ln)

def phase_input_tile(+reduction: Chunks.Reduction,+image_id: U32,original: Buffer.Model<Tile.Tile8>) -> Tile.Tile8:
  Packing.cells_input_tile(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id,Jobs.row_native(Chunks.work(Chunks.selection(reduction)),Chunks.line(reduction)),input_column(reduction),original,Chunks.output(reduction),Chunks.tap_row(reduction),Chunks.tap_column(reduction))
def phase_tile_matches(+reduction: Chunks.Reduction,+image_id: U32,+original: Buffer.Model<Tile.Tile8>) -> {phase_input_tile(reduction,image_id,original) == Packing.cells_tile(original,phase_sample(reduction,image_id)) : Tile.Tile8}:
  {==}
def phase_tile_lane(+reduction: Chunks.Reduction,+image_id: U32,+original: Buffer.Model<Tile.Tile8>,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,phase_input_tile(reduction,image_id,original)) == Packing.cells_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)) : F32}:
  Equal.trans(F32,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),Tile.element_at(lane,Packing.cells_tile(original,phase_sample(reduction,image_id))),Packing.cells_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane)),
    Equal.cong(Tile.Tile8,F32,Tile.element_at(lane),phase_input_tile(reduction,image_id,original),Packing.cells_tile(original,phase_sample(reduction,image_id)),phase_tile_matches(reduction,image_id,original)),Packing.cells_tile_index(original,phase_sample(reduction,image_id),lane))
def input_lane(+reduction: Chunks.Reduction,+valid: Chunks.Valid<reduction>,+original: Buffer.Model<Tile.Tile8>,+image_id: U32,+lane: Sequence.Index(8n),
  +range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(reduction))))>,
  covers: SourceRows.Covers<Chunks.leaf_band(reduction,valid),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(reduction)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(reduction))))>,
  stored: SourceRows.Stored<image(reduction,valid,original,Packing.cells_base(Jobs.context(Chunks.phase(Chunks.selection(reduction))),image_id)),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(reduction))))>) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))) == Tile.element_at(lane,phase_input_tile(reduction,image_id,original)) : F32}:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(reduction)))
  +base = Packing.cells_base(phase,image_id)
  +wanted = SourceRows.global_pixel(image(reduction,valid,original,base),Lookup.read_output(Chunks.sample(reduction)),global_row(reduction,valid),column_at(reduction,lane))
  +sample = Packing.cells_value(original,phase_sample(reduction,image_id),Sequence.index_value(8n,lane))
  Equal.trans(F32,Tile.element_at(lane,Tree.read(Tile.Tile8,Filling.tiles(Chunks.filled(reduction)),Lookup.read_address(Chunks.sample(reduction)))),wanted,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),filled_global_lane(reduction,valid,original,base,lane,range,covers,stored),
    Equal.trans(F32,wanted,sample,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),source_pixel_phase_matches(reduction,valid,original,image_id,lane),Equal.sym(F32,Tile.element_at(lane,phase_input_tile(reduction,image_id,original)),sample,phase_tile_lane(reduction,image_id,original,lane))))
