# Natural allocator extent and initial readback frames for actual chunks.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_capacity.bend as Capacity
import ./row_major_index.bend as RowMajor
import ./input_gradient_tiled_chunks.bend as Partition
import ./input_gradient_tiled_chunk_geometry.bend as Chunks
import ./input_gradient_tiled_fill.bend as Fill
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_refinement.bend as Filling
import ./storage_tree_model.bend as Tree
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../tensor.bend as Tensors
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_extrema.bend as NativeExtrema

# The quantity passed to the allocator is the complete natural tile extent.
def tiled_size(+geometry: Kernel.Geometry,+first: U32,+count: U32) -> Nat:
  Nat.mul(Nat.mul(U32.to_nat(Kernel.band_rows(geometry,first,count)),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry)))),Capacity.block_size(geometry))
def native_size_exact(+native: Capacity.Native,+rows: U32,+groups: U32,+block: Nat,
  +groups_exact: {U32.to_nat(groups) == Capacity.chunk_groups(Capacity.model(native)) : Nat},
  +block_exact: {Capacity.block(Capacity.model(native)) == block : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),Capacity.row_span(Capacity.model(native))),Capacity.workspace(Capacity.model(native))) == True{} : Bool}) ->
  {U32.to_nat(Capacity.tile_product(native,rows)) == Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(groups)),block) : Nat}:
  +rn = U32.to_nat(rows)
  +gn = Capacity.chunk_groups(Capacity.model(native))
  +bn = Capacity.block(Capacity.model(native))
  Equal.trans(Nat,U32.to_nat(Capacity.tile_product(native,rows)),Nat.mul(rn,Capacity.row_span(Capacity.model(native))),Nat.mul(Nat.mul(rn,U32.to_nat(groups)),block),Capacity.tile_product_exact(native,rows,room),
    Equal.trans(Nat,Nat.mul(rn,Capacity.row_span(Capacity.model(native))),Nat.mul(Nat.mul(rn,gn),bn),Nat.mul(Nat.mul(rn,U32.to_nat(groups)),block),Equal.sym(Nat,Nat.mul(Nat.mul(rn,gn),bn),Nat.mul(rn,Capacity.row_span(Capacity.model(native))),RowMajor.assoc(rn,gn,bn)),
      %block_exact : {Nat.mul(Nat.mul(rn,gn),bn) == Nat.mul(Nat.mul(rn,U32.to_nat(groups)),_) : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(rn,value),bn),gn,U32.to_nat(groups),Equal.sym(Nat,U32.to_nat(groups),gn,groups_exact))))
def tiled_block_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Capacity.block(Capacity.model(Capacity.tiled_native(call,envelope,fits))) == Capacity.block_size(Domain.geometry(call)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def tiled_count_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +rows_bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}) ->
  {U32.to_nat(Capacity.tiled_count(Domain.geometry(call),first,count)) == tiled_size(Domain.geometry(call),first,count) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  +native = Capacity.tiled_native(call,envelope,fits)
  +rows = Kernel.band_rows(geometry,first,count)
  +cg = Capacity.layout_chunk_groups(Kernel.tiling(geometry))
  +gn = Capacity.chunk_groups(Capacity.model(native))
  +exact = Equal.trans(Nat,U32.to_nat(cg),U32.to_nat(Capacity.native_groups(native)),gn,
    Equal.cong(U32,Nat,U32.to_nat,cg,Capacity.native_groups(native),Capacity.tiled_groups_matches(call,envelope,fits)),Capacity.groups_exact(native))
  Equal.trans(Nat,U32.to_nat(Capacity.tiled_count(geometry,first,count)),U32.to_nat(Capacity.tile_product(native,rows)),tiled_size(geometry,first,count),
    Equal.cong(U32,Nat,U32.to_nat,Capacity.tiled_count(geometry,first,count),Capacity.tile_product(native,rows),Capacity.tiled_product_matches(call,envelope,fits,first,count)),
    native_size_exact(native,rows,cg,Capacity.block_size(geometry),exact,tiled_block_matches(call,envelope,fits),Capacity.contained_tile_room(call,envelope,fits,first,count,contained,rows_bounded)))


def slot_query(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat) -> Fill.Query:
  +geometry = Chunks.grid_geometry(grid)
  +start = Chunks.selected_start(grid,first,row,group)
  +count = Chunks.selected_rows(grid,row,group)
  Fill.allocated_query(Chunks.selected_filling(grid,first,row,group,source),Execution.source_data(source),U32.to_nat(Kernel.band_rows(geometry,start,count)),position,output,shift,Capacity.tiled_count(geometry,start,count))
def slot_size_matches(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat) ->
  {tiled_size(Chunks.grid_geometry(grid),Chunks.selected_start(grid,first,row,group),Chunks.selected_rows(grid,row,group)) == Fill.tile_count(slot_query(grid,first,row,group,source,position,output,shift)) : Nat}:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,plane,source_first,source_rows} = source
  {==}
def slot_span_ordered(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source) ->
  {Nat.is_le(U32.to_nat(Fill.span(Chunks.selected_filling(grid,first,row,group,source))),U32.to_nat(Fill.groups(Chunks.selected_filling(grid,first,row,group,source)))) == True{} : Bool}:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,plane,source_first,source_rows} = source
  Bounds.as_nat_le(Chunks.selected_groups(grid,row,group),Capacity.layout_chunk_groups(Kernel.tiling(g)),NativeExtrema.minimum_left(Capacity.layout_chunk_groups(Kernel.tiling(g)),Chunks.group_remainder(grid,row,group)))
def slot_count_exact(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat,
  +inside: Partition.Inside<Chunks.query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(Chunks.grid_count(grid))),Domain.height(Chunks.grid_geometry(grid))) == True{} : Bool},
  +fits: {Kernel.fits(Chunks.grid_geometry(grid),Machine.capacity(Chunks.grid_limits(grid))) == True{} : Bool}) ->
  {U32.to_nat(Capacity.tiled_count(Chunks.grid_geometry(grid),Chunks.selected_start(grid,first,row,group),Chunks.selected_rows(grid,row,group))) == Fill.tile_count(slot_query(grid,first,row,group,source,position,output,shift)) : Nat}:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  +start = Chunks.selected_start(grid,first,row,group)
  +count = Chunks.selected_rows(grid,row,group)
  Equal.trans(Nat,U32.to_nat(Capacity.tiled_count(Domain.geometry(call),start,count)),tiled_size(Domain.geometry(call),start,count),Fill.tile_count(slot_query(grid,first,row,group,source,position,output,shift)),
    tiled_count_exact(call,e,fits,start,count,Chunks.selected_band_contains(grid,first,row,group,inside,contained),Chunks.selected_rows_bounded(grid,row,group)),slot_size_matches(grid,first,row,group,source,position,output,shift))

def slot_frame(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat,
  +inside: Partition.Inside<Chunks.query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(Chunks.grid_count(grid))),Domain.height(Chunks.grid_geometry(grid))) == True{} : Bool},
  +fits: {Kernel.fits(Chunks.grid_geometry(grid),Machine.capacity(Chunks.grid_limits(grid))) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Fill.count(slot_query(grid,first,row,group,source,position,output,shift))) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,U32.to_nat(Fill.outputs(Chunks.selected_filling(grid,first,row,group,source)))) == True{} : Bool},
  +shift_inside: {Nat.is_lt(shift,U32.to_nat(Fill.shifts(Chunks.selected_filling(grid,first,row,group,source)))) == True{} : Bool}) ->
  Fill.Frame<slot_query(grid,first,row,group,source,position,output,shift),Fill.initial(slot_query(grid,first,row,group,source,position,output,shift))>:
  +g = Chunks.grid_geometry(grid)
  +start = Chunks.selected_start(grid,first,row,group)
  +count = Chunks.selected_rows(grid,row,group)
  +filling = Chunks.selected_filling(grid,first,row,group,source)
  Fill.allocated_frame(filling,Execution.source_data(source),U32.to_nat(Kernel.band_rows(g,start,count)),position,output,shift,Capacity.tiled_count(g,start,count),slot_count_exact(grid,first,row,group,source,position,output,shift,inside,contained,fits),Chunks.selected_guarded(grid,first,row,group,inside,contained,fits),slot_span_ordered(grid,first,row,group,source),position_inside,output_inside,shift_inside)

def fresh_matches(+shape: Execution.ChunkShape,+quantity: U32,+same: {Execution.shape_tiles(shape) == quantity : U32}) ->
  {Execution.fresh_tiles(shape) == Tree.fresh(Tile.Tile8,Tensors.depth_for((quantity + 1 : U32)),Tile.filled(0.0)) : Tree.Tree<Tile.Tile8>}:
  Equal.cong(U32,Tree.Tree<Tile.Tile8>,count => Tree.fresh(Tile.Tile8,Tensors.depth_for((count + 1 : U32)),Tile.filled(0.0)),Execution.shape_tiles(shape),quantity,same)
def initial_matches_shape(+grid: Chunks.Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat) ->
  {Fill.initial(slot_query(grid,first,row,group,source,position,output,shift)) ==
    Filling.Owners{Execution.source_data(source),Execution.fresh_tiles(Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source))} : Filling.Owners}:
  +g = Chunks.grid_geometry(grid)
  +start = Chunks.selected_start(grid,first,row,group)
  +count = Chunks.selected_rows(grid,row,group)
  +quantity = Capacity.tiled_count(g,start,count)
  +shape = Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{g,Kernel.tiling(g)},first,Chunks.grid_count(grid),target,plane},source)
  Equal.cong(Tree.Tree<Tile.Tile8>,Filling.Owners,tiles => Filling.Owners{Execution.source_data(source),tiles},Tree.fresh(Tile.Tile8,Tensors.depth_for((quantity + 1 : U32)),Tile.filled(0.0)),Execution.fresh_tiles(shape),
    Equal.sym(Tree.Tree<Tile.Tile8>,Execution.fresh_tiles(shape),Tree.fresh(Tile.Tile8,Tensors.depth_for((quantity + 1 : U32)),Tile.filled(0.0)),fresh_matches(shape,quantity,Chunks.selected_shape_tiles_matches(grid,first,target,plane,row,group,source))))

def filling_count(+grid: Chunks.Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source) -> U32:
  (Kernel.band_rows(Chunks.grid_geometry(grid),Chunks.selected_start(grid,first,row,group),Chunks.selected_rows(grid,row,group)) * Fill.span(Chunks.selected_filling(grid,first,row,group,source)) : U32)
def shape_count_matches(+grid: Chunks.Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source) ->
  {Execution.shape_count(Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source)) == U32.to_nat(filling_count(grid,first,row,group,source)) : Nat}:
  Chunks.Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  +start = Chunks.selected_start(grid,first,row,group)
  Equal.cong(U32,Nat,value => U32.to_nat((Kernel.band_rows(g,start,value) * Chunks.selected_groups(grid,row,group) : U32)),Chunks.physical_rows(grid,first,row,group),Chunks.selected_rows(grid,row,group),Chunks.physical_rows_matches(grid,first,row,group))
def shape_count_exact(+grid: Chunks.Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat,
  +facts: Fill.Facts<slot_query(grid,first,row,group,source,position,output,shift)>) ->
  {Execution.shape_count(Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source)) == Fill.count(slot_query(grid,first,row,group,source,position,output,shift)) : Nat}:
  +query = slot_query(grid,first,row,group,source,position,output,shift)
  +shape = Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source)
  Equal.trans(Nat,Execution.shape_count(shape),U32.to_nat(filling_count(grid,first,row,group,source)),Fill.count(query),shape_count_matches(grid,first,target,plane,row,group,source),Fill.count_exact(query,Kernel.band_rows(Chunks.grid_geometry(grid),Chunks.selected_start(grid,first,row,group),Chunks.selected_rows(grid,row,group)),{==},facts))

def fill_matches_slot(+grid: Chunks.Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source,+position: Nat,+output: Nat,+shift: Nat,
  +facts: Fill.Facts<slot_query(grid,first,row,group,source,position,output,shift)>) ->
  {Execution.fill_chunk(Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source),source) ==
    Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(slot_query(grid,first,row,group,source,position,output,shift)),0,Fill.filling(slot_query(grid,first,row,group,source,position,output,shift)),Fill.initial(slot_query(grid,first,row,group,source,position,output,shift))) : Filling.Owners}:
  +query = slot_query(grid,first,row,group,source,position,output,shift)
  +shape = Execution.chunk_shape(Chunks.native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{Chunks.grid_geometry(grid),Kernel.tiling(Chunks.grid_geometry(grid))},first,Chunks.grid_count(grid),target,plane},source)
  +initial = {Filling.Owners{Execution.source_data(source),Execution.fresh_tiles(shape)} : Filling.Owners}
  +count_same = shape_count_exact(grid,first,target,plane,row,group,source,position,output,shift,facts)
  +context_same = Chunks.selected_filling_matches(grid,first,target,plane,row,group,source)
  +initial_same = initial_matches_shape(grid,first,target,plane,row,group,source,position,output,shift)
  Equal.trans(Filling.Owners,Execution.fill_chunk(shape,source),Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Execution.shape_filling(shape),initial),Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Fill.filling(query),Fill.initial(query)),
    Equal.cong(Nat,Filling.Owners,count => Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,count,0,Execution.shape_filling(shape),initial),Execution.shape_count(shape),Fill.count(query),count_same),
    Equal.trans(Filling.Owners,Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Execution.shape_filling(shape),initial),Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Fill.filling(query),initial),Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Fill.filling(query),Fill.initial(query)),
      Equal.cong(Kernel.Filling,Filling.Owners,context => Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,context,initial),Execution.shape_filling(shape),Fill.filling(query),context_same),
      Equal.cong(Filling.Owners,Filling.Owners,initial => Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,Fill.count(query),0,Fill.filling(query),initial),initial,Fill.initial(query),Equal.sym(Filling.Owners,Fill.initial(query),initial,initial_same))))
