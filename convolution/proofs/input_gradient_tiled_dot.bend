# Lane r of the tile row j of a phase job holds the specification value of
# its channel and pixel: its ordered dot over the job's taps is the
# specification's sum in tap order, then + 0.0.
import Base
import ../../lib/proofs/input_gradient_tiled_source_witness.bend as Witness
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/kernel_gemm_spec.bend as GemmSpec
import ../../lib/proofs/kernel_gemm_jobs.bend as GemmJobs
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_value_reads.bend as Input
import ../../lib/proofs/input_gradient_tiled_source_rows.bend as SourceRows
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/kernel_input_gradient_fill_refinement.bend as Filling
import ../../lib/proofs/input_gradient_tiled_entry_reads.bend as Lookup
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_weights.bend as PackedWeights
import ../../lib/proofs/input_gradient_tiled_job_reads.bend as Reads
import ../../lib/proofs/input_gradient_tiled_store_geometry.bend as Channels
import ../../lib/proofs/row_major_coordinates.bend as Coordinates
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_reduction_indices.bend as Indices
import ../../lib/proofs/input_gradient_tiled_band_extents.bend as Bands
import ../../lib/proofs/input_gradient_phase_geometry.bend as PhaseGeometry
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/nat_order.bend as NatOrder
import ../../lib/ordered_matrix.bend as Matrix
import ./input_gradient_cell_sources.bend as Sources
import ./input_gradient_order.bend as Order
import ./input_gradient_tiled_products.bend as Products
import ./matrix_matches_convolution.bend as MatrixLaws
import ../specification.bend as Specification
import ../input_gradient_specification.bend as Strided

# A cell: the job of a seed, its channel and position, and the source rows it reads.
type Point is Data:
  Point{seed: Chunks.Reduction,valid: Chunks.Valid<seed>,bases: Runtime.Bases<Chunks.selection(seed)>,image: U32,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,channel: Nat,position: Nat,
    channel_small: {Nat.is_lt(channel,8n) == True{} : Bool},
    position_small: {Nat.is_lt(position,8n) == True{} : Bool},
    column_inside: {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(seed)),position),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(seed))))))) == True{} : Bool},
    channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(Products.channel_base(seed)),channel),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool},
    range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(seed)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(seed)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(seed))))>}
def seed(point: Point) -> Chunks.Reduction:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  seed
def valid(point: Point) -> Chunks.Valid<seed(point)>:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  valid
def image(point: Point) -> U32:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  image
def gradient(point: Point) -> Buffer.Model<Tile.Tile8>:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  gradient
def weights(point: Point) -> Buffer.Model<Tile.Tile8>:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  weights
def channel(point: Point) -> Nat:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  channel
def position(point: Point) -> Nat:
  Point{seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  position
def count(+point: Point) -> Nat:
  Runtime.count(seed(point))
def flat(+point: Point) -> Order.Flat:
  Products.flat(channel(point),position(point),seed(point),image(point),gradient(point),weights(point))
def selected(+point: Point,index: Nat) -> Chunks.Reduction:
  Indices.at(seed(point),index)
def selected_valid(+point: Point,+index: Nat,+inside: {Nat.is_lt(index,count(point)) == True{} : Bool}) -> Chunks.Valid<selected(point,index)>:
  Indices.valid_at(seed(point),valid(point),index,inside)

# ---- The source rows ----

def Cover(+point: Point,+index: Nat,+inside: {Nat.is_lt(index,count(point)) == True{} : Bool}) -> Type:
  SourceRows.Covers<Chunks.leaf_band(selected(point,index),selected_valid(point,index,inside)),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(seed(point))))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(seed(point)))))>
def Stored(+point: Point,+index: Nat,+inside: {Nat.is_lt(index,count(point)) == True{} : Bool}) -> Type:
  SourceRows.Stored<Input.image(selected(point,index),selected_valid(point,index,inside),gradient(point),Packing.cells_base(Jobs.context(Chunks.phase(Chunks.selection(seed(point)))),image(point))),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(seed(point)))))>
def source_outputs(+point: Point) -> U32:
  Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(seed(point)))))

# The gradient a seed's cells read: the source rows that hold its band, of
# an image of the batch.
def SeedRows(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+image: U32,+gradient: Buffer.Model<Tile.Tile8>) -> Data:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed)))
  +query = Chunks.query(Chunks.selection(seed))
  Witness.Witness<Chunks.leaf_band(seed,valid),Chunks.source(query),gradient,Packing.cells_base(phase,image),Packing.output_groups(Chunks.geometry(query)),Fill.outputs(Chunks.filling(query))>
def SeedSource(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+image: U32,+gradient: Buffer.Model<Tile.Tile8>) -> Data:
  Gemm.Both<SeedRows(seed,valid,image,gradient),Products.ImageInside(Chunks.query(Chunks.selection(seed)),image)>
def Source(+point: Point) -> Data:
  SeedSource(seed(point),valid(point),image(point),gradient(point))
def source_rows(+point: Point,+witness: Source(point)) -> SeedRows(seed(point),valid(point),image(point),gradient(point)):
  Shape.first(SeedRows(seed(point),valid(point),image(point),gradient(point)),Products.ImageInside(Chunks.query(Chunks.selection(seed(point))),image(point)),witness)
def source_image(+point: Point,+witness: Source(point)) -> Products.ImageInside(Chunks.query(Chunks.selection(seed(point))),image(point)):
  Shape.second(SeedRows(seed(point),valid(point),image(point),gradient(point)),Products.ImageInside(Chunks.query(Chunks.selection(seed(point))),image(point)),witness)
type Whole is Data:
  Whole{point: Point,source: Source(point)}
def whole_point(frame: Whole) -> Point:
  Whole{point,source} = frame
  point
def source_cover(+point: Point,+witness: Source(point),+index: Nat,+inside: {Nat.is_lt(index,count(point)) == True{} : Bool}) -> Cover(point,index,inside):
  Point{+seed,+valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  +band = Chunks.leaf_band(seed,valid)
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed)))
  +source = Chunks.source(Chunks.query(Chunks.selection(seed)))
  %Equal.sym(Bands.Query,Chunks.leaf_band(Indices.at(seed,index),Indices.valid_at(seed,valid,index,inside)),band,Indices.leaf_band_at(seed,valid,index,inside)) : SourceRows.Covers<_,Execution.source_first(source),Execution.source_rows(source)>
  Witness.cover(band,source,gradient,Packing.cells_base(phase,image),Packing.output_groups(Chunks.geometry(Chunks.query(Chunks.selection(seed)))),source_outputs(point),source_rows(point,witness))
def source_stored(+point: Point,+witness: Source(point),+index: Nat,+inside: {Nat.is_lt(index,count(point)) == True{} : Bool}) -> Stored(point,index,inside):
  Point{+seed,+valid,bases,+image,+gradient,weights,channel,position,rs,li,ci,chi,range} = point
  Chunks.Reduction{+selection,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selection
  Chunks.Query{+grid,first,target,source0,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,leaf_count,grid_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +band = Chunks.leaf_band(seed,valid)
  +selected = Indices.at(seed,index)
  +source = Chunks.source(query)
  +phase = Jobs.context(Chunks.phase(selection))
  +base = Packing.cells_base(phase,image)
  +groups = Packing.output_groups(g)
  %Equal.sym(Bands.Query,Chunks.leaf_band(selected,Indices.valid_at(seed,valid,index,inside)),band,Indices.leaf_band_at(seed,valid,index,inside)) : SourceRows.Stored<SourceRows.query_slice(_,source,gradient,base,groups),source_outputs(point)>
  Witness.stored(band,source,gradient,base,groups,source_outputs(point),source_rows(point,witness))

# ---- The taps of a cell ----

# The trees the job reads: the filled tiles of the chunk and the packed
# weights of every phase.
def tiles(+point: Point) -> Tree.Tree<Tile.Tile8>:
  Filling.tiles(Chunks.filled(seed(point)))
def weight_tree(+point: Point) -> Tree.Tree<Tile.Tile8>:
  Packed.weight_tree(Weights.full_state(Chunks.phase(Chunks.selection(seed(point))),weights(point)))
def gemm_source(+point: Point) -> GemmSpec.Source:
  GemmSpec.Source{tiles(point),weight_tree(point)}
def key(+point: Point) -> Gemm.Both<Nat,GemmSpec.Grid>:
  Runtime.key(seed(point))

def flat_input() -> Gemm.Both<Order.Flat,Unit> -> Unit -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Specification.Product,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_input)
def flat_weight() -> Gemm.Both<Order.Flat,Unit> -> Unit -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Specification.Product,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_weight)

def steps_count(frame: Whole) -> Nat:
  Whole{point,source} = frame
  count(point)

# Step t of the cell reads the pair the specification multiplies at tap t.
def StepAgrees(+frame: Whole,+step: Nat) -> Data:
  Whole{+point,source} = frame
  Gemm.Both<
    {GemmSpec.gemm_weight()(Gemm.Both{key(point),gemm_source(point)},position(point),step) == flat_input()(Gemm.Both{flat(point),Unit{}},Unit{},step) : F32},
    {GemmSpec.gemm_input()(Gemm.Both{key(point),gemm_source(point)},channel(point),step) == flat_weight()(Gemm.Both{flat(point),Unit{}},Unit{},step) : F32}>

def step_at(+point: Point,+witness: Source(point),+step: Nat,+within: {Nat.is_lt(step,count(point)) == True{} : Bool}) ->
  Gemm.Both<
    {GemmSpec.gemm_weight()(Gemm.Both{key(point),gemm_source(point)},position(point),Runtime.tap(selected(point,step))) == flat_input()(Gemm.Both{flat(point),Unit{}},Unit{},Runtime.tap(selected(point,step))) : F32},
    {GemmSpec.gemm_input()(Gemm.Both{key(point),gemm_source(point)},channel(point),Runtime.tap(selected(point,step))) == flat_weight()(Gemm.Both{flat(point),Unit{}},Unit{},Runtime.tap(selected(point,step))) : F32}>:
  Point{+seed,+valid,+bases,+image,+gradient,+weights,+channel,+position,+rs,+li,+ci,+chi,+range} = point
  Chunks.Reduction{+selection,+line,+group,+unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selection
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,+wb} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{vci,vroom,vfits,+venabled,vji,voi,vri,vcoli} = valid
  +at = Indices.at(seed,step)

  +at_valid = Indices.valid_at(seed,valid,step,within)
  +tap = Runtime.tap(at)
  +phase = Chunks.phase(selection)
  +context = Jobs.context(phase)
  +chunk = Chunks.chunk(query)
  +index = Runtime.job_index(seed)
  +plan = Runtime.plan(seed)
  +product = Products.product(channel,position,at,weights)
  +wanted = Order.flat_product(flat(point),tap)
  +spec = Products.product_spec(channel,rs,position,at,at_valid,image,gradient,weights,li,source_image(point,witness),ci,chi,range,source_cover(point,witness,step,within),source_stored(point,witness,step,within))
  +full = Weights.full_state(phase,weights)
  +packed = Weights.packed_state(phase,weights,unit,tap)
  +query = Weights.packed_query(phase,weights,unit,tap)
  +bounded
 = Indices.bounded(phase,venabled,step,within)
  +rows = Indices.rows(phase)
  +columns = Indices.columns(phase)
  +value = Indices.values(phase,step)
  +outputs = Domain.outputs(Jobs.geometry(phase))
  +entry = Runtime.entry_step(context,chunk,wb,index,Chunks.output(at),Chunks.tap_row(at),Chunks.tap_column(at),
    Coordinates.row_below(outputs,rows,columns,value,bounded),Coordinates.column_below(outputs,rows,columns,value,bounded))
  +slot = Equal.trans(U32,GemmSpec.offset_word(U32.to_nat(Runtime.plan_weight(plan,index)),tap),(Runtime.plan_weight(plan,index) + U32.from_nat(tap) : U32),Packed.address(query),
    Runtime.weight_step(Runtime.plan_weight(plan,index),tap),Runtime.weight_slot(seed,valid,bases,weights,tap))
  Gemm.Both{
    Equal.trans(F32,TileLaws.select_lane(position,Tree.read(Tile.Tile8,tiles(point),GemmSpec.grid_read(Lookup.job_grid(plan,index),tap))),MatrixLaws.product_input(Unit{},Unit{},product),MatrixLaws.product_input(Unit{},Unit{},wanted),
      Equal.cong(U32,F32,address => TileLaws.select_lane(position,Tree.read(Tile.Tile8,tiles(point),address)),GemmSpec.grid_read(Lookup.job_grid(plan,index),tap),Lookup.read_address(Chunks.sample(at)),entry),
      Equal.cong(Specification.Product,F32,value => MatrixLaws.product_input(Unit{},Unit{},value),product,wanted,spec)),
    Equal.trans(F32,TileLaws.select_lane(channel,Tree.read(Tile.Tile8,weight_tree(point),GemmSpec.offset_word(U32.to_nat(Runtime.plan_weight(plan,index)),tap))),MatrixLaws.product_weight(Unit{},Unit{},product),MatrixLaws.product_weight(Unit{},Unit{},wanted),
      Equal.trans(F32,TileLaws.select_lane(channel,Tree.read(Tile.Tile8,weight_tree(point),GemmSpec.offset_word(U32.to_nat(Runtime.plan_weight(plan,index)),tap))),
        TileLaws.select_lane(channel,Tree.read(Tile.Tile8,Packed.weight_tree(full),Packed.address(query))),MatrixLaws.product_weight(Unit{},Unit{},product),
        Equal.cong(U32,F32,address => TileLaws.select_lane(channel,Tree.read(Tile.Tile8,weight_tree(point),address)),GemmSpec.offset_word(U32.to_nat(Runtime.plan_weight(plan,index)),tap),Packed.address(query),slot),
        Equal.cong(PackedWeights.Packed,F32,state => TileLaws.select_lane(channel,Tree.read(Tile.Tile8,Packed.weight_tree(state),Packed.address(query))),full,packed,
          Equal.sym(PackedWeights.Packed,packed,full,Weights.full_state_matches(phase,weights,unit,tap)))),
      Equal.cong(Specification.Product,F32,value => MatrixLaws.product_weight(Unit{},Unit{},value),product,wanted,spec))}

def step_agrees(+frame: Whole,+step: Nat,+within: {Nat.is_lt(step,steps_count(frame)) == True{} : Bool}) -> StepAgrees(frame,step):
  Whole{+point,+witness} = frame
  +same = Indices.tap_exact(seed(point),valid(point),step,within)
  %Equal.sym(Nat,step,Runtime.tap(selected(point,step)),same) :
    Gemm.Both<
      {GemmSpec.gemm_weight()(Gemm.Both{key(point),gemm_source(point)},position(point),_) == flat_input()(Gemm.Both{flat(point),Unit{}},Unit{},_) : F32},
      {GemmSpec.gemm_input()(Gemm.Both{key(point),gemm_source(point)},channel(point),_) == flat_weight()(Gemm.Both{flat(point),Unit{}},Unit{},_) : F32}>
  step_at(point,witness,step,within)

# ---- The value of a cell ----

def keys(+point: Point) -> List<&2,ReductionLaws.AddressPair>:
  Runtime.keyed(count(point),key(point))

# The ordered dot of a cell's taps is the specification's sum in tap order.
def cell_dot(+frame: Whole) ->
  {GemmSpec.dot(GemmLaws.RowFirst{},keys(whole_point(frame)),position(whole_point(frame)),channel(whole_point(frame)),gemm_source(whole_point(frame)),0.0) ==
    Specification.sum(Specification.multiply(Order.flat_range(flat(whole_point(frame)),count(whole_point(frame)),0n)),0.0) : F32}:
  Whole{+point,witness} = frame
  +indices = Ranges.range(count(point),0n)
  +left = {Gemm.Both{key(point),gemm_source(point)} : Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>}
  +right = {Gemm.Both{flat(point),Unit{}} : Gemm.Both<Order.Flat,Unit>}
  +products = Order.flat_range(flat(point),count(point),0n)
  +mapped = Matrix.dot(~Nat,~Nat,~Nat,~Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>,~GemmSpec.gemm_weight(),~GemmSpec.gemm_input(),indices,channel(point),position(point),left,0.0)
  +flat_dot = Matrix.dot(~Unit,~Nat,~Unit,~Gemm.Both<Order.Flat,Unit>,~flat_input(),~flat_weight(),indices,Unit{},Unit{},right,0.0)
  +product_dot = Matrix.dot(~Unit,~Specification.Product,~Unit,~Unit,~MatrixLaws.product_input,~MatrixLaws.product_weight,products,Unit{},Unit{},Unit{},0.0)
  Equal.trans(F32,GemmSpec.dot(GemmLaws.RowFirst{},keys(point),position(point),channel(point),gemm_source(point),0.0),mapped,Specification.sum(Specification.multiply(products),0.0),
    Ordered.dot_reindexed(~Nat,~ReductionLaws.AddressPair,~Nat,~Nat,~GemmSpec.Source,~Gemm.Both<Nat,GemmSpec.Grid>,
      ~GemmSpec.both_key(~Nat,~Nat,~GemmSpec.Grid,~GemmSpec.offset_word,~GemmSpec.grid_read),~GemmSpec.row_element,~GemmSpec.lane_element,indices,key(point),channel(point),position(point),gemm_source(point),0.0),
    Equal.trans(F32,mapped,flat_dot,Specification.sum(Specification.multiply(products),0.0),
      Ordered.dot_congruence(~Nat,~Nat,~Nat,~Gemm.Both<Gemm.Both<Nat,GemmSpec.Grid>,GemmSpec.Source>,~Unit,~Unit,~Gemm.Both<Order.Flat,Unit>,
        ~GemmSpec.gemm_weight(),~GemmSpec.gemm_input(),~flat_input(),~flat_weight(),indices,channel(point),position(point),left,Unit{},Unit{},right,0.0,
        Members.range_all(~Whole,~StepAgrees,~steps_count,~step_agrees,count(point),0n,frame,NatOrder.reflexive(count(point)))),
      Equal.trans(F32,flat_dot,product_dot,Specification.sum(Specification.multiply(products),0.0),
        Equal.sym(F32,product_dot,flat_dot,Ordered.dot_reindexed(~Nat,~Specification.Product,~Unit,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_input,~MatrixLaws.product_weight,indices,flat(point),Unit{},Unit{},Unit{},0.0)),
        MatrixLaws.product_sum(products,0.0))))

def specification_count(+point: Point) -> {count(point) == Order.flat_count(flat(point)) : Nat}:
  Point{+seed,valid,bases,+image,+gradient,+weights,+channel,+position,rs,li,ci,chi,range} = point
  Chunks.Reduction{+selected,+line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Sources.phase_spec_count(call,U32.from_nat(y),U32.from_nat(x),Jobs.row_native(Chunks.work(selected),line),(Input.input_column(seed) + U32.from_nat(position) : U32),image,Nat.add(U32.to_nat(Products.channel_base(seed)),channel),gradient,weights,Reads.phase_y_inside(Chunks.phase(selected)),Reads.phase_x_inside(Chunks.phase(selected)))
def stride_positive(+point: Point) -> {Nat.is_lt(0n,Strided.stride(Order.flat_problem(flat(point)))) == True{} : Bool}:
  Point{+seed,valid,bases,image,gradient,weights,channel,position,rs,li,ci,chi,range} = point
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.stride_positive(call)
def specification_products(+point: Point) ->
  {Order.flat_range(flat(point),count(point),0n) == Strided.products(Order.flat_problem(flat(point)),Order.flat_channel(flat(point)),Order.flat_y(flat(point)),Order.flat_x(flat(point))) : List<&2,Specification.Product>}:
  %Equal.sym(Nat,count(point),Order.flat_count(flat(point)),specification_count(point)) : {Order.flat_range(flat(point),_,0n) == Strided.products(Order.flat_problem(flat(point)),Order.flat_channel(flat(point)),Order.flat_y(flat(point)),Order.flat_x(flat(point))) : List<&2,Specification.Product>}
  Order.flat_spec_products(flat(point),stride_positive(point))

# The readers of a cell's job, whatever the bias tree.
def readers(+point: Point,+bias: Tree.Tree<F32>) -> GemmJobs.Readers:
  GemmJobs.Readers{tiles(point),weight_tree(point),bias}

# Lane r of the tile row j of the job holds the specification value at its
# channel and pixel.
def cell_value(+frame: Whole,+bias: Tree.Tree<F32>) ->
  {TileLaws.select_lane(channel(whole_point(frame)),GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},
    Runtime.job(seed(whole_point(frame))),readers(whole_point(frame),bias),position(whole_point(frame)),0n)) ==
    Strided.value(Order.flat_problem(flat(whole_point(frame))),Order.flat_channel(flat(whole_point(frame))),Order.flat_y(flat(whole_point(frame))),Order.flat_x(flat(whole_point(frame)))) : F32}:
  Whole{+point,+witness} = frame
  Point{+seed,+valid,bases,image,gradient,weights,+channel,+position,+rs,li,ci,+chi,range} = point
  Chunks.Reduction{+selection,line,group,unit,out,+tr,+tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selection
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,+wb} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Chunks.Valid{vci,vroom,vfits,venabled,vji,voi,+vri,+vcoli} = valid
  +job = Channels.channel_job(seed,valid)
  +stored = Channels.lane_stored(seed,valid,channel,rs,
    %Channels.first_word_exact(job) : {Nat.is_lt(Nat.add(_,channel),Domain.channels(Jobs.geometry(Channels.phase(job)))) == True{} : Bool} chi)
  +wanted = Strided.value(Order.flat_problem(flat(point)),Order.flat_channel(flat(point)),Order.flat_y(flat(point)),Order.flat_x(flat(point)))
  +context = Jobs.context(Chunks.phase(selection))
  +chunk = Chunks.chunk(query)
  +index = Runtime.job_index(seed)
  +depth = U32.to_nat(Kernel.phase_inner(context))
  +source = gemm_source(point)
  +taps = GemmSpec.taps(~Kernel.Cursor(),~Kernel.entries,depth,Runtime.job_cursor(Runtime.phase_job(context,chunk,wb,index)))
  +products = Order.flat_range(flat(point),count(point),0n)
  +spec_products = Strided.products(Order.flat_problem(flat(point)),Order.flat_channel(flat(point)),Order.flat_y(flat(point)),Order.flat_x(flat(point)))
  +same_taps = Runtime.phase_taps(context,chunk,wb,index,
    NatOrder.positive_bound(tc,U32.to_nat(Packing.phase_columns(context)),vcoli),NatOrder.positive_bound(tr,U32.to_nat(Packing.phase_rows(context)),vri))
  Equal.trans(F32,TileLaws.select_lane(channel,GemmJobs.cell_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,GemmLaws.Tiles{},Runtime.job(seed),readers(point,bias),position,0n)),
    GemmJobs.lane_value(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,Runtime.job(seed),readers(point,bias),position,channel),wanted,
    GemmJobs.tile_lane(~Kernel.Cursor(),~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.unbiased_value,~Gemm.plus,Runtime.job(seed),readers(point,bias),position,channel,rs,stored),
    Equal.cong(F32,F32,value => (value + 0.0 : F32),GemmSpec.dot(GemmLaws.RowFirst{},taps,position,channel,source,0.0),Specification.sum(Specification.multiply(spec_products),0.0),
      Equal.trans(F32,GemmSpec.dot(GemmLaws.RowFirst{},taps,position,channel,source,0.0),GemmSpec.dot(GemmLaws.RowFirst{},keys(point),position,channel,source,0.0),Specification.sum(Specification.multiply(spec_products),0.0),
        Equal.cong(List<&2,ReductionLaws.AddressPair>,F32,list => GemmSpec.dot(GemmLaws.RowFirst{},list,position,channel,source,0.0),taps,keys(point),same_taps),
        Equal.trans(F32,GemmSpec.dot(GemmLaws.RowFirst{},keys(point),position,channel,source,0.0),Specification.sum(Specification.multiply(products),0.0),Specification.sum(Specification.multiply(spec_products),0.0),
          cell_dot(frame),
          Equal.cong(List<&2,Specification.Product>,F32,list => Specification.sum(Specification.multiply(list),0.0),products,spec_products,specification_products(point))))))
