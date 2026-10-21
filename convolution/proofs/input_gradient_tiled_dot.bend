# Ordered MAC values for one actual chunk job.
import Base
import ../../lib/proofs/input_gradient_tiled_source_witness.bend as Witness
import ../../lib/proofs/input_gradient_tiled_source_coverage.bend as Coverage
import ../../lib/proofs/input_gradient_tiled_copy_geometry.bend as Copy
import ../../lib/traversal_loop.bend as Iteration
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Patches
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_job_reads.bend as Reads
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_value_reads.bend as Input
import ../../lib/proofs/input_gradient_tiled_source_rows.bend as SourceRows
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_phase_geometry.bend as PhaseGeometry
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ./input_gradient_sources.bend as Sources
import ./input_gradient_order.bend as Order
import ../specification.bend as Specification
import ../../lib/proofs/input_gradient_tiled_reduction.bend as Natural
import ../../lib/proofs/input_gradient_tiled_reduction_indices.bend as Indices
import ../../lib/proofs/kernel_reduction_model.bend as Accumulators
import ../../lib/proofs/input_gradient_tiled_reduction.bend as Reduction

import ./input_gradient_tiled_products.bend as Products
import ../../lib/proofs/traversal_reduction_refinement.bend as Reductions
import ../../lib/proofs/input_gradient_tiled_band_extents.bend as Bands
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/nat_order.bend as NatOrder
import ../../lib/ordered_matrix.bend as Matrix
import ../input_gradient_specification.bend as Strided
import ./matrix_matches_convolution.bend as MatrixLaws

type Point<-upper: Bool> is Data:
  Point{seed: Chunks.Reduction,valid: Chunks.Valid<seed>,bases: Runtime.Bases<Chunks.selection(seed)>,image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,row: Accumulator.Row,lane: Nat,
    lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
    column_inside: {Nat.is_lt(Nat.add(U32.to_nat(Input.input_column(seed)),lane),U32.to_nat(PhaseGeometry.axis_count(Packing.phase_x(Jobs.context(Chunks.phase(Chunks.selection(seed))))))) == True{} : Bool},
    channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(upper,row)),Domain.channels(Jobs.geometry(Chunks.phase(Chunks.selection(seed))))) == True{} : Bool},
    range: SourceRows.Range<Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(seed)))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(seed)))),Domain.gradient_rows(Chunks.geometry(Chunks.query(Chunks.selection(seed))))>}
def seed(+upper: Bool,point: Point<upper>) -> Chunks.Reduction:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  seed
def valid(+upper: Bool,point: Point<upper>) -> Chunks.Valid<seed(upper,point)>:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  valid
def bases(+upper: Bool,point: Point<upper>) -> Runtime.Bases<Chunks.selection(seed(upper,point))>:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  bases
def image(+upper: Bool,point: Point<upper>) -> U32:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  image
def gradient(+upper: Bool,point: Point<upper>) -> Buffer.Model<F32>:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  gradient
def weights(+upper: Bool,point: Point<upper>) -> Buffer.Model<F32>:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  weights
def row(+upper: Bool,point: Point<upper>) -> Accumulator.Row:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  row
def lane(+upper: Bool,point: Point<upper>) -> Nat:
  Point{seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  lane
def count(+upper: Bool,point: Point<upper>) -> Nat:
  +point = point
  Indices.count(Chunks.phase(Chunks.selection(seed(upper,point))))
def key(+upper: Bool,+point: Point<upper>) -> Accumulators.RowLane:
  Accumulators.RowLane{row(upper,point),TileLaws.lane_of(lane(upper,point))}
def initial(+upper: Bool,+point: Point<upper>) -> Reduction.Loop:
  Runtime.initial(seed(upper,point),weights(upper,point))
def source(+upper: Bool,+point: Point<upper>) -> Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>:
  Natural.natural_source(initial(upper,point))
def flat(+upper: Bool,+point: Point<upper>) -> Order.Flat:
  Products.flat(upper,row(upper,point),lane(upper,point),seed(upper,point),image(upper,point),gradient(upper,point),weights(upper,point))
def selected(+upper: Bool,+point: Point<upper>,index: Nat) -> Chunks.Reduction:
  Indices.at(seed(upper,point),index)
def selected_valid(+upper: Bool,+point: Point<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool}) -> Chunks.Valid<selected(upper,point,index)>:
  Indices.valid_at(seed(upper,point),valid(upper,point),index,inside)
def Cover(~upper: Bool,+point: Point<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool}) -> Type:
  SourceRows.Covers<Chunks.leaf_band(selected(upper,point,index),selected_valid(upper,point,index,inside)),Execution.source_first(Chunks.source(Chunks.query(Chunks.selection(seed(upper,point))))),Execution.source_rows(Chunks.source(Chunks.query(Chunks.selection(seed(upper,point)))))>
def Stored(~upper: Bool,+point: Point<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool}) -> Type:
  SourceRows.Stored<Input.image(selected(upper,point,index),selected_valid(upper,point,index,inside),gradient(upper,point),Packing.input_base(Jobs.context(Chunks.phase(Chunks.selection(seed(upper,point)))),image(upper,point)),Packing.gradient_step(Jobs.context(Chunks.phase(Chunks.selection(seed(upper,point)))))),Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(seed(upper,point)))))>
def Agreement(~upper: Bool,+point: Point<upper>,+index: Nat) -> Data:
  {Products.natural_product(~upper,row(upper,point),lane(upper,point),seed(upper,point),index,weights(upper,point)) == Order.flat_product(flat(upper,point),index) : Specification.Product}
def cell_given(~upper: Bool,+point: Point<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool},
  covers: Cover(~upper,point,index,inside),storage: Stored(~upper,point,index,inside)) -> Agreement(~upper,point,index):
  Point{+seed,+valid,+bases,+image,+gradient,+weights,+row,+lane,li,ci,chi,range} = point
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +at = Indices.at(seed,index)
  +at_valid = Indices.valid_at(seed,valid,index,inside)
  +target = Order.flat_product(flat(upper,point),index)
  Equal.trans(Specification.Product,Products.natural_product(~upper,row,lane,seed,index,weights),Products.runtime_product(upper,row,lane,at,weights),target,
    Products.natural_product_matches(~upper,row,lane,seed,valid,index,inside,weights),
    %Equal.sym(Nat,index,Runtime.tap(at),Indices.tap_exact(seed,valid,index,inside)) : {Products.runtime_product(upper,row,lane,at,weights) == Order.flat_product(flat(upper,point),_) : Specification.Product}
    Products.runtime_product_spec(~upper,row,lane,at,at_valid,bases,image,gradient,weights,li,ci,chi,range,covers,storage))
def flat_input() -> Shape.Both<Order.Flat,Unit> -> Unit -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Specification.Product,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_input)
def flat_weight() -> Shape.Both<Order.Flat,Unit> -> Unit -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~Specification.Product,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_weight)
def PairAgreement(~upper: Bool,+point: Point<upper>,+index: Nat) -> Data:
  Shape.Both<{Natural.natural_input()(source(upper,point),key(upper,point),index) == flat_input()(Shape.Both{flat(upper,point),Unit{}},Unit{},index) : F32},
    {Natural.natural_weight(~upper)(source(upper,point),key(upper,point),index) == flat_weight()(Shape.Both{flat(upper,point),Unit{}},Unit{},index) : F32}>
def pair_cell(~upper: Bool,~Frame: Data,~view: Frame -> Point<upper>,
  ~cover: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Cover(~upper,view(frame),index,inside),
  ~stored: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Stored(~upper,view(frame),index,inside),
  +frame: Frame,+index: Nat,+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool}) -> PairAgreement(~upper,view(frame),index):
  +point = view(frame)
  +same = cell_given(~upper,point,index,inside,cover(frame,index,inside),stored(frame,index,inside))
  Shape.Both{
    Equal.cong(Specification.Product,F32,product => MatrixLaws.product_input(Unit{},Unit{},product),Products.natural_product(~upper,row(upper,point),lane(upper,point),seed(upper,point),index,weights(upper,point)),Order.flat_product(flat(upper,point),index),same),
    Equal.cong(Specification.Product,F32,product => MatrixLaws.product_weight(Unit{},Unit{},product),Products.natural_product(~upper,row(upper,point),lane(upper,point),seed(upper,point),index,weights(upper,point)),Order.flat_product(flat(upper,point),index),same)}
def samples(~upper: Bool,~Frame: Data,~view: Frame -> Point<upper>,
  ~cover: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Cover(~upper,view(frame),index,inside),
  ~stored: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Stored(~upper,view(frame),index,inside),
  +frame: Frame) -> Ordered.MatchingSamples(~Nat,~Accumulators.RowLane,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>,~Unit,~Unit,~Shape.Both<Order.Flat,Unit>,~Natural.natural_input(),~Natural.natural_weight(~upper),~flat_input(),~flat_weight(),Ranges.range(count(upper,view(frame)),0n),key(upper,view(frame)),key(upper,view(frame)),source(upper,view(frame)),Unit{},Unit{},Shape.Both{flat(upper,view(frame)),Unit{}}):
  +point = view(frame)
  Members.range_all(~Frame,~{+frame => +index => PairAgreement(~upper,view(frame),index) : @+frame: Frame -> @+index: Nat -> Data},~{frame => count(upper,view(frame)) : Frame -> Nat},~pair_cell(~upper,~Frame,~view,~cover,~stored),count(upper,point),0n,frame,NatOrder.reflexive(count(upper,point)))
def natural_sum(~upper: Bool,~Frame: Data,~view: Frame -> Point<upper>,
  ~cover: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Cover(~upper,view(frame),index,inside),
  ~stored: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Stored(~upper,view(frame),index,inside),
  +frame: Frame,+initial_value: F32) ->
  {Matrix.dot(~Accumulators.RowLane,~Nat,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>,~Natural.natural_input(),~Natural.natural_weight(~upper),Ranges.range(count(upper,view(frame)),0n),key(upper,view(frame)),key(upper,view(frame)),source(upper,view(frame)),initial_value) == Specification.sum(Specification.multiply(Order.flat_range(flat(upper,view(frame)),count(upper,view(frame)),0n)),initial_value) : F32}:
  +point = view(frame)
  +indices = Ranges.range(count(upper,point),0n)
  +products = Order.flat_range(flat(upper,point),count(upper,point),0n)
  Equal.trans(F32,Matrix.dot(~Accumulators.RowLane,~Nat,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>,~Natural.natural_input(),~Natural.natural_weight(~upper),indices,key(upper,point),key(upper,point),source(upper,point),initial_value),
    Matrix.dot(~Unit,~Nat,~Unit,~Shape.Both<Order.Flat,Unit>,~flat_input(),~flat_weight(),indices,Unit{},Unit{},Shape.Both{flat(upper,point),Unit{}},initial_value),Specification.sum(Specification.multiply(products),initial_value),
    Ordered.dot_congruence(~Nat,~Accumulators.RowLane,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>,~Unit,~Unit,~Shape.Both<Order.Flat,Unit>,~Natural.natural_input(),~Natural.natural_weight(~upper),~flat_input(),~flat_weight(),indices,key(upper,point),key(upper,point),source(upper,point),Unit{},Unit{},Shape.Both{flat(upper,point),Unit{}},initial_value,samples(~upper,~Frame,~view,~cover,~stored,frame)),
    Equal.trans(F32,Matrix.dot(~Unit,~Nat,~Unit,~Shape.Both<Order.Flat,Unit>,~flat_input(),~flat_weight(),indices,Unit{},Unit{},Shape.Both{flat(upper,point),Unit{}},initial_value),Matrix.dot(~Unit,~Specification.Product,~Unit,~Unit,~MatrixLaws.product_input,~MatrixLaws.product_weight,products,Unit{},Unit{},Unit{},initial_value),Specification.sum(Specification.multiply(products),initial_value),
      Equal.sym(F32,Matrix.dot(~Unit,~Specification.Product,~Unit,~Unit,~MatrixLaws.product_input,~MatrixLaws.product_weight,products,Unit{},Unit{},Unit{},initial_value),Matrix.dot(~Unit,~Nat,~Unit,~Shape.Both<Order.Flat,Unit>,~flat_input(),~flat_weight(),indices,Unit{},Unit{},Shape.Both{flat(upper,point),Unit{}},initial_value),Ordered.dot_reindexed(~Nat,~Specification.Product,~Unit,~Unit,~Unit,~Order.Flat,~Order.flat_product,~MatrixLaws.product_input,~MatrixLaws.product_weight,indices,flat(upper,point),Unit{},Unit{},Unit{},initial_value)),MatrixLaws.product_sum(products,initial_value)))
def specification_count(+upper: Bool,+point: Point<upper>) -> {count(upper,point) == Order.flat_count(flat(upper,point)) : Nat}:
  Point{+seed,valid,bases,+image,+gradient,+weights,+row,+lane,li,ci,chi,range} = point
  Chunks.Reduction{+selected,+line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,+y,+x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Sources.phase_spec_count(call,U32.from_nat(y),U32.from_nat(x),Jobs.row_native(Chunks.work(selected),line),(Input.input_column(seed) + U32.from_nat(lane) : U32),image,Nat.add(U32.to_nat(Products.channel_base(seed)),Patches.patch_key(upper,row)),gradient,weights,Reads.phase_y_inside(Chunks.phase(selected)),Reads.phase_x_inside(Chunks.phase(selected)))
def stride_positive(+upper: Bool,+point: Point<upper>) -> {Nat.is_lt(0n,Strided.stride(Order.flat_problem(flat(upper,point)))) == True{} : Bool}:
  Point{+seed,valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Domain.stride_positive(call)
def specification_products(+upper: Bool,+point: Point<upper>) ->
  {Order.flat_range(flat(upper,point),count(upper,point),0n) == Strided.products(Order.flat_problem(flat(upper,point)),Order.flat_channel(flat(upper,point)),Order.flat_y(flat(upper,point)),Order.flat_x(flat(upper,point))) : List<&2,Specification.Product>}:
  %Equal.sym(Nat,count(upper,point),Order.flat_count(flat(upper,point)),specification_count(upper,point)) : {Order.flat_range(flat(upper,point),_,0n) == Strided.products(Order.flat_problem(flat(upper,point)),Order.flat_channel(flat(upper,point)),Order.flat_y(flat(upper,point)),Order.flat_x(flat(upper,point))) : List<&2,Specification.Product>}
  Order.flat_spec_products(flat(upper,point),stride_positive(upper,point))
def initial_zero(+upper: Bool,+point: Point<upper>) -> {Accumulators.accumulator_at(key(upper,point),Reduction.accumulators(initial(upper,point))) == 0.0 : F32}:
  Point{+seed,valid,bases,image,gradient,+weights,row,lane,li,ci,chi,range} = point
  Chunks.Reduction{+selected,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selected
  Chunks.Query{+grid,first,target,plane,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,leaf_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Accumulators.initial_zero(key(upper,point))
def leaf_mac_value(~upper: Bool,~Frame: Data,~view: Frame -> Point<upper>,
  ~cover: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Cover(~upper,view(frame),index,inside),
  ~stored: @+frame: Frame -> @+index: Nat -> @+inside: {Nat.is_lt(index,count(upper,view(frame))) == True{} : Bool} -> Stored(~upper,view(frame),index,inside),
  +frame: Frame) ->
  {(Accumulators.accumulator_at(key(upper,view(frame)),Reduction.accumulators(Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count(upper,view(frame)),initial(upper,view(frame))))) + 0.0 : F32) == Strided.value(Order.flat_problem(flat(upper,view(frame))),Order.flat_channel(flat(upper,view(frame))),Order.flat_y(flat(upper,view(frame))),Order.flat_x(flat(upper,view(frame)))) : F32}:
  +point = view(frame)
  +products = Order.flat_range(flat(upper,point),count(upper,point),0n)
  +sum = Specification.sum(Specification.multiply(products),0.0)
  +spec_products = Strided.products(Order.flat_problem(flat(upper,point)),Order.flat_channel(flat(upper,point)),Order.flat_y(flat(upper,point)),Order.flat_x(flat(upper,point)))
  Equal.cong(F32,F32,value => (value + 0.0 : F32),Accumulators.accumulator_at(key(upper,point),Reduction.accumulators(Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count(upper,point),initial(upper,point)))),Specification.sum(Specification.multiply(spec_products),0.0),
    Equal.trans(F32,Accumulators.accumulator_at(key(upper,point),Reduction.accumulators(Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count(upper,point),initial(upper,point)))),Natural.natural_dot(~upper,key(upper,point),count(upper,point),initial(upper,point)),Specification.sum(Specification.multiply(spec_products),0.0),
      Equal.trans(F32,Accumulators.accumulator_at(key(upper,point),Reduction.accumulators(Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count(upper,point),initial(upper,point)))),Reduction.ordered_dot(~upper,key(upper,point),count(upper,point),initial(upper,point)),Natural.natural_dot(~upper,key(upper,point),count(upper,point),initial(upper,point)),Reduction.ordered_value(~upper,key(upper,point),count(upper,point),initial(upper,point)),Natural.ordered_as_natural(~upper,key(upper,point),count(upper,point),initial(upper,point))),
      Equal.trans(F32,Natural.natural_dot(~upper,key(upper,point),count(upper,point),initial(upper,point)),sum,Specification.sum(Specification.multiply(spec_products),0.0),
        %Equal.sym(F32,Accumulators.accumulator_at(key(upper,point),Reduction.accumulators(initial(upper,point))),0.0,initial_zero(upper,point)) : {Matrix.dot(~Accumulators.RowLane,~Nat,~Accumulators.RowLane,~Shape.Both<Reductions.AddressLayout,Reductions.SampleSource<Tile.Tile8,Tile.Tile8,Reduction.ReadContext>>,~Natural.natural_input(),~Natural.natural_weight(~upper),Ranges.range(count(upper,point),0n),key(upper,point),key(upper,point),source(upper,point),_) == sum : F32}
        natural_sum(~upper,~Frame,~view,~cover,~stored,frame,0.0),
        Equal.cong(List<&2,Specification.Product>,F32,products => Specification.sum(Specification.multiply(products),0.0),products,spec_products,specification_products(upper,point)))))

# A full-image source constructs the reusable source evidence, not operand claims.
def whole_source(+upper: Bool,+point: Point<upper>) -> Execution.Source:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed(upper,point))))
  SourceRows.full_source(Chunks.leaf_band(seed(upper,point),valid(upper,point)),gradient(upper,point),Packing.input_base(phase,image(upper,point)),Packing.gradient_step(phase))
def source_slice(+upper: Bool,+point: Point<upper>) -> SourceRows.Slice:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed(upper,point))))
  Input.image(seed(upper,point),valid(upper,point),gradient(upper,point),Packing.input_base(phase,image(upper,point)),Packing.gradient_step(phase))
def source_outputs(+upper: Bool,+point: Point<upper>) -> U32:
  Fill.outputs(Chunks.filling(Chunks.query(Chunks.selection(seed(upper,point)))))
def SourceCoverage(+upper: Bool,+point: Point<upper>) -> Data:
  +band = Chunks.leaf_band(seed(upper,point),valid(upper,point))
  +source = Chunks.source(Chunks.query(Chunks.selection(seed(upper,point))))
  Coverage.Contains<Copy.begin(band),Copy.row_count(band),Execution.source_first(source),Execution.source_rows(source),Domain.gradient_rows(Bands.geometry(band))>
# The source rows a seed's cells read hold its band of the gradient.
def SeedSource(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+image: U32,+gradient: Buffer.Model<F32>) -> Data:
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed)))
  +query = Chunks.query(Chunks.selection(seed))
  Witness.Witness<Chunks.leaf_band(seed,valid),Chunks.source(query),gradient,Packing.input_base(phase,image),Packing.gradient_step(phase),Fill.outputs(Chunks.filling(query))>
def Source(+upper: Bool,+point: Point<upper>) -> Data:
  SeedSource(seed(upper,point),valid(upper,point),image(upper,point),gradient(upper,point))
type Whole<-upper: Bool> is Data:
  Whole{point: Point<upper>,source: Source(upper,point)}
def whole_point(+upper: Bool,frame: Whole<upper>) -> Point<upper>:
  Whole{point,source} = frame
  point
def source_cover(~upper: Bool,+point: Point<upper>,+witness: Source(upper,point),+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool}) -> Cover(~upper,point,index,inside):
  Point{+seed,+valid,bases,image,gradient,weights,row,lane,li,ci,chi,range} = point
  +band = Chunks.leaf_band(seed,valid)
  +phase = Jobs.context(Chunks.phase(Chunks.selection(seed)))
  +source = Chunks.source(Chunks.query(Chunks.selection(seed)))
  %Equal.sym(Bands.Query,Chunks.leaf_band(Indices.at(seed,index),Indices.valid_at(seed,valid,index,inside)),band,Indices.leaf_band_at(seed,valid,index,inside)) : SourceRows.Covers<_,Execution.source_first(source),Execution.source_rows(source)>
  Witness.cover(band,source,gradient,Packing.input_base(phase,image),Packing.gradient_step(phase),source_outputs(upper,point),witness)
def source_stored(~upper: Bool,+point: Point<upper>,+witness: Source(upper,point),+index: Nat,+inside: {Nat.is_lt(index,count(upper,point)) == True{} : Bool}) -> Stored(~upper,point,index,inside):
  Point{+seed,+valid,bases,+image,+gradient,weights,row,lane,li,ci,chi,range} = point
  Chunks.Reduction{+selection,line,group,unit,out,tr,tc} = seed
  Chunks.Selection{+query,y,x,yi,xi} = selection
  Chunks.Query{+grid,first,target,plane0,source0,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,leaf_count,grid_range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +band = Chunks.leaf_band(seed,valid)
  +selected = Indices.at(seed,index)
  +source = Chunks.source(query)
  +phase = Jobs.context(Chunks.phase(selection))
  +base = Packing.input_base(phase,image)
  +plane = Packing.gradient_step(phase)
  %Equal.sym(Bands.Query,Chunks.leaf_band(selected,Indices.valid_at(seed,valid,index,inside)),band,Indices.leaf_band_at(seed,valid,index,inside)) : SourceRows.Stored<SourceRows.query_slice(_,source,gradient,base,plane),source_outputs(upper,point)>
  Witness.stored(band,source,gradient,base,plane,source_outputs(upper,point),witness)
def whole_cover(~upper: Bool,+frame: Whole<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,whole_point(upper,frame))) == True{} : Bool}) -> Cover(~upper,whole_point(upper,frame),index,inside):
  Whole{+point,+source} = frame
  source_cover(~upper,point,source,index,inside)
def whole_stored(~upper: Bool,+frame: Whole<upper>,+index: Nat,+inside: {Nat.is_lt(index,count(upper,whole_point(upper,frame))) == True{} : Bool}) -> Stored(~upper,whole_point(upper,frame),index,inside):
  Whole{+point,+source} = frame
  source_stored(~upper,point,source,index,inside)
def whole_mac_value(~upper: Bool,+frame: Whole<upper>) ->
  {(Accumulators.accumulator_at(key(upper,whole_point(upper,frame)),Reduction.accumulators(Iteration.run(~Reduction.Loop,~Reduction.step(~upper),count(upper,whole_point(upper,frame)),initial(upper,whole_point(upper,frame))))) + 0.0 : F32) == Strided.value(Order.flat_problem(flat(upper,whole_point(upper,frame))),Order.flat_channel(flat(upper,whole_point(upper,frame))),Order.flat_y(flat(upper,whole_point(upper,frame))),Order.flat_x(flat(upper,whole_point(upper,frame)))) : F32}:
  leaf_mac_value(~upper,~Whole<upper>,~whole_point(upper),~whole_cover(~upper),~whole_stored(~upper),frame)

