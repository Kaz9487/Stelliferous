# Geometry, allocation and complete block readback for the shared stencil pipeline.
import Base
import ../tensor.bend as Tensors
import ./kernel_gemm_refinement.bend as GemmModel
import ./word_power_of_two.bend as Power
import ./nat_ceiling.bend as Ceiling
import ../kernel_pipeline.bend as Pipeline
import ../kernel_window.bend as Window
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ../math_activation.bend as Activations
import ../machine_limits.bend as Machine
import ./kernel_pipeline_model.bend as Model
import ./kernel_pipeline_window_values.bend as Values
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./kernel_packing_window_readback.bend as PackedReadback
import ./kernel_spatial_packing_refinement.bend as Packed
import ./kernel_pipeline_bounds.bend as BlockBounds
import ./kernel_partition_bounds.bend as Partition
import ./kernel_gemm_readback.bend as GemmReadback
import ./kernel_pipeline_activation.bend as Activation
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as Depth
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as WordBounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./encoded_coordinates.bend as Encoded
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./u32_division.bend as NativeDivision
import ./word_arithmetic.bend as Words
import ./boolean_logic.bend as Boolean

type Layout is Data:
  Layout{coordinates: Coordinates.Geometry,rows: U32,positions: U32,origin: U32,limits: Machine.Limits,budget: U32}

def coordinates(layout: Layout) -> Coordinates.Geometry:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  coordinates

def rows(layout: Layout) -> U32:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  rows

def positions(layout: Layout) -> U32:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  positions

def origin(layout: Layout) -> U32:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  origin

def limits(layout: Layout) -> Machine.Limits:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  limits

def budget(layout: Layout) -> U32:
  Layout{coordinates,rows,positions,origin,limits,budget} = layout
  budget

def inner(layout: Layout) -> U32:
  Coordinates.inner_word(coordinates(layout))

def capacity(layout: Layout) -> U32:
  Coordinates.capacity(coordinates(layout))

def padded(layout: Layout) -> U32:
  Window.padded(positions(layout))

def span(+layout: Layout) -> U32:
  (8 * U32.max(1,(budget(layout) / inner(layout) : U32)) : U32)

def output_count(+layout: Layout) -> U32:
  (rows(layout) * positions(layout) : U32)

def workspace_count(+layout: Layout) -> U32:
  Pipeline.workspace_tiles(budget(layout),inner(layout),positions(layout))

def work(+layout: Layout) -> Pipeline.WindowWork:
  Pipeline.WindowWork{rows(layout),positions(layout),Coordinates.window(coordinates(layout))}

# These are integer geometry and allocator bounds, with no value oracle.
type Bounds<-layout: Layout> is Data:
  Bounds{rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows(layout)),3n),U32.to_nat(capacity(layout))) == True{} : Bool},
    padded_area: {Nat.is_le(Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(padded(layout))),U32.to_nat(capacity(layout))) == True{} : Bool},
    weights_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(inner(layout))),U32.to_nat(capacity(layout))) == True{} : Bool},
    origin_room: {Nat.is_le(Nat.add(U32.to_nat(origin(layout)),U32.to_nat(padded(layout))),U32.to_nat(capacity(layout))) == True{} : Bool},
    loop_room: {Nat.is_le(Nat.add(U32.to_nat(padded(layout)),U32.to_nat(span(layout))),U32.to_nat(capacity(layout))) == True{} : Bool},
    positions_positive: {Nat.is_lt(0n,U32.to_nat(positions(layout))) == True{} : Bool},
    positions_room: {Nat.is_le(U32.to_nat(positions(layout)),U32.to_nat(Machine.capacity(limits(layout)))) == True{} : Bool},
    inner_room: {Nat.is_le(U32.to_nat(inner(layout)),U32.to_nat(Machine.capacity(limits(layout)))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(positions(layout))),U32.to_nat(Machine.capacity(limits(layout)))) == True{} : Bool},
    budget_room: {Nat.is_le(U32.to_nat(budget(layout)),U32.to_nat(Machine.capacity(limits(layout)))) == True{} : Bool},
    budget_span: {Nat.is_le(Nat.mul(U32.to_nat(budget(layout)),16n),U32.to_nat(capacity(layout))) == True{} : Bool},
    allocation_room: {Nat.is_le(U32.to_nat(Machine.capacity(limits(layout))),U32.to_nat(capacity(layout))) == True{} : Bool}}

type Domain is Data:
  Domain{layout: Layout,budget_exact: {budget(layout) == Pipeline.block_tiles() : U32},bounds: Bounds<layout>,output_exact: {U32.to_nat(output_count(layout)) == Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(positions(layout))) : Nat},
    workspace_exact: {U32.to_nat(workspace_count(layout)) == BlockBounds.workspace_product(budget(layout),inner(layout),positions(layout)) : Nat},
    span_exact: {U32.to_nat(span(layout)) == Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n) : Nat},
    output_ready: Depth.Sized<output_count(layout)>,workspace_ready: Depth.Sized<workspace_count(layout)>}

def layout(domain: Domain) -> Layout:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  layout

def domain_bounds(+domain: Domain) -> Bounds<layout(domain)>:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  bounds

def context(+domain: Domain,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>) -> Model.WindowContext:
  Model.WindowContext{Coordinates.window(coordinates(layout(domain))),input,weights}

def coordinates_ok(+geometry: Coordinates.Geometry) -> {U32.is_le(Coordinates.capacity(geometry),2147483648) == True{} : Bool}:
  Coordinates.Geometry{window,channels,capacity,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  ok

def coordinates_inner_positive(+geometry: Coordinates.Geometry) -> {Nat.is_lt(0n,Coordinates.inner(geometry)) == True{} : Bool}:
  Coordinates.Geometry{window,channels,capacity,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  ip

def budget_positive(+budget: U32,exact: {budget == Pipeline.block_tiles() : U32}) -> {Nat.is_lt(0n,U32.to_nat(budget)) == True{} : Bool}:
  %Equal.sym(U32,budget,Pipeline.block_tiles(),exact) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
  {==}

def span_generic(+target: U32,+geometry: Coordinates.Geometry,+limits: Machine.Limits,
  +target_positive: {Nat.is_lt(0n,U32.to_nat(target)) == True{} : Bool},
  +inner_room: {Nat.is_le(Coordinates.inner(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_span: {Nat.is_le(Nat.mul(U32.to_nat(target),16n),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}) ->
  {U32.to_nat((8 * U32.max(1,(target / Coordinates.inner_word(geometry) : U32)) : U32)) == Nat.mul(BlockBounds.groups(target,Coordinates.inner_word(geometry)),8n) : Nat}:
  BlockBounds.span_exact(target,target_positive,limits,Coordinates.inner_word(geometry),Coordinates.capacity(geometry),coordinates_ok(geometry),
    Order.transitive(Nat.mul(U32.to_nat(target),8n),Nat.mul(U32.to_nat(target),16n),U32.to_nat(Coordinates.capacity(geometry)),DivisionOrder.multiply_left_le(U32.to_nat(target),8n,16n,{==}),target_span),coordinates_inner_positive(geometry),inner_room)

def span_facts(+layout: Layout,+bounds: Bounds<layout>,+budget_exact: {budget(layout) == Pipeline.block_tiles() : U32}) ->
  {U32.to_nat(span(layout)) == Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n) : Nat}:
  Layout{+geometry,rows,positions,origin,+limits,+budget_word} = layout
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,+inner_room,output_room,budget_room,+budget_span,allocation_room} = bounds
  span_generic(budget_word,geometry,limits,budget_positive(budget_word,budget_exact),inner_room,budget_span)


def workspace_max_room(+layout: Layout,bounds: Bounds<layout>) -> {Nat.is_le(U32.to_nat(U32.max(budget(layout),inner(layout))),U32.to_nat(Machine.capacity(limits(layout)))) == True{} : Bool}:
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  BlockBounds.tile_count_room(budget(layout),inner(layout),Machine.capacity(limits(layout)),NativeDivision.nat_le_native(budget(layout),Machine.capacity(limits(layout)),budget_room),inner_room)

def workspace_prefix_room(+layout: Layout,bounds: Bounds<layout>) ->
  {Nat.is_le(Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),U32.to_nat(inner(layout))),U32.to_nat(U32.max(budget(layout),inner(layout)))) == True{} : Bool}:
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  BlockBounds.tile_prefix_room(budget(layout),limits(layout),inner(layout),inner_room,coordinates_inner_positive(coordinates(layout)))

def workspace_exact(+layout: Layout,+bounds: Bounds<layout>,+budget_exact: {budget(layout) == Pipeline.block_tiles() : U32}) -> {U32.to_nat(workspace_count(layout)) == BlockBounds.workspace_product(budget(layout),inner(layout),positions(layout)) : Nat}:
  BlockBounds.workspace_count_exact(budget(layout),inner(layout),positions(layout),Machine.capacity(limits(layout)),Limits.capacity_fits(limits(layout)),span_facts(layout,bounds,budget_exact),workspace_prefix_room(layout,bounds),workspace_max_room(layout,bounds))


def output_exact(+layout: Layout,bounds: Bounds<layout>) -> {U32.to_nat(output_count(layout)) == Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(positions(layout))) : Nat}:
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  Arithmetic.multiply(rows(layout),positions(layout),Machine.capacity(limits(layout)),Limits.capacity_fits(limits(layout)),output_room)


def step_readback(+job: Values.JobGeometry,+positions: U32,+start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,
  +tile_depth: Nat,+tile_capacity: U32,+tile_capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == tile_capacity : U32},+tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},+tile_capacity_ok: {U32.is_le(tile_capacity,2147483648) == True{} : Bool},
  +tile_room: {Nat.is_le(PackedReadback.count(Values.job_domain(job)),U32.to_nat(tile_capacity)) == True{} : Bool},
  +output_depth: Nat,output_balanced: Shape.Balanced(F32,output_depth,output),output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},output_room: {Nat.is_le(Nat.mul(U32.to_nat(Values.job_rows(job)),U32.to_nat(positions)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Values.job_span(job))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Coordinates.capacity(PackedReadback.geometry(Values.job_domain(job))))) == True{} : Bool},
  point_inside: {Nat.is_lt(Nat.add(U32.to_nat(start),Values.job_column(job)),U32.to_nat(positions)) == True{} : Bool}) ->
  {Tree.read(F32,Model.result(Model.step_window(Pipeline.WindowWork{Values.job_rows(job),positions,Coordinates.window(PackedReadback.geometry(Values.job_domain(job)))},PackedReadback.start(Values.job_domain(job)),start,Values.job_span(job),Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),U32.from_nat(Nat.add(Nat.mul(Values.job_channel(job),U32.to_nat(positions)),Nat.add(U32.to_nat(start),Values.job_column(job))))) ==
    Model.window_dot(Model.WindowContext{Coordinates.window(PackedReadback.geometry(Values.job_domain(job))),input,weights},(PackedReadback.start(Values.job_domain(job)) + U32.from_nat(Values.job_column(job)) : U32),Values.job_channel(job)) : F32}:
  Values.JobGeometry{+packed_domain,+rows,+span,+channel,+column,+span_exact,+rows_room,+area_room,weights_room,+channel_inside,+column_inside} = job
  +geometry = PackedReadback.geometry(packed_domain)
  +window = Coordinates.window(geometry)
  +origin = PackedReadback.start(packed_domain)
  +groups = PackedReadback.groups(packed_domain)
  +capacity = Coordinates.capacity(geometry)
  +query = Values.query(job,input,workspace,weights,bias_depth)
  +native_query = Model.step_gemm_query(rows,span,window,origin,input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),channel,column,groups,capacity,Values.coordinates_ok(geometry),span_exact,rows_room,area_room,channel_inside,column_inside)
  +actual = Tree.read(F32,Model.result(Model.step_window(Pipeline.WindowWork{rows,positions,window},origin,start,span,Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions)),Nat.add(U32.to_nat(start),column))))
  Equal.trans(F32,actual,GemmReadback.wanted(~True{},native_query),Model.window_dot(Model.WindowContext{window,input,weights},(origin + U32.from_nat(column) : U32),channel),
    Model.window_step_readback(rows,positions,window,origin,start,span,input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output,channel,column,groups,capacity,Values.coordinates_ok(geometry),span_exact,rows_room,area_room,channel_inside,column_inside,output_depth,output_balanced,output_depth_ok,output_room,end_room,positions_room,point_inside),
    %Values.query_matches_native(job,input,workspace,weights,bias_depth) : {GemmReadback.wanted(~True{},_) == Model.window_dot(Model.WindowContext{window,input,weights},(origin + U32.from_nat(column) : U32),channel) : F32}
    Values.wanted_matches(job,input,workspace,weights,bias_depth,tile_depth,tile_capacity,tile_capacity_equal,tile_balanced,tile_depth_ok,tile_capacity_ok,tile_room))


def tiles(+window: Packing.Stencil,+pack_start: U32,+clip: U32,input: Tree.Tree<F32>,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Packed.packed_output(Packed.packed_window(window,U32.to_nat((clip / 8 * Coordinates.stencil_inner(window) : U32)),pack_start,input,workspace))

def computed(+rows: U32,+positions: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+clip: U32,+input: Tree.Tree<F32>,workspace: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  GemmModel.run_at(~True{},rows,tiles(window,pack_start,clip,input,workspace),weights,bias,output,clip,Coordinates.stencil_inner(window),start,positions)

def step_pair(+rows: U32,+positions: U32,+window: Packing.Stencil,+start: U32,+clip: U32,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+pair: Packed.Packed) ->
  {Model.packed(rows,positions,Coordinates.stencil_inner(window),start,clip,weights,bias,output,pair) == Model.State{Packed.packed_input(pair),Packed.packed_output(pair),weights,bias,GemmModel.run_at(~True{},rows,Packed.packed_output(pair),weights,bias,output,clip,Coordinates.stencil_inner(window),start,positions)} : Model.State}:
  Packed.Packed{input,workspace} = pair
  Model.packed_result(rows,positions,Coordinates.stencil_inner(window),start,clip,weights,bias,output,input,workspace)

def step_state(+rows: U32,+positions: U32,+window: Packing.Stencil,+pack_start: U32,+start: U32,+clip: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {Model.step_window(Pipeline.WindowWork{rows,positions,window},pack_start,start,clip,Model.State{input,workspace,weights,bias,output}) ==
    Model.State{input,tiles(window,pack_start,clip,input,workspace),weights,bias,computed(rows,positions,window,pack_start,start,clip,input,workspace,weights,bias,output)} : Model.State}:
  Packing.Stencil{base,h,w,tr,tc,s,ro,co,ow,+inner} = window
  +pair = Packed.packed_window(window,U32.to_nat((clip / 8 * inner : U32)),pack_start,input,workspace)
  %Packed.packed_window_input(window,U32.to_nat((clip / 8 * inner : U32)),pack_start,input,workspace) :
    {Model.packed(rows,positions,inner,start,clip,weights,bias,output,pair) == Model.State{_,Packed.packed_output(pair),weights,bias,GemmModel.run_at(~True{},rows,Packed.packed_output(pair),weights,bias,output,clip,inner,start,positions)} : Model.State}
  step_pair(rows,positions,window,start,clip,weights,bias,output,pair)

def tiles_balanced(+domain: PackedReadback.Domain,+clip: U32,+exact: {U32.to_nat(clip) == Nat.mul(PackedReadback.groups(domain),8n) : Nat},+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,balanced: Shape.Balanced(Tile.Tile8,depth,workspace)) ->
  Shape.Balanced(Tile.Tile8,depth,tiles(Coordinates.window(PackedReadback.geometry(domain)),PackedReadback.start(domain),clip,input,workspace)):
  %Equal.sym(Nat,U32.to_nat((clip / 8 * Coordinates.inner_word(PackedReadback.geometry(domain)) : U32)),PackedReadback.count(domain),Values.packed_count_exact(domain,clip,exact)) :
    Shape.Balanced(Tile.Tile8,depth,Packed.packed_output(Packed.packed_window(Coordinates.window(PackedReadback.geometry(domain)),_,PackedReadback.start(domain),input,workspace)))
  %Equal.sym(Packed.Packed,Packed.packed_window(Coordinates.window(PackedReadback.geometry(domain)),PackedReadback.count(domain),PackedReadback.start(domain),input,workspace),Packed.Packed{input,PackedReadback.filled(PackedReadback.count(domain),0,PackedReadback.Tabulation{input,Coordinates.window(PackedReadback.geometry(domain)),PackedReadback.start(domain)},workspace)},PackedReadback.packed_prefix(domain,input,workspace)) : Shape.Balanced(Tile.Tile8,depth,Packed.packed_output(_))
  PackedReadback.filled_balanced(PackedReadback.count(domain),0,PackedReadback.Tabulation{input,Coordinates.window(PackedReadback.geometry(domain)),PackedReadback.start(domain)},workspace,depth,balanced)


def padded_exact(+domain: Domain) -> {U32.to_nat(padded(layout(domain))) == Encoded.ceiling_eight(U32.to_nat(positions(layout(domain)))) : Nat}:
  Domain{+layout,budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  Encoded.padded_exact(limits(layout),positions(layout),positions_room)

def padded_covers(+domain: Domain) -> {Nat.is_le(U32.to_nat(positions(layout(domain))),U32.to_nat(padded(layout(domain)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(padded(layout(domain))),Encoded.ceiling_eight(U32.to_nat(positions(layout(domain)))),padded_exact(domain)) : {Nat.is_le(U32.to_nat(positions(layout(domain))),_) == True{} : Bool}
  Encoded.within_ceiling(U32.to_nat(positions(layout(domain))))

def padded_positive(+domain: Domain) -> {Nat.is_lt(0n,U32.to_nat(padded(layout(domain)))) == True{} : Bool}:
  Domain{+layout,budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  WordBounds.nat_trans(0n,U32.to_nat(positions(layout)),U32.to_nat(padded(layout)),positions_positive,padded_covers(domain))

def padded_end(+domain: Domain) -> {Nat.is_le(U32.to_nat(padded(layout(domain))),Nat.add(U32.to_nat(positions(layout(domain))),7n)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(padded(layout(domain))),Encoded.ceiling_eight(U32.to_nat(positions(layout(domain)))),padded_exact(domain)) : {Nat.is_le(_,Nat.add(U32.to_nat(positions(layout(domain))),7n)) == True{} : Bool}
  DivisionOrder.product_below(Nat.add(U32.to_nat(positions(layout(domain))),7n),8n,{==})

def padded_alignment(+domain: Domain) -> Partition.BlockAligned<padded(layout(domain))>:
  Partition.BlockAligned{Nat.div(Nat.add(U32.to_nat(positions(layout(domain))),7n),8n),padded_exact(domain)}

def span_alignment(+domain: Domain) -> Partition.BlockAligned<span(layout(domain))>:
  Domain{+layout,budget_exact,bounds,output_exact,workspace_exact,+span_exact,output_ready,workspace_ready} = domain
  Partition.BlockAligned{BlockBounds.groups(budget(layout),inner(layout)),span_exact}

def span_positive(+domain: Domain) -> {Nat.is_lt(0n,U32.to_nat(span(layout(domain)))) == True{} : Bool}:
  Domain{+layout,budget_exact,bounds,output_exact,workspace_exact,+span_exact,output_ready,workspace_ready} = domain
  %Equal.sym(Nat,U32.to_nat(span(layout)),Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n),span_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
  WordBounds.nat_trans(0n,BlockBounds.groups(budget(layout),inner(layout)),Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n),BlockBounds.groups_positive(budget(layout),inner(layout)),RowMajor.product_covers(BlockBounds.groups(budget(layout),inner(layout)),8n,{==}))

def span_double_room(+domain: Domain) -> {Nat.is_le(Nat.double(U32.to_nat(span(layout(domain)))),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{+layout,+budget_exact,+bounds,output_exact,workspace_exact,+span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,+inner_room,output_room,budget_room,+budget_span,allocation_room} = bounds
  %Equal.sym(Nat,U32.to_nat(span(layout)),Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n),span_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(capacity(layout))) == True{} : Bool}
  %Equal.trans(Nat,Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),16n),Nat.add(Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n),Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n)),Nat.double(Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n)),RowMajor.right_distrib(BlockBounds.groups(budget(layout),inner(layout)),8n,8n),Addition.double(Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),8n))) : {Nat.is_le(_,U32.to_nat(capacity(layout))) == True{} : Bool}
  Order.transitive(Nat.mul(BlockBounds.groups(budget(layout),inner(layout)),16n),Nat.mul(U32.to_nat(budget(layout)),16n),U32.to_nat(capacity(layout)),RowMajor.mul_le(BlockBounds.groups(budget(layout),inner(layout)),U32.to_nat(budget(layout)),16n,BlockBounds.groups_room(budget(layout),budget_positive(budget(layout),budget_exact),limits(layout),inner(layout),inner_room,coordinates_inner_positive(coordinates(layout)))),budget_span)

def remaining(+layout: Layout) -> Nat:
  U32.to_nat(((padded(layout) + span(layout) - 1) / span(layout) : U32))

def remaining_exact(+domain: Domain) -> {remaining(layout(domain)) == Ceiling.ceiling(U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain)))) : Nat}:
  Domain{+layout,budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  BlockBounds.remaining_exact(padded(layout),span(layout),capacity(layout),coordinates_ok(coordinates(layout)),padded_positive(domain),span_positive(domain),loop_room,span_double_room(domain))

def initial_schedule(+domain: Domain) -> BlockBounds.Schedule<remaining(layout(domain)),0n,U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain)))>:
  %Equal.sym(Nat,remaining(layout(domain)),Ceiling.ceiling(U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain)))),remaining_exact(domain)) : BlockBounds.Schedule<_,0n,U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain)))>
  BlockBounds.initial_schedule(U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain))),padded_positive(domain),span_positive(domain))

type Memory<-layout: Layout,-workspace: Tree.Tree<Tile.Tile8>,-output: Tree.Tree<F32>,-tile_depth: Nat,-output_depth: Nat> is Data:
  Memory{tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
    tile_room: {Nat.is_le(BlockBounds.workspace_product(budget(layout),inner(layout),positions(layout)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
    output_balanced: Shape.Balanced(F32,output_depth,output),output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(positions(layout))),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}}

def allocated_room(~Element: Data,+count: U32,+fill: Element,+size: Nat,+exact: {U32.to_nat(count) == size : Nat},+ready: Depth.Sized<count>) -> {Nat.is_le(size,U32.to_nat(Tree.capacity(Element,Model.allocated(~Element,fill,count)))) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,Model.allocated(~Element,fill,count)),U32.shln(1,Tensors.depth_for(count)),Shape.capacity(Element,Tensors.depth_for(count),Model.allocated(~Element,fill,count),Shape.fresh(Element,Tensors.depth_for(count),fill))) : {Nat.is_le(size,U32.to_nat(_)) == True{} : Bool}
  %exact : {Nat.is_le(_,U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}
  Depth.allocation_room(count,ready)

def initial_memory(+domain: Domain) -> Memory<layout(domain),Model.allocated(~Tile.Tile8,Tile.filled(0.0),workspace_count(layout(domain))),Model.allocated(~F32,0.0,output_count(layout(domain))),Tensors.depth_for(workspace_count(layout(domain))),Tensors.depth_for(output_count(layout(domain)))>:
  Domain{+layout,budget_exact,bounds,+output_exact,+workspace_exact,span_exact,+output_ready,+workspace_ready} = domain
  Memory{Shape.fresh(Tile.Tile8,Tensors.depth_for(workspace_count(layout)),Tile.filled(0.0)),Activation.ready_depth(workspace_count(layout),workspace_ready),
    allocated_room(~Tile.Tile8,workspace_count(layout),Tile.filled(0.0),BlockBounds.workspace_product(budget(layout),inner(layout),positions(layout)),workspace_exact,workspace_ready),
    Shape.fresh(F32,Tensors.depth_for(output_count(layout)),0.0),Activation.ready_depth(output_count(layout),output_ready),
    allocated_room(~F32,output_count(layout),0.0,Nat.mul(U32.to_nat(rows(layout)),U32.to_nat(positions(layout))),output_exact,output_ready)}


def step_bounds(+domain: Domain) -> BlockBounds.StepBounds<origin(layout(domain)),positions(layout(domain)),padded(layout(domain)),span(layout(domain)),capacity(layout(domain))>:
  Domain{+layout,budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  BlockBounds.StepBounds{coordinates_ok(coordinates(layout)),origin_room,loop_room,padded_end(domain),span_positive(domain),padded_alignment(domain),span_alignment(domain)}

def workspace_geometry_room(+domain: Domain) -> {Nat.is_le(BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{+layout,+budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,+allocation_room} = bounds
  Order.transitive(BlockBounds.workspace_product(budget(layout),inner(layout),positions(layout)),U32.to_nat(U32.max(budget(layout),inner(layout))),U32.to_nat(capacity(layout)),
    BlockBounds.workspace_product_room(budget(layout),inner(layout),positions(layout),span_facts(layout,bounds,budget_exact),workspace_prefix_room(layout,bounds)),
    Order.transitive(U32.to_nat(U32.max(budget(layout),inner(layout))),U32.to_nat(Machine.capacity(limits(layout))),U32.to_nat(capacity(layout)),workspace_max_room(layout,bounds),allocation_room))

def clipped_count_room(+domain: Domain,+clip: U32,+groups: Nat,+exact: {U32.to_nat(clip) == Nat.mul(groups,8n) : Nat},+prefix_bound: {Nat.is_le(U32.to_nat(clip),U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(groups,Coordinates.inner(coordinates(layout(domain)))),BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain)))) == True{} : Bool}:
  +group_exact = {%Equal.sym(Nat,U32.to_nat(clip),Nat.mul(groups,8n),exact) : {Nat.div(_,8n) == groups : Nat} Division.exact_quotient(groups,8n,{==}) : {Nat.div(U32.to_nat(clip),8n) == groups : Nat}}
  %group_exact : {Nat.is_le(Nat.mul(_,Coordinates.inner(coordinates(layout(domain)))),BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain)))) == True{} : Bool}
  RowMajor.mul_le(Nat.div(U32.to_nat(clip),8n),Nat.div(U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain)))),8n),Coordinates.inner(coordinates(layout(domain))),DivisionOrder.quotient_le(U32.to_nat(clip),U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain)))),8n,{==},prefix_bound))

def block_packing(+domain: Domain,+start: U32,+clip: U32,+groups: Nat,+exact: {U32.to_nat(clip) == Nat.mul(groups,8n) : Nat},
  +prefix_bound: {Nat.is_le(U32.to_nat(clip),U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))))) == True{} : Bool},
  +origin_room: {Nat.is_le(Nat.add(U32.to_nat((origin(layout(domain)) + start : U32)),U32.to_nat(clip)),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}) -> PackedReadback.Domain:
  PackedReadback.Domain{coordinates(layout(domain)),groups,(origin(layout(domain)) + start : U32),
    Order.transitive(Nat.mul(groups,Coordinates.inner(coordinates(layout(domain)))),BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))),U32.to_nat(capacity(layout(domain))),clipped_count_room(domain,clip,groups,exact,prefix_bound),workspace_geometry_room(domain)),
    %exact : {Nat.is_le(Nat.add(U32.to_nat((origin(layout(domain)) + start : U32)),_),U32.to_nat(capacity(layout(domain)))) == True{} : Bool} origin_room}

def rows_room(+domain: Domain) -> {Nat.is_le(Nat.add(U32.to_nat(rows(layout(domain))),3n),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  rows_room

def padded_area_room(+domain: Domain) -> {Nat.is_le(Nat.mul(U32.to_nat(rows(layout(domain))),U32.to_nat(padded(layout(domain)))),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  padded_area

def weights_geometry_room(+domain: Domain) -> {Nat.is_le(Nat.mul(U32.to_nat(rows(layout(domain))),Coordinates.inner(coordinates(layout(domain)))),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,positions_room,inner_room,output_room,budget_room,budget_span,allocation_room} = bounds
  weights_room

def block_job(+domain: Domain,+start: U32,+clip: U32,+groups: Nat,+exact: {U32.to_nat(clip) == Nat.mul(groups,8n) : Nat},
  +prefix_bound: {Nat.is_le(U32.to_nat(clip),U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))))) == True{} : Bool},
  +origin_room: {Nat.is_le(Nat.add(U32.to_nat((origin(layout(domain)) + start : U32)),U32.to_nat(clip)),U32.to_nat(capacity(layout(domain)))) == True{} : Bool},
  +clip_bound: {Nat.is_le(U32.to_nat(clip),U32.to_nat(padded(layout(domain)))) == True{} : Bool},+channel: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(clip)) == True{} : Bool}) -> Values.JobGeometry:
  Values.JobGeometry{block_packing(domain,start,clip,groups,exact,prefix_bound,origin_room),rows(layout(domain)),clip,channel,column,exact,
    rows_room(domain),Order.transitive(Nat.mul(U32.to_nat(rows(layout(domain))),U32.to_nat(clip)),Nat.mul(U32.to_nat(rows(layout(domain))),U32.to_nat(padded(layout(domain)))),U32.to_nat(capacity(layout(domain))),DivisionOrder.multiply_left_le(U32.to_nat(rows(layout(domain))),U32.to_nat(clip),U32.to_nat(padded(layout(domain))),clip_bound),padded_area_room(domain)),weights_geometry_room(domain),channel_inside,column_inside}


def tile_capacity_ok(+depth: Nat,+workspace: Tree.Tree<Tile.Tile8>,balanced: Shape.Balanced(Tile.Tile8,depth,workspace),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.is_le(Tree.capacity(Tile.Tile8,workspace),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,workspace),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,workspace,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok))

def memory_advanced(+domain: Domain,+start: U32,+clip: U32,+groups: Nat,+exact: {U32.to_nat(clip) == Nat.mul(groups,8n) : Nat},
  +prefix_bound: {Nat.is_le(U32.to_nat(clip),U32.to_nat(Pipeline.workspace_prefix(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))))) == True{} : Bool},
  +origin_room: {Nat.is_le(Nat.add(U32.to_nat((origin(layout(domain)) + start : U32)),U32.to_nat(clip)),U32.to_nat(capacity(layout(domain)))) == True{} : Bool},
  +input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,
  memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>) -> Memory<layout(domain),tiles(Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),clip,input,workspace),computed(rows(layout(domain)),positions(layout(domain)),Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),start,clip,input,workspace,weights,bias,output),tile_depth,output_depth>:
  Memory{+tile_balanced,+tile_depth_ok,+tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
  +window = Coordinates.window(coordinates(layout(domain)))
  +pack_start = (origin(layout(domain)) + start : U32)
  +packing = block_packing(domain,start,clip,groups,exact,prefix_bound,origin_room)
  +next_tiles = tiles(window,pack_start,clip,input,workspace)
  +next_output = computed(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,bias,output)
  +next_tile_balanced = tiles_balanced(packing,clip,exact,input,workspace,tile_depth,tile_balanced)
  +next_output_balanced = {
    %step_state(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,bias,output) : Shape.Balanced(F32,output_depth,Model.result(_))
    Model.window_step_balance(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,bias,output,output_depth,output_balanced) : Shape.Balanced(F32,output_depth,next_output)}
  Memory{next_tile_balanced,tile_depth_ok,Shape.transfer_room(Tile.Tile8,tile_depth,workspace,next_tiles,BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))),tile_balanced,next_tile_balanced,tile_room),
    next_output_balanced,output_depth_ok,Shape.transfer_room(F32,output_depth,output,next_output,Nat.mul(U32.to_nat(rows(layout(domain))),U32.to_nat(positions(layout(domain)))),output_balanced,next_output_balanced,output_room)}

def step_memory(+domain: Domain,+start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,
  step: BlockBounds.Step<origin(layout(domain)),positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain))>,
  memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>) ->
  Memory<layout(domain),tiles(Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),BlockBounds.current(start,padded(layout(domain)),span(layout(domain))),input,workspace),computed(rows(layout(domain)),positions(layout(domain)),Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),start,BlockBounds.current(start,padded(layout(domain)),span(layout(domain))),input,workspace,weights,bias,output),tile_depth,output_depth>:
  BlockBounds.Step{pack_exact,advance,+clipped,positive,clipped_bound,end_room,position_room,+origin_room,Partition.BlockAligned{+groups,+exact},next_aligned} = step
  memory_advanced(domain,start,BlockBounds.current(start,padded(layout(domain)),span(layout(domain))),groups,exact,BlockBounds.current_prefix_bound(positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),{==},clipped),origin_room,input,workspace,weights,bias,output,tile_depth,output_depth,memory)

def address(+domain: Domain,+channel: Nat,point: Nat) -> U32:
  U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions(layout(domain)))),point))

def wanted(+domain: Domain,input: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,point: Nat,channel: Nat) -> F32:
  Model.window_dot(context(domain,input,weights),(origin(layout(domain)) + U32.from_nat(point) : U32),channel)

def Earlier(+domain: Domain,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+channel: Nat,+point: Nat,+start: U32,+output: Tree.Tree<F32>) -> Data:
  Boolean.When(Nat.is_lt(point,U32.to_nat(start)),{Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32})

def restored_position(+origin: U32,+start: U32,+point: Nat,+lower: {Nat.is_le(U32.to_nat(start),point) == True{} : Bool}) ->
  {((origin + start : U32) + U32.from_nat(Nat.sub(point,U32.to_nat(start))) : U32) == (origin + U32.from_nat(point) : U32) : U32}:
  +local = Nat.sub(point,U32.to_nat(start))
  Equal.trans(U32,((origin + start : U32) + U32.from_nat(local) : U32),(origin + (start + U32.from_nat(local) : U32) : U32),(origin + U32.from_nat(point) : U32),Arithmetic.add_associative(origin,start,U32.from_nat(local)),
    Equal.cong(U32,U32,value => (origin + value : U32),(start + U32.from_nat(local) : U32),U32.from_nat(point),
      Equal.trans(U32,(start + U32.from_nat(local) : U32),U32.from_nat(Nat.add(U32.to_nat(start),local)),U32.from_nat(point),
        %Arithmetic.native_roundtrip(start) : {(_ + U32.from_nat(local) : U32) == U32.from_nat(Nat.add(U32.to_nat(start),local)) : U32}
        Words.native_add(U32.to_nat(start),local),Equal.cong(Nat,U32,U32.from_nat,Nat.add(U32.to_nat(start),local),point,Order.restore_shift(U32.to_nat(start),point,lower)))))


def positions_geometry_room(+domain: Domain) -> {Nat.is_le(U32.to_nat(positions(layout(domain))),U32.to_nat(capacity(layout(domain)))) == True{} : Bool}:
  Domain{+layout,budget_exact,+bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  Bounds{rows_room,padded_area,weights_room,origin_room,loop_room,positions_positive,+positions_room,inner_room,output_room,budget_room,budget_span,+allocation_room} = bounds
  Order.transitive(U32.to_nat(positions(layout)),U32.to_nat(Machine.capacity(limits(layout))),U32.to_nat(capacity(layout)),positions_room,allocation_room)

def point_written(+before: Bool,+domain: Domain,+start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+point: Nat,
  +step: BlockBounds.Step<origin(layout(domain)),positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain))>,
  +memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},+point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool},
  +start_inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded(layout(domain)))) == True{} : Bool},
  +before_exact: {Nat.is_lt(point,U32.to_nat(start)) == before : Bool},+before_next: {Nat.is_lt(point,U32.to_nat((start + span(layout(domain)) : U32))) == True{} : Bool},
  earlier: Boolean.When(before,{Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32})) ->
  {Tree.read(F32,computed(rows(layout(domain)),positions(layout(domain)),Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),start,BlockBounds.current(start,padded(layout(domain)),span(layout(domain))),input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}:
  match before:
    case True{}:
      BlockBounds.Step{pack_exact,advance,+clipped,+positive,+clip_bound,+end_room,position_room,+origin_room,Partition.BlockAligned{+groups,+exact},next_aligned} = step
      Memory{tile_balanced,tile_depth_ok,tile_room,+balanced,+depth_ok,+output_room} = memory
      +geometry = coordinates(layout(domain))
      +window = Coordinates.window(geometry)
      +clip = BlockBounds.current(start,padded(layout(domain)),span(layout(domain)))
      +pack_start = (origin(layout(domain)) + start : U32)
      +job = block_job(domain,start,clip,groups,exact,BlockBounds.current_prefix_bound(positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),{==},clipped),origin_room,clip_bound,channel,0n,channel_inside,positive)
      %step_state(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output) : {Tree.read(F32,Model.result(_),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
      Equal.trans(F32,Tree.read(F32,Model.result(Model.step_window(work(layout(domain)),pack_start,start,clip,Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),address(domain,channel,point)),Tree.read(F32,output,address(domain,channel,point)),wanted(domain,input,weights,point,channel),
        Model.window_step_preserves_before(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output,channel,0n,point,groups,capacity(layout(domain)),Values.coordinates_ok(geometry),exact,rows_room(domain),Values.job_area_room(job),channel_inside,positive,output_depth,balanced,depth_ok,output_room,end_room,positions_geometry_room(domain),point_inside,before_exact),earlier)
    case False{}:
      BlockBounds.Step{pack_exact,+advance,+clipped,positive,+clip_bound,+end_room,position_room,+origin_room,Partition.BlockAligned{+groups,+exact},next_aligned} = step
      Memory{+tile_balanced,+tile_depth_ok,+tile_room,+balanced,+depth_ok,+output_room} = memory
      +geometry = coordinates(layout(domain))
      +window = Coordinates.window(geometry)
      +clip = BlockBounds.current(start,padded(layout(domain)),span(layout(domain)))
      +pack_start = (origin(layout(domain)) + start : U32)
      +lower = Order.ge_true(point,U32.to_nat(start),Order.comparison_not_less(Nat.cmp(point,U32.to_nat(start)),before_exact))
      +local = Nat.sub(point,U32.to_nat(start))
      +restored = Order.restore_shift(U32.to_nat(start),point,lower)
      +padded_inside = WordBounds.nat_trans(point,U32.to_nat(positions(layout(domain))),U32.to_nat(padded(layout(domain))),point_inside,padded_covers(domain))
      +local_inside = {%Equal.sym(Nat,U32.to_nat(clip),Nat.min(U32.to_nat(span(layout(domain))),Nat.sub(U32.to_nat(padded(layout(domain))),U32.to_nat(start))),clipped) : {Nat.is_lt(local,_) == True{} : Bool}
        BlockBounds.clipped_contains(U32.to_nat(start),U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain))),point,WordBounds.strict_le(U32.to_nat(start),U32.to_nat(padded(layout(domain))),start_inside),lower,padded_inside,
          %advance : {Nat.is_lt(point,_) == True{} : Bool} before_next) : {Nat.is_lt(local,U32.to_nat(clip)) == True{} : Bool}}
      +prefix_bound = BlockBounds.current_prefix_bound(positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),{==},clipped)
      +job = block_job(domain,start,clip,groups,exact,prefix_bound,origin_room,clip_bound,channel,local,channel_inside,local_inside)
      %step_state(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output) : {Tree.read(F32,Model.result(_),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
      %Equal.cong(U32,F32,position => Model.window_dot(context(domain,input,weights),position,channel),(pack_start + U32.from_nat(local) : U32),(origin(layout(domain)) + U32.from_nat(point) : U32),restored_position(origin(layout(domain)),start,point,lower)) : {Tree.read(F32,Model.result(Model.step_window(work(layout(domain)),pack_start,start,clip,Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),address(domain,channel,point)) == _ : F32}
      %restored : {Tree.read(F32,Model.result(Model.step_window(work(layout(domain)),pack_start,start,clip,Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(positions(layout(domain)))),_))) == Model.window_dot(context(domain,input,weights),(pack_start + U32.from_nat(local) : U32),channel) : F32}
      step_readback(job,positions(layout(domain)),start,input,workspace,weights,bias_depth,output,tile_depth,Tree.capacity(Tile.Tile8,workspace),{==},tile_balanced,tile_depth_ok,tile_capacity_ok(tile_depth,workspace,tile_balanced,tile_depth_ok),
        Order.transitive(PackedReadback.count(Values.job_domain(job)),BlockBounds.workspace_product(budget(layout(domain)),inner(layout(domain)),positions(layout(domain))),U32.to_nat(Tree.capacity(Tile.Tile8,workspace)),clipped_count_room(domain,clip,groups,exact,prefix_bound),tile_room),
        output_depth,balanced,depth_ok,output_room,end_room,positions_geometry_room(domain),
        %Equal.sym(Nat,Nat.add(U32.to_nat(start),local),point,restored) : {Nat.is_lt(_,U32.to_nat(positions(layout(domain)))) == True{} : Bool} point_inside)


def earlier_selected(+after: Bool,+domain: Domain,+start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+point: Nat,
  +step: BlockBounds.Step<origin(layout(domain)),positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain))>,
  +memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},+point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool},
  +start_inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded(layout(domain)))) == True{} : Bool},
  +after_exact: {Nat.is_lt(point,U32.to_nat((start + span(layout(domain)) : U32))) == after : Bool},
  earlier: Earlier(domain,input,weights,channel,point,start,output)) ->
  Boolean.When(after,{Tree.read(F32,computed(rows(layout(domain)),positions(layout(domain)),Coordinates.window(coordinates(layout(domain))),(origin(layout(domain)) + start : U32),start,BlockBounds.current(start,padded(layout(domain)),span(layout(domain))),input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}):
  match after:
    case False{}: Unit{}
    case True{}:
      point_written(Nat.is_lt(point,U32.to_nat(start)),domain,start,input,workspace,weights,bias_depth,output,tile_depth,output_depth,channel,point,step,memory,channel_inside,point_inside,start_inside,{==},after_exact,earlier)

type LoopEvidence<-domain: Domain,-input: Tree.Tree<F32>,-weights: Tree.Tree<Tile.Tile8>,-channel: Nat,-point: Nat,-output_depth: Nat,-output: Tree.Tree<F32>> is Data:
  LoopEvidence{balanced: Shape.Balanced(F32,output_depth,output),value: {Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}}

def loop_readback(+remaining: Nat,+domain: Domain,+start: U32,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+point: Nat,
  +memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},+point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool},
  +schedule: BlockBounds.Schedule<remaining,U32.to_nat(start),U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain)))>,
  +aligned: Partition.BlockAligned<start>,earlier: Earlier(domain,input,weights,channel,point,start,output)) ->
  LoopEvidence<domain,input,weights,channel,point,output_depth,Model.result(Model.blocks_window(remaining,work(layout(domain)),(origin(layout(domain)) + start : U32),start,padded(layout(domain)),span(layout(domain)),Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output}))>:
  match remaining:
    case 0n:
      Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,output_depth_ok,output_room} = memory
      LoopEvidence{output_balanced,
        %WordBounds.nat_trans(point,U32.to_nat(padded(layout(domain))),U32.to_nat(start),WordBounds.nat_trans(point,U32.to_nat(positions(layout(domain))),U32.to_nat(padded(layout(domain))),point_inside,padded_covers(domain)),BlockBounds.schedule_finished(U32.to_nat(start),U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain))),schedule)) : Boolean.When(_,{Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32})
        earlier}
    case 1n+ +rest:
      +start_inside = BlockBounds.schedule_inside(rest,U32.to_nat(start),U32.to_nat(padded(layout(domain))),U32.to_nat(span(layout(domain))),schedule)
      +step = BlockBounds.step_geometry(origin(layout(domain)),positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain)),step_bounds(domain),start_inside,aligned)
      +window = Coordinates.window(coordinates(layout(domain)))
      +clip = BlockBounds.current(start,padded(layout(domain)),span(layout(domain)))
      +pack_start = (origin(layout(domain)) + start : U32)
      +bias = Tree.fresh(F32,bias_depth,0.0)
      +next_tiles = tiles(window,pack_start,clip,input,workspace)
      +next_output = computed(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,bias,output)
      %Equal.sym(Model.State,Model.step_window(work(layout(domain)),pack_start,start,clip,Model.State{input,workspace,weights,bias,output}),Model.State{input,next_tiles,weights,bias,next_output},step_state(rows(layout(domain)),positions(layout(domain)),window,pack_start,start,clip,input,workspace,weights,bias,output)) :
        LoopEvidence<domain,input,weights,channel,point,output_depth,Model.result(Model.blocks_window(rest,work(layout(domain)),(pack_start + span(layout(domain)) : U32),(start + span(layout(domain)) : U32),padded(layout(domain)),span(layout(domain)),_))>
      %Equal.sym(U32,(pack_start + span(layout(domain)) : U32),(origin(layout(domain)) + (start + span(layout(domain)) : U32) : U32),Arithmetic.add_associative(origin(layout(domain)),start,span(layout(domain)))) :
        LoopEvidence<domain,input,weights,channel,point,output_depth,Model.result(Model.blocks_window(rest,work(layout(domain)),_,(start + span(layout(domain)) : U32),padded(layout(domain)),span(layout(domain)),Model.State{input,next_tiles,weights,bias,next_output}))>
      loop_readback(rest,domain,(start + span(layout(domain)) : U32),input,next_tiles,weights,bias_depth,next_output,tile_depth,output_depth,channel,point,
        step_memory(domain,start,input,workspace,weights,bias,output,tile_depth,output_depth,step,memory),channel_inside,point_inside,
        BlockBounds.step_schedule(origin(layout(domain)),positions(layout(domain)),rest,start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain)),step,schedule),
        BlockBounds.step_alignment(origin(layout(domain)),positions(layout(domain)),start,padded(layout(domain)),span(layout(domain)),capacity(layout(domain)),step),
        earlier_selected(Nat.is_lt(point,U32.to_nat((start + span(layout(domain)) : U32))),domain,start,input,workspace,weights,bias_depth,output,tile_depth,output_depth,channel,point,step,memory,channel_inside,point_inside,start_inside,{==},earlier))

def loop_value(+domain: Domain,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+channel: Nat,+point: Nat,+output_depth: Nat,+output: Tree.Tree<F32>,evidence: LoopEvidence<domain,input,weights,channel,point,output_depth,output>) -> {Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}:
  LoopEvidence{balanced,value} = evidence
  value

# Every selected cell is overwritten from the shared zero accumulator, independent
# of the initial workspace and output contents.
def readback_from_loop(+domain: Domain,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+point: Nat,
  memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>,
  channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool}) ->
  {Tree.read(F32,Model.result(Model.blocks_window(remaining(layout(domain)),work(layout(domain)),origin(layout(domain)),0,padded(layout(domain)),span(layout(domain)),Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}:
  +state = {Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output} : Model.State}
  %Arithmetic.add_zero(origin(layout(domain))) : {Tree.read(F32,Model.result(Model.blocks_window(remaining(layout(domain)),work(layout(domain)),_,0,padded(layout(domain)),span(layout(domain)),state)),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  loop_value(domain,input,weights,channel,point,output_depth,Model.result(Model.blocks_window(remaining(layout(domain)),work(layout(domain)),(origin(layout(domain)) + 0 : U32),0,padded(layout(domain)),span(layout(domain)),state)),
    loop_readback(remaining(layout(domain)),domain,0,input,workspace,weights,bias_depth,output,tile_depth,output_depth,channel,point,memory,channel_inside,point_inside,initial_schedule(domain),Partition.BlockAligned{0n,{==}},
      %Equal.sym(Bool,Nat.is_lt(point,0n),False{},Division.lt_zero(point)) : Boolean.When(_,{Tree.read(F32,output,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}) Unit{}))

def finish_identity(+count: U32,+state: Model.State) -> {Model.finish(~Activations.identity_value,Activations.Identity{},count,state) == Model.result(state) : Tree.Tree<F32>}:
  Model.State{input,workspace,weights,bias,output} = state
  {==}

def run_observation(+layout: Layout,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Model.run_window_with_target(~Activations.identity_value,budget(layout),Activations.Identity{},work(layout),origin(layout),input,weights,bias) == Model.result(Model.blocks_window(remaining(layout),work(layout),origin(layout),0,padded(layout),span(layout),Model.State{input,Model.allocated(~Tile.Tile8,Tile.filled(0.0),workspace_count(layout)),weights,bias,Model.allocated(~F32,0.0,output_count(layout))})) : Tree.Tree<F32>}:
  Layout{Coordinates.Geometry{Packing.Stencil{base,h,w,tr,tc,s,ro,co,ow,inner},channels,cap,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr},rows,positions,origin,limits,budget} = layout
  finish_identity(output_count(layout),Model.blocks_window(remaining(layout),work(layout),origin,0,padded(layout),span(layout),Model.State{input,Model.allocated(~Tile.Tile8,Tile.filled(0.0),workspace_count(layout)),weights,bias,Model.allocated(~F32,0.0,output_count(layout))}))

def budget_exact(+domain: Domain) -> {budget(layout(domain)) == Pipeline.block_tiles() : U32}:
  Domain{layout,budget_exact,bounds,output_exact,workspace_exact,span_exact,output_ready,workspace_ready} = domain
  budget_exact

def from_observation(+layout: Layout,+state: Model.State) ->
  {Model.run_window_with_budget(budget(layout),work(layout),origin(layout),state) == Model.blocks_window(remaining(layout),work(layout),origin(layout),0,padded(layout),span(layout),state) : Model.State}:
  Layout{Coordinates.Geometry{Packing.Stencil{base,h,w,tr,tc,stride,ro,co,ow,inner},channels,capacity,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr},rows,positions,origin,limits,budget} = layout
  {==}

def readback_from(+domain: Domain,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+output: Tree.Tree<F32>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+point: Nat,
  memory: Memory<layout(domain),workspace,output,tile_depth,output_depth>,
  channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool}) ->
  {Tree.read(F32,Model.result(Model.run_window_with(work(layout(domain)),origin(layout(domain)),Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output})),address(domain,channel,point)) == Model.window_dot(context(domain,input,weights),(origin(layout(domain)) + U32.from_nat(point) : U32),channel) : F32}:
  +layout = layout(domain)
  +state = {Model.State{input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),output} : Model.State}
  %budget_exact(domain) : {Tree.read(F32,Model.result(Model.run_window_with_budget(_,work(layout),origin(layout),state)),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  %Equal.sym(Model.State,Model.run_window_with_budget(budget(layout),work(layout),origin(layout),state),Model.blocks_window(remaining(layout),work(layout),origin(layout),0,padded(layout),span(layout),state),from_observation(layout,state)) :
    {Tree.read(F32,Model.result(_),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  readback_from_loop(domain,input,workspace,weights,bias_depth,output,tile_depth,output_depth,channel,point,memory,channel_inside,point_inside)

def run_from_identity(+work: Pipeline.WindowWork,+origin: U32,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Model.run_window(~Activations.identity_value,Activations.Identity{},work,origin,input,weights,bias) == Model.result(Model.run_window_with(work,origin,Model.initial_window_with_target(Pipeline.block_tiles(),work,input,weights,bias))) : Tree.Tree<F32>}:
  Pipeline.WindowWork{+rows,+positions,Packing.Stencil{base,h,w,tr,tc,stride,ro,co,ow,inner}} = work
  finish_identity((rows * positions : U32),Model.run_window_with(work,origin,Model.initial_window_with_target(Pipeline.block_tiles(),work,input,weights,bias)))

def initial_from_observation(+layout: Layout,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>) ->
  {Model.initial_window_with_target(budget(layout),work(layout),input,weights,bias) == Model.State{input,Model.allocated(~Tile.Tile8,Tile.filled(0.0),workspace_count(layout)),weights,bias,Model.allocated(~F32,0.0,output_count(layout))} : Model.State}:
  Layout{Coordinates.Geometry{Packing.Stencil{base,h,w,tr,tc,stride,ro,co,ow,inner},channels,capacity,ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr},rows,positions,origin,limits,budget} = layout
  {==}

def readback(+domain: Domain,+input: Tree.Tree<F32>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+channel: Nat,+point: Nat,
  channel_inside: {Nat.is_lt(channel,U32.to_nat(rows(layout(domain)))) == True{} : Bool},point_inside: {Nat.is_lt(point,U32.to_nat(positions(layout(domain)))) == True{} : Bool}) ->
  {Tree.read(F32,Model.run_window(~Activations.identity_value,Activations.Identity{},work(layout(domain)),origin(layout(domain)),input,weights,Tree.fresh(F32,bias_depth,0.0)),address(domain,channel,point)) == Model.window_dot(context(domain,input,weights),(origin(layout(domain)) + U32.from_nat(point) : U32),channel) : F32}:
  +layout = layout(domain)
  +workspace = Model.allocated(~Tile.Tile8,Tile.filled(0.0),workspace_count(layout))
  +output = Model.allocated(~F32,0.0,output_count(layout))
  +bias = Tree.fresh(F32,bias_depth,0.0)
  +state = {Model.State{input,workspace,weights,bias,output} : Model.State}
  %Equal.sym(Tree.Tree<F32>,Model.run_window(~Activations.identity_value,Activations.Identity{},work(layout),origin(layout),input,weights,bias),Model.result(Model.run_window_with(work(layout),origin(layout),Model.initial_window_with_target(Pipeline.block_tiles(),work(layout),input,weights,bias))),run_from_identity(work(layout),origin(layout),input,weights,bias)) :
    {Tree.read(F32,_,address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  %budget_exact(domain) : {Tree.read(F32,Model.result(Model.run_window_with(work(layout),origin(layout),Model.initial_window_with_target(_,work(layout),input,weights,bias))),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  %Equal.sym(Model.State,Model.initial_window_with_target(budget(layout),work(layout),input,weights,bias),state,initial_from_observation(layout,input,weights,bias)) :
    {Tree.read(F32,Model.result(Model.run_window_with(work(layout),origin(layout),_)),address(domain,channel,point)) == wanted(domain,input,weights,point,channel) : F32}
  readback_from(domain,input,workspace,weights,bias_depth,output,Tensors.depth_for(workspace_count(layout)),Tensors.depth_for(output_count(layout)),channel,point,initial_memory(domain),channel_inside,point_inside)
