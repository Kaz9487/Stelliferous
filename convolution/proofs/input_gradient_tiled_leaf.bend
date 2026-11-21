# Actual leaf tiles from the leaf's write list: the lane of the requested
# channel in the tile of the requested point holds its specified value, and
# the gradient source is kept.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_readback.bend as StorageReadback
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/write_program_lanes.bend as Lanes
import ../../lib/proofs/input_gradient_tiled_producer.bend as Producer
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_output_coordinates.bend as Coordinates
import ../../lib/proofs/input_gradient_tiled_sweep_selected.bend as Selected
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_phase_index.bend as Index
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_written.bend as Phases
import ./input_gradient_tiled_leaf_written.bend as LeafWritten
import ./input_gradient_tiled_empty.bend as EmptyValue
import ../../lib/proofs/input_gradient_tiled_coverage.bend as Coverage
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../input_gradient_specification.bend as Specification

# A leaf, the point asked for, the source rows and packed weights it reads,
# and what the output tiles hold before the leaf runs.
type Context is Data:
  Context{problem: Producer.Problem,image: U32,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>,
    held: Execution.Held,tiles: Buffer.Model<Tile.Tile8>,
    source: Phases.ChunkSource(Producer.query(problem),Producer.leaf_contained(problem),image,gradient),
    fits: Phases.Fits(Producer.query(problem)),
    layout: Output.Layout<Producer.query(problem)>,
    capacity: {Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)) == Output.capacity(Producer.query(problem),layout) : U32},
    depth: {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool},
    initial_zero: {TileLaws.select_lane(Nat.mod(Domain.channel(Producer.point(problem)),8n),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),Coverage.address(problem))) == (0.0 + 0.0 : F32) : F32},
    held_exact: {Execution.held_source(held) == Chunks.source(Producer.query(problem)) : Execution.Source},
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Packed.weight_tree(Weights.full_state(Index.zero(Chunks.grid(Producer.query(problem))),weights)) : Tree.Tree<Tile.Tile8>}}
def problem(context: Context) -> Producer.Problem:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  problem
def image(context: Context) -> U32:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  image
def gradient(context: Context) -> Buffer.Model<Tile.Tile8>:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  gradient
def weights(context: Context) -> Buffer.Model<Tile.Tile8>:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  weights
def output(context: Context) -> Buffer.Model<Tile.Tile8>:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  output
def source(context: Context) -> Phases.ChunkSource(Producer.query(problem(context)),Producer.leaf_contained(problem(context)),image(context),gradient(context)):
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  source
def fits(context: Context) -> Phases.Fits(Producer.query(problem(context))):
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  fits
def layout(context: Context) -> Output.Layout<Producer.query(problem(context))>:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  layout
def capacity(context: Context) -> {Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output(context))) == Output.capacity(Producer.query(problem(context)),layout(context)) : U32}:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  capacity
def depth(context: Context) -> {Nat.is_le(Buffer.depth(~Tile.Tile8,output(context)),30n) == True{} : Bool}:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  depth
def initial_zero(context: Context) ->
  {TileLaws.select_lane(Nat.mod(Domain.channel(Producer.point(problem(context))),8n),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output(context)),Coverage.address(problem(context)))) == (0.0 + 0.0 : F32) : F32}:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  initial_zero
def held(context: Context) -> Execution.Held:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  held
def tiles(context: Context) -> Buffer.Model<Tile.Tile8>:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  tiles
def held_exact(context: Context) -> {Execution.held_source(held(context)) == Chunks.source(Producer.query(problem(context))) : Execution.Source}:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  held_exact
def tiles_exact(context: Context) ->
  {Buffer.tree(~Tile.Tile8,tiles(context)) == Packed.weight_tree(Weights.full_state(Index.zero(Chunks.grid(Producer.query(problem(context)))),weights(context))) : Tree.Tree<Tile.Tile8>}:
  Context{problem,image,gradient,weights,output,held,tiles,source,fits,layout,capacity,depth,initial_zero,held_exact,tiles_exact} = context
  tiles_exact

def query(+context: Context) -> Chunks.Query:
  Producer.query(problem(context))
def leaf(+context: Context) -> LeafWritten.Leaf:
  LeafWritten.Leaf{query(context),Producer.leaf_contained(problem(context)),fits(context),image(context),gradient(context),weights(context),source(context)}
def target(+context: Context) -> Phases.Target:
  LeafWritten.target(leaf(context))
# The tiles the leaf leaves, the requested tile address and the lane of the
# requested channel in it.
def result(+context: Context) -> Tree.Tree<Tile.Tile8>:
  Buffer.tree(~Tile.Tile8,LeafWrites.leaf_tiles(Selected.native_context(query(context)),held(context),tiles(context),output(context)))
def address(+context: Context) -> U32:
  Coverage.address(problem(context))
def lane(+context: Context) -> Nat:
  Nat.mod(Domain.channel(Producer.point(problem(context))),8n)
def wanted(+context: Context) -> F32:
  +point = Producer.point(problem(context))
  Specification.value(Phases.target_problem(target(context)),Domain.channel(point),Domain.row(point),Domain.column(point))
def writes(+context: Context) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  +leaf = leaf(context)
  LeafWrites.leaf_writes(LeafWritten.native(leaf),Chunks.source(query(context)),Packed.weight_tree(LeafWritten.packed(leaf)))
def initial(+context: Context) -> Tree.Tree<Tile.Tile8>:
  Buffer.tree(~Tile.Tile8,output(context))
def tree_capacity(+context: Context) -> U32:
  Tree.capacity(Tile.Tile8,initial(context))
def region(+context: Context) -> Chunks.Query:
  Output.region(query(context))
def region_layout(+context: Context) -> Output.Layout<region(context)>:
  Output.region_layout(query(context),layout(context))

# The leaf's tiles are the initial tiles after the leaf's writes.
def result_writes(+context: Context) -> {result(context) == Writes.pure(Tile.Tile8,writes(context),initial(context)) : Tree.Tree<Tile.Tile8>}:
  +q = query(context)
  +held = held(context)
  +tiles = tiles(context)
  +output = output(context)
  +native = Selected.native_context(q)
  +source = Chunks.source(q)
  +weight_tree = Packed.weight_tree(LeafWritten.packed(leaf(context)))
  +start = initial(context)
  +read = Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(native,Execution.held_source(held),Buffer.tree(~Tile.Tile8,tiles)),start)
  Equal.trans(Tree.Tree<Tile.Tile8>,result(context),read,Writes.pure(Tile.Tile8,writes(context),start),
    LeafWrites.leaf_tiles_writes(native,held,tiles,output),
    Equal.trans(Tree.Tree<Tile.Tile8>,read,Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(native,source,Buffer.tree(~Tile.Tile8,tiles)),start),Writes.pure(Tile.Tile8,writes(context),start),
      Equal.cong(Execution.Source,Tree.Tree<Tile.Tile8>,rows => Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(native,rows,Buffer.tree(~Tile.Tile8,tiles)),start),
        Execution.held_source(held),source,held_exact(context)),
      Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,packed => Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(native,source,packed),start),
        Buffer.tree(~Tile.Tile8,tiles),weight_tree,tiles_exact(context))))

# Every write lands inside the output buffer.
def native_inside(+query: Chunks.Query,+layout: Output.Layout<query>,+point: Output.Point<query>) ->
  {U32.is_lt(Output.native_address(query,point),Output.capacity(query,layout)) == True{} : Bool}:
  +native = Output.native_address(query,point)
  Equal.trans(Bool,U32.is_lt(native,Output.capacity(query,layout)),Nat.is_lt(U32.to_nat(native),U32.to_nat(Output.capacity(query,layout))),True{},
    Comparison.lt(native,Output.capacity(query,layout)),
    %Equal.sym(Nat,U32.to_nat(native),Output.address(query,point),Output.native_address_exact(query,layout,point)) :
      {Nat.is_lt(_,U32.to_nat(Output.capacity(query,layout))) == True{} : Bool}
    Output.address_inside(query,layout,point))
def point_inside(+context: Context,+point: Output.Point<region(context)>) ->
  {U32.is_lt(Output.native_address(region(context),point),tree_capacity(context)) == True{} : Bool}:
  %Equal.sym(U32,tree_capacity(context),Output.capacity(query(context),layout(context)),capacity(context)) :
    {U32.is_lt(Output.native_address(region(context),point),_) == True{} : Bool}
  native_inside(region(context),region_layout(context),point)
def written_inside(+context: Context,+address: U32,+value: Tile.Tile8,holds: Phases.Written(target(context),address,value)) ->
  {U32.is_lt(address,tree_capacity(context)) == True{} : Bool}:
  Phases.Witness{+point,+at,l0,l1,l2,l3,l4,l5,l6,l7} = holds
  %at : {U32.is_lt(_,tree_capacity(context)) == True{} : Bool}
  point_inside(context,point)
def writes_inside(+context: Context) -> Readback.Inside(~Tile.Tile8,tree_capacity(context),writes(context)):
  Every.inside(~Tile.Tile8,~Phases.Target,~Phases.Written,~Context,~target,~tree_capacity,~written_inside,context,writes(context),LeafWritten.leaf_written(leaf(context)))

# A write at the requested address stores the requested value in the lane of
# the requested channel.
def selected_matches(+context: Context,+destination: U32,+value: Tile.Tile8,equal: Bool,+compared: {U32.is_eq(destination,address(context)) == equal : Bool},
  holds: Phases.Written(target(context),destination,value)) -> Boolean.When(equal,{TileLaws.select_lane(lane(context),value) == wanted(context) : F32}):
  match equal:
    case False{}: Unit{}
    case True{}:
      Phases.Witness{+point,+at,+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} = holds
      +area = region(context)
      +arrangement = region_layout(context)
      +requested = Coverage.requested(problem(context))
      +specified = Phases.target_problem(target(context))
      +problem = problem(context)
      +channel = Domain.channel(Producer.point(problem))
      +lane = lane(context)
      +small = DivisionOrder.remainder_small(channel,8n,{==})
      +same = Equal.trans(U32,Output.native_address(area,point),destination,Output.native_address(area,requested),at,Arithmetic.equal_of_is_eq(destination,address(context),compared))
      +fact = Phases.witness_lane(area,specified,destination,value,Phases.Witness{point,at,l0,l1,l2,l3,l4,l5,l6,l7},lane,small)
      +channel_same = Equal.trans(Nat,Phases.lane_channel(area,point,lane),Nat.add(Nat.mul(Output.group(area,requested),8n),lane),channel,
        Equal.cong(Nat,Nat,group => Nat.add(Nat.mul(group,8n),lane),Output.group(area,point),Output.group(area,requested),Coordinates.group_equal(area,arrangement,point,requested,same)),
        Equal.sym(Nat,channel,Nat.add(Nat.mul(Nat.div(channel,8n),8n),lane),DivisionOrder.quotient_exact(channel,8n,{==})))
      +geometry = Chunks.geometry(query(context))
      +inside = {%Equal.sym(Nat,Phases.lane_channel(area,point,lane),channel,channel_same) : {Nat.is_lt(_,Domain.channels(geometry)) == True{} : Bool}
        Domain.channel_inside(geometry,GeometryChunks.grid_batch(Chunks.grid(query(context))),Producer.point(problem),Producer.point_inside(problem)) :
        {Nat.is_lt(Phases.lane_channel(area,point,lane),Domain.channels(geometry)) == True{} : Bool}}
      %channel_same : {TileLaws.select_lane(lane,value) == Specification.value(specified,_,Output.row(area,requested),Output.column(area,requested)) : F32}
      %Coordinates.row_equal(area,arrangement,point,requested,same) :
        {TileLaws.select_lane(lane,value) == Specification.value(specified,Phases.lane_channel(area,point,lane),_,Output.column(area,requested)) : F32}
      %Coordinates.column_equal(area,arrangement,point,requested,same) :
        {TileLaws.select_lane(lane,value) == Specification.value(specified,Phases.lane_channel(area,point,lane),Output.row(area,point),_) : F32}
      Boolean.required({TileLaws.select_lane(lane,value) == Specification.value(specified,Phases.lane_channel(area,point,lane),Output.row(area,point),Output.column(area,point)) : F32},
        Nat.is_lt(Phases.lane_channel(area,point,lane),Domain.channels(Chunks.geometry(area))),fact,inside)
def written_matches(+context: Context,+destination: U32,+value: Tile.Tile8,holds: Phases.Written(target(context),destination,value)) ->
  Boolean.When(U32.is_eq(destination,address(context)),{TileLaws.select_lane(lane(context),value) == wanted(context) : F32}):
  selected_matches(context,destination,value,U32.is_eq(destination,address(context)),{==},holds)
def writes_matching(+context: Context) -> Readback.Matching(~F32,address(context),wanted(context),Lanes.lane_writes(lane(context),writes(context))):
  Lanes.lane_matching(~Phases.Target,~Phases.Written,~Context,~target,~address,~lane,~wanted,~written_matches,context,writes(context),LeafWritten.leaf_written(leaf(context)))

# The requested lane at the requested address: the wanted value if a write is
# there, the initial lane otherwise.
def read_requested(+context: Context) ->
  {TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))) ==
    Bool.pick(F32,Readback.contains(~Tile.Tile8,address(context),writes(context)),wanted(context),TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,initial(context),address(context)))) : F32}:
  +output = output(context)
  +lane = lane(context)
  +query = address(context)
  +written = writes(context)
  +before = Tree.read(Tile.Tile8,initial(context),query)
  +lanes = Lanes.lane_writes(lane,written)
  Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,result(context),query)),TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,written,initial(context)),query)),
    Bool.pick(F32,Readback.contains(~Tile.Tile8,query,written),wanted(context),TileLaws.select_lane(lane,before)),
    Equal.cong(Tree.Tree<Tile.Tile8>,F32,tree => TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,query)),result(context),Writes.pure(Tile.Tile8,written,initial(context)),result_writes(context)),
    Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,written,initial(context)),query)),TileLaws.select_lane(lane,Readback.selected(~Tile.Tile8,written,query,before)),
      Bool.pick(F32,Readback.contains(~Tile.Tile8,query,written),wanted(context),TileLaws.select_lane(lane,before)),
      Equal.cong(Tile.Tile8,F32,TileLaws.select_lane(lane),Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,written,initial(context)),query),Readback.selected(~Tile.Tile8,written,query,before),
        Readback.program(~Tile.Tile8,written,Buffer.depth(~Tile.Tile8,output),initial(context),query,Buffer.balanced(~Tile.Tile8,output),depth(context),
          point_inside(context,Coverage.requested(problem(context))),writes_inside(context))),
      Equal.trans(F32,TileLaws.select_lane(lane,Readback.selected(~Tile.Tile8,written,query,before)),Readback.selected(~F32,lanes,query,TileLaws.select_lane(lane,before)),
        Bool.pick(F32,Readback.contains(~Tile.Tile8,query,written),wanted(context),TileLaws.select_lane(lane,before)),
        Lanes.selected_lane(lane,written,query,before),
        Equal.trans(F32,Readback.selected(~F32,lanes,query,TileLaws.select_lane(lane,before)),Bool.pick(F32,Readback.contains(~F32,query,lanes),wanted(context),TileLaws.select_lane(lane,before)),
          Bool.pick(F32,Readback.contains(~Tile.Tile8,query,written),wanted(context),TileLaws.select_lane(lane,before)),
          Readback.selected_matching(~F32,lanes,query,wanted(context),TileLaws.select_lane(lane,before),writes_matching(context)),
          Equal.cong(Bool,F32,present => Bool.pick(F32,present,wanted(context),TileLaws.select_lane(lane,before)),Readback.contains(~F32,query,lanes),Readback.contains(~Tile.Tile8,query,written),
            Lanes.contains_lane(lane,query,written))))))

# A point of an empty phase is never written: its wanted value is the
# initial +0.0 + 0.0.
def empty_value(+context: Context,+wanted_zero: {wanted(context) == (0.0 + 0.0 : F32) : F32}) ->
  {TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))) == wanted(context) : F32}:
  +before = TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,initial(context),address(context)))
  Equal.trans(F32,TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))),
    Bool.pick(F32,Readback.contains(~Tile.Tile8,address(context),writes(context)),wanted(context),before),wanted(context),
    read_requested(context),
    StorageReadback.pick_same(~F32,Readback.contains(~Tile.Tile8,address(context),writes(context)),wanted(context),before,wanted(context),
      {==},Equal.trans(F32,before,(0.0 + 0.0 : F32),wanted(context),initial_zero(context),Equal.sym(F32,wanted(context),(0.0 + 0.0 : F32),wanted_zero))))

# A point of an enabled phase is written, so it holds its wanted value.
def active_value(+context: Context,+enabled: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0) == False{} : Bool}) ->
  {TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))) == wanted(context) : F32}:
  +state = LeafWritten.packed(leaf(context))
  +active = {Coverage.Active{problem(context),enabled,fits(context)} : Coverage.Active}
  +covered = Coverage.leaf_covered(active,Chunks.source(query(context)),Packed.weight_tree(state))
  +before = TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,initial(context),address(context)))
  Equal.trans(F32,TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))),
    Bool.pick(F32,Readback.contains(~Tile.Tile8,address(context),writes(context)),wanted(context),before),wanted(context),
    read_requested(context),
    %Equal.sym(Bool,Readback.contains(~Tile.Tile8,address(context),writes(context)),True{},covered) : {Bool.pick(F32,_,wanted(context),before) == wanted(context) : F32}
    {==})

# The requested lane holds its wanted value.
def selected_value(disabled: Bool,+context: Context,+compared: {U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0) == disabled : Bool}) ->
  {TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))) == wanted(context) : F32}:
  match disabled:
    case True{}: empty_value(context,EmptyValue.value(problem(context),image(context),gradient(context),weights(context),compared))
    case False{}: active_value(context,compared)
def value(+context: Context) -> {TileLaws.select_lane(lane(context),Tree.read(Tile.Tile8,result(context),address(context))) == wanted(context) : F32}:
  selected_value(U32.is_eq(Kernel.phase_inner(Jobs.context(Producer.phase(problem(context)))),0),context,{==})
