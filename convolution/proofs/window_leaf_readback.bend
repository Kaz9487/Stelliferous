# An Array leaf packs the cells of its rows and multiplies them into fresh
# output cells. Its packed tiles are the problem values filled from its start,
# and its cells agree with the leaf cells of the whole problem.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/machine_limits.bend as Machine
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/u32_shift.bend as Shift
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ../../lib/proofs/kernel_gemm_readback.bend as GemmReadback
import ../../lib/proofs/kernel_gemm_cell_congruence.bend as CellCongruence
import ../../lib/proofs/kernel_packing_values_at.bend as ValuesAt
import ../../lib/proofs/kernel_spatial_packing_refinement.bend as LibPacking
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ../../lib/proofs/kernel_spatial_packing_pixels.bend as Pixels
import ../../lib/proofs/kernel_spatial_packing_call.bend as Call
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../inference.bend as NativeConv
import ../shape.bend as ConvShape
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tile_bounds.bend as PackingBounds
import ./packed_output_bounds.bend as Output
import ./workspace_packing_samples.bend as Prepared
import ./convolution_leaf_contract.bend as LeafContract
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./array_leaf_cells.bend as ArrayLeaf
import ./window_call_ready.bend as Ready
import ./output_values.bend as Values
import ./window_band_values.bend as WindowBandValues
import ./window_samples.bend as Samples
import ./window_problem.bend as WindowProblem
import ./window_positions.bend as Positions
import ./window_start.bend as Start

def work(+problem: ConvSpec.Problem) -> Pipeline.Work:
  NativeConv.work(ConvShape.from_problem(problem))

# The values of the channels of the cells of a problem's input.
def input_values(+problem: ConvSpec.Problem,+cells: Tree.Tree<Tile.Tile8>) -> Tree.Tree<F32>:
  +kernel = U32.from_nat(Guards.kernel(problem))
  ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (kernel * kernel) : U32),U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),cells)

# ---- The pack of a leaf ----

def aligned_ceiling(+blocks: Nat) -> {Encoded.ceiling_eight(Nat.mul(blocks,8n)) == Nat.mul(blocks,8n) : Nat}:
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(blocks,8n),7n),8n),Nat.add(blocks,Nat.div(7n,8n)),DivisionOrder.quotient_shift(7n,8n,blocks,{==})) :
    {Nat.mul(_,8n) == Nat.mul(blocks,8n) : Nat}
  %Equal.sym(Nat,Nat.add(blocks,0n),blocks,Addition.zero_right(blocks)) : {Nat.mul(_,8n) == Nat.mul(blocks,8n) : Nat}
  {==}

# An aligned span is its own padding.
def padded_aligned(+limits: Machine.Limits,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +room: {Nat.is_le(U32.to_nat(span),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(KernelWindow.padded(span)) == Nat.mul(blocks,8n) : Nat}:
  Equal.trans(Nat,U32.to_nat(KernelWindow.padded(span)),Encoded.ceiling_eight(U32.to_nat(span)),Nat.mul(blocks,8n),Encoded.padded_exact(limits,span,room),
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),span_exact) : {Encoded.ceiling_eight(_) == Nat.mul(blocks,8n) : Nat}
    aligned_ceiling(blocks))

# The positions of an aligned span from start fit the loop cap.
def pack_room_of(+limits: Machine.Limits,+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,
  +start: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +padded_value: {U32.to_nat(KernelWindow.padded(span)) == Nat.mul(blocks,8n) : Nat},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(KernelWindow.padded(span))),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(KernelWindow.padded(span)),Nat.mul(blocks,8n),padded_value) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(cap)) == True{} : Bool}
  %span_exact : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(cap)) == True{} : Bool}
  Ready.word_room(limits,allocation_cap,cap,configuration,Nat.add(U32.to_nat(start),U32.to_nat(span)),room)

# The stencil, planes, pairs and packed tiles of a leaf of span positions of a
# problem from start.
def leaf_window(+problem: ConvSpec.Problem) -> Packing.Stencil:
  Ready.band_window(problem,0,U32.from_nat(Guards.height(problem)))

def leaf_planes(+problem: ConvSpec.Problem,+start: U32,+span: U32) -> Packing.Planes:
  Packing.call_planes(leaf_window(problem),start,KernelWindow.padded(span))

def planes_pairs(+problem: ConvSpec.Problem,+planes: Packing.Planes) -> Tree.Tree<Tile.Tile16>:
  Tree.fresh(Tile.Tile16,Tensors.depth_for(Packing.plane_pairs(leaf_window(problem),planes)),Tile.filled_pair(0.0))

def leaf_pairs(+problem: ConvSpec.Problem,+start: U32,+span: U32) -> Tree.Tree<Tile.Tile16>:
  planes_pairs(problem,leaf_planes(problem,start,span))

# The tiles packed from planes, with pairs that hold them.
def planes_packed(+problem: ConvSpec.Problem,+start: U32,+span: U32,+cells: Tree.Tree<Tile.Tile8>,+planes: Packing.Planes) -> Tree.Tree<Tile.Tile8>:
  LibPacking.packed_tiles(leaf_window(problem),planes,U32.to_nat(PackingBounds.count(span,problem)),start,
    LibPacking.transposed(leaf_window(problem),planes,cells,planes_pairs(problem,planes)),ArrayLeaf.fresh_tiles(problem,span))

def leaf_packed(+problem: ConvSpec.Problem,+start: U32,+span: U32,+cells: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  planes_packed(problem,start,span,cells,leaf_planes(problem,start,span))

# A ready pack fills the tiles with the values of the cells.
def packed_fill(+problem: ConvSpec.Problem,+start: U32,+span: U32,+cells: Tree.Tree<Tile.Tile8>,+blocks: Nat,
  +units_exact: {LibPacking.unit_count(leaf_window(problem),U32.to_nat(PackingBounds.count(span,problem))) == blocks : Nat},
  +packable: Model.Packable<leaf_window(problem),blocks,Tensors.depth_for(PackingBounds.count(span,problem)),2147483648>,
  +block: Pixels.Block<leaf_window(problem),Packing.planes_of(leaf_window(problem),start,KernelWindow.padded(span)),start,blocks,
    Tensors.depth_for(Packing.plane_pairs(leaf_window(problem),Packing.planes_of(leaf_window(problem),start,KernelWindow.padded(span)))),2147483648>) ->
  {planes_packed(problem,start,span,cells,Packing.planes_of(leaf_window(problem),start,KernelWindow.padded(span))) ==
    Pixels.filled_of(leaf_window(problem),blocks,start,cells,ArrayLeaf.fresh_tiles(problem,span)) : Tree.Tree<Tile.Tile8>}:
  Model.Packable{+room,+entries_exact,+values_room,+limit} = packable
  +planes = Packing.planes_of(leaf_window(problem),start,KernelWindow.padded(span))
  +depth = Tensors.depth_for(PackingBounds.count(span,problem))
  +pair_depth = Tensors.depth_for(Packing.plane_pairs(leaf_window(problem),planes))
  Pixels.pack_fill(cells,planes_pairs(problem,planes),ArrayLeaf.fresh_tiles(problem,span),leaf_window(problem),planes,U32.to_nat(PackingBounds.count(span,problem)),
    start,blocks,depth,pair_depth,2147483648,room,units_exact,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),Shape.fresh(Tile.Tile16,pair_depth,Tile.filled_pair(0.0)),
    entries_exact,values_room,block,limit)

# A leaf of blocks eight-position blocks packs the problem values.
def packed_blocks(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +cells: Tree.Tree<Tile.Tile8>,+start: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +values: {input_values(problem,cells) == Inputs.input_tree(problem,storage) : Tree.Tree<F32>}) ->
  {leaf_packed(problem,start,span,cells) == ArrayLeaf.packed_tiles(problem,storage,start,span) : Tree.Tree<Tile.Tile8>}:
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +support = Padded.scalar(limits,problem,fast)
  +configuration = BlockBounds.configuration(limits)
  +words = Ready.stencil_room(limits,problem,valid,supported,1073741824,2147483648,configuration)
  +rows = U32.from_nat(Guards.height(problem))
  +window = leaf_window(problem)
  +count = PackingBounds.count(span,problem)
  +n = U32.to_nat(count)
  +tiles = ArrayLeaf.fresh_tiles(problem,span)
  +capacity = Machine.capacity(limits)
  +padded_room = Padded.padded_bound(limits,problem,facts,fast)
  +span_bound = Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem),Order.right_le(U32.to_nat(start),U32.to_nat(span)),range)
  +span_room = Order.transitive(U32.to_nat(span),Support.padded(problem),U32.to_nat(capacity),span_bound,padded_room)
  +padded_value = padded_aligned(limits,span,blocks,span_exact,span_room)
  +bound_le = {
    %Equal.sym(Nat,U32.to_nat(KernelWindow.padded(span)),Nat.mul(blocks,8n),padded_value) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    %span_exact : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    span_bound : {Nat.is_le(U32.to_nat(KernelWindow.padded(span)),Support.padded(problem)) == True{} : Bool}}
  +fit = Ready.plane_fit_within(limits,problem,valid,supported,span,span_room,bound_le)
  +bounds = Ready.bound_room(limits,problem,valid,supported,1073741824,2147483648,configuration,0,rows,span,words,fit)
  +pack_room = pack_room_of(limits,1073741824,2147483648,configuration,start,span,blocks,span_exact,padded_value,
    Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem),U32.to_nat(capacity),range,padded_room))
  +planes = Packing.planes_of(window,start,KernelWindow.padded(span))
  +pairs = Packing.plane_pairs(window,planes)
  +pairs_room = Ready.call_pairs(limits,problem,valid,supported,1073741824,2147483648,configuration,0,rows,start,span,positive,pack_room,span,Order.reflexive(U32.to_nat(span)),
    words,fit,bounds)
  +pair_depth = Tensors.depth_for(pairs)
  +pair_depth_ok = Depth.allocation_depth(pairs,Ready.pairs_sized(limits,problem,0,rows,start,span,pairs_room))
  +stride_count = U32.to_nat(U32.from_nat(Guards.stride(problem)))
  +channel_count = U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))
  +area = Nat.mul(Call.call_rows(window,start,KernelWindow.padded(span)),Call.call_tiles(window))
  +power = U32.to_nat(U32.shln(1,pair_depth))
  +pair_fits = {
    %Equal.sym(Nat,Nat.mul(Nat.mul(stride_count,area),channel_count),Nat.mul(Nat.mul(stride_count,channel_count),area),Ready.swap_factors(stride_count,area,channel_count)) :
      {Nat.is_le(_,power) == True{} : Bool}
    %Ready.pairs_exact(limits,problem,0,rows,start,span,pairs_room) : {Nat.is_le(_,power) == True{} : Bool}
    Depth.allocation_room(pairs,Ready.pairs_sized(limits,problem,0,rows,start,span,pairs_room)) : {Nat.is_le(Nat.mul(Nat.mul(stride_count,area),channel_count),power) == True{} : Bool}}
  +room = Ready.call_room_with(limits,problem,valid,supported,1073741824,2147483648,configuration,0,rows,start,span,positive,pack_room,span,Order.reflexive(U32.to_nat(span)),
    pair_depth,pair_depth_ok,Shift.power_fits(pair_depth,pair_depth_ok),pair_fits,words,fit,bounds)
  +block = Call.call_block(window,start,KernelWindow.padded(span),blocks,pair_depth,2147483648,room,padded_value)
  +tile_depth = Tensors.depth_for(count)
  +tiles_sized = Depth.bounded(count,Arithmetic.count_within(count,n,capacity,1073741824,{==},PackingBounds.count_bound(limits,problem,span,facts,fast,span_bound),
    Limits.capacity_allocatable(limits)))
  +tile_depth_ok = Depth.allocation_depth(count,tiles_sized)
  +count_exact = PackingBounds.count_exact(limits,problem,span,facts,fast,span_bound)
  +groups_exact = {
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),span_exact) : {Nat.div(_,8n) == blocks : Nat}
    Division.exact_quotient(blocks,8n,{==}) : {Nat.div(U32.to_nat(span),8n) == blocks : Nat}}
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,support)
  +native = U32.to_nat(Coordinates.native_inner(problem))
  +native_positive = {
    %Equal.sym(Nat,native,Coordinates.inner(problem),inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Coordinates.inner_positive(problem,facts) : {Nat.is_lt(0n,native) == True{} : Bool}}
  +filled_count = Equal.trans(Nat,Nat.mul(blocks,native),Nat.mul(blocks,Coordinates.inner(problem)),n,
    Equal.cong(Nat,Nat,value => Nat.mul(blocks,value),native,Coordinates.inner(problem),inner_exact),
    Equal.trans(Nat,Nat.mul(blocks,Coordinates.inner(problem)),Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),n,
      Equal.cong(Nat,Nat,value => Nat.mul(value,Coordinates.inner(problem)),blocks,PackingBounds.groups(span),Equal.sym(Nat,PackingBounds.groups(span),blocks,groups_exact)),
      Equal.sym(Nat,n,Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),count_exact)))
  +units_exact = {
    %filled_count : {Nat.div(_,native) == blocks : Nat}
    Division.exact_quotient(blocks,native,native_positive) : {Nat.div(n,native) == blocks : Nat}}
  +room_fits = {
    %groups_exact : {Nat.is_le(Nat.mul(_,Coordinates.inner(problem)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool}
    %count_exact : {Nat.is_le(_,U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool}
    Depth.allocation_room(count,tiles_sized) : {Nat.is_le(Nat.mul(blocks,Coordinates.inner(problem)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool}}
  +rows_height = {
    %Equal.sym(Nat,U32.to_nat(rows),Guards.height(problem),SupportedCoordinateBounds.height_exact(limits,problem,support)) : {Nat.is_le(_,Guards.height(problem)) == True{} : Bool}
    Order.reflexive(Guards.height(problem)) : {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}}
  +packable = Ready.stencil_packable(limits,problem,valid,supported,1073741824,2147483648,configuration,0,rows,blocks,tile_depth,tile_depth_ok,
    Shift.power_fits(tile_depth,tile_depth_ok),room_fits,rows_height,words)
  +planes_exact = Equal.cong(Packing.Planes,Tree.Tree<Tile.Tile8>,value => planes_packed(problem,start,span,cells,value),leaf_planes(problem,start,span),planes,
    Call.call_planes_of(window,start,KernelWindow.padded(span),{
      %Equal.sym(Nat,U32.to_nat(KernelWindow.padded(span)),Nat.mul(blocks,8n),padded_value) : {Nat.is_lt(0n,_) == True{} : Bool}
      %span_exact : {Nat.is_lt(0n,_) == True{} : Bool}
      positive : {Nat.is_lt(0n,U32.to_nat(KernelWindow.padded(span))) == True{} : Bool}}))
  Equal.trans(Tree.Tree<Tile.Tile8>,leaf_packed(problem,start,span,cells),Pixels.filled_of(window,blocks,start,cells,tiles),ArrayLeaf.packed_tiles(problem,storage,start,span),
    Equal.trans(Tree.Tree<Tile.Tile8>,leaf_packed(problem,start,span,cells),planes_packed(problem,start,span,cells,planes),Pixels.filled_of(window,blocks,start,cells,tiles),
      planes_exact,packed_fill(problem,start,span,cells,blocks,units_exact,packable,block)),
    %values : {SpatialSamples.fill(Nat.mul(blocks,SpatialSamples.inner_count(window)),Pixels.window_values(window,cells),window,start,tiles) ==
      SpatialSamples.fill(n,_,window,start,tiles) : Tree.Tree<Tile.Tile8>}
    %filled_count : {SpatialSamples.fill(Nat.mul(blocks,SpatialSamples.inner_count(window)),Pixels.window_values(window,cells),window,start,tiles) ==
      SpatialSamples.fill(_,input_values(problem,cells),window,start,tiles) : Tree.Tree<Tile.Tile8>}
    {==})

# An empty leaf packs nothing: both sides keep the fresh tiles.
def packed_empty(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +cells: Tree.Tree<Tile.Tile8>,+start: U32,+span: U32,+span_exact: {U32.to_nat(span) == 0n : Nat},
  +range: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}) ->
  {leaf_packed(problem,start,span,cells) == ArrayLeaf.packed_tiles(problem,storage,start,span) : Tree.Tree<Tile.Tile8>}:
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +window = leaf_window(problem)
  +planes = leaf_planes(problem,start,span)
  +transposed = LibPacking.transposed(window,planes,cells,leaf_pairs(problem,start,span))
  +tiles = ArrayLeaf.fresh_tiles(problem,span)
  +n = U32.to_nat(PackingBounds.count(span,problem))
  +native = U32.to_nat(Coordinates.native_inner(problem))
  +span_bound = Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem),Order.right_le(U32.to_nat(start),U32.to_nat(span)),range)
  +native_positive = {
    %Equal.sym(Nat,native,Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))) : {Nat.is_lt(0n,_) == True{} : Bool}
    Coordinates.inner_positive(problem,facts) : {Nat.is_lt(0n,native) == True{} : Bool}}
  +n_zero = Equal.trans(Nat,n,Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),0n,PackingBounds.count_exact(limits,problem,span,facts,fast,span_bound),
    %Equal.sym(Nat,U32.to_nat(span),0n,span_exact) : {Nat.mul(Nat.div(_,8n),Coordinates.inner(problem)) == 0n : Nat}
    {==})
  +no_units = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(0n,native),(0n,0n),Division.small(0n,native,native_positive))
  %Equal.sym(Nat,n,0n,n_zero) :
    {LibPacking.packed_tiles(window,planes,_,start,transposed,tiles) == ArrayLeaf.packed_tiles(problem,storage,start,span) : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Nat,n,0n,n_zero) :
    {LibPacking.packed_tiles(window,planes,0n,start,transposed,tiles) == SpatialSamples.fill(_,Inputs.input_tree(problem,storage),Prepared.window(problem),start,tiles) : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Nat,LibPacking.unit_count(window,0n),0n,no_units) :
    {LibPacking.arrays_tiles(LibPacking.units(_,window,planes,start,0,LibPacking.Arrays{transposed,tiles})) == tiles : Tree.Tree<Tile.Tile8>}
  {==}

def packed_filled(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +cells: Tree.Tree<Tile.Tile8>,+start: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +range: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +values: {input_values(problem,cells) == Inputs.input_tree(problem,storage) : Tree.Tree<F32>}) ->
  {leaf_packed(problem,start,span,cells) == ArrayLeaf.packed_tiles(problem,storage,start,span) : Tree.Tree<Tile.Tile8>}:
  match blocks:
    case 0n: packed_empty(limits,problem,storage,valid,supported,cells,start,span,span_exact,range)
    case 1n+ +rest:
      packed_blocks(limits,problem,storage,valid,supported,cells,start,span,1n+rest,span_exact,
        %Equal.sym(Nat,U32.to_nat(span),Nat.mul(1n+rest,8n),span_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(1n+rest,8n,{==},{==}),range,values)

# ---- The cells of a leaf ----

def stride_positive(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool}) -> {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Boolean.every_condition_holds(Guards.valid_conditions(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias}),5n,valid)

# One lane: the accumulator over the whole-input tiles and the one over the
# window tiles are both the matrix dot of the lane's coordinate.
def lane_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +cells: Tree.Tree<Tile.Tile8>,+values: {ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),rows,U32.from_nat(Guards.width(problem)),cells) == tree : Tree.Tree<F32>},
  +channel: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) ->
  {MicroLeaf.leaf_dot(problem,storage_contract,ArrayLeaf.packed_tiles(problem,storage_contract,offset,span),channel,column) ==
    MicroLeaf.leaf_dot(problem,storage_contract,
      leaf_packed(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
        KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span,cells),channel,column) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage_contract
  TilePartitionBounds.BlockAligned{+offset_blocks,+offset_exact} = offset_alignment
  Positions.Cover{+skip,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence,parameters} : Inputs.Contract<problem>}
  +first_rows = U32.to_nat(first)
  +window_rows = U32.to_nat(rows)
  +start_nat = U32.to_nat(offset)
  +span_nat = U32.to_nat(span)
  +cover = {Positions.Cover{skip,first_exact,fits,rows_positive,kernel_fits,top,bottom} : Positions.Cover<problem,first_rows,window_rows,start_nat,span_nat>}
  +offset_alignment = {TilePartitionBounds.BlockAligned{offset_blocks,offset_exact} : TilePartitionBounds.BlockAligned<offset>}
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +support = Padded.scalar(limits,problem,fast)
  +stride_ok = stride_positive(problem,valid)
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(image_width,kernel,stride,padding)
  +width_ok = {{==} : {Nat.is_lt(0n,out_width) == True{} : Bool}}
  +window = Samples.window(problem,first_rows,window_rows)
  +window_contract = WindowProblem.contract(problem,storage_contract,first_rows,window_rows,tree,contents,room)
  +window_start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(out_width),first)
  +full_tiles = ArrayLeaf.packed_tiles(problem,storage_contract,offset,span)
  +window_tiles = ArrayLeaf.packed_tiles(window,window_contract,window_start,span)
  +skipped = Nat.mul(skip,out_width)
  +span_positive = Order.le_lt_trans(0n,column,span_nat,Order.zero_le(column),column_inside)
  +grid = {%Positions.aligned_padded(problem,width_blocks,width) : {Nat.is_le(Nat.add(start_nat,span_nat),_) == True{} : Bool}
    range : {Nat.is_le(Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width)) == True{} : Bool}}
  +offset_room = Order.transitive(start_nat,Nat.add(start_nat,span_nat),U32.to_nat(Machine.capacity(limits)),Order.le_add(start_nat,span_nat),
    Order.transitive(Nat.add(start_nat,span_nat),Support.padded(problem),U32.to_nat(Machine.capacity(limits)),range,Padded.padded_bound(limits,problem,facts,fast)))
  +reach = Order.transitive(skipped,Nat.mul(Nat.div(start_nat,out_width),out_width),start_nat,
    RowMajor.mul_le(skip,Nat.div(start_nat,out_width),out_width,Positions.block_le_start(problem,first_rows,skip,start_nat,span_nat,stride_ok,first_exact,top)),
    DivisionOrder.product_below(start_nat,out_width,width_ok))
  +start_exact = Start.start_exact(limits,problem,facts,support,offset,first,skip,first_exact,reach,offset_room)
  # The lane's position.
  +position = Nat.add(start_nat,column)
  +after = Order.le_add(start_nat,column)
  +inside = {%Equal.sym(Bool,Nat.is_lt(Nat.add(start_nat,column),Nat.add(start_nat,span_nat)),Nat.is_lt(column,span_nat),Order.lt_offset(start_nat,column,span_nat)) :
      {_ == True{} : Bool}
    column_inside : {Nat.is_lt(position,Nat.add(start_nat,span_nat)) == True{} : Bool}}
  +row = Nat.div(position,out_width)
  +column_index = Nat.mod(position,out_width)
  +row_valid = DivisionOrder.quotient_below(position,out_height,out_width,width_ok,Bounds.nat_trans(position,Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width),inside,grid))
  +column_valid = DivisionOrder.remainder_small(position,out_width,width_ok)
  +ordered = Positions.block_le_row(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after)
  +virtual_row = Nat.sub(row,skip)
  # The last lane bounds the window interval.
  +last = Nat.sub(Nat.add(start_nat,span_nat),1n)
  +last_after = Order.transitive(start_nat,position,last,after,Positions.last_row(position,start_nat,span_nat,inside))
  +whole_positive = Order.le_lt_trans(0n,position,Nat.add(start_nat,span_nat),Order.zero_le(position),inside)
  +whole_exact = Order.positive_has_last(Nat.add(start_nat,span_nat),whole_positive)
  +last_inside = {%whole_exact : {Nat.is_lt(last,_) == True{} : Bool}
    Order.lt_succ(last,last,Order.reflexive(last)) : {Nat.is_lt(last,Nat.add(start_nat,span_nat)) == True{} : Bool}}
  +last_row = Nat.div(last,out_width)
  +last_valid = DivisionOrder.quotient_below(last,out_height,out_width,width_ok,Bounds.nat_trans(last,Nat.add(start_nat,span_nat),Nat.mul(out_height,out_width),last_inside,grid))
  +last_ordered = Positions.block_le_row(problem,first_rows,window_rows,start_nat,span_nat,cover,last,stride_ok,width_ok,last_after)
  +last_virtual = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,last,stride_ok,width_ok,last_after,last_inside,last_valid)
  +window_height = ConvSpec.axis_size(window_rows,kernel,stride,padding)
  +window_total = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.is_le(Nat.add(_,span_nat),Nat.mul(window_height,out_width)) == True{} : Bool}
    %Positions.sub_after(start_nat,skipped,span_nat,reach) : {Nat.is_le(_,Nat.mul(window_height,out_width)) == True{} : Bool}
    Order.transitive(Nat.sub(Nat.add(start_nat,span_nat),skipped),Nat.mul(1n+Nat.sub(last_row,skip),out_width),Nat.mul(window_height,out_width),
      %Positions.sub_succ(last_row,skip,last_ordered) : {Nat.is_le(Nat.sub(Nat.add(start_nat,span_nat),skipped),Nat.mul(_,out_width)) == True{} : Bool}
      %RowMajor.subtract_products(1n+last_row,skip,out_width) : {Nat.is_le(Nat.sub(Nat.add(start_nat,span_nat),skipped),_) == True{} : Bool}
      DivisionOrder.subtract_le(Nat.add(start_nat,span_nat),Nat.mul(1n+last_row,out_width),skipped,
        %whole_exact : {Nat.is_le(_,Nat.mul(1n+last_row,out_width)) == True{} : Bool}
        Order.strict_follow(last,Nat.mul(1n+last_row,out_width),DivisionOrder.next_above(last,out_width,width_ok))),
      RowMajor.mul_le(1n+Nat.sub(last_row,skip),window_height,out_width,Order.strict_follow(Nat.sub(last_row,skip),window_height,last_virtual))) :
    {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),Nat.mul(window_height,out_width)) == True{} : Bool}}
  +window_range = {%Equal.sym(Nat,Support.padded(window),Nat.mul(window_height,out_width),Positions.aligned_padded(window,width_blocks,width)) :
      {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),_) == True{} : Bool}
    window_total : {Nat.is_le(Nat.add(U32.to_nat(window_start),span_nat),Support.padded(window)) == True{} : Bool}}
  +window_span_bound = Order.transitive(span_nat,Nat.add(U32.to_nat(window_start),span_nat),Support.padded(window),
    Order.right_le(U32.to_nat(window_start),span_nat),window_range)
  +window_alignment = {TilePartitionBounds.BlockAligned{Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),
    Equal.trans(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n),start_exact,
      %Equal.sym(Nat,start_nat,Nat.mul(offset_blocks,8n),offset_exact) : {Nat.sub(_,skipped) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      %width : {Nat.sub(Nat.mul(offset_blocks,8n),Nat.mul(skip,_)) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      %RowMajor.assoc(skip,width_blocks,8n) :
        {Nat.sub(Nat.mul(offset_blocks,8n),_) == Nat.mul(Nat.sub(offset_blocks,Nat.mul(skip,width_blocks)),8n) : Nat}
      RowMajor.subtract_products(offset_blocks,Nat.mul(skip,width_blocks),8n))} : TilePartitionBounds.BlockAligned<window_start>}
  +window_valid = WindowProblem.valid(problem,first_rows,window_rows,valid,rows_positive,kernel_fits)
  +window_supported = WindowProblem.supported(limits,problem,first_rows,window_rows,valid,supported,fits)
  +window_certificate = ArrayLeaf.evidence(limits,window,window_contract,window_start,span,window_valid,window_supported,window_span_bound)
  +full_certificate = ArrayLeaf.evidence(limits,problem,storage_contract,offset,span,valid,supported,span_bound)
  +window_position = {%Equal.sym(Nat,U32.to_nat(window_start),Nat.sub(start_nat,skipped),start_exact) :
      {Nat.add(_,column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}
    Positions.window_position(start_nat,column,out_width,skip,width_ok,reach,ordered) :
    {Nat.add(U32.to_nat(window_start),column) == Nat.add(Nat.mul(virtual_row,out_width),column_index) : Nat}}
  +window_row = Positions.virtual_row_inside(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside,row_valid)
  +indices = MatrixSpec.kchannels(ci,0n,kernel)
  +full_dot = MicroLeaf.selection_matches(~LeafContract.workspace_evidence,~LeafContract.workspace_samples,limits,problem,storage_contract,valid,supported,full_tiles,offset,span,blocks,channel,column,row,column_index,
    full_certificate,span_exact,span_bound,channel_inside,column_inside,DivisionOrder.quotient_exact(position,out_width,width_ok),row_valid,column_valid)
  +window_dot = MicroLeaf.selection_matches(~LeafContract.workspace_evidence,~LeafContract.workspace_samples,limits,window,window_contract,window_valid,window_supported,window_tiles,window_start,span,blocks,channel,column,
    virtual_row,column_index,window_certificate,span_exact,window_span_bound,channel_inside,column_inside,window_position,window_row,column_valid)
  +transfer = Samples.matrix_dot(problem,first_rows,window_rows,skip,channel,virtual_row,column_index,first_exact,fits,
    Positions.virtual_top(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside),
    Positions.virtual_bottom(problem,first_rows,window_rows,start_nat,span_nat,cover,position,stride_ok,width_ok,after,inside))
  +restored = Equal.cong(Nat,F32,value => Values.matrix_dot(indices,problem,channel,value,column_index,0.0),Nat.add(skip,virtual_row),row,
    Order.restore_shift(skip,row,ordered))
  +left = MicroLeaf.leaf_dot(problem,storage_contract,full_tiles,channel,column)
  +right = MicroLeaf.leaf_dot(problem,storage_contract,window_tiles,channel,column)
  +window_values = {
    %Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)) :
      {ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32),_,U32.from_nat(image_width),cells) == tree : Tree.Tree<F32>}
    values : {ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32),U32.from_nat(U32.to_nat(rows)),U32.from_nat(image_width),cells) == tree : Tree.Tree<F32>}}
  %Equal.sym(Tree.Tree<Tile.Tile8>,leaf_packed(window,window_start,span,cells),window_tiles,
    packed_filled(limits,window,window_contract,window_valid,window_supported,cells,window_start,span,blocks,span_exact,window_range,window_values)) :
    {left == MicroLeaf.leaf_dot(problem,storage_contract,_,channel,column) : F32}
  Equal.trans(F32,left,Values.matrix_dot(indices,problem,channel,row,column_index,0.0),right,full_dot,
    Equal.sym(F32,right,Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
      Equal.trans(F32,right,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
        window_dot,
        Equal.trans(F32,Values.matrix_dot(indices,window,channel,virtual_row,column_index,0.0),
          Values.matrix_dot(indices,problem,channel,Nat.add(skip,virtual_row),column_index,0.0),Values.matrix_dot(indices,problem,channel,row,column_index,0.0),
          transfer,restored))))

# The data an agreement proof needs, attached to one product and one tile tree.
def base_product(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) -> GemmReadback.Product:
  MicroLeaf.leaf_product(limits,problem,storage_contract,ArrayLeaf.packed_tiles(problem,storage_contract,offset,span),span,blocks,0,span,
    Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_exact,Order.le_add(U32.to_nat(span),7n),
    Output.elements_bound(limits,problem,span,Support.fast(limits,problem,supported),span_bound))

type Link<-product: GemmReadback.Product,-tiles: Tree.Tree<Tile.Tile8>> is Data:
  Link{limits: Machine.Limits,problem: ConvSpec.Problem,storage_contract: Inputs.Contract<problem>,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    offset: U32,span: U32,blocks: Nat,span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},offset_alignment: TilePartitionBounds.BlockAligned<offset>,
    range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
    width_blocks: Nat,width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
    tree: Tree.Tree<F32>,first: U32,rows: U32,
    cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
    contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
    room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
    cells: Tree.Tree<Tile.Tile8>,values: {ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),rows,U32.from_nat(Guards.width(problem)),cells) == tree : Tree.Tree<F32>},
    product_exact: {product == base_product(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound) : GemmReadback.Product},
    tiles_exact: {tiles == leaf_packed(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
      KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span,cells) : Tree.Tree<Tile.Tile8>}}

def Linked(product: GemmReadback.Product,tiles: Tree.Tree<Tile.Tile8>) -> Data:
  Link<product,tiles>

def agree(+product: GemmReadback.Product,+tiles: Tree.Tree<Tile.Tile8>,+channel: Nat,+column: Nat,+link: Linked(product,tiles),
  +channel_inside: {Nat.is_lt(channel,GemmReadback.row_count(product)) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(GemmReadback.start_column(product),column),GemmReadback.output_width(product)) == True{} : Bool}) ->
  {GemmReadback.row_dot(product,GemmReadback.product_row_tiles(product),channel,column) == GemmReadback.row_dot(product,tiles,channel,column) : F32}:
  Link{+limits,+problem,+storage_contract,+valid,+supported,+offset,+span,+blocks,+span_exact,+span_bound,+offset_alignment,+range,
    +width_blocks,+width,+tree,+first,+rows,+cover,+contents,+room,+cells,+values,+product_exact,+tiles_exact} = link
  +fast = Support.fast(limits,problem,supported)
  +base = base_product(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound)
  +window_tiles = leaf_packed(Samples.window(problem,U32.to_nat(first),U32.to_nat(rows)),
    KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first),span,cells)
  +channel_bound = {%SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast)) : {Nat.is_lt(channel,_) == True{} : Bool}
    %Equal.cong(GemmReadback.Product,Nat,value => GemmReadback.row_count(value),product,base,product_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}}
  +column_bound = {%Equal.cong(GemmReadback.Product,Bool,value => Nat.is_lt(Nat.add(GemmReadback.start_column(value),column),GemmReadback.output_width(value)),
      product,base,product_exact) : {_ == True{} : Bool}
    column_inside : {Nat.is_lt(Nat.add(GemmReadback.start_column(base),column),GemmReadback.output_width(base)) == True{} : Bool}}
  %Equal.sym(GemmReadback.Product,product,base,product_exact) :
    {GemmReadback.row_dot(_,GemmReadback.product_row_tiles(_),channel,column) == GemmReadback.row_dot(_,tiles,channel,column) : F32}
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles,window_tiles,tiles_exact) :
    {GemmReadback.row_dot(base,GemmReadback.product_row_tiles(base),channel,column) == GemmReadback.row_dot(base,_,channel,column) : F32}
  lane_matches(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents,room,cells,values,channel,column,channel_bound,column_bound)

# A leaf on a band of rows computes the leaf cells of the whole problem.
def band_leaf(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset: U32,+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},+offset_alignment: TilePartitionBounds.BlockAligned<offset>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool},
  +width_blocks: Nat,+width: {Nat.mul(width_blocks,8n) == ConvSpec.out_width(problem) : Nat},
  +tree: Tree.Tree<F32>,+first: U32,+rows: U32,
  +cover: Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>,
  +contents: WindowBandValues.Contents(WindowProblem.count(problem,U32.to_nat(rows)),WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows)),tree),
  +room: {Nat.is_le(WindowProblem.count(problem,U32.to_nat(rows)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +cells: Tree.Tree<Tile.Tile8>,
  +values: {ValuesAt.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),rows,
      U32.from_nat(Guards.width(problem)),cells) == tree : Tree.Tree<F32>}) ->
  {Execution.leaf(work(problem),offset,span,Execution.InputWindow{cells,first,rows},Inputs.weight_cells(problem,storage_contract),Inputs.bias_cells(problem,storage_contract)) ==
    ArrayLeaf.layout_cells(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage_contract
  Positions.Cover{+skip,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage_contract = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence,parameters} : Inputs.Contract<problem>}
  +cover = {Positions.Cover{skip,first_exact,fits,rows_positive,kernel_fits,top,bottom} : Positions.Cover<problem,U32.to_nat(first),U32.to_nat(rows),U32.to_nat(offset),U32.to_nat(span)>}
  +base = base_product(limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound)
  +window = Samples.window(problem,U32.to_nat(first),U32.to_nat(rows))
  +window_start = KernelWindow.start(offset,U32.from_nat(stride),U32.from_nat(ConvSpec.out_width(problem)),first)
  +window_tiles = leaf_packed(window,window_start,span,cells)
  +link = {Link{limits,problem,storage_contract,valid,supported,offset,span,blocks,span_exact,span_bound,offset_alignment,range,width_blocks,width,
    tree,first,rows,cover,contents,room,cells,values,{==},{==}} : Linked(base,window_tiles)}
  +channels = MicroLeaf.rows(problem)
  +inner = Coordinates.native_inner(problem)
  +weights = Inputs.weight_cells(problem,storage_contract)
  +bias = Inputs.bias_cells(problem,storage_contract)
  +output = Execution.allocated_tiles(channels,span)
  +computed = GemmReadback.cell_tree(Execution.leaf_matrix(channels,span,inner),window_tiles,weights,bias,output)
  %Arithmetic.native_roundtrip(rows) :
    {Execution.leaf(work(problem),offset,span,Execution.InputWindow{cells,first,_},weights,bias) ==
      ArrayLeaf.layout_cells(problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}
  Equal.cong(Tree.Tree<Tile.Tile8>,Execution.CellResult(),value => Partition.Leaf{value,offset,span,channels},computed,ArrayLeaf.leaf_cells(problem,storage_contract,offset,span),
    Equal.sym(Tree.Tree<Tile.Tile8>,ArrayLeaf.leaf_cells(problem,storage_contract,offset,span),computed,CellCongruence.product_agree(~Linked,~agree,base,window_tiles,link,bias,output)))

# A leaf on the whole image packs the problem values directly.
def root_leaf(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,+cells: Tree.Tree<Tile.Tile8>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +values: {input_values(problem,cells) == Inputs.input_tree(problem,storage) : Tree.Tree<F32>},+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}) ->
  {Execution.leaf(work(problem),offset,span,Execution.InputWindow{cells,0,U32.from_nat(Guards.height(problem))},Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage)) ==
    ArrayLeaf.layout_cells(problem,storage,Partition.Leaf{Unit{},offset,span,Unit{}}) : Execution.CellResult()}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +channels = MicroLeaf.rows(problem)
  +inner = Coordinates.native_inner(problem)
  +weights = Inputs.weight_cells(problem,storage)
  +bias = Inputs.bias_cells(problem,storage)
  %Equal.sym(Tree.Tree<Tile.Tile8>,leaf_packed(problem,offset,span,cells),ArrayLeaf.packed_tiles(problem,storage,offset,span),
    packed_filled(limits,problem,storage,valid,supported,cells,offset,span,blocks,span_exact,range,values)) :
    {Partition.Leaf{Execution.computed(channels,span,inner,weights,bias,_),offset,span,channels} ==
      Partition.Leaf{ArrayLeaf.leaf_cells(problem,storage,offset,span),offset,span,channels} : Execution.CellResult()}
  {==}
