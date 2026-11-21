# The run of the input gradient kernel: it packs the weights once, and the
# pieces it places in the grid of the input gradient cells hold the input
# gradient of the gradient cells of a Shaped tensor.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/tensor_tasks.bend as Tasks
import ../../lib/tensor_halo.bend as Halo
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weights.bend as WeightPacking
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_grid_ranges.bend as GridRanges
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_tiled_public.bend as TiledPublic
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/input_gradient_tiled_band_source.bend as BandSource
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/tensor_placed.bend as Placed
import ../../lib/proofs/kernel_window_windows.bend as KernelWindows
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_convolution.bend as Conv
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_tiled_relane.bend as Relane
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/storage_buffer.bend as Storage
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/tensor_storage_count.bend as StorageCount
import ../../lib/proofs/tensor_joined.bend as Joined
import ../../lib/proofs/tensor_whole_values.bend as WholeValues
import ../../lib/proofs/tensor_tiled_view_reads.bend as TiledViewReads
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_batch_rows.bend as BatchRows
import ./input_gradient_piece_cells.bend as PieceCells
import ./input_gradient_task.bend as TaskTree
import ./input_gradient_task_reach.bend as TaskReach
import ./input_gradient_windowed.bend as Windowed
import ./input_gradient_call.bend as Call
import ./input_gradient_sources.bend as Sources
import ./input_gradient_cell_sources.bend as CellSources
import ./image_problems.bend as Images
import ../input_gradient_specification.bend as Spec

# ---- The packed weights ----

# The packing of the weights of a call, as the model state.
def packed_state(+call: Domain.Call,+source: Buffer.Model<Tile.Tile8>) -> WeightPacking.Packed:
  +bounds = {PackingBounds.Query{call} : PackingBounds.Query}
  Packed.complete(Packed.allocated_query(bounds,source,0n,0n),Packed.initial(source,bounds))

def represent_eta(+state: WeightPacking.Packed) ->
  {WeightPacking.represent(state) ==
    Kernel.Packed{Buffer.pack(~Tile.Tile8,WeightPacking.source(state)),Buffer.pack(~Tile.Tile8,WeightPacking.weights(state)),WeightPacking.weight_base(state)} : Kernel.Packed}:
  WeightPacking.Packed{source,weights,weight_base} = state
  {==}

# Packing keeps the source weights.
def packed_source(+call: Domain.Call,+source: Buffer.Model<Tile.Tile8>) -> {WeightPacking.source(packed_state(call,source)) == source : Buffer.Model<Tile.Tile8>}:
  +bounds = {PackingBounds.Query{call} : PackingBounds.Query}
  +query = Packed.allocated_query(bounds,source,0n,0n)
  WeightPacking.phases_keep_source(Packed.count(query),0,Packed.geometry(query),Packed.initial(source,bounds))

# The packing the kernel runs is the model packing.
def packing_exact(+call: Domain.Call,+source: Buffer.Model<Tile.Tile8>) ->
  {Rows.packing(call,source) ==
    Kernel.Packed{Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,WeightPacking.weights(packed_state(call,source))),WeightPacking.weight_base(packed_state(call,source))} : Kernel.Packed}:
  +state = packed_state(call,source)
  +matched = Equal.trans(Kernel.Packed,Rows.packing(call,source),WeightPacking.represent(state),
    Kernel.Packed{Buffer.pack(~Tile.Tile8,WeightPacking.source(state)),Buffer.pack(~Tile.Tile8,WeightPacking.weights(state)),WeightPacking.weight_base(state)},
    Packed.native_complete_matches(PackingBounds.Query{call},source,0n,0n),represent_eta(state))
  %packed_source(call,source) :
    {Rows.packing(call,source) == Kernel.Packed{Buffer.pack(~Tile.Tile8,_),Buffer.pack(~Tile.Tile8,WeightPacking.weights(state)),WeightPacking.weight_base(state)} : Kernel.Packed}
  matched

# The packed weights hold the weight cells of the call.
def packed_tiles(+call: Domain.Call,+source: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,WeightPacking.weights(packed_state(call,source))) == Windowed.weight_cells(call,source) : Tree.Tree<Tile.Tile8>}:
  +state = packed_state(call,source)
  Equal.sym(Tree.Tree<Tile.Tile8>,Windowed.weight_cells(call,source),Buffer.tree(~Tile.Tile8,WeightPacking.weights(state)),
    Equal.trans(Tree.Tree<Tile.Tile8>,Rows.native_tree(Rows.packing(call,source)),Rows.native_tree(WeightPacking.represent(state)),Packed.weight_tree(state),
      Equal.cong(Kernel.Packed,Tree.Tree<Tile.Tile8>,packed => Rows.native_tree(packed),Rows.packing(call,source),WeightPacking.represent(state),
        Packed.native_complete_matches(PackingBounds.Query{call},source,0n,0n)),
      Rows.represent_tree(state)))

# ---- The run ----

# The tasks of the run over cells that are not empty.
def windowed_of(+call: Domain.Call,+cells: List<&2,U32>,+gradient_grid: Grid.Grid,+whole: U32,+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),
  +source: Buffer.Model<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  +geometry = Domain.geometry(call)
  +state = packed_state(call,source)
  Kernel.windowed(Kernel.Work{Rows.task_of(geometry),Tensors.grid_of(cells)},gradient_grid,
    Tasks.decide_whole(Tensors.grid_of(cells),workers,whole,Kernel.cell_rows(geometry),Kernel.cell_steps(geometry),
      Kernel.extra_of(geometry,whole,WeightPacking.weight_base(state))),
    gradient,Buffer.pack(~Tile.Tile8,WeightPacking.weights(state)))

# The run over cells that are not empty packs the weights, gives the source
# back and runs its tasks.
def run_exact(+call: Domain.Call,+cells: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +gradient_grid: Grid.Grid,+whole: U32,+workers: U32,-gradient: Pieces.Pieces(~Tile.Tile8),+source: Buffer.Model<Tile.Tile8>) ->
  {Kernel.run(Domain.geometry(call),cells,gradient_grid,whole,workers,gradient,Buffer.pack(~Tile.Tile8,source)) ==
    Kernel.Outcome{Buffer.pack(~Tile.Tile8,source),windowed_of(call,cells,gradient_grid,whole,workers,gradient,source)} : Kernel.Outcome}:
  +geometry = Domain.geometry(call)
  +state = packed_state(call,source)
  +product_exact = TiledPublic.product_exact(Machine.capacity(Domain.limits(call)),Limits.capacity_fits(Domain.limits(call)),cells,room)
  +positive = {%Equal.sym(Nat,U32.to_nat(Grid.product(cells)),ShapeLaws.elements(cells),product_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    nonempty : {Nat.is_lt(0n,U32.to_nat(Grid.product(cells))) == True{} : Bool}}
  +nonzero = Arithmetic.word_positive_nonzero(Grid.product(cells),positive)
  %Equal.sym(Bool,U32.is_zero(Grid.product(cells)),False{},nonzero) :
    {Kernel.run_selected(_,geometry,cells,gradient_grid,whole,workers,gradient,Buffer.pack(~Tile.Tile8,source)) ==
      Kernel.Outcome{Buffer.pack(~Tile.Tile8,source),windowed_of(call,cells,gradient_grid,whole,workers,gradient,source)} : Kernel.Outcome}
  %Equal.sym(Kernel.Packed,Rows.packing(call,source),
      Kernel.Packed{Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,WeightPacking.weights(state)),WeightPacking.weight_base(state)},packing_exact(call,source)) :
    {Kernel.packed_run(geometry,Tensors.grid_of(cells),gradient_grid,whole,workers,gradient,_) ==
      Kernel.Outcome{Buffer.pack(~Tile.Tile8,source),windowed_of(call,cells,gradient_grid,whole,workers,gradient,source)} : Kernel.Outcome}
  {==}

# ---- The windows of the gradient ----

# The windows the tasks of the input gradient read hold the gradient cells
# of a Shaped tensor.
def gradient_windows(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +cells: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: GridRanges.Multiple<GridRanges.last_extent(cells),unit>,
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +d_shape: List<&2,U32>,gradient: Pieces.Pieces(~Tile.Tile8),+d_fill: Tile.Tile8,+d_status: Tensors.Status,+d_workers: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~Holder,~value,holder,d_shape,Tensors.Tensor{gradient,d_shape,d_fill,d_status,d_workers}),
  +d_room: {Nat.is_le(ShapeLaws.elements(d_shape),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +d_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(d_shape)) == True{} : Bool},
  +d_exact: {ShapeLaws.elements(d_shape) == Rows.cells_count(call) : Nat}) ->
  Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),ShapeLaws.elements(d_shape),
    Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),Tensors.grid_of(d_shape),Tile.filled(0.0),
      Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),gradient)>:
  +limits = Domain.limits(call)
  +cap = U32.to_nat(Machine.capacity(limits))
  +total = Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))
  +grid = Grid.grid(cells,Grid.piece_target())
  +counted = U32.to_nat(Grid.count_of(grid))
  +count_exact = GridRanges.grid_count(limits,cells,Grid.piece_target(),room,nonempty)
  +tasks = Runs.decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra))
  +run = Runs.decided_run(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,counted,
    GridRanges.grid_run_of(limits,cells,Grid.piece_target(),room,nonempty,unit,divides),Held.nat_le_moved(counted,ShapeLaws.elements(cells),cap,count_exact,room))
  +reaches = {%Equal.sym(Nat,ShapeLaws.elements(d_shape),Rows.cells_count(call),d_exact) : Windows.Reaches(~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),tasks,_)
    TaskReach.tasks_reach(call,envelope,unit,counted,Held.nat_le_moved(counted,ShapeLaws.elements(cells),total,count_exact,bound_cells),total_room,tasks,run) :
    Windows.Reaches(~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),tasks,ShapeLaws.elements(d_shape))}
  Windows.covered_windows(~Tile.Tile8,~Holder,~value,~Kernel.Task,~Kernel.reach_of,limits,Tensors.grid_of(d_shape),Pieces.ranges_of_grid(Tensors.grid_of(d_shape)),ShapeLaws.elements(d_shape),
    KernelWindows.input_laid(limits,d_shape,d_room,d_nonempty),Rows.task_of(Domain.geometry(call)),holder,Tile.filled(0.0),Machine.capacity(limits),Limits.capacity_allocatable(limits),d_room,
    GridRanges.grid_pieces_word(limits,d_shape,Grid.piece_target(),d_room,d_nonempty),tasks,reaches,
    gradient,Windows.shaped_covered(~Tile.Tile8,~Holder,~value,holder,d_shape,gradient,d_fill,d_status,d_workers,shaped))

# ---- The contract of the input gradient ----

# The tasks of the run over the cells of the input gradient, reading the
# gradient cells of a Shaped tensor, give a tensor of the contract of the
# input gradient.
def output_contract(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +source: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +cells: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: GridRanges.Multiple<GridRanges.last_extent(cells),unit>,
  +counted: Global.Counted<cells>,
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +workers: U32,+whole: U32,+fill: Tile.Tile8,+cells_workers: U32,
  +d_shape: List<&2,U32>,gradient: Pieces.Pieces(~Tile.Tile8),+d_fill: Tile.Tile8,+d_status: Tensors.Status,+d_workers: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~Holder,~value,holder,d_shape,Tensors.Tensor{gradient,d_shape,d_fill,d_status,d_workers}),
  +d_room: {Nat.is_le(ShapeLaws.elements(d_shape),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +d_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(d_shape)) == True{} : Bool},
  +d_exact: {ShapeLaws.elements(d_shape) == Rows.cells_count(call) : Nat}) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    TaskTree.task_values(~Holder,~value,holder,call,source,outer,after),
    Tensors.Tensor{windowed_of(call,cells,Tensors.grid_of(d_shape),whole,workers,gradient,source),cells,fill,Tensors.Usable{},cells_workers}):
  +geometry = Domain.geometry(call)
  +state = packed_state(call,source)
  +bytes_per_cell = Kernel.cell_rows(geometry)
  +operations = Kernel.cell_steps(geometry)
  +extra = Kernel.extra_of(geometry,whole,WeightPacking.weight_base(state))
  +tiles = WeightPacking.weights(state)
  GridRanges.grid_contract(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    TaskTree.task_values(~Holder,~value,holder,call,source,outer,after),Domain.limits(call),cells,fill,cells_workers,
    windowed_of(call,cells,Tensors.grid_of(d_shape),whole,workers,gradient,source),room,nonempty,unit,divides,counted,
    Windowed.windowed_cells(~Holder,~value,holder,call,envelope,fits,source,outer,after,after_exact,outer_exact,values_room,total_room,
      ShapeLaws.elements(d_shape),d_room,cells,room,nonempty,unit,divides,bound_cells,Tensors.grid_of(d_shape),gradient,tiles,packed_tiles(call,source),
      workers,whole,bytes_per_cell,operations,extra,
      gradient_windows(~Holder,~value,holder,call,envelope,total_room,cells,room,nonempty,unit,divides,bound_cells,workers,whole,bytes_per_cell,operations,extra,
        d_shape,gradient,d_fill,d_status,d_workers,shaped,d_room,d_nonempty,d_exact)))

# ---- The specification of a call ----

# The readers of the output gradient and the weights, and the geometry of the
# call.
type InputRead<-Gradient: Data,-Weights: Data> is Data:
  InputRead{gradient: Gradient,weights: Weights,geometry: Kernel.Geometry}

# One image of the call read from the readers.
def problem(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient: Gradient,+weights: Weights,
  +geometry: Kernel.Geometry,+image: Nat) -> Spec.Problem:
  Spec.Problem{Sources.shape(geometry),Images.image_values(~Gradient,~gradient_value,Domain.gradient_size(geometry),Nat.mul(image,Domain.gradient_size(geometry)),gradient),
    Images.image_values(~Weights,~weight_value,Domain.weight_size(geometry),0n,weights)}

# The value of the input gradient at an index: the specification at the point
# the index decodes to.
def input_at(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,read: InputRead<Gradient,Weights>,index: U32) -> F32:
  InputRead{+gradient,+weights,+geometry} = read
  +point = Domain.point(geometry,U32.to_nat(index))
  Spec.value(problem(~Gradient,~gradient_value,~Weights,~weight_value,gradient,weights,geometry,Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point))

# An input gradient call: the input gradient holds the specification of the
# readers' values, and the weights come back holding theirs.
def TiledInputCall(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Tiled.Tiled & Tiled.Tiled) -> Data:
  (dx,weights) = result
  KernelGemm.Both<TiledPublic.Contract(~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
      Gradient.input_shape(Call.arguments(mode,geometry)),dx),
    TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights)>

# ---- Readers that agree ----

# Values that agree with a reader below a bound list, from first on below the
# bound, the values of the reader.
def listed_values(~Logical: Data,~original: Logical -> U32 -> F32,~Related: CellSources.Values -> Logical -> Data,~bound: Nat,
  ~agrees: @+values: CellSources.Values -> @+logical: Logical -> @+related: Related(values,logical) -> @+index: Nat -> @+below: {Nat.is_lt(index,bound) == True{} : Bool} ->
    {CellSources.value_at(values,index) == original(logical,U32.from_nat(index)) : F32},
  +values: CellSources.Values,+logical: Logical,+related: Related(values,logical),count: Nat,+first: Nat,+room: {Nat.is_le(Nat.add(first,count),bound) == True{} : Bool}) ->
  {ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,count,first,values) == Images.image_values(~Logical,~original,count,first,logical) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +moved = {%Addition.successor_right(first,rest) : {Nat.is_le(_,bound) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(first,rest),bound) == True{} : Bool}}
      +below = Bounds.nat_trans(first,1n+Nat.add(first,rest),bound,Order.lt_succ(first,Nat.add(first,rest),Order.le_add(first,rest)),moved)
      %Equal.sym(F32,CellSources.value_at(values,first),original(logical,U32.from_nat(first)),agrees(values,logical,related,first,below)) :
        {_ <> ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,rest,1n+first,values) ==
          original(logical,U32.from_nat(first)) <> Images.image_values(~Logical,~original,rest,1n+first,logical) : List<&2,F32>}
      %Equal.sym(List<&2,F32>,ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,rest,1n+first,values),Images.image_values(~Logical,~original,rest,1n+first,logical),
          listed_values(~Logical,~original,~Related,~bound,~agrees,values,logical,related,rest,1n+first,moved)) :
        {original(logical,U32.from_nat(first)) <> _ == original(logical,U32.from_nat(first)) <> Images.image_values(~Logical,~original,rest,1n+first,logical) : List<&2,F32>}
      {==}

# Values that agree with the readers of the gradient and the weights give the
# problem of an image of the readers.
def problems_agree(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  ~GradientRelated: CellSources.Values -> Gradient -> Data,~gradient_bound: Nat,
  ~gradient_agrees: @+values: CellSources.Values -> @+logical: Gradient -> @+related: GradientRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,gradient_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == gradient_value(logical,U32.from_nat(index)) : F32},
  ~WeightRelated: CellSources.Values -> Weights -> Data,~weight_bound: Nat,
  ~weight_agrees: @+values: CellSources.Values -> @+logical: Weights -> @+related: WeightRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,weight_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == weight_value(logical,U32.from_nat(index)) : F32},
  +gradient_context: Gradient,+weight_context: Weights,+geometry: Kernel.Geometry,+gradient_values: CellSources.Values,+weight_values: CellSources.Values,
  +gradient_related: GradientRelated(gradient_values,gradient_context),+weight_related: WeightRelated(weight_values,weight_context),+images: Nat,
  +gradient_cover: {Nat.is_le(Nat.mul(images,Domain.gradient_size(geometry)),gradient_bound) == True{} : Bool},
  +weight_cover: {Nat.is_le(Domain.weight_size(geometry),weight_bound) == True{} : Bool},
  +image: Nat,+image_inside: {Nat.is_lt(image,images) == True{} : Bool}) ->
  {CellSources.problem(geometry,CellSources.Reader{gradient_values,weight_values},image) ==
    problem(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,image) : Spec.Problem}:
  +size = Domain.gradient_size(geometry)
  +weight_size = Domain.weight_size(geometry)
  +gradient_listed = listed_values(~Gradient,~gradient_value,~GradientRelated,~gradient_bound,~gradient_agrees,gradient_values,gradient_context,gradient_related,size,Nat.mul(image,size),
    Order.transitive(Nat.add(Nat.mul(image,size),size),Nat.mul(images,size),gradient_bound,TiledValues.row_room(image,images,size,Nat.mul(images,size),{==},image_inside),gradient_cover))
  +weight_listed = listed_values(~Weights,~weight_value,~WeightRelated,~weight_bound,~weight_agrees,weight_values,weight_context,weight_related,weight_size,0n,weight_cover)
  +wanted = problem(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,image)
  %Equal.sym(List<&2,F32>,ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,size,Nat.mul(image,size),gradient_values),
      Images.image_values(~Gradient,~gradient_value,size,Nat.mul(image,size),gradient_context),gradient_listed) :
    {Spec.Problem{Sources.shape(geometry),_,ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,weight_size,0n,weight_values)} == wanted : Spec.Problem}
  %Equal.sym(List<&2,F32>,ListRead.enumerate(~F32,~CellSources.Values,~CellSources.value_at,weight_size,0n,weight_values),
      Images.image_values(~Weights,~weight_value,weight_size,0n,weight_context),weight_listed) :
    {Spec.Problem{Sources.shape(geometry),Images.image_values(~Gradient,~gradient_value,size,Nat.mul(image,size),gradient_context),_} == wanted : Spec.Problem}
  {==}

# The setup of a run whose gradient and weights agree with the readers reads
# the input gradient of the readers.
def input_fixed(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  ~GradientRelated: CellSources.Values -> Gradient -> Data,~gradient_bound: Nat,
  ~gradient_agrees: @+values: CellSources.Values -> @+logical: Gradient -> @+related: GradientRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,gradient_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == gradient_value(logical,U32.from_nat(index)) : F32},
  ~WeightRelated: CellSources.Values -> Weights -> Data,~weight_bound: Nat,
  ~weight_agrees: @+values: CellSources.Values -> @+logical: Weights -> @+related: WeightRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,weight_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == weight_value(logical,U32.from_nat(index)) : F32},
  +gradient_context: Gradient,+weight_context: Weights,+call: Domain.Call,+gradient: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +gradient_related: GradientRelated(CellSources.gradient_source(CellSources.models_reader(Domain.geometry(call),gradient,weights)),gradient_context),
  +weight_related: WeightRelated(CellSources.weight_source(CellSources.models_reader(Domain.geometry(call),gradient,weights)),weight_context),
  +gradient_cover: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.gradient_size(Domain.geometry(call))),gradient_bound) == True{} : Bool},
  +weight_cover: {Nat.is_le(Domain.weight_size(Domain.geometry(call)),weight_bound) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call)))) == True{} : Bool}) ->
  {PieceCells.input_value(PieceCells.Setup{call,gradient,weights},index) ==
    input_at(~Gradient,~gradient_value,~Weights,~weight_value,InputRead{gradient_context,weight_context,Domain.geometry(call)},index) : F32}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  +point = Domain.point(geometry,U32.to_nat(index))
  +image = Domain.image(point)
  +reader = CellSources.models_reader(geometry,gradient,weights)
  +image_inside = DivisionOrder.quotient_below(U32.to_nat(index),batch,Domain.image_size(geometry),Domain.image_positive(Domain.Call{geometry,batch,limits,facts}),inside)
  Equal.cong(Spec.Problem,F32,formal => Spec.value(formal,Domain.channel(point),Domain.row(point),Domain.column(point)),
    CellSources.problem(geometry,reader,image),problem(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,image),
    problems_agree(~Gradient,~gradient_value,~Weights,~weight_value,~GradientRelated,~gradient_bound,~gradient_agrees,~WeightRelated,~weight_bound,~weight_agrees,
      gradient_context,weight_context,geometry,CellSources.gradient_source(reader),CellSources.weight_source(reader),gradient_related,weight_related,batch,gradient_cover,weight_cover,
      image,image_inside))

# A setup and a read related as the given ones agree below the values.
def input_agrees(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  ~GradientRelated: CellSources.Values -> Gradient -> Data,~gradient_bound: Nat,
  ~gradient_agrees: @+values: CellSources.Values -> @+logical: Gradient -> @+related: GradientRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,gradient_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == gradient_value(logical,U32.from_nat(index)) : F32},
  ~WeightRelated: CellSources.Values -> Weights -> Data,~weight_bound: Nat,
  ~weight_agrees: @+values: CellSources.Values -> @+logical: Weights -> @+related: WeightRelated(values,logical) -> @+index: Nat ->
    @+below: {Nat.is_lt(index,weight_bound) == True{} : Bool} -> {CellSources.value_at(values,index) == weight_value(logical,U32.from_nat(index)) : F32},
  +gradient_context: Gradient,+weight_context: Weights,+call: Domain.Call,+gradient: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,
  +gradient_related: GradientRelated(CellSources.gradient_source(CellSources.models_reader(Domain.geometry(call),gradient,weights)),gradient_context),
  +weight_related: WeightRelated(CellSources.weight_source(CellSources.models_reader(Domain.geometry(call),gradient,weights)),weight_context),
  +gradient_cover: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.gradient_size(Domain.geometry(call))),gradient_bound) == True{} : Bool},
  +weight_cover: {Nat.is_le(Domain.weight_size(Domain.geometry(call)),weight_bound) == True{} : Bool},
  +left: PieceCells.Setup,+right: InputRead<Gradient,Weights>,
  +related: Global.Fixed(~PieceCells.Setup,~InputRead<Gradient,Weights>,PieceCells.Setup{call,gradient,weights},InputRead{gradient_context,weight_context,Domain.geometry(call)},left,right),
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call)))) == True{} : Bool}) ->
  {PieceCells.input_value(left,index) == input_at(~Gradient,~gradient_value,~Weights,~weight_value,right,index) : F32}:
  KernelGemm.Both{+left_equation,+right_equation} = related
  %Equal.sym(PieceCells.Setup,left,PieceCells.Setup{call,gradient,weights},left_equation) :
    {PieceCells.input_value(_,index) == input_at(~Gradient,~gradient_value,~Weights,~weight_value,right,index) : F32}
  %Equal.sym(InputRead<Gradient,Weights>,right,InputRead{gradient_context,weight_context,Domain.geometry(call)},right_equation) :
    {PieceCells.input_value(PieceCells.Setup{call,gradient,weights},index) == input_at(~Gradient,~gradient_value,~Weights,~weight_value,_,index) : F32}
  input_fixed(~Gradient,~gradient_value,~Weights,~weight_value,~GradientRelated,~gradient_bound,~gradient_agrees,~WeightRelated,~weight_bound,~weight_agrees,
    gradient_context,weight_context,call,gradient,weights,gradient_related,weight_related,gradient_cover,weight_cover,index,inside)

# Values related to fixed values that hold a reader below a bound hold the
# reader below the bound.
def values_agree(~Logical: Data,~original: Logical -> U32 -> F32,~bound: Nat,~fixed_values: CellSources.Values,~fixed_logical: Logical,
  ~fixed: @+index: Nat -> @+below: {Nat.is_lt(index,bound) == True{} : Bool} -> {CellSources.value_at(fixed_values,index) == original(fixed_logical,U32.from_nat(index)) : F32},
  +values: CellSources.Values,+logical: Logical,+related: Global.Fixed(~CellSources.Values,~Logical,fixed_values,fixed_logical,values,logical),+index: Nat,
  +below: {Nat.is_lt(index,bound) == True{} : Bool}) ->
  {CellSources.value_at(values,index) == original(logical,U32.from_nat(index)) : F32}:
  KernelGemm.Both{+values_equation,+logical_equation} = related
  %Equal.sym(CellSources.Values,values,fixed_values,values_equation) : {CellSources.value_at(_,index) == original(logical,U32.from_nat(index)) : F32}
  %Equal.sym(Logical,logical,fixed_logical,logical_equation) : {CellSources.value_at(fixed_values,index) == original(_,U32.from_nat(index)) : F32}
  fixed(index,below)

# ---- The shapes of a call ----

# The cells the specification reads the output gradient through read it as
# the cells of its geometry.
def gradient_cells_value(+mode: Call.Mode,+geometry: Kernel.Geometry,+tree: Tree.Tree<Tile.Tile8>,+word: U32) ->
  {TiledValues.value_value(tree,CellSources.gradient_cells(geometry),word) ==
    TiledValues.value_value(tree,TiledPublic.geometry_of(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),word) : F32}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      Equal.cong(U32,F32,after => TiledValues.value_value(tree,Cells.Cells{0,(gh * after : U32),o,Tiled.groups(o),Grid.Innermost{}},word),gw,(gw * 1 : U32),
        Equal.sym(U32,(gw * 1 : U32),gw,Arithmetic.multiply_one(gw)))
    case Call.Batched{batch}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      Equal.cong(U32,F32,after => TiledValues.value_value(tree,Cells.Cells{0,(gh * after : U32),o,Tiled.groups(o),Grid.Innermost{}},word),gw,(gw * 1 : U32),
        Equal.sym(U32,(gw * 1 : U32),gw,Arithmetic.multiply_one(gw)))

# The cells the specification reads the weights through are the cells of
# their geometry.
def weight_cells_value(+geometry: Kernel.Geometry,+tree: Tree.Tree<Tile.Tile8>,+word: U32) ->
  {TiledValues.value_value(tree,CellSources.weight_cells(geometry),word) ==
    TiledValues.value_value(tree,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{}),word) : F32}:
  Kernel.Geometry{+c,+o,h,w,+k,s,p,gh,gw} = geometry
  +last = (k * 1 : U32)
  Equal.trans(F32,TiledValues.value_value(tree,CellSources.weight_cells(geometry),word),
    TiledValues.value_value(tree,Cells.Cells{(o * (k * last : U32) : U32),(k * last : U32),c,Tiled.groups(c),Grid.Outermost{}},word),
    TiledValues.value_value(tree,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{}),word),
    Equal.cong(U32,F32,value => TiledValues.value_value(tree,Cells.Cells{(o * (k * value : U32) : U32),(k * value : U32),c,Tiled.groups(c),Grid.Outermost{}},word),k,last,
      Equal.sym(U32,last,k,Arithmetic.multiply_one(k))),
    Equal.cong(U32,F32,value => TiledValues.value_value(tree,Cells.Cells{(value * (k * last : U32) : U32),(k * last : U32),c,Tiled.groups(c),Grid.Outermost{}},word),o,(1 * o : U32),
      Equal.sym(U32,(1 * o : U32),o,TiledPublic.one_times(o))))

# The gradient model of a holder of the lifted output gradient reads below the
# values the output gradient.
def gradient_fixed(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+limits: Machine.Limits,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +count: U32,+sized: Depth.Sized<count>,+counted: Global.Counted<Call.input_shape(mode,geometry)>,
  +room: {Nat.is_le(ShapeLaws.elements(Call.input_shape(mode,geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Call.input_shape(mode,geometry))) == True{} : Bool},
  +count_cover: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode)),Grid.Innermost{})),U32.to_nat(count)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Call.input_shape(mode,geometry))) == True{} : Bool}) ->
  {CellSources.value_at(CellSources.Values{Buffer.tree(~Tile.Tile8,BandSource.gradient_model(~Representation.Lifted<Gradient>,~Representation.lifted(~Gradient,~gradient_value),
      Representation.Lifted{gradient_context,TiledPublic.geometry_of(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode)),Grid.Innermost{})},count)),
      CellSources.gradient_cells(geometry)},index) == gradient_value(gradient_context,U32.from_nat(index)) : F32}:
  +shape = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  +holder = {Representation.Lifted{gradient_context,TiledPublic.geometry_of(shape,axis,Grid.Innermost{})} : Representation.Lifted<Gradient>}
  +tree = Buffer.tree(~Tile.Tile8,BandSource.gradient_model(~Representation.Lifted<Gradient>,~Representation.lifted(~Gradient,~gradient_value),holder,count))
  Equal.trans(F32,TiledValues.value_value(tree,CellSources.gradient_cells(geometry),U32.from_nat(index)),
    TiledValues.value_value(tree,TiledPublic.geometry_of(shape,axis,Grid.Innermost{}),U32.from_nat(index)),gradient_value(gradient_context,U32.from_nat(index)),
    gradient_cells_value(mode,geometry,tree,U32.from_nat(index)),
    TiledValues.geometry_value(~Gradient,~gradient_value,gradient_context,limits,tree,shape,axis,Grid.Innermost{},Call.gradient_within(mode,geometry),counted,room,nonempty,
      word => inside_word =>
        %Arithmetic.native_roundtrip(word) : {Tree.read(Tile.Tile8,tree,_) == Representation.lifted(~Gradient,~gradient_value,holder,_) : Tile.Tile8}
        BandSource.gradient_cell(~Representation.Lifted<Gradient>,~Representation.lifted(~Gradient,~gradient_value),holder,count,sized,U32.to_nat(word),
          Bounds.nat_trans(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(shape,axis,Grid.Innermost{})),U32.to_nat(count),inside_word,count_cover)),
      index,inside))

# A buffer that holds the lifted weights reads below the values the weights.
def weight_fixed(~Weights: Data,~weight_value: Weights -> U32 -> F32,+weight_context: Weights,+limits: Machine.Limits,+geometry: Kernel.Geometry,
  +source: Buffer.Model<Tile.Tile8>,+counted: Global.Counted<Call.weight_shape(geometry)>,
  +room: {Nat.is_le(ShapeLaws.elements(Call.weight_shape(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Call.weight_shape(geometry))) == True{} : Bool},
  +length_exact: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)) == ShapeLaws.elements(Tiled.cell_shape(Call.weight_shape(geometry),1n,Grid.Outermost{})) : Nat},
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),
    Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{})},Buffer.content(~Tile.Tile8,source)),
  +index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Call.weight_shape(geometry))) == True{} : Bool}) ->
  {CellSources.value_at(CellSources.Values{Buffer.tree(~Tile.Tile8,source),CellSources.weight_cells(geometry)},index) == weight_value(weight_context,U32.from_nat(index)) : F32}:
  +shape = Call.weight_shape(geometry)
  +holder = {Representation.Lifted{weight_context,TiledPublic.geometry_of(shape,1n,Grid.Outermost{})} : Representation.Lifted<Weights>}
  +tree = Buffer.tree(~Tile.Tile8,source)
  +content = Buffer.content(~Tile.Tile8,source)
  Equal.trans(F32,TiledValues.value_value(tree,CellSources.weight_cells(geometry),U32.from_nat(index)),
    TiledValues.value_value(tree,TiledPublic.geometry_of(shape,1n,Grid.Outermost{}),U32.from_nat(index)),weight_value(weight_context,U32.from_nat(index)),
    weight_cells_value(geometry,tree,U32.from_nat(index)),
    TiledValues.geometry_value(~Weights,~weight_value,weight_context,limits,tree,shape,1n,Grid.Outermost{},Call.weight_within(geometry),counted,room,nonempty,
      word => inside_word =>
        Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,word),Buffer.value(~Tile.Tile8,word,content),Representation.lifted(~Weights,~weight_value,holder,word),
          BandSource.model_value(source,word),
          %Arithmetic.native_roundtrip(word) : {Buffer.value(~Tile.Tile8,_,content) == Representation.lifted(~Weights,~weight_value,holder,_) : Tile.Tile8}
          Contract.read(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,content,U32.to_nat(word),values,
            %Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,content),ShapeLaws.elements(Tiled.cell_shape(shape,1n,Grid.Outermost{})),length_exact) :
              {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} inside_word)),
      index,inside))

# ---- The cells of a call ----

# A call that passes its checks on the cells of its operands.
def Accepted(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits) -> Data:
  {Gradient.input_valid_with(Call.arguments(mode,geometry),Tensors.count_or_zero(Call.gradient_cell_shape(mode,geometry)),Tensors.count_or_zero(Call.weight_cell_shape(geometry)),limits) == True{} : Bool}

def checked_call(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits)) -> Domain.Call:
  Call.call(Call.arguments(mode,geometry),Tensors.count_or_zero(Call.gradient_cell_shape(mode,geometry)),Tensors.count_or_zero(Call.weight_cell_shape(geometry)),limits,accepted)

# The input gradient of a run as tiled cells.
def input_cells_exact(+mode: Call.Mode,+geometry: Kernel.Geometry,+workers: U32,-output: Pieces.Pieces(~Tile.Tile8)) ->
  {Gradient.input_cells(Call.arguments(mode,geometry),workers,output) ==
    Tiled.Tiled{Tensors.Tensor{output,Gradient.input_cell_shape(Call.arguments(mode,geometry)),Tiled.zero_tile(),Tensors.Usable{},workers},Conv.lane_of(Call.batched(mode)),
      Domain.channel_word(geometry),Grid.Innermost{}} : Tiled.Tiled}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

# The cells of the input gradient lay out its values.
def image_laid(+mode: Call.Mode,+geometry: Kernel.Geometry,+counted: Global.Counted<Gradient.input_shape(Call.arguments(mode,geometry))>) ->
  TiledPublic.Laid<Gradient.input_cell_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Domain.channel_word(geometry),Grid.Innermost{},
    Gradient.input_shape(Call.arguments(mode,geometry))>:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      TiledPublic.Laid{{==},{==},{==},counted,{==}}
    case Call.Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      TiledPublic.Laid{{==},{==},{==},counted,{==}}

# The cells the run places the input gradient in are the cells of its
# geometry.
def image_geometry(+mode: Call.Mode,+geometry: Kernel.Geometry) ->
  {PieceCells.cells_of(geometry,Call.image_outer(mode,geometry),Call.image_after(geometry)) ==
    TiledPublic.geometry_of(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}) : Cells.Cells}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Call.Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The cells of the input gradient group its channels by eight.
def image_unit(+mode: Call.Mode,+geometry: Kernel.Geometry) ->
  GridRanges.Multiple<GridRanges.last_extent(Gradient.input_cell_shape(Call.arguments(mode,geometry))),U32.to_nat(Tiled.groups(Domain.channel_word(geometry)))>:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw} = geometry
      +groups = U32.to_nat(Tiled.groups(c))
      GridRanges.Multiple{1n,Equal.sym(Nat,Nat.add(groups,0n),groups,Addition.zero_right(groups))}
    case Call.Batched{batch}:
      Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw} = geometry
      +groups = U32.to_nat(Tiled.groups(c))
      GridRanges.Multiple{1n,Equal.sym(Nat,Nat.add(groups,0n),groups,Addition.zero_right(groups))}

def image_after_exact(+limits: Machine.Limits,+geometry: Kernel.Geometry,+outer: U32,
  +room: {Nat.is_le(Nat.mul(Domain.height(geometry),Domain.width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Nat.mul(Domain.height(geometry),Domain.width(geometry))) == True{} : Bool}) ->
  {Representation.after_count(PieceCells.cells_of(geometry,outer,Call.image_after(geometry))) == Nat.mul(Domain.height(geometry),Domain.width(geometry)) : Nat}:
  +plane = Nat.mul(Domain.height(geometry),Domain.width(geometry))
  %Equal.sym(Nat,U32.to_nat(Call.image_after(geometry)),plane,Call.image_after_word(limits,geometry,room)) : {Representation.least_one(_) == plane : Nat}
  Representation.least_positive(plane,positive)

# ---- Counts of the cells of a call ----

# Groups of eight of positive lanes within the capacity are at most the
# lanes and at least one.
def groups_below(+limits: Machine.Limits,+lanes: U32,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(lanes)) == True{} : Bool}:
  +count = U32.to_nat(lanes)
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(lanes)),Ceiling.ceiling(count,8n),Encoded.eight_blocks_exact(limits,lanes,bound)) : {Nat.is_le(_,count) == True{} : Bool}
  Ceiling.minimal(count,8n,count,{==},RowMajor.product_covers(count,8n,{==}))

def groups_positive(+limits: Machine.Limits,+lanes: U32,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}:
  +count = U32.to_nat(lanes)
  +blocks = Ceiling.ceiling(count,8n)
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(lanes)),blocks,Encoded.eight_blocks_exact(limits,lanes,bound)) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(blocks,8n,Bounds.nat_trans(0n,count,Nat.mul(blocks,8n),positive,Ceiling.covers(count,8n,{==})))

# Cells of fewer groups than lanes are fewer than the values.
def image_cells_below(+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+groups: Nat,+fewer: {Nat.is_le(groups,channels) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(images,height),Nat.mul(width,groups)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width)))) == True{} : Bool}:
  +front = Nat.mul(images,height)
  +exact = Equal.trans(Nat,Nat.mul(front,Nat.mul(width,channels)),Nat.mul(images,Nat.mul(height,Nat.mul(width,channels))),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),
    RowMajor.assoc(images,height,Nat.mul(width,channels)),
    Equal.cong(Nat,Nat,value => Nat.mul(images,value),Nat.mul(height,Nat.mul(width,channels)),Nat.mul(channels,Nat.mul(height,width)),
      Equal.trans(Nat,Nat.mul(height,Nat.mul(width,channels)),Nat.mul(Nat.mul(height,width),channels),Nat.mul(channels,Nat.mul(height,width)),
        Equal.sym(Nat,Nat.mul(Nat.mul(height,width),channels),Nat.mul(height,Nat.mul(width,channels)),RowMajor.assoc(height,width,channels)),
        RowMajor.multiply_commutative(Nat.mul(height,width),channels))))
  Order.transitive(Nat.mul(front,Nat.mul(width,groups)),Nat.mul(front,Nat.mul(width,channels)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),
    RowMajor.mul_le_left(front,Nat.mul(width,groups),Nat.mul(width,channels),RowMajor.mul_le_left(width,groups,channels,fewer)),
    %exact : {Nat.is_le(Nat.mul(front,Nat.mul(width,channels)),_) == True{} : Bool} Order.reflexive(Nat.mul(front,Nat.mul(width,channels))))

def gradient_cells_below(+images: Nat,+outputs: Nat,+rows: Nat,+columns: Nat,+groups: Nat,+fewer: {Nat.is_le(groups,outputs) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,rows),columns),groups),Nat.mul(images,Nat.mul(outputs,Nat.mul(rows,columns)))) == True{} : Bool}:
  +front = Nat.mul(Nat.mul(images,rows),columns)
  +exact = Equal.trans(Nat,Nat.mul(front,outputs),Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),outputs),Nat.mul(images,Nat.mul(outputs,Nat.mul(rows,columns))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,outputs),front,Nat.mul(images,Nat.mul(rows,columns)),RowMajor.assoc(images,rows,columns)),
    Equal.trans(Nat,Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),outputs),Nat.mul(images,Nat.mul(Nat.mul(rows,columns),outputs)),Nat.mul(images,Nat.mul(outputs,Nat.mul(rows,columns))),
      RowMajor.assoc(images,Nat.mul(rows,columns),outputs),
      Equal.cong(Nat,Nat,value => Nat.mul(images,value),Nat.mul(Nat.mul(rows,columns),outputs),Nat.mul(outputs,Nat.mul(rows,columns)),
        RowMajor.multiply_commutative(Nat.mul(rows,columns),outputs))))
  Order.transitive(Nat.mul(front,groups),Nat.mul(front,outputs),Nat.mul(images,Nat.mul(outputs,Nat.mul(rows,columns))),RowMajor.mul_le_left(front,groups,outputs,fewer),
    %exact : {Nat.is_le(Nat.mul(front,outputs),_) == True{} : Bool} Order.reflexive(Nat.mul(front,outputs)))

def weight_cells_below(+groups: Nat,+outputs: Nat,+channels: Nat,+kernel: Nat,+fewer: {Nat.is_le(groups,channels) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(groups,Nat.mul(outputs,Nat.mul(kernel,Nat.mul(kernel,1n)))),Nat.mul(outputs,Nat.mul(channels,Nat.mul(kernel,kernel)))) == True{} : Bool}:
  +area = Nat.mul(kernel,Nat.mul(kernel,1n))
  +swapped = {%Equal.sym(Nat,Nat.mul(groups,Nat.mul(outputs,area)),Nat.mul(outputs,Nat.mul(groups,area)),RowMajor.swap_factors(groups,outputs,area)) :
      {Nat.is_le(_,Nat.mul(outputs,Nat.mul(channels,area))) == True{} : Bool}
    RowMajor.mul_le_left(outputs,Nat.mul(groups,area),Nat.mul(channels,area),RowMajor.mul_le(groups,channels,area,fewer)) :
    {Nat.is_le(Nat.mul(groups,Nat.mul(outputs,area)),Nat.mul(outputs,Nat.mul(channels,area))) == True{} : Bool}}
  %Order.times_one(kernel) : {Nat.is_le(Nat.mul(groups,Nat.mul(outputs,area)),Nat.mul(outputs,Nat.mul(channels,Nat.mul(kernel,_)))) == True{} : Bool}
  swapped

def formed_length(+shape: List<&2,U32>,+content: Buffer.Content<Tile.Tile8>,+formed: Contract.WellFormed<Tile.Tile8,shape,content>) ->
  {Buffer.content_length(~Tile.Tile8,content) == ShapeLaws.elements(shape) : Nat}:
  Contract.WellFormed{+length_exact,+ready} = formed
  length_exact

# ---- The results of a call ----

# The output gradient has rows and columns.
def rows_positive(+call: Domain.Call) -> {Nat.is_lt(0n,Domain.gradient_height(Domain.geometry(call))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  %Equal.sym(Nat,Domain.gradient_height(geometry),KernelShape.axis_size(Domain.height(geometry),Domain.kernel(geometry),Domain.stride(geometry),Domain.padding(geometry)),
    Domain.rows_exact(call)) : {Nat.is_lt(0n,_) == True{} : Bool}
  {==}

def columns_positive(+call: Domain.Call) -> {Nat.is_lt(0n,Domain.gradient_width(Domain.geometry(call))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  %Equal.sym(Nat,Domain.gradient_width(geometry),KernelShape.axis_size(Domain.width(geometry),Domain.kernel(geometry),Domain.stride(geometry),Domain.padding(geometry)),
    Domain.columns_exact(call)) : {Nat.is_lt(0n,_) == True{} : Bool}
  {==}

# The input gradient a run of a call writes.
def run_output(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits),+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),
  +source: Buffer.Model<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  windowed_of(checked_call(mode,geometry,limits,accepted),Gradient.input_cell_shape(Call.arguments(mode,geometry)),Tensors.grid_of(Call.gradient_cell_shape(mode,geometry)),
    Gradient.input_whole(Call.arguments(mode,geometry),Tensors.grid_of(Call.gradient_cell_shape(mode,geometry))),workers,gradient,source)

# The cells of the input gradient of a call that passes its checks fit the
# capacity and hold values.
def image_bound_cells(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits)) ->
  {Nat.is_le(ShapeLaws.elements(Gradient.input_cell_shape(Call.arguments(mode,geometry))),Nat.mul(Nat.mul(U32.to_nat(Call.images(mode)),Domain.height(geometry)),BatchRows.row_size(checked_call(mode,geometry,limits,accepted)))) == True{} : Bool}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +cells = Gradient.input_cell_shape(Call.arguments(mode,g))
  +images = U32.to_nat(Call.images(mode))
  +total = Nat.mul(Nat.mul(images,U32.to_nat(h)),Nat.mul(U32.to_nat(w),Ceiling.ceiling(U32.to_nat(c),8n)))
  +exact = {%Encoded.eight_blocks_exact(limits,c,Domain.channels_capacity_room(checked_call(mode,g,limits,accepted))) :
      {ShapeLaws.elements(cells) == Nat.mul(Nat.mul(images,U32.to_nat(h)),Nat.mul(U32.to_nat(w),_)) : Nat} Call.image_cells_count(mode,g) :
    {ShapeLaws.elements(cells) == total : Nat}}
  %Equal.sym(Nat,ShapeLaws.elements(cells),total,exact) : {Nat.is_le(_,total) == True{} : Bool}
  Order.reflexive(total)

def image_total_room(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits)) -> {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(Call.images(mode)),Domain.height(geometry)),BatchRows.row_size(checked_call(mode,geometry,limits,accepted))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +images = U32.to_nat(Call.images(mode))
  +channels = U32.to_nat(c)
  +rows = U32.to_nat(h)
  +columns = U32.to_nat(w)
  +blocks = Ceiling.ceiling(channels,8n)
  +values_room = Call.values_room(Call.arguments(mode,g),Tensors.count_or_zero(Call.gradient_cell_shape(mode,g)),Tensors.count_or_zero(Call.weight_cell_shape(g)),limits,accepted)
  Order.transitive(Nat.mul(Nat.mul(images,rows),Nat.mul(columns,blocks)),Nat.mul(images,Nat.mul(channels,Nat.mul(rows,columns))),U32.to_nat(Machine.capacity(limits)),
    image_cells_below(images,channels,rows,columns,blocks,Ceiling.minimal(channels,8n,channels,{==},RowMajor.product_covers(channels,8n,{==}))),values_room)

def image_room(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits)) -> {Nat.is_le(ShapeLaws.elements(Gradient.input_cell_shape(Call.arguments(mode,geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(ShapeLaws.elements(Gradient.input_cell_shape(Call.arguments(mode,geometry))),Nat.mul(Nat.mul(U32.to_nat(Call.images(mode)),Domain.height(geometry)),BatchRows.row_size(checked_call(mode,geometry,limits,accepted))),U32.to_nat(Machine.capacity(limits)),image_bound_cells(mode,geometry,limits,accepted),image_total_room(mode,geometry,limits,accepted))

def image_nonempty(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Gradient.input_cell_shape(Call.arguments(mode,geometry)))) == True{} : Bool}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +call = checked_call(mode,g,limits,accepted)
  +images = U32.to_nat(Call.images(mode))
  +rows = U32.to_nat(h)
  +columns = U32.to_nat(w)
  +groups = U32.to_nat(Tiled.groups(c))
  +total = Nat.mul(Nat.mul(images,rows),Nat.mul(columns,groups))
  %Equal.sym(Nat,ShapeLaws.elements(Gradient.input_cell_shape(Call.arguments(mode,g))),total,Call.image_cells_count(mode,g)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(Nat.mul(images,rows),Nat.mul(columns,groups),Order.product_positive(images,rows,images_positive,Domain.height_positive(call)),
    Order.product_positive(columns,groups,Domain.width_positive(call),groups_positive(limits,c,Domain.channels_capacity_room(call),Domain.channels_positive(call))))

# The output gradient and the weights of a call that passes its checks hold
# values.
def gradient_values_nonempty(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Call.input_shape(mode,geometry))) == True{} : Bool}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +call = checked_call(mode,g,limits,accepted)
  +plane = Nat.mul(U32.to_nat(gh),U32.to_nat(gw))
  %Call.gradient_count(mode,g) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(Call.images(mode)),Nat.mul(U32.to_nat(o),plane),images_positive,
    Order.product_positive(U32.to_nat(o),plane,Domain.outputs_positive(call),Order.product_positive(U32.to_nat(gh),U32.to_nat(gw),rows_positive(call),columns_positive(call))))

def weight_values_nonempty(+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+accepted: Accepted(mode,geometry,limits)) -> {Nat.is_lt(0n,ShapeLaws.elements(Call.weight_shape(geometry))) == True{} : Bool}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +call = checked_call(mode,g,limits,accepted)
  +kernel = U32.to_nat(k)
  +area = Nat.mul(kernel,kernel)
  +kernel_positive = Domain.kernel_positive(call)
  %Call.weight_count(g) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(o),Nat.mul(U32.to_nat(c),area),Domain.outputs_positive(call),
    Order.product_positive(U32.to_nat(c),area,Domain.channels_positive(call),Order.product_positive(kernel,kernel,kernel_positive,kernel_positive)))

# The input gradient a run writes from the cells of the output gradient and
# from weights that hold their reader holds the specification of the readers.
def input_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}}),
  +source: Buffer.Model<Tile.Tile8>,+w_counted: Global.Counted<Call.weight_shape(geometry)>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)) == ShapeLaws.elements(Call.weight_cell_shape(geometry)) : Nat},
  +w_values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),
    Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{})},Buffer.content(~Tile.Tile8,source)),
  +workers: U32) ->
  TiledPublic.Contract(~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
    Gradient.input_shape(Call.arguments(mode,geometry)),
    Tiled.Tiled{Tensors.Tensor{run_output(mode,geometry,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),Gradient.input_cell_shape(Call.arguments(mode,geometry)),
      Tiled.zero_tile(),Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),Domain.channel_word(geometry),Grid.Innermost{}}):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  KernelGemm.Both{+d_global,TiledPublic.Laid{+d_inside,+d_shaped,+d_restored,+d_counted,+d_pinned}} = d_contract
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +arguments = Call.arguments(mode,g)
  +call = checked_call(mode,g,limits,accepted)
  +dshape = Call.input_shape(mode,g)
  +axis = Conv.lane_of(Call.batched(mode))
  +dcells = Call.gradient_cell_shape(mode,g)
  +wshape = Call.weight_shape(g)
  +cells = Gradient.input_cell_shape(arguments)
  +values = Gradient.input_shape(arguments)
  +images = U32.to_nat(Call.images(mode))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +dc = Tensors.count_or_zero(dcells)
  +wc = Tensors.count_or_zero(Call.weight_cell_shape(g))
  +channels = U32.to_nat(c)
  +outputs = U32.to_nat(o)
  +rows = U32.to_nat(h)
  +columns = U32.to_nat(w)
  +gradient_rows = U32.to_nat(gh)
  +gradient_columns = U32.to_nat(gw)
  +plane = Nat.mul(rows,columns)
  +image = Nat.mul(channels,plane)
  +gradient_plane = Nat.mul(gradient_rows,gradient_columns)
  +holder = {Representation.Lifted{gradient_context,TiledPublic.geometry_of(dshape,axis,Grid.Innermost{})} : Representation.Lifted<Gradient>}
  +own_contract = {KernelGemm.Both{d_global,TiledPublic.Laid{d_inside,d_shaped,d_restored,d_counted,d_pinned}} :
    TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,dshape,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,Grid.lanes_at(dshape,axis),Grid.Innermost{}})}
  +shaped = TiledPublic.cells_shaped(~Gradient,~gradient_value,gradient_context,dshape,axis,Grid.lanes_at(dshape,axis),Grid.Innermost{},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},own_contract)
  +values_room = Call.values_room(arguments,dc,wc,limits,accepted)
  +plane_room = Domain.plane_room(call)
  +rows_positive_value = Domain.height_positive(call)
  +columns_positive_value = Domain.width_positive(call)
  +channels_positive = Domain.channels_positive(call)
  +plane_positive = Order.product_positive(rows,columns,rows_positive_value,columns_positive_value)
  +plane_below = {%RowMajor.multiply_commutative(plane,channels) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,channels,channels_positive) :
    {Nat.is_le(plane,image) == True{} : Bool}}
  +outer_room = Order.transitive(Nat.mul(images,plane),Nat.mul(images,image),capacity,RowMajor.mul_le_left(images,plane,image,plane_below),values_room)
  +after_exact = image_after_exact(limits,g,Call.image_outer(mode,g),plane_room,plane_positive)
  +outer_exact = Call.image_outer_exact(limits,mode,g,plane_room,outer_room)
  +total_room = image_total_room(mode,g,limits,accepted)
  +bound_cells = image_bound_cells(mode,g,limits,accepted)
  +room = image_room(mode,g,limits,accepted)
  +nonempty = image_nonempty(mode,g,limits,accepted,images_positive)
  +counted = StorageCount.counted_within(limits,cells,room,default_capacity)
  +gradient_rows_positive = rows_positive(call)
  +gradient_columns_positive = columns_positive(call)
  +gradient_plane_positive = Order.product_positive(gradient_rows,gradient_columns,gradient_rows_positive,gradient_columns_positive)
  +outputs_positive = Domain.outputs_positive(call)
  +outputs_room = Order.transitive(outputs,Nat.mul(outputs,gradient_plane),capacity,RowMajor.product_covers(outputs,gradient_plane,gradient_plane_positive),Domain.gradient_image_room(call))
  +output_groups = U32.to_nat(Tiled.groups(o))
  +d_total = Nat.mul(Nat.mul(Nat.mul(images,gradient_rows),gradient_columns),output_groups)
  +d_exact = Call.gradient_cells_count(mode,g)
  +d_room = Order.transitive(ShapeLaws.elements(dcells),Nat.mul(images,Nat.mul(outputs,gradient_plane)),capacity,
    %Equal.sym(Nat,ShapeLaws.elements(dcells),d_total,d_exact) : {Nat.is_le(_,Nat.mul(images,Nat.mul(outputs,gradient_plane))) == True{} : Bool}
      gradient_cells_below(images,outputs,gradient_rows,gradient_columns,output_groups,groups_below(limits,o,outputs_room)),
    Domain.gradient_room(call))
  +d_nonempty = {%Equal.sym(Nat,ShapeLaws.elements(dcells),d_total,d_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(Nat.mul(Nat.mul(images,gradient_rows),gradient_columns),output_groups,
      Order.product_positive(Nat.mul(images,gradient_rows),gradient_columns,Order.product_positive(images,gradient_rows,images_positive,gradient_rows_positive),gradient_columns_positive),
      groups_positive(limits,o,outputs_room,outputs_positive)) : {Nat.is_lt(0n,ShapeLaws.elements(dcells)) == True{} : Bool}}
  +global = output_contract(~Representation.Lifted<Gradient>,~Representation.lifted(~Gradient,~gradient_value),holder,call,Geometry.word_envelope(),Call.fits(arguments,dc,wc,limits,accepted),
    source,Call.image_outer(mode,g),Call.image_after(g),after_exact,outer_exact,values_room,total_room,cells,room,nonempty,U32.to_nat(Tiled.groups(c)),image_unit(mode,g),counted,bound_cells,
    workers,Gradient.input_whole(arguments,Tensors.grid_of(dcells)),Tiled.zero_tile(),dw,dcells,PiecesModel.pack(~Tile.Tile8,dm),df,Tensors.Usable{},dw,shaped,d_room,d_nonempty,d_exact)
  +gradient_model = BandSource.gradient_model(~Representation.Lifted<Gradient>,~Representation.lifted(~Gradient,~gradient_value),holder,Rows.cells_word(call))
  +setup = {PieceCells.Setup{call,gradient_model,source} : PieceCells.Setup}
  +placed = {%image_geometry(mode,g) : Global.Contract(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      Representation.Lifted{setup,_},Tensors.Tensor{run_output(mode,g,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),cells,Tiled.zero_tile(),Tensors.Usable{},dw}) global :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      Representation.Lifted{setup,TiledPublic.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{run_output(mode,g,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),cells,Tiled.zero_tile(),Tensors.Usable{},dw})}
  +values_inside = {%Call.image_count(mode,g) : {Nat.is_le(_,capacity) == True{} : Bool} values_room : {Nat.is_le(ShapeLaws.elements(values),capacity) == True{} : Bool}}
  +laid = image_laid(mode,g,StorageCount.counted_within(limits,values,values_inside,default_capacity))
  +setup_contract = {KernelGemm.Both{placed,laid} : TiledPublic.Contract(~PieceCells.Setup,~PieceCells.input_value,setup,values,
    Tiled.Tiled{Tensors.Tensor{run_output(mode,g,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),cells,Tiled.zero_tile(),Tensors.Usable{},dw},axis,c,Grid.Innermost{}})}
  +read = {InputRead{gradient_context,weight_context,g} : InputRead<Gradient,Weights>}
  +gradient_values = {CellSources.Values{Buffer.tree(~Tile.Tile8,gradient_model),CellSources.gradient_cells(g)} : CellSources.Values}
  +weight_values = {CellSources.Values{Buffer.tree(~Tile.Tile8,source),CellSources.weight_cells(g)} : CellSources.Values}
  +count = Rows.cells_count(call)
  +count_room = {%d_exact : {Nat.is_le(_,capacity) == True{} : Bool} d_room : {Nat.is_le(count,capacity) == True{} : Bool}}
  +word_exact = Bounds.roundtrip(count,Machine.capacity(limits),Limits.capacity_fits(limits),count_room)
  +sized = Depth.bounded(Rows.cells_word(call),Arithmetic.count_within(Rows.cells_word(call),count,Machine.capacity(limits),1073741824,word_exact,count_room,Limits.capacity_allocatable(limits)))
  +count_cover = {%Equal.sym(Nat,U32.to_nat(Rows.cells_word(call)),count,word_exact) : {Nat.is_le(ShapeLaws.elements(dcells),_) == True{} : Bool}
    %d_exact : {Nat.is_le(ShapeLaws.elements(dcells),_) == True{} : Bool} Order.reflexive(ShapeLaws.elements(dcells)) :
    {Nat.is_le(ShapeLaws.elements(dcells),U32.to_nat(Rows.cells_word(call))) == True{} : Bool}}
  +gradient_size = Nat.mul(outputs,gradient_plane)
  +weight_size = Nat.mul(outputs,Nat.mul(channels,Nat.mul(U32.to_nat(k),U32.to_nat(k))))
  +d_values_room = {%Call.gradient_count(mode,g) : {Nat.is_le(_,capacity) == True{} : Bool} Domain.gradient_room(call) : {Nat.is_le(ShapeLaws.elements(dshape),capacity) == True{} : Bool}}
  +d_values_nonempty = gradient_values_nonempty(mode,g,limits,accepted,images_positive)
  +w_values_room = {%Call.weight_count(g) : {Nat.is_le(_,capacity) == True{} : Bool} Domain.weight_room(call) : {Nat.is_le(ShapeLaws.elements(wshape),capacity) == True{} : Bool}}
  +w_values_nonempty = weight_values_nonempty(mode,g,limits,accepted)
  +gradient_cover = {%Call.gradient_count(mode,g) : {Nat.is_le(Nat.mul(images,gradient_size),_) == True{} : Bool} Order.reflexive(Nat.mul(images,gradient_size)) :
    {Nat.is_le(Nat.mul(images,gradient_size),ShapeLaws.elements(dshape)) == True{} : Bool}}
  +weight_cover = {%Call.weight_count(g) : {Nat.is_le(weight_size,_) == True{} : Bool} Order.reflexive(weight_size) : {Nat.is_le(weight_size,ShapeLaws.elements(wshape)) == True{} : Bool}}
  +covered = {%Call.image_count(mode,g) : {Nat.is_le(_,Nat.mul(images,image)) == True{} : Bool} Order.reflexive(Nat.mul(images,image)) :
    {Nat.is_le(ShapeLaws.elements(values),Nat.mul(images,image)) == True{} : Bool}}
  +positive = {%Call.image_count(mode,g) : {Nat.is_lt(0n,_) == True{} : Bool} Order.product_positive(images,image,images_positive,Domain.image_positive(call)) :
    {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}}
  TiledViewReads.contract_below(~PieceCells.Setup,~InputRead<Gradient,Weights>,~PieceCells.input_value,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),
    ~Global.Fixed(~PieceCells.Setup,~InputRead<Gradient,Weights>,setup,read),~Nat.mul(images,image),
    ~input_agrees(~Gradient,~gradient_value,~Weights,~weight_value,
      ~Global.Fixed(~CellSources.Values,~Gradient,gradient_values,gradient_context),~ShapeLaws.elements(dshape),
      ~values_agree(~Gradient,~gradient_value,~ShapeLaws.elements(dshape),~gradient_values,~gradient_context,
        ~gradient_fixed(~Gradient,~gradient_value,gradient_context,limits,mode,g,Rows.cells_word(call),sized,d_counted,d_values_room,d_values_nonempty,count_cover)),
      ~Global.Fixed(~CellSources.Values,~Weights,weight_values,weight_context),~ShapeLaws.elements(wshape),
      ~values_agree(~Weights,~weight_value,~ShapeLaws.elements(wshape),~weight_values,~weight_context,
        ~weight_fixed(~Weights,~weight_value,weight_context,limits,g,source,w_counted,w_values_room,w_values_nonempty,length_exact,w_values)),
      gradient_context,weight_context,call,gradient_model,source,KernelGemm.Both{{==},{==}},KernelGemm.Both{{==},{==}},gradient_cover,weight_cover),
    limits,setup,read,KernelGemm.Both{{==},{==}},values,positive,values_inside,covered,
    Tiled.Tiled{Tensors.Tensor{run_output(mode,g,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),cells,Tiled.zero_tile(),Tensors.Usable{},dw},axis,c,Grid.Innermost{}},setup_contract)

# The weights a run returns hold their reader.
def weights_contract(~Weights: Data,~weight_value: Weights -> U32 -> F32,+weight_context: Weights,+mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,
  +accepted: Accepted(mode,geometry,limits),+wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),
    Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{})},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww}),
  +laid: TiledPublic.Laid<Call.weight_cell_shape(geometry),1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{},Call.weight_shape(geometry)>,
  +source: Buffer.Model<Tile.Tile8>,
  +formed: Contract.WellFormed<Tile.Tile8,Call.weight_cell_shape(geometry),Buffer.content(~Tile.Tile8,source)>,
  +w_values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),
    Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{})},Buffer.content(~Tile.Tile8,source))) ->
  TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Gradient.kept_back(Gradient.Kept{Call.weight_cell_shape(geometry),Grid.lanes_at(Call.weight_shape(geometry),1n),wf,Tensors.Usable{},ww},Buffer.pack(~Tile.Tile8,source))):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +call = checked_call(mode,g,limits,accepted)
  +wcells = Call.weight_cell_shape(g)
  +holder = {Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(g),1n,Grid.Outermost{})} : Representation.Lifted<Weights>}
  +capacity = U32.to_nat(Machine.capacity(limits))
  +channels = U32.to_nat(c)
  +outputs = U32.to_nat(o)
  +kernel = U32.to_nat(k)
  +groups = U32.to_nat(Tiled.groups(c))
  +channels_room = Domain.channels_capacity_room(call)
  +kernel_positive = Domain.kernel_positive(call)
  +room = Order.transitive(ShapeLaws.elements(wcells),Nat.mul(outputs,Nat.mul(channels,Nat.mul(kernel,kernel))),capacity,
    weight_cells_below(groups,outputs,channels,kernel,groups_below(limits,c,channels_room)),Domain.weight_room(call))
  +nonempty = Order.product_positive(groups,Nat.mul(outputs,Nat.mul(kernel,Nat.mul(kernel,1n))),groups_positive(limits,c,channels_room,Domain.channels_positive(call)),
    Order.product_positive(outputs,Nat.mul(kernel,Nat.mul(kernel,1n)),Domain.outputs_positive(call),
      Order.product_positive(kernel,Nat.mul(kernel,1n),kernel_positive,Order.product_positive(kernel,1n,kernel_positive,{==}))))
  +counted = Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,PiecesModel.pack(~Tile.Tile8,wm),wcells,wf,ww,w_global)
  +facts = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,source)),Buffer.content(~Tile.Tile8,source),Buffer.packed_contents(~Tile.Tile8,source)) :
      Contract.Facts(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,wcells,_) KernelGemm.Both{formed,w_values} :
    Contract.Facts(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,wcells,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,source)))}
  KernelGemm.Both{WholeValues.whole_contract(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,limits,wcells,wf,ww,source,counted,room,nonempty,facts),laid}

# The results of a run of a call that passes its checks hold the
# specification of the readers of its operands.
def outcome_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}}),
  +wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}}),
  +source: Buffer.Model<Tile.Tile8>,
  +source_exact: {Buffer.pack(~Tile.Tile8,source) == Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(Call.weight_cell_shape(geometry)),wf,PiecesModel.pack(~Tile.Tile8,wm)) :
    Storage.Buffer<Tile.Tile8>},
  +workers: U32) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    (Gradient.input_cells(Call.arguments(mode,geometry),dw,run_output(mode,geometry,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source)),
      Gradient.kept_back(Gradient.Kept{Call.weight_cell_shape(geometry),Grid.lanes_at(Call.weight_shape(geometry),1n),wf,Tensors.Usable{},ww},Buffer.pack(~Tile.Tile8,source)))):
  KernelGemm.Both{+w_global,TiledPublic.Laid{+w_inside,+w_shaped,+w_restored,+w_counted,+w_pinned}} = w_contract
  +wcells = Call.weight_cell_shape(geometry)
  +holder = {Representation.Lifted{weight_context,TiledPublic.geometry_of(Call.weight_shape(geometry),1n,Grid.Outermost{})} : Representation.Lifted<Weights>}
  +content = Buffer.content(~Tile.Tile8,source)
  +facts = Joined.joined_facts(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,wm,wcells,wf,ww,w_global,source,source_exact)
  +formed = Held.first_of(Contract.WellFormed<Tile.Tile8,wcells,content>,Contract.Values(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,content),facts)
  +w_values = Held.second_of(Contract.WellFormed<Tile.Tile8,wcells,content>,Contract.Values(~Tile.Tile8,~Representation.Lifted<Weights>,~Representation.lifted(~Weights,~weight_value),holder,content),facts)
  %Equal.sym(Tiled.Tiled,Gradient.input_cells(Call.arguments(mode,geometry),dw,run_output(mode,geometry,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source)),
    Tiled.Tiled{Tensors.Tensor{run_output(mode,geometry,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source),Gradient.input_cell_shape(Call.arguments(mode,geometry)),Tiled.zero_tile(),Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Domain.channel_word(geometry),Grid.Innermost{}},input_cells_exact(mode,geometry,dw,run_output(mode,geometry,limits,accepted,workers,PiecesModel.pack(~Tile.Tile8,dm),source))) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      (_,Gradient.kept_back(Gradient.Kept{wcells,Grid.lanes_at(Call.weight_shape(geometry),1n),wf,Tensors.Usable{},ww},Buffer.pack(~Tile.Tile8,source))))
  KernelGemm.Both{
    input_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,default_capacity,accepted,images_positive,dm,df,dw,d_contract,
      source,w_counted,formed_length(wcells,content,formed),w_values,workers),
    weights_contract(~Weights,~weight_value,weight_context,mode,geometry,limits,accepted,wm,wf,ww,w_global,TiledPublic.Laid{w_inside,w_shaped,w_restored,w_counted,w_pinned},
      source,formed,w_values)}

# ---- The call ----

# The output gradient read as an input shape, its lane axis and its rank.
def shape_known(+mode: Call.Mode,+geometry: Kernel.Geometry) ->
  {Conv.input_shape(Call.input_shape(mode,geometry)) ==
    Some{Conv.InputShape{Call.images(mode),Call.batched(mode),Call.outputs_word(geometry),Domain.gradient_rows(geometry),Domain.gradient_columns(geometry)}} : Maybe<&2,Conv.InputShape>}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Call.Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# A call whose run passes its checks returns the input gradient and the
# weights of its run.
def checked_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}}),
  +wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}}),
  +kept: Buffer.Model<Tile.Tile8>,
  +kept_exact: {Buffer.pack(~Tile.Tile8,kept) == Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(Call.weight_cell_shape(geometry)),wf,PiecesModel.pack(~Tile.Tile8,wm)) :
    Storage.Buffer<Tile.Tile8>}) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_checked(True{},Call.arguments(mode,geometry),Call.gradient_cell_shape(mode,geometry),PiecesModel.pack(~Tile.Tile8,dm),dw,
      Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(Call.weight_cell_shape(geometry)),wf,PiecesModel.pack(~Tile.Tile8,wm)),
      Gradient.Kept{Call.weight_cell_shape(geometry),Grid.lanes_at(Call.weight_shape(geometry),1n),wf,Tensors.Usable{},ww})):
  +arguments = Call.arguments(mode,geometry)
  +call = checked_call(mode,geometry,limits,accepted)
  +dcells = Call.gradient_cell_shape(mode,geometry)
  +cells = Gradient.input_cell_shape(arguments)
  +grid = Tensors.grid_of(dcells)
  +whole = Gradient.input_whole(arguments,grid)
  +workers = U32.max(dw,U32.max(ww,1))
  +kept_info = {Gradient.Kept{Call.weight_cell_shape(geometry),Grid.lanes_at(Call.weight_shape(geometry),1n),wf,Tensors.Usable{},ww} : Gradient.Kept}
  %kept_exact : TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_checked(True{},arguments,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,_,kept_info))
  %Equal.sym(Kernel.Outcome,Kernel.run(geometry,cells,grid,whole,workers,PiecesModel.pack(~Tile.Tile8,dm),Buffer.pack(~Tile.Tile8,kept)),
      Kernel.Outcome{Buffer.pack(~Tile.Tile8,kept),windowed_of(call,cells,grid,whole,workers,PiecesModel.pack(~Tile.Tile8,dm),kept)},
      run_exact(call,cells,image_room(mode,geometry,limits,accepted),image_nonempty(mode,geometry,limits,accepted,images_positive),grid,whole,workers,PiecesModel.pack(~Tile.Tile8,dm),kept)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,Gradient.input_result(arguments,dw,kept_info,_))
  outcome_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,default_capacity,accepted,images_positive,
    dm,df,dw,d_contract,wm,wf,ww,w_contract,kept,kept_exact,workers)

# The cells of a call parse as its shapes and pass its checks.
def geometry_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}}),
  +wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}}),
  +observation: Buffer.Observation<Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(Call.weight_cell_shape(geometry)),wf,PiecesModel.pack(~Tile.Tile8,wm))>) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_aligned(
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
        Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  KernelGemm.Both{+d_global,TiledPublic.Laid{+d_inside,+d_shaped,+d_restored,+d_counted,+d_pinned}} = d_contract
  KernelGemm.Both{+w_global,TiledPublic.Laid{+w_inside,+w_shaped,+w_restored,+w_counted,+w_pinned}} = w_contract
  Buffer.Observation{+kept,+kept_exact} = observation
  +g = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +arguments = Call.arguments(mode,g)
  +dshape = Call.input_shape(mode,g)
  +axis = Conv.lane_of(Call.batched(mode))
  +dcells = Call.gradient_cell_shape(mode,g)
  +wshape = Call.weight_shape(g)
  +wcells = Call.weight_cell_shape(g)
  +d_lanes = Grid.lanes_at(dshape,axis)
  +w_lanes = Grid.lanes_at(wshape,1n)
  +dc = Tensors.count_or_zero(dcells)
  +wc = Tensors.count_or_zero(wcells)
  +kept_info = {Gradient.Kept{wcells,w_lanes,wf,Tensors.Usable{},ww} : Gradient.Kept}
  +d_own = {KernelGemm.Both{d_global,TiledPublic.Laid{d_inside,d_shaped,d_restored,d_counted,d_pinned}} :
    TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,dshape,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,d_lanes,Grid.Innermost{}})}
  +w_own = {KernelGemm.Both{w_global,TiledPublic.Laid{w_inside,w_shaped,w_restored,w_counted,w_pinned}} :
    TiledPublic.Contract(~Weights,~weight_value,weight_context,wshape,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),wcells,wf,Tensors.Usable{},ww},1n,w_lanes,Grid.Outermost{}})}
  %limits_exact : TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
    Gradient.input_geometry(s,p,h,w,_,Conv.input_shape(Grid.logical_shape(dcells,Grid.Layout{Tiled.lane_axis(List.length(&2,U32,dcells)),Grid.Innermost{}},d_lanes)),
      Grid.logical_shape(wcells,Grid.Layout{1n,Grid.Outermost{}},w_lanes),dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(wcells,1n,w_lanes,Grid.Outermost{}),wshape,w_restored) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_geometry(s,p,h,w,limits,Conv.input_shape(Grid.logical_shape(dcells,Grid.Layout{Tiled.lane_axis(List.length(&2,U32,dcells)),Grid.Innermost{}},d_lanes)),_,
        dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,dcells)),axis,Call.cells_axis(mode,g)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_geometry(s,p,h,w,limits,Conv.input_shape(Grid.logical_shape(dcells,Grid.Layout{_,Grid.Innermost{}},d_lanes)),wshape,
        dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(dcells,axis,d_lanes,Grid.Innermost{}),dshape,d_restored) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_geometry(s,p,h,w,limits,Conv.input_shape(_),wshape,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(Maybe<&2,Conv.InputShape>,Conv.input_shape(dshape),Some{Conv.InputShape{Call.images(mode),Call.batched(mode),o,gh,gw}},shape_known(mode,g)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_geometry(s,p,h,w,limits,_,wshape,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(Bool,U32.is_eq(o,o),True{},Arithmetic.equal_reflexive(o)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_checked((_ && U32.is_eq(k,k)) && Gradient.input_valid_with(arguments,dc,wc,limits),arguments,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,
        Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(Bool,U32.is_eq(k,k),True{},Arithmetic.equal_reflexive(k)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_checked((True{} && _) && Gradient.input_valid_with(arguments,dc,wc,limits),arguments,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,
        Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  %Equal.sym(Bool,Gradient.input_valid_with(arguments,dc,wc,limits),True{},accepted) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,
      Gradient.input_checked((True{} && True{}) && _,arguments,dcells,PiecesModel.pack(~Tile.Tile8,dm),dw,Tensors.joined(~Tile.Tile8,wc,wf,PiecesModel.pack(~Tile.Tile8,wm)),kept_info))
  checked_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,g,limits,default_capacity,accepted,images_positive,
    dm,df,dw,d_own,wm,wf,ww,w_own,kept,kept_exact)

# The output gradient arranged with its lanes on the channels and the groups
# innermost, as the cells of a model.
def gradient_held(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  gradient: Tiled.Tiled,+held: Relane.Arranged(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),gradient,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),
  +wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}})) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_aligned(Tiled.arranged(gradient,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  KernelGemm.Both{+form,+arranged} = held
  +dshape = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(gradient,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(gradient,axis,Grid.Innermost{})),axis,Grid.lanes_at(dshape,axis),Grid.Innermost{}},form) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))
  Relane.modeled(~Gradient,~gradient_value,(tiled => TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))),
    gradient_context,dshape,Tiled.cells_of(Tiled.arranged(gradient,axis,Grid.Innermost{})),axis,Grid.lanes_at(dshape,axis),Grid.Innermost{},arranged,
    dm => df => dw => d_contract => geometry_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,default_capacity,limits_exact,
      accepted,images_positive,dm,df,dw,d_contract,wm,wf,ww,w_contract,
      Buffer.observe(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(Call.weight_cell_shape(geometry)),wf,PiecesModel.pack(~Tile.Tile8,wm)))))

# The output gradient moved onto its channels, its cells whole.
def gradient_arranged(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,
  +packed: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +wm: PiecesModel.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}})) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_aligned(Gradient.channel_lanes(Tiled.shape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})),
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+shaped,+restored,+counted,+pinned}} = packed
  +dshape = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  +own = {KernelGemm.Both{global,TiledPublic.Laid{inside,shaped,restored,counted,pinned}} :
    TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,dshape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})}
  +rank_room = Order.transitive(List.length(&2,U32,dshape),4n,U32.to_nat(Machine.capacity(limits)),Call.gradient_rank(mode,geometry),Call.four_room(limits))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),dshape,restored) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},
          Tiled.lane_axis(List.length(&2,U32,_)),Grid.Innermost{}),
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},
        Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,dshape)),axis,Call.shape_axis(mode,geometry)) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},_,Grid.Innermost{}),
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(Call.weight_shape(geometry),1n),Grid.Outermost{}},
        Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))
  gradient_held(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,default_capacity,limits_exact,accepted,images_positive,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},
    Relane.arranged_held(~Gradient,~gradient_value,gradient_context,dshape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},axis,Grid.Innermost{},
      limits,default_capacity,Call.gradient_axis_inside(mode,geometry),gradient_values_nonempty(mode,geometry,limits,accepted,images_positive),rank_room,own),
    wm,wf,ww,w_contract)

# The weights arranged with their lanes on the input channels and the groups
# outermost, and the output gradient, as the cells of models.
def weights_held(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  gradient: Tiled.Tiled,+d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),gradient),
  weights: Tiled.Tiled,+held: Relane.Arranged(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights,1n,Grid.Outermost{})) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input_aligned(Gradient.channel_lanes(Tiled.shape(gradient)),Tiled.arranged(weights,1n,Grid.Outermost{}),
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  KernelGemm.Both{+form,+arranged} = held
  +wshape = Call.weight_shape(geometry)
  %Equal.sym(Tiled.Tiled,Tiled.arranged(weights,1n,Grid.Outermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(weights,1n,Grid.Outermost{})),1n,Grid.lanes_at(wshape,1n),Grid.Outermost{}},form) :
    TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(Gradient.channel_lanes(Tiled.shape(gradient)),_,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))
  Relane.modeled(~Weights,~weight_value,(tiled => TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input_aligned(Gradient.channel_lanes(Tiled.shape(gradient)),tiled,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))),
    weight_context,wshape,Tiled.cells_of(Tiled.arranged(weights,1n,Grid.Outermost{})),1n,Grid.lanes_at(wshape,1n),Grid.Outermost{},arranged,
    wm => wf => ww => w_contract => TiledPublic.by_packed(~Gradient,~gradient_value,(tiled => TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,
        mode,geometry,Gradient.input_aligned(Gradient.channel_lanes(Tiled.shape(tiled)),
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),Call.weight_cell_shape(geometry),wf,Tensors.Usable{},ww},1n,Grid.lanes_at(wshape,1n),Grid.Outermost{}},
          Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))),
      gradient_context,Call.input_shape(mode,geometry),gradient,d_contract,
      xm => xs => xf => xw => xa => xl => xp => packed => gradient_arranged(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,
        default_capacity,limits_exact,accepted,images_positive,xm,xs,xf,xw,xa,xl,xp,packed,wm,wf,ww,w_contract)))

# The call at abstract limits that are the default limits.
def limits_input(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +limits_exact: {limits == Default.limits() : Machine.Limits},
  +accepted: Accepted(mode,geometry,limits),+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  gradient: Tiled.Tiled,+d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),gradient),
  weights: Tiled.Tiled,+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights)) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input(gradient,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  +wshape = Call.weight_shape(geometry)
  +rank_room = Order.transitive(List.length(&2,U32,wshape),4n,U32.to_nat(Machine.capacity(limits)),Call.weight_rank(geometry),Call.four_room(limits))
  TiledPublic.by_packed(~Weights,~weight_value,(tiled => TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
      Gradient.input(gradient,tiled,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))),
    weight_context,wshape,weights,w_contract,
    wm => ws => wf => ww => wa => wl => wp => packed => weights_held(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,limits,
      default_capacity,limits_exact,accepted,images_positive,gradient,d_contract,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},
      Relane.arranged_held(~Weights,~weight_value,weight_context,wshape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},1n,Grid.Outermost{},
        limits,default_capacity,Call.weight_axis_inside(geometry),weight_values_nonempty(mode,geometry,limits,accepted),rank_room,packed)))

# The public input gradient: an output gradient and weights of the contract
# of the shapes of a call that passes its checks give the input gradient of
# their readers and the weights with their lanes on the input channels.
def tiled_input(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,
  gradient: Tiled.Tiled,+d_contract: TiledPublic.Contract(~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),gradient),
  weights: Tiled.Tiled,+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  +checks: {Call.checks(mode,geometry) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool}) ->
  TiledInputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,
    Gradient.input(gradient,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  limits_input(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,Default.limits(),{==},{==},checks,images_positive,
    gradient,d_contract,weights,w_contract)
