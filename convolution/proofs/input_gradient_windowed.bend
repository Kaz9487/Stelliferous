# The windowed input gradient: every task holds the gradient cells it reads
# and owns a copy of the packed weights.

import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/tensor_tasks.bend as Tasks
import ../../lib/tensor_halo.bend as Halo
import ../../lib/tensor_move.bend as Move
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_clone.bend as Clone
import ../../lib/proofs/storage_clone_refinement.bend as CloneLaws
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_pieces_grouped_with.bend as Grouped
import ../../lib/proofs/tensor_grid_ranges.bend as GridRanges
import ../../lib/proofs/tensor_grid_pieces.bend as GridLaws
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/tensor_placed.bend as Placed
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_batch_rows.bend as BatchRows
import ./input_gradient_piece_cells.bend as PieceCells
import ./input_gradient_task.bend as TaskTree

# ---- The weights a task owns ----

# The packed weights of a call.
def weight_cells(+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Rows.native_tree(Rows.packing(call,weights))

# A copy of the packed weights.
type WeightModel<-cells: Tree.Tree<Tile.Tile8>> is Data:
  WeightModel{tiles: Buffer.Model<Tile.Tile8>,exact: {Buffer.tree(~Tile.Tile8,tiles) == cells : Tree.Tree<Tile.Tile8>}}

def pack_weights(-cells: Tree.Tree<Tile.Tile8>,model: WeightModel<cells>) -> Storage.Buffer<Tile.Tile8>:
  WeightModel{tiles,exact} = model
  Buffer.pack(~Tile.Tile8,tiles)

# Sharing gives each of the two tasks a copy of the weights.
def split_seen(-cells: Tree.Tree<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>,+exact: {Buffer.tree(~Tile.Tile8,tiles) == cells : Tree.Tree<Tile.Tile8>},
  +seen: CloneLaws.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))>) -> PartitionLaws.Split<WeightModel<cells>>:
  CloneLaws.Observation{+left,+right,+equation} = seen
  PartitionLaws.Split{
    WeightModel{left,CloneLaws.copied_tree(~Tile.Tile8,tiles,left,CloneLaws.left_contents(~Tile.Tile8,tiles,left,right,equation),cells,exact)},
    WeightModel{right,CloneLaws.copied_tree(~Tile.Tile8,tiles,right,CloneLaws.right_contents(~Tile.Tile8,tiles,left,right,equation),cells,exact)}}

def split_weights(-cells: Tree.Tree<Tile.Tile8>,model: WeightModel<cells>) -> PartitionLaws.Split<WeightModel<cells>>:
  WeightModel{+tiles,+exact} = model
  split_seen(cells,tiles,exact,CloneLaws.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))))

def shared_seen(-cells: Tree.Tree<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>,+exact: {Buffer.tree(~Tile.Tile8,tiles) == cells : Tree.Tree<Tile.Tile8>},
  +seen: CloneLaws.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))>) ->
  {Kernel.share(Buffer.pack(~Tile.Tile8,tiles)) ==
    (pack_weights(cells,PartitionLaws.first(~WeightModel<cells>,split_seen(cells,tiles,exact,seen))),
      pack_weights(cells,PartitionLaws.second(~WeightModel<cells>,split_seen(cells,tiles,exact,seen)))) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  CloneLaws.Observation{+left,+right,+equation} = seen
  Equal.sym(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,(Buffer.pack(~Tile.Tile8,left),Buffer.pack(~Tile.Tile8,right)),
    Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),equation)

# Sharing the weights of a task packs the two halves of its split.
def share_matches(-cells: Tree.Tree<Tile.Tile8>,+resource: WeightModel<cells>) ->
  {Kernel.share(pack_weights(cells,resource)) ==
    (pack_weights(cells,PartitionLaws.first(~WeightModel<cells>,split_weights(cells,resource))),
      pack_weights(cells,PartitionLaws.second(~WeightModel<cells>,split_weights(cells,resource)))) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  WeightModel{+tiles,+exact} = resource
  shared_seen(cells,tiles,exact,CloneLaws.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles))))

# ---- The facts of a task ----

def grid_of(+grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32) -> Grid.Grid:
  Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}

# What a task gives as a fact of the tasks grouped with the weights.
def TaskFact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +grid: Grid.Grid,+holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+resource: WeightModel<weight_cells(call,weights)>) -> Data:
  Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Kernel.Work,PiecesModel.Model<Tile.Tile8>,WeightModel<weight_cells(call,weights)>,
    Kernel.share,Kernel.task_leaf,PiecesModel.pack(~Tile.Tile8),pack_weights(weight_cells(call,weights)),split_weights(weight_cells(call,weights)),Grid.Grid,
    GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after))),
    grid,Kernel.Work{Rows.task_of(Domain.geometry(call)),grid},holding,first,count,resource>

# The facts of the tasks of a layout over their windows.
def TaskFacts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +grid: Grid.Grid,tasks: Pieces.Ranges,+windows: Windows.WindowModel<Tile.Tile8>,+resource: WeightModel<weight_cells(call,weights)>) -> Data:
  Grouped.TaskFacts(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~PiecesModel.Model<Tile.Tile8>,~WeightModel<weight_cells(call,weights)>,
    ~Kernel.share,~Kernel.task_leaf,~PiecesModel.pack(~Tile.Tile8),~pack_weights(weight_cells(call,weights)),~split_weights(weight_cells(call,weights)),~Grid.Grid,
    ~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after))),
    grid,Kernel.Work{Rows.task_of(Domain.geometry(call)),grid},tasks,windows,resource)

# What a task gives from its pieces and the weights it hands back.
def fact_of_leaf(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+high: Nat,
  +layout: Held.Spans,+outer: U32,+after: U32,+grid: Grid.Grid,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,
  +resource: WeightModel<weight_cells(call,weights)>,
  +leaf: TaskTree.TaskLeaf<Holder,value,holder,call,weights,TaskTree.task_low(call,first,count),high,layout,outer,after,grid,first,count,
    Kernel.task_leaf(Kernel.Work{Rows.task_of(Domain.geometry(call)),grid},first,count,Held.pack(~Tile.Tile8,holding),pack_weights(weight_cells(call,weights),resource))>) ->
  TaskFact(~Holder,~value,holder,call,weights,outer,after,grid,holding,first,count,resource):
  TaskTree.TaskLeaf{+pieces,+next,+exact,+values,+laid} = leaf
  TaskTree.OperandModel{+next_holding,+next_tiled,+next_values,+next_ordered,+next_tiles,+next_tiles_exact} = next
  Grouped.TaskFact{Partition.Leaf{pieces,first,count,Unit{}},WeightModel{next_tiles,next_tiles_exact},exact,Held.Both{laid,values}}

# A task of whole pieces of the input gradient grid, with the gradient cells
# it holds and its copy of the weights.
def task_fact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +shape: GridLaws.Shape<Domain.limits(call),grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,unit) : Nat},
  +bound_cells: {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(grid_count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,
  +window: Windows.TaskWindow<Tile.Tile8,Holder,value,holder,holding,Windows.span_start(Kernel.reach_of(Rows.task_of(Domain.geometry(call)),first,count)),
    Windows.span_end(Kernel.reach_of(Rows.task_of(Domain.geometry(call)),first,count)),held_total>,
  +fits_task: Runs.Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(grid_count)>,
  +resource: WeightModel<weight_cells(call,weights)>,
  +low_boundary: GridRanges.Boundary<grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),U32.to_nat(first)>,
  +high_boundary: GridRanges.Boundary<grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),Nat.add(U32.to_nat(first),U32.to_nat(count))>) ->
  TaskFact(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),holding,first,count,resource):
  Windows.TaskWindow{+high,+reaches,+high_inside,+tiled,+values} = window
  WeightModel{+tiles,+tiles_exact} = resource
  +limits = Domain.limits(call)
  +grid = grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)
  +bound = U32.to_nat(grid_count)
  +f = U32.to_nat(first)
  +stop = Nat.add(f,U32.to_nat(count))
  +p = U32.to_nat(GridRanges.boundary_piece(grid,f,low_boundary))
  +q = U32.to_nat(GridRanges.boundary_piece(grid,stop,high_boundary))
  +later = Grid.later_pieces(grid,first,count)
  +bounds = GridRanges.task_bounds(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,unit,bound,bound_room,
    fits_task,low_boundary,high_boundary)
  +chain = GridRanges.task_piece_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,unit,bound,bound_room,
    fits_task,low_boundary,high_boundary,units,unit_exact)
  +runs_at = GridRanges.runs_of_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,
    GridLaws.pieces_room(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape),1n+later,p,f,unit,stop,
    GridRanges.task_inside(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,unit,bound,bound_room,
      fits_task,low_boundary,high_boundary,bounds),chain)
  +runs = {%Equal.sym(U32,Grid.piece_of(grid,first),U32.from_nat(p),GridRanges.bounds_first_piece(grid,first,count,p,q,bounds)) :
      GridRanges.PieceRuns(1n+later,grid,_,f,unit,stop)
    runs_at : GridRanges.PieceRuns(1n+later,grid,Grid.piece_of(grid,first),f,unit,stop)}
  +total = Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))
  +inside = Order.transitive(stop,bound,total,Runs.fits_inside(unit,f,stop,bound,fits_task),bound_cells)
  +capacity = Machine.capacity(limits)
  +bounded = Order.transitive(high,held_total,U32.to_nat(capacity),high_inside,held_room)
  +layout = Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding))
  +low = TaskTree.task_low(call,first,count)
  +model = {TaskTree.OperandModel{holding,tiled,values,{==},tiles,tiles_exact} : TaskTree.OperandModel<Holder,value,holder,call,weights,low,high,layout>}
  +owned = {WeightModel{tiles,tiles_exact} : WeightModel<weight_cells(call,weights)>}
  fact_of_leaf(~Holder,~value,holder,call,weights,high,layout,outer,after,grid,first,count,holding,owned,
    TaskTree.task_leaf(~Holder,~value,holder,call,envelope,fits,weights,high,capacity,Limits.capacity_fits(limits),bounded,layout,outer,after,after_exact,outer_exact,
      values_room,total_room,grid,unit,first,count,inside,reaches,runs,model))

# Every task of a layout over the input gradient grid, with the gradient
# cells it holds and its copy of the weights, gives its fact.
def task_facts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +shape: GridLaws.Shape<Domain.limits(call),grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,unit) : Nat},
  +bound_cells: {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(grid_count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  tasks: Pieces.Ranges,+windows: Windows.WindowModel<Tile.Tile8>,+resource: WeightModel<weight_cells(call,weights)>,
  +ok: Windows.WindowsOk(~Tile.Tile8,~Holder,~value,~Kernel.Task,~Kernel.reach_of,holder,Rows.task_of(Domain.geometry(call)),held_total,tasks,windows),
  +run: Runs.Run(tasks,unit,U32.to_nat(grid_count)),
  +bounded: GridRanges.Bounded(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),tasks)) ->
  TaskFacts(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),tasks,windows,resource):
  match tasks:
    case Pieces.Range{+first,+count}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Held.Both{+same_first,+window_facts} = ok
          Held.Both{+same_count,+window} = window_facts
          Held.Both{+low_boundary,+high_boundary} = bounded
          %Equal.sym(U32,task_first,first,same_first) :
            TaskFact(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),holding,_,task_count,resource)
          %Equal.sym(U32,task_count,count,same_count) :
            TaskFact(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),holding,first,_,resource)
          task_fact(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
            grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,unit,units,unit_exact,bound_cells,bound_room,
            first,count,holding,window,run,resource,low_boundary,high_boundary)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Empty.absurd(TaskFacts(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),
            Pieces.Range{first,count},Windows.WindowFork{left_windows,right_windows},resource),ok)
    case Pieces.Split{+left,+right}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Empty.absurd(TaskFacts(~Holder,~value,holder,call,weights,outer,after,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),
            Pieces.Split{left,right},Windows.WindowLeaf{holding,task_first,task_count},resource),ok)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Held.Both{+left_ok,+right_ok} = ok
          Held.Both{+left_run,+run_rest} = run
          Held.Both{+right_run,+glue} = run_rest
          Held.Both{+left_bounded,+right_bounded} = bounded
          +cells = weight_cells(call,weights)
          Held.Both{
            task_facts(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
              grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,unit,units,unit_exact,bound_cells,bound_room,
              left,left_windows,PartitionLaws.first(~WeightModel<cells>,split_weights(cells,resource)),left_ok,left_run,left_bounded),
            task_facts(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
              grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,unit,units,unit_exact,bound_cells,bound_room,
              right,right_windows,PartitionLaws.second(~WeightModel<cells>,split_weights(cells,resource)),right_ok,right_run,right_bounded)}

# ---- The windowed input gradient ----

# The windowed input gradient over a decision of its grid: the pieces of its
# tasks placed in the grid hold the input gradient, given that the tasks of
# the decision cover the grid.
def windowed_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +nonzero: {U32.is_zero(grid_count) == False{} : Bool},
  +shape: GridLaws.Shape<Domain.limits(call),grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,unit) : Nat},
  +bound_cells: {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(grid_count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +gradient_grid: Grid.Grid,-gradient: Pieces.Pieces(~Tile.Tile8),+tiles: Buffer.Model<Tile.Tile8>,
  +tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == weight_cells(call,weights) : Tree.Tree<Tile.Tile8>},
  decision: Tasks.Decision,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,Runs.decision_tasks(decision),
    Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),Runs.decision_tasks(decision),gradient)>,
  +run: Runs.Run(Runs.decision_tasks(decision),unit,U32.to_nat(grid_count)),
  +bounded: GridRanges.Bounded(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),Runs.decision_tasks(decision)),
  cover: @+results: Partition.Tree<&2,PiecesModel.Model<Tile.Tile8>,Unit> ->
    @+laid: GridRanges.TasksLaid(~Tile.Tile8,grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),Runs.decision_tasks(decision),results) ->
    {Placed.tasks_spans(~Tile.Tile8,results,Held.Ended{}) ==
      Held.layout_spans(Pieces.ranges_of_grid(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)),Held.Ended{}) : Held.Spans}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)),
    Kernel.windowed(Kernel.Work{Rows.task_of(Domain.geometry(call)),grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)},
      gradient_grid,decision,gradient,Buffer.pack(~Tile.Tile8,tiles))>:
  Tasks.Decision{+parallel,+tasks} = decision
  Windows.Windowed{+model,+packed,+holds} = windowed
  +limits = Domain.limits(call)
  +task = Rows.task_of(Domain.geometry(call))
  +grid = grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)
  +cells = weight_cells(call,weights)
  +context = {Kernel.Work{task,grid} : Kernel.Work}
  +resource = {WeightModel{tiles,tiles_exact} : WeightModel<cells>}
  +facts = task_facts(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
    grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,unit,units,unit_exact,bound_cells,bound_room,
    tasks,model,resource,holds,run,bounded)
  +results = Grouped.results(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~PiecesModel.Model<Tile.Tile8>,~WeightModel<cells>,
    ~Kernel.share,~Kernel.task_leaf,~PiecesModel.pack(~Tile.Tile8),~pack_weights(cells),~split_weights(cells),~Grid.Grid,
    ~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after))),
    grid,context,tasks,model,resource,facts)
  +laid_hold = GridRanges.laid_results(~Tile.Tile8,~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~WeightModel<cells>,
    ~Kernel.share,~Kernel.task_leaf,~PiecesModel.pack(~Tile.Tile8),~pack_weights(cells),~split_weights(cells),
    ~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    ~Holder,~value,~Kernel.Task,~Kernel.reach_of,holder,task,held_total,grid,context,tasks,model,resource,facts,holds)
  +placed = Placed.placed_model(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(grid),Tile.filled(0.0),results,U32.to_nat(Grid.start(grid,U32.from_nat(0n))),unit,U32.to_nat(grid_count),
    GridRanges.grid_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,unit,units,unit_exact),
    cover(results,Held.first_of(GridRanges.TasksLaid(~Tile.Tile8,grid,tasks,results),
      Placed.TasksHold(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
        TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),results),laid_hold)),
    Held.second_of(GridRanges.TasksLaid(~Tile.Tile8,grid,tasks,results),
      Placed.TasksHold(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
        TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),results),laid_hold))
  # a nonempty grid runs its tasks
  %Equal.sym(Bool,U32.is_zero(grid_count),False{},nonzero) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)),
    Kernel.windowed_when(_,Rows.task_of(Domain.geometry(call)),grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),
      gradient_grid,decision,gradient,Buffer.pack(~Tile.Tile8,tiles))>
  %Equal.sym(Partition.Tree<&1,Move.Holding<Tile.Tile8>,Unit>,Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,task,gradient_grid,Tile.filled(0.0),tasks,gradient),
      Windows.pack_windows(~Tile.Tile8,model),packed) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
        TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
      Pieces.ranges_of_grid(grid),Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),
        Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~Kernel.share,~Kernel.task_leaf,tasks,parallel,context,_,
          Buffer.pack(~Tile.Tile8,tiles)))>
  %Equal.sym(Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit>,
      Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~Kernel.share,~Kernel.task_leaf,tasks,parallel,context,
        Windows.pack_windows(~Tile.Tile8,model),Buffer.pack(~Tile.Tile8,tiles)),
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Unit,~PiecesModel.pack(~Tile.Tile8),results),
      Grouped.grouped(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Kernel.Work,~PiecesModel.Model<Tile.Tile8>,~WeightModel<cells>,
        ~Kernel.share,~Kernel.task_leaf,~PiecesModel.pack(~Tile.Tile8),~pack_weights(cells),~split_weights(cells),~Grid.Grid,
        ~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
          TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after))),
        ~share_matches(cells),tasks,parallel,grid,context,model,resource,facts)) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
        TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
      Pieces.ranges_of_grid(grid),Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),_)>
  placed

# The windowed input gradient over the decision of its grid.
def windowed_decided(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +nonzero: {U32.is_zero(grid_count) == False{} : Bool},
  +shape: GridLaws.Shape<Domain.limits(call),grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,unit) : Nat},
  +bound_cells: {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(grid_count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +gradient_grid: Grid.Grid,-gradient: Pieces.Pieces(~Tile.Tile8),+tiles: Buffer.Model<Tile.Tile8>,
  +tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == weight_cells(call,weights) : Tree.Tree<Tile.Tile8>},
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,
    Runs.decision_tasks(Tasks.decide_whole(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),workers,whole,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),
      Runs.decision_tasks(Tasks.decide_whole(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),workers,whole,bytes_per_cell,operations,extra)),
      gradient)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)),
    Kernel.windowed(Kernel.Work{Rows.task_of(Domain.geometry(call)),grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)},gradient_grid,
      Tasks.decide_whole(grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per),workers,whole,bytes_per_cell,operations,extra),
      gradient,Buffer.pack(~Tile.Tile8,tiles))>:
  +limits = Domain.limits(call)
  +grid = grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)
  +grid_run = GridRanges.grid_run(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,unit,units,unit_exact)
  windowed_output(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
    grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,nonzero,shape,grid_cells,unit,units,unit_exact,bound_cells,bound_room,
    gradient_grid,gradient,tiles,tiles_exact,Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra),windowed,
    Runs.decided_run(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,U32.to_nat(grid_count),grid_run,bound_room),
    GridRanges.bounded_decided(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,U32.to_nat(grid_count),grid_run,bound_room,
      GridRanges.grid_bounded(limits,grid,GridLaws.GridFacts{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,{==},grid_cells})),
    GridRanges.tasks_cover(~Tile.Tile8,limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,unit,units,unit_exact,
      workers,whole,bytes_per_cell,operations,extra))

# The windowed input gradient over the grid of a cell shape, given the facts
# of that grid.
def windowed_grid(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +cells: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +bound_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +gradient_grid: Grid.Grid,-gradient: Pieces.Pieces(~Tile.Tile8),+tiles: Buffer.Model<Tile.Tile8>,
  +tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == weight_cells(call,weights) : Tree.Tree<Tile.Tile8>},
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +facts: GridLaws.GridFacts<Domain.limits(call),Grid.grid(cells,Grid.piece_target())>,+unit: Nat,
  +whole_units: GridRanges.Multiple<U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()))),unit>,
  +counted: {U32.to_nat(Grid.count_of(Grid.grid(cells,Grid.piece_target()))) == ShapeLaws.elements(cells) : Nat},
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,
    Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),
      Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),gradient)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(Grid.grid(cells,Grid.piece_target())),
    Kernel.windowed(Kernel.Work{Rows.task_of(Domain.geometry(call)),Grid.grid(cells,Grid.piece_target())},gradient_grid,
      Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra),gradient,Buffer.pack(~Tile.Tile8,tiles))>:
  GridLaws.GridFacts{+grid_count,+grid_prefix,+grid_outer,+grid_extent,+grid_unit,+grid_block,+grid_per,+shape,+exact,+grid_cells} = facts
  +limits = Domain.limits(call)
  +total = Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))
  +shaped_grid = Grid.grid(cells,Grid.piece_target())
  +units_found = Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target())
  +lit = grid_of(grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per)
  +kept = Equal.trans(U32,grid_unit,GridRanges.grid_unit_cells(shaped_grid),GridRanges.unit_cells_of(units_found),
    Equal.sym(U32,GridRanges.grid_unit_cells(shaped_grid),grid_unit,Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),shaped_grid,lit,exact)),
    Equal.trans(U32,GridRanges.grid_unit_cells(Grid.grid(cells,Grid.piece_target())),GridRanges.grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())),GridRanges.unit_cells_of(units_found),
      Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),Grid.grid(cells,Grid.piece_target()),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target()),GridLaws.grid_units(cells,Grid.piece_target())),
      GridRanges.unit_cells_kept(units_found,Grid.piece_target())))
  +unit_whole = GridRanges.multiple_moved(U32.to_nat(grid_unit),U32.to_nat(GridRanges.unit_cells_of(units_found)),unit,
    Equal.cong(U32,Nat,word => U32.to_nat(word),grid_unit,GridRanges.unit_cells_of(units_found),kept),whole_units)
  +count_exact = Equal.trans(Nat,U32.to_nat(grid_count),U32.to_nat(Grid.count_of(shaped_grid)),ShapeLaws.elements(cells),
    Equal.sym(Nat,U32.to_nat(Grid.count_of(shaped_grid)),U32.to_nat(grid_count),Equal.cong(Grid.Grid,Nat,grid => U32.to_nat(Grid.count_of(grid)),shaped_grid,lit,exact)),counted)
  +nonzero = Arithmetic.word_positive_nonzero(grid_count,{%Equal.sym(Nat,U32.to_nat(grid_count),ShapeLaws.elements(cells),count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    nonempty : {Nat.is_lt(0n,U32.to_nat(grid_count)) == True{} : Bool}})
  +counted_room = {%Equal.sym(Nat,U32.to_nat(grid_count),ShapeLaws.elements(cells),count_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    bound_room : {Nat.is_le(U32.to_nat(grid_count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +counted_cells = {%Equal.sym(Nat,U32.to_nat(grid_count),ShapeLaws.elements(cells),count_exact) : {Nat.is_le(_,total) == True{} : Bool}
    bound_cells : {Nat.is_le(U32.to_nat(grid_count),total) == True{} : Bool}}
  +units = GridRanges.multiple_times(U32.to_nat(grid_unit),unit,unit_whole)
  +unit_exact = GridRanges.multiple_exact(U32.to_nat(grid_unit),unit,unit_whole)
  %Equal.sym(Grid.Grid,shaped_grid,lit,exact) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
        TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
      Pieces.ranges_of_grid(_),
      Kernel.windowed(Kernel.Work{Rows.task_of(Domain.geometry(call)),_},gradient_grid,Tasks.decide_whole(_,workers,whole,bytes_per_cell,operations,extra),gradient,Buffer.pack(~Tile.Tile8,tiles))>
  windowed_decided(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
    grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,nonzero,shape,grid_cells,unit,units,unit_exact,counted_cells,counted_room,
    gradient_grid,gradient,tiles,tiles_exact,workers,whole,bytes_per_cell,operations,extra,
    {%exact : Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,
        Runs.decision_tasks(Tasks.decide_whole(_,workers,whole,bytes_per_cell,operations,extra)),
        Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),
          Runs.decision_tasks(Tasks.decide_whole(_,workers,whole,bytes_per_cell,operations,extra)),gradient)>
      windowed : Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,
        Runs.decision_tasks(Tasks.decide_whole(lit,workers,whole,bytes_per_cell,operations,extra)),
        Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),
          Runs.decision_tasks(Tasks.decide_whole(lit,workers,whole,bytes_per_cell,operations,extra)),gradient)>})

# The windowed input gradient over a cell shape: its pieces placed in the
# grid of the shape hold the input gradient.
def windowed_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +held_total: Nat,+held_room: {Nat.is_le(held_total,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +cells: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: GridRanges.Multiple<GridRanges.last_extent(cells),unit>,
  +bound_cells: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +gradient_grid: Grid.Grid,-gradient: Pieces.Pieces(~Tile.Tile8),+tiles: Buffer.Model<Tile.Tile8>,
  +tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == weight_cells(call,weights) : Tree.Tree<Tile.Tile8>},
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Kernel.Task,Kernel.reach_of,Rows.task_of(Domain.geometry(call)),held_total,
    Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),gradient_grid,Tile.filled(0.0),
      Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra)),gradient)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      TaskTree.task_values(~Holder,~value,holder,call,weights,outer,after)),
    Pieces.ranges_of_grid(Grid.grid(cells,Grid.piece_target())),
    Kernel.windowed(Kernel.Work{Rows.task_of(Domain.geometry(call)),Grid.grid(cells,Grid.piece_target())},gradient_grid,
      Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,whole,bytes_per_cell,operations,extra),gradient,Buffer.pack(~Tile.Tile8,tiles))>:
  +limits = Domain.limits(call)
  +found = Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target())
  windowed_grid(~Holder,~value,holder,call,envelope,fits,weights,outer,after,after_exact,outer_exact,values_room,total_room,held_total,held_room,
    cells,nonempty,bound_cells,room,gradient_grid,gradient,tiles,tiles_exact,workers,whole,bytes_per_cell,operations,extra,
    GridRanges.grid_facts(limits,cells,Grid.piece_target(),room,nonempty),unit,
    GridRanges.multiple_through(U32.to_nat(GridRanges.unit_cells_of(found)),GridRanges.last_extent(cells),unit,
      GridRanges.units_last(limits,cells,U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target(),room,nonempty),divides),
    GridRanges.grid_count(limits,cells,Grid.piece_target(),room,nonempty),windowed)
