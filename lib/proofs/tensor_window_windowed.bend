# A window operation over the windows of its tasks: every task owns a copy of
# the parameters and computes the pieces of its cells from the rows it holds.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../tensor_halo.bend as Halo
import ../tensor_move.bend as Move
import ../tensor_window.bend as Window
import ../traversal_partition.bend as Partition
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_window_piece.bend as Piece
import ./tensor_window_reach.bend as Reach
import ./tensor_window_task.bend as Task
import ./tensor_pieces_spread.bend as Spread
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_grouped_with.bend as Grouped
import ./tensor_placed.bend as Placed
import ./tensor_halo_windows.bend as Windows
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_task_runs.bend as Runs
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_shape_refinement.bend as ShapeLaws

# ---- The parameters of a task ----

# A copy of the parameters holds their cells.
def copied_tree(+source: Buffer.Model<Tile.Tile8>,+copy: Buffer.Model<Tile.Tile8>,
  +same: {Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,copy)) == Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,source)) : Buffer.Content<Tile.Tile8>},
  -cells: Tree.Tree<Tile.Tile8>,+exact: {Buffer.tree(~Tile.Tile8,source) == cells : Tree.Tree<Tile.Tile8>}) -> {Buffer.tree(~Tile.Tile8,copy) == cells : Tree.Tree<Tile.Tile8>}:
  +content = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,copy),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,copy)),Buffer.content(~Tile.Tile8,source),
    Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,copy)),Buffer.content(~Tile.Tile8,copy),Buffer.packed_contents(~Tile.Tile8,copy)),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,copy)),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,source)),Buffer.content(~Tile.Tile8,source),
      same,Buffer.packed_contents(~Tile.Tile8,source)))
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,copy),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,copy)),cells,
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,copy)),Buffer.tree(~Tile.Tile8,copy),Buffer.content_tree_of(~Tile.Tile8,copy)),
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,copy)),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)),cells,
      Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,content => Buffer.content_tree(~Tile.Tile8,content),Buffer.content(~Tile.Tile8,copy),Buffer.content(~Tile.Tile8,source),content),
      Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,source)),Buffer.tree(~Tile.Tile8,source),cells,
        Buffer.content_tree_of(~Tile.Tile8,source),exact)))

# ---- The fact of a task ----

# A task of whole output positions computes the pieces of its cells from the
# window it holds and its copy of the parameters.
def window_fact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+out_bound: Nat,
  +out_cells: {Nat.is_le(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +shape: GridLaws.Shape<limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,U32.to_nat(groups)) : Nat},+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,
  +window: Windows.TaskWindow<Tile.Tile8,Holder,value,holder,holding,Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},first,count)),total>,
  +fits: Runs.Fits<U32.to_nat(groups),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),out_bound>,+resource: Buffer.Model<Tile.Tile8>,
  +resource_exact: {Buffer.tree(~Tile.Tile8,resource) == cells : Tree.Tree<Tile.Tile8>},
  +low_boundary: GridRanges.Boundary<Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},U32.to_nat(first)>,+high_boundary: GridRanges.Boundary<Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Nat.add(U32.to_nat(first),U32.to_nat(count))>) ->
  Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),Spread.split_buffer(~Tile.Tile8),Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},holding,first,count,resource>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Reach.ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  Windows.TaskWindow{+high,+reaches,+inside,+tiled,+values} = window
  +grid = {Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)} : Grid.Grid}
  +g = U32.to_nat(groups)
  +rows = Task.task_rows(Machine.capacity(limits),groups,out_height,out_width,divisors,images,first,count,out_bound,fits,out_cells,out_room,positions_room)
  +held = Task.task_held(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,Machine.capacity(limits),divisors,images,room,first,count,rows,high,reaches)
  +bound_room = Order.transitive(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g),U32.to_nat(Machine.capacity(limits)),out_cells,out_room)
  +p = U32.to_nat(GridRanges.boundary_piece(grid,U32.to_nat(first),low_boundary))
  +q = U32.to_nat(GridRanges.boundary_piece(grid,Nat.add(U32.to_nat(first),U32.to_nat(count)),high_boundary))
  +later = Grid.later_pieces(grid,first,count)
  +bounds = GridRanges.task_bounds(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,g,out_bound,bound_room,fits,low_boundary,high_boundary)
  +chain = GridRanges.task_piece_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,g,out_bound,bound_room,fits,low_boundary,high_boundary,units,unit_exact)
  +runs = GridRanges.runs_of_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,
    GridLaws.pieces_room(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape),1n+later,p,U32.to_nat(first),g,Nat.add(U32.to_nat(first),U32.to_nat(count)),
    GridRanges.task_inside(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,first,count,g,out_bound,bound_room,fits,low_boundary,high_boundary,bounds),chain)
  +first_runs = {%Equal.sym(U32,Grid.piece_of(grid,first),U32.from_nat(p),GridRanges.bounds_first_piece(grid,first,count,p,q,bounds)) :
      GridRanges.PieceRuns(1n+later,grid,_,U32.to_nat(first),g,Nat.add(U32.to_nat(first),U32.to_nat(count)))
    runs : GridRanges.PieceRuns(1n+later,grid,Grid.piece_of(grid,first),U32.to_nat(first),g,Nat.add(U32.to_nat(first),U32.to_nat(count)))}
  +low = U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},first,count)))
  +facts = Task.piece_facts(1n+later,low,high,Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,out_width,lanes,Machine.capacity(limits),divisors,Limits.capacity_allocatable(limits),
    Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),held,sample_room,top_room,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),Order.reflexive(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width))),Order.reflexive(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width))),
    Task.rows_end_room(Machine.capacity(limits),groups,out_height,out_width,images,first,count,rows),Task.rows_end_cap(Machine.capacity(limits),groups,out_height,out_width,images,first,count,rows),
    grid,Grid.piece_of(grid,first),U32.to_nat(first),Order.reflexive(U32.to_nat(first)),first_runs)
  %resource_exact : Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),Spread.split_buffer(~Tile.Tile8),Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,_,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},holding,first,count,resource>
  Task.task_fact(~Holder,~value,~operation,~Spread.split_buffer(~Tile.Tile8),holder,low,high,Machine.capacity(limits),bound_ok,Order.transitive(high,total,U32.to_nat(Machine.capacity(limits)),inside,total_room),
    Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,resource,groups_room,groups_positive,
    Machine.capacity(limits),bound_ok,divisors,first,count,holding,tiled,values,{==},facts)

# Every task of a layout over its window and its share of the parameters gives
# its fact.
def window_facts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+out_bound: Nat,
  +out_cells: {Nat.is_le(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +shape: GridLaws.Shape<limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,U32.to_nat(groups)) : Nat},tasks: Pieces.Ranges,windows: Windows.WindowModel<Tile.Tile8>,+resource: Buffer.Model<Tile.Tile8>,
  +resource_exact: {Buffer.tree(~Tile.Tile8,resource) == cells : Tree.Tree<Tile.Tile8>},
  +ok: Windows.WindowsOk(~Tile.Tile8,~Holder,~value,~Window.Windowing,~Window.reach_of,holder,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},total,tasks,windows),
  +run: Runs.Run(tasks,U32.to_nat(groups),out_bound),+bounded: GridRanges.Bounded(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},tasks)) ->
  Grouped.TaskFacts(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~PiecesModel.Model<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Grid.Grid,~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},tasks,windows,resource):
  match tasks:
    case Pieces.Range{+first,+count}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Held.Both{+same_first,+window_facts} = ok
          Held.Both{+same_count,+window} = window_facts
          Held.Both{+low_boundary,+high_boundary} = bounded
          %Equal.sym(U32,task_first,first,same_first) : Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),Spread.split_buffer(~Tile.Tile8),Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},holding,_,task_count,resource>
          %Equal.sym(U32,task_count,count,same_count) : Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),Spread.split_buffer(~Tile.Tile8),Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},holding,first,_,resource>
          window_fact(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,units,unit_exact,first,count,holding,window,run,resource,resource_exact,low_boundary,high_boundary)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Empty.absurd(Grouped.TaskFacts(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~PiecesModel.Model<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Grid.Grid,~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},Pieces.Range{first,count},Windows.WindowFork{left_windows,right_windows},resource),ok)
    case Pieces.Split{+left,+right}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Empty.absurd(Grouped.TaskFacts(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~PiecesModel.Model<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Grid.Grid,~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},Pieces.Split{left,right},Windows.WindowLeaf{holding,task_first,task_count},resource),ok)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Held.Both{+left_ok,+right_ok} = ok
          Held.Both{+left_run,+run_rest} = run
          Held.Both{+right_run,+glue} = run_rest
          Held.Both{+left_bounded,+right_bounded} = bounded
          +left_resource = PartitionLaws.first(~Buffer.Model<Tile.Tile8>,Spread.split_buffer(~Tile.Tile8,resource))
          +right_resource = PartitionLaws.second(~Buffer.Model<Tile.Tile8>,Spread.split_buffer(~Tile.Tile8,resource))
          Held.Both{
            window_facts(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,units,unit_exact,left,left_windows,left_resource,copied_tree(resource,left_resource,Spread.first_shared(~Tile.Tile8,resource),cells,resource_exact),
              left_ok,left_run,left_bounded),
            window_facts(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,units,unit_exact,right,right_windows,right_resource,copied_tree(resource,right_resource,Spread.second_shared(~Tile.Tile8,resource),cells,resource_exact),
              right_ok,right_run,right_bounded)}

# ---- The windowed operation ----

# The window operation over a decision of the output grid: the output pieces
# of its tasks placed in the output grid hold the values of the operation,
# given that the tasks of the decision cover the grid.
def windowed_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+out_bound: Nat,
  +out_cells: {Nat.is_le(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +nonzero: {U32.is_zero(grid_count) == False{} : Bool},
  +shape: GridLaws.Shape<limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,U32.to_nat(groups)) : Nat},+input_grid: Grid.Grid,-pieces: Pieces.Pieces(~Tile.Tile8),+parameters: Buffer.Model<Tile.Tile8>,
  +parameters_exact: {Buffer.tree(~Tile.Tile8,parameters) == cells : Tree.Tree<Tile.Tile8>},decision: Tasks.Decision,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},total,Runs.decision_tasks(decision),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}}),Runs.decision_tasks(decision),pieces)>,
  +run: Runs.Run(Runs.decision_tasks(decision),U32.to_nat(groups),out_bound),+bounded: GridRanges.Bounded(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Runs.decision_tasks(decision)),
  cover: @+results: Partition.Tree<&2,PiecesModel.Model<Tile.Tile8>,Unit> -> @+laid: GridRanges.TasksLaid(~Tile.Tile8,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},Runs.decision_tasks(decision),results) ->
    {Placed.tasks_spans(~Tile.Tile8,results,Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}),Held.Ended{}) : Held.Spans}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}),Window.windowed_decided(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,decision,pieces,Buffer.pack(~Tile.Tile8,parameters))>:
  Tasks.Decision{+parallel,+tasks} = decision
  Windows.Windowed{+model,+packed,+holds} = windowed
  +grid = {Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)} : Grid.Grid}
  +context = {Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}} : Window.Windowing}
  +facts = window_facts(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,grid_cells,units,unit_exact,tasks,model,parameters,parameters_exact,holds,run,bounded)
  +results = Grouped.results(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~PiecesModel.Model<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Grid.Grid,~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),grid,context,tasks,model,parameters,facts)
  +laid_hold = GridRanges.laid_results(~Tile.Tile8,~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),~Holder,~value,~Window.Windowing,~Window.reach_of,holder,context,total,grid,context,tasks,model,parameters,facts,holds)
  +placed = Placed.placed_model(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(grid),Tile.filled(0.0),results,U32.to_nat(Grid.start(grid,U32.from_nat(0n))),U32.to_nat(groups),
    U32.to_nat(grid_count),GridRanges.grid_chain(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,U32.to_nat(groups),units,unit_exact),
    cover(results,Held.first_of(GridRanges.TasksLaid(~Tile.Tile8,grid,tasks,results),Placed.TasksHold(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),results),laid_hold)),
    Held.second_of(GridRanges.TasksLaid(~Tile.Tile8,grid,tasks,results),Placed.TasksHold(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),results),laid_hold))
  # a nonempty grid runs its tasks
  %Equal.sym(Bool,U32.is_zero(grid_count),False{},nonzero) : Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}),Window.windowed_when(~operation,_,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,decision,pieces,Buffer.pack(~Tile.Tile8,parameters))>
  %Equal.sym(Partition.Tree<&1,Move.Holding<Tile.Tile8>,Unit>,Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,context,input_grid,Window.sample_pad_of(context),tasks,pieces),
      Windows.pack_windows(~Tile.Tile8,model),packed) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(grid),Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),
      Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),
        tasks,parallel,context,_,Buffer.pack(~Tile.Tile8,parameters)))>
  %Equal.sym(Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit>,Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,
        ~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),tasks,parallel,context,Windows.pack_windows(~Tile.Tile8,model),Buffer.pack(~Tile.Tile8,parameters)),
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Unit,~PiecesModel.pack(~Tile.Tile8),results),
      Grouped.grouped(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Window.Windowing,~PiecesModel.Model<Tile.Tile8>,~Buffer.Model<Tile.Tile8>,~Tensors.shared_buffer(~Tile.Tile8),~Window.task_leaf(~operation),~PiecesModel.pack(~Tile.Tile8),~Buffer.pack(~Tile.Tile8),~Spread.split_buffer(~Tile.Tile8),~Grid.Grid,~GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),~Spread.clone_shared(~Tile.Tile8),tasks,parallel,grid,context,model,parameters,facts)) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(grid),Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),_)>
  placed

# The window operation over the decision of its output grid.
def windowed_decided(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +grid_count: U32,+grid_prefix: U32,+grid_outer: U32,+grid_extent: U32,+grid_unit: U32,+grid_block: U32,+grid_per: U32,
  +nonzero: {U32.is_zero(grid_count) == False{} : Bool},
  +shape: GridLaws.Shape<limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per>,
  +grid_cells: {U32.to_nat(grid_count) == Nat.mul(Nat.mul(U32.to_nat(grid_outer),U32.to_nat(grid_extent)),U32.to_nat(grid_unit)) : Nat},
  +units: Nat,+unit_exact: {U32.to_nat(grid_unit) == Nat.mul(units,U32.to_nat(groups)) : Nat},
  +out_cells: {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +input_grid: Grid.Grid,-pieces: Pieces.Pieces(~Tile.Tile8),+parameters: Buffer.Model<Tile.Tile8>,
  +parameters_exact: {Buffer.tree(~Tile.Tile8,parameters) == cells : Tree.Tree<Tile.Tile8>},+workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},total,Runs.decision_tasks(Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}}),Runs.decision_tasks(Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra)),pieces)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}),Window.windowed_decided(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameters))>:
  +g = U32.to_nat(groups)
  +grid_run = GridRanges.grid_run(limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,g,units,unit_exact)
  +bound_room = Order.transitive(U32.to_nat(grid_count),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g),U32.to_nat(Machine.capacity(limits)),out_cells,out_room)
  windowed_output(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,U32.to_nat(grid_count),out_cells,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,nonzero,shape,grid_cells,units,unit_exact,input_grid,pieces,parameters,parameters_exact,Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra),windowed,
    Runs.decided_run(limits,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra,g,U32.to_nat(grid_count),grid_run,bound_room),
    GridRanges.bounded_decided(limits,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra,g,U32.to_nat(grid_count),grid_run,bound_room,
      GridRanges.grid_bounded(limits,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},GridLaws.GridFacts{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,shape,{==},grid_cells})),
    GridRanges.tasks_cover(~Tile.Tile8,limits,grid_outer,grid_extent,grid_unit,grid_block,grid_per,grid_count,grid_prefix,shape,grid_cells,g,units,unit_exact,workers,1,
      bytes_per_cell,operations,extra))

# The window operation over the grid of an output cell shape, given the facts
# of that grid.
def windowed_grid(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+cells_shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells_shape)) == True{} : Bool},
  +out_cells: {Nat.is_le(ShapeLaws.elements(cells_shape),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: GridLaws.GridFacts<limits,Grid.grid(cells_shape,Grid.piece_target())>,
  +whole_units: GridRanges.Multiple<U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()))),U32.to_nat(groups)>,
  +counted: {U32.to_nat(Grid.count_of(Grid.grid(cells_shape,Grid.piece_target()))) == ShapeLaws.elements(cells_shape) : Nat},
  +input_grid: Grid.Grid,-pieces: Pieces.Pieces(~Tile.Tile8),+parameters: Buffer.Model<Tile.Tile8>,
  +parameters_exact: {Buffer.tree(~Tile.Tile8,parameters) == cells : Tree.Tree<Tile.Tile8>},+workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},total,Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())}),Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),pieces)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(Grid.grid(cells_shape,Grid.piece_target())),Window.windowed_decided(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},input_grid,Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameters))>:
  GridLaws.GridFacts{+grid_count,+grid_prefix,+grid_outer,+grid_extent,+grid_unit,+grid_block,+grid_per,+shape,+exact,+grid_cells} = facts
  +g = U32.to_nat(groups)
  +lit = {Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)} : Grid.Grid}
  +kept = Equal.trans(U32,grid_unit,GridRanges.grid_unit_cells(Grid.grid(cells_shape,Grid.piece_target())),GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target())),
    Equal.sym(U32,GridRanges.grid_unit_cells(Grid.grid(cells_shape,Grid.piece_target())),grid_unit,Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),Grid.grid(cells_shape,Grid.piece_target()),lit,exact)),
    Equal.trans(U32,GridRanges.grid_unit_cells(Grid.grid(cells_shape,Grid.piece_target())),GridRanges.grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())),GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target())),
      Equal.cong(Grid.Grid,U32,grid => GridRanges.grid_unit_cells(grid),Grid.grid(cells_shape,Grid.piece_target()),Grid.grid_of_units(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target()),GridLaws.grid_units(cells_shape,Grid.piece_target())),
      GridRanges.unit_cells_kept(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())))
  +unit_whole = GridRanges.multiple_moved(U32.to_nat(grid_unit),U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()))),g,
    Equal.cong(U32,Nat,word => U32.to_nat(word),grid_unit,GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target())),kept),whole_units)
  +count_exact = Equal.trans(Nat,U32.to_nat(grid_count),U32.to_nat(Grid.count_of(Grid.grid(cells_shape,Grid.piece_target()))),ShapeLaws.elements(cells_shape),
    Equal.sym(Nat,U32.to_nat(Grid.count_of(Grid.grid(cells_shape,Grid.piece_target()))),U32.to_nat(grid_count),Equal.cong(Grid.Grid,Nat,grid => U32.to_nat(Grid.count_of(grid)),Grid.grid(cells_shape,Grid.piece_target()),lit,exact)),counted)
  +nonzero = Arithmetic.word_positive_nonzero(grid_count,{%Equal.sym(Nat,U32.to_nat(grid_count),ShapeLaws.elements(cells_shape),count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    nonempty : {Nat.is_lt(0n,U32.to_nat(grid_count)) == True{} : Bool}})
  +counted_cells = {%Equal.sym(Nat,U32.to_nat(grid_count),ShapeLaws.elements(cells_shape),count_exact) : {Nat.is_le(_,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g)) == True{} : Bool} out_cells :
    {Nat.is_le(U32.to_nat(grid_count),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g)) == True{} : Bool}}
  +units = GridRanges.multiple_times(U32.to_nat(grid_unit),g,unit_whole)
  +unit_exact = GridRanges.multiple_exact(U32.to_nat(grid_unit),g,unit_whole)
  %Equal.sym(Grid.Grid,Grid.grid(cells_shape,Grid.piece_target()),lit,exact) : Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(_),Window.windowed_decided(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,_},input_grid,Tasks.decide_whole(_,workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameters))>
  windowed_decided(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,out_room,positions_room,total,total_room,grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,nonzero,shape,grid_cells,
    units,unit_exact,counted_cells,input_grid,pieces,parameters,parameters_exact,workers,bytes_per_cell,operations,extra,{%exact : Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,_},total,Runs.decision_tasks(Tasks.decide_whole(_,workers,1,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,_},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,_}),Runs.decision_tasks(Tasks.decide_whole(_,workers,1,bytes_per_cell,operations,extra)),pieces)> windowed : Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},total,Runs.decision_tasks(Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)}}),Runs.decision_tasks(Tasks.decide_whole(Grid.Grid{grid_count,grid_prefix,grid_outer,grid_extent,grid_unit,grid_block,grid_per,(grid_outer * grid_per : U32)},workers,1,bytes_per_cell,operations,extra)),pieces)>})

# The window operation over an output cell shape whose last extent is a
# multiple of the groups: its output pieces placed in the grid of the shape
# hold the values of the operation.
def windowed_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+cells: Tree.Tree<Tile.Tile8>,
  +kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+cells_shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells_shape)) == True{} : Bool},
  +divides: GridRanges.Multiple<GridRanges.last_extent(cells_shape),U32.to_nat(groups)>,
  +out_cells: {Nat.is_le(ShapeLaws.elements(cells_shape),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: Nat,+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +input_grid: Grid.Grid,-pieces: Pieces.Pieces(~Tile.Tile8),+parameters: Buffer.Model<Tile.Tile8>,
  +parameters_exact: {Buffer.tree(~Tile.Tile8,parameters) == cells : Tree.Tree<Tile.Tile8>},+workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,
  +windowed: Windows.Windowed<Tile.Tile8,Holder,value,holder,Window.Windowing,Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},total,Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),
    Halo.windows(~Tile.Tile8,~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},input_grid,Window.sample_pad_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())}),Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),pieces)>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Holder>,~Task.problem_value(~Holder,~value,~operation),Task.Problem{holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),Pieces.ranges_of_grid(Grid.grid(cells_shape,Grid.piece_target())),Window.windowed_decided(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells_shape,Grid.piece_target())},input_grid,Tasks.decide_whole(Grid.grid(cells_shape,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameters))>:
  +room_cells = Order.transitive(ShapeLaws.elements(cells_shape),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits)),out_cells,out_room)
  windowed_grid(~Holder,~value,~operation,holder,cells,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,limits,divisors,images,room,cells_shape,nonempty,out_cells,out_room,positions_room,total,total_room,GridRanges.grid_facts(limits,cells_shape,Grid.piece_target(),room_cells,nonempty),
    GridRanges.multiple_through(U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(cells_shape,Grid.strides(cells_shape),U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target()))),GridRanges.last_extent(cells_shape),U32.to_nat(groups),
      GridRanges.units_last(limits,cells_shape,U32.is_le(Grid.product(cells_shape),Grid.piece_target()),0,1,1,Grid.piece_target(),room_cells,nonempty),divides),
    GridRanges.grid_count(limits,cells_shape,Grid.piece_target(),room_cells,nonempty),input_grid,pieces,parameters,parameters_exact,workers,bytes_per_cell,operations,extra,windowed)

# ---- The reach of the tasks ----

# The reach of every task of whole output positions lies in the input cells
# and ends at or after it starts.
def window_reaches(+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +grid: Grid.Grid,+limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,+out_bound: Nat,
  +out_cells: {Nat.is_le(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+total: Nat,
  +input_cells: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),total) == True{} : Bool},
  tasks: Pieces.Ranges,+run: Runs.Run(tasks,U32.to_nat(groups),out_bound)) ->
  Windows.Reaches(~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},tasks,total):
  match tasks:
    case Pieces.Range{+first,+count}:
      Task.task_reach(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,Machine.capacity(limits),divisors,images,room,first,count,
        Task.task_rows(Machine.capacity(limits),groups,out_height,out_width,divisors,images,first,count,out_bound,run,out_cells,out_room,positions_room),total,input_cells)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+run_rest} = run
      Held.Both{+right_run,+glue} = run_rest
      Held.Both{
        window_reaches(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,
          total,input_cells,left,left_run),
        window_reaches(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,limits,divisors,images,room,out_bound,out_cells,out_room,positions_room,
          total,input_cells,right,right_run)}
