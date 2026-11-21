# The public form of a window operation: over input cells that hold the
# lifted input of [images, channels, height, width], its output cells hold
# the lifted values of the window. The facts of the machine the window needs
# are premises of its call.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_cells.bend as Cells
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_halo_windows.bend as Windows
import ./kernel_window_windows.bend as KernelWindows
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_task_runs.bend as Runs
import ./tensor_move_readback.bend as Held
import ./tensor_tiled_representation.bend as Representation
import ./tensor_window_values.bend as Values
import ./tensor_window_rows.bend as Rows
import ./tensor_window_piece.bend as Piece
import ./tensor_window_task.bend as Task
import ./tensor_window_reach.bend as Reach
import ./tensor_window_windowed.bend as Windowed
import ./tensor_window_spec.bend as Spec
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_storage_count.bend as StorageCount
import ./tensor_tiled_public.bend as Public
import ../default_machine_limits.bend as Default

# ---- The cells of a window operation ----

# The tasks decided over the output cells of whole output positions read
# inside the input cells.
def decided_reaches(+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +cells: List<&2,U32>,+limits: Machine.Limits,+divisors: Piece.Divisors<Machine.capacity(limits),groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,groups,out_height,images>,
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+total: Nat,
  +input_cells: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),total) == True{} : Bool},
  +shape_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +divides: GridRanges.Multiple<GridRanges.last_extent(cells),U32.to_nat(groups)>,+bounded: {Nat.is_le(ShapeLaws.elements(cells),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra) ->
  Windows.Reaches(~Window.Windowing,~Window.reach_of,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,Grid.grid(cells,Grid.piece_target())},
    Runs.decision_tasks(Tasks.decide_whole(Grid.grid(cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),total):
  +grid = Grid.grid(cells,Grid.piece_target())
  +counted = Held.nat_le_moved(U32.to_nat(Grid.count_of(grid)),ShapeLaws.elements(cells),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),GridRanges.grid_count(limits,cells,Grid.piece_target(),shape_room,nonempty),bounded)
  +pieces = Runs.run_within(Pieces.ranges_of_grid(grid),U32.to_nat(groups),U32.to_nat(Grid.count_of(grid)),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),counted,
    GridRanges.grid_run_of(limits,cells,Grid.piece_target(),shape_room,nonempty,U32.to_nat(groups),divides))
  Windowed.window_reaches(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,limits,divisors,images,room,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),Order.reflexive(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))),
    out_room,positions_room,total,input_cells,Runs.decision_tasks(Tasks.decide_whole(grid,workers,1,bytes_per_cell,operations,extra)),
    Runs.decided_run(limits,grid,workers,1,bytes_per_cell,operations,extra,U32.to_nat(groups),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),pieces,out_room))

# Pieces placed from a model are the pieces of a usable tensor.
def placed_usable(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~layout: Pieces.Ranges,-pieces: Pieces.Pieces(~Element),
  +shape: List<&2,U32>,+fill: Element,+workers: U32,+placed: Placed.PlacedModel<Element,Property,layout,pieces>) ->
  Global.Usable<Element,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>:
  Placed.PlacedModel{+model,+exact,laid,holds} = placed
  Global.Usable{model,fill,workers,
    Equal.cong(Pieces.Pieces(~Element),Tensors.Tensor<Element>,result => Tensors.Tensor{result,shape,fill,Tensors.Usable{},workers},PiecesModel.pack(~Element,model),pieces,
      Equal.sym(Pieces.Pieces(~Element),pieces,PiecesModel.pack(~Element,model),exact))}

# Over input cells that hold the lifted input, the output cells of a window
# operation hold the lifted values of the window.
def window_cells(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: Spec.WeightLanes(~Parameters,~weight,~Related),~bias_lanes: Spec.BiasLanes(~Parameters,~bias,~Related),~operation: F32 -> F32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+out_outer: U32,+out_after: U32,+out_height: U32,+out_width: U32,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+kind_w: Window.Sampling,+kind_exact: {Rows.of_sampling(kind_w) == Values.lifted(kind) : Values.Sampling<Tile.Tile8>},
  +channels_room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+divisors: Piece.Divisors<Machine.capacity(limits),Tiled.groups(lanes),out_height,out_width>,
  +room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,Tiled.groups(lanes),out_height,images>,
  +cells_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +in_cells: List<&2,U32>,+in_total: {ShapeLaws.elements(in_cells) == Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))) : Nat},+in_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(in_cells)) == True{} : Bool},
  +out_cells: List<&2,U32>,+out_total: {ShapeLaws.elements(out_cells) == Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))) : Nat},+out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(out_cells)) == True{} : Bool},
  +divides: GridRanges.Multiple<GridRanges.last_extent(out_cells),U32.to_nat(Tiled.groups(lanes))>,+counted: Global.Counted<out_cells>,
  pieces: Pieces.Pieces(~Tile.Tile8),+fill: Tile.Tile8,+status: Tensors.Status,+in_workers: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},in_cells,Tensors.Tensor{pieces,in_cells,fill,status,in_workers}),
  +parameter_model: Buffer.Model<Tile.Tile8>,+parameters_exact: {Buffer.tree(~Tile.Tile8,parameter_model) == weights : Tree.Tree<Tile.Tile8>},
  +workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,+out_fill: Tile.Tile8) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Spec.Spec<Logical,Parameters>>,~Representation.lifted(~Spec.Spec<Logical,Parameters>,~Spec.spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation)),Representation.Lifted{Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},
    Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())},Tensors.grid_of(in_cells),Tasks.decide_whole(Grid.grid(out_cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameter_model)),out_cells,out_fill,Tensors.Usable{},workers}):
  +total = ShapeLaws.elements(in_cells)
  +total_room = Held.nat_le_moved(total,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits)),in_total,in_room)
  +shape_room = Held.nat_le_moved(ShapeLaws.elements(out_cells),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits)),out_total,cells_room)
  +out_bounded = Held.nat_le_moved(ShapeLaws.elements(out_cells),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),out_total,Order.reflexive(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)))))
  +input_cells = Held.nat_le_moved(Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(Tiled.groups(lanes)))),total,total,
    Equal.trans(Nat,Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(Tiled.groups(lanes)))),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),total,
      Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(Tiled.groups(lanes)))),RowMajor.assoc(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width),U32.to_nat(Tiled.groups(lanes)))),
      Equal.sym(Nat,total,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),in_total)),
    Order.reflexive(total))
  +reaches = decided_reaches(kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width,out_cells,limits,divisors,images,room,cells_room,
    positions_room,total,input_cells,shape_room,out_nonempty,divides,out_bounded,workers,bytes_per_cell,operations,extra)
  +covered = Windows.shaped_covered(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},in_cells,pieces,fill,status,in_workers,shaped)
  +windowed = Windows.covered_windows(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Window.Windowing,~Window.reach_of,limits,Tensors.grid_of(in_cells),Pieces.ranges_of_grid(Tensors.grid_of(in_cells)),
    total,KernelWindows.input_laid(limits,in_cells,total_room,in_nonempty),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())},Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},Window.sample_pad_of(Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())}),Machine.capacity(limits),
    Limits.capacity_allocatable(limits),total_room,GridRanges.grid_pieces_word(limits,in_cells,Grid.piece_target(),total_room,in_nonempty),Runs.decision_tasks(Tasks.decide_whole(Grid.grid(out_cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra)),reaches,pieces,covered)
  +placed = Windowed.windowed_cells(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation),Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,
    out_height,out_width,limits,divisors,images,room,out_cells,out_nonempty,divides,out_bounded,cells_room,positions_room,total,total_room,Tensors.grid_of(in_cells),pieces,
    parameter_model,parameters_exact,workers,bytes_per_cell,operations,extra,windowed)
  +contract = GridRanges.grid_contract(~Tile.Tile8,~Task.Problem<Representation.Lifted<Logical>>,~Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation)),Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},limits,out_cells,out_fill,workers,Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())},Tensors.grid_of(in_cells),Tasks.decide_whole(Grid.grid(out_cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameter_model)),shape_room,out_nonempty,U32.to_nat(Tiled.groups(lanes)),divides,counted,placed)
  Global.relabeled_below(~Tile.Tile8,~Task.Problem<Representation.Lifted<Logical>>,~Representation.Lifted<Spec.Spec<Logical,Parameters>>,~Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation)),~Representation.lifted(~Spec.Spec<Logical,Parameters>,~Spec.spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation)),
    ~(left => right => Global.Fixed(~Task.Problem<Representation.Lifted<Logical>>,~Representation.Lifted<Spec.Spec<Logical,Parameters>>,Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},Representation.Lifted{Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},left,right)),
    ~Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),
    ~(left => right => fixed => index => inside => Global.fixed_below(~Tile.Tile8,~Task.Problem<Representation.Lifted<Logical>>,~Representation.Lifted<Spec.Spec<Logical,Parameters>>,~Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation)),~Representation.lifted(~Spec.Spec<Logical,Parameters>,~Spec.spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation)),~Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},Representation.Lifted{Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},
      cell => below => Spec.cell_agrees(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,kind_w,kind_exact,channels_room,cell,below),
      left,right,fixed,index,inside)),
    Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},Representation.Lifted{Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},KernelGemm.Both{{==},{==}},out_cells,Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())},Tensors.grid_of(in_cells),Tasks.decide_whole(Grid.grid(out_cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameter_model)),out_cells,out_fill,Tensors.Usable{},workers},
    placed_usable(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Task.Problem<Representation.Lifted<Logical>>,~Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation)),Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width}),~Pieces.ranges_of_grid(Grid.grid(out_cells,Grid.piece_target())),Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(out_cells,Grid.piece_target())},Tensors.grid_of(in_cells),Tasks.decide_whole(Grid.grid(out_cells,Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),pieces,Buffer.pack(~Tile.Tile8,parameter_model)),out_cells,out_fill,workers,placed),
    contract,out_bounded,Machine.capacity(limits),cells_room)

# ---- The shape of a window operation ----

def split_length(+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32) -> {List.length(&2,U32,List.append(&2,U32,before,[lanes,rows,columns])) == Nat.add(3n,List.length(&2,U32,before)) : Nat}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,List.append(&2,U32,tail,[lanes,rows,columns])),Nat.add(3n,List.length(&2,U32,tail)),
        split_length(tail,lanes,rows,columns))

def split_below(+count: Nat) -> {Nat.is_le(count,Nat.add(3n,count)) == True{} : Bool}:
  match count:
    case 0n: {==}
    case 1n+ +rest: split_below(rest)

# The lane axis of [before.., lanes, rows, columns] lies inside it.
def split_inside(+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32) -> {Nat.is_le(List.length(&2,U32,before),List.length(&2,U32,List.append(&2,U32,before,[lanes,rows,columns]))) == True{} : Bool}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,rows,columns])),Nat.add(3n,List.length(&2,U32,before)),split_length(before,lanes,rows,columns)) : {Nat.is_le(List.length(&2,U32,before),_) == True{} : Bool}
  split_below(List.length(&2,U32,before))

def split_within(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) ->
  {Nat.is_lt(List.length(&2,U32,before),List.length(&2,U32,List.append(&2,U32,before,Con{lanes,rest}))) == True{} : Bool}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}: split_within(tail,lanes,rest)

def split_lanes(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) -> {Grid.lanes_at(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)) == lanes : U32}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}: split_lanes(tail,lanes,rest)

def split_after(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) -> {Public.lane_after(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)) == rest : List<&2,U32>}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}: split_after(tail,lanes,rest)

def split_removed(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) ->
  {Grid.removed(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)) == List.append(&2,U32,before,rest) : List<&2,U32>}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => Con{head,others},Grid.removed(List.append(&2,U32,tail,Con{lanes,rest}),List.length(&2,U32,tail)),
        List.append(&2,U32,tail,rest),split_removed(tail,lanes,rest))

# The cells of [before.., lanes, rest..] with the groups innermost: the
# other axes and the groups.
def split_cells(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) ->
  {Tiled.cell_shape(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before),Grid.Innermost{}) == List.append(&2,U32,List.append(&2,U32,before,rest),[Tiled.groups(lanes)]) : List<&2,U32>}:
  %Equal.sym(Bool,Nat.is_lt(List.length(&2,U32,before),List.length(&2,U32,List.append(&2,U32,before,Con{lanes,rest}))),True{},split_within(before,lanes,rest)) :
    {Grid.placed(_,Grid.Innermost{},Grid.removed(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)),Grid.groups(Grid.lanes_at(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)))) ==
      List.append(&2,U32,List.append(&2,U32,before,rest),[Tiled.groups(lanes)]) : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Grid.removed(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)),List.append(&2,U32,before,rest),split_removed(before,lanes,rest)) :
    {Grid.placed(True{},Grid.Innermost{},_,Grid.groups(Grid.lanes_at(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)))) ==
      List.append(&2,U32,List.append(&2,U32,before,rest),[Tiled.groups(lanes)]) : List<&2,U32>}
  %Equal.sym(U32,Grid.lanes_at(List.append(&2,U32,before,Con{lanes,rest}),List.length(&2,U32,before)),lanes,split_lanes(before,lanes,rest)) :
    {Grid.placed(True{},Grid.Innermost{},List.append(&2,U32,before,rest),Grid.groups(_)) == List.append(&2,U32,List.append(&2,U32,before,rest),[Tiled.groups(lanes)]) : List<&2,U32>}
  {==}

# The last extent of extents ending with the groups is the groups.
def split_last(+others: List<&2,U32>,+groups: U32) -> {GridRanges.last_extent(List.append(&2,U32,others,[groups])) == U32.to_nat(groups) : Nat}:
  match others:
    case Nil{}: {==}
    case Con{+head,+tail}:
      match tail:
        case Nil{}: {==}
        case Con{+next,+rest}: split_last(Con{next,rest},groups)

# The cells of a window shape: the images, the rows, the columns and the
# groups.
def split_elements(+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32) ->
  {ShapeLaws.elements(Tiled.cell_shape(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{})) == Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{})),ShapeLaws.elements(List.append(&2,U32,List.append(&2,U32,before,[rows,columns]),[Tiled.groups(lanes)])),Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))),
    Equal.cong(List<&2,U32>,Nat,cells => ShapeLaws.elements(cells),Tiled.cell_shape(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{}),List.append(&2,U32,List.append(&2,U32,before,[rows,columns]),[Tiled.groups(lanes)]),split_cells(before,lanes,[rows,columns])),
    Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,List.append(&2,U32,before,[rows,columns]),[Tiled.groups(lanes)])),Nat.mul(ShapeLaws.elements(List.append(&2,U32,before,[rows,columns])),ShapeLaws.elements([Tiled.groups(lanes)])),Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))),
    ShapeLaws.elements_appended(List.append(&2,U32,before,[rows,columns]),[Tiled.groups(lanes)]),
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements(List.append(&2,U32,before,[rows,columns])),ShapeLaws.elements([Tiled.groups(lanes)])),Nat.mul(Nat.mul(ShapeLaws.elements(before),ShapeLaws.elements([rows,columns])),Nat.mul(U32.to_nat(Tiled.groups(lanes)),1n)),Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,ShapeLaws.elements([Tiled.groups(lanes)])),ShapeLaws.elements(List.append(&2,U32,before,[rows,columns])),Nat.mul(ShapeLaws.elements(before),ShapeLaws.elements([rows,columns])),ShapeLaws.elements_appended(before,[rows,columns])),
    Equal.trans(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(before),ShapeLaws.elements([rows,columns])),Nat.mul(U32.to_nat(Tiled.groups(lanes)),1n)),Nat.mul(Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(columns),1n))),U32.to_nat(Tiled.groups(lanes))),Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))),
    Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(columns),1n))),value),Nat.mul(U32.to_nat(Tiled.groups(lanes)),1n),U32.to_nat(Tiled.groups(lanes)),Order.times_one(U32.to_nat(Tiled.groups(lanes)))),
    Equal.trans(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(columns),1n))),U32.to_nat(Tiled.groups(lanes))),Nat.mul(Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),U32.to_nat(Tiled.groups(lanes))),Nat.mul(Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),U32.to_nat(Tiled.groups(lanes))),
    Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),value)),U32.to_nat(Tiled.groups(lanes))),Nat.mul(U32.to_nat(columns),1n),U32.to_nat(columns),Order.times_one(U32.to_nat(columns))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(Tiled.groups(lanes))),Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),Equal.sym(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(rows)),U32.to_nat(columns)),Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),RowMajor.assoc(ShapeLaws.elements(before),U32.to_nat(rows),U32.to_nat(columns)))))))))

# The values of a window shape: the images, the channels and the plane.
def values_elements(+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32) ->
  {ShapeLaws.elements(List.append(&2,U32,before,[lanes,rows,columns])) == Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(lanes)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,before,[lanes,rows,columns])),Nat.mul(ShapeLaws.elements(before),ShapeLaws.elements([lanes,rows,columns])),Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(lanes)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),
    ShapeLaws.elements_appended(before,[lanes,rows,columns]),
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements(before),ShapeLaws.elements([lanes,rows,columns])),Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(lanes),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)))),Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(lanes)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),
    Equal.cong(Nat,Nat,value => Nat.mul(ShapeLaws.elements(before),value),ShapeLaws.elements([lanes,rows,columns]),Nat.mul(U32.to_nat(lanes),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),ShapeLaws.elements_three(lanes,rows,columns)),
    Equal.sym(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(before),U32.to_nat(lanes)),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Nat.mul(ShapeLaws.elements(before),Nat.mul(U32.to_nat(lanes),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)))),RowMajor.assoc(ShapeLaws.elements(before),U32.to_nat(lanes),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))))))

# The values after the lane axis of a window shape are its plane.
def split_plane(+limits: Machine.Limits,+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) == True{} : Bool}) ->
  {Representation.after_count(Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat}:
  Equal.trans(Nat,Representation.least_one(U32.to_nat(Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))))),Representation.least_one(U32.to_nat(Tiled.product([rows,columns]))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),
    Equal.cong(List<&2,U32>,Nat,after => Representation.least_one(U32.to_nat(Tiled.product(after))),Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before)),[rows,columns],split_after(before,lanes,[rows,columns])),
    Equal.trans(Nat,Representation.least_one(U32.to_nat(Tiled.product([rows,columns]))),Representation.least_one(Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),
    Equal.cong(Nat,Nat,value => Representation.least_one(value),U32.to_nat(Tiled.product([rows,columns])),Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),rows,(columns * 1 : U32),U32.to_nat(rows),U32.to_nat(columns),{==},Equal.cong(U32,Nat,word => U32.to_nat(word),(columns * 1 : U32),columns,Arithmetic.multiply_one(columns)),room)),
    Representation.least_positive(Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),positive)))

# The geometry of a window shape with its lanes on the channels.
def split_geometry(+before: List<&2,U32>,+lanes: U32,+rows: U32,+columns: U32) ->
  {Public.geometry_of(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{}) == Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),lanes,Tiled.groups(lanes),Grid.Innermost{}} : Cells.Cells}:
  Equal.trans(Cells.Cells,Public.geometry_of(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{}),
    Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),Grid.lanes_at(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before)),Tiled.groups(Grid.lanes_at(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),Grid.Innermost{}},Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),lanes,Tiled.groups(lanes),Grid.Innermost{}},
    Public.geometry_exact(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),Grid.Innermost{},split_inside(before,lanes,rows,columns)),
    Equal.cong(U32,Cells.Cells,count => Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before))),count,Tiled.groups(count),Grid.Innermost{}},Grid.lanes_at(List.append(&2,U32,before,[lanes,rows,columns]),List.length(&2,U32,before)),lanes,split_lanes(before,lanes,[rows,columns])))

# ---- A window operation over tiled values ----

# What a window needs of the machine.
type Fits<-limits: Machine.Limits,-images: Nat,-lanes: U32,-height: U32,-width: U32,-kernel: U32,-stride: U32,-padding: U32,-out_height: U32,-out_width: U32> is Data:
  Fits{divisors: Piece.Divisors<Machine.capacity(limits),Tiled.groups(lanes),out_height,out_width>,
    room: Reach.ReachRoom<Machine.capacity(limits),kernel,stride,padding,height,width,Tiled.groups(lanes),out_height,images>,
    cells_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    logical_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channels_room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    plane_room: {Nat.is_le(Nat.mul(U32.to_nat(height),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},plane_positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(height),U32.to_nat(width))) == True{} : Bool},
    out_plane_room: {Nat.is_le(Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},images_positive: {Nat.is_lt(0n,images) == True{} : Bool}}

# Over tiled values of [before.., lanes, height, width] with their lanes on
# the channels and the groups innermost, a window operation holds the lifted
# values of the window.
def window_contract(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: Spec.WeightLanes(~Parameters,~weight,~Related),~bias_lanes: Spec.BiasLanes(~Parameters,~bias,~Related),~operation: F32 -> F32 -> F32,+logical: Logical,+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+out_height: U32,+out_width: U32,+kind_w: Window.Sampling,+kind_exact: {Rows.of_sampling(kind_w) == Values.lifted(kind) : Values.Sampling<Tile.Tile8>},
  +fits: Fits<limits,ShapeLaws.elements(before),lanes,height,width,kernel,stride,padding,out_height,out_width>,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+in_workers: U32,
  +contract: Public.Contract(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),fill,Tensors.Usable{},in_workers},List.length(&2,U32,before),
    Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Grid.Innermost{}}),
  +parameter_model: Buffer.Model<Tile.Tile8>,+parameters_exact: {Buffer.tree(~Tile.Tile8,parameter_model) == weights : Tree.Tree<Tile.Tile8>},
  +workers: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra) ->
  Public.Contract(~Spec.Spec<Logical,Parameters>,~Spec.spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},List.append(&2,U32,before,[lanes,out_height,out_width]),Tiled.Tiled{Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},Tensors.grid_of(Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{})),Tasks.decide_whole(Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),PiecesModel.pack(~Tile.Tile8,model),Buffer.pack(~Tile.Tile8,parameter_model)),Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Tiled.zero_tile(),Tensors.Usable{},workers},List.length(&2,U32,before),lanes,Grid.Innermost{}}):
  Fits{+divisors,+room,+cells_room,+positions_room,+in_room,+logical_room,+channels_room,+plane_room,+plane_positive,+out_plane_room,+images_positive} = fits
  Piece.Divisors{+capacity_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  +images = ShapeLaws.elements(before)
  +in_total = split_elements(before,lanes,height,width)
  +out_total = split_elements(before,lanes,out_height,out_width)
  +in_positive = Order.product_positive(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes)),
    Order.product_positive(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width),Order.product_positive(images,U32.to_nat(height),images_positive,
      RowMajor.left_factor_positive(U32.to_nat(height),U32.to_nat(width),plane_positive)),DivisionOrder.right_factor_positive(U32.to_nat(height),U32.to_nat(width),plane_positive)),groups_positive)
  +out_positive = Order.product_positive(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)),
    Order.product_positive(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width),Order.product_positive(images,U32.to_nat(out_height),images_positive,height_positive),width_positive),groups_positive)
  +out_room = Held.nat_le_moved(ShapeLaws.elements(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{})),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits)),out_total,cells_room)
  +values_room = Held.nat_le_moved(ShapeLaws.elements(List.append(&2,U32,before,[lanes,out_height,out_width])),Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits)),
    values_elements(before,lanes,out_height,out_width),logical_room)
  +last_exact = Equal.trans(Nat,GridRanges.last_extent(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{})),GridRanges.last_extent(List.append(&2,U32,List.append(&2,U32,before,[out_height,out_width]),[Tiled.groups(lanes)])),
    Nat.mul(1n,U32.to_nat(Tiled.groups(lanes))),
    Equal.cong(List<&2,U32>,Nat,cells => GridRanges.last_extent(cells),Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),List.append(&2,U32,List.append(&2,U32,before,[out_height,out_width]),[Tiled.groups(lanes)]),
      split_cells(before,lanes,[out_height,out_width])),
    Equal.trans(Nat,GridRanges.last_extent(List.append(&2,U32,List.append(&2,U32,before,[out_height,out_width]),[Tiled.groups(lanes)])),U32.to_nat(Tiled.groups(lanes)),Nat.mul(1n,U32.to_nat(Tiled.groups(lanes))),
      split_last(List.append(&2,U32,before,[out_height,out_width]),Tiled.groups(lanes)),
      Equal.trans(Nat,U32.to_nat(Tiled.groups(lanes)),Nat.mul(U32.to_nat(Tiled.groups(lanes)),1n),Nat.mul(1n,U32.to_nat(Tiled.groups(lanes))),Equal.sym(Nat,Nat.mul(U32.to_nat(Tiled.groups(lanes)),1n),U32.to_nat(Tiled.groups(lanes)),Order.times_one(U32.to_nat(Tiled.groups(lanes)))),
        RowMajor.multiply_commutative(U32.to_nat(Tiled.groups(lanes)),1n))))
  +divides = {GridRanges.Multiple{1n,last_exact} : GridRanges.Multiple<GridRanges.last_extent(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{})),U32.to_nat(Tiled.groups(lanes))>}
  +shaped = Public.cells_shaped(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Grid.Innermost{},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),fill,Tensors.Usable{},in_workers},contract)
  +global = window_cells(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,(Public.lane_prefix(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before))),lanes,height,width,limits,images,
    split_plane(limits,before,lanes,height,width,plane_room,plane_positive),in_room,weights,parameters,kind,kernel,stride,padding,related,(Public.lane_prefix(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before))),out_height,out_width,
    split_plane(limits,before,lanes,out_height,out_width,out_plane_room,Order.product_positive(U32.to_nat(out_height),U32.to_nat(out_width),height_positive,width_positive)),logical_room,
    kind_w,kind_exact,channels_room,divisors,room,cells_room,positions_room,Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),in_total,
    %Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{})),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),in_total) : {Nat.is_lt(0n,_) == True{} : Bool} in_positive,
    Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),out_total,
    %Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{})),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),out_total) : {Nat.is_lt(0n,_) == True{} : Bool} out_positive,
    divides,StorageCount.counted_within(limits,Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),out_room,default_capacity),PiecesModel.pack(~Tile.Tile8,model),fill,Tensors.Usable{},in_workers,
    %split_geometry(before,lanes,height,width) :
      Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),fill,Tensors.Usable{},in_workers})
      shaped,
    parameter_model,parameters_exact,workers,bytes_per_cell,operations,extra,Tiled.zero_tile())
  KernelGemm.Both{
    %Equal.sym(Cells.Cells,Public.geometry_of(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Cells.Cells{(Public.lane_prefix(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),1) * Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before))),lanes,Tiled.groups(lanes),Grid.Innermost{}},
        split_geometry(before,lanes,out_height,out_width)) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Spec.Spec<Logical,Parameters>>,~Representation.lifted(~Spec.Spec<Logical,Parameters>,~Spec.spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation)),Representation.Lifted{Spec.Spec{Spec.Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},_},Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{kind_w,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},Tensors.grid_of(Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{})),Tasks.decide_whole(Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target()),workers,1,bytes_per_cell,operations,extra),PiecesModel.pack(~Tile.Tile8,model),Buffer.pack(~Tile.Tile8,parameter_model)),Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Tiled.zero_tile(),Tensors.Usable{},workers})
      global,
    Public.Laid{split_inside(before,lanes,out_height,out_width),{==},
      %split_lanes(before,lanes,[out_height,out_width]) : {Tiled.logical_shape(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),List.length(&2,U32,before),_,Grid.Innermost{}) == List.append(&2,U32,before,[lanes,out_height,out_width]) : List<&2,U32>}
        Public.cells_restored(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),
      StorageCount.counted_within(limits,List.append(&2,U32,before,[lanes,out_height,out_width]),values_room,default_capacity),split_lanes(before,lanes,[out_height,out_width])}}
