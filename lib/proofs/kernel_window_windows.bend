# The input windows of the tasks of a convolution hold the input cells they read.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_window_pipeline.bend as Window
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../tensor_halo.bend as Halo
import ../machine_limits.bend as Machine
import ../math_activation.bend as Activations
import ./tensor_move_readback.bend as Held
import ./tensor_halo_windows.bend as Windows
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as Ranges
import ./tensor_task_runs.bend as Runs
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_global_shape_count.bend as Global
import ./kernel_window_reach.bend as Reach

# The input grid with its facts, laid out by its ranges over its elements.
def input_laid(+limits: Machine.Limits,+input_shape: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(input_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(input_shape)) == True{} : Bool}) ->
  Windows.Laid<limits,Tensors.grid_of(input_shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(input_shape)),Held.Ended{}),ShapeLaws.elements(input_shape)>:
  %Ranges.grid_count(limits,input_shape,Grid.piece_target(),room,nonempty) :
    Windows.Laid<limits,Tensors.grid_of(input_shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(input_shape)),Held.Ended{}),_>
  Ranges.grid_laid(limits,Tensors.grid_of(input_shape),Ranges.grid_facts(limits,input_shape,Grid.piece_target(),room,nonempty))

# The windows a convolution hands its tasks hold the cells of a Shaped input.
def conv_windows(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logical: Logical,+limits: Machine.Limits,+h: U32,+w: U32,+k: U32,+s: U32,+p: U32,+oh: U32,+ow: U32,+channels: U32,+inner: U32,+gi: U32,+go: U32,+activation: Activations.Activation,+input_shape: List<&2,U32>,
  +output_shape: List<&2,U32>,+workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+cut_cell_ns: F32,+extra: Tasks.Extra,+images: Nat,
  pieces: Pieces.Pieces(~Tile.Tile8),+fill: Tile.Tile8,+status: Tensors.Status,+tensor_workers: U32,
  +shaped: Global.Shaped(~Tile.Tile8,~Logical,~original,logical,input_shape,Tensors.Tensor{pieces,input_shape,fill,status,tensor_workers}),
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(input_shape)) == True{} : Bool},+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(ShapeLaws.elements(input_shape),U32.to_nat(limit)) == True{} : Bool},
  +pieces_ok: {U32.is_le(Grid.pieces_of(Tensors.grid_of(input_shape)),2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(h)) == True{} : Bool},+height_room: {Nat.is_le(U32.to_nat(h),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+height_small: {Nat.is_le(U32.to_nat(U32.max(h,1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(s)) == True{} : Bool},+stride_small: {Nat.is_le(U32.to_nat(s),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+plane_small: {Nat.is_le(U32.to_nat(U32.max((oh * ow : U32),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+groups_small: {Nat.is_le(U32.to_nat(U32.max(go,1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rows_room: {Nat.is_le(Nat.mul(images,U32.to_nat(h)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(images,U32.to_nat((h * w * gi : U32))),ShapeLaws.elements(input_shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(input_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+bound_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(U32.max((oh * ow : U32),1))),U32.to_nat(U32.max(go,1))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_room: {Nat.is_le(ShapeLaws.elements(output_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+output_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(output_shape)) == True{} : Bool},+divides: Ranges.Multiple<Ranges.last_extent(output_shape),U32.to_nat(U32.max(go,1))>,+bounded: {Nat.is_le(ShapeLaws.elements(output_shape),Nat.mul(Nat.mul(images,U32.to_nat(U32.max((oh * ow : U32),1))),U32.to_nat(U32.max(go,1)))) == True{} : Bool},+multiple: Nat,+unit_exact: {U32.to_nat(Runs.cut_unit_of(Grid.grid(output_shape,Grid.piece_target()),whole)) == Nat.mul(multiple,U32.to_nat(U32.max(go,1))) : Nat},+unit_room: {Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.grid(output_shape,Grid.piece_target()),whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Windows.Windowed<Tile.Tile8,Logical,original,logical,Window.Convolution,Window.reach_of,Window.Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation},ShapeLaws.elements(input_shape),
    Runs.decision_tasks(Tasks.decide_cut(Grid.grid(output_shape,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),
    Halo.windows(~Tile.Tile8,~Window.Convolution,~Window.reach_of,Window.Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation},Tensors.grid_of(input_shape),Tile.filled(0.0),Runs.decision_tasks(Tasks.decide_cut(Grid.grid(output_shape,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),pieces)>:
  Windows.covered_windows(~Tile.Tile8,~Logical,~original,~Window.Convolution,~Window.reach_of,limits,Tensors.grid_of(input_shape),Pieces.ranges_of_grid(Tensors.grid_of(input_shape)),ShapeLaws.elements(input_shape),
    input_laid(limits,input_shape,room,nonempty),Window.Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation},logical,Tile.filled(0.0),limit,limit_ok,small,pieces_ok,Runs.decision_tasks(Tasks.decide_cut(Grid.grid(output_shape,Grid.piece_target()),workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),
    Reach.windowed_reaches(limits,h,w,k,s,p,oh,ow,channels,inner,gi,go,activation,output_shape,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra,images,ShapeLaws.elements(input_shape),height_positive,height_room,height_small,stride_positive,stride_small,plane_small,groups_small,rows_room,cells_room,room,bound_room,output_room,output_nonempty,divides,bounded,multiple,unit_exact,unit_room),
    pieces,Windows.shaped_covered(~Tile.Tile8,~Logical,~original,logical,input_shape,pieces,fill,status,tensor_workers,shaped))
