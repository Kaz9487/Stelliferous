# The tasks of a window operation: every task computes the output pieces of
# its cells from the window it holds, and every cell holds the value of its
# window with the lanes past the true extent at +0.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_window.bend as Window
import ./tensor_window_values.bend as Values
import ./tensor_window_rows.bend as Rows
import ./tensor_window_piece.bend as Piece
import ./tensor_pieces_combine_leaf.bend as Updates
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_readback.bend as Readback
import ./tensor_allocation_depth.bend as AllocationDepth
import ./tensor_tiled_padding.bend as Padding
import ./tensor_pieces_content.bend as Content
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./word_power_of_two.bend as Power
import ../storage_buffer.bend as Storage
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_grouped_with.bend as Grouped
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_move_readback.bend as Held
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_halo_windows.bend as Windows
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as U32Extrema
import ./tensor_window_reach.bend as Reach
import ./tensor_task_runs.bend as Runs
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor

# ---- The value of a window operation ----

# A window operation: the input it reads, its parameters, kind and geometry,
# and the true lanes of its output.
type Problem<-Holder: Data> is Data:
  Problem{holder: Holder,weights: Tree.Tree<Tile.Tile8>,kind: Window.Sampling,kernel: U32,stride: U32,padding: U32,height: U32,width: U32,groups: U32,lanes: U32,
    out_height: U32,out_width: U32}

# The output cell of a window operation at a cell index.
def problem_value(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,problem: Problem<Holder>,cell: U32) -> Tile.Tile8:
  Problem{holder,weights,kind,+kernel,+stride,+padding,+height,+width,+groups,+lanes,+out_height,+out_width} = problem
  +index = U32.to_nat(cell)
  Padding.cell(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},index,Values.at(~Tile.Tile8,~operation,~Tile.update_lanes,~Tile.plus,~Rows.Input<Holder>,~Rows.input_read(~Holder,~value),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{holder,height,width,groups},Rows.Weights{weights,kernel,groups},Rows.of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,groups},out_height,out_width,index)))

def initial_of(+kind: Window.Sampling) -> {Window.initial(kind) == Values.start(~Tile.Tile8,Rows.of_sampling(kind)) : Tile.Tile8}:
  match kind:
    case Window.Pooling{+pad}: {==}
    case Window.Weighting{}: {==}

# ---- The output tree of a piece ----

def finish_capacity(+kind: Window.Sampling,+count: U32,+groups: U32,+kernel: U32,+parameters: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,Piece.finish_tree(kind,count,groups,kernel,parameters,tree)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  match kind:
    case Window.Pooling{+pad}: {==}
    case Window.Weighting{}:
      Updates.updated_capacity(~Tile.Tile8,~Rows.Weights,~Tile.Tile8,~Piece.bias_tap,~Tile.plus,U32.to_nat(count),0,Rows.Weights{parameters,kernel,groups},tree)

def finish_balanced(+kind: Window.Sampling,+count: U32,+groups: U32,+kernel: U32,+parameters: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree)) -> Shape.Balanced(Tile.Tile8,depth,Piece.finish_tree(kind,count,groups,kernel,parameters,tree)):
  match kind:
    case Window.Pooling{+pad}: balanced
    case Window.Weighting{}:
      Updates.updated_balanced(~Tile.Tile8,~Rows.Weights,~Tile.Tile8,~Piece.bias_tap,~Tile.plus,U32.to_nat(count),0,Rows.Weights{parameters,kernel,groups},tree,depth,balanced)

# The output tree of a piece is ready at its count of cells.
def piece_ready(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+weights: Tree.Tree<Tile.Tile8>,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,
  +out_height: U32,+out_width: U32,+first: U32,+count: U32,+sized: AllocationDepth.Sized<count>) -> Buffer.ContentReady<Tile.Tile8,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)}>:
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind))
  +fresh_balanced = Shape.fresh(Tile.Tile8,depth,Window.initial(kind))
  +segmented = Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))
  +final = Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind))))
  +capacity_equal = Equal.trans(U32,Tree.capacity(Tile.Tile8,final),Tree.capacity(Tile.Tile8,segmented),U32.shln(1,depth),
    finish_capacity(kind,count,groups,kernel,weights,segmented),
    Equal.trans(U32,Tree.capacity(Tile.Tile8,segmented),Tree.capacity(Tile.Tile8,fresh),U32.shln(1,depth),Piece.segments_capacity(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,fresh),
      Shape.capacity(Tile.Tile8,depth,fresh,fresh_balanced)))
  Buffer.ContentReady{depth,finish_balanced(kind,count,groups,kernel,weights,segmented,depth,Piece.segments_balanced(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,fresh,depth,fresh_balanced)),
    AllocationDepth.allocation_depth(count,sized),
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,final),U32.shln(1,depth),capacity_equal) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    AllocationDepth.allocation_room(count,sized)}

# ---- The cells of a piece ----

# What every cell of a piece reads: the operation, the piece, its words and
# its room.
type PieceCells<-Holder: Data> is Data:
  PieceCells{holder: Holder,weights: Tree.Tree<Tile.Tile8>,kind: Window.Sampling,kernel: U32,stride: U32,padding: U32,height: U32,width: U32,groups: U32,lanes: U32,
    out_height: U32,out_width: U32,first: U32,count: U32,bound: U32,bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},
    divisors: Piece.Divisors<bound,groups,out_height,out_width>,+start_units: Nat,+end_units: Nat,+start_exact: {U32.to_nat(first) == Nat.mul(start_units,U32.to_nat(groups)) : Nat},+end_exact: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == Nat.mul(end_units,U32.to_nat(groups)) : Nat},+end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),end_units),U32.to_nat(bound)) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(end_units,U32.to_nat(groups)),U32.to_nat(bound)) == True{} : Bool},padded: Padding.padding_fits(bound,TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count),sized: AllocationDepth.Sized<count>}

def cells_size(~Holder: Data,cells: PieceCells<Holder>) -> Nat:
  PieceCells{+holder,+weights,+kind,+kernel,+stride,+padding,+height,+width,+groups,+lanes,+out_height,+out_width,+first,+count,+bound,+bound_ok,+divisors,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+sized} = cells
  U32.to_nat(count)

def CellAt(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+cells: PieceCells<Holder>,+index: Nat) -> Data:
  PieceCells{+holder,+weights,+kind,+kernel,+stride,+padding,+height,+width,+groups,+lanes,+out_height,+out_width,+first,+count,+bound,+bound_ok,+divisors,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+sized} = cells
  Content.At(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,weights,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width},first,Buffer.content_tree(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})),Buffer.content_length(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})),index)

# A cell of a piece holds the value of the operation at its cell.
def cell_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+cells: PieceCells<Holder>,+index: Nat,+inside: {Nat.is_lt(index,cells_size(~Holder,cells)) == True{} : Bool}) -> CellAt(~Holder,~value,~operation,cells,index):
  PieceCells{+holder,+weights,+kind,+kernel,+stride,+padding,+height,+width,+groups,+lanes,+out_height,+out_width,+first,+count,+bound,+bound_ok,+divisors,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+sized} = cells
  +f = U32.to_nat(first)
  +g = U32.to_nat(groups)
  +c = U32.to_nat(count)
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind))
  +depth_ok = AllocationDepth.allocation_depth(count,sized)
  +fresh_balanced = Shape.fresh(Tile.Tile8,depth,Window.initial(kind))
  +initial = Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,fresh,U32.from_nat(index)),Window.initial(kind),Values.start(~Tile.Tile8,Rows.of_sampling(kind)),
    Readback.read_fresh(~Tile.Tile8,depth,Window.initial(kind),U32.from_nat(index)),initial_of(kind))
  +value_equal = Piece.piece_value(~Holder,~value,~operation,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,first,count,fresh,depth,U32.shln(1,depth),
    Shape.capacity(Tile.Tile8,depth,fresh,fresh_balanced),fresh_balanced,depth_ok,Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)),bound,divisors,
    start_units,end_units,start_exact,end_exact,end_room,cells_room,AllocationDepth.allocation_room(count,sized),index,inside,initial)
  +final = Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind))))
  +ready = piece_ready(~Holder,~value,~operation,holder,weights,kind,kernel,stride,padding,height,width,groups,out_height,out_width,first,count,sized)
  +sum_bound = {%Equal.sym(Nat,Nat.add(f,c),Nat.mul(end_units,g),end_exact) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} cells_room :
    {Nat.is_le(Nat.add(f,c),U32.to_nat(bound)) == True{} : Bool}}
  +read_padded = Padding.read(bound,bound_ok,TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,final,c,ready,padded,Order.transitive(c,Nat.add(f,c),U32.to_nat(bound),Order.right_le(f,c),sum_bound),
    Order.reflexive(c),index,inside)
  +kept = Padding.kept_ready(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,final,c,ready)
  +length_equal = Shape.second(Buffer.ContentReady<Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})>,{Buffer.content_length(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})) == c : Nat},kept)
  +roundtrip = Bounds.roundtrip(Nat.add(f,index),bound,bound_ok,
    Order.transitive(Nat.add(f,index),Nat.add(f,c),U32.to_nat(bound),Bounds.strict_le(Nat.add(f,index),Nat.add(f,c),Order.offset_lt(f,index,c,inside)),sum_bound))
  KernelGemm.Both{%Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})),c,length_equal) : {Nat.is_lt(index,_) == True{} : Bool} inside,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.add(f,index))),Nat.add(f,index),roundtrip) :
      {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})),U32.from_nat(index)) == Padding.cell(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},_,Values.at(~Tile.Tile8,~operation,~Tile.update_lanes,~Tile.plus,~Rows.Input<Holder>,~Rows.input_read(~Holder,~value),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{holder,height,width,groups},Rows.Weights{weights,kernel,groups},Rows.of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,groups},out_height,out_width,_))) : Tile.Tile8}
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})),U32.from_nat(index)),
      Padding.cell(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Nat.add(f,index),Tree.read(Tile.Tile8,final,U32.from_nat(index))),Padding.cell(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Nat.add(f,index),Values.at(~Tile.Tile8,~operation,~Tile.update_lanes,~Tile.plus,~Rows.Input<Holder>,~Rows.input_read(~Holder,~value),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{holder,height,width,groups},Rows.Weights{weights,kernel,groups},Rows.of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,groups},out_height,out_width,Nat.add(f,index)))),read_padded,
      Equal.cong(Tile.Tile8,Tile.Tile8,cell => Padding.cell(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Nat.add(f,index),cell),Tree.read(Tile.Tile8,final,U32.from_nat(index)),Values.at(~Tile.Tile8,~operation,~Tile.update_lanes,~Tile.plus,~Rows.Input<Holder>,~Rows.input_read(~Holder,~value),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{holder,height,width,groups},Rows.Weights{weights,kernel,groups},Rows.of_sampling(kind),Window.Taps{kernel,stride,padding,height,width,groups},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,groups},out_height,out_width,Nat.add(f,index))),value_equal))}

# Every cell of a piece holds the value of the operation at its cell.
def piece_leaf(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+cells: PieceCells<Holder>) ->
  Members.All(Nat,{index => CellAt(~Holder,~value,~operation,cells,index) : Nat -> Data},Ranges.range(cells_size(~Holder,cells),0n)):
  Members.range_all(~PieceCells<Holder>,~CellAt(~Holder,~value,~operation),~cells_size(~Holder),~cell_at(~Holder,~value,~operation),cells_size(~Holder,cells),0n,cells,Order.reflexive(cells_size(~Holder,cells)))

# ---- The pieces of a task ----

# A piece holds the value of the operation at every cell, as many cells as its
# count, ready.
def piece_holds(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+weights: Tree.Tree<Tile.Tile8>,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+first: U32,+count: U32,+bound: U32,
  +bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+start_units: Nat,+end_units: Nat,+start_exact: {U32.to_nat(first) == Nat.mul(start_units,U32.to_nat(groups)) : Nat},+end_exact: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == Nat.mul(end_units,U32.to_nat(groups)) : Nat},+end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),end_units),U32.to_nat(bound)) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(end_units,U32.to_nat(groups)),U32.to_nat(bound)) == True{} : Bool},
  +padded: Padding.padding_fits(bound,TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count),+sized: AllocationDepth.Sized<count>,+output: Buffer.Model<Tile.Tile8>,
  +tree_equal: {Buffer.tree(~Tile.Tile8,output) == Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),count))))) : Tree.Tree<Tile.Tile8>},+length_equal: {Buffer.length(~Tile.Tile8,output) == Buffer.length(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),count))) : Nat}) ->
  Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,weights,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width},first,count,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))))):
  +c = U32.to_nat(count)
  +final = Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind))))
  +step = Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.Content{Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)))),U32.to_nat(count)})
  +cells = {PieceCells{holder,weights,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,first,count,bound,bound_ok,divisors,start_units,end_units,start_exact,end_exact,end_room,cells_room,padded,sized} :
    PieceCells<Holder>}
  +kept = Padding.kept_ready(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,final,c,piece_ready(~Holder,~value,~operation,holder,weights,kind,kernel,stride,padding,height,width,groups,out_height,out_width,first,count,sized))
  +tree_fresh = Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),count))))),final,tree_equal,
    Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,tree => Piece.finish_tree(kind,count,groups,kernel,weights,Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,weights,tree)),Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),count))),Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Window.initial(kind)),
      AllocationLaws.buffer_tree(~Tile.Tile8,Window.initial(kind),count)))
  +content_equal = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))),Buffer.contents(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))),step,
    Buffer.remodel_content(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))),Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))),step,
      Padding.rule(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),
      Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => Padding.step(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,content),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),Buffer.Content{final,c},
        Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),Buffer.Content{Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output)},
          Buffer.Content{final,c},Held.content_parts(~Tile.Tile8,output),
          Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.Content{Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output)},Buffer.Content{final,Buffer.length(~Tile.Tile8,output)},
            Buffer.Content{final,c},
            Equal.cong(Tree.Tree<Tile.Tile8>,Buffer.Content<Tile.Tile8>,tree => Buffer.Content{tree,Buffer.length(~Tile.Tile8,output)},Buffer.tree(~Tile.Tile8,output),final,tree_fresh),
            Equal.cong(Nat,Buffer.Content<Tile.Tile8>,length => Buffer.Content{final,length},Buffer.length(~Tile.Tile8,output),c,
              Equal.trans(Nat,Buffer.length(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),count))),c,length_equal,
                AllocationLaws.buffer_length(~Tile.Tile8,Window.initial(kind),count))))))))
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))),step,content_equal) :
    {Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,weights,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width},first,count,_) : Data}
  KernelGemm.Both{
    %Equal.sym(Buffer.Content<Tile.Tile8>,step,Buffer.Content{Buffer.content_tree(~Tile.Tile8,step),Buffer.content_length(~Tile.Tile8,step)},Buffer.content_eta(~Tile.Tile8,step)) :
      {Content.piece_values(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,weights,kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width},first,count,_) : Data}
    piece_leaf(~Holder,~value,~operation,cells),
    KernelGemm.Both{Shape.second(Buffer.ContentReady<Tile.Tile8,step>,{Buffer.content_length(~Tile.Tile8,step) == c : Nat},kept),
      Shape.first(Buffer.ContentReady<Tile.Tile8,step>,{Buffer.content_length(~Tile.Tile8,step) == c : Nat},kept)}}

# What one piece of a task needs: its room, its whole groups and its words,
# and the input rows its segments read.
type PieceFact<-low: Nat,-high: Nat,-capacity: U32,-kernel: U32,-stride: U32,-padding: U32,-height: U32,-width: U32,-groups: U32,-out_height: U32,-out_width: U32,-lanes: U32,-bound: U32,-first: U32,-count: U32> is Data:
  PieceFact{sized: AllocationDepth.Sized<count>,start_units: Nat,end_units: Nat,start_exact: {U32.to_nat(first) == Nat.mul(start_units,U32.to_nat(groups)) : Nat},end_exact: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == Nat.mul(end_units,U32.to_nat(groups)) : Nat},end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),end_units),U32.to_nat(bound)) == True{} : Bool},cells_room: {Nat.is_le(Nat.mul(end_units,U32.to_nat(groups)),U32.to_nat(bound)) == True{} : Bool},padded: Padding.padding_fits(bound,TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count),rooms: Piece.SegmentsRoom(U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width)}

def fact_rooms(-low: Nat,-high: Nat,-capacity: U32,-kernel: U32,-stride: U32,-padding: U32,-height: U32,-width: U32,-groups: U32,-out_height: U32,-out_width: U32,-lanes: U32,-bound: U32,-first: U32,-count: U32,+fact: PieceFact<low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,first,count>) -> Piece.SegmentsRoom(U32.to_nat((((first + count) / U32.max(groups,1) : U32) - (first / U32.max(groups,1) : U32) : U32)),U32.is_lt((first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32)),(first / U32.max(groups,1) : U32),((first + count) / U32.max(groups,1) : U32),first,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width):
  PieceFact{+sized,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+rooms} = fact
  rooms

# The facts of the pieces of a task from piece on, remaining of them.
def PieceRest(remaining: Nat,+low: Nat,+high: Nat,+capacity: U32,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+bound: U32,+grid: Grid.Grid,+piece: U32) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest: KernelGemm.Both<PieceFact<low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,Grid.start(grid,piece),Grid.span(grid,piece)>,PieceRest(rest,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,(piece + 1 : U32))>

# The facts of the pieces of a task from piece on, remaining more after it.
def PieceFacts(remaining: Nat,+low: Nat,+high: Nat,+capacity: U32,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+bound: U32,+grid: Grid.Grid,+piece: U32) -> Data:
  PieceRest(1n+remaining,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,piece)

# The pieces a task computes from piece on, what it holds after them, the
# runtime result and the values of the pieces.
type TaskPieces<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,-holder: Holder,-low: Nat,-high: Nat,-layout: Held.Spans,
  -parameters: Buffer.Model<Tile.Tile8>,-kind: Window.Sampling,-kernel: U32,-stride: U32,-padding: U32,-height: U32,-width: U32,-groups: U32,-out_height: U32,-out_width: U32,-lanes: U32,-grid: Grid.Grid,-remaining: Nat,-piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,-result: Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>> is Data:
  TaskPieces{pieces: PiecesModel.Model<Tile.Tile8>,holding: Held.Holding<Tile.Tile8>,tiled: Held.Tiled(~Tile.Tile8,holding,low,high),
    values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
    exact: {result == Pieces.Walked{PiecesModel.pack(~Tile.Tile8,pieces),lefts,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>},
    holds: PiecesModel.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width}),PiecesModel.content(~Tile.Tile8,pieces)),
    laid: {PiecesModel.layout(~Tile.Tile8,pieces) == GridRanges.pieces_layout(grid,remaining,piece) : Pieces.Ranges}}

# A piece followed by the later pieces of a task.
def joined(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+holder: Holder,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+grid: Grid.Grid,+parameters: Buffer.Model<Tile.Tile8>,+groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+piece: U32,+rest: Nat,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,+holding: Held.Holding<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+next: Held.Holding<Tile.Tile8>,
  +packed: {Window.computed(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},Grid.start(grid,piece),Grid.span(grid,piece),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters)) == Window.Computed{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.pack(~Tile.Tile8,output)),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters)} : Window.Computed},
  +held: Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.pack(~Tile.Tile8,output))))),later: TaskPieces<Holder,value,operation,holder,low,high,layout,parameters,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,rest,(piece + 1 : U32),Pieces.Left{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.pack(~Tile.Tile8,output)),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},Window.task_tree(~operation,rest,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,(piece + 1 : U32),Pieces.Walking{Pieces.Left{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.pack(~Tile.Tile8,output)),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},(Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters))})>) -> TaskPieces<Holder,value,operation,holder,low,high,layout,parameters,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,1n+rest,piece,lefts,Window.task_tree(~operation,1n+rest,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,piece,Pieces.Walking{lefts,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))})>:
  TaskPieces{+pieces,+after,+tiled,+values,+ordered,+exact,+holds,+laid} = later
  +first = Grid.start(grid,piece)
  +count = Grid.span(grid,piece)
  +whole = {PiecesModel.Fork{PiecesModel.Leaf{Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))),first,count},pieces} : PiecesModel.Model<Tile.Tile8>}
  TaskPieces{whole,after,tiled,values,ordered,
    %Equal.sym(Window.Computed,Window.computed(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},Grid.start(grid,piece),Grid.span(grid,piece),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters)),Window.Computed{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters)},packed) :
      {Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Window.Held(),Window.task_tree(~operation,rest,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,(piece + 1 : U32),Pieces.pushed(~Storage.Buffer<Tile.Tile8>,~Window.Held(),Window.computed_walked(lefts,_)))) ==
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,whole),lefts,(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>}
    %Equal.sym(Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>,Window.task_tree(~operation,rest,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,(piece + 1 : U32),Pieces.Walking{Pieces.Left{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},lefts},(Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters))}),
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,pieces),Pieces.Left{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},lefts},(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))},exact) :
      {Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Window.Held(),_) == Pieces.Walked{PiecesModel.pack(~Tile.Tile8,whole),lefts,(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>}
    %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))),TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),Buffer.remodeled(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))) :
      {Pieces.Walked{Partition.Fork{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},PiecesModel.pack(~Tile.Tile8,pieces)},lefts,(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))} ==
        Pieces.Walked{Partition.Fork{Partition.Leaf{_,first,count,Unit{}},PiecesModel.pack(~Tile.Tile8,pieces)},lefts,(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>}
    {==},
    KernelGemm.Both{held,holds},
    Equal.cong(Pieces.Ranges,Pieces.Ranges,right => Pieces.Split{Pieces.Range{first,count},right},PiecesModel.layout(~Tile.Tile8,pieces),
      GridRanges.pieces_layout(grid,rest,(piece + 1 : U32)),laid)}

# The pieces of a task from piece on: the first computed, remaining more.
def task_from(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,remaining: Nat,+holder: Holder,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+grid: Grid.Grid,+parameters: Buffer.Model<Tile.Tile8>,+groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,+holding: Held.Holding<Tile.Tile8>,+model: Piece.PieceModel<Holder,value,holder,Window.computed(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},Grid.start(grid,piece),Grid.span(grid,piece),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters)),low,high,layout,parameters,TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Piece.finish_tree(kind,Grid.span(grid,piece),groups,kernel,Buffer.tree(~Tile.Tile8,parameters),Piece.segments_tree(~Holder,~value,~operation,U32.to_nat((((Grid.start(grid,piece) + Grid.span(grid,piece)) / U32.max(groups,1) : U32) - (Grid.start(grid,piece) / U32.max(groups,1) : U32) : U32)),U32.is_lt((Grid.start(grid,piece) / U32.max(groups,1) : U32),((Grid.start(grid,piece) + Grid.span(grid,piece)) / U32.max(groups,1) : U32)),(Grid.start(grid,piece) / U32.max(groups,1) : U32),((Grid.start(grid,piece) + Grid.span(grid,piece)) / U32.max(groups,1) : U32),Grid.start(grid,piece),holder,kind,kernel,stride,padding,height,width,groups,out_height,out_width,Buffer.tree(~Tile.Tile8,parameters),Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),Grid.span(grid,piece)))))),Buffer.length(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Window.initial(kind),Grid.span(grid,piece))))>,+fact: PieceFact<low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,Grid.start(grid,piece),Grid.span(grid,piece)>,+later: PieceRest(remaining,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,(piece + 1 : U32))) -> TaskPieces<Holder,value,operation,holder,low,high,layout,parameters,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,remaining,piece,lefts,Window.task_tree(~operation,remaining,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,piece,Pieces.Walking{lefts,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))})>:
  match remaining:
    case 0n:
      Piece.PieceModel{+next,+output,+packed,+tiled,+values,+ordered,+tree_equal,+length_equal} = model
      PieceFact{+sized,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+rooms} = fact
      +first = Grid.start(grid,piece)
      +count = Grid.span(grid,piece)
          TaskPieces{PiecesModel.Leaf{Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output))),first,count},next,tiled,values,ordered,
        %Equal.sym(Window.Computed,Window.computed(~operation,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},Grid.start(grid,piece),Grid.span(grid,piece),Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters)),Window.Computed{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters)},packed) :
          {Window.computed_walked(lefts,_) == Pieces.Walked{Partition.Leaf{Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))),first,count,Unit{}},lefts,(Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>}
        %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))),TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),Buffer.remodeled(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)))) :
          {Pieces.Walked{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,Buffer.pack(~Tile.Tile8,output)),first,count,Unit{}},lefts,(Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters))} == Pieces.Walked{Partition.Leaf{_,first,count,Unit{}},lefts,(Held.pack(~Tile.Tile8,next),Buffer.pack(~Tile.Tile8,parameters))} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>}
        {==},
        piece_holds(~Holder,~value,~operation,holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,Grid.start(grid,piece),Grid.span(grid,piece),bound,bound_ok,divisors,start_units,end_units,start_exact,end_exact,end_room,cells_room,padded,sized,output,tree_equal,length_equal),{==}}
    case 1n+ +rest:
      Piece.PieceModel{+next,+output,+packed,+tiled,+values,+ordered,+tree_equal,+length_equal} = model
      PieceFact{+sized,+start_units,+end_units,+start_exact,+end_exact,+end_room,+cells_room,+padded,+rooms} = fact
      KernelGemm.Both{+next_fact,+after} = later
      +next_model = Piece.computed_model(~Holder,~value,~operation,holder,next,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,Grid.start(grid,(piece + 1 : U32)),Grid.span(grid,(piece + 1 : U32)),parameters,groups_room,groups_positive,fact_rooms(low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,Grid.start(grid,(piece + 1 : U32)),Grid.span(grid,(piece + 1 : U32)),next_fact))
      joined(~Holder,~value,~operation,holder,low,high,capacity,capacity_ok,bounded,layout,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,parameters,groups_room,groups_positive,bound,bound_ok,divisors,piece,rest,lefts,holding,output,next,packed,piece_holds(~Holder,~value,~operation,holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,Grid.start(grid,piece),Grid.span(grid,piece),bound,bound_ok,divisors,start_units,end_units,start_exact,end_exact,end_room,cells_room,padded,sized,output,tree_equal,length_equal),
        task_from(~Holder,~value,~operation,rest,holder,low,high,capacity,capacity_ok,bounded,layout,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,parameters,groups_room,groups_positive,bound,bound_ok,divisors,(piece + 1 : U32),Pieces.Left{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},Grid.start(grid,piece),Grid.span(grid,piece),Buffer.pack(~Tile.Tile8,output)),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},next,next_model,next_fact,after))

# The pieces of a task from piece on, remaining more after it.
def task_pieces(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,remaining: Nat,+holder: Holder,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+grid: Grid.Grid,+parameters: Buffer.Model<Tile.Tile8>,+groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,+holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),
  +values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),+ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
  +facts: PieceFacts(remaining,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,piece)) -> TaskPieces<Holder,value,operation,holder,low,high,layout,parameters,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,remaining,piece,lefts,Window.task_tree(~operation,remaining,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,piece,Pieces.Walking{lefts,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))})>:
  KernelGemm.Both{+fact,+later} = facts
  task_from(~Holder,~value,~operation,remaining,holder,low,high,capacity,capacity_ok,bounded,layout,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,parameters,groups_room,groups_positive,bound,bound_ok,divisors,piece,lefts,holding,Piece.computed_model(~Holder,~value,~operation,holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,Grid.start(grid,piece),Grid.span(grid,piece),parameters,groups_room,groups_positive,fact_rooms(low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,Grid.start(grid,piece),Grid.span(grid,piece),fact)),fact,later)

# ---- The task fact ----

def fact_of_pieces(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,~split_r: Buffer.Model<Tile.Tile8> -> PartitionLaws.Split<Buffer.Model<Tile.Tile8>>,+holder: Holder,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+grid: Grid.Grid,+parameters: Buffer.Model<Tile.Tile8>,+groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,
  +pieces: TaskPieces<Holder,value,operation,holder,low,high,layout,parameters,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first),Pieces.NoLefts{},Window.task_tree(~operation,Grid.later_pieces(grid,first,count),Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,Grid.piece_of(grid,first),Pieces.Walking{Pieces.NoLefts{},(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))})>) -> Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),split_r,Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),grid,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},holding,first,count,parameters>:
  TaskPieces{+model,+after,+tiled,+values,+ordered,+exact,+holds,+laid} = pieces
  Grouped.TaskFact{Partition.Leaf{model,first,count,Unit{}},parameters,
    %Equal.sym(Pieces.Walked<Storage.Buffer<Tile.Tile8>,Window.Held()>,Window.task_tree(~operation,Grid.later_pieces(grid,first,count),Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},grid,Grid.piece_of(grid,first),Pieces.Walking{Pieces.NoLefts{},(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,parameters))}),Pieces.Walked{PiecesModel.pack(~Tile.Tile8,model),Pieces.NoLefts{},(Held.pack(~Tile.Tile8,after),Buffer.pack(~Tile.Tile8,parameters))},exact) :
      {Window.task_finished(first,count,_) == (Partition.Leaf{PiecesModel.pack(~Tile.Tile8,model),first,count,Unit{}},Buffer.pack(~Tile.Tile8,parameters)) :
        Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>}
    {==},
    Held.Both{laid,holds}}

# A task computes the pieces of its cells from the window it holds: every
# piece holds the value of the operation at every cell.
def task_fact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,~split_r: Buffer.Model<Tile.Tile8> -> PartitionLaws.Split<Buffer.Model<Tile.Tile8>>,+holder: Holder,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+lanes: U32,+grid: Grid.Grid,+parameters: Buffer.Model<Tile.Tile8>,+groups_room: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(capacity)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+first: U32,+count: U32,+holding: Held.Holding<Tile.Tile8>,
  +tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
  +facts: PieceFacts(Grid.later_pieces(grid,first,count),low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,Grid.piece_of(grid,first))) -> Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile8>,Window.Windowing,PiecesModel.Model<Tile.Tile8>,Buffer.Model<Tile.Tile8>,Tensors.shared_buffer(~Tile.Tile8),Window.task_leaf(~operation),PiecesModel.pack(~Tile.Tile8),Buffer.pack(~Tile.Tile8),split_r,Grid.Grid,GridRanges.LaidHolds(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Problem<Holder>,~problem_value(~Holder,~value,~operation),Problem{holder,Buffer.tree(~Tile.Tile8,parameters),kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width})),grid,Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},holding,first,count,parameters>:
  fact_of_pieces(~Holder,~value,~operation,~split_r,holder,low,high,capacity,capacity_ok,bounded,layout,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,parameters,groups_room,groups_positive,bound,bound_ok,divisors,first,count,holding,
    task_pieces(~Holder,~value,~operation,Grid.later_pieces(grid,first,count),holder,low,high,capacity,capacity_ok,bounded,layout,kind,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,grid,parameters,groups_room,groups_positive,bound,bound_ok,divisors,Grid.piece_of(grid,first),Pieces.NoLefts{},holding,tiled,values,ordered,facts))

# ---- The facts of the pieces of a task ----

# A piece of a task of whole output positions inside the rows the task holds:
# its room, its whole groups, its words and the rows its segments read.
def piece_fact(+low: Nat,+high: Nat,+capacity: U32,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,
  +lanes: U32,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+allocatable: {U32.is_le(bound,1073741824) == True{} : Bool},
  +first_row: Nat,+last_row: Nat,+held: Reach.HeldRows<low,high,bound,width,groups,Reach.start_rows(first_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Reach.end_rows(last_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>,
  +sample_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(bound)) == True{} : Bool},
  +top_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(out_height),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(bound)) == True{} : Bool},
  +task_first: Nat,+task_end: Nat,+rows_first: {Nat.is_le(first_row,Nat.div(Nat.div(task_first,U32.to_nat(groups)),U32.to_nat(out_width))) == True{} : Bool},
  +rows_last: {Nat.is_le(Nat.div(Nat.sub(Nat.div(task_end,U32.to_nat(groups)),1n),U32.to_nat(out_width)),last_row) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.div(task_end,U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
  +end_cap: {Nat.is_le(task_end,U32.to_nat(bound)) == True{} : Bool},
  +grid: Grid.Grid,+piece: U32,+start_after: {Nat.is_le(task_first,U32.to_nat(Grid.start(grid,piece))) == True{} : Bool},
  +fits: Runs.Fits<U32.to_nat(groups),U32.to_nat(Grid.start(grid,piece)),Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece))),task_end>) ->
  PieceFact<low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,Grid.start(grid,piece),Grid.span(grid,piece)>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Runs.Fits{+a,+b,+low_exact,+high_exact,+less,+inside} = fits
  +g = U32.to_nat(groups)
  +ow = U32.to_nat(out_width)
  +f = Grid.start(grid,piece)
  +c = Grid.span(grid,piece)
  +s = U32.to_nat(f)
  +n = U32.to_nat(c)
  +end_bound = Order.transitive(Nat.add(s,n),task_end,U32.to_nat(bound),inside,end_cap)
  +n_bound = Order.transitive(n,Nat.add(s,n),U32.to_nat(bound),Order.right_le(s,n),end_bound)
  +a_exact = {%Equal.sym(Nat,s,Nat.mul(a,g),low_exact) : {Nat.div(_,g) == a : Nat} GridRanges.multiple_quotient(a,g,groups_positive) : {Nat.div(s,g) == a : Nat}}
  +b_exact = {%Equal.sym(Nat,Nat.add(s,n),Nat.mul(b,g),high_exact) : {Nat.div(_,g) == b : Nat} GridRanges.multiple_quotient(b,g,groups_positive) :
    {Nat.div(Nat.add(s,n),g) == b : Nat}}
  +b_task = {%b_exact : {Nat.is_le(_,Nat.div(task_end,g)) == True{} : Bool} DivisionOrder.quotient_le(Nat.add(s,n),task_end,g,groups_positive,inside) :
    {Nat.is_le(b,Nat.div(task_end,g)) == True{} : Bool}}
  +b_room = Order.transitive(Nat.add(ow,b),Nat.add(ow,Nat.div(task_end,g)),U32.to_nat(bound),Order.offset_le(ow,b,Nat.div(task_end,g),b_task),end_room)
  +position = (f / U32.max(groups,1) : U32)
  +finish = ((f + c) / U32.max(groups,1) : U32)
  +sum = Arithmetic.sum_exact(bound,bound_ok,f,c,s,n,{==},{==},end_bound)
  +position_exact = Equal.trans(Nat,U32.to_nat(position),Nat.div(s,g),a,Piece.word_quotient(f,groups,bound,bound_ok,groups_room,groups_positive),a_exact)
  +finish_exact = Equal.trans(Nat,U32.to_nat(finish),Nat.div(U32.to_nat((f + c : U32)),g),b,
    Piece.word_quotient((f + c : U32),groups,bound,bound_ok,groups_room,groups_positive),
    {%Equal.sym(Nat,U32.to_nat((f + c : U32)),Nat.add(s,n),sum) : {Nat.div(_,g) == b : Nat} b_exact : {Nat.div(U32.to_nat((f + c : U32)),g) == b : Nat}})
  +first_units = {%a_exact : {Nat.is_le(Nat.div(task_first,g),_) == True{} : Bool} DivisionOrder.quotient_le(task_first,s,g,groups_positive,start_after) :
    {Nat.is_le(Nat.div(task_first,g),a) == True{} : Bool}}
  +after = {%Equal.sym(Nat,U32.to_nat(position),a,position_exact) : {Nat.is_le(first_row,Nat.div(_,ow)) == True{} : Bool}
    Order.transitive(first_row,Nat.div(Nat.div(task_first,g),ow),Nat.div(a,ow),rows_first,DivisionOrder.quotient_le(Nat.div(task_first,g),a,ow,width_positive,first_units)) :
    {Nat.is_le(first_row,Nat.div(U32.to_nat(position),ow)) == True{} : Bool}}
  +ends = {%Equal.sym(Nat,U32.to_nat(finish),b,finish_exact) : {Nat.is_le(Nat.div(Nat.sub(_,1n),ow),last_row) == True{} : Bool}
    Order.transitive(Nat.div(Nat.sub(b,1n),ow),Nat.div(Nat.sub(Nat.div(task_end,g),1n),ow),last_row,
      DivisionOrder.quotient_le(Nat.sub(b,1n),Nat.sub(Nat.div(task_end,g),1n),ow,width_positive,DivisionOrder.subtract_le(b,Nat.div(task_end,g),1n,b_task)),rows_last) :
    {Nat.is_le(Nat.div(Nat.sub(U32.to_nat(finish),1n),ow),last_row) == True{} : Bool}}
  +finish_room = {%Equal.sym(Nat,U32.to_nat(finish),b,finish_exact) : {Nat.is_le(Nat.add(ow,_),U32.to_nat(bound)) == True{} : Bool} b_room :
    {Nat.is_le(Nat.add(ow,U32.to_nat(finish)),U32.to_nat(bound)) == True{} : Bool}}
  +zero_cells = {%Equal.sym(Nat,Nat.mul(g,0n),Nat.mul(0n,g),RowMajor.multiply_commutative(g,0n)) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool}
    Order.zero_le(U32.to_nat(bound)) : {Nat.is_le(Nat.mul(g,U32.to_nat(0)),U32.to_nat(bound)) == True{} : Bool}}
  PieceFact{AllocationDepth.bounded(c,Arithmetic.count_within(c,n,bound,1073741824,{==},n_bound,allocatable)),a,b,low_exact,high_exact,b_room,
    {%high_exact : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} end_bound : {Nat.is_le(Nat.mul(b,g),U32.to_nat(bound)) == True{} : Bool}},
    PaddedCells.fits_when(U32.is_zero((lanes % 8 : U32)),bound,g,U32.to_nat(0),s,n,KernelGemm.Both{groups_positive,KernelGemm.Both{groups_room,KernelGemm.Both{zero_cells,end_bound}}}),
    Reach.segments_room(U32.to_nat((finish - position : U32)),U32.is_lt(position,finish),position,finish,f,low,high,capacity,kernel,stride,padding,height,width,groups,
      out_height,out_width,bound,divisors,first_row,last_row,held,after,ends,finish_room,sample_room,top_room,{==})}

# The pieces of a task from piece on, remaining of them, each a run of whole
# output positions after low_start.
def piece_facts(remaining: Nat,+low: Nat,+high: Nat,+capacity: U32,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,
  +lanes: U32,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+allocatable: {U32.is_le(bound,1073741824) == True{} : Bool},
  +first_row: Nat,+last_row: Nat,+held: Reach.HeldRows<low,high,bound,width,groups,Reach.start_rows(first_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Reach.end_rows(last_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>,
  +sample_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(bound)) == True{} : Bool},
  +top_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(out_height),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(bound)) == True{} : Bool},
  +task_first: Nat,+task_end: Nat,+rows_first: {Nat.is_le(first_row,Nat.div(Nat.div(task_first,U32.to_nat(groups)),U32.to_nat(out_width))) == True{} : Bool},
  +rows_last: {Nat.is_le(Nat.div(Nat.sub(Nat.div(task_end,U32.to_nat(groups)),1n),U32.to_nat(out_width)),last_row) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.div(task_end,U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
  +end_cap: {Nat.is_le(task_end,U32.to_nat(bound)) == True{} : Bool},+grid: Grid.Grid,+piece: U32,+low_start: Nat,+low_after: {Nat.is_le(task_first,low_start) == True{} : Bool},
  +runs: GridRanges.PieceRuns(remaining,grid,piece,low_start,U32.to_nat(groups),task_end)) ->
  PieceRest(remaining,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,grid,piece):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{+fits,+later} = runs
      KernelGemm.Both{+after_low,+rest_runs} = later
      +s = U32.to_nat(Grid.start(grid,piece))
      +start_after = Order.transitive(task_first,low_start,s,low_after,after_low)
      KernelGemm.Both{piece_fact(low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,divisors,allocatable,first_row,last_row,held,sample_room,top_room,task_first,task_end,rows_first,rows_last,end_room,end_cap,grid,piece,start_after,fits),
        piece_facts(rest,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,lanes,bound,divisors,allocatable,first_row,last_row,held,sample_room,top_room,task_first,task_end,rows_first,rows_last,end_room,end_cap,grid,(piece + 1 : U32),Nat.add(s,U32.to_nat(Grid.span(grid,piece))),
          Order.transitive(task_first,s,Nat.add(s,U32.to_nat(Grid.span(grid,piece))),start_after,Order.le_add(s,U32.to_nat(Grid.span(grid,piece)))),rest_runs)}

# ---- The rows a task reads ----

# The first and last output rows of a task of whole output positions, as the
# words of its reach, inside the output rows of the images.
type TaskRows<-bound: U32,-groups: U32,-out_height: U32,-out_width: U32,-images: Nat,-first: U32,-count: U32> is Data:
  TaskRows{first_exact: {U32.to_nat((first / U32.max(groups,1) / U32.max(out_width,1) : U32)) == Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)) : Nat},
    last_exact: {U32.to_nat((((first + count) / U32.max(groups,1) - 1) / U32.max(out_width,1) : U32)) == Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)) : Nat},
    first_below: {Nat.is_lt(Nat.div(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),images) == True{} : Bool},last_below: {Nat.is_lt(Nat.div(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),images) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
    end_cap: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool}}

def rows_end_room(-bound: U32,-groups: U32,-out_height: U32,-out_width: U32,-images: Nat,-first: U32,-count: U32,+rows: TaskRows<bound,groups,out_height,out_width,images,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool}:
  TaskRows{+first_exact,+last_exact,+first_below,+last_below,+end_room,+end_cap} = rows
  end_room

def rows_end_cap(-bound: U32,-groups: U32,-out_height: U32,-out_width: U32,-images: Nat,-first: U32,-count: U32,+rows: TaskRows<bound,groups,out_height,out_width,images,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool}:
  TaskRows{+first_exact,+last_exact,+first_below,+last_below,+end_room,+end_cap} = rows
  end_cap

# A task of whole output positions inside the output cells of the images.
def task_rows(+bound: U32,+groups: U32,+out_height: U32,+out_width: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,+first: U32,+count: U32,
  +out_bound: Nat,+fits: Runs.Fits<U32.to_nat(groups),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),out_bound>,+out_cells: {Nat.is_le(out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups))) == True{} : Bool},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(groups)),U32.to_nat(bound)) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(bound)) == True{} : Bool}) -> TaskRows<bound,groups,out_height,out_width,images,first,count>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Runs.Fits{+a,+b,+low_exact,+high_exact,+less,+inside} = fits
  +g = U32.to_nat(groups)
  +ow = U32.to_nat(out_width)
  +oh = U32.to_nat(out_height)
  +f = U32.to_nat(first)
  +e = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +e_cells = Order.transitive(e,out_bound,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g),inside,out_cells)
  +e_cap = Order.transitive(e,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g),U32.to_nat(bound),e_cells,out_room)
  +q = Nat.div(e,g)
  +position_word = (first / U32.max(groups,1) : U32)
  +position_exact = Piece.word_quotient(first,groups,bound,bound_ok,groups_room,groups_positive)
  +first_exact = Equal.trans(Nat,U32.to_nat((position_word / U32.max(out_width,1) : U32)),Nat.div(U32.to_nat(position_word),ow),Nat.div(Nat.div(f,g),ow),
    Piece.word_quotient(position_word,out_width,bound,bound_ok,width_room,width_positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,ow),U32.to_nat(position_word),Nat.div(f,g),position_exact))
  +sum = Arithmetic.sum_exact(bound,bound_ok,first,count,f,U32.to_nat(count),{==},{==},e_cap)
  +end_word = ((first + count) / U32.max(groups,1) : U32)
  +end_exact = Equal.trans(Nat,U32.to_nat(end_word),Nat.div(U32.to_nat((first + count : U32)),g),q,
    Piece.word_quotient((first + count : U32),groups,bound,bound_ok,groups_room,groups_positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,g),U32.to_nat((first + count : U32)),e,sum))
  +a_exact = {%Equal.sym(Nat,f,Nat.mul(a,g),low_exact) : {Nat.div(_,g) == a : Nat} GridRanges.multiple_quotient(a,g,groups_positive) : {Nat.div(f,g) == a : Nat}}
  +b_exact = {%Equal.sym(Nat,e,Nat.mul(b,g),high_exact) : {Nat.div(_,g) == b : Nat} GridRanges.multiple_quotient(b,g,groups_positive) : {Nat.div(e,g) == b : Nat}}
  +a_less = RowMajor.cancel_multiply_lt(a,b,g,
    {%low_exact : {Nat.is_lt(_,Nat.mul(b,g)) == True{} : Bool} %high_exact : {Nat.is_lt(f,_) == True{} : Bool} less : {Nat.is_lt(Nat.mul(a,g),Nat.mul(b,g)) == True{} : Bool}})
  +q_positive = {%Equal.sym(Nat,q,b,b_exact) : {Nat.is_lt(0n,_) == True{} : Bool} Order.le_lt_trans(0n,a,b,Order.zero_le(a),a_less) : {Nat.is_lt(0n,q) == True{} : Bool}}
  +q_cap = Order.transitive(q,e,U32.to_nat(bound),
    Order.transitive(q,Nat.mul(q,g),e,RowMajor.product_covers(q,g,groups_positive),DivisionOrder.product_below(e,g,groups_positive)),e_cap)
  +last_word = (end_word - 1 : U32)
  +minus = Arithmetic.difference_exact(bound,bound_ok,end_word,1,q,1n,end_exact,{==},Order.strict_follow(0n,q,q_positive),q_cap)
  +last_exact = Equal.trans(Nat,U32.to_nat((last_word / U32.max(out_width,1) : U32)),Nat.div(U32.to_nat(last_word),ow),Nat.div(Nat.sub(q,1n),ow),
    Piece.word_quotient(last_word,out_width,bound,bound_ok,width_room,width_positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,ow),U32.to_nat(last_word),Nat.sub(q,1n),minus))
  +q_positions = {%GridRanges.multiple_quotient(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g,groups_positive) : {Nat.is_le(q,_) == True{} : Bool}
    DivisionOrder.quotient_le(e,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),g),g,groups_positive,e_cells) : {Nat.is_le(q,Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))) == True{} : Bool}}
  +before_q = {%Order.positive_has_last(q,q_positive) : {Nat.is_lt(Nat.sub(q,1n),_) == True{} : Bool}
    Order.lt_succ(Nat.sub(q,1n),Nat.sub(q,1n),Order.reflexive(Nat.sub(q,1n))) : {Nat.is_lt(Nat.sub(q,1n),q) == True{} : Bool}}
  +last_rows = DivisionOrder.quotient_below(Nat.sub(q,1n),Nat.mul(images,oh),ow,width_positive,Bounds.nat_trans(Nat.sub(q,1n),q,Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),before_q,q_positions))
  +last_below = DivisionOrder.quotient_below(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),images,oh,height_positive,last_rows)
  +a_last = {%Equal.sym(Nat,Nat.div(f,g),a,a_exact) : {Nat.is_le(_,Nat.sub(q,1n)) == True{} : Bool}
    %Equal.sym(Nat,q,b,b_exact) : {Nat.is_le(a,Nat.sub(_,1n)) == True{} : Bool} Order.before_last(a,b,a_less) : {Nat.is_le(Nat.div(f,g),Nat.sub(q,1n)) == True{} : Bool}}
  +first_last = DivisionOrder.quotient_le(Nat.div(f,g),Nat.sub(q,1n),ow,width_positive,a_last)
  TaskRows{first_exact,last_exact,Order.le_lt_trans(Nat.div(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),oh),Nat.div(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),oh),images,DivisionOrder.quotient_le(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),oh,height_positive,first_last),last_below),last_below,
    Order.transitive(Nat.add(ow,q),Nat.add(ow,Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))),U32.to_nat(bound),Order.offset_le(ow,q,Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),q_positions),positions_room),e_cap}

# The words of the span a task reads: it starts at its first cell and ends at
# the later of its first and last cells.
type SpanWords<-start: U32,-finish: U32,-low: Nat,-high: Nat> is Data:
  SpanWords{start_exact: {U32.to_nat(start) == low : Nat},finish_exact: {U32.to_nat(finish) == Nat.max(high,low) : Nat}}

def span_start_exact(-start: U32,-finish: U32,-low: Nat,-high: Nat,+words: SpanWords<start,finish,low,high>) -> {U32.to_nat(start) == low : Nat}:
  SpanWords{+start_exact,+finish_exact} = words
  start_exact

def span_finish_exact(-start: U32,-finish: U32,-low: Nat,-high: Nat,+words: SpanWords<start,finish,low,high>) -> {U32.to_nat(finish) == Nat.max(high,low) : Nat}:
  SpanWords{+start_exact,+finish_exact} = words
  finish_exact

# The span of the reach of a task in words.
def span_words(+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +grid: Grid.Grid,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<bound,kernel,stride,padding,height,width,groups,out_height,images>,+first: U32,+count: U32,
  +rows: TaskRows<bound,groups,out_height,out_width,images,first,count>) -> SpanWords<Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Nat.mul(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),Nat.mul(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Reach.ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  TaskRows{+first_exact,+last_exact,+first_below,+last_below,+end_room,+end_cap} = rows
  +h = U32.to_nat(height)
  +cell = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +first_row = (first / U32.max(groups,1) / U32.max(out_width,1) : U32)
  +last_row = (((first + count) / U32.max(groups,1) - 1) / U32.max(out_width,1) : U32)
  +low_word = ((first_row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_le(padding,((first_row % U32.max(out_height,1)) * stride : U32)),
    U32.min(height,(((first_row % U32.max(out_height,1)) * stride : U32) - padding : U32)),0)) * (width * groups : U32) : U32)
  +high_word = ((last_row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_lt(padding,((last_row % U32.max(out_height,1)) * stride + kernel : U32)),
    U32.min(height,(((last_row % U32.max(out_height,1)) * stride + kernel : U32) - padding : U32)),0)) * (width * groups : U32) : U32)
  +low_exact = Reach.low_exact(first_row,Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),first_exact,kernel,stride,padding,height,width,groups,out_height,out_width,bound,divisors,images,room,first_below)
  +high_exact = Reach.high_exact(last_row,Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),last_exact,kernel,stride,padding,height,width,groups,out_height,out_width,bound,divisors,images,room,last_below)
  +low_cells = Nat.mul(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +high_cells = Nat.mul(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +most = Nat.max(high_cells,low_cells)
  +low_bound = Order.transitive(low_cells,Nat.mul(Nat.mul(images,h),cell),U32.to_nat(bound),
    RowMajor.mul_le(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(images,h),cell,Reach.image_rows(Nat.div(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),images,h,
      Reach.low_rows(U32.to_nat(padding),Nat.mul(Nat.mod(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),h),first_below,
      Reach.low_inside(U32.to_nat(padding),Nat.mul(Nat.mod(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),h))),cells_room)
  +high_bound = Order.transitive(high_cells,Nat.mul(Nat.mul(images,h),cell),U32.to_nat(bound),
    RowMajor.mul_le(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(images,h),cell,Reach.image_rows(Nat.div(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),images,h,
      Reach.high_rows(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h),last_below,
      Reach.high_inside(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h))),cells_room)
  +most_word = U32.max(high_word,low_word)
  +most_exact = Equal.trans(Nat,U32.to_nat(most_word),Nat.max(U32.to_nat(high_word),U32.to_nat(low_word)),most,U32Extrema.maximum(high_word,low_word),
    Equal.trans(Nat,Nat.max(U32.to_nat(high_word),U32.to_nat(low_word)),Nat.max(high_cells,U32.to_nat(low_word)),most,
      Equal.cong(Nat,Nat,value => Nat.max(value,U32.to_nat(low_word)),U32.to_nat(high_word),high_cells,high_exact),
      Equal.cong(Nat,Nat,value => Nat.max(high_cells,value),U32.to_nat(low_word),low_cells,low_exact)))
  +ordered = Extrema.maximum_right(high_cells,low_cells)
  +most_bound = Extrema.maximum_upper(high_cells,low_cells,U32.to_nat(bound),high_bound,low_bound)
  +difference = Arithmetic.difference_exact(bound,bound_ok,most_word,low_word,most,low_cells,most_exact,low_exact,ordered,most_bound)
  +restored = Order.restore_shift(low_cells,most,ordered)
  SpanWords{low_exact,
    Equal.trans(Nat,U32.to_nat((low_word + (most_word - low_word : U32) : U32)),Nat.add(low_cells,Nat.sub(most,low_cells)),most,
      Arithmetic.sum_exact(bound,bound_ok,low_word,(most_word - low_word : U32),low_cells,Nat.sub(most,low_cells),low_exact,difference,
        %Equal.sym(Nat,Nat.add(low_cells,Nat.sub(most,low_cells)),most,restored) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} most_bound),
      restored)}

# The rows a task holds from its window: from the start of its reach to at
# least the end of it.
def task_held(+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +grid: Grid.Grid,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<bound,kernel,stride,padding,height,width,groups,out_height,images>,+first: U32,+count: U32,
  +rows: TaskRows<bound,groups,out_height,out_width,images,first,count>,+high: Nat,+reaches: {Nat.is_le(U32.to_nat(Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),high) == True{} : Bool}) ->
  Reach.HeldRows<U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),high,bound,width,groups,Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Reach.ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  TaskRows{+first_exact,+last_exact,+first_below,+last_below,+end_room,+end_cap} = rows
  +h = U32.to_nat(height)
  +cell = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +words = span_words(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,bound,divisors,images,room,first,count,rows)
  +low_cells = Nat.mul(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +high_cells = Nat.mul(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +end_rows = Reach.image_rows(Nat.div(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),images,h,
    Reach.high_rows(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h),last_below,
    Reach.high_inside(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h))
  Reach.HeldRows{
    %Equal.sym(Nat,U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),low_cells,span_start_exact(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),low_cells,high_cells,words)) :
      {Nat.is_le(_,low_cells) == True{} : Bool}
    Order.reflexive(low_cells),
    Order.transitive(high_cells,Nat.max(high_cells,low_cells),high,Extrema.maximum_left(high_cells,low_cells),
      %span_finish_exact(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),low_cells,high_cells,words) : {Nat.is_le(_,high) == True{} : Bool} reaches),
    Order.transitive(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(images,h),U32.to_nat(bound),end_rows,rows_room),
    Order.transitive(high_cells,Nat.mul(Nat.mul(images,h),cell),U32.to_nat(bound),RowMajor.mul_le(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(images,h),cell,end_rows),cells_room),
    groups_positive,allocatable}

# The reach of a task lies in the input cells and ends at or after it starts.
def task_reach(+kind: Window.Sampling,+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,+lanes: U32,+out_height: U32,+out_width: U32,
  +grid: Grid.Grid,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,
  +room: Reach.ReachRoom<bound,kernel,stride,padding,height,width,groups,out_height,images>,+first: U32,+count: U32,
  +rows: TaskRows<bound,groups,out_height,out_width,images,first,count>,+total: Nat,+input_cells: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),total) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(Nat.add(U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),Nat.sub(U32.to_nat(Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))))),total) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count))),U32.to_nat(Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)))) == True{} : Bool}>:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  Reach.ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  TaskRows{+first_exact,+last_exact,+first_below,+last_below,+end_room,+end_cap} = rows
  +h = U32.to_nat(height)
  +cell = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +words = span_words(kind,kernel,stride,padding,height,width,groups,lanes,out_height,out_width,grid,bound,divisors,images,room,first,count,rows)
  +low_cells = Nat.mul(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +high_cells = Nat.mul(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))
  +start = U32.to_nat(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)))
  +finish = U32.to_nat(Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)))
  +start_equal = span_start_exact(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),low_cells,high_cells,words)
  +finish_equal = span_finish_exact(Windows.span_start(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),Windows.span_end(Window.reach_of(Window.Windowing{kind,Window.Taps{kernel,stride,padding,height,width,groups},lanes,out_height,out_width,grid},first,count)),low_cells,high_cells,words)
  +inside = input_cells
  +low_inside = Order.transitive(low_cells,Nat.mul(Nat.mul(images,h),cell),total,
    RowMajor.mul_le(Reach.start_rows(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(images,h),cell,Reach.image_rows(Nat.div(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),images,h,
      Reach.low_rows(U32.to_nat(padding),Nat.mul(Nat.mod(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),h),first_below,
      Reach.low_inside(U32.to_nat(padding),Nat.mul(Nat.mod(Nat.div(Nat.div(U32.to_nat(first),U32.to_nat(groups)),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),h))),inside)
  +high_inside = Order.transitive(high_cells,Nat.mul(Nat.mul(images,h),cell),total,
    RowMajor.mul_le(Reach.end_rows(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(images,h),cell,Reach.image_rows(Nat.div(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),images,h,
      Reach.high_rows(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h),last_below,
      Reach.high_inside(U32.to_nat(padding),Nat.add(Nat.mul(Nat.mod(Nat.div(Nat.sub(Nat.div(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(groups)),1n),U32.to_nat(out_width)),U32.to_nat(out_height)),U32.to_nat(stride)),U32.to_nat(kernel)),h))),inside)
  +ordered = {%Equal.sym(Nat,start,low_cells,start_equal) : {Nat.is_le(_,finish) == True{} : Bool}
    %Equal.sym(Nat,finish,Nat.max(high_cells,low_cells),finish_equal) : {Nat.is_le(low_cells,_) == True{} : Bool}
    Extrema.maximum_right(high_cells,low_cells) : {Nat.is_le(start,finish) == True{} : Bool}}
  +finish_inside = {%Equal.sym(Nat,finish,Nat.max(high_cells,low_cells),finish_equal) : {Nat.is_le(_,total) == True{} : Bool}
    Extrema.maximum_upper(high_cells,low_cells,total,high_inside,low_inside) : {Nat.is_le(finish,total) == True{} : Bool}}
  Held.Both{%Equal.sym(Nat,Nat.add(start,Nat.sub(finish,start)),finish,Order.restore_shift(start,finish,ordered)) : {Nat.is_le(_,total) == True{} : Bool} finish_inside,ordered}
