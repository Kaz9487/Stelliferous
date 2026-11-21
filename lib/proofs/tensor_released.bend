# The pieces of a holding that cover the cells of a shape in the order of its
# grid, released and placed back in that grid, are a tensor of the contract.

import Base
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_move.bend as Move
import ../traversal_partition.bend as Partition
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_placed.bend as Placed
import ./tensor_move_readback.bend as Held
import ./tensor_task_runs.bend as Runs
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order

# A piece of a holding holds the values of the reader, as many cells as its
# count, in a ready buffer.
def piece_holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,
  +piece: Content.Leaf(~Element,~Logical,~original,logical,first,count,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),
  +sized: {Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},+ready: Buffer.ContentReady<Element,Buffer.content(~Element,buffer)>) ->
  Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,buffer)):
  match buffer:
    case Buffer.Model{+tree,+length,+certificate}: KernelGemm.Both{piece,KernelGemm.Both{sized,ready}}

# The pieces of a stack after a tree hold what the tree and each piece hold.
def unstacked_holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+stack: Held.Stack<Element>,+tree: PiecesModel.Model<Element>,
  +low: Nat,+high: Nat,+holds: PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,tree)),
  +tiles: Held.Tiles(~Element,stack,low,high),+values: Held.StackValues(~Element,~Logical,~original,logical,stack)) ->
  PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,Held.unstacked_model(~Element,stack,tree))):
  match stack:
    case Held.Bottom{}: holds
    case Held.Held{+buffer,+first,+count,+below}:
      Held.Both{Held.Starts{start,+sized,+ready},+rest} = tiles
      Held.Both{+piece,+later} = values
      unstacked_holds(~Element,~Logical,~original,logical,below,PiecesModel.Fork{tree,PiecesModel.Leaf{buffer,first,count}},Nat.add(low,U32.to_nat(count)),high,
        KernelGemm.Both{holds,piece_holds(~Element,~Logical,~original,logical,buffer,first,count,piece,sized,ready)},rest,later)

# Pieces in order that cover a range with the values of a reader, their
# ranges those of a run, released and placed in that layout are its placed
# model.
def stack_placed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+stack: Held.Stack<Element>,+fill: Element,+layout: Pieces.Ranges,
  +count: U32,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),+low: Nat,+high: Nat,+tiles: Held.Tiles(~Element,stack,low,high),
  +values: Held.StackValues(~Element,~Logical,~original,logical,stack),+ordered: {Held.spans(~Element,stack) == Held.layout_spans(layout,Held.Ended{}) : Held.Spans}) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),layout,
    Tensors.placed(~Element,layout,fill,Partition.Leaf{Move.released_stack(~Element,Move.first_leaf(~Element,Held.pack_stack(~Element,stack),fill)),0,count,Unit{}})>:
  match stack:
    case Held.Held{+buffer,+first,+pieces,+below}:
      Held.Both{Held.Starts{start,+sized,+ready},+rest} = tiles
      Held.Both{+piece,+later} = values
      +leaf = {PiecesModel.Leaf{buffer,first,pieces} : PiecesModel.Model<Element>}
      +tree = Held.unstacked_model(~Element,below,leaf)
      %Equal.sym(Pieces.Pieces(~Element),Move.unstacked(~Element,Held.pack_stack(~Element,below),PiecesModel.pack(~Element,leaf)),PiecesModel.pack(~Element,tree),
          Held.unstacked_packed(~Element,below,leaf)) :
        Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),layout,Tensors.placed(~Element,layout,fill,Partition.Leaf{_,0,count,Unit{}})>
      Placed.placed_model(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),layout,fill,Partition.Leaf{tree,0,count,Unit{}},Runs.range_low(layout),unit,bound,
        Runs.run_chain(layout,Held.Ended{},unit,bound,run,{==}),
        Equal.trans(Held.Spans,Held.layout_spans(PiecesModel.layout(~Element,tree),Held.Ended{}),Held.spans(~Element,Held.Held{buffer,first,pieces,below}),
          Held.layout_spans(layout,Held.Ended{}),Held.unstacked_spans(~Element,below,leaf),ordered),
        unstacked_holds(~Element,~Logical,~original,logical,below,leaf,Nat.add(low,U32.to_nat(pieces)),high,
          piece_holds(~Element,~Logical,~original,logical,buffer,first,pieces,piece,sized,ready),rest,later))
    case Held.Bottom{}:
      Empty.absurd(Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),layout,
          Tensors.placed(~Element,layout,fill,Partition.Leaf{Move.released_stack(~Element,Move.first_leaf(~Element,Held.pack_stack(~Element,Held.Bottom{}),fill)),0,count,Unit{}})>,
        WordEncoding.false_true(Equal.trans(Bool,False{},Held.spanned(Held.layout_spans(layout,Held.Ended{})),True{},
          Equal.cong(Held.Spans,Bool,Held.spanned,Held.Ended{},Held.layout_spans(layout,Held.Ended{}),ordered),Held.layout_spanned(layout,Held.Ended{}))))

# A holding that covers the cells of a shape in the order of its grid with
# the values of a reader, released and placed back in that grid, is a tensor
# of the contract of that reader.
def released_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -holding: Move.Holding<Element>,+unit: Nat,+counted: Global.Counted<shape>,+layout: GridRanges.GridLayout<shape,Grid.piece_target(),unit>,
  +covered: Held.Covered<Element,Logical,original,logical,holding,0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.placed(~Element,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,
    Partition.Leaf{Move.released(~Element,holding,fill),0,Tensors.count_or_zero(shape),Unit{}}),shape,fill,Tensors.Usable{},workers}):
  Held.Covered{+model,+packed,+tiled,+values,+ordered} = covered
  +ranges = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +total = ShapeLaws.elements(shape)
  %Equal.sym(Move.Holding<Element>,holding,Held.pack(~Element,model),packed) :
    Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.placed(~Element,ranges,fill,
      Partition.Leaf{Move.released(~Element,_,fill),0,Tensors.count_or_zero(shape),Unit{}}),shape,fill,Tensors.Usable{},workers})
  %Equal.sym(Pieces.Pieces(~Element),Move.released(~Element,Held.pack(~Element,model),fill),
      Move.released_stack(~Element,Move.first_leaf(~Element,Held.pack_stack(~Element,Held.order(~Element,model)),fill)),Held.released_order(~Element,model,fill)) :
    Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.placed(~Element,ranges,fill,
      Partition.Leaf{_,0,Tensors.count_or_zero(shape),Unit{}}),shape,fill,Tensors.Usable{},workers})
  GridRanges.layout_contract(~Element,~Logical,~original,logical,shape,fill,workers,
    Tensors.placed(~Element,ranges,fill,Partition.Leaf{Move.released_stack(~Element,Move.first_leaf(~Element,Held.pack_stack(~Element,Held.order(~Element,model)),fill)),0,
      Tensors.count_or_zero(shape),Unit{}}),unit,counted,layout,
    stack_placed(~Element,~Logical,~original,logical,Held.order(~Element,model),fill,ranges,Tensors.count_or_zero(shape),unit,total,
      GridRanges.layout_run(shape,Grid.piece_target(),unit,layout),0n,total,Held.order_tiles(~Element,model,0n,total,tiled),
      Held.order_values(~Element,~Logical,~original,logical,model,values),ordered))

# A covering of the cells of a nonempty counted shape within the capacity,
# released and placed back in its grid, is a tensor of the contract.
def released_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,
  +workers: U32,-holding: Move.Holding<Element>,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +covered: Held.Covered<Element,Logical,original,logical,holding,0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.placed(~Element,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,
    Partition.Leaf{Move.released(~Element,holding,fill),0,Tensors.count_or_zero(shape),Unit{}}),shape,fill,Tensors.Usable{},workers}):
  +divides = {GridRanges.Multiple{GridRanges.last_extent(shape),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(shape),1n),GridRanges.last_extent(shape),
      Order.times_one(GridRanges.last_extent(shape)))} : GridRanges.Multiple<GridRanges.last_extent(shape),1n>}
  released_contract(~Element,~Logical,~original,logical,shape,fill,workers,holding,1n,counted,
    GridRanges.grid_layout(limits,shape,Grid.piece_target(),room,nonempty,1n,divides),covered)
