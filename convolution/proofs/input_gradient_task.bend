# The tasks of the input gradient: the pieces a task computes hold, cell by
# cell, the tiles of the input gradient values of their cells.

import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_band_source.bend as BandSource
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_pieces_content.bend as Content
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_grid_ranges.bend as GridRanges
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/tensor_placed.bend as Placed
import ../../lib/proofs/tensor_tiled_padded_cells.bend as PaddedCells
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_batch_rows.bend as BatchRows
import ./input_gradient_piece_cells.bend as PieceCells
import ./input_gradient_task_reach.bend as TaskReach

# ---- What a task holds ----

# The first gradient cell a task reads.
def task_low(+call: Domain.Call,+task_first: U32,+task_count: U32) -> Nat:
  U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count)))

# The gradient cells a task holds and its packed weights.
type OperandModel<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,
  -low: Nat,-high: Nat,-layout: Held.Spans> is Data:
  OperandModel{holding: Held.Holding<Tile.Tile8>,tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
    ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},tiles: Buffer.Model<Tile.Tile8>,
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(call,weights)) : Tree.Tree<Tile.Tile8>}}

def operands_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,-low: Nat,-high: Nat,
  -layout: Held.Spans,model: OperandModel<Holder,value,holder,call,weights,low,high,layout>) -> Kernel.Operands():
  OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} = model
  (Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))
def holding_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,-low: Nat,-high: Nat,
  -layout: Held.Spans,model: OperandModel<Holder,value,holder,call,weights,low,high,layout>) -> Held.Holding<Tile.Tile8>:
  OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} = model
  holding
def tiles_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,-low: Nat,-high: Nat,
  -layout: Held.Spans,model: OperandModel<Holder,value,holder,call,weights,low,high,layout>) -> Storage.Buffer<Tile.Tile8>:
  OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} = model
  Buffer.pack(~Tile.Tile8,tiles)

# The values the pieces of a task hold: the input gradient of the call, read
# from the gradient of the holder and the weights.
def task_values(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32) ->
  Representation.Lifted<PieceCells.Setup>:
  Representation.Lifted{PieceCells.Setup{call,BandSource.gradient_model(~Holder,~value,holder,Rows.cells_word(call)),weights},PieceCells.cells_of(Domain.geometry(call),outer,after)}

# The pieces a task computes, what it holds after them, the runtime result
# and the values of the pieces.
type TaskPieces<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,
  -low: Nat,-high: Nat,-layout: Held.Spans,-outer: U32,-after: U32,-grid: Grid.Grid,-remaining: Nat,-piece: U32,
  -lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,-result: Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>> is Data:
  TaskPieces{pieces: PiecesModel.Model<Tile.Tile8>,next: OperandModel<Holder,value,holder,call,weights,low,high,layout>,
    exact: {result == Pieces.Walked{PiecesModel.pack(~Tile.Tile8,pieces),lefts,operands_of(~Holder,~value,holder,call,weights,low,high,layout,next)} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>},
    values: PiecesModel.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      task_values(~Holder,~value,holder,call,weights,outer,after)),PiecesModel.content(~Tile.Tile8,pieces)),
    laid: {PiecesModel.layout(~Tile.Tile8,pieces) == GridRanges.pieces_layout(grid,remaining,piece) : Pieces.Ranges}}

# The values of a computed piece, as the values of its content.
def piece_content(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+outer: U32,+after: U32,
  +first: U32,+count: U32,+output: Buffer.Model<Tile.Tile8>,
  +leaf: Content.Leaf(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    task_values(~Holder,~value,holder,call,weights,outer,after),first,count,
    Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))),
  +sized: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) == U32.to_nat(count) : Nat},
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,output)>) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    task_values(~Holder,~value,holder,call,weights,outer,after),first,count,Buffer.content(~Tile.Tile8,output)):
  KernelGemm.Both{{%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,output),
      Buffer.Content{Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))},
      Buffer.content_eta(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))) :
    Content.piece_values(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      task_values(~Holder,~value,holder,call,weights,outer,after),first,count,_)
  leaf : Content.piece_values(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    task_values(~Holder,~value,holder,call,weights,outer,after),first,count,Buffer.content(~Tile.Tile8,output))},KernelGemm.Both{sized,ready}}

# The piece of cells [first, first + count) inside the cells of its task,
# computed from what the task holds.
def computed_piece(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +task_first: U32,+task_count: U32,
  +task_inside: {Nat.is_le(Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),high) == True{} : Bool},
  +task_nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +first: U32,+count: U32,
  +cells_after: {Nat.is_le(U32.to_nat(task_first),U32.to_nat(first)) == True{} : Bool},
  +cells_before: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))) == True{} : Bool},
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout>) ->
  PieceCells.PieceComputed<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,first,count,
    Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),first,count,operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model))>:
  OperandModel{+holding,+tiled,+values,+ordered,+tiles,+tiles_exact} = model
  +total = Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))
  +task_rows = PieceCells.piece_rows(call,task_first,task_count,task_inside,total_room)
  +inside = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),total,cells_before,task_inside)
  +rows = PieceCells.piece_rows(call,first,count,inside,total_room)
  PieceCells.piece_task(~Holder,~value,holder,call,envelope,fits,weights,task_low(call,task_first,task_count),high,capacity,capacity_ok,bounded,layout,first,count,rows,
    outer,after,after_exact,outer_exact,values_room,holding,tiled,values,ordered,tiles,tiles_exact,
    TaskReach.piece_calls(~Holder,~value,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,task_first,task_count,task_rows,task_nonempty,reach_end,
      first,count,rows,cells_after,cells_before))

# ---- The pieces of a task ----

# The last piece of a task.
def last_piece(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+high: Nat,
  +layout: Held.Spans,+outer: U32,+after: U32,+task_first: U32,+task_count: U32,+grid: Grid.Grid,+piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout>,
  +computed: PieceCells.PieceComputed<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,Grid.start(grid,piece),Grid.span(grid,piece),
    Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),Grid.start(grid,piece),Grid.span(grid,piece),
      operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model))>) ->
  TaskPieces<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,grid,0n,piece,lefts,
    Kernel.task_tree(0n,Rows.task_of(Domain.geometry(call)),grid,piece,Pieces.Walking{lefts,operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model)})>:
  PieceCells.PieceComputed{+output,+holding,+tiled,+values,+ordered,+tiles,+tiles_exact,+exact,+leaf,+sized,+ready} = computed
  +low = task_low(call,task_first,task_count)
  +start = Grid.start(grid,piece)
  +span = Grid.span(grid,piece)
  +next = {OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} : OperandModel<Holder,value,holder,call,weights,low,high,layout>}
  TaskPieces{PiecesModel.Leaf{output,start,span},next,
    %Equal.sym(Storage.Buffer<Tile.Tile8> & Kernel.Operands(),Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),start,span,operands_of(~Holder,~value,holder,call,weights,low,high,layout,model)),
        (Buffer.pack(~Tile.Tile8,output),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),exact) :
      {Pieces.piece_walked(~Tile.Tile8,~Kernel.Operands(),lefts,start,span,_) ==
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,PiecesModel.Leaf{output,start,span}),lefts,operands_of(~Holder,~value,holder,call,weights,low,high,layout,next)} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>}
    {==},
    piece_content(~Holder,~value,holder,call,weights,outer,after,start,span,output,leaf,sized,ready),{==}}

# A piece followed by the later pieces of a task.
def joined(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+high: Nat,
  +layout: Held.Spans,+outer: U32,+after: U32,+task_first: U32,+task_count: U32,+grid: Grid.Grid,+rest: Nat,+piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout>,
  +output: Buffer.Model<Tile.Tile8>,+holding: Held.Holding<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>,
  +exact: {Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),Grid.start(grid,piece),Grid.span(grid,piece),
      operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model)) ==
    (Buffer.pack(~Tile.Tile8,output),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))) : Storage.Buffer<Tile.Tile8> & Kernel.Operands()},
  +leaf: Content.Leaf(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
    task_values(~Holder,~value,holder,call,weights,outer,after),Grid.start(grid,piece),Grid.span(grid,piece),
    Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))),
  +sized: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) == U32.to_nat(Grid.span(grid,piece)) : Nat},
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,output)>,
  +later: TaskPieces<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,grid,rest,(piece + 1 : U32),Pieces.Left{Partition.Leaf{Buffer.pack(~Tile.Tile8,output),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},
    Kernel.task_tree(rest,Rows.task_of(Domain.geometry(call)),grid,(piece + 1 : U32),Pieces.Walking{Pieces.Left{Partition.Leaf{Buffer.pack(~Tile.Tile8,output),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))})>) ->
  TaskPieces<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,grid,1n+rest,piece,lefts,
    Kernel.task_tree(1n+rest,Rows.task_of(Domain.geometry(call)),grid,piece,Pieces.Walking{lefts,operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model)})>:
  TaskPieces{+pieces,+after_model,+later_exact,+later_values,+laid} = later
  +low = task_low(call,task_first,task_count)
  +task = Rows.task_of(Domain.geometry(call))
  +start = Grid.start(grid,piece)
  +span = Grid.span(grid,piece)
  +whole = {PiecesModel.Fork{PiecesModel.Leaf{output,start,span},pieces} : PiecesModel.Model<Tile.Tile8>}
  TaskPieces{whole,after_model,
    %Equal.sym(Storage.Buffer<Tile.Tile8> & Kernel.Operands(),Kernel.piece_cells(task,start,span,operands_of(~Holder,~value,holder,call,weights,low,high,layout,model)),
        (Buffer.pack(~Tile.Tile8,output),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),exact) :
      {Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Kernel.Operands(),Kernel.task_tree(rest,task,grid,(piece + 1 : U32),Pieces.pushed(~Storage.Buffer<Tile.Tile8>,~Kernel.Operands(),Pieces.piece_walked(~Tile.Tile8,~Kernel.Operands(),lefts,start,span,_)))) ==
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,whole),lefts,operands_of(~Holder,~value,holder,call,weights,low,high,layout,after_model)} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>}
    %Equal.sym(Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>,Kernel.task_tree(rest,task,grid,(piece + 1 : U32),Pieces.Walking{Pieces.Left{Partition.Leaf{Buffer.pack(~Tile.Tile8,output),start,span,Unit{}},lefts},(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))}),
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,pieces),Pieces.Left{Partition.Leaf{Buffer.pack(~Tile.Tile8,output),start,span,Unit{}},lefts},operands_of(~Holder,~value,holder,call,weights,low,high,layout,after_model)},later_exact) :
      {Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Kernel.Operands(),_) == Pieces.Walked{PiecesModel.pack(~Tile.Tile8,whole),lefts,operands_of(~Holder,~value,holder,call,weights,low,high,layout,after_model)} : Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>}
    {==},
    KernelGemm.Both{piece_content(~Holder,~value,holder,call,weights,outer,after,start,span,output,leaf,sized,ready),later_values},
    Equal.cong(Pieces.Ranges,Pieces.Ranges,right => Pieces.Split{Pieces.Range{start,span},right},PiecesModel.layout(~Tile.Tile8,pieces),
      GridRanges.pieces_layout(grid,rest,(piece + 1 : U32)),laid)}

# The first of the pieces from piece on, from the cells of the runs.
def computed_first(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,remaining: Nat,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +task_first: U32,+task_count: U32,
  +task_inside: {Nat.is_le(Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),high) == True{} : Bool},
  +task_nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +grid: Grid.Grid,+unit: Nat,+piece: U32,+low_cell: Nat,+cells_low: {Nat.is_le(U32.to_nat(task_first),low_cell) == True{} : Bool},
  +runs: GridRanges.PieceRuns(1n+remaining,grid,piece,low_cell,unit,Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))),
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout>) ->
  PieceCells.PieceComputed<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,Grid.start(grid,piece),Grid.span(grid,piece),
    Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),Grid.start(grid,piece),Grid.span(grid,piece),
      operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model))>:
  KernelGemm.Both{+fits_piece,+later_runs} = runs
  KernelGemm.Both{+after_low,+rest_runs} = later_runs
  +start = U32.to_nat(Grid.start(grid,piece))
  +stop = Nat.add(start,U32.to_nat(Grid.span(grid,piece)))
  +bound = Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))
  computed_piece(~Holder,~value,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,outer,after,after_exact,outer_exact,values_room,total_room,
    task_first,task_count,task_inside,reach_end,task_nonempty,Grid.start(grid,piece),Grid.span(grid,piece),
    Order.transitive(U32.to_nat(task_first),low_cell,start,cells_low,after_low),Runs.fits_inside(unit,start,stop,bound,fits_piece),model)

# The pieces of a task from piece on, remaining more after it, the first of
# them computed.
def task_from(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,remaining: Nat,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +task_first: U32,+task_count: U32,
  +task_inside: {Nat.is_le(Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),high) == True{} : Bool},
  +task_nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +grid: Grid.Grid,+unit: Nat,+piece: U32,-lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,+low_cell: Nat,+cells_low: {Nat.is_le(U32.to_nat(task_first),low_cell) == True{} : Bool},
  +runs: GridRanges.PieceRuns(1n+remaining,grid,piece,low_cell,unit,Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))),
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout>,
  +computed: PieceCells.PieceComputed<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,Grid.start(grid,piece),Grid.span(grid,piece),
    Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),Grid.start(grid,piece),Grid.span(grid,piece),
      operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model))>) ->
  TaskPieces<Holder,value,holder,call,weights,task_low(call,task_first,task_count),high,layout,outer,after,grid,remaining,piece,lefts,
    Kernel.task_tree(remaining,Rows.task_of(Domain.geometry(call)),grid,piece,Pieces.Walking{lefts,operands_of(~Holder,~value,holder,call,weights,task_low(call,task_first,task_count),high,layout,model)})>:
  match remaining:
    case 0n:
      last_piece(~Holder,~value,holder,call,weights,high,layout,outer,after,task_first,task_count,grid,piece,lefts,model,computed)
    case 1n+ +rest:
      KernelGemm.Both{+fits_piece,+later_runs} = runs
      KernelGemm.Both{+after_low,+rest_runs} = later_runs
      PieceCells.PieceComputed{+output,+holding,+tiled,+values,+ordered,+tiles,+tiles_exact,+exact,+leaf,+sized,+ready} = computed
      +low = task_low(call,task_first,task_count)
      +next = {OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} : OperandModel<Holder,value,holder,call,weights,low,high,layout>}
      +start = U32.to_nat(Grid.start(grid,piece))
      +stop = Nat.add(start,U32.to_nat(Grid.span(grid,piece)))
      +start_after = Order.transitive(U32.to_nat(task_first),low_cell,start,cells_low,after_low)
      +next_low = Order.transitive(U32.to_nat(task_first),start,stop,start_after,Order.le_add(start,U32.to_nat(Grid.span(grid,piece))))
      +next_piece = (piece + 1 : U32)
      joined(~Holder,~value,holder,call,weights,high,layout,outer,after,task_first,task_count,grid,rest,piece,lefts,model,output,holding,tiles,exact,leaf,sized,ready,
        task_from(~Holder,~value,rest,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,outer,after,after_exact,outer_exact,values_room,total_room,
          task_first,task_count,task_inside,reach_end,task_nonempty,grid,unit,next_piece,Pieces.Left{Partition.Leaf{Buffer.pack(~Tile.Tile8,output),Grid.start(grid,piece),Grid.span(grid,piece),Unit{}},lefts},stop,next_low,rest_runs,next,
          computed_first(~Holder,~value,rest,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,outer,after,after_exact,outer_exact,values_room,total_room,
            task_first,task_count,task_inside,reach_end,task_nonempty,grid,unit,next_piece,stop,next_low,rest_runs,next)))

# ---- A task ----

# The output pieces of a task and the weights it hands back.
type TaskLeaf<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,
  -low: Nat,-high: Nat,-layout: Held.Spans,-outer: U32,-after: U32,-grid: Grid.Grid,-first: U32,-count: U32,
  -result: Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>> is Data:
  TaskLeaf{pieces: PiecesModel.Model<Tile.Tile8>,next: OperandModel<Holder,value,holder,call,weights,low,high,layout>,
    exact: {result == (Partition.Leaf{PiecesModel.pack(~Tile.Tile8,pieces),first,count,Unit{}},tiles_of(~Holder,~value,holder,call,weights,low,high,layout,next)) :
      Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>},
    values: PiecesModel.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<PieceCells.Setup>,~Representation.lifted(~PieceCells.Setup,~PieceCells.input_value),
      task_values(~Holder,~value,holder,call,weights,outer,after)),PiecesModel.content(~Tile.Tile8,pieces)),
    laid: {PiecesModel.layout(~Tile.Tile8,pieces) == GridRanges.pieces_layout(grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first)) : Pieces.Ranges}}

def task_finished(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+high: Nat,
  +layout: Held.Spans,+outer: U32,+after: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,first,count),high,layout>,
  +done: TaskPieces<Holder,value,holder,call,weights,task_low(call,first,count),high,layout,outer,after,grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first),Pieces.NoLefts{},
    Kernel.task_tree(Grid.later_pieces(grid,first,count),Rows.task_of(Domain.geometry(call)),grid,Grid.piece_of(grid,first),
      Pieces.Walking{Pieces.NoLefts{},operands_of(~Holder,~value,holder,call,weights,task_low(call,first,count),high,layout,model)})>) ->
  TaskLeaf<Holder,value,holder,call,weights,task_low(call,first,count),high,layout,outer,after,grid,first,count,
    Kernel.task_finished(first,count,Kernel.task_tree(Grid.later_pieces(grid,first,count),Rows.task_of(Domain.geometry(call)),grid,Grid.piece_of(grid,first),
      Pieces.Walking{Pieces.NoLefts{},operands_of(~Holder,~value,holder,call,weights,task_low(call,first,count),high,layout,model)}))>:
  TaskPieces{+pieces,+next,+exact,+pieces_values,+laid} = done
  OperandModel{+next_holding,+next_tiled,+next_values,+next_ordered,+next_tiles,+next_tiles_exact} = next
  +low = task_low(call,first,count)
  +whole = {OperandModel{next_holding,next_tiled,next_values,next_ordered,next_tiles,next_tiles_exact} : OperandModel<Holder,value,holder,call,weights,low,high,layout>}
  TaskLeaf{pieces,whole,
    %Equal.sym(Pieces.Walked<Storage.Buffer<Tile.Tile8>,Kernel.Operands()>,Kernel.task_tree(Grid.later_pieces(grid,first,count),Rows.task_of(Domain.geometry(call)),grid,Grid.piece_of(grid,first),
        Pieces.Walking{Pieces.NoLefts{},operands_of(~Holder,~value,holder,call,weights,low,high,layout,model)}),
        Pieces.Walked{PiecesModel.pack(~Tile.Tile8,pieces),Pieces.NoLefts{},(Held.pack(~Tile.Tile8,next_holding),Buffer.pack(~Tile.Tile8,next_tiles))},exact) :
      {Kernel.task_finished(first,count,_) == (Partition.Leaf{PiecesModel.pack(~Tile.Tile8,pieces),first,count,Unit{}},Buffer.pack(~Tile.Tile8,next_tiles)) :
        Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>}
    {==},pieces_values,laid}

# Runs of at least one piece end after they start.
def runs_less(remaining: Nat,+grid: Grid.Grid,+piece: U32,+low: Nat,+unit: Nat,+bound: Nat,+runs: GridRanges.PieceRuns(1n+remaining,grid,piece,low,unit,bound)) ->
  {Nat.is_lt(low,bound) == True{} : Bool}:
  KernelGemm.Both{+fits_piece,+later_runs} = runs
  KernelGemm.Both{+after_low,+rest_runs} = later_runs
  +start = U32.to_nat(Grid.start(grid,piece))
  +stop = Nat.add(start,U32.to_nat(Grid.span(grid,piece)))
  Order.le_lt_trans(low,start,bound,after_low,
    Bounds.nat_trans(start,stop,bound,Runs.fits_less(unit,start,stop,bound,fits_piece),Runs.fits_inside(unit,start,stop,bound,fits_piece)))

# A task: the input gradient pieces of cells [first, first + count) from the
# gradient cells and weights it holds.
def task_leaf(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(PieceCells.cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +grid: Grid.Grid,+unit: Nat,+first: U32,+count: U32,
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),high) == True{} : Bool},
  +runs: GridRanges.PieceRuns(1n+Grid.later_pieces(grid,first,count),grid,Grid.piece_of(grid,first),U32.to_nat(first),unit,Nat.add(U32.to_nat(first),U32.to_nat(count))),
  +model: OperandModel<Holder,value,holder,call,weights,task_low(call,first,count),high,layout>) ->
  TaskLeaf<Holder,value,holder,call,weights,task_low(call,first,count),high,layout,outer,after,grid,first,count,
    Kernel.task_leaf(Kernel.Work{Rows.task_of(Domain.geometry(call)),grid},first,count,
      Held.pack(~Tile.Tile8,holding_of(~Holder,~value,holder,call,weights,task_low(call,first,count),high,layout,model)),
      tiles_of(~Holder,~value,holder,call,weights,task_low(call,first,count),high,layout,model))>:
  OperandModel{+holding,+tiled,+values,+ordered,+tiles,+tiles_exact} = model
  +low = task_low(call,first,count)
  +piece = Grid.piece_of(grid,first)
  +later = Grid.later_pieces(grid,first,count)
  +f = U32.to_nat(first)
  +less = runs_less(later,grid,piece,f,unit,Nat.add(f,U32.to_nat(count)),runs)
  +task_rows = PieceCells.piece_rows(call,first,count,inside,total_room)
  +task_nonempty = PieceCells.rows_nonempty(call,first,count,task_rows,less)
  +whole = {OperandModel{holding,tiled,values,ordered,tiles,tiles_exact} : OperandModel<Holder,value,holder,call,weights,low,high,layout>}
  task_finished(~Holder,~value,holder,call,weights,high,layout,outer,after,grid,first,count,whole,
    task_from(~Holder,~value,later,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,outer,after,after_exact,outer_exact,values_room,total_room,
      first,count,inside,reach_end,task_nonempty,grid,unit,piece,Pieces.NoLefts{},f,Order.reflexive(f),runs,whole,
      computed_first(~Holder,~value,later,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,outer,after,after_exact,outer_exact,values_room,total_room,
        first,count,inside,reach_end,task_nonempty,grid,unit,piece,f,Order.reflexive(f),runs,whole)))
