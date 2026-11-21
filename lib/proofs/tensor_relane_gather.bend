# A relane gathers every target piece block by block from the source pieces
# its task holds: the model of the holding and the buffer it leaves.

import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ../tensor_move.bend as Move
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./tensor_move_readback.bend as Held
import ./tensor_moved.bend as Moved
import ./tensor_relane_blocks.bend as Blocks
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./machine_limit_bounds.bend as Limits
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_relane_piece.bend as Piece
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_grid_box.bend as Box
import ./tensor_pieces_content.bend as Content
import ./tensor_placed.bend as Placed
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_relane_values.bend as Values
import ./tensor_relane_reads.bend as Reads
import ./tensor_view_reads.bend as ViewReads
import ./tensor_components.bend as Components
import ./tensor_target_cells.bend as Targets
import ./tensor_taken_stack.bend as Taken
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_task_runs.bend as Runs
import ./boolean_logic.bend as Logic

# ---- The model of a gather ----

# The axes of the blocks of the target piece of count cells from first.
def gather_axes(+context: Tiled.Relaning,+first: U32,+count: U32) -> List<&2,Tiled.Ranged>:
  Tiled.Relaning{+walk,+axes} = context
  Tiled.Relane{source,source_within,source_position,target,target_within,target_position,lanes,+cells} = walk
  +extents = Grid.box_of(Tensors.grid_of(cells),cells,count)
  Tiled.ranged_axes(axes,Grid.coordinates(cells,first),extents,Tasks.strides_of(extents))

def fresh_output(+count: U32) -> Storage.Buffer<Tile.Tile8>:
  Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count)

# The holding and the buffer a relane gathers, as a model.
def relane_model(+context: Tiled.Relaning,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,+holdings: Moved.Holdings<Tile.Tile8>) ->
  Moved.Gathered<Tile.Tile8>:
  match holdings:
    case Moved.Holds{+holding,+rest}:
      +axes = gather_axes(context,first,count)
      +fuel = U32.to_nat(Tiled.ranged_head_count(axes))
      Moved.Gathered{Moved.Holds{Blocks.blocks_holding(fuel,0,axes,holding),rest},Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(count)))}
    case Moved.Unheld{}:
      Moved.Gathered{Moved.Unheld{},Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))}

# A relane gathers what its model says.
def relane_gathered(+context: Tiled.Relaning,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,+holdings: Moved.Holdings<Tile.Tile8>) ->
  {Tiled.relane_gather(context,components,first,count,fill,Moved.pack_holdings(~Tile.Tile8,holdings)) ==
    (Moved.pack_holdings(~Tile.Tile8,Moved.gathered_holdings(~Tile.Tile8,relane_model(context,components,first,count,fill,holdings))),
      Buffer.pack(~Tile.Tile8,Moved.gathered_output(~Tile.Tile8,relane_model(context,components,first,count,fill,holdings)))) :
    Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  Tiled.Relaning{+walk,+axes} = context
  Tiled.Relane{+source,+source_within,+source_position,+target,+target_within,+target_position,+lanes,+cells} = walk
  match holdings:
    case Moved.Holds{+holding,+rest}:
      +ranged = gather_axes(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,target_within,target_position,lanes,cells},axes},first,count)
      +fuel = U32.to_nat(Tiled.ranged_head_count(ranged))
      %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Tiled.blocks(fuel,0,ranged,(Held.pack(~Tile.Tile8,holding),fresh_output(count))),
          (Held.pack(~Tile.Tile8,Blocks.blocks_holding(fuel,0,ranged,holding)),Blocks.blocks_buffer(fuel,0,ranged,holding,fresh_output(count))),
          Blocks.blocks_packed(fuel,0,ranged,holding,fresh_output(count))) :
        {Tiled.relane_written(Moved.pack_holdings(~Tile.Tile8,rest),_) ==
          (Tensors.Holds{Held.pack(~Tile.Tile8,Blocks.blocks_holding(fuel,0,ranged,holding)),Moved.pack_holdings(~Tile.Tile8,rest)},
            Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,ranged,holding,fresh_output(count))))) :
          Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      %Buffer.remodeled(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,ranged,holding,fresh_output(count))) :
        {(Tensors.Holds{Held.pack(~Tile.Tile8,Blocks.blocks_holding(fuel,0,ranged,holding)),Moved.pack_holdings(~Tile.Tile8,rest)},_) ==
          (Tensors.Holds{Held.pack(~Tile.Tile8,Blocks.blocks_holding(fuel,0,ranged,holding)),Moved.pack_holdings(~Tile.Tile8,rest)},
            Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,ranged,holding,fresh_output(count))))) :
          Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      {==}
    case Moved.Unheld{}:
      %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)) :
        {(Tensors.Unheld{},_) == (Tensors.Unheld{},Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)))) :
          Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      {==}

# ---- What a relane writes into a target piece ----

# A target piece of a relane holds no more cells than the machine.
def span_room(+limits: Machine.Limits,+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32,
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Piece.piece_span(values,target,position,piece)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(Piece.piece_span(values,target,position,piece)),ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits)),
    Box.grid_span_room(limits,Tiled.cell_shape(values,target,position),Grid.piece_target(),cells_room,nonempty,piece,piece_inside),cells_room)

# A cell of a target piece of a relane, written by the blocks of the piece
# from a holding of every cell they read, holds the target value at its
# address.
def relane_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),low,high),
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +held_values: Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},holding),
  +held: Piece.BlocksHeld(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding))),+index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool}) ->
  Content.At(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},Piece.piece_first(values,target,position,piece),Blocks.blocks_tree(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),holding,Tree.fresh(Tile.Tile8,Tensors.depth_for(Piece.piece_span(values,target,position,piece)),Tiled.zero_tile())),U32.to_nat(Piece.piece_span(values,target,position,piece)),index):
  +span = Piece.piece_span(values,target,position,piece)
  +first = Piece.piece_first(values,target,position,piece)
  +axes = Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +fuel = U32.to_nat(Tiled.ranged_head_count(axes))
  +depth = Tensors.depth_for(span)
  +fresh = Tree.fresh(Tile.Tile8,depth,Tiled.zero_tile())
  +capacity = Machine.capacity(limits)
  +span_le = span_room(limits,values,target,position,piece,cells_room,nonempty,piece_inside)
  +allocatable = Limits.count_allocatable(limits,span,span_le)
  +back = Bounds.roundtrip(index,capacity,Limits.capacity_fits(limits),
    Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,U32.to_nat(span),U32.to_nat(capacity),index_in,span_le)))
  +query = U32.from_nat(index)
  +query_below = {%Equal.sym(Nat,U32.to_nat(query),index,back) : {Nat.is_lt(_,U32.to_nat(span)) == True{} : Bool}
    index_in : {Nat.is_lt(U32.to_nat(query),U32.to_nat(span)) == True{} : Bool}}
  +context = {Representation.Lifted{logical,Public.geometry_of(values,target,position)} : Representation.Lifted<Logical>}
  +expected = Representation.lifted(~Logical,~original,context,(first + query : U32))
  +sound = Piece.blocks_sound(~Logical,~original,fuel,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,0,query,holding,holding,low,high,{==},sorted,bounded,held_values,held,
    Piece.piece_step_room(within,position,target,source,source_within,source_position,values,piece,limits,source_cells_room,source_positive))
  +read = Blocks.blocks_read(fuel,0,axes,holding,depth,fresh,Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile()),Depth.allocation_depth(span,Depth.bounded(span,allocatable)),
    span,Depth.fresh_fits(~Tile.Tile8,span,Tiled.zero_tile(),allocatable),query,U32Division.nat_lt_native(query,span,query_below),expected,sound)
  +hit = Piece.piece_cover(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,query,query_below)
  +picked = {%hit : {Tree.read(Tile.Tile8,Blocks.blocks_tree(fuel,0,axes,holding,fresh),query) == Bool.pick(Tile.Tile8,_,expected,Tree.read(Tile.Tile8,fresh,query)) : Tile.Tile8}
    read : {Tree.read(Tile.Tile8,Blocks.blocks_tree(fuel,0,axes,holding,fresh),query) == Bool.pick(Tile.Tile8,True{},expected,Tree.read(Tile.Tile8,fresh,query)) : Tile.Tile8}}
  +address = Equal.trans(U32,(first + query : U32),U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(query))),U32.from_nat(Nat.add(U32.to_nat(first),index)),
    Arithmetic.native_add(first,query),Equal.cong(Nat,U32,number => U32.from_nat(Nat.add(U32.to_nat(first),number)),U32.to_nat(query),index,back))
  KernelGemm.Both{index_in,
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Blocks.blocks_tree(fuel,0,axes,holding,fresh),query),expected,
      Representation.lifted(~Logical,~original,context,U32.from_nat(Nat.add(U32.to_nat(first),index))),picked,
      Equal.cong(U32,Tile.Tile8,word => Representation.lifted(~Logical,~original,context,word),(first + query : U32),U32.from_nat(Nat.add(U32.to_nat(first),index)),address))}

# Every cell of a target piece of a relane from offset on holds the target
# value at its address.
def relane_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),low,high),
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +held_values: Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},holding),
  +held: Piece.BlocksHeld(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding))),remaining: Nat,+offset: Nat,+left: {Nat.is_le(Nat.add(offset,remaining),U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool}) ->
  Members.All(Nat,{number => Content.At(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},Piece.piece_first(values,target,position,piece),Blocks.blocks_tree(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),holding,Tree.fresh(Tile.Tile8,Tensors.depth_for(Piece.piece_span(values,target,position,piece)),Tiled.zero_tile())),U32.to_nat(Piece.piece_span(values,target,position,piece)),number) : Nat -> Data},
    Ranges.range(remaining,offset)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{relane_at(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,source_positive,piece_inside,box_cells,span_positive,holding,low,high,sorted,bounded,held_values,held,offset,Order.room_index(offset,rest,U32.to_nat(Piece.piece_span(values,target,position,piece)),left)),
        relane_cells(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,source_positive,piece_inside,box_cells,span_positive,holding,low,high,sorted,bounded,held_values,held,rest,1n+offset,Order.advance_room(offset,rest,U32.to_nat(Piece.piece_span(values,target,position,piece)),left))}

# A target piece of a relane holds the target value of every cell, as many
# cells as the piece, and the allocator can own it.
def relane_target(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+holding: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),low,high),
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +held_values: Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},holding),
  +held: Piece.BlocksHeld(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)))) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),
    Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(U32.to_nat(Tiled.ranged_head_count(Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),holding,fresh_output(Piece.piece_span(values,target,position,piece)))))):
  +span = Piece.piece_span(values,target,position,piece)
  +axes = Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +fuel = U32.to_nat(Tiled.ranged_head_count(axes))
  +depth = Tensors.depth_for(span)
  +fresh = Tree.fresh(Tile.Tile8,depth,Tiled.zero_tile())
  +tree = Blocks.blocks_tree(fuel,0,axes,holding,fresh)
  +allocatable = Limits.count_allocatable(limits,span,span_room(limits,values,target,position,piece,cells_room,nonempty,piece_inside))
  +exact = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(span)))),Buffer.contents(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(span))),
    Buffer.Content{tree,U32.to_nat(span)},Buffer.remodel_content(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(span))),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(span))),
      Buffer.Content{Blocks.blocks_tree(fuel,0,axes,holding,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,fresh_output(span)))),
        Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,fresh_output(span)))},
      Buffer.Content{tree,U32.to_nat(span)},Blocks.blocks_contents(fuel,0,axes,holding,fresh_output(span)),
      Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,
        content => Buffer.Content{Blocks.blocks_tree(fuel,0,axes,holding,Buffer.content_tree(~Tile.Tile8,content)),Buffer.content_length(~Tile.Tile8,content)},
        Buffer.contents(~Tile.Tile8,fresh_output(span)),Buffer.Content{fresh,U32.to_nat(span)},AllocationLaws.contents(~Tile.Tile8,depth,Tiled.zero_tile(),U32.to_nat(span)))))
  +tree_room = {%Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),Tree.capacity(Tile.Tile8,fresh),Blocks.blocks_capacity(fuel,0,axes,holding,fresh)) :
      {Nat.is_le(U32.to_nat(span),U32.to_nat(_)) == True{} : Bool}
    Bounds.as_nat_le(span,Tree.capacity(Tile.Tile8,fresh),Depth.fresh_fits(~Tile.Tile8,span,Tiled.zero_tile(),allocatable)) :
      {Nat.is_le(U32.to_nat(span),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Blocks.blocks_buffer(fuel,0,axes,holding,fresh_output(span)))),Buffer.Content{tree,U32.to_nat(span)},exact) :
    Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},Piece.piece_first(values,target,position,piece),span,_)
  KernelGemm.Both{relane_cells(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,source_positive,piece_inside,box_cells,span_positive,holding,low,high,sorted,bounded,held_values,held,U32.to_nat(span),0n,Order.reflexive(U32.to_nat(span))),
    KernelGemm.Both{{==},Buffer.ContentReady{depth,Blocks.blocks_balanced(fuel,0,axes,holding,depth,fresh,Shape.fresh(Tile.Tile8,depth,Tiled.zero_tile())),
      Depth.allocation_depth(span,Depth.bounded(span,allocatable)),tree_room}}}

# ---- What a relane gathers ----

# A relane gathers a target piece of a component as its model says, keeps
# the holdings in order and writes the target values into the piece.
def relane_law_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+fill: Tile.Tile8,+component: U32,+holding: Held.Holding<Tile.Tile8>,+rest: Moved.Holdings<Tile.Tile8>,
  +model: PiecesModel.Model<Tile.Tile8>,+more: Moved.Models<Tile.Tile8>,
  +held: PiecesModel.ContentAll(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},0),PiecesModel.content(~Tile.Tile8,model)),
  +laid: {Held.spans(~Tile.Tile8,Held.stacked_model(~Tile.Tile8,model,Held.Bottom{})) ==
    Reads.relane_spans(within,position,target,source,source_within,source_position,values,component) : Held.Spans},
  +ordered: {Held.order(~Tile.Tile8,holding) == Held.stacked_model(~Tile.Tile8,model,Held.Bottom{}) : Held.Stack<Tile.Tile8>},+later: Moved.HeldOrder(~Tile.Tile8,rest,more),
  +separate: Reads.RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_ok: Components.ShapeOk(limits,Tiled.cell_shape(values,source,source_position))) ->
  Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Holds{holding,rest},Moved.Models{model,more},
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Holds{holding,rest})):
  Held.Both{+source_fits,Held.Both{+source_positive,+source_rank}} = source_ok
  +axes = Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +fuel = U32.to_nat(Tiled.ranged_head_count(axes))
  +stack = Held.stacked_model(~Tile.Tile8,model,Held.Bottom{})
  +spans = Reads.relane_spans(within,position,target,source,source_within,source_position,values,component)
  +spans_exact = Equal.trans(Held.Spans,Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),Held.spans(~Tile.Tile8,stack),spans,
    Equal.cong(Held.Stack<Tile.Tile8>,Held.Spans,found => Held.spans(~Tile.Tile8,found),Held.order(~Tile.Tile8,holding),stack,ordered),laid)
  +blocks_held = {%Equal.sym(Held.Spans,Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),spans,spans_exact) : Piece.BlocksHeld(fuel,0,axes,_)
    Reads.blocks_held(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,fuel,component,0,separate,blocks,member,source_positive) :
    Piece.BlocksHeld(fuel,0,axes,Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)))}
  +row = Reads.relane_row(within,position,target,source,source_within,source_position,values)
  +classes = Reads.relane_classes(within,position,target,source,source_within,source_position,values,component)
  +sorted = {%Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,holding),stack,ordered) : Held.Sorted(~Tile.Tile8,_,0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)))
    Taken.row_sorted(~Tile.Tile8,~Targets.TargetClassesOk(classes,Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits)),
      limits,Tiled.cell_shape(values,source,source_position),row,classes,0,Tasks.separable([row]),model,source_ok,separate,
      PiecesModel.content_map(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},0),~Runs.piece_sized(~Tile.Tile8),~Placed.holds_sized(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),
        PiecesModel.content(~Tile.Tile8,model),held),laid) :
    Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,holding),0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)))}
  +held_values = Placed.ordered_values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},model,holding,ordered,held)
  Held.Both{relane_gathered(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Holds{holding,rest}),
    Held.Both{Held.Both{Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,Blocks.blocks_holding(fuel,0,axes,holding)),Held.order(~Tile.Tile8,holding),stack,
          Blocks.blocks_order(fuel,0,axes,holding),ordered),later},
      relane_target(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,source_positive,piece_inside,box_cells,span_positive,holding,0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),sorted,source_cells_room,held_values,blocks_held)}}

# A relane gathers a target piece of a component, a piece the component
# lists, as the gather law says.
def relane_law_piece(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},+fill: Tile.Tile8,+component: U32,+holding: Held.Holding<Tile.Tile8>,+rest: Moved.Holdings<Tile.Tile8>,
  +model: PiecesModel.Model<Tile.Tile8>,+more: Moved.Models<Tile.Tile8>,
  +held: PiecesModel.ContentAll(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},0),PiecesModel.content(~Tile.Tile8,model)),
  +laid: {Held.spans(~Tile.Tile8,Held.stacked_model(~Tile.Tile8,model,Held.Bottom{})) ==
    Reads.relane_spans(within,position,target,source,source_within,source_position,values,component) : Held.Spans},
  +ordered: {Held.order(~Tile.Tile8,holding) == Held.stacked_model(~Tile.Tile8,model,Held.Bottom{}) : Held.Stack<Tile.Tile8>},+later: Moved.HeldOrder(~Tile.Tile8,rest,more),
  +separate: Reads.RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Holds{holding,rest},Moved.Models{model,more},
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Holds{holding,rest})):
  +cells = Tiled.cell_shape(values,target,position)
  +cells_rank = Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room)
  +row = Reads.relane_row(within,position,target,source,source_within,source_position,values)
  +classes = Reads.relane_classes(within,position,target,source,source_within,source_position,values,component)
  +piece_inside = Targets.target_piece_below(limits,cells,Grid.piece_target(),cells_room,nonempty,cells_rank,[row],component,
    Targets.when_third(~Components.RowOk(limits,row,Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)),~Components.RowEnds(row,classes),
      ~Targets.TargetClassesOk(classes,Tensors.grid_of(cells),cells,Machine.half_word(limits)),Tasks.separable([row]),separate),piece,member)
  +span_positive = Targets.grid_span_positive(limits,cells,Grid.piece_target(),cells_room,nonempty,piece,piece_inside)
  +box_cells = Targets.grid_box_cells(limits,cells,Grid.piece_target(),cells_room,nonempty,cells_rank,piece,piece_inside)
  +source_ok = {Held.Both{source_cells_room,Held.Both{source_nonempty,Piece.cells_rank_room(limits,values,source,source_position,source_within,source_value,rank_room)}} :
    Components.ShapeOk(limits,Tiled.cell_shape(values,source,source_position))}
  relane_law_at(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,fill,component,holding,rest,model,more,held,laid,ordered,later,separate,blocks,member,source_ok)

# What a relane of a shape needs: the shape and both cell shapes fit the
# machine, the lane axes are apart, and the readings of the relane have class
# lists, blocks and rows that fit.
def RelaneFacts(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool) -> Data:
  Held.Both<{Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  Held.Both<{Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  Held.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  Held.Both<{Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  Held.Both<{Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  Held.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  Held.Both<{Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  Held.Both<{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  Held.Both<ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  Held.Both<Components.ComponentsOk([Reads.relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Machine.half_word(limits)),
  Held.Both<{Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),[Reads.relane_row(within,position,target,source,source_within,source_position,values)]))) == True{} : Bool},
  Held.Both<Components.LimitsAt([Reads.relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Tiled.cell_shape(values,target,position),Tensors.grid_of(Tiled.cell_shape(values,target,position)),0),
  Logic.When(Tasks.separable([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Components.RowOk(limits,Reads.relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)))>>>>>>>>>>>>>>>>>>

# A gather context agrees with a relane of a shape: it walks the relane of the
# shape through its components, from the one source, into the target cells.
def RelaneAgrees(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+context: Tiled.Relaning,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+cells: List<&2,U32>,+cells_grid: Grid.Grid,+fill: Tile.Tile8) -> Data:
  Held.Both<{context == Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)} : Tiled.Relaning},
  Held.Both<{components == Reads.relane_components(within,position,target,source,source_within,source_position,values) : Tasks.Components},
  Held.Both<{origins == [Tensors.origin(Tiled.cell_shape(values,source,source_position),0)] : List<&2,Tensors.Origin>},
  Held.Both<{cells == Tiled.cell_shape(values,target,position) : List<&2,U32>},
  Held.Both<{cells_grid == Tensors.grid_of(Tiled.cell_shape(values,target,position)) : Grid.Grid},
    RelaneFacts(limits,values,source,source_position,source_within,target,position,within)>>>>>

def relane_law_held(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +components_ok: Components.ComponentsOk([Reads.relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),[Reads.relane_row(within,position,target,source,source_within,source_position,values)]))) == True{} : Bool},
  +limits_at: Components.LimitsAt([Reads.relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Tiled.cell_shape(values,target,position),Tensors.grid_of(Tiled.cell_shape(values,target,position)),0),
  +row_ok: Logic.When(Tasks.separable([Reads.relane_row(within,position,target,source,source_within,source_position,values)]),Components.RowOk(limits,Reads.relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position))),+fill: Tile.Tile8,+component: U32,+models: Moved.Models<Tile.Tile8>,+holdings: Moved.Holdings<Tile.Tile8>,
  +ready: Moved.TakenReady(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),Reads.relane_components(within,position,target,source,source_within,source_position,values),[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],models,component),
  +component_in: {Nat.is_lt(U32.to_nat(component),U32.to_nat(Tasks.components_count(Reads.relane_components(within,position,target,source,source_within,source_position,values)))) == True{} : Bool},+order: Moved.HeldOrder(~Tile.Tile8,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece)) ->
  Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,holdings,models,
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,holdings)):
  match models holdings:
    case Moved.Models{+model,+more} Moved.Holds{+holding,+rest}:
      Held.Both{Held.Both{+held,+held_rest},Held.Both{+laid,+laid_rest}} = ready
      Held.Both{+ordered,+later} = order
      +cells_ok = {Held.Both{cells_room,Held.Both{nonempty,Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room)}} :
        Components.ShapeOk(limits,Tiled.cell_shape(values,target,position))}
      +separate = Reads.relane_separate(limits,within,position,target,source,source_within,source_position,values,component,within_value,cells_ok,components_ok,limits_at,
        Targets.component_lists(limits,[Reads.relane_row(within,position,target,source,source_within,source_position,values)],components_ok,lists_positive,component,component_in),row_ok)
      relane_law_piece(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,fill,component,holding,rest,model,more,held,laid,ordered,later,separate,blocks,member,source_nonempty)
    case Moved.NoModels{} +holdings:
      Held.Both{+none,+laid} = ready
      Empty.absurd(Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,holdings,Moved.NoModels{},
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,holdings)),none)
    case Moved.Models{+model,+more} Moved.Unheld{}:
      Empty.absurd(Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Unheld{},Moved.Models{model,more},
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Piece.piece_first(values,target,position,piece),Piece.piece_span(values,target,position,piece),fill,Moved.Unheld{})),order)

# A relane gathers every target piece of its components with the target
# values of its cells, as its model says, keeping the holdings in order.
def relane_law(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+cells: List<&2,U32>,+cells_grid: Grid.Grid,+fill: Tile.Tile8,+context: Tiled.Relaning,+component: U32,+piece: U32,
  +models: Moved.Models<Tile.Tile8>,+holdings: Moved.Holdings<Tile.Tile8>,
  +agrees: RelaneAgrees(limits,values,source,source_position,source_within,target,position,within,context,components,origins,cells,cells_grid,fill),
  +ready: Moved.TakenReady(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),components,origins,models,component),
  +component_in: {Nat.is_lt(U32.to_nat(component),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},+order: Moved.HeldOrder(~Tile.Tile8,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(components,cells_grid,cells,component),piece)) ->
  Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
    relane_model(context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings)):
  Held.Both{+same_context,Held.Both{+same_components,Held.Both{+same_origins,Held.Both{+same_cells,Held.Both{+same_grid,+facts}}}}} = agrees
  Held.Both{+within_value,Held.Both{+inside,Held.Both{+room,Held.Both{+lanes_room,Held.Both{+cells_room,Held.Both{+nonempty,Held.Both{+source_value,Held.Both{+source_inside,Held.Both{+source_room,Held.Both{+source_cells_room,Held.Both{+source_lanes_room,Held.Both{+rank_room,Held.Both{+apart,Held.Both{+source_nonempty,Held.Both{+blocks,Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+row_ok}}}}}}}}}}}}}}}}}} = facts
  %Equal.sym(Tiled.Relaning,context,Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},same_context) :
    Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    _,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
    relane_model(_,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Tasks.Components,components,Reads.relane_components(within,position,target,source,source_within,source_position,values),same_components) :
    Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Grid.Grid,cells_grid,Tensors.grid_of(Tiled.cell_shape(values,target,position)),same_grid) :
    Moved.GatherFacts(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),
    Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Grid.start(_,piece),Grid.span(_,piece),fill,holdings,models,
    relane_model(Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},Reads.relane_components(within,position,target,source,source_within,source_position,values),Grid.start(_,piece),Grid.span(_,piece),fill,holdings))
  relane_law_held(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,source_nonempty,blocks,components_ok,lists_positive,limits_at,row_ok,fill,component,models,holdings,
    {%same_origins : Moved.TakenReady(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),Reads.relane_components(within,position,target,source,source_within,source_position,values),_,models,component)
      %same_components : Moved.TakenReady(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),_,origins,models,component)
      ready : Moved.TakenReady(~Tile.Tile8,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),Reads.relane_components(within,position,target,source,source_within,source_position,values),[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],models,component)},
    {%same_components : {Nat.is_lt(U32.to_nat(component),U32.to_nat(Tasks.components_count(_))) == True{} : Bool} component_in :
      {Nat.is_lt(U32.to_nat(component),U32.to_nat(Tasks.components_count(Reads.relane_components(within,position,target,source,source_within,source_position,values)))) == True{} : Bool}},order,
    {%same_cells : Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,target,position)),_,component),piece)
      %same_grid : Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),_,cells,component),piece)
      %same_components : Moved.Member(Tasks.target_pieces(_,cells_grid,cells,component),piece)
      member : Moved.Member(Tasks.target_pieces(Reads.relane_components(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece)})
