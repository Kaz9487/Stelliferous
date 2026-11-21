# The cells of a relane: the target cells, gathered block by block from the
# source pieces each task holds, hold the lifted values of the reader with
# the lanes on the new axis.

import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_moved_sources.bend as Sources
import ./tensor_global_shape_count.bend as Global
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_relane_values.bend as Values
import ./tensor_relane_piece.bend as Piece
import ./tensor_relane_reads.bend as Reads
import ./tensor_relane_gather.bend as Gather
import ./nat_order.bend as Order
import ./tensor_relane_facts.bend as Facts
import ./tensor_tiled_repositioned.bend as Repositioned
import ./tensor_storage_count.bend as StorageCount
import ../default_machine_limits.bend as Default
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ../tensor_tiled_cells.bend as Cells
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Boolean
import ../kernel_gemm.bend as KernelGemm

# The pieces a relane of source cells laid as canonical pieces moves hold the
# target values of the relane.
def relane_placed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+model: PiecesModel.Model<Tile.Tile8>,+workers: U32,
  +placed: Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),Pieces.ranges_of_grid(Tensors.grid_of(Tiled.cell_shape(values,source,source_position))),PiecesModel.pack(~Tile.Tile8,model)>,
  +facts: Gather.RelaneFacts(limits,values,source,source_position,source_within,target,position,within)) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Tiled.cell_shape(values,target,position))),
    Tensors.moved_with(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,Reads.relane_components(within,position,target,source,source_within,source_position,values),Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},9.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],Tiled.cell_shape(values,target,position),Tiled.zero_tile(),workers)>:
  Placed.PlacedModel{+source_model,+exact,+laid,+holds} = placed
  Held.Both{+within_value,Held.Both{+inside,Held.Both{+room,Held.Both{+lanes_room,Held.Both{+cells_room,Held.Both{+nonempty,Held.Both{+source_value,Held.Both{+source_inside,Held.Both{+source_room,Held.Both{+source_cells_room,Held.Both{+source_lanes_room,Held.Both{+rank_room,Held.Both{+apart,Held.Both{+source_nonempty,Held.Both{+blocks,Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+row_ok}}}}}}}}}}}}}}}}}} = facts
  +row = Reads.relane_row(within,position,target,source,source_within,source_position,values)
  +separate = Tasks.separable([row])
  +origin = Tensors.origin(Tiled.cell_shape(values,source,source_position),0)
  +context = {Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)} : Tiled.Relaning}
  +source_ok = {Held.Both{source_cells_room,Held.Both{source_nonempty,Piece.cells_rank_room(limits,values,source,source_position,source_within,source_value,rank_room)}} :
    Components.ShapeOk(limits,Tiled.cell_shape(values,source,source_position))}
  +target_ok = {Held.Both{cells_room,Held.Both{nonempty,Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room)}} :
    Components.ShapeOk(limits,Tiled.cell_shape(values,target,position))}
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,source_model),exact) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Tiled.cell_shape(values,target,position))),
      Tensors.moved_with(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,Reads.relane_components(within,position,target,source,source_within,source_position,values),Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},9.0,16.0,Tensors.Source{_,Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],Tiled.cell_shape(values,target,position),Tiled.zero_tile(),workers)>
  Components.moved_with_model(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,~Gather.relane_model,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)}),
    ~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)}),~Gather.RelaneAgrees(limits,values,source,source_position,source_within,target,position,within),~Gather.relane_law(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within),
    separate,limits,[row],Tasks.target_rank([row]),[origin],Tiled.cell_shape(values,target,position),Tiled.zero_tile(),context,
    Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},facts}}}}},
    9.0,16.0,workers,Moved.Models{source_model,Moved.NoModels{}},components_ok,{==},
    Held.Both{Held.Both{{==},Held.Both{{==},source_ok}},Held.Both{Sources.origin_when(separate,limits,row,Tiled.cell_shape(values,source,source_position),0,row_ok),Unit{}}},
    lists_positive,target_ok,limits_at,
    Held.Both{laid,Held.Both{Components.shape_ordered(limits,Tiled.cell_shape(values,source,source_position),source_ok),Held.Both{Sources.empty_holds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Tiled.zero_tile()),Unit{}}}},
    Held.Both{Sources.good_of(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},source_model,Held.Ended{},holds),Unit{}})

# A relane of source cells of the contract, counted target cells and the
# facts a relane needs, gives target cells of the contract holding the target
# values.
def relane_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,
  +workers: U32,+contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,source,source_position),fill,Tensors.Usable{},workers}),
  +facts: Gather.RelaneFacts(limits,values,source,source_position,source_within,target,position,within),+counted: Global.Counted<Tiled.cell_shape(values,target,position)>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,Reads.relane_components(within,position,target,source,source_within,source_position,values),Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},9.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],Tiled.cell_shape(values,target,position),Tiled.zero_tile(),workers),Tiled.cell_shape(values,target,position),Tiled.zero_tile(),Tensors.Usable{},workers}):
  Held.Both{+within_value,Held.Both{+inside,Held.Both{+room,Held.Both{+lanes_room,Held.Both{+cells_room,Held.Both{+nonempty,Held.Both{+source_value,Held.Both{+source_inside,Held.Both{+source_room,Held.Both{+source_cells_room,Held.Both{+source_lanes_room,Held.Both{+rank_room,Held.Both{+apart,Held.Both{+source_nonempty,Held.Both{+blocks,Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+row_ok}}}}}}}}}}}}}}}}}} = facts
  +cells = Tiled.cell_shape(values,target,position)
  GridRanges.grid_contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,target,position)},limits,cells,Tiled.zero_tile(),workers,Tensors.moved_with(~Tile.Tile8,~Tiled.Relaning,~Tiled.relane_gather,Reads.relane_components(within,position,target,source,source_within,source_position,values),Tiled.Relaning{Tiled.Relane{source,source_within,source_position,target,within,position,Grid.lanes_at(values,target),Tiled.cell_shape(values,target,position)},Values.values_blocked(within,position,target,source,source_within,source_position,values)},9.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,source,source_position),0)],Tiled.cell_shape(values,target,position),Tiled.zero_tile(),workers),cells_room,nonempty,
    1n,GridRanges.Multiple{GridRanges.last_extent(cells),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(cells),1n),GridRanges.last_extent(cells),Order.times_one(GridRanges.last_extent(cells)))},
    counted,relane_placed(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,model,workers,
      Sources.source_placed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},model,Tiled.cell_shape(values,source,source_position),fill,workers,contract),facts))

# Source cells of the contract relaned onto an axis other than their lane
# axis hold the same values with the lanes on that axis.
def relane_within(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+old: Nat,+position: Grid.Position,+axis: Nat,+target_position: Grid.Position,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,old,position),fill,Tensors.Usable{},workers}),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,
  +facts: Gather.RelaneFacts(limits,values,old,position,Nat.is_lt(old,List.length(&2,U32,values)),axis,target_position,Nat.is_lt(axis,List.length(&2,U32,values)))) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tiled.relane_status(Tensors.Usable{},Tiled.Relane{old,Nat.is_lt(old,List.length(&2,U32,values)),position,axis,Nat.is_lt(axis,List.length(&2,U32,values)),target_position,Grid.lanes_at(values,axis),Tiled.cell_shape(values,axis,target_position)},values,Tiled.cell_shape(values,old,position),workers,PiecesModel.pack(~Tile.Tile8,model)),axis,Grid.lanes_at(values,axis),target_position}):
  Held.Both{+within_value,Held.Both{+target_inside,Held.Both{+room,Held.Both{+lanes_room,Held.Both{+cells_room,+more}}}}} = facts
  KernelGemm.Both{relane_moved(~Logical,~original,logical,limits,values,old,position,Nat.is_lt(old,List.length(&2,U32,values)),axis,target_position,Nat.is_lt(axis,List.length(&2,U32,values)),model,fill,workers,global,facts,
      StorageCount.counted_within(limits,Tiled.cell_shape(values,axis,target_position),cells_room,default_capacity)),
    Public.Laid{inside,{==},Public.cells_restored(values,axis,target_position),counted,{==}}}

def relane_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,+position: Grid.Position,
  +axis: Nat,+target_position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+apart: {Nat.is_eq(axis,old) == False{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +source_inside: {Nat.is_le(old,List.length(&2,U32,values)) == True{} : Bool},+shaped: {Tiled.cell_shape(values,old,position) == shape : List<&2,U32>},
  +counted: Global.Counted<values>,+pinned: {Grid.lanes_at(values,old) == lanes : U32},
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},axis,target_position)):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  +moved_global = {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},_)
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers})}
  +cells_global = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,old,position),shape,shaped) : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers})
    moved_global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,old,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,old,position),model_fill,Tensors.Usable{},model_workers})}
  +facts = Facts.relane_accepted(limits,values,old,position,axis,target_position,StorageCount.counted_room(limits,values,counted,default_capacity),nonempty,rank_room,
    source_inside,inside,apart)
  %equation : Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{_,old,lanes,position},axis,target_position))
  %shaped : Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers},old,lanes,position},axis,target_position))
  %pinned : Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,old,position),model_fill,Tensors.Usable{},model_workers},old,_,position},axis,target_position))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(Tiled.cell_shape(values,old,position),old,Grid.lanes_at(values,old),position),values,Public.cells_restored(values,old,position)) : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tiled.relane_status(Tensors.Usable{},Tiled.Relane{old,Nat.is_lt(old,List.length(&2,U32,_)),position,axis,Nat.is_lt(axis,List.length(&2,U32,_)),target_position,Grid.lanes_at(_,axis),Tiled.cell_shape(_,axis,target_position)},_,Tiled.cell_shape(values,old,position),model_workers,PiecesModel.pack(~Tile.Tile8,model)),axis,Grid.lanes_at(_,axis),target_position})
  relane_within(~Logical,~original,logical,values,limits,default_capacity,old,position,axis,target_position,model,model_fill,model_workers,cells_global,inside,counted,facts)

# A relane of tiled values of the contract with at least one value and a rank
# within the capacity, onto an axis other than their lane axis and at most
# the rank, holds the same values with the lanes on that axis.
def tiled_relane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,+old: Nat,+lanes: U32,+position: Grid.Position,+axis: Nat,+target_position: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+apart: {Nat.is_eq(axis,old) == False{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,old,lanes,position})) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{cells,old,lanes,position},axis,target_position)):
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      KernelGemm.Both{+global,Public.Laid{+source_inside,+shaped,+restored,+counted,+pinned}} = contract
      relane_usable(~Logical,~original,logical,values,pieces,shape,fill,workers,old,lanes,position,axis,target_position,limits,default_capacity,inside,apart,nonempty,rank_room,global,source_inside,shaped,counted,pinned,
        Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}))
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Public.Contract(~Logical,~original,logical,values,Tiled.relane(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},axis,target_position)),contract)

def along_cases(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+values: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,+old: Nat,+lanes: U32,+position: Grid.Position,+axis: Nat,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classified: {Nat.is_eq(old,axis) == same : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,old,lanes,position})) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.along_when(same,Tiled.Tiled{cells,old,lanes,position},axis,position)):
  match same:
    case True{}: contract
    case False{}:
      tiled_relane(~Logical,~original,logical,values,cells,old,lanes,position,axis,position,limits,default_capacity,inside,
        Equal.trans(Bool,Nat.is_eq(axis,old),Nat.is_eq(old,axis),False{},Order.equal_symmetric(axis,old),classified),nonempty,rank_room,contract)

# Tiled values of the contract with their lanes on an axis at most the rank
# hold the same values.
def tiled_along(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+axis: Nat,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Public.Contract(~Logical,~original,logical,values,Tiled.along(tiled,axis)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      along_cases(~Logical,~original,Nat.is_eq(old,axis),logical,values,cells,old,lanes,position,axis,limits,default_capacity,inside,nonempty,rank_room,{==},contract)

# What holds of every tiled form of the contract with its lanes on an axis
# holds of the tiled values of the contract along that axis.
def along_held(~Logical: Data,~original: Logical -> U32 -> F32,-Held: @tiled: Tiled.Tiled -> Data,same: Bool,+logical: Logical,+values: List<&2,U32>,
  +model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,+position: Grid.Position,+axis: Nat,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classified: {Nat.is_eq(old,axis) == same : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},old,lanes,position}),
  body: @cells: Tensors.Tensor<Tile.Tile8> -> @+lanes: U32 -> @+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}) ->
    Held(Tiled.Tiled{cells,axis,lanes,position})) ->
  Held(Tiled.along_when(same,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},old,lanes,position},axis,position)):
  match same:
    case True{}:
      +equal = Order.equal_of_is_eq(old,axis,classified)
      %Equal.sym(Nat,old,axis,equal) : Held(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},_,lanes,position})
      body(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},lanes,%equal : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},_,lanes,position}) contract)
    case False{}:
      +shown = Tiled.logical_shape(shape,old,lanes,position)
      +rank = List.length(&2,U32,shown)
      body(Tiled.relane_status(Tensors.Usable{},Tiled.Relane{old,Nat.is_lt(old,rank),position,axis,Nat.is_lt(axis,rank),position,Grid.lanes_at(shown,axis),Tiled.cell_shape(shown,axis,position)},
          shown,shape,workers,PiecesModel.pack(~Tile.Tile8,model)),Grid.lanes_at(shown,axis),
        tiled_relane(~Logical,~original,logical,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},old,lanes,position,axis,position,limits,default_capacity,inside,
          Equal.trans(Bool,Nat.is_eq(axis,old),Nat.is_eq(old,axis),False{},Order.equal_symmetric(axis,old),classified),nonempty,rank_room,contract))

# The same for usable cells, written with the pieces of their model.
def along_usable(~Logical: Data,~original: Logical -> U32 -> F32,-Held: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,
  -pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,+position: Grid.Position,+axis: Nat,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}),
  body: @cells: Tensors.Tensor<Tile.Tile8> -> @+lanes: U32 -> @+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}) ->
    Held(Tiled.Tiled{cells,axis,lanes,position})) ->
  Held(Tiled.along_when(Nat.is_eq(old,axis),Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},axis,position)):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  %equation : Held(Tiled.along_when(Nat.is_eq(old,axis),Tiled.Tiled{_,old,lanes,position},axis,position))
  along_held(~Logical,~original,Held,Nat.is_eq(old,axis),logical,values,model,shape,model_fill,model_workers,old,lanes,position,axis,limits,default_capacity,inside,nonempty,rank_room,{==},
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
      Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{_,old,lanes,position}) contract,body)

# What holds of every tiled form of the contract with its lanes on an axis
# holds of the tiled values of the contract moved to that axis.
def along_then(~Logical: Data,~original: Logical -> U32 -> F32,-Held: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+axis: Nat,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+contract: Public.Contract(~Logical,~original,logical,values,tiled),
  body: @cells: Tensors.Tensor<Tile.Tile8> -> @+lanes: U32 -> @+position: Grid.Position -> @+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}) ->
    Held(Tiled.Tiled{cells,axis,lanes,position})) -> Held(Tiled.along(tiled,axis)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          along_usable(~Logical,~original,Held,logical,values,pieces,shape,fill,workers,old,lanes,position,axis,limits,default_capacity,inside,nonempty,rank_room,
            Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),contract,cells => lanes => contract => body(cells,lanes,position,contract))
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Held(Tiled.along(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},axis)),contract)

# ---- The cells of an arrangement ----

# Usable cells of the contract are shaped and laid out at the lanes of the
# values.
def constructed_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position}):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  KernelGemm.Both{
    KernelGemm.Both{{%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : Global.Usable<Tile.Tile8,_,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}> usable :
      Global.Usable<Tile.Tile8,Tiled.cell_shape(values,axis,position),Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>},global},
    Public.Laid{inside,{==},Public.cells_restored(values,axis,position),counted,{==}}}

# The cells of packed tiled values of the contract with their groups put at a
# position, shaped and laid out at that position.
def positioned_packed(~Logical: Data,~original: Logical -> U32 -> F32,choice: Bool,+logical: Logical,+values: List<&2,U32>,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+own: Grid.Position,+position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +classified: {Grid.same_position(own,position) || Bool.not(Nat.is_lt(axis,List.length(&2,U32,shape))) == choice : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,own})) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.repositioned_when(choice,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,own},position)),
    axis,Grid.lanes_at(values,axis),position}):
  match choice:
    case True{}:
      +ranks = Repositioned.cells_rank(~Logical,~original,logical,values,PiecesModel.pack(~Tile.Tile8,model),shape,fill,workers,axis,lanes,own,contract)
      constructed_shaped(~Logical,~original,logical,values,PiecesModel.pack(~Tile.Tile8,model),shape,fill,workers,axis,lanes,position,
        Repositioned.position_free(~Logical,~original,logical,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,own,position,limits,default_capacity,
          {%ranks : {Grid.same_position(own,position) || Bool.not(_) == True{} : Bool} classified :
            {Grid.same_position(own,position) || Bool.not(Nat.is_lt(axis,List.length(&2,U32,values))) == True{} : Bool}},contract),
        Global.usable(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},{==}))
    case False{}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
      +same_or = Grid.same_position(own,position)
      +outside = Bool.not(Nat.is_lt(axis,List.length(&2,U32,shape)))
      %Equal.sym(U32,Grid.lanes_at(values,axis),lanes,pinned) :
        Public.ShapedLaid(~Logical,~original,logical,values,
          Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},Tiled.repositioned_view(position,shape),0,[]),axis,_,position})
      Repositioned.repositioned_usable(~Logical,~original,logical,values,PiecesModel.pack(~Tile.Tile8,model),shape,fill,workers,axis,lanes,own,position,limits,default_capacity,rank_room,nonempty,
        Boolean.or_false_left(same_or,outside,classified),Repositioned.not_false(Nat.is_lt(axis,List.length(&2,U32,shape)),Boolean.or_false_right(same_or,outside,classified)),
        global,inside,shaped,counted,pinned,Global.usable(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},{==}))

def arranged_packed(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+values: List<&2,U32>,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +own: Nat,+lanes: U32,+own_position: Grid.Position,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classified: {Nat.is_eq(own,axis) == same : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},own,lanes,own_position})) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(same,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),
    axis,Grid.lanes_at(values,axis),position}):
  match same:
    case True{}:
      %Order.equal_of_is_eq(own,axis,classified) :
        Public.ShapedLaid(~Logical,~original,logical,values,
          Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(True{},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),_,Grid.lanes_at(values,_),position})
      positioned_packed(~Logical,~original,Grid.same_position(own_position,position) || Bool.not(Nat.is_lt(own,List.length(&2,U32,shape))),logical,values,model,shape,fill,workers,own,lanes,own_position,position,
        limits,default_capacity,rank_room,nonempty,{==},contract)
    case False{}:
      +logical_values = Tiled.logical_shape(shape,own,lanes,own_position)
      +rank = List.length(&2,U32,logical_values)
      +walk = {Tiled.Relane{own,Nat.is_lt(own,rank),own_position,axis,Nat.is_lt(axis,rank),position,Grid.lanes_at(logical_values,axis),Tiled.cell_shape(logical_values,axis,position)} : Tiled.Relane}
      constructed_shaped(~Logical,~original,logical,values,Tiled.relane_cells(walk,logical_values,shape,workers,PiecesModel.pack(~Tile.Tile8,model)),Tiled.walk_cells(walk),Tiled.zero_tile(),workers,
        axis,Grid.lanes_at(logical_values,axis),position,
        tiled_relane(~Logical,~original,logical,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},own,lanes,own_position,axis,position,limits,default_capacity,inside,
          Equal.trans(Bool,Nat.is_eq(axis,own),Nat.is_eq(own,axis),False{},Order.equal_symmetric(axis,own),classified),nonempty,rank_room,contract),
        Global.usable(~Tile.Tile8,Tensors.Tensor{Tiled.relane_cells(walk,logical_values,shape,workers,PiecesModel.pack(~Tile.Tile8,model)),Tiled.walk_cells(walk),Tiled.zero_tile(),Tensors.Usable{},workers},{==}))

# The same for usable tiled values, through the model of their pieces.
def arranged_model(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +own: Nat,+lanes: U32,+own_position: Grid.Position,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classified: {Nat.is_eq(own,axis) == same : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position}),
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(same,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),
    axis,Grid.lanes_at(values,axis),position}):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  %equation : Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(same,Tiled.Tiled{_,own,lanes,own_position},axis,position)),axis,Grid.lanes_at(values,axis),position})
  arranged_packed(~Logical,~original,same,logical,values,model,shape,model_fill,model_workers,own,lanes,own_position,axis,position,limits,default_capacity,inside,nonempty,rank_room,classified,
    {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{_,own,lanes,own_position})
      contract : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},own,lanes,own_position})})

# The cells of tiled values of the contract arranged with their lanes on an
# axis at most the rank and their groups at a position are shaped and laid
# out with that lane axis, its lanes and that position.
def arranged_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(tiled,axis,position)),axis,Grid.lanes_at(values,axis),position}):
  match tiled:
    case Tiled.Tiled{cells,+own,+lanes,+own_position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          arranged_model(~Logical,~original,Nat.is_eq(own,axis),logical,values,pieces,shape,fill,workers,own,lanes,own_position,axis,position,limits,default_capacity,inside,nonempty,rank_room,{==},contract,
            Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}))
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.ShapedLaid(~Logical,~original,logical,values,
            Tiled.Tiled{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},own,lanes,own_position},axis,position)),axis,Grid.lanes_at(values,axis),position}),contract)

# ---- The arranged tiled values ----

# Groups put at a position keep the lane axis and its lanes.
def repositioned_eta(choice: Bool,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+own: Grid.Position,
  +position: Grid.Position) ->
  {Tiled.repositioned_when(choice,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,own},position) ==
    Tiled.Tiled{Tiled.cells_of(Tiled.repositioned_when(choice,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,own},position)),axis,lanes,position} :
    Tiled.Tiled}:
  match choice:
    case True{}: {==}
    case False{}: {==}

# Tiled values of the contract arranged on an axis have their lanes on that
# axis, as many as the extent of the values there, and their groups at the
# position asked.
def arranged_eta(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,
  +workers: U32,+own: Nat,+lanes: U32,+own_position: Grid.Position,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_eq(own,axis) == same : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position})) ->
  {Tiled.arranged_when(same,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position) ==
    Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(same,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),axis,Grid.lanes_at(values,axis),position} : Tiled.Tiled}:
  match same:
    case True{}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
      %Order.equal_of_is_eq(own,axis,classified) :
        {Tiled.arranged_when(True{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},_,position) ==
          Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(True{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},_,position)),_,Grid.lanes_at(values,_),position} : Tiled.Tiled}
      %Equal.sym(U32,Grid.lanes_at(values,own),lanes,pinned) :
        {Tiled.arranged_when(True{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},own,position) ==
          Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(True{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},own,position)),own,_,position} : Tiled.Tiled}
      repositioned_eta(Grid.same_position(own_position,position) || Bool.not(Nat.is_lt(own,List.length(&2,U32,shape))),pieces,shape,fill,workers,own,lanes,own_position,position)
    case False{}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
      %restored :
        {Tiled.arranged_when(False{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position) ==
          Tiled.Tiled{Tiled.cells_of(Tiled.arranged_when(False{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),axis,Grid.lanes_at(_,axis),position} : Tiled.Tiled}
      {==}

# What tiled values arranged with their lanes on an axis and their groups at
# a position hold: that layout, and the values of the contract.
def Arranged(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tiled: Tiled.Tiled,+axis: Nat,+position: Grid.Position) -> Data:
  KernelGemm.Both<{Tiled.arranged(tiled,axis,position) ==
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(tiled,axis,position)),axis,Grid.lanes_at(values,axis),position} : Tiled.Tiled},
    Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(tiled,axis,position)),axis,Grid.lanes_at(values,axis),position})>

# Tiled values of the contract arranged with their lanes on an axis at most
# the rank and their groups at a position hold the same values in that layout.
def arranged_held(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Arranged(~Logical,~original,logical,values,tiled,axis,position):
  match tiled:
    case Tiled.Tiled{cells,+own,+lanes,+own_position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{arranged_eta(~Logical,~original,Nat.is_eq(own,axis),logical,values,pieces,shape,fill,workers,own,lanes,own_position,axis,position,{==},contract),
            Public.laid_contract(~Logical,~original,logical,values,Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position)),axis,Grid.lanes_at(values,axis),position,
              arranged_cells(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},own,lanes,own_position},axis,position,limits,default_capacity,inside,nonempty,rank_room,contract))}
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Arranged(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},own,lanes,own_position},axis,position),contract)

# What holds of the packed cells of every model laid out at the lanes of the
# values holds of the cells of the contract.
def modeled_usable(~Logical: Data,~original: Logical -> U32 -> F32,-Held: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,
  -pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +observation: PiecesModel.Observation<Tile.Tile8,pieces>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
  body: @+model: PiecesModel.Model<Tile.Tile8> -> @+fill: Tile.Tile8 -> @+workers: U32 ->
    @+contract: Public.Contract(~Logical,~original,logical,values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position}) ->
    Held(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})) ->
  Held(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}):
  PiecesModel.Observation{+model,+equation} = observation
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  %equation : Held(Tiled.Tiled{Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers},axis,lanes,position})
  %shaped : Held(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,fill,Tensors.Usable{},workers},axis,lanes,position})
  %pinned : Held(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,_,position})
  body(model,fill,workers,
    {%Equal.sym(U32,Grid.lanes_at(values,axis),lanes,pinned) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,_,position})
      %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,fill,Tensors.Usable{},workers},axis,lanes,position})
      %Equal.sym(Pieces.Pieces(~Tile.Tile8),PiecesModel.pack(~Tile.Tile8,model),pieces,equation) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers},axis,lanes,position}) contract :
      Public.Contract(~Logical,~original,logical,values,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})})

def modeled(~Logical: Data,~original: Logical -> U32 -> F32,-Held: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,
  +axis: Nat,+lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}),
  body: @+model: PiecesModel.Model<Tile.Tile8> -> @+fill: Tile.Tile8 -> @+workers: U32 ->
    @+contract: Public.Contract(~Logical,~original,logical,values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position}) ->
    Held(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})) ->
  Held(Tiled.Tiled{cells,axis,lanes,position}):
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      modeled_usable(~Logical,~original,Held,logical,values,pieces,shape,fill,workers,axis,lanes,position,PiecesModel.observe(~Tile.Tile8,pieces),contract,body)
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Held(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)
