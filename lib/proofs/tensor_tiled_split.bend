# The cells of a split of the lane axis: the target cells, gathered cell by
# cell or run by run from the source pieces each task holds, hold the lifted
# values of the reader with the lanes on the second axis of the split.

import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_moved_sources.bend as Sources
import ./tensor_global_shape_count.bend as Global
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_reads.bend as ViewReads
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_split_value.bend as SplitValue
import ./tensor_split_gather.bend as Gather
import ./tensor_split_laws.bend as Laws
import ./tensor_split_reads.bend as Reads
import ./tensor_split_facts.bend as Facts
import ./tensor_split_law.bend as Law
import ./nat_order.bend as Order
import ./boolean_logic.bend as Logic
import ../tensor_tiled_cells.bend as Cells
import ./tensor_storage_count.bend as StorageCount
import ./nat_to_u32_bounds.bend as Bounds
import ../kernel_gemm.bend as KernelGemm
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits
import ./word_multiplication.bend as Multiplication
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_relane_piece.bend as RelanePiece
import ../tensor_tiled_layout.bend as TiledLayout
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison

# ---- The split cells ----

def split_placed_with(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,+model: PiecesModel.Model<Tile.Tile8>,+workers: U32,
  +placed: Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),PiecesModel.pack(~Tile.Tile8,model)>,
  +facts: Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after),
  +counted: Held.Both<Components.ComponentsOk([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))],Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]))) == True{} : Bool}>,
  +row_ok: Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))),Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),workers)>:
  Placed.PlacedModel{+source_model,+exact,+laid,+holds} = placed
  Held.Both{+facts_split,Held.Both{+pre_size,Held.Both{+post_size,Held.Both{+front_size,Held.Both{+back_size,Held.Both{+room,Held.Both{+nonempty,Held.Both{+rank_room,
    Held.Both{+target_room,Held.Both{+target_nonempty,Held.Both{+target_rank,Held.Both{+blocks,+parts}}}}}}}}}}}} = facts
  Held.Both{+components_ok,+lists_positive} = counted
  +row = Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))
  +separate = Tasks.separable([row])
  +source_ok = {Held.Both{room,Held.Both{nonempty,rank_room}} : Components.ShapeOk(limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))}
  +target_ok = {Held.Both{target_room,Held.Both{target_nonempty,target_rank}} : Components.ShapeOk(limits,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))}
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,source_model),exact) :
    Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))),
      Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([row]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{_,Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),
        Tiled.zero_tile(),workers)>
  Components.moved_with_model(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,~Gather.split_model,~Placed.SourceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)}),~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)}),
    ~Law.SplitAgrees(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after),~Law.split_law(~Logical,~original,logical,limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after),
    separate,limits,[row],Tasks.target_rank([row]),[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},
    Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},facts}}}}},
    3.0,16.0,workers,Moved.Models{source_model,Moved.NoModels{}},components_ok,{==},
    Held.Both{Held.Both{{==},Held.Both{{==},source_ok}},Held.Both{Sources.origin_when(separate,limits,row,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Logic.when_held(Components.RowOk(limits,row,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),separate,row_ok)),Unit{}}},
    lists_positive,target_ok,Facts.parts_limits(row,limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),parts),
    Held.Both{laid,Held.Both{Components.shape_ordered(limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),source_ok),Held.Both{Sources.empty_holds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)},Tiled.zero_tile()),Unit{}}}},
    Held.Both{Sources.good_of(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)},source_model,Held.Ended{},holds),Unit{}})

# The pieces source cells laid as canonical pieces split into hold the target
# values of the split.
def split_placed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,+model: PiecesModel.Model<Tile.Tile8>,+workers: U32,
  +placed: Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),PiecesModel.pack(~Tile.Tile8,model)>,
  +facts: Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after)) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)}),Pieces.ranges_of_grid(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))),Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),workers)>:
  Held.Both{+facts_split,Held.Both{+pre_size,Held.Both{+post_size,Held.Both{+front_size,Held.Both{+back_size,Held.Both{+room,Held.Both{+nonempty,Held.Both{+rank_room,
    Held.Both{+target_room,Held.Both{+target_nonempty,Held.Both{+target_rank,Held.Both{+blocks,+parts}}}}}}}}}}}} = facts
  +row = Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))
  +reads = Facts.parts_reads(row,limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),parts,target_room,target_nonempty,target_rank)
  split_placed_with(~Logical,~original,logical,limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,model,workers,placed,facts,
    Facts.row_components_of(row,limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),reads,Facts.parts_cells(row,limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),parts),target_room,target_nonempty),
    Facts.row_ok(row,limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),reads,room,nonempty,blocks))

# Source cells of the contract split into target cells of their count hold
# the target values of the split.
def split_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,
  +workers: U32,+contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),fill,Tensors.Usable{},workers}),
  +facts: Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after),+counted: Global.Counted<Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)},Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),workers),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),Tensors.Usable{},workers}):
  Held.Both{+facts_split,Held.Both{+pre_size,Held.Both{+post_size,Held.Both{+front_size,Held.Both{+back_size,Held.Both{+room,Held.Both{+nonempty,Held.Both{+rank_room,
    Held.Both{+target_room,Held.Both{+target_nonempty,Held.Both{+target_rank,Held.Both{+blocks,+parts}}}}}}}}}}}} = facts
  +cells = Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))
  GridRanges.grid_contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.target_cells(target_outer,target_after,width,position)},limits,cells,Tiled.zero_tile(),workers,Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),workers),target_room,target_nonempty,
    1n,GridRanges.Multiple{GridRanges.last_extent(cells),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(cells),1n),GridRanges.last_extent(cells),Order.times_one(GridRanges.last_extent(cells)))},
    counted,split_placed(~Logical,~original,logical,limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,model,workers,Sources.source_placed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)},model,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),fill,workers,contract),facts))

# ---- The split of tiled values ----

# The axes of new extents with anchors and width after pre pair, around the
# split, with the source axes around lanes after front.
def links_exact(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>) ->
  {Tiled.split_links(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),List.append(&2,U32,pre,anchors <> width <> post),List.length(&2,U32,pre)) == Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))) : Maybe<&2,List<&2,Tiled.Link>>}:
  +front_length = List.length(&2,U32,front)
  +pre_length = List.length(&2,U32,pre)
  %Equal.sym(List<&2,U32>,List.take(&2,U32,List.append(&2,U32,pre,anchors <> width <> post),pre_length),pre,Reads.taken_front(pre,anchors <> width <> post)) :
    {Tiled.joined_links(Tiled.pairing(_,List.take(&2,U32,List.append(&2,U32,front,lanes <> back),front_length),0n),
      Tiled.pairing(List.drop(&2,U32,List.append(&2,U32,pre,anchors <> width <> post),Nat.add(pre_length,2n)),List.drop(&2,U32,List.append(&2,U32,front,lanes <> back),Nat.add(front_length,1n)),Nat.add(front_length,1n))) == Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))) : Maybe<&2,List<&2,Tiled.Link>>}
  %Equal.sym(List<&2,U32>,List.take(&2,U32,List.append(&2,U32,front,lanes <> back),front_length),front,Reads.taken_front(front,lanes <> back)) :
    {Tiled.joined_links(Tiled.pairing(pre,_,0n),
      Tiled.pairing(List.drop(&2,U32,List.append(&2,U32,pre,anchors <> width <> post),Nat.add(pre_length,2n)),List.drop(&2,U32,List.append(&2,U32,front,lanes <> back),Nat.add(front_length,1n)),Nat.add(front_length,1n))) == Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))) : Maybe<&2,List<&2,Tiled.Link>>}
  %Equal.trans(List<&2,U32>,post,List.drop(&2,U32,post,0n),List.drop(&2,U32,List.append(&2,U32,pre,anchors <> width <> post),Nat.add(pre_length,2n)),Equal.sym(List<&2,U32>,List.drop(&2,U32,post,0n),post,ShapeLaws.drop_zero(post)),
      Equal.sym(List<&2,U32>,List.drop(&2,U32,List.append(&2,U32,pre,anchors <> width <> post),Nat.add(pre_length,2n)),List.drop(&2,U32,anchors <> width <> post,2n),Reads.dropped_front(pre,anchors <> width <> post,2n))) :
    {Tiled.joined_links(Tiled.pairing(pre,front,0n),
      Tiled.pairing(_,List.drop(&2,U32,List.append(&2,U32,front,lanes <> back),Nat.add(front_length,1n)),Nat.add(front_length,1n))) == Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))) : Maybe<&2,List<&2,Tiled.Link>>}
  %Equal.trans(List<&2,U32>,back,List.drop(&2,U32,back,0n),List.drop(&2,U32,List.append(&2,U32,front,lanes <> back),Nat.add(front_length,1n)),Equal.sym(List<&2,U32>,List.drop(&2,U32,back,0n),back,ShapeLaws.drop_zero(back)),
      Equal.sym(List<&2,U32>,List.drop(&2,U32,List.append(&2,U32,front,lanes <> back),Nat.add(front_length,1n)),List.drop(&2,U32,lanes <> back,1n),Reads.dropped_front(front,lanes <> back,1n))) :
    {Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,_,Nat.add(front_length,1n))) == Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))) : Maybe<&2,List<&2,Tiled.Link>>}
  {==}

# A split of tiled values whose cells restore values with lanes after front.
def runtime_split(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,
  -pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+outer: U32,+after: U32,
  +restored: {Tiled.logical_shape(cells,List.length(&2,U32,front),lanes,position) == List.append(&2,U32,front,lanes <> back) : List<&2,U32>}) ->
  {Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},List.length(&2,U32,front),lanes,position},List.append(&2,U32,pre,anchors <> width <> post),List.length(&2,U32,pre),outer,anchors,width,
      after) ==
    Tiled.Tiled{Tiled.split_status(Tensors.Usable{},Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),position),cells,List.length(&2,U32,front),workers,pieces),Nat.add(List.length(&2,U32,pre),1n),width,position} :
    Tiled.Tiled}:
  +front_length = List.length(&2,U32,front)
  +pre_length = List.length(&2,U32,pre)
  Equal.cong(Maybe<&2,List<&2,Tiled.Link>>,Tiled.Tiled,
    links => Tiled.Tiled{Tiled.split_status(Tensors.Usable{},Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},links,List.append(&2,U32,pre,anchors <> width <> post),Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),position),cells,front_length,workers,pieces),Nat.add(List.length(&2,U32,pre),1n),width,position},
    Tiled.split_links(Tiled.logical_shape(cells,front_length,lanes,position),front_length,List.append(&2,U32,pre,anchors <> width <> post),pre_length),Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),
    Equal.trans(Maybe<&2,List<&2,Tiled.Link>>,Tiled.split_links(Tiled.logical_shape(cells,front_length,lanes,position),front_length,List.append(&2,U32,pre,anchors <> width <> post),pre_length),
      Tiled.split_links(List.append(&2,U32,front,lanes <> back),front_length,List.append(&2,U32,pre,anchors <> width <> post),pre_length),Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),
      Equal.cong(List<&2,U32>,Maybe<&2,List<&2,Tiled.Link>>,source => Tiled.split_links(source,front_length,List.append(&2,U32,pre,anchors <> width <> post),pre_length),Tiled.logical_shape(cells,front_length,lanes,position),List.append(&2,U32,front,lanes <> back),
        restored),
      links_exact(front,lanes,back,pre,anchors,width,post)))

# The geometry of values with lanes after front.
def lanes_geometry(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+position: Grid.Position,
  +inside: {Nat.is_le(List.length(&2,U32,front),List.length(&2,U32,List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) ->
  {Public.geometry_of(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position) == Laws.source_cells((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))),lanes,position) : Cells.Cells}:
  +axis = List.length(&2,U32,front)
  Equal.trans(Cells.Cells,Public.geometry_of(List.append(&2,U32,front,lanes <> back),axis,position),
    Cells.Cells{(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))),Grid.lanes_at(List.append(&2,U32,front,lanes <> back),axis),Tiled.groups(Grid.lanes_at(List.append(&2,U32,front,lanes <> back),axis)),position},Laws.source_cells((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))),lanes,position),
    Public.geometry_exact(List.append(&2,U32,front,lanes <> back),axis,position,inside),
    Equal.cong(U32,Cells.Cells,extent => Cells.Cells{(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))),extent,Tiled.groups(extent),position},Grid.lanes_at(List.append(&2,U32,front,lanes <> back),axis),lanes,Reads.lanes_front(front,lanes,back)))

# The geometry of extents with anchors and width after pre, the lanes on width.
def width_geometry(+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position) ->
  {Public.geometry_of(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),position) == Laws.target_cells((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),width,position) : Cells.Cells}:
  +axis = Nat.add(List.length(&2,U32,pre),1n)
  Equal.trans(Cells.Cells,Public.geometry_of(List.append(&2,U32,pre,anchors <> width <> post),axis,position),
    Cells.Cells{(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),Grid.lanes_at(List.append(&2,U32,pre,anchors <> width <> post),axis),Tiled.groups(Grid.lanes_at(List.append(&2,U32,pre,anchors <> width <> post),axis)),position},Laws.target_cells((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),width,position),
    Public.geometry_exact(List.append(&2,U32,pre,anchors <> width <> post),axis,position,Bounds.strict_le(axis,List.length(&2,U32,List.append(&2,U32,pre,anchors <> width <> post)),Reads.inside_next(pre,anchors,width,post))),
    Equal.cong(U32,Cells.Cells,extent => Cells.Cells{(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),extent,Tiled.groups(extent),position},Grid.lanes_at(List.append(&2,U32,pre,anchors <> width <> post),axis),width,Reads.lanes_next(pre,anchors,width,post)))

# Tiled values of the contract with lanes after front, split into anchors and
# width after pre with the facts a split needs, hold the same values with the
# lanes on width.
def split_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,
  +pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,-pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,+fill: Tile.Tile8,
  +workers: U32,+outer: U32,+after: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position)},
    Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers}),
  +laid: Public.Laid<cells,List.length(&2,U32,front),lanes,position,List.append(&2,U32,front,lanes <> back)>,+usable: Global.Usable<Tile.Tile8,cells,Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers}>,
  +shape_counted: Global.Counted<List.append(&2,U32,pre,anchors <> width <> post)>,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +facts: Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))))) ->
  Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),
    Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},List.length(&2,U32,front),lanes,position},List.append(&2,U32,pre,anchors <> width <> post),List.length(&2,U32,pre),outer,anchors,width,
      after)):
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  Held.Both{+facts_split,Held.Both{+pre_size,Held.Both{+post_size,Held.Both{+front_size,Held.Both{+back_size,Held.Both{+room,Held.Both{+nonempty,Held.Both{+rank_room,
    Held.Both{+target_room,Held.Both{+target_nonempty,Held.Both{+target_rank,Held.Both{+blocks,+parts}}}}}}}}}}}} = facts
  +source_outer = (Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)
  +source_after = Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))
  +target_outer = (Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32)
  +target_after = Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n)))
  +front_length = List.length(&2,U32,front)
  +lane = Nat.add(List.length(&2,U32,pre),1n)
  +source_exact = Equal.trans(List<&2,U32>,cells,Tiled.cell_shape(List.append(&2,U32,front,lanes <> back),front_length,position),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Equal.sym(List<&2,U32>,Tiled.cell_shape(List.append(&2,U32,front,lanes <> back),front_length,position),cells,shaped),
    Reads.source_cells_of(front,lanes,back,position))
  +source_global = {%lanes_geometry(front,lanes,back,position,inside) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),model_fill,Tensors.Usable{},model_workers})
    %source_exact :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(List.append(&2,U32,front,lanes <> back),front_length,position)},
        Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers})
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},
        Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(List.append(&2,U32,front,lanes <> back),front_length,position)},_)
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Laws.source_cells(source_outer,source_after,lanes,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),model_fill,Tensors.Usable{},model_workers})}
  +target_global = {%Equal.sym(Cells.Cells,Public.geometry_of(List.append(&2,U32,pre,anchors <> width <> post),lane,position),Laws.target_cells((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),width,position),width_geometry(pre,anchors,width,post,position)) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),model_workers),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),Tensors.Usable{},model_workers})
    split_moved(~Logical,~original,logical,limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,model,model_fill,model_workers,source_global,facts,StorageCount.counted_within(limits,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),target_room,default_capacity)) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(List.append(&2,U32,pre,anchors <> width <> post),lane,position)},Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tiled.Split,~Tiled.split_gather,Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),Reads.groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},3.0,16.0,Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0)],Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),model_workers),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.zero_tile(),Tensors.Usable{},model_workers})}
  +target_restored = {%Reads.target_cells_of(pre,anchors,width,post,position) : {Tiled.logical_shape(_,lane,width,position) == List.append(&2,U32,pre,anchors <> width <> post) : List<&2,U32>}
    %Reads.lanes_next(pre,anchors,width,post) : {Tiled.logical_shape(Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),lane,position),lane,_,position) == List.append(&2,U32,pre,anchors <> width <> post) : List<&2,U32>}
    Public.cells_restored(List.append(&2,U32,pre,anchors <> width <> post),lane,position) : {Tiled.logical_shape(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),lane,width,position) == List.append(&2,U32,pre,anchors <> width <> post) : List<&2,U32>}}
  %equation :
    Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),Tiled.split_lanes(Tiled.Tiled{_,front_length,lanes,position},List.append(&2,U32,pre,anchors <> width <> post),List.length(&2,U32,pre),outer,anchors,width,after))
  %Equal.sym(Tiled.Tiled,
      Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},front_length,lanes,position},List.append(&2,U32,pre,anchors <> width <> post),
        List.length(&2,U32,pre),outer,anchors,width,after),
      Tiled.Tiled{Tiled.split_status(Tensors.Usable{},Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),lane,position),cells,front_length,model_workers,PiecesModel.pack(~Tile.Tile8,model)),lane,
        width,position},
      runtime_split(front,lanes,back,pre,anchors,width,post,position,PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,model_workers,outer,after,restored)) :
    Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),_)
  %Equal.sym(List<&2,U32>,cells,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),source_exact) :
    Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),
      Tiled.Tiled{Tiled.split_status(Tensors.Usable{},Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),lane,position),_,front_length,model_workers,PiecesModel.pack(~Tile.Tile8,model)),lane,
        width,position})
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(List.append(&2,U32,pre,anchors <> width <> post),lane,position),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Reads.target_cells_of(pre,anchors,width,post,position)) :
    Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),
      Tiled.Tiled{Tiled.split_status(Tensors.Usable{},Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),_,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),front_length,model_workers,PiecesModel.pack(~Tile.Tile8,model)),lane,width,position})
  KernelGemm.Both{target_global,
    Public.Laid{Bounds.strict_le(lane,List.length(&2,U32,List.append(&2,U32,pre,anchors <> width <> post)),Reads.inside_next(pre,anchors,width,post)),Reads.target_cells_of(pre,anchors,width,post,position),target_restored,
      shape_counted,Reads.lanes_next(pre,anchors,width,post)}}

# ---- The facts of a split from its shapes ----

def right_positive(+x: Nat,+y: Nat,+positive: {Nat.is_lt(0n,Nat.mul(x,y)) == True{} : Bool}) -> {Nat.is_lt(0n,y) == True{} : Bool}:
  RowMajor.left_factor_positive(y,x,{%RowMajor.multiply_commutative(x,y) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(y,x)) == True{} : Bool}})

# A factor of a positive product within a bound lies within it.
def left_room(+x: Nat,+y: Nat,+cap: Nat,+room: {Nat.is_le(Nat.mul(x,y),cap) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(x,y)) == True{} : Bool}) ->
  {Nat.is_le(x,cap) == True{} : Bool}:
  Order.transitive(x,Nat.mul(x,y),cap,RowMajor.product_covers(x,y,right_positive(x,y,positive)),room)

def inner_le(+x: Nat,+y: Nat,+positive: {Nat.is_lt(0n,x) == True{} : Bool}) -> {Nat.is_le(y,Nat.mul(x,y)) == True{} : Bool}:
  {%RowMajor.multiply_commutative(y,x) : {Nat.is_le(y,_) == True{} : Bool} RowMajor.product_covers(y,x,positive) : {Nat.is_le(y,Nat.mul(x,y)) == True{} : Bool}}

def right_room(+x: Nat,+y: Nat,+cap: Nat,+room: {Nat.is_le(Nat.mul(x,y),cap) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(x,y)) == True{} : Bool}) ->
  {Nat.is_le(y,cap) == True{} : Bool}:
  Order.transitive(y,Nat.mul(x,y),cap,inner_le(x,y,RowMajor.left_factor_positive(x,y,positive)),room)

# The extents of a shape within the capacity multiply to its cells.
def outer_of(+limits: Machine.Limits,+values: List<&2,U32>,+outer: U32,+exact: {Grid.product(values) == outer : U32},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(outer) == ShapeLaws.elements(values) : Nat}:
  %exact : {U32.to_nat(_) == ShapeLaws.elements(values) : Nat}
  Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),values,room)

# The axes after the lane axis of values with lanes after front.
def after_front(front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>) -> {Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)) == back : List<&2,U32>}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}: after_front(rest,lanes,back)

def after_next(pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>) -> {Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n)) == post : List<&2,U32>}:
  match pre:
    case Nil{}: {==}
    case Con{+first,+rest}: after_next(rest,anchors,width,post)

# The source rows are the cells of the other source axes.
def source_product(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>) -> {(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32) == Grid.product(List.append(&2,U32,front,back)) : U32}:
  +axis = List.length(&2,U32,front)
  +rest = List.append(&2,U32,front,back)
  Equal.trans(U32,(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),(1 * Grid.product(Grid.removed(List.append(&2,U32,front,lanes <> back),axis)) : U32),Grid.product(rest),Public.prefix_after(List.append(&2,U32,front,lanes <> back),axis,1),
    {%Equal.sym(List<&2,U32>,Grid.removed(List.append(&2,U32,front,lanes <> back),axis),rest,Reads.removed_front(front,lanes,back)) : {(1 * Grid.product(_) : U32) == Grid.product(rest) : U32}
      Public.one_times(Grid.product(rest)) : {(1 * Grid.product(Grid.removed(List.append(&2,U32,front,lanes <> back),axis)) : U32) == Grid.product(rest) : U32}})

def target_product(+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>) -> {(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32) == Grid.product(List.append(&2,U32,pre,anchors <> post)) : U32}:
  +axis = Nat.add(List.length(&2,U32,pre),1n)
  Equal.trans(U32,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),(1 * Grid.product(Grid.removed(List.append(&2,U32,pre,anchors <> width <> post),axis)) : U32),Grid.product(List.append(&2,U32,pre,anchors <> post)),Public.prefix_after(List.append(&2,U32,pre,anchors <> width <> post),axis,1),
    {%Equal.sym(List<&2,U32>,Grid.removed(List.append(&2,U32,pre,anchors <> width <> post),axis),List.append(&2,U32,pre,anchors <> post),Reads.removed_next(pre,anchors,width,post)) : {(1 * Grid.product(_) : U32) == Grid.product(List.append(&2,U32,pre,anchors <> post)) : U32}
      Public.one_times(Grid.product(List.append(&2,U32,pre,anchors <> post))) : {(1 * Grid.product(Grid.removed(List.append(&2,U32,pre,anchors <> width <> post),axis)) : U32) == Grid.product(List.append(&2,U32,pre,anchors <> post)) : U32}})

# The cells of values with lanes after front: the lanes times the other axes.
def values_split(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>) ->
  {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back))) : Nat}:
  +axis = List.length(&2,U32,front)
  {%Reads.lanes_front(front,lanes,back) : {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == Nat.mul(U32.to_nat(_),ShapeLaws.elements(List.append(&2,U32,front,back))) : Nat}
    %Reads.removed_front(front,lanes,back) : {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == Nat.mul(U32.to_nat(Grid.lanes_at(List.append(&2,U32,front,lanes <> back),axis)),ShapeLaws.elements(_)) : Nat}
    Public.elements_split(List.append(&2,U32,front,lanes <> back),axis,Reads.inside_front(front,lanes,back)) :
    {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back))) : Nat}}

def shape_split(+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>) ->
  {ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) == Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))) : Nat}:
  +axis = Nat.add(List.length(&2,U32,pre),1n)
  {%Reads.lanes_next(pre,anchors,width,post) : {ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) == Nat.mul(U32.to_nat(_),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))) : Nat}
    %Reads.removed_next(pre,anchors,width,post) : {ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) == Nat.mul(U32.to_nat(Grid.lanes_at(List.append(&2,U32,pre,anchors <> width <> post),axis)),ShapeLaws.elements(_)) : Nat}
    Public.elements_split(List.append(&2,U32,pre,anchors <> width <> post),axis,Reads.inside_next(pre,anchors,width,post)) : {ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) == Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))) : Nat}}

# The parts of values within the capacity.
def values_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  {%values_split(front,lanes,back) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound :
    {Nat.is_le(Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def values_positive(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back)))) == True{} : Bool}:
  {%values_split(front,lanes,back) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back)))) == True{} : Bool}}

def rest_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  right_room(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(Machine.capacity(limits)),values_room(limits,front,lanes,back,bound),values_positive(front,lanes,back,positive))

def front_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(front),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +rest = List.append(&2,U32,front,back)
  +rest_positive = right_positive(U32.to_nat(lanes),ShapeLaws.elements(rest),values_positive(front,lanes,back,positive))
  left_room(ShapeLaws.elements(front),ShapeLaws.elements(back),U32.to_nat(Machine.capacity(limits)),
    {%ShapeLaws.elements_appended(front,back) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rest_room(limits,front,lanes,back,bound,positive) :
      {Nat.is_le(Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%ShapeLaws.elements_appended(front,back) : {Nat.is_lt(0n,_) == True{} : Bool} rest_positive : {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back))) == True{} : Bool}})

def back_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(back),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +rest = List.append(&2,U32,front,back)
  +rest_positive = right_positive(U32.to_nat(lanes),ShapeLaws.elements(rest),values_positive(front,lanes,back,positive))
  right_room(ShapeLaws.elements(front),ShapeLaws.elements(back),U32.to_nat(Machine.capacity(limits)),
    {%ShapeLaws.elements_appended(front,back) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rest_room(limits,front,lanes,back,bound,positive) :
      {Nat.is_le(Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%ShapeLaws.elements_appended(front,back) : {Nat.is_lt(0n,_) == True{} : Bool} rest_positive : {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back))) == True{} : Bool}})

def lanes_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  left_room(U32.to_nat(lanes),ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(Machine.capacity(limits)),values_room(limits,front,lanes,back,bound),values_positive(front,lanes,back,positive))

# The parts of new extents within the capacity.
def shape_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  {%shape_split(pre,anchors,width,post) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound : {Nat.is_le(Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def shape_positive(+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)))) == True{} : Bool}:
  {%shape_split(pre,anchors,width,post) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)))) == True{} : Bool}}

def others_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  right_room(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)),U32.to_nat(Machine.capacity(limits)),shape_room(limits,pre,anchors,width,post,bound),shape_positive(pre,anchors,width,post,positive))

def width_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(width),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  left_room(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)),U32.to_nat(Machine.capacity(limits)),shape_room(limits,pre,anchors,width,post,bound),shape_positive(pre,anchors,width,post,positive))

# The cells of the other target axes: pre, then anchors and post.
def others_split(+pre: List<&2,U32>,+anchors: U32,+post: List<&2,U32>) ->
  {ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)) == Nat.mul(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post))) : Nat}:
  ShapeLaws.elements_appended(pre,anchors <> post)

def others_positive(+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)))) == True{} : Bool}:
  {%others_split(pre,anchors,post) : {Nat.is_lt(0n,_) == True{} : Bool}
    right_positive(U32.to_nat(width),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)),shape_positive(pre,anchors,width,post,positive)) :
    {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)))) == True{} : Bool}}

def others_parts_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  {%others_split(pre,anchors,post) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} others_room(limits,pre,anchors,width,post,bound,positive) :
    {Nat.is_le(Nat.mul(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def pre_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(pre),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  left_room(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)),U32.to_nat(Machine.capacity(limits)),others_parts_room(limits,pre,anchors,width,post,bound,positive),
    others_positive(pre,anchors,width,post,positive))

def anchors_tail_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  right_room(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)),U32.to_nat(Machine.capacity(limits)),others_parts_room(limits,pre,anchors,width,post,bound,positive),
    others_positive(pre,anchors,width,post,positive))

def anchors_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  left_room(U32.to_nat(anchors),ShapeLaws.elements(post),U32.to_nat(Machine.capacity(limits)),anchors_tail_room(limits,pre,anchors,width,post,bound,positive),
    right_positive(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)),others_positive(pre,anchors,width,post,positive)))

def post_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(post),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  right_room(U32.to_nat(anchors),ShapeLaws.elements(post),U32.to_nat(Machine.capacity(limits)),anchors_tail_room(limits,pre,anchors,width,post,bound,positive),
    right_positive(ShapeLaws.elements(pre),Nat.mul(U32.to_nat(anchors),ShapeLaws.elements(post)),others_positive(pre,anchors,width,post,positive)))

# The anchors times the width lie within the cells of the new extents.
def lanes_product_room(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(anchors),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +a = U32.to_nat(anchors)
  +w = U32.to_nat(width)
  +others = ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post))
  +others_pos = right_positive(w,others,shape_positive(pre,anchors,width,post,positive))
  +anchors_pos = RowMajor.left_factor_positive(a,ShapeLaws.elements(post),
    right_positive(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post)),others_positive(pre,anchors,width,post,positive)))
  +anchors_le = {%Equal.sym(Nat,others,Nat.mul(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post))),others_split(pre,anchors,post)) : {Nat.is_le(a,_) == True{} : Bool}
    Order.transitive(a,Nat.mul(a,ShapeLaws.elements(post)),Nat.mul(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post))),
      RowMajor.product_covers(a,ShapeLaws.elements(post),right_positive(a,ShapeLaws.elements(post),
        right_positive(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post)),others_positive(pre,anchors,width,post,positive)))),
      inner_le(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post)),RowMajor.left_factor_positive(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post)),
        others_positive(pre,anchors,width,post,positive)))) : {Nat.is_le(a,others) == True{} : Bool}}
  Order.transitive(Nat.mul(a,w),Nat.mul(others,w),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(a,others,w,anchors_le),
    {%RowMajor.multiply_commutative(w,others) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} shape_room(limits,pre,anchors,width,post,bound) :
      {Nat.is_le(Nat.mul(others,w),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

# The source and target rows and lanes of a split, as numbers.
def source_rest_exact(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}) -> {U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)) == ShapeLaws.elements(List.append(&2,U32,front,back)) : Nat}:
  +rest = List.append(&2,U32,front,back)
  {%Equal.sym(U32,(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Grid.product(rest),source_product(front,lanes,back)) : {U32.to_nat(_) == ShapeLaws.elements(rest) : Nat}
    Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),rest,rest_room(limits,front,lanes,back,bound,positive)) :
    {U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)) == ShapeLaws.elements(rest) : Nat}}

def source_cells_exact(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+outer: U32,+after: U32,
  +bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool},
  +front_outer: {Grid.product(front) == outer : U32},+back_after: {Grid.product(back) == after : U32}) ->
  {U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)) == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat}:
  %Equal.sym(Nat,U32.to_nat(outer),ShapeLaws.elements(front),outer_of(limits,front,outer,front_outer,front_room(limits,front,lanes,back,bound,positive))) :
    {U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)) == Nat.mul(_,U32.to_nat(after)) : Nat}
  %Equal.sym(Nat,U32.to_nat(after),ShapeLaws.elements(back),outer_of(limits,back,after,back_after,back_room(limits,front,lanes,back,bound,positive))) :
    {U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)) == Nat.mul(ShapeLaws.elements(front),_) : Nat}
  Equal.trans(Nat,U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)),ShapeLaws.elements(List.append(&2,U32,front,back)),Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back)),
    source_rest_exact(limits,front,lanes,back,bound,positive),ShapeLaws.elements_appended(front,back))

def target_cells_exact(+limits: Machine.Limits,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+outer: U32,+after: U32,
  +bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool},
  +pre_outer: {Grid.product(pre) == outer : U32},+post_after: {Grid.product(post) == after : U32}) ->
  {U32.to_nat((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32)) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)) : Nat}:
  +o = U32.to_nat(outer)
  +a = U32.to_nat(anchors)
  +f = U32.to_nat(after)
  %Equal.sym(Nat,Nat.mul(Nat.mul(o,a),f),Nat.mul(o,Nat.mul(a,f)),RowMajor.assoc(o,a,f)) : {U32.to_nat((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32)) == _ : Nat}
  %Equal.sym(Nat,o,ShapeLaws.elements(pre),outer_of(limits,pre,outer,pre_outer,pre_room(limits,pre,anchors,width,post,bound,positive))) :
    {U32.to_nat((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32)) == Nat.mul(_,Nat.mul(a,f)) : Nat}
  %Equal.sym(Nat,f,ShapeLaws.elements(post),outer_of(limits,post,after,post_after,post_room(limits,pre,anchors,width,post,bound,positive))) :
    {U32.to_nat((Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32)) == Nat.mul(ShapeLaws.elements(pre),Nat.mul(a,_)) : Nat}
  %Equal.sym(U32,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Grid.product(List.append(&2,U32,pre,anchors <> post)),target_product(pre,anchors,width,post)) :
    {U32.to_nat(_) == Nat.mul(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post))) : Nat}
  Equal.trans(Nat,U32.to_nat(Grid.product(List.append(&2,U32,pre,anchors <> post))),ShapeLaws.elements(List.append(&2,U32,pre,anchors <> post)),Nat.mul(ShapeLaws.elements(pre),Nat.mul(a,ShapeLaws.elements(post))),
    Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),List.append(&2,U32,pre,anchors <> post),others_room(limits,pre,anchors,width,post,bound,positive)),others_split(pre,anchors,post))

# The groups of the source lanes times the source rows fit half a word.
def source_groups_room(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool},+lane_groups: TiledValues.Groups<limits,lanes>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +rest = ShapeLaws.elements(List.append(&2,U32,front,back))
  +g = U32.to_nat(Tiled.groups(lanes))
  {%Equal.sym(Nat,U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32)),rest,source_rest_exact(limits,front,lanes,back,bound,positive)) :
      {Nat.is_le(Nat.mul(g,_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Limits.within_half(limits,Nat.mul(g,rest),
      Order.transitive(Nat.mul(g,rest),Nat.mul(U32.to_nat(lanes),rest),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(g,U32.to_nat(lanes),rest,TiledValues.fewer_of(limits,lanes,lane_groups)),
        values_room(limits,front,lanes,back,bound))) :
    {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat((Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

# The values of a split of values with lanes after front into new extents with
# anchors and width after pre, of the same cells within the capacity.
def split_value_facts(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +outer: U32,+after: U32,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool},
  +shape_bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool},
  +front_outer: {Grid.product(front) == outer : U32},+pre_outer: {Grid.product(pre) == outer : U32},+back_after: {Grid.product(back) == after : U32},
  +post_after: {Grid.product(post) == after : U32},+split: {(anchors * width : U32) == lanes : U32}) ->
  SplitValue.Facts<limits,outer,anchors,width,after,lanes,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))>:
  +lanes_bound = lanes_room(limits,front,lanes,back,bound,positive)
  +width_bound = width_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty)
  +width_groups = TiledValues.grouped(limits,width,width_bound)
  +lane_groups = TiledValues.grouped(limits,lanes,lanes_bound)
  +after_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product(back),after,back_after)
  {SplitValue.Facts{target_cells_exact(limits,pre,anchors,width,post,outer,after,shape_bound,shape_nonempty,pre_outer,post_after),
      {%Equal.sym(List<&2,U32>,Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n)),post,after_next(pre,anchors,width,post)) : {U32.to_nat(Tiled.product(_)) == U32.to_nat(after) : Nat}
        Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product(post),after,post_after) : {U32.to_nat(Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n)))) == U32.to_nat(after) : Nat}},
      source_cells_exact(limits,front,lanes,back,outer,after,bound,positive,front_outer,back_after),
      {%Equal.sym(List<&2,U32>,Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)),back,after_front(front,lanes,back)) : {U32.to_nat(Tiled.product(_)) == U32.to_nat(after) : Nat}
        after_exact : {U32.to_nat(Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))) == U32.to_nat(after) : Nat}},
      {%split : {U32.to_nat(_) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat}
        Multiplication.bounded_multiply(anchors,width,Machine.capacity(limits),Limits.capacity_fits(limits),
          lanes_product_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty)) : {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat}},
      width_groups,lane_groups,
      {%Equal.sym(Nat,U32.to_nat(outer),ShapeLaws.elements(front),outer_of(limits,front,outer,front_outer,front_room(limits,front,lanes,back,bound,positive))) :
          {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        front_room(limits,front,lanes,back,bound,positive) : {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
      anchors_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty),
      {%Equal.sym(Nat,U32.to_nat(after),ShapeLaws.elements(back),outer_of(limits,back,after,back_after,back_room(limits,front,lanes,back,bound,positive))) :
          {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        back_room(limits,front,lanes,back,bound,positive) : {Nat.is_le(U32.to_nat(after),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
      Order.transitive(U32.to_nat(Tiled.groups(width)),U32.to_nat(width),U32.to_nat(Machine.capacity(limits)),TiledValues.fewer_of(limits,width,width_groups),width_bound),
      source_groups_room(limits,front,lanes,back,bound,positive,lane_groups)} :
    SplitValue.Facts<limits,outer,anchors,width,after,lanes,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))>}

# The shape facts of the cells of a lane axis.
def shape_ok_of(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cells: List<&2,U32>,
  +exact: {Tiled.cell_shape(values,axis,position) == cells : List<&2,U32>},
  +rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rooms: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}>>>) ->
  Components.ShapeOk(limits,cells):
  KernelGemm.Both{+removed_room,KernelGemm.Both{+lanes_bound,KernelGemm.Both{+cells_room,+cells_positive}}} = rooms
  %exact : Components.ShapeOk(limits,_)
  Held.Both{cells_room,Held.Both{cells_positive,rank}}

def split_facts_with(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+facts_split: SplitValue.Facts<limits,outer,anchors,width,after,lanes,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))>,+pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},+front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +source_ok: Components.ShapeOk(limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),+target_ok: Components.ShapeOk(limits,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) ->
  Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))):
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts_split
  Held.Both{+room,Held.Both{+nonempty,+rank_room}} = source_ok
  Held.Both{+target_room,Held.Both{+target_nonempty,+target_rank}} = target_ok
  +blocks = ViewFacts.grid_blocks(limits,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Grid.piece_target(),source_ok)
  +anchor = Facts.anchor_ok(position,limits,List.append(&2,U32,front,back),lanes,width,anchors,room,nonempty,rank_room,blocks,lanes_exact,lane_groups)
  +parts = Facts.split_parts(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),position,limits,List.append(&2,U32,pre,anchors <> width <> post),List.append(&2,U32,pre,anchors <> post),Grid.groups(width),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),List.length(&2,U32,front),width,Grid.groups(lanes),anchors,
    Reads.split_links_in(front,lanes,back,pre,anchors,width,post,position),anchor,room,nonempty,rank_room,blocks,target_rank)
  Held.Both{facts_split,Held.Both{pre_size,Held.Both{post_size,Held.Both{front_size,Held.Both{back_size,Held.Both{room,Held.Both{nonempty,Held.Both{rank_room,
    Held.Both{target_room,Held.Both{target_nonempty,Held.Both{target_rank,Held.Both{blocks,parts}}}}}}}}}}}}

# What a split of values with lanes after front into new extents with anchors
# and width after pre needs, from the cells and ranks within the capacity and
# the products the split matches.
def split_facts(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +same: {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) : Nat},+positive: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +front_outer: {Grid.product(front) == outer : U32},+pre_outer: {Grid.product(pre) == outer : U32},+back_after: {Grid.product(back) == after : U32},
  +post_after: {Grid.product(post) == after : U32},+split: {(anchors * width : U32) == lanes : U32}) ->
  Law.SplitFacts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,(Public.lane_prefix(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n),1) * Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,pre,anchors <> width <> post),Nat.add(List.length(&2,U32,pre),1n))),(Public.lane_prefix(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),1) * Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front))) : U32),Tiled.product(Public.lane_after(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front)))):
  +shape_bound = {%same : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound : {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +shape_nonempty = {%same : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}}
  +lane = Nat.add(List.length(&2,U32,pre),1n)
  +source_ok = shape_ok_of(limits,List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Reads.source_cells_of(front,lanes,back,position),
    RelanePiece.cells_rank_room(limits,List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position,True{},Reads.inside_front(front,lanes,back),rank_room),
    Coordinates.axis_cells(True{},limits,List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position,Reads.inside_front(front,lanes,back),bound,positive))
  +target_ok = shape_ok_of(limits,List.append(&2,U32,pre,anchors <> width <> post),lane,position,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Reads.target_cells_of(pre,anchors,width,post,position),
    RelanePiece.cells_rank_room(limits,List.append(&2,U32,pre,anchors <> width <> post),lane,position,True{},Reads.inside_next(pre,anchors,width,post),shape_rank),
    Coordinates.axis_cells(True{},limits,List.append(&2,U32,pre,anchors <> width <> post),lane,position,Reads.inside_next(pre,anchors,width,post),shape_bound,shape_nonempty))
  split_facts_with(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,
    split_value_facts(limits,front,lanes,back,pre,anchors,width,post,outer,after,bound,positive,shape_bound,shape_nonempty,front_outer,pre_outer,back_after,post_after,split),
    Equal.sym(Nat,U32.to_nat(outer),ShapeLaws.elements(pre),outer_of(limits,pre,outer,pre_outer,pre_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty))),
    Equal.sym(Nat,U32.to_nat(after),ShapeLaws.elements(post),outer_of(limits,post,after,post_after,post_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty))),
    Equal.sym(Nat,U32.to_nat(outer),ShapeLaws.elements(front),outer_of(limits,front,outer,front_outer,front_room(limits,front,lanes,back,bound,positive))),
    Equal.sym(Nat,U32.to_nat(after),ShapeLaws.elements(back),outer_of(limits,back,after,back_after,back_room(limits,front,lanes,back,bound,positive))),
    source_ok,target_ok)

# ---- The split a reshape finds ----

# The axes of a shape before an axis.
def front_of(shape: List<&2,U32>,axis: Nat) -> List<&2,U32>:
  match shape axis:
    case Nil{} axis: []
    case Con{extent,rest} 0n: []
    case Con{+extent,rest} 1n+next: extent <> front_of(rest,next)

# A shape is its axes before an axis inside it, that axis and those after.
def around_split(shape: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {List.append(&2,U32,front_of(shape,axis),Grid.lanes_at(shape,axis) <> Public.lane_after(shape,axis)) == shape : List<&2,U32>}:
  match shape axis:
    case Nil{} axis:
      Empty.absurd({List.append(&2,U32,front_of([],axis),Grid.lanes_at([],axis) <> Public.lane_after([],axis)) == [] : List<&2,U32>},Public.outside_empty(axis,inside))
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,List.append(&2,U32,front_of(rest,next),Grid.lanes_at(rest,next) <> Public.lane_after(rest,next)),rest,
        around_split(rest,next,inside))

def front_length(shape: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {List.length(&2,U32,front_of(shape,axis)) == axis : Nat}:
  match shape axis:
    case Nil{} axis: Empty.absurd({List.length(&2,U32,front_of([],axis)) == axis : Nat},Public.outside_empty(axis,inside))
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next: Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,front_of(rest,next)),next,front_length(rest,next,inside))

# The elements before an axis, from outer on, multiply the axes before it.
def prefix_front(shape: List<&2,U32>,axis: Nat,+outer: U32) -> {Public.lane_prefix(shape,axis,outer) == (outer * Grid.product(front_of(shape,axis)) : U32) : U32}:
  match shape axis:
    case Nil{} axis: Equal.sym(U32,(outer * 1 : U32),outer,Arithmetic.multiply_one(outer))
    case Con{extent,rest} 0n: Equal.sym(U32,(outer * 1 : U32),outer,Arithmetic.multiply_one(outer))
    case Con{+extent,+rest} 1n+ +next:
      Equal.trans(U32,Public.lane_prefix(rest,next,(outer * extent : U32)),((outer * extent : U32) * Grid.product(front_of(rest,next)) : U32),
        (outer * (extent * Grid.product(front_of(rest,next)) : U32) : U32),prefix_front(rest,next,(outer * extent : U32)),
        Arithmetic.multiply_associative(outer,extent,Grid.product(front_of(rest,next))))

def prefix_product(+shape: List<&2,U32>,+axis: Nat) -> {Grid.product(front_of(shape,axis)) == Public.lane_prefix(shape,axis,1) : U32}:
  Equal.trans(U32,Grid.product(front_of(shape,axis)),(1 * Grid.product(front_of(shape,axis)) : U32),Public.lane_prefix(shape,axis,1),
    Equal.sym(U32,(1 * Grid.product(front_of(shape,axis)) : U32),Grid.product(front_of(shape,axis)),Public.one_times(Grid.product(front_of(shape,axis)))),
    Equal.sym(U32,Public.lane_prefix(shape,axis,1),(1 * Grid.product(front_of(shape,axis)) : U32),prefix_front(shape,axis,1)))

# A split of values with lanes on own into new extents with anchors and width
# on axis: the new extents are pre, anchors, width and post, pre and the axes
# before the lanes hold outer, post and the axes after them after, and the
# anchors times the width are the lanes.
type SplitAt<-values: List<&2,U32>,-own: Nat,-shape: List<&2,U32>,-axis: Nat,-outer: U32,-anchors: U32,-width: U32,-after: U32> is Data:
  SplitAt{pre: List<&2,U32>,post: List<&2,U32>,shape_exact: {List.append(&2,U32,pre,anchors <> width <> post) == shape : List<&2,U32>},
    axis_exact: {List.length(&2,U32,pre) == axis : Nat},pre_outer: {Grid.product(pre) == outer : U32},front_outer: {Grid.product(front_of(values,own)) == outer : U32},
    post_after: {Grid.product(post) == after : U32},back_after: {Grid.product(Public.lane_after(values,own)) == after : U32},
    split: {(anchors * width : U32) == Grid.lanes_at(values,own) : U32},anchors_above: {U32.is_lt(1,anchors) == True{} : Bool},
    width_above: {U32.is_lt(1,width) == True{} : Bool}}

def ScanFound(found: Maybe<&2,TiledLayout.Splitting>,+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>) -> Data:
  match found:
    case Some{TiledLayout.Splitting{+axis,+outer,+anchors,+width,+after}}: SplitAt<values,own,shape,axis,outer,anchors,width,after>
    case None{}: Unit

def cond_found(chosen: Bool,+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>,+axis: Nat,+width: U32,+rest: List<&2,U32>,
  +inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+exact: {Public.lane_after(shape,axis) == width <> rest : List<&2,U32>},
  +chosen_exact: {(U32.is_eq(Public.lane_prefix(shape,axis,1),Public.lane_prefix(values,own,1)) && U32.is_lt(1,Grid.lanes_at(shape,axis)) && U32.is_lt(1,width) &&
    U32.is_eq((Grid.lanes_at(shape,axis) * width : U32),Grid.lanes_at(values,own)) && U32.is_eq(Tiled.product(rest),Tiled.product(Public.lane_after(values,own)))) == chosen : Bool}) ->
  ScanFound(Bool.pick(Maybe<&2,TiledLayout.Splitting>,chosen,
    Some{TiledLayout.Splitting{axis,Public.lane_prefix(values,own,1),Grid.lanes_at(shape,axis),width,Tiled.product(rest)}},None{}),values,own,shape):
  match chosen:
    case True{}:
      +anchors = Grid.lanes_at(shape,axis)
      +a = U32.is_eq(Public.lane_prefix(shape,axis,1),Public.lane_prefix(values,own,1))
      +b = U32.is_lt(1,anchors)
      +c = U32.is_lt(1,width)
      +d = U32.is_eq((anchors * width : U32),Grid.lanes_at(values,own))
      +e = U32.is_eq(Tiled.product(rest),Tiled.product(Public.lane_after(values,own)))
      +abcd = Logic.left(a && b && c && d,e,chosen_exact)
      +abc = Logic.left(a && b && c,d,abcd)
      +ab = Logic.left(a && b,c,abc)
      +before = Arithmetic.equal_of_is_eq(Public.lane_prefix(shape,axis,1),Public.lane_prefix(values,own,1),Logic.left(a,b,ab))
      +inner = Arithmetic.equal_of_is_eq(Tiled.product(rest),Tiled.product(Public.lane_after(values,own)),Logic.right(a && b && c && d,e,chosen_exact))
      SplitAt{front_of(shape,axis),rest,
        {%exact : {List.append(&2,U32,front_of(shape,axis),anchors <> _) == shape : List<&2,U32>} around_split(shape,axis,inside) :
          {List.append(&2,U32,front_of(shape,axis),anchors <> width <> rest) == shape : List<&2,U32>}},
        front_length(shape,axis,inside),
        Equal.trans(U32,Grid.product(front_of(shape,axis)),Public.lane_prefix(shape,axis,1),Public.lane_prefix(values,own,1),prefix_product(shape,axis),before),
        prefix_product(values,own),{==},
        Equal.sym(U32,Tiled.product(rest),Tiled.product(Public.lane_after(values,own)),inner),
        Arithmetic.equal_of_is_eq((anchors * width : U32),Grid.lanes_at(values,own),Logic.right(a && b && c,d,abcd)),Logic.right(a,b,ab),Logic.right(a && b,c,abc)}
    case False{}: Unit{}

def after_found(after: List<&2,U32>,+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +exact: {Public.lane_after(shape,axis) == after : List<&2,U32>}) ->
  ScanFound(TiledLayout.split_after(after,axis,Public.lane_prefix(shape,axis,1),Grid.lanes_at(shape,axis),TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape):
  match after:
    case Nil{}: Unit{}
    case Con{+width,+rest}:
      cond_found(U32.is_eq(Public.lane_prefix(shape,axis,1),Public.lane_prefix(values,own,1)) && U32.is_lt(1,Grid.lanes_at(shape,axis)) && U32.is_lt(1,width) &&
        U32.is_eq((Grid.lanes_at(shape,axis) * width : U32),Grid.lanes_at(values,own)) && U32.is_eq(Tiled.product(rest),Tiled.product(Public.lane_after(values,own))),
        values,own,shape,axis,width,rest,inside,exact,{==})

def axis_found(within: Bool,+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>,+axis: Nat,+within_value: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool},
  +inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  ScanFound(TiledLayout.split_of(Tiled.around(shape,axis,1),axis,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape):
  match within:
    case True{}:
      %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,axis,1),Some{Tiled.Around{Public.lane_prefix(shape,axis,1),Grid.lanes_at(shape,axis),Public.lane_after(shape,axis)}},
          Public.located(shape,axis,1,inside)) :
        ScanFound(TiledLayout.split_of(_,axis,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape)
      after_found(Public.lane_after(shape,axis),values,own,shape,axis,within_value,{==})
    case False{}:
      %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,axis,1),Some{Tiled.Around{Public.lane_prefix(shape,axis,1),Grid.lanes_at(shape,axis),Public.lane_after(shape,axis)}},
          Public.located(shape,axis,1,inside)) :
        ScanFound(TiledLayout.split_of(_,axis,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape)
      %Equal.sym(List<&2,U32>,Public.lane_after(shape,axis),[],Public.after_past(shape,axis,within_value)) :
        ScanFound(TiledLayout.split_after(_,axis,Public.lane_prefix(shape,axis,1),Grid.lanes_at(shape,axis),TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape)
      Unit{}

def scan_found(remaining: Nat,found: Maybe<&2,TiledLayout.Splitting>,+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>,+axis: Nat,+held: ScanFound(found,values,own,shape),
  +room: {Nat.is_le(Nat.add(axis,remaining),List.length(&2,U32,shape)) == True{} : Bool}) ->
  ScanFound(TiledLayout.split_scan(remaining,found,shape,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}},axis),values,own,shape):
  match remaining found:
    case remaining Some{+split}: held
    case 0n None{}: Unit{}
    case 1n+ +rest None{}:
      +next = Nat.add(axis,1n)
      +next_room = {%Equal.sym(Nat,Nat.add(next,rest),Nat.add(axis,1n+rest),Addition.associative(axis,1n,rest)) : {Nat.is_le(_,List.length(&2,U32,shape)) == True{} : Bool} room :
        {Nat.is_le(Nat.add(next,rest),List.length(&2,U32,shape)) == True{} : Bool}}
      scan_found(rest,TiledLayout.split_of(Tiled.around(shape,next,1),next,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape,next,
        axis_found(Nat.is_lt(next,List.length(&2,U32,shape)),values,own,shape,next,{==},
          Order.transitive(next,Nat.add(next,rest),List.length(&2,U32,shape),Order.le_add(next,rest),next_room)),next_room)

# The split of the lane axis a reshape finds is a split of the values.
def lane_found(+values: List<&2,U32>,+own: Nat,+shape: List<&2,U32>,+inside: {Nat.is_le(own,List.length(&2,U32,values)) == True{} : Bool}) ->
  ScanFound(TiledLayout.lane_split(Tiled.around(values,own,1),shape),values,own,shape):
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,own,1),Some{Tiled.Around{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Public.lane_after(values,own)}},
      Public.located(values,own,1,inside)) :
    ScanFound(TiledLayout.lane_split(_,shape),values,own,shape)
  scan_found(List.length(&2,U32,shape),TiledLayout.split_of(Tiled.around(shape,0n,1),0n,TiledLayout.Kept{Public.lane_prefix(values,own,1),Grid.lanes_at(values,own),Tiled.product(Public.lane_after(values,own)),False{}}),values,own,shape,0n,
    axis_found(Nat.is_lt(0n,List.length(&2,U32,shape)),values,own,shape,0n,{==},Order.zero_le(List.length(&2,U32,shape))),Order.reflexive(List.length(&2,U32,shape)))

# The lane axis a split splits lies inside the values: on the unit axis past
# them the lanes are one, which no two extents above one multiply to.
def inside_cases(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+own: Nat,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +within_value: {Nat.is_lt(own,List.length(&2,U32,values)) == within : Bool},+split: {(anchors * width : U32) == Grid.lanes_at(values,own) : U32},
  +anchors_above: {U32.is_lt(1,anchors) == True{} : Bool},+width_above: {U32.is_lt(1,width) == True{} : Bool},
  +shape_bound: {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}) ->
  {Nat.is_lt(own,List.length(&2,U32,values)) == True{} : Bool}:
  match within:
    case True{}: within_value
    case False{}:
      +a = U32.to_nat(anchors)
      +w = U32.to_nat(width)
      +unit = Equal.trans(U32,(anchors * width : U32),Grid.lanes_at(values,own),1,split,Public.lanes_past(values,own,within_value))
      +product_one = {%unit : {U32.to_nat(_) == Nat.mul(a,w) : Nat}
        Multiplication.bounded_multiply(anchors,width,Machine.capacity(limits),Limits.capacity_fits(limits),
          lanes_product_room(limits,pre,anchors,width,post,shape_bound,shape_nonempty)) : {U32.to_nat(1) == Nat.mul(a,w) : Nat}}
      +a_above = {%Comparison.lt(1,anchors) : {_ == True{} : Bool} anchors_above : {Nat.is_lt(1n,a) == True{} : Bool}}
      +w_above = {%Comparison.lt(1,width) : {_ == True{} : Bool} width_above : {Nat.is_lt(1n,w) == True{} : Bool}}
      +w_positive = Order.lt_le(0n,1n,w,{==},Bounds.strict_le(1n,w,w_above))
      +a_le = {%Equal.sym(Nat,1n,Nat.mul(a,w),product_one) : {Nat.is_le(a,_) == True{} : Bool} RowMajor.product_covers(a,w,w_positive) : {Nat.is_le(a,1n) == True{} : Bool}}
      Empty.absurd({Nat.is_lt(own,List.length(&2,U32,values)) == True{} : Bool},Held.lt_le_absurd(1n,a,a_above,a_le))

# Tiled values of the contract that a reshape splits hold the same values with
# the lanes on the second axis of the split.
def split_reached(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,
  -pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+own: Nat,+lanes: U32,+position: Grid.Position,+axis: Nat,+outer: U32,
  +anchors: U32,+width: U32,+after: U32,+found: SplitAt<values,own,shape,axis,outer,anchors,width,after>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},own,lanes,position}),
  +bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+same: {ShapeLaws.elements(values) == ShapeLaws.elements(shape) : Nat},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_counted: Global.Counted<shape>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +usable: Global.Usable<Tile.Tile8,cells,Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers}>) ->
  Public.Contract(~Logical,~original,logical,shape,
    Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},own,lanes,position},shape,axis,outer,anchors,width,after)):
  SplitAt{+pre,+post,+shape_exact,+axis_exact,+pre_outer,+front_outer,+post_after,+back_after,+split,+anchors_above,+width_above} = found
  KernelGemm.Both{+global,+laid} = contract
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  +front = front_of(values,own)
  +back = Public.lane_after(values,own)
  +shape_bound = {%Equal.sym(List<&2,U32>,List.append(&2,U32,pre,anchors <> width <> post),shape,shape_exact) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %same : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound : {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +shape_nonempty = {%Equal.sym(List<&2,U32>,List.append(&2,U32,pre,anchors <> width <> post),shape,shape_exact) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool}
    %same : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post))) == True{} : Bool}}
  +inside_lt = inside_cases(Nat.is_lt(own,List.length(&2,U32,values)),limits,values,own,pre,anchors,width,post,{==},split,anchors_above,width_above,shape_bound,shape_nonempty)
  +values_exact = {%pinned : {List.append(&2,U32,front,_ <> back) == values : List<&2,U32>} around_split(values,own,inside_lt) :
    {List.append(&2,U32,front,lanes <> back) == values : List<&2,U32>}}
  +own_exact = front_length(values,own,inside_lt)
  +values_bound = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound :
    {Nat.is_le(ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +values_positive = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive :
    {Nat.is_lt(0n,ShapeLaws.elements(List.append(&2,U32,front,lanes <> back))) == True{} : Bool}}
  +values_rank = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,List.append(&2,U32,front,lanes <> back)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +split_rank = {%Equal.sym(List<&2,U32>,List.append(&2,U32,pre,anchors <> width <> post),shape,shape_exact) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} shape_rank :
    {Nat.is_le(List.length(&2,U32,List.append(&2,U32,pre,anchors <> width <> post)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +split_same = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) : {ShapeLaws.elements(_) == ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) : Nat}
    %Equal.sym(List<&2,U32>,List.append(&2,U32,pre,anchors <> width <> post),shape,shape_exact) : {ShapeLaws.elements(values) == ShapeLaws.elements(_) : Nat} same :
    {ShapeLaws.elements(List.append(&2,U32,front,lanes <> back)) == ShapeLaws.elements(List.append(&2,U32,pre,anchors <> width <> post)) : Nat}}
  +split_counted = {%Equal.sym(List<&2,U32>,List.append(&2,U32,pre,anchors <> width <> post),shape,shape_exact) : Global.Counted<_> shape_counted : Global.Counted<List.append(&2,U32,pre,anchors <> width <> post)>}
  +split_global = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(_,List.length(&2,U32,front),position)},Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers})
    %Equal.sym(Nat,List.length(&2,U32,front),own,own_exact) : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,_,position)},Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers})
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position)},Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers})}
  +split_laid = {%Equal.sym(List<&2,U32>,List.append(&2,U32,front,lanes <> back),values,values_exact) : Public.Laid<cells,List.length(&2,U32,front),lanes,position,_>
    %Equal.sym(Nat,List.length(&2,U32,front),own,own_exact) : Public.Laid<cells,_,lanes,position,values>
    laid : Public.Laid<cells,List.length(&2,U32,front),lanes,position,List.append(&2,U32,front,lanes <> back)>}
  +facts = split_facts(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,values_bound,split_same,values_positive,values_rank,split_rank,front_outer,pre_outer,
    back_after,post_after,Equal.trans(U32,(anchors * width : U32),Grid.lanes_at(values,own),lanes,split,pinned))
  %shape_exact : Public.Contract(~Logical,~original,logical,_,Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},own,lanes,position},_,axis,outer,anchors,width,after))
  %axis_exact : Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},own,lanes,position},List.append(&2,U32,pre,anchors <> width <> post),_,outer,anchors,width,after))
  %own_exact : Public.Contract(~Logical,~original,logical,List.append(&2,U32,pre,anchors <> width <> post),Tiled.split_lanes(Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},_,lanes,position},List.append(&2,U32,pre,anchors <> width <> post),List.length(&2,U32,pre),outer,anchors,width,after))
  split_usable(~Logical,~original,logical,limits,front,lanes,back,pre,anchors,width,post,position,pieces,cells,fill,workers,outer,after,split_global,split_laid,
    usable,split_counted,default_capacity,facts)
