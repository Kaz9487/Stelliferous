# Padding of tiled values. The cells are padded once with the lanes on an axis
# the padding leaves whole: a constant fills the new cells with the value in
# every lane, or in the first lane of a cell of the unit axis, and an edge
# repeats the nearest cell. The padding of every piece puts +0 back in the
# lanes past the lane count, so the result holds the padding of the values.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_pieces_model.bend as PiecesModel
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./tensor_grid_pieces.bend as GridLaws
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_placing_moved.bend as PlacingMoved
import ./tensor_placing_law.bend as Law
import ./tensor_placing_gather.bend as Gathered
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./tensor_tiled_public.bend as Public
import ./tensor_cell_room.bend as Rooms
import ./tensor_tiled_representation.bend as Representation
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_tiled_placing.bend as Placing
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_box_digits.bend as Digits
import ./tensor_target_cells.bend as Targets
import ./tensor_relane_values.bend as Values
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_repositioned.bend as Repositioned
import ../tensor_layout.bend as Layout
import ./tensor_placements.bend as Placements
import ./tensor_placing_reads.bend as Reads
import ./tensor_placing_facts.bend as Facts
import ./tensor_components.bend as Components
import ./tensor_view_facts.bend as ViewFacts
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./nat_extrema.bend as NatExtrema
import ../tensor_shape.bend as Shapes
import ../tensor_tiled_layout.bend as TiledLayout
import ./tensor_padded_extents.bend as Extents
import ./word_encoding.bend as Words
import ./natural_addition.bend as Addition
import ./tensor_tiled_set_slice.bend as SetSlice
import ./tensor_padded_pieces.bend as PaddedPieces
import ./tensor_placing_parts.bend as Parts
import ./tensor_moved_sources.bend as Sources
import ./tensor_storage_count.bend as StorageCount
import ./tensor_count_completion.bend as CountCompletion
import ../default_machine_limits.bend as Default
import ./tensor_view_moved.bend as ViewMoved
import ../tensor_view.bend as Views

# ---- A placing, then padding ----

# Sources of the contract placed in a target of counted cells that the
# padding fits, then padded: a tensor of the contract that holds what padding
# leaves of the placed cells.
def padded_contract(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,
  +segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+fill: Tile.Tile8,+tensor_fill: Tile.Tile8,
  +workers: U32,+models: Moved.Models<Tile.Tile8>,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: Law.PlacingFacts(limits,target,segments,sources,shifts,clamps),+placed: PlacingMoved.SourcesPlaced(~Tile.Tile8,~Logical,~original,logicals,sources,models),
  +padding: TiledMap.Padding,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+fit: {PaddedPieces.groups_fit(cap,padding) == True{} : Bool},+total: {PaddedPieces.padded_cells(padding) == ShapeLaws.elements(target) : Nat},
  +count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>}) ->
  Global.Contract(~Tile.Tile8,~PaddedPieces.Padded<Gathered.Placings<Logical,Tile.Tile8>>,~PaddedPieces.padded_value(~Gathered.Placings<Logical,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Logical,~original)),
    PaddedPieces.Padded{padding,Gathered.Placings{logicals,segments,sources,target,fill}},
    Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,Law.placing_rows(target,sources,shifts,clamps),segments,Moved.pack_sources(~Tile.Tile8,models),
      Law.placing_origins(sources),target,fill,workers),target,tensor_fill,Tensors.Usable{},workers}):
  Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+sources_facts}}}} = facts
  +placings = {Gathered.Placings{logicals,segments,sources,target,fill} : Gathered.Placings<Logical,Tile.Tile8>}
  +divides = {GridRanges.Multiple{GridRanges.last_extent(target),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(target),1n),GridRanges.last_extent(target),Order.times_one(GridRanges.last_extent(target)))} :
    GridRanges.Multiple<GridRanges.last_extent(target),1n>}
  GridRanges.grid_contract(~Tile.Tile8,~PaddedPieces.Padded<Gathered.Placings<Logical,Tile.Tile8>>,~PaddedPieces.padded_value(~Gathered.Placings<Logical,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Logical,~original)),
    PaddedPieces.Padded{padding,placings},limits,target,tensor_fill,workers,
    Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,Law.placing_rows(target,sources,shifts,clamps),segments,Moved.pack_sources(~Tile.Tile8,models),Law.placing_origins(sources),target,fill,workers),
    room,nonempty,1n,divides,Global.Counted{count,counted},
    PaddedPieces.unguarded_model(~Gathered.Placings<Logical,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Logical,~original),cap,padding,placings,Pieces.ranges_of_grid(Tensors.grid_of(target)),
      Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,Law.placing_rows(target,sources,shifts,clamps),segments,Moved.pack_sources(~Tile.Tile8,models),Law.placing_origins(sources),target,fill,workers),
      ShapeLaws.elements(target),1n,PaddedPieces.layout_run(target,Grid.piece_target(),1n,GridRanges.grid_layout(limits,target,Grid.piece_target(),room,nonempty,1n,divides)),fit,total,
      PlacingMoved.placing_placed(~Tile.Tile8,~Logical,~original,~TiledMap.Padding,~TiledMap.padded,~Padding.step,~Padding.rule,
        ~PaddedPieces.Guarded(~Gathered.Placings<Logical,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Logical,~original),cap),
        ~PaddedPieces.guarded_lift(~Gathered.Placings<Logical,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Logical,~original),cap,cap_ok),
        padding,logicals,limits,target,segments,sources,shifts,clamps,fill,workers,models,named,
        {Held.Both{Held.Both{room,Held.Both{nonempty,rank_room}},Held.Both{components_ok,Held.Both{lists_positive,Held.Both{limits_at,sources_facts}}}} : Law.PlacingFacts(limits,target,segments,sources,shifts,clamps)},
        placed)))

# ---- A cell of the lanes inside the values ----

# A lane of a cell below the lane count: the words of the value it reads and
# what the sources give there.
type LaneHit<-Logical: Data,-original: Logical -> U32 -> F32,-logicals: List<&2,Logical>,-sources: List<&2,List<&2,U32>>,-shifts: List<&2,List<&2,U32>>,-total: U32,-rank: Nat,
  -axis: Nat,-others: List<&2,Nat>,-group: Nat,-lane: Nat,-fill_value: F32,-lifted_value: F32> is Data:
  LaneHit{value_words: List<&2,U32>,read: Placing.Read<total,rank,axis,others,group,lane,value_words>,
    equal: {lifted_value == Gathered.sources_value(~F32,~Logical,~original,logicals,Placing.value_segments(sources,shifts),sources,value_words,fill_value) : F32}}

def LaneCase(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+total: U32,
  +rank: Nat,+axis: Nat,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+fill_value: F32,+lifted_value: F32) -> Data:
  match valid:
    case True{}: LaneHit<Logical,original,logicals,sources,shifts,total,rank,axis,others,group,lane,fill_value,lifted_value>
    case False{}: Unit

# A lane of a placed cell agrees with the lane of a target that holds the
# value the sources give below the lane count and +0 past it.
def valid_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,+cap: U32,
  +values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+fill_value: F32,+others: List<&2,Nat>,+group: Nat,+lane: Nat,valid: Bool,
  +classified: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool},+cell_words: List<&2,U32>,
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: Placing.SourcesLanes(limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),sources,shifts,axis,position),
  +spot: Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,lane,cell_words>,+begun: Placing.Begun(Placing.covers(sources,shifts,cell_words,axis,position),fill_cell,fill_value,lane),
  +target: Tile.Tile8,+lifted_value: F32,+lifted_lane: {TileLaws.select_lane(lane,target) == Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),lifted_value,0.0) : F32},
  +hits: LaneCase(~Logical,~original,valid,logicals,sources,shifts,Grid.lanes_at(values,axis),List.length(&2,U32,values),axis,others,group,lane,fill_value,lifted_value)) ->
  {PaddedCells.kept(U32.to_nat(Grid.lanes_at(values,axis)),group,lane,TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),cell_words,fill_cell))) == TileLaws.select_lane(lane,target) : F32}:
  match valid:
    case True{}:
      LaneHit{+value_words,+read,+equal} = hits
      %Equal.sym(Bool,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),True{},classified) : {Bool.pick(F32,_,TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),cell_words,fill_cell)),0.0) == TileLaws.select_lane(lane,target) : F32}
      Equal.trans(F32,TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),cell_words,fill_cell)),Gathered.sources_value(~F32,~Logical,~original,logicals,Placing.value_segments(sources,shifts),sources,value_words,fill_value),TileLaws.select_lane(lane,target),
        Placing.sources_lanes(~Logical,~original,logicals,sources,shifts,limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),axis,position,others,group,lane,cell_words,value_words,fill_cell,fill_value,
          named,facts,spot,read,begun),
        Equal.sym(F32,TileLaws.select_lane(lane,target),Gathered.sources_value(~F32,~Logical,~original,logicals,Placing.value_segments(sources,shifts),sources,value_words,fill_value),
          Equal.trans(F32,TileLaws.select_lane(lane,target),lifted_value,Gathered.sources_value(~F32,~Logical,~original,logicals,Placing.value_segments(sources,shifts),sources,value_words,fill_value),
            {%classified : {TileLaws.select_lane(lane,target) == Bool.pick(F32,_,lifted_value,0.0) : F32} lifted_lane : {TileLaws.select_lane(lane,target) == lifted_value : F32}},equal)))
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),False{},classified) : {Bool.pick(F32,_,TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),cell_words,fill_cell)),0.0) == TileLaws.select_lane(lane,target) : F32}
      Equal.sym(F32,TileLaws.select_lane(lane,target),0.0,%classified : {TileLaws.select_lane(lane,target) == Bool.pick(F32,_,lifted_value,0.0) : F32} lifted_lane)

# The value a lane of a cell reads below the lane count, placed as it is.
def constant_hit(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,
  +values: List<&2,U32>,+axis: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +values_room: {Nat.is_le(Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap)) == True{} : Bool},+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +others: List<&2,Nat>,+others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),+group: Nat,+lane: Nat,+fill_value: F32,
  +classified: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool}) ->
  LaneCase(~Logical,~original,valid,logicals,sources,shifts,Grid.lanes_at(values,axis),List.length(&2,U32,values),axis,others,group,lane,fill_value,Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))))))):
  match valid:
    case True{}:
      +digits = Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))
      +below = Values.inserted_below(values,axis,others,Nat.add(lane,Nat.mul(group,8n)),others_below,classified,within)
      LaneHit{Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),Placing.Read{Placing.digits_words(values,digits,cap,cap_ok,values_room,below),Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),classified},{==}}
    case False{}: Unit{}

# ---- A cell of a constant ----

# A cell of the placing of a constant whose lanes lie inside the values
# agrees with the cell of the lifted placing of the values.
def constant_lanes(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,
  +cap: U32,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_value: F32,+word: U32,+target: Placing.Target<limits,cap,values,axis,position>,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: Placing.SourcesLanes(limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),sources,shifts,axis,position),
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,position)),Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tile.filled(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>:
  Placing.Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +below_cells = {%within : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(Grid.placed(True{},position,removed,groups),digits)}
  +others = Values.cell_others(True{},position,digits)
  +group = Values.cell_group(True{},position,digits)
  +others_below = Values.cell_below(True{},position,removed,groups,digits,below_cells)
  +group_inside = Values.group_below(True{},position,removed,groups,digits,below_cells)
  +whole = Values.cell_whole(True{},position,removed,groups,digits,below_cells)
  +exact = {%Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole) : {U32.to_nat(word) == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word)) :
    {U32.to_nat(word) == ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,others,group)) : Nat}}
  +cell_digits = Equal.trans(List<&2,Nat>,Digits.words_nat(Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),digits,Coordinates.placed_digits(True{},position,others,group),
    Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word),Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole))
  +others_same = Equal.trans(Nat,1n+List.length(&2,Nat,others),1n+List.length(&2,U32,removed),List.length(&2,U32,values),
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,others),List.length(&2,U32,removed),Values.below_length(removed,others,others_below)),
    Public.removed_length(values,axis,within))
  +cell_count = Digits.coordinates_count(cells,word)
  +lanes_same = Coordinates.geometry_lanes(values,axis,position,inside)
  +group_same = Coordinates.group_within(limits,values,axis,position,others,group,word,within,removed_room,others_below,group_inside,exact)
  %Equal.sym(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),U32.to_nat(Grid.lanes_at(values,axis)),lanes_same) :
    PaddedCells.Valid<_,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tile.filled(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>
  %Equal.sym(Nat,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),group,group_same) :
    PaddedCells.Valid<U32.to_nat(Grid.lanes_at(values,axis)),_,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tile.filled(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>
  PaddedCells.Valid{
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,0n,
        Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,0n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,0n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(0n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,0n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,0n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,1n,
        Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,1n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,1n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(1n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,1n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,1n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,2n,
        Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,2n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,2n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(2n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,2n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,2n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,3n,
        Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,3n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,3n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(3n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,3n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,3n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,4n,
        Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,4n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,4n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(4n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,4n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,4n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,5n,
        Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,5n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,5n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(5n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,5n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,5n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,6n,
        Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,6n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,6n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(6n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,6n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,6n,fill_value,{==})),
      valid_lane(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,Tile.filled(fill_value),fill_value,others,group,7n,
        Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},Grid.coordinates(Tiled.cell_shape(values,axis,position),word),named,facts,
        {Placing.Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,7n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tile.filled(fill_value),fill_value,7n,{==}),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(7n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,7n,{==}),
        constant_hit(~Logical,~original,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logicals,sources,shifts,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,7n,fill_value,{==}))}

# ---- A cell of the unit axis ----

# A cell of the placing of a constant with the lanes on the unit axis agrees
# with the cell of the lifted placing of the values.
def constant_unit(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,
  +cap: U32,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_value: F32,+word: U32,+target: Placing.Target<limits,cap,values,axis,position>,
  +past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: Placing.SourcesUnits(limits,cap,List.length(&2,U32,values),sources,shifts,axis),+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,position)),Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tiled.unit_tile(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>:
  Placing.Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +removed_same = Public.removed_past(values,axis,past)
  +others_below = {%past : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(removed,digits)}
  +values_below = {%removed_same : Coordinates.DigitsBelow(_,digits) others_below : Coordinates.DigitsBelow(values,digits)}
  +exact = Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word))
  +cell_digits = Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word)
  +one = Public.lanes_past(values,axis,past)
  +group_inside = {%Equal.sym(U32,Grid.lanes_at(values,axis),1,one) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
    {==} : {Nat.is_lt(0n,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool}}
  +lanes_one = Equal.trans(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),U32.to_nat(Grid.lanes_at(values,axis)),1n,Coordinates.geometry_lanes(values,axis,position,inside),Equal.cong(U32,Nat,U32.to_nat,Grid.lanes_at(values,axis),1,one))
  +group_same = Repositioned.group_unit(limits,values,axis,position,word,past,inside,removed_room,inside_word)
  %Equal.sym(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),1n,lanes_one) : PaddedCells.Valid<_,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tiled.unit_tile(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>
  %Equal.sym(Nat,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),0n,group_same) : PaddedCells.Valid<1n,_,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tiled.unit_tile(fill_value)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)>
  PaddedCells.Valid{
    Equal.trans(F32,TileLaws.select_lane(0n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,logicals,sources,axis,position),Placing.cell_segments(sources,shifts,axis,position),Placing.cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),Tiled.unit_tile(fill_value)},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      Placing.sources_units(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits))),Tiled.unit_tile(fill_value),fill_value,named,facts,
        {Placing.UnitSpot{cap_ok,cell_digits,Placing.digits_words(values,digits,cap,cap_ok,values_room,values_below),Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))} :
          Placing.UnitSpot<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))>},
        Placing.begun_of(Placing.covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),Tiled.unit_tile(fill_value),fill_value,0n,{==})),
      Equal.sym(F32,TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),
        %one : {TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(0n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,0n,{==}))),
    Equal.sym(F32,TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(1n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,1n,{==})),
    Equal.sym(F32,TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(2n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,2n,{==})),
    Equal.sym(F32,TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(3n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,3n,{==})),
    Equal.sym(F32,TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(4n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,4n,{==})),
    Equal.sym(F32,TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(5n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,5n,{==})),
    Equal.sym(F32,TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(6n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,6n,{==})),
    Equal.sym(F32,TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(7n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,Placing.value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,7n,{==}))}

# ---- An edge reads the clamped index ----

def le_false(+left: Nat,+right: Nat,+below: {Nat.is_lt(right,left) == True{} : Bool}) -> {Nat.is_le(left,right) == False{} : Bool}:
  match left right:
    case 0n +right: Empty.absurd({Nat.is_le(0n,right) == False{} : Bool},Order.below_zero(right,below))
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +more: le_false(rest,more,below)

# A segment holds an index, as numbers.
def holds_exact(+first: U32,+count: U32,+index: U32,+sum: {U32.to_nat((first + count : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}) ->
  {U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)) == Nat.is_le(U32.to_nat(first),U32.to_nat(index)) && Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) : Bool}:
  +left = U32.is_le(first,index)
  +right = U32.is_lt(index,(first + count : U32))
  +low = Nat.is_le(U32.to_nat(first),U32.to_nat(index))
  +high = Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)))
  Equal.trans(Bool,left && right,low && right,low && high,Equal.cong(Bool,Bool,value => value && right,left,low,Comparison.le(first,index)),
    Equal.cong(Bool,Bool,value => low && value,right,high,
      Equal.trans(Bool,right,Nat.is_lt(U32.to_nat(index),U32.to_nat((first + count : U32))),high,Comparison.lt(index,(first + count : U32)),
        Equal.cong(Nat,Bool,value => Nat.is_lt(U32.to_nat(index),value),U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),sum))))

# A segment that holds an index finds it; one that does not keeps what was
# found.
def step_hit(+first: U32,+count: U32,+source: U32,+step: U32,+rest: List<&2,Tasks.Segment>,+index: U32,+found: Maybe<&2,U32>,
  +hit: {U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)) == True{} : Bool}) ->
  {Placements.segment_index(Con{Tasks.Segment{first,count,source,step},rest},index,found) == Placements.segment_index(rest,index,Some{(source + (index - first) * step : U32)}) : Maybe<&2,U32>}:
  %Equal.sym(Bool,U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),True{},hit) :
    {Placements.segment_index(rest,index,Bool.pick(Maybe<&2,U32>,_,Some{(source + (index - first) * step : U32)},found)) == Placements.segment_index(rest,index,Some{(source + (index - first) * step : U32)}) : Maybe<&2,U32>}
  {==}

def step_miss(+first: U32,+count: U32,+source: U32,+step: U32,+rest: List<&2,Tasks.Segment>,+index: U32,+found: Maybe<&2,U32>,
  +miss: {U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)) == False{} : Bool}) ->
  {Placements.segment_index(Con{Tasks.Segment{first,count,source,step},rest},index,found) == Placements.segment_index(rest,index,found) : Maybe<&2,U32>}:
  %Equal.sym(Bool,U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),False{},miss) :
    {Placements.segment_index(rest,index,Bool.pick(Maybe<&2,U32>,_,Some{(source + (index - first) * step : U32)},found)) == Placements.segment_index(rest,index,found) : Maybe<&2,U32>}
  {==}

def kept_value(+clamped: Bool,+shift: U32,+bound: U32,+index: U32,+value: U32,
  +both: Held.Both<Reads.KeptAt(clamped,shift,bound,index),{U32.to_nat(value) == Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>) ->
  {U32.to_nat(value) == Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}:
  Held.Both{+kept,+equal} = both
  equal

def first_equal(+a: Nat,+b: Nat,+c: Nat,+d: Nat,+both: Held.Both<{a == b : Nat},{c == d : Nat}>) -> {a == b : Nat}:
  Held.Both{+first,+second} = both
  first

def second_equal(+a: Nat,+b: Nat,+c: Nat,+d: Nat,+both: Held.Both<{a == b : Nat},{c == d : Nat}>) -> {c == d : Nat}:
  Held.Both{+first,+second} = both
  second

# Two found words of one number are one found word.
def found_same(+left: U32,+right: U32,+kept: Nat,+left_kept: {U32.to_nat(left) == kept : Nat},+right_kept: {U32.to_nat(right) == kept : Nat}) ->
  {Some{left} == Some{right} : Maybe<&2,U32>}:
  Equal.cong(U32,Maybe<&2,U32>,value => {Some{value} : Maybe<&2,U32>},left,right,
    Arithmetic.to_nat_injective(left,right,Equal.trans(Nat,U32.to_nat(left),kept,U32.to_nat(right),left_kept,Equal.sym(Nat,U32.to_nat(right),kept,right_kept))))

# A clamped index lies before the shift and the bound.
def clamp_inside(+shift: Nat,+bound: Nat,+coordinate: Nat,+positive: {Nat.is_lt(0n,bound) == True{} : Bool}) ->
  {Nat.is_lt(Reads.clamped_index(shift,bound,coordinate),Nat.add(shift,bound)) == True{} : Bool}:
  +index = Reads.clamped_index(shift,bound,coordinate)
  %Order.restore_shift(shift,index,Reads.clamped_after(shift,bound,coordinate,positive)) : {Nat.is_lt(_,Nat.add(shift,bound)) == True{} : Bool}
  Order.offset_lt(shift,Nat.sub(index,shift),bound,Reads.clamped_below(shift,bound,coordinate,positive))

# The clamped word lies inside a placed source.
def clamp_after(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(shift),U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))),Reads.clamped_index(U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word)),
      Reads.clamp_exact(word,shift,bound,cap,cap_ok,shift_room,positive)) :
    {Nat.is_le(U32.to_nat(shift),_) == True{} : Bool}
  Reads.clamped_after(U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word),positive)

def clamp_before(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))),Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))),Reads.clamped_index(U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word)),
      Reads.clamp_exact(word,shift,bound,cap,cap_ok,shift_room,positive)) :
    {Nat.is_lt(_,Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == True{} : Bool}
  clamp_inside(U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word),positive)

# A placed source finds the clamped word.
def placed_found(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat}) ->
  {Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) == Some{(0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32)} : Maybe<&2,U32>}:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  step_hit(shift,bound,0,1,[],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{},Equal.trans(Bool,U32.is_le(shift,Tasks.clamp(word,shift,(shift + bound - 1 : U32))) && U32.is_lt(Tasks.clamp(word,shift,(shift + bound - 1 : U32)),(shift + bound : U32)),
    Nat.is_le(s,U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32)))) && Nat.is_lt(U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))),Nat.add(s,b)),True{},holds_exact(shift,bound,Tasks.clamp(word,shift,(shift + bound - 1 : U32)),sum_word),
    Boolean.and_true(Nat.is_le(s,U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32)))),Nat.is_lt(U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))),Nat.add(s,b)),clamp_after(cap,cap_ok,bound,shift,word,positive,shift_room),
      clamp_before(cap,cap_ok,bound,shift,word,positive,shift_room))))

# What a placed source reads at the clamped word is the index an edge keeps.
def placed_kept(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32)) == Reads.kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word)) : Nat}:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  %Reads.clamp_exact(word,shift,bound,cap,cap_ok,shift_room,positive) : {U32.to_nat((0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32)) == Nat.sub(_,s) : Nat}
  kept_value(False{},shift,bound,Tasks.clamp(word,shift,(shift + bound - 1 : U32)),(0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32),Reads.segment_kept(False{},shift,bound,0,1,shift,bound,{Reads.Middle{{==},{==},Order.reflexive(b)} : Reads.SegmentRead<shift,bound,0,1,shift,bound,False{}>},
    Tasks.clamp(word,shift,(shift + bound - 1 : U32)),cap,cap_ok,shift_room,clamp_after(cap,cap_ok,bound,shift,word,positive,shift_room),clamp_before(cap,cap_ok,bound,shift,word,positive,shift_room)))

# ---- The three reads of an edge ----

# A kept number found at the clamped word ends the same as the placed source.
def found_end(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+bound: U32,+shift: U32,+word: U32,+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},+kept: {U32.to_nat(value) == Reads.kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word)) : Nat}) ->
  {Some{value} == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  Equal.trans(Maybe<&2,U32>,Some{value},Some{(0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32)},Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}),found_same(value,(0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32),Reads.kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(word)),kept,placed_kept(cap,cap_ok,bound,shift,word,positive,shift_room)),
    Equal.sym(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}),Some{(0 + (Tasks.clamp(word,shift,(shift + bound - 1 : U32)) - shift) * 1 : U32)},placed_found(cap,cap_ok,bound,shift,word,positive,shift_room,sum_word)))

# An index before the shift reads the first index of the source.
def edge_before(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+total: U32,+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(total)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},+whole: {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat},
  +before: {Nat.is_lt(U32.to_nat(word),U32.to_nat(shift)) == True{} : Bool}) -> {Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}) == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  +t = U32.to_nat(word)
  +sum1 = Equal.cong(U32,Nat,U32.to_nat,(0 + shift : U32),shift,Arithmetic.add_zero_left(shift))
  +hit1 = Equal.trans(Bool,U32.is_le(0,word) && U32.is_lt(word,(0 + shift : U32)),Nat.is_le(U32.to_nat(0),t) && Nat.is_lt(t,Nat.add(U32.to_nat(0),s)),True{},holds_exact(0,shift,word,sum1),
    Boolean.and_true(Nat.is_le(U32.to_nat(0),t),Nat.is_lt(t,Nat.add(U32.to_nat(0),s)),Order.zero_le(t),before))
  +miss2 = Equal.trans(Bool,U32.is_le(shift,word) && U32.is_lt(word,(shift + bound : U32)),Nat.is_le(s,t) && Nat.is_lt(t,Nat.add(s,b)),False{},holds_exact(shift,bound,word,sum_word),
    %Equal.sym(Bool,Nat.is_le(s,t),False{},le_false(s,t,before)) : {_ && Nat.is_lt(t,Nat.add(s,b)) == False{} : Bool} {==})
  +below_sum = {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),Nat.add(s,b),sum_word) : {Nat.is_lt(t,_) == True{} : Bool}
    Bounds.nat_trans(t,s,Nat.add(s,b),before,Order.le_add(s,b)) : {Nat.is_lt(t,U32.to_nat((shift + bound : U32))) == True{} : Bool}}
  +miss3 = Equal.trans(Bool,U32.is_le((shift + bound : U32),word) && U32.is_lt(word,((shift + bound : U32) + (total - shift - bound : U32) : U32)),Nat.is_le(U32.to_nat((shift + bound : U32)),t) && Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))),False{},holds_exact((shift + bound : U32),(total - shift - bound : U32),word,FullRange.add((shift + bound : U32),(total - shift - bound : U32),total,%Equal.sym(Nat,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),whole) :
      {Nat.is_le(_,U32.to_nat(total)) == True{} : Bool} Order.reflexive(U32.to_nat(total)))),
    %Equal.sym(Bool,Nat.is_le(U32.to_nat((shift + bound : U32)),t),False{},le_false(U32.to_nat((shift + bound : U32)),t,below_sum)) : {_ && Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))) == False{} : Bool} {==})
  Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{(0 + (word - 0) * 0 : U32)},Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}),
    Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,Some{(0 + (word - 0) * 0 : U32)}),Some{(0 + (word - 0) * 0 : U32)},step_hit(0,shift,0,0,[Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{},hit1),
      Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,Some{(0 + (word - 0) * 0 : U32)}),Placements.segment_index([Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,Some{(0 + (word - 0) * 0 : U32)}),Some{(0 + (word - 0) * 0 : U32)},
        step_miss(shift,bound,0,1,[Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,Some{(0 + (word - 0) * 0 : U32)},miss2),step_miss((shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,[],word,Some{(0 + (word - 0) * 0 : U32)},miss3))),
    found_end(cap,cap_ok,bound,shift,word,(0 + (word - 0) * 0 : U32),positive,shift_room,sum_word,
      kept_value(True{},shift,bound,word,(0 + (word - 0) * 0 : U32),Reads.segment_kept(True{},0,shift,0,0,shift,bound,
        {Reads.Before{{==},{==},{==},positive,Facts.le_one_more(s)} : Reads.SegmentRead<0,shift,0,0,shift,bound,True{}>},word,cap,cap_ok,shift_room,Order.zero_le(t),before))))

# An index of the source reads itself.
def edge_middle(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+total: U32,+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(total)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},+whole: {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat},
  +after: {Nat.is_lt(U32.to_nat(word),U32.to_nat(shift)) == False{} : Bool},+before: {Nat.is_lt(U32.to_nat(word),Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == True{} : Bool}) -> {Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}) == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  +t = U32.to_nat(word)
  +sum1 = Equal.cong(U32,Nat,U32.to_nat,(0 + shift : U32),shift,Arithmetic.add_zero_left(shift))
  +le_st = Order.not_less(t,s,after)
  +miss1 = Equal.trans(Bool,U32.is_le(0,word) && U32.is_lt(word,(0 + shift : U32)),Nat.is_le(U32.to_nat(0),t) && Nat.is_lt(t,Nat.add(U32.to_nat(0),s)),False{},holds_exact(0,shift,word,sum1),
    %Equal.sym(Bool,Nat.is_le(U32.to_nat(0),t),True{},Order.zero_le(t)) : {_ && Nat.is_lt(t,Nat.add(U32.to_nat(0),s)) == False{} : Bool} after)
  +hit2 = Equal.trans(Bool,U32.is_le(shift,word) && U32.is_lt(word,(shift + bound : U32)),Nat.is_le(s,t) && Nat.is_lt(t,Nat.add(s,b)),True{},holds_exact(shift,bound,word,sum_word),Boolean.and_true(Nat.is_le(s,t),Nat.is_lt(t,Nat.add(s,b)),le_st,before))
  +below_sum = {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),Nat.add(s,b),sum_word) : {Nat.is_lt(t,_) == True{} : Bool} before : {Nat.is_lt(t,U32.to_nat((shift + bound : U32))) == True{} : Bool}}
  +miss3 = Equal.trans(Bool,U32.is_le((shift + bound : U32),word) && U32.is_lt(word,((shift + bound : U32) + (total - shift - bound : U32) : U32)),Nat.is_le(U32.to_nat((shift + bound : U32)),t) && Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))),False{},holds_exact((shift + bound : U32),(total - shift - bound : U32),word,FullRange.add((shift + bound : U32),(total - shift - bound : U32),total,%Equal.sym(Nat,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),whole) :
      {Nat.is_le(_,U32.to_nat(total)) == True{} : Bool} Order.reflexive(U32.to_nat(total)))),
    %Equal.sym(Bool,Nat.is_le(U32.to_nat((shift + bound : U32)),t),False{},le_false(U32.to_nat((shift + bound : U32)),t,below_sum)) : {_ && Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))) == False{} : Bool} {==})
  Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{(0 + (word - shift) * 1 : U32)},Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}),
    Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{(0 + (word - shift) * 1 : U32)},step_miss(0,shift,0,0,[Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{},miss1),
      Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Placements.segment_index([Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,Some{(0 + (word - shift) * 1 : U32)}),Some{(0 + (word - shift) * 1 : U32)},
        step_hit(shift,bound,0,1,[Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{},hit2),step_miss((shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,[],word,Some{(0 + (word - shift) * 1 : U32)},miss3))),
    found_end(cap,cap_ok,bound,shift,word,(0 + (word - shift) * 1 : U32),positive,shift_room,sum_word,
      kept_value(True{},shift,bound,word,(0 + (word - shift) * 1 : U32),Reads.segment_kept(True{},shift,bound,0,1,shift,bound,
        {Reads.Middle{{==},{==},Order.reflexive(b)} : Reads.SegmentRead<shift,bound,0,1,shift,bound,True{}>},word,cap,cap_ok,shift_room,le_st,before))))

# An index past the source reads its last index.
def edge_after(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+total: U32,+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(total)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},+whole: {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat},
  +after: {Nat.is_lt(U32.to_nat(word),U32.to_nat(shift)) == False{} : Bool},+past: {Nat.is_lt(U32.to_nat(word),Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == False{} : Bool}) -> {Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}) == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  +t = U32.to_nat(word)
  +sum1 = Equal.cong(U32,Nat,U32.to_nat,(0 + shift : U32),shift,Arithmetic.add_zero_left(shift))
  +le_st = Order.not_less(t,s,after)
  +miss1 = Equal.trans(Bool,U32.is_le(0,word) && U32.is_lt(word,(0 + shift : U32)),Nat.is_le(U32.to_nat(0),t) && Nat.is_lt(t,Nat.add(U32.to_nat(0),s)),False{},holds_exact(0,shift,word,sum1),
    %Equal.sym(Bool,Nat.is_le(U32.to_nat(0),t),True{},Order.zero_le(t)) : {_ && Nat.is_lt(t,Nat.add(U32.to_nat(0),s)) == False{} : Bool} after)
  +miss2 = Equal.trans(Bool,U32.is_le(shift,word) && U32.is_lt(word,(shift + bound : U32)),Nat.is_le(s,t) && Nat.is_lt(t,Nat.add(s,b)),False{},holds_exact(shift,bound,word,sum_word),
    %Equal.sym(Bool,Nat.is_le(s,t),True{},le_st) : {_ && Nat.is_lt(t,Nat.add(s,b)) == False{} : Bool} past)
  +sum_le = {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),Nat.add(s,b),sum_word) : {Nat.is_le(_,t) == True{} : Bool} Order.not_less(t,Nat.add(s,b),past) :
    {Nat.is_le(U32.to_nat((shift + bound : U32)),t) == True{} : Bool}}
  +below_end = {%Equal.sym(Nat,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),whole) : {Nat.is_lt(t,_) == True{} : Bool} inside :
    {Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))) == True{} : Bool}}
  +hit3 = Equal.trans(Bool,U32.is_le((shift + bound : U32),word) && U32.is_lt(word,((shift + bound : U32) + (total - shift - bound : U32) : U32)),Nat.is_le(U32.to_nat((shift + bound : U32)),t) && Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))),True{},holds_exact((shift + bound : U32),(total - shift - bound : U32),word,FullRange.add((shift + bound : U32),(total - shift - bound : U32),total,%Equal.sym(Nat,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),whole) :
      {Nat.is_le(_,U32.to_nat(total)) == True{} : Bool} Order.reflexive(U32.to_nat(total)))),
    Boolean.and_true(Nat.is_le(U32.to_nat((shift + bound : U32)),t),Nat.is_lt(t,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32)))),sum_le,below_end))
  +last = FullRange.subtract(bound,1,bound,Order.positive_one(b,positive),Order.sub_le(b,1n))
  +read = {Reads.After{{==},{==},
      {%Equal.sym(Nat,U32.to_nat((bound - 1 : U32)),Nat.sub(b,1n),last) : {Nat.add(_,1n) == b : Nat}
        Equal.trans(Nat,Nat.add(Nat.sub(b,1n),1n),1n+Nat.sub(b,1n),b,ViewFacts.plus_one(Nat.sub(b,1n)),Order.positive_has_last(b,positive)) :
        {Nat.add(U32.to_nat((bound - 1 : U32)),1n) == b : Nat}},
      {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),Nat.add(s,b),sum_word) : {Nat.is_le(Nat.add(s,b),1n+_) == True{} : Bool} Facts.le_one_more(Nat.add(s,b)) :
        {Nat.is_le(Nat.add(s,b),1n+U32.to_nat((shift + bound : U32))) == True{} : Bool}}} :
    Reads.SegmentRead<(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,shift,bound,True{}>}
  Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{((bound - 1 : U32) + (word - (shift + bound : U32)) * 0 : U32)},Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}),
    Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{((bound - 1 : U32) + (word - (shift + bound : U32)) * 0 : U32)},step_miss(0,shift,0,0,[Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{},miss1),
      Equal.trans(Maybe<&2,U32>,Placements.segment_index([Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Placements.segment_index([Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}),Some{((bound - 1 : U32) + (word - (shift + bound : U32)) * 0 : U32)},
        step_miss(shift,bound,0,1,[Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{},miss2),step_hit((shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,[],word,None{},hit3))),
    found_end(cap,cap_ok,bound,shift,word,((bound - 1 : U32) + (word - (shift + bound : U32)) * 0 : U32),positive,shift_room,sum_word,
      kept_value(True{},shift,bound,word,((bound - 1 : U32) + (word - (shift + bound : U32)) * 0 : U32),Reads.segment_kept(True{},(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,shift,bound,read,word,cap,cap_ok,shift_room,sum_le,below_end))))

def edge_cases(low: Bool,high: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+total: U32,+bound: U32,+shift: U32,+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(total)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +sum_word: {U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},+whole: {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat},
  +classified_low: {Nat.is_lt(U32.to_nat(word),U32.to_nat(shift)) == low : Bool},
  +classified_high: {Nat.is_lt(U32.to_nat(word),Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == high : Bool}) -> {Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}) == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  match low high:
    case True{} +high: edge_before(cap,cap_ok,total,bound,shift,word,positive,inside,shift_room,sum_word,whole,classified_low)
    case False{} True{}: edge_middle(cap,cap_ok,total,bound,shift,word,positive,inside,shift_room,sum_word,whole,classified_low,classified_high)
    case False{} False{}: edge_after(cap,cap_ok,total,bound,shift,word,positive,inside,shift_room,sum_word,whole,classified_low,classified_high)

# Along an axis an edge reads at an index what a placed source reads at the
# index clamped to the source.
def edge_index(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+total: U32,+bound: U32,+shift: U32,+word: U32,
  +fit: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(total)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(total)) == True{} : Bool}) -> {Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{}) == Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{}) : Maybe<&2,U32>}:
  +s = U32.to_nat(shift)
  +sum = Nat.add(s,U32.to_nat(bound))
  +room = Facts.after_room(total,bound,shift,fit)
  edge_cases(Nat.is_lt(U32.to_nat(word),s),Nat.is_lt(U32.to_nat(word),sum),cap,cap_ok,total,bound,shift,word,positive,inside,shift_room,
    first_equal(U32.to_nat((shift + bound : U32)),sum,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),room),
    second_equal(U32.to_nat((shift + bound : U32)),sum,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),room),{==},{==})

# ---- An edge reads the clamped words ----

# The words clamped to a source placed at shifts, and their numbers.
def clamp_words(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>) -> List<&2,U32>:
  match source shifts words:
    case Con{+extent,source} Con{+shift,shifts} Con{+word,words}: Tasks.clamp(word,shift,(shift + extent - 1 : U32)) <> clamp_words(source,shifts,words)
    case source shifts words: []

def clamped_digits(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>) -> List<&2,Nat>:
  match source shifts digits:
    case Con{+extent,source} Con{+shift,shifts} Con{+digit,digits}: Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit) <> clamped_digits(source,shifts,digits)
    case source shifts digits: []

# Along every axis an edge reads at words what the placed source reads at the
# words clamped to it.
def edge_cell(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +within: Facts.PlacedIn(target,source,shifts),+room: Placing.Room(source,shifts,cap),+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +below: Coordinates.DigitsBelow(target,Digits.words_nat(words))) -> {Placements.segments_cell(Layout.edge_segments(target,source,shifts),words,Tasks.strides_of(source)) == Placements.segments_cell(Layout.placed_segments(source,shifts),clamp_words(source,shifts,words),Tasks.strides_of(source)) : Maybe<&2,U32>}:
  match target source shifts words:
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} Con{+word,+more}:
      Held.Both{+fit,+fits} = within
      KernelGemm.Both{+shift_room,+rooms} = room
      KernelGemm.Both{+inside,+belows} = below
      +inner = ShapeLaws.elements(rest)
      +stride = Grid.product(rest)
      +edge = Placements.segment_index([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],word,None{})
      +placed = Placements.segment_index([Tasks.Segment{shift,bound,0,1}],Tasks.clamp(word,shift,(shift + bound - 1 : U32)),None{})
      +edge_rest = Placements.segments_cell(Layout.edge_segments(later,rest,moves),more,Tasks.strides_of(rest))
      +placed_rest = Placements.segments_cell(Layout.placed_segments(rest,moves),clamp_words(rest,moves,more),Tasks.strides_of(rest))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{bound,rest}),Con{stride,Grid.strides(rest)},GridLaws.strides_con(bound,rest)) :
        {Placements.segments_cell(Layout.edge_segments(Con{total,later},Con{bound,rest},Con{shift,moves}),Con{word,more},_) ==
          Placements.segments_cell(Layout.placed_segments(Con{bound,rest},Con{shift,moves}),clamp_words(Con{bound,rest},Con{shift,moves},Con{word,more}),_) : Maybe<&2,U32>}
      Equal.trans(Maybe<&2,U32>,Placements.combined(Placements.strided(edge,stride),edge_rest),Placements.combined(Placements.strided(placed,stride),edge_rest),
        Placements.combined(Placements.strided(placed,stride),placed_rest),
        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,found => Placements.combined(Placements.strided(found,stride),edge_rest),edge,placed,
          edge_index(cap,cap_ok,total,bound,shift,word,fit,shift_room,RowMajor.left_factor_positive(U32.to_nat(bound),inner,positive),inside)),
        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,found => Placements.combined(Placements.strided(placed,stride),found),edge_rest,placed_rest,
          edge_cell(later,rest,moves,more,cap,cap_ok,fits,rooms,DivisionOrder.right_factor_positive(U32.to_nat(bound),inner,positive),belows)))
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} Nil{}: {==}
    case Nil{} Nil{} Nil{} +words: {==}
    case Con{+total,+later} Nil{} +shifts +words: Empty.absurd({Placements.segments_cell(Layout.edge_segments(Con{total,later},Nil{},shifts),words,Tasks.strides_of(Nil{})) == Placements.segments_cell(Layout.placed_segments(Nil{},shifts),clamp_words(Nil{},shifts,words),Tasks.strides_of(Nil{})) : Maybe<&2,U32>},within)
    case Con{+total,+later} Con{+bound,+rest} Nil{} +words: Empty.absurd({Placements.segments_cell(Layout.edge_segments(Con{total,later},Con{bound,rest},Nil{}),words,Tasks.strides_of(Con{bound,rest})) == Placements.segments_cell(Layout.placed_segments(Con{bound,rest},Nil{}),clamp_words(Con{bound,rest},Nil{},words),Tasks.strides_of(Con{bound,rest})) : Maybe<&2,U32>},within)
    case Nil{} Con{+bound,+rest} +shifts +words: Empty.absurd({Placements.segments_cell(Layout.edge_segments(Nil{},Con{bound,rest},shifts),words,Tasks.strides_of(Con{bound,rest})) == Placements.segments_cell(Layout.placed_segments(Con{bound,rest},shifts),clamp_words(Con{bound,rest},shifts,words),Tasks.strides_of(Con{bound,rest})) : Maybe<&2,U32>},within)
    case Nil{} Nil{} Con{+shift,+moves} +words: Empty.absurd({Placements.segments_cell(Layout.edge_segments(Nil{},Nil{},Con{shift,moves}),words,Tasks.strides_of(Nil{})) == Placements.segments_cell(Layout.placed_segments(Nil{},Con{shift,moves}),clamp_words(Nil{},Con{shift,moves},words),Tasks.strides_of(Nil{})) : Maybe<&2,U32>},within)

# One source placed with its edges gives at words what it gives placed at the
# clamped words.
def edge_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +words: List<&2,U32>,+old: Element,+same: {Placements.segments_cell(Layout.edge_segments(target,source,shifts),words,Tasks.strides_of(source)) == Placements.segments_cell(Layout.placed_segments(source,shifts),clamp_words(source,shifts,words),Tasks.strides_of(source)) : Maybe<&2,U32>}) ->
  {Gathered.sources_value(~Element,~Logical,~original,[logical],[Layout.edge_segments(target,source,shifts)],[source],words,old) ==
    Gathered.sources_value(~Element,~Logical,~original,[logical],[Layout.placed_segments(source,shifts)],[source],clamp_words(source,shifts,words),old) : Element}:
  Equal.cong(Maybe<&2,U32>,Element,found => Placements.placed_result(~Element,~Logical,~original,logical,found,old),
    Placements.segments_cell(Layout.edge_segments(target,source,shifts),words,Tasks.strides_of(source)),
    Placements.segments_cell(Layout.placed_segments(source,shifts),clamp_words(source,shifts,words),Tasks.strides_of(source)),same)

# Clamped words lie inside the source.
def words_inside(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: Placing.Room(source,shifts,cap),+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}) ->
  {Placing.inside_words(source,shifts,clamp_words(source,shifts,words)) == True{} : Bool}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      KernelGemm.Both{+fit,+others} = room
      +inner = ShapeLaws.elements(rest)
      +bound_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +clamped = Tasks.clamp(word,shift,(shift + extent - 1 : U32))
      +s = U32.to_nat(shift)
      +here = Equal.trans(Bool,U32.is_le(shift,clamped) && U32.is_lt(clamped,(shift + extent : U32)),
        Nat.is_le(s,U32.to_nat(clamped)) && Nat.is_lt(U32.to_nat(clamped),Nat.add(s,U32.to_nat(extent))),True{},
        holds_exact(shift,extent,clamped,Arithmetic.add(shift,extent,cap,cap_ok,fit)),
        Boolean.and_true(Nat.is_le(s,U32.to_nat(clamped)),Nat.is_lt(U32.to_nat(clamped),Nat.add(s,U32.to_nat(extent))),
          clamp_after(cap,cap_ok,extent,shift,word,bound_positive,fit),clamp_before(cap,cap_ok,extent,shift,word,bound_positive,fit)))
      Boolean.and_true(U32.is_le(shift,clamped) && U32.is_lt(clamped,(shift + extent : U32)),Placing.inside_words(rest,later,clamp_words(rest,later,more)),here,
        words_inside(rest,later,more,cap,cap_ok,others,DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)))
    case Nil{} +shifts +words: {==}
    case Con{+extent,+rest} Nil{} +words: {==}
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}

# The numbers of clamped words are the clamped numbers of the words.
def clamp_nat(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: Placing.Room(source,shifts,cap),+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}) ->
  {Digits.words_nat(clamp_words(source,shifts,words)) == clamped_digits(source,shifts,Digits.words_nat(words)) : List<&2,Nat>}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      KernelGemm.Both{+fit,+others} = room
      +inner = ShapeLaws.elements(rest)
      +clamped = Tasks.clamp(word,shift,(shift + extent - 1 : U32))
      +index = Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),U32.to_nat(word))
      +tail = Digits.words_nat(clamp_words(rest,later,more))
      +digits = clamped_digits(rest,later,Digits.words_nat(more))
      Equal.trans(List<&2,Nat>,{Con{U32.to_nat(clamped),tail} : List<&2,Nat>},{Con{index,tail} : List<&2,Nat>},{Con{index,digits} : List<&2,Nat>},
        Equal.cong(Nat,List<&2,Nat>,value => {Con{value,tail} : List<&2,Nat>},U32.to_nat(clamped),index,
          Reads.clamp_exact(word,shift,extent,cap,cap_ok,fit,RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive))),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,value => {Con{index,value} : List<&2,Nat>},tail,digits,
          clamp_nat(rest,later,more,cap,cap_ok,others,DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive))))
    case Nil{} +shifts +words: {==}
    case Con{+extent,+rest} Nil{} +words: {==}
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}

# Clamped words count as the axes of the target.
def clamp_count(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+within: Facts.PlacedIn(target,source,shifts),
  +count: {List.length(&2,U32,words) == List.length(&2,U32,target) : Nat}) -> {List.length(&2,U32,clamp_words(source,shifts,words)) == List.length(&2,U32,target) : Nat}:
  match target source shifts words:
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} Con{+word,+more}:
      Held.Both{+fit,+fits} = within
      Equal.cong(Nat,Nat,n => 1n+n,List.length(&2,U32,clamp_words(rest,moves,more)),List.length(&2,U32,later),
        clamp_count(later,rest,moves,more,fits,Placing.pred_same(List.length(&2,U32,more),List.length(&2,U32,later),count)))
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} Nil{}:
      Empty.absurd({List.length(&2,U32,clamp_words(Con{bound,rest},Con{shift,moves},Nil{})) == List.length(&2,U32,Con{total,later}) : Nat},Digits.zero_successor(List.length(&2,U32,later),count))
    case Nil{} Nil{} Nil{} +words: {==}
    case Con{+total,+later} Nil{} +shifts +words: Empty.absurd({List.length(&2,U32,clamp_words(Nil{},shifts,words)) == List.length(&2,U32,Con{total,later}) : Nat},within)
    case Con{+total,+later} Con{+bound,+rest} Nil{} +words: Empty.absurd({List.length(&2,U32,clamp_words(Con{bound,rest},Nil{},words)) == List.length(&2,U32,Con{total,later}) : Nat},within)
    case Nil{} Con{+bound,+rest} +shifts +words: Empty.absurd({List.length(&2,U32,clamp_words(Con{bound,rest},shifts,words)) == List.length(&2,U32,Nil{}) : Nat},within)
    case Nil{} Nil{} Con{+shift,+moves} +words: Empty.absurd({List.length(&2,U32,clamp_words(Nil{},Con{shift,moves},words)) == List.length(&2,U32,Nil{}) : Nat},within)

# Clamped digits count as the digits.
def clamped_length(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+cap: U32,+room: Placing.Room(source,shifts,cap),
  +count: {List.length(&2,Nat,digits) == List.length(&2,U32,source) : Nat}) -> {List.length(&2,Nat,clamped_digits(source,shifts,digits)) == List.length(&2,Nat,digits) : Nat}:
  match source shifts digits:
    case Con{+extent,+rest} Con{+shift,+later} Con{+digit,+more}:
      KernelGemm.Both{+fit,+others} = room
      Equal.cong(Nat,Nat,n => 1n+n,List.length(&2,Nat,clamped_digits(rest,later,more)),List.length(&2,Nat,more),
        clamped_length(rest,later,more,cap,others,Placing.pred_same(List.length(&2,Nat,more),List.length(&2,U32,rest),count)))
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}
    case Nil{} +shifts +digits: Equal.sym(Nat,List.length(&2,Nat,digits),0n,count)
    case Con{+extent,+rest} Nil{} +digits:
      Empty.absurd({List.length(&2,Nat,clamped_digits(Con{extent,rest},Nil{},digits)) == List.length(&2,Nat,digits) : Nat},room)

# A digit below the extent of an axis without a shift stays as it is.
def clamped_first(+bound: Nat,+digit: Nat,+below: {Nat.is_lt(digit,bound) == True{} : Bool}) -> {Reads.clamped_index(0n,bound,digit) == digit : Nat}:
  +positive = Order.le_lt_trans(0n,digit,bound,Order.zero_le(digit),below)
  Equal.trans(Nat,Nat.min(Nat.max(digit,0n),Nat.sub(bound,1n)),Nat.max(digit,0n),digit,
    NatExtrema.minimum_when_ordered(Nat.max(digit,0n),Nat.sub(bound,1n),
      %Equal.sym(Nat,Nat.max(digit,0n),digit,Held.max_zero(digit)) : {Nat.is_le(_,Nat.sub(bound,1n)) == True{} : Bool} Padding.below_last(digit,bound,positive,below)),
    Held.max_zero(digit))

def clamp_unmoved(+shift: U32,+bound: U32,+digit: Nat,+front: {U32.to_nat(shift) == 0n : Nat},+below: {Nat.is_lt(digit,U32.to_nat(bound)) == True{} : Bool}) ->
  {Reads.clamped_index(U32.to_nat(shift),U32.to_nat(bound),digit) == digit : Nat}:
  %Equal.sym(Nat,U32.to_nat(shift),0n,front) : {Reads.clamped_index(_,U32.to_nat(bound),digit) == digit : Nat}
  clamped_first(U32.to_nat(bound),digit,below)

# Clamping digits with a lane inserted along an axis without a shift clamps
# the other digits.
def clamped_inserted(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,axis: Nat,others: List<&2,Nat>,+channel: Nat,+within: Facts.PlacedIn(target,source,shifts),
  +below: Coordinates.DigitsBelow(Grid.removed(target,axis),others),+axis_in: {Nat.is_lt(axis,List.length(&2,U32,target)) == True{} : Bool},
  +front: {U32.to_nat(Grid.lanes_at(shifts,axis)) == 0n : Nat},+channel_in: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(source,axis))) == True{} : Bool}) ->
  {clamped_digits(source,shifts,Coordinates.inserted_digit(axis,others,channel)) == Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(source,axis),Grid.removed(shifts,axis),others),channel) : List<&2,Nat>}:
  match target source shifts axis others:
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} 0n +others:
      Equal.cong(Nat,List<&2,Nat>,value => {Con{value,clamped_digits(rest,moves,others)} : List<&2,Nat>},Reads.clamped_index(U32.to_nat(shift),U32.to_nat(bound),channel),channel,
        clamp_unmoved(shift,bound,channel,front,channel_in))
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} 1n+ +next Con{+digit,+digits}:
      Held.Both{+fit,+fits} = within
      KernelGemm.Both{+inside,+belows} = below
      Equal.cong(List<&2,Nat>,List<&2,Nat>,value => {Con{Reads.clamped_index(U32.to_nat(shift),U32.to_nat(bound),digit),value} : List<&2,Nat>},
        clamped_digits(rest,moves,Coordinates.inserted_digit(next,digits,channel)),Coordinates.inserted_digit(next,clamped_digits(Grid.removed(rest,next),Grid.removed(moves,next),digits),channel),
        clamped_inserted(later,rest,moves,next,digits,channel,fits,belows,axis_in,front,channel_in))
    case Con{+total,+later} Con{+bound,+rest} Con{+shift,+moves} 1n+ +next Nil{}:
      Empty.absurd({clamped_digits(Con{bound,rest},Con{shift,moves},Coordinates.inserted_digit(1n+next,Nil{},channel)) == Coordinates.inserted_digit(1n+next,clamped_digits(Grid.removed(Con{bound,rest},1n+next),Grid.removed(Con{shift,moves},1n+next),Nil{}),channel) : List<&2,Nat>},below)
    case Nil{} +source +shifts +axis +others: Empty.absurd({clamped_digits(source,shifts,Coordinates.inserted_digit(axis,others,channel)) == Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(source,axis),Grid.removed(shifts,axis),others),channel) : List<&2,Nat>},Order.below_zero(axis,axis_in))
    case Con{+total,+later} Nil{} +shifts +axis +others: Empty.absurd({clamped_digits(Nil{},shifts,Coordinates.inserted_digit(axis,others,channel)) == Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(Nil{},axis),Grid.removed(shifts,axis),others),channel) : List<&2,Nat>},within)
    case Con{+total,+later} Con{+bound,+rest} Nil{} +axis +others: Empty.absurd({clamped_digits(Con{bound,rest},Nil{},Coordinates.inserted_digit(axis,others,channel)) == Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(Con{bound,rest},axis),Grid.removed(Nil{},axis),others),channel) : List<&2,Nat>},within)

# Clamping digits with one more axis clamps the digits and the last digit.
def clamped_append(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+extent: U32,+shift: U32,+digit: Nat,+cap: U32,+room: Placing.Room(source,shifts,cap),
  +count: {List.length(&2,Nat,digits) == List.length(&2,U32,source) : Nat}) -> {clamped_digits(List.append(&2,U32,source,[extent]),List.append(&2,U32,shifts,[shift]),List.append(&2,Nat,digits,[digit])) == List.append(&2,Nat,clamped_digits(source,shifts,digits),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]) : List<&2,Nat>}:
  match source shifts digits:
    case Con{+bound,+rest} Con{+moved,+later} Con{+first,+more}:
      KernelGemm.Both{+fit,+others} = room
      Equal.cong(List<&2,Nat>,List<&2,Nat>,value => {Con{Reads.clamped_index(U32.to_nat(moved),U32.to_nat(bound),first),value} : List<&2,Nat>},
        clamped_digits(List.append(&2,U32,rest,[extent]),List.append(&2,U32,later,[shift]),List.append(&2,Nat,more,[digit])),
        List.append(&2,Nat,clamped_digits(rest,later,more),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]),
        clamped_append(rest,later,more,extent,shift,digit,cap,others,Placing.pred_same(List.length(&2,Nat,more),List.length(&2,U32,rest),count)))
    case Nil{} Nil{} Nil{}: {==}
    case Con{+bound,+rest} Con{+moved,+later} Nil{}:
      Empty.absurd({clamped_digits(List.append(&2,U32,Con{bound,rest},[extent]),List.append(&2,U32,Con{moved,later},[shift]),List.append(&2,Nat,Nil{},[digit])) == List.append(&2,Nat,clamped_digits(Con{bound,rest},Con{moved,later},Nil{}),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,U32,rest),count))
    case Nil{} Nil{} Con{+first,+more}:
      Empty.absurd({clamped_digits(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Nil{},[shift]),List.append(&2,Nat,Con{first,more},[digit])) == List.append(&2,Nat,clamped_digits(Nil{},Nil{},Con{first,more}),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,count)))
    case Con{+bound,+rest} Nil{} +digits: Empty.absurd({clamped_digits(List.append(&2,U32,Con{bound,rest},[extent]),List.append(&2,U32,Nil{},[shift]),List.append(&2,Nat,digits,[digit])) == List.append(&2,Nat,clamped_digits(Con{bound,rest},Nil{},digits),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]) : List<&2,Nat>},room)
    case Nil{} Con{+moved,+later} +digits: Empty.absurd({clamped_digits(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Con{moved,later},[shift]),List.append(&2,Nat,digits,[digit])) == List.append(&2,Nat,clamped_digits(Nil{},Con{moved,later},digits),[Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),digit)]) : List<&2,Nat>},room)

# Clamping the digits of a cell with its group on an axis without a shift
# clamps the other digits.
def clamped_placed(position: Grid.Position,+source: List<&2,U32>,+shifts: List<&2,U32>,+others: List<&2,Nat>,+groups: U32,+group: Nat,+cap: U32,+room: Placing.Room(source,shifts,cap),
  +count: {List.length(&2,Nat,others) == List.length(&2,U32,source) : Nat},+group_in: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}) ->
  {clamped_digits(Grid.placed(True{},position,source,groups),Grid.placed(True{},position,shifts,0),Coordinates.placed_digits(True{},position,others,group)) ==
    Coordinates.placed_digits(True{},position,clamped_digits(source,shifts,others),group) : List<&2,Nat>}:
  match position:
    case Grid.Innermost{}:
      Equal.trans(List<&2,Nat>,clamped_digits(List.append(&2,U32,source,[groups]),List.append(&2,U32,shifts,[0]),List.append(&2,Nat,others,[group])),
        List.append(&2,Nat,clamped_digits(source,shifts,others),[Reads.clamped_index(0n,U32.to_nat(groups),group)]),List.append(&2,Nat,clamped_digits(source,shifts,others),[group]),
        clamped_append(source,shifts,others,groups,0,group,cap,room,count),
        Equal.cong(Nat,List<&2,Nat>,value => List.append(&2,Nat,clamped_digits(source,shifts,others),[value]),Reads.clamped_index(0n,U32.to_nat(groups),group),group,
          clamped_first(U32.to_nat(groups),group,group_in)))
    case Grid.Outermost{}:
      Equal.cong(Nat,List<&2,Nat>,value => {Con{value,clamped_digits(source,shifts,others)} : List<&2,Nat>},Reads.clamped_index(0n,U32.to_nat(groups),group),group,
        clamped_first(U32.to_nat(groups),group,group_in))

# ---- A cell of an edge ----

# The lanes of a cell clamped to the source: some source holds the cell.
def edge_begun(+source: List<&2,U32>,+before: List<&2,U32>,+cell_words: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+lane: Nat,
  +inside: {Placing.inside_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),cell_words) == True{} : Bool}) ->
  Placing.Begun(Placing.covers([source],[before],cell_words,axis,position),fill_cell,0.0,lane):
  %Equal.sym(Bool,Placing.inside_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),cell_words),True{},inside) : Placing.Begun(_ || False{},fill_cell,0.0,lane)
  Unit{}

# Clamping the digits of a cell of a source on an axis without a shift clamps
# its other digits.
def edge_placed_digits(+source: List<&2,U32>,+before: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+cap: U32,
  +source_within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},+shifts_in: {Nat.is_lt(axis,List.length(&2,U32,before)) == True{} : Bool},+front: {Grid.lanes_at(before,axis) == 0 : U32},
  +rooms: Placing.Room(Grid.removed(source,axis),Grid.removed(before,axis),cap),+count: {List.length(&2,Nat,others) == List.length(&2,U32,Grid.removed(source,axis)) : Nat},
  +group_in: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(source,axis)))) == True{} : Bool}) ->
  {clamped_digits(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),Coordinates.placed_digits(True{},position,others,group)) == Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group) : List<&2,Nat>}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,source)),True{},source_within) :
    {clamped_digits(Grid.placed(_,position,Grid.removed(source,axis),Grid.groups(Grid.lanes_at(source,axis))),Grid.placed(Nat.is_lt(axis,List.length(&2,U32,before)),position,Grid.removed(before,axis),(Grid.lanes_at(before,axis) / 8 : U32)),Coordinates.placed_digits(True{},position,others,group)) ==
      Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group) : List<&2,Nat>}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,before)),True{},shifts_in) :
    {clamped_digits(Grid.placed(True{},position,Grid.removed(source,axis),Grid.groups(Grid.lanes_at(source,axis))),Grid.placed(_,position,Grid.removed(before,axis),(Grid.lanes_at(before,axis) / 8 : U32)),Coordinates.placed_digits(True{},position,others,group)) == Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group) : List<&2,Nat>}
  %Equal.sym(U32,Grid.lanes_at(before,axis),0,front) :
    {clamped_digits(Grid.placed(True{},position,Grid.removed(source,axis),Grid.groups(Grid.lanes_at(source,axis))),Grid.placed(True{},position,Grid.removed(before,axis),(_ / 8 : U32)),Coordinates.placed_digits(True{},position,others,group)) == Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group) : List<&2,Nat>}
  clamped_placed(position,Grid.removed(source,axis),Grid.removed(before,axis),others,Grid.groups(Grid.lanes_at(source,axis)),group,cap,rooms,count,group_in)

# The value a lane of a cell of an edge reads below the lane count: the
# placed source at the clamped words.
def edge_hit(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+logical: Logical,+source: List<&2,U32>,+before: List<&2,U32>,+values: List<&2,U32>,+axis: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+values_room: {Nat.is_le(Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+others: List<&2,Nat>,+others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
  +group: Nat,+lane: Nat,+value_within: Facts.PlacedIn(values,source,before),+value_room: Placing.Room(source,before,cap),
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+front: {U32.to_nat(Grid.lanes_at(before,axis)) == 0n : Nat},+lane_same: {Grid.lanes_at(source,axis) == Grid.lanes_at(values,axis) : U32},
  +classified: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool}) ->
  LaneCase(~Logical,~original,valid,[logical],[source],[before],Grid.lanes_at(values,axis),List.length(&2,U32,values),axis,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group,lane,0.0,Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))))))):
  match valid:
    case True{}:
      +channel = Nat.add(lane,Nat.mul(group,8n))
      +digits = Coordinates.inserted_digit(axis,others,channel)
      +coords = Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))))))
      +below = Values.inserted_below(values,axis,others,channel,others_below,classified,within)
      +words_exact = Placing.digits_words(values,digits,cap,cap_ok,values_room,below)
      +channel_in = {%Equal.sym(U32,Grid.lanes_at(source,axis),Grid.lanes_at(values,axis),lane_same) : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} classified :
        {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(source,axis))) == True{} : Bool}}
      +value_digits = Equal.trans(List<&2,Nat>,Digits.words_nat(clamp_words(source,before,coords)),clamped_digits(source,before,Digits.words_nat(coords)),
        Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),channel),clamp_nat(source,before,coords,cap,cap_ok,value_room,positive),
        Equal.trans(List<&2,Nat>,clamped_digits(source,before,Digits.words_nat(coords)),clamped_digits(source,before,digits),Coordinates.inserted_digit(axis,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),channel),
          Equal.cong(List<&2,Nat>,List<&2,Nat>,found => clamped_digits(source,before,found),Digits.words_nat(coords),digits,words_exact),
          clamped_inserted(values,source,before,axis,others,channel,value_within,others_below,within,front,channel_in)))
      +below_words = {%Equal.sym(List<&2,Nat>,Digits.words_nat(coords),digits,words_exact) : Coordinates.DigitsBelow(values,_) below :
        Coordinates.DigitsBelow(values,Digits.words_nat(coords))}
      LaneHit{clamp_words(source,before,coords),
        Placing.Read{value_digits,clamp_count(values,source,before,coords,value_within,Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))))))),classified},
        edge_value(~F32,~Logical,~original,logical,values,source,before,coords,0.0,edge_cell(values,source,before,coords,cap,cap_ok,value_within,value_room,positive,below_words))}
    case False{}: Unit{}

# A cell of an edge whose lanes lie inside the values agrees with the cell of
# the lifted edge of the values.
def edge_lanes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+before: List<&2,U32>,+limits: Machine.Limits,+cap: U32,
  +values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+word: U32,+target: Placing.Target<limits,cap,values,axis,position>,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+facts: Placing.SourcesLanes(limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),[source],[before],axis,position),+value_within: Facts.PlacedIn(values,source,before),
  +cell_within: Facts.PlacedIn(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position)),+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +cell_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(source,axis,position))) == True{} : Bool},+front: {Grid.lanes_at(before,axis) == 0 : U32},+lane_same: {Grid.lanes_at(source,axis) == Grid.lanes_at(values,axis) : U32},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,position)),Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>:
  Placing.Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  KernelGemm.Both{+lanes,+others_lanes} = facts
  Placing.Lanes{+source_within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+source_removed,+aligned,+ending,+grouped,+lane_fits} = lanes
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +below_cells = {%within : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(Grid.placed(True{},position,removed,groups),digits)}
  +others = Values.cell_others(True{},position,digits)
  +group = Values.cell_group(True{},position,digits)
  +others_below = Values.cell_below(True{},position,removed,groups,digits,below_cells)
  +group_inside = Values.group_below(True{},position,removed,groups,digits,below_cells)
  +whole = Values.cell_whole(True{},position,removed,groups,digits,below_cells)
  +exact = {%Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole) : {U32.to_nat(word) == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word)) :
    {U32.to_nat(word) == ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,others,group)) : Nat}}
  +coords = Grid.coordinates(cells,word)
  +cell_digits = Equal.trans(List<&2,Nat>,Digits.words_nat(coords),digits,Coordinates.placed_digits(True{},position,others,group),
    Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word),Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole))
  +others_same = Equal.trans(Nat,1n+List.length(&2,Nat,others),1n+List.length(&2,U32,removed),List.length(&2,U32,values),
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,others),List.length(&2,U32,removed),Values.below_length(removed,others,others_below)),
    Public.removed_length(values,axis,within))
  +lanes_same = Coordinates.geometry_lanes(values,axis,position,inside)
  +group_same = Coordinates.group_within(limits,values,axis,position,others,group,word,within,removed_room,others_below,group_inside,exact)
  +cv = Tiled.cell_shape(source,axis,position)
  +csh = Placing.cell_shifts(before,axis,position)
  +cell_words = clamp_words(cv,csh,coords)
  +others_c = clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others)
  +removed_source = Grid.removed(source,axis)
  +removed_rooms = Placing.room_removed(source,before,axis,cap,value_room)
  +removed_count = Placing.pred_same(List.length(&2,U32,removed),List.length(&2,U32,removed_source),
    Equal.trans(Nat,1n+List.length(&2,U32,removed),List.length(&2,U32,values),1n+List.length(&2,U32,removed_source),Public.removed_length(values,axis,within),
      Equal.trans(Nat,List.length(&2,U32,values),List.length(&2,U32,source),1n+List.length(&2,U32,removed_source),Equal.sym(Nat,List.length(&2,U32,source),List.length(&2,U32,values),ranked),
        Equal.sym(Nat,1n+List.length(&2,U32,removed_source),List.length(&2,U32,source),Public.removed_length(source,axis,source_within)))))
  +others_count = Equal.trans(Nat,List.length(&2,Nat,others),List.length(&2,U32,removed),List.length(&2,U32,removed_source),Values.below_length(removed,others,others_below),removed_count)
  +group_source = {%Equal.sym(U32,Grid.lanes_at(source,axis),Grid.lanes_at(values,axis),lane_same) : {Nat.is_lt(group,U32.to_nat(Grid.groups(_))) == True{} : Bool} group_inside :
    {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(source,axis)))) == True{} : Bool}}
  +cell_digits_c = Equal.trans(List<&2,Nat>,Digits.words_nat(cell_words),clamped_digits(cv,csh,Digits.words_nat(coords)),Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group),
    clamp_nat(cv,csh,coords,cap,cap_ok,cell_room,cell_positive),
    Equal.trans(List<&2,Nat>,clamped_digits(cv,csh,Digits.words_nat(coords)),clamped_digits(cv,csh,Coordinates.placed_digits(True{},position,others,group)),Coordinates.placed_digits(True{},position,clamped_digits(Grid.removed(source,axis),Grid.removed(before,axis),others),group),
      Equal.cong(List<&2,Nat>,List<&2,Nat>,found => clamped_digits(cv,csh,found),Digits.words_nat(coords),Coordinates.placed_digits(True{},position,others,group),cell_digits),
      edge_placed_digits(source,before,axis,position,others,group,cap,source_within,Placing.shifts_within(source,before,axis,source_within,shifts_same),front,removed_rooms,
        others_count,group_source)))
  +others_same_c = Equal.trans(Nat,1n+List.length(&2,Nat,others_c),1n+List.length(&2,Nat,others),List.length(&2,U32,values),
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,others_c),List.length(&2,Nat,others),clamped_length(removed_source,Grid.removed(before,axis),others,cap,removed_rooms,others_count)),
    others_same)
  +cell_count_c = clamp_count(cells,cv,csh,coords,cell_within,Digits.coordinates_count(cells,word))
  +clamped_inside = words_inside(cv,csh,coords,cap,cap_ok,cell_room,cell_positive)
  +front_nat = Equal.cong(U32,Nat,U32.to_nat,Grid.lanes_at(before,axis),0,front)
  +below_coords = {%Equal.sym(List<&2,Nat>,Digits.words_nat(coords),digits,Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word)) : Coordinates.DigitsBelow(cells,_)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(cells,Digits.words_nat(coords))}
  +tile_same = Equal.sym(Tile.Tile8,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,[logical],[source],axis,position),Placing.cell_segments([source],[before],axis,position),Placing.cell_sources([source],axis,position),clamp_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),fill_cell),
    edge_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      cells,cv,csh,coords,fill_cell,edge_cell(cells,cv,csh,coords,cap,cap_ok,cell_within,cell_room,cell_positive,below_coords)))
  %Equal.sym(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),U32.to_nat(Grid.lanes_at(values,axis)),lanes_same) : PaddedCells.Valid<_,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  %Equal.sym(Nat,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),group,group_same) : PaddedCells.Valid<U32.to_nat(Grid.lanes_at(values,axis)),_,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  %tile_same : PaddedCells.Valid<U32.to_nat(Grid.lanes_at(values,axis)),group,_,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  PaddedCells.Valid{
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,0n,
        Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,0n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,0n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(0n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,0n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,0n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,1n,
        Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,1n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,1n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(1n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,1n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,1n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,2n,
        Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,2n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,2n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(2n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,2n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,2n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,3n,
        Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,3n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,3n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(3n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,3n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,3n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,4n,
        Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,4n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,4n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(4n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,4n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,4n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,5n,
        Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,5n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,5n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(5n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,5n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,5n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,6n,
        Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,6n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,6n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(6n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,6n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,6n,value_within,value_room,positive,front_nat,lane_same,{==})),
      valid_lane(~Logical,~original,[logical],[source],[before],limits,cap,values,axis,position,fill_cell,0.0,others_c,group,7n,
        Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cell_words,{==},facts,
        {Placing.Spot{cap_ok,others_same_c,cell_digits_c,cell_count_c,{==}} : Placing.Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others_c,group,7n,cell_words>},
        edge_begun(source,before,cell_words,axis,position,fill_cell,7n,clamped_inside),
        Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(7n,Nat.mul(group,8n)))))),
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,7n,{==}),
        edge_hit(~Logical,~original,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),logical,source,before,values,axis,cap,cap_ok,values_room,within,others,others_below,
          group,7n,value_within,value_room,positive,front_nat,lane_same,{==}))}

# A source with its lanes on the unit axis keeps its room in cells.
def past_room(+source: List<&2,U32>,+before: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cap: U32,+source_past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool},
  +shifts_past: {Nat.is_lt(axis,List.length(&2,U32,before)) == False{} : Bool},+room: Placing.Room(source,before,cap)) -> Placing.Room(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),cap):
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(source,axis,position),source,Placing.cells_past(source,axis,position,source_past)) : Placing.Room(_,Placing.cell_shifts(before,axis,position),cap)
  %Equal.sym(List<&2,U32>,Placing.cell_shifts(before,axis,position),before,Placing.moved_past(before,axis,position,shifts_past)) : Placing.Room(source,_,cap)
  room

# The numbers of the cell words of a source with its lanes on the unit axis
# clamped to it.
def past_digits(+source: List<&2,U32>,+before: List<&2,U32>,+axis: Nat,+position: Grid.Position,+words: List<&2,U32>,+digits: List<&2,Nat>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+source_past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool},
  +shifts_past: {Nat.is_lt(axis,List.length(&2,U32,before)) == False{} : Bool},+room: Placing.Room(source,before,cap),
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+exact: {Digits.words_nat(words) == digits : List<&2,Nat>}) ->
  {Digits.words_nat(clamp_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),words)) == clamped_digits(source,before,digits) : List<&2,Nat>}:
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(source,axis,position),source,Placing.cells_past(source,axis,position,source_past)) :
    {Digits.words_nat(clamp_words(_,Placing.cell_shifts(before,axis,position),words)) == clamped_digits(source,before,digits) : List<&2,Nat>}
  %Equal.sym(List<&2,U32>,Placing.cell_shifts(before,axis,position),before,Placing.moved_past(before,axis,position,shifts_past)) :
    {Digits.words_nat(clamp_words(source,_,words)) == clamped_digits(source,before,digits) : List<&2,Nat>}
  Equal.trans(List<&2,Nat>,Digits.words_nat(clamp_words(source,before,words)),clamped_digits(source,before,Digits.words_nat(words)),clamped_digits(source,before,digits),
    clamp_nat(source,before,words,cap,cap_ok,room,positive),Equal.cong(List<&2,Nat>,List<&2,Nat>,found => clamped_digits(source,before,found),Digits.words_nat(words),digits,exact))

# A cell of an edge with its lanes on the unit axis agrees with the cell of the
# lifted edge of the values.
def edge_unit(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+before: List<&2,U32>,+limits: Machine.Limits,+cap: U32,
  +values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+word: U32,+target: Placing.Target<limits,cap,values,axis,position>,
  +past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},+facts: Placing.SourcesUnits(limits,cap,List.length(&2,U32,values),[source],[before],axis),+value_within: Facts.PlacedIn(values,source,before),
  +cell_within: Facts.PlacedIn(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position)),+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +cell_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(source,axis,position))) == True{} : Bool},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,position)),Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>:
  Placing.Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  KernelGemm.Both{+units,+others_units} = facts
  Placing.Units{+source_past,+inside_le,+ranked,+shifts_same,+value_room,+value_fits,+source_removed} = units
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +removed_same = Public.removed_past(values,axis,past)
  +others_below = {%past : Coordinates.DigitsBelow(Grid.placed(_,position,removed,Grid.groups(Grid.lanes_at(values,axis))),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(removed,digits)}
  +values_below = {%removed_same : Coordinates.DigitsBelow(_,digits) others_below : Coordinates.DigitsBelow(values,digits)}
  +exact = Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word))
  +one = Public.lanes_past(values,axis,past)
  +group_inside = {%Equal.sym(U32,Grid.lanes_at(values,axis),1,one) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
    {==} : {Nat.is_lt(0n,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool}}
  +lanes_one = Equal.trans(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),U32.to_nat(Grid.lanes_at(values,axis)),1n,Coordinates.geometry_lanes(values,axis,position,inside),Equal.cong(U32,Nat,U32.to_nat,Grid.lanes_at(values,axis),1,one))
  +group_same = Repositioned.group_unit(limits,values,axis,position,word,past,inside,removed_room,inside_word)
  +cv = Tiled.cell_shape(source,axis,position)
  +csh = Placing.cell_shifts(before,axis,position)
  +coords = Grid.coordinates(cells,word)
  +vcoords = Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))
  +cell_words = clamp_words(cv,csh,coords)
  +value_words = clamp_words(source,before,vcoords)
  +others_c = clamped_digits(source,before,digits)
  +shifts_past = Placing.shifts_past(source,before,axis,source_past,shifts_same)
  +cell_room = past_room(source,before,axis,position,cap,source_past,shifts_past,value_room)
  +coords_exact = Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word)
  +words_exact = Placing.digits_words(values,digits,cap,cap_ok,values_room,values_below)
  +cell_digits_c = past_digits(source,before,axis,position,coords,digits,cap,cap_ok,source_past,shifts_past,value_room,positive,coords_exact)
  +value_digits_c = Equal.trans(List<&2,Nat>,Digits.words_nat(value_words),clamped_digits(source,before,Digits.words_nat(vcoords)),others_c,
    clamp_nat(source,before,vcoords,cap,cap_ok,value_room,positive),
    Equal.cong(List<&2,Nat>,List<&2,Nat>,found => clamped_digits(source,before,found),Digits.words_nat(vcoords),digits,words_exact))
  +value_count_c = clamp_count(values,source,before,vcoords,value_within,Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,digits))))
  +clamped_inside = words_inside(cv,csh,coords,cap,cap_ok,cell_room,cell_positive)
  +below_coords = {%Equal.sym(List<&2,Nat>,Digits.words_nat(coords),digits,coords_exact) : Coordinates.DigitsBelow(cells,_)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(cells,Digits.words_nat(coords))}
  +below_values = {%Equal.sym(List<&2,Nat>,Digits.words_nat(vcoords),digits,words_exact) : Coordinates.DigitsBelow(values,_) values_below :
    Coordinates.DigitsBelow(values,Digits.words_nat(vcoords))}
  +value_same = edge_value(~F32,~Logical,~original,logical,values,source,before,vcoords,0.0,edge_cell(values,source,before,vcoords,cap,cap_ok,value_within,value_room,positive,below_values))
  +tile_same = Equal.sym(Tile.Tile8,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,[logical],[source],axis,position),Placing.cell_segments([source],[before],axis,position),Placing.cell_sources([source],axis,position),clamp_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),fill_cell),
    edge_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      cells,cv,csh,coords,fill_cell,edge_cell(cells,cv,csh,coords,cap,cap_ok,cell_within,cell_room,cell_positive,below_coords)))
  %Equal.sym(Nat,Representation.lane_count(Public.geometry_of(values,axis,position)),1n,lanes_one) : PaddedCells.Valid<_,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  %Equal.sym(Nat,Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)),0n,group_same) : PaddedCells.Valid<1n,_,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[source],axis,position),[Layout.edge_segments(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position))],Placing.cell_sources([source],axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  %tile_same : PaddedCells.Valid<1n,0n,_,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)>
  PaddedCells.Valid{
    Equal.trans(F32,TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,[logical],[source],axis,position),Placing.cell_segments([source],[before],axis,position),Placing.cell_sources([source],axis,position),clamp_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),fill_cell)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),
      Equal.trans(F32,TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,[logical],[source],axis,position),Placing.cell_segments([source],[before],axis,position),Placing.cell_sources([source],axis,position),clamp_words(Tiled.cell_shape(source,axis,position),Placing.cell_shifts(before,axis,position),Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),fill_cell)),Gathered.sources_value(~F32,~Logical,~original,[logical],Placing.value_segments([source],[before]),[source],clamp_words(source,before,Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))),0.0),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),
        Placing.sources_units(~Logical,~original,[logical],[source],[before],limits,cap,List.length(&2,U32,values),axis,position,others_c,cell_words,value_words,fill_cell,0.0,{==},facts,
          {Placing.UnitSpot{cap_ok,cell_digits_c,value_digits_c,value_count_c} : Placing.UnitSpot<cap,List.length(&2,U32,values),others_c,cell_words,value_words>},
          edge_begun(source,before,cell_words,axis,position,fill_cell,0n,clamped_inside)),
        Equal.sym(F32,Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),Gathered.sources_value(~F32,~Logical,~original,[logical],Placing.value_segments([source],[before]),[source],clamp_words(source,before,Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))),0.0),value_same)),
      Equal.sym(F32,TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),
        %one : {TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(0n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,0n,{==}))),
    Equal.sym(F32,TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(1n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,1n,{==})),
    Equal.sym(F32,TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(2n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,2n,{==})),
    Equal.sym(F32,TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(3n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,3n,{==})),
    Equal.sym(F32,TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(4n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,4n,{==})),
    Equal.sym(F32,TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(5n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,5n,{==})),
    Equal.sym(F32,TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(6n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,6n,{==})),
    Equal.sym(F32,TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)),0.0,
      %one : {TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(7n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{[logical],[Layout.edge_segments(values,source,before)],[source],values,0.0},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,7n,{==}))}

# ---- The cells of padded values ----

def head_extent(extents: List<&2,U32>) -> U32:
  match extents:
    case Con{+head,rest}: head
    case Nil{}: 0

def first_extent(found: Maybe<&2,List<&2,U32>>) -> U32:
  match found:
    case Some{+extents}: head_extent(extents)
    case None{}: 0

# A prepended extent keeps its head before any tail.
def prepend_swap(found: Maybe<&2,U32>,+tail: List<&2,U32>,+other: List<&2,U32>,+total: U32,
  +exact: {Shapes.prepend_extent(found,Some{tail}) == Some{total <> tail} : Maybe<&2,List<&2,U32>>}) ->
  {Shapes.prepend_extent(found,Some{other}) == Some{total <> other} : Maybe<&2,List<&2,U32>>}:
  match found:
    case None{}:
      Empty.absurd({Shapes.prepend_extent(None{},Some{other}) == Some{total <> other} : Maybe<&2,List<&2,U32>>},
        Words.false_true(Equal.cong(Maybe<&2,List<&2,U32>>,Bool,value => Extents.present_extents(value),None{},Some{total <> tail},exact)))
    case Some{+head}:
      Equal.cong(U32,Maybe<&2,List<&2,U32>>,value => {Some{value <> other} : Maybe<&2,List<&2,U32>>},head,total,
        Equal.cong(Maybe<&2,List<&2,U32>>,U32,value => first_extent(value),Some{head <> tail},Some{total <> tail},exact))

# Zero is below every word, and adding zero on either side keeps a word.
def zero_below(+value: U32) -> {U32.is_le(0,value) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(0,value),Nat.is_le(U32.to_nat(0),U32.to_nat(value)),True{},Comparison.le(0,value),Order.zero_le(U32.to_nat(value)))

def checked_zero(+extent: U32) -> {Shapes.checked_add_to(Shapes.checked_add(0,extent),0) == Some{extent} : Maybe<&2,U32>}:
  %Equal.sym(Bool,U32.is_le(extent,4294967295),True{},FullRange.maximum(extent)) :
    {Shapes.checked_add_to(Shapes.sum_when(_,(0 + extent : U32)),0) == Some{extent} : Maybe<&2,U32>}
  %Equal.sym(U32,(0 + extent : U32),extent,Arithmetic.add_zero_left(extent)) : {Shapes.checked_add_to(Some{_},0) == Some{extent} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(0,(4294967295 - extent : U32)),True{},zero_below((4294967295 - extent : U32))) :
    {Shapes.sum_when(_,(extent + 0 : U32)) == Some{extent} : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,value => {Some{value} : Maybe<&2,U32>},(extent + 0 : U32),extent,
    Equal.trans(U32,(extent + 0 : U32),(0 + extent : U32),extent,Arithmetic.add_commutative(extent,0),Arithmetic.add_zero_left(extent)))

# Padded extents: as many fronts and backs as extents.
def padded_lengths(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) ->
  Held.Both<{List.length(&2,U32,before) == List.length(&2,U32,extents) : Nat},{List.length(&2,U32,after) == List.length(&2,U32,extents) : Nat}>:
  match extents before after:
    case Nil{} Nil{} Nil{}: Held.Both{{==},{==}}
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(front,extent),back),Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,{==},{==},exact)
      +tail = Extents.prepended_tail(front,extent,back,rest,fronts,backs,padded,found)
      +later = padded_lengths(rest,fronts,backs,tail,Extents.prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found))
      Held.Both{Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,fronts),List.length(&2,U32,rest),
          first_equal(List.length(&2,U32,fronts),List.length(&2,U32,rest),List.length(&2,U32,backs),List.length(&2,U32,rest),later)),
        Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,backs),List.length(&2,U32,rest),
          second_equal(List.length(&2,U32,fronts),List.length(&2,U32,rest),List.length(&2,U32,backs),List.length(&2,U32,rest),later))}
    case Con{+extent,+rest} Nil{} +after:
      Empty.absurd(Held.Both<{List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{extent,rest}) : Nat},{List.length(&2,U32,after) == List.length(&2,U32,Con{extent,rest}) : Nat}>,
        Extents.unpadded(padded,exact))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{}:
      Empty.absurd(Held.Both<{List.length(&2,U32,Con{front,fronts}) == List.length(&2,U32,Con{extent,rest}) : Nat},{List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{extent,rest}) : Nat}>,
        Extents.unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after:
      Empty.absurd(Held.Both<{List.length(&2,U32,Con{front,fronts}) == List.length(&2,U32,Nil{}) : Nat},{List.length(&2,U32,after) == List.length(&2,U32,Nil{}) : Nat}>,
        Extents.unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs}:
      Empty.absurd(Held.Both<{List.length(&2,U32,Nil{}) == List.length(&2,U32,Nil{}) : Nat},{List.length(&2,U32,Con{back,backs}) == List.length(&2,U32,Nil{}) : Nat}>,
        Extents.unpadded(padded,exact))

# The extents the cells grow to are the padded extents.
def padded_grown(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) -> {TiledLayout.grown(extents,before,after) == padded : List<&2,U32>}:
  match extents before after:
    case Nil{} Nil{} Nil{}:
      Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => Extents.known_extents(found),Some{[]},Some{padded},exact)
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(front,extent),back),Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,{==},{==},exact)
      +total = Extents.prepended_total(front,extent,back,rest,fronts,backs,padded,found)
      +tail = Extents.prepended_tail(front,extent,back,rest,fronts,backs,padded,found)
      +widened = Extents.prepended_widened(front,extent,back,rest,fronts,backs,padded,found)
      +f = U32.to_nat(front)
      +e = U32.to_nat(extent)
      +g = U32.to_nat(back)
      +t = U32.to_nat(total)
      +inner = (extent + front : U32)
      +front_room = {%widened : {Nat.is_le(Nat.add(f,e),_) == True{} : Bool} Order.le_add(Nat.add(f,e),g) : {Nat.is_le(Nat.add(f,e),t) == True{} : Bool}}
      +inner_room = {%Order.commutative(f,e) : {Nat.is_le(_,t) == True{} : Bool} front_room : {Nat.is_le(Nat.add(e,f),t) == True{} : Bool}}
      +inner_exact = FullRange.add(extent,front,total,inner_room)
      +sum_same = Equal.trans(Nat,Nat.add(U32.to_nat(inner),g),Nat.add(Nat.add(e,f),g),t,
        Equal.cong(Nat,Nat,number => Nat.add(number,g),U32.to_nat(inner),Nat.add(e,f),inner_exact),
        Equal.trans(Nat,Nat.add(Nat.add(e,f),g),Nat.add(Nat.add(f,e),g),t,Equal.cong(Nat,Nat,number => Nat.add(number,g),Nat.add(e,f),Nat.add(f,e),Order.commutative(e,f)),widened))
      +sum_room = {%Equal.sym(Nat,Nat.add(U32.to_nat(inner),g),t,sum_same) : {Nat.is_le(_,t) == True{} : Bool} Order.reflexive(t) : {Nat.is_le(Nat.add(U32.to_nat(inner),g),t) == True{} : Bool}}
      +word = Arithmetic.to_nat_injective((extent + front + back : U32),total,Equal.trans(Nat,U32.to_nat((extent + front + back : U32)),Nat.add(U32.to_nat(inner),g),t,
        FullRange.add(inner,back,total,sum_room),sum_same))
      %Extents.prepended_exact(front,extent,back,rest,fronts,backs,padded,found) : {TiledLayout.grown(Con{extent,rest},Con{front,fronts},Con{back,backs}) == _ : List<&2,U32>}
      Equal.trans(List<&2,U32>,{(extent + front + back : U32) <> TiledLayout.grown(rest,fronts,backs) : List<&2,U32>},{total <> TiledLayout.grown(rest,fronts,backs) : List<&2,U32>},{total <> tail : List<&2,U32>},
        Equal.cong(U32,List<&2,U32>,head => {head <> TiledLayout.grown(rest,fronts,backs) : List<&2,U32>},(extent + front + back : U32),total,word),
        Equal.cong(List<&2,U32>,List<&2,U32>,later => {total <> later : List<&2,U32>},TiledLayout.grown(rest,fronts,backs),tail,
          padded_grown(rest,fronts,backs,tail,Extents.prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found))))
    case Con{+extent,+rest} Nil{} +after:
      Empty.absurd({TiledLayout.grown(Con{extent,rest},Nil{},after) == padded : List<&2,U32>},Extents.unpadded(padded,exact))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{}:
      Empty.absurd({TiledLayout.grown(Con{extent,rest},Con{front,fronts},Nil{}) == padded : List<&2,U32>},Extents.unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after:
      Empty.absurd({TiledLayout.grown(Nil{},Con{front,fronts},after) == padded : List<&2,U32>},Extents.unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs}:
      Empty.absurd({TiledLayout.grown(Nil{},Nil{},Con{back,backs}) == padded : List<&2,U32>},Extents.unpadded(padded,exact))

# Padding without one axis pads the other axes.
def padded_removed(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,axis: Nat,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) ->
  {Shapes.padded(Grid.removed(extents,axis),Grid.removed(before,axis),Grid.removed(after,axis)) == Some{Grid.removed(padded,axis)} : Maybe<&2,List<&2,U32>>}:
  match extents before after axis:
    case Nil{} Nil{} Nil{} +axis:
      %Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => Extents.known_extents(found),Some{[]},Some{padded},exact) :
        {Shapes.padded(Nil{},Nil{},Nil{}) == Some{Grid.removed(_,axis)} : Maybe<&2,List<&2,U32>>}
      {==}
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs} 0n:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(front,extent),back),Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,{==},{==},exact)
      %Extents.prepended_exact(front,extent,back,rest,fronts,backs,padded,found) : {Shapes.padded(rest,fronts,backs) == Some{Grid.removed(_,0n)} : Maybe<&2,List<&2,U32>>}
      Extents.prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found)
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs} 1n+ +next:
      +sum = Shapes.checked_add_to(Shapes.checked_add(front,extent),back)
      +found = Extents.prepended(sum,Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,{==},{==},exact)
      +total = Extents.prepended_total(front,extent,back,rest,fronts,backs,padded,found)
      +tail = Extents.prepended_tail(front,extent,back,rest,fronts,backs,padded,found)
      +tail_exact = Extents.prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found)
      +padded_exact = Extents.prepended_exact(front,extent,back,rest,fronts,backs,padded,found)
      +head_exact = {%tail_exact : {Shapes.prepend_extent(sum,_) == Some{total <> tail} : Maybe<&2,List<&2,U32>>}
        %Equal.sym(List<&2,U32>,total <> tail,padded,padded_exact) : {Shapes.prepend_extent(sum,Shapes.padded(rest,fronts,backs)) == Some{_} : Maybe<&2,List<&2,U32>>}
        exact : {Shapes.prepend_extent(sum,Some{tail}) == Some{total <> tail} : Maybe<&2,List<&2,U32>>}}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.padded(Grid.removed(rest,next),Grid.removed(fronts,next),Grid.removed(backs,next)),Some{Grid.removed(tail,next)},
          padded_removed(rest,fronts,backs,next,tail,tail_exact)) :
        {Shapes.prepend_extent(sum,_) == Some{Grid.removed(padded,1n+next)} : Maybe<&2,List<&2,U32>>}
      %padded_exact : {Shapes.prepend_extent(sum,Some{Grid.removed(tail,next)}) == Some{Grid.removed(_,1n+next)} : Maybe<&2,List<&2,U32>>}
      prepend_swap(sum,tail,Grid.removed(tail,next),total,head_exact)
    case Con{+extent,+rest} Nil{} +after +axis:
      Empty.absurd({Shapes.padded(Grid.removed(Con{extent,rest},axis),Grid.removed(Nil{},axis),Grid.removed(after,axis)) == Some{Grid.removed(padded,axis)} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{} +axis:
      Empty.absurd({Shapes.padded(Grid.removed(Con{extent,rest},axis),Grid.removed(Con{front,fronts},axis),Grid.removed(Nil{},axis)) == Some{Grid.removed(padded,axis)} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after +axis:
      Empty.absurd({Shapes.padded(Grid.removed(Nil{},axis),Grid.removed(Con{front,fronts},axis),Grid.removed(after,axis)) == Some{Grid.removed(padded,axis)} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs} +axis:
      Empty.absurd({Shapes.padded(Grid.removed(Nil{},axis),Grid.removed(Nil{},axis),Grid.removed(Con{back,backs},axis)) == Some{Grid.removed(padded,axis)} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))

# Padding with one more unpadded axis pads it to its extent.
def padded_appended(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+padded: List<&2,U32>,+extent: U32,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) ->
  {Shapes.padded(List.append(&2,U32,extents,[extent]),List.append(&2,U32,before,[0]),List.append(&2,U32,after,[0])) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>}:
  match extents before after:
    case Nil{} Nil{} Nil{}:
      %Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => Extents.known_extents(found),Some{[]},Some{padded},exact) :
        {Shapes.padded([extent],[0],[0]) == Some{List.append(&2,U32,_,[extent])} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_add_to(Shapes.checked_add(0,extent),0),Some{extent},checked_zero(extent)) :
        {Shapes.prepend_extent(_,Some{[]}) == Some{[extent]} : Maybe<&2,List<&2,U32>>}
      {==}
    case Con{+first,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +sum = Shapes.checked_add_to(Shapes.checked_add(front,first),back)
      +found = Extents.prepended(sum,Shapes.padded(rest,fronts,backs),front,first,back,rest,fronts,backs,padded,{==},{==},exact)
      +total = Extents.prepended_total(front,first,back,rest,fronts,backs,padded,found)
      +tail = Extents.prepended_tail(front,first,back,rest,fronts,backs,padded,found)
      +tail_exact = Extents.prepended_tail_exact(front,first,back,rest,fronts,backs,padded,found)
      +padded_exact = Extents.prepended_exact(front,first,back,rest,fronts,backs,padded,found)
      +head_exact = {%tail_exact : {Shapes.prepend_extent(sum,_) == Some{total <> tail} : Maybe<&2,List<&2,U32>>}
        %Equal.sym(List<&2,U32>,total <> tail,padded,padded_exact) : {Shapes.prepend_extent(sum,Shapes.padded(rest,fronts,backs)) == Some{_} : Maybe<&2,List<&2,U32>>}
        exact : {Shapes.prepend_extent(sum,Some{tail}) == Some{total <> tail} : Maybe<&2,List<&2,U32>>}}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.padded(List.append(&2,U32,rest,[extent]),List.append(&2,U32,fronts,[0]),List.append(&2,U32,backs,[0])),
          Some{List.append(&2,U32,tail,[extent])},padded_appended(rest,fronts,backs,tail,extent,tail_exact)) :
        {Shapes.prepend_extent(sum,_) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>}
      %padded_exact : {Shapes.prepend_extent(sum,Some{List.append(&2,U32,tail,[extent])}) == Some{List.append(&2,U32,_,[extent])} : Maybe<&2,List<&2,U32>>}
      prepend_swap(sum,tail,List.append(&2,U32,tail,[extent]),total,head_exact)
    case Con{+first,+rest} Nil{} +after:
      Empty.absurd({Shapes.padded(List.append(&2,U32,Con{first,rest},[extent]),List.append(&2,U32,Nil{},[0]),List.append(&2,U32,after,[0])) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Con{+first,+rest} Con{+front,+fronts} Nil{}:
      Empty.absurd({Shapes.padded(List.append(&2,U32,Con{first,rest},[extent]),List.append(&2,U32,Con{front,fronts},[0]),List.append(&2,U32,Nil{},[0])) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after:
      Empty.absurd({Shapes.padded(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Con{front,fronts},[0]),List.append(&2,U32,after,[0])) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs}:
      Empty.absurd({Shapes.padded(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Nil{},[0]),List.append(&2,U32,Con{back,backs},[0])) == Some{List.append(&2,U32,padded,[extent])} : Maybe<&2,List<&2,U32>>},
        Extents.unpadded(padded,exact))

# Padding with the groups placed unpadded pads the groups to themselves.
def padded_placed(position: Grid.Position,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,+groups: U32,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) ->
  {Shapes.padded(Grid.placed(True{},position,extents,groups),Grid.placed(True{},position,before,0),Grid.placed(True{},position,after,0)) ==
    Some{Grid.placed(True{},position,padded,groups)} : Maybe<&2,List<&2,U32>>}:
  match position:
    case Grid.Innermost{}: padded_appended(extents,before,after,padded,groups,exact)
    case Grid.Outermost{}:
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_add_to(Shapes.checked_add(0,groups),0),Some{groups},checked_zero(groups)) :
        {Shapes.prepend_extent(_,Shapes.padded(extents,before,after)) == Some{groups <> padded} : Maybe<&2,List<&2,U32>>}
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.padded(extents,before,after),Some{padded},exact) :
        {Shapes.prepend_extent(Some{groups},_) == Some{groups <> padded} : Maybe<&2,List<&2,U32>>}
      {==}

# An axis without padding keeps its extent.
def lane_same(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,axis: Nat,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>},
  +front: {Grid.lanes_at(before,axis) == 0 : U32},+back: {Grid.lanes_at(after,axis) == 0 : U32},
  +within: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) -> {Grid.lanes_at(padded,axis) == Grid.lanes_at(extents,axis) : U32}:
  match extents before after axis:
    case Con{+extent,+rest} Con{+first,+fronts} Con{+last,+backs} 0n:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(first,extent),last),Shapes.padded(rest,fronts,backs),first,extent,last,rest,fronts,backs,padded,{==},{==},exact)
      +total = Extents.prepended_total(first,extent,last,rest,fronts,backs,padded,found)
      +widened = Extents.prepended_widened(first,extent,last,rest,fronts,backs,padded,found)
      +e = U32.to_nat(extent)
      +front_zero = Equal.cong(U32,Nat,U32.to_nat,first,0,front)
      +back_zero = Equal.cong(U32,Nat,U32.to_nat,last,0,back)
      +number = Equal.trans(Nat,U32.to_nat(total),Nat.add(Nat.add(U32.to_nat(first),e),U32.to_nat(last)),e,Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),e),U32.to_nat(last)),U32.to_nat(total),widened),
        %Equal.sym(Nat,U32.to_nat(first),0n,front_zero) : {Nat.add(Nat.add(_,e),U32.to_nat(last)) == e : Nat}
        %Equal.sym(Nat,U32.to_nat(last),0n,back_zero) : {Nat.add(e,_) == e : Nat}
        Addition.zero_right(e))
      %Extents.prepended_exact(first,extent,last,rest,fronts,backs,padded,found) : {Grid.lanes_at(_,0n) == extent : U32}
      Arithmetic.to_nat_injective(total,extent,number)
    case Con{+extent,+rest} Con{+first,+fronts} Con{+last,+backs} 1n+ +next:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(first,extent),last),Shapes.padded(rest,fronts,backs),first,extent,last,rest,fronts,backs,padded,{==},{==},exact)
      +tail = Extents.prepended_tail(first,extent,last,rest,fronts,backs,padded,found)
      %Extents.prepended_exact(first,extent,last,rest,fronts,backs,padded,found) : {Grid.lanes_at(_,1n+next) == Grid.lanes_at(rest,next) : U32}
      lane_same(rest,fronts,backs,next,tail,Extents.prepended_tail_exact(first,extent,last,rest,fronts,backs,padded,found),front,back,within)
    case Nil{} +before +after +axis:
      Empty.absurd({Grid.lanes_at(padded,axis) == Grid.lanes_at(Nil{},axis) : U32},Order.below_zero(axis,within))
    case Con{+extent,+rest} Nil{} +after +axis:
      Empty.absurd({Grid.lanes_at(padded,axis) == Grid.lanes_at(Con{extent,rest},axis) : U32},Extents.unpadded(padded,exact))
    case Con{+extent,+rest} Con{+first,+fronts} Nil{} +axis:
      Empty.absurd({Grid.lanes_at(padded,axis) == Grid.lanes_at(Con{extent,rest},axis) : U32},Extents.unpadded(padded,exact))

# The padding of the lane axis: none in front or behind when it lies inside
# the values.
def FreeLane(within: Bool,+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat) -> Data:
  match within:
    case True{}: Held.Both<{Grid.lanes_at(before,axis) == 0 : U32},{Grid.lanes_at(after,axis) == 0 : U32}>
    case False{}: Unit

def free_of(within: Bool,+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat,+front: {Grid.lanes_at(before,axis) == 0 : U32},
  +back: {Grid.lanes_at(after,axis) == 0 : U32}) -> FreeLane(within,before,after,axis):
  match within:
    case True{}: Held.Both{front,back}
    case False{}: Unit{}

def free_front(+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat,+free: FreeLane(True{},before,after,axis)) -> {Grid.lanes_at(before,axis) == 0 : U32}:
  Held.Both{+front,+back} = free
  front

def free_back(+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat,+free: FreeLane(True{},before,after,axis)) -> {Grid.lanes_at(after,axis) == 0 : U32}:
  Held.Both{+front,+back} = free
  back

def cells_cases(within: Bool,+values: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+exact: {Shapes.padded(values,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>},
  +free: FreeLane(within,before,after,axis)) ->
  {Shapes.padded(Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(within,position,Grid.removed(before,axis),0),
    Grid.placed(within,position,Grid.removed(after,axis),0)) == Some{Grid.placed(within,position,Grid.removed(padded,axis),Grid.groups(Grid.lanes_at(padded,axis)))} :
    Maybe<&2,List<&2,U32>>}:
  match within:
    case True{}:
      %Equal.sym(U32,Grid.lanes_at(padded,axis),Grid.lanes_at(values,axis),lane_same(values,before,after,axis,padded,exact,free_front(before,after,axis,free),free_back(before,after,axis,free),classified)) :
        {Shapes.padded(Grid.placed(True{},position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(True{},position,Grid.removed(before,axis),0),
          Grid.placed(True{},position,Grid.removed(after,axis),0)) == Some{Grid.placed(True{},position,Grid.removed(padded,axis),Grid.groups(_))} : Maybe<&2,List<&2,U32>>}
      padded_placed(position,Grid.removed(values,axis),Grid.removed(before,axis),Grid.removed(after,axis),Grid.removed(padded,axis),Grid.groups(Grid.lanes_at(values,axis)),
        padded_removed(values,before,after,axis,padded,exact))
    case False{}:
      padded_removed(values,before,after,axis,padded,exact)

# The cells of padded values are the cells padded on the other axes.
def cells_padded(within: Bool,+values: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+exact: {Shapes.padded(values,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>},
  +free: FreeLane(within,before,after,axis)) ->
  {Shapes.padded(Tiled.cell_shape(values,axis,position),Tiled.cell_values(before,axis,position),Tiled.cell_values(after,axis,position)) ==
    Some{Tiled.cell_shape(padded,axis,position)} : Maybe<&2,List<&2,U32>>}:
  +lengths = padded_lengths(values,before,after,padded,exact)
  +rank = List.length(&2,U32,values)
  +front_length = first_equal(List.length(&2,U32,before),rank,List.length(&2,U32,after),rank,lengths)
  +back_length = second_equal(List.length(&2,U32,before),rank,List.length(&2,U32,after),rank,lengths)
  +padded_length = Equal.sym(Nat,rank,List.length(&2,U32,padded),Facts.placed_length(padded,values,before,Extents.padded_in(values,before,after,padded,exact)))
  +front_in = {%Equal.sym(Nat,List.length(&2,U32,before),rank,front_length) : {Nat.is_lt(axis,_) == within : Bool} classified :
    {Nat.is_lt(axis,List.length(&2,U32,before)) == within : Bool}}
  +back_in = {%Equal.sym(Nat,List.length(&2,U32,after),rank,back_length) : {Nat.is_lt(axis,_) == within : Bool} classified :
    {Nat.is_lt(axis,List.length(&2,U32,after)) == within : Bool}}
  +padded_in = {%Equal.sym(Nat,List.length(&2,U32,padded),rank,padded_length) : {Nat.is_lt(axis,_) == within : Bool} classified :
    {Nat.is_lt(axis,List.length(&2,U32,padded)) == within : Bool}}
  +removed = Grid.removed(values,axis)
  +removed_before = Grid.removed(before,axis)
  +removed_after = Grid.removed(after,axis)
  +removed_padded = Grid.removed(padded,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +padded_groups = Grid.groups(Grid.lanes_at(padded,axis))
  %Equal.sym(Bool,Nat.is_lt(axis,rank),within,classified) :
    {Shapes.padded(Grid.placed(_,position,removed,groups),Grid.placed(Nat.is_lt(axis,List.length(&2,U32,before)),position,removed_before,0),
      Grid.placed(Nat.is_lt(axis,List.length(&2,U32,after)),position,removed_after,0)) == Some{Grid.placed(Nat.is_lt(axis,List.length(&2,U32,padded)),position,removed_padded,padded_groups)} :
      Maybe<&2,List<&2,U32>>}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,before)),within,front_in) :
    {Shapes.padded(Grid.placed(within,position,removed,groups),Grid.placed(_,position,removed_before,0),
      Grid.placed(Nat.is_lt(axis,List.length(&2,U32,after)),position,removed_after,0)) == Some{Grid.placed(Nat.is_lt(axis,List.length(&2,U32,padded)),position,removed_padded,padded_groups)} :
      Maybe<&2,List<&2,U32>>}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,after)),within,back_in) :
    {Shapes.padded(Grid.placed(within,position,removed,groups),Grid.placed(within,position,removed_before,0),
      Grid.placed(_,position,removed_after,0)) == Some{Grid.placed(Nat.is_lt(axis,List.length(&2,U32,padded)),position,removed_padded,padded_groups)} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,padded)),within,padded_in) :
    {Shapes.padded(Grid.placed(within,position,removed,groups),Grid.placed(within,position,removed_before,0),
      Grid.placed(within,position,removed_after,0)) == Some{Grid.placed(_,position,removed_padded,padded_groups)} : Maybe<&2,List<&2,U32>>}
  cells_cases(within,values,before,after,padded,axis,position,classified,exact,free)

def front_cases(within: Bool,+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat,+position: Grid.Position,+free: FreeLane(within,before,after,axis)) ->
  {Grid.placed(within,position,Grid.removed(before,axis),0) == Grid.placed(within,position,Grid.removed(before,axis),(Grid.lanes_at(before,axis) / 8 : U32)) : List<&2,U32>}:
  match within:
    case True{}:
      %Equal.sym(U32,Grid.lanes_at(before,axis),0,free_front(before,after,axis,free)) :
        {Grid.placed(True{},position,Grid.removed(before,axis),0) == Grid.placed(True{},position,Grid.removed(before,axis),(_ / 8 : U32)) : List<&2,U32>}
      {==}
    case False{}: {==}

# The cells shift from the front of the other axes, and not along an
# unpadded lane axis.
def front_cells(within: Bool,+before: List<&2,U32>,+after: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,before)) == within : Bool},+free: FreeLane(within,before,after,axis)) ->
  {Tiled.cell_values(before,axis,position) == Placing.cell_shifts(before,axis,position) : List<&2,U32>}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,before)),within,classified) :
    {Grid.placed(_,position,Grid.removed(before,axis),0) == Grid.placed(_,position,Grid.removed(before,axis),(Grid.lanes_at(before,axis) / 8 : U32)) : List<&2,U32>}
  front_cases(within,before,after,axis,position,free)

# ---- The lane axis padding keeps ----

def self_below(+value: Nat) -> {Nat.is_lt(value,value) == False{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+ +rest: self_below(rest)

# Where the search for an unpadded axis ends: past the axes, or at an axis
# without padding.
type Unpadded<-before: List<&2,U32>,-after: List<&2,U32>,-index: Nat,-rank: Nat,-found: Nat> is Data:
  PastAxes{same: {found == rank : Nat}}
  FreeAxis{offset: Nat,at: {found == Nat.add(index,offset) : Nat},front: {Grid.lanes_at(before,offset) == 0 : U32},back: {Grid.lanes_at(after,offset) == 0 : U32}}

def unpadded_later(+front: U32,+back: U32,+before: List<&2,U32>,+after: List<&2,U32>,+index: Nat,+rank: Nat,+found: Nat,
  +later: Unpadded<before,after,1n+index,rank,found>) -> Unpadded<Con{front,before},Con{back,after},index,rank,found>:
  match later:
    case PastAxes{+same}: PastAxes{same}
    case FreeAxis{+offset,+at,+zero_front,+zero_back}:
      FreeAxis{1n+offset,Equal.trans(Nat,found,Nat.add(1n+index,offset),Nat.add(index,1n+offset),at,Equal.sym(Nat,Nat.add(index,1n+offset),1n+Nat.add(index,offset),Addition.successor_right(index,offset))),
        zero_front,zero_back}

def unpadded_step(zero: Bool,+front: U32,+back: U32,+before: List<&2,U32>,+after: List<&2,U32>,+index: Nat,+rank: Nat,
  +classified: {U32.is_zero(front) && U32.is_zero(back) == zero : Bool},+later: Unpadded<before,after,1n+index,rank,TiledLayout.unpadded_from(before,after,1n+index,rank)>) ->
  Unpadded<Con{front,before},Con{back,after},index,rank,Bool.pick(Nat,zero,index,TiledLayout.unpadded_from(before,after,1n+index,rank))>:
  match zero:
    case True{}:
      FreeAxis{0n,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),Arithmetic.word_zero(front,Boolean.left(U32.is_zero(front),U32.is_zero(back),classified)),
        Arithmetic.word_zero(back,Boolean.right(U32.is_zero(front),U32.is_zero(back),classified))}
    case False{}: unpadded_later(front,back,before,after,index,rank,TiledLayout.unpadded_from(before,after,1n+index,rank),later)

def unpadded_found(before: List<&2,U32>,after: List<&2,U32>,+index: Nat,+rank: Nat) -> Unpadded<before,after,index,rank,TiledLayout.unpadded_from(before,after,index,rank)>:
  match before after:
    case Con{+front,+fronts} Con{+back,+backs}:
      unpadded_step(U32.is_zero(front) && U32.is_zero(back),front,back,fronts,backs,index,rank,{==},unpadded_found(fronts,backs,1n+index,rank))
    case Nil{} +after: PastAxes{{==}}
    case Con{+front,+fronts} Nil{}: PastAxes{{==}}

def unpadded_free(+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,+found: Unpadded<before,after,0n,rank,TiledLayout.unpadded_from(before,after,0n,rank)>) ->
  FreeLane(Nat.is_lt(TiledLayout.unpadded_from(before,after,0n,rank),rank),before,after,TiledLayout.unpadded_from(before,after,0n,rank)):
  match found:
    case PastAxes{+same}:
      %Equal.sym(Nat,TiledLayout.unpadded_from(before,after,0n,rank),rank,same) : FreeLane(Nat.is_lt(_,rank),before,after,_)
      %Equal.sym(Bool,Nat.is_lt(rank,rank),False{},self_below(rank)) : FreeLane(_,before,after,rank)
      Unit{}
    case FreeAxis{+offset,+at,+zero_front,+zero_back}:
      %Equal.sym(Nat,TiledLayout.unpadded_from(before,after,0n,rank),offset,at) : FreeLane(Nat.is_lt(_,rank),before,after,_)
      free_of(Nat.is_lt(offset,rank),before,after,offset,zero_front,zero_back)

# The lane axis padding keeps: the lane axis without padding, or another
# axis without padding, or the unit axis.
def kept_lane(inside: Bool,+lane: Nat,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,
  +classified: {Nat.is_lt(lane,rank) == inside : Bool},+front_length: {List.length(&2,U32,before) == rank : Nat},+back_length: {List.length(&2,U32,after) == rank : Nat},
  +zero: {U32.is_zero(Layout.extent_at(before,lane)) && U32.is_zero(Layout.extent_at(after,lane)) == True{} : Bool}) -> FreeLane(inside,before,after,lane):
  match inside:
    case True{}:
      +front_in = {%Equal.sym(Nat,List.length(&2,U32,before),rank,front_length) : {Nat.is_lt(lane,_) == True{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,before)) == True{} : Bool}}
      +back_in = {%Equal.sym(Nat,List.length(&2,U32,after),rank,back_length) : {Nat.is_lt(lane,_) == True{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,after)) == True{} : Bool}}
      Held.Both{%SetSlice.extent_lanes(before,lane,front_in) : {_ == 0 : U32}
          Arithmetic.word_zero(Layout.extent_at(before,lane),Boolean.left(U32.is_zero(Layout.extent_at(before,lane)),U32.is_zero(Layout.extent_at(after,lane)),zero)),
        %SetSlice.extent_lanes(after,lane,back_in) : {_ == 0 : U32}
          Arithmetic.word_zero(Layout.extent_at(after,lane),Boolean.right(U32.is_zero(Layout.extent_at(before,lane)),U32.is_zero(Layout.extent_at(after,lane)),zero))}
    case False{}: Unit{}

def lane_choice(chosen: Bool,+lane: Nat,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,
  +classified: {U32.is_zero(Layout.extent_at(before,lane)) && U32.is_zero(Layout.extent_at(after,lane)) == chosen : Bool},
  +front_length: {List.length(&2,U32,before) == rank : Nat},+back_length: {List.length(&2,U32,after) == rank : Nat}) ->
  FreeLane(Nat.is_lt(Bool.pick(Nat,chosen,lane,TiledLayout.unpadded_from(before,after,0n,rank)),rank),before,after,Bool.pick(Nat,chosen,lane,TiledLayout.unpadded_from(before,after,0n,rank))):
  match chosen:
    case True{}: kept_lane(Nat.is_lt(lane,rank),lane,before,after,rank,{==},front_length,back_length,classified)
    case False{}: unpadded_free(before,after,rank,unpadded_found(before,after,0n,rank))

def pad_free(+lane: Nat,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,+front_length: {List.length(&2,U32,before) == rank : Nat},
  +back_length: {List.length(&2,U32,after) == rank : Nat}) ->
  FreeLane(Nat.is_lt(TiledLayout.pad_lane(lane,before,after,rank),rank),before,after,TiledLayout.pad_lane(lane,before,after,rank)):
  lane_choice(U32.is_zero(Layout.extent_at(before,lane)) && U32.is_zero(Layout.extent_at(after,lane)),lane,before,after,rank,{==},front_length,back_length)

# ---- Padding of zeros ----

# Values padded with zeros keep their extents.
def padded_zeros(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>},
  +front_zero: {Layout.all_zero(before) == True{} : Bool},+back_zero: {Layout.all_zero(after) == True{} : Bool}) -> {padded == extents : List<&2,U32>}:
  match extents before after:
    case Nil{} Nil{} Nil{}:
      Equal.sym(List<&2,U32>,[],padded,Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => Extents.known_extents(found),Some{[]},Some{padded},exact))
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +found = Extents.prepended(Shapes.checked_add_to(Shapes.checked_add(front,extent),back),Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,{==},{==},exact)
      +total = Extents.prepended_total(front,extent,back,rest,fronts,backs,padded,found)
      +tail = Extents.prepended_tail(front,extent,back,rest,fronts,backs,padded,found)
      +widened = Extents.prepended_widened(front,extent,back,rest,fronts,backs,padded,found)
      +e = U32.to_nat(extent)
      +front_word = Arithmetic.word_zero(front,Boolean.left(U32.is_zero(front),Layout.all_zero(fronts),front_zero))
      +back_word = Arithmetic.word_zero(back,Boolean.left(U32.is_zero(back),Layout.all_zero(backs),back_zero))
      +number = Equal.trans(Nat,U32.to_nat(total),Nat.add(Nat.add(U32.to_nat(front),e),U32.to_nat(back)),e,Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(front),e),U32.to_nat(back)),U32.to_nat(total),widened),
        %Equal.sym(Nat,U32.to_nat(front),0n,Equal.cong(U32,Nat,U32.to_nat,front,0,front_word)) : {Nat.add(Nat.add(_,e),U32.to_nat(back)) == e : Nat}
        %Equal.sym(Nat,U32.to_nat(back),0n,Equal.cong(U32,Nat,U32.to_nat,back,0,back_word)) : {Nat.add(e,_) == e : Nat}
        Addition.zero_right(e))
      %Extents.prepended_exact(front,extent,back,rest,fronts,backs,padded,found) : {_ == Con{extent,rest} : List<&2,U32>}
      Equal.trans(List<&2,U32>,{total <> tail : List<&2,U32>},{extent <> tail : List<&2,U32>},{extent <> rest : List<&2,U32>},
        Equal.cong(U32,List<&2,U32>,head => {head <> tail : List<&2,U32>},total,extent,Arithmetic.to_nat_injective(total,extent,number)),
        Equal.cong(List<&2,U32>,List<&2,U32>,later => {extent <> later : List<&2,U32>},tail,rest,
          padded_zeros(rest,fronts,backs,tail,Extents.prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found),
            Boolean.right(U32.is_zero(front),Layout.all_zero(fronts),front_zero),Boolean.right(U32.is_zero(back),Layout.all_zero(backs),back_zero))))
    case Con{+extent,+rest} Nil{} +after: Empty.absurd({padded == Con{extent,rest} : List<&2,U32>},Extents.unpadded(padded,exact))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{}: Empty.absurd({padded == Con{extent,rest} : List<&2,U32>},Extents.unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after: Empty.absurd({padded == Nil{} : List<&2,U32>},Extents.unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs}: Empty.absurd({padded == Nil{} : List<&2,U32>},Extents.unpadded(padded,exact))

# Zero shifts hold digits below a source, and keep them.
def zero_inside(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+zero: {Layout.all_zero(shifts) == True{} : Bool},
  +below: Coordinates.DigitsBelow(source,digits)) -> {Placing.inside(source,shifts,digits) == True{} : Bool}:
  match source shifts digits:
    case Con{+extent,+rest} Con{+shift,+later} Con{+digit,+more}:
      KernelGemm.Both{+small,+others} = below
      +shift_zero = Equal.cong(U32,Nat,U32.to_nat,shift,0,Arithmetic.word_zero(shift,Boolean.left(U32.is_zero(shift),Layout.all_zero(later),zero)))
      Boolean.and_true(Placing.axis_inside(extent,shift,digit),Placing.inside(rest,later,more),
        %Equal.sym(Nat,U32.to_nat(shift),0n,shift_zero) : {Nat.is_le(_,digit) && Nat.is_lt(digit,Nat.add(_,U32.to_nat(extent))) == True{} : Bool}
        Boolean.and_true(Nat.is_le(0n,digit),Nat.is_lt(digit,U32.to_nat(extent)),Order.zero_le(digit),small),
        zero_inside(rest,later,more,Boolean.right(U32.is_zero(shift),Layout.all_zero(later),zero),others))
    case Nil{} +shifts +digits: {==}
    case Con{+extent,+rest} Nil{} +digits: {==}
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}

def zero_shifted(shifts: List<&2,U32>,digits: List<&2,Nat>,+zero: {Layout.all_zero(shifts) == True{} : Bool},
  +count: {List.length(&2,Nat,digits) == List.length(&2,U32,shifts) : Nat}) -> {Placing.shifted(shifts,digits) == digits : List<&2,Nat>}:
  match shifts digits:
    case Con{+shift,+later} Con{+digit,+more}:
      +shift_zero = Equal.cong(U32,Nat,U32.to_nat,shift,0,Arithmetic.word_zero(shift,Boolean.left(U32.is_zero(shift),Layout.all_zero(later),zero)))
      %Equal.sym(Nat,U32.to_nat(shift),0n,shift_zero) : {Con{Nat.sub(digit,_),Placing.shifted(later,more)} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(Nat,Nat.sub(digit,0n),digit,Order.subtract_zero(digit)) : {Con{_,Placing.shifted(later,more)} == Con{digit,more} : List<&2,Nat>}
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => {Con{digit,tail} : List<&2,Nat>},Placing.shifted(later,more),more,
        zero_shifted(later,more,Boolean.right(U32.is_zero(shift),Layout.all_zero(later),zero),Placing.pred_same(List.length(&2,Nat,more),List.length(&2,U32,later),count)))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}:
      Empty.absurd({Placing.shifted(Nil{},Con{digit,more}) == Con{digit,more} : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,count)))
    case Con{+shift,+later} Nil{}: {==}

# A source and its shifts within a room have as many axes.
def room_length(source: List<&2,U32>,shifts: List<&2,U32>,+cap: U32,+room: Placing.Room(source,shifts,cap)) -> {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat}:
  match source shifts:
    case Con{+extent,+rest} Con{+shift,+later}:
      KernelGemm.Both{+fit,+others} = room
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,later),List.length(&2,U32,rest),room_length(rest,later,cap,others))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+shift,+later}: Empty.absurd({List.length(&2,U32,Con{shift,later}) == List.length(&2,U32,Nil{}) : Nat},room)
    case Con{+extent,+rest} Nil{}: Empty.absurd({List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{extent,rest}) : Nat},room)

# A source placed in itself with zero shifts reads its values where they lie.
def zero_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: List<&2,U32>,+shifts: List<&2,U32>,+index: U32,+old: Element,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: Placing.Room(source,shifts,cap),+zero: {Layout.all_zero(shifts) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(source)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(source)) == True{} : Bool}) ->
  {Gathered.sources_value(~Element,~Logical,~original,[logical],[Layout.placed_segments(source,shifts)],[source],Grid.coordinates(source,index),old) == original(logical,index) : Element}:
  +words = Grid.coordinates(source,index)
  +digits = Targets.cell_coordinates(source,U32.to_nat(index))
  +words_exact = Digits.coordinates_exact(source,index,cap,cap_ok,cells_room,inside)
  +below = Digits.coordinates_below(source,U32.to_nat(index),inside)
  +count = Digits.coordinates_count(source,index)
  +inside_digits = zero_inside(source,shifts,digits,zero,below)
  +inside_words = {%Equal.sym(List<&2,Nat>,Digits.words_nat(words),digits,words_exact) : {Placing.inside(source,shifts,_) == True{} : Bool} inside_digits :
    {Placing.inside(source,shifts,Digits.words_nat(words)) == True{} : Bool}}
  +found = Equal.trans(Bool,Placing.inside_words(source,shifts,words),Placing.inside(source,shifts,Digits.words_nat(words)),True{},Placing.inside_nat(source,shifts,words,cap,cap_ok,room),inside_words)
  +fits = Order.transitive(ShapeLaws.elements(source),Nat.double(ShapeLaws.elements(source)),U32.to_nat(cap),Order.le_double(ShapeLaws.elements(source)),cells_room)
  +shifts_length = room_length(source,shifts,cap,room)
  +digits_count = Equal.trans(Nat,List.length(&2,Nat,digits),List.length(&2,U32,source),List.length(&2,U32,shifts),Digits.coordinates_length(source,U32.to_nat(index)),
    Equal.sym(Nat,List.length(&2,U32,shifts),List.length(&2,U32,source),shifts_length))
  +offset = Placing.offset(source,shifts,words)
  +offset_number = Equal.trans(Nat,U32.to_nat(offset),ShapeLaws.digits_value(source,Placing.shifted(shifts,Digits.words_nat(words))),U32.to_nat(index),
    Placing.offset_nat(source,shifts,words,cap,cap_ok,room,count,fits,inside_words),
    %Equal.sym(List<&2,Nat>,Digits.words_nat(words),digits,words_exact) : {ShapeLaws.digits_value(source,Placing.shifted(shifts,_)) == U32.to_nat(index) : Nat}
    %Equal.sym(List<&2,Nat>,Placing.shifted(shifts,digits),digits,zero_shifted(shifts,digits,zero,digits_count)) : {ShapeLaws.digits_value(source,_) == U32.to_nat(index) : Nat}
    Digits.value_coordinates(source,U32.to_nat(index),inside))
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(source,shifts),words,Tasks.strides_of(source)),
      Bool.pick(Maybe<&2,U32>,Placing.inside_words(source,shifts,words),Some{offset},None{}),Placing.segments_words(source,shifts,words)) :
    {Placements.placed_result(~Element,~Logical,~original,logical,_,old) == original(logical,index) : Element}
  %Equal.sym(Bool,Placing.inside_words(source,shifts,words),True{},found) :
    {Placements.placed_result(~Element,~Logical,~original,logical,Bool.pick(Maybe<&2,U32>,_,Some{offset},None{}),old) == original(logical,index) : Element}
  %Equal.sym(U32,offset,index,Arithmetic.to_nat_injective(offset,index,offset_number)) : {original(logical,_) == original(logical,index) : Element}
  {==}

# Zero shifts clamp words below a source to themselves.
def clamp_zero(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Placing.Room(source,shifts,cap),+zero: {Layout.all_zero(shifts) == True{} : Bool},+below: Coordinates.DigitsBelow(source,Digits.words_nat(words))) ->
  {clamp_words(source,shifts,words) == words : List<&2,U32>}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      KernelGemm.Both{+fit,+others} = room
      KernelGemm.Both{+small,+belows} = below
      +shift_word = Arithmetic.word_zero(shift,Boolean.left(U32.is_zero(shift),Layout.all_zero(later),zero))
      +shift_zero = Equal.cong(U32,Nat,U32.to_nat,shift,0,shift_word)
      +positive = Order.le_lt_trans(0n,U32.to_nat(word),U32.to_nat(extent),Order.zero_le(U32.to_nat(word)),small)
      +clamped = Tasks.clamp(word,shift,(shift + extent - 1 : U32))
      +number = Equal.trans(Nat,U32.to_nat(clamped),Reads.clamped_index(U32.to_nat(shift),U32.to_nat(extent),U32.to_nat(word)),U32.to_nat(word),
        Reads.clamp_exact(word,shift,extent,cap,cap_ok,fit,positive),
        clamp_unmoved(shift,extent,U32.to_nat(word),shift_zero,small))
      Equal.trans(List<&2,U32>,{clamped <> clamp_words(rest,later,more) : List<&2,U32>},{word <> clamp_words(rest,later,more) : List<&2,U32>},{word <> more : List<&2,U32>},
        Equal.cong(U32,List<&2,U32>,head => {head <> clamp_words(rest,later,more) : List<&2,U32>},clamped,word,Arithmetic.to_nat_injective(clamped,word,number)),
        Equal.cong(List<&2,U32>,List<&2,U32>,tail => {word <> tail : List<&2,U32>},clamp_words(rest,later,more),more,
          clamp_zero(rest,later,more,cap,cap_ok,others,Boolean.right(U32.is_zero(shift),Layout.all_zero(later),zero),belows)))
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}
    case Nil{} +shifts Nil{}: {==}
    case Nil{} +shifts Con{+word,+more}: Empty.absurd({clamp_words(Nil{},shifts,Con{word,more}) == Con{word,more} : List<&2,U32>},below)
    case Con{+extent,+rest} Nil{} +words: Empty.absurd({clamp_words(Con{extent,rest},Nil{},words) == words : List<&2,U32>},room)

# A lane kept from a value kept on its own lane is the value.
def stable_lane(choice: Bool,+value: F32,+target: F32,+kept: {Bool.pick(F32,choice,Bool.pick(F32,choice,value,0.0),0.0) == target : F32}) ->
  {Bool.pick(F32,choice,value,0.0) == target : F32}:
  match choice:
    case True{}: kept
    case False{}: kept

# A lifted cell that agrees with a target below its lane count is the target.
def stable_valid(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+word: U32,+target: Tile.Tile8,
  +valid: PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word),target>) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word) == target : Tile.Tile8}:
  PaddedCells.Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = valid
  +number = U32.to_nat(word)
  +outer = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  TileLaws.tiles_equal(Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word),target,
    stable_lane(Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(0n,Nat.mul(group,8n))))),Tile.lane0(target),lane0),
    stable_lane(Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(1n,Nat.mul(group,8n))))),Tile.lane1(target),lane1),
    stable_lane(Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(2n,Nat.mul(group,8n))))),Tile.lane2(target),lane2),
    stable_lane(Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(3n,Nat.mul(group,8n))))),Tile.lane3(target),lane3),
    stable_lane(Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(4n,Nat.mul(group,8n))))),Tile.lane4(target),lane4),
    stable_lane(Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(5n,Nat.mul(group,8n))))),Tile.lane5(target),lane5),
    stable_lane(Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(6n,Nat.mul(group,8n))))),Tile.lane6(target),lane6),
    stable_lane(Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(7n,Nat.mul(group,8n))))),Tile.lane7(target),lane7))

# ---- The padded tiled values ----

# What padded values read: the values placed from their front padding in the
# padded extents, the new elements holding the value or the nearest element
# at an edge.
def pad_values(~Logical: Data,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,+before: List<&2,U32>) -> Gathered.Placings<Logical,F32>:
  match border:
    case Layout.Constant{+value}: Gathered.Placings{[logical],[Layout.placed_segments(values,before)],[values],target,value}
    case Layout.Edge{}: Gathered.Placings{[logical],[Layout.edge_segments(target,values,before)],[values],target,0.0}

# ---- Cells placed, then padded ----

# One source fits half a word.
def one_room(+limits: Machine.Limits) -> {Nat.is_le(1n,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(1n,16n,U32.to_nat(Machine.half_word(limits)),{==},Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)))

# One source laid as the canonical pieces of its shape, holding the values.
def one_placed(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logical: Logical,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,
  +held: Sources.SourceHeld(~Tile.Tile8,~Logical,~original,logical,model,shape)) ->
  PlacingMoved.SourcesPlaced(~Tile.Tile8,~Logical,~original,[logical],[shape],Moved.Models{model,Moved.NoModels{}}):
  Held.Both{+laid,+holds} = held
  Held.Both{laid,Held.Both{holds,Unit{}}}

# Cells of the contract placed from the front padding in the padded cells,
# then padded, hold the padded values when every cell agrees with them.
def placed_case(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+spec: Gathered.Placings<Logical,F32>,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,+segment: List<&2,List<&2,Tasks.Segment>>,+clamped: Bool,+fill_cell: Tile.Tile8,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers}),+cells_exact: {Shapes.padded(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>},
  +axes: Parts.SourceAxes(segment,Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),clamped,Machine.half_word(limits)),
  +laid: Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>,+arranged: PaddedCells.Placed<Tiled.cell_shape(target,lane,position),target,lane,lanes,position,Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(target,lane,position)))>,
  +cells_counted: Global.Counted<Tiled.cell_shape(target,lane,position)>,
  +room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},
  agree: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool} ->
    PaddedCells.Valid<Representation.lane_count(Public.geometry_of(target,lane,position)),Representation.cell_group(Public.geometry_of(target,lane,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[segment],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(target,lane,position)},word)>) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),spec,target,Tiled.Tiled{Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position}):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = arranged
  +within = Extents.padded_in(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),Tiled.cell_shape(target,lane,position),cells_exact)
  +source_rank = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,U32,Tiled.cell_shape(target,lane,position)),Facts.placed_length(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    rank_room : {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +target_ok = {Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,Tiled.cell_shape(target,lane,position))}
  +source_ok = {Held.Both{source_room,Held.Both{source_positive,source_rank}} : Components.ShapeOk(limits,Tiled.cell_shape(values,lane,position))}
  +shaped = {Held.Both{source_ok,Held.Both{axes,Unit{}}} : Facts.SourcesShaped(limits,Tiled.cell_shape(target,lane,position),[segment],[Tiled.cell_shape(values,lane,position)],[Placing.cell_shifts(before,lane,position)],[clamped])}
  +facts = Facts.placing_facts(limits,Tiled.cell_shape(target,lane,position),[segment],[Tiled.cell_shape(values,lane,position)],[Placing.cell_shifts(before,lane,position)],[clamped],target_ok,shaped,{==},one_room(limits))
  +padding_eq = Equal.trans(TiledMap.Padding,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),TiledMap.padding_of(Tiled.cell_shape(target,lane,position),lane,lanes,position),PaddedCells.layout_padding(layout),
    Equal.cong(List<&2,U32>,TiledMap.Padding,cells => TiledMap.padding_of(cells,lane,lanes,position),TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),Tiled.cell_shape(target,lane,position),
      padded_grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),Tiled.cell_shape(target,lane,position),cells_exact)),
    Equal.sym(TiledMap.Padding,PaddedCells.layout_padding(layout),TiledMap.padding_of(Tiled.cell_shape(target,lane,position),lane,lanes,position),padding_exact))
  +padding_back = Equal.sym(TiledMap.Padding,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),PaddedCells.layout_padding(layout),padding_eq)
  +cap = Machine.half_word(PaddedPieces.layout_limits(layout))
  +total_positive = {%Equal.sym(Nat,PaddedCells.layout_total(layout),ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),total_exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,PaddedCells.layout_total(layout)) == True{} : Bool}}
  +fit = {%padding_back : {PaddedPieces.groups_fit(cap,_) == True{} : Bool} PaddedPieces.layout_fit(layout,total_positive) : {PaddedPieces.groups_fit(cap,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position)) == True{} : Bool}}
  +total = {%padding_back : {PaddedPieces.padded_cells(_) == ShapeLaws.elements(Tiled.cell_shape(target,lane,position)) : Nat}
    Equal.trans(Nat,PaddedPieces.padded_cells(PaddedCells.layout_padding(layout)),PaddedCells.layout_total(layout),ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),PaddedPieces.layout_cells(layout),total_exact) :
    {PaddedPieces.padded_cells(TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position)) == ShapeLaws.elements(Tiled.cell_shape(target,lane,position)) : Nat}}
  +contract = padded_contract(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Placing.lifted_logicals(~Logical,[logical],[values],lane,position),limits,Tiled.cell_shape(target,lane,position),[segment],[Tiled.cell_shape(values,lane,position)],[Placing.cell_shifts(before,lane,position)],[clamped],fill_cell,model_fill,model_workers,
    Moved.Models{model,Moved.NoModels{}},{==},facts,
    one_placed(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},model,Tiled.cell_shape(values,lane,position),Sources.source_held(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},model,Tiled.cell_shape(values,lane,position),model_fill,model_workers,global)),
    TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),cap,Limits.half_word_fits(PaddedPieces.layout_limits(layout)),fit,total,Global.count(Tiled.cell_shape(target,lane,position),cells_counted),Global.accepted(Tiled.cell_shape(target,lane,position),cells_counted))
  +shaped_cells = Global.shaped(~Tile.Tile8,~PaddedPieces.Padded<Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>>,~PaddedPieces.padded_value(~Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),PaddedPieces.Padded{TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[segment],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell}},Tiled.cell_shape(target,lane,position),Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers},{==},{==},contract)
  +moved = {%padding_eq : Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>>,~PaddedPieces.padded_value(~Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),PaddedPieces.Padded{_,Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[segment],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell}},Tiled.cell_shape(target,lane,position),Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers}) shaped_cells :
    Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>>,~PaddedPieces.padded_value(~Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),PaddedPieces.Padded{PaddedCells.layout_padding(layout),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[segment],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell}},Tiled.cell_shape(target,lane,position),Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers})}
  +restated = PaddedPieces.padded_restated(~Gathered.Placings<Representation.Lifted<Logical>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Representation.Lifted<Gathered.Placings<Logical,F32>>,~Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original)),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[segment],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell},Representation.Lifted{spec,Public.geometry_of(target,lane,position)},layout,Tiled.cell_shape(target,lane,position),Public.geometry_of(target,lane,position),geometry_exact,total_exact,agree,
    Machine.capacity(limits),room,Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers},moved)
  Public.laid_contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),spec,target,Tensors.Tensor{Tensors.placed_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),[Layout.placed_readings(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tensors.grid_of(Tiled.cell_shape(target,lane,position)),Tensors.grid_of(Tiled.cell_shape(values,lane,position)),clamped,0)],[segment],Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,lane,position),0)],Tiled.cell_shape(target,lane,position),fill_cell,model_workers),Tiled.cell_shape(target,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position,KernelGemm.Both{restated,laid})

# ---- Cells padding leaves as they are ----

# A cell that reads the values and agrees with the padded values of the same
# extents holds them.
def kept_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+spec: Gathered.Placings<Logical,F32>,+values: List<&2,U32>,+target: List<&2,U32>,+lane: Nat,
  +position: Grid.Position,+word: U32,+cell: Tile.Tile8,+read: {cell == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8},
  +same: {target == values : List<&2,U32>},
  +valid: PaddedCells.Valid<Representation.lane_count(Public.geometry_of(target,lane,position)),Representation.cell_group(Public.geometry_of(target,lane,position),U32.to_nat(word)),cell,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(target,lane,position)},word)>) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) == Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}:
  stable_valid(~Logical,~original,logical,Public.geometry_of(values,lane,position),word,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(values,lane,position)},word),
    %same : PaddedCells.Valid<Representation.lane_count(Public.geometry_of(_,lane,position)),Representation.cell_group(Public.geometry_of(_,lane,position),U32.to_nat(word)),
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(_,lane,position)},word)>
    %read : PaddedCells.Valid<Representation.lane_count(Public.geometry_of(target,lane,position)),Representation.cell_group(Public.geometry_of(target,lane,position),U32.to_nat(word)),_,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(target,lane,position)},word)>
    valid)

# A cell placed from no padding in cells of the same extents reads the
# values.
def constant_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,+before: List<&2,U32>,+lane: Nat,
  +position: Grid.Position,+limits: Machine.Limits,+fill_cell: Tile.Tile8,+word: U32,+same: {target == values : List<&2,U32>},
  +room: Placing.Room(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Machine.half_word(limits)),+zero: {Layout.all_zero(Placing.cell_shifts(before,lane,position)) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell},word) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}:
  %Equal.sym(List<&2,U32>,target,values,same) :
    {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(_,lane,position),fill_cell},word) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}
  zero_read(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),word,fill_cell,Machine.half_word(limits),Limits.half_word_fits(limits),room,zero,cells_room,inside)

# A cell of an edge from no padding in cells of the same extents reads the
# values.
def edge_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,+before: List<&2,U32>,+lane: Nat,
  +position: Grid.Position,+limits: Machine.Limits,+fill_cell: Tile.Tile8,+word: U32,+same: {target == values : List<&2,U32>},
  +within: Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},
  +room: Placing.Room(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Machine.half_word(limits)),+zero: {Layout.all_zero(Placing.cell_shifts(before,lane,position)) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.edge_segments(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell},word) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}:
  +words = Grid.coordinates(Tiled.cell_shape(values,lane,position),word)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +below = {%Equal.sym(List<&2,Nat>,Digits.words_nat(words),Targets.cell_coordinates(Tiled.cell_shape(values,lane,position),U32.to_nat(word)),Digits.coordinates_exact(Tiled.cell_shape(values,lane,position),word,cap,cap_ok,cells_room,inside)) :
      Coordinates.DigitsBelow(Tiled.cell_shape(values,lane,position),_)
    Digits.coordinates_below(Tiled.cell_shape(values,lane,position),U32.to_nat(word),inside) : Coordinates.DigitsBelow(Tiled.cell_shape(values,lane,position),Digits.words_nat(words))}
  %Equal.sym(List<&2,U32>,target,values,same) :
    {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.edge_segments(Tiled.cell_shape(_,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(_,lane,position),fill_cell},word) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),[Representation.Lifted{logical,Public.geometry_of(values,lane,position)}],[Layout.edge_segments(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],words,fill_cell),
    Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),[Representation.Lifted{logical,Public.geometry_of(values,lane,position)}],[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],clamp_words(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),words),fill_cell),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word),
    edge_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),words,fill_cell,edge_cell(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),words,cap,cap_ok,within,room,positive,below)),
    %Equal.sym(List<&2,U32>,clamp_words(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),words),words,clamp_zero(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),words,cap,cap_ok,room,zero,below)) :
      {Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),[Representation.Lifted{logical,Public.geometry_of(values,lane,position)}],[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],_,fill_cell) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}
    zero_read(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),word,fill_cell,cap,cap_ok,room,zero,cells_room,inside))

# Cells padding leaves as they are hold the padded values of the same
# extents when every cell of the values agrees with them.
def unchanged_case(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+spec: Gathered.Placings<Logical,F32>,+values: List<&2,U32>,+target: List<&2,U32>,+lane: Nat,+lanes: U32,
  +position: Grid.Position,+limits: Machine.Limits,+model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers}),+laid: Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>,+same: {target == values : List<&2,U32>},
  +room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  agree: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool} ->
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},word) == Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{spec,Public.geometry_of(values,lane,position)},word) : Tile.Tile8}) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),spec,target,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position}):
  %Equal.sym(List<&2,U32>,target,values,same) : Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),spec,_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position})
  Public.laid_contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),spec,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position,
    KernelGemm.Both{Global.restated_below(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.Lifted<Gathered.Placings<Logical,F32>>,~Representation.lifted(~Logical,~original),~Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original)),~ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Representation.Lifted{spec,Public.geometry_of(values,lane,position)},agree,Tiled.cell_shape(values,lane,position),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},
      KernelGemm.Both{Global.usable(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},{==}),global},Order.reflexive(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),Machine.capacity(limits),room),laid})

# Padded values of the extents of the cells of the values are the values.
def target_same(+values: List<&2,U32>,+target: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,
  +laid: Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>,+laid_target: Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>,+cells_same: {Tiled.cell_shape(target,lane,position) == Tiled.cell_shape(values,lane,position) : List<&2,U32>}) ->
  {target == values : List<&2,U32>}:
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  Public.Laid{+target_inside,+target_shaped,+target_restored,+target_counted,+target_pinned} = laid_target
  Equal.trans(List<&2,U32>,target,Tiled.logical_shape(Tiled.cell_shape(target,lane,position),lane,lanes,position),values,Equal.sym(List<&2,U32>,Tiled.logical_shape(Tiled.cell_shape(target,lane,position),lane,lanes,position),target,target_restored),
    %Equal.sym(List<&2,U32>,Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),cells_same) : {Tiled.logical_shape(_,lane,lanes,position) == values : List<&2,U32>} restored)

# ---- A cell of the padded values ----

# A source placed from the front of the lane axis of a target of the same
# lanes starts every group and ends the lane axis.
def front_lane(+values: List<&2,U32>,+target: List<&2,U32>,+before: List<&2,U32>,+lane: Nat,+front: {Grid.lanes_at(before,lane) == 0 : U32},
  +same: {Grid.lanes_at(values,lane) == Grid.lanes_at(target,lane) : U32}) ->
  KernelGemm.Both<{U32.is_zero((Grid.lanes_at(before,lane) % 8 : U32)) == True{} : Bool},
    Placing.Ending<Grid.lanes_at(values,lane),Grid.lanes_at(before,lane),Grid.lanes_at(target,lane)>>:
  %Equal.sym(U32,Grid.lanes_at(before,lane),0,front) :
    KernelGemm.Both<{U32.is_zero((_ % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(values,lane),_,Grid.lanes_at(target,lane)>>
  %Equal.sym(U32,Grid.lanes_at(values,lane),Grid.lanes_at(target,lane),same) :
    KernelGemm.Both<{U32.is_zero((0 % 8 : U32)) == True{} : Bool},Placing.Ending<_,0,Grid.lanes_at(target,lane)>>
  KernelGemm.Both{{==},Placing.Last{{==}}}

# Padding keeps the lanes of the axis it leaves whole or of the unit axis.
def lanes_kept(within: Bool,+values: List<&2,U32>,+target: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},
  +ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+free: FreeLane(within,before,after,lane)) ->
  {Grid.lanes_at(target,lane) == Grid.lanes_at(values,lane) : U32}:
  match within:
    case True{}: lane_same(values,before,after,lane,target,exact,free_front(before,after,lane,free),free_back(before,after,lane,free),classified)
    case False{}:
      +past = {%ranked : {Nat.is_lt(lane,_) == False{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,target)) == False{} : Bool}}
      Equal.trans(U32,Grid.lanes_at(target,lane),1,Grid.lanes_at(values,lane),Public.lanes_past(target,lane,past),
        Equal.sym(U32,Grid.lanes_at(values,lane),1,Public.lanes_past(values,lane,classified)))

# A cell of padded cells of a value agrees with the padded values.
def constant_agree(~Logical: Data,~original: Logical -> U32 -> F32,within: Bool,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+position: Grid.Position,+limits: Machine.Limits,+value: F32,+exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),+placed: Facts.PlacedIn(target,values,before),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+shifts_same: {List.length(&2,U32,before) == List.length(&2,U32,values) : Nat},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},+source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},+word: U32,+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool}) -> PaddedCells.Valid<Representation.lane_count(Public.geometry_of(target,lane,position)),Representation.cell_group(Public.geometry_of(target,lane,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),Bool.pick(Tile.Tile8,Bool.not(within),Tiled.unit_tile(value),Tile.filled(value))},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{pad_values(~Logical,Layout.Constant{value},logical,values,target,before),Public.geometry_of(target,lane,position)},word)>:
  match within:
    case True{}:
      +inside_target = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} inside : {Nat.is_le(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +within_target = {%ranked : {Nat.is_lt(lane,_) == True{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +same = Equal.sym(U32,Grid.lanes_at(target,lane),Grid.lanes_at(values,lane),
        lane_same(values,before,after,lane,target,exact,free_front(before,after,lane,free),free_back(before,after,lane,free),classified))
      +facts = {KernelGemm.Both{SetSlice.source_lanes(limits,target,values,before,lane,position,within_target,classified,ranked,shifts_same,placed,cell_placed,room,positive,
          cells_room,cells_positive,source_room,source_positive,source_cells_room,front_lane(values,target,before,lane,free_front(before,after,lane,free),same)),Unit{}} :
        Placing.SourcesLanes(limits,Machine.half_word(limits),Grid.lanes_at(target,lane),List.length(&2,U32,target),List.length(&2,U32,Tiled.cell_shape(target,lane,position)),[values],[before],lane,position)}
      +goal = {Placing.Target{Limits.half_word_fits(limits),inside_target,Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),cells_room),
          Limits.double_bound(limits,ShapeLaws.elements(target),room),Rooms.removed_room(limits,target,lane,within_target,positive,room)} :
        Placing.Target<limits,Machine.half_word(limits),target,lane,position>}
      constant_lanes(~Logical,~original,[logical],[values],[before],limits,Machine.half_word(limits),target,lane,position,value,word,goal,within_target,{==},facts,inside_word)
    case False{}:
      +inside_target = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} inside : {Nat.is_le(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +past_target = {%ranked : {Nat.is_lt(lane,_) == False{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,target)) == False{} : Bool}}
      +facts = {KernelGemm.Both{SetSlice.source_units(limits,target,values,before,lane,classified,inside,ranked,shifts_same,placed,room,positive,source_room),Unit{}} :
        Placing.SourcesUnits(limits,Machine.half_word(limits),List.length(&2,U32,target),[values],[before],lane)}
      +goal = {Placing.Target{Limits.half_word_fits(limits),inside_target,Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),cells_room),
          Limits.double_bound(limits,ShapeLaws.elements(target),room),
          {%Equal.sym(List<&2,U32>,Grid.removed(target,lane),target,Public.removed_past(target,lane,past_target)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            room : {Nat.is_le(ShapeLaws.elements(Grid.removed(target,lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}} :
        Placing.Target<limits,Machine.half_word(limits),target,lane,position>}
      constant_unit(~Logical,~original,[logical],[values],[before],limits,Machine.half_word(limits),target,lane,position,value,word,goal,past_target,{==},facts,inside_word)

# A cell of padded cells of an edge agrees with the padded values.
def edge_agree(~Logical: Data,~original: Logical -> U32 -> F32,within: Bool,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+position: Grid.Position,+limits: Machine.Limits,+fill_cell: Tile.Tile8,+exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),+placed: Facts.PlacedIn(target,values,before),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+shifts_same: {List.length(&2,U32,before) == List.length(&2,U32,values) : Nat},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},+source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},+word: U32,+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool}) -> PaddedCells.Valid<Representation.lane_count(Public.geometry_of(target,lane,position)),Representation.cell_group(Public.geometry_of(target,lane,position),U32.to_nat(word)),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.edge_segments(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{pad_values(~Logical,Layout.Edge{},logical,values,target,before),Public.geometry_of(target,lane,position)},word)>:
  match within:
    case True{}:
      +inside_target = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} inside : {Nat.is_le(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +within_target = {%ranked : {Nat.is_lt(lane,_) == True{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +same = Equal.sym(U32,Grid.lanes_at(target,lane),Grid.lanes_at(values,lane),
        lane_same(values,before,after,lane,target,exact,free_front(before,after,lane,free),free_back(before,after,lane,free),classified))
      +facts = {KernelGemm.Both{SetSlice.source_lanes(limits,target,values,before,lane,position,within_target,classified,ranked,shifts_same,placed,cell_placed,room,positive,
          cells_room,cells_positive,source_room,source_positive,source_cells_room,front_lane(values,target,before,lane,free_front(before,after,lane,free),same)),Unit{}} :
        Placing.SourcesLanes(limits,Machine.half_word(limits),Grid.lanes_at(target,lane),List.length(&2,U32,target),List.length(&2,U32,Tiled.cell_shape(target,lane,position)),[values],[before],lane,position)}
      +goal = {Placing.Target{Limits.half_word_fits(limits),inside_target,Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),cells_room),
          Limits.double_bound(limits,ShapeLaws.elements(target),room),Rooms.removed_room(limits,target,lane,within_target,positive,room)} :
        Placing.Target<limits,Machine.half_word(limits),target,lane,position>}
      edge_lanes(~Logical,~original,logical,values,before,limits,Machine.half_word(limits),target,lane,position,fill_cell,word,goal,within_target,facts,placed,cell_placed,
        source_positive,source_cells_positive,free_front(before,after,lane,free),same,inside_word)
    case False{}:
      +inside_target = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} inside : {Nat.is_le(lane,List.length(&2,U32,target)) == True{} : Bool}}
      +past_target = {%ranked : {Nat.is_lt(lane,_) == False{} : Bool} classified : {Nat.is_lt(lane,List.length(&2,U32,target)) == False{} : Bool}}
      +facts = {KernelGemm.Both{SetSlice.source_units(limits,target,values,before,lane,classified,inside,ranked,shifts_same,placed,room,positive,source_room),Unit{}} :
        Placing.SourcesUnits(limits,Machine.half_word(limits),List.length(&2,U32,target),[values],[before],lane)}
      +goal = {Placing.Target{Limits.half_word_fits(limits),inside_target,Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),cells_room),
          Limits.double_bound(limits,ShapeLaws.elements(target),room),
          {%Equal.sym(List<&2,U32>,Grid.removed(target,lane),target,Public.removed_past(target,lane,past_target)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            room : {Nat.is_le(ShapeLaws.elements(Grid.removed(target,lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}} :
        Placing.Target<limits,Machine.half_word(limits),target,lane,position>}
      edge_unit(~Logical,~original,logical,values,before,limits,Machine.half_word(limits),target,lane,position,fill_cell,word,goal,past_target,facts,placed,cell_placed,
        source_positive,source_cells_positive,inside_word)

# ---- The padded cells ----

# Extents without a zero extent have an element to repeat at every edge.
def edges_held(shape: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Layout.edge_axes(shape,before,after) == True{} : Bool}:
  match shape before after:
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +positive = Equal.trans(Bool,U32.is_lt(0,extent),Nat.is_lt(U32.to_nat(0),U32.to_nat(extent)),True{},Comparison.lt(0,extent),
        Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)))
      %Equal.sym(Bool,U32.is_lt(0,extent),True{},positive) :
        {(_ || (U32.is_zero(front) && U32.is_zero(back))) && Layout.edge_axes(rest,fronts,backs) == True{} : Bool}
      edges_held(rest,fronts,backs,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{}: {==}
    case Con{+extent,+rest} Nil{} +after: {==}
    case Nil{} +before +after: {==}

# Cells of a value padded, or left as they are.
def constant_change(~Logical: Data,~original: Logical -> U32 -> F32,unchanged: Bool,+logical: Logical,+value: F32,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,+within: Bool,
  +changed: {Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)) == unchanged : Bool},
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),+global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers}),+laid: Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>,+laid_target: Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>,+exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+cells_exact: {Shapes.padded(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>},+placed: Facts.PlacedIn(target,values,before),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+cells_counted: Global.Counted<Tiled.cell_shape(target,lane,position)>,+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+shifts_same: {List.length(&2,U32,before) == List.length(&2,U32,values) : Nat},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},+source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},+cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,Layout.Constant{value},logical,values,target,before),target,
    Tiled.Tiled{Layout.pad_valid(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),Layout.Constant{Bool.pick(Tile.Tile8,Bool.not(within),Tiled.unit_tile(value),Tile.filled(value))},True{},unchanged,PiecesModel.pack(~Tile.Tile8,model),
      Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_shape(target,lane,position),model_fill,model_workers),lane,lanes,position}):
  match unchanged:
    case True{}:
      +front_zero = Boolean.left(Layout.all_zero(Placing.cell_shifts(before,lane,position)),Layout.all_zero(Tiled.cell_values(after,lane,position)),changed)
      +back_zero = Boolean.right(Layout.all_zero(Placing.cell_shifts(before,lane,position)),Layout.all_zero(Tiled.cell_values(after,lane,position)),changed)
      +cells_same = padded_zeros(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),Tiled.cell_shape(target,lane,position),cells_exact,front_zero,back_zero)
      +same = target_same(values,target,lane,lanes,position,laid,laid_target,cells_same)
      +kept_room = SetSlice.placed_room(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),limits,{%cells_same : Facts.PlacedIn(_,Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)) cell_placed : Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))},
        source_cells_room,source_cells_positive)
      +kept_half = Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),source_cells_room)
      +cells_back = Equal.sym(List<&2,U32>,Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),cells_same)
      unchanged_case(~Logical,~original,logical,pad_values(~Logical,Layout.Constant{value},logical,values,target,before),values,target,lane,lanes,position,limits,model,model_fill,model_workers,global,laid,same,source_cells_room,
        word => inside_word => kept_cell(~Logical,~original,logical,pad_values(~Logical,Layout.Constant{value},logical,values,target,before),values,target,lane,position,word,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),Bool.pick(Tile.Tile8,Bool.not(within),Tiled.unit_tile(value),Tile.filled(value))},word),
          constant_read(~Logical,~original,logical,values,target,before,lane,position,limits,Bool.pick(Tile.Tile8,Bool.not(within),Tiled.unit_tile(value),Tile.filled(value)),word,same,kept_room,front_zero,kept_half,inside_word),same,
          constant_agree(~Logical,~original,within,logical,values,target,before,after,lane,position,limits,value,exact,classified,free,placed,cell_placed,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,word,
            {%cells_back : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(_)) == True{} : Bool} inside_word : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool}})))
    case False{}:
      placed_case(~Logical,~original,logical,pad_values(~Logical,Layout.Constant{value},logical,values,target,before),values,target,before,after,lane,lanes,position,limits,model,model_fill,model_workers,
        Layout.placed_segments(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),False{},Bool.pick(Tile.Tile8,Bool.not(within),Tiled.unit_tile(value),Tile.filled(value)),global,cells_exact,Facts.placed_axes(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),limits,cell_placed,cells_room,cells_positive),
        laid_target,PaddedCells.placed(Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(target,lane,position))),target,lane,lanes,position,Tiled.cell_shape(target,lane,position),{==},laid_target),cells_counted,
        cells_room,cells_positive,cells_rank,source_cells_room,source_cells_positive,
        word => inside_word => constant_agree(~Logical,~original,within,logical,values,target,before,after,lane,position,limits,value,exact,classified,free,placed,cell_placed,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,word,inside_word))

# Cells of an edge padded, or left as they are.
def edge_change(~Logical: Data,~original: Logical -> U32 -> F32,unchanged: Bool,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,+within: Bool,
  +changed: {Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)) == unchanged : Bool},
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),+global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers}),+laid: Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>,+laid_target: Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>,+exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+cells_exact: {Shapes.padded(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>},+placed: Facts.PlacedIn(target,values,before),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+cells_counted: Global.Counted<Tiled.cell_shape(target,lane,position)>,+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+shifts_same: {List.length(&2,U32,before) == List.length(&2,U32,values) : Nat},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},+source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},+cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,Layout.Edge{},logical,values,target,before),target,
    Tiled.Tiled{Layout.pad_valid(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),Layout.Edge{},True{},unchanged,PiecesModel.pack(~Tile.Tile8,model),
      Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_shape(target,lane,position),model_fill,model_workers),lane,lanes,position}):
  match unchanged:
    case True{}:
      +front_zero = Boolean.left(Layout.all_zero(Placing.cell_shifts(before,lane,position)),Layout.all_zero(Tiled.cell_values(after,lane,position)),changed)
      +back_zero = Boolean.right(Layout.all_zero(Placing.cell_shifts(before,lane,position)),Layout.all_zero(Tiled.cell_values(after,lane,position)),changed)
      +cells_same = padded_zeros(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),Tiled.cell_shape(target,lane,position),cells_exact,front_zero,back_zero)
      +same = target_same(values,target,lane,lanes,position,laid,laid_target,cells_same)
      +kept_in = {%cells_same : Facts.PlacedIn(_,Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)) cell_placed : Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))}
      +kept_room = SetSlice.placed_room(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),limits,kept_in,source_cells_room,source_cells_positive)
      +kept_half = Limits.double_bound(limits,ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),source_cells_room)
      +cells_back = Equal.sym(List<&2,U32>,Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),cells_same)
      unchanged_case(~Logical,~original,logical,pad_values(~Logical,Layout.Edge{},logical,values,target,before),values,target,lane,lanes,position,limits,model,model_fill,model_workers,global,laid,same,source_cells_room,
        word => inside_word => kept_cell(~Logical,~original,logical,pad_values(~Logical,Layout.Edge{},logical,values,target,before),values,target,lane,position,word,
          Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{Placing.lifted_logicals(~Logical,[logical],[values],lane,position),[Layout.edge_segments(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position))],[Tiled.cell_shape(values,lane,position)],Tiled.cell_shape(target,lane,position),model_fill},word),
          edge_read(~Logical,~original,logical,values,target,before,lane,position,limits,model_fill,word,same,kept_in,source_cells_positive,kept_room,front_zero,kept_half,inside_word),same,
          edge_agree(~Logical,~original,within,logical,values,target,before,after,lane,position,limits,model_fill,exact,classified,free,placed,cell_placed,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,word,
            {%cells_back : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(_)) == True{} : Bool} inside_word : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool}})))
    case False{}:
      placed_case(~Logical,~original,logical,pad_values(~Logical,Layout.Edge{},logical,values,target,before),values,target,before,after,lane,lanes,position,limits,model,model_fill,model_workers,
        Layout.edge_segments(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),True{},model_fill,global,cells_exact,
        Facts.edge_axes(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),limits,cell_placed,cells_room,cells_positive,source_cells_positive),
        laid_target,PaddedCells.placed(Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(target,lane,position))),target,lane,lanes,position,Tiled.cell_shape(target,lane,position),{==},laid_target),cells_counted,
        cells_room,cells_positive,cells_rank,source_cells_room,source_cells_positive,
        word => inside_word => edge_agree(~Logical,~original,within,logical,values,target,before,after,lane,position,limits,model_fill,exact,classified,free,placed,cell_placed,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,word,inside_word))

# Cells padded at their border.
def pad_border(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,+within: Bool,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),+global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers}),+laid: Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>,+laid_target: Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>,+exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+cells_exact: {Shapes.padded(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>},+placed: Facts.PlacedIn(target,values,before),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(target,lane,position),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position)),+cells_counted: Global.Counted<Tiled.cell_shape(target,lane,position)>,+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},+ranked: {List.length(&2,U32,values) == List.length(&2,U32,target) : Nat},+shifts_same: {List.length(&2,U32,before) == List.length(&2,U32,values) : Nat},+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(target,lane,position))) == True{} : Bool},+source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},+cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
    Tiled.Tiled{Layout.pad_valid(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),TiledLayout.cell_border(border,Bool.not(within)),
      Views.has_count(Some{Global.count(Tiled.cell_shape(target,lane,position),cells_counted)}) && Layout.border_fits(~Tile.Tile8,TiledLayout.cell_border(border,Bool.not(within)),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),
      Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)),PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_shape(target,lane,position),model_fill,model_workers),lane,lanes,position}):
  match border:
    case Layout.Constant{+value}:
      constant_change(~Logical,~original,Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)),logical,value,values,target,before,after,lane,lanes,position,limits,
        model,model_fill,model_workers,within,{==},classified,free,global,laid,laid_target,exact,cells_exact,placed,cell_placed,cells_counted,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,cells_rank)
    case Layout.Edge{}:
      %Equal.sym(Bool,Layout.edge_axes(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),True{},edges_held(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),ViewMoved.none_zero(Tiled.cell_shape(values,lane,position),source_cells_positive))) :
        Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,Layout.Edge{},logical,values,target,before),target,
          Tiled.Tiled{Layout.pad_valid(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),Layout.Edge{},Views.has_count(Some{Global.count(Tiled.cell_shape(target,lane,position),cells_counted)}) && _,
            Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)),PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_shape(target,lane,position),model_fill,model_workers),lane,lanes,position})
      edge_change(~Logical,~original,Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)),logical,values,target,before,after,lane,lanes,position,limits,
        model,model_fill,model_workers,within,{==},classified,free,global,laid,laid_target,exact,cells_exact,placed,cell_placed,cells_counted,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,cells_rank)

# Cells of the contract along an axis padding leaves whole, padded: the
# padded values.
def pad_model(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +model: PiecesModel.Model<Tile.Tile8>,+model_fill: Tile.Tile8,+model_workers: U32,+within: Bool,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+values_counted,+pinned}} = contract
  +laid = {Public.Laid{inside,shaped,restored,values_counted,pinned} : Public.Laid<Tiled.cell_shape(values,lane,position),lane,lanes,position,values>}
  +lengths = padded_lengths(values,before,after,target,exact)
  +shifts_same = first_equal(List.length(&2,U32,before),List.length(&2,U32,values),List.length(&2,U32,after),List.length(&2,U32,values),lengths)
  +classified_before = {%Equal.sym(Nat,List.length(&2,U32,before),List.length(&2,U32,values),shifts_same) : {Nat.is_lt(lane,_) == within : Bool} classified :
    {Nat.is_lt(lane,List.length(&2,U32,before)) == within : Bool}}
  +front_eq = front_cells(within,before,after,lane,position,classified_before,free)
  +cells_found = cells_padded(within,values,before,after,target,lane,position,classified,exact,free)
  +cells_exact = {%front_eq : {Shapes.padded(Tiled.cell_shape(values,lane,position),_,Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>} cells_found :
    {Shapes.padded(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)) == Some{Tiled.cell_shape(target,lane,position)} : Maybe<&2,List<&2,U32>>}}
  +placed = Extents.padded_in(values,before,after,target,exact)
  +cell_placed = Extents.padded_in(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),Tiled.cell_shape(target,lane,position),cells_exact)
  +ranked = Facts.placed_length(target,values,before,placed)
  +room = StorageCount.counted_room(limits,target,Global.Counted{count,counted},default_capacity)
  +source_positive = CountCompletion.positive_elements(values,nonempty)
  +positive = Facts.placed_positive(target,values,before,placed,source_positive)
  +source_room = StorageCount.counted_room(limits,values,values_counted,default_capacity)
  +cells_room = Rooms.cells_room(limits,target,lane,position,positive,room)
  +cells_positive = Rooms.cells_positive(limits,target,lane,position,positive,room)
  +source_cells_room = Rooms.cells_room(limits,values,lane,position,source_positive,source_room)
  +source_cells_positive = Rooms.cells_positive(limits,values,lane,position,source_positive,source_room)
  +cells_rank = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(target,lane,position)),List.length(&2,U32,target),Rooms.cell_rank(target,lane,position)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %ranked : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    rank_room : {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(target,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_counted = StorageCount.counted_within(limits,Tiled.cell_shape(target,lane,position),cells_room,default_capacity)
  +lanes_same = Equal.trans(U32,Grid.lanes_at(target,lane),Grid.lanes_at(values,lane),lanes,lanes_kept(within,values,target,before,after,lane,exact,classified,ranked,free),pinned)
  +inside_target = {%ranked : {Nat.is_le(lane,_) == True{} : Bool} inside : {Nat.is_le(lane,List.length(&2,U32,target)) == True{} : Bool}}
  +laid_target = {Public.Laid{inside_target,{==},
      {%lanes_same : {Tiled.logical_shape(Tiled.cell_shape(target,lane,position),lane,_,position) == target : List<&2,U32>} Public.cells_restored(target,lane,position) :
        {Tiled.logical_shape(Tiled.cell_shape(target,lane,position),lane,lanes,position) == target : List<&2,U32>}},
      Global.Counted{count,counted},lanes_same} : Public.Laid<Tiled.cell_shape(target,lane,position),lane,lanes,position,target>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.padded(Tiled.cell_shape(values,lane,position),Tiled.cell_values(before,lane,position),Tiled.cell_values(after,lane,position)),Some{Tiled.cell_shape(target,lane,position)},cells_found) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      Tiled.Tiled{Layout.pad_status(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Tiled.cell_values(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),
        TiledLayout.cell_border(border,Bool.not(Nat.is_lt(lane,List.length(&2,U32,values)))),Tensors.Usable{},_,PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),Tiled.cell_values(before,lane,position),Tiled.cell_values(after,lane,position),
        model_fill,model_workers),lane,lanes,position})
  %Equal.sym(List<&2,U32>,Tiled.cell_values(before,lane,position),Placing.cell_shifts(before,lane,position),front_eq) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      Tiled.Tiled{Layout.pad_status(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),_,Tiled.cell_values(after,lane,position)),lane,lanes,position),
        TiledLayout.cell_border(border,Bool.not(Nat.is_lt(lane,List.length(&2,U32,values)))),Tensors.Usable{},Some{Tiled.cell_shape(target,lane,position)},PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),_,Tiled.cell_values(after,lane,position),
        model_fill,model_workers),lane,lanes,position})
  %Equal.sym(Bool,Nat.is_lt(lane,List.length(&2,U32,values)),within,classified) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      Tiled.Tiled{Layout.pad_status(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),TiledLayout.cell_border(border,Bool.not(_)),Tensors.Usable{},Some{Tiled.cell_shape(target,lane,position)},
        PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position),model_fill,model_workers),lane,lanes,position})
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Tiled.cell_shape(target,lane,position)),Some{Global.count(Tiled.cell_shape(target,lane,position),cells_counted)},Global.accepted(Tiled.cell_shape(target,lane,position),cells_counted)) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      Tiled.Tiled{Layout.pad_valid(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(TiledLayout.grown(Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),lane,lanes,position),TiledLayout.cell_border(border,Bool.not(within)),
        Views.has_count(_) && Layout.border_fits(~Tile.Tile8,TiledLayout.cell_border(border,Bool.not(within)),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_values(after,lane,position)),
        Layout.all_zero(Placing.cell_shifts(before,lane,position)) && Layout.all_zero(Tiled.cell_values(after,lane,position)),PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),Placing.cell_shifts(before,lane,position),Tiled.cell_shape(target,lane,position),model_fill,model_workers),lane,lanes,position})
  pad_border(~Logical,~original,border,logical,values,target,before,after,lane,lanes,position,limits,model,model_fill,model_workers,within,classified,free,global,laid,laid_target,exact,cells_exact,placed,cell_placed,cells_counted,inside,ranked,shifts_same,room,positive,source_room,source_positive,cells_room,cells_positive,source_cells_room,source_cells_positive,cells_rank)

# Usable cells of the contract padded.
def pad_shaped(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,-cells: Tensors.Tensor<Tile.Tile8>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+within: Bool,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tiled.cell_shape(values,lane,position),cells),+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,lane,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),Tiled.Tiled{cells,lane,lanes,position})):
  KernelGemm.Both{Global.Usable{+model,+model_fill,+model_workers,+equation},+global} = shaped
  %equation : Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),Tiled.Tiled{_,lane,lanes,position}))
  pad_model(~Logical,~original,border,logical,values,target,before,after,lane,lanes,position,limits,default_capacity,model,model_fill,model_workers,within,classified,free,
    exact,count,counted,nonempty,rank_room,
    {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},cells,equation) : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{_,lane,lanes,position}) contract :
      Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),model_fill,Tensors.Usable{},model_workers},lane,lanes,position})})

# Cells of the contract along an axis padding leaves whole, padded.
def pad_cells(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+within: Bool,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+free: FreeLane(within,before,after,lane),
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,lane,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),Tiled.Tiled{cells,lane,lanes,position})):
  pad_shaped(~Logical,~original,border,logical,values,target,before,after,cells,lane,lanes,position,limits,default_capacity,within,classified,free,exact,count,counted,
    nonempty,rank_room,Public.cells_shaped(~Logical,~original,logical,values,lane,lanes,position,cells,contract),contract)

# ---- The lane axis of padding ----

def unpadded_pick(chosen: Bool,+index: Nat,+rest: Nat,+rank: Nat,+index_in: {Nat.is_le(index,rank) == True{} : Bool},+rest_in: {Nat.is_le(rest,rank) == True{} : Bool}) ->
  {Nat.is_le(Bool.pick(Nat,chosen,index,rest),rank) == True{} : Bool}:
  match chosen:
    case True{}: index_in
    case False{}: rest_in

# The search for an unpadded axis ends at most at the rank.
def unpadded_le(before: List<&2,U32>,after: List<&2,U32>,+index: Nat,+rank: Nat,+fits: {Nat.is_le(Nat.add(index,List.length(&2,U32,before)),rank) == True{} : Bool}) ->
  {Nat.is_le(TiledLayout.unpadded_from(before,after,index,rank),rank) == True{} : Bool}:
  match before after:
    case Con{+front,+fronts} Con{+back,+backs}:
      unpadded_pick(U32.is_zero(front) && U32.is_zero(back),index,TiledLayout.unpadded_from(fronts,backs,1n+index,rank),rank,
        Order.transitive(index,Nat.add(index,1n+List.length(&2,U32,fronts)),rank,Order.le_add(index,1n+List.length(&2,U32,fronts)),fits),
        unpadded_le(fronts,backs,1n+index,rank,
          {%Addition.successor_right(index,List.length(&2,U32,fronts)) : {Nat.is_le(_,rank) == True{} : Bool} fits :
            {Nat.is_le(Nat.add(1n+index,List.length(&2,U32,fronts)),rank) == True{} : Bool}}))
    case Nil{} +after: Order.reflexive(rank)
    case Con{+front,+fronts} Nil{}: Order.reflexive(rank)

# The lanes padding keeps lie at most at the rank.
def pad_lane_le(+lane: Nat,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,+inside: {Nat.is_le(lane,rank) == True{} : Bool},
  +front_length: {List.length(&2,U32,before) == rank : Nat}) -> {Nat.is_le(TiledLayout.pad_lane(lane,before,after,rank),rank) == True{} : Bool}:
  unpadded_pick(U32.is_zero(Layout.extent_at(before,lane)) && U32.is_zero(Layout.extent_at(after,lane)),lane,TiledLayout.unpadded_from(before,after,0n,rank),rank,inside,
    unpadded_le(before,after,0n,rank,
      {%Equal.sym(Nat,List.length(&2,U32,before),rank,front_length) : {Nat.is_le(_,rank) == True{} : Bool} Order.reflexive(rank) :
        {Nat.is_le(List.length(&2,U32,before),rank) == True{} : Bool}}))

# Cells of the contract relaned to the axis padding keeps, padded.
def pad_relaned(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+axis: Nat,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classified: {Nat.is_eq(old,axis) == False{} : Bool},+inside_axis: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +free: FreeLane(Nat.is_lt(axis,List.length(&2,U32,values)),before,after,axis),
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),
    Tiled.along_when(False{},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},axis,position))):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  +packed = {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
      Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{_,old,lanes,position})
    contract : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},old,lanes,position})}
  +shown = Tiled.logical_shape(shape,old,lanes,position)
  +rank = List.length(&2,U32,shown)
  %equation : Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),
    Tiled.along_when(False{},Tiled.Tiled{_,old,lanes,position},axis,position)))
  pad_cells(~Logical,~original,border,logical,values,target,before,after,
    Tiled.relane_status(Tensors.Usable{},Tiled.Relane{old,Nat.is_lt(old,rank),position,axis,Nat.is_lt(axis,rank),position,Grid.lanes_at(shown,axis),Tiled.cell_shape(shown,axis,position)},
      shown,shape,model_workers,PiecesModel.pack(~Tile.Tile8,model)),
    axis,Grid.lanes_at(shown,axis),position,limits,default_capacity,Nat.is_lt(axis,List.length(&2,U32,values)),{==},free,exact,count,counted,nonempty,rank_room,
    Relane.tiled_relane(~Logical,~original,logical,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},old,lanes,position,
      axis,position,limits,default_capacity,inside_axis,Equal.trans(Bool,Nat.is_eq(axis,old),Nat.is_eq(old,axis),False{},Order.equal_symmetric(axis,old),classified),
      CountCompletion.positive_elements(values,nonempty),rank_room,packed))

# Cells of the contract with their lanes moved to the axis padding keeps,
# padded.
def pad_lanes(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+axis: Nat,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classified: {Nat.is_eq(old,axis) == same : Bool},+inside_axis: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +free: FreeLane(Nat.is_lt(axis,List.length(&2,U32,values)),before,after,axis),
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.padded_cells(border,before,after,List.length(&2,U32,values),
    Tiled.along_when(same,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},axis,position))):
  match same:
    case True{}:
      +equal = Order.equal_of_is_eq(old,axis,classified)
      pad_cells(~Logical,~original,border,logical,values,target,before,after,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position,limits,default_capacity,
        Nat.is_lt(old,List.length(&2,U32,values)),{==},
        {%Equal.sym(Nat,old,axis,equal) : FreeLane(Nat.is_lt(_,List.length(&2,U32,values)),before,after,_) free :
          FreeLane(Nat.is_lt(old,List.length(&2,U32,values)),before,after,old)},
        exact,count,counted,nonempty,rank_room,contract)
    case False{}:
      pad_relaned(~Logical,~original,border,logical,values,target,before,after,pieces,shape,fill,workers,old,lanes,position,axis,limits,default_capacity,
        exact,count,counted,nonempty,rank_room,classified,inside_axis,free,Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),contract)

# Usable tiled values of the contract padded.
def pad_ready(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +fits: {Layout.border_fits(~F32,border,values,before,after) == True{} : Bool},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.pad(border,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},before,after)):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+values_counted,+pinned}} = contract
  +rank = List.length(&2,U32,values)
  +lengths = padded_lengths(values,before,after,target,exact)
  +front_length = first_equal(List.length(&2,U32,before),rank,List.length(&2,U32,after),rank,lengths)
  +back_length = second_equal(List.length(&2,U32,before),rank,List.length(&2,U32,after),rank,lengths)
  +axis = TiledLayout.pad_lane(old,before,after,rank)
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,old,lanes,position),values,restored) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      TiledLayout.pad_valid(border,Tensors.Usable{},Views.has_count(Layout.padded_count(Shapes.padded(_,before,after))) && Layout.border_fits(~F32,border,_,before,after),Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},_,before,after))
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.padded(values,before,after),Some{target},exact) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      TiledLayout.pad_valid(border,Tensors.Usable{},Views.has_count(Layout.padded_count(_)) && Layout.border_fits(~F32,border,values,before,after),Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},values,before,after))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(target),Some{count},counted) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
      TiledLayout.pad_valid(border,Tensors.Usable{},Views.has_count(_) && Layout.border_fits(~F32,border,values,before,after),Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},values,before,after))
  %Equal.sym(Bool,Layout.border_fits(~F32,border,values,before,after),True{},fits) :
    Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.pad_valid(border,Tensors.Usable{},Views.has_count(Some{count}) && _,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},values,before,after))
  pad_lanes(~Logical,~original,Nat.is_eq(old,axis),border,logical,values,target,before,after,pieces,shape,fill,workers,old,lanes,position,axis,limits,default_capacity,
    exact,count,counted,nonempty,rank_room,{==},pad_lane_le(old,before,after,rank,inside,front_length),pad_free(old,before,after,rank,front_length,back_length),contract)

# ---- The tiled form of pad ----

# Tiled values of the contract without a zero extent, padded to counted
# extents within the capacity where the border fits: a tiled tensor of the
# padded values.
def pad(~Logical: Data,~original: Logical -> U32 -> F32,border: Layout.Border<F32>,+logical: Logical,+values: List<&2,U32>,+target: List<&2,U32>,
  +before: List<&2,U32>,+after: List<&2,U32>,tiled: Tiled.Tiled,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +exact: {Shapes.padded(values,before,after) == Some{target} : Maybe<&2,List<&2,U32>>},+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +fits: {Layout.border_fits(~F32,border,values,before,after) == True{} : Bool},
  +nonempty: {Shapes.has_zero(values) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,TiledLayout.pad(border,tiled,before,after)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          pad_ready(~Logical,~original,border,logical,values,target,before,after,pieces,shape,fill,workers,old,lanes,position,limits,default_capacity,
            exact,count,counted,fits,nonempty,rank_room,contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),pad_values(~Logical,border,logical,values,target,before),target,
            TiledLayout.pad(border,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},before,after)),contract)
