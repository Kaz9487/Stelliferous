# Tiled reductions. A Ready reduction of tiled values of the contract holds, at
# the row-major index of the kept axes, the finish of the fold of the values of
# the reduced axes there in row-major order: the input read through the view
# with the kept axes first and the reduced axes last.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_reduce.bend as TiledReduce
import ../tensor_partials.bend as Partials
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_placing.bend as Placing
import ./tensor_tiled_view.bend as TiledViewValues
import ./tensor_view_gather.bend as ViewGather
import ./tensor_reduce_values.bend as Reduce
import ./tensor_move_readback.bend as Held
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./list_storage_readback.bend as ListReadback
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./tensor_view_reads.bend as ViewReads
import ./tensor_tiled_view_reads.bend as TiledViewReads
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_count_completion.bend as CountCompletion
import ./u32_full_range.bend as FullRange
import ./tensor_box_digits.bend as Digits
import ./tensor_target_cells.bend as Targets
import ./tensor_tiled_padding.bend as Padding
import ./tensor_relane_values.bend as Values
import ./tensor_view_metadata_refinement.bend as MetadataLaws
import ./nat_division_order.bend as DivisionOrder
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tasks.bend as Tasks
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_content.bend as Content
import ./storage_buffer_model.bend as Buffer
import ./tensor_tiled_values.bend as TiledValues
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_ceiling.bend as Ceiling
import ./boolean_logic.bend as Boolean

# ---- The specification ----

# The tiled values of a reduction of the listed axes: at the row-major index of
# the kept axes, the finish of the fold of the values of the reduced axes there.
def Folded(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+initial: Accumulator,+values: List<&2,U32>,+axes: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes)))},Layout.extents_at(values,Reduce.kept_axes(values,axes)),tiled)

# ---- Values on the unit axis ----

# A cell of the unit axis: the value in its first lane, +0 in the others.
def unit_of(~Logical: Data,~original: Logical -> U32 -> F32,logical: Logical,index: U32) -> Tile.Tile8:
  Tiled.unit_tile(original(logical,index))

def unit_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+position: Grid.Position,+index: U32,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(values)) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},index) == unit_of(~Logical,~original,logical,index) : Tile.Tile8}:
  TileLaws.tiles_equal(Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},index),unit_of(~Logical,~original,logical,index),
    Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,0n,{==}),Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,1n,{==}),
    Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,2n,{==}),Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,3n,{==}),
    Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,4n,{==}),Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,5n,{==}),
    Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,6n,{==}),Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,index,room,inside,7n,{==}))

# Cells that hold the values on the unit axis are tiled values of the
# contract with their lanes there.
def unit_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,-cells: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Logical,~unit_of(~Logical,~original),logical,values,cells),
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+count: U32,+counted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>}) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,List.length(&2,U32,values),1,Grid.Innermost{}}):
  +rank = List.length(&2,U32,values)
  +same = Placing.cells_past(values,rank,Grid.Innermost{},Order.not_below(rank,rank,Order.reflexive(rank)))
  +lifted = Global.restated_below(~Tile.Tile8,~Logical,~Representation.Lifted<Logical>,~unit_of(~Logical,~original),~Representation.lifted(~Logical,~original),~ShapeLaws.elements(values),
    logical,Representation.Lifted{logical,Public.geometry_of(values,rank,Grid.Innermost{})},
    index => inside => Equal.sym(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,rank,Grid.Innermost{})},index),
      unit_of(~Logical,~original,logical,index),unit_agree(~Logical,~original,logical,limits,values,Grid.Innermost{},index,room,inside)),
    values,cells,shaped,Order.reflexive(ShapeLaws.elements(values)),Machine.capacity(limits),room)
  Public.laid_contract(~Logical,~original,logical,values,cells,rank,1,Grid.Innermost{},
    KernelGemm.Both{%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,rank,Grid.Innermost{}),values,same) :
        Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,rank,Grid.Innermost{})},_,cells)
      lifted,
      TiledViewValues.unit_laid(values,count,counted)})

# Unit cells fold their first lanes as the values fold.
def unit_fold_some(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axes: List<&2,Views.Axis>,+offset: U32,count: Nat,+position: Nat,+state: Accumulator) ->
  {Reduce.line_fold(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original)),Reduce.Ordered{logical,Some{Views.View{axes,offset}}},count,position,state) == Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Some{Views.View{axes,offset}}},count,position,state) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      unit_fold_some(~Accumulator,~operation,~finish,~Logical,~original,logical,axes,offset,rest,1n+position,
        operation(state,ViewGather.viewed(~F32,~Logical,~original,ViewGather.Viewing{logical,axes,offset,[]},U32.from_nat(position))))

def unit_fold_none(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,count: Nat,+position: Nat,+state: Accumulator) ->
  {Reduce.line_fold(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original)),Reduce.Ordered{logical,None{}},count,position,state) == Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,None{}},count,position,state) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest: unit_fold_none(~Accumulator,~operation,~finish,~Logical,~original,logical,rest,1n+position,operation(state,original(logical,U32.from_nat(position))))

def unit_fold(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,view: Maybe<&2,Views.View>,+logical: Logical,+count: Nat,+position: Nat,+state: Accumulator) ->
  {Reduce.line_fold(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original)),Reduce.Ordered{logical,view},count,position,state) == Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,view},count,position,state) : Accumulator}:
  match view:
    case Some{Views.View{+axes,+offset}}: unit_fold_some(~Accumulator,~operation,~finish,~Logical,~original,logical,axes,offset,count,position,state)
    case None{}: unit_fold_none(~Accumulator,~operation,~finish,~Logical,~original,logical,count,position,state)

# A line of unit cells finishes on the unit axis as the line of the values
# finishes.
def unit_line(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+initial: Accumulator,+logical: Logical,+view: Maybe<&2,Views.View>,+length: Nat,+index: U32) ->
  {Reduce.line_value(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~TiledReduce.unit_finish(~Accumulator,~finish),~Reduce.Ordered<Logical>,
      ~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,view},length},index) ==
    unit_of(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,view},length},index) : Tile.Tile8}:
  Equal.cong(Accumulator,Tile.Tile8,state => Tiled.unit_tile(finish(state)),Reduce.line_fold(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original)),Reduce.Ordered{logical,view},length,Nat.mul(U32.to_nat(index),length),initial),
    Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,view},length,Nat.mul(U32.to_nat(index),length),initial),unit_fold(~Accumulator,~operation,~finish,~Logical,~original,view,logical,length,Nat.mul(U32.to_nat(index),length),initial))

# ---- The lanes of the result ----

def arranged_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tiled: Tiled.Tiled,+axis: Nat,
  +arranged: Relane.Arranged(~Logical,~original,logical,values,tiled,axis,Grid.Innermost{})) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.arranged(tiled,axis,Grid.Innermost{})):
  KernelGemm.Both{+eta,+contract} = arranged
  %Equal.sym(Tiled.Tiled,Tiled.arranged(tiled,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(tiled,axis,Grid.Innermost{})),axis,Grid.lanes_at(values,axis),Grid.Innermost{}},eta) :
    Public.Contract(~Logical,~original,logical,values,_)
  contract

# A result of one value a cell with its lanes on the longest kept axis.
def lanes_contract(~Logical: Data,~original: Logical -> U32 -> F32,choice: Bool,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),+count: Nat,+classified: {Nat.is_lt(axis,count) == choice : Bool},+ranked: {count == List.length(&2,U32,values) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Public.Contract(~Logical,~original,logical,values,TiledReduce.result_arranged(choice,axis,tiled)):
  match choice:
    case True{}:
      +inside = Bounds.strict_le(axis,List.length(&2,U32,values),Held.nat_lt_moved(axis,List.length(&2,U32,values),count,Equal.sym(Nat,count,List.length(&2,U32,values),ranked),classified))
      arranged_contract(~Logical,~original,logical,values,tiled,axis,
        Relane.arranged_held(~Logical,~original,logical,values,tiled,axis,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,contract))
    case False{}: contract

# ---- One value a cell ----

# Unit cells that hold the values of a reader, packed, with their lanes moved
# to the longest kept axis.
def unit_packed(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+limits: Machine.Limits,+axes: List<&2,U32>,+values: List<&2,U32>,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+restated: Global.Shaped(~Tile.Tile8,~Context,~unit_of(~Context,~reader),context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Public.Contract(~Context,~reader,context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),TiledReduce.result_lanes(values,Layout.naturals(axes),Tiled.unit(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers}))):
  +rank = List.length(&2,U32,values)
  +listed = Layout.naturals(axes)
  +kept = Reduce.kept_axes(values,axes)
  +reduced = Reduce.reduced_axes(values,axes)
  +counted = Global.shaped_counted(~Tile.Tile8,~Context,~unit_of(~Context,~reader),context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers},restated)
  +room = StorageCount.counted_room(limits,Layout.extents_at(values,Reduce.kept_axes(values,axes)),counted,default_capacity)
  +product = Equal.trans(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Reduce.kept_axes(values,axes))),ShapeLaws.elements(Layout.extents_at(values,reduced))),ShapeLaws.elements(List.drop(&2,U32,values,0n)),ShapeLaws.elements(values),
    Reduce.partition_elements(values,rank,0n,listed,{==}),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,values,0n),values,Reduce.dropped_none(~U32,values)))
  +kept_positive = RowMajor.left_factor_positive(ShapeLaws.elements(Layout.extents_at(values,Reduce.kept_axes(values,axes))),ShapeLaws.elements(Layout.extents_at(values,reduced)),
    Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Reduce.kept_axes(values,axes))),ShapeLaws.elements(Layout.extents_at(values,reduced))),ShapeLaws.elements(values),product,nonempty))
  +widths = Reduce.partition_length(rank,0n,listed)
  +kept_rank = Held.nat_le_right_moved(List.length(&2,Nat,kept),rank,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),
    Equal.sym(Nat,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),rank,widths),Order.le_add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)))
  +lengths = Reduce.extents_length(values,kept)
  +out_rank = Held.nat_le_moved(List.length(&2,U32,Layout.extents_at(values,Reduce.kept_axes(values,axes))),List.length(&2,Nat,kept),U32.to_nat(Machine.capacity(limits)),lengths,Order.transitive(List.length(&2,Nat,kept),rank,U32.to_nat(Machine.capacity(limits)),kept_rank,rank_room))
  +count = List.length(&2,Nat,kept)
  +longest = TiledReduce.longest_kept(kept,values,0n,count,0)
  lanes_contract(~Context,~reader,Nat.is_lt(longest,count),context,limits,Layout.extents_at(values,Reduce.kept_axes(values,axes)),longest,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers},List.length(&2,U32,Layout.extents_at(values,Reduce.kept_axes(values,axes))),1,Grid.Innermost{}},
    unit_contract(~Context,~reader,context,limits,Layout.extents_at(values,Reduce.kept_axes(values,axes)),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers},restated,room,Global.count(Layout.extents_at(values,Reduce.kept_axes(values,axes)),counted),Global.accepted(Layout.extents_at(values,Reduce.kept_axes(values,axes)),counted)),
    count,{==},Equal.sym(Nat,List.length(&2,U32,Layout.extents_at(values,Reduce.kept_axes(values,axes))),count,lengths),default_capacity,kept_positive,out_rank)

# The same for any Shaped unit cells.
def unit_result(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+limits: Machine.Limits,+axes: List<&2,U32>,+values: List<&2,U32>,-result: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Context,~unit_of(~Context,~reader),context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),result),+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Public.Contract(~Context,~reader,context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),TiledReduce.result_lanes(values,Layout.naturals(axes),Tiled.unit(result))):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},contract} = shaped
  %equation : Public.Contract(~Context,~reader,context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),TiledReduce.result_lanes(values,Layout.naturals(axes),Tiled.unit(_)))
  unit_packed(~Context,~reader,context,limits,axes,values,model,fill,workers,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(values,Reduce.kept_axes(values,axes)),fill,Tensors.Usable{},workers},result,equation) : Global.Shaped(~Tile.Tile8,~Context,~unit_of(~Context,~reader),context,Layout.extents_at(values,Reduce.kept_axes(values,axes)),_) shaped,
    default_capacity,rank_room,nonempty)

# The unit cells of a reduction finished on the unit axis.
def single_result(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+initial: Accumulator,+axes: List<&2,U32>,+values: List<&2,U32>,-result: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~TiledReduce.unit_finish(~Accumulator,~finish),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original))),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes)))},Layout.extents_at(values,Reduce.kept_axes(values,axes)),result),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,axes,TiledReduce.result_lanes(values,Layout.naturals(axes),Tiled.unit(result))):
  unit_result(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes)))},limits,axes,values,result,
    Global.restated(~Tile.Tile8,~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~TiledReduce.unit_finish(~Accumulator,~finish),~Reduce.Ordered<Logical>,~Reduce.through(~Tile.Tile8,~Logical,~unit_of(~Logical,~original))),~unit_of(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes)))},Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes)))},
      index => unit_line(~Accumulator,~operation,~finish,~Logical,~original,initial,logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,axes),Reduce.reduced_axes(values,axes))),ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,axes))),index),Layout.extents_at(values,Reduce.kept_axes(values,axes)),result,shaped),
    default_capacity,rank_room,nonempty)

# The reduction of tiled values of the contract with their lanes on the unit
# axis.
def single_route(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+axes: List<&2,U32>,+values: List<&2,U32>,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool},
  +unique: {Views.unique_axes(Layout.naturals(axes)) == True{} : Bool},+below: {Layout.all_below(axes,List.length(&2,U32,values)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,axes,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Single{},name,operations,initial,values,axes,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position})):
  +rank = List.length(&2,U32,values)
  +same = Placing.cells_past(values,rank,position,Order.not_below(rank,rank,Order.reflexive(rank)))
  +lifted = {%same : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,rank,position)},_,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers})
    Public.cells_shaped(~Logical,~original,logical,values,rank,Grid.lanes_at(values,rank),position,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},contract) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,rank,position)},values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers})}
  +room = StorageCount.counted_room(limits,values,
    Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,rank,position)},values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},lifted),
    default_capacity)
  +units = Global.restated_below(~Tile.Tile8,~Representation.Lifted<Logical>,~Logical,~Representation.lifted(~Logical,~original),~unit_of(~Logical,~original),~ShapeLaws.elements(values),
    Representation.Lifted{logical,Public.geometry_of(values,rank,position)},logical,index => inside => unit_agree(~Logical,~original,logical,limits,values,position,index,room,inside),
    values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},lifted,Order.reflexive(ShapeLaws.elements(values)),Machine.capacity(limits),room)
  +kept = Reduce.kept_axes(values,axes)
  +count = List.length(&2,Nat,kept)
  +longest = TiledReduce.longest_kept(kept,values,0n,count,0)
  single_result(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,initial,axes,values,Layout.reduce_axes(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~TiledReduce.unit_finish(~Accumulator,~finish),name,operations,initial,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},axes),
    Reduce.reduce_axes(~Tile.Tile8,~Accumulator,~TiledReduce.lane_operation(~Accumulator,~operation),~TiledReduce.unit_finish(~Accumulator,~finish),~Logical,~unit_of(~Logical,~original),
      logical,limits,name,operations,initial,axes,model,values,fill,workers,Global.contract_of(~Tile.Tile8,~Logical,~unit_of(~Logical,~original),logical,values,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},units),
      default_capacity,rank_room,nonempty,cap,cap_ok,count_room,
      Boolean.and_true(Views.unique_axes(Layout.naturals(axes)),Layout.all_below(axes,List.length(&2,U32,values)),unique,below)),
    default_capacity,rank_room,nonempty)

# ---- The address of a line of a reduction in digits ----

def flags_length(remaining: Nat,+index: Nat,+axes: List<&2,Nat>) -> {List.length(&2,Bool,Layout.reduced_flags(remaining,index,axes)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Bool,Layout.reduced_flags(rest,1n+index,axes)),rest,flags_length(rest,1n+index,axes))

# The axes from index on whose flag is listed.
def flagged(flags: List<&2,Bool>,+index: Nat,+listed: Bool) -> List<&2,Nat>:
  match flags:
    case []: []
    case flag <> rest: Layout.chosen(flag,listed,index,flagged(rest,1n+index,listed))

def where_flagged(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+listed: Bool) ->
  {Layout.axes_where(remaining,index,axes,listed) == flagged(Layout.reduced_flags(remaining,index,axes),index,listed) : List<&2,Nat>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Layout.chosen(Views.contains_index(axes,index),listed,index,tail),
        Layout.axes_where(rest,1n+index,axes,listed),flagged(Layout.reduced_flags(rest,1n+index,axes),1n+index,listed),where_flagged(rest,1n+index,axes,listed))

# The digits of the kept axes and of the listed axes put back in axis order.
def merged(flags: List<&2,Bool>,kept: List<&2,Nat>,reduced: List<&2,Nat>) -> List<&2,Nat>:
  match flags:
    case []: []
    case flag <> rest:
      match flag:
        case True{}:
          match reduced:
            case []: []
            case digit <> more: digit <> merged(rest,kept,more)
        case False{}:
          match kept:
            case []: []
            case digit <> more: digit <> merged(rest,more,reduced)

def digits_none(extents: List<&2,U32>) -> {ShapeLaws.digits_value(extents,[]) == 0n : Nat}:
  match extents:
    case []: {==}
    case extent <> rest: {==}

# The steps of a view at digits after digits.
def sum_appended(axes: List<&2,Views.Axis>,digits: List<&2,Nat>,+more_axes: List<&2,Views.Axis>,+more: List<&2,Nat>,+below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits)) ->
  {TiledViewReads.digit_sum(List.append(&2,Views.Axis,axes,more_axes),[],List.append(&2,Nat,digits,more)) ==
    Nat.add(TiledViewReads.digit_sum(axes,[],digits),TiledViewReads.digit_sum(more_axes,[],more)) : Nat}:
  match axes digits:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+tail}:
      KernelGemm.Both{here,+later} = below
      +step = Nat.mul(digit,U32.to_nat(stride))
      +inner = TiledViewReads.digit_sum(rest,[],tail)
      +outer = TiledViewReads.digit_sum(more_axes,[],more)
      %Equal.sym(Nat,TiledViewReads.digit_sum(List.append(&2,Views.Axis,rest,more_axes),[],List.append(&2,Nat,tail,more)),Nat.add(inner,outer),sum_appended(rest,tail,more_axes,more,later)) :
        {Nat.add(step,_) == Nat.add(Nat.add(step,inner),outer) : Nat}
      Equal.sym(Nat,Nat.add(Nat.add(step,inner),outer),Nat.add(step,Nat.add(inner,outer)),Addition.associative(step,inner,outer))
    case Nil{} Con{+digit,+tail}:
      Empty.absurd({TiledViewReads.digit_sum(List.append(&2,Views.Axis,[],more_axes),[],List.append(&2,Nat,Con{digit,tail},more)) ==
        Nat.add(TiledViewReads.digit_sum([],[],Con{digit,tail}),TiledViewReads.digit_sum(more_axes,[],more)) : Nat},below)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd({TiledViewReads.digit_sum(List.append(&2,Views.Axis,Con{Views.Axis{extent,stride},rest},more_axes),[],List.append(&2,Nat,[],more)) ==
        Nat.add(TiledViewReads.digit_sum(Con{Views.Axis{extent,stride},rest},[],[]),TiledViewReads.digit_sum(more_axes,[],more)) : Nat},below)

def order_appended(left: List<&2,Nat>,+right: List<&2,Nat>,+shape: List<&2,U32>) ->
  {Reduce.order_axes(List.append(&2,Nat,left,right),shape) == List.append(&2,Views.Axis,Reduce.order_axes(left,shape),Reduce.order_axes(right,shape)) : List<&2,Views.Axis>}:
  match left:
    case []: {==}
    case +axis <> +rest:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => {Views.Axis{Layout.extent_at(shape,axis),Grid.product(List.drop(&2,U32,shape,1n+axis))} : Views.Axis} <> tail,
        Reduce.order_axes(List.append(&2,Nat,rest,right),shape),List.append(&2,Views.Axis,Reduce.order_axes(rest,shape),Reduce.order_axes(right,shape)),order_appended(rest,right,shape))

# One axis further: its digit times the cells after it.
def split_kept(+rest: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+digit: Nat,+more: List<&2,Nat>,+reduced: List<&2,Nat>,
  +fits: {Nat.add(index,1n+List.length(&2,Bool,rest)) == List.length(&2,U32,shape) : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool},
  +sum: {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,False{}),shape),[],more),TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,True{}),shape),[],reduced)) ==
    ShapeLaws.digits_value(List.drop(&2,U32,shape,1n+index),merged(rest,more,reduced)) : Nat}) ->
  {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(False{} <> rest,index,False{}),shape),[],digit <> more),TiledViewReads.digit_sum(Reduce.order_axes(flagged(False{} <> rest,index,True{}),shape),[],reduced)) ==
    ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged(False{} <> rest,digit <> more,reduced)) : Nat}:
  +after = List.drop(&2,U32,shape,1n+index)
  +step = U32.to_nat(Grid.product(after))
  +kept_sum = TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,False{}),shape),[],more)
  +reduced_sum = TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,True{}),shape),[],reduced)
  +value = ShapeLaws.digits_value(after,merged(rest,more,reduced))
  +inside = {%fits : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,List.length(&2,Bool,rest)) : {Nat.is_lt(index,List.length(&2,U32,shape)) == True{} : Bool}}
  +product = Public.product_exact(cap,cap_ok,after,CountCompletion.suffix_room(shape,1n+index,U32.to_nat(cap),positive,room))
  %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> after,Reduce.drop_next(shape,index,inside)) :
    {Nat.add(Nat.add(Nat.mul(digit,step),kept_sum),reduced_sum) == ShapeLaws.digits_value(_,digit <> merged(rest,more,reduced)) : Nat}
  %product : {Nat.add(Nat.add(Nat.mul(digit,step),kept_sum),reduced_sum) == Nat.add(Nat.mul(digit,_),value) : Nat}
  %sum : {Nat.add(Nat.add(Nat.mul(digit,step),kept_sum),reduced_sum) == Nat.add(Nat.mul(digit,step),_) : Nat}
  Addition.associative(Nat.mul(digit,step),kept_sum,reduced_sum)

def split_reduced(+rest: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+kept: List<&2,Nat>,+digit: Nat,+more: List<&2,Nat>,
  +fits: {Nat.add(index,1n+List.length(&2,Bool,rest)) == List.length(&2,U32,shape) : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool},
  +sum: {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,False{}),shape),[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,True{}),shape),[],more)) ==
    ShapeLaws.digits_value(List.drop(&2,U32,shape,1n+index),merged(rest,kept,more)) : Nat}) ->
  {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(True{} <> rest,index,False{}),shape),[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(True{} <> rest,index,True{}),shape),[],digit <> more)) ==
    ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged(True{} <> rest,kept,digit <> more)) : Nat}:
  +after = List.drop(&2,U32,shape,1n+index)
  +step = U32.to_nat(Grid.product(after))
  +kept_sum = TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,False{}),shape),[],kept)
  +reduced_sum = TiledViewReads.digit_sum(Reduce.order_axes(flagged(rest,1n+index,True{}),shape),[],more)
  +value = ShapeLaws.digits_value(after,merged(rest,kept,more))
  +inside = {%fits : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,List.length(&2,Bool,rest)) : {Nat.is_lt(index,List.length(&2,U32,shape)) == True{} : Bool}}
  +product = Public.product_exact(cap,cap_ok,after,CountCompletion.suffix_room(shape,1n+index,U32.to_nat(cap),positive,room))
  %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> after,Reduce.drop_next(shape,index,inside)) :
    {Nat.add(kept_sum,Nat.add(Nat.mul(digit,step),reduced_sum)) == ShapeLaws.digits_value(_,digit <> merged(rest,kept,more)) : Nat}
  %product : {Nat.add(kept_sum,Nat.add(Nat.mul(digit,step),reduced_sum)) == Nat.add(Nat.mul(digit,_),value) : Nat}
  %sum : {Nat.add(kept_sum,Nat.add(Nat.mul(digit,step),reduced_sum)) == Nat.add(Nat.mul(digit,step),_) : Nat}
  Equal.trans(Nat,Nat.add(kept_sum,Nat.add(Nat.mul(digit,step),reduced_sum)),Nat.add(Nat.add(kept_sum,Nat.mul(digit,step)),reduced_sum),
    Nat.add(Nat.mul(digit,step),Nat.add(kept_sum,reduced_sum)),
    Equal.sym(Nat,Nat.add(Nat.add(kept_sum,Nat.mul(digit,step)),reduced_sum),Nat.add(kept_sum,Nat.add(Nat.mul(digit,step),reduced_sum)),Addition.associative(kept_sum,Nat.mul(digit,step),reduced_sum)),
    Equal.trans(Nat,Nat.add(Nat.add(kept_sum,Nat.mul(digit,step)),reduced_sum),Nat.add(Nat.add(Nat.mul(digit,step),kept_sum),reduced_sum),Nat.add(Nat.mul(digit,step),Nat.add(kept_sum,reduced_sum)),
      Equal.cong(Nat,Nat,first => Nat.add(first,reduced_sum),Nat.add(kept_sum,Nat.mul(digit,step)),Nat.add(Nat.mul(digit,step),kept_sum),Order.commutative(kept_sum,Nat.mul(digit,step))),
      Addition.associative(Nat.mul(digit,step),kept_sum,reduced_sum)))

# The steps of the kept axes at their digits and of the listed axes at theirs
# number the digits put back in axis order.
def split_sum(flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,kept: List<&2,Nat>,reduced: List<&2,Nat>,
  +kept_below: Coordinates.DigitsBelow(Layout.extents_at(shape,flagged(flags,index,False{})),kept),
  +reduced_below: Coordinates.DigitsBelow(Layout.extents_at(shape,flagged(flags,index,True{})),reduced),
  +fits: {Nat.add(index,List.length(&2,Bool,flags)) == List.length(&2,U32,shape) : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(flags,index,False{}),shape),[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(flags,index,True{}),shape),[],reduced)) == ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged(flags,kept,reduced)) : Nat}:
  match flags:
    case []:
      match kept reduced:
        case Nil{} Nil{}: Equal.sym(Nat,ShapeLaws.digits_value(List.drop(&2,U32,shape,index),[]),0n,digits_none(List.drop(&2,U32,shape,index)))
        case Con{+digit,+more} +reduced:
          Empty.absurd({Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged([],index,False{}),shape),[],digit <> more),TiledViewReads.digit_sum(Reduce.order_axes(flagged([],index,True{}),shape),[],reduced)) ==
            ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged([],digit <> more,reduced)) : Nat},kept_below)
        case Nil{} Con{+digit,+more}:
          Empty.absurd({Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged([],index,False{}),shape),[],[]),TiledViewReads.digit_sum(Reduce.order_axes(flagged([],index,True{}),shape),[],digit <> more)) ==
            ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged([],[],digit <> more)) : Nat},reduced_below)
    case +flag <> +rest:
      match flag kept reduced:
        case True{} +kept Nil{}:
          Empty.absurd({Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(True{} <> rest,index,False{}),shape),[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(True{} <> rest,index,True{}),shape),[],[])) ==
            ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged(True{} <> rest,kept,[])) : Nat},reduced_below)
        case True{} +kept Con{+digit,+more}:
          KernelGemm.Both{here,+later} = reduced_below
          split_reduced(rest,index,shape,kept,digit,more,fits,cap,cap_ok,positive,room,
            split_sum(rest,1n+index,shape,kept,more,kept_below,later,
              Equal.trans(Nat,Nat.add(1n+index,List.length(&2,Bool,rest)),Nat.add(index,1n+List.length(&2,Bool,rest)),List.length(&2,U32,shape),
                Equal.sym(Nat,Nat.add(index,1n+List.length(&2,Bool,rest)),1n+Nat.add(index,List.length(&2,Bool,rest)),Addition.successor_right(index,List.length(&2,Bool,rest))),fits),
              cap,cap_ok,positive,room))
        case False{} Nil{} +reduced:
          Empty.absurd({Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(flagged(False{} <> rest,index,False{}),shape),[],[]),TiledViewReads.digit_sum(Reduce.order_axes(flagged(False{} <> rest,index,True{}),shape),[],reduced)) ==
            ShapeLaws.digits_value(List.drop(&2,U32,shape,index),merged(False{} <> rest,[],reduced)) : Nat},kept_below)
        case False{} Con{+digit,+more} +reduced:
          KernelGemm.Both{here,+later} = kept_below
          split_kept(rest,index,shape,digit,more,reduced,fits,cap,cap_ok,positive,room,
            split_sum(rest,1n+index,shape,more,reduced,later,reduced_below,
              Equal.trans(Nat,Nat.add(1n+index,List.length(&2,Bool,rest)),Nat.add(index,1n+List.length(&2,Bool,rest)),List.length(&2,U32,shape),
                Equal.sym(Nat,Nat.add(index,1n+List.length(&2,Bool,rest)),1n+Nat.add(index,List.length(&2,Bool,rest)),Addition.successor_right(index,List.length(&2,Bool,rest))),fits),
              cap,cap_ok,positive,room))

# The value a reduction reads at the digits of the kept axes and of the
# listed axes is the value at those digits put back in axis order.
def line_address(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+axes: List<&2,Nat>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +kept_below: Coordinates.DigitsBelow(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),kept),+reduced_below: Coordinates.DigitsBelow(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),reduced),+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool}) ->
  {Reduce.through(~Element,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))},
      U32.from_nat(Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),kept),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),reduced)))) ==
    original(logical,U32.from_nat(ShapeLaws.digits_value(shape,merged(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),kept,reduced)))) : Element}:
  +rank = List.length(&2,U32,shape)
  +flags = Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)
  +order = List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))
  +view_axes = Reduce.order_axes(order,shape)
  +kept_axes = Reduce.order_axes(Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),shape)
  +reduced_axes = Reduce.order_axes(Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}),shape)
  +number = Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),kept),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),reduced))
  +joined = List.append(&2,Nat,kept,reduced)
  +merged_digits = merged(flags,kept,reduced)
  +extents_same = Equal.trans(List<&2,U32>,Views.axis_extents(view_axes),Layout.extents_at(shape,order),List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),
    Reduce.order_axes_extents(order,shape),Reduce.appended_extents(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))
  +view_below = {%Equal.sym(List<&2,U32>,Views.axis_extents(view_axes),List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),extents_same) : Coordinates.DigitsBelow(_,joined)
    Coordinates.digits_joined(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),kept,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),reduced,kept_below,reduced_below) : Coordinates.DigitsBelow(Views.axis_extents(view_axes),joined)}
  +number_view = {%Equal.sym(List<&2,U32>,Views.axis_extents(view_axes),List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),extents_same) : {ShapeLaws.digits_value(_,joined) == number : Nat}
    Coordinates.digits_appended(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),kept,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),reduced,kept_below) : {ShapeLaws.digits_value(Views.axis_extents(view_axes),joined) == number : Nat}}
  +cells_same = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(view_axes)),ShapeLaws.elements(List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),ShapeLaws.elements(shape),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Views.axis_extents(view_axes),List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),extents_same),
    Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),ShapeLaws.elements(shape),
      ShapeLaws.elements_appended(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),
      Equal.trans(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})))),ShapeLaws.elements(List.drop(&2,U32,shape,0n)),ShapeLaws.elements(shape),
        Reduce.partition_elements(shape,rank,0n,axes,{==}),
        Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,shape,0n),shape,Reduce.dropped_none(~U32,shape)))))
  +number_inside = {%number_view : {Nat.is_lt(_,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}
    Coordinates.digits_below(Views.axis_extents(view_axes),joined,view_below) : {Nat.is_lt(number,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}}
  +number_room = Order.transitive(number,ShapeLaws.elements(shape),U32.to_nat(cap),
    Bounds.strict_le(number,ShapeLaws.elements(shape),{%cells_same : {Nat.is_lt(number,_) == True{} : Bool} number_inside : {Nat.is_lt(number,ShapeLaws.elements(shape)) == True{} : Bool}}),room)
  +number_exact = FullRange.roundtrip(number,cap,number_room)
  +kept_view_below = {%Equal.sym(List<&2,U32>,Views.axis_extents(kept_axes),Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),Reduce.order_axes_extents(Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),shape)) : Coordinates.DigitsBelow(_,kept) kept_below : Coordinates.DigitsBelow(Views.axis_extents(kept_axes),kept)}
  +split = split_sum(flags,0n,shape,kept,reduced,
    {%where_flagged(rank,0n,axes,False{}) : Coordinates.DigitsBelow(Layout.extents_at(shape,_),kept) kept_below : Coordinates.DigitsBelow(Layout.extents_at(shape,flagged(flags,0n,False{})),kept)},
    {%where_flagged(rank,0n,axes,True{}) : Coordinates.DigitsBelow(Layout.extents_at(shape,_),reduced) reduced_below : Coordinates.DigitsBelow(Layout.extents_at(shape,flagged(flags,0n,True{})),reduced)},
    flags_length(rank,0n,axes),cap,cap_ok,positive,room)
  +split_kept = {%Equal.sym(List<&2,Nat>,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),flagged(flags,0n,False{}),where_flagged(rank,0n,axes,False{})) :
      {Nat.add(TiledViewReads.digit_sum(Reduce.order_axes(_,shape),[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(flags,0n,True{}),shape),[],reduced)) ==
        ShapeLaws.digits_value(List.drop(&2,U32,shape,0n),merged_digits) : Nat}
    split : {Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(Reduce.order_axes(flagged(flags,0n,True{}),shape),[],reduced)) ==
      ShapeLaws.digits_value(List.drop(&2,U32,shape,0n),merged_digits) : Nat}}
  +split_both = {%Equal.sym(List<&2,Nat>,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}),flagged(flags,0n,True{}),where_flagged(rank,0n,axes,True{})) :
      {Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(Reduce.order_axes(_,shape),[],reduced)) == ShapeLaws.digits_value(List.drop(&2,U32,shape,0n),merged_digits) : Nat}
    split_kept : {Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)) == ShapeLaws.digits_value(List.drop(&2,U32,shape,0n),merged_digits) : Nat}}
  +split_shape = {%Reduce.dropped_none(~U32,shape) : {Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)) == ShapeLaws.digits_value(_,merged_digits) : Nat}
    split_both : {Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)) == ShapeLaws.digits_value(shape,merged_digits) : Nat}}
  +view_digits = {%number_view : {ViewReads.view_sum(view_axes,[],_) == TiledViewReads.digit_sum(view_axes,[],joined) : Nat}
    TiledViewReads.view_sum_digits(view_axes,[],joined,view_below) : {ViewReads.view_sum(view_axes,[],number) == TiledViewReads.digit_sum(view_axes,[],joined) : Nat}}
  +view_split = {%Equal.sym(List<&2,Views.Axis>,view_axes,List.append(&2,Views.Axis,kept_axes,reduced_axes),order_appended(Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}),shape)) :
      {TiledViewReads.digit_sum(_,[],joined) == Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)) : Nat}
    sum_appended(kept_axes,kept,reduced_axes,reduced,kept_view_below) :
      {TiledViewReads.digit_sum(view_axes,[],joined) == Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)) : Nat}}
  +chain = Equal.trans(Nat,ViewReads.view_sum(view_axes,[],number),TiledViewReads.digit_sum(view_axes,[],joined),ShapeLaws.digits_value(shape,merged_digits),view_digits,
    Equal.trans(Nat,TiledViewReads.digit_sum(view_axes,[],joined),Nat.add(TiledViewReads.digit_sum(kept_axes,[],kept),TiledViewReads.digit_sum(reduced_axes,[],reduced)),ShapeLaws.digits_value(shape,merged_digits),
      view_split,split_shape))
  %Equal.sym(Maybe<&2,Views.View>,Layout.line_view(shape,order),Some{Views.View{view_axes,0}},Reduce.line_picked(shape,axes)) :
    {Reduce.through(~Element,~Logical,~original,Reduce.Ordered{logical,_},U32.from_nat(number)) == original(logical,U32.from_nat(ShapeLaws.digits_value(shape,merged_digits))) : Element}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,number_exact) :
    {original(logical,(Views.mirrored_offset(view_axes,[],0) + Box.stepped_sum(Views.axis_extents(view_axes),ViewPart.view_steps(Views.mirrored_axes(view_axes,[])),_) : U32)) ==
      original(logical,U32.from_nat(ShapeLaws.digits_value(shape,merged_digits))) : Element}
  %Equal.sym(U32,(Views.mirrored_offset(view_axes,[],0) + Box.stepped_sum(Views.axis_extents(view_axes),ViewPart.view_steps(Views.mirrored_axes(view_axes,[])),number) : U32),
      U32.from_nat(Nat.add(U32.to_nat(0),ViewReads.view_sum(view_axes,[],number))),ViewReads.view_address(view_axes,[],0,number,number_inside)) :
    {original(logical,_) == original(logical,U32.from_nat(ShapeLaws.digits_value(shape,merged_digits))) : Element}
  %Equal.sym(Nat,ViewReads.view_sum(view_axes,[],number),ShapeLaws.digits_value(shape,merged_digits),chain) :
    {original(logical,U32.from_nat(_)) == original(logical,U32.from_nat(ShapeLaws.digits_value(shape,merged_digits))) : Element}
  {==}

# ---- The axes of the cells of a reduction ----

def predecessor(value: Nat) -> Nat:
  match value:
    case 0n: 0n
    case 1n+rest: rest

def successor_same(+left: Nat,+right: Nat,+same: {1n+left == 1n+right : Nat}) -> {left == right : Nat}:
  Equal.cong(Nat,Nat,value => predecessor(value),1n+left,1n+right,same)

def flag_inside(+index: Nat,+rest: List<&2,Bool>,+shape: List<&2,U32>,+fits: {Nat.add(index,1n+List.length(&2,Bool,rest)) == List.length(&2,U32,shape) : Nat}) ->
  {Nat.is_lt(index,List.length(&2,U32,shape)) == True{} : Bool}:
  %fits : {Nat.is_lt(index,_) == True{} : Bool}
  Order.below_sum(index,List.length(&2,Bool,rest))

def flag_next(+index: Nat,+rest: List<&2,Bool>,+shape: List<&2,U32>,+fits: {Nat.add(index,1n+List.length(&2,Bool,rest)) == List.length(&2,U32,shape) : Nat}) ->
  {Nat.add(1n+index,List.length(&2,Bool,rest)) == List.length(&2,U32,shape) : Nat}:
  Equal.trans(Nat,Nat.add(1n+index,List.length(&2,Bool,rest)),Nat.add(index,1n+List.length(&2,Bool,rest)),List.length(&2,U32,shape),
    Equal.sym(Nat,Nat.add(index,1n+List.length(&2,Bool,rest)),1n+Nat.add(index,List.length(&2,Bool,rest)),Addition.successor_right(index,List.length(&2,Bool,rest))),fits)

# The words of the axes from a word on whose flag is set are those axes.
def words_flagged(flags: List<&2,Bool>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(index),List.length(&2,Bool,flags)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Layout.naturals(TiledReduce.flagged_words(flags,index)) == flagged(flags,U32.to_nat(index),True{}) : List<&2,Nat>}:
  match flags:
    case []: {==}
    case +flag <> +rest:
      match flag:
        case True{}:
          +start = U32.to_nat(index)
          +count = List.length(&2,Bool,rest)
          +next = Padding.successor_exact(cap,cap_ok,index,start,{==},Bounds.nat_trans(start,Nat.add(start,1n+count),U32.to_nat(cap),Order.below_sum(start,count),room))
          +later_room = {%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+start,next) : {Nat.is_le(Nat.add(_,count),U32.to_nat(cap)) == True{} : Bool}
            %Addition.successor_right(start,count) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
            {Nat.is_le(Nat.add(U32.to_nat((index + 1 : U32)),count),U32.to_nat(cap)) == True{} : Bool}}
          %next :
            {start <> Layout.naturals(TiledReduce.flagged_words(rest,(index + 1 : U32))) == start <> flagged(rest,_,True{}) : List<&2,Nat>}
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => start <> tail,Layout.naturals(TiledReduce.flagged_words(rest,(index + 1 : U32))),flagged(rest,U32.to_nat((index + 1 : U32)),True{}),
            words_flagged(rest,(index + 1 : U32),cap,cap_ok,later_room))
        case False{}:
          +start = U32.to_nat(index)
          +count = List.length(&2,Bool,rest)
          +next = Padding.successor_exact(cap,cap_ok,index,start,{==},Bounds.nat_trans(start,Nat.add(start,1n+count),U32.to_nat(cap),Order.below_sum(start,count),room))
          +later_room = {%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+start,next) : {Nat.is_le(Nat.add(_,count),U32.to_nat(cap)) == True{} : Bool}
            %Addition.successor_right(start,count) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
            {Nat.is_le(Nat.add(U32.to_nat((index + 1 : U32)),count),U32.to_nat(cap)) == True{} : Bool}}
          %next :
            {Layout.naturals(TiledReduce.flagged_words(rest,(index + 1 : U32))) == flagged(rest,_,True{}) : List<&2,Nat>}
          words_flagged(rest,(index + 1 : U32),cap,cap_ok,later_room)

# No axis below the start is flagged from it on.
def flagged_above(flags: List<&2,Bool>,+index: Nat,+listed: Bool,+axis: Nat,+below: {Nat.is_lt(axis,index) == True{} : Bool}) ->
  {Views.contains_index(flagged(flags,index,listed),axis) == False{} : Bool}:
  match flags:
    case []: {==}
    case flag <> +rest:
      Reduce.chosen_absent(flag,listed,index,flagged(rest,1n+index,listed),axis,
        Equal.trans(Bool,Nat.is_eq(index,axis),Nat.is_eq(axis,index),False{},Order.equal_symmetric(index,axis),Order.less_not_equal(axis,index,below)),
        flagged_above(rest,1n+index,listed,axis,Order.lt_succ(axis,index,Bounds.strict_le(axis,index,below))))

# Every axis of a list lies below a rank.
def below_all(axes: List<&2,Nat>,+rank: Nat) -> Bool:
  match axes:
    case []: True{}
    case axis <> rest: Nat.is_lt(axis,rank) && below_all(rest,rank)

# Words lie below a rank as their numbers do.
def below_naturals(words: List<&2,U32>,+rank: Nat) -> {Layout.all_below(words,rank) == below_all(Layout.naturals(words),rank) : Bool}:
  match words:
    case []: {==}
    case +word <> +rest:
      Equal.cong(Bool,Bool,later => Nat.is_lt(U32.to_nat(word),rank) && later,Layout.all_below(rest,rank),below_all(Layout.naturals(rest),rank),below_naturals(rest,rank))

def chosen_below(member: Bool,listed: Bool,+index: Nat,+rest: List<&2,Nat>,+rank: Nat,+inside: {Nat.is_lt(index,rank) == True{} : Bool},+later: {below_all(rest,rank) == True{} : Bool}) ->
  {below_all(Layout.chosen(member,listed,index,rest),rank) == True{} : Bool}:
  match member listed:
    case True{} True{}: %Equal.sym(Bool,Nat.is_lt(index,rank),True{},inside) : {_ && below_all(rest,rank) == True{} : Bool} later
    case False{} False{}: %Equal.sym(Bool,Nat.is_lt(index,rank),True{},inside) : {_ && below_all(rest,rank) == True{} : Bool} later
    case True{} False{}: later
    case False{} True{}: later

def chosen_unique(member: Bool,listed: Bool,+index: Nat,+rest: List<&2,Nat>,+absent: {Views.contains_index(rest,index) == False{} : Bool},+later: {Views.unique_axes(rest) == True{} : Bool}) ->
  {Views.unique_axes(Layout.chosen(member,listed,index,rest)) == True{} : Bool}:
  match member listed:
    case True{} True{}: %Equal.sym(Bool,Views.contains_index(rest,index),False{},absent) : {Bool.not(_) && Views.unique_axes(rest) == True{} : Bool} later
    case False{} False{}: %Equal.sym(Bool,Views.contains_index(rest,index),False{},absent) : {Bool.not(_) && Views.unique_axes(rest) == True{} : Bool} later
    case True{} False{}: later
    case False{} True{}: later

# The axes flagged from a start on lie below the start and the count of the
# flags.
def flagged_below(flags: List<&2,Bool>,+index: Nat,+listed: Bool,+rank: Nat,+fits: {Nat.is_le(Nat.add(index,List.length(&2,Bool,flags)),rank) == True{} : Bool}) ->
  {below_all(flagged(flags,index,listed),rank) == True{} : Bool}:
  match flags:
    case []: {==}
    case +flag <> +rest:
      +count = List.length(&2,Bool,rest)
      +inside = Bounds.nat_trans(index,Nat.add(index,1n+count),rank,Order.below_sum(index,count),fits)
      +later_fits = {%Addition.successor_right(index,count) : {Nat.is_le(_,rank) == True{} : Bool} fits : {Nat.is_le(Nat.add(1n+index,count),rank) == True{} : Bool}}
      chosen_below(flag,listed,index,flagged(rest,1n+index,listed),rank,inside,flagged_below(rest,1n+index,listed,rank,later_fits))

# The axes flagged from a start on are unique.
def flagged_unique(flags: List<&2,Bool>,+index: Nat,+listed: Bool) -> {Views.unique_axes(flagged(flags,index,listed)) == True{} : Bool}:
  match flags:
    case []: {==}
    case +flag <> +rest:
      chosen_unique(flag,listed,index,flagged(rest,1n+index,listed),flagged_above(rest,1n+index,listed,index,Order.lt_succ(index,index,Order.reflexive(index))),
        flagged_unique(rest,1n+index,listed))

# An axis below the start leaves the flags from it on.
def flags_skip(remaining: Nat,+index: Nat,+axis: Nat,+rest: List<&2,Nat>,+below: {Nat.is_lt(axis,index) == True{} : Bool}) ->
  {Layout.reduced_flags(remaining,index,axis <> rest) == Layout.reduced_flags(remaining,index,rest) : List<&2,Bool>}:
  match remaining:
    case 0n: {==}
    case 1n+ +more:
      %Equal.sym(Bool,Nat.is_eq(axis,index),False{},Order.less_not_equal(axis,index,below)) :
        {(_ || Views.contains_index(rest,index)) <> Layout.reduced_flags(more,1n+index,axis <> rest) == Views.contains_index(rest,index) <> Layout.reduced_flags(more,1n+index,rest) : List<&2,Bool>}
      Equal.cong(List<&2,Bool>,List<&2,Bool>,tail => Views.contains_index(rest,index) <> tail,Layout.reduced_flags(more,1n+index,axis <> rest),Layout.reduced_flags(more,1n+index,rest),
        flags_skip(more,1n+index,axis,rest,Order.lt_succ(axis,index,Bounds.strict_le(axis,index,below))))

# The flags of the axes flagged from the start on are the flags.
def flags_flagged(flags: List<&2,Bool>,+index: Nat) -> {Layout.reduced_flags(List.length(&2,Bool,flags),index,flagged(flags,index,True{})) == flags : List<&2,Bool>}:
  match flags:
    case []: {==}
    case +flag <> +rest:
      match flag:
        case True{}:
          +later = flagged(rest,1n+index,True{})
          +count = List.length(&2,Bool,rest)
          %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) :
            {(_ || Views.contains_index(later,index)) <> Layout.reduced_flags(count,1n+index,index <> later) == True{} <> rest : List<&2,Bool>}
          %Equal.sym(List<&2,Bool>,Layout.reduced_flags(count,1n+index,index <> later),Layout.reduced_flags(count,1n+index,later),flags_skip(count,1n+index,index,later,Order.lt_succ(index,index,Order.reflexive(index)))) :
            {True{} <> _ == True{} <> rest : List<&2,Bool>}
          Equal.cong(List<&2,Bool>,List<&2,Bool>,tail => True{} <> tail,Layout.reduced_flags(count,1n+index,later),rest,flags_flagged(rest,1n+index))
        case False{}:
          +later = flagged(rest,1n+index,True{})
          +count = List.length(&2,Bool,rest)
          %Equal.sym(Bool,Views.contains_index(later,index),False{},flagged_above(rest,1n+index,True{},index,Order.lt_succ(index,index,Order.reflexive(index)))) :
            {_ <> Layout.reduced_flags(count,1n+index,later) == False{} <> rest : List<&2,Bool>}
          Equal.cong(List<&2,Bool>,List<&2,Bool>,tail => False{} <> tail,Layout.reduced_flags(count,1n+index,later),rest,flags_flagged(rest,1n+index))

# ---- The extents of the flagged axes ----

def kept_extent(flag: Bool,+listed: Bool,+extent: U32,rest: List<&2,U32>) -> List<&2,U32>:
  match flag listed:
    case True{} True{}: extent <> rest
    case False{} False{}: extent <> rest
    case flag listed: rest

# The extents of the axes whose flag is listed.
def picked_extents(extents: List<&2,U32>,flags: List<&2,Bool>,+listed: Bool) -> List<&2,U32>:
  match extents flags:
    case Con{extent,rest} Con{flag,more}: kept_extent(flag,listed,extent,picked_extents(rest,more,listed))
    case extents flags: []

def picked_none(extents: List<&2,U32>,+listed: Bool) -> {picked_extents(extents,[],listed) == [] : List<&2,U32>}:
  match extents:
    case []: {==}
    case extent <> rest: {==}

def extents_flagged(flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+listed: Bool,+fits: {Nat.add(index,List.length(&2,Bool,flags)) == List.length(&2,U32,shape) : Nat}) ->
  {Layout.extents_at(shape,flagged(flags,index,listed)) == picked_extents(List.drop(&2,U32,shape,index),flags,listed) : List<&2,U32>}:
  match flags:
    case []: Equal.sym(List<&2,U32>,picked_extents(List.drop(&2,U32,shape,index),[],listed),[],picked_none(List.drop(&2,U32,shape,index),listed))
    case +flag <> +rest:
      match flag listed:
        case True{} True{}:
          %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),Reduce.drop_next(shape,index,flag_inside(index,rest,shape,fits))) :
            {Layout.extents_at(shape,flagged(True{} <> rest,index,True{})) == picked_extents(_,True{} <> rest,True{}) : List<&2,U32>}
          Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(shape,index) <> tail,Layout.extents_at(shape,flagged(rest,1n+index,True{})),picked_extents(List.drop(&2,U32,shape,1n+index),rest,True{}),
            extents_flagged(rest,1n+index,shape,True{},flag_next(index,rest,shape,fits)))
        case False{} False{}:
          %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),Reduce.drop_next(shape,index,flag_inside(index,rest,shape,fits))) :
            {Layout.extents_at(shape,flagged(False{} <> rest,index,False{})) == picked_extents(_,False{} <> rest,False{}) : List<&2,U32>}
          Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(shape,index) <> tail,Layout.extents_at(shape,flagged(rest,1n+index,False{})),picked_extents(List.drop(&2,U32,shape,1n+index),rest,False{}),
            extents_flagged(rest,1n+index,shape,False{},flag_next(index,rest,shape,fits)))
        case True{} False{}:
          %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),Reduce.drop_next(shape,index,flag_inside(index,rest,shape,fits))) :
            {Layout.extents_at(shape,flagged(True{} <> rest,index,False{})) == picked_extents(_,True{} <> rest,False{}) : List<&2,U32>}
          extents_flagged(rest,1n+index,shape,False{},flag_next(index,rest,shape,fits))
        case False{} True{}:
          %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),Reduce.drop_next(shape,index,flag_inside(index,rest,shape,fits))) :
            {Layout.extents_at(shape,flagged(False{} <> rest,index,True{})) == picked_extents(_,False{} <> rest,True{}) : List<&2,U32>}
          extents_flagged(rest,1n+index,shape,True{},flag_next(index,rest,shape,fits))

def kept_appended(flag: Bool,+listed: Bool,+extent: U32,+joined: List<&2,U32>,+rest: List<&2,U32>,+more: List<&2,U32>,+same: {joined == List.append(&2,U32,rest,more) : List<&2,U32>}) ->
  {kept_extent(flag,listed,extent,joined) == List.append(&2,U32,kept_extent(flag,listed,extent,rest),more) : List<&2,U32>}:
  match flag listed:
    case True{} True{}: Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,joined,List.append(&2,U32,rest,more),same)
    case False{} False{}: Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,joined,List.append(&2,U32,rest,more),same)
    case True{} False{}: same
    case False{} True{}: same

def picked_appended(extents: List<&2,U32>,flags: List<&2,Bool>,+more_extents: List<&2,U32>,+more_flags: List<&2,Bool>,+listed: Bool,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {picked_extents(List.append(&2,U32,extents,more_extents),List.append(&2,Bool,flags,more_flags),listed) ==
    List.append(&2,U32,picked_extents(extents,flags,listed),picked_extents(more_extents,more_flags,listed)) : List<&2,U32>}:
  match extents flags:
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Con{+flag,+more}:
      kept_appended(flag,listed,extent,picked_extents(List.append(&2,U32,rest,more_extents),List.append(&2,Bool,more,more_flags),listed),picked_extents(rest,more,listed),picked_extents(more_extents,more_flags,listed),
        picked_appended(rest,more,more_extents,more_flags,listed,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Nil{} Con{+flag,+more}:
      Empty.absurd({picked_extents(List.append(&2,U32,Nil{},more_extents),List.append(&2,Bool,Con{flag,more},more_flags),listed) ==
        List.append(&2,U32,picked_extents(Nil{},Con{flag,more},listed),picked_extents(more_extents,more_flags,listed)) : List<&2,U32>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({picked_extents(List.append(&2,U32,Con{extent,rest},more_extents),List.append(&2,Bool,Nil{},more_flags),listed) ==
        List.append(&2,U32,picked_extents(Con{extent,rest},Nil{},listed),picked_extents(more_extents,more_flags,listed)) : List<&2,U32>},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# The extents of the cells with the flag of the groups at their position.
def placed_extents(position: Grid.Position,+extents: List<&2,U32>,+flag: Bool,+listed: Bool,+groups: U32) -> List<&2,U32>:
  match position:
    case Grid.Innermost{}: List.append(&2,U32,extents,kept_extent(flag,listed,groups,[]))
    case Grid.Outermost{}: kept_extent(flag,listed,groups,extents)

def picked_placed(position: Grid.Position,+extents: List<&2,U32>,+flags: List<&2,Bool>,+flag: Bool,+listed: Bool,+groups: U32,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {picked_extents(Grid.placed(True{},position,extents,groups),TiledReduce.flags_placed(position,flags,flag),listed) == placed_extents(position,picked_extents(extents,flags,listed),flag,listed,groups) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}: picked_appended(extents,flags,[groups],[flag],listed,same)
    case Grid.Outermost{}: {==}

# ---- The lane axis taken out ----

# The kept axes before an axis.
def kept_before(flags: List<&2,Bool>,axis: Nat) -> Nat:
  match flags axis:
    case Con{flag,+rest} 1n+ +next: Bool.pick(Nat,flag,kept_before(rest,next),1n+kept_before(rest,next))
    case flags axis: 0n

def removed_length(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {List.length(&2,U32,Grid.removed(extents,axis)) == List.length(&2,Bool,Partials.flag_removed(flags,axis)) : Nat}:
  match extents flags axis:
    case Con{+extent,+rest} Con{+flag,+more} 0n: successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)
    case Con{+extent,+rest} Con{+flag,+more} 1n+ +next:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Grid.removed(rest,next)),List.length(&2,Bool,Partials.flag_removed(more,next)),removed_length(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Nil{} Nil{} +axis: {==}
    case Nil{} Con{+flag,+more} +axis: Empty.absurd({List.length(&2,U32,Grid.removed(Nil{},axis)) == List.length(&2,Bool,Partials.flag_removed(Con{flag,more},axis)) : Nat},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({List.length(&2,U32,Grid.removed(Con{extent,rest},axis)) == List.length(&2,Bool,Partials.flag_removed(Nil{},axis)) : Nat},Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# Taking out a kept axis leaves the listed extents.
def removed_unpicked(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool}) ->
  {picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),True{}) == picked_extents(extents,flags,True{}) : List<&2,U32>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n: Empty.absurd({picked_extents(rest,more,True{}) == picked_extents(Con{extent,rest},Con{True{},more},True{}) : List<&2,U32>},WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{+extent,+rest} Con{False{},+more} 0n: {==}
    case Con{+extent,+rest} Con{+flag,+more} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => kept_extent(flag,True{},extent,tail),picked_extents(Grid.removed(rest,next),Partials.flag_removed(more,next),True{}),picked_extents(rest,more,True{}),
        removed_unpicked(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged))
    case Nil{} Nil{} +axis: {==}
    case Nil{} Con{+flag,+more} +axis:
      Empty.absurd({picked_extents(Grid.removed(Nil{},axis),Partials.flag_removed(Con{flag,more},axis),True{}) == picked_extents(Nil{},Con{flag,more},True{}) : List<&2,U32>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({picked_extents(Grid.removed(Con{extent,rest},axis),Partials.flag_removed(Nil{},axis),True{}) == picked_extents(Con{extent,rest},Nil{},True{}) : List<&2,U32>},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# Taking out a reduced axis leaves the kept extents.
def removed_kept(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+flagged: {Partials.flag_at(flags,axis) == True{} : Bool}) ->
  {picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),False{}) == picked_extents(extents,flags,False{}) : List<&2,U32>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{False{},+more} 0n: Empty.absurd({picked_extents(rest,more,False{}) == picked_extents(Con{extent,rest},Con{False{},more},False{}) : List<&2,U32>},WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{True{},+more} 0n: {==}
    case Con{+extent,+rest} Con{+flag,+more} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => kept_extent(flag,False{},extent,tail),picked_extents(Grid.removed(rest,next),Partials.flag_removed(more,next),False{}),picked_extents(rest,more,False{}),
        removed_kept(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),flagged))
    case Nil{} Nil{} +axis: {==}
    case Nil{} Con{+flag,+more} +axis:
      Empty.absurd({picked_extents(Grid.removed(Nil{},axis),Partials.flag_removed(Con{flag,more},axis),False{}) == picked_extents(Nil{},Con{flag,more},False{}) : List<&2,U32>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({picked_extents(Grid.removed(Con{extent,rest},axis),Partials.flag_removed(Nil{},axis),False{}) == picked_extents(Con{extent,rest},Nil{},False{}) : List<&2,U32>},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# Taking out a kept axis takes its place out of the kept extents.
def removed_picked(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool}) ->
  {picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),False{}) == Grid.removed(picked_extents(extents,flags,False{}),kept_before(flags,axis)) : List<&2,U32>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      Empty.absurd({picked_extents(rest,more,False{}) == Grid.removed(picked_extents(Con{extent,rest},Con{True{},more},False{}),0n) : List<&2,U32>},WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{+extent,+rest} Con{False{},+more} 0n: {==}
    case Con{+extent,+rest} Con{True{},+more} 1n+ +next: removed_picked(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged)
    case Con{+extent,+rest} Con{False{},+more} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,picked_extents(Grid.removed(rest,next),Partials.flag_removed(more,next),False{}),Grid.removed(picked_extents(rest,more,False{}),kept_before(more,next)),
        removed_picked(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged))
    case Nil{} Nil{} +axis: {==}
    case Nil{} Con{+flag,+more} +axis:
      Empty.absurd({picked_extents(Grid.removed(Nil{},axis),Partials.flag_removed(Con{flag,more},axis),False{}) == Grid.removed(picked_extents(Nil{},Con{flag,more},False{}),kept_before(Con{flag,more},axis)) : List<&2,U32>},
        Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({picked_extents(Grid.removed(Con{extent,rest},axis),Partials.flag_removed(Nil{},axis),False{}) == Grid.removed(picked_extents(Con{extent,rest},Nil{},False{}),kept_before(Nil{},axis)) : List<&2,U32>},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# A kept axis has its extent at its place among the kept extents.
def lanes_picked(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {Grid.lanes_at(picked_extents(extents,flags,False{}),kept_before(flags,axis)) == Grid.lanes_at(extents,axis) : U32}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      Empty.absurd({Grid.lanes_at(picked_extents(Con{extent,rest},Con{True{},more},False{}),0n) == extent : U32},WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{+extent,+rest} Con{False{},+more} 0n: {==}
    case Con{+extent,+rest} Con{True{},+more} 1n+ +next: lanes_picked(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside)
    case Con{+extent,+rest} Con{False{},+more} 1n+ +next: lanes_picked(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside)
    case Nil{} flags +axis:
      Empty.absurd({Grid.lanes_at(picked_extents(Nil{},flags,False{}),kept_before(flags,axis)) == Grid.lanes_at(Nil{},axis) : U32},Order.below_zero(axis,inside))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Grid.lanes_at(picked_extents(Con{extent,rest},Nil{},False{}),kept_before(Nil{},axis)) == Grid.lanes_at(Con{extent,rest},axis) : U32},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

def kept_inside(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {Nat.is_lt(kept_before(flags,axis),List.length(&2,U32,picked_extents(extents,flags,False{}))) == True{} : Bool}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      Empty.absurd({Nat.is_lt(0n,List.length(&2,U32,picked_extents(Con{extent,rest},Con{True{},more},False{}))) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{+extent,+rest} Con{False{},+more} 0n: {==}
    case Con{+extent,+rest} Con{True{},+more} 1n+ +next: kept_inside(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside)
    case Con{+extent,+rest} Con{False{},+more} 1n+ +next: kept_inside(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside)
    case Nil{} flags +axis:
      Empty.absurd({Nat.is_lt(kept_before(flags,axis),List.length(&2,U32,picked_extents(Nil{},flags,False{}))) == True{} : Bool},Order.below_zero(axis,inside))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Nat.is_lt(kept_before(Nil{},axis),List.length(&2,U32,picked_extents(Con{extent,rest},Nil{},False{}))) == True{} : Bool},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# The place of a kept axis among the kept axes is the count of kept axes
# before it.
def place_kept(flags: List<&2,Bool>,+index: Nat,axis: Nat,+base: Nat,+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool},+inside: {Nat.is_lt(axis,List.length(&2,Bool,flags)) == True{} : Bool}) ->
  {TiledReduce.kept_place(flagged(flags,index,False{}),Nat.add(index,axis),base) == Nat.add(kept_before(flags,axis),base) : Nat}:
  match flags axis:
    case Con{True{},+more} 0n:
      Empty.absurd({TiledReduce.kept_place(flagged(Con{True{},more},index,False{}),Nat.add(index,0n),base) == base : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{False{},+more} 0n:
      +later = flagged(more,1n+index,False{})
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) : {TiledReduce.kept_place(index <> later,_,base) == base : Nat}
      %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) : {Bool.pick(Nat,_,base,TiledReduce.kept_place(later,index,1n+base)) == base : Nat}
      {==}
    case Con{True{},+more} 1n+ +next:
      %Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)) :
        {TiledReduce.kept_place(flagged(more,1n+index,False{}),_,base) == Nat.add(kept_before(more,next),base) : Nat}
      place_kept(more,1n+index,next,base,unflagged,inside)
    case Con{False{},+more} 1n+ +next:
      +later = flagged(more,1n+index,False{})
      +before = kept_before(more,next)
      %Equal.sym(Bool,Nat.is_eq(index,Nat.add(index,1n+next)),False{},Order.less_not_equal(index,Nat.add(index,1n+next),Order.below_sum(index,next))) :
        {Bool.pick(Nat,_,base,TiledReduce.kept_place(later,Nat.add(index,1n+next),1n+base)) == 1n+Nat.add(before,base) : Nat}
      %Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)) :
        {TiledReduce.kept_place(later,_,1n+base) == 1n+Nat.add(before,base) : Nat}
      Equal.trans(Nat,TiledReduce.kept_place(later,Nat.add(1n+index,next),1n+base),Nat.add(before,1n+base),1n+Nat.add(before,base),
        place_kept(more,1n+index,next,1n+base,unflagged,inside),Addition.successor_right(before,base))
    case Nil{} +axis:
      Empty.absurd({TiledReduce.kept_place(flagged(Nil{},index,False{}),Nat.add(index,axis),base) == Nat.add(kept_before(Nil{},axis),base) : Nat},Order.below_zero(axis,inside))

# ---- The digits of the cells ----

# A last kept digit stays last.
def merged_last(extents: List<&2,U32>,flags: List<&2,Bool>,kept: List<&2,Nat>,reduced: List<&2,Nat>,+digit: Nat,
  +kept_below: Coordinates.DigitsBelow(picked_extents(extents,flags,False{}),kept),+reduced_below: Coordinates.DigitsBelow(picked_extents(extents,flags,True{}),reduced),
  +same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {merged(List.append(&2,Bool,flags,[False{}]),List.append(&2,Nat,kept,[digit]),reduced) == List.append(&2,Nat,merged(flags,kept,reduced),[digit]) : List<&2,Nat>}:
  match extents flags:
    case Nil{} Nil{}:
      match kept reduced:
        case Nil{} Nil{}: {==}
        case Con{+first,+more} +reduced:
          Empty.absurd({merged(List.append(&2,Bool,Nil{},[False{}]),List.append(&2,Nat,Con{first,more},[digit]),reduced) ==
            List.append(&2,Nat,merged(Nil{},Con{first,more},reduced),[digit]) : List<&2,Nat>},kept_below)
        case Nil{} Con{+first,+more}:
          Empty.absurd({merged(List.append(&2,Bool,Nil{},[False{}]),List.append(&2,Nat,Nil{},[digit]),Con{first,more}) ==
            List.append(&2,Nat,merged(Nil{},Nil{},Con{first,more}),[digit]) : List<&2,Nat>},reduced_below)
    case Con{+extent,+rest} Con{True{},+more}:
      match reduced:
        case Nil{}:
          Empty.absurd({merged(List.append(&2,Bool,Con{True{},more},[False{}]),List.append(&2,Nat,kept,[digit]),Nil{}) ==
            List.append(&2,Nat,merged(Con{True{},more},kept,Nil{}),[digit]) : List<&2,Nat>},reduced_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = reduced_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,merged(List.append(&2,Bool,more,[False{}]),List.append(&2,Nat,kept,[digit]),later),
            List.append(&2,Nat,merged(more,kept,later),[digit]),merged_last(rest,more,kept,later,digit,kept_below,tail_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Con{+extent,+rest} Con{False{},+more}:
      match kept:
        case Nil{}:
          Empty.absurd({merged(List.append(&2,Bool,Con{False{},more},[False{}]),List.append(&2,Nat,Nil{},[digit]),reduced) ==
            List.append(&2,Nat,merged(Con{False{},more},Nil{},reduced),[digit]) : List<&2,Nat>},kept_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = kept_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,merged(List.append(&2,Bool,more,[False{}]),List.append(&2,Nat,later,[digit]),reduced),
            List.append(&2,Nat,merged(more,later,reduced),[digit]),merged_last(rest,more,later,reduced,digit,tail_below,reduced_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Nil{} Con{+flag,+more}:
      Empty.absurd({merged(List.append(&2,Bool,Con{flag,more},[False{}]),List.append(&2,Nat,kept,[digit]),reduced) ==
        List.append(&2,Nat,merged(Con{flag,more},kept,reduced),[digit]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({merged(List.append(&2,Bool,Nil{},[False{}]),List.append(&2,Nat,kept,[digit]),reduced) ==
        List.append(&2,Nat,merged(Nil{},kept,reduced),[digit]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# The digits of the cells with the group of the kept lanes at its position.
def merged_placed(position: Grid.Position,+extents: List<&2,U32>,+flags: List<&2,Bool>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+group: Nat,
  +kept_below: Coordinates.DigitsBelow(picked_extents(extents,flags,False{}),kept),+reduced_below: Coordinates.DigitsBelow(picked_extents(extents,flags,True{}),reduced),
  +same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {merged(TiledReduce.flags_placed(position,flags,False{}),Coordinates.cell_digits(position,kept,group),reduced) ==
    Coordinates.cell_digits(position,merged(flags,kept,reduced),group) : List<&2,Nat>}:
  match position:
    case Grid.Innermost{}: merged_last(extents,flags,kept,reduced,group,kept_below,reduced_below,same)
    case Grid.Outermost{}: {==}

# The digit of the lanes put back at the lane axis.
def merged_inserted(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,kept: List<&2,Nat>,reduced: List<&2,Nat>,+digit: Nat,
  +kept_below: Coordinates.DigitsBelow(picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),False{}),kept),
  +reduced_below: Coordinates.DigitsBelow(picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),True{}),reduced),
  +same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+unflagged: {Partials.flag_at(flags,axis) == False{} : Bool},+inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {Coordinates.inserted_digit(axis,merged(Partials.flag_removed(flags,axis),kept,reduced),digit) == merged(flags,Coordinates.inserted_digit(kept_before(flags,axis),kept,digit),reduced) : List<&2,Nat>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      Empty.absurd({Coordinates.inserted_digit(0n,merged(more,kept,reduced),digit) == merged(Con{True{},more},Coordinates.inserted_digit(0n,kept,digit),reduced) : List<&2,Nat>},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},unflagged)))
    case Con{+extent,+rest} Con{False{},+more} 0n: {==}
    case Con{+extent,+rest} Con{True{},+more} 1n+ +next:
      match reduced:
        case Nil{}:
          Empty.absurd({Coordinates.inserted_digit(1n+next,merged(Con{True{},Partials.flag_removed(more,next)},kept,Nil{}),digit) ==
            merged(Con{True{},more},Coordinates.inserted_digit(kept_before(more,next),kept,digit),Nil{}) : List<&2,Nat>},reduced_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = reduced_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,Coordinates.inserted_digit(next,merged(Partials.flag_removed(more,next),kept,later),digit),
            merged(more,Coordinates.inserted_digit(kept_before(more,next),kept,digit),later),
            merged_inserted(rest,more,next,kept,later,digit,kept_below,tail_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside))
    case Con{+extent,+rest} Con{False{},+more} 1n+ +next:
      match kept:
        case Nil{}:
          Empty.absurd({Coordinates.inserted_digit(1n+next,merged(Con{False{},Partials.flag_removed(more,next)},Nil{},reduced),digit) ==
            merged(Con{False{},more},Coordinates.inserted_digit(1n+kept_before(more,next),Nil{},digit),reduced) : List<&2,Nat>},kept_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = kept_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,Coordinates.inserted_digit(next,merged(Partials.flag_removed(more,next),later,reduced),digit),
            merged(more,Coordinates.inserted_digit(kept_before(more,next),later,digit),reduced),
            merged_inserted(rest,more,next,later,reduced,digit,tail_below,reduced_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),unflagged,inside))
    case Nil{} +flags +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,merged(Partials.flag_removed(flags,axis),kept,reduced),digit) == merged(flags,Coordinates.inserted_digit(kept_before(flags,axis),kept,digit),reduced) : List<&2,Nat>},
        Order.below_zero(axis,inside))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,merged(Partials.flag_removed(Nil{},axis),kept,reduced),digit) == merged(Nil{},Coordinates.inserted_digit(kept_before(Nil{},axis),kept,digit),reduced) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# Kept and listed digits below their extents put back below the extents.
def merged_below(extents: List<&2,U32>,flags: List<&2,Bool>,kept: List<&2,Nat>,reduced: List<&2,Nat>,+kept_below: Coordinates.DigitsBelow(picked_extents(extents,flags,False{}),kept),+reduced_below: Coordinates.DigitsBelow(picked_extents(extents,flags,True{}),reduced),
  +same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) -> Coordinates.DigitsBelow(extents,merged(flags,kept,reduced)):
  match extents flags:
    case Nil{} Nil{}: Unit{}
    case Con{+extent,+rest} Con{True{},+more}:
      match reduced:
        case Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},merged(Con{True{},more},kept,Nil{})),reduced_below)
        case Con{+first,+later}:
          KernelGemm.Both{+here,+tail_below} = reduced_below
          KernelGemm.Both{here,merged_below(rest,more,kept,later,kept_below,tail_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same))}
    case Con{+extent,+rest} Con{False{},+more}:
      match kept:
        case Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},merged(Con{False{},more},Nil{},reduced)),kept_below)
        case Con{+first,+later}:
          KernelGemm.Both{+here,+tail_below} = kept_below
          KernelGemm.Both{here,merged_below(rest,more,later,reduced,tail_below,reduced_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same))}
    case Nil{} Con{+flag,+more}: Empty.absurd(Coordinates.DigitsBelow(Nil{},merged(Con{flag,more},kept,reduced)),Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},merged(Nil{},kept,reduced)),Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# ---- The cells of the kept lanes ----

# The flag of an axis is whether it is listed.
def flag_listed(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,axis: Nat,+inside: {Nat.is_lt(axis,remaining) == True{} : Bool}) ->
  {Partials.flag_at(Layout.reduced_flags(remaining,index,axes),axis) == Views.contains_index(axes,Nat.add(index,axis)) : Bool}:
  match remaining axis:
    case 0n +axis:
      Empty.absurd({Partials.flag_at(Layout.reduced_flags(0n,index,axes),axis) == Views.contains_index(axes,Nat.add(index,axis)) : Bool},Order.below_zero(axis,inside))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) : {Views.contains_index(axes,index) == Views.contains_index(axes,_) : Bool}
      {==}
    case 1n+ +rest 1n+ +next:
      %Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)) :
        {Partials.flag_at(Layout.reduced_flags(rest,1n+index,axes),next) == Views.contains_index(axes,_) : Bool}
      flag_listed(rest,1n+index,axes,next,inside)

def unflagged_of(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool}) -> {Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == False{} : Bool}:
  Equal.trans(Bool,Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Views.contains_index(axes,lane),False{},flag_listed(List.length(&2,U32,values),0n,axes,lane,within),absent)

def same_of(+values: List<&2,U32>,+axes: List<&2,Nat>) -> {List.length(&2,U32,values) == List.length(&2,Bool,Layout.reduced_flags(List.length(&2,U32,values),0n,axes)) : Nat}:
  Equal.sym(Nat,List.length(&2,Bool,Layout.reduced_flags(List.length(&2,U32,values),0n,axes)),List.length(&2,U32,values),flags_length(List.length(&2,U32,values),0n,axes))

# The extents of the axes where the listed flag is set.
def extents_listed(+values: List<&2,U32>,+axes: List<&2,Nat>,+listed: Bool) ->
  {Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,listed)) == picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),listed) : List<&2,U32>}:
  +rank = List.length(&2,U32,values)
  +flags = Layout.reduced_flags(List.length(&2,U32,values),0n,axes)
  Equal.trans(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(rank,0n,axes,listed)),Layout.extents_at(values,flagged(flags,0n,listed)),picked_extents(values,flags,listed),
    Equal.cong(List<&2,Nat>,List<&2,U32>,order => Layout.extents_at(values,order),Layout.axes_where(rank,0n,axes,listed),flagged(flags,0n,listed),where_flagged(rank,0n,axes,listed)),
    Equal.trans(List<&2,U32>,Layout.extents_at(values,flagged(flags,0n,listed)),picked_extents(List.drop(&2,U32,values,0n),flags,listed),picked_extents(values,flags,listed),
      extents_flagged(flags,0n,values,listed,flags_length(rank,0n,axes)),
      Equal.cong(List<&2,U32>,List<&2,U32>,items => picked_extents(items,flags,listed),List.drop(&2,U32,values,0n),values,Reduce.dropped_none(~U32,values))))

# The place of the lane axis among the kept axes.
def place_listed(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool}) -> {TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n) == kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) : Nat}:
  +rank = List.length(&2,U32,values)
  +flags = Layout.reduced_flags(List.length(&2,U32,values),0n,axes)
  +unflagged = Equal.trans(Bool,Partials.flag_at(flags,lane),Views.contains_index(axes,lane),False{},flag_listed(rank,0n,axes,lane,within),absent)
  +inside = {%Equal.sym(Nat,List.length(&2,Bool,flags),rank,flags_length(rank,0n,axes)) : {Nat.is_lt(lane,_) == True{} : Bool} within :
    {Nat.is_lt(lane,List.length(&2,Bool,flags)) == True{} : Bool}}
  %Equal.sym(List<&2,Nat>,Layout.axes_where(rank,0n,axes,False{}),flagged(flags,0n,False{}),where_flagged(rank,0n,axes,False{})) :
    {TiledReduce.kept_place(_,lane,0n) == kept_before(flags,lane) : Nat}
  Equal.trans(Nat,TiledReduce.kept_place(flagged(flags,0n,False{}),lane,0n),Nat.add(kept_before(flags,lane),0n),kept_before(flags,lane),
    place_kept(flags,0n,lane,0n,unflagged,inside),Addition.zero_right(kept_before(flags,lane)))

def kept_lanes(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool}) -> {Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)) == Grid.lanes_at(values,lane) : U32}:
  %Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),extents_listed(values,axes,False{})) : {Grid.lanes_at(_,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)) == Grid.lanes_at(values,lane) : U32}
  %Equal.sym(Nat,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),place_listed(values,axes,lane,within,absent)) : {Grid.lanes_at(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),_) == Grid.lanes_at(values,lane) : U32}
  lanes_picked(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same_of(values,axes),unflagged_of(values,axes,lane,position,within,absent),within)

def kept_within(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool}) -> {Nat.is_lt(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),List.length(&2,U32,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool}:
  %Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),extents_listed(values,axes,False{})) : {Nat.is_lt(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),List.length(&2,U32,_)) == True{} : Bool}
  %Equal.sym(Nat,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),place_listed(values,axes,lane,within,absent)) : {Nat.is_lt(_,List.length(&2,U32,picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}))) == True{} : Bool}
  kept_inside(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same_of(values,axes),unflagged_of(values,axes,lane,position,within,absent),within)

# The cells of values with their lanes inside them.
def cells_placed(+values: List<&2,U32>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Tiled.cell_shape(values,lane,position) == Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))) : List<&2,U32>}:
  %within : {Tiled.cell_shape(values,lane,position) == Grid.placed(_,position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))) : List<&2,U32>}
  {==}

def removed_count(extents: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {1n+List.length(&2,U32,Grid.removed(extents,axis)) == List.length(&2,U32,extents) : Nat}:
  match extents axis:
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next:
      Equal.cong(Nat,Nat,count => 1n+count,1n+List.length(&2,U32,Grid.removed(rest,next)),List.length(&2,U32,rest),removed_count(rest,next,inside))
    case Nil{} +axis: Empty.absurd({1n+List.length(&2,U32,Grid.removed(Nil{},axis)) == 0n : Nat},Order.below_zero(axis,inside))

def flags_count(position: Grid.Position,+flags: List<&2,Bool>,+flag: Bool) -> {List.length(&2,Bool,TiledReduce.flags_placed(position,flags,flag)) == 1n+List.length(&2,Bool,flags) : Nat}:
  match position:
    case Grid.Innermost{}:
      Equal.trans(Nat,List.length(&2,Bool,List.append(&2,Bool,flags,[flag])),Nat.add(List.length(&2,Bool,flags),1n),1n+List.length(&2,Bool,flags),
        ListReadback.append_length(~Bool,flags,[flag]),Addition.successor_width(List.length(&2,Bool,flags)))
    case Grid.Outermost{}: {==}

# The cells have the rank of the values.
def cells_rank(+values: List<&2,U32>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool}) -> {List.length(&2,U32,Tiled.cell_shape(values,lane,position)) == List.length(&2,U32,values) : Nat}:
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,lane,position),Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,position,within)) : {List.length(&2,U32,_) == List.length(&2,U32,values) : Nat}
  Equal.trans(Nat,List.length(&2,U32,Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane)))),1n+List.length(&2,U32,Grid.removed(values,lane)),List.length(&2,U32,values),Coordinates.placed_count(position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),removed_count(values,lane,within))

# The flags of the cells of a reduction are as many as the axes.
def placed_rank(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool}) -> {List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))) == List.length(&2,U32,values) : Nat}:
  +same = same_of(values,axes)
  +removed_same = removed_length(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same)
  Equal.trans(Nat,List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),1n+List.length(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),List.length(&2,U32,values),flags_count(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),
    Equal.trans(Nat,1n+List.length(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),1n+List.length(&2,U32,Grid.removed(values,lane)),List.length(&2,U32,values),
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),List.length(&2,U32,Grid.removed(values,lane)),Equal.sym(Nat,List.length(&2,U32,Grid.removed(values,lane)),List.length(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),removed_same)),
      removed_count(values,lane,within)))

# The cell axes of a reduction are the axes flagged among the flags of the
# cells.
def cell_words(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) ->
  {Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)) == flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{}) : List<&2,Nat>}:
  +flags_rank = placed_rank(values,axes,lane,position,within)
  +room = {%Equal.sym(Nat,List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),List.length(&2,U32,values),flags_rank) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} rank_room :
    {Nat.is_le(Nat.add(U32.to_nat(0),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)))),U32.to_nat(cap)) == True{} : Bool}}
  words_flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),0,cap,cap_ok,room)

# The cell axes of a reduction are unique.
def cells_unique(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) ->
  {Views.unique_axes(Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))) == True{} : Bool}:
  %Equal.sym(List<&2,Nat>,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{}),cell_words(values,axes,lane,position,within,cap,cap_ok,rank_room)) : {Views.unique_axes(_) == True{} : Bool}
  flagged_unique(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{})

# The cell axes of a reduction lie below the rank of the cells.
def cells_below(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) ->
  {Layout.all_below(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position),List.length(&2,U32,Tiled.cell_shape(values,lane,position))) == True{} : Bool}:
  +lengths = Equal.trans(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,U32,values),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),cells_rank(values,lane,position,within),Equal.sym(Nat,List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),List.length(&2,U32,values),placed_rank(values,axes,lane,position,within)))
  +fits = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),lengths) : {Nat.is_le(List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),_) == True{} : Bool} Order.reflexive(List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)))) : {Nat.is_le(Nat.add(U32.to_nat(0),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)))),List.length(&2,U32,Tiled.cell_shape(values,lane,position))) == True{} : Bool}}
  %Equal.sym(Bool,Layout.all_below(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position),List.length(&2,U32,Tiled.cell_shape(values,lane,position))),below_all(Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),List.length(&2,U32,Tiled.cell_shape(values,lane,position))),below_naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position),List.length(&2,U32,Tiled.cell_shape(values,lane,position)))) :
    {_ == True{} : Bool}
  %Equal.sym(List<&2,Nat>,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{}),cell_words(values,axes,lane,position,within,cap,cap_ok,rank_room)) : {below_all(_,List.length(&2,U32,Tiled.cell_shape(values,lane,position))) == True{} : Bool}
  flagged_below(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{},List.length(&2,U32,Tiled.cell_shape(values,lane,position)),fits)

# The flags of the cell axes a reduction reduces: the flags of the other axes
# and the flag of the lane axis for the groups at their position.
def cells_flags(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) -> {Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))) == TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)) : List<&2,Bool>}:
  +flags_rank = placed_rank(values,axes,lane,position,within)
  +lengths = Equal.trans(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,U32,values),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),cells_rank(values,lane,position,within),Equal.sym(Nat,List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),List.length(&2,U32,values),flags_rank))
  %Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),lengths) : {Layout.reduced_flags(_,0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))) == TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)) : List<&2,Bool>}
  %Equal.sym(List<&2,Nat>,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),U32.to_nat(0),True{}),cell_words(values,axes,lane,position,within,cap,cap_ok,rank_room)) : {Layout.reduced_flags(List.length(&2,Bool,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane))),0n,_) == TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)) : List<&2,Bool>}
  flags_flagged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),0n)

def placed_unlisted(position: Grid.Position,+extents: List<&2,U32>,+groups: U32) -> {placed_extents(position,extents,False{},True{},groups) == extents : List<&2,U32>}:
  match position:
    case Grid.Innermost{}: MetadataLaws.append_empty(extents)
    case Grid.Outermost{}: {==}

def placed_listed(position: Grid.Position,+extents: List<&2,U32>,+groups: U32) -> {placed_extents(position,extents,False{},False{},groups) == Grid.placed(True{},position,extents,groups) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: {==}

# The reduced cell axes have the extents of the reduced axes.
def cells_reduced(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) -> {Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})) : List<&2,U32>}:
  +same = same_of(values,axes)
  +removed_same = removed_length(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same)
  +unflagged = unflagged_of(values,axes,lane,position,within,absent)
  +cell_flags = cells_flags(values,axes,lane,position,within,cap,cap_ok,rank_room)
  Equal.trans(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),picked_extents(Tiled.cell_shape(values,lane,position),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),extents_listed(Tiled.cell_shape(values,lane,position),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{}),Equal.trans(List<&2,U32>,picked_extents(Tiled.cell_shape(values,lane,position),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),True{}),picked_extents(Tiled.cell_shape(values,lane,position),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Equal.cong(List<&2,Bool>,List<&2,U32>,flags => picked_extents(Tiled.cell_shape(values,lane,position),flags,True{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),cell_flags),Equal.trans(List<&2,U32>,picked_extents(Tiled.cell_shape(values,lane,position),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),True{}),picked_extents(Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Equal.cong(List<&2,U32>,List<&2,U32>,items => picked_extents(items,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),True{}),Tiled.cell_shape(values,lane,position),Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,position,within)),Equal.trans(List<&2,U32>,picked_extents(Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),True{}),placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},Grid.groups(Grid.lanes_at(values,lane))),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),picked_placed(position,Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},Grid.groups(Grid.lanes_at(values,lane)),removed_same),Equal.trans(List<&2,U32>,placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},Grid.groups(Grid.lanes_at(values,lane))),placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),False{},True{},Grid.groups(Grid.lanes_at(values,lane))),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Equal.cong(Bool,List<&2,U32>,flag => placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),flag,True{},Grid.groups(Grid.lanes_at(values,lane))),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},unflagged),Equal.trans(List<&2,U32>,placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),False{},True{},Grid.groups(Grid.lanes_at(values,lane))),picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),placed_unlisted(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Grid.groups(Grid.lanes_at(values,lane))),Equal.trans(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),removed_unpicked(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same,unflagged),Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),True{}),extents_listed(values,axes,True{})))))))))

# The kept cell axes are the cells of the kept extents with the lanes at the
# place of the lane axis among the kept axes.
def cells_kept(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(cap)) == True{} : Bool}) -> {Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})) == Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position) : List<&2,U32>}:
  +same = same_of(values,axes)
  +removed_same = removed_length(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same)
  +unflagged = unflagged_of(values,axes,lane,position,within,absent)
  +cell_flags = cells_flags(values,axes,lane,position,within,cap,cap_ok,rank_room)
  Equal.trans(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),picked_extents(Tiled.cell_shape(values,lane,position),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),False{}),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),extents_listed(Tiled.cell_shape(values,lane,position),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Equal.trans(List<&2,U32>,picked_extents(Tiled.cell_shape(values,lane,position),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),False{}),picked_extents(Tiled.cell_shape(values,lane,position),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),False{}),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(List<&2,Bool>,List<&2,U32>,flags => picked_extents(Tiled.cell_shape(values,lane,position),flags,False{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),cell_flags),Equal.trans(List<&2,U32>,picked_extents(Tiled.cell_shape(values,lane,position),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),False{}),picked_extents(Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),False{}),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(List<&2,U32>,List<&2,U32>,items => picked_extents(items,TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),False{}),Tiled.cell_shape(values,lane,position),Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,position,within)),Equal.trans(List<&2,U32>,picked_extents(Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),False{}),placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),picked_placed(position,Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},Grid.groups(Grid.lanes_at(values,lane)),removed_same),Equal.trans(List<&2,U32>,placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},Grid.groups(Grid.lanes_at(values,lane))),placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),False{},False{},Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(Bool,List<&2,U32>,flag => placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),flag,False{},Grid.groups(Grid.lanes_at(values,lane))),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},unflagged),Equal.trans(List<&2,U32>,placed_extents(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),False{},False{},Grid.groups(Grid.lanes_at(values,lane))),Grid.placed(True{},position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),placed_listed(position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.groups(Grid.lanes_at(values,lane))),Equal.trans(List<&2,U32>,Grid.placed(True{},position,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.groups(Grid.lanes_at(values,lane))),Grid.placed(True{},position,Grid.removed(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(List<&2,U32>,List<&2,U32>,items => Grid.placed(True{},position,items,Grid.groups(Grid.lanes_at(values,lane))),picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.removed(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),removed_picked(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same,unflagged)),Equal.trans(List<&2,U32>,Grid.placed(True{},position,Grid.removed(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.groups(Grid.lanes_at(values,lane))),Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(List<&2,U32>,List<&2,U32>,items => Grid.placed(True{},position,Grid.removed(items,kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.groups(Grid.lanes_at(values,lane))),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),extents_listed(values,axes,False{}))),Equal.trans(List<&2,U32>,Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.groups(Grid.lanes_at(values,lane))),Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(values,lane))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(Nat,List<&2,U32>,place => Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),place),Grid.groups(Grid.lanes_at(values,lane))),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),Equal.sym(Nat,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),place_listed(values,axes,lane,within,absent))),Equal.trans(List<&2,U32>,Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(values,lane))),Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Equal.cong(U32,List<&2,U32>,extent => Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(extent)),Grid.lanes_at(values,lane),Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Equal.sym(U32,Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),kept_lanes(values,axes,lane,position,within,absent))),Equal.sym(List<&2,U32>,Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))),cells_placed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,kept_within(values,axes,lane,position,within,absent)))))))))))))

# ---- A cell of the kept values ----

# A cell below the cells of values with their lanes inside them: the digits of
# the other axes and its group.
type KeptWord<-values: List<&2,U32>,-axis: Nat,-position: Grid.Position,-word: U32> is Data:
  KeptWord{others: List<&2,Nat>,group: Nat,others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
    group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
    exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),Coordinates.cell_digits(position,others,group)) : Nat},
    placed_exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),Coordinates.placed_digits(True{},position,others,group)) : Nat},
    group_same: {Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)) == group : Nat}}

def kept_word(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+word: U32,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) -> KeptWord<values,axis,position,word>:
  +number = U32.to_nat(word)
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,number)
  +below_placed = {%within : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits) Digits.coordinates_below(cells,number,inside_word) : Coordinates.DigitsBelow(Grid.placed(True{},position,removed,groups),digits)}
  +others = Values.cell_others(True{},position,digits)
  +group = Values.cell_group(True{},position,digits)
  +others_below = Values.cell_below(True{},position,removed,groups,digits,below_placed)
  +group_inside = Values.group_below(True{},position,removed,groups,digits,below_placed)
  +whole = Values.cell_whole(True{},position,removed,groups,digits,below_placed)
  +placed = Coordinates.placed_digits(True{},position,others,group)
  +placed_exact = {%Equal.sym(List<&2,Nat>,placed,digits,whole) : {number == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),number,Digits.value_coordinates(cells,number,inside_word)) : {number == ShapeLaws.digits_value(cells,placed) : Nat}}
  KeptWord{others,group,others_below,group_inside,
    {%Coordinates.placed_cell(position,others,group) : {number == ShapeLaws.digits_value(cells,_) : Nat} placed_exact :
      {number == ShapeLaws.digits_value(cells,Coordinates.cell_digits(position,others,group)) : Nat}},
    placed_exact,
    Coordinates.group_within(limits,values,axis,position,others,group,word,within,room,others_below,group_inside,placed_exact)}

# ---- The lanes of eight accumulators ----

def folds_lane(~Accumulator: Data,folds: TiledReduce.Folds<Accumulator>,lane: Nat) -> Accumulator:
  match folds lane:
    case TiledReduce.Folds{a,b,c,d,e,f,g,h} 0n: a
    case TiledReduce.Folds{a,b,c,d,e,f,g,h} 1n+next:
      match next:
        case 0n: b
        case 1n+next:
          match next:
            case 0n: c
            case 1n+next:
              match next:
                case 0n: d
                case 1n+next:
                  match next:
                    case 0n: e
                    case 1n+next:
                      match next:
                        case 0n: f
                        case 1n+next:
                          match next:
                            case 0n: g
                            case 1n+next: h

def started_lane(~Accumulator: Data,+initial: Accumulator,lane: Nat) -> {folds_lane(~Accumulator,TiledReduce.started(~Accumulator,initial),lane) == initial : Accumulator}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

def finished_lane(~Accumulator: Data,~finish: Accumulator -> F32,folds: TiledReduce.Folds<Accumulator>,lane: Nat) ->
  {TileLaws.select_lane(lane,TiledReduce.finished(~Accumulator,~finish,folds)) == finish(folds_lane(~Accumulator,folds,lane)) : F32}:
  match folds lane:
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} 0n: {==}
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def stepped_lane(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,folds: TiledReduce.Folds<Accumulator>,tile: Tile.Tile8,lane: Nat) ->
  {folds_lane(~Accumulator,TiledReduce.stepped(~Accumulator,~operation,folds,tile),lane) == operation(folds_lane(~Accumulator,folds,lane),TileLaws.select_lane(lane,tile)) : Accumulator}:
  match folds tile lane:
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} Tile.Tile{+i,+j,+k,+l,+m,+n,+o,+p} 0n: {==}
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} Tile.Tile{+i,+j,+k,+l,+m,+n,+o,+p} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def closed_lane(~Row: Data,~Partial: Data,~close: Row -> Partial,folds: TiledReduce.Folds<Row>,lane: Nat) ->
  {folds_lane(~Partial,TiledReduce.closed(~Row,~Partial,~close,folds),lane) == close(folds_lane(~Row,folds,lane)) : Partial}:
  match folds lane:
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} 0n: {==}
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def merged_lane(~Partial: Data,~merge: Partial -> Partial -> Partial,left: TiledReduce.Folds<Partial>,right: TiledReduce.Folds<Partial>,lane: Nat) ->
  {folds_lane(~Partial,TiledReduce.merged(~Partial,~merge,left,right),lane) == merge(folds_lane(~Partial,left,lane),folds_lane(~Partial,right,lane)) : Partial}:
  match left right lane:
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} TiledReduce.Folds{+i,+j,+k,+l,+m,+n,+o,+p} 0n: {==}
    case TiledReduce.Folds{+a,+b,+c,+d,+e,+f,+g,+h} TiledReduce.Folds{+i,+j,+k,+l,+m,+n,+o,+p} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

# One lane of the tiles a reader gives.
type LaneOf<-Context: Data> is Data:
  LaneOf{lane: Nat,context: Context}

def lane_read(~Context: Data,~value: Context -> U32 -> Tile.Tile8,read: LaneOf<Context>,index: U32) -> F32:
  LaneOf{lane,context} = read
  TileLaws.select_lane(lane,value(context,index))

# Lane lane of the fold of eight accumulators is the fold of that lane.
def lane_fold(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+lane: Nat,count: Nat,
  +position: Nat,+state: TiledReduce.Folds<Accumulator>) ->
  {folds_lane(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Context,~value,context,count,position,state),lane) ==
    Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Context>,~lane_read(~Context,~value),LaneOf{lane,context},count,position,folds_lane(~Accumulator,state,lane)) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %stepped_lane(~Accumulator,~operation,state,value(context,U32.from_nat(position)),lane) :
        {folds_lane(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Context,~value,context,rest,1n+position,
            TiledReduce.stepped(~Accumulator,~operation,state,value(context,U32.from_nat(position)))),lane) ==
          Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Context>,~lane_read(~Context,~value),LaneOf{lane,context},rest,1n+position,_) : Accumulator}
      lane_fold(~Accumulator,~operation,~Context,~value,context,lane,rest,1n+position,TiledReduce.stepped(~Accumulator,~operation,state,value(context,U32.from_nat(position))))

# ---- Folds of readers that agree ----

# Two readers that agree at every position of two lines.
def Agrees(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,count: Nat,+left_position: Nat,
  +right_position: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<{left_value(left,U32.from_nat(left_position)) == right_value(right,U32.from_nat(right_position)) : F32},
        Agrees(~Left,~left_value,~Right,~right_value,left,right,rest,1n+left_position,1n+right_position)>

# Folds of two readers that agree at every position agree.
def folds_agree(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,count: Nat,+left_position: Nat,+right_position: Nat,+state: Accumulator,
  +agrees: Agrees(~Left,~left_value,~Right,~right_value,left,right,count,left_position,right_position)) ->
  {Reduce.line_fold(~F32,~Accumulator,~operation,~Left,~left_value,left,count,left_position,state) ==
    Reduce.line_fold(~F32,~Accumulator,~operation,~Right,~right_value,right,count,right_position,state) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+first,+later} = agrees
      %first :
        {Reduce.line_fold(~F32,~Accumulator,~operation,~Left,~left_value,left,rest,1n+left_position,operation(state,left_value(left,U32.from_nat(left_position)))) ==
          Reduce.line_fold(~F32,~Accumulator,~operation,~Right,~right_value,right,rest,1n+right_position,operation(state,_)) : Accumulator}
      folds_agree(~Accumulator,~operation,~Left,~left_value,~Right,~right_value,left,right,rest,1n+left_position,1n+right_position,
        operation(state,left_value(left,U32.from_nat(left_position))),later)

# Readers that agree at every offset below a total from two bases agree on
# the lines from there.
def agrees_all(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~left: Left,~right: Right,~total: Nat,~left_base: Nat,~right_base: Nat,
  ~point: @+at: Nat -> @+inside: {Nat.is_lt(at,total) == True{} : Bool} -> {left_value(left,U32.from_nat(Nat.add(left_base,at))) == right_value(right,U32.from_nat(Nat.add(right_base,at))) : F32},
  remaining: Nat,+offset: Nat,+fits: {Nat.add(offset,remaining) == total : Nat}) ->
  Agrees(~Left,~left_value,~Right,~right_value,left,right,remaining,Nat.add(left_base,offset),Nat.add(right_base,offset)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = {%fits : {Nat.is_lt(offset,_) == True{} : Bool} Order.below_sum(offset,rest) : {Nat.is_lt(offset,total) == True{} : Bool}}
      +later_fits = Equal.trans(Nat,Nat.add(1n+offset,rest),Nat.add(offset,1n+rest),total,Equal.sym(Nat,Nat.add(offset,1n+rest),1n+Nat.add(offset,rest),Addition.successor_right(offset,rest)),fits)
      KernelGemm.Both{point(offset,inside),
        {%Addition.successor_right(left_base,offset) : Agrees(~Left,~left_value,~Right,~right_value,left,right,rest,_,1n+Nat.add(right_base,offset))
          %Addition.successor_right(right_base,offset) : Agrees(~Left,~left_value,~Right,~right_value,left,right,rest,Nat.add(left_base,1n+offset),_)
          agrees_all(~Left,~left_value,~Right,~right_value,~left,~right,~total,~left_base,~right_base,~point,rest,1n+offset,later_fits) :
          Agrees(~Left,~left_value,~Right,~right_value,left,right,rest,1n+Nat.add(left_base,offset),1n+Nat.add(right_base,offset))}}

# ---- One lane of a kept cell ----

# What the cells of values within a capacity give.
def AxisCells(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> Data:
  KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
      KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
        {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}>>>

def removed_room_of(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cells: AxisCells(limits,values,axis,position)) ->
  {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KernelGemm.Both{+removed,+rest} = cells
  removed

def lanes_room_of(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cells: AxisCells(limits,values,axis,position)) ->
  {Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KernelGemm.Both{+removed,KernelGemm.Both{+lanes,+rest}} = cells
  lanes

def cells_room_of(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cells: AxisCells(limits,values,axis,position)) ->
  {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KernelGemm.Both{+removed,KernelGemm.Both{+lanes,KernelGemm.Both{+room,+positive}}} = cells
  room

def cells_positive_of(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+cells: AxisCells(limits,values,axis,position)) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}:
  KernelGemm.Both{+removed,KernelGemm.Both{+lanes,KernelGemm.Both{+room,+positive}}} = cells
  positive

# The cells of values, of the kept values and of the listed values within a
# capacity.
type KeptRooms<-limits: Machine.Limits,-values: List<&2,U32>,-axes: List<&2,Nat>,-lane: Nat,-position: Grid.Position> is Data:
  KeptRooms{values_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},values_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
    removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},
    kept_room: {Nat.is_le(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},kept_positive: {Nat.is_lt(0n,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool},
    kept_removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    reduced_positive: {Nat.is_lt(0n,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))) == True{} : Bool},
    cells_fewer: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),ShapeLaws.elements(values)) == True{} : Bool}}

def kept_rooms(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+values_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +values_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) -> KeptRooms<limits,values,axes,lane,position>:
  +cells = Coordinates.axis_cells(True{},limits,values,lane,position,within,values_room,values_positive)
  +product = Equal.trans(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(List.drop(&2,U32,values,0n)),ShapeLaws.elements(values),
    Reduce.partition_elements(values,List.length(&2,U32,values),0n,axes,{==}),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,values,0n),values,Reduce.dropped_none(~U32,values)))
  +product_positive = {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(values),product) : {Nat.is_lt(0n,_) == True{} : Bool} values_positive :
    {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))))) == True{} : Bool}}
  +kept_positive = RowMajor.left_factor_positive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),product_positive)
  +reduced_positive = DivisionOrder.right_factor_positive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),product_positive)
  +kept_room = Order.transitive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),reduced_positive),
    {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(values),product) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} values_room :
      {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
  +kept_cells = Coordinates.axis_cells(True{},limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,kept_within(values,axes,lane,position,within,absent),kept_room,kept_positive)
  +lanes_room = lanes_room_of(limits,values,lane,position,cells)
  +cells_fewer = {%Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane))),Public.cells_split(values,lane,position,within)) :
      {Nat.is_le(_,ShapeLaws.elements(values)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(U32.to_nat(Grid.lanes_at(values,lane)),ShapeLaws.elements(Grid.removed(values,lane))),Public.elements_split(values,lane,within)) :
      {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane))),_) == True{} : Bool}
    RowMajor.mul_le(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),U32.to_nat(Grid.lanes_at(values,lane)),ShapeLaws.elements(Grid.removed(values,lane)),
      Coordinates.groups_fewer(limits,Grid.lanes_at(values,lane),TiledValues.grouped(limits,Grid.lanes_at(values,lane),lanes_room))) :
    {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),ShapeLaws.elements(values)) == True{} : Bool}}
  KeptRooms{values_room,values_positive,removed_room_of(limits,values,lane,position,cells),cells_room_of(limits,values,lane,position,cells),
    cells_positive_of(limits,values,lane,position,cells),kept_room,kept_positive,removed_room_of(limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,kept_cells),reduced_positive,cells_fewer}

# The digits of a cell lie below the cells.
def cell_below_of(position: Grid.Position,+removed: List<&2,U32>,+groups: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(removed,others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}) -> Coordinates.DigitsBelow(Grid.placed(True{},position,removed,groups),Coordinates.cell_digits(position,others,group)):
  match position:
    case Grid.Innermost{}: Coordinates.digits_joined(removed,others,[groups],[group],others_below,KernelGemm.Both{group_inside,Unit{}})
    case Grid.Outermost{}: KernelGemm.Both{group_inside,others_below}

# A valid lane of a kept cell at any reduced digits reads the value of the
# specification at the digits of the value with them.
def lane_merged(~Logical: Data,~original: Logical -> U32 -> F32,~Context: Data,~read: Context -> U32 -> Tile.Tile8,~context_at: Logical -> Cells.Cells -> Context,~agree: Coordinates.CellsAgree(~Logical,~original,~Context,~read,~context_at),+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},+digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},+reduced: List<&2,Nat>,+reduced_below: Coordinates.DigitsBelow(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced)) ->
  {TileLaws.select_lane(digit,read(context_at(logical,Public.geometry_of(values,lane,position)),U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),reduced))))) == original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))) : F32}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  +same = same_of(values,axes)
  +unflagged = unflagged_of(values,axes,lane,position,within,absent)
  +removed_same = removed_length(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same)
  +lanes_same = kept_lanes(values,axes,lane,position,within,absent)
  +kept_exact = extents_listed(values,axes,False{})
  +reduced_exact = extents_listed(values,axes,True{})
  +place_exact = place_listed(values,axes,lane,within,absent)
  +cell_flags = cells_flags(values,axes,lane,position,within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)
  +removed_exact = Equal.trans(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.removed(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),removed_picked(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same,unflagged),
    Equal.trans(List<&2,U32>,Grid.removed(picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),
      Equal.cong(List<&2,U32>,List<&2,U32>,items => Grid.removed(items,kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),kept_exact)),
      Equal.cong(Nat,List<&2,U32>,place => Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),place),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),Equal.sym(Nat,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),place_exact))))
  +others_picked = {%Equal.sym(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),removed_exact) : Coordinates.DigitsBelow(_,others) others_below : Coordinates.DigitsBelow(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),others)}
  +unpicked_exact = Equal.trans(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),removed_unpicked(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same,unflagged),Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),True{}),reduced_exact))
  +reduced_picked = {%Equal.sym(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),unpicked_exact) : Coordinates.DigitsBelow(_,reduced) reduced_below : Coordinates.DigitsBelow(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),reduced)}
  +merged_digits_below = merged_below(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced,others_picked,reduced_picked,removed_same)
  +group_inside_values = {%lanes_same : {Nat.is_lt(group,U32.to_nat(Grid.groups(_))) == True{} : Bool} group_inside :
    {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool}}
  +cell_digits_below = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,lane,position),Grid.placed(True{},position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,position,within)) : Coordinates.DigitsBelow(_,Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))
    cell_below_of(position,Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane)),merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group,merged_digits_below,group_inside_values) : Coordinates.DigitsBelow(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))}
  +cell_value_room = Order.transitive(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group)),ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits)),
    Bounds.strict_le(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group)),ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),Coordinates.digits_below(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group),cell_digits_below)),cells_room)
  +lane_digits = Coordinates.lane_digits(~Logical,~original,logical,limits,values,lane,position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group,U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))),within,removed_room,merged_digits_below,group_inside_values,
    FullRange.roundtrip(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group)),Machine.capacity(limits),cell_value_room),digit,small)
  +lane_value = {%valid : {TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))) ==
      Bool.pick(F32,_,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),Nat.add(digit,Nat.mul(group,8n)))))),0.0) : F32}
    lane_digits : {TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))) ==
      original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),Nat.add(digit,Nat.mul(group,8n)))))) : F32}}
  +inserted_eq = merged_inserted(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,others,reduced,Nat.add(digit,Nat.mul(group,8n)),others_picked,reduced_picked,same,unflagged,within)
  +merged_cells = Equal.trans(List<&2,Nat>,merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),reduced),merged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Coordinates.cell_digits(position,others,group),reduced),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group),
    Equal.cong(List<&2,Bool>,List<&2,Nat>,flags => merged(flags,Coordinates.cell_digits(position,others,group),reduced),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),cell_flags),
    Equal.trans(List<&2,Nat>,merged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),Coordinates.cell_digits(position,others,group),reduced),merged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Coordinates.cell_digits(position,others,group),reduced),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group),
      Equal.cong(Bool,List<&2,Nat>,flag => merged(TiledReduce.flags_placed(position,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),flag),Coordinates.cell_digits(position,others,group),reduced),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{},unflagged),
      merged_placed(position,Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced,group,others_picked,reduced_picked,removed_same)))
  Equal.trans(F32,TileLaws.select_lane(digit,read(context_at(logical,Public.geometry_of(values,lane,position)),U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),reduced))))),TileLaws.select_lane(digit,read(context_at(logical,Public.geometry_of(values,lane,position)),U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),Equal.cong(List<&2,Nat>,F32,digits => TileLaws.select_lane(digit,read(context_at(logical,Public.geometry_of(values,lane,position)),U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),digits)))),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),reduced),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group),merged_cells),
    Equal.trans(F32,TileLaws.select_lane(digit,read(context_at(logical,Public.geometry_of(values,lane,position)),U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))),TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),
      agree(logical,limits,values,lane,position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group,U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))),within,removed_room,merged_digits_below,group_inside_values,
            FullRange.roundtrip(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group)),Machine.capacity(limits),cell_value_room),digit,small,valid),
    Equal.trans(F32,TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),Coordinates.cell_digits(position,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),group))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),Nat.add(digit,Nat.mul(group,8n)))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),lane_value,
      Equal.trans(F32,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),Nat.add(digit,Nat.mul(group,8n)))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),Equal.cong(List<&2,Nat>,F32,digits => original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))),Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,reduced),Nat.add(digit,Nat.mul(group,8n))),merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),others,Nat.add(digit,Nat.mul(group,8n))),reduced),inserted_eq),Equal.cong(Nat,F32,place => original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(place,others,Nat.add(digit,Nat.mul(group,8n))),reduced)))),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),Equal.sym(Nat,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),kept_before(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),place_exact))))))

# A valid lane of a kept cell reads at a position of its line the value of
# the specification at that position of the line of the value.
def offset_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},
  +offset: Nat,+inside: {Nat.is_lt(offset,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))) == True{} : Bool}) ->
  {lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))) == Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))) : F32}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  +within_kept = kept_within(values,axes,lane,position,within,absent)
  +lanes_same = kept_lanes(values,axes,lane,position,within,absent)
  +kept_cells = cells_kept(values,axes,lane,position,within,absent,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)
  +reduced_cells = cells_reduced(values,axes,lane,position,within,absent,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)
  +reduced_below = Digits.coordinates_below(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset,inside)
  +reduced_value = Digits.value_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset,inside)
  +kept_shape_below = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Grid.placed(True{},position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))),
      cells_placed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,within_kept)) : Coordinates.DigitsBelow(_,Coordinates.cell_digits(position,others,group))
    cell_below_of(position,Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),others,group,others_below,group_inside) : Coordinates.DigitsBelow(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group))}
  +kept_cells_below = {%Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),kept_cells) : Coordinates.DigitsBelow(_,Coordinates.cell_digits(position,others,group)) kept_shape_below :
    Coordinates.DigitsBelow(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Coordinates.cell_digits(position,others,group))}
  +reduced_cells_below = {%Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced_cells) : Coordinates.DigitsBelow(_,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset)) reduced_below : Coordinates.DigitsBelow(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))}
  +cell_position = {
    %Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),kept_cells) :
      {Nat.add(Nat.mul(ShapeLaws.digits_value(_,Coordinates.cell_digits(position,others,group)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    %Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced_cells) :
      {Nat.add(Nat.mul(ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)),ShapeLaws.elements(_)),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    %Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced_cells) :
      {Nat.add(Nat.mul(ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(_,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    %exact : {Nat.add(Nat.mul(_,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    %Equal.sym(Nat,ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset)),offset,reduced_value) : {Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),_) == Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    {==} : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Coordinates.cell_digits(position,others,group)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) ==
      Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}}
  +cell_address = {%cell_position : {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},U32.from_nat(_)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},
        U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))) : Tile.Tile8}
    line_address(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Tiled.cell_shape(values,lane,position),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset),kept_cells_below,reduced_cells_below,
      Machine.capacity(limits),Limits.capacity_fits(limits),cells_positive,cells_room) :
    {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},
      U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))) : Tile.Tile8}}
  +spec_below = Values.inserted_below(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)),others_below,
    {%Equal.sym(U32,Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),lanes_same) : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(_)) == True{} : Bool} valid :
      {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))) == True{} : Bool}},within_kept)
  +spec_room = Order.transitive(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),Coordinates.digits_below(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),spec_below)),kept_room)
  +spec_exact = FullRange.roundtrip(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),Machine.capacity(limits),spec_room)
  +spec_position = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),spec_exact) : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(_,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}
    %reduced_value : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),_) : Nat}
    {==} : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))) == Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset) : Nat}}
  +spec_address = line_address(~F32,~Logical,~original,logical,values,axes,Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset),spec_below,reduced_below,Machine.capacity(limits),Limits.capacity_fits(limits),values_positive,values_room)
  Equal.trans(F32,TileLaws.select_lane(digit,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset)))),TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset)))))),Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))),Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(digit,tile),Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))),Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),cell_address),
    Equal.trans(F32,TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,position)},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,position),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Coordinates.cell_digits(position,others,group),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset)))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))),lane_merged(~Logical,~original,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Public.lifted_at(~Logical),~Coordinates.lifted_agree(~Logical,~original),logical,limits,values,axes,lane,position,within,absent,rank_room,rooms,others,group,others_below,group_inside,digit,small,valid,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset),reduced_below),
      Equal.trans(F32,original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset))),Equal.sym(F32,Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))))),spec_address),Equal.cong(Nat,F32,number => Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(number)),Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),offset))),Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset),spec_position))))

# Every position of the line of a valid lane of a kept cell agrees with the
# line of its value.
def lanes_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},
  remaining: Nat,+offset: Nat,+fits: {Nat.add(offset,remaining) == ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))) : Nat}) ->
  Agrees(~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},remaining,Nat.add(Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset),Nat.add(Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),offset)):
  agrees_all(~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),~LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},~Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},~ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),~Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),~Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),
    ~(at => below => offset_agree(~Logical,~original,logical,limits,values,axes,lane,position,within,absent,rank_room,rooms,word,others,group,others_below,group_inside,exact,digit,small,valid,at,below)),remaining,offset,fits)

# ---- The lanes of a kept cell ----

# A valid lane of a kept cell holds the value of its line.
def kept_lane(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},+initial: Accumulator) ->
  {TileLaws.select_lane(digit,Reduce.line_value(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Lines{TiledReduce.started(~Accumulator,initial),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},word)) == Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))) : F32}:
  +reduced_cells = cells_reduced(values,axes,lane,position,within,absent,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)
  +position_cells = Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))))
  +position_spec = Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))))
  +agrees = {%Addition.zero_right(position_cells) : Agrees(~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),_,position_spec)
    %Addition.zero_right(position_spec) : Agrees(~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.add(position_cells,0n),_)
    lanes_agree(~Logical,~original,logical,limits,values,axes,lane,position,within,absent,rank_room,rooms,word,others,group,others_below,group_inside,exact,digit,small,valid,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),0n,{==}) :
    Agrees(~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),position_cells,position_spec)}
  %Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced_cells) :
    {TileLaws.select_lane(digit,TiledReduce.finished(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),
        ~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(_),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))),TiledReduce.started(~Accumulator,initial)))) ==
      finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial)) : F32}
  %Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),reduced_cells) :
    {TileLaws.select_lane(digit,TiledReduce.finished(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),
        ~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(_)),TiledReduce.started(~Accumulator,initial)))) ==
      finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial)) : F32}
  Equal.trans(F32,TileLaws.select_lane(digit,TiledReduce.finished(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)))),finish(folds_lane(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)),digit)),finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial)),
    finished_lane(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)),digit),
    Equal.cong(Accumulator,F32,state => finish(state),folds_lane(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)),digit),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial),
      Equal.trans(Accumulator,folds_lane(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)),digit),Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),folds_lane(~Accumulator,TiledReduce.started(~Accumulator,initial),digit)),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial),
        lane_fold(~Accumulator,~operation,~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},digit,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),TiledReduce.started(~Accumulator,initial)),
        Equal.trans(Accumulator,Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),folds_lane(~Accumulator,TiledReduce.started(~Accumulator,initial),digit)),Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial),
          Equal.cong(Accumulator,Accumulator,state => Reduce.line_fold(~F32,~Accumulator,~operation,~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),state),folds_lane(~Accumulator,TiledReduce.started(~Accumulator,initial),digit),initial,
            started_lane(~Accumulator,initial,digit)),
          folds_agree(~Accumulator,~operation,~LaneOf<Reduce.Ordered<Representation.Lifted<Logical>>>,~lane_read(~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),LaneOf{digit,Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}},Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),
            Nat.mul(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.mul(U32.to_nat(U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial,agrees)))))

# What a lane of a kept cell holds: below the lanes, the value of the target
# at the kept digits with the lane put back.
def KeptLanes(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target) -> Type:
  @+spec: Spec -> @+limits: Machine.Limits -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> @+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool} -> @+absent: {Views.contains_index(axes,lane) == False{} : Bool} -> @+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} -> @+rooms: KeptRooms<limits,values,axes,lane,position> -> @+arranged: Fits(spec,values,axes,lane,position) -> @+word: U32 -> @+others: List<&2,Nat> -> @+group: Nat -> @+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others) -> @+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool} -> @+exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat} -> @+digit: Nat -> @+small: {Nat.is_le(digit,7n) == True{} : Bool} -> @+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool} ->
    {TileLaws.select_lane(digit,cells_read(cells_at(spec,values,axes,lane,position),word)) == target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))) : F32}

# Lane digit of a kept cell below the lanes holds the lifted value of the
# specification, and the padding holds +0 past them like it.
def lane_cases(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~core: KeptLanes(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at),valid: Bool,+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+count: Nat,+count_exact: {count == U32.to_nat(Grid.lanes_at(values,lane)) : Nat},
  +classified: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),count) == valid : Bool}) ->
  {PaddedCells.kept(count,group,digit,TileLaws.select_lane(digit,cells_read(cells_at(spec,values,axes,lane,position),word))) == TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) : F32}:
  match valid:
    case True{}:
      KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
      +valid_values = {%count_exact : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),_) == True{} : Bool} classified : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}}
      +lanes_same = kept_lanes(values,axes,lane,position,within,absent)
      +valid_kept = {%Equal.sym(U32,Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),lanes_same) : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(_)) == True{} : Bool} valid_values :
        {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))) == True{} : Bool}}
      +target = Coordinates.lane_digits(~Target,~target_read,target_at(spec,values,axes),limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,others,group,word,kept_within(values,axes,lane,position,within,absent),
        kept_removed_room,others_below,group_inside,exact,digit,small)
      +target_valid = {%valid_kept : {TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) == Bool.pick(F32,_,target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),0.0) : F32} target :
        {TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) == target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))) : F32}}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),count),True{},classified) :
        {Bool.pick(F32,_,TileLaws.select_lane(digit,cells_read(cells_at(spec,values,axes,lane,position),word)),0.0) == TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) : F32}
      Equal.trans(F32,TileLaws.select_lane(digit,cells_read(cells_at(spec,values,axes,lane,position),word)),target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)),
        core(spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,digit,small,valid_values),
        Equal.sym(F32,TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)),target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),target_valid))
    case False{}:
      KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
      +invalid_values = {%count_exact : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),_) == False{} : Bool} classified : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == False{} : Bool}}
      +lanes_same = kept_lanes(values,axes,lane,position,within,absent)
      +invalid_kept = {%Equal.sym(U32,Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),lanes_same) : {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(_)) == False{} : Bool} invalid_values :
        {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)))) == False{} : Bool}}
      +target = Coordinates.lane_digits(~Target,~target_read,target_at(spec,values,axes),limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,others,group,word,kept_within(values,axes,lane,position,within,absent),
        kept_removed_room,others_below,group_inside,exact,digit,small)
      +target_zero = {%invalid_kept : {TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) == Bool.pick(F32,_,target_read(target_at(spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))),0.0) : F32} target :
        {TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) == 0.0 : F32}}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),count),False{},classified) :
        {Bool.pick(F32,_,TileLaws.select_lane(digit,cells_read(cells_at(spec,values,axes,lane,position),word)),0.0) == TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)) : F32}
      Equal.sym(F32,TileLaws.select_lane(digit,Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)),0.0,target_zero)

# Every lane of a kept cell below the lanes holds the lifted value of the
# specification.
def kept_valid(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~core: KeptLanes(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at),+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},+count: Nat,
  +count_exact: {count == U32.to_nat(Grid.lanes_at(values,lane)) : Nat}) ->
  PaddedCells.Valid<count,group,cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},word)>:
  PaddedCells.Valid{lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,0n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,1n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,2n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,3n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,4n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,5n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,6n,{==},count,count_exact,{==}),
    lane_cases(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),count),spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,7n,{==},count,count_exact,{==})}

# ---- The kept result ----

def rooms_cells_positive(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+rooms: KeptRooms<limits,values,axes,lane,position>) -> {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  cells_positive

def rooms_cells_fewer(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+rooms: KeptRooms<limits,values,axes,lane,position>) -> {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),ShapeLaws.elements(values)) == True{} : Bool}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  cells_fewer

def rooms_kept_room(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+rooms: KeptRooms<limits,values,axes,lane,position>) -> {Nat.is_le(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  kept_room

def rooms_kept_removed(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+rooms: KeptRooms<limits,values,axes,lane,position>) -> {Nat.is_le(ShapeLaws.elements(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  kept_removed_room

# The specification a fold of kept cells reads: the logical values and the
# initial accumulator.
type KeptFold<-Accumulator: Data,-Logical: Data> is Data:
  KeptFold{logical: Logical,initial: Accumulator}

def fold_cells(~Accumulator: Data,~Logical: Data,+spec: KeptFold<Accumulator,Logical>,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position) -> Reduce.Lines<TiledReduce.Folds<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>:
  KeptFold{+logical,+initial} = spec
  Reduce.Lines{TiledReduce.started(~Accumulator,initial),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Layout.line_view(Tiled.cell_shape(values,lane,position),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),True{})))}

def fold_target(~Accumulator: Data,~Logical: Data,+spec: KeptFold<Accumulator,Logical>,+values: List<&2,U32>,+axes: List<&2,Nat>) -> Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>:
  KeptFold{+logical,+initial} = spec
  Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))}

# What a kept result asks of its reduction: nothing, or its groups where a
# reduction on its kept axis puts them.
def free(~Spec: Data,spec: Spec,values: List<&2,U32>,axes: List<&2,Nat>,lane: Nat,position: Grid.Position) -> Data:
  Unit

def pinned(~Spec: Data,spec: Spec,values: List<&2,U32>,axes: List<&2,Nat>,lane: Nat,position: Grid.Position) -> Data:
  {position == TiledReduce.kept_position(lane,List.length(&2,U32,values)) : Grid.Position}

# What every piece of a kept result reads: the facts of the reduction and the
# layout of the kept cells.
type KeptScope<-Spec: Data,-Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data> is Data:
  KeptScope{spec: Spec,limits: Machine.Limits,values: List<&2,U32>,axes: List<&2,Nat>,lane: Nat,position: Grid.Position,
    within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},absent: {Views.contains_index(axes,lane) == False{} : Bool},
    rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},rooms: KeptRooms<limits,values,axes,lane,position>,arranged: Fits(spec,values,axes,lane,position),layout: PaddedCells.Layout,
    geometry_exact: {PaddedCells.layout_geometry(layout) == Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position) : Cells.Cells},
    total_exact: {PaddedCells.layout_total(layout) == ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}))) : Nat}}

def kept_layout(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,+scope: KeptScope<Spec,Fits>) -> PaddedCells.Layout:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  layout

def kept_padding(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,+scope: KeptScope<Spec,Fits>) -> TiledMap.Padding:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  PaddedCells.layout_padding(layout)

def kept_total(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,+scope: KeptScope<Spec,Fits>) -> Nat:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  PaddedCells.layout_total(layout)

def kept_lines(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,+scope: KeptScope<Spec,Fits>) -> Cells:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  cells_at(spec,values,axes,lane,position)

def kept_target(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Target: Data,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,+scope: KeptScope<Spec,Fits>) -> Representation.Lifted<Target>:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  Representation.Lifted{target_at(spec,values,axes),PaddedCells.layout_geometry(layout)}

# A lane of a fold of kept cells holds the fold of its line.
def fold_lane(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+spec: KeptFold<Accumulator,Logical>,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Unit,+word: U32,+others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}) ->
  {TileLaws.select_lane(digit,Reduce.line_value(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),fold_cells(~Accumulator,~Logical,spec,values,axes,lane,position),word)) == Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),fold_target(~Accumulator,~Logical,spec,values,axes),U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))) : F32}:
  KeptFold{+logical,+initial} = spec
  kept_lane(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,position,within,absent,rank_room,rooms,word,others,group,others_below,group_inside,exact,digit,small,valid,initial)

# A kept cell of a word agrees below the lanes with the lifted values of the
# specification in the layout of the kept cells.
def kept_word_valid(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~core: KeptLanes(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at),+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+layout: PaddedCells.Layout,
  +geometry_exact: {PaddedCells.layout_geometry(layout) == Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position) : Cells.Cells},+word: U32,+kept: KeptWord<Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,word>) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(layout)),Representation.cell_group(PaddedCells.layout_geometry(layout),U32.to_nat(word)),cells_read(cells_at(spec,values,axes,lane,position),word),
    Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),PaddedCells.layout_geometry(layout)},word)>:
  KeptWord{+others,+group,+others_below,+group_inside,+exact,+placed_exact,+group_same} = kept
  +geometry = PaddedCells.layout_geometry(layout)
  +kept_geometry = Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)
  %Equal.sym(Cells.Cells,geometry,kept_geometry,geometry_exact) :
    PaddedCells.Valid<Representation.lane_count(_),Representation.cell_group(geometry,U32.to_nat(word)),cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),geometry},word)>
  %Equal.sym(Cells.Cells,geometry,kept_geometry,geometry_exact) :
    PaddedCells.Valid<Representation.lane_count(kept_geometry),Representation.cell_group(_,U32.to_nat(word)),cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),geometry},word)>
  %Equal.sym(Cells.Cells,geometry,kept_geometry,geometry_exact) :
    PaddedCells.Valid<Representation.lane_count(kept_geometry),Representation.cell_group(kept_geometry,U32.to_nat(word)),cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),_},word)>
  %Equal.sym(Nat,Representation.lane_count(kept_geometry),U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),
      Coordinates.geometry_lanes(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,Bounds.strict_le(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),List.length(&2,U32,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),kept_within(values,axes,lane,position,within,absent)))) :
    PaddedCells.Valid<_,Representation.cell_group(kept_geometry,U32.to_nat(word)),cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),kept_geometry},word)>
  %Equal.sym(Nat,Representation.cell_group(kept_geometry,U32.to_nat(word)),group,group_same) :
    PaddedCells.Valid<U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),_,cells_read(cells_at(spec,values,axes,lane,position),word),Representation.lifted(~Target,~target_read,Representation.Lifted{target_at(spec,values,axes),kept_geometry},word)>
  kept_valid(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,word,others,group,others_below,group_inside,exact,
    U32.to_nat(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))),Equal.cong(U32,Nat,extent => U32.to_nat(extent),Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),kept_lanes(values,axes,lane,position,within,absent)))

def kept_agree(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~core: KeptLanes(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at),+scope: KeptScope<Spec,Fits>,+word: U32,+inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(kept_layout(~Spec,~Fits,scope))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(kept_layout(~Spec,~Fits,scope))),
    Representation.cell_group(PaddedCells.layout_geometry(kept_layout(~Spec,~Fits,scope)),U32.to_nat(word)),
    cells_read(kept_lines(~Spec,~Fits,~Cells,~cells_at,scope),word),Representation.lifted(~Target,~target_read,kept_target(~Spec,~Fits,~Target,~target_at,scope),word)>:
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  +inside_cells = {%total_exact : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} inside : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})))) == True{} : Bool}}
  +inside_kept = {%cells_kept(values,axes,lane,position,within,absent,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room) : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(_)) == True{} : Bool} inside_cells :
    {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position))) == True{} : Bool}}
  kept_word_valid(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~core,spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,layout,geometry_exact,word,
    kept_word(limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position,word,kept_within(values,axes,lane,position,within,absent),rooms_kept_removed(limits,values,axes,lane,position,rooms),inside_kept))

# Padding a piece of kept cells leaves the lifted values of the
# specification.
def kept_rule(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~agree: PaddedCells.Agreement(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at)),+scope: KeptScope<Spec,Fits>,+first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),kept_total(~Spec,~Fits,scope)) == True{} : Bool},
  +held: Content.piece_values(~Tile.Tile8,~Cells,~cells_read,kept_lines(~Spec,~Fits,~Cells,~cells_at,scope),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),kept_target(~Spec,~Fits,~Target,~target_at,scope),first,count,input,
    Padding.step(kept_padding(~Spec,~Fits,scope),first,count,input)):
  KeptScope{+spec,+limits,+values,+axes,+lane,+position,+within,+absent,+rank_room,+rooms,+arranged,+layout,+geometry_exact,+total_exact} = scope
  +rebuilt = {KeptScope{spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,layout,geometry_exact,total_exact} : KeptScope<Spec,Fits>}
  PaddedCells.padded_piece(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),
    ~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at),~agree,
    rebuilt,first,count,input,ready,length_exact,inside,held)

def kept_scoped(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~agree: PaddedCells.Agreement(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at)),+scope: KeptScope<Spec,Fits>,+parallel: Bool,+tasks: Pieces.Ranges,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +total_exact: {kept_total(~Spec,~Fits,scope) == ShapeLaws.elements(shape) : Nat},
  +contract: Global.Contract(~Tile.Tile8,~Cells,~cells_read,kept_lines(~Spec,~Fits,~Cells,~cells_at,scope),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),kept_target(~Spec,~Fits,~Target,~target_at,scope),
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,kept_padding(~Spec,~Fits,scope),Tasks.Decision{parallel,tasks},PiecesModel.pack(~Tile.Tile8,model),shape,fill,workers)):
  PiecesMap.decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,~Padding.step,~Padding.rule,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),
    ~KeptScope<Spec,Fits>,~kept_padding(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at),~kept_total(~Spec,~Fits),
    ~kept_rule(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~agree),scope,Tasks.Decision{parallel,tasks},model,shape,fill,workers,total_exact,contract)

def kept_placed(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~agree: PaddedCells.Agreement(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at)),+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+lanes: U32,+within_cells: Bool,+parallel: Bool,+tasks: Pieces.Ranges,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Cells,~cells_read,cells_at(spec,values,axes,lane,position),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,Tensors.Usable{},workers}),
  +placed: PaddedCells.Placed<Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position,within_cells>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_when(within_cells,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),lanes,position),Tasks.Decision{parallel,tasks},
      PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,workers)):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  %padding_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),Representation.Lifted{target_at(spec,values,axes),Public.geometry_of(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,_,Tasks.Decision{parallel,tasks},PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,workers))
  %geometry_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),Representation.Lifted{target_at(spec,values,axes),_},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,PaddedCells.layout_padding(layout),Tasks.Decision{parallel,tasks},PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,workers))
  kept_scoped(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~agree,KeptScope{spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,layout,geometry_exact,total_exact},parallel,tasks,model,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,workers,total_exact,global)

def kept_decided(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~agree: PaddedCells.Agreement(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at)),+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+lanes: U32,decision: Tasks.Decision,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Cells,~cells_read,cells_at(spec,values,axes,lane,position),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,Tensors.Usable{},workers}),
  +laid: Public.Laid<Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))>) ->
  Public.Within(~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~Target,~Public.lifted_at(~Target),target_at(spec,values,axes),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position,
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position),decision,PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,workers)):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      +within_cells = Nat.is_lt(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),List.length(&2,U32,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}))))
      KernelGemm.Both{kept_placed(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~agree,spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,lanes,within_cells,parallel,tasks,model,fill,workers,global,
        PaddedCells.placed(within_cells,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),{==},laid)),laid}

# Kept cells that hold the folds of the lines of the cells, padded, are the
# tiled values of the specification.
def kept_contract(~Spec: Data,~Fits: Spec -> List<&2,U32> -> List<&2,Nat> -> Nat -> Grid.Position -> Data,~Cells: Data,~cells_read: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> F32,~cells_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> @+lane: Nat -> @+position: Grid.Position -> Cells,~target_at: @+spec: Spec -> @+values: List<&2,U32> -> @+axes: List<&2,Nat> -> Target,~agree: PaddedCells.Agreement(~KeptScope<Spec,Fits>,~Cells,~cells_read,~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_read),~kept_layout(~Spec,~Fits),~kept_lines(~Spec,~Fits,~Cells,~cells_at),~kept_target(~Spec,~Fits,~Target,~target_at)),+spec: Spec,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: KeptRooms<limits,values,axes,lane,position>,+arranged: Fits(spec,values,axes,lane,position),+lanes: U32,
  +pinned: {Grid.lanes_at(values,lane) == lanes : U32},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},-result: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Cells,~cells_read,cells_at(spec,values,axes,lane,position),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),result)) ->
  Public.Contract(~Target,~target_read,target_at(spec,values,axes),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_result(Tiled.Tiled{result,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position})):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},+global} = shaped
  +kept_cells_eq = cells_kept(values,axes,lane,position,within,absent,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)
  +lanes_eq = Equal.trans(U32,Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),Grid.lanes_at(values,lane),lanes,kept_lanes(values,axes,lane,position,within,absent),pinned)
  +restored = {%Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),kept_cells_eq) : {Tiled.logical_shape(_,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}
    %lanes_eq : {Tiled.logical_shape(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),_,position) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}
    Public.cells_restored(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position) : {Tiled.logical_shape(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}}
  +laid = {Public.Laid{Bounds.strict_le(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),List.length(&2,U32,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),kept_within(values,axes,lane,position,within,absent)),
    Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),kept_cells_eq),restored,StorageCount.counted_within(limits,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),rooms_kept_room(limits,values,axes,lane,position,rooms),default_capacity),lanes_eq} :
    Public.Laid<Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))>}
  +packed_global = {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,Tensors.Usable{},workers},result,equation) :
      Global.Contract(~Tile.Tile8,~Cells,~cells_read,cells_at(spec,values,axes,lane,position),_) global :
    Global.Contract(~Tile.Tile8,~Cells,~cells_read,cells_at(spec,values,axes,lane,position),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{})),fill,Tensors.Usable{},workers})}
  %equation : Public.Contract(~Target,~target_read,target_at(spec,values,axes),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_result(Tiled.Tiled{_,TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),lanes,position}))
  kept_decided(~Spec,~Fits,~Cells,~cells_read,~Target,~target_read,~cells_at,~target_at,~agree,spec,limits,values,axes,lane,position,within,absent,rank_room,rooms,arranged,lanes,Tasks.decide(Tensors.grid_of(Layout.extents_at(Tiled.cell_shape(values,lane,position),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),False{}))),workers,Tensors.pass_rows(1.0,0.0),Tensors.pass_steps(),Tasks.no_extra()),
    model,fill,workers,packed_global,laid)

# ---- Lanes on a kept axis ----

# A reduction with the lanes on a kept axis folds eight lanes a cell over the
# reduced cell axes and pads the result: the tiled values of the
# specification.
def kept_route(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+lanes: U32,
  +position: Grid.Position,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,lanes,position}),
  +within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(Layout.naturals(listed),lane) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Kept{},name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,lanes,position})):
  KernelGemm.Both{+global,+laid} = contract
  Public.Laid{+inside,+shaped_cells,+restored,+counted,+pinned} = laid
  +axes = Layout.naturals(listed)
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +rooms = kept_rooms(limits,values,axes,lane,position,within,absent,values_room,nonempty)
  +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,position)),List.length(&2,U32,values),cells_rank(values,lane,position,within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_count_room = Order.transitive(Nat.add(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),4n),Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap),
    Order.add_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),Nat.double(ShapeLaws.elements(values)),4n,Order.double_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),ShapeLaws.elements(values),rooms_cells_fewer(limits,values,axes,lane,position,rooms))),
    count_room)
  kept_contract(~KeptFold<Accumulator,Logical>,~free(~KeptFold<Accumulator,Logical>),~Reduce.Lines<TiledReduce.Folds<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>,~Reduce.line_value(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),~fold_cells(~Accumulator,~Logical),~fold_target(~Accumulator,~Logical),~kept_agree(~KeptFold<Accumulator,Logical>,~free(~KeptFold<Accumulator,Logical>),~Reduce.Lines<TiledReduce.Folds<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>,~Reduce.line_value(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),~fold_cells(~Accumulator,~Logical),~fold_target(~Accumulator,~Logical),~fold_lane(~Accumulator,~operation,~finish,~Logical,~original)),KeptFold{logical,initial},limits,values,axes,lane,position,within,absent,rank_room,rooms,Unit{},lanes,pinned,default_capacity,Layout.reduce_axes(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),name,(operations * 8.0 : F32),TiledReduce.started(~Accumulator,initial),Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position)),
    Reduce.reduce_axes(~Tile.Tile8,~TiledReduce.Folds<Accumulator>,~TiledReduce.stepped(~Accumulator,~operation),~TiledReduce.finished(~Accumulator,~finish),~Representation.Lifted<Logical>,
      ~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},limits,name,(operations * 8.0 : F32),TiledReduce.started(~Accumulator,initial),TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position),model,Tiled.cell_shape(values,lane,position),fill,workers,global,
      default_capacity,cells_rank_room,rooms_cells_positive(limits,values,axes,lane,position,rooms),cap,cap_ok,cells_count_room,Boolean.and_true(Views.unique_axes(Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position))),Layout.all_below(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,position),List.length(&2,U32,Tiled.cell_shape(values,lane,position))),cells_unique(values,axes,lane,position,within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room),cells_below(values,axes,lane,position,within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room))))

# ---- Lanes along a reduced axis ----

# The first lane of a cell past lane.
def lane_first(+first: U32,lane: Nat) -> U32:
  match lane:
    case 0n: first
    case 1n+ +rest: (first + U32.from_nat(1n+rest) : U32)

# The lanes of a tile from lane on, each applied when its condition holds.
def lanes_applied(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,count: Nat,+first: U32,+lanes: U32,+tile: Tile.Tile8,+lane: Nat,value: Accumulator) -> Accumulator:
  match count:
    case 0n: value
    case 1n+rest:
      lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,1n+lane,
        TiledReduce.lane_applied(~Accumulator,~operation,U32.is_lt(lane_first(first,lane),lanes),value,TileLaws.select_lane(lane,tile)))

# The lanes of a tile from lane on, each applied.
def lanes_from(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,count: Nat,+tile: Tile.Tile8,+lane: Nat,value: Accumulator) -> Accumulator:
  match count:
    case 0n: value
    case 1n+rest: lanes_from(~Accumulator,~operation,rest,tile,1n+lane,operation(value,TileLaws.select_lane(lane,tile)))

# Lanes that all hold values: their condition holds.
def LanesAll(+first: U32,+lanes: U32,count: Nat,+lane: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: KernelGemm.Both<{U32.is_lt(lane_first(first,lane),lanes) == True{} : Bool},LanesAll(first,lanes,rest,1n+lane)>

def lanes_every(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,count: Nat,+first: U32,+lanes: U32,+tile: Tile.Tile8,+lane: Nat,+value: Accumulator,
  +every: LanesAll(first,lanes,count,lane)) ->
  {lanes_applied(~Accumulator,~operation,count,first,lanes,tile,lane,value) == lanes_from(~Accumulator,~operation,count,tile,lane,value) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+holds,+later} = every
      %Equal.sym(Bool,U32.is_lt(lane_first(first,lane),lanes),True{},holds) :
        {lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,1n+lane,TiledReduce.lane_applied(~Accumulator,~operation,_,value,TileLaws.select_lane(lane,tile))) ==
          lanes_from(~Accumulator,~operation,rest,tile,1n+lane,operation(value,TileLaws.select_lane(lane,tile))) : Accumulator}
      lanes_every(~Accumulator,~operation,rest,first,lanes,tile,1n+lane,operation(value,TileLaws.select_lane(lane,tile)),later)

# Each condition of all_hold holds when all of them do.
def all_lanes(+first: U32,+lanes: U32,+all: {TiledReduce.all_hold(first,lanes) == True{} : Bool}) -> LanesAll(first,lanes,8n,0n):
  +c0 = U32.is_lt(first,lanes)
  +c1 = U32.is_lt((first + 1 : U32),lanes)
  +c2 = U32.is_lt((first + 2 : U32),lanes)
  +c3 = U32.is_lt((first + 3 : U32),lanes)
  +c4 = U32.is_lt((first + 4 : U32),lanes)
  +c5 = U32.is_lt((first + 5 : U32),lanes)
  +c6 = U32.is_lt((first + 6 : U32),lanes)
  +c7 = U32.is_lt((first + 7 : U32),lanes)
  +h6 = Boolean.and_true_left(c0 && c1 && c2 && c3 && c4 && c5 && c6,c7,all)
  +h5 = Boolean.and_true_left(c0 && c1 && c2 && c3 && c4 && c5,c6,h6)
  +h4 = Boolean.and_true_left(c0 && c1 && c2 && c3 && c4,c5,h5)
  +h3 = Boolean.and_true_left(c0 && c1 && c2 && c3,c4,h4)
  +h2 = Boolean.and_true_left(c0 && c1 && c2,c3,h3)
  +h1 = Boolean.and_true_left(c0 && c1,c2,h2)
  KernelGemm.Both{Boolean.and_true_left(c0,c1,h1),KernelGemm.Both{Boolean.and_true_right(c0,c1,h1),
    KernelGemm.Both{Boolean.and_true_right(c0 && c1,c2,h2),KernelGemm.Both{Boolean.and_true_right(c0 && c1 && c2,c3,h3),
      KernelGemm.Both{Boolean.and_true_right(c0 && c1 && c2 && c3,c4,h4),KernelGemm.Both{Boolean.and_true_right(c0 && c1 && c2 && c3 && c4,c5,h5),
        KernelGemm.Both{Boolean.and_true_right(c0 && c1 && c2 && c3 && c4 && c5,c6,h6),KernelGemm.Both{Boolean.and_true_right(c0 && c1 && c2 && c3 && c4 && c5 && c6,c7,all),
          Unit{}}}}}}}}}

def all_from(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+value: Accumulator,tile: Tile.Tile8) ->
  {TiledReduce.lanes_all(~Accumulator,~operation,value,tile) == lanes_from(~Accumulator,~operation,8n,tile,0n,value) : Accumulator}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

def masked_applied(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+first: U32,+lanes: U32,+value: Accumulator,tile: Tile.Tile8) ->
  {TiledReduce.lanes_masked(~Accumulator,~operation,first,lanes,value,tile) == lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,value) : Accumulator}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

def whole_applied(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+whole: Bool,+first: U32,+lanes: U32,+value: Accumulator,+tile: Tile.Tile8,
  +same: {TiledReduce.all_hold(first,lanes) == whole : Bool}) ->
  {TiledReduce.lanes_whole(~Accumulator,~operation,whole,first,lanes,value,tile) == lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,value) : Accumulator}:
  match whole:
    case True{}:
      Equal.trans(Accumulator,TiledReduce.lanes_all(~Accumulator,~operation,value,tile),lanes_from(~Accumulator,~operation,8n,tile,0n,value),
        lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,value),all_from(~Accumulator,~operation,value,tile),
        Equal.sym(Accumulator,lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,value),lanes_from(~Accumulator,~operation,8n,tile,0n,value),
          lanes_every(~Accumulator,~operation,8n,first,lanes,tile,0n,value,all_lanes(first,lanes,same))))
    case False{}: masked_applied(~Accumulator,~operation,first,lanes,value,tile)

# A cell steps its lanes as they are applied one by one: all eight at once
# when each holds a value, otherwise each when it does.
def stepped_applied(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+first: U32,+lanes: U32,+value: Accumulator,+tile: Tile.Tile8) ->
  {TiledReduce.lanes_stepped(~Accumulator,~operation,first,lanes,value,tile) == lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,value) : Accumulator}:
  whole_applied(~Accumulator,~operation,TiledReduce.all_hold(first,lanes),first,lanes,value,tile,{==})

# Lanes past the values: their condition fails.
def LanesPast(+first: U32,+lanes: U32,count: Nat,+lane: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: KernelGemm.Both<{U32.is_lt(lane_first(first,lane),lanes) == False{} : Bool},LanesPast(first,lanes,rest,1n+lane)>

def lanes_past(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,count: Nat,+first: U32,+lanes: U32,+tile: Tile.Tile8,+lane: Nat,+value: Accumulator,
  +past: LanesPast(first,lanes,count,lane)) -> {lanes_applied(~Accumulator,~operation,count,first,lanes,tile,lane,value) == value : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+fails,+later} = past
      %Equal.sym(Bool,U32.is_lt(lane_first(first,lane),lanes),False{},fails) :
        {lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,1n+lane,TiledReduce.lane_applied(~Accumulator,~operation,_,value,TileLaws.select_lane(lane,tile))) == value : Accumulator}
      lanes_past(~Accumulator,~operation,rest,first,lanes,tile,1n+lane,value,later)

# Lanes that hold values: their condition holds and each reads the next value
# of a line.
def LanesHeld(~Context: Data,~value: Context -> U32 -> F32,+first: U32,+lanes: U32,+tile: Tile.Tile8,+context: Context,count: Nat,+lane: Nat,+position: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<{U32.is_lt(lane_first(first,lane),lanes) == True{} : Bool},
        KernelGemm.Both<{TileLaws.select_lane(lane,tile) == value(context,U32.from_nat(position)) : F32},LanesHeld(~Context,~value,first,lanes,tile,context,rest,1n+lane,1n+position)>>

def lanes_held(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~Context: Data,~value: Context -> U32 -> F32,count: Nat,+rest: Nat,+first: U32,+lanes: U32,+tile: Tile.Tile8,
  +context: Context,+lane: Nat,+position: Nat,+state: Accumulator,+held: LanesHeld(~Context,~value,first,lanes,tile,context,count,lane,position)) ->
  {lanes_applied(~Accumulator,~operation,Nat.add(count,rest),first,lanes,tile,lane,state) ==
    lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,Nat.add(count,lane),Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,count,position,state)) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +more:
      KernelGemm.Both{+holds,KernelGemm.Both{+reads,+later}} = held
      %Equal.sym(Bool,U32.is_lt(lane_first(first,lane),lanes),True{},holds) :
        {lanes_applied(~Accumulator,~operation,Nat.add(more,rest),first,lanes,tile,1n+lane,TiledReduce.lane_applied(~Accumulator,~operation,_,state,TileLaws.select_lane(lane,tile))) ==
          lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,1n+Nat.add(more,lane),Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,more,1n+position,operation(state,value(context,U32.from_nat(position))))) : Accumulator}
      %Equal.sym(F32,TileLaws.select_lane(lane,tile),value(context,U32.from_nat(position)),reads) :
        {lanes_applied(~Accumulator,~operation,Nat.add(more,rest),first,lanes,tile,1n+lane,operation(state,_)) ==
          lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,1n+Nat.add(more,lane),Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,more,1n+position,operation(state,value(context,U32.from_nat(position))))) : Accumulator}
      %Addition.successor_right(more,lane) :
        {lanes_applied(~Accumulator,~operation,Nat.add(more,rest),first,lanes,tile,1n+lane,operation(state,value(context,U32.from_nat(position)))) ==
          lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,_,Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,more,1n+position,operation(state,value(context,U32.from_nat(position))))) : Accumulator}
      lanes_held(~Accumulator,~operation,~Context,~value,more,rest,first,lanes,tile,context,1n+lane,1n+position,operation(state,value(context,U32.from_nat(position))),later)

# A cell whose first count lanes hold the next values of a line and whose
# other lanes hold none steps the fold of those values.
def cell_fold(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~Context: Data,~value: Context -> U32 -> F32,+count: Nat,+rest: Nat,+first: U32,+lanes: U32,+tile: Tile.Tile8,
  +context: Context,+position: Nat,+state: Accumulator,+eight: {Nat.add(count,rest) == 8n : Nat},+held: LanesHeld(~Context,~value,first,lanes,tile,context,count,0n,position),
  +past: LanesPast(first,lanes,rest,Nat.add(count,0n))) ->
  {TiledReduce.lanes_stepped(~Accumulator,~operation,first,lanes,state,tile) == Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,count,position,state) : Accumulator}:
  +folded = Reduce.line_fold(~F32,~Accumulator,~operation,~Context,~value,context,count,position,state)
  +applied = {%eight : {lanes_applied(~Accumulator,~operation,_,first,lanes,tile,0n,state) == folded : Accumulator}
    Equal.trans(Accumulator,lanes_applied(~Accumulator,~operation,Nat.add(count,rest),first,lanes,tile,0n,state),lanes_applied(~Accumulator,~operation,rest,first,lanes,tile,Nat.add(count,0n),folded),folded,
      lanes_held(~Accumulator,~operation,~Context,~value,count,rest,first,lanes,tile,context,0n,position,state,held),
      lanes_past(~Accumulator,~operation,rest,first,lanes,tile,Nat.add(count,0n),folded,past)) : {lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,state) == folded : Accumulator}}
  Equal.trans(Accumulator,TiledReduce.lanes_stepped(~Accumulator,~operation,first,lanes,state,tile),lanes_applied(~Accumulator,~operation,8n,first,lanes,tile,0n,state),folded,
    stepped_applied(~Accumulator,~operation,first,lanes,state,tile),applied)

# A fold of count and more values folds the first count, then the others.
def fold_split(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+more: Nat,+position: Nat,+state: Accumulator) ->
  {Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,Nat.add(count,more),position,state) ==
    Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,more,Nat.add(count,position),Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,count,position,state)) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Addition.successor_right(rest,position) :
        {Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,Nat.add(rest,more),1n+position,step(state,value(context,U32.from_nat(position)))) ==
          Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,more,_,Reduce.line_fold(~Element,~Accumulator,~step,~Context,~value,context,rest,1n+position,step(state,value(context,U32.from_nat(position))))) : Accumulator}
      fold_split(~Element,~Accumulator,~step,~Context,~value,context,rest,more,1n+position,step(state,value(context,U32.from_nat(position))))

# The condition of a lane of a cell whose first lane is eight times its group
# compares the lane of the values as a number.
def lane_condition(+limits: Machine.Limits,+first: U32,+lanes: U32,lane: Nat,+base: Nat,+first_exact: {U32.to_nat(first) == base : Nat},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+room: {Nat.is_le(Nat.add(base,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.is_lt(lane_first(first,lane),lanes) == Nat.is_lt(Nat.add(lane,base),U32.to_nat(lanes)) : Bool}:
  match lane:
    case 0n:
      %first_exact : {U32.is_lt(first,lanes) == Nat.is_lt(_,U32.to_nat(lanes)) : Bool}
      Comparison.lt(first,lanes)
    case 1n+ +rest:
      +offset = Nat.add(1n,rest)
      +offset_bound = Bounds.strict_le(offset,8n,small)
      +sum_room = Order.transitive(Nat.add(base,offset),Nat.add(base,8n),U32.to_nat(Machine.half_word(limits)),Order.offset_le(base,offset,8n,offset_bound),room)
      +offset_room = Order.transitive(offset,Nat.add(base,offset),U32.to_nat(Machine.half_word(limits)),Order.right_le(base,offset),sum_room)
      +sum = Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),first,U32.from_nat(offset),base,offset,first_exact,Bounds.roundtrip(offset,Machine.half_word(limits),Limits.half_word_fits(limits),offset_room),sum_room)
      %Order.commutative(base,offset) : {U32.is_lt((first + U32.from_nat(offset) : U32),lanes) == Nat.is_lt(_,U32.to_nat(lanes)) : Bool}
      %sum : {U32.is_lt((first + U32.from_nat(offset) : U32),lanes) == Nat.is_lt(_,U32.to_nat(lanes)) : Bool}
      Comparison.lt((first + U32.from_nat(offset) : U32),lanes)

# Whether the first lane eight past that of a cell lies before the lanes.
def next_compared(choice: Bool,+limits: Machine.Limits,+lanes: U32,+first: U32,+group: Nat,+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +half_room: {Nat.is_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+chosen: {U32.is_lt((first + 8 : U32),lanes) == choice : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(lanes)) == choice : Bool}:
  +sum = Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),first,8,Nat.mul(group,8n),8n,first_exact,{==},half_room)
  Equal.trans(Bool,Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(lanes)),U32.is_lt((first + 8 : U32),lanes),choice,
    Equal.sym(Bool,U32.is_lt((first + 8 : U32),lanes),Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(lanes)),
      {%sum : {U32.is_lt((first + 8 : U32),lanes) == Nat.is_lt(_,U32.to_nat(lanes)) : Bool} Comparison.lt((first + 8 : U32),lanes) :
        {U32.is_lt((first + 8 : U32),lanes) == Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(lanes)) : Bool}}),
    chosen)

# Whether a group follows the group of a cell.
def group_next(choice: Bool,+limits: Machine.Limits,+lanes: U32,+first: U32,+group: Nat,+facts: TiledValues.Groups<limits,lanes>,+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +half_room: {Nat.is_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+chosen: {U32.is_lt((first + 8 : U32),lanes) == choice : Bool}) ->
  {Nat.is_lt(1n+group,U32.to_nat(Grid.groups(lanes))) == choice : Bool}:
  TiledValues.Groups{+exact,+fewer,+room} = facts
  +compared = next_compared(choice,limits,lanes,first,group,first_exact,half_room,chosen)
  %Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),exact) : {Nat.is_lt(1n+group,_) == choice : Bool}
  %Equal.sym(Bool,Nat.is_lt(1n+group,Ceiling.ceiling(U32.to_nat(lanes),8n)),Nat.is_lt(Nat.mul(1n+group,8n),U32.to_nat(lanes)),Ceiling.upper_comparison(U32.to_nat(lanes),8n,1n+group,{==})) : {_ == choice : Bool}
  %Order.commutative(Nat.mul(group,8n),8n) : {Nat.is_lt(_,U32.to_nat(lanes)) == choice : Bool}
  compared

# The lanes of a group lie within half the word.
def group_half(+limits: Machine.Limits,+lanes: U32,+group: Nat,+facts: TiledValues.Groups<limits,lanes>,+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(lanes))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+room} = facts
  %Order.commutative(8n,Nat.mul(group,8n)) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(1n+group,8n),Nat.mul(U32.to_nat(Grid.groups(lanes)),8n),U32.to_nat(Machine.half_word(limits)),
    RowMajor.mul_le(1n+group,U32.to_nat(Grid.groups(lanes)),8n,Order.strict_follow(group,U32.to_nat(Grid.groups(lanes)),group_inside)),room)

def below_self(+value: Nat,+inside: {Nat.is_lt(value,value) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,value),True{},Equal.sym(Bool,Nat.is_lt(value,value),False{},Order.not_below(value,value,Order.reflexive(value))),inside))

def add_swap(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(first,Nat.add(second,third)) == Nat.add(second,Nat.add(first,third)) : Nat}:
  Equal.trans(Nat,Nat.add(first,Nat.add(second,third)),Nat.add(Nat.add(first,second),third),Nat.add(second,Nat.add(first,third)),
    Equal.sym(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Addition.associative(first,second,third)),
    Equal.trans(Nat,Nat.add(Nat.add(first,second),third),Nat.add(Nat.add(second,first),third),Nat.add(second,Nat.add(first,third)),
      Equal.cong(Nat,Nat,number => Nat.add(number,third),Nat.add(first,second),Nat.add(second,first),Order.commutative(first,second)),
      Addition.associative(second,first,third)))

# The next position of a line past two counts.
def position_next(+position: Nat,+front: Nat,+middle: Nat,+back: Nat,+exact: {position == Nat.add(front,Nat.add(middle,back)) : Nat}) ->
  {1n+position == Nat.add(front,Nat.add(middle,1n+back)) : Nat}:
  %Equal.sym(Nat,Nat.add(middle,1n+back),1n+Nat.add(middle,back),Addition.successor_right(middle,back)) : {1n+position == Nat.add(front,_) : Nat}
  %Equal.sym(Nat,Nat.add(front,1n+Nat.add(middle,back)),1n+Nat.add(front,Nat.add(middle,back)),Addition.successor_right(front,Nat.add(middle,back))) : {1n+position == _ : Nat}
  Equal.cong(Nat,Nat,number => 1n+number,position,Nat.add(front,Nat.add(middle,back)),exact)

def position_shifted(shift: Nat,+position: Nat,+front: Nat,+middle: Nat,+back: Nat,+exact: {position == Nat.add(front,Nat.add(middle,back)) : Nat}) ->
  {Nat.add(shift,position) == Nat.add(front,Nat.add(middle,Nat.add(shift,back))) : Nat}:
  match shift:
    case 0n: exact
    case 1n+ +rest: position_next(Nat.add(rest,position),front,middle,Nat.add(rest,back),position_shifted(rest,position,front,middle,back,exact))

def position_added(shift: Nat,+position: Nat,+front: Nat,+back: Nat,+exact: {position == Nat.add(front,back) : Nat}) ->
  {Nat.add(shift,position) == Nat.add(front,Nat.add(shift,back)) : Nat}:
  match shift:
    case 0n: exact
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(front,1n+Nat.add(rest,back)),1n+Nat.add(front,Nat.add(rest,back)),Addition.successor_right(front,Nat.add(rest,back))) : {1n+Nat.add(rest,position) == _ : Nat}
      Equal.cong(Nat,Nat,number => 1n+number,Nat.add(rest,position),Nat.add(front,Nat.add(rest,back)),position_added(rest,position,front,back,exact))

def line_value_of(~Accumulator: Data,line: TiledReduce.LaneLine<Accumulator>) -> Accumulator:
  TiledReduce.LaneLine{value,first,lanes} = line
  value

def line_finished(~Accumulator: Data,~finish: Accumulator -> F32,line: TiledReduce.LaneLine<Accumulator>) ->
  {TiledReduce.line_finish(~Accumulator,~finish,line) == Tiled.unit_tile(finish(line_value_of(~Accumulator,line))) : Tile.Tile8}:
  match line:
    case TiledReduce.LaneLine{+value,+first,+lanes}: {==}

# ---- The flags of a reduction along its last reduced axis ----

def none_picked(extents: List<&2,U32>,flags: List<&2,Bool>,+none: {TiledReduce.none_flagged(flags) == True{} : Bool}) -> {picked_extents(extents,flags,True{}) == [] : List<&2,U32>}:
  match extents flags:
    case Con{+extent,+rest} Con{True{},+more}: Empty.absurd({picked_extents(Con{extent,rest},Con{True{},more},True{}) == [] : List<&2,U32>},WordEncoding.false_true(none))
    case Con{+extent,+rest} Con{False{},+more}: none_picked(rest,more,none)
    case Nil{} Nil{}: {==}
    case Nil{} Con{+flag,+more}: {==}
    case Con{+extent,+rest} Nil{}: {==}

# Removing a reduced axis keeps the kept extents.
def removed_flagged(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat},+flagged: {Partials.flag_at(flags,axis) == True{} : Bool}) ->
  {picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),False{}) == picked_extents(extents,flags,False{}) : List<&2,U32>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n: {==}
    case Con{+extent,+rest} Con{False{},+more} 0n:
      Empty.absurd({picked_extents(Grid.removed(Con{extent,rest},0n),Partials.flag_removed(Con{False{},more},0n),False{}) == picked_extents(Con{extent,rest},Con{False{},more},False{}) : List<&2,U32>},WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{+flag,+more} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => kept_extent(flag,False{},extent,tail),picked_extents(Grid.removed(rest,next),Partials.flag_removed(more,next),False{}),picked_extents(rest,more,False{}),
        removed_flagged(rest,more,next,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same),flagged))
    case Nil{} Nil{} +axis: {==}
    case Nil{} Con{+flag,+more} +axis:
      Empty.absurd({picked_extents(Grid.removed(Nil{},axis),Partials.flag_removed(Con{flag,more},axis),False{}) == picked_extents(Nil{},Con{flag,more},False{}) : List<&2,U32>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({picked_extents(Grid.removed(Con{extent,rest},axis),Partials.flag_removed(Nil{},axis),False{}) == picked_extents(Con{extent,rest},Nil{},False{}) : List<&2,U32>},WordEncoding.false_true(flagged))

# The reduced extents end with the last reduced axis.
def picked_last_reduced(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+flagged: {Partials.flag_at(flags,axis) == True{} : Bool},+after: {TiledReduce.none_after(flags,axis) == True{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {picked_extents(extents,flags,True{}) == List.append(&2,U32,picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),True{}),[Grid.lanes_at(extents,axis)]) : List<&2,U32>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      +empty = Equal.sym(List<&2,U32>,picked_extents(rest,more,True{}),[],none_picked(rest,more,after))
      %empty : {extent <> _ == List.append(&2,U32,picked_extents(rest,more,True{}),[extent]) : List<&2,U32>}
      %empty : {extent <> [] == List.append(&2,U32,_,[extent]) : List<&2,U32>}
      {==}
    case Con{+extent,+rest} Con{False{},+more} 0n:
      Empty.absurd({picked_extents(Con{extent,rest},Con{False{},more},True{}) == List.append(&2,U32,picked_extents(Grid.removed(Con{extent,rest},0n),Partials.flag_removed(Con{False{},more},0n),True{}),[Grid.lanes_at(Con{extent,rest},0n)]) : List<&2,U32>},
        WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{+flag,+more} 1n+ +next:
      kept_appended(flag,True{},extent,picked_extents(rest,more,True{}),picked_extents(Grid.removed(rest,next),Partials.flag_removed(more,next),True{}),[Grid.lanes_at(rest,next)],
        picked_last_reduced(rest,more,next,flagged,after,inside))
    case Nil{} +flags +axis:
      Empty.absurd({picked_extents(Nil{},flags,True{}) == List.append(&2,U32,picked_extents(Grid.removed(Nil{},axis),Partials.flag_removed(flags,axis),True{}),[Grid.lanes_at(Nil{},axis)]) : List<&2,U32>},Order.below_zero(axis,inside))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({picked_extents(Con{extent,rest},Nil{},True{}) == List.append(&2,U32,picked_extents(Grid.removed(Con{extent,rest},axis),Partials.flag_removed(Nil{},axis),True{}),[Grid.lanes_at(Con{extent,rest},axis)]) : List<&2,U32>},
        WordEncoding.false_true(flagged))

# The digit of the last reduced axis put back at its axis.
def merged_reduced_last(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,kept: List<&2,Nat>,reduced: List<&2,Nat>,+digit: Nat,
  +kept_below: Coordinates.DigitsBelow(picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),False{}),kept),
  +reduced_below: Coordinates.DigitsBelow(picked_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis),True{}),reduced),
  +flagged: {Partials.flag_at(flags,axis) == True{} : Bool},+after: {TiledReduce.none_after(flags,axis) == True{} : Bool},+inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {Coordinates.inserted_digit(axis,merged(Partials.flag_removed(flags,axis),kept,reduced),digit) == merged(flags,kept,List.append(&2,Nat,reduced,[digit])) : List<&2,Nat>}:
  match extents flags axis:
    case Con{+extent,+rest} Con{True{},+more} 0n:
      match reduced:
        case Nil{}: {==}
        case Con{+first,+later}:
          Empty.absurd({Coordinates.inserted_digit(0n,merged(Partials.flag_removed(Con{True{},more},0n),kept,Con{first,later}),digit) == merged(Con{True{},more},kept,List.append(&2,Nat,Con{first,later},[digit])) : List<&2,Nat>},
            {%none_picked(rest,more,after) : Coordinates.DigitsBelow(_,Con{first,later}) reduced_below : Coordinates.DigitsBelow([],Con{first,later})})
    case Con{+extent,+rest} Con{False{},+more} 0n:
      Empty.absurd({Coordinates.inserted_digit(0n,merged(Partials.flag_removed(Con{False{},more},0n),kept,reduced),digit) == merged(Con{False{},more},kept,List.append(&2,Nat,reduced,[digit])) : List<&2,Nat>},
        WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{True{},+more} 1n+ +next:
      match reduced:
        case Nil{}:
          Empty.absurd({Coordinates.inserted_digit(1n+next,merged(Partials.flag_removed(Con{True{},more},1n+next),kept,Nil{}),digit) == merged(Con{True{},more},kept,List.append(&2,Nat,Nil{},[digit])) : List<&2,Nat>},reduced_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = reduced_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,Coordinates.inserted_digit(next,merged(Partials.flag_removed(more,next),kept,later),digit),merged(more,kept,List.append(&2,Nat,later,[digit])),
            merged_reduced_last(rest,more,next,kept,later,digit,kept_below,tail_below,flagged,after,inside))
    case Con{+extent,+rest} Con{False{},+more} 1n+ +next:
      match kept:
        case Nil{}:
          Empty.absurd({Coordinates.inserted_digit(1n+next,merged(Partials.flag_removed(Con{False{},more},1n+next),Nil{},reduced),digit) == merged(Con{False{},more},Nil{},List.append(&2,Nat,reduced,[digit])) : List<&2,Nat>},kept_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = kept_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,Coordinates.inserted_digit(next,merged(Partials.flag_removed(more,next),later,reduced),digit),merged(more,later,List.append(&2,Nat,reduced,[digit])),
            merged_reduced_last(rest,more,next,later,reduced,digit,tail_below,reduced_below,flagged,after,inside))
    case Nil{} +flags +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,merged(Partials.flag_removed(flags,axis),kept,reduced),digit) == merged(flags,kept,List.append(&2,Nat,reduced,[digit])) : List<&2,Nat>},Order.below_zero(axis,inside))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,merged(Partials.flag_removed(Nil{},axis),kept,reduced),digit) == merged(Nil{},kept,List.append(&2,Nat,reduced,[digit])) : List<&2,Nat>},WordEncoding.false_true(flagged))

# The digits of the cells with the group of the reduced lanes last.
def merged_last_reduced(extents: List<&2,U32>,flags: List<&2,Bool>,kept: List<&2,Nat>,reduced: List<&2,Nat>,+digit: Nat,
  +kept_below: Coordinates.DigitsBelow(picked_extents(extents,flags,False{}),kept),+reduced_below: Coordinates.DigitsBelow(picked_extents(extents,flags,True{}),reduced),
  +same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {merged(List.append(&2,Bool,flags,[True{}]),kept,List.append(&2,Nat,reduced,[digit])) == List.append(&2,Nat,merged(flags,kept,reduced),[digit]) : List<&2,Nat>}:
  match extents flags:
    case Nil{} Nil{}:
      match kept reduced:
        case Nil{} Nil{}: {==}
        case Con{+first,+more} +reduced:
          Empty.absurd({merged(List.append(&2,Bool,Nil{},[True{}]),Con{first,more},List.append(&2,Nat,reduced,[digit])) ==
            List.append(&2,Nat,merged(Nil{},Con{first,more},reduced),[digit]) : List<&2,Nat>},kept_below)
        case Nil{} Con{+first,+more}:
          Empty.absurd({merged(List.append(&2,Bool,Nil{},[True{}]),Nil{},List.append(&2,Nat,Con{first,more},[digit])) ==
            List.append(&2,Nat,merged(Nil{},Nil{},Con{first,more}),[digit]) : List<&2,Nat>},reduced_below)
    case Con{+extent,+rest} Con{True{},+more}:
      match reduced:
        case Nil{}:
          Empty.absurd({merged(List.append(&2,Bool,Con{True{},more},[True{}]),kept,List.append(&2,Nat,Nil{},[digit])) ==
            List.append(&2,Nat,merged(Con{True{},more},kept,Nil{}),[digit]) : List<&2,Nat>},reduced_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = reduced_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,merged(List.append(&2,Bool,more,[True{}]),kept,List.append(&2,Nat,later,[digit])),
            List.append(&2,Nat,merged(more,kept,later),[digit]),merged_last_reduced(rest,more,kept,later,digit,kept_below,tail_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Con{+extent,+rest} Con{False{},+more}:
      match kept:
        case Nil{}:
          Empty.absurd({merged(List.append(&2,Bool,Con{False{},more},[True{}]),Nil{},List.append(&2,Nat,reduced,[digit])) ==
            List.append(&2,Nat,merged(Con{False{},more},Nil{},reduced),[digit]) : List<&2,Nat>},kept_below)
        case Con{+first,+later}:
          KernelGemm.Both{here,+tail_below} = kept_below
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => first <> tail,merged(List.append(&2,Bool,more,[True{}]),later,List.append(&2,Nat,reduced,[digit])),
            List.append(&2,Nat,merged(more,later,reduced),[digit]),merged_last_reduced(rest,more,later,reduced,digit,tail_below,reduced_below,successor_same(List.length(&2,U32,rest),List.length(&2,Bool,more),same)))
    case Nil{} Con{+flag,+more}:
      Empty.absurd({merged(List.append(&2,Bool,Con{flag,more},[True{}]),kept,List.append(&2,Nat,reduced,[digit])) ==
        List.append(&2,Nat,merged(Con{flag,more},kept,reduced),[digit]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,Bool,more),same))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({merged(List.append(&2,Bool,Nil{},[True{}]),kept,List.append(&2,Nat,reduced,[digit])) ==
        List.append(&2,Nat,merged(Nil{},kept,reduced),[digit]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# ---- The shapes of a reduction along its last reduced axis ----

def flagged_of(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool}) -> {Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool}:
  Equal.trans(Bool,Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Views.contains_index(axes,lane),True{},flag_listed(List.length(&2,U32,values),0n,axes,lane,within),member)

def removed_same(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool}) -> {List.length(&2,U32,Grid.removed(values,lane)) == List.length(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)) : Nat}:
  removed_length(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same_of(values,axes))

def kept_along(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool}) -> {picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}:
  Equal.trans(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),removed_flagged(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,same_of(values,axes),flagged_of(values,axes,lane,within,member)),
    Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),False{}),extents_listed(values,axes,False{})))

def cells_flags_along(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+limits: Machine.Limits,+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))) == List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]) : List<&2,Bool>}:
  Equal.trans(List<&2,Bool>,Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),TiledReduce.flags_placed(Grid.Innermost{},Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane)),List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),
    cells_flags(values,axes,lane,Grid.Innermost{},within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room),
    Equal.cong(Bool,List<&2,Bool>,flag => TiledReduce.flags_placed(Grid.Innermost{},Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),flag),Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},flagged_of(values,axes,lane,within,member)))

# The kept cell axes are the kept axes.
def cells_kept_along(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+limits: Machine.Limits,+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}:
  +placed = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,lane,Grid.Innermost{}),Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,Grid.Innermost{},within)) : {picked_extents(_,Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),False{}) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}
    %Equal.sym(List<&2,Bool>,Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),cells_flags_along(values,axes,lane,within,member,limits,rank_room)) : {picked_extents(Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),_,False{}) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}
    Equal.trans(List<&2,U32>,picked_extents(Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),False{}),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),[]),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),
      picked_placed(Grid.Innermost{},Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},False{},Grid.groups(Grid.lanes_at(values,lane)),removed_same(values,axes,lane,within,member)),
      Equal.trans(List<&2,U32>,List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),[]),picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),MetadataLaws.append_empty(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{})),kept_along(values,axes,lane,within,member))) : {picked_extents(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),False{}) == Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})) : List<&2,U32>}}
  Equal.trans(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),picked_extents(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),extents_listed(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),placed)

# The reduced cell axes are the reduced axes but the lane axis, then the
# groups.
def cells_reduced_along(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+limits: Machine.Limits,+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})) == List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]) : List<&2,U32>}:
  +placed = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,lane,Grid.Innermost{}),Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,Grid.Innermost{},within)) : {picked_extents(_,Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),True{}) == List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]) : List<&2,U32>}
    %Equal.sym(List<&2,Bool>,Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),cells_flags_along(values,axes,lane,within,member,limits,rank_room)) : {picked_extents(Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),_,True{}) == List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]) : List<&2,U32>}
    picked_placed(Grid.Innermost{},Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{},True{},Grid.groups(Grid.lanes_at(values,lane)),removed_same(values,axes,lane,within,member)) : {picked_extents(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),True{}) == List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]) : List<&2,U32>}}
  Equal.trans(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),picked_extents(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),True{}),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),extents_listed(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{}),placed)

# The reduced axes are the reduced axes but the lane axis, then the lane axis.
def reduced_along(+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool}) -> {Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})) == List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]) : List<&2,U32>}:
  Equal.trans(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),picked_extents(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),True{}),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),extents_listed(values,axes,True{}),picked_last_reduced(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,flagged_of(values,axes,lane,within,member),after,within))

# What the cells of values and the kept values give within a capacity.
type AlongRooms<-limits: Machine.Limits,-values: List<&2,U32>,-axes: List<&2,Nat>,-lane: Nat> is Data:
  AlongRooms{values_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},values_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
    cells: AxisCells(limits,values,lane,Grid.Innermost{}),groups: TiledValues.Groups<limits,Grid.lanes_at(values,lane)>,cells_fewer: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),ShapeLaws.elements(values)) == True{} : Bool},
    kept_room: {Nat.is_le(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},kept_positive: {Nat.is_lt(0n,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool}}

def along_rooms(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},
  +values_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+values_positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) -> AlongRooms<limits,values,axes,lane>:
  +cells = Coordinates.axis_cells(True{},limits,values,lane,Grid.Innermost{},within,values_room,values_positive)
  +groups = TiledValues.grouped(limits,Grid.lanes_at(values,lane),lanes_room_of(limits,values,lane,Grid.Innermost{},cells))
  +product = Equal.trans(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(List.drop(&2,U32,values,0n)),ShapeLaws.elements(values),
    Reduce.partition_elements(values,List.length(&2,U32,values),0n,axes,{==}),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,values,0n),values,Reduce.dropped_none(~U32,values)))
  +product_positive = {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(values),product) : {Nat.is_lt(0n,_) == True{} : Bool} values_positive :
    {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))))) == True{} : Bool}}
  +kept_positive = RowMajor.left_factor_positive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),product_positive)
  +reduced_positive = DivisionOrder.right_factor_positive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),product_positive)
  +kept_room = Order.transitive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),reduced_positive),
    {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.elements(values),product) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} values_room :
      {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
  +cells_fewer = {%Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane))),Public.cells_split(values,lane,Grid.Innermost{},within)) :
      {Nat.is_le(_,ShapeLaws.elements(values)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(U32.to_nat(Grid.lanes_at(values,lane)),ShapeLaws.elements(Grid.removed(values,lane))),Public.elements_split(values,lane,within)) :
      {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane))),_) == True{} : Bool}
    RowMajor.mul_le(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),U32.to_nat(Grid.lanes_at(values,lane)),ShapeLaws.elements(Grid.removed(values,lane)),Coordinates.groups_fewer(limits,Grid.lanes_at(values,lane),groups)) :
    {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),ShapeLaws.elements(values)) == True{} : Bool}}
  AlongRooms{values_room,values_positive,cells,groups,cells_fewer,kept_room,kept_positive}

def along_cells_of(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+rooms: AlongRooms<limits,values,axes,lane>) -> AxisCells(limits,values,lane,Grid.Innermost{}):
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  cells

def along_fewer_of(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+rooms: AlongRooms<limits,values,axes,lane>) -> {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),ShapeLaws.elements(values)) == True{} : Bool}:
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  cells_fewer

def along_kept_room_of(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+rooms: AlongRooms<limits,values,axes,lane>) -> {Nat.is_le(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  kept_room

def groups_positive_of(+limits: Machine.Limits,+values: List<&2,U32>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+cells: AxisCells(limits,values,lane,Grid.Innermost{})) ->
  {Nat.is_lt(0n,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool}:
  RowMajor.left_factor_positive(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane)),
    {%Public.cells_split(values,lane,Grid.Innermost{},within) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive_of(limits,values,lane,Grid.Innermost{},cells) : {Nat.is_lt(0n,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),ShapeLaws.elements(Grid.removed(values,lane)))) == True{} : Bool}})

# ---- One lane of a cell along the lane axis ----

# A valid lane of the cell of a group of a block of a line reads the value of
# the specification at its place in the line.
def along_lane(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: AlongRooms<limits,values,axes,lane>,+word: Nat,+word_inside: {Nat.is_lt(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool},+block: Nat,+block_inside: {Nat.is_lt(block,ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}))) == True{} : Bool},+group: Nat,+digit: Nat,+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool},+small: {Nat.is_le(digit,7n) == True{} : Bool},
  +valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},+cell_position: Nat,+cell_exact: {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)) : Nat},
  +position: Nat,+position_exact: {position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))) : Nat}) ->
  {TileLaws.select_lane(digit,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))) == Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)) : F32}:
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  +flagged = flagged_of(values,axes,lane,within,member)
  +same_removed = removed_same(values,axes,lane,within,member)
  +kept_below = Digits.coordinates_below(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word,word_inside)
  +kept_value = Digits.value_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word,word_inside)
  +reduced_below = Digits.coordinates_below(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block,block_inside)
  +reduced_value = Digits.value_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block,block_inside)
  +kept_removed_below = {%Equal.sym(List<&2,U32>,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),kept_along(values,axes,lane,within,member)) : Coordinates.DigitsBelow(_,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)) kept_below : Coordinates.DigitsBelow(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),False{}),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word))}
  +cells_kept = cells_kept_along(values,axes,lane,within,member,limits,rank_room)
  +cells_reduced = cells_reduced_along(values,axes,lane,within,member,limits,rank_room)
  +kept_cells_below = {%Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),cells_kept) : Coordinates.DigitsBelow(_,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)) kept_below : Coordinates.DigitsBelow(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word))}
  +reduced_cells_below = {%Equal.sym(List<&2,U32>,Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),cells_reduced) : Coordinates.DigitsBelow(_,List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))
    Coordinates.digits_joined(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Grid.groups(Grid.lanes_at(values,lane))],[group],reduced_below,KernelGemm.Both{group_inside,Unit{}}) : Coordinates.DigitsBelow(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))}
  +cell_address = line_address(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]),kept_cells_below,reduced_cells_below,
    Machine.capacity(limits),Limits.capacity_fits(limits),cells_positive_of(limits,values,lane,Grid.Innermost{},cells),cells_room_of(limits,values,lane,Grid.Innermost{},cells))
  +kept_number = Equal.trans(Nat,ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),word,
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.digits_value(extents,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),cells_kept),kept_value)
  +cells_number = Equal.trans(Nat,ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group),
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.digits_value(extents,List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),cells_reduced),
    {%reduced_value : {ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])) == Nat.add(Nat.mul(_,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group) : Nat}
      Coordinates.digits_last(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),Grid.groups(Grid.lanes_at(values,lane)),group,reduced_below) : {ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])) == Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group) : Nat}})
  +cell_sum = {%kept_number : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))) == Nat.add(Nat.mul(_,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)) : Nat}
    %cells_number : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))) == Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),_) : Nat}
    {==} : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))) == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)) : Nat}}
  +cell_number = Equal.trans(Nat,Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),ShapeLaws.digits_value(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))),Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)),cell_position,
    cell_sum,Equal.sym(Nat,cell_position,Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)),cell_exact))
  +cell_read = {%cell_number : {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(_)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))))) : Tile.Tile8}
    cell_address : {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group]))))) : Tile.Tile8}}
  +merged_cells = Equal.trans(List<&2,Nat>,merged(Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),merged(List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group),
    Equal.cong(List<&2,Bool>,List<&2,Nat>,flags => merged(flags,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[group])),Layout.reduced_flags(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),List.append(&2,Bool,Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),[True{}]),cells_flags_along(values,axes,lane,within,member,limits,rank_room)),
    merged_last_reduced(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),group,kept_removed_below,reduced_below,same_removed))
  +others_below = merged_below(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),kept_removed_below,reduced_below,same_removed)
  +cell_below = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,lane,Grid.Innermost{}),Grid.placed(True{},Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane))),cells_placed(values,lane,Grid.Innermost{},within)) : Coordinates.DigitsBelow(_,Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))
    cell_below_of(Grid.Innermost{},Grid.removed(values,lane),Grid.groups(Grid.lanes_at(values,lane)),merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group,others_below,group_inside) : Coordinates.DigitsBelow(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))}
  +cell_room = Order.transitive(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group)),ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits)),
    Bounds.strict_le(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group)),ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),Coordinates.digits_below(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group),cell_below)),cells_room_of(limits,values,lane,Grid.Innermost{},cells))
  +lane_read = Coordinates.lane_digits(~Logical,~original,logical,limits,values,lane,Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group,U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))),within,removed_room_of(limits,values,lane,Grid.Innermost{},cells),others_below,group_inside,
    Bounds.roundtrip(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group)),Machine.capacity(limits),Limits.capacity_fits(limits),cell_room),digit,small)
  +cell_lane = Equal.trans(F32,TileLaws.select_lane(digit,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))))),
    original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n)))))),
    Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(digit,tile),Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)),Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group)))),
      {%merged_cells : {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),_))) : Tile.Tile8} cell_read :
        {Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group)))) : Tile.Tile8}}),
    {%valid : {TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))))) ==
        Bool.pick(F32,_,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n)))))),0.0) : F32} lane_read :
      {TileLaws.select_lane(digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},U32.from_nat(ShapeLaws.digits_value(Tiled.cell_shape(values,lane,Grid.Innermost{}),Coordinates.cell_digits(Grid.Innermost{},merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),group))))) == original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n)))))) : F32}})
  +reduced_last = reduced_along(values,axes,lane,within,member,after)
  +reduced_spec_below = {%Equal.sym(List<&2,U32>,Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),reduced_last) : Coordinates.DigitsBelow(_,List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))
    Coordinates.digits_joined(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Grid.lanes_at(values,lane)],[Nat.add(digit,Nat.mul(group,8n))],reduced_below,KernelGemm.Both{valid,Unit{}}) : Coordinates.DigitsBelow(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))}
  +spec_address = line_address(~F32,~Logical,~original,logical,values,axes,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]),kept_below,reduced_spec_below,Machine.capacity(limits),Limits.capacity_fits(limits),values_positive,values_room)
  +spec_reduced = Equal.trans(Nat,ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])),ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n))),
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.digits_value(extents,List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),reduced_last),
    {%reduced_value : {ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])) == Nat.add(Nat.mul(_,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n))) : Nat}
      Coordinates.digits_last(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),Grid.lanes_at(values,lane),Nat.add(digit,Nat.mul(group,8n)),reduced_below) : {ShapeLaws.digits_value(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])) == Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n))) : Nat}})
  +spec_sum = {%kept_value : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))) == Nat.add(Nat.mul(_,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))) : Nat}
    %spec_reduced : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))) == Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),_) : Nat}
    {==} : {Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))) == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))) : Nat}}
  +spec_number = Equal.trans(Nat,Nat.add(Nat.mul(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word)),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))),Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))),position,
    spec_sum,Equal.sym(Nat,position,Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))),position_exact))
  +spec_read = {%spec_number : {Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(_)) == original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))))) : F32} spec_address :
    {Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)) == original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))))) : F32}}
  +merged_spec = merged_reduced_last(values,Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane,Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),Nat.add(digit,Nat.mul(group,8n)),kept_removed_below,reduced_below,flagged,after,within)
  Equal.trans(F32,TileLaws.select_lane(digit,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n)))))),Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)),
    cell_lane,
    {%Equal.sym(List<&2,Nat>,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n))),merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))])),merged_spec) : {original(logical,U32.from_nat(ShapeLaws.digits_value(values,_))) == Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)) : F32}
      Equal.sym(F32,Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)),original(logical,U32.from_nat(ShapeLaws.digits_value(values,merged(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),List.append(&2,Nat,Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block),[Nat.add(digit,Nat.mul(group,8n))]))))),spec_read) :
      {original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(lane,merged(Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),Targets.cell_coordinates(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),word),Targets.cell_coordinates(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),block)),Nat.add(digit,Nat.mul(group,8n)))))) == Reduce.through(~F32,~Logical,~original,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.from_nat(position)) : F32}})

# ---- The cells of a line along the lane axis ----

# The lanes of a cell from lane on that hold values read the next values of
# the specification.
def held_lanes(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: AlongRooms<limits,values,axes,lane>,+word: Nat,+word_inside: {Nat.is_lt(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool},+block: Nat,+block_inside: {Nat.is_lt(block,ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}))) == True{} : Bool},count: Nat,+digit: Nat,+group: Nat,+first: U32,+cell_position: Nat,+position: Nat,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool},+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +half_room: {Nat.is_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +eight_room: {Nat.is_le(Nat.add(count,digit),8n) == True{} : Bool},+covered: {Nat.is_le(Nat.add(Nat.add(count,digit),Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool},
  +cell_exact: {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)) : Nat},
  +position_exact: {position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)))) : Nat}) ->
  LanesHeld(~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),first,Grid.lanes_at(values,lane),Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},count,digit,position):
  match count:
    case 0n: Unit{}
    case 1n+ +more:
      +inner = Order.right_le(more,digit)
      +small_lane = Bounds.nat_trans(digit,1n+Nat.add(more,digit),8n,Order.lt_succ(digit,Nat.add(more,digit),inner),eight_room)
      +valid = Bounds.nat_trans(Nat.add(digit,Nat.mul(group,8n)),1n+Nat.add(Nat.add(more,digit),Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane)),
        Order.lt_succ(Nat.add(digit,Nat.mul(group,8n)),Nat.add(Nat.add(more,digit),Nat.mul(group,8n)),Order.add_le(digit,Nat.add(more,digit),Nat.mul(group,8n),inner)),covered)
      +holds = Equal.trans(Bool,U32.is_lt(lane_first(first,digit),Grid.lanes_at(values,lane)),Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))),True{},
        lane_condition(limits,first,Grid.lanes_at(values,lane),digit,Nat.mul(group,8n),first_exact,small_lane,half_room),valid)
      +reads = along_lane(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,group,digit,group_inside,Order.strict_follow(digit,8n,small_lane),valid,cell_position,cell_exact,position,position_exact)
      +successor = Equal.sym(Nat,Nat.add(more,1n+digit),1n+Nat.add(more,digit),Addition.successor_right(more,digit))
      +later_room = {%successor : {Nat.is_le(_,8n) == True{} : Bool} eight_room : {Nat.is_le(Nat.add(more,1n+digit),8n) == True{} : Bool}}
      +later_covered = {%successor : {Nat.is_le(Nat.add(_,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool} covered : {Nat.is_le(Nat.add(Nat.add(more,1n+digit),Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}}
      +later_position = position_next(position,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(digit,Nat.mul(group,8n)),position_exact)
      KernelGemm.Both{holds,KernelGemm.Both{reads,held_lanes(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,more,1n+digit,group,first,cell_position,1n+position,group_inside,first_exact,half_room,later_room,later_covered,cell_exact,later_position)}}

# The lanes of a cell from lane on past the values hold none.
def past_lanes(+limits: Machine.Limits,+lanes: U32,count: Nat,+lane: Nat,+group: Nat,+first: U32,+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +half_room: {Nat.is_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+eight_room: {Nat.is_le(Nat.add(count,lane),8n) == True{} : Bool},
  +beyond: {Nat.is_le(U32.to_nat(lanes),Nat.add(lane,Nat.mul(group,8n))) == True{} : Bool}) -> LanesPast(first,lanes,count,lane):
  match count:
    case 0n: Unit{}
    case 1n+ +more:
      +small_lane = Bounds.nat_trans(lane,1n+Nat.add(more,lane),8n,Order.lt_succ(lane,Nat.add(more,lane),Order.right_le(more,lane)),eight_room)
      +fails = Equal.trans(Bool,U32.is_lt(lane_first(first,lane),lanes),Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)),False{},
        lane_condition(limits,first,lanes,lane,Nat.mul(group,8n),first_exact,small_lane,half_room),Order.not_below(U32.to_nat(lanes),Nat.add(lane,Nat.mul(group,8n)),beyond))
      +later_room = {%Equal.sym(Nat,Nat.add(more,1n+lane),1n+Nat.add(more,lane),Addition.successor_right(more,lane)) : {Nat.is_le(_,8n) == True{} : Bool} eight_room :
        {Nat.is_le(Nat.add(more,1n+lane),8n) == True{} : Bool}}
      KernelGemm.Both{fails,past_lanes(limits,lanes,more,1n+lane,group,first,first_exact,half_room,later_room,Order.successor(U32.to_nat(lanes),Nat.add(lane,Nat.mul(group,8n)),beyond))}

# The cells of a block of a line from a group on fold the values of the
# specification there, and the first lane is 0 again past the last group.
def along_block(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: AlongRooms<limits,values,axes,lane>,+word: Nat,+word_inside: {Nat.is_lt(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool},+block: Nat,+block_inside: {Nat.is_lt(block,ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}))) == True{} : Bool},remaining: Nat,choice: Bool,+group: Nat,+first: U32,+cell_position: Nat,+spec_position: Nat,+spec_remaining: Nat,+state: Accumulator,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool},+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +groups_fit: {Nat.add(group,remaining) == U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))) : Nat},+spec_fit: {Nat.add(Nat.mul(group,8n),spec_remaining) == U32.to_nat(Grid.lanes_at(values,lane)) : Nat},
  +cell_exact: {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group)) : Nat},
  +spec_exact: {spec_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.mul(group,8n))) : Nat},+chosen: {U32.is_lt((first + 8 : U32),Grid.lanes_at(values,lane)) == choice : Bool}) ->
  {Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},remaining,cell_position,TiledReduce.LaneLine{state,first,Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}:
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  match remaining:
    case 0n:
      +same = Equal.trans(Nat,group,Nat.add(group,0n),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Equal.sym(Nat,Nat.add(group,0n),group,Addition.zero_right(group)),groups_fit)
      Empty.absurd({Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},0n,cell_position,TiledReduce.LaneLine{state,first,Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>},
        below_self(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),{%same : {Nat.is_lt(_,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool} group_inside : {Nat.is_lt(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool}}))
    case 1n+ +rest:
      match rest choice:
        case 0n True{}:
          +half_room = group_half(limits,Grid.lanes_at(values,lane),group,groups,group_inside)
          +next_inside = group_next(True{},limits,Grid.lanes_at(values,lane),first,group,groups,first_exact,half_room,chosen)
          +last = Equal.trans(Nat,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Nat.add(group,1n),1n+group,Equal.sym(Nat,Nat.add(group,1n),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),groups_fit),
            Equal.trans(Nat,Nat.add(group,1n),1n+Nat.add(group,0n),1n+group,Addition.successor_right(group,0n),Equal.cong(Nat,Nat,number => 1n+number,Nat.add(group,0n),group,Addition.zero_right(group))))
          Empty.absurd({Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n,cell_position,TiledReduce.LaneLine{state,first,Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>},
            below_self(1n+group,{%last : {Nat.is_lt(1n+group,_) == True{} : Bool} next_inside : {Nat.is_lt(1n+group,1n+group) == True{} : Bool}}))
        case 0n False{}:
          +half_room = group_half(limits,Grid.lanes_at(values,lane),group,groups,group_inside)
          +compared = next_compared(False{},limits,Grid.lanes_at(values,lane),first,group,first_exact,half_room,chosen)
          +beyond = Order.not_less(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Grid.lanes_at(values,lane)),compared)
          +spec_le = Order.cancel_offset_le(Nat.mul(group,8n),spec_remaining,8n,
            {%Equal.sym(Nat,Nat.add(Nat.mul(group,8n),spec_remaining),U32.to_nat(Grid.lanes_at(values,lane)),spec_fit) : {Nat.is_le(_,Nat.add(Nat.mul(group,8n),8n)) == True{} : Bool} beyond : {Nat.is_le(Nat.add(Nat.mul(group,8n),spec_remaining),Nat.add(Nat.mul(group,8n),8n)) == True{} : Bool}})
          +eight = Order.restore_shift(spec_remaining,8n,spec_le)
          +ends = {%Order.commutative(Nat.mul(group,8n),spec_remaining) : {_ == U32.to_nat(Grid.lanes_at(values,lane)) : Nat} spec_fit : {Nat.add(spec_remaining,Nat.mul(group,8n)) == U32.to_nat(Grid.lanes_at(values,lane)) : Nat}}
          +zero_lane = Equal.sym(Nat,Nat.add(spec_remaining,0n),spec_remaining,Addition.zero_right(spec_remaining))
          +exact_end = {%ends : {Nat.is_le(Nat.add(spec_remaining,Nat.mul(group,8n)),_) == True{} : Bool} Order.reflexive(Nat.add(spec_remaining,Nat.mul(group,8n))) : {Nat.is_le(Nat.add(spec_remaining,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}}
          +covered = {%zero_lane : {Nat.is_le(Nat.add(_,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool} exact_end : {Nat.is_le(Nat.add(Nat.add(spec_remaining,0n),Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}}
          +held_room = {%zero_lane : {Nat.is_le(_,8n) == True{} : Bool} spec_le : {Nat.is_le(Nat.add(spec_remaining,0n),8n) == True{} : Bool}}
          +held = held_lanes(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,spec_remaining,0n,group,first,cell_position,spec_position,group_inside,first_exact,half_room,held_room,covered,cell_exact,spec_exact)
          +past_count = Nat.sub(8n,spec_remaining)
          +eight_le = {%Equal.sym(Nat,Nat.add(spec_remaining,past_count),8n,eight) : {Nat.is_le(_,8n) == True{} : Bool} Order.reflexive(8n) : {Nat.is_le(Nat.add(spec_remaining,past_count),8n) == True{} : Bool}}
          +past_room = {%zero_lane : {Nat.is_le(Nat.add(past_count,_),8n) == True{} : Bool}
            %Order.commutative(spec_remaining,past_count) : {Nat.is_le(_,8n) == True{} : Bool}
            eight_le : {Nat.is_le(Nat.add(past_count,Nat.add(spec_remaining,0n)),8n) == True{} : Bool}}
          +beyond_past = {%zero_lane : {Nat.is_le(U32.to_nat(Grid.lanes_at(values,lane)),Nat.add(_,Nat.mul(group,8n))) == True{} : Bool}
            %ends : {Nat.is_le(_,Nat.add(spec_remaining,Nat.mul(group,8n))) == True{} : Bool}
            Order.reflexive(Nat.add(spec_remaining,Nat.mul(group,8n))) : {Nat.is_le(U32.to_nat(Grid.lanes_at(values,lane)),Nat.add(Nat.add(spec_remaining,0n),Nat.mul(group,8n))) == True{} : Bool}}
          +past = past_lanes(limits,Grid.lanes_at(values,lane),past_count,Nat.add(spec_remaining,0n),group,first,first_exact,half_room,past_room,beyond_past)
          +cell = cell_fold(~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),spec_remaining,past_count,first,Grid.lanes_at(values,lane),Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_position,state,eight,held,past)
          %Equal.sym(Bool,U32.is_lt((first + 8 : U32),Grid.lanes_at(values,lane)),False{},chosen) :
            {TiledReduce.LaneLine{TiledReduce.lanes_stepped(~Accumulator,~operation,first,Grid.lanes_at(values,lane),state,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),Bool.pick(U32,_,(first + 8 : U32),0),Grid.lanes_at(values,lane)} ==
              TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}
          Equal.cong(Accumulator,TiledReduce.LaneLine<Accumulator>,value => TiledReduce.LaneLine{value,0,Grid.lanes_at(values,lane)},TiledReduce.lanes_stepped(~Accumulator,~operation,first,Grid.lanes_at(values,lane),state,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),cell)
        case 1n+ +later True{}:
          +half_room = group_half(limits,Grid.lanes_at(values,lane),group,groups,group_inside)
          +compared = next_compared(True{},limits,Grid.lanes_at(values,lane),first,group,first_exact,half_room,chosen)
          +next_inside = group_next(True{},limits,Grid.lanes_at(values,lane),first,group,groups,first_exact,half_room,chosen)
          +more_spec = Order.cancel_offset_lt(Nat.mul(group,8n),8n,spec_remaining,
            {%Equal.sym(Nat,Nat.add(Nat.mul(group,8n),spec_remaining),U32.to_nat(Grid.lanes_at(values,lane)),spec_fit) : {Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),_) == True{} : Bool} compared : {Nat.is_lt(Nat.add(Nat.mul(group,8n),8n),Nat.add(Nat.mul(group,8n),spec_remaining)) == True{} : Bool}})
          +spec_later = Nat.sub(spec_remaining,8n)
          +restore = Order.restore_shift(8n,spec_remaining,Bounds.strict_le(8n,spec_remaining,more_spec))
          +covered = {%Order.commutative(Nat.mul(group,8n),8n) : {Nat.is_le(_,U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool} Bounds.strict_le(Nat.add(Nat.mul(group,8n),8n),U32.to_nat(Grid.lanes_at(values,lane)),compared) : {Nat.is_le(Nat.add(8n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}}
          +held = held_lanes(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,8n,0n,group,first,cell_position,spec_position,group_inside,first_exact,half_room,{==},covered,cell_exact,spec_exact)
          +cell = cell_fold(~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),8n,0n,first,Grid.lanes_at(values,lane),Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position)),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_position,state,{==},held,Unit{})
          +sum = Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),first,8,Nat.mul(group,8n),8n,first_exact,{==},half_room)
          +next_first = Equal.trans(Nat,U32.to_nat((first + 8 : U32)),Nat.add(Nat.mul(group,8n),8n),Nat.mul(1n+group,8n),sum,Order.commutative(Nat.mul(group,8n),8n))
          +next_fit = Equal.trans(Nat,Nat.add(1n+group,1n+later),Nat.add(group,1n+1n+later),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Equal.sym(Nat,Nat.add(group,1n+1n+later),1n+Nat.add(group,1n+later),Addition.successor_right(group,1n+later)),groups_fit)
          +next_spec_fit = Equal.trans(Nat,Nat.add(Nat.mul(1n+group,8n),spec_later),Nat.add(8n,Nat.add(Nat.mul(group,8n),spec_later)),U32.to_nat(Grid.lanes_at(values,lane)),Addition.associative(8n,Nat.mul(group,8n),spec_later),
            Equal.trans(Nat,Nat.add(8n,Nat.add(Nat.mul(group,8n),spec_later)),Nat.add(Nat.mul(group,8n),Nat.add(8n,spec_later)),U32.to_nat(Grid.lanes_at(values,lane)),add_swap(8n,Nat.mul(group,8n),spec_later),
              {%Equal.sym(Nat,Nat.add(8n,spec_later),spec_remaining,restore) : {Nat.add(Nat.mul(group,8n),_) == U32.to_nat(Grid.lanes_at(values,lane)) : Nat} spec_fit : {Nat.add(Nat.mul(group,8n),Nat.add(8n,spec_later)) == U32.to_nat(Grid.lanes_at(values,lane)) : Nat}}))
          +next_cell = position_next(cell_position,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),group,cell_exact)
          +next_spec = position_shifted(8n,spec_position,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Nat.mul(group,8n),spec_exact)
          +rest_fold = along_block(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,1n+later,U32.is_lt(((first + 8 : U32) + 8 : U32),Grid.lanes_at(values,lane)),1n+group,(first + 8 : U32),1n+cell_position,Nat.add(8n,spec_position),spec_later,
            Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_position,state),next_inside,next_first,next_fit,next_spec_fit,next_cell,next_spec,{==})
          %Equal.sym(Bool,U32.is_lt((first + 8 : U32),Grid.lanes_at(values,lane)),True{},chosen) :
            {Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n+later,1n+cell_position,TiledReduce.LaneLine{TiledReduce.lanes_stepped(~Accumulator,~operation,first,Grid.lanes_at(values,lane),state,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),Bool.pick(U32,_,(first + 8 : U32),0),Grid.lanes_at(values,lane)}) ==
              TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}
          %Equal.sym(Accumulator,TiledReduce.lanes_stepped(~Accumulator,~operation,first,Grid.lanes_at(values,lane),state,Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.from_nat(cell_position))),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_position,state),cell) :
            {Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n+later,1n+cell_position,TiledReduce.LaneLine{_,(first + 8 : U32),Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}
          %restore :
            {Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n+later,1n+cell_position,TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_position,state),(first + 8 : U32),Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},_,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}
          %Equal.sym(Accumulator,Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.add(8n,spec_later),spec_position,state),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_later,Nat.add(8n,spec_position),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_position,state)),
              fold_split(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_later,spec_position,state)) :
            {Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n+later,1n+cell_position,TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},8n,spec_position,state),(first + 8 : U32),Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{_,0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>}
          rest_fold
        case 1n+ +later False{}:
          +half_room = group_half(limits,Grid.lanes_at(values,lane),group,groups,group_inside)
          +next_outside = group_next(False{},limits,Grid.lanes_at(values,lane),first,group,groups,first_exact,half_room,chosen)
          +more_groups = Equal.trans(Nat,1n+Nat.add(group,1n+later),Nat.add(group,1n+1n+later),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Equal.sym(Nat,Nat.add(group,1n+1n+later),1n+Nat.add(group,1n+later),Addition.successor_right(group,1n+later)),groups_fit)
          +inside_more = {%more_groups : {Nat.is_lt(1n+group,_) == True{} : Bool} Order.below_sum(1n+group,later) : {Nat.is_lt(1n+group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))) == True{} : Bool}}
          Empty.absurd({Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},1n+1n+later,cell_position,TiledReduce.LaneLine{state,first,Grid.lanes_at(values,lane)}) == TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},spec_remaining,spec_position,state),0,Grid.lanes_at(values,lane)} : TiledReduce.LaneLine<Accumulator>},
            WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(1n+group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),True{},Equal.sym(Bool,Nat.is_lt(1n+group,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),False{},next_outside),inside_more)))

# The blocks of a line from a block on fold the values of the specification
# there.
def along_lines(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: AlongRooms<limits,values,axes,lane>,+word: Nat,+word_inside: {Nat.is_lt(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool},blocks: Nat,+block: Nat,+cell_position: Nat,+spec_position: Nat,+state: Accumulator,
  +blocks_fit: {Nat.add(block,blocks) == ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})) : Nat},
  +cell_exact: {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))))) : Nat},
  +spec_exact: {spec_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane)))) : Nat}) ->
  {line_value_of(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(blocks,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)})) ==
    Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(blocks,U32.to_nat(Grid.lanes_at(values,lane))),spec_position,state) : Accumulator}:
  AlongRooms{+values_room,+values_positive,+cells,+groups,+cells_fewer,+kept_room,+kept_positive} = rooms
  match blocks:
    case 0n: {==}
    case 1n+ +more:
      +block_inside = {%blocks_fit : {Nat.is_lt(block,_) == True{} : Bool} Order.below_sum(block,more) : {Nat.is_lt(block,ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}))) == True{} : Bool}}
      +block_cells = along_block(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,block,block_inside,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),U32.is_lt((0 + 8 : U32),Grid.lanes_at(values,lane)),0n,0,cell_position,spec_position,U32.to_nat(Grid.lanes_at(values,lane)),state,
        groups_positive_of(limits,values,lane,within,cells),{==},{==},{==},
        {%Equal.sym(Nat,Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),0n),Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Addition.zero_right(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))))) : {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),_) : Nat} cell_exact :
          {cell_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),0n)) : Nat}},
        {%Equal.sym(Nat,Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),0n),Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),Addition.zero_right(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))))) : {spec_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),_) : Nat} spec_exact :
          {spec_position == Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.add(Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),0n)) : Nat}},{==})
      +next_fit = Equal.trans(Nat,Nat.add(1n+block,more),Nat.add(block,1n+more),ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),Equal.sym(Nat,Nat.add(block,1n+more),1n+Nat.add(block,more),Addition.successor_right(block,more)),blocks_fit)
      +next_cell = position_added(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.mul(block,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),cell_exact)
      +next_spec = position_added(U32.to_nat(Grid.lanes_at(values,lane)),spec_position,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Nat.mul(block,U32.to_nat(Grid.lanes_at(values,lane))),spec_exact)
      +rest_lines = along_lines(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,word_inside,more,1n+block,Nat.add(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position),Nat.add(U32.to_nat(Grid.lanes_at(values,lane)),spec_position),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),spec_position,state),next_fit,next_cell,next_spec)
      %Equal.sym(Accumulator,Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.add(U32.to_nat(Grid.lanes_at(values,lane)),Nat.mul(more,U32.to_nat(Grid.lanes_at(values,lane)))),spec_position,state),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(more,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(U32.to_nat(Grid.lanes_at(values,lane)),spec_position),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),spec_position,state)),
          fold_split(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),Nat.mul(more,U32.to_nat(Grid.lanes_at(values,lane))),spec_position,state)) :
        {line_value_of(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.add(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Nat.mul(more,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)})) == _ : Accumulator}
      %Equal.sym(TiledReduce.LaneLine<Accumulator>,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.add(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Nat.mul(more,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)}),Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(more,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.add(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position),Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)})),
          fold_split(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Nat.mul(more,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)})) :
        {line_value_of(~Accumulator,_) == Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(more,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(U32.to_nat(Grid.lanes_at(values,lane)),spec_position),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),spec_position,state)) : Accumulator}
      %Equal.sym(TiledReduce.LaneLine<Accumulator>,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position,TiledReduce.LaneLine{state,0,Grid.lanes_at(values,lane)}),TiledReduce.LaneLine{Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),spec_position,state),0,Grid.lanes_at(values,lane)},block_cells) :
        {line_value_of(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(more,U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.add(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),cell_position),_)) == Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(more,U32.to_nat(Grid.lanes_at(values,lane))),Nat.add(U32.to_nat(Grid.lanes_at(values,lane)),spec_position),Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},U32.to_nat(Grid.lanes_at(values,lane)),spec_position,state)) : Accumulator}
      rest_lines

# A line of the cells along the lane axis finishes on the unit axis as the
# line of the specification.
def along_line(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(axes,lane) == True{} : Bool},+after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: AlongRooms<limits,values,axes,lane>,+initial: Accumulator,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))) == True{} : Bool}) ->
  {Reduce.line_value(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Lines{TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},index) == unit_of(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},index) : Tile.Tile8}:
  +word = U32.to_nat(index)
  +cells_count = Equal.trans(Nat,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{}))),ShapeLaws.elements(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))])),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),cells_reduced_along(values,axes,lane,within,member,limits,rank_room)),
    Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))])),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),1n)),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),ShapeLaws.elements_appended(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.groups(Grid.lanes_at(values,lane))]),
      Equal.cong(Nat,Nat,number => Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),number),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),1n),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane))),Order.times_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))))))
  +spec_count = Equal.trans(Nat,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),ShapeLaws.elements(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)])),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})),List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),reduced_along(values,axes,lane,within,member,after)),
    Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)])),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),Nat.mul(U32.to_nat(Grid.lanes_at(values,lane)),1n)),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),ShapeLaws.elements_appended(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{}),[Grid.lanes_at(values,lane)]),
      Equal.cong(Nat,Nat,number => Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),number),Nat.mul(U32.to_nat(Grid.lanes_at(values,lane)),1n),U32.to_nat(Grid.lanes_at(values,lane)),Order.times_one(U32.to_nat(Grid.lanes_at(values,lane))))))
  +lines = along_lines(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,word,inside,ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),0n,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial,{==},
    Equal.sym(Nat,Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),0n),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),Addition.zero_right(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))))),
    Equal.sym(Nat,Nat.add(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),0n),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),Addition.zero_right(Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))))))
  %Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{}))),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),spec_count) :
    {TiledReduce.line_finish(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{}))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})) == Tiled.unit_tile(finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},_,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial))) : Tile.Tile8}
  %Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{}))),Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),cells_count) :
    {TiledReduce.line_finish(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},_,Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})) == Tiled.unit_tile(finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial))) : Tile.Tile8}
  Equal.trans(Tile.Tile8,TiledReduce.line_finish(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})),
    Tiled.unit_tile(finish(line_value_of(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})))),
    Tiled.unit_tile(finish(Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial))),
    line_finished(~Accumulator,~finish,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})),
    Equal.cong(Accumulator,Tile.Tile8,value => Tiled.unit_tile(finish(value)),line_value_of(~Accumulator,Reduce.line_fold(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.groups(Grid.lanes_at(values,lane)))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)})),
      Reduce.line_fold(~F32,~Accumulator,~operation,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original),Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},Nat.mul(ShapeLaws.elements(picked_extents(Grid.removed(values,lane),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,values),0n,axes),lane),True{})),U32.to_nat(Grid.lanes_at(values,lane))),Nat.mul(word,ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))),initial),lines))

# ---- Lanes along the last reduced axis ----

# A reduction along the lane axis, the last reduced axis, folds the lanes of
# the cells of each line that hold values in order: the tiled values of the
# specification on the unit axis, their lanes moved to the longest kept axis.
def along_route(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+lanes: U32,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},lane,lanes,Grid.Innermost{}}),
  +within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+member: {Views.contains_index(Layout.naturals(listed),lane) == True{} : Bool},
  +after: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,Layout.naturals(listed)),lane) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Along{},name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},lane,lanes,Grid.Innermost{}})):
  KernelGemm.Both{+global,+laid} = contract
  Public.Laid{+rank_inside,+shaped_cells,+restored,+counted,+pinned} = laid
  +axes = Layout.naturals(listed)
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +rooms = along_rooms(limits,values,axes,lane,within,values_room,nonempty)
  +cells = along_cells_of(limits,values,axes,lane,rooms)
  +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),List.length(&2,U32,values),cells_rank(values,lane,Grid.Innermost{},within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_count_room = Order.transitive(Nat.add(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{}))),4n),Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap),
    Order.add_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{}))),Nat.double(ShapeLaws.elements(values)),4n,Order.double_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,Grid.Innermost{})),ShapeLaws.elements(values),along_fewer_of(limits,values,axes,lane,rooms))),
    count_room)
  +kept = Reduce.kept_axes(values,listed)
  +count = List.length(&2,Nat,kept)
  +longest = TiledReduce.longest_kept(kept,values,0n,count,0)
  +reduced_cells = Reduce.reduce_axes(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},limits,name,(operations * 8.0 : F32),
    TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}),model,Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,workers,global,default_capacity,cells_rank_room,cells_positive_of(limits,values,lane,Grid.Innermost{},cells),cap,cap_ok,cells_count_room,Boolean.and_true(Views.unique_axes(Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))),Layout.all_below(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}),List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{}))),cells_unique(values,axes,lane,Grid.Innermost{},within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room),cells_below(values,axes,lane,Grid.Innermost{},within,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room)))
  +shaped_kept = {%cells_kept_along(values,axes,lane,within,member,limits,rank_room) : Global.Shaped(~Tile.Tile8,~Reduce.Lines<TiledReduce.LaneLine<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>,~Reduce.line_value(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),Reduce.Lines{TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},_,Layout.reduce_axes(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),name,(operations * 8.0 : F32),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{}))) reduced_cells : Global.Shaped(~Tile.Tile8,~Reduce.Lines<TiledReduce.LaneLine<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>,~Reduce.line_value(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),Reduce.Lines{TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Layout.reduce_axes(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),name,(operations * 8.0 : F32),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})))}
  +restated = Global.restated_below(~Tile.Tile8,~Reduce.Lines<TiledReduce.LaneLine<Accumulator>,Reduce.Ordered<Representation.Lifted<Logical>>>,~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),~Reduce.Ordered<Representation.Lifted<Logical>>,~Reduce.through(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),~unit_of(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original))),~ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}))),Reduce.Lines{TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Reduce.Ordered{Representation.Lifted{logical,Public.geometry_of(values,lane,Grid.Innermost{})},Layout.line_view(Tiled.cell_shape(values,lane,Grid.Innermost{}),List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),False{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},ShapeLaws.elements(Layout.extents_at(Tiled.cell_shape(values,lane,Grid.Innermost{}),Layout.axes_where(List.length(&2,U32,Tiled.cell_shape(values,lane,Grid.Innermost{})),0n,Layout.naturals(TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),True{})))},Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,True{})))},
    index => inside => along_line(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,values,axes,lane,within,member,after,rank_room,rooms,initial,index,inside),Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Layout.reduce_axes(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),name,(operations * 8.0 : F32),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),shaped_kept,Order.reflexive(ShapeLaws.elements(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})))),Machine.capacity(limits),along_kept_room_of(limits,values,axes,lane,rooms))
  %pinned : Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Along{},name,operations,initial,values,listed,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},lane,_,Grid.Innermost{}}))
  unit_result(~Reduce.Lines<Accumulator,Reduce.Ordered<Logical>>,~Reduce.line_value(~F32,~Accumulator,~operation,~finish,~Reduce.Ordered<Logical>,~Reduce.through(~F32,~Logical,~original)),Reduce.Lines{initial,Reduce.Ordered{logical,Layout.line_view(values,List.append(&2,Nat,Reduce.kept_axes(values,listed),Reduce.reduced_axes(values,listed)))},ShapeLaws.elements(Layout.extents_at(values,Reduce.reduced_axes(values,listed)))},limits,listed,values,Layout.reduce_axes(~Tile.Tile8,~TiledReduce.LaneLine<Accumulator>,~TiledReduce.line_step(~Accumulator,~operation),~TiledReduce.line_finish(~Accumulator,~finish),name,(operations * 8.0 : F32),TiledReduce.LaneLine{initial,0,Grid.lanes_at(values,lane)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,Grid.Innermost{}),fill,Tensors.Usable{},workers},TiledReduce.cell_axes(axes,List.length(&2,U32,values),lane,Grid.Innermost{})),restated,default_capacity,rank_room,nonempty)

# ---- Lanes across a reduced axis ----

# The flags of the axes hold none after the last.
def last_unflagged(remaining: Nat,+index: Nat,+axes: List<&2,Nat>) ->
  {TiledReduce.none_after(Layout.reduced_flags(1n+remaining,index,axes),remaining) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+rest: last_unflagged(rest,1n+index,axes)

def last_after(rank: Nat,+axes: List<&2,Nat>,+positive: {Nat.is_lt(0n,rank) == True{} : Bool}) ->
  {TiledReduce.none_after(Layout.reduced_flags(rank,0n,axes),Nat.sub(rank,1n)) == True{} : Bool}:
  match rank:
    case 0n: Empty.absurd({TiledReduce.none_after(Layout.reduced_flags(0n,0n,axes),Nat.sub(0n,1n)) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {TiledReduce.none_after(Layout.reduced_flags(1n+rest,0n,axes),_) == True{} : Bool}
      last_unflagged(rest,0n,axes)

def last_below(rank: Nat,+positive: {Nat.is_lt(0n,rank) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(rank,1n),rank) == True{} : Bool}:
  match rank:
    case 0n: Empty.absurd({Nat.is_lt(Nat.sub(0n,1n),0n) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.is_lt(_,1n+rest) == True{} : Bool}
      Order.lt_succ(rest,rest,Order.reflexive(rest))

# What a reduction of every model of the cells of the contract holds, it
# holds of the cells.
def modeled_usable(~Logical: Data,~original: Logical -> U32 -> F32,~Holds: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,
  -pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +observation: PiecesModel.Observation<Tile.Tile8,pieces>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
  body: @+model: PiecesModel.Model<Tile.Tile8> -> @+fill: Tile.Tile8 -> @+workers: U32 -> @+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position}) ->
    Holds(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})) ->
  Holds(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}):
  PiecesModel.Observation{+model,+equation} = observation
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  %equation : Holds(Tiled.Tiled{Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers},axis,lanes,position})
  %shaped : Holds(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,fill,Tensors.Usable{},workers},axis,lanes,position})
  %pinned : Holds(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,_,position})
  body(model,fill,workers,
    {%Equal.sym(U32,Grid.lanes_at(values,axis),lanes,pinned) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,_,position})
      %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,fill,Tensors.Usable{},workers},axis,lanes,position})
      %Equal.sym(Pieces.Pieces(~Tile.Tile8),PiecesModel.pack(~Tile.Tile8,model),pieces,equation) :
        Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers},axis,lanes,position}) contract : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})})

def modeled_cells(~Logical: Data,~original: Logical -> U32 -> F32,~Holds: @tiled: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,
  cells: Tensors.Tensor<Tile.Tile8>,+axis: Nat,+lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}),
  body: @+model: PiecesModel.Model<Tile.Tile8> -> @+fill: Tile.Tile8 -> @+workers: U32 -> @+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position}) ->
    Holds(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position})) ->
  Holds(Tiled.Tiled{cells,axis,lanes,position}):
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      modeled_usable(~Logical,~original,~Holds,logical,values,pieces,shape,fill,workers,axis,lanes,position,PiecesModel.observe(~Tile.Tile8,pieces),contract,body)
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Holds(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)

# Tiled values with their lanes on the last axis, the groups innermost,
# reduce along it when it is reduced, otherwise on a kept axis.
def last_route(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,reduced: Bool,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{}),fill,Tensors.Usable{},workers},Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),Grid.Innermost{}}),
  +decided: {Views.contains_index(Layout.naturals(listed),Nat.sub(List.length(&2,U32,values),1n)) == reduced : Bool},+positive: {Nat.is_lt(0n,List.length(&2,U32,values)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded_last(~Accumulator,~operation,~finish,reduced,name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{}),fill,Tensors.Usable{},workers},Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),Grid.Innermost{}})):
  match reduced:
    case True{}:
      along_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),model,fill,workers,contract,
        last_below(List.length(&2,U32,values),positive),decided,last_after(List.length(&2,U32,values),Layout.naturals(listed),positive),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
    case False{}:
      kept_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),Grid.Innermost{},model,fill,workers,contract,
        last_below(List.length(&2,U32,values),positive),decided,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)

def across_arranged(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+position: Grid.Position,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +arranged: Relane.Arranged(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{}),+positive: {Nat.is_lt(0n,List.length(&2,U32,values)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Across{},name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position})):
  KernelGemm.Both{+equation,+moved} = arranged
  %Equal.sym(Tiled.Tiled,Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{}),Tiled.Tiled{Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{})),Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),Grid.Innermost{}},equation) : Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded_last(~Accumulator,~operation,~finish,Views.contains_index(Layout.naturals(listed),Nat.sub(List.length(&2,U32,values),1n)),name,operations,initial,values,listed,_))
  modeled_cells(~Logical,~original,~(tiled => Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded_last(~Accumulator,~operation,~finish,Views.contains_index(Layout.naturals(listed),Nat.sub(List.length(&2,U32,values),1n)),name,operations,initial,values,listed,tiled))),logical,values,Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{})),Nat.sub(List.length(&2,U32,values),1n),Grid.lanes_at(values,Nat.sub(List.length(&2,U32,values),1n)),Grid.Innermost{},moved,
    moved_model => moved_fill => moved_workers => contract => last_route(~Accumulator,~operation,~finish,~Logical,~original,Views.contains_index(Layout.naturals(listed),Nat.sub(List.length(&2,U32,values),1n)),logical,limits,name,operations,initial,listed,values,moved_model,moved_fill,moved_workers,contract,{==},positive,default_capacity,rank_room,nonempty,cap,cap_ok,count_room))

# Tiled values of the contract with their lanes across a reduced axis move
# them to the last axis, the groups innermost, and reduce there.
def across_route(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+position: Grid.Position,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position}),+positive: {Nat.is_lt(0n,List.length(&2,U32,values)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Across{},name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position})):
  across_arranged(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,position,model,fill,workers,
    Relane.arranged_held(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Nat.sub(List.length(&2,U32,values),1n),Grid.Innermost{},limits,default_capacity,Order.sub_le(List.length(&2,U32,values),1n),nonempty,rank_room,contract),
    positive,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)

# ---- The route of a reduction ----

def positive_of(rank: Nat,lane: Nat,+below: {Nat.is_lt(lane,rank) == True{} : Bool}) -> {Nat.is_lt(0n,rank) == True{} : Bool}:
  match rank lane:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(below))
    case 0n 1n+rest: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(below))
    case 1n+rest lane: {==}

# Lanes on the unit axis: one value a cell.
def single_at(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+position: Grid.Position,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position}),+past: {Nat.is_lt(lane,List.length(&2,U32,values)) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool},
  +unique: {Views.unique_axes(Layout.naturals(listed)) == True{} : Bool},+below: {Layout.all_below(listed,List.length(&2,U32,values)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Single{},name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  +unit = Order.antisymmetric(lane,List.length(&2,U32,values),inside,Order.not_less(lane,List.length(&2,U32,values),past))
  +moved = Equal.trans(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position},
    Equal.cong(Nat,Tiled.Tiled,axis => Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,position),fill,Tensors.Usable{},workers},axis,Grid.lanes_at(values,axis),position},lane,List.length(&2,U32,values),unit),
    Equal.cong(List<&2,U32>,Tiled.Tiled,cells => Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position},
      Tiled.cell_shape(values,List.length(&2,U32,values),position),values,Public.unit_shape(values,position)))
  %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position},moved) : Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.Single{},name,operations,initial,values,listed,_))
  single_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,model,fill,workers,position,{%moved : Public.Contract(~Logical,~original,logical,values,_) contract : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),values,fill,Tensors.Usable{},workers},List.length(&2,U32,values),Grid.lanes_at(values,List.length(&2,U32,values)),position})},default_capacity,rank_room,nonempty,cap,cap_ok,count_room,unique,below)

# Tiled values of the contract reduce on the route of their lanes: one value
# a cell on the unit axis, the lanes on a kept axis, along a reduced axis with
# no reduced axis after it and the groups innermost, otherwise across.
def routed(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,within: Bool,member: Bool,position: Grid.Position,after: Bool,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,
  +listed: List<&2,U32>,+values: List<&2,U32>,+lane: Nat,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position}),
  +within_eq: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+member_eq: {Views.contains_index(Layout.naturals(listed),lane) == member : Bool},
  +after_eq: {TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,Layout.naturals(listed)),lane) == after : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool},
  +unique: {Views.unique_axes(Layout.naturals(listed)) == True{} : Bool},+below: {Layout.all_below(listed,List.length(&2,U32,values)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,Bool.pick(TiledReduce.Route,within,Bool.pick(TiledReduce.Route,member,Bool.pick(TiledReduce.Route,Bool.not(Grid.is_outermost(position)) && after,TiledReduce.Along{},TiledReduce.Across{}),TiledReduce.Kept{}),TiledReduce.Single{}),name,operations,initial,values,listed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,lane,position),fill,Tensors.Usable{},workers},lane,Grid.lanes_at(values,lane),position})):
  match within member position after:
    case False{} member position after:
      single_at(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,position,model,fill,workers,contract,within_eq,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,unique,below)
    case True{} False{} position after:
      kept_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,Grid.lanes_at(values,lane),position,model,fill,workers,contract,within_eq,member_eq,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
    case True{} True{} Grid.Innermost{} True{}:
      along_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,Grid.lanes_at(values,lane),model,fill,workers,contract,within_eq,member_eq,after_eq,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
    case True{} True{} Grid.Innermost{} False{}:
      across_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,Grid.Innermost{},model,fill,workers,contract,positive_of(List.length(&2,U32,values),lane,within_eq),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
    case True{} True{} Grid.Outermost{} after:
      across_route(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,lane,Grid.Outermost{},model,fill,workers,contract,positive_of(List.length(&2,U32,values),lane,within_eq),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)

# A valid list of axes of usable, nonempty values.
def valid_true(+values: List<&2,U32>,+listed: List<&2,U32>,+unique: {Views.unique_axes(Layout.naturals(listed)) == True{} : Bool},
  +below: {Layout.all_below(listed,List.length(&2,U32,values)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  {TiledReduce.valid_axes(Tensors.Usable{},values,listed) == True{} : Bool}:
  +zero = CountCompletion.positive_not_zero(values,nonempty)
  %Equal.sym(Bool,Views.unique_axes(Layout.naturals(listed)),True{},unique) :
    {Tensors.usable(Tensors.Usable{}) && _ && Layout.all_below(listed,List.length(&2,U32,values)) && Bool.not(Shapes.has_zero(values)) == True{} : Bool}
  %Equal.sym(Bool,Layout.all_below(listed,List.length(&2,U32,values)),True{},below) :
    {Tensors.usable(Tensors.Usable{}) && True{} && _ && Bool.not(Shapes.has_zero(values)) == True{} : Bool}
  %Equal.sym(Bool,Shapes.has_zero(values),False{},zero) :
    {Tensors.usable(Tensors.Usable{}) && True{} && True{} && Bool.not(_) == True{} : Bool}
  {==}

# Usable cells of the contract, a valid list of axes.
def valid_usable(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,
  pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +unique: {Views.unique_axes(Layout.naturals(listed)) == True{} : Bool},+below: {Layout.all_below(listed,List.length(&2,U32,values)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},listed)):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  +same = Equal.trans(Tiled.Tiled,
    TiledReduce.folded_valid(~Accumulator,~operation,~finish,TiledReduce.valid_axes(Tensors.Usable{},Tiled.logical_shape(shape,lane,lanes,position),listed),name,operations,initial,Tiled.logical_shape(shape,lane,lanes,position),listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),
    TiledReduce.folded_valid(~Accumulator,~operation,~finish,TiledReduce.valid_axes(Tensors.Usable{},values,listed),name,operations,initial,values,listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),
    TiledReduce.folded_valid(~Accumulator,~operation,~finish,True{},name,operations,initial,values,listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),
    Equal.cong(List<&2,U32>,Tiled.Tiled,shaped => TiledReduce.folded_valid(~Accumulator,~operation,~finish,TiledReduce.valid_axes(Tensors.Usable{},shaped,listed),name,operations,initial,shaped,listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),
      Tiled.logical_shape(shape,lane,lanes,position),values,restored),
    Equal.cong(Bool,Tiled.Tiled,valid => TiledReduce.folded_valid(~Accumulator,~operation,~finish,valid,name,operations,initial,values,listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),TiledReduce.valid_axes(Tensors.Usable{},values,listed),True{},valid_true(values,listed,unique,below,nonempty)))
  %Equal.sym(Tiled.Tiled,TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},listed),TiledReduce.folded_valid(~Accumulator,~operation,~finish,True{},name,operations,initial,values,listed,(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},lane)),same) : Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,_)
  modeled_usable(~Logical,~original,~(tiled => Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.folded(~Accumulator,~operation,~finish,TiledReduce.route_of(lane,List.length(&2,U32,values),Layout.naturals(listed),Bool.not(Grid.is_outermost(position)) && TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,Layout.naturals(listed)),lane)),name,operations,initial,values,listed,tiled))),logical,values,pieces,shape,fill,workers,lane,lanes,position,
    PiecesModel.observe(~Tile.Tile8,pieces),contract,
    model => fill => workers => contract =>
      routed(~Accumulator,~operation,~finish,~Logical,~original,Nat.is_lt(lane,List.length(&2,U32,values)),Views.contains_index(Layout.naturals(listed),lane),position,TiledReduce.none_after(Layout.reduced_flags(List.length(&2,U32,values),0n,Layout.naturals(listed)),lane),
        logical,limits,name,operations,initial,listed,values,lane,model,fill,workers,contract,{==},{==},{==},default_capacity,rank_room,nonempty,cap,cap_ok,count_room,unique,below))

# Tiled values of the contract reduce a valid list of axes as Layout.reduce_axes
# reduces the values: every output folds its line in increasing index.
def reduce_axes(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+listed: List<&2,U32>,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),
  +unique: {Views.unique_axes(Layout.naturals(listed)) == True{} : Bool},+below: {Layout.all_below(listed,List.length(&2,U32,values)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,tiled,listed)):
  match tiled:
    case Tiled.Tiled{cells,+lane,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          valid_usable(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,listed,values,pieces,shape,fill,workers,lane,lanes,position,contract,unique,below,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,listed,TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},lane,lanes,position},listed)),contract)

# ---- Every axis ----

# The words of the axes from index on, count of them, are the numbers from
# start on.
def every_words(remaining: Nat,+index: U32,+start: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(index) == start : Nat},
  +room: {Nat.is_le(Nat.add(start,remaining),U32.to_nat(cap)) == True{} : Bool}) -> {Layout.naturals(TiledReduce.every_axis(remaining,index)) == Layout.every_index(remaining,start) : List<&2,Nat>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next_room = {%Addition.successor_right(start,rest) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(1n+start,rest),U32.to_nat(cap)) == True{} : Bool}}
      +one_le = {%Equal.sym(Bool,Nat.is_le(Nat.add(start,1n),Nat.add(start,1n+rest)),Nat.is_le(1n,1n+rest),Order.le_offset(start,1n,1n+rest)) : {_ == True{} : Bool}
        Order.zero_le(rest) : {Nat.is_le(Nat.add(start,1n),Nat.add(start,1n+rest)) == True{} : Bool}}
      +next_exact = Equal.trans(Nat,U32.to_nat((index + 1 : U32)),Nat.add(start,1n),1n+start,
        Arithmetic.sum_exact(cap,cap_ok,index,1,start,1n,exact,{==},Order.transitive(Nat.add(start,1n),Nat.add(start,1n+rest),U32.to_nat(cap),one_le,room)),Order.commutative(start,1n))
      Equal.trans(List<&2,Nat>,U32.to_nat(index) <> Layout.naturals(TiledReduce.every_axis(rest,(index + 1 : U32))),start <> Layout.naturals(TiledReduce.every_axis(rest,(index + 1 : U32))),
        start <> Layout.every_index(rest,1n+start),
        Equal.cong(Nat,List<&2,Nat>,head => head <> Layout.naturals(TiledReduce.every_axis(rest,(index + 1 : U32))),U32.to_nat(index),start,exact),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => start <> tail,Layout.naturals(TiledReduce.every_axis(rest,(index + 1 : U32))),Layout.every_index(rest,1n+start),
          every_words(rest,(index + 1 : U32),1n+start,cap,cap_ok,next_exact,next_room)))

# The numbers from start on, count of them, lie below a rank they fit in.
def every_index_below(remaining: Nat,+start: Nat,+rank: Nat,+fits: {Nat.is_le(Nat.add(start,remaining),rank) == True{} : Bool}) ->
  {below_all(Layout.every_index(remaining,start),rank) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next_fits = {%Addition.successor_right(start,rest) : {Nat.is_le(_,rank) == True{} : Bool} fits : {Nat.is_le(Nat.add(1n+start,rest),rank) == True{} : Bool}}
      +below = Bounds.nat_trans(start,1n+Nat.add(start,rest),rank,Order.lt_succ(start,Nat.add(start,rest),Order.le_add(start,rest)),next_fits)
      %Equal.sym(Bool,Nat.is_lt(start,rank),True{},below) : {_ && below_all(Layout.every_index(rest,1n+start),rank) == True{} : Bool}
      every_index_below(rest,1n+start,rank,next_fits)

# Every axis of a rank is listed once.
def every_unique_words(+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(rank,U32.to_nat(cap)) == True{} : Bool}) ->
  {Views.unique_axes(Layout.naturals(TiledReduce.every_axis(rank,0))) == True{} : Bool}:
  %Equal.sym(List<&2,Nat>,Layout.naturals(TiledReduce.every_axis(rank,0)),Layout.every_index(rank,0n),every_words(rank,0,0n,cap,cap_ok,{==},room)) : {Views.unique_axes(_) == True{} : Bool}
  Reduce.every_unique(rank,0n)

# Every axis of a rank lies below it.
def every_below_words(+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(rank,U32.to_nat(cap)) == True{} : Bool}) ->
  {Layout.all_below(TiledReduce.every_axis(rank,0),rank) == True{} : Bool}:
  %Equal.sym(Bool,Layout.all_below(TiledReduce.every_axis(rank,0),rank),below_all(Layout.naturals(TiledReduce.every_axis(rank,0)),rank),below_naturals(TiledReduce.every_axis(rank,0),rank)) :
    {_ == True{} : Bool}
  %Equal.sym(List<&2,Nat>,Layout.naturals(TiledReduce.every_axis(rank,0)),Layout.every_index(rank,0n),every_words(rank,0,0n,cap,cap_ok,{==},room)) : {below_all(_,rank) == True{} : Bool}
  every_index_below(rank,0n,rank,Order.reflexive(rank))

# Usable cells of the contract reduce every axis.
def all_usable(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+values: List<&2,U32>,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,TiledReduce.every_axis(List.length(&2,U32,values),0),TiledReduce.reduce_all(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  +same = Equal.cong(List<&2,U32>,Tiled.Tiled,extents => TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},TiledReduce.every_axis(List.length(&2,U32,extents),0)),Tiled.logical_shape(shape,lane,lanes,position),values,restored)
  %Equal.sym(Tiled.Tiled,TiledReduce.reduce_all(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),TiledReduce.reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},TiledReduce.every_axis(List.length(&2,U32,values),0)),same) : Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,TiledReduce.every_axis(List.length(&2,U32,values),0),_)
  reduce_axes(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,TiledReduce.every_axis(List.length(&2,U32,values),0),values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},contract,every_unique_words(List.length(&2,U32,values),Machine.capacity(limits),Limits.capacity_fits(limits),rank_room),every_below_words(List.length(&2,U32,values),Machine.capacity(limits),Limits.capacity_fits(limits),rank_room),default_capacity,rank_room,nonempty,cap,cap_ok,count_room)

# Tiled values of the contract reduce every axis as Layout.reduce_axes reduces
# every axis of the values.
def reduce_all(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(values)),4n),U32.to_nat(cap)) == True{} : Bool}) -> Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,TiledReduce.every_axis(List.length(&2,U32,values),0),TiledReduce.reduce_all(~Accumulator,~operation,~finish,name,operations,initial,tiled)):
  match tiled:
    case Tiled.Tiled{cells,+lane,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          all_usable(~Accumulator,~operation,~finish,~Logical,~original,logical,limits,name,operations,initial,values,pieces,shape,fill,workers,lane,lanes,position,contract,default_capacity,rank_room,nonempty,cap,cap_ok,count_room)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Folded(~Accumulator,~operation,~finish,~Logical,~original,logical,initial,values,TiledReduce.every_axis(List.length(&2,U32,values),0),TiledReduce.reduce_all(~Accumulator,~operation,~finish,name,operations,initial,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},lane,lanes,position})),contract)
