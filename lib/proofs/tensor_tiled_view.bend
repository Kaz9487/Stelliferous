# Gathers of tiled tensors: the tiled form of a view of the logical values.
# A view that keeps no axis of lanes reads the cells of the input with its
# lanes on the unit axis, one value a cell, as a view of those cells. A view
# that keeps the lanes reads, through the view of the cells, the cell of every
# group of target lanes, and the padding sets the lanes past the target to +0.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_view.bend as TiledView
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./machine_limit_bounds.bend as Limits
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_cell_room.bend as Rooms
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_placing.bend as Placing
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_view_reads.bend as ViewReads
import ./nat_order.bend as Order
import ./tensor_grid_box.bend as Box
import ./tensor_view_part.bend as ViewPart
import ./tensor_tiled_view_reads.bend as KeptReads
import ./tensor_padded_pieces.bend as PaddedPieces
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_set_slice.bend as SetSlice
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ../tensor_tiled_layout.bend as TiledLayout
import ../tensor_layout.bend as Layout
import ./checked_word_refinement.bend as Checked
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./nat_ceiling.bend as Ceiling
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./natural_addition.bend as Addition
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_box_digits.bend as Digits
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_relabeled.bend as Relabeled
import ../tensor_tiled_cells.bend as Cells

# ---- Lanes on the unit axis ----

# One lane of a cell on the unit axis read where a view reads is that lane
# of the cell of the viewed values.
def unit_lane_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+index: U32,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +read: {Nat.is_lt(U32.to_nat((Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32)),
    ShapeLaws.elements(values)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},axes,offset,reversed},index)) == TileLaws.select_lane(lane,Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(Views.axis_extents(axes),List.length(&2,U32,Views.axis_extents(axes)),Grid.Innermost{})},index)) : F32}:
  +address = (Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32)
  +target = Views.axis_extents(axes)
  Equal.trans(F32,TileLaws.select_lane(lane,ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},axes,offset,reversed},index)),Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,address),0.0),TileLaws.select_lane(lane,Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(Views.axis_extents(axes),List.length(&2,U32,Views.axis_extents(axes)),Grid.Innermost{})},index)),
    Coordinates.unit_lane(~Logical,~original,logical,limits,values,position,address,room,read,lane,small),
    Equal.sym(F32,TileLaws.select_lane(lane,Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(Views.axis_extents(axes),List.length(&2,U32,Views.axis_extents(axes)),Grid.Innermost{})},index)),Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,address),0.0),
      Coordinates.unit_lane(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},limits,target,Grid.Innermost{},index,
        target_room,inside,lane,small)))

# A cell of values with their lanes on the unit axis, read where a view
# reads, is the cell of the viewed values on the unit axis of the target.
def unit_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+index: U32,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +read: {Nat.is_lt(U32.to_nat((Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32)),
    ShapeLaws.elements(values)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},axes,offset,reversed},index) ==
    Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(Views.axis_extents(axes),List.length(&2,U32,Views.axis_extents(axes)),Grid.Innermost{})},index) : Tile.Tile8}:
  TileLaws.tiles_equal(ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},axes,offset,reversed},index),Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(Views.axis_extents(axes),List.length(&2,U32,Views.axis_extents(axes)),Grid.Innermost{})},index),
    unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,0n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,1n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,2n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,3n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,4n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,5n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,6n,{==}),unit_lane_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,read,inside,7n,{==}))

# A view that fits counted values reads below their cells.
def unit_read(+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+source_count: U32,
  +accepted: {Tensors.count_of(values) == Some{source_count} : Maybe<&2,U32>},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32)),ShapeLaws.elements(values)) == True{} : Bool}:
  +length = Tensors.count_or_zero(values)
  +cells = ViewMoved.counted_cells(values,source_count,accepted)
  +length_room = {%Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(values),cells) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
    {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +read = ViewReads.address_at(axes,reversed,offset,U32.to_nat(index),length,Machine.capacity(limits),Limits.capacity_fits(limits),target_room,length_room,fits,inside)
  +sum = Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,reversed,U32.to_nat(index)))
  %Equal.sym(Nat,U32.to_nat((Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32)),sum,ViewGather.read_exact_of(offset,axes,reversed,U32.to_nat(index),(Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32),length,read)) :
    {Nat.is_lt(_,ShapeLaws.elements(values)) == True{} : Bool}
  %cells : {Nat.is_lt(sum,_) == True{} : Bool}
  ViewGather.read_below_of(offset,axes,reversed,U32.to_nat(index),(Views.mirrored_offset(axes,reversed,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,reversed)),U32.to_nat(index)) : U32),length,read)

# The laid out cells of the unit axis of extents.
def unit_laid(+target: List<&2,U32>,+count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>}) ->
  Public.Laid<Tiled.cell_shape(target,List.length(&2,U32,target),Grid.Innermost{}),List.length(&2,U32,target),1,Grid.Innermost{},target>:
  +rank = List.length(&2,U32,target)
  +past = Order.not_below(rank,rank,Order.reflexive(rank))
  +pinned = Public.lanes_past(target,rank,past)
  Public.Laid{Order.reflexive(rank),{==},
    %pinned : {Tiled.logical_shape(Tiled.cell_shape(target,rank,Grid.Innermost{}),rank,_,Grid.Innermost{}) == target : List<&2,U32>} Public.cells_restored(target,rank,Grid.Innermost{}),
    Global.Counted{count,counted},pinned}

# The cells of a contract with their lanes on the unit axis, of the pieces of
# a model, through a view the runtime accepts: the tiled form of the view.
def unit_model(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+values: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers},axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}}):
  +rank = List.length(&2,U32,values)
  +target = Views.axis_extents(axes)
  +cells_same = Placing.cells_past(values,rank,position,Order.not_below(rank,rank,Order.reflexive(rank)))
  +source_counted = {%cells_same : Global.Counted<_>
    Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,workers,global) :
    Global.Counted<values>}
  +source_count = Global.count(values,source_counted)
  +room = StorageCount.counted_room(limits,values,source_counted,default_capacity)
  +target_room = StorageCount.counted_room(limits,target,Global.Counted{count,counted},default_capacity)
  +fits_cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,List.length(&2,U32,values),position),values,cells_same) : {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(_)) == True{} : Bool} fits :
    {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(Tiled.cell_shape(values,List.length(&2,U32,values),position))) == True{} : Bool}}
  +source_rank = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,List.length(&2,U32,values),position),values,cells_same) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,List.length(&2,U32,values),position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +fits_nonempty = {%ViewMoved.none_zero(target,ViewMoved.counted_positive(target,count,counted,nonzero)) : {Views.fits_nonempty(_,Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool} fits :
    {Views.fits_nonempty(False{},Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool}}
  +moved = ViewMoved.view_moved(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},limits,name,model,Tiled.cell_shape(values,List.length(&2,U32,values),position),axes,offset,reversed,fill,workers,global,
    default_capacity,source_rank,target_rank,count,counted,nonzero,fits_cells)
  +restated = Global.restated_below(~Tile.Tile8,~ViewGather.Viewing<Representation.Lifted<Logical>>,~Representation.Lifted<ViewGather.Viewing<Logical>>,
    ~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original)),~ShapeLaws.elements(target),
    ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},axes,offset,reversed},Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(target,List.length(&2,U32,target),Grid.Innermost{})},
    index => inside => unit_agree(~Logical,~original,logical,limits,values,position,axes,offset,reversed,index,room,target_room,
      unit_read(limits,values,axes,offset,reversed,source_count,Global.accepted(values,source_counted),room,target_room,fits_nonempty,index,inside),inside),
    target,Tensors.moved(~Tile.Tile8,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers},axes,offset,reversed),moved,Order.reflexive(ShapeLaws.elements(target)),Machine.capacity(limits),target_room)
  +target_rank_n = List.length(&2,U32,target)
  +target_same = Placing.cells_past(target,target_rank_n,Grid.Innermost{},Order.not_below(target_rank_n,target_rank_n,Order.reflexive(target_rank_n)))
  Public.laid_contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},target,Tensors.moved(~Tile.Tile8,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers},axes,offset,reversed),target_rank_n,1,Grid.Innermost{},
    KernelGemm.Both{%Equal.sym(List<&2,U32>,Tiled.cell_shape(target,target_rank_n,Grid.Innermost{}),target,target_same) :
        Global.Shaped(~Tile.Tile8,~Representation.Lifted<ViewGather.Viewing<Logical>>,~Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original)),
          Representation.Lifted{ViewGather.Viewing{logical,axes,offset,reversed},Public.geometry_of(target,target_rank_n,Grid.Innermost{})},_,Tensors.moved(~Tile.Tile8,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),fill,Tensors.Usable{},workers},axes,offset,reversed))
      restated,unit_laid(target,count,counted)})

# The same for usable cells.
def unit_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+values: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,-cells: Tensors.Tensor<Tile.Tile8>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},Tiled.cell_shape(values,List.length(&2,U32,values),position),cells)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,cells,axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}}):
  KernelGemm.Both{Global.Usable{+model,+model_fill,+model_workers,+equation},+global} = shaped
  %equation : Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,_,axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}})
  unit_model(~Logical,~original,logical,limits,name,values,position,axes,offset,reversed,model,model_fill,model_workers,default_capacity,rank_room,target_rank,count,counted,nonzero,fits,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,List.length(&2,U32,values),position),model_fill,Tensors.Usable{},model_workers},cells,equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},_)
    global)

# Tiled values of the contract with their lanes on the unit axis, through a
# view the runtime accepts: the tiled form of the view.
def unit_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+values: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,cells: Tensors.Tensor<Tile.Tile8>,+lanes: U32,+position: Grid.Position,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,List.length(&2,U32,values),lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,cells,axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}}):
  unit_usable(~Logical,~original,logical,limits,name,values,position,axes,offset,reversed,cells,default_capacity,rank_room,target_rank,count,counted,nonzero,fits,
    Public.cells_shaped(~Logical,~original,logical,values,List.length(&2,U32,values),lanes,position,cells,contract))

# The shape the contract restores.
def contract_restored(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,
  +fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>}:
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  restored

# Tiled values of the contract, through a view the runtime accepts that
# keeps no lanes: the tiled form of the view.
def unit_gathered(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+values: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,tiled: Tiled.Tiled,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),TiledView.gather_values(TiledView.Gathering{name,axes,offset,reversed,values,Views.axis_extents(axes)},tiled)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          +rank = List.length(&2,U32,values)
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,axis,lanes,position),values,
              contract_restored(~Logical,~original,logical,values,pieces,shape,fill,workers,axis,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,Tiled.cells_of(Tiled.along(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position},List.length(&2,U32,_))),axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}})
          Relane.along_then(~Logical,~original,(tiled => Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,Tiled.cells_of(tiled),axes,offset,reversed),List.length(&2,U32,Views.axis_extents(axes)),1,Grid.Innermost{}})),logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position},rank,limits,default_capacity,Order.reflexive(rank),nonempty,rank_room,contract,
            cells => lanes => position => contract => unit_cells(~Logical,~original,logical,limits,name,values,axes,offset,reversed,cells,lanes,position,default_capacity,rank_room,target_rank,count,counted,nonzero,fits,contract))
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),TiledView.gather_values(TiledView.Gathering{name,axes,offset,reversed,values,Views.axis_extents(axes)},Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position})),contract)

# ---- Lanes the view keeps ----

# The laid out cells of the target of a kept view.
def target_laid(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>}) ->
  Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>:
  KeptReads.KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
  Public.Laid{Bounds.strict_le(lane,List.length(&2,U32,target),lane_inside),{==},
    {%Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact) : {Tiled.logical_shape(cells,lane,_,Grid.Innermost{}) == target : List<&2,U32>}
      Public.cells_restored(target,lane,Grid.Innermost{}) : {Tiled.logical_shape(cells,lane,extent,Grid.Innermost{}) == target : List<&2,U32>}},
    Global.Counted{count,counted},Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact)}

# Cells of the contract with their lanes on the axis a view keeps, moved
# through the view of the cells and padded: the tiled form of the view.
def kept_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +laid: Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>,
  +arranged: PaddedCells.Placed<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Views.axis_extents(axes),lane,extent,Grid.Innermost{},
    Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})))>,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    Tiled.Tiled{Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,
      TiledMap.padding_of(Views.axis_extents(KeptReads.cell_axes_of(kept,extent,shape,axis,position)),lane,extent,Grid.Innermost{}),name,
      Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers},KeptReads.cell_axes_of(kept,extent,shape,axis,position),
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),List.append(&2,Bool,kept_flags,[False{}])),lane,extent,Grid.Innermost{}}):
  KeptReads.KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  KeptReads.SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = arranged
  +target = Views.axis_extents(axes)
  +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
  +cell_axes = KeptReads.cell_axes_of(kept,extent,shape,axis,position)
  +cell_flags = List.append(&2,Bool,kept_flags,[False{}])
  +offset_cell = TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)
  +source_cells = Tiled.cell_shape(shape,axis,position)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +extents = Views.axis_extents(cell_axes)
  +padding = TiledMap.padding_of(extents,lane,extent,Grid.Innermost{})
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  +geometry = Public.geometry_of(target,lane,Grid.Innermost{})
  +inner = {ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},cell_axes,offset_cell,cell_flags} : ViewGather.Viewing<Representation.Lifted<Logical>>}
  +extents_same = KeptReads.cell_extents(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +padding_same = Equal.trans(TiledMap.Padding,padding,TiledMap.padding_of(cells,lane,extent,Grid.Innermost{}),PaddedCells.layout_padding(layout),
    Equal.cong(List<&2,U32>,TiledMap.Padding,found => TiledMap.padding_of(found,lane,extent,Grid.Innermost{}),extents,cells,extents_same),
    Equal.sym(TiledMap.Padding,PaddedCells.layout_padding(layout),TiledMap.padding_of(cells,lane,extent,Grid.Innermost{}),padding_exact))
  +padding_back = Equal.sym(TiledMap.Padding,padding,PaddedCells.layout_padding(layout),padding_same)
  +nonempty = KeptReads.cell_nonempty(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +positive = CountCompletion.positive_elements(extents,nonempty)
  +total_positive = {%Equal.sym(Nat,PaddedCells.layout_total(layout),ShapeLaws.elements(cells),total_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    %extents_same : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool}
    positive : {Nat.is_lt(0n,PaddedCells.layout_total(layout)) == True{} : Bool}}
  +cap = Machine.half_word(PaddedPieces.layout_limits(layout))
  +fit = {%padding_back : {PaddedPieces.groups_fit(cap,_) == True{} : Bool}
    PaddedPieces.layout_fit(layout,total_positive) : {PaddedPieces.groups_fit(cap,padding) == True{} : Bool}}
  +total = {%padding_back : {PaddedPieces.padded_cells(_) == ShapeLaws.elements(extents) : Nat}
    Equal.trans(Nat,PaddedPieces.padded_cells(PaddedCells.layout_padding(layout)),PaddedCells.layout_total(layout),ShapeLaws.elements(extents),PaddedPieces.layout_cells(layout),
      Equal.trans(Nat,PaddedCells.layout_total(layout),ShapeLaws.elements(cells),ShapeLaws.elements(extents),total_exact,
        Equal.cong(List<&2,U32>,Nat,found => ShapeLaws.elements(found),cells,extents,Equal.sym(List<&2,U32>,extents,cells,extents_same)))) :
    {PaddedPieces.padded_cells(padding) == ShapeLaws.elements(extents) : Nat}}
  +source_rank = {%Equal.sym(Nat,List.length(&2,U32,source_cells),List.length(&2,U32,shape),Rooms.cell_rank(shape,axis,position)) : {Nat.is_le(_,capacity) == True{} : Bool}
    rank_room : {Nat.is_le(List.length(&2,U32,source_cells),capacity) == True{} : Bool}}
  +cells_rank = {%Equal.sym(List<&2,U32>,extents,cells,extents_same) : {Nat.is_le(List.length(&2,U32,_),capacity) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,U32,cells),List.length(&2,U32,target),Rooms.cell_rank(target,lane,Grid.Innermost{})) : {Nat.is_le(_,capacity) == True{} : Bool}
    target_rank : {Nat.is_le(List.length(&2,U32,extents),capacity) == True{} : Bool}}
  +cells_fit = {%Equal.sym(List<&2,U32>,extents,cells,extents_same) : {Nat.is_le(ShapeLaws.elements(_),capacity) == True{} : Bool} target_cells_room : {Nat.is_le(ShapeLaws.elements(extents),capacity) == True{} : Bool}}
  +cells_counted = StorageCount.counted_within(limits,extents,cells_fit,default_capacity)
  +cells_total = Global.count(extents,cells_counted)
  +cells_nonzero = Arithmetic.word_positive_nonzero(cells_total,
    {%Count.count_exact(ShapeLaws.elements(extents),cells_total,Global.evidence(extents,cells_counted)) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
      {Nat.is_lt(0n,U32.to_nat(cells_total)) == True{} : Bool}})
  +shaped = PaddedPieces.padded_view(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},limits,name,model,
    source_cells,cell_axes,offset_cell,cell_flags,fill,workers,global,default_capacity,source_rank,cells_rank,cells_total,Global.accepted(extents,cells_counted),cells_nonzero,
    KeptReads.cell_fits(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view),padding,cap,Limits.half_word_fits(PaddedPieces.layout_limits(layout)),fit,total)
  +moved = {%padding_same : Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<ViewGather.Viewing<Representation.Lifted<Logical>>>,
        ~PaddedPieces.padded_value(~ViewGather.Viewing<Representation.Lifted<Logical>>,~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),
        PaddedPieces.Padded{_,inner},cells,Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},cell_axes,offset_cell,cell_flags))
    %extents_same : Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<ViewGather.Viewing<Representation.Lifted<Logical>>>,
        ~PaddedPieces.padded_value(~ViewGather.Viewing<Representation.Lifted<Logical>>,~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),
        PaddedPieces.Padded{padding,inner},_,Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},cell_axes,offset_cell,cell_flags))
    shaped : Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<ViewGather.Viewing<Representation.Lifted<Logical>>>,
      ~PaddedPieces.padded_value(~ViewGather.Viewing<Representation.Lifted<Logical>>,~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original))),
      PaddedPieces.Padded{PaddedCells.layout_padding(layout),inner},cells,Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},cell_axes,offset_cell,cell_flags))}
  +restated = PaddedPieces.padded_restated(~ViewGather.Viewing<Representation.Lifted<Logical>>,~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),
    ~Representation.Lifted<ViewGather.Viewing<Logical>>,~Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original)),
    inner,Representation.Lifted{viewing,geometry},layout,cells,geometry,geometry_exact,total_exact,
    word => inside => KeptReads.word_valid(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,word,inside),
    Machine.capacity(limits),target_cells_room,Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},cell_axes,offset_cell,cell_flags),moved)
  Public.laid_contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),viewing,target,Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},cell_axes,offset_cell,cell_flags),lane,extent,Grid.Innermost{},KernelGemm.Both{restated,laid})

# ---- Views that read every cell in order ----

# Flags none of which is set reverse no axis.
def none_forward(flags: List<&2,Bool>,+none: {TiledView.none_flagged(flags) == True{} : Bool}) -> {ViewGather.forward(flags) == True{} : Bool}:
  match flags:
    case Nil{}: {==}
    case Con{+flag,+rest}:
      Boolean.and_true(Bool.not(flag),ViewGather.forward(rest),Boolean.and_true_left(Bool.not(flag),TiledView.none_flagged(rest),none),
        none_forward(rest,Boolean.and_true_right(Bool.not(flag),TiledView.none_flagged(rest),none)))

def none_some(+count: U32,same: {None{} == Some{count} : Maybe<&2,U32>}) -> Empty:
  WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,found => Maybe.is_some(&2,U32,found),None{},Some{count},same))

def one_times(+value: Nat) -> {Nat.mul(1n,value) == value : Nat}:
  Equal.trans(Nat,Nat.mul(1n,value),Nat.mul(value,1n),value,RowMajor.multiply_commutative(1n,value),Order.times_one(value))

def one_quotient(+value: Nat) -> {Nat.div(value,1n) == value : Nat}:
  Equal.trans(Nat,Nat.div(value,1n),Nat.div(Nat.mul(value,1n),1n),value,
    Equal.cong(Nat,Nat,number => Nat.div(number,1n),value,Nat.mul(value,1n),Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value))),
    Division.exact_quotient(value,1n,{==}))

# The inner count of an axis of contiguous axes, its stride test and the
# count of the axis.
type AxisCount<-extent: U32,-stride: U32,-count: U32,-inner: Maybe<&2,U32>> is Data:
  AxisCount{size: U32,found: {inner == Some{size} : Maybe<&2,U32>},valid: {U32.is_le(extent,1) || U32.is_eq(stride,size) == True{} : Bool},
    product: {Nat.mul(U32.to_nat(extent),U32.to_nat(size)) == U32.to_nat(count) : Nat}}

def product_count(valid: Bool,+extent: U32,+stride: U32,+size: U32,+count: U32,+classified: {U32.is_le(extent,1) || U32.is_eq(stride,size) == valid : Bool},
  +sized: {Views.contiguous_product(valid,extent,size) == Some{count} : Maybe<&2,U32>}) -> AxisCount<extent,stride,count,Some{size}>:
  match valid:
    case True{}: AxisCount{size,{==},classified,Checked.multiply_sound(extent,size,count,sized)}
    case False{}: Empty.absurd(AxisCount<extent,stride,count,Some{size}>,none_some(count,sized))

def axis_count(inner: Maybe<&2,U32>,+extent: U32,+stride: U32,+count: U32,+sized: {Views.contiguous_axis(extent,stride,inner) == Some{count} : Maybe<&2,U32>}) ->
  AxisCount<extent,stride,count,inner>:
  match inner:
    case Some{+size}: product_count(U32.is_le(extent,1) || U32.is_eq(stride,size),extent,stride,size,count,{==},sized)
    case None{}: Empty.absurd(AxisCount<extent,stride,count,None{}>,none_some(count,sized))

def count_size(+extent: U32,+stride: U32,+count: U32,+inner: Maybe<&2,U32>,+counted: AxisCount<extent,stride,count,inner>) -> U32:
  AxisCount{+size,+found,+valid,+product} = counted
  size

def count_found(+extent: U32,+stride: U32,+count: U32,+inner: Maybe<&2,U32>,+counted: AxisCount<extent,stride,count,inner>) ->
  {inner == Some{count_size(extent,stride,count,inner,counted)} : Maybe<&2,U32>}:
  AxisCount{+size,+found,+valid,+product} = counted
  found

def count_valid(+extent: U32,+stride: U32,+count: U32,+inner: Maybe<&2,U32>,+counted: AxisCount<extent,stride,count,inner>) ->
  {U32.is_le(extent,1) || U32.is_eq(stride,count_size(extent,stride,count,inner,counted)) == True{} : Bool}:
  AxisCount{+size,+found,+valid,+product} = counted
  valid

def count_product(+extent: U32,+stride: U32,+count: U32,+inner: Maybe<&2,U32>,+counted: AxisCount<extent,stride,count,inner>) ->
  {Nat.mul(U32.to_nat(extent),U32.to_nat(count_size(extent,stride,count,inner,counted))) == U32.to_nat(count) : Nat}:
  AxisCount{+size,+found,+valid,+product} = counted
  product

# Contiguous axes count the cells of their extents.
def contiguous_count(axes: List<&2,Views.Axis>,+count: U32,+sized: {Views.contiguous_size(axes) == Some{count} : Maybe<&2,U32>}) ->
  {U32.to_nat(count) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}:
  match axes:
    case Nil{}: Equal.sym(Nat,1n,U32.to_nat(count),Checked.accepted_value(1,count,sized))
    case Con{Views.Axis{+extent,+stride},+tail}:
      +inner = Views.contiguous_size(tail)
      +counted = axis_count(inner,extent,stride,count,sized)
      +size = count_size(extent,stride,count,inner,counted)
      Equal.trans(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(extent),U32.to_nat(size)),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Views.axis_extents(tail))),
        Equal.sym(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(size)),U32.to_nat(count),count_product(extent,stride,count,inner,counted)),
        Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),U32.to_nat(size),ShapeLaws.elements(Views.axis_extents(tail)),
          contiguous_count(tail,size,count_found(extent,stride,count,inner,counted))))

# One step of the read of contiguous axes: an axis of one index adds nothing,
# another steps the cells of the axes after it.
def contiguous_step(small: Bool,+extent: U32,+stride: U32,+size: U32,+cells: Nat,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rest: U32,+classified: {U32.is_le(extent,1) == small : Bool},+valid: {U32.is_le(extent,1) || U32.is_eq(stride,size) == True{} : Bool},
  +size_exact: {U32.to_nat(size) == cells : Nat},+positive: {Nat.is_lt(0n,cells) == True{} : Bool},
  +below: {Nat.is_lt(index,Nat.mul(U32.to_nat(extent),cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(U32.to_nat(extent),cells),U32.to_nat(cap)) == True{} : Bool},
  +rest_exact: {U32.to_nat(rest) == Nat.mod(index,cells) : Nat}) ->
  {U32.to_nat((U32.from_nat(Nat.div(index,cells)) * stride + rest : U32)) == index : Nat}:
  match small:
    case True{}:
      +quotient = Nat.div(index,cells)
      +remainder = Nat.mod(index,cells)
      +split = DivisionOrder.quotient_exact(index,cells,positive)
      +single = {%Comparison.le(extent,1) : {_ == True{} : Bool} classified : {Nat.is_le(U32.to_nat(extent),1n) == True{} : Bool}}
      +quotient_one = Order.lt_le(quotient,U32.to_nat(extent),1n,DivisionOrder.quotient_below(index,U32.to_nat(extent),cells,positive,below),single)
      +quotient_zero = Order.zero_below(quotient,Division.lt_succ_le(quotient,0n,quotient_one))
      +remainder_index = {%quotient_zero : {index == Nat.add(Nat.mul(_,cells),remainder) : Nat} split : {index == remainder : Nat}}
      %Equal.sym(Nat,quotient,0n,quotient_zero) : {U32.to_nat((U32.from_nat(_) * stride + rest : U32)) == index : Nat}
      %Equal.sym(U32,U32.from_nat(0n),0,Arithmetic.native_roundtrip(0)) : {U32.to_nat((_ * stride + rest : U32)) == index : Nat}
      %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {U32.to_nat((_ + rest : U32)) == index : Nat}
      %Equal.sym(U32,(0 + rest : U32),rest,Arithmetic.add_zero_left(rest)) : {U32.to_nat(_) == index : Nat}
      Equal.trans(Nat,U32.to_nat(rest),remainder,index,rest_exact,Equal.sym(Nat,index,remainder,remainder_index))
    case False{}:
      +quotient = Nat.div(index,cells)
      +remainder = Nat.mod(index,cells)
      +split = DivisionOrder.quotient_exact(index,cells,positive)
      +same = {%classified : {_ || U32.is_eq(stride,size) == True{} : Bool} valid : {U32.is_eq(stride,size) == True{} : Bool}}
      +stride_value = Equal.trans(Nat,U32.to_nat(stride),U32.to_nat(size),cells,Equal.cong(U32,Nat,U32.to_nat,stride,size,Arithmetic.equal_of_is_eq(stride,size,same)),size_exact)
      +product_le = {%Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,cells),remainder),split) : {Nat.is_le(Nat.mul(quotient,cells),_) == True{} : Bool}
        Order.le_add(Nat.mul(quotient,cells),remainder) : {Nat.is_le(Nat.mul(quotient,cells),index) == True{} : Bool}}
      +index_room = Order.transitive(index,Nat.mul(U32.to_nat(extent),cells),U32.to_nat(cap),Bounds.strict_le(index,Nat.mul(U32.to_nat(extent),cells),below),room)
      +product_room = Order.transitive(Nat.mul(quotient,cells),index,U32.to_nat(cap),product_le,index_room)
      +quotient_le = {%RowMajor.multiply_commutative(cells,quotient) : {Nat.is_le(quotient,_) == True{} : Bool}
        DivisionOrder.le_multiple(cells,quotient,positive) : {Nat.is_le(quotient,Nat.mul(quotient,cells)) == True{} : Bool}}
      +quotient_room = Order.transitive(quotient,Nat.mul(quotient,cells),U32.to_nat(cap),quotient_le,product_room)
      +scaled = Arithmetic.product_exact(cap,cap_ok,U32.from_nat(quotient),stride,quotient,cells,FullRange.roundtrip(quotient,cap,quotient_room),stride_value,product_room)
      +summed = Arithmetic.sum_exact(cap,cap_ok,(U32.from_nat(quotient) * stride : U32),rest,Nat.mul(quotient,cells),remainder,scaled,rest_exact,
        {%split : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} index_room : {Nat.is_le(Nat.add(Nat.mul(quotient,cells),remainder),U32.to_nat(cap)) == True{} : Bool}})
      Equal.trans(Nat,U32.to_nat((U32.from_nat(quotient) * stride + rest : U32)),Nat.add(Nat.mul(quotient,cells),remainder),index,summed,
        Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,cells),remainder),split))

# Contiguous axes read the index itself.
def contiguous_read(axes: List<&2,Views.Axis>,+count: U32,+sized: {Views.contiguous_size(axes) == Some{count} : Maybe<&2,U32>},+index: Nat,
  +below: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index)) == index : Nat}:
  match axes:
    case Nil{}: Equal.sym(Nat,index,0n,Order.zero_below(index,Division.lt_succ_le(index,0n,below)))
    case Con{Views.Axis{+extent,+stride},+tail}:
      +inner = Views.contiguous_size(tail)
      +counted = axis_count(inner,extent,stride,count,sized)
      +size = count_size(extent,stride,count,inner,counted)
      +found = count_found(extent,stride,count,inner,counted)
      +cells = ShapeLaws.elements(Views.axis_extents(tail))
      +total = Nat.mul(U32.to_nat(extent),cells)
      +total_positive = Representation.below_positive(index,total,below)
      +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),cells,total_positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),cells,total_positive)
      +tail_room = Order.transitive(cells,total,U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(extent),cells,extent_positive),room)
      +remainder = Nat.mod(index,cells)
      +rest = contiguous_read(tail,size,found,remainder,DivisionOrder.remainder_small(index,cells,positive),cap,cap_ok,tail_room)
      contiguous_step(U32.is_le(extent,1),extent,stride,size,cells,index,cap,cap_ok,Box.stepped_sum(Views.axis_extents(tail),ViewPart.view_steps(tail),remainder),
        {==},count_valid(extent,stride,count,inner,counted),contiguous_count(tail,size,found),positive,below,room,rest)

def contiguous_word(found: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,+exact: {Views.contiguous_size(axes) == found : Maybe<&2,U32>},
  +held: {Views.has_count(found) == True{} : Bool},+index: Nat,+below: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index)) == index : Nat}:
  match found:
    case Some{+count}: contiguous_read(axes,count,exact,index,below,cap,cap_ok,room)
    case None{}:
      Empty.absurd({U32.to_nat(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index)) == index : Nat},WordEncoding.false_true(held))

def product_valid(valid: Bool,+extent: U32,+inner: U32,+held: {Views.has_count(Views.contiguous_product(valid,extent,inner)) == True{} : Bool}) -> {valid == True{} : Bool}:
  match valid:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(held))

def axis_held(inner: Maybe<&2,U32>,+extent: U32,+stride: U32,+held: {Views.has_count(Views.contiguous_axis(extent,stride,inner)) == True{} : Bool}) ->
  {Views.has_count(inner) == True{} : Bool}:
  match inner:
    case Some{+count}: {==}
    case None{}: Empty.absurd({Views.has_count(None{}) == True{} : Bool},WordEncoding.false_true(held))

# The last axis of contiguous axes has one index or a unit stride.
def last_contiguous(axes: List<&2,Views.Axis>,+extent: U32,+stride: U32,
  +held: {Views.has_count(Views.contiguous_size(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]))) == True{} : Bool}) ->
  {U32.is_le(extent,1) || U32.is_eq(stride,1) == True{} : Bool}:
  match axes:
    case Nil{}: product_valid(U32.is_le(extent,1) || U32.is_eq(stride,1),extent,1,held)
    case Con{Views.Axis{+outer,+step},+rest}:
      last_contiguous(rest,extent,stride,axis_held(Views.contiguous_size(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])),outer,step,held))

# ---- The group of a cell read in order ----

# What makes the groups of an outermost source innermost: one group or one
# outer cell.
def Single(position: Grid.Position,+groups: Nat,+outer: Nat) -> Data:
  match position:
    case Grid.Innermost{}: Unit
    case Grid.Outermost{}: {Nat.is_le(groups,1n) || Nat.is_eq(outer,1n) == True{} : Bool}

def single_split(alone: Bool,+outer: Nat,+groups: Nat,+index: Nat,+classified: {Nat.is_le(groups,1n) == alone : Bool},
  +single: {alone || Nat.is_eq(outer,1n) == True{} : Bool},+below: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {Nat.div(index,outer) == Nat.mod(index,groups) : Nat}:
  match alone:
    case True{}:
      +one = Order.antisymmetric(groups,1n,classified,Order.positive_one(groups,groups_positive))
      +below_one = {%one : {Nat.is_lt(index,Nat.mul(_,outer)) == True{} : Bool} below : {Nat.is_lt(index,Nat.mul(1n,outer)) == True{} : Bool}}
      +below_outer = {%one_times(outer) : {Nat.is_lt(index,_) == True{} : Bool} below_one : {Nat.is_lt(index,outer) == True{} : Bool}}
      +remainder = Nat.mod(index,1n)
      +remainder_zero = Order.zero_below(remainder,Division.lt_succ_le(remainder,0n,DivisionOrder.remainder_small(index,1n,{==})))
      %Equal.sym(Nat,groups,1n,one) : {Nat.div(index,outer) == Nat.mod(index,_) : Nat}
      Equal.trans(Nat,Nat.div(index,outer),0n,remainder,DivisionOrder.small_quotient(index,outer,below_outer),Equal.sym(Nat,remainder,0n,remainder_zero))
    case False{}:
      +one = Order.equal_of_is_eq(outer,1n,single)
      +below_one = {%one : {Nat.is_lt(index,Nat.mul(groups,_)) == True{} : Bool} below : {Nat.is_lt(index,Nat.mul(groups,1n)) == True{} : Bool}}
      +below_groups = {%Order.times_one(groups) : {Nat.is_lt(index,_) == True{} : Bool} below_one : {Nat.is_lt(index,groups) == True{} : Bool}}
      %Equal.sym(Nat,outer,1n,one) : {Nat.div(index,_) == Nat.mod(index,groups) : Nat}
      Equal.trans(Nat,Nat.div(index,1n),index,Nat.mod(index,groups),one_quotient(index),
        Equal.sym(Nat,Nat.mod(index,groups),index,DivisionOrder.small_remainder(index,groups,below_groups)))

def single_group(alone: Bool,+outer: Nat,+groups: Nat,+index: Nat,+classified: {Nat.is_le(groups,1n) == alone : Bool},
  +single: {alone || Nat.is_eq(outer,1n) == True{} : Bool},+below: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Nat.div(index,Representation.least_one(outer)) == Nat.mod(index,Representation.least_one(groups)) : Nat}:
  +total_positive = Representation.below_positive(index,Nat.mul(groups,outer),below)
  +outer_positive = DivisionOrder.right_factor_positive(groups,outer,total_positive)
  +groups_positive = RowMajor.left_factor_positive(groups,outer,total_positive)
  %Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,outer_positive)) :
    {Nat.div(index,_) == Nat.mod(index,Representation.least_one(groups)) : Nat}
  %Equal.sym(Nat,Representation.least_one(groups),groups,Representation.least_positive(groups,groups_positive)) :
    {Nat.div(index,outer) == Nat.mod(index,_) : Nat}
  single_split(alone,outer,groups,index,classified,single,below,groups_positive)

# The group of a cell of a source whose groups count as innermost.
def outer_group(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+single: Single(position,groups,outer),
  +below: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Representation.group_at(position,outer,groups,index) == Nat.mod(index,Representation.least_one(groups)) : Nat}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: single_group(Nat.is_le(groups,1n),outer,groups,index,{==},single,below)

# The cells of a shape are its groups times the cells of its other axes.
def cell_elements(position: Grid.Position,+shape: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  +removed = Grid.removed(shape,axis)
  +groups = Grid.groups(Grid.lanes_at(shape,axis))
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},within) :
    {ShapeLaws.elements(Grid.placed(_,position,removed,groups)) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(removed)) : Nat}
  Box.placed_elements(position,removed,groups)

def zeros_below(shape: List<&2,U32>,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) -> Coordinates.DigitsBelow(shape,KeptReads.zeros(List.length(&2,U32,shape))):
  match shape:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      KernelGemm.Both{Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)),
        zeros_below(rest,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))}

# Digits of an amount below the lanes at the lane axis lie below a shape
# without a zero.
def unit_below(shape: List<&2,U32>,axis: Nat,+amount: Nat,+inside: {Nat.is_lt(amount,U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool},
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool}) -> Coordinates.DigitsBelow(shape,KeptReads.unit_digits(List.length(&2,U32,shape),axis,amount)):
  match shape axis:
    case Nil{} +axis: Unit{}
    case Con{+extent,+rest} 0n:
      KernelGemm.Both{inside,zeros_below(rest,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))}
    case Con{+extent,+rest} 1n+ +next:
      KernelGemm.Both{Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)),
        unit_below(rest,next,amount,inside,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))}

def empty_value(extents: List<&2,U32>) -> {ShapeLaws.digits_value(extents,[]) == 0n : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}: {==}

# The other digits of a unit at an axis number zero.
def removed_unit_value(rank: Nat,axis: Nat,extents: List<&2,U32>,+amount: Nat) ->
  {ShapeLaws.digits_value(extents,Geometry.removed_at(KeptReads.unit_digits(rank,axis,amount),axis)) == 0n : Nat}:
  match rank axis extents:
    case 0n +axis +extents: empty_value(extents)
    case 1n+ +rest 0n +extents: KeptReads.zeros_value(extents,rest)
    case 1n+ +rest 1n+ +next Nil{}: {==}
    case 1n+ +rest 1n+ +next Con{+extent,+tail}: removed_unit_value(rest,next,tail,amount)

# The cell of eight lanes into an outermost lane axis is the count of the
# other axes.
def unit_cell(+shape: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {KeptReads.cell_value(shape,axis,Grid.Outermost{},KeptReads.unit_digits(List.length(&2,U32,shape),axis,8n)) == ShapeLaws.elements(Grid.removed(shape,axis)) : Nat}:
  +rank = List.length(&2,U32,shape)
  +unit = KeptReads.unit_digits(rank,axis,8n)
  +others = Geometry.removed_at(unit,axis)
  +removed = Grid.removed(shape,axis)
  +groups = Grid.groups(Grid.lanes_at(shape,axis))
  +cells = ShapeLaws.elements(removed)
  %Equal.sym(Nat,Digits.digit_at(unit,axis),Bool.pick(Nat,Nat.is_eq(axis,axis),8n,0n),KeptReads.digit_unit(rank,axis,axis,8n,within)) :
    {ShapeLaws.digits_value(Tiled.cell_shape(shape,axis,Grid.Outermost{}),Coordinates.placed_digits(True{},Grid.Outermost{},others,Nat.div(_,8n))) == cells : Nat}
  %Equal.sym(Bool,Nat.is_eq(axis,axis),True{},Order.equal_reflexive(axis)) :
    {ShapeLaws.digits_value(Tiled.cell_shape(shape,axis,Grid.Outermost{}),Coordinates.placed_digits(True{},Grid.Outermost{},others,Nat.div(Bool.pick(Nat,_,8n,0n),8n))) == cells : Nat}
  %Equal.sym(Bool,Nat.is_lt(axis,rank),True{},within) : {ShapeLaws.digits_value(Grid.placed(_,Grid.Outermost{},removed,groups),1n <> others) == cells : Nat}
  %Equal.sym(Nat,ShapeLaws.digits_value(removed,others),0n,removed_unit_value(rank,axis,removed,8n)) : {Nat.add(Nat.mul(1n,cells),_) == cells : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(1n,cells),0n),Nat.mul(1n,cells),cells,Addition.zero_right(Nat.mul(1n,cells)),one_times(cells))

# Lanes of more than one group go past eight.
def eight_below(small: Bool,+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+classified: {Nat.is_le(U32.to_nat(lanes),8n) == small : Bool},
  +several: {Nat.is_le(U32.to_nat(Grid.groups(lanes)),1n) == False{} : Bool}) -> {Nat.is_lt(8n,U32.to_nat(lanes)) == True{} : Bool}:
  match small:
    case True{}:
      TiledValues.Groups{+exact,+fewer,+room} = facts
      +one = {%Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),exact) : {Nat.is_le(_,1n) == True{} : Bool}
        Ceiling.minimal(U32.to_nat(lanes),8n,1n,{==},classified) : {Nat.is_le(U32.to_nat(Grid.groups(lanes)),1n) == True{} : Bool}}
      Empty.absurd({Nat.is_lt(8n,U32.to_nat(lanes)) == True{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(U32.to_nat(Grid.groups(lanes)),1n),True{},
          Equal.sym(Bool,Nat.is_le(U32.to_nat(Grid.groups(lanes)),1n),False{},several),one)))
    case False{}: Order.ge_false(8n,U32.to_nat(lanes),Equal.trans(Bool,Nat.is_ge(8n,U32.to_nat(lanes)),Nat.is_le(U32.to_nat(lanes),8n),False{},Order.ge_le(8n,U32.to_nat(lanes)),classified))

def outer_single(small: Bool,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+facts: KeptReads.SourceFacts<limits,shape,axis,Grid.Outermost{}>,
  +grouped: TiledValues.Groups<limits,Grid.lanes_at(shape,axis)>,
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classified: {Nat.is_le(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),1n) == small : Bool},
  +last: {small || U32.is_eq(KeptReads.group_step(shape,axis,Grid.Outermost{}),1) == True{} : Bool}) ->
  {small || Nat.is_eq(U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)),1n) == True{} : Bool}:
  match small:
    case True{}: {==}
    case False{}:
      KeptReads.SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
      +lanes = Grid.lanes_at(shape,axis)
      +rank = List.length(&2,U32,shape)
      +step = KeptReads.group_step(shape,axis,Grid.Outermost{})
      +outer = (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)
      +one = Arithmetic.equal_of_is_eq(step,1,last)
      +eight = eight_below(Nat.is_le(U32.to_nat(lanes),8n),limits,lanes,grouped,{==},classified)
      +term = KeptReads.group_term(1n,shape,axis,Grid.Outermost{},limits,KeptReads.SourceFacts{within,rank_room,room,cells_room,nonzero},unit_below(shape,axis,8n,eight,nonzero))
      +cell = KeptReads.cell_value(shape,axis,Grid.Outermost{},KeptReads.unit_digits(rank,axis,8n))
      +outer_value = Equal.trans(Nat,U32.to_nat(outer),ShapeLaws.elements(Grid.removed(shape,axis)),1n,Coordinates.outer_word(limits,shape,axis,removed_room),
        Equal.trans(Nat,ShapeLaws.elements(Grid.removed(shape,axis)),cell,1n,Equal.sym(Nat,cell,ShapeLaws.elements(Grid.removed(shape,axis)),unit_cell(shape,axis,within)),
          Equal.trans(Nat,cell,Nat.mul(1n,U32.to_nat(step)),1n,Equal.sym(Nat,Nat.mul(1n,U32.to_nat(step)),cell,term),
            Equal.trans(Nat,Nat.mul(1n,U32.to_nat(step)),U32.to_nat(step),1n,one_times(U32.to_nat(step)),Equal.cong(U32,Nat,U32.to_nat,step,1,one)))))
      %Equal.sym(Nat,U32.to_nat(outer),1n,outer_value) : {Nat.is_eq(_,1n) == True{} : Bool}
      {==}

# A source whose group step is one or whose groups are one has its groups
# innermost.
def single_of(position: Grid.Position,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+facts: KeptReads.SourceFacts<limits,shape,axis,position>,
  +grouped: TiledValues.Groups<limits,Grid.lanes_at(shape,axis)>,
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +last: {U32.is_le(Grid.groups(Grid.lanes_at(shape,axis)),1) || U32.is_eq(KeptReads.group_step(shape,axis,position),1) == True{} : Bool}) ->
  Single(position,U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32))):
  match position:
    case Grid.Innermost{}: Unit{}
    case Grid.Outermost{}:
      +groups = Grid.groups(Grid.lanes_at(shape,axis))
      outer_single(Nat.is_le(U32.to_nat(groups),1n),limits,shape,axis,facts,grouped,removed_room,{==},
        {%Comparison.le(groups,1) : {_ || U32.is_eq(KeptReads.group_step(shape,axis,Grid.Outermost{}),1) == True{} : Bool} last :
          {Nat.is_le(U32.to_nat(groups),1n) || U32.is_eq(KeptReads.group_step(shape,axis,Grid.Outermost{}),1) == True{} : Bool}})

# A cell read in order has the group of the target cell.
def kept_group(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +lanes_same: {Grid.lanes_at(shape,axis) == extent : U32},
  +single: Single(position,U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32))),
  +index: Nat,+below: {Nat.is_lt(index,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),
    U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)))) == True{} : Bool}) ->
  {Representation.cell_group(Public.geometry_of(shape,axis,position),index) == Representation.cell_group(Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{}),index) : Nat}:
  KeptReads.KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +within = KeptReads.source_within(limits,shape,axis,position,facts)
  +outer = (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)
  +groups = U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis)))
  +target_lanes = Equal.trans(U32,Grid.lanes_at(shape,axis),extent,Grid.lanes_at(target,lane),lanes_same,extent_exact)
  %Equal.sym(Cells.Cells,Public.geometry_of(shape,axis,position),Coordinates.geometry_inside(shape,axis,position),Coordinates.geometry_same(shape,axis,position,within)) :
    {Representation.cell_group(_,index) == Representation.cell_group(Public.geometry_of(target,lane,Grid.Innermost{}),index) : Nat}
  %Equal.sym(Cells.Cells,Public.geometry_of(target,lane,Grid.Innermost{}),Coordinates.geometry_inside(target,lane,Grid.Innermost{}),
      Coordinates.geometry_same(target,lane,Grid.Innermost{},lane_inside)) :
    {Representation.cell_group(Coordinates.geometry_inside(shape,axis,position),index) == Representation.cell_group(_,index) : Nat}
  %target_lanes : {Representation.group_at(position,U32.to_nat(outer),groups,index) == Nat.mod(index,Representation.least_one(U32.to_nat(Grid.groups(_)))) : Nat}
  outer_group(position,U32.to_nat(outer),groups,index,single,below)

def pick_twice(choice: Bool,-value: F32) -> {Bool.pick(F32,choice,value,0.0) == Bool.pick(F32,choice,Bool.pick(F32,choice,value,0.0),0.0) : F32}:
  match choice:
    case True{}: {==}
    case False{}: {==}

# A lane of a lifted cell is the lane kept below the lane count.
def lifted_kept(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+word: U32,+selected: Nat,
  +small: {Nat.is_le(selected,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(selected,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)) ==
    PaddedCells.kept(Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),selected,
      TileLaws.select_lane(selected,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word))) : F32}:
  +number = U32.to_nat(word)
  +outer = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +channel = Nat.add(selected,Nat.mul(group,8n))
  %Equal.sym(F32,TileLaws.select_lane(selected,Representation.cell(~Logical,~original,logical,geometry,outer,group)),
      Representation.lane(~Logical,~original,logical,geometry,outer,group,selected),Representation.cell_lane(~Logical,~original,logical,geometry,outer,group,selected,small)) :
    {_ == PaddedCells.kept(Representation.lane_count(geometry),group,selected,_) : F32}
  pick_twice(Nat.is_lt(channel,Representation.lane_count(geometry)),original(logical,U32.from_nat(Representation.logical_index(geometry,outer,channel))))

def held_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: Cells.Cells,+geometry: Cells.Cells,+word: U32,
  +count_same: {Representation.lane_count(source) == Representation.lane_count(geometry) : Nat},
  +group_same: {Representation.cell_group(source,U32.to_nat(word)) == Representation.cell_group(geometry,U32.to_nat(word)) : Nat},
  +selected: Nat,+small: {Nat.is_le(selected,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(selected,Representation.lifted(~Logical,~original,Representation.Lifted{logical,source},word)) ==
    PaddedCells.kept(Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),selected,
      TileLaws.select_lane(selected,Representation.lifted(~Logical,~original,Representation.Lifted{logical,source},word))) : F32}:
  +tile = Representation.lifted(~Logical,~original,Representation.Lifted{logical,source},word)
  %count_same : {TileLaws.select_lane(selected,tile) == PaddedCells.kept(_,Representation.cell_group(geometry,U32.to_nat(word)),selected,TileLaws.select_lane(selected,tile)) : F32}
  %group_same : {TileLaws.select_lane(selected,tile) == PaddedCells.kept(Representation.lane_count(source),_,selected,TileLaws.select_lane(selected,tile)) : F32}
  lifted_kept(~Logical,~original,logical,source,word,selected,small)

# A cell whose lanes hold the target lanes below the lane count, with the
# lane count and the group of the target, is the target cell.
def kept_lanes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source_geometry: Cells.Cells,+geometry: Cells.Cells,+word: U32,+target_word: Tile.Tile8,
  +count_same: {Representation.lane_count(source_geometry) == Representation.lane_count(geometry) : Nat},
  +group_same: {Representation.cell_group(source_geometry,U32.to_nat(word)) == Representation.cell_group(geometry,U32.to_nat(word)) : Nat},
  +valid: PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_geometry},word),target_word>) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_geometry},word) == target_word : Tile.Tile8}:
  PaddedCells.Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = valid
  +source_word = Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_geometry},word)
  +count = Representation.lane_count(geometry)
  +group = Representation.cell_group(geometry,U32.to_nat(word))
  TileLaws.tiles_equal(source_word,target_word,
    Equal.trans(F32,Tile.lane0(source_word),PaddedCells.kept(count,group,0n,Tile.lane0(source_word)),Tile.lane0(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,0n,{==}),lane0),
    Equal.trans(F32,Tile.lane1(source_word),PaddedCells.kept(count,group,1n,Tile.lane1(source_word)),Tile.lane1(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,1n,{==}),lane1),
    Equal.trans(F32,Tile.lane2(source_word),PaddedCells.kept(count,group,2n,Tile.lane2(source_word)),Tile.lane2(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,2n,{==}),lane2),
    Equal.trans(F32,Tile.lane3(source_word),PaddedCells.kept(count,group,3n,Tile.lane3(source_word)),Tile.lane3(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,3n,{==}),lane3),
    Equal.trans(F32,Tile.lane4(source_word),PaddedCells.kept(count,group,4n,Tile.lane4(source_word)),Tile.lane4(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,4n,{==}),lane4),
    Equal.trans(F32,Tile.lane5(source_word),PaddedCells.kept(count,group,5n,Tile.lane5(source_word)),Tile.lane5(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,5n,{==}),lane5),
    Equal.trans(F32,Tile.lane6(source_word),PaddedCells.kept(count,group,6n,Tile.lane6(source_word)),Tile.lane6(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,6n,{==}),lane6),
    Equal.trans(F32,Tile.lane7(source_word),PaddedCells.kept(count,group,7n,Tile.lane7(source_word)),Tile.lane7(target_word),
      held_lane(~Logical,~original,logical,source_geometry,geometry,word,count_same,group_same,7n,{==}),lane7))

# A cell a view reads in order holds the target cell of the viewed values.
def kept_word(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +lanes_same: {Grid.lanes_at(shape,axis) == extent : U32},
  +zero: {U32.is_zero(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)) == True{} : Bool},
  +none: {TiledView.none_flagged(kept_flags) == True{} : Bool},
  +contiguous: {Views.has_count(Views.contiguous_size(KeptReads.cell_axes_of(kept,extent,shape,axis,position))) == True{} : Bool},
  +single: Single(position,U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32))),
  +cells_total: {ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})) ==
    Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32))) : Nat},
  +word: U32,+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}))) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},word) ==
    Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})},word) : Tile.Tile8}:
  KeptReads.KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
  +cell_axes = KeptReads.cell_axes_of(kept,extent,shape,axis,position)
  +extents = Views.axis_extents(cell_axes)
  +cell_flags = List.append(&2,Bool,kept_flags,[False{}])
  +offset_cell = TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)
  +number = U32.to_nat(word)
  +within = KeptReads.source_within(limits,shape,axis,position,facts)
  +source_geometry = Public.geometry_of(shape,axis,position)
  +geometry = Public.geometry_of(target,lane,Grid.Innermost{})
  +source = {Representation.Lifted{logical,source_geometry} : Representation.Lifted<Logical>}
  +source_word = Representation.lifted(~Logical,~original,source,word)
  +target_word = Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
    Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},geometry},word)
  +extents_same = KeptReads.cell_extents(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +inside_extents = {%Equal.sym(List<&2,U32>,extents,cells,extents_same) : {Nat.is_lt(number,ShapeLaws.elements(_)) == True{} : Bool} inside :
    {Nat.is_lt(number,ShapeLaws.elements(extents)) == True{} : Bool}}
  +extents_room = {%Equal.sym(List<&2,U32>,extents,cells,extents_same) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    target_cells_room : {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +sum = Box.stepped_sum(extents,ViewPart.view_steps(cell_axes),number)
  +sum_word = Arithmetic.to_nat_injective(sum,word,contiguous_word(Views.contiguous_size(cell_axes),cell_axes,{==},contiguous,number,inside_extents,
    Machine.capacity(limits),Limits.capacity_fits(limits),extents_room))
  +forward = ViewGather.forward_append(kept_flags,none_forward(kept_flags,none))
  +read = {%Equal.sym(U32,Views.mirrored_offset(cell_axes,cell_flags,offset_cell),offset_cell,ViewGather.forward_offset(cell_axes,cell_flags,offset_cell,forward)) :
      {Representation.lifted(~Logical,~original,source,(_ + Box.stepped_sum(extents,ViewPart.view_steps(Views.mirrored_axes(cell_axes,cell_flags)),number) : U32)) == source_word : Tile.Tile8}
    %Equal.sym(List<&2,Views.Axis>,Views.mirrored_axes(cell_axes,cell_flags),cell_axes,ViewGather.forward_axes(cell_axes,cell_flags,forward)) :
      {Representation.lifted(~Logical,~original,source,(offset_cell + Box.stepped_sum(extents,ViewPart.view_steps(_),number) : U32)) == source_word : Tile.Tile8}
    %Equal.sym(U32,offset_cell,0,Arithmetic.word_zero(offset_cell,zero)) : {Representation.lifted(~Logical,~original,source,(_ + sum : U32)) == source_word : Tile.Tile8}
    %Equal.sym(U32,(0 + sum : U32),sum,Arithmetic.add_zero_left(sum)) : {Representation.lifted(~Logical,~original,source,_) == source_word : Tile.Tile8}
    Equal.cong(U32,Tile.Tile8,at => Representation.lifted(~Logical,~original,source,at),sum,word,sum_word) :
    {ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{source,cell_axes,offset_cell,cell_flags},word) ==
      source_word : Tile.Tile8}}
  +valid_word = {%read : PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,number),_,target_word>
    KeptReads.word_valid(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,word,inside) :
    PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,number),source_word,target_word>}
  +below = {%cells_total : {Nat.is_lt(number,_) == True{} : Bool} inside :
    {Nat.is_lt(number,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)))) == True{} : Bool}}
  +group_same = kept_group(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,lanes_same,single,number,below)
  +count_same = Equal.trans(Nat,Representation.lane_count(source_geometry),U32.to_nat(Grid.lanes_at(shape,axis)),Representation.lane_count(geometry),
    Coordinates.geometry_lanes(shape,axis,position,Bounds.strict_le(axis,List.length(&2,U32,shape),within)),
    Equal.trans(Nat,U32.to_nat(Grid.lanes_at(shape,axis)),U32.to_nat(Grid.lanes_at(target,lane)),Representation.lane_count(geometry),
      Equal.cong(U32,Nat,U32.to_nat,Grid.lanes_at(shape,axis),Grid.lanes_at(target,lane),Equal.trans(U32,Grid.lanes_at(shape,axis),extent,Grid.lanes_at(target,lane),lanes_same,extent_exact)),
      Equal.sym(Nat,Representation.lane_count(geometry),U32.to_nat(Grid.lanes_at(target,lane)),
        Coordinates.geometry_lanes(target,lane,Grid.Innermost{},Bounds.strict_le(lane,List.length(&2,U32,target),lane_inside)))))
  kept_lanes(~Logical,~original,logical,source_geometry,geometry,word,target_word,count_same,group_same,valid_word)

# Usable cells a view reads in order from the first, kept under the target
# shape: the contract on the axis the view keeps.
def kept_relabeled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +lanes: U32,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +laid: Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers}),
  +pinned: {Grid.lanes_at(shape,axis) == lanes : U32},
  +classified: {TiledView.in_order(Tensors.Usable{},KeptReads.cell_axes_of(kept,extent,shape,axis,position),kept_flags,
    TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),lanes,extent,Tiled.cell_shape(shape,axis,position)) == True{} : Bool}) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    Tiled.Tiled{Tensors.relabeled(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers},
      Views.axis_extents(KeptReads.cell_axes_of(kept,extent,shape,axis,position))),lane,extent,Grid.Innermost{}}):
  KeptReads.KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
  +cell_axes = KeptReads.cell_axes_of(kept,extent,shape,axis,position)
  +extents = Views.axis_extents(cell_axes)
  +offset_cell = TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)
  +step = KeptReads.group_step(shape,axis,position)
  +source_cells = Tiled.cell_shape(shape,axis,position)
  +within = KeptReads.source_within(limits,shape,axis,position,facts)
  +source = {Representation.Lifted{logical,Public.geometry_of(shape,axis,position)} : Representation.Lifted<Logical>}
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  +geometry = Public.geometry_of(target,lane,Grid.Innermost{})
  +source_count = Tensors.count_or_zero(source_cells)
  +target_count = Tensors.count_or_zero(extents)
  +usable_zero = Tensors.usable(Tensors.Usable{}) && U32.is_zero(offset_cell)
  +usable_none = usable_zero && TiledView.none_flagged(kept_flags)
  +usable_lanes = usable_none && U32.is_eq(lanes,extent)
  +usable_count = usable_lanes && U32.is_lt(0,source_count)
  +usable_same = usable_count && U32.is_eq(source_count,target_count)
  +contiguous = Views.is_contiguous(Views.View{cell_axes,offset_cell})
  +held_same = Boolean.left(usable_same,contiguous,classified)
  +held_contiguous = Boolean.right(usable_same,contiguous,classified)
  +held_count = Boolean.left(usable_count,U32.is_eq(source_count,target_count),held_same)
  +counts_equal = Boolean.right(usable_count,U32.is_eq(source_count,target_count),held_same)
  +held_lanes = Boolean.left(usable_lanes,U32.is_lt(0,source_count),held_count)
  +held_none = Boolean.left(usable_none,U32.is_eq(lanes,extent),held_lanes)
  +lanes_equal = Boolean.right(usable_none,U32.is_eq(lanes,extent),held_lanes)
  +held_zero = Boolean.left(usable_zero,TiledView.none_flagged(kept_flags),held_none)
  +none = Boolean.right(usable_zero,TiledView.none_flagged(kept_flags),held_none)
  +zero = Boolean.right(Tensors.usable(Tensors.Usable{}),U32.is_zero(offset_cell),held_zero)
  +nonempty = KeptReads.cell_nonempty(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +counted_contiguous = {%nonempty : {Views.contiguous_nonempty(_,cell_axes) == True{} : Bool} held_contiguous :
    {Views.has_count(Views.contiguous_size(cell_axes)) == True{} : Bool}}
  +lanes_same = Equal.trans(U32,Grid.lanes_at(shape,axis),lanes,extent,pinned,Arithmetic.equal_of_is_eq(lanes,extent,lanes_equal))
  +step_last = last_contiguous(kept,Tiled.groups(extent),step,counted_contiguous)
  +last = {%Equal.sym(U32,Grid.lanes_at(shape,axis),extent,lanes_same) : {U32.is_le(Grid.groups(_),1) || U32.is_eq(step,1) == True{} : Bool} step_last :
    {U32.is_le(Grid.groups(Grid.lanes_at(shape,axis)),1) || U32.is_eq(step,1) == True{} : Bool}}
  +single = single_of(position,limits,shape,axis,facts,grouped,removed_room,last)
  +extents_same = KeptReads.cell_extents(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +cells_fit = {%Equal.sym(List<&2,U32>,extents,cells,extents_same) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} target_cells_room :
    {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_counted = StorageCount.counted_within(limits,extents,cells_fit,default_capacity)
  +cells_total = Global.count(extents,cells_counted)
  +positive = CountCompletion.positive_elements(extents,nonempty)
  +cells_nonzero = Arithmetic.word_positive_nonzero(cells_total,
    {%Count.count_exact(ShapeLaws.elements(extents),cells_total,Global.evidence(extents,cells_counted)) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
      {Nat.is_lt(0n,U32.to_nat(cells_total)) == True{} : Bool}})
  +target_known = Equal.cong(Maybe<&2,U32>,U32,found => Tensors.known_count(found),Tensors.count_of(extents),Some{cells_total},Global.accepted(extents,cells_counted))
  +counts_same = Equal.trans(U32,source_count,target_count,cells_total,Arithmetic.equal_of_is_eq(source_count,target_count,counts_equal),target_known)
  +source_counted = StorageCount.counted_within(limits,source_cells,KeptReads.source_cells_room(limits,shape,axis,position,facts),default_capacity)
  +source_total = Global.count(source_cells,source_counted)
  +source_accepted = Global.accepted(source_cells,source_counted)
  +source_known = Equal.cong(Maybe<&2,U32>,U32,found => Tensors.known_count(found),Tensors.count_of(source_cells),Some{source_total},source_accepted)
  +totals_same = Equal.trans(U32,source_total,source_count,cells_total,Equal.sym(U32,source_count,source_total,source_known),counts_same)
  +source_exact = {%totals_same : {Tensors.count_of(source_cells) == Some{_} : Maybe<&2,U32>} source_accepted :
    {Tensors.count_of(source_cells) == Some{cells_total} : Maybe<&2,U32>}}
  +removed_cells = ShapeLaws.elements(Grid.removed(shape,axis))
  +groups = U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis)))
  +outer = (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)
  +cells_split = Equal.trans(Nat,ShapeLaws.elements(cells),ShapeLaws.elements(extents),Nat.mul(groups,U32.to_nat(outer)),
    Equal.cong(List<&2,U32>,Nat,list => ShapeLaws.elements(list),cells,extents,Equal.sym(List<&2,U32>,extents,cells,extents_same)),
    Equal.trans(Nat,ShapeLaws.elements(extents),U32.to_nat(cells_total),Nat.mul(groups,U32.to_nat(outer)),
      Count.count_exact(ShapeLaws.elements(extents),cells_total,Global.evidence(extents,cells_counted)),
      Equal.trans(Nat,U32.to_nat(cells_total),U32.to_nat(source_count),Nat.mul(groups,U32.to_nat(outer)),
        Equal.cong(U32,Nat,U32.to_nat,cells_total,source_count,Equal.sym(U32,source_count,cells_total,counts_same)),
        Equal.trans(Nat,U32.to_nat(source_count),ShapeLaws.elements(source_cells),Nat.mul(groups,U32.to_nat(outer)),cells_count,
          Equal.trans(Nat,ShapeLaws.elements(source_cells),Nat.mul(groups,removed_cells),Nat.mul(groups,U32.to_nat(outer)),cell_elements(position,shape,axis,within),
            Equal.cong(Nat,Nat,count => Nat.mul(groups,count),removed_cells,U32.to_nat(outer),
              Equal.sym(Nat,U32.to_nat(outer),removed_cells,Coordinates.outer_word(limits,shape,axis,removed_room))))))))
  +relabeled = Relabeled.relabeled(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),source,limits,source_cells,extents,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},
    KernelGemm.Both{Global.Usable{model,fill,workers,{==}},global},cells_total,source_exact,Global.accepted(extents,cells_counted),cells_nonzero,default_capacity)
  +on_cells = {%extents_same : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),source,_,Tensors.relabeled(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},extents))
    relabeled : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),source,cells,Tensors.relabeled(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},extents))}
  +restated = Global.restated_below(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.Lifted<ViewGather.Viewing<Logical>>,~Representation.lifted(~Logical,~original),
    ~Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original)),~ShapeLaws.elements(cells),source,Representation.Lifted{viewing,geometry},
    word => inside => kept_word(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,lanes_same,zero,none,counted_contiguous,
      single,cells_split,word,inside),
    cells,Tensors.relabeled(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},extents),on_cells,Order.reflexive(ShapeLaws.elements(cells)),Machine.capacity(limits),target_cells_room)
  Public.laid_contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),viewing,target,Tensors.relabeled(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source_cells,fill,Tensors.Usable{},workers},extents),lane,extent,
    Grid.Innermost{},KernelGemm.Both{restated,laid})

# The cells of the contract on the axis a view keeps: kept when the view
# reads every cell in order, moved through the view otherwise.
def kept_choice(~Logical: Data,~original: Logical -> U32 -> F32,chosen: Bool,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,
  +position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +lanes: U32,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +laid: Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>,
  +arranged: PaddedCells.Placed<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Views.axis_extents(axes),lane,extent,Grid.Innermost{},
    Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})))>,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers}),
  +pinned: {Grid.lanes_at(shape,axis) == lanes : U32},
  +classified: {TiledView.in_order(Tensors.Usable{},KeptReads.cell_axes_of(kept,extent,shape,axis,position),kept_flags,
    TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),lanes,extent,Tiled.cell_shape(shape,axis,position)) == chosen : Bool}) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.moved_or_kept(chosen,name,lane,extent,KeptReads.cell_axes_of(kept,extent,shape,axis,position),kept_flags,
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),
      Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),fill,Tensors.Usable{},workers})):
  match chosen:
    case True{}:
      kept_relabeled(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,model,fill,workers,lanes,default_capacity,laid,
        global,pinned,classified)
    case False{}:
      kept_cells(~Logical,~original,logical,limits,name,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,model,fill,workers,default_capacity,target_rank,
        laid,arranged,global)

# The cells of a contract are a Shaped tensor of the cell shape whose lane
# axis has the lanes of the tiled values.
def shaped_pinned(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,
  cells: Tensors.Tensor<Tile.Tile8>,contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position})) ->
  KernelGemm.Both<Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),cells),
    {Grid.lanes_at(values,axis) == lanes : U32}>:
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      KernelGemm.Both{global,Public.Laid{inside,+shaped,restored,counted,+pinned}} = contract
      KernelGemm.Both{KernelGemm.Both{
        %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : Global.Usable<Tile.Tile8,_,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>
          Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),
        global},pinned}
    case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}:
      Empty.absurd(KernelGemm.Both<Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
          Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}),
        {Grid.lanes_at(values,axis) == lanes : U32}>,contract)

# Usable cells of the contract on the axis a view keeps.
def kept_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,-cells: Tensors.Tensor<Tile.Tile8>,+lanes: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +laid: Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>,
  +arranged: PaddedCells.Placed<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Views.axis_extents(axes),lane,extent,Grid.Innermost{},
    Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})))>,
  +held: KernelGemm.Both<Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},Tiled.cell_shape(shape,axis,position),cells),
    {Grid.lanes_at(shape,axis) == lanes : U32}>) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.moved_cells(name,lane,extent,kept,kept_flags,TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),(8 * TiledView.lane_inner(TiledView.lanes_of(shape,axis,position)) : U32)),Tiled.Tiled{cells,axis,lanes,position})):
  KernelGemm.Both{KernelGemm.Both{Global.Usable{+model,+model_fill,+model_workers,+equation},+global},+pinned} = held
  %equation : Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.moved_cells(name,lane,extent,kept,kept_flags,TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),(8 * TiledView.lane_inner(TiledView.lanes_of(shape,axis,position)) : U32)),Tiled.Tiled{_,axis,lanes,position}))
  kept_choice(~Logical,~original,TiledView.in_order(Tensors.Usable{},KeptReads.cell_axes_of(kept,extent,shape,axis,position),kept_flags,
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),lanes,extent,Tiled.cell_shape(shape,axis,position)),
    logical,limits,name,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,model,model_fill,model_workers,lanes,default_capacity,target_rank,laid,arranged,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(shape,axis,position),model_fill,Tensors.Usable{},model_workers},cells,equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},_)
    global,pinned,{==})

# Tiled values of the contract on the axis a view keeps.
def kept_along(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,cells: Tensors.Tensor<Tile.Tile8>,+lanes: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +laid: Public.Laid<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),lane,extent,Grid.Innermost{},Views.axis_extents(axes)>,
  +arranged: PaddedCells.Placed<Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Views.axis_extents(axes),lane,extent,Grid.Innermost{},
    Nat.is_lt(lane,List.length(&2,U32,Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})))>,
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{cells,axis,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.moved_cells(name,lane,extent,kept,kept_flags,TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),
      TiledView.cell_index(TiledView.lanes_of(shape,axis,position),(8 * TiledView.lane_inner(TiledView.lanes_of(shape,axis,position)) : U32)),Tiled.Tiled{cells,axis,lanes,position})):
  kept_usable(~Logical,~original,logical,limits,name,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,cells,lanes,default_capacity,target_rank,laid,arranged,
    shaped_pinned(~Logical,~original,logical,shape,axis,lanes,position,cells,contract))

# The lane of the mapping of a view the runtime keeps the lanes of: the
# cells moved through the view of the cells.
def kept_lane(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,Nat>,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+valid: Bool,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+reach: List<&2,U32>,
  +same: {TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)) ==
    TiledView.Mapped{valid,found,extent,kept,kept_flags,reach} : TiledView.Mapped},
  +held: {TiledView.mapped_keeps(TiledView.Mapped{valid,found,extent,kept,kept_flags,reach},shape,TiledView.coordinate_at(Grid.coordinates(shape,offset),axis)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.gathered_lane(found,name,offset,TiledView.lanes_of(shape,axis,position),extent,kept,kept_flags,
      Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})):
  match found:
    case Some{+lane}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+source_counted,+pinned}} = contract
      +target = Views.axis_extents(axes)
      +lanes_found = TiledView.lanes_of(shape,axis,position)
      +mapped = TiledView.mapped_axes(axes,flags,0n,shape,lanes_found,Grid.coordinates(shape,offset))
      +first = TiledView.coordinate_at(Grid.coordinates(shape,offset),axis)
      +inside_reach = TiledView.all_inside(reach,shape)
      +aligned = U32.is_zero((first % 8 : U32))
      +kept_reach = Boolean.left((valid && Maybe.is_some(&2,Nat,Some{lane})) && inside_reach,aligned,held)
      +kept_lane = Boolean.left(valid && Maybe.is_some(&2,Nat,Some{lane}),inside_reach,kept_reach)
      +valid_true = Equal.trans(Bool,KeptReads.mapped_valid(mapped),valid,True{},
        Equal.cong(TiledView.Mapped,Bool,value => KeptReads.mapped_valid(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same),
        Boolean.left(valid,Maybe.is_some(&2,Nat,Some{lane}),kept_lane))
      +reach_inside = {%Equal.sym(List<&2,U32>,KeptReads.mapped_reach(mapped),reach,
          Equal.cong(TiledView.Mapped,List<&2,U32>,value => KeptReads.mapped_reach(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same)) :
          {TiledView.all_inside(_,shape) == True{} : Bool}
        Boolean.right(valid && Maybe.is_some(&2,Nat,Some{lane}),inside_reach,kept_reach) : {TiledView.all_inside(KeptReads.mapped_reach(mapped),shape) == True{} : Bool}}
      +view = KeptReads.kept_view(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,valid_true,
        Equal.cong(TiledView.Mapped,Maybe<&2,Nat>,value => KeptReads.mapped_lane(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same),
        Equal.cong(TiledView.Mapped,U32,value => KeptReads.mapped_extent(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same),
        Equal.cong(TiledView.Mapped,List<&2,Views.Axis>,value => KeptReads.mapped_kept(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same),
        Equal.cong(TiledView.Mapped,List<&2,Bool>,value => KeptReads.mapped_flags(value),mapped,TiledView.Mapped{valid,Some{lane},extent,kept,kept_flags,reach},same),
        reach_inside,Boolean.right((valid && Maybe.is_some(&2,Nat,Some{lane})) && inside_reach,aligned,held),
        within,nonzero,target_nonzero,default_capacity,rank_room,source_counted,Global.Counted{count,counted},fits)
      +laid = target_laid(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,count,counted)
      +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
      +arranged = PaddedCells.placed(Nat.is_lt(lane,List.length(&2,U32,cells)),target,lane,extent,Grid.Innermost{},cells,{==},laid)
      +offset_cell = TiledView.cell_index(lanes_found,offset)
      +step = TiledView.cell_index(lanes_found,(8 * TiledView.lane_inner(lanes_found) : U32))
      +axis_exact = KeptReads.lane_axis_of(shape,axis,position,Bounds.strict_le(axis,List.length(&2,U32,shape),within))
      %Equal.sym(Nat,TiledView.lane_axis(lanes_found),axis,axis_exact) :
        Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},target,
          TiledView.moved_cells(name,lane,extent,kept,kept_flags,offset_cell,step,Tiled.along(Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position},_)))
      Relane.along_usable(~Logical,~original,(tiled => Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},target,
          TiledView.moved_cells(name,lane,extent,kept,kept_flags,offset_cell,step,tiled))),
        logical,shape,pieces,cells_shape,fill,workers,old,lanes,position,axis,limits,default_capacity,Bounds.strict_le(axis,List.length(&2,U32,shape),within),
        CountCompletion.positive_elements(shape,nonzero),rank_room,Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},{==}),contract,
        cells => lanes => contract => kept_along(~Logical,~original,logical,limits,name,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,cells,lanes,
          default_capacity,target_rank,laid,arranged,contract))
    case None{}:
      +inside_reach = TiledView.all_inside(reach,shape)
      +kept_reach = Boolean.left((valid && Maybe.is_some(&2,Nat,None{})) && inside_reach,U32.is_zero((TiledView.coordinate_at(Grid.coordinates(shape,offset),axis) % 8 : U32)),held)
      +kept_lane = Boolean.left(valid && Maybe.is_some(&2,Nat,None{}),inside_reach,kept_reach)
      Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
          Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position}),
        WordEncoding.false_true(Boolean.right(valid,Maybe.is_some(&2,Nat,None{}),kept_lane)))

# The mapping of a view the runtime keeps the lanes of.
def kept_mapped(~Logical: Data,~original: Logical -> U32 -> F32,mapped: TiledView.Mapped,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,
  +same: {TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)) == mapped : TiledView.Mapped},
  +keeps: {TiledView.keeps(axes,offset,flags,shape,TiledView.lanes_of(shape,axis,position)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.gather_kept(mapped,name,offset,TiledView.lanes_of(shape,axis,position),Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})):
  match mapped:
    case TiledView.Mapped{+valid,found,+extent,+kept,+kept_flags,+reach}:
      +lanes_found = TiledView.lanes_of(shape,axis,position)
      +words = Grid.coordinates(shape,offset)
      +axis_exact = KeptReads.lane_axis_of(shape,axis,position,Bounds.strict_le(axis,List.length(&2,U32,shape),within))
      +held = {%same : {TiledView.mapped_keeps(_,shape,TiledView.coordinate_at(words,axis)) == True{} : Bool}
        %axis_exact : {TiledView.mapped_keeps(TiledView.mapped_axes(axes,flags,0n,shape,lanes_found,words),shape,TiledView.coordinate_at(words,_)) == True{} : Bool}
        keeps : {TiledView.mapped_keeps(TiledView.Mapped{valid,found,extent,kept,kept_flags,reach},shape,TiledView.coordinate_at(words,axis)) == True{} : Bool}}
      kept_lane(~Logical,~original,found,logical,limits,name,shape,axis,axes,flags,offset,pieces,cells_shape,fill,workers,old,lanes,position,valid,extent,kept,kept_flags,reach,
        same,held,within,nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)

# Usable tiled values of the contract through a view the runtime keeps the
# lanes of on an axis of the input: the tiled form of the view.
def kept_gathered(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axis: Nat,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+keeps: {TiledView.keeps(axes,offset,flags,shape,TiledView.lanes_of(shape,axis,position)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    TiledView.gather_at(name,axes,offset,flags,shape,TiledView.lanes_of(shape,axis,position),Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})):
  kept_mapped(~Logical,~original,TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)),logical,limits,name,shape,axis,axes,flags,offset,
    pieces,cells_shape,fill,workers,old,lanes,position,{==},keeps,within,nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)

# ---- Gathers ----

# Where the search for an axis whose lanes a view keeps stops: an axis of the
# input the runtime keeps the lanes of, or none.
def KeptFound(found: Maybe<&2,Nat>,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+position: Grid.Position) -> Data:
  match found:
    case Some{+axis}:
      KernelGemm.Both<{Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},{TiledView.keeps(axes,offset,reversed,shape,TiledView.lanes_of(shape,axis,position)) == True{} : Bool}>
    case None{}: Unit

def kept_pick(chosen: Bool,+axis: Nat,+later: Maybe<&2,Nat>,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+position: Grid.Position,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) && TiledView.keeps(axes,offset,reversed,shape,TiledView.lanes_of(shape,axis,position)) == chosen : Bool},
  +found: KeptFound(later,axes,offset,reversed,shape,position)) -> KeptFound(Bool.pick(Maybe<&2,Nat>,chosen,Some{axis},later),axes,offset,reversed,shape,position):
  match chosen:
    case True{}:
      KernelGemm.Both{Boolean.left(Nat.is_lt(axis,List.length(&2,U32,shape)),TiledView.keeps(axes,offset,reversed,shape,TiledView.lanes_of(shape,axis,position)),classified),
        Boolean.right(Nat.is_lt(axis,List.length(&2,U32,shape)),TiledView.keeps(axes,offset,reversed,shape,TiledView.lanes_of(shape,axis,position)),classified)}
    case False{}: found

# The search for an axis whose lanes a view keeps finds one or none.
def kept_among_found(candidates: List<&2,Nat>,+name: String,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,
  +position: Grid.Position) ->
  KeptFound(TiledView.kept_among(candidates,TiledView.Gathering{name,axes,offset,reversed,shape,target},position),axes,offset,reversed,shape,position):
  match candidates:
    case Nil{}: Unit{}
    case Con{+axis,+rest}:
      kept_pick(Nat.is_lt(axis,List.length(&2,U32,shape)) && TiledView.keeps(axes,offset,reversed,shape,TiledView.lanes_of(shape,axis,position)),axis,
        TiledView.kept_among(rest,TiledView.Gathering{name,axes,offset,reversed,shape,target},position),axes,offset,reversed,shape,position,{==},
        kept_among_found(rest,name,axes,offset,reversed,shape,target,position))

# The axis the search finds, or the unit axis when it finds none.
def gather_found(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,Nat>,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+old: Nat,+lanes: U32,
  +position: Grid.Position,+kept: KeptFound(found,axes,offset,reversed,shape,position),
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
    TiledView.gather_chosen(found,TiledView.Gathering{name,axes,offset,reversed,shape,Views.axis_extents(axes)},position,
      Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position})):
  match found:
    case Some{+axis}:
      KernelGemm.Both{+within,+keeps} = kept
      kept_gathered(~Logical,~original,logical,limits,name,shape,axis,axes,reversed,offset,pieces,cells_shape,fill,workers,old,lanes,position,keeps,within,nonzero,target_nonzero,
        default_capacity,rank_room,target_rank,count,counted,fits,contract)
    case None{}:
      +target = Views.axis_extents(axes)
      +positive = CountCompletion.positive_elements(target,target_nonzero)
      +count_nonzero = Arithmetic.word_positive_nonzero(count,
        {%Count.count_exact(ShapeLaws.elements(target),count,Count.count_of(target,count,counted)) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
          {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}})
      unit_gathered(~Logical,~original,logical,limits,name,shape,axes,offset,reversed,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position},
        default_capacity,rank_room,target_rank,count,counted,count_nonzero,fits,CountCompletion.positive_elements(shape,nonzero),contract)

# Tiled values of the contract through a view of their extents onto extents
# without a zero, read the runtime way: the tiled form of the view.
def gathered(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,tiled: Tiled.Tiled,+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
    TiledView.gather(name,Tensors.Usable{},Some{Views.View{axes,offset}},reversed,shape,Views.axis_extents(axes),tiled)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Usable{},+workers}:
          +candidates = TiledView.lane_candidates(shape,old,axes,Tiled.lane_axis(List.length(&2,U32,Views.axis_extents(axes))))
          %Equal.sym(Bool,Shapes.has_zero(shape),False{},nonzero) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
              TiledView.gather_started(_ || Shapes.has_zero(Views.axis_extents(axes)),name,axes,offset,reversed,shape,Views.axis_extents(axes),
                (Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position},old)))
          %Equal.sym(Bool,Shapes.has_zero(Views.axis_extents(axes)),False{},target_nonzero) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
              TiledView.gather_started(_,name,axes,offset,reversed,shape,Views.axis_extents(axes),
                (Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},old,lanes,position},old)))
          gather_found(~Logical,~original,TiledView.kept_among(candidates,TiledView.Gathering{name,axes,offset,reversed,shape,Views.axis_extents(axes)},position),
            logical,limits,name,shape,axes,offset,reversed,pieces,cells_shape,fill,workers,old,lanes,position,
            kept_among_found(candidates,name,axes,offset,reversed,shape,Views.axis_extents(axes),position),
            nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
            TiledView.gather(name,Tensors.Usable{},Some{Views.View{axes,offset}},reversed,shape,Views.axis_extents(axes),
              Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},old,lanes,position})),contract)

# ---- Layout gathers ----

# Tiled values of the contract through a view of their extents, labelled with
# the extents of the view: the tiled form of the view.
def gathered_extents(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+name: String,+values: List<&2,U32>,+axes: List<&2,Views.Axis>,
  +offset: U32,+reversed: List<&2,Bool>,tiled: Tiled.Tiled,+nonzero: {Shapes.has_zero(values) == False{} : Bool},
  +target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
    TiledLayout.gather_extents(name,Tensors.Usable{},Some{Views.View{axes,offset}},reversed,values,tiled)):
  gathered(~Logical,~original,logical,limits,name,values,axes,offset,reversed,tiled,nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)

# The result's axis i is the input axis order[i]: the tiled form of the view
# that permutes the axes.
def transposed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+order: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,tiled: Tiled.Tiled,
  +viewed: {Layout.permuted(Tensors.view_of(values),Layout.naturals(order)) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
    TiledLayout.transpose(tiled,order)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,old,lanes,position),values,contract_restored(~Logical,~original,logical,values,pieces,shape,fill,workers,old,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
              TiledLayout.gather_extents("transpose",Tensors.Usable{},Layout.permuted(Tensors.view_of(_),Layout.naturals(order)),[],_,
                Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          %Equal.sym(Maybe<&2,Views.View>,Layout.permuted(Tensors.view_of(values),Layout.naturals(order)),Some{Views.View{axes,offset}},viewed) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
              TiledLayout.gather_extents("transpose",Tensors.Usable{},_,[],values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          gathered_extents(~Logical,~original,logical,limits,"transpose",values,axes,offset,[],Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},
            nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
            TiledLayout.transpose(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},order)),contract)

# count elements from start along axis, every step-th one: the tiled form of
# the view that slices the axis.
def sliced(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: U32,+start: U32,+count: U32,+step: U32,
  +axes: List<&2,Views.Axis>,+offset: U32,tiled: Tiled.Tiled,
  +viewed: {Layout.sliced(Tensors.view_of(values),axis,start,count,step) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +total: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{total} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
    TiledLayout.slice(tiled,axis,start,count,step)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,old,lanes,position),values,contract_restored(~Logical,~original,logical,values,pieces,shape,fill,workers,old,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
              TiledLayout.gather_extents("slice",Tensors.Usable{},Layout.sliced(Tensors.view_of(_),axis,start,count,step),[],_,
                Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          %Equal.sym(Maybe<&2,Views.View>,Layout.sliced(Tensors.view_of(values),axis,start,count,step),Some{Views.View{axes,offset}},viewed) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
              TiledLayout.gather_extents("slice",Tensors.Usable{},_,[],values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          gathered_extents(~Logical,~original,logical,limits,"slice",values,axes,offset,[],Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},
            nonzero,target_nonzero,default_capacity,rank_room,target_rank,total,counted,fits,contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Views.axis_extents(axes),
            TiledLayout.slice(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},axis,start,count,step)),contract)

# The elements in reverse order along every listed axis: the tiled form of
# the view of the values read with those axes in reverse.
def flipped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+listed: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,tiled: Tiled.Tiled,
  +viewed: {Layout.flipped(Views.unique_axes(Layout.naturals(listed)) && Layout.all_below(listed,List.length(&2,U32,values)),Tensors.view_of(values)) ==
    Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
    ViewGather.Viewing{logical,axes,offset,Layout.reversed_flags(List.length(&2,U32,values),0,listed)},Views.axis_extents(axes),TiledLayout.flip(tiled,listed)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          +reversed = Layout.reversed_flags(List.length(&2,U32,values),0,listed)
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,old,lanes,position),values,contract_restored(~Logical,~original,logical,values,pieces,shape,fill,workers,old,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
              TiledLayout.gather_extents("flip",Tensors.Usable{},Layout.flipped(Views.unique_axes(Layout.naturals(listed)) && Layout.all_below(listed,List.length(&2,U32,_)),Tensors.view_of(_)),
                Layout.reversed_flags(List.length(&2,U32,_),0,listed),_,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          %Equal.sym(Maybe<&2,Views.View>,Layout.flipped(Views.unique_axes(Layout.naturals(listed)) && Layout.all_below(listed,List.length(&2,U32,values)),Tensors.view_of(values)),
              Some{Views.View{axes,offset}},viewed) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,reversed},Views.axis_extents(axes),
              TiledLayout.gather_extents("flip",Tensors.Usable{},_,reversed,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          gathered_extents(~Logical,~original,logical,limits,"flip",values,axes,offset,reversed,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},
            nonzero,target_nonzero,default_capacity,rank_room,target_rank,count,counted,fits,contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
            ViewGather.Viewing{logical,axes,offset,Layout.reversed_flags(List.length(&2,U32,values),0,listed)},Views.axis_extents(axes),
            TiledLayout.flip(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},listed)),contract)

# ---- Broadcasting ----

# The values repeated to the extents target under NumPy broadcasting: the
# tiled form of the stretched view.
def broadcast(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+target: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,tiled: Tiled.Tiled,
  +stretched: {Tensors.stretched(Tensors.view_of(values),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(target) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},target,
    TiledLayout.broadcast_to(tiled,target)):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          +same = ViewReads.stretch_extents(Tensors.view_of(values),target,axes,offset,stretched)
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(shape,old,lanes,position),values,contract_restored(~Logical,~original,logical,values,pieces,shape,fill,workers,old,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},target,
              TiledLayout.broadcast_view(Tensors.Usable{},Tensors.stretched(Tensors.view_of(_),target),_,target,workers,
                Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(values),target),Some{Views.View{axes,offset}},stretched) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},target,
              TiledLayout.broadcast_view(Tensors.Usable{},_,values,target,workers,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          %same :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},_,
              TiledView.gather("broadcast_to",Tensors.Usable{},Some{Views.View{axes,offset}},[],values,_,
                Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position}))
          gathered(~Logical,~original,logical,limits,"broadcast_to",values,axes,offset,[],Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},old,lanes,position},
            nonzero,{%Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,same) : {Shapes.has_zero(_) == False{} : Bool} target_nonzero : {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}},default_capacity,rank_room,
            {%Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,same) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} target_rank : {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
            count,{%Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,same) : {Tensors.count_of(_) == Some{count} : Maybe<&2,U32>} counted : {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>}},fits,contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},target,
            TiledLayout.broadcast_to(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},old,lanes,position},target)),contract)
