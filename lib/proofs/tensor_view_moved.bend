# A view of a tensor: the tensor a view of the cells of a tensor of the
# contract makes, when the runtime accepts the view, holds the viewed values.

import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_moved_sources.bend as Sources
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_then_gather.bend as Then
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_contract.bend as Contract
import ./tensor_shape_refinement.bend as ShapeLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./boolean_logic.bend as Logic

# ---- The view ----

# A shape of positive cells has no zero extent.
def none_zero(extents: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> {Shapes.has_zero(extents) == False{} : Bool}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      %Equal.sym(Bool,U32.is_zero(extent),False{},Arithmetic.word_positive_nonzero(extent,RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive))) :
        {(_ || Shapes.has_zero(rest)) == False{} : Bool}
      none_zero(rest,DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive))

# What a target piece of a view holds is what the view holds of it.
def kept_all(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,
  content: PiecesModel.Content<Element>,
  +holds: PiecesModel.ContentAll(~Element,~Then.Kept(~Element,~ViewGather.ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),Unit{}),content)) ->
  PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),
    ViewGather.Viewing{logical,axes,offset,flags}),content):
  match content:
    case PiecesModel.PieceContent{+buffer,+first,+rows}: holds
    case PiecesModel.Joined{+left,+right}:
      KernelGemm.Both{+left_holds,+right_holds} = holds
      KernelGemm.Both{kept_all(~Element,~Logical,~original,logical,axes,offset,flags,left,left_holds),kept_all(~Element,~Logical,~original,logical,axes,offset,flags,right,right_holds)}

# The pieces a view of a source laid as canonical pieces moves, then a map of
# every piece, hold what the map keeps of the values of the view.
def view_then(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),
  ~step: @+context: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: PiecesMap.Rule(~Element,~Later,~after,~step),
  ~Outer: @+later: Later -> @+viewing: ViewGather.Viewing<Logical> -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~lift: @+later: Later -> @+viewing: ViewGather.Viewing<Logical> -> @+first: U32 -> @+rows: U32 -> @+inner: Buffer.Content<Element> -> @+holds: Placed.PieceHolds(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,first,rows,inner) -> Outer(later,viewing,first,rows,step(later,first,rows,inner)),
  +later: Later,+logical: Logical,+limits: Machine.Limits,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+fill: Element,+workers: U32,
  +placed: Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Tensors.grid_of(source)),PiecesModel.pack(~Element,model)>,
  +facts: ViewGather.ViewFacts(limits,source,axes,offset,flags,Grid.piece_target())) ->
  Placed.PlacedModel<Element,Outer(later,ViewGather.Viewing{logical,axes,offset,flags}),Pieces.ranges_of_grid(Tensors.grid_of(Views.axis_extents(axes))),
    Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Later>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Later,~Tensors.view_gather(~Element),~after),
      Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,offset},flags),
      Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,offset,flags},later},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],Views.axis_extents(axes),fill,workers)>:
  Placed.PlacedModel{+source_model,+exact,+laid,+holds} = placed
  Held.Both{+source_ok,Held.Both{+target_ok,Held.Both{+fits,Held.Both{+count_le,Held.Both{+axes_fit,Held.Both{+blocks,Held.Both{+strides,
    Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+row_ok}}}}}}}}}} = facts
  +extents = Views.axis_extents(axes)
  +sgrid = Tensors.grid_of(source)
  +tgrid = Tensors.grid_of(extents)
  +row = Tasks.readings(source,sgrid,tgrid,Grid.coordinates(source,offset),axes,flags,0)
  +separate = Tasks.separable([row])
  +origin = Tensors.origin(source,offset)
  +context = {Tensors.Then{Tensors.Viewed{source,tgrid,extents,axes,offset,flags},later} : Tensors.Then<Tensors.Viewed,Later>}
  +all_facts = {Held.Both{source_ok,Held.Both{target_ok,Held.Both{fits,Held.Both{count_le,Held.Both{axes_fit,Held.Both{blocks,Held.Both{strides,
      Held.Both{components_ok,Held.Both{lists_positive,Held.Both{limits_at,row_ok}}}}}}}}}} : ViewGather.ViewFacts(limits,source,axes,offset,flags,Grid.piece_target())}
  %Equal.sym(Pieces.Pieces(~Element),PiecesModel.pack(~Element,model),PiecesModel.pack(~Element,source_model),exact) :
    Placed.PlacedModel<Element,Outer(later,ViewGather.Viewing{logical,axes,offset,flags}),Pieces.ranges_of_grid(tgrid),
      Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Later>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Later,~Tensors.view_gather(~Element),~after),
        Tasks.components(source,sgrid,tgrid,Views.View{axes,offset},flags),context,Tensors.move_rows(),0.0,Tensors.Source{_,Tensors.Ended{}},[origin],extents,fill,workers)>
  Components.moved_with_model(~Element,~Tensors.Then<Tensors.Viewed,Later>,
    ~Tensors.then_gather(~Element,~Tensors.Viewed,~Later,~Tensors.view_gather(~Element),~after),
    ~Then.then_model(~Element,~Tensors.Viewed,~Later,~after,~ViewGather.view_model(~Element)),
    ~Placed.SourceHolds(~Element,~Logical,~original,logical),~Outer(later,ViewGather.Viewing{logical,axes,offset,flags}),
    ~Then.ThenAgrees(~Element,~Tensors.Viewed,~Later,~ViewGather.ViewAgrees(~Element,limits,source,axes,offset,flags),later),
    ~Then.then_law(~Element,~Tensors.Viewed,~Later,~Tensors.view_gather(~Element),~ViewGather.view_model(~Element),~after,~step,~rule,
      ~Placed.SourceHolds(~Element,~Logical,~original,logical),~ViewGather.ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
      ~(later => first => rows => buffer => Outer(later,ViewGather.Viewing{logical,axes,offset,flags},first,rows,buffer)),~ViewGather.ViewAgrees(~Element,limits,source,axes,offset,flags),
      ~ViewGather.view_law(~Element,~Logical,~original,logical,limits,source,axes,offset,flags),
      ~(later => first => rows => inner => holds => lift(later,ViewGather.Viewing{logical,axes,offset,flags},first,rows,inner,holds)),later),
    separate,limits,[row],Tasks.target_rank([row]),[origin],extents,fill,context,
    Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},all_facts}}}}}},
    Tensors.move_rows(),0.0,workers,Moved.Models{source_model,Moved.NoModels{}},components_ok,{==},
    Held.Both{Held.Both{{==},Held.Both{{==},source_ok}},Held.Both{Sources.origin_when(separate,limits,row,source,offset,row_ok),Unit{}}},
    lists_positive,target_ok,limits_at,
    Held.Both{laid,Held.Both{Components.shape_ordered(limits,source,source_ok),Held.Both{Sources.empty_holds(~Element,~Logical,~original,logical,fill),Unit{}}}},
    Held.Both{Sources.good_of(~Element,~Logical,~original,logical,source_model,Held.Ended{},holds),Unit{}})

# The pieces a view of a source laid as canonical pieces moves hold the
# values of the view.
def view_placed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+fill: Element,+workers: U32,
  +placed: Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Tensors.grid_of(source)),PiecesModel.pack(~Element,model)>,
  +facts: ViewGather.ViewFacts(limits,source,axes,offset,flags,Grid.piece_target())) ->
  Placed.PlacedModel<Element,Then.Kept(~Element,~ViewGather.ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),Unit{}),Pieces.ranges_of_grid(Tensors.grid_of(Views.axis_extents(axes))),
    Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Element),~Tensors.unchanged(~Element)),
      Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,offset},flags),
      Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,offset,flags},Unit{}},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],Views.axis_extents(axes),fill,workers)>:
  view_then(~Element,~Logical,~original,~Unit,~Tensors.unchanged(~Element),~Then.kept_step(~Element),~Then.kept_rule(~Element),
    ~(later => viewing => first => rows => buffer => Then.Kept(~Element,~(each_first => each_rows => each_buffer => Placed.PieceHolds(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,each_first,each_rows,each_buffer)),later,first,rows,buffer)),
    ~(later => viewing => first => rows => inner => holds => Then.kept_lift(~Element,~(each_first => each_rows => each_buffer => Placed.PieceHolds(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,each_first,each_rows,each_buffer)),later,first,rows,inner,holds)),
    Unit{},logical,limits,model,source,axes,offset,flags,fill,workers,placed,facts)

# A target piece kept as the view gave it holds the viewed values.
def placed_view(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,
  +layout: Pieces.Ranges,-pieces: Pieces.Pieces(~Element),
  +placed: Placed.PlacedModel<Element,Then.Kept(~Element,~ViewGather.ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),Unit{}),layout,pieces>) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags}),
    layout,pieces>:
  Placed.PlacedModel{+model,+exact,+laid,+holds} = placed
  Placed.PlacedModel{model,exact,laid,kept_all(~Element,~Logical,~original,logical,axes,offset,flags,PiecesModel.content(~Element,model),holds)}

# The count of a counted shape is its number of cells.
def counted_cells(+shape: List<&2,U32>,+count: U32,+accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {U32.to_nat(Tensors.count_or_zero(shape)) == ShapeLaws.elements(shape) : Nat}:
  Equal.trans(Nat,U32.to_nat(Tensors.count_or_zero(shape)),U32.to_nat(count),ShapeLaws.elements(shape),
    Equal.cong(U32,Nat,U32.to_nat,Tensors.count_or_zero(shape),count,Global.whole_width(shape,count,accepted)),
    Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),Global.whole_counted(shape,count,accepted)))

# A shape that counts a nonzero number of cells has cells.
def counted_positive(+shape: List<&2,U32>,+count: U32,+accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}:
  %Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),Global.whole_counted(shape,count,accepted)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Arithmetic.word_nonzero_positive(count,nonzero)

# A view the runtime accepts of a counted source, whose target counts a
# nonzero number of cells, gives its gather every fact it needs.
def accepted_facts(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,
  +source_count: U32,+source_accepted: {Tensors.count_of(source) == Some{source_count} : Maybe<&2,U32>},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool}) ->
  ViewGather.ViewFacts(limits,source,axes,offset,flags,Grid.piece_target()):
  +extents = Views.axis_extents(axes)
  +length = Tensors.count_or_zero(source)
  +cells = counted_cells(source,source_count,source_accepted)
  +positive = counted_positive(extents,count,counted,nonzero)
  +fits_nonempty = {%none_zero(extents,positive) : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool}
    fits : {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}}
  +location = ViewBounds.location(axes,Views.last_coordinates(axes),U32.to_nat(offset))
  +source_positive = {%cells : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,location,U32.to_nat(length),Order.zero_le(location),ViewBounds.last_address(axes,offset,length,fits_nonempty)) :
    {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}}
  +count_le = {%cells : {Nat.is_le(U32.to_nat(length),_) == True{} : Bool}
    Order.reflexive(U32.to_nat(length)) : {Nat.is_le(U32.to_nat(length),ShapeLaws.elements(source)) == True{} : Bool}}
  ViewFacts.view_facts(limits,source,axes,offset,flags,Grid.piece_target(),
    Held.Both{StorageCount.counted_room(limits,source,Global.Counted{source_count,source_accepted},default_capacity),Held.Both{source_positive,source_rank}},
    Held.Both{StorageCount.counted_room(limits,extents,Global.Counted{count,counted},default_capacity),Held.Both{positive,target_rank}},
    fits_nonempty,count_le)

# The cells a view of a tensor of the contract that the runtime accepts
# gathers, with a target of nonzero cells, hold the viewed values.
def view_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool}) ->
  Global.Contract(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},
    Tensors.Tensor{Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Element),~Tensors.unchanged(~Element)),
    Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,offset},flags),
    Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,offset,flags},Unit{}},Tensors.move_rows(),0.0,
    Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],Views.axis_extents(axes),fill,workers),Views.axis_extents(axes),fill,Tensors.Usable{},workers}):
  +extents = Views.axis_extents(axes)
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  +source_counted = Global.contract_counted(~Element,~Logical,~original,logical,PiecesModel.pack(~Element,model),source,fill,workers,contract)
  +source_count = Global.count(source,source_counted)
  +facts = accepted_facts(limits,source,axes,offset,flags,source_count,Global.accepted(source,source_counted),count,counted,nonzero,default_capacity,source_rank,target_rank,fits)
  GridRanges.grid_contract(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,limits,extents,fill,workers,Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Element),~Tensors.unchanged(~Element)),
    Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
    Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},Unit{}},Tensors.move_rows(),0.0,
    Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),
    StorageCount.counted_room(limits,extents,Global.Counted{count,counted},default_capacity),
    counted_positive(extents,count,counted,nonzero),
    1n,GridRanges.Multiple{GridRanges.last_extent(extents),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(extents),1n),GridRanges.last_extent(extents),Order.times_one(GridRanges.last_extent(extents)))},
    Global.Counted{count,counted},
    placed_view(~Element,~Logical,~original,logical,axes,offset,flags,Pieces.ranges_of_grid(Tensors.grid_of(extents)),Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Element),~Tensors.unchanged(~Element)),
    Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
    Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},Unit{}},Tensors.move_rows(),0.0,
    Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),
      view_placed(~Element,~Logical,~original,logical,limits,model,source,axes,offset,flags,fill,workers,
        Sources.source_placed(~Element,~Logical,~original,logical,model,source,fill,workers,contract),facts)))

# A view of a tensor of the contract that the runtime accepts, with a target
# of nonzero cells, gives a ready tensor of its extents holding the viewed
# values.
def view_moved(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+name: String,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool}) ->
  Global.Shaped(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),ViewGather.Viewing{logical,axes,offset,flags},Views.axis_extents(axes),
    Tensors.moved(~Element,name,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers},axes,offset,flags)):
  +extents = Views.axis_extents(axes)
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  %Equal.sym(Bool,Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)),True{},fits) :
    Global.Shaped(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,extents,
      Tensors.moved_status(~Element,~Unit,~Tensors.unchanged(~Element),Unit{},name,Tensors.Usable{},_,Tensors.count_of(extents),PiecesModel.pack(~Element,model),source,extents,axes,offset,flags,fill,workers))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(extents),Some{count},counted) :
    Global.Shaped(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,extents,
      Tensors.moved_status(~Element,~Unit,~Tensors.unchanged(~Element),Unit{},name,Tensors.Usable{},True{},_,PiecesModel.pack(~Element,model),source,extents,axes,offset,flags,fill,workers))
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) :
    Global.Shaped(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,extents,
      Tensors.moved_counted(~Element,~Unit,~Tensors.unchanged(~Element),Unit{},_,PiecesModel.pack(~Element,model),source,extents,axes,offset,flags,fill,workers))
  Global.shaped(~Element,~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Element,~Logical,~original),viewing,extents,
    Tensors.Tensor{Tensors.moved_with(~Element,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Element,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Element),~Tensors.unchanged(~Element)),
      Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
      Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},Unit{}},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Element,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),extents,fill,Tensors.Usable{},workers},{==},{==},
    view_contract(~Element,~Logical,~original,logical,limits,model,source,axes,offset,flags,fill,workers,contract,default_capacity,source_rank,target_rank,count,counted,nonzero,fits))
