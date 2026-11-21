# Binary operations of tiled operands under NumPy broadcasting: every value of
# the result is the operation of the values the two operands read through
# their views stretched to the common extents.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_broadcast.bend as TiledBroadcast
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_map.bend as TiledMapLaws
import ./tensor_tiled_view.bend as TiledViewLaws
import ../tensor_tiled_view.bend as TiledView
import ./tensor_tiled_view_reads.bend as KeptReads
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_reads.bend as ViewReads
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./nat_order.bend as Order
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_representation.bend as Representation
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./tensor_stretched_digits.bend as Stretched
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_cell_word.bend as CellWords
import ./tensor_cell_room.bend as Rooms
import ./tensor_relane_values.bend as Values
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_view_moved.bend as ViewMoved
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./kernel_tile_refinement.bend as TileLaws
import ../tensor_tiled_cells.bend as Cells

# ---- Operands of one shape ----

# Operands arranged with their lanes on one axis and their groups innermost
# combine cell by cell.
def arranged_combined(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+values: List<&2,U32>,+name: String,+operations: F32,left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,
  +axis: Nat,+left_held: Relane.Arranged(~Left,~left_value,left,values,left_tiled,axis,Grid.Innermost{}),
  +right_held: Relane.Arranged(~Right,~right_value,right,values,right_tiled,axis,Grid.Innermost{})) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.arranged(left_tiled,axis,Grid.Innermost{}),Tiled.arranged(right_tiled,axis,Grid.Innermost{}))):
  KernelGemm.Both{+left_form,+left_contract} = left_held
  KernelGemm.Both{+right_form,+right_contract} = right_held
  +lanes = Grid.lanes_at(values,axis)
  %Equal.sym(Tiled.Tiled,Tiled.arranged(left_tiled,axis,Grid.Innermost{}),Tiled.Tiled{Tiled.cells_of(Tiled.arranged(left_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},left_form) :
    Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledMap.combine(~TiledMap.pairs(~operation),name,operations,_,Tiled.arranged(right_tiled,axis,Grid.Innermost{})))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(right_tiled,axis,Grid.Innermost{}),Tiled.Tiled{Tiled.cells_of(Tiled.arranged(right_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},right_form) :
    Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(left_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},_))
  TiledMapLaws.combine(~Left,~left_value,~Right,~right_value,~operation,left,right,values,name,operations,axis,lanes,lanes,Grid.Innermost{},Tiled.cells_of(Tiled.arranged(left_tiled,axis,Grid.Innermost{})),Tiled.cells_of(Tiled.arranged(right_tiled,axis,Grid.Innermost{})),left_contract,right_contract)

# The lane axis of packed tiled values of the contract lies inside them or
# just past them.
def packed_inside(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lx: Nat,+ll: U32,+lp: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp})) -> {Nat.is_le(lx,List.length(&2,U32,values)) == True{} : Bool}:
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  inside

def same_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+values: List<&2,U32>,+name: String,+operations: F32,+lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lx: Nat,+ll: U32,+lp: Grid.Position,+rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+rx: Nat,+rl: U32,+rp: Grid.Position,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp}),+right_contract: Public.Contract(~Right,~right_value,right,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp})) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp})):
  +inside = packed_inside(~Left,~left_value,left,values,lm,lc,lf,lw,lx,ll,lp,left_contract)
  arranged_combined(~Left,~left_value,~Right,~right_value,~operation,left,right,values,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp},lx,
    Relane.arranged_held(~Left,~left_value,left,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},lx,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,left_contract),
    Relane.arranged_held(~Right,~right_value,right,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp},lx,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,right_contract))

def same_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+values: List<&2,U32>,+name: String,+operations: F32,-left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,
  -right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,values,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,values,right_tiled)) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledBroadcast.same(~operation,name,operations,left_tiled,right_tiled)):
  Public.Packed{+lm,+lc,+lf,+lw,+lx,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+rx,+rl,+rp,+right_equation} = right_packed
  %left_equation : Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledBroadcast.same(~operation,name,operations,_,right_tiled))
  %right_equation : Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledBroadcast.same(~operation,name,operations,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},_))
  same_packed(~Left,~left_value,~Right,~right_value,~operation,left,right,values,name,operations,lm,lc,lf,lw,lx,ll,lp,rm,rc,rf,rw,rx,rl,rp,limits,default_capacity,nonempty,rank_room,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},left_tiled,left_equation) : Public.Contract(~Left,~left_value,left,values,_) left_contract :
      Public.Contract(~Left,~left_value,left,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp},right_tiled,right_equation) : Public.Contract(~Right,~right_value,right,values,_) right_contract :
      Public.Contract(~Right,~right_value,right,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp})})

# Two tiled operands of the contract and of one shape combine into the
# operation of their values at every index.
def same(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+values: List<&2,U32>,+name: String,+operations: F32,left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,values,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,values,right_tiled)) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledBroadcast.same(~operation,name,operations,left_tiled,right_tiled)):
  same_described(~Left,~left_value,~Right,~right_value,~operation,left,right,values,name,operations,left_tiled,Public.packed(~Left,~left_value,left,values,left_tiled,left_contract),
    right_tiled,Public.packed(~Right,~right_value,right,values,right_tiled,right_contract),limits,default_capacity,nonempty,rank_room,left_contract,right_contract)

# ---- Operands repeated to the common extents ----

# The view of values stretched to their own extents reads each value at its
# index.
def own_agree(~Logical: Data,~original: Logical -> U32 -> F32,+shape: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,+offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +logical: Logical,+other: ViewGather.Viewing<Logical>,+related: {other == ViewGather.Viewing{logical,axes,offset,[]} : ViewGather.Viewing<Logical>},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {original(logical,index) == ViewGather.viewed(~F32,~Logical,~original,other,index) : F32}:
  %Equal.sym(ViewGather.Viewing<Logical>,other,ViewGather.Viewing{logical,axes,offset,[]},related) : {original(logical,index) == ViewGather.viewed(~F32,~Logical,~original,_,index) : F32}
  Equal.cong(U32,F32,word => original(logical,word),index,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),U32.to_nat(index)) : U32),
    Equal.sym(U32,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),U32.to_nat(index)) : U32),index,ViewReads.stretched_own_address(shape,count,exact,axes,offset,stretched,index,inside)))

# Tiled values of the contract hold the values read through their view
# stretched to their own extents.
def own_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +limits: Machine.Limits,+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,shape,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},shape,tiled):
  KeptReads.contract_below(~Logical,~ViewGather.Viewing<Logical>,~original,~ViewGather.viewed(~F32,~Logical,~original),
    ~(other => viewing => {viewing == ViewGather.Viewing{other,axes,offset,[]} : ViewGather.Viewing<Logical>}),~ShapeLaws.elements(shape),
    ~own_agree(~Logical,~original,shape,count,exact,axes,offset,stretched),limits,logical,ViewGather.Viewing{logical,axes,offset,[]},{==},shape,positive,room,
    Order.reflexive(ShapeLaws.elements(shape)),tiled,contract)

# Tiled values of the contract repeated to the common extents hold the values
# read through their view stretched to those extents.
def repeated_contract(~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+logical: Logical,+values: List<&2,U32>,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,tiled: Tiled.Tiled,
  +classified: {Shapes.same_extents(values,shape) == same : Bool},+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +stretched: {Tensors.stretched(Tensors.view_of(values),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},shape,TiledBroadcast.repeated(same,shape,tiled)):
  match same:
    case True{}:
      +equal = ShapeLaws.same_extents(values,shape,classified)
      +exact = Count.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,counted))
      own_contract(~Logical,~original,logical,shape,axes,offset,count,exact,
        {%equal : {Tensors.stretched(Tensors.view_of(_),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>} stretched :
          {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}},
        limits,CountCompletion.positive_elements(shape,target_nonzero),StorageCount.counted_room(limits,shape,Global.Counted{count,counted},default_capacity),tiled,
        {%equal : Public.Contract(~Logical,~original,logical,_,tiled) contract : Public.Contract(~Logical,~original,logical,shape,tiled)})
    case False{}:
      TiledViewLaws.broadcast(~Logical,~original,logical,limits,values,shape,axes,offset,tiled,stretched,nonzero,target_nonzero,default_capacity,rank_room,target_rank,
        count,counted,fits,contract)

# Packed operands each repeated to the common extents combine into the
# operation of the values both read through their stretched views.
def values_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+name: String,+operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,+shape: List<&2,U32>,+lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lx: Nat,+ll: U32,+lp: Grid.Position,+rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+rx: Nat,+rl: U32,+rp: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+left_axes: List<&2,Views.Axis>,+left_offset: U32,+left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{left_axes,left_offset}} : Maybe<&2,Views.View>},+left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+left_rank: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+left_fits: {Views.fits(Views.View{left_axes,left_offset},Tensors.count_or_zero(ls)) == True{} : Bool},+right_axes: List<&2,Views.Axis>,+right_offset: U32,+right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{right_axes,right_offset}} : Maybe<&2,Views.View>},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+right_rank: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_fits: {Views.fits(Views.View{right_axes,right_offset},Tensors.count_or_zero(rs)) == True{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+count: U32,+counted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp}),+right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp})) ->
  Public.Contract(~CombineLeaf.Context<ViewGather.Viewing<Left>,ViewGather.Viewing<Right>>,~CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~ViewGather.Viewing<Right>,~operation,
    ~ViewGather.viewed(~F32,~Left,~left_value),~ViewGather.viewed(~F32,~Right,~right_value)),CombineLeaf.Context{ViewGather.Viewing{left,left_axes,left_offset,[]},ViewGather.Viewing{right,right_axes,right_offset,[]}},shape,
    TiledBroadcast.values(~operation,name,operations,ls,rs,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp})):
  same(~ViewGather.Viewing<Left>,~ViewGather.viewed(~F32,~Left,~left_value),~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),~operation,ViewGather.Viewing{left,left_axes,left_offset,[]},ViewGather.Viewing{right,right_axes,right_offset,[]},shape,name,operations,TiledBroadcast.repeated(Shapes.same_extents(ls,shape),shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp}),
    TiledBroadcast.repeated(Shapes.same_extents(rs,shape),shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp}),limits,default_capacity,CountCompletion.positive_elements(shape,shape_nonzero),shape_rank,
    repeated_contract(~Left,~left_value,Shapes.same_extents(ls,shape),left,ls,shape,left_axes,left_offset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lx,ll,lp},{==},limits,default_capacity,left_stretched,left_nonzero,
      shape_nonzero,left_rank,shape_rank,count,counted,left_fits,left_contract),
    repeated_contract(~Right,~right_value,Shapes.same_extents(rs,shape),right,rs,shape,right_axes,right_offset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},rx,rl,rp},{==},limits,default_capacity,right_stretched,right_nonzero,
      shape_nonzero,right_rank,shape_rank,count,counted,right_fits,right_contract))

# ---- A right operand whose stretched view keeps its lanes ----

# The target axes of the stretched view of the right read on its cells.
def mapped(+rs: List<&2,U32>,+rx: Nat,+raxes: List<&2,Views.Axis>,+roffset: U32) -> TiledView.Mapped:
  TiledView.mapped_axes(raxes,[],0n,rs,TiledView.lanes_of(rs,rx,Grid.Innermost{}),Grid.coordinates(rs,roffset))

# The view of the right cells: the other target axes, then the groups of the
# lane axis of the left.
def cell_axes(+rs: List<&2,U32>,+rx: Nat,+raxes: List<&2,Views.Axis>,+roffset: U32) -> List<&2,Views.Axis>:
  KeptReads.cell_axes_of(KeptReads.mapped_kept(mapped(rs,rx,raxes,roffset)),KeptReads.mapped_extent(mapped(rs,rx,raxes,roffset)),rs,rx,Grid.Innermost{})

def cell_offset(+rs: List<&2,U32>,+rx: Nat,+roffset: U32) -> U32:
  TiledView.cell_index(TiledView.lanes_of(rs,rx,Grid.Innermost{}),roffset)

# What the runtime checks before it reads the right cells through the view
# of the cells of the stretched view: the view keeps the lanes of the right,
# on its axis rx, on the lane axis lx of the left.
type Held<-rs: List<&2,U32>,-rx: Nat,-lx: Nat,-raxes: List<&2,Views.Axis>,-roffset: U32> is Data:
  Held{within: {Nat.is_lt(rx,List.length(&2,U32,rs)) == True{} : Bool},
    valid: {KeptReads.mapped_valid(mapped(rs,rx,raxes,roffset)) == True{} : Bool},
    lane_exact: {KeptReads.mapped_lane(mapped(rs,rx,raxes,roffset)) == Some{lx} : Maybe<&2,Nat>},
    inside_reach: {TiledView.all_inside(KeptReads.mapped_reach(mapped(rs,rx,raxes,roffset)),rs) == True{} : Bool},
    aligned: {U32.is_zero((TiledView.coordinate_at(Grid.coordinates(rs,roffset),rx) % 8 : U32)) == True{} : Bool}}

def held_inside(+rs: List<&2,U32>,+rx: Nat,+lx: Nat,+raxes: List<&2,Views.Axis>,+roffset: U32,+held: Held<rs,rx,lx,raxes,roffset>) ->
  {Nat.is_le(rx,List.length(&2,U32,rs)) == True{} : Bool}:
  Held{+within,+valid,+lane_exact,+inside_reach,+aligned} = held
  Bounds.strict_le(rx,List.length(&2,U32,rs),within)

# The operation of the left values and the right values read through their
# view stretched to the left extents.
def StretchedContract(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +ls: List<&2,U32>,+raxes: List<&2,Views.Axis>,+roffset: U32,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value)),
    CombineLeaf.Context{left,ViewGather.Viewing{right,raxes,roffset,[]}},ls,tiled)

# The tile the view of the right cells reads agrees with the right values
# read through their view at the lanes of the target cell.
def held_word(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+ls: List<&2,U32>,+rs: List<&2,U32>,+rx: Nat,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptReads.KeptView<limits,rs,rx,Grid.Innermost{},raxes,[],roffset,lx,extent,kept,kept_flags>,
  +extents: {Views.axis_extents(raxes) == ls : List<&2,U32>},+forward: {ViewGather.forward(kept_flags) == True{} : Bool},
  +word: U32,+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(ls,lx,Grid.Innermost{}))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(ls,lx,Grid.Innermost{})),Representation.cell_group(Public.geometry_of(ls,lx,Grid.Innermost{}),U32.to_nat(word)),
    ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,rx,Grid.Innermost{})},KeptReads.cell_axes_of(kept,extent,rs,rx,Grid.Innermost{}),
        TiledView.cell_index(TiledView.lanes_of(rs,rx,Grid.Innermost{}),roffset),[]},word),
    Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,raxes,roffset,[]},Public.geometry_of(ls,lx,Grid.Innermost{})},word)>:
  %extents : PaddedCells.Valid<Representation.lane_count(Public.geometry_of(_,lx,Grid.Innermost{})),Representation.cell_group(Public.geometry_of(_,lx,Grid.Innermost{}),U32.to_nat(word)),
    ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,rx,Grid.Innermost{})},KeptReads.cell_axes_of(kept,extent,rs,rx,Grid.Innermost{}),
        TiledView.cell_index(TiledView.lanes_of(rs,rx,Grid.Innermost{}),roffset),[]},word),
    Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,raxes,roffset,[]},Public.geometry_of(_,lx,Grid.Innermost{})},word)>
  %ViewGather.forward_viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Public.geometry_of(rs,rx,Grid.Innermost{})},
      KeptReads.cell_axes_of(kept,extent,rs,rx,Grid.Innermost{}),TiledView.cell_index(TiledView.lanes_of(rs,rx,Grid.Innermost{}),roffset),List.append(&2,Bool,kept_flags,[False{}]),
      ViewGather.forward_append(kept_flags,forward),word) :
    PaddedCells.Valid<Representation.lane_count(Public.geometry_of(Views.axis_extents(raxes),lx,Grid.Innermost{})),
      Representation.cell_group(Public.geometry_of(Views.axis_extents(raxes),lx,Grid.Innermost{}),U32.to_nat(word)),_,
      Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
        Representation.Lifted{ViewGather.Viewing{right,raxes,roffset,[]},Public.geometry_of(Views.axis_extents(raxes),lx,Grid.Innermost{})},word)>
  KeptReads.word_valid(~Right,~right_value,right,limits,rs,rx,Grid.Innermost{},raxes,[],roffset,lx,extent,kept,kept_flags,view,word,
    {%Equal.sym(List<&2,U32>,Views.axis_extents(raxes),ls,extents) : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(_,lx,Grid.Innermost{}))) == True{} : Bool} inside_word :
      {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(raxes),lx,Grid.Innermost{}))) == True{} : Bool}})

# Packed cells of the left and of the right whose stretched view keeps its
# lanes on the lane axis of the left combine into the operation of the left
# values and the right values read through that view.
def held_cells(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,+rx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+lm: PiecesModel.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,+rm: PiecesModel.Model<Tile.Tile8>,+rf: Tile.Tile8,+rw: U32,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+held: Held<rs,rx,lx,raxes,roffset>,
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),Tiled.cell_shape(ls,lx,Grid.Innermost{}),lf,Tensors.Usable{},lw},lx,Grid.lanes_at(ls,lx),Grid.Innermost{}}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),Tiled.cell_shape(rs,rx,Grid.Innermost{}),rf,Tensors.Usable{},rw},rx,Grid.lanes_at(rs,rx),Grid.Innermost{}})) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),Tiled.cell_shape(ls,lx,Grid.Innermost{}),lf,Tensors.Usable{},lw},lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),Tiled.cell_shape(rs,rx,Grid.Innermost{}),rf,Tensors.Usable{},rw},rx,Grid.lanes_at(rs,rx),Grid.Innermost{}},
    cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset))):
  Held{+within,+valid,+lane_exact,+inside_reach,+aligned} = held
  KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
  KernelGemm.Both{+right_global,Public.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  +left_laid = {Public.Laid{left_inside,left_shaped,left_restored,left_counted,left_pinned} :
    Public.Laid<Tiled.cell_shape(ls,lx,Grid.Innermost{}),lx,Grid.lanes_at(ls,lx),Grid.Innermost{},ls>}
  +found = mapped(rs,rx,raxes,roffset)
  +extents = ViewReads.stretch_extents(Tensors.view_of(rs),ls,raxes,roffset,right_stretched)
  +ls_nonzero = CountCompletion.positive_not_zero(ls,left_nonempty)
  +right_count = Global.count(rs,right_counted)
  +right_accepted = Global.accepted(rs,right_counted)
  +right_exact = Global.whole_counted(rs,right_count,right_accepted)
  +view_fits = {%Equal.sym(Maybe<&2,U32>,Tensors.count_of(rs),Some{right_count},right_accepted) : {Views.fits(Views.View{raxes,roffset},Tensors.known_count(_)) == True{} : Bool}
    Stretched.stretched_fits(rs,ls,right_count,right_exact,raxes,roffset,right_stretched,ls_nonzero) : {Views.fits(Views.View{raxes,roffset},Tensors.count_or_zero(rs)) == True{} : Bool}}
  +view = KeptReads.kept_view(limits,rs,rx,Grid.Innermost{},raxes,[],roffset,lx,KeptReads.mapped_extent(found),KeptReads.mapped_kept(found),KeptReads.mapped_flags(found),
    valid,lane_exact,{==},{==},{==},inside_reach,aligned,within,CountCompletion.positive_not_zero(rs,right_nonempty),
    {%Equal.sym(List<&2,U32>,Views.axis_extents(raxes),ls,extents) : {Shapes.has_zero(_) == False{} : Bool} ls_nonzero : {Shapes.has_zero(Views.axis_extents(raxes)) == False{} : Bool}},
    default_capacity,right_rank_room,right_counted,
    {%Equal.sym(List<&2,U32>,Views.axis_extents(raxes),ls,extents) : Global.Counted<_> left_counted : Global.Counted<Views.axis_extents(raxes)>},view_fits)
  +forward = KeptReads.mapped_forward(raxes,0n,rs,TiledView.lanes_of(rs,rx,Grid.Innermost{}),Grid.coordinates(rs,roffset))
  +cells_extents = {%extents : {Views.axis_extents(cell_axes(rs,rx,raxes,roffset)) == Tiled.cell_shape(_,lx,Grid.Innermost{}) : List<&2,U32>}
    KeptReads.cell_extents(limits,rs,rx,Grid.Innermost{},raxes,[],roffset,lx,KeptReads.mapped_extent(found),KeptReads.mapped_kept(found),KeptReads.mapped_flags(found),view) :
    {Views.axis_extents(cell_axes(rs,rx,raxes,roffset)) == Tiled.cell_shape(ls,lx,Grid.Innermost{}) : List<&2,U32>}}
  TiledMapLaws.viewed_models(~Left,~left_value,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,
    ~CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value)),~TiledMap.pairs(~operation),
    left,Representation.Lifted{right,Public.geometry_of(rs,rx,Grid.Innermost{})},CombineLeaf.Context{left,ViewGather.Viewing{right,raxes,roffset,[]}},ls,operations,lx,limits,default_capacity,lm,lf,lw,
    rm,Tiled.cell_shape(rs,rx,Grid.Innermost{}),rf,rw,rx,Grid.lanes_at(rs,rx),Grid.Innermost{},cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset),cells_extents,
    KeptReads.cell_fits(limits,rs,rx,Grid.Innermost{},raxes,[],roffset,lx,KeptReads.mapped_extent(found),KeptReads.mapped_kept(found),KeptReads.mapped_flags(found),view),
    left_nonempty,rank_room,left_laid,left_global,right_global,
    word => inside => TiledMapLaws.paired_valid(~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),~operation,left,
      ViewGather.Viewing{right,raxes,roffset,[]},Public.geometry_of(ls,lx,Grid.Innermost{}),word,
      ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
        ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,rx,Grid.Innermost{})},cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset),[]},word),
      held_word(~Right,~right_value,right,limits,ls,rs,rx,lx,raxes,roffset,KeptReads.mapped_extent(found),KeptReads.mapped_kept(found),KeptReads.mapped_flags(found),
        view,extents,forward,word,inside)))

# Both operands arranged with their lanes on the axes the view keeps and their
# groups innermost, read through the packed cells of their models.
def held_arranged(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,+rx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+held: Held<rs,rx,lx,raxes,roffset>,
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_arranged: Relane.Arranged(~Left,~left_value,left,ls,left_tiled,lx,Grid.Innermost{}),
  +right_arranged: Relane.Arranged(~Right,~right_value,right,rs,right_tiled,rx,Grid.Innermost{})) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.arranged(right_tiled,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset))):
  KernelGemm.Both{+left_form,+left_contract} = left_arranged
  KernelGemm.Both{+right_form,+right_contract} = right_arranged
  %Equal.sym(Tiled.Tiled,Tiled.arranged(left_tiled,lx,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(left_tiled,lx,Grid.Innermost{})),lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},left_form) :
    StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
      _,Tiled.arranged(right_tiled,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(right_tiled,rx,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(right_tiled,rx,Grid.Innermost{})),rx,Grid.lanes_at(rs,rx),Grid.Innermost{}},right_form) :
    StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(left_tiled,lx,Grid.Innermost{})),lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},_,cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))
  Relane.modeled(~Left,~left_value,
    (tiled => StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
      tiled,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(right_tiled,rx,Grid.Innermost{})),rx,Grid.lanes_at(rs,rx),Grid.Innermost{}},cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))),
    left,ls,Tiled.cells_of(Tiled.arranged(left_tiled,lx,Grid.Innermost{})),lx,Grid.lanes_at(ls,lx),Grid.Innermost{},left_contract,
    lm => lf => lw => left_modeled => Relane.modeled(~Right,~right_value,
      (tiled => StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),Tiled.cell_shape(ls,lx,Grid.Innermost{}),lf,Tensors.Usable{},lw},lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},tiled,
        cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))),
      right,rs,Tiled.cells_of(Tiled.arranged(right_tiled,rx,Grid.Innermost{})),rx,Grid.lanes_at(rs,rx),Grid.Innermost{},right_contract,
      rm => rf => rw => right_modeled => held_cells(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,rx,raxes,roffset,
        lm,lf,lw,rm,rf,rw,right_stretched,held,left_nonempty,right_nonempty,rank_room,right_rank_room,left_modeled,right_modeled)))

# Packed tiled operands, read through the cells of their models.
def held_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,+rx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,-left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,-right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+held: Held<rs,rx,lx,raxes,roffset>,
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.arranged(right_tiled,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset))):
  Public.Packed{+lm,+lc,+lf,+lw,+lo,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+ro,+rl,+rp,+right_equation} = right_packed
  %left_equation : StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.arranged(_,lx,Grid.Innermost{}),Tiled.arranged(right_tiled,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))
  %right_equation : StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lx,Grid.Innermost{}),Tiled.arranged(_,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)))
  held_arranged(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,rx,raxes,roffset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
    right_stretched,held,left_nonempty,right_nonempty,rank_room,right_rank_room,
    Relane.arranged_held(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lx,Grid.Innermost{},limits,default_capacity,inside_axis,left_nonempty,rank_room,
      {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) : Public.Contract(~Left,~left_value,left,ls,_)
      left_contract : Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})}),
    Relane.arranged_held(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},rx,Grid.Innermost{},limits,default_capacity,held_inside(rs,rx,lx,raxes,roffset,held),right_nonempty,
      right_rank_room,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) : Public.Contract(~Right,~right_value,right,rs,_)
      right_contract : Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})}))

# A left operand of the common extents and a right operand whose view
# stretched to them keeps the lanes of the right on the lane axis of the left:
# with their groups innermost, every left cell combines with the right cell
# the view of the cells reads, into the operation of the left values and the
# right values read through the stretched view.
def held(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,+rx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+held: Held<rs,rx,lx,raxes,roffset>,
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.arranged(right_tiled,rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset))):
  held_described(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,rx,raxes,roffset,
    left_tiled,Public.packed(~Left,~left_value,left,ls,left_tiled,left_contract),right_tiled,Public.packed(~Right,~right_value,right,rs,right_tiled,right_contract),
    right_stretched,held,left_nonempty,right_nonempty,rank_room,right_rank_room,inside_axis,left_contract,right_contract)

# ---- A right operand repeated along the lane axis of the left ----

# The view of the right cells: the stretched view without the lane axis of the
# left, then the groups of the left at stride 0.
def spread_axes_of(+ls: List<&2,U32>,+lx: Nat,+raxes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  TiledBroadcast.spread_axes(Nat.is_lt(lx,List.length(&2,U32,ls)),raxes,lx,Tiled.groups(Grid.lanes_at(ls,lx)))

# The extents of axes without one are the extents without it.
def removed_extents(axes: List<&2,Views.Axis>,axis: Nat) -> {Views.axis_extents(Views.removed_axis(axes,axis)) == Grid.removed(Views.axis_extents(axes),axis) : List<&2,U32>}:
  match axes axis:
    case Nil{} axis: {==}
    case Con{Views.Axis{+extent,+stride},+rest} 0n: {==}
    case Con{Views.Axis{+extent,+stride},+rest} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => extent <> extents,Views.axis_extents(Views.removed_axis(rest,next)),Grid.removed(Views.axis_extents(rest),next),
        removed_extents(rest,next))

# One more axis at stride 0 adds nothing to the sum of digits below the
# extents, whatever its digit.
def appended_sum(axes: List<&2,Views.Axis>,digits: List<&2,Nat>,+extent: U32,+last: Nat,+below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits)) ->
  {KeptReads.digit_sum(List.append(&2,Views.Axis,axes,[Views.Axis{extent,0}]),[],List.append(&2,Nat,digits,[last])) == KeptReads.digit_sum(axes,[],digits) : Nat}:
  match axes digits:
    case Con{Views.Axis{+size,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(digit,U32.to_nat(stride)),sum),
        KeptReads.digit_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,0}]),[],List.append(&2,Nat,more,[last])),KeptReads.digit_sum(rest,[],more),
        appended_sum(rest,more,extent,last,later))
    case Nil{} Nil{}:
      %Equal.sym(Nat,Nat.mul(last,0n),0n,RowMajor.multiply_zero_right(last)) : {Nat.add(_,0n) == 0n : Nat}
      {==}
    case Nil{} Con{+digit,+more}:
      Empty.absurd({KeptReads.digit_sum(List.append(&2,Views.Axis,Nil{},[Views.Axis{extent,0}]),[],List.append(&2,Nat,Con{digit,more},[last])) ==
        KeptReads.digit_sum(Nil{},[],Con{digit,more}) : Nat},below)
    case Con{Views.Axis{+size,+stride},+rest} Nil{}:
      Empty.absurd({KeptReads.digit_sum(List.append(&2,Views.Axis,Con{Views.Axis{size,stride},rest},[Views.Axis{extent,0}]),[],List.append(&2,Nat,Nil{},[last])) ==
        KeptReads.digit_sum(Con{Views.Axis{size,stride},rest},[],Nil{}) : Nat},below)

# An axis at stride 0 adds nothing to the sum of digits with a digit put at it.
def removed_sum(axes: List<&2,Views.Axis>,axis: Nat,others: List<&2,Nat>,+channel: Nat,+zero: {Views.stride_at(axes,axis) == 0 : U32},
  +below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),axis),others)) ->
  {KeptReads.digit_sum(axes,[],Coordinates.inserted_digit(axis,others,channel)) == KeptReads.digit_sum(Views.removed_axis(axes,axis),[],others) : Nat}:
  match axes axis others:
    case Nil{} axis others: {==}
    case Con{Views.Axis{+extent,+stride},+rest} 0n +others:
      %Equal.sym(U32,stride,0,zero) : {Nat.add(Nat.mul(channel,U32.to_nat(_)),KeptReads.digit_sum(rest,[],others)) == KeptReads.digit_sum(rest,[],others) : Nat}
      %Equal.sym(Nat,Nat.mul(channel,0n),0n,RowMajor.multiply_zero_right(channel)) : {Nat.add(_,KeptReads.digit_sum(rest,[],others)) == KeptReads.digit_sum(rest,[],others) : Nat}
      {==}
    case Con{Views.Axis{+extent,+stride},+rest} 1n+ +next Nil{}:
      Empty.absurd({KeptReads.digit_sum(Con{Views.Axis{extent,stride},rest},[],Coordinates.inserted_digit(1n+next,Nil{},channel)) ==
        KeptReads.digit_sum(Views.removed_axis(Con{Views.Axis{extent,stride},rest},1n+next),[],Nil{}) : Nat},below)
    case Con{Views.Axis{+extent,+stride},+rest} 1n+ +next Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(digit,U32.to_nat(stride)),sum),
        KeptReads.digit_sum(rest,[],Coordinates.inserted_digit(next,more,channel)),KeptReads.digit_sum(Views.removed_axis(rest,next),[],more),
        removed_sum(rest,next,more,channel,zero,later))

# Nor does one more axis at stride 0 reach farther.
def appended_span(axes: List<&2,Views.Axis>,+extent: U32) -> {ViewBounds.last_span(List.append(&2,Views.Axis,axes,[Views.Axis{extent,0}])) == ViewBounds.last_span(axes) : Nat}:
  match axes:
    case Nil{}:
      %Equal.sym(Nat,Nat.mul(U32.to_nat((extent - 1 : U32)),0n),0n,RowMajor.multiply_zero_right(U32.to_nat((extent - 1 : U32)))) : {Nat.add(_,0n) == 0n : Nat}
      {==}
    case Con{Views.Axis{+size,+stride},+rest}:
      Equal.cong(Nat,Nat,span => Nat.add(Nat.mul(U32.to_nat((size - 1 : U32)),U32.to_nat(stride)),span),
        ViewBounds.last_span(List.append(&2,Views.Axis,rest,[Views.Axis{extent,0}])),ViewBounds.last_span(rest),appended_span(rest,extent))

# Nor does an axis at stride 0 removed.
def removed_span(axes: List<&2,Views.Axis>,axis: Nat,+zero: {Views.stride_at(axes,axis) == 0 : U32}) ->
  {ViewBounds.last_span(axes) == ViewBounds.last_span(Views.removed_axis(axes,axis)) : Nat}:
  match axes axis:
    case Nil{} axis: {==}
    case Con{Views.Axis{+extent,+stride},+rest} 0n:
      %Equal.sym(U32,stride,0,zero) : {Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(_)),ViewBounds.last_span(rest)) == ViewBounds.last_span(rest) : Nat}
      %Equal.sym(Nat,Nat.mul(U32.to_nat((extent - 1 : U32)),0n),0n,RowMajor.multiply_zero_right(U32.to_nat((extent - 1 : U32)))) :
        {Nat.add(_,ViewBounds.last_span(rest)) == ViewBounds.last_span(rest) : Nat}
      {==}
    case Con{Views.Axis{+extent,+stride},+rest} 1n+ +next:
      Equal.cong(Nat,Nat,span => Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),span),
        ViewBounds.last_span(rest),ViewBounds.last_span(Views.removed_axis(rest,next)),removed_span(rest,next,zero))

# The spread view has the extents of the cells with their lanes on the axis.
def spread_extents(within: Bool,+axes: List<&2,Views.Axis>,+axis: Nat,+groups: U32,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,Views.axis_extents(axes))) == within : Bool}) ->
  {Views.axis_extents(TiledBroadcast.spread_axes(within,axes,axis,groups)) == Grid.placed(within,Grid.Innermost{},Grid.removed(Views.axis_extents(axes),axis),groups) : List<&2,U32>}:
  match within:
    case True{}:
      %removed_extents(axes,axis) :
        {Views.axis_extents(List.append(&2,Views.Axis,Views.removed_axis(axes,axis),[Views.Axis{groups,0}])) == List.append(&2,U32,_,[groups]) : List<&2,U32>}
      KeptReads.extents_appended(Views.removed_axis(axes,axis),groups,0)
    case False{}:
      Equal.sym(List<&2,U32>,Grid.removed(Views.axis_extents(axes),axis),Views.axis_extents(axes),Public.removed_past(Views.axis_extents(axes),axis,classified))

# It sums the digits of a cell as the stretched view sums the logical digits
# of any channel of the cell.
def spread_sum(within: Bool,+axes: List<&2,Views.Axis>,+axis: Nat,+groups: U32,+others: List<&2,Nat>,+group: Nat,+channel: Nat,
  +zero: {Views.stride_at(axes,axis) == 0 : U32},+below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),axis),others)) ->
  {KeptReads.digit_sum(TiledBroadcast.spread_axes(within,axes,axis,groups),[],Coordinates.placed_digits(within,Grid.Innermost{},others,group)) ==
    KeptReads.digit_sum(axes,[],Coordinates.logical_digits(within,axis,others,channel)) : Nat}:
  match within:
    case True{}:
      Equal.trans(Nat,KeptReads.digit_sum(List.append(&2,Views.Axis,Views.removed_axis(axes,axis),[Views.Axis{groups,0}]),[],List.append(&2,Nat,others,[group])),
        KeptReads.digit_sum(Views.removed_axis(axes,axis),[],others),KeptReads.digit_sum(axes,[],Coordinates.inserted_digit(axis,others,channel)),
        appended_sum(Views.removed_axis(axes,axis),others,groups,group,
          {%Equal.sym(List<&2,U32>,Views.axis_extents(Views.removed_axis(axes,axis)),Grid.removed(Views.axis_extents(axes),axis),removed_extents(axes,axis)) :
            Coordinates.DigitsBelow(_,others) below : Coordinates.DigitsBelow(Views.axis_extents(Views.removed_axis(axes,axis)),others)}),
        Equal.sym(Nat,KeptReads.digit_sum(axes,[],Coordinates.inserted_digit(axis,others,channel)),KeptReads.digit_sum(Views.removed_axis(axes,axis),[],others),
          removed_sum(axes,axis,others,channel,zero,below)))
    case False{}: {==}

# It reaches as far as the stretched view.
def spread_span(within: Bool,+axes: List<&2,Views.Axis>,+axis: Nat,+groups: U32,+zero: {Views.stride_at(axes,axis) == 0 : U32}) ->
  {ViewBounds.last_span(TiledBroadcast.spread_axes(within,axes,axis,groups)) == ViewBounds.last_span(axes) : Nat}:
  match within:
    case True{}:
      Equal.trans(Nat,ViewBounds.last_span(List.append(&2,Views.Axis,Views.removed_axis(axes,axis),[Views.Axis{groups,0}])),ViewBounds.last_span(Views.removed_axis(axes,axis)),
        ViewBounds.last_span(axes),appended_span(Views.removed_axis(axes,axis),groups),
        Equal.sym(Nat,ViewBounds.last_span(axes),ViewBounds.last_span(Views.removed_axis(axes,axis)),removed_span(axes,axis,zero)))
    case False{}: {==}

# So it fits where the stretched view fits.
def spread_fits(+within: Bool,+axes: List<&2,Views.Axis>,+axis: Nat,+groups: U32,+offset: U32,+length: U32,+zero: {Views.stride_at(axes,axis) == 0 : U32},
  +nonempty: {Shapes.has_zero(Views.axis_extents(TiledBroadcast.spread_axes(within,axes,axis,groups))) == False{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}) ->
  {Views.fits(Views.View{TiledBroadcast.spread_axes(within,axes,axis,groups),offset},length) == True{} : Bool}:
  +bound = {%ViewBounds.last_location(axes,U32.to_nat(offset)) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} ViewBounds.last_address(axes,offset,length,fits) :
    {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),U32.to_nat(length)) == True{} : Bool}}
  ViewBounds.fits_span(TiledBroadcast.spread_axes(within,axes,axis,groups),offset,length,nonempty,
    %Equal.sym(Nat,ViewBounds.last_span(TiledBroadcast.spread_axes(within,axes,axis,groups)),ViewBounds.last_span(axes),spread_span(within,axes,axis,groups,zero)) :
      {Nat.is_lt(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool} bound)

# A lane below the extent of a target cell: the value at the address the spread
# view reads is the value the stretched view reads at the lane.
def spread_hit(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+axis: Nat,
  +zero: {Views.stride_at(axes,axis) == 0 : U32},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: U32,+length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +others: List<&2,Nat>,+group: Nat,+others_below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),axis),others),
  +value: F32,+address: U32,+value_exact: {value == right_value(right,address) : F32},
  +address_exact: {U32.to_nat(address) == Nat.add(U32.to_nat(offset),KeptReads.digit_sum(spread_axes_of(Views.axis_extents(axes),axis,axes),[],
    Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,Views.axis_extents(axes))),Grid.Innermost{},others,group))) : Nat},
  +lane: Nat,+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(Views.axis_extents(axes),axis))) == True{} : Bool}) ->
  {value == ViewGather.viewed(~F32,~Right,~right_value,ViewGather.Viewing{right,axes,offset,[]},
    U32.from_nat(ShapeLaws.digits_value(Views.axis_extents(axes),
      Coordinates.logical_digits(Nat.is_lt(axis,List.length(&2,U32,Views.axis_extents(axes))),axis,others,Nat.add(lane,Nat.mul(group,8n)))))) : F32}:
  +target = Views.axis_extents(axes)
  +within = Nat.is_lt(axis,List.length(&2,U32,target))
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +digits = Coordinates.logical_digits(within,axis,others,channel)
  +digits_below = Values.logical_below(within,target,axis,others,channel,{==},others_below,inside)
  +index_value = ShapeLaws.digits_value(target,digits)
  +index = U32.from_nat(index_value)
  +index_below = Coordinates.digits_below(target,digits,digits_below)
  +index_exact = FullRange.roundtrip(index_value,Machine.capacity(limits),
    Order.transitive(index_value,ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(index_value,ShapeLaws.elements(target),index_below),target_room))
  +index_inside = {%Equal.sym(Nat,U32.to_nat(index),index_value,index_exact) : {Nat.is_lt(_,ShapeLaws.elements(target)) == True{} : Bool} index_below :
    {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(target)) == True{} : Bool}}
  +target_address = (Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(target,ViewPart.view_steps(Views.mirrored_axes(axes,[])),U32.to_nat(index)) : U32)
  +target_read = ViewReads.address_at(axes,[],offset,U32.to_nat(index),length,Machine.capacity(limits),Limits.capacity_fits(limits),target_room,length_room,fits,index_inside)
  +target_sum = Equal.trans(Nat,ViewReads.view_sum(axes,[],U32.to_nat(index)),ViewReads.view_sum(axes,[],index_value),KeptReads.digit_sum(axes,[],digits),
    Equal.cong(Nat,Nat,number => ViewReads.view_sum(axes,[],number),U32.to_nat(index),index_value,index_exact),KeptReads.view_sum_digits(axes,[],digits,digits_below))
  +target_exact = Equal.trans(Nat,U32.to_nat(target_address),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],U32.to_nat(index))),
    Nat.add(U32.to_nat(offset),KeptReads.digit_sum(axes,[],digits)),ViewGather.read_exact_of(offset,axes,[],U32.to_nat(index),target_address,length,target_read),
    Equal.cong(Nat,Nat,sum => Nat.add(U32.to_nat(offset),sum),ViewReads.view_sum(axes,[],U32.to_nat(index)),KeptReads.digit_sum(axes,[],digits),target_sum))
  +spread = spread_axes_of(target,axis,axes)
  +placed = Coordinates.placed_digits(within,Grid.Innermost{},others,group)
  +sums = spread_sum(within,axes,axis,Tiled.groups(Grid.lanes_at(target,axis)),others,group,channel,zero,others_below)
  +same_nat = Equal.trans(Nat,U32.to_nat(address),Nat.add(U32.to_nat(offset),KeptReads.digit_sum(spread,[],placed)),U32.to_nat(target_address),address_exact,
    Equal.trans(Nat,Nat.add(U32.to_nat(offset),KeptReads.digit_sum(spread,[],placed)),Nat.add(U32.to_nat(offset),KeptReads.digit_sum(axes,[],digits)),U32.to_nat(target_address),
      Equal.cong(Nat,Nat,sum => Nat.add(U32.to_nat(offset),sum),KeptReads.digit_sum(spread,[],placed),KeptReads.digit_sum(axes,[],digits),sums),
      Equal.sym(Nat,U32.to_nat(target_address),Nat.add(U32.to_nat(offset),KeptReads.digit_sum(axes,[],digits)),target_exact)))
  +same = Equal.trans(U32,address,U32.from_nat(U32.to_nat(address)),target_address,Equal.sym(U32,U32.from_nat(U32.to_nat(address)),address,Arithmetic.native_roundtrip(address)),
    Equal.trans(U32,U32.from_nat(U32.to_nat(address)),U32.from_nat(U32.to_nat(target_address)),target_address,
      Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat(address),U32.to_nat(target_address),same_nat),Arithmetic.native_roundtrip(target_address)))
  Equal.trans(F32,value,right_value(right,address),right_value(right,target_address),value_exact,Equal.cong(U32,F32,place => right_value(right,place),address,target_address,same))

# A lane of a target cell: the value the spread view reads holds where the
# lane lies below the extent, +0 past it.
def spread_lane(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+axis: Nat,
  +zero: {Views.stride_at(axes,axis) == 0 : U32},+inside_axis: {Nat.is_le(axis,List.length(&2,U32,Views.axis_extents(axes))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(Views.axis_extents(axes),axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: U32,+length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},+word: U32,+others: List<&2,Nat>,+group: Nat,
  +others_below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),axis),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Views.axis_extents(axes),axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Views.axis_extents(axes),axis,Grid.Innermost{}),
    Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,Views.axis_extents(axes))),Grid.Innermost{},others,group)) : Nat},
  +value: F32,+address: U32,+value_exact: {value == right_value(right,address) : F32},
  +address_exact: {U32.to_nat(address) == Nat.add(U32.to_nat(offset),KeptReads.digit_sum(spread_axes_of(Views.axis_extents(axes),axis,axes),[],
    Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,Views.axis_extents(axes))),Grid.Innermost{},others,group))) : Nat},
  +lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {PaddedCells.kept(U32.to_nat(Grid.lanes_at(Views.axis_extents(axes),axis)),group,lane,value) ==
    TileLaws.select_lane(lane,Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,axes,offset,[]},Public.geometry_of(Views.axis_extents(axes),axis,Grid.Innermost{})},word)) : F32}:
  +target = Views.axis_extents(axes)
  +within = Nat.is_lt(axis,List.length(&2,U32,target))
  +lanes = U32.to_nat(Grid.lanes_at(target,axis))
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +viewing = {ViewGather.Viewing{right,axes,offset,[]} : ViewGather.Viewing<Right>}
  +read = ViewGather.viewed(~F32,~Right,~right_value,viewing,U32.from_nat(ShapeLaws.digits_value(target,Coordinates.logical_digits(within,axis,others,channel))))
  +outer = TileLaws.select_lane(lane,Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
    Representation.Lifted{viewing,Public.geometry_of(target,axis,Grid.Innermost{})},word))
  +target_lane = Coordinates.lane_values(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),viewing,limits,target,axis,Grid.Innermost{},within,others,group,word,
    {==},inside_axis,removed_room,others_below,group_inside,exact,lane,small)
  Equal.trans(F32,PaddedCells.kept(lanes,group,lane,value),PaddedCells.kept(lanes,group,lane,read),outer,
    PaddedCells.kept_agree(lanes,group,lane,value,read,Nat.is_lt(channel,lanes),{==},
      inside => spread_hit(~Right,~right_value,right,limits,axes,offset,axis,zero,target_room,length,length_room,fits,others,group,others_below,value,address,value_exact,
        address_exact,lane,inside)),
    Equal.sym(F32,outer,PaddedCells.kept(lanes,group,lane,read),target_lane))

# A target cell at a word split into the digits of the other axes and its
# group, the lanes of the right cell the spread view reads at the word.
def spread_split(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+rs: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+axis: Nat,+zero: {Views.stride_at(axes,axis) == 0 : U32},
  +inside_axis: {Nat.is_le(axis,List.length(&2,U32,Views.axis_extents(axes))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(Views.axis_extents(axes),axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: U32,+length_exact: {U32.to_nat(length) == ShapeLaws.elements(rs) : Nat},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +spread_fits: {Views.fits_nonempty(False{},Views.View{spread_axes_of(Views.axis_extents(axes),axis,axes),offset},length) == True{} : Bool},
  +word: U32,+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),axis,Grid.Innermost{}))) == True{} : Bool},
  +split: CellWords.CellWord<Views.axis_extents(axes),axis,word>) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(Views.axis_extents(axes),axis,Grid.Innermost{})),
    Representation.cell_group(Public.geometry_of(Views.axis_extents(axes),axis,Grid.Innermost{}),U32.to_nat(word)),
    Tile.filled(Tile.lane0(ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},spread_axes_of(Views.axis_extents(axes),axis,axes),offset,[]},word))),
    Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,axes,offset,[]},Public.geometry_of(Views.axis_extents(axes),axis,Grid.Innermost{})},word)>:
  CellWords.CellWord{+others,+group,+others_below,+group_inside,+split_exact,+group_same,+lanes_same,+split_below} = split
  +target = Views.axis_extents(axes)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +spread = spread_axes_of(target,axis,axes)
  +geometry = Public.geometry_of(target,axis,Grid.Innermost{})
  +number = U32.to_nat(word)
  +placed = Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,target)),Grid.Innermost{},others,group)
  +extents = {spread_extents(Nat.is_lt(axis,List.length(&2,U32,target)),axes,axis,Tiled.groups(Grid.lanes_at(target,axis)),{==}) :
    {Views.axis_extents(spread) == Tiled.cell_shape(target,axis,Grid.Innermost{}) : List<&2,U32>}}
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(spread),Tiled.cell_shape(target,axis,Grid.Innermost{}),extents)
  +word_digits = {%back : {number == ShapeLaws.digits_value(_,placed) : Nat} split_exact : {number == ShapeLaws.digits_value(Views.axis_extents(spread),placed) : Nat}}
  +placed_below = {%back : Coordinates.DigitsBelow(_,placed) split_below : Coordinates.DigitsBelow(Views.axis_extents(spread),placed)}
  +spread_room = {%back : {Nat.is_le(ShapeLaws.elements(_),capacity) == True{} : Bool} cells_room : {Nat.is_le(ShapeLaws.elements(Views.axis_extents(spread)),capacity) == True{} : Bool}}
  +spread_inside = {%back : {Nat.is_lt(number,ShapeLaws.elements(_)) == True{} : Bool} inside_word : {Nat.is_lt(number,ShapeLaws.elements(Views.axis_extents(spread))) == True{} : Bool}}
  +length_room = {%Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(rs),length_exact) : {Nat.is_le(_,capacity) == True{} : Bool} room : {Nat.is_le(U32.to_nat(length),capacity) == True{} : Bool}}
  +address = (Views.mirrored_offset(spread,[],offset) + Box.stepped_sum(Views.axis_extents(spread),ViewPart.view_steps(Views.mirrored_axes(spread,[])),number) : U32)
  +read = ViewReads.address_at(spread,[],offset,number,length,Machine.capacity(limits),Limits.capacity_fits(limits),spread_room,length_room,spread_fits,spread_inside)
  +read_exact = ViewGather.read_exact_of(offset,spread,[],number,address,length,read)
  +read_below = {%Equal.sym(Nat,U32.to_nat(address),Nat.add(U32.to_nat(offset),ViewReads.view_sum(spread,[],number)),read_exact) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
    ViewGather.read_below_of(offset,spread,[],number,address,length,read) : {Nat.is_lt(U32.to_nat(address),U32.to_nat(length)) == True{} : Bool}}
  +address_inside = {%length_exact : {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool} read_below : {Nat.is_lt(U32.to_nat(address),ShapeLaws.elements(rs)) == True{} : Bool}}
  +value = Tile.lane0(Representation.lifted(~Right,~right_value,Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},address))
  +value_exact = {Coordinates.unit_lane(~Right,~right_value,right,limits,rs,position,address,room,address_inside,0n,{==}) : {value == right_value(right,address) : F32}}
  +address_sum = Equal.trans(Nat,U32.to_nat(address),Nat.add(U32.to_nat(offset),ViewReads.view_sum(spread,[],number)),Nat.add(U32.to_nat(offset),KeptReads.digit_sum(spread,[],placed)),
    read_exact,Equal.cong(Nat,Nat,sum => Nat.add(U32.to_nat(offset),sum),ViewReads.view_sum(spread,[],number),KeptReads.digit_sum(spread,[],placed),
      Equal.trans(Nat,ViewReads.view_sum(spread,[],number),ViewReads.view_sum(spread,[],ShapeLaws.digits_value(Views.axis_extents(spread),placed)),KeptReads.digit_sum(spread,[],placed),
        Equal.cong(Nat,Nat,index => ViewReads.view_sum(spread,[],index),number,ShapeLaws.digits_value(Views.axis_extents(spread),placed),word_digits),
        KeptReads.view_sum_digits(spread,[],placed,placed_below))))
  +lanes = U32.to_nat(Grid.lanes_at(target,axis))
  +outer = Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),Representation.Lifted{ViewGather.Viewing{right,axes,offset,[]},geometry},word)
  %Equal.sym(Nat,Representation.lane_count(geometry),lanes,lanes_same) : PaddedCells.Valid<_,Representation.cell_group(geometry,number),Tile.filled(value),outer>
  %Equal.sym(Nat,Representation.cell_group(geometry,number),group,group_same) : PaddedCells.Valid<lanes,_,Tile.filled(value),outer>
  PaddedCells.Valid{
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,0n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,1n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,2n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,3n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,4n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,5n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,6n,{==}),
    spread_lane(~Right,~right_value,right,limits,axes,offset,axis,zero,inside_axis,target_room,removed_room,length,length_room,fits,word,others,group,others_below,group_inside,
      split_exact,value,address,value_exact,address_sum,7n,{==})}

# The same at the extents of the left, which the stretched view reads.
def spread_word(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+ls: List<&2,U32>,+rs: List<&2,U32>,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+axis: Nat,+extents: {Views.axis_extents(axes) == ls : List<&2,U32>},+zero: {Views.stride_at(axes,axis) == 0 : U32},
  +inside_axis: {Nat.is_le(axis,List.length(&2,U32,Views.axis_extents(axes))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(Views.axis_extents(axes),axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: U32,+length_exact: {U32.to_nat(length) == ShapeLaws.elements(rs) : Nat},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +spread_fits: {Views.fits_nonempty(False{},Views.View{spread_axes_of(Views.axis_extents(axes),axis,axes),offset},length) == True{} : Bool},
  +word: U32,+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(ls,axis,Grid.Innermost{}))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(ls,axis,Grid.Innermost{})),Representation.cell_group(Public.geometry_of(ls,axis,Grid.Innermost{}),U32.to_nat(word)),
    Tile.filled(Tile.lane0(ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},spread_axes_of(ls,axis,axes),offset,[]},word))),
    Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,axes,offset,[]},Public.geometry_of(ls,axis,Grid.Innermost{})},word)>:
  +inside_target = {%Equal.sym(List<&2,U32>,Views.axis_extents(axes),ls,extents) : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(_,axis,Grid.Innermost{}))) == True{} : Bool}
    inside_word : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),axis,Grid.Innermost{}))) == True{} : Bool}}
  %extents : PaddedCells.Valid<Representation.lane_count(Public.geometry_of(_,axis,Grid.Innermost{})),Representation.cell_group(Public.geometry_of(_,axis,Grid.Innermost{}),U32.to_nat(word)),
    Tile.filled(Tile.lane0(ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},spread_axes_of(_,axis,axes),offset,[]},word))),
    Representation.lifted(~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),
      Representation.Lifted{ViewGather.Viewing{right,axes,offset,[]},Public.geometry_of(_,axis,Grid.Innermost{})},word)>
  spread_split(~Right,~right_value,right,limits,rs,position,axes,offset,axis,zero,inside_axis,target_room,removed_room,cells_room,room,length,length_exact,fits,spread_fits,
    word,inside_target,CellWords.cell_word(limits,Views.axis_extents(axes),axis,word,inside_axis,removed_room,inside_target))

# A left cell spread with the first lane of a tile agrees with the combined
# reader when that lane agrees with the right reader at every lane of the
# cell.
def spread_valid(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +geometry: Cells.Cells,+word: U32,tile: Tile.Tile8,
  +valid: PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),Tile.filled(Tile.lane0(tile)),
    Representation.lifted(~Right,~right_value,Representation.Lifted{right,geometry},word)>) ->
  PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),
    TiledBroadcast.spread_pairs(~operation,Representation.lifted(~Left,~left_value,Representation.Lifted{left,geometry},word),tile),
    Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),
      Representation.Lifted{CombineLeaf.Context{left,right},geometry},word)>:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}:
      TiledMapLaws.paired_valid(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,word,Tile.filled(a),valid)

# Packed cells of the left and the unit cells of the right, whose stretched
# view repeats along the lane axis of the left, combine into the operation of
# the left values and the right values read through that view.
def spread_cells(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+position: Grid.Position,+lm: PiecesModel.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,+rm: PiecesModel.Model<Tile.Tile8>,+rf: Tile.Tile8,+rw: U32,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+zero: {Views.stride_at(raxes,lx) == 0 : U32},
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),Tiled.cell_shape(ls,lx,Grid.Innermost{}),lf,Tensors.Usable{},lw},lx,Grid.lanes_at(ls,lx),Grid.Innermost{}}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),Tiled.cell_shape(rs,List.length(&2,U32,rs),position),rf,
    Tensors.Usable{},rw},List.length(&2,U32,rs),Grid.lanes_at(rs,List.length(&2,U32,rs)),position})) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),Tiled.cell_shape(ls,lx,Grid.Innermost{}),lf,Tensors.Usable{},lw},lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},
    Tiled.unit(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),Tiled.cell_shape(rs,List.length(&2,U32,rs),position),rf,Tensors.Usable{},rw}),spread_axes_of(ls,lx,raxes),roffset)):
  KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
  KernelGemm.Both{+right_global,Public.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  +left_laid = {Public.Laid{left_inside,left_shaped,left_restored,left_counted,left_pinned} :
    Public.Laid<Tiled.cell_shape(ls,lx,Grid.Innermost{}),lx,Grid.lanes_at(ls,lx),Grid.Innermost{},ls>}
  +extents = ViewReads.stretch_extents(Tensors.view_of(rs),ls,raxes,roffset,right_stretched)
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(raxes),ls,extents)
  +target = Views.axis_extents(raxes)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +target_room = {%back : {Nat.is_le(ShapeLaws.elements(_),capacity) == True{} : Bool} StorageCount.counted_room(limits,ls,left_counted,default_capacity) :
    {Nat.is_le(ShapeLaws.elements(target),capacity) == True{} : Bool}}
  +target_positive = {%back : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} left_nonempty : {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}}
  +target_inside = {%back : {Nat.is_le(lx,List.length(&2,U32,_)) == True{} : Bool} inside_axis : {Nat.is_le(lx,List.length(&2,U32,target)) == True{} : Bool}}
  +removed_room = KeptReads.removed_fits(Nat.is_lt(lx,List.length(&2,U32,target)),limits,target,lx,{==},target_positive,target_room)
  +cells_room = Rooms.cells_room(limits,target,lx,Grid.Innermost{},target_positive,target_room)
  +room = StorageCount.counted_room(limits,rs,right_counted,default_capacity)
  +right_count = Global.count(rs,right_counted)
  +right_accepted = Global.accepted(rs,right_counted)
  +length = Tensors.count_or_zero(rs)
  +length_exact = ViewMoved.counted_cells(rs,right_count,right_accepted)
  +stretched_fits = {%Equal.sym(Maybe<&2,U32>,Tensors.count_of(rs),Some{right_count},right_accepted) : {Views.fits(Views.View{raxes,roffset},Tensors.known_count(_)) == True{} : Bool}
    Stretched.stretched_fits(rs,ls,right_count,Global.whole_counted(rs,right_count,right_accepted),raxes,roffset,right_stretched,CountCompletion.positive_not_zero(ls,left_nonempty)) :
    {Views.fits(Views.View{raxes,roffset},length) == True{} : Bool}}
  +fits = {%CountCompletion.positive_not_zero(target,target_positive) : {Views.fits_nonempty(_,Views.View{raxes,roffset},length) == True{} : Bool} stretched_fits :
    {Views.fits_nonempty(False{},Views.View{raxes,roffset},length) == True{} : Bool}}
  +spread = spread_axes_of(target,lx,raxes)
  +spread_extents_of = {spread_extents(Nat.is_lt(lx,List.length(&2,U32,target)),raxes,lx,Tiled.groups(Grid.lanes_at(target,lx)),{==}) :
    {Views.axis_extents(spread) == Tiled.cell_shape(target,lx,Grid.Innermost{}) : List<&2,U32>}}
  +spread_nonzero = {%Equal.sym(List<&2,U32>,Views.axis_extents(spread),Tiled.cell_shape(target,lx,Grid.Innermost{}),spread_extents_of) : {Shapes.has_zero(_) == False{} : Bool}
    CountCompletion.positive_not_zero(Tiled.cell_shape(target,lx,Grid.Innermost{}),Rooms.cells_positive(limits,target,lx,Grid.Innermost{},target_positive,target_room)) :
    {Shapes.has_zero(Views.axis_extents(spread)) == False{} : Bool}}
  +spread_fitting = {spread_fits(Nat.is_lt(lx,List.length(&2,U32,target)),raxes,lx,Tiled.groups(Grid.lanes_at(target,lx)),roffset,length,zero,spread_nonzero,fits) :
    {Views.fits(Views.View{spread,roffset},length) == True{} : Bool}}
  +spread_fits_nonempty = {%spread_nonzero : {Views.fits_nonempty(_,Views.View{spread,roffset},length) == True{} : Bool} spread_fitting :
    {Views.fits_nonempty(False{},Views.View{spread,roffset},length) == True{} : Bool}}
  +viewed_extents = {%extents : {Views.axis_extents(spread_axes_of(_,lx,raxes)) == Tiled.cell_shape(_,lx,Grid.Innermost{}) : List<&2,U32>} spread_extents_of :
    {Views.axis_extents(spread_axes_of(ls,lx,raxes)) == Tiled.cell_shape(ls,lx,Grid.Innermost{}) : List<&2,U32>}}
  +viewed_left = {%extents : {Views.fits(Views.View{spread_axes_of(_,lx,raxes),roffset},length) == True{} : Bool} spread_fitting :
    {Views.fits(Views.View{spread_axes_of(ls,lx,raxes),roffset},length) == True{} : Bool}}
  +viewed_fits = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(rs,List.length(&2,U32,rs),position),rs,Public.unit_shape(rs,position)) :
      {Views.fits(Views.View{spread_axes_of(ls,lx,raxes),roffset},Tensors.count_or_zero(_)) == True{} : Bool} viewed_left :
    {Views.fits(Views.View{spread_axes_of(ls,lx,raxes),roffset},Tensors.count_or_zero(Tiled.cell_shape(rs,List.length(&2,U32,rs),position))) == True{} : Bool}}
  TiledMapLaws.viewed_models(~Left,~left_value,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,
    ~CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value)),~TiledBroadcast.spread_pairs(~operation),
    left,Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},CombineLeaf.Context{left,ViewGather.Viewing{right,raxes,roffset,[]}},ls,operations,lx,limits,
    default_capacity,lm,lf,lw,rm,Tiled.cell_shape(rs,List.length(&2,U32,rs),position),rf,rw,List.length(&2,U32,Tiled.cell_shape(rs,List.length(&2,U32,rs),position)),1,Grid.Innermost{},
    spread_axes_of(ls,lx,raxes),roffset,viewed_extents,viewed_fits,left_nonempty,rank_room,left_laid,left_global,right_global,
    word => inside => spread_valid(~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),~operation,left,ViewGather.Viewing{right,raxes,roffset,[]},
      Public.geometry_of(ls,lx,Grid.Innermost{}),word,
      ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
        ViewGather.Viewing{Representation.Lifted{right,Public.geometry_of(rs,List.length(&2,U32,rs),position)},spread_axes_of(ls,lx,raxes),roffset,[]},word),
      spread_word(~Right,~right_value,right,limits,ls,rs,position,raxes,roffset,lx,extents,zero,target_inside,target_room,removed_room,cells_room,room,length,length_exact,fits,
        spread_fits_nonempty,word,inside)))

# Cells of the left of the contract and packed tiled values of the right,
# read one value a cell.
def spread_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,left_cells: Tensors.Tensor<Tile.Tile8>,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+zero: {Views.stride_at(raxes,lx) == 0 : U32},
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{}}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.Tiled{left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},Tiled.unit(Tiled.unit_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})),
    spread_axes_of(ls,lx,raxes),roffset)):
  KernelGemm.Both{+right_global,Public.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(rc,ro,rl,rp),rs,right_restored) :
    StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
      Tiled.Tiled{left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},
      Tiled.unit(Tiled.cells_of(Tiled.along(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},List.length(&2,U32,_)))),
      spread_axes_of(ls,lx,raxes),roffset))
  Relane.along_then(~Right,~right_value,
    (tiled => StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
      Tiled.Tiled{left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},Tiled.unit(Tiled.cells_of(tiled)),spread_axes_of(ls,lx,raxes),roffset))),
    right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},List.length(&2,U32,rs),limits,default_capacity,
    Order.reflexive(List.length(&2,U32,rs)),right_nonempty,right_rank_room,right_contract,
    cells => lanes => position => along_contract => Relane.modeled(~Right,~right_value,
      (tiled => StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
        Tiled.Tiled{left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},Tiled.unit(Tiled.cells_of(tiled)),spread_axes_of(ls,lx,raxes),roffset))),
      right,rs,cells,List.length(&2,U32,rs),lanes,position,along_contract,
      unit_model => unit_fill => unit_workers => right_modeled => Relane.modeled(~Left,~left_value,
        (tiled => StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
          tiled,Tiled.unit(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,unit_model),Tiled.cell_shape(rs,List.length(&2,U32,rs),position),unit_fill,Tensors.Usable{},unit_workers}),
          spread_axes_of(ls,lx,raxes),roffset))),
        left,ls,left_cells,lx,Grid.lanes_at(ls,lx),Grid.Innermost{},left_contract,
        lm => lf => lw => left_modeled => spread_cells(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,raxes,roffset,position,
          lm,lf,lw,unit_model,unit_fill,unit_workers,right_stretched,zero,left_nonempty,rank_room,inside_axis,left_modeled,right_modeled))))

def spread_arranged(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,left_tiled: Tiled.Tiled,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+zero: {Views.stride_at(raxes,lx) == 0 : U32},
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_arranged: Relane.Arranged(~Left,~left_value,left,ls,left_tiled,lx,Grid.Innermost{}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.unit(Tiled.unit_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})),spread_axes_of(ls,lx,raxes),roffset)):
  KernelGemm.Both{+left_form,+left_contract} = left_arranged
  %Equal.sym(Tiled.Tiled,Tiled.arranged(left_tiled,lx,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(left_tiled,lx,Grid.Innermost{})),lx,Grid.lanes_at(ls,lx),Grid.Innermost{}},left_form) :
    StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
      _,Tiled.unit(Tiled.unit_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})),spread_axes_of(ls,lx,raxes),roffset))
  spread_packed(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,raxes,roffset,
    Tiled.cells_of(Tiled.arranged(left_tiled,lx,Grid.Innermost{})),rm,rc,rf,rw,ro,rl,rp,right_stretched,zero,left_nonempty,right_nonempty,rank_room,right_rank_room,inside_axis,
    left_contract,right_contract)

# Packed tiled operands, the left arranged with its lanes on its axis.
def spread_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,-left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,-right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},+zero: {Views.stride_at(raxes,lx) == 0 : U32},
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.unit(Tiled.unit_cells(right_tiled)),spread_axes_of(ls,lx,raxes),roffset)):
  Public.Packed{+lm,+lc,+lf,+lw,+lo,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+ro,+rl,+rp,+right_equation} = right_packed
  %left_equation : StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.arranged(_,lx,Grid.Innermost{}),Tiled.unit(Tiled.unit_cells(right_tiled)),spread_axes_of(ls,lx,raxes),roffset))
  %right_equation : StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lx,Grid.Innermost{}),Tiled.unit(Tiled.unit_cells(_)),spread_axes_of(ls,lx,raxes),roffset))
  spread_arranged(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,raxes,roffset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
    rm,rc,rf,rw,ro,rl,rp,right_stretched,zero,left_nonempty,right_nonempty,rank_room,right_rank_room,inside_axis,
    Relane.arranged_held(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lx,Grid.Innermost{},limits,default_capacity,inside_axis,left_nonempty,rank_room,
      {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) : Public.Contract(~Left,~left_value,left,ls,_)
      left_contract : Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})}),
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) : Public.Contract(~Right,~right_value,right,rs,_)
      right_contract : Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})})

# A left operand of the common extents and a right operand whose view
# stretched to them repeats along the lane axis of the left: with the groups
# of the left innermost, every left cell combines with the one right value its
# cell reads, into the operation of the left values and the right values read
# through the stretched view.
def spread(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,+operations: F32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,+lx: Nat,
  +raxes: List<&2,Views.Axis>,+roffset: U32,left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +repeated: {U32.is_zero(Views.stride_at(raxes,lx)) == True{} : Bool},
  +left_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(ls)) == True{} : Bool},+right_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(rs)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_axis: {Nat.is_le(lx,List.length(&2,U32,ls)) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  StretchedContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,raxes,roffset,TiledMap.combine_viewed(~TiledBroadcast.spread_pairs(~operation),operations,
    Tiled.arranged(left_tiled,lx,Grid.Innermost{}),Tiled.unit(Tiled.unit_cells(right_tiled)),spread_axes_of(ls,lx,raxes),roffset)):
  spread_described(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lx,raxes,roffset,
    left_tiled,Public.packed(~Left,~left_value,left,ls,left_tiled,left_contract),right_tiled,Public.packed(~Right,~right_value,right,rs,right_tiled,right_contract),
    right_stretched,Arithmetic.word_zero(Views.stride_at(raxes,lx),repeated),left_nonempty,right_nonempty,rank_room,right_rank_room,inside_axis,left_contract,right_contract)

# ---- The broadcast of two operands ----

# The operation of the two operands read through their views stretched to the
# common extents.
def BroadcastContract(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~CombineLeaf.Context<ViewGather.Viewing<Left>,ViewGather.Viewing<Right>>,~CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~ViewGather.Viewing<Right>,~operation,
    ~ViewGather.viewed(~F32,~Left,~left_value),~ViewGather.viewed(~F32,~Right,~right_value)),
    CombineLeaf.Context{ViewGather.Viewing{left,laxes,loffset,[]},ViewGather.Viewing{right,raxes,roffset,[]}},shape,tiled)

# The left of a context read through a view.
def left_viewed(~Left: Data,~Right: Data,context: CombineLeaf.Context<Left,Right>,+axes: List<&2,Views.Axis>,+offset: U32) -> CombineLeaf.Context<ViewGather.Viewing<Left>,Right>:
  CombineLeaf.Context{left,right} = context
  CombineLeaf.Context{ViewGather.Viewing{left,axes,offset,[]},right}

# A context with its left read through the view stretched to its own extents
# combines the same values.
def own_pair(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+shape: List<&2,U32>,+count: U32,
  +exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,+offset: U32,
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +before: CombineLeaf.Context<Left,Right>,+after: CombineLeaf.Context<ViewGather.Viewing<Left>,Right>,
  +related: {after == left_viewed(~Left,~Right,before,axes,offset) : CombineLeaf.Context<ViewGather.Viewing<Left>,Right>},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value,before,index) ==
    CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~Right,~operation,~ViewGather.viewed(~F32,~Left,~left_value),~right_value,after,index) : F32}:
  match before:
    case CombineLeaf.Context{+left,+right}:
      %Equal.sym(CombineLeaf.Context<ViewGather.Viewing<Left>,Right>,after,left_viewed(~Left,~Right,CombineLeaf.Context{left,right},axes,offset),related) :
        {CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value,CombineLeaf.Context{left,right},index) ==
          CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~Right,~operation,~ViewGather.viewed(~F32,~Left,~left_value),~right_value,_,index) : F32}
      Equal.cong(F32,F32,value => operation(value,right_value(right,index)),left_value(left,index),
        ViewGather.viewed(~F32,~Left,~left_value,ViewGather.Viewing{left,axes,offset,[]},index),
        own_agree(~Left,~left_value,shape,count,exact,axes,offset,stretched,left,ViewGather.Viewing{left,axes,offset,[]},{==},index,inside))

# Tiled values of the operation of a left of the extents and a right reader
# hold it with the left read through its view stretched to its own extents.
def own_left(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+limits: Machine.Limits,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  tiled: Tiled.Tiled,
  +contract: Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},shape,tiled)) ->
  Public.Contract(~CombineLeaf.Context<ViewGather.Viewing<Left>,Right>,~CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~Right,~operation,~ViewGather.viewed(~F32,~Left,~left_value),
    ~right_value),CombineLeaf.Context{ViewGather.Viewing{left,axes,offset,[]},right},shape,tiled):
  KeptReads.contract_below(~CombineLeaf.Context<Left,Right>,~CombineLeaf.Context<ViewGather.Viewing<Left>,Right>,
    ~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),
    ~CombineLeaf.value(~F32,~ViewGather.Viewing<Left>,~Right,~operation,~ViewGather.viewed(~F32,~Left,~left_value),~right_value),
    ~(before => after => {after == left_viewed(~Left,~Right,before,axes,offset) : CombineLeaf.Context<ViewGather.Viewing<Left>,Right>}),~ShapeLaws.elements(shape),
    ~own_pair(~Left,~left_value,~Right,~right_value,~operation,shape,count,exact,axes,offset,stretched),limits,CombineLeaf.Context{left,right},
    CombineLeaf.Context{ViewGather.Viewing{left,axes,offset,[]},right},{==},shape,positive,room,Order.reflexive(ShapeLaws.elements(shape)),tiled,contract)

# The right of a context read through a view.
def right_viewed(~Left: Data,~Right: Data,context: CombineLeaf.Context<Left,Right>,+axes: List<&2,Views.Axis>,+offset: U32) -> CombineLeaf.Context<Left,ViewGather.Viewing<Right>>:
  CombineLeaf.Context{left,right} = context
  CombineLeaf.Context{left,ViewGather.Viewing{right,axes,offset,[]}}

# The same with the right read through its own view.
def own_second(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+shape: List<&2,U32>,+count: U32,
  +exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,+offset: U32,
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +before: CombineLeaf.Context<Left,Right>,+after: CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,
  +related: {after == right_viewed(~Left,~Right,before,axes,offset) : CombineLeaf.Context<Left,ViewGather.Viewing<Right>>},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value,before,index) ==
    CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value),after,index) : F32}:
  match before:
    case CombineLeaf.Context{+left,+right}:
      %Equal.sym(CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,after,right_viewed(~Left,~Right,CombineLeaf.Context{left,right},axes,offset),related) :
        {CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value,CombineLeaf.Context{left,right},index) ==
          CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value),_,index) : F32}
      Equal.cong(F32,F32,value => operation(left_value(left,index),value),right_value(right,index),
        ViewGather.viewed(~F32,~Right,~right_value,ViewGather.Viewing{right,axes,offset,[]},index),
        own_agree(~Right,~right_value,shape,count,exact,axes,offset,stretched,right,ViewGather.Viewing{right,axes,offset,[]},{==},index,inside))

def own_right(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+limits: Machine.Limits,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  tiled: Tiled.Tiled,
  +contract: Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},shape,tiled)) ->
  Public.Contract(~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,
    ~ViewGather.viewed(~F32,~Right,~right_value)),CombineLeaf.Context{left,ViewGather.Viewing{right,axes,offset,[]}},shape,tiled):
  KeptReads.contract_below(~CombineLeaf.Context<Left,Right>,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,
    ~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),
    ~CombineLeaf.value(~F32,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~F32,~Right,~right_value)),
    ~(before => after => {after == right_viewed(~Left,~Right,before,axes,offset) : CombineLeaf.Context<Left,ViewGather.Viewing<Right>>}),~ShapeLaws.elements(shape),
    ~own_second(~Left,~left_value,~Right,~right_value,~operation,shape,count,exact,axes,offset,stretched),limits,CombineLeaf.Context{left,right},
    CombineLeaf.Context{left,ViewGather.Viewing{right,axes,offset,[]}},{==},shape,positive,room,Order.reflexive(ShapeLaws.elements(shape)),tiled,contract)

# Both operands repeated to the common extents.
def values_decided(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.values(~operation,name,operations,ls,rs,shape,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
  KernelGemm.Both{+right_global,Public.Laid{+right_inside,+right_shaped,+right_restored,+right_counted,+right_pinned}} = right_contract
  +left_count = Global.count(ls,left_counted)
  +left_accepted = Global.accepted(ls,left_counted)
  +right_count = Global.count(rs,right_counted)
  +right_accepted = Global.accepted(rs,right_counted)
  values_packed(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,ls,rs,shape,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,limits,default_capacity,
    laxes,loffset,left_stretched,left_nonzero,rank_room,
    {%Equal.sym(Maybe<&2,U32>,Tensors.count_of(ls),Some{left_count},left_accepted) : {Views.fits(Views.View{laxes,loffset},Tensors.known_count(_)) == True{} : Bool}
      Stretched.stretched_fits(ls,shape,left_count,Global.whole_counted(ls,left_count,left_accepted),laxes,loffset,left_stretched,shape_nonzero) :
      {Views.fits(Views.View{laxes,loffset},Tensors.count_or_zero(ls)) == True{} : Bool}},
    raxes,roffset,right_stretched,right_nonzero,right_rank_room,
    {%Equal.sym(Maybe<&2,U32>,Tensors.count_of(rs),Some{right_count},right_accepted) : {Views.fits(Views.View{raxes,roffset},Tensors.known_count(_)) == True{} : Bool}
      Stretched.stretched_fits(rs,shape,right_count,Global.whole_counted(rs,right_count,right_accepted),raxes,roffset,right_stretched,shape_nonzero) :
      {Views.fits(Views.View{raxes,roffset},Tensors.count_or_zero(rs)) == True{} : Bool}},
    shape_nonzero,shape_rank,Global.count(shape,counted),Global.accepted(shape,counted),left_contract,right_contract)

# A right operand that repeats along the lane axis of the left is read one
# value a cell, otherwise both operands are repeated.
def spread_decided(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,repeats: Bool,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_own: {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +classified: {U32.is_zero(Views.stride_at(raxes,lo)) == repeats : Bool},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,laxes,loffset,raxes,roffset,TiledBroadcast.spread_or(~operation,name,operations,repeats,lo,ls,rs,
    raxes,roffset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  match repeats:
    case True{}:
      KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
      +left_count = Global.count(ls,left_counted)
      +exact = Global.whole_counted(ls,left_count,Global.accepted(ls,left_counted))
      +positive = CountCompletion.positive_elements(ls,left_nonzero)
      +room = StorageCount.counted_room(limits,ls,left_counted,default_capacity)
      own_left(~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),~operation,left,ViewGather.Viewing{right,raxes,roffset,[]},ls,laxes,loffset,
        left_count,exact,left_own,limits,positive,room,
        TiledBroadcast.spread_with(~operation,operations,lo,ls,raxes,roffset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}),
        spread(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lo,raxes,roffset,
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
          right_stretched,classified,positive,CountCompletion.positive_elements(rs,right_nonzero),rank_room,right_rank_room,left_inside,left_contract,right_contract))
    case False{}:
      KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
      values_decided(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
        ls,laxes,loffset,raxes,roffset,left_own,right_stretched,left_nonzero,right_nonzero,left_nonzero,rank_room,right_rank_room,rank_room,left_counted,left_contract,right_contract)

# A lane test that holds names the lane.
def lane_exact_of(lane: Maybe<&2,Nat>,+axis: Nat,+tested: {TiledBroadcast.lane_is(lane,axis) == True{} : Bool}) -> {lane == Some{axis} : Maybe<&2,Nat>}:
  match lane:
    case Some{+found}:
      Equal.cong(Nat,Maybe<&2,Nat>,value => Some{value},found,axis,Order.equal_of_is_eq(found,axis,tested))
    case None{}: Empty.absurd({None{} == Some{axis} : Maybe<&2,Nat>},WordEncoding.false_true(tested))

# The right cells read through the view of the cells when the mapped axes
# keep the lanes of the right on the lane axis of the left.
def aligned_decided(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,keeps: Bool,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +rx: Nat,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_own: {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +valid: Bool,+lane: Maybe<&2,Nat>,+lane_extent: U32,+kept: List<&2,Views.Axis>,+flags: List<&2,Bool>,+reach: List<&2,U32>,
  +found: {mapped(rs,rx,raxes,roffset) == TiledView.Mapped{valid,lane,lane_extent,kept,flags,reach} : TiledView.Mapped},
  +classified: {Nat.is_lt(rx,List.length(&2,U32,rs)) && valid && TiledBroadcast.lane_is(lane,lo) && TiledView.all_inside(reach,rs) &&
    U32.is_zero((TiledView.coordinate_at(Grid.coordinates(rs,roffset),rx) % 8 : U32)) == keeps : Bool},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,laxes,loffset,raxes,roffset,TiledBroadcast.aligned_with(~operation,name,operations,keeps,lo,rx,ls,rs,
    raxes,roffset,List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(lane_extent),KeptReads.group_step(rs,rx,Grid.Innermost{})}]),cell_offset(rs,rx,roffset),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  match keeps:
    case True{}:
      KernelGemm.Both{+left_global,Public.Laid{+left_inside,+left_shaped,+left_restored,+left_counted,+left_pinned}} = left_contract
      +left_count = Global.count(ls,left_counted)
      +exact = Global.whole_counted(ls,left_count,Global.accepted(ls,left_counted))
      +positive = CountCompletion.positive_elements(ls,left_nonzero)
      +room = StorageCount.counted_room(limits,ls,left_counted,default_capacity)
      +within = Nat.is_lt(rx,List.length(&2,U32,rs))
      +lane_held = TiledBroadcast.lane_is(lane,lo)
      +inside = TiledView.all_inside(reach,rs)
      +aligned = U32.is_zero((TiledView.coordinate_at(Grid.coordinates(rs,roffset),rx) % 8 : U32))
      +back = Equal.sym(TiledView.Mapped,mapped(rs,rx,raxes,roffset),TiledView.Mapped{valid,lane,lane_extent,kept,flags,reach},found)
      +four = Boolean.and_true_left(within && valid && lane_held && inside,aligned,classified)
      +three = Boolean.and_true_left(within && valid && lane_held,inside,four)
      +two = Boolean.and_true_left(within && valid,lane_held,three)
      +facts = {Held{Boolean.and_true_left(within,valid,two),
          {%back : {KeptReads.mapped_valid(_) == True{} : Bool} Boolean.and_true_right(within,valid,two) :
            {KeptReads.mapped_valid(mapped(rs,rx,raxes,roffset)) == True{} : Bool}},
          {%back : {KeptReads.mapped_lane(_) == Some{lo} : Maybe<&2,Nat>} lane_exact_of(lane,lo,Boolean.and_true_right(within && valid,lane_held,three)) :
            {KeptReads.mapped_lane(mapped(rs,rx,raxes,roffset)) == Some{lo} : Maybe<&2,Nat>}},
          {%back : {TiledView.all_inside(KeptReads.mapped_reach(_),rs) == True{} : Bool} Boolean.and_true_right(within && valid && lane_held,inside,four) :
            {TiledView.all_inside(KeptReads.mapped_reach(mapped(rs,rx,raxes,roffset)),rs) == True{} : Bool}},
          Boolean.and_true_right(within && valid && lane_held && inside,aligned,classified)} : Held<rs,rx,lo,raxes,roffset>}
      %found : BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,laxes,loffset,raxes,roffset,TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
        Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lo,Grid.Innermost{}),
        Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},rx,Grid.Innermost{}),
        KeptReads.cell_axes_of(KeptReads.mapped_kept(_),KeptReads.mapped_extent(_),rs,rx,Grid.Innermost{}),cell_offset(rs,rx,roffset)))
      own_left(~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~F32,~Right,~right_value),~operation,left,ViewGather.Viewing{right,raxes,roffset,[]},ls,laxes,loffset,
        left_count,exact,left_own,limits,positive,room,
        TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,
          Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},lo,Grid.Innermost{}),
          Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},rx,Grid.Innermost{}),cell_axes(rs,rx,raxes,roffset),cell_offset(rs,rx,roffset)),
        held(~Left,~left_value,~Right,~right_value,~operation,left,right,operations,limits,default_capacity,ls,rs,lo,rx,raxes,roffset,
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},
          right_stretched,facts,positive,CountCompletion.positive_elements(rs,right_nonzero),rank_room,right_rank_room,left_inside,left_contract,right_contract))
    case False{}:
      spread_decided(~Left,~left_value,~Right,~right_value,~operation,U32.is_zero(Views.stride_at(raxes,lo)),left,right,name,operations,limits,default_capacity,ls,rs,
        lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,laxes,loffset,raxes,roffset,left_own,right_stretched,{==},left_nonzero,right_nonzero,rank_room,right_rank_room,left_contract,right_contract)

# The axes the stretched view maps on the right cells decide.
def mapped_decided(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,mapping: TiledView.Mapped,+left: Left,
  +right: Right,+name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,
  +rs: List<&2,U32>,+lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +rx: Nat,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_own: {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +found: {mapped(rs,rx,raxes,roffset) == mapping : TiledView.Mapped},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,laxes,loffset,raxes,roffset,TiledBroadcast.held_cells(~operation,name,operations,mapping,lo,rx,ls,rs,
    TiledView.lanes_of(rs,rx,Grid.Innermost{}),raxes,roffset,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  match mapping:
    case TiledView.Mapped{+valid,+lane,+lane_extent,+kept,+flags,+reach}:
      aligned_decided(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(rx,List.length(&2,U32,rs)) && valid && TiledBroadcast.lane_is(lane,lo) && TiledView.all_inside(reach,rs) &&
          U32.is_zero((TiledView.coordinate_at(Grid.coordinates(rs,roffset),rx) % 8 : U32)),
        left,right,name,operations,limits,default_capacity,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
        rx,laxes,loffset,raxes,roffset,left_own,right_stretched,valid,lane,lane_extent,kept,flags,reach,found,{==},left_nonzero,right_nonzero,rank_room,right_rank_room,
        left_contract,right_contract)

# A left of the common extents keeps its lanes and reads the right through its
# stretched view, otherwise both operands are repeated.
def full_decided(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,full: Bool,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +classified: {Shapes.same_extents(ls,shape) && Bool.not(Shapes.has_zero(ls)) && Bool.not(Shapes.has_zero(rs)) == full : Bool},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.full_cells(~operation,name,full,operations,ls,rs,shape,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  match full:
    case True{}:
      +same_shape = Shapes.same_extents(ls,shape)
      +nonzero_left = Bool.not(Shapes.has_zero(ls))
      +same = ShapeLaws.same_extents(ls,shape,Boolean.and_true_left(same_shape,nonzero_left,
        Boolean.and_true_left(same_shape && nonzero_left,Bool.not(Shapes.has_zero(rs)),classified)))
      +back = Equal.sym(List<&2,U32>,ls,shape,same)
      +rx = Nat.sub(lo,Nat.sub(List.length(&2,U32,ls),List.length(&2,U32,rs)))
      +right_own = {%back : {Tensors.stretched(Tensors.view_of(rs),_) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>} right_stretched :
        {Tensors.stretched(Tensors.view_of(rs),ls) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>}}
      +left_own = {%back : {Tensors.stretched(Tensors.view_of(ls),_) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>} left_stretched :
        {Tensors.stretched(Tensors.view_of(ls),ls) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>}}
      %same : BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,_,laxes,loffset,raxes,roffset,TiledBroadcast.held_view(~operation,name,operations,
        Tensors.spread_view(Tensors.stretched(Tensors.view_of(rs),ls)),lo,rx,ls,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
      %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(rs),ls),Some{Views.View{raxes,roffset}},right_own) :
        BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,ls,laxes,loffset,raxes,roffset,TiledBroadcast.held_view(~operation,name,operations,
          Tensors.spread_view(_),lo,rx,ls,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
      mapped_decided(~Left,~left_value,~Right,~right_value,~operation,mapped(rs,rx,raxes,roffset),left,right,name,operations,limits,default_capacity,ls,rs,
        lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,rx,laxes,loffset,raxes,roffset,left_own,right_own,{==},left_nonzero,right_nonzero,rank_room,right_rank_room,
        left_contract,right_contract)
    case False{}:
      values_decided(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
        shape,laxes,loffset,raxes,roffset,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,
        left_contract,right_contract)

# Packed tiled operands of the contract under NumPy broadcasting.
def combined_packed(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lm: PiecesModel.Model<Tile.Tile8>,+lc: List<&2,U32>,+lf: Tile.Tile8,+lw: U32,+lo: Nat,+ll: U32,+lp: Grid.Position,
  +rm: PiecesModel.Model<Tile.Tile8>,+rc: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+ro: Nat,+rl: U32,+rp: Grid.Position,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp}),
  +right_contract: Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.combine(~operation,name,operations,ls,rs,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})):
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(ls,rs),Some{shape},common) :
    BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.common_shape(~operation,name,_,operations,ls,rs,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp}))
  full_decided(~Left,~left_value,~Right,~right_value,~operation,Shapes.same_extents(ls,shape) && Bool.not(Shapes.has_zero(ls)) && Bool.not(Shapes.has_zero(rs)),left,right,name,
    operations,limits,default_capacity,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,shape,laxes,loffset,raxes,roffset,left_stretched,right_stretched,{==},
    left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,left_contract,right_contract)

def combined_described(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  -left_tiled: Tiled.Tiled,+left_packed: Public.Packed<left_tiled>,-right_tiled: Tiled.Tiled,+right_packed: Public.Packed<right_tiled>,
  +shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.combine(~operation,name,operations,ls,rs,left_tiled,right_tiled)):
  Public.Packed{+lm,+lc,+lf,+lw,+lo,+ll,+lp,+left_equation} = left_packed
  Public.Packed{+rm,+rc,+rf,+rw,+ro,+rl,+rp,+right_equation} = right_packed
  %left_equation : BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    TiledBroadcast.combine(~operation,name,operations,ls,rs,_,right_tiled))
  %right_equation : BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,
    TiledBroadcast.combine(~operation,name,operations,ls,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},_))
  combined_packed(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,lm,lc,lf,lw,lo,ll,lp,rm,rc,rf,rw,ro,rl,rp,
    shape,laxes,loffset,raxes,roffset,common,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp},left_tiled,left_equation) :
      Public.Contract(~Left,~left_value,left,ls,_) left_contract :
      Public.Contract(~Left,~left_value,left,ls,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,lm),lc,lf,Tensors.Usable{},lw},lo,ll,lp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp},right_tiled,right_equation) :
      Public.Contract(~Right,~right_value,right,rs,_) right_contract :
      Public.Contract(~Right,~right_value,right,rs,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,rm),rc,rf,Tensors.Usable{},rw},ro,rl,rp})})

# Tiled operands of the contract and of extents without a zero under NumPy
# broadcasting: the result holds at every index of the common extents the
# operation of the two values the operands read through their views
# stretched to those extents.
def combined(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +name: String,+operations: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+ls: List<&2,U32>,+rs: List<&2,U32>,
  left_tiled: Tiled.Tiled,right_tiled: Tiled.Tiled,+shape: List<&2,U32>,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +common: {Shapes.broadcast(ls,rs) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +left_stretched: {Tensors.stretched(Tensors.view_of(ls),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of(rs),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +left_nonzero: {Shapes.has_zero(ls) == False{} : Bool},+right_nonzero: {Shapes.has_zero(rs) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,ls),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_rank_room: {Nat.is_le(List.length(&2,U32,rs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +left_contract: Public.Contract(~Left,~left_value,left,ls,left_tiled),+right_contract: Public.Contract(~Right,~right_value,right,rs,right_tiled)) ->
  BroadcastContract(~Left,~left_value,~Right,~right_value,~operation,left,right,shape,laxes,loffset,raxes,roffset,TiledBroadcast.combine(~operation,name,operations,ls,rs,left_tiled,right_tiled)):
  combined_described(~Left,~left_value,~Right,~right_value,~operation,left,right,name,operations,limits,default_capacity,ls,rs,
    left_tiled,Public.packed(~Left,~left_value,left,ls,left_tiled,left_contract),right_tiled,Public.packed(~Right,~right_value,right,rs,right_tiled,right_contract),
    shape,laxes,loffset,raxes,roffset,common,left_stretched,right_stretched,left_nonzero,right_nonzero,shape_nonzero,rank_room,right_rank_room,shape_rank,counted,
    left_contract,right_contract)
