# Binary operations of tiled operands that broadcast. When the left operand
# has the common extents, the right takes its lanes on the axis NumPy
# broadcasting aligns with the lane axis of the left, both groups innermost,
# and the cells broadcast as the values do: the right cells are read through
# the view of the cells of its stretched view, the view a gather of the cells
# reads. A right operand repeated along that axis has its
# values one a cell, every lane of a left cell combining with the value of
# its right cell. A left operand of other extents, and an empty operand, is
# repeated to the common extents first.
import Base
import ./kernel_tile.bend as Tile
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_shape.bend as Shapes
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_view.bend as TiledView
import ./tensor_view.bend as Views
import ./tensor_tiled_layout.bend as TiledLayout

def innermost(tiled: Tiled.Tiled,+axis: Nat) -> Tiled.Tiled:
  Tiled.arranged(tiled,axis,Grid.Innermost{})

# Operands of one shape combine cell by cell, the lanes of the right on the
# axis of the left.
def same_axes(~operation: F32 -> F32 -> F32,name: String,operations: F32,left: Tiled.Tiled & Nat,right: Tiled.Tiled) -> Tiled.Tiled:
  (left,+axis) = left
  TiledMap.combine(~TiledMap.pairs(~operation),name,operations,innermost(left,axis),innermost(right,axis))

def same(~operation: F32 -> F32 -> F32,name: String,operations: F32,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  same_axes(~operation,name,operations,Tiled.lane_axis_of(left),right)

# An operand of other extents is repeated to the common ones.
def repeated(same: Bool,+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match same:
    case True{}: tiled
    case False{}: TiledLayout.broadcast_to(tiled,shape)

# Both operands repeated to the common extents combine cell by cell.
def values(~operation: F32 -> F32 -> F32,name: String,operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,+shape: List<&2,U32>,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  same(~operation,name,operations,repeated(Shapes.same_extents(ls,shape),shape,left),repeated(Shapes.same_extents(rs,shape),shape,right))

# The operation of every lane of the left tile with the first lane of the
# right.
def spread_pairs(~operation: F32 -> F32 -> F32,left: Tile.Tile8,right: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = left
  Tile.Tile{+value,j,k,l,m,n,o,p} = right
  Tile.Tile{operation(a,value),operation(b,value),operation(c,value),operation(d,value),operation(e,value),operation(f,value),operation(g,value),operation(h,value)}

# A view of values that repeats their lane axis, as a view of their cells:
# the other axes, then the groups at stride 0. Lanes past the axes leave the
# view as it is.
def spread_axes(within: Bool,+axes: List<&2,Views.Axis>,+axis: Nat,+groups: U32) -> List<&2,Views.Axis>:
  match within:
    case True{}: List.append(&2,Views.Axis,Views.removed_axis(axes,axis),[Views.Axis{groups,0}])
    case False{}: axes

# A right operand repeated along the lane axis of the left has its values one
# a cell, read through its stretched view as a view of the left cells.
def spread_with(~operation: F32 -> F32 -> F32,operations: F32,+axis: Nat,+ls: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,left: Tiled.Tiled,
  right: Tiled.Tiled) -> Tiled.Tiled:
  TiledMap.combine_viewed(~spread_pairs(~operation),operations,innermost(left,axis),Tiled.unit(Tiled.unit_cells(right)),
    spread_axes(Nat.is_lt(axis,List.length(&2,U32,ls)),axes,axis,Tiled.groups(Grid.lanes_at(ls,axis))),offset)

def spread_or(~operation: F32 -> F32 -> F32,name: String,operations: F32,spread: Bool,+axis: Nat,+ls: List<&2,U32>,+rs: List<&2,U32>,+axes: List<&2,Views.Axis>,
  +offset: U32,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  match spread:
    case True{}: spread_with(~operation,operations,axis,ls,axes,offset,left,right)
    case False{}: values(~operation,name,operations,ls,rs,ls,left,right)

def aligned_with(~operation: F32 -> F32 -> F32,name: String,operations: F32,held: Bool,+axis: Nat,+right_axis: Nat,+ls: List<&2,U32>,+rs: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+cell_axes: List<&2,Views.Axis>,+cell_offset: U32,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  match held:
    case True{}: TiledMap.combine_viewed(~TiledMap.pairs(~operation),operations,innermost(left,axis),innermost(right,right_axis),cell_axes,cell_offset)
    case False{}: spread_or(~operation,name,operations,U32.is_zero(Views.stride_at(axes,axis)),axis,ls,rs,axes,offset,left,right)

def lane_is(lane: Maybe<&2,Nat>,+axis: Nat) -> Bool:
  match lane:
    case Some{+found}: Nat.is_eq(found,axis)
    case None{}: False{}

# The right keeps its lanes when its stretched view keeps them on the lane
# axis of the left, as a gather of the cells does: the right cells are read
# through the view of the cells of that gather. Otherwise a right operand
# that repeats along the lane axis of the left is read one value a cell.
def held_cells(~operation: F32 -> F32 -> F32,name: String,operations: F32,mapped: TiledView.Mapped,+axis: Nat,+right_axis: Nat,+ls: List<&2,U32>,+rs: List<&2,U32>,
  +lanes: TiledView.Lanes,+axes: List<&2,Views.Axis>,+offset: U32,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  TiledView.Mapped{valid,lane,+lane_extent,+kept,flags,reach} = mapped
  aligned_with(~operation,name,operations,
    Nat.is_lt(right_axis,List.length(&2,U32,rs)) && valid && lane_is(lane,axis) && TiledView.all_inside(reach,rs) &&
      U32.is_zero((TiledView.coordinate_at(Grid.coordinates(rs,offset),right_axis) % 8 : U32)),
    axis,right_axis,ls,rs,axes,offset,
    List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(lane_extent),TiledView.cell_index(lanes,(8 * TiledView.lane_inner(lanes) : U32))}]),
    TiledView.cell_index(lanes,offset),left,right)

def held_view(~operation: F32 -> F32 -> F32,name: String,operations: F32,view: Views.View,+axis: Nat,+right_axis: Nat,+ls: List<&2,U32>,+rs: List<&2,U32>,
  left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  Views.View{+axes,+offset} = view
  +lanes = TiledView.lanes_of(rs,right_axis,Grid.Innermost{})
  held_cells(~operation,name,operations,TiledView.mapped_axes(axes,[],0n,rs,lanes,Grid.coordinates(rs,offset)),axis,right_axis,ls,rs,lanes,axes,offset,left,right)

# The lanes of the left on axis: the right axis aligned with it holds the same
# extent or repeats one index.
def aligned_cells(~operation: F32 -> F32 -> F32,name: String,operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,started: Tiled.Tiled & Nat,right: Tiled.Tiled) -> Tiled.Tiled:
  (left,+axis) = started
  held_view(~operation,name,operations,Tensors.spread_view(Tensors.stretched(Tensors.view_of(rs),ls)),axis,
    Nat.sub(axis,Nat.sub(List.length(&2,U32,ls),List.length(&2,U32,rs))),ls,rs,left,right)

def full_cells(~operation: F32 -> F32 -> F32,name: String,full: Bool,operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,+shape: List<&2,U32>,left: Tiled.Tiled,right: Tiled.Tiled) ->
  Tiled.Tiled:
  match full:
    case True{}: aligned_cells(~operation,name,operations,ls,rs,Tiled.lane_axis_of(left),right)
    case False{}: values(~operation,name,operations,ls,rs,shape,left,right)

def common_shape(~operation: F32 -> F32 -> F32,name: String,common: Maybe<&2,List<&2,U32>>,operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  match common:
    case Some{+shape}:
      full_cells(~operation,name,Shapes.same_extents(ls,shape) && Bool.not(Shapes.has_zero(ls)) && Bool.not(Shapes.has_zero(rs)),operations,ls,rs,shape,left,right)
    case None{}: Tiled.with_status(Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{ls,rs}}},left)

# Usable tiled operands of other extents under NumPy broadcasting, on the
# cells. Incompatible extents fail and keep the left tensor.
def combine(~operation: F32 -> F32 -> F32,name: String,operations: F32,+ls: List<&2,U32>,+rs: List<&2,U32>,left: Tiled.Tiled,right: Tiled.Tiled) -> Tiled.Tiled:
  common_shape(~operation,name,Shapes.broadcast(ls,rs),operations,ls,rs,left,right)
