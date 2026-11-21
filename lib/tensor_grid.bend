# The canonical order of cells and the pieces that store them.
#
# A tiled tensor of a logical shape keeps eight values of its lane axis in one
# cell. Its cell axes are the other axes in order with the groups of the lane
# axis, ceil(lanes / 8), innermost, or outermost for operands a kernel streams
# by group. Cells are row-major over the cell axes.
#
# Storage is pieces: aligned blocks of units, a unit being the trailing block
# of the shortest prefix of the cell axes that holds at most target cells.
# Pieces depend on the cell axes and the target alone, so two tensors of one
# shape have the same pieces and a reshape that keeps the cell axes keeps them.
import Base
import ./tensor_view.bend as Views

type Position is Data:
  Innermost{}
  Outermost{}

type Layout is Data:
  Layout{axis: Nat,position: Position}

def is_outermost(position: Position) -> Bool:
  match position:
    case Innermost{}: False{}
    case Outermost{}: True{}

def same_position(left: Position,right: Position) -> Bool:
  match left right:
    case Innermost{} Innermost{}: True{}
    case Outermost{} Outermost{}: True{}
    case left right: False{}

# The pieces of a grid. The cell axes split into a prefix and a unit: outer
# indices of the prefix axes but its last, extent indices of its last, and
# unit_cells cells in a unit. A piece is an aligned block of piece_units
# indices of the last prefix axis inside one outer index, per_outer of them
# for each outer index, so every piece is a box of the cell axes. prefix is the
# number of prefix axes.
type Grid is Data:
  Grid{count: U32,prefix: U32,outer: U32,extent: U32,unit_cells: U32,piece_units: U32,per_outer: U32,pieces: U32}

def groups(+lanes: U32) -> U32:
  ((lanes + 7) / 8 : U32)

# The piece size of every tensor: one piece of the cheapest pass costs far more
# than dispatching it as a task and stays in the second level cache across a
# fused chain.
def piece_target() -> U32:
  4096

def product(extents: List<&2,U32>) -> U32:
  match extents:
    case Nil{}: 1
    case Con{extent,rest}: (extent * product(rest) : U32)

# The row-major strides of extents, every one the product of the axes after
# it, and the product of them all, in one walk.
type Strided is Data:
  Strided{strides: List<&2,U32>,product: U32}

def stride_before(+extent: U32,inner: Strided) -> Strided:
  Strided{strides,+product} = inner
  Strided{product <> strides,(extent * product : U32)}

def strided(extents: List<&2,U32>) -> Strided:
  match extents:
    case Nil{}: Strided{[],1}
    case Con{+extent,rest}: stride_before(extent,strided(rest))

def strides_listed(found: Strided) -> List<&2,U32>:
  Strided{strides,product} = found
  strides

def strides(extents: List<&2,U32>) -> List<&2,U32>:
  strides_listed(strided(extents))

def removed(extents: List<&2,U32>,axis: Nat) -> List<&2,U32>:
  match extents axis:
    case Nil{} axis: []
    case Con{extent,rest} 0n: rest
    case Con{extent,rest} 1n+next: extent <> removed(rest,next)

def inserted(extents: List<&2,U32>,axis: Nat,+value: U32) -> List<&2,U32>:
  match extents axis:
    case extents 0n: value <> extents
    case Nil{} 1n+next: [value]
    case Con{extent,rest} 1n+next: extent <> inserted(rest,next,value)

def lanes_at(extents: List<&2,U32>,axis: Nat) -> U32:
  match extents axis:
    case Nil{} axis: 1
    case Con{extent,rest} 0n: extent
    case Con{extent,rest} 1n+next: lanes_at(rest,next)

def placed(lane: Bool,position: Position,others: List<&2,U32>,+groups: U32) -> List<&2,U32>:
  match lane position:
    case False{} position: others
    case True{} Innermost{}: List.append(&2,U32,others,[groups])
    case True{} Outermost{}: groups <> others

# The cell axes of a logical shape. The lane axis of a shape of rank 0 is a
# unit axis past its end: its one cell has no cell axes.
def cell_shape(+shape: List<&2,U32>,+layout: Layout) -> List<&2,U32>:
  Layout{+axis,+position} = layout
  placed(Nat.is_lt(axis,List.length(&2,U32,shape)),position,removed(shape,axis),groups(lanes_at(shape,axis)))

def without_last(extents: List<&2,U32>) -> List<&2,U32>:
  match extents:
    case Nil{}: []
    case Con{extent,Nil{}}: []
    case Con{extent,rest}: extent <> without_last(rest)

def last_of(extents: List<&2,U32>) -> U32:
  match extents:
    case Nil{}: 1
    case Con{extent,Nil{}}: extent
    case Con{extent,rest}: last_of(rest)

def others_of(position: Position,cells: List<&2,U32>) -> List<&2,U32>:
  match position cells:
    case Innermost{} cells: without_last(cells)
    case Outermost{} Nil{}: []
    case Outermost{} Con{groups,rest}: rest

# The logical shape of cell axes: the lane axis comes back with lanes values.
# Cell axes of a lane axis past the end of the shape are the shape itself.
def logical_when(within: Bool,cells: List<&2,U32>,+axis: Nat,+position: Position,+lanes: U32) -> List<&2,U32>:
  match within:
    case True{}: inserted(others_of(position,cells),axis,lanes)
    case False{}: cells

def logical_shape(+cells: List<&2,U32>,+layout: Layout,+lanes: U32) -> List<&2,U32>:
  Layout{+axis,+position} = layout
  logical_when(Nat.is_lt(axis,List.length(&2,U32,cells)),cells,axis,position,lanes)

def outer_count(+cells: List<&2,U32>,+position: Position) -> U32:
  product(others_of(position,cells))

def groups_of(+position: Position,+cells: List<&2,U32>) -> U32:
  match position cells:
    case Innermost{} cells: last_of(cells)
    case Outermost{} Nil{}: 1
    case Outermost{} Con{groups,rest}: groups

# The unit of cell axes: the axes after the shortest prefix whose rest holds at
# most target cells. The last axis is never split. outer is the product of the
# prefix axes but its last, extent the last.
type Units is Data:
  Units{prefix: U32,outer: U32,extent: U32,unit_cells: U32}

# The cells of the axes from the first of extents on, by its stride.
def cells_from(extents: List<&2,U32>,strides: List<&2,U32>) -> U32:
  match extents strides:
    case Con{+extent,rest} Con{+stride,more}: (extent * stride : U32)
    case extents strides: 1

# strides are the strides of extents.
def units_from(extents: List<&2,U32>,strides: List<&2,U32>,fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32) -> Units:
  match extents strides fits:
    case extents strides True{}: Units{prefix,outer,extent,cells_from(extents,strides)}
    case Nil{} strides False{}: Units{prefix,outer,extent,1}
    case Con{last,Nil{}} strides False{}: Units{prefix,outer,extent,last}
    case Con{axis,+rest} Con{+stride,more} False{}: units_from(rest,more,U32.is_le(stride,target),(prefix + 1 : U32),(outer * extent : U32),axis,target)
    case extents strides False{}: Units{prefix,outer,extent,1}

# The most indices of the last prefix axis whose cells fit the target, at
# least one and at most the extent: storage comes in powers of two, and the
# target is one, so a piece fills its storage but for less than a unit.
def piece_units(+unit_cells: U32,+extent: U32,+target: U32) -> U32:
  U32.max(U32.min(extent,(target / U32.max(unit_cells,1) : U32)),1)

def divided_up(+count: U32,+size: U32) -> U32:
  (count / U32.max(size,1) + Bool.pick(U32,U32.is_zero((count % U32.max(size,1) : U32)),0,1) : U32)

def grid_of_units(+found: Units,+target: U32) -> Grid:
  Units{+prefix,+outer,+extent,+unit_cells} = found
  +block = piece_units(unit_cells,extent,target)
  +per_outer = divided_up(extent,block)
  Grid{(outer * extent * unit_cells : U32),prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)}

def grid_strided(+cells: List<&2,U32>,+target: U32,found: Strided) -> Grid:
  Strided{strides,+product} = found
  grid_of_units(units_from(cells,strides,U32.is_le(product,target),0,1,1,target),target)

def grid(+cells: List<&2,U32>,+target: U32) -> Grid:
  grid_strided(cells,target,strided(cells))

def same_grid(left: Grid,right: Grid) -> Bool:
  Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = left
  Grid{other_count,other_prefix,other_outer,other_extent,other_unit_cells,other_piece_units,other_per_outer,other_pieces} = right
  U32.is_eq(count,other_count) && U32.is_eq(outer,other_outer) && U32.is_eq(extent,other_extent) && U32.is_eq(unit_cells,other_unit_cells) && U32.is_eq(piece_units,other_piece_units)

# The first cell and the cells of a piece, and the piece of a cell.
def start(+grid: Grid,+piece: U32) -> U32:
  Grid{count,prefix,outer,+extent,+unit_cells,+piece_units,+per_outer,pieces} = grid
  +per = U32.max(per_outer,1)
  ((piece / per * extent + piece % per * piece_units) * unit_cells : U32)

def span(+grid: Grid,+piece: U32) -> U32:
  Grid{+count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  (U32.min(start(grid,(piece + 1 : U32)),count) - U32.min(start(grid,piece),count) : U32)

def piece_of(+grid: Grid,+index: U32) -> U32:
  Grid{count,prefix,outer,+extent,+unit_cells,+piece_units,+per_outer,pieces} = grid
  +image = U32.max((extent * unit_cells : U32),1)
  (index / image * per_outer + index % image / U32.max((piece_units * unit_cells : U32),1) : U32)

# The pieces after the first of the pieces that hold cells
# [first, first + count), count at least one.
def later_pieces(+grid: Grid,+first: U32,+count: U32) -> Nat:
  U32.to_nat((piece_of(grid,(first + count - 1 : U32)) - piece_of(grid,first) : U32))

# The block of a piece along a cell axis of the given extent: one index on a
# prefix axis but the last, piece_units on the last, the whole axis after it.
def block(+grid: Grid,+axis: U32,+extent: U32) -> U32:
  Grid{count,+prefix,outer,last,unit_cells,+piece_units,per_outer,pieces} = grid
  Bool.pick(U32,U32.is_lt((axis + 1 : U32),prefix),1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),piece_units,extent))

def count_of(grid: Grid) -> U32:
  Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  count

# The pieces are equal blocks of the cell order, the last one perhaps shorter:
# one outer index, or whole blocks of every outer index.
def flat_blocks(+grid: Grid) -> Bool:
  Grid{count,prefix,+outer,+extent,unit_cells,+piece_units,per_outer,pieces} = grid
  U32.is_le(outer,1) || U32.is_zero((extent % U32.max(piece_units,1) : U32))

def piece_cells(+grid: Grid) -> U32:
  Grid{count,prefix,outer,extent,+unit_cells,+piece_units,per_outer,pieces} = grid
  (unit_cells * piece_units : U32)

# Two grids of as many cells cut them at the same places.
def same_pieces(+left: Grid,+right: Grid) -> Bool:
  U32.is_eq(count_of(left),count_of(right)) && flat_blocks(left) && flat_blocks(right) && U32.is_eq(piece_cells(left),piece_cells(right))

def pieces_of(grid: Grid) -> U32:
  Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  pieces

# The coordinates of a cell index over the cell axes.
def coordinates_by(extents: List<&2,U32>,strides: List<&2,U32>,+index: U32) -> List<&2,U32>:
  match extents strides:
    case Con{+extent,rest} Con{+stride,more}:
      +inner = U32.max(stride,1)
      (index / inner % U32.max(extent,1) : U32) <> coordinates_by(rest,more,(index % inner : U32))
    case extents strides: []

def coordinates(+extents: List<&2,U32>,+index: U32) -> List<&2,U32>:
  coordinates_by(extents,strides(extents),index)

# The extents of the box of a piece of count cells: one index of a prefix axis
# but the last, its units of the last, every index after it.
def box(extents: List<&2,U32>,+axis: U32,+prefix: U32,+units: U32) -> List<&2,U32>:
  match extents:
    case Nil{}: []
    case Con{+extent,rest}:
      Bool.pick(U32,U32.is_lt((axis + 1 : U32),prefix),1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),units,extent)) <> box(rest,(axis + 1 : U32),prefix,units)

def box_of(+grid: Grid,+cells: List<&2,U32>,+count: U32) -> List<&2,U32>:
  Grid{total,+prefix,outer,extent,+unit_cells,piece_units,per_outer,pieces} = grid
  box(cells,0,prefix,(count / U32.max(unit_cells,1) : U32))

def prepended(+axis: Views.Axis,view: Views.View) -> Views.View:
  Views.View{axes,offset} = view
  Views.View{axis <> axes,offset}

# One walk over the axes: a View used twice would make the compiler share
# every View constructor.
def restricted(axes: List<&2,Views.Axis>,coordinates: List<&2,U32>,extents: List<&2,U32>,+offset: U32) -> Views.View:
  match axes coordinates extents:
    case Con{Views.Axis{extent,+stride},axes} Con{+coordinate,coordinates} Con{+box_extent,extents}:
      prepended(Views.Axis{box_extent,stride},restricted(axes,coordinates,extents,(offset + coordinate * stride : U32)))
    case axes coordinates extents: Views.View{[],offset}

# The contiguous axes of extents.
def axes_by(extents: List<&2,U32>,strides: List<&2,U32>) -> List<&2,Views.Axis>:
  match extents strides:
    case Con{+extent,rest} Con{+stride,more}: Views.Axis{extent,stride} <> axes_by(rest,more)
    case extents strides: []

def dense_axes(+extents: List<&2,U32>) -> List<&2,Views.Axis>:
  axes_by(extents,strides(extents))

# A piece of count cells read in its own order over its box.
def box_view(+grid: Grid,+cells: List<&2,U32>,+count: U32) -> Views.View:
  Views.View{dense_axes(box_of(grid,cells,count)),0}

# A view of the cells of shape cells read only over the box of the piece that
# starts at first and holds count cells, in the piece's own order.
def piece_view(+grid: Grid,+cells: List<&2,U32>,+first: U32,+count: U32,view: Views.View) -> Views.View:
  Views.View{axes,+offset} = view
  restricted(axes,coordinates(cells,first),box_of(grid,cells,count),offset)
