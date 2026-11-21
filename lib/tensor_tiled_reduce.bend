# Reductions of tiled tensors, read in the layout of the input. With the
# lanes on a kept axis every lane reduces its own values: the cells reduce
# their cell axes with eight accumulators a cell, and every piece of the
# result sets the lanes past the extent to +0. With the lanes on a reduced
# axis the result holds one value a cell, its lanes then moved to the longest
# kept axis. A fold reads the values in increasing index: along the cells of
# a lane axis reduced last every cell steps its lanes in order. A sum
# follows Partials: the rows of the last axis folded by Sum and the rows
# merged in the tree of the reduced axes. With the lanes on a reduced last
# axis a row is its cells in order and every lane of a cell that holds a
# value adds it to one lane of the sum. Lanes on another reduced axis first
# move to the last axis, the groups innermost.
import Base
import ./kernel_tile.bend as Tile
import ./math_sum.bend as Sum
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_layout.bend as Layout
import ./tensor_operations.bend as Operations
import ./tensor_partials.bend as Partials
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./traversal_array.bend as Traversal
import ./proofs/traversal_witness.bend as TraversalWitness

# ---- Eight values at once ----

# One accumulator a lane.
type Folds<-Accumulator: Data> is Data:
  Folds{a: Accumulator,b: Accumulator,c: Accumulator,d: Accumulator,e: Accumulator,f: Accumulator,g: Accumulator,h: Accumulator}

def started(~Accumulator: Data,+initial: Accumulator) -> Folds<Accumulator>:
  Folds{initial,initial,initial,initial,initial,initial,initial,initial}

def stepped(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,folds: Folds<Accumulator>,tile: Tile.Tile8) -> Folds<Accumulator>:
  Folds{a,b,c,d,e,f,g,h} = folds
  Tile.Tile{i,j,k,l,m,n,o,p} = tile
  Folds{operation(a,i),operation(b,j),operation(c,k),operation(d,l),operation(e,m),operation(f,n),operation(g,o),operation(h,p)}

def finished(~Accumulator: Data,~finish: Accumulator -> F32,folds: Folds<Accumulator>) -> Tile.Tile8:
  Folds{a,b,c,d,e,f,g,h} = folds
  Tile.Tile{finish(a),finish(b),finish(c),finish(d),finish(e),finish(f),finish(g),finish(h)}

def closed(~Row: Data,~Partial: Data,~close: Row -> Partial,folds: Folds<Row>) -> Folds<Partial>:
  Folds{a,b,c,d,e,f,g,h} = folds
  Folds{close(a),close(b),close(c),close(d),close(e),close(f),close(g),close(h)}

def merged(~Partial: Data,~merge: Partial -> Partial -> Partial,left: Folds<Partial>,right: Folds<Partial>) -> Folds<Partial>:
  Folds{a,b,c,d,e,f,g,h} = left
  Folds{i,j,k,l,m,n,o,p} = right
  Folds{merge(a,i),merge(b,j),merge(c,k),merge(d,l),merge(e,m),merge(f,n),merge(g,o),merge(h,p)}

# The operation on a cell of the unit axis reads its first lane; the finished
# value is a cell of the unit axis.
def lane_operation(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,accumulator: Accumulator,tile: Tile.Tile8) -> Accumulator:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  operation(accumulator,a)

def unit_finish(~Accumulator: Data,~finish: Accumulator -> F32,accumulator: Accumulator) -> Tile.Tile8:
  Tiled.unit_tile(finish(accumulator))

# ---- The lanes ----

# The place among the kept axes of the one of the largest extent, the later
# among equal extents, or the count of kept axes when none is kept.
def longest_kept(kept: List<&2,Nat>,+shape: List<&2,U32>,+place: Nat,+best: Nat,+extent: U32) -> Nat:
  match kept:
    case []: best
    case axis <> rest:
      +own = Grid.lanes_at(shape,axis)
      +longer = U32.is_le(extent,own)
      longest_kept(rest,shape,1n+place,Bool.pick(Nat,longer,place,best),Bool.pick(U32,longer,own,extent))

# The flags but the one of an axis.
# The flags of the other axes with the flag of the groups at their position.
def flags_placed(position: Grid.Position,flags: List<&2,Bool>,+flag: Bool) -> List<&2,Bool>:
  match position:
    case Grid.Innermost{}: List.append(&2,Bool,flags,[flag])
    case Grid.Outermost{}: flag <> flags

# The axes from index on whose flag is set.
def flagged_words(flags: List<&2,Bool>,+index: U32) -> List<&2,U32>:
  match flags:
    case []: []
    case flag <> rest:
      match flag:
        case True{}: index <> flagged_words(rest,(index + 1 : U32))
        case False{}: flagged_words(rest,(index + 1 : U32))

# The reduced cell axes with the lanes on lane: the reduced axes but the lane
# axis in order, and the groups at their position when the lane axis is
# reduced.
def cell_axes(+reduced: List<&2,Nat>,+rank: Nat,+lane: Nat,position: Grid.Position) -> List<&2,U32>:
  +flags = Layout.reduced_flags(rank,0n,reduced)
  flagged_words(flags_placed(position,Partials.flag_removed(flags,lane),Partials.flag_at(flags,lane)),0)

# The place of an axis among the kept axes, from place on.
def kept_place(kept: List<&2,Nat>,+axis: Nat,+place: Nat) -> Nat:
  match kept:
    case []: place
    case found <> rest: Bool.pick(Nat,Nat.is_eq(found,axis),place,kept_place(rest,axis,1n+place))

def trimmed(pair: Tiled.Tiled & TiledMap.Padding) -> Tiled.Tiled:
  (Tiled.Tiled{cells,axis,lanes,position},padding) = pair
  Tiled.Tiled{Tensors.map_pieces(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,0.0,cells),axis,lanes,position}

# The result of a reduction with the lanes on a kept axis: lanes past the
# extent hold what the accumulators made of +0, set to +0 here.
def kept_result(tiled: Tiled.Tiled) -> Tiled.Tiled:
  trimmed(TiledMap.padding_cells(tiled))

def kept_cells(+reduced: List<&2,Nat>,+rank: Nat,+lane: Nat,+lanes: U32,+position: Grid.Position,cells: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  kept_result(Tiled.Tiled{cells,kept_place(Layout.axes_where(rank,0n,reduced,False{}),lane,0n),lanes,position})

# ---- Folds ----

def folded_kept(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,unit: Bool,name: String,+operations: F32,+initial: Accumulator,
  +axes: List<&2,U32>,+rank: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match unit:
    case True{}:
      Tiled.Tiled{cells,lane,lanes,position} = tiled
      Tiled.unit(Layout.reduce_axes(~Tile.Tile8,~Accumulator,~lane_operation(~Accumulator,~operation),~unit_finish(~Accumulator,~finish),name,operations,initial,cells,axes))
    case False{}:
      Tiled.Tiled{cells,+lane,+lanes,+position} = tiled
      kept_cells(Layout.naturals(axes),rank,lane,lanes,position,
        Layout.reduce_axes(~Tile.Tile8,~Folds<Accumulator>,~stepped(~Accumulator,~operation),~finished(~Accumulator,~finish),name,(operations * 8.0 : F32),started(~Accumulator,initial),
          cells,cell_axes(Layout.naturals(axes),rank,lane,position)))

# How a reduction reads the lanes of its input: one value a cell, the lanes
# on a kept axis, along a reduced lane axis, or across it.
type Route is Data:
  Single{}
  Kept{}
  Along{}
  Across{}

def route_of(+lane: Nat,+rank: Nat,+reduced: List<&2,Nat>,along: Bool) -> Route:
  Bool.pick(Route,Nat.is_lt(lane,rank),Bool.pick(Route,Views.contains_index(reduced,lane),Bool.pick(Route,along,Along{},Across{}),Kept{}),Single{})

# Whether no flag is set.
def none_flagged(flags: List<&2,Bool>) -> Bool:
  match flags:
    case []: True{}
    case flag <> rest: Bool.not(flag) && none_flagged(rest)

# Whether no flag after the one of an axis is set.
def none_after(flags: List<&2,Bool>,axis: Nat) -> Bool:
  match flags axis:
    case Nil{} axis: True{}
    case Con{flag,rest} 0n: none_flagged(rest)
    case Con{flag,rest} 1n+next: none_after(rest,next)

def result_arranged(kept: Bool,+axis: Nat,result: Tiled.Tiled) -> Tiled.Tiled:
  match kept:
    case True{}: Tiled.arranged(result,axis,Grid.Innermost{})
    case False{}: result

# A result of one value a cell with its lanes on the longest kept axis, or
# on the unit axis when no axis is kept.
def result_lanes(+shape: List<&2,U32>,+reduced: List<&2,Nat>,result: Tiled.Tiled) -> Tiled.Tiled:
  +rank = List.length(&2,U32,shape)
  +kept = Layout.axes_where(rank,0n,reduced,False{})
  +count = List.length(&2,Nat,kept)
  +longest = longest_kept(kept,shape,0n,count,0)
  result_arranged(Nat.is_lt(longest,count),longest,result)

# A line along the lane axis: its cells in order, each stepping the lanes
# that hold values. The first lane of a cell is eight past that of the cell
# before, and 0 again after the last group.
type LaneLine<-Accumulator: Data> is Data:
  LaneLine{value: Accumulator,first: U32,lanes: U32}

def lane_applied(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,valid: Bool,value: Accumulator,lane: F32) -> Accumulator:
  match valid:
    case True{}: operation(value,lane)
    case False{}: value

# The eight lanes of a tile, one after another.
def lanes_all(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,value: Accumulator,tile: Tile.Tile8) -> Accumulator:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  operation(operation(operation(operation(operation(operation(operation(operation(value,a),b),c),d),e),f),g),h)

# The lanes of a tile that hold values, one after another.
def lanes_masked(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+first: U32,+lanes: U32,value: Accumulator,tile: Tile.Tile8) -> Accumulator:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  lane_applied(~Accumulator,~operation,U32.is_lt((first + 7 : U32),lanes),
    lane_applied(~Accumulator,~operation,U32.is_lt((first + 6 : U32),lanes),
      lane_applied(~Accumulator,~operation,U32.is_lt((first + 5 : U32),lanes),
        lane_applied(~Accumulator,~operation,U32.is_lt((first + 4 : U32),lanes),
          lane_applied(~Accumulator,~operation,U32.is_lt((first + 3 : U32),lanes),
            lane_applied(~Accumulator,~operation,U32.is_lt((first + 2 : U32),lanes),
              lane_applied(~Accumulator,~operation,U32.is_lt((first + 1 : U32),lanes),
                lane_applied(~Accumulator,~operation,U32.is_lt(first,lanes),value,a),b),c),d),e),f),g),h)

def lanes_whole(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,whole: Bool,+first: U32,+lanes: U32,value: Accumulator,tile: Tile.Tile8) -> Accumulator:
  match whole:
    case True{}: lanes_all(~Accumulator,~operation,value,tile)
    case False{}: lanes_masked(~Accumulator,~operation,first,lanes,value,tile)

# Whether all eight lanes from first on hold values.
def all_hold(+first: U32,+lanes: U32) -> Bool:
  U32.is_lt(first,lanes) && U32.is_lt((first + 1 : U32),lanes) && U32.is_lt((first + 2 : U32),lanes) && U32.is_lt((first + 3 : U32),lanes) &&
    U32.is_lt((first + 4 : U32),lanes) && U32.is_lt((first + 5 : U32),lanes) && U32.is_lt((first + 6 : U32),lanes) && U32.is_lt((first + 7 : U32),lanes)

# A cell steps the lanes that hold values: all eight without a test when
# each of them does.
def lanes_stepped(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,+first: U32,+lanes: U32,value: Accumulator,tile: Tile.Tile8) -> Accumulator:
  lanes_whole(~Accumulator,~operation,all_hold(first,lanes),first,lanes,value,tile)

def line_step(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,line: LaneLine<Accumulator>,tile: Tile.Tile8) -> LaneLine<Accumulator>:
  LaneLine{value,+first,+lanes} = line
  +next = (first + 8 : U32)
  LaneLine{lanes_stepped(~Accumulator,~operation,first,lanes,value,tile),Bool.pick(U32,U32.is_lt(next,lanes),next,0),lanes}

def line_finish(~Accumulator: Data,~finish: Accumulator -> F32,line: LaneLine<Accumulator>) -> Tile.Tile8:
  LaneLine{value,first,lanes} = line
  Tiled.unit_tile(finish(value))

# Lanes along a reduced axis with no reduced axis after it, the groups
# innermost: every line steps its cells in order.
def folded_along(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,name: String,+operations: F32,+initial: Accumulator,
  +shape: List<&2,U32>,+axes: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{cells,+lane,+lanes,+position} = tiled
  +rank = List.length(&2,U32,shape)
  result_lanes(shape,Layout.naturals(axes),Tiled.unit(
    Layout.reduce_axes(~Tile.Tile8,~LaneLine<Accumulator>,~line_step(~Accumulator,~operation),~line_finish(~Accumulator,~finish),name,(operations * 8.0 : F32),
      LaneLine{initial,0,lanes},cells,cell_axes(Layout.naturals(axes),rank,lane,position))))

# Lanes on the last axis, the groups innermost: along it when it is reduced,
# otherwise on a kept axis.
def folded_last(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,reduced: Bool,name: String,+operations: F32,+initial: Accumulator,
  +shape: List<&2,U32>,+axes: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match reduced:
    case True{}: folded_along(~Accumulator,~operation,~finish,name,operations,initial,shape,axes,tiled)
    case False{}: folded_kept(~Accumulator,~operation,~finish,False{},name,operations,initial,axes,List.length(&2,U32,shape),tiled)

def folded(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,route: Route,name: String,+operations: F32,+initial: Accumulator,
  +shape: List<&2,U32>,+axes: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match route:
    case Single{}: result_lanes(shape,Layout.naturals(axes),folded_kept(~Accumulator,~operation,~finish,True{},name,operations,initial,axes,List.length(&2,U32,shape),tiled))
    case Kept{}: folded_kept(~Accumulator,~operation,~finish,False{},name,operations,initial,axes,List.length(&2,U32,shape),tiled)
    case Along{}: folded_along(~Accumulator,~operation,~finish,name,operations,initial,shape,axes,tiled)
    case Across{}:
      +last = Nat.sub(List.length(&2,U32,shape),1n)
      folded_last(~Accumulator,~operation,~finish,Views.contains_index(Layout.naturals(axes),last),name,operations,initial,shape,axes,Tiled.arranged(tiled,last,Grid.Innermost{}))

# Valid axes reduce the lanes where they lie. Invalid axes, an empty tensor
# and a failed tensor reduce on the unit axis, which reports them.
def folded_valid(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,valid: Bool,name: String,+operations: F32,+initial: Accumulator,
  +shape: List<&2,U32>,+axes: List<&2,U32>,started: Tiled.Tiled & Nat) -> Tiled.Tiled:
  match valid:
    case True{}:
      (tiled,+lane) = started
      Tiled.Tiled{cells,own,lanes,+position} = tiled
      +rank = List.length(&2,U32,shape)
      +reduced = Layout.naturals(axes)
      folded(~Accumulator,~operation,~finish,route_of(lane,rank,reduced,Bool.not(Grid.is_outermost(position)) && none_after(Layout.reduced_flags(rank,0n,reduced),lane)),name,operations,initial,shape,axes,
        Tiled.Tiled{cells,own,lanes,position})
    case False{}:
      (tiled,lane) = started
      folded(~Accumulator,~operation,~finish,Single{},name,operations,initial,shape,axes,Tiled.along(tiled,List.length(&2,U32,shape)))

# A valid list of axes of a usable, nonempty tensor.
def valid_axes(status: Tensors.Status,+shape: List<&2,U32>,+axes: List<&2,U32>) -> Bool:
  Tensors.usable(status) && Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)) && Bool.not(Shapes.has_zero(shape))

# Layout.reduce_axes of a tiled tensor: every value folded in increasing
# index. An invalid list of axes, an empty tensor and a failed tensor reduce
# the cells on the unit axis, which reports them.
def reduce_axes(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,name: String,+operations: F32,+initial: Accumulator,
  tiled: Tiled.Tiled,+axes: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    folded_valid(~Accumulator,~operation,~finish,valid_axes(status,shape,axes),name,operations,initial,shape,axes,Tiled.lane_axis_of(tiled)))

def every_axis(remaining: Nat,+index: U32) -> List<&2,U32>:
  match remaining:
    case 0n: []
    case 1n+rest: index <> every_axis(rest,(index + 1 : U32))

def reduce_all(~Accumulator: Data,~operation: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,name: String,+operations: F32,+initial: Accumulator,
  tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    reduce_axes(~Accumulator,~operation,~finish,name,operations,initial,tiled,every_axis(List.length(&2,U32,shape),0)))

# ---- Sums ----

def parts_zero() -> Sum.Parts:
  Sum.Parts{0.0,0.0}

def lane_rounded(parts: Sum.Parts) -> Tile.Tile8:
  Tiled.unit_tile(Sum.rounded(parts))

def row_kept(lo: Tile.Tile8,parts: Sum.TileParts) -> Sum.TileParts:
  Sum.TileParts{hi,error} = parts
  Sum.TileParts{hi,Tile.plus(lo,error)}

# A row of cells: lane j of the parts adds the values of lane j of the sum.
def row_added(parts: Sum.TileParts,value: Tile.Tile8) -> Sum.TileParts:
  Sum.TileParts{hi,lo} = parts
  row_kept(lo,Sum.tile_two_sum(hi,value))

def row_closed(parts: Sum.TileParts) -> Sum.Parts:
  Sum.TileParts{hi,lo} = parts
  Sum.parts(Sum.Lanes{hi,lo,0})

def sum_begun(context: Unit) -> Sum.Lanes:
  Sum.zero()

# A row along the lane axis: its cells in order, lane j of the parts the sum
# of lane j of Sum, every cell adding only its lanes that hold values. The
# first lane of a cell is eight past that of the cell before, and 0 again
# after the last group.
type SumLine is Data:
  SumLine{parts: Sum.TileParts,first: U32,lanes: U32}

# The lanes of added below count, the others of kept.
def lanes_below(+count: U32,added: Tile.Tile8,kept: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = added
  Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = kept
  Tile.Tile{Bool.pick(F32,U32.is_lt(0,count),a0,k0),Bool.pick(F32,U32.is_lt(1,count),a1,k1),Bool.pick(F32,U32.is_lt(2,count),a2,k2),Bool.pick(F32,U32.is_lt(3,count),a3,k3),
    Bool.pick(F32,U32.is_lt(4,count),a4,k4),Bool.pick(F32,U32.is_lt(5,count),a5,k5),Bool.pick(F32,U32.is_lt(6,count),a6,k6),Bool.pick(F32,U32.is_lt(7,count),a7,k7)}

def parts_below(+count: U32,added: Sum.TileParts,kept: Sum.TileParts) -> Sum.TileParts:
  Sum.TileParts{added_hi,added_lo} = added
  Sum.TileParts{kept_hi,kept_lo} = kept
  Sum.TileParts{lanes_below(count,added_hi,kept_hi),lanes_below(count,added_lo,kept_lo)}

# A cell whose lanes all hold values adds them at once.
def cell_added(whole: Bool,+count: U32,+parts: Sum.TileParts,value: Tile.Tile8) -> Sum.TileParts:
  match whole:
    case True{}: row_added(parts,value)
    case False{}: parts_below(count,row_added(parts,value),parts)

def line_added(line: SumLine,tile: Tile.Tile8) -> SumLine:
  SumLine{+parts,+first,+lanes} = line
  +count = (lanes - first : U32)
  +next = (first + 8 : U32)
  SumLine{cell_added(U32.is_le(8,count),count,parts,tile),Bool.pick(U32,U32.is_lt(next,lanes),next,0),lanes}

def line_begun(lanes: U32) -> SumLine:
  SumLine{Sum.TileParts{Tile.filled(0.0),Tile.filled(0.0)},0,lanes}

def line_closed(line: SumLine) -> Sum.Parts:
  SumLine{parts,first,lanes} = line
  row_closed(parts)

# The groups of lanes on a kept axis lie innermost when the lanes are on the
# last axis, every cell then a row, otherwise outermost, the last axis then
# the last of the cells too.
def kept_position(+lane: Nat,+rank: Nat) -> Grid.Position:
  Bool.pick(Grid.Position,Nat.is_eq(lane,Nat.sub(rank,1n)),Grid.Innermost{},Grid.Outermost{})

# Lanes on a kept axis, the groups at kept_position.
def summed_kept(name: String,+axes: List<&2,U32>,+rank: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{cells,+lane,+lanes,+position} = tiled
  +reduced = Layout.naturals(axes)
  kept_cells(reduced,rank,lane,lanes,position,
    Partials.reduce(~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),
      ~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,
      ~Sum.tile_rounded_total,name,48.0,Unit{},Sum.tile_parts_zero(),cells,cell_axes(reduced,rank,lane,position)))

# Lanes along the reduced last axis, the groups innermost: a row is its cells
# in order.
def summed_along(name: String,+shape: List<&2,U32>,+axes: List<&2,U32>,+rank: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{cells,+lane,+lanes,position} = tiled
  result_lanes(shape,Layout.naturals(axes),Tiled.unit(Partials.reduce(~Tile.Tile8,~U32,~SumLine,~Sum.Parts,~line_begun,~Operations.fold_range(~Tile.Tile8,~SumLine,~line_added),~line_closed,~Sum.merge,~lane_rounded,name,6.0,lanes,
    parts_zero(),cells,cell_axes(Layout.naturals(axes),rank,lane,Grid.Innermost{}))))

# Lanes on the last axis, the groups innermost: along it when it is reduced,
# otherwise on a kept axis.
def summed_last(reduced: Bool,name: String,+shape: List<&2,U32>,+axes: List<&2,U32>,+rank: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match reduced:
    case True{}: summed_along(name,shape,axes,rank,tiled)
    case False{}: summed_kept(name,axes,rank,tiled)

def summed(route: Route,name: String,+shape: List<&2,U32>,+axes: List<&2,U32>,+rank: Nat,+lane: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match route:
    case Single{}:
      result_lanes(shape,Layout.naturals(axes),Tiled.unit(Partials.reduce(~Tile.Tile8,~Unit,~Sum.Lanes,~Sum.Parts,~sum_begun,~Operations.fold_range(~Tile.Tile8,~Sum.Lanes,~lane_operation(~Sum.Lanes,~Sum.step)),~Sum.parts,~Sum.merge,~lane_rounded,
        name,6.0,Unit{},parts_zero(),Tiled.cells_of(tiled),axes)))
    case Kept{}: summed_kept(name,axes,rank,Tiled.arranged(tiled,lane,kept_position(lane,rank)))
    case Along{}: summed_along(name,shape,axes,rank,Tiled.arranged(tiled,lane,Grid.Innermost{}))
    case Across{}:
      +last = Nat.sub(rank,1n)
      summed_last(Views.contains_index(Layout.naturals(axes),last),name,shape,axes,rank,Tiled.arranged(tiled,last,Grid.Innermost{}))

# Valid axes sum the lanes where they lie. Invalid axes, an empty tensor and
# a failed tensor sum on the unit axis, which reports them.
def sum_valid(valid: Bool,name: String,+shape: List<&2,U32>,+axes: List<&2,U32>,started: Tiled.Tiled & Nat) -> Tiled.Tiled:
  match valid:
    case True{}:
      (tiled,+lane) = started
      +rank = List.length(&2,U32,shape)
      summed(route_of(lane,rank,Layout.naturals(axes),Nat.is_eq(lane,Nat.sub(rank,1n))),name,shape,axes,rank,lane,tiled)
    case False{}:
      (tiled,lane) = started
      +rank = List.length(&2,U32,shape)
      summed(Single{},name,shape,axes,rank,rank,Tiled.along(tiled,rank))

# The sums of the listed axes in the order of Partials. An invalid list of
# axes, an empty tensor and a failed tensor sum the cells on the unit axis,
# which reports them.
def sum_axes(name: String,tiled: Tiled.Tiled,+axes: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    sum_valid(valid_axes(status,shape,axes),name,shape,axes,Tiled.lane_axis_of(tiled)))

def sum_all(name: String,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers => sum_axes(name,tiled,every_axis(List.length(&2,U32,shape),0)))
