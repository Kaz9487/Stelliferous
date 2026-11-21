# Gathers of tiled tensors. A view of the logical shape reads the cells when
# one target axis reads the lane axis forward one index a step from a group
# boundary, every other target axis repeats or steps whole cells along one
# axis, and no axis runs past its extent: the cells move once through the view
# of the cells, the groups innermost, and every piece sets the lanes past the
# extent of the result to +0. A view that reads every cell in order keeps the
# cells. A view that does not keep the lanes of the input keeps them on the
# axis it reads at the lane axis of its target, or else on the longest axis it
# keeps whole, and the input takes its lanes there. A view that keeps no axis
# whole reads the input with its lanes on the unit axis after the last, one
# lane a cell.
import Base
import ./kernel_tile.bend as Tile
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_tasks.bend as Tasks
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap

# ---- The view of the cells ----

# The lanes of an input: their axis and its extent, the elements after it,
# the cells of one group and where the groups lie.
type Lanes is Data:
  Lanes{axis: Nat,extent: U32,inner: U32,outer: U32,position: Grid.Position}

def lanes_around(around: Maybe<&2,Tiled.Around>,+axis: Nat,+position: Grid.Position) -> Lanes:
  match around:
    case Some{Tiled.Around{+before,+extent,after}}:
      +inner = Tiled.product(after)
      Lanes{axis,extent,inner,(before * inner : U32),position}
    case None{}: Lanes{axis,1,1,1,position}

def lanes_of(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> Lanes:
  lanes_around(Tiled.around(shape,axis,1),axis,position)

def lane_axis(lanes: Lanes) -> Nat:
  Lanes{axis,extent,inner,outer,position} = lanes
  axis

def lane_inner(lanes: Lanes) -> U32:
  Lanes{axis,extent,inner,outer,position} = lanes
  inner

def cell_at(position: Grid.Position,+outer_index: U32,+group: U32,+groups: U32,+outer: U32) -> U32:
  match position:
    case Grid.Innermost{}: (outer_index * groups + group : U32)
    case Grid.Outermost{}: (group * outer + outer_index : U32)

# The cell of the logical element at index, or the cell step of a logical
# stride that reads whole cells.
def cell_index(+lanes: Lanes,+index: U32) -> U32:
  Lanes{axis,+extent,+inner,+outer,position} = lanes
  cell_at(position,((index / (extent * inner)) * inner + index % inner : U32),((index / inner) % extent / 8 : U32),Tiled.groups(extent),outer)

# How one target axis reads the input: it repeats one index, it reads the
# lane axis one index a step, it steps whole cells along a source axis, or it
# does not read the cells.
type Reading is Data:
  Repeated{}
  Lane{}
  Stepping{axis: U32,step: U32,cells: U32}
  Unkept{}

def lane_reading(+step: U32,+stride: U32,+lanes: Lanes,+axis: U32) -> Reading:
  Bool.pick(Reading,U32.is_eq(step,1),Lane{},Bool.pick(Reading,U32.is_zero((step % 8 : U32)),Stepping{axis,step,cell_index(lanes,stride)},Unkept{}))

def reading_of(found: Tasks.Stepped,+stride: U32,+lanes: Lanes) -> Reading:
  match found:
    case Tasks.Stepped{+axis,+step,extent,first}:
      Bool.pick(Reading,U32.is_eq(axis,U32.from_nat(lane_axis(lanes))),lane_reading(step,stride,lanes,axis),Stepping{axis,step,cell_index(lanes,stride)})
    case Tasks.Unstepped{}: Unkept{}

def classified(+shape: List<&2,U32>,+lanes: Lanes,+extent: U32,+stride: U32) -> Reading:
  Bool.pick(Reading,Tasks.unread(extent,stride),Repeated{},reading_of(Tasks.stepped_axis(shape,Grid.coordinates(shape,0),stride,0),stride,lanes))

# The highest index of every source axis a view reaches.
def raised(reach: List<&2,U32>,axis: Nat,+amount: U32) -> List<&2,U32>:
  match reach axis:
    case [] axis: []
    case value <> rest 0n: (value + amount : U32) <> rest
    case value <> rest 1n+next: value <> raised(rest,next,amount)

def all_inside(reach: List<&2,U32>,shape: List<&2,U32>) -> Bool:
  match reach shape:
    case value <> rest extent <> others: U32.is_lt(value,extent) && all_inside(rest,others)
    case reach shape: True{}

# The target axes of a view read on the cells: whether every axis keeps the
# cells, the target axis that reads the lanes and its extent, the view of
# the cells on the other axes with their flags, and the reach of the view.
type Mapped is Data:
  Mapped{valid: Bool,lane: Maybe<&2,Nat>,lane_extent: U32,axes: List<&2,Views.Axis>,flags: List<&2,Bool>,reach: List<&2,U32>}

def with_reading(reading: Reading,+reversed: Bool,+index: Nat,+extent: U32,+lane_axis: Nat,mapped: Mapped) -> Mapped:
  match reading:
    case Repeated{}:
      Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
      Mapped{valid,lane,lane_extent,Views.Axis{extent,0} <> axes,reversed <> flags,reach}
    case Lane{}:
      Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
      Mapped{valid && Bool.not(reversed) && Bool.not(Maybe.is_some(&2,Nat,lane)),Some{index},extent,axes,flags,raised(reach,lane_axis,(extent - 1 : U32))}
    case Stepping{+axis,+step,+cells}:
      Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
      Mapped{valid,lane,lane_extent,Views.Axis{extent,cells} <> axes,reversed <> flags,raised(reach,U32.to_nat(axis),(step * (extent - 1) : U32))}
    case Unkept{}:
      Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
      Mapped{False{},lane,lane_extent,axes,flags,reach}

def mapped_axes(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+lanes: Lanes,+start: List<&2,U32>) -> Mapped:
  match axes:
    case []: Mapped{True{},None{},0,[],[],start}
    case Views.Axis{+extent,+stride} <> rest:
      with_reading(classified(shape,lanes,extent,stride),Views.flag(flags),index,extent,lane_axis(lanes),
        mapped_axes(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,start))

# ---- Gathers on the cells ----

# The cells moved through the view of the cells, the other target axes then
# the groups of the lane axis of the target, every piece setting the lanes
# past the extent of the result to +0 as it is filled.
def moved_view(+name: String,+axis: Nat,+lane_extent: U32,+cell_axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,cells: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(Views.axis_extents(cell_axes),axis,lane_extent,Grid.Innermost{}),name,cells,
    cell_axes,offset,List.append(&2,Bool,flags,[False{}])),axis,lane_extent,Grid.Innermost{}}

def none_flagged(flags: List<&2,Bool>) -> Bool:
  match flags:
    case Nil{}: True{}
    case Con{flag,rest}: Bool.not(flag) && none_flagged(rest)

# A usable input whose view of the cells reads every cell in order from the
# first, keeping the lane extent, keeps its cells: the lanes past the extent
# are +0 already.
def in_order(status: Tensors.Status,+cell_axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lanes: U32,+lane_extent: U32,+source: List<&2,U32>) -> Bool:
  +count = Tensors.count_or_zero(source)
  Tensors.usable(status) && U32.is_zero(offset) && none_flagged(flags) && U32.is_eq(lanes,lane_extent) && U32.is_lt(0,count) &&
    U32.is_eq(count,Tensors.count_or_zero(Views.axis_extents(cell_axes))) && Views.is_contiguous(Views.View{cell_axes,offset})

def moved_or_kept(kept: Bool,+name: String,+axis: Nat,+lane_extent: U32,+cell_axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,cells: Tensors.Tensor<Tile.Tile8>) ->
  Tiled.Tiled:
  match kept:
    case True{}: Tiled.Tiled{Tensors.relabeled(~Tile.Tile8,cells,Views.axis_extents(cell_axes)),axis,lane_extent,Grid.Innermost{}}
    case False{}: moved_view(name,axis,lane_extent,cell_axes,flags,offset,cells)

def moved_cells(+name: String,+axis: Nat,+lane_extent: U32,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+group_step: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.Tensor{pieces,+source,fill,status,workers},own,+lanes,position} = tiled
  +cell_axes = List.append(&2,Views.Axis,axes,[Views.Axis{Tiled.groups(lane_extent),group_step}])
  +status = status
  moved_or_kept(in_order(status,cell_axes,flags,offset,lanes,lane_extent,source),name,axis,lane_extent,cell_axes,flags,offset,
    Tensors.Tensor{pieces,source,fill,status,workers})

# The first index of the view along the lane axis.
def coordinate_at(coordinates: List<&2,U32>,axis: Nat) -> U32:
  match coordinates axis:
    case [] axis: 0
    case value <> rest 0n: value
    case value <> rest 1n+next: coordinate_at(rest,next)

# What a gather on the cells reads beside the mapped axes: its name, the
# offset of the view, the input extents and the lanes of the input.
type Gathering is Data:
  Gathering{name: String,axes: List<&2,Views.Axis>,offset: U32,reversed: List<&2,Bool>,shape: List<&2,U32>,target: List<&2,U32>}

def gathered_lane(lane: Maybe<&2,Nat>,+name: String,+offset: U32,+lanes: Lanes,+lane_extent: U32,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,tiled: Tiled.Tiled) ->
  Tiled.Tiled:
  match lane:
    case Some{+axis}:
      moved_cells(name,axis,lane_extent,axes,flags,cell_index(lanes,offset),cell_index(lanes,(8 * lane_inner(lanes) : U32)),Tiled.along(tiled,lane_axis(lanes)))
    case None{}: tiled

def gather_kept(mapped: Mapped,+name: String,+offset: U32,+lanes: Lanes,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Mapped{valid,lane,+lane_extent,+axes,+flags,reach} = mapped
  gathered_lane(lane,name,offset,lanes,lane_extent,axes,flags,tiled)

# The gather with the lanes of the input on the axis of lanes, which the view
# keeps.
def gather_at(+name: String,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+lanes: Lanes,tiled: Tiled.Tiled) -> Tiled.Tiled:
  +start = Grid.coordinates(shape,offset)
  gather_kept(mapped_axes(axes,reversed,0n,shape,lanes,start),name,offset,lanes,tiled)

def mapped_keeps(mapped: Mapped,+shape: List<&2,U32>,+first: U32) -> Bool:
  Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  valid && Maybe.is_some(&2,Nat,lane) && all_inside(reach,shape) && U32.is_zero((first % 8 : U32))

# The view keeps the lanes on their axis: one target axis reads it one index
# a step from a group boundary, every other target axis repeats or steps
# whole cells, and the view stays inside the input.
def keeps(+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+lanes: Lanes) -> Bool:
  +start = Grid.coordinates(shape,offset)
  mapped_keeps(mapped_axes(axes,reversed,0n,shape,lanes,start),shape,coordinate_at(start,lane_axis(lanes)))

# The first of the axes whose lanes the view keeps.
def kept_among(axes: List<&2,Nat>,+context: Gathering,+position: Grid.Position) -> Maybe<&2,Nat>:
  match axes:
    case []: None{}
    case axis <> rest:
      Gathering{name,+view,+offset,+reversed,+shape,target} = context
      +axis = axis
      Bool.pick(Maybe<&2,Nat>,Nat.is_lt(axis,List.length(&2,U32,shape)) && keeps(view,offset,reversed,shape,lanes_of(shape,axis,position)),Some{axis},
        kept_among(rest,context,position))

# The view keeps no axis whole: the input takes its lanes on its unit axis,
# one value a cell in the logical shape, the view reads the cells as they are
# and the result keeps the lanes on the unit axis of the target.
def gather_values(+context: Gathering,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Gathering{+name,+axes,+offset,+reversed,shape,+target} = context
  Tiled.Tiled{Tensors.moved(~Tile.Tile8,name,Tiled.unit_cells(tiled),axes,offset,reversed),List.length(&2,U32,target),1,Grid.Innermost{}}

# ---- The axes to tile along ----

# The axes of a shape of an extent at a stride, as a view of the shape reads
# one of its axes.
def reading(shape: List<&2,U32>,+index: Nat,+extent: U32,+stride: U32) -> List<&2,Nat>:
  match shape:
    case []: []
    case axis_extent <> rest:
      +rest = rest
      +found = reading(rest,1n+index,extent,stride)
      Bool.pick(List<&2,Nat>,U32.is_eq(axis_extent,extent) && U32.is_eq(Tiled.product(rest),stride),index <> found,found)

def read_at(axes: List<&2,Views.Axis>,+position: Nat,+shape: List<&2,U32>) -> List<&2,Nat>:
  match axes position:
    case Views.Axis{+extent,+stride} <> rest 0n: reading(shape,0n,extent,stride)
    case axis <> rest 1n+next: read_at(rest,next,shape)
    case [] position: []

def longest_free(shape: List<&2,U32>,+taken: List<&2,Nat>,+index: Nat,+best: Nat,+extent: U32) -> Nat:
  match shape:
    case []: best
    case axis_extent <> rest:
      +axis_extent = axis_extent
      +longer = U32.is_le(extent,axis_extent) && Bool.not(Views.contains_index(taken,index))
      longest_free(rest,taken,1n+index,Bool.pick(Nat,longer,index,best),Bool.pick(U32,longer,axis_extent,extent))

# The axes from the longest down, the later among equal extents first.
def by_extent(remaining: Nat,+shape: List<&2,U32>,+taken: List<&2,Nat>) -> List<&2,Nat>:
  match remaining:
    case 0n: []
    case 1n+rest:
      +next = longest_free(shape,taken,0n,List.length(&2,U32,shape),0)
      next <> by_extent(rest,shape,next <> taken)

# The axes a gather may keep the lanes on: the input's own lane axis, the axis
# the view reads at the lane axis of its target, then every axis from the
# longest down.
def lane_candidates(+shape: List<&2,U32>,+axis: Nat,+axes: List<&2,Views.Axis>,+target_axis: Nat) -> List<&2,Nat>:
  axis <> List.append(&2,Nat,read_at(axes,target_axis,shape),by_extent(List.length(&2,U32,shape),shape,[]))

# ---- Gathers ----

# An empty target holds no value. A target with values read from an empty
# input reads past it.
def empty_cells(empty: Bool,count: Maybe<&2,U32>,+name: String,+shape: List<&2,U32>,+target: List<&2,U32>,+workers: U32) -> Tiled.Tiled:
  match empty count:
    case True{} Some{count}: Tiled.empty(target,workers)
    case empty count: Tiled.failed(name,Tensors.InvalidArgument{shape},target,workers)

def gather_empty(+name: String,+shape: List<&2,U32>,+target: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => extents => status => workers =>
    empty_cells(Shapes.has_zero(target),Tensors.count_of(target),name,shape,target,workers))

# A failed input keeps its problem; a gather without a view fails.
def unviewed_cells(status: Tensors.Status,+name: String,+shape: List<&2,U32>,+target: List<&2,U32>,+workers: U32) -> Tiled.Tiled:
  match status:
    case Tensors.Usable{}: Tiled.failed(name,Tensors.InvalidArgument{shape},target,workers)
    case Tensors.Failed{problem}: Tiled.with_status(Tensors.Failed{problem},Tiled.empty(target,workers))

def gather_unviewed(status: Tensors.Status,+name: String,+shape: List<&2,U32>,+target: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => extents => own_status => workers => unviewed_cells(status,name,shape,target,workers))

def gather_chosen(chosen: Maybe<&2,Nat>,+context: Gathering,+position: Grid.Position,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match chosen:
    case Some{+axis}:
      Gathering{+name,+axes,+offset,+reversed,+shape,target} = context
      gather_at(name,axes,offset,reversed,shape,lanes_of(shape,axis,position),tiled)
    case None{}: gather_values(context,tiled)

def gather_nonempty(empty: Bool,+name: String,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,
  +axis: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match empty:
    case False{}:
      Tiled.Tiled{cells,own,lanes,+position} = tiled
      +context = {Gathering{name,axes,offset,reversed,shape,target} : Gathering}
      gather_chosen(kept_among(lane_candidates(shape,axis,axes,Tiled.lane_axis(List.length(&2,U32,target))),context,position),context,position,
        Tiled.Tiled{cells,own,lanes,position})
    case True{}: gather_empty(name,shape,target,tiled)

def gather_started(empty: Bool,+name: String,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,
  started: Tiled.Tiled & Nat) -> Tiled.Tiled:
  (tiled,+axis) = started
  gather_nonempty(empty,name,axes,offset,reversed,shape,target,axis,tiled)

# The elements a view of a tiled tensor of extents shape selects, in the
# view's row-major order with the flagged axes in reverse, as a tiled tensor
# of extents target.
def gather(+name: String,status: Tensors.Status,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match status source:
    case Tensors.Usable{} Some{Views.View{+axes,+offset}}:
      gather_started(Shapes.has_zero(shape) || Shapes.has_zero(target),name,axes,offset,reversed,shape,target,Tiled.lane_axis_of(tiled))
    case status source: gather_unviewed(status,name,shape,target,tiled)
