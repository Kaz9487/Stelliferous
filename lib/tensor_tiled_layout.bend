# Layout operations of tiled tensors. Gathers compute the view of the call
# from the extents, as Layout does for its tensors, and read it through
# TiledView. Writes, padding and reshapes work on the cells with the lanes on
# an axis the call leaves whole, so the cells move as they are.
import Base
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_layout.bend as Layout
import ./tensor_tiled.bend as Tiled
import ./tensor_grid.bend as Grid
import ./tensor_tiled_view.bend as TiledView
import ./tensor_tiled_map.bend as TiledMap
import ./kernel_tile.bend as Tile

# The gather labelled with the extents of the view.
def gather_extents(+name: String,status: Tensors.Status,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match source:
    case Some{Views.View{+axes,offset}}: TiledView.gather(name,status,Some{Views.View{axes,offset}},reversed,shape,Views.axis_extents(axes),tiled)
    case None{}: TiledView.gather(name,status,None{},reversed,shape,[],tiled)

# The result's axis i is the input axis order[i].
def transpose(tiled: Tiled.Tiled,order: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    gather_extents("transpose",status,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),[],shape,tiled))

# count elements from start along axis, every step-th one.
def slice(tiled: Tiled.Tiled,axis: U32,start: U32,count: U32,step: U32) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    gather_extents("slice",status,Layout.sliced(Tensors.view_of(shape),axis,start,count,step),[],shape,tiled))

# The elements in reverse order along every listed axis; an axis is listed once.
def flip(tiled: Tiled.Tiled,+axes: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    +rank = List.length(&2,U32,shape)
    gather_extents("flip",status,Layout.flipped(Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,rank),Tensors.view_of(shape)),
      Layout.reversed_flags(rank,0,axes),shape,tiled))

def broadcast_view(status: Tensors.Status,view: Maybe<&2,Views.View>,+shape: List<&2,U32>,+target: List<&2,U32>,+workers: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match status view:
    case Tensors.Usable{} None{}: Tiled.failed("broadcast_to",Tensors.ShapeMismatch{shape,target},target,workers)
    case status view: TiledView.gather("broadcast_to",status,view,[],shape,target,tiled)

# The tensor repeated to the extents target under NumPy broadcasting: one
# gather through the stretched view.
def broadcast_to(tiled: Tiled.Tiled,+target: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    broadcast_view(status,Tensors.stretched(Tensors.view_of(shape),target),shape,target,workers,tiled))

# ---- Writes ----

# The part, its lanes on the lane axis of the target and its groups at the
# same position, written into the cells of the target along another axis.
def written_cells(+axis: Nat,+start: U32,target: Tiled.Tiled,part: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},+lane,+lanes,+position} = target
  Tiled.Tiled{Layout.set_slice(~Tile.Tile8,Tensors.Tensor{pieces,cells,fill,status,workers},U32.from_nat(Tiled.cell_axis(position,axis,lane,List.length(&2,U32,cells))),
    start,Tiled.cells_of(Tiled.arranged(part,lane,position))),lane,lanes,position}

def written_on(lanes: Bool,+axis: Nat,+start: U32,target: Tiled.Tiled,part: Tiled.Tiled) -> Tiled.Tiled:
  match lanes:
    case True{}: Tiled.lanes_written(target,start,part)
    case False{}: written_cells(axis,start,target,part)

def written_from(started: Tiled.Tiled & Nat,+axis: Nat,+start: U32,part: Tiled.Tiled) -> Tiled.Tiled:
  (target,+lane) = started
  written_on(Nat.is_eq(axis,lane),axis,start,target,part)

def written(valid: Bool,+shape: List<&2,U32>,+part_shape: List<&2,U32>,+axis: U32,+start: U32,target: Tiled.Tiled,part: Tiled.Tiled) -> Tiled.Tiled:
  match valid:
    case True{}: written_from(Tiled.lane_axis_of(target),U32.to_nat(axis),start,part)
    case False{}: Tiled.with_status(Tensors.Failed{Tensors.Problem{"set_slice",Tensors.ShapeMismatch{shape,part_shape}}},target)

def write_status(target_status: Tensors.Status,part_status: Tensors.Status,+shape: List<&2,U32>,+part_shape: List<&2,U32>,+axis: U32,+start: U32,
  target: Tiled.Tiled,part: Tiled.Tiled) -> Tiled.Tiled:
  match target_status part_status:
    case Tensors.Usable{} Tensors.Usable{}: written(Layout.slice_fits(shape,part_shape,axis,start,0),shape,part_shape,axis,start,target,part)
    case Tensors.Usable{} Tensors.Failed{+problem}: Tiled.with_status(Tensors.Failed{problem},target)
    case Tensors.Failed{problem} part_status: target

# target[..., start : start + part extent, ...] = part along axis, written
# into the cells of the target: whole cells along another axis, the lanes
# the part reaches along the lane axis.
def set_slice(target: Tiled.Tiled,+axis: U32,+start: U32,part: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,target,target => shape => target_status => target_workers =>
    Tiled.with_metadata(~Tiled.Tiled,part,part => part_shape => part_status => part_workers =>
      +workers = U32.max(target_workers,part_workers)
      write_status(target_status,part_status,shape,part_shape,axis,start,Tiled.with_workers(target,workers),Tiled.with_workers(part,workers))))

# ---- Padding ----

# The first axis padding leaves whole, or the unit axis.
def unpadded_from(before: List<&2,U32>,after: List<&2,U32>,+index: Nat,+rank: Nat) -> Nat:
  match before after:
    case Con{+front,before} Con{+back,after}: Bool.pick(Nat,U32.is_zero(front) && U32.is_zero(back),index,unpadded_from(before,after,1n+index,rank))
    case before after: rank

# The lanes stay on their axis when padding leaves it whole.
def pad_lane(+lane: Nat,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat) -> Nat:
  Bool.pick(Nat,U32.is_zero(Layout.extent_at(before,lane)) && U32.is_zero(Layout.extent_at(after,lane)),lane,unpadded_from(before,after,0n,rank))

# The border of the cells: the value in every lane, or in the first lane of a
# cell of the unit axis.
def cell_border(border: Layout.Border<F32>,+unit: Bool) -> Layout.Border<Tile.Tile8>:
  match border:
    case Layout.Constant{+value}: Layout.Constant{Bool.pick(Tile.Tile8,unit,Tiled.unit_tile(value),Tile.filled(value))}
    case Layout.Edge{}: Layout.Edge{}

# The extents of padded cells.
def grown(cells: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>) -> List<&2,U32>:
  match cells before after:
    case Con{+extent,cells} Con{+front,before} Con{+back,after}: (extent + front + back : U32) <> grown(cells,before,after)
    case cells before after: cells

# Every piece of the padded cells sets the lanes past the extent of the lane
# axis to +0 as it is filled: cells of the value hold it in every lane.
def padded_cells(+border: Layout.Border<F32>,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},+lane,+lanes,+position} = tiled
  +front = Tiled.cell_values(before,lane,position)
  +back = Tiled.cell_values(after,lane,position)
  Tiled.Tiled{Layout.pad_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,TiledMap.padding_of(grown(cells,front,back),lane,lanes,position),
    cell_border(border,Bool.not(Nat.is_lt(lane,rank))),Tensors.Tensor{pieces,cells,fill,status,workers},front,back),lane,lanes,position}

def padded_from(started: Tiled.Tiled & Nat,+border: Layout.Border<F32>,+before: List<&2,U32>,+after: List<&2,U32>,+rank: Nat) -> Tiled.Tiled:
  (tiled,+lane) = started
  padded_cells(border,before,after,rank,Tiled.along(tiled,pad_lane(lane,before,after,rank)))

def pad_valid(+border: Layout.Border<F32>,status: Tensors.Status,valid: Bool,tiled: Tiled.Tiled,+shape: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>) -> Tiled.Tiled:
  match status valid:
    case Tensors.Usable{} True{}: padded_from(Tiled.lane_axis_of(tiled),border,before,after,List.length(&2,U32,shape))
    case Tensors.Usable{} False{}: Tiled.with_status(Tensors.Failed{Tensors.Problem{Layout.border_name(~F32,border),Tensors.InvalidArgument{shape}}},tiled)
    case Tensors.Failed{problem} valid: tiled

# before[i] elements are added in front of axis i and after[i] behind it. Both
# lists have one entry per axis and the padded extents must be allocatable.
# The cells are padded once with the lanes on an axis the padding leaves
# whole.
def pad(+border: Layout.Border<F32>,tiled: Tiled.Tiled,+before: List<&2,U32>,+after: List<&2,U32>) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    pad_valid(border,status,Views.has_count(Layout.padded_count(Shapes.padded(shape,before,after))) && Layout.border_fits(~F32,border,shape,before,after),
      tiled,shape,before,after))

# ---- Reshape ----

# The axes around the lane axis of the source: the elements before it, the
# lanes and the elements after it.
type Kept is Data:
  Kept{outer: U32,lanes: U32,inner: U32,innermost: Bool}

# An axis of the new extents keeps the cells when the axes before it hold as
# many elements as those before the lane axis, it has the lanes as extent and
# the axes after it hold as many elements as those after the lane axis.
def same_around(found: Maybe<&2,Tiled.Around>,+kept: Kept) -> Bool:
  match found:
    case Some{Tiled.Around{+before,+extent,+after}}:
      Kept{+outer,+lanes,+inner,innermost} = kept
      U32.is_eq(before,outer) && U32.is_eq(extent,lanes) && U32.is_eq(Tiled.product(after),inner)
    case None{}: False{}

# The first axis from axis on that fits, found telling whether axis does; the
# unit axis past the last is included.
def scan(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: Kept -> Bool,remaining: Nat,found: Bool,+shape: List<&2,U32>,+kept: Kept,+axis: Nat) -> Maybe<&2,Nat>:
  match remaining found:
    case remaining True{}: Some{axis}
    case 0n False{}: None{}
    case 1n+rest False{}:
      +next = Nat.add(axis,1n)
      scan(~fits,rest,fits(Tiled.around(shape,next,1),kept),shape,kept,next)

# The lane axis of the new rank is tried first.
def preferred_found(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: Kept -> Bool,found: Bool,+shape: List<&2,U32>,+kept: Kept,+preferred: Nat) -> Maybe<&2,Nat>:
  match found:
    case True{}: Some{preferred}
    case False{}: scan(~fits,List.length(&2,U32,shape),fits(Tiled.around(shape,0n,1),kept),shape,kept,0n)

def axis_fitting(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: Kept -> Bool,+shape: List<&2,U32>,+kept: Kept) -> Maybe<&2,Nat>:
  +preferred = Tiled.lane_axis(List.length(&2,U32,shape))
  preferred_found(~fits,fits(Tiled.around(shape,preferred,1),kept),shape,kept,preferred)

# A usable reshape fails when the extents count other elements.
def mismatched(+source: List<&2,U32>,+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_status(Tensors.Failed{Tensors.Problem{"reshape",Tensors.ShapeMismatch{source,shape}}},tiled)

# An empty source and empty extents hold no value; a failed source keeps its
# problem.
def reshape_other(status: Tensors.Status,empty: Bool,count: Maybe<&2,U32>,+source: List<&2,U32>,+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match status empty count:
    case Tensors.Usable{} True{} Some{count}:
      Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => extents => status => workers => Tiled.empty(shape,workers))
    case Tensors.Usable{} empty count: mismatched(source,shape,tiled)
    case Tensors.Failed{problem} empty count: tiled

# Whole groups of a last lane axis keep their cells in order under any number
# of groups when the groups are innermost: an axis of the new extents fits
# when it is last and it and the lanes fill their groups.
def whole_groups(found: Maybe<&2,Tiled.Around>,+kept: Kept) -> Bool:
  match found:
    case Some{Tiled.Around{+before,+extent,+after}}:
      Kept{+outer,+lanes,+inner,+innermost} = kept
      innermost && U32.is_eq(inner,1) && U32.is_zero((lanes % 8 : U32)) && U32.is_zero((extent % 8 : U32)) && U32.is_eq(Tiled.product(after),1)
    case None{}: False{}

def regrouped_around(around: Maybe<&2,Tiled.Around>,+axis: Nat,tiled: Tiled.Tiled,+shape: List<&2,U32>) -> Tiled.Attempt<Tiled.Tiled>:
  match around:
    case Some{Tiled.Around{+before,+extent,+after}}:
      Tiled.Tiled{cells,lane_axis,lanes,+position} = tiled
      Tiled.Reached{Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,axis,position)),axis,extent,position}}
    case None{}: Tiled.Untried{tiled}

def regrouped(found: Maybe<&2,Nat>,tiled: Tiled.Tiled,+shape: List<&2,U32>) -> Tiled.Attempt<Tiled.Tiled>:
  match found:
    case Some{+axis}: regrouped_around(Tiled.around(shape,axis,1),axis,tiled,shape)
    case None{}: Tiled.Untried{tiled}

def relabelled(found: Maybe<&2,Nat>,+kept: Kept,tiled: Tiled.Tiled,+shape: List<&2,U32>) -> Tiled.Attempt<Tiled.Tiled>:
  match found:
    case Some{+axis}:
      Tiled.Tiled{cells,lane_axis,+lanes,+position} = tiled
      Tiled.Reached{Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,axis,position)),axis,lanes,position}}
    case None{}: regrouped(axis_fitting(~whole_groups,shape,kept),tiled,shape)

# A usable source of as many elements, none of them on an empty axis, keeps
# its cells when an axis of the new extents keeps the lanes and the axes
# around them, or holds whole groups of them.
def reshape_around(axes: Maybe<&2,Tiled.Around>,valid: Bool,+innermost: Bool,status: Tensors.Status,tiled: Tiled.Tiled,+source: List<&2,U32>,+shape: List<&2,U32>) ->
  Tiled.Attempt<Tiled.Tiled>:
  match axes valid:
    case Some{Tiled.Around{+outer,+lanes,+after}} True{}:
      +kept = {Kept{outer,lanes,Tiled.product(after),innermost} : Kept}
      relabelled(axis_fitting(~same_around,shape,kept),kept,tiled,shape)
    case axes valid: Tiled.Reached{reshape_other(status,Shapes.has_zero(source) && Shapes.has_zero(shape),Tensors.count_of(shape),source,shape,tiled)}

def reshape_positioned(+source: List<&2,U32>,+status: Tensors.Status,+axis: Nat,started: Tiled.Tiled & Grid.Position,+shape: List<&2,U32>) -> Tiled.Attempt<Tiled.Tiled>:
  (tiled,position) = started
  reshape_around(Tiled.around(source,axis,1),
    Tensors.usable(status) && Bool.not(Shapes.has_zero(source)) && Layout.same_count(Tensors.count_of(source),Tensors.count_of(shape)),Bool.not(Grid.is_outermost(position)),
    status,tiled,source,shape)

def reshape_from(+source: List<&2,U32>,+status: Tensors.Status,started: Tiled.Tiled & Nat,+shape: List<&2,U32>) -> Tiled.Attempt<Tiled.Tiled>:
  (tiled,+axis) = started
  reshape_positioned(source,status,axis,Tiled.position_of(tiled),shape)

def reshape_at(+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Tiled.Attempt<Tiled.Tiled>:
  Tiled.with_metadata(~Tiled.Attempt<Tiled.Tiled>,tiled,tiled => source => status => workers =>
    +source = source
    reshape_from(source,status,Tiled.lane_axis_of(tiled),shape))

# Whether an axis of the new extents keeps lanes and the axes around them.
def kept_around(axes: Maybe<&2,Tiled.Around>,+innermost: Bool,+shape: List<&2,U32>) -> Bool:
  match axes:
    case Some{Tiled.Around{+outer,+lanes,after}}: Maybe.is_some(&2,Nat,axis_fitting(~same_around,shape,Kept{outer,lanes,Tiled.product(after),innermost}))
    case None{}: False{}

# The longest axis of the source but own, the later among equal extents, of
# an extent above 1 whose lanes and the axes around them an axis of the new
# extents keeps, or the unit axis when there is none.
def matched_axis(remaining: Nat,+index: Nat,+source: List<&2,U32>,+shape: List<&2,U32>,+own: Nat,+innermost: Bool,+best: Nat,+extent: U32) -> Nat:
  match remaining:
    case 0n: best
    case 1n+rest:
      +axis_extent = Grid.lanes_at(source,index)
      +fits = Bool.not(Nat.is_eq(index,own)) && U32.is_le(extent,axis_extent) && kept_around(Tiled.around(source,index,1),innermost,shape)
      matched_axis(rest,1n+index,source,shape,own,innermost,Bool.pick(Nat,fits,index,best),Bool.pick(U32,fits,axis_extent,extent))

def reshape_again(attempt: Tiled.Attempt<Tiled.Tiled>,+source: List<&2,U32>,+shape: List<&2,U32>) -> Tiled.Tiled:
  match attempt:
    case Tiled.Reached{result}: result
    case Tiled.Untried{tiled}: mismatched(source,shape,tiled)

def reshape_relaned(+source: List<&2,U32>,+shape: List<&2,U32>,+own: Nat,started: Tiled.Tiled & Grid.Position) -> Tiled.Tiled:
  (tiled,+position) = started
  +rank = List.length(&2,U32,source)
  +axis = matched_axis(rank,0n,source,shape,own,Bool.not(Grid.is_outermost(position)),rank,2)
  reshape_again(reshape_at(shape,Tiled.along(tiled,axis)),source,shape)

# An axis of the new extents and the next split the lane axis when their
# extents, both above 1, multiply to the lanes, the axes before them hold as
# many elements as those before the lane axis and the axes after them as many
# as those after it.
type Splitting is Data:
  Splitting{axis: Nat,outer: U32,anchors: U32,width: U32,after: U32}

def split_after(after: List<&2,U32>,+axis: Nat,+before: U32,+anchors: U32,+kept: Kept) -> Maybe<&2,Splitting>:
  match after:
    case Con{+width,+rest}:
      Kept{+outer,+lanes,+inner,innermost} = kept
      +inside = Tiled.product(rest)
      Bool.pick(Maybe<&2,Splitting>,U32.is_eq(before,outer) && U32.is_lt(1,anchors) && U32.is_lt(1,width) && U32.is_eq((anchors * width : U32),lanes) && U32.is_eq(inside,inner),
        Some{Splitting{axis,outer,anchors,width,inside}},None{})
    case Nil{}: None{}

def split_of(found: Maybe<&2,Tiled.Around>,+axis: Nat,+kept: Kept) -> Maybe<&2,Splitting>:
  match found:
    case Some{Tiled.Around{+before,+anchors,after}}: split_after(after,axis,before,anchors,kept)
    case None{}: None{}

def split_scan(remaining: Nat,found: Maybe<&2,Splitting>,+shape: List<&2,U32>,+kept: Kept,+axis: Nat) -> Maybe<&2,Splitting>:
  match remaining found:
    case remaining Some{split}: Some{split}
    case 0n None{}: None{}
    case 1n+rest None{}:
      +next = Nat.add(axis,1n)
      split_scan(rest,split_of(Tiled.around(shape,next,1),next,kept),shape,kept,next)

# The first split of the lane axis among the new extents.
def lane_split(around: Maybe<&2,Tiled.Around>,+shape: List<&2,U32>) -> Maybe<&2,Splitting>:
  match around:
    case Some{Tiled.Around{+outer,+lanes,after}}:
      +kept = {Kept{outer,lanes,Tiled.product(after),False{}} : Kept}
      split_scan(List.length(&2,U32,shape),split_of(Tiled.around(shape,0n,1),0n,kept),shape,kept,0n)
    case None{}: None{}

def split_or_relaned(found: Maybe<&2,Splitting>,tiled: Tiled.Tiled,+source: List<&2,U32>,+shape: List<&2,U32>,+own: Nat) -> Tiled.Tiled:
  match found:
    case Some{Splitting{+axis,+outer,+anchors,+width,+after}}: Tiled.split_lanes(tiled,shape,axis,outer,anchors,width,after)
    case None{}: reshape_relaned(source,shape,own,Tiled.position_of(tiled))

def reshape_moved(attempt: Tiled.Attempt<Tiled.Tiled>,+source: List<&2,U32>,+shape: List<&2,U32>,+own: Nat) -> Tiled.Tiled:
  match attempt:
    case Tiled.Reached{result}: result
    case Tiled.Untried{tiled}: split_or_relaned(lane_split(Tiled.around(source,own,1),shape),tiled,source,shape,own)

def reshape_lane(+shape: List<&2,U32>,+source: List<&2,U32>,started: Tiled.Tiled & Nat) -> Tiled.Tiled:
  (tiled,+own) = started
  reshape_moved(reshape_at(shape,tiled),source,shape,own)

def reshape_started(+shape: List<&2,U32>,started: Tiled.Tiled & List<&2,U32>) -> Tiled.Tiled:
  (tiled,+source) = started
  reshape_lane(shape,source,Tiled.lane_axis_of(tiled))

# Same elements in the same row-major order under new extents: the cells
# themselves when the lanes keep their axes around them or fill their groups.
# Extents that split the lane axis in two take the lanes on the second, each
# cell read from the one or two cells that hold its lanes. Otherwise the
# lanes move to the longest axis of the source whose lanes and
# the axes around them an axis of the new extents keeps, or to the unit axis
# when none does, and the cells take the new extents.
def reshape(tiled: Tiled.Tiled,+shape: List<&2,U32>) -> Tiled.Tiled:
  reshape_started(shape,Tiled.shape(tiled))

# ---- Upsampling ----

def upsample_checked(valid: Bool,planes: Maybe<&2,U32>,+shape: List<&2,U32>,+scale: U32,+workers: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match valid planes:
    case True{} Some{planes}:
      reshape(gather_extents("upsample_nearest",Tensors.Usable{},Layout.repeated(Tensors.view_of(shape),scale),[],shape,tiled),Layout.scaled_extents(shape,scale))
    case valid planes: Tiled.failed("upsample_nearest",Tensors.InvalidArgument{shape},Layout.scaled_extents(shape,scale),workers)

def upsample_status(status: Tensors.Status,+shape: List<&2,U32>,+scale: U32,+workers: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match status:
    case Tensors.Failed{problem}: tiled
    case Tensors.Usable{}: upsample_checked(Layout.upsample_valid(shape,scale),Layout.planes(shape),shape,scale,workers,tiled)

# Nearest-neighbour upsampling of the last two axes: one gather through the
# view that repeats each row and each column scale times, under the extents
# of that view, then each axis joined with its repeats.
def upsample_nearest(tiled: Tiled.Tiled,+scale: U32) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers => upsample_status(status,shape,scale,workers,tiled))
