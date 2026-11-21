# Layout operations consume their arguments. Reshape relabels the cells.
# set_slice, concat and pad place their sources in a new tensor, every
# component of the placement a task that reads only its own pieces. A
# reduction folds the lines of every piece in place when the pieces hold
# whole lines, and the lines of all cells in order otherwise.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_line_chain.bend as LineChain
import ./tensor_tasks.bend as Tasks
import ./traversal_partition.bend as Partition

def naturals(values: List<&2,U32>) -> List<&2,Nat>:
  match values:
    case []: []
    case value <> rest: U32.to_nat(value) <> naturals(rest)

def permuted(view: Maybe<&2,Views.View>,order: List<&2,Nat>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.permute(view,order)
    case None{}: None{}

def sliced(view: Maybe<&2,Views.View>,axis: U32,start: U32,count: U32,step: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,U32.to_nat(axis),start,count,step)
    case None{}: None{}

def same_count(left: Maybe<&2,U32>,right: Maybe<&2,U32>) -> Bool:
  match left right:
    case Some{left} Some{right}: U32.is_eq(left,right)
    case left right: False{}

def reshape_status(~Element: Data,status: Tensors.Status,valid: Bool,pieces: Pieces.Pieces(~Element),+source: List<&2,U32>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Usable{} True{}: Tensors.relabeled(~Element,Tensors.Tensor{pieces,source,fill,Tensors.Usable{},workers},shape)
    case Tensors.Usable{} False{}: Tensors.Tensor{pieces,source,fill,Tensors.Failed{Tensors.Problem{"reshape",Tensors.ShapeMismatch{source,shape}}},workers}
    case Tensors.Failed{problem} valid: Tensors.Tensor{pieces,source,fill,Tensors.Failed{problem},workers}

# Same elements in the same row-major order under new extents.
def reshape(~Element: Data,tensor: Tensors.Tensor<Element>,+shape: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+source,+fill,status,+workers} = tensor
  reshape_status(~Element,status,same_count(Tensors.count_of(source),Tensors.count_of(shape)),pieces,source,shape,fill,workers)

def extent_at(extents: List<&2,U32>,axis: Nat) -> U32:
  Tensors.known_count(List.get(&2,U32,extents,axis))

def leading(ranked: Bool,+shape: List<&2,U32>) -> Maybe<&2,U32>:
  match ranked:
    case True{}: Shapes.product(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),Some{1})
    case False{}: None{}

# The number of planes before the last two axes; none below rank two.
def planes(+shape: List<&2,U32>) -> Maybe<&2,U32>:
  leading(Nat.is_le(2n,List.length(&2,U32,shape)),shape)

# Nearest-neighbour upsampling repeats each element of the last two axes
# scale times, by copying from a view whose repeated axes have stride zero.
def prepend_repeated(axis: Views.Axis,rest: Maybe<&2,List<&2,Views.Axis>>) -> Maybe<&2,List<&2,Views.Axis>>:
  match rest:
    case Some{axes}: Some{axis <> axes}
    case None{}: None{}

def repeated_axes(axes: List<&2,Views.Axis>,+scale: U32) -> Maybe<&2,List<&2,Views.Axis>>:
  match axes:
    case [Views.Axis{height,height_stride},Views.Axis{width,width_stride}]:
      Some{[Views.Axis{height,height_stride},Views.Axis{scale,0},Views.Axis{width,width_stride},Views.Axis{scale,0}]}
    case [axis] : None{}
    case []: None{}
    case axis <> rest: prepend_repeated(axis,repeated_axes(rest,scale))

def with_offset(axes: Maybe<&2,List<&2,Views.Axis>>,offset: U32) -> Maybe<&2,Views.View>:
  match axes:
    case Some{axes}: Some{Views.View{axes,offset}}
    case None{}: None{}

def repeated(view: Maybe<&2,Views.View>,scale: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{Views.View{axes,offset}}: with_offset(repeated_axes(axes,scale),offset)
    case None{}: None{}

def scaled_extents(extents: List<&2,U32>,+scale: U32) -> List<&2,U32>:
  match extents:
    case [height,width]: [(height * scale : U32),(width * scale : U32)]
    case extent <> rest: extent <> scaled_extents(rest,scale)
    case []: []

# The repeated extents and the output count fit the machine word.
def upsample_valid(+shape: List<&2,U32>,+scale: U32) -> Bool:
  +height = extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n))
  +width = extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
  Views.has_count(Shapes.checked_multiply(height,scale)) && Views.has_count(Shapes.checked_multiply(width,scale)) &&
    Views.has_count(Tensors.count_of(scaled_extents(shape,scale)))

def with_extent(extents: List<&2,U32>,axis: Nat,+extent: U32) -> List<&2,U32>:
  List.set(&2,U32,extents,axis,extent)

# ---- Placing sources ----

# A source of extents source placed in a target of extents target at shifts:
# target index i of every axis reads source index i - shift.
def placed_readings(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32) ->
  List<&2,Tasks.Reading>:
  match target source shifts:
    case Con{+extent,target} Con{+bound,source} Con{+shift,shifts}:
      Tasks.Reading{extent,Grid.block(target_grid,axis,extent),Tasks.Read{axis,Grid.block(source_grid,axis,bound),1,0,False{},1,1,shift,bound,clamped}} <>
        placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))
    case target source shifts: []

# Along every axis the source fills the indices from its shift on.
def placed_segments(source: List<&2,U32>,shifts: List<&2,U32>) -> List<&2,List<&2,Tasks.Segment>>:
  match source shifts:
    case Con{+extent,source} Con{+shift,shifts}: {[Tasks.Segment{shift,extent,0,1}] : List<&2,Tasks.Segment>} <> placed_segments(source,shifts)
    case source shifts: []

# Along every axis the indices before the source repeat its first index and
# the indices after it its last.
def edge_segments(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>) -> List<&2,List<&2,Tasks.Segment>>:
  match target source shifts:
    case Con{+total,target} Con{+extent,source} Con{+shift,shifts}:
      {[Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,extent,0,1},Tasks.Segment{(shift + extent : U32),(total - shift - extent : U32),(extent - 1 : U32),0}] : List<&2,Tasks.Segment>} <>
        edge_segments(target,source,shifts)
    case target source shifts: []

# start on axis, zero on every other of remaining axes.
def shifts_at(remaining: Nat,+axis: U32,+start: U32,+index: U32) -> List<&2,U32>:
  match remaining:
    case 0n: []
    case 1n+rest: Bool.pick(U32,U32.is_eq(index,axis),start,0) <> shifts_at(rest,axis,start,(index + 1 : U32))

# The part has the extents of the target but along axis, where it fits from
# start on.
def slice_fits(target: List<&2,U32>,part: List<&2,U32>,+axis: U32,+start: U32,+index: U32) -> Bool:
  match target part:
    case Nil{} Nil{}: U32.is_lt(axis,index)
    case Con{+extent,target} Con{+width,part}:
      Bool.pick(Bool,U32.is_eq(index,axis),U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),U32.is_eq(extent,width)) &&
        slice_fits(target,part,axis,start,(index + 1 : U32))
    case target part: False{}

def slice_placed(~Element: Data,empty: Bool,pieces: Pieces.Pieces(~Element),part: Pieces.Pieces(~Element),+shape: List<&2,U32>,+part_shape: List<&2,U32>,+axis: U32,+start: U32,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match empty:
    case True{}: Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}
    case False{}:
      +grid = Tensors.grid_of(shape)
      +rank = List.length(&2,U32,shape)
      +shifts = shifts_at(rank,axis,start,0)
      +whole = shifts_at(rank,0,0,0)
      Tensors.Tensor{Tensors.placed_from(~Element,
        [placed_readings(shape,shape,whole,grid,grid,False{},0),placed_readings(shape,part_shape,shifts,grid,Tensors.grid_of(part_shape),False{},0)],
        [placed_segments(shape,whole),placed_segments(part_shape,shifts)],
        Tensors.Source{pieces,Tensors.Source{part,Tensors.Ended{}}},[Tensors.origin(shape,0),Tensors.origin(part_shape,0)],shape,fill,workers),shape,fill,Tensors.Usable{},workers}

def slice_status(~Element: Data,target: Tensors.Status,part: Tensors.Status,valid: Bool,pieces: Pieces.Pieces(~Element),part_pieces: Pieces.Pieces(~Element),
  +shape: List<&2,U32>,+part_shape: List<&2,U32>,+axis: U32,+start: U32,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match target part valid:
    case Tensors.Usable{} Tensors.Usable{} True{}: slice_placed(~Element,Shapes.has_zero(part_shape),pieces,part_pieces,shape,part_shape,axis,start,fill,workers)
    case Tensors.Usable{} Tensors.Usable{} False{}: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{Tensors.Problem{"set_slice",Tensors.ShapeMismatch{shape,part_shape}}},workers}
    case Tensors.Usable{} Tensors.Failed{+problem} valid: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Failed{problem} part valid: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}

# target[..., start : start + part extent, ...] = part along axis.
def set_slice(~Element: Data,target: Tensors.Tensor<Element>,+axis: U32,+start: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+shape,+fill,target_status,target_workers} = target
  Tensors.Tensor{part_pieces,+part_shape,part_fill,part_status,part_workers} = part
  slice_status(~Element,target_status,part_status,slice_fits(shape,part_shape,axis,start,0),pieces,part_pieces,shape,part_shape,axis,start,fill,U32.max(target_workers,part_workers))

# ---- Concatenation along an axis ----

type Survey is Data:
  Survey{valid: Bool,total: U32,status: Tensors.Status,workers: U32}

def surveyed(survey: Survey,+shape: List<&2,U32>,+axis: Nat,status: Tensors.Status,workers: U32) -> Survey:
  Survey{valid,+total,previous,most} = survey
  +width = extent_at(shape,axis)
  Survey{valid && U32.is_le(width,(4294967295 - total : U32)),(total + width : U32),Tensors.first_failure(previous,status),U32.max(most,workers)}

def prepend_survey(~Element: Data,tensor: Tensors.Tensor<Element>,result: Tensors.Parts<Element> & Survey) -> Tensors.Parts<Element> & Survey:
  (rest,summary) = result
  (Tensors.Cons{tensor,rest},summary)

# Summarizes the extents of every part and returns the parts in order.
def survey(~Element: Data,parts: Tensors.Parts<Element>,+axis: Nat,summary: Survey) -> Tensors.Parts<Element> & Survey:
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        (Tensors.Single{tensor},surveyed(summary,shape,axis,status,workers)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        prepend_survey(~Element,tensor,survey(~Element,rest,axis,surveyed(summary,shape,axis,status,workers))))

def prepend_fit(~Element: Data,tensor: Tensors.Tensor<Element>,result: Tensors.Parts<Element> & Bool) -> Tensors.Parts<Element> & Bool:
  (rest,fits) = result
  (Tensors.Cons{tensor,rest},fits)

# Every part fits the joined extents along axis from where the parts before
# it end, as a part of set_slice does.
def fitting(~Element: Data,parts: Tensors.Parts<Element>,+target: List<&2,U32>,+axis: U32,+start: U32,fits: Bool) -> Tensors.Parts<Element> & Bool:
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Bool),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        (Tensors.Single{tensor},fits && slice_fits(target,shape,axis,start,0)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Bool),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        prepend_fit(~Element,tensor,fitting(~Element,rest,target,axis,(start + extent_at(shape,U32.to_nat(axis)) : U32),fits && slice_fits(target,shape,axis,start,0))))

# The origins, readings and segments of the parts of a concatenation.
type Planned is Data:
  Planned{origins: List<&2,Tensors.Origin>,readings: List<&2,List<&2,Tasks.Reading>>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>}

def planned_part(~Element: Data,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+axis: U32,+start: U32,+target: List<&2,U32>,+target_grid: Grid.Grid,
  later: Tensors.Sources<Element> & Planned) -> Tensors.Sources<Element> & Planned:
  (sources,planned) = later
  Planned{origins,readings,segments} = planned
  +shifts = shifts_at(List.length(&2,U32,shape),axis,start,0)
  (Tensors.Source{pieces,sources},Planned{Tensors.origin(shape,0) <> origins,placed_readings(target,shape,shifts,target_grid,Tensors.grid_of(shape),False{},0) <> readings,
    placed_segments(shape,shifts) <> segments})

# Every part placed along axis from start on, in order.
def concat_sources(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,+start: U32,+target: List<&2,U32>,+target_grid: Grid.Grid) -> Tensors.Sources<Element> & Planned:
  match parts:
    case Tensors.Single{Tensors.Tensor{pieces,+shape,fill,status,workers}}:
      planned_part(~Element,pieces,shape,axis,start,target,target_grid,(Tensors.Ended{},Planned{[],[],[]}))
    case Tensors.Cons{Tensors.Tensor{pieces,+shape,fill,status,workers},rest}:
      planned_part(~Element,pieces,shape,axis,start,target,target_grid,concat_sources(~Element,rest,axis,(start + extent_at(shape,U32.to_nat(axis)) : U32),target,target_grid))

def joined_parts(~Element: Data,+shape: List<&2,U32>,+fill: Element,+workers: U32,planned: Tensors.Sources<Element> & Planned) -> Tensors.Tensor<Element>:
  (sources,plan) = planned
  Planned{+origins,+readings,+segments} = plan
  Tensors.Tensor{Tensors.placed_from(~Element,readings,segments,sources,origins,shape,fill,workers),shape,fill,Tensors.Usable{},workers}

def join_counted(~Element: Data,count: Maybe<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match count:
    case Some{count}: joined_parts(~Element,shape,fill,workers,concat_sources(~Element,parts,axis,0,shape,Tensors.grid_of(shape)))
    case None{}: Tensors.failed(~Element,"concat",Tensors.InvalidShape{shape},shape,fill,workers)

# An empty result has no element to place.
def join_shaped(~Element: Data,empty: Bool,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match empty:
    case False{}: join_counted(~Element,Tensors.count_of(shape),shape,axis,fill,workers,parts)
    case True{}: Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),workers)

def join_status(~Element: Data,status: Tensors.Status,valid: Bool,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Usable{} True{}: join_shaped(~Element,Shapes.has_zero(shape),shape,axis,fill,workers,parts)
    case Tensors.Usable{} False{}: Tensors.failed(~Element,"concat",Tensors.InvalidArgument{first},first,fill,workers)
    case Tensors.Failed{problem} other: Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),first,fill,Tensors.Failed{problem},workers)

def fitted_status(~Element: Data,status: Tensors.Status,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,
  fitted: Tensors.Parts<Element> & Bool) -> Tensors.Tensor<Element>:
  (parts,valid) = fitted
  join_status(~Element,status,valid,first,shape,axis,fill,workers,parts)

def joined(~Element: Data,+first: List<&2,U32>,+axis: U32,+fill: Element,surveyed: Tensors.Parts<Element> & Survey) -> Tensors.Tensor<Element>:
  (parts,summary) = surveyed
  Survey{valid,+total,status,+workers} = summary
  +shape = with_extent(first,U32.to_nat(axis),total)
  fitted_status(~Element,status,first,shape,axis,fill,workers,fitting(~Element,parts,shape,axis,0,valid))

def lead(~Element: Data,parts: Tensors.Parts<Element>) -> Tensors.Parts<Element> & (List<&2,U32> & Element):
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & (List<&2,U32> & Element)),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +fill = fill
        (Tensors.Single{tensor},(shape,fill)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & (List<&2,U32> & Element)),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +fill = fill
        (Tensors.Cons{tensor,rest},(shape,fill)))

def concat_first(~Element: Data,+axis: U32,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+fill: Element) -> Tensors.Tensor<Element>:
  joined(~Element,shape,axis,fill,
    survey(~Element,parts,U32.to_nat(axis),Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),0,Tensors.Usable{},1}))

def concat_led(~Element: Data,+axis: U32,led: Tensors.Parts<Element> & (List<&2,U32> & Element)) -> Tensors.Tensor<Element>:
  (parts,first) = led
  (shape,fill) = first
  concat_first(~Element,axis,parts,shape,fill)

# Parts agree on every extent except axis; the result stacks them in order.
def concat(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32) -> Tensors.Tensor<Element>:
  concat_led(~Element,axis,lead(~Element,parts))

# ---- Padding ----

# Constant fills the new elements with one value; Edge repeats the nearest
# element of the tensor along every axis, so a corner holds the nearest
# corner element.
type Border<-Element: Data> is Data:
  Constant{value: Element}
  Edge{}

def border_name(~Element: Data,border: Border<Element>) -> String:
  match border:
    case Constant{value}: "pad"
    case Edge{}: "pad_edge"

def all_zero(values: List<&2,U32>) -> Bool:
  match values:
    case []: True{}
    case value <> rest: U32.is_zero(value) && all_zero(rest)

def padded_count(shape: Maybe<&2,List<&2,U32>>) -> Maybe<&2,U32>:
  match shape:
    case Some{shape}: Tensors.count_of(shape)
    case None{}: None{}

# An axis padded at an edge has an element to repeat.
def edge_axes(shape: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>) -> Bool:
  match shape before after:
    case Con{+extent,shape} Con{+front,before} Con{+back,after}:
      (U32.is_lt(0,extent) || (U32.is_zero(front) && U32.is_zero(back))) && edge_axes(shape,before,after)
    case shape before after: True{}

def border_fits(~Element: Data,border: Border<Element>,+shape: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>) -> Bool:
  match border:
    case Constant{value}: True{}
    case Edge{}: edge_axes(shape,before,after)

def pad_placed(~Element: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),+later: Later,border: Border<Element>,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  +before: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match border:
    case Constant{+value}:
      Tensors.Tensor{Tensors.placed_then(~Element,~Later,~after,later,[placed_readings(target,shape,before,Tensors.grid_of(target),Tensors.grid_of(shape),False{},0)],[placed_segments(shape,before)],
        Tensors.Source{pieces,Tensors.Ended{}},[Tensors.origin(shape,0)],target,value,workers),target,fill,Tensors.Usable{},workers}
    case Edge{}:
      Tensors.Tensor{Tensors.placed_then(~Element,~Later,~after,later,[placed_readings(target,shape,before,Tensors.grid_of(target),Tensors.grid_of(shape),True{},0)],[edge_segments(target,shape,before)],
        Tensors.Source{pieces,Tensors.Ended{}},[Tensors.origin(shape,0)],target,fill,workers),target,fill,Tensors.Usable{},workers}

def pad_valid(~Element: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),+later: Later,border: Border<Element>,valid: Bool,unchanged: Bool,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+before: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match valid unchanged:
    case True{} True{}: Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}
    case True{} False{}: pad_placed(~Element,~Later,~after,later,border,pieces,shape,before,target,fill,workers)
    case False{} unchanged: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{Tensors.Problem{border_name(~Element,border),Tensors.InvalidArgument{shape}}},workers}

def pad_status(~Element: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),+later: Later,+border: Border<Element>,status: Tensors.Status,padded: Maybe<&2,List<&2,U32>>,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  +before: List<&2,U32>,+behind: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status padded:
    case Tensors.Usable{} Some{+target}:
      pad_valid(~Element,~Later,~after,later,border,Views.has_count(Tensors.count_of(target)) && border_fits(~Element,border,shape,before,behind),all_zero(before) && all_zero(behind),
        pieces,shape,before,target,fill,workers)
    case Tensors.Usable{} None{}: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{Tensors.Problem{border_name(~Element,border),Tensors.InvalidArgument{shape}}},workers}
    case Tensors.Failed{problem} padded: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}

# The padding, then a map of every piece it fills.
def pad_then(~Element: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),+later: Later,+border: Border<Element>,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,
  +behind: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+shape,+fill,status,+workers} = tensor
  pad_status(~Element,~Later,~after,later,border,status,Shapes.padded(shape,before,behind),pieces,shape,before,behind,fill,workers)

# ---- Reductions ----

# Every listed axis is below the rank.
# Whether each of remaining axes from index on is listed.
def reduced_flags(remaining: Nat,+index: Nat,+reduced: List<&2,Nat>) -> List<&2,Bool>:
  match remaining:
    case 0n: []
    case 1n+rest: Views.contains_index(reduced,index) <> reduced_flags(rest,1n+index,reduced)

def all_below(axes: List<&2,U32>,+rank: Nat) -> Bool:
  match axes:
    case []: True{}
    case axis <> rest: Nat.is_lt(U32.to_nat(axis),rank) && all_below(rest,rank)

# Reduction over listed axes. Output k, at the row-major index of the kept
# axes, folds the values of the reduced axes in row-major order, from
# initial, and holds finish of the fold:
# finish(operation(... operation(operation(initial, x[k, 0]), x[k, 1]) ..., x[k, R - 1])).
def chosen(member: Bool,+listed: Bool,+index: Nat,rest: List<&2,Nat>) -> List<&2,Nat>:
  match member listed:
    case True{} True{}: index <> rest
    case False{} False{}: index <> rest
    case member listed: rest

def axes_where(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+listed: Bool) -> List<&2,Nat>:
  match remaining:
    case 0n: []
    case 1n+rest: chosen(Views.contains_index(axes,index),listed,index,axes_where(rest,1n+index,axes,listed))

def extents_at(+shape: List<&2,U32>,indices: List<&2,Nat>) -> List<&2,U32>:
  match indices:
    case []: []
    case index <> rest: extent_at(shape,index) <> extents_at(shape,rest)

# The cells of extents read with the kept axes first and the reduced last.
def line_view(+extents: List<&2,U32>,+order: List<&2,Nat>) -> Maybe<&2,Views.View>:
  Views.permute(Views.View{Tasks.piece_cells(extents),0},order)

# The axes index, index + 1, ... of a shape, remaining of them.
def every_index(remaining: Nat,+index: Nat) -> List<&2,Nat>:
  match remaining:
    case 0n: []
    case 1n+rest: index <> every_index(rest,1n+index)

def cells_per_outer(+grid: Grid.Grid) -> U32:
  Grid.Grid{+count,prefix,+outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  (count / U32.max(outer,1) : U32)

# The lines of the reduced axes, last in the cells, fold from initial across
# the pieces in order. A task holds the pieces of whole outer indices.
def reduce_trailing(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,+length: U32,
  +initial: Accumulator,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+outputs: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  +grid = Tensors.grid_of(shape)
  Tensors.with_workers(~Element,Tensors.from_list(~Element,outputs,
    Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,
      LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),
        ~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(cells_per_outer(grid),1) : U32),1.0,operations,grid,fill,workers,pieces)),fill),workers)

# The reduced axes last: kept of the axes kept, the rest reduced.
def reduce_arranged(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,+length: U32,
  +initial: Accumulator,+kept: Nat,arranged: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+shape,+fill,status,+workers} = arranged
  match status:
    case Tensors.Usable{}: reduce_trailing(~Element,~Accumulator,~operation,~finish,operations,length,initial,pieces,shape,extents_at(shape,every_index(kept,0n)),fill,workers)
    case Tensors.Failed{problem}: Tensors.Tensor{pieces,extents_at(shape,every_index(kept,0n)),fill,Tensors.Failed{problem},workers}

# The reduced axes move last unless they are.
def reduce_moved(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,+length: U32,
  +initial: Accumulator,trailing: Bool,view: Maybe<&2,Views.View>,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+kept: List<&2,Nat>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  match trailing view:
    case True{} view:
      reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})
    case False{} Some{Views.View{+axes,+offset}}:
      reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),
        Tensors.moved(~Element,"reduce",Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axes,offset,[]))
    case False{} None{}: Tensors.failed(~Element,"reduce",Tensors.InvalidArgument{shape},extents_at(shape,kept),fill,workers)

# The axes index, index + 1, ... in this order.
def in_order(axes: List<&2,Nat>,+index: Nat) -> Bool:
  match axes:
    case Nil{}: True{}
    case Con{axis,rest}: Nat.is_eq(axis,index) && in_order(rest,1n+index)

def reduce_lined(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,+length: U32,
  +initial: Accumulator,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  +order = List.append(&2,Nat,kept,reduced)
  reduce_moved(~Element,~Accumulator,~operation,~finish,operations,length,initial,in_order(order,0n),line_view(shape,order),pieces,shape,kept,fill,workers)

# No output, or outputs that fold no value.
def reduce_sized(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,empty: Bool,unread: Bool,
  +count: U32,+length: U32,+initial: Accumulator,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  match empty unread:
    case True{} unread: Tensors.Tensor{Tensors.filled(~Element,extents_at(shape,kept),fill),extents_at(shape,kept),fill,Tensors.Usable{},workers}
    case False{} True{}: Tensors.Tensor{Tensors.filled(~Element,extents_at(shape,kept),finish(initial)),extents_at(shape,kept),fill,Tensors.Usable{},workers}
    case False{} False{}:
      reduce_lined(~Element,~Accumulator,~operation,~finish,operations,length,initial,pieces,shape,kept,reduced,fill,workers)

def reduce_counted(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,count: Maybe<&2,U32>,length: Maybe<&2,U32>,+initial: Accumulator,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match count length:
    case Some{+count} Some{+length}:
      reduce_sized(~Element,~Accumulator,~operation,~finish,operations,U32.is_zero(count),U32.is_zero(length),count,length,initial,pieces,shape,kept,reduced,fill,workers)
    case count length: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},extents_at(shape,kept),fill,workers)

# The reduction of the axes reduced, the others kept, both in increasing
# index. A failed tensor keeps its problem; an invalid list of axes fails.
def reduce_listed(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,name: String,
  +operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<Element>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,valid: Bool) -> Tensors.Tensor<Element>:
  Tensors.Tensor{pieces,+shape,+fill,status,+workers} = tensor
  match status valid:
    case Tensors.Usable{} True{}:
      reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Tensors.count_of(extents_at(shape,kept)),Tensors.count_of(extents_at(shape,reduced)),
        initial,pieces,shape,kept,reduced,fill,workers)
    case Tensors.Usable{} False{}: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{shape}}},workers}
    case Tensors.Failed{problem} valid: Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}

# The result has the extents of the kept axes. Every listed axis is below the
# rank and listed once. operations prices the combination of one value.
def reduce_axes(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,name: String,
  +operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<Element>,+axes: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_metadata(~Element,~Tensors.Tensor<Element>,tensor,tensor => shape => fill => status => workers =>
    +rank = List.length(&2,U32,shape)
    reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,tensor,
      axes_where(rank,0n,naturals(axes),False{}),axes_where(rank,0n,naturals(axes),True{}),
      Views.unique_axes(naturals(axes)) && all_below(axes,rank)))

# ---- Reversal ----

# The axes of a shape read in reverse: one flag an axis.
def reversed_flags(remaining: Nat,+index: U32,+axes: List<&2,U32>) -> List<&2,Bool>:
  match remaining:
    case 0n: []
    case 1n+rest: Views.contains_index(naturals(axes),U32.to_nat(index)) <> reversed_flags(rest,(index + 1 : U32),axes)

def flipped(valid: Bool,view: Maybe<&2,Views.View>) -> Maybe<&2,Views.View>:
  match valid:
    case True{}: view
    case False{}: None{}
