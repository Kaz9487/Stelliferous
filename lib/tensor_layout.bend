# Layout operations consume their arguments. Copy keeps the bands and reshape
# relabels them. Transpose, slice, flip, upsampling and set_slice go through a view
# of the input: bands whose images and rows the view keeps in place run it on
# their own rows, other inputs first arrive as one band.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./traversal_partition.bend as Partition
import ./tensor_band_concat.bend as BandConcat
import ./tensor_band_schedule.bend as BandSchedule
import ./tensor_band_view.bend as BandViews
import ./tensor_band_reduce.bend as BandReduce

def extents(view: Maybe<&2,Views.View>) -> List<&2,U32>:
  match view:
    case Some{view}: Views.axis_extents_of(view)
    case None{}: []

def naturals(values: List<&2,U32>) -> List<&2,Nat>:
  match values:
    case []: []
    case value <> rest: U32.to_nat(value) <> naturals(rest)

# The only layout operation that keeps its argument: (original, copy).
def copy_bands(~Element: Data,status: Tensors.Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  match status:
    case Tensors.Usable{}:
      Storage.bind_pair(~Bands.Bands(~Element),~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Tensors.Tensor<Element>),
        Bands.copy(~Element,bands),left => right =>
          (Tensors.Tensor{left,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{right,geometry,shape,fill,Tensors.Usable{},workers}))
    case Tensors.Failed{+problem}:
      (Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers},
        Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),shape,fill,Tensors.Failed{problem},workers))

def copy(~Element: Data,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  Tensors.Tensor{bands,geometry,shape,fill,status,workers} = tensor
  copy_bands(~Element,status,bands,geometry,shape,fill,workers)

def permuted(view: Maybe<&2,Views.View>,order: List<&2,Nat>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.permute(view,order)
    case None{}: None{}

# The result's axis i is the source axis order[i].
def reordered(~Element: Data,name: String,tensor: Tensors.Tensor<Element>,order: List<&2,Nat>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  Tensors.gather_extents(~Element,name,status,permuted(Tensors.view_of(shape),order),[],bands,geometry,shape,fill,workers)

def transpose(~Element: Data,tensor: Tensors.Tensor<Element>,order: List<&2,U32>) -> Tensors.Tensor<Element>:
  reordered(~Element,"transpose",tensor,naturals(order))

def sliced(view: Maybe<&2,Views.View>,axis: U32,start: U32,count: U32,step: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,U32.to_nat(axis),start,count,step)
    case None{}: None{}

# Elements start, start + step, ... (count of them) along one axis.
def slice(~Element: Data,tensor: Tensors.Tensor<Element>,axis: U32,start: U32,count: U32,step: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  Tensors.gather_extents(~Element,"slice",status,sliced(Tensors.view_of(shape),axis,start,count,step),[],bands,geometry,shape,fill,workers)

def same_count(left: Maybe<&2,U32>,right: Maybe<&2,U32>) -> Bool:
  match left right:
    case Some{left} Some{right}: U32.is_eq(left,right)
    case left right: False{}

# The bands keep their geometry and order under the target shape when it
# names as many elements.
def relabel_ready(~Element: Data,accepted: Bool,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,fill: Element,
  workers: U32) -> Tensors.Tensor<Element>:
  match accepted:
    case True{}: Tensors.Tensor{bands,geometry,target,fill,Tensors.Usable{},workers}
    case False{}: Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{Tensors.Problem{"reshape",Tensors.ShapeMismatch{shape,target}}},workers}

def relabel(~Element: Data,accepted: Bool,current: Tensors.Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  fill: Element,workers: U32) -> Tensors.Tensor<Element>:
  match current:
    case Tensors.Usable{}: relabel_ready(~Element,accepted,bands,geometry,shape,target,fill,workers)
    case Tensors.Failed{problem}: Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}

# Same elements in the same row-major order under new extents.
def reshape(~Element: Data,tensor: Tensors.Tensor<Element>,+shape: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+source_shape,fill,status,workers} = tensor
  relabel(~Element,same_count(Tensors.count_of(source_shape),Tensors.count_of(shape)),status,bands,geometry,source_shape,shape,fill,workers)

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

def upsample_checked(~Element: Data,valid: Bool,planes: Maybe<&2,U32>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match valid planes:
    case True{} Some{planes}:
      Tensors.gather(~Element,"upsample_nearest",Tensors.Usable{},repeated(Tensors.view_of(shape),scale),[],bands,geometry,shape,scaled_extents(shape,scale),fill,workers)
    case valid planes: Tensors.failed(~Element,"upsample_nearest",Tensors.InvalidArgument{shape},scaled_extents(shape,scale),fill,workers)

# The repeated extents and the output count fit the machine word.
def upsample_valid(+shape: List<&2,U32>,+scale: U32) -> Bool:
  +height = extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n))
  +width = extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
  Views.has_count(Shapes.checked_multiply(height,scale)) && Views.has_count(Shapes.checked_multiply(width,scale)) &&
    Views.has_count(Tensors.count_of(scaled_extents(shape,scale)))

def upsample_status(~Element: Data,status: Tensors.Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,
  +fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status:
    case Tensors.Failed{problem}: Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Usable{}: upsample_checked(~Element,upsample_valid(shape,scale),planes(shape),bands,geometry,shape,scale,fill,workers)

# Nearest-neighbour upsampling of the last two axes: one gather through the
# view that repeats each row and each column scale times.
def upsample_nearest(~Element: Data,tensor: Tensors.Tensor<Element>,+scale: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  upsample_status(~Element,status,bands,geometry,shape,scale,fill,workers)

def settle(~Element: Data,valid: Bool,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool:
  match result:
    case Operations.Accepted{owners}: (owners,valid)
    case Operations.Rejected{owners}: (owners,False{})

# What every band of a placement shares: the banded slice of the target
# geometry.
type Placing is Data:
  Placing{banded: BandViews.Banded,geometry: Bands.Geometry}

def placed_outcome(~Element: Data,+first: U32,+rows: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Bands.Transformed(~Element):
  match result:
    case Operations.Accepted{(part,target)}: Partition.Leaf{Bands.Checked{target,True{}},first,rows,Unit{}}
    case Operations.Rejected{(part,target)}: Partition.Leaf{Bands.Checked{target,False{}},first,rows,Unit{}}

def placed_viewed(~Element: Data,source: Maybe<&2,Views.View>,slice: Views.View,+first: U32,+rows: U32,target: Storage.Buffer<Element>,part: Storage.Buffer<Element>) ->
  Bands.Transformed(~Element):
  match source:
    case Some{source}: placed_outcome(~Element,first,rows,Operations.scatter(~Element,part,source,target,slice))
    case None{}: Partition.Leaf{Bands.Checked{target,False{}},first,rows,Unit{}}

# A band of the target takes the band of the part of the same rows, written
# through the local view of the slice.
def placed_band(~Element: Data,context: Placing,first: U32,rows: U32,target: Storage.Buffer<Element>,part: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  Placing{+banded,+geometry} = context
  +rows = rows
  placed_viewed(~Element,Tensors.view_of(BandViews.extents(banded,rows)),BandViews.local(banded,geometry,rows),first,rows,target,part)

def place_planned(~Element: Data,+name: String,+banded: BandViews.Banded,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  part: Tensors.Tensor<Element>,owners: Bands.Bands(~Element) & Bands.Plan) -> Tensors.Tensor<Element>:
  (target,+layout) = owners
  Tensors.transformed(~Element,name,Tensors.InvalidArgument{target_shape},geometry,target_shape,fill,workers,
    Bands.transform_paired(~Element,~Placing,~placed_band(~Element),layout,BandSchedule.parallel(geometry,workers,12.0,0.0,layout),Placing{banded,geometry},
      Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,target,Tensors.arrive(~Element,BandViews.output(banded,geometry),layout,part))))

def mismatched(~Element: Data,name: String,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{target_shape,part_shape}}},workers}

def place_whole(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+whole: Bands.Geometry,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,
  part: Tensors.Tensor<Element>,target: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match found:
    case Some{+banded}: place_planned(~Element,name,banded,whole,target_shape,fill,workers,part,Bands.plan(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),target)))
    case None{}: mismatched(~Element,name,Tensors.arrive(~Element,whole,Bands.unsplit(whole),target),whole,target_shape,part_shape,fill,workers)

# The target first arrives as one band of one row.
def place_target(~Element: Data,+name: String,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)} : Bands.Geometry}
  place_whole(~Element,name,BandViews.classify(axes,offset,whole),whole,target_shape,part_shape,fill,workers,part,Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Usable{},workers})

# The part arrives in the rows of the target, so a slice writes band by band
# only when it repeats no row.
def place_rows(~Element: Data,+name: String,repeats: List<&2,U32>,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+axes: List<&2,Views.Axis>,
  +offset: U32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,
  part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match repeats:
    case []: place_planned(~Element,name,BandViews.Banded{before,[],after,start},geometry,target_shape,fill,workers,part,Bands.plan(~Element,bands))
    case extent <> rest: place_target(~Element,name,axes,offset,bands,geometry,target_shape,part_shape,fill,workers,part)

# Target bands whose images and rows the slice keeps in place take the part
# band by band; other targets first arrive as one band of one row.
def place_found(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,
  +target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match found:
    case Some{BandViews.Banded{+before,repeats,+after,+start}}:
      place_rows(~Element,name,repeats,before,after,start,axes,offset,bands,geometry,target_shape,part_shape,fill,workers,part)
    case None{}: place_target(~Element,name,axes,offset,bands,geometry,target_shape,part_shape,fill,workers,part)

# A part of the extents of the slice is written; a part with no element
# writes nothing.
def place_checked(~Element: Data,+name: String,same: Bool,empty: Bool,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match same empty:
    case True{} True{}: Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Usable{},workers}
    case True{} False{}: place_found(~Element,name,BandViews.classify(axes,offset,geometry),axes,offset,bands,geometry,target_shape,part_shape,fill,workers,part)
    case False{} empty: mismatched(~Element,name,bands,geometry,target_shape,part_shape,fill,workers)

# The slice comes taken apart, so that no View value is used twice: that
# would make every View constructor shared.
def place_slice(~Element: Data,+name: String,slice: Maybe<&2,Views.View>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,
  +fill: Element,+workers: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match slice:
    case Some{Views.View{+axes,+offset}}:
      place_checked(~Element,name,Shapes.same_extents(Views.axis_extents(axes),part_shape),Shapes.has_zero(part_shape),axes,offset,bands,geometry,target_shape,part_shape,fill,
        workers,part)
    case None{}: mismatched(~Element,name,bands,geometry,target_shape,part_shape,fill,workers)

def place_status(~Element: Data,+name: String,target_status: Tensors.Status,part_status: Tensors.Status,slice: Maybe<&2,Views.View>,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>,+fill: Element,+workers: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match target_status part_status:
    case Tensors.Usable{} Tensors.Usable{}: place_slice(~Element,name,slice,bands,geometry,target_shape,part_shape,fill,workers,part)
    case Tensors.Usable{} Tensors.Failed{+problem}: Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Failed{problem} part_status: Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Failed{problem},workers}

# target[..., start : start + part extent, ...] = part along axis: the part is
# written through the slice of the target that it replaces.
def set_slice(~Element: Data,target: Tensors.Tensor<Element>,+axis: U32,start: U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+target_shape,+fill,target_status,target_workers} = target
  Tensors.Tensor{part_bands,part_geometry,+part_shape,part_fill,part_status,part_workers} = part
  +workers = U32.max(target_workers,part_workers)
  place_status(~Element,"set_slice",target_status,part_status,sliced(Tensors.view_of(target_shape),axis,start,extent_at(part_shape,U32.to_nat(axis)),1),bands,geometry,
    target_shape,part_shape,fill,workers,Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,Tensors.Usable{},workers})

# Concatenation along an axis.
type Survey is Data:
  Survey{valid: Bool,total: U32,status: Tensors.Status,workers: U32}

def surveyed(survey: Survey,+first: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,status: Tensors.Status,workers: U32) -> Survey:
  Survey{valid,+total,previous,most} = survey
  +width = extent_at(shape,axis)
  Survey{valid && Nat.is_eq(List.length(&2,U32,first),List.length(&2,U32,shape)) &&
      Shapes.same_extents(List.take(&2,U32,first,axis),List.take(&2,U32,shape,axis)) &&
      Shapes.same_extents(List.drop(&2,U32,first,1n+axis),List.drop(&2,U32,shape,1n+axis)) &&
      U32.is_le(width,(4294967295 - total : U32)),
    (total + width : U32),Tensors.first_failure(previous,status),U32.max(most,workers)}

def prepend_survey(~Element: Data,tensor: Tensors.Tensor<Element>,result: Tensors.Parts<Element> & Survey) -> Tensors.Parts<Element> & Survey:
  (rest,summary) = result
  (Tensors.Cons{tensor,rest},summary)

# Summarizes the extents of every part and returns the parts in order.
def survey(~Element: Data,parts: Tensors.Parts<Element>,+first: List<&2,U32>,+axis: Nat,summary: Survey) -> Tensors.Parts<Element> & Survey:
  match parts:
    case Tensors.Single{tensor}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        (Tensors.Single{tensor},surveyed(summary,first,shape,axis,status,workers)))
    case Tensors.Cons{tensor,rest}:
      Tensors.with_metadata(~Element,~(Tensors.Parts<Element> & Survey),tensor,tensor => shape => fill => status => workers =>
        +shape = shape
        +status = status
        +workers = workers
        prepend_survey(~Element,tensor,survey(~Element,rest,first,axis,surveyed(summary,first,shape,axis,status,workers))))

def with_extent(extents: List<&2,U32>,axis: Nat,+extent: U32) -> List<&2,U32>:
  List.set(&2,U32,extents,axis,extent)

def appended_shaped(~Element: Data,+axis: U32,+start: U32,output: Tensors.Tensor<Element>,shaped: Tensors.Tensor<Element> & List<&2,U32>) -> Tensors.Tensor<Element> & U32:
  (part,+shape) = shaped
  (set_slice(~Element,output,axis,start,part),(start + extent_at(shape,U32.to_nat(axis)) : U32))

def append_at(~Element: Data,+axis: U32,+start: U32,output: Tensors.Tensor<Element>,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & U32:
  appended_shaped(~Element,axis,start,output,Tensors.shape(~Element,part))

def append_part(~Element: Data,+axis: U32,state: Tensors.Tensor<Element> & U32,part: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & U32:
  (output,start) = state
  append_at(~Element,axis,start,output,part)

def appended_all(~Element: Data,state: Tensors.Tensor<Element> & U32) -> Tensors.Tensor<Element>:
  (output,start) = state
  output

def append_parts(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32,state: Tensors.Tensor<Element> & U32) -> Tensors.Tensor<Element>:
  match parts:
    case Tensors.Single{part}: appended_all(~Element,append_part(~Element,axis,state,part))
    case Tensors.Cons{part,rest}: append_parts(~Element,rest,axis,append_part(~Element,axis,state,part))

# The leaf concat of the bands of the parts at one band of the output, along
# the middle axis of [outer, extent, inner], into a tensor of that band shape.
def band_joined(~Element: Data,parts: Tensors.Parts<Element>,shape: List<&2,U32>,fill: Element) -> Tensors.Tensor<Element>:
  append_parts(~Element,parts,1,(Tensors.with_workers(~Element,Tensors.full(~Element,shape,fill),1),0))

# The parts meet in one plan of the rows of their joining, and every band of
# the output appends the bands of the parts there.
def join_counted(~Element: Data,count: Maybe<&2,U32>,+shape: List<&2,U32>,+axis: U32,+fill: Element,+workers: U32,parts: Tensors.Parts<Element>) -> Tensors.Tensor<Element>:
  match count:
    case Some{count}:
      +joining = BandConcat.joining(shape,axis)
      BandConcat.met(~Element,~band_joined(~Element),joining,shape,fill,workers,BandConcat.offers(~Element,joining,parts))
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

def joined(~Element: Data,+first: List<&2,U32>,+axis: U32,+fill: Element,surveyed: Tensors.Parts<Element> & Survey) -> Tensors.Tensor<Element>:
  (parts,summary) = surveyed
  Survey{valid,total,status,workers} = summary
  join_status(~Element,status,valid,first,with_extent(first,U32.to_nat(axis),total),axis,fill,workers,parts)

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
    survey(~Element,parts,shape,U32.to_nat(axis),Survey{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),0,Tensors.Usable{},1}))

def concat_led(~Element: Data,+axis: U32,led: Tensors.Parts<Element> & (List<&2,U32> & Element)) -> Tensors.Tensor<Element>:
  (parts,first) = led
  (shape,fill) = first
  concat_first(~Element,axis,parts,shape,fill)

# Parts agree on every extent except axis; the result stacks them in order.
def concat(~Element: Data,parts: Tensors.Parts<Element>,+axis: U32) -> Tensors.Tensor<Element>:
  concat_led(~Element,axis,lead(~Element,parts))

# Every listed axis is below the rank.
def all_below(axes: List<&2,U32>,+rank: Nat) -> Bool:
  match axes:
    case []: True{}
    case axis <> rest: Nat.is_lt(U32.to_nat(axis),rank) && all_below(rest,rank)

# Reduction over listed axes. Output k, at the row-major index of the kept
# axes, folds the values of the reduced axes in row-major order, from
# initial, and holds finish of the fold:
# finish(operation(... operation(operation(initial, x[k, 0]), x[k, 1]) ..., x[k, R - 1])).
# The axes after the kept ones, in increasing index, are the reduced ones.
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

def reduce_counted(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,count: Maybe<&2,U32>,length: Maybe<&2,U32>,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,
  +workers: U32,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match count length:
    case Some{+count} Some{+length}:
      BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,extents_at(shape,kept),fill,workers,
        tensor,permuted(Tensors.view_of(shape),List.append(&2,Nat,kept,reduced)),List.length(&2,Nat,reduced))
    case count length: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},extents_at(shape,kept),fill,workers)

# The reduction of the axes reduced, the others kept, both in increasing
# index. A failed tensor keeps its problem; an invalid list of axes fails.
def reduce_listed(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,name: String,
  +operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<Element>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,valid: Bool) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  match status valid:
    case Tensors.Usable{} True{}:
      reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Tensors.count_of(extents_at(shape,kept)),Tensors.count_of(extents_at(shape,reduced)),
        initial,shape,kept,reduced,fill,workers,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})
    case Tensors.Usable{} False{}: Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{shape}}},workers}
    case Tensors.Failed{problem} valid: Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}

# The result has the extents of the kept axes. Every listed axis is below the
# rank and listed once. operations prices the combination of one value.
def reduce_axes(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,name: String,
  +operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<Element>,+axes: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_metadata(~Element,~Tensors.Tensor<Element>,tensor,tensor => shape => fill => status => workers =>
    +rank = List.length(&2,U32,shape)
    reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,tensor,
      axes_where(rank,0n,naturals(axes),False{}),axes_where(rank,0n,naturals(axes),True{}),
      Views.unique_axes(naturals(axes)) && all_below(axes,rank)))

# The axes index, index + 1, ... of a shape, remaining of them.
def every_index(remaining: Nat,+index: Nat) -> List<&2,Nat>:
  match remaining:
    case 0n: []
    case 1n+rest: index <> every_index(rest,1n+index)

# The reduction of every element: no axis kept, every axis reduced.
def reduce_all(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,name: String,
  +operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.with_metadata(~Element,~Tensors.Tensor<Element>,tensor,tensor => shape => fill => status => workers =>
    reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,tensor,[],every_index(List.length(&2,U32,shape),0n),True{}))

# Reversal along axes: one gather of the view of the tensor with the listed
# axes read in reverse.
def reversed_flags(remaining: Nat,+index: U32,+axes: List<&2,U32>) -> List<&2,Bool>:
  match remaining:
    case 0n: []
    case 1n+rest: Views.contains_index(naturals(axes),U32.to_nat(index)) <> reversed_flags(rest,(index + 1 : U32),axes)

def flipped(valid: Bool,view: Maybe<&2,Views.View>) -> Maybe<&2,Views.View>:
  match valid:
    case True{}: view
    case False{}: None{}

# The elements in reverse order along every listed axis; an axis is listed once.
def flip(~Element: Data,tensor: Tensors.Tensor<Element>,+axes: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  +rank = List.length(&2,U32,shape)
  Tensors.gather_extents(~Element,"flip",status,flipped(Views.unique_axes(naturals(axes)) && all_below(axes,rank),Tensors.view_of(shape)),
    reversed_flags(rank,0,axes),bands,geometry,shape,fill,workers)

# Padding. Constant fills the new elements with one value and writes the
# tensor once, whatever the axes; Edge repeats the nearest element of the
# tensor one axis after another, so a corner holds the nearest corner element.
type Border<-Element: Data> is Data:
  Constant{value: Element}
  Edge{}

def border_name(~Element: Data,border: Border<Element>) -> String:
  match border:
    case Constant{value}: "pad"
    case Edge{}: "pad_edge"

# The elements at one end of axis, repeated count times: (strip, tensor).
def strip_of(~Element: Data,+axis: U32,from_end: Bool,+count: U32,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  +strip = with_extent(shape,U32.to_nat(axis),count)
  Tensors.gather(~Element,"pad_edge",status,
    Tensors.stretched(sliced(Tensors.view_of(shape),axis,Bool.pick(U32,from_end,(extent_at(shape,U32.to_nat(axis)) - 1 : U32),0),1,1),strip),[],
    bands,geometry,shape,strip,fill,workers)

# The elements at one end of axis, repeated count times: (strip, tensor).
def edge_strip(~Element: Data,tensor: Tensors.Tensor<Element>,+axis: U32,from_end: Bool,+count: U32) -> Tensors.Tensor<Element> & Tensors.Tensor<Element>:
  Tensors.then(~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,~(Tensors.Tensor<Element> & Tensors.Tensor<Element>),copy(~Element,tensor),original => copied =>
    (strip_of(~Element,axis,from_end,count,copied),original))

def pad_edge_axis(~Element: Data,unchanged: Bool,tensor: Tensors.Tensor<Element>,+axis: U32,+before: U32,+after: U32) -> Tensors.Tensor<Element>:
  match unchanged:
    case True{}: tensor
    case False{}:
      Tensors.then(~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,edge_strip(~Element,tensor,axis,False{},before),first => tensor =>
        Tensors.then(~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,~Tensors.Tensor<Element>,edge_strip(~Element,tensor,axis,True{},after),last => tensor =>
          concat(~Element,Tensors.Cons{first,Tensors.Cons{tensor,Tensors.Single{last}}},axis)))

def pad_edge_axes(~Element: Data,before: List<&2,U32>,after: List<&2,U32>,tensor: Tensors.Tensor<Element>,+axis: U32) -> Tensors.Tensor<Element>:
  match before after:
    case Con{+front,before} Con{+back,after}:
      pad_edge_axes(~Element,before,after,pad_edge_axis(~Element,U32.is_zero(front) && U32.is_zero(back),tensor,axis,front,back),(axis + 1 : U32))
    case before after: tensor

# The view of the padded extents that holds the tensor: on every axis, the
# extent of the tensor starting after the elements added in front.
def interior(before: List<&2,U32>,shape: List<&2,U32>,view: Maybe<&2,Views.View>,+axis: U32) -> Maybe<&2,Views.View>:
  match before shape:
    case Con{front,before} Con{extent,shape}: interior(before,shape,sliced(view,axis,front,extent,1),(axis + 1 : U32))
    case before shape: view

def all_zero(values: List<&2,U32>) -> Bool:
  match values:
    case []: True{}
    case value <> rest: U32.is_zero(value) && all_zero(rest)

# Constant padding: storage of the padded extents filled with the value, then
# the tensor written through the interior view. One copy whatever the axes.
def padded_into(~Element: Data,count: Maybe<&2,U32>,+value: Element,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+padded: List<&2,U32>) -> Tensors.Tensor<Element>:
  match count:
    case Some{+count}:
      Tensors.Tensor{part_bands,part_geometry,+part_shape,+part_fill,part_status,+part_workers} = tensor
      +whole = {Bands.Geometry{1,1,1,count} : Bands.Geometry}
      place_status(~Element,"pad",Tensors.Usable{},part_status,interior(before,part_shape,Tensors.view_of(padded),0),
        Partition.Leaf{Tensors.buffer(~Element,value,count),0,1,Unit{}},whole,padded,part_shape,part_fill,part_workers,
        Tensors.Tensor{part_bands,part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers})
    case None{}: Tensors.with_status(~Element,Tensors.Failed{Tensors.Problem{"pad",Tensors.InvalidShape{padded}}},tensor)

def pad_constant(~Element: Data,unchanged: Bool,+value: Element,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+padded: List<&2,U32>) -> Tensors.Tensor<Element>:
  match unchanged:
    case True{}: tensor
    case False{}: padded_into(~Element,Tensors.count_of(padded),value,tensor,before,padded)

def pad_border(~Element: Data,border: Border<Element>,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>,padded: Maybe<&2,List<&2,U32>>) -> Tensors.Tensor<Element>:
  match border padded:
    case Constant{value} Some{padded}: pad_constant(~Element,all_zero(before) && all_zero(after),value,tensor,before,padded)
    case border padded: pad_edge_axes(~Element,before,after,tensor,0)

def padded_count(shape: Maybe<&2,List<&2,U32>>) -> Maybe<&2,U32>:
  match shape:
    case Some{shape}: Tensors.count_of(shape)
    case None{}: None{}

def pad_valid(~Element: Data,+border: Border<Element>,status: Tensors.Status,valid: Bool,tensor: Tensors.Tensor<Element>,shape: List<&2,U32>,
  before: List<&2,U32>,after: List<&2,U32>,padded: Maybe<&2,List<&2,U32>>) -> Tensors.Tensor<Element>:
  match status valid:
    case Tensors.Usable{} True{}: pad_border(~Element,border,tensor,before,after,padded)
    case Tensors.Usable{} False{}: Tensors.with_status(~Element,Tensors.Failed{Tensors.Problem{border_name(~Element,border),Tensors.InvalidArgument{shape}}},tensor)
    case Tensors.Failed{problem} valid: tensor

# before[i] elements are added in front of axis i and after[i] behind it. Both
# lists have one entry per axis and the padded extents must be allocatable.
def pad(~Element: Data,+border: Border<Element>,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>) -> Tensors.Tensor<Element>:
  Tensors.with_metadata(~Element,~Tensors.Tensor<Element>,tensor,tensor => shape => fill => status => workers =>
    +shape = shape
    +padded = Shapes.padded(shape,before,after)
    pad_valid(~Element,border,status,Views.has_count(padded_count(padded)),tensor,shape,before,after,padded))
