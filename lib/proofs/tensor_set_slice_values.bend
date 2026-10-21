# The values of set_slice and of every write through a view of storage.
# The part is stored by one checked scatter through a view of the target: a
# storage index holds the part element of the last coordinates of the view at
# that index, or the target element when the view does not reach it.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_contract.bend as Contract
import ./tensor_copy_values.bend as CopyValues
import ./view_writes.bend as ViewWrites
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./tensor_gather_values.bend as GatherValues
import ./tensor_scatter_values.bend as Scatter
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_full_values.bend as Full
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ./tensor_canonical_view.bend as CanonicalView
import ./tensor_count_completion.bend as CountCompletion
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./checked_word_completion.bend as Completion
import ./u32_full_range.bend as FullRange
import ./u32_division.bend as U32Division
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean
import ../tensor_band.bend as Bands
import ./tensor_band_model.bend as Model
import ./tensor_global_shape_count.bend as Global
import ./tensor_whole_leaf.bend as WholeLeaf
import ../tensor_band_view.bend as BandViews
import ../default_machine_limits.bend as Default
import ../tensor_band_schedule.bend as BandSchedule
import ../traversal_partition.bend as Partition
import ./tensor_band_view_writes.bend as BandWrites
import ./tensor_band_view_values.bend as ViewValues
import ./tensor_band_view_classified.bend as Classified
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_target.bend as Target
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_band_each.bend as Each
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_inspection.bend as Inspection
import ./tensor_row_major_coordinates.bend as Digits
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./consecutive_writes.bend as Consecutive
import ./tensor_line_values.bend as LineValues

# The reader of a write through a view: the scatter of the part through the
# view over the target, with the count of the view; the target itself when
# there is no view.
type Placed<-Before: Data,-Part: Data> is Data:
  Placed{before: Before,part: Part,slice: Maybe<&2,Views.View>}

def count_of(view: Views.View) -> Nat:
  U32.to_nat(Tensors.known_count(Views.numel(view)))

# The element of the part that the view stores at a storage index, or
# previous.
def slice_element(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,slice: Maybe<&2,Views.View>,part: Part,previous: Element,position: U32) -> Element:
  match slice:
    case Some{+view}: Scatter.scattered(~Element,~Part,~part_value,count_of(view),view,part,previous,position)
    case None{}: previous

def placed_value(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,context: Placed<Before,Part>,position: U32) -> Element:
  Placed{before,part,slice} = context
  +position = position
  slice_element(~Element,~Part,~part_value,slice,part,before_value(before,position),position)

# A write through a view that returned Ready: the result has the target shape,
# and its reader is the scatter of the part through the view.
def Placing(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,slice: Maybe<&2,Views.View>,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Contract(~Element,~Placed<Before,Part>,~placed_value(~Element,~Before,~Part,~before_value,~part_value),Placed{before,part,slice},tensor),{Contract.extents(~Element,tensor) == shape : List<&2,U32>}>

def ReadyPlacing(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,slice: Maybe<&2,Views.View>,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Type:
  {Contract.tensor_ready(~Element,tensor) == True{} : Bool} -> Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice,shape,tensor)

# The part as the write passes it on: its bands as a Ready tensor of the
# workers of the write.
def part_tensor(~Element: Data,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}

# The result of the target bands and the part bands of one plan written band
# by band.
def placed_tensor(~Element: Data,+name: String,+banded: BandViews.Banded,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+layout: Bands.Plan,
  target: Bands.Bands(~Element),part: Bands.Bands(~Element)) -> Tensors.Tensor<Element>:
  Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},geometry,shape,fill,workers,
    Bands.transform_paired(~Element,~Layout.Placing,~Layout.placed_band(~Element),layout,BandSchedule.parallel(geometry,workers,12.0,0.0,layout),Layout.Placing{banded,geometry},
      Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,target,part)))

# ---- The counts of a view ----

# A view whose count the machine holds counts the elements of its extents.
def view_count(result: Maybe<&2,U32>,+extents: List<&2,U32>,count: ShapeLaws.Count(result,ShapeLaws.elements(extents)),+word: U32,
  allocated: {Tensors.allocatable(result) == Some{word} : Maybe<&2,U32>}) ->
  {U32.to_nat(Tensors.known_count(result)) == ShapeLaws.elements(extents) : Nat}:
  match result:
    case Some{+value}: Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(value),count)
    case None{}:
      Empty.absurd({U32.to_nat(Tensors.known_count(None{})) == ShapeLaws.elements(extents) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,maybe => Views.has_count(maybe),None{},Some{word},allocated)))

# The axes of a banded view without repeats hold the elements of every
# image, channel block, row and column block.
def unrepeated_elements(+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +batch_positive: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}))) ==
    Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(after))))) : Nat}:
  +images = Classified.images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32))
  +rowed = BandViews.rows_axes(U32.is_lt(1,height),height,width,[])
  +rest = List.append(&2,Views.Axis,rowed,after)
  +middle = List.append(&2,Views.Axis,before,rest)
  +b = U32.to_nat(batch)
  +h = U32.to_nat(height)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +rows = ShapeLaws.elements(Views.axis_extents(rowed))
  +rows_exact = Equal.trans(Nat,rows,Nat.mul(h,1n),h,ViewValues.rows_elements(U32.is_lt(1,height),height,width,[],{==},height_positive),Order.times_one(h))
  Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,images,middle))),
    Nat.mul(ShapeLaws.elements(Views.axis_extents(images)),ShapeLaws.elements(Views.axis_extents(middle))),Nat.mul(b,Nat.mul(eb,Nat.mul(h,ea))),
    LineValues.axes_elements(images,middle),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(images)),b,ViewValues.images_elements(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32),{==},batch_positive)) :
      {Nat.mul(_,ShapeLaws.elements(Views.axis_extents(middle))) == Nat.mul(b,Nat.mul(eb,Nat.mul(h,ea))) : Nat}
    Equal.cong(Nat,Nat,value => Nat.mul(b,value),ShapeLaws.elements(Views.axis_extents(middle)),Nat.mul(eb,Nat.mul(h,ea)),
      Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(middle)),Nat.mul(eb,ShapeLaws.elements(Views.axis_extents(rest))),Nat.mul(eb,Nat.mul(h,ea)),
        LineValues.axes_elements(before,rest),
        Equal.cong(Nat,Nat,value => Nat.mul(eb,value),ShapeLaws.elements(Views.axis_extents(rest)),Nat.mul(h,ea),
          Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(rest)),Nat.mul(rows,ea),Nat.mul(h,ea),LineValues.axes_elements(rowed,after),
            Equal.cong(Nat,Nat,value => Nat.mul(value,ea),rows,h,rows_exact))))))

# The band writes of a part of the extents of a classified view are as many
# as the view counts.
def written_count(+view_axes: List<&2,Views.Axis>,+offset: U32,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+part_shape: List<&2,U32>,
  +axes_exact: {view_axes == Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width}) : List<&2,Views.Axis>},
  +fitted: ViewValues.Fitted<before,[],after,start,batch,channels,height,width>,+same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  counted: Global.Counted<part_shape>) ->
  {Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(after))))) ==
    count_of(Views.View{view_axes,offset}) : Nat}:
  Global.Counted{+count,+accepted} = counted
  +extents = Views.axis_extents(view_axes)
  +whole = Classified.whole_axes(BandViews.Banded{before,[],after,start},Bands.Geometry{batch,channels,height,width})
  +writes = Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(after)))))
  +counts = {%Equal.sym(List<&2,U32>,extents,part_shape,same) : {Tensors.allocatable(Shapes.numel(Shapes.Shape{_})) == Some{count} : Maybe<&2,U32>}
    accepted : {Tensors.allocatable(Shapes.numel(Shapes.Shape{extents})) == Some{count} : Maybe<&2,U32>}}
  +viewed = view_count(Shapes.numel(Shapes.Shape{extents}),extents,ShapeLaws.numel(extents),count,counts)
  +elements = Equal.trans(Nat,ShapeLaws.elements(extents),ShapeLaws.elements(Views.axis_extents(whole)),writes,
    Equal.cong(List<&2,Views.Axis>,Nat,axes => ShapeLaws.elements(Views.axis_extents(axes)),view_axes,whole,axes_exact),
    unrepeated_elements(before,after,start,batch,channels,height,width,ViewValues.fitted_batch(before,[],after,start,batch,channels,height,width,fitted),
      ViewValues.fitted_height(before,[],after,start,batch,channels,height,width,fitted)))
  Equal.sym(Nat,count_of(Views.View{view_axes,offset}),writes,Equal.trans(Nat,count_of(Views.View{view_axes,offset}),ShapeLaws.elements(extents),writes,viewed,elements))

# ---- The values of a write through a view ----

# A tensor of the contract of a reader is a tensor of the placement through a
# view when the two agree at every index.
def placing(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,~Context: Data,~reader: Context -> U32 -> Element,
  +context: Context,+before: Before,+part: Part,+view: Views.View,
  agrees: @+index: U32 -> {reader(context,index) == placed_value(~Element,~Before,~Part,~before_value,~part_value,Placed{before,part,Some{view}},index) : Element},
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{view},shape,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  +placement = {Placed{before,part,Some{view}} : Placed<Before,Part>}
  Shape.Both{Global.relabeled(~Element,~Context,~Placed<Before,Part>,~reader,~placed_value(~Element,~Before,~Part,~before_value,~part_value),
      ~(left => right => Global.Fixed(~Context,~Placed<Before,Part>,context,placement,left,right)),
      ~(left => right => related => index => Global.fixed_agrees(~Element,~Context,~Placed<Before,Part>,~reader,~placed_value(~Element,~Before,~Part,~before_value,~part_value),
        context,placement,agrees,left,right,related,index)),
      context,placement,Shape.Both{{==},{==}},Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},contract),{==}}

# The scatter of as many writes as the view counts reads as the placement.
def scatter_agrees(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+view: Views.View,
  +writes: Nat,+exact: {writes == count_of(view) : Nat},+index: U32) ->
  {Scatter.value(~Element,~Before,~Part,~before_value,~part_value,Scatter.Scatter{before,part,view,writes},index) ==
    placed_value(~Element,~Before,~Part,~before_value,~part_value,Placed{before,part,Some{view}},index) : Element}:
  Equal.cong(Nat,Element,amount => Scatter.scattered(~Element,~Part,~part_value,amount,view,part,before_value(before,index),index),writes,count_of(view),exact)

# A view that counts no element leaves the target as it is.
def kept_agrees(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+view: Views.View,
  +empty: {Views.numel(view) == Some{0} : Maybe<&2,U32>},+index: U32) ->
  {before_value(before,index) == placed_value(~Element,~Before,~Part,~before_value,~part_value,Placed{before,part,Some{view}},index) : Element}:
  %Equal.sym(Maybe<&2,U32>,Views.numel(view),Some{0},empty) :
    {before_value(before,index) == Scatter.scattered(~Element,~Part,~part_value,U32.to_nat(Tensors.known_count(_)),view,part,before_value(before,index),index) : Element}
  {==}

# Bands made in a plan cover the rows of every geometry of the same images and
# height.
def formed_planned(~Element: Data,+geometry: Bands.Geometry,+model: Model.Model<Element>,formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>) ->
  Target.Planned(geometry,Model.layout(~Element,model)):
  BandShape.WellFormed{metadata,ready,coverage} = formed
  Shape.Both{BandShape.spans(~Element,geometry,model,metadata),
    %Equal.sym(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),Model.layout_matches_content(~Element,model)) :
      BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage}

# ---- A write band by band ----

# The target and the arrived part make the scope of the band writes; every
# band writes its part band through its local view.
def placed_bands(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+axes_before: List<&2,Views.Axis>,+axes_after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+target_model: Model.Model<Element>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,+part_model: Model.Model<Element>,
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{axes_before,[],axes_after,start}>,
  +fitted: ViewValues.Fitted<axes_before,[],axes_after,start,batch,channels,height,width>,
  +exact: {Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(axes_before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(axes_after))))) ==
    count_of(Views.View{view_axes,offset}) : Nat},
  well_formed: BandShape.WellFormed<Element,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)>,
  values: BandValues.Values(~Element,~Before,~before_value,before,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)),
  +counted: Global.Counted<target_shape>,+coherent: Global.Coherent(target_shape,Bands.Geometry{batch,channels,height,width}),
  +same_layout: {Model.content_layout(~Element,Model.content(~Element,part_model)) == Model.layout(~Element,target_model) : Bands.Plan},
  part_formed: BandShape.WellFormed<Element,BandViews.output(BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width}),Model.content(~Element,part_model)>,
  part_values: BandValues.Values(~Element,~Part,~part_value,part,BandViews.output(BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width}),
    Model.content(~Element,part_model)),
  accepted: {Contract.tensor_ready(~Element,placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
    Model.layout(~Element,target_model),Model.pack(~Element,target_model),Model.pack(~Element,part_model))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
      Model.layout(~Element,target_model),Model.pack(~Element,target_model),Model.pack(~Element,part_model))):
  BandShape.WellFormed{+metadata,+ready,coverage} = well_formed
  BandShape.WellFormed{+part_metadata,+part_ready,part_coverage} = part_formed
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +view = {Views.View{view_axes,offset} : Views.View}
  +writes = Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(axes_before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(axes_after)))))
  +words = ViewValues.words(axes_before,[],axes_after,start,batch,channels,height,width,target_shape,fitted,counted,coherent)
  +room = ViewValues.fitted_domain(axes_before,[],axes_after,start,batch,channels,height,width,target_shape,fitted,counted,coherent)
  +scope = {BandWrites.Scope{before,part,view_axes,offset,axes_before,axes_after,start,geometry,Default.limits(),words,classified,room} : BandWrites.Scope<Before,Part>}
  +target_facts = Model.content_both(~Element,~BandWrites.Scope<Before,Part>,
    ~Model.Both(~Element,~BandWrites.Scope<Before,Part>,~BandWrites.target_fields(~Element,~Before,~Part),~Model.always(~Element,~BandWrites.Scope<Before,Part>,~Model.ready_band(~Element))),
    ~BandWrites.target_values(~Element,~Before,~Part,~before_value),scope,Model.content(~Element,target_model),
    Model.content_both(~Element,~BandWrites.Scope<Before,Part>,~BandWrites.target_fields(~Element,~Before,~Part),~Model.always(~Element,~BandWrites.Scope<Before,Part>,~Model.ready_band(~Element)),
      scope,Model.content(~Element,target_model),metadata,ready),values)
  +part_facts = Model.content_both(~Element,~BandWrites.Scope<Before,Part>,
    ~Model.Both(~Element,~BandWrites.Scope<Before,Part>,~BandWrites.part_fields(~Element,~Before,~Part),~Model.always(~Element,~BandWrites.Scope<Before,Part>,~Model.ready_band(~Element))),
    ~BandWrites.part_values(~Element,~Before,~Part,~part_value),scope,Model.content(~Element,part_model),
    Model.content_both(~Element,~BandWrites.Scope<Before,Part>,~BandWrites.part_fields(~Element,~Before,~Part),~Model.always(~Element,~BandWrites.Scope<Before,Part>,~Model.ready_band(~Element)),
      scope,Model.content(~Element,part_model),part_metadata,part_ready),part_values)
  +common = Each.matched(~Element,target_model,part_model,Equal.sym(Bands.Plan,Model.layout(~Element,part_model),Model.layout(~Element,target_model),
    Equal.trans(Bands.Plan,Model.layout(~Element,part_model),Model.content_layout(~Element,Model.content(~Element,part_model)),Model.layout(~Element,target_model),
      Model.layout_matches_content(~Element,part_model),same_layout)))
  BandWrites.placed(~Element,~Before,~Part,~before_value,~part_value,~(scope => tensor => Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{view},target_shape,tensor)),
    scope,name,target_shape,fill,workers,target_model,part_model,common,target_facts,part_facts,counted,coherent,
    model => placed => placing(~Element,~Before,~Part,~before_value,~part_value,~Scatter.Scatter<Before,Part>,~Scatter.value(~Element,~Before,~Part,~before_value,~part_value),
      Scatter.Scatter{before,part,view,writes},before,part,view,index => scatter_agrees(~Element,~Before,~Part,~before_value,~part_value,before,part,view,writes,exact,index),
      Model.pack(~Element,model),geometry,target_shape,fill,workers,placed),
    accepted)

# The arrived part is read through its model.
def arrived_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+axes_before: List<&2,Views.Axis>,+axes_after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+target_model: Model.Model<Element>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  -arrived: Bands.Bands(~Element),observation: Model.Observation<Element,arrived>,
  facts: Arrive.Arrived(~Element,~Part,~part_value,part,BandViews.output(BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width}),
    Model.layout(~Element,target_model),arrived),
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{axes_before,[],axes_after,start}>,
  +fitted: ViewValues.Fitted<axes_before,[],axes_after,start,batch,channels,height,width>,
  +exact: {Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(axes_before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(Views.axis_extents(axes_after))))) ==
    count_of(Views.View{view_axes,offset}) : Nat},
  well_formed: BandShape.WellFormed<Element,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)>,
  values: BandValues.Values(~Element,~Before,~before_value,before,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)),
  +counted: Global.Counted<target_shape>,+coherent: Global.Coherent(target_shape,Bands.Geometry{batch,channels,height,width}),
  accepted: {Contract.tensor_ready(~Element,placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
    Model.layout(~Element,target_model),Model.pack(~Element,target_model),arrived)) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
      Model.layout(~Element,target_model),Model.pack(~Element,target_model),arrived)):
  Model.Observation{+model,+equation} = observation
  Shape.Both{+same_layout,Shape.Both{part_formed,part_values}} = facts
  +output = BandViews.output(BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width})
  +contents = Model.observed_contents(~Element,model,arrived,equation)
  %equation : Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
      Model.layout(~Element,target_model),Model.pack(~Element,target_model),_))
  placed_bands(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,axes_before,axes_after,start,batch,channels,height,width,
    target_model,target_shape,fill,workers,model,classified,fitted,exact,well_formed,values,counted,coherent,
    %contents : {Model.content_layout(~Element,_) == Model.layout(~Element,target_model) : Bands.Plan} same_layout,
    %contents : BandShape.WellFormed<Element,output,_> part_formed,
    %contents : BandValues.Values(~Element,~Part,~part_value,part,output,_) part_values,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),arrived,equation) :
      {Contract.tensor_ready(~Element,placed_tensor(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,workers,
        Model.layout(~Element,target_model),Model.pack(~Element,target_model),_)) == True{} : Bool}
      accepted)

# Bands whose images and rows a classified view without repeats keeps in place
# take the part in their own plan.
def banded_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+axes_before: List<&2,Views.Axis>,+axes_after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+target_model: Model.Model<Element>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{axes_before,[],axes_after,start}>,
  +well_formed: BandShape.WellFormed<Element,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)>,
  +values: BandValues.Values(~Element,~Before,~before_value,before,Bands.Geometry{batch,channels,height,width},Model.content(~Element,target_model)),
  +counted: Global.Counted<target_shape>,+coherent: Global.Coherent(target_shape,Bands.Geometry{batch,channels,height,width}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  +same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_planned(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,
    workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),Bands.plan(~Element,Model.pack(~Element,target_model)))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_planned(~Element,name,BandViews.Banded{axes_before,[],axes_after,start},Bands.Geometry{batch,channels,height,width},target_shape,fill,
      workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),Bands.plan(~Element,Model.pack(~Element,target_model)))):
  Classified.Classified{+axes_exact,+offset_exact,+fits} = classified
  Shape.Both{Shape.Both{part_formed,Shape.Both{+part_counted,part_coherent}},part_values} = part_contract
  +classified = {Classified.Classified{axes_exact,offset_exact,fits} :
    Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{axes_before,[],axes_after,start}>}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +banded = {BandViews.Banded{axes_before,[],axes_after,start} : BandViews.Banded}
  +output = BandViews.output(banded,geometry)
  +layout = Model.layout(~Element,target_model)
  +fitted = ViewValues.fitted(axes_before,[],axes_after,start,batch,channels,height,width,fits)
  +whole = Classified.whole_axes(banded,geometry)
  +part_output = Equal.trans(Nat,ShapeLaws.elements(part_shape),ShapeLaws.elements(Views.axis_extents(view_axes)),BandGather.elements(output),
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),part_shape,Views.axis_extents(view_axes),Equal.sym(List<&2,U32>,Views.axis_extents(view_axes),part_shape,same)),
    Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(view_axes)),ShapeLaws.elements(Views.axis_extents(whole)),BandGather.elements(output),
      Equal.cong(List<&2,Views.Axis>,Nat,axes => ShapeLaws.elements(Views.axis_extents(axes)),view_axes,whole,axes_exact),
      ViewValues.whole_elements(axes_before,[],axes_after,start,batch,channels,height,width,fitted)))
  +one = Equal.sym(U32,(height * 1 : U32),height,Arithmetic.multiply_one(height))
  +planned = {%one : Target.Planned(Bands.Geometry{batch,channels,_,width},layout)
    formed_planned(~Element,geometry,target_model,well_formed) : Target.Planned(output,layout)}
  +arrival = {Target.Arrival{part_counted,part_output,planned} : Target.Arrival<part_shape,output,layout>}
  +exact = written_count(view_axes,offset,axes_before,axes_after,start,batch,channels,height,width,part_shape,axes_exact,fitted,same,part_counted)
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,target_model)),(Model.pack(~Element,target_model),layout),Model.plan(~Element,target_model)) :
    Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
      Layout.place_planned(~Element,name,banded,geometry,target_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),_))
  arrived_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,axes_before,axes_after,start,batch,channels,height,width,
    target_model,target_shape,fill,workers,
    Tensors.arrive(~Element,output,layout,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
    Model.observe(~Element,Tensors.arrive(~Element,output,layout,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))),
    Arrive.arrived(~Element,~Part,~part_value,part,part_shape,output,layout,part_geometry,part_fill,Tensors.Usable{},workers,part_model,arrival,
      %Model.packed_contents(~Element,part_model) : BandShape.WellFormed<Element,part_geometry,_> part_formed,part_coherent,
      %Model.packed_contents(~Element,part_model) : BandValues.Values(~Element,~Part,~part_value,part,part_geometry,_) part_values),
    classified,fitted,exact,well_formed,values,counted,coherent,
    %Model.plan(~Element,target_model) :
      {Contract.tensor_ready(~Element,Layout.place_planned(~Element,name,banded,geometry,target_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),_)) ==
        True{} : Bool}
      accepted)

# ---- The routes of a write ----

# The target arrived as one band of one row is read through its model.
def arrived_target(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  -arrived: Bands.Bands(~Element),observation: Model.Observation<Element,arrived>,
  facts: Arrive.Arrived(~Element,~Before,~before_value,before,Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},
    Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)}),arrived),
  +counted: Global.Counted<target_shape>,+coherent: Global.Coherent(target_shape,Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)}),
  classified: Classified.Classified<view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},BandViews.Banded{[],[],view_axes,offset}>,
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  +same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_planned(~Element,name,BandViews.Banded{[],[],view_axes,offset},Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},
    target_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),Bands.plan(~Element,arrived))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_planned(~Element,name,BandViews.Banded{[],[],view_axes,offset},Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},
      target_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),Bands.plan(~Element,arrived))):
  Model.Observation{+model,+equation} = observation
  Shape.Both{same_layout,Shape.Both{well_formed,values}} = facts
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)} : Bands.Geometry}
  +contents = Model.observed_contents(~Element,model,arrived,equation)
  %equation : Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_planned(~Element,name,BandViews.Banded{[],[],view_axes,offset},whole,target_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),
      Bands.plan(~Element,_)))
  banded_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,[],view_axes,offset,1,1,1,Tensors.count_or_zero(target_shape),
    model,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,classified,
    %contents : BandShape.WellFormed<Element,whole,_> well_formed,
    %contents : BandValues.Values(~Element,~Before,~before_value,before,whole,_) values,
    counted,coherent,part_contract,same,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),arrived,equation) :
      {Contract.tensor_ready(~Element,Layout.place_planned(~Element,name,BandViews.Banded{[],[],view_axes,offset},whole,target_shape,fill,workers,
        part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),Bands.plan(~Element,_))) == True{} : Bool}
      accepted)

# A target of one image of one row takes every view its addresses fit.
def whole_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,valid: Bool,+before: Before,+part: Part,
  +name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  classification: Classified.Classification(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},
    BandViews.kept_when(valid,BandViews.Banded{[],[],view_axes,offset})),
  +same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_whole(~Element,name,BandViews.kept_when(valid,BandViews.Banded{[],[],view_axes,offset}),
    Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),
    Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_whole(~Element,name,BandViews.kept_when(valid,BandViews.Banded{[],[],view_axes,offset}),
      Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),
      Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers})):
  match valid:
    case True{}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,coherent}},values} = target_contract
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)} : Bands.Geometry}
      +arrival = WholeLeaf.whole_arrival(target_shape,counted)
      arrived_target(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,
        Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
        Model.observe(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers})),
        Arrive.arrived(~Element,~Before,~before_value,before,target_shape,whole,Bands.unsplit(whole),geometry,fill,Tensors.Usable{},workers,target_model,arrival,
          %Model.packed_contents(~Element,target_model) : BandShape.WellFormed<Element,geometry,_> well_formed,coherent,
          %Model.packed_contents(~Element,target_model) : BandValues.Values(~Element,~Before,~before_value,before,geometry,_) values),
        counted,Target.arrival_coherent(target_shape,whole,Bands.unsplit(whole),arrival),classification,part_contract,same,accepted)
    case False{}:
      Empty.absurd(Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
          Layout.place_whole(~Element,name,None{},Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)},target_shape,part_shape,fill,workers,
            part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers),
            Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers})),
        WordEncoding.false_true(accepted))

# Other targets first arrive as one band of one row.
def target_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,
  +name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  +same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_target(~Element,name,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
    part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_target(~Element,name,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
      part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(target_shape)} : Bands.Geometry}
  whole_slice(~Element,~Before,~Part,~before_value,~part_value,BandViews.fits(BandViews.Banded{[],[],view_axes,offset},whole),before,part,name,view_axes,offset,
    target_model,geometry,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,
    Classified.classify(view_axes,offset,whole),same,accepted)

# A classification without repeats writes band by band; other targets first
# arrive as one band of one row.
def rows_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,repeats: List<&2,U32>,
  +axes_before: List<&2,Views.Axis>,+axes_after: List<&2,Views.Axis>,+start: U32,+before: Before,+part: Part,
  +name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  classified: Classified.Classified<view_axes,offset,geometry,BandViews.Banded{axes_before,repeats,axes_after,start}>,
  +same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_rows(~Element,name,repeats,axes_before,axes_after,start,view_axes,offset,Model.pack(~Element,target_model),geometry,
    target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_rows(~Element,name,repeats,axes_before,axes_after,start,view_axes,offset,Model.pack(~Element,target_model),geometry,
      target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  match repeats:
    case []:
      Bands.Geometry{+batch,+channels,+height,+width} = geometry
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,+coherent}},values} = target_contract
      +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +contents = Model.packed_contents(~Element,target_model)
      banded_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,axes_before,axes_after,start,batch,channels,height,width,
        target_model,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,classified,
        %contents : BandShape.WellFormed<Element,geometry,_> well_formed,
        %contents : BandValues.Values(~Element,~Before,~before_value,before,geometry,_) values,
        counted,coherent,part_contract,same,accepted)
    case extent <> rest:
      target_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,target_model,geometry,target_shape,fill,workers,
        part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,same,accepted)

# The view is classified by the geometry of the target first.
def found_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,found: Maybe<&2,BandViews.Banded>,
  +before: Before,+part: Part,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,
  +fill: Element,+workers: U32,+part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  classification: Classified.Classification(view_axes,offset,geometry,found),+same: {Views.axis_extents(view_axes) == part_shape : List<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.place_found(~Element,name,found,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
    part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_found(~Element,name,found,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
      part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  match found:
    case Some{BandViews.Banded{+axes_before,repeats,+axes_after,+start}}:
      rows_slice(~Element,~Before,~Part,~before_value,~part_value,repeats,axes_before,axes_after,start,before,part,name,view_axes,offset,target_model,geometry,target_shape,fill,workers,
        part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,classification,same,accepted)
    case None{}:
      target_slice(~Element,~Before,~Part,~before_value,~part_value,before,part,name,view_axes,offset,target_model,geometry,target_shape,fill,workers,
        part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,same,accepted)

# A part of the extents of the view is written; a part without elements
# writes nothing, since the view then counts none.
def checked_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,same: Bool,empty: Bool,
  +before: Before,+part: Part,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,
  +fill: Element,+workers: U32,+part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  +same_exact: {Shapes.same_extents(Views.axis_extents(view_axes),part_shape) == same : Bool},+emptiness: {Shapes.has_zero(part_shape) == empty : Bool},
  accepted: {Contract.tensor_ready(~Element,Layout.place_checked(~Element,name,same,empty,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
    part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
    Layout.place_checked(~Element,name,same,empty,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
      part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  match same empty:
    case True{} True{}:
      +view = {Views.View{view_axes,offset} : Views.View}
      +zero = {%Equal.sym(List<&2,U32>,Views.axis_extents(view_axes),part_shape,ShapeLaws.same_extents(Views.axis_extents(view_axes),part_shape,same_exact)) :
          {Shapes.count_nonempty(Shapes.has_zero(_),_) == Some{0} : Maybe<&2,U32>}
        %Equal.sym(Bool,Shapes.has_zero(part_shape),True{},emptiness) : {Shapes.count_nonempty(_,part_shape) == Some{0} : Maybe<&2,U32>}
        {==} : {Views.numel(view) == Some{0} : Maybe<&2,U32>}}
      placing(~Element,~Before,~Part,~before_value,~part_value,~Before,~before_value,before,before,part,view,
        index => kept_agrees(~Element,~Before,~Part,~before_value,~part_value,before,part,view,zero,index),
        Model.pack(~Element,target_model),geometry,target_shape,fill,workers,target_contract)
    case True{} False{}:
      found_slice(~Element,~Before,~Part,~before_value,~part_value,BandViews.classify(view_axes,offset,geometry),before,part,name,view_axes,offset,target_model,geometry,target_shape,
        fill,workers,part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,Classified.classify(view_axes,offset,geometry),
        ShapeLaws.same_extents(Views.axis_extents(view_axes),part_shape,same_exact),accepted)
    case False{} empty:
      Empty.absurd(Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,Some{Views.View{view_axes,offset}},target_shape,
          Layout.place_checked(~Element,name,False{},empty,view_axes,offset,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
            part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))),
        WordEncoding.false_true(accepted))

# A write of a part of any layout through a view of the target bands: when
# the call returns Ready, the result has the target shape and holds the part
# through the view, and the target elsewhere.
def modeled_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,slice: Maybe<&2,Views.View>,
  +before: Before,+part: Part,+name: String,+target_model: Model.Model<Element>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  +part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_model),geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  accepted: {Contract.tensor_ready(~Element,Layout.place_slice(~Element,name,slice,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
    part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice,target_shape,
    Layout.place_slice(~Element,name,slice,Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
      part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  match slice:
    case Some{Views.View{+view_axes,+offset}}:
      checked_slice(~Element,~Before,~Part,~before_value,~part_value,Shapes.same_extents(Views.axis_extents(view_axes),part_shape),Shapes.has_zero(part_shape),
        before,part,name,view_axes,offset,target_model,geometry,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,target_contract,part_contract,{==},{==},accepted)
    case None{}:
      Empty.absurd(Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,None{},target_shape,
          Layout.place_slice(~Element,name,None{},Model.pack(~Element,target_model),geometry,target_shape,part_shape,fill,workers,
            part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))),
        WordEncoding.false_true(accepted))

# The target bands are read through their model.
def observed_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,slice: Maybe<&2,Views.View>,
  +before: Before,+part: Part,+name: String,-bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,+geometry: Bands.Geometry,+target_shape: List<&2,U32>,
  +fill: Element,+workers: U32,+part_model: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{bands,geometry,target_shape,fill,Tensors.Usable{},workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)),
  accepted: {Contract.tensor_ready(~Element,Layout.place_slice(~Element,name,slice,bands,geometry,target_shape,part_shape,fill,workers,
    part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice,target_shape,
    Layout.place_slice(~Element,name,slice,bands,geometry,target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))):
  Model.Observation{+model,+equation} = observation
  %equation : Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice,target_shape,
    Layout.place_slice(~Element,name,slice,_,geometry,target_shape,part_shape,fill,workers,part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers)))
  modeled_slice(~Element,~Before,~Part,~before_value,~part_value,slice,before,part,name,model,geometry,target_shape,fill,workers,part_model,part_geometry,part_shape,part_fill,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{_,geometry,target_shape,fill,Tensors.Usable{},workers})
      target_contract,
    part_contract,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      {Contract.tensor_ready(~Element,Layout.place_slice(~Element,name,slice,_,geometry,target_shape,part_shape,fill,workers,
        part_tensor(~Element,part_model,part_geometry,part_shape,part_fill,workers))) == True{} : Bool}
      accepted)

# ---- The public writes ----

# The reader of a set_slice result: the part through the slice of the target
# extents that it replaces.
def slice_of(+axis: U32,start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  Layout.sliced(Tensors.view_of(target_shape),axis,start,Layout.extent_at(part_shape,U32.to_nat(axis)),1)

def placement(~Before: Data,~Part: Data,before: Before,part: Part,+axis: U32,start: U32,+target_shape: List<&2,U32>,+part_shape: List<&2,U32>) -> Placed<Before,Part>:
  Placed{before,part,slice_of(axis,start,target_shape,part_shape)}

# set_slice of a target and a part of any layouts: when the call returns
# Ready, the result has the target shape and holds the part through the slice
# of the target that it replaces, and the target elsewhere.
def set_slice(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+axis: U32,+start: U32,
  +target_bands: Model.Model<Element>,+target_geometry: Bands.Geometry,+target_shape: List<&2,U32>,+target_fill: Element,+target_workers: U32,
  +part_bands: Model.Model<Element>,+part_geometry: Bands.Geometry,+part_shape: List<&2,U32>,+part_fill: Element,+part_workers: U32,
  target_contract: Global.Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers}),
  part_contract: Global.Contract(~Element,~Part,~part_value,part,Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},axis,start,
    Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers})) == True{} : Bool}) ->
  Placing(~Element,~Before,~Part,~before_value,~part_value,before,part,slice_of(axis,start,target_shape,part_shape),target_shape,
    Layout.set_slice(~Element,Tensors.Tensor{Model.pack(~Element,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},axis,start,
      Tensors.Tensor{Model.pack(~Element,part_bands),part_geometry,part_shape,part_fill,Tensors.Usable{},part_workers})):
  modeled_slice(~Element,~Before,~Part,~before_value,~part_value,slice_of(axis,start,target_shape,part_shape),before,part,"set_slice",target_bands,target_geometry,target_shape,
    target_fill,U32.max(target_workers,part_workers),part_bands,part_geometry,part_shape,part_fill,target_contract,part_contract,accepted)

def pad_counted(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Maybe<&2,U32>,+part: Part,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,
  +part_model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Part,~part_value,part,part_tensor(~Element,part_model,geometry,shape,fill,workers)),
  +counts: {Tensors.count_of(padded) == count : Maybe<&2,U32>}) ->
  ReadyPlacing(~Element,~Element,~Part,~Full.constant(~Element),~part_value,value,part,Layout.interior(before,shape,Tensors.view_of(padded),0),padded,
    Layout.padded_into(~Element,count,value,part_tensor(~Element,part_model,geometry,shape,fill,workers),before,padded)):
  match count:
    case Some{+count}:
      accepted => observed_slice(~Element,~Element,~Part,~Full.constant(~Element),~part_value,Layout.interior(before,shape,Tensors.view_of(padded),0),value,part,"pad",
        Partition.Leaf{Tensors.buffer(~Element,value,count),0,1,Unit{}},Model.observe(~Element,Partition.Leaf{Tensors.buffer(~Element,value,count),0,1,Unit{}}),
        Bands.Geometry{1,1,1,count},padded,fill,workers,part_model,geometry,shape,fill,
        %Global.whole_width(padded,count,counts) :
          Global.Contract(~Element,~Element,~Full.constant(~Element),value,
            Tensors.Tensor{Partition.Leaf{Tensors.buffer(~Element,value,count),0,1,Unit{}},Bands.Geometry{1,1,1,_},padded,fill,Tensors.Usable{},workers})
          WholeLeaf.stored(~Element,~Element,~Full.constant(~Element),value,Tensors.buffer(~Element,value,count),padded,fill,workers,
            Buffer.observe(~Element,Tensors.buffer(~Element,value,count)),Full.initial_counted(~Element,padded,value,count,Count.count_of(padded,count,counts))),
        contract,accepted)
    case None{}:
      accepted => Empty.absurd(Placing(~Element,~Element,~Part,~Full.constant(~Element),~part_value,value,part,Layout.interior(before,shape,Tensors.view_of(padded),0),padded,
        Layout.padded_into(~Element,None{},value,part_tensor(~Element,part_model,geometry,shape,fill,workers),before,padded)),WordEncoding.false_true(accepted))

# Constant padding of an input of any layout: when the call returns Ready, the
# result has the padded extents and holds the input through the interior view
# of those extents, and the value elsewhere.
def pad_constant(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+value: Element,+before: List<&2,U32>,+padded: List<&2,U32>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.pad_constant(~Element,False{},value,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},before,padded)) == True{} : Bool}) ->
  Placing(~Element,~Element,~Logical,~Full.constant(~Element),~original,value,logical,Layout.interior(before,shape,Tensors.view_of(padded),0),padded,
    Layout.pad_constant(~Element,False{},value,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},before,padded)):
  pad_counted(~Element,~Logical,~original,Tensors.count_of(padded),logical,value,before,padded,bands,geometry,shape,fill,workers,contract,{==},accepted)

# ---- A part written along the middle axis of three axes ----

# The row-major view of [outer,extent,inner] with elements: each stride is the
# product of the extents inside it.
def line_stride(+extent: U32,+inner: U32) -> U32:
  U32.mul(U32.mul(1,inner),extent)

def three_axes(+outer: U32,+extent: U32,+inner: U32) -> List<&2,Views.Axis>:
  [Views.Axis{outer,line_stride(extent,inner)},Views.Axis{extent,U32.mul(1,inner)},Views.Axis{inner,1}]

def three_canonical(+outer: U32,+extent: U32,+inner: U32,+count: U32,+nonzero: {U32.is_zero(count) == False{} : Bool}) ->
  {CanonicalView.canonical_view([outer,extent,inner],count) == Views.View{three_axes(outer,extent,inner),0} : Views.View}:
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) :
    {Views.View{CanonicalView.axis_model(List.reverse(&2,U32,[outer,extent,inner]),Bool.pick(U32,_,0,1),[]),0} == Views.View{three_axes(outer,extent,inner),0} : Views.View}
  {==}

def three_view(+outer: U32,+extent: U32,+inner: U32,+count: U32,+exact: {ShapeLaws.elements([outer,extent,inner]) == U32.to_nat(count) : Nat},
  +nonzero: {U32.is_zero(count) == False{} : Bool}) ->
  {Tensors.view_of([outer,extent,inner]) == Some{Views.View{three_axes(outer,extent,inner),0}} : Maybe<&2,Views.View>}:
  Equal.trans(Maybe<&2,Views.View>,Tensors.view_of([outer,extent,inner]),Some{CanonicalView.canonical_view([outer,extent,inner],count)},Some{Views.View{three_axes(outer,extent,inner),0}},
    CanonicalView.view([outer,extent,inner],count,exact),
    Equal.cong(Views.View,Maybe<&2,Views.View>,view => Some{view},CanonicalView.canonical_view([outer,extent,inner],count),Views.View{three_axes(outer,extent,inner),0},
      three_canonical(outer,extent,inner,count,nonzero)))

# The Nat facts of a slice of count elements from start that ends inside an
# axis of size elements.
type Fits<-start: Nat,-count: Nat,-size: Nat> is Data:
  Fits{start_inside: {Nat.is_lt(start,size) == True{} : Bool},start_last: {Nat.is_le(start,Nat.sub(size,1n)) == True{} : Bool},
    size_positive: {Nat.is_le(1n,size) == True{} : Bool},rest: {Nat.is_le(Nat.sub(count,1n),Nat.sub(Nat.sub(size,1n),start)) == True{} : Bool}}

def fits(count: Nat,+start: Nat,size: Nat,positive: {Nat.is_lt(0n,count) == True{} : Bool},room: {Nat.is_le(Nat.add(start,count),size) == True{} : Bool}) ->
  Fits<start,count,size>:
  match count size:
    case 0n size: Empty.absurd(Fits<start,0n,size>,WordEncoding.false_true(positive))
    case 1n+ +more 0n:
      Empty.absurd(Fits<start,1n+more,0n>,WordEncoding.false_true(%Addition.successor_right(start,more) : {Nat.is_le(_,0n) == True{} : Bool} room))
    case 1n+ +more 1n+ +last:
      +inner = {
        %Addition.successor_right(start,more) : {Nat.is_le(_,1n+last) == True{} : Bool}
        room
        : {Nat.is_le(Nat.add(start,more),last) == True{} : Bool}}
      +start_room = Order.transitive(start,Nat.add(start,more),last,Order.le_add(start,more),inner)
      Fits{Order.lt_succ(start,last,start_room),
        %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_le(start,_) == True{} : Bool} start_room,
        Order.zero_le(last),
        %Equal.sym(Nat,Nat.sub(more,0n),more,Order.subtract_zero(more)) : {Nat.is_le(_,Nat.sub(Nat.sub(1n+last,1n),start)) == True{} : Bool}
        %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_le(more,Nat.sub(_,start)) == True{} : Bool}
        Order.cancel_offset_le(start,more,Nat.sub(last,start),
          %Equal.sym(Nat,Nat.add(start,Nat.sub(last,start)),last,Order.restore_shift(start,last,start_room)) : {Nat.is_le(Nat.add(start,more),_) == True{} : Bool}
          inner)}

# The difference of two words in order.
def word_difference(+left: U32,+right: U32,+ordered: {Nat.is_le(U32.to_nat(right),U32.to_nat(left)) == True{} : Bool}) ->
  {U32.to_nat(U32.sub(left,right)) == Nat.sub(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  %Equal.sym(U32,U32.sub(left,right),U32.from_nat(Nat.sub(U32.to_nat(left),U32.to_nat(right))),Arithmetic.native_subtract(left,right,ordered)) :
    {U32.to_nat(_) == Nat.sub(U32.to_nat(left),U32.to_nat(right)) : Nat}
  FullRange.roundtrip(Nat.sub(U32.to_nat(left),U32.to_nat(right)),left,Order.sub_le(U32.to_nat(left),U32.to_nat(right)))

# A slice of count elements from start fits an axis of size elements.
def start_fits(+start: U32,+count: U32,+size: U32,+room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool}) ->
  {U32.is_le(start,size) == True{} : Bool}:
  U32Division.nat_le_native(start,size,Order.transitive(U32.to_nat(start),Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size),Order.le_add(U32.to_nat(start),U32.to_nat(count)),room))

def slice_steps_fit(+start: U32,+count: U32,+size: U32,+classified: {U32.is_zero(count) == False{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool},facts: Fits<U32.to_nat(start),U32.to_nat(count),U32.to_nat(size)>) ->
  {Views.slice_step(False{},start,count,1,size) == True{} : Bool}:
  Fits{start_inside,+start_last,+size_positive,rest} = facts
  +last = word_difference(size,1,size_positive)
  +span = {
    Equal.trans(Nat,U32.to_nat(((size - 1 - start) / 1 : U32)),U32.to_nat((size - 1 - start : U32)),Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start)),
      U32Division.divided_by_one((size - 1 - start : U32)),
      %last : {U32.to_nat((size - 1 - start : U32)) == Nat.sub(_,U32.to_nat(start)) : Nat}
      word_difference((size - 1 : U32),start,
        %Equal.sym(Nat,U32.to_nat((size - 1 : U32)),Nat.sub(U32.to_nat(size),1n),last) : {Nat.is_le(U32.to_nat(start),_) == True{} : Bool} start_last))
    : {U32.to_nat(((size - 1 - start) / 1 : U32)) == Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start)) : Nat}}
  +below = word_difference(count,1,Order.strict_follow(0n,U32.to_nat(count),Arithmetic.word_nonzero_positive(count,classified)))
  %Equal.sym(Bool,U32.is_zero(count),False{},classified) :
    {U32.is_le(start,size) && (_ || (U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)))) == True{} : Bool}
  Boolean.and_true(U32.is_le(start,size),U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)),start_fits(start,count,size,room),
    Boolean.and_true(U32.is_lt(start,size),U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)),U32Division.nat_lt_native(start,size,start_inside),
      U32Division.nat_le_native((count - 1 : U32),((size - 1 - start) / 1 : U32),
        %Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),below) : {Nat.is_le(_,U32.to_nat(((size - 1 - start) / 1 : U32))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(((size - 1 - start) / 1 : U32)),Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start)),span) : {Nat.is_le(Nat.sub(U32.to_nat(count),1n),_) == True{} : Bool}
        rest)))

def slice_steps_when(empty: Bool,+start: U32,+count: U32,+size: U32,+classified: {U32.is_zero(count) == empty : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool}) ->
  {Views.slice_step(False{},start,count,1,size) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(Bool,U32.is_zero(count),True{},classified) :
        {U32.is_le(start,size) && (_ || (U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)))) == True{} : Bool}
      Boolean.and_true(U32.is_le(start,size),True{},start_fits(start,count,size,room),{==})
    case False{}:
      slice_steps_fit(start,count,size,classified,room,fits(U32.to_nat(count),U32.to_nat(start),U32.to_nat(size),Arithmetic.word_nonzero_positive(count,classified),room))

# The cells of extent lines of inner cells.
def cells(+extent: U32,+inner: U32) -> Nat:
  Nat.mul(U32.to_nat(extent),U32.to_nat(inner))

# Where the slice of a part starts: start lines of inner cells in.
def middle_offset(+start: U32,+inner: U32) -> U32:
  (0 + U32.from_nat(Nat.mul(U32.to_nat(start),U32.to_nat(U32.mul(1,inner)))) : U32)

# The slice of a part: every line of the target over the extent of the part.
def middle_axes(+outer: U32,+extent: U32,+part_extent: U32,+inner: U32) -> List<&2,Views.Axis>:
  [Views.Axis{outer,line_stride(extent,inner)},Views.Axis{part_extent,U32.mul(1,inner)},Views.Axis{inner,1}]

def middle_view(+outer: U32,+extent: U32,+part_extent: U32,+inner: U32,+start: U32) -> Views.View:
  Views.View{middle_axes(outer,extent,part_extent,inner),middle_offset(start,inner)}

# The cell offset of line past the start of the slice.
def middle_cell(+start: U32,+extent: U32,+inner: U32,+line: Nat,+offset: Nat) -> Nat:
  Nat.add(Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner))),offset)

# The bounds of one part written along the middle axis: the part ends inside
# the target, whose cells fit the capacity.
type MiddleFit<-outer: U32,-extent: U32,-inner: U32,-count: U32,-start: U32,-part_extent: U32,-capacity: U32> is Data:
  MiddleFit{exact: {Nat.mul(U32.to_nat(outer),cells(extent,inner)) == U32.to_nat(count) : Nat},nonzero: {U32.is_zero(count) == False{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent)) == True{} : Bool},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}

def fit_cells(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {Nat.is_lt(0n,Nat.mul(U32.to_nat(outer),cells(extent,inner))) == True{} : Bool}:
  MiddleFit{+exact,+nonzero,room,capacity_ok,count_room} = fit
  %Equal.sym(Nat,Nat.mul(U32.to_nat(outer),cells(extent,inner)),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
  Arithmetic.word_nonzero_positive(count,nonzero)

# The lines of the target fit the capacity.
def fit_lines(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {Nat.is_le(Nat.mul(U32.to_nat(outer),cells(extent,inner)),U32.to_nat(capacity)) == True{} : Bool}:
  MiddleFit{+exact,nonzero,room,capacity_ok,+count_room} = fit
  %Equal.sym(Nat,Nat.mul(U32.to_nat(outer),cells(extent,inner)),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  count_room

# One line of the target fits the capacity.
def fit_line(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {Nat.is_le(cells(extent,inner),U32.to_nat(capacity)) == True{} : Bool}:
  +line = cells(extent,inner)
  +outer_positive = RowMajor.left_factor_positive(U32.to_nat(outer),line,fit_cells(outer,extent,inner,count,start,part_extent,capacity,fit))
  Order.transitive(line,Nat.mul(U32.to_nat(outer),line),U32.to_nat(capacity),
    %RowMajor.multiply_commutative(line,U32.to_nat(outer)) : {Nat.is_le(line,_) == True{} : Bool}
    RowMajor.product_covers(line,U32.to_nat(outer),outer_positive),
    fit_lines(outer,extent,inner,count,start,part_extent,capacity,fit))

# The inner axis fits the capacity.
def fit_inner(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {Nat.is_le(U32.to_nat(inner),U32.to_nat(capacity)) == True{} : Bool}:
  +wide = U32.to_nat(inner)
  +tall = U32.to_nat(extent)
  +line_positive = DivisionOrder.right_factor_positive(U32.to_nat(outer),cells(extent,inner),fit_cells(outer,extent,inner,count,start,part_extent,capacity,fit))
  Order.transitive(wide,Nat.mul(tall,wide),U32.to_nat(capacity),
    %RowMajor.multiply_commutative(wide,tall) : {Nat.is_le(wide,_) == True{} : Bool}
    RowMajor.product_covers(wide,tall,RowMajor.left_factor_positive(tall,wide,line_positive)),
    fit_line(outer,extent,inner,count,start,part_extent,capacity,fit))

# The slice ends inside a line: its start and its cells fit one line.
def fit_slice(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) ->
  {Nat.is_le(Nat.add(cells(start,inner),cells(part_extent,inner)),cells(extent,inner)) == True{} : Bool}:
  MiddleFit{exact,nonzero,+room,capacity_ok,count_room} = fit
  +wide = U32.to_nat(inner)
  %RowMajor.distrib(U32.to_nat(start),U32.to_nat(part_extent),wide) : {Nat.is_le(_,cells(extent,inner)) == True{} : Bool}
  RowMajor.mul_le(Nat.add(U32.to_nat(start),U32.to_nat(part_extent)),U32.to_nat(extent),wide,room)

def fit_run(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {Nat.is_le(cells(part_extent,inner),cells(extent,inner)) == True{} : Bool}:
  Order.drop_left_le(cells(start,inner),cells(part_extent,inner),cells(extent,inner),fit_slice(outer,extent,inner,count,start,part_extent,capacity,fit))

# The strides of the view count the cells inside them.
def unit_value(+inner: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(capacity)) == True{} : Bool}) -> {U32.to_nat(U32.mul(1,inner)) == U32.to_nat(inner) : Nat}:
  +wide = U32.to_nat(inner)
  Equal.trans(Nat,U32.to_nat(U32.mul(1,inner)),Nat.add(wide,0n),wide,
    Arithmetic.multiply(1,inner,capacity,capacity_ok,
      %Equal.sym(Nat,Nat.add(wide,0n),wide,Addition.zero_right(wide)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} inner_room),
    Addition.zero_right(wide))

def line_value(+extent: U32,+inner: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(capacity)) == True{} : Bool},+line_room: {Nat.is_le(cells(extent,inner),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(line_stride(extent,inner)) == cells(extent,inner) : Nat}:
  +wide = U32.to_nat(inner)
  +tall = U32.to_nat(extent)
  +unit = unit_value(inner,capacity,capacity_ok,inner_room)
  Equal.trans(Nat,U32.to_nat(line_stride(extent,inner)),Nat.mul(U32.to_nat(U32.mul(1,inner)),tall),Nat.mul(tall,wide),
    Arithmetic.multiply(U32.mul(1,inner),extent,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(U32.mul(1,inner)),wide,unit) : {Nat.is_le(Nat.mul(_,tall),U32.to_nat(capacity)) == True{} : Bool}
      %RowMajor.multiply_commutative(tall,wide) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      line_room),
    %Equal.sym(Nat,U32.to_nat(U32.mul(1,inner)),wide,unit) : {Nat.mul(_,tall) == Nat.mul(tall,wide) : Nat}
    RowMajor.multiply_commutative(wide,tall))

def offset_value(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) -> {U32.to_nat(middle_offset(start,inner)) == cells(start,inner) : Nat}:
  MiddleFit{exact,nonzero,room,+capacity_ok,count_room} = fit
  +unit = unit_value(inner,capacity,capacity_ok,fit_inner(outer,extent,inner,count,start,part_extent,capacity,fit))
  +shifted = Nat.mul(U32.to_nat(start),U32.to_nat(U32.mul(1,inner)))
  +start_room = Order.transitive(cells(start,inner),cells(extent,inner),U32.to_nat(capacity),
    Order.transitive(cells(start,inner),Nat.add(cells(start,inner),cells(part_extent,inner)),cells(extent,inner),
      Order.le_add(cells(start,inner),cells(part_extent,inner)),fit_slice(outer,extent,inner,count,start,part_extent,capacity,fit)),
    fit_line(outer,extent,inner,count,start,part_extent,capacity,fit))
  %unit : {U32.to_nat(middle_offset(start,inner)) == Nat.mul(U32.to_nat(start),_) : Nat}
  %Equal.sym(U32,(0 + U32.from_nat(shifted) : U32),U32.from_nat(shifted),Arithmetic.add_zero_left(U32.from_nat(shifted))) : {U32.to_nat(_) == shifted : Nat}
  FullRange.roundtrip(shifted,capacity,
    %Equal.sym(Nat,U32.to_nat(U32.mul(1,inner)),U32.to_nat(inner),unit) : {Nat.is_le(Nat.mul(U32.to_nat(start),_),U32.to_nat(capacity)) == True{} : Bool} start_room)

# The slice set_slice writes a part of extent part_extent through, from start
# along the middle axis of [outer,extent,inner].
def middle_slice(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>) ->
  {slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]) == Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}:
  MiddleFit{+exact,+nonzero,+room,capacity_ok,count_room} = fit
  +step = U32.mul(1,inner)
  +unit = unit_value(inner,capacity,capacity_ok,fit_inner(outer,extent,inner,count,start,part_extent,capacity,fit))
  +offset = U32.from_nat(Nat.mul(U32.to_nat(start),U32.to_nat(step)))
  +head = {Views.Axis{outer,line_stride(extent,inner)} : Views.Axis}
  +tail = {[Views.Axis{inner,1}] : List<&2,Views.Axis>}
  +counted = Equal.trans(Nat,ShapeLaws.elements([outer,extent,inner]),Nat.mul(U32.to_nat(outer),cells(extent,inner)),U32.to_nat(count),
    ShapeLaws.elements_three(outer,extent,inner),exact)
  +start_room = Order.transitive(Nat.mul(U32.to_nat(start),U32.to_nat(step)),cells(extent,inner),U32.to_nat(capacity),
    %Equal.sym(Nat,U32.to_nat(U32.mul(1,inner)),U32.to_nat(inner),unit) : {Nat.is_le(Nat.mul(U32.to_nat(start),_),cells(extent,inner)) == True{} : Bool}
    Order.transitive(cells(start,inner),Nat.add(cells(start,inner),cells(part_extent,inner)),cells(extent,inner),
      Order.le_add(cells(start,inner),cells(part_extent,inner)),fit_slice(outer,extent,inner,count,start,part_extent,capacity,fit)),
    fit_line(outer,extent,inner,count,start,part_extent,capacity,fit))
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of([outer,extent,inner]),Some{Views.View{three_axes(outer,extent,inner),0}},three_view(outer,extent,inner,count,counted,nonzero)) :
    {Layout.sliced(_,1,start,part_extent,1) == Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}
  %Equal.sym(Bool,Views.slice_step(False{},start,part_extent,1,extent),True{},slice_steps_when(U32.is_zero(part_extent),start,part_extent,extent,{==},room)) :
    {Views.prepend_axis(Some{head},Views.sliced_axis(_,part_extent,Shapes.checked_multiply(step,1),Views.add_offset(Shapes.checked_multiply(start,step),0),tail)) ==
      Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(step,1),Some{step},
    Completion.actual(step,1,Machine.word_max(),step,Order.times_one(U32.to_nat(step)),FullRange.maximum(step),{==})) :
    {Views.prepend_axis(Some{head},Views.sliced_axis(True{},part_extent,_,Views.add_offset(Shapes.checked_multiply(start,step),0),tail)) ==
      Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(start,step),Some{offset},
    Completion.actual(start,step,Machine.word_max(),offset,
      Equal.sym(Nat,U32.to_nat(offset),Nat.mul(U32.to_nat(start),U32.to_nat(step)),FullRange.roundtrip(Nat.mul(U32.to_nat(start),U32.to_nat(step)),capacity,start_room)),
      FullRange.maximum(offset),{==})) :
    {Views.prepend_axis(Some{head},Views.sliced_axis(True{},part_extent,Some{step},Views.add_offset(_,0),tail)) ==
      Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}
  %Equal.sym(Bool,U32.is_le(offset,(Machine.word_max() - 0 : U32)),True{},FullRange.maximum(offset)) :
    {Views.prepend_axis(Some{head},Views.sliced_axis(True{},part_extent,Some{step},Views.address_when(_,0,offset),tail)) ==
      Some{middle_view(outer,extent,part_extent,inner,start)} : Maybe<&2,Views.View>}
  {==}

# A part slice scatters the cells of the part.
def middle_count(+outer: U32,+extent: U32,+part_extent: U32,+inner: U32,+start: U32,+part_count: U32,
  +part_exact: {Nat.mul(U32.to_nat(outer),cells(part_extent,inner)) == U32.to_nat(part_count) : Nat}) ->
  {count_of(middle_view(outer,extent,part_extent,inner,start)) == U32.to_nat(part_count) : Nat}:
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{[outer,part_extent,inner]}),Some{part_count},
    CountCompletion.numel([outer,part_extent,inner],part_count,
      Equal.trans(Nat,ShapeLaws.elements([outer,part_extent,inner]),Nat.mul(U32.to_nat(outer),cells(part_extent,inner)),U32.to_nat(part_count),
        ShapeLaws.elements_three(outer,part_extent,inner),part_exact))) :
    {U32.to_nat(Tensors.known_count(_)) == U32.to_nat(part_count) : Nat}
  {==}

# a + b + c in the order c + a + b.
def rotated(+a: Nat,+b: Nat,+c: Nat) -> {Nat.add(Nat.add(a,b),c) == Nat.add(Nat.add(c,a),b) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(c,Nat.add(a,b)),Nat.add(Nat.add(c,a),b),Order.commutative(Nat.add(a,b),c),
    Equal.sym(Nat,Nat.add(Nat.add(c,a),b),Nat.add(c,Nat.add(a,b)),Addition.associative(c,a,b)))

# Element offset of a line of the slice lies at the start of the slice, that
# many lines of the target and offset cells further on.
def middle_address(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,cells(part_extent,inner)) == True{} : Bool}) ->
  {CopyValues.address(middle_view(outer,extent,part_extent,inner,start),Nat.add(Nat.mul(line,cells(part_extent,inner)),offset)) ==
    U32.from_nat(middle_cell(start,extent,inner,line,offset)) : U32}:
  MiddleFit{exact,nonzero,room,+capacity_ok,count_room} = fit
  +wide = U32.to_nat(inner)
  +width = U32.to_nat(part_extent)
  +run = cells(part_extent,inner)
  +length = cells(extent,inner)
  +positive = DivisionOrder.right_factor_positive(width,wide,Order.le_lt_trans(0n,offset,run,Order.zero_le(offset),offset_inside))
  +quotient = Nat.div(offset,wide)
  +remainder = Nat.mod(offset,wide)
  +points = {[Digits.Point{outer,line,line_stride(extent,inner),False{}},Digits.Point{part_extent,quotient,U32.mul(1,inner),False{}},Digits.Point{inner,remainder,1,False{}}] :
    List<&2,Digits.Point>}
  +valid = {Shape.Both{line_inside,Shape.Both{DivisionOrder.quotient_below(offset,width,wide,positive,offset_inside),Shape.Both{DivisionOrder.remainder_small(offset,wide,positive),Unit{}}}} :
    Digits.Valid(points)}
  +inner_room = fit_inner(outer,extent,inner,count,start,part_extent,capacity,fit)
  +line_exact = line_value(extent,inner,capacity,capacity_ok,inner_room,fit_line(outer,extent,inner,count,start,part_extent,capacity,fit))
  +unit = unit_value(inner,capacity,capacity_ok,inner_room)
  +split = DivisionOrder.quotient_exact(offset,wide,positive)
  +joined = Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(line,run),value),Nat.add(Nat.mul(quotient,wide),remainder),offset,Equal.sym(Nat,offset,Nat.add(Nat.mul(quotient,wide),remainder),split))
  +position = {
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(line,width),quotient),wide),remainder),Nat.add(Nat.mul(line,run),Nat.add(Nat.mul(quotient,wide),remainder)),
      Nat.add(Nat.mul(line,run),offset),Digits.digit_join(line,width,quotient,wide,remainder),joined)
    : {Digits.position(points,0n) == Nat.add(Nat.mul(line,run),offset) : Nat}}
  +unit_digit = Equal.trans(Nat,Nat.add(Nat.mul(remainder,1n),0n),Nat.mul(remainder,1n),remainder,Addition.zero_right(Nat.mul(remainder,1n)),Order.times_one(remainder))
  +digits = {
    %Equal.sym(Nat,U32.to_nat(line_stride(extent,inner)),length,line_exact) :
      {Nat.add(Nat.mul(line,_),Nat.add(Nat.mul(quotient,U32.to_nat(U32.mul(1,inner))),Nat.add(Nat.mul(remainder,1n),0n))) == Nat.add(Nat.mul(line,length),offset) : Nat}
    %Equal.sym(Nat,U32.to_nat(U32.mul(1,inner)),wide,unit) :
      {Nat.add(Nat.mul(line,length),Nat.add(Nat.mul(quotient,_),Nat.add(Nat.mul(remainder,1n),0n))) == Nat.add(Nat.mul(line,length),offset) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.mul(remainder,1n),0n),remainder,unit_digit) :
      {Nat.add(Nat.mul(line,length),Nat.add(Nat.mul(quotient,wide),_)) == Nat.add(Nat.mul(line,length),offset) : Nat}
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(line,length),value),Nat.add(Nat.mul(quotient,wide),remainder),offset,Equal.sym(Nat,offset,Nat.add(Nat.mul(quotient,wide),remainder),split))
    : {Digits.weight(points) == Nat.add(Nat.mul(line,length),offset) : Nat}}
  +total = {
    %Equal.sym(Nat,U32.to_nat(middle_offset(start,inner)),cells(start,inner),offset_value(outer,extent,inner,count,start,part_extent,capacity,fit)) :
      {Nat.add(Digits.weight(points),_) == middle_cell(start,extent,inner,line,offset) : Nat}
    %Equal.sym(Nat,Digits.weight(points),Nat.add(Nat.mul(line,length),offset),digits) :
      {Nat.add(_,cells(start,inner)) == middle_cell(start,extent,inner,line,offset) : Nat}
    rotated(Nat.mul(line,length),offset,cells(start,inner))
    : {Nat.add(Digits.weight(points),U32.to_nat(middle_offset(start,inner))) == middle_cell(start,extent,inner,line,offset) : Nat}}
  %position : {CopyValues.address(middle_view(outer,extent,part_extent,inner,start),_) == U32.from_nat(middle_cell(start,extent,inner,line,offset)) : U32}
  Equal.trans(U32,CopyValues.address(Views.View{Digits.axes(points),middle_offset(start,inner)},Digits.position(points,0n)),
    U32.from_nat(Nat.add(Digits.weight(points),U32.to_nat(middle_offset(start,inner)))),U32.from_nat(middle_cell(start,extent,inner,line,offset)),
    Digits.weighed(points,middle_offset(start,inner),valid),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(Digits.weight(points),U32.to_nat(middle_offset(start,inner))),middle_cell(start,extent,inner,line,offset),total))

# The base of a line of the slice: its cells lie at the base plus their
# position in the part.
def line_base(+start: U32,+extent: U32,+part_extent: U32,+inner: U32,+line: Nat) -> Nat:
  Nat.sub(Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner))),Nat.mul(line,cells(part_extent,inner)))

def scaled_le(+a: Nat,+b: Nat,+factor: Nat,ordered: {Nat.is_le(a,b) == True{} : Bool}) -> {Nat.is_le(Nat.mul(factor,a),Nat.mul(factor,b)) == True{} : Bool}:
  %RowMajor.multiply_commutative(a,factor) : {Nat.is_le(_,Nat.mul(factor,b)) == True{} : Bool}
  %RowMajor.multiply_commutative(b,factor) : {Nat.is_le(Nat.mul(a,factor),_) == True{} : Bool}
  RowMajor.mul_le(a,b,factor,ordered)

# A line of the slice starts at its base plus its position in the part.
def line_start(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat) ->
  {Nat.add(line_base(start,extent,part_extent,inner,line),Nat.mul(line,cells(part_extent,inner))) == Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner))) : Nat}:
  +first = Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner)))
  +skipped = Nat.mul(line,cells(part_extent,inner))
  +base = line_base(start,extent,part_extent,inner,line)
  +run_room = Order.drop_left_le(cells(start,inner),cells(part_extent,inner),cells(extent,inner),fit_slice(outer,extent,inner,count,start,part_extent,capacity,fit))
  +below = Order.transitive(skipped,Nat.mul(line,cells(extent,inner)),first,scaled_le(cells(part_extent,inner),cells(extent,inner),line,run_room),
    Order.drop_left_le(cells(start,inner),Nat.mul(line,cells(extent,inner)),first,Order.reflexive(first)))
  Equal.trans(Nat,Nat.add(base,skipped),Nat.add(skipped,base),first,Order.commutative(base,skipped),Order.restore_shift(skipped,first,below))

def line_cell(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+offset: Nat) ->
  {Nat.add(line_base(start,extent,part_extent,inner,line),Nat.add(Nat.mul(line,cells(part_extent,inner)),offset)) == middle_cell(start,extent,inner,line,offset) : Nat}:
  +skipped = Nat.mul(line,cells(part_extent,inner))
  +base = line_base(start,extent,part_extent,inner,line)
  Equal.trans(Nat,Nat.add(base,Nat.add(skipped,offset)),Nat.add(Nat.add(base,skipped),offset),middle_cell(start,extent,inner,line,offset),
    Equal.sym(Nat,Nat.add(Nat.add(base,skipped),offset),Nat.add(base,Nat.add(skipped,offset)),Addition.associative(base,skipped,offset)),
    Equal.cong(Nat,Nat,value => Nat.add(value,offset),Nat.add(base,skipped),Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner))),
      line_start(outer,extent,inner,count,start,part_extent,capacity,fit,line)))

# The cells of a line of the slice end before the next line of the target.
def line_past(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat) ->
  {Nat.is_le(middle_cell(start,extent,inner,line,cells(part_extent,inner)),Nat.mul(1n+line,cells(extent,inner))) == True{} : Bool}:
  +opening = cells(start,inner)
  +run = cells(part_extent,inner)
  +length = cells(extent,inner)
  +lines = Nat.mul(line,length)
  +reordered = Equal.trans(Nat,Nat.add(Nat.add(opening,lines),run),Nat.add(opening,Nat.add(lines,run)),Nat.add(Nat.add(opening,run),lines),Addition.associative(opening,lines,run),
    Equal.trans(Nat,Nat.add(opening,Nat.add(lines,run)),Nat.add(opening,Nat.add(run,lines)),Nat.add(Nat.add(opening,run),lines),
      Equal.cong(Nat,Nat,value => Nat.add(opening,value),Nat.add(lines,run),Nat.add(run,lines),Order.commutative(lines,run)),
      Equal.sym(Nat,Nat.add(Nat.add(opening,run),lines),Nat.add(opening,Nat.add(run,lines)),Addition.associative(opening,run,lines))))
  %Equal.sym(Nat,Nat.add(Nat.add(opening,lines),run),Nat.add(Nat.add(opening,run),lines),reordered) : {Nat.is_le(_,Nat.mul(1n+line,length)) == True{} : Bool}
  Order.add_le(Nat.add(opening,run),length,lines,fit_slice(outer,extent,inner,count,start,part_extent,capacity,fit))

# The cells of the line after a line of the target lie inside the capacity.
def next_line(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+line,cells(extent,inner)),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(Nat.mul(1n+line,cells(extent,inner)),Nat.mul(U32.to_nat(outer),cells(extent,inner)),U32.to_nat(capacity),
    RowMajor.mul_le(1n+line,U32.to_nat(outer),cells(extent,inner),Order.strict_follow(line,U32.to_nat(outer),line_inside)),
    fit_lines(outer,extent,inner,count,start,part_extent,capacity,fit))

# The last cell of a line of the slice lies inside the capacity.
def line_end(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool}) ->
  {Nat.is_le(middle_cell(start,extent,inner,line,cells(part_extent,inner)),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(middle_cell(start,extent,inner,line,cells(part_extent,inner)),Nat.mul(1n+line,cells(extent,inner)),U32.to_nat(capacity),
    line_past(outer,extent,inner,count,start,part_extent,capacity,fit,line),next_line(outer,extent,inner,count,start,part_extent,capacity,fit,line,line_inside))

# A later line starts past the next one.
def later_line(+line: Nat,+more: Nat,+length: Nat) -> {Nat.is_le(Nat.mul(1n+line,length),Nat.mul(Nat.add(line,1n+more),length)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(line,1n+more),1n+Nat.add(line,more),Addition.successor_right(line,more)) : {Nat.is_le(Nat.mul(1n+line,length),Nat.mul(_,length)) == True{} : Bool}
  RowMajor.mul_le(1n+line,1n+Nat.add(line,more),length,Order.le_add(line,more))

# The cells of a line of the slice from offset on lie at the base of the line
# plus their position.
def slice_line(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,remaining: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+room: {Nat.is_le(Nat.add(offset,remaining),cells(part_extent,inner)) == True{} : Bool}) ->
  CopyValues.Line(middle_view(outer,extent,part_extent,inner,start),line_base(start,extent,part_extent,inner,line),remaining,Nat.add(Nat.mul(line,cells(part_extent,inner)),offset)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +skipped = Nat.mul(line,cells(part_extent,inner))
      +base = line_base(start,extent,part_extent,inner,line)
      +cell = middle_cell(start,extent,inner,line,offset)
      Shape.Both{
        Equal.trans(U32,CopyValues.address(middle_view(outer,extent,part_extent,inner,start),Nat.add(skipped,offset)),U32.from_nat(cell),U32.from_nat(Nat.add(base,Nat.add(skipped,offset))),
          middle_address(outer,extent,inner,count,start,part_extent,capacity,fit,line,offset,line_inside,Order.room_index(offset,rest,cells(part_extent,inner),room)),
          Equal.cong(Nat,U32,U32.from_nat,cell,Nat.add(base,Nat.add(skipped,offset)),
            Equal.sym(Nat,Nat.add(base,Nat.add(skipped,offset)),cell,line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,offset)))),
        %Addition.successor_right(skipped,offset) : CopyValues.Line(middle_view(outer,extent,part_extent,inner,start),base,rest,_)
        slice_line(outer,extent,inner,count,start,part_extent,capacity,fit,line,rest,1n+offset,line_inside,Order.advance_room(offset,rest,cells(part_extent,inner),room))}

# The writes of a line of the slice are consecutive from its base.
def line_writes(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+part: Part,+line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool}) ->
  {ViewWrites.writes_from(~Element,~Part,~value,cells(part_extent,inner),Nat.mul(line,cells(part_extent,inner)),middle_view(outer,extent,part_extent,inner,start),part) ==
    Consecutive.writes(~Element,~Part,~value,cells(part_extent,inner),line_base(start,extent,part_extent,inner,line),Nat.mul(line,cells(part_extent,inner)),part) :
    List<&2,Writes.Assignment<Element>>}:
  +run = cells(part_extent,inner)
  +base = line_base(start,extent,part_extent,inner,line)
  ViewWrites.on_line(~Element,~Part,~value,run,Nat.mul(line,run),middle_view(outer,extent,part_extent,inner,start),part,base,
    %Addition.zero_right(Nat.mul(line,run)) : CopyValues.Line(middle_view(outer,extent,part_extent,inner,start),base,run,_)
    slice_line(outer,extent,inner,count,start,part_extent,capacity,fit,line,run,0n,line_inside,Order.reflexive(run)))

# The writes of the lines from line on: the first line, then the others.
def lines_step(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+run: Nat,+rest: Nat,+line: Nat,+view: Views.View,+part: Part,
  +query: U32,+previous: Element) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),query,previous) ==
    Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part),query,previous)) : Element}:
  +position = Nat.mul(line,run)
  +first = ViewWrites.writes_from(~Element,~Part,~value,run,position,view,part)
  +later = ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.add(position,run),view,part)
  %Order.commutative(position,run) :
    {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),position,view,part),query,previous) ==
      Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),_,view,part),query,Readback.selected(~Element,first,query,previous)) : Element}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,ViewWrites.writes_from(~Element,~Part,~value,Nat.add(run,Nat.mul(rest,run)),position,view,part),
    List.append(&2,Writes.Assignment<Element>,first,later),ViewWrites.split(~Element,~Part,~value,run,Nat.mul(rest,run),position,view,part)) :
    {Readback.selected(~Element,_,query,previous) == Readback.selected(~Element,later,query,Readback.selected(~Element,first,query,previous)) : Element}
  Readback.selected_append(~Element,first,later,query,previous)

# Lines from line on whose cells all lie past the query keep the previous
# element there.
def lines_before(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part: Part,+previous: Element,lines: Nat,+line: Nat,+query: Nat,
  +inside: {Nat.is_le(Nat.add(line,lines),U32.to_nat(outer)) == True{} : Bool},
  +below: {Nat.is_lt(query,Nat.add(cells(start,inner),Nat.mul(line,cells(extent,inner)))) == True{} : Bool}) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(lines,cells(part_extent,inner)),Nat.mul(line,cells(part_extent,inner)),
    middle_view(outer,extent,part_extent,inner,start),part),U32.from_nat(query),previous) == previous : Element}:
  match lines:
    case 0n: {==}
    case 1n+ +rest:
      +run = cells(part_extent,inner)
      +length = cells(extent,inner)
      +view = middle_view(outer,extent,part_extent,inner,start)
      +base = line_base(start,extent,part_extent,inner,line)
      +line_inside = Order.room_index(line,rest,U32.to_nat(outer),inside)
      +first = ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part)
      +kept = {
        %Equal.sym(List<&2,Writes.Assignment<Element>>,first,Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),
          line_writes(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,line,part,line_inside)) :
          {Readback.selected(~Element,_,U32.from_nat(query),previous) == previous : Element}
        Consecutive.before(~Element,~Part,~value,run,base,Nat.mul(line,run),part,previous,query,capacity,
          %Equal.sym(Nat,Nat.add(base,Nat.add(Nat.mul(line,run),run)),middle_cell(start,extent,inner,line,run),line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,run)) :
            {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          line_end(outer,extent,inner,count,start,part_extent,capacity,fit,line,line_inside),
          %Equal.sym(Nat,Nat.add(base,Nat.mul(line,run)),Nat.add(cells(start,inner),Nat.mul(line,length)),
            line_start(outer,extent,inner,count,start,part_extent,capacity,fit,line)) : {Nat.is_lt(query,_) == True{} : Bool}
          below)
        : {Readback.selected(~Element,first,U32.from_nat(query),previous) == previous : Element}}
      %Equal.sym(Element,Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),U32.from_nat(query),previous),
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),U32.from_nat(query),
          Readback.selected(~Element,first,U32.from_nat(query),previous)),
        lines_step(~Element,~Part,~value,run,rest,line,view,part,U32.from_nat(query),previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,first,U32.from_nat(query),previous),previous,kept) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),U32.from_nat(query),_) == previous : Element}
      lines_before(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,rest,1n+line,query,
        Order.advance_room(line,rest,U32.to_nat(outer),inside),
        Bounds.nat_trans(query,Nat.add(cells(start,inner),Nat.mul(line,length)),Nat.add(cells(start,inner),Nat.add(length,Nat.mul(line,length))),below,
          Order.offset_le(cells(start,inner),Nat.mul(line,length),Nat.add(length,Nat.mul(line,length)),
            Order.drop_left_le(length,Nat.mul(line,length),Nat.add(length,Nat.mul(line,length)),Order.reflexive(Nat.add(length,Nat.mul(line,length)))))))

# A cell of a line of the slice lies before the cells of the next line.
def line_below(+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+line: Nat,+offset: Nat,
  +offset_inside: {Nat.is_lt(offset,cells(part_extent,inner)) == True{} : Bool}) ->
  {Nat.is_lt(middle_cell(start,extent,inner,line,offset),Nat.add(cells(start,inner),Nat.mul(1n+line,cells(extent,inner)))) == True{} : Bool}:
  +opening = cells(start,inner)
  +length = cells(extent,inner)
  +lines = Nat.mul(line,length)
  %Equal.trans(Nat,Nat.add(Nat.add(opening,lines),length),Nat.add(opening,Nat.add(lines,length)),Nat.add(opening,Nat.add(length,lines)),Addition.associative(opening,lines,length),
    Equal.cong(Nat,Nat,value => Nat.add(opening,value),Nat.add(lines,length),Nat.add(length,lines),Order.commutative(lines,length))) :
    {Nat.is_lt(middle_cell(start,extent,inner,line,offset),_) == True{} : Bool}
  Order.offset_lt(Nat.add(opening,lines),offset,length,
    Bounds.nat_trans(offset,cells(part_extent,inner),length,offset_inside,fit_run(outer,extent,inner,count,start,part_extent,capacity,fit)))

# A line before the line of the query keeps the previous element there.
def line_passed(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part: Part,+previous: Element,+line: Nat,+query: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},
  +past: {Nat.is_le(Nat.mul(1n+line,cells(extent,inner)),query) == True{} : Bool},+query_inside: {Nat.is_lt(query,U32.to_nat(capacity)) == True{} : Bool}) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,cells(part_extent,inner),Nat.mul(line,cells(part_extent,inner)),middle_view(outer,extent,part_extent,inner,start),part),
    U32.from_nat(query),previous) == previous : Element}:
  +run = cells(part_extent,inner)
  +base = line_base(start,extent,part_extent,inner,line)
  %Equal.sym(List<&2,Writes.Assignment<Element>>,ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),middle_view(outer,extent,part_extent,inner,start),part),
    Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),line_writes(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,line,part,line_inside)) :
    {Readback.selected(~Element,_,U32.from_nat(query),previous) == previous : Element}
  Consecutive.after(~Element,~Part,~value,run,base,Nat.mul(line,run),part,previous,query,capacity,
    %Equal.sym(Nat,Nat.add(base,Nat.add(Nat.mul(line,run),run)),middle_cell(start,extent,inner,line,run),line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,run)) :
      {Nat.is_le(_,query) == True{} : Bool}
    Order.transitive(middle_cell(start,extent,inner,line,run),Nat.mul(1n+line,cells(extent,inner)),query,line_past(outer,extent,inner,count,start,part_extent,capacity,fit,line),past),
    query_inside)

# The writes of the lines from line on store, at the cell offset of the line
# distance further on, element offset of that line of the part.
def lines_reach(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part: Part,+previous: Element,lines: Nat,+line: Nat,distance: Nat,+offset: Nat,
  +inside: {Nat.is_le(Nat.add(line,lines),U32.to_nat(outer)) == True{} : Bool},+reached: {Nat.is_lt(distance,lines) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,cells(part_extent,inner)) == True{} : Bool}) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(lines,cells(part_extent,inner)),Nat.mul(line,cells(part_extent,inner)),
    middle_view(outer,extent,part_extent,inner,start),part),U32.from_nat(middle_cell(start,extent,inner,Nat.add(line,distance),offset)),previous) ==
    value(part,Nat.add(Nat.mul(Nat.add(line,distance),cells(part_extent,inner)),offset)) : Element}:
  match lines distance:
    case 0n distance:
      Empty.absurd({Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(0n,cells(part_extent,inner)),Nat.mul(line,cells(part_extent,inner)),
        middle_view(outer,extent,part_extent,inner,start),part),U32.from_nat(middle_cell(start,extent,inner,Nat.add(line,distance),offset)),previous) ==
        value(part,Nat.add(Nat.mul(Nat.add(line,distance),cells(part_extent,inner)),offset)) : Element},Order.below_zero(distance,reached))
    case 1n+ +rest 0n:
      +run = cells(part_extent,inner)
      +view = middle_view(outer,extent,part_extent,inner,start)
      +base = line_base(start,extent,part_extent,inner,line)
      +line_inside = Order.room_index(line,rest,U32.to_nat(outer),inside)
      +first = ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part)
      +query = U32.from_nat(middle_cell(start,extent,inner,line,offset))
      +stored = value(part,Nat.add(Nat.mul(line,run),offset))
      +hit = {
        %Equal.sym(List<&2,Writes.Assignment<Element>>,first,Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),
          line_writes(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,line,part,line_inside)) :
          {Readback.selected(~Element,_,query,previous) == stored : Element}
        %line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,offset) :
          {Readback.selected(~Element,Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),U32.from_nat(_),previous) == stored : Element}
        Consecutive.hit(~Element,~Part,~value,run,base,Nat.mul(line,run),part,previous,offset,capacity,
          %Equal.sym(Nat,Nat.add(base,Nat.add(Nat.mul(line,run),run)),middle_cell(start,extent,inner,line,run),line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,run)) :
            {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          line_end(outer,extent,inner,count,start,part_extent,capacity,fit,line,line_inside),
          offset_inside)
        : {Readback.selected(~Element,first,query,previous) == stored : Element}}
      %Equal.sym(Nat,Nat.add(line,0n),line,Addition.zero_right(line)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),U32.from_nat(middle_cell(start,extent,inner,_,offset)),previous) ==
          value(part,Nat.add(Nat.mul(_,run),offset)) : Element}
      %Equal.sym(Element,Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),query,previous),
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,Readback.selected(~Element,first,query,previous)),
        lines_step(~Element,~Part,~value,run,rest,line,view,part,query,previous)) : {_ == stored : Element}
      %Equal.sym(Element,Readback.selected(~Element,first,query,previous),stored,hit) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,_) == stored : Element}
      lines_before(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,stored,rest,1n+line,middle_cell(start,extent,inner,line,offset),
        Order.advance_room(line,rest,U32.to_nat(outer),inside),line_below(outer,extent,inner,count,start,part_extent,capacity,fit,line,offset,offset_inside))
    case 1n+ +rest 1n+ +more:
      +run = cells(part_extent,inner)
      +length = cells(extent,inner)
      +view = middle_view(outer,extent,part_extent,inner,start)
      +target = Nat.add(line,1n+more)
      +line_inside = Order.room_index(line,rest,U32.to_nat(outer),inside)
      +target_inside = Bounds.nat_trans(target,Nat.add(line,1n+rest),U32.to_nat(outer),Order.offset_lt(line,1n+more,1n+rest,reached),inside)
      +first = ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part)
      +cell = middle_cell(start,extent,inner,target,offset)
      +query = U32.from_nat(cell)
      +stored = value(part,Nat.add(Nat.mul(target,run),offset))
      +opened = Nat.add(cells(start,inner),Nat.mul(target,length))
      +passed = line_passed(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,line,cell,line_inside,
        Order.transitive(Nat.mul(1n+line,length),Nat.mul(target,length),cell,later_line(line,more,length),
          Order.transitive(Nat.mul(target,length),opened,cell,Order.drop_left_le(cells(start,inner),Nat.mul(target,length),opened,Order.reflexive(opened)),Order.le_add(opened,offset))),
        Bounds.nat_trans(cell,middle_cell(start,extent,inner,target,run),U32.to_nat(capacity),Order.offset_lt(opened,offset,run,offset_inside),
          line_end(outer,extent,inner,count,start,part_extent,capacity,fit,target,target_inside)))
      %Equal.sym(Element,Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),query,previous),
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,Readback.selected(~Element,first,query,previous)),
        lines_step(~Element,~Part,~value,run,rest,line,view,part,query,previous)) : {_ == stored : Element}
      %Equal.sym(Element,Readback.selected(~Element,first,query,previous),previous,passed) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,_) == stored : Element}
      %Equal.sym(Nat,Nat.add(line,1n+more),1n+Nat.add(line,more),Addition.successor_right(line,more)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),U32.from_nat(middle_cell(start,extent,inner,_,offset)),previous) ==
          value(part,Nat.add(Nat.mul(_,run),offset)) : Element}
      lines_reach(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,rest,1n+line,more,offset,
        Order.advance_room(line,rest,U32.to_nat(outer),inside),reached,offset_inside)

# The line of the query keeps the previous element at a cell outside the
# slice.
def line_missed(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part: Part,+previous: Element,+line: Nat,+cell: Nat,
  outside: Either<&2,&2,{Nat.is_lt(cell,cells(start,inner)) == True{} : Bool},{Nat.is_le(Nat.add(cells(start,inner),cells(part_extent,inner)),cell) == True{} : Bool}>,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+cell_inside: {Nat.is_lt(cell,cells(extent,inner)) == True{} : Bool}) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,cells(part_extent,inner),Nat.mul(line,cells(part_extent,inner)),middle_view(outer,extent,part_extent,inner,start),part),
    U32.from_nat(Nat.add(Nat.mul(line,cells(extent,inner)),cell)),previous) == previous : Element}:
  match outside:
    case Inl{before}:
      +run = cells(part_extent,inner)
      +lines = Nat.mul(line,cells(extent,inner))
      +base = line_base(start,extent,part_extent,inner,line)
      %Equal.sym(List<&2,Writes.Assignment<Element>>,ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),middle_view(outer,extent,part_extent,inner,start),part),
        Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),line_writes(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,line,part,line_inside)) :
        {Readback.selected(~Element,_,U32.from_nat(Nat.add(lines,cell)),previous) == previous : Element}
      Consecutive.before(~Element,~Part,~value,run,base,Nat.mul(line,run),part,previous,Nat.add(lines,cell),capacity,
        %Equal.sym(Nat,Nat.add(base,Nat.add(Nat.mul(line,run),run)),middle_cell(start,extent,inner,line,run),line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,run)) :
          {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        line_end(outer,extent,inner,count,start,part_extent,capacity,fit,line,line_inside),
        %Equal.sym(Nat,Nat.add(base,Nat.mul(line,run)),Nat.add(cells(start,inner),lines),line_start(outer,extent,inner,count,start,part_extent,capacity,fit,line)) :
          {Nat.is_lt(Nat.add(lines,cell),_) == True{} : Bool}
        %Order.commutative(lines,cells(start,inner)) : {Nat.is_lt(Nat.add(lines,cell),_) == True{} : Bool}
        Order.offset_lt(lines,cell,cells(start,inner),before))
    case Inr{after}:
      +run = cells(part_extent,inner)
      +lines = Nat.mul(line,cells(extent,inner))
      +base = line_base(start,extent,part_extent,inner,line)
      +ended = Equal.trans(Nat,middle_cell(start,extent,inner,line,run),Nat.add(Nat.add(lines,cells(start,inner)),run),Nat.add(lines,Nat.add(cells(start,inner),run)),
        Equal.cong(Nat,Nat,value => Nat.add(value,run),Nat.add(cells(start,inner),lines),Nat.add(lines,cells(start,inner)),Order.commutative(cells(start,inner),lines)),
        Addition.associative(lines,cells(start,inner),run))
      %Equal.sym(List<&2,Writes.Assignment<Element>>,ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),middle_view(outer,extent,part_extent,inner,start),part),
        Consecutive.writes(~Element,~Part,~value,run,base,Nat.mul(line,run),part),line_writes(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,line,part,line_inside)) :
        {Readback.selected(~Element,_,U32.from_nat(Nat.add(lines,cell)),previous) == previous : Element}
      Consecutive.after(~Element,~Part,~value,run,base,Nat.mul(line,run),part,previous,Nat.add(lines,cell),capacity,
        %Equal.sym(Nat,Nat.add(base,Nat.add(Nat.mul(line,run),run)),middle_cell(start,extent,inner,line,run),line_cell(outer,extent,inner,count,start,part_extent,capacity,fit,line,run)) :
          {Nat.is_le(_,Nat.add(lines,cell)) == True{} : Bool}
        %Equal.sym(Nat,middle_cell(start,extent,inner,line,run),Nat.add(lines,Nat.add(cells(start,inner),run)),ended) : {Nat.is_le(_,Nat.add(lines,cell)) == True{} : Bool}
        Order.offset_le(lines,Nat.add(cells(start,inner),run),cell,after),
        Bounds.nat_trans(Nat.add(lines,cell),Nat.mul(1n+line,cells(extent,inner)),U32.to_nat(capacity),
          %Order.commutative(lines,cells(extent,inner)) : {Nat.is_lt(Nat.add(lines,cell),_) == True{} : Bool}
          Order.offset_lt(lines,cell,cells(extent,inner),cell_inside),
          next_line(outer,extent,inner,count,start,part_extent,capacity,fit,line,line_inside)))

# The writes of the lines from line on keep the previous element at a cell of
# the line distance further on that lies outside the slice.
def lines_pass(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,
  +outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part: Part,+previous: Element,lines: Nat,+line: Nat,distance: Nat,+cell: Nat,
  +outside: Either<&2,&2,{Nat.is_lt(cell,cells(start,inner)) == True{} : Bool},{Nat.is_le(Nat.add(cells(start,inner),cells(part_extent,inner)),cell) == True{} : Bool}>,
  +inside: {Nat.is_le(Nat.add(line,lines),U32.to_nat(outer)) == True{} : Bool},+reached: {Nat.is_lt(distance,lines) == True{} : Bool},
  +cell_inside: {Nat.is_lt(cell,cells(extent,inner)) == True{} : Bool}) ->
  {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(lines,cells(part_extent,inner)),Nat.mul(line,cells(part_extent,inner)),
    middle_view(outer,extent,part_extent,inner,start),part),U32.from_nat(Nat.add(Nat.mul(Nat.add(line,distance),cells(extent,inner)),cell)),previous) == previous : Element}:
  match lines distance:
    case 0n distance:
      Empty.absurd({Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(0n,cells(part_extent,inner)),Nat.mul(line,cells(part_extent,inner)),
        middle_view(outer,extent,part_extent,inner,start),part),U32.from_nat(Nat.add(Nat.mul(Nat.add(line,distance),cells(extent,inner)),cell)),previous) == previous : Element},
        Order.below_zero(distance,reached))
    case 1n+ +rest 0n:
      +run = cells(part_extent,inner)
      +length = cells(extent,inner)
      +view = middle_view(outer,extent,part_extent,inner,start)
      +line_inside = Order.room_index(line,rest,U32.to_nat(outer),inside)
      +first = ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part)
      +position = Nat.add(Nat.mul(line,length),cell)
      +query = U32.from_nat(position)
      +kept = line_missed(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,line,cell,outside,line_inside,cell_inside)
      %Equal.sym(Nat,Nat.add(line,0n),line,Addition.zero_right(line)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),U32.from_nat(Nat.add(Nat.mul(_,length),cell)),previous) == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),query,previous),
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,Readback.selected(~Element,first,query,previous)),
        lines_step(~Element,~Part,~value,run,rest,line,view,part,query,previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,first,query,previous),previous,kept) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,_) == previous : Element}
      lines_before(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,rest,1n+line,position,
        Order.advance_room(line,rest,U32.to_nat(outer),inside),
        Bounds.nat_trans(position,Nat.add(length,Nat.mul(line,length)),Nat.add(cells(start,inner),Nat.add(length,Nat.mul(line,length))),
          %Order.commutative(Nat.mul(line,length),length) : {Nat.is_lt(position,_) == True{} : Bool}
          Order.offset_lt(Nat.mul(line,length),cell,length,cell_inside),
          Order.drop_left_le(cells(start,inner),Nat.add(length,Nat.mul(line,length)),Nat.add(cells(start,inner),Nat.add(length,Nat.mul(line,length))),
            Order.reflexive(Nat.add(cells(start,inner),Nat.add(length,Nat.mul(line,length)))))))
    case 1n+ +rest 1n+ +more:
      +run = cells(part_extent,inner)
      +length = cells(extent,inner)
      +view = middle_view(outer,extent,part_extent,inner,start)
      +target = Nat.add(line,1n+more)
      +line_inside = Order.room_index(line,rest,U32.to_nat(outer),inside)
      +target_inside = Bounds.nat_trans(target,Nat.add(line,1n+rest),U32.to_nat(outer),Order.offset_lt(line,1n+more,1n+rest,reached),inside)
      +first = ViewWrites.writes_from(~Element,~Part,~value,run,Nat.mul(line,run),view,part)
      +position = Nat.add(Nat.mul(target,length),cell)
      +query = U32.from_nat(position)
      +passed = line_passed(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,line,position,line_inside,
        Order.transitive(Nat.mul(1n+line,length),Nat.mul(target,length),position,later_line(line,more,length),Order.le_add(Nat.mul(target,length),cell)),
        Bounds.nat_trans(position,Nat.mul(1n+target,length),U32.to_nat(capacity),
          %Order.commutative(Nat.mul(target,length),length) : {Nat.is_lt(position,_) == True{} : Bool}
          Order.offset_lt(Nat.mul(target,length),cell,length,cell_inside),
          next_line(outer,extent,inner,count,start,part_extent,capacity,fit,target,target_inside)))
      %Equal.sym(Element,Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(1n+rest,run),Nat.mul(line,run),view,part),query,previous),
        Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,Readback.selected(~Element,first,query,previous)),
        lines_step(~Element,~Part,~value,run,rest,line,view,part,query,previous)) : {_ == previous : Element}
      %Equal.sym(Element,Readback.selected(~Element,first,query,previous),previous,passed) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),query,_) == previous : Element}
      %Equal.sym(Nat,Nat.add(line,1n+more),1n+Nat.add(line,more),Addition.successor_right(line,more)) :
        {Readback.selected(~Element,ViewWrites.writes_from(~Element,~Part,~value,Nat.mul(rest,run),Nat.mul(1n+line,run),view,part),U32.from_nat(Nat.add(Nat.mul(_,length),cell)),previous) == previous : Element}
      lines_pass(~Element,~Part,~value,outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,rest,1n+line,more,cell,outside,
        Order.advance_room(line,rest,U32.to_nat(outer),inside),reached,cell_inside)

# A part slice along the middle axis holds the part element offset of a line
# at the cell offset of that line past the start of the slice.
def middle_hit(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part_count: U32,+part_exact: {Nat.mul(U32.to_nat(outer),cells(part_extent,inner)) == U32.to_nat(part_count) : Nat},
  +part: Part,+previous: Element,+line: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,cells(part_extent,inner)) == True{} : Bool}) ->
  {slice_element(~Element,~Part,~part_value,slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,U32.from_nat(middle_cell(start,extent,inner,line,offset))) ==
    part_value(part,U32.from_nat(Nat.add(Nat.mul(line,cells(part_extent,inner)),offset))) : Element}:
  +view = middle_view(outer,extent,part_extent,inner,start)
  +query = U32.from_nat(middle_cell(start,extent,inner,line,offset))
  +stored = part_value(part,U32.from_nat(Nat.add(Nat.mul(line,cells(part_extent,inner)),offset)))
  %Equal.sym(Maybe<&2,Views.View>,slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),Some{view},middle_slice(outer,extent,inner,count,start,part_extent,capacity,fit)) :
    {slice_element(~Element,~Part,~part_value,_,part,previous,query) == stored : Element}
  %Equal.sym(Nat,count_of(view),U32.to_nat(part_count),middle_count(outer,extent,part_extent,inner,start,part_count,part_exact)) :
    {Scatter.scattered(~Element,~Part,~part_value,_,view,part,previous,query) == stored : Element}
  %part_exact : {Scatter.scattered(~Element,~Part,~part_value,_,view,part,previous,query) == stored : Element}
  lines_reach(~Element,~Part,~Scatter.logical(~Element,~Part,~part_value),outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,U32.to_nat(outer),0n,line,offset,
    Order.reflexive(U32.to_nat(outer)),line_inside,offset_inside)

# A part slice along the middle axis keeps the previous element at a cell of
# a line outside the slice.
def middle_miss(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+outer: U32,+extent: U32,+inner: U32,+count: U32,+start: U32,+part_extent: U32,+capacity: U32,
  +fit: MiddleFit<outer,extent,inner,count,start,part_extent,capacity>,+part_count: U32,+part_exact: {Nat.mul(U32.to_nat(outer),cells(part_extent,inner)) == U32.to_nat(part_count) : Nat},
  +part: Part,+previous: Element,+line: Nat,+cell: Nat,
  +outside: Either<&2,&2,{Nat.is_lt(cell,cells(start,inner)) == True{} : Bool},{Nat.is_le(Nat.add(cells(start,inner),cells(part_extent,inner)),cell) == True{} : Bool}>,
  +line_inside: {Nat.is_lt(line,U32.to_nat(outer)) == True{} : Bool},+cell_inside: {Nat.is_lt(cell,cells(extent,inner)) == True{} : Bool}) ->
  {slice_element(~Element,~Part,~part_value,slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),part,previous,
    U32.from_nat(Nat.add(Nat.mul(line,cells(extent,inner)),cell))) == previous : Element}:
  +view = middle_view(outer,extent,part_extent,inner,start)
  +query = U32.from_nat(Nat.add(Nat.mul(line,cells(extent,inner)),cell))
  %Equal.sym(Maybe<&2,Views.View>,slice_of(1,start,[outer,extent,inner],[outer,part_extent,inner]),Some{view},middle_slice(outer,extent,inner,count,start,part_extent,capacity,fit)) :
    {slice_element(~Element,~Part,~part_value,_,part,previous,query) == previous : Element}
  %Equal.sym(Nat,count_of(view),U32.to_nat(part_count),middle_count(outer,extent,part_extent,inner,start,part_count,part_exact)) :
    {Scatter.scattered(~Element,~Part,~part_value,_,view,part,previous,query) == previous : Element}
  %part_exact : {Scatter.scattered(~Element,~Part,~part_value,_,view,part,previous,query) == previous : Element}
  lines_pass(~Element,~Part,~Scatter.logical(~Element,~Part,~part_value),outer,extent,inner,count,start,part_extent,capacity,fit,part,previous,U32.to_nat(outer),0n,line,cell,outside,
    Order.reflexive(U32.to_nat(outer)),line_inside,cell_inside)
