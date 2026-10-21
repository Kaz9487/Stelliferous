# The layout an operation runs in is offered by one of its operands. A
# tensor arrives in it when its bands each lie in one image and cover the
# rows of the batch; those facts name only the images and the height, so
# operands of other channels take the same plan.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_band_schedule.bend as Schedule
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_band_count.bend as BandCount
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_plan.bend as Coverage
import ./tensor_band_inspection.bend as Inspection
import ./storage_shape.bend as Shape
import ../machine_limits.bend as Machine
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_band_gather.bend as BandGather
import ./kernel_band_planning_refinement.bend as Planning

type Owner<-Element: Data,-shape: List<&2,U32>,-geometry: Bands.Geometry,-layout: Bands.Plan> is Data:
  Owner{model: Model.Model<Element>,well_formed: Metadata.WellFormed<Element,geometry,Model.content(~Element,model)>,counted: Global.Counted<shape>,
    coherent: Global.Coherent(shape,geometry),same: {Model.layout(~Element,model) == layout : Bands.Plan}}

# A property of the geometry and the layout of a target.
def Holds(~Property: Bands.Geometry -> Bands.Plan -> Data,target: Schedule.Target) -> Data:
  Schedule.Target{geometry,layout,split} = target
  Property(geometry,layout)

def owner_at(~Element: Data,+shape: List<&2,U32>,geometry: Bands.Geometry,layout: Bands.Plan) -> Data:
  Owner<Element,shape,geometry,layout>

def Owned(~Element: Data,+shape: List<&2,U32>,target: Schedule.Target) -> Data:
  Holds(~owner_at(~Element,shape),target)

def owner_counted(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,owner: Owner<Element,shape,geometry,layout>) -> Global.Counted<shape>:
  Owner{model,well_formed,counted,coherent,same} = owner
  counted

def owner_coherent(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,owner: Owner<Element,shape,geometry,layout>) -> Global.Coherent(shape,geometry):
  Owner{model,well_formed,counted,coherent,same} = owner
  coherent

def covered(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,owner: Owner<Element,shape,geometry,layout>) ->
  Coverage.Coverage(layout,0n,Inspection.total_rows(geometry)):
  Owner{+model,Metadata.WellFormed{metadata,ready,coverage},counted,coherent,+same} = owner
  %Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.content(~Element,model)),Model.layout(~Element,model),layout,
    Equal.sym(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),Model.layout_matches_content(~Element,model)),same) :
    Coverage.Coverage(_,0n,Inspection.total_rows(geometry))
  coverage

# A Normal tensor offers its own geometry and plan.
def offered(~Element: Data,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  normal: Global.Normal(~Element,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Owner<Element,shape,geometry,Model.layout(~Element,model)>:
  Shape.Both{well_formed,Shape.Both{counted,coherent}} = normal
  Owner{model,%Model.packed_contents(~Element,model) : Metadata.WellFormed<Element,geometry,_> well_formed,counted,coherent,{==}}

def offer(~Element: Data,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Tensors.offer(~Element,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}) ==
    (Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers},Schedule.Target{geometry,Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model))}) :
    Tensors.Tensor<Element> & Schedule.Target}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,model)),(Model.pack(~Element,model),Model.layout(~Element,model)),Model.plan(~Element,model)) :
    {Tensors.offered(~Element,geometry,shape,fill,Tensors.Usable{},workers,_) ==
      (Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers},Schedule.Target{geometry,Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model))}) :
      Tensors.Tensor<Element> & Schedule.Target}
  {==}

# What a tensor needs to arrive in a layout of the geometry: its shape counts
# the elements of the geometry, and the plan lies in the images.
def Planned(+geometry: Bands.Geometry,+layout: Bands.Plan) -> Data:
  Shape.Both<Metadata.Spans(geometry,layout),Coverage.Coverage(layout,0n,Inspection.total_rows(geometry))>

type Arrival<-shape: List<&2,U32>,-geometry: Bands.Geometry,-layout: Bands.Plan> is Data:
  Arrival{counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),planned: Planned(geometry,layout)}

def arrival_counted(+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,arrival: Arrival<shape,geometry,layout>) -> Global.Counted<shape>:
  Arrival{counted,coherent,planned} = arrival
  counted

def arrival_coherent(+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,arrival: Arrival<shape,geometry,layout>) -> Global.Coherent(shape,geometry):
  Arrival{counted,coherent,planned} = arrival
  coherent

def arrival_spans(+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,arrival: Arrival<shape,geometry,layout>) -> Metadata.Spans(geometry,layout):
  Arrival{counted,coherent,Shape.Both{spans,coverage}} = arrival
  spans

def arrival_covered(+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,arrival: Arrival<shape,geometry,layout>) ->
  Coverage.Coverage(layout,0n,Inspection.total_rows(geometry)):
  Arrival{counted,coherent,Shape.Both{spans,coverage}} = arrival
  coverage

# The layout of well formed bands lies where their metadata states.
def owner_planned(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+owner: Owner<Element,shape,geometry,layout>) -> Planned(geometry,layout):
  Owner{+model,Metadata.WellFormed{metadata,ready,coverage},counted,coherent,+same} = owner
  Shape.Both{%same : Metadata.Spans(geometry,_) Metadata.spans(~Element,geometry,model,metadata),covered(~Element,shape,geometry,layout,owner)}

def owner_arrival(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+owner: Owner<Element,shape,geometry,layout>) -> Arrival<shape,geometry,layout>:
  Arrival{owner_counted(~Element,shape,geometry,layout,owner),owner_coherent(~Element,shape,geometry,layout,owner),owner_planned(~Element,shape,geometry,layout,owner)}

# A plan that lies in the images of the output and covers its rows, whatever
# geometry the operand that offers it has.
def planned_in(+output: Bands.Geometry,geometry: Bands.Geometry,layout: Bands.Plan) -> Data:
  Planned(output,layout)

# A plan that fits the geometry and covers its rows lies in its images.
def fitted_planned(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+layout: Bands.Plan,
  facts: Shape.Both<FromWhole.Fits(geometry,layout,capacity,Machine.half_word(limits)),Coverage.Coverage(layout,0n,Inspection.total_rows(geometry))>) -> Planned(geometry,layout):
  Shape.Both{fits,coverage} = facts
  Shape.Both{FromWhole.fitted_spans(geometry,layout,capacity,Machine.half_word(limits),fits),coverage}

# Every image of a geometry split depth times lies in its images and covers
# its rows. Depth 0 keeps every image as one band.
def split_planned(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+depth: Nat,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  Planned(geometry,FromWhole.split_plan(geometry,depth)):
  fitted_planned(geometry,limits,capacity,FromWhole.split_plan(geometry,depth),Planning.image_plan(geometry,limits,capacity,depth,domain))

def chosen(~Property: Bands.Geometry -> Bands.Plan -> Data,choose_left: Bool,+geometry: Bands.Geometry,+left: Bands.Plan,+right: Bands.Plan,
  left_holds: Property(geometry,left),right_holds: Property(geometry,right)) -> Property(geometry,Bool.pick(Bands.Plan,choose_left,left,right)):
  match choose_left:
    case True{}: left_holds
    case False{}: right_holds

def joined(~Property: Bands.Geometry -> Bands.Plan -> Data,same: Bool,+geometry: Bands.Geometry,+layout: Bands.Plan,+other: Bands.Geometry,+other_layout: Bands.Plan,
  compared: {Bands.same_geometry(geometry,other) == same : Bool},left: Property(geometry,layout),right: Property(other,other_layout)) ->
  Holds(~Property,Schedule.joined(same,geometry,layout,other_layout)):
  match same:
    case True{}:
      +right = {%Equal.sym(Bands.Geometry,geometry,other,BandCount.same_geometry_equal(geometry,other,compared)) : Property(_,other_layout) right : Property(geometry,other_layout)}
      chosen(~Property,Bands.same_plan(layout,other_layout) || Nat.is_le(Schedule.allocation_elements(layout,geometry),Schedule.allocation_elements(other_layout,geometry)),
        geometry,layout,other_layout,left,right)
    case False{}: left

def met(~Property: Bands.Geometry -> Bands.Plan -> Data,split: Bool,other_split: Bool,+geometry: Bands.Geometry,+layout: Bands.Plan,+other: Bands.Geometry,+other_layout: Bands.Plan,
  left: Property(geometry,layout),right: Property(other,other_layout)) -> Holds(~Property,Schedule.met(split,other_split,geometry,layout,other,other_layout)):
  match split other_split:
    case True{} True{}: joined(~Property,Bands.same_geometry(geometry,other),geometry,layout,other,other_layout,{==},left,right)
    case True{} False{}: left
    case False{} True{}: right
    case False{} False{}: left

# A property of the geometry and the layout of both operands holds where they
# meet: the layout they meet in is the layout of one of them.
def meet(~Property: Bands.Geometry -> Bands.Plan -> Data,+left: Schedule.Target,+right: Schedule.Target,
  left_holds: Holds(~Property,left),right_holds: Holds(~Property,right)) -> Holds(~Property,Schedule.meet(left,right)):
  match left right:
    case Schedule.Target{+geometry,+layout,split} Schedule.Target{+other,+other_layout,other_split}:
      met(~Property,split,other_split,geometry,layout,other,other_layout,left_holds,right_holds)
