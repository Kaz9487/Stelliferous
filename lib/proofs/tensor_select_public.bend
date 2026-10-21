# Select on three Normal tensors of one shape is a Normal tensor whose cells
# hold the selected values, whatever storage each input has: the inputs meet
# in the layout of one of them, each arrives in it, and one band kernel runs.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_select.bend as Select
import ../tensor_band.bend as Bands
import ../tensor_shape.bend as Shapes
import ../tensor_band_schedule.bend as Schedule
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as Values
import ./tensor_band_shape_refinement.bend as Metadata
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_target.bend as Target
import ./tensor_band_arrive.bend as Arrive
import ./tensor_select_band_contract.bend as Contract
import ./tensor_select_band_values.bend as SelectValues
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape

def tensor(+model: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32) -> Tensors.Tensor<F32>:
  Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers}

def target(+model: Model.Model<F32>,+geometry: Bands.Geometry) -> Schedule.Target:
  Schedule.Target{geometry,Model.layout(~F32,model),Schedule.splits(Model.layout(~F32,model))}

# The operands after each one offered its layout.
def offers(+shape: List<&2,U32>,+fill: F32,+workers: U32,mask: Tensors.Tensor<F32> & Schedule.Target,yes: Tensors.Tensor<F32> & Schedule.Target,no: Tensors.Tensor<F32> & Schedule.Target) -> Tensors.Tensor<F32>:
  Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,mask,mask => mask_target =>
    Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,yes,yes => yes_target =>
      Tensors.then(~Tensors.Tensor<F32>,~Schedule.Target,~Tensors.Tensor<F32>,no,no => no_target =>
        Select.selected(Schedule.meet(mask_target,Schedule.meet(yes_target,no_target)),shape,fill,workers,mask,yes,no))))

def met(+shape: List<&2,U32>,+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mask_geometry: Bands.Geometry,+yes_geometry: Bands.Geometry,+no_geometry: Bands.Geometry,
  +mask_fill: F32,+yes_fill: F32,+no_fill: F32,+mask_workers: U32,+yes_workers: U32,+no_workers: U32,+fill: F32,+workers: U32) -> Tensors.Tensor<F32>:
  Select.selected(Schedule.meet(target(mask,mask_geometry),Schedule.meet(target(yes,yes_geometry),target(no,no_geometry))),shape,fill,workers,
    tensor(mask,mask_geometry,shape,mask_fill,mask_workers),tensor(yes,yes_geometry,shape,yes_fill,yes_workers),tensor(no,no_geometry,shape,no_fill,no_workers))

# Inputs of one shape offer their layouts and meet.
def offered_call(+shape: List<&2,U32>,+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,
  +mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,+fill: F32,+workers: U32) ->
  {Select.matching(shape,fill,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) == met(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers) : Tensors.Tensor<F32>}:
  %Equal.sym(Tensors.Tensor<F32> & Schedule.Target,Tensors.offer(~F32,tensor(mask,mg,shape,mf,mw)),(tensor(mask,mg,shape,mf,mw),target(mask,mg)),Target.offer(~F32,mask,mg,shape,mf,mw)) :
    {offers(shape,fill,workers,_,Tensors.offer(~F32,tensor(yes,yg,shape,yf,yw)),Tensors.offer(~F32,tensor(no,ng,shape,nf,nw))) == met(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers) : Tensors.Tensor<F32>}
  %Equal.sym(Tensors.Tensor<F32> & Schedule.Target,Tensors.offer(~F32,tensor(yes,yg,shape,yf,yw)),(tensor(yes,yg,shape,yf,yw),target(yes,yg)),Target.offer(~F32,yes,yg,shape,yf,yw)) :
    {offers(shape,fill,workers,(tensor(mask,mg,shape,mf,mw),target(mask,mg)),_,Tensors.offer(~F32,tensor(no,ng,shape,nf,nw))) == met(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers) : Tensors.Tensor<F32>}
  %Equal.sym(Tensors.Tensor<F32> & Schedule.Target,Tensors.offer(~F32,tensor(no,ng,shape,nf,nw)),(tensor(no,ng,shape,nf,nw),target(no,ng)),Target.offer(~F32,no,ng,shape,nf,nw)) :
    {offers(shape,fill,workers,(tensor(mask,mg,shape,mf,mw),target(mask,mg)),(tensor(yes,yg,shape,yf,yw),target(yes,yg)),_) == met(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers) : Tensors.Tensor<F32>}
  {==}

# Inputs of one shape with a count select through the operands that meet.
def call(+shape: List<&2,U32>,+count: U32,+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,
  +mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,counted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {F.select(tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) ==
    Select.matching(shape,yf,U32.max(mw,U32.max(yw,nw)),tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) : Tensors.Tensor<F32>}:
  +workers = U32.max(mw,U32.max(yw,nw))
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(shape,shape),Some{shape},UnitBroadcast.self_broadcast(shape)) :
    {Select.values_shape(_,shape,shape,shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) == Select.matching(shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) : Tensors.Tensor<F32>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(shape,shape),Some{shape},UnitBroadcast.self_broadcast(shape)) :
    {Select.masked_shape(_,shape,shape,shape,shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) == Select.matching(shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) : Tensors.Tensor<F32>}
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
    {Select.shapes(_ && _,shape,Tensors.count_of(shape),shape,shape,shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) == Select.matching(shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) : Tensors.Tensor<F32>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counted) :
    {Select.shapes(True{},shape,_,shape,shape,shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) == Select.matching(shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)) : Tensors.Tensor<F32>}
  {==}

# Bands that arrived in one layout give the facts of the kernel.
def facts(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,
  mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32),
  mask_arrived: Arrive.Arrived(~F32,~Logical,~mv,logical,geometry,layout,mask),yes_arrived: Arrive.Arrived(~F32,~Logical,~yv,logical,geometry,layout,yes),no_arrived: Arrive.Arrived(~F32,~Logical,~nv,logical,geometry,layout,no)) ->
  Contract.Facts(~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,geometry,Select.selected_bands(geometry,layout,parallel,mask,yes,no)):
  Shape.Both{mask_layout,Shape.Both{mask_shape,mask_values}} = mask_arrived
  Shape.Both{yes_layout,Shape.Both{yes_shape,yes_values}} = yes_arrived
  Shape.Both{no_layout,Shape.Both{no_shape,no_values}} = no_arrived
  Contract.selected(~Logical,~mv,~yv,~nv,logical,geometry,layout,parallel,mask,yes,no,mask_layout,yes_layout,no_layout,yes_shape,no_shape,mask_values,yes_values,no_values)

# Each input arrives in the layout the inputs met in.
def arrived(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +target: Schedule.Target,+owner: Target.Owned(~F32,shape,target),+mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,
  +mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  mask_formed: Metadata.WellFormed<F32,mg,Model.content(~F32,mask)>,yes_formed: Metadata.WellFormed<F32,yg,Model.content(~F32,yes)>,no_formed: Metadata.WellFormed<F32,ng,Model.content(~F32,no)>,
  mask_coherent: Global.Coherent(shape,mg),yes_coherent: Global.Coherent(shape,yg),no_coherent: Global.Coherent(shape,ng),
  mask_values: Values.Values(~F32,~Logical,~mv,logical,mg,Model.content(~F32,mask)),yes_values: Values.Values(~F32,~Logical,~yv,logical,yg,Model.content(~F32,yes)),no_values: Values.Values(~F32,~Logical,~nv,logical,ng,Model.content(~F32,no))) ->
  Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    Select.selected(target,shape,fill,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw))):
  match target:
    case Schedule.Target{+geometry,+layout,split}:
      +parallel = Schedule.parallel(geometry,workers,16.0,0.875,layout)
      +arrival = Target.owner_arrival(~F32,shape,geometry,layout,owner)
      Global.contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,geometry,shape,fill,workers,
        Select.selected_bands(geometry,layout,parallel,Tensors.arrive(~F32,geometry,layout,tensor(mask,mg,shape,mf,mw)),
          Tensors.arrive(~F32,geometry,layout,tensor(yes,yg,shape,yf,yw)),Tensors.arrive(~F32,geometry,layout,tensor(no,ng,shape,nf,nw))),
        Target.owner_counted(~F32,shape,geometry,layout,owner),Target.owner_coherent(~F32,shape,geometry,layout,owner),
        facts(~Logical,~mv,~yv,~nv,logical,geometry,layout,parallel,Tensors.arrive(~F32,geometry,layout,tensor(mask,mg,shape,mf,mw)),
          Tensors.arrive(~F32,geometry,layout,tensor(yes,yg,shape,yf,yw)),Tensors.arrive(~F32,geometry,layout,tensor(no,ng,shape,nf,nw)),
          Arrive.arrived(~F32,~Logical,~mv,logical,shape,geometry,layout,mg,mf,Tensors.Usable{},mw,mask,arrival,mask_formed,mask_coherent,mask_values),
          Arrive.arrived(~F32,~Logical,~yv,logical,shape,geometry,layout,yg,yf,Tensors.Usable{},yw,yes,arrival,yes_formed,yes_coherent,yes_values),
          Arrive.arrived(~F32,~Logical,~nv,logical,shape,geometry,layout,ng,nf,Tensors.Usable{},nw,no,arrival,no_formed,no_coherent,no_values)))

# Inputs of one shape that offer their layouts select into a Normal tensor.
def offered(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  +fill: F32,+workers: U32,
  mask_contract: Global.Contract(~F32,~Logical,~mv,logical,tensor(mask,mg,shape,mf,mw)),yes_contract: Global.Contract(~F32,~Logical,~yv,logical,tensor(yes,yg,shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,logical,tensor(no,ng,shape,nf,nw))) ->
  Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    Select.matching(shape,fill,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw))):
  Shape.Both{Shape.Both{+mask_shape,Shape.Both{+mask_counted,+mask_coherent}},mask_values} = mask_contract
  Shape.Both{Shape.Both{+yes_shape,Shape.Both{+yes_counted,+yes_coherent}},yes_values} = yes_contract
  Shape.Both{Shape.Both{+no_shape,Shape.Both{+no_counted,+no_coherent}},no_values} = no_contract
  +mask_owner = Target.offered(~F32,mask,mg,shape,mf,mw,Shape.Both{mask_shape,Shape.Both{mask_counted,mask_coherent}})
  +yes_owner = Target.offered(~F32,yes,yg,shape,yf,yw,Shape.Both{yes_shape,Shape.Both{yes_counted,yes_coherent}})
  +no_owner = Target.offered(~F32,no,ng,shape,nf,nw,Shape.Both{no_shape,Shape.Both{no_counted,no_coherent}})
  +met_target = Schedule.meet(target(mask,mg),Schedule.meet(target(yes,yg),target(no,ng)))
  %Equal.sym(Tensors.Tensor<F32>,Select.matching(shape,fill,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)),
    met(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers),offered_call(shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,fill,workers)) :
    Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,_)
  arrived(~Logical,~mv,~yv,~nv,logical,shape,fill,workers,met_target,
    Target.meet(~Target.owner_at(~F32,shape),target(mask,mg),Schedule.meet(target(yes,yg),target(no,ng)),mask_owner,Target.meet(~Target.owner_at(~F32,shape),target(yes,yg),target(no,ng),yes_owner,no_owner)),
    mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,
    %Model.packed_contents(~F32,mask) : Metadata.WellFormed<F32,mg,_> mask_shape,%Model.packed_contents(~F32,yes) : Metadata.WellFormed<F32,yg,_> yes_shape,
    %Model.packed_contents(~F32,no) : Metadata.WellFormed<F32,ng,_> no_shape,mask_coherent,yes_coherent,no_coherent,
    %Model.packed_contents(~F32,mask) : Values.Values(~F32,~Logical,~mv,logical,mg,_) mask_values,%Model.packed_contents(~F32,yes) : Values.Values(~F32,~Logical,~yv,logical,yg,_) yes_values,
    %Model.packed_contents(~F32,no) : Values.Values(~F32,~Logical,~nv,logical,ng,_) no_values)

def modeled(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,
  +mask: Model.Model<F32>,+yes: Model.Model<F32>,+no: Model.Model<F32>,+mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,
  mask_contract: Global.Contract(~F32,~Logical,~mv,logical,tensor(mask,mg,shape,mf,mw)),yes_contract: Global.Contract(~F32,~Logical,~yv,logical,tensor(yes,yg,shape,yf,yw)),
  no_contract: Global.Contract(~F32,~Logical,~nv,logical,tensor(no,ng,shape,nf,nw))) ->
  Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,F.select(tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw))):
  Shape.Both{Shape.Both{+mask_shape,Shape.Both{Global.Counted{+count,+accepted},+mask_coherent}},mask_values} = mask_contract
  +workers = U32.max(mw,U32.max(yw,nw))
  %Equal.sym(Tensors.Tensor<F32>,F.select(tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)),
    Select.matching(shape,yf,workers,tensor(mask,mg,shape,mf,mw),tensor(yes,yg,shape,yf,yw),tensor(no,ng,shape,nf,nw)),call(shape,count,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,accepted)) :
    Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,_)
  offered(~Logical,~mv,~yv,~nv,logical,shape,mask,yes,no,mg,yg,ng,mf,yf,nf,mw,yw,nw,yf,workers,
    Shape.Both{Shape.Both{mask_shape,Shape.Both{Global.Counted{count,accepted},mask_coherent}},mask_values},yes_contract,no_contract)

def observed(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,
  +mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,-mask: Bands.Bands(~F32),-yes: Bands.Bands(~F32),-no: Bands.Bands(~F32),
  mask_observed: Model.Observation<F32,mask>,yes_observed: Model.Observation<F32,yes>,no_observed: Model.Observation<F32,no>,
  mask_contract: Global.Contract(~F32,~Logical,~mv,logical,Tensors.Tensor{mask,mg,shape,mf,Tensors.Usable{},mw}),yes_contract: Global.Contract(~F32,~Logical,~yv,logical,Tensors.Tensor{yes,yg,shape,yf,Tensors.Usable{},yw}),
  no_contract: Global.Contract(~F32,~Logical,~nv,logical,Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw})) ->
  Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    F.select(Tensors.Tensor{mask,mg,shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw})):
  Model.Observation{+m,+me} = mask_observed
  Model.Observation{+y,+ye} = yes_observed
  Model.Observation{+n,+ne} = no_observed
  %me : Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    F.select(Tensors.Tensor{_,mg,shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw}))
  %ye : Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    F.select(tensor(m,mg,shape,mf,mw),Tensors.Tensor{_,yg,shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw}))
  %ne : Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    F.select(tensor(m,mg,shape,mf,mw),tensor(y,yg,shape,yf,yw),Tensors.Tensor{_,ng,shape,nf,Tensors.Usable{},nw}))
  modeled(~Logical,~mv,~yv,~nv,logical,shape,m,y,n,mg,yg,ng,mf,yf,nf,mw,yw,nw,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,m),mask,me) : Global.Contract(~F32,~Logical,~mv,logical,Tensors.Tensor{_,mg,shape,mf,Tensors.Usable{},mw}) mask_contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,y),yes,ye) : Global.Contract(~F32,~Logical,~yv,logical,Tensors.Tensor{_,yg,shape,yf,Tensors.Usable{},yw}) yes_contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,n),no,ne) : Global.Contract(~F32,~Logical,~nv,logical,Tensors.Tensor{_,ng,shape,nf,Tensors.Usable{},nw}) no_contract)

# Three Normal tensors of one shape select into a Normal tensor that holds,
# at every flat index, the yes value where the mask is nonzero and the no
# value elsewhere.
def select(~Logical: Data,~mv: Logical -> U32 -> F32,~yv: Logical -> U32 -> F32,~nv: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,
  +mg: Bands.Geometry,+yg: Bands.Geometry,+ng: Bands.Geometry,+mf: F32,+yf: F32,+nf: F32,+mw: U32,+yw: U32,+nw: U32,mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32),
  mask_contract: Global.Contract(~F32,~Logical,~mv,logical,Tensors.Tensor{mask,mg,shape,mf,Tensors.Usable{},mw}),yes_contract: Global.Contract(~F32,~Logical,~yv,logical,Tensors.Tensor{yes,yg,shape,yf,Tensors.Usable{},yw}),
  no_contract: Global.Contract(~F32,~Logical,~nv,logical,Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw})) ->
  Global.Contract(~F32,~Logical,~SelectValues.value(~Logical,~mv,~yv,~nv),logical,
    F.select(Tensors.Tensor{mask,mg,shape,mf,Tensors.Usable{},mw},Tensors.Tensor{yes,yg,shape,yf,Tensors.Usable{},yw},Tensors.Tensor{no,ng,shape,nf,Tensors.Usable{},nw})):
  observed(~Logical,~mv,~yv,~nv,logical,shape,mg,yg,ng,mf,yf,nf,mw,yw,nw,mask,yes,no,Model.observe(~F32,mask),Model.observe(~F32,yes),Model.observe(~F32,no),mask_contract,yes_contract,no_contract)
