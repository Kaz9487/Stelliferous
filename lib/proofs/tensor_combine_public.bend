# Combining two Normal tensors of one shape gives a Normal tensor whose cells
# hold the operation of the two values, whatever storage each operand has:
# the operands meet in the layout of one of them, each arrives in it, and one
# band kernel runs.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_shape.bend as Shapes
import ../tensor_band_schedule.bend as Schedule
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_target.bend as Target
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_combine.bend as Kernel
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape

def tensor(~Element: Data,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}

def target(~Element: Data,+model: Model.Model<Element>,+geometry: Bands.Geometry) -> Schedule.Target:
  Schedule.Target{geometry,Model.layout(~Element,model),Schedule.splits(Model.layout(~Element,model))}

# The operands after each one offered its layout.
def offers(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),operations: F32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left: Tensors.Tensor<Element> & Schedule.Target,right: Tensors.Tensor<Element> & Schedule.Target) -> Tensors.Tensor<Element>:
  Tensors.then(~Tensors.Tensor<Element>,~Schedule.Target,~Tensors.Tensor<Element>,left,left => left_target =>
    Tensors.then(~Tensors.Tensor<Element>,~Schedule.Target,~Tensors.Tensor<Element>,right,right => right_target =>
      Tensors.combine_met_with(~Element,~kernel,operations,Schedule.meet(left_target,right_target),shape,fill,workers,left,right)))

def met(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +left_geometry: Bands.Geometry,+right_geometry: Bands.Geometry,+left_fill: Element,+right_fill: Element,+left_workers: U32,+right_workers: U32,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.combine_met_with(~Element,~kernel,operations,Schedule.meet(target(~Element,left,left_geometry),target(~Element,right,right_geometry)),shape,fill,workers,
    tensor(~Element,left,left_geometry,shape,left_fill,left_workers),tensor(~Element,right,right_geometry,shape,right_fill,right_workers))

# Operands that offer their layouts meet in one of them.
def offered_call(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,+fill: Element,+workers: U32) ->
  {Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw)) ==
    met(~Element,~kernel,operations,shape,left,right,lg,rg,lf,rf,lw,rw,fill,workers) : Tensors.Tensor<Element>}:
  %Equal.sym(Tensors.Tensor<Element> & Schedule.Target,Tensors.offer(~Element,tensor(~Element,left,lg,shape,lf,lw)),(tensor(~Element,left,lg,shape,lf,lw),target(~Element,left,lg)),
    Target.offer(~Element,left,lg,shape,lf,lw)) :
    {offers(~Element,~kernel,operations,shape,fill,workers,_,Tensors.offer(~Element,tensor(~Element,right,rg,shape,rf,rw))) == met(~Element,~kernel,operations,shape,left,right,lg,rg,lf,rf,lw,rw,fill,workers) : Tensors.Tensor<Element>}
  %Equal.sym(Tensors.Tensor<Element> & Schedule.Target,Tensors.offer(~Element,tensor(~Element,right,rg,shape,rf,rw)),(tensor(~Element,right,rg,shape,rf,rw),target(~Element,right,rg)),
    Target.offer(~Element,right,rg,shape,rf,rw)) :
    {offers(~Element,~kernel,operations,shape,fill,workers,(tensor(~Element,left,lg,shape,lf,lw),target(~Element,left,lg)),_) == met(~Element,~kernel,operations,shape,left,right,lg,rg,lf,rf,lw,rw,fill,workers) : Tensors.Tensor<Element>}
  {==}

# Operands of one shape take the equal route with their largest worker count.
def call(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+name: String,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32) ->
  {Tensors.combine_with(~Element,~kernel,name,operations,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw)) ==
    Tensors.combine_offered_with(~Element,~kernel,operations,U32.max(lw,rw),shape,lf,tensor(~Element,left,lg,shape,lf,U32.max(lw,rw)),tensor(~Element,right,rg,shape,rf,U32.max(lw,rw))) :
    Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
    {Tensors.combine_shapes(~Element,~kernel,name,operations,_,Model.pack(~Element,left),lg,shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,shape,rf,Tensors.Usable{},rw) ==
      Tensors.combine_offered_with(~Element,~kernel,operations,U32.max(lw,rw),shape,lf,tensor(~Element,left,lg,shape,lf,U32.max(lw,rw)),tensor(~Element,right,rg,shape,rf,U32.max(lw,rw))) :
      Tensors.Tensor<Element>}
  {==}

# Bands that arrived in one layout give the facts of the kernel.
def facts(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  left_arrived: Arrive.Arrived(~Element,~Left,~left_value,left_context,geometry,layout,left),right_arrived: Arrive.Arrived(~Element,~Right,~right_value,right_context,geometry,layout,right)) ->
  Shape.Both<BandValues.Values(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},geometry,
      Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))),
    BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_with(~Element,~kernel,geometry,parallel,left,right))>>:
  Shape.Both{left_layout,Shape.Both{left_shape,left_values}} = left_arrived
  Shape.Both{right_layout,Shape.Both{right_shape,right_values}} = right_arrived
  Kernel.combined(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,parallel,left,right,
    Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,left)),layout,Model.content_layout(~Element,Model.contents(~Element,right)),left_layout,
      Equal.sym(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,right)),layout,right_layout)),
    left_shape,right_shape,left_values,right_values,Arrive.dimension_choice(geometry))

# Each operand arrives in the layout the operands met in.
def arrived(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+fill: Element,+workers: U32,+target: Schedule.Target,+owner: Target.Owned(~Element,shape,target),
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  left_formed: BandShape.WellFormed<Element,lg,Model.content(~Element,left)>,right_formed: BandShape.WellFormed<Element,rg,Model.content(~Element,right)>,
  left_coherent: Global.Coherent(shape,lg),right_coherent: Global.Coherent(shape,rg),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,lg,Model.content(~Element,left)),right_values: BandValues.Values(~Element,~Right,~right_value,right_context,rg,Model.content(~Element,right))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_met_with(~Element,~kernel,operations,target,shape,fill,workers,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw))):
  match target:
    case Schedule.Target{+geometry,+layout,split}:
      +parallel = Schedule.parallel(geometry,workers,12.0,operations,layout)
      +arrival = Target.owner_arrival(~Element,shape,geometry,layout,owner)
      Global.contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},geometry,shape,fill,workers,
        Bands.combine_with(~Element,~kernel,geometry,parallel,Tensors.arrive(~Element,geometry,layout,tensor(~Element,left,lg,shape,lf,lw)),Tensors.arrive(~Element,geometry,layout,tensor(~Element,right,rg,shape,rf,rw))),
        Target.owner_counted(~Element,shape,geometry,layout,owner),Target.owner_coherent(~Element,shape,geometry,layout,owner),
        facts(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,geometry,layout,parallel,
          Tensors.arrive(~Element,geometry,layout,tensor(~Element,left,lg,shape,lf,lw)),Tensors.arrive(~Element,geometry,layout,tensor(~Element,right,rg,shape,rf,rw)),
          Arrive.arrived(~Element,~Left,~left_value,left_context,shape,geometry,layout,lg,lf,Tensors.Usable{},lw,left,arrival,left_formed,left_coherent,left_values),
          Arrive.arrived(~Element,~Right,~right_value,right_context,shape,geometry,layout,rg,rf,Tensors.Usable{},rw,right,arrival,right_formed,right_coherent,right_values)))

# Operands of one shape that offer their layouts combine into a Normal
# tensor holding the operation of the two values at every flat index.
def offered(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,+fill: Element,+workers: U32,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,tensor(~Element,left,lg,shape,lf,lw)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,tensor(~Element,right,rg,shape,rf,rw))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw))):
  Shape.Both{Shape.Both{+left_shape,Shape.Both{+left_counted,+left_coherent}},left_values} = left_contract
  Shape.Both{Shape.Both{+right_shape,Shape.Both{+right_counted,+right_coherent}},right_values} = right_contract
  +left_owner = Target.offered(~Element,left,lg,shape,lf,lw,Shape.Both{left_shape,Shape.Both{left_counted,left_coherent}})
  +right_owner = Target.offered(~Element,right,rg,shape,rf,rw,Shape.Both{right_shape,Shape.Both{right_counted,right_coherent}})
  %Equal.sym(Tensors.Tensor<Element>,Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw)),
    met(~Element,~kernel,operations,shape,left,right,lg,rg,lf,rf,lw,rw,fill,workers),offered_call(~Element,~kernel,operations,shape,left,right,lg,rg,lf,rf,lw,rw,fill,workers)) :
    Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},_)
  arrived(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,fill,workers,
    Schedule.meet(target(~Element,left,lg),target(~Element,right,rg)),Target.meet(~Target.owner_at(~Element,shape),target(~Element,left,lg),target(~Element,right,rg),left_owner,right_owner),
    left,right,lg,rg,lf,rf,lw,rw,
    %Model.packed_contents(~Element,left) : BandShape.WellFormed<Element,lg,_> left_shape,%Model.packed_contents(~Element,right) : BandShape.WellFormed<Element,rg,_> right_shape,
    left_coherent,right_coherent,
    %Model.packed_contents(~Element,left) : BandValues.Values(~Element,~Left,~left_value,left_context,lg,_) left_values,
    %Model.packed_contents(~Element,right) : BandValues.Values(~Element,~Right,~right_value,right_context,rg,_) right_values)

def modeled(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,tensor(~Element,left,lg,shape,lf,lw)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,tensor(~Element,right,rg,shape,rf,rw))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_with(~Element,~kernel,name,operations,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw))):
  +workers = U32.max(lw,rw)
  %Equal.sym(Tensors.Tensor<Element>,Tensors.combine_with(~Element,~kernel,name,operations,tensor(~Element,left,lg,shape,lf,lw),tensor(~Element,right,rg,shape,rf,rw)),
    Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,lf,tensor(~Element,left,lg,shape,lf,workers),tensor(~Element,right,rg,shape,rf,workers)),
    call(~Element,~kernel,name,operations,shape,left,right,lg,rg,lf,rf,lw,rw)) :
    Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},_)
  offered(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,left,right,lg,rg,lf,rf,workers,workers,lf,workers,left_contract,right_contract)

def observed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+shape: List<&2,U32>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  -left: Bands.Bands(~Element),-right: Bands.Bands(~Element),left_observed: Model.Observation<Element,left>,right_observed: Model.Observation<Element,right>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{left,lg,shape,lf,Tensors.Usable{},lw}),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{right,rg,shape,rf,Tensors.Usable{},rw})) ->
  Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left,lg,shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,shape,rf,Tensors.Usable{},rw})):
  Model.Observation{+l,+le} = left_observed
  Model.Observation{+r,+re} = right_observed
  %le : Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{_,lg,shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,shape,rf,Tensors.Usable{},rw}))
  %re : Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_with(~Element,~kernel,name,operations,tensor(~Element,l,lg,shape,lf,lw),Tensors.Tensor{_,rg,shape,rf,Tensors.Usable{},rw}))
  modeled(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,shape,l,r,lg,rg,lf,rf,lw,rw,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,l),left,le) : Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{_,lg,shape,lf,Tensors.Usable{},lw}) left_contract,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,r),right,re) : Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{_,rg,shape,rf,Tensors.Usable{},rw}) right_contract)

# Two Normal tensors of one shape combine into a Normal tensor that holds, at
# every flat index, the operation of the left value and the right value.
def combine(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+shape: List<&2,U32>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{left,lg,shape,lf,Tensors.Usable{},lw}),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{right,rg,shape,rf,Tensors.Usable{},rw})) ->
  Global.Contract(~Element,~Leaf.Context<Left,Right>,~Leaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left_context,right_context},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left,lg,shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,shape,rf,Tensors.Usable{},rw})):
  observed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,shape,lg,rg,lf,rf,lw,rw,left,right,
    Model.observe(~Element,left),Model.observe(~Element,right),left_contract,right_contract)
