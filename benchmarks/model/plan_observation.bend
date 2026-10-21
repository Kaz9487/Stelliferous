# Diagnostic metadata replay: no tensor allocation, convolution or timing.
# Choices and costs are read from the same public caller/planners as the model.
import Base
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/tensor_band_pool.bend as BandPool
import ../../lib/tensor_band_concat.bend as BandConcat
import ../../lib/tensor_shape.bend as Shapes
import ../../lib/tensor_layout.bend as Layout
import ../../lib/tensor_window.bend as Window
import ../../lib/tensor_access.bend as Access
import ../../lib/tensor_band_schedule.bend as BandSchedule
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_planning.bend as KernelPlanning
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/math_activation.bend as Activations
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/default_machine_limits.bend as Default

type Receipt is Data:
  Receipt{layout: Maybe<&2,Bands.Plan>,route: String,depth: U32,parallel: Bool,estimate: Maybe<&2,F32>,repartition: Maybe<&2,F32>,gather: Bool}

def unplanned() -> Receipt:
  Receipt{None{},"unplanned",0,False{},None{},None{},False{}}

def layout(receipt: Receipt) -> Maybe<&2,Bands.Plan>:
  Receipt{layout,route,depth,parallel,estimate,repartition,gather} = receipt
  layout

def output_rows(rows: BandWindow.RowWindow) -> Bands.Plan:
  BandWindow.RowWindow{BandWindow.Rows{first,count},above,below,window} = rows
  Bands.Rows{first,count}

def output_plan(plan: Bands.Plan,+window: BandWindow.Geometry) -> Bands.Plan:
  match plan:
    case Bands.Rows{first,count}: output_rows(BandWindow.window(window,first,count))
    case Bands.Split{left,right}: Bands.Split{output_plan(left,window),output_plan(right,window)}

def selected_initial(+window: BandWindow.Geometry,selection: BandPlanning.Plan) -> Receipt:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},k,s,p,oh,ow} = window
  BandPlanning.Plan{+depth,parallel,time} = selection
  Receipt{Some{output_plan(Bands.image_plan(U32.to_nat(images),0,U32.to_nat(depth),height),window)},"band_initial",depth,parallel,Some{time},Some{0.0},False{}}

def selected_existing(+window: BandWindow.Geometry,selection: BandPlanning.LayoutPlan) -> Receipt:
  BandPlanning.LayoutPlan{+plan,parallel,time,repartition} = selection
  Receipt{Some{output_plan(plan,window)},"band_retained_or_repartitioned",0,parallel,Some{time},Some{repartition},False{}}

def convolution_layout(previous: Maybe<&2,Bands.Plan>,+workers: U32,+work: KernelPlanning.Work,+window: BandWindow.Geometry) -> Receipt:
  match previous:
    case Some{plan}: selected_existing(window,BandPlanning.replan(plan,work,window,workers,DefaultCost.configuration()))
    case None{}: selected_initial(window,BandPlanning.select(work,window,workers,DefaultCost.configuration()))

def convolution(+geometry: TensorConvolution.Geometry,+activation: Activations.Activation,+workers: U32,+bias_count: U32,previous: Receipt) -> Receipt:
  +context = TensorConvolution.band_context(geometry,True{},activation)
  convolution_layout(layout(previous),workers,TensorConvolution.band_work(context,U32.shln(1,Storage.depth_for(bias_count))),TensorConvolution.context_window(context))

def convolution_valid(valid: Bool,+geometry: TensorConvolution.Geometry,activation: Activations.Activation,workers: U32,bias_count: U32,previous: Receipt) -> Receipt:
  match valid:
    case True{}: convolution(geometry,activation,workers,bias_count,previous)
    case False{}: Receipt{None{},"invalid_convolution",0,False{},None{},None{},False{}}

def convolution_geometry(parsed: Maybe<&2,TensorConvolution.Geometry>,+weight_count: U32,+bias_count: U32,activation: Activations.Activation,workers: U32,previous: Receipt) -> Receipt:
  match parsed:
    case None{}: Receipt{None{},"invalid_geometry",0,False{},None{},None{},False{}}
    case Some{+geometry}:
      TensorConvolution.Geometry{images,batched,+dimensions} = geometry
      KernelShape.Dimensions{+ci,co,+h,+w,k,s,p} = dimensions
      convolution_valid(KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),U32.to_nat(weight_count),U32.to_nat(bias_count)) && TensorConvolution.machine_support(geometry,Default.limits()),geometry,activation,workers,bias_count,previous)

def checked_convolution(input_shape: List<&2,U32>,weight_shape: List<&2,U32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,
  weight_count: U32,bias_count: U32,activation: Activations.Activation,workers: U32,previous: Receipt) -> Receipt:
  convolution_geometry(TensorConvolution.geometry(TensorConvolution.input_shape(input_shape),weight_shape,bias_shape,stride,padding),weight_count,bias_count,activation,workers,previous)

# What an operand offers: its plan when its bands split the rows of its
# image, its storage as one band otherwise.
def whole_target(geometry: Bands.Geometry) -> BandSchedule.Target:
  Bands.Geometry{batch,channels,height,width} = geometry
  BandSchedule.Target{Bands.Geometry{1,1,1,(batch * channels * height * width : U32)},Bands.Rows{0,1},False{}}

def operand_target(+geometry: Bands.Geometry,layout: Maybe<&2,Bands.Plan>) -> BandSchedule.Target:
  match layout:
    case Some{+plan}: BandSchedule.Target{geometry,plan,BandSchedule.splits(plan)}
    case None{}: whole_target(geometry)

def combined_image(image: Bool,+workers: U32,+geometry: Bands.Geometry,+plan: Bands.Plan) -> Receipt:
  match image:
    case True{}: Receipt{Some{plan},"binary",0,BandSchedule.parallel(geometry,workers,12.0,1.0,plan),None{},None{},False{}}
    case False{}: Receipt{None{},"binary_whole",0,False{},None{},None{},False{}}

def combined_target(+geometry: Bands.Geometry,+workers: U32,target: BandSchedule.Target) -> Receipt:
  BandSchedule.Target{+met,+plan,split} = target
  combined_image(Bands.same_geometry(met,geometry),workers,geometry,plan)

# The operands meet as the public combine meets them.
def combined(+geometry: Bands.Geometry,workers: U32,left: Receipt,right: Receipt) -> Receipt:
  combined_target(geometry,workers,BandSchedule.meet(operand_target(geometry,layout(left)),operand_target(geometry,layout(right))))

# The parts of a concat offer as the public concat offers them. A part with
# a plan is in the geometry of the joining.
def offered(+geometry: Bands.Geometry,layout: Maybe<&2,Bands.Plan>) -> BandSchedule.Target:
  match layout:
    case Some{plan}: BandConcat.part_target(True{},geometry,plan)
    case None{}: BandConcat.part_target(False{},geometry,Bands.Rows{0,0})

def concat_target(+shape: List<&2,U32>,+axis: U32,layout: Maybe<&2,Bands.Plan>) -> BandSchedule.Target:
  +joining = BandConcat.joining(shape,axis)
  offered(BandConcat.geometry(joining,BandConcat.extent(joining,shape)),layout)

def concat_last(+shape: List<&2,U32>,+axis: U32,part: Receipt) -> BandSchedule.Target:
  concat_target(shape,axis,layout(part))

def concat_cons(+shape: List<&2,U32>,+axis: U32,part: Receipt,rest: BandSchedule.Target) -> BandSchedule.Target:
  BandSchedule.meet(concat_target(shape,axis,layout(part)),rest)

def concat_band(target: BandSchedule.Target) -> Receipt:
  BandSchedule.Target{met,plan,split} = target
  Receipt{Some{plan},"concat",0,False{},None{},None{},False{}}

def concat_selected(empty: Bool,count: Maybe<&2,U32>,target: BandSchedule.Target) -> Receipt:
  match empty count:
    case False{} Some{count}: concat_band(target)
    case False{} None{}: Receipt{None{},"invalid_concat",0,False{},None{},None{},False{}}
    case True{} count: Receipt{None{},"empty_concat",0,False{},None{},None{},False{}}

def concatenated(+shape: List<&2,U32>,axis: U32,target: BandSchedule.Target) -> Receipt:
  concat_selected(Shapes.has_zero(shape),Tensors.count_of(shape),target)

def append_plan(+geometry: Bands.Geometry,+workers: U32,input: Maybe<&2,Bands.Plan>,target: Maybe<&2,Bands.Plan>) -> Bool:
  match input target:
    case Some{input} Some{+target}: Bands.same_plan(input,target) && BandSchedule.parallel(geometry,workers,8.0,0.0,target)
    case input target: False{}

def append_parallel(geometry: Bands.Geometry,workers: U32,input: Receipt,target: Receipt) -> Bool:
  append_plan(geometry,workers,layout(input),layout(target))

# The plan a per-plane operation runs in: the input's own, or its planes as
# one band. A batch of one image keeps the image geometry; other batches run
# in a geometry the receipts do not describe.
def image_band(geometry: Bands.Geometry) -> Maybe<&2,Bands.Plan>:
  Bands.Geometry{batch,channels,+height,width} = geometry
  Bool.pick(Maybe<&2,Bands.Plan>,U32.is_eq(batch,1),Some{Bands.Rows{0,height}},None{})

def plane_plan(+geometry: Bands.Geometry,layout: Maybe<&2,Bands.Plan>) -> Maybe<&2,Bands.Plan>:
  match layout:
    case Some{plan}: Some{plan}
    case None{}: image_band(geometry)

def pool_receipt(+window: BandWindow.Geometry,workers: U32,plan: Maybe<&2,Bands.Plan>) -> Receipt:
  match plan:
    case None{}: Receipt{None{},"plane_pool",0,False{},None{},None{},False{}}
    case Some{+plan}: Receipt{Some{output_plan(plan,window)},"band_pool",0,BandPool.parallel(window,workers,BandPool.plan_size(plan,window)),None{},None{},False{}}

def pooled(+window: BandWindow.Geometry,workers: U32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  pool_receipt(window,workers,plane_plan(BandWindow.input_geometry(window),plan))

def pool_dimensions(pool: Access.Pool,+geometry: Bands.Geometry,workers: U32,previous: Receipt) -> Receipt:
  Access.Pool{height,width,kernel,stride,padding,oh,ow} = pool
  pooled(BandWindow.Geometry{geometry,kernel,stride,padding,oh,ow},workers,previous)

def pool_selected(valid: Bool,empty: Bool,pool: Access.Pool,geometry: Bands.Geometry,workers: U32,previous: Receipt) -> Receipt:
  match valid empty:
    case True{} False{}: pool_dimensions(pool,geometry,workers,previous)
    case True{} True{}: Receipt{None{},"empty_pool",0,False{},None{},None{},False{}}
    case False{} empty: Receipt{None{},"invalid_pool",0,False{},None{},None{},False{}}

def pool_public_geometry(+shape: List<&2,U32>,+geometry: Bands.Geometry,+pool: Access.Pool,workers: U32,previous: Receipt) -> Receipt:
  pool_selected(Window.valid(shape,pool) && Maybe.is_some(&2,U32,Layout.planes(shape)),Shapes.has_zero(Window.output_shape(shape,pool)),pool,geometry,workers,previous)

def pooled_public(+shape: List<&2,U32>,geometry: Bands.Geometry,kernel: U32,stride: U32,padding: U32,workers: U32,previous: Receipt) -> Receipt:
  pool_public_geometry(shape,geometry,Window.geometry(shape,kernel,stride,padding),workers,previous)

def scaled_plan(plan: Bands.Plan,+scale: U32) -> Bands.Plan:
  match plan:
    case Bands.Rows{first,count}: Bands.Rows{(first * scale : U32),(count * scale : U32)}
    case Bands.Split{left,right}: Bands.Split{scaled_plan(left,scale),scaled_plan(right,scale)}

def upsample_receipt(+geometry: Bands.Geometry,+workers: U32,+scale: U32,plan: Maybe<&2,Bands.Plan>) -> Receipt:
  match plan:
    case None{}: Receipt{None{},"plane_upsample",0,False{},None{},None{},False{}}
    case Some{+plan}: Receipt{Some{scaled_plan(plan,scale)},"band_upsample",0,BandSchedule.parallel(geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),0.0,plan),None{},None{},False{}}

def upsampled(+geometry: Bands.Geometry,+workers: U32,+scale: U32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  upsample_receipt(geometry,workers,scale,plane_plan(geometry,plan))

def upsample_selected(valid: Bool,empty: Bool,geometry: Bands.Geometry,workers: U32,scale: U32,previous: Receipt) -> Receipt:
  match valid empty:
    case True{} False{}: upsampled(geometry,workers,scale,previous)
    case True{} True{}: Receipt{None{},"empty_upsample",0,False{},None{},None{},False{}}
    case False{} empty: Receipt{None{},"invalid_upsample",0,False{},None{},None{},False{}}

def upsample_public(+shape: List<&2,U32>,geometry: Bands.Geometry,workers: U32,+scale: U32,previous: Receipt) -> Receipt:
  upsample_selected(Layout.upsample_valid(shape,scale) && Maybe.is_some(&2,U32,Layout.planes(shape)),Shapes.has_zero(Layout.scaled_extents(shape,scale)),
    geometry,workers,scale,previous)

# The indexed decode keeps the logits' bands; the reshape after it gathers.
def decoded(geometry: Bands.Geometry,workers: U32,operations: F32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  match plan:
    case None{}: Receipt{None{},"unplanned_decode",0,False{},None{},None{},False{}}
    case Some{plan}: Receipt{None{},"band_decode",0,BandSchedule.parallel(geometry,workers,12.0,operations,plan),None{},None{},True{}}

def emit_plan(plan: Bands.Plan) -> IO(Unit):
  match plan:
    case Bands.Rows{first,count}:
      do IO<Unit>:
        IO.write("{\"first\":")
        IO.write(Nat.show(U32.to_nat(first)))
        IO.write(",\"rows\":")
        IO.write(Nat.show(U32.to_nat(count)))
        IO.write("}")
    case Bands.Split{left,right}:
      do IO<Unit>:
        IO.write("{\"left\":")
        emit_plan(left)
        IO.write(",\"right\":")
        emit_plan(right)
        IO.write("}")

def emit_layout(plan: Maybe<&2,Bands.Plan>) -> IO(Unit):
  match plan:
    case None{}: IO.write("null")
    case Some{plan}: emit_plan(plan)

def emit_estimate(estimate: Maybe<&2,F32>) -> IO(Unit):
  match estimate:
    case None{}: IO.write("null")
    case Some{time}: IO.write(F32.show(time))

def emit(index: U32,receipt: Receipt) -> IO(Unit):
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = receipt
  do IO<Unit>:
    IO.write("{\"operation\":")
    IO.write(Nat.show(U32.to_nat(index)))
    IO.write(",\"route\":\"")
    IO.write(route)
    IO.write("\",\"output_layout\":")
    emit_layout(plan)
    IO.write(",\"initial_depth\":")
    IO.write(Nat.show(U32.to_nat(depth)))
    IO.write(",\"parallel\":")
    IO.write(Bool.pick(String,parallel,"true","false"))
    IO.write(",\"estimated_ns\":")
    emit_estimate(estimate)
    IO.write(",\"repartition_ns\":")
    emit_estimate(repartition)
    IO.write(",\"gather_input\":")
    IO.write(Bool.pick(String,gather,"true","false"))
    IO.write("}\n")

def emit_append(index: U32,part: U32,parallel: Bool) -> IO(Unit):
  do IO<Unit>:
    IO.write("{\"operation\":")
    IO.write(Nat.show(U32.to_nat(index)))
    IO.write(",\"part\":")
    IO.write(Nat.show(U32.to_nat(part)))
    IO.write(",\"append_parallel\":")
    IO.write(Bool.pick(String,parallel,"true","false"))
    IO.write("}\n")

def emit_count(index: U32,count: Maybe<&2,U32>) -> IO(Unit):
  match count:
    case None{}:
      do IO<Unit>:
        IO.write("{\"tensor\":")
        IO.write(Nat.show(U32.to_nat(index)))
        IO.write(",\"count\":null}\n")
    case Some{count}:
      do IO<Unit>:
        IO.write("{\"tensor\":")
        IO.write(Nat.show(U32.to_nat(index)))
        IO.write(",\"count\":")
        IO.write(Nat.show(U32.to_nat(count)))
        IO.write("}\n")

def emit_tensor_count(index: U32,shape: List<&2,U32>) -> IO(Unit):
  emit_count(index,Tensors.count_of(shape))
