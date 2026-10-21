# The public input gradient of an output gradient and weights of any layout
# that hold the values of their readers. A Ready result has the Normal state
# and holds, at every element, the specification of the readers' values; the
# weights come back holding theirs.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_read.bend as BandRead
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/tensor_convolution.bend as Conv
import ../../lib/storage_buffer.bend as Storage
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/tensor_whole_leaf.bend as WholeLeaf
import ../../lib/proofs/tensor_gather_values.bend as GatherValues
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_band_from_whole.bend as FromWhole
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/word_encoding.bend as Words
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../input_gradient_specification.bend as Specification
import ./input_gradient_sources.bend as Sources
import ./band_window_contents.bend as Contents
import ./input_gradient_call.bend as Call
import ./input_gradient_tiled_images.bend as Images
import ./input_gradient_tiled_run.bend as Run

# The readers of the output gradient and the weights, and the geometry of the call.
type InputRead<-Gradient: Data,-Weights: Data> is Data:
  InputRead{gradient: Gradient,weights: Weights,geometry: Kernel.Geometry}

# One image of the call read from the readers, and the value of the cell at
# an index of the input: the specification at the point the index decodes to.
def problem(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient: Gradient,+weights: Weights,
  +geometry: Kernel.Geometry,+image: Nat) -> Specification.Problem:
  Specification.Problem{Sources.shape(geometry),Contents.image_values(~Gradient,~gradient_value,Sources.gradient_size(geometry),Nat.mul(image,Sources.gradient_size(geometry)),gradient),
    Contents.image_values(~Weights,~weight_value,Sources.weight_size(geometry),0n,weights)}
def input_at(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,read: InputRead<Gradient,Weights>,index: U32) -> F32:
  InputRead{+gradient,+weights,+geometry} = read
  +point = Domain.point(geometry,U32.to_nat(index))
  Specification.value(problem(~Gradient,~gradient_value,~Weights,~weight_value,gradient,weights,geometry,Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point))

def returned_weights(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  (result,weights) = pair
  weights

# An input gradient call: the gradient holds the specification of the
# readers' values, and the weights come back holding theirs.
def InputCall(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +geometry: Kernel.Geometry,result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Data:
  Shape.Both<Global.Contract(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
      Tensors.result_of(~F32,result)),
    Global.Shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),returned_weights(result))>

# A buffer that holds a reader's values lists them on a run of its indices.
def stored(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+content: Buffer.Content<F32>,+values: Contract.Values(~F32,~Logical,~original,logical,content),
  count: Nat,+index: Nat,+room: {Nat.is_le(Nat.add(index,count),Buffer.content_length(~F32,content)) == True{} : Bool}) ->
  {Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,count,index,content) == Contents.image_values(~Logical,~original,count,index,logical) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(F32,Buffer.value(~F32,U32.from_nat(index),content),original(logical,U32.from_nat(index)),
        Contract.read(~F32,~Logical,~original,logical,content,index,values,Order.room_index(index,rest,Buffer.content_length(~F32,content),room))) :
        {_ <> Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,rest,1n+index,content) == Contents.image_values(~Logical,~original,1n+rest,index,logical) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => original(logical,U32.from_nat(index)) <> tail,
        Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,rest,1n+index,content),Contents.image_values(~Logical,~original,rest,1n+index,logical),
        stored(~Logical,~original,logical,content,values,rest,1n+index,Order.advance_room(index,rest,Buffer.content_length(~F32,content),room)))

# The kernel's buffers hold the readers' values for the geometry of the
# readers, and the bands are those of the batch.
type Link<-Gradient: Data,-gradient_value: Gradient -> U32 -> F32,-Weights: Data,-weight_value: Weights -> U32 -> F32,
  -geometry: Kernel.Geometry,-batch: Nat,-gradient: Buffer.Model<F32>,-weights: Buffer.Model<F32>,
  -shape: Kernel.Geometry,-gradient_context: Gradient,-weight_context: Weights,-bands: Bands.Geometry> is Data:
  Link{same: {shape == geometry : Kernel.Geometry},
    elements: {BandGather.elements(bands) == Nat.mul(batch,Domain.image_size(geometry)) : Nat},
    gradient_values: Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.content(~F32,gradient)),
    gradient_room: {Nat.is_le(Nat.mul(batch,Sources.gradient_size(geometry)),Buffer.content_length(~F32,Buffer.content(~F32,gradient))) == True{} : Bool},
    weight_values: Contract.Values(~F32,~Weights,~weight_value,weight_context,Buffer.content(~F32,weights)),
    weight_room: {Nat.is_le(Sources.weight_size(geometry),Buffer.content_length(~F32,Buffer.content(~F32,weights))) == True{} : Bool}}
def Linked(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +setup: Images.Setup,+read: InputRead<Gradient,Weights>,+bands: Bands.Geometry) -> Data:
  Images.Setup{Domain.Call{geometry,batch,limits,facts},gradient,weights} = setup
  InputRead{gradient_context,weight_context,shape} = read
  Link<Gradient,gradient_value,Weights,weight_value,geometry,batch,gradient,weights,shape,gradient_context,weight_context,bands>

# At every index of the bands the kernel's value is the readers' value.
def agrees(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +setup: Images.Setup,+read: InputRead<Gradient,Weights>,+bands: Bands.Geometry,+related: Linked(~Gradient,~gradient_value,~Weights,~weight_value,setup,read,bands),
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(bands)) == True{} : Bool}) ->
  {Images.value(setup,index) == input_at(~Gradient,~gradient_value,~Weights,~weight_value,read,index) : F32}:
  Images.Setup{Domain.Call{+geometry,+batch,+limits,+facts},+gradient,+weights} = setup
  InputRead{+gradient_context,+weight_context,+shape} = read
  Link{+same,+elements,+gradient_values,+gradient_room,+weight_values,+weight_room} = related
  +call = {Domain.Call{geometry,batch,limits,facts} : Domain.Call}
  +point = Domain.point(geometry,U32.to_nat(index))
  +image = Domain.image(point)
  +size = Domain.image_size(geometry)
  +area = Sources.gradient_size(geometry)
  +positive = Order.product_positive(Domain.channels(geometry),Nat.mul(Domain.height(geometry),Domain.width(geometry)),Domain.channels_positive(call),
    Order.product_positive(Domain.height(geometry),Domain.width(geometry),Domain.height_positive(call),Domain.width_positive(call)))
  +below = DivisionOrder.quotient_below(U32.to_nat(index),batch,size,positive,%elements : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} inside)
  +gradient_fits = {
    %Order.commutative(area,Nat.mul(image,area)) : {Nat.is_le(_,Buffer.content_length(~F32,Buffer.content(~F32,gradient))) == True{} : Bool}
    Order.transitive(Nat.mul(1n+image,area),Nat.mul(batch,area),Buffer.content_length(~F32,Buffer.content(~F32,gradient)),
      RowMajor.mul_le(1n+image,batch,area,Order.strict_follow(image,batch,below)),gradient_room) :
    {Nat.is_le(Nat.add(Nat.mul(image,area),area),Buffer.content_length(~F32,Buffer.content(~F32,gradient))) == True{} : Bool}}
  %Equal.sym(Kernel.Geometry,shape,geometry,same) :
    {Images.value(Images.Setup{call,gradient,weights},index) == input_at(~Gradient,~gradient_value,~Weights,~weight_value,InputRead{gradient_context,weight_context,_},index) : F32}
  %Equal.sym(List<&2,F32>,Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,area,Nat.mul(image,area),Buffer.content(~F32,gradient)),
    Contents.image_values(~Gradient,~gradient_value,area,Nat.mul(image,area),gradient_context),
    stored(~Gradient,~gradient_value,gradient_context,Buffer.content(~F32,gradient),gradient_values,area,Nat.mul(image,area),gradient_fits)) :
    {Specification.value(Specification.Problem{Sources.shape(geometry),_,
      Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,Sources.weight_size(geometry),0n,Buffer.content(~F32,weights))},Domain.channel(point),Domain.row(point),Domain.column(point)) ==
      input_at(~Gradient,~gradient_value,~Weights,~weight_value,InputRead{gradient_context,weight_context,geometry},index) : F32}
  %Equal.sym(List<&2,F32>,Contents.image_values(~Buffer.Content<F32>,~Sources.source_value,Sources.weight_size(geometry),0n,Buffer.content(~F32,weights)),
    Contents.image_values(~Weights,~weight_value,Sources.weight_size(geometry),0n,weight_context),
    stored(~Weights,~weight_value,weight_context,Buffer.content(~F32,weights),weight_values,Sources.weight_size(geometry),0n,weight_room)) :
    {Specification.value(Specification.Problem{Sources.shape(geometry),Contents.image_values(~Gradient,~gradient_value,area,Nat.mul(image,area),gradient_context),_},
      Domain.channel(point),Domain.row(point),Domain.column(point)) ==
      input_at(~Gradient,~gradient_value,~Weights,~weight_value,InputRead{gradient_context,weight_context,geometry},index) : F32}
  {==}

# Well formed bands that hold the kernel's values hold the readers' values.
def transported(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +setup: Images.Setup,+read: InputRead<Gradient,Weights>,+bands: Bands.Geometry,+related: Linked(~Gradient,~gradient_value,~Weights,~weight_value,setup,read,bands),
  +content: Model.Content<F32>,+well_formed: BandShape.WellFormed<F32,bands,content>,+values: BandValues.Values(~F32,~Images.Setup,~Images.value,setup,bands,content),
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(BandGather.elements(bands),U32.to_nat(capacity)) == True{} : Bool}) ->
  BandValues.Values(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),read,bands,content):
  BandShape.WellFormed{+metadata,+ready,+coverage} = well_formed
  BandRead.agreed(~F32,~Images.Setup,~InputRead<Gradient,Weights>,~Images.value,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),
    ~Linked(~Gradient,~gradient_value,~Weights,~weight_value),~agrees(~Gradient,~gradient_value,~Weights,~weight_value),
    setup,read,bands,related,content,metadata,values,capacity,capacity_ok,room)

# ---- The call a validated input gradient runs ----

def three(+channels: U32,+height: U32,+width: U32) ->
  {ShapeLaws.elements([channels,height,width]) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),Order.times_one(U32.to_nat(width))) :
    {Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),_)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}
  {==}
def shape_count(+mode: Call.Mode,+geometry: Kernel.Geometry) ->
  {ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry))) == Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)) : Nat}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      Equal.trans(Nat,ShapeLaws.elements([c,h,w]),Domain.image_size(geometry),Nat.mul(1n,Domain.image_size(geometry)),three(c,h,w),
        Equal.sym(Nat,Nat.add(Domain.image_size(geometry),0n),Domain.image_size(geometry),Addition.zero_right(Domain.image_size(geometry))))
    case Call.Batched{+batch}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(batch),value),ShapeLaws.elements([c,h,w]),Domain.image_size(geometry),three(c,h,w))

# The element count of the input is its batch of images.
def counted_exact(+mode: Call.Mode,+geometry: Kernel.Geometry,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>}) ->
  {Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)) == U32.to_nat(count) : Nat}:
  Equal.trans(Nat,Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)),ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry))),U32.to_nat(count),
    Equal.sym(Nat,ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry))),Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)),shape_count(mode,geometry)),
    Count.count_exact(ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry))),count,Count.count_of(Gradient.input_shape(Call.arguments(mode,geometry)),count,counted)))
def count_bounded(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  bounded
def machine_room(+mode: Call.Mode,+geometry: Kernel.Geometry,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +limits: Machine.Limits,+limits_equal: {limits == Default.limits() : Machine.Limits}) ->
  {Nat.is_le(Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Call.batch(Call.arguments(mode,geometry)),Domain.image_size(geometry)),U32.to_nat(count),counted_exact(mode,geometry,count,counted)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Bounds.as_nat_le(count,Machine.capacity(limits),
    %Equal.sym(Machine.Limits,limits,Default.limits(),limits_equal) : {U32.is_le(count,Machine.capacity(_)) == True{} : Bool}
    count_bounded(ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry))),count,Count.count_of(Gradient.input_shape(Call.arguments(mode,geometry)),count,counted)))

def well_formed_ready(+shape: List<&2,U32>,+content: Buffer.Content<F32>,well_formed: Contract.WellFormed<F32,shape,content>) -> Buffer.ContentReady<F32,content>:
  Contract.WellFormed{length_exact,ready} = well_formed
  ready

# The kernel call of a validated input gradient.
def validated_run(+mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +limits: Machine.Limits,+limits_equal: {limits == Default.limits() : Machine.Limits},
  +valid: {Gradient.input_valid_with(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits) == True{} : Bool},
  +ready: Buffer.ContentReady<F32,Buffer.content(~F32,gradient)>) -> Run.Run:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +arguments = Call.arguments(mode,shape)
  +setup = {Images.Setup{Domain.Call{shape,Call.batch(arguments),limits,Call.facts(arguments,gradient,weights,limits,limits_equal,valid,ready)},gradient,weights} : Images.Setup}
  Run.Run{setup,Geometry.word_envelope(),Call.validated_condition(arguments,Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits,0n,valid),workers,limits,
    machine_room(mode,shape,count,counted,limits,limits_equal)}

# The kernel's buffers hold the readers' values for the bands of the input.
def validated_link(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +limits: Machine.Limits,
  +valid: {Gradient.input_valid_with(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits) == True{} : Bool}) ->
  Link<Gradient,gradient_value,Weights,weight_value,geometry,Call.batch(Call.arguments(mode,geometry)),gradient,weights,geometry,gradient_context,weight_context,
    Gradient.input_band_geometry(Call.arguments(mode,geometry))>:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Buffer.Model{+dt,+dl,+dc} = gradient
  Buffer.Model{+wt,+wl,+wc} = weights
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +arguments = Call.arguments(mode,shape)
  +gradient_length = Order.equal_of_is_eq(dl,Nat.mul(Call.batch(arguments),Sources.gradient_size(shape)),Call.validated_condition(arguments,dl,wl,limits,1n,valid))
  +weight_length = Order.equal_of_is_eq(wl,Sources.weight_size(shape),Call.valid_condition(arguments,dl,wl,limits,1n,valid))
  Link{{==},{==},Shape.second(Contract.WellFormed<F32,Call.input_shape(mode,shape),Buffer.Content{dt,dl}>,Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.Content{dt,dl}),d_facts),
    %gradient_length : {Nat.is_le(_,dl) == True{} : Bool} Order.reflexive(dl),
    Shape.second(Contract.WellFormed<F32,Call.weight_shape(shape),Buffer.Content{wt,wl}>,Contract.Values(~F32,~Weights,~weight_value,weight_context,Buffer.Content{wt,wl}),w_facts),
    %weight_length : {Nat.is_le(_,wl) == True{} : Bool} Order.reflexive(wl)}

# ---- The result of a validated input gradient ----

# The bands of an outcome that hold the readers' values make the result, and
# the weights the outcome gives back are the weights it read.
def outcome_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+workers: U32,+weight_fill: F32,+weight_workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +weights: Buffer.Model<F32>,-contents: Model.Content<F32>,
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +evidence: Shape.Both<BandValues.Values(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
      Gradient.input_band_geometry(Call.arguments(mode,geometry)),contents),
    BandShape.WellFormed<F32,Gradient.input_band_geometry(Call.arguments(mode,geometry)),contents>>,
  outcome: Kernel.Outcome,+matched: Run.Outcome(Buffer.pack(~F32,weights),contents,outcome)) ->
  InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_result(Call.arguments(mode,geometry),workers,Call.weight_shape(geometry),weight_fill,weight_workers,outcome)):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  match outcome:
    case Kernel.Outcome{out_weights,output}:
      Shape.Both{+same_weights,+same_contents} = matched
      Shape.Both{
        Global.contract(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
          Gradient.input_band_geometry(Call.arguments(mode,geometry)),Gradient.input_shape(Call.arguments(mode,geometry)),0.0,workers,output,
          Global.Counted{count,counted},shape_count(mode,geometry),
          %Equal.sym(Model.Content<F32>,Model.contents(~F32,output),contents,same_contents) :
            Shape.Both<BandValues.Values(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
              Gradient.input_band_geometry(Call.arguments(mode,geometry)),_),BandShape.WellFormed<F32,Gradient.input_band_geometry(Call.arguments(mode,geometry)),_>>
          evidence),
        %Equal.sym(Storage.Buffer<F32>,out_weights,Buffer.pack(~F32,weights),same_weights) :
          Global.Shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
            Tensors.whole(~F32,_,Call.weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers))
        WholeLeaf.held_shaped(~F32,~Weights,~weight_value,weight_context,weights,Call.weight_shape(geometry),weight_fill,weight_workers,w_facts)}

# A batch of at least one image: the kernel makes the bands of the batch.
def nonempty_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+weight_fill: F32,+weight_workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +limits: Machine.Limits,+limits_equal: {limits == Default.limits() : Machine.Limits},
  +valid: {Gradient.input_valid_with(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits) == True{} : Bool},
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +rest: Nat,+nonempty: {Call.batch(Call.arguments(mode,geometry)) == 1n+rest : Nat}) ->
  InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_result(Call.arguments(mode,geometry),workers,Call.weight_shape(geometry),weight_fill,weight_workers,
      Kernel.run(geometry,U32.to_nat(count),U32.max(workers,U32.max(weight_workers,1)),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)))):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +run_workers = U32.max(workers,U32.max(weight_workers,1))
  +read = {InputRead{gradient_context,weight_context,geometry} : InputRead<Gradient,Weights>}
  +run = validated_run(mode,geometry,gradient,weights,run_workers,count,counted,limits,limits_equal,valid,
    well_formed_ready(Call.input_shape(mode,geometry),Buffer.content(~F32,gradient),
      Shape.first(Contract.WellFormed<F32,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)>,
        Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.content(~F32,gradient)),d_facts)))
  +facts = {%Arithmetic.native_roundtrip(Call.images(mode)) :
    Shape.Both<BandValues.Values(~F32,~Images.Setup,~Images.value,Run.run_setup(run),Bands.Geometry{_,c,h,w},
        Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))),
      BandShape.WellFormed<F32,Bands.Geometry{_,c,h,w},Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))>>
    Images.batch_facts(Run.run_batch(run),rest,nonempty) :
    Shape.Both<BandValues.Values(~F32,~Images.Setup,~Images.value,Run.run_setup(run),Bands.Geometry{Call.images(mode),c,h,w},
        Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))),
      BandShape.WellFormed<F32,Bands.Geometry{Call.images(mode),c,h,w},Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))>>}
  +well_formed = Shape.second(BandValues.Values(~F32,~Images.Setup,~Images.value,Run.run_setup(run),Bands.Geometry{Call.images(mode),c,h,w},
      Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))),
    BandShape.WellFormed<F32,Bands.Geometry{Call.images(mode),c,h,w},Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))>,facts)
  +values = transported(~Gradient,~gradient_value,~Weights,~weight_value,Run.run_setup(run),read,Bands.Geometry{Call.images(mode),c,h,w},
    validated_link(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,gradient,weights,d_facts,w_facts,limits,valid),
    Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run))),well_formed,
    Shape.first(BandValues.Values(~F32,~Images.Setup,~Images.value,Run.run_setup(run),Bands.Geometry{Call.images(mode),c,h,w},
        Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))),
      BandShape.WellFormed<F32,Bands.Geometry{Call.images(mode),c,h,w},Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run)))>,facts),
    Machine.capacity(limits),Limits.capacity_fits(limits),machine_room(mode,geometry,count,counted,limits,limits_equal))
  +matched = {%counted_exact(mode,geometry,count,counted) :
    Run.Outcome(Buffer.pack(~F32,weights),Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run))),
      Kernel.run(geometry,_,run_workers,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)))
    Run.run_matches(run,rest,nonempty) :
    Run.Outcome(Buffer.pack(~F32,weights),Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run))),
      Kernel.run(geometry,U32.to_nat(count),run_workers,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)))}
  outcome_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,workers,weight_fill,weight_workers,count,counted,
    weights,Model.planned(~F32,~Images.batch_part(Run.run_batch(run)),Images.batch_plan(Run.run_batch(run))),w_facts,Shape.Both{values,well_formed},
    Kernel.run(geometry,U32.to_nat(count),run_workers,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)),matched)

def leaf_metadata(+geometry: Bands.Geometry,+first: U32,+rows: U32,+capacity: U32,+division_capacity: U32,facts: FromWhole.LeafFacts<geometry,first,rows,capacity,division_capacity>) ->
  BandShape.Leaf(geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows))):
  FromWhole.LeafFacts{allocation,contents,metadata} = facts
  metadata

# No image: one band of no rows, which holds any values.
def empty_bands(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +empty: {Call.batch(Call.arguments(mode,geometry)) == 0n : Nat}) ->
  Global.Contract(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),InputRead{gradient_context,weight_context,geometry},
    Tensors.Tensor{Partition.Leaf{Tensors.buffer(~F32,0.0,0),0,0,Unit{}},Gradient.input_band_geometry(Call.arguments(mode,geometry)),Gradient.input_shape(Call.arguments(mode,geometry)),
      0.0,Tensors.Usable{},workers}):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +bands = Gradient.input_band_geometry(Call.arguments(mode,geometry))
  +read = {InputRead{gradient_context,weight_context,geometry} : InputRead<Gradient,Weights>}
  +tree = Tree.fresh(F32,Tensors.depth_for(0),0.0)
  +metadata = {%Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(bands,0),0,FromWhole.empty_count(bands,0,{==})) : BandShape.Leaf(bands,0,0,_)
    leaf_metadata(bands,0,0,0,0,FromWhole.empty_leaf(bands,0,0,0,0,{==},Order.zero_le(Inspection.total_rows(bands)))) : BandShape.Leaf(bands,0,0,0n)}
  +ready = Buffer.content_ready(~F32,Buffer.fresh(~F32,Tensors.depth_for(0),0.0,0n),Buffer.fresh_ready(~F32,Tensors.depth_for(0),0.0,0n,{==},{==}))
  +coverage = {BandPlan.LeafCoverage{{==},%Equal.sym(Nat,U32.to_nat(Call.images(mode)),0n,empty) : {U32.to_nat(0) == Nat.mul(_,U32.to_nat(h)) : Nat} {==}} :
    BandPlan.LeafCoverage<0,0,0n,Inspection.total_rows(bands)>}
  +values = BandValues.empty_leaf(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),read,bands,0,0,tree,0n,
    RowMajor.multiply_zero_right(U32.to_nat(c)))
  Global.contract(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),read,bands,Gradient.input_shape(Call.arguments(mode,geometry)),0.0,workers,
    Partition.Leaf{Tensors.buffer(~F32,0.0,0),0,0,Unit{}},Global.Counted{count,counted},shape_count(mode,geometry),
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Tensors.buffer(~F32,0.0,0)),Buffer.Content{tree,0n},Contract.empty_contents(~F32,0.0)) :
      Shape.Both<BandValues.Values(~F32,~InputRead<Gradient,Weights>,~input_at(~Gradient,~gradient_value,~Weights,~weight_value),read,bands,Model.RowBand{_,0,0}),
        BandShape.WellFormed<F32,bands,Model.RowBand{_,0,0}>>
    Shape.Both{values,BandShape.WellFormed{metadata,ready,coverage}})

# A batch of no image: the kernel gives one band of no rows.
def empty_contract(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+weight_fill: F32,+weight_workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +limits: Machine.Limits,+limits_equal: {limits == Default.limits() : Machine.Limits},
  +valid: {Gradient.input_valid_with(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits) == True{} : Bool},
  +ready: Buffer.ContentReady<F32,Buffer.content(~F32,gradient)>,
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +empty: {Call.batch(Call.arguments(mode,geometry)) == 0n : Nat}) ->
  InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_result(Call.arguments(mode,geometry),workers,Call.weight_shape(geometry),weight_fill,weight_workers,
      Kernel.run(geometry,U32.to_nat(count),U32.max(workers,U32.max(weight_workers,1)),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)))):
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +run_workers = U32.max(workers,U32.max(weight_workers,1))
  +arguments = Call.arguments(mode,geometry)
  +call = {Domain.Call{geometry,Call.batch(arguments),limits,Call.facts(arguments,gradient,weights,limits,limits_equal,valid,ready)} : Domain.Call}
  +quotient = Equal.trans(Nat,Nat.div(U32.to_nat(count),U32.to_nat((c * h * w : U32))),Call.batch(arguments),0n,
    Domain.run_images(call,U32.to_nat(count),Equal.sym(Nat,Nat.mul(Call.batch(arguments),Domain.image_size(geometry)),U32.to_nat(count),counted_exact(mode,geometry,count,counted))),empty)
  %Equal.sym(Nat,Nat.div(U32.to_nat(count),U32.to_nat((c * h * w : U32))),0n,quotient) :
    InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
      Gradient.input_result(arguments,workers,Call.weight_shape(geometry),weight_fill,weight_workers,
        Kernel.run_selected(Nat.is_eq(_,0n),geometry,_,run_workers,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights))))
  Shape.Both{empty_bands(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,workers,count,counted,empty),
    WholeLeaf.held_shaped(~F32,~Weights,~weight_value,weight_context,weights,Call.weight_shape(geometry),weight_fill,weight_workers,w_facts)}

# ---- The public input gradient ----

def InputHolds(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +geometry: Kernel.Geometry,result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Type:
  @accepted: {Call.ready(result) == True{} : Bool} -> InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,result)

# The batch has no image or at least one.
def images_selected(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,batch: Nat,
  +gradient_context: Gradient,+weight_context: Weights,+mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +workers: U32,+weight_fill: F32,+weight_workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +valid: {Gradient.input_valid(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights)) == True{} : Bool},
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +equation: {Call.batch(Call.arguments(mode,geometry)) == batch : Nat}) ->
  InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_result(Call.arguments(mode,geometry),workers,Call.weight_shape(geometry),weight_fill,weight_workers,
      Kernel.run(geometry,U32.to_nat(count),U32.max(workers,U32.max(weight_workers,1)),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights)))):
  match batch:
    case 0n:
      empty_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,gradient,weights,workers,weight_fill,weight_workers,count,counted,
        Default.limits(),{==},valid,
        well_formed_ready(Call.input_shape(mode,geometry),Buffer.content(~F32,gradient),
          Shape.first(Contract.WellFormed<F32,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)>,
            Contract.Values(~F32,~Gradient,~gradient_value,gradient_context,Buffer.content(~F32,gradient)),d_facts)),w_facts,equation)
    case 1n+ +rest:
      nonempty_contract(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,gradient,weights,workers,weight_fill,weight_workers,count,counted,
        Default.limits(),{==},valid,d_facts,w_facts,rest,equation)

# The validation either rejects the call or runs the kernel.
def valid_selected(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,accepted: Bool,
  +gradient_context: Gradient,+weight_context: Weights,+mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +workers: U32,+weight_fill: F32,+weight_workers: U32,+count: U32,
  +counted: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == Some{count} : Maybe<&2,U32>},
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +compared: {Gradient.input_valid(Call.arguments(mode,geometry),Buffer.length(~F32,gradient),Buffer.length(~F32,weights)) == accepted : Bool}) ->
  InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_checked(accepted,Call.arguments(mode,geometry),count,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,Call.weight_shape(geometry),weight_fill,weight_workers)):
  match accepted:
    case False{}:
      ready => Empty.absurd(InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
        Gradient.input_checked(False{},Call.arguments(mode,geometry),count,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,Call.weight_shape(geometry),
          weight_fill,weight_workers)),Words.false_true(ready))
    case True{}:
      ready => images_selected(~Gradient,~gradient_value,~Weights,~weight_value,Call.batch(Call.arguments(mode,geometry)),gradient_context,weight_context,mode,geometry,gradient,weights,
        workers,weight_fill,weight_workers,count,counted,compared,d_facts,w_facts,{==})

# The shape either has no count or runs the validation.
def counted_selected(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,result: Maybe<&2,U32>,
  +gradient_context: Gradient,+weight_context: Weights,+mode: Call.Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +workers: U32,+weight_fill: F32,+weight_workers: U32,
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights)),
  +equation: {Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))) == result : Maybe<&2,U32>}) ->
  InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_count(result,Call.arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,Call.weight_shape(geometry),weight_fill,weight_workers)):
  match result:
    case None{}:
      ready => Empty.absurd(InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
        Gradient.input_count(None{},Call.arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,Call.weight_shape(geometry),
          weight_fill,weight_workers)),Words.false_true(ready))
    case Some{+count}:
      Buffer.Model{+dt,+dl,+dc} = gradient
      Buffer.Model{+wt,+wl,+wc} = weights
      valid_selected(~Gradient,~gradient_value,~Weights,~weight_value,Gradient.input_valid(Call.arguments(mode,geometry),dl,wl),gradient_context,weight_context,mode,geometry,
        Buffer.Model{dt,dl,dc},Buffer.Model{wt,wl,wc},workers,weight_fill,weight_workers,count,equation,d_facts,w_facts,{==})

# Both operands arrived: the shapes parse to the call of the geometry.
def input_stored(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,
  +gradient: Buffer.Model<F32>,+d_fill: F32,+d_workers: U32,
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +weights: Buffer.Model<F32>,+w_fill: F32,+w_workers: U32,
  +w_facts: GatherValues.Facts(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Buffer.content(~F32,weights))) ->
  InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input_status(Tensors.first_failure(Tensors.Usable{},Tensors.Usable{}),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
      Buffer.pack(~F32,gradient),Call.input_shape(mode,geometry),d_fill,d_workers,Buffer.pack(~F32,weights),Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers)):
  +parsed = {%Equal.sym(Maybe<&2,Conv.InputShape>,Conv.input_shape(Call.input_shape(mode,geometry)),Some{Call.parsed_shape(mode,geometry)},Call.parse(mode,geometry)) :
    {Gradient.input_geometry(Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),_,Call.weight_shape(geometry),
      Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),d_workers,w_fill,w_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))),Call.arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),
        d_workers,Call.weight_shape(geometry),w_fill,w_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
    Call.parsed_call(mode,geometry,gradient,weights,d_workers,w_fill,w_workers) :
    {Gradient.input_status(Tensors.first_failure(Tensors.Usable{},Tensors.Usable{}),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
      Buffer.pack(~F32,gradient),Call.input_shape(mode,geometry),d_fill,d_workers,Buffer.pack(~F32,weights),Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))),Call.arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),
        d_workers,Call.weight_shape(geometry),w_fill,w_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}}
  %Equal.sym(Tensors.Tensor<F32> & Tensors.Tensor<F32>,
    Gradient.input_status(Tensors.first_failure(Tensors.Usable{},Tensors.Usable{}),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
      Buffer.pack(~F32,gradient),Call.input_shape(mode,geometry),d_fill,d_workers,Buffer.pack(~F32,weights),Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers),
    Gradient.input_count(Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))),Call.arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),
      d_workers,Call.weight_shape(geometry),w_fill,w_workers),parsed) :
    InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,_)
  counted_selected(~Gradient,~gradient_value,~Weights,~weight_value,Tensors.count_of(Gradient.input_shape(Call.arguments(mode,geometry))),gradient_context,weight_context,mode,geometry,
    gradient,weights,d_workers,w_fill,w_workers,d_facts,w_facts,{==})

# The output gradient arrived: the weights arrive in storage.
def input_weights(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,
  +gradient: Buffer.Model<F32>,+d_fill: F32,+d_workers: U32,
  +d_facts: GatherValues.Facts(~F32,~Gradient,~gradient_value,gradient_context,Call.input_shape(mode,geometry),Buffer.content(~F32,gradient)),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers})) ->
  InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      weights => weight_shape => weight_fill => weight_status => weight_workers =>
        +weight_status = weight_status
        Gradient.input_status(Tensors.first_failure(Tensors.Usable{},weight_status),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
          Buffer.pack(~F32,gradient),Call.input_shape(mode,geometry),d_fill,d_workers,weights,weight_shape,weight_fill,weight_status,weight_workers))):
  WholeLeaf.with_whole(~F32,~Weights,~weight_value,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),
    ~(result => InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,result)),
    ~(weights => weight_shape => weight_fill => weight_status => weight_workers =>
      +weight_status = weight_status
      Gradient.input_status(Tensors.first_failure(Tensors.Usable{},weight_status),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
        Buffer.pack(~F32,gradient),Call.input_shape(mode,geometry),d_fill,d_workers,weights,weight_shape,weight_fill,weight_status,weight_workers)),
    weight_context,w_bands,w_geometry,Call.weight_shape(geometry),w_fill,w_workers,w_contract,
    weights => w_facts => input_stored(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,gradient,d_fill,d_workers,d_facts,
      weights,w_fill,w_workers,w_facts))

# The input gradient of an output gradient and weights of any layout. A
# Ready result is Normal and holds, at every element, the specification of
# the readers' values.
def input(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,+gradient_context: Gradient,+weight_context: Weights,
  +mode: Call.Mode,+geometry: Kernel.Geometry,
  +d_bands: Model.Model<F32>,+d_geometry: Bands.Geometry,+d_fill: F32,+d_workers: U32,
  +d_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,Call.input_shape(mode,geometry),d_fill,Tensors.Usable{},d_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  accepted: {Call.ready(Gradient.input(Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,Call.input_shape(mode,geometry),d_fill,Tensors.Usable{},d_workers},
    Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
    Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))) == True{} : Bool}) ->
  InputCall(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,
    Gradient.input(Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,Call.input_shape(mode,geometry),d_fill,Tensors.Usable{},d_workers},
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))):
  holds = WholeLeaf.with_whole(~F32,~Gradient,~gradient_value,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),
    ~(result => InputHolds(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,geometry,result)),
    ~(gradient => d_shape => d_fill => d_status => workers =>
      Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),
        Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
        weights => weight_shape => weight_fill => weight_status => weight_workers =>
          +weight_status = weight_status
          Gradient.input_status(Tensors.first_failure(d_status,weight_status),Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry),
            gradient,d_shape,d_fill,workers,weights,weight_shape,weight_fill,weight_status,weight_workers))),
    gradient_context,d_bands,d_geometry,Call.input_shape(mode,geometry),d_fill,d_workers,d_contract,
    gradient => d_facts => input_weights(~Gradient,~gradient_value,~Weights,~weight_value,gradient_context,weight_context,mode,geometry,gradient,d_fill,d_workers,d_facts,
      w_bands,w_geometry,w_fill,w_workers,w_contract))
  holds(accepted)
