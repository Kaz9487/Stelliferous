# Train.forward, Train.evaluate, Train.backward and Train.sgd_step_with of an
# activation, the average pool, a batch normalization and a layer
# normalization: each result holds what the public theorem of its operation
# states, and the layer comes back as its description states.
import Base
import ../math_activation.bend as Activations
import ../math_sum.bend as Sum
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_tiled.bend as Tiled
import ../tensor_training.bend as Train
import ../tensor_optimizer.bend as Optimizer
import ../tensor_batch_norm.bend as Norm
import ../tensor_layer_norm.bend as LineNorm
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_tiled_public.bend as Public
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_partials_values.bend as PartialsValues
import ./tensor_reduce_values.bend as Reduce
import ./tensor_tiled_sums.bend as Sums
import ./tensor_tiled_reshape.bend as TiledReshape
import ./tensor_tiled_view.bend as TiledView
import ./tensor_view_gather.bend as ViewGather
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_function_values.bend as FunctionValues
import ./tensor_optimizer_values.bend as OptimizerValues
import ./tensor_batch_norm_public.bend as NormPublic
import ./tensor_layer_norm_values.bend as LineValues
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./word_encoding.bend as Words

# ---- Results ----

# A result of forward: the output, what backward reads and the layer.
def Forwarded(~Output: Tiled.Tiled -> Data,~Saved: Train.Saved -> Data,~Layer: Train.Layer -> Data,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  (saved,layer) = rest
  KernelGemm.Both<Output(output),KernelGemm.Both<Saved(saved),Layer(layer)>>

# A result of evaluate or backward: the output and the layer.
def Answered(~Output: Tiled.Tiled -> Data,~Layer: Train.Layer -> Data,result: Tiled.Tiled & Train.Layer) -> Data:
  (output,layer) = result
  KernelGemm.Both<Output(output),Layer(layer)>

# Train.recorded of a result of forward goes on with its parts.
def recorded_holds(~Output: Tiled.Tiled -> Data,~Saved: Train.Saved -> Data,~Layer: Train.Layer -> Data,~Goal: Tiled.Tiled & Train.Tape -> Data,
  ~next: Tiled.Tiled -> Train.Saved -> Train.Layer -> Tiled.Tiled & Train.Tape,result: Tiled.Tiled & (Train.Saved & Train.Layer),+held: Forwarded(~Output,~Saved,~Layer,result),
  body: @output: Tiled.Tiled -> @saved: Train.Saved -> @layer: Train.Layer -> @+output_held: Output(output) -> @+saved_held: Saved(saved) -> @+layer_held: Layer(layer) ->
    Goal(next(output,saved,layer))) -> Goal(Train.recorded(result,next)):
  (output,rest) = result
  (saved,layer) = rest
  KernelGemm.Both{+output_held,KernelGemm.Both{+saved_held,+layer_held}} = held
  body(output,saved,layer,output_held,saved_held,layer_held)

# Train.returned of a result of evaluate or backward goes on with its parts.
def returned_holds(~Output: Tiled.Tiled -> Data,~Layer: Train.Layer -> Data,~Goal: Tiled.Tiled & Train.Network -> Data,
  ~next: Tiled.Tiled -> Train.Layer -> Tiled.Tiled & Train.Network,result: Tiled.Tiled & Train.Layer,+held: Answered(~Output,~Layer,result),
  body: @output: Tiled.Tiled -> @layer: Train.Layer -> @+output_held: Output(output) -> @+layer_held: Layer(layer) -> Goal(next(output,layer))) ->
  Goal(Train.returned(result,next)):
  (output,layer) = result
  KernelGemm.Both{+output_held,+layer_held} = held
  body(output,layer,output_held,layer_held)

# Four axes fit every machine.
def four_room(+limits: Machine.Limits) -> {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(4n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))

# ---- Activation ----

def ActivationLayer(+activation: Activations.Activation,layer: Train.Layer) -> Data:
  {layer == Train.Activation{activation} : Train.Layer}

# backward reads the input of the activation.
def ActivationSaved(~X: Data,~x_value: X -> U32 -> F32,+x: X,+shape: List<&2,U32>,saved: Train.Saved) -> Data:
  match saved:
    case Train.Activated{input}: Public.Contract(~X,~x_value,x,shape,input)
    case saved: {False{} == True{} : Bool}

# The output holds the activation of x, backward reads x, and the layer comes
# back.
def ActivationForward(~X: Data,~x_value: X -> U32 -> F32,+activation: Activations.Activation,+x: X,+shape: List<&2,U32>,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  Forwarded(~(output => FunctionValues.Activated(~X,~x_value,activation,x,shape,output)),~(saved => ActivationSaved(~X,~x_value,x,shape,saved)),
    ~(layer => ActivationLayer(activation,layer)),result)

def activated_holds(~X: Data,~x_value: X -> U32 -> F32,+activation: Activations.Activation,+x: X,+shape: List<&2,U32>,pair: Tiled.Tiled & Tiled.Tiled,
  +held: FunctionValues.ActivatedForward(~X,~x_value,activation,x,shape,pair)) -> ActivationForward(~X,~x_value,activation,x,shape,Train.activated(activation,pair)):
  (output,input) = pair
  KernelGemm.Both{+output_held,+input_contract} = held
  KernelGemm.Both{output_held,KernelGemm.Both{input_contract,{==}}}

# forward of an activation layer with a built-in activation on x of the
# contract.
def activation_forward(~X: Data,~x_value: X -> U32 -> F32,+activation: Activations.Activation,+x: X,+shape: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~X,~x_value,x,shape,tiled),layer: Train.Layer,+described: ActivationLayer(activation,layer),
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) -> ActivationForward(~X,~x_value,activation,x,shape,Train.forward(tiled,layer)):
  %Equal.sym(Train.Layer,layer,Train.Activation{activation},described) : ActivationForward(~X,~x_value,activation,x,shape,Train.forward(tiled,_))
  Public.by_packed(~X,~x_value,(that => ActivationForward(~X,~x_value,activation,x,shape,Train.forward(that,Train.Activation{activation}))),x,shape,tiled,contract,
    xm => xs => xf => xw => xa => xl => xp => packed => activated_holds(~X,~x_value,activation,x,shape,F.activate_forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},activation),
      FunctionValues.activate_forward(~X,~x_value,x,activation,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},packed,built)))

# evaluate of an activation layer: the output holds the activation of x.
def activation_evaluate(~X: Data,~x_value: X -> U32 -> F32,+activation: Activations.Activation,+x: X,+shape: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~X,~x_value,x,shape,tiled),layer: Train.Layer,+described: ActivationLayer(activation,layer),
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Answered(~(output => FunctionValues.Activated(~X,~x_value,activation,x,shape,output)),~(layer => ActivationLayer(activation,layer)),Train.evaluate(tiled,layer)):
  %Equal.sym(Train.Layer,layer,Train.Activation{activation},described) :
    Answered(~(output => FunctionValues.Activated(~X,~x_value,activation,x,shape,output)),~(layer => ActivationLayer(activation,layer)),Train.evaluate(tiled,_))
  KernelGemm.Both{FunctionValues.activate(~X,~x_value,x,activation,shape,tiled,contract,built),{==}}

# backward of an activation layer: the output holds dy times the derivative
# at the saved input.
def activation_backward(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+activation: Activations.Activation,+dy: DY,+x: X,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+shape: List<&2,U32>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool},gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,shape,gradient),
  saved: Train.Saved,+kept: ActivationSaved(~X,~x_value,x,shape,saved),layer: Train.Layer,+described: ActivationLayer(activation,layer)) ->
  Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
    Train.backward(gradient,saved,layer)):
  match saved:
    case Train.Activated{input}:
      %Equal.sym(Train.Layer,layer,Train.Activation{activation},described) :
        Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
          Train.backward(gradient,Train.Activated{input},_))
      KernelGemm.Both{FunctionValues.activate_backward(~DY,~dy_value,~X,~x_value,activation,dy,x,limits,default_capacity,shape,nonempty,rank_room,built,gradient,dy_contract,
        input,kept),{==}}
    case Train.Convolved{other}:
      Empty.absurd(Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
        Train.backward(gradient,Train.Convolved{other},layer)),Words.false_true(kept))
    case Train.Normalized{other}:
      Empty.absurd(Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
        Train.backward(gradient,Train.Normalized{other},layer)),Words.false_true(kept))
    case Train.LineNormalized{other}:
      Empty.absurd(Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
        Train.backward(gradient,Train.LineNormalized{other},layer)),Words.false_true(kept))
    case Train.Pooled{other}:
      Empty.absurd(Answered(~(output => FunctionValues.ActivatedBackward(~DY,~dy_value,~X,~x_value,activation,dy,x,shape,output)),~(layer => ActivationLayer(activation,layer)),
        Train.backward(gradient,Train.Pooled{other},layer)),Words.false_true(kept))

# ---- Average pool ----

def PoolLayer(layer: Train.Layer) -> Data:
  {layer == Train.AveragePool{} : Train.Layer}

# The kept extents of the mean over H and W.
def pool_kept(+images: U32,+channels: U32,+height: U32,+width: U32) -> List<&2,U32>:
  Layout.extents_at([images,channels,height,width],Reduce.kept_axes([images,channels,height,width],[2,3]))

# The mean of each channel of each image over H and W.
def PoolMean() -> Data:
  PiecesMap.Context<F32,PartialsValues.Box<Sum.Parts>>

def pool_mean_value() -> PoolMean() -> U32 -> F32:
  PiecesMap.value(~F32,~F32,~PartialsValues.Box<Sum.Parts>,~F.scalar_value(~F32.div),~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded))

def pool_context(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32) -> PoolMean():
  PiecesMap.Context{U32.to_f32(Tensors.count_or_zero(Layout.extents_at([images,channels,height,width],
    Layout.axes_where(List.length(&2,U32,[images,channels,height,width]),0n,Layout.naturals([2,3]),True{})))),
    Sums.sum_box(~X,~x_value,x,[images,channels,height,width],[2,3])}

def PoolOutput(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,output: Tiled.Tiled) -> Data:
  Public.Contract(~PoolMean(),~pool_mean_value(),pool_context(~X,~x_value,x,images,channels,height,width),[images,channels,1,1],output)

# The pool of [N, C, H, W]: the mean of each channel of each image fits the
# machine and reshapes to [N, C, 1, 1].
type PoolFacts<-limits: Machine.Limits,-images: U32,-channels: U32,-height: U32,-width: U32> is Data:
  PoolFacts{nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},cap: U32,
    cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,channels,height,width])),4n),U32.to_nat(cap)) == True{} : Bool},
    kept_nonzero: {Shapes.has_zero(pool_kept(images,channels,height,width)) == False{} : Bool},
    same: {Tensors.count_of([images,channels,1,1]) == Tensors.count_of(pool_kept(images,channels,height,width)) : Maybe<&2,U32>},
    kept_rank: {Nat.is_le(List.length(&2,U32,pool_kept(images,channels,height,width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The pool of x, as the cells of a model, holds the means of its channels.
def pool_packed(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})) ->
  PoolOutput(~X,~x_value,x,images,channels,height,width,F.reshape(F.mean_axes(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[2,3]),[images,channels,1,1])):
  PoolFacts{+nonempty,+cap,+cap_ok,+count_room,+kept_nonzero,+same,+kept_rank} = facts
  TiledReshape.reshaped(~PoolMean(),~pool_mean_value(),pool_context(~X,~x_value,x,images,channels,height,width),limits,pool_kept(images,channels,height,width),
    [images,channels,1,1],F.mean_axes(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[2,3]),kept_nonzero,same,kept_rank,four_room(limits),default_capacity,
    Sums.mean_axes(~X,~x_value,x,limits,[2,3],[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},x_packed,{==},{==},default_capacity,four_room(limits),nonempty,cap,cap_ok,count_room))

# The output holds the mean of each channel of each image as [N, C, 1, 1],
# backward reads [N, C, H, W], and the layer comes back.
def PoolForward(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  Forwarded(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(saved => {saved == Train.Pooled{[images,channels,height,width]} : Train.Saved}),
    ~(layer => PoolLayer(layer)),result)

def pool_forward_model(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})) ->
  PoolForward(~X,~x_value,x,images,channels,height,width,Train.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Train.AveragePool{})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = x_packed
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),[images,channels,height,width],restored) :
    PoolForward(~X,~x_value,x,images,channels,height,width,
      (F.reshape(F.mean_axes(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[2,3]),Train.pooled_extents(_)),(Train.Pooled{_},Train.AveragePool{})))
  KernelGemm.Both{pool_packed(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,xm,xs,xf,xw,xa,xl,xp,
      KernelGemm.Both{global,Public.Laid{inside,shaped,restored,counted,pinned}}),KernelGemm.Both{{==},{==}}}

def pool_forward_packed(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  -tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),+packed: Public.Packed<tiled>) ->
  PoolForward(~X,~x_value,x,images,channels,height,width,Train.forward(tiled,Train.AveragePool{})):
  Public.Packed{+xm,+xs,+xf,+xw,+xa,+xl,+xp,+equation} = packed
  %equation : PoolForward(~X,~x_value,x,images,channels,height,width,Train.forward(_,Train.AveragePool{}))
  pool_forward_model(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,xm,xs,xf,xw,xa,xl,xp,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},tiled,equation) : Public.Contract(~X,~x_value,x,[images,channels,height,width],_) contract :
      Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})})

# forward of the pool on x [N, C, H, W] of the contract.
def pool_forward(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),layer: Train.Layer,+described: PoolLayer(layer)) ->
  PoolForward(~X,~x_value,x,images,channels,height,width,Train.forward(tiled,layer)):
  %Equal.sym(Train.Layer,layer,Train.AveragePool{},described) : PoolForward(~X,~x_value,x,images,channels,height,width,Train.forward(tiled,_))
  pool_forward_packed(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,tiled,contract,
    Public.packed(~X,~x_value,x,[images,channels,height,width],tiled,contract))

def pool_evaluate_model(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})) ->
  Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),Train.evaluate(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Train.AveragePool{})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = x_packed
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),[images,channels,height,width],restored) :
    Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),
      (F.reshape(F.mean_axes(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},[2,3]),Train.pooled_extents(_)),Train.AveragePool{}))
  KernelGemm.Both{pool_packed(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,xm,xs,xf,xw,xa,xl,xp,
      KernelGemm.Both{global,Public.Laid{inside,shaped,restored,counted,pinned}}),{==}}

def pool_evaluate_packed(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  -tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),+packed: Public.Packed<tiled>) ->
  Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),Train.evaluate(tiled,Train.AveragePool{})):
  Public.Packed{+xm,+xs,+xf,+xw,+xa,+xl,+xp,+equation} = packed
  %equation : Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),Train.evaluate(_,Train.AveragePool{}))
  pool_evaluate_model(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,xm,xs,xf,xw,xa,xl,xp,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},tiled,equation) : Public.Contract(~X,~x_value,x,[images,channels,height,width],_) contract :
      Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})})

# evaluate of the pool: the output of forward.
def pool_evaluate(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: PoolFacts<limits,images,channels,height,width>,
  tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),layer: Train.Layer,+described: PoolLayer(layer)) ->
  Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),Train.evaluate(tiled,layer)):
  %Equal.sym(Train.Layer,layer,Train.AveragePool{},described) :
    Answered(~(output => PoolOutput(~X,~x_value,x,images,channels,height,width,output)),~(layer => PoolLayer(layer)),Train.evaluate(tiled,_))
  pool_evaluate_packed(~X,~x_value,x,images,channels,height,width,limits,default_capacity,facts,tiled,contract,
    Public.packed(~X,~x_value,x,[images,channels,height,width],tiled,contract))

# The gradient of the pool spreads dy / (H * W) over [N, C, H, W].
def PoolSpreadValue(~DY: Data) -> Data:
  ViewGather.Viewing<PiecesMap.Context<F32,DY>>

def pool_spread_value(~DY: Data,~dy_value: DY -> U32 -> F32) -> PoolSpreadValue(~DY) -> U32 -> F32:
  ViewGather.viewed(~F32,~PiecesMap.Context<F32,DY>,~PiecesMap.value(~F32,~F32,~DY,~F.scalar_value(~F32.div),~dy_value))

def PoolGradient(~DY: Data,~dy_value: DY -> U32 -> F32,+dy: DY,+images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  output: Tiled.Tiled) -> Data:
  Public.Contract(~PoolSpreadValue(~DY),~pool_spread_value(~DY,~dy_value),ViewGather.Viewing{PiecesMap.Context{U32.to_f32((height * width : U32)),dy},axes,offset,[]},
    [images,channels,height,width],output)

# The view that stretches [N, C, 1, 1] over [N, C, H, W].
type PoolSpread<-images: U32,-channels: U32,-height: U32,-width: U32,-axes: List<&2,Views.Axis>,-offset: U32> is Data:
  PoolSpread{stretched: {Tensors.stretched(Tensors.view_of([images,channels,1,1]),[images,channels,height,width]) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
    nonzero: {Shapes.has_zero([images,channels,1,1]) == False{} : Bool},target_nonzero: {Shapes.has_zero([images,channels,height,width]) == False{} : Bool},
    count: U32,counted: {Tensors.count_of([images,channels,height,width]) == Some{count} : Maybe<&2,U32>},
    fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero([images,channels,1,1])) == True{} : Bool}}

def pool_backward_packed(~DY: Data,~dy_value: DY -> U32 -> F32,+dy: DY,+images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+spread: PoolSpread<images,channels,height,width,axes,offset>,
  +dm: PiecesModel.Model<Tile.Tile8>,+ds: List<&2,U32>,+df: Tile.Tile8,+dw: U32,+da: Nat,+dl: U32,+dp: Grid.Position,+d_packed: Public.Contract(~DY,~dy_value,dy,[images,channels,1,1],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp})) ->
  PoolGradient(~DY,~dy_value,dy,images,channels,height,width,axes,offset,Train.pooled_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},[images,channels,height,width])):
  PoolSpread{+stretched,+nonzero,+target_nonzero,+count,+counted,+fits} = spread
  TiledView.broadcast(~PiecesMap.Context<F32,DY>,~PiecesMap.value(~F32,~F32,~DY,~F.scalar_value(~F32.div),~dy_value),PiecesMap.Context{U32.to_f32((height * width : U32)),dy},
    limits,[images,channels,1,1],[images,channels,height,width],axes,offset,F.div_scalar(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},U32.to_f32((height * width : U32))),stretched,nonzero,target_nonzero,
    default_capacity,four_room(limits),four_room(limits),count,counted,fits,
    FunctionValues.scalar_public(~DY,~dy_value,~F32.div,U32.to_f32((height * width : U32)),1.0)(dy,[images,channels,1,1],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},d_packed))

# backward of the pool, for dy [N, C, 1, 1] of the contract and the extents
# forward saved.
def pool_backward(~DY: Data,~dy_value: DY -> U32 -> F32,+dy: DY,+images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+spread: PoolSpread<images,channels,height,width,axes,offset>,
  gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,[images,channels,1,1],gradient),
  saved: Train.Saved,+kept: {saved == Train.Pooled{[images,channels,height,width]} : Train.Saved},layer: Train.Layer,+described: PoolLayer(layer)) ->
  Answered(~(output => PoolGradient(~DY,~dy_value,dy,images,channels,height,width,axes,offset,output)),~(layer => PoolLayer(layer)),Train.backward(gradient,saved,layer)):
  %Equal.sym(Train.Saved,saved,Train.Pooled{[images,channels,height,width]},kept) :
    Answered(~(output => PoolGradient(~DY,~dy_value,dy,images,channels,height,width,axes,offset,output)),~(layer => PoolLayer(layer)),Train.backward(gradient,_,layer))
  %Equal.sym(Train.Layer,layer,Train.AveragePool{},described) :
    Answered(~(output => PoolGradient(~DY,~dy_value,dy,images,channels,height,width,axes,offset,output)),~(layer => PoolLayer(layer)),
      Train.backward(gradient,Train.Pooled{[images,channels,height,width]},_))
  KernelGemm.Both{Public.by_packed(~DY,~dy_value,(that => PoolGradient(~DY,~dy_value,dy,images,channels,height,width,axes,offset,
      Train.pooled_backward(that,[images,channels,height,width]))),dy,[images,channels,1,1],gradient,dy_contract,
    dm => ds => df => dw => da => dl => dp => d_packed => pool_backward_packed(~DY,~dy_value,dy,images,channels,height,width,axes,offset,limits,default_capacity,spread,dm,ds,df,dw,da,dl,dp,d_packed)),{==}}

# ---- Parameters ----

# A parameter: its values, the velocity of the last step, and whether a step
# with momentum ran.
type Parameter<-X: Data> is Data:
  Parameter{value: X,velocity: X,moving: Bool}

def parameter_value(~X: Data,+parameter: Parameter<X>) -> X:
  Parameter{+value,velocity,moving} = parameter
  value

def parameter_velocity(~X: Data,+parameter: Parameter<X>) -> X:
  Parameter{value,+velocity,moving} = parameter
  velocity

def parameter_moving(~X: Data,+parameter: Parameter<X>) -> Bool:
  Parameter{value,velocity,+moving} = parameter
  moving

# No gradient.
def Cleared(gradient: Train.Gradient) -> Data:
  match gradient:
    case Train.Empty{}: {True{} == True{} : Bool}
    case Train.Held{tensor}: {False{} == True{} : Bool}

# A gradient that holds the values of its reader.
def Computed(~R: Data,~r_value: R -> U32 -> F32,+context: R,+shape: List<&2,U32>,gradient: Train.Gradient) -> Data:
  match gradient:
    case Train.Empty{}: {False{} == True{} : Bool}
    case Train.Held{tensor}: Public.Contract(~R,~r_value,context,shape,tensor)

# A parameter in a layer: its tensor holds the values, its gradient what
# Graded states, and its velocity the velocity.
def Slot(~X: Data,~x_value: X -> U32 -> F32,~Graded: Train.Gradient -> Data,+parameter: Parameter<X>,+shape: List<&2,U32>,tensor: Tiled.Tiled,
  gradient: Train.Gradient,velocity: Optimizer.Velocity) -> Data:
  KernelGemm.Both<Public.Contract(~X,~x_value,parameter_value(~X,parameter),shape,tensor),
    KernelGemm.Both<Graded(gradient),OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,parameter),shape,parameter_moving(~X,parameter),velocity)>>

# ---- Batch normalization ----

# A batch normalization of C channels: gamma and beta in their slots, the
# running statistics, epsilon and momentum.
def NormLayer(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+channels: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,
  layer: Train.Layer) -> Data:
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<{layer_momentum == momentum : F32},
        KernelGemm.Both<Slot(~X,~x_value,~GammaGraded,gamma,[channels],gammas,gamma_gradient,gamma_velocity),
          KernelGemm.Both<Slot(~X,~x_value,~BetaGraded,beta,[channels],betas,beta_gradient,beta_velocity),
            KernelGemm.Both<Public.Contract(~R,~r_value,mean,[channels],means),Public.Contract(~R,~r_value,variance,[channels],variances)>>>>>
    case layer: {False{} == True{} : Bool}

# The velocities of gamma and beta over [features], kept.
def ParametersKept(~X: Data,~x_value: X -> U32 -> F32,+gamma: Parameter<X>,+beta: Parameter<X>,+features: U32,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity) -> Data:
  KernelGemm.Both<OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,gamma),[features],parameter_moving(~X,gamma),gamma_velocity),
    OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,beta),[features],parameter_moving(~X,beta),beta_velocity)>

# forward: what the public theorem states of the output, xhat, the deviation,
# gamma, beta and the moved running statistics, the gradients and velocities
# as they were.
def NormForward(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.Normalized{pair}:
      match layer:
        case Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
          KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<{layer_momentum == momentum : F32},
            KernelGemm.Both<NormPublic.ForwardHolds(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
                images,channels,height,width,epsilon,momentum,(output,(pair,((gammas,betas),(means,variances))))),
              KernelGemm.Both<GammaGraded(gamma_gradient),KernelGemm.Both<BetaGraded(beta_gradient),ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)>>>>>
        case layer: {False{} == True{} : Bool}
    case saved: {False{} == True{} : Bool}

# The room the statistics of [N, C, H, W] need.
type NormRoom<-images: U32,-channels: U32,-height: U32,-width: U32> is Data:
  NormRoom{nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},cap: U32,
    cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}}

def normalized_holds(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,result: Norm.Forward(),
  +held: NormPublic.ForwardHolds(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
    images,channels,height,width,epsilon,momentum,result),
  +gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)) ->
  NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.normalized(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,result)):
  (output,(saved,((gammas,betas),(means,variances)))) = result
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{gamma_graded,KernelGemm.Both{beta_graded,kept}}}}}

def norm_forward_packed(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+room: NormRoom<images,channels,height,width>,
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),
  +mm: PiecesModel.Model<Tile.Tile8>,+ms: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+ma: Nat,+ml: U32,+mp: Grid.Position,+m_packed: Public.Contract(~R,~r_value,mean,[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp}),
  +vm: PiecesModel.Model<Tile.Tile8>,+vs: List<&2,U32>,+vf: Tile.Tile8,+vw: U32,+va: Nat,+vl: U32,+vp: Grid.Position,+v_packed: Public.Contract(~R,~r_value,variance,[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp})) ->
  NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
    Train.normalized(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp}),Norm.settings(epsilon,momentum)))):
  NormRoom{+nonempty,+cap,+cap_ok,+count_room} = room
  normalized_holds(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,
    Norm.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp}),Norm.settings(epsilon,momentum)),
    NormPublic.forward(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
      images,channels,height,width,epsilon,momentum,limits,default_capacity,nonempty,four_room(limits),cap,cap_ok,count_room,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},x_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},g_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},b_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},m_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp},v_packed),
    gamma_graded,beta_graded,kept)

# forward of a batch normalization of the description on its tensors.
def NormForwardOf(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,gamma_gradient: Train.Gradient,beta_gradient: Train.Gradient,
  gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,input: Tiled.Tiled,parameters: Tiled.Tiled & Tiled.Tiled,running: Tiled.Tiled & Tiled.Tiled) -> Data:
  NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
    Train.normalized(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.forward(input,parameters,running,Norm.settings(epsilon,momentum))))

# forward of a batch normalization of the description on x [N, C, H, W] of
# the contract.
def norm_forward(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+room: NormRoom<images,channels,height,width>,
  tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,channels,epsilon,momentum,gamma,beta,mean,variance,layer)) ->
  NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.forward(tiled,layer)):
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}},KernelGemm.Both{+mean_contract,+variance_contract}}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
          Train.forward(tiled,Train.BatchNorm{gammas,betas,means,variances,_,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      %Equal.sym(F32,layer_momentum,momentum,momentum_exact) :
        NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
          Train.forward(tiled,Train.BatchNorm{gammas,betas,means,variances,epsilon,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      Public.by_packed(~X,~x_value,(that => NormForwardOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,that,(gammas,betas),(means,variances))),x,[images,channels,height,width],tiled,contract,
        xm => xs => xf => xw => xa => xl => xp => x_packed => Public.by_packed(~X,~x_value,(that => NormForwardOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(that,betas),(means,variances))),parameter_value(~X,gamma),[channels],gammas,gamma_contract,
          gm => gs => gf => gw => ga => gl => gp => g_packed => Public.by_packed(~X,~x_value,(that => NormForwardOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},that),(means,variances))),parameter_value(~X,beta),[channels],betas,beta_contract,
            bm => bs => bf => bw => ba => bl => bp => b_packed => Public.by_packed(~R,~r_value,(that => NormForwardOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(that,variances))),mean,[channels],means,mean_contract,
              mm => ms => mf => mw => ma => ml => mp => m_packed => Public.by_packed(~R,~r_value,(that => NormForwardOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},that))),variance,[channels],variances,variance_contract,
                vm => vs => vf => vw => va => vl => vp => v_packed => norm_forward_packed(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,limits,default_capacity,room,
                  gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,KernelGemm.Both{gamma_kept,beta_kept},
                  xm,xs,xf,xw,xa,xl,xp,x_packed,gm,gs,gf,gw,ga,gl,gp,g_packed,bm,bs,bf,bw,ba,bl,bp,b_packed,mm,ms,mf,mw,ma,ml,mp,m_packed,vm,vs,vf,vw,va,vl,vp,v_packed))))))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.forward(tiled,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.forward(tiled,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.forward(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(NormForward(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.forward(tiled,Train.AveragePool{})),Words.false_true(held))

# evaluate: what the public theorem states of the output, gamma, beta and the
# running statistics, the gradients and velocities as they were.
def NormEvaluate(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,result: Tiled.Tiled & Train.Layer) -> Data:
  (output,layer) = result
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<{layer_momentum == momentum : F32},
        KernelGemm.Both<NormPublic.EvaluateHolds(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
            images,channels,height,width,epsilon,(output,((gammas,betas),(means,variances)))),
          KernelGemm.Both<GammaGraded(gamma_gradient),KernelGemm.Both<BetaGraded(beta_gradient),ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)>>>>>
    case layer: {False{} == True{} : Bool}

def norm_evaluated_holds(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,result: Norm.Evaluated(),
  +held: NormPublic.EvaluateHolds(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
    images,channels,height,width,epsilon,result),
  +gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)) ->
  NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.norm_evaluated(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,result)):
  (output,((gammas,betas),(means,variances))) = result
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{gamma_graded,KernelGemm.Both{beta_graded,kept}}}}}

def norm_evaluate_packed(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),
  +mm: PiecesModel.Model<Tile.Tile8>,+ms: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+ma: Nat,+ml: U32,+mp: Grid.Position,+m_packed: Public.Contract(~R,~r_value,mean,[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp}),
  +vm: PiecesModel.Model<Tile.Tile8>,+vs: List<&2,U32>,+vf: Tile.Tile8,+vw: U32,+va: Nat,+vl: U32,+vp: Grid.Position,+v_packed: Public.Contract(~R,~r_value,variance,[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp})) ->
  NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
    Train.norm_evaluated(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.evaluate(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp}),Norm.settings(epsilon,momentum)))):
  norm_evaluated_holds(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,
    Norm.evaluate(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp}),Norm.settings(epsilon,momentum)),
    NormPublic.evaluate(~X,~x_value,~R,~r_value,~R,~r_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),mean,variance,
      images,channels,height,width,epsilon,momentum,limits,default_capacity,nonempty,four_room(limits),
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},x_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},g_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},b_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},m_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,vm),vs,vf,Tensors.Usable{},vw},va,vl,vp},v_packed),
    gamma_graded,beta_graded,kept)

# evaluate of a batch normalization of the description on its tensors.
def NormEvaluateOf(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,gamma_gradient: Train.Gradient,beta_gradient: Train.Gradient,
  gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,input: Tiled.Tiled,parameters: Tiled.Tiled & Tiled.Tiled,running: Tiled.Tiled & Tiled.Tiled) -> Data:
  NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
    Train.norm_evaluated(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.evaluate(input,parameters,running,Norm.settings(epsilon,momentum))))

# evaluate of a batch normalization of the description on x [N, C, H, W] of
# the contract.
def norm_evaluate(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},
  tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,[images,channels,height,width],tiled),
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,channels,epsilon,momentum,gamma,beta,mean,variance,layer)) ->
  NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.evaluate(tiled,layer)):
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}},KernelGemm.Both{+mean_contract,+variance_contract}}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
          Train.evaluate(tiled,Train.BatchNorm{gammas,betas,means,variances,_,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      %Equal.sym(F32,layer_momentum,momentum,momentum_exact) :
        NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,
          Train.evaluate(tiled,Train.BatchNorm{gammas,betas,means,variances,epsilon,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      Public.by_packed(~X,~x_value,(that => NormEvaluateOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,that,(gammas,betas),(means,variances))),x,[images,channels,height,width],tiled,contract,
        xm => xs => xf => xw => xa => xl => xp => x_packed => Public.by_packed(~X,~x_value,(that => NormEvaluateOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(that,betas),(means,variances))),parameter_value(~X,gamma),[channels],gammas,gamma_contract,
          gm => gs => gf => gw => ga => gl => gp => g_packed => Public.by_packed(~X,~x_value,(that => NormEvaluateOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},that),(means,variances))),parameter_value(~X,beta),[channels],betas,beta_contract,
            bm => bs => bf => bw => ba => bl => bp => b_packed => Public.by_packed(~R,~r_value,(that => NormEvaluateOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(that,variances))),mean,[channels],means,mean_contract,
              mm => ms => mf => mw => ma => ml => mp => m_packed => Public.by_packed(~R,~r_value,(that => NormEvaluateOf(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,mm),ms,mf,Tensors.Usable{},mw},ma,ml,mp},that))),variance,[channels],variances,variance_contract,
                vm => vs => vf => vw => va => vl => vp => v_packed => norm_evaluate_packed(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,limits,default_capacity,nonempty,
                  gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,KernelGemm.Both{gamma_kept,beta_kept},
                  xm,xs,xf,xw,xa,xl,xp,x_packed,gm,gs,gf,gw,ga,gl,gp,g_packed,bm,bs,bf,bw,ba,bl,bp,b_packed,mm,ms,mf,mw,ma,ml,mp,m_packed,vm,vs,vf,vw,va,vl,vp,v_packed))))))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.evaluate(tiled,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.evaluate(tiled,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.evaluate(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(NormEvaluate(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,x,mean,variance,Train.evaluate(tiled,Train.AveragePool{})),Words.false_true(held))

# What backward reads: xhat and 1 / sqrt(variance + epsilon).
def NormSavedPair(~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,+xhat: H,+deviation: V,+images: U32,+channels: U32,+height: U32,+width: U32,
  pair: Tiled.Tiled & Tiled.Tiled) -> Data:
  (normalized,inverse) = pair
  KernelGemm.Both<Public.Contract(~H,~h_value,xhat,[images,channels,height,width],normalized),Public.Contract(~V,~v_value,deviation,[channels],inverse)>

# backward: what the public theorem states of dx, the gradients of gamma and
# beta and gamma, and beta, the running statistics and the velocities as
# they were.
def NormBackward(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,result: Tiled.Tiled & Train.Layer) -> Data:
  (dx,layer) = result
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      match gamma_gradient:
        case Train.Held{dgamma}:
          match beta_gradient:
            case Train.Held{dbeta}:
              KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<{layer_momentum == momentum : F32},
                KernelGemm.Both<NormPublic.BackwardHolds(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,dy,xhat,deviation,parameter_value(~X,gamma),images,channels,height,width,
                    (dx,(dgamma,(dbeta,gammas)))),
                  KernelGemm.Both<Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],betas),
                    KernelGemm.Both<Public.Contract(~R,~r_value,mean,[channels],means),KernelGemm.Both<Public.Contract(~R,~r_value,variance,[channels],variances),
                      ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)>>>>>>
            case Train.Empty{}: {False{} == True{} : Bool}
        case Train.Empty{}: {False{} == True{} : Bool}
    case layer: {False{} == True{} : Bool}

def norm_back_holds(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,-betas: Tiled.Tiled,-means: Tiled.Tiled,-variances: Tiled.Tiled,
  -gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,result: Norm.Backward(),
  +held: NormPublic.BackwardHolds(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,dy,xhat,deviation,parameter_value(~X,gamma),images,channels,height,width,result),
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],betas),+mean_contract: Public.Contract(~R,~r_value,mean,[channels],means),
  +variance_contract: Public.Contract(~R,~r_value,variance,[channels],variances),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity)) ->
  NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,
    Train.norm_back(betas,means,variances,epsilon,momentum,gamma_velocity,beta_velocity,result)):
  (dx,(dgamma,(dbeta,gammas))) = result
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{beta_contract,KernelGemm.Both{mean_contract,KernelGemm.Both{variance_contract,kept}}}}}}

def norm_backward_packed(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+room: NormRoom<images,channels,height,width>,
  -betas: Tiled.Tiled,-means: Tiled.Tiled,-variances: Tiled.Tiled,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],betas),+mean_contract: Public.Contract(~R,~r_value,mean,[channels],means),
  +variance_contract: Public.Contract(~R,~r_value,variance,[channels],variances),+kept: ParametersKept(~X,~x_value,gamma,beta,channels,gamma_velocity,beta_velocity),
  +ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_packed: Public.Contract(~DY,~dy_value,dy,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}),
  +hm: PiecesModel.Model<Tile.Tile8>,+hs: List<&2,U32>,+hf: Tile.Tile8,+hw: U32,+ha: Nat,+hl: U32,+hp: Grid.Position,+h_packed: Public.Contract(~H,~h_value,xhat,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp}),
  +dm: PiecesModel.Model<Tile.Tile8>,+ds: List<&2,U32>,+df: Tile.Tile8,+dw: U32,+da: Nat,+dl: U32,+dp: Grid.Position,+d_packed: Public.Contract(~V,~v_value,deviation,[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[channels],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp})) ->
  NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,
    Train.norm_back(betas,means,variances,epsilon,momentum,gamma_velocity,beta_velocity,Norm.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp}))):
  NormRoom{+nonempty,+cap,+cap_ok,+count_room} = room
  norm_back_holds(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,betas,means,variances,gamma_velocity,beta_velocity,Norm.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp}),
    NormPublic.backward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,dy,xhat,deviation,parameter_value(~X,gamma),images,channels,height,width,
      limits,default_capacity,nonempty,four_room(limits),cap,cap_ok,count_room,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},y_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp},h_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},d_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,Tensors.Usable{},gw},ga,gl,gp},g_packed),
    beta_contract,mean_contract,variance_contract,kept)

# backward of a batch normalization of the description on its tensors.
def NormBackwardOf(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,betas: Tiled.Tiled,means: Tiled.Tiled,variances: Tiled.Tiled,gamma_velocity: Optimizer.Velocity,
  beta_velocity: Optimizer.Velocity,gradient: Tiled.Tiled,saved: Tiled.Tiled & Tiled.Tiled,gammas: Tiled.Tiled) -> Data:
  NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,
    Train.norm_back(betas,means,variances,epsilon,momentum,gamma_velocity,beta_velocity,Norm.backward(gradient,saved,gammas)))

def norm_backward_pair(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  ~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+room: NormRoom<images,channels,height,width>,
  gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,[images,channels,height,width],gradient),
  pair: Tiled.Tiled & Tiled.Tiled,+kept: NormSavedPair(~H,~h_value,~V,~v_value,xhat,deviation,images,channels,height,width,pair),
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,channels,epsilon,momentum,gamma,beta,mean,variance,layer)) ->
  NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{pair},layer)):
  (normalized,inverse) = pair
  KernelGemm.Both{+xhat_contract,+deviation_contract} = kept
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}},KernelGemm.Both{+mean_contract,+variance_contract}}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},
          Train.BatchNorm{gammas,betas,means,variances,_,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      %Equal.sym(F32,layer_momentum,momentum,momentum_exact) :
        NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},
          Train.BatchNorm{gammas,betas,means,variances,epsilon,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      Public.by_packed(~DY,~dy_value,(that => NormBackwardOf(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,betas,means,variances,gamma_velocity,beta_velocity,that,(normalized,inverse),gammas)),dy,[images,channels,height,width],gradient,dy_contract,
        ym => ys => yf => yw => ya => yl => yp => y_packed => Public.by_packed(~H,~h_value,(that => NormBackwardOf(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,betas,means,variances,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(that,inverse),gammas)),xhat,[images,channels,height,width],normalized,xhat_contract,
          hm => hs => hf => hw => ha => hl => hp => h_packed => Public.by_packed(~V,~v_value,(that => NormBackwardOf(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,betas,means,variances,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp},that),gammas)),deviation,[channels],inverse,deviation_contract,
            dm => ds => df => dw => da => dl => dp => d_packed => Public.by_packed(~X,~x_value,(that => NormBackwardOf(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,betas,means,variances,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,Tensors.Usable{},hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),that)),parameter_value(~X,gamma),[channels],gammas,gamma_contract,
              gm => gs => gf => gw => ga => gl => gp => g_packed => norm_backward_packed(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,limits,default_capacity,room,
                betas,means,variances,gamma_velocity,beta_velocity,beta_contract,mean_contract,variance_contract,KernelGemm.Both{gamma_kept,beta_kept},
                ym,ys,yf,yw,ya,yl,yp,y_packed,hm,hs,hf,hw,ha,hl,hp,h_packed,dm,ds,df,dw,da,dl,dp,d_packed,gm,gs,gf,gw,ga,gl,gp,g_packed)))))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Normalized{(normalized,inverse)},Train.AveragePool{})),Words.false_true(held))

# What backward reads of a batch normalization.
def NormSaved(~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,+xhat: H,+deviation: V,+images: U32,+channels: U32,+height: U32,+width: U32,
  saved: Train.Saved) -> Data:
  match saved:
    case Train.Normalized{pair}: NormSavedPair(~H,~h_value,~V,~v_value,xhat,deviation,images,channels,height,width,pair)
    case saved: {False{} == True{} : Bool}

# backward of a batch normalization of the description, for dy of the
# contract and what forward saved.
def norm_backward(~DY: Data,~dy_value: DY -> U32 -> F32,~H: Data,~h_value: H -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,
  ~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+dy: DY,+xhat: H,+deviation: V,+mean: R,+variance: R,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+room: NormRoom<images,channels,height,width>,
  gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,[images,channels,height,width],gradient),
  saved: Train.Saved,+kept: NormSaved(~H,~h_value,~V,~v_value,xhat,deviation,images,channels,height,width,saved),
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,channels,epsilon,momentum,gamma,beta,mean,variance,layer)) ->
  NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,saved,layer)):
  match saved:
    case Train.Normalized{pair}:
      norm_backward_pair(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,~GammaGraded,~BetaGraded,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,limits,default_capacity,room,gradient,dy_contract,pair,kept,layer,held)
    case Train.Convolved{other_saved}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Convolved{other_saved},layer)),Words.false_true(kept))
    case Train.LineNormalized{other_saved}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.LineNormalized{other_saved},layer)),Words.false_true(kept))
    case Train.Activated{other_input}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Activated{other_input},layer)),Words.false_true(kept))
    case Train.Pooled{other_shape}:
      Empty.absurd(NormBackward(~DY,~dy_value,~H,~h_value,~V,~v_value,~X,~x_value,~R,~r_value,images,channels,height,width,epsilon,momentum,gamma,beta,dy,xhat,deviation,mean,variance,Train.backward(gradient,Train.Pooled{other_shape},layer)),Words.false_true(kept))

# ---- Layer normalization ----

# A layer normalization over the last axis of shape, [features]: gamma and
# beta in their slots, and epsilon.
def LineLayer(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,layer: Train.Layer) -> Data:
  match layer:
    case Train.LayerNorm{gammas,betas,layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},
        KernelGemm.Both<Slot(~X,~x_value,~GammaGraded,gamma,[features],gammas,gamma_gradient,gamma_velocity),
          Slot(~X,~x_value,~BetaGraded,beta,[features],betas,beta_gradient,beta_velocity)>>
    case layer: {False{} == True{} : Bool}

# The room and the views the layer normalization of shape reads.
type LineFacts<-limits: Machine.Limits,-shape: List<&2,U32>,-features: U32,-laxes: List<&2,Views.Axis>,-loffset: U32,-raxes: List<&2,Views.Axis>,-roffset: U32,
  -gaxes: List<&2,Views.Axis>,-goffset: U32> is Data:
  LineFacts{rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
    own_stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
    row_stretched: {Tensors.stretched(Tensors.view_of(LineNorm.unit_extents(shape)),shape) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
    fits: {Shapes.same_extents(LineNorm.last_of(shape),[features]) == True{} : Bool},
    gamma_stretched: {Tensors.stretched(Tensors.view_of([features]),shape) == Some{Views.View{gaxes,goffset}} : Maybe<&2,Views.View>}}

# forward: what the layer normalization theorem states of the output, xhat,
# the deviation, gamma and beta, the gradients and velocities as they were.
def LineForward(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.LineNormalized{kept}:
      match layer:
        case Train.LayerNorm{gammas,betas,layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
          KernelGemm.Both<{layer_epsilon == epsilon : F32},
            KernelGemm.Both<LineValues.ForwardHolds(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,(output,(kept,(gammas,betas)))),
              KernelGemm.Both<GammaGraded(gamma_gradient),KernelGemm.Both<BetaGraded(beta_gradient),ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)>>>>
        case layer: {False{} == True{} : Bool}
    case saved: {False{} == True{} : Bool}

def line_forward_holds(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,result: LineNorm.Forward(),
  +held: LineValues.ForwardHolds(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,result),+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)) ->
  LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_normalized(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,result)):
  (output,(saved,(gammas,betas))) = result
  KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{gamma_graded,KernelGemm.Both{beta_graded,kept}}}}

def line_forward_packed(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xt: Tensors.Status,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp}),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gt: Tensors.Status,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bt: Tensors.Status,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp})) ->
  LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_normalized(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp}),epsilon))):
  LineFacts{+rank_room,+nonempty,+cap,+cap_ok,+count_room,+own_stretched,+row_stretched,+fits,+gamma_stretched} = facts
  line_forward_holds(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp}),epsilon),
    LineValues.forward(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,own_stretched,row_stretched,fits,gamma_stretched,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},x_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},g_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp},b_packed),
    gamma_graded,beta_graded,kept)

# forward of a layer normalization of the description on its tensors.
def LineForwardOf(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,gamma_gradient: Train.Gradient,beta_gradient: Train.Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,
  input: Tiled.Tiled,gammas: Tiled.Tiled,betas: Tiled.Tiled) -> Data:
  LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_normalized(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.forward(input,(gammas,betas),epsilon)))

def line_forward_tensors(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  -tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,shape,tiled),+x_observed: Public.Observed<tiled>,
  -gammas: Tiled.Tiled,+gamma_contract: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],gammas),+g_observed: Public.Observed<gammas>,
  -betas: Tiled.Tiled,+beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+b_observed: Public.Observed<betas>) ->
  LineForwardOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,tiled,gammas,betas):
  Public.Observed{+xm,+xs,+xf,+xt,+xw,+xa,+xl,+xp,+x_equation} = x_observed
  Public.Observed{+gm,+gs,+gf,+gt,+gw,+ga,+gl,+gp,+g_equation} = g_observed
  Public.Observed{+bm,+bs,+bf,+bt,+bw,+ba,+bl,+bp,+b_equation} = b_observed
  %x_equation : LineForwardOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,_,gammas,betas)
  %g_equation : LineForwardOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},_,betas)
  %b_equation : LineForwardOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},_)
  line_forward_packed(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,limits,default_capacity,facts,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,kept,
    xm,xs,xf,xt,xw,xa,xl,xp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},tiled,x_equation) : Public.Contract(~X,~x_value,x,shape,_) contract : Public.Contract(~X,~x_value,x,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp})},
    gm,gs,gf,gt,gw,ga,gl,gp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},gammas,g_equation) : Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],_) gamma_contract :
      Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp})},
    bm,bs,bf,bt,bw,ba,bl,bp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp},betas,b_equation) : Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],_) beta_contract :
      Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp})})

# forward of a layer normalization of the description on x of the contract
# whose output is usable.
def line_forward(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,shape,tiled),
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~GammaGraded,~BetaGraded,features,epsilon,gamma,beta,layer)) ->
  LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,layer)):
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,Train.LayerNorm{gammas,betas,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      line_forward_tensors(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,limits,default_capacity,facts,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,KernelGemm.Both{gamma_kept,beta_kept},
        tiled,contract,Public.observe(tiled),gammas,gamma_contract,Public.observe(gammas),betas,beta_contract,Public.observe(betas))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LineForward(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.forward(tiled,Train.AveragePool{})),Words.false_true(held))

# evaluate: what the layer normalization theorem states of the output, gamma
# and beta, the gradients and velocities as they were.
def LineEvaluate(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,result: Tiled.Tiled & Train.Layer) -> Data:
  (output,layer) = result
  match layer:
    case Train.LayerNorm{gammas,betas,layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},
        KernelGemm.Both<LineValues.EvaluateHolds(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,(output,(gammas,betas))),
          KernelGemm.Both<GammaGraded(gamma_gradient),KernelGemm.Both<BetaGraded(beta_gradient),ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)>>>>
    case layer: {False{} == True{} : Bool}

def line_evaluated_holds(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,result: LineNorm.Evaluated(),
  +held: LineValues.EvaluateHolds(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,result),+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)) ->
  LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_norm_evaluated(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,result)):
  (output,(gammas,betas)) = result
  KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{gamma_graded,KernelGemm.Both{beta_graded,kept}}}}

def line_evaluate_packed(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xt: Tensors.Status,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp}),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gt: Tensors.Status,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bt: Tensors.Status,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp})) ->
  LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_norm_evaluated(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.evaluate(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp}),epsilon))):
  LineFacts{+rank_room,+nonempty,+cap,+cap_ok,+count_room,+own_stretched,+row_stretched,+fits,+gamma_stretched} = facts
  line_evaluated_holds(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.evaluate(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp}),epsilon),
    LineValues.evaluate(~X,~x_value,~X,~x_value,~X,~x_value,x,parameter_value(~X,gamma),parameter_value(~X,beta),shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,own_stretched,row_stretched,fits,gamma_stretched,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},x_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},g_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp},b_packed),
    gamma_graded,beta_graded,kept)

# evaluate of a layer normalization of the description on its tensors.
def LineEvaluateOf(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,gamma_gradient: Train.Gradient,beta_gradient: Train.Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,
  input: Tiled.Tiled,gammas: Tiled.Tiled,betas: Tiled.Tiled) -> Data:
  LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.line_norm_evaluated(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.evaluate(input,(gammas,betas),epsilon)))

def line_evaluate_tensors(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,-gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,+gamma_graded: GammaGraded(gamma_gradient),+beta_graded: BetaGraded(beta_gradient),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  -tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,shape,tiled),+x_observed: Public.Observed<tiled>,
  -gammas: Tiled.Tiled,+gamma_contract: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],gammas),+g_observed: Public.Observed<gammas>,
  -betas: Tiled.Tiled,+beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+b_observed: Public.Observed<betas>) ->
  LineEvaluateOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,tiled,gammas,betas):
  Public.Observed{+xm,+xs,+xf,+xt,+xw,+xa,+xl,+xp,+x_equation} = x_observed
  Public.Observed{+gm,+gs,+gf,+gt,+gw,+ga,+gl,+gp,+g_equation} = g_observed
  Public.Observed{+bm,+bs,+bf,+bt,+bw,+ba,+bl,+bp,+b_equation} = b_observed
  %x_equation : LineEvaluateOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,_,gammas,betas)
  %g_equation : LineEvaluateOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},_,betas)
  %b_equation : LineEvaluateOf(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},_)
  line_evaluate_packed(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,limits,default_capacity,facts,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,kept,
    xm,xs,xf,xt,xw,xa,xl,xp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp},tiled,x_equation) : Public.Contract(~X,~x_value,x,shape,_) contract : Public.Contract(~X,~x_value,x,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,xt,xw},xa,xl,xp})},
    gm,gs,gf,gt,gw,ga,gl,gp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},gammas,g_equation) : Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],_) gamma_contract :
      Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp})},
    bm,bs,bf,bt,bw,ba,bl,bp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp},betas,b_equation) : Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],_) beta_contract :
      Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,bt,bw},ba,bl,bp})})

# evaluate of a layer normalization of the description on x of the contract
# whose output is usable.
def line_evaluate(~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,shape,tiled),
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~GammaGraded,~BetaGraded,features,epsilon,gamma,beta,layer)) ->
  LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,layer)):
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,Train.LayerNorm{gammas,betas,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      line_evaluate_tensors(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,limits,default_capacity,facts,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,gamma_graded,beta_graded,KernelGemm.Both{gamma_kept,beta_kept},
        tiled,contract,Public.observe(tiled),gammas,gamma_contract,Public.observe(gammas),betas,beta_contract,Public.observe(betas))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LineEvaluate(~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,Train.evaluate(tiled,Train.AveragePool{})),Words.false_true(held))

# What backward reads of a layer normalization of x: xhat and the deviation.
def LineSaved(~X: Data,~x_value: X -> U32 -> F32,+x: X,+shape: List<&2,U32>,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,saved: Train.Saved) -> Data:
  match saved:
    case Train.LineNormalized{kept}: LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,kept)
    case saved: {False{} == True{} : Bool}

# backward: what the layer normalization theorem states of dx, the gradients
# of gamma and beta and gamma, and beta and the velocities as they were.
def LineBackward(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,result: Tiled.Tiled & Train.Layer) -> Data:
  (dx,layer) = result
  match layer:
    case Train.LayerNorm{gammas,betas,layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      match gamma_gradient:
        case Train.Held{dgamma}:
          match beta_gradient:
            case Train.Held{dbeta}:
              KernelGemm.Both<{layer_epsilon == epsilon : F32},
                KernelGemm.Both<LineValues.BackwardHolds(~X,~x_value,~X,~x_value,~DY,~dy_value,x,parameter_value(~X,gamma),dy,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,(dx,(dgamma,(dbeta,gammas)))),
                  KernelGemm.Both<Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)>>>
            case Train.Empty{}: {False{} == True{} : Bool}
        case Train.Empty{}: {False{} == True{} : Bool}
    case layer: {False{} == True{} : Bool}

def line_back_holds(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,-betas: Tiled.Tiled,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  result: LineNorm.Backward(),+held: LineValues.BackwardHolds(~X,~x_value,~X,~x_value,~DY,~dy_value,x,parameter_value(~X,gamma),dy,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,result),
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity)) ->
  LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.line_norm_back(betas,epsilon,gamma_velocity,beta_velocity,result)):
  (dx,(dgamma,(dbeta,gammas))) = result
  KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{beta_contract,kept}}}

def line_backward_packed(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  -betas: Tiled.Tiled,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  +ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yt: Tensors.Status,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_packed: Public.Contract(~DY,~dy_value,dy,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp}),
  +hm: PiecesModel.Model<Tile.Tile8>,+hs: List<&2,U32>,+hf: Tile.Tile8,+ht: Tensors.Status,+hw: U32,+ha: Nat,+hl: U32,+hp: Grid.Position,+dm: PiecesModel.Model<Tile.Tile8>,+ds: List<&2,U32>,+df: Tile.Tile8,+dt: Tensors.Status,+dw: U32,+da: Nat,+dl: U32,+dp: Grid.Position,+holds: LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp})),
  +gm: PiecesModel.Model<Tile.Tile8>,+gs: List<&2,U32>,+gf: Tile.Tile8,+gt: Tensors.Status,+gw: U32,+ga: Nat,+gl: U32,+gp: Grid.Position,+g_packed: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp})) ->
  LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.line_norm_back(betas,epsilon,gamma_velocity,beta_velocity,LineNorm.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp}))):
  LineFacts{+rank_room,+nonempty,+cap,+cap_ok,+count_room,+own_stretched,+row_stretched,+fits,+gamma_stretched} = facts
  line_back_holds(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,LineNorm.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp}),
    LineValues.backward(~X,~x_value,~X,~x_value,~DY,~dy_value,x,parameter_value(~X,gamma),dy,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,limits,default_capacity,rank_room,nonempty,cap,cap_ok,count_room,own_stretched,row_stretched,fits,gamma_stretched,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},y_packed,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp}),holds,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},g_packed),
    beta_contract,kept)

# backward of a layer normalization of the description on its tensors.
def LineBackwardOf(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,betas: Tiled.Tiled,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,
  gradient: Tiled.Tiled,xhat: Tiled.Tiled,deviation: Tiled.Tiled,gammas: Tiled.Tiled) -> Data:
  LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.line_norm_back(betas,epsilon,gamma_velocity,beta_velocity,LineNorm.backward(gradient,(xhat,deviation),gammas)))

def line_backward_tensors(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  -betas: Tiled.Tiled,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+kept: ParametersKept(~X,~x_value,gamma,beta,features,gamma_velocity,beta_velocity),
  -gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,shape,gradient),+y_observed: Public.Observed<gradient>,
  -xhat: Tiled.Tiled,+h_observed: Public.Observed<xhat>,-deviation: Tiled.Tiled,+d_observed: Public.Observed<deviation>,
  +holds: LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(xhat,deviation)),
  -gammas: Tiled.Tiled,+gamma_contract: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],gammas),+g_observed: Public.Observed<gammas>) ->
  LineBackwardOf(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,gradient,xhat,deviation,gammas):
  Public.Observed{+ym,+ys,+yf,+yt,+yw,+ya,+yl,+yp,+y_equation} = y_observed
  Public.Observed{+hm,+hs,+hf,+ht,+hw,+ha,+hl,+hp,+h_equation} = h_observed
  Public.Observed{+dm,+ds,+df,+dt,+dw,+da,+dl,+dp,+d_equation} = d_observed
  Public.Observed{+gm,+gs,+gf,+gt,+gw,+ga,+gl,+gp,+g_equation} = g_observed
  %y_equation : LineBackwardOf(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,_,xhat,deviation,gammas)
  %h_equation : LineBackwardOf(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},_,deviation,gammas)
  %d_equation : LineBackwardOf(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},_,gammas)
  %g_equation : LineBackwardOf(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,betas,gamma_velocity,beta_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},_)
  line_backward_packed(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,limits,default_capacity,facts,betas,gamma_velocity,beta_velocity,beta_contract,kept,
    ym,ys,yf,yt,yw,ya,yl,yp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp},gradient,y_equation) : Public.Contract(~DY,~dy_value,dy,shape,_) dy_contract : Public.Contract(~DY,~dy_value,dy,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,yt,yw},ya,yl,yp})},
    hm,hs,hf,ht,hw,ha,hl,hp,dm,ds,df,dt,dw,da,dl,dp,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},deviation,d_equation) : LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},_))
      {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},xhat,h_equation) : LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(_,deviation)) holds :
        LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},deviation))} :
      LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,hm),hs,hf,ht,hw},ha,hl,hp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp}))},
    gm,gs,gf,gt,gw,ga,gl,gp,{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp},gammas,g_equation) : Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],_) gamma_contract :
      Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gs,gf,gt,gw},ga,gl,gp})})

def line_backward_pair(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,shape,gradient),
  pair: Tiled.Tiled & Tiled.Tiled,+holds: LineValues.SavedHolds(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,pair),
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~GammaGraded,~BetaGraded,features,epsilon,gamma,beta,layer)) ->
  LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{pair},layer)):
  (xhat,deviation) = pair
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}}}} = held
      %Equal.sym(F32,layer_epsilon,epsilon,epsilon_exact) :
        LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{(xhat,deviation)},
          Train.LayerNorm{gammas,betas,_,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))
      line_backward_tensors(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,limits,default_capacity,facts,betas,gamma_velocity,beta_velocity,beta_contract,KernelGemm.Both{gamma_kept,beta_kept},
        gradient,dy_contract,Public.observe(gradient),xhat,Public.observe(xhat),deviation,Public.observe(deviation),holds,
        gammas,gamma_contract,Public.observe(gammas))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{(xhat,deviation)},Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{(xhat,deviation)},Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{(xhat,deviation)},Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.LineNormalized{(xhat,deviation)},Train.AveragePool{})),Words.false_true(held))

# backward of a layer normalization of the description, for dy of the
# contract and what forward saved, whose results are usable.
def line_backward(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~GammaGraded: Train.Gradient -> Data,~BetaGraded: Train.Gradient -> Data,+shape: List<&2,U32>,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Parameter<X>,+beta: Parameter<X>,+x: X,+dy: DY,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+facts: LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,shape,gradient),
  saved: Train.Saved,+kept: LineSaved(~X,~x_value,x,shape,epsilon,laxes,loffset,raxes,roffset,saved),
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~GammaGraded,~BetaGraded,features,epsilon,gamma,beta,layer)) ->
  LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,saved,layer)):
  match saved:
    case Train.LineNormalized{pair}:
      line_backward_pair(~DY,~dy_value,~X,~x_value,~GammaGraded,~BetaGraded,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,limits,default_capacity,facts,gradient,dy_contract,pair,kept,layer,held)
    case Train.Convolved{other_saved}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.Convolved{other_saved},layer)),Words.false_true(kept))
    case Train.Normalized{other_pair}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.Normalized{other_pair},layer)),Words.false_true(kept))
    case Train.Activated{other_input}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.Activated{other_input},layer)),Words.false_true(kept))
    case Train.Pooled{other_shape}:
      Empty.absurd(LineBackward(~DY,~dy_value,~X,~x_value,shape,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,x,dy,Train.backward(gradient,Train.Pooled{other_shape},layer)),Words.false_true(kept))

# ---- One step ----

# A parameter after one step of Optimizer.sgd_with for the gradient moved, and
# the velocity it leaves.
def ParameterStepped(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,+parameter: Parameter<X>,+moved: G,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>,result: Tiled.Tiled & Optimizer.Velocity) -> Data:
  OptimizerValues.Stepped(~X,~x_value,~OptimizerValues.Applied<OptimizerValues.Direction<X,G>,X>,
    ~OptimizerValues.applied_value(~OptimizerValues.Direction<X,G>,~OptimizerValues.direction_value(~X,~x_value,~G,~g_value),~X,~x_value),
    parameter_value(~X,parameter),OptimizerValues.Applied{F32.is_eq(momentum,0.0),parameter_moving(~X,parameter),momentum,
      OptimizerValues.Direction{F32.is_eq(decay,0.0),decay,parameter_value(~X,parameter),moved},parameter_velocity(~X,parameter)},
    rate,shape,Bool.not(F32.is_eq(momentum,0.0)),result)

def stepped_observed(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~Goal: Tiled.Tiled & Optimizer.Velocity -> Data,+parameter: Parameter<X>,+moved: G,+rate: F32,+momentum: F32,+decay: F32,
  +shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -tensor: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,parameter_value(~X,parameter),shape,tensor),+t_observed: Public.Observed<tensor>,
  -gradient: Tiled.Tiled,+graded: Public.Contract(~G,~g_value,moved,shape,gradient),+g_observed: Public.Observed<gradient>,
  -velocity: Optimizer.Velocity,+kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,parameter),shape,parameter_moving(~X,parameter),velocity),
  +v_observed: OptimizerValues.VelocityObserved<velocity>,
  goal: @+tm: PiecesModel.Model<Tile.Tile8> -> @+ts: List<&2,U32> -> @+tf: Tile.Tile8 -> @+tt: Tensors.Status -> @+tw: U32 -> @+ta: Nat -> @+tl: U32 -> @+tp: Grid.Position -> @+dm: PiecesModel.Model<Tile.Tile8> -> @+ds: List<&2,U32> -> @+df: Tile.Tile8 -> @+dt: Tensors.Status -> @+dw: U32 -> @+da: Nat -> @+dl: U32 -> @+dp: Grid.Position -> @+velocity_model: OptimizerValues.VelocityModel ->
    @+held: ParameterStepped(~X,~x_value,~G,~g_value,parameter,moved,rate,momentum,decay,shape,Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(velocity_model),rate,momentum,decay)) ->
    Goal(Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(velocity_model),rate,momentum,decay))) ->
  Goal(Optimizer.sgd_with(tensor,gradient,velocity,rate,momentum,decay)):
  Public.Observed{+tm,+ts,+tf,+tt,+tw,+ta,+tl,+tp,+t_equation} = t_observed
  Public.Observed{+dm,+ds,+df,+dt,+dw,+da,+dl,+dp,+d_equation} = g_observed
  OptimizerValues.VelocityObserved{+velocity_model,+v_equation} = v_observed
  %t_equation : Goal(Optimizer.sgd_with(_,gradient,velocity,rate,momentum,decay))
  %d_equation : Goal(Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},_,velocity,rate,momentum,decay))
  %v_equation : Goal(Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},_,rate,momentum,decay))
  goal(tm,ts,tf,tt,tw,ta,tl,tp,dm,ds,df,dt,dw,da,dl,dp,velocity_model,
    OptimizerValues.sgd_with(~X,~x_value,~G,~g_value,~X,~x_value,parameter_value(~X,parameter),moved,parameter_velocity(~X,parameter),rate,momentum,decay,shape,
      limits,default_capacity,nonempty,rank_room,parameter_moving(~X,parameter),
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},tensor,t_equation) : Public.Contract(~X,~x_value,parameter_value(~X,parameter),shape,_) contract :
        Public.Contract(~X,~x_value,parameter_value(~X,parameter),shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp})},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},{%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},gradient,d_equation) : Public.Contract(~G,~g_value,moved,shape,_) graded : Public.Contract(~G,~g_value,moved,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp})},
      OptimizerValues.velocity_of(velocity_model),
      {%Equal.sym(Optimizer.Velocity,OptimizerValues.velocity_of(velocity_model),velocity,v_equation) :
        OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,parameter),shape,parameter_moving(~X,parameter),_) kept :
        OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,parameter),shape,parameter_moving(~X,parameter),OptimizerValues.velocity_of(velocity_model))}))

# One step of a parameter whose gradient was computed: the goal of the stepped
# parameter, from the goal of each step on packed tensors.
def slot_stepped(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~Goal: Tiled.Tiled & Optimizer.Velocity -> Data,+parameter: Parameter<X>,+moved: G,+rate: F32,+momentum: F32,+decay: F32,
  +shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  tensor: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,parameter_value(~X,parameter),shape,tensor),
  gradient: Train.Gradient,+graded: Computed(~G,~g_value,moved,shape,gradient),
  velocity: Optimizer.Velocity,+kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,parameter),shape,parameter_moving(~X,parameter),velocity),
  goal: @+tm: PiecesModel.Model<Tile.Tile8> -> @+ts: List<&2,U32> -> @+tf: Tile.Tile8 -> @+tt: Tensors.Status -> @+tw: U32 -> @+ta: Nat -> @+tl: U32 -> @+tp: Grid.Position -> @+dm: PiecesModel.Model<Tile.Tile8> -> @+ds: List<&2,U32> -> @+df: Tile.Tile8 -> @+dt: Tensors.Status -> @+dw: U32 -> @+da: Nat -> @+dl: U32 -> @+dp: Grid.Position -> @+velocity_model: OptimizerValues.VelocityModel ->
    @+held: ParameterStepped(~X,~x_value,~G,~g_value,parameter,moved,rate,momentum,decay,shape,Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(velocity_model),rate,momentum,decay)) ->
    Goal(Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(velocity_model),rate,momentum,decay))) ->
  Goal(Train.stepped(tensor,gradient,velocity,Train.Descent{rate,momentum,decay})):
  match gradient:
    case Train.Held{computed}:
      stepped_observed(~X,~x_value,~G,~g_value,~Goal,parameter,moved,rate,momentum,decay,shape,limits,default_capacity,nonempty,rank_room,
        tensor,contract,Public.observe(tensor),computed,graded,Public.observe(computed),velocity,kept,OptimizerValues.observe_velocity(velocity),goal)
    case Train.Empty{}: Empty.absurd(Goal(Train.stepped(tensor,Train.Empty{},velocity,Train.Descent{rate,momentum,decay})),Words.false_true(graded))

# One axis fits every machine.
def one_room(+limits: Machine.Limits,+features: U32) -> {Nat.is_le(List.length(&2,U32,[features]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(1n,4n,U32.to_nat(Machine.capacity(limits)),{==},four_room(limits))

# ---- A step of a batch normalization ----

# After a step: gamma and beta stepped, the gradients cleared, the running
# statistics, epsilon and momentum as they were.
def NormStepped(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+epsilon: F32,+running_momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,layer: Train.Layer) -> Data:
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<{layer_momentum == running_momentum : F32},
        KernelGemm.Both<Cleared(gamma_gradient),KernelGemm.Both<Cleared(beta_gradient),
          KernelGemm.Both<ParameterStepped(~X,~x_value,~GG,~gg_value,gamma,gamma_moved,rate,momentum,decay,[channels],(gammas,gamma_velocity)),
            KernelGemm.Both<ParameterStepped(~X,~x_value,~BG,~bg_value,beta,beta_moved,rate,momentum,decay,[channels],(betas,beta_velocity)),
              KernelGemm.Both<Public.Contract(~R,~r_value,mean,[channels],means),Public.Contract(~R,~r_value,variance,[channels],variances)>>>>>>>
    case layer: {False{} == True{} : Bool}

def norm_stepped_holds(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+epsilon: F32,+running_momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,-means: Tiled.Tiled,-variances: Tiled.Tiled,+layer_epsilon: F32,+layer_momentum: F32,
  +epsilon_exact: {layer_epsilon == epsilon : F32},+momentum_exact: {layer_momentum == running_momentum : F32},
  +mean_contract: Public.Contract(~R,~r_value,mean,[channels],means),+variance_contract: Public.Contract(~R,~r_value,variance,[channels],variances),
  gamma_result: Tiled.Tiled & Optimizer.Velocity,+gamma_held: ParameterStepped(~X,~x_value,~GG,~gg_value,gamma,gamma_moved,rate,momentum,decay,[channels],gamma_result),
  beta_result: Tiled.Tiled & Optimizer.Velocity,+beta_held: ParameterStepped(~X,~x_value,~BG,~bg_value,beta,beta_moved,rate,momentum,decay,[channels],beta_result)) ->
  NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,gamma_result,beta_result)):
  (gammas,gamma_velocity) = gamma_result
  (betas,beta_velocity) = beta_result
  KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,KernelGemm.Both{{==},KernelGemm.Both{{==},
    KernelGemm.Both{gamma_held,KernelGemm.Both{beta_held,KernelGemm.Both{mean_contract,variance_contract}}}}}}}

# Train.sgd_step_with of a batch normalization whose gradients were computed.
def norm_step(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+epsilon: F32,+running_momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements([channels])) == True{} : Bool},
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~(gradient => Computed(~GG,~gg_value,gamma_moved,[channels],gradient)),
    ~(gradient => Computed(~BG,~bg_value,beta_moved,[channels],gradient)),channels,epsilon,running_momentum,gamma,beta,mean,variance,layer)) ->
  NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}},KernelGemm.Both{+mean_contract,+variance_contract}}}}} = held
      slot_stepped(~X,~x_value,~GG,~gg_value,~(stepped => NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,
          Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,stepped,Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay})))),
        gamma,gamma_moved,rate,momentum,decay,[channels],limits,default_capacity,nonempty,one_room(limits,channels),gammas,gamma_contract,gamma_gradient,gamma_graded,gamma_velocity,gamma_kept,
        tm => ts => tf => tt => tw => ta => tl => tp => dm => ds => df => dt => dw => da => dl => dp => gamma_model => gamma_held =>
          slot_stepped(~X,~x_value,~BG,~bg_value,~(stepped => NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,
              Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(gamma_model),rate,momentum,decay),stepped))),
            beta,beta_moved,rate,momentum,decay,[channels],limits,default_capacity,nonempty,one_room(limits,channels),betas,beta_contract,beta_gradient,beta_graded,beta_velocity,beta_kept,
            um => us => uf => ut => uw => ua => ul => up => em => es => ef => et => ew => ea => el => ep => beta_model => beta_held =>
              norm_stepped_holds(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,means,variances,layer_epsilon,layer_momentum,epsilon_exact,momentum_exact,mean_contract,variance_contract,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(gamma_model),rate,momentum,decay),gamma_held,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,um),us,uf,ut,uw},ua,ul,up},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,et,ew},ea,el,ep},OptimizerValues.velocity_of(beta_model),rate,momentum,decay),beta_held)))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(NormStepped(~X,~x_value,~R,~r_value,~GG,~gg_value,~BG,~bg_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))

def norm_cleared_slots(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,+channels: U32,+epsilon: F32,+running_momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,+rate: F32,+momentum: F32,+decay: F32,
  -gammas: Tiled.Tiled,gamma_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-betas: Tiled.Tiled,beta_gradient: Train.Gradient,-beta_velocity: Optimizer.Velocity,
  -means: Tiled.Tiled,-variances: Tiled.Tiled,+layer_epsilon: F32,+layer_momentum: F32,
  +epsilon_exact: {layer_epsilon == epsilon : F32},+momentum_exact: {layer_momentum == running_momentum : F32},
  +gamma_contract: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[channels],gammas),+gamma_graded: Cleared(gamma_gradient),
  +gamma_kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,gamma),[channels],parameter_moving(~X,gamma),gamma_velocity),
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[channels],betas),+beta_graded: Cleared(beta_gradient),
  +beta_kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,beta),[channels],parameter_moving(~X,beta),beta_velocity),
  +mean_contract: Public.Contract(~R,~r_value,mean,[channels],means),+variance_contract: Public.Contract(~R,~r_value,variance,[channels],variances)) ->
  NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,
    Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,Train.stepped(gammas,gamma_gradient,gamma_velocity,Train.Descent{rate,momentum,decay}),
      Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay}))):
  match gamma_gradient:
    case Train.Empty{}:
      match beta_gradient:
        case Train.Empty{}:
          KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
            KernelGemm.Both{KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}},KernelGemm.Both{mean_contract,variance_contract}}}}}
        case Train.Held{other}:
          Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,
            Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,(gammas,gamma_velocity),Train.stepped(betas,Train.Held{other},beta_velocity,Train.Descent{rate,momentum,decay}))),
            Words.false_true(beta_graded))
    case Train.Held{other}:
      Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,
        Train.norm_stepped(means,variances,layer_epsilon,layer_momentum,Train.stepped(gammas,Train.Held{other},gamma_velocity,Train.Descent{rate,momentum,decay}),
          Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay}))),Words.false_true(gamma_graded))

# Train.sgd_step_with of a batch normalization without gradients: the layer
# as it was.
def norm_step_cleared(~X: Data,~x_value: X -> U32 -> F32,~R: Data,~r_value: R -> U32 -> F32,+channels: U32,+epsilon: F32,+running_momentum: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+mean: R,+variance: R,+rate: F32,+momentum: F32,+decay: F32,
  layer: Train.Layer,+held: NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,layer)) ->
  NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}},KernelGemm.Both{+mean_contract,+variance_contract}}}}} = held
      norm_cleared_slots(~X,~x_value,~R,~r_value,channels,epsilon,running_momentum,gamma,beta,mean,variance,rate,momentum,decay,gammas,gamma_gradient,gamma_velocity,betas,beta_gradient,beta_velocity,
        means,variances,layer_epsilon,layer_momentum,epsilon_exact,momentum_exact,gamma_contract,gamma_graded,gamma_kept,beta_contract,beta_graded,beta_kept,
        mean_contract,variance_contract)
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,Train.sgd_step_with(Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,Train.sgd_step_with(Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(NormLayer(~X,~x_value,~R,~r_value,~Cleared,~Cleared,channels,epsilon,running_momentum,gamma,beta,mean,variance,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))

# ---- A step of a layer normalization ----

# After a step: gamma and beta stepped, the gradients cleared and epsilon as
# it was.
def LineStepped(~X: Data,~x_value: X -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,layer: Train.Layer) -> Data:
  match layer:
    case Train.LayerNorm{gammas,betas,layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both<{layer_epsilon == epsilon : F32},KernelGemm.Both<Cleared(gamma_gradient),KernelGemm.Both<Cleared(beta_gradient),
        KernelGemm.Both<ParameterStepped(~X,~x_value,~GG,~gg_value,gamma,gamma_moved,rate,momentum,decay,[features],(gammas,gamma_velocity)),
          ParameterStepped(~X,~x_value,~BG,~bg_value,beta,beta_moved,rate,momentum,decay,[features],(betas,beta_velocity))>>>>
    case layer: {False{} == True{} : Bool}

def line_stepped_holds(~X: Data,~x_value: X -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,+layer_epsilon: F32,+epsilon_exact: {layer_epsilon == epsilon : F32},
  gamma_result: Tiled.Tiled & Optimizer.Velocity,+gamma_held: ParameterStepped(~X,~x_value,~GG,~gg_value,gamma,gamma_moved,rate,momentum,decay,[features],gamma_result),
  beta_result: Tiled.Tiled & Optimizer.Velocity,+beta_held: ParameterStepped(~X,~x_value,~BG,~bg_value,beta,beta_moved,rate,momentum,decay,[features],beta_result)) ->
  LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.line_norm_stepped(layer_epsilon,gamma_result,beta_result)):
  (gammas,gamma_velocity) = gamma_result
  (betas,beta_velocity) = beta_result
  KernelGemm.Both{epsilon_exact,KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{gamma_held,beta_held}}}}

# Train.sgd_step_with of a layer normalization whose gradients were computed.
def line_step(~X: Data,~x_value: X -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+gamma_moved: GG,+beta_moved: BG,+rate: F32,+momentum: F32,+decay: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements([features])) == True{} : Bool},
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~(gradient => Computed(~GG,~gg_value,gamma_moved,[features],gradient)),
    ~(gradient => Computed(~BG,~bg_value,beta_moved,[features],gradient)),features,epsilon,gamma,beta,layer)) ->
  LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}}}} = held
      slot_stepped(~X,~x_value,~GG,~gg_value,~(stepped => LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,
          Train.line_norm_stepped(layer_epsilon,stepped,Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay})))),
        gamma,gamma_moved,rate,momentum,decay,[features],limits,default_capacity,nonempty,one_room(limits,features),gammas,gamma_contract,gamma_gradient,gamma_graded,gamma_velocity,gamma_kept,
        tm => ts => tf => tt => tw => ta => tl => tp => dm => ds => df => dt => dw => da => dl => dp => gamma_model => gamma_held =>
          slot_stepped(~X,~x_value,~BG,~bg_value,~(stepped => LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,
              Train.line_norm_stepped(layer_epsilon,Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(gamma_model),rate,momentum,decay),stepped))),
            beta,beta_moved,rate,momentum,decay,[features],limits,default_capacity,nonempty,one_room(limits,features),betas,beta_contract,beta_gradient,beta_graded,beta_velocity,beta_kept,
            um => us => uf => ut => uw => ua => ul => up => em => es => ef => et => ew => ea => el => ep => beta_model => beta_held =>
              line_stepped_holds(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,layer_epsilon,epsilon_exact,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(gamma_model),rate,momentum,decay),gamma_held,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,um),us,uf,ut,uw},ua,ul,up},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,et,ew},ea,el,ep},OptimizerValues.velocity_of(beta_model),rate,momentum,decay),beta_held)))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LineStepped(~X,~x_value,~GG,~gg_value,~BG,~bg_value,features,epsilon,gamma,beta,gamma_moved,beta_moved,rate,momentum,decay,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))

def line_cleared_slots(~X: Data,~x_value: X -> U32 -> F32,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+rate: F32,+momentum: F32,+decay: F32,
  -gammas: Tiled.Tiled,gamma_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-betas: Tiled.Tiled,beta_gradient: Train.Gradient,-beta_velocity: Optimizer.Velocity,
  +layer_epsilon: F32,+epsilon_exact: {layer_epsilon == epsilon : F32},
  +gamma_contract: Public.Contract(~X,~x_value,parameter_value(~X,gamma),[features],gammas),+gamma_graded: Cleared(gamma_gradient),
  +gamma_kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,gamma),[features],parameter_moving(~X,gamma),gamma_velocity),
  +beta_contract: Public.Contract(~X,~x_value,parameter_value(~X,beta),[features],betas),+beta_graded: Cleared(beta_gradient),
  +beta_kept: OptimizerValues.Kept(~X,~x_value,parameter_velocity(~X,beta),[features],parameter_moving(~X,beta),beta_velocity)) ->
  LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,
    Train.line_norm_stepped(layer_epsilon,Train.stepped(gammas,gamma_gradient,gamma_velocity,Train.Descent{rate,momentum,decay}),Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay}))):
  match gamma_gradient:
    case Train.Empty{}:
      match beta_gradient:
        case Train.Empty{}:
          KernelGemm.Both{epsilon_exact,KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
            KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}}}}
        case Train.Held{other}:
          Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,
            Train.line_norm_stepped(layer_epsilon,(gammas,gamma_velocity),Train.stepped(betas,Train.Held{other},beta_velocity,Train.Descent{rate,momentum,decay}))),Words.false_true(beta_graded))
    case Train.Held{other}:
      Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,
        Train.line_norm_stepped(layer_epsilon,Train.stepped(gammas,Train.Held{other},gamma_velocity,Train.Descent{rate,momentum,decay}),Train.stepped(betas,beta_gradient,beta_velocity,Train.Descent{rate,momentum,decay}))),
        Words.false_true(gamma_graded))

# Train.sgd_step_with of a layer normalization without gradients: the layer as
# it was.
def line_step_cleared(~X: Data,~x_value: X -> U32 -> F32,+features: U32,+epsilon: F32,+gamma: Parameter<X>,+beta: Parameter<X>,+rate: F32,+momentum: F32,+decay: F32,
  layer: Train.Layer,+held: LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,layer)) ->
  LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+gamma_contract,KernelGemm.Both{+gamma_graded,+gamma_kept}},
        KernelGemm.Both{+beta_contract,KernelGemm.Both{+beta_graded,+beta_kept}}}} = held
      line_cleared_slots(~X,~x_value,features,epsilon,gamma,beta,rate,momentum,decay,gammas,gamma_gradient,gamma_velocity,betas,beta_gradient,beta_velocity,layer_epsilon,epsilon_exact,
        gamma_contract,gamma_graded,gamma_kept,beta_contract,beta_graded,beta_kept)
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,Train.sgd_step_with(Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,Train.sgd_step_with(Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LineLayer(~X,~x_value,~Cleared,~Cleared,features,epsilon,gamma,beta,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))

# ---- Layers as models ----

# A tensor, a gradient, a layer or what forward saved whose tensors are the
# packed cells of models: a proof reads it as often as it needs.
type TensorModel is Data:
  TensorModel{model: PiecesModel.Model<Tile.Tile8>,shape: List<&2,U32>,fill: Tile.Tile8,status: Tensors.Status,workers: U32,axis: Nat,lanes: U32,position: Grid.Position}

def tensor_of(+model: TensorModel) -> Tiled.Tiled:
  match model:
    case TensorModel{+cells,+shape,+fill,+status,+workers,+axis,+lanes,+position}:
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cells),shape,fill,status,workers},axis,lanes,position}

type TensorObserved<-tensor: Tiled.Tiled> is Data:
  TensorObserved{model: TensorModel,equation: {tensor_of(model) == tensor : Tiled.Tiled}}

def tensor_observed(-tensor: Tiled.Tiled,+observed: Public.Observed<tensor>) -> TensorObserved<tensor>:
  Public.Observed{+model,+shape,+fill,+status,+workers,+axis,+lanes,+position,+equation} = observed
  TensorObserved{TensorModel{model,shape,fill,status,workers,axis,lanes,position},equation}

# Any tensor as the tensor of a model.
def observe_tensor(tensor: Tiled.Tiled) -> TensorObserved<tensor>:
  tensor_observed(tensor,Public.observe(tensor))

type PairModel is Data:
  PairModel{first: TensorModel,second: TensorModel}

def pair_of(+model: PairModel) -> Tiled.Tiled & Tiled.Tiled:
  match model:
    case PairModel{+first,+second}: (tensor_of(first),tensor_of(second))

type PairObserved<-pair: Tiled.Tiled & Tiled.Tiled> is Data:
  PairObserved{model: PairModel,equation: {pair_of(model) == pair : Tiled.Tiled & Tiled.Tiled}}

def pair_equation(+first_model: TensorModel,+second_model: TensorModel,-first: Tiled.Tiled,-second: Tiled.Tiled,
  +first_equation: {tensor_of(first_model) == first : Tiled.Tiled},+second_equation: {tensor_of(second_model) == second : Tiled.Tiled}) ->
  {pair_of(PairModel{first_model,second_model}) == (first,second) : Tiled.Tiled & Tiled.Tiled}:
  %first_equation : {pair_of(PairModel{first_model,second_model}) == (_,second) : Tiled.Tiled & Tiled.Tiled}
  %second_equation : {pair_of(PairModel{first_model,second_model}) == (tensor_of(first_model),_) : Tiled.Tiled & Tiled.Tiled}
  {==}

def pair_observed(-first: Tiled.Tiled,+first_observed: TensorObserved<first>,-second: Tiled.Tiled,+second_observed: TensorObserved<second>) ->
  PairObserved<(first,second)>:
  TensorObserved{+first_model,+first_equation} = first_observed
  TensorObserved{+second_model,+second_equation} = second_observed
  PairObserved{PairModel{first_model,second_model},pair_equation(first_model,second_model,first,second,first_equation,second_equation)}

def observe_pair(pair: Tiled.Tiled & Tiled.Tiled) -> PairObserved<pair>:
  (first,second) = pair
  pair_observed(first,observe_tensor(first),second,observe_tensor(second))

type GradientModel is Data:
  EmptyModel{}
  HeldModel{tensor: TensorModel}

def gradient_of(+model: GradientModel) -> Train.Gradient:
  match model:
    case EmptyModel{}: Train.Empty{}
    case HeldModel{+tensor}: Train.Held{tensor_of(tensor)}

type GradientObserved<-gradient: Train.Gradient> is Data:
  GradientObserved{model: GradientModel,equation: {gradient_of(model) == gradient : Train.Gradient}}

def held_equation(+model: TensorModel,-tensor: Tiled.Tiled,+equation: {tensor_of(model) == tensor : Tiled.Tiled}) ->
  {gradient_of(HeldModel{model}) == Train.Held{tensor} : Train.Gradient}:
  %equation : {gradient_of(HeldModel{model}) == Train.Held{_} : Train.Gradient}
  {==}

def held_observed(-tensor: Tiled.Tiled,+observed: TensorObserved<tensor>) -> GradientObserved<Train.Held{tensor}>:
  TensorObserved{+model,+equation} = observed
  GradientObserved{HeldModel{model},held_equation(model,tensor,equation)}

def observe_gradient(gradient: Train.Gradient) -> GradientObserved<gradient>:
  match gradient:
    case Train.Empty{}: GradientObserved{EmptyModel{},{==}}
    case Train.Held{tensor}: held_observed(tensor,observe_tensor(tensor))

type LayerModel is Data:
  ConvolutionModel{weights: TensorModel,bias: TensorModel,stride: U32,padding: U32,weight_gradient: GradientModel,bias_gradient: GradientModel,
    weight_velocity: OptimizerValues.VelocityModel,bias_velocity: OptimizerValues.VelocityModel}
  BatchNormModel{gamma: TensorModel,beta: TensorModel,running_mean: TensorModel,running_variance: TensorModel,epsilon: F32,momentum: F32,
    gamma_gradient: GradientModel,beta_gradient: GradientModel,gamma_velocity: OptimizerValues.VelocityModel,beta_velocity: OptimizerValues.VelocityModel}
  LayerNormModel{gamma: TensorModel,beta: TensorModel,epsilon: F32,gamma_gradient: GradientModel,beta_gradient: GradientModel,
    gamma_velocity: OptimizerValues.VelocityModel,beta_velocity: OptimizerValues.VelocityModel}
  ActivationModel{activation: Activations.Activation}
  AveragePoolModel{}

def layer_of(+model: LayerModel) -> Train.Layer:
  match model:
    case ConvolutionModel{+weights,+bias,+stride,+padding,+weight_gradient,+bias_gradient,+weight_velocity,+bias_velocity}:
      Train.Convolution{tensor_of(weights),tensor_of(bias),stride,padding,gradient_of(weight_gradient),gradient_of(bias_gradient),
        OptimizerValues.velocity_of(weight_velocity),OptimizerValues.velocity_of(bias_velocity)}
    case BatchNormModel{+gamma,+beta,+running_mean,+running_variance,+epsilon,+momentum,+gamma_gradient,+beta_gradient,+gamma_velocity,+beta_velocity}:
      Train.BatchNorm{tensor_of(gamma),tensor_of(beta),tensor_of(running_mean),tensor_of(running_variance),epsilon,momentum,gradient_of(gamma_gradient),
        gradient_of(beta_gradient),OptimizerValues.velocity_of(gamma_velocity),OptimizerValues.velocity_of(beta_velocity)}
    case LayerNormModel{+gamma,+beta,+epsilon,+gamma_gradient,+beta_gradient,+gamma_velocity,+beta_velocity}:
      Train.LayerNorm{tensor_of(gamma),tensor_of(beta),epsilon,gradient_of(gamma_gradient),gradient_of(beta_gradient),
        OptimizerValues.velocity_of(gamma_velocity),OptimizerValues.velocity_of(beta_velocity)}
    case ActivationModel{+activation}: Train.Activation{activation}
    case AveragePoolModel{}: Train.AveragePool{}

type LayerObserved<-layer: Train.Layer> is Data:
  LayerObserved{model: LayerModel,equation: {layer_of(model) == layer : Train.Layer}}

def convolution_equation(+wm: TensorModel,+bm: TensorModel,+stride: U32,+padding: U32,+wgm: GradientModel,+bgm: GradientModel,
  +wvm: OptimizerValues.VelocityModel,+bvm: OptimizerValues.VelocityModel,-weights: Tiled.Tiled,-bias: Tiled.Tiled,-weight_gradient: Train.Gradient,-bias_gradient: Train.Gradient,
  -weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,+w_equation: {tensor_of(wm) == weights : Tiled.Tiled},+b_equation: {tensor_of(bm) == bias : Tiled.Tiled},
  +wg_equation: {gradient_of(wgm) == weight_gradient : Train.Gradient},+bg_equation: {gradient_of(bgm) == bias_gradient : Train.Gradient},
  +wv_equation: {OptimizerValues.velocity_of(wvm) == weight_velocity : Optimizer.Velocity},+bv_equation: {OptimizerValues.velocity_of(bvm) == bias_velocity : Optimizer.Velocity}) ->
  {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity} : Train.Layer}:
  %w_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{_,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity} : Train.Layer}
  %b_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{tensor_of(wm),_,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity} : Train.Layer}
  %wg_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{tensor_of(wm),tensor_of(bm),stride,padding,_,bias_gradient,weight_velocity,bias_velocity} : Train.Layer}
  %bg_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{tensor_of(wm),tensor_of(bm),stride,padding,gradient_of(wgm),_,weight_velocity,bias_velocity} : Train.Layer}
  %wv_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{tensor_of(wm),tensor_of(bm),stride,padding,gradient_of(wgm),gradient_of(bgm),_,bias_velocity} : Train.Layer}
  %bv_equation : {layer_of(ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm}) == Train.Convolution{tensor_of(wm),tensor_of(bm),stride,padding,gradient_of(wgm),gradient_of(bgm),OptimizerValues.velocity_of(wvm),_} : Train.Layer}
  {==}

def convolution_observed(-weights: Tiled.Tiled,+wo: TensorObserved<weights>,-bias: Tiled.Tiled,+bo: TensorObserved<bias>,+stride: U32,+padding: U32,
  -weight_gradient: Train.Gradient,+wgo: GradientObserved<weight_gradient>,-bias_gradient: Train.Gradient,+bgo: GradientObserved<bias_gradient>,
  -weight_velocity: Optimizer.Velocity,+wvo: OptimizerValues.VelocityObserved<weight_velocity>,-bias_velocity: Optimizer.Velocity,+bvo: OptimizerValues.VelocityObserved<bias_velocity>) ->
  LayerObserved<Train.Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}>:
  TensorObserved{+wm,+w_equation} = wo
  TensorObserved{+bm,+b_equation} = bo
  GradientObserved{+wgm,+wg_equation} = wgo
  GradientObserved{+bgm,+bg_equation} = bgo
  OptimizerValues.VelocityObserved{+wvm,+wv_equation} = wvo
  OptimizerValues.VelocityObserved{+bvm,+bv_equation} = bvo
  LayerObserved{ConvolutionModel{wm,bm,stride,padding,wgm,bgm,wvm,bvm},
    convolution_equation(wm,bm,stride,padding,wgm,bgm,wvm,bvm,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,
      w_equation,b_equation,wg_equation,bg_equation,wv_equation,bv_equation)}

def norm_equation(+gm: TensorModel,+bm: TensorModel,+mm: TensorModel,+vm: TensorModel,+epsilon: F32,+momentum: F32,+ggm: GradientModel,+bgm: GradientModel,
  +gvm: OptimizerValues.VelocityModel,+bvm: OptimizerValues.VelocityModel,-gamma: Tiled.Tiled,-beta: Tiled.Tiled,-running_mean: Tiled.Tiled,-running_variance: Tiled.Tiled,
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +g_equation: {tensor_of(gm) == gamma : Tiled.Tiled},+b_equation: {tensor_of(bm) == beta : Tiled.Tiled},+m_equation: {tensor_of(mm) == running_mean : Tiled.Tiled},
  +v_equation: {tensor_of(vm) == running_variance : Tiled.Tiled},+gg_equation: {gradient_of(ggm) == gamma_gradient : Train.Gradient},
  +bg_equation: {gradient_of(bgm) == beta_gradient : Train.Gradient},+gv_equation: {OptimizerValues.velocity_of(gvm) == gamma_velocity : Optimizer.Velocity},
  +bv_equation: {OptimizerValues.velocity_of(bvm) == beta_velocity : Optimizer.Velocity}) ->
  {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) ==
    Train.BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}:
  %g_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{_,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %b_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),_,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %m_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),_,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %v_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),tensor_of(mm),_,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %gg_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),tensor_of(mm),tensor_of(vm),epsilon,momentum,_,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %bg_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),tensor_of(mm),tensor_of(vm),epsilon,momentum,gradient_of(ggm),_,gamma_velocity,beta_velocity} : Train.Layer}
  %gv_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),tensor_of(mm),tensor_of(vm),epsilon,momentum,gradient_of(ggm),gradient_of(bgm),_,beta_velocity} : Train.Layer}
  %bv_equation : {layer_of(BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm}) == Train.BatchNorm{tensor_of(gm),tensor_of(bm),tensor_of(mm),tensor_of(vm),epsilon,momentum,gradient_of(ggm),gradient_of(bgm),OptimizerValues.velocity_of(gvm),_} : Train.Layer}
  {==}

def norm_observed(-gamma: Tiled.Tiled,+go: TensorObserved<gamma>,-beta: Tiled.Tiled,+bo: TensorObserved<beta>,-running_mean: Tiled.Tiled,+mo: TensorObserved<running_mean>,
  -running_variance: Tiled.Tiled,+vo: TensorObserved<running_variance>,+epsilon: F32,+momentum: F32,
  -gamma_gradient: Train.Gradient,+ggo: GradientObserved<gamma_gradient>,-beta_gradient: Train.Gradient,+bgo: GradientObserved<beta_gradient>,
  -gamma_velocity: Optimizer.Velocity,+gvo: OptimizerValues.VelocityObserved<gamma_velocity>,-beta_velocity: Optimizer.Velocity,+bvo: OptimizerValues.VelocityObserved<beta_velocity>) ->
  LayerObserved<Train.BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}>:
  TensorObserved{+gm,+g_equation} = go
  TensorObserved{+bm,+b_equation} = bo
  TensorObserved{+mm,+m_equation} = mo
  TensorObserved{+vm,+v_equation} = vo
  GradientObserved{+ggm,+gg_equation} = ggo
  GradientObserved{+bgm,+bg_equation} = bgo
  OptimizerValues.VelocityObserved{+gvm,+gv_equation} = gvo
  OptimizerValues.VelocityObserved{+bvm,+bv_equation} = bvo
  LayerObserved{BatchNormModel{gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm},
    norm_equation(gm,bm,mm,vm,epsilon,momentum,ggm,bgm,gvm,bvm,gamma,beta,running_mean,running_variance,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,
      g_equation,b_equation,m_equation,v_equation,gg_equation,bg_equation,gv_equation,bv_equation)}

def line_equation(+gm: TensorModel,+bm: TensorModel,+epsilon: F32,+ggm: GradientModel,+bgm: GradientModel,
  +gvm: OptimizerValues.VelocityModel,+bvm: OptimizerValues.VelocityModel,-gamma: Tiled.Tiled,-beta: Tiled.Tiled,
  -gamma_gradient: Train.Gradient,-beta_gradient: Train.Gradient,-gamma_velocity: Optimizer.Velocity,-beta_velocity: Optimizer.Velocity,
  +g_equation: {tensor_of(gm) == gamma : Tiled.Tiled},+b_equation: {tensor_of(bm) == beta : Tiled.Tiled},
  +gg_equation: {gradient_of(ggm) == gamma_gradient : Train.Gradient},+bg_equation: {gradient_of(bgm) == beta_gradient : Train.Gradient},
  +gv_equation: {OptimizerValues.velocity_of(gvm) == gamma_velocity : Optimizer.Velocity},+bv_equation: {OptimizerValues.velocity_of(bvm) == beta_velocity : Optimizer.Velocity}) ->
  {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}:
  %g_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{_,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %b_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{tensor_of(gm),_,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %gg_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{tensor_of(gm),tensor_of(bm),epsilon,_,beta_gradient,gamma_velocity,beta_velocity} : Train.Layer}
  %bg_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{tensor_of(gm),tensor_of(bm),epsilon,gradient_of(ggm),_,gamma_velocity,beta_velocity} : Train.Layer}
  %gv_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{tensor_of(gm),tensor_of(bm),epsilon,gradient_of(ggm),gradient_of(bgm),_,beta_velocity} : Train.Layer}
  %bv_equation : {layer_of(LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm}) == Train.LayerNorm{tensor_of(gm),tensor_of(bm),epsilon,gradient_of(ggm),gradient_of(bgm),OptimizerValues.velocity_of(gvm),_} : Train.Layer}
  {==}

def line_observed(-gamma: Tiled.Tiled,+go: TensorObserved<gamma>,-beta: Tiled.Tiled,+bo: TensorObserved<beta>,+epsilon: F32,
  -gamma_gradient: Train.Gradient,+ggo: GradientObserved<gamma_gradient>,-beta_gradient: Train.Gradient,+bgo: GradientObserved<beta_gradient>,
  -gamma_velocity: Optimizer.Velocity,+gvo: OptimizerValues.VelocityObserved<gamma_velocity>,-beta_velocity: Optimizer.Velocity,+bvo: OptimizerValues.VelocityObserved<beta_velocity>) ->
  LayerObserved<Train.LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}>:
  TensorObserved{+gm,+g_equation} = go
  TensorObserved{+bm,+b_equation} = bo
  GradientObserved{+ggm,+gg_equation} = ggo
  GradientObserved{+bgm,+bg_equation} = bgo
  OptimizerValues.VelocityObserved{+gvm,+gv_equation} = gvo
  OptimizerValues.VelocityObserved{+bvm,+bv_equation} = bvo
  LayerObserved{LayerNormModel{gm,bm,epsilon,ggm,bgm,gvm,bvm},
    line_equation(gm,bm,epsilon,ggm,bgm,gvm,bvm,gamma,beta,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,
      g_equation,b_equation,gg_equation,bg_equation,gv_equation,bv_equation)}

# Any layer as the layer of a model.
def observe_layer(layer: Train.Layer) -> LayerObserved<layer>:
  match layer:
    case Train.Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      convolution_observed(weights,observe_tensor(weights),bias,observe_tensor(bias),stride,padding,weight_gradient,observe_gradient(weight_gradient),
        bias_gradient,observe_gradient(bias_gradient),weight_velocity,OptimizerValues.observe_velocity(weight_velocity),bias_velocity,OptimizerValues.observe_velocity(bias_velocity))
    case Train.BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      norm_observed(gamma,observe_tensor(gamma),beta,observe_tensor(beta),running_mean,observe_tensor(running_mean),running_variance,observe_tensor(running_variance),
        epsilon,momentum,gamma_gradient,observe_gradient(gamma_gradient),beta_gradient,observe_gradient(beta_gradient),
        gamma_velocity,OptimizerValues.observe_velocity(gamma_velocity),beta_velocity,OptimizerValues.observe_velocity(beta_velocity))
    case Train.LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      line_observed(gamma,observe_tensor(gamma),beta,observe_tensor(beta),epsilon,gamma_gradient,observe_gradient(gamma_gradient),beta_gradient,observe_gradient(beta_gradient),
        gamma_velocity,OptimizerValues.observe_velocity(gamma_velocity),beta_velocity,OptimizerValues.observe_velocity(beta_velocity))
    case Train.Activation{+activation}: LayerObserved{ActivationModel{activation},{==}}
    case Train.AveragePool{}: LayerObserved{AveragePoolModel{},{==}}

type SavedModel is Data:
  ConvolvedModel{pair: PairModel}
  NormalizedModel{pair: PairModel}
  LineNormalizedModel{pair: PairModel}
  ActivatedModel{input: TensorModel}
  PooledModel{shape: List<&2,U32>}

def saved_of(+model: SavedModel) -> Train.Saved:
  match model:
    case ConvolvedModel{+pair}: Train.Convolved{pair_of(pair)}
    case NormalizedModel{+pair}: Train.Normalized{pair_of(pair)}
    case LineNormalizedModel{+pair}: Train.LineNormalized{pair_of(pair)}
    case ActivatedModel{+input}: Train.Activated{tensor_of(input)}
    case PooledModel{+shape}: Train.Pooled{shape}

type SavedObserved<-saved: Train.Saved> is Data:
  SavedObserved{model: SavedModel,equation: {saved_of(model) == saved : Train.Saved}}

def saved_equation(~Part: Type,~make: Part -> Train.Saved,-made: Part,-part: Part,+equation: {made == part : Part}) -> {make(made) == make(part) : Train.Saved}:
  %equation : {make(made) == make(_) : Train.Saved}
  {==}

def convolved_observed(-pair: Tiled.Tiled & Tiled.Tiled,+observed: PairObserved<pair>) -> SavedObserved<Train.Convolved{pair}>:
  PairObserved{+model,+equation} = observed
  SavedObserved{ConvolvedModel{model},saved_equation(~(Tiled.Tiled & Tiled.Tiled),~(kept => Train.Convolved{kept}),pair_of(model),pair,equation)}

def normalized_observed(-pair: Tiled.Tiled & Tiled.Tiled,+observed: PairObserved<pair>) -> SavedObserved<Train.Normalized{pair}>:
  PairObserved{+model,+equation} = observed
  SavedObserved{NormalizedModel{model},saved_equation(~(Tiled.Tiled & Tiled.Tiled),~(kept => Train.Normalized{kept}),pair_of(model),pair,equation)}

def line_normalized_observed(-pair: Tiled.Tiled & Tiled.Tiled,+observed: PairObserved<pair>) -> SavedObserved<Train.LineNormalized{pair}>:
  PairObserved{+model,+equation} = observed
  SavedObserved{LineNormalizedModel{model},saved_equation(~(Tiled.Tiled & Tiled.Tiled),~(kept => Train.LineNormalized{kept}),pair_of(model),pair,equation)}

def activated_observed(-input: Tiled.Tiled,+observed: TensorObserved<input>) -> SavedObserved<Train.Activated{input}>:
  TensorObserved{+model,+equation} = observed
  SavedObserved{ActivatedModel{model},saved_equation(~Tiled.Tiled,~(kept => Train.Activated{kept}),tensor_of(model),input,equation)}

# Anything forward saved as what a model saved.
def observe_saved(saved: Train.Saved) -> SavedObserved<saved>:
  match saved:
    case Train.Convolved{pair}: convolved_observed(pair,observe_pair(pair))
    case Train.Normalized{pair}: normalized_observed(pair,observe_pair(pair))
    case Train.LineNormalized{pair}: line_normalized_observed(pair,observe_pair(pair))
    case Train.Activated{input}: activated_observed(input,observe_tensor(input))
    case Train.Pooled{+shape}: SavedObserved{PooledModel{shape},{==}}
