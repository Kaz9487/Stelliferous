# The images of a batch as convolution problems, and the value of every
# output element: the activation of the matrix value of its image.
import Base
import ../specification.bend as ConvSpec
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./input_storage.bend as Inputs
import ./output_values.bend as Values
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues

# The values of count elements of the input from element index on.
def image_values(~Logical: Data,~original: Logical -> U32 -> F32,count: Nat,+index: Nat,+logical: Logical) -> List<&2,F32>:
  match count:
    case 0n: []
    case 1n+rest: original(logical,U32.from_nat(index)) <> image_values(~Logical,~original,rest,1n+index,logical)

def listed(~Logical: Data,~original: Logical -> U32 -> F32,count: Nat,+index: Nat,+logical: Logical) ->
  {image_values(~Logical,~original,count,index,logical) == Contract.listed(~F32,~Logical,~original,count,index,logical) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => original(logical,U32.from_nat(index)) <> tail,
        image_values(~Logical,~original,rest,1n+index,logical),Contract.listed(~F32,~Logical,~original,rest,1n+index,logical),listed(~Logical,~original,rest,1n+index,logical))

def image_length(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical) ->
  {ConvSpec.length(image_values(~Logical,~original,count,index,logical)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,n => 1n+n,ConvSpec.length(image_values(~Logical,~original,rest,1n+index,logical)),rest,image_length(~Logical,~original,rest,1n+index,logical))

def image_get(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical,+position: Nat,
  +inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position) == Some{original(logical,U32.from_nat(Nat.add(index,position)))} : Maybe<&2,F32>}:
  match count position:
    case 0n 0n:
      Empty.absurd({None{} == Some{original(logical,U32.from_nat(Nat.add(index,0n)))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 0n 1n+previous:
      Empty.absurd({None{} == Some{original(logical,U32.from_nat(Nat.add(index,1n+previous)))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {Some{original(logical,U32.from_nat(index))} == Some{original(logical,U32.from_nat(_))} : Maybe<&2,F32>}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {List.get(&2,F32,image_values(~Logical,~original,rest,1n+index,logical),previous) == Some{original(logical,U32.from_nat(_))} : Maybe<&2,F32>}
      image_get(~Logical,~original,rest,1n+index,logical,previous,inside)

def image_at(~Logical: Data,~original: Logical -> U32 -> F32,+count: Nat,+index: Nat,+logical: Logical,+position: Nat,
  +inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {ConvSpec.at(position,image_values(~Logical,~original,count,index,logical)) == original(logical,U32.from_nat(Nat.add(index,position))) : F32}:
  %Equal.sym(F32,ConvSpec.at(position,image_values(~Logical,~original,count,index,logical)),
    Maybe.default(&2,F32,List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position),0.0),
    Inputs.lookup_matches_list(position,image_values(~Logical,~original,count,index,logical))) : {_ == original(logical,U32.from_nat(Nat.add(index,position))) : F32}
  %Equal.sym(Maybe<&2,F32>,List.get(&2,F32,image_values(~Logical,~original,count,index,logical),position),Some{original(logical,U32.from_nat(Nat.add(index,position)))},
    image_get(~Logical,~original,count,index,logical,position,inside)) :
    {Maybe.default(&2,F32,_,0.0) == original(logical,U32.from_nat(Nat.add(index,position))) : F32}
  {==}

type Context<-Logical: Data> is Data:
  Context{logical: Logical,activation: Activations.Activation,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,
    weights: List<&2,F32>,bias: List<&2,F32>}

# Image `image` of the input as a convolution problem.
def problem(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) -> ConvSpec.Problem:
  +area = Nat.mul(channels,Nat.mul(height,width))
  ConvSpec.Problem{ConvSpec.Shape{channels,outputs,height,width,kernel,stride,padding},
    image_values(~Logical,~original,area,Nat.mul(image,area),logical),weights,bias}

def at_position(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+channel: Nat,+position: Nat) -> F32:
  +output_width = ConvSpec.axis_size(width,kernel,stride,padding)
  ActivationProofs.activation_value(~custom,activation,
    Values.matrix_value(problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),channel,
      Nat.div(position,output_width),Nat.mod(position,output_width)))

def plane(+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat) -> Nat:
  Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding))

def at_rest(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+rest: Nat) -> F32:
  +area = plane(height,width,kernel,stride,padding)
  at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,Nat.div(rest,area),Nat.mod(rest,area))

def value_at(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+flat: Nat) -> F32:
  +volume = Nat.mul(outputs,plane(height,width,kernel,stride,padding))
  at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume))

def value(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,context: Context<Logical>,index: U32) -> F32:
  Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,U32.to_nat(index))

# An output index built from (image, channel, position) decodes to them.
def decoded(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+channel: Nat,+position: Nat,
  +channel_inside: {Nat.is_lt(channel,outputs) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,plane(height,width,kernel,stride,padding)) == True{} : Bool}) ->
  {value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,
      Nat.add(Nat.mul(image,Nat.mul(outputs,plane(height,width,kernel,stride,padding))),Nat.add(Nat.mul(channel,plane(height,width,kernel,stride,padding)),position))) ==
    at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position) : F32}:
  +area = plane(height,width,kernel,stride,padding)
  +volume = Nat.mul(outputs,area)
  +inner = Nat.add(Nat.mul(channel,area),position)
  +flat = Nat.add(Nat.mul(image,volume),inner)
  +area_positive = Order.le_lt_trans(0n,position,area,Order.zero_le(position),position_inside)
  +inner_inside = RowMajor.radix(channel,outputs,position,area,channel_inside,position_inside)
  +volume_positive = Order.le_lt_trans(0n,inner,volume,Order.zero_le(inner),inner_inside)
  Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume)),
    at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
    at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position),
    Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume)),
      at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,Nat.mod(flat,volume)),
      at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
      Equal.cong(Nat,F32,value => at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,value,Nat.mod(flat,volume)),
        Nat.div(flat,volume),image,Radix.quotient(image,inner,volume,volume_positive,inner_inside)),
      Equal.cong(Nat,F32,value => at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,value),
        Nat.mod(flat,volume),inner,Radix.remainder(image,inner,volume,volume_positive,inner_inside))),
    Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
      at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,Nat.mod(inner,area)),
      at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position),
      Equal.cong(Nat,F32,value => at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,value,Nat.mod(inner,area)),
        Nat.div(inner,area),channel,Radix.quotient(channel,position,area,area_positive,position_inside)),
      Equal.cong(Nat,F32,value => at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,value),
        Nat.mod(inner,area),position,Radix.remainder(channel,position,area,area_positive,position_inside))))
