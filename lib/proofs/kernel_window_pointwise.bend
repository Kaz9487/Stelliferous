# A convolution the windowed pipeline runs pointwise: a kernel of one at
# stride one without padding and an inner extent of whole groups of sixteen.
import Base
import ../kernel_window_pipeline.bend as WindowPipeline
import ../kernel_gemm.bend as Gemm
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./u32_extrema.bend as Extrema
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ../math_activation.bend as Activations

type PointFacts<-kernel: U32,-stride: U32,-padding: U32,-inner: U32> is Data:
  PointFacts{kernel_one: {kernel == 1 : U32},stride_one: {stride == 1 : U32},padding_zero: {padding == 0 : U32},
    whole: {(inner % 16 : U32) == 0 : U32}}

def nat_zero(+value: Nat,small: {Nat.is_le(value,0n) == True{} : Bool}) -> {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(small))

# A word below a zero word is zero.
def below_zero(+word: U32,+upper: U32,+ordered: {U32.is_le(word,upper) == True{} : Bool},+zero: {upper == 0 : U32}) -> {word == 0 : U32}:
  Arithmetic.to_nat_injective(word,0,nat_zero(U32.to_nat(word),
    %zero : {Nat.is_le(U32.to_nat(word),U32.to_nat(_)) == True{} : Bool}
    Bounds.as_nat_le(word,upper,ordered)))

# Both words of a zero maximum are zero.
def zero_maximum(+left: U32,+right: U32,+maximum: {U32.max(left,right) == 0 : U32}) -> Gemm.Both<{left == 0 : U32},{right == 0 : U32}>:
  Gemm.Both{below_zero(left,U32.max(left,right),Extrema.maximum_left(left,right),maximum),below_zero(right,U32.max(left,right),Extrema.maximum_right(left,right),maximum)}

# A word one past zero is one.
def one_of(+word: U32,+zero: {(word - 1 : U32) == 0 : U32}) -> {word == 1 : U32}:
  Equal.trans(U32,word,((word - 1 : U32) + 1 : U32),1,Equal.sym(U32,((word - 1 : U32) + 1 : U32),word,Subtraction.add_subtract(word,1)),
    Equal.cong(U32,U32,value => (value + 1 : U32),(word - 1 : U32),0,zero))

def facts_words(+kernel: U32,+stride: U32,+padding: U32,+inner: U32,
  steps: Gemm.Both<{(kernel - 1 : U32) == 0 : U32},{(stride - 1 : U32) == 0 : U32}>,rest: Gemm.Both<{padding == 0 : U32},{(inner % 16 : U32) == 0 : U32}>) ->
  PointFacts<kernel,stride,padding,inner>:
  Gemm.Both{+kernel_zero,+stride_zero} = steps
  Gemm.Both{+padding_zero,+whole} = rest
  PointFacts{one_of(kernel,kernel_zero),one_of(stride,stride_zero),padding_zero,whole}

def facts_halves(+kernel: U32,+stride: U32,+padding: U32,+inner: U32,
  halves: Gemm.Both<{U32.max((kernel - 1 : U32),(stride - 1 : U32)) == 0 : U32},{U32.max(padding,(inner % 16 : U32)) == 0 : U32}>) ->
  PointFacts<kernel,stride,padding,inner>:
  Gemm.Both{+steps,+rest} = halves
  facts_words(kernel,stride,padding,inner,zero_maximum((kernel - 1 : U32),(stride - 1 : U32),steps),zero_maximum(padding,(inner % 16 : U32),rest))

def facts_picked(+kernel: U32,+stride: U32,+padding: U32,+inner: U32,
  +zero: {U32.is_zero(U32.max(U32.max((kernel - 1 : U32),(stride - 1 : U32)),U32.max(padding,(inner % 16 : U32)))) == True{} : Bool}) ->
  PointFacts<kernel,stride,padding,inner>:
  facts_halves(kernel,stride,padding,inner,
    zero_maximum(U32.max((kernel - 1 : U32),(stride - 1 : U32)),U32.max(padding,(inner % 16 : U32)),
      Arithmetic.word_zero(U32.max(U32.max((kernel - 1 : U32),(stride - 1 : U32)),U32.max(padding,(inner % 16 : U32))),zero)))

def pointwise_facts(+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+output_height: U32,+output_width: U32,+channels: U32,+inner: U32,
  +input_groups: U32,+output_groups: U32,+activation: Activations.Activation,
  +direct: {WindowPipeline.pointwise(WindowPipeline.Convolution{height,width,kernel,stride,padding,output_height,output_width,channels,inner,input_groups,output_groups,activation}) == True{} : Bool}) ->
  PointFacts<kernel,stride,padding,inner>:
  facts_picked(kernel,stride,padding,inner,direct)
