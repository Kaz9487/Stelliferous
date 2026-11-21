# The products of a matrix: each row of left [count,rows,inner] with each row
# of a shared weight [outputs,inner], or with each column of the matrix of its
# image of right [count,inner,outputs], added in increasing inner index from
# +0.0, then the bias of the output. They are the values of the convolution of
# a one-cell window over each image of left read at the exchanged index: its
# inner extent the channels, its rows the positions of a column, the weights
# those of its image.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/math_activation.bend as Activations
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_tiled_swapped.bend as Swapped
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./image_problems.bend as ImageProblems
import ./output_values.bend as Values

# ---- The products ----

# The products of left at left_base + k and weight at weight_base + k for k from
# column, added to the accumulator in increasing k.
def ordered_sum(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,remaining: Nat,+column: Nat,
  +left: Left,+left_base: Nat,+weight: Weight,+weight_base: Nat,+accumulator: F32) -> F32:
  match remaining:
    case 0n: accumulator
    case 1n+rest:
      ordered_sum(~Left,~left_value,~Weight,~weight_value,rest,1n+column,left,left_base,weight,weight_base,
        (accumulator + left_value(left,U32.from_nat(Nat.add(left_base,column))) * weight_value(weight,U32.from_nat(Nat.add(weight_base,column))) : F32))

# Left [count,rows,inner] by a weight [outputs,inner] shared by every row, and a
# bias [outputs].
type Shared<-Left: Data,-Weight: Data,-Bias: Data> is Data:
  Shared{left: Left,weight: Weight,bias: Bias,inner: Nat,outputs: Nat}

# Index i of [count,rows,outputs]: the ordered sum of row i / outputs of left with
# the weight row of output i % outputs from +0.0, then the bias of that output.
def shared_value(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,context: Shared<Left,Weight,Bias>,index: U32) -> F32:
  Shared{+left,+weight,+bias,+inner,+outputs} = context
  +position = U32.to_nat(index)
  +output = Nat.mod(position,outputs)
  (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.div(position,outputs),inner),weight,Nat.mul(output,inner),0.0) +
    bias_value(bias,U32.from_nat(output)) : F32)

# ---- The one-cell window ----

# A window of one cell keeps the extent of an axis.
def unit_axis(+size: Nat,+positive: {Nat.is_lt(0n,size) == True{} : Bool}) -> {ConvSpec.axis_size(size,1n,1n,0n) == size : Nat}:
  match size:
    case 0n: Empty.absurd({ConvSpec.axis_size(0n,1n,1n,0n) == 0n : Nat},Order.below_zero(0n,positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(1n+rest,0n),1n+rest,Addition.zero_right(1n+rest)) :
        {1n+Nat.div(Nat.sub(_,1n),1n) == 1n+rest : Nat}
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {1n+Nat.div(_,1n) == 1n+rest : Nat}
      Equal.cong(Nat & Nat,Nat,pair => 1n+Pair.fst(Nat,Nat,pair),Nat.divmod(rest,1n),(rest,0n),Swapped.by_one(rest))

# The positions of a column of rows under a one-cell window.
def plane_exact(+rows: Nat,+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool}) -> {ImageProblems.plane(rows,1n,1n,1n,0n) == rows : Nat}:
  Equal.trans(Nat,Nat.mul(ConvSpec.axis_size(rows,1n,1n,0n),1n),ConvSpec.axis_size(rows,1n,1n,0n),rows,
    Order.times_one(ConvSpec.axis_size(rows,1n,1n,0n)),unit_axis(rows,rows_positive))

# Image `image` of left as the input [inner,rows,1] of a one-cell window.
def window(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+inner: Nat,+outputs: Nat,+rows: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) -> ConvSpec.Problem:
  ImageProblems.problem(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},inner,outputs,rows,1n,1n,1n,0n,weights,bias,image)

# The window reads the left value of its row and channel.
def sample_read(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+input_room: {Nat.is_le(Nat.mul(Nat.mul(count,inner),rows),U32.to_nat(cap)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +image: Nat,+channel: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,inner) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {MatrixSpec.patch(window(~Left,~left_value,left,inner,outputs,rows,weights,bias,image),MatrixSpec.Position{row,0n},MatrixSpec.KIndex{channel,0n,0n}) ==
    left_value(left,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),channel))) : F32}:
  +logical = {Swapped.Swapped{left,inner,rows} : Swapped.Swapped<Left>}
  +target = left_value(left,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),channel)))
  +unit_row = {Equal.trans(Nat,Nat.add(Nat.mul(row,1n),0n),Nat.mul(row,1n),row,Addition.zero_right(Nat.mul(row,1n)),Order.times_one(row)) : {Nat.add(Nat.mul(row,1n),0n) == row : Nat}}
  +plane = Nat.mul(inner,rows)
  +position = Nat.add(Nat.mul(channel,rows),row)
  %Equal.sym(Nat,Nat.add(Nat.mul(row,1n),0n),row,unit_row) :
    {ConvSpec.sample_if(Nat.is_le(0n,_) && Nat.is_lt(_,Nat.add(0n,rows)) && Nat.is_le(0n,0n) && Nat.is_lt(0n,Nat.add(0n,1n)),
      Nat.add(Nat.mul(channel,Nat.mul(rows,1n)),Nat.add(Nat.mul(Nat.sub(_,0n),1n),Nat.sub(0n,0n))),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,Nat.mul(rows,1n)),Nat.mul(image,Nat.mul(inner,Nat.mul(rows,1n))),logical)) == target : F32}
  %Equal.sym(Bool,Nat.is_le(0n,row),True{},Order.zero_le(row)) :
    {ConvSpec.sample_if(_ && Nat.is_lt(row,Nat.add(0n,rows)) && Nat.is_le(0n,0n) && Nat.is_lt(0n,Nat.add(0n,1n)),
      Nat.add(Nat.mul(channel,Nat.mul(rows,1n)),Nat.add(Nat.mul(Nat.sub(row,0n),1n),Nat.sub(0n,0n))),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,Nat.mul(rows,1n)),Nat.mul(image,Nat.mul(inner,Nat.mul(rows,1n))),logical)) == target : F32}
  %Equal.sym(Bool,Nat.is_lt(row,rows),True{},row_inside) :
    {ConvSpec.sample_if(True{} && _ && Nat.is_le(0n,0n) && Nat.is_lt(0n,Nat.add(0n,1n)),
      Nat.add(Nat.mul(channel,Nat.mul(rows,1n)),Nat.add(Nat.mul(Nat.sub(row,0n),1n),Nat.sub(0n,0n))),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,Nat.mul(rows,1n)),Nat.mul(image,Nat.mul(inner,Nat.mul(rows,1n))),logical)) == target : F32}
  %Equal.sym(Nat,Nat.sub(row,0n),row,Order.subtract_zero(row)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(rows,1n)),Nat.add(Nat.mul(_,1n),0n)),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,Nat.mul(rows,1n)),Nat.mul(image,Nat.mul(inner,Nat.mul(rows,1n))),logical)) == target : F32}
  %Equal.sym(Nat,Nat.add(Nat.mul(row,1n),0n),row,unit_row) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(rows,1n)),_),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,Nat.mul(rows,1n)),Nat.mul(image,Nat.mul(inner,Nat.mul(rows,1n))),logical)) == target : F32}
  %Equal.sym(Nat,Nat.mul(rows,1n),rows,Order.times_one(rows)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,_),row),ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Nat.mul(inner,_),Nat.mul(image,Nat.mul(inner,_)),logical)) == target : F32}
  Equal.trans(F32,ConvSpec.at(position,ImageProblems.image_values(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),plane,Nat.mul(image,plane),logical)),
    Swapped.swapped_value(~Left,~left_value,logical,U32.from_nat(RowMajor.index(image,inner,rows,channel,row))),target,
    ImageProblems.image_at(~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),plane,Nat.mul(image,plane),logical,position,RowMajor.radix(channel,inner,row,rows,channel_inside,row_inside)),
    %RowMajor.mixed(image,inner,rows,channel,row) : {Swapped.swapped_value(~Left,~left_value,logical,U32.from_nat(_)) == target : F32}
    Swapped.into_values(~Left,~left_value,left,count,inner,rows,cap,cap_ok,input_room,inner_positive,rows_positive,image,channel,row,image_inside,channel_inside,row_inside))

# The window reads the weight of its output and channel from the weights of its
# image.
def weight_read(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,+left: Left,+inner: Nat,+outputs: Nat,+rows: Nat,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+weight: Weight,+weight_count: Nat,+weight_base: Nat,
  +weights_at: {weights == Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight) : List<&2,F32>},
  +weights_room: {Nat.is_le(Nat.mul(outputs,inner),weight_count) == True{} : Bool},+output: Nat,+channel: Nat,+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,inner) == True{} : Bool}) ->
  {MatrixSpec.weight(window(~Left,~left_value,left,inner,outputs,rows,weights,bias,image),output,MatrixSpec.KIndex{channel,0n,0n}) == weight_value(weight,U32.from_nat(Nat.add(weight_base,Nat.add(Nat.mul(output,inner),channel)))) : F32}:
  +line = Nat.add(Nat.mul(output,inner),channel)
  +unit = {Equal.trans(Nat,Nat.add(Nat.mul(line,1n),0n),Nat.mul(line,1n),line,Addition.zero_right(Nat.mul(line,1n)),Order.times_one(line)) : {Nat.add(Nat.mul(line,1n),0n) == line : Nat}}
  +target = weight_value(weight,U32.from_nat(Nat.add(weight_base,line)))
  %Equal.sym(Nat,Nat.add(Nat.mul(line,1n),0n),line,unit) : {ConvSpec.at(Nat.add(Nat.mul(_,1n),0n),weights) == target : F32}
  %Equal.sym(Nat,Nat.add(Nat.mul(line,1n),0n),line,unit) : {ConvSpec.at(_,weights) == target : F32}
  %Equal.sym(List<&2,F32>,weights,Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight),weights_at) : {ConvSpec.at(line,_) == target : F32}
  %ImageProblems.listed(~Weight,~weight_value,weight_count,weight_base,weight) : {ConvSpec.at(line,_) == target : F32}
  ImageProblems.image_at(~Weight,~weight_value,weight_count,weight_base,weight,line,
    Bounds.nat_trans(line,Nat.mul(outputs,inner),weight_count,RowMajor.radix(output,outputs,channel,inner,output_inside,channel_inside),weights_room))

# The window reads the bias of its output.
def bias_read(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+inner: Nat,+outputs: Nat,+rows: Nat,+weights: List<&2,F32>,+bias: Bias,+bias_count: Nat,+image: Nat,+output: Nat,
  +output_inside: {Nat.is_lt(output,bias_count) == True{} : Bool}) ->
  {ConvSpec.bias_at(window(~Left,~left_value,left,inner,outputs,rows,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image),output) == bias_value(bias,U32.from_nat(output)) : F32}:
  %ImageProblems.listed(~Bias,~bias_value,bias_count,0n,bias) : {ConvSpec.at(output,_) == bias_value(bias,U32.from_nat(output)) : F32}
  ImageProblems.image_at(~Bias,~bias_value,bias_count,0n,bias,output,output_inside)

# The dot of the window over the channels from channel on is the ordered sum.
def dot_sum(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weight: Weight,+weight_count: Nat,+biases: List<&2,F32>,+weights: List<&2,F32>,+weight_base: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+input_room: {Nat.is_le(Nat.mul(Nat.mul(count,inner),rows),U32.to_nat(cap)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+weights_room: {Nat.is_le(Nat.mul(outputs,inner),weight_count) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+weights_at: {weights == Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight) : List<&2,F32>},
  remaining: Nat,+channel: Nat,+fits: {Nat.is_le(Nat.add(remaining,channel),inner) == True{} : Bool},+accumulator: F32) ->
  {Values.matrix_dot(MatrixSpec.kchannels(remaining,channel,1n),window(~Left,~left_value,left,inner,outputs,rows,weights,biases,image),output,row,0n,accumulator) == ordered_sum(~Left,~left_value,~Weight,~weight_value,remaining,channel,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),accumulator) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +problem = window(~Left,~left_value,left,inner,outputs,rows,weights,biases,image)
      +reduction = {MatrixSpec.KIndex{channel,0n,0n} : MatrixSpec.Reduction}
      +channel_inside = Bounds.nat_trans(channel,1n+Nat.add(rest,channel),inner,Order.lt_succ(channel,Nat.add(rest,channel),Order.right_le(rest,channel)),fits)
      +read_left = left_value(left,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),channel)))
      +read_weight = weight_value(weight,U32.from_nat(Nat.add(Nat.add(weight_base,Nat.mul(output,inner)),channel)))
      +weight_exact = {Equal.trans(F32,MatrixSpec.weight(problem,output,reduction),weight_value(weight,U32.from_nat(Nat.add(weight_base,Nat.add(Nat.mul(output,inner),channel)))),read_weight,
          weight_read(~Left,~left_value,~Weight,~weight_value,left,inner,outputs,rows,weights,biases,image,weight,weight_count,weight_base,weights_at,weights_room,output,channel,output_inside,channel_inside),
          Equal.cong(Nat,F32,position => weight_value(weight,U32.from_nat(position)),Nat.add(weight_base,Nat.add(Nat.mul(output,inner),channel)),Nat.add(Nat.add(weight_base,Nat.mul(output,inner)),channel),
            Equal.sym(Nat,Nat.add(Nat.add(weight_base,Nat.mul(output,inner)),channel),Nat.add(weight_base,Nat.add(Nat.mul(output,inner),channel)),Addition.associative(weight_base,Nat.mul(output,inner),channel)))) :
        {MatrixSpec.weight(problem,output,reduction) == read_weight : F32}}
      +next_fits = {%Equal.sym(Nat,Nat.add(rest,1n+channel),1n+Nat.add(rest,channel),Addition.successor_right(rest,channel)) : {Nat.is_le(_,inner) == True{} : Bool}
        fits : {Nat.is_le(Nat.add(rest,1n+channel),inner) == True{} : Bool}}
      +next = (accumulator + read_left * read_weight : F32)
      %Equal.sym(F32,MatrixSpec.patch(problem,MatrixSpec.Position{row,0n},reduction),read_left,
          sample_read(~Left,~left_value,left,count,inner,outputs,rows,weights,biases,cap,cap_ok,input_room,inner_positive,rows_positive,image,channel,row,image_inside,channel_inside,row_inside)) :
        {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(rest,1n+channel,1n),output,MatrixSpec.Position{row,0n},problem,(accumulator + _ * MatrixSpec.weight(problem,output,reduction) : F32)) ==
          ordered_sum(~Left,~left_value,~Weight,~weight_value,rest,1n+channel,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),next) : F32}
      %Equal.sym(F32,MatrixSpec.weight(problem,output,reduction),read_weight,weight_exact) :
        {OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~MatrixSpec.patch,~MatrixSpec.weight,MatrixSpec.kchannels(rest,1n+channel,1n),output,MatrixSpec.Position{row,0n},problem,(accumulator + read_left * _ : F32)) ==
          ordered_sum(~Left,~left_value,~Weight,~weight_value,rest,1n+channel,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),next) : F32}
      dot_sum(~Left,~left_value,~Weight,~weight_value,left,count,inner,outputs,rows,weight,weight_count,biases,weights,weight_base,cap,cap_ok,input_room,inner_positive,rows_positive,weights_room,image,output,row,image_inside,output_inside,row_inside,weights_at,
        rest,1n+channel,next_fits,next)

# The matrix value of the window at its output and row: the ordered sum, then the bias.
def matrix_read(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weight: Weight,+weight_count: Nat,+bias: Bias,+bias_count: Nat,+weights: List<&2,F32>,+weight_base: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+input_room: {Nat.is_le(Nat.mul(Nat.mul(count,inner),rows),U32.to_nat(cap)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+weights_room: {Nat.is_le(Nat.mul(outputs,inner),weight_count) == True{} : Bool},+bias_room: {Nat.is_le(outputs,bias_count) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+weights_at: {weights == Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight) : List<&2,F32>}) ->
  {Values.matrix_value(window(~Left,~left_value,left,inner,outputs,rows,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image),output,row,0n) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}:
  +problem = window(~Left,~left_value,left,inner,outputs,rows,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image)
  +sum = ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0)
  +fits = {%Equal.sym(Nat,Nat.add(inner,0n),inner,Addition.zero_right(inner)) : {Nat.is_le(_,inner) == True{} : Bool} Order.reflexive(inner) : {Nat.is_le(Nat.add(inner,0n),inner) == True{} : Bool}}
  %Equal.sym(F32,ConvSpec.bias_at(problem,output),bias_value(bias,U32.from_nat(output)),
      bias_read(~Left,~left_value,~Bias,~bias_value,left,inner,outputs,rows,weights,bias,bias_count,image,output,Bounds.nat_trans(output,outputs,bias_count,output_inside,bias_room))) :
    {(Values.matrix_dot(MatrixSpec.kchannels(inner,0n,1n),problem,output,row,0n,0.0) + _ : F32) == (sum + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  %Equal.sym(F32,Values.matrix_dot(MatrixSpec.kchannels(inner,0n,1n),problem,output,row,0n,0.0),sum,
      dot_sum(~Left,~left_value,~Weight,~weight_value,left,count,inner,outputs,rows,weight,weight_count,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),weights,weight_base,cap,cap_ok,input_room,inner_positive,rows_positive,weights_room,image,output,row,image_inside,output_inside,row_inside,weights_at,inner,0n,fits,0.0)) :
    {(_ + bias_value(bias,U32.from_nat(output)) : F32) == (sum + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  {==}

# The output of the window at its output and row.
def window_read(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weight: Weight,+weight_count: Nat,+bias: Bias,+bias_count: Nat,+weights: List<&2,F32>,+weight_base: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+input_room: {Nat.is_le(Nat.mul(Nat.mul(count,inner),rows),U32.to_nat(cap)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+weights_room: {Nat.is_le(Nat.mul(outputs,inner),weight_count) == True{} : Bool},+bias_room: {Nat.is_le(outputs,bias_count) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+weights_at: {weights == Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight) : List<&2,F32>}) ->
  {ImageProblems.at_position(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image,output,row) ==
    (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}:
  %Equal.sym(Nat & Nat,Nat.divmod(row,1n),(row,0n),Swapped.by_one(row)) :
    {Values.matrix_value(window(~Left,~left_value,left,inner,outputs,rows,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image),output,Pair.fst(Nat,Nat,_),Pair.snd(Nat,Nat,_)) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  matrix_read(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,count,inner,outputs,rows,weight,weight_count,bias,bias_count,weights,weight_base,cap,cap_ok,input_room,inner_positive,rows_positive,weights_room,bias_room,image,output,row,image_inside,output_inside,row_inside,weights_at)

# The output of the window at index (image, output, row) of [count,outputs,rows,1].
def window_value(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weight: Weight,+weight_count: Nat,+bias: Bias,+bias_count: Nat,+weights: List<&2,F32>,+weight_base: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+input_room: {Nat.is_le(Nat.mul(Nat.mul(count,inner),rows),U32.to_nat(cap)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+weights_room: {Nat.is_le(Nat.mul(outputs,inner),weight_count) == True{} : Bool},+bias_room: {Nat.is_le(outputs,bias_count) == True{} : Bool},+output_room: {Nat.is_le(Nat.mul(Nat.mul(count,outputs),rows),U32.to_nat(cap)) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+weights_at: {weights == Contract.listed(~F32,~Weight,~weight_value,weight_count,weight_base,weight) : List<&2,F32>}) ->
  {ImageProblems.value(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),ImageProblems.Context{Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias)},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),row))) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}:
  +index = Nat.add(Nat.mul(Nat.add(Nat.mul(image,outputs),output),rows),row)
  +plane_rows = plane_exact(rows,rows_positive)
  +row_plane = {%Equal.sym(Nat,ImageProblems.plane(rows,1n,1n,1n,0n),rows,plane_rows) : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(row,ImageProblems.plane(rows,1n,1n,1n,0n)) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,Swapped.index_room(image,output,row,count,outputs,rows,cap,image_inside,output_inside,row_inside,output_room))) :
    {ImageProblems.value_at(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),_) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  %Equal.sym(Nat,index,RowMajor.index(image,outputs,rows,output,row),RowMajor.mixed(image,outputs,rows,output,row)) :
    {ImageProblems.value_at(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),_) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  %plane_rows : {ImageProblems.value_at(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),Nat.add(Nat.mul(image,Nat.mul(outputs,_)),Nat.add(Nat.mul(output,_),row))) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  Equal.trans(F32,ImageProblems.value_at(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),Nat.add(Nat.mul(image,Nat.mul(outputs,ImageProblems.plane(rows,1n,1n,1n,0n))),Nat.add(Nat.mul(output,ImageProblems.plane(rows,1n,1n,1n,0n)),row))),
    ImageProblems.at_position(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image,output,row),
    (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.add(weight_base,Nat.mul(output,inner)),0.0) + bias_value(bias,U32.from_nat(output)) : F32),
    ImageProblems.decoded(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),Swapped.Swapped{left,inner,rows},Activations.identity(),inner,outputs,rows,1n,1n,1n,0n,weights,Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias),image,output,row,output_inside,row_plane),
    window_read(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,count,inner,outputs,rows,weight,weight_count,bias,bias_count,weights,weight_base,cap,cap_ok,input_room,inner_positive,rows_positive,weights_room,bias_room,image,output,row,image_inside,output_inside,row_inside,weights_at))

# The shared product at index (image, row, output) of [count,rows,outputs].
def shared_read(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,+weight: Weight,+bias: Bias,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +last_room: {Nat.is_le(Nat.mul(Nat.mul(count,rows),outputs),U32.to_nat(cap)) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,outputs) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,Shared{left,weight,bias,inner,outputs},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),outputs),output))) == (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.mul(output,inner),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}:
  +index = Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),outputs),output)
  +line = Nat.add(Nat.mul(image,rows),row)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,Swapped.index_room(image,row,output,count,rows,outputs,cap,image_inside,row_inside,output_inside,last_room))) :
    {(ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.div(_,outputs),inner),weight,Nat.mul(Nat.mod(_,outputs),inner),0.0) + bias_value(bias,U32.from_nat(Nat.mod(_,outputs))) : F32) ==
      (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.mul(output,inner),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(index,outputs),(line,output),Division.quotient_remainder(line,output,outputs,outputs_positive,output_inside)) :
    {(ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Pair.fst(Nat,Nat,_),inner),weight,Nat.mul(Pair.snd(Nat,Nat,_),inner),0.0) + bias_value(bias,U32.from_nat(Pair.snd(Nat,Nat,_))) : F32) ==
      (ordered_sum(~Left,~left_value,~Weight,~weight_value,inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),weight,Nat.mul(output,inner),0.0) + bias_value(bias,U32.from_nat(output)) : F32) : F32}
  {==}

# ---- The window read at the exchanged index ----

# A bound of a product of three extents in another order.
def swap_room(+first: Nat,+second: Nat,+third: Nat,+cap: Nat,+room: {Nat.is_le(Nat.mul(Nat.mul(first,second),third),cap) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(first,third),second),cap) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Nat.mul(first,third),second),Nat.mul(first,Nat.mul(third,second)),RowMajor.assoc(first,third,second)) : {Nat.is_le(_,cap) == True{} : Bool}
  %RowMajor.multiply_commutative(second,third) : {Nat.is_le(Nat.mul(first,_),cap) == True{} : Bool}
  %RowMajor.assoc(first,second,third) : {Nat.is_le(_,cap) == True{} : Bool}
  room

# What the window and the shared product read, and their rooms.
type Facts<-count: U32,-lanes: U32,-rows: U32,-inner: U32,-weight_count: Nat,-bias_count: Nat> is Data:
  Facts{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(count),U32.to_nat(lanes)),U32.to_nat(rows)),U32.to_nat(cap)) == True{} : Bool},
    input_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(rows)),U32.to_nat(cap)) == True{} : Bool},
    inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},lanes_positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    weights_room: {Nat.is_le(Nat.mul(U32.to_nat(lanes),U32.to_nat(inner)),weight_count) == True{} : Bool},bias_room: {Nat.is_le(U32.to_nat(lanes),bias_count) == True{} : Bool}}

# The context of the window over left with the listed weights and bias.
def window_context(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+weight: Weight,+bias: Bias,+inner: U32,+weight_count: Nat,+bias_count: Nat,+lanes: U32,+rows: U32) ->
  ImageProblems.Context<Swapped.Swapped<Left>>:
  ImageProblems.Context{Swapped.Swapped{left,U32.to_nat(inner),U32.to_nat(rows)},Activations.identity(),U32.to_nat(inner),U32.to_nat(lanes),U32.to_nat(rows),1n,1n,1n,0n,Contract.listed(~F32,~Weight,~weight_value,weight_count,0n,weight),Contract.listed(~F32,~Bias,~bias_value,bias_count,0n,bias)}

# The window over left [count,rows,inner] in the middle, the shared product in the last.
def SharedWindow(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+weight: Weight,+bias: Bias,+inner: U32,+weight_count: Nat,+bias_count: Nat,
  +middle: ImageProblems.Context<Swapped.Swapped<Left>>,+last: Shared<Left,Weight,Bias>,+count: U32,+lanes: U32,+rows: U32) -> Data:
  KernelGemm.Both<{middle == window_context(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,inner,weight_count,bias_count,lanes,rows) : ImageProblems.Context<Swapped.Swapped<Left>>},
    KernelGemm.Both<{last == Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(lanes)} : Shared<Left,Weight,Bias>},Facts<count,lanes,rows,inner,weight_count,bias_count>>>

# The window and the shared product agree at the exchanged indices.
def window_agrees(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+weight: Weight,+bias: Bias,+inner: U32,+weight_count: Nat,+bias_count: Nat,
  +middle: ImageProblems.Context<Swapped.Swapped<Left>>,+last: Shared<Left,Weight,Bias>,+count: U32,+lanes: U32,+rows: U32,
  +related: SharedWindow(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,inner,weight_count,bias_count,middle,last,count,lanes,rows),
  +image: Nat,+lane: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,U32.to_nat(count)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,U32.to_nat(lanes)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {ImageProblems.value(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),middle,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),lane),U32.to_nat(rows)),row))) ==
    shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,last,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(lanes)),lane))) : F32}:
  KernelGemm.Both{+same_middle,KernelGemm.Both{+same_last,Facts{+cap,+cap_ok,+room,+input_room,+inner_positive,+lanes_positive,+rows_positive,+weights_room,+bias_room}}} = related
  +context = window_context(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,inner,weight_count,bias_count,lanes,rows)
  +shared = {Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(lanes)} : Shared<Left,Weight,Bias>}
  +middle_index = U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),lane),U32.to_nat(rows)),row))
  +last_index = U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(lanes)),lane))
  +total = ordered_sum(~Left,~left_value,~Weight,~weight_value,U32.to_nat(inner),0n,left,Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(inner)),weight,Nat.mul(lane,U32.to_nat(inner)),0.0)
  +read = (total + bias_value(bias,U32.from_nat(lane)) : F32)
  %Equal.sym(ImageProblems.Context<Swapped.Swapped<Left>>,middle,context,same_middle) : {ImageProblems.value(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),_,middle_index) == shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,last,last_index) : F32}
  %Equal.sym(Shared<Left,Weight,Bias>,last,shared,same_last) : {ImageProblems.value(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),context,middle_index) == shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,_,last_index) : F32}
  Equal.trans(F32,ImageProblems.value(~Activations.identity_value,~Swapped.Swapped<Left>,~Swapped.swapped_value(~Left,~left_value),context,middle_index),read,shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,shared,last_index),
    window_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,U32.to_nat(count),U32.to_nat(inner),U32.to_nat(lanes),U32.to_nat(rows),weight,weight_count,bias,bias_count,Contract.listed(~F32,~Weight,~weight_value,weight_count,0n,weight),0n,cap,cap_ok,input_room,inner_positive,rows_positive,
      weights_room,bias_room,room,image,lane,row,image_inside,lane_inside,row_inside,{==}),
    Equal.sym(F32,shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,shared,last_index),read,
      shared_read(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,U32.to_nat(count),U32.to_nat(inner),U32.to_nat(lanes),U32.to_nat(rows),weight,bias,cap,cap_ok,
        swap_room(U32.to_nat(count),U32.to_nat(lanes),U32.to_nat(rows),U32.to_nat(cap),room),lanes_positive,image,lane,row,image_inside,lane_inside,row_inside)))

# ---- The products of matrices ----

# The matrix of image `image` of right [count,inner,outputs] read as weights
# [outputs,inner]: index j is output j / inner and inner index j % inner.
type Transposed<-Right: Data> is Data:
  Transposed{right: Right,image: Nat,inner: Nat,outputs: Nat}

def transposed_value(~Right: Data,~right_value: Right -> U32 -> F32,context: Transposed<Right>,index: U32) -> F32:
  Transposed{+right,+image,+inner,+outputs} = context
  +position = U32.to_nat(index)
  right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,inner),Nat.mod(position,inner)),outputs),Nat.div(position,inner))))

# Left [count,rows,inner] by right [count,inner,outputs], a matrix of right for
# each image.
type Batched<-Left: Data,-Right: Data> is Data:
  Batched{left: Left,right: Right,rows: Nat,inner: Nat,outputs: Nat}

# Index i of [count,rows,outputs]: the ordered sum of row i / outputs of left with
# column i % outputs of the matrix of right of its image from +0.0, then +0.0.
def batched_value(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,context: Batched<Left,Right>,index: U32) -> F32:
  Batched{+left,+right,+rows,+inner,+outputs} = context
  +position = U32.to_nat(index)
  (ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(Nat.div(position,outputs),inner),
    Transposed{right,Nat.div(Nat.div(position,outputs),rows),inner,outputs},Nat.mul(Nat.mod(position,outputs),inner),0.0) + 0.0 : F32)

# The product of matrices at index (image, row, output) of [count,rows,outputs].
def batched_read(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,+left: Left,+right: Right,+count: Nat,+inner: Nat,+outputs: Nat,+rows: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+last_room: {Nat.is_le(Nat.mul(Nat.mul(count,rows),outputs),U32.to_nat(cap)) == True{} : Bool},
  +outputs_positive: {Nat.is_lt(0n,outputs) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+image: Nat,+output: Nat,+row: Nat,
  +image_inside: {Nat.is_lt(image,count) == True{} : Bool},+output_inside: {Nat.is_lt(output,outputs) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {batched_value(~Left,~left_value,~Right,~right_value,Batched{left,right,rows,inner,outputs},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),outputs),output))) ==
    (ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(Nat.add(Nat.mul(image,rows),row),inner),Transposed{right,image,inner,outputs},
      Nat.mul(output,inner),0.0) + 0.0 : F32) : F32}:
  +line = Nat.add(Nat.mul(image,rows),row)
  +index = Nat.add(Nat.mul(line,outputs),output)
  +read = (ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(line,inner),Transposed{right,image,inner,outputs},
    Nat.mul(output,inner),0.0) + 0.0 : F32)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,Swapped.index_room(image,row,output,count,rows,outputs,cap,image_inside,row_inside,output_inside,last_room))) :
    {(ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(Nat.div(_,outputs),inner),
      Transposed{right,Nat.div(Nat.div(_,outputs),rows),inner,outputs},Nat.mul(Nat.mod(_,outputs),inner),0.0) + 0.0 : F32) == read : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(index,outputs),(line,output),Division.quotient_remainder(line,output,outputs,outputs_positive,output_inside)) :
    {(ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(Pair.fst(Nat,Nat,_),inner),
      Transposed{right,Nat.div(Pair.fst(Nat,Nat,_),rows),inner,outputs},Nat.mul(Pair.snd(Nat,Nat,_),inner),0.0) + 0.0 : F32) == read : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(line,rows),(image,row),Division.quotient_remainder(image,row,rows,rows_positive,row_inside)) :
    {(ordered_sum(~Left,~left_value,~Transposed<Right>,~transposed_value(~Right,~right_value),inner,0n,left,Nat.mul(line,inner),
      Transposed{right,Pair.fst(Nat,Nat,_),inner,outputs},Nat.mul(output,inner),0.0) + 0.0 : F32) == read : F32}
  {==}
