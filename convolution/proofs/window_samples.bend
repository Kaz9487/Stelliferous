# A window problem keeps rows [first, first + rows) of the original image as
# a shorter image. Where the window covers every row a read needs, its sample
# at output row r is the original sample at row block + r, first = block * stride.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ./matrix_reduction_bounds.bend as ReductionBounds
import ./output_values.bend as Values
import ./window_band_values.bend as WindowBandValues
import ./convolution_guard_facts.bend as Guards
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor

def window(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat) -> ConvSpec.Problem:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{+ci,co,height,+width,kernel,stride,padding} = shape
  ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},
    WindowBandValues.values(Nat.mul(ci,Nat.mul(rows,width)),0n,WindowBandValues.Band{input,height,width,first,rows}),weights,bias}

def le_after(+first: Nat,+row: Nat) -> {Nat.is_le(row,Nat.add(first,row)) == True{} : Bool}:
  %Order.commutative(row,first) : {Nat.is_le(row,_) == True{} : Bool}
  Order.le_add(row,first)

# Input row block * stride + (r * stride + kernel_row) of original output row block + r.
def row_shift(+block: Nat,+row: Nat,+stride: Nat,+kernel_row: Nat,+first: Nat,+first_exact: {first == Nat.mul(block,stride) : Nat}) ->
  {Nat.add(Nat.mul(Nat.add(block,row),stride),kernel_row) == Nat.add(first,Nat.add(Nat.mul(row,stride),kernel_row)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(block,row),stride),Nat.add(Nat.mul(block,stride),Nat.mul(row,stride)),RowMajor.distrib(block,row,stride)) :
    {Nat.add(_,kernel_row) == Nat.add(first,Nat.add(Nat.mul(row,stride),kernel_row)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(block,stride),Nat.mul(row,stride)),kernel_row),Nat.add(Nat.mul(block,stride),Nat.add(Nat.mul(row,stride),kernel_row)),
    Addition.associative(Nat.mul(block,stride),Nat.mul(row,stride),kernel_row)) :
    {_ == Nat.add(first,Nat.add(Nat.mul(row,stride),kernel_row)) : Nat}
  %Equal.sym(Nat,first,Nat.mul(block,stride),first_exact) :
    {Nat.add(Nat.mul(block,stride),Nat.add(Nat.mul(row,stride),kernel_row)) == Nat.add(_,Nat.add(Nat.mul(row,stride),kernel_row)) : Nat}
  {==}

# How the window's top meets the read rows from row to row + reach: the
# window starts at row 0, the rows lie at or below the top padding of the
# window, or the original rows lie in the padding above the image.
type Top<-first: Nat,-padding: Nat,-row: Nat,-reach: Nat> is Data:
  Start{zero: {first == 0n : Nat}}
  Below{inside: {Nat.is_le(padding,row) == True{} : Bool}}
  Above{outside: {Nat.is_le(Nat.add(Nat.add(first,row),reach),padding) == True{} : Bool}}

def not_reached(+row: Nat,+padding: Nat,below: {Nat.is_lt(row,padding) == True{} : Bool}) -> {Nat.is_le(padding,row) == False{} : Bool}:
  match row padding:
    case 0n 0n: Empty.absurd({Nat.is_le(0n,0n) == False{} : Bool},Order.below_zero(0n,below))
    case 1n+row 0n: Empty.absurd({Nat.is_le(0n,1n+row) == False{} : Bool},Order.below_zero(1n+row,below))
    case 0n 1n+padding: {==}
    case 1n+row 1n+padding: not_reached(row,padding,below)

# A row whose successor fits below a bound is below it.
def before_bound(+row: Nat,+bound: Nat,+room: {Nat.is_le(Nat.add(row,1n),bound) == True{} : Bool}) -> {Nat.is_lt(row,bound) == True{} : Bool}:
  Bounds.nat_trans(row,1n+row,bound,Order.lt_succ(row,row,Order.reflexive(row)),
    %Order.commutative(row,1n) : {Nat.is_le(_,bound) == True{} : Bool}
    room)

# No read needs a row above the window, unless the window starts at row 0 or
# the read lies in the padding above the image in both problems.
def low(+first: Nat,+padding: Nat,+row: Nat,+top: Top<first,padding,row,1n>) ->
  {Nat.is_le(padding,Nat.add(first,row)) == Nat.is_le(padding,row) : Bool}:
  match top:
    case Start{+zero}:
      %Equal.sym(Nat,first,0n,zero) : {Nat.is_le(padding,Nat.add(_,row)) == Nat.is_le(padding,row) : Bool}
      {==}
    case Below{+below}:
      %Equal.sym(Bool,Nat.is_le(padding,Nat.add(first,row)),True{},Order.transitive(padding,row,Nat.add(first,row),below,le_after(first,row))) :
        {_ == Nat.is_le(padding,row) : Bool}
      %Equal.sym(Bool,Nat.is_le(padding,row),True{},below) : {True{} == _ : Bool}
      {==}
    case Above{+outside}:
      +shifted = before_bound(Nat.add(first,row),padding,outside)
      %Equal.sym(Bool,Nat.is_le(padding,Nat.add(first,row)),False{},not_reached(Nat.add(first,row),padding,shifted)) :
        {_ == Nat.is_le(padding,row) : Bool}
      %Equal.sym(Bool,Nat.is_le(padding,row),False{},not_reached(row,padding,Order.le_lt_trans(row,Nat.add(first,row),padding,le_after(first,row),shifted))) :
        {False{} == _ : Bool}
      {==}

# No read needs a row below the window, unless the window ends at the image end.
def high(+first: Nat,+rows: Nat,+height: Nat,+padding: Nat,+row: Nat,+fits: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool},
  +bottom: Either<&2,&2,{height == Nat.add(first,rows) : Nat},{Nat.is_lt(row,Nat.add(padding,rows)) == True{} : Bool}>) ->
  {Nat.is_lt(Nat.add(first,row),Nat.add(padding,height)) == Nat.is_lt(row,Nat.add(padding,rows)) : Bool}:
  match bottom:
    case Inl{+full}:
      %Equal.sym(Nat,height,Nat.add(first,rows),full) : {Nat.is_lt(Nat.add(first,row),Nat.add(padding,_)) == Nat.is_lt(row,Nat.add(padding,rows)) : Bool}
      %Equal.sym(Nat,Nat.add(padding,Nat.add(first,rows)),Nat.add(first,Nat.add(padding,rows)),RowMajor.add_swap(padding,first,rows)) :
        {Nat.is_lt(Nat.add(first,row),_) == Nat.is_lt(row,Nat.add(padding,rows)) : Bool}
      Order.lt_offset(first,row,Nat.add(padding,rows))
    case Inr{+inside}:
      +shifted = {%Equal.sym(Bool,Nat.is_lt(Nat.add(first,row),Nat.add(first,Nat.add(padding,rows))),Nat.is_lt(row,Nat.add(padding,rows)),
          Order.lt_offset(first,row,Nat.add(padding,rows))) : {_ == True{} : Bool}
        inside : {Nat.is_lt(Nat.add(first,row),Nat.add(first,Nat.add(padding,rows))) == True{} : Bool}}
      +bounded = {%Equal.sym(Nat,Nat.add(first,Nat.add(padding,rows)),Nat.add(padding,Nat.add(first,rows)),RowMajor.add_swap(first,padding,rows)) :
          {Nat.is_le(_,Nat.add(padding,height)) == True{} : Bool}
        %Equal.sym(Bool,Nat.is_le(Nat.add(padding,Nat.add(first,rows)),Nat.add(padding,height)),Nat.is_le(Nat.add(first,rows),height),
          Order.le_offset(padding,Nat.add(first,rows),height)) : {_ == True{} : Bool}
        fits : {Nat.is_le(Nat.add(first,Nat.add(padding,rows)),Nat.add(padding,height)) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(first,row),Nat.add(padding,height)),True{},
        Bounds.nat_trans(Nat.add(first,row),Nat.add(first,Nat.add(padding,rows)),Nat.add(padding,height),shifted,bounded)) :
        {_ == Nat.is_lt(row,Nat.add(padding,rows)) : Bool}
      %Equal.sym(Bool,Nat.is_lt(row,Nat.add(padding,rows)),True{},inside) : {True{} == _ : Bool}
      {==}

# A covered read: band index channel * plane + r holds original index
# channel * height * width + (first + row - padding) * width + column - padding.
def inside_read(+input: List<&2,F32>,+ci: Nat,+height: Nat,+width: Nat,+first: Nat,+rows: Nat,+padding: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,ci) == True{} : Bool},
  +low_row: {Nat.is_le(padding,row) == True{} : Bool},+high_row: {Nat.is_lt(row,Nat.add(padding,rows)) == True{} : Bool},
  +low_column: {Nat.is_le(padding,column) == True{} : Bool},+high_column: {Nat.is_lt(column,Nat.add(padding,width)) == True{} : Bool}) ->
  {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(rows,width)),Nat.add(Nat.mul(Nat.sub(row,padding),width),Nat.sub(column,padding))),
      WindowBandValues.values(Nat.mul(ci,Nat.mul(rows,width)),0n,WindowBandValues.Band{input,height,width,first,rows})) ==
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}:
  +plane = Nat.mul(rows,width)
  +offset = Nat.add(Nat.mul(Nat.sub(row,padding),width),Nat.sub(column,padding))
  +band = {WindowBandValues.Band{input,height,width,first,rows} : WindowBandValues.Band}
  +offset_inside = RowMajor.radix(Nat.sub(row,padding),rows,Nat.sub(column,padding),width,
    Order.sub_lt(row,padding,rows,low_row,high_row),Order.sub_lt(column,padding,width,low_column,high_column))
  +positive = Order.le_lt_trans(0n,offset,plane,Order.zero_le(offset),offset_inside)
  +index_inside = RowMajor.radix(channel,ci,offset,plane,channel_inside,offset_inside)
  +split = Division.quotient_remainder(channel,offset,plane,positive,offset_inside)
  %Equal.sym(F32,ConvSpec.at(Nat.add(Nat.mul(channel,plane),offset),WindowBandValues.values(Nat.mul(ci,plane),0n,band)),WindowBandValues.value(band,Nat.add(Nat.mul(channel,plane),offset)),
    WindowBandValues.at(Nat.mul(ci,plane),band,Nat.add(Nat.mul(channel,plane),offset),index_inside)) :
    {_ == ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(channel,plane),offset),plane),(channel,offset),split) :
    {ConvSpec.at(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),Nat.mul(height,width)),Nat.add(Nat.mul(first,width),Pair.snd(Nat,Nat,_))),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Nat,Nat.sub(Nat.add(first,row),padding),Nat.add(first,Nat.sub(row,padding)),Order.subtract_from_sum(first,row,padding,low_row)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(first,width),offset)),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(_,width),Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Nat,Nat.mul(Nat.add(first,Nat.sub(row,padding)),width),Nat.add(Nat.mul(first,width),Nat.mul(Nat.sub(row,padding),width)),
    RowMajor.distrib(first,Nat.sub(row,padding),width)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(first,width),offset)),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(_,Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(Nat.sub(row,padding),width)),Nat.sub(column,padding)),
    Nat.add(Nat.mul(first,width),offset),Addition.associative(Nat.mul(first,width),Nat.mul(Nat.sub(row,padding),width),Nat.sub(column,padding))) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(first,width),offset)),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),_),input) : F32}
  {==}

def selected(+inside: Bool,+input: List<&2,F32>,+ci: Nat,+height: Nat,+width: Nat,+first: Nat,+rows: Nat,+padding: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,ci) == True{} : Bool},
  +exact: {Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)) && Nat.is_le(padding,column) && Nat.is_lt(column,Nat.add(padding,width)) == inside : Bool}) ->
  {ConvSpec.sample_if(inside,Nat.add(Nat.mul(channel,Nat.mul(rows,width)),Nat.add(Nat.mul(Nat.sub(row,padding),width),Nat.sub(column,padding))),
      WindowBandValues.values(Nat.mul(ci,Nat.mul(rows,width)),0n,WindowBandValues.Band{input,height,width,first,rows})) ==
    ConvSpec.sample_if(inside,Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}:
  match inside:
    case False{}: {==}
    case True{}:
      +three = Boolean.left(Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)) && Nat.is_le(padding,column),
        Nat.is_lt(column,Nat.add(padding,width)),exact)
      +two = Boolean.left(Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)),Nat.is_le(padding,column),three)
      inside_read(input,ci,height,width,first,rows,padding,channel,row,column,channel_inside,
        Boolean.left(Nat.is_le(padding,row),Nat.is_lt(row,Nat.add(padding,rows)),two),
        Boolean.right(Nat.is_le(padding,row),Nat.is_lt(row,Nat.add(padding,rows)),two),
        Boolean.right(Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)),Nat.is_le(padding,column),three),
        Boolean.right(Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)) && Nat.is_le(padding,column),
          Nat.is_lt(column,Nat.add(padding,width)),exact))

# The window sample at output row r equals the original sample at row block + r.
def sample(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+block: Nat,+channel: Nat,+output_row: Nat,+output_column: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.input_channel_count(problem)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +top: Top<first,Guards.pad(problem),Nat.add(Nat.mul(output_row,Guards.stride(problem)),kernel_row),1n>,
  +bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_lt(Nat.add(Nat.mul(output_row,Guards.stride(problem)),kernel_row),Nat.add(Guards.pad(problem),rows)) == True{} : Bool}>) ->
  {ConvSpec.sample(window(problem,first,rows),channel,output_row,output_column,kernel_row,kernel_column) ==
    ConvSpec.sample(problem,channel,Nat.add(block,output_row),output_column,kernel_row,kernel_column) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +row = Nat.add(Nat.mul(output_row,stride),kernel_row)
  +column = Nat.add(Nat.mul(output_column,stride),kernel_column)
  +windowed = {ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},
    WindowBandValues.values(Nat.mul(ci,Nat.mul(rows,width)),0n,WindowBandValues.Band{input,height,width,first,rows}),weights,bias} : ConvSpec.Problem}
  %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(block,output_row),stride),kernel_row),Nat.add(first,row),row_shift(block,output_row,stride,kernel_row,first,first_exact)) :
    {ConvSpec.sample(windowed,channel,output_row,output_column,kernel_row,kernel_column) ==
      ConvSpec.sample_if(Nat.is_le(padding,_) && Nat.is_lt(_,Nat.add(padding,height)) && Nat.is_le(padding,column) && Nat.is_lt(column,Nat.add(padding,width)),
        Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(_,padding),width),Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Bool,Nat.is_le(padding,Nat.add(first,row)),Nat.is_le(padding,row),low(first,padding,row,top)) :
    {ConvSpec.sample(windowed,channel,output_row,output_column,kernel_row,kernel_column) ==
      ConvSpec.sample_if(_ && Nat.is_lt(Nat.add(first,row),Nat.add(padding,height)) && Nat.is_le(padding,column) && Nat.is_lt(column,Nat.add(padding,width)),
        Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}
  %Equal.sym(Bool,Nat.is_lt(Nat.add(first,row),Nat.add(padding,height)),Nat.is_lt(row,Nat.add(padding,rows)),high(first,rows,height,padding,row,fits,bottom)) :
    {ConvSpec.sample(windowed,channel,output_row,output_column,kernel_row,kernel_column) ==
      ConvSpec.sample_if(Nat.is_le(padding,row) && _ && Nat.is_le(padding,column) && Nat.is_lt(column,Nat.add(padding,width)),
        Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(Nat.add(first,row),padding),width),Nat.sub(column,padding))),input) : F32}
  selected(Nat.is_le(padding,row) && Nat.is_lt(row,Nat.add(padding,rows)) && Nat.is_le(padding,column) && Nat.is_lt(column,Nat.add(padding,width)),
    input,ci,height,width,first,rows,padding,channel,row,column,channel_inside,{==})

# Row-level cover facts give every kernel row's cover facts.
def top_at(+first: Nat,+padding: Nat,+base: Nat,+kernel: Nat,+kernel_row: Nat,+kernel_row_inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},
  +top: Top<first,padding,base,kernel>) -> Top<first,padding,Nat.add(base,kernel_row),1n>:
  match top:
    case Start{+zero}: Start{zero}
    case Below{+below}: Below{Order.transitive(padding,base,Nat.add(base,kernel_row),below,Order.le_add(base,kernel_row))}
    case Above{+outside}:
      +reached = Nat.add(first,base)
      +tap = {%Addition.associative(first,base,kernel_row) : {Nat.is_le(Nat.add(_,1n),Nat.add(reached,kernel)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(reached,kernel_row),1n),Nat.add(reached,Nat.add(kernel_row,1n)),Addition.associative(reached,kernel_row,1n)) :
          {Nat.is_le(_,Nat.add(reached,kernel)) == True{} : Bool}
        %Order.commutative(1n,kernel_row) : {Nat.is_le(Nat.add(reached,_),Nat.add(reached,kernel)) == True{} : Bool}
        Order.offset_le(reached,1n+kernel_row,kernel,Order.strict_follow(kernel_row,kernel,kernel_row_inside)) :
        {Nat.is_le(Nat.add(Nat.add(first,Nat.add(base,kernel_row)),1n),Nat.add(reached,kernel)) == True{} : Bool}}
      Above{Order.transitive(Nat.add(Nat.add(first,Nat.add(base,kernel_row)),1n),Nat.add(reached,kernel),padding,tap,outside)}

def bottom_at(+height: Nat,+first: Nat,+rows: Nat,+padding: Nat,+base: Nat,+kernel: Nat,+kernel_row: Nat,
  +kernel_row_inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},
  +bottom: Either<&2,&2,{height == Nat.add(first,rows) : Nat},{Nat.is_le(Nat.add(base,kernel),Nat.add(padding,rows)) == True{} : Bool}>) ->
  Either<&2,&2,{height == Nat.add(first,rows) : Nat},{Nat.is_lt(Nat.add(base,kernel_row),Nat.add(padding,rows)) == True{} : Bool}>:
  match bottom:
    case Inl{+full}: Inl{full}
    case Inr{+room}:
      Inr{Bounds.nat_trans(Nat.add(base,kernel_row),Nat.add(base,kernel),Nat.add(padding,rows),
        %Equal.sym(Bool,Nat.is_lt(Nat.add(base,kernel_row),Nat.add(base,kernel)),Nat.is_lt(kernel_row,kernel),Order.lt_offset(base,kernel_row,kernel)) : {_ == True{} : Bool}
        kernel_row_inside,room)}

def patch_agrees(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+block: Nat,+output_row: Nat,+output_column: Nat,+index: MatrixSpec.Reduction,
  +first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +top: Top<first,Guards.pad(problem),Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)>,
  +bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)),Nat.add(Guards.pad(problem),rows)) == True{} : Bool}>,
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {MatrixSpec.patch(window(problem,first,rows),MatrixSpec.Position{output_row,output_column},index) ==
    MatrixSpec.patch(problem,MatrixSpec.Position{Nat.add(block,output_row),output_column},index) : F32}:
  MatrixSpec.KIndex{+channel,+kernel_row,+kernel_column} = index
  Shape.Both{+channel_inside,+spatial} = bounds
  Shape.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  +base = Nat.mul(output_row,Guards.stride(problem))
  sample(problem,first,rows,block,channel,output_row,output_column,kernel_row,kernel_column,first_exact,channel_inside,fits,
    top_at(first,Guards.pad(problem),base,Guards.kernel(problem),kernel_row,kernel_row_inside,top),
    bottom_at(Guards.height(problem),first,rows,Guards.pad(problem),base,Guards.kernel(problem),kernel_row,kernel_row_inside,bottom))

def weight_agrees(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+channel: Nat,+index: MatrixSpec.Reduction) ->
  {MatrixSpec.weight(window(problem,first,rows),channel,index) == MatrixSpec.weight(problem,channel,index) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
  {==}

def agreements(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+block: Nat,+channel: Nat,+output_row: Nat,+output_column: Nat,
  +indices: List<&2,MatrixSpec.Reduction>,
  +first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +top: Top<first,Guards.pad(problem),Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)>,
  +bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)),Nat.add(Guards.pad(problem),rows)) == True{} : Bool}>,
  bounds: ReductionBounds.AllBounds(problem,indices)) ->
  Ordered.MatchingSamples(~MatrixSpec.Reduction,~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~MatrixSpec.patch,~MatrixSpec.weight,~MatrixSpec.patch,~MatrixSpec.weight,indices,channel,MatrixSpec.Position{output_row,output_column},window(problem,first,rows),
    channel,MatrixSpec.Position{Nat.add(block,output_row),output_column},problem):
  match indices:
    case Nil{}: Unit{}
    case +index <> +rest:
      Shape.Both{+head,tail} = bounds
      Shape.Both{
        Shape.Both{patch_agrees(problem,first,rows,block,output_row,output_column,index,first_exact,fits,top,bottom,head),
          weight_agrees(problem,first,rows,channel,index)},
        agreements(problem,first,rows,block,channel,output_row,output_column,rest,first_exact,fits,top,bottom,tail)}

# The window dot at output row r is the original dot at row block + r.
def matrix_dot(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+block: Nat,+channel: Nat,+output_row: Nat,+output_column: Nat,
  +first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +top: Top<first,Guards.pad(problem),Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)>,
  +bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(output_row,Guards.stride(problem)),Guards.kernel(problem)),Nat.add(Guards.pad(problem),rows)) == True{} : Bool}>) ->
  {Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),window(problem,first,rows),channel,output_row,output_column,0.0) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,Nat.add(block,output_row),output_column,0.0) : F32}:
  +indices = MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem))
  Ordered.dot_congruence(~MatrixSpec.Reduction,~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~MatrixSpec.patch,~MatrixSpec.weight,~MatrixSpec.patch,~MatrixSpec.weight,indices,
    channel,MatrixSpec.Position{output_row,output_column},window(problem,first,rows),channel,MatrixSpec.Position{Nat.add(block,output_row),output_column},problem,0.0,
    agreements(problem,first,rows,block,channel,output_row,output_column,indices,first_exact,fits,top,bottom,ReductionBounds.matrix_reduction_in_bounds(problem)))
