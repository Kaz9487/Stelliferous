# Output-position arithmetic shared by window leaves: rows of a flat position,
# cancelled stride products, and the padded length of aligned rows.
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./concrete_support_guard.bend as Support
import ./convolution_guard_facts.bend as Guards
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ./window_samples.bend as Samples

def padded_multiple(+count: Nat,+blocks: Nat,+exact: {count == Nat.mul(blocks,8n) : Nat}) ->
  {Nat.mul(Nat.div(Nat.add(count,7n),8n),8n) == count : Nat}:
  %Equal.sym(Nat,count,Nat.mul(blocks,8n),exact) : {Nat.mul(Nat.div(Nat.add(_,7n),8n),8n) == _ : Nat}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(blocks,8n),7n),8n),(blocks,7n),Division.quotient_remainder(blocks,7n,8n,{==},{==})) :
    {Nat.mul(Pair.fst(Nat,Nat,_),8n) == Nat.mul(blocks,8n) : Nat}
  {==}

# Rows of whole eight-position blocks need no padding.
def aligned_padded(+problem: ConvSpec.Problem,+blocks: Nat,+width: {Nat.mul(blocks,8n) == ConvSpec.out_width(problem) : Nat}) ->
  {Support.padded(problem) == Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)) : Nat}:
  +area = Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))
  padded_multiple(area,Nat.mul(ConvSpec.out_height(problem),blocks),
    %width : {Nat.mul(ConvSpec.out_height(problem),_) == Nat.mul(Nat.mul(ConvSpec.out_height(problem),blocks),8n) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.mul(ConvSpec.out_height(problem),blocks),8n),Nat.mul(ConvSpec.out_height(problem),Nat.mul(blocks,8n)),
      RowMajor.assoc(ConvSpec.out_height(problem),blocks,8n)))

def sub_after(+value: Nat,+amount: Nat,+extra: Nat,+ordered: {Nat.is_le(amount,value) == True{} : Bool}) ->
  {Nat.sub(Nat.add(value,extra),amount) == Nat.add(Nat.sub(value,amount),extra) : Nat}:
  %Order.commutative(extra,value) : {Nat.sub(_,amount) == Nat.add(Nat.sub(value,amount),extra) : Nat}
  %Order.commutative(extra,Nat.sub(value,amount)) : {Nat.sub(Nat.add(extra,value),amount) == _ : Nat}
  Order.subtract_from_sum(extra,value,amount,ordered)

def zero_factor(+value: Nat,+factor: Nat,+positive: {Nat.is_lt(0n,factor) == True{} : Bool},+zero: {Nat.mul(value,factor) == 0n : Nat}) ->
  {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest:
      match factor:
        case 0n: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(positive))
        case 1n+more: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),1n+Nat.add(more,Nat.mul(rest,1n+more)),0n,zero)))

# Rows [first, first + rows) cover every read of output positions
# [offset, offset + span): no read needs a row above first unless it is 0,
# and none needs a row at or past first + rows unless that is the image end.
# The window's first row against the span's reads: row 0, at least the top
# padding before the first row read, or not after the first output's row
# while every read lies in the padding above the image.
type Top<-problem: ConvSpec.Problem,-first: Nat,-offset: Nat,-span: Nat> is Data:
  Start{zero: {first == 0n : Nat}}
  Within{above: {Nat.is_le(Nat.add(first,Guards.pad(problem)),Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}}
  Over{before: {Nat.is_le(first,Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool},
    reads: {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(offset,span),1n),ConvSpec.out_width(problem)),Guards.stride(problem)),Guards.kernel(problem)),Guards.pad(problem)) == True{} : Bool}}

def top_of(+problem: ConvSpec.Problem,+first: Nat,+offset: Nat,+span: Nat,
  top: Either<&2,&2,{first == 0n : Nat},{Nat.is_le(Nat.add(first,Guards.pad(problem)),Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}>) ->
  Top<problem,first,offset,span>:
  match top:
    case Inl{zero}: Start{zero}
    case Inr{above}: Within{above}

type Cover<-problem: ConvSpec.Problem,-first: Nat,-rows: Nat,-offset: Nat,-span: Nat> is Data:
  Cover{block: Nat,
    first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
    fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
    kernel_fits: {Nat.is_le(Guards.kernel(problem),Nat.add(rows,Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool},
    top: Top<problem,first,offset,span>,
    bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
      {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(offset,span),1n),ConvSpec.out_width(problem)),Guards.stride(problem)),Guards.kernel(problem)),
        Nat.add(Nat.add(first,rows),Guards.pad(problem))) == True{} : Bool}>}

def block(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Cover<problem,first,rows,offset,span>) -> Nat:
  Cover{+block,first_exact,fits,rows_positive,kernel_fits,top,bottom} = cover
  block

def last_row(+position: Nat,+offset: Nat,+span: Nat,+inside: {Nat.is_lt(position,Nat.add(offset,span)) == True{} : Bool}) ->
  {Nat.is_le(position,Nat.sub(Nat.add(offset,span),1n)) == True{} : Bool}:
  +positive = Order.le_lt_trans(0n,position,Nat.add(offset,span),Order.zero_le(position),inside)
  Division.lt_succ_le(position,Nat.sub(Nat.add(offset,span),1n),
    %Equal.sym(Nat,1n+Nat.sub(Nat.add(offset,span),1n),Nat.add(offset,span),Order.positive_has_last(Nat.add(offset,span),positive)) :
      {Nat.is_lt(position,_) == True{} : Bool}
    inside)

# Every top case starts the window at or before the first output's row.
def top_before(+problem: ConvSpec.Problem,+first: Nat,+offset: Nat,+span: Nat,+top: Top<problem,first,offset,span>) ->
  {Nat.is_le(first,Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}:
  match top:
    case Start{+zero}:
      %Equal.sym(Nat,first,0n,zero) : {Nat.is_le(_,Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}
      Order.zero_le(Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem)))
    case Within{+above}:
      Order.transitive(first,Nat.add(first,Guards.pad(problem)),Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem)),
        Order.le_add(first,Guards.pad(problem)),above)
    case Over{+before,reads}: before

def block_le_start(+problem: ConvSpec.Problem,+first: Nat,+block: Nat,+offset: Nat,+span: Nat,
  +stride_positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},
  +first_exact: {first == Nat.mul(block,Guards.stride(problem)) : Nat},
  +top: Top<problem,first,offset,span>) ->
  {Nat.is_le(block,Nat.div(offset,ConvSpec.out_width(problem))) == True{} : Bool}:
  Ceiling.cancel_product_le(block,Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem),stride_positive,
    %first_exact : {Nat.is_le(_,Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem))) == True{} : Bool}
    top_before(problem,first,offset,span,top))

# The window starts block output rows before the lane's output row.
def block_le_row(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Cover<problem,first,rows,offset,span>,
  +position: Nat,+stride_positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},+after: {Nat.is_le(offset,position) == True{} : Bool}) ->
  {Nat.is_le(block(problem,first,rows,offset,span,cover),Nat.div(position,ConvSpec.out_width(problem))) == True{} : Bool}:
  Cover{+block,+first_exact,fits,rows_positive,kernel_fits,+top,bottom} = cover
  +stride = Guards.stride(problem)
  +width = ConvSpec.out_width(problem)
  +start = Nat.div(offset,width)
  Order.transitive(block,start,Nat.div(position,width),
    block_le_start(problem,first,block,offset,span,stride_positive,first_exact,top),
    DivisionOrder.quotient_le(offset,position,width,width_positive,after))

# A lane row is block + virtual row, so its stride product is first + virtual product.
def row_product(+block: Nat,+row: Nat,+stride: Nat,+first: Nat,+first_exact: {first == Nat.mul(block,stride) : Nat},
  +ordered: {Nat.is_le(block,row) == True{} : Bool}) -> {Nat.mul(row,stride) == Nat.add(first,Nat.mul(Nat.sub(row,block),stride)) : Nat}:
  %Equal.sym(Nat,first,Nat.mul(block,stride),first_exact) : {Nat.mul(row,stride) == Nat.add(_,Nat.mul(Nat.sub(row,block),stride)) : Nat}
  %RowMajor.distrib(block,Nat.sub(row,block),stride) : {Nat.mul(row,stride) == _ : Nat}
  %Equal.sym(Nat,Nat.add(block,Nat.sub(row,block)),row,Order.restore_shift(block,row,ordered)) : {Nat.mul(row,stride) == Nat.mul(_,stride) : Nat}
  {==}

def rows_from_product(+row: Nat,+limit: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +product: {Nat.is_le(Nat.mul(row,stride),limit) == True{} : Bool}) -> {Nat.is_lt(row,1n+Nat.div(limit,stride)) == True{} : Bool}:
  Order.lt_succ(row,Nat.div(limit,stride),
    %Division.exact_quotient(row,stride,positive) : {Nat.is_le(_,Nat.div(limit,stride)) == True{} : Bool}
    DivisionOrder.quotient_le(Nat.mul(row,stride),limit,stride,positive,product))

# Virtual top cover: the lane's reads start at or below virtual row 0, or
# lie in the padding above the image.
def virtual_top(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Cover<problem,first,rows,offset,span>,
  +position: Nat,+stride_positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},+after: {Nat.is_le(offset,position) == True{} : Bool},
  +inside: {Nat.is_lt(position,Nat.add(offset,span)) == True{} : Bool}) ->
  Samples.Top<first,Guards.pad(problem),Nat.mul(Nat.sub(Nat.div(position,ConvSpec.out_width(problem)),block(problem,first,rows,offset,span,cover)),Guards.stride(problem)),
    Guards.kernel(problem)>:
  Cover{+block,+first_exact,fits,rows_positive,kernel_fits,+top,bottom} = cover
  match top:
    case Start{+zero}: Samples.Start{zero}
    case Within{+above}:
      +stride = Guards.stride(problem)
      +padding = Guards.pad(problem)
      +width = ConvSpec.out_width(problem)
      +start = Nat.div(offset,width)
      +row = Nat.div(position,width)
      +later = DivisionOrder.quotient_le(offset,position,width,width_positive,after)
      +ordered = Order.transitive(block,start,row,block_le_start(problem,first,block,offset,span,stride_positive,first_exact,Within{above}),later)
      +product = row_product(block,row,stride,first,first_exact,ordered)
      Samples.Below{%Order.le_offset(first,padding,Nat.mul(Nat.sub(row,block),stride)) : {_ == True{} : Bool}
        %product : {Nat.is_le(Nat.add(first,padding),_) == True{} : Bool}
        Order.transitive(Nat.add(first,padding),Nat.mul(start,stride),Nat.mul(row,stride),above,RowMajor.mul_le(start,row,stride,later))}
    case Over{+before,+reads}:
      +stride = Guards.stride(problem)
      +padding = Guards.pad(problem)
      +kernel = Guards.kernel(problem)
      +width = ConvSpec.out_width(problem)
      +row = Nat.div(position,width)
      +last = Nat.div(Nat.sub(Nat.add(offset,span),1n),width)
      +ordered = Order.transitive(block,Nat.div(offset,width),row,block_le_start(problem,first,block,offset,span,stride_positive,first_exact,Over{before,reads}),
        DivisionOrder.quotient_le(offset,position,width,width_positive,after))
      +product = row_product(block,row,stride,first,first_exact,ordered)
      Samples.Above{%product : {Nat.is_le(Nat.add(_,kernel),padding) == True{} : Bool}
        Order.transitive(Nat.add(Nat.mul(row,stride),kernel),Nat.add(Nat.mul(last,stride),kernel),padding,
          Order.add_le(Nat.mul(row,stride),Nat.mul(last,stride),kernel,
            RowMajor.mul_le(row,last,stride,DivisionOrder.quotient_le(position,Nat.sub(Nat.add(offset,span),1n),width,width_positive,last_row(position,offset,span,inside)))),
          reads)}

# Virtual bottom cover: the lane's reads end inside the window unless it ends the image.
def virtual_bottom(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Cover<problem,first,rows,offset,span>,
  +position: Nat,+stride_positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},+after: {Nat.is_le(offset,position) == True{} : Bool},
  +inside: {Nat.is_lt(position,Nat.add(offset,span)) == True{} : Bool}) ->
  Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},{Nat.is_le(Nat.add(Nat.mul(Nat.sub(Nat.div(position,ConvSpec.out_width(problem)),
    block(problem,first,rows,offset,span,cover)),Guards.stride(problem)),Guards.kernel(problem)),Nat.add(Guards.pad(problem),rows)) == True{} : Bool}>:
  Cover{+block,+first_exact,fits,rows_positive,kernel_fits,+top,+bottom} = cover
  match bottom:
    case Inl{+full}: Inl{full}
    case Inr{+below}:
      +stride = Guards.stride(problem)
      +padding = Guards.pad(problem)
      +kernel = Guards.kernel(problem)
      +width = ConvSpec.out_width(problem)
      +row = Nat.div(position,width)
      +last = Nat.div(Nat.sub(Nat.add(offset,span),1n),width)
      +ordered = Order.transitive(block,Nat.div(offset,width),row,block_le_start(problem,first,block,offset,span,stride_positive,first_exact,top),
        DivisionOrder.quotient_le(offset,position,width,width_positive,after))
      +product = row_product(block,row,stride,first,first_exact,ordered)
      +virtual = Nat.mul(Nat.sub(row,block),stride)
      +reach = Order.transitive(Nat.add(Nat.mul(row,stride),kernel),Nat.add(Nat.mul(last,stride),kernel),Nat.add(Nat.add(first,rows),padding),
        Order.add_le(Nat.mul(row,stride),Nat.mul(last,stride),kernel,
          RowMajor.mul_le(row,last,stride,DivisionOrder.quotient_le(position,Nat.sub(Nat.add(offset,span),1n),width,width_positive,last_row(position,offset,span,inside)))),
        below)
      Inr{%Order.commutative(rows,padding) : {Nat.is_le(Nat.add(virtual,kernel),_) == True{} : Bool}
        %Order.le_offset(first,Nat.add(virtual,kernel),Nat.add(rows,padding)) : {_ == True{} : Bool}
        %Addition.associative(first,virtual,kernel) : {Nat.is_le(_,Nat.add(first,Nat.add(rows,padding))) == True{} : Bool}
        %Addition.associative(first,rows,padding) : {Nat.is_le(Nat.add(Nat.add(first,virtual),kernel),_) == True{} : Bool}
        %product : {Nat.is_le(Nat.add(_,kernel),Nat.add(Nat.add(first,rows),padding)) == True{} : Bool}
        reach}

def virtual_inside_when(+height: Nat,+first: Nat,+rows: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+row: Nat,+block: Nat,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +product: {Nat.mul(row,stride) == Nat.add(first,Nat.mul(Nat.sub(row,block),stride)) : Nat},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(rows,Nat.mul(2n,padding))) == True{} : Bool},
  +row_valid: {Nat.is_lt(row,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +bottom: Either<&2,&2,{height == Nat.add(first,rows) : Nat},{Nat.is_le(Nat.add(Nat.mul(Nat.sub(row,block),stride),kernel),Nat.add(padding,rows)) == True{} : Bool}>) ->
  {Nat.is_lt(Nat.sub(row,block),ConvSpec.axis_size(rows,kernel,stride,padding)) == True{} : Bool}:
  match bottom:
    case Inl{+full}:
      +whole = Nat.sub(Nat.add(height,Nat.mul(2n,padding)),kernel)
      +limit = Nat.sub(Nat.add(rows,Nat.mul(2n,padding)),kernel)
      +virtual = Nat.mul(Nat.sub(row,block),stride)
      rows_from_product(Nat.sub(row,block),limit,stride,positive,
        %Order.le_offset(first,virtual,limit) : {_ == True{} : Bool}
        %Order.subtract_from_sum(first,Nat.add(rows,Nat.mul(2n,padding)),kernel,kernel_fits) : {Nat.is_le(Nat.add(first,virtual),_) == True{} : Bool}
        %Addition.associative(first,rows,Nat.mul(2n,padding)) : {Nat.is_le(Nat.add(first,virtual),Nat.sub(_,kernel)) == True{} : Bool}
        %full : {Nat.is_le(Nat.add(first,virtual),Nat.sub(Nat.add(_,Nat.mul(2n,padding)),kernel)) == True{} : Bool}
        %product : {Nat.is_le(_,whole) == True{} : Bool}
        Order.transitive(Nat.mul(row,stride),Nat.mul(Nat.div(whole,stride),stride),whole,
          RowMajor.mul_le(row,Nat.div(whole,stride),stride,Division.lt_succ_le(row,Nat.div(whole,stride),row_valid)),
          DivisionOrder.product_below(whole,stride,positive)))
    case Inr{+fitted}:
      +limit = Nat.sub(Nat.add(rows,Nat.mul(2n,padding)),kernel)
      +virtual = Nat.mul(Nat.sub(row,block),stride)
      +reach = {%Order.commutative(rows,padding) : {Nat.is_le(_,Nat.add(rows,Nat.mul(2n,padding))) == True{} : Bool}
          %Equal.sym(Bool,Nat.is_le(Nat.add(rows,padding),Nat.add(rows,Nat.mul(2n,padding))),Nat.is_le(padding,Nat.mul(2n,padding)),
            Order.le_offset(rows,padding,Nat.mul(2n,padding))) : {_ == True{} : Bool}
          Order.le_add(padding,Nat.mul(1n,padding)) : {Nat.is_le(Nat.add(padding,rows),Nat.add(rows,Nat.mul(2n,padding))) == True{} : Bool}}
      rows_from_product(Nat.sub(row,block),limit,stride,positive,
        Order.transitive(virtual,Nat.sub(Nat.add(padding,rows),kernel),limit,
          %DivisionOrder.subtract_sum(virtual,kernel) : {Nat.is_le(_,Nat.sub(Nat.add(padding,rows),kernel)) == True{} : Bool}
          DivisionOrder.subtract_le(Nat.add(virtual,kernel),Nat.add(padding,rows),kernel,fitted),
          DivisionOrder.subtract_le(Nat.add(padding,rows),Nat.add(rows,Nat.mul(2n,padding)),kernel,reach)))

def virtual_row_inside(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Cover<problem,first,rows,offset,span>,
  +position: Nat,+stride_positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},+after: {Nat.is_le(offset,position) == True{} : Bool},
  +inside: {Nat.is_lt(position,Nat.add(offset,span)) == True{} : Bool},
  +row_valid: {Nat.is_lt(Nat.div(position,ConvSpec.out_width(problem)),
    ConvSpec.axis_size(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(Nat.div(position,ConvSpec.out_width(problem)),block(problem,first,rows,offset,span,cover)),
    ConvSpec.axis_size(rows,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem))) == True{} : Bool}:
  Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +width = ConvSpec.out_width(problem)
  +row = Nat.div(position,width)
  +ordered = Order.transitive(block,Nat.div(offset,width),row,block_le_start(problem,first,block,offset,span,stride_positive,first_exact,top),
    DivisionOrder.quotient_le(offset,position,width,width_positive,after))
  virtual_inside_when(Guards.height(problem),first,rows,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),row,block,stride_positive,
    row_product(block,row,Guards.stride(problem),first,first_exact,ordered),kernel_fits,row_valid,
    virtual_bottom(problem,first,rows,offset,span,Cover{block,first_exact,fits,rows_positive,kernel_fits,top,bottom},position,stride_positive,width_positive,after,inside))

def sub_succ(+value: Nat,+amount: Nat,+ordered: {Nat.is_le(amount,value) == True{} : Bool}) ->
  {Nat.sub(1n+value,amount) == 1n+Nat.sub(value,amount) : Nat}:
  match value amount:
    case 0n 0n: {==}
    case 1n+rest 0n: {==}
    case 0n 1n+less: Empty.absurd({Nat.sub(1n,1n+less) == 1n+Nat.sub(0n,1n+less) : Nat},WordEncoding.false_true(ordered))
    case 1n+ +rest 1n+ +less: sub_succ(rest,less,ordered)

# Position offset + column of the original problem, counted from a window that
# skips block output rows, is its virtual row and the same column.
def window_position(+offset: Nat,+column: Nat,+width: Nat,+block: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +reach: {Nat.is_le(Nat.mul(block,width),offset) == True{} : Bool},
  +ordered: {Nat.is_le(block,Nat.div(Nat.add(offset,column),width)) == True{} : Bool}) ->
  {Nat.add(Nat.sub(offset,Nat.mul(block,width)),column) ==
    Nat.add(Nat.mul(Nat.sub(Nat.div(Nat.add(offset,column),width),block),width),Nat.mod(Nat.add(offset,column),width)) : Nat}:
  +position = Nat.add(offset,column)
  +row = Nat.div(position,width)
  +skipped = Nat.mul(block,width)
  +virtual = Nat.mul(Nat.sub(row,block),width)
  +rest = Nat.mod(position,width)
  %sub_after(offset,skipped,column,reach) : {_ == Nat.add(virtual,rest) : Nat}
  %Equal.sym(Nat,position,Nat.add(Nat.mul(row,width),rest),DivisionOrder.quotient_exact(position,width,positive)) :
    {Nat.sub(_,skipped) == Nat.add(virtual,rest) : Nat}
  %Equal.sym(Nat,Nat.mul(row,width),Nat.add(skipped,virtual),row_product(block,row,width,skipped,{==},ordered)) :
    {Nat.sub(Nat.add(_,rest),skipped) == Nat.add(virtual,rest) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(skipped,virtual),rest),Nat.add(skipped,Nat.add(virtual,rest)),Addition.associative(skipped,virtual,rest)) :
    {Nat.sub(_,skipped) == Nat.add(virtual,rest) : Nat}
  Order.subtract_shift(skipped,Nat.add(virtual,rest))

def top_within(+problem: ConvSpec.Problem,+first: Nat,+offset: Nat,+span: Nat,+inner: Nat,+inner_span: Nat,
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},+after: {Nat.is_le(offset,inner) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(inner,inner_span),Nat.add(offset,span)) == True{} : Bool},
  +top: Top<problem,first,offset,span>) -> Top<problem,first,inner,inner_span>:
  match top:
    case Start{+zero}: Start{zero}
    case Within{+above}:
      Within{Order.transitive(Nat.add(first,Guards.pad(problem)),Nat.mul(Nat.div(offset,ConvSpec.out_width(problem)),Guards.stride(problem)),
        Nat.mul(Nat.div(inner,ConvSpec.out_width(problem)),Guards.stride(problem)),above,
        RowMajor.mul_le(Nat.div(offset,ConvSpec.out_width(problem)),Nat.div(inner,ConvSpec.out_width(problem)),Guards.stride(problem),
          DivisionOrder.quotient_le(offset,inner,ConvSpec.out_width(problem),width_positive,after)))}
    case Over{+before,+reads}:
      +width = ConvSpec.out_width(problem)
      +stride = Guards.stride(problem)
      +outer_row = Nat.div(Nat.sub(Nat.add(offset,span),1n),width)
      +inner_row = Nat.div(Nat.sub(Nat.add(inner,inner_span),1n),width)
      Over{Order.transitive(first,Nat.mul(Nat.div(offset,width),stride),Nat.mul(Nat.div(inner,width),stride),before,
          RowMajor.mul_le(Nat.div(offset,width),Nat.div(inner,width),stride,DivisionOrder.quotient_le(offset,inner,width,width_positive,after))),
        Order.transitive(Nat.add(Nat.mul(inner_row,stride),Guards.kernel(problem)),Nat.add(Nat.mul(outer_row,stride),Guards.kernel(problem)),Guards.pad(problem),
          Order.add_le(Nat.mul(inner_row,stride),Nat.mul(outer_row,stride),Guards.kernel(problem),
            RowMajor.mul_le(inner_row,outer_row,stride,
              DivisionOrder.quotient_le(Nat.sub(Nat.add(inner,inner_span),1n),Nat.sub(Nat.add(offset,span),1n),width,width_positive,
                DivisionOrder.subtract_le(Nat.add(inner,inner_span),Nat.add(offset,span),1n,within)))),
          reads)}

def bottom_within(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+inner: Nat,+inner_span: Nat,
  +width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(inner,inner_span),Nat.add(offset,span)) == True{} : Bool},
  +bottom: Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(offset,span),1n),ConvSpec.out_width(problem)),Guards.stride(problem)),Guards.kernel(problem)),
      Nat.add(Nat.add(first,rows),Guards.pad(problem))) == True{} : Bool}>) ->
  Either<&2,&2,{Guards.height(problem) == Nat.add(first,rows) : Nat},
    {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(inner,inner_span),1n),ConvSpec.out_width(problem)),Guards.stride(problem)),Guards.kernel(problem)),
      Nat.add(Nat.add(first,rows),Guards.pad(problem))) == True{} : Bool}>:
  match bottom:
    case Inl{+full}: Inl{full}
    case Inr{+below}:
      +width = ConvSpec.out_width(problem)
      +stride = Guards.stride(problem)
      +outer_row = Nat.div(Nat.sub(Nat.add(offset,span),1n),width)
      +inner_row = Nat.div(Nat.sub(Nat.add(inner,inner_span),1n),width)
      Inr{Order.transitive(Nat.add(Nat.mul(inner_row,stride),Guards.kernel(problem)),Nat.add(Nat.mul(outer_row,stride),Guards.kernel(problem)),
        Nat.add(Nat.add(first,rows),Guards.pad(problem)),
        Order.add_le(Nat.mul(inner_row,stride),Nat.mul(outer_row,stride),Guards.kernel(problem),
          RowMajor.mul_le(inner_row,outer_row,stride,
            DivisionOrder.quotient_le(Nat.sub(Nat.add(inner,inner_span),1n),Nat.sub(Nat.add(offset,span),1n),width,width_positive,
              DivisionOrder.subtract_le(Nat.add(inner,inner_span),Nat.add(offset,span),1n,within)))),
        below)}

# A window that covers a span covers every span inside it.
def cover_within(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+inner: Nat,+inner_span: Nat,
  +cover: Cover<problem,first,rows,offset,span>,+width_positive: {Nat.is_lt(0n,ConvSpec.out_width(problem)) == True{} : Bool},
  +after: {Nat.is_le(offset,inner) == True{} : Bool},+within: {Nat.is_le(Nat.add(inner,inner_span),Nat.add(offset,span)) == True{} : Bool}) ->
  Cover<problem,first,rows,inner,inner_span>:
  Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  Cover{block,first_exact,fits,rows_positive,kernel_fits,top_within(problem,first,offset,span,inner,inner_span,width_positive,after,within,top),
    bottom_within(problem,first,rows,offset,span,inner,inner_span,width_positive,within,bottom)}
