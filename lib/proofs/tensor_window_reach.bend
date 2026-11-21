# The input rows the segments of a window task read lie inside the rows the
# task holds, and the rows a task holds lie inside the input.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as Extrema
import ./row_major_index.bend as RowMajor
import ./u32_extrema.bend as U32Extrema
import ./u32_comparison.bend as Comparison
import ./tensor_window_piece.bend as Piece
import ../kernel_gemm.bend as KernelGemm
import ./tensor_window_rows.bend as Rows
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./boolean_logic.bend as Boolean
import ./natural_addition.bend as Addition

# ---- Rows of the stacked images ----

# The rows past the rows of its image that a task reads from, for its first
# output row at top, and up to, for its last output row ending at bottom.
def low_rows(+padding: Nat,+top: Nat,+height: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_le(padding,top),Nat.min(height,Nat.sub(top,padding)),0n)

def high_rows(+padding: Nat,+bottom: Nat,+height: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(padding,bottom),Nat.min(height,Nat.sub(bottom,padding)),0n)

def low_inside(+padding: Nat,+top: Nat,+height: Nat) -> {Nat.is_le(low_rows(padding,top,height),height) == True{} : Bool}:
  Extrema.pick_upper(Nat.is_le(padding,top),Nat.min(height,Nat.sub(top,padding)),0n,height,Extrema.minimum_left(height,Nat.sub(top,padding)),Order.zero_le(height))

def high_inside(+padding: Nat,+bottom: Nat,+height: Nat) -> {Nat.is_le(high_rows(padding,bottom,height),height) == True{} : Bool}:
  Extrema.pick_upper(Nat.is_lt(padding,bottom),Nat.min(height,Nat.sub(bottom,padding)),0n,height,Extrema.minimum_left(height,Nat.sub(bottom,padding)),
    Order.zero_le(height))

# A row y that kernel row kernel_row of an output row at top reads, past
# padding, lies at or after the first row of a task from an output row at
# top_first above it.
def low_before(+top_first: Nat,+top: Nat,+kernel_row: Nat,+padding: Nat,+height: Nat,+y: Nat,+ordered: {Nat.is_le(top_first,top) == True{} : Bool},
  +sample: {Nat.add(y,padding) == Nat.add(top,kernel_row) : Nat}) -> {Nat.is_le(low_rows(padding,top_first,height),y) == True{} : Bool}:
  +read = Equal.trans(Nat,Nat.sub(Nat.add(top,kernel_row),padding),Nat.sub(Nat.add(y,padding),padding),y,
    Equal.cong(Nat,Nat,whole => Nat.sub(whole,padding),Nat.add(top,kernel_row),Nat.add(y,padding),Equal.sym(Nat,Nat.add(y,padding),Nat.add(top,kernel_row),sample)),
    DivisionOrder.subtract_sum(y,padding))
  Extrema.pick_upper(Nat.is_le(padding,top_first),Nat.min(height,Nat.sub(top_first,padding)),0n,y,
    Order.transitive(Nat.min(height,Nat.sub(top_first,padding)),Nat.sub(top_first,padding),y,Extrema.minimum_right(height,Nat.sub(top_first,padding)),
      Order.transitive(Nat.sub(top_first,padding),Nat.sub(top,padding),y,DivisionOrder.subtract_le(top_first,top,padding,ordered),
        %read : {Nat.is_le(Nat.sub(top,padding),_) == True{} : Bool}
        DivisionOrder.subtract_le(top,Nat.add(top,kernel_row),padding,Order.le_add(top,kernel_row)))),
    Order.zero_le(y))

# The same row lies before the end row of a task to an output row at top_last
# below it.
def high_after(+top: Nat,+top_last: Nat,+kernel: Nat,+kernel_row: Nat,+padding: Nat,+height: Nat,+y: Nat,+ordered: {Nat.is_le(top,top_last) == True{} : Bool},
  +inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},+stored: {Nat.is_lt(y,height) == True{} : Bool},
  +sample: {Nat.add(y,padding) == Nat.add(top,kernel_row) : Nat}) -> {Nat.is_lt(y,high_rows(padding,Nat.add(top_last,kernel),height)) == True{} : Bool}:
  +bottom = Nat.add(top_last,kernel)
  +sum_below = {%Equal.trans(Nat,Nat.add(top,kernel_row),Nat.add(y,padding),Nat.add(padding,y),Equal.sym(Nat,Nat.add(y,padding),Nat.add(top,kernel_row),sample),
        Order.commutative(y,padding)) : {Nat.is_lt(_,bottom) == True{} : Bool}
    Bounds.nat_trans(Nat.add(top,kernel_row),Nat.add(top,kernel),bottom,Order.offset_lt(top,kernel_row,kernel,inside),Order.add_le(top,top_last,kernel,ordered)) :
    {Nat.is_lt(Nat.add(padding,y),bottom) == True{} : Bool}}
  +padding_below = Order.le_lt_trans(padding,Nat.add(padding,y),bottom,Order.le_add(padding,y),sum_below)
  %Equal.sym(Bool,Nat.is_lt(padding,bottom),True{},padding_below) : {Nat.is_lt(y,Bool.pick(Nat,_,Nat.min(height,Nat.sub(bottom,padding)),0n)) == True{} : Bool}
  Extrema.minimum_strict_lower(height,Nat.sub(bottom,padding),y,stored,
    %Equal.sym(Bool,Nat.is_lt(y,Nat.sub(bottom,padding)),Nat.is_lt(Nat.add(padding,y),bottom),Order.subtraction_lt(bottom,padding,y)) : {_ == True{} : Bool}
    sum_below)

# Two output rows of the same image lie in the order of their offsets in it.
def same_image_order(+low: Nat,+high: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+ordered: {Nat.is_le(low,high) == True{} : Bool},
  +same: {Nat.div(low,divisor) == Nat.div(high,divisor) : Nat}) -> {Nat.is_le(Nat.mod(low,divisor),Nat.mod(high,divisor)) == True{} : Bool}:
  Order.cancel_offset_le(Nat.mul(Nat.div(high,divisor),divisor),Nat.mod(low,divisor),Nat.mod(high,divisor),
    %same : {Nat.is_le(Nat.add(Nat.mul(_,divisor),Nat.mod(low,divisor)),Nat.add(Nat.mul(Nat.div(high,divisor),divisor),Nat.mod(high,divisor))) == True{} : Bool}
    %DivisionOrder.quotient_exact(high,divisor,positive) : {Nat.is_le(Nat.add(Nat.mul(Nat.div(low,divisor),divisor),Nat.mod(low,divisor)),_) == True{} : Bool}
    %DivisionOrder.quotient_exact(low,divisor,positive) : {Nat.is_le(_,high) == True{} : Bool}
    ordered)

# The rows before an image: the next image starts past them.
def next_image(+image: Nat,+later: Nat,+height: Nat,+apart: {Nat.is_lt(image,later) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(image,height),height),Nat.mul(later,height)) == True{} : Bool}:
  %Order.commutative(height,Nat.mul(image,height)) : {Nat.is_le(_,Nat.mul(later,height)) == True{} : Bool}
  RowMajor.mul_le(1n+image,later,height,Order.strict_follow(image,later,apart))

# ---- A stored row lies inside the rows of a task ----

def lower_cases(apart: Bool,+first_row: Nat,+row: Nat,+out_height: Nat,+positive: {Nat.is_lt(0n,out_height) == True{} : Bool},+stride: Nat,+padding: Nat,
  +height: Nat,+kernel_row: Nat,+y: Nat,+ordered: {Nat.is_le(first_row,row) == True{} : Bool},
  +sample: {Nat.add(y,padding) == Nat.add(Nat.mul(Nat.mod(row,out_height),stride),kernel_row) : Nat},
  +classified: {Nat.is_lt(Nat.div(first_row,out_height),Nat.div(row,out_height)) == apart : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.div(first_row,out_height),height),low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height)),
    Nat.add(Nat.mul(Nat.div(row,out_height),height),y)) == True{} : Bool}:
  match apart:
    case True{}:
      +before = Nat.mul(Nat.div(first_row,out_height),height)
      +image = Nat.mul(Nat.div(row,out_height),height)
      +low = low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height)
      Order.transitive(Nat.add(before,low),Nat.add(before,height),Nat.add(image,y),Order.offset_le(before,low,height,low_inside(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height)),
        Order.transitive(Nat.add(before,height),image,Nat.add(image,y),next_image(Nat.div(first_row,out_height),Nat.div(row,out_height),height,classified),Order.le_add(image,y)))
    case False{}:
      +first_image = Nat.div(first_row,out_height)
      +same = Order.antisymmetric(first_image,Nat.div(row,out_height),DivisionOrder.quotient_le(first_row,row,out_height,positive,ordered),
        Order.not_less(first_image,Nat.div(row,out_height),classified))
      +tops = RowMajor.mul_le(Nat.mod(first_row,out_height),Nat.mod(row,out_height),stride,same_image_order(first_row,row,out_height,positive,ordered,same))
      %Equal.sym(Nat,first_image,Nat.div(row,out_height),same) :
        {Nat.is_le(Nat.add(Nat.mul(_,height),low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height)),Nat.add(Nat.mul(Nat.div(row,out_height),height),y)) ==
          True{} : Bool}
      Order.offset_le(Nat.mul(Nat.div(row,out_height),height),low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height),y,
        low_before(Nat.mul(Nat.mod(first_row,out_height),stride),Nat.mul(Nat.mod(row,out_height),stride),kernel_row,padding,height,y,tops,sample))

# A stored row of an output row of a task lies at or after the first row the
# task reads.
def lower_row(+first_row: Nat,+row: Nat,+out_height: Nat,+positive: {Nat.is_lt(0n,out_height) == True{} : Bool},+stride: Nat,+padding: Nat,+height: Nat,
  +kernel_row: Nat,+y: Nat,+ordered: {Nat.is_le(first_row,row) == True{} : Bool},
  +sample: {Nat.add(y,padding) == Nat.add(Nat.mul(Nat.mod(row,out_height),stride),kernel_row) : Nat}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.div(first_row,out_height),height),low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height)),
    Nat.add(Nat.mul(Nat.div(row,out_height),height),y)) == True{} : Bool}:
  lower_cases(Nat.is_lt(Nat.div(first_row,out_height),Nat.div(row,out_height)),first_row,row,out_height,positive,stride,padding,height,kernel_row,y,ordered,sample,{==})

def upper_cases(apart: Bool,+row: Nat,+last_row: Nat,+out_height: Nat,+positive: {Nat.is_lt(0n,out_height) == True{} : Bool},+stride: Nat,+padding: Nat,
  +height: Nat,+kernel: Nat,+kernel_row: Nat,+y: Nat,+ordered: {Nat.is_le(row,last_row) == True{} : Bool},+inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},
  +stored: {Nat.is_lt(y,height) == True{} : Bool},+sample: {Nat.add(y,padding) == Nat.add(Nat.mul(Nat.mod(row,out_height),stride),kernel_row) : Nat},
  +classified: {Nat.is_lt(Nat.div(row,out_height),Nat.div(last_row,out_height)) == apart : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.div(row,out_height),height),y),
    Nat.add(Nat.mul(Nat.div(last_row,out_height),height),high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height))) == True{} : Bool}:
  match apart:
    case True{}:
      +image = Nat.mul(Nat.div(row,out_height),height)
      +last = Nat.mul(Nat.div(last_row,out_height),height)
      Bounds.nat_trans(Nat.add(image,y),Nat.add(image,height),Nat.add(last,high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height)),
        Order.offset_lt(image,y,height,stored),
        Order.transitive(Nat.add(image,height),last,Nat.add(last,high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height)),
          next_image(Nat.div(row,out_height),Nat.div(last_row,out_height),height,classified),
          Order.le_add(last,high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height))))
    case False{}:
      +image = Nat.div(row,out_height)
      +same = Order.antisymmetric(image,Nat.div(last_row,out_height),DivisionOrder.quotient_le(row,last_row,out_height,positive,ordered),
        Order.not_less(image,Nat.div(last_row,out_height),classified))
      +tops = RowMajor.mul_le(Nat.mod(row,out_height),Nat.mod(last_row,out_height),stride,same_image_order(row,last_row,out_height,positive,ordered,same))
      %Equal.sym(Nat,image,Nat.div(last_row,out_height),same) : {Nat.is_lt(Nat.add(Nat.mul(_,height),y),
          Nat.add(Nat.mul(Nat.div(last_row,out_height),height),high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height))) == True{} : Bool}
      Order.offset_lt(Nat.mul(Nat.div(last_row,out_height),height),y,high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height),
        high_after(Nat.mul(Nat.mod(row,out_height),stride),Nat.mul(Nat.mod(last_row,out_height),stride),kernel,kernel_row,padding,height,y,tops,inside,stored,sample))

# A stored row of an output row of a task lies before the end row the task
# reads.
def upper_row(+row: Nat,+last_row: Nat,+out_height: Nat,+positive: {Nat.is_lt(0n,out_height) == True{} : Bool},+stride: Nat,+padding: Nat,+height: Nat,
  +kernel: Nat,+kernel_row: Nat,+y: Nat,+ordered: {Nat.is_le(row,last_row) == True{} : Bool},+inside: {Nat.is_lt(kernel_row,kernel) == True{} : Bool},
  +stored: {Nat.is_lt(y,height) == True{} : Bool},+sample: {Nat.add(y,padding) == Nat.add(Nat.mul(Nat.mod(row,out_height),stride),kernel_row) : Nat}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.div(row,out_height),height),y),
    Nat.add(Nat.mul(Nat.div(last_row,out_height),height),high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height))) == True{} : Bool}:
  upper_cases(Nat.is_lt(Nat.div(row,out_height),Nat.div(last_row,out_height)),row,last_row,out_height,positive,stride,padding,height,kernel,kernel_row,y,ordered,inside,stored,
    sample,{==})

# ---- The rows of a segment ----

# The rows a task reads from and up to, past the rows of the images before.
def start_rows(+first_row: Nat,+out_height: Nat,+stride: Nat,+padding: Nat,+height: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(first_row,out_height),height),low_rows(padding,Nat.mul(Nat.mod(first_row,out_height),stride),height))

def end_rows(+last_row: Nat,+out_height: Nat,+stride: Nat,+padding: Nat,+height: Nat,+kernel: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(last_row,out_height),height),high_rows(padding,Nat.add(Nat.mul(Nat.mod(last_row,out_height),stride),kernel),height))

# A stored input row between the rows a task reads lies inside what it holds.
def row_room(+low: Nat,+high: Nat,+capacity: U32,+image: U32,+height: U32,+y: U32,+width: U32,+groups: U32,+bound: U32,
  +bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+first: Nat,+last: Nat,
  +low_at: {Nat.is_le(low,Nat.mul(first,Nat.mul(U32.to_nat(width),U32.to_nat(groups)))) == True{} : Bool},
  +high_at: {Nat.is_le(Nat.mul(last,Nat.mul(U32.to_nat(width),U32.to_nat(groups))),high) == True{} : Bool},
  +after_rows: {Nat.is_le(first,Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y))) == True{} : Bool},
  +before_rows: {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),last) == True{} : Bool},
  +last_room: {Nat.is_le(last,U32.to_nat(bound)) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(last,Nat.mul(U32.to_nat(width),U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+allocatable: {U32.is_le((width * groups : U32),1073741824) == True{} : Bool}) ->
  Rows.RowRoom<low,high,capacity,image,height,y,width,groups>:
  +w = U32.to_nat(width)
  +g = U32.to_nat(groups)
  +cell = Nat.mul(w,g)
  +rows = Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y))
  +rows_bound = Order.transitive(rows,last,U32.to_nat(bound),Bounds.strict_le(rows,last,before_rows),last_room)
  +next_cells = RowMajor.mul_le(1n+rows,last,cell,Order.strict_follow(rows,last,before_rows))
  +cells_bound = Order.transitive(Nat.mul(rows,cell),Nat.add(cell,Nat.mul(rows,cell)),U32.to_nat(bound),Order.right_le(cell,Nat.mul(rows,cell)),
    Order.transitive(Nat.add(cell,Nat.mul(rows,cell)),Nat.mul(last,cell),U32.to_nat(bound),next_cells,cells_room))
  +start_equal = RowMajor.assoc(rows,w,g)
  +start_bound = {%Equal.sym(Nat,Nat.mul(Nat.mul(rows,w),g),Nat.mul(rows,cell),start_equal) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} cells_bound :
    {Nat.is_le(Nat.mul(Nat.mul(rows,w),g),U32.to_nat(bound)) == True{} : Bool}}
  +wide = {%RowMajor.multiply_commutative(g,Nat.mul(rows,w)) : {Nat.is_le(Nat.mul(rows,w),_) == True{} : Bool} DivisionOrder.le_multiple(g,Nat.mul(rows,w),groups_positive) :
    {Nat.is_le(Nat.mul(rows,w),Nat.mul(Nat.mul(rows,w),g)) == True{} : Bool}}
  +product_bound = Order.transitive(Nat.mul(U32.to_nat(image),U32.to_nat(height)),rows,U32.to_nat(bound),Order.le_add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y)),
    rows_bound)
  +cell_bound = Order.transitive(cell,Nat.mul(last,cell),U32.to_nat(bound),DivisionOrder.le_multiple(last,cell,Order.positive_bound(rows,last,before_rows)),cells_room)
  Rows.RowRoom{
    Order.transitive(low,Nat.mul(first,cell),Nat.mul(Nat.mul(rows,w),g),low_at,
      %Equal.sym(Nat,Nat.mul(Nat.mul(rows,w),g),Nat.mul(rows,cell),start_equal) : {Nat.is_le(Nat.mul(first,cell),_) == True{} : Bool}
      RowMajor.mul_le(first,rows,cell,after_rows)),
    %Equal.sym(Nat,Nat.mul(Nat.mul(rows,w),g),Nat.mul(rows,cell),start_equal) : {Nat.is_le(Nat.add(_,cell),high) == True{} : Bool}
    %Order.commutative(cell,Nat.mul(rows,cell)) : {Nat.is_le(_,high) == True{} : Bool}
    Order.transitive(Nat.add(cell,Nat.mul(rows,cell)),Nat.mul(last,cell),high,next_cells,high_at),
    Arithmetic.product_exact(bound,bound_ok,((image * height + y) * width : U32),groups,Nat.mul(rows,w),g,
      Arithmetic.product_exact(bound,bound_ok,(image * height + y : U32),width,rows,w,
        Arithmetic.sum_exact(bound,bound_ok,(image * height : U32),y,Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(y),
          Arithmetic.product_exact(bound,bound_ok,image,height,U32.to_nat(image),U32.to_nat(height),{==},{==},product_bound),{==},rows_bound),
        {==},Order.transitive(Nat.mul(rows,w),Nat.mul(Nat.mul(rows,w),g),U32.to_nat(bound),wide,start_bound)),
      {==},start_bound),
    Arithmetic.product_exact(bound,bound_ok,width,groups,w,g,{==},{==},cell_bound),
    allocatable}

# Where a segment row lies: its image and top, its rows within the task, and
# room for its words.
type SegmentRow<-bound: U32,-image: U32,-top: U32,-padding: U32,-height: U32,-kernel: U32,-first_row: Nat,-row: Nat,-last_row: Nat,-out_height: Nat,-stride: Nat> is Data:
  SegmentRow{positive: {Nat.is_lt(0n,out_height) == True{} : Bool},image_exact: {U32.to_nat(image) == Nat.div(row,out_height) : Nat},
    top_exact: {U32.to_nat(top) == Nat.mul(Nat.mod(row,out_height),stride) : Nat},after: {Nat.is_le(first_row,row) == True{} : Bool},
    before: {Nat.is_le(row,last_row) == True{} : Bool},sample_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(bound)) == True{} : Bool},
    top_room: {Nat.is_le(Nat.add(Nat.mul(out_height,stride),U32.to_nat(kernel)),U32.to_nat(bound)) == True{} : Bool}}

# The rows a task holds and the room of their cells.
type HeldRows<-low: Nat,-high: Nat,-bound: U32,-width: U32,-groups: U32,-first: Nat,-last: Nat> is Data:
  HeldRows{low_at: {Nat.is_le(low,Nat.mul(first,Nat.mul(U32.to_nat(width),U32.to_nat(groups)))) == True{} : Bool},
    high_at: {Nat.is_le(Nat.mul(last,Nat.mul(U32.to_nat(width),U32.to_nat(groups))),high) == True{} : Bool},
    last_room: {Nat.is_le(last,U32.to_nat(bound)) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.mul(last,Nat.mul(U32.to_nat(width),U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
    groups_positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},allocatable: {U32.is_le((width * groups : U32),1073741824) == True{} : Bool}}

# A kernel row of a segment row that is stored lies inside what the task holds.
def when_row(stored: Bool,+low: Nat,+high: Nat,+capacity: U32,+image: U32,+height: U32,+top: U32,+kernel_row: U32,+padding: U32,+width: U32,+groups: U32,
  +bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+kernel: U32,+first_row: Nat,+row: Nat,+last_row: Nat,+out_height: Nat,+stride: Nat,
  +held: HeldRows<low,high,bound,width,groups,start_rows(first_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height)),end_rows(last_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>,+placed: SegmentRow<bound,image,top,padding,height,kernel,first_row,row,last_row,out_height,stride>,
  +offset: Nat,+kernel_exact: {U32.to_nat(kernel_row) == offset : Nat},+kernel_inside: {Nat.is_lt(offset,U32.to_nat(kernel)) == True{} : Bool},
  +classified: {U32.is_lt((top + kernel_row - padding : U32),height) == stored : Bool}) ->
  Boolean.When(stored,Rows.RowRoom<low,high,capacity,image,height,(top + kernel_row - padding : U32),width,groups>):
  match stored:
    case False{}: Unit{}
    case True{}:
      HeldRows{+low_at,+high_at,+last_room,+cells_room,+groups_positive,+allocatable} = held
      SegmentRow{+positive,+image_exact,+top_exact,+after,+before,+sample_room,+top_room} = placed
      +y = (top + kernel_row - padding : U32)
      +y_value = U32.to_nat(y)
      +p = U32.to_nat(padding)
      +h = U32.to_nat(height)
      +top_value = Nat.mul(Nat.mod(row,out_height),stride)
      +stored_below = Bounds.as_nat_lt(y,height,classified)
      +sum_y = Arithmetic.sum_exact(bound,bound_ok,y,padding,y_value,p,{==},{==},
        Order.transitive(Nat.add(y_value,p),Nat.add(h,p),U32.to_nat(bound),Order.add_le(y_value,h,p,Bounds.strict_le(y_value,h,stored_below)),sample_room))
      +top_below = RowMajor.mul_le(Nat.mod(row,out_height),out_height,stride,Bounds.strict_le(Nat.mod(row,out_height),out_height,DivisionOrder.remainder_small(row,out_height,positive)))
      +sum_top = Arithmetic.sum_exact(bound,bound_ok,top,kernel_row,top_value,offset,top_exact,kernel_exact,
        Order.transitive(Nat.add(top_value,offset),Nat.add(Nat.mul(out_height,stride),U32.to_nat(kernel)),U32.to_nat(bound),
          Order.transitive(Nat.add(top_value,offset),Nat.add(Nat.mul(out_height,stride),offset),Nat.add(Nat.mul(out_height,stride),U32.to_nat(kernel)),
            Order.add_le(top_value,Nat.mul(out_height,stride),offset,top_below),
            Order.offset_le(Nat.mul(out_height,stride),offset,U32.to_nat(kernel),Bounds.strict_le(offset,U32.to_nat(kernel),kernel_inside))),
          top_room))
      +sample = Equal.trans(Nat,Nat.add(y_value,p),U32.to_nat((y + padding : U32)),Nat.add(top_value,offset),Equal.sym(Nat,U32.to_nat((y + padding : U32)),Nat.add(y_value,p),sum_y),
        Equal.trans(Nat,U32.to_nat((y + padding : U32)),U32.to_nat((top + kernel_row : U32)),Nat.add(top_value,offset),
          Equal.cong(U32,Nat,U32.to_nat,(y + padding : U32),(top + kernel_row : U32),Subtraction.add_subtract((top + kernel_row : U32),padding)),sum_top))
      row_room(low,high,capacity,image,height,y,width,groups,bound,bound_ok,start_rows(first_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height)),end_rows(last_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),low_at,high_at,
        %Equal.sym(Nat,U32.to_nat(image),Nat.div(row,out_height),image_exact) : {Nat.is_le(start_rows(first_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height)),Nat.add(Nat.mul(_,h),y_value)) == True{} : Bool}
        lower_row(first_row,row,out_height,positive,stride,p,h,offset,y_value,after,sample),
        %Equal.sym(Nat,U32.to_nat(image),Nat.div(row,out_height),image_exact) : {Nat.is_lt(Nat.add(Nat.mul(_,h),y_value),end_rows(last_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))) == True{} : Bool}
        upper_row(row,last_row,out_height,positive,stride,p,h,U32.to_nat(kernel),offset,y_value,before,kernel_inside,stored_below,sample),
        last_room,cells_room,groups_positive,allocatable)

# The kernel rows of a segment row from kernel_row on, count of them, lie
# inside what the task holds.
def rows_room(count: Nat,+low: Nat,+high: Nat,+capacity: U32,+image: U32,+height: U32,+top: U32,+kernel_row: U32,+padding: U32,+width: U32,+groups: U32,
  +bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},+kernel: U32,+first_row: Nat,+row: Nat,+last_row: Nat,+out_height: Nat,+stride: Nat,
  +held: HeldRows<low,high,bound,width,groups,start_rows(first_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height)),end_rows(last_row,out_height,stride,U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>,+placed: SegmentRow<bound,image,top,padding,height,kernel,first_row,row,last_row,out_height,stride>,
  +offset: Nat,+kernel_exact: {U32.to_nat(kernel_row) == offset : Nat},+kernel_room: {Nat.is_le(Nat.add(offset,count),U32.to_nat(kernel)) == True{} : Bool},
  +kernel_bound: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(bound)) == True{} : Bool}) ->
  Rows.RowsRoom(count,kernel_row,low,high,capacity,image,height,top,padding,width,groups):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +next_room = Order.advance_room(offset,rest,U32.to_nat(kernel),kernel_room)
      KernelGemm.Both{
        when_row(U32.is_lt((top + kernel_row - padding : U32),height),low,high,capacity,image,height,top,kernel_row,padding,width,groups,bound,bound_ok,kernel,
          first_row,row,last_row,out_height,stride,held,placed,offset,kernel_exact,Order.room_index(offset,rest,U32.to_nat(kernel),kernel_room),{==}),
        rows_room(rest,low,high,capacity,image,height,top,(kernel_row + 1 : U32),padding,width,groups,bound,bound_ok,kernel,first_row,row,last_row,out_height,stride,
          held,placed,1n+offset,
          Equal.trans(Nat,U32.to_nat((kernel_row + 1 : U32)),Nat.add(offset,1n),1n+offset,
            Arithmetic.sum_exact(bound,bound_ok,kernel_row,1,offset,1n,kernel_exact,{==},
              Order.transitive(Nat.add(offset,1n),Nat.add(offset,1n+rest),U32.to_nat(bound),Order.offset_le(offset,1n,1n+rest,Order.zero_le(rest)),
                Order.transitive(Nat.add(offset,1n+rest),U32.to_nat(kernel),U32.to_nat(bound),kernel_room,kernel_bound))),
            Addition.successor_width(offset)),
          next_room,kernel_bound)}

# The kernel rows of the segments from position on lie inside what the task
# holds, when the rows of the segments lie between its first and last rows.
def segments_room(fuel: Nat,more: Bool,+position: U32,+end: U32,+first: U32,+low: Nat,+high: Nat,+capacity: U32,+kernel: U32,+stride: U32,+padding: U32,
  +height: U32,+width: U32,+groups: U32,+out_height: U32,+out_width: U32,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,
  +first_row: Nat,+last_row: Nat,+held: HeldRows<low,high,bound,width,groups,start_rows(first_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),end_rows(last_row,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel))>,
  +after: {Nat.is_le(first_row,Nat.div(U32.to_nat(position),U32.to_nat(out_width))) == True{} : Bool},
  +ends: {Nat.is_le(Nat.div(Nat.sub(U32.to_nat(end),1n),U32.to_nat(out_width)),last_row) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(out_width),U32.to_nat(end)),U32.to_nat(bound)) == True{} : Bool},
  +sample_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(bound)) == True{} : Bool},
  +top_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(out_height),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(bound)) == True{} : Bool},
  +agrees: {U32.is_lt(position,end) == more : Bool}) ->
  Piece.SegmentsRoom(fuel,more,position,end,first,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width):
  match fuel more:
    case 0n more: Unit{}
    case 1n+rest False{}: Unit{}
    case 1n+ +rest True{}:
      Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
      +p = U32.to_nat(position)
      +ow = U32.to_nat(out_width)
      +oh = U32.to_nat(out_height)
      +s = U32.to_nat(stride)
      +line = Piece.segment_row(position,out_width)
      +row = Nat.div(p,ow)
      +line_exact = Piece.word_quotient(position,out_width,bound,bound_ok,width_room,width_positive)
      +image_exact = Equal.trans(Nat,U32.to_nat((line / U32.max(out_height,1) : U32)),Nat.div(U32.to_nat(line),oh),Nat.div(row,oh),
        Piece.word_quotient(line,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,v => Nat.div(v,oh),U32.to_nat(line),row,line_exact))
      +remainder = Nat.mod(row,oh)
      +remainder_exact = Equal.trans(Nat,U32.to_nat((line % U32.max(out_height,1) : U32)),Nat.mod(U32.to_nat(line),oh),remainder,
        Piece.word_remainder(line,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,v => Nat.mod(v,oh),U32.to_nat(line),row,line_exact))
      +top_below = RowMajor.mul_le(remainder,oh,s,Bounds.strict_le(remainder,oh,DivisionOrder.remainder_small(row,oh,height_positive)))
      +top_exact = Arithmetic.product_exact(bound,bound_ok,(line % U32.max(out_height,1) : U32),stride,remainder,s,remainder_exact,{==},
        Order.transitive(Nat.mul(remainder,s),Nat.mul(oh,s),U32.to_nat(bound),top_below,
          Order.transitive(Nat.mul(oh,s),Nat.add(Nat.mul(oh,s),U32.to_nat(kernel)),U32.to_nat(bound),Order.le_add(Nat.mul(oh,s),U32.to_nat(kernel)),top_room)))
      +inside = Bounds.as_nat_lt(position,end,agrees)
      +before = Order.transitive(row,Nat.div(Nat.sub(U32.to_nat(end),1n),ow),last_row,
        DivisionOrder.quotient_le(p,Nat.sub(U32.to_nat(end),1n),ow,width_positive,Order.before_last(p,U32.to_nat(end),inside)),ends)
      +stop = Piece.segment_stop(position,end,out_width)
      +stop_exact = Piece.stop_exact(position,end,out_width,bound,bound_ok,width_room,width_positive,end_room,agrees)
      +ahead = {%Equal.sym(Nat,U32.to_nat(stop),Piece.stop_value(position,end,out_width),stop_exact) : {Nat.is_lt(p,_) == True{} : Bool}
        Piece.stop_ahead(position,end,out_width,width_positive,agrees) : {Nat.is_lt(p,U32.to_nat(stop)) == True{} : Bool}}
      +kernel_bound = Order.transitive(U32.to_nat(kernel),Nat.add(Nat.mul(oh,s),U32.to_nat(kernel)),U32.to_nat(bound),Order.right_le(Nat.mul(oh,s),U32.to_nat(kernel)),top_room)
      KernelGemm.Both{
        rows_room(U32.to_nat(kernel),low,high,capacity,(line / U32.max(out_height,1) : U32),height,((line % U32.max(out_height,1)) * stride : U32),0,padding,width,groups,
          bound,bound_ok,kernel,first_row,row,last_row,oh,s,held,SegmentRow{height_positive,image_exact,top_exact,after,before,sample_room,top_room},0n,{==},
          Order.reflexive(U32.to_nat(kernel)),kernel_bound),
        segments_room(rest,U32.is_lt(stop,end),stop,end,first,low,high,capacity,kernel,stride,padding,height,width,groups,out_height,out_width,bound,divisors,
          first_row,last_row,held,
          Order.transitive(first_row,row,Nat.div(U32.to_nat(stop),ow),after,
            DivisionOrder.quotient_le(p,U32.to_nat(stop),ow,width_positive,Bounds.strict_le(p,U32.to_nat(stop),ahead))),
          ends,end_room,sample_room,top_room,{==})}

# ---- The words of a reach ----

# The rows past its image a task reads from, as the word of the reach.
def low_pick(choice: Bool,+padding: U32,+top: U32,+height: U32,+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},
  +top_room: {Nat.is_le(U32.to_nat(top),U32.to_nat(bound)) == True{} : Bool},+classified: {U32.is_le(padding,top) == choice : Bool}) ->
  {U32.to_nat(Bool.pick(U32,choice,U32.min(height,(top - padding : U32)),0)) == low_rows(U32.to_nat(padding),U32.to_nat(top),U32.to_nat(height)) : Nat}:
  match choice:
    case True{}:
      +p = U32.to_nat(padding)
      +t = U32.to_nat(top)
      +h = U32.to_nat(height)
      +agrees = Equal.trans(Bool,Nat.is_le(p,t),U32.is_le(padding,top),True{},Equal.sym(Bool,U32.is_le(padding,top),Nat.is_le(p,t),Comparison.le(padding,top)),classified)
      %Equal.sym(Bool,Nat.is_le(p,t),True{},agrees) : {U32.to_nat(U32.min(height,(top - padding : U32))) == Bool.pick(Nat,_,Nat.min(h,Nat.sub(t,p)),0n) : Nat}
      Equal.trans(Nat,U32.to_nat(U32.min(height,(top - padding : U32))),Nat.min(h,U32.to_nat((top - padding : U32))),Nat.min(h,Nat.sub(t,p)),
        U32Extrema.minimum(height,(top - padding : U32)),
        Equal.cong(Nat,Nat,value => Nat.min(h,value),U32.to_nat((top - padding : U32)),Nat.sub(t,p),
          Arithmetic.difference_exact(bound,bound_ok,top,padding,t,p,{==},{==},agrees,top_room)))
    case False{}:
      +p = U32.to_nat(padding)
      +t = U32.to_nat(top)
      +h = U32.to_nat(height)
      +agrees = Equal.trans(Bool,Nat.is_le(p,t),U32.is_le(padding,top),False{},Equal.sym(Bool,U32.is_le(padding,top),Nat.is_le(p,t),Comparison.le(padding,top)),classified)
      %Equal.sym(Bool,Nat.is_le(p,t),False{},agrees) : {U32.to_nat(0) == Bool.pick(Nat,_,Nat.min(h,Nat.sub(t,p)),0n) : Nat}
      {==}

# The rows past its image a task reads up to, as the word of the reach.
def high_pick(choice: Bool,+padding: U32,+bottom: U32,+height: U32,+bound: U32,+bound_ok: {U32.is_le(bound,2147483648) == True{} : Bool},
  +bottom_room: {Nat.is_le(U32.to_nat(bottom),U32.to_nat(bound)) == True{} : Bool},+classified: {U32.is_lt(padding,bottom) == choice : Bool}) ->
  {U32.to_nat(Bool.pick(U32,choice,U32.min(height,(bottom - padding : U32)),0)) == high_rows(U32.to_nat(padding),U32.to_nat(bottom),U32.to_nat(height)) : Nat}:
  match choice:
    case True{}:
      +p = U32.to_nat(padding)
      +t = U32.to_nat(bottom)
      +h = U32.to_nat(height)
      +agrees = Equal.trans(Bool,Nat.is_lt(p,t),U32.is_lt(padding,bottom),True{},Equal.sym(Bool,U32.is_lt(padding,bottom),Nat.is_lt(p,t),Comparison.lt(padding,bottom)),classified)
      %Equal.sym(Bool,Nat.is_lt(p,t),True{},agrees) : {U32.to_nat(U32.min(height,(bottom - padding : U32))) == Bool.pick(Nat,_,Nat.min(h,Nat.sub(t,p)),0n) : Nat}
      Equal.trans(Nat,U32.to_nat(U32.min(height,(bottom - padding : U32))),Nat.min(h,U32.to_nat((bottom - padding : U32))),Nat.min(h,Nat.sub(t,p)),
        U32Extrema.minimum(height,(bottom - padding : U32)),
        Equal.cong(Nat,Nat,value => Nat.min(h,value),U32.to_nat((bottom - padding : U32)),Nat.sub(t,p),
          Arithmetic.difference_exact(bound,bound_ok,bottom,padding,t,p,{==},{==},Bounds.strict_le(p,t,agrees),bottom_room)))
    case False{}:
      +p = U32.to_nat(padding)
      +t = U32.to_nat(bottom)
      +h = U32.to_nat(height)
      +agrees = Equal.trans(Bool,Nat.is_lt(p,t),U32.is_lt(padding,bottom),False{},Equal.sym(Bool,U32.is_lt(padding,bottom),Nat.is_lt(p,t),Comparison.lt(padding,bottom)),classified)
      %Equal.sym(Bool,Nat.is_lt(p,t),False{},agrees) : {U32.to_nat(0) == Bool.pick(Nat,_,Nat.min(h,Nat.sub(t,p)),0n) : Nat}
      {==}

# The room of the words of the reaches of the tasks over images images.
type ReachRoom<-bound: U32,-kernel: U32,-stride: U32,-padding: U32,-height: U32,-width: U32,-groups: U32,-out_height: U32,-images: Nat> is Data:
  ReachRoom{top_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(out_height),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(bound)) == True{} : Bool},
    sample_room: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(bound)) == True{} : Bool},
    rows_room: {Nat.is_le(Nat.mul(images,U32.to_nat(height)),U32.to_nat(bound)) == True{} : Bool},
    row_room: {Nat.is_le(Nat.mul(U32.to_nat(width),U32.to_nat(groups)),U32.to_nat(bound)) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))),U32.to_nat(bound)) == True{} : Bool},
    allocatable: {U32.is_le((width * groups : U32),1073741824) == True{} : Bool}}

# Rows of an image below the rows of the images: the rows before it and some
# of its own.
def image_rows(+image: Nat,+images: Nat,+height: Nat,+rows: Nat,+below: {Nat.is_lt(image,images) == True{} : Bool},+inside: {Nat.is_le(rows,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(image,height),rows),Nat.mul(images,height)) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(image,height),rows),Nat.add(height,Nat.mul(image,height)),Nat.mul(images,height),
    %Order.commutative(Nat.mul(image,height),height) : {Nat.is_le(Nat.add(Nat.mul(image,height),rows),_) == True{} : Bool}
    Order.offset_le(Nat.mul(image,height),rows,height,inside),
    RowMajor.mul_le(1n+image,images,height,Order.strict_follow(image,images,below)))

# The cell a task reads from, as the word of the reach, for its first output
# row row.
def low_exact(+row: U32,+row_value: Nat,+row_exact: {U32.to_nat(row) == row_value : Nat},+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,
  +out_height: U32,+out_width: U32,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,
  +room: ReachRoom<bound,kernel,stride,padding,height,width,groups,out_height,images>,
  +below: {Nat.is_lt(Nat.div(row_value,U32.to_nat(out_height)),images) == True{} : Bool}) ->
  {U32.to_nat(((row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_le(padding,((row % U32.max(out_height,1)) * stride : U32)),
      U32.min(height,(((row % U32.max(out_height,1)) * stride : U32) - padding : U32)),0)) * (width * groups : U32) : U32)) ==
    Nat.mul(start_rows(row_value,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))) : Nat}:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  +oh = U32.to_nat(out_height)
  +s = U32.to_nat(stride)
  +p = U32.to_nat(padding)
  +h = U32.to_nat(height)
  +cell = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +image = Nat.div(row_value,oh)
  +remainder = Nat.mod(row_value,oh)
  +top = ((row % U32.max(out_height,1)) * stride : U32)
  +top_value = Nat.mul(remainder,s)
  +image_exact = Equal.trans(Nat,U32.to_nat((row / U32.max(out_height,1) : U32)),Nat.div(U32.to_nat(row),oh),image,
    Piece.word_quotient(row,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,value => Nat.div(value,oh),U32.to_nat(row),row_value,row_exact))
  +remainder_exact = Equal.trans(Nat,U32.to_nat((row % U32.max(out_height,1) : U32)),Nat.mod(U32.to_nat(row),oh),remainder,
    Piece.word_remainder(row,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,value => Nat.mod(value,oh),U32.to_nat(row),row_value,row_exact))
  +top_bound = Order.transitive(top_value,Nat.mul(oh,s),U32.to_nat(bound),
    RowMajor.mul_le(remainder,oh,s,Bounds.strict_le(remainder,oh,DivisionOrder.remainder_small(row_value,oh,height_positive))),
    Order.transitive(Nat.mul(oh,s),Nat.add(Nat.mul(oh,s),U32.to_nat(kernel)),U32.to_nat(bound),Order.le_add(Nat.mul(oh,s),U32.to_nat(kernel)),top_room))
  +top_exact = Arithmetic.product_exact(bound,bound_ok,(row % U32.max(out_height,1) : U32),stride,remainder,s,remainder_exact,{==},top_bound)
  +pick = Bool.pick(U32,U32.is_le(padding,top),U32.min(height,(top - padding : U32)),0)
  +pick_exact = Equal.trans(Nat,U32.to_nat(pick),low_rows(p,U32.to_nat(top),h),low_rows(p,top_value,h),
    low_pick(U32.is_le(padding,top),padding,top,height,bound,bound_ok,
      {%Equal.sym(Nat,U32.to_nat(top),top_value,top_exact) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} top_bound :
        {Nat.is_le(U32.to_nat(top),U32.to_nat(bound)) == True{} : Bool}},{==}),
    Equal.cong(Nat,Nat,value => low_rows(p,value,h),U32.to_nat(top),top_value,top_exact))
  +rows = start_rows(row_value,oh,s,p,h)
  +rows_below = image_rows(image,images,h,low_rows(p,top_value,h),below,low_inside(p,top_value,h))
  +rows_bound = Order.transitive(rows,Nat.mul(images,h),U32.to_nat(bound),rows_below,rows_room)
  +before_bound = Order.transitive(Nat.mul(image,h),rows,U32.to_nat(bound),Order.le_add(Nat.mul(image,h),low_rows(p,top_value,h)),rows_bound)
  Arithmetic.product_exact(bound,bound_ok,(row / U32.max(out_height,1) * height + pick : U32),(width * groups : U32),rows,cell,
    Arithmetic.sum_exact(bound,bound_ok,(row / U32.max(out_height,1) * height : U32),pick,Nat.mul(image,h),low_rows(p,top_value,h),
      Arithmetic.product_exact(bound,bound_ok,(row / U32.max(out_height,1) : U32),height,image,h,image_exact,{==},before_bound),pick_exact,rows_bound),
    Arithmetic.product_exact(bound,bound_ok,width,groups,U32.to_nat(width),U32.to_nat(groups),{==},{==},row_room),
    Order.transitive(Nat.mul(rows,cell),Nat.mul(Nat.mul(images,h),cell),U32.to_nat(bound),RowMajor.mul_le(rows,Nat.mul(images,h),cell,rows_below),cells_room))

# The cell a task reads up to, as the word of the reach, for its last output
# row row.
def high_exact(+row: U32,+row_value: Nat,+row_exact: {U32.to_nat(row) == row_value : Nat},+kernel: U32,+stride: U32,+padding: U32,+height: U32,+width: U32,+groups: U32,
  +out_height: U32,+out_width: U32,+bound: U32,+divisors: Piece.Divisors<bound,groups,out_height,out_width>,+images: Nat,
  +room: ReachRoom<bound,kernel,stride,padding,height,width,groups,out_height,images>,
  +below: {Nat.is_lt(Nat.div(row_value,U32.to_nat(out_height)),images) == True{} : Bool}) ->
  {U32.to_nat(((row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_lt(padding,((row % U32.max(out_height,1)) * stride + kernel : U32)),
      U32.min(height,(((row % U32.max(out_height,1)) * stride + kernel : U32) - padding : U32)),0)) * (width * groups : U32) : U32)) ==
    Nat.mul(end_rows(row_value,U32.to_nat(out_height),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(height),U32.to_nat(kernel)),Nat.mul(U32.to_nat(width),U32.to_nat(groups))) : Nat}:
  Piece.Divisors{+bound_ok,+groups_room,+groups_positive,+height_room,+height_positive,+width_room,+width_positive} = divisors
  ReachRoom{+top_room,+sample_room,+rows_room,+row_room,+cells_room,+allocatable} = room
  +oh = U32.to_nat(out_height)
  +s = U32.to_nat(stride)
  +p = U32.to_nat(padding)
  +h = U32.to_nat(height)
  +k = U32.to_nat(kernel)
  +cell = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +image = Nat.div(row_value,oh)
  +remainder = Nat.mod(row_value,oh)
  +bottom = ((row % U32.max(out_height,1)) * stride + kernel : U32)
  +bottom_value = Nat.add(Nat.mul(remainder,s),k)
  +image_exact = Equal.trans(Nat,U32.to_nat((row / U32.max(out_height,1) : U32)),Nat.div(U32.to_nat(row),oh),image,
    Piece.word_quotient(row,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,value => Nat.div(value,oh),U32.to_nat(row),row_value,row_exact))
  +remainder_exact = Equal.trans(Nat,U32.to_nat((row % U32.max(out_height,1) : U32)),Nat.mod(U32.to_nat(row),oh),remainder,
    Piece.word_remainder(row,out_height,bound,bound_ok,height_room,height_positive),Equal.cong(Nat,Nat,value => Nat.mod(value,oh),U32.to_nat(row),row_value,row_exact))
  +bottom_bound = Order.transitive(bottom_value,Nat.add(Nat.mul(oh,s),k),U32.to_nat(bound),
    Order.add_le(Nat.mul(remainder,s),Nat.mul(oh,s),k,RowMajor.mul_le(remainder,oh,s,Bounds.strict_le(remainder,oh,DivisionOrder.remainder_small(row_value,oh,height_positive)))),
    top_room)
  +top_bound = Order.transitive(Nat.mul(remainder,s),bottom_value,U32.to_nat(bound),Order.le_add(Nat.mul(remainder,s),k),bottom_bound)
  +bottom_exact = Arithmetic.sum_exact(bound,bound_ok,((row % U32.max(out_height,1)) * stride : U32),kernel,Nat.mul(remainder,s),k,
    Arithmetic.product_exact(bound,bound_ok,(row % U32.max(out_height,1) : U32),stride,remainder,s,remainder_exact,{==},top_bound),{==},bottom_bound)
  +pick = Bool.pick(U32,U32.is_lt(padding,bottom),U32.min(height,(bottom - padding : U32)),0)
  +pick_exact = Equal.trans(Nat,U32.to_nat(pick),high_rows(p,U32.to_nat(bottom),h),high_rows(p,bottom_value,h),
    high_pick(U32.is_lt(padding,bottom),padding,bottom,height,bound,bound_ok,
      {%Equal.sym(Nat,U32.to_nat(bottom),bottom_value,bottom_exact) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} bottom_bound :
        {Nat.is_le(U32.to_nat(bottom),U32.to_nat(bound)) == True{} : Bool}},{==}),
    Equal.cong(Nat,Nat,value => high_rows(p,value,h),U32.to_nat(bottom),bottom_value,bottom_exact))
  +rows = end_rows(row_value,oh,s,p,h,k)
  +rows_below = image_rows(image,images,h,high_rows(p,bottom_value,h),below,high_inside(p,bottom_value,h))
  +rows_bound = Order.transitive(rows,Nat.mul(images,h),U32.to_nat(bound),rows_below,rows_room)
  +before_bound = Order.transitive(Nat.mul(image,h),rows,U32.to_nat(bound),Order.le_add(Nat.mul(image,h),high_rows(p,bottom_value,h)),rows_bound)
  Arithmetic.product_exact(bound,bound_ok,(row / U32.max(out_height,1) * height + pick : U32),(width * groups : U32),rows,cell,
    Arithmetic.sum_exact(bound,bound_ok,(row / U32.max(out_height,1) * height : U32),pick,Nat.mul(image,h),high_rows(p,bottom_value,h),
      Arithmetic.product_exact(bound,bound_ok,(row / U32.max(out_height,1) : U32),height,image,h,image_exact,{==},before_bound),pick_exact,rows_bound),
    Arithmetic.product_exact(bound,bound_ok,width,groups,U32.to_nat(width),U32.to_nat(groups),{==},{==},row_room),
    Order.transitive(Nat.mul(rows,cell),Nat.mul(Nat.mul(images,h),cell),U32.to_nat(bound),RowMajor.mul_le(rows,Nat.mul(images,h),cell,rows_below),cells_room))
