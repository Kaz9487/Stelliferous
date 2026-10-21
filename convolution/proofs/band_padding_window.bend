# A band leaf whose window holds no image row: its outputs read only the
# padding above the image. The leaf packs instead the one row at the stride
# position of its first output; that row lies in its core and covers it.
import Base
import ../specification.bend as ConvSpec
import ./convolution_guard_facts.bend as Guards
import ./window_positions.bend as Positions
import ./window_samples.bend as Samples
import ./band_window_contents.bend as WindowContents
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/tensor_cursor_refinement.bend as CursorLaws
import ../../lib/proofs/word_encoding.bend as WordEncoding

# A sum that does not exceed its first term adds nothing.
def absorbed(+base: Nat,+extra: Nat,+bounded: {Nat.is_le(Nat.add(base,extra),base) == True{} : Bool}) -> {extra == 0n : Nat}:
  Order.zero_below(extra,
    %Order.le_offset(base,extra,0n) : {_ == True{} : Bool}
    %Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base)) : {Nat.is_le(Nat.add(base,extra),_) == True{} : Bool}
    bounded)

# A positive kernel never ends at or before the row it starts on.
def kernel_absent(+row: Nat,+kernel: Nat,+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +bounded: {Nat.is_le(Nat.add(row,kernel),row) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(%absorbed(row,kernel,bounded) : {Nat.is_lt(0n,_) == True{} : Bool} kernel_positive)

def strictly_below_itself(+value: Nat,+below: {Nat.is_lt(value,value) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,value),True{},Equal.sym(Bool,Nat.is_lt(value,value),False{},CursorLaws.less_reflexive(value)),below))

# Reads that pass the top padding but end inside the image leave the window a row.
def reads_inside(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+output_first: Nat,+output_end: Nat,
  +kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +beyond: {Nat.is_lt(padding,Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.sub(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding),height) == True{} : Bool},
  +empty: {Nat.is_le(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) == True{} : Bool}) -> Empty:
  +first = Nat.mul(output_first,stride)
  +last = Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel)
  +reach = Nat.sub(last,padding)
  +needed = Nat.min(height,Nat.sub(first,padding))
  +start = WindowContents.window_first(height,output_first,stride,padding)
  +reach_room = Order.transitive(reach,start,Nat.sub(first,padding),
    Order.transitive(reach,Nat.min(height,reach),start,Extrema.minimum_lower(height,reach,reach,inside,Order.reflexive(reach)),empty),
    Order.transitive(start,needed,Nat.sub(first,padding),WindowContents.aligned_below(needed,stride),Extrema.minimum_right(height,Nat.sub(first,padding))))
  +first_positive = Bounds.nat_trans(0n,reach,Nat.sub(first,padding),Order.positive_difference(last,padding,beyond),reach_room)
  +first_after = Bounds.strict_le(padding,first,Order.difference_positive(padding,first,first_positive))
  +last_first = {%Order.restore_shift(padding,last,Bounds.strict_le(padding,last,beyond)) : {Nat.is_le(_,first) == True{} : Bool}
    %Order.restore_shift(padding,first,first_after) : {Nat.is_le(Nat.add(padding,reach),_) == True{} : Bool}
    Order.offset_le(padding,reach,Nat.sub(first,padding),reach_room) : {Nat.is_le(last,first) == True{} : Bool}}
  +first_last = Order.add_le(first,Nat.mul(Nat.sub(output_end,1n),stride),kernel,
    RowMajor.mul_le(output_first,Nat.sub(output_end,1n),stride,Order.before_last(output_first,output_end,ordered)))
  kernel_absent(first,kernel,kernel_positive,Order.transitive(Nat.add(first,kernel),last,first,first_last,last_first))

# Reads that end below the image leave the window a row too: such a window
# starts at the image end, which then is a stride multiple, while the first
# output, at the stride position of a core row, starts inside the image.
def reads_below(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+core_first: Nat,+output_first: Nat,+output_end: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +first_ceiling: {output_first == Ceiling.ceiling(core_first,stride) : Nat},+core_first_inside: {Nat.is_le(core_first,height) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +outside: {Nat.is_le(height,Nat.sub(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding)) == True{} : Bool},
  +empty: {Nat.is_le(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) == True{} : Bool}) -> Empty:
  +first = Nat.mul(output_first,stride)
  +reach = Nat.sub(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding)
  +needed = Nat.min(height,Nat.sub(first,padding))
  +start = WindowContents.window_first(height,output_first,stride,padding)
  +start_high = Order.transitive(height,Nat.min(height,reach),start,Extrema.minimum_lower(height,reach,height,Order.reflexive(height),outside),empty)
  +start_blocks = {%WindowContents.aligned_exact(needed,stride,stride_positive) : {Nat.is_le(height,_) == True{} : Bool}
    start_high : {Nat.is_le(height,Nat.mul(Nat.div(needed,stride),stride)) == True{} : Bool}}
  +needed_high = Order.transitive(height,Nat.mul(Nat.div(needed,stride),stride),needed,start_blocks,DivisionOrder.product_below(needed,stride,stride_positive))
  +needed_exact = Order.antisymmetric(needed,height,Extrema.minimum_left(height,Nat.sub(first,padding)),needed_high)
  +height_blocks = {%needed_exact : {Nat.is_le(height,Nat.mul(Nat.div(_,stride),stride)) == True{} : Bool}
    start_blocks : {Nat.is_le(height,Nat.mul(Nat.div(height,stride),stride)) == True{} : Bool}}
  +first_blocks = {%Equal.sym(Nat,output_first,Ceiling.ceiling(core_first,stride),first_ceiling) : {Nat.is_le(Nat.mul(_,stride),Nat.mul(Nat.div(height,stride),stride)) == True{} : Bool}
    RowMajor.mul_le(Ceiling.ceiling(core_first,stride),Nat.div(height,stride),stride,
      Ceiling.minimal(core_first,stride,Nat.div(height,stride),stride_positive,
        Order.transitive(core_first,height,Nat.mul(Nat.div(height,stride),stride),core_first_inside,height_blocks))) :
    {Nat.is_le(first,Nat.mul(Nat.div(height,stride),stride)) == True{} : Bool}}
  +first_low = Order.transitive(first,Nat.mul(Nat.div(height,stride),stride),height,first_blocks,DivisionOrder.product_below(height,stride,stride_positive))
  +clear = Order.transitive(height,needed,Nat.sub(first,padding),needed_high,Extrema.minimum_right(height,Nat.sub(first,padding)))
  +padding_before = Bounds.strict_le(padding,first,Order.difference_positive(padding,first,Bounds.nat_trans(0n,height,Nat.sub(first,padding),height_positive,clear)))
  +shifted = {%Order.restore_shift(padding,first,padding_before) : {Nat.is_le(Nat.add(padding,height),_) == True{} : Bool}
    Order.offset_le(padding,height,Nat.sub(first,padding),clear) : {Nat.is_le(Nat.add(padding,height),first) == True{} : Bool}}
  +padding_zero = absorbed(height,padding,
    %Order.commutative(padding,height) : {Nat.is_le(_,height) == True{} : Bool}
    Order.transitive(Nat.add(padding,height),first,height,shifted,first_low))
  +span = Nat.sub(Nat.add(height,Nat.mul(2n,padding)),kernel)
  +first_span = RowMajor.mul_le(output_first,Nat.div(span,stride),stride,
    Division.lt_succ_le(output_first,Nat.div(span,stride),Bounds.nat_trans(output_first,output_end,1n+Nat.div(span,stride),ordered,bounded)))
  +first_room = Order.transitive(first,Nat.mul(Nat.div(span,stride),stride),span,first_span,DivisionOrder.product_below(span,stride,stride_positive))
  +reads_room = {%Order.restore_shift(kernel,Nat.add(height,Nat.mul(2n,padding)),kernel_fits) : {Nat.is_le(Nat.add(first,kernel),_) == True{} : Bool}
    %Order.commutative(span,kernel) : {Nat.is_le(Nat.add(first,kernel),_) == True{} : Bool}
    Order.add_le(first,span,kernel,first_room) : {Nat.is_le(Nat.add(first,kernel),Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool}}
  +unpadded = {%padding_zero : {Nat.is_le(Nat.add(first,kernel),Nat.add(height,Nat.mul(2n,_))) == True{} : Bool}
    reads_room : {Nat.is_le(Nat.add(first,kernel),Nat.add(height,0n)) == True{} : Bool}}
  +height_low = {%padding_zero : {Nat.is_le(Nat.add(_,height),first) == True{} : Bool} shifted : {Nat.is_le(height,first) == True{} : Bool}}
  kernel_absent(first,kernel,kernel_positive,
    Order.transitive(Nat.add(first,kernel),height,first,
      %Addition.zero_right(height) : {Nat.is_le(Nat.add(first,kernel),_) == True{} : Bool} unpadded,
      height_low))

def reads_beyond(inside: Bool,+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+core_first: Nat,+output_first: Nat,+output_end: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +first_ceiling: {output_first == Ceiling.ceiling(core_first,stride) : Nat},+core_first_inside: {Nat.is_le(core_first,height) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +guard: {Nat.is_le(Nat.sub(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding),height) == inside : Bool},
  +beyond: {Nat.is_lt(padding,Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel)) == True{} : Bool},
  +empty: {Nat.is_le(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) == True{} : Bool}) -> Empty:
  match inside:
    case True{}: reads_inside(height,kernel,stride,padding,output_first,output_end,kernel_positive,ordered,beyond,guard,empty)
    case False{}:
      reads_below(height,kernel,stride,padding,core_first,output_first,output_end,stride_positive,kernel_positive,height_positive,kernel_fits,first_ceiling,core_first_inside,ordered,bounded,
        WindowContents.not_le(Nat.sub(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding),height,guard),empty)

def reads_when(reached: Bool,+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+core_first: Nat,+output_first: Nat,+output_end: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +first_ceiling: {output_first == Ceiling.ceiling(core_first,stride) : Nat},+core_first_inside: {Nat.is_le(core_first,height) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +guard: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding) == reached : Bool},
  +empty: {Nat.is_le(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding) == True{} : Bool}:
  match reached:
    case True{}: guard
    case False{}:
      +last = Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel)
      +beyond = {%Equal.sym(Bool,Nat.is_lt(padding,last),Bool.not(Nat.is_le(last,padding)),Order.strict_complement(padding,last)) : {_ == True{} : Bool}
        %Equal.sym(Bool,Nat.is_le(last,padding),False{},guard) : {Bool.not(_) == True{} : Bool}
        {==} : {Nat.is_lt(padding,last) == True{} : Bool}}
      Empty.absurd({Nat.is_le(last,padding) == True{} : Bool},
        reads_beyond(Nat.is_le(Nat.sub(last,padding),height),height,kernel,stride,padding,core_first,output_first,output_end,stride_positive,kernel_positive,height_positive,
          kernel_fits,first_ceiling,core_first_inside,ordered,bounded,{==},beyond,empty))

# When the window of an active leaf holds no row, every read of its outputs
# lies in the padding above the image.
def padding_reads(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+core_first: Nat,+output_first: Nat,+output_end: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +first_ceiling: {output_first == Ceiling.ceiling(core_first,stride) : Nat},+core_first_inside: {Nat.is_le(core_first,height) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +bounded: {Nat.is_le(output_end,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +empty: {Nat.is_le(WindowContents.window_end(height,output_end,stride,kernel,padding),WindowContents.window_first(height,output_first,stride,padding)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding) == True{} : Bool}:
  reads_when(Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding),height,kernel,stride,padding,core_first,output_first,output_end,
    stride_positive,kernel_positive,height_positive,kernel_fits,first_ceiling,core_first_inside,ordered,bounded,{==},empty)

def minimum_when(smaller: Bool,+left: Nat,+right: Nat,+choice: {Nat.is_lt(left,right) == smaller : Bool},
  +below: {Nat.is_lt(Nat.min(left,right),left) == True{} : Bool}) -> {Nat.min(left,right) == right : Nat}:
  match smaller:
    case True{}:
      +picked = Equal.trans(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),left,Extrema.minimum_choice(left,right),
        Equal.cong(Bool,Nat,value => Bool.pick(Nat,value,left,right),Nat.is_lt(left,right),True{},choice))
      Empty.absurd({Nat.min(left,right) == right : Nat},strictly_below_itself(left,%picked : {Nat.is_lt(_,left) == True{} : Bool} below))
    case False{}:
      Equal.trans(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),right,Extrema.minimum_choice(left,right),
        Equal.cong(Bool,Nat,value => Bool.pick(Nat,value,left,right),Nat.is_lt(left,right),False{},choice))

# A minimum below its left bound is its right bound.
def minimum_below_left(+left: Nat,+right: Nat,+below: {Nat.is_lt(Nat.min(left,right),left) == True{} : Bool}) -> {Nat.min(left,right) == right : Nat}:
  minimum_when(Nat.is_lt(left,right),left,right,{==},below)

# How the output rows of a core end: at the output height for the core that
# ends the image, else at the ceiling of the core end.
type Ending<-core_end: Nat,-height: Nat,-stride: Nat,-output_end: Nat,-output_height: Nat> is Data:
  Last{whole: {core_end == height : Nat},full: {output_end == output_height : Nat}}
  Inner{ceiling: {Nat.is_le(output_end,Ceiling.ceiling(core_end,stride)) == True{} : Bool}}

# A core that ends before the image end owns output rows below its ceiling:
# the first output's stride position lies before the core end.
def row_before_boundary(+core_end: Nat,+stride: Nat,+output_first: Nat,+output_end: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},+ending: {Nat.is_le(output_end,Ceiling.ceiling(core_end,stride)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(output_first,stride),core_end) == True{} : Bool}:
  %Ceiling.upper_comparison(core_end,stride,output_first,stride_positive) : {_ == True{} : Bool}
  Bounds.nat_trans(output_first,output_end,Ceiling.ceiling(core_end,stride),ordered,ending)

# A last core owns every output row up to the output height. When those rows
# read only the top padding, the stride exceeds the padded image, so the first
# output is row 0.
def last_band_first(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+previous: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +ordered: {Nat.is_lt(1n+previous,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +reads: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(ConvSpec.axis_size(height,kernel,stride,padding),1n),stride),kernel),padding) == True{} : Bool}) -> Empty:
  +total = Nat.add(height,Nat.mul(2n,padding))
  +span = Nat.sub(total,kernel)
  +quotient = Nat.div(span,stride)
  +blocks = Nat.mul(quotient,stride)
  +reads_blocks = {%Order.subtract_zero(quotient) : {Nat.is_le(Nat.add(Nat.mul(_,stride),kernel),padding) == True{} : Bool}
    reads : {Nat.is_le(Nat.add(blocks,kernel),padding) == True{} : Bool}}
  +stride_blocks = Order.transitive(stride,Nat.mul(1n+previous,stride),blocks,Order.le_add(stride,Nat.mul(previous,stride)),
    RowMajor.mul_le(1n+previous,quotient,stride,Division.lt_succ_le(1n+previous,quotient,ordered)))
  +stride_reads = Order.transitive(Nat.add(stride,kernel),Nat.add(blocks,kernel),padding,Order.add_le(stride,blocks,kernel,stride_blocks),reads_blocks)
  +stride_inside = {%Addition.zero_right(stride) : {Nat.is_lt(_,Nat.add(stride,kernel)) == True{} : Bool}
    Order.offset_lt(stride,0n,kernel,kernel_positive) : {Nat.is_lt(stride,Nat.add(stride,kernel)) == True{} : Bool}}
  +stride_small = Bounds.nat_trans(stride,Nat.add(stride,kernel),padding,stride_inside,stride_reads)
  +span_below = {%Order.restore_shift(kernel,total,kernel_fits) : {Nat.is_lt(_,Nat.add(kernel,Nat.add(stride,blocks))) == True{} : Bool}
    Order.offset_lt(kernel,span,Nat.add(stride,blocks),DivisionOrder.next_above(span,stride,stride_positive)) :
    {Nat.is_lt(total,Nat.add(kernel,Nat.add(stride,blocks))) == True{} : Bool}}
  +swapped = {%RowMajor.add_swap(kernel,stride,blocks) : {Nat.is_lt(total,_) == True{} : Bool}
    span_below : {Nat.is_lt(total,Nat.add(stride,Nat.add(kernel,blocks))) == True{} : Bool}}
  +kernel_blocks = {%Order.commutative(blocks,kernel) : {Nat.is_le(_,padding) == True{} : Bool} reads_blocks : {Nat.is_le(Nat.add(kernel,blocks),padding) == True{} : Bool}}
  +total_below = Bounds.nat_trans(total,Nat.add(stride,Nat.add(kernel,blocks)),Nat.add(stride,padding),swapped,Order.offset_le(stride,Nat.add(kernel,blocks),padding,kernel_blocks))
  +doubled = {%Equal.sym(Nat,Nat.mul(2n,padding),Nat.add(padding,padding),Order.twice(padding)) : {Nat.is_lt(Nat.add(stride,padding),_) == True{} : Bool}
    Order.add_lt(stride,padding,padding,stride_small) : {Nat.is_lt(Nat.add(stride,padding),Nat.mul(2n,padding)) == True{} : Bool}}
  strictly_below_itself(total,
    Bounds.nat_trans(total,Nat.add(stride,padding),total,total_below,Order.transitive(Nat.add(stride,padding),Nat.mul(2n,padding),total,Bounds.strict_le(Nat.add(stride,padding),Nat.mul(2n,padding),doubled),Samples.le_after(height,Nat.mul(2n,padding)))))

def row_in_last_band(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,output_first: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,ConvSpec.axis_size(height,kernel,stride,padding)) == True{} : Bool},
  +reads: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(ConvSpec.axis_size(height,kernel,stride,padding),1n),stride),kernel),padding) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(output_first,stride),height) == True{} : Bool}:
  match output_first:
    case 0n: height_positive
    case 1n+ +previous:
      Empty.absurd({Nat.is_lt(Nat.mul(1n+previous,stride),height) == True{} : Bool},
        last_band_first(height,kernel,stride,padding,previous,stride_positive,kernel_positive,kernel_fits,ordered,reads))

# The stride position of the first output of a leaf whose outputs read only
# padding lies before its core end.
def padding_row(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+core_end: Nat,+output_first: Nat,+output_end: Nat,
  +stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},+kernel_positive: {Nat.is_lt(0n,kernel) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},
  +kernel_fits: {Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +reads: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding) == True{} : Bool},
  +ending: Ending<core_end,height,stride,output_end,ConvSpec.axis_size(height,kernel,stride,padding)>) ->
  {Nat.is_lt(Nat.mul(output_first,stride),core_end) == True{} : Bool}:
  match ending:
    case Last{+whole,+full}:
      %Equal.sym(Nat,core_end,height,whole) : {Nat.is_lt(Nat.mul(output_first,stride),_) == True{} : Bool}
      row_in_last_band(height,kernel,stride,padding,output_first,stride_positive,kernel_positive,height_positive,kernel_fits,
        %full : {Nat.is_lt(output_first,_) == True{} : Bool} ordered,
        %full : {Nat.is_le(Nat.add(Nat.mul(Nat.sub(_,1n),stride),kernel),padding) == True{} : Bool} reads)
    case Inner{+ceiling}: row_before_boundary(core_end,stride,output_first,output_end,stride_positive,ordered,ceiling)

# The one row at the first output's stride position covers a leaf whose
# outputs read only the padding above the image.
def padding_cover(+problem: ConvSpec.Problem,+output_first: Nat,+output_end: Nat,+ordered: {Nat.is_lt(output_first,output_end) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(Nat.mul(output_first,Guards.stride(problem)),1n),Guards.height(problem)) == True{} : Bool},
  +reads: {Nat.is_le(Nat.add(Nat.mul(Nat.sub(output_end,1n),Guards.stride(problem)),Guards.kernel(problem)),Guards.pad(problem)) == True{} : Bool}) ->
  Positions.Cover<problem,Nat.mul(output_first,Guards.stride(problem)),1n,Nat.mul(output_first,ConvSpec.out_width(problem)),
    Nat.mul(Nat.sub(output_end,output_first),ConvSpec.out_width(problem))>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +output_width = ConvSpec.axis_size(width,kernel,stride,padding)
  +width_ok = {{==} : {Nat.is_lt(0n,output_width) == True{} : Bool}}
  +first = Nat.mul(output_first,stride)
  +offset = Nat.mul(output_first,output_width)
  +positions = Nat.mul(Nat.sub(output_end,output_first),output_width)
  +last = Nat.div(Nat.sub(Nat.add(offset,positions),1n),output_width)
  +last_reads = Order.transitive(Nat.add(Nat.mul(last,stride),kernel),Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding,
    Order.add_le(Nat.mul(last,stride),Nat.mul(Nat.sub(output_end,1n),stride),kernel,
      RowMajor.mul_le(last,Nat.sub(output_end,1n),stride,WindowContents.last_output(output_first,output_end,output_width,width_ok,ordered))),reads)
  +before = {%Equal.sym(Nat,Nat.div(offset,output_width),output_first,Division.exact_quotient(output_first,output_width,width_ok)) :
      {Nat.is_le(first,Nat.mul(_,stride)) == True{} : Bool}
    Order.reflexive(first) : {Nat.is_le(first,Nat.mul(Nat.div(offset,output_width),stride)) == True{} : Bool}}
  +kernel_fits = Order.transitive(kernel,padding,Nat.add(1n,Nat.mul(2n,padding)),
    Order.transitive(kernel,Nat.add(Nat.mul(Nat.sub(output_end,1n),stride),kernel),padding,Samples.le_after(Nat.mul(Nat.sub(output_end,1n),stride),kernel),reads),
    Order.transitive(padding,Nat.mul(2n,padding),Nat.add(1n,Nat.mul(2n,padding)),WindowContents.padding_doubled(padding),Samples.le_after(1n,Nat.mul(2n,padding))))
  +bottom = Order.transitive(Nat.add(Nat.mul(last,stride),kernel),padding,Nat.add(Nat.add(first,1n),padding),last_reads,Samples.le_after(Nat.add(first,1n),padding))
  Positions.Cover{output_first,{==},fits,{==},kernel_fits,Positions.Over{before,last_reads},Inr{bottom}}
