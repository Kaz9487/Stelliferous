# The band a call of the windowed convolution reads for positions
# [offset, offset + span) of one image covers every read of those positions.
# The pipeline computes the band without the coverage check, so the facts
# come from how the band is computed.
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_extrema.bend as Extrema
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./window_start.bend as Start
import ./window_contents.bend as Contents
import ./window_covers.bend as Covers
import ./window_positions.bend as Positions

# The rows of the band, as natural numbers: the first row the first output
# row reads rounded down to the stride, kept at or below the last
# stride-aligned row of the image, and the end the last output row reads,
# at least one row past the first and at most the image end.
def row_start(+stride: Nat,+padding: Nat,+row: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_le(padding,Nat.mul(row,stride)),Nat.mul(Nat.div(Nat.sub(Nat.mul(row,stride),padding),stride),stride),0n)

def last_start(+height: Nat,+stride: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.sub(height,1n),stride),stride)

def band_first(+height: Nat,+stride: Nat,+padding: Nat,+row: Nat) -> Nat:
  Nat.min(row_start(stride,padding,row),last_start(height,stride))

def reach(+kernel: Nat,+stride: Nat,+row: Nat) -> Nat:
  Nat.add(Nat.mul(row,stride),kernel)

def row_end(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+row: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_le(reach(kernel,stride,row),padding),0n,Nat.min(height,Nat.sub(reach(kernel,stride,row),padding)))

def band_end(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first_row: Nat,+last_row: Nat) -> Nat:
  Nat.min(height,Nat.max(row_end(height,kernel,stride,padding,last_row),Nat.add(band_first(height,stride,padding,first_row),1n)))

def Multiple(+value: Nat,+stride: Nat) -> Data:
  {value == Nat.mul(Nat.div(value,stride),stride) : Nat}

def multiple(+quotient: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) -> Multiple(Nat.mul(quotient,stride),stride):
  %Equal.sym(Nat,Nat.div(Nat.mul(quotient,stride),stride),quotient,Division.exact_quotient(quotient,stride,positive)) :
    {Nat.mul(quotient,stride) == Nat.mul(_,stride) : Nat}
  {==}

def picked(-Property: Nat -> Data,choice: Bool,+left: Nat,+right: Nat,left_holds: Property(left),right_holds: Property(right)) ->
  Property(Bool.pick(Nat,choice,left,right)):
  match choice:
    case True{}: left_holds
    case False{}: right_holds

def minimum_holds(-Property: Nat -> Data,+left: Nat,+right: Nat,left_holds: Property(left),right_holds: Property(right)) -> Property(Nat.min(left,right)):
  %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),NatExtrema.minimum_choice(left,right)) : Property(_)
  picked(Property,Nat.is_lt(left,right),left,right,left_holds,right_holds)

# The first row is a multiple of the stride.
def first_multiple(+height: Nat,+stride: Nat,+padding: Nat,+row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  Multiple(band_first(height,stride,padding,row),stride):
  minimum_holds({value => Multiple(value,stride) : Nat -> Data},row_start(stride,padding,row),last_start(height,stride),
    picked({value => Multiple(value,stride) : Nat -> Data},Nat.is_le(padding,Nat.mul(row,stride)),
      Nat.mul(Nat.div(Nat.sub(Nat.mul(row,stride),padding),stride),stride),0n,
      multiple(Nat.div(Nat.sub(Nat.mul(row,stride),padding),stride),stride,positive),multiple(0n,stride,positive)),
    multiple(Nat.div(Nat.sub(height,1n),stride),stride,positive))

# The rows above the first row hold no read of the first output row.
def start_top(reached: Bool,+stride: Nat,+padding: Nat,+row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +guard: {Nat.is_le(padding,Nat.mul(row,stride)) == reached : Bool}) ->
  Either<&2,&2,{Bool.pick(Nat,reached,Nat.mul(Nat.div(Nat.sub(Nat.mul(row,stride),padding),stride),stride),0n) == 0n : Nat},
    {Nat.is_le(Nat.add(Bool.pick(Nat,reached,Nat.mul(Nat.div(Nat.sub(Nat.mul(row,stride),padding),stride),stride),0n),padding),Nat.mul(row,stride)) == True{} : Bool}>:
  match reached:
    case False{}: Inl{{==}}
    case True{}:
      +top = Nat.mul(row,stride)
      +rest = Nat.sub(top,padding)
      Inr{%Order.restore_shift(padding,top,guard) : {Nat.is_le(Nat.add(Nat.mul(Nat.div(rest,stride),stride),padding),_) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(padding,rest),Nat.add(rest,padding),Order.commutative(padding,rest)) : {Nat.is_le(Nat.add(Nat.mul(Nat.div(rest,stride),stride),padding),_) == True{} : Bool}
        Order.add_le(Nat.mul(Nat.div(rest,stride),stride),rest,padding,DivisionOrder.product_below(rest,stride,positive))}

def StartTop(+stride: Nat,+padding: Nat,+row: Nat,+value: Nat) -> Data:
  Either<&2,&2,{value == 0n : Nat},{Nat.is_le(Nat.add(value,padding),Nat.mul(row,stride)) == True{} : Bool}>

def top_below(+stride: Nat,+padding: Nat,+row: Nat,+start: Nat,+first: Nat,+below: {Nat.is_le(first,start) == True{} : Bool},
  +side: StartTop(stride,padding,row,start)) -> StartTop(stride,padding,row,first):
  match side:
    case Inl{+zero}:
      Inl{Order.zero_below(first,%zero : {Nat.is_le(first,_) == True{} : Bool} below)}
    case Inr{+above}:
      Inr{Order.transitive(Nat.add(first,padding),Nat.add(start,padding),Nat.mul(row,stride),Order.add_le(first,start,padding,below),above)}

def first_top(+height: Nat,+stride: Nat,+padding: Nat,+row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  StartTop(stride,padding,row,band_first(height,stride,padding,row)):
  top_below(stride,padding,row,row_start(stride,padding,row),band_first(height,stride,padding,row),
    NatExtrema.minimum_left(row_start(stride,padding,row),last_start(height,stride)),
    start_top(Nat.is_le(padding,Nat.mul(row,stride)),stride,padding,row,positive,{==}))

def top_before(+stride: Nat,+padding: Nat,+row: Nat,+first: Nat,+side: StartTop(stride,padding,row,first)) ->
  {Nat.is_le(first,Nat.mul(row,stride)) == True{} : Bool}:
  match side:
    case Inl{+zero}: %Equal.sym(Nat,first,0n,zero) : {Nat.is_le(_,Nat.mul(row,stride)) == True{} : Bool} Order.zero_le(Nat.mul(row,stride))
    case Inr{+above}: Order.transitive(first,Nat.add(first,padding),Nat.mul(row,stride),Order.le_add(first,padding),above)

def first_before(+height: Nat,+stride: Nat,+padding: Nat,+row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(band_first(height,stride,padding,row),Nat.mul(row,stride)) == True{} : Bool}:
  top_before(stride,padding,row,band_first(height,stride,padding,row),first_top(height,stride,padding,row,positive))

# The first row lies inside the image.
def first_inside(+height: Nat,+stride: Nat,+padding: Nat,+row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(band_first(height,stride,padding,row),1n),height) == True{} : Bool}:
  +last = Nat.sub(height,1n)
  +first = band_first(height,stride,padding,row)
  +below = Order.transitive(first,last_start(height,stride),last,NatExtrema.minimum_right(row_start(stride,padding,row),last_start(height,stride)),
    DivisionOrder.product_below(last,stride,positive))
  %Order.restore_shift(1n,height,Order.positive_one(height,height_positive)) : {Nat.is_le(Nat.add(first,1n),_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(1n,last),Nat.add(last,1n),Order.commutative(1n,last)) : {Nat.is_le(Nat.add(first,1n),_) == True{} : Bool}
  Order.add_le(first,last,1n,below)

# The end of the band lies past the first row and inside the image.
def end_after(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first_row: Nat,+last_row: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(band_first(height,stride,padding,first_row),1n),band_end(height,kernel,stride,padding,first_row,last_row)) == True{} : Bool}:
  +next = Nat.add(band_first(height,stride,padding,first_row),1n)
  NatExtrema.minimum_lower(height,Nat.max(row_end(height,kernel,stride,padding,last_row),next),next,first_inside(height,stride,padding,first_row,positive,height_positive),
    NatExtrema.maximum_right(row_end(height,kernel,stride,padding,last_row),next))

def end_inside(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first_row: Nat,+last_row: Nat) ->
  {Nat.is_le(band_end(height,kernel,stride,padding,first_row,last_row),height) == True{} : Bool}:
  NatExtrema.minimum_left(height,Nat.max(row_end(height,kernel,stride,padding,last_row),Nat.add(band_first(height,stride,padding,first_row),1n)))

# The band ends at the image end or after every row the last output row reads.
def EndBottom(+height: Nat,+padding: Nat,+reached: Nat,+value: Nat) -> Data:
  Either<&2,&2,{height == value : Nat},{Nat.is_le(reached,Nat.add(value,padding)) == True{} : Bool}>

def clip_bottom(choice: Bool,+height: Nat,+padding: Nat,+reached: Nat,+next: Nat,
  +guard: {Nat.is_lt(height,Nat.sub(reached,padding)) == choice : Bool},+beyond: {Nat.is_le(padding,reached) == True{} : Bool},
  +bounded: {Nat.is_le(Nat.max(Bool.pick(Nat,choice,height,Nat.sub(reached,padding)),next),height) == True{} : Bool}) ->
  EndBottom(height,padding,reached,Nat.max(Bool.pick(Nat,choice,height,Nat.sub(reached,padding)),next)):
  match choice:
    case True{}:
      Inl{Order.antisymmetric(height,Nat.max(height,next),NatExtrema.maximum_left(height,next),bounded)}
    case False{}:
      +rest = Nat.sub(reached,padding)
      Inr{%Order.restore_shift(padding,reached,beyond) : {Nat.is_le(_,Nat.add(Nat.max(rest,next),padding)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(padding,rest),Nat.add(rest,padding),Order.commutative(padding,rest)) : {Nat.is_le(_,Nat.add(Nat.max(rest,next),padding)) == True{} : Bool}
        Order.add_le(rest,Nat.max(rest,next),padding,NatExtrema.maximum_left(rest,next))}

def last_bottom(choice: Bool,+height: Nat,+padding: Nat,+reached: Nat,+next: Nat,
  +guard: {Nat.is_le(reached,padding) == choice : Bool},
  +bounded: {Nat.is_le(Nat.max(Bool.pick(Nat,choice,0n,Nat.min(height,Nat.sub(reached,padding))),next),height) == True{} : Bool}) ->
  EndBottom(height,padding,reached,Nat.max(Bool.pick(Nat,choice,0n,Nat.min(height,Nat.sub(reached,padding))),next)):
  match choice:
    case True{}:
      Inr{Order.transitive(reached,padding,Nat.add(Nat.max(0n,next),padding),guard,Order.right_le(Nat.max(0n,next),padding))}
    case False{}:
      +rest = Nat.sub(reached,padding)
      +beyond = {%Equal.sym(Bool,Nat.is_lt(padding,reached),Bool.not(Nat.is_le(reached,padding)),Order.strict_complement(padding,reached)) : {_ == True{} : Bool}
        %Equal.sym(Bool,Nat.is_le(reached,padding),False{},guard) : {Bool.not(_) == True{} : Bool}
        {==} : {Nat.is_lt(padding,reached) == True{} : Bool}}
      %Equal.sym(Nat,Nat.min(height,rest),Bool.pick(Nat,Nat.is_lt(height,rest),height,rest),NatExtrema.minimum_choice(height,rest)) :
        EndBottom(height,padding,reached,Nat.max(_,next))
      clip_bottom(Nat.is_lt(height,rest),height,padding,reached,next,{==},Bounds.strict_le(padding,reached,beyond),
        %NatExtrema.minimum_choice(height,rest) : {Nat.is_le(Nat.max(_,next),height) == True{} : Bool} bounded)

def end_bottom(choice: Bool,+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+row: Nat,+next: Nat,
  +guard: {Nat.is_lt(height,Nat.max(row_end(height,kernel,stride,padding,row),next)) == choice : Bool}) ->
  EndBottom(height,padding,reach(kernel,stride,row),Bool.pick(Nat,choice,height,Nat.max(row_end(height,kernel,stride,padding,row),next))):
  match choice:
    case True{}: Inl{{==}}
    case False{}:
      last_bottom(Nat.is_le(reach(kernel,stride,row),padding),height,padding,reach(kernel,stride,row),next,{==},
        Order.not_less(height,Nat.max(row_end(height,kernel,stride,padding,row),next),guard))

def band_bottom(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first_row: Nat,+last_row: Nat) ->
  EndBottom(height,padding,reach(kernel,stride,last_row),band_end(height,kernel,stride,padding,first_row,last_row)):
  +most = Nat.max(row_end(height,kernel,stride,padding,last_row),Nat.add(band_first(height,stride,padding,first_row),1n))
  %Equal.sym(Nat,Nat.min(height,most),Bool.pick(Nat,Nat.is_lt(height,most),height,most),NatExtrema.minimum_choice(height,most)) :
    EndBottom(height,padding,reach(kernel,stride,last_row),_)
  end_bottom(Nat.is_lt(height,most),height,kernel,stride,padding,last_row,Nat.add(band_first(height,stride,padding,first_row),1n),{==})

# The reads of the last output row, from the first row on, end inside the
# band with its padding on both sides.
def bottom_room(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: Nat,+last: Nat,+row: Nat,
  +front: {Nat.is_le(Nat.add(first,kernel),reach(kernel,stride,row)) == True{} : Bool},
  +reads: {Nat.is_le(reach(kernel,stride,row),Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +bottom: EndBottom(height,padding,reach(kernel,stride,row),last)) ->
  {Nat.is_le(Nat.add(first,kernel),Nat.add(last,Nat.mul(2n,padding))) == True{} : Bool}:
  match bottom:
    case Inl{+same}:
      %same : {Nat.is_le(Nat.add(first,kernel),Nat.add(_,Nat.mul(2n,padding))) == True{} : Bool}
      Order.transitive(Nat.add(first,kernel),reach(kernel,stride,row),Nat.add(height,Nat.mul(2n,padding)),front,reads)
    case Inr{+after}:
      Order.transitive(Nat.add(first,kernel),Nat.add(last,padding),Nat.add(last,Nat.mul(2n,padding)),
        Order.transitive(Nat.add(first,kernel),reach(kernel,stride,row),Nat.add(last,padding),front,after),
        Order.offset_le(last,padding,Nat.mul(2n,padding),Order.le_add(padding,Nat.mul(1n,padding))))

# The kernel fits the band with its padding: the first row lies at or above
# the reads of the last output row, which end inside the padded image.
def band_kernel(+height: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+first: Nat,+last: Nat,+row: Nat,
  +before: {Nat.is_le(first,Nat.mul(row,stride)) == True{} : Bool},+ordered: {Nat.is_le(first,last) == True{} : Bool},
  +reads: {Nat.is_le(reach(kernel,stride,row),Nat.add(height,Nat.mul(2n,padding))) == True{} : Bool},
  +bottom: EndBottom(height,padding,reach(kernel,stride,row),last)) ->
  {Nat.is_le(kernel,Nat.add(Nat.sub(last,first),Nat.mul(2n,padding))) == True{} : Bool}:
  +doubled = Nat.mul(2n,padding)
  Order.cancel_offset_le(first,kernel,Nat.add(Nat.sub(last,first),doubled),
    %Addition.associative(first,Nat.sub(last,first),doubled) : {Nat.is_le(Nat.add(first,kernel),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,Nat.sub(last,first)),last,Order.restore_shift(first,last,ordered)) :
      {Nat.is_le(Nat.add(first,kernel),Nat.add(_,doubled)) == True{} : Bool}
    bottom_room(height,kernel,stride,padding,first,last,row,Order.add_le(first,Nat.mul(row,stride),kernel,before),reads,bottom))

# The band of positions [offset, offset + span) of an image of a problem.
def first_row(+problem: ConvSpec.Problem,+offset: Nat) -> Nat:
  Nat.div(offset,ConvSpec.out_width(problem))

def last_row(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat) -> Nat:
  Nat.div(Nat.sub(Nat.add(offset,span),1n),ConvSpec.out_width(problem))

def nat_first(+problem: ConvSpec.Problem,+offset: Nat) -> Nat:
  band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,offset))

def nat_end(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat) -> Nat:
  band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,offset),last_row(problem,offset,span))

def nat_cover(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat,
  +positive: {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,Guards.height(problem)) == True{} : Bool},
  +ordered: {Nat.is_le(first_row(problem,offset),last_row(problem,offset,span)) == True{} : Bool},
  +reads: {Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,offset,span)),Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}) ->
  Positions.Cover<problem,nat_first(problem,offset),Nat.sub(nat_end(problem,offset,span),nat_first(problem,offset)),offset,span>:
  +height = Guards.height(problem)
  +kernel = Guards.kernel(problem)
  +stride = Guards.stride(problem)
  +padding = Guards.pad(problem)
  +top_row = first_row(problem,offset)
  +bottom_row = last_row(problem,offset,span)
  +first = nat_first(problem,offset)
  +last = nat_end(problem,offset,span)
  +after = end_after(height,kernel,stride,padding,top_row,bottom_row,positive,height_positive)
  +less = Bounds.predecessor_strict(first,last,%Order.commutative(first,1n) : {Nat.is_le(_,last) == True{} : Bool} after)
  +first_le = Bounds.strict_le(first,last,less)
  +joined = Equal.sym(Nat,Nat.add(first,Nat.sub(last,first)),last,Order.restore_shift(first,last,first_le))
  +before = Order.transitive(first,Nat.mul(top_row,stride),Nat.mul(bottom_row,stride),first_before(height,stride,padding,top_row,positive),
    RowMajor.mul_le(top_row,bottom_row,stride,ordered))
  +bottom = band_bottom(height,kernel,stride,padding,top_row,bottom_row)
  Positions.Cover{Nat.div(first,stride),first_multiple(height,stride,padding,top_row,positive),
    %joined : {Nat.is_le(_,height) == True{} : Bool} end_inside(height,kernel,stride,padding,top_row,bottom_row),
    Order.positive_difference(last,first,less),
    band_kernel(height,kernel,stride,padding,first,last,bottom_row,before,first_le,reads,bottom),
    Positions.top_of(problem,first,offset,span,first_top(height,stride,padding,top_row,positive)),
    %joined : EndBottom(height,padding,reach(kernel,stride,bottom_row),_) bottom}

# The band of a call as the pipeline computes it.
def height_word(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.height(problem))

def kernel_word(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.kernel(problem))

def stride_word(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.stride(problem))

def padding_word(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.pad(problem))

def width_word(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(ConvSpec.out_width(problem))

def first_row_word(+problem: ConvSpec.Problem,+offset: U32) -> U32:
  (offset / width_word(problem) : U32)

def last_row_word(+problem: ConvSpec.Problem,+offset: U32,+span: U32) -> U32:
  ((offset + span - 1) / width_word(problem) : U32)

def last_start_word(+problem: ConvSpec.Problem) -> U32:
  (((0 + height_word(problem) : U32) - 1 : U32) / stride_word(problem) * stride_word(problem) : U32)

def first_word(+problem: ConvSpec.Problem,+offset: U32) -> U32:
  U32.max(0,U32.min(KernelWindow.first_row(stride_word(problem),padding_word(problem),first_row_word(problem,offset)),last_start_word(problem)))

def end_word(+problem: ConvSpec.Problem,+offset: U32,+span: U32) -> U32:
  U32.min((0 + height_word(problem) : U32),U32.max(KernelWindow.end_row(height_word(problem),kernel_word(problem),stride_word(problem),padding_word(problem),last_row_word(problem,offset,span)),(first_word(problem,offset) + 1 : U32)))

def rows_word(+problem: ConvSpec.Problem,+offset: U32,+span: U32) -> U32:
  (end_word(problem,offset,span) - first_word(problem,offset) : U32)

def padded_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Covers.padded_rows_bound(limits,problem,valid,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

def height_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {Nat.is_le(Guards.height(problem),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Guards.height(problem),Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),U32.to_nat(Machine.coordinate(limits)),Order.le_add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),padded_room(limits,problem,valid,supported))

def stride_double(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {Nat.is_le(Nat.double(Guards.stride(problem)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  %Start.stride_word(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  Start.stride_divisor(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

def max_zero(+value: Nat) -> {Nat.max(0n,value) == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

# An output row of the problem reads rows inside the padded image.
def row_reach(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+row: Nat,+inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool}) ->
  {Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),row),Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +whole = Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem)))
  +rest = Nat.sub(whole,Guards.kernel(problem))
  +stride_positive = Contents.valid_condition(problem,valid,5n)
  +kernel_fits = Contents.valid_condition(problem,valid,4n)
  +below = Division.lt_succ_le(row,Nat.div(rest,Guards.stride(problem)),inside)
  %Order.restore_shift(Guards.kernel(problem),whole,kernel_fits) : {Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),row),_) == True{} : Bool}
  %Order.commutative(rest,Guards.kernel(problem)) : {Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),row),_) == True{} : Bool}
  Order.add_le(Nat.mul(row,Guards.stride(problem)),rest,Guards.kernel(problem),Order.transitive(Nat.mul(row,Guards.stride(problem)),Nat.mul(Nat.div(rest,Guards.stride(problem)),Guards.stride(problem)),rest,RowMajor.mul_le(row,Nat.div(rest,Guards.stride(problem)),Guards.stride(problem),below),
    DivisionOrder.product_below(rest,Guards.stride(problem),stride_positive)))

# The words of the band equal their natural-number values.
def last_start_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(last_start_word(problem)) == last_start(Guards.height(problem),Guards.stride(problem)) : Nat}:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,support)
  +stride_exact = Start.stride_word(limits,problem,support)
  +stride_positive = Contents.valid_condition(problem,valid,5n)
  +room = height_room(limits,problem,valid,supported)
  +end_exact = Equal.trans(Nat,U32.to_nat((0 + height_word(problem) : U32)),U32.to_nat(height_word(problem)),Guards.height(problem),Equal.cong(U32,Nat,word => U32.to_nat(word),(0 + height_word(problem) : U32),height_word(problem),Arithmetic.add_zero_left(height_word(problem))),height_exact)
  +lower = Nat.sub(Guards.height(problem),1n)
  +lower_exact = Arithmetic.difference_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(0 + height_word(problem) : U32),1,Guards.height(problem),1n,end_exact,{==},Order.positive_one(Guards.height(problem),Contents.valid_condition(problem,valid,8n)),room)
  +quotient_exact = U32Division.quotient_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),((0 + height_word(problem) : U32) - 1 : U32),stride_word(problem),lower,Guards.stride(problem),lower_exact,stride_exact,stride_double(limits,problem,valid,supported),stride_positive)
  Arithmetic.product_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(((0 + height_word(problem) : U32) - 1 : U32) / stride_word(problem) : U32),stride_word(problem),Nat.div(lower,Guards.stride(problem)),Guards.stride(problem),quotient_exact,stride_exact,
    Order.transitive(Nat.mul(Nat.div(lower,Guards.stride(problem)),Guards.stride(problem)),lower,U32.to_nat(Machine.coordinate(limits)),DivisionOrder.product_below(lower,Guards.stride(problem),stride_positive),
      Order.transitive(lower,Guards.height(problem),U32.to_nat(Machine.coordinate(limits)),Order.sub_le(Guards.height(problem),1n),room)))

def first_row_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+inside: {Nat.is_le(U32.to_nat(offset),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  {U32.to_nat(first_row_word(problem,offset)) == first_row(problem,U32.to_nat(offset)) : Nat}:
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  U32Division.quotient_exact(Machine.half_word(limits),Limits.half_word_fits(limits),offset,width_word(problem),U32.to_nat(offset),ConvSpec.out_width(problem),{==},Coordinates.width_exact(limits,problem,facts,support),
    Limits.double_bound(limits,ConvSpec.out_width(problem),Coordinates.width_bound(limits,problem,facts,support)),Coordinates.width_positive(problem))

# The first row of an output row, rounded down to the stride.
def row_start_word(choice: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+row: U32,+row_value: Nat,
  +product_exact: {U32.to_nat((row * stride_word(problem) : U32)) == Nat.mul(row_value,Guards.stride(problem)) : Nat},
  +product_room: {Nat.is_le(Nat.mul(row_value,Guards.stride(problem)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
  +guard: {Nat.is_le(Guards.pad(problem),Nat.mul(row_value,Guards.stride(problem))) == choice : Bool}) ->
  {U32.to_nat(Bool.pick(U32,choice,(((row * stride_word(problem) : U32) - padding_word(problem) : U32) / stride_word(problem) * stride_word(problem) : U32),0)) ==
    Bool.pick(Nat,choice,Nat.mul(Nat.div(Nat.sub(Nat.mul(row_value,Guards.stride(problem)),Guards.pad(problem)),Guards.stride(problem)),Guards.stride(problem)),0n) : Nat}:
  match choice:
    case False{}: {==}
    case True{}:
      +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
      +stride_exact = Start.stride_word(limits,problem,support)
      +stride_positive = Contents.valid_condition(problem,valid,5n)
      +top = Nat.mul(row_value,Guards.stride(problem))
      +rest = Nat.sub(top,Guards.pad(problem))
      +rest_room = Order.transitive(rest,top,U32.to_nat(Machine.coordinate(limits)),Order.sub_le(top,Guards.pad(problem)),product_room)
      +rest_exact = Arithmetic.difference_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(row * stride_word(problem) : U32),padding_word(problem),top,Guards.pad(problem),product_exact,SupportedCoordinateBounds.pad_exact(limits,problem,support),guard,product_room)
      +quotient_exact = U32Division.quotient_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),((row * stride_word(problem) : U32) - padding_word(problem) : U32),stride_word(problem),rest,Guards.stride(problem),rest_exact,stride_exact,stride_double(limits,problem,valid,supported),stride_positive)
      Arithmetic.product_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(((row * stride_word(problem) : U32) - padding_word(problem) : U32) / stride_word(problem) : U32),stride_word(problem),Nat.div(rest,Guards.stride(problem)),Guards.stride(problem),quotient_exact,stride_exact,
        Order.transitive(Nat.mul(Nat.div(rest,Guards.stride(problem)),Guards.stride(problem)),rest,U32.to_nat(Machine.coordinate(limits)),DivisionOrder.product_below(rest,Guards.stride(problem),stride_positive),rest_room))

def le_exact(+left: U32,+right: U32,+left_value: Nat,+right_value: Nat,+left_exact: {U32.to_nat(left) == left_value : Nat},
  +right_exact: {U32.to_nat(right) == right_value : Nat}) -> {U32.is_le(left,right) == Nat.is_le(left_value,right_value) : Bool}:
  %left_exact : {U32.is_le(left,right) == Nat.is_le(_,right_value) : Bool}
  %right_exact : {U32.is_le(left,right) == Nat.is_le(U32.to_nat(left),_) : Bool}
  Comparison.le(left,right)

def row_room(+problem: ConvSpec.Problem,+limits: Machine.Limits,+support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +row: Nat,+inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool}) -> {Nat.is_le(Nat.mul(row,Guards.stride(problem)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(row,Guards.stride(problem)),Nat.mul(ConvSpec.out_height(problem),Guards.stride(problem)),U32.to_nat(Machine.coordinate(limits)),RowMajor.mul_le(row,ConvSpec.out_height(problem),Guards.stride(problem),Bounds.strict_le(row,ConvSpec.out_height(problem),inside)),
    Order.transitive(Nat.mul(ConvSpec.out_height(problem),Guards.stride(problem)),Nat.add(Nat.mul(ConvSpec.out_height(problem),Guards.stride(problem)),Guards.kernel(problem)),U32.to_nat(Machine.coordinate(limits)),Order.le_add(Nat.mul(ConvSpec.out_height(problem),Guards.stride(problem)),Guards.kernel(problem)),SupportedCoordinateBounds.row_envelope(limits,problem,support)))

def row_start_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+row: U32,+row_value: Nat,+row_exact: {U32.to_nat(row) == row_value : Nat},+inside: {Nat.is_lt(row_value,ConvSpec.out_height(problem)) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.first_row(stride_word(problem),padding_word(problem),row)) == row_start(Guards.stride(problem),Guards.pad(problem),row_value) : Nat}:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +room = row_room(problem,limits,support,row_value,inside)
  +product_exact = Arithmetic.product_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),row,stride_word(problem),row_value,Guards.stride(problem),row_exact,Start.stride_word(limits,problem,support),room)
  %Equal.sym(Bool,U32.is_le(padding_word(problem),(row * stride_word(problem) : U32)),Nat.is_le(Guards.pad(problem),Nat.mul(row_value,Guards.stride(problem))),
    le_exact(padding_word(problem),(row * stride_word(problem) : U32),Guards.pad(problem),Nat.mul(row_value,Guards.stride(problem)),SupportedCoordinateBounds.pad_exact(limits,problem,support),product_exact)) :
    {U32.to_nat(Bool.pick(U32,_,(((row * stride_word(problem) : U32) - padding_word(problem) : U32) / stride_word(problem) * stride_word(problem) : U32),0)) == row_start(Guards.stride(problem),Guards.pad(problem),row_value) : Nat}
  row_start_word(Nat.is_le(Guards.pad(problem),Nat.mul(row_value,Guards.stride(problem))),limits,problem,valid,supported,row,row_value,product_exact,room,{==})

def first_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+inside: {Nat.is_lt(U32.to_nat(offset),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  {U32.to_nat(first_word(problem,offset)) == band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))) : Nat}:
  +facts = Guards.valid(problem,valid)
  +row = first_row_word(problem,offset)
  +row_exact = first_row_exact(limits,problem,valid,supported,offset,Bounds.strict_le(U32.to_nat(offset),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),inside))
  +row_inside = DivisionOrder.quotient_below(U32.to_nat(offset),ConvSpec.out_height(problem),ConvSpec.out_width(problem),Coordinates.width_positive(problem),inside)
  +start = KernelWindow.first_row(stride_word(problem),padding_word(problem),row)
  +lower = U32.min(start,last_start_word(problem))
  +lower_exact = Equal.trans(Nat,U32.to_nat(lower),Nat.min(U32.to_nat(start),U32.to_nat(last_start_word(problem))),band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),Extrema.minimum(start,last_start_word(problem)),
    %Equal.sym(Nat,U32.to_nat(start),row_start(Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),row_start_exact(limits,problem,valid,supported,row,first_row(problem,U32.to_nat(offset)),row_exact,row_inside)) :
      {Nat.min(_,U32.to_nat(last_start_word(problem))) == band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))) : Nat}
    %Equal.sym(Nat,U32.to_nat(last_start_word(problem)),last_start(Guards.height(problem),Guards.stride(problem)),last_start_exact(limits,problem,valid,supported)) : {Nat.min(row_start(Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),_) == band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))) : Nat}
    {==})
  %lower_exact : {U32.to_nat(U32.max(0,lower)) == _ : Nat}
  %max_zero(U32.to_nat(lower)) : {U32.to_nat(U32.max(0,lower)) == _ : Nat}
  Extrema.maximum(0,lower)

def parent_end_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat((0 + height_word(problem) : U32)) == Guards.height(problem) : Nat}:
  Equal.trans(Nat,U32.to_nat((0 + height_word(problem) : U32)),U32.to_nat(height_word(problem)),Guards.height(problem),Equal.cong(U32,Nat,word => U32.to_nat(word),(0 + height_word(problem) : U32),height_word(problem),Arithmetic.add_zero_left(height_word(problem))),
    SupportedCoordinateBounds.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))

def not_le_above(+low: Nat,+high: Nat,+guard: {Nat.is_le(high,low) == False{} : Bool}) -> {Nat.is_le(low,high) == True{} : Bool}:
  Bounds.strict_le(low,high,
    %Equal.sym(Bool,Nat.is_lt(low,high),Bool.not(Nat.is_le(high,low)),Order.strict_complement(low,high)) : {_ == True{} : Bool}
    %Equal.sym(Bool,Nat.is_le(high,low),False{},guard) : {Bool.not(_) == True{} : Bool}
    {==})

# The positions of a call lie in one image: they start inside it, the last
# one is on the first one's row or after it, and its row is an output row.
def offset_after(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {Nat.is_lt(offset,Nat.add(offset,span)) == True{} : Bool}:
  %Addition.zero_right(offset) : {Nat.is_lt(_,Nat.add(offset,span)) == True{} : Bool}
  Order.offset_lt(offset,0n,span,positive)

def offset_inside(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {Nat.is_lt(offset,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}:
  Bounds.nat_trans(offset,Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),offset_after(problem,offset,span,positive,grid),grid)

def rows_ordered(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {Nat.is_le(first_row(problem,offset),last_row(problem,offset,span)) == True{} : Bool}:
  DivisionOrder.quotient_le(offset,Nat.sub(Nat.add(offset,span),1n),ConvSpec.out_width(problem),Coordinates.width_positive(problem),
    Order.before_last(offset,Nat.add(offset,span),offset_after(problem,offset,span,positive,grid)))

def last_inside(+problem: ConvSpec.Problem,+offset: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {Nat.is_lt(last_row(problem,offset,span),ConvSpec.out_height(problem)) == True{} : Bool}:
  +end = Nat.add(offset,span)
  +end_positive = Bounds.nat_trans(0n,span,end,positive,Order.right_le(offset,span))
  DivisionOrder.quotient_below(Nat.sub(end,1n),ConvSpec.out_height(problem),ConvSpec.out_width(problem),Coordinates.width_positive(problem),
    Bounds.nat_trans(Nat.sub(end,1n),end,Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),Order.last_below(end,end_positive),grid))

def last_row_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {U32.to_nat(last_row_word(problem,offset,span)) == last_row(problem,U32.to_nat(offset),U32.to_nat(span)) : Nat}:
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +room = Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),U32.to_nat(Machine.capacity(limits)),grid,Padded.spatial_bound_facts(limits,problem,facts,support))
  +sum_exact = Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),offset,span,U32.to_nat(offset),U32.to_nat(span),{==},{==},room)
  +end_positive = Bounds.nat_trans(0n,U32.to_nat(span),Nat.add(U32.to_nat(offset),U32.to_nat(span)),positive,Order.right_le(U32.to_nat(offset),U32.to_nat(span)))
  +last_exact = Arithmetic.difference_exact(Machine.capacity(limits),Limits.capacity_fits(limits),(offset + span : U32),1,Nat.add(U32.to_nat(offset),U32.to_nat(span)),1n,sum_exact,{==},Order.positive_one(Nat.add(U32.to_nat(offset),U32.to_nat(span)),end_positive),room)
  U32Division.quotient_exact(Machine.half_word(limits),Limits.half_word_fits(limits),((offset + span : U32) - 1 : U32),width_word(problem),Nat.sub(Nat.add(U32.to_nat(offset),U32.to_nat(span)),1n),ConvSpec.out_width(problem),last_exact,Coordinates.width_exact(limits,problem,facts,support),
    Limits.double_bound(limits,ConvSpec.out_width(problem),Coordinates.width_bound(limits,problem,facts,support)),Coordinates.width_positive(problem))

def reach_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {U32.to_nat((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32)) == reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +inside = last_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid)
  +product_exact = Arithmetic.product_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),last_row_word(problem,offset,span),stride_word(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span)),Guards.stride(problem),last_row_exact(limits,problem,valid,supported,offset,span,positive,grid),Start.stride_word(limits,problem,support),
    row_room(problem,limits,support,last_row(problem,U32.to_nat(offset),U32.to_nat(span)),inside))
  Arithmetic.sum_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(last_row_word(problem,offset,span) * stride_word(problem) : U32),kernel_word(problem),Nat.mul(last_row(problem,U32.to_nat(offset),U32.to_nat(span)),Guards.stride(problem)),Guards.kernel(problem),product_exact,SupportedCoordinateBounds.kernel_exact(limits,problem,support),
    Order.transitive(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),U32.to_nat(Machine.coordinate(limits)),row_reach(problem,valid,last_row(problem,U32.to_nat(offset),U32.to_nat(span)),inside),padded_room(limits,problem,valid,supported)))

# The end the last output row reads, clipped to the image.
def row_end_word(choice: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},+guard: {Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem)) == choice : Bool}) ->
  {U32.to_nat(Bool.pick(U32,choice,0,U32.min(height_word(problem),((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32) - padding_word(problem) : U32)))) == Bool.pick(Nat,choice,0n,Nat.min(Guards.height(problem),Nat.sub(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem)))) : Nat}:
  match choice:
    case True{}: {==}
    case False{}:
      +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
      +inside = last_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid)
      +bounded = Order.transitive(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(Guards.height(problem),Nat.mul(2n,Guards.pad(problem))),U32.to_nat(Machine.coordinate(limits)),row_reach(problem,valid,last_row(problem,U32.to_nat(offset),U32.to_nat(span)),inside),padded_room(limits,problem,valid,supported))
      +rest_exact = Arithmetic.difference_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),(last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32),padding_word(problem),reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem),reach_exact(limits,problem,valid,supported,offset,span,positive,grid),SupportedCoordinateBounds.pad_exact(limits,problem,support),
        not_le_above(Guards.pad(problem),reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),guard),bounded)
      %SupportedCoordinateBounds.height_exact(limits,problem,support) : {U32.to_nat(U32.min(height_word(problem),((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32) - padding_word(problem) : U32))) == Nat.min(_,Nat.sub(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem))) : Nat}
      %rest_exact : {U32.to_nat(U32.min(height_word(problem),((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32) - padding_word(problem) : U32))) == Nat.min(U32.to_nat(height_word(problem)),_) : Nat}
      Extrema.minimum(height_word(problem),((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32) - padding_word(problem) : U32))

def row_end_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.end_row(height_word(problem),kernel_word(problem),stride_word(problem),padding_word(problem),last_row_word(problem,offset,span))) == row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  %Equal.sym(Bool,U32.is_le((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32),padding_word(problem)),Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem)),le_exact((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32),padding_word(problem),reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem),reach_exact(limits,problem,valid,supported,offset,span,positive,grid),SupportedCoordinateBounds.pad_exact(limits,problem,support))) :
    {U32.to_nat(Bool.pick(U32,_,0,U32.min(height_word(problem),((last_row_word(problem,offset,span) * stride_word(problem) + kernel_word(problem) : U32) - padding_word(problem) : U32)))) == row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}
  row_end_word(Nat.is_le(reach(Guards.kernel(problem),Guards.stride(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.pad(problem)),limits,problem,valid,supported,offset,span,positive,grid,{==})

def end_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {U32.to_nat(end_word(problem,offset,span)) == band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}:
  +facts = offset_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid)
  +first = first_word(problem,offset)
  +end_row = KernelWindow.end_row(height_word(problem),kernel_word(problem),stride_word(problem),padding_word(problem),last_row_word(problem,offset,span))
  +next = (first + 1 : U32)
  +first_exact = first_exact(limits,problem,valid,supported,offset,facts)
  +next_room = Order.transitive(Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n),Guards.height(problem),U32.to_nat(Machine.coordinate(limits)),
    first_inside(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),Contents.valid_condition(problem,valid,5n),Contents.valid_condition(problem,valid,8n)),height_room(limits,problem,valid,supported))
  +next_exact = Arithmetic.sum_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),first,1,band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n,first_exact,{==},next_room)
  +most = U32.max(end_row,next)
  +most_exact = Equal.trans(Nat,U32.to_nat(most),Nat.max(U32.to_nat(end_row),U32.to_nat(next)),Nat.max(row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n)),Extrema.maximum(end_row,next),
    %Equal.sym(Nat,U32.to_nat(end_row),row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),row_end_exact(limits,problem,valid,supported,offset,span,positive,grid)) : {Nat.max(_,U32.to_nat(next)) == Nat.max(row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n)) : Nat}
    %Equal.sym(Nat,U32.to_nat(next),Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n),next_exact) : {Nat.max(row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),_) == Nat.max(row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n)) : Nat}
    {==})
  Equal.trans(Nat,U32.to_nat(end_word(problem,offset,span)),Nat.min(U32.to_nat((0 + height_word(problem) : U32)),U32.to_nat(most)),band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Extrema.minimum((0 + height_word(problem) : U32),most),
    %Equal.sym(Nat,U32.to_nat((0 + height_word(problem) : U32)),Guards.height(problem),parent_end_exact(limits,problem,valid,supported)) : {Nat.min(_,U32.to_nat(most)) == band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}
    %Equal.sym(Nat,U32.to_nat(most),Nat.max(row_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Nat.add(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n)),most_exact) : {Nat.min(Guards.height(problem),_) == band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))) : Nat}
    {==})

def rows_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> {U32.to_nat(rows_word(problem,offset,span)) == Nat.sub(band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)))) : Nat}:
  +facts = offset_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid)
  +stride_positive = Contents.valid_condition(problem,valid,5n)
  +height_positive = Contents.valid_condition(problem,valid,8n)
  +after = end_after(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span)),stride_positive,height_positive)
  +ordered = Bounds.strict_le(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Bounds.predecessor_strict(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),%Order.commutative(band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),1n) : {Nat.is_le(_,band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span)))) == True{} : Bool} after))
  Arithmetic.difference_exact(Machine.coordinate(limits),Limits.coordinate_fits(limits),end_word(problem,offset,span),first_word(problem,offset),band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),end_exact(limits,problem,valid,supported,offset,span,positive,grid),first_exact(limits,problem,valid,supported,offset,facts),ordered,
    Order.transitive(band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),Guards.height(problem),U32.to_nat(Machine.coordinate(limits)),end_inside(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),height_room(limits,problem,valid,supported)))

# The band a call computes for positions [offset, offset + span) of one image
# covers every read of the positions.
def band_cover(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  Positions.Cover<problem,U32.to_nat(first_word(problem,offset)),U32.to_nat(rows_word(problem,offset,span)),U32.to_nat(offset),U32.to_nat(span)>:
  %Equal.sym(Nat,U32.to_nat(first_word(problem,offset)),band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),first_exact(limits,problem,valid,supported,offset,offset_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid))) :
    Positions.Cover<problem,_,U32.to_nat(rows_word(problem,offset,span)),U32.to_nat(offset),U32.to_nat(span)>
  %Equal.sym(Nat,U32.to_nat(rows_word(problem,offset,span)),Nat.sub(band_end(Guards.height(problem),Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)),last_row(problem,U32.to_nat(offset),U32.to_nat(span))),band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset)))),rows_exact(limits,problem,valid,supported,offset,span,positive,grid)) :
    Positions.Cover<problem,band_first(Guards.height(problem),Guards.stride(problem),Guards.pad(problem),first_row(problem,U32.to_nat(offset))),_,U32.to_nat(offset),U32.to_nat(span)>
  nat_cover(problem,U32.to_nat(offset),U32.to_nat(span),Contents.valid_condition(problem,valid,5n),Contents.valid_condition(problem,valid,8n),rows_ordered(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid),
    row_reach(problem,valid,last_row(problem,U32.to_nat(offset),U32.to_nat(span)),last_inside(problem,U32.to_nat(offset),U32.to_nat(span),positive,grid)))
