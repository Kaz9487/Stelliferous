# Bounds for the fixed tile workspace and the aligned block schedule.
import Base
import ../kernel_pipeline.bend as Pipeline
import ../kernel_window.bend as Window
import ./binary_radix.bend as Radix
import ./encoded_coordinates.bend as Encoded
import ./kernel_partition_bounds.bend as Partition
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as U32Extrema
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./nat_ceiling.bend as Ceiling

def groups(target: U32,inner: U32) -> Nat:
  Nat.max(1n,U32.to_nat((target / inner : U32)))

def quotient_exact(+target: U32,+limits: Machine.Limits,+inner: U32,+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((target / inner : U32)) == Nat.div(U32.to_nat(target),U32.to_nat(inner)) : Nat}:
  U32Division.divide(target,inner,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(inner),bound),positive)

def groups_exact(+target: U32,+inner: U32) ->
  {U32.to_nat(U32.max(1,(target / inner : U32))) == groups(target,inner) : Nat}:
  U32Extrema.maximum(1,(target / inner : U32))

def groups_room(+target: U32,+target_positive: {Nat.is_lt(0n,U32.to_nat(target)) == True{} : Bool},+limits: Machine.Limits,+inner: U32,+bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) -> {Nat.is_le(groups(target,inner),U32.to_nat(target)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((target / inner : U32)),Nat.div(U32.to_nat(target),U32.to_nat(inner)),quotient_exact(target,limits,inner,positive,bound)) : {Nat.is_le(Nat.max(1n,_),U32.to_nat(target)) == True{} : Bool}
  +quotient = Nat.div(U32.to_nat(target),U32.to_nat(inner))
  Extrema.maximum_upper(1n,quotient,U32.to_nat(target),Order.strict_follow(0n,U32.to_nat(target),target_positive),Order.transitive(quotient,Nat.mul(quotient,U32.to_nat(inner)),U32.to_nat(target),
    RowMajor.product_covers(quotient,U32.to_nat(inner),positive),DivisionOrder.product_below(U32.to_nat(target),U32.to_nat(inner),positive)))

def groups_positive(+target: U32,+inner: U32) -> {Nat.is_lt(0n,groups(target,inner)) == True{} : Bool}:
  Bounds.predecessor_strict(0n,groups(target,inner),Extrema.maximum_left(1n,U32.to_nat((target / inner : U32))))

def span_room(+target: U32,+target_positive: {Nat.is_lt(0n,U32.to_nat(target)) == True{} : Bool},+limits: Machine.Limits,+inner: U32,+bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) -> {Nat.is_le(Nat.mul(groups(target,inner),8n),Nat.mul(U32.to_nat(target),8n)) == True{} : Bool}:
  RowMajor.mul_le(groups(target,inner),U32.to_nat(target),8n,groups_room(target,target_positive,limits,inner,bound,positive))

def span_exact(+target: U32,+target_positive: {Nat.is_lt(0n,U32.to_nat(target)) == True{} : Bool},+limits: Machine.Limits,+inner: U32,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +span_fit: {Nat.is_le(Nat.mul(U32.to_nat(target),8n),U32.to_nat(upper)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == Nat.mul(groups(target,inner),8n) : Nat}:
  %RowMajor.multiply_commutative(8n,groups(target,inner)) : {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == _ : Nat}
  %groups_exact(target,inner) : {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == Nat.mul(8n,_) : Nat}
  Arithmetic.multiply(8,U32.max(1,(target / inner : U32)),upper,upper_ok,
    %Equal.sym(Nat,U32.to_nat(U32.max(1,(target / inner : U32))),groups(target,inner),groups_exact(target,inner)) : {Nat.is_le(Nat.mul(8n,_),U32.to_nat(upper)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(8n,groups(target,inner)),Nat.mul(groups(target,inner),8n),RowMajor.multiply_commutative(8n,groups(target,inner))) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
    Order.transitive(Nat.mul(groups(target,inner),8n),Nat.mul(U32.to_nat(target),8n),U32.to_nat(upper),span_room(target,target_positive,limits,inner,bound,positive),span_fit))

def tile_count_room(+target: U32,+inner: U32,+upper: U32,+target_bound: {U32.is_le(target,upper) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(upper)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.max(target,inner)),U32.to_nat(upper)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(target,inner)),Nat.max(U32.to_nat(target),U32.to_nat(inner)),U32Extrema.maximum(target,inner)) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
  Extrema.maximum_upper(U32.to_nat(target),U32.to_nat(inner),U32.to_nat(upper),Bounds.as_nat_le(target,upper,target_bound),bound)

def group_tiles(+target: U32,+inner: Nat,+quotient: Nat,+product_room: {Nat.is_le(Nat.mul(quotient,inner),U32.to_nat(target)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.max(1n,quotient),inner),Nat.max(U32.to_nat(target),inner)) == True{} : Bool}:
  match quotient:
    case 0n:
      %Equal.sym(Nat,Nat.add(inner,0n),inner,Addition.zero_right(inner)) : {Nat.is_le(_,Nat.max(U32.to_nat(target),inner)) == True{} : Bool}
      Extrema.maximum_right(U32.to_nat(target),inner)
    case 1n+ +rest:
      Order.transitive(Nat.mul(1n+rest,inner),U32.to_nat(target),Nat.max(U32.to_nat(target),inner),product_room,Extrema.maximum_left(U32.to_nat(target),inner))

def tile_prefix_room(+target: U32,+limits: Machine.Limits,+inner: U32,+bound: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(groups(target,inner),U32.to_nat(inner)),U32.to_nat(U32.max(target,inner))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(target,inner)),Nat.max(U32.to_nat(target),U32.to_nat(inner)),U32Extrema.maximum(target,inner)) : {Nat.is_le(Nat.mul(groups(target,inner),U32.to_nat(inner)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((target / inner : U32)),Nat.div(U32.to_nat(target),U32.to_nat(inner)),quotient_exact(target,limits,inner,positive,bound)) : {Nat.is_le(Nat.mul(Nat.max(1n,_),U32.to_nat(inner)),Nat.max(U32.to_nat(target),U32.to_nat(inner))) == True{} : Bool}
  group_tiles(target,U32.to_nat(inner),Nat.div(U32.to_nat(target),U32.to_nat(inner)),DivisionOrder.product_below(U32.to_nat(target),U32.to_nat(inner),positive))

# The final scheduled block starts inside the padded extent.
def Last(remaining: Nat,+start: Nat,+end: Nat,+span: Nat) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest: {Nat.is_lt(Nat.add(start,Nat.mul(rest,span)),end) == True{} : Bool}

type Schedule<-remaining: Nat,-start: Nat,-end: Nat,-span: Nat> is Data:
  Schedule{covers: {Nat.is_le(end,Nat.add(start,Nat.mul(remaining,span))) == True{} : Bool},last: Last(remaining,start,end,span)}

def ceiling_last(+count: Nat,+end: Nat,+span: Nat,+exact: {count == Ceiling.ceiling(end,span) : Nat},
  +nonempty: {Nat.is_lt(0n,end) == True{} : Bool},+positive: {Nat.is_lt(0n,span) == True{} : Bool}) -> Last(count,0n,end,span):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      %Order.subtract_zero(rest) : {Nat.is_lt(Nat.mul(_,span),end) == True{} : Bool}
      %Equal.sym(Nat,1n+rest,Ceiling.ceiling(end,span),exact) : {Nat.is_lt(Nat.mul(Nat.sub(_,1n),span),end) == True{} : Bool}
      Ceiling.previous_below(end,span,nonempty,positive)

def initial_schedule(+end: Nat,+span: Nat,+nonempty: {Nat.is_lt(0n,end) == True{} : Bool},+positive: {Nat.is_lt(0n,span) == True{} : Bool}) -> Schedule<Ceiling.ceiling(end,span),0n,end,span>:
  Schedule{Ceiling.covers(end,span,positive),ceiling_last(Ceiling.ceiling(end,span),end,span,{==},nonempty,positive)}

def schedule_inside(+rest: Nat,+start: Nat,+end: Nat,+span: Nat,schedule: Schedule<1n+rest,start,end,span>) -> {Nat.is_lt(start,end) == True{} : Bool}:
  Schedule{covers,+last} = schedule
  Order.le_lt_trans(start,Nat.add(start,Nat.mul(rest,span)),end,Order.le_add(start,Nat.mul(rest,span)),last)

def last_advanced(+remaining: Nat,+start: Nat,+end: Nat,+span: Nat,last: Last(1n+remaining,start,end,span)) -> Last(remaining,Nat.add(start,span),end,span):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(Nat.add(start,span),Nat.mul(rest,span)),Nat.add(start,Nat.add(span,Nat.mul(rest,span))),Addition.associative(start,span,Nat.mul(rest,span))) : {Nat.is_lt(_,end) == True{} : Bool}
      last

def schedule_advanced(+rest: Nat,+start: Nat,+end: Nat,+span: Nat,schedule: Schedule<1n+rest,start,end,span>) -> Schedule<rest,Nat.add(start,span),end,span>:
  Schedule{+covers,+last} = schedule
  Schedule{
    %Equal.sym(Nat,Nat.add(Nat.add(start,span),Nat.mul(rest,span)),Nat.add(start,Nat.add(span,Nat.mul(rest,span))),Addition.associative(start,span,Nat.mul(rest,span))) : {Nat.is_le(end,_) == True{} : Bool}
    covers,
    last_advanced(rest,start,end,span,last)}

def schedule_finished(+start: Nat,+end: Nat,+span: Nat,schedule: Schedule<0n,start,end,span>) -> {Nat.is_le(end,start) == True{} : Bool}:
  Schedule{+covers,last} = schedule
  %Addition.zero_right(start) : {Nat.is_le(end,_) == True{} : Bool}
  covers

# Native loop arithmetic is exact under a bound on the padded extent plus one block.
def remaining_exact(+end: U32,+span: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,U32.to_nat(end)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +sum_room: {Nat.is_le(Nat.add(U32.to_nat(end),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(((end + span - 1) / span : U32)) == Ceiling.ceiling(U32.to_nat(end),U32.to_nat(span)) : Nat}:
  +sum = (end + span : U32)
  +sum_exact = Arithmetic.add(end,span,cap,cap_ok,sum_room)
  +sub_exact = Arithmetic.subtract(sum,1,cap,cap_ok,
    %Equal.sym(Nat,U32.to_nat(sum),Nat.add(U32.to_nat(end),U32.to_nat(span)),sum_exact) : {Nat.is_le(1n,_) == True{} : Bool}
    Order.strict_follow(0n,Nat.add(U32.to_nat(end),U32.to_nat(span)),Bounds.nat_trans(0n,U32.to_nat(end),Nat.add(U32.to_nat(end),U32.to_nat(span)),nonempty,Order.le_add(U32.to_nat(end),U32.to_nat(span)))),
    Order.transitive(Nat.sub(U32.to_nat(sum),1n),U32.to_nat(sum),U32.to_nat(cap),Order.sub_le(U32.to_nat(sum),1n),
      %Equal.sym(Nat,U32.to_nat(sum),Nat.add(U32.to_nat(end),U32.to_nat(span)),sum_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} sum_room))
  Equal.trans(Nat,U32.to_nat(((end + span - 1) / span : U32)),Nat.div(Nat.sub(Nat.add(U32.to_nat(end),U32.to_nat(span)),1n),U32.to_nat(span)),Ceiling.ceiling(U32.to_nat(end),U32.to_nat(span)),
    %sum_exact : {U32.to_nat(((end + span - 1) / span : U32)) == Nat.div(Nat.sub(_,1n),U32.to_nat(span)) : Nat}
    %sub_exact : {U32.to_nat(((end + span - 1) / span : U32)) == Nat.div(_,U32.to_nat(span)) : Nat}
    U32Division.divide((sum - 1 : U32),span,cap,cap_ok,divisor_room,positive),
    Ceiling.added_formula(U32.to_nat(end),U32.to_nat(span),nonempty,positive))

def aligned_choice(choice: Bool,+left: U32,+right: U32,left_aligned: Partition.BlockAligned<left>,right_aligned: Partition.BlockAligned<right>) -> Partition.BlockAligned<Bool.pick(U32,choice,left,right)>:
  match choice:
    case False{}: right_aligned
    case True{}: left_aligned

def minimum_aligned(+left: U32,+right: U32,left_aligned: Partition.BlockAligned<left>,right_aligned: Partition.BlockAligned<right>) -> Partition.BlockAligned<U32.min(left,right)>:
  %Equal.sym(U32,U32.min(left,right),Bool.pick(U32,U32.is_lt(left,right),left,right),U32Extrema.minimum_pick(left,right)) : Partition.BlockAligned<_>
  aligned_choice(U32.is_lt(left,right),left,right,left_aligned,right_aligned)

def difference_aligned(+end: U32,+start: U32,+exact: {U32.to_nat((end - start : U32)) == Nat.sub(U32.to_nat(end),U32.to_nat(start)) : Nat},
  end_aligned: Partition.BlockAligned<end>,start_aligned: Partition.BlockAligned<start>) -> Partition.BlockAligned<(end - start : U32)>:
  Partition.BlockAligned{+end_groups,+end_exact} = end_aligned
  Partition.BlockAligned{+start_groups,+start_exact} = start_aligned
  Partition.BlockAligned{Nat.sub(end_groups,start_groups),
    %Equal.sym(Nat,U32.to_nat((end - start : U32)),Nat.sub(U32.to_nat(end),U32.to_nat(start)),exact) : {_ == Nat.mul(Nat.sub(end_groups,start_groups),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(end),Nat.mul(end_groups,8n),end_exact) : {Nat.sub(_,U32.to_nat(start)) == Nat.mul(Nat.sub(end_groups,start_groups),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(start),Nat.mul(start_groups,8n),start_exact) : {Nat.sub(Nat.mul(end_groups,8n),_) == Nat.mul(Nat.sub(end_groups,start_groups),8n) : Nat}
    RowMajor.subtract_products(end_groups,start_groups,8n)}

def sum_aligned(+start: U32,+span: U32,+exact: {U32.to_nat((start + span : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(span)) : Nat},
  start_aligned: Partition.BlockAligned<start>,span_aligned: Partition.BlockAligned<span>) -> Partition.BlockAligned<(start + span : U32)>:
  Partition.BlockAligned{+start_groups,+start_exact} = start_aligned
  Partition.BlockAligned{+span_groups,+span_exact} = span_aligned
  Partition.BlockAligned{Nat.add(start_groups,span_groups),
    %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(U32.to_nat(start),U32.to_nat(span)),exact) : {_ == Nat.mul(Nat.add(start_groups,span_groups),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(start),Nat.mul(start_groups,8n),start_exact) : {Nat.add(_,U32.to_nat(span)) == Nat.mul(Nat.add(start_groups,span_groups),8n) : Nat}
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(span_groups,8n),span_exact) : {Nat.add(Nat.mul(start_groups,8n),_) == Nat.mul(Nat.add(start_groups,span_groups),8n) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.add(start_groups,span_groups),8n),Nat.add(Nat.mul(start_groups,8n),Nat.mul(span_groups,8n)),RowMajor.distrib(start_groups,span_groups,8n))}

def clipped_exact(+start: U32,+end: U32,+span: U32,+difference: {U32.to_nat((end - start : U32)) == Nat.sub(U32.to_nat(end),U32.to_nat(start)) : Nat}) ->
  {U32.to_nat(U32.min(span,(end - start : U32))) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(end),U32.to_nat(start))) : Nat}:
  %difference : {U32.to_nat(U32.min(span,(end - start : U32))) == Nat.min(U32.to_nat(span),_) : Nat}
  U32Extrema.minimum(span,(end - start : U32))

def clipped_end(+start: Nat,+end: Nat,+span: Nat,+ordered: {Nat.is_le(start,end) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,Nat.min(span,Nat.sub(end,start))),end) == True{} : Bool}:
  %Order.restore_shift(start,end,ordered) : {Nat.is_le(Nat.add(start,Nat.min(span,Nat.sub(end,start))),_) == True{} : Bool}
  Order.offset_le(start,Nat.min(span,Nat.sub(end,start)),Nat.sub(end,start),Extrema.minimum_right(span,Nat.sub(end,start)))

def clipped_bound(+start: Nat,+end: Nat,+span: Nat) -> {Nat.is_le(Nat.min(span,Nat.sub(end,start)),end) == True{} : Bool}:
  Order.transitive(Nat.min(span,Nat.sub(end,start)),Nat.sub(end,start),end,Extrema.minimum_right(span,Nat.sub(end,start)),Order.sub_le(end,start))

# A point between this start and the next start belongs to the clipped block.
def clipped_contains(+start: Nat,+end: Nat,+span: Nat,+point: Nat,
  +start_before_end: {Nat.is_le(start,end) == True{} : Bool},+start_before_point: {Nat.is_le(start,point) == True{} : Bool},
  +inside: {Nat.is_lt(point,end) == True{} : Bool},+before_next: {Nat.is_lt(point,Nat.add(start,span)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(point,start),Nat.min(span,Nat.sub(end,start))) == True{} : Bool}:
  +local = Nat.sub(point,start)
  %Equal.sym(Bool,Nat.is_lt(local,Nat.min(span,Nat.sub(end,start))),Nat.is_lt(local,span) && Nat.is_lt(local,Nat.sub(end,start)),Extrema.below_minimum(local,span,Nat.sub(end,start))) : {_ == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(local,span),True{},Order.sub_lt(point,start,span,start_before_point,before_next)) : {_ && Nat.is_lt(local,Nat.sub(end,start)) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(local,Nat.sub(end,start)),Nat.is_lt(Nat.add(start,local),end),Order.below_difference(start,local,end,start_before_end)) : {_ == True{} : Bool}
  %Equal.sym(Nat,Nat.add(start,local),point,Order.restore_shift(start,point,start_before_point)) : {Nat.is_lt(_,end) == True{} : Bool}
  inside

def ceiling_eight(+count: Nat) -> Nat:
  Encoded.ceiling_eight(count)


def workspace_product(+target: U32,+inner: U32,+positions: U32) -> Nat:
  Nat.mul(Nat.div(U32.to_nat(Pipeline.workspace_prefix(target,inner,positions)),8n),U32.to_nat(inner))

def workspace_product_room(+target: U32,+inner: U32,+positions: U32,+full_exact: {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == Nat.mul(groups(target,inner),8n) : Nat},
  old_room: {Nat.is_le(Nat.mul(groups(target,inner),U32.to_nat(inner)),U32.to_nat(U32.max(target,inner))) == True{} : Bool}) ->
  {Nat.is_le(workspace_product(target,inner,positions),U32.to_nat(U32.max(target,inner))) == True{} : Bool}:
  +full = (8 * U32.max(1,(target / inner : U32)) : U32)
  +prefix = Pipeline.workspace_prefix(target,inner,positions)
  +group_bound = {
    %full_exact : {Nat.is_le(Nat.div(U32.to_nat(prefix),8n),Nat.div(_,8n)) == True{} : Bool}
    DivisionOrder.quotient_le(U32.to_nat(prefix),U32.to_nat(full),8n,{==},
      %Equal.sym(Nat,U32.to_nat(prefix),Nat.min(U32.to_nat(full),U32.to_nat(Window.padded(positions))),U32Extrema.minimum(full,Window.padded(positions))) : {Nat.is_le(_,U32.to_nat(full)) == True{} : Bool}
      Extrema.minimum_left(U32.to_nat(full),U32.to_nat(Window.padded(positions)))) :
    {Nat.is_le(Nat.div(U32.to_nat(prefix),8n),Nat.div(Nat.mul(groups(target,inner),8n),8n)) == True{} : Bool}}
  +group_exact =
    {%Addition.zero_right(Nat.mul(groups(target,inner),8n)) : {Nat.div(_,8n) == groups(target,inner) : Nat}
      Radix.quotient(groups(target,inner),0n,8n,{==},{==}) : {Nat.div(Nat.mul(groups(target,inner),8n),8n) == groups(target,inner) : Nat}}
  Order.transitive(workspace_product(target,inner,positions),Nat.mul(groups(target,inner),U32.to_nat(inner)),U32.to_nat(U32.max(target,inner)),
    RowMajor.mul_le(Nat.div(U32.to_nat(prefix),8n),groups(target,inner),U32.to_nat(inner),
      %group_exact : {Nat.is_le(Nat.div(U32.to_nat(prefix),8n),_) == True{} : Bool} group_bound),old_room)

def workspace_count_exact(+target: U32,+inner: U32,+positions: U32,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +full_exact: {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == Nat.mul(groups(target,inner),8n) : Nat},
  old_room: {Nat.is_le(Nat.mul(groups(target,inner),U32.to_nat(inner)),U32.to_nat(U32.max(target,inner))) == True{} : Bool},
  max_room: {Nat.is_le(U32.to_nat(U32.max(target,inner)),U32.to_nat(upper)) == True{} : Bool}) ->
  {U32.to_nat(Pipeline.workspace_tiles(target,inner,positions)) == workspace_product(target,inner,positions) : Nat}:
  +prefix = Pipeline.workspace_prefix(target,inner,positions)
  +quotient_exact = U32Division.divide(prefix,8,2147483648,{==},{==},{==})
  %quotient_exact : {U32.to_nat(Pipeline.workspace_tiles(target,inner,positions)) == Nat.mul(_,U32.to_nat(inner)) : Nat}
  Arithmetic.multiply((prefix / 8 : U32),inner,upper,upper_ok,
    %Equal.sym(Nat,U32.to_nat((prefix / 8 : U32)),Nat.div(U32.to_nat(prefix),8n),quotient_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(inner)),U32.to_nat(upper)) == True{} : Bool}
    Order.transitive(workspace_product(target,inner,positions),U32.to_nat(U32.max(target,inner)),U32.to_nat(upper),workspace_product_room(target,inner,positions,full_exact,old_room),max_room))

def workspace_count_room(+target: U32,+inner: U32,+positions: U32,+upper: U32,+exact: {U32.to_nat(Pipeline.workspace_tiles(target,inner,positions)) == workspace_product(target,inner,positions) : Nat},
  +full_exact: {U32.to_nat((8 * U32.max(1,(target / inner : U32)) : U32)) == Nat.mul(groups(target,inner),8n) : Nat},old_room: {Nat.is_le(Nat.mul(groups(target,inner),U32.to_nat(inner)),U32.to_nat(U32.max(target,inner))) == True{} : Bool},max_room: {Nat.is_le(U32.to_nat(U32.max(target,inner)),U32.to_nat(upper)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(target,inner,positions)),U32.to_nat(upper)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Pipeline.workspace_tiles(target,inner,positions)),workspace_product(target,inner,positions),exact) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
  Order.transitive(workspace_product(target,inner,positions),U32.to_nat(U32.max(target,inner)),U32.to_nat(upper),workspace_product_room(target,inner,positions,full_exact,old_room),max_room)

# One aligned block clips to the remaining real padded schedule.
def current(+start: U32,+padded: U32,+span: U32) -> U32:
  U32.min(span,(padded - start : U32))

type StepBounds<-origin: U32,-positions: U32,-padded: U32,-span: U32,-cap: U32> is Data:
  StepBounds{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    pack_room: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap)) == True{} : Bool},
    loop_room: {Nat.is_le(Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
    padded_end: {Nat.is_le(U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
    padded_aligned: Partition.BlockAligned<padded>,span_aligned: Partition.BlockAligned<span>}

type Step<-origin: U32,-positions: U32,-start: U32,-padded: U32,-span: U32,-cap: U32> is Data:
  Step{pack_exact: {U32.to_nat((origin + start : U32)) == Nat.add(U32.to_nat(origin),U32.to_nat(start)) : Nat},
    advance: {U32.to_nat((start + span : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(span)) : Nat},
    clipped: {U32.to_nat(current(start,padded,span)) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))) : Nat},
    positive: {Nat.is_lt(0n,U32.to_nat(current(start,padded,span))) == True{} : Bool},
    clipped_bound: {Nat.is_le(U32.to_nat(current(start,padded,span)),U32.to_nat(padded)) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    position_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),U32.to_nat(padded)) == True{} : Bool},
    origin_room: {Nat.is_le(Nat.add(U32.to_nat((origin + start : U32)),U32.to_nat(current(start,padded,span))),U32.to_nat(cap)) == True{} : Bool},
    aligned: Partition.BlockAligned<current(start,padded,span)>,next_aligned: Partition.BlockAligned<(start + span : U32)>}

def step_geometry(+origin: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +bounds: StepBounds<origin,positions,padded,span,cap>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},+aligned: Partition.BlockAligned<start>) -> Step<origin,positions,start,padded,span,cap>:
  StepBounds{+cap_ok,+pack_room,+loop_room,+padded_end,+span_positive,+padded_aligned,+span_aligned} = bounds
  +ordered = Bounds.strict_le(U32.to_nat(start),U32.to_nat(padded),inside)
  +padded_room = Order.transitive(U32.to_nat(padded),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.le_add(U32.to_nat(padded),U32.to_nat(span)),loop_room)
  +difference = Arithmetic.subtract(padded,start,cap,cap_ok,ordered,Order.transitive(Nat.sub(U32.to_nat(padded),U32.to_nat(start)),U32.to_nat(padded),U32.to_nat(cap),Order.sub_le(U32.to_nat(padded),U32.to_nat(start)),padded_room))
  +advance = Arithmetic.add(start,span,cap,cap_ok,Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.add_le(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered),loop_room))
  +clipped = clipped_exact(start,padded,span,difference)
  +pack_exact = Arithmetic.add(origin,start,cap,cap_ok,Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(start)),Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap),Order.offset_le(U32.to_nat(origin),U32.to_nat(start),U32.to_nat(padded),ordered),pack_room))
  +position_room = {
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(padded)) == True{} : Bool}
    clipped_end(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered) : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),U32.to_nat(padded)) == True{} : Bool}}
  Step{pack_exact,advance,clipped,
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_lt(0n,_) == True{} : Bool}
    Extrema.minimum_strict_lower(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start)),0n,span_positive,Order.positive_difference(U32.to_nat(padded),U32.to_nat(start),inside)),
    %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(_,U32.to_nat(padded)) == True{} : Bool}
    clipped_bound(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)),
    Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n),position_room,padded_end),position_room,
    %Equal.sym(Nat,U32.to_nat((origin + start : U32)),Nat.add(U32.to_nat(origin),U32.to_nat(start)),pack_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(current(start,padded,span))),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(origin),U32.to_nat(start)),U32.to_nat(current(start,padded,span))),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span)))),Addition.associative(U32.to_nat(origin),U32.to_nat(start),U32.to_nat(current(start,padded,span)))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span)))),Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap),Order.offset_le(U32.to_nat(origin),Nat.add(U32.to_nat(start),U32.to_nat(current(start,padded,span))),U32.to_nat(padded),position_room),pack_room),
    minimum_aligned(span,(padded - start : U32),span_aligned,difference_aligned(padded,start,difference,padded_aligned,aligned)),sum_aligned(start,span,advance,aligned,span_aligned)}

def step_schedule(+origin: U32,+positions: U32,+rest: Nat,+start: U32,+padded: U32,+span: U32,+cap: U32,
  step: Step<origin,positions,start,padded,span,cap>,schedule: Schedule<1n+rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>) -> Schedule<rest,U32.to_nat((start + span : U32)),U32.to_nat(padded),U32.to_nat(span)>:
  Step{pack_exact,+advance,clipped,positive,clipped_bound,end_room,position_room,origin_room,aligned,next_aligned} = step
  %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(U32.to_nat(start),U32.to_nat(span)),advance) : Schedule<rest,_,U32.to_nat(padded),U32.to_nat(span)>
  schedule_advanced(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)

def step_alignment(+origin: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,+cap: U32,step: Step<origin,positions,start,padded,span,cap>) -> Partition.BlockAligned<(start + span : U32)>:
  Step{pack_exact,advance,clipped,positive,clipped_bound,end_room,position_room,origin_room,aligned,next_aligned} = step
  next_aligned

def current_prefix_bound(+positions: U32,+start: U32,+padded: U32,+span: U32,+padded_value: {padded == Window.padded(positions) : U32},
  clipped: {U32.to_nat(current(start,padded,span)) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))) : Nat}) -> {Nat.is_le(U32.to_nat(current(start,padded,span)),U32.to_nat(U32.min(span,Window.padded(positions)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.min(span,Window.padded(positions))),Nat.min(U32.to_nat(span),U32.to_nat(Window.padded(positions))),U32Extrema.minimum(span,Window.padded(positions))) : {Nat.is_le(U32.to_nat(current(start,padded,span)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(_,Nat.min(U32.to_nat(span),U32.to_nat(Window.padded(positions)))) == True{} : Bool}
  %padded_value : {Nat.is_le(Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),Nat.min(U32.to_nat(span),U32.to_nat(_))) == True{} : Bool}
  Extrema.minimum_lower(U32.to_nat(span),U32.to_nat(padded),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),Extrema.minimum_left(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),
    Order.transitive(Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),Nat.sub(U32.to_nat(padded),U32.to_nat(start)),U32.to_nat(padded),Extrema.minimum_right(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),Order.sub_le(U32.to_nat(padded),U32.to_nat(start))))
