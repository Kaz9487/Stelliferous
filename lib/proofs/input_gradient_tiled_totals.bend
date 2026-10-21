# Phase tap counts form a partition of the kernel, including padded rotations.
import Base
import ./input_gradient_phase_taps.bend as Taps
import ./natural_range_sums.bend as Sums
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as Words
import ./binary_radix.bend as Radix

def descending_length(count: Nat,+residue: Nat,+stride: Nat) -> {List.length(&2,Nat,Taps.descending(count,residue,stride)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Taps.descending(rest,residue,stride)),rest,descending_length(rest,residue,stride))

def filtered_length(+count: Nat,+residue: Nat,+stride: Nat,positive: {Nat.is_lt(0n,stride) == True{} : Bool},small: {Nat.is_lt(residue,stride) == True{} : Bool}) ->
  {List.length(&2,Nat,Taps.filtered(count,residue,stride)) == Taps.count(count,residue,stride) : Nat}:
  Equal.trans(Nat,List.length(&2,Nat,Taps.filtered(count,residue,stride)),List.length(&2,Nat,Taps.descending(Taps.count(count,residue,stride),residue,stride)),Taps.count(count,residue,stride),
    Equal.cong(List<&2,Nat>,Nat,values => List.length(&2,Nat,values),Taps.filtered(count,residue,stride),Taps.descending(Taps.count(count,residue,stride),residue,stride),Taps.filtered_count(count,stride,residue,positive,small)),
    descending_length(Taps.count(count,residue,stride),residue,stride))

def selected_length(enabled: Bool,+head: Nat,+tail: List<&2,Nat>) ->
  {List.length(&2,Nat,Taps.selected(enabled,head,tail)) == Nat.add(Bool.pick(Nat,enabled,1n,0n),List.length(&2,Nat,tail)) : Nat}:
  match enabled:
    case False{}: {==}
    case True{}: {==}

def count_successor(+count: Nat,+residue: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},+small: {Nat.is_lt(residue,stride) == True{} : Bool}) ->
  {Taps.count(1n+count,residue,stride) == Nat.add(Bool.pick(Nat,Nat.is_eq(residue,Nat.mod(count,stride)),1n,0n),Taps.count(count,residue,stride)) : Nat}:
  +values = Taps.filtered(count,residue,stride)
  +enabled = Nat.is_eq(residue,Nat.mod(count,stride))
  Equal.trans(Nat,Taps.count(1n+count,residue,stride),List.length(&2,Nat,Taps.filtered(1n+count,residue,stride)),Nat.add(Bool.pick(Nat,enabled,1n,0n),Taps.count(count,residue,stride)),
    Equal.sym(Nat,List.length(&2,Nat,Taps.filtered(1n+count,residue,stride)),Taps.count(1n+count,residue,stride),filtered_length(1n+count,residue,stride,positive,small)),
    Equal.trans(Nat,List.length(&2,Nat,Taps.selected(enabled,count,values)),Nat.add(Bool.pick(Nat,enabled,1n,0n),List.length(&2,Nat,values)),Nat.add(Bool.pick(Nat,enabled,1n,0n),Taps.count(count,residue,stride)),
      selected_length(enabled,count,values),Equal.cong(Nat,Nat,tail => Nat.add(Bool.pick(Nat,enabled,1n,0n),tail),List.length(&2,Nat,values),Taps.count(count,residue,stride),filtered_length(count,residue,stride,positive,small))))

def count_zero(+residue: Nat,+stride: Nat) -> {Taps.count(0n,residue,stride) == 0n : Nat}:
  %Equal.sym(Bool,Nat.is_lt(residue,0n),False{},Division.lt_zero(residue)) :
    {Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(0n,1n),residue),stride),0n) == 0n : Nat}
  {==}

type Indicator is Data:
  Indicator{target: Nat,limit: Nat}

def indicator(context: Indicator,index: Nat) -> Nat:
  Indicator{target,limit} = context
  Bool.pick(Nat,Nat.is_eq(index,target),1n,0n)
def indicator_limit(context: Indicator) -> Nat:
  Indicator{target,limit} = context
  limit
def advance_indicator(context: Indicator) -> Indicator:
  Indicator{target,limit} = context
  Indicator{1n+target,limit}
def advanced_indicator(context: Indicator,index: Nat) -> Nat:
  indicator(advance_indicator(context),index)
def advanced_same(+context: Indicator,+index: Nat) -> {advanced_indicator(context,index) == indicator(advance_indicator(context),index) : Nat}:
  {==}
def one(context: Indicator) -> Nat:
  1n

def indicator_step(+context: Indicator,+index: Nat,inside: {Nat.is_lt(index,indicator_limit(context)) == True{} : Bool}) ->
  {Sums.shifted(~Indicator,~advanced_indicator,~one,context,index) == indicator(context,index) : Nat}:
  Indicator{target,limit} = context
  {==}

def indicator_shift(+count: Nat,+start: Nat,+context: Indicator,room: {Nat.is_le(Nat.add(start,count),indicator_limit(context)) == True{} : Bool}) ->
  {Sums.sum(~Indicator,~indicator,count,1n+start,advance_indicator(context)) == Sums.sum(~Indicator,~indicator,count,start,context) : Nat}:
  %Sums.project_context(~Indicator,~Indicator,~advance_indicator,~advanced_indicator,~indicator,~advanced_same,count,1n+start,context) :
    {_ == Sums.sum(~Indicator,~indicator,count,start,context) : Nat}
  %Sums.shifting(~Indicator,~advanced_indicator,~one,count,start,context) :
    {_ == Sums.sum(~Indicator,~indicator,count,start,context) : Nat}
  Sums.agrees(~Indicator,~indicator_limit,~Sums.shifted(~Indicator,~advanced_indicator,~one),~indicator,~indicator_step,count,start,context,room)

type Above is Data:
  Above{target: Nat,start: Nat,count: Nat,limit: Nat,before: {Nat.is_lt(target,start) == True{} : Bool}}

def above_indicator(context: Above) -> Indicator:
  Above{target,start,count,limit,before} = context
  Indicator{target,limit}
def above_start(context: Above) -> Nat:
  Above{target,start,count,limit,before} = context
  start
def above_count(context: Above) -> Nat:
  Above{target,start,count,limit,before} = context
  count
def above_delta(context: Above,index: Nat) -> Nat:
  indicator(above_indicator(context),index)
def above_same(+context: Above,+index: Nat) -> {above_delta(context,index) == indicator(above_indicator(context),index) : Nat}:
  {==}
def zero(context: Above) -> Nat:
  0n

def above_empty(+context: Above,+index: Nat,inside: {Nat.is_lt(index,above_count(context)) == True{} : Bool}) ->
  {Sums.shifted(~Above,~above_delta,~above_start,context,index) == Sums.constant(~Above,~zero,context,index) : Nat}:
  Above{+target,+start,count,limit,before} = context
  +different = {
    %Order.equal_symmetric(target,Nat.add(start,index)) : {_ == False{} : Bool}
    Order.less_not_equal(target,Nat.add(start,index),Bounds.nat_trans(target,start,Nat.add(start,index),before,Order.le_add(start,index))) :
      {Nat.is_eq(Nat.add(start,index),target) == False{} : Bool}}
  %Equal.sym(Bool,Nat.is_eq(Nat.add(start,index),target),False{},different) : {Bool.pick(Nat,_,1n,0n) == 0n : Nat}
  {==}

def indicator_above(+count: Nat,+start: Nat,+target: Nat,+limit: Nat,before: {Nat.is_lt(target,start) == True{} : Bool}) ->
  {Sums.sum(~Indicator,~indicator,count,start,Indicator{target,limit}) == 0n : Nat}:
  +context = {Above{target,start,count,limit,before} : Above}
  %Sums.project_context(~Above,~Indicator,~above_indicator,~above_delta,~indicator,~above_same,count,start,context) : {_ == 0n : Nat}
  %Addition.zero_right(start) : {Sums.sum(~Above,~above_delta,count,_,context) == 0n : Nat}
  %Sums.shifting(~Above,~above_delta,~above_start,count,0n,context) : {_ == 0n : Nat}
  Equal.trans(Nat,Sums.sum(~Above,~Sums.shifted(~Above,~above_delta,~above_start),count,0n,context),Sums.sum(~Above,~Sums.constant(~Above,~zero),count,0n,context),0n,
    Sums.agrees(~Above,~above_count,~Sums.shifted(~Above,~above_delta,~above_start),~Sums.constant(~Above,~zero),~above_empty,count,0n,context,Order.reflexive(count)),
    Equal.trans(Nat,Sums.sum(~Above,~Sums.constant(~Above,~zero),count,0n,context),Nat.mul(count,0n),0n,Sums.repeated(~Above,~zero,count,0n,context),RowMajor.multiply_zero_right(count)))

def indicator_once(count: Nat,target: Nat,+limit: Nat,+room: {Nat.is_le(count,limit) == True{} : Bool},inside: {Nat.is_lt(target,count) == True{} : Bool}) ->
  {Sums.sum(~Indicator,~indicator,count,0n,Indicator{target,limit}) == 1n : Nat}:
  match count target:
    case 0n target: Empty.absurd({Sums.sum(~Indicator,~indicator,0n,0n,Indicator{target,limit}) == 1n : Nat},Words.false_true(%Division.lt_zero(target) : {_ == True{} : Bool} inside))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Sums.sum(~Indicator,~indicator,1n+rest,0n,Indicator{0n,limit}),Nat.add(1n,Sums.sum(~Indicator,~indicator,rest,1n,Indicator{0n,limit})),
        Sums.successor(~Indicator,~indicator,rest,0n,Indicator{0n,limit})) : {_ == 1n : Nat}
      Equal.cong(Nat,Nat,tail => 1n+tail,Sums.sum(~Indicator,~indicator,rest,1n,Indicator{0n,limit}),0n,indicator_above(rest,1n,0n,limit,{==}))
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Sums.sum(~Indicator,~indicator,1n+rest,0n,Indicator{1n+previous,limit}),Sums.sum(~Indicator,~indicator,rest,1n,Indicator{1n+previous,limit}),
        Sums.successor(~Indicator,~indicator,rest,0n,Indicator{1n+previous,limit})) : {_ == 1n : Nat}
      %Equal.sym(Nat,Sums.sum(~Indicator,~indicator,rest,1n,Indicator{1n+previous,limit}),Sums.sum(~Indicator,~indicator,rest,0n,Indicator{previous,limit}),
        indicator_shift(rest,0n,Indicator{previous,limit},Order.transitive(rest,1n+rest,limit,Bounds.strict_le(rest,1n+rest,Order.lt_succ(rest,rest,Order.reflexive(rest))),room))) : {_ == 1n : Nat}
      indicator_once(rest,previous,limit,Order.transitive(rest,1n+rest,limit,Bounds.strict_le(rest,1n+rest,Order.lt_succ(rest,rest,Order.reflexive(rest))),room),inside)

type Axis is Data:
  Axis{kernel: Nat,stride: Nat,padding: Nat,positive: {Nat.is_lt(0n,stride) == True{} : Bool}}

def axis_stride(context: Axis) -> Nat:
  Axis{kernel,stride,padding,positive} = context
  stride
def raw_taps(context: Axis,index: Nat) -> Nat:
  Axis{kernel,stride,padding,positive} = context
  Taps.count(kernel,index,stride)
def advance_axis(context: Axis) -> Axis:
  Axis{kernel,stride,padding,positive} = context
  Axis{1n+kernel,stride,padding,positive}
def next_taps(context: Axis,index: Nat) -> Nat:
  raw_taps(advance_axis(context),index)
def next_same(+context: Axis,+index: Nat) -> {next_taps(context,index) == raw_taps(advance_axis(context),index) : Nat}:
  {==}
def axis_indicator(context: Axis) -> Indicator:
  Axis{kernel,+stride,padding,positive} = context
  Indicator{Nat.mod(kernel,stride),stride}
def axis_delta(context: Axis,index: Nat) -> Nat:
  indicator(axis_indicator(context),index)
def axis_delta_same(+context: Axis,+index: Nat) -> {axis_delta(context,index) == indicator(axis_indicator(context),index) : Nat}:
  {==}

def tap_increment(+context: Axis,+index: Nat,inside: {Nat.is_lt(index,axis_stride(context)) == True{} : Bool}) ->
  {next_taps(context,index) == Sums.added(~Axis,~axis_delta,~raw_taps,context,index) : Nat}:
  Axis{+kernel,+stride,padding,+positive} = context
  count_successor(kernel,index,stride,positive,inside)

def sum_increment(+context: Axis) ->
  {Sums.sum(~Axis,~raw_taps,axis_stride(context),0n,advance_axis(context)) == Nat.add(1n,Sums.sum(~Axis,~raw_taps,axis_stride(context),0n,context)) : Nat}:
  Axis{+kernel,+stride,padding,+positive} = context
  +delta = Sums.sum(~Axis,~axis_delta,stride,0n,context)
  +current = Sums.sum(~Axis,~raw_taps,stride,0n,context)
  +delta_once = {
    %Equal.sym(Nat,delta,Sums.sum(~Indicator,~indicator,stride,0n,axis_indicator(context)),
      Sums.project_context(~Axis,~Indicator,~axis_indicator,~axis_delta,~indicator,~axis_delta_same,stride,0n,context)) : {_ == 1n : Nat}
    indicator_once(stride,Nat.mod(kernel,stride),stride,Order.reflexive(stride),DivisionOrder.remainder_small(kernel,stride,positive)) : {delta == 1n : Nat}}
  %Sums.project_context(~Axis,~Axis,~advance_axis,~next_taps,~raw_taps,~next_same,stride,0n,context) : {_ == Nat.add(1n,current) : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~next_taps,stride,0n,context),Sums.sum(~Axis,~Sums.added(~Axis,~axis_delta,~raw_taps),stride,0n,context),
    Sums.agrees(~Axis,~axis_stride,~next_taps,~Sums.added(~Axis,~axis_delta,~raw_taps),~tap_increment,stride,0n,context,Order.reflexive(stride))) : {_ == Nat.add(1n,current) : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~Sums.added(~Axis,~axis_delta,~raw_taps),stride,0n,context),Nat.add(delta,current),Sums.addition(~Axis,~axis_delta,~raw_taps,stride,0n,context)) : {_ == Nat.add(1n,current) : Nat}
  Equal.cong(Nat,Nat,delta => Nat.add(delta,current),delta,1n,delta_once)

def zero_axis(context: Axis) -> Axis:
  Axis{kernel,stride,padding,positive} = context
  Axis{0n,stride,padding,positive}
def zero_taps(context: Axis,index: Nat) -> Nat:
  raw_taps(zero_axis(context),index)
def zero_taps_same(+context: Axis,+index: Nat) -> {zero_taps(context,index) == raw_taps(zero_axis(context),index) : Nat}:
  {==}
def axis_zero(context: Axis) -> Nat:
  0n
def zero_taps_empty(+context: Axis,+index: Nat,inside: {Nat.is_lt(index,axis_stride(context)) == True{} : Bool}) ->
  {zero_taps(context,index) == Sums.constant(~Axis,~axis_zero,context,index) : Nat}:
  Axis{kernel,+stride,padding,positive} = context
  count_zero(index,stride)

def raw_complete(kernel: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Sums.sum(~Axis,~raw_taps,stride,0n,Axis{kernel,stride,0n,positive}) == kernel : Nat}:
  match kernel:
    case 0n:
      +context = {Axis{0n,stride,0n,positive} : Axis}
      %Sums.project_context(~Axis,~Axis,~zero_axis,~zero_taps,~raw_taps,~zero_taps_same,stride,0n,context) : {_ == 0n : Nat}
      Equal.trans(Nat,Sums.sum(~Axis,~zero_taps,stride,0n,context),Sums.sum(~Axis,~Sums.constant(~Axis,~axis_zero),stride,0n,context),0n,
        Sums.agrees(~Axis,~axis_stride,~zero_taps,~Sums.constant(~Axis,~axis_zero),~zero_taps_empty,stride,0n,context,Order.reflexive(stride)),
        Equal.trans(Nat,Sums.sum(~Axis,~Sums.constant(~Axis,~axis_zero),stride,0n,context),Nat.mul(stride,0n),0n,Sums.repeated(~Axis,~axis_zero,stride,0n,context),RowMajor.multiply_zero_right(stride)))
    case 1n+ +rest:
      %Equal.sym(Nat,Sums.sum(~Axis,~raw_taps,stride,0n,Axis{1n+rest,stride,0n,positive}),Nat.add(1n,Sums.sum(~Axis,~raw_taps,stride,0n,Axis{rest,stride,0n,positive})),
        sum_increment(Axis{rest,stride,0n,positive})) : {_ == 1n+rest : Nat}
      Equal.cong(Nat,Nat,tail => 1n+tail,Sums.sum(~Axis,~raw_taps,stride,0n,Axis{rest,stride,0n,positive}),rest,raw_complete(rest,stride,positive))

def axis_kernel(context: Axis) -> Nat:
  Axis{kernel,stride,padding,positive} = context
  kernel
def axis_padding(context: Axis) -> Nat:
  Axis{kernel,stride,padding,positive} = context
  padding
def clear_padding(context: Axis) -> Axis:
  Axis{kernel,stride,padding,positive} = context
  Axis{kernel,stride,0n,positive}
def raw_padding_same(+context: Axis,+index: Nat) -> {raw_taps(context,index) == raw_taps(clear_padding(context),index) : Nat}:
  Axis{kernel,stride,padding,positive} = context
  {==}
def periodic_taps(context: Axis,index: Nat) -> Nat:
  Axis{kernel,+stride,padding,positive} = context
  Taps.count(kernel,Nat.mod(index,stride),stride)

def periodic_raw(+context: Axis,+index: Nat,inside: {Nat.is_lt(index,axis_stride(context)) == True{} : Bool}) ->
  {periodic_taps(context,index) == raw_taps(context,index) : Nat}:
  Axis{+kernel,+stride,padding,positive} = context
  Equal.cong(Nat,Nat,residue => Taps.count(kernel,residue,stride),Nat.mod(index,stride),index,
    Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(index,stride),(0n,index),Division.small(index,stride,inside)))

def periodic_complete(+context: Axis) -> {Sums.sum(~Axis,~periodic_taps,axis_stride(context),0n,context) == axis_kernel(context) : Nat}:
  Axis{+kernel,+stride,padding,+positive} = context
  %Equal.sym(Nat,Sums.sum(~Axis,~periodic_taps,stride,0n,context),Sums.sum(~Axis,~raw_taps,stride,0n,context),
    Sums.agrees(~Axis,~axis_stride,~periodic_taps,~raw_taps,~periodic_raw,stride,0n,context,Order.reflexive(stride))) : {_ == kernel : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~raw_taps,stride,0n,context),Sums.sum(~Axis,~raw_taps,stride,0n,clear_padding(context)),
    Sums.project_context(~Axis,~Axis,~clear_padding,~raw_taps,~raw_taps,~raw_padding_same,stride,0n,context)) : {_ == kernel : Nat}
  raw_complete(kernel,stride,positive)

def periodic_step(+context: Axis,+index: Nat,inside: {Nat.is_lt(index,axis_stride(context)) == True{} : Bool}) ->
  {Sums.shifted(~Axis,~periodic_taps,~axis_stride,context,index) == periodic_taps(context,index) : Nat}:
  Axis{+kernel,+stride,padding,+positive} = context
  Equal.cong(Nat,Nat,residue => Taps.count(kernel,residue,stride),Nat.mod(Nat.add(stride,index),stride),Nat.mod(index,stride),
    %Addition.zero_right(stride) : {Nat.mod(Nat.add(_,index),stride) == Nat.mod(index,stride) : Nat}
      DivisionOrder.remainder_shift(index,stride,1n,positive))

def periodic_shift(+context: Axis,+count: Nat,room: {Nat.is_le(count,axis_stride(context)) == True{} : Bool}) ->
  {Sums.sum(~Axis,~periodic_taps,count,axis_stride(context),context) == Sums.sum(~Axis,~periodic_taps,count,0n,context) : Nat}:
  Axis{kernel,+stride,padding,positive} = context
  %Addition.zero_right(stride) : {Sums.sum(~Axis,~periodic_taps,count,_,context) == Sums.sum(~Axis,~periodic_taps,count,0n,context) : Nat}
  %Sums.shifting(~Axis,~periodic_taps,~axis_stride,count,0n,context) : {_ == Sums.sum(~Axis,~periodic_taps,count,0n,context) : Nat}
  Sums.agrees(~Axis,~axis_stride,~Sums.shifted(~Axis,~periodic_taps,~axis_stride),~periodic_taps,~periodic_step,count,0n,context,room)

def rotated_complete(+context: Axis,+start: Nat,+small: {Nat.is_lt(start,axis_stride(context)) == True{} : Bool}) ->
  {Sums.sum(~Axis,~periodic_taps,axis_stride(context),start,context) == axis_kernel(context) : Nat}:
  Axis{kernel,+stride,padding,positive} = context
  +tail = Nat.sub(stride,start)
  +prefix = Sums.sum(~Axis,~periodic_taps,start,0n,context)
  +suffix = Sums.sum(~Axis,~periodic_taps,tail,start,context)
  +ordered = Bounds.strict_le(start,stride,small)
  +end = Order.restore_shift(start,stride,ordered)
  +size = Equal.trans(Nat,Nat.add(tail,start),Nat.add(start,tail),stride,Order.commutative(tail,start),end)
  %size : {Sums.sum(~Axis,~periodic_taps,_,start,context) == kernel : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~periodic_taps,Nat.add(tail,start),start,context),Nat.add(suffix,Sums.sum(~Axis,~periodic_taps,start,Nat.add(start,tail),context)),
    Sums.split(~Axis,~periodic_taps,tail,start,start,context)) : {_ == kernel : Nat}
  %Equal.sym(Nat,Nat.add(start,tail),stride,end) : {Nat.add(suffix,Sums.sum(~Axis,~periodic_taps,start,_,context)) == kernel : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~periodic_taps,start,stride,context),prefix,periodic_shift(context,start,ordered)) : {Nat.add(suffix,_) == kernel : Nat}
  %Order.commutative(prefix,suffix) : {_ == kernel : Nat}
  %Sums.split(~Axis,~periodic_taps,start,tail,0n,context) : {_ == kernel : Nat}
  %Equal.sym(Nat,Nat.add(start,tail),stride,end) : {Sums.sum(~Axis,~periodic_taps,_,0n,context) == kernel : Nat}
  periodic_complete(context)

def padded_residue(+index: Nat,+padding: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.mod(Nat.add(index,padding),stride) == Nat.mod(Nat.add(index,Nat.mod(padding,stride)),stride) : Nat}:
  +quotient = Nat.div(padding,stride)
  +remainder = Nat.mod(padding,stride)
  +prefix = Nat.mul(quotient,stride)
  Equal.trans(Nat,Nat.mod(Nat.add(index,padding),stride),Nat.mod(Nat.add(prefix,Nat.add(index,remainder)),stride),Nat.mod(Nat.add(index,remainder),stride),
    Equal.cong(Nat,Nat,value => Nat.mod(value,stride),Nat.add(index,padding),Nat.add(prefix,Nat.add(index,remainder)),
      Equal.trans(Nat,Nat.add(index,padding),Nat.add(index,Nat.add(prefix,remainder)),Nat.add(prefix,Nat.add(index,remainder)),
        Equal.cong(Nat,Nat,padding => Nat.add(index,padding),padding,Nat.add(prefix,remainder),DivisionOrder.quotient_exact(padding,stride,positive)),RowMajor.add_swap(index,prefix,remainder))),
    DivisionOrder.remainder_shift(Nat.add(index,remainder),stride,quotient,positive))

def cyclic_taps(context: Axis,index: Nat) -> Nat:
  Axis{kernel,+stride,padding,positive} = context
  Taps.count(kernel,Nat.mod(Nat.add(index,padding),stride),stride)
def normalized_axis(context: Axis) -> Axis:
  Axis{kernel,+stride,padding,positive} = context
  Axis{kernel,stride,Nat.mod(padding,stride),positive}
def normalized_same(+context: Axis,+index: Nat) -> {cyclic_taps(context,index) == cyclic_taps(normalized_axis(context),index) : Nat}:
  Axis{+kernel,+stride,+padding,+positive} = context
  Equal.cong(Nat,Nat,residue => Taps.count(kernel,residue,stride),Nat.mod(Nat.add(index,padding),stride),Nat.mod(Nat.add(index,Nat.mod(padding,stride)),stride),padded_residue(index,padding,stride,positive))

def cyclic_shifted_same(+context: Axis,+index: Nat,inside: {Nat.is_lt(index,axis_stride(context)) == True{} : Bool}) ->
  {cyclic_taps(context,index) == Sums.shifted(~Axis,~periodic_taps,~axis_padding,context,index) : Nat}:
  Axis{+kernel,+stride,+padding,positive} = context
  Equal.cong(Nat,Nat,value => Taps.count(kernel,Nat.mod(value,stride),stride),Nat.add(index,padding),Nat.add(padding,index),Order.commutative(index,padding))

def cyclic_complete(+context: Axis) -> {Sums.sum(~Axis,~cyclic_taps,axis_stride(context),0n,context) == axis_kernel(context) : Nat}:
  Axis{+kernel,+stride,+padding,+positive} = context
  +normalized = normalized_axis(context)
  +start = Nat.mod(padding,stride)
  %Equal.sym(Nat,Sums.sum(~Axis,~cyclic_taps,stride,0n,context),Sums.sum(~Axis,~cyclic_taps,stride,0n,normalized),
    Sums.project_context(~Axis,~Axis,~normalized_axis,~cyclic_taps,~cyclic_taps,~normalized_same,stride,0n,context)) : {_ == kernel : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~cyclic_taps,stride,0n,normalized),Sums.sum(~Axis,~Sums.shifted(~Axis,~periodic_taps,~axis_padding),stride,0n,normalized),
    Sums.agrees(~Axis,~axis_stride,~cyclic_taps,~Sums.shifted(~Axis,~periodic_taps,~axis_padding),~cyclic_shifted_same,stride,0n,normalized,Order.reflexive(stride))) : {_ == kernel : Nat}
  %Equal.sym(Nat,Sums.sum(~Axis,~Sums.shifted(~Axis,~periodic_taps,~axis_padding),stride,0n,normalized),Sums.sum(~Axis,~periodic_taps,stride,Nat.add(start,0n),normalized),
    Sums.shifting(~Axis,~periodic_taps,~axis_padding,stride,0n,normalized)) : {_ == kernel : Nat}
  %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Sums.sum(~Axis,~periodic_taps,stride,_,normalized) == kernel : Nat}
  rotated_complete(normalized,start,DivisionOrder.remainder_small(padding,stride,positive))

def cyclic_bounded(+context: Axis,+count: Nat,room: {Nat.is_le(count,axis_stride(context)) == True{} : Bool}) ->
  {Nat.is_le(Sums.sum(~Axis,~cyclic_taps,count,0n,context),axis_kernel(context)) == True{} : Bool}:
  %cyclic_complete(context) : {Nat.is_le(Sums.sum(~Axis,~cyclic_taps,count,0n,context),_) == True{} : Bool}
  Sums.prefix_bounded(~Axis,~cyclic_taps,count,axis_stride(context),0n,context,room)

type Matrix is Data:
  Matrix{axis: Axis,columns: Nat,positive: {Nat.is_lt(0n,columns) == True{} : Bool}}

def matrix_columns(context: Matrix) -> Nat:
  Matrix{axis,columns,positive} = context
  columns
def matrix_axis(context: Matrix) -> Axis:
  Matrix{axis,columns,positive} = context
  axis
def matrix_sample(context: Matrix,index: Nat) -> Nat:
  Matrix{+axis,+columns,positive} = context
  +index = index
  Nat.mul(cyclic_taps(axis,Nat.div(index,columns)),cyclic_taps(axis,Nat.mod(index,columns)))
def matrix_row(context: Matrix,index: Nat) -> Nat:
  Matrix{+axis,columns,positive} = context
  Nat.mul(cyclic_taps(axis,index),Sums.sum(~Axis,~cyclic_taps,columns,0n,axis))

type Row is Data:
  Row{matrix: Matrix,row: Nat}

def row_matrix(context: Row) -> Matrix:
  Row{matrix,row} = context
  matrix
def row_axis(context: Row) -> Axis:
  Row{matrix,row} = context
  matrix_axis(matrix)
def row_columns(context: Row) -> Nat:
  Row{matrix,row} = context
  matrix_columns(matrix)
def row_offset(context: Row) -> Nat:
  Row{matrix,row} = context
  Nat.mul(row,matrix_columns(matrix))
def row_factor(context: Row) -> Nat:
  Row{matrix,row} = context
  cyclic_taps(matrix_axis(matrix),row)
def row_column(context: Row,index: Nat) -> Nat:
  cyclic_taps(row_axis(context),index)
def row_flat(context: Row,index: Nat) -> Nat:
  matrix_sample(row_matrix(context),index)
def row_flat_same(+context: Row,+index: Nat) -> {row_flat(context,index) == matrix_sample(row_matrix(context),index) : Nat}:
  {==}
def row_column_same(+context: Row,+index: Nat) -> {row_column(context,index) == cyclic_taps(row_axis(context),index) : Nat}:
  {==}

def row_shifted_same(+context: Row,+index: Nat,inside: {Nat.is_lt(index,row_columns(context)) == True{} : Bool}) ->
  {Sums.shifted(~Row,~row_flat,~row_offset,context,index) == Sums.scaled(~Row,~row_column,~row_factor,context,index) : Nat}:
  Row{Matrix{+axis,+columns,+positive},+row} = context
  +inside = inside
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(row,columns),index),columns),row,Radix.quotient(row,index,columns,positive,inside)) :
    {Nat.mul(cyclic_taps(axis,_),cyclic_taps(axis,Nat.mod(Nat.add(Nat.mul(row,columns),index),columns))) == Nat.mul(cyclic_taps(axis,row),cyclic_taps(axis,index)) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(row,columns),index),columns),index,Radix.remainder(row,index,columns,positive,inside)) :
    {Nat.mul(cyclic_taps(axis,row),cyclic_taps(axis,_)) == Nat.mul(cyclic_taps(axis,row),cyclic_taps(axis,index)) : Nat}
  {==}

def row_sum(+context: Matrix,+row: Nat) ->
  {Sums.sum(~Matrix,~matrix_sample,matrix_columns(context),Nat.mul(row,matrix_columns(context)),context) == matrix_row(context,row) : Nat}:
  Matrix{+axis,+columns,positive} = context
  +cell = {Row{context,row} : Row}
  +offset = Nat.mul(row,columns)
  %Sums.project_context(~Row,~Matrix,~row_matrix,~row_flat,~matrix_sample,~row_flat_same,columns,offset,cell) : {_ == matrix_row(context,row) : Nat}
  %Addition.zero_right(offset) : {Sums.sum(~Row,~row_flat,columns,_,cell) == matrix_row(context,row) : Nat}
  %Sums.shifting(~Row,~row_flat,~row_offset,columns,0n,cell) : {_ == matrix_row(context,row) : Nat}
  %Equal.sym(Nat,Sums.sum(~Row,~Sums.shifted(~Row,~row_flat,~row_offset),columns,0n,cell),Sums.sum(~Row,~Sums.scaled(~Row,~row_column,~row_factor),columns,0n,cell),
    Sums.agrees(~Row,~row_columns,~Sums.shifted(~Row,~row_flat,~row_offset),~Sums.scaled(~Row,~row_column,~row_factor),~row_shifted_same,columns,0n,cell,Order.reflexive(columns))) : {_ == matrix_row(context,row) : Nat}
  %Equal.sym(Nat,Sums.sum(~Row,~Sums.scaled(~Row,~row_column,~row_factor),columns,0n,cell),Nat.mul(row_factor(cell),Sums.sum(~Row,~row_column,columns,0n,cell)),Sums.scaling(~Row,~row_column,~row_factor,columns,0n,cell)) : {_ == matrix_row(context,row) : Nat}
  %Equal.sym(Nat,Sums.sum(~Row,~row_column,columns,0n,cell),Sums.sum(~Axis,~cyclic_taps,columns,0n,axis),Sums.project_context(~Row,~Axis,~row_axis,~row_column,~cyclic_taps,~row_column_same,columns,0n,cell)) :
    {Nat.mul(row_factor(cell),_) == matrix_row(context,row) : Nat}
  {==}

def matrix_rows(count: Nat,+start: Nat,+context: Matrix) ->
  {Sums.sum(~Matrix,~matrix_sample,Nat.mul(count,matrix_columns(context)),Nat.mul(start,matrix_columns(context)),context) ==
    Sums.sum(~Matrix,~matrix_row,count,start,context) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Matrix{axis,+columns,positive} = context
      +offset = Nat.mul(start,columns)
      +tail_count = Nat.mul(rest,columns)
      +next = Nat.mul(1n+start,columns)
      +tail = Sums.sum(~Matrix,~matrix_sample,tail_count,next,context)
      %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_sample,Nat.add(columns,tail_count),offset,context),
        Nat.add(Sums.sum(~Matrix,~matrix_sample,columns,offset,context),Sums.sum(~Matrix,~matrix_sample,tail_count,Nat.add(offset,columns),context)),
        Sums.split(~Matrix,~matrix_sample,columns,tail_count,offset,context)) : {_ == Sums.sum(~Matrix,~matrix_row,1n+rest,start,context) : Nat}
      %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_sample,columns,offset,context),matrix_row(context,start),row_sum(context,start)) :
        {Nat.add(_,Sums.sum(~Matrix,~matrix_sample,tail_count,Nat.add(offset,columns),context)) == Sums.sum(~Matrix,~matrix_row,1n+rest,start,context) : Nat}
      %Order.commutative(columns,offset) :
        {Nat.add(matrix_row(context,start),Sums.sum(~Matrix,~matrix_sample,tail_count,_,context)) == Sums.sum(~Matrix,~matrix_row,1n+rest,start,context) : Nat}
      %Equal.sym(Nat,tail,Sums.sum(~Matrix,~matrix_row,rest,1n+start,context),matrix_rows(rest,1n+start,context)) :
        {Nat.add(matrix_row(context,start),_) == Sums.sum(~Matrix,~matrix_row,1n+rest,start,context) : Nat}
      Equal.sym(Nat,Sums.sum(~Matrix,~matrix_row,1n+rest,start,context),Nat.add(matrix_row(context,start),Sums.sum(~Matrix,~matrix_row,rest,1n+start,context)),Sums.successor(~Matrix,~matrix_row,rest,start,context))

def matrix_limit(context: Matrix) -> Nat:
  Matrix{axis,columns,positive} = context
  axis_stride(axis)
def matrix_tap(context: Matrix,index: Nat) -> Nat:
  cyclic_taps(matrix_axis(context),index)
def matrix_column_total(context: Matrix) -> Nat:
  Matrix{axis,columns,positive} = context
  Sums.sum(~Axis,~cyclic_taps,columns,0n,axis)
def matrix_tap_same(+context: Matrix,+index: Nat) -> {matrix_tap(context,index) == cyclic_taps(matrix_axis(context),index) : Nat}:
  {==}
def matrix_scaled_same(+context: Matrix,+index: Nat,inside: {Nat.is_lt(index,matrix_limit(context)) == True{} : Bool}) ->
  {matrix_row(context,index) == Sums.scaled(~Matrix,~matrix_tap,~matrix_column_total,context,index) : Nat}:
  Matrix{+axis,+columns,positive} = context
  RowMajor.multiply_commutative(cyclic_taps(axis,index),Sums.sum(~Axis,~cyclic_taps,columns,0n,axis))

def matrix_product(+rows: Nat,+context: Matrix,room: {Nat.is_le(rows,matrix_limit(context)) == True{} : Bool}) ->
  {Sums.sum(~Matrix,~matrix_sample,Nat.mul(rows,matrix_columns(context)),0n,context) ==
    Nat.mul(Sums.sum(~Axis,~cyclic_taps,rows,0n,matrix_axis(context)),matrix_column_total(context)) : Nat}:
  Matrix{+axis,+columns,positive} = context
  %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_sample,Nat.mul(rows,columns),0n,context),Sums.sum(~Matrix,~matrix_row,rows,0n,context),matrix_rows(rows,0n,context)) :
    {_ == Nat.mul(Sums.sum(~Axis,~cyclic_taps,rows,0n,axis),matrix_column_total(context)) : Nat}
  %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_row,rows,0n,context),Sums.sum(~Matrix,~Sums.scaled(~Matrix,~matrix_tap,~matrix_column_total),rows,0n,context),
    Sums.agrees(~Matrix,~matrix_limit,~matrix_row,~Sums.scaled(~Matrix,~matrix_tap,~matrix_column_total),~matrix_scaled_same,rows,0n,context,room)) :
    {_ == Nat.mul(Sums.sum(~Axis,~cyclic_taps,rows,0n,axis),matrix_column_total(context)) : Nat}
  %Equal.sym(Nat,Sums.sum(~Matrix,~Sums.scaled(~Matrix,~matrix_tap,~matrix_column_total),rows,0n,context),Nat.mul(matrix_column_total(context),Sums.sum(~Matrix,~matrix_tap,rows,0n,context)),
    Sums.scaling(~Matrix,~matrix_tap,~matrix_column_total,rows,0n,context)) : {_ == Nat.mul(Sums.sum(~Axis,~cyclic_taps,rows,0n,axis),matrix_column_total(context)) : Nat}
  %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_tap,rows,0n,context),Sums.sum(~Axis,~cyclic_taps,rows,0n,axis),Sums.project_context(~Matrix,~Axis,~matrix_axis,~matrix_tap,~cyclic_taps,~matrix_tap_same,rows,0n,context)) :
    {Nat.mul(matrix_column_total(context),_) == Nat.mul(Sums.sum(~Axis,~cyclic_taps,rows,0n,axis),matrix_column_total(context)) : Nat}
  RowMajor.multiply_commutative(matrix_column_total(context),Sums.sum(~Axis,~cyclic_taps,rows,0n,axis))

def matrix_bounded(+rows: Nat,+context: Matrix,+row_room: {Nat.is_le(rows,matrix_limit(context)) == True{} : Bool},
  column_room: {Nat.is_le(matrix_columns(context),matrix_limit(context)) == True{} : Bool}) ->
  {Nat.is_le(Sums.sum(~Matrix,~matrix_sample,Nat.mul(rows,matrix_columns(context)),0n,context),Nat.mul(axis_kernel(matrix_axis(context)),axis_kernel(matrix_axis(context)))) == True{} : Bool}:
  Matrix{+axis,+columns,positive} = context
  +kernel = axis_kernel(axis)
  +row_sum = Sums.sum(~Axis,~cyclic_taps,rows,0n,axis)
  +column_sum = matrix_column_total(context)
  %Equal.sym(Nat,Sums.sum(~Matrix,~matrix_sample,Nat.mul(rows,columns),0n,context),Nat.mul(row_sum,column_sum),matrix_product(rows,context,row_room)) :
    {Nat.is_le(_,Nat.mul(kernel,kernel)) == True{} : Bool}
  Order.transitive(Nat.mul(row_sum,column_sum),Nat.mul(kernel,column_sum),Nat.mul(kernel,kernel),
    RowMajor.mul_le(row_sum,kernel,column_sum,cyclic_bounded(axis,rows,row_room)),DivisionOrder.multiply_left_le(kernel,column_sum,kernel,cyclic_bounded(axis,columns,column_room)))

type Weighted is Data:
  Weighted{matrix: Matrix,factor: Nat}

def weighted_matrix(context: Weighted) -> Matrix:
  Weighted{matrix,factor} = context
  matrix
def weighted_factor(context: Weighted) -> Nat:
  Weighted{matrix,factor} = context
  factor
def weighted_plain(context: Weighted,index: Nat) -> Nat:
  matrix_sample(weighted_matrix(context),index)
def weighted_plain_same(+context: Weighted,+index: Nat) -> {weighted_plain(context,index) == matrix_sample(weighted_matrix(context),index) : Nat}:
  {==}
def weighted_size(+rows: Nat,context: Weighted) -> Nat:
  Nat.mul(rows,matrix_columns(weighted_matrix(context)))
def weighted_limit(context: Weighted) -> Nat:
  Weighted{+matrix,factor} = context
  Nat.mul(factor,Nat.mul(axis_kernel(matrix_axis(matrix)),axis_kernel(matrix_axis(matrix))))

def weighted_bounded(+rows: Nat,+context: Weighted,
  row_room: {Nat.is_le(rows,matrix_limit(weighted_matrix(context))) == True{} : Bool},
  column_room: {Nat.is_le(matrix_columns(weighted_matrix(context)),matrix_limit(weighted_matrix(context))) == True{} : Bool}) ->
  {Nat.is_le(Sums.sum(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),weighted_size(rows,context),0n,context),weighted_limit(context)) == True{} : Bool}:
  Weighted{+matrix,+factor} = context
  +size = weighted_size(rows,context)
  %Equal.sym(Nat,Sums.sum(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),size,0n,context),Nat.mul(factor,Sums.sum(~Weighted,~weighted_plain,size,0n,context)),
    Sums.scaling(~Weighted,~weighted_plain,~weighted_factor,size,0n,context)) : {Nat.is_le(_,weighted_limit(context)) == True{} : Bool}
  %Equal.sym(Nat,Sums.sum(~Weighted,~weighted_plain,size,0n,context),Sums.sum(~Matrix,~matrix_sample,size,0n,matrix),
    Sums.project_context(~Weighted,~Matrix,~weighted_matrix,~weighted_plain,~matrix_sample,~weighted_plain_same,size,0n,context)) : {Nat.is_le(Nat.mul(factor,_),weighted_limit(context)) == True{} : Bool}
  DivisionOrder.multiply_left_le(factor,Sums.sum(~Matrix,~matrix_sample,size,0n,matrix),Nat.mul(axis_kernel(matrix_axis(matrix)),axis_kernel(matrix_axis(matrix))),matrix_bounded(rows,matrix,row_room,column_room))

def weighted_prefix(+prefix: Nat,+rows: Nat,+context: Weighted,
  selected: {Nat.is_le(prefix,weighted_size(rows,context)) == True{} : Bool},
  row_room: {Nat.is_le(rows,matrix_limit(weighted_matrix(context))) == True{} : Bool},
  column_room: {Nat.is_le(matrix_columns(weighted_matrix(context)),matrix_limit(weighted_matrix(context))) == True{} : Bool}) ->
  {Nat.is_le(Sums.sum(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),prefix,0n,context),weighted_limit(context)) == True{} : Bool}:
  Order.transitive(Sums.sum(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),prefix,0n,context),Sums.sum(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),weighted_size(rows,context),0n,context),weighted_limit(context),
    Sums.prefix_bounded(~Weighted,~Sums.scaled(~Weighted,~weighted_plain,~weighted_factor),prefix,weighted_size(rows,context),0n,context,selected),weighted_bounded(rows,context,row_room,column_room))
