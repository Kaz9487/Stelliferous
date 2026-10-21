# A view in its banded form read by a band and by the whole tensor: the band
# reads, in its own rows, the channel and column the whole view reads in the
# whole tensor.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ./tensor_row_major_coordinates.bend as Coordinates
import ./tensor_band_view_classified.bend as Classified
import ./tensor_copy_values.bend as Copy
import ./storage_shape.bend as Shape
import ./u32_comparison.bend as Comparison
import ./tensor_shape_refinement.bend as ShapeLaws
import ./checked_word_refinement.bend as Checked
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_division.bend as U32Division
import ./u32_full_range.bend as FullRange
import ./word_encoding.bend as WordEncoding

# ---- Channels and columns of digits ----

# The channels, or the columns, the digits of a position over axes move by.
def moved(+plane: U32,+column: Bool,view_axes: List<&2,Views.Axis>,+value: Nat) -> Nat:
  match view_axes:
    case []: 0n
    case Views.Axis{extent,+stride} <> +tail:
      Nat.add(Nat.mul(Nat.div(value,ShapeLaws.elements(Views.axis_extents(tail))),U32.to_nat(BandViews.component(plane,column,stride))),
        moved(plane,column,tail,Nat.mod(value,ShapeLaws.elements(Views.axis_extents(tail)))))

# The most channels, or columns, the axes move by.
def span(+plane: U32,+column: Bool,view_axes: List<&2,Views.Axis>) -> Nat:
  match view_axes:
    case []: 0n
    case Views.Axis{+extent,+stride} <> tail:
      Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(BandViews.component(plane,column,stride))),span(plane,column,tail))

def zero_sum(+factor: Nat) -> {Nat.add(Nat.mul(factor,0n),0n) == 0n : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(factor,0n),0n),Nat.mul(factor,0n),0n,Addition.zero_right(Nat.mul(factor,0n)),RowMajor.multiply_zero_right(factor))

def interchange(+a: Nat,+b: Nat,+c: Nat,+d: Nat) -> {Nat.add(Nat.add(a,b),Nat.add(c,d)) == Nat.add(Nat.add(a,c),Nat.add(b,d)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),Nat.add(c,d)),Nat.add(a,Nat.add(b,Nat.add(c,d))),Nat.add(Nat.add(a,c),Nat.add(b,d)),Addition.associative(a,b,Nat.add(c,d)),
    Equal.trans(Nat,Nat.add(a,Nat.add(b,Nat.add(c,d))),Nat.add(a,Nat.add(c,Nat.add(b,d))),Nat.add(Nat.add(a,c),Nat.add(b,d)),
      Equal.cong(Nat,Nat,sum => Nat.add(a,sum),Nat.add(b,Nat.add(c,d)),Nat.add(c,Nat.add(b,d)),RowMajor.add_swap(b,c,d)),
      Equal.sym(Nat,Nat.add(Nat.add(a,c),Nat.add(b,d)),Nat.add(a,Nat.add(c,Nat.add(b,d))),Addition.associative(a,c,Nat.add(b,d)))))

# A digit times a word of quotient and remainder, added to sums of quotients
# and remainders.
def spread(+factor: Nat,+digit: Nat,+quotient: Nat,+remainder: Nat,+quotients: Nat,+remainders: Nat) ->
  {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),remainder)),Nat.add(Nat.mul(factor,quotients),remainders)) ==
    Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),quotients)),Nat.add(Nat.mul(digit,remainder),remainders)) : Nat}:
  +scaled = Nat.mul(factor,Nat.mul(digit,quotient))
  +own = Nat.mul(digit,remainder)
  +others = Nat.mul(factor,quotients)
  +turned = Equal.trans(Nat,Nat.mul(digit,Nat.mul(quotient,factor)),Nat.mul(Nat.mul(digit,quotient),factor),scaled,
    Equal.sym(Nat,Nat.mul(Nat.mul(digit,quotient),factor),Nat.mul(digit,Nat.mul(quotient,factor)),RowMajor.assoc(digit,quotient,factor)),
    RowMajor.multiply_commutative(Nat.mul(digit,quotient),factor))
  %Equal.sym(Nat,Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),quotients)),Nat.add(scaled,others),RowMajor.right_distrib(factor,Nat.mul(digit,quotient),quotients)) : {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),remainder)),Nat.add(others,remainders)) == Nat.add(_,Nat.add(own,remainders)) : Nat}
  %Equal.sym(Nat,Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),remainder)),Nat.add(Nat.mul(digit,Nat.mul(quotient,factor)),own),RowMajor.right_distrib(digit,Nat.mul(quotient,factor),remainder)) : {Nat.add(_,Nat.add(others,remainders)) == Nat.add(Nat.add(scaled,others),Nat.add(own,remainders)) : Nat}
  %Equal.sym(Nat,Nat.mul(digit,Nat.mul(quotient,factor)),scaled,turned) : {Nat.add(Nat.add(_,own),Nat.add(others,remainders)) == Nat.add(Nat.add(scaled,others),Nat.add(own,remainders)) : Nat}
  interchange(scaled,own,others,remainders)

# A positive plane a word can be divided by.
type Divisor<-plane: U32> is Data:
  Divisor{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},bound: {Nat.is_le(Nat.double(U32.to_nat(plane)),U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool}}

def split_word(+plane: U32,+word: U32,+divisor: Divisor<plane>) ->
  {U32.to_nat(word) == Nat.add(Nat.mul(U32.to_nat(BandViews.component(plane,False{},word)),U32.to_nat(plane)),U32.to_nat(BandViews.component(plane,True{},word))) : Nat}:
  Divisor{+cap,+cap_ok,+bound,+positive} = divisor
  %Equal.sym(Nat,U32.to_nat((word / plane : U32)),Nat.div(U32.to_nat(word),U32.to_nat(plane)),U32Division.divide(word,plane,cap,cap_ok,bound,positive)) :
    {U32.to_nat(word) == Nat.add(Nat.mul(_,U32.to_nat(plane)),U32.to_nat((word % plane : U32))) : Nat}
  %Equal.sym(Nat,U32.to_nat((word % plane : U32)),Nat.mod(U32.to_nat(word),U32.to_nat(plane)),U32Division.modulo(word,plane,cap,cap_ok,bound,positive)) :
    {U32.to_nat(word) == Nat.add(Nat.mul(Nat.div(U32.to_nat(word),U32.to_nat(plane)),U32.to_nat(plane)),_) : Nat}
  DivisionOrder.quotient_exact(U32.to_nat(word),U32.to_nat(plane),positive)

# The digits of a position weigh their channels times the plane and their
# columns.
def weight_whole(+plane: U32,+divisor: Divisor<plane>,+view_axes: List<&2,Views.Axis>,+value: Nat) ->
  {Coordinates.weight(Coordinates.points_of(view_axes,value)) == Nat.add(Nat.mul(U32.to_nat(plane),moved(plane,False{},view_axes,value)),moved(plane,True{},view_axes,value)) : Nat}:
  match view_axes:
    case []: Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(plane),0n),0n),0n,zero_sum(U32.to_nat(plane)))
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +digit = Nat.div(value,rest)
      +remainder = Nat.mod(value,rest)
      +quotient = U32.to_nat(BandViews.component(plane,False{},stride))
      +leftover = U32.to_nat(BandViews.component(plane,True{},stride))
      +factor = U32.to_nat(plane)
      %Equal.sym(Nat,U32.to_nat(stride),Nat.add(Nat.mul(quotient,factor),leftover),split_word(plane,stride,divisor)) :
        {Nat.add(Nat.mul(digit,_),Coordinates.weight(Coordinates.points_of(tail,remainder))) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),moved(plane,False{},tail,remainder))),Nat.add(Nat.mul(digit,leftover),moved(plane,True{},tail,remainder))) : Nat}
      %Equal.sym(Nat,Coordinates.weight(Coordinates.points_of(tail,remainder)),Nat.add(Nat.mul(factor,moved(plane,False{},tail,remainder)),moved(plane,True{},tail,remainder)),
          weight_whole(plane,divisor,tail,remainder)) :
        {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),_) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),moved(plane,False{},tail,remainder))),Nat.add(Nat.mul(digit,leftover),moved(plane,True{},tail,remainder))) : Nat}
      spread(factor,digit,quotient,leftover,moved(plane,False{},tail,remainder),moved(plane,True{},tail,remainder))

# ---- Bounds of digits ----

# A digit below the extent is at most the extent less one.
def digit_steps(+extent: U32,+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_le(digit,U32.to_nat((extent - 1 : U32))) == True{} : Bool}:
  +size = U32.to_nat(extent)
  +positive = Order.le_lt_trans(0n,digit,size,Order.zero_le(digit),inside)
  %Equal.sym(Nat,U32.to_nat(U32.sub(extent,1)),Nat.sub(size,U32.to_nat(1)),FullRange.subtract(extent,1,extent,Order.strict_follow(0n,size,positive),Order.sub_le(size,U32.to_nat(1)))) :
    {Nat.is_le(digit,_) == True{} : Bool}
  Division.lt_succ_le(digit,Nat.sub(size,1n),%Equal.sym(Nat,1n+Nat.sub(size,1n),size,Order.positive_has_last(size,positive)) : {Nat.is_lt(digit,_) == True{} : Bool} inside)

def moved_within(+plane: U32,+column: Bool,+view_axes: List<&2,Views.Axis>,+value: Nat,+inside: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}) ->
  {Nat.is_le(moved(plane,column,view_axes,value),span(plane,column,view_axes)) == True{} : Bool}:
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +size = U32.to_nat(extent)
      +positive = DivisionOrder.right_factor_positive(size,rest,Order.le_lt_trans(0n,value,Nat.mul(size,rest),Order.zero_le(value),inside))
      +digit = Nat.div(value,rest)
      +word = U32.to_nat(BandViews.component(plane,column,stride))
      +steps = U32.to_nat((extent - 1 : U32))
      +others = moved(plane,column,tail,Nat.mod(value,rest))
      Order.transitive(Nat.add(Nat.mul(digit,word),others),Nat.add(Nat.mul(steps,word),others),Nat.add(Nat.mul(steps,word),span(plane,column,tail)),
        Order.add_le(Nat.mul(digit,word),Nat.mul(steps,word),others,RowMajor.mul_le(digit,steps,word,digit_steps(extent,digit,DivisionOrder.quotient_below(value,size,rest,positive,inside)))),
        Order.offset_le(Nat.mul(steps,word),others,span(plane,column,tail),moved_within(plane,column,tail,Nat.mod(value,rest),DivisionOrder.remainder_small(value,rest,positive))))

# ---- The reach of axes ----

def reach_none(+plane: U32,+column: Bool,view_axes: List<&2,Views.Axis>) -> {BandViews.reach(plane,column,view_axes,None{}) == None{} : Maybe<&2,U32>}:
  match view_axes:
    case []: {==}
    case Views.Axis{extent,stride} <> tail: {==}

def none_some(+total: U32,found: {None{} == Some{total} : Maybe<&2,U32>}) -> Empty:
  WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,None{},Some{total},found))

# A start and an amount that sum to a total, when there is a start.
def Sum(start: Maybe<&2,U32>,amount: Nat,+total: U32) -> Data:
  match start:
    case Some{start}: {Nat.add(U32.to_nat(start),amount) == U32.to_nat(total) : Nat}
    case None{}: Empty

# A word added to a start without passing the largest word.
def sum_room(+largest: U32,+moved_word: U32,+start: U32,+named: {largest == 4294967295 : U32},within: {U32.is_le(start,(4294967295 - moved_word : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(moved_word),U32.to_nat(start)),U32.to_nat(largest)) == True{} : Bool}:
  +bound = U32.to_nat(largest)
  +own = U32.to_nat(moved_word)
  +ordered = Bounds.as_nat_le(moved_word,largest,%Equal.sym(U32,largest,4294967295,named) : {U32.is_le(moved_word,_) == True{} : Bool} FullRange.maximum(moved_word))
  +difference = FullRange.subtract(largest,moved_word,largest,ordered,Order.sub_le(bound,own))
  %Order.restore_shift(own,bound,ordered) : {Nat.is_le(Nat.add(own,U32.to_nat(start)),_) == True{} : Bool}
  Order.offset_le(own,U32.to_nat(start),Nat.sub(bound,own),
    %difference : {Nat.is_le(U32.to_nat(start),_) == True{} : Bool}
    Bounds.as_nat_le(start,(largest - moved_word : U32),%Equal.sym(U32,largest,4294967295,named) : {U32.is_le(start,(_ - moved_word : U32)) == True{} : Bool} within))

def sum_step(valid: Bool,+largest: U32,+named: {largest == 4294967295 : U32},+moved_word: U32,+start: U32,+rest: Nat,+total: U32,+computed: {U32.is_le(start,(4294967295 - moved_word : U32)) == valid : Bool},
  later: Sum(Shapes.sum_when(valid,(moved_word + start : U32)),rest,total)) ->
  {Nat.add(U32.to_nat(start),Nat.add(U32.to_nat(moved_word),rest)) == U32.to_nat(total) : Nat}:
  match valid:
    case False{}: Empty.absurd({Nat.add(U32.to_nat(start),Nat.add(U32.to_nat(moved_word),rest)) == U32.to_nat(total) : Nat},later)
    case True{}:
      +own = U32.to_nat(moved_word)
      +first = U32.to_nat(start)
      Equal.trans(Nat,Nat.add(first,Nat.add(own,rest)),Nat.add(Nat.add(own,first),rest),U32.to_nat(total),
        Equal.trans(Nat,Nat.add(first,Nat.add(own,rest)),Nat.add(own,Nat.add(first,rest)),Nat.add(Nat.add(own,first),rest),RowMajor.add_swap(first,own,rest),
          Equal.sym(Nat,Nat.add(Nat.add(own,first),rest),Nat.add(own,Nat.add(first,rest)),Addition.associative(own,first,rest))),
        %FullRange.add(moved_word,start,largest,sum_room(largest,moved_word,start,named,computed)) : {Nat.add(_,rest) == U32.to_nat(total) : Nat} later)

def product_step(product: Maybe<&2,U32>,+largest: U32,+named: {largest == 4294967295 : U32},+extent: U32,+word: U32,+start: U32,+rest: Nat,+total: U32,
  +computed: {Shapes.checked_multiply((extent - 1 : U32),word) == product : Maybe<&2,U32>},later: Sum(Shapes.checked_add_to(product,start),rest,total)) ->
  {Nat.add(U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(word)),rest)) == U32.to_nat(total) : Nat}:
  match product:
    case None{}: Empty.absurd({Nat.add(U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(word)),rest)) == U32.to_nat(total) : Nat},later)
    case Some{+moved_word}:
      %Equal.sym(Nat,Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(word)),U32.to_nat(moved_word),Checked.multiply_sound((extent - 1 : U32),word,moved_word,computed)) :
        {Nat.add(U32.to_nat(start),Nat.add(_,rest)) == U32.to_nat(total) : Nat}
      sum_step(U32.is_le(start,(4294967295 - moved_word : U32)),largest,named,moved_word,start,rest,total,{==},later)

# A reach the axes accept is the start and the span.
def reached(+plane: U32,+column: Bool,view_axes: List<&2,Views.Axis>,start: Maybe<&2,U32>,+total: U32,
  found: {BandViews.reach(plane,column,view_axes,start) == Some{total} : Maybe<&2,U32>}) -> Sum(start,span(plane,column,view_axes),total):
  match view_axes start:
    case [] Some{+start}: Equal.trans(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),U32.to_nat(total),Addition.zero_right(U32.to_nat(start)),Checked.accepted_value(start,total,found))
    case [] None{}: none_some(total,found)
    case Views.Axis{+extent,+stride} <> +tail Some{+start}:
      +word = BandViews.component(plane,column,stride)
      product_step(Shapes.checked_multiply((extent - 1 : U32),word),4294967295,{==},extent,word,start,span(plane,column,tail),total,{==},
        reached(plane,column,tail,Shapes.checked_add_to(Shapes.checked_multiply((extent - 1 : U32),word),start),total,found))
    case Views.Axis{extent,stride} <> tail None{}: none_some(total,found)

def reach_outer(outer: Maybe<&2,U32>,+plane: U32,+column: Bool,+after: List<&2,Views.Axis>,+middle: U32,+limit: U32,
  +computed: {BandViews.reach(plane,column,after,Some{middle}) == outer : Maybe<&2,U32>},below: {BandViews.below(outer,limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(middle),span(plane,column,after)),U32.to_nat(limit)) == True{} : Bool}:
  match outer:
    case Some{+total}:
      %Equal.sym(Nat,Nat.add(U32.to_nat(middle),span(plane,column,after)),U32.to_nat(total),reached(plane,column,after,Some{middle},total,computed)) : {Nat.is_lt(_,U32.to_nat(limit)) == True{} : Bool}
      Bounds.as_nat_lt(total,limit,below)
    case None{}: Empty.absurd({Nat.is_lt(Nat.add(U32.to_nat(middle),span(plane,column,after)),U32.to_nat(limit)) == True{} : Bool},WordEncoding.false_true(below))

def reach_inner(inner: Maybe<&2,U32>,+plane: U32,+column: Bool,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+limit: U32,
  +computed: {BandViews.reach(plane,column,before,Some{start}) == inner : Maybe<&2,U32>},
  below: {BandViews.below(BandViews.reach(plane,column,after,inner),limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(start),Nat.add(span(plane,column,before),span(plane,column,after))),U32.to_nat(limit)) == True{} : Bool}:
  match inner:
    case Some{+middle}:
      %Addition.associative(U32.to_nat(start),span(plane,column,before),span(plane,column,after)) :
        {Nat.is_lt(_,U32.to_nat(limit)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(U32.to_nat(start),span(plane,column,before)),U32.to_nat(middle),reached(plane,column,before,Some{start},middle,computed)) : {Nat.is_lt(Nat.add(_,span(plane,column,after)),U32.to_nat(limit)) == True{} : Bool}
      reach_outer(BandViews.reach(plane,column,after,Some{middle}),plane,column,after,middle,limit,{==},below)
    case None{}:
      Empty.absurd({Nat.is_lt(Nat.add(U32.to_nat(start),Nat.add(span(plane,column,before),span(plane,column,after))),U32.to_nat(limit)) == True{} : Bool},
        WordEncoding.false_true(%reach_none(plane,column,after) : {BandViews.below(_,limit) == True{} : Bool} below))

# Axes before and after the rows that pass the check of a classification stay
# below the limit from their start.
def reach_within(+plane: U32,+column: Bool,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+start: U32,+limit: U32,
  below: {BandViews.below(BandViews.reach(plane,column,after,BandViews.reach(plane,column,before,Some{start})),limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(start),Nat.add(span(plane,column,before),span(plane,column,after))),U32.to_nat(limit)) == True{} : Bool}:
  reach_inner(BandViews.reach(plane,column,before,Some{start}),plane,column,before,after,start,limit,{==},below)

# ---- Strides of a band ----

def extents_localized(+plane: U32,+band: U32,+view_axes: List<&2,Views.Axis>) ->
  {Views.axis_extents(BandViews.localized_axes(plane,band,view_axes)) == Views.axis_extents(view_axes) : List<&2,U32>}:
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => extent <> others,Views.axis_extents(BandViews.localized_axes(plane,band,tail)),Views.axis_extents(tail),
        extents_localized(plane,band,tail))

def localized_length(+plane: U32,+band: U32,+axes: List<&2,Views.Axis>) -> {List.length(&2,Views.Axis,BandViews.localized_axes(plane,band,axes)) == List.length(&2,Views.Axis,axes) : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Views.Axis,BandViews.localized_axes(plane,band,tail)),List.length(&2,Views.Axis,tail),localized_length(plane,band,tail))

# A word of the geometry as the same channel and column in a band, when its
# channel and column stay within bounds whose band words fit a capacity.
def local_exact(+plane: U32,+band: U32,+capacity: U32,+word: U32,+channels: Nat,+columns: Nat,
  channel_bound: {Nat.is_le(U32.to_nat(BandViews.component(plane,False{},word)),channels) == True{} : Bool},
  column_bound: {Nat.is_le(U32.to_nat(BandViews.component(plane,True{},word)),columns) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(channels,U32.to_nat(band)),columns),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(BandViews.localized(plane,band,word)) == Nat.add(Nat.mul(U32.to_nat(BandViews.component(plane,False{},word)),U32.to_nat(band)),U32.to_nat(BandViews.component(plane,True{},word))) : Nat}:
  +channel = BandViews.component(plane,False{},word)
  +column = BandViews.component(plane,True{},word)
  +quotient = U32.to_nat(channel)
  +remainder = U32.to_nat(column)
  +factor = U32.to_nat(band)
  +bound = U32.to_nat(capacity)
  +sum_room = Order.transitive(Nat.add(Nat.mul(quotient,factor),remainder),Nat.add(Nat.mul(channels,factor),columns),bound,
    Order.transitive(Nat.add(Nat.mul(quotient,factor),remainder),Nat.add(Nat.mul(channels,factor),remainder),Nat.add(Nat.mul(channels,factor),columns),
      Order.add_le(Nat.mul(quotient,factor),Nat.mul(channels,factor),remainder,RowMajor.mul_le(quotient,channels,factor,channel_bound)),
      Order.offset_le(Nat.mul(channels,factor),remainder,columns,column_bound)),
    room)
  +product = FullRange.multiply(channel,band,capacity,
    Order.transitive(Nat.mul(quotient,factor),Nat.add(Nat.mul(quotient,factor),remainder),bound,Order.le_add(Nat.mul(quotient,factor),remainder),sum_room))
  Equal.trans(Nat,U32.to_nat(U32.add(U32.mul(channel,band),column)),Nat.add(U32.to_nat(U32.mul(channel,band)),remainder),Nat.add(Nat.mul(quotient,factor),remainder),
    FullRange.add(U32.mul(channel,band),column,capacity,
      %Equal.sym(Nat,U32.to_nat(U32.mul(channel,band)),Nat.mul(quotient,factor),product) : {Nat.is_le(Nat.add(_,remainder),bound) == True{} : Bool} sum_room),
    Equal.cong(Nat,Nat,moved_part => Nat.add(moved_part,remainder),U32.to_nat(U32.mul(channel,band)),Nat.mul(quotient,factor),product))

# A word below a bound is below the bound times a positive factor.
def scaled_bound(+value: Nat,+steps: Nat,+bound: Nat,+positive: {Nat.is_lt(0n,steps) == True{} : Bool},
  spanned: {Nat.is_le(Nat.mul(steps,value),bound) == True{} : Bool}) -> {Nat.is_le(value,bound) == True{} : Bool}:
  Order.transitive(value,Nat.mul(steps,value),bound,
    %RowMajor.multiply_commutative(value,steps) : {Nat.is_le(value,_) == True{} : Bool} RowMajor.product_covers(value,steps,positive),spanned)

# A digit times the local stride of its axis is the digit times its channel
# times the band words plus its column.
def local_word(+plane: U32,+band: U32,+capacity: U32,+extent: U32,+stride: U32,+digit: Nat,+channels: Nat,+columns: Nat,
  +inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  channel_span: {Nat.is_le(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(BandViews.component(plane,False{},stride))),channels) == True{} : Bool},
  column_span: {Nat.is_le(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(BandViews.component(plane,True{},stride))),columns) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(channels,U32.to_nat(band)),columns),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.mul(digit,U32.to_nat(BandViews.localized(plane,band,stride))) ==
    Nat.mul(digit,Nat.add(Nat.mul(U32.to_nat(BandViews.component(plane,False{},stride)),U32.to_nat(band)),U32.to_nat(BandViews.component(plane,True{},stride)))) : Nat}:
  match digit:
    case 0n: {==}
    case 1n+ +rest:
      +steps = U32.to_nat((extent - 1 : U32))
      +positive = Bounds.nat_trans(0n,1n+rest,steps,{==},digit_steps(extent,1n+rest,inside))
      Equal.cong(Nat,Nat,word => Nat.mul(1n+rest,word),U32.to_nat(BandViews.localized(plane,band,stride)),
        Nat.add(Nat.mul(U32.to_nat(BandViews.component(plane,False{},stride)),U32.to_nat(band)),U32.to_nat(BandViews.component(plane,True{},stride))),
        local_exact(plane,band,capacity,stride,channels,columns,
          scaled_bound(U32.to_nat(BandViews.component(plane,False{},stride)),steps,channels,positive,channel_span),
          scaled_bound(U32.to_nat(BandViews.component(plane,True{},stride)),steps,columns,positive,column_span),room))

def left_part(+left: Nat,+right: Nat,+bound: Nat,+sum: {Nat.is_le(Nat.add(left,right),bound) == True{} : Bool}) -> {Nat.is_le(left,bound) == True{} : Bool}:
  Order.transitive(left,Nat.add(left,right),bound,Order.le_add(left,right),sum)

def right_part(+left: Nat,+right: Nat,+bound: Nat,+sum: {Nat.is_le(Nat.add(left,right),bound) == True{} : Bool}) -> {Nat.is_le(right,bound) == True{} : Bool}:
  Order.transitive(right,Nat.add(left,right),bound,%Order.commutative(right,left) : {Nat.is_le(right,_) == True{} : Bool} Order.le_add(right,left),sum)

# The digits of a position over the local axes weigh their channels times the
# band words and their columns.
def weight_local(+plane: U32,+band: U32,+capacity: U32,+channels: Nat,+columns: Nat,+view_axes: List<&2,Views.Axis>,+value: Nat,
  +inside: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool},
  +channel_span: {Nat.is_le(span(plane,False{},view_axes),channels) == True{} : Bool},+column_span: {Nat.is_le(span(plane,True{},view_axes),columns) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(channels,U32.to_nat(band)),columns),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Coordinates.weight(Coordinates.points_of(BandViews.localized_axes(plane,band,view_axes),value)) ==
    Nat.add(Nat.mul(U32.to_nat(band),moved(plane,False{},view_axes,value)),moved(plane,True{},view_axes,value)) : Nat}:
  match view_axes:
    case []: Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(band),0n),0n),0n,zero_sum(U32.to_nat(band)))
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +size = U32.to_nat(extent)
      +positive = DivisionOrder.right_factor_positive(size,rest,Order.le_lt_trans(0n,value,Nat.mul(size,rest),Order.zero_le(value),inside))
      +digit = Nat.div(value,rest)
      +remainder = Nat.mod(value,rest)
      +quotient = U32.to_nat(BandViews.component(plane,False{},stride))
      +leftover = U32.to_nat(BandViews.component(plane,True{},stride))
      +factor = U32.to_nat(band)
      +steps = U32.to_nat((extent - 1 : U32))
      +local_tail = BandViews.localized_axes(plane,band,tail)
      +same = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(local_tail),Views.axis_extents(tail),extents_localized(plane,band,tail))
      +others = moved(plane,False{},tail,remainder)
      +columns_left = moved(plane,True{},tail,remainder)
      %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(local_tail)),rest,same) :
        {Nat.add(Nat.mul(Nat.div(value,_),U32.to_nat(BandViews.localized(plane,band,stride))),Coordinates.weight(Coordinates.points_of(local_tail,Nat.mod(value,_)))) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),columns_left)) : Nat}
      %Equal.sym(Nat,Nat.mul(digit,U32.to_nat(BandViews.localized(plane,band,stride))),Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),
          local_word(plane,band,capacity,extent,stride,digit,channels,columns,DivisionOrder.quotient_below(value,size,rest,positive,inside),
            left_part(Nat.mul(steps,quotient),span(plane,False{},tail),channels,channel_span),left_part(Nat.mul(steps,leftover),span(plane,True{},tail),columns,column_span),room)) :
        {Nat.add(_,Coordinates.weight(Coordinates.points_of(local_tail,remainder))) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),columns_left)) : Nat}
      %Equal.sym(Nat,Coordinates.weight(Coordinates.points_of(local_tail,remainder)),Nat.add(Nat.mul(factor,others),columns_left),
          weight_local(plane,band,capacity,channels,columns,tail,remainder,DivisionOrder.remainder_small(value,rest,positive),
            right_part(Nat.mul(steps,quotient),span(plane,False{},tail),channels,channel_span),right_part(Nat.mul(steps,leftover),span(plane,True{},tail),columns,column_span),room)) :
        {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),_) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),columns_left)) : Nat}
      spread(factor,digit,quotient,leftover,others,columns_left)

# ---- The images, rows and repeats of a cell ----

def extents_repeated(+repeats: List<&2,U32>) -> {Views.axis_extents(BandViews.repeated(repeats)) == repeats : List<&2,U32>}:
  match repeats:
    case []: {==}
    case +extent <> +tail:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => extent <> others,Views.axis_extents(BandViews.repeated(tail)),tail,extents_repeated(tail))

def weight_repeated(+repeats: List<&2,U32>,+value: Nat) -> {Coordinates.weight(Coordinates.points_of(BandViews.repeated(repeats),value)) == 0n : Nat}:
  match repeats:
    case []: {==}
    case +extent <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(BandViews.repeated(tail)))
      %Equal.sym(Nat,Coordinates.weight(Coordinates.points_of(BandViews.repeated(tail),Nat.mod(value,rest))),0n,weight_repeated(tail,Nat.mod(value,rest))) :
        {Nat.add(Nat.mul(Nat.div(value,rest),0n),_) == 0n : Nat}
      zero_sum(Nat.div(value,rest))

# An axis that a geometry of one image, or of one row, leaves out has one
# step and the coordinate zero.
def Present(present: Bool,+extent: Nat,+coordinate: Nat) -> Data:
  match present:
    case True{}: Unit
    case False{}: Shape.Both<{extent == 1n : Nat},{coordinate == 0n : Nat}>

def absent(+count: Nat,+index: Nat,many: {Nat.is_lt(1n,count) == False{} : Bool},inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  Shape.Both<{count == 1n : Nat},{index == 0n : Nat}>:
  match count:
    case 0n: Empty.absurd(Shape.Both<{0n == 1n : Nat},{index == 0n : Nat}>,Order.below_zero(index,inside))
    case 1n: Shape.Both{{==},Coordinates.zero_below_one(index,inside)}
    case 2n+ +rest: Empty.absurd(Shape.Both<{2n+rest == 1n : Nat},{index == 0n : Nat}>,WordEncoding.false_true(Equal.sym(Bool,Nat.is_lt(1n,2n+rest),False{},many)))

def presence(present: Bool,+extent: U32,+coordinate: Nat,+computed: {U32.is_lt(1,extent) == present : Bool},inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  Present(present,U32.to_nat(extent),coordinate):
  match present:
    case True{}: Unit{}
    case False{}:
      absent(U32.to_nat(extent),coordinate,Equal.trans(Bool,Nat.is_lt(1n,U32.to_nat(extent)),U32.is_lt(1,extent),False{},Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(1n,U32.to_nat(extent)),Comparison.lt(1,extent)),computed),inside)

def images_points(many: Bool,+batch: U32,+image: Nat,+stride: U32) -> List<&2,Coordinates.Point>:
  match many:
    case True{}: [Coordinates.Point{batch,image,stride,False{}}]
    case False{}: []

def images_axes(many: Bool,+batch: U32,+image: Nat,+stride: U32) -> {Coordinates.axes(images_points(many,batch,image,stride)) == Classified.images_axes(many,batch,stride) : List<&2,Views.Axis>}:
  match many:
    case True{}: {==}
    case False{}: {==}

def images_position(many: Bool,+batch: U32,+image: Nat,+stride: U32,+present: Present(many,U32.to_nat(batch),image)) ->
  {Coordinates.position(images_points(many,batch,image,stride),0n) == image : Nat}:
  match many:
    case True{}: {==}
    case False{}:
      Shape.Both{one,+zero} = present
      Equal.sym(Nat,image,0n,zero)

def images_weight(many: Bool,+batch: U32,+image: Nat,+stride: U32,+present: Present(many,U32.to_nat(batch),image)) ->
  {Coordinates.weight(images_points(many,batch,image,stride)) == Nat.mul(image,U32.to_nat(stride)) : Nat}:
  match many:
    case True{}: Addition.zero_right(Nat.mul(image,U32.to_nat(stride)))
    case False{}:
      Shape.Both{one,+zero} = present
      %Equal.sym(Nat,image,0n,zero) : {0n == Nat.mul(_,U32.to_nat(stride)) : Nat}
      {==}

def images_valid(many: Bool,+batch: U32,+image: Nat,+stride: U32,+inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool}) ->
  Coordinates.Valid(images_points(many,batch,image,stride)):
  match many:
    case True{}: Shape.Both{inside,Unit{}}
    case False{}: Unit{}

def rows_points(rowed: Bool,+rows: U32,+row: Nat,+width: U32,repeats: List<&2,U32>,+repeat: Nat) -> List<&2,Coordinates.Point>:
  match rowed:
    case True{}: Coordinates.Point{rows,row,width,False{}} <> Coordinates.points_of(BandViews.repeated(repeats),repeat)
    case False{}: Coordinates.points_of(BandViews.repeated(repeats),repeat)

def rows_axes(rowed: Bool,+rows: U32,+row: Nat,+width: U32,+repeats: List<&2,U32>,+repeat: Nat) ->
  {Coordinates.axes(rows_points(rowed,rows,row,width,repeats,repeat)) == BandViews.rows_axes(rowed,rows,width,repeats) : List<&2,Views.Axis>}:
  match rowed:
    case True{}:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{rows,width} <> others,Coordinates.axes(Coordinates.points_of(BandViews.repeated(repeats),repeat)),
        BandViews.repeated(repeats),Coordinates.axes_of(BandViews.repeated(repeats),repeat))
    case False{}: Coordinates.axes_of(BandViews.repeated(repeats),repeat)

def repeats_inside(+repeats: List<&2,U32>,+repeat: Nat,inside: {Nat.is_lt(repeat,ShapeLaws.elements(repeats)) == True{} : Bool}) ->
  {Nat.is_lt(repeat,ShapeLaws.elements(Views.axis_extents(BandViews.repeated(repeats)))) == True{} : Bool}:
  %Equal.sym(List<&2,U32>,Views.axis_extents(BandViews.repeated(repeats)),repeats,extents_repeated(repeats)) : {Nat.is_lt(repeat,ShapeLaws.elements(_)) == True{} : Bool}
  inside

def rows_position(rowed: Bool,+rows: U32,+row: Nat,+width: U32,+repeats: List<&2,U32>,+repeat: Nat,+prefix: Nat,+present: Present(rowed,U32.to_nat(rows),row),
  +inside: {Nat.is_lt(repeat,ShapeLaws.elements(repeats)) == True{} : Bool}) ->
  {Coordinates.position(rows_points(rowed,rows,row,width,repeats,repeat),prefix) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,U32.to_nat(rows)),row),ShapeLaws.elements(repeats)),repeat) : Nat}:
  match rowed:
    case True{}:
      +same = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(BandViews.repeated(repeats)),repeats,extents_repeated(repeats))
      +repeated_inside = repeats_inside(repeats,repeat,inside)
      %same : {Coordinates.position(Coordinates.points_of(BandViews.repeated(repeats),repeat),Nat.add(Nat.mul(prefix,U32.to_nat(rows)),row)) ==
        Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,U32.to_nat(rows)),row),_),repeat) : Nat}
      Coordinates.position_of(BandViews.repeated(repeats),repeat,Nat.add(Nat.mul(prefix,U32.to_nat(rows)),row),repeated_inside)
    case False{}:
      Shape.Both{+one,+zero} = present
      +same = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(BandViews.repeated(repeats)),repeats,extents_repeated(repeats))
      +repeated_inside = repeats_inside(repeats,repeat,inside)
      %Equal.sym(Nat,row,0n,zero) : {Coordinates.position(Coordinates.points_of(BandViews.repeated(repeats),repeat),prefix) ==
        Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,U32.to_nat(rows)),_),ShapeLaws.elements(repeats)),repeat) : Nat}
      %Equal.sym(Nat,U32.to_nat(rows),1n,one) : {Coordinates.position(Coordinates.points_of(BandViews.repeated(repeats),repeat),prefix) ==
        Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,_),0n),ShapeLaws.elements(repeats)),repeat) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.mul(prefix,1n),0n),prefix,Equal.trans(Nat,Nat.add(Nat.mul(prefix,1n),0n),Nat.mul(prefix,1n),prefix,Addition.zero_right(Nat.mul(prefix,1n)),Order.times_one(prefix))) :
        {Coordinates.position(Coordinates.points_of(BandViews.repeated(repeats),repeat),prefix) == Nat.add(Nat.mul(_,ShapeLaws.elements(repeats)),repeat) : Nat}
      %same : {Coordinates.position(Coordinates.points_of(BandViews.repeated(repeats),repeat),prefix) == Nat.add(Nat.mul(prefix,_),repeat) : Nat}
      Coordinates.position_of(BandViews.repeated(repeats),repeat,prefix,repeated_inside)

def rows_weight(rowed: Bool,+rows: U32,+row: Nat,+width: U32,+repeats: List<&2,U32>,+repeat: Nat,+present: Present(rowed,U32.to_nat(rows),row)) ->
  {Coordinates.weight(rows_points(rowed,rows,row,width,repeats,repeat)) == Nat.mul(row,U32.to_nat(width)) : Nat}:
  match rowed:
    case True{}:
      %Equal.sym(Nat,Coordinates.weight(Coordinates.points_of(BandViews.repeated(repeats),repeat)),0n,weight_repeated(repeats,repeat)) :
        {Nat.add(Nat.mul(row,U32.to_nat(width)),_) == Nat.mul(row,U32.to_nat(width)) : Nat}
      Addition.zero_right(Nat.mul(row,U32.to_nat(width)))
    case False{}:
      Shape.Both{one,+zero} = present
      %Equal.sym(Nat,row,0n,zero) : {Coordinates.weight(Coordinates.points_of(BandViews.repeated(repeats),repeat)) == Nat.mul(_,U32.to_nat(width)) : Nat}
      weight_repeated(repeats,repeat)

def rows_valid(rowed: Bool,+rows: U32,+row: Nat,+width: U32,+repeats: List<&2,U32>,+repeat: Nat,+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +inside: {Nat.is_lt(repeat,ShapeLaws.elements(repeats)) == True{} : Bool}) -> Coordinates.Valid(rows_points(rowed,rows,row,width,repeats,repeat)):
  match rowed:
    case True{}: Shape.Both{row_inside,Coordinates.valid_of(BandViews.repeated(repeats),repeat,repeats_inside(repeats,repeat,inside))}
    case False{}: Coordinates.valid_of(BandViews.repeated(repeats),repeat,repeats_inside(repeats,repeat,inside))

# ---- The channel and column a cell reads ----

# The channel, or the column, of the input a cell reads: the start of the
# view and the digits of the cell's channel and column.
def at(+plane: U32,+column: Bool,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+offset: U32,+channel_value: Nat,+column_value: Nat) -> Nat:
  Nat.add(U32.to_nat(BandViews.component(plane,column,offset)),Nat.add(moved(plane,column,before,channel_value),moved(plane,column,after,column_value)))

# The most the start and the axes reach.
def reach_total(+plane: U32,+column: Bool,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+offset: U32) -> Nat:
  Nat.add(U32.to_nat(BandViews.component(plane,column,offset)),Nat.add(span(plane,column,before),span(plane,column,after)))

def at_within(+plane: U32,+column: Bool,+before: List<&2,Views.Axis>,+after: List<&2,Views.Axis>,+offset: U32,+channel_value: Nat,+column_value: Nat,
  +before_inside: {Nat.is_lt(channel_value,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +after_inside: {Nat.is_lt(column_value,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {Nat.is_le(at(plane,column,before,after,offset,channel_value,column_value),reach_total(plane,column,before,after,offset)) == True{} : Bool}:
  +early = moved(plane,column,before,channel_value)
  +late = moved(plane,column,after,column_value)
  Order.offset_le(U32.to_nat(BandViews.component(plane,column,offset)),Nat.add(early,late),Nat.add(span(plane,column,before),span(plane,column,after)),
    Order.transitive(Nat.add(early,late),Nat.add(span(plane,column,before),late),Nat.add(span(plane,column,before),span(plane,column,after)),
      Order.add_le(early,span(plane,column,before),late,moved_within(plane,column,before,channel_value,before_inside)),
      Order.offset_le(span(plane,column,before),late,span(plane,column,after),moved_within(plane,column,after,column_value,after_inside))))

# ---- Sums of the address ----

def collect(+factor: Nat,+a: Nat,+b: Nat,+c: Nat,+d: Nat) ->
  {Nat.add(Nat.add(Nat.mul(factor,a),b),Nat.add(Nat.mul(factor,c),d)) == Nat.add(Nat.mul(factor,Nat.add(a,c)),Nat.add(b,d)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(factor,a),b),Nat.add(Nat.mul(factor,c),d)),Nat.add(Nat.add(Nat.mul(factor,a),Nat.mul(factor,c)),Nat.add(b,d)),
    Nat.add(Nat.mul(factor,Nat.add(a,c)),Nat.add(b,d)),interchange(Nat.mul(factor,a),b,Nat.mul(factor,c),d),
    Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.add(b,d)),Nat.add(Nat.mul(factor,a),Nat.mul(factor,c)),Nat.mul(factor,Nat.add(a,c)),
      Equal.sym(Nat,Nat.mul(factor,Nat.add(a,c)),Nat.add(Nat.mul(factor,a),Nat.mul(factor,c)),RowMajor.right_distrib(factor,a,c))))

# The weight of the images, the axes before the rows, the rows and the axes
# after the rows, with the offset, as channels times a factor and columns.
def assembled(+factor: Nat,+images: Nat,+early_channels: Nat,+early_columns: Nat,+rows: Nat,+late_channels: Nat,+late_columns: Nat,+channel: Nat,+column: Nat) ->
  {Nat.add(Nat.add(images,Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(rows,Nat.add(Nat.mul(factor,late_channels),late_columns)))),Nat.add(Nat.mul(channel,factor),column)) ==
    Nat.add(images,Nat.add(Nat.mul(factor,Nat.add(channel,Nat.add(early_channels,late_channels))),Nat.add(rows,Nat.add(column,Nat.add(early_columns,late_columns))))) : Nat}:
  +early = Nat.add(Nat.mul(factor,early_channels),early_columns)
  +late = Nat.add(Nat.mul(factor,late_channels),late_columns)
  +channels = Nat.add(early_channels,late_channels)
  +columns = Nat.add(early_columns,late_columns)
  +start = Nat.add(Nat.mul(factor,channel),column)
  +grouped = Nat.add(Nat.mul(factor,channels),columns)
  +total = Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(column,columns))
  +inner = Equal.trans(Nat,Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),Nat.add(Nat.add(rows,grouped),start),Nat.add(total,rows),
    Equal.trans(Nat,Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),Nat.add(Nat.add(rows,Nat.add(early,late)),start),Nat.add(Nat.add(rows,grouped),start),
      Equal.trans(Nat,Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),Nat.add(Nat.add(rows,Nat.add(early,late)),Nat.add(Nat.mul(channel,factor),column)),
        Nat.add(Nat.add(rows,Nat.add(early,late)),start),
        Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.add(Nat.mul(channel,factor),column)),Nat.add(early,Nat.add(rows,late)),Nat.add(rows,Nat.add(early,late)),RowMajor.add_swap(early,rows,late)),
        Equal.cong(Nat,Nat,product => Nat.add(Nat.add(rows,Nat.add(early,late)),Nat.add(product,column)),Nat.mul(channel,factor),Nat.mul(factor,channel),RowMajor.multiply_commutative(channel,factor))),
      Equal.cong(Nat,Nat,sum => Nat.add(Nat.add(rows,sum),start),Nat.add(early,late),grouped,collect(factor,early_channels,early_columns,late_channels,late_columns))),
    Equal.trans(Nat,Nat.add(Nat.add(rows,grouped),start),Nat.add(rows,Nat.add(grouped,start)),Nat.add(total,rows),Addition.associative(rows,grouped,start),
      Equal.trans(Nat,Nat.add(rows,Nat.add(grouped,start)),Nat.add(Nat.add(grouped,start),rows),Nat.add(total,rows),Order.commutative(rows,Nat.add(grouped,start)),
        Equal.cong(Nat,Nat,sum => Nat.add(sum,rows),Nat.add(grouped,start),total,
          Equal.trans(Nat,Nat.add(grouped,start),Nat.add(start,grouped),total,Order.commutative(grouped,start),collect(factor,channel,column,channels,columns))))))
  Equal.trans(Nat,Nat.add(Nat.add(images,Nat.add(early,Nat.add(rows,late))),Nat.add(Nat.mul(channel,factor),column)),
    Nat.add(images,Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column))),
    Nat.add(images,Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(rows,Nat.add(column,columns)))),
    Addition.associative(images,Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),
    Equal.cong(Nat,Nat,sum => Nat.add(images,sum),Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),
      Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(rows,Nat.add(column,columns))),
      Equal.trans(Nat,Nat.add(Nat.add(early,Nat.add(rows,late)),Nat.add(Nat.mul(channel,factor),column)),Nat.add(total,rows),
        Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(rows,Nat.add(column,columns))),inner,
        Equal.trans(Nat,Nat.add(total,rows),Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(Nat.add(column,columns),rows)),
          Nat.add(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(rows,Nat.add(column,columns))),
          Addition.associative(Nat.mul(factor,Nat.add(channel,channels)),Nat.add(column,columns),rows),
          Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(factor,Nat.add(channel,channels)),sum),Nat.add(Nat.add(column,columns),rows),Nat.add(rows,Nat.add(column,columns)),
            Order.commutative(Nat.add(column,columns),rows))))))

# ---- The two reads ----

def appended_axes(+left: List<&2,Coordinates.Point>,+right: List<&2,Coordinates.Point>,+left_axes: List<&2,Views.Axis>,+right_axes: List<&2,Views.Axis>,
  left_exact: {Coordinates.axes(left) == left_axes : List<&2,Views.Axis>},right_exact: {Coordinates.axes(right) == right_axes : List<&2,Views.Axis>}) ->
  {Coordinates.axes(List.append(&2,Coordinates.Point,left,right)) == List.append(&2,Views.Axis,left_axes,right_axes) : List<&2,Views.Axis>}:
  Equal.trans(List<&2,Views.Axis>,Coordinates.axes(List.append(&2,Coordinates.Point,left,right)),List.append(&2,Views.Axis,Coordinates.axes(left),Coordinates.axes(right)),
    List.append(&2,Views.Axis,left_axes,right_axes),Coordinates.axes_append(left,right),
    Equal.trans(List<&2,Views.Axis>,List.append(&2,Views.Axis,Coordinates.axes(left),Coordinates.axes(right)),List.append(&2,Views.Axis,left_axes,Coordinates.axes(right)),
      List.append(&2,Views.Axis,left_axes,right_axes),
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,others,Coordinates.axes(right)),Coordinates.axes(left),left_axes,left_exact),
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,left_axes,others),Coordinates.axes(right),right_axes,right_exact)))

def appended_weight(+left: List<&2,Coordinates.Point>,+right: List<&2,Coordinates.Point>,+left_weight: Nat,+right_weight: Nat,
  left_exact: {Coordinates.weight(left) == left_weight : Nat},right_exact: {Coordinates.weight(right) == right_weight : Nat}) ->
  {Coordinates.weight(List.append(&2,Coordinates.Point,left,right)) == Nat.add(left_weight,right_weight) : Nat}:
  Equal.trans(Nat,Coordinates.weight(List.append(&2,Coordinates.Point,left,right)),Nat.add(Coordinates.weight(left),Coordinates.weight(right)),Nat.add(left_weight,right_weight),
    Coordinates.weight_append(left,right),
    Equal.trans(Nat,Nat.add(Coordinates.weight(left),Coordinates.weight(right)),Nat.add(left_weight,Coordinates.weight(right)),Nat.add(left_weight,right_weight),
      Equal.cong(Nat,Nat,sum => Nat.add(sum,Coordinates.weight(right)),Coordinates.weight(left),left_weight,left_exact),
      Equal.cong(Nat,Nat,sum => Nat.add(left_weight,sum),Coordinates.weight(right),right_weight,right_exact)))

# Digits of the axes before the rows, the rows and repeats, and the axes after
# the rows, from a prefix.
def joined_position(+prefix: Nat,+rows: Nat,+row: Nat,+repeats: Nat,+repeat: Nat,+late: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,rows),row),repeats),repeat),late),column) ==
    RowMajor.index(prefix,Nat.mul(rows,repeats),late,Nat.add(Nat.mul(row,repeats),repeat),column) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,rows),row),repeats),repeat),Nat.add(Nat.mul(prefix,Nat.mul(rows,repeats)),Nat.add(Nat.mul(row,repeats),repeat)),Coordinates.digit_join(prefix,rows,row,repeats,repeat)) : {Nat.add(Nat.mul(_,late),column) == RowMajor.index(prefix,Nat.mul(rows,repeats),late,Nat.add(Nat.mul(row,repeats),repeat),column) : Nat}
  Coordinates.digit_join(prefix,Nat.mul(rows,repeats),Nat.add(Nat.mul(row,repeats),repeat),late,column)

# The bound of the local words: a channel below the channels and a column
# below the width of a band of at least one row.
def band_room(+channel: Nat,+channels: Nat,+column: Nat,+width: Nat,+rows: Nat,+band: Nat,+capacity: Nat,
  channel_below: {Nat.is_lt(channel,channels) == True{} : Bool},column_below: {Nat.is_lt(column,width) == True{} : Bool},
  rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},band_exact: {band == Nat.mul(rows,width) : Nat},
  room: {Nat.is_le(Nat.mul(channels,band),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(channel,band),column),capacity) == True{} : Bool}:
  +scaled = Nat.mul(channel,band)
  +wide = {%Equal.sym(Nat,band,Nat.mul(rows,width),band_exact) : {Nat.is_le(width,_) == True{} : Bool} DivisionOrder.le_multiple(rows,width,rows_positive) : {Nat.is_le(width,band) == True{} : Bool}}
  Order.transitive(Nat.add(scaled,column),Nat.add(scaled,band),capacity,
    Order.offset_le(scaled,column,band,Order.transitive(column,width,band,Bounds.strict_le(column,width,column_below),wide)),
    Order.transitive(Nat.add(scaled,band),Nat.mul(channels,band),capacity,
      %Order.commutative(band,scaled) : {Nat.is_le(_,Nat.mul(channels,band)) == True{} : Bool}
      RowMajor.mul_le(1n+channel,channels,band,Order.strict_follow(channel,channels,channel_below)),room))

# A band of rows rows reads through its local view, at the cell of a channel,
# a row and its repeat, and a column of its output, the cell of its own rows at
# the channel and column the view reads.
def band_address(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+capacity: U32,+channel: Nat,+row: Nat,+repeat: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +repeat_inside: {Nat.is_lt(repeat,ShapeLaws.elements(repeats)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool},
  +channel_reach: {Nat.is_lt(reach_total((height * width : U32),False{},before,after,offset),U32.to_nat(channels)) == True{} : Bool},
  +column_reach: {Nat.is_lt(reach_total((height * width : U32),True{},before,after,offset),U32.to_nat(width)) == True{} : Bool},
  +band_exact: {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat((rows * width : U32))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Copy.address(BandViews.local(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width},rows),
      RowMajor.index(channel,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)),ShapeLaws.elements(Views.axis_extents(after)),Nat.add(Nat.mul(row,ShapeLaws.elements(repeats)),repeat),column)) ==
    U32.from_nat(RowMajor.index(at((height * width : U32),False{},before,after,offset,channel,column),U32.to_nat(rows),U32.to_nat(width),row,
      at((height * width : U32),True{},before,after,offset,channel,column))) : U32}:
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +factor = U32.to_nat(band)
  +local_before = BandViews.localized_axes(plane,band,before)
  +local_after = BandViews.localized_axes(plane,band,after)
  +rowed_flag = {True{} : Bool}
  +early = Coordinates.points_of(local_before,channel)
  +middle = rows_points(rowed_flag,rows,row,width,repeats,repeat)
  +late = Coordinates.points_of(local_after,column)
  +tail = List.append(&2,Coordinates.Point,middle,late)
  +points = List.append(&2,Coordinates.Point,early,tail)
  +repeated = ShapeLaws.elements(repeats)
  +late_size = ShapeLaws.elements(Views.axis_extents(after))
  +same_before = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(local_before),Views.axis_extents(before),extents_localized(plane,band,before))
  +same_after = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(local_after),Views.axis_extents(after),extents_localized(plane,band,after))
  +early_inside = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(local_before)),ShapeLaws.elements(Views.axis_extents(before)),same_before) : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside :
    {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(local_before))) == True{} : Bool}}
  +late_inside = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(local_after)),ShapeLaws.elements(Views.axis_extents(after)),same_after) : {Nat.is_lt(column,_) == True{} : Bool} column_inside :
    {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(local_after))) == True{} : Bool}}
  +channel_total = reach_total(plane,False{},before,after,offset)
  +column_total = reach_total(plane,True{},before,after,offset)
  +start_channel = U32.to_nat(BandViews.component(plane,False{},offset))
  +start_column = U32.to_nat(BandViews.component(plane,True{},offset))
  +channel_spans = right_part(start_channel,Nat.add(span(plane,False{},before),span(plane,False{},after)),channel_total,Order.reflexive(channel_total))
  +column_spans = right_part(start_column,Nat.add(span(plane,True{},before),span(plane,True{},after)),column_total,Order.reflexive(column_total))
  +local_room = band_room(channel_total,U32.to_nat(channels),column_total,U32.to_nat(width),U32.to_nat(rows),factor,U32.to_nat(capacity),channel_reach,column_reach,
    Order.le_lt_trans(0n,row,U32.to_nat(rows),Order.zero_le(row),row_inside),band_exact,room)
  +valid = Coordinates.valid_append(early,tail,Coordinates.valid_of(local_before,channel,early_inside),
    Coordinates.valid_append(middle,late,rows_valid(rowed_flag,rows,row,width,repeats,repeat,row_inside,repeat_inside),Coordinates.valid_of(local_after,column,late_inside)))
  +axes_exact = appended_axes(early,tail,local_before,List.append(&2,Views.Axis,BandViews.rows_axes(rowed_flag,rows,width,repeats),local_after),
    Coordinates.axes_of(local_before,channel),
    appended_axes(middle,late,BandViews.rows_axes(rowed_flag,rows,width,repeats),local_after,rows_axes(rowed_flag,rows,row,width,repeats,repeat),Coordinates.axes_of(local_after,column)))
  +inner = Nat.add(Nat.mul(Nat.add(Nat.mul(channel,U32.to_nat(rows)),row),repeated),repeat)
  +position_exact = Equal.trans(Nat,Coordinates.position(points,0n),Coordinates.position(late,Coordinates.position(middle,channel)),
    RowMajor.index(channel,Nat.mul(U32.to_nat(rows),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column),
    Equal.trans(Nat,Coordinates.position(points,0n),Coordinates.position(tail,channel),Coordinates.position(late,Coordinates.position(middle,channel)),
      %Coordinates.position_of(local_before,channel,0n,early_inside) : {Coordinates.position(points,0n) == Coordinates.position(tail,_) : Nat}
      Coordinates.position_append(early,tail,0n),Coordinates.position_append(middle,late,channel)),
    %Equal.sym(Nat,Coordinates.position(middle,channel),inner,rows_position(rowed_flag,rows,row,width,repeats,repeat,channel,Unit{},repeat_inside)) :
      {Coordinates.position(late,_) == RowMajor.index(channel,Nat.mul(U32.to_nat(rows),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column) : Nat}
    Equal.trans(Nat,Coordinates.position(late,inner),Nat.add(Nat.mul(inner,late_size),column),
      RowMajor.index(channel,Nat.mul(U32.to_nat(rows),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column),
      %same_after : {Coordinates.position(late,inner) == Nat.add(Nat.mul(inner,_),column) : Nat} Coordinates.position_of(local_after,column,inner,late_inside),
      joined_position(channel,U32.to_nat(rows),row,repeated,repeat,late_size,column)))
  +early_channels = moved(plane,False{},before,channel)
  +early_columns = moved(plane,True{},before,channel)
  +late_channels = moved(plane,False{},after,column)
  +late_columns = moved(plane,True{},after,column)
  +row_words = Nat.mul(row,U32.to_nat(width))
  +weight_exact = appended_weight(early,tail,Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns)),
    weight_local(plane,band,capacity,channel_total,column_total,before,channel,channel_inside,
      left_part(span(plane,False{},before),span(plane,False{},after),channel_total,channel_spans),left_part(span(plane,True{},before),span(plane,True{},after),column_total,column_spans),local_room),
    appended_weight(middle,late,row_words,Nat.add(Nat.mul(factor,late_channels),late_columns),rows_weight(rowed_flag,rows,row,width,repeats,repeat,Unit{}),
      weight_local(plane,band,capacity,channel_total,column_total,after,column,column_inside,
        right_part(span(plane,False{},before),span(plane,False{},after),channel_total,channel_spans),right_part(span(plane,True{},before),span(plane,True{},after),column_total,column_spans),local_room)))
  +offset_exact = local_exact(plane,band,capacity,offset,channel_total,column_total,
    left_part(start_channel,Nat.add(span(plane,False{},before),span(plane,False{},after)),channel_total,Order.reflexive(channel_total)),
    left_part(start_column,Nat.add(span(plane,True{},before),span(plane,True{},after)),column_total,Order.reflexive(column_total)),local_room)
  +channel_read = at(plane,False{},before,after,offset,channel,column)
  +column_read = at(plane,True{},before,after,offset,channel,column)
  %axes_exact : {Copy.address(Views.View{_,BandViews.localized(plane,band,offset)},
      RowMajor.index(channel,Nat.mul(U32.to_nat(rows),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column)) ==
    U32.from_nat(RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read)) : U32}
  %position_exact : {Copy.address(Views.View{Coordinates.axes(points),BandViews.localized(plane,band,offset)},_) ==
    U32.from_nat(RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read)) : U32}
  Equal.trans(U32,Copy.address(Views.View{Coordinates.axes(points),BandViews.localized(plane,band,offset)},Coordinates.position(points,0n)),
    U32.from_nat(Nat.add(Coordinates.weight(points),U32.to_nat(BandViews.localized(plane,band,offset)))),
    U32.from_nat(RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read)),
    Coordinates.weighed(points,BandViews.localized(plane,band,offset),valid),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(Coordinates.weight(points),U32.to_nat(BandViews.localized(plane,band,offset))),
      RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read),
      %Equal.sym(Nat,Coordinates.weight(points),Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns))),weight_exact) :
        {Nat.add(_,U32.to_nat(BandViews.localized(plane,band,offset))) == RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read) : Nat}
      %Equal.sym(Nat,U32.to_nat(BandViews.localized(plane,band,offset)),Nat.add(Nat.mul(start_channel,factor),start_column),offset_exact) :
        {Nat.add(Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns))),_) ==
          RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read) : Nat}
      Equal.trans(Nat,Nat.add(Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns))),Nat.add(Nat.mul(start_channel,factor),start_column)),
        Nat.add(Nat.mul(factor,channel_read),Nat.add(row_words,column_read)),RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read),
        assembled(factor,0n,early_channels,early_columns,row_words,late_channels,late_columns,start_channel,start_column),
        %Equal.sym(Nat,U32.to_nat(band),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),band_exact) : {Nat.add(Nat.mul(_,channel_read),Nat.add(row_words,column_read)) == RowMajor.index(channel_read,U32.to_nat(rows),U32.to_nat(width),row,column_read) : Nat}
        Equal.cong(Nat,Nat,product => Nat.add(product,Nat.add(row_words,column_read)),Nat.mul(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),channel_read),
          Nat.mul(channel_read,Nat.mul(U32.to_nat(rows),U32.to_nat(width))),RowMajor.multiply_commutative(Nat.mul(U32.to_nat(rows),U32.to_nat(width)),channel_read)))))

# An image, a channel and a column read whole.
def image_join(+image: Nat,+channels: Nat,+plane: Nat,+channel: Nat,+rest: Nat) ->
  {Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(plane,channel),rest)) == Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),rest) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),Nat.add(Nat.mul(Nat.mul(image,channels),plane),Nat.mul(channel,plane)),RowMajor.distrib(Nat.mul(image,channels),channel,plane)) : {Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(plane,channel),rest)) == Nat.add(_,rest) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),plane),Nat.mul(image,Nat.mul(channels,plane)),RowMajor.assoc(image,channels,plane)) : {Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(plane,channel),rest)) == Nat.add(Nat.add(_,Nat.mul(channel,plane)),rest) : Nat}
  %RowMajor.multiply_commutative(channel,plane) : {Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(_,rest)) == Nat.add(Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.mul(channel,plane)),rest) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.mul(channel,plane)),rest),Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),rest)),
    Addition.associative(Nat.mul(image,Nat.mul(channels,plane)),Nat.mul(channel,plane),rest))

# The whole view of a banded form reads, at the output cell of an image, a
# channel, a row and its repeat, and a column, the input pixel of that image
# and row at the channel and column the view reads.
def whole_address(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+channel: Nat,+row: Nat,+repeat: Nat,+column: Nat,
  +divisor: Divisor<(height * width : U32)>,+plane_exact: {U32.to_nat((height * width : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +image_exact: {U32.to_nat((channels * (height * width : U32) : U32)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat},
  +image_inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},+repeat_inside: {Nat.is_lt(repeat,ShapeLaws.elements(repeats)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {Copy.address(Views.View{Classified.whole_axes(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width}),offset},
      RowMajor.index(Nat.add(Nat.mul(image,ShapeLaws.elements(Views.axis_extents(before))),channel),Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)),
        ShapeLaws.elements(Views.axis_extents(after)),Nat.add(Nat.mul(row,ShapeLaws.elements(repeats)),repeat),column)) ==
    U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),at((height * width : U32),False{},before,after,offset,channel,column)),U32.to_nat(height),U32.to_nat(width),row,
      at((height * width : U32),True{},before,after,offset,channel,column))) : U32}:
  +plane = (height * width : U32)
  +factor = U32.to_nat(plane)
  +stride = (channels * plane : U32)
  +many = U32.is_lt(1,batch)
  +rowed_flag = U32.is_lt(1,height)
  +images = images_points(many,batch,image,stride)
  +early = Coordinates.points_of(before,channel)
  +middle = rows_points(rowed_flag,height,row,width,repeats,repeat)
  +late = Coordinates.points_of(after,column)
  +tail = List.append(&2,Coordinates.Point,middle,late)
  +rest = List.append(&2,Coordinates.Point,early,tail)
  +points = List.append(&2,Coordinates.Point,images,rest)
  +image_present = presence(many,batch,image,{==},image_inside)
  +row_present = presence(rowed_flag,height,row,{==},row_inside)
  +repeated = ShapeLaws.elements(repeats)
  +early_size = ShapeLaws.elements(Views.axis_extents(before))
  +late_size = ShapeLaws.elements(Views.axis_extents(after))
  +valid = Coordinates.valid_append(images,rest,images_valid(many,batch,image,stride,image_inside),
    Coordinates.valid_append(early,tail,Coordinates.valid_of(before,channel,channel_inside),
      Coordinates.valid_append(middle,late,rows_valid(rowed_flag,height,row,width,repeats,repeat,row_inside,repeat_inside),Coordinates.valid_of(after,column,column_inside))))
  +axes_exact = appended_axes(images,rest,Classified.images_axes(many,batch,stride),List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,BandViews.rows_axes(rowed_flag,height,width,repeats),after)),
    images_axes(many,batch,image,stride),
    appended_axes(early,tail,before,List.append(&2,Views.Axis,BandViews.rows_axes(rowed_flag,height,width,repeats),after),Coordinates.axes_of(before,channel),
      appended_axes(middle,late,BandViews.rows_axes(rowed_flag,height,width,repeats),after,rows_axes(rowed_flag,height,row,width,repeats,repeat),Coordinates.axes_of(after,column))))
  +leading = Nat.add(Nat.mul(image,early_size),channel)
  +inner = Nat.add(Nat.mul(Nat.add(Nat.mul(leading,U32.to_nat(height)),row),repeated),repeat)
  +position_exact = Equal.trans(Nat,Coordinates.position(points,0n),Coordinates.position(late,Coordinates.position(middle,leading)),
    RowMajor.index(leading,Nat.mul(U32.to_nat(height),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column),
    Equal.trans(Nat,Coordinates.position(points,0n),Coordinates.position(tail,leading),Coordinates.position(late,Coordinates.position(middle,leading)),
      Equal.trans(Nat,Coordinates.position(points,0n),Coordinates.position(rest,image),Coordinates.position(tail,leading),
        %images_position(many,batch,image,stride,image_present) : {Coordinates.position(points,0n) == Coordinates.position(rest,_) : Nat}
        Coordinates.position_append(images,rest,0n),
        %Coordinates.position_of(before,channel,image,channel_inside) : {Coordinates.position(rest,image) == Coordinates.position(tail,_) : Nat}
        Coordinates.position_append(early,tail,image)),
      Coordinates.position_append(middle,late,leading)),
    %Equal.sym(Nat,Coordinates.position(middle,leading),inner,rows_position(rowed_flag,height,row,width,repeats,repeat,leading,row_present,repeat_inside)) :
      {Coordinates.position(late,_) == RowMajor.index(leading,Nat.mul(U32.to_nat(height),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column) : Nat}
    Equal.trans(Nat,Coordinates.position(late,inner),Nat.add(Nat.mul(inner,late_size),column),
      RowMajor.index(leading,Nat.mul(U32.to_nat(height),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column),
      Coordinates.position_of(after,column,inner,column_inside),joined_position(leading,U32.to_nat(height),row,repeated,repeat,late_size,column)))
  +early_channels = moved(plane,False{},before,channel)
  +early_columns = moved(plane,True{},before,channel)
  +late_channels = moved(plane,False{},after,column)
  +late_columns = moved(plane,True{},after,column)
  +row_words = Nat.mul(row,U32.to_nat(width))
  +image_words = Nat.mul(image,U32.to_nat(stride))
  +rest_weight = Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns)))
  +weight_exact = appended_weight(images,rest,image_words,rest_weight,images_weight(many,batch,image,stride,image_present),
    appended_weight(early,tail,Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns)),
      weight_whole(plane,divisor,before,channel),
      appended_weight(middle,late,row_words,Nat.add(Nat.mul(factor,late_channels),late_columns),rows_weight(rowed_flag,height,row,width,repeats,repeat,row_present),
        weight_whole(plane,divisor,after,column))))
  +start_channel = U32.to_nat(BandViews.component(plane,False{},offset))
  +start_column = U32.to_nat(BandViews.component(plane,True{},offset))
  +channel_read = at(plane,False{},before,after,offset,channel,column)
  +column_read = at(plane,True{},before,after,offset,channel,column)
  +read = RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel_read),U32.to_nat(height),U32.to_nat(width),row,column_read)
  %axes_exact : {Copy.address(Views.View{_,offset},RowMajor.index(leading,Nat.mul(U32.to_nat(height),repeated),late_size,Nat.add(Nat.mul(row,repeated),repeat),column)) == U32.from_nat(read) : U32}
  %position_exact : {Copy.address(Views.View{Coordinates.axes(points),offset},_) == U32.from_nat(read) : U32}
  Equal.trans(U32,Copy.address(Views.View{Coordinates.axes(points),offset},Coordinates.position(points,0n)),U32.from_nat(Nat.add(Coordinates.weight(points),U32.to_nat(offset))),U32.from_nat(read),
    Coordinates.weighed(points,offset,valid),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(Coordinates.weight(points),U32.to_nat(offset)),read,
      %Equal.sym(Nat,Coordinates.weight(points),Nat.add(image_words,rest_weight),weight_exact) : {Nat.add(_,U32.to_nat(offset)) == read : Nat}
      %Equal.sym(Nat,U32.to_nat(offset),Nat.add(Nat.mul(start_channel,factor),start_column),split_word(plane,offset,divisor)) : {Nat.add(Nat.add(image_words,rest_weight),_) == read : Nat}
      Equal.trans(Nat,Nat.add(Nat.add(image_words,rest_weight),Nat.add(Nat.mul(start_channel,factor),start_column)),
        Nat.add(image_words,Nat.add(Nat.mul(factor,channel_read),Nat.add(row_words,column_read))),read,
        assembled(factor,image_words,early_channels,early_columns,row_words,late_channels,late_columns,start_channel,start_column),
        %Equal.sym(Nat,U32.to_nat(stride),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),image_exact) : {Nat.add(Nat.mul(image,_),Nat.add(Nat.mul(factor,channel_read),Nat.add(row_words,column_read))) == read : Nat}
        %Equal.sym(Nat,factor,Nat.mul(U32.to_nat(height),U32.to_nat(width)),plane_exact) : {Nat.add(Nat.mul(image,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),Nat.add(Nat.mul(_,channel_read),Nat.add(row_words,column_read))) == read : Nat}
        image_join(image,U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)),channel_read,Nat.add(row_words,column_read)))))
