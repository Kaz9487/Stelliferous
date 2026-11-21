# A placing of tiled cells. Sources placed in a target at shifts whose shift
# along the lane axis starts a group, each ending the lane axis or filling
# its groups, are placed as cells at the groups of the shift: lane l of a
# target cell below the lane count reads what the placing of the values reads
# at its channel, and past the count +0 or what the cell held before.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ./tensor_grid_pieces.bend as GridLaws
import ../tensor_tasks.bend as Tasks
import ../tensor_layout.bend as Layout
import ./tensor_placements.bend as Placements
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_shape_refinement.bend as ShapeLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as Ceiling
import ./binary_radix.bend as Radix
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ../tensor_tiled.bend as Tiled
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ../machine_limits.bend as Machine
import ./tensor_placing_gather.bend as Gathered
import ./tensor_target_cells.bend as Targets
import ./tensor_relane_values.bend as Values
import ../kernel_gemm.bend as KernelGemm

# ---- A source placed at shifts ----

# Along one axis the digit lies from the shift to the shift and the extent.
def axis_inside(+extent: U32,+shift: U32,+digit: Nat) -> Bool:
  Nat.is_le(U32.to_nat(shift),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent)))

# The digits lie inside a source placed at shifts along every axis.
def inside(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>) -> Bool:
  match source shifts digits:
    case Con{+extent,source} Con{+shift,shifts} Con{+digit,digits}: axis_inside(extent,shift,digit) && inside(source,shifts,digits)
    case source shifts digits: True{}

# The digits of the source a target digit reads.
def shifted(shifts: List<&2,U32>,digits: List<&2,Nat>) -> List<&2,Nat>:
  match shifts digits:
    case Con{+shift,shifts} Con{+digit,digits}: Nat.sub(digit,U32.to_nat(shift)) <> shifted(shifts,digits)
    case shifts digits: []

# Every shift and extent of a source within a word bound, as many shifts as
# extents.
def Room(source: List<&2,U32>,shifts: List<&2,U32>,+cap: U32) -> Data:
  match source shifts:
    case Con{+extent,source} Con{+shift,shifts}:
      KernelGemm.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},Room(source,shifts,cap)>
    case Nil{} Nil{}: Unit
    case source shifts: Empty

# ---- The segments of a placed source at words ----

# What the segments of a placed source compute at words: whether every word
# lies inside the source, and the offset of the words in the source.
def inside_words(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>) -> Bool:
  match source shifts words:
    case Con{+extent,source} Con{+shift,shifts} Con{+word,words}:
      (U32.is_le(shift,word) && U32.is_lt(word,(shift + extent : U32))) && inside_words(source,shifts,words)
    case source shifts words: True{}

def offset(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>) -> U32:
  match source shifts words:
    case Con{extent,+source} Con{+shift,shifts} Con{+word,words}:
      (((0 + (word - shift) * 1 : U32) * Grid.product(source) : U32) + offset(source,shifts,words) : U32)
    case source shifts words: 0

def joined(here: Bool,rest: Bool,+found: U32,+stride: U32,+offset: U32) ->
  {Placements.combined(Placements.strided(Bool.pick(Maybe<&2,U32>,here,Some{found},None{}),stride),Bool.pick(Maybe<&2,U32>,rest,Some{offset},None{})) ==
    Bool.pick(Maybe<&2,U32>,here && rest,Some{((found * stride : U32) + offset : U32)},None{}) : Maybe<&2,U32>}:
  match here rest:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# The segments of a source placed at shifts find words inside it at their
# offset.
def segments_words(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>) ->
  {Placements.segments_cell(Layout.placed_segments(source,shifts),words,Tasks.strides_of(source)) ==
    Bool.pick(Maybe<&2,U32>,inside_words(source,shifts,words),Some{offset(source,shifts,words)},None{}) : Maybe<&2,U32>}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      +here = U32.is_le(shift,word) && U32.is_lt(word,(shift + extent : U32))
      +found = (0 + (word - shift) * 1 : U32)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {Placements.segments_cell(Layout.placed_segments(Con{extent,rest},Con{shift,later}),Con{word,more},_) ==
          Bool.pick(Maybe<&2,U32>,inside_words(Con{extent,rest},Con{shift,later},Con{word,more}),Some{offset(Con{extent,rest},Con{shift,later},Con{word,more})},None{}) : Maybe<&2,U32>}
      %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(rest,later),more,Tasks.strides_of(rest)),
          Bool.pick(Maybe<&2,U32>,inside_words(rest,later,more),Some{offset(rest,later,more)},None{}),segments_words(rest,later,more)) :
        {Placements.combined(Placements.strided(Bool.pick(Maybe<&2,U32>,here,Some{found},None{}),Grid.product(rest)),_) ==
          Bool.pick(Maybe<&2,U32>,here && inside_words(rest,later,more),Some{((found * Grid.product(rest) : U32) + offset(rest,later,more) : U32)},None{}) : Maybe<&2,U32>}
      joined(here,inside_words(rest,later,more),found,Grid.product(rest),offset(rest,later,more))
    case Nil{} +shifts +words: {==}
    case Con{+extent,+rest} Nil{} +words: {==}
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}

# ---- The words as numbers ----

def axis_word(+extent: U32,+shift: U32,+word: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_le(shift,word) && U32.is_lt(word,(shift + extent : U32)) == axis_inside(extent,shift,U32.to_nat(word)) : Bool}:
  %Equal.sym(Bool,U32.is_le(shift,word),Nat.is_le(U32.to_nat(shift),U32.to_nat(word)),Comparison.le(shift,word)) :
    {_ && U32.is_lt(word,(shift + extent : U32)) == axis_inside(extent,shift,U32.to_nat(word)) : Bool}
  %Equal.sym(Bool,U32.is_lt(word,(shift + extent : U32)),Nat.is_lt(U32.to_nat(word),U32.to_nat((shift + extent : U32))),Comparison.lt(word,(shift + extent : U32))) :
    {Nat.is_le(U32.to_nat(shift),U32.to_nat(word)) && _ == axis_inside(extent,shift,U32.to_nat(word)) : Bool}
  %Equal.sym(Nat,U32.to_nat((shift + extent : U32)),Nat.add(U32.to_nat(shift),U32.to_nat(extent)),Arithmetic.add(shift,extent,cap,cap_ok,room)) :
    {Nat.is_le(U32.to_nat(shift),U32.to_nat(word)) && Nat.is_lt(U32.to_nat(word),_) == axis_inside(extent,shift,U32.to_nat(word)) : Bool}
  {==}

# Words lie inside a source exactly when their numbers do.
def inside_nat(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Room(source,shifts,cap)) -> {inside_words(source,shifts,words) == inside(source,shifts,Digits.words_nat(words)) : Bool}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      KernelGemm.Both{+fit,+others} = room
      %Equal.sym(Bool,U32.is_le(shift,word) && U32.is_lt(word,(shift + extent : U32)),axis_inside(extent,shift,U32.to_nat(word)),axis_word(extent,shift,word,cap,cap_ok,fit)) :
        {_ && inside_words(rest,later,more) == axis_inside(extent,shift,U32.to_nat(word)) && inside(rest,later,Digits.words_nat(more)) : Bool}
      Equal.cong(Bool,Bool,value => axis_inside(extent,shift,U32.to_nat(word)) && value,inside_words(rest,later,more),inside(rest,later,Digits.words_nat(more)),
        inside_nat(rest,later,more,cap,cap_ok,others))
    case Nil{} +shifts +words: {==}
    case Con{+extent,+rest} Nil{} +words: {==}
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: {==}

# A digit inside an axis lies from the shift, before the shift and the
# extent, and below the extent once shifted.
def axis_lower(+extent: U32,+shift: U32,+digit: Nat,+inside_ok: {axis_inside(extent,shift,digit) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(shift),digit) == True{} : Bool}:
  Boolean.left(Nat.is_le(U32.to_nat(shift),digit),Nat.is_lt(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent))),inside_ok)

def axis_upper(+extent: U32,+shift: U32,+digit: Nat,+inside_ok: {axis_inside(extent,shift,digit) == True{} : Bool}) -> {Nat.is_lt(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent))) == True{} : Bool}:
  Boolean.right(Nat.is_le(U32.to_nat(shift),digit),Nat.is_lt(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent))),inside_ok)

def axis_small(+extent: U32,+shift: U32,+digit: Nat,+inside_ok: {axis_inside(extent,shift,digit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(digit,U32.to_nat(shift)),U32.to_nat(extent)) == True{} : Bool}:
  Order.sub_lt(digit,U32.to_nat(shift),U32.to_nat(extent),axis_lower(extent,shift,digit,inside_ok),axis_upper(extent,shift,digit,inside_ok))


# Digits inside a source of their rank lie below it once shifted.
def below_shifted(source: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+cap: U32,+room: Room(source,shifts,cap),
  +same: {List.length(&2,Nat,digits) == List.length(&2,U32,source) : Nat},+inside_ok: {inside(source,shifts,digits) == True{} : Bool}) ->
  Coordinates.DigitsBelow(source,shifted(shifts,digits)):
  match source shifts digits:
    case Con{+extent,+rest} Con{+shift,+later} Con{+digit,+more}:
      KernelGemm.Both{+fit,+others} = room
      +here = axis_inside(extent,shift,digit)
      KernelGemm.Both{axis_small(extent,shift,digit,Boolean.left(here,inside(rest,later,more),inside_ok)),
        below_shifted(rest,later,more,cap,others,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,U32,rest),same),
          Boolean.right(here,inside(rest,later,more),inside_ok))}
    case Nil{} Nil{} Nil{}: Unit{}
    case Nil{} Nil{} Con{+digit,+more}: Empty.absurd(Coordinates.DigitsBelow(Nil{},shifted(Nil{},Con{digit,more})),Digits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))
    case Nil{} Con{+shift,+later} +digits: Empty.absurd(Coordinates.DigitsBelow(Nil{},shifted(Con{shift,later},digits)),room)
    case Con{+extent,+rest} Nil{} +digits: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},shifted(Nil{},digits)),room)
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},shifted(Con{shift,later},Nil{})),Digits.zero_successor(List.length(&2,U32,rest),same))

def words_length(words: List<&2,U32>) -> {List.length(&2,Nat,Digits.words_nat(words)) == List.length(&2,U32,words) : Nat}:
  match words:
    case Nil{}: {==}
    case Con{+word,+more}: Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Digits.words_nat(more)),List.length(&2,U32,more),words_length(more))

# The offset of words inside a source is the number of their shifted digits.
def offset_nat(source: List<&2,U32>,shifts: List<&2,U32>,words: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Room(source,shifts,cap),+same: {List.length(&2,U32,words) == List.length(&2,U32,source) : Nat},
  +fits: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(cap)) == True{} : Bool},+inside_ok: {inside(source,shifts,Digits.words_nat(words)) == True{} : Bool}) ->
  {U32.to_nat(offset(source,shifts,words)) == ShapeLaws.digits_value(source,shifted(shifts,Digits.words_nat(words))) : Nat}:
  match source shifts words:
    case Con{+extent,+rest} Con{+shift,+later} Con{+word,+more}:
      KernelGemm.Both{+fit,+others} = room
      +digit = U32.to_nat(word)
      +here = axis_inside(extent,shift,digit)
      +here_ok = Boolean.left(here,inside(rest,later,Digits.words_nat(more)),inside_ok)
      +rest_ok = Boolean.right(here,inside(rest,later,Digits.words_nat(more)),inside_ok)
      +lower = axis_lower(extent,shift,digit,here_ok)
      +upper = axis_upper(extent,shift,digit,here_ok)
      +small = axis_small(extent,shift,digit,here_ok)
      +inner = ShapeLaws.elements(rest)
      +local = Nat.sub(digit,U32.to_nat(shift))
      +extent_positive = Order.positive_bound(local,U32.to_nat(extent),small)
      +inner_room = Order.transitive(inner,Nat.mul(U32.to_nat(extent),inner),U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),fits)
      +rest_same = Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,more),1n+List.length(&2,U32,rest),same)
      +rest_exact = offset_nat(rest,later,more,cap,cap_ok,others,rest_same,inner_room,rest_ok)
      +rest_below = below_shifted(rest,later,Digits.words_nat(more),cap,others,
        Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(more)),List.length(&2,U32,more),List.length(&2,U32,rest),words_length(more),rest_same),rest_ok)
      +tail = ShapeLaws.digits_value(rest,shifted(later,Digits.words_nat(more)))
      +tail_small = Coordinates.digits_below(rest,shifted(later,Digits.words_nat(more)),rest_below)
      +whole = RowMajor.radix(local,U32.to_nat(extent),tail,inner,small,tail_small)
      +whole_room = Bounds.strict_le(Nat.add(Nat.mul(local,inner),tail),Nat.mul(U32.to_nat(extent),inner),whole)
      +sum_room = Order.transitive(Nat.add(Nat.mul(local,inner),tail),Nat.mul(U32.to_nat(extent),inner),U32.to_nat(cap),whole_room,fits)
      +product_room = Order.transitive(Nat.mul(local,inner),Nat.add(Nat.mul(local,inner),tail),U32.to_nat(cap),Order.le_add(Nat.mul(local,inner),tail),sum_room)
      +word_room = Order.transitive(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent)),U32.to_nat(cap),Bounds.strict_le(digit,Nat.add(U32.to_nat(shift),U32.to_nat(extent)),upper),fit)
      +difference = Arithmetic.difference_exact(cap,cap_ok,word,shift,digit,U32.to_nat(shift),{==},{==},lower,word_room)
      +found_same = {%Equal.sym(U32,((word - shift : U32) * 1 : U32),(word - shift : U32),Arithmetic.multiply_one((word - shift : U32))) : {(0 + _ : U32) == (word - shift : U32) : U32}
        Arithmetic.add_zero_left((word - shift : U32)) : {(0 + (word - shift) * 1 : U32) == (word - shift : U32) : U32}}
      +stride = Equal.trans(Nat,U32.to_nat(Grid.product(rest)),U32.to_nat(U32.from_nat(inner)),inner,
        Equal.cong(U32,Nat,U32.to_nat,Grid.product(rest),U32.from_nat(inner),Equal.sym(U32,U32.from_nat(inner),Grid.product(rest),Digits.product_from(rest))),
        Bounds.roundtrip(inner,cap,cap_ok,inner_room))
      +found_exact = {%Equal.sym(U32,(0 + (word - shift) * 1 : U32),(word - shift : U32),found_same) : {U32.to_nat(_) == local : Nat} difference :
        {U32.to_nat((0 + (word - shift) * 1 : U32)) == local : Nat}}
      +product = Arithmetic.product_exact(cap,cap_ok,(0 + (word - shift) * 1 : U32),Grid.product(rest),local,inner,found_exact,stride,product_room)
      Arithmetic.sum_exact(cap,cap_ok,((0 + (word - shift) * 1 : U32) * Grid.product(rest) : U32),offset(rest,later,more),Nat.mul(local,inner),tail,product,rest_exact,sum_room)
    case Nil{} Nil{} +words: {==}
    case Nil{} Con{+shift,+later} +words: Empty.absurd({U32.to_nat(offset(Nil{},Con{shift,later},words)) == ShapeLaws.digits_value(Nil{},shifted(Con{shift,later},Digits.words_nat(words))) : Nat},room)
    case Con{+extent,+rest} Nil{} +words: Empty.absurd({U32.to_nat(offset(Con{extent,rest},Nil{},words)) == ShapeLaws.digits_value(Con{extent,rest},shifted(Nil{},Digits.words_nat(words))) : Nat},room)
    case Con{+extent,+rest} Con{+shift,+later} Nil{}: Empty.absurd({U32.to_nat(offset(Con{extent,rest},Con{shift,later},Nil{})) == ShapeLaws.digits_value(Con{extent,rest},shifted(Con{shift,later},Digits.words_nat(Nil{}))) : Nat},Digits.zero_successor(List.length(&2,U32,rest),same))

# ---- The lane axis ----

# How a source ends along the lane axis: it fills its groups, or it ends the
# axis of the target.
type Ending<-extent: U32,-shift: U32,-total: U32> is Data:
  Whole{aligned: {U32.is_zero((extent % 8 : U32)) == True{} : Bool}}
  Last{ends: {Nat.add(U32.to_nat(shift),U32.to_nat(extent)) == U32.to_nat(total) : Nat}}

def lane_below(+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) -> {Nat.is_lt(lane,8n) == True{} : Bool}:
  Order.lt_succ(lane,7n,small)

# A channel of a group lies before the next group.
def channel_next(+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Nat.mul(1n+group,8n)) == True{} : Bool}:
  Order.add_lt(lane,8n,Nat.mul(group,8n),lane_below(lane,small))

# A group lies from a group exactly when its channels do.
def lower_lane(+first: Nat,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {Nat.is_le(first,group) == Nat.is_le(Nat.mul(first,8n),Nat.add(lane,Nat.mul(group,8n))) : Bool}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Boolean.equivalent(Nat.is_le(first,group),Nat.is_le(Nat.mul(first,8n),channel),
    below => Order.transitive(Nat.mul(first,8n),Nat.mul(group,8n),channel,RowMajor.mul_le(first,group,8n,below),Order.right_le(lane,Nat.mul(group,8n))),
    covered => Division.lt_succ_le(first,group,RowMajor.cancel_multiply_lt(first,1n+group,8n,
      Order.le_lt_trans(Nat.mul(first,8n),channel,Nat.mul(1n+group,8n),covered,channel_next(group,lane,small)))))

# Below a multiple of eight a group lies exactly when its channels do.
def whole_lane(+groups: Nat,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(group,8n),Nat.mul(groups,8n)) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Nat.mul(groups,8n)) : Bool}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Boolean.equivalent(Nat.is_lt(Nat.mul(group,8n),Nat.mul(groups,8n)),Nat.is_lt(channel,Nat.mul(groups,8n)),
    below => Bounds.nat_trans(channel,Nat.mul(1n+group,8n),Nat.mul(groups,8n),channel_next(group,lane,small),
      RowMajor.mul_le(1n+group,groups,8n,Order.strict_follow(group,groups,RowMajor.cancel_multiply_lt(group,groups,8n,below)))),
    covered => Order.le_lt_trans(Nat.mul(group,8n),channel,Nat.mul(groups,8n),Order.right_le(lane,Nat.mul(group,8n)),covered))

# Before the end of the target both lie below it.
def last_lane(+end: Nat,+group: Nat,+lane: Nat,+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),end) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(group,8n),end) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),end) : Bool}:
  %Equal.sym(Bool,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),end),True{},inside) : {Nat.is_lt(Nat.mul(group,8n),end) == _ : Bool}
  Order.le_lt_trans(Nat.mul(group,8n),Nat.add(lane,Nat.mul(group,8n)),end,Order.right_le(lane,Nat.mul(group,8n)),inside)

def ending_lane(+extent: U32,+shift: U32,+total: U32,+first: Nat,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +ending: Ending<extent,shift,total>,+shift_exact: {Nat.mul(first,8n) == U32.to_nat(shift) : Nat},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(group,8n),Nat.add(Nat.mul(first,8n),U32.to_nat(extent))) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Nat.add(Nat.mul(first,8n),U32.to_nat(extent))) : Bool}:
  match ending:
    case Whole{+aligned}:
      +count = U32.to_nat((extent / 8 : U32))
      +multiple = Equal.trans(Nat,Nat.add(Nat.mul(first,8n),U32.to_nat(extent)),Nat.add(Nat.mul(first,8n),Nat.mul(count,8n)),Nat.mul(Nat.add(first,count),8n),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,8n),value),U32.to_nat(extent),Nat.mul(count,8n),
          Equal.sym(Nat,Nat.mul(count,8n),U32.to_nat(extent),Radix.divide_eight_exact_multiple(extent,aligned))),
        Equal.sym(Nat,Nat.mul(Nat.add(first,count),8n),Nat.add(Nat.mul(first,8n),Nat.mul(count,8n)),RowMajor.distrib(first,count,8n)))
      %Equal.sym(Nat,Nat.add(Nat.mul(first,8n),U32.to_nat(extent)),Nat.mul(Nat.add(first,count),8n),multiple) :
        {Nat.is_lt(Nat.mul(group,8n),_) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),_) : Bool}
      whole_lane(Nat.add(first,count),group,lane,small)
    case Last{+ends}:
      %Equal.sym(Nat,Nat.mul(first,8n),U32.to_nat(shift),shift_exact) : {Nat.is_lt(Nat.mul(group,8n),Nat.add(_,U32.to_nat(extent))) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Nat.add(_,U32.to_nat(extent))) : Bool}
      %Equal.sym(Nat,Nat.add(U32.to_nat(shift),U32.to_nat(extent)),U32.to_nat(total),ends) : {Nat.is_lt(Nat.mul(group,8n),_) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),_) : Bool}
      last_lane(U32.to_nat(total),group,lane,inside)

# A group lies inside a source placed at a shift starting a group exactly when
# a channel of it below the lanes of the target does.
def lane_axis(+extent: U32,+shift: U32,+total: U32,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +aligned: {U32.is_zero((shift % 8 : U32)) == True{} : Bool},+ending: Ending<extent,shift,total>,
  +grouped: {U32.to_nat(Grid.groups(extent)) == Ceiling.ceiling(U32.to_nat(extent),8n) : Nat},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == True{} : Bool}) ->
  {axis_inside(Grid.groups(extent),(shift / 8 : U32),group) == axis_inside(extent,shift,Nat.add(lane,Nat.mul(group,8n))) : Bool}:
  +first = U32.to_nat((shift / 8 : U32))
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +shift_exact = Radix.divide_eight_exact_multiple(shift,aligned)
  %Equal.sym(Nat,U32.to_nat(Grid.groups(extent)),Ceiling.ceiling(U32.to_nat(extent),8n),grouped) :
    {Nat.is_le(first,group) && Nat.is_lt(group,Nat.add(first,_)) == axis_inside(extent,shift,channel) : Bool}
  %shift_exact : {Nat.is_le(first,group) && Nat.is_lt(group,Nat.add(first,Ceiling.ceiling(U32.to_nat(extent),8n))) == Nat.is_le(_,channel) && Nat.is_lt(channel,Nat.add(_,U32.to_nat(extent))) : Bool}
  %Equal.sym(Bool,Nat.is_le(first,group),Nat.is_le(Nat.mul(first,8n),channel),lower_lane(first,group,lane,small)) :
    {_ && Nat.is_lt(group,Nat.add(first,Ceiling.ceiling(U32.to_nat(extent),8n))) == Nat.is_le(Nat.mul(first,8n),channel) && Nat.is_lt(channel,Nat.add(Nat.mul(first,8n),U32.to_nat(extent))) : Bool}
  %Equal.sym(Nat,Nat.add(first,Ceiling.ceiling(U32.to_nat(extent),8n)),Ceiling.ceiling(Nat.add(Nat.mul(first,8n),U32.to_nat(extent)),8n),Ceiling.ceiling_after(first,U32.to_nat(extent))) :
    {Nat.is_le(Nat.mul(first,8n),channel) && Nat.is_lt(group,_) == Nat.is_le(Nat.mul(first,8n),channel) && Nat.is_lt(channel,Nat.add(Nat.mul(first,8n),U32.to_nat(extent))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(Nat.add(Nat.mul(first,8n),U32.to_nat(extent)),8n)),Nat.is_lt(Nat.mul(group,8n),Nat.add(Nat.mul(first,8n),U32.to_nat(extent))),
      Ceiling.upper_comparison(Nat.add(Nat.mul(first,8n),U32.to_nat(extent)),8n,group,{==})) :
    {Nat.is_le(Nat.mul(first,8n),channel) && _ == Nat.is_le(Nat.mul(first,8n),channel) && Nat.is_lt(channel,Nat.add(Nat.mul(first,8n),U32.to_nat(extent))) : Bool}
  Equal.cong(Bool,Bool,value => Nat.is_le(Nat.mul(first,8n),channel) && value,Nat.is_lt(Nat.mul(group,8n),Nat.add(Nat.mul(first,8n),U32.to_nat(extent))),
    Nat.is_lt(channel,Nat.add(Nat.mul(first,8n),U32.to_nat(extent))),
    ending_lane(extent,shift,total,first,group,lane,small,ending,shift_exact,inside))

# ---- Digits around the lane axis ----

def and_true_right(+value: Bool) -> {value && True{} == value : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

def and_commutative(+first: Bool,+second: Bool) -> {first && second == second && first : Bool}:
  match first second:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: {==}

# Digits with one more axis at the end lie inside a source with one more axis
# when the digits do and the last digit does.
def inside_appended(extents: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+extent: U32,+shift: U32,+digit: Nat,
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,extents) : Nat},+digits_same: {List.length(&2,Nat,digits) == List.length(&2,U32,extents) : Nat}) ->
  {inside(List.append(&2,U32,extents,[extent]),List.append(&2,U32,shifts,[shift]),List.append(&2,Nat,digits,[digit])) == inside(extents,shifts,digits) && axis_inside(extent,shift,digit) : Bool}:
  match extents shifts digits:
    case Con{+first,+rest} Con{+low,+lows} Con{+value,+values}:
      +here = axis_inside(first,low,value)
      +last = axis_inside(extent,shift,digit)
      +later = inside_appended(rest,lows,values,extent,shift,digit,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,lows),1n+List.length(&2,U32,rest),shifts_same),
        Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,values),1n+List.length(&2,U32,rest),digits_same))
      %Equal.sym(Bool,inside(List.append(&2,U32,rest,[extent]),List.append(&2,U32,lows,[shift]),List.append(&2,Nat,values,[digit])),inside(rest,lows,values) && last,later) :
        {here && _ == (here && inside(rest,lows,values)) && last : Bool}
      Equal.sym(Bool,(here && inside(rest,lows,values)) && last,here && (inside(rest,lows,values) && last),Boolean.and_associative(here,inside(rest,lows,values),last))
    case Nil{} Nil{} Nil{}: and_true_right(axis_inside(extent,shift,digit))
    case Nil{} Nil{} Con{+value,+values}: Empty.absurd({inside(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Nil{},[shift]),List.append(&2,Nat,Con{value,values},[digit])) == inside(Nil{},Nil{},Con{value,values}) && axis_inside(extent,shift,digit) : Bool},Digits.zero_successor(List.length(&2,Nat,values),Equal.sym(Nat,1n+List.length(&2,Nat,values),0n,digits_same)))
    case Nil{} Con{+low,+lows} +digits: Empty.absurd({inside(List.append(&2,U32,Nil{},[extent]),List.append(&2,U32,Con{low,lows},[shift]),List.append(&2,Nat,digits,[digit])) == inside(Nil{},Con{low,lows},digits) && axis_inside(extent,shift,digit) : Bool},Digits.zero_successor(List.length(&2,U32,lows),Equal.sym(Nat,1n+List.length(&2,U32,lows),0n,shifts_same)))
    case Con{+first,+rest} Nil{} +digits: Empty.absurd({inside(List.append(&2,U32,Con{first,rest},[extent]),List.append(&2,U32,Nil{},[shift]),List.append(&2,Nat,digits,[digit])) == inside(Con{first,rest},Nil{},digits) && axis_inside(extent,shift,digit) : Bool},Digits.zero_successor(List.length(&2,U32,rest),shifts_same))
    case Con{+first,+rest} Con{+low,+lows} Nil{}: Empty.absurd({inside(List.append(&2,U32,Con{first,rest},[extent]),List.append(&2,U32,Con{low,lows},[shift]),List.append(&2,Nat,Nil{},[digit])) == inside(Con{first,rest},Con{low,lows},Nil{}) && axis_inside(extent,shift,digit) : Bool},Digits.zero_successor(List.length(&2,U32,rest),digits_same))

# Digits with a channel put back at the lane axis lie inside a source when the
# other digits lie inside its other axes and the channel inside its lane axis.
def inside_inserted(extents: List<&2,U32>,shifts: List<&2,U32>,lane: Nat,others: List<&2,Nat>,+channel: Nat,
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,extents) : Nat},+others_same: {1n+List.length(&2,Nat,others) == List.length(&2,U32,extents) : Nat},
  +within: {Nat.is_lt(lane,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {inside(extents,shifts,Coordinates.inserted_digit(lane,others,channel)) == inside(Grid.removed(extents,lane),Grid.removed(shifts,lane),others) && axis_inside(Grid.lanes_at(extents,lane),Grid.lanes_at(shifts,lane),channel) : Bool}:
  match extents shifts lane others:
    case Nil{} +shifts +lane +others: Empty.absurd({inside(Nil{},shifts,Coordinates.inserted_digit(lane,others,channel)) == inside(Grid.removed(Nil{},lane),Grid.removed(shifts,lane),others) && axis_inside(Grid.lanes_at(Nil{},lane),Grid.lanes_at(shifts,lane),channel) : Bool},Order.below_zero(lane,within))
    case Con{+first,+rest} Nil{} +lane +others: Empty.absurd({inside(Con{first,rest},Nil{},Coordinates.inserted_digit(lane,others,channel)) == inside(Grid.removed(Con{first,rest},lane),Grid.removed(Nil{},lane),others) && axis_inside(Grid.lanes_at(Con{first,rest},lane),Grid.lanes_at(Nil{},lane),channel) : Bool},Digits.zero_successor(List.length(&2,U32,rest),shifts_same))
    case Con{+first,+rest} Con{+low,+lows} 0n +others: and_commutative(axis_inside(first,low,channel),inside(rest,lows,others))
    case Con{+first,+rest} Con{+low,+lows} 1n+ +next Con{+value,+values}:
      +here = axis_inside(first,low,value)
      +last = axis_inside(Grid.lanes_at(rest,next),Grid.lanes_at(lows,next),channel)
      +kept = inside(Grid.removed(rest,next),Grid.removed(lows,next),values)
      +later = inside_inserted(rest,lows,next,values,channel,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,lows),1n+List.length(&2,U32,rest),shifts_same),
        Equal.cong(Nat,Nat,count => Digits.before(count),1n+1n+List.length(&2,Nat,values),1n+List.length(&2,U32,rest),others_same),within)
      %Equal.sym(Bool,inside(rest,lows,Coordinates.inserted_digit(next,values,channel)),kept && last,later) : {here && _ == (here && kept) && last : Bool}
      Equal.sym(Bool,(here && kept) && last,here && (kept && last),Boolean.and_associative(here,kept,last))
    case Con{+first,+rest} Con{+low,+lows} 1n+ +next Nil{}:
      +empty = Equal.sym(Nat,0n,List.length(&2,U32,rest),Equal.cong(Nat,Nat,count => Digits.before(count),1n,1n+List.length(&2,U32,rest),others_same))
      Empty.absurd({inside(Con{first,rest},Con{low,lows},Coordinates.inserted_digit(1n+next,Nil{},channel)) == inside(Grid.removed(Con{first,rest},1n+next),Grid.removed(Con{low,lows},1n+next),Nil{}) && axis_inside(Grid.lanes_at(Con{first,rest},1n+next),Grid.lanes_at(Con{low,lows},1n+next),channel) : Bool},Order.below_zero(next,%empty : {Nat.is_lt(next,_) == True{} : Bool} within))

# Inside the cells of a source exactly when inside its other axes and, when
# its lanes lie inside it, inside the groups.
def placed_inside(within: Bool,rest: Bool,group: Bool) -> Bool:
  match within:
    case True{}: rest && group
    case False{}: rest

def inside_placed(within: Bool,position: Grid.Position,extents: List<&2,U32>,shifts: List<&2,U32>,digits: List<&2,Nat>,+extent: U32,+shift: U32,+digit: Nat,
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,extents) : Nat},+digits_same: {List.length(&2,Nat,digits) == List.length(&2,U32,extents) : Nat}) ->
  {inside(Grid.placed(within,position,extents,extent),Grid.placed(within,position,shifts,shift),Coordinates.placed_digits(within,position,digits,digit)) ==
    placed_inside(within,inside(extents,shifts,digits),axis_inside(extent,shift,digit)) : Bool}:
  match within position:
    case False{} +position: {==}
    case True{} Grid.Innermost{}: inside_appended(extents,shifts,digits,extent,shift,digit,shifts_same,digits_same)
    case True{} Grid.Outermost{}: and_commutative(axis_inside(extent,shift,digit),inside(extents,shifts,digits))

# Shifted digits with one more axis at the end.
def shifted_appended(shifts: List<&2,U32>,digits: List<&2,Nat>,+shift: U32,+digit: Nat,+same: {List.length(&2,Nat,digits) == List.length(&2,U32,shifts) : Nat}) ->
  {shifted(List.append(&2,U32,shifts,[shift]),List.append(&2,Nat,digits,[digit])) == List.append(&2,Nat,shifted(shifts,digits),[Nat.sub(digit,U32.to_nat(shift))]) : List<&2,Nat>}:
  match shifts digits:
    case Con{+low,+lows} Con{+value,+values}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,rest => Nat.sub(value,U32.to_nat(low)) <> rest,
        shifted(List.append(&2,U32,lows,[shift]),List.append(&2,Nat,values,[digit])),List.append(&2,Nat,shifted(lows,values),[Nat.sub(digit,U32.to_nat(shift))]),
        shifted_appended(lows,values,shift,digit,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,values),1n+List.length(&2,U32,lows),same)))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+value,+values}: Empty.absurd({shifted(List.append(&2,U32,Nil{},[shift]),List.append(&2,Nat,Con{value,values},[digit])) == List.append(&2,Nat,shifted(Nil{},Con{value,values}),[Nat.sub(digit,U32.to_nat(shift))]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,values),Equal.sym(Nat,1n+List.length(&2,Nat,values),0n,same)))
    case Con{+low,+lows} Nil{}: Empty.absurd({shifted(List.append(&2,U32,Con{low,lows},[shift]),List.append(&2,Nat,Nil{},[digit])) == List.append(&2,Nat,shifted(Con{low,lows},Nil{}),[Nat.sub(digit,U32.to_nat(shift))]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,U32,lows),same))

# Shifted digits with a channel put back at the lane axis: the other digits
# shifted by the other shifts and the channel by the lane shift.
def shifted_inserted(shifts: List<&2,U32>,lane: Nat,others: List<&2,Nat>,+channel: Nat,
  +same: {1n+List.length(&2,Nat,others) == List.length(&2,U32,shifts) : Nat},+within: {Nat.is_lt(lane,List.length(&2,U32,shifts)) == True{} : Bool}) ->
  {shifted(shifts,Coordinates.inserted_digit(lane,others,channel)) == Coordinates.inserted_digit(lane,shifted(Grid.removed(shifts,lane),others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,lane)))) : List<&2,Nat>}:
  match shifts lane others:
    case Nil{} +lane +others: Empty.absurd({shifted(Nil{},Coordinates.inserted_digit(lane,others,channel)) == Coordinates.inserted_digit(lane,shifted(Grid.removed(Nil{},lane),others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(Nil{},lane)))) : List<&2,Nat>},Order.below_zero(lane,within))
    case Con{+low,+lows} 0n +others: {==}
    case Con{+low,+lows} 1n+ +next Con{+value,+values}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,rest => Nat.sub(value,U32.to_nat(low)) <> rest,
        shifted(lows,Coordinates.inserted_digit(next,values,channel)),
        Coordinates.inserted_digit(next,shifted(Grid.removed(lows,next),values),Nat.sub(channel,U32.to_nat(Grid.lanes_at(lows,next)))),
        shifted_inserted(lows,next,values,channel,Equal.cong(Nat,Nat,count => Digits.before(count),1n+1n+List.length(&2,Nat,values),1n+List.length(&2,U32,lows),same),within))
    case Con{+low,+lows} 1n+ +next Nil{}:
      +empty = Equal.sym(Nat,0n,List.length(&2,U32,lows),Equal.cong(Nat,Nat,count => Digits.before(count),1n,1n+List.length(&2,U32,lows),same))
      Empty.absurd({shifted(Con{low,lows},Coordinates.inserted_digit(1n+next,Nil{},channel)) == Coordinates.inserted_digit(1n+next,shifted(Grid.removed(Con{low,lows},1n+next),Nil{}),Nat.sub(channel,U32.to_nat(Grid.lanes_at(Con{low,lows},1n+next)))) : List<&2,Nat>},Order.below_zero(next,%empty : {Nat.is_lt(next,_) == True{} : Bool} within))

# Shifted digits of a cell: the other digits shifted and the group shifted by
# the groups of the shift.
def shifted_placed(within: Bool,position: Grid.Position,shifts: List<&2,U32>,digits: List<&2,Nat>,+shift: U32,+group: Nat,
  +same: {List.length(&2,Nat,digits) == List.length(&2,U32,shifts) : Nat}) ->
  {shifted(Grid.placed(within,position,shifts,shift),Coordinates.placed_digits(within,position,digits,group)) ==
    Coordinates.placed_digits(within,position,shifted(shifts,digits),Nat.sub(group,U32.to_nat(shift))) : List<&2,Nat>}:
  match within position:
    case False{} +position: {==}
    case True{} Grid.Innermost{}: shifted_appended(shifts,digits,shift,group,same)
    case True{} Grid.Outermost{}: {==}

# ---- The channel of a group of a placed source ----

def swap_front(+a: Nat,+b: Nat,+c: Nat) -> {Nat.add(a,Nat.add(b,c)) == Nat.add(b,Nat.add(a,c)) : Nat}:
  Equal.trans(Nat,Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(a,b),c),Nat.add(b,Nat.add(a,c)),
    Equal.sym(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Addition.associative(a,b,c)),
    Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(Nat.add(b,a),c),Nat.add(b,Nat.add(a,c)),
      Equal.cong(Nat,Nat,value => Nat.add(value,c),Nat.add(a,b),Nat.add(b,a),Order.commutative(a,b)),
      Addition.associative(b,a,c)))
# The channel of a lane of a group from the groups of the shift on is the
# channel past the shift.
def channel_local(+first: Nat,+group: Nat,+lane: Nat,+lower: {Nat.is_le(first,group) == True{} : Bool}) ->
  {Nat.add(lane,Nat.mul(Nat.sub(group,first),8n)) == Nat.sub(Nat.add(lane,Nat.mul(group,8n)),Nat.mul(first,8n)) : Nat}:
  +local = Nat.sub(group,first)
  +split = Equal.trans(Nat,Nat.mul(group,8n),Nat.mul(Nat.add(first,local),8n),Nat.add(Nat.mul(first,8n),Nat.mul(local,8n)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,8n),group,Nat.add(first,local),Equal.sym(Nat,Nat.add(first,local),group,Order.restore_shift(first,group,lower))),
    RowMajor.distrib(first,local,8n))
  +moved = Equal.trans(Nat,Nat.add(lane,Nat.mul(group,8n)),Nat.add(lane,Nat.add(Nat.mul(first,8n),Nat.mul(local,8n))),Nat.add(Nat.mul(first,8n),Nat.add(lane,Nat.mul(local,8n))),
    Equal.cong(Nat,Nat,value => Nat.add(lane,value),Nat.mul(group,8n),Nat.add(Nat.mul(first,8n),Nat.mul(local,8n)),split),
    swap_front(lane,Nat.mul(first,8n),Nat.mul(local,8n)))
  %Equal.sym(Nat,Nat.add(lane,Nat.mul(group,8n)),Nat.add(Nat.mul(first,8n),Nat.add(lane,Nat.mul(local,8n))),moved) : {Nat.add(lane,Nat.mul(local,8n)) == Nat.sub(_,Nat.mul(first,8n)) : Nat}
  Equal.sym(Nat,Nat.sub(Nat.add(Nat.mul(first,8n),Nat.add(lane,Nat.mul(local,8n))),Nat.mul(first,8n)),Nat.add(lane,Nat.mul(local,8n)),
    Order.subtract_shift(Nat.mul(first,8n),Nat.add(lane,Nat.mul(local,8n))))

# ---- One source at one lane of a cell ----

# The shifts of the cells of a source: the other shifts and the groups of the
# lane shift at their position.
def cell_shifts(+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> List<&2,U32>:
  Grid.placed(Nat.is_lt(axis,List.length(&2,U32,shifts)),position,Grid.removed(shifts,axis),(Grid.lanes_at(shifts,axis) / 8 : U32))

# Rooms of a source and its shifts without one axis.
def room_removed(source: List<&2,U32>,shifts: List<&2,U32>,axis: Nat,+cap: U32,+room: Room(source,shifts,cap)) -> Room(Grid.removed(source,axis),Grid.removed(shifts,axis),cap):
  match source shifts axis:
    case Con{+extent,+rest} Con{+shift,+later} 0n:
      KernelGemm.Both{+fit,+others} = room
      others
    case Con{+extent,+rest} Con{+shift,+later} 1n+ +next:
      KernelGemm.Both{+fit,+others} = room
      KernelGemm.Both{fit,room_removed(rest,later,next,cap,others)}
    case Nil{} Nil{} +axis: Unit{}
    case Nil{} Con{+shift,+later} +axis: Empty.absurd(Room(Grid.removed(Nil{},axis),Grid.removed(Con{shift,later},axis),cap),room)
    case Con{+extent,+rest} Nil{} +axis: Empty.absurd(Room(Grid.removed(Con{extent,rest},axis),Grid.removed(Nil{},axis),cap),room)

# What a source placed in cells with its lanes inside it needs: its rank, its
# rooms, the lane shift starting a group and the source ending the lane axis
# of the target or filling its groups.
type Lanes<-limits: Machine.Limits,-cap: U32,-total: U32,-rank: Nat,-cell_rank: Nat,-source: List<&2,U32>,-shifts: List<&2,U32>,-axis: Nat,-position: Grid.Position> is Data:
  Lanes{within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
    ranked: {List.length(&2,U32,source) == rank : Nat},
    cell_ranked: {List.length(&2,U32,Tiled.cell_shape(source,axis,position)) == cell_rank : Nat},
    shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat},
    cell_room: Room(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cap),
    value_room: Room(source,shifts,cap),
    cell_fits: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(source,axis,position)),U32.to_nat(cap)) == True{} : Bool},
    value_fits: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(cap)) == True{} : Bool},
    removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(source,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    aligned: {U32.is_zero((Grid.lanes_at(shifts,axis) % 8 : U32)) == True{} : Bool},
    ending: Ending<Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),total>,
    grouped: {U32.to_nat(Grid.groups(Grid.lanes_at(source,axis))) == Ceiling.ceiling(U32.to_nat(Grid.lanes_at(source,axis)),8n) : Nat},
    lane_fits: {Nat.is_le(Nat.add(U32.to_nat(Grid.lanes_at(shifts,axis)),U32.to_nat(Grid.lanes_at(source,axis))),U32.to_nat(total)) == True{} : Bool}}

# A cell of the target at its other digits and its group, a lane of it, and
# the words of the cell.
type Spot<-cap: U32,-rank: Nat,-cell_rank: Nat,-position: Grid.Position,-others: List<&2,Nat>,-group: Nat,-lane: Nat,-cell_words: List<&2,U32>> is Data:
  Spot{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    others_same: {1n+List.length(&2,Nat,others) == rank : Nat},
    cell_digits: {Digits.words_nat(cell_words) == Coordinates.placed_digits(True{},position,others,group) : List<&2,Nat>},
    cell_count: {List.length(&2,U32,cell_words) == cell_rank : Nat},
    small: {Nat.is_le(lane,7n) == True{} : Bool}}

# A lane of the spot below the lanes of the target and the words of its value.
type Read<-total: U32,-rank: Nat,-axis: Nat,-others: List<&2,Nat>,-group: Nat,-lane: Nat,-value_words: List<&2,U32>> is Data:
  Read{value_digits: {Digits.words_nat(value_words) == Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))) : List<&2,Nat>},
    value_count: {List.length(&2,U32,value_words) == rank : Nat},
    in_lane: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == True{} : Bool}}

def pred_same(+left: Nat,+right: Nat,+same: {1n+left == 1n+right : Nat}) -> {left == right : Nat}:
  Equal.cong(Nat,Nat,count => Digits.before(count),1n+left,1n+right,same)

def shifts_within(+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat}) -> {Nat.is_lt(axis,List.length(&2,U32,shifts)) == True{} : Bool}:
  %Equal.sym(Nat,List.length(&2,U32,shifts),List.length(&2,U32,source),shifts_same) : {Nat.is_lt(axis,_) == True{} : Bool}
  within

def others_source(+source: List<&2,U32>,+others: List<&2,Nat>,+rank: Nat,+ranked: {List.length(&2,U32,source) == rank : Nat},
  +others_same: {1n+List.length(&2,Nat,others) == rank : Nat}) -> {1n+List.length(&2,Nat,others) == List.length(&2,U32,source) : Nat}:
  Equal.trans(Nat,1n+List.length(&2,Nat,others),rank,List.length(&2,U32,source),others_same,Equal.sym(Nat,List.length(&2,U32,source),rank,ranked))

def others_removed(+source: List<&2,U32>,+others: List<&2,Nat>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
  +same: {1n+List.length(&2,Nat,others) == List.length(&2,U32,source) : Nat}) -> {List.length(&2,Nat,others) == List.length(&2,U32,Grid.removed(source,axis)) : Nat}:
  pred_same(List.length(&2,Nat,others),List.length(&2,U32,Grid.removed(source,axis)),
    Equal.trans(Nat,1n+List.length(&2,Nat,others),List.length(&2,U32,source),1n+List.length(&2,U32,Grid.removed(source,axis)),same,
      Equal.sym(Nat,1n+List.length(&2,U32,Grid.removed(source,axis)),List.length(&2,U32,source),Public.removed_length(source,axis,within))))

def shifts_removed(+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat}) ->
  {List.length(&2,U32,Grid.removed(shifts,axis)) == List.length(&2,U32,Grid.removed(source,axis)) : Nat}:
  pred_same(List.length(&2,U32,Grid.removed(shifts,axis)),List.length(&2,U32,Grid.removed(source,axis)),
    Equal.trans(Nat,1n+List.length(&2,U32,Grid.removed(shifts,axis)),List.length(&2,U32,shifts),1n+List.length(&2,U32,Grid.removed(source,axis)),
      Public.removed_length(shifts,axis,shifts_within(source,shifts,axis,within,shifts_same)),
      Equal.trans(Nat,List.length(&2,U32,shifts),List.length(&2,U32,source),1n+List.length(&2,U32,Grid.removed(source,axis)),shifts_same,
        Equal.sym(Nat,1n+List.length(&2,U32,Grid.removed(source,axis)),List.length(&2,U32,source),Public.removed_length(source,axis,within)))))

# The source holds the cell of the spot when its other digits lie inside its
# other axes and its group inside the groups of the source.
def cell_split(+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>) ->
  {inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) ==
    inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.groups(Grid.lanes_at(source,axis)),(Grid.lanes_at(shifts,axis) / 8 : U32),group) : Bool}:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +groups = Grid.groups(Grid.lanes_at(source,axis))
  +first = (Grid.lanes_at(shifts,axis) / 8 : U32)
  +cells = Tiled.cell_shape(source,axis,position)
  +moved = cell_shifts(shifts,axis,position)
  %Equal.sym(Bool,inside_words(cells,moved,cell_words),inside(cells,moved,Digits.words_nat(cell_words)),inside_nat(cells,moved,cell_words,cap,cap_ok,cell_room)) :
    {_ == kept && axis_inside(groups,first,group) : Bool}
  %Equal.sym(List<&2,Nat>,Digits.words_nat(cell_words),Coordinates.placed_digits(True{},position,others,group),cell_digits) :
    {inside(cells,moved,_) == kept && axis_inside(groups,first,group) : Bool}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,source)),True{},within) :
    {inside(Grid.placed(_,position,Grid.removed(source,axis),groups),moved,Coordinates.placed_digits(True{},position,others,group)) ==
      kept && axis_inside(groups,first,group) : Bool}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shifts)),True{},shifts_within(source,shifts,axis,within,shifts_same)) :
    {inside(Grid.placed(True{},position,Grid.removed(source,axis),groups),Grid.placed(_,position,Grid.removed(shifts,axis),first),Coordinates.placed_digits(True{},position,others,group)) ==
      kept && axis_inside(groups,first,group) : Bool}
  %Equal.sym(Bool,inside(Grid.placed(True{},position,Grid.removed(source,axis),groups),Grid.placed(True{},position,Grid.removed(shifts,axis),first),Coordinates.placed_digits(True{},position,others,group)),
      placed_inside(True{},kept,axis_inside(groups,first,group)),
      inside_placed(True{},position,Grid.removed(source,axis),Grid.removed(shifts,axis),others,groups,first,group,shifts_removed(source,shifts,axis,within,shifts_same),
        others_removed(source,others,axis,within,others_source(source,others,rank,ranked,others_same)))) :
    {_ == kept && axis_inside(groups,first,group) : Bool}
  {==}

# The source holds the cell of the spot exactly when it holds its value.
def cell_inside(+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,+value_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+read: Read<total,rank,axis,others,group,lane,value_words>) ->
  {inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) ==
    inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Nat.add(lane,Nat.mul(group,8n))) : Bool}:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  Read{+value_digits,+value_count,+in_lane} = read
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +groups = Grid.groups(Grid.lanes_at(source,axis))
  +first = (Grid.lanes_at(shifts,axis) / 8 : U32)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Equal.trans(Bool,inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),kept && axis_inside(groups,first,group),
    kept && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel),
    cell_split(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot),
    Equal.cong(Bool,Bool,value => kept && value,axis_inside(groups,first,group),axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel),
      lane_axis(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),total,group,lane,small,aligned,ending,grouped,in_lane)))

# What the values of a source placed in a target need at a lane axis inside
# it, read at other digits of the rank: the bounds of the words, the axis
# inside, the rank of the source and of its shifts, and the room of the
# values.
type ValueFacts<-cap: U32,-rank: Nat,-source: List<&2,U32>,-shifts: List<&2,U32>,-axis: Nat,-others: List<&2,Nat>> is Data:
  ValueFacts{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
    ranked: {List.length(&2,U32,source) == rank : Nat},
    shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat},
    value_room: Room(source,shifts,cap),
    value_fits: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(cap)) == True{} : Bool},
    others_same: {1n+List.length(&2,Nat,others) == rank : Nat}}

# The value facts of a source placed in cells at a spot.
def value_facts(+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>) ->
  ValueFacts<cap,rank,source,shifts,axis,others>:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  ValueFacts{cap_ok,within,ranked,shifts_same,value_room,value_fits,others_same}

# The source holds the value of the spot when its other digits lie inside its
# other axes and its channel inside its lane axis.
def value_inside(+cap: U32,+total: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,+value_words: List<&2,U32>,
  +values: ValueFacts<cap,rank,source,shifts,axis,others>,+read: Read<total,rank,axis,others,group,lane,value_words>) ->
  {inside_words(source,shifts,value_words) ==
    inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Nat.add(lane,Nat.mul(group,8n))) : Bool}:
  ValueFacts{+cap_ok,+within,+ranked,+shifts_same,+value_room,+value_fits,+others_same} = values
  Read{+value_digits,+value_count,+in_lane} = read
  +channel = Nat.add(lane,Nat.mul(group,8n))
  %Equal.sym(Bool,inside_words(source,shifts,value_words),inside(source,shifts,Digits.words_nat(value_words)),inside_nat(source,shifts,value_words,cap,cap_ok,value_room)) :
    {_ == inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel) : Bool}
  %Equal.sym(List<&2,Nat>,Digits.words_nat(value_words),Coordinates.inserted_digit(axis,others,channel),value_digits) :
    {inside(source,shifts,_) == inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel) : Bool}
  inside_inserted(source,shifts,axis,others,channel,shifts_same,others_source(source,others,rank,ranked,others_same),within)

# The other digits of the spot shifted into the source.
def others_moved(+shifts: List<&2,U32>,+axis: Nat,+others: List<&2,Nat>) -> List<&2,Nat>:
  shifted(Grid.removed(shifts,axis),others)

# The value of the source the spot reads: the other digits and the channel
# shifted into the source.
def source_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+others: List<&2,Nat>,+channel: Nat) -> F32:
  original(logical,U32.from_nat(ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,others_moved(shifts,axis,others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis)))))))

# A lane of the source cell the spot reads holds the source value when the
# channel lies inside the source, and +0 past it.
def cell_pick(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,
  +hit: {inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.groups(Grid.lanes_at(source,axis)),(Grid.lanes_at(shifts,axis) / 8 : U32),group) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words))) ==
    Bool.pick(F32,Nat.is_lt(Nat.sub(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(shifts,axis))),U32.to_nat(Grid.lanes_at(source,axis))),
      source_value(~Logical,~original,logical,source,shifts,axis,others,Nat.add(lane,Nat.mul(group,8n))),0.0) : F32}:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +groups = Grid.groups(Grid.lanes_at(source,axis))
  +first_word = (Grid.lanes_at(shifts,axis) / 8 : U32)
  +cells = Tiled.cell_shape(source,axis,position)
  +moved = cell_shifts(shifts,axis,position)
  +cell = offset(cells,moved,cell_words)
  +first = U32.to_nat(first_word)
  +kept_ok = Boolean.left(kept,axis_inside(groups,first_word,group),hit)
  +group_ok = Boolean.right(kept,axis_inside(groups,first_word,group),hit)
  +local = Nat.sub(group,first)
  +lower = axis_lower(groups,first_word,group,group_ok)
  +group_small = axis_small(groups,first_word,group,group_ok)
  +local_others = others_moved(shifts,axis,others)
  +others_same_source = others_source(source,others,rank,ranked,others_same)
  +below = below_shifted(Grid.removed(source,axis),Grid.removed(shifts,axis),others,cap,room_removed(source,shifts,axis,cap,value_room),
    others_removed(source,others,axis,within,others_same_source),kept_ok)
  +cell_ok = Equal.trans(Bool,inside(cells,moved,Digits.words_nat(cell_words)),inside_words(cells,moved,cell_words),True{},
    Equal.sym(Bool,inside_words(cells,moved,cell_words),inside(cells,moved,Digits.words_nat(cell_words)),inside_nat(cells,moved,cell_words,cap,cap_ok,cell_room)),
    Equal.trans(Bool,inside_words(cells,moved,cell_words),kept && axis_inside(groups,first_word,group),True{},
      cell_split(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot),hit))
  +offset_exact = offset_nat(cells,moved,cell_words,cap,cap_ok,cell_room,Equal.trans(Nat,List.length(&2,U32,cell_words),cell_rank,List.length(&2,U32,cells),cell_count,
    Equal.sym(Nat,List.length(&2,U32,cells),cell_rank,cell_ranked)),cell_fits,cell_ok)
  +exact = {%shifted_placed(True{},position,Grid.removed(shifts,axis),others,first_word,group,
        Equal.sym(Nat,List.length(&2,U32,Grid.removed(shifts,axis)),List.length(&2,Nat,others),
          Equal.trans(Nat,List.length(&2,U32,Grid.removed(shifts,axis)),List.length(&2,U32,Grid.removed(source,axis)),List.length(&2,Nat,others),
            shifts_removed(source,shifts,axis,within,shifts_same),
            Equal.sym(Nat,List.length(&2,Nat,others),List.length(&2,U32,Grid.removed(source,axis)),others_removed(source,others,axis,within,others_same_source))))) :
      {U32.to_nat(cell) == ShapeLaws.digits_value(cells,_) : Nat}
    %shifts_within(source,shifts,axis,within,shifts_same) :
      {U32.to_nat(cell) == ShapeLaws.digits_value(cells,shifted(Grid.placed(_,position,Grid.removed(shifts,axis),first_word),Coordinates.placed_digits(True{},position,others,group))) : Nat}
    %cell_digits : {U32.to_nat(cell) == ShapeLaws.digits_value(cells,shifted(moved,_)) : Nat}
    offset_exact : {U32.to_nat(cell) == ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,local_others,local)) : Nat}}
  +values = Coordinates.lane_values(~Logical,~original,logical,limits,source,axis,position,True{},local_others,local,cell,within,Bounds.strict_le(axis,List.length(&2,U32,source),within),
    removed_room,below,group_small,exact,lane,small)
  +shift_exact = Radix.divide_eight_exact_multiple(Grid.lanes_at(shifts,axis),aligned)
  +channel_same = {%shift_exact : {Nat.add(lane,Nat.mul(local,8n)) == Nat.sub(channel,_) : Nat} channel_local(first,group,lane,lower) :
    {Nat.add(lane,Nat.mul(local,8n)) == Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis))) : Nat}}
  %channel_same : {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},cell)) ==
      Bool.pick(F32,Nat.is_lt(_,U32.to_nat(Grid.lanes_at(source,axis))),original(logical,U32.from_nat(ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,local_others,_)))),0.0) : F32}
  values

# A lane of the source cell the spot reads holds the source value.
def cell_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,+value_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+read: Read<total,rank,axis,others,group,lane,value_words>,
  +hit: {inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Nat.add(lane,Nat.mul(group,8n))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words))) ==
    source_value(~Logical,~original,logical,source,shifts,axis,others,Nat.add(lane,Nat.mul(group,8n))) : F32}:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  Read{+value_digits,+value_count,+in_lane} = read
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +groups = Grid.groups(Grid.lanes_at(source,axis))
  +first_word = (Grid.lanes_at(shifts,axis) / 8 : U32)
  +cells = Tiled.cell_shape(source,axis,position)
  +moved = cell_shifts(shifts,axis,position)
  +cell = offset(cells,moved,cell_words)
  +along = axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel)
  +hit_cell = Equal.trans(Bool,kept && axis_inside(groups,first_word,group),kept && along,True{},
    Equal.cong(Bool,Bool,value => kept && value,axis_inside(groups,first_word,group),along,
      lane_axis(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),total,group,lane,small,aligned,ending,grouped,in_lane)),hit)
  +lane_ok = axis_small(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel,Boolean.right(kept,along,hit))
  %lane_ok : {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},cell)) == Bool.pick(F32,_,source_value(~Logical,~original,logical,source,shifts,axis,others,channel),0.0) : F32}
  cell_pick(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot,hit_cell)

# A lane of a source cell past the lanes of the target holds +0.
def cell_zero(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,
  +facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,
  +out: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == False{} : Bool},
  +hit: {inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words))) == 0.0 : F32}:
  Lanes{+within,+ranked,+cell_ranked,+shifts_same,+cell_room,+value_room,+cell_fits,+value_fits,+removed_room,+aligned,+ending,+grouped,+lane_fits} = facts
  Spot{+cap_ok,+others_same,+cell_digits,+cell_count,+small} = spot
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +groups = Grid.groups(Grid.lanes_at(source,axis))
  +first_word = (Grid.lanes_at(shifts,axis) / 8 : U32)
  +cells = Tiled.cell_shape(source,axis,position)
  +moved = cell_shifts(shifts,axis,position)
  +cell = offset(cells,moved,cell_words)
  +shift = U32.to_nat(Grid.lanes_at(shifts,axis))
  +extent = U32.to_nat(Grid.lanes_at(source,axis))
  +hit_cell = Equal.trans(Bool,kept && axis_inside(groups,first_word,group),inside_words(cells,moved,cell_words),True{},
    Equal.sym(Bool,inside_words(cells,moved,cell_words),kept && axis_inside(groups,first_word,group),cell_split(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot)),hit)
  +past = Order.not_less(channel,U32.to_nat(total),out)
  +ends = Order.transitive(Nat.add(shift,extent),U32.to_nat(total),channel,lane_fits,past)
  +lower = Order.transitive(shift,Nat.add(shift,extent),channel,Order.le_add(shift,extent),ends)
  +beyond = Order.cancel_offset_le(shift,extent,Nat.sub(channel,shift),
    %Equal.sym(Nat,Nat.add(shift,Nat.sub(channel,shift)),channel,Order.restore_shift(shift,channel,lower)) : {Nat.is_le(Nat.add(shift,extent),_) == True{} : Bool} ends)
  %Order.not_below(extent,Nat.sub(channel,shift),beyond) : {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},cell)) == Bool.pick(F32,_,source_value(~Logical,~original,logical,source,shifts,axis,others,channel),0.0) : F32}
  cell_pick(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot,hit_cell)

# The source value the placing of the values reads at the spot.
def value_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+cap: U32,+total: U32,+rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+value_words: List<&2,U32>,
  +values: ValueFacts<cap,rank,source,shifts,axis,others>,+read: Read<total,rank,axis,others,group,lane,value_words>,
  +hit: {inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Nat.add(lane,Nat.mul(group,8n))) == True{} : Bool}) ->
  {original(logical,offset(source,shifts,value_words)) == source_value(~Logical,~original,logical,source,shifts,axis,others,Nat.add(lane,Nat.mul(group,8n))) : F32}:
  ValueFacts{+cap_ok,+within,+ranked,+shifts_same,+value_room,+value_fits,+others_same} = values
  Read{+value_digits,+value_count,+in_lane} = read
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +kept = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others)
  +along = axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),channel)
  +word = offset(source,shifts,value_words)
  +value_ok = Equal.trans(Bool,inside(source,shifts,Digits.words_nat(value_words)),inside_words(source,shifts,value_words),True{},
    Equal.sym(Bool,inside_words(source,shifts,value_words),inside(source,shifts,Digits.words_nat(value_words)),inside_nat(source,shifts,value_words,cap,cap_ok,value_room)),
    Equal.trans(Bool,inside_words(source,shifts,value_words),kept && along,True{},
      value_inside(cap,total,rank,source,shifts,axis,others,group,lane,value_words,values,read),hit))
  +offset_exact = offset_nat(source,shifts,value_words,cap,cap_ok,value_room,Equal.trans(Nat,List.length(&2,U32,value_words),rank,List.length(&2,U32,source),value_count,
    Equal.sym(Nat,List.length(&2,U32,source),rank,ranked)),value_fits,value_ok)
  +number = {%shifted_inserted(shifts,axis,others,channel,
        Equal.trans(Nat,1n+List.length(&2,Nat,others),List.length(&2,U32,source),List.length(&2,U32,shifts),others_source(source,others,rank,ranked,others_same),
          Equal.sym(Nat,List.length(&2,U32,shifts),List.length(&2,U32,source),shifts_same)),
        shifts_within(source,shifts,axis,within,shifts_same)) : {U32.to_nat(word) == ShapeLaws.digits_value(source,_) : Nat}
    %value_digits : {U32.to_nat(word) == ShapeLaws.digits_value(source,shifted(shifts,_)) : Nat}
    offset_exact : {U32.to_nat(word) == ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,others_moved(shifts,axis,others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis))))) : Nat}}
  Equal.cong(U32,F32,at => original(logical,at),word,U32.from_nat(ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,others_moved(shifts,axis,others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis)))))),
    Equal.trans(U32,word,U32.from_nat(U32.to_nat(word)),U32.from_nat(ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,others_moved(shifts,axis,others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis)))))),
      Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)),
      Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(word),ShapeLaws.digits_value(source,Coordinates.inserted_digit(axis,others_moved(shifts,axis,others),Nat.sub(channel,U32.to_nat(Grid.lanes_at(shifts,axis))))),number)))

# ---- One source at one lane of a cell, lanes on the unit axis ----

# What a source placed in cells with its lanes on the unit axis needs.
type Units<-limits: Machine.Limits,-cap: U32,-rank: Nat,-source: List<&2,U32>,-shifts: List<&2,U32>,-axis: Nat> is Data:
  Units{past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool},
    inside_le: {Nat.is_le(axis,List.length(&2,U32,source)) == True{} : Bool},
    ranked: {List.length(&2,U32,source) == rank : Nat},
    shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat},
    value_room: Room(source,shifts,cap),
    value_fits: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(cap)) == True{} : Bool},
    removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(source,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# A cell of a target with its lanes on the unit axis: the words of the cell
# and of its value are those of its digits.
type UnitSpot<-cap: U32,-rank: Nat,-others: List<&2,Nat>,-cell_words: List<&2,U32>,-value_words: List<&2,U32>> is Data:
  UnitSpot{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    cell_digits: {Digits.words_nat(cell_words) == others : List<&2,Nat>},
    value_digits: {Digits.words_nat(value_words) == others : List<&2,Nat>},
    value_count: {List.length(&2,U32,value_words) == rank : Nat}}

def shifts_past(+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool},
  +shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat}) -> {Nat.is_lt(axis,List.length(&2,U32,shifts)) == False{} : Bool}:
  %Equal.sym(Nat,List.length(&2,U32,shifts),List.length(&2,U32,source),shifts_same) : {Nat.is_lt(axis,_) == False{} : Bool}
  past

# The cells of a source with its lanes on the unit axis are the source, and
# their shifts its shifts.
def cells_past(+source: List<&2,U32>,+axis: Nat,+position: Grid.Position,+past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool}) ->
  {Tiled.cell_shape(source,axis,position) == source : List<&2,U32>}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,source)),False{},past) : {Grid.placed(_,position,Grid.removed(source,axis),Grid.groups(Grid.lanes_at(source,axis))) == source : List<&2,U32>}
  Public.removed_past(source,axis,past)

def moved_past(+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+past: {Nat.is_lt(axis,List.length(&2,U32,shifts)) == False{} : Bool}) ->
  {cell_shifts(shifts,axis,position) == shifts : List<&2,U32>}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shifts)),False{},past) : {Grid.placed(_,position,Grid.removed(shifts,axis),(Grid.lanes_at(shifts,axis) / 8 : U32)) == shifts : List<&2,U32>}
  Public.removed_past(shifts,axis,past)

# A cell of a target with its lanes on the unit axis: the words of the cell
# are those of its digits.
type UnitCell<-cap: U32,-rank: Nat,-others: List<&2,Nat>,-words: List<&2,U32>> is Data:
  UnitCell{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    digits: {Digits.words_nat(words) == others : List<&2,Nat>},
    count: {List.length(&2,U32,words) == rank : Nat}}

# A lane of the source cell at a found word holds the source value in the
# first lane and +0 past it.
def unit_pick(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+value_words: List<&2,U32>,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +facts: Units<limits,cap,rank,source,shifts,axis>,+cell: UnitCell<cap,rank,others,value_words>,+hit: {inside_words(source,shifts,value_words) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},offset(source,shifts,value_words))) ==
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),1n),original(logical,offset(source,shifts,value_words)),0.0) : F32}:
  Units{+past,+inside_le,+ranked,+shifts_same,+value_room,+value_fits,+removed_room} = facts
  UnitCell{+cap_ok,+value_digits,+value_count} = cell
  +word = offset(source,shifts,value_words)
  +local = shifted(shifts,others)
  +inside_ok = {%value_digits : {inside(source,shifts,_) == True{} : Bool}
    Equal.trans(Bool,inside(source,shifts,Digits.words_nat(value_words)),inside_words(source,shifts,value_words),True{},
      Equal.sym(Bool,inside_words(source,shifts,value_words),inside(source,shifts,Digits.words_nat(value_words)),inside_nat(source,shifts,value_words,cap,cap_ok,value_room)),hit) :
    {inside(source,shifts,others) == True{} : Bool}}
  +value_same = Equal.trans(Nat,List.length(&2,U32,value_words),rank,List.length(&2,U32,source),value_count,Equal.sym(Nat,List.length(&2,U32,source),rank,ranked))
  +others_same = {%value_digits : {List.length(&2,Nat,_) == List.length(&2,U32,source) : Nat}
    Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(value_words)),List.length(&2,U32,value_words),List.length(&2,U32,source),words_length(value_words),value_same) :
    {List.length(&2,Nat,others) == List.length(&2,U32,source) : Nat}}
  +number = {%value_digits : {U32.to_nat(word) == ShapeLaws.digits_value(source,shifted(shifts,_)) : Nat}
    offset_nat(source,shifts,value_words,cap,cap_ok,value_room,value_same,value_fits,
      %Equal.sym(List<&2,Nat>,Digits.words_nat(value_words),others,value_digits) : {inside(source,shifts,_) == True{} : Bool} inside_ok) :
    {U32.to_nat(word) == ShapeLaws.digits_value(source,local) : Nat}}
  +removed_same = Public.removed_past(source,axis,past)
  +below = {%Equal.sym(List<&2,U32>,Grid.removed(source,axis),source,removed_same) : Coordinates.DigitsBelow(_,local)
    below_shifted(source,shifts,others,cap,value_room,others_same,inside_ok) : Coordinates.DigitsBelow(Grid.removed(source,axis),local)}
  +exact = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(source,axis,position),source,cells_past(source,axis,position,past)) : {U32.to_nat(word) == ShapeLaws.digits_value(_,local) : Nat}
    number : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(source,axis,position),Coordinates.placed_digits(False{},position,local,0n)) : Nat}}
  +one = Public.lanes_past(source,axis,past)
  +group_inside = {%Equal.sym(U32,Grid.lanes_at(source,axis),1,one) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
    {==} : {Nat.is_lt(0n,U32.to_nat(Grid.groups(Grid.lanes_at(source,axis)))) == True{} : Bool}}
  +values = Coordinates.lane_values(~Logical,~original,logical,limits,source,axis,position,False{},local,0n,word,past,inside_le,removed_room,below,group_inside,exact,lane,small)
  +picked = {%one :
      {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},word)) ==
        Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(0n,8n)),U32.to_nat(_)),original(logical,U32.from_nat(ShapeLaws.digits_value(source,local))),0.0) : F32}
    values : {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},word)) ==
      Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),1n),original(logical,U32.from_nat(ShapeLaws.digits_value(source,local))),0.0) : F32}}
  Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},word)),
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),1n),original(logical,U32.from_nat(ShapeLaws.digits_value(source,local))),0.0),
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),1n),original(logical,word),0.0),picked,
    Equal.cong(U32,F32,at => Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),1n),original(logical,at),0.0),U32.from_nat(ShapeLaws.digits_value(source,local)),word,
      Equal.trans(U32,U32.from_nat(ShapeLaws.digits_value(source,local)),U32.from_nat(U32.to_nat(word)),word,
        Equal.cong(Nat,U32,U32.from_nat,ShapeLaws.digits_value(source,local),U32.to_nat(word),Equal.sym(Nat,U32.to_nat(word),ShapeLaws.digits_value(source,local),number)),
        Arithmetic.native_roundtrip(word))))

# The first lane of the source cell at a found word holds the source value.
def unit_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+cell_words: List<&2,U32>,+value_words: List<&2,U32>,
  +facts: Units<limits,cap,rank,source,shifts,axis>,+spot: UnitSpot<cap,rank,others,cell_words,value_words>,+hit: {inside_words(source,shifts,value_words) == True{} : Bool}) ->
  {TileLaws.select_lane(0n,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},offset(source,shifts,value_words))) ==
    original(logical,offset(source,shifts,value_words)) : F32}:
  UnitSpot{+cap_ok,+cell_digits,+value_digits,+value_count} = spot
  unit_pick(~Logical,~original,logical,limits,cap,rank,source,shifts,axis,position,others,value_words,0n,{==},facts,
    {UnitCell{cap_ok,value_digits,value_count} : UnitCell<cap,rank,others,value_words>},hit)

# ---- Every source at one lane of a cell ----

# What a lane of a cell holds before a source: the value before it, unless a
# source still to come holds the cell.
def Begun(covered: Bool,+old_cell: Tile.Tile8,+old_value: F32,+lane: Nat) -> Data:
  match covered:
    case True{}: Unit
    case False{}: {TileLaws.select_lane(lane,old_cell) == old_value : F32}

def begun_of(covered: Bool,+old_cell: Tile.Tile8,+old_value: F32,+lane: Nat,+same: {TileLaws.select_lane(lane,old_cell) == old_value : F32}) -> Begun(covered,old_cell,old_value,lane):
  match covered:
    case True{}: Unit{}
    case False{}: same

# A source found at a cell leaves its lane what it leaves at the value; one
# not found leaves both as they were.
def begun_found(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+context: Representation.Lifted<Logical>,found: Bool,+covered: Bool,+cell: U32,+word: U32,
  +lane: Nat,+old_cell: Tile.Tile8,+old_value: F32,+begun: Begun(found || covered,old_cell,old_value,lane),
  same_value: @+hit: {found == True{} : Bool} -> {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,cell)) == original(logical,word) : F32}) ->
  Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Bool.pick(Maybe<&2,U32>,found,Some{cell},None{}),old_cell),
    Placements.placed_result(~F32,~Logical,~original,logical,Bool.pick(Maybe<&2,U32>,found,Some{word},None{}),old_value),lane):
  match found:
    case True{}: begun_of(covered,Representation.lifted(~Logical,~original,context,cell),original(logical,word),lane,same_value({==}))
    case False{}: begun

# A source with its lanes inside it at a lane of a cell below the lanes of the
# target.
def lane_source(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,+value_words: List<&2,U32>,
  +old_cell: Tile.Tile8,+old_value: F32,+covered: Bool,+facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,
  +spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+read: Read<total,rank,axis,others,group,lane,value_words>,
  +begun: Begun(inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) || covered,old_cell,old_value,lane)) ->
  Begun(covered,
    Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
    Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),lane):
  +cells = Tiled.cell_shape(source,axis,position)
  +moved = cell_shifts(shifts,axis,position)
  +context = {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>}
  +found = inside(Grid.removed(source,axis),Grid.removed(shifts,axis),others) && axis_inside(Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Nat.add(lane,Nat.mul(group,8n)))
  +cell_same = cell_inside(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,value_words,facts,spot,read)
  +value_same = value_inside(cap,total,rank,source,shifts,axis,others,group,lane,value_words,value_facts(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot),read)
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(cells,moved),cell_words,Tasks.strides_of(cells)),Bool.pick(Maybe<&2,U32>,inside_words(cells,moved,cell_words),Some{offset(cells,moved,cell_words)},None{}),
      segments_words(cells,moved,cell_words)) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,_,old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),lane)
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),Bool.pick(Maybe<&2,U32>,inside_words(source,shifts,value_words),Some{offset(source,shifts,value_words)},None{}),segments_words(source,shifts,value_words)) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Bool.pick(Maybe<&2,U32>,inside_words(cells,moved,cell_words),Some{offset(cells,moved,cell_words)},None{}),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,_,old_value),lane)
  %Equal.sym(Bool,inside_words(cells,moved,cell_words),found,cell_same) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Bool.pick(Maybe<&2,U32>,_,Some{offset(cells,moved,cell_words)},None{}),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Bool.pick(Maybe<&2,U32>,inside_words(source,shifts,value_words),Some{offset(source,shifts,value_words)},None{}),old_value),lane)
  %Equal.sym(Bool,inside_words(source,shifts,value_words),found,value_same) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Bool.pick(Maybe<&2,U32>,found,Some{offset(cells,moved,cell_words)},None{}),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Bool.pick(Maybe<&2,U32>,_,Some{offset(source,shifts,value_words)},None{}),old_value),lane)
  begun_found(~Logical,~original,logical,context,found,covered,offset(cells,moved,cell_words),offset(source,shifts,value_words),lane,old_cell,old_value,
    %cell_same : Begun(_ || covered,old_cell,old_value,lane) begun,
    hit => Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,offset(cells,moved,cell_words))),source_value(~Logical,~original,logical,source,shifts,axis,others,Nat.add(lane,Nat.mul(group,8n))),
      original(logical,offset(source,shifts,value_words)),
      cell_value(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,value_words,facts,spot,read,hit),
      Equal.sym(F32,original(logical,offset(source,shifts,value_words)),source_value(~Logical,~original,logical,source,shifts,axis,others,Nat.add(lane,Nat.mul(group,8n))),value_value(~Logical,~original,logical,cap,total,rank,source,shifts,axis,others,group,lane,value_words,value_facts(limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot),read,hit))))

# A source with its lanes on the unit axis at the first lane of a cell.
def unit_step(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+cell_words: List<&2,U32>,+value_words: List<&2,U32>,+old_cell: Tile.Tile8,+old_value: F32,+covered: Bool,
  +facts: Units<limits,cap,rank,source,shifts,axis>,+spot: UnitSpot<cap,rank,others,cell_words,value_words>,
  +begun: Begun(inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) || covered,old_cell,old_value,0n)) ->
  Begun(covered,
    Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
      Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
    Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),0n):
  Units{+past,+inside_le,+ranked,+shifts_same,+value_room,+value_fits,+removed_room} = facts
  UnitSpot{+cap_ok,+cell_digits,+value_digits,+value_count} = spot
  +context = {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>}
  +cells_same = cells_past(source,axis,position,past)
  +moved_same = moved_past(shifts,axis,position,shifts_past(source,shifts,axis,past,shifts_same))
  +words_equal = Gathered.words_same(cell_words,value_words,
    Equal.trans(List<&2,Nat>,Digits.words_nat(cell_words),others,Digits.words_nat(value_words),cell_digits,Equal.sym(List<&2,Nat>,Digits.words_nat(value_words),others,value_digits)))
  +found = inside_words(source,shifts,value_words)
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(source,axis,position),source,cells_same) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Placements.segments_cell(Layout.placed_segments(_,cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(_)),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),0n)
  %Equal.sym(List<&2,U32>,cell_shifts(shifts,axis,position),shifts,moved_same) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Placements.segments_cell(Layout.placed_segments(source,_),cell_words,Tasks.strides_of(source)),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),0n)
  %Equal.sym(List<&2,U32>,cell_words,value_words,words_equal) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Placements.segments_cell(Layout.placed_segments(source,shifts),_,Tasks.strides_of(source)),old_cell),Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),old_value),0n)
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(source,shifts),value_words,Tasks.strides_of(source)),Bool.pick(Maybe<&2,U32>,found,Some{offset(source,shifts,value_words)},None{}),segments_words(source,shifts,value_words)) :
    Begun(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,_,old_cell),Placements.placed_result(~F32,~Logical,~original,logical,_,old_value),0n)
  begun_found(~Logical,~original,logical,context,found,covered,offset(source,shifts,value_words),offset(source,shifts,value_words),0n,old_cell,old_value,
    %words_equal : Begun(inside_words(source,shifts,_) || covered,old_cell,old_value,0n)
    %moved_same : Begun(inside_words(source,_,cell_words) || covered,old_cell,old_value,0n)
    %cells_same : Begun(inside_words(_,cell_shifts(shifts,axis,position),cell_words) || covered,old_cell,old_value,0n)
    begun,
    hit => unit_value(~Logical,~original,logical,limits,cap,rank,source,shifts,axis,position,others,cell_words,value_words,facts,spot,hit))

# ---- The lists of a placing of cells ----

# The cells of every source lifted at its geometry, the segments of its cells,
# its cells, and the segments of its values.
def lifted_logicals(~Logical: Data,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,+axis: Nat,+position: Grid.Position) -> List<&2,Representation.Lifted<Logical>>:
  match logicals sources:
    case Con{+logical,logicals} Con{+source,sources}:
      {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>} <> lifted_logicals(~Logical,logicals,sources,axis,position)
    case logicals sources: []

def cell_segments(sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+axis: Nat,+position: Grid.Position) -> List<&2,List<&2,List<&2,Tasks.Segment>>>:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}:
      Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position)) <> cell_segments(sources,shifts,axis,position)
    case sources shifts: []

def cell_sources(sources: List<&2,List<&2,U32>>,+axis: Nat,+position: Grid.Position) -> List<&2,List<&2,U32>>:
  match sources:
    case Con{+source,sources}: Tiled.cell_shape(source,axis,position) <> cell_sources(sources,axis,position)
    case Nil{}: []

def value_segments(sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>) -> List<&2,List<&2,List<&2,Tasks.Segment>>>:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}: Layout.placed_segments(source,shift) <> value_segments(sources,shifts)
    case sources shifts: []

# Some source holds the cell at the words.
def covers(sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+cell_words: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> Bool:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}:
      inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position),cell_words) || covers(sources,shifts,cell_words,axis,position)
    case sources shifts: False{}

# What every source of a placing of cells with its lanes inside it needs.
def SourcesLanes(+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+axis: Nat,+position: Grid.Position) -> Data:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}:
      KernelGemm.Both<Lanes<limits,cap,total,rank,cell_rank,source,shift,axis,position>,SourcesLanes(limits,cap,total,rank,cell_rank,sources,shifts,axis,position)>
    case Nil{} Nil{}: Unit
    case sources shifts: Empty

def SourcesUnits(+limits: Machine.Limits,+cap: U32,+rank: Nat,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,+axis: Nat) -> Data:
  match sources shifts:
    case Con{+source,sources} Con{+shift,shifts}:
      KernelGemm.Both<Units<limits,cap,rank,source,shift,axis>,SourcesUnits(limits,cap,rank,sources,shifts,axis)>
    case Nil{} Nil{}: Unit
    case sources shifts: Empty

# ---- Every source at a lane of a cell ----

def named_rest(~Logical: Data,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+first: Logical,+source: List<&2,U32>,
  +named: {List.length(&2,Logical,Con{first,logicals}) == List.length(&2,List<&2,U32>,Con{source,sources}) : Nat}) ->
  {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat}:
  pred_same(List.length(&2,Logical,logicals),List.length(&2,List<&2,U32>,sources),named)

# A lane of a cell below the lanes of the target holds after the sources with
# their lanes inside them what the values hold there after the sources.
def sources_lanes(~Logical: Data,~original: Logical -> U32 -> F32,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,
  +limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,
  +cell_words: List<&2,U32>,+value_words: List<&2,U32>,+old_cell: Tile.Tile8,+old_value: F32,
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: SourcesLanes(limits,cap,total,rank,cell_rank,sources,shifts,axis,position),
  +spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+read: Read<total,rank,axis,others,group,lane,value_words>,
  +begun: Begun(covers(sources,shifts,cell_words,axis,position),old_cell,old_value,lane)) ->
  {TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,logicals,value_segments(sources,shifts),sources,value_words,old_value) : F32}:
  match logicals sources shifts:
    case Con{+logical,+rest} Con{+source,+later} Con{+shift,+moves}:
      KernelGemm.Both{+here,+more} = facts
      sources_lanes(~Logical,~original,rest,later,moves,limits,cap,total,rank,cell_rank,axis,position,others,group,lane,cell_words,value_words,
        Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
          Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
        Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shift),value_words,Tasks.strides_of(source)),old_value),
        named_rest(~Logical,rest,later,logical,source,named),more,spot,read,
        lane_source(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shift,axis,position,others,group,lane,cell_words,value_words,old_cell,old_value,
          covers(later,moves,cell_words,axis,position),here,spot,read,begun))
    case Nil{} Nil{} Nil{}: begun
    case Nil{} Nil{} Con{+shift,+moves}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Nil{},axis,position),cell_segments(Nil{},Con{shift,moves},axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Nil{},value_segments(Nil{},Con{shift,moves}),Nil{},value_words,old_value) : F32},facts)
    case Nil{} Con{+source,+later} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Con{source,later},axis,position),cell_segments(Con{source,later},shifts,axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Nil{},value_segments(Con{source,later},shifts),Con{source,later},value_words,old_value) : F32},Digits.zero_successor(List.length(&2,List<&2,U32>,later),named))
    case Con{+logical,+rest} Nil{} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Nil{},axis,position),cell_segments(Nil{},shifts,axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Con{logical,rest},value_segments(Nil{},shifts),Nil{},value_words,old_value) : F32},Digits.zero_successor(List.length(&2,Logical,rest),Equal.sym(Nat,1n+List.length(&2,Logical,rest),0n,named)))
    case Con{+logical,+rest} Con{+source,+later} Nil{}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Con{source,later},axis,position),cell_segments(Con{source,later},Nil{},axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Con{logical,rest},value_segments(Con{source,later},Nil{}),Con{source,later},value_words,old_value) : F32},facts)

# The first lane of a cell holds after the sources with their lanes on the
# unit axis what the values hold there after the sources.
def sources_units(~Logical: Data,~original: Logical -> U32 -> F32,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,
  +limits: Machine.Limits,+cap: U32,+rank: Nat,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,
  +cell_words: List<&2,U32>,+value_words: List<&2,U32>,+old_cell: Tile.Tile8,+old_value: F32,
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: SourcesUnits(limits,cap,rank,sources,shifts,axis),+spot: UnitSpot<cap,rank,others,cell_words,value_words>,
  +begun: Begun(covers(sources,shifts,cell_words,axis,position),old_cell,old_value,0n)) ->
  {TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,logicals,value_segments(sources,shifts),sources,value_words,old_value) : F32}:
  match logicals sources shifts:
    case Con{+logical,+rest} Con{+source,+later} Con{+shift,+moves}:
      KernelGemm.Both{+here,+more} = facts
      sources_units(~Logical,~original,rest,later,moves,limits,cap,rank,axis,position,others,cell_words,value_words,
        Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
          Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
        Placements.placed_result(~F32,~Logical,~original,logical,Placements.segments_cell(Layout.placed_segments(source,shift),value_words,Tasks.strides_of(source)),old_value),
        named_rest(~Logical,rest,later,logical,source,named),more,spot,
        unit_step(~Logical,~original,logical,limits,cap,rank,source,shift,axis,position,others,cell_words,value_words,old_cell,old_value,
          covers(later,moves,cell_words,axis,position),here,spot,begun))
    case Nil{} Nil{} Nil{}: begun
    case Nil{} Nil{} Con{+shift,+moves}: Empty.absurd({TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Nil{},axis,position),cell_segments(Nil{},Con{shift,moves},axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Nil{},value_segments(Nil{},Con{shift,moves}),Nil{},value_words,old_value) : F32},facts)
    case Nil{} Con{+source,+later} +shifts: Empty.absurd({TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Con{source,later},axis,position),cell_segments(Con{source,later},shifts,axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Nil{},value_segments(Con{source,later},shifts),Con{source,later},value_words,old_value) : F32},Digits.zero_successor(List.length(&2,List<&2,U32>,later),named))
    case Con{+logical,+rest} Nil{} +shifts: Empty.absurd({TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Nil{},axis,position),cell_segments(Nil{},shifts,axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Con{logical,rest},value_segments(Nil{},shifts),Nil{},value_words,old_value) : F32},Digits.zero_successor(List.length(&2,Logical,rest),Equal.sym(Nat,1n+List.length(&2,Logical,rest),0n,named)))
    case Con{+logical,+rest} Con{+source,+later} Nil{}: Empty.absurd({TileLaws.select_lane(0n,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Con{source,later},axis,position),cell_segments(Con{source,later},Nil{},axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == Gathered.sources_value(~F32,~Logical,~original,Con{logical,rest},value_segments(Con{source,later},Nil{}),Con{source,later},value_words,old_value) : F32},facts)

# ---- Every source at a lane past the lanes of the target ----

# What a lane of a cell past the lanes holds before a source: +0, unless a
# source still to come holds the cell.
def Zeroed(covered: Bool,+old_cell: Tile.Tile8,+lane: Nat) -> Data:
  match covered:
    case True{}: Unit
    case False{}: {TileLaws.select_lane(lane,old_cell) == 0.0 : F32}

def zero_of(covered: Bool,+old_cell: Tile.Tile8,+lane: Nat,+zero: {TileLaws.select_lane(lane,old_cell) == 0.0 : F32}) -> Zeroed(covered,old_cell,lane):
  match covered:
    case True{}: Unit{}
    case False{}: zero

def zero_found(~Logical: Data,~original: Logical -> U32 -> F32,+context: Representation.Lifted<Logical>,found: Bool,+covered: Bool,+cell: U32,+lane: Nat,+old_cell: Tile.Tile8,
  +zeroed: Zeroed(found || covered,old_cell,lane),
  zero: @+hit: {found == True{} : Bool} -> {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,cell)) == 0.0 : F32}) ->
  Zeroed(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Bool.pick(Maybe<&2,U32>,found,Some{cell},None{}),old_cell),lane):
  match found:
    case True{}: zero_of(covered,Representation.lifted(~Logical,~original,context,cell),lane,zero({==}))
    case False{}: zeroed

# A source with its lanes on the unit axis holds +0 past the first lane.
def unit_zero(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+cell_words: List<&2,U32>,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +facts: Units<limits,cap,rank,source,shifts,axis>,+cell: UnitCell<cap,rank,others,cell_words>,+out: {Nat.is_lt(Nat.add(lane,0n),1n) == False{} : Bool},
  +hit: {inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(source,axis,position)},offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words))) == 0.0 : F32}:
  Units{+past,+inside_le,+ranked,+shifts_same,+value_room,+value_fits,+removed_room} = facts
  +context = {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>}
  +cells_same = cells_past(source,axis,position,past)
  +moved_same = moved_past(shifts,axis,position,shifts_past(source,shifts,axis,past,shifts_same))
  %Equal.sym(List<&2,U32>,Tiled.cell_shape(source,axis,position),source,cells_same) :
    {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,offset(_,cell_shifts(shifts,axis,position),cell_words))) == 0.0 : F32}
  %Equal.sym(List<&2,U32>,cell_shifts(shifts,axis,position),shifts,moved_same) :
    {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,offset(source,_,cell_words))) == 0.0 : F32}
  %out : {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,context,offset(source,shifts,cell_words))) ==
    Bool.pick(F32,_,original(logical,offset(source,shifts,cell_words)),0.0) : F32}
  unit_pick(~Logical,~original,logical,limits,cap,rank,source,shifts,axis,position,others,cell_words,lane,small,facts,cell,
    %moved_same : {inside_words(source,_,cell_words) == True{} : Bool}
    %cells_same : {inside_words(_,cell_shifts(shifts,axis,position),cell_words) == True{} : Bool}
    hit)

def lane_zero_step(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,
  +source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,+cell_words: List<&2,U32>,
  +old_cell: Tile.Tile8,+covered: Bool,+facts: Lanes<limits,cap,total,rank,cell_rank,source,shifts,axis,position>,+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+out: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == False{} : Bool},
  +zeroed: Zeroed(inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) || covered,old_cell,lane)) ->
  Zeroed(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),lane):
  +context = {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>}
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),Bool.pick(Maybe<&2,U32>,inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),Some{offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words)},None{}),
      segments_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words)) :
    Zeroed(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,_,old_cell),lane)
  zero_found(~Logical,~original,context,inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),covered,offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),lane,old_cell,zeroed,
    hit => cell_zero(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shifts,axis,position,others,group,lane,cell_words,facts,spot,out,hit))

def unit_zero_step(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+cap: U32,+rank: Nat,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+cell_words: List<&2,U32>,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},+old_cell: Tile.Tile8,+covered: Bool,
  +facts: Units<limits,cap,rank,source,shifts,axis>,+cell: UnitCell<cap,rank,others,cell_words>,+out: {Nat.is_lt(Nat.add(lane,0n),1n) == False{} : Bool},
  +zeroed: Zeroed(inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words) || covered,old_cell,lane)) ->
  Zeroed(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),lane):
  +context = {Representation.Lifted{logical,Public.geometry_of(source,axis,position)} : Representation.Lifted<Logical>}
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),Bool.pick(Maybe<&2,U32>,inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),Some{offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words)},None{}),
      segments_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words)) :
    Zeroed(covered,Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,_,old_cell),lane)
  zero_found(~Logical,~original,context,inside_words(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),covered,offset(Tiled.cell_shape(source,axis,position),cell_shifts(shifts,axis,position),cell_words),lane,old_cell,zeroed,
    hit => unit_zero(~Logical,~original,logical,limits,cap,rank,source,shifts,axis,position,others,cell_words,lane,small,facts,cell,out,hit))

# A lane of a cell past the lanes of the target holds +0 after the sources
# with their lanes inside them when some source holds the cell.
def sources_zero(~Logical: Data,~original: Logical -> U32 -> F32,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,
  +limits: Machine.Limits,+cap: U32,+total: U32,+rank: Nat,+cell_rank: Nat,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+lane: Nat,
  +cell_words: List<&2,U32>,+old_cell: Tile.Tile8,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: SourcesLanes(limits,cap,total,rank,cell_rank,sources,shifts,axis,position),+spot: Spot<cap,rank,cell_rank,position,others,group,lane,cell_words>,+out: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(total)) == False{} : Bool},
  +zeroed: Zeroed(covers(sources,shifts,cell_words,axis,position),old_cell,lane)) ->
  {TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),cell_words,old_cell)) == 0.0 : F32}:
  match logicals sources shifts:
    case Con{+logical,+rest} Con{+source,+later} Con{+shift,+moves}:
      KernelGemm.Both{+here,+more} = facts
      sources_zero(~Logical,~original,rest,later,moves,limits,cap,total,rank,cell_rank,axis,position,others,group,lane,cell_words,
        Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
          Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
        named_rest(~Logical,rest,later,logical,source,named),more,spot,out,
        lane_zero_step(~Logical,~original,logical,limits,cap,total,rank,cell_rank,source,shift,axis,position,others,group,lane,cell_words,old_cell,
          covers(later,moves,cell_words,axis,position),here,spot,out,zeroed))
    case Nil{} Nil{} Nil{}: zeroed
    case Nil{} Nil{} Con{+shift,+moves}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Nil{},axis,position),cell_segments(Nil{},Con{shift,moves},axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == 0.0 : F32},facts)
    case Nil{} Con{+source,+later} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Con{source,later},axis,position),cell_segments(Con{source,later},shifts,axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == 0.0 : F32},Digits.zero_successor(List.length(&2,List<&2,U32>,later),named))
    case Con{+logical,+rest} Nil{} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Nil{},axis,position),cell_segments(Nil{},shifts,axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == 0.0 : F32},Digits.zero_successor(List.length(&2,Logical,rest),Equal.sym(Nat,1n+List.length(&2,Logical,rest),0n,named)))
    case Con{+logical,+rest} Con{+source,+later} Nil{}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Con{source,later},axis,position),cell_segments(Con{source,later},Nil{},axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == 0.0 : F32},facts)

# A lane past the first of a cell holds +0 after the sources with their lanes
# on the unit axis when some source holds the cell.
def units_zero(~Logical: Data,~original: Logical -> U32 -> F32,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,
  +limits: Machine.Limits,+cap: U32,+rank: Nat,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+cell_words: List<&2,U32>,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +old_cell: Tile.Tile8,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +facts: SourcesUnits(limits,cap,rank,sources,shifts,axis),+cell: UnitCell<cap,rank,others,cell_words>,+out: {Nat.is_lt(Nat.add(lane,0n),1n) == False{} : Bool},
  +zeroed: Zeroed(covers(sources,shifts,cell_words,axis,position),old_cell,lane)) ->
  {TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),cell_words,old_cell)) == 0.0 : F32}:
  match logicals sources shifts:
    case Con{+logical,+rest} Con{+source,+later} Con{+shift,+moves}:
      KernelGemm.Both{+here,+more} = facts
      units_zero(~Logical,~original,rest,later,moves,limits,cap,rank,axis,position,others,cell_words,lane,small,
        Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(source,axis,position)},
          Placements.segments_cell(Layout.placed_segments(Tiled.cell_shape(source,axis,position),cell_shifts(shift,axis,position)),cell_words,Tasks.strides_of(Tiled.cell_shape(source,axis,position))),old_cell),
        named_rest(~Logical,rest,later,logical,source,named),more,cell,out,
        unit_zero_step(~Logical,~original,logical,limits,cap,rank,source,shift,axis,position,others,cell_words,lane,small,old_cell,
          covers(later,moves,cell_words,axis,position),here,cell,out,zeroed))
    case Nil{} Nil{} Nil{}: zeroed
    case Nil{} Nil{} Con{+shift,+moves}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Nil{},axis,position),cell_segments(Nil{},Con{shift,moves},axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == 0.0 : F32},facts)
    case Nil{} Con{+source,+later} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Nil{},Con{source,later},axis,position),cell_segments(Con{source,later},shifts,axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == 0.0 : F32},Digits.zero_successor(List.length(&2,List<&2,U32>,later),named))
    case Con{+logical,+rest} Nil{} +shifts: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Nil{},axis,position),cell_segments(Nil{},shifts,axis,position),cell_sources(Nil{},axis,position),cell_words,old_cell)) == 0.0 : F32},Digits.zero_successor(List.length(&2,Logical,rest),Equal.sym(Nat,1n+List.length(&2,Logical,rest),0n,named)))
    case Con{+logical,+rest} Con{+source,+later} Nil{}: Empty.absurd({TileLaws.select_lane(lane,Gathered.sources_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),lifted_logicals(~Logical,Con{logical,rest},Con{source,later},axis,position),cell_segments(Con{source,later},Nil{},axis,position),cell_sources(Con{source,later},axis,position),cell_words,old_cell)) == 0.0 : F32},facts)

# ---- A cell of the target ----

# What the target of a placing of cells needs: its cells and values within
# the bound, and its other axes within the machine.
type Target<-limits: Machine.Limits,-cap: U32,-values: List<&2,U32>,-axis: Nat,-position: Grid.Position> is Data:
  Target{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,axis,position))),U32.to_nat(cap)) == True{} : Bool},
    values_room: {Nat.is_le(Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap)) == True{} : Bool},
    removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The words of a value of the target at digits below it are its digits.
def digits_words(+values: List<&2,U32>,+digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap)) == True{} : Bool},+below: Coordinates.DigitsBelow(values,digits)) ->
  {Digits.words_nat(Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))) == digits : List<&2,Nat>}:
  +number = ShapeLaws.digits_value(values,digits)
  +small = Coordinates.digits_below(values,digits,below)
  +fits = Order.transitive(number,Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap),
    Order.transitive(number,ShapeLaws.elements(values),Nat.double(ShapeLaws.elements(values)),Bounds.strict_le(number,ShapeLaws.elements(values),small),Order.le_double(ShapeLaws.elements(values))),room)
  +back = Bounds.roundtrip(number,cap,cap_ok,fits)
  +inside = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,back) : {Nat.is_lt(_,ShapeLaws.elements(values)) == True{} : Bool} small :
    {Nat.is_lt(U32.to_nat(U32.from_nat(number)),ShapeLaws.elements(values)) == True{} : Bool}}
  Equal.trans(List<&2,Nat>,Digits.words_nat(Grid.coordinates(values,U32.from_nat(number))),Targets.cell_coordinates(values,U32.to_nat(U32.from_nat(number))),digits,
    Digits.coordinates_exact(values,U32.from_nat(number),cap,cap_ok,room,inside),
    Equal.trans(List<&2,Nat>,Targets.cell_coordinates(values,U32.to_nat(U32.from_nat(number))),Targets.cell_coordinates(values,number),digits,
      Equal.cong(Nat,List<&2,Nat>,at => Targets.cell_coordinates(values,at),U32.to_nat(U32.from_nat(number)),number,back),
      Digits.coordinates_value(values,digits,below)))

# A lane of a cell of the target with its lanes inside it: below the lanes
# the value the placing of the values reads at its channel, +0 past them.
def lane_cases(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,+cap: U32,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+fill_value: F32,+word: U32,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,valid: Bool,+classified: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool},
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+values_room: {Nat.is_le(Nat.double(ShapeLaws.elements(values)),U32.to_nat(cap)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: SourcesLanes(limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),sources,shifts,axis,position),
  +spot: Spot<cap,List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),position,others,group,lane,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>,+covered: {covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position) == True{} : Bool},
  +lifted_lane: {TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32}) ->
  {TileLaws.select_lane(lane,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)) == TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) : F32}:
  match valid:
    case True{}:
      +digits = Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))
      +below = Values.inserted_below(values,axis,others,Nat.add(lane,Nat.mul(group,8n)),others_below,classified,within)
      +value_words = Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n))))))
      +read = {Read{digits_words(values,digits,cap,cap_ok,values_room,below),Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),classified} :
        Read<Grid.lanes_at(values,axis),List.length(&2,U32,values),axis,others,group,lane,value_words>}
      Equal.trans(F32,TileLaws.select_lane(lane,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
        sources_lanes(~Logical,~original,logicals,sources,shifts,limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),axis,position,
          others,group,lane,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),value_words,fill_cell,fill_value,named,facts,spot,read,{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Begun(_,fill_cell,fill_value,lane) Unit{} : Begun(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,fill_value,lane)}),
        Equal.sym(F32,TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),
          %classified : {TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,_,Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32} lifted_lane))
    case False{}:
      Equal.trans(F32,TileLaws.select_lane(lane,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
        sources_zero(~Logical,~original,logicals,sources,shifts,limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),axis,position,
          others,group,lane,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),fill_cell,named,facts,spot,classified,{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,lane) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,lane)}),
        Equal.sym(F32,TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
          %classified : {TileLaws.select_lane(lane,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,_,Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32} lifted_lane))

# A cell of a target with its lanes inside it, held by some source, holds the
# cell of the lifted placing of the values.
def cell_within(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,+cap: U32,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+fill_value: F32,+word: U32,
  +target: Target<limits,cap,values,axis,position>,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: SourcesLanes(limits,cap,Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),sources,shifts,axis,position),+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},+covered: {covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word) == Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word) : Tile.Tile8}:
  Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +below_cells = {%within : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(Grid.placed(True{},position,removed,groups),digits)}
  +others = Values.cell_others(True{},position,digits)
  +group = Values.cell_group(True{},position,digits)
  +others_below = Values.cell_below(True{},position,removed,groups,digits,below_cells)
  +group_inside = Values.group_below(True{},position,removed,groups,digits,below_cells)
  +whole = Values.cell_whole(True{},position,removed,groups,digits,below_cells)
  +exact = {%Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole) : {U32.to_nat(word) == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word)) :
    {U32.to_nat(word) == ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,others,group)) : Nat}}
  +cell_digits = Equal.trans(List<&2,Nat>,Digits.words_nat(Grid.coordinates(Tiled.cell_shape(values,axis,position),word)),digits,Coordinates.placed_digits(True{},position,others,group),
    Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word),Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),digits,whole))
  +others_same = Equal.trans(Nat,1n+List.length(&2,Nat,others),1n+List.length(&2,U32,removed),List.length(&2,U32,values),
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,others),List.length(&2,U32,removed),Values.below_length(removed,others,others_below)),
    Public.removed_length(values,axis,within))
  +cell_count = Digits.coordinates_count(cells,word)
  TileLaws.tiles_equal(Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,0n,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,0n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,0n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,1n,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,1n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,1n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,2n,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,2n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,2n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,3n,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,3n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,3n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,4n,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,4n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,4n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,5n,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,5n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,5n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,6n,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,6n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,6n,{==})),
    lane_cases(~Logical,~original,logicals,sources,shifts,limits,cap,values,axis,position,fill_cell,fill_value,word,others,group,7n,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),{==},cap_ok,values_room,within,others_below,named,facts,
      {Spot{cap_ok,others_same,cell_digits,cell_count,{==}} : Spot<cap,List.length(&2,U32,values),List.length(&2,U32,cells),position,others,group,7n,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},covered,
      Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,True{},others,group,word,within,inside,removed_room,others_below,group_inside,exact,7n,{==})))

# A cell of a target with its lanes on the unit axis, held by some source,
# holds the cell of the lifted placing of the values.
def cell_unit(~Logical: Data,~original: Logical -> U32 -> F32,+logicals: List<&2,Logical>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+limits: Machine.Limits,+cap: U32,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+fill_cell: Tile.Tile8,+fill_value: F32,+word: U32,
  +target: Target<limits,cap,values,axis,position>,+past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: SourcesUnits(limits,cap,List.length(&2,U32,values),sources,shifts,axis),+inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},+covered: {covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word) == Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word) : Tile.Tile8}:
  Target{+cap_ok,+inside,+cells_room,+values_room,+removed_room} = target
  +cells = Tiled.cell_shape(values,axis,position)
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,U32.to_nat(word))
  +removed_same = Public.removed_past(values,axis,past)
  +others_below = {%past : Coordinates.DigitsBelow(Grid.placed(_,position,removed,groups),digits)
    Digits.coordinates_below(cells,U32.to_nat(word),inside_word) : Coordinates.DigitsBelow(removed,digits)}
  +values_below = {%removed_same : Coordinates.DigitsBelow(_,digits) others_below : Coordinates.DigitsBelow(values,digits)}
  +exact = Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),U32.to_nat(word),Digits.value_coordinates(cells,U32.to_nat(word),inside_word))
  +cell_digits = Digits.coordinates_exact(cells,word,cap,cap_ok,cells_room,inside_word)
  +cell_count = {%removed_same : {List.length(&2,U32,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)) == List.length(&2,U32,_) : Nat}
    %past : {List.length(&2,U32,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)) == List.length(&2,U32,Grid.placed(_,position,removed,groups)) : Nat}
    Digits.coordinates_count(cells,word) : {List.length(&2,U32,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)) == List.length(&2,U32,values) : Nat}}
  +one = Public.lanes_past(values,axis,past)
  +group_inside = {%Equal.sym(U32,Grid.lanes_at(values,axis),1,one) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
    {==} : {Nat.is_lt(0n,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool}}
  TileLaws.tiles_equal(Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word),Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word),
    Equal.trans(F32,TileLaws.select_lane(0n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      sources_units(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits))),fill_cell,fill_value,named,facts,
        {UnitSpot{cap_ok,cell_digits,digits_words(values,digits,cap,cap_ok,values_room,values_below),Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))} :
          UnitSpot<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))>},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Begun(_,fill_cell,fill_value,0n) Unit{} : Begun(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,fill_value,0n)}),
      Equal.sym(F32,TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),
        %one : {TileLaws.select_lane(0n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(0n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,0n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(1n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),1n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,1n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,1n)}),
      Equal.sym(F32,TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(1n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(1n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,1n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(2n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),2n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,2n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,2n)}),
      Equal.sym(F32,TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(2n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(2n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,2n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(3n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),3n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,3n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,3n)}),
      Equal.sym(F32,TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(3n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(3n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,3n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(4n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),4n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,4n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,4n)}),
      Equal.sym(F32,TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(4n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(4n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,4n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(5n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),5n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,5n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,5n)}),
      Equal.sym(F32,TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(5n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(5n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,5n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(6n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),6n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,6n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,6n)}),
      Equal.sym(F32,TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(6n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(6n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,6n,{==}))),
    Equal.trans(F32,TileLaws.select_lane(7n,Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Gathered.Placings{lifted_logicals(~Logical,logicals,sources,axis,position),cell_segments(sources,shifts,axis,position),cell_sources(sources,axis,position),Tiled.cell_shape(values,axis,position),fill_cell},word)),0.0,TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),
      units_zero(~Logical,~original,logicals,sources,shifts,limits,cap,List.length(&2,U32,values),axis,position,digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),7n,{==},fill_cell,named,facts,
        {UnitCell{cap_ok,cell_digits,cell_count} : UnitCell<cap,List.length(&2,U32,values),digits,Grid.coordinates(Tiled.cell_shape(values,axis,position),word)>},{==},{%Equal.sym(Bool,covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),True{},covered) : Zeroed(_,fill_cell,7n) Unit{} : Zeroed(covers(sources,shifts,Grid.coordinates(Tiled.cell_shape(values,axis,position),word),axis,position),fill_cell,7n)}),
      Equal.sym(F32,TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)),0.0,
        %one : {TileLaws.select_lane(7n,Representation.lifted(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Representation.Lifted{Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},Public.geometry_of(values,axis,position)},word)) == Bool.pick(F32,Nat.is_lt(Nat.add(7n,Nat.mul(0n,8n)),U32.to_nat(_)),Gathered.placing_value(~F32,~Logical,~original,Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},U32.from_nat(ShapeLaws.digits_value(values,digits))),0.0) : F32}
        Coordinates.lane_values(~Gathered.Placings<Logical,F32>,~Gathered.placing_value(~F32,~Logical,~original),Gathered.Placings{logicals,value_segments(sources,shifts),sources,values,fill_value},limits,values,axis,position,False{},digits,0n,word,past,inside,removed_room,others_below,group_inside,exact,7n,{==}))))
