# The cells a view reads for a target piece of a component lie in the box the
# component takes of the source: the address of a target cell is the view
# offset plus the strides times its coordinates, every read source axis
# lands in the footprint of the class of its target axis, and every other
# source axis keeps the coordinate of the offset.

import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./tensor_mirror_address.bend as MirrorAddress
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_grid_box.bend as Box
import ./tensor_copy_values.bend as CopyValues
import ./tensor_row_major_coordinates.bend as RowMajorCoordinates
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_grid_pieces.bend as Laws
import ./tensor_view_part.bend as ViewPart
import ./tensor_components.bend as Components
import ./tensor_move_readback.bend as Held
import ./tensor_target_cells.bend as Targets
import ./nat_ceiling.bend as NatCeiling
import ./u32_full_range.bend as FullRange
import ../tensor_shape.bend as Shapes
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_canonical_view.bend as Canonical
import ./tensor_box_digits.bend as Digits
import ./tensor_view_metadata_refinement.bend as MetadataLaws

# ---- The address of a target cell ----

# The view offset plus the strides times the coordinates of index, every
# flagged coordinate reflected.
def view_sum(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat) -> Nat:
  match axes:
    case Nil{}: 0n
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),Nat.div(index,inner)),U32.to_nat(stride)),
        view_sum(rest,Views.later_flags(flags),Nat.mod(index,inner)))

def encoded_times(+count: Nat,+word: U32) -> {(U32.from_nat(count) * word : U32) == U32.from_nat(Nat.mul(count,U32.to_nat(word))) : U32}:
  %Arithmetic.native_roundtrip(word) : {(U32.from_nat(count) * _ : U32) == U32.from_nat(Nat.mul(count,U32.to_nat(word))) : U32}
  Multiplication.native_encoded_multiply(count,U32.to_nat(word))

def offset_kept(+offset: U32) -> {offset == U32.from_nat(Nat.add(U32.to_nat(offset),0n)) : U32}:
  Equal.trans(U32,offset,U32.from_nat(U32.to_nat(offset)),U32.from_nat(Nat.add(U32.to_nat(offset),0n)),Equal.sym(U32,U32.from_nat(U32.to_nat(offset)),offset,
    Arithmetic.native_roundtrip(offset)),Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(offset),Nat.add(U32.to_nat(offset),0n),
      Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset)))))

def small_less(+value: Nat,+small: {Nat.is_le(value,1n) == True{} : Bool}) -> {Nat.sub(value,1n) == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n: {==}
    case 2n+ +more: Empty.absurd({Nat.sub(2n+more,1n) == 0n : Nat},WordEncoding.false_true(small))

def start_turned(choice: Bool,+extent: U32,+stride: U32,+offset: U32,+test: {U32.is_lt(1,extent) == choice : Bool}) ->
  {Bool.pick(U32,choice,(offset + (extent - 1) * stride : U32),offset) ==
    U32.from_nat(Nat.add(U32.to_nat(offset),Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)))) : U32}:
  match choice:
    case True{}:
      +size = U32.to_nat(extent)
      +above = {%Comparison.lt(1,extent) : {_ == True{} : Bool} test : {Nat.is_lt(1n,size) == True{} : Bool}}
      +less = Arithmetic.native_subtract(extent,1,Bounds.strict_le(1n,size,above))
      %Equal.sym(U32,(extent - 1 : U32),U32.from_nat(Nat.sub(size,1n)),less) :
        {(offset + _ * stride : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),Nat.mul(Nat.sub(size,1n),U32.to_nat(stride)))) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.sub(size,1n)) * stride : U32),U32.from_nat(Nat.mul(Nat.sub(size,1n),U32.to_nat(stride))),encoded_times(Nat.sub(size,1n),stride)) :
        {(offset + _ : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),Nat.mul(Nat.sub(size,1n),U32.to_nat(stride)))) : U32}
      %Arithmetic.native_roundtrip(offset) :
        {(_ + U32.from_nat(Nat.mul(Nat.sub(size,1n),U32.to_nat(stride))) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),Nat.mul(Nat.sub(size,1n),U32.to_nat(stride)))) : U32}
      WordArithmetic.native_add(U32.to_nat(offset),Nat.mul(Nat.sub(size,1n),U32.to_nat(stride)))
    case False{}:
      +size = U32.to_nat(extent)
      +below = Order.not_less(1n,size,{%Comparison.lt(1,extent) : {_ == False{} : Bool} test : {Nat.is_lt(1n,size) == False{} : Bool}})
      %Equal.sym(Nat,Nat.sub(size,1n),0n,small_less(size,below)) :
        {offset == U32.from_nat(Nat.add(U32.to_nat(offset),Nat.mul(_,U32.to_nat(stride)))) : U32}
      offset_kept(offset)

# The offset of an axis read in reverse moves to its last index.
def start_value(reverse: Bool,+extent: U32,+stride: U32,+offset: U32) ->
  {Views.mirrored_start(reverse,extent,stride,offset) == U32.from_nat(Nat.add(U32.to_nat(offset),MirrorAddress.turn_axis(reverse,U32.to_nat(extent),stride))) : U32}:
  match reverse:
    case False{}: offset_kept(offset)
    case True{}: start_turned(U32.is_lt(1,extent),extent,stride,offset,{==})

def sum_moved(+offset: Nat,+turn: Nat,+rest: Nat,+term: Nat) ->
  {Nat.add(Nat.add(Nat.add(offset,turn),rest),term) == Nat.add(offset,Nat.add(Nat.add(term,turn),rest)) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.add(offset,turn),rest),term),Nat.add(Nat.add(offset,turn),Nat.add(rest,term)),
      Addition.associative(Nat.add(offset,turn),rest,term)) :
    {_ == Nat.add(offset,Nat.add(Nat.add(term,turn),rest)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(offset,turn),Nat.add(rest,term)),Nat.add(offset,Nat.add(turn,Nat.add(rest,term))),
      Addition.associative(offset,turn,Nat.add(rest,term))) :
    {_ == Nat.add(offset,Nat.add(Nat.add(term,turn),rest)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(term,turn),rest),Nat.add(term,Nat.add(turn,rest)),Addition.associative(term,turn,rest)) :
    {Nat.add(offset,Nat.add(turn,Nat.add(rest,term))) == Nat.add(offset,_) : Nat}
  Equal.cong(Nat,Nat,value => Nat.add(offset,value),Nat.add(turn,Nat.add(rest,term)),Nat.add(term,Nat.add(turn,rest)),
    Equal.trans(Nat,Nat.add(turn,Nat.add(rest,term)),Nat.add(turn,Nat.add(term,rest)),Nat.add(term,Nat.add(turn,rest)),
      Equal.cong(Nat,Nat,value => Nat.add(turn,value),Nat.add(rest,term),Nat.add(term,rest),Order.commutative(rest,term)),
      RowMajor.add_swap(turn,term,rest)))

def words_moved(+head: U32,+term: U32,+tail: U32) -> {(head + (term + tail : U32) : U32) == ((head + tail : U32) + term : U32) : U32}:
  Equal.trans(U32,(head + (term + tail : U32) : U32),(head + (tail + term : U32) : U32),((head + tail : U32) + term : U32),
    Equal.cong(U32,U32,value => (head + value : U32),(term + tail : U32),(tail + term : U32),Arithmetic.add_commutative(term,tail)),
    Equal.sym(U32,((head + tail : U32) + term : U32),(head + (tail + term : U32) : U32),Arithmetic.add_associative(head,tail,term)))

# ---- The sums of a view ----

def last_split(size: Nat,inner: Nat,+size_positive: {Nat.is_lt(0n,size) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.sub(Nat.mul(size,inner),1n) == Nat.add(Nat.mul(Nat.sub(size,1n),inner),Nat.sub(inner,1n)) : Nat}:
  match size inner:
    case 0n +inner:
      Empty.absurd({Nat.sub(Nat.mul(0n,inner),1n) == Nat.add(Nat.mul(Nat.sub(0n,1n),inner),Nat.sub(inner,1n)) : Nat},WordEncoding.false_true(size_positive))
    case 1n+ +rest 0n:
      Empty.absurd({Nat.sub(Nat.mul(1n+rest,0n),1n) == Nat.add(Nat.mul(Nat.sub(1n+rest,1n),0n),Nat.sub(0n,1n)) : Nat},WordEncoding.false_true(inner_positive))
    case 1n+ +rest 1n+ +below:
      +tail = Nat.mul(rest,1n+below)
      %Equal.sym(Nat,Nat.sub(1n+rest,1n),rest,Order.subtract_zero(rest)) :
        {Nat.sub(Nat.mul(1n+rest,1n+below),1n) == Nat.add(Nat.mul(_,1n+below),Nat.sub(1n+below,1n)) : Nat}
      %Equal.sym(Nat,Nat.sub(1n+below,1n),below,Order.subtract_zero(below)) :
        {Nat.sub(Nat.mul(1n+rest,1n+below),1n) == Nat.add(tail,_) : Nat}
      Equal.trans(Nat,Nat.sub(1n+Nat.add(below,tail),1n),Nat.add(below,tail),Nat.add(tail,below),
        Order.subtract_zero(Nat.add(below,tail)),Order.commutative(below,tail))

# A view reads its last index at its last span.
def last_sum(axes: List<&2,Views.Axis>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {view_sum(axes,[],Nat.sub(ShapeLaws.elements(Views.axis_extents(axes)),1n)) == ViewBounds.last_span(axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +size = U32.to_nat(extent)
      +step = U32.to_nat(stride)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +last = Nat.sub(Nat.mul(size,inner),1n)
      +quotient = Nat.sub(size,1n)
      +remainder = Nat.sub(inner,1n)
      +pair = Division.unique(last,inner,quotient,remainder,last_split(size,inner,size_positive,inner_positive),inner_positive,Order.last_below(inner,inner_positive))
      +word = FullRange.subtract(extent,1,extent,Order.positive_one(size,size_positive),Order.sub_le(size,1n))
      %Equal.sym(Nat & Nat,Nat.divmod(last,inner),(quotient,remainder),pair) :
        {Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),step),view_sum(rest,[],Pair.snd(Nat,Nat,_))) == Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),step),ViewBounds.last_span(rest)) : Nat}
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),quotient,word) :
        {Nat.add(Nat.mul(quotient,step),view_sum(rest,[],remainder)) == Nat.add(Nat.mul(_,step),ViewBounds.last_span(rest)) : Nat}
      Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(quotient,step),sum),view_sum(rest,[],remainder),ViewBounds.last_span(rest),last_sum(rest,inner_positive))

# The extents of axes with one more appended.
def appended_extents(axes: List<&2,Views.Axis>,+extent: U32,+stride: U32) ->
  {Views.axis_extents(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}])) == List.append(&2,U32,Views.axis_extents(axes),[extent]) : List<&2,U32>}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+size,+step},+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => size <> later,Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])),
        List.append(&2,U32,Views.axis_extents(rest),[extent]),appended_extents(rest,extent,stride))

def appended_count(+axes: List<&2,Views.Axis>,+extent: U32,+stride: U32) ->
  {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]))) == Nat.mul(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(extent)) : Nat}:
  +front = ShapeLaws.elements(Views.axis_extents(axes))
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}])),List.append(&2,U32,Views.axis_extents(axes),[extent]),
      appended_extents(axes,extent,stride)) :
    {ShapeLaws.elements(_) == Nat.mul(front,U32.to_nat(extent)) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,Views.axis_extents(axes),[extent])),Nat.mul(front,Nat.mul(U32.to_nat(extent),1n)),Nat.mul(front,U32.to_nat(extent)),
    ShapeLaws.elements_appended(Views.axis_extents(axes),[extent]),
    Equal.cong(Nat,Nat,last => Nat.mul(front,last),Nat.mul(U32.to_nat(extent),1n),U32.to_nat(extent),Order.times_one(U32.to_nat(extent))))

def divided_one(+value: Nat) -> {Nat.div(value,1n) == value : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,1n),(value,0n),Division.unique(value,1n,value,0n,
      Equal.sym(Nat,Nat.add(Nat.mul(value,1n),0n),value,Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,
        Addition.zero_right(Nat.mul(value,1n)),Order.times_one(value))),{==},{==})) :
    {Pair.fst(Nat,Nat,_) == value : Nat}
  {==}

def remainder_below(+value: Nat,+divisor: Nat,+below: {Nat.is_lt(value,divisor) == True{} : Bool}) -> {Nat.mod(value,divisor) == value : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,divisor),(0n,value),Division.small(value,divisor,below)) : {Pair.snd(Nat,Nat,_) == value : Nat}
  {==}

# Axes with one more appended read an index at the quotient by its extent on
# the others and the remainder on it.
def appended_sum(axes: List<&2,Views.Axis>,+extent: U32,+stride: U32,+index: Nat,
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}])))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {view_sum(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),[],index) == Nat.add(view_sum(axes,[],Nat.div(index,U32.to_nat(extent))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))) : Nat}:
  match axes:
    case Nil{}:
      +below = {%Order.times_one(U32.to_nat(extent)) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,U32.to_nat(extent)) == True{} : Bool}}
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,1n),U32.to_nat(stride)),0n),Nat.mul(Nat.div(index,1n),U32.to_nat(stride)),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride)),Addition.zero_right(Nat.mul(Nat.div(index,1n),U32.to_nat(stride))),
        Equal.trans(Nat,Nat.mul(Nat.div(index,1n),U32.to_nat(stride)),Nat.mul(index,U32.to_nat(stride)),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride)),Equal.cong(Nat,Nat,number => Nat.mul(number,U32.to_nat(stride)),Nat.div(index,1n),index,divided_one(index)),
        Equal.cong(Nat,Nat,number => Nat.mul(number,U32.to_nat(stride)),index,Nat.mod(index,U32.to_nat(extent)),Equal.sym(Nat,Nat.mod(index,U32.to_nat(extent)),index,remainder_below(index,U32.to_nat(extent),below)))))
    case Con{Views.Axis{+size,+step},+rest}:
      +whole_positive = DivisionOrder.right_factor_positive(U32.to_nat(size),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]))),Order.le_lt_trans(0n,index,Nat.mul(U32.to_nat(size),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])))),Order.zero_le(index),inside))
      +product_positive = {%appended_count(rest,extent,stride) : {Nat.is_lt(0n,_) == True{} : Bool} whole_positive : {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))) == True{} : Bool}}
      +rest_positive = RowMajor.left_factor_positive(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent),product_positive)
      +later = appended_sum(rest,extent,stride,Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),
        {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]))),Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent)),appended_count(rest,extent,stride)) : {Nat.is_lt(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),_) == True{} : Bool}
          DivisionOrder.remainder_small(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent)),product_positive) : {Nat.is_lt(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])))) == True{} : Bool}},positive)
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])))),U32.to_nat(step)),view_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),[],Nat.mod(index,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])))))),Nat.add(Nat.mul(Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(step)),view_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),[],Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))))),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Equal.cong(Nat,Nat,count => Nat.add(Nat.mul(Nat.div(index,count),U32.to_nat(step)),view_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),[],Nat.mod(index,count))),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]))),Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent)),appended_count(rest,extent,stride)),
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(step)),view_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),[],Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))))),Nat.add(Nat.mul(Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(step)),sum),view_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),[],Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent)))),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride))),later),
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Equal.cong(Nat,Nat,quotient => Nat.add(Nat.mul(quotient,U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.div(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),DivisionOrder.product_quotient(index,ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent),rest_positive,positive)),
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Equal.cong(Nat,Nat,digit => Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],digit),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.div(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),DivisionOrder.remainder_quotient(index,ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent),rest_positive,positive)),
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride)))),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Equal.cong(Nat,Nat,digit => Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(digit,U32.to_nat(stride)))),Nat.mod(Nat.mod(index,Nat.mul(ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent))),U32.to_nat(extent)),Nat.mod(index,U32.to_nat(extent)),DivisionOrder.remainder_remainder(index,ShapeLaws.elements(Views.axis_extents(rest)),U32.to_nat(extent),rest_positive,positive)),
        Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride))),Nat.add(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),Nat.add(view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride)))),Addition.associative(Nat.mul(Nat.div(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest))),U32.to_nat(step)),view_sum(rest,[],Nat.mod(Nat.div(index,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(rest)))),Nat.mul(Nat.mod(index,U32.to_nat(extent)),U32.to_nat(stride)))))))))

# The mirrored view addresses a target index at the view offset plus its
# reflected sum.
def view_address(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+index: Nat,
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),index) : U32) ==
    U32.from_nat(Nat.add(U32.to_nat(offset),view_sum(axes,flags,index))) : U32}:
  match axes:
    case Nil{}:
      Equal.trans(U32,(offset + 0 : U32),offset,U32.from_nat(Nat.add(U32.to_nat(offset),0n)),Arithmetic.add_zero(offset),offset_kept(offset))
    case Con{Views.Axis{+extent,+stride},+rest}:
      +reverse = Views.flag(flags)
      +later = Views.later_flags(flags)
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +size = U32.to_nat(extent)
      +inner_positive = Components.inner_positive(size,inner,index,inside)
      +coordinate = Nat.div(index,inner)
      +remainder = Nat.mod(index,inner)
      +start = Views.mirrored_start(reverse,extent,stride,offset)
      +step = Views.directed(reverse,stride)
      +moved = Views.mirrored_offset(rest,later,start)
      +tail = Box.stepped_sum(Views.axis_extents(rest),ViewPart.view_steps(Views.mirrored_axes(rest,later)),remainder)
      +term = (U32.from_nat(coordinate) * step : U32)
      +turn = MirrorAddress.turn_axis(reverse,size,stride)
      +sum = view_sum(rest,later,remainder)
      +base = U32.to_nat(offset)
      +product = Nat.mul(coordinate,U32.to_nat(step))
      +reflected = Nat.mul(MirrorAddress.reflected_coordinate(reverse,size,coordinate),U32.to_nat(stride))
      +recursive = view_address(rest,later,start,remainder,DivisionOrder.remainder_small(index,inner,inner_positive))
      +start_word = Equal.trans(U32,U32.from_nat(U32.to_nat(start)),start,U32.from_nat(Nat.add(base,turn)),Arithmetic.native_roundtrip(start),
        start_value(reverse,extent,stride,offset))
      +joined = Equal.trans(U32,((moved + tail : U32) + term : U32),(U32.from_nat(Nat.add(U32.to_nat(start),sum)) + U32.from_nat(product) : U32),
        U32.from_nat(Nat.add(Nat.add(U32.to_nat(start),sum),product)),
        MirrorAddress.both((moved + tail : U32),U32.from_nat(Nat.add(U32.to_nat(start),sum)),term,U32.from_nat(product),recursive,encoded_times(coordinate,step)),
        WordArithmetic.native_add(Nat.add(U32.to_nat(start),sum),product))
      +turned = MirrorAddress.sum_congruent(Nat.add(U32.to_nat(start),sum),product,Nat.add(Nat.add(base,turn),sum),product,
        MirrorAddress.sum_congruent(U32.to_nat(start),sum,Nat.add(base,turn),sum,start_word,{==}),{==})
      +reflected_word = MirrorAddress.sum_congruent(base,Nat.add(Nat.add(product,turn),sum),base,Nat.add(reflected,sum),{==},
        MirrorAddress.sum_congruent(Nat.add(product,turn),sum,reflected,sum,
          MirrorAddress.axis_address(reverse,size,coordinate,stride,DivisionOrder.quotient_below(index,size,inner,inner_positive,inside)),{==}))
      Equal.trans(U32,(moved + (term + tail : U32) : U32),((moved + tail : U32) + term : U32),U32.from_nat(Nat.add(base,Nat.add(reflected,sum))),
        words_moved(moved,term,tail),
        Equal.trans(U32,((moved + tail : U32) + term : U32),U32.from_nat(Nat.add(Nat.add(Nat.add(base,turn),sum),product)),U32.from_nat(Nat.add(base,Nat.add(reflected,sum))),
          Equal.trans(U32,((moved + tail : U32) + term : U32),U32.from_nat(Nat.add(Nat.add(U32.to_nat(start),sum),product)),
            U32.from_nat(Nat.add(Nat.add(Nat.add(base,turn),sum),product)),joined,turned),
          Equal.trans(U32,U32.from_nat(Nat.add(Nat.add(Nat.add(base,turn),sum),product)),U32.from_nat(Nat.add(base,Nat.add(Nat.add(product,turn),sum))),
            U32.from_nat(Nat.add(base,Nat.add(reflected,sum))),
            Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(Nat.add(base,turn),sum),product),Nat.add(base,Nat.add(Nat.add(product,turn),sum)),
              sum_moved(base,turn,sum,product)),
            reflected_word)))

def reflected_le(reverse: Bool,+extent: Nat,+coordinate: Nat,+inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {Nat.is_le(MirrorAddress.reflected_coordinate(reverse,extent,coordinate),Nat.sub(extent,1n)) == True{} : Bool}:
  match reverse:
    case False{}: Components.pred_le(coordinate,extent,inside)
    case True{}: Order.sub_le(Nat.sub(extent,1n),coordinate)

def sum_le(+a: Nat,+b: Nat,+c: Nat,+d: Nat,+left: {Nat.is_le(a,c) == True{} : Bool},+right: {Nat.is_le(b,d) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(a,b),Nat.add(c,d)) == True{} : Bool}:
  Order.transitive(Nat.add(a,b),Nat.add(c,b),Nat.add(c,d),Order.add_le(a,c,b,left),Order.offset_le(c,b,d,right))

# The reflected sum of a target index stays at or below the sum at the last
# coordinates.
def view_sum_bound(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {Nat.is_le(view_sum(axes,flags,index),ViewBounds.last_span(axes)) == True{} : Bool}:
  match axes:
    case Nil{}: Order.reflexive(0n)
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +size = U32.to_nat(extent)
      +inner_positive = Components.inner_positive(size,inner,index,inside)
      +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(size,inner),Order.zero_le(index),inside)
      +size_positive = RowMajor.left_factor_positive(size,inner,whole_positive)
      +size_room = Order.transitive(size,Nat.mul(size,inner),U32.to_nat(cap),RowMajor.product_covers(size,inner,inner_positive),room)
      +inner_room = Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(cap),DivisionOrder.le_multiple(size,inner,size_positive),room)
      +last = Arithmetic.subtract(extent,1,cap,cap_ok,Order.strict_follow(0n,size,size_positive),
        Order.transitive(Nat.sub(size,1n),size,U32.to_nat(cap),Order.sub_le(size,1n),size_room))
      +coordinate = Nat.div(index,inner)
      sum_le(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),size,coordinate),U32.to_nat(stride)),view_sum(rest,Views.later_flags(flags),Nat.mod(index,inner)),
        Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ViewBounds.last_span(rest),
        {%Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(size,1n),last) :
            {Nat.is_le(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),size,coordinate),U32.to_nat(stride)),Nat.mul(_,U32.to_nat(stride))) == True{} : Bool}
          RowMajor.mul_le(MirrorAddress.reflected_coordinate(Views.flag(flags),size,coordinate),Nat.sub(size,1n),U32.to_nat(stride),
            reflected_le(Views.flag(flags),size,coordinate,DivisionOrder.quotient_below(index,size,inner,inner_positive,inside))) :
          {Nat.is_le(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),size,coordinate),U32.to_nat(stride)),
            Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))) == True{} : Bool}},
        view_sum_bound(rest,Views.later_flags(flags),Nat.mod(index,inner),cap,cap_ok,inner_room,DivisionOrder.remainder_small(index,inner,inner_positive)))

# The address of a target index: the view offset plus its reflected sum,
# inside the length the view fits.
def ReadAt(+offset: U32,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+address: U32,+length: U32) -> Data:
  Held.Both<{U32.to_nat(address) == Nat.add(U32.to_nat(offset),view_sum(axes,flags,index)) : Nat},
    {Nat.is_lt(Nat.add(U32.to_nat(offset),view_sum(axes,flags,index)),U32.to_nat(length)) == True{} : Bool}>

def address_at(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+index: Nat,+length: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},+length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  ReadAt(offset,axes,flags,index,
    (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),index) : U32),length):
  +base = U32.to_nat(offset)
  +sum = view_sum(axes,flags,index)
  +bound = Order.le_lt_trans(Nat.add(base,sum),Nat.add(base,ViewBounds.last_span(axes)),U32.to_nat(length),
    Order.offset_le(base,sum,ViewBounds.last_span(axes),view_sum_bound(axes,flags,index,cap,cap_ok,room,inside)),
    {%ViewBounds.last_location(axes,base) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} ViewBounds.last_address(axes,offset,length,fits) :
      {Nat.is_lt(Nat.add(base,ViewBounds.last_span(axes)),U32.to_nat(length)) == True{} : Bool}})
  +address = (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),index) : U32)
  Held.Both{Equal.trans(Nat,U32.to_nat(address),U32.to_nat(U32.from_nat(Nat.add(base,sum))),Nat.add(base,sum),
      Equal.cong(U32,Nat,word => U32.to_nat(word),address,U32.from_nat(Nat.add(base,sum)),view_address(axes,flags,offset,index,inside)),
      Bounds.roundtrip(Nat.add(base,sum),cap,cap_ok,Order.transitive(Nat.add(base,sum),U32.to_nat(length),U32.to_nat(cap),Bounds.strict_le(Nat.add(base,sum),U32.to_nat(length),bound),
        length_room))),
    bound}

# ---- The source digits a target cell reads ----

# The source index a read gives a target coordinate.
def read_value(mirrored: Bool,+offset: Nat,+step: Nat,+coordinate: Nat) -> Nat:
  match mirrored:
    case True{}: Nat.sub(offset,Nat.mul(step,coordinate))
    case False{}: Nat.add(offset,Nat.mul(step,coordinate))

# The index of source axis axis: the first target axis that reads it, or the
# fixed index of the offset.
def read_digit(readings: List<&2,Tasks.Reading>,coordinates: List<&2,Nat>,+axis: U32,+fixed: Nat) -> Nat:
  match readings coordinates:
    case Con{Tasks.Reading{extent,block,Tasks.Read{+other,source_block,+step,+offset,+mirrored,width,divisor,shift,bound,clamped}},+rest} Con{+coordinate,+more}:
      Bool.pick(Nat,U32.is_eq(other,axis),read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate),read_digit(rest,more,axis,fixed))
    case Con{reading,+rest} Con{coordinate,+more}:
      read_digit(rest,more,axis,fixed)
    case readings coordinates:
      fixed

def source_digits(+readings: List<&2,Tasks.Reading>,+coordinates: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+axis: U32) -> List<&2,Nat>:
  match extents fixed:
    case Con{extent,+rest} Con{+index,+more}:
      Con{read_digit(readings,coordinates,axis,U32.to_nat(index)),source_digits(readings,coordinates,rest,more,(axis + 1 : U32))}
    case extents fixed:
      Nil{}

# A value inside a span.
def InSpan(span: Tasks.Span,+value: Nat) -> Data:
  match span:
    case Tasks.Span{+first,+count}:
      Held.Both<{Nat.is_le(U32.to_nat(first),value) == True{} : Bool},{Nat.is_lt(value,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}>

# A digit inside a span and below its extent.
def AxisIn(+span: Tasks.Span,+digit: Nat,+extent: U32) -> Data:
  Held.Both<InSpan(span,digit),{Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}>

# Every digit lies in the span of its axis and below its extent.
def DigitsIn(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,digits: List<&2,Nat>) -> Data:
  match spans extents digits:
    case Con{+span,spans} Con{+extent,extents} Con{+digit,digits}:
      Held.Both<AxisIn(span,digit,extent),DigitsIn(spans,extents,digits)>
    case Nil{} Nil{} Nil{}:
      Unit
    case spans extents digits:
      Empty

def BoxDigits(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+digits: List<&2,Nat>) -> Data:
  Held.Both<Components.InBox(spans,extents,ShapeLaws.digits_value(extents,digits)),{Nat.is_lt(ShapeLaws.digits_value(extents,digits),ShapeLaws.elements(extents)) == True{} : Bool}>

# One more axis whose digit lies in its span and below its extent.
def digits_step(+first: U32,+count: U32,+rest_spans: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+digit: Nat,+more: List<&2,Nat>,
  +held: InSpan(Tasks.Span{first,count},digit),
  +small: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+later: BoxDigits(rest_spans,rest,more)) ->
  BoxDigits(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},Con{digit,more}):
  Held.Both{+low,+high} = held
  Held.Both{+box,+below} = later
  +inner = ShapeLaws.elements(rest)
  +value = ShapeLaws.digits_value(rest,more)
  +inner_positive = Order.le_lt_trans(0n,value,inner,Order.zero_le(value),below)
  +whole = Nat.add(Nat.mul(digit,inner),value)
  +pair = Division.unique(whole,inner,digit,value,{==},inner_positive,below)
  +quotient = Equal.cong(Nat & Nat,Nat,found => Pair.fst(Nat,Nat,found),Nat.divmod(whole,inner),(digit,value),pair)
  +remainder = Equal.cong(Nat & Nat,Nat,found => Pair.snd(Nat,Nat,found),Nat.divmod(whole,inner),(digit,value),pair)
  Held.Both{
    Held.Both{{%Equal.sym(Nat,Nat.div(whole,inner),digit,quotient) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} low :
        {Nat.is_le(U32.to_nat(first),Nat.div(whole,inner)) == True{} : Bool}},
      Held.Both{{%Equal.sym(Nat,Nat.div(whole,inner),digit,quotient) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool} high :
          {Nat.is_lt(Nat.div(whole,inner),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}},
        {%Equal.sym(Nat,Nat.mod(whole,inner),value,remainder) : Components.InBox(rest_spans,rest,_) box : Components.InBox(rest_spans,rest,Nat.mod(whole,inner))}}},
    RowMajor.radix(digit,U32.to_nat(extent),value,inner,small,below)}

# Digits inside the spans make a number inside the box.
def digits_box(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,digits: List<&2,Nat>,+inside: DigitsIn(spans,extents,digits)) -> BoxDigits(spans,extents,digits):
  match spans extents digits:
    case Con{Tasks.Span{+first,+count},+rest_spans} Con{+extent,+rest} Con{+digit,+more}:
      Held.Both{+axis,+others} = inside
      Held.Both{+held,+small} = axis
      digits_step(first,count,rest_spans,extent,rest,digit,more,held,small,digits_box(rest_spans,rest,more,others))
    case Nil{} Nil{} Nil{}:
      Held.Both{Unit{},{==}}
    case Nil{} Nil{} Con{+digit,+more}:
      Empty.absurd(BoxDigits(Nil{},Nil{},Con{digit,more}),inside)
    case Nil{} Con{+extent,+rest} +digits:
      Empty.absurd(BoxDigits(Nil{},Con{extent,rest},digits),inside)
    case Con{Tasks.Span{+first,+count},+rest_spans} Nil{} +digits:
      Empty.absurd(BoxDigits(Con{Tasks.Span{first,count},rest_spans},Nil{},digits),inside)
    case Con{Tasks.Span{+first,+count},+rest_spans} Con{+extent,+rest} Nil{}:
      Empty.absurd(BoxDigits(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},Nil{}),inside)

# ---- Source indices inside the spans of a box ----

def none_in(+extent: U32,+block: U32,+coordinate: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(coordinate),U32.to_nat(extent)) == True{} : Bool}) ->
  InSpan(Tasks.box_span(None{},extent,block,coordinate),U32.to_nat(coordinate)):
  +total = U32.to_nat(extent)
  +size = U32.to_nat(block)
  +value = U32.to_nat(coordinate)
  +unit = U32.max(block,1)
  +unit_exact = Extrema.max_one_exact(block,size,{==},positive)
  +quotient = Nat.div(value,size)
  +total_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +value_room = Order.transitive(value,total,U32.to_nat(cap),Bounds.strict_le(value,total,inside),total_room)
  +quotient_exact = U32Division.quotient_exact(cap,cap_ok,coordinate,unit,value,size,{==},unit_exact,
    {%unit_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} block_room : {Nat.is_le(Nat.double(size),U32.to_nat(cap)) == True{} : Bool}},positive)
  +below = DivisionOrder.product_below(value,size,positive)
  +scaled = (coordinate / unit * block : U32)
  +start = Nat.mul(quotient,size)
  +start_total = Order.transitive(start,value,total,below,Bounds.strict_le(value,total,inside))
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,(coordinate / unit : U32),block,quotient,size,quotient_exact,{==},
    Order.transitive(start,value,U32.to_nat(cap),below,value_room))
  +low = U32.min(scaled,extent)
  +low_exact = Equal.trans(Nat,U32.to_nat(low),Nat.min(U32.to_nat(scaled),total),start,Extrema.minimum(scaled,extent),
    Equal.trans(Nat,Nat.min(U32.to_nat(scaled),total),Nat.min(start,total),start,
      Equal.cong(Nat,Nat,found => Nat.min(found,total),U32.to_nat(scaled),start,scaled_exact),NatExtrema.minimum_when_ordered(start,total,start_total)))
  +reach = (low + block : U32)
  +reach_exact = Arithmetic.sum_exact(cap,cap_ok,low,block,start,size,low_exact,{==},
    Order.transitive(Nat.add(start,size),Nat.add(total,size),U32.to_nat(cap),Order.add_le(start,total,size,start_total),room))
  +ordered = {%Equal.sym(Nat,U32.to_nat(low),start,low_exact) : {Nat.is_le(_,Nat.min(U32.to_nat(reach),total)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(reach),Nat.add(start,size),reach_exact) : {Nat.is_le(start,Nat.min(_,total)) == True{} : Bool}
    NatExtrema.minimum_lower(Nat.add(start,size),total,start,Order.le_add(start,size),start_total) :
    {Nat.is_le(U32.to_nat(low),Nat.min(U32.to_nat(reach),total)) == True{} : Bool}}
  +ends = Components.span_sum(low,reach,extent,cap,cap_ok,ordered,total_room)
  Held.Both{{%Equal.sym(Nat,U32.to_nat(low),start,low_exact) : {Nat.is_le(_,value) == True{} : Bool} below : {Nat.is_le(U32.to_nat(low),value) == True{} : Bool}},
    {%Equal.sym(Nat,Nat.add(U32.to_nat(low),U32.to_nat((U32.min(reach,extent) - low : U32))),Nat.min(U32.to_nat(reach),total),ends) :
        {Nat.is_lt(value,_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(reach),Nat.add(start,size),reach_exact) : {Nat.is_lt(value,Nat.min(_,total)) == True{} : Bool}
      NatExtrema.minimum_strict_lower(Nat.add(start,size),total,value,
        {%Equal.sym(Nat,Nat.add(start,size),Nat.mul(1n+quotient,size),Equal.sym(Nat,Nat.mul(1n+quotient,size),Nat.add(start,size),
            Equal.trans(Nat,Nat.mul(1n+quotient,size),Nat.add(size,start),Nat.add(start,size),{==},Order.commutative(size,start)))) :
            {Nat.is_lt(value,_) == True{} : Bool}
          DivisionOrder.next_above(value,size,positive) : {Nat.is_lt(value,Nat.add(start,size)) == True{} : Bool}},inside) :
      {Nat.is_lt(value,Nat.add(U32.to_nat(low),U32.to_nat((U32.min(reach,extent) - low : U32)))) == True{} : Bool}}}

# A footprint read from shift zero, unclamped, whose first index lies at or
# below its last and its last inside the bound, is not empty.
def placed_value(+first: U32,+last: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +first_below: {Nat.is_lt(U32.to_nat(first),U32.to_nat(bound)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(bound),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.placed(False{},first,last,0,bound,source_block) ==
    Tasks.Footprint{((U32.max(first,0) - 0) / U32.max(source_block,1) : U32),((U32.min(last,(0 + bound - 1 : U32)) - 0) / U32.max(source_block,1) : U32),False{}} :
    Tasks.Footprint}:
  +total = U32.to_nat(bound)
  +positive = Order.le_lt_trans(0n,U32.to_nat(first),total,Order.zero_le(U32.to_nat(first)),first_below)
  +top = (0 + bound - 1 : U32)
  +top_exact = {%Equal.sym(U32,(0 + bound : U32),bound,Arithmetic.add_zero_left(bound)) : {U32.to_nat((_ - 1 : U32)) == Nat.sub(total,1n) : Nat}
    Arithmetic.subtract(bound,1,cap,cap_ok,Order.strict_follow(0n,total,positive),Order.transitive(Nat.sub(total,1n),total,U32.to_nat(cap),Order.sub_le(total,1n),room)) :
    {U32.to_nat(top) == Nat.sub(total,1n) : Nat}}
  +zero = Arithmetic.word_positive_nonzero(bound,positive)
  +under = Equal.trans(Bool,U32.is_lt(last,0),Nat.is_lt(U32.to_nat(last),0n),False{},Comparison.lt(last,0),Division.lt_zero(U32.to_nat(last)))
  +above = Equal.trans(Bool,U32.is_lt(top,first),Nat.is_lt(U32.to_nat(top),U32.to_nat(first)),False{},Comparison.lt(top,first),
    Order.not_below(U32.to_nat(first),U32.to_nat(top),{%Equal.sym(Nat,U32.to_nat(top),Nat.sub(total,1n),top_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
      Components.pred_le(U32.to_nat(first),total,first_below) : {Nat.is_le(U32.to_nat(first),U32.to_nat(top)) == True{} : Bool}}))
  +low = ((U32.max(first,0) - 0) / U32.max(source_block,1) : U32)
  +high = ((U32.min(last,top) - 0) / U32.max(source_block,1) : U32)
  Equal.cong(Bool,Tasks.Footprint,empty => Tasks.Footprint{low,high,empty},(U32.is_zero(bound) || U32.is_lt(last,0) || U32.is_lt(top,first)),False{},
    {%Equal.sym(Bool,U32.is_zero(bound),False{},zero) : {(_ || U32.is_lt(last,0) || U32.is_lt(top,first)) == False{} : Bool}
      %Equal.sym(Bool,U32.is_lt(last,0),False{},under) : {(False{} || _ || U32.is_lt(top,first)) == False{} : Bool}
      %Equal.sym(Bool,U32.is_lt(top,first),False{},above) : {(False{} || False{} || _) == False{} : Bool}
      {==} : {(U32.is_zero(bound) || U32.is_lt(last,0) || U32.is_lt(top,first)) == False{} : Bool}})

# A value from the first to the last index a footprint keeps, below its
# bound, lies in the span of the footprint.
def footprint_in(+low_word: U32,+high_word: U32,+bound: U32,+source_block: U32,+coordinate: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +start: Nat,+end: Nat,+first_kept: {U32.to_nat(low_word) == start : Nat},+last_kept: {U32.to_nat(high_word) == end : Nat},
  +low_le: {Nat.is_le(start,value) == True{} : Bool},+value_end: {Nat.is_le(value,end) == True{} : Bool},+end_below: {Nat.is_lt(end,U32.to_nat(bound)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(bound),U32.to_nat(source_block)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(source_block)) == True{} : Bool}) ->
  InSpan(Tasks.box_span(Some{Tasks.Footprint{(low_word / U32.max(source_block,1) : U32),(high_word / U32.max(source_block,1) : U32),False{}}},bound,source_block,coordinate),value):
  +total = U32.to_nat(bound)
  +size = U32.to_nat(source_block)
  +bound_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +unit = U32.max(source_block,1)
  +unit_exact = Extrema.max_one_exact(source_block,size,{==},block_positive)
  +divisor_room = {%unit_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} block_room : {Nat.is_le(Nat.double(size),U32.to_nat(cap)) == True{} : Bool}}
  +value_below = Order.le_lt_trans(value,end,total,value_end,end_below)
  +end_room = Order.transitive(end,total,U32.to_nat(cap),Bounds.strict_le(end,total,end_below),bound_room)
  +ordered = Order.transitive(start,value,end,low_le,value_end)
  +low_value = Nat.div(start,size)
  +high_value = Nat.div(end,size)
  +low = (low_word / unit : U32)
  +high = (high_word / unit : U32)
  +low_exact = U32Division.quotient_exact(cap,cap_ok,low_word,unit,start,size,first_kept,unit_exact,divisor_room,block_positive)
  +high_exact = U32Division.quotient_exact(cap,cap_ok,high_word,unit,end,size,last_kept,unit_exact,divisor_room,block_positive)
  +low_below = DivisionOrder.product_below(start,size,block_positive)
  +scaled = (low * source_block : U32)
  +scaled_value = Nat.mul(low_value,size)
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,low,source_block,low_value,size,low_exact,{==},
    Order.transitive(scaled_value,start,U32.to_nat(cap),low_below,Order.transitive(start,end,U32.to_nat(cap),ordered,end_room)))
  +scaled_total = Order.transitive(scaled_value,start,total,low_below,Order.transitive(start,end,total,ordered,Bounds.strict_le(end,total,end_below)))
  +begin = U32.min(scaled,bound)
  +begin_exact = Equal.trans(Nat,U32.to_nat(begin),Nat.min(U32.to_nat(scaled),total),scaled_value,Extrema.minimum(scaled,bound),
    Equal.trans(Nat,Nat.min(U32.to_nat(scaled),total),Nat.min(scaled_value,total),scaled_value,
      Equal.cong(Nat,Nat,found => Nat.min(found,total),U32.to_nat(scaled),scaled_value,scaled_exact),NatExtrema.minimum_when_ordered(scaled_value,total,scaled_total)))
  +high_below = DivisionOrder.product_below(end,size,block_positive)
  +next_exact = Arithmetic.sum_exact(cap,cap_ok,high,1,high_value,1n,high_exact,{==},
    Order.transitive(Nat.add(high_value,1n),Nat.add(end,1n),U32.to_nat(cap),
      Order.add_le(high_value,end,1n,Order.transitive(high_value,Nat.mul(high_value,size),end,RowMajor.product_covers(high_value,size,block_positive),high_below)),
      Order.transitive(Nat.add(end,1n),Nat.add(total,size),U32.to_nat(cap),
        Order.transitive(Nat.add(end,1n),Nat.add(total,1n),Nat.add(total,size),Order.add_le(end,total,1n,Bounds.strict_le(end,total,end_below)),
          Order.offset_le(total,1n,size,Order.strict_follow(0n,size,block_positive))),room)))
  +next_value = Nat.mul(Nat.add(high_value,1n),size)
  +next_same = Equal.trans(Nat,Nat.mul(Nat.add(high_value,1n),size),Nat.mul(1n+high_value,size),Nat.add(size,Nat.mul(high_value,size)),
    Equal.cong(Nat,Nat,found => Nat.mul(found,size),Nat.add(high_value,1n),1n+high_value,Addition.successor_width(high_value)),{==})
  +next_room = {%Equal.sym(Nat,next_value,Nat.add(size,Nat.mul(high_value,size)),next_same) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(size,Nat.mul(high_value,size)),Nat.add(size,total),U32.to_nat(cap),
      Order.offset_le(size,Nat.mul(high_value,size),total,Order.transitive(Nat.mul(high_value,size),end,total,high_below,Bounds.strict_le(end,total,end_below))),
      {%Order.commutative(total,size) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(size,total),U32.to_nat(cap)) == True{} : Bool}}) :
    {Nat.is_le(next_value,U32.to_nat(cap)) == True{} : Bool}}
  +reach = ((high + 1 : U32) * source_block : U32)
  +reach_exact = Arithmetic.product_exact(cap,cap_ok,(high + 1 : U32),source_block,Nat.add(high_value,1n),size,next_exact,{==},next_room)
  +value_next = {%Equal.sym(Nat,next_value,Nat.add(size,Nat.mul(high_value,size)),next_same) : {Nat.is_lt(value,_) == True{} : Bool}
    Bounds.nat_trans(value,1n+end,Nat.add(size,Nat.mul(high_value,size)),Order.le_lt_trans(value,end,1n+end,value_end,Order.lt_succ(end,end,Order.reflexive(end))),
      Order.strict_follow(end,Nat.add(size,Nat.mul(high_value,size)),DivisionOrder.next_above(end,size,block_positive))) :
    {Nat.is_lt(value,next_value) == True{} : Bool}}
  +ordered_span = {%Equal.sym(Nat,U32.to_nat(begin),scaled_value,begin_exact) : {Nat.is_le(_,Nat.min(U32.to_nat(reach),total)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(reach),next_value,reach_exact) : {Nat.is_le(scaled_value,Nat.min(_,total)) == True{} : Bool}
    NatExtrema.minimum_lower(next_value,total,scaled_value,
      Order.transitive(scaled_value,value,next_value,Order.transitive(scaled_value,start,value,low_below,low_le),Bounds.strict_le(value,next_value,value_next)),
      scaled_total) :
    {Nat.is_le(U32.to_nat(begin),Nat.min(U32.to_nat(reach),total)) == True{} : Bool}}
  +ends = Components.span_sum(begin,reach,bound,cap,cap_ok,ordered_span,bound_room)
  Held.Both{{%Equal.sym(Nat,U32.to_nat(begin),scaled_value,begin_exact) : {Nat.is_le(_,value) == True{} : Bool}
      Order.transitive(scaled_value,start,value,low_below,low_le) : {Nat.is_le(U32.to_nat(begin),value) == True{} : Bool}},
    {%Equal.sym(Nat,Nat.add(U32.to_nat(begin),U32.to_nat((U32.min(reach,bound) - begin : U32))),Nat.min(U32.to_nat(reach),total),ends) :
        {Nat.is_lt(value,_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(reach),next_value,reach_exact) : {Nat.is_lt(value,Nat.min(_,total)) == True{} : Bool}
      NatExtrema.minimum_strict_lower(next_value,total,value,value_next,value_below) :
      {Nat.is_lt(value,Nat.add(U32.to_nat(begin),U32.to_nat((U32.min(reach,bound) - begin : U32)))) == True{} : Bool}}}

# A value below the bound, from the first to the last virtual index, lies in
# the span of the footprint.
def placed_in(+first: U32,+last: U32,+bound: U32,+source_block: U32,+coordinate: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_le: {Nat.is_le(U32.to_nat(first),value) == True{} : Bool},+high_ge: {Nat.is_le(value,U32.to_nat(last)) == True{} : Bool},
  +value_below: {Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(bound),U32.to_nat(source_block)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(source_block)) == True{} : Bool}) ->
  InSpan(Tasks.box_span(Some{Tasks.placed(False{},first,last,0,bound,source_block)},bound,source_block,coordinate),value):
  +total = U32.to_nat(bound)
  +size = U32.to_nat(source_block)
  +start = U32.to_nat(first)
  +bound_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +top = (0 + bound - 1 : U32)
  +unit = U32.max(source_block,1)
  +first_below = Order.le_lt_trans(start,value,total,low_le,value_below)
  +same = placed_value(first,last,bound,source_block,cap,cap_ok,first_below,bound_room)
  +low = ((U32.max(first,0) - 0) / unit : U32)
  +high = ((U32.min(last,top) - 0) / unit : U32)
  +positive = Order.le_lt_trans(0n,value,total,Order.zero_le(value),value_below)
  +top_exact = {%Equal.sym(U32,(0 + bound : U32),bound,Arithmetic.add_zero_left(bound)) : {U32.to_nat((_ - 1 : U32)) == Nat.sub(total,1n) : Nat}
    Arithmetic.subtract(bound,1,cap,cap_ok,Order.strict_follow(0n,total,positive),Order.transitive(Nat.sub(total,1n),total,U32.to_nat(cap),Order.sub_le(total,1n),bound_room)) :
    {U32.to_nat(top) == Nat.sub(total,1n) : Nat}}
  +end = Nat.min(U32.to_nat(last),Nat.sub(total,1n))
  +first_kept = Equal.trans(Nat,U32.to_nat((U32.max(first,0) - 0 : U32)),U32.to_nat(U32.max(first,0)),start,
    Equal.cong(U32,Nat,word => U32.to_nat(word),(U32.max(first,0) - 0 : U32),U32.max(first,0),Arithmetic.subtract_zero(U32.max(first,0))),
    Equal.cong(U32,Nat,word => U32.to_nat(word),U32.max(first,0),first,Laws.maximum_kept(first,0,Order.zero_le(start))))
  +last_kept = Equal.trans(Nat,U32.to_nat((U32.min(last,top) - 0 : U32)),U32.to_nat(U32.min(last,top)),end,
    Equal.cong(U32,Nat,word => U32.to_nat(word),(U32.min(last,top) - 0 : U32),U32.min(last,top),Arithmetic.subtract_zero(U32.min(last,top))),
    Equal.trans(Nat,U32.to_nat(U32.min(last,top)),Nat.min(U32.to_nat(last),U32.to_nat(top)),end,Extrema.minimum(last,top),
      Equal.cong(Nat,Nat,found => Nat.min(U32.to_nat(last),found),U32.to_nat(top),Nat.sub(total,1n),top_exact)))
  +value_end = NatExtrema.minimum_lower(U32.to_nat(last),Nat.sub(total,1n),value,high_ge,Components.pred_le(value,total,value_below))
  +end_below = Order.le_lt_trans(end,Nat.sub(total,1n),total,NatExtrema.minimum_right(U32.to_nat(last),Nat.sub(total,1n)),Order.last_below(total,positive))
  %Equal.sym(Tasks.Footprint,Tasks.placed(False{},first,last,0,bound,source_block),Tasks.Footprint{low,high,False{}},same) :
    InSpan(Tasks.box_span(Some{_},bound,source_block,coordinate),value)
  footprint_in((U32.max(first,0) - 0 : U32),(U32.min(last,top) - 0 : U32),bound,source_block,coordinate,value,cap,cap_ok,start,end,first_kept,last_kept,low_le,value_end,
    end_below,room,block_room,block_positive)

# A word divided by one keeps its value.
def by_one(+value: U32,+value_nat: Nat,+exact: {U32.to_nat(value) == value_nat : Nat}) -> {U32.to_nat((value / U32.max(1,1) : U32)) == value_nat : Nat}:
  Equal.trans(Nat,U32.to_nat((value / 1 : U32)),U32.to_nat(value),value_nat,U32Division.divided_by_one(value),exact)

# A word widened by one and divided by one keeps its value.
def last_by_one(+value: U32,+value_nat: Nat,+exact: {U32.to_nat(value) == value_nat : Nat}) -> {U32.to_nat(((value + 1 - 1 : U32) / U32.max(1,1) : U32)) == value_nat : Nat}:
  +same = Equal.trans(U32,((value + 1 : U32) - 1 : U32),(value + 0 : U32),value,Arithmetic.subtract_from_sum(value,1,1),Arithmetic.add_zero(value))
  %Equal.sym(U32,((value + 1 : U32) - 1 : U32),value,same) : {U32.to_nat((_ / U32.max(1,1) : U32)) == value_nat : Nat}
  by_one(value,value_nat,exact)

# The last index a view read reaches stays inside the source axis.
def ReadInside(mirrored: Bool,+step: U32,+offset: U32,+high: U32,+bound: U32) -> Data:
  match mirrored:
    case False{}:
      {Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),U32.to_nat(high))),U32.to_nat(bound)) == True{} : Bool}
    case True{}:
      Held.Both<{Nat.is_le(Nat.mul(U32.to_nat(step),U32.to_nat(high)),U32.to_nat(offset)) == True{} : Bool},{Nat.is_lt(U32.to_nat(offset),U32.to_nat(bound)) == True{} : Bool}>

# A target coordinate inside the range of a view read reads a source index
# inside the span of the footprint of the range.
def read_in(mirrored: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+bound: U32,+low: U32,+high: U32,+fixed: U32,+coordinate: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_le: {Nat.is_le(U32.to_nat(low),coordinate) == True{} : Bool},+high_ge: {Nat.is_le(coordinate,U32.to_nat(high)) == True{} : Bool},
  +inside: ReadInside(mirrored,step,offset,high,bound),
  +room: {Nat.is_le(Nat.add(U32.to_nat(bound),U32.to_nat(source_block)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(source_block)) == True{} : Bool}) ->
  InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,1,1,0,bound,False{}},low,high)},bound,source_block,fixed),
    read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate)):
  match mirrored:
    case False{}:
      +origin = U32.to_nat(offset)
      +near = Nat.mul(U32.to_nat(step),U32.to_nat(low))
      +here = Nat.mul(U32.to_nat(step),coordinate)
      +far = Nat.mul(U32.to_nat(step),U32.to_nat(high))
      +total = U32.to_nat(bound)
      +bound_room = Order.transitive(total,Nat.add(total,U32.to_nat(source_block)),U32.to_nat(cap),Order.le_add(total,U32.to_nat(source_block)),room)
      +near_here = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(low),coordinate,low_le)
      +here_far = DivisionOrder.multiply_left_le(U32.to_nat(step),coordinate,U32.to_nat(high),high_ge)
      +far_sum = Nat.add(origin,far)
      +far_room = Order.transitive(far_sum,total,U32.to_nat(cap),Bounds.strict_le(far_sum,total,inside),bound_room)
      +near_room = Order.transitive(Nat.add(origin,near),far_sum,U32.to_nat(cap),
        Order.offset_le(origin,near,far,Order.transitive(near,here,far,near_here,here_far)),far_room)
      +first = ((offset + step * low : U32) / U32.max(1,1) : U32)
      +last = (((offset + step * high : U32) + 1 - 1 : U32) / U32.max(1,1) : U32)
      +first_exact = by_one((offset + step * low : U32),Nat.add(origin,near),Components.forward_sum(cap,cap_ok,step,offset,low,near_room))
      +last_exact = last_by_one((offset + step * high : U32),far_sum,Components.forward_sum(cap,cap_ok,step,offset,high,far_room))
      placed_in(first,last,bound,source_block,fixed,Nat.add(origin,here),cap,cap_ok,
        {%Equal.sym(Nat,U32.to_nat(first),Nat.add(origin,near),first_exact) : {Nat.is_le(_,Nat.add(origin,here)) == True{} : Bool}
          Order.offset_le(origin,near,here,near_here) : {Nat.is_le(U32.to_nat(first),Nat.add(origin,here)) == True{} : Bool}},
        {%Equal.sym(Nat,U32.to_nat(last),far_sum,last_exact) : {Nat.is_le(Nat.add(origin,here),_) == True{} : Bool}
          Order.offset_le(origin,here,far,here_far) : {Nat.is_le(Nat.add(origin,here),U32.to_nat(last)) == True{} : Bool}},
        Order.le_lt_trans(Nat.add(origin,here),far_sum,total,Order.offset_le(origin,here,far,here_far),inside),
        room,block_room,block_positive)
    case True{}:
      Held.Both{+reach,+below} = inside
      +origin = U32.to_nat(offset)
      +near = Nat.mul(U32.to_nat(step),U32.to_nat(low))
      +here = Nat.mul(U32.to_nat(step),coordinate)
      +far = Nat.mul(U32.to_nat(step),U32.to_nat(high))
      +total = U32.to_nat(bound)
      +bound_room = Order.transitive(total,Nat.add(total,U32.to_nat(source_block)),U32.to_nat(cap),Order.le_add(total,U32.to_nat(source_block)),room)
      +near_here = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(low),coordinate,low_le)
      +here_far = DivisionOrder.multiply_left_le(U32.to_nat(step),coordinate,U32.to_nat(high),high_ge)
      +origin_room = Order.transitive(origin,total,U32.to_nat(cap),Bounds.strict_le(origin,total,below),bound_room)
      +near_inside = Order.transitive(near,far,origin,Order.transitive(near,here,far,near_here,here_far),reach)
      +first = ((offset - step * high : U32) / U32.max(1,1) : U32)
      +last = (((offset - step * low : U32) + 1 - 1 : U32) / U32.max(1,1) : U32)
      +first_exact = by_one((offset - step * high : U32),Nat.sub(origin,far),Components.backward_difference(cap,cap_ok,step,offset,high,reach,origin_room))
      +last_exact = last_by_one((offset - step * low : U32),Nat.sub(origin,near),Components.backward_difference(cap,cap_ok,step,offset,low,near_inside,origin_room))
      placed_in(first,last,bound,source_block,fixed,Nat.sub(origin,here),cap,cap_ok,
        {%Equal.sym(Nat,U32.to_nat(first),Nat.sub(origin,far),first_exact) : {Nat.is_le(_,Nat.sub(origin,here)) == True{} : Bool}
          Order.sub_antitone(origin,here,far,here_far) : {Nat.is_le(U32.to_nat(first),Nat.sub(origin,here)) == True{} : Bool}},
        {%Equal.sym(Nat,U32.to_nat(last),Nat.sub(origin,near),last_exact) : {Nat.is_le(Nat.sub(origin,here),_) == True{} : Bool}
          Order.sub_antitone(origin,near,here,near_here) : {Nat.is_le(Nat.sub(origin,here),U32.to_nat(last)) == True{} : Bool}},
        Order.le_lt_trans(Nat.sub(origin,here),origin,total,Order.sub_le(origin,here),below),
        room,block_room,block_positive)

# ---- The digits a target cell reads lie in the source box ----

# A view read reads below its bound.
def read_below(mirrored: Bool,+step: U32,+offset: U32,+high: U32,+bound: U32,+coordinate: Nat,
  +high_ge: {Nat.is_le(coordinate,U32.to_nat(high)) == True{} : Bool},+inside: ReadInside(mirrored,step,offset,high,bound)) ->
  {Nat.is_lt(read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate),U32.to_nat(bound)) == True{} : Bool}:
  match mirrored:
    case False{}:
      Order.le_lt_trans(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),U32.to_nat(high))),U32.to_nat(bound),
        Order.offset_le(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate),Nat.mul(U32.to_nat(step),U32.to_nat(high)),
          DivisionOrder.multiply_left_le(U32.to_nat(step),coordinate,U32.to_nat(high),high_ge)),inside)
    case True{}:
      Held.Both{+reach,+below} = inside
      Order.le_lt_trans(Nat.sub(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),U32.to_nat(offset),U32.to_nat(bound),
        Order.sub_le(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),below)

# A target axis that reads a source axis reads it as a view does, from a
# coordinate inside its class.
def ReadIn(reading: Tasks.Reading,class: Tasks.Class,+coordinate: Nat) -> Data:
  match reading class:
    case Tasks.Reading{+extent,+block,Tasks.Read{other,source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}} Tasks.Class{+first,+count,source_first,source_count}:
      Held.Both<{width == 1 : U32},Held.Both<{divisor == 1 : U32},Held.Both<{shift == 0 : U32},Held.Both<{clamped == False{} : Bool},
        Held.Both<{Nat.is_le(U32.to_nat((first * block : U32)),coordinate) == True{} : Bool},
          Held.Both<{Nat.is_le(coordinate,U32.to_nat((U32.min(((first + count) * block : U32),extent) - 1 : U32))) == True{} : Bool},
            ReadInside(mirrored,step,offset,(U32.min(((first + count) * block : U32),extent) - 1 : U32),bound)>>>>>>
    case reading class:
      Unit

# Every target axis of a row reads as a view does, at its class and coordinate.
def RowIn(readings: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,coordinates: List<&2,Nat>) -> Data:
  match readings classes coordinates:
    case Con{+reading,readings} Con{+class,classes} Con{+coordinate,coordinates}:
      Held.Both<ReadIn(reading,class,coordinate),RowIn(readings,classes,coordinates)>
    case Nil{} Nil{} Nil{}:
      Unit
    case readings classes coordinates:
      Empty

# The source axis a target axis reads, when it reads it, holds the read digit.
def axis_found(hit: Bool,+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +target_extent: U32,+target_block: U32,+first: U32,+count: U32,+source_first: U32,+source_count: U32,+later_found: Maybe<&2,Tasks.Footprint>,+later_digit: Nat,
  +coordinate: Nat,+grid: Grid.Grid,+axis: U32,+extent: U32,+fixed: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +here: ReadIn(Tasks.Reading{target_extent,target_block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},
    Tasks.Class{first,count,source_first,source_count},coordinate),
  +facts: Components.AxisFacts(hit,source_block,bound,grid,axis,extent),
  +later: AxisIn(Tasks.box_span(later_found,extent,Grid.block(grid,axis,extent),fixed),later_digit,extent),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,extent),1))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,axis,extent))) == True{} : Bool}) ->
  AxisIn(Tasks.box_span(Tasks.found_footprint(hit,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},target_extent,target_block,first,count,
      later_found),extent,Grid.block(grid,axis,extent),fixed),
    Bool.pick(Nat,hit,read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate),later_digit),extent):
  match hit:
    case True{}:
      Held.Both{+width_one,+here_divisor} = here
      Held.Both{+divisor_one,+here_shift} = here_divisor
      Held.Both{+shift_zero,+here_clamped} = here_shift
      Held.Both{+unclamped,+here_low} = here_clamped
      Held.Both{+low_le,+here_high} = here_low
      Held.Both{+high_ge,+inside} = here_high
      Held.Both{+same_block,+same_bound} = facts
      +block = Grid.block(grid,axis,extent)
      +low = (first * target_block : U32)
      +high = (U32.min(((first + count) * target_block : U32),target_extent) - 1 : U32)
      +value = read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate)
      +inside_extent = {%same_bound : ReadInside(mirrored,step,offset,high,_) inside : ReadInside(mirrored,step,offset,high,extent)}
      +spanned = read_in(mirrored,other,block,step,offset,extent,low,high,fixed,coordinate,cap,cap_ok,low_le,high_ge,inside_extent,room,
        block_room,positive)
      Held.Both{
        {%Equal.sym(U32,width,1,width_one) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,_,divisor,shift,bound,clamped},low,high)},extent,block,fixed),value)
          %Equal.sym(U32,divisor,1,divisor_one) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,1,_,shift,bound,clamped},low,high)},extent,block,fixed),value)
          %Equal.sym(U32,shift,0,shift_zero) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,1,1,_,bound,clamped},low,high)},extent,block,fixed),value)
          %Equal.sym(Bool,clamped,False{},unclamped) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,1,1,0,bound,_},low,high)},extent,block,fixed),value)
          %Equal.sym(U32,source_block,block,same_block) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,_,step,offset,mirrored,1,1,0,bound,False{}},low,high)},extent,block,fixed),value)
          %Equal.sym(U32,bound,extent,same_bound) :
            InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,block,step,offset,mirrored,1,1,0,_,False{}},low,high)},extent,block,fixed),value)
          spanned :
          InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},low,high)},extent,block,fixed),value)},
        read_below(mirrored,step,offset,high,extent,coordinate,high_ge,inside_extent)}
    case False{}:
      later

# The source axis axis holds the digit a target cell reads from it.
def axis_in(readings: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,coordinates: List<&2,Nat>,+grid: Grid.Grid,+axis: U32,+extent: U32,+fixed: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+row: RowIn(readings,classes,coordinates),+reads: Components.ReadsAxis(readings,grid,axis,extent),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,extent),1))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,axis,extent))) == True{} : Bool},
  +fixed_in: {Nat.is_lt(U32.to_nat(fixed),U32.to_nat(extent)) == True{} : Bool}) ->
  AxisIn(Tasks.box_span(Tasks.read_footprint(readings,classes,axis),extent,Grid.block(grid,axis,extent),fixed),read_digit(readings,coordinates,axis,U32.to_nat(fixed)),extent):
  match readings classes coordinates:
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest}
        Con{Tasks.Class{+first,+count,+source_first,+source_count},+others} Con{+coordinate,+more}:
      Held.Both{+here,+later} = row
      Held.Both{+facts,+reads_rest} = reads
      axis_found(U32.is_eq(other,axis),other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,target_extent,target_block,first,count,source_first,source_count,
        Tasks.read_footprint(rest,others,axis),read_digit(rest,more,axis,U32.to_nat(fixed)),coordinate,grid,axis,extent,fixed,cap,cap_ok,here,facts,
        axis_in(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads_rest,room,block_room,positive,fixed_in),room,block_room,positive)
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Unread{}},+rest} Con{+class,+others} Con{+coordinate,+more}:
      Held.Both{+here,+later} = row
      axis_in(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads,room,block_room,positive,fixed_in)
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Crossing{}},+rest} Con{+class,+others} Con{+coordinate,+more}:
      Held.Both{+here,+later} = row
      axis_in(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads,room,block_room,positive,fixed_in)
    case Nil{} Nil{} Nil{}:
      +block = Grid.block(grid,axis,extent)
      Held.Both{none_in(extent,block,fixed,cap,cap_ok,room,block_room,positive,fixed_in),fixed_in}
    case Nil{} Nil{} Con{+coordinate,+more}:
      Empty.absurd(AxisIn(Tasks.box_span(Tasks.read_footprint(Nil{},Nil{},axis),extent,Grid.block(grid,axis,extent),fixed),
        read_digit(Nil{},Con{coordinate,more},axis,U32.to_nat(fixed)),extent),row)
    case Nil{} Con{+class,+others} +coordinates:
      Empty.absurd(AxisIn(Tasks.box_span(Tasks.read_footprint(Nil{},Con{class,others},axis),extent,Grid.block(grid,axis,extent),fixed),
        read_digit(Nil{},coordinates,axis,U32.to_nat(fixed)),extent),row)
    case Con{+reading,+rest} Nil{} +coordinates:
      Empty.absurd(AxisIn(Tasks.box_span(Tasks.read_footprint(Con{reading,rest},Nil{},axis),extent,Grid.block(grid,axis,extent),fixed),
        read_digit(Con{reading,rest},coordinates,axis,U32.to_nat(fixed)),extent),row)
    case Con{+reading,+rest} Con{+class,+others} Nil{}:
      Empty.absurd(AxisIn(Tasks.box_span(Tasks.read_footprint(Con{reading,rest},Con{class,others},axis),extent,Grid.block(grid,axis,extent),fixed),
        read_digit(Con{reading,rest},Nil{},axis,U32.to_nat(fixed)),extent),row)

# Every source axis has a positive block that twice fits the word.
def BlocksIn(+grid: Grid.Grid,extents: List<&2,U32>,+at: U32,+cap: U32) -> Data:
  match extents:
    case Nil{}:
      Unit
    case Con{+extent,rest}:
      Held.Both<{Nat.is_lt(0n,U32.to_nat(Grid.block(grid,at,extent))) == True{} : Bool},
        Held.Both<{Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,at,extent),1))),U32.to_nat(cap)) == True{} : Bool},BlocksIn(grid,rest,(at + 1 : U32),cap)>>

# Every fixed index lies below its extent.
def FixedIn(fixed: List<&2,U32>,extents: List<&2,U32>) -> Data:
  match fixed extents:
    case Con{+index,fixed} Con{+extent,extents}:
      Held.Both<{Nat.is_lt(U32.to_nat(index),U32.to_nat(extent)) == True{} : Bool},FixedIn(fixed,extents)>
    case Nil{} Nil{}:
      Unit
    case fixed extents:
      Empty

# Every source axis holds the digit a target cell reads from it.
def digits_in(+readings: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+coordinates: List<&2,Nat>,+grid: Grid.Grid,extents: List<&2,U32>,fixed: List<&2,U32>,
  +at: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+row: RowIn(readings,classes,coordinates),
  +axes: Components.AxesOk(readings,grid,extents,at,cap),+blocks: BlocksIn(grid,extents,at,cap),+fixed_in: FixedIn(fixed,extents)) ->
  DigitsIn(Tasks.boxed(readings,classes,grid,extents,fixed,at),extents,source_digits(readings,coordinates,extents,fixed,at)):
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      Held.Both{+here,+later} = axes
      Held.Both{+reads,+room} = here
      Held.Both{+positive,+blocks_rest} = blocks
      Held.Both{+block_room,+blocks_more} = blocks_rest
      Held.Both{+index_in,+fixed_rest} = fixed_in
      Held.Both{axis_in(readings,classes,coordinates,grid,at,extent,index,cap,cap_ok,row,reads,room,block_room,positive,index_in),
        digits_in(readings,classes,coordinates,grid,rest,more,(at + 1 : U32),cap,cap_ok,row,later,blocks_more,fixed_rest)}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+index,+more}:
      Empty.absurd(DigitsIn(Tasks.boxed(readings,classes,grid,Nil{},Con{index,more},at),Nil{},source_digits(readings,coordinates,Nil{},Con{index,more},at)),fixed_in)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(DigitsIn(Tasks.boxed(readings,classes,grid,Con{extent,rest},Nil{},at),Con{extent,rest},source_digits(readings,coordinates,Con{extent,rest},Nil{},at)),fixed_in)

# ---- The number the digits of a target cell make ----

# The left of an or that does not hold does not hold.
def left_false(a: Bool,+b: Bool,+none: {(a || b) == False{} : Bool}) -> {a == False{} : Bool}:
  match a:
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},none)))
    case False{}: {==}

# The right of an or that does not hold does not hold.
def right_false(a: Bool,+b: Bool,+none: {(a || b) == False{} : Bool}) -> {b == False{} : Bool}:
  match a:
    case True{}: Empty.absurd({b == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},none)))
    case False{}: none

# A source axis no target axis reads keeps its fixed index.
def read_unheld(readings: List<&2,Tasks.Reading>,coordinates: List<&2,Nat>,+axis: U32,+fixed: Nat,
  +unheld: {Tasks.holds(Tasks.read_axes(readings),axis) == False{} : Bool}) -> {read_digit(readings,coordinates,axis,fixed) == fixed : Nat}:
  match readings coordinates:
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Con{+coordinate,+more}:
      +others = Tasks.holds(Tasks.read_axes(rest),axis)
      +missed = left_false(U32.is_eq(other,axis),others,unheld)
      +later = read_unheld(rest,more,axis,fixed,right_false(U32.is_eq(other,axis),others,unheld))
      %Equal.sym(Bool,U32.is_eq(other,axis),False{},missed) :
        {Bool.pick(Nat,_,read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate),read_digit(rest,more,axis,fixed)) == fixed : Nat}
      later
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} Con{+coordinate,+more}:
      read_unheld(rest,more,axis,fixed,unheld)
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} Con{+coordinate,+more}:
      read_unheld(rest,more,axis,fixed,unheld)
    case Nil{} +coordinates:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Nil{}:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} Nil{}:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} Nil{}:
      {==}

# The digits with the one of source axis axis replaced by value.
def put(digits: List<&2,Nat>,+at: U32,+axis: U32,+value: Nat) -> List<&2,Nat>:
  match digits:
    case Nil{}: Nil{}
    case Con{+digit,+rest}: Con{Bool.pick(Nat,U32.is_eq(axis,at),value,digit),put(rest,(at + 1 : U32),axis,value)}

# The digit of source axis axis, or zero.
def digit_at(digits: List<&2,Nat>,+at: U32,+axis: U32) -> Nat:
  match digits:
    case Nil{}: 0n
    case Con{+digit,+rest}: Bool.pick(Nat,U32.is_eq(axis,at),digit,digit_at(rest,(at + 1 : U32),axis))

# The size of the axes after source axis axis, or zero.
def axis_weight(extents: List<&2,U32>,digits: List<&2,Nat>,+at: U32,+axis: U32) -> Nat:
  match extents digits:
    case Con{extent,+rest} Con{digit,+more}: Bool.pick(Nat,U32.is_eq(axis,at),ShapeLaws.elements(rest),axis_weight(rest,more,(at + 1 : U32),axis))
    case extents digits: 0n

# A target axis that reads source axis other replaces its digit.
def digits_put(+extent: U32,+block: U32,+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,
  +clamped: Bool,+rest: List<&2,Tasks.Reading>,+coordinate: Nat,+more: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {source_digits(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},Con{coordinate,more},extents,fixed,at) ==
    put(source_digits(rest,more,extents,fixed,at),at,other,read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate)) : List<&2,Nat>}:
  match extents fixed:
    case Con{+size,+others} Con{+index,+indices}:
      +value = read_value(mirrored,U32.to_nat(offset),U32.to_nat(step),coordinate)
      +head = Bool.pick(Nat,U32.is_eq(other,at),value,read_digit(rest,more,at,U32.to_nat(index)))
      +next = (at + 1 : U32)
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{head,tail},
        source_digits(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},Con{coordinate,more},others,indices,next),
        put(source_digits(rest,more,others,indices,next),next,other,value),
        digits_put(extent,block,other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,rest,coordinate,more,others,indices,next))
    case Nil{} +fixed:
      {==}
    case Con{+size,+others} Nil{}:
      {==}

# A target axis that reads no source axis keeps every digit.
def digits_unread(+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+coordinate: Nat,+more: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {source_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{coordinate,more},extents,fixed,at) == source_digits(rest,more,extents,fixed,at) : List<&2,Nat>}:
  match extents fixed:
    case Con{+size,+others} Con{+index,+indices}:
      +next = (at + 1 : U32)
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{read_digit(rest,more,at,U32.to_nat(index)),tail},
        source_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{coordinate,more},others,indices,next),source_digits(rest,more,others,indices,next),
        digits_unread(extent,block,rest,coordinate,more,others,indices,next))
    case Nil{} +fixed:
      {==}
    case Con{+size,+others} Nil{}:
      {==}

# A source axis before the first digit leaves the digits as they are.
def put_before(digits: List<&2,Nat>,+at: U32,+position: Nat,+axis: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+room: {Nat.is_le(Nat.add(position,List.length(&2,Nat,digits)),U32.to_nat(cap)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(axis),position) == True{} : Bool}) -> {put(digits,at,axis,value) == digits : List<&2,Nat>}:
  match digits:
    case Nil{}:
      {==}
    case Con{+digit,+rest}:
      +missed = Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),False{},Comparison.eq(axis,at),
        {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(U32.to_nat(axis),_) == False{} : Bool}
          Order.less_not_equal(U32.to_nat(axis),position,before) : {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == False{} : Bool}})
      +length = List.length(&2,Nat,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
      +next_exact = Components.next_word(at,position,cap,cap_ok,exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),U32.to_nat(cap),Order.le_add(1n+position,length),shifted))
      +later = put_before(rest,(at + 1 : U32),1n+position,axis,value,cap,cap_ok,next_exact,shifted,Order.lt_succ(U32.to_nat(axis),position,Bounds.strict_le(U32.to_nat(axis),position,before)))
      %Equal.sym(Bool,U32.is_eq(axis,at),False{},missed) : {Con{Bool.pick(Nat,_,value,digit),put(rest,(at + 1 : U32),axis,value)} == Con{digit,rest} : List<&2,Nat>}
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{digit,tail},put(rest,(at + 1 : U32),axis,value),rest,later)

# Moving the first of three sums to the end of two.
def outer_swap(+a: Nat,+middle: Nat,+b: Nat) -> {Nat.add(Nat.add(a,middle),b) == Nat.add(Nat.add(b,middle),a) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,middle),b),Nat.add(a,Nat.add(middle,b)),Nat.add(Nat.add(b,middle),a),Addition.associative(a,middle,b),
    Equal.trans(Nat,Nat.add(a,Nat.add(middle,b)),Nat.add(a,Nat.add(b,middle)),Nat.add(Nat.add(b,middle),a),
      Equal.cong(Nat,Nat,sum => Nat.add(a,sum),Nat.add(middle,b),Nat.add(b,middle),Order.commutative(middle,b)),
      Equal.trans(Nat,Nat.add(a,Nat.add(b,middle)),Nat.add(b,Nat.add(a,middle)),Nat.add(Nat.add(b,middle),a),RowMajor.add_swap(a,b,middle),
        Equal.trans(Nat,Nat.add(b,Nat.add(a,middle)),Nat.add(b,Nat.add(middle,a)),Nat.add(Nat.add(b,middle),a),
          Equal.cong(Nat,Nat,sum => Nat.add(b,sum),Nat.add(a,middle),Nat.add(middle,a),Order.commutative(a,middle)),
          Equal.sym(Nat,Nat.add(Nat.add(b,middle),a),Nat.add(b,Nat.add(middle,a)),Addition.associative(b,middle,a))))))

# One digit of the number, replaced or kept.
def put_found(hit: Bool,+extent: U32,+rest: List<&2,U32>,+digit: Nat,+more: List<&2,Nat>,+at: U32,+position: Nat,+axis: U32,+value: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+same: {U32.is_eq(axis,at) == hit : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(1n+Nat.add(position,List.length(&2,Nat,more)),U32.to_nat(cap)) == True{} : Bool},
  +next_exact: {U32.to_nat((at + 1 : U32)) == 1n+position : Nat},
  +later: {Nat.add(ShapeLaws.digits_value(rest,put(more,(at + 1 : U32),axis,value)),Nat.mul(digit_at(more,(at + 1 : U32),axis),axis_weight(rest,more,(at + 1 : U32),axis))) ==
    Nat.add(ShapeLaws.digits_value(rest,more),Nat.mul(value,axis_weight(rest,more,(at + 1 : U32),axis))) : Nat}) ->
  {Nat.add(ShapeLaws.digits_value(Con{extent,rest},Con{Bool.pick(Nat,hit,value,digit),put(more,(at + 1 : U32),axis,value)}),
      Nat.mul(Bool.pick(Nat,hit,digit,digit_at(more,(at + 1 : U32),axis)),Bool.pick(Nat,hit,ShapeLaws.elements(rest),axis_weight(rest,more,(at + 1 : U32),axis)))) ==
    Nat.add(ShapeLaws.digits_value(Con{extent,rest},Con{digit,more}),Nat.mul(value,Bool.pick(Nat,hit,ShapeLaws.elements(rest),axis_weight(rest,more,(at + 1 : U32),axis)))) : Nat}:
  match hit:
    case True{}:
      +inner = ShapeLaws.elements(rest)
      +tail = ShapeLaws.digits_value(rest,more)
      +equal_axis = Arithmetic.equal_of_is_eq(axis,at,same)
      +before = {%Equal.sym(U32,axis,at,equal_axis) : {Nat.is_lt(U32.to_nat(_),1n+position) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_lt(_,1n+position) == True{} : Bool}
        Order.lt_succ(position,position,Order.reflexive(position)) : {Nat.is_lt(U32.to_nat(axis),1n+position) == True{} : Bool}}
      +kept = put_before(more,(at + 1 : U32),1n+position,axis,value,cap,cap_ok,next_exact,room,before)
      %Equal.sym(List<&2,Nat>,put(more,(at + 1 : U32),axis,value),more,kept) :
        {Nat.add(Nat.add(Nat.mul(value,inner),ShapeLaws.digits_value(rest,_)),Nat.mul(digit,inner)) == Nat.add(Nat.add(Nat.mul(digit,inner),tail),Nat.mul(value,inner)) : Nat}
      outer_swap(Nat.mul(value,inner),tail,Nat.mul(digit,inner))
    case False{}:
      +head = Nat.mul(digit,ShapeLaws.elements(rest))
      +changed = ShapeLaws.digits_value(rest,put(more,(at + 1 : U32),axis,value))
      +tail = ShapeLaws.digits_value(rest,more)
      +old_part = Nat.mul(digit_at(more,(at + 1 : U32),axis),axis_weight(rest,more,(at + 1 : U32),axis))
      +new_part = Nat.mul(value,axis_weight(rest,more,(at + 1 : U32),axis))
      Equal.trans(Nat,Nat.add(Nat.add(head,changed),old_part),Nat.add(head,Nat.add(changed,old_part)),Nat.add(Nat.add(head,tail),new_part),
        Addition.associative(head,changed,old_part),
        Equal.trans(Nat,Nat.add(head,Nat.add(changed,old_part)),Nat.add(head,Nat.add(tail,new_part)),Nat.add(Nat.add(head,tail),new_part),
          Equal.cong(Nat,Nat,sum => Nat.add(head,sum),Nat.add(changed,old_part),Nat.add(tail,new_part),later),
          Equal.sym(Nat,Nat.add(Nat.add(head,tail),new_part),Nat.add(head,Nat.add(tail,new_part)),Addition.associative(head,tail,new_part))))

# Replacing the digit of a source axis by value moves the number by the
# difference times the weight of the axis.
def put_value(extents: List<&2,U32>,digits: List<&2,Nat>,+at: U32,+position: Nat,+axis: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+room: {Nat.is_le(Nat.add(position,List.length(&2,Nat,digits)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.add(ShapeLaws.digits_value(extents,put(digits,at,axis,value)),Nat.mul(digit_at(digits,at,axis),axis_weight(extents,digits,at,axis))) ==
    Nat.add(ShapeLaws.digits_value(extents,digits),Nat.mul(value,axis_weight(extents,digits,at,axis))) : Nat}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      +length = List.length(&2,Nat,more)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
      +next_exact = Components.next_word(at,position,cap,cap_ok,exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),U32.to_nat(cap),Order.le_add(1n+position,length),shifted))
      put_found(U32.is_eq(axis,at),extent,rest,digit,more,at,position,axis,value,cap,cap_ok,{==},exact,shifted,next_exact,
        put_value(rest,more,(at + 1 : U32),1n+position,axis,value,cap,cap_ok,next_exact,shifted))
    case Nil{} +digits:
      Equal.trans(Nat,Nat.mul(digit_at(digits,at,axis),0n),0n,Nat.mul(value,0n),RowMajor.multiply_zero_right(digit_at(digits,at,axis)),
        Equal.sym(Nat,Nat.mul(value,0n),0n,RowMajor.multiply_zero_right(value)))
    case Con{+extent,+rest} Nil{}:
      Equal.sym(Nat,Nat.mul(value,0n),0n,RowMajor.multiply_zero_right(value))

# ---- The fixed digits and the axis a stride steps ----

# A natural number below one is zero.
def below_one(value: Nat,+small: {Nat.is_lt(value,1n) == True{} : Bool}) -> {0n == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +rest: Empty.absurd({0n == 1n+rest : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

# A zero word has the value zero.
def zero_value(value: Nat,+word: U32,+exact: {U32.to_nat(word) == value : Nat},+zero: {U32.is_zero(word) == True{} : Bool}) -> {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      +nonzero = Arithmetic.word_positive_nonzero(word,{%Equal.sym(Nat,U32.to_nat(word),1n+rest,exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
        {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}})
      Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_zero(word),True{},Equal.sym(Bool,U32.is_zero(word),False{},nonzero),zero)))

# The left of an and that holds holds.
def and_left(a: Bool,+b: Bool,+both: {(a && b) == True{} : Bool}) -> {a == True{} : Bool}:
  match a:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(both))

# The right of an and that holds holds.
def and_right(a: Bool,+b: Bool,+both: {(a && b) == True{} : Bool}) -> {b == True{} : Bool}:
  match a:
    case True{}: both
    case False{}: Empty.absurd({b == True{} : Bool},WordEncoding.false_true(both))

# A word equals itself.
def word_same(+word: U32) -> {U32.is_eq(word,word) == True{} : Bool}:
  Equal.trans(Bool,U32.is_eq(word,word),Nat.is_eq(U32.to_nat(word),U32.to_nat(word)),True{},Comparison.eq(word,word),Order.equal_reflexive(U32.to_nat(word)))

# A word above the position of another differs from it.
def word_apart(+axis: U32,+at: U32,+position: Nat,+exact: {U32.to_nat(at) == position : Nat},+after: {Nat.is_lt(position,U32.to_nat(axis)) == True{} : Bool}) ->
  {U32.is_eq(axis,at) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),False{},Comparison.eq(axis,at),
    {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(U32.to_nat(axis),_) == False{} : Bool}
      %Order.equal_symmetric(position,U32.to_nat(axis)) : {_ == False{} : Bool}
      Order.less_not_equal(position,U32.to_nat(axis),after) :
      {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == False{} : Bool}})

# The size of the extents after an axis covers it, when the extents hold a
# cell.
def inner_le(+size: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,Nat.mul(size,inner)) == True{} : Bool}) -> {Nat.is_le(inner,Nat.mul(size,inner)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(size,inner),Nat.mul(inner,size),RowMajor.multiply_commutative(size,inner)) : {Nat.is_le(inner,_) == True{} : Bool}
  RowMajor.product_covers(inner,size,RowMajor.left_factor_positive(size,inner,positive))

# The product word of extents is their number of cells.
def product_value(extents: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(extents)) == ShapeLaws.elements(extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +inner_room = Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(cap),inner_le(size,inner,positive),room)
      Arithmetic.product_exact(cap,cap_ok,extent,Grid.product(rest),size,inner,{==},
        product_value(rest,cap,cap_ok,inner_room,DivisionOrder.right_factor_positive(size,inner,positive)),room)

# The source digits of the coordinates of an index make the index.
def coordinates_value(extents: List<&2,U32>,+index: U32,+at: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,Grid.coordinates(extents,index),at)) == U32.to_nat(index) : Nat}:
  match extents:
    case Nil{}:
      below_one(U32.to_nat(index),inside)
    case Con{+extent,+rest}:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +value = U32.to_nat(index)
      +whole = Nat.mul(size,inner)
      +whole_positive = Order.le_lt_trans(0n,value,whole,Order.zero_le(value),inside)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,whole_positive)
      +size_positive = RowMajor.left_factor_positive(size,inner,whole_positive)
      +inner_double = Order.transitive(Nat.double(inner),Nat.double(whole),U32.to_nat(cap),Order.double_le(inner,whole,inner_le(size,inner,whole_positive)),room)
      +size_double = Order.transitive(Nat.double(size),Nat.double(whole),U32.to_nat(cap),
        Order.double_le(size,whole,RowMajor.product_covers(size,inner,inner_positive)),room)
      +inner_room = Order.transitive(inner,Nat.double(inner),U32.to_nat(cap),Order.le_double(inner),inner_double)
      +inner_word = U32.max(Grid.product(rest),1)
      +inner_exact = Extrema.max_one_exact(Grid.product(rest),inner,product_value(rest,cap,cap_ok,inner_room,inner_positive),inner_positive)
      +size_word = U32.max(extent,1)
      +size_exact = Extrema.max_one_exact(extent,size,{==},size_positive)
      +quotient_word = (index / inner_word : U32)
      +quotient = Nat.div(value,inner)
      +quotient_exact = U32Division.quotient_exact(cap,cap_ok,index,inner_word,value,inner,{==},inner_exact,inner_double,inner_positive)
      +below = DivisionOrder.quotient_below(value,size,inner,inner_positive,inside)
      +kept = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(quotient,size),(0n,quotient),Division.small(quotient,size,below))
      +digit_word = (index / inner_word % size_word : U32)
      +digit_exact = {%kept : {U32.to_nat(digit_word) == _ : Nat}
        %quotient_exact : {U32.to_nat(digit_word) == Nat.mod(_,size) : Nat}
        %size_exact : {U32.to_nat(digit_word) == Nat.mod(U32.to_nat(quotient_word),_) : Nat}
        U32Division.modulo(quotient_word,size_word,cap,cap_ok,
          %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} size_double,
          %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_lt(0n,_) == True{} : Bool} size_positive) :
        {U32.to_nat(digit_word) == quotient : Nat}}
      +remainder_word = (index % inner_word : U32)
      +remainder_exact = {%inner_exact : {U32.to_nat(remainder_word) == Nat.mod(value,_) : Nat}
        U32Division.modulo(index,inner_word,cap,cap_ok,
          %Equal.sym(Nat,U32.to_nat(inner_word),inner,inner_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} inner_double,
          %Equal.sym(Nat,U32.to_nat(inner_word),inner,inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive) :
        {U32.to_nat(remainder_word) == Nat.mod(value,inner) : Nat}}
      +later = coordinates_value(rest,remainder_word,(at + 1 : U32),cap,cap_ok,inner_double,
        {%Equal.sym(Nat,U32.to_nat(remainder_word),Nat.mod(value,inner),remainder_exact) : {Nat.is_lt(_,inner) == True{} : Bool}
          DivisionOrder.remainder_small(value,inner,inner_positive) : {Nat.is_lt(U32.to_nat(remainder_word),inner) == True{} : Bool}})
      +tail = ShapeLaws.digits_value(rest,source_digits(Nil{},Nil{},rest,Grid.coordinates(rest,remainder_word),(at + 1 : U32)))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},Laws.strides_con(extent,rest)) :
        {ShapeLaws.digits_value(Con{extent,rest},source_digits(Nil{},Nil{},Con{extent,rest},Grid.coordinates_by(Con{extent,rest},_,index),at)) == value : Nat}
      %Equal.sym(Nat,U32.to_nat(digit_word),quotient,digit_exact) : {Nat.add(Nat.mul(_,inner),tail) == value : Nat}
      %Equal.sym(Nat,tail,U32.to_nat(remainder_word),later) : {Nat.add(Nat.mul(quotient,inner),_) == value : Nat}
      %Equal.sym(Nat,U32.to_nat(remainder_word),Nat.mod(value,inner),remainder_exact) : {Nat.add(Nat.mul(quotient,inner),_) == value : Nat}
      Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,inner),Nat.mod(value,inner)),DivisionOrder.quotient_exact(value,inner,inner_positive))

# What the axis a stride steps tells: it lies at or after position, its digit
# is the one the target axes read there from its fixed coordinate, and the
# step times its weight is the stride.
def StepAt(found: Tasks.Stepped,+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,+extents: List<&2,U32>,+fixed: List<&2,U32>,+at: U32,+position: Nat,
  +stride: U32) -> Data:
  match found:
    case Tasks.Unstepped{}:
      Unit
    case Tasks.Stepped{+axis,+step,+extent,+first}:
      Held.Both<{Nat.is_le(position,U32.to_nat(axis)) == True{} : Bool},
        Held.Both<{digit_at(source_digits(readings,cells,extents,fixed,at),at,axis) == read_digit(readings,cells,axis,U32.to_nat(first)) : Nat},
          {Nat.mul(U32.to_nat(step),axis_weight(extents,source_digits(readings,cells,extents,fixed,at),at,axis)) == U32.to_nat(stride) : Nat}>>

# An axis found after the first keeps its facts past the first.
def step_widen(found: Tasks.Stepped,+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,+extent: U32,+rest: List<&2,U32>,+first: U32,+more: List<&2,U32>,
  +at: U32,+position: Nat,+stride: U32,+exact: {U32.to_nat(at) == position : Nat},
  +later: StepAt(found,readings,cells,rest,more,(at + 1 : U32),1n+position,stride)) -> StepAt(found,readings,cells,Con{extent,rest},Con{first,more},at,position,stride):
  match found:
    case Tasks.Unstepped{}:
      Unit{}
    case Tasks.Stepped{+axis,+step,+source_extent,+fixed}:
      Held.Both{+after,+tail} = later
      Held.Both{+digit,+weight} = tail
      +head = read_digit(readings,cells,at,U32.to_nat(first))
      +digits = source_digits(readings,cells,rest,more,(at + 1 : U32))
      +apart = word_apart(axis,at,position,exact,Order.succ_le_lt(position,U32.to_nat(axis),after))
      Held.Both{Order.transitive(position,1n+position,U32.to_nat(axis),Bounds.strict_le(position,1n+position,Order.lt_succ(position,position,Order.reflexive(position))),after),
        Held.Both{
          {%Equal.sym(Bool,U32.is_eq(axis,at),False{},apart) :
              {Bool.pick(Nat,_,head,digit_at(digits,(at + 1 : U32),axis)) == read_digit(readings,cells,axis,U32.to_nat(fixed)) : Nat}
            digit :
            {Bool.pick(Nat,U32.is_eq(axis,at),head,digit_at(digits,(at + 1 : U32),axis)) == read_digit(readings,cells,axis,U32.to_nat(fixed)) : Nat}},
          {%Equal.sym(Bool,U32.is_eq(axis,at),False{},apart) :
              {Nat.mul(U32.to_nat(step),Bool.pick(Nat,_,ShapeLaws.elements(rest),axis_weight(rest,digits,(at + 1 : U32),axis))) == U32.to_nat(stride) : Nat}
            weight :
            {Nat.mul(U32.to_nat(step),Bool.pick(Nat,U32.is_eq(axis,at),ShapeLaws.elements(rest),axis_weight(rest,digits,(at + 1 : U32),axis))) == U32.to_nat(stride) : Nat}}}}

# The stride divides into whole steps of the axis it finds.
def step_found(+stride: U32,+own: U32,+inner: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inner_exact: {U32.to_nat(own) == inner : Nat},+room: {Nat.is_le(Nat.double(inner),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,inner) == True{} : Bool},+whole: {U32.is_zero((stride % own : U32)) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat((stride / own : U32)),inner) == U32.to_nat(stride) : Nat}:
  +value = U32.to_nat(stride)
  +quotient_exact = U32Division.quotient_exact(cap,cap_ok,stride,own,value,inner,{==},inner_exact,room,positive)
  +remainder_exact = {%inner_exact : {U32.to_nat((stride % own : U32)) == Nat.mod(value,_) : Nat}
    U32Division.modulo(stride,own,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(own),inner,inner_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room,
      %Equal.sym(Nat,U32.to_nat(own),inner,inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive) :
    {U32.to_nat((stride % own : U32)) == Nat.mod(value,inner) : Nat}}
  +none = zero_value(Nat.mod(value,inner),(stride % own : U32),remainder_exact,whole)
  %Equal.sym(Nat,U32.to_nat((stride / own : U32)),Nat.div(value,inner),quotient_exact) : {Nat.mul(_,inner) == value : Nat}
  Equal.trans(Nat,Nat.mul(Nat.div(value,inner),inner),Nat.add(Nat.mul(Nat.div(value,inner),inner),0n),value,
    Equal.sym(Nat,Nat.add(Nat.mul(Nat.div(value,inner),inner),0n),Nat.mul(Nat.div(value,inner),inner),Addition.zero_right(Nat.mul(Nat.div(value,inner),inner))),
    {%none : {Nat.add(Nat.mul(Nat.div(value,inner),inner),_) == value : Nat}
      Equal.sym(Nat,value,Nat.add(Nat.mul(Nat.div(value,inner),inner),Nat.mod(value,inner)),DivisionOrder.quotient_exact(value,inner,positive)) :
      {Nat.add(Nat.mul(Nat.div(value,inner),inner),0n) == value : Nat}})

# The first axis that passes the test, or one after it.
def step_choice(choice: Bool,+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,+extent: U32,+rest: List<&2,U32>,+first: U32,+more: List<&2,U32>,
  +at: U32,+position: Nat,+stride: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +test: {Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride) == choice : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+inner_exact: {U32.to_nat(Grid.product(rest)) == ShapeLaws.elements(rest) : Nat},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(rest)) == True{} : Bool},
  +later: StepAt(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),readings,cells,rest,more,(at + 1 : U32),1n+position,stride)) ->
  StepAt(Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Con{first,more},choice,stride,at),readings,cells,Con{extent,rest},Con{first,more},at,position,stride):
  match choice:
    case True{}:
      +own = U32.max(Grid.product(rest),1)
      +inner = ShapeLaws.elements(rest)
      +whole = and_right((U32.is_lt(1,extent) && U32.is_lt(0,Grid.product(rest)) && U32.is_le(Grid.product(rest),stride)),U32.is_zero((stride % own : U32)),test)
      +head = read_digit(readings,cells,at,U32.to_nat(first))
      +digits = source_digits(readings,cells,rest,more,(at + 1 : U32))
      +same = word_same(at)
      Held.Both{{%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_le(position,_) == True{} : Bool} Order.reflexive(position) :
          {Nat.is_le(position,U32.to_nat(at)) == True{} : Bool}},
        Held.Both{
          {%Equal.sym(Bool,U32.is_eq(at,at),True{},same) : {Bool.pick(Nat,_,head,digit_at(digits,(at + 1 : U32),at)) == head : Nat}
            {==} :
            {Bool.pick(Nat,U32.is_eq(at,at),head,digit_at(digits,(at + 1 : U32),at)) == head : Nat}},
          {%Equal.sym(Bool,U32.is_eq(at,at),True{},same) :
              {Nat.mul(U32.to_nat((stride / own : U32)),Bool.pick(Nat,_,inner,axis_weight(rest,digits,(at + 1 : U32),at))) == U32.to_nat(stride) : Nat}
            step_found(stride,own,inner,cap,cap_ok,Extrema.max_one_exact(Grid.product(rest),inner,inner_exact,positive),room,positive,whole) :
            {Nat.mul(U32.to_nat((stride / own : U32)),Bool.pick(Nat,U32.is_eq(at,at),inner,axis_weight(rest,digits,(at + 1 : U32),at))) == U32.to_nat(stride) : Nat}}}}
    case False{}:
      step_widen(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),readings,cells,extent,rest,first,more,at,position,stride,exact,later)

# The axis a stride steps, from a first axis, walks the strides of the axes.
def stepped_con(+extent: U32,+rest: List<&2,U32>,+fixed: List<&2,U32>,+stride: U32,+at: U32) ->
  {Tasks.stepped_axis(Con{extent,rest},fixed,stride,at) ==
    Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},fixed,Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),stride,at) : Tasks.Stepped}:
  Equal.cong(List<&2,U32>,Tasks.Stepped,strides => Tasks.stepped_on(Con{extent,rest},strides,fixed,Tasks.head_steps(Con{extent,rest},strides,stride),stride,at),
    Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},Laws.strides_con(extent,rest))

# The axis a stride steps, walked with the fixed coordinates.
def stepped_at(extents: List<&2,U32>,fixed: List<&2,U32>,+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,+stride: U32,+at: U32,+position: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  StepAt(Tasks.stepped_axis(extents,fixed,stride,at),readings,cells,extents,fixed,at,position,stride):
  match extents fixed:
    case Con{+extent,+rest} Con{+first,+more}:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(size,inner)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +inner_double = Order.transitive(Nat.double(inner),Nat.double(whole),U32.to_nat(cap),Order.double_le(inner,whole,inner_le(size,inner,positive)),size_room)
      +inner_room = Order.transitive(inner,Nat.double(inner),U32.to_nat(cap),Order.le_double(inner),inner_double)
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
      +next_exact = Components.next_word(at,position,cap,cap_ok,exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),U32.to_nat(cap),Order.le_add(1n+position,length),shifted))
      %Equal.sym(Tasks.Stepped,Tasks.stepped_axis(Con{extent,rest},Con{first,more},stride,at),
          Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Con{first,more},Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),stride,at),stepped_con(extent,rest,Con{first,more},stride,at)) :
        StepAt(_,readings,cells,Con{extent,rest},Con{first,more},at,position,stride)
      step_choice(Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),readings,cells,extent,rest,first,more,
        at,position,stride,cap,cap_ok,{==},exact,product_value(rest,cap,cap_ok,inner_room,inner_positive),inner_double,inner_positive,
        stepped_at(rest,more,readings,cells,stride,(at + 1 : U32),1n+position,cap,cap_ok,next_exact,shifted,inner_double,inner_positive))
    case Nil{} +fixed:
      Unit{}
    case Con{+extent,+rest} Nil{}:
      %Equal.sym(Tasks.Stepped,Tasks.stepped_axis(Con{extent,rest},Nil{},stride,at),
          Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Nil{},Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),stride,at),stepped_con(extent,rest,Nil{},stride,at)) :
        StepAt(_,readings,cells,Con{extent,rest},Nil{},at,position,stride)
      Unit{}

# ---- The digits of a target cell make its view address ----

# The source digits number no more than the source axes.
def digits_length(+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {Nat.is_le(List.length(&2,Nat,source_digits(readings,cells,extents,fixed,at)),List.length(&2,U32,extents)) == True{} : Bool}:
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      digits_length(readings,cells,rest,more,(at + 1 : U32))
    case Nil{} +fixed:
      {==}
    case Con{+extent,+rest} Nil{}:
      {==}

# A choice whose negation holds does not hold.
def not_true(value: Bool,+negated: {Bool.not(value) == True{} : Bool}) -> {value == False{} : Bool}:
  match value:
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(negated))
    case False{}: {==}

# A natural number at most zero is zero.
def le_zero(value: Nat,+small: {Nat.is_le(value,0n) == True{} : Bool}) -> {value == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +rest: Empty.absurd({1n+rest == 0n : Nat},WordEncoding.false_true(small))

# What a view axis guarantees its reading: an axis it does not step adds
# nothing, and the last index of an axis it steps fits the word.
def AxisFit(found: Tasks.Stepped,+extent: U32,+stride: U32,+cap: U32) -> Data:
  match found:
    case Tasks.Unstepped{}:
      {Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)) == 0n : Nat}
    case Tasks.Stepped{axis,+step,source_extent,+first}:
      {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(cap)) == True{} : Bool}

def AxesFit(+extents: List<&2,U32>,+fixed: List<&2,U32>,axes: List<&2,Views.Axis>,+cap: U32) -> Data:
  match axes:
    case Nil{}:
      Unit
    case Con{Views.Axis{+extent,+stride},rest}:
      Held.Both<AxisFit(Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(extents,fixed,stride,0)),extent,stride,cap),
        AxesFit(extents,fixed,rest,cap)>

# The view read of a stepped axis, or a crossing when it runs past the axis.
def stepped_reading(fits: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+flag: Bool,+bound: U32,+extent: U32,+block: U32) -> Tasks.Reading:
  Bool.pick(Tasks.Reading,fits,Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,flag,1,1,0,bound,False{}}},Tasks.Reading{extent,block,Tasks.Crossing{}})

# The readings after one that is no crossing are no crossing.
def crossing_pick(fits: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+flag: Bool,+bound: U32,+extent: U32,+block: U32,+readings: List<&2,Tasks.Reading>,
  +none: {Tasks.crossing(Con{stepped_reading(fits,axis,source_block,step,offset,flag,bound,extent,block),readings}) == False{} : Bool}) ->
  {Tasks.crossing(readings) == False{} : Bool}:
  match fits:
    case True{}: none
    case False{}: Empty.absurd({Tasks.crossing(readings) == False{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},none)))

def tail_crossing(found: Tasks.Stepped,+source_grid: Grid.Grid,+extent: U32,+block: U32,+flag: Bool,+readings: List<&2,Tasks.Reading>,
  +none: {Tasks.crossing(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings}) == False{} : Bool}) -> {Tasks.crossing(readings) == False{} : Bool}:
  match found:
    case Tasks.Unstepped{}:
      none
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      crossing_pick(U32.is_lt(last,source_extent),axis,Grid.block(source_grid,axis,source_extent),step,Bool.pick(U32,flag,last,first),flag,source_extent,extent,block,
        readings,none)

# The readings after one keep their source axes apart.
def distinct_pick(fits: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+flag: Bool,+bound: U32,+extent: U32,+block: U32,+readings: List<&2,Tasks.Reading>,
  +apart: {Tasks.distinct(Tasks.read_axes(Con{stepped_reading(fits,axis,source_block,step,offset,flag,bound,extent,block),readings})) == True{} : Bool}) ->
  {Tasks.distinct(Tasks.read_axes(readings)) == True{} : Bool}:
  match fits:
    case True{}: and_right(Bool.not(Tasks.holds(Tasks.read_axes(readings),axis)),Tasks.distinct(Tasks.read_axes(readings)),apart)
    case False{}: apart

def tail_distinct(found: Tasks.Stepped,+source_grid: Grid.Grid,+extent: U32,+block: U32,+flag: Bool,+readings: List<&2,Tasks.Reading>,
  +apart: {Tasks.distinct(Tasks.read_axes(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings})) == True{} : Bool}) ->
  {Tasks.distinct(Tasks.read_axes(readings)) == True{} : Bool}:
  match found:
    case Tasks.Unstepped{}:
      apart
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      distinct_pick(U32.is_lt(last,source_extent),axis,Grid.block(source_grid,axis,source_extent),step,Bool.pick(U32,flag,last,first),flag,source_extent,extent,block,
        readings,apart)

# The facts of the axis a stride steps, or none when the stride is zero.
def step_or_none(zero: Bool,+extents: List<&2,U32>,+fixed: List<&2,U32>,+readings: List<&2,Tasks.Reading>,+cells: List<&2,Nat>,+stride: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  StepAt(Bool.pick(Tasks.Stepped,zero,Tasks.Unstepped{},Tasks.stepped_axis(extents,fixed,stride,0)),readings,cells,extents,fixed,0,0n,stride):
  match zero:
    case True{}: Unit{}
    case False{}: stepped_at(extents,fixed,readings,cells,stride,0,0n,cap,cap_ok,{==},room,size_room,positive)

# The last index a stepped read reaches: its fixed coordinate plus the step
# times the last coordinate of the target axis.
def last_exact(+first: U32,+step: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)) == Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))) : Nat}:
  +origin = U32.to_nat(first)
  +stride = U32.to_nat(step)
  +size = U32.to_nat(extent)
  +span = Nat.sub(size,1n)
  +size_exact = Extrema.max_one_exact(extent,size,{==},positive)
  +top = (U32.max(extent,1) - 1 : U32)
  +top_exact = {%size_exact : {U32.to_nat(top) == Nat.sub(_,1n) : Nat}
    Arithmetic.subtract(U32.max(extent,1),1,cap,cap_ok,
      {%Equal.sym(Nat,U32.to_nat(U32.max(extent,1)),size,size_exact) : {Nat.is_le(1n,_) == True{} : Bool} Order.strict_follow(0n,size,positive) :
        {Nat.is_le(1n,U32.to_nat(U32.max(extent,1))) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(U32.max(extent,1)),size,size_exact) : {Nat.is_le(Nat.sub(_,1n),U32.to_nat(cap)) == True{} : Bool}
        Order.transitive(span,size,U32.to_nat(cap),Order.sub_le(size,1n),extent_room) :
        {Nat.is_le(Nat.sub(U32.to_nat(U32.max(extent,1)),1n),U32.to_nat(cap)) == True{} : Bool}}) :
    {U32.to_nat(top) == span : Nat}}
  +reach = Nat.mul(stride,span)
  +reach_exact = Arithmetic.product_exact(cap,cap_ok,step,top,stride,span,{==},top_exact,
    Order.transitive(reach,Nat.add(origin,reach),U32.to_nat(cap),Order.right_le(origin,reach),fit))
  Arithmetic.sum_exact(cap,cap_ok,first,(step * top : U32),origin,reach,{==},reach_exact,fit)

# The source index a stepped read gives a coordinate: its fixed coordinate
# plus the step times the reflected coordinate.
def read_shift(flag: Bool,+first: U32,+step: U32,+extent: U32,+coordinate: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  {read_value(flag,U32.to_nat(Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first)),U32.to_nat(step),coordinate) ==
    Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate))) : Nat}:
  match flag:
    case False{}:
      {==}
    case True{}:
      +origin = U32.to_nat(first)
      +stride = U32.to_nat(step)
      +size = U32.to_nat(extent)
      +span = Nat.sub(size,1n)
      +positive = Order.le_lt_trans(0n,coordinate,size,Order.zero_le(coordinate),inside)
      +reach = Nat.mul(stride,span)
      +reached = last_exact(first,step,extent,cap,cap_ok,fit,positive,extent_room)
      +covered = Components.pred_le(coordinate,size,inside)
      +near = Nat.mul(stride,coordinate)
      +near_le = DivisionOrder.multiply_left_le(stride,coordinate,span,covered)
      +turned = Equal.trans(Nat,Nat.sub(reach,near),Nat.sub(Nat.mul(span,stride),Nat.mul(coordinate,stride)),Nat.mul(stride,Nat.sub(span,coordinate)),
        Equal.trans(Nat,Nat.sub(reach,near),Nat.sub(Nat.mul(span,stride),near),Nat.sub(Nat.mul(span,stride),Nat.mul(coordinate,stride)),
          Equal.cong(Nat,Nat,value => Nat.sub(value,near),reach,Nat.mul(span,stride),RowMajor.multiply_commutative(stride,span)),
          Equal.cong(Nat,Nat,value => Nat.sub(Nat.mul(span,stride),value),near,Nat.mul(coordinate,stride),RowMajor.multiply_commutative(stride,coordinate))),
        Equal.trans(Nat,Nat.sub(Nat.mul(span,stride),Nat.mul(coordinate,stride)),Nat.mul(Nat.sub(span,coordinate),stride),Nat.mul(stride,Nat.sub(span,coordinate)),
          RowMajor.subtract_products(span,coordinate,stride),RowMajor.multiply_commutative(Nat.sub(span,coordinate),stride)))
      %Equal.sym(Nat,U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)),Nat.add(origin,reach),reached) :
        {Nat.sub(_,near) == Nat.add(origin,Nat.mul(stride,Nat.sub(span,coordinate))) : Nat}
      Equal.trans(Nat,Nat.sub(Nat.add(origin,reach),near),Nat.add(origin,Nat.sub(reach,near)),Nat.add(origin,Nat.mul(stride,Nat.sub(span,coordinate))),
        Order.subtract_from_sum(origin,reach,near,near_le),Equal.cong(Nat,Nat,value => Nat.add(origin,value),Nat.sub(reach,near),Nat.mul(stride,Nat.sub(span,coordinate)),turned))

# The number a read digit makes, with the moved digit cancelled.
def balance(+changed: Nat,+kept: Nat,+fixed_part: Nat,+moved_part: Nat,+base: Nat,+rest_sum: Nat,
  +equation: {Nat.add(changed,fixed_part) == Nat.add(kept,Nat.add(fixed_part,moved_part)) : Nat},+later: {kept == Nat.add(base,rest_sum) : Nat}) ->
  {changed == Nat.add(base,Nat.add(moved_part,rest_sum)) : Nat}:
  +swapped = Equal.trans(Nat,Nat.add(fixed_part,changed),Nat.add(changed,fixed_part),Nat.add(fixed_part,Nat.add(kept,moved_part)),Order.commutative(fixed_part,changed),
    Equal.trans(Nat,Nat.add(changed,fixed_part),Nat.add(kept,Nat.add(fixed_part,moved_part)),Nat.add(fixed_part,Nat.add(kept,moved_part)),equation,
      RowMajor.add_swap(kept,fixed_part,moved_part)))
  +cancelled = Order.cancel_sum(fixed_part,changed,Nat.add(kept,moved_part),swapped)
  Equal.trans(Nat,changed,Nat.add(kept,moved_part),Nat.add(base,Nat.add(moved_part,rest_sum)),cancelled,
    Equal.trans(Nat,Nat.add(kept,moved_part),Nat.add(Nat.add(base,rest_sum),moved_part),Nat.add(base,Nat.add(moved_part,rest_sum)),
      Equal.cong(Nat,Nat,value => Nat.add(value,moved_part),kept,Nat.add(base,rest_sum),later),
      Equal.trans(Nat,Nat.add(Nat.add(base,rest_sum),moved_part),Nat.add(base,Nat.add(rest_sum,moved_part)),Nat.add(base,Nat.add(moved_part,rest_sum)),
        Addition.associative(base,rest_sum,moved_part),
        Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(rest_sum,moved_part),Nat.add(moved_part,rest_sum),Order.commutative(rest_sum,moved_part)))))

# A stepped target axis moves the number by its reflected coordinate times its
# stride.
def read_step(fits: Bool,+axis: U32,+step: U32,+first: U32,+source_extent: U32,+extent: U32,+stride: U32,+flag: Bool,+block: U32,+source_grid: Grid.Grid,
  +readings: List<&2,Tasks.Reading>,+coordinate: Nat,+cells: List<&2,Nat>,+extents: List<&2,U32>,+fixed: List<&2,U32>,+base: Nat,+rest_sum: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(cap)) == True{} : Bool},
  +whole: {Tasks.crossing(Con{stepped_reading(fits,axis,Grid.block(source_grid,axis,source_extent),step,
    Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block),readings}) == False{} : Bool},
  +apart: {Tasks.distinct(Tasks.read_axes(Con{stepped_reading(fits,axis,Grid.block(source_grid,axis,source_extent),step,
    Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block),readings})) == True{} : Bool},
  +steps: StepAt(Tasks.Stepped{axis,step,source_extent,first},readings,cells,extents,fixed,0,0n,stride),
  +later: {ShapeLaws.digits_value(extents,source_digits(readings,cells,extents,fixed,0)) == Nat.add(base,rest_sum) : Nat},
  +coordinate_in: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(extents,source_digits(Con{stepped_reading(fits,axis,Grid.block(source_grid,axis,source_extent),step,
      Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block),readings},Con{coordinate,cells},extents,fixed,0)) ==
    Nat.add(base,Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate),U32.to_nat(stride)),rest_sum)) : Nat}:
  match fits:
    case True{}:
      Held.Both{+after,+tail} = steps
      Held.Both{+digit,+weight} = tail
      +offset = Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first)
      +source_block = Grid.block(source_grid,axis,source_extent)
      +digits = source_digits(readings,cells,extents,fixed,0)
      +origin = U32.to_nat(first)
      +steps_nat = U32.to_nat(step)
      +turned = MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate)
      +value = read_value(flag,U32.to_nat(offset),steps_nat,coordinate)
      +weight_of = axis_weight(extents,digits,0,axis)
      +stride_nat = U32.to_nat(stride)
      +changed = ShapeLaws.digits_value(extents,put(digits,0,axis,value))
      +kept = ShapeLaws.digits_value(extents,digits)
      +unheld = not_true(Tasks.holds(Tasks.read_axes(readings),axis),
        and_left(Bool.not(Tasks.holds(Tasks.read_axes(readings),axis)),Tasks.distinct(Tasks.read_axes(readings)),apart))
      +digit_fixed = Equal.trans(Nat,digit_at(digits,0,axis),read_digit(readings,cells,axis,origin),origin,digit,read_unheld(readings,cells,axis,origin,unheld))
      +shift = read_shift(flag,first,step,extent,coordinate,cap,cap_ok,fit,coordinate_in,extent_room)
      +moved = put_value(extents,digits,0,0n,axis,value,cap,cap_ok,{==},
        Order.transitive(List.length(&2,Nat,digits),List.length(&2,U32,extents),U32.to_nat(cap),digits_length(readings,cells,extents,fixed,0),room))
      +scaled = Equal.trans(Nat,Nat.mul(Nat.mul(steps_nat,turned),weight_of),Nat.mul(Nat.mul(turned,steps_nat),weight_of),Nat.mul(turned,stride_nat),
        Equal.cong(Nat,Nat,value => Nat.mul(value,weight_of),Nat.mul(steps_nat,turned),Nat.mul(turned,steps_nat),RowMajor.multiply_commutative(steps_nat,turned)),
        Equal.trans(Nat,Nat.mul(Nat.mul(turned,steps_nat),weight_of),Nat.mul(turned,Nat.mul(steps_nat,weight_of)),Nat.mul(turned,stride_nat),
          RowMajor.assoc(turned,steps_nat,weight_of),Equal.cong(Nat,Nat,value => Nat.mul(turned,value),Nat.mul(steps_nat,weight_of),stride_nat,weight)))
      +equation = {%scaled : {Nat.add(changed,Nat.mul(origin,weight_of)) == Nat.add(kept,Nat.add(Nat.mul(origin,weight_of),_)) : Nat}
        %RowMajor.distrib(origin,Nat.mul(steps_nat,turned),weight_of) : {Nat.add(changed,Nat.mul(origin,weight_of)) == Nat.add(kept,_) : Nat}
        %shift : {Nat.add(changed,Nat.mul(origin,weight_of)) == Nat.add(kept,Nat.mul(_,weight_of)) : Nat}
        %digit_fixed : {Nat.add(changed,Nat.mul(_,weight_of)) == Nat.add(kept,Nat.mul(value,weight_of)) : Nat}
        moved :
        {Nat.add(changed,Nat.mul(origin,weight_of)) == Nat.add(kept,Nat.add(Nat.mul(origin,weight_of),Nat.mul(turned,stride_nat))) : Nat}}
      +put_eq = digits_put(extent,block,axis,source_block,step,offset,flag,1,1,0,source_extent,False{},readings,coordinate,cells,extents,fixed,0)
      %Equal.sym(List<&2,Nat>,source_digits(Con{Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,flag,1,1,0,source_extent,False{}}},readings},
          Con{coordinate,cells},extents,fixed,0),put(digits,0,axis,value),put_eq) :
        {ShapeLaws.digits_value(extents,_) == Nat.add(base,Nat.add(Nat.mul(turned,stride_nat),rest_sum)) : Nat}
      balance(changed,kept,Nat.mul(origin,weight_of),Nat.mul(turned,stride_nat),base,rest_sum,equation,later)
    case False{}:
      Empty.absurd({ShapeLaws.digits_value(extents,source_digits(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},readings},Con{coordinate,cells},extents,fixed,0)) ==
          Nat.add(base,Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate),U32.to_nat(stride)),rest_sum)) : Nat},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},whole)))

# One target axis adds its reflected coordinate times its stride.
def axis_step(found: Tasks.Stepped,+extent: U32,+stride: U32,+flag: Bool,+block: U32,+source_grid: Grid.Grid,+readings: List<&2,Tasks.Reading>,+coordinate: Nat,
  +cells: List<&2,Nat>,+extents: List<&2,U32>,+fixed: List<&2,U32>,+base: Nat,+rest_sum: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: AxisFit(found,extent,stride,cap),
  +whole: {Tasks.crossing(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings}) == False{} : Bool},
  +apart: {Tasks.distinct(Tasks.read_axes(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings})) == True{} : Bool},
  +steps: StepAt(found,readings,cells,extents,fixed,0,0n,stride),
  +later: {ShapeLaws.digits_value(extents,source_digits(readings,cells,extents,fixed,0)) == Nat.add(base,rest_sum) : Nat},
  +coordinate_in: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(extents,source_digits(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings},Con{coordinate,cells},extents,fixed,0)) ==
    Nat.add(base,Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate),U32.to_nat(stride)),rest_sum)) : Nat}:
  match found:
    case Tasks.Unstepped{}:
      +turned = MirrorAddress.reflected_coordinate(flag,U32.to_nat(extent),coordinate)
      +stride_nat = U32.to_nat(stride)
      +span = Nat.sub(U32.to_nat(extent),1n)
      +none = le_zero(Nat.mul(turned,stride_nat),
        {%fit : {Nat.is_le(Nat.mul(turned,stride_nat),_) == True{} : Bool}
          RowMajor.mul_le(turned,span,stride_nat,reflected_le(flag,U32.to_nat(extent),coordinate,coordinate_in)) :
          {Nat.is_le(Nat.mul(turned,stride_nat),0n) == True{} : Bool}})
      +unread = digits_unread(extent,block,readings,coordinate,cells,extents,fixed,0)
      %Equal.sym(List<&2,Nat>,source_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},readings},Con{coordinate,cells},extents,fixed,0),
          source_digits(readings,cells,extents,fixed,0),unread) :
        {ShapeLaws.digits_value(extents,_) == Nat.add(base,Nat.add(Nat.mul(turned,stride_nat),rest_sum)) : Nat}
      %Equal.sym(Nat,Nat.mul(turned,stride_nat),0n,none) :
        {ShapeLaws.digits_value(extents,source_digits(readings,cells,extents,fixed,0)) == Nat.add(base,Nat.add(_,rest_sum)) : Nat}
      later
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      read_step(U32.is_lt(last,source_extent),axis,step,first,source_extent,extent,stride,flag,block,source_grid,readings,coordinate,cells,extents,fixed,base,rest_sum,
        cap,cap_ok,fit,whole,apart,steps,later,coordinate_in,extent_room,room)

# The source digits a target cell reads make the view offset plus its
# reflected digit sum.
def view_digits(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+extents: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+fixed: List<&2,U32>,
  +axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +apart: {Tasks.distinct(Tasks.read_axes(Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,axis))) == True{} : Bool},
  +whole: {Tasks.crossing(Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,axis)) == False{} : Bool},
  +fit: AxesFit(extents,fixed,axes,cap)) ->
  {ShapeLaws.digits_value(extents,source_digits(Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,axis),Targets.cell_coordinates(Views.axis_extents(axes),index),
      extents,fixed,0)) ==
    Nat.add(ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,fixed,0)),view_sum(axes,flags,index)) : Nat}:
  match axes:
    case Nil{}:
      +base = ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,fixed,0))
      Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base))
    case Con{Views.Axis{+extent,+stride},+rest}:
      Held.Both{+here,+later_fit} = fit
      +base = ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,fixed,0))
      +flag = Views.flag(flags)
      +later_flags = Views.later_flags(flags)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(extents,fixed,stride,0))
      +block = Grid.block(target_grid,axis,extent)
      +next = (axis + 1 : U32)
      +later_readings = Tasks.readings(extents,source_grid,target_grid,fixed,rest,later_flags,next)
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +inner_positive = Components.inner_positive(size,inner,index,inside)
      +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(size,inner),Order.zero_le(index),inside)
      +coordinate = Nat.div(index,inner)
      +remainder = Nat.mod(index,inner)
      +cells = Targets.cell_coordinates(Views.axis_extents(rest),remainder)
      +later = view_digits(rest,later_flags,remainder,extents,source_grid,target_grid,fixed,next,cap,cap_ok,room,size_room,positive,
        Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(cap),inner_le(size,inner,whole_positive),target_room),
        DivisionOrder.remainder_small(index,inner,inner_positive),
        tail_distinct(found,source_grid,extent,block,flag,later_readings,apart),tail_crossing(found,source_grid,extent,block,flag,later_readings,whole),later_fit)
      axis_step(found,extent,stride,flag,block,source_grid,later_readings,coordinate,cells,extents,fixed,base,view_sum(rest,later_flags,remainder),cap,cap_ok,
        here,whole,apart,step_or_none(Tasks.unread(extent,stride),extents,fixed,later_readings,cells,stride,cap,cap_ok,room,size_room,positive),later,
        DivisionOrder.quotient_below(index,size,inner,inner_positive,inside),
        Order.transitive(size,Nat.mul(size,inner),U32.to_nat(cap),RowMajor.product_covers(size,inner,inner_positive),target_room),room)

# ---- A target cell inside the classes of its component reads as a view ----

# The last cell of a class range, as a word, holds the coordinate and stays
# inside its axis.
def HighIn(+reach: U32,+extent: U32,+coordinate: Nat) -> Data:
  Held.Both<{Nat.is_le(coordinate,U32.to_nat((reach - 1 : U32))) == True{} : Bool},
    {Nat.is_le(U32.to_nat((reach - 1 : U32)),Nat.sub(U32.to_nat(extent),1n)) == True{} : Bool}>

def high_in(+reach: U32,+extent: U32,+coordinate: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(coordinate,U32.to_nat(reach)) == True{} : Bool},+within: {Nat.is_le(U32.to_nat(reach),U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) -> HighIn(reach,extent,coordinate):
  +size = U32.to_nat(reach)
  +positive = Order.le_lt_trans(0n,coordinate,size,Order.zero_le(coordinate),below)
  +exact = Arithmetic.subtract(reach,1,cap,cap_ok,Order.strict_follow(0n,size,positive),
    Order.transitive(Nat.sub(size,1n),size,U32.to_nat(cap),Order.sub_le(size,1n),Order.transitive(size,U32.to_nat(extent),U32.to_nat(cap),within,extent_room)))
  Held.Both{{%Equal.sym(Nat,U32.to_nat((reach - 1 : U32)),Nat.sub(size,1n),exact) : {Nat.is_le(coordinate,_) == True{} : Bool}
      Components.pred_le(coordinate,size,below) : {Nat.is_le(coordinate,U32.to_nat((reach - 1 : U32))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((reach - 1 : U32)),Nat.sub(size,1n),exact) : {Nat.is_le(_,Nat.sub(U32.to_nat(extent),1n)) == True{} : Bool}
      DivisionOrder.subtract_le(size,U32.to_nat(extent),1n,within) : {Nat.is_le(U32.to_nat((reach - 1 : U32)),Nat.sub(U32.to_nat(extent),1n)) == True{} : Bool}}}

def high_low(+reach: U32,+extent: U32,+coordinate: Nat,+held: HighIn(reach,extent,coordinate)) ->
  {Nat.is_le(coordinate,U32.to_nat((reach - 1 : U32))) == True{} : Bool}:
  Held.Both{+low,+top} = held
  low

def high_top(+reach: U32,+extent: U32,+coordinate: Nat,+held: HighIn(reach,extent,coordinate)) ->
  {Nat.is_le(U32.to_nat((reach - 1 : U32)),Nat.sub(U32.to_nat(extent),1n)) == True{} : Bool}:
  Held.Both{+low,+top} = held
  top

# The indices from the first a stepped read reaches up to a high coordinate
# stay at or below its last index.
def reached_le(+first: U32,+step: U32,+extent: U32,+high: U32,
  +last_value: {U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)) ==
    Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))) : Nat},
  +high_le: {Nat.is_le(U32.to_nat(high),Nat.sub(U32.to_nat(extent),1n)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),U32.to_nat(high))),U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32))) == True{} : Bool}:
  +origin = U32.to_nat(first)
  +far = Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))
  +reached = Nat.mul(U32.to_nat(step),U32.to_nat(high))
  %Equal.sym(Nat,U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)),Nat.add(origin,far),last_value) : {Nat.is_le(Nat.add(origin,reached),_) == True{} : Bool}
  Order.offset_le(origin,reached,far,DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(high),Nat.sub(U32.to_nat(extent),1n),high_le))

# A stepped read reaches no further than its last index.
def inside_of(flag: Bool,+first: U32,+step: U32,+extent: U32,+bound: U32,+high: U32,
  +last_ge: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),U32.to_nat(high))),U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32))) == True{} : Bool},
  +last_below: {Nat.is_lt(U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)),U32.to_nat(bound)) == True{} : Bool}) ->
  ReadInside(flag,step,Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),high,bound):
  match flag:
    case False{}:
      Order.le_lt_trans(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),U32.to_nat(high))),U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)),
        U32.to_nat(bound),last_ge,last_below)
    case True{}:
      +reached = Nat.mul(U32.to_nat(step),U32.to_nat(high))
      Held.Both{Order.transitive(reached,Nat.add(U32.to_nat(first),reached),U32.to_nat((first + step * (U32.max(extent,1) - 1 : U32) : U32)),
          Order.right_le(U32.to_nat(first),reached),last_ge),last_below}

# A view read of a coordinate inside its class.
def read_in_pick(fits: Bool,+axis: U32,+step: U32,+first: U32,+source_extent: U32,+extent: U32,+flag: Bool,+block: U32,+source_grid: Grid.Grid,
  +readings: List<&2,Tasks.Reading>,+class_first: U32,+class_count: U32,+source_first: U32,+source_count: U32,+coordinate: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same: {U32.is_lt((first + step * (U32.max(extent,1) - 1 : U32) : U32),source_extent) == fits : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(cap)) == True{} : Bool},
  +whole: {Tasks.crossing(Con{stepped_reading(fits,axis,Grid.block(source_grid,axis,source_extent),step,
    Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block),readings}) == False{} : Bool},
  +class_in: Targets.ClassIn(Tasks.Class{class_first,class_count,source_first,source_count},block,extent,coordinate),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  ReadIn(stepped_reading(fits,axis,Grid.block(source_grid,axis,source_extent),step,Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),
    flag,source_extent,extent,block),Tasks.Class{class_first,class_count,source_first,source_count},coordinate):
  match fits:
    case True{}:
      Held.Both{+low_le,+below} = class_in
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      +reach = U32.min(((class_first + class_count) * block : U32),extent)
      +within = {%Equal.sym(Nat,U32.to_nat(reach),Nat.min(U32.to_nat(((class_first + class_count) * block : U32)),U32.to_nat(extent)),
          Extrema.minimum(((class_first + class_count) * block : U32),extent)) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
        NatExtrema.minimum_right(U32.to_nat(((class_first + class_count) * block : U32)),U32.to_nat(extent)) :
        {Nat.is_le(U32.to_nat(reach),U32.to_nat(extent)) == True{} : Bool}}
      +positive = Order.le_lt_trans(0n,coordinate,U32.to_nat(extent),Order.zero_le(coordinate),Bounds.nat_trans(coordinate,U32.to_nat(reach),U32.to_nat(extent),below,within))
      +high_ge = high_low(reach,extent,coordinate,high_in(reach,extent,coordinate,cap,cap_ok,below,within,extent_room))
      +high_le = high_top(reach,extent,coordinate,high_in(reach,extent,coordinate,cap,cap_ok,below,within,extent_room))
      +last_below = Equal.trans(Bool,Nat.is_lt(U32.to_nat(last),U32.to_nat(source_extent)),U32.is_lt(last,source_extent),True{},
        Equal.sym(Bool,U32.is_lt(last,source_extent),Nat.is_lt(U32.to_nat(last),U32.to_nat(source_extent)),Comparison.lt(last,source_extent)),same)
      +last_ge = reached_le(first,step,extent,(reach - 1 : U32),last_exact(first,step,extent,cap,cap_ok,fit,positive,extent_room),high_le)
      Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{low_le,Held.Both{high_ge,
        inside_of(flag,first,step,extent,source_extent,(reach - 1 : U32),last_ge,last_below)}}}}}}
    case False{}:
      Empty.absurd(ReadIn(Tasks.Reading{extent,block,Tasks.Crossing{}},Tasks.Class{class_first,class_count,source_first,source_count},coordinate),
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},whole)))

def read_in_of(found: Tasks.Stepped,+extent: U32,+stride: U32,+flag: Bool,+block: U32,+source_grid: Grid.Grid,+readings: List<&2,Tasks.Reading>,
  +class_first: U32,+class_count: U32,+source_first: U32,+source_count: U32,+coordinate: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: AxisFit(found,extent,stride,cap),
  +whole: {Tasks.crossing(Con{Tasks.source_reading(source_grid,extent,block,flag,found),readings}) == False{} : Bool},
  +class_in: Targets.ClassIn(Tasks.Class{class_first,class_count,source_first,source_count},block,extent,coordinate),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  ReadIn(Tasks.source_reading(source_grid,extent,block,flag,found),Tasks.Class{class_first,class_count,source_first,source_count},coordinate):
  match found:
    case Tasks.Unstepped{}:
      Unit{}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      read_in_pick(U32.is_lt(last,source_extent),axis,step,first,source_extent,extent,flag,block,source_grid,readings,class_first,class_count,source_first,source_count,
        coordinate,cap,cap_ok,{==},fit,whole,class_in,extent_room)

# Every target axis of a cell inside the classes of its component reads as a
# view does.
def view_row_in(axes: List<&2,Views.Axis>,classes: List<&2,Tasks.Class>,cells: List<&2,Nat>,+flags: List<&2,Bool>,+extents: List<&2,U32>,+source_grid: Grid.Grid,
  +target_grid: Grid.Grid,+fixed: List<&2,U32>,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +target_positive: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +whole: {Tasks.crossing(Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,axis)) == False{} : Bool},
  +fit: AxesFit(extents,fixed,axes,cap),+held: Targets.ClassesIn(classes,target_grid,Views.axis_extents(axes),cells,axis)) ->
  RowIn(Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,axis),classes,cells):
  match axes classes cells:
    case Con{Views.Axis{+extent,+stride},+rest} Con{Tasks.Class{+class_first,+class_count,+source_first,+source_count},+others} Con{+coordinate,+more}:
      Held.Both{+here,+later_fit} = fit
      Held.Both{+class_in,+later_held} = held
      +flag = Views.flag(flags)
      +later_flags = Views.later_flags(flags)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(extents,fixed,stride,0))
      +block = Grid.block(target_grid,axis,extent)
      +next = (axis + 1 : U32)
      +later_readings = Tasks.readings(extents,source_grid,target_grid,fixed,rest,later_flags,next)
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,target_positive)
      Held.Both{read_in_of(found,extent,stride,flag,block,source_grid,later_readings,class_first,class_count,source_first,source_count,coordinate,cap,cap_ok,
          here,whole,class_in,Order.transitive(size,Nat.mul(size,inner),U32.to_nat(cap),RowMajor.product_covers(size,inner,inner_positive),target_room)),
        view_row_in(rest,others,more,later_flags,extents,source_grid,target_grid,fixed,next,cap,cap_ok,
          Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(cap),inner_le(size,inner,target_positive),target_room),inner_positive,
          tail_crossing(found,source_grid,extent,block,flag,later_readings,whole),later_fit,later_held)}
    case Nil{} Nil{} Nil{}:
      Unit{}
    case Nil{} Nil{} Con{+coordinate,+more}:
      Empty.absurd(RowIn(Tasks.readings(extents,source_grid,target_grid,fixed,Nil{},flags,axis),Nil{},Con{coordinate,more}),held)
    case Nil{} Con{+class,+others} +cells:
      Empty.absurd(RowIn(Tasks.readings(extents,source_grid,target_grid,fixed,Nil{},flags,axis),Con{class,others},cells),held)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{} +cells:
      Empty.absurd(RowIn(Tasks.readings(extents,source_grid,target_grid,fixed,Con{Views.Axis{extent,stride},rest},flags,axis),Nil{},cells),held)
    case Con{Views.Axis{+extent,+stride},+rest} Con{+class,+others} Nil{}:
      Empty.absurd(RowIn(Tasks.readings(extents,source_grid,target_grid,fixed,Con{Views.Axis{extent,stride},rest},flags,axis),Con{class,others},Nil{}),held)

# ---- The view address of a target cell inside the source box ----

# Every coordinate of an index lies below its extent.
def coordinates_in(extents: List<&2,U32>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  FixedIn(Grid.coordinates(extents,index),extents):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(size,inner)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_double = Order.transitive(Nat.double(inner),Nat.double(whole),U32.to_nat(cap),Order.double_le(inner,whole,inner_le(size,inner,positive)),room)
      +size_double = Order.transitive(Nat.double(size),Nat.double(whole),U32.to_nat(cap),
        Order.double_le(size,whole,RowMajor.product_covers(size,inner,inner_positive)),room)
      +inner_word = U32.max(Grid.product(rest),1)
      +size_word = U32.max(extent,1)
      +size_exact = Extrema.max_one_exact(extent,size,{==},size_positive)
      +quotient_word = (index / inner_word : U32)
      +digit_word = (index / inner_word % size_word : U32)
      +digit_exact = {%size_exact : {U32.to_nat(digit_word) == Nat.mod(U32.to_nat(quotient_word),_) : Nat}
        U32Division.modulo(quotient_word,size_word,cap,cap_ok,
          %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} size_double,
          %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_lt(0n,_) == True{} : Bool} size_positive) :
        {U32.to_nat(digit_word) == Nat.mod(U32.to_nat(quotient_word),size) : Nat}}
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},Laws.strides_con(extent,rest)) :
        FixedIn(Grid.coordinates_by(Con{extent,rest},_,index),Con{extent,rest})
      Held.Both{{%Equal.sym(Nat,U32.to_nat(digit_word),Nat.mod(U32.to_nat(quotient_word),size),digit_exact) : {Nat.is_lt(_,size) == True{} : Bool}
          DivisionOrder.remainder_small(U32.to_nat(quotient_word),size,size_positive) : {Nat.is_lt(U32.to_nat(digit_word),size) == True{} : Bool}},
        coordinates_in(rest,(index % inner_word : U32),cap,cap_ok,inner_double,inner_positive)}

# Every index below the cells lies inside the whole box.
def whole_in(extents: List<&2,U32>,+value: Nat,+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Components.InBox(Tasks.whole_spans(extents),extents,value):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +inner_positive = Components.inner_positive(size,inner,value,below)
      Held.Both{Order.zero_le(Nat.div(value,inner)),
        Held.Both{DivisionOrder.quotient_below(value,size,inner,inner_positive,below),
          whole_in(rest,Nat.mod(value,inner),DivisionOrder.remainder_small(value,inner,inner_positive))}}

# A source index inside a box of the extents and below their cells.
def InSource(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+value: Nat) -> Data:
  Held.Both<Components.InBox(spans,extents,value),{Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}>

# A target cell inside the classes of its component reads a source index
# inside the box the component reads: the view offset plus its reflected
# digit sum.
def view_box(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+extents: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+offset: U32,
  +classes: List<&2,Tasks.Class>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(extents)) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +apart: {Tasks.distinct(Tasks.read_axes(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0))) == True{} : Bool},
  +whole: {Tasks.crossing(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)) == False{} : Bool},
  +fit: AxesFit(extents,Grid.coordinates(extents,offset),axes,cap),
  +held: Targets.ClassesIn(classes,target_grid,Views.axis_extents(axes),Targets.cell_coordinates(Views.axis_extents(axes),index),0),
  +axes_ok: Components.AxesOk(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0),source_grid,extents,0,cap),
  +blocks: BlocksIn(source_grid,extents,0,cap)) ->
  InSource(Tasks.boxed(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0),classes,source_grid,extents,
      Grid.coordinates(extents,offset),0),extents,Nat.add(U32.to_nat(offset),view_sum(axes,flags,index))):
  +fixed = Grid.coordinates(extents,offset)
  +readings = Tasks.readings(extents,source_grid,target_grid,fixed,axes,flags,0)
  +cells = Targets.cell_coordinates(Views.axis_extents(axes),index)
  +digits = source_digits(readings,cells,extents,fixed,0)
  +box = Tasks.boxed(readings,classes,source_grid,extents,fixed,0)
  +sum = view_sum(axes,flags,index)
  +target_positive = Order.le_lt_trans(0n,index,ShapeLaws.elements(Views.axis_extents(axes)),Order.zero_le(index),inside)
  +row = view_row_in(axes,classes,cells,flags,extents,source_grid,target_grid,fixed,0,cap,cap_ok,target_room,target_positive,whole,fit,held)
  +found = digits_box(box,extents,digits,digits_in(readings,classes,cells,source_grid,extents,fixed,0,cap,cap_ok,row,axes_ok,blocks,
    coordinates_in(extents,offset,cap,cap_ok,size_room,positive)))
  +value = Equal.trans(Nat,ShapeLaws.digits_value(extents,digits),Nat.add(ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,fixed,0)),sum),Nat.add(U32.to_nat(offset),sum),
    view_digits(axes,flags,index,extents,source_grid,target_grid,fixed,0,cap,cap_ok,room,size_room,positive,target_room,inside,apart,whole,fit),
    Equal.cong(Nat,Nat,base => Nat.add(base,sum),ShapeLaws.digits_value(extents,source_digits(Nil{},Nil{},extents,fixed,0)),U32.to_nat(offset),
      coordinates_value(extents,offset,0,cap,cap_ok,size_room,offset_in)))
  %value : InSource(box,extents,_)
  found

# The classes of a component hold a target cell when the view is separable;
# one component of an inseparable view reads every source whole.
def ClassesWhen(separate: Bool,+classes: List<&2,Tasks.Class>,+target_grid: Grid.Grid,+axes: List<&2,Views.Axis>,+cells: List<&2,Nat>) -> Data:
  match separate:
    case True{}: Targets.ClassesIn(classes,target_grid,Views.axis_extents(axes),cells,0)
    case False{}: Unit

def source_choice(separate: Bool,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+extents: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,
  +offset: U32,+classes: List<&2,Tasks.Class>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same: {Tasks.separable([Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)]) == separate : Bool},
  +held: ClassesWhen(separate,classes,target_grid,axes,Targets.cell_coordinates(Views.axis_extents(axes),index)),
  +room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(extents)) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +fit: AxesFit(extents,Grid.coordinates(extents,offset),axes,cap),
  +axes_ok: Components.AxesOk(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0),source_grid,extents,0,cap),
  +blocks: BlocksIn(source_grid,extents,0,cap),
  +address_below: {Nat.is_lt(Nat.add(U32.to_nat(offset),view_sum(axes,flags,index)),ShapeLaws.elements(extents)) == True{} : Bool}) ->
  InSource(Tasks.box_when(separate,Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0),classes,source_grid,extents,offset),
    extents,Nat.add(U32.to_nat(offset),view_sum(axes,flags,index))):
  match separate:
    case True{}:
      +readings = Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)
      +apart_whole = and_left((Tasks.distinct(Tasks.read_axes(readings)) && Bool.not(Tasks.crossing(readings))),Tasks.separable(Nil{}),same)
      +apart = and_left(Tasks.distinct(Tasks.read_axes(readings)),Bool.not(Tasks.crossing(readings)),apart_whole)
      +whole = not_true(Tasks.crossing(readings),and_right(Tasks.distinct(Tasks.read_axes(readings)),Bool.not(Tasks.crossing(readings)),apart_whole))
      view_box(axes,flags,index,extents,source_grid,target_grid,offset,classes,cap,cap_ok,room,size_room,positive,offset_in,target_room,inside,apart,whole,fit,held,
        axes_ok,blocks)
    case False{}:
      +value = Nat.add(U32.to_nat(offset),view_sum(axes,flags,index))
      Held.Both{whole_in(extents,value,address_below),address_below}

# A target cell of a component of a view reads a view address inside the
# source box of the component.
def view_source_box(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+extents: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,
  +offset: U32,+component: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +held: ClassesWhen(Tasks.separable([Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)]),
    Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)]),
      [Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)]),component),
    target_grid,axes,Targets.cell_coordinates(Views.axis_extents(axes),index)),
  +room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +size_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(extents)) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +fit: AxesFit(extents,Grid.coordinates(extents,offset),axes,cap),
  +axes_ok: Components.AxesOk(Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0),source_grid,extents,0,cap),
  +blocks: BlocksIn(source_grid,extents,0,cap),
  +address_below: {Nat.is_lt(Nat.add(U32.to_nat(offset),view_sum(axes,flags,index)),ShapeLaws.elements(extents)) == True{} : Bool}) ->
  InSource(Tasks.source_box(Tasks.components(extents,source_grid,target_grid,Views.View{axes,offset},flags),0,source_grid,extents,offset,component),
    extents,Nat.add(U32.to_nat(offset),view_sum(axes,flags,index))):
  +readings = Tasks.readings(extents,source_grid,target_grid,Grid.coordinates(extents,offset),axes,flags,0)
  +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank([readings]),[readings]),component)
  source_choice(Tasks.separable([readings]),axes,flags,index,extents,source_grid,target_grid,offset,classes,cap,cap_ok,{==},held,room,size_room,positive,offset_in,
    target_room,inside,fit,axes_ok,blocks,address_below)

# ---- The classes of a view end inside the blocks of its target ----

def read_of(+reading: Tasks.Reading) -> Tasks.Source:
  match reading:
    case Tasks.Reading{+extent,+block,+source}: source

def pick_parts(fits: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+flag: Bool,+bound: U32,+extent: U32,+block: U32) ->
  {stepped_reading(fits,axis,source_block,step,offset,flag,bound,extent,block) ==
    Tasks.Reading{extent,block,read_of(stepped_reading(fits,axis,source_block,step,offset,flag,bound,extent,block))} : Tasks.Reading}:
  match fits:
    case True{}: {==}
    case False{}: {==}

# A source reading reads its target extent in its target block.
def reading_parts(found: Tasks.Stepped,+source_grid: Grid.Grid,+extent: U32,+block: U32,+flag: Bool) ->
  {Tasks.source_reading(source_grid,extent,block,flag,found) == Tasks.Reading{extent,block,read_of(Tasks.source_reading(source_grid,extent,block,flag,found))} : Tasks.Reading}:
  match found:
    case Tasks.Unstepped{}: {==}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      pick_parts(U32.is_lt(last,source_extent),axis,Grid.block(source_grid,axis,source_extent),step,Bool.pick(U32,flag,last,first),flag,source_extent,extent,block)

def ends_head(+extent: U32,+block: U32,+source: Tasks.Source,+row: List<&2,Tasks.Reading>,+class: Tasks.Class,+classes: List<&2,Tasks.Class>,
  +ends: Components.RowEnds(Con{Tasks.Reading{extent,block,source},row},Con{class,classes})) ->
  Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  Held.Both{+here,+later} = ends
  here

def ends_tail(+extent: U32,+block: U32,+source: Tasks.Source,+row: List<&2,Tasks.Reading>,+class: Tasks.Class,+classes: List<&2,Tasks.Class>,
  +ends: Components.RowEnds(Con{Tasks.Reading{extent,block,source},row},Con{class,classes})) -> Components.RowEnds(row,classes):
  Held.Both{+here,+later} = ends
  later

def room_sum(+extent: U32,+block: U32,+source: Tasks.Source,+row: List<&2,Tasks.Reading>,+cap: U32,
  +rooms: Components.ReadingsRoom(Con{Tasks.Reading{extent,block,source},row},cap)) ->
  {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}:
  Held.Both{Held.Both{Held.Both{+blocks,Held.Both{+ceiling,Held.Both{+positive,+sum}}},+read},+later} = rooms
  sum

def rooms_tail(+extent: U32,+block: U32,+source: Tasks.Source,+row: List<&2,Tasks.Reading>,+cap: U32,
  +rooms: Components.ReadingsRoom(Con{Tasks.Reading{extent,block,source},row},cap)) -> Components.ReadingsRoom(row,cap):
  Held.Both{+here,+later} = rooms
  later

# The classes a row of view readings ends inside end inside the blocks of
# the target.
def view_class_ends(+source: List<&2,U32>,+source_grid: Grid.Grid,+grid: Grid.Grid,+coordinates: List<&2,U32>,axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,
  +axis: U32,classes: List<&2,Tasks.Class>,+cap: U32,+same: {List.length(&2,Tasks.Class,classes) == List.length(&2,Views.Axis,axes) : Nat},
  +rooms: Components.ReadingsRoom(Tasks.readings(source,source_grid,grid,coordinates,axes,flags,axis),cap),
  +ends: Components.RowEnds(Tasks.readings(source,source_grid,grid,coordinates,axes,flags,axis),classes)) ->
  Targets.ClassEnds(classes,grid,Views.axis_extents(axes),axis,cap):
  match axes classes:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+class,+more}:
      +flag = Views.flag(flags)
      +block = Grid.block(grid,axis,extent)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(source,coordinates,stride,0))
      +read = read_of(Tasks.source_reading(source_grid,extent,block,flag,found))
      +later = Tasks.readings(source,source_grid,grid,coordinates,rest,Views.drop_flags(1n,flags),(axis + 1 : U32))
      +parts = reading_parts(found,source_grid,extent,block,flag)
      +split_ends = {%parts : Components.RowEnds(Con{_,later},Con{class,more}) ends : Components.RowEnds(Con{Tasks.Reading{extent,block,read},later},Con{class,more})}
      +split_rooms = {%parts : Components.ReadingsRoom(Con{_,later},cap) rooms : Components.ReadingsRoom(Con{Tasks.Reading{extent,block,read},later},cap)}
      Held.Both{ends_head(extent,block,read,later,class,more,split_ends),
        Held.Both{room_sum(extent,block,read,later,cap,split_rooms),
          view_class_ends(source,source_grid,grid,coordinates,rest,Views.drop_flags(1n,flags),(axis + 1 : U32),more,cap,
            Components.length_back(List.length(&2,Tasks.Class,more),List.length(&2,Views.Axis,rest),same),rooms_tail(extent,block,read,later,cap,split_rooms),
            ends_tail(extent,block,read,later,class,more,split_ends))}}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+class,+more}:
      Empty.absurd(Targets.ClassEnds(Con{class,more},grid,Views.axis_extents(Nil{}),axis,cap),
        Components.zero_successor(List.length(&2,Tasks.Class,more),Equal.sym(Nat,1n+List.length(&2,Tasks.Class,more),0n,same)))
    case Con{+head,+rest} Nil{}:
      Empty.absurd(Targets.ClassEnds(Nil{},grid,Views.axis_extents(Con{head,rest}),axis,cap),Components.zero_successor(List.length(&2,Views.Axis,rest),same))

# A view reads one reading of every axis.
def readings_length(+source: List<&2,U32>,+source_grid: Grid.Grid,+grid: Grid.Grid,+coordinates: List<&2,U32>,axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,
  +axis: U32) -> {List.length(&2,Tasks.Reading,Tasks.readings(source,source_grid,grid,coordinates,axes,flags,axis)) == List.length(&2,Views.Axis,axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Equal.cong(Nat,Nat,value => 1n+value,
        List.length(&2,Tasks.Reading,Tasks.readings(source,source_grid,grid,coordinates,rest,Views.drop_flags(1n,flags),(axis + 1 : U32))),List.length(&2,Views.Axis,rest),
        readings_length(source,source_grid,grid,coordinates,rest,Views.drop_flags(1n,flags),(axis + 1 : U32)))

# ---- The address the copy order gives an index ----

def points_sum(+axes: List<&2,Views.Axis>,+index: Nat) -> {RowMajorCoordinates.weight(RowMajorCoordinates.points_of(axes,index)) == view_sum(axes,[],index) : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +inner = ShapeLaws.elements(Views.axis_extents(tail))
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.div(index,inner),U32.to_nat(stride)),value),
        RowMajorCoordinates.weight(RowMajorCoordinates.points_of(tail,Nat.mod(index,inner))),view_sum(tail,[],Nat.mod(index,inner)),points_sum(tail,Nat.mod(index,inner)))

# The copy order reads an index of a view at its offset plus its sum.
def address_sum(+axes: List<&2,Views.Axis>,+offset: U32,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {CopyValues.address(Views.View{axes,offset},index) == U32.from_nat(Nat.add(U32.to_nat(offset),view_sum(axes,[],index))) : U32}:
  +points = RowMajorCoordinates.points_of(axes,index)
  +sum = view_sum(axes,[],index)
  +base = U32.to_nat(offset)
  %RowMajorCoordinates.axes_of(axes,index) :
    {CopyValues.address(Views.View{_,offset},index) == U32.from_nat(Nat.add(base,sum)) : U32}
  %RowMajorCoordinates.position_of(axes,index,0n,inside) :
    {CopyValues.address(Views.View{RowMajorCoordinates.axes(points),offset},_) == U32.from_nat(Nat.add(base,sum)) : U32}
  Equal.trans(U32,CopyValues.address(Views.View{RowMajorCoordinates.axes(points),offset},RowMajorCoordinates.position(points,0n)),
    U32.from_nat(Nat.add(RowMajorCoordinates.weight(points),base)),U32.from_nat(Nat.add(base,sum)),
    RowMajorCoordinates.weighed(points,offset,RowMajorCoordinates.valid_of(axes,index,inside)),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(RowMajorCoordinates.weight(points),base),Nat.add(base,sum),
      Equal.trans(Nat,Nat.add(RowMajorCoordinates.weight(points),base),Nat.add(sum,base),Nat.add(base,sum),
        Equal.cong(Nat,Nat,value => Nat.add(value,base),RowMajorCoordinates.weight(points),sum,points_sum(axes,index)),Order.commutative(sum,base))))

# The copy order reads an index of a view at its offset plus the strides times
# the digits of the index.
def copy_address(+axes: List<&2,Views.Axis>,+offset: U32,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {CopyValues.address(Views.View{axes,offset},index) == (offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index) : U32) : U32}:
  +sum = view_sum(axes,[],index)
  +base = U32.to_nat(offset)
  Equal.trans(U32,CopyValues.address(Views.View{axes,offset},index),U32.from_nat(Nat.add(base,sum)),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index) : U32),
    address_sum(axes,offset,index,inside),
    %MirrorAddress.unmirrored_offset(axes,offset) :
      {U32.from_nat(Nat.add(base,sum)) == (_ + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),index) : U32) : U32}
    %MirrorAddress.unmirrored_axes(axes) :
      {U32.from_nat(Nat.add(base,sum)) == (Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(_),index) : U32) : U32}
    Equal.sym(U32,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),index) : U32),U32.from_nat(Nat.add(base,sum)),
      view_address(axes,[],offset,index,inside)))

def boxed_extents(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+same: ViewPart.SameLength(axes,boxes)) ->
  {Views.axis_extents(ViewPart.boxed_axes(axes,boxes)) == boxes : List<&2,U32>}:
  match axes boxes:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => box <> tail,Views.axis_extents(ViewPart.boxed_axes(rest,later)),later,boxed_extents(rest,later,same))
    case Nil{} Con{+box,+later}: Empty.absurd({Views.axis_extents(ViewPart.boxed_axes(Nil{},Con{box,later})) == Con{box,later} : List<&2,U32>},same)
    case Con{+axis,+rest} Nil{}: Empty.absurd({Views.axis_extents(ViewPart.boxed_axes(Con{axis,rest},Nil{})) == Nil{} : List<&2,U32>},same)

def boxed_steps(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+same: ViewPart.SameLength(axes,boxes)) ->
  {ViewPart.view_steps(ViewPart.boxed_axes(axes,boxes)) == ViewPart.view_steps(axes) : List<&2,U32>}:
  match axes boxes:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => stride <> tail,ViewPart.view_steps(ViewPart.boxed_axes(rest,later)),ViewPart.view_steps(rest),boxed_steps(rest,later,same))
    case Nil{} Con{+box,+later}: Empty.absurd({ViewPart.view_steps(ViewPart.boxed_axes(Nil{},Con{box,later})) == ViewPart.view_steps(Nil{}) : List<&2,U32>},same)
    case Con{+axis,+rest} Nil{}: Empty.absurd({ViewPart.view_steps(ViewPart.boxed_axes(Con{axis,rest},Nil{})) == ViewPart.view_steps(Con{axis,rest}) : List<&2,U32>},same)

# A view read without mirrored axes reads an index where the copy order does.
def viewed_address(+axes: List<&2,Views.Axis>,+offset: U32,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),index) : U32) ==
    CopyValues.address(Views.View{axes,offset},index) : U32}:
  Equal.trans(U32,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),index) : U32),
    U32.from_nat(Nat.add(U32.to_nat(offset),view_sum(axes,[],index))),CopyValues.address(Views.View{axes,offset},index),
    view_address(axes,[],offset,index,inside),
    Equal.sym(U32,CopyValues.address(Views.View{axes,offset},index),U32.from_nat(Nat.add(U32.to_nat(offset),view_sum(axes,[],index))),address_sum(axes,offset,index,inside)))

# One less keeps the order of two numbers.
def less_one(+low: Nat,+high: Nat,+ordered: {Nat.is_le(low,high) == True{} : Bool}) -> {Nat.is_le(Nat.sub(low,1n),Nat.sub(high,1n)) == True{} : Bool}:
  match low high:
    case 0n +high: Order.zero_le(Nat.sub(high,1n))
    case 1n+ +below 0n: Empty.absurd({Nat.is_le(Nat.sub(1n+below,1n),Nat.sub(0n,1n)) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+ +below 1n+ +above:
      %Equal.sym(Nat,Nat.sub(1n+below,1n),below,Order.subtract_zero(below)) : {Nat.is_le(_,Nat.sub(1n+above,1n)) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(1n+above,1n),above,Order.subtract_zero(above)) : {Nat.is_le(below,_) == True{} : Bool}
      ordered

# A box of a view whose axes span no more than the axes of the view ends no
# further than the view.
def boxed_span(axes: List<&2,Views.Axis>,boxes: List<&2,U32>,+within: Box.BoxWithin(boxes,Views.axis_extents(axes)),
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(boxes)) == True{} : Bool}) ->
  {Nat.is_le(ViewBounds.last_span(ViewPart.boxed_axes(axes,boxes)),ViewBounds.last_span(axes)) == True{} : Bool}:
  match axes boxes:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      Held.Both{+head,+tail} = within
      +size = U32.to_nat(box)
      +inner = ShapeLaws.elements(later)
      +box_one = Order.positive_one(size,RowMajor.left_factor_positive(size,inner,positive))
      +extent_one = Order.transitive(1n,size,U32.to_nat(extent),box_one,head)
      +box_last = FullRange.subtract(box,1,box,box_one,Order.sub_le(size,1n))
      +extent_last = FullRange.subtract(extent,1,extent,extent_one,Order.sub_le(U32.to_nat(extent),1n))
      sum_le(Nat.mul(U32.to_nat((box - 1 : U32)),U32.to_nat(stride)),ViewBounds.last_span(ViewPart.boxed_axes(rest,later)),
        Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ViewBounds.last_span(rest),
        {%Equal.sym(Nat,U32.to_nat((box - 1 : U32)),Nat.sub(size,1n),box_last) :
            {Nat.is_le(Nat.mul(_,U32.to_nat(stride)),Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),extent_last) :
            {Nat.is_le(Nat.mul(Nat.sub(size,1n),U32.to_nat(stride)),Nat.mul(_,U32.to_nat(stride))) == True{} : Bool}
          RowMajor.mul_le(Nat.sub(size,1n),Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride),less_one(size,U32.to_nat(extent),head)) :
          {Nat.is_le(Nat.mul(U32.to_nat((box - 1 : U32)),U32.to_nat(stride)),Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))) == True{} : Bool}},
        boxed_span(rest,later,tail,DivisionOrder.right_factor_positive(size,inner,positive)))
    case Nil{} +boxes: Order.zero_le(0n)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}: Order.zero_le(ViewBounds.last_span(Con{Views.Axis{extent,stride},rest}))

# The view of a piece of the grid of the extents of a view is the box of the
# piece at the address of its start.
def piece_view_boxed(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+piece: U32,
  +start_in: {Nat.is_lt(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}) == Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)} : Views.View}:
  +same = ViewPart.same_box_of(axes,Grid.grid(Views.axis_extents(axes),target),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))
  %ViewPart.view_eta(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),offset)) : {_ == Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)} : Views.View}
  %Equal.sym(List<&2,Views.Axis>,ViewPart.view_axes_of(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),offset)),ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),ViewPart.restricted_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),offset)) :
    {Views.View{_,ViewPart.view_offset(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),offset))} == Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)} : Views.View}
  %Equal.sym(U32,ViewPart.view_offset(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),offset)),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32),
      ViewPart.restricted_offset(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),offset,Machine.half_word(limits),LimitBounds.half_word_fits(limits),
        LimitBounds.double_bound(limits,ShapeLaws.elements(Views.axis_extents(axes)),room),start_in,same)) :
    {Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),_} == Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)} : Views.View}
  {==}

def piece_address_with(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+piece: U32,+index: Nat,
  +boxed: Box.PieceBox(Views.axis_extents(axes),ViewPart.view_steps(axes),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),index)) ->
  {CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),index) ==
    CopyValues.address(Views.View{axes,offset},Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  +same = ViewPart.same_box_of(axes,Grid.grid(Views.axis_extents(axes),target),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))
  +start_in = Order.le_lt_trans(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),ShapeLaws.elements(Views.axis_extents(axes)),Order.le_add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),ending)
  +moved = (Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)
  +boxed_view = {Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)} : Views.View}
  +boxed_in = {%Equal.sym(List<&2,U32>,Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),same)) :
      {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool} index_box : {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))))) == True{} : Bool}}
  Equal.trans(U32,CopyValues.address(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),offset),index),((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),CopyValues.address(Views.View{axes,offset},Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)),
    %Equal.sym(Views.View,Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)},piece_view_boxed(limits,axes,offset,target,room,piece,start_in)) :
      {CopyValues.address(_,index) == ((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32) : U32}
    Equal.trans(U32,CopyValues.address(boxed_view,index),
      ((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),ViewPart.view_steps(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),index) : U32),
      ((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),
      copy_address(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32),index,boxed_in),
      %Equal.sym(List<&2,U32>,Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),same)) :
        {((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(_,ViewPart.view_steps(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),index) : U32) == ((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32) : U32}
      %Equal.sym(List<&2,U32>,ViewPart.view_steps(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),ViewPart.view_steps(axes),boxed_steps(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),same)) :
        {((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),_,index) : U32) == ((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32) : U32}
      {==}),
    Equal.trans(U32,((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),CopyValues.address(Views.View{axes,offset},Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)),
      Equal.trans(U32,((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),(offset + (Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32) : U32),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),
        Arithmetic.add_associative(offset,Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))),Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index)),
        Equal.cong(U32,U32,value => (offset + value : U32),(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)),
          Equal.trans(U32,(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) + Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) : U32),(Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32),Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)),Arithmetic.add_commutative(Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))),Box.stepped_sum(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ViewPart.view_steps(axes),index)),shift))),
      Equal.sym(U32,CopyValues.address(Views.View{axes,offset},Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),copy_address(axes,offset,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),ending))))

# The view of a piece of the grid of the extents of a view reads index index
# of the piece where the view reads the start of the piece plus index.
def piece_address(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  {CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),index) ==
    CopyValues.address(Views.View{axes,offset},Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32}:
  piece_address_with(limits,axes,offset,target,room,piece,index,
    Box.grid_piece_box(limits,Views.axis_extents(axes),ViewPart.view_steps(axes),target,room,nonempty,rank_room,piece,inside,index,index_in))

# A nonempty view fits a length by the walk to its last cell.
def nonempty_fits(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},length) == True{} : Bool}) -> {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}:
  %CountCompletion.positive_not_zero(Views.axis_extents(axes),nonempty) : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool}
  fits

# The view of a piece of the grid of the extents of a view fits every length
# the view fits: it ends at the address of its last cell.
def piece_fits_with(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+piece: U32,+length: U32,+length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +within: Box.BoxWithin(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Views.axis_extents(axes)),
  +boxed: Box.PieceBox(Views.axis_extents(axes),ViewPart.view_steps(axes),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Nat.sub(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),1n))) ->
  {Views.fits(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),length) == True{} : Bool}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  +capacity = U32.to_nat(Machine.capacity(limits))
  +length_nat = U32.to_nat(length)
  +base = U32.to_nat(offset)
  +span = ViewBounds.last_span(axes)
  +box_span = ViewBounds.last_span(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))
  +start_nat = U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))
  +last = Nat.sub(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),1n)
  +same = ViewPart.same_box_of(axes,Grid.grid(Views.axis_extents(axes),target),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))
  +start_in = Order.le_lt_trans(start_nat,Nat.add(start_nat,last),ShapeLaws.elements(Views.axis_extents(axes)),Order.le_add(start_nat,last),ending)
  +boxed_view = Equal.sym(Views.View,Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)},piece_view_boxed(limits,axes,offset,target,room,piece,start_in))
  +counted = {%Equal.sym(List<&2,U32>,Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),same)) : {ShapeLaws.elements(_) == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)) : Nat} box_cells :
    {ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))) == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)) : Nat}}
  +last_in = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))),ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),
      Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),same))) :
      {Nat.is_lt(last,_) == True{} : Bool}
    index_box : {Nat.is_lt(last,ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))))) == True{} : Bool}}
  +positive = Order.le_lt_trans(0n,last,ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))),Order.zero_le(last),last_in)
  +box_positive = Order.le_lt_trans(0n,last,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),Order.zero_le(last),index_box)
  +last_sum_eq = {%counted : {view_sum(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),[],Nat.sub(_,1n)) == box_span : Nat} last_sum(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),positive) :
    {view_sum(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),[],last) == box_span : Nat}}
  +start_below = {%copy_address(axes,offset,start_nat,start_in) : {Nat.is_lt(U32.to_nat(_),length_nat) == True{} : Bool}
    CopyValues.address_inside(Views.View{axes,offset},start_nat,length,fits) : {Nat.is_lt(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),length_nat) == True{} : Bool}}
  +span_in = {%ViewBounds.last_location(axes,base) : {Nat.is_lt(_,length_nat) == True{} : Bool} ViewBounds.last_address(axes,offset,length,fits) :
    {Nat.is_lt(Nat.add(base,span),length_nat) == True{} : Bool}}
  +start_room = Order.transitive(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),length_nat,capacity,Bounds.strict_le(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),length_nat,start_below),length_room)
  +box_room = Order.transitive(box_span,span,capacity,boxed_span(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),within,box_positive),
    Order.transitive(span,Nat.add(base,span),capacity,Order.right_le(base,span),
      Order.transitive(Nat.add(base,span),length_nat,capacity,Bounds.strict_le(Nat.add(base,span),length_nat,span_in),length_room)))
  +end = Nat.add(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),box_span)
  +doubled = {%Equal.sym(Nat,Nat.add(capacity,capacity),Nat.double(capacity),Addition.double(capacity)) :
      {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    LimitBounds.double_bound(limits,capacity,Order.reflexive(capacity)) : {Nat.is_le(Nat.add(capacity,capacity),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +round = FullRange.roundtrip(end,Machine.half_word(limits),
    Order.transitive(end,Nat.add(capacity,capacity),U32.to_nat(Machine.half_word(limits)),sum_le(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),box_span,capacity,capacity,start_room,box_room),doubled))
  +read_end = Equal.trans(U32,CopyValues.address(Views.View{axes,offset},Nat.add(start_nat,last)),CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),last),U32.from_nat(end),
    Equal.sym(U32,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),last),CopyValues.address(Views.View{axes,offset},Nat.add(start_nat,last)),piece_address_with(limits,axes,offset,target,room,piece,last,boxed)),
    %boxed_view : {CopyValues.address(_,last) == U32.from_nat(end) : U32}
    %last_sum_eq : {CopyValues.address(Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)},last) == U32.from_nat(Nat.add(U32.to_nat((offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32)),_)) : U32}
    address_sum(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32),last,last_in))
  +end_below = {%round : {Nat.is_lt(_,length_nat) == True{} : Bool}
    %read_end : {Nat.is_lt(U32.to_nat(_),length_nat) == True{} : Bool}
    CopyValues.address_inside(Views.View{axes,offset},Nat.add(start_nat,last),length,fits) : {Nat.is_lt(end,length_nat) == True{} : Bool}}
  %boxed_view : {Views.fits(_,length) == True{} : Bool}
  ViewBounds.fits_span(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece))) : U32),length,CountCompletion.positive_not_zero(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),positive),end_below)

def piece_fits(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+length: U32,+length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}) ->
  {Views.fits(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.View{axes,offset}),length) == True{} : Bool}:
  +last_in = Order.last_below(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Targets.grid_span_positive(limits,Views.axis_extents(axes),target,room,nonempty,piece,inside))
  piece_fits_with(limits,axes,offset,target,room,piece,length,length_room,fits,
    Box.grid_within(limits,Views.axis_extents(axes),target,room,nonempty,rank_room,piece,inside),
    Box.grid_piece_box(limits,Views.axis_extents(axes),ViewPart.view_steps(axes),target,room,nonempty,rank_room,piece,inside,Nat.sub(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),1n),last_in))

# ---- The box view of a piece ----

# The box view of extents opens with the first extent stepping over the
# cells of the rest.
def dense_con(+extent: U32,+rest: List<&2,U32>) -> {Grid.dense_axes(Con{extent,rest}) == Con{Views.Axis{extent,Grid.product(rest)},Grid.dense_axes(rest)} : List<&2,Views.Axis>}:
  Equal.cong(List<&2,U32>,List<&2,Views.Axis>,strides => Grid.axes_by(Con{extent,rest},strides),Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},
    Laws.strides_con(extent,rest))

# The box view of a piece reads the extents of its box.
def dense_extents(+extents: List<&2,U32>) -> {Views.axis_extents(Grid.dense_axes(extents)) == extents : List<&2,U32>}:
  match extents:
    case []: {==}
    case +extent <> +rest:
      %Equal.sym(List<&2,Views.Axis>,Grid.dense_axes(Con{extent,rest}),Con{Views.Axis{extent,Grid.product(rest)},Grid.dense_axes(rest)},dense_con(extent,rest)) :
        {Views.axis_extents(_) == Con{extent,rest} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(Grid.dense_axes(rest)),rest,dense_extents(rest))

# The cells inside an axis of a box of no more cells than a word.
def inner_cells(+extent: U32,+rest: List<&2,U32>,+limits: Machine.Limits,
  +room: {Nat.is_le(ShapeLaws.elements(extent <> rest),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extent <> rest)) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(rest)) == ShapeLaws.elements(rest) : Nat}:
  +size = U32.to_nat(extent)
  +inner = ShapeLaws.elements(rest)
  %Digits.product_from(rest) : {U32.to_nat(_) == inner : Nat}
  FullRange.roundtrip(inner,Machine.capacity(limits),
    Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(size,inner,RowMajor.left_factor_positive(size,inner,positive)),room))

# Every axis of a box view steps over the cells inside it.
def dense_size(+extents: List<&2,U32>,+limit: U32,+maximum: {limit == Machine.word_max() : U32},+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Views.contiguous_size(Grid.dense_axes(extents)) == Some{Grid.product(extents)} : Maybe<&2,U32>}:
  match extents:
    case []: {==}
    case +extent <> +rest:
      %Equal.sym(List<&2,Views.Axis>,Grid.dense_axes(Con{extent,rest}),Con{Views.Axis{extent,Grid.product(rest)},Grid.dense_axes(rest)},dense_con(extent,rest)) :
        {Views.contiguous_size(_) == Some{Grid.product(Con{extent,rest})} : Maybe<&2,U32>}
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +inner_exact = inner_cells(extent,rest,limits,room,positive)
      Equal.trans(Maybe<&2,U32>,Views.contiguous_size(Views.Axis{extent,Grid.product(rest)} <> Grid.dense_axes(rest)),Shapes.checked_multiply(extent,Grid.product(rest)),
        Some{Grid.product(extent <> rest)},
        Canonical.prepended_size(extent,Grid.product(rest),Grid.dense_axes(rest),
          dense_size(rest,limit,maximum,limits,Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(size,inner,RowMajor.left_factor_positive(size,inner,positive)),room),
            DivisionOrder.right_factor_positive(size,inner,positive))),
        CountCompletion.checked_product(extent,Grid.product(rest),limit,maximum,
          Order.transitive(Nat.mul(size,U32.to_nat(Grid.product(rest))),U32.to_nat(Machine.capacity(limits)),U32.to_nat(limit),
            %Equal.sym(Nat,U32.to_nat(Grid.product(rest)),inner,inner_exact) : {Nat.is_le(Nat.mul(size,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room,
            Bounds.as_nat_le(Machine.capacity(limits),limit,
              %Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(Machine.capacity(limits),_) == True{} : Bool}
              FullRange.maximum(Machine.capacity(limits))))))

# A box view is contiguous.
def dense_contiguous(+extents: List<&2,U32>,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Views.is_contiguous(Views.View{Grid.dense_axes(extents),0}) == True{} : Bool}:
  Canonical.contiguous_nonempty(Shapes.has_zero(Views.axis_extents(Grid.dense_axes(extents))),Grid.dense_axes(extents),
    %Equal.sym(Maybe<&2,U32>,Views.contiguous_size(Grid.dense_axes(extents)),Some{Grid.product(extents)},dense_size(extents,Machine.word_max(),{==},limits,room,positive)) :
      {Views.has_count(_) == True{} : Bool}
    {==})

# A box view ends at its last cell.
def dense_span(+extents: List<&2,U32>,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {ViewBounds.last_span(Grid.dense_axes(extents)) == Nat.sub(ShapeLaws.elements(extents),1n) : Nat}:
  match extents:
    case []: {==}
    case +extent <> +rest:
      %Equal.sym(List<&2,Views.Axis>,Grid.dense_axes(Con{extent,rest}),Con{Views.Axis{extent,Grid.product(rest)},Grid.dense_axes(rest)},dense_con(extent,rest)) :
        {ViewBounds.last_span(_) == Nat.sub(ShapeLaws.elements(Con{extent,rest}),1n) : Nat}
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +extent_last = FullRange.subtract(extent,1,extent,Order.positive_one(size,size_positive),Order.sub_le(size,1n))
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(size,1n),extent_last) :
        {Nat.add(Nat.mul(_,U32.to_nat(Grid.product(rest))),ViewBounds.last_span(Grid.dense_axes(rest))) == Nat.sub(Nat.mul(size,inner),1n) : Nat}
      %Equal.sym(Nat,U32.to_nat(Grid.product(rest)),inner,inner_cells(extent,rest,limits,room,positive)) :
        {Nat.add(Nat.mul(Nat.sub(size,1n),_),ViewBounds.last_span(Grid.dense_axes(rest))) == Nat.sub(Nat.mul(size,inner),1n) : Nat}
      %Equal.sym(Nat,ViewBounds.last_span(Grid.dense_axes(rest)),Nat.sub(inner,1n),
          dense_span(rest,limits,Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive)) :
        {Nat.add(Nat.mul(Nat.sub(size,1n),inner),_) == Nat.sub(Nat.mul(size,inner),1n) : Nat}
      Equal.sym(Nat,Nat.sub(Nat.mul(size,inner),1n),Nat.add(Nat.mul(Nat.sub(size,1n),inner),Nat.sub(inner,1n)),last_split(size,inner,size_positive,inner_positive))

# A box view fits a length of its cells.
def dense_fits(+extents: List<&2,U32>,+length: U32,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +exact: {U32.to_nat(length) == ShapeLaws.elements(extents) : Nat}) ->
  {Views.fits(Views.View{Grid.dense_axes(extents),0},length) == True{} : Bool}:
  ViewBounds.fits_span(Grid.dense_axes(extents),0,length,
    %Equal.sym(List<&2,U32>,Views.axis_extents(Grid.dense_axes(extents)),extents,dense_extents(extents)) : {Shapes.has_zero(_) == False{} : Bool}
      CountCompletion.positive_not_zero(extents,positive),
    %Equal.sym(Nat,ViewBounds.last_span(Grid.dense_axes(extents)),Nat.sub(ShapeLaws.elements(extents),1n),dense_span(extents,limits,room,positive)) :
      {Nat.is_lt(Nat.add(U32.to_nat(0),_),U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(extents),exact) :
      {Nat.is_lt(Nat.add(U32.to_nat(0),Nat.sub(ShapeLaws.elements(extents),1n)),_) == True{} : Bool}
    Order.last_below(ShapeLaws.elements(extents),positive))

# A box view counts the cells of its box.
def dense_numel(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.numel(Views.View{Grid.dense_axes(extents),0}) == Some{count} : Maybe<&2,U32>}:
  %Equal.sym(List<&2,U32>,Views.axis_extents(Grid.dense_axes(extents)),extents,dense_extents(extents)) : {Shapes.numel(Shapes.Shape{_}) == Some{count} : Maybe<&2,U32>}
  CountCompletion.numel(extents,count,exact)

# ---- A stretched view ----

def found_axes(view: Maybe<&2,Views.View>) -> List<&2,Views.Axis>:
  match view:
    case Some{Views.View{axes,offset}}: axes
    case None{}: []

def AxisExtent(axis: Maybe<&2,Views.Axis>,+extent: U32) -> Data:
  match axis:
    case Some{Views.Axis{size,stride}}: {size == extent : U32}
    case None{}: Unit

def StretchedExtents(result: Maybe<&2,List<&2,Views.Axis>>,+target: List<&2,U32>) -> Data:
  match result:
    case Some{out}: {Views.axis_extents(out) == target : List<&2,U32>}
    case None{}: Unit

def stretched_axis_extent(same: Bool,single: Bool,+size: U32,+stride: U32,+extent: U32,equal: {U32.is_eq(size,extent) == same : Bool}) ->
  AxisExtent(Views.stretched_axis(same,single,size,stride,extent),extent):
  match same single:
    case True{} True{}: Arithmetic.equal_of_is_eq(size,extent,equal)
    case True{} False{}: Arithmetic.equal_of_is_eq(size,extent,equal)
    case False{} True{}: {==}
    case False{} False{}: Unit{}

def prepended_extents(axis: Maybe<&2,Views.Axis>,rest: Maybe<&2,List<&2,Views.Axis>>,+extent: U32,+others: List<&2,U32>,head: AxisExtent(axis,extent),
  tail: StretchedExtents(rest,others)) -> StretchedExtents(Views.stretched_prepend(axis,rest),extent <> others):
  match axis rest:
    case Some{Views.Axis{+size,+stride}} Some{+out}:
      %Equal.sym(U32,size,extent,head) : {_ <> Views.axis_extents(out) == extent <> others : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => extent <> extents,Views.axis_extents(out),others,tail)
    case Some{axis} None{}: Unit{}
    case None{} Some{out}: Unit{}
    case None{} None{}: Unit{}

# A stretch that succeeds reads the extents it stretched to.
def stretched_extents(+axes: List<&2,Views.Axis>,+target: List<&2,U32>) -> StretchedExtents(Views.stretch_axes(axes,target),target):
  match axes target:
    case Nil{} Nil{}: {==}
    case Nil{} Con{extent,rest}: Unit{}
    case Con{axis,tail} Nil{}: Unit{}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
      prepended_extents(Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent),Views.stretch_axes(tail,rest),extent,rest,
        stretched_axis_extent(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent,{==}),stretched_extents(tail,rest))

def stretch_found(result: Maybe<&2,List<&2,Views.Axis>>,+source: List<&2,Views.Axis>,+source_offset: U32,+target: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,
  equation: {Views.stretch_axes(source,target) == result : Maybe<&2,List<&2,Views.Axis>>},
  seen: {Views.stretched_view(result,source_offset) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}) -> {Views.axis_extents(axes) == target : List<&2,U32>}:
  match result:
    case Some{+out}:
      %Equal.cong(Maybe<&2,Views.View>,List<&2,Views.Axis>,view => found_axes(view),Some{Views.View{out,source_offset}},Some{Views.View{axes,offset}},seen) :
        {Views.axis_extents(_) == target : List<&2,U32>}
      %equation : StretchedExtents(_,target)
      stretched_extents(source,target)
    case None{}:
      Empty.absurd({Views.axis_extents(axes) == target : List<&2,U32>},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,view => Maybe.is_some(&2,Views.View,view),None{},Some{Views.View{axes,offset}},seen)))

# A view stretched to the extents target reads those extents.
def stretch_extents(view: Maybe<&2,Views.View>,+target: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,
  stretched: {Tensors.stretched(view,target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}) -> {Views.axis_extents(axes) == target : List<&2,U32>}:
  match view:
    case Some{Views.View{+source,+source_offset}}:
      stretch_found(Views.stretch_axes(Views.with_rank(source,List.length(&2,U32,target)),target),Views.with_rank(source,List.length(&2,U32,target)),source_offset,target,
        axes,offset,{==},stretched)
    case None{}:
      Empty.absurd({Views.axis_extents(axes) == target : List<&2,U32>},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,view => Maybe.is_some(&2,Views.View,view),None{},Some{Views.View{axes,offset}},stretched)))

# ---- A view stretched to the extents it has ----

def sub_self(index: Nat) -> {Nat.sub(index,index) == 0n : Nat}:
  match index:
    case 0n: {==}
    case 1n+ +rest: sub_self(rest)

# Axes stretched to their own extents stay as they are.
def stretch_own(+axes: List<&2,Views.Axis>) -> {Views.stretch_axes(axes,Views.axis_extents(axes)) == Some{axes} : Maybe<&2,List<&2,Views.Axis>>}:
  match axes:
    case []: {==}
    case Views.Axis{+size,+stride} <> +tail:
      %Equal.sym(Bool,U32.is_eq(size,size),True{},word_same(size)) :
        {Views.stretched_prepend(Views.stretched_axis(_,U32.is_eq(size,1),size,stride,size),Views.stretch_axes(tail,Views.axis_extents(tail))) ==
          Some{Views.Axis{size,stride} <> tail} : Maybe<&2,List<&2,Views.Axis>>}
      %Equal.sym(Maybe<&2,List<&2,Views.Axis>>,Views.stretch_axes(tail,Views.axis_extents(tail)),Some{tail},stretch_own(tail)) :
        {Views.stretched_prepend(Some{Views.Axis{size,stride}},_) == Some{Views.Axis{size,stride} <> tail} : Maybe<&2,List<&2,Views.Axis>>}
      {==}

# Axes brought to their own rank stay as they are.
def rank_own(+axes: List<&2,Views.Axis>) -> {Views.with_rank(axes,List.length(&2,U32,Views.axis_extents(axes))) == axes : List<&2,Views.Axis>}:
  +rank = List.length(&2,Views.Axis,axes)
  %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes)),rank,MetadataLaws.extents_length(axes)) : {Views.with_rank(axes,_) == axes : List<&2,Views.Axis>}
  %Equal.sym(Nat,Nat.sub(rank,rank),0n,sub_self(rank)) :
    {List.append(&2,Views.Axis,List.replicate(Views.Axis,_,Views.Axis{1,0}),axes) == axes : List<&2,Views.Axis>}
  {==}

# A view stretched to its own extents is the view.
def stretch_same(+view: Views.View) -> {Views.stretch(view,Views.axis_extents_of(view)) == Some{view} : Maybe<&2,Views.View>}:
  Views.View{+axes,+offset} = view
  %Equal.sym(List<&2,Views.Axis>,Views.with_rank(axes,List.length(&2,U32,Views.axis_extents(axes))),axes,rank_own(axes)) :
    {Views.stretched_view(Views.stretch_axes(_,Views.axis_extents(axes)),offset) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}
  %Equal.sym(Maybe<&2,List<&2,Views.Axis>>,Views.stretch_axes(axes,Views.axis_extents(axes)),Some{axes},stretch_own(axes)) :
    {Views.stretched_view(_,offset) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}
  {==}

def view_or(view: Maybe<&2,Views.View>,+default: Views.View) -> Views.View:
  match view:
    case Some{found}: found
    case None{}: default

# The view of a shape stretched to the shape is its row-major view.
def stretched_own(+shape: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,+offset: U32,
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}) ->
  {Views.View{axes,offset} == Canonical.canonical_view(shape,count) : Views.View}:
  +canonical = Canonical.canonical_view(shape,count)
  +read = {%Canonical.view(shape,count,exact) : {Some{Views.View{axes,offset}} == Tensors.stretched(_,shape) : Maybe<&2,Views.View>}
    Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(shape),shape),Some{Views.View{axes,offset}},stretched) :
    {Some{Views.View{axes,offset}} == Tensors.stretched(Some{canonical},shape) : Maybe<&2,Views.View>}}
  +kept = {%Canonical.canonical_shape(shape,count,exact) : {Views.stretch(canonical,_) == Some{canonical} : Maybe<&2,Views.View>}
    stretch_same(canonical) : {Views.stretch(canonical,shape) == Some{canonical} : Maybe<&2,Views.View>}}
  Equal.cong(Maybe<&2,Views.View>,Views.View,view => view_or(view,canonical),Some{Views.View{axes,offset}},Some{canonical},
    Equal.trans(Maybe<&2,Views.View>,Some{Views.View{axes,offset}},Views.stretch(canonical,shape),Some{canonical},read,kept))

# The view of a shape stretched to the shape reads every index at itself.
def stretched_own_address(+shape: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,+offset: U32,
  +stretched: {Tensors.stretched(Tensors.view_of(shape),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),U32.to_nat(index)) : U32) == index : U32}:
  +canonical = Canonical.canonical_view(shape,count)
  +number = U32.to_nat(index)
  +extents = stretch_extents(Tensors.view_of(shape),shape,axes,offset,stretched)
  +inside_axes = {%Equal.sym(List<&2,U32>,Views.axis_extents(axes),shape,extents) : {Nat.is_lt(number,ShapeLaws.elements(_)) == True{} : Bool} inside :
    {Nat.is_lt(number,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +inside_canonical = {%Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical),shape,Canonical.canonical_shape(shape,count,exact)) :
      {Nat.is_lt(number,ShapeLaws.elements(_)) == True{} : Bool}
    inside : {Nat.is_lt(number,ShapeLaws.elements(Views.axis_extents_of(canonical))) == True{} : Bool}}
  +same = stretched_own(shape,count,exact,axes,offset,stretched)
  +zero = CopyValues.zero_contiguous_address(canonical,number,Canonical.canonical_contiguous(shape,count,exact),inside_canonical)
  +at = {%Equal.sym(Views.View,Views.View{axes,offset},canonical,same) : {CopyValues.address(_,number) == U32.from_nat(number) : U32}
    zero : {CopyValues.address(Views.View{axes,offset},number) == U32.from_nat(number) : U32}}
  Equal.trans(U32,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),number) : U32),
    CopyValues.address(Views.View{axes,offset},number),index,viewed_address(axes,offset,number,inside_axes),
    Equal.trans(U32,CopyValues.address(Views.View{axes,offset},number),U32.from_nat(number),index,at,Arithmetic.native_roundtrip(index)))
