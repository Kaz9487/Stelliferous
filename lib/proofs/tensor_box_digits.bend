# The digits of a cell of a piece: on every axis the digit of the first cell
# of the piece plus the digit of the cell in the box of the piece.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_grid_box.bend as Box
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_target_cells.bend as Targets
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_move_readback.bend as Held
import ../kernel_gemm.bend as KernelGemm
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./u32_extrema.bend as Extrema
import ./tensor_tiled_public.bend as TiledPublic
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits

# ---- Digits ----

def zero_flag(+value: Nat) -> Bool:
  match value:
    case 0n: True{}
    case 1n+previous: False{}

def before(+value: Nat) -> Nat:
  match value:
    case 0n: 0n
    case 1n+previous: previous

def follow_strict(+value: Nat,+limit: Nat,+follow: {Nat.is_le(1n+value,limit) == True{} : Bool}) -> {Nat.is_lt(value,limit) == True{} : Bool}:
  match value limit:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(follow))
    case 0n 1n+ +rest: {==}
    case 1n+ +previous 0n: Empty.absurd({Nat.is_lt(1n+previous,0n) == True{} : Bool},WordEncoding.false_true(follow))
    case 1n+ +previous 1n+ +rest: follow_strict(previous,rest,follow)

def zero_successor(+value: Nat,+same: {0n == 1n+value : Nat}) -> Empty:
  WordEncoding.false_true(Equal.sym(Bool,True{},False{},Equal.cong(Nat,Bool,number => zero_flag(number),0n,1n+value,same)))

def digit_at(digits: List<&2,Nat>,axis: Nat) -> Nat:
  match digits axis:
    case Nil{} axis: 0n
    case Con{+digit,rest} 0n: digit
    case Con{digit,rest} 1n+next: digit_at(rest,next)

# The digits of two lists added axis by axis.
def added(left: List<&2,Nat>,right: List<&2,Nat>) -> List<&2,Nat>:
  match left right:
    case Con{+digit,left} Con{+other,right}: Nat.add(digit,other) <> added(left,right)
    case left right: []

# The numbers of words.
def words_nat(words: List<&2,U32>) -> List<&2,Nat>:
  match words:
    case Nil{}: []
    case Con{+word,rest}: U32.to_nat(word) <> words_nat(rest)

def coordinates_length(extents: List<&2,U32>,+index: Nat) -> {List.length(&2,Nat,Targets.cell_coordinates(extents,index)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Targets.cell_coordinates(rest,Nat.mod(index,ShapeLaws.elements(rest)))),List.length(&2,U32,rest),
        coordinates_length(rest,Nat.mod(index,ShapeLaws.elements(rest))))

def added_length(left: List<&2,Nat>,right: List<&2,Nat>,+same: {List.length(&2,Nat,left) == List.length(&2,Nat,right) : Nat}) ->
  {List.length(&2,Nat,added(left,right)) == List.length(&2,Nat,left) : Nat}:
  match left right:
    case Con{+digit,+more} Con{+other,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,added(more,rest)),List.length(&2,Nat,more),
        added_length(more,rest,Equal.cong(Nat,Nat,count => before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Nat,rest),same)))
    case Nil{} +right: {==}
    case Con{+digit,+more} Nil{}: Empty.absurd({0n == 1n+List.length(&2,Nat,more) : Nat},zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))

def added_digit(left: List<&2,Nat>,right: List<&2,Nat>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,Nat,left)) == True{} : Bool},
  +same: {List.length(&2,Nat,left) == List.length(&2,Nat,right) : Nat}) ->
  {digit_at(added(left,right),axis) == Nat.add(digit_at(left,axis),digit_at(right,axis)) : Nat}:
  match left right axis:
    case Con{+digit,+more} Con{+other,+rest} 0n: {==}
    case Con{+digit,+more} Con{+other,+rest} 1n+ +next:
      added_digit(more,rest,next,inside,Equal.cong(Nat,Nat,count => before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Nat,rest),same))
    case Nil{} +right +axis: Empty.absurd({digit_at(added(Nil{},right),axis) == Nat.add(digit_at(Nil{},axis),digit_at(right,axis)) : Nat},Order.below_zero(axis,inside))
    case Con{+digit,+more} Nil{} +axis:
      Empty.absurd({digit_at(added(Con{digit,more},Nil{}),axis) == Nat.add(digit_at(Con{digit,more},axis),digit_at(Nil{},axis)) : Nat},
        zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))

# The digits of two lists agree on count axes from axis on.
def Agree(count: Nat,+left: List<&2,Nat>,+right: List<&2,Nat>,+axis: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: KernelGemm.Both<{digit_at(left,axis) == digit_at(right,axis) : Nat},Agree(rest,left,right,1n+axis)>

def agree_shift(count: Nat,+digit: Nat,+left: List<&2,Nat>,+other: Nat,+right: List<&2,Nat>,+axis: Nat,+agree: Agree(count,Con{digit,left},Con{other,right},1n+axis)) ->
  Agree(count,left,right,axis):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{+here,+later} = agree
      KernelGemm.Both{here,agree_shift(rest,digit,left,other,right,1n+axis,later)}

# Lists of one length with the same digit on every axis are the same.
def same_digits(left: List<&2,Nat>,right: List<&2,Nat>,+same: {List.length(&2,Nat,left) == List.length(&2,Nat,right) : Nat},
  +agree: Agree(List.length(&2,Nat,left),left,right,0n)) -> {left == right : List<&2,Nat>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Con{+digit,+more} Con{+other,+rest}:
      KernelGemm.Both{+head,+later} = agree
      +tail = same_digits(more,rest,Equal.cong(Nat,Nat,count => before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Nat,rest),same),
        agree_shift(List.length(&2,Nat,more),digit,more,other,rest,0n,later))
      %head : {Con{digit,more} == Con{_,rest} : List<&2,Nat>}
      %tail : {Con{digit,more} == Con{digit,_} : List<&2,Nat>}
      {==}
    case Nil{} Con{+other,+rest}: Empty.absurd({Nil{} == Con{other,rest} : List<&2,Nat>},zero_successor(List.length(&2,Nat,rest),same))
    case Con{+digit,+more} Nil{}:
      Empty.absurd({Con{digit,more} == Nil{} : List<&2,Nat>},zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))

# ---- Digits times steps ----

# The cells of a shape, as a word.
def product_from(extents: List<&2,U32>) -> {U32.from_nat(ShapeLaws.elements(extents)) == Grid.product(extents) : U32}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      %Multiplication.native_encoded_multiply(U32.to_nat(extent),ShapeLaws.elements(rest)) : {_ == (extent * Grid.product(rest) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(extent)),extent,Arithmetic.native_roundtrip(extent)) :
        {(_ * U32.from_nat(ShapeLaws.elements(rest)) : U32) == (extent * Grid.product(rest) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(ShapeLaws.elements(rest)),Grid.product(rest),product_from(rest)) : {(extent * _ : U32) == (extent * Grid.product(rest) : U32) : U32}
      {==}


def stepped_digits(digits: List<&2,Nat>,steps: List<&2,U32>) -> U32:
  match digits steps:
    case Con{+digit,digits} Con{+step,steps}: (U32.from_nat(digit) * step + stepped_digits(digits,steps) : U32)
    case digits steps: 0

# The number of digits is every digit times the stride of its axis.
def digits_stepped(extents: List<&2,U32>,digits: List<&2,Nat>) ->
  {U32.from_nat(ShapeLaws.digits_value(extents,digits)) == stepped_digits(digits,Tasks.strides_of(extents)) : U32}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      +inner = ShapeLaws.elements(rest)
      +value = ShapeLaws.digits_value(rest,more)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {U32.from_nat(ShapeLaws.digits_value(Con{extent,rest},Con{digit,more})) == stepped_digits(Con{digit,more},_) : U32}
      %WordArithmetic.native_add(Nat.mul(digit,inner),value) :
        {_ == ((U32.from_nat(digit) * Grid.product(rest) : U32) + stepped_digits(more,Tasks.strides_of(rest)) : U32) : U32}
      %Multiplication.native_encoded_multiply(digit,inner) :
        {(_ + U32.from_nat(value) : U32) == ((U32.from_nat(digit) * Grid.product(rest) : U32) + stepped_digits(more,Tasks.strides_of(rest)) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(inner),Grid.product(rest),product_from(rest)) :
        {((U32.from_nat(digit) * _ : U32) + U32.from_nat(value) : U32) == ((U32.from_nat(digit) * Grid.product(rest) : U32) + stepped_digits(more,Tasks.strides_of(rest)) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(value),stepped_digits(more,Tasks.strides_of(rest)),digits_stepped(rest,more)) :
        {((U32.from_nat(digit) * Grid.product(rest) : U32) + _ : U32) == ((U32.from_nat(digit) * Grid.product(rest) : U32) + stepped_digits(more,Tasks.strides_of(rest)) : U32) : U32}
      {==}
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: {==}
    case Con{+extent,+rest} Nil{}: {==}

# The steps of a piece over a cell are its steps over the coordinates of the
# cell.
def stepped_coordinates(extents: List<&2,U32>,steps: List<&2,U32>,+index: Nat) ->
  {Box.stepped_sum(extents,steps,index) == stepped_digits(Targets.cell_coordinates(extents,index),steps) : U32}:
  match extents steps:
    case Con{+extent,+rest} Con{+step,+later}:
      +inner = ShapeLaws.elements(rest)
      Equal.cong(U32,U32,tail => (U32.from_nat(Nat.div(index,inner)) * step + tail : U32),Box.stepped_sum(rest,later,Nat.mod(index,inner)),
        stepped_digits(Targets.cell_coordinates(rest,Nat.mod(index,inner)),later),stepped_coordinates(rest,later,Nat.mod(index,inner)))
    case Nil{} +steps: {==}
    case Con{+extent,+rest} Nil{}: {==}

# One step at an axis, none elsewhere.
def zero_steps(rank: Nat) -> List<&2,U32>:
  match rank:
    case 0n: []
    case 1n+rest: 0 <> zero_steps(rest)

def unit_steps(rank: Nat,axis: Nat) -> List<&2,U32>:
  match rank axis:
    case 0n axis: []
    case 1n+rest 0n: 1 <> zero_steps(rest)
    case 1n+rest 1n+next: 0 <> unit_steps(rest,next)

def zero_stepped(digits: List<&2,Nat>,rank: Nat) -> {stepped_digits(digits,zero_steps(rank)) == 0 : U32}:
  match digits rank:
    case Con{+digit,+more} 1n+ +rest:
      %Equal.sym(U32,stepped_digits(more,zero_steps(rest)),0,zero_stepped(more,rest)) : {((U32.from_nat(digit) * 0 : U32) + _ : U32) == 0 : U32}
      %Equal.sym(U32,(U32.from_nat(digit) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(digit))) : {(_ + 0 : U32) == 0 : U32}
      Arithmetic.add_zero(0)
    case Nil{} +rank: {==}
    case Con{+digit,+more} 0n: {==}

# The steps of one axis read its digit.
def unit_stepped(digits: List<&2,Nat>,rank: Nat,axis: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool}) ->
  {stepped_digits(digits,unit_steps(rank,axis)) == U32.from_nat(digit_at(digits,axis)) : U32}:
  match digits rank axis:
    case Con{+digit,+more} 1n+ +rest 0n:
      %Equal.sym(U32,stepped_digits(more,zero_steps(rest)),0,zero_stepped(more,rest)) : {((U32.from_nat(digit) * 1 : U32) + _ : U32) == U32.from_nat(digit) : U32}
      %Equal.sym(U32,(U32.from_nat(digit) * 1 : U32),U32.from_nat(digit),Arithmetic.multiply_one(U32.from_nat(digit))) : {(_ + 0 : U32) == U32.from_nat(digit) : U32}
      Arithmetic.add_zero(U32.from_nat(digit))
    case Con{+digit,+more} 1n+ +rest 1n+ +next:
      %Equal.sym(U32,(U32.from_nat(digit) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(digit))) :
        {(_ + stepped_digits(more,unit_steps(rest,next)) : U32) == U32.from_nat(digit_at(more,next)) : U32}
      %Equal.sym(U32,(0 + stepped_digits(more,unit_steps(rest,next)) : U32),stepped_digits(more,unit_steps(rest,next)),Arithmetic.add_zero_left(stepped_digits(more,unit_steps(rest,next)))) :
        {_ == U32.from_nat(digit_at(more,next)) : U32}
      unit_stepped(more,rest,next,inside)
    case Con{+digit,+more} 0n +axis: Empty.absurd({stepped_digits(Con{digit,more},unit_steps(0n,axis)) == U32.from_nat(digit_at(Con{digit,more},axis)) : U32},Order.below_zero(axis,inside))
    case Nil{} +rank +axis: {==}

# ---- Coordinates ----

def elements_positive(+extent: U32,+rest: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(Con{extent,rest})) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(rest)) == True{} : Bool}:
  DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),Order.positive_bound(index,ShapeLaws.elements(Con{extent,rest}),inside))

# The coordinates of the number of digits below their extents are the
# digits.
def coordinates_value(extents: List<&2,U32>,digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(extents,digits)) ->
  {Targets.cell_coordinates(extents,ShapeLaws.digits_value(extents,digits)) == digits : List<&2,Nat>}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +inner = ShapeLaws.elements(rest)
      +tail = ShapeLaws.digits_value(rest,more)
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(digit,inner),tail),inner),(digit,tail),
          Division.quotient_remainder(digit,tail,inner,Coordinates.digits_positive(rest,more,later),Coordinates.digits_below(rest,more,later))) :
        {Con{Pair.fst(Nat,Nat,_),Targets.cell_coordinates(rest,Pair.snd(Nat,Nat,_))} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Targets.cell_coordinates(rest,tail),more,coordinates_value(rest,more,later)) : {Con{digit,_} == Con{digit,more} : List<&2,Nat>}
      {==}
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({Targets.cell_coordinates(Nil{},ShapeLaws.digits_value(Nil{},Con{digit,more})) == Con{digit,more} : List<&2,Nat>},below)
    case Con{+extent,+rest} Nil{}: Empty.absurd({Targets.cell_coordinates(Con{extent,rest},ShapeLaws.digits_value(Con{extent,rest},Nil{})) == Nil{} : List<&2,Nat>},below)

# The number of the coordinates of a cell is the cell.
def value_coordinates(extents: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(extents,Targets.cell_coordinates(extents,index)) == index : Nat}:
  match extents:
    case Nil{}: Equal.sym(Nat,index,0n,Order.zero_below(index,Division.lt_succ_le(index,0n,inside)))
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +positive = elements_positive(extent,rest,index,inside)
      +remainder = Nat.mod(index,inner)
      %Equal.sym(Nat,ShapeLaws.digits_value(rest,Targets.cell_coordinates(rest,remainder)),remainder,
          value_coordinates(rest,remainder,DivisionOrder.remainder_small(index,inner,positive))) :
        {Nat.add(Nat.mul(Nat.div(index,inner),inner),_) == index : Nat}
      Equal.sym(Nat,index,Nat.add(Nat.mul(Nat.div(index,inner),inner),remainder),DivisionOrder.quotient_exact(index,inner,positive))

# The coordinates of a cell lie below their extents.
def coordinates_below(extents: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Coordinates.DigitsBelow(extents,Targets.cell_coordinates(extents,index)):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +positive = elements_positive(extent,rest,index,inside)
      KernelGemm.Both{DivisionOrder.quotient_below(index,U32.to_nat(extent),inner,positive,inside),
        coordinates_below(rest,Nat.mod(index,inner),DivisionOrder.remainder_small(index,inner,positive))}

# Every coordinate of a cell is at most the cell.
def coordinate_le(extents: List<&2,U32>,+index: Nat,axis: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Nat.is_le(digit_at(Targets.cell_coordinates(extents,index),axis),index) == True{} : Bool}:
  match extents axis:
    case Nil{} +axis: Order.zero_le(index)
    case Con{+extent,+rest} 0n:
      +inner = ShapeLaws.elements(rest)
      +positive = elements_positive(extent,rest,index,inside)
      Order.transitive(Nat.div(index,inner),Nat.mul(Nat.div(index,inner),inner),index,RowMajor.product_covers(Nat.div(index,inner),inner,positive),
        DivisionOrder.product_below(index,inner,positive))
    case Con{+extent,+rest} 1n+ +next:
      +inner = ShapeLaws.elements(rest)
      +positive = elements_positive(extent,rest,index,inside)
      +remainder = Nat.mod(index,inner)
      Order.transitive(digit_at(Targets.cell_coordinates(rest,remainder),next),remainder,index,
        coordinate_le(rest,remainder,next,DivisionOrder.remainder_small(index,inner,positive)),DivisionOrder.remainder_le(index,inner,positive))

# The coordinates of a word are the coordinates of its number.
def coordinates_exact(extents: List<&2,U32>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {words_nat(Grid.coordinates(extents,index)) == Targets.cell_coordinates(extents,U32.to_nat(index)) : List<&2,Nat>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +cells = ShapeLaws.elements(rest)
      +value = U32.to_nat(index)
      +whole = Nat.mul(U32.to_nat(extent),cells)
      +whole_positive = Order.positive_bound(value,whole,inside)
      +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +whole_room = Order.transitive(whole,Nat.double(whole),U32.to_nat(cap),Order.le_double(whole),room)
      +cells_le = DivisionOrder.le_multiple(U32.to_nat(extent),cells,extent_positive)
      +extent_le = RowMajor.product_covers(U32.to_nat(extent),cells,positive)
      +inner_room = Order.transitive(Nat.double(cells),Nat.double(whole),U32.to_nat(cap),Order.double_le(cells,whole,cells_le),room)
      +extent_room = Order.transitive(Nat.double(U32.to_nat(extent)),Nat.double(whole),U32.to_nat(cap),Order.double_le(U32.to_nat(extent),whole,extent_le),room)
      +width = U32.max(Grid.product(rest),1)
      +width_exact = Extrema.max_one_exact(Grid.product(rest),cells,TiledPublic.product_exact(cap,cap_ok,rest,Order.transitive(cells,whole,U32.to_nat(cap),cells_le,whole_room)),positive)
      +quotient = Nat.div(value,cells)
      +remainder = Nat.mod(value,cells)
      +exact = DivisionOrder.quotient_exact(value,cells,positive)
      +remainder_in = DivisionOrder.remainder_small(value,cells,positive)
      +quotient_in = DivisionOrder.quotient_below(value,U32.to_nat(extent),cells,positive,inside)
      +quotient_exact = Radix.quotient_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
      +remainder_exact = Radix.remainder_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
      +extent_width = Extrema.max_one_exact(extent,U32.to_nat(extent),{==},extent_positive)
      +digit_exact = Radix.remainder_word((index / width : U32),U32.max(extent,1),0n,quotient,U32.to_nat(extent),cap,cap_ok,quotient_exact,extent_width,extent_positive,quotient_in,extent_room)
      +later = coordinates_exact(rest,(index % width : U32),cap,cap_ok,inner_room,
        {%Equal.sym(Nat,U32.to_nat((index % width : U32)),remainder,remainder_exact) : {Nat.is_lt(_,cells) == True{} : Bool} remainder_in :
          {Nat.is_lt(U32.to_nat((index % width : U32)),cells) == True{} : Bool}})
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {words_nat(Grid.coordinates_by(Con{extent,rest},_,index)) == Targets.cell_coordinates(Con{extent,rest},U32.to_nat(index)) : List<&2,Nat>}
      %Equal.sym(Nat,U32.to_nat(((index / width : U32) % U32.max(extent,1) : U32)),quotient,digit_exact) :
        {Con{_,words_nat(Grid.coordinates(rest,(index % width : U32)))} == Con{quotient,Targets.cell_coordinates(rest,remainder)} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,words_nat(Grid.coordinates(rest,(index % width : U32))),Targets.cell_coordinates(rest,U32.to_nat((index % width : U32))),later) :
        {Con{quotient,_} == Con{quotient,Targets.cell_coordinates(rest,remainder)} : List<&2,Nat>}
      %Equal.sym(Nat,U32.to_nat((index % width : U32)),remainder,remainder_exact) :
        {Con{quotient,Targets.cell_coordinates(rest,_)} == Con{quotient,Targets.cell_coordinates(rest,remainder)} : List<&2,Nat>}
      {==}

# ---- The digits of a cell of a piece ----

# The digit of the first cell of a piece and the digit of a cell in its box
# add to the digit of the cell, on an axis the steps of the piece count.
def piece_digit(+extents: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+axis: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool},
  +boxed: Box.PieceBox(extents,unit_steps(List.length(&2,U32,extents),axis),boxes,start,cells,index)) ->
  {Nat.add(digit_at(Targets.cell_coordinates(extents,start),axis),digit_at(Targets.cell_coordinates(boxes,index),axis)) ==
    digit_at(Targets.cell_coordinates(extents,Nat.add(start,index)),axis) : Nat}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  +rank = List.length(&2,U32,extents)
  +steps = unit_steps(rank,axis)
  +first = digit_at(Targets.cell_coordinates(extents,start),axis)
  +inner = digit_at(Targets.cell_coordinates(boxes,index),axis)
  +whole = digit_at(Targets.cell_coordinates(extents,Nat.add(start,index)),axis)
  +start_in = Order.le_lt_trans(start,Nat.add(start,index),ShapeLaws.elements(extents),Order.le_add(start,index),ending)
  +first_le = coordinate_le(extents,start,axis,start_in)
  +inner_le = coordinate_le(boxes,index,axis,index_box)
  +sum_le = Order.transitive(Nat.add(first,inner),Nat.add(start,inner),Nat.add(start,index),Order.add_le(first,start,inner,first_le),
    %Order.commutative(inner,start) : {Nat.is_le(_,Nat.add(start,index)) == True{} : Bool}
    %Order.commutative(index,start) : {Nat.is_le(Nat.add(inner,start),_) == True{} : Bool}
    Order.add_le(inner,index,start,inner_le))
  +total_room = Order.transitive(Nat.add(start,index),ShapeLaws.elements(extents),U32.to_nat(cap),Bounds.strict_le(Nat.add(start,index),ShapeLaws.elements(extents),ending),room)
  +box_digits = stepped_digits(Targets.cell_coordinates(boxes,index),steps)
  +start_digits = stepped_digits(Targets.cell_coordinates(extents,start),steps)
  +whole_digits = stepped_digits(Targets.cell_coordinates(extents,Nat.add(start,index)),steps)
  +box_sum = Box.stepped_sum(boxes,steps,index)
  +start_sum = Box.stepped_sum(extents,steps,start)
  +words = {%stepped_coordinates(boxes,steps,index) : {(_ + start_digits : U32) == whole_digits : U32}
    %stepped_coordinates(extents,steps,start) : {(box_sum + _ : U32) == whole_digits : U32}
    %stepped_coordinates(extents,steps,Nat.add(start,index)) : {(box_sum + start_sum : U32) == _ : U32}
    shift : {(box_digits + start_digits : U32) == whole_digits : U32}}
  +digits = {%unit_stepped(Targets.cell_coordinates(boxes,index),rank,axis,inside) : {(_ + U32.from_nat(first) : U32) == U32.from_nat(whole) : U32}
    %unit_stepped(Targets.cell_coordinates(extents,start),rank,axis,inside) : {(box_digits + _ : U32) == U32.from_nat(whole) : U32}
    %unit_stepped(Targets.cell_coordinates(extents,Nat.add(start,index)),rank,axis,inside) : {(box_digits + start_digits : U32) == _ : U32}
    words : {(U32.from_nat(inner) + U32.from_nat(first) : U32) == U32.from_nat(whole) : U32}}
  +joined = {%WordArithmetic.native_add(inner,first) : {_ == U32.from_nat(whole) : U32} digits : {U32.from_nat(Nat.add(inner,first)) == U32.from_nat(whole) : U32}}
  +ordered = Equal.trans(Nat,Nat.add(first,inner),Nat.add(inner,first),whole,Order.commutative(first,inner),
    Equal.trans(Nat,Nat.add(inner,first),U32.to_nat(U32.from_nat(Nat.add(inner,first))),whole,
      Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.add(inner,first))),Nat.add(inner,first),
        Bounds.roundtrip(Nat.add(inner,first),cap,cap_ok,
          %Order.commutative(first,inner) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} Order.transitive(Nat.add(first,inner),Nat.add(start,index),U32.to_nat(cap),sum_le,total_room))),
      Equal.trans(Nat,U32.to_nat(U32.from_nat(Nat.add(inner,first))),U32.to_nat(U32.from_nat(whole)),whole,
        Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(Nat.add(inner,first)),U32.from_nat(whole),joined),
        Bounds.roundtrip(whole,cap,cap_ok,Order.transitive(whole,Nat.add(start,index),U32.to_nat(cap),
          coordinate_le(extents,Nat.add(start,index),axis,ending),total_room)))))
  ordered

# A piece over the steps of every axis from axis on, count of them.
def Boxes(count: Nat,+extents: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+axis: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<Box.PieceBox(extents,unit_steps(List.length(&2,U32,extents),axis),boxes,start,cells,index),Boxes(rest,extents,boxes,start,cells,index,1n+axis)>

def boxes_agree(count: Nat,+extents: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+axis: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},
  +inside: {Nat.is_le(Nat.add(axis,count),List.length(&2,U32,extents)) == True{} : Bool},+boxed: Boxes(count,extents,boxes,start,cells,index,axis)) ->
  Agree(count,added(Targets.cell_coordinates(extents,start),Targets.cell_coordinates(boxes,index)),Targets.cell_coordinates(extents,Nat.add(start,index)),axis):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{+here,+later} = boxed
      +rank = List.length(&2,U32,extents)
      +first = Targets.cell_coordinates(extents,start)
      +inner = Targets.cell_coordinates(boxes,index)
      +within = Order.le_lt_trans(axis,Nat.add(axis,rest),rank,Order.le_add(axis,rest),follow_strict(Nat.add(axis,rest),rank,
        {%Addition.successor_right(axis,rest) : {Nat.is_le(_,rank) == True{} : Bool} inside : {Nat.is_le(1n+Nat.add(axis,rest),rank) == True{} : Bool}}))
      KernelGemm.Both{
        Equal.trans(Nat,digit_at(added(first,inner),axis),Nat.add(digit_at(first,axis),digit_at(inner,axis)),
          digit_at(Targets.cell_coordinates(extents,Nat.add(start,index)),axis),
          added_digit(first,inner,axis,{%Equal.sym(Nat,List.length(&2,Nat,first),rank,coordinates_length(extents,start)) : {Nat.is_lt(axis,_) == True{} : Bool} within :
            {Nat.is_lt(axis,List.length(&2,Nat,first)) == True{} : Bool}},
            Equal.trans(Nat,List.length(&2,Nat,first),rank,List.length(&2,Nat,inner),coordinates_length(extents,start),
              Equal.sym(Nat,List.length(&2,Nat,inner),rank,Equal.trans(Nat,List.length(&2,Nat,inner),List.length(&2,U32,boxes),rank,coordinates_length(boxes,index),same)))),
          piece_digit(extents,boxes,start,cells,index,axis,cap,cap_ok,room,within,here)),
        boxes_agree(rest,extents,boxes,start,cells,index,1n+axis,cap,cap_ok,room,same,
          {%Addition.successor_right(axis,rest) : {Nat.is_le(_,rank) == True{} : Bool} inside : {Nat.is_le(Nat.add(1n+axis,rest),rank) == True{} : Bool}},later)}

# The coordinates of the first cell of a piece and of a cell in its box add
# to the coordinates of the cell.
def piece_coordinates(+extents: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},+boxed: Boxes(List.length(&2,U32,extents),extents,boxes,start,cells,index,0n)) ->
  {added(Targets.cell_coordinates(extents,start),Targets.cell_coordinates(boxes,index)) == Targets.cell_coordinates(extents,Nat.add(start,index)) : List<&2,Nat>}:
  +rank = List.length(&2,U32,extents)
  +first = Targets.cell_coordinates(extents,start)
  +inner = Targets.cell_coordinates(boxes,index)
  +first_length = coordinates_length(extents,start)
  +inner_length = Equal.trans(Nat,List.length(&2,Nat,inner),List.length(&2,U32,boxes),rank,coordinates_length(boxes,index),same)
  +lengths = Equal.trans(Nat,List.length(&2,Nat,first),rank,List.length(&2,Nat,inner),first_length,Equal.sym(Nat,List.length(&2,Nat,inner),rank,inner_length))
  +added_rank = Equal.trans(Nat,List.length(&2,Nat,added(first,inner)),List.length(&2,Nat,first),rank,added_length(first,inner,lengths),first_length)
  same_digits(added(first,inner),Targets.cell_coordinates(extents,Nat.add(start,index)),
    Equal.trans(Nat,List.length(&2,Nat,added(first,inner)),rank,List.length(&2,Nat,Targets.cell_coordinates(extents,Nat.add(start,index))),added_rank,
      Equal.sym(Nat,List.length(&2,Nat,Targets.cell_coordinates(extents,Nat.add(start,index))),rank,coordinates_length(extents,Nat.add(start,index)))),
    {%Equal.sym(Nat,List.length(&2,Nat,added(first,inner)),rank,added_rank) :
        Agree(_,added(first,inner),Targets.cell_coordinates(extents,Nat.add(start,index)),0n)
      boxes_agree(rank,extents,boxes,start,cells,index,0n,cap,cap_ok,room,same,Order.reflexive(rank),boxed) :
      Agree(List.length(&2,Nat,added(first,inner)),added(first,inner),Targets.cell_coordinates(extents,Nat.add(start,index)),0n)})

# ---- A piece box lies inside its cells ----

# The last index of every axis.
def lasts(extents: List<&2,U32>) -> List<&2,Nat>:
  match extents:
    case Nil{}: []
    case Con{+extent,rest}: Nat.sub(U32.to_nat(extent),1n) <> lasts(rest)

def last_split(+extent: Nat,+inner: Nat,+extent_positive: {Nat.is_lt(0n,extent) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.divmod(Nat.sub(Nat.mul(extent,inner),1n),inner) == (Nat.sub(extent,1n),Nat.sub(inner,1n)) : Nat & Nat}:
  match extent inner:
    case 1n+ +before 1n+ +inner_before:
      +whole = Nat.add(inner_before,Nat.mul(before,1n+inner_before))
      %Equal.sym(Nat,Nat.sub(whole,0n),whole,Order.subtract_zero(whole)) :
        {Nat.divmod(_,1n+inner_before) == (Nat.sub(before,0n),Nat.sub(inner_before,0n)) : Nat & Nat}
      %Equal.sym(Nat,Nat.sub(before,0n),before,Order.subtract_zero(before)) : {Nat.divmod(whole,1n+inner_before) == (_,Nat.sub(inner_before,0n)) : Nat & Nat}
      %Equal.sym(Nat,Nat.sub(inner_before,0n),inner_before,Order.subtract_zero(inner_before)) : {Nat.divmod(whole,1n+inner_before) == (before,_) : Nat & Nat}
      Division.unique(Nat.add(inner_before,Nat.mul(before,1n+inner_before)),1n+inner_before,before,inner_before,
        Order.commutative(inner_before,Nat.mul(before,1n+inner_before)),inner_positive,Order.lt_succ(inner_before,inner_before,Order.reflexive(inner_before)))
    case 0n +inner: Empty.absurd({Nat.divmod(Nat.sub(Nat.mul(0n,inner),1n),inner) == (Nat.sub(0n,1n),Nat.sub(inner,1n)) : Nat & Nat},WordEncoding.false_true(extent_positive))
    case 1n+ +before 0n:
      Empty.absurd({Nat.divmod(Nat.sub(Nat.mul(1n+before,0n),1n),0n) == (Nat.sub(1n+before,1n),Nat.sub(0n,1n)) : Nat & Nat},WordEncoding.false_true(inner_positive))

# The coordinates of the last cell are the last index of every axis.
def last_coordinates(extents: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Targets.cell_coordinates(extents,Nat.sub(ShapeLaws.elements(extents),1n)) == lasts(extents) : List<&2,Nat>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.sub(Nat.mul(U32.to_nat(extent),inner),1n),inner),(Nat.sub(U32.to_nat(extent),1n),Nat.sub(inner,1n)),
          last_split(U32.to_nat(extent),inner,extent_positive,inner_positive)) :
        {Con{Pair.fst(Nat,Nat,_),Targets.cell_coordinates(rest,Pair.snd(Nat,Nat,_))} == Con{Nat.sub(U32.to_nat(extent),1n),lasts(rest)} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Targets.cell_coordinates(rest,Nat.sub(inner,1n)),lasts(rest),last_coordinates(rest,inner_positive)) :
        {Con{Nat.sub(U32.to_nat(extent),1n),_} == Con{Nat.sub(U32.to_nat(extent),1n),lasts(rest)} : List<&2,Nat>}
      {==}

# Every axis of a box from its first index lies inside its extent.
def BoxInside(lows: List<&2,Nat>,boxes: List<&2,U32>,extents: List<&2,U32>) -> Data:
  match lows boxes extents:
    case Con{+low,lows} Con{+box,boxes} Con{+extent,extents}:
      KernelGemm.Both<{Nat.is_le(Nat.add(low,U32.to_nat(box)),U32.to_nat(extent)) == True{} : Bool},BoxInside(lows,boxes,extents)>
    case lows boxes extents: Unit

# Every extent of cells is positive.
def Positive(extents: List<&2,U32>) -> Data:
  match extents:
    case Nil{}: Unit
    case Con{+extent,extents}: KernelGemm.Both<{Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},Positive(extents)>

def positive_extents(extents: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> Positive(extents):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      KernelGemm.Both{RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive),
        positive_extents(rest,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))}

def inside_axis(+low: Nat,+box: Nat,+extent: Nat,+positive: {Nat.is_lt(0n,box) == True{} : Bool},+below: {Nat.is_lt(Nat.add(low,Nat.sub(box,1n)),extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(low,box),extent) == True{} : Bool}:
  match box:
    case 0n: Empty.absurd({Nat.is_le(Nat.add(low,0n),extent) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +before:
      %Equal.sym(Nat,Nat.add(low,1n+before),1n+Nat.add(low,before),Addition.successor_right(low,before)) : {Nat.is_le(_,extent) == True{} : Bool}
      Order.strict_follow(Nat.add(low,before),extent,
        {%Order.subtract_zero(before) : {Nat.is_lt(Nat.add(low,_),extent) == True{} : Bool} below : {Nat.is_lt(Nat.add(low,before),extent) == True{} : Bool}})

def inside_of(lows: List<&2,Nat>,boxes: List<&2,U32>,extents: List<&2,U32>,+positive: Positive(boxes),+below: Coordinates.DigitsBelow(extents,added(lows,lasts(boxes)))) ->
  BoxInside(lows,boxes,extents):
  match lows boxes extents:
    case Con{+low,+more} Con{+box,+rest} Con{+extent,+later}:
      KernelGemm.Both{+box_positive,+rest_positive} = positive
      KernelGemm.Both{+here,+after} = below
      KernelGemm.Both{inside_axis(low,U32.to_nat(box),U32.to_nat(extent),box_positive,here),inside_of(more,rest,later,rest_positive,after)}
    case Nil{} +boxes +extents: Unit{}
    case Con{+low,+more} Nil{} +extents: Unit{}
    case Con{+low,+more} Con{+box,+rest} Nil{}: Unit{}

# The cell of a piece lies in its box, and the piece in its cells.
def box_index(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+boxed: Box.PieceBox(extents,steps,boxes,start,cells,index)) ->
  {Nat.is_lt(index,ShapeLaws.elements(boxes)) == True{} : Bool}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  index_box

def box_ending(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,+boxed: Box.PieceBox(extents,steps,boxes,start,cells,index)) ->
  {Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  ending

# The box of a piece from its first cell lies inside the cells.
def piece_inside(extents: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},
  +boxed: Boxes(List.length(&2,U32,extents),extents,boxes,start,cells,Nat.sub(ShapeLaws.elements(boxes),1n),0n)) ->
  BoxInside(Targets.cell_coordinates(extents,start),boxes,extents):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      KernelGemm.Both{+first,+others} = boxed
      +whole = {Con{extent,rest} : List<&2,U32>}
      +last = Nat.sub(ShapeLaws.elements(boxes),1n)
      +ending = box_ending(whole,unit_steps(List.length(&2,U32,whole),0n),boxes,start,cells,last,first)
      +positive = Order.positive_bound(last,ShapeLaws.elements(boxes),box_index(whole,unit_steps(List.length(&2,U32,whole),0n),boxes,start,cells,last,first))
      +coordinates = Equal.trans(List<&2,Nat>,added(Targets.cell_coordinates(whole,start),lasts(boxes)),
        added(Targets.cell_coordinates(whole,start),Targets.cell_coordinates(boxes,last)),Targets.cell_coordinates(whole,Nat.add(start,last)),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,inner => added(Targets.cell_coordinates(whole,start),inner),lasts(boxes),Targets.cell_coordinates(boxes,last),
          Equal.sym(List<&2,Nat>,Targets.cell_coordinates(boxes,last),lasts(boxes),last_coordinates(boxes,positive))),
        piece_coordinates(whole,boxes,start,cells,last,cap,cap_ok,room,same,KernelGemm.Both{first,others}))
      inside_of(Targets.cell_coordinates(whole,start),boxes,whole,positive_extents(boxes,positive),
        {%Equal.sym(List<&2,Nat>,added(Targets.cell_coordinates(whole,start),lasts(boxes)),Targets.cell_coordinates(whole,Nat.add(start,last)),coordinates) :
            Coordinates.DigitsBelow(whole,_)
          coordinates_below(whole,Nat.add(start,last),ending) : Coordinates.DigitsBelow(whole,added(Targets.cell_coordinates(whole,start),lasts(boxes)))})

# The cell a piece holds at the number of digits inside its box: its number is
# the first cell plus that number, its digits are the first coordinates plus
# the box digits, and they lie inside the cells.
def CellFacts(+extents: List<&2,U32>,+first: U32,+address: U32,+digits: List<&2,Nat>) -> Data:
  Held.Both<{U32.to_nat((first + address : U32)) == ShapeLaws.digits_value(extents,digits) : Nat},Coordinates.DigitsBelow(extents,digits)>

# A piece box covers no more cells than the cells.
def piece_fits(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,
  +boxed: Box.PieceBox(extents,steps,boxes,start,cells,index)) -> {Nat.is_le(ShapeLaws.elements(boxes),ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+shift,Held.Both{+inside,Held.Both{+fits,Held.Both{+ending,+counted}}}} = boxed
  fits

# The cell a piece box covers at an index lies inside the cells.
def piece_ending(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,
  +boxed: Box.PieceBox(extents,steps,boxes,start,cells,index)) -> {Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+shift,Held.Both{+inside,Held.Both{+fits,Held.Both{+ending,+counted}}}} = boxed
  ending

# The digits of the cell a piece box covers at an index are the coordinates
# of the first cell of the piece plus the index.
def piece_coordinates_of(+extents: List<&2,U32>,+boxes: List<&2,U32>,+first: U32,+cells: Nat,+digits: List<&2,Nat>,+box: List<&2,Nat>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},
  +shifted: {digits == added(words_nat(Grid.coordinates(extents,first)),box) : List<&2,Nat>},+inside: Coordinates.DigitsBelow(boxes,box),
  +ending: {Nat.is_lt(Nat.add(U32.to_nat(first),ShapeLaws.digits_value(boxes,box)),ShapeLaws.elements(extents)) == True{} : Bool},
  +boxed: Boxes(List.length(&2,U32,extents),extents,boxes,U32.to_nat(first),cells,ShapeLaws.digits_value(boxes,box),0n)) ->
  {digits == Targets.cell_coordinates(extents,Nat.add(U32.to_nat(first),ShapeLaws.digits_value(boxes,box))) : List<&2,Nat>}:
  +start = U32.to_nat(first)
  +index = ShapeLaws.digits_value(boxes,box)
  +cell = Nat.add(start,index)
  +elements = ShapeLaws.elements(extents)
  +single = Order.transitive(elements,Nat.double(elements),U32.to_nat(cap),Order.le_double(elements),room)
  +lows = coordinates_exact(extents,first,cap,cap_ok,room,Order.le_lt_trans(start,cell,elements,Order.le_add(start,index),ending))
  +corner = words_nat(Grid.coordinates(extents,first))
  Equal.trans(List<&2,Nat>,digits,added(corner,box),Targets.cell_coordinates(extents,cell),shifted,
    Equal.trans(List<&2,Nat>,added(corner,box),added(Targets.cell_coordinates(extents,start),box),Targets.cell_coordinates(extents,cell),
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => added(list,box),corner,Targets.cell_coordinates(extents,start),lows),
      Equal.trans(List<&2,Nat>,added(Targets.cell_coordinates(extents,start),box),added(Targets.cell_coordinates(extents,start),Targets.cell_coordinates(boxes,index)),
        Targets.cell_coordinates(extents,cell),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,list => added(Targets.cell_coordinates(extents,start),list),box,Targets.cell_coordinates(boxes,index),
          Equal.sym(List<&2,Nat>,Targets.cell_coordinates(boxes,index),box,coordinates_value(boxes,box,inside))),
        piece_coordinates(extents,boxes,start,cells,index,cap,cap_ok,single,same,boxed))))

def piece_cell(+extents: List<&2,U32>,+boxes: List<&2,U32>,+first: U32,+cells: Nat,+digits: List<&2,Nat>,+box: List<&2,Nat>,+address: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},+at: {address == U32.from_nat(ShapeLaws.digits_value(boxes,box)) : U32},
  +shifted: {digits == added(words_nat(Grid.coordinates(extents,first)),box) : List<&2,Nat>},+inside: Coordinates.DigitsBelow(boxes,box),
  +ending: {Nat.is_lt(Nat.add(U32.to_nat(first),ShapeLaws.digits_value(boxes,box)),ShapeLaws.elements(extents)) == True{} : Bool},
  +boxed: Boxes(List.length(&2,U32,extents),extents,boxes,U32.to_nat(first),cells,ShapeLaws.digits_value(boxes,box),0n)) -> CellFacts(extents,first,address,digits):
  +start = U32.to_nat(first)
  +index = ShapeLaws.digits_value(boxes,box)
  +cell = Nat.add(start,index)
  +elements = ShapeLaws.elements(extents)
  +single = Order.transitive(elements,Nat.double(elements),U32.to_nat(cap),Order.le_double(elements),room)
  +cell_room = Order.transitive(cell,elements,U32.to_nat(cap),Bounds.strict_le(cell,elements,ending),single)
  +index_room = Order.transitive(index,cell,U32.to_nat(cap),
    {%Order.commutative(index,start) : {Nat.is_le(index,_) == True{} : Bool} Order.le_add(index,start) : {Nat.is_le(index,cell) == True{} : Bool}},cell_room)
  +coordinates = piece_coordinates_of(extents,boxes,first,cells,digits,box,cap,cap_ok,room,same,shifted,inside,ending,boxed)
  +number = Equal.trans(Nat,U32.to_nat((first + address : U32)),U32.to_nat((first + U32.from_nat(index) : U32)),cell,
    Equal.cong(U32,Nat,word => U32.to_nat((first + word : U32)),address,U32.from_nat(index),at),
    Arithmetic.sum_exact(cap,cap_ok,first,U32.from_nat(index),start,index,{==},Bounds.roundtrip(index,cap,cap_ok,index_room),cell_room))
  +value = Equal.trans(Nat,ShapeLaws.digits_value(extents,digits),ShapeLaws.digits_value(extents,Targets.cell_coordinates(extents,cell)),cell,
    Equal.cong(List<&2,Nat>,Nat,list => ShapeLaws.digits_value(extents,list),digits,Targets.cell_coordinates(extents,cell),coordinates),
    value_coordinates(extents,cell,ending))
  Held.Both{Equal.trans(Nat,U32.to_nat((first + address : U32)),cell,ShapeLaws.digits_value(extents,digits),number,Equal.sym(Nat,ShapeLaws.digits_value(extents,digits),cell,value)),
    {%Equal.sym(List<&2,Nat>,digits,Targets.cell_coordinates(extents,cell),coordinates) : Coordinates.DigitsBelow(extents,_)
      coordinates_below(extents,cell,ending) : Coordinates.DigitsBelow(extents,digits)}}

# A piece of the grid of cells inside the machine is its box at every unit
# step from an axis on.
def grid_boxes(count: Nat,+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool},+axis: Nat) ->
  Boxes(count,extents,Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece)),U32.to_nat(Grid.start(Grid.grid(extents,target),piece)),
    U32.to_nat(Grid.span(Grid.grid(extents,target),piece)),index,axis):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{Box.grid_piece_box(limits,extents,unit_steps(List.length(&2,U32,extents),axis),target,room,nonempty,rank_room,piece,inside,index,index_in),
        grid_boxes(rest,limits,extents,target,room,nonempty,rank_room,piece,inside,index,index_in,1n+axis)}

# The box of a piece has an extent on every axis of the cells, and the
# coordinates of a cell a digit on every axis.
def box_length(extents: List<&2,U32>,+axis: U32,+prefix: U32,+units: U32) -> {List.length(&2,U32,Grid.box(extents,axis,prefix,units)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Grid.box(rest,(axis + 1 : U32),prefix,units)),List.length(&2,U32,rest),box_length(rest,(axis + 1 : U32),prefix,units))

def box_of_length(+grid: Grid.Grid,+cells: List<&2,U32>,+count: U32) -> {List.length(&2,U32,Grid.box_of(grid,cells,count)) == List.length(&2,U32,cells) : Nat}:
  match grid:
    case Grid.Grid{+total,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces}: box_length(cells,0,prefix,(count / U32.max(unit_cells,1) : U32))

def coordinates_count(extents: List<&2,U32>,+index: U32) -> {List.length(&2,U32,Grid.coordinates(extents,index)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +inner = (index % U32.max(Grid.product(rest),1) : U32)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {List.length(&2,U32,Grid.coordinates_by(Con{extent,rest},_,index)) == List.length(&2,U32,Con{extent,rest}) : Nat}
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Grid.coordinates(rest,inner)),List.length(&2,U32,rest),coordinates_count(rest,inner))

# Every axis of a box from its first index fits half the capacity.
def Rooms(lows: List<&2,Nat>,boxes: List<&2,U32>,+cap: U32) -> Data:
  match lows boxes:
    case Con{+low,lows} Con{+box,boxes}:
      KernelGemm.Both<{Nat.is_le(Nat.double(Nat.add(low,U32.to_nat(box))),U32.to_nat(cap)) == True{} : Bool},Rooms(lows,boxes,cap)>
    case lows boxes: Unit

def inside_rooms(lows: List<&2,Nat>,boxes: List<&2,U32>,extents: List<&2,U32>,+cap: U32,
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},+same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},
  +inside: BoxInside(lows,boxes,extents)) -> Rooms(lows,boxes,cap):
  match lows boxes extents:
    case Con{+low,+more} Con{+box,+rest} Con{+extent,+later}:
      KernelGemm.Both{+here,+after} = inside
      +inner = ShapeLaws.elements(later)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +extent_le = RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive)
      +inner_le = DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive)
      KernelGemm.Both{
        Order.transitive(Nat.double(Nat.add(low,U32.to_nat(box))),Nat.double(ShapeLaws.elements(Con{extent,later})),U32.to_nat(cap),
          Order.double_le(Nat.add(low,U32.to_nat(box)),ShapeLaws.elements(Con{extent,later}),Order.transitive(Nat.add(low,U32.to_nat(box)),U32.to_nat(extent),
            ShapeLaws.elements(Con{extent,later}),here,extent_le)),room),
        inside_rooms(more,rest,later,cap,Order.transitive(Nat.double(inner),Nat.double(ShapeLaws.elements(Con{extent,later})),U32.to_nat(cap),
          Order.double_le(inner,ShapeLaws.elements(Con{extent,later}),inner_le),room),inner_positive,
          Equal.cong(Nat,Nat,count => before(count),1n+List.length(&2,U32,rest),1n+List.length(&2,U32,later),same),after)}
    case Nil{} +boxes +extents: Unit{}
    case Con{+low,+more} Nil{} +extents: Unit{}
    case Con{+low,+more} Con{+box,+rest} Nil{}:
      Empty.absurd(Rooms(Con{low,more},Con{box,rest},cap),zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))

# Every axis of the box of a piece of a grid from the first cell of the piece
# fits half the capacity.
def grid_box_rooms(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  Rooms(words_nat(Grid.coordinates(extents,Grid.start(Grid.grid(extents,target),piece))),Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece)),Machine.half_word(limits)):
  +first = Grid.start(Grid.grid(extents,target),piece)
  +span = Grid.span(Grid.grid(extents,target),piece)
  +boxes = Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece))
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +elements = ShapeLaws.elements(extents)
  +span_positive = Targets.grid_span_positive(limits,extents,target,room,nonempty,piece,inside)
  +box_cells = Targets.grid_box_cells(limits,extents,target,room,nonempty,rank_room,piece,inside)
  +half_room = Order.transitive(elements,U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),room,Bounds.as_nat_le(Machine.capacity(limits),cap,Limits.capacity_within_half(limits)))
  +last = Nat.sub(ShapeLaws.elements(boxes),1n)
  +last_in = {%Equal.sym(Nat,ShapeLaws.elements(boxes),U32.to_nat(span),box_cells) : {Nat.is_lt(Nat.sub(_,1n),U32.to_nat(span)) == True{} : Bool}
    Order.last_below(U32.to_nat(span),span_positive) : {Nat.is_lt(last,U32.to_nat(span)) == True{} : Bool}}
  +steps = unit_steps(List.length(&2,U32,extents),0n)
  +first_in = Order.le_lt_trans(U32.to_nat(first),Nat.add(U32.to_nat(first),last),elements,Order.le_add(U32.to_nat(first),last),
    piece_ending(extents,steps,boxes,U32.to_nat(first),U32.to_nat(span),last,Box.grid_piece_box(limits,extents,steps,target,room,nonempty,rank_room,piece,inside,last,last_in)))
  +inside_box = {%Equal.sym(List<&2,Nat>,words_nat(Grid.coordinates(extents,first)),Targets.cell_coordinates(extents,U32.to_nat(first)),
      coordinates_exact(extents,first,cap,cap_ok,Limits.double_bound(limits,elements,room),first_in)) : BoxInside(_,boxes,extents)
    piece_inside(extents,boxes,U32.to_nat(first),U32.to_nat(span),cap,cap_ok,half_room,box_of_length(Grid.grid(extents,target),extents,span),
      grid_boxes(List.length(&2,U32,extents),limits,extents,target,room,nonempty,rank_room,piece,inside,last,last_in,0n)) :
    BoxInside(words_nat(Grid.coordinates(extents,first)),boxes,extents)}
  inside_rooms(words_nat(Grid.coordinates(extents,first)),boxes,extents,cap,Limits.double_bound(limits,elements,room),nonempty,box_of_length(Grid.grid(extents,target),extents,span),inside_box)

# ---- Strides that step at most half a word ----

# Every stride steps at most half a word.
def StridesIn(strides: List<&2,U32>,+word: U32) -> Data:
  match strides:
    case Nil{}: Unit
    case Con{+stride,rest}: Held.Both<{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},StridesIn(rest,word)>

# Every stride of extents that fit capacity steps at most half a word.
def strides_in(extents: List<&2,U32>,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  StridesIn(Tasks.strides_of(extents),Machine.half_word(limits)):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      +e = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +inner_le = {%Equal.trans(Nat,Nat.mul(1n,inner),Nat.mul(inner,1n),inner,RowMajor.multiply_commutative(1n,inner),Order.times_one(inner)) : {Nat.is_le(_,Nat.mul(e,inner)) == True{} : Bool}
        RowMajor.mul_le(1n,e,inner,Order.positive_one(e,RowMajor.left_factor_positive(e,inner,positive))) : {Nat.is_le(inner,Nat.mul(e,inner)) == True{} : Bool}}
      +inner_room = Order.transitive(inner,Nat.mul(e,inner),U32.to_nat(Machine.capacity(limits)),inner_le,room)
      +stride_exact = Equal.trans(Nat,U32.to_nat(Grid.product(rest)),U32.to_nat(U32.from_nat(inner)),inner,
        Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product(rest),U32.from_nat(inner),Equal.sym(U32,U32.from_nat(inner),Grid.product(rest),product_from(rest))),
        Bounds.roundtrip(inner,Machine.capacity(limits),Limits.capacity_fits(limits),inner_room))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        StridesIn(_,Machine.half_word(limits))
      Held.Both{{%Equal.sym(Nat,U32.to_nat(Grid.product(rest)),inner,stride_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} Limits.double_bound(limits,inner,inner_room) :
          {Nat.is_le(Nat.double(U32.to_nat(Grid.product(rest))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},
        strides_in(rest,limits,inner_room,DivisionOrder.right_factor_positive(e,inner,positive))}

# A stride picked from strides that step at most half a word steps at most
# half a word, and so does the zero past their end.
def picked_in(choice: Bool,+first: U32,+second: U32,+word: U32,+first_in: {Nat.is_le(Nat.double(U32.to_nat(first)),U32.to_nat(word)) == True{} : Bool},
  +second_in: {Nat.is_le(Nat.double(U32.to_nat(second)),U32.to_nat(word)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(Bool.pick(U32,choice,first,second))),U32.to_nat(word)) == True{} : Bool}:
  match choice:
    case True{}: first_in
    case False{}: second_in

def nth_in(strides: List<&2,U32>,+index: U32,+word: U32,+bounded: StridesIn(strides,word)) -> {Nat.is_le(Nat.double(U32.to_nat(Tasks.nth(strides,index))),U32.to_nat(word)) == True{} : Bool}:
  match strides:
    case Nil{}: Order.zero_le(U32.to_nat(word))
    case Con{+stride,+rest}:
      Held.Both{+here,+later} = bounded
      picked_in(U32.is_zero(index),stride,Tasks.nth(rest,(index - 1 : U32)),word,here,nth_in(rest,(index - 1 : U32),word,later))

