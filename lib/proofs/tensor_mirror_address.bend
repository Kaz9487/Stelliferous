# The addresses of a mirrored view (Views.mirrored), which reads its flagged
# axes in reverse through wrapping U32 arithmetic. Its address of an index is
# the address the view itself gives to the row-major digits of that index
# with the digit d of every flagged axis replaced by extent - 1 - d, and that
# address is inside every length the view fits.
import Base
import ../tensor_view.bend as Views
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_cursor_bounds.bend as CursorBounds
import ./tensor_copy_values.bend as CopyValues
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./storage_shape.bend as Shape

# One axis of extent above one: its digit, its stride in the view and whether
# it is read in reverse. The least significant axis is first.
type Axis is Data:
  Axis{extent: Nat,coordinate: Nat,stride: U32,reverse: Bool}

def active(mirror: Bool,reverse: Bool) -> Bool:
  match mirror:
    case False{}: False{}
    case True{}: reverse

def stride_of(mirror: Bool,reverse: Bool,stride: U32) -> Nat:
  U32.to_nat(Views.directed(active(mirror,reverse),stride))

# The coordinates of the cursor model: with the wrapped strides of the
# mirrored view (mirror), or with the strides of the view itself.
def modeled(+mirror: Bool,axes: List<&2,Axis>) -> List<&2,CursorModel.Coordinate>:
  match axes:
    case []: []
    case Axis{extent,coordinate,stride,reverse} <> tail:
      CursorModel.Coordinate{extent,coordinate,stride_of(mirror,reverse,stride)} <> modeled(mirror,tail)

def reflected_coordinate(reverse: Bool,extent: Nat,coordinate: Nat) -> Nat:
  match reverse:
    case False{}: coordinate
    case True{}: Nat.sub(Nat.sub(extent,1n),coordinate)

# The coordinates of the view itself, with every reversed digit reflected.
def reflected(axes: List<&2,Axis>) -> List<&2,CursorModel.Coordinate>:
  match axes:
    case []: []
    case Axis{+extent,coordinate,stride,reverse} <> tail:
      CursorModel.Coordinate{extent,reflected_coordinate(reverse,extent,coordinate),U32.to_nat(stride)} <> reflected(tail)

# The distance from the offset of the view to the first address read.
def turn_axis(reverse: Bool,extent: Nat,stride: U32) -> Nat:
  match reverse:
    case False{}: 0n
    case True{}: Nat.mul(Nat.sub(extent,1n),U32.to_nat(stride))

def turn(axes: List<&2,Axis>) -> Nat:
  match axes:
    case []: 0n
    case Axis{extent,coordinate,stride,reverse} <> tail: Nat.add(turn_axis(reverse,extent,stride),turn(tail))

def advance(axes: List<&2,Axis>) -> List<&2,Axis>:
  match axes:
    case []: []
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      Bool.pick(List<&2,Axis>,Nat.is_lt(1n+coordinate,extent),Axis{extent,1n+coordinate,stride,reverse} <> tail,Axis{extent,0n,stride,reverse} <> advance(tail))

def iterate(count: Nat,axes: List<&2,Axis>) -> List<&2,Axis>:
  match count:
    case 0n: axes
    case 1n+rest: iterate(rest,advance(axes))

def build(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,reversed: List<&2,Axis>) -> List<&2,Axis>:
  match axes:
    case []: reversed
    case Views.Axis{extent,stride} <> tail:
      +extent = U32.to_nat(extent)
      +reversed = reversed
      build(tail,Views.later_flags(flags),Bool.pick(List<&2,Axis>,Nat.is_lt(1n,extent),Axis{extent,0n,stride,Views.flag(flags)} <> reversed,reversed))

# Natural sums, and sums whose terms agree as U32.
def interchange(+a: Nat,+b: Nat,+c: Nat,+d: Nat) -> {Nat.add(Nat.add(a,b),Nat.add(c,d)) == Nat.add(Nat.add(a,c),Nat.add(b,d)) : Nat}:
  match a:
    case 0n: RowMajor.add_swap(b,c,d)
    case 1n+ +rest: Equal.cong(Nat,Nat,value => 1n+value,Nat.add(Nat.add(rest,b),Nat.add(c,d)),Nat.add(Nat.add(rest,c),Nat.add(b,d)),interchange(rest,b,c,d))

def both(+a: U32,+b: U32,+c: U32,+d: U32,left: {a == b : U32},right: {c == d : U32}) -> {U32.add(a,c) == U32.add(b,d) : U32}:
  Equal.trans(U32,U32.add(a,c),U32.add(b,c),U32.add(b,d),
    Equal.cong(U32,U32,value => U32.add(value,c),a,b,left),Equal.cong(U32,U32,value => U32.add(b,value),c,d,right))

def sum_congruent(+a: Nat,+b: Nat,+c: Nat,+d: Nat,left: {U32.from_nat(a) == U32.from_nat(c) : U32},right: {U32.from_nat(b) == U32.from_nat(d) : U32}) ->
  {U32.from_nat(Nat.add(a,b)) == U32.from_nat(Nat.add(c,d)) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.add(a,b)),U32.add(U32.from_nat(c),U32.from_nat(d)),U32.from_nat(Nat.add(c,d)),
    Equal.trans(U32,U32.from_nat(Nat.add(a,b)),U32.add(U32.from_nat(a),U32.from_nat(b)),U32.add(U32.from_nat(c),U32.from_nat(d)),
      Equal.sym(U32,U32.add(U32.from_nat(a),U32.from_nat(b)),U32.from_nat(Nat.add(a,b)),WordArithmetic.native_add(a,b)),
      both(U32.from_nat(a),U32.from_nat(c),U32.from_nat(b),U32.from_nat(d),left,right)),
    WordArithmetic.native_add(c,d))

# A negated stride and the stride add up to zero.
def opposite(+stride: U32) -> {U32.from_nat(Nat.add(U32.to_nat(U32.sub(0,stride)),U32.to_nat(stride))) == U32.from_nat(0n) : U32}:
  Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(U32.sub(0,stride)),U32.to_nat(stride))),U32.add(U32.sub(0,stride),stride),U32.from_nat(0n),
    Equal.sym(U32,U32.add(U32.sub(0,stride),stride),U32.from_nat(Nat.add(U32.to_nat(U32.sub(0,stride)),U32.to_nat(stride))),Arithmetic.native_add(U32.sub(0,stride),stride)),
    Subtraction.add_subtract(0,stride))

# count steps back from the last element are last - count steps forward.
def stepped_back(+count: Nat,+last: Nat,+backward: Nat,+forward: Nat,+zero: {U32.from_nat(Nat.add(backward,forward)) == U32.from_nat(0n) : U32},
  ordered: {Nat.is_le(count,last) == True{} : Bool}) ->
  {U32.from_nat(Nat.add(Nat.mul(count,backward),Nat.mul(last,forward))) == U32.from_nat(Nat.mul(Nat.sub(last,count),forward)) : U32}:
  match count last:
    case 0n 0n: {==}
    case 0n 1n+last: {==}
    case 1n+count 0n:
      Empty.absurd({U32.from_nat(Nat.add(Nat.mul(1n+count,backward),Nat.mul(0n,forward))) == U32.from_nat(Nat.mul(Nat.sub(0n,1n+count),forward)) : U32},WordEncoding.false_true(ordered))
    case 1n+ +count 1n+ +last:
      Equal.trans(U32,U32.from_nat(Nat.add(Nat.add(backward,Nat.mul(count,backward)),Nat.add(forward,Nat.mul(last,forward)))),
        U32.from_nat(Nat.add(Nat.add(backward,forward),Nat.add(Nat.mul(count,backward),Nat.mul(last,forward)))),
        U32.from_nat(Nat.mul(Nat.sub(last,count),forward)),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(backward,Nat.mul(count,backward)),Nat.add(forward,Nat.mul(last,forward))),
          Nat.add(Nat.add(backward,forward),Nat.add(Nat.mul(count,backward),Nat.mul(last,forward))),
          interchange(backward,Nat.mul(count,backward),forward,Nat.mul(last,forward))),
        sum_congruent(Nat.add(backward,forward),Nat.add(Nat.mul(count,backward),Nat.mul(last,forward)),0n,Nat.mul(Nat.sub(last,count),forward),
          zero,stepped_back(count,last,backward,forward,zero,ordered)))

# One axis: its term in the mirrored view, with its share of the turn, is its
# term in the view at the reflected digit.
def axis_address(reverse: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {U32.from_nat(Nat.add(Nat.mul(coordinate,U32.to_nat(Views.directed(reverse,stride))),turn_axis(reverse,extent,stride))) ==
    U32.from_nat(Nat.mul(reflected_coordinate(reverse,extent,coordinate),U32.to_nat(stride))) : U32}:
  match reverse extent coordinate:
    case False{} extent coordinate:
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),0n),Nat.mul(coordinate,U32.to_nat(stride)),Addition.zero_right(Nat.mul(coordinate,U32.to_nat(stride))))
    case True{} 0n 0n:
      Empty.absurd({U32.from_nat(Nat.add(Nat.mul(0n,U32.to_nat(U32.sub(0,stride))),Nat.mul(0n,U32.to_nat(stride)))) == U32.from_nat(Nat.mul(0n,U32.to_nat(stride))) : U32},WordEncoding.false_true(inside))
    case True{} 0n 1n+coordinate:
      Empty.absurd({U32.from_nat(Nat.add(Nat.mul(1n+coordinate,U32.to_nat(U32.sub(0,stride))),Nat.mul(0n,U32.to_nat(stride)))) == U32.from_nat(Nat.mul(0n,U32.to_nat(stride))) : U32},WordEncoding.false_true(inside))
    case True{} 1n+ +last coordinate:
      %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) :
        {U32.from_nat(Nat.add(Nat.mul(coordinate,U32.to_nat(U32.sub(0,stride))),Nat.mul(_,U32.to_nat(stride)))) == U32.from_nat(Nat.mul(Nat.sub(_,coordinate),U32.to_nat(stride))) : U32}
      stepped_back(coordinate,last,U32.to_nat(U32.sub(0,stride)),U32.to_nat(stride),opposite(stride),Order.strict_follow(coordinate,1n+last,inside))

def shifted(+axes: List<&2,CursorModel.Coordinate>,+amount: Nat,+offset: Nat) ->
  {CursorModel.address(axes,Nat.add(amount,offset)) == Nat.add(amount,CursorModel.address(axes,offset)) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      Equal.trans(Nat,Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,Nat.add(amount,offset))),
        Nat.add(Nat.mul(coordinate,stride),Nat.add(amount,CursorModel.address(tail,offset))),
        Nat.add(amount,Nat.add(Nat.mul(coordinate,stride),CursorModel.address(tail,offset))),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(coordinate,stride),value),CursorModel.address(tail,Nat.add(amount,offset)),Nat.add(amount,CursorModel.address(tail,offset)),shifted(tail,amount,offset)),
        RowMajor.add_swap(Nat.mul(coordinate,stride),amount,CursorModel.address(tail,offset)))

def offset_congruent(+axes: List<&2,CursorModel.Coordinate>,+left: Nat,+right: Nat,same: {U32.from_nat(left) == U32.from_nat(right) : U32}) ->
  {U32.from_nat(CursorModel.address(axes,left)) == U32.from_nat(CursorModel.address(axes,right)) : U32}:
  match axes:
    case []: same
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      sum_congruent(Nat.mul(coordinate,stride),CursorModel.address(tail,left),Nat.mul(coordinate,stride),CursorModel.address(tail,right),{==},offset_congruent(tail,left,right,same))

# The mirrored coordinates, read from the turned offset, give the address of
# the reflected coordinates of the view.
def turned(+axes: List<&2,Axis>,+offset: Nat,valid: CursorLaws.Valid(modeled(True{},axes))) ->
  {U32.from_nat(CursorModel.address(modeled(True{},axes),Nat.add(turn(axes),offset))) == U32.from_nat(CursorModel.address(reflected(axes),offset)) : U32}:
  match axes:
    case []: {==}
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,tail_valid} = valid
      +term = Nat.mul(coordinate,U32.to_nat(Views.directed(reverse,stride)))
      +share = turn_axis(reverse,extent,stride)
      +rest = CursorModel.address(modeled(True{},tail),Nat.add(turn(tail),offset))
      Equal.trans(U32,U32.from_nat(Nat.add(term,CursorModel.address(modeled(True{},tail),Nat.add(Nat.add(share,turn(tail)),offset)))),
        U32.from_nat(Nat.add(Nat.add(term,share),rest)),
        U32.from_nat(Nat.add(Nat.mul(reflected_coordinate(reverse,extent,coordinate),U32.to_nat(stride)),CursorModel.address(reflected(tail),offset))),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(term,CursorModel.address(modeled(True{},tail),Nat.add(Nat.add(share,turn(tail)),offset))),Nat.add(Nat.add(term,share),rest),
          Equal.trans(Nat,Nat.add(term,CursorModel.address(modeled(True{},tail),Nat.add(Nat.add(share,turn(tail)),offset))),Nat.add(term,Nat.add(share,rest)),Nat.add(Nat.add(term,share),rest),
            Equal.cong(Nat,Nat,value => Nat.add(term,value),CursorModel.address(modeled(True{},tail),Nat.add(Nat.add(share,turn(tail)),offset)),Nat.add(share,rest),
              Equal.trans(Nat,CursorModel.address(modeled(True{},tail),Nat.add(Nat.add(share,turn(tail)),offset)),CursorModel.address(modeled(True{},tail),Nat.add(share,Nat.add(turn(tail),offset))),Nat.add(share,rest),
                Equal.cong(Nat,Nat,value => CursorModel.address(modeled(True{},tail),value),Nat.add(Nat.add(share,turn(tail)),offset),Nat.add(share,Nat.add(turn(tail),offset)),Addition.associative(share,turn(tail),offset)),
                shifted(modeled(True{},tail),share,Nat.add(turn(tail),offset)))),
            Equal.sym(Nat,Nat.add(Nat.add(term,share),rest),Nat.add(term,Nat.add(share,rest)),Addition.associative(term,share,rest)))),
        sum_congruent(Nat.add(term,share),rest,Nat.mul(reflected_coordinate(reverse,extent,coordinate),U32.to_nat(stride)),CursorModel.address(reflected(tail),offset),
          axis_address(reverse,extent,coordinate,stride,inside),turned(tail,offset,tail_valid)))

# A step of the model is a step of the axes, and no step changes the turn.
def selected_modeled(+mirror: Bool,+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+tail: List<&2,Axis>,
  step: {CursorModel.advance(modeled(mirror,tail)) == modeled(mirror,advance(tail)) : List<&2,CursorModel.Coordinate>}) ->
  {Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride_of(mirror,reverse,stride)} <> modeled(mirror,tail),
      CursorModel.Coordinate{extent,0n,stride_of(mirror,reverse,stride)} <> CursorModel.advance(modeled(mirror,tail))) ==
    modeled(mirror,Bool.pick(List<&2,Axis>,selected,Axis{extent,1n+coordinate,stride,reverse} <> tail,Axis{extent,0n,stride,reverse} <> advance(tail))) : List<&2,CursorModel.Coordinate>}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(List<&2,CursorModel.Coordinate>,List<&2,CursorModel.Coordinate>,rest => CursorModel.Coordinate{extent,0n,stride_of(mirror,reverse,stride)} <> rest,
        CursorModel.advance(modeled(mirror,tail)),modeled(mirror,advance(tail)),step)

def modeled_advance(+mirror: Bool,+axes: List<&2,Axis>) -> {CursorModel.advance(modeled(mirror,axes)) == modeled(mirror,advance(axes)) : List<&2,CursorModel.Coordinate>}:
  match axes:
    case []: {==}
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      selected_modeled(mirror,Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,reverse,tail,modeled_advance(mirror,tail))

def modeled_iterate(+mirror: Bool,count: Nat,+axes: List<&2,Axis>) ->
  {CursorOrder.iterate(count,modeled(mirror,axes)) == modeled(mirror,iterate(count,axes)) : List<&2,CursorModel.Coordinate>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(List<&2,CursorModel.Coordinate>,CursorModel.advance(modeled(mirror,axes)),modeled(mirror,advance(axes)),modeled_advance(mirror,axes)) :
        {CursorOrder.iterate(rest,_) == modeled(mirror,iterate(rest,advance(axes))) : List<&2,CursorModel.Coordinate>}
      modeled_iterate(mirror,rest,advance(axes))

def selected_turn(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+tail: List<&2,Axis>,step: {turn(advance(tail)) == turn(tail) : Nat}) ->
  {turn(Bool.pick(List<&2,Axis>,selected,Axis{extent,1n+coordinate,stride,reverse} <> tail,Axis{extent,0n,stride,reverse} <> advance(tail))) ==
    turn(Axis{extent,coordinate,stride,reverse} <> tail) : Nat}:
  match selected:
    case True{}: {==}
    case False{}: Equal.cong(Nat,Nat,value => Nat.add(turn_axis(reverse,extent,stride),value),turn(advance(tail)),turn(tail),step)

def advance_turn(+axes: List<&2,Axis>) -> {turn(advance(axes)) == turn(axes) : Nat}:
  match axes:
    case []: {==}
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      selected_turn(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,reverse,tail,advance_turn(tail))

def iterate_turn(count: Nat,+axes: List<&2,Axis>) -> {turn(iterate(count,axes)) == turn(axes) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.trans(Nat,turn(iterate(rest,advance(axes))),turn(advance(axes)),turn(axes),iterate_turn(rest,advance(axes)),advance_turn(axes))

# The reflected coordinates are coordinates of the view, with its span.
def reflected_inside(reverse: Bool,+extent: Nat,+coordinate: Nat,inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {Nat.is_lt(reflected_coordinate(reverse,extent,coordinate),extent) == True{} : Bool}:
  match reverse extent coordinate:
    case False{} extent coordinate: inside
    case True{} 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(inside))
    case True{} 0n 1n+coordinate: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(inside))
    case True{} 1n+ +last coordinate:
      %Equal.sym(Nat,Nat.sub(last,0n),last,Order.subtract_zero(last)) : {Nat.is_lt(Nat.sub(_,coordinate),1n+last) == True{} : Bool}
      Order.lt_succ(Nat.sub(last,coordinate),last,Order.sub_le(last,coordinate))

def reflected_valid(+axes: List<&2,Axis>,valid: CursorLaws.Valid(modeled(True{},axes))) -> CursorLaws.Valid(reflected(axes)):
  match axes:
    case []: Unit{}
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,tail_valid} = valid
      Shape.Both{reflected_inside(reverse,extent,coordinate,inside),reflected_valid(tail,tail_valid)}

def reflected_span(+axes: List<&2,Axis>) -> {CursorModel.span(reflected(axes)) == CursorModel.span(modeled(False{},axes)) : Nat}:
  match axes:
    case []: {==}
    case Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.sub(extent,1n),U32.to_nat(stride)),value),CursorModel.span(reflected(tail)),CursorModel.span(modeled(False{},tail)),reflected_span(tail))

# The axes built from a view are the prepared axes of the mirrored view and
# of the view itself.
def picked(+mirror: Bool,+selected: Bool,+extent: Nat,+stride: U32,+reverse: Bool,+reversed: List<&2,Axis>) ->
  {Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,0n,stride_of(mirror,reverse,stride)} <> modeled(mirror,reversed),modeled(mirror,reversed)) ==
    modeled(mirror,Bool.pick(List<&2,Axis>,selected,Axis{extent,0n,stride,reverse} <> reversed,reversed)) : List<&2,CursorModel.Coordinate>}:
  match selected:
    case True{}: {==}
    case False{}: {==}

def built_mirrored(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+reversed: List<&2,Axis>) ->
  {CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),modeled(True{},reversed)) == modeled(True{},build(axes,flags,reversed)) : List<&2,CursorModel.Coordinate>}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      %Equal.sym(List<&2,CursorModel.Coordinate>,
        Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,stride_of(True{},Views.flag(flags),stride)} <> modeled(True{},reversed),modeled(True{},reversed)),
        modeled(True{},Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed)),
        picked(True{},Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),stride,Views.flag(flags),reversed)) :
        {CursorModel.prepare_axes(Views.mirrored_axes(tail,Views.later_flags(flags)),_) == modeled(True{},build(Views.Axis{extent,stride} <> tail,flags,reversed)) : List<&2,CursorModel.Coordinate>}
      built_mirrored(tail,Views.later_flags(flags),Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed))

def built_forward(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+reversed: List<&2,Axis>) ->
  {CursorModel.prepare_axes(axes,modeled(False{},reversed)) == modeled(False{},build(axes,flags,reversed)) : List<&2,CursorModel.Coordinate>}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      %Equal.sym(List<&2,CursorModel.Coordinate>,
        Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,stride_of(False{},Views.flag(flags),stride)} <> modeled(False{},reversed),modeled(False{},reversed)),
        modeled(False{},Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed)),
        picked(False{},Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),stride,Views.flag(flags),reversed)) :
        {CursorModel.prepare_axes(tail,_) == modeled(False{},build(Views.Axis{extent,stride} <> tail,flags,reversed)) : List<&2,CursorModel.Coordinate>}
      built_forward(tail,Views.later_flags(flags),Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed))

# The offset of the mirrored view is the offset of the view plus the turn.
def start_when(reverse: Bool,selected: Bool,extent: U32,stride: U32,+offset: U32) -> U32:
  match reverse:
    case False{}: offset
    case True{}: Bool.pick(U32,selected,(offset + (extent - 1) * stride : U32),offset)

def start_same(reverse: Bool,+extent: U32,+stride: U32,+offset: U32) ->
  {Views.mirrored_start(reverse,extent,stride,offset) == start_when(reverse,U32.is_lt(1,extent),extent,stride,offset) : U32}:
  match reverse:
    case False{}: {==}
    case True{}: {==}

def start_selected(reverse: Bool,selected: Bool,+extent: U32,+stride: U32,+reversed: List<&2,Axis>,+offset: U32,+base: Nat,
  guard: {Nat.is_lt(1n,U32.to_nat(extent)) == selected : Bool},start: {offset == U32.from_nat(Nat.add(turn(reversed),base)) : U32}) ->
  {start_when(reverse,selected,extent,stride,offset) ==
    U32.from_nat(Nat.add(turn(Bool.pick(List<&2,Axis>,selected,Axis{U32.to_nat(extent),0n,stride,reverse} <> reversed,reversed)),base)) : U32}:
  match reverse selected:
    case False{} False{}: start
    case False{} True{}: start
    case True{} False{}: start
    case True{} True{}:
      +last = Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride))
      Equal.trans(U32,U32.add(offset,U32.mul(U32.sub(extent,1),stride)),U32.from_nat(Nat.add(Nat.add(turn(reversed),base),last)),U32.from_nat(Nat.add(Nat.add(last,turn(reversed)),base)),
        Equal.trans(U32,U32.add(offset,U32.mul(U32.sub(extent,1),stride)),U32.add(U32.from_nat(Nat.add(turn(reversed),base)),U32.from_nat(last)),U32.from_nat(Nat.add(Nat.add(turn(reversed),base),last)),
          both(offset,U32.from_nat(Nat.add(turn(reversed),base)),U32.mul(U32.sub(extent,1),stride),U32.from_nat(last),start,CursorLaws.rewind(extent,stride,guard)),
          WordArithmetic.native_add(Nat.add(turn(reversed),base),last)),
        Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(turn(reversed),base),last),Nat.add(Nat.add(last,turn(reversed)),base),
          Equal.trans(Nat,Nat.add(Nat.add(turn(reversed),base),last),Nat.add(last,Nat.add(turn(reversed),base)),Nat.add(Nat.add(last,turn(reversed)),base),
            Order.commutative(Nat.add(turn(reversed),base),last),
            Equal.sym(Nat,Nat.add(Nat.add(last,turn(reversed)),base),Nat.add(last,Nat.add(turn(reversed),base)),Addition.associative(last,turn(reversed),base)))))

def started(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+reversed: List<&2,Axis>,+offset: U32,+base: Nat,
  start: {offset == U32.from_nat(Nat.add(turn(reversed),base)) : U32}) ->
  {Views.mirrored_offset(axes,flags,offset) == U32.from_nat(Nat.add(turn(build(axes,flags,reversed)),base)) : U32}:
  match axes:
    case []: start
    case Views.Axis{+extent,+stride} <> +tail:
      started(tail,Views.later_flags(flags),Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed),
        Views.mirrored_start(Views.flag(flags),extent,stride,offset),base,
        Equal.trans(U32,Views.mirrored_start(Views.flag(flags),extent,stride,offset),start_when(Views.flag(flags),U32.is_lt(1,extent),extent,stride,offset),
          U32.from_nat(Nat.add(turn(Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed)),base)),
          start_same(Views.flag(flags),extent,stride,offset),{
            %Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(1n,U32.to_nat(extent)),Comparison.lt(1,extent)) :
              {start_when(Views.flag(flags),_,extent,stride,offset) ==
                U32.from_nat(Nat.add(turn(Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed)),base)) : U32}
            start_selected(Views.flag(flags),Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,reversed,offset,base,{==},start) :
              {start_when(Views.flag(flags),U32.is_lt(1,extent),extent,stride,offset) ==
                U32.from_nat(Nat.add(turn(Bool.pick(List<&2,Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Axis{U32.to_nat(extent),0n,stride,Views.flag(flags)} <> reversed,reversed)),base)) : U32}}))

# The model of the mirrored view at an index, and its validity.
def current(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat) ->
  {CursorOrder.iterate(index,CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),[])) == modeled(True{},iterate(index,build(axes,flags,[]))) : List<&2,CursorModel.Coordinate>}:
  Equal.trans(List<&2,CursorModel.Coordinate>,CursorOrder.iterate(index,CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),[])),
    CursorOrder.iterate(index,modeled(True{},build(axes,flags,[]))),modeled(True{},iterate(index,build(axes,flags,[]))),
    Equal.cong(List<&2,CursorModel.Coordinate>,List<&2,CursorModel.Coordinate>,value => CursorOrder.iterate(index,value),
      CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),[]),modeled(True{},build(axes,flags,[])),built_mirrored(axes,flags,[])),
    modeled_iterate(True{},index,build(axes,flags,[])))

def current_valid(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat) -> CursorLaws.Valid(modeled(True{},iterate(index,build(axes,flags,[])))):
  %current(axes,flags,index) : CursorLaws.Valid(_)
  CursorOrder.iterate_valid(index,CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),[]),CursorLaws.prepared_valid(Views.mirrored_axes(axes,flags),[],Unit{}))

# The digits of an index over the extents of the view (axes of extent one
# carry no digit), with the digit of every flagged axis reflected.
def coordinates(source: Views.View,+flags: List<&2,Bool>,index: Nat) -> List<&2,CursorModel.Coordinate>:
  Views.View{axes,offset} = source
  reflected(iterate(index,build(axes,flags,[])))

# The address the view gives to those digits: their dot product with its
# strides, plus its offset.
def address(source: Views.View,+flags: List<&2,Bool>,index: Nat) -> U32:
  Views.View{axes,offset} = source
  U32.from_nat(CursorModel.address(reflected(iterate(index,build(axes,flags,[]))),U32.to_nat(offset)))

# The mirrored view addresses every index by that address.
def mirrored(+source: Views.View,+flags: List<&2,Bool>,+index: Nat) -> {CopyValues.address(Views.mirrored(source,flags),index) == address(source,flags,index) : U32}:
  Views.View{+axes,+offset} = source
  +start = build(axes,flags,[])
  +reached = iterate(index,start)
  +base = U32.to_nat(offset)
  +moved = Views.mirrored_offset(axes,flags,offset)
  %Equal.sym(List<&2,CursorModel.Coordinate>,CursorOrder.iterate(index,CursorModel.prepare_axes(Views.mirrored_axes(axes,flags),[])),modeled(True{},reached),current(axes,flags,index)) :
    {U32.from_nat(CursorModel.address(_,U32.to_nat(moved))) == U32.from_nat(CursorModel.address(reflected(reached),base)) : U32}
  Equal.trans(U32,U32.from_nat(CursorModel.address(modeled(True{},reached),U32.to_nat(moved))),U32.from_nat(CursorModel.address(modeled(True{},reached),Nat.add(turn(reached),base))),
    U32.from_nat(CursorModel.address(reflected(reached),base)),
    offset_congruent(modeled(True{},reached),U32.to_nat(moved),Nat.add(turn(reached),base),
      Equal.trans(U32,U32.from_nat(U32.to_nat(moved)),moved,U32.from_nat(Nat.add(turn(reached),base)),Arithmetic.native_roundtrip(moved),
        Equal.trans(U32,moved,U32.from_nat(Nat.add(turn(start),base)),U32.from_nat(Nat.add(turn(reached),base)),
          started(axes,flags,[],offset,base,Equal.sym(U32,U32.from_nat(U32.to_nat(offset)),offset,Arithmetic.native_roundtrip(offset))),
          Equal.cong(Nat,U32,value => U32.from_nat(Nat.add(value,base)),turn(start),turn(reached),Equal.sym(Nat,turn(reached),turn(start),iterate_turn(index,start)))))),
    turned(reached,base,current_valid(axes,flags,index)))

# That address is inside every length the view fits.
def inside(+source: Views.View,+flags: List<&2,Bool>,+index: Nat,+length: U32,fits: {Views.fits_nonempty(False{},source,length) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(address(source,flags,index)),U32.to_nat(length)) == True{} : Bool}:
  Views.View{+axes,+offset} = source
  +start = build(axes,flags,[])
  +reached = iterate(index,start)
  CursorBounds.read_inside(reflected(reached),U32.to_nat(offset),length,reflected_valid(reached,current_valid(axes,flags,index)),{
    %Equal.sym(Nat,CursorModel.span(reflected(reached)),CursorModel.span(CursorModel.prepare_axes(axes,[])),
      Equal.trans(Nat,CursorModel.span(reflected(reached)),CursorModel.span(modeled(False{},start)),CursorModel.span(CursorModel.prepare_axes(axes,[])),
        Equal.trans(Nat,CursorModel.span(reflected(reached)),CursorModel.span(modeled(False{},reached)),CursorModel.span(modeled(False{},start)),reflected_span(reached),
          Equal.trans(Nat,CursorModel.span(modeled(False{},reached)),CursorModel.span(CursorOrder.iterate(index,modeled(False{},start))),CursorModel.span(modeled(False{},start)),
            Equal.cong(List<&2,CursorModel.Coordinate>,Nat,value => CursorModel.span(value),modeled(False{},reached),CursorOrder.iterate(index,modeled(False{},start)),
              Equal.sym(List<&2,CursorModel.Coordinate>,CursorOrder.iterate(index,modeled(False{},start)),modeled(False{},reached),modeled_iterate(False{},index,start))),
            CopyValues.iterate_span(index,modeled(False{},start)))),
        Equal.cong(List<&2,CursorModel.Coordinate>,Nat,value => CursorModel.span(value),modeled(False{},start),CursorModel.prepare_axes(axes,[]),
          Equal.sym(List<&2,CursorModel.Coordinate>,CursorModel.prepare_axes(axes,[]),modeled(False{},start),built_forward(axes,flags,[]))))) :
      {Nat.is_lt(Nat.add(_,U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}
    CursorBounds.fits_bound(axes,offset,length,fits) : {Nat.is_lt(Nat.add(CursorModel.span(reflected(reached)),U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}})

# A view mirrored with no flag is the view itself, so its address of an
# index is the address the view gives to it.
def unmirrored_axes(+axes: List<&2,Views.Axis>) -> {Views.mirrored_axes(axes,[]) == axes : List<&2,Views.Axis>}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,rest => Views.Axis{extent,stride} <> rest,Views.mirrored_axes(tail,[]),tail,unmirrored_axes(tail))

def unmirrored_offset(+axes: List<&2,Views.Axis>,+offset: U32) -> {Views.mirrored_offset(axes,[],offset) == offset : U32}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail: unmirrored_offset(tail,offset)

def forward(+source: Views.View,+index: Nat) -> {address(source,[],index) == CopyValues.address(source,index) : U32}:
  Views.View{+axes,+offset} = source
  %unmirrored_axes(axes) : {address(Views.View{axes,offset},[],index) == CopyValues.address(Views.View{_,offset},index) : U32}
  %unmirrored_offset(axes,offset) : {address(Views.View{axes,offset},[],index) == CopyValues.address(Views.View{Views.mirrored_axes(axes,[]),_},index) : U32}
  Equal.sym(U32,CopyValues.address(Views.mirrored(Views.View{axes,offset},[]),index),address(Views.View{axes,offset},[],index),mirrored(Views.View{axes,offset},[],index))
