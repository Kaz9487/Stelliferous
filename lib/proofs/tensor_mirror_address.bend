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
import ../kernel_gemm.bend as KernelGemm

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

def picked(+mirror: Bool,+selected: Bool,+extent: Nat,+stride: U32,+reverse: Bool,+reversed: List<&2,Axis>) ->
  {Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,0n,stride_of(mirror,reverse,stride)} <> modeled(mirror,reversed),modeled(mirror,reversed)) ==
    modeled(mirror,Bool.pick(List<&2,Axis>,selected,Axis{extent,0n,stride,reverse} <> reversed,reversed)) : List<&2,CursorModel.Coordinate>}:
  match selected:
    case True{}: {==}
    case False{}: {==}

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

def unmirrored_axes(+axes: List<&2,Views.Axis>) -> {Views.mirrored_axes(axes,[]) == axes : List<&2,Views.Axis>}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,rest => Views.Axis{extent,stride} <> rest,Views.mirrored_axes(tail,[]),tail,unmirrored_axes(tail))

def unmirrored_offset(+axes: List<&2,Views.Axis>,+offset: U32) -> {Views.mirrored_offset(axes,[],offset) == offset : U32}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail: unmirrored_offset(tail,offset)
