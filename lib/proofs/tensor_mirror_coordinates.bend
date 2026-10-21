# Recover mirrored cursor digits from their row-major position.
import Base
import ./tensor_mirror_address.bend as Mirror
import ./tensor_cursor_model.bend as Model
import ./tensor_cursor_order.bend as Order
import ./tensor_cursor_refinement.bend as Cursor
import ./tensor_coordinate_uniqueness.bend as Unique
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as Words

def flags(axes: List<&2,Mirror.Axis>) -> List<&2,Bool>:
  match axes:
    case []: []
    case Mirror.Axis{extent,coordinate,stride,reverse} <> tail: reverse <> flags(tail)

def later_flags(axes: List<&2,Bool>) -> List<&2,Bool>:
  match axes:
    case []: []
    case head <> tail: tail

def first_flag(axes: List<&2,Bool>) -> Bool:
  match axes:
    case []: False{}
    case head <> tail: head

def first_coordinate(axes: List<&2,Model.Coordinate>) -> Nat:
  match axes:
    case []: 0n
    case Model.Coordinate{extent,coordinate,stride} <> tail: coordinate

def selected_flags(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+tail: List<&2,Mirror.Axis>,
  equal: {flags(Mirror.advance(tail)) == flags(tail) : List<&2,Bool>}) ->
  {flags(Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,1n+coordinate,stride,reverse} <> tail,
    Mirror.Axis{extent,0n,stride,reverse} <> Mirror.advance(tail))) == flags(Mirror.Axis{extent,coordinate,stride,reverse} <> tail) : List<&2,Bool>}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(List<&2,Bool>,List<&2,Bool>,rest => reverse <> rest,flags(Mirror.advance(tail)),flags(tail),equal)

def advance_flags(+axes: List<&2,Mirror.Axis>) -> {flags(Mirror.advance(axes)) == flags(axes) : List<&2,Bool>}:
  match axes:
    case []: {==}
    case Mirror.Axis{+extent,+coordinate,+stride,+reverse} <> +tail:
      selected_flags(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,reverse,tail,advance_flags(tail))

def iterate_flags(count: Nat,+axes: List<&2,Mirror.Axis>) -> {flags(Mirror.iterate(count,axes)) == flags(axes) : List<&2,Bool>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(List<&2,Bool>,flags(Mirror.iterate(rest,Mirror.advance(axes))),flags(Mirror.advance(axes)),flags(axes),
        iterate_flags(rest,Mirror.advance(axes)),advance_flags(axes))

# The forward coordinates and the flags together determine the mirrored axes.
def injective(+left: List<&2,Mirror.Axis>,+right: List<&2,Mirror.Axis>,
  +same: {Mirror.modeled(False{},left) == Mirror.modeled(False{},right) : List<&2,Model.Coordinate>},
  +reversed: {flags(left) == flags(right) : List<&2,Bool>}) -> {left == right : List<&2,Mirror.Axis>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Nil{} Con{Mirror.Axis{+e,+c,+s,+r},+tail}:
      Empty.absurd({Nil{} == Mirror.Axis{e,c,s,r} <> tail : List<&2,Mirror.Axis>},Words.false_true(
        Equal.sym(Bool,True{},False{},Equal.cong(List<&2,Model.Coordinate>,Bool,Unique.empty,Mirror.modeled(False{},[]),Mirror.modeled(False{},Mirror.Axis{e,c,s,r} <> tail),same))))
    case Con{Mirror.Axis{+e,+c,+s,+r},+tail} Nil{}:
      Empty.absurd({Mirror.Axis{e,c,s,r} <> tail == Nil{} : List<&2,Mirror.Axis>},Words.false_true(
        Equal.cong(List<&2,Model.Coordinate>,Bool,Unique.empty,Mirror.modeled(False{},Mirror.Axis{e,c,s,r} <> tail),Mirror.modeled(False{},[]),same)))
    case Con{Mirror.Axis{+le,+lc,+ls,+lr},+lt} Con{Mirror.Axis{+re,+rc,+rs,+rr},+rt}:
      +ee = Equal.cong(List<&2,Model.Coordinate>,Nat,Unique.extent,Mirror.modeled(False{},left),Mirror.modeled(False{},right),same)
      +cc = Equal.cong(List<&2,Model.Coordinate>,Nat,first_coordinate,Mirror.modeled(False{},left),Mirror.modeled(False{},right),same)
      +ss = Arithmetic.to_nat_injective(ls,rs,Equal.cong(List<&2,Model.Coordinate>,Nat,Unique.stride,Mirror.modeled(False{},left),Mirror.modeled(False{},right),same))
      +ff = Equal.cong(List<&2,Bool>,Bool,first_flag,flags(left),flags(right),reversed)
      +tt = injective(lt,rt,Equal.cong(List<&2,Model.Coordinate>,List<&2,Model.Coordinate>,Unique.later,Mirror.modeled(False{},left),Mirror.modeled(False{},right),same),
        Equal.cong(List<&2,Bool>,List<&2,Bool>,later_flags,flags(left),flags(right),reversed))
      %ee : {Mirror.Axis{le,lc,ls,lr} <> lt == Mirror.Axis{_,rc,rs,rr} <> rt : List<&2,Mirror.Axis>}
      %cc : {Mirror.Axis{le,lc,ls,lr} <> lt == Mirror.Axis{le,_,rs,rr} <> rt : List<&2,Mirror.Axis>}
      %ss : {Mirror.Axis{le,lc,ls,lr} <> lt == Mirror.Axis{le,lc,_,rr} <> rt : List<&2,Mirror.Axis>}
      %ff : {Mirror.Axis{le,lc,ls,lr} <> lt == Mirror.Axis{le,lc,ls,_} <> rt : List<&2,Mirror.Axis>}
      %tt : {Mirror.Axis{le,lc,ls,lr} <> lt == Mirror.Axis{le,lc,ls,lr} <> _ : List<&2,Mirror.Axis>}
      {==}

def reached(+count: Nat,+initial: List<&2,Mirror.Axis>,+candidate: List<&2,Mirror.Axis>,
  +initial_valid: Cursor.Valid(Mirror.modeled(False{},initial)),candidate_valid: Cursor.Valid(Mirror.modeled(False{},candidate)),
  +same: {Unique.layout(Mirror.modeled(False{},initial)) == Unique.layout(Mirror.modeled(False{},candidate)) : List<&2,Model.Coordinate>},
  +reversed: {flags(initial) == flags(candidate) : List<&2,Bool>},
  +place: {Model.position(Mirror.modeled(False{},candidate)) == Nat.add(Model.position(Mirror.modeled(False{},initial)),count) : Nat},
  room: {Nat.is_lt(Nat.add(Model.position(Mirror.modeled(False{},initial)),count),Model.extent_product(Mirror.modeled(False{},initial))) == True{} : Bool}) ->
  {Mirror.iterate(count,initial) == candidate : List<&2,Mirror.Axis>}:
  injective(Mirror.iterate(count,initial),candidate,
    Equal.trans(List<&2,Model.Coordinate>,Mirror.modeled(False{},Mirror.iterate(count,initial)),Order.iterate(count,Mirror.modeled(False{},initial)),Mirror.modeled(False{},candidate),
      Equal.sym(List<&2,Model.Coordinate>,Order.iterate(count,Mirror.modeled(False{},initial)),Mirror.modeled(False{},Mirror.iterate(count,initial)),Mirror.modeled_iterate(False{},count,initial)),
      Unique.reached(count,Mirror.modeled(False{},initial),Mirror.modeled(False{},candidate),initial_valid,candidate_valid,same,place,room)),
    Equal.trans(List<&2,Bool>,flags(Mirror.iterate(count,initial)),flags(initial),flags(candidate),iterate_flags(count,initial),reversed))
