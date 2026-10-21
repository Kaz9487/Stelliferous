# Valid cursor digits are determined by their layout and row-major position.
import Base
import ./tensor_cursor_model.bend as Model
import ./tensor_cursor_refinement.bend as Cursor
import ./tensor_cursor_order.bend as Order
import ./storage_shape.bend as Shape
import ./binary_radix.bend as Radix
import ./nat_order.bend as NatOrder
import ./word_encoding.bend as Words
import ./row_major_index.bend as RowMajor

def layout(axes: List<&2,Model.Coordinate>) -> List<&2,Model.Coordinate>:
  match axes:
    case []: []
    case Model.Coordinate{extent,coordinate,stride} <> tail:
      Model.Coordinate{extent,0n,stride} <> layout(tail)

def empty(axes: List<&2,Model.Coordinate>) -> Bool:
  match axes:
    case []: True{}
    case head <> tail: False{}

def extent(axes: List<&2,Model.Coordinate>) -> Nat:
  match axes:
    case []: 0n
    case Model.Coordinate{extent,coordinate,stride} <> tail: extent

def stride(axes: List<&2,Model.Coordinate>) -> Nat:
  match axes:
    case []: 0n
    case Model.Coordinate{extent,coordinate,stride} <> tail: stride

def later(axes: List<&2,Model.Coordinate>) -> List<&2,Model.Coordinate>:
  match axes:
    case []: []
    case head <> tail: tail

def selected(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,Model.Coordinate>,
  equal: {layout(Model.advance(tail)) == layout(tail) : List<&2,Model.Coordinate>}) ->
  {layout(Bool.pick(List<&2,Model.Coordinate>,selected,Model.Coordinate{extent,1n+coordinate,stride} <> tail,
    Model.Coordinate{extent,0n,stride} <> Model.advance(tail))) == layout(Model.Coordinate{extent,coordinate,stride} <> tail) : List<&2,Model.Coordinate>}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(List<&2,Model.Coordinate>,List<&2,Model.Coordinate>,rest => Model.Coordinate{extent,0n,stride} <> rest,
        layout(Model.advance(tail)),layout(tail),equal)

def advanced(+axes: List<&2,Model.Coordinate>) -> {layout(Model.advance(axes)) == layout(axes) : List<&2,Model.Coordinate>}:
  match axes:
    case []: {==}
    case Model.Coordinate{+extent,+coordinate,+stride} <> +tail:
      selected(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,advanced(tail))

def iterated(count: Nat,+axes: List<&2,Model.Coordinate>) -> {layout(Order.iterate(count,axes)) == layout(axes) : List<&2,Model.Coordinate>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(List<&2,Model.Coordinate>,layout(Order.iterate(rest,Model.advance(axes))),layout(Model.advance(axes)),layout(axes),
        iterated(rest,Model.advance(axes)),advanced(axes))

def position_form(+extent: Nat,+coordinate: Nat,+tail: List<&2,Model.Coordinate>) ->
  {Nat.add(coordinate,Nat.mul(extent,Model.position(tail))) == Nat.add(Nat.mul(Model.position(tail),extent),coordinate) : Nat}:
  Equal.trans(Nat,Nat.add(coordinate,Nat.mul(extent,Model.position(tail))),Nat.add(Nat.mul(extent,Model.position(tail)),coordinate),
    Nat.add(Nat.mul(Model.position(tail),extent),coordinate),NatOrder.commutative(coordinate,Nat.mul(extent,Model.position(tail))),
    Equal.cong(Nat,Nat,value => Nat.add(value,coordinate),Nat.mul(extent,Model.position(tail)),Nat.mul(Model.position(tail),extent),
      RowMajor.multiply_commutative(extent,Model.position(tail))))


def aligned(+extent: Nat,+a: Nat,+b: Nat,+stride: Nat,+left: List<&2,Model.Coordinate>,+right: List<&2,Model.Coordinate>,
  +left_inside: {Nat.is_lt(a,extent) == True{} : Bool},+right_inside: {Nat.is_lt(b,extent) == True{} : Bool},
  +same: {layout(left) == layout(right) : List<&2,Model.Coordinate>},
  +position: {Model.position(Model.Coordinate{extent,a,stride} <> left) == Model.position(Model.Coordinate{extent,b,stride} <> right) : Nat},
  tails: {left == right : List<&2,Model.Coordinate>}) ->
  {Model.Coordinate{extent,a,stride} <> left == Model.Coordinate{extent,b,stride} <> right : List<&2,Model.Coordinate>}:
  +number = Equal.trans(Nat,Nat.add(Nat.mul(Model.position(left),extent),a),Model.position(Model.Coordinate{extent,a,stride} <> left),Nat.add(Nat.mul(Model.position(right),extent),b),
    Equal.sym(Nat,Model.position(Model.Coordinate{extent,a,stride} <> left),Nat.add(Nat.mul(Model.position(left),extent),a),position_form(extent,a,left)),
    Equal.trans(Nat,Model.position(Model.Coordinate{extent,a,stride} <> left),Model.position(Model.Coordinate{extent,b,stride} <> right),Nat.add(Nat.mul(Model.position(right),extent),b),position,position_form(extent,b,right)))
  %Radix.col_unique(Model.position(left),Model.position(right),extent,a,b,NatOrder.positive_bound(a,extent,left_inside),left_inside,right_inside,number) :
    {Model.Coordinate{extent,a,stride} <> left == Model.Coordinate{extent,_,stride} <> right : List<&2,Model.Coordinate>}
  %tails : {Model.Coordinate{extent,a,stride} <> left == Model.Coordinate{extent,a,stride} <> _ : List<&2,Model.Coordinate>}
  {==}

def tail_position(+extent: Nat,+a: Nat,+b: Nat,+stride: Nat,+left: List<&2,Model.Coordinate>,+right: List<&2,Model.Coordinate>,
  +left_inside: {Nat.is_lt(a,extent) == True{} : Bool},+right_inside: {Nat.is_lt(b,extent) == True{} : Bool},
  position: {Model.position(Model.Coordinate{extent,a,stride} <> left) == Model.position(Model.Coordinate{extent,b,stride} <> right) : Nat}) ->
  {Model.position(left) == Model.position(right) : Nat}:
  Radix.row_unique(Model.position(left),Model.position(right),extent,a,b,NatOrder.positive_bound(a,extent,left_inside),left_inside,right_inside,
    Equal.trans(Nat,Nat.add(Nat.mul(Model.position(left),extent),a),Model.position(Model.Coordinate{extent,a,stride} <> left),Nat.add(Nat.mul(Model.position(right),extent),b),
      Equal.sym(Nat,Model.position(Model.Coordinate{extent,a,stride} <> left),Nat.add(Nat.mul(Model.position(left),extent),a),position_form(extent,a,left)),
      Equal.trans(Nat,Model.position(Model.Coordinate{extent,a,stride} <> left),Model.position(Model.Coordinate{extent,b,stride} <> right),Nat.add(Nat.mul(Model.position(right),extent),b),position,position_form(extent,b,right))))

def unique(+left: List<&2,Model.Coordinate>,+right: List<&2,Model.Coordinate>,
  valid_left: Cursor.Valid(left),valid_right: Cursor.Valid(right),
  +same: {layout(left) == layout(right) : List<&2,Model.Coordinate>},
  +position: {Model.position(left) == Model.position(right) : Nat}) -> {left == right : List<&2,Model.Coordinate>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Nil{} Con{Model.Coordinate{+extent,+coordinate,+stride},+tail}:
      +head = {Model.Coordinate{extent,coordinate,stride} : Model.Coordinate}
      Empty.absurd({Nil{} == Con{head,tail} : List<&2,Model.Coordinate>},Words.false_true(Equal.sym(Bool,True{},False{},
        Equal.cong(List<&2,Model.Coordinate>,Bool,empty,layout([]),layout(head <> tail),same))))
    case Con{Model.Coordinate{+extent,+coordinate,+stride},+tail} Nil{}:
      +head = {Model.Coordinate{extent,coordinate,stride} : Model.Coordinate}
      Empty.absurd({Con{head,tail} == Nil{} : List<&2,Model.Coordinate>},Words.false_true(
        Equal.cong(List<&2,Model.Coordinate>,Bool,empty,layout(head <> tail),layout([]),same)))
    case Con{Model.Coordinate{+le,+lc,+ls},+lt} Con{Model.Coordinate{+re,+rc,+rs},+rt}:
      Shape.Both{+li,+lv} = valid_left
      Shape.Both{+ri,+rv} = valid_right
      +ee = Equal.cong(List<&2,Model.Coordinate>,Nat,extent,layout(Model.Coordinate{le,lc,ls} <> lt),layout(Model.Coordinate{re,rc,rs} <> rt),same)
      +ss = Equal.cong(List<&2,Model.Coordinate>,Nat,stride,layout(Model.Coordinate{le,lc,ls} <> lt),layout(Model.Coordinate{re,rc,rs} <> rt),same)
      +tt = Equal.cong(List<&2,Model.Coordinate>,List<&2,Model.Coordinate>,later,layout(Model.Coordinate{le,lc,ls} <> lt),layout(Model.Coordinate{re,rc,rs} <> rt),same)
      %ee : {Model.Coordinate{le,lc,ls} <> lt == Model.Coordinate{_,rc,rs} <> rt : List<&2,Model.Coordinate>}
      %ss : {Model.Coordinate{le,lc,ls} <> lt == Model.Coordinate{le,rc,_} <> rt : List<&2,Model.Coordinate>}
      aligned(le,lc,rc,ls,lt,rt,li,
        %Equal.sym(Nat,le,re,ee) : {Nat.is_lt(rc,_) == True{} : Bool} ri,tt,
        %Equal.sym(Nat,le,re,ee) : {Model.position(Model.Coordinate{le,lc,ls} <> lt) == Model.position(Model.Coordinate{_,rc,ls} <> rt) : Nat}
        %Equal.sym(Nat,ls,rs,ss) : {Model.position(Model.Coordinate{le,lc,ls} <> lt) == Model.position(Model.Coordinate{re,rc,_} <> rt) : Nat} position,
        unique(lt,rt,lv,rv,tt,tail_position(le,lc,rc,ls,lt,rt,li,
          %Equal.sym(Nat,le,re,ee) : {Nat.is_lt(rc,_) == True{} : Bool} ri,
          %Equal.sym(Nat,le,re,ee) : {Model.position(Model.Coordinate{le,lc,ls} <> lt) == Model.position(Model.Coordinate{_,rc,rs} <> rt) : Nat}
          %Equal.sym(Nat,ls,rs,ss) : {Model.position(Model.Coordinate{le,lc,ls} <> lt) == Model.position(Model.Coordinate{re,rc,_} <> rt) : Nat} position)))

# A candidate with the initial layout and the requested position is the
# actual cursor state, including layouts with omitted extent-one axes.
def reached(+count: Nat,+initial: List<&2,Model.Coordinate>,+candidate: List<&2,Model.Coordinate>,
  +initial_valid: Cursor.Valid(initial),candidate_valid: Cursor.Valid(candidate),
  +same: {layout(initial) == layout(candidate) : List<&2,Model.Coordinate>},
  +place: {Model.position(candidate) == Nat.add(Model.position(initial),count) : Nat},
  room: {Nat.is_lt(Nat.add(Model.position(initial),count),Model.extent_product(initial)) == True{} : Bool}) ->
  {Order.iterate(count,initial) == candidate : List<&2,Model.Coordinate>}:
  unique(Order.iterate(count,initial),candidate,Order.iterate_valid(count,initial,initial_valid),candidate_valid,
    Equal.trans(List<&2,Model.Coordinate>,layout(Order.iterate(count,initial)),layout(initial),layout(candidate),iterated(count,initial),same),
    Equal.trans(Nat,Model.position(Order.iterate(count,initial)),Nat.add(Model.position(initial),count),Model.position(candidate),
      Order.iterate_position(count,initial,initial_valid,room),Equal.sym(Nat,Model.position(candidate),Nat.add(Model.position(initial),count),place)))
