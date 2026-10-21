# Addresses of views derived from one another axis by axis: a view stretched
# to more elements, sliced along one axis, or compacted where a source holds
# one element for an axis. Each view is read at explicit coordinates, one per
# axis, and its address is their dot product with the strides.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_row_major_coordinates.bend as Digits
import ./tensor_copy_values.bend as Copy
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_linear_layout.bend as Linear
import ./checked_word_completion.bend as Completion
import ./boolean_logic.bend as Boolean
import ./u32_full_range.bend as FullRange
import ./u32_division.bend as U32Division
import ./checked_word_refinement.bend as CheckedLaws
import ../machine_limits.bend as Machine
import ../tensor_band.bend as Bands

# The points of axes at coordinates, read forward.
def points(axes: List<&2,Views.Axis>,coordinates: List<&2,Nat>) -> List<&2,Digits.Point>:
  match axes coordinates:
    case Con{Views.Axis{extent,stride},tail} Con{coordinate,rest}: Digits.Point{extent,coordinate,stride,False{}} <> points(tail,rest)
    case axes coordinates: []

# One coordinate per axis, each inside its extent.
def Inside(axes: List<&2,Views.Axis>,coordinates: List<&2,Nat>) -> Data:
  match axes coordinates:
    case Nil{} Nil{}: Unit
    case Con{Views.Axis{extent,stride},tail} Con{coordinate,rest}:
      Shape.Both<{Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},Inside(tail,rest)>
    case axes coordinates: Empty

def axes_of(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,inside: Inside(axes,coordinates)) ->
  {Digits.axes(points(axes,coordinates)) == axes : List<&2,Views.Axis>}:
  match axes coordinates:
    case Nil{} Nil{}: {==}
    case Nil{} Con{coordinate,rest}: Empty.absurd({Digits.axes(points(Nil{},Con{coordinate,rest})) == Nil{} : List<&2,Views.Axis>},inside)
    case Con{axis,tail} Nil{}: Empty.absurd({Digits.axes(points(Con{axis,tail},Nil{})) == Con{axis,tail} : List<&2,Views.Axis>},inside)
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      Shape.Both{below,others} = inside
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,axes => Views.Axis{extent,stride} <> axes,Digits.axes(points(tail,rest)),tail,axes_of(tail,rest,others))

def valid(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,inside: Inside(axes,coordinates)) -> Digits.Valid(points(axes,coordinates)):
  match axes coordinates:
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{coordinate,rest}: Empty.absurd(Digits.Valid(points(Nil{},Con{coordinate,rest})),inside)
    case Con{axis,tail} Nil{}: Empty.absurd(Digits.Valid(points(Con{axis,tail},Nil{})),inside)
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      Shape.Both{below,others} = inside
      Shape.Both{below,valid(tail,rest,others)}

# The row-major position of coordinates over extents.
def position(extents: List<&2,U32>,coordinates: List<&2,Nat>,prefix: Nat) -> Nat:
  match extents coordinates:
    case Con{extent,tail} Con{coordinate,rest}: position(tail,rest,Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate))
    case extents coordinates: prefix

def positioned(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+prefix: Nat) ->
  {Digits.position(points(axes,coordinates),prefix) == position(Views.axis_extents(axes),coordinates,prefix) : Nat}:
  match axes coordinates:
    case Nil{} Nil{}: {==}
    case Nil{} Con{coordinate,rest}: {==}
    case Con{Views.Axis{extent,stride},tail} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      positioned(tail,rest,Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate))

# The address of a view at the position of inside coordinates is the dot
# product of the coordinates with the strides, after the offset.
def address(+axes: List<&2,Views.Axis>,+offset: U32,+coordinates: List<&2,Nat>,+inside: Inside(axes,coordinates)) ->
  {Copy.address(Views.View{axes,offset},position(Views.axis_extents(axes),coordinates,0n)) ==
    U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})) : U32}:
  +shaped = axes_of(axes,coordinates,inside)
  +valid_points = valid(axes,coordinates,inside)
  %positioned(axes,coordinates,0n) :
    {Copy.address(Views.View{axes,offset},_) == U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})) : U32}
  %shaped : {Copy.address(Views.View{_,offset},Digits.position(points(axes,coordinates),0n)) == U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})) : U32}
  Digits.copied(points(axes,coordinates),offset,valid_points,Digits.inside(points(axes,coordinates),valid_points))

# That position is inside the elements of the extents.
def position_inside(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+inside: Inside(axes,coordinates)) ->
  {Nat.is_lt(position(Views.axis_extents(axes),coordinates,0n),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}:
  +valid_points = valid(axes,coordinates,inside)
  %positioned(axes,coordinates,0n) : {Nat.is_lt(_,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}
  %axes_of(axes,coordinates,inside) : {Nat.is_lt(Digits.position(points(axes,coordinates),0n),ShapeLaws.elements(Views.axis_extents(_))) == True{} : Bool}
  Digits.inside(points(axes,coordinates),valid_points)

# A dot product started from a larger prefix is larger by the difference.
def shifted_prefix(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+prefix: Nat,+amount: Nat) ->
  {Digits.address(points(axes,coordinates),Nat.add(prefix,amount),False{}) == Nat.add(Digits.address(points(axes,coordinates),prefix,False{}),amount) : Nat}:
  match axes coordinates:
    case Nil{} Nil{}: {==}
    case Nil{} Con{coordinate,rest}: {==}
    case Con{Views.Axis{extent,stride},tail} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      +term = Nat.mul(coordinate,U32.to_nat(stride))
      %Addition.associative(term,prefix,amount) :
        {Digits.address(points(tail,rest),_,False{}) == Nat.add(Digits.address(points(tail,rest),Nat.add(term,prefix),False{}),amount) : Nat}
      shifted_prefix(tail,rest,Nat.add(term,prefix),amount)

# The coordinates a view reads through axes of these extents: an axis of one
# element reads coordinate zero.
def masked(extents: List<&2,U32>,coordinates: List<&2,Nat>) -> List<&2,Nat>:
  match extents coordinates:
    case Con{extent,tail} Con{coordinate,rest}: Bool.pick(Nat,U32.is_eq(extent,1),0n,coordinate) <> masked(tail,rest)
    case extents coordinates: []

def one_of(+value: U32,classified: {U32.is_eq(value,1) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}:
  %Equal.sym(U32,value,1,Arithmetic.equal_of_is_eq(value,1,classified)) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
  {==}

# A coordinate below an extent of one element is zero.
def below_one(+coordinate: Nat,+extent: U32,classified: {U32.is_eq(extent,1) == True{} : Bool},below: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  {coordinate == 0n : Nat}:
  Digits.zero_below_one(coordinate,
    %Arithmetic.equal_of_is_eq(extent,1,classified) : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below)

# How one axis of a stretched view comes from the axis of the view.
type Lane<-size: U32,-stride: U32,-extent: U32,-step: U32> is Data:
  Kept{same: {size == extent : U32},kept: {step == stride : U32}}
  Repeated{single: {size == 1 : U32},zero: {step == 0 : U32}}

def Stretched(axes: List<&2,Views.Axis>,target: List<&2,U32>,out: List<&2,Views.Axis>) -> Data:
  match axes target out:
    case Nil{} Nil{} Nil{}: Unit
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Con{Views.Axis{result,step},others}:
      Shape.Both<Shape.Both<{result == extent : U32},Lane<size,stride,extent,step>>,Stretched(tail,rest,others)>
    case axes target out: Empty

def StretchResult(axes: List<&2,Views.Axis>,target: List<&2,U32>,result: Maybe<&2,List<&2,Views.Axis>>) -> Data:
  match result:
    case Some{out}: Stretched(axes,target,out)
    case None{}: Unit

def prepended_lanes(same: Bool,single: Bool,+size: U32,+stride: U32,+extent: U32,+tail: List<&2,Views.Axis>,+rest: List<&2,U32>,tail_result: Maybe<&2,List<&2,Views.Axis>>,
  same_value: {U32.is_eq(size,extent) == same : Bool},single_value: {U32.is_eq(size,1) == single : Bool},tail_lanes: StretchResult(tail,rest,tail_result)) ->
  StretchResult(Views.Axis{size,stride} <> tail,extent <> rest,Views.stretched_prepend(Views.stretched_axis(same,single,size,stride,extent),tail_result)):
  match same single tail_result:
    case True{} single Some{others}:
      +equal = Arithmetic.equal_of_is_eq(size,extent,same_value)
      Shape.Both{Shape.Both{equal,Kept{equal,{==}}},tail_lanes}
    case True{} single None{}: Unit{}
    case False{} True{} Some{others}: Shape.Both{Shape.Both{{==},Repeated{Arithmetic.equal_of_is_eq(size,1,single_value),{==}}},tail_lanes}
    case False{} True{} None{}: Unit{}
    case False{} False{} tail_result: Unit{}

def stretch_lanes(+axes: List<&2,Views.Axis>,+target: List<&2,U32>) -> StretchResult(axes,target,Views.stretch_axes(axes,target)):
  match axes target:
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{extent,rest}: Unit{}
    case Con{axis,tail} Nil{}: Unit{}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
      prepended_lanes(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent,tail,rest,Views.stretch_axes(tail,rest),{==},{==},stretch_lanes(tail,rest))

# The premise form: a stretch that succeeded.
def stretched(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,
  result: {Views.stretch_axes(axes,target) == Some{out} : Maybe<&2,List<&2,Views.Axis>>}) -> Stretched(axes,target,out):
  %result : StretchResult(axes,target,_)
  stretch_lanes(axes,target)

# One axis of a stretched view reads the coordinate of the view, or zero
# where the view repeats its one element.
def lane_term(single: Bool,+size: U32,+stride: U32,+extent: U32,+result: U32,+step: U32,+coordinate: Nat,
  classified: {U32.is_eq(size,1) == single : Bool},resulting: {result == extent : U32},lane: Lane<size,stride,extent,step>,
  below: {Nat.is_lt(coordinate,U32.to_nat(result)) == True{} : Bool}) ->
  {Nat.mul(coordinate,U32.to_nat(step)) == Nat.mul(Bool.pick(Nat,single,0n,coordinate),U32.to_nat(stride)) : Nat}:
  match single lane:
    case True{} Kept{+same,kept}:
      +narrow = {
        %Equal.trans(U32,result,extent,size,resulting,Equal.sym(U32,size,extent,same)) : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below
        : {Nat.is_lt(coordinate,U32.to_nat(size)) == True{} : Bool}}
      %Equal.sym(Nat,coordinate,0n,below_one(coordinate,size,classified,narrow)) : {Nat.mul(_,U32.to_nat(step)) == Nat.mul(0n,U32.to_nat(stride)) : Nat}
      {==}
    case True{} Repeated{one,zero}:
      %Equal.sym(U32,step,0,zero) : {Nat.mul(coordinate,U32.to_nat(_)) == Nat.mul(0n,U32.to_nat(stride)) : Nat}
      RowMajor.multiply_zero_right(coordinate)
    case False{} Kept{same,kept}:
      %Equal.sym(U32,step,stride,kept) : {Nat.mul(coordinate,U32.to_nat(_)) == Nat.mul(coordinate,U32.to_nat(stride)) : Nat}
      {==}
    case False{} Repeated{one,zero}:
      Empty.absurd({Nat.mul(coordinate,U32.to_nat(step)) == Nat.mul(coordinate,U32.to_nat(stride)) : Nat},
        Words.false_true(Equal.sym(Bool,True{},False{},
          %one : {U32.is_eq(_,1) == False{} : Bool} classified)))

def lane_bound(single: Bool,+size: U32,+stride: U32,+extent: U32,+result: U32,+step: U32,+coordinate: Nat,
  classified: {U32.is_eq(size,1) == single : Bool},resulting: {result == extent : U32},lane: Lane<size,stride,extent,step>,
  below: {Nat.is_lt(coordinate,U32.to_nat(result)) == True{} : Bool}) ->
  {Nat.is_lt(Bool.pick(Nat,single,0n,coordinate),U32.to_nat(size)) == True{} : Bool}:
  match single lane:
    case True{} lane: one_of(size,classified)
    case False{} Kept{same,kept}:
      %Equal.trans(U32,result,extent,size,resulting,Equal.sym(U32,size,extent,same)) : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below
    case False{} Repeated{one,zero}:
      Empty.absurd({Nat.is_lt(coordinate,U32.to_nat(size)) == True{} : Bool},
        Words.false_true(Equal.sym(Bool,True{},False{},
          %one : {U32.is_eq(_,1) == False{} : Bool} classified)))

# A stretched view at inside coordinates has the address of the view at the
# coordinates its axes read.
def stretched_address(+out: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+prefix: Nat,
  lanes: Stretched(axes,target,out),inside: Inside(out,coordinates)) ->
  {Digits.address(points(out,coordinates),prefix,False{}) == Digits.address(points(axes,masked(Views.axis_extents(axes),coordinates)),prefix,False{}) : Nat}:
  match out coordinates:
    case Nil{} Nil{}:
      match axes target:
        case Nil{} Nil{}: {==}
        case Nil{} Con{extent,rest}: Empty.absurd({Digits.address(points(Nil{},Nil{}),prefix,False{}) == Digits.address(points(Nil{},masked(Views.axis_extents(Nil{}),Nil{})),prefix,False{}) : Nat},lanes)
        case Con{Views.Axis{size,stride},tail} Nil{}: Empty.absurd({Digits.address(points(Nil{},Nil{}),prefix,False{}) == Digits.address(points(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{})),prefix,False{}) : Nat},lanes)
        case Con{Views.Axis{size,stride},tail} Con{extent,rest}: Empty.absurd({Digits.address(points(Nil{},Nil{}),prefix,False{}) == Digits.address(points(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{})),prefix,False{}) : Nat},lanes)
    case Nil{} Con{coordinate,more}: Empty.absurd({Digits.address(points(Nil{},Con{coordinate,more}),prefix,False{}) == Digits.address(points(axes,masked(Views.axis_extents(axes),Con{coordinate,more})),prefix,False{}) : Nat},inside)
    case Con{axis,others} Nil{}: Empty.absurd({Digits.address(points(Con{axis,others},Nil{}),prefix,False{}) == Digits.address(points(axes,masked(Views.axis_extents(axes),Nil{})),prefix,False{}) : Nat},inside)
    case Con{Views.Axis{+result,+step},+others} Con{+coordinate,+more}:
      match axes target:
        case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
          Shape.Both{Shape.Both{+resulting,+lane},tail_lanes} = lanes
          Shape.Both{+below,more_inside} = inside
          +pick = Bool.pick(Nat,U32.is_eq(size,1),0n,coordinate)
          %Equal.sym(Nat,Nat.mul(coordinate,U32.to_nat(step)),Nat.mul(pick,U32.to_nat(stride)),lane_term(U32.is_eq(size,1),size,stride,extent,result,step,coordinate,{==},resulting,lane,below)) :
            {Digits.address(points(others,more),Nat.add(_,prefix),False{}) ==
              Digits.address(points(tail,masked(Views.axis_extents(tail),more)),Nat.add(Nat.mul(pick,U32.to_nat(stride)),prefix),False{}) : Nat}
          stretched_address(others,more,tail,rest,Nat.add(Nat.mul(pick,U32.to_nat(stride)),prefix),tail_lanes,more_inside)
        case Nil{} Nil{}: Empty.absurd({Digits.address(points(Con{Views.Axis{result,step},others},Con{coordinate,more}),prefix,False{}) == Digits.address(points(Nil{},masked(Views.axis_extents(Nil{}),Con{coordinate,more})),prefix,False{}) : Nat},lanes)
        case Nil{} Con{extent,rest}: Empty.absurd({Digits.address(points(Con{Views.Axis{result,step},others},Con{coordinate,more}),prefix,False{}) == Digits.address(points(Nil{},masked(Views.axis_extents(Nil{}),Con{coordinate,more})),prefix,False{}) : Nat},lanes)
        case Con{Views.Axis{size,stride},tail} Nil{}: Empty.absurd({Digits.address(points(Con{Views.Axis{result,step},others},Con{coordinate,more}),prefix,False{}) == Digits.address(points(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Con{coordinate,more})),prefix,False{}) : Nat},lanes)

def stretched_inside(+out: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+axes: List<&2,Views.Axis>,+target: List<&2,U32>,
  lanes: Stretched(axes,target,out),inside: Inside(out,coordinates)) -> Inside(axes,masked(Views.axis_extents(axes),coordinates)):
  match out coordinates:
    case Nil{} Nil{}:
      match axes target:
        case Nil{} Nil{}: Unit{}
        case Nil{} Con{extent,rest}: Empty.absurd(Inside(Nil{},masked(Views.axis_extents(Nil{}),Nil{})),lanes)
        case Con{Views.Axis{size,stride},tail} Nil{}: Empty.absurd(Inside(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{})),lanes)
        case Con{Views.Axis{size,stride},tail} Con{extent,rest}: Empty.absurd(Inside(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{})),lanes)
    case Nil{} Con{coordinate,more}: Empty.absurd(Inside(axes,masked(Views.axis_extents(axes),Con{coordinate,more})),inside)
    case Con{axis,others} Nil{}: Empty.absurd(Inside(axes,masked(Views.axis_extents(axes),Nil{})),inside)
    case Con{Views.Axis{+result,+step},+others} Con{+coordinate,+more}:
      match axes target:
        case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
          Shape.Both{Shape.Both{resulting,lane},tail_lanes} = lanes
          Shape.Both{below,more_inside} = inside
          Shape.Both{lane_bound(U32.is_eq(size,1),size,stride,extent,result,step,coordinate,{==},resulting,lane,below),
            stretched_inside(others,more,tail,rest,tail_lanes,more_inside)}
        case Nil{} Nil{}: Empty.absurd(Inside(Nil{},masked(Views.axis_extents(Nil{}),Con{coordinate,more})),lanes)
        case Nil{} Con{extent,rest}: Empty.absurd(Inside(Nil{},masked(Views.axis_extents(Nil{}),Con{coordinate,more})),lanes)
        case Con{Views.Axis{size,stride},tail} Nil{}: Empty.absurd(Inside(Con{Views.Axis{size,stride},tail},masked(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Con{coordinate,more})),lanes)

# An axis of a view that its source holds once has one element, or does not
# move.
type Repeats<-extent: U32,-stride: U32> is Data:
  Narrow{small: {Nat.is_le(U32.to_nat(extent),1n) == True{} : Bool}}
  Still{zero: {stride == 0 : U32}}

def Spreads(single: Bool,extent: U32,stride: U32) -> Data:
  match single:
    case True{}: Repeats<extent,stride>
    case False{}: Unit

def Spreadable(source: List<&2,U32>,axes: List<&2,Views.Axis>) -> Data:
  match source axes:
    case Nil{} Nil{}: Unit
    case Con{single,rest} Con{Views.Axis{extent,stride},others}: Shape.Both<Spreads(U32.is_eq(single,1),extent,stride),Spreadable(rest,others)>
    case source axes: Empty

def kept_term(unit: Bool,+extent: U32,+stride: U32,+coordinate: Nat,classified: {U32.is_eq(extent,1) == unit : Bool},
  below: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.mul(coordinate,U32.to_nat(stride)) == Nat.mul(Bool.pick(Nat,unit,0n,coordinate),U32.to_nat(stride)) : Nat}:
  match unit:
    case True{}:
      %Equal.sym(Nat,coordinate,0n,below_one(coordinate,extent,classified,below)) : {Nat.mul(_,U32.to_nat(stride)) == Nat.mul(0n,U32.to_nat(stride)) : Nat}
      {==}
    case False{}: {==}

def kept_bound(unit: Bool,+extent: U32,+coordinate: Nat,classified: {U32.is_eq(extent,1) == unit : Bool},
  below: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_lt(Bool.pick(Nat,unit,0n,coordinate),U32.to_nat(extent)) == True{} : Bool}:
  match unit:
    case True{}: one_of(extent,classified)
    case False{}: below

# One axis of a compacted view: where the source holds one element the
# coordinate is zero, which reads the same address.
def compact_term(single: Bool,+extent: U32,+stride: U32,+coordinate: Nat,spreads: Spreads(single,extent,stride),
  below: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.mul(coordinate,U32.to_nat(stride)) ==
    Nat.mul(Bool.pick(Nat,U32.is_eq(Bool.pick(U32,single,1,extent),1),0n,coordinate),U32.to_nat(stride)) : Nat}:
  match single spreads:
    case True{} Narrow{small}:
      %Equal.sym(Nat,coordinate,0n,Digits.zero_below_one(coordinate,Bounds.nat_trans(coordinate,U32.to_nat(extent),1n,below,small)))
        : {Nat.mul(_,U32.to_nat(stride)) == Nat.mul(0n,U32.to_nat(stride)) : Nat}
      {==}
    case True{} Still{zero}:
      %Equal.sym(U32,stride,0,zero) : {Nat.mul(coordinate,U32.to_nat(_)) == Nat.mul(0n,U32.to_nat(_)) : Nat}
      RowMajor.multiply_zero_right(coordinate)
    case False{} spreads: kept_term(U32.is_eq(extent,1),extent,stride,coordinate,{==},below)

def compact_bound(single: Bool,+extent: U32,+coordinate: Nat,below: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_lt(Bool.pick(Nat,U32.is_eq(Bool.pick(U32,single,1,extent),1),0n,coordinate),U32.to_nat(Bool.pick(U32,single,1,extent))) == True{} : Bool}:
  match single:
    case True{}: {==}
    case False{}: kept_bound(U32.is_eq(extent,1),extent,coordinate,{==},below)

# A view at inside coordinates has the address of its compacted view at the
# coordinates the compacted axes read.
def compact_address(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+source: List<&2,U32>,+prefix: Nat,
  spread: Spreadable(source,axes),inside: Inside(axes,coordinates)) ->
  {Digits.address(points(axes,coordinates),prefix,False{}) ==
    Digits.address(points(Bands.compact_axes(source,axes),masked(Views.axis_extents(Bands.compact_axes(source,axes)),coordinates)),prefix,False{}) : Nat}:
  match axes coordinates:
    case Nil{} Nil{}:
      match source:
        case Nil{}: {==}
        case Con{single,rest}: {==}
    case Nil{} Con{coordinate,more}:
      Empty.absurd({Digits.address(points(Nil{},Con{coordinate,more}),prefix,False{}) ==
        Digits.address(points(Bands.compact_axes(source,Nil{}),masked(Views.axis_extents(Bands.compact_axes(source,Nil{})),Con{coordinate,more})),prefix,False{}) : Nat},inside)
    case Con{axis,others} Nil{}:
      Empty.absurd({Digits.address(points(Con{axis,others},Nil{}),prefix,False{}) ==
        Digits.address(points(Bands.compact_axes(source,Con{axis,others}),masked(Views.axis_extents(Bands.compact_axes(source,Con{axis,others})),Nil{})),prefix,False{}) : Nat},inside)
    case Con{Views.Axis{+extent,+stride},+others} Con{+coordinate,+more}:
      match source:
        case Con{+single,+rest}:
          Shape.Both{spreads,rest_spread} = spread
          Shape.Both{+below,more_inside} = inside
          +term = Nat.mul(coordinate,U32.to_nat(stride))
          %compact_term(U32.is_eq(single,1),extent,stride,coordinate,spreads,below) :
            {Digits.address(points(others,more),Nat.add(term,prefix),False{}) ==
              Digits.address(points(Bands.compact_axes(rest,others),masked(Views.axis_extents(Bands.compact_axes(rest,others)),more)),Nat.add(_,prefix),False{}) : Nat}
          compact_address(others,more,rest,Nat.add(term,prefix),rest_spread,more_inside)
        case Nil{}:
          Empty.absurd({Digits.address(points(Con{Views.Axis{extent,stride},others},Con{coordinate,more}),prefix,False{}) ==
            Digits.address(points(Bands.compact_axes(Nil{},Con{Views.Axis{extent,stride},others}),masked(Views.axis_extents(Bands.compact_axes(Nil{},Con{Views.Axis{extent,stride},others})),Con{coordinate,more})),prefix,False{}) : Nat},spread)

def compact_inside(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+source: List<&2,U32>,
  spread: Spreadable(source,axes),inside: Inside(axes,coordinates)) ->
  Inside(Bands.compact_axes(source,axes),masked(Views.axis_extents(Bands.compact_axes(source,axes)),coordinates)):
  match axes coordinates:
    case Nil{} Nil{}:
      match source:
        case Nil{}: Unit{}
        case Con{single,rest}: Unit{}
    case Nil{} Con{coordinate,more}:
      Empty.absurd(Inside(Bands.compact_axes(source,Nil{}),masked(Views.axis_extents(Bands.compact_axes(source,Nil{})),Con{coordinate,more})),inside)
    case Con{axis,others} Nil{}:
      Empty.absurd(Inside(Bands.compact_axes(source,Con{axis,others}),masked(Views.axis_extents(Bands.compact_axes(source,Con{axis,others})),Nil{})),inside)
    case Con{Views.Axis{+extent,+stride},+others} Con{+coordinate,+more}:
      match source:
        case Con{+single,+rest}:
          Shape.Both{spreads,rest_spread} = spread
          Shape.Both{below,more_inside} = inside
          Shape.Both{compact_bound(U32.is_eq(single,1),extent,coordinate,below),compact_inside(others,more,rest,rest_spread,more_inside)}
        case Nil{}:
          Empty.absurd(Inside(Bands.compact_axes(Nil{},Con{Views.Axis{extent,stride},others}),
            masked(Views.axis_extents(Bands.compact_axes(Nil{},Con{Views.Axis{extent,stride},others})),Con{coordinate,more})),spread)

# Coordinates with start added to the coordinate of one axis.
def shifted(axis: Nat,start: U32,coordinates: List<&2,Nat>) -> List<&2,Nat>:
  match axis coordinates:
    case 0n Con{coordinate,rest}: Nat.add(coordinate,U32.to_nat(start)) <> rest
    case 1n+axis Con{coordinate,rest}: coordinate <> shifted(axis,start,rest)
    case axis coordinates: []

# Extents with the extent of one axis replaced by count.
def replaced(extents: List<&2,U32>,axis: Nat,count: U32) -> List<&2,U32>:
  match extents axis:
    case Con{extent,tail} 0n: count <> tail
    case Con{extent,tail} 1n+axis: extent <> replaced(tail,axis,count)
    case extents axis: []

# How a view sliced along one axis with step one comes from the view.
def Sliced(axes: List<&2,Views.Axis>,axis: Nat,start: U32,count: U32,offset: U32,out: List<&2,Views.Axis>,out_offset: U32) -> Data:
  match axes axis out:
    case Con{Views.Axis{size,stride},tail} 0n Con{Views.Axis{extent,step},others}:
      Shape.Both<Shape.Both<{extent == count : U32},{step == stride : U32}>,
        Shape.Both<{others == tail : List<&2,Views.Axis>},
          Shape.Both<{Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool},
            {U32.to_nat(out_offset) == Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(start),U32.to_nat(stride))) : Nat}>>>
    case Con{Views.Axis{size,stride},tail} 1n+axis Con{Views.Axis{extent,step},others}:
      Shape.Both<Shape.Both<{extent == size : U32},{step == stride : U32}>,Sliced(tail,axis,start,count,offset,others,out_offset)>
    case axes axis out: Empty

def SliceResult(axes: List<&2,Views.Axis>,axis: Nat,start: U32,count: U32,offset: U32,result: Maybe<&2,Views.View>) -> Data:
  match result:
    case Some{Views.View{out,out_offset}}: Sliced(axes,axis,start,count,offset,out,out_offset)
    case None{}: Unit

# A slice step of one from start that passed its check ends inside the axis.
def slice_room_when(zero: Bool,+start: U32,+count: U32,+size: U32,+classified: {U32.is_zero(count) == zero : Bool},
  checked: {Views.slice_step(False{},start,count,1,size) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool}:
  match zero:
    case True{}:
      +first = Boolean.left(U32.is_le(start,size),U32.is_zero(count) || (U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32))),checked)
      %Equal.sym(U32,count,0,Arithmetic.word_zero(count,classified)) : {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(_)),U32.to_nat(size)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),Addition.zero_right(U32.to_nat(start))) : {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool}
      Bounds.as_nat_le(start,size,first)
    case False{}:
      +rest = {
        %classified : {Bool.or(_,U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32))) == True{} : Bool}
        Boolean.right(U32.is_le(start,size),U32.is_zero(count) || (U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32))),checked)
        : {U32.is_lt(start,size) && U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)) == True{} : Bool}}
      +below = Bounds.as_nat_lt(start,size,Boolean.left(U32.is_lt(start,size),U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)),rest))
      +last = Boolean.right(U32.is_lt(start,size),U32.is_le((count - 1 : U32),((size - 1 - start) / 1 : U32)),rest)
      +positive = Arithmetic.word_nonzero_positive(count,classified)
      +size_positive = Order.le_lt_trans(0n,U32.to_nat(start),U32.to_nat(size),Order.zero_le(U32.to_nat(start)),below)
      +before = Order.before_last(U32.to_nat(start),U32.to_nat(size),below)
      +size_last = FullRange.subtract(size,1,size,Order.positive_one(U32.to_nat(size),size_positive),Order.sub_le(U32.to_nat(size),1n))
      +span = FullRange.subtract((size - 1 : U32),start,size,
        %Equal.sym(Nat,U32.to_nat((size - 1 : U32)),Nat.sub(U32.to_nat(size),1n),size_last) : {Nat.is_le(U32.to_nat(start),_) == True{} : Bool} before,
        Order.transitive(Nat.sub(U32.to_nat((size - 1 : U32)),U32.to_nat(start)),U32.to_nat((size - 1 : U32)),U32.to_nat(size),Order.sub_le(U32.to_nat((size - 1 : U32)),U32.to_nat(start)),
          %Equal.sym(Nat,U32.to_nat((size - 1 : U32)),Nat.sub(U32.to_nat(size),1n),size_last) : {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool} Order.sub_le(U32.to_nat(size),1n)))
      +count_last = FullRange.subtract(count,1,count,Order.positive_one(U32.to_nat(count),positive),Order.sub_le(U32.to_nat(count),1n))
      +room = {
        %count_last : {Nat.is_le(_,Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start))) == True{} : Bool}
        %size_last : {Nat.is_le(U32.to_nat((count - 1 : U32)),Nat.sub(_,U32.to_nat(start))) == True{} : Bool}
        %span : {Nat.is_le(U32.to_nat((count - 1 : U32)),_) == True{} : Bool}
        %U32Division.divided_by_one((size - 1 - start : U32)) : {Nat.is_le(U32.to_nat((count - 1 : U32)),_) == True{} : Bool}
        Bounds.as_nat_le((count - 1 : U32),((size - 1 - start) / 1 : U32),last)
        : {Nat.is_le(Nat.sub(U32.to_nat(count),1n),Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start))) == True{} : Bool}}
      +shift = Order.restore_shift(U32.to_nat(start),Nat.sub(U32.to_nat(size),1n),before)
      %Order.positive_has_last(U32.to_nat(count),positive) :
        {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(size)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(U32.to_nat(start),1n+Nat.sub(U32.to_nat(count),1n)),1n+Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(count),1n)),
        Addition.successor_right(U32.to_nat(start),Nat.sub(U32.to_nat(count),1n))) : {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool}
      %Order.positive_has_last(U32.to_nat(size),size_positive) :
        {Nat.is_le(1n+Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(count),1n)),_) == True{} : Bool}
      %shift : {Nat.is_le(1n+Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(count),1n)),1n+_) == True{} : Bool}
      Order.offset_le(U32.to_nat(start),Nat.sub(U32.to_nat(count),1n),Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(start)),room)

def word_or_zero(result: Maybe<&2,U32>) -> U32:
  match result:
    case Some{word}: word
    case None{}: 0

# An offset that passed the check of add_offset is the sum of the two.
def offset_sum(fits: Bool,+offset: U32,+shift: U32,+out: U32,+limit: U32,maximum: {limit == Machine.word_max() : U32},
  classified: {U32.is_le(shift,(limit - offset : U32)) == fits : Bool},result: {Views.address_when(fits,offset,shift) == Some{out} : Maybe<&2,U32>}) ->
  {U32.to_nat(out) == Nat.add(U32.to_nat(offset),U32.to_nat(shift)) : Nat}:
  match fits:
    case True{}:
      +within = {
        %Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(offset,_) == True{} : Bool} FullRange.maximum(offset)
        : {U32.is_le(offset,limit) == True{} : Bool}}
      +ordered = Bounds.as_nat_le(offset,limit,within)
      +rest = FullRange.subtract(limit,offset,limit,ordered,Order.sub_le(U32.to_nat(limit),U32.to_nat(offset)))
      +small = {
        %rest : {Nat.is_le(U32.to_nat(shift),_) == True{} : Bool} Bounds.as_nat_le(shift,(limit - offset : U32),classified)
        : {Nat.is_le(U32.to_nat(shift),Nat.sub(U32.to_nat(limit),U32.to_nat(offset))) == True{} : Bool}}
      +inside = {
        %Order.restore_shift(U32.to_nat(offset),U32.to_nat(limit),ordered) : {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(shift)),_) == True{} : Bool}
        Order.offset_le(U32.to_nat(offset),U32.to_nat(shift),Nat.sub(U32.to_nat(limit),U32.to_nat(offset)),small)
        : {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(shift)),U32.to_nat(limit)) == True{} : Bool}}
      %Equal.cong(Maybe<&2,U32>,U32,word_or_zero,Some{(offset + shift : U32)},Some{out},result) :
        {U32.to_nat(_) == Nat.add(U32.to_nat(offset),U32.to_nat(shift)) : Nat}
      FullRange.add(offset,shift,limit,inside)
    case False{}:
      Empty.absurd({U32.to_nat(out) == Nat.add(U32.to_nat(offset),U32.to_nat(shift)) : Nat},Words.false_true(Equal.cong(Maybe<&2,U32>,Bool,value => Maybe.is_some(&2,U32,value),None{},Some{out},result)))

def offset_lanes(fits: Bool,+size: U32,+stride: U32,+tail: List<&2,Views.Axis>,+start: U32,+count: U32,+offset: U32,+step: U32,+product: U32,
  room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool},
  stepped: {Nat.mul(U32.to_nat(stride),U32.to_nat(1)) == U32.to_nat(step) : Nat},
  produced: {Nat.mul(U32.to_nat(start),U32.to_nat(stride)) == U32.to_nat(product) : Nat},
  classified: {U32.is_le(product,(Machine.word_max() - offset : U32)) == fits : Bool}) ->
  SliceResult(Views.Axis{size,stride} <> tail,0n,start,count,offset,Views.sliced_axis(True{},count,Some{step},Views.address_when(fits,offset,product),tail)):
  match fits:
    case True{}:
      +step_value = Arithmetic.to_nat_injective(step,stride,
        Equal.trans(Nat,U32.to_nat(step),Nat.mul(U32.to_nat(stride),1n),U32.to_nat(stride),Equal.sym(Nat,Nat.mul(U32.to_nat(stride),1n),U32.to_nat(step),stepped),Order.times_one(U32.to_nat(stride))))
      Shape.Both{Shape.Both{{==},step_value},Shape.Both{{==},Shape.Both{room,
        %Equal.sym(Nat,Nat.mul(U32.to_nat(start),U32.to_nat(stride)),U32.to_nat(product),produced) : {U32.to_nat((offset + product : U32)) == Nat.add(U32.to_nat(offset),_) : Nat}
        offset_sum(True{},offset,product,(offset + product : U32),Machine.word_max(),{==},classified,{==})}}}
    case False{}: Unit{}

def first_lanes(valid: Bool,step: Maybe<&2,U32>,product: Maybe<&2,U32>,+size: U32,+stride: U32,+tail: List<&2,Views.Axis>,+start: U32,+count: U32,+offset: U32,
  checked: {Views.slice_step(False{},start,count,1,size) == valid : Bool},
  stepped: {Shapes.checked_multiply(stride,1) == step : Maybe<&2,U32>},produced: {Shapes.checked_multiply(start,stride) == product : Maybe<&2,U32>}) ->
  SliceResult(Views.Axis{size,stride} <> tail,0n,start,count,offset,Views.sliced_axis(valid,count,step,Views.add_offset(product,offset),tail)):
  match valid step product:
    case True{} Some{+word} Some{+shift}:
      offset_lanes(U32.is_le(shift,(Machine.word_max() - offset : U32)),size,stride,tail,start,count,offset,word,shift,
        slice_room_when(U32.is_zero(count),start,count,size,{==},checked),
        CheckedLaws.multiply_sound(stride,1,word,stepped),CheckedLaws.multiply_sound(start,stride,shift,produced),{==})
    case True{} Some{word} None{}: Unit{}
    case True{} None{} product: Unit{}
    case False{} step product: Unit{}

def prepended_slice(inner: Maybe<&2,Views.View>,+size: U32,+stride: U32,+tail: List<&2,Views.Axis>,+rest: Nat,+start: U32,+count: U32,+offset: U32,
  inner_lanes: SliceResult(tail,rest,start,count,offset,inner)) ->
  SliceResult(Views.Axis{size,stride} <> tail,1n+rest,start,count,offset,Views.prepend_axis(Some{Views.Axis{size,stride}},inner)):
  match inner:
    case Some{Views.View{others,out_offset}}: Shape.Both{Shape.Both{{==},{==}},inner_lanes}
    case None{}: Unit{}

def slice_lanes(+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32) ->
  SliceResult(axes,axis,start,count,offset,Views.slice_axes(axes,axis,start,count,1,offset)):
  match axes axis:
    case Nil{} axis: Unit{}
    case Con{Views.Axis{+size,+stride},+tail} 0n:
      first_lanes(Views.slice_step(False{},start,count,1,size),Shapes.checked_multiply(stride,1),Shapes.checked_multiply(start,stride),size,stride,tail,start,count,offset,{==},{==},{==})
    case Con{Views.Axis{+size,+stride},+tail} 1n+ +rest:
      prepended_slice(Views.slice_axes(tail,rest,start,count,1,offset),size,stride,tail,rest,start,count,offset,slice_lanes(tail,rest,start,count,offset))

# The premise form: a slice that succeeded.
def sliced(+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32,+out: List<&2,Views.Axis>,+out_offset: U32,
  result: {Views.slice(Views.View{axes,offset},axis,start,count,1) == Some{Views.View{out,out_offset}} : Maybe<&2,Views.View>}) ->
  Sliced(axes,axis,start,count,offset,out,out_offset):
  %result : SliceResult(axes,axis,start,count,offset,_)
  slice_lanes(axes,axis,start,count,offset)

def sliced_term(+coordinate: Nat,+start: U32,+stride: U32,+offset: U32,+out_offset: U32,
  placed: {U32.to_nat(out_offset) == Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(start),U32.to_nat(stride))) : Nat}) ->
  {Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),U32.to_nat(out_offset)) == Nat.add(Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),U32.to_nat(offset)) : Nat}:
  +a = Nat.mul(coordinate,U32.to_nat(stride))
  +b = Nat.mul(U32.to_nat(start),U32.to_nat(stride))
  +o = U32.to_nat(offset)
  %Equal.sym(Nat,U32.to_nat(out_offset),Nat.add(o,b),placed) : {Nat.add(a,_) == Nat.add(Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),o) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),Nat.add(a,b),RowMajor.distrib(coordinate,U32.to_nat(start),U32.to_nat(stride))) : {Nat.add(a,Nat.add(o,b)) == Nat.add(_,o) : Nat}
  %Order.commutative(b,o) : {Nat.add(a,_) == Nat.add(Nat.add(a,b),o) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(a,b),o),Nat.add(a,Nat.add(b,o)),Addition.associative(a,b,o))

# A sliced view at inside coordinates has the address of the view at the
# coordinates moved by start along the axis.
def sliced_address(+out: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32,+out_offset: U32,
  inside: Inside(out,coordinates),lanes: Sliced(axes,axis,start,count,offset,out,out_offset)) ->
  {Digits.address(points(out,coordinates),U32.to_nat(out_offset),False{}) == Digits.address(points(axes,shifted(axis,start,coordinates)),U32.to_nat(offset),False{}) : Nat}:
  match out coordinates:
    case Con{Views.Axis{+extent,+step},+others} Con{+coordinate,+more}:
      match axes axis:
        case Con{Views.Axis{+size,+stride},+tail} 0n:
          Shape.Both{Shape.Both{counted,+stepped},Shape.Both{+rest_same,Shape.Both{room,+placed}}} = lanes
          %Equal.sym(U32,step,stride,stepped) : {Digits.address(points(others,more),Nat.add(Nat.mul(coordinate,U32.to_nat(_)),U32.to_nat(out_offset)),False{}) ==
            Digits.address(points(tail,more),Nat.add(Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),U32.to_nat(offset)),False{}) : Nat}
          %Equal.sym(List<&2,Views.Axis>,others,tail,rest_same) : {Digits.address(points(_,more),Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),U32.to_nat(out_offset)),False{}) ==
            Digits.address(points(tail,more),Nat.add(Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),U32.to_nat(offset)),False{}) : Nat}
          Equal.cong(Nat,Nat,prefix => Digits.address(points(tail,more),prefix,False{}),
            Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),U32.to_nat(out_offset)),Nat.add(Nat.mul(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(stride)),U32.to_nat(offset)),
            sliced_term(coordinate,start,stride,offset,out_offset,placed))
        case Con{Views.Axis{+size,+stride},+tail} 1n+ +rest:
          Shape.Both{below,more_inside} = inside
          Shape.Both{Shape.Both{counted,+stepped},inner} = lanes
          +term = Nat.mul(coordinate,U32.to_nat(stride))
          +moved = shifted(rest,start,more)
          %Equal.sym(U32,step,stride,stepped) : {Digits.address(points(others,more),Nat.add(Nat.mul(coordinate,U32.to_nat(_)),U32.to_nat(out_offset)),False{}) ==
            Digits.address(points(tail,moved),Nat.add(term,U32.to_nat(offset)),False{}) : Nat}
          %Order.commutative(U32.to_nat(out_offset),term) : {Digits.address(points(others,more),_,False{}) == Digits.address(points(tail,moved),Nat.add(term,U32.to_nat(offset)),False{}) : Nat}
          %Order.commutative(U32.to_nat(offset),term) : {Digits.address(points(others,more),Nat.add(U32.to_nat(out_offset),term),False{}) == Digits.address(points(tail,moved),_,False{}) : Nat}
          %Equal.sym(Nat,Digits.address(points(others,more),Nat.add(U32.to_nat(out_offset),term),False{}),Nat.add(Digits.address(points(others,more),U32.to_nat(out_offset),False{}),term),
            shifted_prefix(others,more,U32.to_nat(out_offset),term)) : {_ == Digits.address(points(tail,moved),Nat.add(U32.to_nat(offset),term),False{}) : Nat}
          %Equal.sym(Nat,Digits.address(points(tail,moved),Nat.add(U32.to_nat(offset),term),False{}),Nat.add(Digits.address(points(tail,moved),U32.to_nat(offset),False{}),term),
            shifted_prefix(tail,moved,U32.to_nat(offset),term)) : {Nat.add(Digits.address(points(others,more),U32.to_nat(out_offset),False{}),term) == _ : Nat}
          Equal.cong(Nat,Nat,value => Nat.add(value,term),Digits.address(points(others,more),U32.to_nat(out_offset),False{}),Digits.address(points(tail,moved),U32.to_nat(offset),False{}),
            sliced_address(others,more,tail,rest,start,count,offset,out_offset,more_inside,inner))
        case Nil{} axis:
          Empty.absurd({Digits.address(points(Con{Views.Axis{extent,step},others},Con{coordinate,more}),U32.to_nat(out_offset),False{}) == Digits.address(points(Nil{},shifted(axis,start,Con{coordinate,more})),U32.to_nat(offset),False{}) : Nat},lanes)
    case Nil{} Nil{}:
      match axes axis:
        case Nil{} axis:
          Empty.absurd({Digits.address(points(Nil{},Nil{}),U32.to_nat(out_offset),False{}) == Digits.address(points(Nil{},shifted(axis,start,Nil{})),U32.to_nat(offset),False{}) : Nat},lanes)
        case Con{Views.Axis{size,stride},tail} 0n:
          Empty.absurd({Digits.address(points(Nil{},Nil{}),U32.to_nat(out_offset),False{}) == Digits.address(points(Con{Views.Axis{size,stride},tail},shifted(0n,start,Nil{})),U32.to_nat(offset),False{}) : Nat},lanes)
        case Con{Views.Axis{size,stride},tail} 1n+rest:
          Empty.absurd({Digits.address(points(Nil{},Nil{}),U32.to_nat(out_offset),False{}) == Digits.address(points(Con{Views.Axis{size,stride},tail},shifted(1n+rest,start,Nil{})),U32.to_nat(offset),False{}) : Nat},lanes)
    case Nil{} Con{coordinate,more}:
      Empty.absurd({Digits.address(points(Nil{},Con{coordinate,more}),U32.to_nat(out_offset),False{}) == Digits.address(points(axes,shifted(axis,start,Con{coordinate,more})),U32.to_nat(offset),False{}) : Nat},inside)
    case Con{axis_out,others} Nil{}:
      Empty.absurd({Digits.address(points(Con{axis_out,others},Nil{}),U32.to_nat(out_offset),False{}) == Digits.address(points(axes,shifted(axis,start,Nil{})),U32.to_nat(offset),False{}) : Nat},inside)

def sliced_below(+coordinate: Nat,+start: U32,+count: U32,+size: U32,below: {Nat.is_lt(coordinate,U32.to_nat(count)) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(coordinate,U32.to_nat(start)),U32.to_nat(size)) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(coordinate,U32.to_nat(start)),Nat.add(U32.to_nat(count),U32.to_nat(start)),U32.to_nat(size),Order.add_lt(coordinate,U32.to_nat(count),U32.to_nat(start),below),
    %Order.commutative(U32.to_nat(start),U32.to_nat(count)) : {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool} room)

def sliced_inside(+out: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32,+out_offset: U32,
  inside: Inside(out,coordinates),lanes: Sliced(axes,axis,start,count,offset,out,out_offset)) -> Inside(axes,shifted(axis,start,coordinates)):
  match out coordinates:
    case Con{Views.Axis{+extent,+step},+others} Con{+coordinate,+more}:
      match axes axis:
        case Con{Views.Axis{+size,+stride},+tail} 0n:
          Shape.Both{below,more_inside} = inside
          Shape.Both{Shape.Both{+counted,stepped},Shape.Both{+rest_same,Shape.Both{+room,placed}}} = lanes
          Shape.Both{sliced_below(coordinate,start,count,size,%counted : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below,room),
            %rest_same : Inside(_,more) more_inside}
        case Con{Views.Axis{+size,+stride},+tail} 1n+ +rest:
          Shape.Both{below,more_inside} = inside
          Shape.Both{Shape.Both{+counted,stepped},inner} = lanes
          Shape.Both{%counted : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below,
            sliced_inside(others,more,tail,rest,start,count,offset,out_offset,more_inside,inner)}
        case Nil{} axis: Empty.absurd(Inside(Nil{},shifted(axis,start,Con{coordinate,more})),lanes)
    case Nil{} Nil{}:
      match axes axis:
        case Nil{} axis: Empty.absurd(Inside(Nil{},shifted(axis,start,Nil{})),lanes)
        case Con{Views.Axis{size,stride},tail} 0n: Empty.absurd(Inside(Con{Views.Axis{size,stride},tail},shifted(0n,start,Nil{})),lanes)
        case Con{Views.Axis{size,stride},tail} 1n+rest: Empty.absurd(Inside(Con{Views.Axis{size,stride},tail},shifted(1n+rest,start,Nil{})),lanes)
    case Nil{} Con{coordinate,more}: Empty.absurd(Inside(axes,shifted(axis,start,Con{coordinate,more})),inside)
    case Con{axis_out,others} Nil{}: Empty.absurd(Inside(axes,shifted(axis,start,Nil{})),inside)

def spreads_sliced(single: Bool,+size: U32,+stride: U32,+start: U32,+count: U32,
  room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size)) == True{} : Bool},spreads: Spreads(single,size,stride)) -> Spreads(single,count,stride):
  match single spreads:
    case True{} Narrow{small}:
      +fewer = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(size),
        %Order.commutative(U32.to_nat(count),U32.to_nat(start)) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} Order.le_add(U32.to_nat(count),U32.to_nat(start)),room)
      Narrow{Order.transitive(U32.to_nat(count),U32.to_nat(size),1n,fewer,small)}
    case True{} Still{zero}: Still{zero}
    case False{} spreads: Unit{}

# The axes a source holds once stay so in a slice of the view.
def sliced_spreadable(+out: List<&2,Views.Axis>,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32,+out_offset: U32,
  spread: Spreadable(source,axes),lanes: Sliced(axes,axis,start,count,offset,out,out_offset)) -> Spreadable(source,out):
  match out source:
    case Con{Views.Axis{+extent,+step},+others} Con{+single,+rest}:
      match axes axis:
        case Con{Views.Axis{+size,+stride},+tail} 0n:
          Shape.Both{spreads,rest_spread} = spread
          Shape.Both{Shape.Both{+counted,+stepped},Shape.Both{+rest_same,Shape.Both{+room,placed}}} = lanes
          Shape.Both{%Equal.sym(U32,extent,count,counted) : Spreads(U32.is_eq(single,1),_,step) %Equal.sym(U32,step,stride,stepped) : Spreads(U32.is_eq(single,1),count,_)
              spreads_sliced(U32.is_eq(single,1),size,stride,start,count,room,spreads),
            %Equal.sym(List<&2,Views.Axis>,others,tail,rest_same) : Spreadable(rest,_) rest_spread}
        case Con{Views.Axis{+size,+stride},+tail} 1n+ +inner_axis:
          Shape.Both{spreads,rest_spread} = spread
          Shape.Both{Shape.Both{+counted,+stepped},inner} = lanes
          Shape.Both{%Equal.sym(U32,extent,size,counted) : Spreads(U32.is_eq(single,1),_,step) %Equal.sym(U32,step,stride,stepped) : Spreads(U32.is_eq(single,1),size,_) spreads,
            sliced_spreadable(others,rest,tail,inner_axis,start,count,offset,out_offset,rest_spread,inner)}
        case Nil{} axis: Empty.absurd(Spreadable(Con{single,rest},Con{Views.Axis{extent,step},others}),lanes)
    case Nil{} Nil{}:
      match axes axis:
        case Nil{} axis: Unit{}
        case Con{Views.Axis{size,stride},tail} 0n: Empty.absurd(Spreadable(Nil{},Nil{}),lanes)
        case Con{Views.Axis{size,stride},tail} 1n+rest: Empty.absurd(Spreadable(Nil{},Nil{}),lanes)
    case Nil{} Con{single,rest}:
      match axes axis:
        case Nil{} axis: Empty.absurd(Spreadable(Con{single,rest},Nil{}),spread)
        case Con{Views.Axis{size,stride},tail} 0n: Empty.absurd(Spreadable(Con{single,rest},Nil{}),lanes)
        case Con{Views.Axis{size,stride},tail} 1n+inner_axis: Empty.absurd(Spreadable(Con{single,rest},Nil{}),lanes)
    case Con{Views.Axis{extent,step},others} Nil{}:
      match axes axis:
        case Nil{} axis: Empty.absurd(Spreadable(Nil{},Con{Views.Axis{extent,step},others}),lanes)
        case Con{Views.Axis{size,stride},tail} axis: Empty.absurd(Spreadable(Nil{},Con{Views.Axis{extent,step},others}),spread)

# The extents of a slice replace the extent of its axis by its count.
def sliced_extents(+out: List<&2,Views.Axis>,+axes: List<&2,Views.Axis>,+axis: Nat,+start: U32,+count: U32,+offset: U32,+out_offset: U32,
  lanes: Sliced(axes,axis,start,count,offset,out,out_offset)) -> {Views.axis_extents(out) == replaced(Views.axis_extents(axes),axis,count) : List<&2,U32>}:
  match out axes axis:
    case Con{Views.Axis{+extent,step},+others} Con{Views.Axis{+size,stride},+tail} 0n:
      Shape.Both{Shape.Both{+counted,stepped},Shape.Both{+rest_same,others_room}} = lanes
      %Equal.sym(U32,extent,count,counted) : {_ <> Views.axis_extents(others) == count <> Views.axis_extents(tail) : List<&2,U32>}
      %Equal.sym(List<&2,Views.Axis>,others,tail,rest_same) : {count <> Views.axis_extents(_) == count <> Views.axis_extents(tail) : List<&2,U32>}
      {==}
    case Con{Views.Axis{+extent,step},+others} Con{Views.Axis{+size,stride},+tail} 1n+ +rest:
      Shape.Both{Shape.Both{+counted,stepped},inner} = lanes
      %Equal.sym(U32,extent,size,counted) : {_ <> Views.axis_extents(others) == size <> replaced(Views.axis_extents(tail),rest,count) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => size <> extents,Views.axis_extents(others),replaced(Views.axis_extents(tail),rest,count),
        sliced_extents(others,tail,rest,start,count,offset,out_offset,inner))
    case Nil{} Nil{} axis: Empty.absurd({Views.axis_extents(Nil{}) == replaced(Views.axis_extents(Nil{}),axis,count) : List<&2,U32>},lanes)
    case Nil{} Con{Views.Axis{size,stride},tail} 0n:
      Empty.absurd({Views.axis_extents(Nil{}) == replaced(Views.axis_extents(Con{Views.Axis{size,stride},tail}),0n,count) : List<&2,U32>},lanes)
    case Nil{} Con{Views.Axis{size,stride},tail} 1n+rest:
      Empty.absurd({Views.axis_extents(Nil{}) == replaced(Views.axis_extents(Con{Views.Axis{size,stride},tail}),1n+rest,count) : List<&2,U32>},lanes)
    case Con{Views.Axis{extent,step},others} Nil{} axis:
      Empty.absurd({Views.axis_extents(Con{Views.Axis{extent,step},others}) == replaced(Views.axis_extents(Nil{}),axis,count) : List<&2,U32>},lanes)

def head_extent(extents: List<&2,U32>) -> U32:
  match extents:
    case Con{extent,tail}: extent
    case Nil{}: 0

def tail_extents(extents: List<&2,U32>) -> List<&2,U32>:
  match extents:
    case Con{extent,tail}: tail
    case Nil{}: []

def has_extents(extents: List<&2,U32>) -> Bool:
  match extents:
    case Con{extent,tail}: True{}
    case Nil{}: False{}

# The extents of a stretched view are the target.
def stretched_extents(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Stretched(axes,target,out)) ->
  {Views.axis_extents(out) == target : List<&2,U32>}:
  match axes target out:
    case Nil{} Nil{} Nil{}: {==}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{+result,+step},+others}:
      Shape.Both{Shape.Both{+resulting,lane},tail_lanes} = lanes
      %Equal.sym(U32,result,extent,resulting) : {_ <> Views.axis_extents(others) == extent <> rest : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => extent <> extents,Views.axis_extents(others),rest,stretched_extents(tail,rest,others,tail_lanes))
    case Nil{} Nil{} Con{axis,others}: Empty.absurd({Views.axis_extents(Con{axis,others}) == Nil{} : List<&2,U32>},lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd({Views.axis_extents(out) == Con{extent,rest} : List<&2,U32>},lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out: Empty.absurd({Views.axis_extents(out) == Nil{} : List<&2,U32>},lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}: Empty.absurd({Views.axis_extents(Nil{}) == Con{extent,rest} : List<&2,U32>},lanes)

# The extents of a compacted view.
def compact_extents_of(+source: List<&2,U32>,+axes: List<&2,Views.Axis>) ->
  {Views.axis_extents(Bands.compact_axes(source,axes)) == Bands.compact_extents(source,Views.axis_extents(axes)) : List<&2,U32>}:
  match source axes:
    case Con{+single,+rest} Con{Views.Axis{+extent,+stride},+others}:
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => Bool.pick(U32,U32.is_eq(single,1),1,extent) <> extents,
        Views.axis_extents(Bands.compact_axes(rest,others)),Bands.compact_extents(rest,Views.axis_extents(others)),compact_extents_of(rest,others))
    case Con{single,rest} Nil{}: {==}
    case Nil{} axes: {==}

# A contiguous view at offset zero addresses inside coordinates by their
# row-major position.
def canonical_address(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,layout: Linear.Layout(axes),+inside: Inside(axes,coordinates)) ->
  {Digits.address(points(axes,coordinates),0n,False{}) == position(Views.axis_extents(axes),coordinates,0n) : Nat}:
  +valid_points = valid(axes,coordinates,inside)
  Equal.trans(Nat,Digits.address(points(axes,coordinates),0n,False{}),Digits.position(points(axes,coordinates),0n),position(Views.axis_extents(axes),coordinates,0n),
    Digits.canonical(points(axes,coordinates),valid_points,
      %Equal.sym(List<&2,Views.Axis>,Digits.axes(points(axes,coordinates)),axes,axes_of(axes,coordinates,inside)) : Linear.Layout(_) layout,
      Digits.inside(points(axes,coordinates),valid_points)),
    positioned(axes,coordinates,0n))

# The last coordinate of every axis.
def lasts(axes: List<&2,Views.Axis>) -> List<&2,Nat>:
  match axes:
    case Con{Views.Axis{extent,stride},tail}: Nat.sub(U32.to_nat(extent),1n) <> lasts(tail)
    case Nil{}: []

def lasts_inside(+axes: List<&2,Views.Axis>,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) -> Inside(axes,lasts(axes)):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+tail}:
      Shape.Both{
        Order.last_below(U32.to_nat(extent),Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))),
        lasts_inside(tail,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))}

# A dot product is at least its prefix.
def prefix_le(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+prefix: Nat) ->
  {Nat.is_le(prefix,Digits.address(points(axes,coordinates),prefix,False{})) == True{} : Bool}:
  match axes coordinates:
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      +term = Nat.mul(coordinate,U32.to_nat(stride))
      Order.transitive(prefix,Nat.add(term,prefix),Digits.address(points(tail,rest),Nat.add(term,prefix),False{}),
        %Order.commutative(prefix,term) : {Nat.is_le(prefix,_) == True{} : Bool} Order.le_add(prefix,term),
        prefix_le(tail,rest,Nat.add(term,prefix)))
    case Con{axis,tail} Nil{}: Order.reflexive(prefix)
    case Nil{} coordinates: Order.reflexive(prefix)

def located_step(+extent: U32,+stride: U32,+tail: List<&2,Views.Axis>,+offset: U32,+length: U32,
  +nonzero: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+tail_nonempty: {Shapes.has_zero(Views.axis_extents(tail)) == False{} : Bool},
  +bound: {Nat.is_lt(Digits.address(points(tail,lasts(tail)),Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)),U32.to_nat(offset)),False{}),U32.to_nat(length)) == True{} : Bool},
  tail_located: @+offset: U32 -> {Nat.is_lt(Digits.address(points(tail,lasts(tail)),U32.to_nat(offset),False{}),U32.to_nat(length)) == True{} : Bool} ->
    {Views.found(Views.locate(tail,Views.last_coordinates(tail),Some{offset},length)) == True{} : Bool}) ->
  {Views.found(Views.locate(Views.Axis{extent,stride} <> tail,Views.last_coordinates(Views.Axis{extent,stride} <> tail),Some{offset},length)) == True{} : Bool}:
  +term = Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride))
  +total = Digits.address(points(tail,lasts(tail)),Nat.add(term,U32.to_nat(offset)),False{})
  +before = prefix_le(tail,lasts(tail),Nat.add(term,U32.to_nat(offset)))
  +reach = Order.le_lt_trans(Nat.add(term,U32.to_nat(offset)),total,U32.to_nat(length),before,bound)
  +offset_below = Order.le_lt_trans(U32.to_nat(offset),Nat.add(term,U32.to_nat(offset)),U32.to_nat(length),
    %Order.commutative(U32.to_nat(offset),term) : {Nat.is_le(U32.to_nat(offset),_) == True{} : Bool} Order.le_add(U32.to_nat(offset),term),reach)
  +last = FullRange.subtract(extent,1,extent,Order.positive_one(U32.to_nat(extent),nonzero),Order.sub_le(U32.to_nat(extent),1n))
  +last_below = {
    %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),last) : {Nat.is_lt(_,U32.to_nat(extent)) == True{} : Bool}
    Order.last_below(U32.to_nat(extent),nonzero) : {Nat.is_lt(U32.to_nat((extent - 1 : U32)),U32.to_nat(extent)) == True{} : Bool}}
  +term_room = Bounds.strict_le(term,U32.to_nat(length),Order.le_lt_trans(term,Nat.add(term,U32.to_nat(offset)),U32.to_nat(length),Order.le_add(term,U32.to_nat(offset)),reach))
  +product = ((extent - 1 : U32) * stride : U32)
  +product_value = {
    %last : {U32.to_nat(product) == Nat.mul(_,U32.to_nat(stride)) : Nat}
    FullRange.multiply((extent - 1 : U32),stride,length,
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),last) : {Nat.is_le(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(length)) == True{} : Bool} term_room)
    : {U32.to_nat(product) == term : Nat}}
  +multiplied = Completion.actual((extent - 1 : U32),stride,Machine.word_max(),product,
    %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),last) : {Nat.mul(_,U32.to_nat(stride)) == U32.to_nat(product) : Nat} Equal.sym(Nat,U32.to_nat(product),term,product_value),
    FullRange.maximum(product),{==})
  +length_positive = Order.le_lt_trans(0n,U32.to_nat(offset),U32.to_nat(length),Order.zero_le(U32.to_nat(offset)),offset_below)
  +one_less = FullRange.subtract(length,1,length,Order.positive_one(U32.to_nat(length),length_positive),Order.sub_le(U32.to_nat(length),1n))
  +offset_last = Order.before_last(U32.to_nat(offset),U32.to_nat(length),offset_below)
  +room = FullRange.subtract((length - 1 : U32),offset,length,
    %Equal.sym(Nat,U32.to_nat((length - 1 : U32)),Nat.sub(U32.to_nat(length),1n),one_less) : {Nat.is_le(U32.to_nat(offset),_) == True{} : Bool} offset_last,
    Order.transitive(Nat.sub(U32.to_nat((length - 1 : U32)),U32.to_nat(offset)),U32.to_nat((length - 1 : U32)),U32.to_nat(length),Order.sub_le(U32.to_nat((length - 1 : U32)),U32.to_nat(offset)),
      %Equal.sym(Nat,U32.to_nat((length - 1 : U32)),Nat.sub(U32.to_nat(length),1n),one_less) : {Nat.is_le(_,U32.to_nat(length)) == True{} : Bool} Order.sub_le(U32.to_nat(length),1n)))
  +fits_room = {
    %Equal.sym(Nat,U32.to_nat(product),term,product_value) : {Nat.is_le(_,U32.to_nat((length - 1 - offset : U32))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((length - 1 - offset : U32)),Nat.sub(U32.to_nat((length - 1 : U32)),U32.to_nat(offset)),room) : {Nat.is_le(term,_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((length - 1 : U32)),Nat.sub(U32.to_nat(length),1n),one_less) : {Nat.is_le(term,Nat.sub(_,U32.to_nat(offset))) == True{} : Bool}
    Order.cancel_offset_le(U32.to_nat(offset),term,Nat.sub(Nat.sub(U32.to_nat(length),1n),U32.to_nat(offset)),
      %Equal.sym(Nat,Nat.add(U32.to_nat(offset),Nat.sub(Nat.sub(U32.to_nat(length),1n),U32.to_nat(offset))),Nat.sub(U32.to_nat(length),1n),
        Order.restore_shift(U32.to_nat(offset),Nat.sub(U32.to_nat(length),1n),offset_last)) : {Nat.is_le(Nat.add(U32.to_nat(offset),term),_) == True{} : Bool}
      Order.before_last(Nat.add(U32.to_nat(offset),term),U32.to_nat(length),
        %Order.commutative(term,U32.to_nat(offset)) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} reach))
    : {Nat.is_le(U32.to_nat(product),U32.to_nat((length - 1 - offset : U32))) == True{} : Bool}}
  +moved = (offset + product : U32)
  +moved_value = {
    %product_value : {U32.to_nat(moved) == Nat.add(U32.to_nat(offset),_) : Nat}
    FullRange.add(offset,product,length,
      %Equal.sym(Nat,U32.to_nat(product),term,product_value) : {Nat.is_le(Nat.add(U32.to_nat(offset),_),U32.to_nat(length)) == True{} : Bool}
      %Order.commutative(term,U32.to_nat(offset)) : {Nat.is_le(_,U32.to_nat(length)) == True{} : Bool} Bounds.strict_le(Nat.add(term,U32.to_nat(offset)),U32.to_nat(length),reach))
    : {U32.to_nat(moved) == Nat.add(U32.to_nat(offset),term) : Nat}}
  %Equal.sym(Bool,U32.is_lt((extent - 1 : U32),extent),True{},U32Division.nat_lt_native((extent - 1 : U32),extent,last_below)) :
    {Views.found(Views.locate(tail,Views.last_coordinates(tail),Views.advance_inside(_ && U32.is_lt(offset,length),offset,(extent - 1 : U32),stride,length),length)) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(offset,length),True{},U32Division.nat_lt_native(offset,length,offset_below)) :
    {Views.found(Views.locate(tail,Views.last_coordinates(tail),Views.advance_inside(True{} && _,offset,(extent - 1 : U32),stride,length),length)) == True{} : Bool}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply((extent - 1 : U32),stride),Some{product},multiplied) :
    {Views.found(Views.locate(tail,Views.last_coordinates(tail),Views.advance_product(_,offset,(length - 1 - offset : U32)),length)) == True{} : Bool}
  %Equal.sym(Bool,U32.is_le(product,(length - 1 - offset : U32)),True{},U32Division.nat_le_native(product,(length - 1 - offset : U32),fits_room)) :
    {Views.found(Views.locate(tail,Views.last_coordinates(tail),Views.address_when(_,offset,product),length)) == True{} : Bool}
  tail_located(moved,
    %Equal.sym(Nat,U32.to_nat(moved),Nat.add(U32.to_nat(offset),term),moved_value) : {Nat.is_lt(Digits.address(points(tail,lasts(tail)),_,False{}),U32.to_nat(length)) == True{} : Bool}
    %Order.commutative(term,U32.to_nat(offset)) : {Nat.is_lt(Digits.address(points(tail,lasts(tail)),_,False{}),U32.to_nat(length)) == True{} : Bool} bound)

def located(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  bound: {Nat.is_lt(Digits.address(points(axes,lasts(axes)),U32.to_nat(offset),False{}),U32.to_nat(length)) == True{} : Bool}) ->
  {Views.found(Views.locate(axes,Views.last_coordinates(axes),Some{offset},length)) == True{} : Bool}:
  match axes:
    case Nil{}:
      %Equal.sym(Bool,U32.is_lt(offset,length),True{},U32Division.nat_lt_native(offset,length,bound)) : {Views.found(Views.finish_address(_,offset)) == True{} : Bool}
      {==}
    case Con{Views.Axis{+extent,+stride},+tail}:
      +nonzero = Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty))
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      located_step(extent,stride,tail,offset,length,nonzero,tail_nonempty,bound,next => fits => located(tail,next,length,tail_nonempty,fits))

# A nonempty view whose last address is inside a length fits it.
def fits_bound(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  bound: {Nat.is_lt(Digits.address(points(axes,lasts(axes)),U32.to_nat(offset),False{}),U32.to_nat(length)) == True{} : Bool}) ->
  {Views.fits(Views.View{axes,offset},length) == True{} : Bool}:
  %Equal.sym(Bool,Shapes.has_zero(Views.axis_extents(axes)),False{},nonempty) : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool}
  located(axes,offset,length,nonempty,bound)
