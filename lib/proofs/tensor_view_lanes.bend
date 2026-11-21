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
import ./encoded_coordinates.bend as Encoded
import ./word_arithmetic.bend as WordArithmetic
import ../kernel_gemm.bend as KernelGemm

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
      KernelGemm.Both<{Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},Inside(tail,rest)>
    case axes coordinates: Empty

def axes_of(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,inside: Inside(axes,coordinates)) ->
  {Digits.axes(points(axes,coordinates)) == axes : List<&2,Views.Axis>}:
  match axes coordinates:
    case Nil{} Nil{}: {==}
    case Nil{} Con{coordinate,rest}: Empty.absurd({Digits.axes(points(Nil{},Con{coordinate,rest})) == Nil{} : List<&2,Views.Axis>},inside)
    case Con{axis,tail} Nil{}: Empty.absurd({Digits.axes(points(Con{axis,tail},Nil{})) == Con{axis,tail} : List<&2,Views.Axis>},inside)
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      KernelGemm.Both{below,others} = inside
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,axes => Views.Axis{extent,stride} <> axes,Digits.axes(points(tail,rest)),tail,axes_of(tail,rest,others))

def valid(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,inside: Inside(axes,coordinates)) -> Digits.Valid(points(axes,coordinates)):
  match axes coordinates:
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{coordinate,rest}: Empty.absurd(Digits.Valid(points(Nil{},Con{coordinate,rest})),inside)
    case Con{axis,tail} Nil{}: Empty.absurd(Digits.Valid(points(Con{axis,tail},Nil{})),inside)
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest}:
      KernelGemm.Both{below,others} = inside
      KernelGemm.Both{below,valid(tail,rest,others)}

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
def native_terms(axes: List<&2,Views.Axis>,coordinates: List<&2,Nat>,prefix: U32) -> U32:
  match axes coordinates:
    case Con{Views.Axis{extent,stride},tail} Con{coordinate,rest}:
      native_terms(tail,rest,(U32.from_nat(coordinate) * stride + prefix : U32))
    case axes coordinates: prefix

def encoded_terms(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+prefix: Nat) ->
  {native_terms(axes,coordinates,U32.from_nat(prefix)) == U32.from_nat(Digits.address(points(axes,coordinates),prefix,False{})) : U32}:
  match axes coordinates:
    case Nil{} coordinates: {==}
    case Con{axis,tail} Nil{}: {==}
    case Con{Views.Axis{extent,+stride},+tail} Con{+coordinate,+rest}:
      +term = Nat.mul(coordinate,U32.to_nat(stride))
      +next = Equal.trans(U32,(U32.from_nat(coordinate) * stride + U32.from_nat(prefix) : U32),(U32.from_nat(term) + U32.from_nat(prefix) : U32),U32.from_nat(Nat.add(term,prefix)),
        Equal.cong(U32,U32,value => (value + U32.from_nat(prefix) : U32),(U32.from_nat(coordinate) * stride : U32),U32.from_nat(term),Encoded.scale_encoded(coordinate,stride)),WordArithmetic.native_add(term,prefix))
      %Equal.sym(U32,(U32.from_nat(coordinate) * stride + U32.from_nat(prefix) : U32),U32.from_nat(Nat.add(term,prefix)),next) :
        {native_terms(tail,rest,_) == U32.from_nat(Digits.address(points(tail,rest),Nat.add(term,prefix),False{})) : U32}
      encoded_terms(tail,rest,Nat.add(term,prefix))

def native_address(+axes: List<&2,Views.Axis>,+offset: U32,+coordinates: List<&2,Nat>,+inside: Inside(axes,coordinates)) ->
  {Copy.address(Views.View{axes,offset},position(Views.axis_extents(axes),coordinates,0n)) == native_terms(axes,coordinates,offset) : U32}:
  Equal.trans(U32,Copy.address(Views.View{axes,offset},position(Views.axis_extents(axes),coordinates,0n)),U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})),native_terms(axes,coordinates,offset),
    address(axes,offset,coordinates,inside),
    %Arithmetic.native_roundtrip(offset) : {U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})) == native_terms(axes,coordinates,_) : U32}
    Equal.sym(U32,native_terms(axes,coordinates,U32.from_nat(U32.to_nat(offset))),U32.from_nat(Digits.address(points(axes,coordinates),U32.to_nat(offset),False{})),encoded_terms(axes,coordinates,U32.to_nat(offset))))

def view_terms(view: Views.View,coordinates: List<&2,Nat>) -> U32:
  Views.View{axes,offset} = view
  native_terms(axes,coordinates,offset)

def ViewInside(view: Views.View,coordinates: List<&2,Nat>) -> Data:
  Views.View{axes,offset} = view
  Inside(axes,coordinates)

def view_address(+view: Views.View,+coordinates: List<&2,Nat>,inside: ViewInside(view,coordinates)) ->
  {Copy.address(view,position(Views.axis_extents_of(view),coordinates,0n)) == view_terms(view,coordinates) : U32}:
  Views.View{+axes,+offset} = view
  native_address(axes,offset,coordinates,inside)
