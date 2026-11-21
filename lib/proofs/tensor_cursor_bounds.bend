# Bounds of every countdown-cursor read, obtained from the checked last address.
import Base
import ../tensor_view.bend as Views
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_view_bounds.bend as ViewBounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_full_range.bend as FullRange
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor_shape.bend as Shapes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./encoded_coordinates.bend as Encoding
import ../kernel_gemm.bend as KernelGemm

def address_bounded(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,valid: CursorLaws.Valid(axes)) ->
  {Nat.is_le(CursorModel.address(axes,offset),Nat.add(CursorModel.span(axes),offset)) == True{} : Bool}:
  match axes:
    case []: Order.reflexive(offset)
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      KernelGemm.Both{inside,tail_valid} = valid
      %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.sub(extent,1n),stride),CursorModel.span(tail)),offset),
        Nat.add(Nat.mul(Nat.sub(extent,1n),stride),Nat.add(CursorModel.span(tail),offset)),Addition.associative(Nat.mul(Nat.sub(extent,1n),stride),CursorModel.span(tail),offset)) :
        {Nat.is_le(CursorModel.address(CursorModel.Coordinate{extent,coordinate,stride} <> tail,offset),_) == True{} : Bool}
      Encoding.sum_bounds(Nat.mul(coordinate,stride),CursorModel.address(tail,offset),Nat.mul(Nat.sub(extent,1n),stride),Nat.add(CursorModel.span(tail),offset),
        RowMajor.mul_le(coordinate,Nat.sub(extent,1n),stride,Order.before_last(coordinate,extent,inside)),address_bounded(tail,offset,tail_valid))

def selected_span(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: Nat,+tail: List<&2,CursorModel.Coordinate>,
  preserved: {CursorModel.span(CursorModel.advance(tail)) == CursorModel.span(tail) : Nat}) ->
  {CursorModel.span(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{extent,1n+coordinate,stride} <> tail,
      CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance(tail))) == CursorModel.span(CursorModel.Coordinate{extent,coordinate,stride} <> tail) : Nat}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.sub(extent,1n),stride),value),CursorModel.span(CursorModel.advance(tail)),CursorModel.span(tail),preserved)

def advance_span(+axes: List<&2,CursorModel.Coordinate>) -> {CursorModel.span(CursorModel.advance(axes)) == CursorModel.span(axes) : Nat}:
  match axes:
    case []: {==}
    case CursorModel.Coordinate{+extent,+coordinate,+stride} <> +tail:
      selected_span(Nat.is_lt(1n+coordinate,extent),extent,coordinate,stride,tail,advance_span(tail))

def included_span(+selected: Bool,+extent: U32,+stride: U32,+reversed: List<&2,CursorModel.Coordinate>,
  guard: {Nat.is_lt(1n,U32.to_nat(extent)) == selected : Bool}) ->
  {Nat.is_le(CursorModel.span(Bool.pick(List<&2,CursorModel.Coordinate>,selected,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)),
    Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),CursorModel.span(reversed))) == True{} : Bool}:
  match selected:
    case False{}:
      %Order.commutative(CursorModel.span(reversed),Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))) :
        {Nat.is_le(CursorModel.span(reversed),_) == True{} : Bool}
      Order.le_add(CursorModel.span(reversed),Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)))
    case True{}:
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),
        FullRange.subtract(extent,1,extent,Bounds.strict_le(1n,U32.to_nat(extent),guard),Order.sub_le(U32.to_nat(extent),1n))) :
        {Nat.is_le(CursorModel.span(CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed),Nat.add(Nat.mul(_,U32.to_nat(stride)),CursorModel.span(reversed))) == True{} : Bool}
      Order.reflexive(Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)),CursorModel.span(reversed)))

def prepended_bound(+head: Nat,+tail: Nat,+base: Nat,+current: Nat,+result: Nat,
  selected: {Nat.is_le(current,Nat.add(head,base)) == True{} : Bool},
  computed: {Nat.is_le(result,Nat.add(tail,current)) == True{} : Bool}) ->
  {Nat.is_le(result,Nat.add(Nat.add(head,tail),base)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(head,tail),base),Nat.add(tail,Nat.add(head,base)),
    Equal.trans(Nat,Nat.add(Nat.add(head,tail),base),Nat.add(head,Nat.add(tail,base)),Nat.add(tail,Nat.add(head,base)),
      Addition.associative(head,tail,base),RowMajor.add_swap(head,tail,base))) : {Nat.is_le(result,_) == True{} : Bool}
  Order.transitive(result,Nat.add(tail,current),Nat.add(tail,Nat.add(head,base)),computed,
    Encoding.sum_bounds(tail,current,tail,Nat.add(head,base),Order.reflexive(tail),selected))

def prepared_span(+axes: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>) ->
  {Nat.is_le(CursorModel.span(CursorModel.prepare_axes(axes,reversed)),Nat.add(ViewBounds.last_span(axes),CursorModel.span(reversed))) == True{} : Bool}:
  match axes:
    case []: Order.reflexive(CursorModel.span(reversed))
    case Views.Axis{+extent,+stride} <> +tail:
      prepended_bound(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ViewBounds.last_span(tail),CursorModel.span(reversed),
        CursorModel.span(Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)),
        CursorModel.span(CursorModel.prepare_axes(Views.Axis{extent,stride} <> tail,reversed)),
        included_span(Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,reversed,{==}),
        prepared_span(tail,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)))

def initial_span(+axes: List<&2,Views.Axis>) ->
  {Nat.is_le(CursorModel.span(CursorModel.prepare_axes(axes,[])),ViewBounds.last_span(axes)) == True{} : Bool}:
  %Addition.zero_right(ViewBounds.last_span(axes)) : {Nat.is_le(CursorModel.span(CursorModel.prepare_axes(axes,[])),_) == True{} : Bool}
  prepared_span(axes,[])

def fits_bound(+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,
  fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(CursorModel.span(CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}:
  Order.le_lt_trans(Nat.add(CursorModel.span(CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)),Nat.add(ViewBounds.last_span(axes),U32.to_nat(offset)),U32.to_nat(length),
    Order.add_le(CursorModel.span(CursorModel.prepare_axes(axes,[])),ViewBounds.last_span(axes),U32.to_nat(offset),initial_span(axes)),{
      %Equal.sym(Nat,Nat.add(ViewBounds.last_span(axes),U32.to_nat(offset)),Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),Order.commutative(ViewBounds.last_span(axes),U32.to_nat(offset))) :
        {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
      %ViewBounds.last_location(axes,U32.to_nat(offset)) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
      ViewBounds.last_address(axes,offset,length,fits) : {Nat.is_lt(Nat.add(ViewBounds.last_span(axes),U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}})

def read_inside(+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,+length: U32,valid: CursorLaws.Valid(axes),
  bound: {Nat.is_lt(Nat.add(CursorModel.span(axes),offset),U32.to_nat(length)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(CursorModel.address(axes,offset))),U32.to_nat(length)) == True{} : Bool}:
  +inside = Order.le_lt_trans(CursorModel.address(axes,offset),Nat.add(CursorModel.span(axes),offset),U32.to_nat(length),address_bounded(axes,offset,valid),bound)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(CursorModel.address(axes,offset))),CursorModel.address(axes,offset),
    FullRange.roundtrip(CursorModel.address(axes,offset),length,Bounds.strict_le(CursorModel.address(axes,offset),U32.to_nat(length),inside))) :
    {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
  inside
