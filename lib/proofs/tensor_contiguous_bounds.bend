# The successful contiguous guard describes one interval of exactly numel cells.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_bounds.bend as ViewBounds
import ./checked_word_refinement.bend as Checked
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../kernel_gemm.bend as KernelGemm

def stride_guard(+extent: U32,+stride: U32,+count: U32,
  valid: {U32.is_le(extent,1) || U32.is_eq(stride,count) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(extent),1n) || Nat.is_eq(U32.to_nat(stride),U32.to_nat(count)) == True{} : Bool}:
  %Comparison.le(extent,1) : {_ || Nat.is_eq(U32.to_nat(stride),U32.to_nat(count)) == True{} : Bool}
  %Comparison.eq(stride,count) : {U32.is_le(extent,1) || _ == True{} : Bool}
  valid

def head_span(+extent: Nat,+stride: Nat,+count: Nat,+span: Nat,
  positive: {Nat.is_lt(0n,extent) == True{} : Bool},
  valid: {Nat.is_le(extent,1n) || Nat.is_eq(stride,count) == True{} : Bool},
  previous: {1n+span == count : Nat}) ->
  {1n+Nat.add(Nat.mul(Nat.sub(extent,1n),stride),span) == Nat.mul(extent,count) : Nat}:
  match extent:
    case 0n: Empty.absurd({1n+span == 0n : Nat},WordEncoding.false_true(positive))
    case 1n:
      %Equal.sym(Nat,Nat.add(count,0n),count,Addition.zero_right(count)) : {1n+span == _ : Nat}
      previous
    case 2n+ +rest:
      %Equal.sym(Nat,stride,count,Order.equal_of_is_eq(stride,count,valid)) :
        {1n+Nat.add(Nat.mul(1n+rest,_),span) == Nat.mul(2n+rest,count) : Nat}
      %previous : {1n+Nat.add(Nat.mul(1n+rest,count),span) == Nat.add(_,Nat.mul(1n+rest,count)) : Nat}
      %Addition.successor_right(Nat.mul(1n+rest,count),span) :
        {_ == Nat.add(1n+span,Nat.mul(1n+rest,count)) : Nat}
      Order.commutative(Nat.mul(1n+rest,count),1n+span)

law SpanCount:
  for result: Maybe<&2,U32>
  for product: Nat
  for span: Nat
  Data

def SpanCount(result,product,span):
  match result:
    case None{}: Unit
    case Some{count}: KernelGemm.Both<{product == U32.to_nat(count) : Nat},{1n+span == U32.to_nat(count) : Nat}>

def multiplied(+result: Maybe<&2,U32>,+extent: U32,+count: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  computed: {Shapes.checked_multiply(extent,count) == result : Maybe<&2,U32>},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  valid: {U32.is_le(extent,1) || U32.is_eq(stride,count) == True{} : Bool},
  product_equal: {tail_product == U32.to_nat(count) : Nat},span_equal: {1n+tail_span == U32.to_nat(count) : Nat}) ->
  SpanCount(result,Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match result:
    case None{}: Unit{}
    case Some{+output}:
      +product = Checked.multiply_sound(extent,count,output,computed)
      KernelGemm.Both{Equal.trans(Nat,Nat.mul(U32.to_nat(extent),tail_product),Nat.mul(U32.to_nat(extent),U32.to_nat(count)),U32.to_nat(output),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),tail_product,U32.to_nat(count),product_equal),product),{
        %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),
          FullRange.subtract(extent,1,extent,Order.strict_follow(0n,U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))) :
          {1n+Nat.add(Nat.mul(_,U32.to_nat(stride)),tail_span) == U32.to_nat(output) : Nat}
        Equal.trans(Nat,1n+Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)),tail_span),Nat.mul(U32.to_nat(extent),U32.to_nat(count)),U32.to_nat(output),
          head_span(U32.to_nat(extent),U32.to_nat(stride),U32.to_nat(count),tail_span,positive,stride_guard(extent,stride,count,valid),span_equal),product) :
          {1n+Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span) == U32.to_nat(output) : Nat}}}

def selected(+valid: Bool,+extent: U32,+count: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  guard: {U32.is_le(extent,1) || U32.is_eq(stride,count) == valid : Bool},
  product_equal: {tail_product == U32.to_nat(count) : Nat},span_equal: {1n+tail_span == U32.to_nat(count) : Nat}) ->
  SpanCount(Views.contiguous_product(valid,extent,count),Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match valid:
    case False{}: Unit{}
    case True{}: multiplied(Shapes.checked_multiply(extent,count),extent,count,stride,tail_product,tail_span,{==},positive,guard,product_equal,span_equal)

def axis(+inner: Maybe<&2,U32>,+extent: U32,+stride: U32,+tail_product: Nat,+tail_span: Nat,
  positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},previous: SpanCount(inner,tail_product,tail_span)) ->
  SpanCount(Views.contiguous_axis(extent,stride,inner),Nat.mul(U32.to_nat(extent),tail_product),Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail_span)):
  match inner:
    case None{}: Unit{}
    case Some{+count}:
      KernelGemm.Both{product_equal,span_equal} = previous
      selected(U32.is_le(extent,1) || U32.is_eq(stride,count),extent,count,stride,tail_product,tail_span,positive,{==},product_equal,span_equal)

def contiguous(+axes: List<&2,Views.Axis>,nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  SpanCount(Views.contiguous_size(axes),ShapeLaws.elements(Views.axis_extents(axes)),ViewBounds.last_span(axes)):
  match axes:
    case []: KernelGemm.Both{{==},{==}}
    case Views.Axis{+extent,+stride} <> +tail:
      +nonempty = nonempty
      axis(Views.contiguous_size(tail),extent,stride,ShapeLaws.elements(Views.axis_extents(tail)),ViewBounds.last_span(tail),
        Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)),
        contiguous(tail,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)))

def offset(view: Views.View) -> U32:
  Views.View{axes,offset} = view
  offset
