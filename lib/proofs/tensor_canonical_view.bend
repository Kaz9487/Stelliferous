# The row-major view of a shape: the view that view_of builds, its extents,
# its element count and its contiguity.
import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../machine_limits.bend as Machine
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./tensor_view_metadata_refinement.bend as Metadata
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./word_multiplication.bend as Multiplication
import ./nat_to_u32_bounds.bend as Bounds
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison
import ./checked_word_refinement.bend as Checked
import ./checked_word_completion.bend as Completion
import ./u32_division.bend as U32Division

# The axes built from the reversed extents, innermost first, each stride the
# product of the extents inside it.
def axis_model(reversed: List<&2,U32>,+stride: U32,axes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match reversed:
    case []: axes
    case +head <> tail: axis_model(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes)

def canonical_view(+extents: List<&2,U32>,+count: U32) -> Views.View:
  Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,U32.is_zero(count),0,1),[]),0}

def axes(view: Views.View) -> List<&2,Views.Axis>:
  Views.View{axes,offset} = view
  axes

def zero_product(+right: U32) -> {U32.mul(0,right) == 0 : U32}:
  Multiplication.native_multiply(0,right)

def built_empty(+reversed: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32}) ->
  {Views.build_axes(reversed,Some{0},axes,offset) == Some{Views.View{axis_model(reversed,0,axes),offset}} : Maybe<&2,Views.View>}:
  match reversed:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(0,head),Some{0},
        %zero_product(head) : {Shapes.checked_multiply(0,head) == Some{_} : Maybe<&2,U32>}
        CountCompletion.checked_product(0,head,limit,maximum,Order.zero_le(U32.to_nat(limit)))) :
        {Views.build_axes(tail,_,Views.Axis{head,0} <> axes,offset) == Some{Views.View{axis_model(tail,U32.mul(0,head),Views.Axis{head,0} <> axes),offset}} : Maybe<&2,Views.View>}
      %Equal.sym(U32,U32.mul(0,head),0,zero_product(head)) : {Views.build_axes(tail,Some{0},Views.Axis{head,0} <> axes,offset) == Some{Views.View{axis_model(tail,_,Views.Axis{head,0} <> axes),offset}} : Maybe<&2,Views.View>}
      built_empty(tail,Views.Axis{head,0} <> axes,offset,limit,maximum)

def built_nonempty(+reversed: List<&2,U32>,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +nonempty: {Shapes.has_zero(reversed) == False{} : Bool},+room: {Nat.is_le(Nat.mul(U32.to_nat(stride),ShapeLaws.elements(reversed)),U32.to_nat(limit)) == True{} : Bool}) ->
  {Views.build_axes(reversed,Some{stride},axes,offset) == Some{Views.View{axis_model(reversed,stride,axes),offset}} : Maybe<&2,Views.View>}:
  match reversed:
    case []: {==}
    case +head <> +tail:
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(head),Shapes.has_zero(tail),nonempty)
      +prefix_room = Order.transitive(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),Nat.mul(U32.to_nat(stride),ShapeLaws.elements(head <> tail)),U32.to_nat(limit),
        %RowMajor.assoc(U32.to_nat(stride),U32.to_nat(head),ShapeLaws.elements(tail)) : {Nat.is_le(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),_) == True{} : Bool}
        RowMajor.product_covers(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),ShapeLaws.elements(tail),CountCompletion.positive_elements(tail,tail_nonempty)),room)
      +product_exact = FullRange.multiply(stride,head,limit,prefix_room)
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(stride,head),Some{U32.mul(stride,head)},CountCompletion.checked_product(stride,head,limit,maximum,prefix_room)) :
        {Views.build_axes(tail,_,Views.Axis{head,stride} <> axes,offset) == Some{Views.View{axis_model(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes),offset}} : Maybe<&2,Views.View>}
      built_nonempty(tail,U32.mul(stride,head),Views.Axis{head,stride} <> axes,offset,limit,maximum,tail_nonempty,
        %Equal.sym(Nat,U32.to_nat(U32.mul(stride,head)),Nat.mul(U32.to_nat(stride),U32.to_nat(head)),product_exact) : {Nat.is_le(Nat.mul(_,ShapeLaws.elements(tail)),U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(stride),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(U32.to_nat(stride),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),RowMajor.assoc(U32.to_nat(stride),U32.to_nat(head),ShapeLaws.elements(tail))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} room)

def reversed_elements(+extents: List<&2,U32>,+tail: List<&2,U32>) ->
  {ShapeLaws.elements(List.reverse.go(&2,U32,extents,tail)) == Nat.mul(ShapeLaws.elements(extents),ShapeLaws.elements(tail)) : Nat}:
  match extents:
    case []: Equal.sym(Nat,Nat.add(ShapeLaws.elements(tail),0n),ShapeLaws.elements(tail),Addition.zero_right(ShapeLaws.elements(tail)))
    case +head <> +rest:
      Equal.trans(Nat,ShapeLaws.elements(List.reverse.go(&2,U32,rest,head <> tail)),Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),
        Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),ShapeLaws.elements(tail)),reversed_elements(rest,head <> tail),
        Equal.trans(Nat,Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),Nat.mul(Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),ShapeLaws.elements(tail)),
          Equal.sym(Nat,Nat.mul(Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),ShapeLaws.elements(tail)),Nat.mul(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail))),RowMajor.assoc(ShapeLaws.elements(rest),U32.to_nat(head),ShapeLaws.elements(tail))),
          Equal.cong(Nat,Nat,value => Nat.mul(value,ShapeLaws.elements(tail)),Nat.mul(ShapeLaws.elements(rest),U32.to_nat(head)),Nat.mul(U32.to_nat(head),ShapeLaws.elements(rest)),RowMajor.multiply_commutative(ShapeLaws.elements(rest),U32.to_nat(head)))))

def reverse_elements(+extents: List<&2,U32>) -> {ShapeLaws.elements(List.reverse(&2,U32,extents)) == ShapeLaws.elements(extents) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),Nat.mul(ShapeLaws.elements(extents),1n),ShapeLaws.elements(extents),reversed_elements(extents,[]),Order.times_one(ShapeLaws.elements(extents)))

def view_when_bounded(+extents: List<&2,U32>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  empty: Bool,+classified: {U32.is_zero(count) == empty : Bool},+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},
  +within: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Views.contiguous_count(Some{count},extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  match empty:
    case True{}:
      %Equal.sym(Bool,U32.is_zero(count),True{},classified) :
        {Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0) == Some{Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,_,0,1),[]),0}} : Maybe<&2,Views.View>}
      built_empty(List.reverse(&2,U32,extents),[],0,limit,maximum)
    case False{}:
      +positive = Arithmetic.word_nonzero_positive(count,classified)
      %Equal.sym(Bool,U32.is_zero(count),False{},classified) :
        {Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0) == Some{Views.View{axis_model(List.reverse(&2,U32,extents),Bool.pick(U32,_,0,1),[]),0}} : Maybe<&2,Views.View>}
      built_nonempty(List.reverse(&2,U32,extents),1,[],0,limit,maximum,
        CountCompletion.positive_not_zero(List.reverse(&2,U32,extents),%Equal.sym(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),ShapeLaws.elements(extents),reverse_elements(extents)) :
          {Nat.is_lt(0n,_) == True{} : Bool} %Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive),
        %Equal.sym(Nat,Nat.add(ShapeLaws.elements(List.reverse(&2,U32,extents)),0n),ShapeLaws.elements(List.reverse(&2,U32,extents)),Addition.zero_right(ShapeLaws.elements(List.reverse(&2,U32,extents)))) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(List.reverse(&2,U32,extents)),ShapeLaws.elements(extents),reverse_elements(extents)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} Bounds.as_nat_le(count,limit,within))

def view_when(+extents: List<&2,U32>,+count: U32,empty: Bool,+classified: {U32.is_zero(count) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.contiguous_count(Some{count},extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  view_when_bounded(extents,count,Machine.word_max(),{==},empty,classified,exact,FullRange.maximum(count))

# view_of builds the canonical view of a shape with count elements.
def view(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Tensors.view_of(extents) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{extents}),Some{count},CountCompletion.numel(extents,count,exact)) :
    {Views.contiguous_count(_,extents,0) == Some{canonical_view(extents,count)} : Maybe<&2,Views.View>}
  view_when(extents,count,U32.is_zero(count),{==},exact)

def canonical_shape(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.axis_extents_of(canonical_view(extents,count)) == extents : List<&2,U32>}:
  %view(extents,count,exact) : {Metadata.observed_extents(_,extents) == extents : List<&2,U32>}
  Metadata.contiguous_shape(extents,0)

def target_counted_from_count(+extents: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(extents),count>) ->
  {Views.numel(canonical_view(extents,count)) == Some{count} : Maybe<&2,U32>}:
  Count.Count{+exact,bounded} = evidence
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical_view(extents,count)),extents,
    canonical_shape(extents,count,exact)) : {Shapes.numel(Shapes.Shape{_}) == Some{count} : Maybe<&2,U32>}
  CountCompletion.numel(extents,count,exact)

# A successful build from an already contiguous suffix remains contiguous.
def ContiguousResult(result: Maybe<&2,Views.View>) -> Data:
  match result:
    case None{}: Unit
    case Some{view}: {Views.is_contiguous(view) == True{} : Bool}

def failed_build(reversed: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32) -> ContiguousResult(Views.build_axes(reversed,None{},axes,offset)):
  match reversed:
    case Nil{}: Unit{}
    case Con{extent,tail}: Unit{}

def contiguous_nonempty(+empty: Bool,+axes: List<&2,Views.Axis>,accepted: {Views.has_count(Views.contiguous_size(axes)) == True{} : Bool}) ->
  {Views.contiguous_nonempty(empty,axes) == True{} : Bool}:
  match empty:
    case True{}: {==}
    case False{}: accepted

def same_stride(+extent: U32,+stride: U32) -> {U32.is_le(extent,1) || U32.is_eq(stride,stride) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(stride,stride),True{},
    %Equal.sym(Bool,U32.is_eq(stride,stride),Nat.is_eq(U32.to_nat(stride),U32.to_nat(stride)),Comparison.eq(stride,stride)) : {_ == True{} : Bool} Order.equal_reflexive(U32.to_nat(stride))) :
    {U32.is_le(extent,1) || _ == True{} : Bool}
  Boolean.or_true_right(U32.is_le(extent,1))

def prepended_size(+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>}) ->
  {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,Views.contiguous_size(axes),Some{stride},inner) : {Views.contiguous_axis(extent,stride,_) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(extent,1) || U32.is_eq(stride,stride),True{},same_stride(extent,stride)) :
    {Views.contiguous_product(_,extent,stride) == Shapes.checked_multiply(extent,stride) : Maybe<&2,U32>}
  {==}

def built_step_bounded(+reversed: List<&2,U32>,+result: Maybe<&2,U32>,+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,+word_limit: U32,+maximum: {word_limit == Machine.word_max() : U32},
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>},computed: {Shapes.checked_multiply(stride,extent) == result : Maybe<&2,U32>},
  next: @value: U32 -> {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Some{value} : Maybe<&2,U32>} ->
    ContiguousResult(Views.build_axes(reversed,Some{value},Views.Axis{extent,stride} <> axes,offset))) ->
  ContiguousResult(Views.build_axes(reversed,result,Views.Axis{extent,stride} <> axes,offset)):
  match result:
    case None{}: failed_build(reversed,Views.Axis{extent,stride} <> axes,offset)
    case Some{+value}:
      next(value)(
        Equal.trans(Maybe<&2,U32>,Views.contiguous_size(Views.Axis{extent,stride} <> axes),Shapes.checked_multiply(extent,stride),Some{value},
          prepended_size(extent,stride,axes,inner),
          Completion.actual(extent,stride,word_limit,value,
            Equal.trans(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(stride)),Nat.mul(U32.to_nat(stride),U32.to_nat(extent)),U32.to_nat(value),
              RowMajor.multiply_commutative(U32.to_nat(extent),U32.to_nat(stride)),Checked.multiply_sound(stride,extent,value,computed)),
            %Equal.sym(U32,word_limit,Machine.word_max(),maximum) : {U32.is_le(value,_) == True{} : Bool} FullRange.maximum(value),maximum)))

def built_step(+reversed: List<&2,U32>,+result: Maybe<&2,U32>,+extent: U32,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>},computed: {Shapes.checked_multiply(stride,extent) == result : Maybe<&2,U32>},
  next: @value: U32 -> {Views.contiguous_size(Views.Axis{extent,stride} <> axes) == Some{value} : Maybe<&2,U32>} ->
    ContiguousResult(Views.build_axes(reversed,Some{value},Views.Axis{extent,stride} <> axes,offset))) ->
  ContiguousResult(Views.build_axes(reversed,result,Views.Axis{extent,stride} <> axes,offset)):
  built_step_bounded(reversed,result,extent,stride,axes,offset,Machine.word_max(),{==},inner,computed,next)

def built_contiguous(+reversed: List<&2,U32>,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  inner: {Views.contiguous_size(axes) == Some{stride} : Maybe<&2,U32>}) -> ContiguousResult(Views.build_axes(reversed,Some{stride},axes,offset)):
  match reversed:
    case []:
      contiguous_nonempty(Shapes.has_zero(Views.axis_extents(axes)),axes,
        Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,Views.contiguous_size(axes),Some{stride},inner))
    case +extent <> +tail:
      built_step(tail,Shapes.checked_multiply(stride,extent),extent,stride,axes,offset,inner,{==},
        value => inner => built_contiguous(tail,value,Views.Axis{extent,stride} <> axes,offset,inner))

def canonical_contiguous_when(+extents: List<&2,U32>,+count: U32,empty: Bool,+classified: {Shapes.has_zero(extents) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) -> {Views.is_contiguous(canonical_view(extents,count)) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical_view(extents,count)),extents,canonical_shape(extents,count,exact)) :
        {Views.contiguous_nonempty(Shapes.has_zero(_),axes(canonical_view(extents,count))) == True{} : Bool}
      %Equal.sym(Bool,Shapes.has_zero(extents),True{},classified) : {Views.contiguous_nonempty(_,axes(canonical_view(extents,count))) == True{} : Bool}
      {==}
    case False{}:
      +nonzero = U32Division.nonzero(count,%exact : {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_elements(extents,classified))
      %view_when(extents,count,False{},nonzero,exact) : ContiguousResult(_)
      %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) : ContiguousResult(Views.build_axes(List.reverse(&2,U32,extents),Some{Bool.pick(U32,_,0,1)},[],0))
      built_contiguous(List.reverse(&2,U32,extents),1,[],0,{==})

def canonical_contiguous(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.is_contiguous(canonical_view(extents,count)) == True{} : Bool}:
  canonical_contiguous_when(extents,count,Shapes.has_zero(extents),{==},exact)
