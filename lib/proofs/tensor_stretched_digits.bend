# What a view stretched under NumPy broadcasting reads at the digits of a
# target index: the row-major address of the source digits, the digits past
# the leading axes the source lacks with 0 on every axis it repeats.
import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_tiled_view_reads.bend as KeptReads
import ./tensor_view_reads.bend as ViewReads
import ./tensor_canonical_view.bend as Canonical
import ./tensor_copy_values.bend as CopyValues
import ./tensor_view_metadata_refinement.bend as MetadataLaws
import ./tensor_grid_box.bend as Box
import ./tensor_view_part.bend as ViewPart
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./encoded_coordinates.bend as EncodedCoordinates
import ./u32_comparison.bend as Comparison
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_vector_descriptor_refinement.bend as Descriptor
import ./boolean_logic.bend as Boolean
import ../tensor_shape.bend as Shapes
import ./natural_addition.bend as Addition
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_linear_layout.bend as Linear
import ./tensor_row_major_coordinates.bend as RowMajorCoordinates
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds

# ---- The digits a stretch reads ----

# The digit where the extents agree, 0 where the source repeats its one index.
def repeated_digits(sizes: List<&2,U32>,target: List<&2,U32>,digits: List<&2,Nat>) -> List<&2,Nat>:
  match sizes target digits:
    case Con{+size,sizes} Con{+extent,target} Con{+digit,digits}: Bool.pick(Nat,U32.is_eq(size,extent),digit,0n) <> repeated_digits(sizes,target,digits)
    case sizes target digits: []

def skip(count: Nat,digits: List<&2,Nat>) -> List<&2,Nat>:
  match count digits:
    case 0n digits: digits
    case 1n+next Con{digit,rest}: skip(next,rest)
    case 1n+next Nil{}: []

def skip_extents(count: Nat,extents: List<&2,U32>) -> List<&2,U32>:
  match count extents:
    case 0n extents: extents
    case 1n+next Con{extent,rest}: skip_extents(next,rest)
    case 1n+next Nil{}: []

# The source digits of the digits of a target index under a stretch.
def source_digits(+source: List<&2,U32>,+target: List<&2,U32>,+digits: List<&2,Nat>) -> List<&2,Nat>:
  +lacked = Nat.sub(List.length(&2,U32,target),List.length(&2,U32,source))
  repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))

def axes_or(found: Maybe<&2,List<&2,Views.Axis>>,+default: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match found:
    case Some{axes}: axes
    case None{}: default

def axis_or(found: Maybe<&2,Views.Axis>,+default: Views.Axis) -> Views.Axis:
  match found:
    case Some{axis}: axis
    case None{}: default

def axis_stride(+axis: Views.Axis) -> Nat:
  Views.Axis{+extent,+stride} = axis
  U32.to_nat(stride)

# ---- The sum of a stretch at its digits ----

# One stretched axis steps its digit as the source axis steps the repeated
# digit.
def axis_term(same: Bool,single: Bool,+size: U32,+stride: U32,+extent: U32,+digit: Nat,+axis: Views.Axis,
  +seen: {Views.stretched_axis(same,single,size,stride,extent) == Some{axis} : Maybe<&2,Views.Axis>}) ->
  {Nat.mul(digit,axis_stride(axis)) == Nat.mul(Bool.pick(Nat,same,digit,0n),U32.to_nat(stride)) : Nat}:
  match same single:
    case True{} +single:
      %Equal.cong(Maybe<&2,Views.Axis>,Views.Axis,found => axis_or(found,axis),Some{Views.Axis{size,stride}},Some{axis},seen) :
        {Nat.mul(digit,axis_stride(_)) == Nat.mul(digit,U32.to_nat(stride)) : Nat}
      {==}
    case False{} True{}:
      %Equal.cong(Maybe<&2,Views.Axis>,Views.Axis,found => axis_or(found,axis),Some{Views.Axis{extent,0}},Some{axis},seen) :
        {Nat.mul(digit,axis_stride(_)) == Nat.mul(0n,U32.to_nat(stride)) : Nat}
      RowMajor.multiply_zero_right(digit)
    case False{} False{}:
      Empty.absurd({Nat.mul(digit,axis_stride(axis)) == Nat.mul(Bool.pick(Nat,False{},digit,0n),U32.to_nat(stride)) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.Axis>,Bool,found => Maybe.is_some(&2,Views.Axis,found),None{},Some{axis},seen)))

# The sum of a view of no digits is 0.
def sum_none(axes: List<&2,Views.Axis>) -> {KeptReads.digit_sum(axes,[],[]) == 0n : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +rest: {==}

# The sum of an axis and the rest of a stretch at some digits.
def digits_sum(+size: U32,+stride: U32,+extent: U32,+rest: List<&2,Views.Axis>,+others: List<&2,U32>,axis: Views.Axis,digits: List<&2,Nat>,
  +found: List<&2,Views.Axis>,+axis_seen: {Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent) == Some{axis} : Maybe<&2,Views.Axis>},
  inner: @+more: List<&2,Nat> -> {KeptReads.digit_sum(found,[],more) == KeptReads.digit_sum(rest,[],repeated_digits(Views.axis_extents(rest),others,more)) : Nat}) ->
  {KeptReads.digit_sum(axis <> found,[],digits) ==
    KeptReads.digit_sum(Views.Axis{size,stride} <> rest,[],repeated_digits(size <> Views.axis_extents(rest),extent <> others,digits)) : Nat}:
  match axis:
    case Views.Axis{+axis_extent,+axis_step}:
      match digits:
        case []: {==}
        case +digit <> +more:
          +term = axis_term(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent,digit,Views.Axis{axis_extent,axis_step},axis_seen)
          +own = Nat.mul(digit,U32.to_nat(axis_step))
          +repeated = Nat.mul(Bool.pick(Nat,U32.is_eq(size,extent),digit,0n),U32.to_nat(stride))
          +found_sum = KeptReads.digit_sum(found,[],more)
          +rest_sum = KeptReads.digit_sum(rest,[],repeated_digits(Views.axis_extents(rest),others,more))
          Equal.trans(Nat,Nat.add(own,found_sum),Nat.add(repeated,found_sum),Nat.add(repeated,rest_sum),
            Equal.cong(Nat,Nat,value => Nat.add(value,found_sum),own,repeated,term),
            Equal.cong(Nat,Nat,value => Nat.add(repeated,value),found_sum,rest_sum,inner(more)))

# An axis and the rest of a stretch put together.
def prepend_sum(axis_found: Maybe<&2,Views.Axis>,rest_found: Maybe<&2,List<&2,Views.Axis>>,+size: U32,+stride: U32,+extent: U32,+rest: List<&2,Views.Axis>,
  +others: List<&2,U32>,+digits: List<&2,Nat>,+out: List<&2,Views.Axis>,
  +axis_seen: {Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent) == axis_found : Maybe<&2,Views.Axis>},
  +rest_seen: {Views.stretch_axes(rest,others) == rest_found : Maybe<&2,List<&2,Views.Axis>>},
  +seen: {Views.stretched_prepend(axis_found,rest_found) == Some{out} : Maybe<&2,List<&2,Views.Axis>>},
  inner: @+found: List<&2,Views.Axis> -> @+found_seen: {Views.stretch_axes(rest,others) == Some{found} : Maybe<&2,List<&2,Views.Axis>>} -> @+more: List<&2,Nat> ->
    {KeptReads.digit_sum(found,[],more) == KeptReads.digit_sum(rest,[],repeated_digits(Views.axis_extents(rest),others,more)) : Nat}) ->
  {KeptReads.digit_sum(out,[],digits) ==
    KeptReads.digit_sum(Views.Axis{size,stride} <> rest,[],repeated_digits(size <> Views.axis_extents(rest),extent <> others,digits)) : Nat}:
  match axis_found rest_found:
    case Some{+axis} Some{+found}:
      %Equal.cong(Maybe<&2,List<&2,Views.Axis>>,List<&2,Views.Axis>,result => axes_or(result,out),Some{axis <> found},Some{out},seen) :
        {KeptReads.digit_sum(_,[],digits) ==
          KeptReads.digit_sum(Views.Axis{size,stride} <> rest,[],repeated_digits(size <> Views.axis_extents(rest),extent <> others,digits)) : Nat}
      digits_sum(size,stride,extent,rest,others,axis,digits,found,axis_seen,inner(found,rest_seen))
    case Some{+axis} None{}:
      Empty.absurd({KeptReads.digit_sum(out,[],digits) ==
          KeptReads.digit_sum(Views.Axis{size,stride} <> rest,[],repeated_digits(size <> Views.axis_extents(rest),extent <> others,digits)) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case None{} +rest_found:
      Empty.absurd({KeptReads.digit_sum(out,[],digits) ==
          KeptReads.digit_sum(Views.Axis{size,stride} <> rest,[],repeated_digits(size <> Views.axis_extents(rest),extent <> others,digits)) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))

# Axes stretched to a target sum the digits of a target index as the source
# axes sum the digits they repeat.
def stretch_sum(axes: List<&2,Views.Axis>,target: List<&2,U32>,+digits: List<&2,Nat>,+out: List<&2,Views.Axis>,
  +seen: {Views.stretch_axes(axes,target) == Some{out} : Maybe<&2,List<&2,Views.Axis>>}) ->
  {KeptReads.digit_sum(out,[],digits) == KeptReads.digit_sum(axes,[],repeated_digits(Views.axis_extents(axes),target,digits)) : Nat}:
  match axes target:
    case Nil{} Nil{}:
      %Equal.cong(Maybe<&2,List<&2,Views.Axis>>,List<&2,Views.Axis>,result => axes_or(result,out),Some{[]},Some{out},seen) :
        {KeptReads.digit_sum(_,[],digits) == KeptReads.digit_sum([],[],repeated_digits([],[],digits)) : Nat}
      {==}
    case Nil{} Con{+extent,+others}:
      Empty.absurd({KeptReads.digit_sum(out,[],digits) == KeptReads.digit_sum([],[],repeated_digits([],extent <> others,digits)) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case Con{+axis,+rest} Nil{}:
      Empty.absurd({KeptReads.digit_sum(out,[],digits) == KeptReads.digit_sum(axis <> rest,[],repeated_digits(Views.axis_extents(axis <> rest),[],digits)) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case Con{Views.Axis{+size,+stride},+rest} Con{+extent,+others}:
      prepend_sum(Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent),Views.stretch_axes(rest,others),size,stride,extent,rest,others,digits,
        out,{==},{==},seen,found => found_seen => more => stretch_sum(rest,others,more,found,found_seen))

# ---- Leading axes the source lacks ----

def repeated_no_target(sizes: List<&2,U32>,+digits: List<&2,Nat>) -> {repeated_digits(sizes,[],digits) == [] : List<&2,Nat>}:
  match sizes:
    case []: {==}
    case +size <> +rest: {==}

def repeated_no_digits(sizes: List<&2,U32>,target: List<&2,U32>) -> {repeated_digits(sizes,target,[]) == [] : List<&2,Nat>}:
  match sizes target:
    case Nil{} +target: {==}
    case Con{+size,+rest} Nil{}: {==}
    case Con{+size,+rest} Con{+extent,+others}: {==}

# The extents of axes with leading axes of one index.
def ones_extents(count: Nat,+axes: List<&2,Views.Axis>) ->
  {Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),axes)) == List.append(&2,U32,List.replicate(U32,count,1),Views.axis_extents(axes)) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => 1 <> extents,Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,next,Views.Axis{1,0}),axes)),
        List.append(&2,U32,List.replicate(U32,next,1),Views.axis_extents(axes)),ones_extents(next,axes))

# Leading axes of one index and step 0 add nothing to the sum.
def ones_sum(count: Nat,+axes: List<&2,Views.Axis>,digits: List<&2,Nat>) ->
  {KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),axes),[],digits) == KeptReads.digit_sum(axes,[],skip(count,digits)) : Nat}:
  match count digits:
    case 0n +digits: {==}
    case 1n+ +next Nil{}: Equal.sym(Nat,KeptReads.digit_sum(axes,[],[]),0n,sum_none(axes))
    case 1n+ +next Con{+digit,+more}:
      %Equal.sym(Nat,Nat.mul(digit,0n),0n,RowMajor.multiply_zero_right(digit)) :
        {Nat.add(_,KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,next,Views.Axis{1,0}),axes),[],more)) ==
          KeptReads.digit_sum(axes,[],skip(next,more)) : Nat}
      ones_sum(next,axes,more)

# The repeated digits past leading extents of one.
def ones_skipped(count: Nat,+source: List<&2,U32>,target: List<&2,U32>,digits: List<&2,Nat>) ->
  {skip(count,repeated_digits(List.append(&2,U32,List.replicate(U32,count,1),source),target,digits)) == repeated_digits(source,skip_extents(count,target),skip(count,digits)) : List<&2,Nat>}:
  match count target digits:
    case 0n +target +digits: {==}
    case 1n+ +next Nil{} +digits: Equal.sym(List<&2,Nat>,repeated_digits(source,[],skip(1n+next,digits)),[],repeated_no_target(source,skip(1n+next,digits)))
    case 1n+ +next Con{+extent,+others} Nil{}:
      Equal.sym(List<&2,Nat>,repeated_digits(source,skip_extents(next,others),[]),[],repeated_no_digits(source,skip_extents(next,others)))
    case 1n+ +next Con{+extent,+others} Con{+digit,+more}: ones_skipped(next,source,others,more)

# Digits below leading extents of one and the source are below the source past them.
def ones_below(count: Nat,+source: List<&2,U32>,digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(List.append(&2,U32,List.replicate(U32,count,1),source),digits)) ->
  Coordinates.DigitsBelow(source,skip(count,digits)):
  match count digits:
    case 0n +digits: below
    case 1n+ +next Nil{}: Empty.absurd(Coordinates.DigitsBelow(source,skip(1n+next,[])),below)
    case 1n+ +next Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      ones_below(next,source,more,later)

# ---- Repeated digits below the source ----

def digit_below(same: Bool,single: Bool,+size: U32,+stride: U32,+extent: U32,+digit: Nat,+axis: Views.Axis,
  +seen: {Views.stretched_axis(same,single,size,stride,extent) == Some{axis} : Maybe<&2,Views.Axis>},
  +same_value: {U32.is_eq(size,extent) == same : Bool},+single_value: {U32.is_eq(size,1) == single : Bool},
  +below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) -> {Nat.is_lt(Bool.pick(Nat,same,digit,0n),U32.to_nat(size)) == True{} : Bool}:
  match same single:
    case True{} +single:
      %Equal.sym(U32,size,extent,Arithmetic.equal_of_is_eq(size,extent,same_value)) : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool}
      below
    case False{} True{}:
      %Equal.sym(U32,size,1,Arithmetic.equal_of_is_eq(size,1,single_value)) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
      {==}
    case False{} False{}:
      Empty.absurd({Nat.is_lt(Bool.pick(Nat,False{},digit,0n),U32.to_nat(size)) == True{} : Bool},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.Axis>,Bool,found => Maybe.is_some(&2,Views.Axis,found),None{},Some{axis},seen)))

def prepend_below(axis_found: Maybe<&2,Views.Axis>,rest_found: Maybe<&2,List<&2,Views.Axis>>,+size: U32,+stride: U32,+extent: U32,+rest: List<&2,Views.Axis>,
  +others: List<&2,U32>,+digit: Nat,+more: List<&2,Nat>,+out: List<&2,Views.Axis>,
  +axis_seen: {Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent) == axis_found : Maybe<&2,Views.Axis>},
  +rest_seen: {Views.stretch_axes(rest,others) == rest_found : Maybe<&2,List<&2,Views.Axis>>},
  +seen: {Views.stretched_prepend(axis_found,rest_found) == Some{out} : Maybe<&2,List<&2,Views.Axis>>},
  +below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  inner: @+found: List<&2,Views.Axis> -> @+found_seen: {Views.stretch_axes(rest,others) == Some{found} : Maybe<&2,List<&2,Views.Axis>>} ->
    Coordinates.DigitsBelow(Views.axis_extents(rest),repeated_digits(Views.axis_extents(rest),others,more))) ->
  Coordinates.DigitsBelow(size <> Views.axis_extents(rest),repeated_digits(size <> Views.axis_extents(rest),extent <> others,digit <> more)):
  match axis_found rest_found:
    case Some{+axis} Some{+found}:
      KernelGemm.Both{digit_below(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent,digit,axis,axis_seen,{==},{==},below),inner(found,rest_seen)}
    case Some{+axis} None{}:
      Empty.absurd(Coordinates.DigitsBelow(size <> Views.axis_extents(rest),repeated_digits(size <> Views.axis_extents(rest),extent <> others,digit <> more)),
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case None{} +rest_found:
      Empty.absurd(Coordinates.DigitsBelow(size <> Views.axis_extents(rest),repeated_digits(size <> Views.axis_extents(rest),extent <> others,digit <> more)),
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))

# The digits a stretch repeats of digits below the target lie below the axes.
def repeated_below(axes: List<&2,Views.Axis>,target: List<&2,U32>,digits: List<&2,Nat>,+out: List<&2,Views.Axis>,
  +seen: {Views.stretch_axes(axes,target) == Some{out} : Maybe<&2,List<&2,Views.Axis>>},+below: Coordinates.DigitsBelow(target,digits)) ->
  Coordinates.DigitsBelow(Views.axis_extents(axes),repeated_digits(Views.axis_extents(axes),target,digits)):
  match axes target digits:
    case Nil{} Nil{} Nil{}: Unit{}
    case Nil{} Nil{} Con{+digit,+more}: Empty.absurd(Coordinates.DigitsBelow([],repeated_digits([],[],digit <> more)),below)
    case Nil{} Con{+extent,+others} +digits:
      Empty.absurd(Coordinates.DigitsBelow([],repeated_digits([],extent <> others,digits)),
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case Con{+axis,+rest} Nil{} +digits:
      Empty.absurd(Coordinates.DigitsBelow(Views.axis_extents(axis <> rest),repeated_digits(Views.axis_extents(axis <> rest),[],digits)),
        WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Bool,found => Maybe.is_some(&2,List<&2,Views.Axis>,found),None{},Some{out},seen)))
    case Con{Views.Axis{+size,+stride},+rest} Con{+extent,+others} Nil{}:
      Empty.absurd(Coordinates.DigitsBelow(size <> Views.axis_extents(rest),repeated_digits(size <> Views.axis_extents(rest),extent <> others,[])),below)
    case Con{Views.Axis{+size,+stride},+rest} Con{+extent,+others} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      prepend_below(Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent),Views.stretch_axes(rest,others),size,stride,extent,rest,others,
        digit,more,out,{==},{==},seen,here,found => found_seen => repeated_below(rest,others,more,found,found_seen,later))

# ---- The word a stretched view reads ----

def view_offset(+view: Views.View) -> U32:
  Views.View{+axes,+offset} = view
  offset

def stretched_axes_of(found: Maybe<&2,List<&2,Views.Axis>>,+source_offset: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  +seen: {Views.stretched_view(found,source_offset) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}) -> {found == Some{axes} : Maybe<&2,List<&2,Views.Axis>>}:
  match found:
    case Some{+out}:
      +same = Equal.cong(Maybe<&2,Views.View>,Views.View,view => ViewReads.view_or(view,Views.View{axes,offset}),Some{Views.View{out,source_offset}},Some{Views.View{axes,offset}},seen)
      Equal.cong(Views.View,Maybe<&2,List<&2,Views.Axis>>,view => Some{Canonical.axes(view)},Views.View{out,source_offset},Views.View{axes,offset},same)
    case None{}:
      Empty.absurd({None{} == Some{axes} : Maybe<&2,List<&2,Views.Axis>>},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,view => Maybe.is_some(&2,Views.View,view),None{},Some{Views.View{axes,offset}},seen)))

def stretched_offset_of(found: Maybe<&2,List<&2,Views.Axis>>,+source_offset: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  +seen: {Views.stretched_view(found,source_offset) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}) -> {source_offset == offset : U32}:
  match found:
    case Some{+out}:
      +same = Equal.cong(Maybe<&2,Views.View>,Views.View,view => ViewReads.view_or(view,Views.View{axes,offset}),Some{Views.View{out,source_offset}},Some{Views.View{axes,offset}},seen)
      Equal.cong(Views.View,U32,view => view_offset(view),Views.View{out,source_offset},Views.View{axes,offset},same)
    case None{}:
      Empty.absurd({source_offset == offset : U32},
        WordEncoding.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,view => Maybe.is_some(&2,Views.View,view),None{},Some{Views.View{axes,offset}},seen)))

# The row-major view of a shape sums the digits of an index to the index.
def canonical_sum(+source: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+index: Nat,
  +inside: {Nat.is_lt(index,ShapeLaws.elements(source)) == True{} : Bool}) -> {ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],index) == index : Nat}:
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +model_back = Equal.sym(List<&2,U32>,Views.axis_extents(model),source,Canonical.canonical_shape(source,count,exact))
  +inside_model = {%model_back : {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool} inside :
    {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(model))) == True{} : Bool}}
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(model),
    Order.le_lt_trans(0n,index,ShapeLaws.elements(Views.axis_extents(model)),Order.zero_le(index),inside_model))
  +layout = Linear.checked_layout(model,0,nonempty,Canonical.canonical_contiguous(source,count,exact))
  +points = RowMajorCoordinates.points_of(model,index)
  +axes_back = Equal.sym(List<&2,Views.Axis>,RowMajorCoordinates.axes(points),model,RowMajorCoordinates.axes_of(model,index))
  +position_exact = {RowMajorCoordinates.position_of(model,index,0n,inside_model) : {RowMajorCoordinates.position(points,0n) == index : Nat}}
  +inside_points = {%Equal.sym(Nat,RowMajorCoordinates.position(points,0n),index,position_exact) :
      {Nat.is_lt(_,ShapeLaws.elements(Views.axis_extents(RowMajorCoordinates.axes(points)))) == True{} : Bool}
    %axes_back : {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(_))) == True{} : Bool}
    inside_model : {Nat.is_lt(RowMajorCoordinates.position(points,0n),ShapeLaws.elements(Views.axis_extents(RowMajorCoordinates.axes(points)))) == True{} : Bool}}
  +canonical = RowMajorCoordinates.canonical(points,RowMajorCoordinates.valid_of(model,index,inside_model),
    {%axes_back : Linear.Layout(_) layout : Linear.Layout(RowMajorCoordinates.axes(points))},inside_points)
  +weight = RowMajorCoordinates.weight(points)
  Equal.trans(Nat,ViewReads.view_sum(model,[],index),weight,index,Equal.sym(Nat,weight,ViewReads.view_sum(model,[],index),ViewReads.points_sum(model,index)),
    Equal.trans(Nat,weight,Nat.add(weight,0n),index,Equal.sym(Nat,Nat.add(weight,0n),weight,Addition.zero_right(weight)),
      Equal.trans(Nat,Nat.add(weight,0n),RowMajorCoordinates.address(points,0n,False{}),index,
        Equal.sym(Nat,RowMajorCoordinates.address(points,0n,False{}),Nat.add(weight,0n),RowMajorCoordinates.address_weight(points,0n)),
        Equal.trans(Nat,RowMajorCoordinates.address(points,0n,False{}),RowMajorCoordinates.position(points,0n),index,canonical,position_exact))))

# The leading axes the source lacks, of one index and step 0, stretch to the
# axes of the stretched view.
def ranked_axes(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +lacked: Nat,+lacked_exact: {Nat.sub(List.length(&2,U32,target),List.length(&2,Views.Axis,Canonical.axes(Canonical.canonical_view(source,count)))) == lacked : Nat}) ->
  {Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),target) == Some{axes} : Maybe<&2,List<&2,Views.Axis>>}:
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +found = {%Canonical.view(source,count,exact) : {Tensors.stretched(_,target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>} stretched :
    {Views.stretched_view(Views.stretch_axes(Views.with_rank(model,List.length(&2,U32,target)),target),0) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}}
  +ranked_seen = {%lacked_exact : {Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,_,Views.Axis{1,0}),model),target) == Some{axes} : Maybe<&2,List<&2,Views.Axis>>}
    stretched_axes_of(Views.stretch_axes(Views.with_rank(model,List.length(&2,U32,target)),target),0,axes,offset,found) :
    {Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),target) == Some{axes} : Maybe<&2,List<&2,Views.Axis>>}}
  ranked_seen

# Their extents are ones before the source.
def ranked_ones(+source: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+lacked: Nat) ->
  {Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count)))) == List.append(&2,U32,List.replicate(U32,lacked,1),source) : List<&2,U32>}:
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +model_extents = {Canonical.canonical_shape(source,count,exact) : {Views.axis_extents(model) == source : List<&2,U32>}}
  +ranked_extents = Equal.trans(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count)))),List.append(&2,U32,List.replicate(U32,lacked,1),Views.axis_extents(model)),List.append(&2,U32,List.replicate(U32,lacked,1),source),
    ones_extents(lacked,model),Equal.cong(List<&2,U32>,List<&2,U32>,extents => List.append(&2,U32,List.replicate(U32,lacked,1),extents),Views.axis_extents(model),source,model_extents))
  ranked_extents

# The digits a stretch names of digits below the target lie below the source.
def stretched_below_at(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(target,digits),+lacked: Nat,+lacked_exact: {Nat.sub(List.length(&2,U32,target),List.length(&2,Views.Axis,Canonical.axes(Canonical.canonical_view(source,count)))) == lacked : Nat}) ->
  Coordinates.DigitsBelow(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))):
  +ranked_seen = ranked_axes(source,target,count,exact,axes,offset,stretched,lacked,lacked_exact)
  +ranked_extents = ranked_ones(source,count,exact,lacked)
  +repeated = {%ranked_extents : Coordinates.DigitsBelow(_,repeated_digits(_,target,digits))
    repeated_below(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),target,digits,axes,ranked_seen,below) : Coordinates.DigitsBelow(List.append(&2,U32,List.replicate(U32,lacked,1),source),repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits))}
  +below_source = {%ones_skipped(lacked,source,target,digits) : Coordinates.DigitsBelow(source,_)
    ones_below(lacked,source,repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits),repeated) : Coordinates.DigitsBelow(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))}
  below_source

# The view of a source stretched to a target reads at the digits of a target
# index the source cell at the source digits, past the leading axes the
# source lacks.
def stretched_sum_at(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(target,digits),+lacked: Nat,+lacked_exact: {Nat.sub(List.length(&2,U32,target),List.length(&2,Views.Axis,Canonical.axes(Canonical.canonical_view(source,count)))) == lacked : Nat}) ->
  {Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],ShapeLaws.digits_value(target,digits))) == ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))) : Nat}:
  +canonical = Canonical.canonical_view(source,count)
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +found = {%Canonical.view(source,count,exact) : {Tensors.stretched(_,target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>} stretched :
    {Views.stretched_view(Views.stretch_axes(Views.with_rank(model,List.length(&2,U32,target)),target),0) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>}}
  +ranked_seen = ranked_axes(source,target,count,exact,axes,offset,stretched,lacked,lacked_exact)
  +offset_zero = stretched_offset_of(Views.stretch_axes(Views.with_rank(model,List.length(&2,U32,target)),target),0,axes,offset,found)
  +index = ShapeLaws.digits_value(target,digits)
  +target_extents = ViewReads.stretch_extents(Tensors.view_of(source),target,axes,offset,stretched)
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,target_extents)
  +below_axes = {%back : Coordinates.DigitsBelow(_,digits) below : Coordinates.DigitsBelow(Views.axis_extents(axes),digits)}
  +inside_axes = {%back : {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool} Coordinates.digits_below(target,digits,below) :
    {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +model_extents = {Canonical.canonical_shape(source,count,exact) : {Views.axis_extents(model) == source : List<&2,U32>}}
  +ranked_extents = ranked_ones(source,count,exact,lacked)
  +below_source = stretched_below_at(source,target,count,exact,axes,offset,stretched,digits,below,lacked,lacked_exact)
  +model_back = Equal.sym(List<&2,U32>,Views.axis_extents(model),source,model_extents)
  +below_model = {%model_back : Coordinates.DigitsBelow(_,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))) below_source : Coordinates.DigitsBelow(Views.axis_extents(model),repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))}
  +inside_model = {%model_back : {Nat.is_lt(ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),ShapeLaws.elements(_)) == True{} : Bool} Coordinates.digits_below(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)),below_source) :
    {Nat.is_lt(ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),ShapeLaws.elements(Views.axis_extents(model))) == True{} : Bool}}
  +step_offset = {%offset_zero : {Nat.add(U32.to_nat(_),ViewReads.view_sum(axes,[],index)) == ViewReads.view_sum(axes,[],index) : Nat} {==} : {Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],index)) == ViewReads.view_sum(axes,[],index) : Nat}}
  +step_digits = {%target_extents : {ViewReads.view_sum(axes,[],ShapeLaws.digits_value(_,digits)) == KeptReads.digit_sum(axes,[],digits) : Nat}
    KeptReads.view_sum_digits(axes,[],digits,below_axes) : {ViewReads.view_sum(axes,[],index) == KeptReads.digit_sum(axes,[],digits) : Nat}}
  +step_stretch = stretch_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),target,digits,axes,ranked_seen)
  +step_extents = Equal.cong(List<&2,U32>,Nat,extents => KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),[],repeated_digits(extents,target,digits)),
    Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count)))),List.append(&2,U32,List.replicate(U32,lacked,1),source),ranked_extents)
  +step_ones = ones_sum(lacked,model,repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits))
  +step_skip = Equal.cong(List<&2,Nat>,Nat,more => KeptReads.digit_sum(model,[],more),skip(lacked,repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits)),repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)),
    ones_skipped(lacked,source,target,digits))
  +step_model = {%model_extents : {KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))) == ViewReads.view_sum(model,[],ShapeLaws.digits_value(_,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))) : Nat}
    Equal.sym(Nat,ViewReads.view_sum(model,[],ShapeLaws.digits_value(Views.axis_extents(model),repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),KeptReads.view_sum_digits(model,[],repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)),below_model)) :
    {KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))) == ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))) : Nat}}
  +sums = Equal.trans(Nat,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],index)),ViewReads.view_sum(axes,[],index),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_offset,
    Equal.trans(Nat,ViewReads.view_sum(axes,[],index),KeptReads.digit_sum(axes,[],digits),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_digits,
      Equal.trans(Nat,KeptReads.digit_sum(axes,[],digits),KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),[],repeated_digits(Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count)))),target,digits)),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_stretch,
        Equal.trans(Nat,KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),[],repeated_digits(Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count)))),target,digits)),KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),[],repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits)),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_extents,
          Equal.trans(Nat,KeptReads.digit_sum(List.append(&2,Views.Axis,List.replicate(Views.Axis,lacked,Views.Axis{1,0}),Canonical.axes(Canonical.canonical_view(source,count))),[],repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits)),KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],skip(lacked,repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits))),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_ones,
            Equal.trans(Nat,KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],skip(lacked,repeated_digits(List.append(&2,U32,List.replicate(U32,lacked,1),source),target,digits))),KeptReads.digit_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),step_skip,step_model))))))
  Equal.trans(Nat,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],index)),ViewReads.view_sum(Canonical.axes(Canonical.canonical_view(source,count)),[],ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)))),ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),sums,canonical_sum(source,count,exact,ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits))),Coordinates.digits_below(source,repeated_digits(source,skip_extents(lacked,target),skip(lacked,digits)),below_source)))

# The row-major axes of a source have its rank.
def model_length(+source: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat}) -> {List.length(&2,Views.Axis,Canonical.axes(Canonical.canonical_view(source,count))) == List.length(&2,U32,source) : Nat}:
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  Equal.trans(Nat,List.length(&2,Views.Axis,model),List.length(&2,U32,Views.axis_extents(model)),List.length(&2,U32,source),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(model)),List.length(&2,Views.Axis,model),MetadataLaws.extents_length(model)),
    Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),Views.axis_extents(model),source,Canonical.canonical_shape(source,count,exact)))

# The view of a source stretched to a target sums the digits of a target index
# to the row-major index of the source digits the stretch names.
def stretched_sum(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(target,digits)) -> {Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],ShapeLaws.digits_value(target,digits))) == ShapeLaws.digits_value(source,source_digits(source,target,digits)) : Nat}:
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +rank = List.length(&2,U32,target)
  +model_length = model_length(source,count,exact)
  %Equal.cong(Nat,Nat,length => Nat.sub(rank,length),List.length(&2,Views.Axis,model),List.length(&2,U32,source),model_length) :
    {Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],ShapeLaws.digits_value(target,digits))) == ShapeLaws.digits_value(source,repeated_digits(source,skip_extents(_,target),skip(_,digits))) : Nat}
  stretched_sum_at(source,target,count,exact,axes,offset,stretched,digits,below,Nat.sub(rank,List.length(&2,Views.Axis,model)),{==})

# The digits a stretch names lie below the source.
def stretched_below(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(target,digits)) -> Coordinates.DigitsBelow(source,source_digits(source,target,digits)):
  +model = Canonical.axes(Canonical.canonical_view(source,count))
  +rank = List.length(&2,U32,target)
  +model_length = model_length(source,count,exact)
  %Equal.cong(Nat,Nat,length => Nat.sub(rank,length),List.length(&2,Views.Axis,model),List.length(&2,U32,source),model_length) :
    Coordinates.DigitsBelow(source,repeated_digits(source,skip_extents(_,target),skip(_,digits)))
  stretched_below_at(source,target,count,exact,axes,offset,stretched,digits,below,Nat.sub(rank,List.length(&2,Views.Axis,model)),{==})

# The last index of every axis.
def last_digits(axes: List<&2,Views.Axis>) -> List<&2,Nat>:
  match axes:
    case []: []
    case Views.Axis{+extent,+stride} <> rest: U32.to_nat((extent - 1 : U32)) <> last_digits(rest)

# A view sums its last digits to its last span.
def last_sum(axes: List<&2,Views.Axis>) -> {KeptReads.digit_sum(axes,[],last_digits(axes)) == ViewBounds.last_span(axes) : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +rest:
      Equal.cong(Nat,Nat,tail => Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),tail),KeptReads.digit_sum(rest,[],last_digits(rest)),
        ViewBounds.last_span(rest),last_sum(rest))

# The last digits of nonzero extents lie below them.
def last_below(axes: List<&2,Views.Axis>,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  Coordinates.DigitsBelow(Views.axis_extents(axes),last_digits(axes)):
  match axes:
    case []: Unit{}
    case Views.Axis{+extent,+stride} <> +rest:
      KernelGemm.Both{Bounds.as_nat_lt((extent - 1 : U32),extent,
          Descriptor.predecessor_inside(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))),
        last_below(rest,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))}

# The view of a source stretched to nonzero target extents fits the source.
def stretched_fits(+source: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},+axes: List<&2,Views.Axis>,
  +offset: U32,+stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonempty: {Shapes.has_zero(target) == False{} : Bool}) -> {Views.fits(Views.View{axes,offset},count) == True{} : Bool}:
  +target_extents = ViewReads.stretch_extents(Tensors.view_of(source),target,axes,offset,stretched)
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,target_extents)
  +nonempty_axes = {%back : {Shapes.has_zero(_) == False{} : Bool} nonempty : {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}}
  +last = last_digits(axes)
  +below_axes = last_below(axes,nonempty_axes)
  +below_target = {%target_extents : Coordinates.DigitsBelow(_,last) below_axes : Coordinates.DigitsBelow(target,last)}
  +index = ShapeLaws.digits_value(target,last)
  +sum = stretched_sum(source,target,count,exact,axes,offset,stretched,last,below_target)
  +digits_sum = {%target_extents : {ViewReads.view_sum(axes,[],ShapeLaws.digits_value(_,last)) == KeptReads.digit_sum(axes,[],last) : Nat}
    KeptReads.view_sum_digits(axes,[],last,below_axes) : {ViewReads.view_sum(axes,[],index) == KeptReads.digit_sum(axes,[],last) : Nat}}
  +span = Equal.trans(Nat,ViewReads.view_sum(axes,[],index),KeptReads.digit_sum(axes,[],last),ViewBounds.last_span(axes),digits_sum,last_sum(axes))
  +named = source_digits(source,target,last)
  +inside = Coordinates.digits_below(source,named,stretched_below(source,target,count,exact,axes,offset,stretched,last,below_target))
  +inside_count = {%exact : {Nat.is_lt(ShapeLaws.digits_value(source,named),_) == True{} : Bool} inside :
    {Nat.is_lt(ShapeLaws.digits_value(source,named),U32.to_nat(count)) == True{} : Bool}}
  +inside_sum = {%Equal.sym(Nat,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],index)),ShapeLaws.digits_value(source,named),sum) :
      {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
    inside_count : {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,[],index)),U32.to_nat(count)) == True{} : Bool}}
  +bound = {%Equal.cong(Nat,Nat,spanned => Nat.add(U32.to_nat(offset),spanned),ViewReads.view_sum(axes,[],index),ViewBounds.last_span(axes),span) :
      {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
    inside_sum : {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewBounds.last_span(axes)),U32.to_nat(count)) == True{} : Bool}}
  ViewBounds.fits_span(axes,offset,count,nonempty_axes,bound)

# ---- Extents that stretch to a target ----
