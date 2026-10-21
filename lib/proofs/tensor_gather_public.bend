# The layout operations that are one gather of a view of their input:
# transpose, slice, nearest upsampling and flip. For an input of any layout,
# when the call returns Ready, the result holds at every index the input at
# the address the view the operation computes gives to that index, with the
# axes flip lists read in reverse.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_global_shape_count.bend as Global
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_view_values.bend as Values
import ./tensor_band_model.bend as Model
import ./word_encoding.bend as WordEncoding
import ../tensor_shape.bend as Shapes
import ./tensor_view_metadata_refinement.bend as Metadata
import ./checked_word_refinement.bend as Checked
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean

def Gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~GatherValues.Gather<Logical>,~GatherValues.value(~Element,~Logical,~original),GatherValues.Gather{logical,source},tensor)

# A Ready result of a view an operation computes has a source view, and its
# values are read through it.
def Viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,source: Maybe<&2,Views.View>,-tensor: Tensors.Tensor<Element>) -> Data:
  match source:
    case Some{source}: Gathered(~Element,~Logical,~original,logical,source,tensor)
    case None{}: Empty

# The same, with the flagged axes read in reverse.
def Directed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,
  -tensor: Tensors.Tensor<Element>) -> Data:
  match source:
    case Some{source}: Values.Gathered(~Element,~Logical,~original,logical,source,reversed,tensor)
    case None{}: Empty

# The gather labelled with the extents of the view it reads through.
def extents_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,+name: String,+reversed: List<&2,Bool>,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_extents(~Element,name,Tensors.Usable{},source,reversed,bands,geometry,shape,fill,workers)) == True{} : Bool}) ->
  Directed(~Element,~Logical,~original,logical,source,reversed,Tensors.gather_extents(~Element,name,Tensors.Usable{},source,reversed,bands,geometry,shape,fill,workers)):
  match source:
    case Some{Views.View{+axes,+offset}}:
      Values.gather(~Element,~Logical,~original,logical,name,axes,offset,reversed,bands,geometry,shape,Views.axis_extents(axes),fill,workers,contract,{==},accepted)
    case None{}: WordEncoding.false_true(accepted)

# The same with no flag, read forward.
def forward_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,+name: String,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_extents(~Element,name,Tensors.Usable{},source,[],bands,geometry,shape,fill,workers)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,source,Tensors.gather_extents(~Element,name,Tensors.Usable{},source,[],bands,geometry,shape,fill,workers)):
  match source:
    case Some{Views.View{+axes,+offset}}:
      Values.forward_gather(~Element,~Logical,~original,logical,name,axes,offset,bands,geometry,shape,Views.axis_extents(axes),fill,workers,Model.observe(~Element,bands),
        contract,{==},accepted)
    case None{}: WordEncoding.false_true(accepted)

# transpose: the result's axis i is the input axis order[i].
def transpose(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+order: List<&2,U32>,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.transpose(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},order)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),
    Layout.transpose(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},order)):
  forward_gather(~Element,~Logical,~original,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),logical,"transpose",bands,geometry,shape,fill,workers,contract,accepted)

# slice keeps count elements of an axis from start, step apart.
def slice(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axis: U32,+start: U32,+count: U32,+step: U32,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.slice(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis,start,count,step)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.sliced(Tensors.view_of(shape),axis,start,count,step),
    Layout.slice(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis,start,count,step)):
  forward_gather(~Element,~Logical,~original,Layout.sliced(Tensors.view_of(shape),axis,start,count,step),logical,"slice",bands,geometry,shape,fill,workers,contract,accepted)

# ---- Nearest upsampling ----

# The extents of the repeated view: each of the last two extents is followed
# by a scale axis of stride zero.
def repeated_extents(extents: List<&2,U32>,+scale: U32) -> List<&2,U32>:
  match extents:
    case [height,width]: [height,scale,width,scale]
    case head <> tail: head <> repeated_extents(tail,scale)
    case []: []

def repeated_extents_or(result: Maybe<&2,List<&2,Views.Axis>>,fallback: List<&2,U32>) -> List<&2,U32>:
  match result:
    case Some{axes}: Views.axis_extents(axes)
    case None{}: fallback

def prepended_extents(result: Maybe<&2,List<&2,Views.Axis>>,+extent: U32,+stride: U32,+fallback: List<&2,U32>,
  previous: {repeated_extents_or(result,fallback) == fallback : List<&2,U32>}) ->
  {repeated_extents_or(Layout.prepend_repeated(Views.Axis{extent,stride},result),extent <> fallback) == extent <> fallback : List<&2,U32>}:
  match result:
    case Some{+axes}: Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(axes),fallback,previous)
    case None{}: {==}

def repeated_axes_extents(+axes: List<&2,Views.Axis>,+scale: U32) ->
  {repeated_extents_or(Layout.repeated_axes(axes,scale),repeated_extents(Views.axis_extents(axes),scale)) ==
    repeated_extents(Views.axis_extents(axes),scale) : List<&2,U32>}:
  match axes:
    case [Views.Axis{height,height_stride},Views.Axis{width,width_stride}]: {==}
    case Con{Views.Axis{extent,stride},Nil{}}: {==}
    case Con{Views.Axis{+extent,+stride},Con{Views.Axis{+second,+second_stride},Con{Views.Axis{+third,+third_stride},+rest}}}:
      +tail = {Views.Axis{second,second_stride} <> Views.Axis{third,third_stride} <> rest : List<&2,Views.Axis>}
      prepended_extents(Layout.repeated_axes(tail,scale),extent,stride,repeated_extents(Views.axis_extents(tail),scale),repeated_axes_extents(tail,scale))
    case []: {==}

def has_view(view: Maybe<&2,Views.View>) -> Bool:
  match view:
    case Some{view}: True{}
    case None{}: False{}

def offset_extents(result: Maybe<&2,List<&2,Views.Axis>>,+view: Views.View,+offset: U32,+fallback: List<&2,U32>,
  previous: {repeated_extents_or(result,fallback) == fallback : List<&2,U32>},
  actual: {Layout.with_offset(result,offset) == Some{view} : Maybe<&2,Views.View>}) -> {Views.axis_extents_of(view) == fallback : List<&2,U32>}:
  match result:
    case Some{+axes}:
      Views.View{+view_axes,+view_offset} = view
      Equal.trans(List<&2,U32>,Views.axis_extents(view_axes),Views.axis_extents(axes),fallback,
        Equal.sym(List<&2,U32>,Views.axis_extents(axes),Views.axis_extents(view_axes),
          Equal.cong(Maybe<&2,Views.View>,List<&2,U32>,found => Metadata.observed_extents(found,fallback),Some{Views.View{axes,offset}},Some{Views.View{view_axes,view_offset}},actual)),previous)
    case None{}:
      Empty.absurd({Views.axis_extents_of(view) == fallback : List<&2,U32>},WordEncoding.false_true(
        Equal.cong(Maybe<&2,Views.View>,Bool,has_view,None{},Some{view},actual)))

# The repeated view has the repeated extents of the shape.
def repeated_view_extents(base: Maybe<&2,Views.View>,+shape: List<&2,U32>,+scale: U32,+view: Views.View,
  +based: {Tensors.view_of(shape) == base : Maybe<&2,Views.View>},
  actual: {Layout.repeated(base,scale) == Some{view} : Maybe<&2,Views.View>}) -> {Views.axis_extents_of(view) == repeated_extents(shape,scale) : List<&2,U32>}:
  match base:
    case Some{Views.View{+axes,+offset}}:
      +axes_shape = {
        %based : {Metadata.observed_extents(_,shape) == shape : List<&2,U32>}
        Metadata.contiguous_shape(shape,0) : {Views.axis_extents(axes) == shape : List<&2,U32>}}
      %axes_shape : {Views.axis_extents_of(view) == repeated_extents(_,scale) : List<&2,U32>}
      offset_extents(Layout.repeated_axes(axes,scale),view,offset,repeated_extents(Views.axis_extents(axes),scale),repeated_axes_extents(axes,scale),actual)
    case None{}:
      Empty.absurd({Views.axis_extents_of(view) == repeated_extents(shape,scale) : List<&2,U32>},WordEncoding.false_true(
        Equal.cong(Maybe<&2,Views.View>,Bool,has_view,None{},Some{view},actual)))

def found(+result: Maybe<&2,U32>,truth: {Views.has_count(result) == True{} : Bool}) -> {result == Some{Checked.value(result)} : Maybe<&2,U32>}:
  match result:
    case None{}: Empty.absurd({None{} == Some{0} : Maybe<&2,U32>},WordEncoding.false_true(truth))
    case Some{value}: {==}

def scaled_exact(+extent: U32,+scale: U32,truth: {Views.has_count(Shapes.checked_multiply(extent,scale)) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat(extent),U32.to_nat(scale)) == U32.to_nat((extent * scale : U32)) : Nat}:
  +value = Checked.value(Shapes.checked_multiply(extent,scale))
  +exact = Checked.multiply_sound(extent,scale,value,found(Shapes.checked_multiply(extent,scale),truth))
  Equal.sym(Nat,U32.to_nat((extent * scale : U32)),Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),
    FullRange.multiply(extent,scale,value,
      %Equal.sym(Nat,Nat.mul(U32.to_nat(extent),U32.to_nat(scale)),U32.to_nat(value),exact) : {Nat.is_le(_,U32.to_nat(value)) == True{} : Bool}
      Order.reflexive(U32.to_nat(value))))

# A shape of rank two or more scales and repeats its last two axes.
def scaled_planes(+shape: List<&2,U32>,+scale: U32,ranked: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Layout.scaled_extents(shape,scale) == List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),
    [(Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)) * scale : U32),(Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)) * scale : U32)]) : List<&2,U32>}:
  match shape:
    case []: Empty.absurd({Layout.scaled_extents([],scale) == [(0 * scale : U32),(0 * scale : U32)] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a]: Empty.absurd({Layout.scaled_extents([a],scale) == [(a * scale : U32),(a * scale : U32)] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a,+b]: {==}
    case +a <> +b <> +c <> +rest:
      +remaining = List.length(&2,U32,rest)
      +tail = {b <> c <> rest : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,value => a <> value,Layout.scaled_extents(tail,scale),
        List.append(&2,U32,List.take(&2,U32,tail,remaining),[(Layout.extent_at(tail,remaining) * scale : U32),(Layout.extent_at(tail,1n+remaining) * scale : U32)]),
        %Order.subtract_zero(remaining) :
          {Layout.scaled_extents(tail,scale) == List.append(&2,U32,List.take(&2,U32,tail,_),[(Layout.extent_at(tail,_) * scale : U32),(Layout.extent_at(tail,1n+remaining) * scale : U32)]) : List<&2,U32>}
        scaled_planes(tail,scale,Order.zero_le(remaining)))

def repeated_planes(+shape: List<&2,U32>,+scale: U32,ranked: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {repeated_extents(shape,scale) == List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),
    [Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),scale,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)),scale]) : List<&2,U32>}:
  match shape:
    case []: Empty.absurd({repeated_extents([],scale) == [0,scale,0,scale] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a]: Empty.absurd({repeated_extents([a],scale) == [a,scale,a,scale] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a,+b]: {==}
    case +a <> +b <> +c <> +rest:
      +remaining = List.length(&2,U32,rest)
      +tail = {b <> c <> rest : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,value => a <> value,repeated_extents(tail,scale),
        List.append(&2,U32,List.take(&2,U32,tail,remaining),[Layout.extent_at(tail,remaining),scale,Layout.extent_at(tail,1n+remaining),scale]),
        %Order.subtract_zero(remaining) :
          {repeated_extents(tail,scale) == List.append(&2,U32,List.take(&2,U32,tail,_),[Layout.extent_at(tail,_),scale,Layout.extent_at(tail,1n+remaining),scale]) : List<&2,U32>}
        repeated_planes(tail,scale,Order.zero_le(remaining)))

def merged_pair_elements(+height: U32,+width: U32,+scale: U32,
  +height_exact: {Nat.mul(U32.to_nat(height),U32.to_nat(scale)) == U32.to_nat((height * scale : U32)) : Nat},
  +width_exact: {Nat.mul(U32.to_nat(width),U32.to_nat(scale)) == U32.to_nat((width * scale : U32)) : Nat}) ->
  {ShapeLaws.elements([height,scale,width,scale]) == ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]) : Nat}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(scale),1n),U32.to_nat(scale),Order.times_one(U32.to_nat(scale))) :
    {Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(width),_))) == ShapeLaws.elements([(height * scale : U32),(width * scale : U32)]) : Nat}
  %Equal.sym(Nat,Nat.mul(U32.to_nat((width * scale : U32)),1n),U32.to_nat((width * scale : U32)),Order.times_one(U32.to_nat((width * scale : U32)))) :
    {Nat.mul(U32.to_nat(height),Nat.mul(U32.to_nat(scale),Nat.mul(U32.to_nat(width),U32.to_nat(scale)))) == Nat.mul(U32.to_nat((height * scale : U32)),_) : Nat}
  %RowMajor.assoc(U32.to_nat(height),U32.to_nat(scale),Nat.mul(U32.to_nat(width),U32.to_nat(scale))) :
    {_ == Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32))) : Nat}
  Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(height),U32.to_nat(scale)),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),
    Nat.mul(U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(width),U32.to_nat(scale))),Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat((width * scale : U32))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,Nat.mul(U32.to_nat(width),U32.to_nat(scale))),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),U32.to_nat((height * scale : U32)),height_exact),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat((height * scale : U32)),value),Nat.mul(U32.to_nat(width),U32.to_nat(scale)),U32.to_nat((width * scale : U32)),width_exact))

# When the scaled rows and columns fit a word, the scaled extents have as many
# elements as the repeated ones.
def scaled_elements(+shape: List<&2,U32>,+scale: U32,+ranked: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool},
  +height_exact: {Nat.mul(U32.to_nat(Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n))),U32.to_nat(scale)) ==
    U32.to_nat((Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)) * scale : U32)) : Nat},
  +width_exact: {Nat.mul(U32.to_nat(Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))),U32.to_nat(scale)) ==
    U32.to_nat((Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)) * scale : U32)) : Nat}) ->
  {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == ShapeLaws.elements(repeated_extents(shape,scale)) : Nat}:
  +rank = List.length(&2,U32,shape)
  +height = Layout.extent_at(shape,Nat.sub(rank,2n))
  +width = Layout.extent_at(shape,Nat.sub(rank,1n))
  +leading = List.take(&2,U32,shape,Nat.sub(rank,2n))
  +scaled = {[(height * scale : U32),(width * scale : U32)] : List<&2,U32>}
  +repeated = {[height,scale,width,scale] : List<&2,U32>}
  +leading_elements = ShapeLaws.elements(leading)
  Equal.trans(Nat,ShapeLaws.elements(Layout.scaled_extents(shape,scale)),Nat.mul(leading_elements,ShapeLaws.elements(scaled)),ShapeLaws.elements(repeated_extents(shape,scale)),
    %Equal.sym(List<&2,U32>,Layout.scaled_extents(shape,scale),List.append(&2,U32,leading,scaled),scaled_planes(shape,scale,ranked)) :
      {ShapeLaws.elements(_) == Nat.mul(leading_elements,ShapeLaws.elements(scaled)) : Nat}
    ShapeLaws.elements_appended(leading,scaled),
    Equal.trans(Nat,Nat.mul(leading_elements,ShapeLaws.elements(scaled)),Nat.mul(leading_elements,ShapeLaws.elements(repeated)),ShapeLaws.elements(repeated_extents(shape,scale)),
      Equal.cong(Nat,Nat,value => Nat.mul(leading_elements,value),ShapeLaws.elements(scaled),ShapeLaws.elements(repeated),
        Equal.sym(Nat,ShapeLaws.elements(repeated),ShapeLaws.elements(scaled),merged_pair_elements(height,width,scale,height_exact,width_exact))),
      %Equal.sym(List<&2,U32>,repeated_extents(shape,scale),List.append(&2,U32,leading,repeated),repeated_planes(shape,scale,ranked)) :
        {Nat.mul(leading_elements,ShapeLaws.elements(repeated)) == ShapeLaws.elements(_) : Nat}
      Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading,repeated)),Nat.mul(leading_elements,ShapeLaws.elements(repeated)),ShapeLaws.elements_appended(leading,repeated))))

# A shape with planes has rank two or more.
def leading_ranked(ranked: Bool,+shape: List<&2,U32>,+classified: {Nat.is_le(2n,List.length(&2,U32,shape)) == ranked : Bool},+count: U32,
  found: {Layout.leading(ranked,shape) == Some{count} : Maybe<&2,U32>}) -> {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}:
  match ranked:
    case True{}: classified
    case False{}:
      Empty.absurd({Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool},WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,None{},Some{count},found)))

# The gather through the repeated view, labelled with the scaled extents.
def repeated_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,source: Maybe<&2,Views.View>,+logical: Logical,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  +sourced: {Layout.repeated(Tensors.view_of(shape),scale) == source : Maybe<&2,Views.View>},
  +same: {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == ShapeLaws.elements(repeated_extents(shape,scale)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather(~Element,"upsample_nearest",Tensors.Usable{},source,[],bands,geometry,shape,Layout.scaled_extents(shape,scale),fill,workers)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,source,Tensors.gather(~Element,"upsample_nearest",Tensors.Usable{},source,[],bands,geometry,shape,Layout.scaled_extents(shape,scale),fill,workers)):
  match source:
    case Some{Views.View{+axes,+offset}}:
      +extents = repeated_view_extents(Tensors.view_of(shape),shape,scale,Views.View{axes,offset},{==},sourced)
      Values.forward_gather(~Element,~Logical,~original,logical,"upsample_nearest",axes,offset,bands,geometry,shape,Layout.scaled_extents(shape,scale),fill,workers,
        Model.observe(~Element,bands),contract,
        %Equal.sym(List<&2,U32>,Views.axis_extents(axes),repeated_extents(shape,scale),extents) :
          {ShapeLaws.elements(Layout.scaled_extents(shape,scale)) == ShapeLaws.elements(_) : Nat}
        same,accepted)
    case None{}: WordEncoding.false_true(accepted)

def checked_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,valid: Bool,planes: Maybe<&2,U32>,+logical: Logical,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+scale: U32,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  +checks: {Layout.upsample_valid(shape,scale) == valid : Bool},+counted: {Layout.planes(shape) == planes : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Layout.upsample_checked(~Element,valid,planes,bands,geometry,shape,scale,fill,workers)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.repeated(Tensors.view_of(shape),scale),Layout.upsample_checked(~Element,valid,planes,bands,geometry,shape,scale,fill,workers)):
  match valid planes:
    case True{} Some{+count}:
      +rank = List.length(&2,U32,shape)
      +height_ok = Views.has_count(Shapes.checked_multiply(Layout.extent_at(shape,Nat.sub(rank,2n)),scale))
      +width_ok = Views.has_count(Shapes.checked_multiply(Layout.extent_at(shape,Nat.sub(rank,1n)),scale))
      +pair = Boolean.left((height_ok && width_ok : Bool),Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale))),checks)
      +ranked = leading_ranked(Nat.is_le(2n,rank),shape,{==},count,counted)
      repeated_gather(~Element,~Logical,~original,Layout.repeated(Tensors.view_of(shape),scale),logical,bands,geometry,shape,scale,fill,workers,contract,{==},
        scaled_elements(shape,scale,ranked,scaled_exact(Layout.extent_at(shape,Nat.sub(rank,2n)),scale,Boolean.left(height_ok,width_ok,pair)),
          scaled_exact(Layout.extent_at(shape,Nat.sub(rank,1n)),scale,Boolean.right(height_ok,width_ok,pair))),accepted)
    case True{} None{}:
      Empty.absurd(Viewed(~Element,~Logical,~original,logical,Layout.repeated(Tensors.view_of(shape),scale),
        Layout.upsample_checked(~Element,True{},None{},bands,geometry,shape,scale,fill,workers)),WordEncoding.false_true(accepted))
    case False{} planes:
      Empty.absurd(Viewed(~Element,~Logical,~original,logical,Layout.repeated(Tensors.view_of(shape),scale),
        Layout.upsample_checked(~Element,False{},planes,bands,geometry,shape,scale,fill,workers)),WordEncoding.false_true(accepted))

# upsample_nearest repeats each element of the last two axes scale times.
def upsample_nearest(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+scale: U32,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.upsample_nearest(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},scale)) == True{} : Bool}) ->
  Viewed(~Element,~Logical,~original,logical,Layout.repeated(Tensors.view_of(shape),scale),
    Layout.upsample_nearest(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},scale)):
  checked_gather(~Element,~Logical,~original,Layout.upsample_valid(shape,scale),Layout.planes(shape),logical,bands,geometry,shape,scale,fill,workers,contract,{==},{==},accepted)

# ---- Flip ----

def flip_checked(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,valid: Bool,+logical: Logical,+axes: List<&2,U32>,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_extents(~Element,"flip",Tensors.Usable{},Layout.flipped(valid,Tensors.view_of(shape)),
    Layout.reversed_flags(List.length(&2,U32,shape),0,axes),bands,geometry,shape,fill,workers)) == True{} : Bool}) ->
  Directed(~Element,~Logical,~original,logical,Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
    Tensors.gather_extents(~Element,"flip",Tensors.Usable{},Layout.flipped(valid,Tensors.view_of(shape)),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
      bands,geometry,shape,fill,workers)):
  match valid:
    case True{}:
      extents_gather(~Element,~Logical,~original,Tensors.view_of(shape),logical,"flip",Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
        bands,geometry,shape,fill,workers,contract,accepted)
    case False{}:
      Empty.absurd(Directed(~Element,~Logical,~original,logical,Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
        Tensors.gather_extents(~Element,"flip",Tensors.Usable{},None{},Layout.reversed_flags(List.length(&2,U32,shape),0,axes),bands,geometry,shape,fill,workers)),
        WordEncoding.false_true(accepted))

# flip reads the listed axes in reverse: the result holds at every index the
# input at the address its view gives to the digits of that index with the
# digit of every listed axis reflected.
def flip(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axes: List<&2,U32>,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axes)) == True{} : Bool}) ->
  Directed(~Element,~Logical,~original,logical,Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
    Layout.flip(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axes)):
  flip_checked(~Element,~Logical,~original,Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)),logical,axes,
    bands,geometry,shape,fill,workers,contract,accepted)
