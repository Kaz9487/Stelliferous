# Broadcasting a Normal tensor to extents its own extents stretch to gathers
# it through the stretched view into a tensor that holds, at each index, the
# value at the address of that index in that view. Extents that broadcast with others
# stretch to the common extents, and an operand read at them is read as it is
# or through that view.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_contract.bend as Contract
import ./tensor_copy_values.bend as CopyValues
import ./tensor_gather_values.bend as GatherValues
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_storage_count.bend as StorageCount
import ./tensor_gather_public.bend as GatherPublic
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_broadcast.bend as Broadcast
import ./tensor_band_spread.bend as Spread
import ./tensor_canonical_view.bend as Canonical
import ./tensor_view_lanes.bend as Lanes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_whole_storage_values.bend as WholeStorageValues
import ./tensor_band_count.bend as BandCount
import ./tensor_allocation_depth.bend as Depth
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_view_ready.bend as Ready
import ./tensor_row_major_coordinates.bend as Coordinates
import ./word_encoding.bend as WordEncoding
import ./tensor_view_metadata_refinement.bend as Metadata

# An operand read at the common extents: as it is, or through the view its
# own extents stretch to.
type Expanded<-Logical: Data> is Data:
  Expanded{logical: Logical,view: Maybe<&2,Views.View>}

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: Expanded<Logical>,index: U32) -> Element:
  match context:
    case Expanded{logical,None{}}: original(logical,index)
    case Expanded{logical,Some{view}}: original(logical,CopyValues.address(view,U32.to_nat(index)))

def expansion_when(same: Bool,+shape: List<&2,U32>,+target: List<&2,U32>) -> Maybe<&2,Views.View>:
  match same:
    case True{}: None{}
    case False{}: Tensors.stretched(Tensors.view_of(shape),target)

# The view an operand of the extents shape is read through at target.
def expansion(+shape: List<&2,U32>,+target: List<&2,U32>) -> Maybe<&2,Views.View>:
  expansion_when(Shapes.same_extents(shape,target),shape,target)

# The common extents have the rank of both operands.
def Ranks(shape: List<&2,U32>,left: List<&2,U32>,right: List<&2,U32>) -> Data:
  Shape.Both<{List.length(&2,U32,shape) == List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == List.length(&2,U32,right) : Nat}>

def absent(~Value: Data,+value: Value,missing: {None{} == Some{value} : Maybe<&2,Value>}) -> Empty:
  WordEncoding.false_true(Equal.cong(Maybe<&2,Value>,Bool,maybe => Maybe.is_some(&2,Value,maybe),None{},Some{value},missing))

def prepended_lengths(+head: U32,+others: List<&2,U32>,+shape: List<&2,U32>,+left: List<&2,U32>,+right: List<&2,U32>,
  +same: {head <> others == shape : List<&2,U32>},ranks: Ranks(others,left,right)) ->
  Shape.Both<{List.length(&2,U32,shape) == 1n+List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == 1n+List.length(&2,U32,right) : Nat}>:
  Shape.Both{left_rank,right_rank} = ranks
  Shape.Both{
    %same : {List.length(&2,U32,_) == 1n+List.length(&2,U32,left) : Nat}
    Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,others),List.length(&2,U32,left),left_rank),
    %same : {List.length(&2,U32,_) == 1n+List.length(&2,U32,right) : Nat}
    Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,others),List.length(&2,U32,right),right_rank)}

def prepended_rank(extent: Maybe<&2,U32>,rest: Maybe<&2,List<&2,U32>>,+shape: List<&2,U32>,+left: List<&2,U32>,+right: List<&2,U32>,
  common: {Shapes.prepend_extent(extent,rest) == Some{shape} : Maybe<&2,List<&2,U32>>},
  tail: @+tail_shape: List<&2,U32> -> {rest == Some{tail_shape} : Maybe<&2,List<&2,U32>>} -> Ranks(tail_shape,left,right)) ->
  Shape.Both<{List.length(&2,U32,shape) == 1n+List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == 1n+List.length(&2,U32,right) : Nat}>:
  match extent rest:
    case Some{+head} Some{+others}:
      prepended_lengths(head,others,shape,left,right,
        Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,maybe => Maybe.default(&2,List<&2,U32>,maybe,[]),Some{head <> others},Some{shape},common),tail(others,{==}))
    case Some{head} None{}:
      Empty.absurd(Shape.Both<{List.length(&2,U32,shape) == 1n+List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == 1n+List.length(&2,U32,right) : Nat}>,
        absent(~List<&2,U32>,shape,common))
    case None{} Some{others}:
      Empty.absurd(Shape.Both<{List.length(&2,U32,shape) == 1n+List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == 1n+List.length(&2,U32,right) : Nat}>,
        absent(~List<&2,U32>,shape,common))
    case None{} None{}:
      Empty.absurd(Shape.Both<{List.length(&2,U32,shape) == 1n+List.length(&2,U32,left) : Nat},{List.length(&2,U32,shape) == 1n+List.length(&2,U32,right) : Nat}>,
        absent(~List<&2,U32>,shape,common))

def common_rank(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,
  common: {Shapes.broadcast_same_rank(left,right) == Some{shape} : Maybe<&2,List<&2,U32>>}) -> Ranks(shape,left,right):
  match left right:
    case Nil{} Nil{}:
      +same = Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,maybe => Maybe.default(&2,List<&2,U32>,maybe,[]),Some{[]},Some{shape},common)
      +empty = Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),shape,[],Equal.sym(List<&2,U32>,[],shape,same))
      Shape.Both{empty,empty}
    case Con{+x,+xs} Con{+y,+ys}:
      prepended_rank(Shapes.broadcast_extent(U32.is_eq(x,y),U32.is_eq(x,1),U32.is_eq(y,1),x,y),Shapes.broadcast_same_rank(xs,ys),shape,xs,ys,common,
        tail_shape => tail_common => common_rank(xs,ys,tail_shape,tail_common))
    case Nil{} Con{y,ys}: Empty.absurd(Ranks(shape,Nil{},Con{y,ys}),absent(~List<&2,U32>,shape,common))
    case Con{x,xs} Nil{}: Empty.absurd(Ranks(shape,Con{x,xs},Nil{}),absent(~List<&2,U32>,shape,common))

def replicated_length(count: Nat,+value: U32,+extents: List<&2,U32>) ->
  {List.length(&2,U32,List.append(&2,U32,List.replicate(U32,count,value),extents)) == Nat.add(count,List.length(&2,U32,extents)) : Nat}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,List.append(&2,U32,List.replicate(U32,rest,value),extents)),Nat.add(rest,List.length(&2,U32,extents)),
        replicated_length(rest,value,extents))

# The canonical axes of a source have its rank.
def axes_rank(+source: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat}) ->
  {List.length(&2,U32,source) == List.length(&2,Views.Axis,Canonical.axes(Canonical.canonical_view(source,count))) : Nat}:
  +axes = Canonical.axes(Canonical.canonical_view(source,count))
  Equal.trans(Nat,List.length(&2,U32,source),List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),
    Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),source,Views.axis_extents(axes),
      Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(source,count)),source,Canonical.canonical_shape(source,count,exact))),
    Metadata.extents_length(axes))

# Padding to the rank of padded extents pads as much as padding to the rank.
def ranked_axes(+source: List<&2,U32>,+count: U32,+rank: Nat,+shape: List<&2,U32>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  same_rank: {List.length(&2,U32,shape) == List.length(&2,U32,Shapes.with_rank(source,rank)) : Nat}) ->
  {Spread.source_axes(source,count,List.length(&2,U32,shape)) == Spread.source_axes(source,count,rank) : List<&2,Views.Axis>}:
  +axes = Canonical.axes(Canonical.canonical_view(source,count))
  +own = List.length(&2,Views.Axis,axes)
  +own_rank = axes_rank(source,count,exact)
  +padded = Equal.trans(Nat,List.length(&2,U32,shape),List.length(&2,U32,Shapes.with_rank(source,rank)),Nat.add(Nat.sub(rank,own),own),same_rank,
    Equal.trans(Nat,List.length(&2,U32,Shapes.with_rank(source,rank)),Nat.add(Nat.sub(rank,List.length(&2,U32,source)),List.length(&2,U32,source)),Nat.add(Nat.sub(rank,own),own),
      replicated_length(Nat.sub(rank,List.length(&2,U32,source)),1,source),
      Equal.cong(Nat,Nat,length => Nat.add(Nat.sub(rank,length),length),List.length(&2,U32,source),own,own_rank)))
  +shift = Equal.trans(Nat,Nat.sub(List.length(&2,U32,shape),own),Nat.sub(Nat.add(Nat.sub(rank,own),own),own),Nat.sub(rank,own),
    Equal.cong(Nat,Nat,length => Nat.sub(length,own),List.length(&2,U32,shape),Nat.add(Nat.sub(rank,own),own),padded),
    DivisionOrder.subtract_sum(Nat.sub(rank,own),own))
  Equal.cong(Nat,List<&2,Views.Axis>,pad => List.append(&2,Views.Axis,List.replicate(Views.Axis,pad,Views.Axis{1,0}),axes),
    Nat.sub(List.length(&2,U32,shape),own),Nat.sub(rank,own),shift)

# Extents that broadcast with other extents stretch to the common extents.
def Stretches(left: List<&2,Views.Axis>,right: List<&2,Views.Axis>,shape: List<&2,U32>) -> Data:
  Shape.Both<{Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(left,shape)) == True{} : Bool},
    {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(right,shape)) == True{} : Bool}>

def ranked_stretches(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+right_count: U32,
  +left_exact: {ShapeLaws.elements(left) == U32.to_nat(left_count) : Nat},+right_exact: {ShapeLaws.elements(right) == U32.to_nat(right_count) : Nat},
  ranks: Ranks(shape,Shapes.with_rank(left,List.length(&2,U32,right)),Shapes.with_rank(right,List.length(&2,U32,left))),
  compatible: Broadcast.Compatible(Spread.source_axes(left,left_count,List.length(&2,U32,right)),Spread.source_axes(right,right_count,List.length(&2,U32,left)),Some{shape})) ->
  Stretches(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Spread.source_axes(right,right_count,List.length(&2,U32,shape)),shape):
  Shape.Both{left_rank,right_rank} = ranks
  Shape.Both{left_some,right_some} = compatible
  Shape.Both{
    %Equal.sym(List<&2,Views.Axis>,Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Spread.source_axes(left,left_count,List.length(&2,U32,right)),
      ranked_axes(left,left_count,List.length(&2,U32,right),shape,left_exact,left_rank)) :
      {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(_,shape)) == True{} : Bool}
    left_some,
    %Equal.sym(List<&2,Views.Axis>,Spread.source_axes(right,right_count,List.length(&2,U32,shape)),Spread.source_axes(right,right_count,List.length(&2,U32,left)),
      ranked_axes(right,right_count,List.length(&2,U32,left),shape,right_exact,right_rank)) :
      {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(_,shape)) == True{} : Bool}
    right_some}

def stretches(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+right_count: U32,
  +left_exact: {ShapeLaws.elements(left) == U32.to_nat(left_count) : Nat},+right_exact: {ShapeLaws.elements(right) == U32.to_nat(right_count) : Nat},
  +common: {Shapes.broadcast(left,right) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Stretches(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Spread.source_axes(right,right_count,List.length(&2,U32,shape)),shape):
  +left_axes = Spread.source_axes(left,left_count,List.length(&2,U32,right))
  +right_axes = Spread.source_axes(right,right_count,List.length(&2,U32,left))
  ranked_stretches(left,right,shape,left_count,right_count,left_exact,right_exact,
    common_rank(Shapes.with_rank(left,List.length(&2,U32,right)),Shapes.with_rank(right,List.length(&2,U32,left)),shape,common),
    %common : Broadcast.Compatible(left_axes,right_axes,_)
    %Spread.source_extents(right,right_count,List.length(&2,U32,left),right_exact) :
      Broadcast.Compatible(left_axes,right_axes,Shapes.broadcast_same_rank(Shapes.with_rank(left,List.length(&2,U32,right)),_))
    %Spread.source_extents(left,left_count,List.length(&2,U32,right),left_exact) :
      Broadcast.Compatible(left_axes,right_axes,Shapes.broadcast_same_rank(_,Views.axis_extents(right_axes)))
    Broadcast.broadcast_stretches(left_axes,right_axes))

# The axes a source stretches to.
type Stretched<-source: List<&2,U32>,-count: U32,-shape: List<&2,U32>> is Data:
  Stretched{full: List<&2,Views.Axis>,
    equation: {Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>}}

def stretched_when(result: Maybe<&2,List<&2,Views.Axis>>,+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,
  equation: {Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape) == result : Maybe<&2,List<&2,Views.Axis>>},
  some: {Maybe.is_some(&2,List<&2,Views.Axis>,result) == True{} : Bool}) -> Stretched<source,count,shape>:
  match result:
    case Some{+full}: Stretched{full,equation}
    case None{}: Empty.absurd(Stretched<source,count,shape>,WordEncoding.false_true(some))

def stretched_pair(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+right_count: U32,
  stretches: Stretches(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Spread.source_axes(right,right_count,List.length(&2,U32,shape)),shape)) ->
  Shape.Both<Stretched<left,left_count,shape>,Stretched<right,right_count,shape>>:
  Shape.Both{left_some,right_some} = stretches
  Shape.Both{
    stretched_when(Views.stretch_axes(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),shape),left,left_count,shape,{==},left_some),
    stretched_when(Views.stretch_axes(Spread.source_axes(right,right_count,List.length(&2,U32,shape)),shape),right,right_count,shape,{==},right_some)}

# Both extents of a broadcast stretch to the common extents.
def stretched(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+right_count: U32,
  +left_exact: {ShapeLaws.elements(left) == U32.to_nat(left_count) : Nat},+right_exact: {ShapeLaws.elements(right) == U32.to_nat(right_count) : Nat},
  common: {Shapes.broadcast(left,right) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Shape.Both<Stretched<left,left_count,shape>,Stretched<right,right_count,shape>>:
  stretched_pair(left,right,shape,left_count,right_count,stretches(left,right,shape,left_count,right_count,left_exact,right_exact,common))

# The axes of extents read with stride zero.
def axes_of(extents: List<&2,U32>) -> List<&2,Views.Axis>:
  match extents:
    case Nil{}: Nil{}
    case Con{extent,rest}: Views.Axis{extent,0} <> axes_of(rest)

def axes_extents(+extents: List<&2,U32>) -> {Views.axis_extents(axes_of(extents)) == extents : List<&2,U32>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(axes_of(rest)),rest,axes_extents(rest))

def axes_length(+extents: List<&2,U32>) -> {List.length(&2,Views.Axis,axes_of(extents)) == List.length(&2,U32,extents) : Nat}:
  Equal.trans(Nat,List.length(&2,Views.Axis,axes_of(extents)),List.length(&2,U32,Views.axis_extents(axes_of(extents))),List.length(&2,U32,extents),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes_of(extents))),List.length(&2,Views.Axis,axes_of(extents)),Metadata.extents_length(axes_of(extents))),
    Equal.cong(List<&2,U32>,Nat,list => List.length(&2,U32,list),Views.axis_extents(axes_of(extents)),extents,axes_extents(extents)))

# Whether an axis stretches does not depend on its stride.
def axis_some(same: Bool,single: Bool,+size: U32,+stride: U32,+other: U32,+extent: U32,
  some: {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(same,single,size,stride,extent)) == True{} : Bool}) ->
  {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(same,single,size,other,extent)) == True{} : Bool}:
  match same single:
    case True{} True{}: {==}
    case True{} False{}: {==}
    case False{} True{}: {==}
    case False{} False{}: Empty.absurd({Maybe.is_some(&2,Views.Axis,Views.stretched_axis(False{},False{},size,other,extent)) == True{} : Bool},WordEncoding.false_true(some))

# An axis of size one stretches to any extent.
def single_some(same: Bool,+size: U32,+stride: U32,+extent: U32) ->
  {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(same,True{},size,stride,extent)) == True{} : Bool}:
  match same:
    case True{}: {==}
    case False{}: {==}

# An axis that stretches to a middle extent that stretches to an extent
# stretches to that extent.
def axis_through(same: Bool,single: Bool,+size: U32,+stride: U32,+middle: U32,+middle_stride: U32,+extent: U32,
  +same_value: {U32.is_eq(size,middle) == same : Bool},+single_value: {U32.is_eq(size,1) == single : Bool},
  inner: {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(same,single,size,stride,middle)) == True{} : Bool},
  outer: {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(middle,extent),U32.is_eq(middle,1),middle,middle_stride,extent)) == True{} : Bool}) ->
  {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(size,extent),single,size,stride,extent)) == True{} : Bool}:
  match same single:
    case True{} True{}: single_some(U32.is_eq(size,extent),size,stride,extent)
    case False{} True{}: single_some(U32.is_eq(size,extent),size,stride,extent)
    case True{} False{}:
      +equal = Arithmetic.equal_of_is_eq(size,middle,same_value)
      axis_some(U32.is_eq(size,extent),False{},size,middle_stride,stride,extent,
        %single_value : {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(size,extent),_,size,middle_stride,extent)) == True{} : Bool}
        %Equal.sym(U32,size,middle,equal) : {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(_,extent),U32.is_eq(_,1),_,middle_stride,extent)) == True{} : Bool}
        outer)
    case False{} False{}:
      Empty.absurd({Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(size,extent),False{},size,stride,extent)) == True{} : Bool},WordEncoding.false_true(inner))

def prepended_head(axis: Maybe<&2,Views.Axis>,rest: Maybe<&2,List<&2,Views.Axis>>,
  some: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretched_prepend(axis,rest)) == True{} : Bool}) -> {Maybe.is_some(&2,Views.Axis,axis) == True{} : Bool}:
  match axis rest:
    case Some{axis} Some{rest}: {==}
    case Some{axis} None{}: {==}
    case None{} Some{rest}: Empty.absurd({Maybe.is_some(&2,Views.Axis,None{}) == True{} : Bool},WordEncoding.false_true(some))
    case None{} None{}: Empty.absurd({Maybe.is_some(&2,Views.Axis,None{}) == True{} : Bool},WordEncoding.false_true(some))

def prepended_tail(axis: Maybe<&2,Views.Axis>,rest: Maybe<&2,List<&2,Views.Axis>>,
  some: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretched_prepend(axis,rest)) == True{} : Bool}) -> {Maybe.is_some(&2,List<&2,Views.Axis>,rest) == True{} : Bool}:
  match axis rest:
    case Some{axis} Some{rest}: {==}
    case Some{axis} None{}: Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,None{}) == True{} : Bool},WordEncoding.false_true(some))
    case None{} Some{rest}: {==}
    case None{} None{}: Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,None{}) == True{} : Bool},WordEncoding.false_true(some))

# Axes that stretch to the extents of middle axes that stretch to target
# extents stretch to the target extents.
def axes_through(+axes: List<&2,Views.Axis>,+middle: List<&2,Views.Axis>,+target: List<&2,U32>,
  +inner: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(axes,Views.axis_extents(middle))) == True{} : Bool},
  +outer: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(middle,target)) == True{} : Bool}) ->
  {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(axes,target)) == True{} : Bool}:
  match axes middle target:
    case Nil{} Nil{} Nil{}: {==}
    case Nil{} Nil{} Con{extent,rest}:
      Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Nil{},Con{extent,rest})) == True{} : Bool},WordEncoding.false_true(outer))
    case Nil{} Con{Views.Axis{size,stride},tail} target:
      Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Nil{},target)) == True{} : Bool},WordEncoding.false_true(inner))
    case Con{Views.Axis{size,stride},tail} Nil{} target:
      Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Con{Views.Axis{size,stride},tail},target)) == True{} : Bool},WordEncoding.false_true(inner))
    case Con{Views.Axis{size,stride},tail} Con{Views.Axis{middle_size,middle_stride},middle_tail} Nil{}:
      Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Con{Views.Axis{size,stride},tail},Nil{})) == True{} : Bool},WordEncoding.false_true(outer))
    case Con{Views.Axis{+size,+stride},+tail} Con{Views.Axis{+middle_size,+middle_stride},+middle_tail} Con{+extent,+rest}:
      +inner_axis = Views.stretched_axis(U32.is_eq(size,middle_size),U32.is_eq(size,1),size,stride,middle_size)
      +outer_axis = Views.stretched_axis(U32.is_eq(middle_size,extent),U32.is_eq(middle_size,1),middle_size,middle_stride,extent)
      Broadcast.prepended_exists(Views.stretched_axis(U32.is_eq(size,extent),U32.is_eq(size,1),size,stride,extent),Views.stretch_axes(tail,rest),
        axis_through(U32.is_eq(size,middle_size),U32.is_eq(size,1),size,stride,middle_size,middle_stride,extent,{==},{==},
          prepended_head(inner_axis,Views.stretch_axes(tail,Views.axis_extents(middle_tail)),inner),
          prepended_head(outer_axis,Views.stretch_axes(middle_tail,rest),outer)),
        axes_through(tail,middle_tail,rest,
          prepended_tail(inner_axis,Views.stretch_axes(tail,Views.axis_extents(middle_tail)),inner),
          prepended_tail(outer_axis,Views.stretch_axes(middle_tail,rest),outer)))

# Leading axes of extent one added to both stretch to the leading target
# extents.
def padded_through(count: Nat,+axes: List<&2,Views.Axis>,+middle: List<&2,Views.Axis>,+target: List<&2,U32>,
  +inner: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(axes,Views.axis_extents(middle))) == True{} : Bool},
  outer: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),middle),target)) == True{} : Bool}) ->
  {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),axes),target)) == True{} : Bool}:
  match count target:
    case 0n target: axes_through(axes,middle,target,inner,outer)
    case 1n+rest Nil{}:
      Empty.absurd({Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,1n+rest,Views.Axis{1,0}),axes),Nil{})) == True{} : Bool},
        WordEncoding.false_true(outer))
    case 1n+ +rest Con{+extent,+others}:
      Broadcast.prepended_exists(Views.stretched_axis(U32.is_eq(1,extent),U32.is_eq(1,1),1,0,extent),
        Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,rest,Views.Axis{1,0}),axes),others),
        %Equal.sym(Bool,U32.is_eq(1,1),True{},BandShape.equal_words(1,1,{==})) : {Maybe.is_some(&2,Views.Axis,Views.stretched_axis(U32.is_eq(1,extent),_,1,0,extent)) == True{} : Bool}
        single_some(U32.is_eq(1,extent),1,0,extent),
        padded_through(rest,axes,middle,others,inner,
          prepended_tail(Views.stretched_axis(U32.is_eq(1,extent),U32.is_eq(1,1),1,0,extent),
            Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,rest,Views.Axis{1,0}),middle),others),outer)))

def replicated_append(count: Nat,+more: Nat,+axes: List<&2,Views.Axis>) ->
  {List.append(&2,Views.Axis,List.replicate(Views.Axis,Nat.add(count,more),Views.Axis{1,0}),axes) ==
    List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),List.append(&2,Views.Axis,List.replicate(Views.Axis,more,Views.Axis{1,0}),axes)) : List<&2,Views.Axis>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => Views.Axis{1,0} <> tail,
        List.append(&2,Views.Axis,List.replicate(Views.Axis,Nat.add(rest,more),Views.Axis{1,0}),axes),
        List.append(&2,Views.Axis,List.replicate(Views.Axis,rest,Views.Axis{1,0}),List.append(&2,Views.Axis,List.replicate(Views.Axis,more,Views.Axis{1,0}),axes)),
        replicated_append(rest,more,axes))

# Padding from low to high ranks pads from low to middle and from middle to
# high.
def difference_through(+low: Nat,+middle: Nat,+high: Nat,lower: {Nat.is_le(low,middle) == True{} : Bool},upper: {Nat.is_le(middle,high) == True{} : Bool}) ->
  {Nat.sub(high,low) == Nat.add(Nat.sub(high,middle),Nat.sub(middle,low)) : Nat}:
  +inner = Nat.sub(middle,low)
  +outer = Nat.sub(high,middle)
  %Order.restore_shift(middle,high,upper) : {Nat.sub(_,low) == Nat.add(outer,inner) : Nat}
  %Order.restore_shift(low,middle,lower) : {Nat.sub(Nat.add(_,outer),low) == Nat.add(outer,inner) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(low,inner),outer),Nat.add(low,Nat.add(inner,outer)),Addition.associative(low,inner,outer)) : {Nat.sub(_,low) == Nat.add(outer,inner) : Nat}
  Equal.trans(Nat,Nat.sub(Nat.add(low,Nat.add(inner,outer)),low),Nat.add(inner,outer),Nat.add(outer,inner),Order.subtract_shift(low,Nat.add(inner,outer)),Order.commutative(inner,outer))

# The common extents have the rank of each operand padded to the other.
def padded_ranks(+shape: List<&2,U32>,+left: List<&2,U32>,+right: List<&2,U32>,
  ranks: Ranks(shape,Shapes.with_rank(left,List.length(&2,U32,right)),Shapes.with_rank(right,List.length(&2,U32,left)))) ->
  Shape.Both<{List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,right),List.length(&2,U32,left)),List.length(&2,U32,left)) : Nat},
    {List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,left),List.length(&2,U32,right)),List.length(&2,U32,right)) : Nat}>:
  Shape.Both{left_rank,right_rank} = ranks
  Shape.Both{
    Equal.trans(Nat,List.length(&2,U32,shape),List.length(&2,U32,Shapes.with_rank(left,List.length(&2,U32,right))),
      Nat.add(Nat.sub(List.length(&2,U32,right),List.length(&2,U32,left)),List.length(&2,U32,left)),
      left_rank,replicated_length(Nat.sub(List.length(&2,U32,right),List.length(&2,U32,left)),1,left)),
    Equal.trans(Nat,List.length(&2,U32,shape),List.length(&2,U32,Shapes.with_rank(right,List.length(&2,U32,left))),
      Nat.add(Nat.sub(List.length(&2,U32,left),List.length(&2,U32,right)),List.length(&2,U32,right)),
      right_rank,replicated_length(Nat.sub(List.length(&2,U32,left),List.length(&2,U32,right)),1,right))}

def broadcast_ranks(+left: List<&2,U32>,+right: List<&2,U32>,+shape: List<&2,U32>,common: {Shapes.broadcast(left,right) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Shape.Both<{List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,right),List.length(&2,U32,left)),List.length(&2,U32,left)) : Nat},
    {List.length(&2,U32,shape) == Nat.add(Nat.sub(List.length(&2,U32,left),List.length(&2,U32,right)),List.length(&2,U32,right)) : Nat}>:
  padded_ranks(shape,left,right,common_rank(Shapes.with_rank(left,List.length(&2,U32,right)),Shapes.with_rank(right,List.length(&2,U32,left)),shape,common))

# A rank padded to another is at least that rank.
def padded_covers(+low: Nat,+padding: Nat,+length: Nat,equation: {length == Nat.add(padding,low) : Nat}) -> {Nat.is_le(low,length) == True{} : Bool}:
  %Equal.sym(Nat,length,Nat.add(padding,low),equation) : {Nat.is_le(low,_) == True{} : Bool}
  %Order.commutative(low,padding) : {Nat.is_le(low,_) == True{} : Bool}
  Order.le_add(low,padding)

def flat_ranked(+left: List<&2,U32>,+middle: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+left_exact: {ShapeLaws.elements(left) == U32.to_nat(left_count) : Nat},
  ranks: Ranks(shape,Shapes.with_rank(left,List.length(&2,U32,middle)),Shapes.with_rank(middle,List.length(&2,U32,left))),
  compatible: Broadcast.Compatible(Spread.source_axes(left,left_count,List.length(&2,U32,middle)),Views.with_rank(axes_of(middle),List.length(&2,U32,left)),Some{shape})) ->
  Stretches(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Views.with_rank(axes_of(middle),List.length(&2,U32,shape)),shape):
  Shape.Both{left_rank,middle_rank} = ranks
  Shape.Both{left_some,middle_some} = compatible
  +middle_axes = axes_of(middle)
  +own = List.length(&2,Views.Axis,middle_axes)
  +rank = List.length(&2,U32,middle)
  +padding = Nat.sub(List.length(&2,U32,left),rank)
  +own_rank = axes_length(middle)
  +length = Equal.trans(Nat,Nat.sub(List.length(&2,U32,shape),rank),Nat.sub(Nat.add(padding,rank),rank),padding,
    Equal.cong(Nat,Nat,total => Nat.sub(total,rank),List.length(&2,U32,shape),Nat.add(padding,rank),
      Equal.trans(Nat,List.length(&2,U32,shape),List.length(&2,U32,Shapes.with_rank(middle,List.length(&2,U32,left))),Nat.add(padding,rank),
        middle_rank,replicated_length(padding,1,middle))),
    DivisionOrder.subtract_sum(padding,rank))
  +pads = Equal.trans(Nat,Nat.sub(List.length(&2,U32,left),own),padding,Nat.sub(List.length(&2,U32,shape),own),
    Equal.cong(Nat,Nat,length => Nat.sub(List.length(&2,U32,left),length),own,rank,own_rank),
    Equal.trans(Nat,padding,Nat.sub(List.length(&2,U32,shape),rank),Nat.sub(List.length(&2,U32,shape),own),
      Equal.sym(Nat,Nat.sub(List.length(&2,U32,shape),rank),padding,length),
      Equal.cong(Nat,Nat,length => Nat.sub(List.length(&2,U32,shape),length),rank,own,Equal.sym(Nat,own,rank,own_rank))))
  Shape.Both{
    %Equal.sym(List<&2,Views.Axis>,Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Spread.source_axes(left,left_count,List.length(&2,U32,middle)),
      ranked_axes(left,left_count,List.length(&2,U32,middle),shape,left_exact,left_rank)) :
      {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(_,shape)) == True{} : Bool}
    left_some,
    %Equal.cong(Nat,List<&2,Views.Axis>,pad => List.append(&2,Views.Axis,List.replicate(Views.Axis,pad,Views.Axis{1,0}),middle_axes),
        Nat.sub(List.length(&2,U32,left),own),Nat.sub(List.length(&2,U32,shape),own),pads) :
      {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(_,shape)) == True{} : Bool}
    middle_some}

# Both operands of a broadcast stretch to the common extents, the right one
# read with stride zero.
def flat_stretches(+left: List<&2,U32>,+middle: List<&2,U32>,+shape: List<&2,U32>,+left_count: U32,+left_exact: {ShapeLaws.elements(left) == U32.to_nat(left_count) : Nat},
  +common: {Shapes.broadcast(left,middle) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Stretches(Spread.source_axes(left,left_count,List.length(&2,U32,shape)),Views.with_rank(axes_of(middle),List.length(&2,U32,shape)),shape):
  +left_axes = Spread.source_axes(left,left_count,List.length(&2,U32,middle))
  +middle_axes = Views.with_rank(axes_of(middle),List.length(&2,U32,left))
  flat_ranked(left,middle,shape,left_count,left_exact,
    common_rank(Shapes.with_rank(left,List.length(&2,U32,middle)),Shapes.with_rank(middle,List.length(&2,U32,left)),shape,common),
    %common : Broadcast.Compatible(left_axes,middle_axes,_)
    %Spread.source_extents(left,left_count,List.length(&2,U32,middle),left_exact) :
      Broadcast.Compatible(left_axes,middle_axes,Shapes.broadcast_same_rank(_,Shapes.with_rank(middle,List.length(&2,U32,left))))
    %Equal.trans(List<&2,U32>,Views.axis_extents(middle_axes),Shapes.with_rank(Views.axis_extents(axes_of(middle)),List.length(&2,U32,left)),
        Shapes.with_rank(middle,List.length(&2,U32,left)),Spread.padded_extents(axes_of(middle),List.length(&2,U32,left)),
        Equal.cong(List<&2,U32>,List<&2,U32>,extents => Shapes.with_rank(extents,List.length(&2,U32,left)),Views.axis_extents(axes_of(middle)),middle,axes_extents(middle))) :
      Broadcast.Compatible(left_axes,middle_axes,Shapes.broadcast_same_rank(Views.axis_extents(left_axes),_))
    Broadcast.broadcast_stretches(left_axes,middle_axes))

# Extents that stretch to middle extents stretch to the extents the middle
# extents stretch to.
def stretched_through(+source: List<&2,U32>,+count: U32,+middle: List<&2,U32>,+shape: List<&2,U32>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  source_rank: {Nat.is_le(List.length(&2,U32,source),List.length(&2,U32,middle)) == True{} : Bool},
  +middle_rank: {Nat.is_le(List.length(&2,U32,middle),List.length(&2,U32,shape)) == True{} : Bool},
  inner: Stretched<source,count,middle>,
  outer: {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Views.with_rank(axes_of(middle),List.length(&2,U32,shape)),shape)) == True{} : Bool}) ->
  Stretched<source,count,shape>:
  Stretched{+full,+equation} = inner
  +axes = Canonical.axes(Canonical.canonical_view(source,count))
  +own = List.length(&2,Views.Axis,axes)
  +rank = List.length(&2,U32,middle)
  +outer_padding = Nat.sub(List.length(&2,U32,shape),rank)
  +padded = Equal.trans(List<&2,Views.Axis>,Views.with_rank(axes,List.length(&2,U32,shape)),
    List.append(&2,Views.Axis,List.replicate(Views.Axis,Nat.add(outer_padding,Nat.sub(rank,own)),Views.Axis{1,0}),axes),
    List.append(&2,Views.Axis,List.replicate(Views.Axis,outer_padding,Views.Axis{1,0}),Views.with_rank(axes,rank)),
    Equal.cong(Nat,List<&2,Views.Axis>,pad => List.append(&2,Views.Axis,List.replicate(Views.Axis,pad,Views.Axis{1,0}),axes),
      Nat.sub(List.length(&2,U32,shape),own),Nat.add(outer_padding,Nat.sub(rank,own)),
      difference_through(own,rank,List.length(&2,U32,shape),%axes_rank(source,count,exact) : {Nat.is_le(_,rank) == True{} : Bool} source_rank,middle_rank)),
    replicated_append(outer_padding,Nat.sub(rank,own),axes))
  stretched_when(Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape),source,count,shape,{==},
    %Equal.sym(List<&2,Views.Axis>,Views.with_rank(axes,List.length(&2,U32,shape)),
        List.append(&2,Views.Axis,List.replicate(Views.Axis,outer_padding,Views.Axis{1,0}),Views.with_rank(axes,rank)),padded) :
      {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(_,shape)) == True{} : Bool}
    padded_through(outer_padding,Views.with_rank(axes,rank),axes_of(middle),shape,
      %Equal.sym(List<&2,U32>,Views.axis_extents(axes_of(middle)),middle,axes_extents(middle)) :
        {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(Views.with_rank(axes,rank),_)) == True{} : Bool}
      %Equal.sym(Maybe<&2,List<&2,Views.Axis>>,Views.stretch_axes(Views.with_rank(axes,rank),middle),Some{full},equation) : {Maybe.is_some(&2,List<&2,Views.Axis>,_) == True{} : Bool}
      {==},
      %Equal.cong(Nat,Nat,length => Nat.sub(List.length(&2,U32,shape),length),List.length(&2,Views.Axis,axes_of(middle)),rank,axes_length(middle)) :
        {Maybe.is_some(&2,List<&2,Views.Axis>,Views.stretch_axes(List.append(&2,Views.Axis,List.replicate(Views.Axis,_,Views.Axis{1,0}),axes_of(middle)),shape)) == True{} : Bool}
      outer))

# The stretched view of the source is the view of those axes.
def stretched_view(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  stretched: {Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>}) ->
  {Tensors.stretched(Tensors.view_of(source),shape) == Some{Views.View{full,0}} : Maybe<&2,Views.View>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(source),Some{Canonical.canonical_view(source,count)},Canonical.view(source,count,exact)) :
    {Tensors.stretched(_,shape) == Some{Views.View{full,0}} : Maybe<&2,Views.View>}
  Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Maybe<&2,Views.View>,result => Views.stretched_view(result,0),
    Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape),Some{full},stretched)

# A stretched view of the source comes from axes the source stretches to.
type Viewed<-source: List<&2,U32>,-count: U32,-shape: List<&2,U32>,-view: Views.View> is Data:
  Viewed{full: List<&2,Views.Axis>,
    equation: {Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
    same: {Views.View{full,0} == view : Views.View}}

def viewed_when(result: Maybe<&2,List<&2,Views.Axis>>,+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+view: Views.View,
  equation: {Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape) == result : Maybe<&2,List<&2,Views.Axis>>},
  seen: {Views.stretched_view(result,0) == Some{view} : Maybe<&2,Views.View>}) -> Viewed<source,count,shape,view>:
  match result:
    case Some{+full}:
      Viewed{full,equation,Equal.cong(Maybe<&2,Views.View>,Views.View,maybe => Maybe.default(&2,Views.View,maybe,Views.View{[],0}),Some{Views.View{full,0}},Some{view},seen)}
    case None{}: Empty.absurd(Viewed<source,count,shape,view>,absent(~Views.View,view,seen))

def viewed(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+view: Views.View,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  stretched: {Tensors.stretched(Tensors.view_of(source),shape) == Some{view} : Maybe<&2,Views.View>}) -> Viewed<source,count,shape,view>:
  viewed_when(Views.stretch_axes(Spread.source_axes(source,count,List.length(&2,U32,shape)),shape),source,count,shape,view,{==},
    %Canonical.view(source,count,exact) : {Tensors.stretched(_,shape) == Some{view} : Maybe<&2,Views.View>} stretched)

# Whole storage of a shape holds its count.
def whole_length(~Element: Data,+shape: List<&2,U32>,+count: U32,+input: Buffer.Model<Element>,formed: Contract.WellFormed<Element,shape,Buffer.content(~Element,input)>,
  +exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat}) -> {Buffer.length(~Element,input) == U32.to_nat(count) : Nat}:
  Contract.WellFormed{length_exact,ready} = formed
  Equal.trans(Nat,Buffer.length(~Element,input),Buffer.content_length(~Element,Buffer.content(~Element,input)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,input)),Buffer.length(~Element,input),Buffer.content_length_of(~Element,input)),
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,input)),ShapeLaws.elements(shape),U32.to_nat(count),length_exact,exact))

# The input read through the stretched view: its elements are those of the
# target, and when the target has elements its last address lies inside the
# input.
def broadcast_viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+source_count: U32,+source_exact: {ShapeLaws.elements(shape) == U32.to_nat(source_count) : Nat},+full: List<&2,Views.Axis>,
  +equation: {Views.stretch_axes(Spread.source_axes(shape,source_count,List.length(&2,U32,target)),target) == Some{full} : Maybe<&2,List<&2,Views.Axis>>}) ->
  Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{Views.View{full,0}}},target,
    Tensors.gather(~Element,"broadcast_to",Tensors.Usable{},Some{Views.View{full,0}},[],Model.pack(~Element,bands),geometry,shape,target,fill,workers)):
  +axes = Spread.source_axes(shape,source_count,List.length(&2,U32,target))
  +lanes = Lanes.stretched(axes,target,full,equation)
  +same_elements = Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),target,Views.axis_extents(full),
    Equal.sym(List<&2,U32>,Views.axis_extents(full),target,Lanes.stretched_extents(axes,target,full,lanes)))
  Global.restated(~Element,~GatherValues.Directed<Logical>,~Expanded<Logical>,~GatherValues.directed(~Element,~Logical,~original),~value(~Element,~Logical,~original),
    GatherValues.Directed{logical,Views.View{full,0},[]},Expanded{logical,Some{Views.View{full,0}}},
    (index => GatherValues.unflagged_value(~Element,~Logical,~original,GatherValues.Gather{logical,Views.View{full,0}},index)),target,
    Tensors.gather(~Element,"broadcast_to",Tensors.Usable{},Some{Views.View{full,0}},[],Model.pack(~Element,bands),geometry,shape,target,fill,workers),
    Ready.shaped(~Element,~Logical,~original,logical,"broadcast_to",full,0,[],bands,geometry,shape,target,fill,workers,contract,counted,same_elements,
      nonempty =>
        %Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(source_count),source_exact) :
          {Nat.is_lt(Coordinates.address(Lanes.points(full,Lanes.lasts(full)),0n,False{}),_) == True{} : Bool}
        Spread.source_bound(shape,source_count,target,full,source_exact,lanes,nonempty)))

def broadcast_seen(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,+view: Views.View,
  +fill: Element,+workers: U32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+source_count: U32,+source_exact: {ShapeLaws.elements(shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Tensors.stretched(Tensors.view_of(shape),target) == Some{view} : Maybe<&2,Views.View>},seen: Viewed<shape,source_count,target,view>) ->
  Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{view}},target,
    Tensors.broadcast_to(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},target)):
  Viewed{+full,+equation,+same} = seen
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(shape),target),Some{view},stretched) :
    Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{view}},target,
      Tensors.broadcast_view(~Element,Tensors.Usable{},_,Model.pack(~Element,bands),geometry,shape,target,fill,workers))
  %same :
    Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{_}},target,
      Tensors.broadcast_view(~Element,Tensors.Usable{},Some{_},Model.pack(~Element,bands),geometry,shape,target,fill,workers))
  broadcast_viewed(~Element,~Logical,~original,logical,shape,target,fill,workers,bands,geometry,contract,counted,source_count,source_exact,full,equation)

def broadcast_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,+view: Views.View,
  +fill: Element,+workers: U32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,source_counted: Global.Counted<shape>,+stretched: {Tensors.stretched(Tensors.view_of(shape),target) == Some{view} : Maybe<&2,Views.View>}) ->
  Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{view}},target,
    Tensors.broadcast_to(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},target)):
  Global.Counted{+source_count,+source_accepted} = source_counted
  +source_exact = Global.whole_counted(shape,source_count,source_accepted)
  broadcast_seen(~Element,~Logical,~original,logical,shape,target,view,fill,workers,bands,geometry,contract,counted,source_count,source_exact,stretched,
    viewed(shape,source_count,target,view,source_exact,stretched))

def contract_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Counted<shape>:
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values} = contract
  counted

# Broadcasting a tensor of the contract to extents its own extents stretch to
# gives a tensor of those extents that holds, at each index, the value at the
# address of that index in the stretched view.
def broadcast_to(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,+view: Views.View,
  +fill: Element,+workers: U32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  +counted: Global.Counted<target>,+stretched: {Tensors.stretched(Tensors.view_of(shape),target) == Some{view} : Maybe<&2,Views.View>}) ->
  Global.Shaped(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,Some{view}},target,
    Tensors.broadcast_to(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},target)):
  broadcast_counted(~Element,~Logical,~original,logical,shape,target,view,fill,workers,bands,geometry,contract,counted,
    contract_counted(~Element,~Logical,~original,logical,shape,fill,workers,bands,geometry,contract),stretched)

# An operand of the common extents is read as it is.
def Plain(~Logical: Data,before: Logical,after: Expanded<Logical>) -> Data:
  {Expanded{before,None{}} == after : Expanded<Logical>}

def plain_agrees(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+before: Logical,+after: Expanded<Logical>,+related: Plain(~Logical,before,after),+index: U32) ->
  {original(before,index) == value(~Element,~Logical,~original,after,index) : Element}:
  %related : {original(before,index) == value(~Element,~Logical,~original,_,index) : Element}
  {==}

def plain(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~Expanded<Logical>,~value(~Element,~Logical,~original),Expanded{logical,None{}},Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}):
  Global.relabeled(~Element,~Logical,~Expanded<Logical>,~original,~value(~Element,~Logical,~original),~Plain(~Logical),~plain_agrees(~Element,~Logical,~original),
    logical,Expanded{logical,None{}},{==},Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},contract)

