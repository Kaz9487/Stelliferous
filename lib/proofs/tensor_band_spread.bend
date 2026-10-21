# The bands of a left operand combined with the parts of a right source they
# read. The source is read with the extents of the left operand through its
# contiguous view stretched to them. Each band reads its part through a view
# of the extents of its cells: a band of a whole image reads the source at the
# addresses of the stretched view, a band of image rows the part copied for it.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ../tensor.bend as Tensors
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_coordinates.bend as Coordinates
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./tensor_band_values.bend as BandValues
import ./tensor_count_completion.bend as CountCompletion
import ./write_program.bend as Writes
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./binary_radix.bend as Radix
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_band_combine.bend as BandCombine
import ./tensor_combine_kernel_values.bend as CombineKernel
import ./tensor_contract.bend as Contract
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./tensor_band_each.bend as Each
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_gather_values.bend as GatherValues
import ./tensor_band_plan.bend as BandPlan
import ./tensor_view_lanes.bend as Lanes
import ./tensor_row_major_coordinates.bend as Digits
import ./tensor_linear_layout.bend as Linear
import ./tensor_canonical_view.bend as Canonical
import ./tensor_copy_values.bend as Copy
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_view_metadata_refinement.bend as Metadata

# Axes and extents with leading axes of one element.
def padded_extents_of(count: Nat,+axes: List<&2,Views.Axis>) ->
  {Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),axes)) == List.append(&2,U32,List.replicate(U32,count,1),Views.axis_extents(axes)) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,extents => 1 <> extents,
        Views.axis_extents(List.append(&2,Views.Axis,List.replicate(Views.Axis,rest,Views.Axis{1,0}),axes)),List.append(&2,U32,List.replicate(U32,rest,1),Views.axis_extents(axes)),
        padded_extents_of(rest,axes))

def padded_extents(+axes: List<&2,Views.Axis>,+rank: Nat) -> {Views.axis_extents(Views.with_rank(axes,rank)) == Shapes.with_rank(Views.axis_extents(axes),rank) : List<&2,U32>}:
  %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),Metadata.extents_length(axes)) :
    {Views.axis_extents(Views.with_rank(axes,rank)) == List.append(&2,U32,List.replicate(U32,Nat.sub(rank,_),1),Views.axis_extents(axes)) : List<&2,U32>}
  padded_extents_of(Nat.sub(rank,List.length(&2,Views.Axis,axes)),axes)

def padded_layout(count: Nat,+axes: List<&2,Views.Axis>,+layout: Linear.Layout(axes)) ->
  Linear.Layout(List.append(&2,Views.Axis,List.replicate(Views.Axis,count,Views.Axis{1,0}),axes)):
  match count:
    case 0n: layout
    case 1n+rest: Shape.Both{Unit{},padded_layout(rest,axes,layout)}

def rank_layout(+axes: List<&2,Views.Axis>,+rank: Nat,+layout: Linear.Layout(axes)) -> Linear.Layout(Views.with_rank(axes,rank)):
  padded_layout(Nat.sub(rank,List.length(&2,Views.Axis,axes)),axes,layout)

def padded_zero(count: Nat,+extents: List<&2,U32>) ->
  {Shapes.has_zero(List.append(&2,U32,List.replicate(U32,count,1),extents)) == Shapes.has_zero(extents) : Bool}:
  match count:
    case 0n: {==}
    case 1n+rest: padded_zero(rest,extents)

def padded_elements(count: Nat,+extents: List<&2,U32>) ->
  {ShapeLaws.elements(List.append(&2,U32,List.replicate(U32,count,1),extents)) == ShapeLaws.elements(extents) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(Nat,Nat.add(ShapeLaws.elements(List.append(&2,U32,List.replicate(U32,rest,1),extents)),0n),
        ShapeLaws.elements(List.append(&2,U32,List.replicate(U32,rest,1),extents)),ShapeLaws.elements(extents),
        Addition.zero_right(ShapeLaws.elements(List.append(&2,U32,List.replicate(U32,rest,1),extents))),padded_elements(rest,extents))

def rank_elements(+extents: List<&2,U32>,+rank: Nat) -> {ShapeLaws.elements(Shapes.with_rank(extents,rank)) == ShapeLaws.elements(extents) : Nat}:
  padded_elements(Nat.sub(rank,List.length(&2,U32,extents)),extents)

def rank_zero(+extents: List<&2,U32>,+rank: Nat) -> {Shapes.has_zero(Shapes.with_rank(extents,rank)) == Shapes.has_zero(extents) : Bool}:
  padded_zero(Nat.sub(rank,List.length(&2,U32,extents)),extents)

# Two axes read the same addresses when they have the same extents and the
# same stride wherever an axis has more than one element.
def Agree(left: List<&2,Views.Axis>,right: List<&2,Views.Axis>) -> Data:
  match left right:
    case Nil{} Nil{}: Unit
    case Con{Views.Axis{extent,stride},tail} Con{Views.Axis{other,step},others}:
      Shape.Both<Shape.Both<{extent == other : U32},Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == U32.to_nat(step) : Nat})>,Agree(tail,others)>
    case left right: Empty

def lane_pick(spread: Bool,+extent: U32,+stride: U32,+step: U32,+reversed: List<&2,CursorModel.Coordinate>,
  classified: {Nat.is_lt(1n,U32.to_nat(extent)) == spread : Bool},kept: Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == U32.to_nat(step) : Nat})) ->
  {Bool.pick(List<&2,CursorModel.Coordinate>,spread,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed) ==
    Bool.pick(List<&2,CursorModel.Coordinate>,spread,CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(step)} <> reversed,reversed) : List<&2,CursorModel.Coordinate>}:
  match spread:
    case True{}:
      Equal.cong(Nat,List<&2,CursorModel.Coordinate>,value => CursorModel.Coordinate{U32.to_nat(extent),0n,value} <> reversed,U32.to_nat(stride),U32.to_nat(step),
        %classified : Boolean.When(_,{U32.to_nat(stride) == U32.to_nat(step) : Nat}) kept)
    case False{}: {==}

def agreed_prepared(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>,+reversed: List<&2,CursorModel.Coordinate>,agree: Agree(left,right)) ->
  {CursorModel.prepare_axes(left,reversed) == CursorModel.prepare_axes(right,reversed) : List<&2,CursorModel.Coordinate>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+tail} Con{Views.Axis{+other,+step},+others}:
      Shape.Both{Shape.Both{+same,kept},rest} = agree
      +picked = Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(extent)),CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(stride)} <> reversed,reversed)
      %same :
        {CursorModel.prepare_axes(tail,picked) ==
          CursorModel.prepare_axes(others,Bool.pick(List<&2,CursorModel.Coordinate>,Nat.is_lt(1n,U32.to_nat(_)),CursorModel.Coordinate{U32.to_nat(_),0n,U32.to_nat(step)} <> reversed,reversed)) : List<&2,CursorModel.Coordinate>}
      %lane_pick(Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,step,reversed,{==},kept) :
        {CursorModel.prepare_axes(tail,picked) == CursorModel.prepare_axes(others,_) : List<&2,CursorModel.Coordinate>}
      agreed_prepared(tail,others,picked,rest)
    case Nil{} Con{axis,others}: Empty.absurd({CursorModel.prepare_axes(Nil{},reversed) == CursorModel.prepare_axes(Con{axis,others},reversed) : List<&2,CursorModel.Coordinate>},agree)
    case Con{axis,tail} Nil{}: Empty.absurd({CursorModel.prepare_axes(Con{axis,tail},reversed) == CursorModel.prepare_axes(Nil{},reversed) : List<&2,CursorModel.Coordinate>},agree)

# Views of axes that agree have the same address at every index.
def agreed_address(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>,+offset: U32,+index: Nat,agree: Agree(left,right)) ->
  {Copy.address(Views.View{left,offset},index) == Copy.address(Views.View{right,offset},index) : U32}:
  Equal.cong(List<&2,CursorModel.Coordinate>,U32,axes => U32.from_nat(CursorModel.address(CursorOrder.iterate(index,axes),U32.to_nat(offset))),
    CursorModel.prepare_axes(left,[]),CursorModel.prepare_axes(right,[]),agreed_prepared(left,right,[],agree))

def zero_steps(enabled: Bool) -> Boolean.When(enabled,{U32.to_nat(0) == U32.to_nat(0) : Nat}):
  match enabled:
    case True{}: {==}
    case False{}: Unit{}

# Two lanes stretched to one extent from axes that agree agree.
def lane_steps(+size: U32,+stride: U32,+extent: U32,+step: U32,+result: U32,+other_size: U32,+other_stride: U32,+other_step: U32,
  lane: Lanes.Lane<size,stride,extent,step>,other_lane: Lanes.Lane<other_size,other_stride,extent,other_step>,+resulting: {result == extent : U32},
  +same: {size == other_size : U32},kept: Boolean.When(Nat.is_lt(1n,U32.to_nat(size)),{U32.to_nat(stride) == U32.to_nat(other_stride) : Nat})) ->
  Boolean.When(Nat.is_lt(1n,U32.to_nat(result)),{U32.to_nat(step) == U32.to_nat(other_step) : Nat}):
  match lane other_lane:
    case Lanes.Kept{+same_size,+kept_step} Lanes.Kept{other_same,+other_kept}:
      %Equal.trans(U32,size,extent,result,same_size,Equal.sym(U32,result,extent,resulting)) : Boolean.When(Nat.is_lt(1n,U32.to_nat(_)),{U32.to_nat(step) == U32.to_nat(other_step) : Nat})
      %Equal.sym(U32,step,stride,kept_step) : Boolean.When(Nat.is_lt(1n,U32.to_nat(size)),{U32.to_nat(_) == U32.to_nat(other_step) : Nat})
      %Equal.sym(U32,other_step,other_stride,other_kept) : Boolean.When(Nat.is_lt(1n,U32.to_nat(size)),{U32.to_nat(stride) == U32.to_nat(_) : Nat})
      kept
    case Lanes.Repeated{single,+zero} Lanes.Repeated{other_single,+other_zero}:
      %Equal.sym(U32,step,0,zero) : Boolean.When(Nat.is_lt(1n,U32.to_nat(result)),{U32.to_nat(_) == U32.to_nat(other_step) : Nat})
      %Equal.sym(U32,other_step,0,other_zero) : Boolean.When(Nat.is_lt(1n,U32.to_nat(result)),{U32.to_nat(0) == U32.to_nat(_) : Nat})
      zero_steps(Nat.is_lt(1n,U32.to_nat(result)))
    case Lanes.Kept{+same_size,kept_step} Lanes.Repeated{+other_single,zero}:
      %Equal.trans(U32,1,size,result,Equal.trans(U32,1,other_size,size,Equal.sym(U32,other_size,1,other_single),Equal.sym(U32,size,other_size,same)),
        Equal.trans(U32,size,extent,result,same_size,Equal.sym(U32,result,extent,resulting))) :
        Boolean.When(Nat.is_lt(1n,U32.to_nat(_)),{U32.to_nat(step) == U32.to_nat(other_step) : Nat})
      Unit{}
    case Lanes.Repeated{+single,zero} Lanes.Kept{+other_same,other_kept}:
      %Equal.trans(U32,1,other_size,result,Equal.trans(U32,1,size,other_size,Equal.sym(U32,size,1,single),same),
        Equal.trans(U32,other_size,extent,result,other_same,Equal.sym(U32,result,extent,resulting))) :
        Boolean.When(Nat.is_lt(1n,U32.to_nat(_)),{U32.to_nat(step) == U32.to_nat(other_step) : Nat})
      Unit{}

def stretched_agree(+left_out: List<&2,Views.Axis>,+right_out: List<&2,Views.Axis>,+left: List<&2,Views.Axis>,+target: List<&2,U32>,+right: List<&2,Views.Axis>,
  left_lanes: Lanes.Stretched(left,target,left_out),right_lanes: Lanes.Stretched(right,target,right_out),agree: Agree(left,right)) -> Agree(left_out,right_out):
  match left_out right_out:
    case Nil{} Nil{}: Unit{}
    case Con{Views.Axis{+result,+step},+others} Con{Views.Axis{+other_result,+other_step},+other_others}:
      match left target:
        case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
          match right:
            case Con{Views.Axis{+other_size,+other_stride},+other_tail}:
              Shape.Both{Shape.Both{+resulting,lane},tail_lanes} = left_lanes
              Shape.Both{Shape.Both{+other_resulting,other_lane},other_tail_lanes} = right_lanes
              Shape.Both{Shape.Both{+same,kept},tail_agree} = agree
              Shape.Both{Shape.Both{Equal.trans(U32,result,extent,other_result,resulting,Equal.sym(U32,other_result,extent,other_resulting)),
                  lane_steps(size,stride,extent,step,result,other_size,other_stride,other_step,lane,other_lane,resulting,same,kept)},
                stretched_agree(others,other_others,tail,rest,other_tail,tail_lanes,other_tail_lanes,tail_agree)}
            case Nil{}: Empty.absurd(Agree(Con{Views.Axis{result,step},others},Con{Views.Axis{other_result,other_step},other_others}),right_lanes)
        case Con{Views.Axis{size,stride},tail} Nil{}: Empty.absurd(Agree(Con{Views.Axis{result,step},others},Con{Views.Axis{other_result,other_step},other_others}),left_lanes)
        case Nil{} Con{extent,rest}: Empty.absurd(Agree(Con{Views.Axis{result,step},others},Con{Views.Axis{other_result,other_step},other_others}),left_lanes)
        case Nil{} Nil{}: Empty.absurd(Agree(Con{Views.Axis{result,step},others},Con{Views.Axis{other_result,other_step},other_others}),left_lanes)
    case Nil{} Con{Views.Axis{+other_result,+other_step},+other_others}:
      Empty.absurd(Agree(Nil{},Con{Views.Axis{other_result,other_step},other_others}),Words.false_true(
        Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],other_result <> Views.axis_extents(other_others),
          Equal.trans(List<&2,U32>,[],target,other_result <> Views.axis_extents(other_others),Lanes.stretched_extents(left,target,Nil{},left_lanes),
            Equal.sym(List<&2,U32>,Views.axis_extents(Con{Views.Axis{other_result,other_step},other_others}),target,
              Lanes.stretched_extents(right,target,Con{Views.Axis{other_result,other_step},other_others},right_lanes))))))
    case Con{Views.Axis{+result,+step},+others} Nil{}:
      Empty.absurd(Agree(Con{Views.Axis{result,step},others},Nil{}),Words.false_true(
        Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],result <> Views.axis_extents(others),
          Equal.trans(List<&2,U32>,[],target,result <> Views.axis_extents(others),Lanes.stretched_extents(right,target,Nil{},right_lanes),
            Equal.sym(List<&2,U32>,Views.axis_extents(Con{Views.Axis{result,step},others}),target,
              Lanes.stretched_extents(left,target,Con{Views.Axis{result,step},others},left_lanes))))))

# Axes laid out row-major with the same extents agree.
def layout_stride(spread: Bool,+extent: U32,+stride: U32,+step: U32,+count: Nat,+other_count: Nat,
  classified: {Nat.is_lt(1n,U32.to_nat(extent)) == spread : Bool},+same: {count == other_count : Nat},
  left: Boolean.When(spread,{U32.to_nat(stride) == count : Nat}),right: Boolean.When(spread,{U32.to_nat(step) == other_count : Nat})) ->
  Boolean.When(Nat.is_lt(1n,U32.to_nat(extent)),{U32.to_nat(stride) == U32.to_nat(step) : Nat}):
  match spread:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(1n,U32.to_nat(extent)),spread,classified) : Boolean.When(_,{U32.to_nat(stride) == U32.to_nat(step) : Nat})
      Equal.trans(Nat,U32.to_nat(stride),count,U32.to_nat(step),left,Equal.trans(Nat,count,other_count,U32.to_nat(step),same,Equal.sym(Nat,U32.to_nat(step),other_count,right)))
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(1n,U32.to_nat(extent)),spread,classified) : Boolean.When(_,{U32.to_nat(stride) == U32.to_nat(step) : Nat})
      Unit{}

def layouts_agree(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>,left_layout: Linear.Layout(left),right_layout: Linear.Layout(right),
  +same: {Views.axis_extents(left) == Views.axis_extents(right) : List<&2,U32>}) -> Agree(left,right):
  match left right:
    case Nil{} Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+tail} Con{Views.Axis{+other,+step},+others}:
      Shape.Both{left_stride,tail_layout} = left_layout
      Shape.Both{right_stride,others_layout} = right_layout
      +heads = Equal.cong(List<&2,U32>,U32,Lanes.head_extent,extent <> Views.axis_extents(tail),other <> Views.axis_extents(others),same)
      +tails = Equal.cong(List<&2,U32>,List<&2,U32>,Lanes.tail_extents,extent <> Views.axis_extents(tail),other <> Views.axis_extents(others),same)
      Shape.Both{Shape.Both{heads,
          layout_stride(Nat.is_lt(1n,U32.to_nat(extent)),extent,stride,step,ShapeLaws.elements(Views.axis_extents(tail)),ShapeLaws.elements(Views.axis_extents(others)),{==},
            Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(tail),Views.axis_extents(others),tails),left_stride,
            %Equal.sym(U32,extent,other,heads) : Boolean.When(Nat.is_lt(1n,U32.to_nat(_)),{U32.to_nat(step) == ShapeLaws.elements(Views.axis_extents(others)) : Nat}) right_stride)},
        layouts_agree(tail,others,tail_layout,others_layout,tails)}
    case Nil{} Con{Views.Axis{other,step},others}:
      Empty.absurd(Agree(Nil{},Con{Views.Axis{other,step},others}),Words.false_true(Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],other <> Views.axis_extents(others),same)))
    case Con{Views.Axis{extent,stride},tail} Nil{}:
      Empty.absurd(Agree(Con{Views.Axis{extent,stride},tail},Nil{}),Words.false_true(Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],extent <> Views.axis_extents(tail),Equal.sym(List<&2,U32>,extent <> Views.axis_extents(tail),[],same))))

# The source read with the extents of the left operand: its contiguous axes,
# with leading axes of one element up to the rank of those extents, stretched
# to them.
def source_axes(+source: List<&2,U32>,+count: U32,+rank: Nat) -> List<&2,Views.Axis>:
  Views.with_rank(Canonical.axes(Canonical.canonical_view(source,count)),rank)

def source_extents(+source: List<&2,U32>,+count: U32,+rank: Nat,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat}) ->
  {Views.axis_extents(source_axes(source,count,rank)) == Shapes.with_rank(source,rank) : List<&2,U32>}:
  %Canonical.canonical_shape(source,count,exact) : {Views.axis_extents(source_axes(source,count,rank)) == Shapes.with_rank(_,rank) : List<&2,U32>}
  padded_extents(Canonical.axes(Canonical.canonical_view(source,count)),rank)

def stretched_source(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat}) ->
  {Bands.stretched_source(source,shape) == Views.stretched_view(Views.stretch_axes(source_axes(source,count,List.length(&2,U32,shape)),shape),0) : Maybe<&2,Views.View>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(source),Some{Canonical.canonical_view(source,count)},Canonical.view(source,count,exact)) :
    {Bands.stretched_view(_,shape) == Views.stretched_view(Views.stretch_axes(source_axes(source,count,List.length(&2,U32,shape)),shape),0) : Maybe<&2,Views.View>}
  {==}

def lane_nonzero(+size: U32,+stride: U32,+extent: U32,+step: U32,lane: Lanes.Lane<size,stride,extent,step>,nonzero: {U32.is_zero(extent) == False{} : Bool}) ->
  {U32.is_zero(size) == False{} : Bool}:
  match lane:
    case Lanes.Kept{same,kept}: %Equal.sym(U32,size,extent,same) : {U32.is_zero(_) == False{} : Bool} nonzero
    case Lanes.Repeated{single,zero}: %Equal.sym(U32,size,1,single) : {U32.is_zero(_) == False{} : Bool} {==}

# Every axis stretched to extents without zero has an element.
def stretched_nonempty(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Lanes.Stretched(axes,target,out),
  +nonempty: {Shapes.has_zero(target) == False{} : Bool}) -> {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}:
  match axes target out:
    case Nil{} Nil{} Nil{}: {==}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{result,step},+others}:
      Shape.Both{Shape.Both{resulting,lane},tail_lanes} = lanes
      %Equal.sym(Bool,U32.is_zero(size),False{},lane_nonzero(size,stride,extent,step,lane,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonempty))) :
        {_ || Shapes.has_zero(Views.axis_extents(tail)) == False{} : Bool}
      stretched_nonempty(tail,rest,others,tail_lanes,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonempty))
    case Nil{} Nil{} Con{axis,others}: Empty.absurd({Shapes.has_zero(Views.axis_extents(Nil{})) == False{} : Bool},lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd({Shapes.has_zero(Views.axis_extents(Nil{})) == False{} : Bool},lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out: Empty.absurd({Shapes.has_zero(Views.axis_extents(Con{Views.Axis{size,stride},tail})) == False{} : Bool},lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}: Empty.absurd({Shapes.has_zero(Views.axis_extents(Con{Views.Axis{size,stride},tail})) == False{} : Bool},lanes)

# A stretched axis repeats one element where the axis it came from has one.
def spreads_lane(single: Bool,+size: U32,+stride: U32,+extent: U32,+result: U32,+step: U32,classified: {U32.is_eq(size,1) == single : Bool},
  +resulting: {result == extent : U32},lane: Lanes.Lane<size,stride,extent,step>) -> Lanes.Spreads(single,result,step):
  match single lane:
    case True{} Lanes.Kept{+same,kept}:
      %Equal.trans(U32,1,size,result,Equal.sym(U32,size,1,Arithmetic.equal_of_is_eq(size,1,classified)),Equal.trans(U32,size,extent,result,same,Equal.sym(U32,result,extent,resulting))) :
        Lanes.Repeats<_,step>
      Lanes.Narrow{{==}}
    case True{} Lanes.Repeated{single,zero}: Lanes.Still{zero}
    case False{} lane: Unit{}

def stretched_spreadable(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Lanes.Stretched(axes,target,out)) ->
  Lanes.Spreadable(Views.axis_extents(axes),out):
  match axes target out:
    case Nil{} Nil{} Nil{}: Unit{}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{+result,+step},+others}:
      Shape.Both{Shape.Both{resulting,lane},tail_lanes} = lanes
      Shape.Both{spreads_lane(U32.is_eq(size,1),size,stride,extent,result,step,{==},resulting,lane),stretched_spreadable(tail,rest,others,tail_lanes)}
    case Nil{} Nil{} Con{axis,others}: Empty.absurd(Lanes.Spreadable(Views.axis_extents(Nil{}),Con{axis,others}),lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd(Lanes.Spreadable(Views.axis_extents(Nil{}),out),lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out: Empty.absurd(Lanes.Spreadable(Views.axis_extents(Con{Views.Axis{size,stride},tail}),out),lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}: Empty.absurd(Lanes.Spreadable(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{}),lanes)

# The facts of the stretched source.
def source_nonempty(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source,count,List.length(&2,U32,shape)),shape,full),nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Shapes.has_zero(source) == False{} : Bool}:
  +rank = List.length(&2,U32,shape)
  Equal.trans(Bool,Shapes.has_zero(source),Shapes.has_zero(Shapes.with_rank(source,rank)),False{},Equal.sym(Bool,Shapes.has_zero(Shapes.with_rank(source,rank)),Shapes.has_zero(source),rank_zero(source,rank)),
    %source_extents(source,count,rank,exact) : {Shapes.has_zero(_) == False{} : Bool}
    stretched_nonempty(source_axes(source,count,rank),shape,full,lanes,nonempty))

def source_layout(+source: List<&2,U32>,+count: U32,+rank: Nat,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},nonempty: {Shapes.has_zero(source) == False{} : Bool}) ->
  Linear.Layout(source_axes(source,count,rank)):
  +axes = Canonical.axes(Canonical.canonical_view(source,count))
  rank_layout(axes,rank,Linear.checked_layout(axes,0,
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(source,count)),source,Canonical.canonical_shape(source,count,exact)) : {Shapes.has_zero(_) == False{} : Bool} nonempty,
    Canonical.canonical_contiguous(source,count,exact)))

def full_spreadable(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source,count,List.length(&2,U32,shape)),shape,full)) -> Lanes.Spreadable(Shapes.with_rank(source,List.length(&2,U32,shape)),full):
  %source_extents(source,count,List.length(&2,U32,shape),exact) : Lanes.Spreadable(_,full)
  stretched_spreadable(source_axes(source,count,List.length(&2,U32,shape)),shape,full,lanes)

# Coordinates below extents, one per axis.
def Below(extents: List<&2,U32>,coordinates: List<&2,Nat>) -> Data:
  match extents coordinates:
    case Nil{} Nil{}: Unit
    case Con{extent,rest} Con{coordinate,more}: Shape.Both<{Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},Below(rest,more)>
    case extents coordinates: Empty

def below_inside(+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+extents: List<&2,U32>,+same: {Views.axis_extents(axes) == extents : List<&2,U32>},
  below: Below(extents,coordinates)) -> Lanes.Inside(axes,coordinates):
  match axes coordinates:
    case Nil{} Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+more}:
      match extents:
        case Con{+other,+rest}:
          Shape.Both{below_here,below_more} = below
          Shape.Both{
            %Equal.sym(U32,extent,other,Equal.cong(List<&2,U32>,U32,Lanes.head_extent,extent <> Views.axis_extents(tail),other <> rest,same)) : {Nat.is_lt(coordinate,U32.to_nat(_)) == True{} : Bool} below_here,
            below_inside(tail,more,rest,Equal.cong(List<&2,U32>,List<&2,U32>,Lanes.tail_extents,extent <> Views.axis_extents(tail),other <> rest,same),below_more)}
        case Nil{}:
          Empty.absurd(Lanes.Inside(Con{Views.Axis{extent,stride},tail},Con{coordinate,more}),Words.false_true(
            Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],extent <> Views.axis_extents(tail),Equal.sym(List<&2,U32>,extent <> Views.axis_extents(tail),[],same))))
    case Nil{} Con{coordinate,more}:
      match extents:
        case Nil{}: Empty.absurd(Lanes.Inside(Nil{},Con{coordinate,more}),below)
        case Con{other,rest}: Empty.absurd(Lanes.Inside(Nil{},Con{coordinate,more}),Words.false_true(Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],other <> rest,same)))
    case Con{Views.Axis{extent,stride},tail} Nil{}:
      match extents:
        case Con{other,rest}: Empty.absurd(Lanes.Inside(Con{Views.Axis{extent,stride},tail},Nil{}),below)
        case Nil{}:
          Empty.absurd(Lanes.Inside(Con{Views.Axis{extent,stride},tail},Nil{}),Words.false_true(
            Equal.cong(List<&2,U32>,Bool,Lanes.has_extents,[],extent <> Views.axis_extents(tail),Equal.sym(List<&2,U32>,extent <> Views.axis_extents(tail),[],same))))

# A band of image rows reads its image and its rows from the stretched source:
# one slice of the rows for extents of three axes, a slice of the image and
# then of the rows for four.
type ImageRows<-full: List<&2,Views.Axis>,-image: U32,-start: U32,-rows: U32,-band: List<&2,Views.Axis>,-offset: U32> is Data:
  ImageRows{axes: List<&2,Views.Axis>,image_offset: U32,outer: Lanes.Sliced(full,0n,image,1,0,axes,image_offset),
    inner: Lanes.Sliced(axes,2n,start,rows,image_offset,band,offset)}

def Slices(shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,full: List<&2,Views.Axis>,band: List<&2,Views.Axis>,offset: U32) -> Data:
  match shape:
    case [c,h,w]: Lanes.Sliced(full,1n,(first % height : U32),rows,0,band,offset)
    case [n,c,h,w]: ImageRows<full,(first / height : U32),(first % height : U32),rows,band,offset>
    case shape: Empty

# The coordinates in the stretched source of coordinates in a band.
def pixel_of(shape: List<&2,U32>,+height: U32,+first: U32,coordinates: List<&2,Nat>) -> List<&2,Nat>:
  match shape:
    case [c,h,w]: Lanes.shifted(1n,(first % height : U32),coordinates)
    case [n,c,h,w]: Lanes.shifted(0n,(first / height : U32),Lanes.shifted(2n,(first % height : U32),coordinates))
    case shape: coordinates

# The extents of the cells of a band of image rows.
def band_cells(shape: List<&2,U32>,rows: U32) -> List<&2,U32>:
  match shape:
    case [c,h,w]: [c,rows,w]
    case [n,c,h,w]: [1,c,rows,w]
    case shape: []

def image_slices(image: Maybe<&2,Views.View>,+full: List<&2,Views.Axis>,+index: U32,+start: U32,+rows: U32,+band: List<&2,Views.Axis>,+offset: U32,
  outer: {Views.slice(Views.View{full,0},0n,index,1,1) == image : Maybe<&2,Views.View>},
  inner: {Bands.sliced_rows(image,start,rows) == Some{Views.View{band,offset}} : Maybe<&2,Views.View>}) -> ImageRows<full,index,start,rows,band,offset>:
  match image:
    case Some{Views.View{+axes,+image_offset}}:
      ImageRows{axes,image_offset,Lanes.sliced(full,0n,index,1,0,axes,image_offset,outer),Lanes.sliced(axes,2n,start,rows,image_offset,band,offset,inner)}
    case None{}:
      Empty.absurd(ImageRows<full,index,start,rows,band,offset>,Words.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,Bands.has_view,None{},Some{Views.View{band,offset}},inner)))

def slices_of(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  result: {Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}) == Some{Views.View{band,offset}} : Maybe<&2,Views.View>}) ->
  Slices(shape,height,first,rows,full,band,offset):
  match shape:
    case [+c,+h,+w]: Lanes.sliced(full,1n,(first % height : U32),rows,0,band,offset,result)
    case [+n,+c,+h,+w]:
      image_slices(Views.slice(Views.View{full,0},0n,(first / height : U32),1,1),full,(first / height : U32),(first % height : U32),rows,band,offset,{==},result)
    case []: Empty.absurd(Slices([],height,first,rows,full,band,offset),Words.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,Bands.has_view,None{},Some{Views.View{band,offset}},result)))
    case [a]: Empty.absurd(Slices([a],height,first,rows,full,band,offset),Words.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,Bands.has_view,None{},Some{Views.View{band,offset}},result)))
    case [a,b]: Empty.absurd(Slices([a,b],height,first,rows,full,band,offset),Words.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,Bands.has_view,None{},Some{Views.View{band,offset}},result)))
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd(Slices(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},height,first,rows,full,band,offset),
        Words.false_true(Equal.cong(Maybe<&2,Views.View>,Bool,Bands.has_view,None{},Some{Views.View{band,offset}},result)))

def slices_extents(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  slices: Slices(shape,height,first,rows,full,band,offset),+full_extents: {Views.axis_extents(full) == shape : List<&2,U32>}) ->
  {Views.axis_extents(band) == band_cells(shape,rows) : List<&2,U32>}:
  match shape:
    case [+c,+h,+w]:
      %full_extents : {Views.axis_extents(band) == Lanes.replaced(_,1n,rows) : List<&2,U32>}
      Lanes.sliced_extents(band,full,1n,(first % height : U32),rows,0,offset,slices)
    case [+n,+c,+h,+w]:
      ImageRows{+axes,+image_offset,outer,inner} = slices
      %Equal.trans(List<&2,U32>,Views.axis_extents(axes),Lanes.replaced(Views.axis_extents(full),0n,1),[1,c,h,w],
        Lanes.sliced_extents(axes,full,0n,(first / height : U32),1,0,image_offset,outer),
        Equal.cong(List<&2,U32>,List<&2,U32>,extents => Lanes.replaced(extents,0n,1),Views.axis_extents(full),[n,c,h,w],full_extents)) :
        {Views.axis_extents(band) == Lanes.replaced(_,2n,rows) : List<&2,U32>}
      Lanes.sliced_extents(band,axes,2n,(first % height : U32),rows,image_offset,offset,inner)
    case []: Empty.absurd({Views.axis_extents(band) == band_cells([],rows) : List<&2,U32>},slices)
    case [a]: Empty.absurd({Views.axis_extents(band) == band_cells([a],rows) : List<&2,U32>},slices)
    case [a,b]: Empty.absurd({Views.axis_extents(band) == band_cells([a,b],rows) : List<&2,U32>},slices)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}: Empty.absurd({Views.axis_extents(band) == band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows) : List<&2,U32>},slices)

def slices_spreadable(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,+source: List<&2,U32>,
  slices: Slices(shape,height,first,rows,full,band,offset),spread: Lanes.Spreadable(source,full)) -> Lanes.Spreadable(source,band):
  match shape:
    case [+c,+h,+w]: Lanes.sliced_spreadable(band,source,full,1n,(first % height : U32),rows,0,offset,spread,slices)
    case [+n,+c,+h,+w]:
      ImageRows{+axes,+image_offset,outer,inner} = slices
      Lanes.sliced_spreadable(band,source,axes,2n,(first % height : U32),rows,image_offset,offset,
        Lanes.sliced_spreadable(axes,source,full,0n,(first / height : U32),1,0,image_offset,spread,outer),inner)
    case []: Empty.absurd(Lanes.Spreadable(source,band),slices)
    case [a]: Empty.absurd(Lanes.Spreadable(source,band),slices)
    case [a,b]: Empty.absurd(Lanes.Spreadable(source,band),slices)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}: Empty.absurd(Lanes.Spreadable(source,band),slices)

def slices_address(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  +coordinates: List<&2,Nat>,slices: Slices(shape,height,first,rows,full,band,offset),+inside: Lanes.Inside(band,coordinates)) ->
  {Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}) == Digits.address(Lanes.points(full,pixel_of(shape,height,first,coordinates)),0n,False{}) : Nat}:
  match shape:
    case [+c,+h,+w]: Lanes.sliced_address(band,coordinates,full,1n,(first % height : U32),rows,0,offset,inside,slices)
    case [+n,+c,+h,+w]:
      ImageRows{+axes,+image_offset,outer,+inner} = slices
      +moved = Lanes.shifted(2n,(first % height : U32),coordinates)
      Equal.trans(Nat,Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}),Digits.address(Lanes.points(axes,moved),U32.to_nat(image_offset),False{}),
        Digits.address(Lanes.points(full,Lanes.shifted(0n,(first / height : U32),moved)),0n,False{}),
        Lanes.sliced_address(band,coordinates,axes,2n,(first % height : U32),rows,image_offset,offset,inside,inner),
        Lanes.sliced_address(axes,moved,full,0n,(first / height : U32),1,0,image_offset,
          Lanes.sliced_inside(band,coordinates,axes,2n,(first % height : U32),rows,image_offset,offset,inside,inner),outer))
    case []: Empty.absurd({Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}) == Digits.address(Lanes.points(full,pixel_of([],height,first,coordinates)),0n,False{}) : Nat},slices)
    case [a]: Empty.absurd({Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}) == Digits.address(Lanes.points(full,pixel_of([a],height,first,coordinates)),0n,False{}) : Nat},slices)
    case [a,b]: Empty.absurd({Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}) == Digits.address(Lanes.points(full,pixel_of([a,b],height,first,coordinates)),0n,False{}) : Nat},slices)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd({Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}) ==
        Digits.address(Lanes.points(full,pixel_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},height,first,coordinates)),0n,False{}) : Nat},slices)

def slices_inside(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  +coordinates: List<&2,Nat>,slices: Slices(shape,height,first,rows,full,band,offset),inside: Lanes.Inside(band,coordinates)) ->
  Lanes.Inside(full,pixel_of(shape,height,first,coordinates)):
  match shape:
    case [+c,+h,+w]: Lanes.sliced_inside(band,coordinates,full,1n,(first % height : U32),rows,0,offset,inside,slices)
    case [+n,+c,+h,+w]:
      ImageRows{+axes,+image_offset,outer,inner} = slices
      Lanes.sliced_inside(axes,Lanes.shifted(2n,(first % height : U32),coordinates),full,0n,(first / height : U32),1,0,image_offset,
        Lanes.sliced_inside(band,coordinates,axes,2n,(first % height : U32),rows,image_offset,offset,inside,inner),outer)
    case []: Empty.absurd(Lanes.Inside(full,pixel_of([],height,first,coordinates)),slices)
    case [a]: Empty.absurd(Lanes.Inside(full,pixel_of([a],height,first,coordinates)),slices)
    case [a,b]: Empty.absurd(Lanes.Inside(full,pixel_of([a,b],height,first,coordinates)),slices)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}: Empty.absurd(Lanes.Inside(full,pixel_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},height,first,coordinates)),slices)

# The row-major position of three coordinates.
def position_index(+channel: Nat,+rows: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(channel,rows),row),width),column) == RowMajor.index(channel,rows,width,row,column) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(channel,rows),row),width),Nat.add(Nat.mul(Nat.mul(channel,rows),width),Nat.mul(row,width)),RowMajor.distrib(Nat.mul(channel,rows),row,width)) :
    {Nat.add(_,column) == RowMajor.index(channel,rows,width,row,column) : Nat}
  RowMajor.expanded(channel,rows,width,row,column)

# A pixel of a band of image rows: the image is the quotient of the first row
# by the height and the row inside it follows the remainder.
def band_pixel(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,+limits: Machine.Limits,
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) ==
    RowMajor.index(Nat.add(Nat.mul(U32.to_nat((first / height : U32)),U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),
      Nat.add(row,U32.to_nat((first % height : U32))),column) : Nat}:
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +start = U32.to_nat(first)
  +base = U32.to_nat(height)
  +image = Nat.div(start,base)
  +offset = Nat.mod(start,base)
  +span = Inspection.local_span(height,first,rows,cap,cap_ok,positive,height_bound,local)
  +inside = Coordinates.row_inside(offset,U32.to_nat(rows),base,row,row_inside,span)
  +located = Equal.trans(Nat,Nat.add(Nat.mul(image,base),Nat.add(offset,row)),Nat.add(Nat.add(Nat.mul(image,base),offset),row),Nat.add(start,row),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,base),offset),row),Nat.add(Nat.mul(image,base),Nat.add(offset,row)),Addition.associative(Nat.mul(image,base),offset,row)),
    Equal.cong(Nat,Nat,value => Nat.add(value,row),Nat.add(Nat.mul(image,base),offset),start,
      Equal.sym(Nat,start,Nat.add(Nat.mul(image,base),offset),DivisionOrder.quotient_exact(start,base,positive))))
  %Equal.sym(Nat,U32.to_nat((first / height : U32)),image,U32Division.divide(first,height,cap,cap_ok,height_bound,positive)) :
    {BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(start,row),column) ==
      RowMajor.index(Nat.add(Nat.mul(_,U32.to_nat(channels)),channel),base,U32.to_nat(width),Nat.add(row,U32.to_nat((first % height : U32))),column) : Nat}
  %Equal.sym(Nat,U32.to_nat((first % height : U32)),offset,U32Division.modulo(first,height,cap,cap_ok,height_bound,positive)) :
    {BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(start,row),column) ==
      RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),base,U32.to_nat(width),Nat.add(row,_),column) : Nat}
  %Order.commutative(offset,row) :
    {BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(start,row),column) ==
      RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),base,U32.to_nat(width),_,column) : Nat}
  BandValues.located_pixel(Bands.Geometry{images,channels,height,width},image,channel,Nat.add(offset,row),column,Nat.add(start,row),inside,located)

# Stretched axes have one axis per extent of the target.
def stretched_length(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Lanes.Stretched(axes,target,out)) ->
  {List.length(&2,Views.Axis,axes) == List.length(&2,U32,target) : Nat}:
  match axes target out:
    case Nil{} Nil{} Nil{}: {==}
    case Con{Views.Axis{size,stride},+tail} Con{extent,+rest} Con{Views.Axis{result,step},+others}:
      Shape.Both{head,tail_lanes} = lanes
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Views.Axis,tail),List.length(&2,U32,rest),stretched_length(tail,rest,others,tail_lanes))
    case Nil{} Nil{} Con{axis,others}: Empty.absurd({List.length(&2,Views.Axis,Nil{}) == List.length(&2,U32,Nil{}) : Nat},lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd({List.length(&2,Views.Axis,Nil{}) == List.length(&2,U32,Con{extent,rest}) : Nat},lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out: Empty.absurd({List.length(&2,Views.Axis,Con{Views.Axis{size,stride},tail}) == List.length(&2,U32,Nil{}) : Nat},lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}:
      Empty.absurd({List.length(&2,Views.Axis,Con{Views.Axis{size,stride},tail}) == List.length(&2,U32,Con{extent,rest}) : Nat},lanes)

# Axes of a rank keep their axes when brought to it.
def axes_ranked(+axes: List<&2,Views.Axis>,+rank: Nat,same: {List.length(&2,Views.Axis,axes) == rank : Nat}) -> {Views.with_rank(axes,rank) == axes : List<&2,Views.Axis>}:
  +length = List.length(&2,Views.Axis,axes)
  %same : {List.append(&2,Views.Axis,List.replicate(Views.Axis,Nat.sub(_,length),Views.Axis{1,0}),axes) == axes : List<&2,Views.Axis>}
  %Addition.zero_right(length) : {List.append(&2,Views.Axis,List.replicate(Views.Axis,Nat.sub(_,length),Views.Axis{1,0}),axes) == axes : List<&2,Views.Axis>}
  %Equal.sym(Nat,Nat.sub(Nat.add(length,0n),length),0n,Order.subtract_shift(length,0n)) :
    {List.append(&2,Views.Axis,List.replicate(Views.Axis,_,Views.Axis{1,0}),axes) == axes : List<&2,Views.Axis>}
  {==}

# A lane keeps its extent when compacted against the extents it came from.
def kept_extent(single: Bool,+size: U32,+stride: U32,+extent: U32,+step: U32,classified: {U32.is_eq(size,1) == single : Bool},lane: Lanes.Lane<size,stride,extent,step>) ->
  {Bool.pick(U32,single,1,extent) == size : U32}:
  match single lane:
    case True{} lane: Equal.sym(U32,size,1,Arithmetic.equal_of_is_eq(size,1,classified))
    case False{} Lanes.Kept{same,kept}: Equal.sym(U32,size,extent,same)
    case False{} Lanes.Repeated{one,zero}:
      Empty.absurd({extent == size : U32},Words.false_true(Equal.sym(Bool,True{},False{},%one : {U32.is_eq(_,1) == False{} : Bool} classified)))

# Extents stretched to a target are their own compaction against it.
def compact_source(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Lanes.Stretched(axes,target,out)) ->
  {Bands.compact_extents(Views.axis_extents(axes),target) == Views.axis_extents(axes) : List<&2,U32>}:
  match axes target out:
    case Nil{} Nil{} Nil{}: {==}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{result,+step},+others}:
      Shape.Both{Shape.Both{resulting,lane},tail_lanes} = lanes
      %Equal.sym(List<&2,U32>,Bands.compact_extents(Views.axis_extents(tail),rest),Views.axis_extents(tail),compact_source(tail,rest,others,tail_lanes)) :
        {Bool.pick(U32,U32.is_eq(size,1),1,extent) <> _ == size <> Views.axis_extents(tail) : List<&2,U32>}
      Equal.cong(U32,List<&2,U32>,head => head <> Views.axis_extents(tail),Bool.pick(U32,U32.is_eq(size,1),1,extent),size,
        kept_extent(U32.is_eq(size,1),size,stride,extent,step,{==},lane))
    case Nil{} Nil{} Con{axis,others}: Empty.absurd({Bands.compact_extents(Views.axis_extents(Nil{}),Nil{}) == Views.axis_extents(Nil{}) : List<&2,U32>},lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd({Bands.compact_extents(Views.axis_extents(Nil{}),Con{extent,rest}) == Views.axis_extents(Nil{}) : List<&2,U32>},lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out:
      Empty.absurd({Bands.compact_extents(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{}) == Views.axis_extents(Con{Views.Axis{size,stride},tail}) : List<&2,U32>},lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}:
      Empty.absurd({Bands.compact_extents(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Con{extent,rest}) == Views.axis_extents(Con{Views.Axis{size,stride},tail}) : List<&2,U32>},lanes)

# A stretched lane compacted against the extents it came from agrees with
# any lane that agrees with the lane it came from.
def compact_lane(single: Bool,+size: U32,+stride: U32,+extent: U32,+result: U32,+step: U32,+other: U32,+other_stride: U32,
  classified: {U32.is_eq(size,1) == single : Bool},+resulting: {result == extent : U32},lane: Lanes.Lane<size,stride,extent,step>,
  +same: {size == other : U32},kept: Boolean.When(Nat.is_lt(1n,U32.to_nat(size)),{U32.to_nat(stride) == U32.to_nat(other_stride) : Nat})) ->
  Shape.Both<{Bool.pick(U32,single,1,result) == other : U32},Boolean.When(Nat.is_lt(1n,U32.to_nat(Bool.pick(U32,single,1,result))),{U32.to_nat(step) == U32.to_nat(other_stride) : Nat})>:
  match single lane:
    case True{} lane: Shape.Both{Equal.trans(U32,1,size,other,Equal.sym(U32,size,1,Arithmetic.equal_of_is_eq(size,1,classified)),same),Unit{}}
    case False{} Lanes.Kept{same_size,kept_step}:
      +grown = Equal.trans(U32,size,extent,result,same_size,Equal.sym(U32,result,extent,resulting))
      Shape.Both{Equal.trans(U32,result,size,other,Equal.sym(U32,size,result,grown),same),
        %grown : Boolean.When(Nat.is_lt(1n,U32.to_nat(_)),{U32.to_nat(step) == U32.to_nat(other_stride) : Nat})
        %Equal.sym(U32,step,stride,kept_step) : Boolean.When(Nat.is_lt(1n,U32.to_nat(size)),{U32.to_nat(_) == U32.to_nat(other_stride) : Nat})
        kept}
    case False{} Lanes.Repeated{one,zero}:
      Empty.absurd(Shape.Both<{result == other : U32},Boolean.When(Nat.is_lt(1n,U32.to_nat(result)),{U32.to_nat(step) == U32.to_nat(other_stride) : Nat})>,
        Words.false_true(Equal.sym(Bool,True{},False{},%one : {U32.is_eq(_,1) == False{} : Bool} classified)))

def compact_agree(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,+other: List<&2,Views.Axis>,
  lanes: Lanes.Stretched(axes,target,out),agree: Agree(axes,other)) -> Agree(Bands.compact_axes(Views.axis_extents(axes),out),other):
  match axes target out:
    case Nil{} Nil{} Nil{}:
      match other:
        case Nil{}: Unit{}
        case Con{Views.Axis{size,stride},others}: Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Nil{}),Nil{}),Con{Views.Axis{size,stride},others}),agree)
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{+result,+step},+others}:
      match other:
        case Con{Views.Axis{+other_size,+other_stride},+other_tail}:
          Shape.Both{Shape.Both{+resulting,lane},tail_lanes} = lanes
          Shape.Both{Shape.Both{+same,kept},tail_agree} = agree
          Shape.Both{compact_lane(U32.is_eq(size,1),size,stride,extent,result,step,other_size,other_stride,{==},resulting,lane,same,kept),
            compact_agree(tail,rest,others,other_tail,tail_lanes,tail_agree)}
        case Nil{}:
          Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Con{Views.Axis{result,step},others}),Nil{}),agree)
    case Nil{} Nil{} Con{axis,others}: Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Nil{}),Con{axis,others}),other),lanes)
    case Nil{} Con{extent,rest} out: Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Nil{}),out),other),lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out: Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Con{Views.Axis{size,stride},tail}),out),other),lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}:
      Empty.absurd(Agree(Bands.compact_axes(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{}),other),lanes)

# Extents stretch to a target when each is the target extent or one: the
# axes stretched again keep their stride, or repeat with stride zero.
def Fit(sizes: List<&2,U32>,target: List<&2,U32>) -> Data:
  match sizes target:
    case Nil{} Nil{}: Unit
    case Con{size,rest} Con{extent,others}: Shape.Both<{U32.is_eq(size,extent) || U32.is_eq(size,1) == True{} : Bool},Fit(rest,others)>
    case sizes target: Empty

def restretched_axis(same: Bool,size: U32,stride: U32,extent: U32) -> Views.Axis:
  match same:
    case True{}: Views.Axis{size,stride}
    case False{}: Views.Axis{extent,0}

def restretched(axes: List<&2,Views.Axis>,target: List<&2,U32>) -> List<&2,Views.Axis>:
  match axes target:
    case Con{Views.Axis{+size,stride},tail} Con{+extent,rest}: restretched_axis(U32.is_eq(size,extent),size,stride,extent) <> restretched(tail,rest)
    case axes target: []

def axis_some(same: Bool,single: Bool,+size: U32,+stride: U32,+extent: U32,allowed: {same || single == True{} : Bool}) ->
  {Views.stretched_axis(same,single,size,stride,extent) == Some{restretched_axis(same,size,stride,extent)} : Maybe<&2,Views.Axis>}:
  match same single:
    case True{} single: {==}
    case False{} True{}: {==}
    case False{} False{}:
      Empty.absurd({Views.stretched_axis(False{},False{},size,stride,extent) == Some{restretched_axis(False{},size,stride,extent)} : Maybe<&2,Views.Axis>},Words.false_true(allowed))

def restretch(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,fit: Fit(Views.axis_extents(axes),target)) ->
  {Views.stretch_axes(axes,target) == Some{restretched(axes,target)} : Maybe<&2,List<&2,Views.Axis>>}:
  match axes target:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest}:
      Shape.Both{+allowed,tail_fit} = fit
      +equal = U32.is_eq(size,extent)
      +single = U32.is_eq(size,1)
      %Equal.sym(Maybe<&2,List<&2,Views.Axis>>,Views.stretch_axes(tail,rest),Some{restretched(tail,rest)},restretch(tail,rest,tail_fit)) :
        {Views.stretched_prepend(Views.stretched_axis(equal,single,size,stride,extent),_) ==
          Some{restretched_axis(equal,size,stride,extent) <> restretched(tail,rest)} : Maybe<&2,List<&2,Views.Axis>>}
      %Equal.sym(Maybe<&2,Views.Axis>,Views.stretched_axis(equal,single,size,stride,extent),Some{restretched_axis(equal,size,stride,extent)},
        axis_some(equal,single,size,stride,extent,allowed)) :
        {Views.stretched_prepend(_,Some{restretched(tail,rest)}) == Some{restretched_axis(equal,size,stride,extent) <> restretched(tail,rest)} : Maybe<&2,List<&2,Views.Axis>>}
      {==}
    case Nil{} Con{extent,rest}: Empty.absurd({Views.stretch_axes(Nil{},Con{extent,rest}) == Some{restretched(Nil{},Con{extent,rest})} : Maybe<&2,List<&2,Views.Axis>>},fit)
    case Con{Views.Axis{size,stride},tail} Nil{}:
      Empty.absurd({Views.stretch_axes(Con{Views.Axis{size,stride},tail},Nil{}) == Some{restretched(Con{Views.Axis{size,stride},tail},Nil{})} : Maybe<&2,List<&2,Views.Axis>>},fit)

# Extents compacted against a target of their rank stretch to it.
def predecessor(value: Nat) -> Nat:
  match value:
    case 0n: 0n
    case 1n+rest: rest

def pick_fit(single: Bool,+extent: U32) -> {U32.is_eq(Bool.pick(U32,single,1,extent),extent) || U32.is_eq(Bool.pick(U32,single,1,extent),1) == True{} : Bool}:
  match single:
    case True{}: Boolean.or_true_right(U32.is_eq(1,extent))
    case False{}:
      %Equal.sym(Bool,U32.is_eq(extent,extent),True{},Arithmetic.equal_reflexive(extent)) : {_ || U32.is_eq(extent,1) == True{} : Bool}
      {==}

def compact_fit(+source: List<&2,U32>,+target: List<&2,U32>,same: {List.length(&2,U32,source) == List.length(&2,U32,target) : Nat}) ->
  Fit(Bands.compact_extents(source,target),target):
  match source target:
    case Nil{} Nil{}: Unit{}
    case Con{+single,+rest} Con{+extent,+others}:
      Shape.Both{pick_fit(U32.is_eq(single,1),extent),
        compact_fit(rest,others,Equal.cong(Nat,Nat,predecessor,1n+List.length(&2,U32,rest),1n+List.length(&2,U32,others),same))}
    case Nil{} Con{extent,others}:
      Empty.absurd(Fit(Bands.compact_extents(Nil{},Con{extent,others}),Con{extent,others}),
        Words.false_true(Equal.cong(Nat,Bool,length => Nat.is_lt(0n,length),0n,1n+List.length(&2,U32,others),same)))
    case Con{single,rest} Nil{}:
      Empty.absurd(Fit(Bands.compact_extents(Con{single,rest},Nil{}),Nil{}),
        Words.false_true(Equal.cong(Nat,Bool,length => Nat.is_lt(0n,length),0n,1n+List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same))))

def out_nonempty(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,+lanes: Lanes.Stretched(axes,target,out),
  +nonempty: {Shapes.has_zero(target) == False{} : Bool}) -> {Shapes.has_zero(Views.axis_extents(out)) == False{} : Bool}:
  %Equal.sym(List<&2,U32>,Views.axis_extents(out),target,Lanes.stretched_extents(axes,target,out,lanes)) : {Shapes.has_zero(_) == False{} : Bool}
  nonempty

# The last address a contiguous view stretched to a nonempty target reads
# lies inside the elements of the view.
def kernel_bound(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,+count: U32,+lanes: Lanes.Stretched(axes,target,out),
  layout: Linear.Layout(axes),+nonempty: {Shapes.has_zero(target) == False{} : Bool},exact: {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat}) ->
  {Nat.is_lt(Digits.address(Lanes.points(out,Lanes.lasts(out)),0n,False{}),U32.to_nat(count)) == True{} : Bool}:
  +lasts = Lanes.lasts(out)
  +masked = Lanes.masked(Views.axis_extents(axes),lasts)
  +inside = Lanes.lasts_inside(out,out_nonempty(axes,target,out,lanes,nonempty))
  +inner = Lanes.stretched_inside(out,lasts,axes,target,lanes,inside)
  %Equal.sym(Nat,Digits.address(Lanes.points(out,lasts),0n,False{}),Lanes.position(Views.axis_extents(axes),masked,0n),
    Equal.trans(Nat,Digits.address(Lanes.points(out,lasts),0n,False{}),Digits.address(Lanes.points(axes,masked),0n,False{}),Lanes.position(Views.axis_extents(axes),masked,0n),
      Lanes.stretched_address(out,lasts,axes,target,0n,lanes,inside),Lanes.canonical_address(axes,masked,layout,inner))) :
    {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
  %exact : {Nat.is_lt(Lanes.position(Views.axis_extents(axes),masked,0n),_) == True{} : Bool}
  Lanes.position_inside(axes,masked,inner)

# A contiguous view stretched to a target fits the elements of the view, and
# reads at the position of inside coordinates the position of the
# coordinates its axes read.
def kernel_fits(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,+count: U32,+lanes: Lanes.Stretched(axes,target,out),
  layout: Linear.Layout(axes),+nonempty: {Shapes.has_zero(target) == False{} : Bool},exact: {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat}) ->
  {Views.fits(Views.View{out,0},count) == True{} : Bool}:
  Lanes.fits_bound(out,0,count,out_nonempty(axes,target,out,lanes,nonempty),kernel_bound(axes,target,out,count,lanes,layout,nonempty,exact))

def kernel_address(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,
  +lanes: Lanes.Stretched(axes,target,out),layout: Linear.Layout(axes),+inside: Lanes.Inside(out,coordinates)) ->
  {Copy.address(Views.View{out,0},Lanes.position(target,coordinates,0n)) ==
    U32.from_nat(Lanes.position(Views.axis_extents(axes),Lanes.masked(Views.axis_extents(axes),coordinates),0n)) : U32}:
  +masked = Lanes.masked(Views.axis_extents(axes),coordinates)
  %Lanes.stretched_extents(axes,target,out,lanes) :
    {Copy.address(Views.View{out,0},Lanes.position(_,coordinates,0n)) == U32.from_nat(Lanes.position(Views.axis_extents(axes),masked,0n)) : U32}
  Equal.trans(U32,Copy.address(Views.View{out,0},Lanes.position(Views.axis_extents(out),coordinates,0n)),U32.from_nat(Digits.address(Lanes.points(out,coordinates),0n,False{})),
    U32.from_nat(Lanes.position(Views.axis_extents(axes),masked,0n)),
    Lanes.address(out,0,coordinates,inside),
    Equal.cong(Nat,U32,U32.from_nat,Digits.address(Lanes.points(out,coordinates),0n,False{}),Lanes.position(Views.axis_extents(axes),masked,0n),
      Equal.trans(Nat,Digits.address(Lanes.points(out,coordinates),0n,False{}),Digits.address(Lanes.points(axes,masked),0n,False{}),Lanes.position(Views.axis_extents(axes),masked,0n),
        Lanes.stretched_address(out,coordinates,axes,target,0n,lanes,inside),
        Lanes.canonical_address(axes,masked,layout,Lanes.stretched_inside(out,coordinates,axes,target,lanes,inside)))))

# A band of a whole image stores each cell at its pixel.
def whole_position(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  +whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) ==
    BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat}:
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +leading = (U32.is_eq(images,1) && U32.is_zero(first) : Bool)
  +first_zero = Arithmetic.equal_of_is_eq(first,0,Boolean.right(U32.is_eq(images,1),U32.is_zero(first),Boolean.left(leading,U32.is_eq(rows,height),whole)))
  +rows_height = Arithmetic.equal_of_is_eq(rows,height,Boolean.right(leading,U32.is_eq(rows,height),whole))
  +inside = {
    %rows_height : {Nat.is_lt(row,U32.to_nat(_)) == True{} : Bool}
    row_inside : {Nat.is_lt(row,BandValues.height(geometry)) == True{} : Bool}}
  %Equal.sym(U32,first,0,first_zero) :
    {BandValues.position(geometry,rows,channel,row,column) == BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(_),row),column) : Nat}
  %Equal.sym(U32,rows,height,rows_height) : {BandValues.position(geometry,_,channel,row,column) == BandValues.pixel(geometry,channel,row,column) : Nat}
  Equal.sym(Nat,BandValues.pixel(geometry,channel,row,column),BandValues.point(geometry,0n,channel,row,column),
    BandValues.located_pixel(geometry,0n,channel,row,column,row,inside,{==}))

# Axes that agree with the stretched source read, at the position of a cell
# of a band of a whole image, the stretched source at its pixel.
def whole_address(+axes: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,
  +channel: Nat,+row: Nat,+column: Nat,agree: Agree(axes,full),whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {Copy.address(Views.View{axes,0},BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)) ==
    Copy.address(Views.View{full,0},BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column)) : U32}:
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +position = BandValues.position(geometry,rows,channel,row,column)
  +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
  Equal.trans(U32,Copy.address(Views.View{axes,0},position),Copy.address(Views.View{full,0},position),Copy.address(Views.View{full,0},pixel),
    agreed_address(axes,full,0,position,agree),
    Equal.cong(Nat,U32,index => Copy.address(Views.View{full,0},index),position,pixel,whole_position(images,channels,height,width,first,rows,channel,row,column,whole,row_inside)))

# The part a band copies from the source: the source at the addresses of
# the compacted band view.
def piece_tree(~Element: Data,+view: Views.View,count: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Writes.pure(Element,Copy.copy_writes(~Element,Views.is_contiguous(view),U32.to_nat(count),view,0,input),output)

# A band of a whole image reads its part at its pixels: the source itself, or
# its copy through axes that agree with the contiguous view of its extents.
def whole_copy_read(~Element: Data,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+length: Nat,+copied: List<&2,Views.Axis>,+compact: List<&2,U32>,+count: U32,
  +address: U32,
  copy_agree: Agree(copied,Canonical.axes(Canonical.canonical_view(compact,count))),+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},
  +copied_extents: {Views.axis_extents(copied) == compact : List<&2,U32>},
  ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(address),U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,piece_tree(~Element,Views.View{copied,0},count,input,output),address) == Tree.read(Element,input,address) : Element}:
  +index = U32.to_nat(address)
  +canonical = Canonical.canonical_view(compact,count)
  +canonical_inside = {
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(canonical),compact,Canonical.canonical_shape(compact,count,exact)) : {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(compact),U32.to_nat(count),exact) : {Nat.is_lt(index,_) == True{} : Bool}
    inside : {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents_of(canonical))) == True{} : Bool}}
  +copied_exact = {
    %Equal.sym(List<&2,U32>,Views.axis_extents(copied),compact,copied_extents) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    exact : {ShapeLaws.elements(Views.axis_extents_of(Views.View{copied,0})) == U32.to_nat(count) : Nat}}
  %Arithmetic.native_roundtrip(address) : {Tree.read(Element,piece_tree(~Element,Views.View{copied,0},count,input,output),_) == Tree.read(Element,input,_) : Element}
  Equal.trans(Element,Tree.read(Element,piece_tree(~Element,Views.View{copied,0},count,input,output),U32.from_nat(index)),
    Tree.read(Element,input,Copy.address(Views.View{copied,0},index)),Tree.read(Element,input,U32.from_nat(index)),
    Copy.copied_value(~Element,Views.View{copied,0},input,output,length,U32.to_nat(count),index,ready,capacity_ok,room,inside,copied_exact),
    Equal.cong(U32,Element,word => Tree.read(Element,input,word),Copy.address(Views.View{copied,0},index),U32.from_nat(index),
      Equal.trans(U32,Copy.address(Views.View{copied,0},index),Copy.address(canonical,index),U32.from_nat(index),
        agreed_address(copied,Canonical.axes(canonical),0,index,copy_agree),
        Copy.zero_contiguous_address(canonical,index,Canonical.canonical_contiguous(compact,count,exact),canonical_inside))))

# A band of image rows reads its part through the contiguous view of the
# part stretched to its cells: the compacted band view at the coordinates
# its axes read, which is the stretched source at the pixel of the cell.
def rows_address(+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+coordinates: List<&2,Nat>,+slices: Slices(shape,height,first,rows,full,band,offset),
  +spread: Lanes.Spreadable(source,band),+same: {Views.axis_extents(axes) == Views.axis_extents(Bands.compact_axes(source,band)) : List<&2,U32>},
  +full_extents: {Views.axis_extents(full) == shape : List<&2,U32>},+inside: Lanes.Inside(band,coordinates)) ->
  {Copy.address(Views.View{Bands.compact_axes(source,band),offset},Lanes.position(Views.axis_extents(axes),Lanes.masked(Views.axis_extents(axes),coordinates),0n)) ==
    Copy.address(Views.View{full,0},Lanes.position(shape,pixel_of(shape,height,first,coordinates),0n)) : U32}:
  +copied = Bands.compact_axes(source,band)
  +masked = Lanes.masked(Views.axis_extents(copied),coordinates)
  +pixel = pixel_of(shape,height,first,coordinates)
  %Equal.sym(List<&2,U32>,Views.axis_extents(axes),Views.axis_extents(copied),same) :
    {Copy.address(Views.View{copied,offset},Lanes.position(_,Lanes.masked(_,coordinates),0n)) == Copy.address(Views.View{full,0},Lanes.position(shape,pixel,0n)) : U32}
  %full_extents : {Copy.address(Views.View{copied,offset},Lanes.position(Views.axis_extents(copied),masked,0n)) == Copy.address(Views.View{full,0},Lanes.position(_,pixel,0n)) : U32}
  Equal.trans(U32,Copy.address(Views.View{copied,offset},Lanes.position(Views.axis_extents(copied),masked,0n)),
    U32.from_nat(Digits.address(Lanes.points(copied,masked),U32.to_nat(offset),False{})),Copy.address(Views.View{full,0},Lanes.position(Views.axis_extents(full),pixel,0n)),
    Lanes.address(copied,offset,masked,Lanes.compact_inside(band,coordinates,source,spread,inside)),
    Equal.trans(U32,U32.from_nat(Digits.address(Lanes.points(copied,masked),U32.to_nat(offset),False{})),U32.from_nat(Digits.address(Lanes.points(full,pixel),0n,False{})),
      Copy.address(Views.View{full,0},Lanes.position(Views.axis_extents(full),pixel,0n)),
      Equal.cong(Nat,U32,U32.from_nat,Digits.address(Lanes.points(copied,masked),U32.to_nat(offset),False{}),Digits.address(Lanes.points(full,pixel),0n,False{}),
        Equal.trans(Nat,Digits.address(Lanes.points(copied,masked),U32.to_nat(offset),False{}),Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}),
          Digits.address(Lanes.points(full,pixel),0n,False{}),
          Equal.sym(Nat,Digits.address(Lanes.points(band,coordinates),U32.to_nat(offset),False{}),Digits.address(Lanes.points(copied,masked),U32.to_nat(offset),False{}),
            Lanes.compact_address(band,coordinates,source,U32.to_nat(offset),spread,inside)),
          slices_address(shape,height,first,rows,full,band,offset,coordinates,slices,inside))),
      Equal.sym(U32,Copy.address(Views.View{full,0},Lanes.position(Views.axis_extents(full),pixel,0n)),U32.from_nat(Digits.address(Lanes.points(full,pixel),0n,False{})),
        Lanes.address(full,0,pixel,slices_inside(shape,height,first,rows,full,band,offset,coordinates,slices,inside)))))

def rows_read(~Element: Data,+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+kernel: List<&2,Views.Axis>,+count: U32,+coordinates: List<&2,Nat>,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+length: Nat,
  +slices: Slices(shape,height,first,rows,full,band,offset),+spread: Lanes.Spreadable(source,band),
  +lanes: Lanes.Stretched(axes,band_cells(shape,rows),kernel),layout: Linear.Layout(axes),
  +same: {Views.axis_extents(axes) == Views.axis_extents(Bands.compact_axes(source,band)) : List<&2,U32>},
  +full_extents: {Views.axis_extents(full) == shape : List<&2,U32>},+exact: {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat},
  ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},+below: Below(band_cells(shape,rows),coordinates)) ->
  {Tree.read(Element,piece_tree(~Element,Views.View{Bands.compact_axes(source,band),offset},count,input,output),
      Copy.address(Views.View{kernel,0},Lanes.position(band_cells(shape,rows),coordinates,0n))) ==
    Tree.read(Element,input,Copy.address(Views.View{full,0},Lanes.position(shape,pixel_of(shape,height,first,coordinates),0n))) : Element}:
  +cells = band_cells(shape,rows)
  +copied = Bands.compact_axes(source,band)
  +masked = Lanes.masked(Views.axis_extents(axes),coordinates)
  +index = Lanes.position(Views.axis_extents(axes),masked,0n)
  +kernel_inside = below_inside(kernel,coordinates,cells,Lanes.stretched_extents(axes,cells,kernel,lanes),below)
  +band_inside = below_inside(band,coordinates,cells,slices_extents(shape,height,first,rows,full,band,offset,slices,full_extents),below)
  +index_count = {
    %exact : {Nat.is_lt(index,_) == True{} : Bool}
    Lanes.position_inside(axes,masked,Lanes.stretched_inside(kernel,coordinates,axes,cells,lanes,kernel_inside)) : {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  +copied_exact = {
    %same : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    exact : {ShapeLaws.elements(Views.axis_extents_of(Views.View{copied,offset})) == U32.to_nat(count) : Nat}}
  %Equal.sym(U32,Copy.address(Views.View{kernel,0},Lanes.position(cells,coordinates,0n)),U32.from_nat(index),
    kernel_address(axes,cells,kernel,coordinates,lanes,layout,kernel_inside)) :
    {Tree.read(Element,piece_tree(~Element,Views.View{copied,offset},count,input,output),_) ==
      Tree.read(Element,input,Copy.address(Views.View{full,0},Lanes.position(shape,pixel_of(shape,height,first,coordinates),0n))) : Element}
  Equal.trans(Element,Tree.read(Element,piece_tree(~Element,Views.View{copied,offset},count,input,output),U32.from_nat(index)),
    Tree.read(Element,input,Copy.address(Views.View{copied,offset},index)),
    Tree.read(Element,input,Copy.address(Views.View{full,0},Lanes.position(shape,pixel_of(shape,height,first,coordinates),0n))),
    Copy.copied_value(~Element,Views.View{copied,offset},input,output,length,U32.to_nat(count),index,ready,capacity_ok,room,index_count,copied_exact),
    Equal.cong(U32,Element,word => Tree.read(Element,input,word),Copy.address(Views.View{copied,offset},index),
      Copy.address(Views.View{full,0},Lanes.position(shape,pixel_of(shape,height,first,coordinates),0n)),
      rows_address(shape,height,first,rows,full,band,offset,source,axes,coordinates,slices,spread,same,full_extents,band_inside)))

# A shape of image rows names the extents of its geometry.
def ImageShape(shape: List<&2,U32>,geometry: Bands.Geometry) -> Data:
  match shape geometry:
    case [c,h,w] Bands.Geometry{n,ci,hi,wi}: Shape.Both<{n == 1 : U32},Shape.Both<{c == ci : U32},Shape.Both<{h == hi : U32},{w == wi : U32}>>>
    case [n,c,h,w] Bands.Geometry{ni,ci,hi,wi}: Shape.Both<{n == ni : U32},Shape.Both<{c == ci : U32},Shape.Both<{h == hi : U32},{w == wi : U32}>>>
    case shape geometry: Empty

def four_equal(+a: U32,+b: U32,+c: U32,+d: U32,+e: U32,+f: U32,+g: U32,+h: U32,
  +equal: {U32.is_eq(a,b) && U32.is_eq(c,d) && U32.is_eq(e,f) && U32.is_eq(g,h) == True{} : Bool}) ->
  Shape.Both<{a == b : U32},Shape.Both<{c == d : U32},Shape.Both<{e == f : U32},{g == h : U32}>>>:
  +two = (U32.is_eq(a,b) && U32.is_eq(c,d) : Bool)
  +three = (U32.is_eq(a,b) && U32.is_eq(c,d) && U32.is_eq(e,f) : Bool)
  +three_true = Boolean.left(three,U32.is_eq(g,h),equal)
  +two_true = Boolean.left(two,U32.is_eq(e,f),three_true)
  Shape.Both{Arithmetic.equal_of_is_eq(a,b,Boolean.left(U32.is_eq(a,b),U32.is_eq(c,d),two_true)),
    Shape.Both{Arithmetic.equal_of_is_eq(c,d,Boolean.right(U32.is_eq(a,b),U32.is_eq(c,d),two_true)),
      Shape.Both{Arithmetic.equal_of_is_eq(e,f,Boolean.right(two,U32.is_eq(e,f),three_true)),
        Arithmetic.equal_of_is_eq(g,h,Boolean.right(three,U32.is_eq(g,h),equal))}}}

def image_shape(+shape: List<&2,U32>,+geometry: Bands.Geometry,rows_shaped: {Bands.image_rows(shape,geometry) == True{} : Bool}) -> ImageShape(shape,geometry):
  match shape geometry:
    case [+c,+h,+w] Bands.Geometry{+n,+ci,+hi,+wi}: four_equal(n,1,c,ci,h,hi,w,wi,rows_shaped)
    case [+n,+c,+h,+w] Bands.Geometry{+ni,+ci,+hi,+wi}: four_equal(n,ni,c,ci,h,hi,w,wi,rows_shaped)
    case [] Bands.Geometry{n,c,h,w}: Empty.absurd(ImageShape([],Bands.Geometry{n,c,h,w}),Words.false_true(rows_shaped))
    case [a] Bands.Geometry{n,c,h,w}: Empty.absurd(ImageShape([a],Bands.Geometry{n,c,h,w}),Words.false_true(rows_shaped))
    case [a,b] Bands.Geometry{n,c,h,w}: Empty.absurd(ImageShape([a,b],Bands.Geometry{n,c,h,w}),Words.false_true(rows_shaped))
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}} Bands.Geometry{n,ci,hi,wi}:
      Empty.absurd(ImageShape(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},Bands.Geometry{n,ci,hi,wi}),Words.false_true(rows_shaped))

# The coordinates of a cell of a band of image rows: channel, row and column,
# after the image for four axes.
def cell_of(shape: List<&2,U32>,channel: Nat,row: Nat,column: Nat) -> List<&2,Nat>:
  match shape:
    case [c,h,w]: [channel,row,column]
    case [n,c,h,w]: [0n,channel,row,column]
    case shape: []

def cell_below(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  Below(band_cells(shape,rows),cell_of(shape,channel,row,column)):
  match shape:
    case [+c,+h,+w]:
      Shape.Both{one,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      Shape.Both{%Equal.sym(U32,c,channels,same_c) : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} channel_inside,
        Shape.Both{row_inside,Shape.Both{%Equal.sym(U32,w,width,same_w) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool} column_inside,Unit{}}}}
    case [+n,+c,+h,+w]:
      Shape.Both{same_n,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      Shape.Both{{==},Shape.Both{%Equal.sym(U32,c,channels,same_c) : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} channel_inside,
        Shape.Both{row_inside,Shape.Both{%Equal.sym(U32,w,width,same_w) : {Nat.is_lt(column,U32.to_nat(_)) == True{} : Bool} column_inside,Unit{}}}}}
    case []: Empty.absurd(Below(band_cells([],rows),cell_of([],channel,row,column)),image)
    case [a]: Empty.absurd(Below(band_cells([a],rows),cell_of([a],channel,row,column)),image)
    case [a,b]: Empty.absurd(Below(band_cells([a,b],rows),cell_of([a,b],channel,row,column)),image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd(Below(band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows),cell_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},channel,row,column)),image)

def cell_position(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width})) ->
  {Lanes.position(band_cells(shape,rows),cell_of(shape,channel,row,column),0n) == BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) : Nat}:
  match shape:
    case [+c,+h,+w]:
      Shape.Both{one,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      %same_w :
        {Nat.add(Nat.mul(Nat.add(Nat.mul(channel,U32.to_nat(rows)),row),U32.to_nat(w)),column) == RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(_),row,column) : Nat}
      position_index(channel,U32.to_nat(rows),U32.to_nat(w),row,column)
    case [+n,+c,+h,+w]:
      Shape.Both{same_n,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      %same_w :
        {Nat.add(Nat.mul(Nat.add(Nat.mul(channel,U32.to_nat(rows)),row),U32.to_nat(w)),column) == RowMajor.index(channel,U32.to_nat(rows),U32.to_nat(_),row,column) : Nat}
      position_index(channel,U32.to_nat(rows),U32.to_nat(w),row,column)
    case []: Empty.absurd({Lanes.position(band_cells([],rows),cell_of([],channel,row,column),0n) == BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) : Nat},image)
    case [a]: Empty.absurd({Lanes.position(band_cells([a],rows),cell_of([a],channel,row,column),0n) == BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) : Nat},image)
    case [a,b]: Empty.absurd({Lanes.position(band_cells([a,b],rows),cell_of([a,b],channel,row,column),0n) == BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) : Nat},image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd({Lanes.position(band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows),cell_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},channel,row,column),0n) ==
        BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column) : Nat},image)

# The row-major index of a pixel named by its image, channel, row and column
# over the extents of the shape is the pixel of the geometry.
def index_pixel(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,+limits: Machine.Limits,
  +image: Nat,+c: U32,+h: U32,+w: U32,same_c: {c == channels : U32},same_h: {h == height : U32},same_w: {w == width : U32},
  image_exact: {image == U32.to_nat((first / height : U32)) : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(c)),channel),U32.to_nat(h),U32.to_nat(w),Nat.add(row,U32.to_nat((first % height : U32))),column) ==
    BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat}:
  +inner = Nat.add(row,U32.to_nat((first % height : U32)))
  +pixel = BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column)
  %Equal.sym(U32,c,channels,same_c) : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(_)),channel),U32.to_nat(h),U32.to_nat(w),inner,column) == pixel : Nat}
  %Equal.sym(U32,h,height,same_h) : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(_),U32.to_nat(w),inner,column) == pixel : Nat}
  %Equal.sym(U32,w,width,same_w) : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(_),inner,column) == pixel : Nat}
  %Equal.sym(Nat,image,U32.to_nat((first / height : U32)),image_exact) :
    {RowMajor.index(Nat.add(Nat.mul(_,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),inner,column) == pixel : Nat}
  Equal.sym(Nat,pixel,RowMajor.index(Nat.add(Nat.mul(U32.to_nat((first / height : U32)),U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),inner,column),
    band_pixel(images,channels,height,width,first,rows,channel,row,column,limits,positive,height_bound,local,row_inside))

# A band of one image starts in its first image.
def first_image(+height: U32,+first: U32,+rows: U32,+row: Nat,+limits: Machine.Limits,
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(1n,U32.to_nat(height))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) -> {0n == U32.to_nat((first / height : U32)) : Nat}:
  +start = U32.to_nat(first)
  +base = U32.to_nat(height)
  +fits = {
    %Addition.zero_right(base) : {Nat.is_le(Nat.add(start,U32.to_nat(rows)),_) == True{} : Bool}
    rows_fit : {Nat.is_le(Nat.add(start,U32.to_nat(rows)),base) == True{} : Bool}}
  +below = Order.le_lt_trans(start,Nat.add(start,row),base,Order.le_add(start,row),Coordinates.row_inside(start,U32.to_nat(rows),base,row,row_inside,fits))
  Equal.trans(Nat,0n,Nat.div(start,base),U32.to_nat((first / height : U32)),
    Equal.sym(Nat,Nat.div(start,base),0n,Radix.quotient(0n,start,base,positive,below)),
    Equal.sym(Nat,U32.to_nat((first / height : U32)),Nat.div(start,base),U32Division.divide(first,height,Machine.half_word(limits),Limits.half_word_fits(limits),height_bound,positive)))

def cell_pixel(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,+limits: Machine.Limits,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {Lanes.position(shape,pixel_of(shape,height,first,cell_of(shape,channel,row,column)),0n) ==
    BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat}:
  match shape:
    case [+c,+h,+w]:
      Shape.Both{one,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      +inner = Nat.add(row,U32.to_nat((first % height : U32)))
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(channel,U32.to_nat(h)),inner),U32.to_nat(w)),column),RowMajor.index(channel,U32.to_nat(h),U32.to_nat(w),inner,column),
        BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column),
        position_index(channel,U32.to_nat(h),U32.to_nat(w),inner,column),
        index_pixel(images,channels,height,width,first,rows,channel,row,column,limits,0n,c,h,w,same_c,same_h,same_w,
          first_image(height,first,rows,row,limits,positive,height_bound,
            %one : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(_),U32.to_nat(height))) == True{} : Bool} rows_fit,row_inside),
          positive,height_bound,local,row_inside))
    case [+n,+c,+h,+w]:
      Shape.Both{same_n,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      +image_index = U32.to_nat((first / height : U32))
      +inner = Nat.add(row,U32.to_nat((first % height : U32)))
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image_index,U32.to_nat(c)),channel),U32.to_nat(h)),inner),U32.to_nat(w)),column),
        RowMajor.index(Nat.add(Nat.mul(image_index,U32.to_nat(c)),channel),U32.to_nat(h),U32.to_nat(w),inner,column),
        BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column),
        position_index(Nat.add(Nat.mul(image_index,U32.to_nat(c)),channel),U32.to_nat(h),U32.to_nat(w),inner,column),
        index_pixel(images,channels,height,width,first,rows,channel,row,column,limits,image_index,c,h,w,same_c,same_h,same_w,{==},positive,height_bound,local,row_inside))
    case []: Empty.absurd({Lanes.position([],pixel_of([],height,first,cell_of([],channel,row,column)),0n) ==
      BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat},image)
    case [a]: Empty.absurd({Lanes.position([a],pixel_of([a],height,first,cell_of([a],channel,row,column)),0n) ==
      BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat},image)
    case [a,b]: Empty.absurd({Lanes.position([a,b],pixel_of([a,b],height,first,cell_of([a,b],channel,row,column)),0n) ==
      BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat},image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd({Lanes.position(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},pixel_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},height,first,cell_of(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},channel,row,column)),0n) ==
        BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column) : Nat},image)

# Axes stretched to their own extents keep every axis.
def self_stretched(+axes: List<&2,Views.Axis>) -> Lanes.Stretched(axes,Views.axis_extents(axes),axes):
  match axes:
    case []: Unit{}
    case Views.Axis{+extent,+stride} <> +tail: Shape.Both{Shape.Both{{==},Lanes.Kept{{==},{==}}},self_stretched(tail)}

# The contiguous view of extents without zero is laid out row-major and fits
# its count.
def canonical_layout(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},
  nonempty: {Shapes.has_zero(extents) == False{} : Bool}) -> Linear.Layout(Canonical.axes(Canonical.canonical_view(extents,count))):
  Linear.checked_layout(Canonical.axes(Canonical.canonical_view(extents,count)),0,
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(extents,count)),extents,Canonical.canonical_shape(extents,count,exact)) :
      {Shapes.has_zero(_) == False{} : Bool} nonempty,
    Canonical.canonical_contiguous(extents,count,exact))

def fits_empty(+axes: List<&2,Views.Axis>,+count: U32,empty: {Shapes.has_zero(Views.axis_extents(axes)) == True{} : Bool}) ->
  {Views.fits(Views.View{axes,0},count) == True{} : Bool}:
  %Equal.sym(Bool,Shapes.has_zero(Views.axis_extents(axes)),True{},empty) : {Views.fits_nonempty(_,Views.View{axes,0},count) == True{} : Bool}
  U32Division.nat_le_native(0,count,Order.zero_le(U32.to_nat(count)))

def canonical_fits_when(empty: Bool,+extents: List<&2,U32>,+count: U32,+classified: {Shapes.has_zero(extents) == empty : Bool},
  +exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) -> {Views.fits(Canonical.canonical_view(extents,count),count) == True{} : Bool}:
  match empty:
    case True{}:
      +axes = Canonical.axes(Canonical.canonical_view(extents,count))
      +shaped = Canonical.canonical_shape(extents,count,exact)
      fits_empty(axes,count,%Equal.sym(List<&2,U32>,Views.axis_extents(axes),extents,shaped) : {Shapes.has_zero(_) == True{} : Bool} classified)
    case False{}:
      +axes = Canonical.axes(Canonical.canonical_view(extents,count))
      +shaped = Canonical.canonical_shape(extents,count,exact)
      kernel_fits(axes,Views.axis_extents(axes),axes,count,self_stretched(axes),canonical_layout(extents,count,exact,classified),
        %Equal.sym(List<&2,U32>,Views.axis_extents(axes),extents,shaped) : {Shapes.has_zero(_) == False{} : Bool} classified,
        %Equal.sym(List<&2,U32>,Views.axis_extents(axes),extents,shaped) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat} exact)

def canonical_fits(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat}) ->
  {Views.fits(Canonical.canonical_view(extents,count),count) == True{} : Bool}:
  canonical_fits_when(Shapes.has_zero(extents),extents,count,{==},exact)

# The contiguous view of compacted extents stretched to a target fits their
# count.
def stretched_fits_when(empty: Bool,+compact: List<&2,U32>,+count: U32,+target: List<&2,U32>,+out: List<&2,Views.Axis>,
  +lanes: Lanes.Stretched(Canonical.axes(Canonical.canonical_view(compact,count)),target,out),+classified: {Shapes.has_zero(target) == empty : Bool},
  +exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat}) -> {Views.fits(Views.View{out,0},count) == True{} : Bool}:
  match empty:
    case True{}:
      +axes = Canonical.axes(Canonical.canonical_view(compact,count))
      fits_empty(out,count,%Equal.sym(List<&2,U32>,Views.axis_extents(out),target,Lanes.stretched_extents(axes,target,out,lanes)) : {Shapes.has_zero(_) == True{} : Bool} classified)
    case False{}:
      +axes = Canonical.axes(Canonical.canonical_view(compact,count))
      +shaped = Canonical.canonical_shape(compact,count,exact)
      kernel_fits(axes,target,out,count,lanes,
        canonical_layout(compact,count,exact,
          %shaped : {Shapes.has_zero(_) == False{} : Bool} stretched_nonempty(axes,target,out,lanes,classified)),
        classified,%Equal.sym(List<&2,U32>,Views.axis_extents(axes),compact,shaped) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat} exact)

def stretched_fits(+compact: List<&2,U32>,+count: U32,+target: List<&2,U32>,+out: List<&2,Views.Axis>,
  +lanes: Lanes.Stretched(Canonical.axes(Canonical.canonical_view(compact,count)),target,out),+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat}) ->
  {Views.fits(Views.View{out,0},count) == True{} : Bool}:
  stretched_fits_when(Shapes.has_zero(target),compact,count,target,out,lanes,{==},exact)

# A ready owner whose length fits a view is a valid owner of it.
def valid_view(~Element: Data,+input: Buffer.Model<Element>,+view: Views.View,ready: Buffer.Sized<Element,input>,
  fits: {Views.fits(view,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}) ->
  {OperationBounds.valid(~Element,Buffer.pack(~Element,input),view) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Buffer.Sized{depth_ok,+room} = ready
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) : {Operations.valid_buffer(length,_,view) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))),True{},room) : {_ && Views.fits(view,U32.from_nat(length)) == True{} : Bool}
  %Equal.sym(Bool,Views.fits(view,U32.from_nat(length)),True{},fits) : {True{} && _ == True{} : Bool}
  {==}

# The kernel on a band and its part passes the guard of the checked
# combination with the contiguous view of the cells on the left.
def spread_guard(~Element: Data,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+kernel: List<&2,Views.Axis>,+count: U32,
  left_ready: Buffer.Sized<Element,left>,piece_ready: Buffer.Sized<Element,piece>,
  +exact: {ShapeLaws.elements(Views.axis_extents(kernel)) == U32.to_nat(count) : Nat},written: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},
  fits: {Views.fits(Views.View{kernel,0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool}) ->
  {OperationBounds.combine_guard(~Element,Buffer.pack(~Element,left),Canonical.canonical_view(Views.axis_extents(kernel),count),Buffer.pack(~Element,piece),Views.View{kernel,0}) == True{} : Bool}:
  +extents = Views.axis_extents(kernel)
  +target = Canonical.canonical_view(extents,count)
  +source = {Views.View{kernel,0} : Views.View}
  +same = {
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(target),extents,Canonical.canonical_shape(extents,count,exact)) : {Shapes.same_extents(_,extents) == True{} : Bool}
    ShapeLaws.same_extents_reflexive(extents) : {Shapes.same(Views.shape(target),Views.shape(source)) == True{} : Bool}}
  +left_valid = valid_view(~Element,left,target,left_ready,
    %Equal.sym(U32,U32.from_nat(Buffer.length(~Element,left)),count,Arithmetic.from_nat_exact(count,Buffer.length(~Element,left),written)) : {Views.fits(target,_) == True{} : Bool}
    canonical_fits(extents,count,exact))
  +right_valid = valid_view(~Element,piece,source,piece_ready,fits)
  %Equal.sym(Bool,Shapes.same(Views.shape(target),Views.shape(source)),True{},same) :
    {_ && Views.is_contiguous(target) && OperationBounds.valid(~Element,Buffer.pack(~Element,left),target) && OperationBounds.valid(~Element,Buffer.pack(~Element,piece),source) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(target),True{},Canonical.canonical_contiguous(extents,count,exact)) :
    {True{} && _ && OperationBounds.valid(~Element,Buffer.pack(~Element,left),target) && OperationBounds.valid(~Element,Buffer.pack(~Element,piece),source) == True{} : Bool}
  %Equal.sym(Bool,OperationBounds.valid(~Element,Buffer.pack(~Element,left),target),True{},left_valid) :
    {True{} && True{} && _ && OperationBounds.valid(~Element,Buffer.pack(~Element,piece),source) == True{} : Bool}
  %Equal.sym(Bool,OperationBounds.valid(~Element,Buffer.pack(~Element,piece),source),True{},right_valid) : {True{} && True{} && True{} && _ == True{} : Bool}
  {==}

# The left owner of a kernel call holds the left content the call observes.
def kernel_left_content(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.kernel_left(~Element,pair)) == BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.observe_pair(~Element,pair)}) : Buffer.Content<Element>}:
  match pair:
    case (Storage.Buffer{storage,length,certificate},right): {==}

# A band combined with its part holds the kernel combination through the
# kernel view.
def kernel_content(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+axes: List<&2,Views.Axis>,+count: U32,
  counted: {Views.numel(Canonical.canonical_view(Views.axis_extents(axes),count)) == Some{count} : Maybe<&2,U32>},
  accepted: {OperationBounds.combine_guard(~Element,Buffer.pack(~Element,left),Canonical.canonical_view(Views.axis_extents(axes),count),Buffer.pack(~Element,piece),Views.View{axes,0}) == True{} : Bool}) ->
  {Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,count,Some{Views.View{axes,0}},Buffer.pack(~Element,left),Buffer.pack(~Element,piece))) ==
    CombineKernel.combined_content(~Element,~operation,count,Views.View{axes,0},Buffer.content(~Element,left),Buffer.content(~Element,piece)) : Buffer.Content<Element>}:
  match left piece:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+piece_tree,+piece_length,+piece_certificate}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +piece = {Buffer.Model{piece_tree,piece_length,piece_certificate} : Buffer.Model<Element>}
      +source = {Views.View{axes,0} : Views.View}
      +result = AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,piece_tree,left_tree)
      Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.kernel_left(~Element,kernel(U32.to_nat(count),source,0,Buffer.pack(~Element,left),Buffer.pack(~Element,piece)))),
        BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.observe_pair(~Element,kernel(U32.to_nat(count),source,0,Buffer.pack(~Element,left),Buffer.pack(~Element,piece)))}),
        Buffer.Content{BandCombineLeaf.readback_output(~Element,result),left_length},
        kernel_left_content(~Element,kernel(U32.to_nat(count),source,0,Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),
        Equal.trans(Buffer.Content<Element>,BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.observe_pair(~Element,kernel(U32.to_nat(count),source,0,Buffer.pack(~Element,left),Buffer.pack(~Element,piece)))}),
          BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,piece_length,result)}),
          Buffer.Content{BandCombineLeaf.readback_output(~Element,result),left_length},
          Equal.cong(OperationsLaws.PairOwners(Element),Buffer.Content<Element>,owners => BandCombineLeaf.left_content(~Element,Operations.Accepted{owners}),
            OperationsLaws.observe_pair(~Element,kernel(U32.to_nat(count),source,0,Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),OperationsLaws.combined_pair(~Element,left_length,piece_length,result),
            OperationsLaws.checked_kernel_finish(~Element,~operation,~kernel,~kernel_law,left,Canonical.canonical_view(Views.axis_extents(axes),count),piece,source,count,counted,accepted)),
          BandCombineLeaf.model_contents(~Element,left_length,piece_length,result)))

# What a band reads of its part at a cell: the part at the address of the
# kernel view is the source at the address of the stretched source.
def ReadAt(~Element: Data,+piece: Tree.Tree<Element>,+source: Tree.Tree<Element>,+kernel: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,+geometry: Bands.Geometry,
  +first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  {Tree.read(Element,piece,Copy.address(Views.View{kernel,0},BandValues.position(geometry,rows,channel,row,column))) ==
    Tree.read(Element,source,Copy.address(Views.View{full,0},BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))) : Element}

def Reads(~Element: Data,+piece: Tree.Tree<Element>,+source: Tree.Tree<Element>,+kernel: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,+geometry: Bands.Geometry,
  +first: U32,+rows: U32) -> Data:
  Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => ReadAt(~Element,piece,source,kernel,full,geometry,first,rows,channel,row,column) : Nat -> Data},
    Ranges.range(BandValues.width(geometry),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows),0n)) : Nat -> Data},Ranges.range(BandValues.channels(geometry),0n))

def read_at(~Element: Data,+piece: Tree.Tree<Element>,+source: Tree.Tree<Element>,+kernel: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,+geometry: Bands.Geometry,
  +first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,reads: Reads(~Element,piece,source,kernel,full,geometry,first,rows),
  channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) -> ReadAt(~Element,piece,source,kernel,full,geometry,first,rows,channel,row,column):
  Members.all_member(Nat,{column => ReadAt(~Element,piece,source,kernel,full,geometry,first,rows,channel,row,column) : Nat -> Data},column,Ranges.range(BandValues.width(geometry),0n),
    Members.all_member(Nat,{row => Members.All(Nat,{column => ReadAt(~Element,piece,source,kernel,full,geometry,first,rows,channel,row,column) : Nat -> Data},
        Ranges.range(BandValues.width(geometry),0n)) : Nat -> Data},row,Ranges.range(U32.to_nat(rows),0n),
      Members.all_member(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => ReadAt(~Element,piece,source,kernel,full,geometry,first,rows,channel,row,column) : Nat -> Data},
          Ranges.range(BandValues.width(geometry),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows),0n)) : Nat -> Data},channel,Ranges.range(BandValues.channels(geometry),0n),
        reads,Members.range_member(BandValues.channels(geometry),0n,channel,channel_inside)),
      Members.range_member(U32.to_nat(rows),0n,row,row_inside)),
    Members.range_member(BandValues.width(geometry),0n,column,column_inside))

# A combined cell holds the operation of the left value at its pixel and the
# source value at the address of the pixel in the stretched source.
def combined_cell(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+kernel: List<&2,Views.Axis>,+geometry: Bands.Geometry,+first: U32,+rows: U32,+count: U32,
  +limits: Machine.Limits,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+source: Buffer.Content<Element>,+source_count: U32,+channel: Nat,+row: Nat,+column: Nat,
  +left_ready: Buffer.Sized<Element,left>,+piece_ready: Buffer.Sized<Element,piece>,
  left_cell: BandValues.At(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left),channel,row,column),
  +read: ReadAt(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),kernel,full,geometry,first,rows,channel,row,column),
  +written: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},+exact: {ShapeLaws.elements(Views.axis_extents(kernel)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(Views.View{kernel,0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool},
  +source_values: Contract.Values(~Element,~Right,~right_value,right_context,source),+source_length: {Buffer.content_length(~Element,source) == U32.to_nat(source_count) : Nat},
  +source_fits: {Views.fits_nonempty(False{},Views.View{full,0},source_count) == True{} : Bool},
  +pixel_room: {Nat.is_le(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandValues.At(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,first,rows,
    Buffer.content_tree(~Element,CombineKernel.combined_content(~Element,~operation,count,Views.View{kernel,0},Buffer.content(~Element,left),Buffer.content(~Element,piece))),
    Buffer.content_length(~Element,CombineKernel.combined_content(~Element,~operation,count,Views.View{kernel,0},Buffer.content(~Element,left),Buffer.content(~Element,piece))),
    channel,row,column):
  Buffer.Model{+left_tree,+left_length,+left_certificate} = left
  Buffer.Model{+piece_tree,+piece_length,+piece_certificate} = piece
  Buffer.Content{+source_tree,+source_size} = source
  Buffer.Sized{+depth_ok,+room} = left_ready
  Shape.Both{+selected,+left_equal} = left_cell
  +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
  +piece = {Buffer.Model{piece_tree,piece_length,piece_certificate} : Buffer.Model<Element>}
  +source = {Buffer.Content{source_tree,source_size} : Buffer.Content<Element>}
  +view = {Views.View{kernel,0} : Views.View}
  +combined = CombineKernel.combined_content(~Element,~operation,count,view,Buffer.content(~Element,left),Buffer.content(~Element,piece))
  +position = BandValues.position(geometry,rows,channel,row,column)
  +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
  +index = U32.from_nat(position)
  +global = U32.from_nat(pixel)
  +capacity = Tree.capacity(Element,left_tree)
  +position_exact = Bounds.roundtrip(position,capacity,Buffer.capacity_bound(~Element,left,depth_ok),
    Bounds.strict_le(position,U32.to_nat(capacity),Bounds.nat_trans(position,left_length,U32.to_nat(capacity),selected,room)))
  +pixel_exact = Bounds.roundtrip(pixel,Machine.capacity(limits),Limits.capacity_fits(limits),pixel_room)
  +inside_count = {
    %Equal.sym(Nat,U32.to_nat(count),left_length,written) : {Nat.is_lt(position,_) == True{} : Bool}
    selected : {Nat.is_lt(position,U32.to_nat(count)) == True{} : Bool}}
  +source_address = Copy.address(Views.View{full,0},pixel)
  +source_inside = {
    %Equal.sym(Nat,source_size,U32.to_nat(source_count),source_length) : {Nat.is_lt(U32.to_nat(source_address),_) == True{} : Bool}
    Copy.address_inside(Views.View{full,0},pixel,source_count,source_fits) : {Nat.is_lt(U32.to_nat(source_address),source_size) == True{} : Bool}}
  +right_read = {
    %Arithmetic.native_roundtrip(source_address) : {Tree.read(Element,source_tree,_) == right_value(right_context,_) : Element}
    Contract.read(~Element,~Right,~right_value,right_context,source,U32.to_nat(source_address),source_values,source_inside)
    : {Tree.read(Element,source_tree,source_address) == right_value(right_context,source_address) : Element}}
  +kernel_cell = CombineKernel.cell(~Element,~Buffer.Content<Element>,~Buffer.Content<Element>,~Contract.stored(~Element),~Contract.stored(~Element),~operation,
    Buffer.content(~Element,left),Buffer.content(~Element,piece),view,left,piece,count,position,Buffer.Sized{depth_ok,room},piece_ready,
    Contract.stored_values(~Element,Buffer.content(~Element,left)),Contract.stored_values(~Element,Buffer.content(~Element,piece)),written,exact,fits,inside_count)
  Shape.Both{selected,
    %Equal.sym(Nat,U32.to_nat(global),pixel,pixel_exact) :
      {Tree.read(Element,Buffer.content_tree(~Element,combined),index) == operation(left_value(left_context,global),right_value(right_context,Copy.address(Views.View{full,0},_))) : Element}
    Equal.trans(Element,Tree.read(Element,Buffer.content_tree(~Element,combined),index),
      operation(Tree.read(Element,left_tree,index),Tree.read(Element,piece_tree,Copy.address(view,U32.to_nat(index)))),
      operation(left_value(left_context,global),right_value(right_context,source_address)),
      kernel_cell,
      %Equal.sym(Nat,U32.to_nat(index),position,position_exact) :
        {operation(Tree.read(Element,left_tree,index),Tree.read(Element,piece_tree,Copy.address(view,_))) == operation(left_value(left_context,global),right_value(right_context,source_address)) : Element}
      Equal.trans(Element,operation(Tree.read(Element,left_tree,index),Tree.read(Element,piece_tree,Copy.address(view,position))),
        operation(left_value(left_context,global),Tree.read(Element,piece_tree,Copy.address(view,position))),
        operation(left_value(left_context,global),right_value(right_context,source_address)),
        Equal.cong(Element,Element,element => operation(element,Tree.read(Element,piece_tree,Copy.address(view,position))),Tree.read(Element,left_tree,index),left_value(left_context,global),left_equal),
        Equal.cong(Element,Element,element => operation(left_value(left_context,global),element),Tree.read(Element,piece_tree,Copy.address(view,position)),
          right_value(right_context,source_address),
          Equal.trans(Element,Tree.read(Element,piece_tree,Copy.address(view,position)),Tree.read(Element,source_tree,source_address),right_value(right_context,source_address),read,right_read))))}

# What one band and its part supply to every cell.
type SpreadScope<-Element: Data,-Left: Data,-Right: Data,-left_value: Left -> U32 -> Element,-right_value: Right -> U32 -> Element> is Data:
  SpreadScope{left_context: Left,right_context: Right,full: List<&2,Views.Axis>,kernel: List<&2,Views.Axis>,geometry: Bands.Geometry,first: U32,rows: U32,count: U32,
    limits: Machine.Limits,left: Buffer.Model<Element>,piece: Buffer.Model<Element>,source: Buffer.Content<Element>,source_count: U32,
    left_ready: Buffer.Sized<Element,left>,piece_ready: Buffer.Sized<Element,piece>,
    left_values: BandValues.Leaf(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left)),
    written: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},exact: {ShapeLaws.elements(Views.axis_extents(kernel)) == U32.to_nat(count) : Nat},
    fits: {Views.fits(Views.View{kernel,0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool},
    source_values: Contract.Values(~Element,~Right,~right_value,right_context,source),source_length: {Buffer.content_length(~Element,source) == U32.to_nat(source_count) : Nat},
    source_fits: {Views.fits_nonempty(False{},Views.View{full,0},source_count) == True{} : Bool},
    reads: Reads(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),kernel,full,geometry,first,rows),
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool},
    height_positive: {BandGather.height_positive(geometry) == True{} : Bool},
    pixels: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def scope_channels(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+scope: SpreadScope<Element,Left,Right,left_value,right_value>) -> Nat:
  SpreadScope{left_context,right_context,full,kernel,geometry,first,rows,count,limits,left,piece,source,source_count,left_ready,piece_ready,left_values,written,exact,fits,
    source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels} = scope
  BandValues.channels(geometry)

def scope_rows(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+scope: SpreadScope<Element,Left,Right,left_value,right_value>) -> Nat:
  SpreadScope{left_context,right_context,full,kernel,geometry,first,rows,count,limits,left,piece,source,source_count,left_ready,piece_ready,left_values,written,exact,fits,
    source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels} = scope
  U32.to_nat(rows)

def scope_width(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+scope: SpreadScope<Element,Left,Right,left_value,right_value>) -> Nat:
  SpreadScope{left_context,right_context,full,kernel,geometry,first,rows,count,limits,left,piece,source,source_count,left_ready,piece_ready,left_values,written,exact,fits,
    source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels} = scope
  BandValues.width(geometry)

def Cell(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +scope: SpreadScope<Element,Left,Right,left_value,right_value>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  SpreadScope{left_context,right_context,full,kernel,geometry,first,rows,count,limits,left,piece,source,source_count,left_ready,piece_ready,left_values,written,exact,fits,
    source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels} = scope
  BandValues.At(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,first,rows,
    Buffer.content_tree(~Element,CombineKernel.combined_content(~Element,~operation,count,Views.View{kernel,0},Buffer.content(~Element,left),Buffer.content(~Element,piece))),
    Buffer.content_length(~Element,CombineKernel.combined_content(~Element,~operation,count,Views.View{kernel,0},Buffer.content(~Element,left),Buffer.content(~Element,piece))),
    channel,row,column)

def spread_cell(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +scope: SpreadScope<Element,Left,Right,left_value,right_value>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,scope_channels(~Element,~Left,~Right,~left_value,~right_value,scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,scope_rows(~Element,~Left,~Right,~left_value,~right_value,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,scope_width(~Element,~Left,~Right,~left_value,~right_value,scope)) == True{} : Bool}) ->
  Cell(~Element,~Left,~Right,~left_value,~right_value,~operation,scope,channel,row,column):
  SpreadScope{+left_context,+right_context,+full,+kernel,+geometry,+first,+rows,+count,+limits,+left,+piece,+source,+source_count,+left_ready,+piece_ready,+left_values,+written,+exact,+fits,
    +source_values,+source_length,+source_fits,+reads,+rows_fit,+height_positive,+pixels} = scope
  +global_row = Bounds.nat_trans(Nat.add(U32.to_nat(first),row),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),
    Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),row_inside),rows_fit)
  +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
  +pixel_room = Bounds.strict_le(pixel,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(pixel,BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits)),
      BandGather.pixel_below(geometry,channel,Nat.add(U32.to_nat(first),row),column,channel_inside,global_row,column_inside,height_positive),pixels))
  combined_cell(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,kernel,geometry,first,rows,count,limits,left,piece,source,source_count,
    channel,row,column,left_ready,piece_ready,
    BandValues.leaf_cell(~Element,~Left,~left_value,left_context,geometry,first,rows,Buffer.tree(~Element,left),Buffer.length(~Element,left),channel,row,column,left_values,
      channel_inside,row_inside,column_inside),
    read_at(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),kernel,full,geometry,first,rows,channel,row,column,reads,channel_inside,row_inside,column_inside),
    written,exact,fits,source_values,source_length,source_fits,pixel_room)

def spread_values(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+kernel: List<&2,Views.Axis>,+geometry: Bands.Geometry,+first: U32,+rows: U32,+count: U32,
  +limits: Machine.Limits,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+source: Buffer.Content<Element>,+source_count: U32,
  left_ready: Buffer.Sized<Element,left>,piece_ready: Buffer.Sized<Element,piece>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.RowBand{Buffer.content(~Element,left),first,rows}),
  written: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},exact: {ShapeLaws.elements(Views.axis_extents(kernel)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(Views.View{kernel,0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool},
  source_values: Contract.Values(~Element,~Right,~right_value,right_context,source),source_length: {Buffer.content_length(~Element,source) == U32.to_nat(source_count) : Nat},
  source_fits: {Views.fits_nonempty(False{},Views.View{full,0},source_count) == True{} : Bool},
  reads: Reads(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),kernel,full,geometry,first,rows),
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool},
  height_positive: {BandGather.height_positive(geometry) == True{} : Bool},
  pixels: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandValues.Values(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,
    Model.RowBand{CombineKernel.combined_content(~Element,~operation,count,Views.View{kernel,0},Buffer.content(~Element,left),Buffer.content(~Element,piece)),first,rows}):
  match left piece:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+piece_tree,+piece_length,+piece_certificate}:
      BandValues.table(~SpreadScope<Element,Left,Right,left_value,right_value>,~Cell(~Element,~Left,~Right,~left_value,~right_value,~operation),
        ~scope_channels(~Element,~Left,~Right,~left_value,~right_value),~scope_rows(~Element,~Left,~Right,~left_value,~right_value),~scope_width(~Element,~Left,~Right,~left_value,~right_value),
        ~spread_cell(~Element,~Left,~Right,~left_value,~right_value,~operation),
        SpreadScope{left_context,right_context,full,kernel,geometry,first,rows,count,limits,Buffer.Model{left_tree,left_length,left_certificate},
          Buffer.Model{piece_tree,piece_length,piece_certificate},source,source_count,left_ready,piece_ready,left_values,written,exact,fits,
          source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels})

# A band and its part combine into a band that holds, at every cell, the
# operation of the left value and the source value at the address of the
# pixel in the stretched source.
def band_contract(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+axes: List<&2,Views.Axis>,+geometry: Bands.Geometry,+first: U32,+rows: U32,+count: U32,
  +limits: Machine.Limits,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+source: Buffer.Content<Element>,+source_count: U32,
  +left_ready: Buffer.Sized<Element,left>,+piece_ready: Buffer.Sized<Element,piece>,+left_metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.RowBand{Buffer.content(~Element,left),first,rows}),
  +written: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},+exact: {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(Views.View{axes,0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool},
  source_values: Contract.Values(~Element,~Right,~right_value,right_context,source),source_length: {Buffer.content_length(~Element,source) == U32.to_nat(source_count) : Nat},
  source_fits: {Views.fits_nonempty(False{},Views.View{full,0},source_count) == True{} : Bool},
  reads: Reads(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),axes,full,geometry,first,rows),
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool},
  height_positive: {BandGather.height_positive(geometry) == True{} : Bool},
  pixels: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandCombine.LeafContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,first,rows,
    Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,count,Some{Views.View{axes,0}},Buffer.pack(~Element,left),Buffer.pack(~Element,piece)))):
  +combined = CombineKernel.combined_content(~Element,~operation,count,Views.View{axes,0},Buffer.content(~Element,left),Buffer.content(~Element,piece))
  +counted = {
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(Views.axis_extents(axes),count)),Views.axis_extents(axes),
      Canonical.canonical_shape(Views.axis_extents(axes),count,exact)) : {Shapes.numel(Shapes.Shape{_}) == Some{count} : Maybe<&2,U32>}
    CountCompletion.numel(Views.axis_extents(axes),count,exact) : {Views.numel(Canonical.canonical_view(Views.axis_extents(axes),count)) == Some{count} : Maybe<&2,U32>}}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,count,Some{Views.View{axes,0}},Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),
    combined,kernel_content(~Element,~operation,~kernel,~kernel_law,left,piece,axes,count,counted,
      spread_guard(~Element,left,piece,axes,count,left_ready,piece_ready,exact,written,fits))) :
    BandCombine.LeafContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,first,rows,_)
  Shape.Both{spread_values(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,axes,geometry,first,rows,count,limits,left,piece,source,source_count,
      left_ready,piece_ready,left_values,written,exact,fits,source_values,source_length,source_fits,reads,rows_fit,height_positive,pixels),
    Shape.Both{CombineKernel.program_ready(~Element,~operation,count,Views.View{axes,0},left,piece,left_ready),
      %Equal.sym(Nat,Buffer.content_length(~Element,combined),Buffer.length(~Element,left),CombineKernel.program_length(~Element,~operation,count,Views.View{axes,0},left,piece)) :
        BandShape.Leaf(geometry,first,rows,_)
      left_metadata}}

# Coordinates below extents leave no extent of zero.
def below_nonempty(+extents: List<&2,U32>,+coordinates: List<&2,Nat>,below: Below(extents,coordinates)) -> {Shapes.has_zero(extents) == False{} : Bool}:
  match extents coordinates:
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Con{+coordinate,+more}:
      Shape.Both{+here,more_below} = below
      %Equal.sym(Bool,U32.is_zero(extent),False{},
        U32Division.nonzero(extent,Order.le_lt_trans(0n,coordinate,U32.to_nat(extent),Order.zero_le(coordinate),here))) :
        {_ || Shapes.has_zero(rest) == False{} : Bool}
      below_nonempty(rest,more,more_below)
    case Nil{} Con{coordinate,more}: Empty.absurd({Shapes.has_zero(Nil{}) == False{} : Bool},below)
    case Con{extent,rest} Nil{}: Empty.absurd({Shapes.has_zero(Con{extent,rest}) == False{} : Bool},below)

# How a band reads its part: the source itself, a copy of all of it, or a
# copy of the compacted view of its image rows.
type How<-Element: Data,-piece: Tree.Tree<Element>,-source: Tree.Tree<Element>,-kernel: List<&2,Views.Axis>,-full: List<&2,Views.Axis>,
  -images: U32,-channels: U32,-height: U32,-width: U32,-first: U32,-rows: U32> is Data:
  Same{agree: Agree(kernel,full),whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
    made: {piece == source : Tree.Tree<Element>}}
  Whole{copied: List<&2,Views.Axis>,compact: List<&2,U32>,count: U32,output: Tree.Tree<Element>,length: Nat,
    agree: Agree(kernel,full),whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
    copy_agree: Agree(copied,Canonical.axes(Canonical.canonical_view(compact,count))),exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},
    copied_extents: {Views.axis_extents(copied) == compact : List<&2,U32>},
    ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
    fits: {Views.fits_nonempty(False{},Views.View{kernel,0},count) == True{} : Bool},
    made: {piece == piece_tree(~Element,Views.View{copied,0},count,source,output) : Tree.Tree<Element>}}
  Rows{shape: List<&2,U32>,band: List<&2,Views.Axis>,offset: U32,compacted: List<&2,U32>,compact: List<&2,U32>,count: U32,output: Tree.Tree<Element>,length: Nat,
    limits: Machine.Limits,slices: Slices(shape,height,first,rows,full,band,offset),spread: Lanes.Spreadable(compacted,band),
    lanes: Lanes.Stretched(Canonical.axes(Canonical.canonical_view(compact,count)),band_cells(shape,rows),kernel),
    same: {Views.axis_extents(Canonical.axes(Canonical.canonical_view(compact,count))) == Views.axis_extents(Bands.compact_axes(compacted,band)) : List<&2,U32>},
    full_extents: {Views.axis_extents(full) == shape : List<&2,U32>},exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},
    ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
    image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
    height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool},
    made: {piece == piece_tree(~Element,Views.View{Bands.compact_axes(compacted,band),offset},count,source,output) : Tree.Tree<Element>}}

def read_cell(~Element: Data,+piece: Tree.Tree<Element>,+source: Tree.Tree<Element>,+kernel: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,
  +images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat,
  how: How<Element,piece,source,kernel,full,images,channels,height,width,first,rows>,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  ReadAt(~Element,piece,source,kernel,full,Bands.Geometry{images,channels,height,width},first,rows,channel,row,column):
  match how:
    case Same{agree,whole,made}:
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +position = BandValues.position(geometry,rows,channel,row,column)
      +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
      +address = Copy.address(Views.View{kernel,0},position)
      %Equal.sym(Tree.Tree<Element>,piece,source,made) : {Tree.read(Element,_,address) == Tree.read(Element,source,Copy.address(Views.View{full,0},pixel)) : Element}
      Equal.cong(U32,Element,word => Tree.read(Element,source,word),address,Copy.address(Views.View{full,0},pixel),
        whole_address(kernel,full,images,channels,height,width,first,rows,channel,row,column,agree,whole,row_inside))
    case Whole{+copied,+compact,+count,+output,+length,agree,whole,copy_agree,+exact,+copied_extents,ready,capacity_ok,room,fits,made}:
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +position = BandValues.position(geometry,rows,channel,row,column)
      +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
      +address = Copy.address(Views.View{kernel,0},position)
      %Equal.sym(Tree.Tree<Element>,piece,piece_tree(~Element,Views.View{copied,0},count,source,output),made) :
        {Tree.read(Element,_,address) == Tree.read(Element,source,Copy.address(Views.View{full,0},pixel)) : Element}
      Equal.trans(Element,Tree.read(Element,piece_tree(~Element,Views.View{copied,0},count,source,output),address),Tree.read(Element,source,address),
        Tree.read(Element,source,Copy.address(Views.View{full,0},pixel)),
        whole_copy_read(~Element,source,output,length,copied,compact,count,address,copy_agree,exact,copied_extents,ready,capacity_ok,room,
          Copy.address_inside(Views.View{kernel,0},position,count,fits)),
        Equal.cong(U32,Element,word => Tree.read(Element,source,word),address,Copy.address(Views.View{full,0},pixel),
          whole_address(kernel,full,images,channels,height,width,first,rows,channel,row,column,agree,whole,row_inside)))
    case Rows{+shape,+band,+offset,+compacted,+compact,+count,+output,+length,limits,slices,spread,+lanes,same,full_extents,+exact,ready,capacity_ok,room,
      +image,positive,height_bound,local,rows_fit,made}:
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +position = BandValues.position(geometry,rows,channel,row,column)
      +pixel = BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column)
      +address = Copy.address(Views.View{kernel,0},position)
      +axes = Canonical.axes(Canonical.canonical_view(compact,count))
      +shaped = Canonical.canonical_shape(compact,count,exact)
      +cells = band_cells(shape,rows)
      +coordinates = cell_of(shape,channel,row,column)
      +below = cell_below(shape,images,channels,height,width,rows,channel,row,column,image,channel_inside,row_inside,column_inside)
      +axes_exact = {
        %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(compact,count)),compact,shaped) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
        exact : {ShapeLaws.elements(Views.axis_extents(axes)) == U32.to_nat(count) : Nat}}
      +layout = canonical_layout(compact,count,exact,
        %shaped : {Shapes.has_zero(_) == False{} : Bool} stretched_nonempty(axes,cells,kernel,lanes,below_nonempty(cells,coordinates,below)))
      %Equal.sym(Tree.Tree<Element>,piece,piece_tree(~Element,Views.View{Bands.compact_axes(compacted,band),offset},count,source,output),made) :
        {Tree.read(Element,_,address) == Tree.read(Element,source,Copy.address(Views.View{full,0},pixel)) : Element}
      %cell_position(shape,images,channels,height,width,rows,channel,row,column,image) :
        {Tree.read(Element,piece_tree(~Element,Views.View{Bands.compact_axes(compacted,band),offset},count,source,output),Copy.address(Views.View{kernel,0},_)) ==
          Tree.read(Element,source,Copy.address(Views.View{full,0},pixel)) : Element}
      %cell_pixel(shape,images,channels,height,width,first,rows,channel,row,column,limits,image,positive,height_bound,local,rows_fit,row_inside) :
        {Tree.read(Element,piece_tree(~Element,Views.View{Bands.compact_axes(compacted,band),offset},count,source,output),Copy.address(Views.View{kernel,0},Lanes.position(cells,coordinates,0n))) ==
          Tree.read(Element,source,Copy.address(Views.View{full,0},_)) : Element}
      rows_read(~Element,shape,height,first,rows,full,band,offset,compacted,axes,kernel,count,coordinates,source,output,length,slices,spread,lanes,layout,same,full_extents,axes_exact,
        ready,capacity_ok,room,below)

# Every cell of a band reads its part as the stretched source.
type ReadScope<-Element: Data> is Data:
  ReadScope{piece: Tree.Tree<Element>,source: Tree.Tree<Element>,kernel: List<&2,Views.Axis>,full: List<&2,Views.Axis>,images: U32,channels: U32,height: U32,width: U32,
    first: U32,rows: U32,how: How<Element,piece,source,kernel,full,images,channels,height,width,first,rows>}

def read_channels(~Element: Data,+scope: ReadScope<Element>) -> Nat:
  ReadScope{piece,source,kernel,full,images,channels,height,width,first,rows,how} = scope
  U32.to_nat(channels)

def read_rows(~Element: Data,+scope: ReadScope<Element>) -> Nat:
  ReadScope{piece,source,kernel,full,images,channels,height,width,first,rows,how} = scope
  U32.to_nat(rows)

def read_width(~Element: Data,+scope: ReadScope<Element>) -> Nat:
  ReadScope{piece,source,kernel,full,images,channels,height,width,first,rows,how} = scope
  U32.to_nat(width)

def ReadCell(~Element: Data,+scope: ReadScope<Element>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  ReadScope{piece,source,kernel,full,images,channels,height,width,first,rows,how} = scope
  ReadAt(~Element,piece,source,kernel,full,Bands.Geometry{images,channels,height,width},first,rows,channel,row,column)

def read_rule(~Element: Data,+scope: ReadScope<Element>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,read_channels(~Element,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,read_rows(~Element,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,read_width(~Element,scope)) == True{} : Bool}) -> ReadCell(~Element,scope,channel,row,column):
  ReadScope{+piece,+source,+kernel,+full,+images,+channels,+height,+width,+first,+rows,how} = scope
  read_cell(~Element,piece,source,kernel,full,images,channels,height,width,first,rows,channel,row,column,how,channel_inside,row_inside,column_inside)

def reads(~Element: Data,+piece: Tree.Tree<Element>,+source: Tree.Tree<Element>,+kernel: List<&2,Views.Axis>,+full: List<&2,Views.Axis>,
  +images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+how: How<Element,piece,source,kernel,full,images,channels,height,width,first,rows>) ->
  Reads(~Element,piece,source,kernel,full,Bands.Geometry{images,channels,height,width},first,rows):
  BandValues.table(~ReadScope<Element>,~ReadCell(~Element),~read_channels(~Element),~read_rows(~Element),~read_width(~Element),~read_rule(~Element),
    ReadScope{piece,source,kernel,full,images,channels,height,width,first,rows,how})

# The part a band copies, as contents: the source through the view of the
# band into a fresh buffer of the count of its compacted extents. A band
# without a view makes an empty buffer.
def part_views(~Element: Data,source_view: Maybe<&2,Views.View>,piece_view: Maybe<&2,Views.View>,count: Maybe<&2,U32>,+fill: Element,+source: Buffer.Content<Element>) ->
  Buffer.Content<Element>:
  match source_view piece_view count:
    case Some{source_view} Some{piece_view} Some{+count}:
      Copy.program_content(~Element,count,source_view,piece_view,source,Buffer.content(~Element,Buffer.fresh(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))
    case source_view piece_view count: Buffer.Content{Tree.fresh(Element,Storage.depth_for(0),fill),U32.to_nat(0)}

def part(~Element: Data,context: Bands.Pieces<Element>,source: Buffer.Content<Element>,+first: U32,+rows: U32) -> Buffer.Content<Element>:
  Bands.Pieces{+spread,+fill} = context
  part_views(~Element,Bands.compact_of(Bands.source_of(spread),Bands.band_view(spread,first,rows)),Bands.contiguous_of(Bands.band_compact(spread,first,rows)),
    Bands.count_of_extents(Bands.band_compact(spread,first,rows)),fill,source)

# The owners a copy into a part leaves: the source and the part.
def copied_seen(~Element: Data,+first: U32,+rows: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,result)) ==
    (BandCombineLeaf.left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)),
      Model.RowBand{Copy.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}:
  match result:
    case Operations.Accepted{(Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{piece,piece_length,piece_certificate})}: {==}
    case Operations.Rejected{(Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{piece,piece_length,piece_certificate})}: {==}

def modeled_left(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}) ==
    Buffer.Content{Copy.reader_tree(~Element,result),input_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{+reader,output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,input_length},Tree.reflect(Element,Tree.pack(Element,reader)),reader,
    StorageLaws.reflect_packed(Element,reader))

# A copy keeps its source and writes the copied elements into its output.
def copy_kept(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {BandCombineLeaf.left_content(~Element,OperationsLaws.copy_when_denotation(~Element,True{},Some{count},input,source,output,target)) ==
    Buffer.contents(~Element,input) : Buffer.Content<Element>}:
  Views.View{+axes,+destination} = target
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  +input_tree = Tree.reflect(Element,input)
  +output_tree = Tree.reflect(Element,output)
  +result = AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input_tree,output_tree)
  Equal.trans(Buffer.Content<Element>,BandCombineLeaf.left_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}),
    Buffer.Content{Copy.reader_tree(~Element,result),input_length},Buffer.Content{input_tree,input_length},
    modeled_left(~Element,input_length,output_length,result),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,input_length},Copy.reader_tree(~Element,result),input_tree,
      Copy.copy_reader(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input_tree,output_tree)))

def copy_made(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {Copy.right_content(~Element,OperationsLaws.copy_when_denotation(~Element,True{},Some{count},input,source,output,target)) ==
    Copy.program_content(~Element,count,source,target,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  Copy.accepted_content(~Element,count,source,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

def empty_seen(~Element: Data,+fill: Element,+first: U32,+rows: U32,+source: Buffer.Model<Element>) ->
  {(Buffer.contents(~Element,Buffer.pack(~Element,source)),Model.RowBand{Buffer.contents(~Element,Bands.allocate(~Element,fill,0)),first,rows}) ==
    (Buffer.content(~Element,source),Model.RowBand{Buffer.Content{Tree.fresh(Element,Storage.depth_for(0),fill),U32.to_nat(0)},first,rows}) : Buffer.Content<Element> & Model.Content<Element>}:
  %Buffer.packed_contents(~Element,source) :
    {(Buffer.contents(~Element,Buffer.pack(~Element,source)),Model.RowBand{Buffer.contents(~Element,Bands.allocate(~Element,fill,0)),first,rows}) ==
      (_,Model.RowBand{Buffer.Content{Tree.fresh(Element,Storage.depth_for(0),fill),U32.to_nat(0)},first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  %AllocationLaws.contents(~Element,Storage.depth_for(0),fill,U32.to_nat(0)) :
    {(Buffer.contents(~Element,Buffer.pack(~Element,source)),Model.RowBand{Buffer.contents(~Element,Bands.allocate(~Element,fill,0)),first,rows}) ==
      (Buffer.contents(~Element,Buffer.pack(~Element,source)),Model.RowBand{_,first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  {==}

def copied_views(~Element: Data,+source_view: Views.View,+piece_view: Views.View,+count: U32,+fill: Element,+first: U32,+rows: U32,+source: Buffer.Model<Element>) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Bands.piece_views(~Element,Some{source_view},Some{piece_view},Some{count},fill,first,rows,Buffer.pack(~Element,source))) ==
    (Buffer.content(~Element,source),Model.RowBand{part_views(~Element,Some{source_view},Some{piece_view},Some{count},fill,Buffer.content(~Element,source)),first,rows}) :
    Buffer.Content<Element> & Model.Content<Element>}:
  +fresh = {Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  %Buffer.packed_contents(~Element,source) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))) ==
      (_,Model.RowBand{Copy.program_content(~Element,count,source_view,piece_view,_,fresh),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  %AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count)) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))) ==
      (Buffer.contents(~Element,Buffer.pack(~Element,source)),Model.RowBand{Copy.program_content(~Element,count,source_view,piece_view,Buffer.contents(~Element,Buffer.pack(~Element,source)),_),first,rows}) :
      Buffer.Content<Element> & Model.Content<Element>}
  %copy_kept(~Element,count,source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))) ==
      (_,Model.RowBand{Copy.program_content(~Element,count,source_view,piece_view,Buffer.contents(~Element,Buffer.pack(~Element,source)),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),first,rows}) :
      Buffer.Content<Element> & Model.Content<Element>}
  %copy_made(~Element,count,source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))) ==
      (BandCombineLeaf.left_content(~Element,OperationsLaws.copy_when_denotation(~Element,True{},Some{count},Buffer.pack(~Element,source),source_view,Bands.allocate(~Element,fill,count),piece_view)),Model.RowBand{_,first,rows}) :
      Buffer.Content<Element> & Model.Content<Element>}
  %OperationsLaws.copy_when(~Element,True{},Some{count},Buffer.pack(~Element,source),source_view,Bands.allocate(~Element,fill,count),piece_view) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))) ==
      (BandCombineLeaf.left_content(~Element,_),Model.RowBand{Copy.right_content(~Element,_),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  copied_seen(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,Buffer.pack(~Element,source),Bands.allocate(~Element,fill,count)))

def views_seen(~Element: Data,source_view: Maybe<&2,Views.View>,piece_view: Maybe<&2,Views.View>,count: Maybe<&2,U32>,+fill: Element,+first: U32,+rows: U32,+source: Buffer.Model<Element>) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
      Bands.piece_views(~Element,source_view,piece_view,count,fill,first,rows,Buffer.pack(~Element,source))) ==
    (Buffer.content(~Element,source),Model.RowBand{part_views(~Element,source_view,piece_view,count,fill,Buffer.content(~Element,source)),first,rows}) :
    Buffer.Content<Element> & Model.Content<Element>}:
  match source_view piece_view count:
    case Some{+source_view} Some{+piece_view} Some{+count}: copied_views(~Element,source_view,piece_view,count,fill,first,rows,source)
    case Some{source_view} Some{piece_view} None{}: empty_seen(~Element,fill,first,rows,source)
    case Some{source_view} None{} Some{count}: empty_seen(~Element,fill,first,rows,source)
    case Some{source_view} None{} None{}: empty_seen(~Element,fill,first,rows,source)
    case None{} Some{piece_view} Some{count}: empty_seen(~Element,fill,first,rows,source)
    case None{} Some{piece_view} None{}: empty_seen(~Element,fill,first,rows,source)
    case None{} None{} Some{count}: empty_seen(~Element,fill,first,rows,source)
    case None{} None{} None{}: empty_seen(~Element,fill,first,rows,source)

# Every band copies its part and keeps the source.
def part_rule(~Element: Data,+context: Bands.Pieces<Element>,+first: U32,+rows: U32,+kept: Buffer.Model<Element>) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_band(~Element,context,first,rows,Buffer.pack(~Element,kept))) ==
    (Buffer.contents(~Element,Buffer.pack(~Element,kept)),Model.RowBand{part(~Element,context,Buffer.contents(~Element,Buffer.pack(~Element,kept)),first,rows),first,rows}) :
    Buffer.Content<Element> & Model.Content<Element>}:
  Bands.Pieces{+spread,+fill} = context
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,kept)),Buffer.content(~Element,kept),Buffer.packed_contents(~Element,kept)) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.piece_band(~Element,Bands.Pieces{spread,fill},first,rows,Buffer.pack(~Element,kept))) ==
      (_,Model.RowBand{part(~Element,Bands.Pieces{spread,fill},_,first,rows),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  views_seen(~Element,Bands.compact_of(Bands.source_of(spread),Bands.band_view(spread,first,rows)),Bands.contiguous_of(Bands.band_compact(spread,first,rows)),
    Bands.count_of_extents(Bands.band_compact(spread,first,rows)),fill,first,rows,kept)

def parts_of(~Element: Data,made: Buffer.Content<Element> & Model.Content<Element>) -> Model.Content<Element>:
  (source,parts) = made
  parts

def result_parts(~Element: Data,made: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {Model.contents(~Element,Bands.result(~Element,made)) == parts_of(~Element,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),made)) :
    Model.Content<Element>}:
  (source,bands) = made
  {==}

# The parts of the bands of a plan.
def parts(~Element: Data,+context: Bands.Spread,+fill: Element,+source: Buffer.Content<Element>,layout: Bands.Plan) -> Model.Content<Element>:
  Model.build(~Element,~part(~Element,Bands.Pieces{context,fill},source),layout)

def built_parts(~Element: Data,+context: Bands.Spread,+fill: Element,+source: Buffer.Model<Element>,+layout: Bands.Plan) ->
  {Model.contents(~Element,Bands.result(~Element,Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,
      Bands.Pieces{context,fill},Buffer.pack(~Element,source)))) == parts(~Element,context,fill,Buffer.content(~Element,source),layout) : Model.Content<Element>}:
  %Buffer.packed_contents(~Element,source) : {Model.contents(~Element,Bands.result(~Element,Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,Bands.Pieces{context,fill},Buffer.pack(~Element,source)))) == parts(~Element,context,fill,_,layout) : Model.Content<Element>}
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,Bands.Pieces{context,fill},Buffer.pack(~Element,source)))),
    parts_of(~Element,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,Bands.Pieces{context,fill},Buffer.pack(~Element,source)))),
    parts(~Element,context,fill,Buffer.contents(~Element,Buffer.pack(~Element,source)),layout),
    result_parts(~Element,Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,Bands.Pieces{context,fill},Buffer.pack(~Element,source))),
    Equal.cong(Buffer.Content<Element> & Model.Content<Element>,Model.Content<Element>,seen => parts_of(~Element,seen),
      Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Bands.piece_band(~Element),layout,Bands.Pieces{context,fill},Buffer.pack(~Element,source))),
      (Buffer.contents(~Element,Buffer.pack(~Element,source)),parts(~Element,context,fill,Buffer.contents(~Element,Buffer.pack(~Element,source)),layout)),
      Each.built(~Element,~Storage.Buffer<Element>,~Bands.Pieces<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Each.buffer_held(~Element),~Buffer.Content<Element>,~Buffer.contents(~Element),
        ~Bands.piece_band(~Element),~part(~Element),~part_rule(~Element),layout,Bands.Pieces{context,fill},source)))

# Extents at most those of a shape, axis by axis.
def Within(extents: List<&2,U32>,shape: List<&2,U32>) -> Data:
  match extents shape:
    case Nil{} Nil{}: Unit
    case Con{extent,rest} Con{size,sizes}: Shape.Both<{Nat.is_le(U32.to_nat(extent),U32.to_nat(size)) == True{} : Bool},Within(rest,sizes)>
    case extents shape: Empty

def within_self(+shape: List<&2,U32>) -> Within(shape,shape):
  match shape:
    case Nil{}: Unit{}
    case Con{+extent,+rest}: Shape.Both{Order.reflexive(U32.to_nat(extent)),within_self(rest)}

# A lane compacted against an extent within its target keeps at most the
# size it came from.
def lane_within(single: Bool,+size: U32,+stride: U32,+extent: U32,+step: U32,+wanted: U32,classified: {U32.is_eq(size,1) == single : Bool},
  lane: Lanes.Lane<size,stride,extent,step>,below: {Nat.is_le(U32.to_nat(wanted),U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,single,1,wanted)),U32.to_nat(size)) == True{} : Bool}:
  match single lane:
    case True{} lane:
      %Equal.sym(U32,size,1,Arithmetic.equal_of_is_eq(size,1,classified)) : {Nat.is_le(U32.to_nat(1),U32.to_nat(_)) == True{} : Bool}
      Order.reflexive(U32.to_nat(1))
    case False{} Lanes.Kept{same,kept}:
      %Equal.sym(U32,size,extent,same) : {Nat.is_le(U32.to_nat(wanted),U32.to_nat(_)) == True{} : Bool}
      below
    case False{} Lanes.Repeated{unit,zero}:
      Empty.absurd({Nat.is_le(U32.to_nat(wanted),U32.to_nat(size)) == True{} : Bool},
        Words.false_true(Equal.sym(Bool,True{},False{},%unit : {U32.is_eq(_,1) == False{} : Bool} classified)))

# A target within the extents the source was stretched to compacts to at
# most the elements of the source.
def compact_within(+axes: List<&2,Views.Axis>,+shape: List<&2,U32>,+out: List<&2,Views.Axis>,+target: List<&2,U32>,lanes: Lanes.Stretched(axes,shape,out),
  within: Within(target,shape)) ->
  {Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(axes),target)),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}:
  match axes shape out:
    case Nil{} Nil{} Nil{}: Order.reflexive(1n)
    case Con{Views.Axis{+size,+stride},+tail} Con{+extent,+rest} Con{Views.Axis{+result,+step},+others}:
      match target:
        case Con{+wanted,+more}:
          Shape.Both{Shape.Both{resulting,lane},tail_lanes} = lanes
          Shape.Both{below,more_within} = within
          +picked = U32.to_nat(Bool.pick(U32,U32.is_eq(size,1),1,wanted))
          +inner = ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(tail),more))
          +sizes = ShapeLaws.elements(Views.axis_extents(tail))
          Order.transitive(Nat.mul(picked,inner),Nat.mul(U32.to_nat(size),inner),Nat.mul(U32.to_nat(size),sizes),
            RowMajor.mul_le(picked,U32.to_nat(size),inner,lane_within(U32.is_eq(size,1),size,stride,extent,step,wanted,{==},lane,below)),
            DivisionOrder.multiply_left_le(U32.to_nat(size),inner,sizes,compact_within(tail,rest,others,more,tail_lanes,more_within)))
        case Nil{}:
          Empty.absurd({Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(Con{Views.Axis{size,stride},tail}),Nil{})),
            ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{size,stride},tail}))) == True{} : Bool},within)
    case Nil{} Nil{} Con{axis,others}:
      Empty.absurd({Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(Nil{}),target)),ShapeLaws.elements(Views.axis_extents(Nil{}))) == True{} : Bool},lanes)
    case Nil{} Con{extent,rest} out:
      Empty.absurd({Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(Nil{}),target)),ShapeLaws.elements(Views.axis_extents(Nil{}))) == True{} : Bool},lanes)
    case Con{Views.Axis{size,stride},tail} Nil{} out:
      Empty.absurd({Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(Con{Views.Axis{size,stride},tail}),target)),
        ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{size,stride},tail}))) == True{} : Bool},lanes)
    case Con{Views.Axis{size,stride},tail} Con{extent,rest} Nil{}:
      Empty.absurd({Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Views.axis_extents(Con{Views.Axis{size,stride},tail}),target)),
        ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{size,stride},tail}))) == True{} : Bool},lanes)

# The cells of a band of image rows lie within the image shape.
def cells_within(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  span: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows)),U32.to_nat(height)) == True{} : Bool}) ->
  Within(band_cells(shape,rows),shape):
  match shape:
    case [+c,+h,+w]:
      Shape.Both{one,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      Shape.Both{Order.reflexive(U32.to_nat(c)),
        Shape.Both{%Equal.sym(U32,h,height,same_h) : {Nat.is_le(U32.to_nat(rows),U32.to_nat(_)) == True{} : Bool}
            Order.drop_left_le(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows),U32.to_nat(height),span),
          Shape.Both{Order.reflexive(U32.to_nat(w)),Unit{}}}}
    case [+n,+c,+h,+w]:
      Shape.Both{same_n,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      Shape.Both{Order.positive_one(U32.to_nat(n),Arithmetic.word_nonzero_positive(n,Boolean.or_false_left(U32.is_zero(n),Shapes.has_zero([c,h,w]),nonempty))),
        Shape.Both{Order.reflexive(U32.to_nat(c)),
          Shape.Both{%Equal.sym(U32,h,height,same_h) : {Nat.is_le(U32.to_nat(rows),U32.to_nat(_)) == True{} : Bool}
              Order.drop_left_le(Nat.mod(U32.to_nat(first),U32.to_nat(height)),U32.to_nat(rows),U32.to_nat(height),span),
            Shape.Both{Order.reflexive(U32.to_nat(w)),Unit{}}}}}
    case []: Empty.absurd(Within(band_cells([],rows),[]),image)
    case [a]: Empty.absurd(Within(band_cells([a],rows),[a]),image)
    case [a,b]: Empty.absurd(Within(band_cells([a,b],rows),[a,b]),image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd(Within(band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows),Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}),image)

# The count of the part a band copies: its elements, at most those of the
# source and so a word.
def part_count(compact: List<&2,U32>) -> U32:
  U32.from_nat(ShapeLaws.elements(compact))

def part_exact(+compact: List<&2,U32>,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  bound: {Nat.is_le(ShapeLaws.elements(compact),U32.to_nat(capacity)) == True{} : Bool}) -> {ShapeLaws.elements(compact) == U32.to_nat(part_count(compact)) : Nat}:
  Equal.sym(Nat,U32.to_nat(U32.from_nat(ShapeLaws.elements(compact))),ShapeLaws.elements(compact),
    Bounds.roundtrip(ShapeLaws.elements(compact),capacity,capacity_ok,bound))

# The fresh buffer a part is copied into holds its count.
def fresh_of(~Element: Data,+fill: Element,+count: U32,sized: Depth.Sized<count>) -> Buffer.Sized<Element,Buffer.fresh(~Element,Storage.depth_for(count),fill,U32.to_nat(count))>:
  Depth.Sized{depth_ok,room} = sized
  Buffer.fresh_ready(~Element,Storage.depth_for(count),fill,U32.to_nat(count),depth_ok,room)

# Extents that fit a target have its length.
def fit_length(+sizes: List<&2,U32>,+target: List<&2,U32>,fit: Fit(sizes,target)) -> {List.length(&2,U32,sizes) == List.length(&2,U32,target) : Nat}:
  match sizes target:
    case Nil{} Nil{}: {==}
    case Con{+size,+rest} Con{+extent,+others}:
      Shape.Both{allowed,more} = fit
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,rest),List.length(&2,U32,others),fit_length(rest,others,more))
    case Nil{} Con{extent,others}: Empty.absurd({List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{extent,others}) : Nat},fit)
    case Con{size,rest} Nil{}: Empty.absurd({List.length(&2,U32,Con{size,rest}) == List.length(&2,U32,Nil{}) : Nat},fit)

# The contiguous view of extents that fit a target fits it, and stretches to
# it again: the kernel reads the part through that view.
def canonical_fit(+compact: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},fit: Fit(compact,target)) ->
  Fit(Views.axis_extents(Canonical.axes(Canonical.canonical_view(compact,count))),target):
  %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(compact,count)),compact,Canonical.canonical_shape(compact,count,exact)) : Fit(_,target)
  fit

def kernel_lanes(+compact: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},fit: Fit(compact,target)) ->
  Lanes.Stretched(Canonical.axes(Canonical.canonical_view(compact,count)),target,restretched(Canonical.axes(Canonical.canonical_view(compact,count)),target)):
  +axes = Canonical.axes(Canonical.canonical_view(compact,count))
  Lanes.stretched(axes,target,restretched(axes,target),restretch(axes,target,canonical_fit(compact,target,count,exact,fit)))

def kernel_view(+compact: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},+fit: Fit(compact,target)) ->
  {Bands.stretched_to(Bands.contiguous_of(Some{compact}),Some{target}) ==
    Some{Views.View{restretched(Canonical.axes(Canonical.canonical_view(compact,count)),target),0}} : Maybe<&2,Views.View>}:
  +axes = Canonical.axes(Canonical.canonical_view(compact,count))
  +length = Equal.trans(Nat,List.length(&2,Views.Axis,axes),List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,U32,target),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),Metadata.extents_length(axes)),
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(compact,count)),compact,Canonical.canonical_shape(compact,count,exact)) :
      {List.length(&2,U32,_) == List.length(&2,U32,target) : Nat}
    fit_length(compact,target,fit))
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(compact),Some{Canonical.canonical_view(compact,count)},Canonical.view(compact,count,exact)) :
    {Bands.stretched_to(_,Some{target}) == Some{Views.View{restretched(axes,target),0}} : Maybe<&2,Views.View>}
  %Equal.sym(List<&2,Views.Axis>,Views.with_rank(axes,List.length(&2,U32,target)),axes,axes_ranked(axes,List.length(&2,U32,target),length)) :
    {Views.stretched_view(Views.stretch_axes(_,target),0) == Some{Views.View{restretched(axes,target),0}} : Maybe<&2,Views.View>}
  %Equal.sym(Maybe<&2,List<&2,Views.Axis>>,Views.stretch_axes(axes,target),Some{restretched(axes,target)},
      restretch(axes,target,canonical_fit(compact,target,count,exact,fit))) :
    {Views.stretched_view(_,0) == Some{Views.View{restretched(axes,target),0}} : Maybe<&2,Views.View>}
  {==}

# A band of a whole image holds the elements of the shape.
def whole_cells(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,
  +whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
  coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat}) ->
  {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  +leading = (U32.is_eq(images,1) && U32.is_zero(first) : Bool)
  +whole_rows = Boolean.left(leading,U32.is_eq(rows,height),whole)
  +one = Arithmetic.equal_of_is_eq(images,1,Boolean.left(U32.is_eq(images,1),U32.is_zero(first),whole_rows))
  +rows_height = Arithmetic.equal_of_is_eq(rows,height,Boolean.right(leading,U32.is_eq(rows,height),whole))
  +inner = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  %Equal.sym(U32,rows,height,rows_height) : {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(_),U32.to_nat(width))) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(shape),BandGather.elements(Bands.Geometry{images,channels,height,width}),inner,coherent,
    %Equal.sym(U32,images,1,one) : {Nat.mul(U32.to_nat(_),inner) == inner : Nat}
    Addition.zero_right(inner))

# The cells of a band of image rows: its channels, rows and width.
def three_cells(+c: U32,+w: U32,+channels: U32,+width: U32,+rows: U32,same_c: {c == channels : U32},same_w: {w == width : U32}) ->
  {ShapeLaws.elements([c,rows,w]) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  %same_c : {ShapeLaws.elements([c,rows,w]) == Nat.mul(U32.to_nat(_),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}
  %same_w : {ShapeLaws.elements([c,rows,w]) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(rows),U32.to_nat(_))) : Nat}
  ShapeLaws.elements_three(c,rows,w)

def cells_count(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width})) ->
  {ShapeLaws.elements(band_cells(shape,rows)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  match shape:
    case [+c,+h,+w]:
      Shape.Both{one,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      three_cells(c,w,channels,width,rows,same_c,same_w)
    case [+n,+c,+h,+w]:
      Shape.Both{same_n,Shape.Both{same_c,Shape.Both{same_h,same_w}}} = image
      Equal.trans(Nat,ShapeLaws.elements([1,c,rows,w]),ShapeLaws.elements([c,rows,w]),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
        Addition.zero_right(ShapeLaws.elements([c,rows,w])),three_cells(c,w,channels,width,rows,same_c,same_w))
    case []: Empty.absurd({ShapeLaws.elements(band_cells([],rows)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},image)
    case [a]: Empty.absurd({ShapeLaws.elements(band_cells([a],rows)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},image)
    case [a,b]: Empty.absurd({ShapeLaws.elements(band_cells([a,b],rows)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd({ShapeLaws.elements(band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows)) ==
        Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},image)

# The metadata of a band, field by field.
def metadata_local(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length)) -> {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool}:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  local

def metadata_rows(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(images),U32.to_nat(height))) == True{} : Bool}:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  rows_fit

def metadata_length(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,
  metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,length)) ->
  {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat}:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  length_exact

# A band combined with its part through the contiguous view of the compacted
# extents stretched to the extents of its cells: the band holds the combined
# values when the part reads the stretched source at every cell.
def part_contract(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+limits: Machine.Limits,
  +compact: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},+fit: Fit(compact,target),
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+source: Buffer.Content<Element>,+source_count: U32,
  +left_ready: Buffer.Sized<Element,left>,+piece_ready: Buffer.Sized<Element,piece>,
  +left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  +cells: {ShapeLaws.elements(target) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  piece_fits: {Views.fits(Views.View{restretched(Canonical.axes(Canonical.canonical_view(compact,count)),target),0},U32.from_nat(Buffer.length(~Element,piece))) == True{} : Bool},
  source_values: Contract.Values(~Element,~Right,~right_value,right_context,source),source_length: {Buffer.content_length(~Element,source) == U32.to_nat(source_count) : Nat},
  source_fits: {Views.fits_nonempty(False{},Views.View{full,0},source_count) == True{} : Bool},
  how: How<Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),restretched(Canonical.axes(Canonical.canonical_view(compact,count)),target),full,
    images,channels,height,width,first,rows>,
  width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandCombine.LeafContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},first,rows,
    Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),
      Bands.stretched_to(Bands.contiguous_of(Some{compact}),Some{target}),Buffer.pack(~Element,left),Buffer.pack(~Element,piece)))):
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +axes = Canonical.axes(Canonical.canonical_view(compact,count))
  +kernel_axes = restretched(axes,target)
  +local = Bands.local_count(geometry,rows)
  +written = BandShapeCount.count_refinement_exact(geometry,rows,Buffer.length(~Element,left),
    BandShape.count(~Element,geometry,first,rows,left,width_positive,left_metadata,left_ready))
  +cell_count = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +length_exact = metadata_length(images,channels,height,width,first,rows,Buffer.length(~Element,left),left_metadata)
  +kernel_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(kernel_axes)),ShapeLaws.elements(target),U32.to_nat(local),
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),Views.axis_extents(kernel_axes),target,
      Lanes.stretched_extents(axes,target,kernel_axes,kernel_lanes(compact,target,count,exact,fit))),
    Equal.trans(Nat,ShapeLaws.elements(target),cell_count,U32.to_nat(local),cells,
      Equal.trans(Nat,cell_count,Buffer.length(~Element,left),U32.to_nat(local),Equal.sym(Nat,Buffer.length(~Element,left),cell_count,length_exact),
        Equal.sym(Nat,U32.to_nat(local),Buffer.length(~Element,left),written))))
  %Equal.sym(Maybe<&2,Views.View>,Bands.stretched_to(Bands.contiguous_of(Some{compact}),Some{target}),Some{Views.View{kernel_axes,0}},
      kernel_view(compact,target,count,exact,fit)) :
    BandCombine.LeafContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,first,rows,
      Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,local,_,Buffer.pack(~Element,left),Buffer.pack(~Element,piece))))
  band_contract(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,full,kernel_axes,geometry,first,rows,local,limits,
    left,piece,source,source_count,left_ready,piece_ready,left_metadata,left_values,written,kernel_exact,piece_fits,source_values,source_length,source_fits,
    reads(~Element,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,source),kernel_axes,full,images,channels,height,width,first,rows,how),
    metadata_rows(images,channels,height,width,first,rows,Buffer.length(~Element,left),left_metadata),height_positive,pixels)

# The source every band reads: the view of the source stretched to the
# shape, its axes and the room of its owner.
def source_view(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  stretched: {Views.stretch_axes(source_axes(source,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>}) ->
  {Bands.stretched_source(source,shape) == Some{Views.View{full,0}} : Maybe<&2,Views.View>}:
  +axes = source_axes(source,count,List.length(&2,U32,shape))
  Equal.trans(Maybe<&2,Views.View>,Bands.stretched_source(source,shape),Views.stretched_view(Views.stretch_axes(axes,shape),0),Some{Views.View{full,0}},
    stretched_source(source,count,shape,exact),
    Equal.cong(Maybe<&2,List<&2,Views.Axis>>,Maybe<&2,Views.View>,result => Views.stretched_view(result,0),Views.stretch_axes(axes,shape),Some{full},stretched))

def out_length(+axes: List<&2,Views.Axis>,+target: List<&2,U32>,+out: List<&2,Views.Axis>,lanes: Lanes.Stretched(axes,target,out)) ->
  {List.length(&2,Views.Axis,out) == List.length(&2,U32,target) : Nat}:
  Equal.trans(Nat,List.length(&2,Views.Axis,out),List.length(&2,U32,Views.axis_extents(out)),List.length(&2,U32,target),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(out)),List.length(&2,Views.Axis,out),Metadata.extents_length(out)),
    Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),Views.axis_extents(out),target,Lanes.stretched_extents(axes,target,out,lanes)))

def ranked_source(+source: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source,count,List.length(&2,U32,shape)),shape,full)) ->
  {List.length(&2,U32,Shapes.with_rank(source,List.length(&2,U32,shape))) == List.length(&2,U32,shape) : Nat}:
  +rank = List.length(&2,U32,shape)
  +axes = source_axes(source,count,rank)
  Equal.trans(Nat,List.length(&2,U32,Shapes.with_rank(source,rank)),List.length(&2,Views.Axis,axes),rank,
    Equal.trans(Nat,List.length(&2,U32,Shapes.with_rank(source,rank)),List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),
      Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),Shapes.with_rank(source,rank),Views.axis_extents(axes),
        Equal.sym(List<&2,U32>,Views.axis_extents(axes),Shapes.with_rank(source,rank),source_extents(source,count,rank,exact))),
      Metadata.extents_length(axes)),
    stretched_length(axes,shape,full,lanes))

def within_length(+extents: List<&2,U32>,+shape: List<&2,U32>,within: Within(extents,shape)) -> {List.length(&2,U32,extents) == List.length(&2,U32,shape) : Nat}:
  match extents shape:
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Con{+size,+sizes}:
      Shape.Both{below,more} = within
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,rest),List.length(&2,U32,sizes),within_length(rest,sizes,more))
    case Nil{} Con{size,sizes}: Empty.absurd({List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{size,sizes}) : Nat},within)
    case Con{extent,rest} Nil{}: Empty.absurd({List.length(&2,U32,Con{extent,rest}) == List.length(&2,U32,Nil{}) : Nat},within)

def source_capacity(~Element: Data,+source: Buffer.Model<Element>,ready: Buffer.Sized<Element,source>) ->
  {U32.is_le(Tree.capacity(Element,Buffer.tree(~Element,source)),2147483648) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = ready
  Buffer.capacity_bound(~Element,source,depth_ok)

def source_allocatable(~Element: Data,+source: Buffer.Model<Element>,ready: Buffer.Sized<Element,source>) ->
  {U32.is_le(Tree.capacity(Element,Buffer.tree(~Element,source)),1073741824) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = ready
  Buffer.capacity_allocatable(~Element,source,depth_ok)

def source_room(~Element: Data,+source_shape: List<&2,U32>,+count: U32,+rank: Nat,+source: Buffer.Model<Element>,ready: Buffer.Sized<Element,source>,
  length_exact: {Buffer.length(~Element,source) == U32.to_nat(count) : Nat},exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat}) ->
  {Nat.is_le(ShapeLaws.elements(Shapes.with_rank(source_shape,rank)),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,source)))) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = ready
  +capacity = U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,source)))
  %Equal.sym(Nat,ShapeLaws.elements(Shapes.with_rank(source_shape,rank)),ShapeLaws.elements(source_shape),rank_elements(source_shape,rank)) :
    {Nat.is_le(_,capacity) == True{} : Bool}
  %Equal.sym(Nat,ShapeLaws.elements(source_shape),U32.to_nat(count),exact) : {Nat.is_le(_,capacity) == True{} : Bool}
  %length_exact : {Nat.is_le(_,capacity) == True{} : Bool}
  room

# The part of a band within the shape holds at most the elements of the
# source, and its compacted extents stretch to the cells of the band.
def part_bound(~Element: Data,+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+target: List<&2,U32>,+source: Buffer.Model<Element>,
  ready: Buffer.Sized<Element,source>,length_exact: {Buffer.length(~Element,source) == U32.to_nat(count) : Nat},+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+within: Within(target,shape)) ->
  {Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,target)),target)),
    U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,source)))) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  +axes = source_axes(source_shape,count,rank)
  +ranked = Shapes.with_rank(source_shape,rank)
  +capacity = U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,source)))
  %Equal.sym(Nat,List.length(&2,U32,target),rank,within_length(target,shape,within)) :
    {Nat.is_le(ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,_),target)),capacity) == True{} : Bool}
  Order.transitive(ShapeLaws.elements(Bands.compact_extents(ranked,target)),ShapeLaws.elements(ranked),capacity,
    %source_extents(source_shape,count,rank,exact) : {Nat.is_le(ShapeLaws.elements(Bands.compact_extents(_,target)),ShapeLaws.elements(_)) == True{} : Bool}
    compact_within(axes,shape,full,target,lanes,within),
    source_room(~Element,source_shape,count,rank,source,ready,length_exact,exact))

def part_fit(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+target: List<&2,U32>,
  +exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),
  within: Within(target,shape)) -> Fit(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,target)),target),target):
  compact_fit(Shapes.with_rank(source_shape,List.length(&2,U32,target)),target,
    %Equal.sym(Nat,List.length(&2,U32,target),List.length(&2,U32,shape),within_length(target,shape,within)) :
      {List.length(&2,U32,Shapes.with_rank(source_shape,_)) == _ : Nat}
    ranked_source(source_shape,count,shape,full,exact,lanes))

def source_elements(+source_shape: List<&2,U32>,+count: U32,+rank: Nat,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat}) ->
  {ShapeLaws.elements(Views.axis_extents(source_axes(source_shape,count,rank))) == U32.to_nat(count) : Nat}:
  %Equal.sym(List<&2,U32>,Views.axis_extents(source_axes(source_shape,count,rank)),Shapes.with_rank(source_shape,rank),source_extents(source_shape,count,rank,exact)) :
    {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(Shapes.with_rank(source_shape,rank)),ShapeLaws.elements(source_shape),U32.to_nat(count),rank_elements(source_shape,rank),exact)

# The last address the stretched source reads lies inside the source.
def source_bound(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Nat.is_lt(Digits.address(Lanes.points(full,Lanes.lasts(full)),0n,False{}),U32.to_nat(count)) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  kernel_bound(source_axes(source_shape,count,rank),shape,full,count,lanes,
    source_layout(source_shape,count,rank,exact,source_nonempty(source_shape,count,shape,full,exact,lanes,nonempty)),nonempty,source_elements(source_shape,count,rank,exact))

# The stretched source fits the count of the source.
def source_fits(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Views.fits_nonempty(False{},Views.View{full,0},count) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  +axes = source_axes(source_shape,count,rank)
  %Equal.trans(Bool,Shapes.has_zero(Views.axis_extents(full)),Shapes.has_zero(shape),False{},
      Equal.cong(List<&2,U32>,Bool,extents => Shapes.has_zero(extents),Views.axis_extents(full),shape,Lanes.stretched_extents(axes,shape,full,lanes)),nonempty) :
    {Views.fits_nonempty(_,Views.View{full,0},count) == True{} : Bool}
  kernel_fits(axes,shape,full,count,lanes,source_layout(source_shape,count,rank,exact,source_nonempty(source_shape,count,shape,full,exact,lanes,nonempty)),nonempty,
    source_elements(source_shape,count,rank,exact))

# A band of a whole image compacts the source to itself, and reads through
# axes that agree with the source axes.
def whole_compact(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full)) ->
  {Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape) == Shapes.with_rank(source_shape,List.length(&2,U32,shape)) : List<&2,U32>}:
  +rank = List.length(&2,U32,shape)
  %source_extents(source_shape,count,rank,exact) : {Bands.compact_extents(_,shape) == _ : List<&2,U32>}
  compact_source(source_axes(source_shape,count,rank),shape,full,lanes)

def whole_extents(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)) == U32.to_nat(parts) : Nat}) ->
  {Views.axis_extents(Canonical.axes(Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),parts))) ==
    Views.axis_extents(source_axes(source_shape,count,List.length(&2,U32,shape))) : List<&2,U32>}:
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +ranked = Shapes.with_rank(source_shape,rank)
  Equal.trans(List<&2,U32>,Views.axis_extents_of(Canonical.canonical_view(compact,parts)),compact,Views.axis_extents(source_axes(source_shape,count,rank)),
    Canonical.canonical_shape(compact,parts,parts_exact),
    Equal.trans(List<&2,U32>,compact,ranked,Views.axis_extents(source_axes(source_shape,count,rank)),whole_compact(source_shape,count,shape,full,exact,lanes),
      Equal.sym(List<&2,U32>,Views.axis_extents(source_axes(source_shape,count,rank)),ranked,source_extents(source_shape,count,rank,exact))))

def whole_layout(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+nonempty: {Shapes.has_zero(shape) == False{} : Bool},+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)) == U32.to_nat(parts) : Nat}) ->
  Linear.Layout(Canonical.axes(Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),parts))):
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +ranked = Shapes.with_rank(source_shape,rank)
  canonical_layout(compact,parts,parts_exact,
    Equal.trans(Bool,Shapes.has_zero(compact),Shapes.has_zero(ranked),False{},
      Equal.cong(List<&2,U32>,Bool,extents => Shapes.has_zero(extents),compact,ranked,whole_compact(source_shape,count,shape,full,exact,lanes)),
      Equal.trans(Bool,Shapes.has_zero(ranked),Shapes.has_zero(source_shape),False{},rank_zero(source_shape,rank),
        source_nonempty(source_shape,count,shape,full,exact,lanes,nonempty))))

# What a band combined with its part holds, for the view the kernel reads
# the part through.
def SpreadLeaf(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),+left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+images: U32,+channels: U32,+height: U32,+width: U32,
  +first: U32,+rows: U32,+view: Maybe<&2,Views.View>,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>) -> Data:
  BandCombine.LeafContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},first,rows,
    Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),view,
      Buffer.pack(~Element,left),Buffer.pack(~Element,piece))))

def sized_room(~Element: Data,+model: Buffer.Model<Element>,ready: Buffer.Sized<Element,model>) ->
  {Nat.is_le(Buffer.length(~Element,model),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,model)))) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = ready
  room

# The part of a band of a whole image: the source through the compacted
# stretched view.
def whole_part(~Element: Data,+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  stretched: {Views.stretch_axes(source_axes(source_shape,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)) == U32.to_nat(parts) : Nat},
  +fill: Element,+source: Buffer.Content<Element>) ->
  {part_views(~Element,Bands.compact_of(source_shape,Bands.stretched_source(source_shape,shape)),
      Bands.contiguous_of(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),
      Bands.count_of_extents(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),fill,source) ==
    part_views(~Element,Some{Views.View{Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full),0}},
      Some{Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),parts)},Some{parts},fill,source) : Buffer.Content<Element>}:
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)
  +copied = {Views.View{Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full),0} : Views.View}
  +whole = part_views(~Element,Some{copied},Some{Canonical.canonical_view(compact,parts)},Some{parts},fill,source)
  %Equal.sym(Maybe<&2,Views.View>,Bands.stretched_source(source_shape,shape),Some{Views.View{full,0}},source_view(source_shape,count,shape,full,exact,stretched)) :
    {part_views(~Element,Bands.compact_of(source_shape,_),Bands.contiguous_of(Some{compact}),Bands.count_of_extents(Some{compact}),fill,source) == whole : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(compact),Some{Canonical.canonical_view(compact,parts)},Canonical.view(compact,parts,parts_exact)) :
    {part_views(~Element,Some{copied},_,Bands.count_of_extents(Some{compact}),fill,source) == whole : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{compact}),Some{parts},CountCompletion.numel(compact,parts,parts_exact)) :
    {part_views(~Element,Some{copied},Some{Canonical.canonical_view(compact,parts)},_,fill,source) == whole : Buffer.Content<Element>}
  {==}

# The kernel of a band of a whole image reads the axes of the stretched
# source.
def whole_agree(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+nonempty: {Shapes.has_zero(shape) == False{} : Bool},+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)) == U32.to_nat(parts) : Nat},
  +fit: Fit(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),shape)) ->
  Agree(restretched(Canonical.axes(Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),parts)),shape),full):
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +axes = Canonical.axes(Canonical.canonical_view(compact,parts))
  +source = source_axes(source_shape,count,rank)
  stretched_agree(restretched(axes,shape),full,axes,shape,source,kernel_lanes(compact,shape,parts,parts_exact,fit),lanes,
    layouts_agree(axes,source,whole_layout(source_shape,count,shape,full,exact,lanes,nonempty,parts,parts_exact),
      source_layout(source_shape,count,rank,exact,source_nonempty(source_shape,count,shape,full,exact,lanes,nonempty)),
      whole_extents(source_shape,count,shape,full,exact,lanes,parts,parts_exact)))

def whole_copy_agree(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full),+nonempty: {Shapes.has_zero(shape) == False{} : Bool},+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)) == U32.to_nat(parts) : Nat}) ->
  Agree(Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full),
    Canonical.axes(Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape),parts))):
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +axes = Canonical.axes(Canonical.canonical_view(compact,parts))
  +source = source_axes(source_shape,count,rank)
  %Equal.sym(Nat,List.length(&2,Views.Axis,full),rank,out_length(source,shape,full,lanes)) :
    Agree(Bands.compact_axes(Shapes.with_rank(source_shape,_),full),axes)
  %source_extents(source_shape,count,rank,exact) : Agree(Bands.compact_axes(_,full),axes)
  compact_agree(source,shape,full,axes,lanes,
    layouts_agree(source,axes,source_layout(source_shape,count,rank,exact,source_nonempty(source_shape,count,shape,full,exact,lanes,nonempty)),
      whole_layout(source_shape,count,shape,full,exact,lanes,nonempty,parts,parts_exact),
      Equal.sym(List<&2,U32>,Views.axis_extents(axes),Views.axis_extents(source),whole_extents(source_shape,count,shape,full,exact,lanes,parts,parts_exact))))

def whole_copied_extents(+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,
  +lanes: Lanes.Stretched(source_axes(source_shape,count,List.length(&2,U32,shape)),shape,full)) ->
  {Views.axis_extents(Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full)) ==
    Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape) : List<&2,U32>}:
  +rank = List.length(&2,U32,shape)
  +copied = Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full)
  %out_length(source_axes(source_shape,count,rank),shape,full,lanes) :
    {Views.axis_extents(copied) == Bands.compact_extents(Shapes.with_rank(source_shape,_),shape) : List<&2,U32>}
  %Lanes.stretched_extents(source_axes(source_shape,count,rank),shape,full,lanes) :
    {Views.axis_extents(copied) == Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),_) : List<&2,U32>}
  Lanes.compact_extents_of(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full)

# A stretched view reads within the count it fits when its extents have no
# zero.
def kernel_inside(+compact: List<&2,U32>,+target: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(compact) == U32.to_nat(count) : Nat},+fit: Fit(compact,target),
  +nonempty: {Shapes.has_zero(target) == False{} : Bool}) ->
  {Views.fits_nonempty(False{},Views.View{restretched(Canonical.axes(Canonical.canonical_view(compact,count)),target),0},count) == True{} : Bool}:
  +axes = Canonical.axes(Canonical.canonical_view(compact,count))
  +lanes = kernel_lanes(compact,target,count,exact,fit)
  %Equal.trans(Bool,Shapes.has_zero(Views.axis_extents(restretched(axes,target))),Shapes.has_zero(target),False{},
      Equal.cong(List<&2,U32>,Bool,extents => Shapes.has_zero(extents),Views.axis_extents(restretched(axes,target)),target,
        Lanes.stretched_extents(axes,target,restretched(axes,target),lanes)),nonempty) :
    {Views.fits_nonempty(_,Views.View{restretched(axes,target),0},count) == True{} : Bool}
  stretched_fits(compact,count,target,restretched(axes,target),lanes,exact)

# A band of a whole image combined with the source itself.
def whole_taken(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+source: Buffer.Model<Element>,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  +made: {Buffer.content(~Element,piece) == Buffer.content(~Element,source) : Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),Some{shape}),left,piece):
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +lanes = Lanes.stretched(source_axes(source_shape,source_count,rank),shape,full,stretched)
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,source))
  +parts = part_count(compact)
  +parts_exact = part_exact(compact,capacity,source_capacity(~Element,source,source_ready),
    part_bound(~Element,source_shape,source_count,shape,full,shape,source,source_ready,source_length,source_exact,lanes,within_self(shape)))
  +fit = part_fit(source_shape,source_count,shape,full,shape,source_exact,lanes,within_self(shape))
  +kernel_axes = restretched(Canonical.axes(Canonical.canonical_view(compact,parts)),shape)
  +length_parts = Equal.trans(Nat,Buffer.length(~Element,source),U32.to_nat(source_count),ShapeLaws.elements(compact),source_length,
    Equal.trans(Nat,U32.to_nat(source_count),ShapeLaws.elements(source_shape),ShapeLaws.elements(compact),Equal.sym(Nat,ShapeLaws.elements(source_shape),U32.to_nat(source_count),source_exact),
      Equal.trans(Nat,ShapeLaws.elements(source_shape),ShapeLaws.elements(Shapes.with_rank(source_shape,rank)),ShapeLaws.elements(compact),
        Equal.sym(Nat,ShapeLaws.elements(Shapes.with_rank(source_shape,rank)),ShapeLaws.elements(source_shape),rank_elements(source_shape,rank)),
        Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),Shapes.with_rank(source_shape,rank),compact,
          Equal.sym(List<&2,U32>,compact,Shapes.with_rank(source_shape,rank),whole_compact(source_shape,source_count,shape,full,source_exact,lanes))))))
  +piece_length = Equal.trans(Nat,Buffer.length(~Element,piece),Buffer.length(~Element,source),ShapeLaws.elements(compact),
    Equal.trans(Nat,Buffer.length(~Element,piece),Buffer.content_length(~Element,Buffer.content(~Element,source)),Buffer.length(~Element,source),
      Equal.trans(Nat,Buffer.length(~Element,piece),Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.content_length(~Element,Buffer.content(~Element,source)),
        Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.length(~Element,piece),Buffer.content_length_of(~Element,piece)),
        Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,piece),Buffer.content(~Element,source),made)),
      Buffer.content_length_of(~Element,source)),
    length_parts)
  part_contract(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,full,images,channels,height,width,first,rows,limits,
    compact,shape,parts,parts_exact,fit,left,piece,Buffer.content(~Element,source),source_count,left_ready,
    Buffer.ready_content(~Element,piece,
      %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,piece),Buffer.content(~Element,source),made) : Buffer.ContentReady<Element,_>
      Buffer.content_ready(~Element,source,source_ready)),
    left_metadata,left_values,
    whole_cells(shape,images,channels,height,width,first,rows,whole,coherent),
    %Equal.sym(U32,U32.from_nat(Buffer.length(~Element,piece)),parts,
        Equal.cong(Nat,U32,U32.from_nat,Buffer.length(~Element,piece),ShapeLaws.elements(compact),piece_length)) :
      {Views.fits(Views.View{kernel_axes,0},_) == True{} : Bool}
    stretched_fits(compact,parts,shape,kernel_axes,kernel_lanes(compact,shape,parts,parts_exact,fit),parts_exact),
    source_values,Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,source)),Buffer.length(~Element,source),U32.to_nat(source_count),
      Buffer.content_length_of(~Element,source),source_length),
    source_fits(source_shape,source_count,shape,full,source_exact,lanes,nonempty),
    Same{whole_agree(source_shape,source_count,shape,full,source_exact,lanes,nonempty,parts,parts_exact,fit),whole,
      Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,Buffer.content(~Element,piece)),
        Buffer.content_tree(~Element,Buffer.content(~Element,source)),
        Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,piece)),Buffer.tree(~Element,piece),Buffer.content_tree_of(~Element,piece)),
        Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,content => Buffer.content_tree(~Element,content),Buffer.content(~Element,piece),Buffer.content(~Element,source),made))},
    width_positive,height_positive,pixels)

# A band of a whole image combined with its copy of the source.
def whole_copied(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +whole: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  made: {Buffer.content(~Element,piece) ==
    part_views(~Element,Bands.compact_of(source_shape,Bands.stretched_source(source_shape,shape)),
      Bands.contiguous_of(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),
      Bands.count_of_extents(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),fill,Buffer.content(~Element,source)) :
    Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,shape)),shape)}),Some{shape}),left,piece):
  Buffer.Model{+source_tree,+source_size,+source_certificate} = source
  +source = {Buffer.Model{source_tree,source_size,source_certificate} : Buffer.Model<Element>}
  +rank = List.length(&2,U32,shape)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,rank),shape)
  +lanes = Lanes.stretched(source_axes(source_shape,source_count,rank),shape,full,stretched)
  +capacity = Tree.capacity(Element,source_tree)
  +bound = part_bound(~Element,source_shape,source_count,shape,full,shape,source,source_ready,source_length,source_exact,lanes,within_self(shape))
  +parts = part_count(compact)
  +parts_exact = part_exact(compact,capacity,source_capacity(~Element,source,source_ready),bound)
  +fit = part_fit(source_shape,source_count,shape,full,shape,source_exact,lanes,within_self(shape))
  +kernel_axes = restretched(Canonical.axes(Canonical.canonical_view(compact,parts)),shape)
  +copied = Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,full)),full)
  +target = Canonical.canonical_view(compact,parts)
  +output = Buffer.fresh(~Element,Storage.depth_for(parts),fill,U32.to_nat(parts))
  +output_ready = fresh_of(~Element,fill,parts,Depth.bounded(parts,
    Arithmetic.count_within(parts,ShapeLaws.elements(compact),capacity,1073741824,Equal.sym(Nat,ShapeLaws.elements(compact),U32.to_nat(parts),parts_exact),bound,
      source_allocatable(~Element,source,source_ready))))
  +program = Copy.program_content(~Element,parts,Views.View{copied,0},target,Buffer.content(~Element,source),Buffer.content(~Element,output))
  +copied_content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,piece),
    part_views(~Element,Bands.compact_of(source_shape,Bands.stretched_source(source_shape,shape)),Bands.contiguous_of(Some{compact}),Bands.count_of_extents(Some{compact}),
      fill,Buffer.content(~Element,source)),
    program,made,whole_part(~Element,source_shape,source_count,shape,full,source_exact,stretched,parts,parts_exact,fill,Buffer.content(~Element,source)))
  +piece_length = Equal.trans(Nat,Buffer.length(~Element,piece),Buffer.content_length(~Element,Buffer.content(~Element,piece)),U32.to_nat(parts),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.length(~Element,piece),Buffer.content_length_of(~Element,piece)),
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.content_length(~Element,program),U32.to_nat(parts),
      Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,piece),program,copied_content),
      GatherValues.program_length(~Element,parts,Views.View{copied,0},target,source,output)))
  part_contract(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,full,images,channels,height,width,first,rows,limits,
    compact,shape,parts,parts_exact,fit,left,piece,Buffer.content(~Element,source),source_count,left_ready,
    Buffer.ready_content(~Element,piece,
      %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,piece),program,copied_content) : Buffer.ContentReady<Element,_>
      GatherValues.program_ready(~Element,parts,Views.View{copied,0},target,source,output,output_ready)),
    left_metadata,left_values,whole_cells(shape,images,channels,height,width,first,rows,whole,coherent),
    %Equal.sym(U32,U32.from_nat(Buffer.length(~Element,piece)),parts,
        Equal.trans(U32,U32.from_nat(Buffer.length(~Element,piece)),U32.from_nat(U32.to_nat(parts)),parts,
          Equal.cong(Nat,U32,U32.from_nat,Buffer.length(~Element,piece),U32.to_nat(parts),piece_length),Arithmetic.from_nat_exact(parts,U32.to_nat(parts),{==}))) :
      {Views.fits(Views.View{kernel_axes,0},_) == True{} : Bool}
    stretched_fits(compact,parts,shape,kernel_axes,kernel_lanes(compact,shape,parts,parts_exact,fit),parts_exact),
    source_values,source_length,source_fits(source_shape,source_count,shape,full,source_exact,lanes,nonempty),
    Whole{copied,compact,parts,Tree.fresh(Element,Storage.depth_for(parts),fill),U32.to_nat(parts),
      whole_agree(source_shape,source_count,shape,full,source_exact,lanes,nonempty,parts,parts_exact,fit),whole,
      whole_copy_agree(source_shape,source_count,shape,full,source_exact,lanes,nonempty,parts,parts_exact),parts_exact,
      whole_copied_extents(source_shape,source_count,shape,full,lanes),
      Buffer.content_ready(~Element,output,output_ready),source_capacity(~Element,output,output_ready),sized_room(~Element,output,output_ready),
      kernel_inside(compact,shape,parts,parts_exact,fit,nonempty),
      Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,Buffer.content(~Element,piece)),
        piece_tree(~Element,Views.View{copied,0},parts,source_tree,Tree.fresh(Element,Storage.depth_for(parts),fill)),
        Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,piece)),Buffer.tree(~Element,piece),Buffer.content_tree_of(~Element,piece)),
        Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,content => Buffer.content_tree(~Element,content),Buffer.content(~Element,piece),program,copied_content))},
    width_positive,height_positive,pixels)

# The cells of a band of image rows of an image shape.
def rows_cells(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,
  image: ImageShape(shape,Bands.Geometry{images,channels,height,width})) -> {Bands.rows_extents(shape,rows) == Some{band_cells(shape,rows)} : Maybe<&2,List<&2,U32>>}:
  match shape:
    case [c,h,w]: {==}
    case [n,c,h,w]: {==}
    case []: Empty.absurd({Bands.rows_extents([],rows) == Some{band_cells([],rows)} : Maybe<&2,List<&2,U32>>},image)
    case [a]: Empty.absurd({Bands.rows_extents([a],rows) == Some{band_cells([a],rows)} : Maybe<&2,List<&2,U32>>},image)
    case [a,b]: Empty.absurd({Bands.rows_extents([a,b],rows) == Some{band_cells([a,b],rows)} : Maybe<&2,List<&2,U32>>},image)
    case Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}}:
      Empty.absurd({Bands.rows_extents(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows) == Some{band_cells(Con{a,Con{b,Con{c,Con{d,Con{e,more}}}}},rows)} :
        Maybe<&2,List<&2,U32>>},image)

# The part of a band of image rows: the source through the compacted view of
# its image and its rows.
def rows_part(~Element: Data,+source_shape: List<&2,U32>,+count: U32,+shape: List<&2,U32>,+full: List<&2,Views.Axis>,+exact: {ShapeLaws.elements(source_shape) == U32.to_nat(count) : Nat},
  stretched: {Views.stretch_axes(source_axes(source_shape,count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +height: U32,+first: U32,+rows: U32,+band: List<&2,Views.Axis>,+offset: U32,
  result: {Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}) == Some{Views.View{band,offset}} : Maybe<&2,Views.View>},+parts: U32,
  +parts_exact: {ShapeLaws.elements(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,band_cells(shape,rows))),band_cells(shape,rows))) == U32.to_nat(parts) : Nat},
  +fill: Element,+source: Buffer.Content<Element>) ->
  {part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,Bands.stretched_source(source_shape,shape))),
      Bands.contiguous_of(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,band_cells(shape,rows))),band_cells(shape,rows))}),
      Bands.count_of_extents(Some{Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,band_cells(shape,rows))),band_cells(shape,rows))}),fill,source) ==
    part_views(~Element,Some{Views.View{Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),band),offset}},
      Some{Canonical.canonical_view(Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,band_cells(shape,rows))),band_cells(shape,rows)),parts)},
      Some{parts},fill,source) : Buffer.Content<Element>}:
  +cells = band_cells(shape,rows)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,cells)),cells)
  +copied = {Views.View{Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),band),offset} : Views.View}
  +copy = part_views(~Element,Some{copied},Some{Canonical.canonical_view(compact,parts)},Some{parts},fill,source)
  %Equal.sym(Maybe<&2,Views.View>,Bands.stretched_source(source_shape,shape),Some{Views.View{full,0}},source_view(source_shape,count,shape,full,exact,stretched)) :
    {part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,_)),Bands.contiguous_of(Some{compact}),Bands.count_of_extents(Some{compact}),
      fill,source) == copy : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}),Some{Views.View{band,offset}},result) :
    {part_views(~Element,Bands.compact_of(source_shape,_),Bands.contiguous_of(Some{compact}),Bands.count_of_extents(Some{compact}),fill,source) == copy : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(compact),Some{Canonical.canonical_view(compact,parts)},Canonical.view(compact,parts,parts_exact)) :
    {part_views(~Element,Some{copied},_,Bands.count_of_extents(Some{compact}),fill,source) == copy : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Shapes.numel(Shapes.Shape{compact}),Some{parts},CountCompletion.numel(compact,parts,parts_exact)) :
    {part_views(~Element,Some{copied},Some{Canonical.canonical_view(compact,parts)},_,fill,source) == copy : Buffer.Content<Element>}
  {==}

# The axes a band of image rows copies through have the extents of the
# compacted cells.
def rows_same(+source_shape: List<&2,U32>,+shape: List<&2,U32>,+height: U32,+first: U32,+rows: U32,+full: List<&2,Views.Axis>,+band: List<&2,Views.Axis>,+offset: U32,
  +slices: Slices(shape,height,first,rows,full,band,offset),+full_extents: {Views.axis_extents(full) == shape : List<&2,U32>},
  +length: {List.length(&2,Views.Axis,band) == List.length(&2,U32,band_cells(shape,rows)) : Nat}) ->
  {Views.axis_extents(Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),band)) ==
    Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,band_cells(shape,rows))),band_cells(shape,rows)) : List<&2,U32>}:
  +cells = band_cells(shape,rows)
  +copied = Bands.compact_axes(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),band)
  %length : {Views.axis_extents(copied) == Bands.compact_extents(Shapes.with_rank(source_shape,_),cells) : List<&2,U32>}
  %slices_extents(shape,height,first,rows,full,band,offset,slices,full_extents) :
    {Views.axis_extents(copied) == Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),_) : List<&2,U32>}
  Lanes.compact_extents_of(Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band)),band)

# A band of image rows combined with its copy of the source.
def rows_copied(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  ~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,+band: List<&2,Views.Axis>,+offset: U32,
  +result: {Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}) == Some{Views.View{band,offset}} : Maybe<&2,Views.View>},
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  made: {Buffer.content(~Element,piece) ==
    part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,Bands.stretched_source(source_shape,shape))),
      Bands.contiguous_of(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),Bands.count_of_extents(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),
      fill,Buffer.content(~Element,source)) : Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),Bands.rows_extents(shape,rows)),left,piece):
  Buffer.Model{+source_tree,+source_size,+source_certificate} = source
  +source = {Buffer.Model{source_tree,source_size,source_certificate} : Buffer.Model<Element>}
  +rank = List.length(&2,U32,shape)
  +cells = band_cells(shape,rows)
  +compact = Bands.compact_extents(Shapes.with_rank(source_shape,List.length(&2,U32,cells)),cells)
  +lanes = Lanes.stretched(source_axes(source_shape,source_count,rank),shape,full,stretched)
  +full_extents = Lanes.stretched_extents(source_axes(source_shape,source_count,rank),shape,full,lanes)
  +local = metadata_local(images,channels,height,width,first,rows,Buffer.length(~Element,left),left_metadata)
  +span = Inspection.local_span(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,height_bound,local)
  +within = cells_within(shape,images,channels,height,width,first,rows,image,nonempty,span)
  +slices = slices_of(shape,height,first,rows,full,band,offset,result)
  +band_length = Equal.trans(Nat,List.length(&2,Views.Axis,band),List.length(&2,U32,Views.axis_extents(band)),List.length(&2,U32,cells),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(band)),List.length(&2,Views.Axis,band),Metadata.extents_length(band)),
    Equal.cong(List<&2,U32>,Nat,extents => List.length(&2,U32,extents),Views.axis_extents(band),cells,slices_extents(shape,height,first,rows,full,band,offset,slices,full_extents)))
  +capacity = Tree.capacity(Element,source_tree)
  +bound = part_bound(~Element,source_shape,source_count,shape,full,cells,source,source_ready,source_length,source_exact,lanes,within)
  +parts = part_count(compact)
  +parts_exact = part_exact(compact,capacity,source_capacity(~Element,source,source_ready),bound)
  +fit = part_fit(source_shape,source_count,shape,full,cells,source_exact,lanes,within)
  +kernel_axes = restretched(Canonical.axes(Canonical.canonical_view(compact,parts)),cells)
  +compacted = Shapes.with_rank(source_shape,List.length(&2,Views.Axis,band))
  +copied = Bands.compact_axes(compacted,band)
  +target = Canonical.canonical_view(compact,parts)
  +output = Buffer.fresh(~Element,Storage.depth_for(parts),fill,U32.to_nat(parts))
  +output_ready = fresh_of(~Element,fill,parts,Depth.bounded(parts,
    Arithmetic.count_within(parts,ShapeLaws.elements(compact),capacity,1073741824,Equal.sym(Nat,ShapeLaws.elements(compact),U32.to_nat(parts),parts_exact),bound,
      source_allocatable(~Element,source,source_ready))))
  +program = Copy.program_content(~Element,parts,Views.View{copied,offset},target,Buffer.content(~Element,source),Buffer.content(~Element,output))
  +extents = rows_cells(shape,images,channels,height,width,rows,image)
  +copied_content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,piece),
    part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,Bands.stretched_source(source_shape,shape))),Bands.contiguous_of(Some{compact}),
      Bands.count_of_extents(Some{compact}),fill,Buffer.content(~Element,source)),
    program,
    %extents :
      {Buffer.content(~Element,piece) ==
        part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,Bands.stretched_source(source_shape,shape))),
          Bands.contiguous_of(Bands.compact_with(source_shape,_)),Bands.count_of_extents(Bands.compact_with(source_shape,_)),fill,Buffer.content(~Element,source)) :
        Buffer.Content<Element>}
    made,
    rows_part(~Element,source_shape,source_count,shape,full,source_exact,stretched,height,first,rows,band,offset,result,parts,parts_exact,fill,Buffer.content(~Element,source)))
  +piece_length = Equal.trans(Nat,Buffer.length(~Element,piece),Buffer.content_length(~Element,Buffer.content(~Element,piece)),U32.to_nat(parts),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.length(~Element,piece),Buffer.content_length_of(~Element,piece)),
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,piece)),Buffer.content_length(~Element,program),U32.to_nat(parts),
      Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,piece),program,copied_content),
      GatherValues.program_length(~Element,parts,Views.View{copied,offset},target,source,output)))
  %Equal.sym(Maybe<&2,List<&2,U32>>,Bands.rows_extents(shape,rows),Some{cells},extents) :
    SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
      Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,_)),_),left,piece)
  part_contract(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,full,images,channels,height,width,first,rows,limits,
    compact,cells,parts,parts_exact,fit,left,piece,Buffer.content(~Element,source),source_count,left_ready,
    Buffer.ready_content(~Element,piece,
      %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,piece),program,copied_content) : Buffer.ContentReady<Element,_>
      GatherValues.program_ready(~Element,parts,Views.View{copied,offset},target,source,output,output_ready)),
    left_metadata,left_values,cells_count(shape,images,channels,height,width,rows,image),
    %Equal.sym(U32,U32.from_nat(Buffer.length(~Element,piece)),parts,
        Equal.trans(U32,U32.from_nat(Buffer.length(~Element,piece)),U32.from_nat(U32.to_nat(parts)),parts,
          Equal.cong(Nat,U32,U32.from_nat,Buffer.length(~Element,piece),U32.to_nat(parts),piece_length),Arithmetic.from_nat_exact(parts,U32.to_nat(parts),{==}))) :
      {Views.fits(Views.View{kernel_axes,0},_) == True{} : Bool}
    stretched_fits(compact,parts,cells,kernel_axes,kernel_lanes(compact,cells,parts,parts_exact,fit),parts_exact),
    source_values,source_length,source_fits(source_shape,source_count,shape,full,source_exact,lanes,nonempty),
    Rows{shape,band,offset,compacted,compact,parts,Tree.fresh(Element,Storage.depth_for(parts),fill),U32.to_nat(parts),limits,slices,
      slices_spreadable(shape,height,first,rows,full,band,offset,compacted,slices,
        %Equal.trans(Nat,rank,List.length(&2,U32,cells),List.length(&2,Views.Axis,band),
            Equal.sym(Nat,List.length(&2,U32,cells),rank,within_length(cells,shape,within)),
            Equal.sym(Nat,List.length(&2,Views.Axis,band),List.length(&2,U32,cells),band_length)) :
          Lanes.Spreadable(Shapes.with_rank(source_shape,_),full)
        full_spreadable(source_shape,source_count,shape,full,source_exact,lanes)),
      kernel_lanes(compact,cells,parts,parts_exact,fit),
      Equal.trans(List<&2,U32>,Views.axis_extents_of(target),compact,Views.axis_extents(copied),Canonical.canonical_shape(compact,parts,parts_exact),
        Equal.sym(List<&2,U32>,Views.axis_extents(copied),compact,rows_same(source_shape,shape,height,first,rows,full,band,offset,slices,full_extents,band_length))),
      full_extents,parts_exact,
      Buffer.content_ready(~Element,output,output_ready),source_capacity(~Element,output,output_ready),sized_room(~Element,output,output_ready),
      image,height_positive,height_bound,local,metadata_rows(images,channels,height,width,first,rows,Buffer.length(~Element,left),left_metadata),
      Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,piece),Buffer.content_tree(~Element,Buffer.content(~Element,piece)),
        piece_tree(~Element,Views.View{copied,offset},parts,source_tree,Tree.fresh(Element,Storage.depth_for(parts),fill)),
        Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,piece)),Buffer.tree(~Element,piece),Buffer.content_tree_of(~Element,piece)),
        Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,content => Buffer.content_tree(~Element,content),Buffer.content(~Element,piece),program,copied_content))},
    width_positive,height_positive,pixels)

def rows_found(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  view: Maybe<&2,Views.View>,  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +result: {Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}) == view : Maybe<&2,Views.View>},boxed: {Bands.has_view(view) == True{} : Bool},
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +image: ImageShape(shape,Bands.Geometry{images,channels,height,width}),
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  made: {Buffer.content(~Element,piece) ==
    part_views(~Element,Bands.compact_of(source_shape,Bands.rows_view(shape,height,first,rows,Bands.stretched_source(source_shape,shape))),
      Bands.contiguous_of(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),Bands.count_of_extents(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),
      fill,Buffer.content(~Element,source)) : Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),Bands.rows_extents(shape,rows)),left,piece):
  match view:
    case Some{Views.View{+band,+offset}}:
      rows_copied(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
        full,source_count,limits,fill,source,first,rows,left,piece,band,offset,result,source_ready,source_values,source_length,source_exact,stretched,nonempty,
        width_positive,height_positive,height_bound,pixels,image,left_ready,left_metadata,left_values,made)
    case None{}:
      Empty.absurd(SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
        Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.rows_extents(shape,rows))),Bands.rows_extents(shape,rows)),left,piece),Words.false_true(boxed))

# A band copies the part of its view: of the whole image, or of its image
# rows of an image shape. Every band of a boxed plan has one of these views.
def copied_when(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  whole: Bool,rows_shaped: Bool,  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  +whole_value: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == whole : Bool},
  +shaped_value: {Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}) == rows_shaped : Bool},
  boxed: {Bands.has_view(Bands.band_view_when(whole,rows_shaped,shape,height,first,rows,Bands.stretched_source(source_shape,shape))) == True{} : Bool},
  made: {Buffer.content(~Element,piece) ==
    part_views(~Element,Bands.compact_of(source_shape,Bands.band_view_when(whole,rows_shaped,shape,height,first,rows,Bands.stretched_source(source_shape,shape))),
      Bands.contiguous_of(Bands.compact_with(source_shape,Bands.band_extents_when(whole,rows_shaped,shape,rows))),
      Bands.count_of_extents(Bands.compact_with(source_shape,Bands.band_extents_when(whole,rows_shaped,shape,rows))),fill,Buffer.content(~Element,source)) :
    Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.band_extents_when(whole,rows_shaped,shape,rows))),
      Bands.band_extents_when(whole,rows_shaped,shape,rows)),left,piece):
  match whole rows_shaped:
    case True{} rows_shaped:
      whole_copied(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
        full,source_count,limits,fill,source,first,rows,left,piece,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
        width_positive,height_positive,pixels,whole_value,left_ready,left_metadata,left_values,made)
    case False{} True{}:
      rows_found(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Bands.rows_view(shape,height,first,rows,Some{Views.View{full,0}}),
        left_context,right_context,images,channels,height,width,shape,source_shape,full,source_count,limits,fill,source,first,rows,left,piece,{==},
        %source_view(source_shape,source_count,shape,full,source_exact,stretched) : {Bands.has_view(Bands.rows_view(shape,height,first,rows,_)) == True{} : Bool} boxed,
        source_ready,source_values,source_length,source_exact,stretched,nonempty,width_positive,height_positive,height_bound,pixels,
        image_shape(shape,Bands.Geometry{images,channels,height,width},shaped_value),left_ready,left_metadata,left_values,made)
    case False{} False{}:
      Empty.absurd(SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
        Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.band_extents_when(False{},False{},shape,rows))),
          Bands.band_extents_when(False{},False{},shape,rows)),left,piece),Words.false_true(boxed))

def taken_when(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  whole: Bool,rows_shaped: Bool,  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  +whole_value: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == whole : Bool},
  taken: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == True{} : Bool},
  made: {Buffer.content(~Element,piece) == Buffer.content(~Element,source) : Buffer.Content<Element>}) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.band_extents_when(whole,rows_shaped,shape,rows))),
      Bands.band_extents_when(whole,rows_shaped,shape,rows)),left,piece):
  match whole:
    case True{}:
      whole_taken(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
        full,source_count,limits,source,first,rows,left,piece,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
        width_positive,height_positive,pixels,whole_value,left_ready,left_metadata,left_values,made)
    case False{}:
      Empty.absurd(SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
        Bands.stretched_to(Bands.contiguous_of(Bands.compact_with(source_shape,Bands.band_extents_when(False{},rows_shaped,shape,rows))),
          Bands.band_extents_when(False{},rows_shaped,shape,rows)),left,piece),
        Words.false_true(Equal.trans(Bool,False{},Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows),True{},
          Equal.sym(Bool,Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows),False{},whole_value),taken)))

# The part a band reads: the source itself for a band of a whole image that
# takes it as is, or its copy.
type Piece<-Element: Data,-context: Bands.Spread,-fill: Element,-source: Buffer.Model<Element>,-first: U32,-rows: U32,-piece: Buffer.Model<Element>> is Data:
  Taken{whole: {Bands.whole_image(Bands.spread_geometry(context),first,rows) == True{} : Bool},
    made: {Buffer.content(~Element,piece) == Buffer.content(~Element,source) : Buffer.Content<Element>}}
  Copied{made: {Buffer.content(~Element,piece) == part(~Element,Bands.Pieces{context,fill},Buffer.content(~Element,source),first,rows) : Buffer.Content<Element>}}

# A band of a boxed plan combined with the part it reads holds the combined
# values of its cells.
def spread_leaf(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,
  +left: Buffer.Model<Element>,+piece: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  boxed: {Bands.has_view(Bands.band_view(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},first,rows)) == True{} : Bool},
  kind: Piece<Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,source,first,rows,piece>) ->
  SpreadLeaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,left_context,right_context,full,images,channels,height,width,first,rows,
    Bands.stretched_to(Bands.contiguous_of(Bands.band_compact(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},first,rows)),
      Bands.band_extents(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},first,rows)),left,piece):
  match kind:
    case Taken{whole,made}:
      taken_when(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows),Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}),
        left_context,right_context,images,channels,height,width,shape,source_shape,full,source_count,limits,source,first,rows,left,piece,source_ready,source_values,
        source_length,source_exact,stretched,nonempty,coherent,width_positive,height_positive,pixels,left_ready,left_metadata,left_values,{==},whole,made)
    case Copied{made}:
      copied_when(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows),Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}),
        left_context,right_context,images,channels,height,width,shape,source_shape,full,source_count,limits,fill,source,first,rows,left,piece,source_ready,source_values,
        source_length,source_exact,stretched,nonempty,coherent,width_positive,height_positive,height_bound,pixels,left_ready,left_metadata,left_values,{==},{==},boxed,made)

# The bands spread makes from a source for a plan: one band of a whole image
# takes the source as is, every other band its part.
def taken_or_part(~Element: Data,whole: Bool,+context: Bands.Spread,+fill: Element,+source: Buffer.Content<Element>,+first: U32,+rows: U32) -> Model.Content<Element>:
  match whole:
    case True{}: Model.RowBand{source,first,rows}
    case False{}: Model.RowBand{part(~Element,Bands.Pieces{context,fill},source,first,rows),first,rows}

def spread_content(~Element: Data,+context: Bands.Spread,+fill: Element,+source: Buffer.Content<Element>,layout: Bands.Plan) -> Model.Content<Element>:
  match layout:
    case Bands.Rows{+first,+rows}: taken_or_part(~Element,Bands.whole_image(Bands.spread_geometry(context),first,rows),context,fill,source,first,rows)
    case Bands.Split{left,right}: parts(~Element,context,fill,source,Bands.Split{left,right})

def leaf_spread(~Element: Data,whole: Bool,+context: Bands.Spread,+fill: Element,+first: U32,+rows: U32,+source: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.spread_leaf(~Element,whole,context,fill,first,rows,Buffer.pack(~Element,source))) ==
    taken_or_part(~Element,whole,context,fill,Buffer.content(~Element,source),first,rows) : Model.Content<Element>}:
  match whole:
    case True{}:
      Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.RowBand{buffer,first,rows},Buffer.contents(~Element,Buffer.pack(~Element,source)),
        Buffer.content(~Element,source),Buffer.packed_contents(~Element,source))
    case False{}:
      %Buffer.packed_contents(~Element,source) :
        {Model.contents(~Element,Bands.result(~Element,Bands.piece_band(~Element,Bands.Pieces{context,fill},first,rows,Buffer.pack(~Element,source)))) ==
          Model.RowBand{part(~Element,Bands.Pieces{context,fill},_,first,rows),first,rows} : Model.Content<Element>}
      Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.piece_band(~Element,Bands.Pieces{context,fill},first,rows,Buffer.pack(~Element,source)))),
        parts_of(~Element,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
          Bands.piece_band(~Element,Bands.Pieces{context,fill},first,rows,Buffer.pack(~Element,source)))),
        Model.RowBand{part(~Element,Bands.Pieces{context,fill},Buffer.contents(~Element,Buffer.pack(~Element,source)),first,rows),first,rows},
        result_parts(~Element,Bands.piece_band(~Element,Bands.Pieces{context,fill},first,rows,Buffer.pack(~Element,source))),
        Equal.cong(Buffer.Content<Element> & Model.Content<Element>,Model.Content<Element>,seen => parts_of(~Element,seen),
          Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),
            Bands.piece_band(~Element,Bands.Pieces{context,fill},first,rows,Buffer.pack(~Element,source))),
          (Buffer.contents(~Element,Buffer.pack(~Element,source)),
            Model.RowBand{part(~Element,Bands.Pieces{context,fill},Buffer.contents(~Element,Buffer.pack(~Element,source)),first,rows),first,rows}),
          part_rule(~Element,Bands.Pieces{context,fill},first,rows,source)))

def spread_contents(~Element: Data,+context: Bands.Spread,+fill: Element,+layout: Bands.Plan,+source: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.spread(~Element,context,fill,layout,Buffer.pack(~Element,source))) ==
    spread_content(~Element,context,fill,Buffer.content(~Element,source),layout) : Model.Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_spread(~Element,Bands.whole_image(Bands.spread_geometry(context),first,rows),context,fill,first,rows,source)
    case Bands.Split{+left,+right}: built_parts(~Element,context,fill,source,Bands.Split{left,right})

# A band and its part, combined.
def spread_band(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),context: Bands.Spread,first: U32,rows: U32,left: Buffer.Model<Element>,right: Buffer.Model<Element>) ->
  Model.Content<Element>:
  +context = context
  +first = first
  +rows = rows
  Model.RowBand{Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.spread_geometry(context),rows),
    Bands.stretched_to(Bands.contiguous_of(Bands.band_compact(context,first,rows)),Bands.band_extents(context,first,rows)),Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    first,rows}

def spread_band_rule(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+context: Bands.Spread,+first: U32,+rows: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.combine_spread_band_with(~Element,~kernel,context,first,rows,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) ==
    spread_band(~Element,~kernel,context,first,rows,left,right) : Model.Content<Element>}:
  {==}

# The combined bands keep the layout of the left bands.
def spread_layout(~Element: Data,~kernel: OperationsLaws.Kernel(~Element),+context: Bands.Spread,left: Model.Model<Element>,right: Model.Model<Element>) ->
  {Model.content_layout(~Element,Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,left,right)) == Model.layout(~Element,left) : Bands.Plan}:
  match left right:
    case Model.Leaf{buffer,first,rows} Model.Leaf{other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,left,other_left)),Model.layout(~Element,left),
        spread_layout(~Element,~kernel,context,left,other_left)) :
        {Bands.Split{_,Model.content_layout(~Element,Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,right,other_right))} ==
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,right,other_right)),Model.layout(~Element,right),
        spread_layout(~Element,~kernel,context,right,other_right)) :
        {Bands.Split{Model.layout(~Element,left),_} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{other_left,other_right}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Leaf{buffer,first,rows}),Model.content_layout(~Element,Model.content(~Element,Model.Leaf{buffer,first,rows})),
        Model.layout_matches_content(~Element,Model.Leaf{buffer,first,rows}))
    case Model.Fork{+left,+right} Model.Leaf{buffer,first,rows}:
      Equal.sym(Bands.Plan,Model.layout(~Element,Model.Fork{left,right}),Model.content_layout(~Element,Model.content(~Element,Model.Fork{left,right})),
        Model.layout_matches_content(~Element,Model.Fork{left,right}))

# What the bands of a content are: one band, its buffer, or its two halves.
def is_band(~Element: Data,content: Model.Content<Element>) -> Bool:
  match content:
    case Model.RowBand{buffer,first,rows}: True{}
    case Model.Joined{left,right}: False{}

def band_buffer(~Element: Data,content: Model.Content<Element>) -> Buffer.Content<Element>:
  match content:
    case Model.RowBand{buffer,first,rows}: buffer
    case Model.Joined{left,right}: band_buffer(~Element,left)

def left_half(~Element: Data,content: Model.Content<Element>) -> Model.Content<Element>:
  match content:
    case Model.RowBand{buffer,first,rows}: Model.RowBand{buffer,first,rows}
    case Model.Joined{left,right}: left

def right_half(~Element: Data,content: Model.Content<Element>) -> Model.Content<Element>:
  match content:
    case Model.RowBand{buffer,first,rows}: Model.RowBand{buffer,first,rows}
    case Model.Joined{left,right}: right

def taken_band(~Element: Data,whole: Bool,+context: Bands.Spread,+fill: Element,+source: Buffer.Content<Element>,+first: U32,+rows: U32) ->
  {is_band(~Element,taken_or_part(~Element,whole,context,fill,source,first,rows)) == True{} : Bool}:
  match whole:
    case True{}: {==}
    case False{}: {==}

# What the combined bands hold: the combined values, ready, with the metadata
# of the geometry.
def Combined(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +left_context: Left,+right_context: Right,+full: List<&2,Views.Axis>,+geometry: Bands.Geometry,+content: Model.Content<Element>) -> Data:
  BandValues.PipelineContract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,content)

# Every band of a plan of copied parts, combined with its part.
def pipeline_parts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  left: Model.Model<Element>,right: Model.Model<Element>,
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  left_metadata: BandShape.Contents(~Element,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),left_ready: Model.Ready(~Element,left),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),
  +boxed: {Bands.spread_boxed(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},Model.layout(~Element,left)) == True{} : Bool},
  +made: {Model.content(~Element,right) ==
    parts(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),Model.layout(~Element,left)) :
    Model.Content<Element>}) ->
  Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
    Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},left,right)):
  match left right:
    case Model.Leaf{Buffer.Model{+left_tree,+left_length,+left_certificate},+first,+rows} Model.Leaf{+piece,+other_first,+other_rows}:
      +left = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
      BandCombine.leaf_contract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
        CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},first,rows,
        Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),
          Bands.stretched_to(Bands.contiguous_of(Bands.band_compact(context,first,rows)),Bands.band_extents(context,first,rows)),Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),
        spread_leaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
          full,source_count,limits,fill,source,first,rows,left,piece,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
          width_positive,height_positive,height_bound,pixels,left_ready,left_metadata,left_values,boxed,
          Copied{Equal.cong(Model.Content<Element>,Buffer.Content<Element>,content => band_buffer(~Element,content),Model.content(~Element,Model.Leaf{piece,other_first,other_rows}),
            parts(~Element,context,fill,Buffer.content(~Element,source),Bands.Rows{first,rows}),made)}))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_left_metadata,left_right_metadata} = left_metadata
      Shape.Both{left_left_ready,left_right_ready} = left_ready
      Shape.Both{left_left_values,left_right_values} = left_values
      +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
      +whole = {Model.Joined{Model.content(~Element,other_left),Model.content(~Element,other_right)} : Model.Content<Element>}
      +planned = parts(~Element,context,fill,Buffer.content(~Element,source),Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)})
      BandValues.pipeline_joined(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
        CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},
        Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,left,other_left),
        Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,right,other_right),
        pipeline_parts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,other_left,left_context,right_context,images,channels,height,width,
          shape,source_shape,full,source_count,limits,fill,source,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
          width_positive,height_positive,height_bound,pixels,left_left_metadata,left_left_ready,left_left_values,
          Boolean.left(Bands.spread_boxed(context,Model.layout(~Element,left)),Bands.spread_boxed(context,Model.layout(~Element,right)),boxed),
          Equal.cong(Model.Content<Element>,Model.Content<Element>,content => left_half(~Element,content),whole,planned,made)),
        pipeline_parts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,right,other_right,left_context,right_context,images,channels,height,width,
          shape,source_shape,full,source_count,limits,fill,source,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
          width_positive,height_positive,height_bound,pixels,left_right_metadata,left_right_ready,left_right_values,
          Boolean.right(Bands.spread_boxed(context,Model.layout(~Element,left)),Bands.spread_boxed(context,Model.layout(~Element,right)),boxed),
          Equal.cong(Model.Content<Element>,Model.Content<Element>,content => right_half(~Element,content),whole,planned,made)))
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd(Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
          Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},
            Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})),
        Words.false_true(Equal.cong(Model.Content<Element>,Bool,content => is_band(~Element,content),
          Model.Joined{Model.content(~Element,other_left),Model.content(~Element,other_right)},
          parts(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),Bands.Rows{first,rows}),made)))
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd(Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
          Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},
            Model.Fork{left,right},Model.Leaf{buffer,first,rows})),
        Words.false_true(Equal.sym(Bool,True{},False{},Equal.cong(Model.Content<Element>,Bool,content => is_band(~Element,content),
          Model.RowBand{Buffer.content(~Element,buffer),first,rows},
          parts(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),
            Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)}),made))))

# One band of the plan: it takes the source as is when it holds the whole
# image, its part otherwise.
def top_leaf(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  whole: Bool,right: Model.Model<Element>,
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,+first: U32,+rows: U32,+left: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_ready: Buffer.Sized<Element,left>,+left_metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,width},first,rows,Buffer.length(~Element,left)),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.RowBand{Buffer.content(~Element,left),first,rows}),
  boxed: {Bands.has_view(Bands.band_view(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},first,rows)) == True{} : Bool},
  whole_value: {Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows) == whole : Bool},
  +made: {Model.content(~Element,right) ==
    taken_or_part(~Element,whole,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),first,rows) :
    Model.Content<Element>}) ->
  Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
    Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},
      Model.Leaf{left,first,rows},right)):
  match whole right:
    case True{} Model.Leaf{+piece,+other_first,+other_rows}:
      +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
      BandCombine.leaf_contract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
        CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},first,rows,
        Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),
          Bands.stretched_to(Bands.contiguous_of(Bands.band_compact(context,first,rows)),Bands.band_extents(context,first,rows)),Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),
        spread_leaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
          full,source_count,limits,fill,source,first,rows,left,piece,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
          width_positive,height_positive,height_bound,pixels,left_ready,left_metadata,left_values,boxed,
          Taken{whole_value,Equal.cong(Model.Content<Element>,Buffer.Content<Element>,content => band_buffer(~Element,content),
            Model.content(~Element,Model.Leaf{piece,other_first,other_rows}),Model.RowBand{Buffer.content(~Element,source),first,rows},made)}))
    case False{} Model.Leaf{+piece,+other_first,+other_rows}:
      +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
      BandCombine.leaf_contract(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
        CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},first,rows,
        Buffer.contents(~Element,Bands.spread_views(~Element,~kernel,Bands.local_count(Bands.Geometry{images,channels,height,width},rows),
          Bands.stretched_to(Bands.contiguous_of(Bands.band_compact(context,first,rows)),Bands.band_extents(context,first,rows)),Buffer.pack(~Element,left),Buffer.pack(~Element,piece))),
        spread_leaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left_context,right_context,images,channels,height,width,shape,source_shape,
          full,source_count,limits,fill,source,first,rows,left,piece,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
          width_positive,height_positive,height_bound,pixels,left_ready,left_metadata,left_values,boxed,
          Copied{Equal.cong(Model.Content<Element>,Buffer.Content<Element>,content => band_buffer(~Element,content),
            Model.content(~Element,Model.Leaf{piece,other_first,other_rows}),
            Model.RowBand{part(~Element,Bands.Pieces{context,fill},Buffer.content(~Element,source),first,rows),first,rows},made)}))
    case whole Model.Fork{+other_left,+other_right}:
      +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
      +taken = taken_or_part(~Element,whole,context,fill,Buffer.content(~Element,source),first,rows)
      Empty.absurd(Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
          Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,Model.Leaf{left,first,rows},Model.Fork{other_left,other_right})),
        Words.false_true(Equal.trans(Bool,False{},is_band(~Element,taken),True{},
          Equal.cong(Model.Content<Element>,Bool,content => is_band(~Element,content),Model.Joined{Model.content(~Element,other_left),Model.content(~Element,other_right)},
            taken,made),
          taken_band(~Element,whole,context,fill,Buffer.content(~Element,source),first,rows))))

# The bands of a boxed plan, combined with the bands spread makes for them.
def spread_pipeline(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  left: Model.Model<Element>,right: Model.Model<Element>,
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  left_metadata: BandShape.Contents(~Element,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),left_ready: Model.Ready(~Element,left),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),
  boxed: {Bands.spread_boxed(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},Model.layout(~Element,left)) == True{} : Bool},
  made: {Model.content(~Element,right) ==
    spread_content(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),Model.layout(~Element,left)) :
    Model.Content<Element>}) ->
  Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,Bands.Geometry{images,channels,height,width},
    Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},left,right)):
  match left:
    case Model.Leaf{Buffer.Model{+left_tree,+left_length,+left_certificate},+first,+rows}:
      top_leaf(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Bands.whole_image(Bands.Geometry{images,channels,height,width},first,rows),right,
        left_context,right_context,images,channels,height,width,shape,source_shape,full,source_count,limits,fill,source,first,rows,
        Buffer.Model{left_tree,left_length,left_certificate},source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
        width_positive,height_positive,height_bound,pixels,left_ready,left_metadata,left_values,boxed,{==},made)
    case Model.Fork{+before,+after}:
      pipeline_parts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,Model.Fork{before,after},right,left_context,right_context,
        images,channels,height,width,shape,source_shape,full,source_count,limits,fill,source,source_ready,source_values,source_length,source_exact,stretched,nonempty,
        coherent,width_positive,height_positive,height_bound,pixels,left_metadata,left_ready,left_values,boxed,made)

def spread_observed(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +parallel: Bool,+left: Model.Model<Element>,-right: Bands.Bands(~Element),observation: Model.Observation<Element,right>,
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,
  contents: {Model.contents(~Element,right) ==
    spread_content(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Buffer.content(~Element,source),Model.layout(~Element,left)) :
    Model.Content<Element>},
  +source_ready: Buffer.Sized<Element,source>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  left_metadata: BandShape.Contents(~Element,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),left_ready: Model.Ready(~Element,left),
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),
  coverage: BandPlan.Coverage(Model.layout(~Element,left),0n,Inspection.total_rows(Bands.Geometry{images,channels,height,width})),
  boxed: {Bands.spread_boxed(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},Model.layout(~Element,left)) == True{} : Bool}) ->
  Shape.Both<BandValues.Values(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},
      Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},parallel,
        Model.pack(~Element,left),right))),
    BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},
      Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},parallel,
        Model.pack(~Element,left),right))>>:
  Model.Observation{+model,+equation} = observation
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +context = {Bands.Spread{geometry,shape,source_shape} : Bands.Spread}
  +combined = Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,context,parallel,Model.pack(~Element,left),Model.pack(~Element,model)))
  +zipped = Each.content_zip(~Element,~Bands.Spread,~spread_band(~Element,~kernel),context,left,model)
  +zip_equal = {
    %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
      {Model.contents(~Element,Bands.combine_spread_planned_with(~Element,~kernel,context,parallel,_,Model.pack(~Element,model))) == zipped : Model.Content<Element>}
    Each.zipped(~Element,~Bands.Spread,~Bands.combine_spread_band_with(~Element,~kernel),~spread_band(~Element,~kernel),~spread_band_rule(~Element,~kernel),
      left,model,parallel,context) : {combined == zipped : Model.Content<Element>}}
  +made = Equal.trans(Model.Content<Element>,Model.content(~Element,model),Model.contents(~Element,right),
    spread_content(~Element,context,fill,Buffer.content(~Element,source),Model.layout(~Element,left)),
    Equal.sym(Model.Content<Element>,Model.contents(~Element,right),Model.content(~Element,model),Model.observed_contents(~Element,model,right,equation)),contents)
  %equation :
    Shape.Both<BandValues.Values(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
        CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,
        Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,context,parallel,Model.pack(~Element,left),_))),
      BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,context,parallel,Model.pack(~Element,left),_))>>
  BandValues.pipeline_well_formed(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
    CombineKernel.Combine{left_context,right_context,Views.View{full,0}},geometry,combined,
    %Equal.sym(Model.Content<Element>,combined,zipped,zip_equal) :
      Combined(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,full,geometry,_)
    spread_pipeline(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,model,left_context,right_context,images,channels,height,width,
      shape,source_shape,full,source_count,limits,fill,source,source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,
      width_positive,height_positive,height_bound,pixels,left_metadata,left_ready,left_values,boxed,made),
    %Equal.sym(Model.Content<Element>,combined,zipped,zip_equal) : BandPlan.Coverage(Model.content_layout(~Element,_),0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,zipped),Model.layout(~Element,left),spread_layout(~Element,~kernel,context,left,model)) :
      BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage)

# The bands of a boxed plan combined with the parts spread makes for them
# from a source stretched to the shape: at every cell the operation of the
# left value and the source value at the address of the pixel in the
# stretched source, in well formed bands of the geometry.
def spread_combined(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  ~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  +parallel: Bool,+left: Model.Model<Element>,
  +left_context: Left,+right_context: Right,+images: U32,+channels: U32,+height: U32,+width: U32,+shape: List<&2,U32>,+source_shape: List<&2,U32>,
  +full: List<&2,Views.Axis>,+source_count: U32,+limits: Machine.Limits,+fill: Element,+source: Buffer.Model<Element>,
  +source_ready: Buffer.Sized<Element,source>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  +source_length: {Buffer.length(~Element,source) == U32.to_nat(source_count) : Nat},+source_exact: {ShapeLaws.elements(source_shape) == U32.to_nat(source_count) : Nat},
  +stretched: {Views.stretch_axes(source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +coherent: {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  left_shape: BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)>,
  left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),
  boxed: {Bands.spread_boxed(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},Model.layout(~Element,left)) == True{} : Bool}) ->
  Shape.Both<BandValues.Values(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,Views.View{full,0}},Bands.Geometry{images,channels,height,width},
      Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},parallel,
        Model.pack(~Element,left),Bands.spread(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Model.layout(~Element,left),
          Buffer.pack(~Element,source))))),
    BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},
      Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},parallel,
        Model.pack(~Element,left),Bands.spread(~Element,Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},fill,Model.layout(~Element,left),
          Buffer.pack(~Element,source))))>>:
  BandShape.WellFormed{left_metadata,left_ready,coverage} = left_shape
  +context = {Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape} : Bands.Spread}
  spread_observed(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,parallel,left,
    Bands.spread(~Element,context,fill,Model.layout(~Element,left),Buffer.pack(~Element,source)),
    Model.observe(~Element,Bands.spread(~Element,context,fill,Model.layout(~Element,left),Buffer.pack(~Element,source))),
    left_context,right_context,images,channels,height,width,shape,source_shape,full,source_count,limits,fill,source,
    spread_contents(~Element,context,fill,Model.layout(~Element,left),source),
    source_ready,source_values,source_length,source_exact,stretched,nonempty,coherent,width_positive,height_positive,height_bound,pixels,
    left_metadata,Model.ready(~Element,left,left_ready),left_values,
    %Equal.sym(Bands.Plan,Model.layout(~Element,left),Model.content_layout(~Element,Model.content(~Element,left)),Model.layout_matches_content(~Element,left)) :
      BandPlan.Coverage(_,0n,Inspection.total_rows(Bands.Geometry{images,channels,height,width}))
    coverage,boxed)
