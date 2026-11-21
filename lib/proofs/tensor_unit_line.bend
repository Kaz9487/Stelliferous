# One reduced axis of a shape: the reduced axes are that axis, and the kept
# extents hold as many cells as the shape with that axis of extent 1, so a
# reduction over the axis reshapes to that unit shape.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_layout.bend as TiledLayout
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../kernel_gemm.bend as KernelGemm
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_reduce_values.bend as Reduce
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_global_shape_count.bend as Global
import ./tensor_relabeled.bend as Relabeled
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_reshape.bend as TiledReshape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor

def unit_cells(+shape: List<&2,U32>,+axis: Nat,inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Nat.mul(ShapeLaws.elements(Layout.with_extent(shape,axis,1)),U32.to_nat(Layout.extent_at(shape,axis))) == ShapeLaws.elements(shape) : Nat}:
  match shape axis:
    case [] axis:
      Empty.absurd({Nat.mul(ShapeLaws.elements(Layout.with_extent([],axis,1)),U32.to_nat(Layout.extent_at([],axis))) == ShapeLaws.elements([]) : Nat},Order.below_zero(axis,inside))
    case +head <> +tail 0n:
      %Equal.sym(Nat,Nat.add(ShapeLaws.elements(tail),0n),ShapeLaws.elements(tail),Addition.zero_right(ShapeLaws.elements(tail))) :
        {Nat.mul(_,U32.to_nat(head)) == Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail)) : Nat}
      RowMajor.multiply_commutative(ShapeLaws.elements(tail),U32.to_nat(head))
    case +head <> +tail 1n+ +rest:
      Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(Layout.with_extent(tail,rest,1))),U32.to_nat(Layout.extent_at(tail,rest))),
        Nat.mul(U32.to_nat(head),Nat.mul(ShapeLaws.elements(Layout.with_extent(tail,rest,1)),U32.to_nat(Layout.extent_at(tail,rest)))),
        Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail)),
        RowMajor.assoc(U32.to_nat(head),ShapeLaws.elements(Layout.with_extent(tail,rest,1)),U32.to_nat(Layout.extent_at(tail,rest))),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(head),value),Nat.mul(ShapeLaws.elements(Layout.with_extent(tail,rest,1)),U32.to_nat(Layout.extent_at(tail,rest))),
          ShapeLaws.elements(tail),unit_cells(tail,rest,inside)))

# No axis from past it on is the axis.
def single_after(remaining: Nat,+index: Nat,+axis: Nat,+below: {Nat.is_lt(axis,index) == True{} : Bool}) -> {Layout.axes_where(remaining,index,[axis],True{}) == [] : List<&2,Nat>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Bool,Nat.is_eq(axis,index),False{},Order.less_not_equal(axis,index,below)) :
        {Layout.chosen(_ || False{},True{},index,Layout.axes_where(rest,1n+index,[axis],True{})) == [] : List<&2,Nat>}
      single_after(rest,1n+index,axis,Order.lt_succ(axis,index,Bounds.strict_le(axis,index,below)))

# The axis index + count is the one reduced axis of the axes from index on.
def single_at(count: Nat,+index: Nat,+rest: Nat) -> {Layout.axes_where(Nat.add(count,1n+rest),index,[Nat.add(index,count)],True{}) == [Nat.add(index,count)] : List<&2,Nat>}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {Layout.axes_where(1n+rest,index,[_],True{}) == [_] : List<&2,Nat>}
      %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) :
        {Layout.chosen(_ || False{},True{},index,Layout.axes_where(rest,1n+index,[index],True{})) == [index] : List<&2,Nat>}
      Equal.cong(List<&2,Nat>,List<&2,Nat>,items => index <> items,Layout.axes_where(rest,1n+index,[index],True{}),[],
        single_after(rest,1n+index,index,Order.lt_succ(index,index,Order.reflexive(index))))
    case 1n+ +previous:
      +apart = Equal.trans(Bool,Nat.is_eq(Nat.add(index,1n+previous),index),Nat.is_eq(index,Nat.add(index,1n+previous)),False{},
        Order.equal_symmetric(Nat.add(index,1n+previous),index),Order.less_not_equal(index,Nat.add(index,1n+previous),Order.below_sum(index,previous)))
      %Equal.sym(Bool,Nat.is_eq(Nat.add(index,1n+previous),index),False{},apart) :
        {Layout.chosen(_ || False{},True{},index,Layout.axes_where(Nat.add(previous,1n+rest),1n+index,[Nat.add(index,1n+previous)],True{})) == [Nat.add(index,1n+previous)] : List<&2,Nat>}
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {Layout.axes_where(Nat.add(previous,1n+rest),1n+index,[_],True{}) == [_] : List<&2,Nat>}
      single_at(previous,1n+index,rest)

# An axis below the rank leaves rank - axis - 1 axes after it.
def axis_split(+axis: Nat,+rank: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool}) -> {Nat.add(axis,1n+Nat.sub(rank,1n+axis)) == rank : Nat}:
  Equal.trans(Nat,Nat.add(axis,1n+Nat.sub(rank,1n+axis)),1n+Nat.add(axis,Nat.sub(rank,1n+axis)),rank,
    Addition.successor_right(axis,Nat.sub(rank,1n+axis)),Order.restore_shift(1n+axis,rank,Order.strict_follow(axis,rank,inside)))

# The one reduced axis.
def axis_reduced(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Layout.axes_where(List.length(&2,U32,shape),0n,[axis],True{}) == [axis] : List<&2,Nat>}:
  %axis_split(axis,List.length(&2,U32,shape),inside) : {Layout.axes_where(_,0n,[axis],True{}) == [axis] : List<&2,Nat>}
  single_at(axis,0n,Nat.sub(List.length(&2,U32,shape),1n+axis))

# The extents kept by a reduction over the axis.
def Kept(+shape: List<&2,U32>,+axis: Nat) -> List<&2,U32>:
  Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,[axis],False{}))

# The kept extents times the extent of the axis are the cells of the shape.
def kept_cells(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Nat.mul(ShapeLaws.elements(Kept(shape,axis)),U32.to_nat(Layout.extent_at(shape,axis))) == ShapeLaws.elements(shape) : Nat}:
  %Order.times_one(U32.to_nat(Layout.extent_at(shape,axis))) : {Nat.mul(ShapeLaws.elements(Kept(shape,axis)),_) == ShapeLaws.elements(shape) : Nat}
  %axis_reduced(shape,axis,inside) : {Nat.mul(ShapeLaws.elements(Kept(shape,axis)),ShapeLaws.elements(Layout.extents_at(shape,_))) == ShapeLaws.elements(shape) : Nat}
  %ShapeLaws.drop_zero(shape) : {Nat.mul(ShapeLaws.elements(Kept(shape,axis)),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,[axis],True{})))) == ShapeLaws.elements(_) : Nat}
  Reduce.partition_elements(shape,List.length(&2,U32,shape),0n,[axis],{==})

# The axis of a nonempty shape holds cells.
def axis_positive(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Layout.extent_at(shape,axis))) == True{} : Bool}:
  +kept = ShapeLaws.elements(Kept(shape,axis))
  +extent = U32.to_nat(Layout.extent_at(shape,axis))
  RowMajor.left_factor_positive(extent,kept,
    %RowMajor.multiply_commutative(kept,extent) : {Nat.is_lt(0n,_) == True{} : Bool}
      %Equal.sym(Nat,Nat.mul(kept,extent),ShapeLaws.elements(shape),kept_cells(shape,axis,inside)) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def kept_positive(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_lt(0n,ShapeLaws.elements(Kept(shape,axis))) == True{} : Bool}:
  RowMajor.left_factor_positive(ShapeLaws.elements(Kept(shape,axis)),U32.to_nat(Layout.extent_at(shape,axis)),
    %Equal.sym(Nat,Nat.mul(ShapeLaws.elements(Kept(shape,axis)),U32.to_nat(Layout.extent_at(shape,axis))),ShapeLaws.elements(shape),kept_cells(shape,axis,inside)) :
      {Nat.is_lt(0n,_) == True{} : Bool} positive)

# The kept extents hold as many cells as the shape with the axis of extent 1.
def kept_unit(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {ShapeLaws.elements(Kept(shape,axis)) == ShapeLaws.elements(Layout.with_extent(shape,axis,1)) : Nat}:
  +kept = ShapeLaws.elements(Kept(shape,axis))
  +unit = ShapeLaws.elements(Layout.with_extent(shape,axis,1))
  +extent = U32.to_nat(Layout.extent_at(shape,axis))
  +same = Equal.trans(Nat,Nat.mul(kept,extent),ShapeLaws.elements(shape),Nat.mul(unit,extent),kept_cells(shape,axis,inside),
    Equal.sym(Nat,Nat.mul(unit,extent),ShapeLaws.elements(shape),unit_cells(shape,axis,inside)))
  +nonzero = axis_positive(shape,axis,inside,positive)
  Order.antisymmetric(kept,unit,
    Ceiling.cancel_product_le(kept,unit,extent,nonzero,%same : {Nat.is_le(Nat.mul(kept,extent),_) == True{} : Bool} Order.reflexive(Nat.mul(kept,extent))),
    Ceiling.cancel_product_le(unit,kept,extent,nonzero,%same : {Nat.is_le(_,Nat.mul(kept,extent)) == True{} : Bool} Order.reflexive(Nat.mul(kept,extent))))

# The kept extents have no more axes than the shape.
def kept_rank(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Nat.is_le(List.length(&2,U32,Kept(shape,axis)),List.length(&2,U32,shape)) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  +kept = List.length(&2,Nat,Layout.axes_where(rank,0n,[axis],False{}))
  +parts = {%axis_reduced(shape,axis,inside) : {Nat.add(kept,List.length(&2,Nat,_)) == rank : Nat} Reduce.partition_length(rank,0n,[axis]) : {Nat.add(kept,1n) == rank : Nat}}
  %Equal.sym(Nat,List.length(&2,U32,Kept(shape,axis)),kept,Reduce.extents_length(shape,Layout.axes_where(rank,0n,[axis],False{}))) : {Nat.is_le(_,rank) == True{} : Bool}
  %parts : {Nat.is_le(kept,_) == True{} : Bool}
  Order.le_add(kept,1n)

def kept_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(List.length(&2,U32,Kept(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(List.length(&2,U32,Kept(shape,axis)),List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits)),kept_rank(shape,axis,inside),rank_room)

def kept_count(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+counted: Global.Counted<Kept(shape,axis)>) ->
  {Tensors.count_of(Layout.with_extent(shape,axis,1)) == Tensors.count_of(Kept(shape,axis)) : Maybe<&2,U32>}:
  Equal.trans(Maybe<&2,U32>,Tensors.count_of(Layout.with_extent(shape,axis,1)),Some{Global.count(Kept(shape,axis),counted)},Tensors.count_of(Kept(shape,axis)),
    Relabeled.counted_same(Kept(shape,axis),Layout.with_extent(shape,axis,1),counted,kept_unit(shape,axis,inside,positive)),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of(Kept(shape,axis)),Some{Global.count(Kept(shape,axis),counted)},Global.accepted(Kept(shape,axis),counted)))

def kept_nonzero(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Shapes.has_zero(Kept(shape,axis)) == False{} : Bool}:
  CountCompletion.positive_not_zero(Kept(shape,axis),kept_positive(shape,axis,inside,positive))

def unit_nonzero(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Shapes.has_zero(Layout.with_extent(shape,axis,1)) == False{} : Bool}:
  CountCompletion.positive_not_zero(Layout.with_extent(shape,axis,1),
    %kept_unit(shape,axis,inside,positive) : {Nat.is_lt(0n,_) == True{} : Bool} kept_positive(shape,axis,inside,positive))

def unit_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(List.length(&2,U32,Layout.with_extent(shape,axis,1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,List.length(&2,U32,Layout.with_extent(shape,axis,1)),List.length(&2,U32,shape),UnitBroadcast.rank(shape,axis)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  rank_room

# The values kept by a reduction over the axis, reshaped to the shape with the
# axis of extent 1.
def kept_reshaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,tiled: Tiled.Tiled,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,Kept(shape,axis),tiled)) ->
  Public.Contract(~Logical,~original,logical,Layout.with_extent(shape,axis,1),TiledLayout.reshape(tiled,Layout.with_extent(shape,axis,1))):
  match tiled:
    case Tiled.Tiled{cells,+lane,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+cell_shape,+fill,status,+workers}:
          match status:
            case Tensors.Usable{}:
              KernelGemm.Both{+global,Public.Laid{+within,+shaped,+restored,+counted,+pinned}} = contract
              TiledReshape.reshaped(~Logical,~original,logical,limits,Kept(shape,axis),Layout.with_extent(shape,axis,1),
                Tiled.Tiled{Tensors.Tensor{pieces,cell_shape,fill,Tensors.Usable{},workers},lane,lanes,position},kept_nonzero(shape,axis,inside,positive),
                kept_count(shape,axis,inside,positive,counted),kept_room(limits,shape,axis,inside,rank_room),unit_room(limits,shape,axis,rank_room),default_capacity,contract)
            case Tensors.Failed{+problem}:
              Empty.absurd(Public.Contract(~Logical,~original,logical,Layout.with_extent(shape,axis,1),
                TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{pieces,cell_shape,fill,Tensors.Failed{problem},workers},lane,lanes,position},Layout.with_extent(shape,axis,1))),contract)
