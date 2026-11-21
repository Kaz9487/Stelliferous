# The public tiled form of a tensor: cells of the logical shape with the lanes
# of one axis grouped eight a cell, the groups at a position, that hold the
# lifted values of a logical reader in the geometry the runtime reads them
# with. A failed tensor carries no guarantee.
import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./tensor_global_shape_count.bend as Global
import ./tensor_pieces_content.bend as Content
import ./tensor_tiled_representation.bend as Representation
import ./tensor_shape_refinement.bend as ShapeLaws
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as WordMultiplication
import ./word_encoding.bend as WordEncoding
import ./tensor_pieces_model.bend as PiecesModel
import ../tensor_pieces.bend as Pieces
import ../kernel_gemm.bend as KernelGemm
import ./kernel_tile_refinement.bend as TileLaws

# ---- The lane axis of a shape ----

# The product of the axes before the lane axis, its extent and the axes after
# it, as the runtime finds them.
def lane_prefix(shape: List<&2,U32>,axis: Nat,+outer: U32) -> U32:
  match shape axis:
    case Nil{} axis: outer
    case Con{extent,after} 0n: outer
    case Con{extent,after} 1n+rest: lane_prefix(after,rest,(outer * extent : U32))

def lane_after(shape: List<&2,U32>,axis: Nat) -> List<&2,U32>:
  match shape axis:
    case Nil{} axis: Nil{}
    case Con{extent,after} 0n: after
    case Con{extent,after} 1n+rest: lane_after(after,rest)

def located(+shape: List<&2,U32>,+axis: Nat,+outer: U32,+inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Tiled.around(shape,axis,outer) == Some{Tiled.Around{lane_prefix(shape,axis,outer),Grid.lanes_at(shape,axis),lane_after(shape,axis)}} : Maybe<&2,Tiled.Around>}:
  match shape axis:
    case Nil{} 0n: {==}
    case Nil{} 1n+rest: Empty.absurd({Tiled.around([],1n+rest,outer) == Some{Tiled.Around{outer,1,[]}} : Maybe<&2,Tiled.Around>},WordEncoding.false_true(inside))
    case Con{+extent,+after} 0n: {==}
    case Con{+extent,+after} 1n+ +rest: located(after,rest,(outer * extent : U32),inside)

# The lane axis of a rank lies inside it, or is the unit axis of rank 0.
def axis_within(+rank: Nat) -> {Nat.is_le(Tiled.lane_axis(rank),rank) == True{} : Bool}:
  match rank:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n+ +rest: %Order.commutative(rest,3n) : {Nat.is_le(rest,_) == True{} : Bool} Order.le_add(rest,3n)

def axis_picked(picked: Bool,+first: Nat,+second: Nat,+rank: Nat,+first_within: {Nat.is_le(first,rank) == True{} : Bool},+second_within: {Nat.is_le(second,rank) == True{} : Bool}) ->
  {Nat.is_le(Bool.pick(Nat,picked,first,second),rank) == True{} : Bool}:
  match picked:
    case True{}: first_within
    case False{}: second_within

# The lane axis a shape takes lies inside it.
def axis_for_within(+shape: List<&2,U32>) -> {Nat.is_le(Tiled.lane_axis_for(shape),List.length(&2,U32,shape)) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  +axis = Tiled.lane_axis(rank)
  axis_picked(U32.is_eq(Tensors.known_count(List.get(&2,U32,shape,axis)),1) && Nat.is_lt(1n,rank),Nat.sub(rank,1n),axis,rank,Order.sub_le(rank,1n),axis_within(rank))

# ---- The geometry of the values ----

# The geometry of the values of a shape around its lane axis, the groups at a
# position: the outer indices before and after the lane axis, the values
# after it, the lanes and their groups, as the runtime reads them back.
def geometry_around(axes: Maybe<&2,Tiled.Around>,+position: Grid.Position) -> Cells.Cells:
  match axes:
    case Some{Tiled.Around{+before,+lanes,after}}:
      +inner = Tiled.product(after)
      Cells.Cells{(before * inner : U32),inner,lanes,Tiled.groups(lanes),position}
    case None{}: Cells.Cells{0,1,1,1,position}

def geometry_of(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> Cells.Cells:
  geometry_around(Tiled.around(values,axis,1),position)

# The geometry of values whose lane axis lies inside them or just past them.
def geometry_exact(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {geometry_of(values,axis,position) == Cells.Cells{(lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32),Tiled.product(lane_after(values,axis)),
    Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position} : Cells.Cells}:
  Equal.cong(Maybe<&2,Tiled.Around>,Cells.Cells,axes => geometry_around(axes,position),Tiled.around(values,axis,1),
    Some{Tiled.Around{lane_prefix(values,axis,1),Grid.lanes_at(values,axis),lane_after(values,axis)}},located(values,axis,1,inside))

# ---- The cells of the values ----

# The other axes multiply alike before and after the lane axis.
def prefix_after(+shape: List<&2,U32>,+axis: Nat,+outer: U32) ->
  {(lane_prefix(shape,axis,outer) * Tiled.product(lane_after(shape,axis)) : U32) == (outer * Grid.product(Grid.removed(shape,axis)) : U32) : U32}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+after} 0n: {==}
    case Con{+extent,+after} 1n+ +rest:
      +others = Grid.product(Grid.removed(after,rest))
      Equal.trans(U32,(lane_prefix(after,rest,(outer * extent : U32)) * Tiled.product(lane_after(after,rest)) : U32),((outer * extent : U32) * others : U32),
        (outer * (extent * others : U32) : U32),prefix_after(after,rest,(outer * extent : U32)),Arithmetic.multiply_associative(outer,extent,others))

def one_times(+word: U32) -> {(1 * word : U32) == word : U32}:
  Equal.trans(U32,(1 * word : U32),(word * 1 : U32),word,Arithmetic.multiply_commutative(1,word),Arithmetic.multiply_one(word))

# Past the end of a shape the lane axis is the unit axis: one lane, nothing
# after it and every axis before it.
def lanes_past(+shape: List<&2,U32>,+axis: Nat,+past: {Nat.is_lt(axis,List.length(&2,U32,shape)) == False{} : Bool}) -> {Grid.lanes_at(shape,axis) == 1 : U32}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+after} 0n: Empty.absurd({extent == 1 : U32},WordEncoding.false_true(Equal.sym(Bool,True{},False{},past)))
    case Con{+extent,+after} 1n+ +rest: lanes_past(after,rest,past)

def after_past(+shape: List<&2,U32>,+axis: Nat,+past: {Nat.is_lt(axis,List.length(&2,U32,shape)) == False{} : Bool}) -> {lane_after(shape,axis) == Nil{} : List<&2,U32>}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+after} 0n: Empty.absurd({after == Nil{} : List<&2,U32>},WordEncoding.false_true(Equal.sym(Bool,True{},False{},past)))
    case Con{+extent,+after} 1n+ +rest: after_past(after,rest,past)

def removed_past(+shape: List<&2,U32>,+axis: Nat,+past: {Nat.is_lt(axis,List.length(&2,U32,shape)) == False{} : Bool}) -> {Grid.removed(shape,axis) == shape : List<&2,U32>}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+after} 0n: Empty.absurd({after == Con{extent,after} : List<&2,U32>},WordEncoding.false_true(Equal.sym(Bool,True{},False{},past)))
    case Con{+extent,+after} 1n+ +rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => extent <> others,Grid.removed(after,rest),after,removed_past(after,rest,past))

# The cells of values with their lanes on the unit axis have the extents of
# the values.
def unit_shape(+values: List<&2,U32>,+position: Grid.Position) -> {Tiled.cell_shape(values,List.length(&2,U32,values),position) == values : List<&2,U32>}:
  +rank = List.length(&2,U32,values)
  +past = Order.not_below(rank,rank,Order.reflexive(rank))
  %Equal.sym(Bool,Nat.is_lt(rank,rank),False{},past) : {Grid.placed(_,position,Grid.removed(values,rank),Grid.groups(Grid.lanes_at(values,rank))) == values : List<&2,U32>}
  removed_past(values,rank,past)

def outside_empty(+axis: Nat,inside: {Nat.is_lt(axis,0n) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(axis,0n),True{},Equal.sym(Bool,Nat.is_lt(axis,0n),False{},Division.lt_zero(axis)),inside))

# Inside a shape the lane axis comes back one longer than the other axes.
def removed_length(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {1n+List.length(&2,U32,Grid.removed(shape,axis)) == List.length(&2,U32,shape) : Nat}:
  match shape axis:
    case Nil{} axis: Empty.absurd({1n+List.length(&2,U32,Grid.removed(Nil{},axis)) == 0n : Nat},outside_empty(axis,inside))
    case Con{+extent,+after} 0n: {==}
    case Con{+extent,+after} 1n+ +rest:
      Equal.cong(Nat,Nat,length => 1n+length,1n+List.length(&2,U32,Grid.removed(after,rest)),List.length(&2,U32,after),removed_length(after,rest,inside))

def appended_length(+others: List<&2,U32>,+last: U32) -> {List.length(&2,U32,List.append(&2,U32,others,[last])) == 1n+List.length(&2,U32,others) : Nat}:
  match others:
    case Nil{}: {==}
    case Con{+head,+tail}: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,List.append(&2,U32,tail,[last])),1n+List.length(&2,U32,tail),appended_length(tail,last))

def placed_length(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  {List.length(&2,U32,Grid.placed(True{},position,others,groups)) == 1n+List.length(&2,U32,others) : Nat}:
  match position:
    case Grid.Innermost{}: appended_length(others,groups)
    case Grid.Outermost{}: {==}

def within_cases(within: Bool,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool}) ->
  {Nat.is_lt(axis,List.length(&2,U32,Grid.placed(within,position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis))))) == within : Bool}:
  match within:
    case True{}:
      +others = Grid.removed(shape,axis)
      %Equal.sym(Nat,List.length(&2,U32,Grid.placed(True{},position,others,Grid.groups(Grid.lanes_at(shape,axis)))),1n+List.length(&2,U32,others),
        placed_length(position,others,Grid.groups(Grid.lanes_at(shape,axis)))) : {Nat.is_lt(axis,_) == True{} : Bool}
      %Equal.sym(Nat,1n+List.length(&2,U32,others),List.length(&2,U32,shape),removed_length(shape,axis,classified)) : {Nat.is_lt(axis,_) == True{} : Bool}
      classified
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),shape,removed_past(shape,axis,classified)) : {Nat.is_lt(axis,List.length(&2,U32,_)) == False{} : Bool}
      classified

# The lane axis lies inside the cells of values exactly when it lies inside the
# values.
def cells_within(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position) ->
  {Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,position))) == Nat.is_lt(axis,List.length(&2,U32,values)) : Bool}:
  within_cases(Nat.is_lt(axis,List.length(&2,U32,values)),values,axis,position,{==})

# The values of a shape are the lanes of its lane axis at every index of the
# other axes.
def elements_split(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  match shape axis:
    case Nil{} axis:
      Empty.absurd({ShapeLaws.elements(Nil{}) == Nat.mul(U32.to_nat(Grid.lanes_at(Nil{},axis)),ShapeLaws.elements(Grid.removed(Nil{},axis))) : Nat},outside_empty(axis,inside))
    case Con{+extent,+after} 0n: {==}
    case Con{+extent,+after} 1n+ +rest:
      +lanes = U32.to_nat(Grid.lanes_at(after,rest))
      +others = ShapeLaws.elements(Grid.removed(after,rest))
      Equal.trans(Nat,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(after)),Nat.mul(U32.to_nat(extent),Nat.mul(lanes,others)),Nat.mul(lanes,Nat.mul(U32.to_nat(extent),others)),
        Equal.cong(Nat,Nat,count => Nat.mul(U32.to_nat(extent),count),ShapeLaws.elements(after),Nat.mul(lanes,others),elements_split(after,rest,inside)),
        RowMajor.swap_factors(U32.to_nat(extent),lanes,others))

def placed_cells(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)) : Nat}:
  match position:
    case Grid.Innermost{}:
      +count = ShapeLaws.elements(others)
      Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,others,[groups])),Nat.mul(count,Nat.mul(U32.to_nat(groups),1n)),Nat.mul(U32.to_nat(groups),count),
        ShapeLaws.elements_appended(others,[groups]),
        %Equal.sym(Nat,Nat.mul(U32.to_nat(groups),1n),U32.to_nat(groups),Order.times_one(U32.to_nat(groups))) : {Nat.mul(count,_) == Nat.mul(U32.to_nat(groups),count) : Nat}
          RowMajor.multiply_commutative(count,U32.to_nat(groups)))
    case Grid.Outermost{}: {==}

def without_last_appended(+others: List<&2,U32>,+last: U32) -> {Grid.without_last(List.append(&2,U32,others,[last])) == others : List<&2,U32>}:
  match others:
    case Nil{}: {==}
    case Con{+head,+tail}:
      match tail:
        case Nil{}: {==}
        case Con{+next,+rest}:
          Equal.cong(List<&2,U32>,List<&2,U32>,kept => head <> kept,Grid.without_last(List.append(&2,U32,next <> rest,[last])),next <> rest,without_last_appended(next <> rest,last))

def placed_others(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  {Grid.outer_count(Grid.placed(True{},position,others,groups),position) == Grid.product(others) : U32}:
  match position:
    case Grid.Innermost{}:
      Equal.cong(List<&2,U32>,U32,kept => Grid.product(kept),Grid.without_last(List.append(&2,U32,others,[groups])),others,without_last_appended(others,groups))
    case Grid.Outermost{}: {==}

# The cells of values with the lane axis inside them: the groups of the lanes
# at every index of the other axes, and the other cell axes.
def cells_split(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  +others = Grid.removed(shape,axis)
  +groups = Grid.groups(Grid.lanes_at(shape,axis))
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},inside) :
    {ShapeLaws.elements(Grid.placed(_,position,others,groups)) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)) : Nat}
  placed_cells(position,others,groups)

def others_split(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Grid.outer_count(Tiled.cell_shape(shape,axis,position),position) == Grid.product(Grid.removed(shape,axis)) : U32}:
  +others = Grid.removed(shape,axis)
  +groups = Grid.groups(Grid.lanes_at(shape,axis))
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},inside) :
    {Grid.outer_count(Grid.placed(_,position,others,groups),position) == Grid.product(others) : U32}
  placed_others(position,others,groups)

def inserted_front(extents: List<&2,U32>,+value: U32) -> {Grid.inserted(extents,0n,value) == value <> extents : List<&2,U32>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}: {==}

# Values with their extent at an axis inside them put back are the values.
def inserted_removed(values: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Grid.inserted(Grid.removed(values,axis),axis,Grid.lanes_at(values,axis)) == values : List<&2,U32>}:
  match values axis:
    case Con{+extent,+rest} 0n: inserted_front(rest,extent)
    case Con{+extent,+rest} 1n+ +k:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => extent <> later,Grid.inserted(Grid.removed(rest,k),k,Grid.lanes_at(rest,k)),rest,inserted_removed(rest,k,inside))
    case Nil{} +axis:
      Empty.absurd({Grid.inserted(Grid.removed(Nil{},axis),axis,Grid.lanes_at(Nil{},axis)) == Nil{} : List<&2,U32>},outside_empty(axis,inside))

def others_placed(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  {Grid.others_of(position,Grid.placed(True{},position,others,groups)) == others : List<&2,U32>}:
  match position:
    case Grid.Innermost{}: without_last_appended(others,groups)
    case Grid.Outermost{}: {==}

def restored_cases(within: Bool,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool}) ->
  {Grid.logical_when(within,Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),axis,position,Grid.lanes_at(values,axis)) == values : List<&2,U32>}:
  match within:
    case True{}:
      +others = Grid.removed(values,axis)
      +groups = Grid.groups(Grid.lanes_at(values,axis))
      %Equal.sym(List<&2,U32>,Grid.others_of(position,Grid.placed(True{},position,others,groups)),others,others_placed(position,others,groups)) :
        {Grid.inserted(_,axis,Grid.lanes_at(values,axis)) == values : List<&2,U32>}
      inserted_removed(values,axis,classified)
    case False{}: removed_past(values,axis,classified)

# The cells of values give the values back.
def cells_restored(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position) ->
  {Tiled.logical_shape(Tiled.cell_shape(values,axis,position),axis,Grid.lanes_at(values,axis),position) == values : List<&2,U32>}:
  +within = Nat.is_lt(axis,List.length(&2,U32,values))
  +cells = Tiled.cell_shape(values,axis,position)
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,cells)),within,cells_within(values,axis,position)) :
    {Grid.logical_when(_,cells,axis,position,Grid.lanes_at(values,axis)) == values : List<&2,U32>}
  restored_cases(within,values,axis,position,{==})

# The product of the extents of a shape within a bound is the count of its
# values: a zero extent makes both zero.
def product_head(zero: Bool,+extent: U32,+rest: List<&2,U32>,+classified: {U32.is_zero(extent) == zero : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool},
  later: {Nat.is_le(ShapeLaws.elements(rest),U32.to_nat(cap)) == True{} : Bool} -> {U32.to_nat(Grid.product(rest)) == ShapeLaws.elements(rest) : Nat}) ->
  {U32.to_nat((extent * Grid.product(rest) : U32)) == Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)) : Nat}:
  match zero:
    case True{}:
      +others = Grid.product(rest)
      +empty = Arithmetic.word_zero(extent,classified)
      %Equal.sym(U32,extent,0,empty) : {U32.to_nat((_ * others : U32)) == Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)) : Nat}
      %Equal.sym(U32,extent,0,empty) : {U32.to_nat((0 * others : U32)) == Nat.mul(U32.to_nat(_),ShapeLaws.elements(rest)) : Nat}
      %Equal.sym(U32,(0 * others : U32),0,Arithmetic.multiply_zero_left(others)) : {U32.to_nat(_) == 0n : Nat}
      {==}
    case False{}:
      +others = Grid.product(rest)
      +positive = Arithmetic.word_nonzero_positive(extent,classified)
      +exact = later(Order.transitive(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),U32.to_nat(cap),
        DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),positive),bound))
      Equal.trans(Nat,U32.to_nat((extent * others : U32)),Nat.mul(U32.to_nat(extent),U32.to_nat(others)),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),
        WordMultiplication.bounded_multiply(extent,others,cap,cap_ok,
          %Equal.sym(Nat,U32.to_nat(others),ShapeLaws.elements(rest),exact) : {Nat.is_le(Nat.mul(U32.to_nat(extent),_),U32.to_nat(cap)) == True{} : Bool} bound),
        Equal.cong(Nat,Nat,count => Nat.mul(U32.to_nat(extent),count),U32.to_nat(others),ShapeLaws.elements(rest),exact))

def product_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shape: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(shape)) == ShapeLaws.elements(shape) : Nat}:
  match shape:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      product_head(U32.is_zero(extent),extent,rest,{==},cap,cap_ok,bound,later => product_exact(cap,cap_ok,rest,later))

# The values before the lanes, the lanes and the values after them multiply
# to the values.
def around_product(+shape: List<&2,U32>,+axis: Nat,+outer: U32) ->
  {((lane_prefix(shape,axis,outer) * Grid.lanes_at(shape,axis) : U32) * Tiled.product(lane_after(shape,axis)) : U32) == (outer * Grid.product(shape) : U32) : U32}:
  match shape axis:
    case Nil{} axis: Arithmetic.multiply_one((outer * 1 : U32))
    case Con{+extent,+after} 0n: Arithmetic.multiply_associative(outer,extent,Grid.product(after))
    case Con{+extent,+after} 1n+ +rest:
      +next = (outer * extent : U32)
      Equal.trans(U32,((lane_prefix(after,rest,next) * Grid.lanes_at(after,rest) : U32) * Tiled.product(lane_after(after,rest)) : U32),
        (next * Grid.product(after) : U32),(outer * (extent * Grid.product(after) : U32) : U32),around_product(after,rest,next),
        Arithmetic.multiply_associative(outer,extent,Grid.product(after)))

# Nonempty values within a capacity are the values before the lanes, the lanes
# and the values after them.
def count_split(+shape: List<&2,U32>,+axis: Nat,+outer: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.mul(U32.to_nat(outer),ShapeLaws.elements(shape)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat(outer),ShapeLaws.elements(shape)) ==
    Nat.mul(Nat.mul(U32.to_nat(lane_prefix(shape,axis,outer)),U32.to_nat(Grid.lanes_at(shape,axis))),ShapeLaws.elements(lane_after(shape,axis))) : Nat}:
  match shape axis:
    case Nil{} axis:
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),1n),1n),Nat.mul(U32.to_nat(outer),1n),Order.times_one(Nat.mul(U32.to_nat(outer),1n)))
    case Con{+extent,+after} 0n:
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(after)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(after))),
        RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),ShapeLaws.elements(after)))
    case Con{+extent,+after} 1n+ +rest:
      +whole = U32.to_nat(outer)
      +size = U32.to_nat(extent)
      +later = ShapeLaws.elements(after)
      +next = (outer * extent : U32)
      +grouped = RowMajor.assoc(whole,size,later)
      +later_positive = DivisionOrder.right_factor_positive(size,later,positive)
      +small = Order.transitive(Nat.mul(whole,size),Nat.mul(Nat.mul(whole,size),later),U32.to_nat(cap),RowMajor.product_covers(Nat.mul(whole,size),later,later_positive),
        %Equal.sym(Nat,Nat.mul(Nat.mul(whole,size),later),Nat.mul(whole,Nat.mul(size,later)),grouped) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} bound)
      +exact = WordMultiplication.bounded_multiply(outer,extent,cap,cap_ok,small)
      +moved = Equal.trans(Nat,Nat.mul(whole,Nat.mul(size,later)),Nat.mul(Nat.mul(whole,size),later),Nat.mul(U32.to_nat(next),later),
        Equal.sym(Nat,Nat.mul(Nat.mul(whole,size),later),Nat.mul(whole,Nat.mul(size,later)),grouped),
        Equal.cong(Nat,Nat,value => Nat.mul(value,later),Nat.mul(whole,size),U32.to_nat(next),Equal.sym(Nat,U32.to_nat(next),Nat.mul(whole,size),exact)))
      Equal.trans(Nat,Nat.mul(whole,Nat.mul(size,later)),Nat.mul(U32.to_nat(next),later),
        Nat.mul(Nat.mul(U32.to_nat(lane_prefix(after,rest,next)),U32.to_nat(Grid.lanes_at(after,rest))),ShapeLaws.elements(lane_after(after,rest))),moved,
        count_split(after,rest,next,cap,cap_ok,%moved : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} bound,later_positive))

# Nonempty values laid around their lanes: whole blocks of the values after
# the lanes make the outer indices, and the blocks of lanes by the values
# after them make the values.
type Split<-limits: Machine.Limits,-values: List<&2,U32>,-axis: Nat,-lanes: U32> is Data:
  Split{blocks: Nat,
    outer_exact: {U32.to_nat((lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32)) ==
      Nat.mul(blocks,Representation.least_one(U32.to_nat(Tiled.product(lane_after(values,axis))))) : Nat},
    elements_exact: {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.least_one(U32.to_nat(Tiled.product(lane_after(values,axis))))) : Nat},
    lanes_within: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    after_positive: {Nat.is_lt(0n,U32.to_nat(Tiled.product(lane_after(values,axis)))) == True{} : Bool}}

def values_split(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +values_room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) -> Split<limits,values,axis,lanes>:
  +elements = ShapeLaws.elements(values)
  +prefix = lane_prefix(values,axis,1)
  +extent = Grid.lanes_at(values,axis)
  +after_axes = lane_after(values,axis)
  +inner = Tiled.product(after_axes)
  +outer = (prefix * inner : U32)
  +blocks = U32.to_nat(prefix)
  +lane_count = U32.to_nat(extent)
  +rest_count = ShapeLaws.elements(after_axes)
  +front = Nat.mul(blocks,lane_count)
  +cap = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +split = Equal.trans(Nat,elements,Nat.mul(1n,elements),Nat.mul(front,rest_count),Equal.sym(Nat,Nat.add(elements,0n),elements,Addition.zero_right(elements)),
    count_split(values,axis,1,cap,cap_ok,
      %Equal.sym(Nat,Nat.add(elements,0n),elements,Addition.zero_right(elements)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} values_room,nonempty))
  +positive = {%split : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,Nat.mul(front,rest_count)) == True{} : Bool}}
  +after_positive = DivisionOrder.right_factor_positive(front,rest_count,positive)
  +front_positive = RowMajor.left_factor_positive(front,rest_count,positive)
  +lanes_positive = DivisionOrder.right_factor_positive(blocks,lane_count,front_positive)
  +blocks_positive = RowMajor.left_factor_positive(blocks,lane_count,front_positive)
  +total_split = {%split : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} values_room : {Nat.is_le(Nat.mul(front,rest_count),U32.to_nat(cap)) == True{} : Bool}}
  +rest_room = Order.transitive(rest_count,Nat.mul(front,rest_count),U32.to_nat(cap),DivisionOrder.le_multiple(front,rest_count,front_positive),total_split)
  +inner_exact = product_exact(cap,cap_ok,after_axes,rest_room)
  +inner_back = Equal.sym(Nat,U32.to_nat(inner),rest_count,inner_exact)
  +inner_positive = {%inner_back : {Nat.is_lt(0n,_) == True{} : Bool} after_positive : {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}}
  +least = Representation.least_positive(U32.to_nat(inner),inner_positive)
  +least_back = Equal.sym(Nat,Representation.least_one(U32.to_nat(inner)),U32.to_nat(inner),least)
  +front_room = Order.transitive(Nat.mul(blocks,rest_count),Nat.mul(front,rest_count),U32.to_nat(cap),
    RowMajor.mul_le(blocks,front,rest_count,RowMajor.product_covers(blocks,lane_count,lanes_positive)),total_split)
  +outer_word = WordMultiplication.bounded_multiply(prefix,inner,cap,cap_ok,%inner_back : {Nat.is_le(Nat.mul(blocks,_),U32.to_nat(cap)) == True{} : Bool} front_room)
  +outer_exact = Equal.trans(Nat,U32.to_nat(outer),Nat.mul(blocks,U32.to_nat(inner)),Nat.mul(blocks,Representation.least_one(U32.to_nat(inner))),outer_word,
    Equal.cong(Nat,Nat,value => Nat.mul(blocks,value),U32.to_nat(inner),Representation.least_one(U32.to_nat(inner)),least_back))
  +lanes_exact = Equal.cong(U32,Nat,word => Nat.mul(Nat.mul(blocks,U32.to_nat(word)),rest_count),extent,lanes,pinned)
  +after_exact = Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),value),rest_count,Representation.least_one(U32.to_nat(inner)),
    Equal.trans(Nat,rest_count,U32.to_nat(inner),Representation.least_one(U32.to_nat(inner)),inner_back,least_back))
  +elements_exact = Equal.trans(Nat,elements,Nat.mul(front,rest_count),Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.least_one(U32.to_nat(inner))),split,
    Equal.trans(Nat,Nat.mul(front,rest_count),Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),rest_count),
      Nat.mul(Nat.mul(blocks,U32.to_nat(lanes)),Representation.least_one(U32.to_nat(inner))),lanes_exact,after_exact))
  +lanes_within = Order.transitive(lane_count,front,U32.to_nat(cap),
    %RowMajor.multiply_commutative(lane_count,blocks) : {Nat.is_le(lane_count,_) == True{} : Bool} RowMajor.product_covers(lane_count,blocks,blocks_positive),
    Order.transitive(front,Nat.mul(front,rest_count),U32.to_nat(cap),RowMajor.product_covers(front,rest_count,after_positive),total_split))
  +bound = {%pinned : {Nat.is_le(U32.to_nat(_),U32.to_nat(cap)) == True{} : Bool} lanes_within : {Nat.is_le(U32.to_nat(lanes),U32.to_nat(cap)) == True{} : Bool}}
  Split{blocks,outer_exact,elements_exact,bound,inner_positive}

# What a piece map needs of the cells of values: how the padding of the cells
# and the geometry of the values place the groups, the lanes at every index
# of the other cells within the machine, and the groups at every index of the
# other cells counted both ways.
type Placement<-lanes: U32,-padded_outer: U32,-padded_position: Grid.Position,-outer: U32,-position: Grid.Position> is Data:
  Along{outer_same: {padded_outer == outer : U32},position_same: {padded_position == position : Grid.Position}}
  Single{one: {lanes == 1 : U32},inner: {padded_position == Grid.Innermost{} : Grid.Position}}

type Cover<-limits: Machine.Limits,-lanes: U32,-padded_outer: U32,-padded_position: Grid.Position,-outer: U32,-position: Grid.Position,-total: Nat> is Data:
  Cover{counted: {Nat.is_le(Nat.mul(U32.to_nat(lanes),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells: {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)) == total : Nat},
    padded_cells: {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer)) == total : Nat},
    placement: Placement<lanes,padded_outer,padded_position,outer,position>}

# The cover of the product of the other axes, for lanes that are none or some.
def product_cover(zero: Bool,+limits: Machine.Limits,+lanes: U32,+others: List<&2,U32>,+position: Grid.Position,+total: Nat,+classified: {U32.is_zero(lanes) == zero : Bool},
  +counted: {Nat.is_le(Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(others)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells: {total == Nat.mul(U32.to_nat(Grid.groups(lanes)),ShapeLaws.elements(others)) : Nat}) ->
  Cover<limits,lanes,Grid.product(others),position,Grid.product(others),position,total>:
  match zero:
    case True{}:
      +product = Grid.product(others)
      +count = ShapeLaws.elements(others)
      +empty = Arithmetic.word_zero(lanes,classified)
      +none = {%Equal.sym(Nat,total,Nat.mul(U32.to_nat(Grid.groups(lanes)),count),cells) : {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(product)) == _ : Nat}
        %Equal.sym(U32,lanes,0,empty) : {Nat.mul(U32.to_nat(Grid.groups(_)),U32.to_nat(product)) == Nat.mul(U32.to_nat(Grid.groups(lanes)),count) : Nat}
        %Equal.sym(U32,lanes,0,empty) : {Nat.mul(U32.to_nat(Grid.groups(0)),U32.to_nat(product)) == Nat.mul(U32.to_nat(Grid.groups(_)),count) : Nat}
        {==} : {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(product)) == total : Nat}}
      Cover{%Equal.sym(U32,lanes,0,empty) : {Nat.is_le(Nat.mul(U32.to_nat(_),U32.to_nat(product)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} Order.zero_le(U32.to_nat(Machine.capacity(limits))),
        none,none,Along{{==},{==}}}
    case False{}:
      +product = Grid.product(others)
      +count = ShapeLaws.elements(others)
      +positive = Arithmetic.word_nonzero_positive(lanes,classified)
      +exact = product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),others,
        Order.transitive(count,Nat.mul(U32.to_nat(lanes),count),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(lanes),count,positive),counted))
      +some = {%Equal.sym(Nat,U32.to_nat(product),count,exact) : {Nat.mul(U32.to_nat(Grid.groups(lanes)),_) == total : Nat}
        Equal.sym(Nat,total,Nat.mul(U32.to_nat(Grid.groups(lanes)),count),cells) : {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(product)) == total : Nat}}
      Cover{%Equal.sym(Nat,U32.to_nat(product),count,exact) : {Nat.is_le(Nat.mul(U32.to_nat(lanes),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} counted,
        some,some,Along{{==},{==}}}

# The cover of values with the lane axis inside them, the cells at the cell
# shape and the values counted within the limits.
def along_cover(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+count: U32,
  +exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32}) ->
  Cover<limits,lanes,Grid.outer_count(Tiled.cell_shape(values,axis,position),position),position,(lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32),position,
    ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>:
  +others = Grid.removed(values,axis)
  +product = Grid.product(others)
  +outer = (lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32)
  +total = ShapeLaws.elements(Tiled.cell_shape(values,axis,position))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +counted = {%pinned : {Nat.is_le(Nat.mul(U32.to_nat(_),ShapeLaws.elements(others)),capacity) == True{} : Bool}
    %elements_split(values,axis,within) : {Nat.is_le(_,capacity) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(values),U32.to_nat(count),exact) : {Nat.is_le(_,capacity) == True{} : Bool} Bounds.as_nat_le(count,Machine.capacity(limits),bounded) :
    {Nat.is_le(Nat.mul(U32.to_nat(lanes),ShapeLaws.elements(others)),capacity) == True{} : Bool}}
  +cells = {%pinned : {total == Nat.mul(U32.to_nat(Grid.groups(_)),ShapeLaws.elements(others)) : Nat} cells_split(values,axis,position,within) :
    {total == Nat.mul(U32.to_nat(Grid.groups(lanes)),ShapeLaws.elements(others)) : Nat}}
  +cover = product_cover(U32.is_zero(lanes),limits,lanes,others,position,total,{==},counted,cells)
  +outer_exact = Equal.trans(U32,outer,(1 * product : U32),product,prefix_after(values,axis,1),one_times(product))
  %Equal.sym(U32,outer,product,outer_exact) : Cover<limits,lanes,Grid.outer_count(Tiled.cell_shape(values,axis,position),position),position,_,position,total>
  %Equal.sym(U32,Grid.outer_count(Tiled.cell_shape(values,axis,position),position),product,others_split(values,axis,position,within)) :
    Cover<limits,lanes,_,position,product,position,total>
  cover

# The cover of values with the lane axis past them: one lane, the cells the
# values themselves.
def single_cover(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+count: U32,
  +exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool}) ->
  Cover<limits,1,Grid.product(Tiled.cell_shape(values,axis,position)),Grid.Innermost{},(lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32),position,
    ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>:
  +product = Grid.product(values)
  +count_values = ShapeLaws.elements(values)
  +outer = (lane_prefix(values,axis,1) * Tiled.product(lane_after(values,axis)) : U32)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +bound = {%Equal.sym(Nat,count_values,U32.to_nat(count),exact) : {Nat.is_le(_,capacity) == True{} : Bool} Bounds.as_nat_le(count,Machine.capacity(limits),bounded) : {Nat.is_le(count_values,capacity) == True{} : Bool}}
  +product_count = product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),values,bound)
  +once = {Equal.trans(Nat,Nat.mul(1n,U32.to_nat(product)),U32.to_nat(product),count_values,Addition.zero_right(U32.to_nat(product)),product_count) :
    {Nat.mul(1n,U32.to_nat(product)) == count_values : Nat}}
  +cover = {Cover{%Equal.sym(Nat,Nat.mul(1n,U32.to_nat(product)),count_values,once) : {Nat.is_le(_,capacity) == True{} : Bool} bound,once,once,Single{{==},{==}}} :
    Cover<limits,1,product,Grid.Innermost{},product,position,count_values>}
  +shape_exact = {%Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),False{},past) :
      {Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) == values : List<&2,U32>}
    removed_past(values,axis,past) : {Tiled.cell_shape(values,axis,position) == values : List<&2,U32>}}
  +outer_exact = {Equal.trans(U32,outer,(1 * Grid.product(Grid.removed(values,axis)) : U32),Grid.product(Grid.removed(values,axis)),prefix_after(values,axis,1),
      one_times(Grid.product(Grid.removed(values,axis)))) : {outer == Grid.product(Grid.removed(values,axis)) : U32}}
  %Equal.sym(U32,outer,Grid.product(Grid.removed(values,axis)),outer_exact) :
    Cover<limits,1,Grid.product(Tiled.cell_shape(values,axis,position)),Grid.Innermost{},_,position,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>
  %Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,removed_past(values,axis,past)) :
    Cover<limits,1,Grid.product(Tiled.cell_shape(values,axis,position)),Grid.Innermost{},Grid.product(_),position,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>
  +shape_back = Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),values,shape_exact)
  %shape_back : Cover<limits,1,Grid.product(_),Grid.Innermost{},product,position,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>
  %shape_back : Cover<limits,1,product,Grid.Innermost{},product,position,ShapeLaws.elements(_)>
  cover

# ---- The contract ----

# The layout of the cells of a tiled tensor of values of a shape: the lane
# axis lies inside the values or just past them, the cells are the cell
# shape of the values and give them back, the values are counted, and the
# lanes are the extent of the values along the axis (1 past them).
type Laid<-cells: List<&2,U32>,-axis: Nat,-lanes: U32,-position: Grid.Position,-values: List<&2,U32>> is Data:
  Laid{inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
    shaped: {Tiled.cell_shape(values,axis,position) == cells : List<&2,U32>},
    restored: {Tiled.logical_shape(cells,axis,lanes,position) == values : List<&2,U32>},
    counted: Global.Counted<values>,
    pinned: {Grid.lanes_at(values,axis) == lanes : U32}}

# Usable laid out cells that hold the cells of a reader at a context made
# from a base and the geometry of the values.
def Within(~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Base: Data,~context: Base -> Cells.Cells -> Reader,+base: Base,+values: List<&2,U32>,+axis: Nat,+lanes: U32,
  +position: Grid.Position,cells: Tensors.Tensor<Tile.Tile8>) -> Data:
  match cells:
    case Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}:
      KernelGemm.Both<Global.Contract(~Tile.Tile8,~Reader,~read,context(base,geometry_of(values,axis,position)),Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
        Laid<shape,axis,lanes,position,values>>
    case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}: Empty

def Holds(~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Base: Data,~context: Base -> Cells.Cells -> Reader,+base: Base,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  match tiled:
    case Tiled.Tiled{cells,axis,lanes,position}: Within(~Reader,~read,~Base,~context,base,values,axis,lanes,position,cells)

def lifted_at(~Logical: Data,logical: Logical,geometry: Cells.Cells) -> Representation.Lifted<Logical>:
  Representation.Lifted{logical,geometry}

# What the tiled form of a tensor holds: laid out cells that hold the lifted
# values of the logical input in the geometry of the values.
def Contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Holds(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Logical,~lifted_at(~Logical),logical,values,tiled)

# Ready cells of the cell shape that hold the lifted values, laid out: what a
# contract holds, for cells that a run builds without reading them.
def ShapedLaid(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  match tiled:
    case Tiled.Tiled{cells,axis,lanes,position}:
      KernelGemm.Both<Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry_of(values,axis,position)},
          Tiled.cell_shape(values,axis,position),cells),
        Laid<Tiled.cell_shape(values,axis,position),axis,lanes,position,values>>

# Tiled values as the cells of a model, packed.
type Packed<-tiled: Tiled.Tiled> is Data:
  Packed{model: PiecesModel.Model<Tile.Tile8>,shape: List<&2,U32>,fill: Tile.Tile8,workers: U32,axis: Nat,lanes: U32,position: Grid.Position,
    equation: {Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position} == tiled : Tiled.Tiled}}

def packed_observed(-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +observation: PiecesModel.Observation<Tile.Tile8,pieces>) -> Packed<Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}>:
  PiecesModel.Observation{+model,+equation} = observation
  Packed{model,shape,fill,workers,axis,lanes,position,
    Equal.cong(Pieces.Pieces(~Tile.Tile8),Tiled.Tiled,stored => Tiled.Tiled{Tensors.Tensor{stored,shape,fill,Tensors.Usable{},workers},axis,lanes,position},
      PiecesModel.pack(~Tile.Tile8,model),pieces,equation)}

# Tiled values of the contract are the packed cells of a model.
def packed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+contract: Contract(~Logical,~original,logical,values,tiled)) ->
  Packed<tiled>:
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          packed_observed(pieces,shape,fill,workers,axis,lanes,position,PiecesModel.observe(~Tile.Tile8,pieces))
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}: Empty.absurd(Packed<Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}>,contract)

type Observed<-tiled: Tiled.Tiled> is Data:
  Observed{model: PiecesModel.Model<Tile.Tile8>,shape: List<&2,U32>,fill: Tile.Tile8,status: Tensors.Status,workers: U32,axis: Nat,lanes: U32,position: Grid.Position,
    equation: {Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,status,workers},axis,lanes,position} == tiled : Tiled.Tiled}}

def observed_cells(-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+status: Tensors.Status,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +observation: PiecesModel.Observation<Tile.Tile8,pieces>) -> Observed<Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,status,workers},axis,lanes,position}>:
  PiecesModel.Observation{+model,+equation} = observation
  Observed{model,shape,fill,status,workers,axis,lanes,position,
    Equal.cong(Pieces.Pieces(~Tile.Tile8),Tiled.Tiled,stored => Tiled.Tiled{Tensors.Tensor{stored,shape,fill,status,workers},axis,lanes,position},
      PiecesModel.pack(~Tile.Tile8,model),pieces,equation)}

def observe(tiled: Tiled.Tiled) -> Observed<tiled>:
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,+status,+workers}:
          observed_cells(pieces,shape,fill,status,workers,axis,lanes,position,PiecesModel.observe(~Tile.Tile8,pieces))

# Shaped laid out cells are cells of the contract.
def laid_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-cells: Tensors.Tensor<Tile.Tile8>,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +shaped: ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position})) -> Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}):
  KernelGemm.Both{KernelGemm.Both{Global.Usable{+model,+model_fill,+model_workers,+equation},+global},+laid} = shaped
  +cells_shape = Tiled.cell_shape(values,axis,position)
  %equation : Contract(~Logical,~original,logical,values,Tiled.Tiled{_,axis,lanes,position})
  KernelGemm.Both{{%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,model_fill,Tensors.Usable{},model_workers},cells,equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry_of(values,axis,position)},_)
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry_of(values,axis,position)},
      Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,model_fill,Tensors.Usable{},model_workers})},laid}

# The cells of a contract are a Shaped tensor of the cell shape and the
# lifted reader of the geometry of the values.
def cells_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,
  cells: Tensors.Tensor<Tile.Tile8>,contract: Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position})) ->
  Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),cells):
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      KernelGemm.Both{global,Laid{inside,+shaped,counted,restored,pinned}} = contract
      KernelGemm.Both{
        %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : Global.Usable<Tile.Tile8,_,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>
          Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),
        global}
    case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}:
      Empty.absurd(Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
        Representation.Lifted{logical,geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}),contract)

def lifted_logical(~Logical: Data,lifted: Representation.Lifted<Logical>) -> Logical:
  Representation.Lifted{+logical,+geometry} = lifted
  logical

def lifted_geometry(~Logical: Data,lifted: Representation.Lifted<Logical>) -> Cells.Cells:
  Representation.Lifted{+logical,+geometry} = lifted
  geometry

# Related readers lifted in one geometry.
def Relifted(~Before: Data,~After: Data,~Related: Before -> After -> Data,lifted: Representation.Lifted<Before>,other: Representation.Lifted<After>) -> Data:
  KernelGemm.Both<Related(lifted_logical(~Before,lifted),lifted_logical(~After,other)),
    {lifted_geometry(~Before,lifted) == lifted_geometry(~After,other) : Cells.Cells}>

def lane_agrees(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~F32,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat) ->
  {Representation.lane(~Before,~before_value,before,geometry,outer,group,lane) == Representation.lane(~After,~after_value,after,geometry,outer,group,lane) : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +index = U32.from_nat(Representation.logical_index(geometry,outer,channel))
  Equal.cong(F32,F32,value => Bool.pick(F32,Nat.is_lt(channel,Representation.lane_count(geometry)),value,0.0),before_value(before,index),after_value(after,index),
    agreement(before,after,related,index))

def cell_agrees(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~F32,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +geometry: Cells.Cells,+outer: Nat,+group: Nat) ->
  {Representation.cell(~Before,~before_value,before,geometry,outer,group) == Representation.cell(~After,~after_value,after,geometry,outer,group) : Tile.Tile8}:
  TileLaws.tiles_equal(Representation.cell(~Before,~before_value,before,geometry,outer,group),Representation.cell(~After,~after_value,after,geometry,outer,group),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,0n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,1n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,2n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,3n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,4n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,5n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,6n),
    lane_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,outer,group,7n))

def relifted_agree(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~F32,~Before,~After,~before_value,~after_value,~Related),+lifted: Representation.Lifted<Before>,+other: Representation.Lifted<After>,
  +related: Relifted(~Before,~After,~Related,lifted,other),+index: U32) ->
  {Representation.lifted(~Before,~before_value,lifted,index) == Representation.lifted(~After,~after_value,other,index) : Tile.Tile8}:
  match lifted:
    case Representation.Lifted{+logical,+geometry}:
      match other:
        case Representation.Lifted{+other_logical,+other_geometry}:
          KernelGemm.Both{+paired,+same} = related
          %same : {Representation.lifted(~Before,~before_value,Representation.Lifted{logical,geometry},index) ==
            Representation.lifted(~After,~after_value,Representation.Lifted{other_logical,_},index) : Tile.Tile8}
          +number = U32.to_nat(index)
          cell_agrees(~Before,~After,~before_value,~after_value,~Related,~agreement,logical,other_logical,paired,geometry,
            Representation.cell_outer(geometry,number),Representation.cell_group(geometry,number))

# The cells of a reader hold the values of every reader that agrees with it
# at every index.
def reread(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~F32,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  +values: List<&2,U32>,tiled: Tiled.Tiled,contract: Contract(~Before,~before_value,before,values,tiled)) -> Contract(~After,~after_value,after,values,tiled):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{global,+laid} = contract
          +geometry = geometry_of(values,axis,position)
          KernelGemm.Both{
            Global.relabeled(~Tile.Tile8,~Representation.Lifted<Before>,~Representation.Lifted<After>,~Representation.lifted(~Before,~before_value),
              ~Representation.lifted(~After,~after_value),~Relifted(~Before,~After,~Related),~relifted_agree(~Before,~After,~before_value,~after_value,~Related,~agreement),
              Representation.Lifted{before,geometry},Representation.Lifted{after,geometry},KernelGemm.Both{related,{==}},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},global),
            laid}
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Contract(~After,~after_value,after,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)

# ---- The packed cells of a contract ----

def by_packed_of(~Logical: Data,~original: Logical -> U32 -> F32,-Goal: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,-tiled: Tiled.Tiled,
  +described: Packed<tiled>,+contract: Contract(~Logical,~original,logical,values,tiled),
  goal: @+model: PiecesModel.Model<Tile.Tile8> -> @+shape: List<&2,U32> -> @+fill: Tile.Tile8 -> @+workers: U32 -> @+axis: Nat -> @+lanes: U32 -> @+position: Grid.Position ->
    @+contract: Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}) ->
    Goal(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})) -> Goal(tiled):
  Packed{+model,+shape,+fill,+workers,+axis,+lanes,+position,+equation} = described
  %equation : Goal(_)
  goal(model,shape,fill,workers,axis,lanes,position,
    %Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position},tiled,equation) :
      Contract(~Logical,~original,logical,values,_) contract)

# What holds of the packed cells of every tensor of the contract holds of a
# tensor of the contract.
def by_packed(~Logical: Data,~original: Logical -> U32 -> F32,-Goal: Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Contract(~Logical,~original,logical,values,tiled),
  goal: @+model: PiecesModel.Model<Tile.Tile8> -> @+shape: List<&2,U32> -> @+fill: Tile.Tile8 -> @+workers: U32 -> @+axis: Nat -> @+lanes: U32 -> @+position: Grid.Position ->
    @+contract: Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}) ->
    Goal(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})) -> Goal(tiled):
  by_packed_of(~Logical,~original,Goal,logical,values,tiled,packed(~Logical,~original,logical,values,tiled,contract),contract,goal)
