# TiledLayout.reshape: tiled values of a shape under extents of as many
# values hold the same values in row-major order.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_layout.bend as TiledLayout
import ../tensor_layout.bend as Layout
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./machine_limit_bounds.bend as Limits
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_values.bend as Values
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_view_reads.bend as ViewReads
import ../tensor_view.bend as Views
import ./tensor_view_gather.bend as ViewGather
import ./tensor_tiled_view.bend as TiledViewProof
import ./nat_ceiling.bend as Ceiling
import ./u32_division.bend as NativeDivision
import ./tensor_relabeled.bend as Relabeled
import ./tensor_tiled_view_reads.bend as KeptReads
import ./tensor_move_readback.bend as Held
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./tensor_tiled_split.bend as TiledSplit

# ---- The cells of a shape ----

def or_rest(+extent: U32,+rest: List<&2,U32>,+nonzero: {Shapes.has_zero(extent <> rest) == False{} : Bool}) -> {Shapes.has_zero(rest) == False{} : Bool}:
  Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)

def or_head(+extent: U32,+rest: List<&2,U32>,+nonzero: {Shapes.has_zero(extent <> rest) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))

# The lanes of any axis of a shape without a zero extent are some of its
# values.
def lanes_within(shape: List<&2,U32>,axis: Nat,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(shape)) == True{} : Bool}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n:
      RowMajor.product_covers(U32.to_nat(extent),ShapeLaws.elements(rest),CountCompletion.positive_elements(rest,or_rest(extent,rest,nonzero)))
    case Con{+extent,+rest} 1n+ +next:
      Order.transitive(U32.to_nat(Grid.lanes_at(rest,next)),ShapeLaws.elements(rest),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),
        lanes_within(rest,next,or_rest(extent,rest,nonzero)),
        %RowMajor.multiply_commutative(ShapeLaws.elements(rest),U32.to_nat(extent)) : {Nat.is_le(ShapeLaws.elements(rest),_) == True{} : Bool}
        RowMajor.product_covers(ShapeLaws.elements(rest),U32.to_nat(extent),or_head(extent,rest,nonzero)))

def lanes_positive(shape: List<&2,U32>,axis: Nat,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool}:
  match shape axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n: or_head(extent,rest,nonzero)
    case Con{+extent,+rest} 1n+next: lanes_positive(rest,next,or_rest(extent,rest,nonzero))

# The values of a shape off one axis are some of its values.
def removed_cases(within: Bool,+shape: List<&2,U32>,+axis: Nat,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool},
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),ShapeLaws.elements(shape)) == True{} : Bool}:
  match within:
    case True{}:
      +lanes = U32.to_nat(Grid.lanes_at(shape,axis))
      +others = ShapeLaws.elements(Grid.removed(shape,axis))
      %Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(lanes,others),Public.elements_split(shape,axis,classified)) : {Nat.is_le(others,_) == True{} : Bool}
      %RowMajor.multiply_commutative(others,lanes) : {Nat.is_le(others,_) == True{} : Bool}
      RowMajor.product_covers(others,lanes,lanes_positive(shape,axis,nonzero))
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),shape,Public.removed_past(shape,axis,classified)) : {Nat.is_le(ShapeLaws.elements(_),ShapeLaws.elements(shape)) == True{} : Bool}
      Order.reflexive(ShapeLaws.elements(shape))

def removed_within(+shape: List<&2,U32>,+axis: Nat,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),ShapeLaws.elements(shape)) == True{} : Bool}:
  removed_cases(Nat.is_lt(axis,List.length(&2,U32,shape)),shape,axis,{==},nonzero)

# The outer indices of an axis count the values off it.
def removed_exact(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32)) == ShapeLaws.elements(Grid.removed(shape,axis)) : Nat}:
  +others = Grid.product(Grid.removed(shape,axis))
  %Equal.sym(U32,(Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32),others,Equal.trans(U32,(Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32),(1 * others : U32),others,
      Public.prefix_after(shape,axis,1),Public.one_times(others))) :
    {U32.to_nat(_) == ShapeLaws.elements(Grid.removed(shape,axis)) : Nat}
  Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),Grid.removed(shape,axis),room)

# The cells of a shape are the groups of its lanes at every outer index.
def cells_cases(within: Bool,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool}) ->
  {ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  match within:
    case True{}: Public.cells_split(shape,axis,position,classified)
    case False{}:
      +others = Grid.removed(shape,axis)
      %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),False{},classified) :
        {ShapeLaws.elements(Grid.placed(_,position,others,Grid.groups(Grid.lanes_at(shape,axis)))) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(others)) : Nat}
      %Equal.sym(U32,Grid.lanes_at(shape,axis),1,Public.lanes_past(shape,axis,classified)) : {ShapeLaws.elements(others) == Nat.mul(U32.to_nat(Grid.groups(_)),ShapeLaws.elements(others)) : Nat}
      Equal.sym(Nat,Nat.add(ShapeLaws.elements(others),0n),ShapeLaws.elements(others),Addition.zero_right(ShapeLaws.elements(others)))

def cells_elements(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) ->
  {ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  cells_cases(Nat.is_lt(axis,List.length(&2,U32,shape)),shape,axis,position,{==})

# A shape without a zero extent within the capacity has cells.
def cells_positive(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(shape,axis,position))) == True{} : Bool}:
  +lanes = Grid.lanes_at(shape,axis)
  +groups = Values.group_positive(limits,lanes,
    Values.grouped(limits,lanes,Order.transitive(U32.to_nat(lanes),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),lanes_within(shape,axis,nonzero),bound)),
    lanes_positive(shape,axis,nonzero))
  %Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),Nat.mul(U32.to_nat(Grid.groups(lanes)),ShapeLaws.elements(Grid.removed(shape,axis))),
      cells_elements(shape,axis,position)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(Grid.groups(lanes)),ShapeLaws.elements(Grid.removed(shape,axis)),groups,
    CountCompletion.positive_elements(Grid.removed(shape,axis),KeptReads.removed_nonzero(shape,axis,nonzero)))

# A shape of values has no zero extent.
def nonzero_of(shape: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Shapes.has_zero(shape) == False{} : Bool}:
  match shape:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +head = Arithmetic.word_positive_nonzero(extent,RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))
      %Equal.sym(Bool,U32.is_zero(extent),False{},head) : {_ || Shapes.has_zero(rest) == False{} : Bool}
      nonzero_of(rest,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))

# ---- Cells under the cell shape of the new extents ----

# Cells of the lifted values of the new extents, reshaped to the cell shape
# of an axis of the new extents with the lanes of that axis, are the tiled
# form of the new extents.
def moved_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,
  +other: Nat,+lanes: U32,+position: Grid.Position,-cells: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(shape,other,position)},Tiled.cell_shape(values,axis,position),cells),
  +cells_same: {ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) == ShapeLaws.elements(Tiled.cell_shape(shape,other,position)) : Nat},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},
  +inside: {Nat.is_le(other,List.length(&2,U32,shape)) == True{} : Bool},+pinned: {Grid.lanes_at(shape,other) == lanes : U32},+counted: Global.Counted<shape>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,other,position)),other,lanes,position}):
  +cell_shape = Tiled.cell_shape(shape,other,position)
  Public.laid_contract(~Logical,~original,logical,shape,Layout.reshape(~Tile.Tile8,cells,cell_shape),other,lanes,position,
    KernelGemm.Both{
      Relabeled.reshaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(shape,other,position)},
        limits,Tiled.cell_shape(values,axis,position),cell_shape,cells,shaped,cells_same,positive,default_capacity),
      Public.Laid{inside,{==},{%pinned : {Tiled.logical_shape(cell_shape,other,_,position) == shape : List<&2,U32>} Public.cells_restored(shape,other,position) :
        {Tiled.logical_shape(cell_shape,other,lanes,position) == shape : List<&2,U32>}},counted,pinned}})

# ---- An axis that keeps the lanes and the axes around them ----

def kept_of(+values: List<&2,U32>,+axis: Nat,+innermost: Bool) -> TiledLayout.Kept:
  TiledLayout.Kept{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Tiled.product(Public.lane_after(values,axis)),innermost}

def triple_first(+first: Bool,+second: Bool,+third: Bool,+all: {first && second && third == True{} : Bool}) -> {first == True{} : Bool}:
  Boolean.left(first,second,Boolean.left(first && second,third,all))

def triple_second(+first: Bool,+second: Bool,+third: Bool,+all: {first && second && third == True{} : Bool}) -> {second == True{} : Bool}:
  Boolean.right(first,second,Boolean.left(first && second,third,all))

def triple_third(+first: Bool,+second: Bool,+third: Bool,+all: {first && second && third == True{} : Bool}) -> {third == True{} : Bool}:
  Boolean.right(first && second,third,all)

# An axis of the new extents that keeps the lanes and the axes around them
# has the geometry of the lane axis and as many cells.
type Around<-values: List<&2,U32>,-axis: Nat,-shape: List<&2,U32>,-other: Nat> is Data:
  Around{outer: {Public.lane_prefix(shape,other,1) == Public.lane_prefix(values,axis,1) : U32},lanes: {Grid.lanes_at(shape,other) == Grid.lanes_at(values,axis) : U32},
    inner: {Tiled.product(Public.lane_after(shape,other)) == Tiled.product(Public.lane_after(values,axis)) : U32}}

def around_of(+values: List<&2,U32>,+axis: Nat,+innermost: Bool,+shape: List<&2,U32>,+other: Nat,+inside: {Nat.is_le(other,List.length(&2,U32,shape)) == True{} : Bool},
  +fits: {TiledLayout.same_around(Tiled.around(shape,other,1),kept_of(values,axis,innermost)) == True{} : Bool}) -> Around<values,axis,shape,other>:
  +outer = U32.is_eq(Public.lane_prefix(shape,other,1),Public.lane_prefix(values,axis,1))
  +lanes = U32.is_eq(Grid.lanes_at(shape,other),Grid.lanes_at(values,axis))
  +inner = U32.is_eq(Tiled.product(Public.lane_after(shape,other)),Tiled.product(Public.lane_after(values,axis)))
  +all = {%Public.located(shape,other,1,inside) : {TiledLayout.same_around(_,kept_of(values,axis,innermost)) == True{} : Bool} fits : {outer && lanes && inner == True{} : Bool}}
  Around{Arithmetic.equal_of_is_eq(Public.lane_prefix(shape,other,1),Public.lane_prefix(values,axis,1),triple_first(outer,lanes,inner,all)),
    Arithmetic.equal_of_is_eq(Grid.lanes_at(shape,other),Grid.lanes_at(values,axis),triple_second(outer,lanes,inner,all)),
    Arithmetic.equal_of_is_eq(Tiled.product(Public.lane_after(shape,other)),Tiled.product(Public.lane_after(values,axis)),triple_third(outer,lanes,inner,all))}

def around_geometry(+values: List<&2,U32>,+axis: Nat,+shape: List<&2,U32>,+other: Nat,+position: Grid.Position,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+other_inside: {Nat.is_le(other,List.length(&2,U32,shape)) == True{} : Bool},
  +around: Around<values,axis,shape,other>) -> {Public.geometry_of(values,axis,position) == Public.geometry_of(shape,other,position) : Cells.Cells}:
  Around{+outer,+lanes,+inner} = around
  +geometry = {Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
    Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position} : Cells.Cells}
  %Equal.sym(Cells.Cells,Public.geometry_of(shape,other,position),Cells.Cells{(Public.lane_prefix(shape,other,1) * Tiled.product(Public.lane_after(shape,other)) : U32),
      Tiled.product(Public.lane_after(shape,other)),Grid.lanes_at(shape,other),Tiled.groups(Grid.lanes_at(shape,other)),position},
      Public.geometry_exact(shape,other,position,other_inside)) : {Public.geometry_of(values,axis,position) == _ : Cells.Cells}
  %Equal.sym(U32,Public.lane_prefix(shape,other,1),Public.lane_prefix(values,axis,1),outer) : {Public.geometry_of(values,axis,position) == Cells.Cells{(_ * Tiled.product(Public.lane_after(shape,other)) : U32),Tiled.product(Public.lane_after(shape,other)),
    Grid.lanes_at(shape,other),Tiled.groups(Grid.lanes_at(shape,other)),position} : Cells.Cells}
  %Equal.sym(U32,Tiled.product(Public.lane_after(shape,other)),Tiled.product(Public.lane_after(values,axis)),inner) :
    {Public.geometry_of(values,axis,position) == Cells.Cells{(Public.lane_prefix(values,axis,1) * _ : U32),_,
    Grid.lanes_at(shape,other),Tiled.groups(Grid.lanes_at(shape,other)),position} : Cells.Cells}
  %Equal.sym(U32,Grid.lanes_at(shape,other),Grid.lanes_at(values,axis),lanes) : {Public.geometry_of(values,axis,position) == Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),
    Tiled.product(Public.lane_after(values,axis)),_,Tiled.groups(_),position} : Cells.Cells}
  Public.geometry_exact(values,axis,position,inside)

def around_cells(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+shape: List<&2,U32>,+other: Nat,+position: Grid.Position,
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+around: Around<values,axis,shape,other>) ->
  {ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) == ShapeLaws.elements(Tiled.cell_shape(shape,other,position)) : Nat}:
  Around{+outer,+lanes,+inner} = around
  +capacity = U32.to_nat(Machine.capacity(limits))
  +outer_inner = {(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32) : U32}
  +removed_same = Equal.trans(Nat,ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(outer_inner),ShapeLaws.elements(Grid.removed(shape,other)),
    Equal.sym(Nat,U32.to_nat(outer_inner),ShapeLaws.elements(Grid.removed(values,axis)),
      removed_exact(limits,values,axis,Order.transitive(ShapeLaws.elements(Grid.removed(values,axis)),ShapeLaws.elements(values),capacity,removed_within(values,axis,nonzero),bound))),
    %outer : {U32.to_nat((_ * Tiled.product(Public.lane_after(values,axis)) : U32)) == ShapeLaws.elements(Grid.removed(shape,other)) : Nat}
    %inner : {U32.to_nat((Public.lane_prefix(shape,other,1) * _ : U32)) == ShapeLaws.elements(Grid.removed(shape,other)) : Nat}
    removed_exact(limits,shape,other,Order.transitive(ShapeLaws.elements(Grid.removed(shape,other)),ShapeLaws.elements(shape),capacity,removed_within(shape,other,shape_nonzero),shape_bound)))
  %Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),
      Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),cells_elements(values,axis,position)) :
    {_ == ShapeLaws.elements(Tiled.cell_shape(shape,other,position)) : Nat}
  %Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(shape,other,position)),
      Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,other))),ShapeLaws.elements(Grid.removed(shape,other))),cells_elements(shape,other,position)) :
    {Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))) == _ : Nat}
  %Equal.sym(U32,Grid.lanes_at(shape,other),Grid.lanes_at(values,axis),lanes) :
    {Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))) ==
    Nat.mul(U32.to_nat(Grid.groups(_)),ShapeLaws.elements(Grid.removed(shape,other))) : Nat}
  %removed_same : {Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))) ==
    Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),_) : Nat}
  {==}

# Cells of a contract under the cell shape of an axis of the new extents that
# keeps the lanes and the axes around them are the tiled form of the new
# extents.
def relabel_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,
  +other: Nat,+lanes: U32,+position: Grid.Position,-cells: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),cells),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<shape>,
  +other_inside: {Nat.is_le(other,List.length(&2,U32,shape)) == True{} : Bool},+around: Around<values,axis,shape,other>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,other,position)),other,lanes,position}):
  Around{+outer,+around_lanes,+inner} = around
  +kept = {Around{outer,around_lanes,inner} : Around<values,axis,shape,other>}
  moved_cells(~Logical,~original,logical,limits,values,shape,axis,other,lanes,position,cells,
    {%around_geometry(values,axis,shape,other,position,inside,other_inside,kept) :
        Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},Tiled.cell_shape(values,axis,position),cells)
      shaped : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
        Representation.Lifted{logical,Public.geometry_of(shape,other,position)},Tiled.cell_shape(values,axis,position),cells)},
    around_cells(limits,values,axis,shape,other,position,nonzero,shape_nonzero,bound,shape_bound,kept),cells_positive(limits,values,axis,position,nonzero,bound),
    other_inside,Equal.trans(U32,Grid.lanes_at(shape,other),Grid.lanes_at(values,axis),lanes,around_lanes,pinned),counted,default_capacity)

# ---- The first attempt ----

# A usable source of values without a zero extent and of the count of the new
# extents tries the axes that keep its lanes and the axes around them.
def around_relabelled(+values: List<&2,U32>,+shape: List<&2,U32>,-tiled: Tiled.Tiled,+axis: Nat,+innermost: Bool,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonzero: {Shapes.has_zero(values) == False{} : Bool},+count: U32,
  +counted_values: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},+counted_shape: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {TiledLayout.reshape_around(Tiled.around(values,axis,1),
      Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(values)) && Layout.same_count(Tensors.count_of(values),Tensors.count_of(shape)),innermost,
      Tensors.Usable{},tiled,values,shape) ==
    TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) :
    Tiled.Attempt<Tiled.Tiled>}:
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,axis,1),Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      Public.located(values,axis,1,inside)) :
    {TiledLayout.reshape_around(_,Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(values)) && Layout.same_count(Tensors.count_of(values),Tensors.count_of(shape)),
      innermost,Tensors.Usable{},tiled,values,shape) == TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) : Tiled.Attempt<Tiled.Tiled>}
  %Equal.sym(Bool,Shapes.has_zero(values),False{},nonzero) :
    {TiledLayout.reshape_around(Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      Tensors.usable(Tensors.Usable{}) && Bool.not(_) && Layout.same_count(Tensors.count_of(values),Tensors.count_of(shape)),
      innermost,Tensors.Usable{},tiled,values,shape) == TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) : Tiled.Attempt<Tiled.Tiled>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(values),Some{count},counted_values) :
    {TiledLayout.reshape_around(Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      Layout.same_count(_,Tensors.count_of(shape)),innermost,Tensors.Usable{},tiled,values,shape) == TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) : Tiled.Attempt<Tiled.Tiled>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counted_shape) :
    {TiledLayout.reshape_around(Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      Layout.same_count(Some{count},_),innermost,Tensors.Usable{},tiled,values,shape) == TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) : Tiled.Attempt<Tiled.Tiled>}
  %Equal.sym(Bool,U32.is_eq(count,count),True{},Arithmetic.equal_reflexive(count)) :
    {TiledLayout.reshape_around(Some{Tiled.Around{Public.lane_prefix(values,axis,1),Grid.lanes_at(values,axis),Public.lane_after(values,axis)}},
      _,innermost,Tensors.Usable{},tiled,values,shape) == TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),tiled,shape) : Tiled.Attempt<Tiled.Tiled>}
  {==}

# The runtime reads the values of the cells back as the source extents.
def at_relabelled(+values: List<&2,U32>,+shape: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,
  +lanes: U32,+position: Grid.Position,+restored: {Tiled.logical_shape(cells_shape,axis,lanes,position) == values : List<&2,U32>},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+nonzero: {Shapes.has_zero(values) == False{} : Bool},+count: U32,
  +counted_values: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},+counted_shape: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {TiledLayout.reshape_at(shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}) ==
    TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),
      kept_of(values,axis,Bool.not(Grid.is_outermost(position))),Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape) :
    Tiled.Attempt<Tiled.Tiled>}:
  +innermost = Bool.not(Grid.is_outermost(position))
  Equal.trans(Tiled.Attempt<Tiled.Tiled>,
    TiledLayout.reshape_at(shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
    TiledLayout.reshape_around(Tiled.around(values,axis,1),
      Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(values)) && Layout.same_count(Tensors.count_of(values),Tensors.count_of(shape)),innermost,
      Tensors.Usable{},Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},values,shape),
    TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost)),kept_of(values,axis,innermost),
      Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),
    Equal.cong(List<&2,U32>,Tiled.Attempt<Tiled.Tiled>,
      source => TiledLayout.reshape_around(Tiled.around(source,axis,1),
        Tensors.usable(Tensors.Usable{}) && Bool.not(Shapes.has_zero(source)) && Layout.same_count(Tensors.count_of(source),Tensors.count_of(shape)),innermost,
        Tensors.Usable{},Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},source,shape),
      Tiled.logical_shape(cells_shape,axis,lanes,position),values,restored),
    around_relabelled(values,shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},axis,innermost,inside,nonzero,count,
      counted_values,counted_shape))

# ---- The search for an axis ----

# Where a search for an axis of the new extents stops: an axis that fits, or
# none.
def Searched(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,found: Maybe<&2,Nat>,+shape: List<&2,U32>,+kept: TiledLayout.Kept) -> Data:
  match found:
    case Some{+axis}:
      KernelGemm.Both<{Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool},{fits(Tiled.around(shape,axis,1),kept) == True{} : Bool}>
    case None{}: Unit

def scan_searched(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,remaining: Nat,found: Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,
  +axis: Nat,+classified: {fits(Tiled.around(shape,axis,1),kept) == found : Bool},+room: {Nat.is_le(Nat.add(axis,remaining),List.length(&2,U32,shape)) == True{} : Bool}) ->
  Searched(~fits,TiledLayout.scan(~fits,remaining,found,shape,kept,axis),shape,kept):
  match remaining found:
    case remaining True{}:
      KernelGemm.Both{Order.transitive(axis,Nat.add(axis,remaining),List.length(&2,U32,shape),Order.le_add(axis,remaining),room),classified}
    case 0n False{}: Unit{}
    case 1n+ +rest False{}:
      +next = Nat.add(axis,1n)
      scan_searched(~fits,rest,fits(Tiled.around(shape,next,1),kept),shape,kept,next,{==},
        %Equal.sym(Nat,Nat.add(axis,1n),1n+axis,Addition.successor_width(axis)) : {Nat.is_le(Nat.add(_,rest),List.length(&2,U32,shape)) == True{} : Bool}
        Order.advance_room(axis,rest,List.length(&2,U32,shape),room))

def preferred_searched(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,found: Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,
  +preferred: Nat,+classified: {fits(Tiled.around(shape,preferred,1),kept) == found : Bool},+within: {Nat.is_le(preferred,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Searched(~fits,TiledLayout.preferred_found(~fits,found,shape,kept,preferred),shape,kept):
  match found:
    case True{}: KernelGemm.Both{within,classified}
    case False{}:
      scan_searched(~fits,List.length(&2,U32,shape),fits(Tiled.around(shape,0n,1),kept),shape,kept,0n,{==},Order.reflexive(List.length(&2,U32,shape)))

def search_found(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept) ->
  Searched(~fits,TiledLayout.axis_fitting(~fits,shape,kept),shape,kept):
  +preferred = Tiled.lane_axis(List.length(&2,U32,shape))
  preferred_searched(~fits,fits(Tiled.around(shape,preferred,1),kept),shape,kept,preferred,{==},Public.axis_within(List.length(&2,U32,shape)))

# ---- Whole groups ----

# The values of a shape are the lanes of any axis at every index of the
# others; the unit axis has one lane.
def lanes_cases(within: Bool,+shape: List<&2,U32>,+axis: Nat,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool}) ->
  {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  match within:
    case True{}: Public.elements_split(shape,axis,classified)
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),shape,Public.removed_past(shape,axis,classified)) :
        {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(_)) : Nat}
      %Equal.sym(U32,Grid.lanes_at(shape,axis),1,Public.lanes_past(shape,axis,classified)) : {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(_),ShapeLaws.elements(shape)) : Nat}
      Equal.sym(Nat,Nat.add(ShapeLaws.elements(shape),0n),ShapeLaws.elements(shape),Addition.zero_right(ShapeLaws.elements(shape)))

def lanes_split(+shape: List<&2,U32>,+axis: Nat) ->
  {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(Grid.removed(shape,axis))) : Nat}:
  lanes_cases(Nat.is_lt(axis,List.length(&2,U32,shape)),shape,axis,{==})

def groups_exact(+limits: Machine.Limits,+channels: U32,+facts: Values.Groups<limits,channels>) ->
  {U32.to_nat(Grid.groups(channels)) == Ceiling.ceiling(U32.to_nat(channels),8n) : Nat}:
  Values.Groups{+exact,+fewer,+lanes} = facts
  exact

# Lanes that fill their groups are eight times the groups.
def eight_groups(+limits: Machine.Limits,+lanes: U32,+bound: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +whole: {U32.is_zero((lanes % 8 : U32)) == True{} : Bool}) -> {U32.to_nat(lanes) == Nat.mul(U32.to_nat(Grid.groups(lanes)),8n) : Nat}:
  +count = U32.to_nat(lanes)
  +quotient = Nat.div(count,8n)
  +remainder = Equal.trans(Nat,Nat.mod(count,8n),U32.to_nat((lanes % 8 : U32)),0n,
    Equal.sym(Nat,U32.to_nat((lanes % 8 : U32)),Nat.mod(count,8n),NativeDivision.modulo(lanes,8,16,{==},{==},{==})),
    Equal.cong(U32,Nat,word => U32.to_nat(word),(lanes % 8 : U32),0,Arithmetic.word_zero((lanes % 8 : U32),whole)))
  %Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(count,8n),groups_exact(limits,lanes,Values.grouped(limits,lanes,bound))) :
    {count == Nat.mul(_,8n) : Nat}
  %Equal.sym(Nat,Nat.mod(count,8n),0n,remainder) : {count == Nat.mul(Ceiling.rounded(quotient,_),8n) : Nat}
  Equal.trans(Nat,count,Nat.add(Nat.mul(quotient,8n),Nat.mod(count,8n)),Nat.mul(quotient,8n),DivisionOrder.quotient_exact(count,8n,{==}),
    Equal.trans(Nat,Nat.add(Nat.mul(quotient,8n),Nat.mod(count,8n)),Nat.add(Nat.mul(quotient,8n),0n),Nat.mul(quotient,8n),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(quotient,8n),value),Nat.mod(count,8n),0n,remainder),Addition.zero_right(Nat.mul(quotient,8n))))

# The lane of a cell of grouped cells and the value it reads.
def lane_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+index: Nat,+lane: Nat) -> F32:
  Representation.lane(~Logical,~original,logical,geometry,Representation.cell_outer(geometry,index),Representation.cell_group(geometry,index),lane)

def grouped_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+index: Nat,+lane: Nat) -> F32:
  original(logical,U32.from_nat(Nat.add(Nat.mul(index,8n),lane)))

# Lane lane of cell index of cells whose lanes fill their groups and that
# have nothing after the lane axis reads value eight times the index past
# the lane.
def grouped_lane_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+lanes: U32,+groups: U32,+divisor: Nat,+quotient: Nat,+remainder: Nat,
  +index: Nat,+lane: Nat,+exact: {U32.to_nat(lanes) == Nat.mul(divisor,8n) : Nat},+split: {Nat.add(Nat.mul(quotient,divisor),remainder) == index : Nat},
  +small: {Nat.is_lt(remainder,divisor) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Representation.lane(~Logical,~original,logical,Cells.Cells{outer,1,lanes,groups,Grid.Innermost{}},quotient,remainder,lane) ==
    grouped_value(~Logical,~original,logical,index,lane) : F32}:
  +geometry = {Cells.Cells{outer,1,lanes,groups,Grid.Innermost{}} : Cells.Cells}
  +channel = Nat.add(lane,Nat.mul(remainder,8n))
  +block = Nat.mul(quotient,Nat.mul(divisor,8n))
  +inside = {%Equal.sym(Nat,U32.to_nat(lanes),Nat.mul(divisor,8n),exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    %Order.commutative(Nat.mul(remainder,8n),lane) : {Nat.is_lt(_,Nat.mul(divisor,8n)) == True{} : Bool}
    RowMajor.radix(remainder,divisor,lane,8n,small,lane_small) : {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool}}
  +read = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(quotient,1n),U32.to_nat(lanes)),channel),1n),Nat.mod(quotient,1n))
  +position = {%Equal.sym(Nat,Nat.mod(quotient,1n),0n,PaddedCells.mod_one(quotient)) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(quotient,1n),U32.to_nat(lanes)),channel),1n),_) == Nat.add(Nat.mul(index,8n),lane) : Nat}
    %Equal.sym(Nat,Nat.div(quotient,1n),quotient,ViewReads.divided_one(quotient)) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(_,U32.to_nat(lanes)),channel),1n),0n) == Nat.add(Nat.mul(index,8n),lane) : Nat}
    %Equal.sym(Nat,U32.to_nat(lanes),Nat.mul(divisor,8n),exact) : {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,_),channel),1n),0n) == Nat.add(Nat.mul(index,8n),lane) : Nat}
    %split : {Nat.add(Nat.mul(Nat.add(block,channel),1n),0n) == Nat.add(Nat.mul(_,8n),lane) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(block,channel),1n),0n),Nat.mul(Nat.add(block,channel),1n),Addition.zero_right(Nat.mul(Nat.add(block,channel),1n))) :
      {_ == Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),8n),lane) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.add(block,channel),1n),Nat.add(block,channel),Order.times_one(Nat.add(block,channel))) :
      {_ == Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),8n),lane) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),8n),Nat.add(Nat.mul(Nat.mul(quotient,divisor),8n),Nat.mul(remainder,8n)),
        RowMajor.distrib(Nat.mul(quotient,divisor),remainder,8n)) :
      {Nat.add(block,channel) == Nat.add(_,lane) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.mul(quotient,divisor),8n),block,RowMajor.assoc(quotient,divisor,8n)) :
      {Nat.add(block,channel) == Nat.add(Nat.add(_,Nat.mul(remainder,8n)),lane) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.add(block,Nat.mul(remainder,8n)),lane),Nat.add(block,Nat.add(Nat.mul(remainder,8n),lane)),Addition.associative(block,Nat.mul(remainder,8n),lane)) :
      {Nat.add(block,channel) == _ : Nat}
    %Order.commutative(lane,Nat.mul(remainder,8n)) : {Nat.add(block,channel) == Nat.add(block,_) : Nat}
    {==} : {read == Nat.add(Nat.mul(index,8n),lane) : Nat}}
  %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(lanes)),True{},inside) :
    {Bool.pick(F32,_,original(logical,U32.from_nat(read)),0.0) == grouped_value(~Logical,~original,logical,index,lane) : F32}
  Equal.cong(Nat,F32,number => original(logical,U32.from_nat(number)),read,Nat.add(Nat.mul(index,8n),lane),position)

def grouped_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+lanes: U32,+groups: U32,+index: Nat,+lane: Nat,
  +exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(groups),8n) : Nat},+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {lane_at(~Logical,~original,logical,Cells.Cells{outer,1,lanes,groups,Grid.Innermost{}},index,lane) == grouped_value(~Logical,~original,logical,index,lane) : F32}:
  +count = U32.to_nat(groups)
  +divisor = Representation.least_one(count)
  +least = Equal.sym(Nat,divisor,count,Representation.least_positive(count,positive))
  +divisor_positive = {%least : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,divisor) == True{} : Bool}}
  grouped_lane_at(~Logical,~original,logical,outer,lanes,groups,divisor,Nat.div(index,divisor),Nat.mod(index,divisor),index,lane,
    {%least : {U32.to_nat(lanes) == Nat.mul(_,8n) : Nat} exact : {U32.to_nat(lanes) == Nat.mul(divisor,8n) : Nat}},
    Equal.sym(Nat,index,Nat.add(Nat.mul(Nat.div(index,divisor),divisor),Nat.mod(index,divisor)),DivisionOrder.quotient_exact(index,divisor,divisor_positive)),
    DivisionOrder.remainder_small(index,divisor,divisor_positive),lane_small)

def tile_same(+a0: F32,+a1: F32,+a2: F32,+a3: F32,+a4: F32,+a5: F32,+a6: F32,+a7: F32,+b0: F32,+b1: F32,+b2: F32,+b3: F32,+b4: F32,+b5: F32,+b6: F32,+b7: F32,
  +e0: {a0 == b0 : F32},+e1: {a1 == b1 : F32},+e2: {a2 == b2 : F32},+e3: {a3 == b3 : F32},+e4: {a4 == b4 : F32},+e5: {a5 == b5 : F32},+e6: {a6 == b6 : F32},
  +e7: {a7 == b7 : F32}) -> {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} : Tile.Tile8}:
  %e0 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{_,b1,b2,b3,b4,b5,b6,b7} : Tile.Tile8}
  %e1 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,_,b2,b3,b4,b5,b6,b7} : Tile.Tile8}
  %e2 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,_,b3,b4,b5,b6,b7} : Tile.Tile8}
  %e3 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,a2,_,b4,b5,b6,b7} : Tile.Tile8}
  %e4 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,a2,a3,_,b5,b6,b7} : Tile.Tile8}
  %e5 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,a2,a3,a4,_,b6,b7} : Tile.Tile8}
  %e6 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,a2,a3,a4,a5,_,b7} : Tile.Tile8}
  %e7 : {Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} == Tile.Tile{a0,a1,a2,a3,a4,a5,a6,_} : Tile.Tile8}
  {==}

# The cell at an index of grouped cells holds the values eight times the
# index on.
def grouped_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+lanes: U32,+groups: U32,+index: U32,
  +exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(groups),8n) : Nat},+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,1,lanes,groups,Grid.Innermost{}}},index) ==
    Tile.Tile{grouped_value(~Logical,~original,logical,U32.to_nat(index),0n),grouped_value(~Logical,~original,logical,U32.to_nat(index),1n),
      grouped_value(~Logical,~original,logical,U32.to_nat(index),2n),grouped_value(~Logical,~original,logical,U32.to_nat(index),3n),
      grouped_value(~Logical,~original,logical,U32.to_nat(index),4n),grouped_value(~Logical,~original,logical,U32.to_nat(index),5n),
      grouped_value(~Logical,~original,logical,U32.to_nat(index),6n),grouped_value(~Logical,~original,logical,U32.to_nat(index),7n)} : Tile.Tile8}:
  +geometry = {Cells.Cells{outer,1,lanes,groups,Grid.Innermost{}} : Cells.Cells}
  +number = U32.to_nat(index)
  tile_same(lane_at(~Logical,~original,logical,geometry,number,0n),lane_at(~Logical,~original,logical,geometry,number,1n),lane_at(~Logical,~original,logical,geometry,number,2n),
    lane_at(~Logical,~original,logical,geometry,number,3n),lane_at(~Logical,~original,logical,geometry,number,4n),lane_at(~Logical,~original,logical,geometry,number,5n),
    lane_at(~Logical,~original,logical,geometry,number,6n),lane_at(~Logical,~original,logical,geometry,number,7n),
    grouped_value(~Logical,~original,logical,number,0n),grouped_value(~Logical,~original,logical,number,1n),grouped_value(~Logical,~original,logical,number,2n),
    grouped_value(~Logical,~original,logical,number,3n),grouped_value(~Logical,~original,logical,number,4n),grouped_value(~Logical,~original,logical,number,5n),
    grouped_value(~Logical,~original,logical,number,6n),grouped_value(~Logical,~original,logical,number,7n),
    grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,0n,exact,positive,{==}),grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,1n,exact,positive,{==}),
    grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,2n,exact,positive,{==}),grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,3n,exact,positive,{==}),
    grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,4n,exact,positive,{==}),grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,5n,exact,positive,{==}),
    grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,6n,exact,positive,{==}),grouped_lane(~Logical,~original,logical,outer,lanes,groups,number,7n,exact,positive,{==}))

def grouped_tile(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+index: Nat) -> Tile.Tile8:
  Tile.Tile{grouped_value(~Logical,~original,logical,index,0n),grouped_value(~Logical,~original,logical,index,1n),grouped_value(~Logical,~original,logical,index,2n),
    grouped_value(~Logical,~original,logical,index,3n),grouped_value(~Logical,~original,logical,index,4n),grouped_value(~Logical,~original,logical,index,5n),
    grouped_value(~Logical,~original,logical,index,6n),grouped_value(~Logical,~original,logical,index,7n)}

# Innermost cells with nothing after the lane axis whose lanes fill their
# groups.
def Grouped(geometry: Cells.Cells) -> Data:
  match geometry:
    case Cells.Cells{outer,+after,+lanes,+groups,position}:
      match position:
        case Grid.Innermost{}:
          KernelGemm.Both<{after == 1 : U32},
            KernelGemm.Both<{U32.to_nat(lanes) == Nat.mul(U32.to_nat(groups),8n) : Nat},{Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}>>
        case Grid.Outermost{}: Empty

def grouped_of(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,geometry: Cells.Cells,+grouped: Grouped(geometry),+index: U32) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},index) == grouped_tile(~Logical,~original,logical,U32.to_nat(index)) : Tile.Tile8}:
  match geometry:
    case Cells.Cells{+outer,+after,+lanes,+groups,position}:
      match position:
        case Grid.Innermost{}:
          KernelGemm.Both{+one,+rest} = grouped
          KernelGemm.Both{+exact,+positive} = rest
          %Equal.sym(U32,after,1,one) :
            {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,_,lanes,groups,Grid.Innermost{}}},index) ==
              grouped_tile(~Logical,~original,logical,U32.to_nat(index)) : Tile.Tile8}
          grouped_cell(~Logical,~original,logical,outer,lanes,groups,index,exact,positive)
        case Grid.Outermost{}:
          Empty.absurd({Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,lanes,groups,Grid.Outermost{}}},index) ==
            grouped_tile(~Logical,~original,logical,U32.to_nat(index)) : Tile.Tile8},grouped)

# Two grouped geometries of the same values read the same cells.
def Regrouped(~Logical: Data,before: Representation.Lifted<Logical>,after: Representation.Lifted<Logical>) -> Data:
  match before after:
    case Representation.Lifted{+logical,geometry} Representation.Lifted{+other,other_geometry}:
      KernelGemm.Both<{logical == other : Logical},KernelGemm.Both<Grouped(geometry),Grouped(other_geometry)>>

def regrouped_agree(~Logical: Data,~original: Logical -> U32 -> F32,+before: Representation.Lifted<Logical>,+after: Representation.Lifted<Logical>,
  +related: Regrouped(~Logical,before,after),+index: U32) ->
  {Representation.lifted(~Logical,~original,before,index) == Representation.lifted(~Logical,~original,after,index) : Tile.Tile8}:
  match before after:
    case Representation.Lifted{+logical,+geometry} Representation.Lifted{+other,+other_geometry}:
      KernelGemm.Both{+same,+groupings} = related
      KernelGemm.Both{+grouped,+other_grouped} = groupings
      %same : {Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},index) ==
        Representation.lifted(~Logical,~original,Representation.Lifted{_,other_geometry},index) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},index),grouped_tile(~Logical,~original,logical,U32.to_nat(index)),
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,other_geometry},index),grouped_of(~Logical,~original,logical,geometry,grouped,index),
        Equal.sym(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,other_geometry},index),grouped_tile(~Logical,~original,logical,U32.to_nat(index)),
          grouped_of(~Logical,~original,logical,other_geometry,other_grouped,index)))

# Cells of grouped values hold the values of another grouping.
def regrouped_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+other_geometry: Cells.Cells,+shape: List<&2,U32>,
  -cells: Tensors.Tensor<Tile.Tile8>,+grouped: Grouped(geometry),+other_grouped: Grouped(other_geometry),
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},shape,cells)) ->
  Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,other_geometry},shape,cells):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},+contract} = shaped
  %equation : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,other_geometry},shape,_)
  KernelGemm.Both{Global.Usable{model,fill,workers,{==}},
    Global.relabeled(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.lifted(~Logical,~original),
      ~Regrouped(~Logical),~regrouped_agree(~Logical,~original),Representation.Lifted{logical,geometry},Representation.Lifted{logical,other_geometry},
      KernelGemm.Both{{==},KernelGemm.Both{grouped,other_grouped}},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},
      {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},cells,equation) :
          Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},_)
        contract : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},
          Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})})}

# Shapes of as many values whose lane axes fill their groups have as many
# cells.
def eights(+groups: Nat,+others: Nat) -> {Nat.mul(Nat.mul(groups,8n),others) == Nat.mul(Nat.mul(groups,others),8n) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(groups,8n),others),Nat.mul(groups,Nat.mul(8n,others)),Nat.mul(Nat.mul(groups,others),8n),RowMajor.assoc(groups,8n,others),
    Equal.trans(Nat,Nat.mul(groups,Nat.mul(8n,others)),Nat.mul(groups,Nat.mul(others,8n)),Nat.mul(Nat.mul(groups,others),8n),
      Equal.cong(Nat,Nat,value => Nat.mul(groups,value),Nat.mul(8n,others),Nat.mul(others,8n),RowMajor.multiply_commutative(8n,others)),
      Equal.sym(Nat,Nat.mul(Nat.mul(groups,others),8n),Nat.mul(groups,Nat.mul(others,8n)),RowMajor.assoc(groups,others,8n))))

def split_eights(+shape: List<&2,U32>,+axis: Nat,+exact: {U32.to_nat(Grid.lanes_at(shape,axis)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),8n) : Nat}) ->
  {ShapeLaws.elements(shape) == Nat.mul(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),ShapeLaws.elements(Grid.removed(shape,axis))),8n) : Nat}:
  +groups = U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis)))
  +others = ShapeLaws.elements(Grid.removed(shape,axis))
  Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(Nat.mul(groups,8n),others),Nat.mul(Nat.mul(groups,others),8n),
    {%exact : {ShapeLaws.elements(shape) == Nat.mul(_,others) : Nat} lanes_split(shape,axis) : {ShapeLaws.elements(shape) == Nat.mul(Nat.mul(groups,8n),others) : Nat}},
    eights(groups,others))

def cancel_eight(+left: Nat,+right: Nat,+same: {Nat.mul(left,8n) == Nat.mul(right,8n) : Nat}) -> {left == right : Nat}:
  Order.antisymmetric(left,right,
    RowMajor.cancel_multiply_le(left,right,8n,{==},{%same : {Nat.is_le(Nat.mul(left,8n),_) == True{} : Bool} Order.reflexive(Nat.mul(left,8n)) :
      {Nat.is_le(Nat.mul(left,8n),Nat.mul(right,8n)) == True{} : Bool}}),
    RowMajor.cancel_multiply_le(right,left,8n,{==},{%same : {Nat.is_le(_,Nat.mul(left,8n)) == True{} : Bool} Order.reflexive(Nat.mul(left,8n)) :
      {Nat.is_le(Nat.mul(right,8n),Nat.mul(left,8n)) == True{} : Bool}}))

def regrouped_cells(+values: List<&2,U32>,+axis: Nat,+shape: List<&2,U32>,+other: Nat,+position: Grid.Position,
  +elements: {ShapeLaws.elements(values) == ShapeLaws.elements(shape) : Nat},
  +exact: {U32.to_nat(Grid.lanes_at(values,axis)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),8n) : Nat},
  +other_exact: {U32.to_nat(Grid.lanes_at(shape,other)) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,other))),8n) : Nat}) ->
  {ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) == ShapeLaws.elements(Tiled.cell_shape(shape,other,position)) : Nat}:
  +cells = Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))
  +other_cells = Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,other))),ShapeLaws.elements(Grid.removed(shape,other)))
  +products = Equal.trans(Nat,Nat.mul(cells,8n),ShapeLaws.elements(values),Nat.mul(other_cells,8n),Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(cells,8n),split_eights(values,axis,exact)),
    Equal.trans(Nat,ShapeLaws.elements(values),ShapeLaws.elements(shape),Nat.mul(other_cells,8n),elements,split_eights(shape,other,other_exact)))
  Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),cells,ShapeLaws.elements(Tiled.cell_shape(shape,other,position)),cells_elements(values,axis,position),
    Equal.trans(Nat,cells,other_cells,ShapeLaws.elements(Tiled.cell_shape(shape,other,position)),cancel_eight(cells,other_cells,products),
      Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(shape,other,position)),other_cells,cells_elements(shape,other,position))))

# The grouping of a lane axis of whole groups at an innermost position.
def grouped_geometry(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inner: {U32.is_eq(Tiled.product(Public.lane_after(values,axis)),1) == True{} : Bool},+whole: {U32.is_zero((Grid.lanes_at(values,axis) % 8 : U32)) == True{} : Bool}) ->
  Grouped(Public.geometry_of(values,axis,Grid.Innermost{})):
  +lanes = Grid.lanes_at(values,axis)
  +room = Order.transitive(U32.to_nat(lanes),ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits)),lanes_within(values,axis,nonzero),bound)
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Innermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),
      Tiled.product(Public.lane_after(values,axis)),lanes,Tiled.groups(lanes),Grid.Innermost{}},Public.geometry_exact(values,axis,Grid.Innermost{},inside)) : Grouped(_)
  KernelGemm.Both{Arithmetic.equal_of_is_eq(Tiled.product(Public.lane_after(values,axis)),1,inner),
    KernelGemm.Both{eight_groups(limits,lanes,room,whole),Values.group_positive(limits,lanes,Values.grouped(limits,lanes,room),lanes_positive(values,axis,nonzero))}}

# ---- The source ----

# A source of values without a zero extent and new extents of its count,
# both within the capacity of default limits.
type Source<-limits: Machine.Limits,-values: List<&2,U32>,-shape: List<&2,U32>> is Data:
  Source{count: U32,values_counted: {Tensors.count_of(values) == Some{count} : Maybe<&2,U32>},shape_counted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
    nonzero: {Shapes.has_zero(values) == False{} : Bool},shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},
    bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    shape_bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    elements: {ShapeLaws.elements(values) == ShapeLaws.elements(shape) : Nat},default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}}

def source_of(+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+counted: Global.Counted<values>,
  +same: {Tensors.count_of(shape) == Tensors.count_of(values) : Maybe<&2,U32>},+nonzero: {Shapes.has_zero(values) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) -> Source<limits,values,shape>:
  +count = Global.count(values,counted)
  +values_counted = Global.accepted(values,counted)
  +shape_counted = Equal.trans(Maybe<&2,U32>,Tensors.count_of(shape),Tensors.count_of(values),Some{count},same,values_counted)
  +elements = Equal.trans(Nat,ShapeLaws.elements(values),U32.to_nat(count),ShapeLaws.elements(shape),Global.whole_counted(values,count,values_counted),
    Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),Global.whole_counted(shape,count,shape_counted)))
  +shape_positive = {%elements : {Nat.is_lt(0n,_) == True{} : Bool} CountCompletion.positive_elements(values,nonzero) : {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}}
  Source{count,values_counted,shape_counted,nonzero,nonzero_of(shape,shape_positive),StorageCount.counted_room(limits,values,counted,default_capacity),
    StorageCount.counted_room(limits,shape,Global.Counted{count,shape_counted},default_capacity),elements,default_capacity}

# Cells of a lane axis of whole groups under the cell shape of a last axis of
# the new extents of whole groups are the tiled form of the new extents.
def regroup_contract(~Logical: Data,~original: Logical -> U32 -> F32,position: Grid.Position,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,
  +shape: List<&2,U32>,+axis: Nat,+other: Nat,-cells: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),cells),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+source: Source<limits,values,shape>,
  +other_inside: {Nat.is_le(other,List.length(&2,U32,shape)) == True{} : Bool},
  +fits: {TiledLayout.whole_groups(Tiled.around(shape,other,1),kept_of(values,axis,Bool.not(Grid.is_outermost(position)))) == True{} : Bool}) ->
  Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,other,position)),other,Grid.lanes_at(shape,other),position}):
  match position:
    case Grid.Innermost{}:
      Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
      +lanes = Grid.lanes_at(values,axis)
      +other_lanes = Grid.lanes_at(shape,other)
      +inner = U32.is_eq(Tiled.product(Public.lane_after(values,axis)),1)
      +whole = U32.is_zero((lanes % 8 : U32))
      +other_whole = U32.is_zero((other_lanes % 8 : U32))
      +other_inner = U32.is_eq(Tiled.product(Public.lane_after(shape,other)),1)
      +all = {%Public.located(shape,other,1,other_inside) : {TiledLayout.whole_groups(_,kept_of(values,axis,True{})) == True{} : Bool} fits :
        {inner && whole && other_whole && other_inner == True{} : Bool}}
      +three = Boolean.left(inner && whole && other_whole,other_inner,all)
      +two = Boolean.left(inner && whole,other_whole,three)
      +room = Order.transitive(U32.to_nat(lanes),ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits)),lanes_within(values,axis,nonzero),bound)
      +other_room = Order.transitive(U32.to_nat(other_lanes),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),lanes_within(shape,other,shape_nonzero),shape_bound)
      +whole_true = Boolean.right(inner,whole,two)
      +other_whole_true = Boolean.right(inner && whole,other_whole,three)
      moved_cells(~Logical,~original,logical,limits,values,shape,axis,other,other_lanes,Grid.Innermost{},cells,
        regrouped_shaped(~Logical,~original,logical,Public.geometry_of(values,axis,Grid.Innermost{}),Public.geometry_of(shape,other,Grid.Innermost{}),
          Tiled.cell_shape(values,axis,Grid.Innermost{}),cells,
          grouped_geometry(limits,values,axis,inside,nonzero,bound,Boolean.left(inner,whole,two),whole_true),
          grouped_geometry(limits,shape,other,other_inside,shape_nonzero,shape_bound,Boolean.right(inner && whole && other_whole,other_inner,all),other_whole_true),shaped),
        regrouped_cells(values,axis,shape,other,Grid.Innermost{},elements,eight_groups(limits,lanes,room,whole_true),eight_groups(limits,other_lanes,other_room,other_whole_true)),
        cells_positive(limits,values,axis,Grid.Innermost{},nonzero,bound),other_inside,{==},Global.Counted{count,shape_counted},default_capacity)
    case Grid.Outermost{}:
      Empty.absurd(Public.Contract(~Logical,~original,logical,shape,
          Tiled.Tiled{Layout.reshape(~Tile.Tile8,cells,Tiled.cell_shape(shape,other,Grid.Outermost{})),other,Grid.lanes_at(shape,other),Grid.Outermost{}}),
        WordEncoding.false_true({%Public.located(shape,other,1,other_inside) : {TiledLayout.whole_groups(_,kept_of(values,axis,False{})) == True{} : Bool} fits :
          {False{} == True{} : Bool}}))

# ---- The attempts ----

# Through an axis the search is known to find, the tiled form of the new
# extents.
def relabelled_reached(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,Nat>,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,
  +shape: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +searched: Searched(~TiledLayout.same_around,found,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),+some: {Maybe.is_some(&2,Nat,found) == True{} : Bool},
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,axis,position)},Tiled.cell_shape(values,axis,position),
    Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers}),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},+source: Source<limits,values,shape>) ->
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_again(TiledLayout.relabelled(found,kept_of(values,axis,Bool.not(Grid.is_outermost(position))),
    Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),values,shape)):
  match found:
    case Some{+other}:
      KernelGemm.Both{+other_inside,+fits} = searched
      Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
      relabel_contract(~Logical,~original,logical,limits,values,shape,axis,other,lanes,position,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},shaped,
        inside,pinned,nonzero,shape_nonzero,bound,shape_bound,Global.Counted{count,shape_counted},other_inside,around_of(values,axis,Bool.not(Grid.is_outermost(position)),shape,other,other_inside,fits),
        default_capacity)
    case None{}:
      Empty.absurd(Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_again(TiledLayout.relabelled(None{},kept_of(values,axis,Bool.not(Grid.is_outermost(position))),
        Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),values,shape)),WordEncoding.false_true(some))

# ---- The axis the lanes move to ----

def scan_absurd(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,+axis: Nat,+witness: Nat,
  +classified: {fits(Tiled.around(shape,axis,1),kept) == False{} : Bool},+same: {witness == axis : Nat},+hit: {fits(Tiled.around(shape,witness,1),kept) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},fits(Tiled.around(shape,axis,1),kept),True{},Equal.sym(Bool,fits(Tiled.around(shape,axis,1),kept),False{},classified),
    {%same : {fits(Tiled.around(shape,_,1),kept) == True{} : Bool} hit : {fits(Tiled.around(shape,axis,1),kept) == True{} : Bool}}))

# A scan over axes one of which fits finds an axis.
def scan_some(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,remaining: Nat,found: Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,
  +axis: Nat,+classified: {fits(Tiled.around(shape,axis,1),kept) == found : Bool},+witness: Nat,+after: {Nat.is_le(axis,witness) == True{} : Bool},
  +before: {Nat.is_le(witness,Nat.add(axis,remaining)) == True{} : Bool},+hit: {fits(Tiled.around(shape,witness,1),kept) == True{} : Bool}) ->
  {Maybe.is_some(&2,Nat,TiledLayout.scan(~fits,remaining,found,shape,kept,axis)) == True{} : Bool}:
  match remaining found:
    case remaining True{}: {==}
    case 0n False{}:
      Empty.absurd({Maybe.is_some(&2,Nat,TiledLayout.scan(~fits,0n,False{},shape,kept,axis)) == True{} : Bool},
        scan_absurd(~fits,shape,kept,axis,witness,classified,
          Order.antisymmetric(witness,axis,{%Addition.zero_right(axis) : {Nat.is_le(witness,_) == True{} : Bool} before : {Nat.is_le(witness,axis) == True{} : Bool}},after),hit))
    case 1n+ +rest False{}:
      +next = Nat.add(axis,1n)
      Held.by_cases({Maybe.is_some(&2,Nat,TiledLayout.scan(~fits,1n+rest,False{},shape,kept,axis)) == True{} : Bool},Nat.is_lt(axis,witness),
        below => scan_some(~fits,rest,fits(Tiled.around(shape,next,1),kept),shape,kept,next,{==},witness,
          {%Equal.sym(Nat,Nat.add(axis,1n),1n+axis,Addition.successor_width(axis)) : {Nat.is_le(_,witness) == True{} : Bool}
            Order.strict_follow(axis,witness,below) : {Nat.is_le(next,witness) == True{} : Bool}},
          {%Equal.sym(Nat,Nat.add(axis,1n),1n+axis,Addition.successor_width(axis)) : {Nat.is_le(witness,Nat.add(_,rest)) == True{} : Bool}
            %Addition.successor_right(axis,rest) : {Nat.is_le(witness,_) == True{} : Bool} before : {Nat.is_le(witness,Nat.add(next,rest)) == True{} : Bool}},hit),
        beyond => Empty.absurd({Maybe.is_some(&2,Nat,TiledLayout.scan(~fits,1n+rest,False{},shape,kept,axis)) == True{} : Bool},
          scan_absurd(~fits,shape,kept,axis,witness,classified,Order.antisymmetric(witness,axis,Order.not_less(axis,witness,beyond),after),hit)))

def preferred_some(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,found: Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,+preferred: Nat,
  +witness: Nat,+within: {Nat.is_le(witness,List.length(&2,U32,shape)) == True{} : Bool},+hit: {fits(Tiled.around(shape,witness,1),kept) == True{} : Bool}) ->
  {Maybe.is_some(&2,Nat,TiledLayout.preferred_found(~fits,found,shape,kept,preferred)) == True{} : Bool}:
  match found:
    case True{}: {==}
    case False{}:
      scan_some(~fits,List.length(&2,U32,shape),fits(Tiled.around(shape,0n,1),kept),shape,kept,0n,{==},witness,Order.zero_le(witness),within,hit)

def found_some(~fits: @found: Maybe<&2,Tiled.Around> -> @+kept: TiledLayout.Kept -> Bool,+shape: List<&2,U32>,+kept: TiledLayout.Kept,+witness: Nat,
  +within: {Nat.is_le(witness,List.length(&2,U32,shape)) == True{} : Bool},+hit: {fits(Tiled.around(shape,witness,1),kept) == True{} : Bool}) ->
  {Maybe.is_some(&2,Nat,TiledLayout.axis_fitting(~fits,shape,kept)) == True{} : Bool}:
  +preferred = Tiled.lane_axis(List.length(&2,U32,shape))
  preferred_some(~fits,fits(Tiled.around(shape,preferred,1),kept),shape,kept,preferred,witness,within,hit)

# The values before the unit axis are all of them.
def prefix_all(+values: List<&2,U32>) -> {Public.lane_prefix(values,List.length(&2,U32,values),1) == Grid.product(values) : U32}:
  +rank = List.length(&2,U32,values)
  +past = Order.not_below(rank,rank,Order.reflexive(rank))
  +prefix = Public.lane_prefix(values,rank,1)
  %Arithmetic.multiply_one(prefix) : {_ == Grid.product(values) : U32}
  %Public.after_past(values,rank,past) : {(prefix * Tiled.product(_) : U32) == Grid.product(values) : U32}
  %Public.removed_past(values,rank,past) : {(prefix * Tiled.product(Public.lane_after(values,rank)) : U32) == Grid.product(_) : U32}
  %Public.one_times(Grid.product(Grid.removed(values,rank))) : {(prefix * Tiled.product(Public.lane_after(values,rank)) : U32) == _ : U32}
  Public.prefix_after(values,rank,1)

def prefix_count(+limits: Machine.Limits,+values: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Public.lane_prefix(values,List.length(&2,U32,values),1)) == ShapeLaws.elements(values) : Nat}:
  %Equal.sym(U32,Public.lane_prefix(values,List.length(&2,U32,values),1),Grid.product(values),prefix_all(values)) : {U32.to_nat(_) == ShapeLaws.elements(values) : Nat}
  Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),values,bound)

def triple_true(+first: Bool,+second: Bool,+third: Bool,+first_true: {first == True{} : Bool},+second_true: {second == True{} : Bool},+third_true: {third == True{} : Bool}) ->
  {first && second && third == True{} : Bool}:
  %Equal.sym(Bool,first,True{},first_true) : {_ && second && third == True{} : Bool}
  %Equal.sym(Bool,second,True{},second_true) : {True{} && _ && third == True{} : Bool}
  %Equal.sym(Bool,third,True{},third_true) : {True{} && True{} && _ == True{} : Bool}
  {==}

# The unit axis of the new extents keeps the lanes on the unit axis of the
# source and the axes around them.
def unit_fits(+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+innermost: Bool,+source: Source<limits,values,shape>) ->
  {TiledLayout.same_around(Tiled.around(shape,List.length(&2,U32,shape),1),kept_of(values,List.length(&2,U32,values),innermost)) == True{} : Bool}:
  Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
  +rank = List.length(&2,U32,values)
  +other_rank = List.length(&2,U32,shape)
  +past = Order.not_below(rank,rank,Order.reflexive(rank))
  +other_past = Order.not_below(other_rank,other_rank,Order.reflexive(other_rank))
  +prefix = Public.lane_prefix(values,rank,1)
  +other_prefix = Public.lane_prefix(shape,other_rank,1)
  +prefixes = Arithmetic.to_nat_injective(other_prefix,prefix,
    Equal.trans(Nat,U32.to_nat(other_prefix),ShapeLaws.elements(shape),U32.to_nat(prefix),prefix_count(limits,shape,shape_bound),
      Equal.trans(Nat,ShapeLaws.elements(shape),ShapeLaws.elements(values),U32.to_nat(prefix),Equal.sym(Nat,ShapeLaws.elements(values),ShapeLaws.elements(shape),elements),
        Equal.sym(Nat,U32.to_nat(prefix),ShapeLaws.elements(values),prefix_count(limits,values,bound)))))
  +outer = U32.is_eq(other_prefix,prefix)
  +lanes = U32.is_eq(Grid.lanes_at(shape,other_rank),Grid.lanes_at(values,rank))
  +inner = U32.is_eq(Tiled.product(Public.lane_after(shape,other_rank)),Tiled.product(Public.lane_after(values,rank)))
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,other_rank,1),Some{Tiled.Around{other_prefix,Grid.lanes_at(shape,other_rank),Public.lane_after(shape,other_rank)}},
      Public.located(shape,other_rank,1,Order.reflexive(other_rank))) :
    {TiledLayout.same_around(_,kept_of(values,rank,innermost)) == True{} : Bool}
  triple_true(outer,lanes,inner,
    {%Equal.sym(U32,other_prefix,prefix,prefixes) : {U32.is_eq(_,prefix) == True{} : Bool} Arithmetic.equal_reflexive(prefix) : {outer == True{} : Bool}},
    {%Equal.sym(U32,Grid.lanes_at(shape,other_rank),1,Public.lanes_past(shape,other_rank,other_past)) : {U32.is_eq(_,Grid.lanes_at(values,rank)) == True{} : Bool}
      %Equal.sym(U32,Grid.lanes_at(values,rank),1,Public.lanes_past(values,rank,past)) : {U32.is_eq(1,_) == True{} : Bool} {==} : {lanes == True{} : Bool}},
    {%Equal.sym(List<&2,U32>,Public.lane_after(shape,other_rank),[],Public.after_past(shape,other_rank,other_past)) :
        {U32.is_eq(Tiled.product(_),Tiled.product(Public.lane_after(values,rank))) == True{} : Bool}
      %Equal.sym(List<&2,U32>,Public.lane_after(values,rank),[],Public.after_past(values,rank,past)) : {U32.is_eq(Tiled.product([]),Tiled.product(_)) == True{} : Bool}
      {==} : {inner == True{} : Bool}})

# The axis the lanes move to: the unit axis, or an axis inside the source
# whose lanes and the axes around them an axis of the new extents keeps.
type Matched<-values: List<&2,U32>,-shape: List<&2,U32>,-innermost: Bool,-axis: Nat> is Data:
  Past{past: {axis == List.length(&2,U32,values) : Nat}}
  Keeps{inside: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
    keeps: {TiledLayout.kept_around(Tiled.around(values,axis,1),innermost,shape) == True{} : Bool}}

def picked_matched(chosen: Bool,+values: List<&2,U32>,+shape: List<&2,U32>,+innermost: Bool,+own: Nat,+extent: U32,+index: Nat,+best: Nat,
  +classified: {Bool.not(Nat.is_eq(index,own)) && U32.is_le(extent,Grid.lanes_at(values,index)) && TiledLayout.kept_around(Tiled.around(values,index,1),innermost,shape) ==
    chosen : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,values)) == True{} : Bool},+matched: Matched<values,shape,innermost,best>) ->
  Matched<values,shape,innermost,Bool.pick(Nat,chosen,index,best)>:
  match chosen:
    case True{}:
      Keeps{inside,Boolean.right(Bool.not(Nat.is_eq(index,own)) && U32.is_le(extent,Grid.lanes_at(values,index)),
        TiledLayout.kept_around(Tiled.around(values,index,1),innermost,shape),classified)}
    case False{}: matched

def matched_facts(remaining: Nat,+index: Nat,+values: List<&2,U32>,+shape: List<&2,U32>,+own: Nat,+innermost: Bool,+best: Nat,+extent: U32,
  +matched: Matched<values,shape,innermost,best>,+room: {Nat.add(index,remaining) == List.length(&2,U32,values) : Nat}) ->
  Matched<values,shape,innermost,TiledLayout.matched_axis(remaining,index,values,shape,own,innermost,best,extent)>:
  match remaining:
    case 0n: matched
    case 1n+ +rest:
      +rank = List.length(&2,U32,values)
      +axis_extent = Grid.lanes_at(values,index)
      +fits = Bool.not(Nat.is_eq(index,own)) && U32.is_le(extent,axis_extent) && TiledLayout.kept_around(Tiled.around(values,index,1),innermost,shape)
      matched_facts(rest,1n+index,values,shape,own,innermost,Bool.pick(Nat,fits,index,best),Bool.pick(U32,fits,axis_extent,extent),
        picked_matched(fits,values,shape,innermost,own,extent,index,best,{==},Order.below_complement(index,rest,rank,room),matched),
        Equal.trans(Nat,1n+Nat.add(index,rest),Nat.add(index,1n+rest),rank,
          Equal.sym(Nat,Nat.add(index,1n+rest),1n+Nat.add(index,rest),Addition.successor_right(index,rest)),room))

def matched_inside(+values: List<&2,U32>,+shape: List<&2,U32>,+innermost: Bool,+axis: Nat,+matched: Matched<values,shape,innermost,axis>) ->
  {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool}:
  match matched:
    case Past{+past}:
      %Equal.sym(Nat,axis,List.length(&2,U32,values),past) : {Nat.is_le(_,List.length(&2,U32,values)) == True{} : Bool}
      Order.reflexive(List.length(&2,U32,values))
    case Keeps{+inside,+keeps}:
      Order.transitive(axis,1n+axis,List.length(&2,U32,values),Order.successor(axis,axis,Order.reflexive(axis)),Order.strict_follow(axis,List.length(&2,U32,values),inside))

# After the lanes move, the search finds an axis that keeps them.
def matched_some(+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+innermost: Bool,+axis: Nat,+matched: Matched<values,shape,innermost,axis>,
  +source: Source<limits,values,shape>) ->
  {Maybe.is_some(&2,Nat,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost))) == True{} : Bool}:
  match matched:
    case Past{+past}:
      %Equal.sym(Nat,axis,List.length(&2,U32,values),past) :
        {Maybe.is_some(&2,Nat,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,_,innermost))) == True{} : Bool}
      found_some(~TiledLayout.same_around,shape,kept_of(values,List.length(&2,U32,values),innermost),List.length(&2,U32,shape),Order.reflexive(List.length(&2,U32,shape)),
        unit_fits(limits,values,shape,innermost,source))
    case Keeps{+inside,+keeps}:
      +rank = List.length(&2,U32,values)
      {%Public.located(values,axis,1,Order.transitive(axis,1n+axis,rank,Order.successor(axis,axis,Order.reflexive(axis)),Order.strict_follow(axis,rank,inside))) :
          {TiledLayout.kept_around(_,innermost,shape) == True{} : Bool}
        keeps : {Maybe.is_some(&2,Nat,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,innermost))) == True{} : Bool}}

# ---- The lanes moved ----

def Reached(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+shape: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_again(TiledLayout.reshape_at(shape,tiled),values,shape))

def relaned_reached(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+axis: Nat,
  +position: Grid.Position,moved: Tensors.Tensor<Tile.Tile8>,+lanes: U32,+source: Source<limits,values,shape>,
  +some: {Maybe.is_some(&2,Nat,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position))))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{moved,axis,lanes,position})) ->
  Reached(~Logical,~original,logical,values,shape,Tiled.Tiled{moved,axis,lanes,position}):
  match moved:
    case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Usable{},+workers}:
      Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped_exact,+restored,+counted,+pinned}} = contract
      +kept = kept_of(values,axis,Bool.not(Grid.is_outermost(position)))
      %Equal.sym(Tiled.Attempt<Tiled.Tiled>,
          TiledLayout.reshape_at(shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
          TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept),kept,
            Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),
          at_relabelled(values,shape,pieces,cells_shape,fill,workers,axis,lanes,position,restored,inside,nonzero,count,values_counted,shape_counted)) :
        Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_again(_,values,shape))
      relabelled_reached(~Logical,~original,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept),logical,limits,values,shape,pieces,cells_shape,fill,workers,axis,lanes,
        position,search_found(~TiledLayout.same_around,shape,kept),some,
        Public.cells_shaped(~Logical,~original,logical,values,axis,lanes,position,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},contract),inside,pinned,
        Source{count,values_counted,shape_counted,nonzero,shape_nonzero,bound,shape_bound,elements,default_capacity})
    case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Reached(~Logical,~original,logical,values,shape,
        Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)

# A source no axis of the new extents keeps the lanes of moves its lanes to
# the matched axis and tries again.
def relaned_held(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,+own: Nat,
  tiled: Tiled.Tiled,+source: Source<limits,values,shape>,+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_relaned(values,shape,own,Tiled.position_of(tiled))):
  match tiled:
    case Tiled.Tiled{cells,+old,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Usable{},+workers}:
          Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
          +rebuilt = {Source{count,values_counted,shape_counted,nonzero,shape_nonzero,bound,shape_bound,elements,default_capacity} : Source<limits,values,shape>}
          +rank = List.length(&2,U32,values)
          +innermost = Bool.not(Grid.is_outermost(position))
          +axis = TiledLayout.matched_axis(rank,0n,values,shape,own,innermost,rank,2)
          +matched = matched_facts(rank,0n,values,shape,own,innermost,rank,2,Past{{==}},{==})
          Relane.along_usable(~Logical,~original,Reached(~Logical,~original,logical,values,shape),logical,values,pieces,cells_shape,fill,workers,old,lanes,position,axis,limits,
            default_capacity,matched_inside(values,shape,innermost,axis,matched),CountCompletion.positive_elements(values,nonzero),rank_room,
            Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},{==}),contract,
            moved => moved_lanes => moved_contract =>
              relaned_reached(~Logical,~original,logical,limits,values,shape,axis,position,moved,moved_lanes,rebuilt,matched_some(limits,values,shape,innermost,axis,matched,rebuilt),
                moved_contract))
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_relaned(values,shape,own,
            Tiled.position_of(Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},old,lanes,position}))),contract)

# A split of the lane axis the new extents find, or the lanes moved.
def split_or_held(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,TiledLayout.Splitting>,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,
  +shape: List<&2,U32>,pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +evidence: TiledSplit.ScanFound(found,values,axis,shape),+source: Source<limits,values,shape>,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Public.Contract(~Logical,~original,logical,shape,
    TiledLayout.split_or_relaned(found,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},values,shape,axis)):
  match found:
    case Some{TiledLayout.Splitting{+split_axis,+outer,+anchors,+width,+after}}:
      Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
      TiledSplit.split_reached(~Logical,~original,logical,limits,values,shape,pieces,cells_shape,fill,workers,axis,lanes,position,split_axis,outer,anchors,width,after,evidence,
        contract,bound,elements,CountCompletion.positive_elements(values,nonzero),rank_room,shape_rank,Global.Counted{count,shape_counted},default_capacity,
        Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},{==}))
    case None{}:
      relaned_held(~Logical,~original,logical,limits,values,shape,axis,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},source,
        rank_room,contract)

# Through whole groups, or with the lanes moved.
def regrouped_moved(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,Nat>,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,
  pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +searched: Searched(~TiledLayout.whole_groups,found,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},+source: Source<limits,values,shape>,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_moved(TiledLayout.regrouped(found,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),values,shape,axis)):
  match found:
    case Some{+other}:
      KernelGemm.Both{+other_inside,+fits} = searched
      %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,other,1),Some{Tiled.Around{Public.lane_prefix(shape,other,1),Grid.lanes_at(shape,other),Public.lane_after(shape,other)}},
          Public.located(shape,other,1,other_inside)) :
        Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_moved(TiledLayout.regrouped_around(_,other,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),values,shape,axis))
      regroup_contract(~Logical,~original,position,logical,limits,values,shape,axis,other,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},Public.cells_shaped(~Logical,~original,logical,values,axis,lanes,position,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},contract),
        inside,source,other_inside,fits)
    case None{}:
      split_or_held(~Logical,~original,TiledLayout.lane_split(Tiled.around(values,axis,1),shape),logical,limits,values,shape,pieces,cells_shape,fill,workers,axis,lanes,position,
        TiledSplit.lane_found(values,axis,shape,inside),source,rank_room,shape_rank,contract)

# Through an axis that keeps the lanes and the axes around them, through
# whole groups, or with the lanes moved.
def relabelled_moved(~Logical: Data,~original: Logical -> U32 -> F32,found: Maybe<&2,Nat>,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,
  pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +searched: Searched(~TiledLayout.same_around,found,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32},+source: Source<limits,values,shape>,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_moved(TiledLayout.relabelled(found,kept_of(values,axis,Bool.not(Grid.is_outermost(position))),Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),values,shape,axis)):
  match found:
    case Some{+other}:
      KernelGemm.Both{+other_inside,+fits} = searched
      Source{+count,+values_counted,+shape_counted,+nonzero,+shape_nonzero,+bound,+shape_bound,+elements,+default_capacity} = source
      relabel_contract(~Logical,~original,logical,limits,values,shape,axis,other,lanes,position,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},Public.cells_shaped(~Logical,~original,logical,values,axis,lanes,position,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},contract),
        inside,pinned,nonzero,shape_nonzero,bound,shape_bound,Global.Counted{count,shape_counted},other_inside,
        around_of(values,axis,Bool.not(Grid.is_outermost(position)),shape,other,other_inside,fits),default_capacity)
    case None{}:
      regrouped_moved(~Logical,~original,TiledLayout.axis_fitting(~TiledLayout.whole_groups,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),logical,limits,values,shape,pieces,cells_shape,fill,workers,axis,lanes,
        position,search_found(~TiledLayout.whole_groups,shape,kept_of(values,axis,Bool.not(Grid.is_outermost(position)))),inside,pinned,source,rank_room,shape_rank,contract)

# ---- Reshape ----

# Tiled values of a shape without a zero extent, under extents of the same
# count, read the runtime way: the tiled form of the new extents.
def reshaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+shape: List<&2,U32>,tiled: Tiled.Tiled,
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+same: {Tensors.count_of(shape) == Tensors.count_of(values) : Maybe<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape(tiled,shape)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{+global,Public.Laid{+inside,+shaped_exact,+restored,+counted,+pinned}} = contract
          +count = Global.count(values,counted)
          +values_counted = Global.accepted(values,counted)
          +shape_counted = Equal.trans(Maybe<&2,U32>,Tensors.count_of(shape),Tensors.count_of(values),Some{count},same,values_counted)
          +kept = kept_of(values,axis,Bool.not(Grid.is_outermost(position)))
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells_shape,axis,lanes,position),values,restored) :
            Public.Contract(~Logical,~original,logical,shape,
              TiledLayout.reshape_moved(TiledLayout.reshape_at(shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),_,shape,axis))
          %Equal.sym(Tiled.Attempt<Tiled.Tiled>,
              TiledLayout.reshape_at(shape,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
              TiledLayout.relabelled(TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept),kept,
                Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},shape),
              at_relabelled(values,shape,pieces,cells_shape,fill,workers,axis,lanes,position,restored,inside,nonzero,count,values_counted,shape_counted)) :
            Public.Contract(~Logical,~original,logical,shape,TiledLayout.reshape_moved(_,values,shape,axis))
          +source = source_of(limits,values,shape,counted,same,nonzero,default_capacity)
          relabelled_moved(~Logical,~original,TiledLayout.axis_fitting(~TiledLayout.same_around,shape,kept),logical,limits,values,shape,pieces,cells_shape,fill,workers,
            axis,lanes,position,search_found(~TiledLayout.same_around,shape,kept),inside,pinned,source,rank_room,shape_rank,contract)
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~Logical,~original,logical,shape,
            TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},axis,lanes,position},shape)),contract)

# Extents with two axes appended have one rank, whatever the two axes.
def tail_length(front: List<&2,U32>,+first: U32,+second: U32,+third: U32,+fourth: U32) ->
  {List.length(&2,U32,List.append(&2,U32,front,[first,second])) == List.length(&2,U32,List.append(&2,U32,front,[third,fourth])) : Nat}:
  match front:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,List.append(&2,U32,rest,[first,second])),List.length(&2,U32,List.append(&2,U32,rest,[third,fourth])),
        tail_length(rest,first,second,third,fourth))

# ---- Upsampling ----

# Upsampling keeps the rank.
def scaled_length(extents: List<&2,U32>,+scale: U32) -> {List.length(&2,U32,Layout.scaled_extents(extents,scale)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,rest}:
      match rest:
        case Nil{}: {==}
        case Con{+width,more}:
          match more:
            case Nil{}: {==}
            case Con{+next,+tail}:
              Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Layout.scaled_extents(width <> next <> tail,scale)),List.length(&2,U32,width <> next <> tail),
                scaled_length(width <> next <> tail,scale))


def upsampled_model(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+scale: U32,+axes: List<&2,Views.Axis>,
  +offset: U32,+model: PiecesModel.Model<Tile.Tile8>,+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +scaled: {Tensors.count_of(Layout.scaled_extents(values,scale)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
    TiledLayout.reshape(TiledLayout.gather_extents("upsample_nearest",Tensors.Usable{},Some{Views.View{axes,offset}},[],values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),Layout.scaled_extents(values,scale))):
  reshaped(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},limits,Views.axis_extents(axes),
    Layout.scaled_extents(values,scale),TiledLayout.gather_extents("upsample_nearest",Tensors.Usable{},Some{Views.View{axes,offset}},[],values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),target_nonzero,
    Equal.trans(Maybe<&2,U32>,Tensors.count_of(Layout.scaled_extents(values,scale)),Some{count},Tensors.count_of(Views.axis_extents(axes)),scaled,
      Equal.sym(Maybe<&2,U32>,Tensors.count_of(Views.axis_extents(axes)),Some{count},counted)),
    target_rank,{%Equal.sym(Nat,List.length(&2,U32,Layout.scaled_extents(values,scale)),List.length(&2,U32,values),scaled_length(values,scale)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,Layout.scaled_extents(values,scale)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},default_capacity,
    TiledViewProof.gathered_extents(~Logical,~original,logical,limits,"upsample_nearest",values,axes,offset,[],
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position},nonzero,target_nonzero,default_capacity,
      rank_room,target_rank,count,counted,fits,contract))

def upsampled_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+scale: U32,+axes: List<&2,Views.Axis>,
  +offset: U32,-pieces: Pieces.Pieces(~Tile.Tile8),+cells_shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +scaled: {Tensors.count_of(Layout.scaled_extents(values,scale)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +usable: Global.Usable<Tile.Tile8,cells_shape,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers}>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
    TiledLayout.reshape(TiledLayout.gather_extents("upsample_nearest",Tensors.Usable{},Some{Views.View{axes,offset}},[],values,
      Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),Layout.scaled_extents(values,scale))):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  %equation : Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
      TiledLayout.reshape(TiledLayout.gather_extents("upsample_nearest",Tensors.Usable{},Some{Views.View{axes,offset}},[],values,Tiled.Tiled{_,axis,lanes,position}),
        Layout.scaled_extents(values,scale)))
  upsampled_model(~Logical,~original,logical,limits,values,scale,axes,offset,model,cells_shape,model_fill,model_workers,axis,lanes,position,nonzero,target_nonzero,
    default_capacity,rank_room,target_rank,count,counted,scaled,fits,
    {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,model_fill,Tensors.Usable{},model_workers},
        Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},equation) :
      Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{_,axis,lanes,position})
      contract : Public.Contract(~Logical,~original,logical,values,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells_shape,model_fill,Tensors.Usable{},model_workers},axis,lanes,position})})

# Nearest-neighbour upsampling of tiled values of a shape without a zero
# extent, read the runtime way: the values of the view that repeats each row
# and column, under the scaled extents.
def upsampled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+scale: U32,+axes: List<&2,Views.Axis>,
  +offset: U32,tiled: Tiled.Tiled,+valid: {Layout.upsample_valid(values,scale) == True{} : Bool},+planes: U32,+leading: {Layout.planes(values) == Some{planes} : Maybe<&2,U32>},
  +viewed: {Layout.repeated(Tensors.view_of(values),scale) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +nonzero: {Shapes.has_zero(values) == False{} : Bool},+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},
  +scaled: {Tensors.count_of(Layout.scaled_extents(values,scale)) == Some{count} : Maybe<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(values)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
    TiledLayout.upsample_nearest(tiled,scale)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Usable{},+workers}:
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells_shape,axis,lanes,position),values,
              TiledViewProof.contract_restored(~Logical,~original,logical,values,pieces,cells_shape,fill,workers,axis,lanes,position,contract)) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
              TiledLayout.upsample_checked(Layout.upsample_valid(_,scale),Layout.planes(_),_,scale,workers,
                Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}))
          %Equal.sym(Bool,Layout.upsample_valid(values,scale),True{},valid) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
              TiledLayout.upsample_checked(_,Layout.planes(values),values,scale,workers,
                Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}))
          %Equal.sym(Maybe<&2,U32>,Layout.planes(values),Some{planes},leading) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
              TiledLayout.upsample_checked(True{},_,values,scale,workers,
                Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}))
          %Equal.sym(Maybe<&2,Views.View>,Layout.repeated(Tensors.view_of(values),scale),Some{Views.View{axes,offset}},viewed) :
            Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},Layout.scaled_extents(values,scale),
              TiledLayout.reshape(TiledLayout.gather_extents("upsample_nearest",Tensors.Usable{},_,[],values,
                Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},axis,lanes,position}),Layout.scaled_extents(values,scale)))
          upsampled_usable(~Logical,~original,logical,limits,values,scale,axes,offset,pieces,cells_shape,fill,workers,axis,lanes,position,nonzero,target_nonzero,
            default_capacity,rank_room,target_rank,count,counted,scaled,fits,
            Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,cells_shape,fill,Tensors.Usable{},workers},{==}),contract)
        case Tensors.Tensor{pieces,+cells_shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),ViewGather.Viewing{logical,axes,offset,[]},
            Layout.scaled_extents(values,scale),
            TiledLayout.upsample_nearest(Tiled.Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},axis,lanes,position},scale)),contract)
