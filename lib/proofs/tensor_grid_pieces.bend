# The pieces of a grid in order: where each starts, and the first piece that
# starts at or after a cell. These are the grid laws the halo walk assumes.
import Base
import ../tensor_grid.bend as Grid
import ../tensor_view.bend as Views
import ../tensor_halo.bend as Halo
import ./u32_extrema.bend as Extrema
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./boolean_logic.bend as Logic
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_ceiling.bend as NatCeiling
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./u32_comparison.bend as Comparison
import ./nat_extrema.bend as NatExtrema
import ./natural_addition.bend as Addition
import ./binary_radix.bend as Radix
import ./word_multiplication.bend as Multiplication

# ---- Strides ----

# One walk gives the strides of extents and the product of them all.
def strided_exact(extents: List<&2,U32>) -> {Grid.strided(extents) == Grid.Strided{Grid.strides(extents),Grid.product(extents)} : Grid.Strided}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      %Equal.sym(Grid.Strided,Grid.strided(rest),Grid.Strided{Grid.strides(rest),Grid.product(rest)},strided_exact(rest)) :
        {Grid.stride_before(extent,_) == Grid.Strided{Grid.strides(Con{extent,rest}),(extent * Grid.product(rest) : U32)} : Grid.Strided}
      %Equal.sym(Grid.Strided,Grid.strided(rest),Grid.Strided{Grid.strides(rest),Grid.product(rest)},strided_exact(rest)) :
        {Grid.Strided{Con{Grid.product(rest),Grid.strides(rest)},(extent * Grid.product(rest) : U32)} ==
          Grid.Strided{Grid.strides_listed(Grid.stride_before(extent,_)),(extent * Grid.product(rest) : U32)} : Grid.Strided}
      {==}

# The stride of an axis is the product of the axes after it.
def strides_con(+extent: U32,+rest: List<&2,U32>) -> {Grid.strides(Con{extent,rest}) == Con{Grid.product(rest),Grid.strides(rest)} : List<&2,U32>}:
  %Equal.sym(Grid.Strided,Grid.strided(rest),Grid.Strided{Grid.strides(rest),Grid.product(rest)},strided_exact(rest)) :
    {Grid.strides_listed(Grid.stride_before(extent,_)) == Con{Grid.product(rest),Grid.strides(rest)} : List<&2,U32>}
  {==}

# Axes whose cells fit the target are one unit of all their cells.
def units_fit(extents: List<&2,U32>,+prefix: U32,+outer: U32,+extent: U32,+target: U32) ->
  {Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target) == Grid.Units{prefix,outer,extent,Grid.product(extents)} : Grid.Units}:
  match extents:
    case Nil{}:
      {==}
    case Con{+axis,Nil{}}:
      {==}
    case Con{+axis,Con{+next,+tail}}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},strides_con(axis,Con{next,tail})) :
        {Grid.units_from(Con{axis,Con{next,tail}},_,True{},prefix,outer,extent,target) == Grid.Units{prefix,outer,extent,(axis * Grid.product(Con{next,tail}) : U32)} : Grid.Units}
      {==}

# The grid of cell axes is the grid of their units.
def grid_units(+cells: List<&2,U32>,+target: U32) ->
  {Grid.grid(cells,target) == Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target) : Grid.Grid}:
  %Equal.sym(Grid.Strided,Grid.strided(cells),Grid.Strided{Grid.strides(cells),Grid.product(cells)},strided_exact(cells)) :
    {Grid.grid_strided(cells,target,_) == Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target) : Grid.Grid}
  {==}

# The units of cell axes, outermost first: the axes past the shortest prefix
# whose rest has at most target cells by their product.
def units_model(extents: List<&2,U32>,fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32) -> Grid.Units:
  match extents fits:
    case extents True{}: Grid.Units{prefix,outer,extent,Grid.product(extents)}
    case Nil{} False{}: Grid.Units{prefix,outer,extent,1}
    case Con{last,Nil{}} False{}: Grid.Units{prefix,outer,extent,last}
    case Con{axis,+rest} False{}: units_model(rest,U32.is_le(Grid.product(rest),target),(prefix + 1 : U32),(outer * extent : U32),axis,target)

def units_exact(extents: List<&2,U32>,fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32) ->
  {Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target) == units_model(extents,fits,prefix,outer,extent,target) : Grid.Units}:
  match extents fits:
    case +extents True{}:
      units_fit(extents,prefix,outer,extent,target)
    case Nil{} False{}:
      {==}
    case Con{+last,Nil{}} False{}:
      {==}
    case Con{+axis,Con{+next,+tail}} False{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},strides_con(axis,Con{next,tail})) :
        {Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target) == units_model(Con{axis,Con{next,tail}},False{},prefix,outer,extent,target) : Grid.Units}
      units_exact(Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target)

# The grid of cell axes is the grid of their model units.
def grid_model(+cells: List<&2,U32>,+target: U32) ->
  {Grid.grid(cells,target) == Grid.grid_of_units(units_model(cells,U32.is_le(Grid.product(cells),target),0,1,1,target),target) : Grid.Grid}:
  Equal.trans(Grid.Grid,Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),
    Grid.grid_of_units(units_model(cells,U32.is_le(Grid.product(cells),target),0,1,1,target),target),grid_units(cells,target),
    Equal.cong(Grid.Units,Grid.Grid,found => Grid.grid_of_units(found,target),Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),
      units_model(cells,U32.is_le(Grid.product(cells),target),0,1,1,target),units_exact(cells,U32.is_le(Grid.product(cells),target),0,1,1,target)))

# The coordinates of a cell index over cell axes, outermost first: each is
# the quotient by the cells of the axes after it.
def coordinates_model(extents: List<&2,U32>,+index: U32) -> List<&2,U32>:
  match extents:
    case Nil{}: []
    case Con{+extent,+rest}:
      +inner = U32.max(Grid.product(rest),1)
      (index / inner % U32.max(extent,1) : U32) <> coordinates_model(rest,(index % inner : U32))

def coordinates_exact(extents: List<&2,U32>,+index: U32) -> {Grid.coordinates(extents,index) == coordinates_model(extents,index) : List<&2,U32>}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},strides_con(extent,rest)) :
        {Grid.coordinates_by(Con{extent,rest},_,index) == coordinates_model(Con{extent,rest},index) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => (index / U32.max(Grid.product(rest),1) % U32.max(extent,1) : U32) <> tail,
        Grid.coordinates(rest,(index % U32.max(Grid.product(rest),1) : U32)),coordinates_model(rest,(index % U32.max(Grid.product(rest),1) : U32)),
        coordinates_exact(rest,(index % U32.max(Grid.product(rest),1) : U32)))

# The contiguous axes of extents, each stepping by the cells after it.
def dense_model(extents: List<&2,U32>) -> List<&2,Views.Axis>:
  match extents:
    case Nil{}: []
    case Con{+extent,+rest}: Views.Axis{extent,Grid.product(rest)} <> dense_model(rest)

def dense_exact(extents: List<&2,U32>) -> {Grid.dense_axes(extents) == dense_model(extents) : List<&2,Views.Axis>}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},strides_con(extent,rest)) :
        {Grid.axes_by(Con{extent,rest},_) == dense_model(Con{extent,rest}) : List<&2,Views.Axis>}
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => Views.Axis{extent,Grid.product(rest)} <> tail,Grid.dense_axes(rest),dense_model(rest),dense_exact(rest))

# The first contiguous axis steps by the cells after it.
def dense_con(+extent: U32,+rest: List<&2,U32>) ->
  {Grid.dense_axes(Con{extent,rest}) == Views.Axis{extent,Grid.product(rest)} <> Grid.dense_axes(rest) : List<&2,Views.Axis>}:
  %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},strides_con(extent,rest)) :
    {Grid.axes_by(Con{extent,rest},_) == Views.Axis{extent,Grid.product(rest)} <> Grid.dense_axes(rest) : List<&2,Views.Axis>}
  {==}

# ---- What the halo walk needs of the grid ----

# piece_from(index) is the first piece that starts at or after index.
def FromMonotone(+grid: Grid.Grid,+total: Nat) -> Type:
  @+lower: U32 -> @+upper: U32 -> @+bounded: {Nat.is_le(U32.to_nat(upper),total) == True{} : Bool} -> @+ordered: {Nat.is_le(U32.to_nat(lower),U32.to_nat(upper)) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat(Halo.piece_from(grid,lower)),U32.to_nat(Halo.piece_from(grid,upper))) == True{} : Bool}

def FromBounded(+grid: Grid.Grid) -> Type:
  @+index: U32 -> {U32.is_le(Halo.piece_from(grid,index),Grid.pieces_of(grid)) == True{} : Bool}

def StartsAfter(+grid: Grid.Grid,+total: Nat) -> Type:
  @+index: U32 -> @+bounded: {Nat.is_le(U32.to_nat(index),total) == True{} : Bool} -> @+inside: {Nat.is_lt(U32.to_nat(Halo.piece_from(grid,index)),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat(index),U32.to_nat(Grid.start(grid,Halo.piece_from(grid,index)))) == True{} : Bool}

def StartsBefore(+grid: Grid.Grid,+total: Nat) -> Type:
  @+index: U32 -> @+bounded: {Nat.is_le(U32.to_nat(index),total) == True{} : Bool} -> @+piece: U32 -> @+below: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Halo.piece_from(grid,index))) == True{} : Bool} ->
    {Nat.is_lt(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(index)) == True{} : Bool}

def false_unless(+value: Bool,absurd: {value == True{} : Bool} -> Empty) -> {value == False{} : Bool}:
  match value:
    case True{}: Empty.absurd({True{} == False{} : Bool},absurd({==}))
    case False{}: {==}

def maximum_kept(+left: U32,+right: U32,+order: {Nat.is_le(U32.to_nat(right),U32.to_nat(left)) == True{} : Bool}) -> {U32.max(left,right) == left : U32}:
  %Extrema.pick_maximum(left,right) : {_ == left : U32}
  %Equal.sym(Bool,U32.is_lt(left,right),False{},
    Equal.trans(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),False{},Comparison.lt(left,right),
      false_unless(Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),less => Held.lt_le_absurd(U32.to_nat(left),U32.to_nat(right),less,order)))) :
    {Bool.pick(U32,_,right,left) == left : U32}
  {==}

# The piece from a cell is never past the last piece.
def from_bounded(+grid: Grid.Grid) -> FromBounded(grid):
  +index => Extrema.minimum_right(Bool.pick(U32,U32.is_lt(Grid.start(grid,Grid.piece_of(grid,index)),index),(Grid.piece_of(grid,index) + 1 : U32),Grid.piece_of(grid,index)),
    Grid.pieces_of(grid))

# ---- The block of a grid ----

# A block of units: at least one index, at most the extent or one.
type Block<-units: U32,-extent: U32> is Data:
  Block{positive: {Nat.is_lt(0n,U32.to_nat(units)) == True{} : Bool},below: {Nat.is_le(U32.to_nat(units),U32.to_nat(U32.max(extent,1))) == True{} : Bool}}

def block_positive(+units: U32,+extent: U32,+block: Block<units,extent>) -> {Nat.is_lt(0n,U32.to_nat(units)) == True{} : Bool}:
  Block{positive,below} = block
  positive

# The block of every grid made from units.
def grid_block(+unit_cells: U32,+extent: U32,+target: U32) -> Block<Grid.piece_units(unit_cells,extent,target),extent>:
  +fitted = U32.min(extent,(target / U32.max(unit_cells,1) : U32))
  +units = U32.max(fitted,1)
  Block{Bounds.nat_trans(0n,1n,U32.to_nat(units),{==},Bounds.as_nat_le(1,units,Extrema.maximum_right(fitted,1))),
    Bounds.as_nat_le(units,U32.max(extent,1),Extrema.maximum_upper(fitted,1,U32.max(extent,1),
      {%Equal.sym(Bool,U32.is_le(fitted,U32.max(extent,1)),Nat.is_le(U32.to_nat(fitted),U32.to_nat(U32.max(extent,1))),Comparison.le(fitted,U32.max(extent,1))) : {_ == True{} : Bool}
        Order.transitive(U32.to_nat(fitted),U32.to_nat(extent),U32.to_nat(U32.max(extent,1)),
          Bounds.as_nat_le(fitted,extent,Extrema.minimum_left(extent,(target / U32.max(unit_cells,1) : U32))),Bounds.as_nat_le(extent,U32.max(extent,1),Extrema.maximum_left(extent,1))) :
        {U32.is_le(fitted,U32.max(extent,1)) == True{} : Bool}},
      Extrema.maximum_right(extent,1)))}

# ---- Words of a grid ----

def word_quotient(+limits: Machine.Limits,+value: U32,+divisor: U32,+small: {Nat.is_le(U32.to_nat(divisor),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}) -> {U32.to_nat((value / divisor : U32)) == Nat.div(U32.to_nat(value),U32.to_nat(divisor)) : Nat}:
  U32Division.divide(value,divisor,Machine.half_word(limits),LimitBounds.half_word_fits(limits),LimitBounds.double_bound(limits,U32.to_nat(divisor),small),positive)

def word_remainder(+limits: Machine.Limits,+value: U32,+divisor: U32,+small: {Nat.is_le(U32.to_nat(divisor),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}) -> {U32.to_nat((value % divisor : U32)) == Nat.mod(U32.to_nat(value),U32.to_nat(divisor)) : Nat}:
  U32Division.modulo(value,divisor,Machine.half_word(limits),LimitBounds.half_word_fits(limits),LimitBounds.double_bound(limits,U32.to_nat(divisor),small),positive)

# One more block for a remainder that is not zero.
def picked_with(+word: U32,+rest: Nat,+exact: {U32.to_nat(word) == rest : Nat},+zero: Bool,+is_zero: {U32.is_zero(word) == zero : Bool}) ->
  {U32.to_nat(Bool.pick(U32,zero,0,1)) == Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n) : Nat}:
  match zero:
    case True{}:
      %exact : {0n == Bool.pick(Nat,Nat.is_eq(_,0n),0n,1n) : Nat}
      %Equal.sym(U32,word,0,Arithmetic.word_zero(word,is_zero)) : {0n == Bool.pick(Nat,Nat.is_eq(U32.to_nat(_),0n),0n,1n) : Nat}
      {==}
    case False{}:
      %exact : {1n == Bool.pick(Nat,Nat.is_eq(_,0n),0n,1n) : Nat}
      %Order.positive_has_last(U32.to_nat(word),Arithmetic.word_nonzero_positive(word,is_zero)) :
        {1n == Bool.pick(Nat,Nat.is_eq(_,0n),0n,1n) : Nat}
      {==}

def picked(+word: U32,+rest: Nat,+exact: {U32.to_nat(word) == rest : Nat}) ->
  {U32.to_nat(Bool.pick(U32,U32.is_zero(word),0,1)) == Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n) : Nat}:
  picked_with(word,rest,exact,U32.is_zero(word),{==})

# The rounded up quotient of words is the ceiling of their values.
def rounded_word(+limits: Machine.Limits,+count: U32,+size: U32,+count_small: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +size_small: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(size)) == True{} : Bool}) ->
  {U32.to_nat((count / size + Bool.pick(U32,U32.is_zero((count % size : U32)),0,1) : U32)) == NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)) : Nat}:
  +quotient = Nat.div(U32.to_nat(count),U32.to_nat(size))
  +rest = Nat.mod(U32.to_nat(count),U32.to_nat(size))
  +whole = (count / size : U32)
  +more = Bool.pick(U32,U32.is_zero((count % size : U32)),0,1)
  +parts = Equal.trans(Nat,Nat.add(U32.to_nat(whole),U32.to_nat(more)),Nat.add(quotient,U32.to_nat(more)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(more)),U32.to_nat(whole),quotient,word_quotient(limits,count,size,size_small,positive)),
    Equal.trans(Nat,Nat.add(quotient,U32.to_nat(more)),Nat.add(quotient,Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)),
      Equal.cong(Nat,Nat,value => Nat.add(quotient,value),U32.to_nat(more),Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n),
        picked((count % size : U32),rest,word_remainder(limits,count,size,size_small,positive))),
      Equal.sym(Nat,NatCeiling.rounded(quotient,rest),Nat.add(quotient,Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n)),NatCeiling.rounded_choice(quotient,rest))))
  Equal.trans(Nat,U32.to_nat((whole + more : U32)),Nat.add(U32.to_nat(whole),U32.to_nat(more)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)),
    Arithmetic.add(whole,more,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
      Held.nat_le_moved(Nat.add(U32.to_nat(whole),U32.to_nat(more)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)),U32.to_nat(Machine.capacity(limits)),parts,
        Order.transitive(NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),
          NatCeiling.bounded(U32.to_nat(count),U32.to_nat(size),positive),count_small))),
    parts)

# The pieces along the last prefix axis: the ceiling of its extent over the block.
def divided_exact(+limits: Machine.Limits,+count: U32,+size: U32,+count_small: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +size_small: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(size)) == True{} : Bool}) ->
  {U32.to_nat(Grid.divided_up(count,size)) == NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)) : Nat}:
  %Equal.sym(U32,U32.max(size,1),size,maximum_kept(size,1,Order.positive_one(U32.to_nat(size),positive))) :
    {U32.to_nat((count / _ + Bool.pick(U32,U32.is_zero((count % _ : U32)),0,1) : U32)) == NatCeiling.ceiling(U32.to_nat(count),U32.to_nat(size)) : Nat}
  rounded_word(limits,count,size,count_small,size_small,positive)

# ---- Where pieces start ----

# The first cell of piece j of outer index q: image cells for each outer index
# and cells for each piece.
def place(+outer: Nat,+inner: Nat,+image: Nat,+cells: Nat) -> Nat:
  Nat.add(Nat.mul(outer,image),Nat.mul(inner,cells))

# One outer index past the pieces before the end of its image.
def next_outer(+outer: Nat,+size: Nat) -> {Nat.add(Nat.mul(outer,size),size) == Nat.mul(1n+outer,size) : Nat}:
  Order.commutative(Nat.mul(outer,size),size)

def place_before_with(+choice: Bool,+lower_outer: Nat,+lower_inner: Nat,+upper_outer: Nat,+upper_inner: Nat,+per: Nat,+image: Nat,+cells: Nat,
  +test: {Nat.is_lt(lower_outer,upper_outer) == choice : Bool},+cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},
  +last: {Nat.is_lt(Nat.mul(lower_inner,cells),image) == True{} : Bool},+upper_inside: {Nat.is_lt(upper_inner,per) == True{} : Bool},
  +before: {Nat.is_lt(Nat.add(Nat.mul(lower_outer,per),lower_inner),Nat.add(Nat.mul(upper_outer,per),upper_inner)) == True{} : Bool}) ->
  {Nat.is_lt(place(lower_outer,lower_inner,image,cells),place(upper_outer,upper_inner,image,cells)) == True{} : Bool}:
  match choice:
    case True{}:
      +end = Nat.add(Nat.mul(lower_outer,image),image)
      Bounds.nat_trans(place(lower_outer,lower_inner,image,cells),end,place(upper_outer,upper_inner,image,cells),
        Order.offset_lt(Nat.mul(lower_outer,image),Nat.mul(lower_inner,cells),image,last),
        Order.transitive(end,Nat.mul(upper_outer,image),place(upper_outer,upper_inner,image,cells),
          Held.nat_le_moved(end,Nat.mul(1n+lower_outer,image),Nat.mul(upper_outer,image),next_outer(lower_outer,image),
            RowMajor.mul_le(1n+lower_outer,upper_outer,image,Order.strict_follow(lower_outer,upper_outer,test))),
          Order.le_add(Nat.mul(upper_outer,image),Nat.mul(upper_inner,cells))))
    case False{}:
      +upper_end = Bounds.nat_trans(Nat.add(Nat.mul(lower_outer,per),lower_inner),Nat.add(Nat.mul(upper_outer,per),upper_inner),Nat.mul(1n+upper_outer,per),before,
        Bounds.strict_le(Nat.add(Nat.mul(upper_outer,per),upper_inner),Nat.mul(1n+upper_outer,per),
          %next_outer(upper_outer,per) :
            {Nat.is_lt(Nat.add(Nat.mul(upper_outer,per),upper_inner),_) == True{} : Bool}
          Order.offset_lt(Nat.mul(upper_outer,per),upper_inner,per,upper_inside)))
      +below_next = Division.lt_succ_le(lower_outer,upper_outer,RowMajor.cancel_multiply_lt(lower_outer,1n+upper_outer,per,
        Order.le_lt_trans(Nat.mul(lower_outer,per),Nat.add(Nat.mul(lower_outer,per),lower_inner),Nat.mul(1n+upper_outer,per),
          Order.le_add(Nat.mul(lower_outer,per),lower_inner),upper_end)))
      +same = Order.antisymmetric(lower_outer,upper_outer,below_next,Order.not_less(lower_outer,upper_outer,test))
      +inner_before = Order.cancel_offset_lt(Nat.mul(upper_outer,per),lower_inner,upper_inner,
        {%same : {Nat.is_lt(Nat.add(Nat.mul(_,per),lower_inner),Nat.add(Nat.mul(upper_outer,per),upper_inner)) == True{} : Bool} before :
          {Nat.is_lt(Nat.add(Nat.mul(upper_outer,per),lower_inner),Nat.add(Nat.mul(upper_outer,per),upper_inner)) == True{} : Bool}})
      %Equal.sym(Nat,lower_outer,upper_outer,same) : {Nat.is_lt(place(_,lower_inner,image,cells),place(upper_outer,upper_inner,image,cells)) == True{} : Bool}
      Order.offset_lt(Nat.mul(upper_outer,image),Nat.mul(lower_inner,cells),Nat.mul(upper_inner,cells),
        RowMajor.multiply_lt(lower_inner,upper_inner,cells,inner_before,cells_positive))

# A piece before another starts before it.
def place_before(+lower_outer: Nat,+lower_inner: Nat,+upper_outer: Nat,+upper_inner: Nat,+per: Nat,+image: Nat,+cells: Nat,
  +cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},+last: {Nat.is_lt(Nat.mul(lower_inner,cells),image) == True{} : Bool},
  +upper_inside: {Nat.is_lt(upper_inner,per) == True{} : Bool},
  +before: {Nat.is_lt(Nat.add(Nat.mul(lower_outer,per),lower_inner),Nat.add(Nat.mul(upper_outer,per),upper_inner)) == True{} : Bool}) ->
  {Nat.is_lt(place(lower_outer,lower_inner,image,cells),place(upper_outer,upper_inner,image,cells)) == True{} : Bool}:
  place_before_with(Nat.is_lt(lower_outer,upper_outer),lower_outer,lower_inner,upper_outer,upper_inner,per,image,cells,{==},cells_positive,last,upper_inside,before)

def true_unless(+value: Bool,absurd: {value == False{} : Bool} -> Empty) -> {value == True{} : Bool}:
  match value:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},absurd({==}))

# A piece before the ceiling of the extent over the block starts inside the extent.
def inner_last(+inner: Nat,+per: Nat,+extent: Nat,+block: Nat,+positive: {Nat.is_lt(0n,block) == True{} : Bool},
  +per_exact: {per == NatCeiling.ceiling(extent,block) : Nat},+inside: {Nat.is_lt(inner,per) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(inner,block),extent) == True{} : Bool}:
  true_unless(Nat.is_lt(Nat.mul(inner,block),extent),outside => Held.lt_le_absurd(inner,per,inside,
    Held.nat_le_moved(per,NatCeiling.ceiling(extent,block),inner,per_exact,
      NatCeiling.minimal(extent,block,inner,positive,Held.lt_false_le(Nat.mul(inner,block),extent,outside)))))

def within_with(+choice: Bool,+outer: Nat,+inner: Nat,+outers: Nat,+per: Nat,+extent: Nat,+block: Nat,+test: {Nat.is_lt(outer,outers) == choice : Bool},
  +per_positive: {Nat.is_lt(0n,per) == True{} : Bool},+last: {Nat.is_lt(Nat.mul(inner,block),extent) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(Nat.mul(outer,per),inner),Nat.mul(outers,per)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(outer,extent),Nat.mul(inner,block)),Nat.mul(outers,extent)) == True{} : Bool}:
  match choice:
    case True{}:
      Bounds.strict_le(Nat.add(Nat.mul(outer,extent),Nat.mul(inner,block)),Nat.mul(outers,extent),
        Bounds.nat_trans(Nat.add(Nat.mul(outer,extent),Nat.mul(inner,block)),Nat.add(Nat.mul(outer,extent),extent),Nat.mul(outers,extent),
          Order.offset_lt(Nat.mul(outer,extent),Nat.mul(inner,block),extent,last),
          Held.nat_le_moved(Nat.add(Nat.mul(outer,extent),extent),Nat.mul(1n+outer,extent),Nat.mul(outers,extent),next_outer(outer,extent),
            RowMajor.mul_le(1n+outer,outers,extent,Order.strict_follow(outer,outers,test)))))
    case False{}:
      +upper = RowMajor.cancel_multiply_le(outer,outers,per,per_positive,
        Order.transitive(Nat.mul(outer,per),Nat.add(Nat.mul(outer,per),inner),Nat.mul(outers,per),Order.le_add(Nat.mul(outer,per),inner),inside))
      +same = Order.antisymmetric(outer,outers,upper,Order.not_less(outer,outers,test))
      +empty_inner = Equal.trans(Bool,Nat.is_le(inner,0n),Nat.is_le(Nat.add(Nat.mul(outers,per),inner),Nat.add(Nat.mul(outers,per),0n)),True{},
        Equal.sym(Bool,Nat.is_le(Nat.add(Nat.mul(outers,per),inner),Nat.add(Nat.mul(outers,per),0n)),Nat.is_le(inner,0n),Order.le_offset(Nat.mul(outers,per),inner,0n)),
        Held.nat_le_right_moved(Nat.add(Nat.mul(outers,per),inner),Nat.add(Nat.mul(outers,per),0n),Nat.mul(outers,per),Addition.zero_right(Nat.mul(outers,per)),
          {%same : {Nat.is_le(Nat.add(Nat.mul(_,per),inner),Nat.mul(outers,per)) == True{} : Bool} inside :
            {Nat.is_le(Nat.add(Nat.mul(outers,per),inner),Nat.mul(outers,per)) == True{} : Bool}}))
      +zero = Order.antisymmetric(inner,0n,empty_inner,Order.zero_le(inner))
      %Equal.sym(Nat,outer,outers,same) : {Nat.is_le(Nat.add(Nat.mul(_,extent),Nat.mul(inner,block)),Nat.mul(outers,extent)) == True{} : Bool}
      %Equal.sym(Nat,inner,0n,zero) : {Nat.is_le(Nat.add(Nat.mul(outers,extent),Nat.mul(_,block)),Nat.mul(outers,extent)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(Nat.mul(outers,extent),0n),Nat.mul(outers,extent),Addition.zero_right(Nat.mul(outers,extent))) :
        {Nat.is_le(_,Nat.mul(outers,extent)) == True{} : Bool}
      Order.reflexive(Nat.mul(outers,extent))

# A piece at or before the last starts at or before the end of the cells.
def within(+outer: Nat,+inner: Nat,+outers: Nat,+per: Nat,+extent: Nat,+block: Nat,
  +per_positive: {Nat.is_lt(0n,per) == True{} : Bool},+last: {Nat.is_lt(Nat.mul(inner,block),extent) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(Nat.mul(outer,per),inner),Nat.mul(outers,per)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(outer,extent),Nat.mul(inner,block)),Nat.mul(outers,extent)) == True{} : Bool}:
  within_with(Nat.is_lt(outer,outers),outer,inner,outers,per,extent,block,{==},per_positive,last,inside)

# The word of a piece's first cell is its place: q outer indices and j pieces
# for a piece q * per + j.
def start_place(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32,
  +piece: U32,+q: Nat,+j: Nat,+per_positive: {Nat.is_lt(0n,U32.to_nat(per)) == True{} : Bool},
  +per_small: {Nat.is_le(U32.to_nat(per),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},
  +index: {U32.to_nat(piece) == Nat.add(Nat.mul(q,U32.to_nat(per)),j) : Nat},+inner_inside: {Nat.is_lt(j,U32.to_nat(per)) == True{} : Bool},
  +bound: {Nat.is_le(Nat.mul(Nat.add(Nat.mul(q,U32.to_nat(extent)),Nat.mul(j,U32.to_nat(block))),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},piece)) ==
    place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +whole = (piece / per : U32)
  +part = (piece % per : U32)
  +q_exact = Equal.trans(Nat,U32.to_nat(whole),Nat.div(U32.to_nat(piece),U32.to_nat(per)),q,word_quotient(limits,piece,per,per_small,per_positive),
    Equal.trans(Nat,Nat.div(U32.to_nat(piece),U32.to_nat(per)),Nat.div(Nat.add(Nat.mul(q,U32.to_nat(per)),j),U32.to_nat(per)),q,
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(per)),U32.to_nat(piece),Nat.add(Nat.mul(q,U32.to_nat(per)),j),index),Radix.quotient(q,j,U32.to_nat(per),per_positive,inner_inside)))
  +j_exact = Equal.trans(Nat,U32.to_nat(part),Nat.mod(U32.to_nat(piece),U32.to_nat(per)),j,word_remainder(limits,piece,per,per_small,per_positive),
    Equal.trans(Nat,Nat.mod(U32.to_nat(piece),U32.to_nat(per)),Nat.mod(Nat.add(Nat.mul(q,U32.to_nat(per)),j),U32.to_nat(per)),j,
      Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(per)),U32.to_nat(piece),Nat.add(Nat.mul(q,U32.to_nat(per)),j),index),Radix.remainder(q,j,U32.to_nat(per),per_positive,inner_inside)))
  +base = Nat.add(Nat.mul(q,U32.to_nat(extent)),Nat.mul(j,U32.to_nat(block)))
  +base_room = Order.transitive(base,Nat.mul(base,U32.to_nat(unit_cells)),U32.to_nat(cap),RowMajor.product_covers(base,U32.to_nat(unit_cells),unit_positive),bound)
  +offset = (part * block : U32)
  +offset_exact = Multiplication.bounded_multiply(part,block,cap,cap_ok,
    Held.nat_le_moved(Nat.mul(U32.to_nat(part),U32.to_nat(block)),Nat.mul(j,U32.to_nat(block)),U32.to_nat(cap),Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat(part),j,j_exact),
      Order.transitive(Nat.mul(j,U32.to_nat(block)),base,U32.to_nat(cap),Order.right_le(Nat.mul(q,U32.to_nat(extent)),Nat.mul(j,U32.to_nat(block))),base_room)))
  +offset_nat = Equal.trans(Nat,U32.to_nat(offset),Nat.mul(U32.to_nat(part),U32.to_nat(block)),Nat.mul(j,U32.to_nat(block)),offset_exact,
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat(part),j,j_exact))
  +parts = Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(extent)),U32.to_nat(offset)),Nat.add(Nat.mul(q,U32.to_nat(extent)),U32.to_nat(offset)),base,
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(offset)),U32.to_nat(whole),q,q_exact),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(q,U32.to_nat(extent)),value),U32.to_nat(offset),Nat.mul(j,U32.to_nat(block)),offset_nat))
  +sum = (whole * extent + offset : U32)
  +sum_nat = Equal.trans(Nat,U32.to_nat(sum),Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(extent)),U32.to_nat(offset)),base,
    Arithmetic.multiply_add_exact(whole,extent,offset,cap,cap_ok,
      Held.nat_le_moved(Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(extent)),U32.to_nat(offset)),base,U32.to_nat(cap),parts,base_room)),
    parts)
  +total_exact = Multiplication.bounded_multiply(sum,unit_cells,cap,cap_ok,
    Held.nat_le_moved(Nat.mul(U32.to_nat(sum),U32.to_nat(unit_cells)),Nat.mul(base,U32.to_nat(unit_cells)),U32.to_nat(cap),
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(unit_cells)),U32.to_nat(sum),base,sum_nat),bound))
  %Equal.sym(U32,U32.max(per,1),per,maximum_kept(per,1,Order.positive_one(U32.to_nat(per),per_positive))) :
    {U32.to_nat(((piece / _ * extent + piece % _ * block) * unit_cells : U32)) == place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}
  Equal.trans(Nat,U32.to_nat((sum * unit_cells : U32)),Nat.mul(U32.to_nat(sum),U32.to_nat(unit_cells)),place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),total_exact,
    Equal.trans(Nat,Nat.mul(U32.to_nat(sum),U32.to_nat(unit_cells)),Nat.mul(base,U32.to_nat(unit_cells)),place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(unit_cells)),U32.to_nat(sum),base,sum_nat),
      Equal.trans(Nat,Nat.mul(base,U32.to_nat(unit_cells)),Nat.add(Nat.mul(Nat.mul(q,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells))),
        place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
        RowMajor.distrib(Nat.mul(q,U32.to_nat(extent)),Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells)),
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.mul(q,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells))),
          Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells))),
          place(q,j,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
          Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(q,U32.to_nat(extent)),U32.to_nat(unit_cells)),
            Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(q,U32.to_nat(extent),U32.to_nat(unit_cells))),
          Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),value),Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells)),
            Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),RowMajor.assoc(j,U32.to_nat(block),U32.to_nat(unit_cells)))))))

# ---- The shape of a grid ----

# A grid of outer indices, an extent of the last prefix axis, units of cells,
# blocks of the extent and the pieces of each outer index, that fits the machine.
type Shape<-limits: Machine.Limits,-outer: U32,-extent: U32,-unit_cells: U32,-block: U32,-per: U32> is Data:
  Shape{outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},
    block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},block_below: {Nat.is_le(U32.to_nat(block),U32.to_nat(extent)) == True{} : Bool},
    per_exact: {U32.to_nat(per) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def shape_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  room

def shape_unit(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool}:
  Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  unit_positive

def shape_block(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool}:
  Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  block_positive

def shape_per(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat(per) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat}:
  Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  per_exact

def extent_positive(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  Shape{outer_positive,unit_positive,+block_positive,+block_below,per_exact,room} = shape
  Bounds.nat_trans(0n,U32.to_nat(block),U32.to_nat(extent),block_positive,block_below)

# The extent lies inside the cells of one outer index and inside all of them.
def extent_inside(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(extent),Nat.mul(U32.to_nat(outer),U32.to_nat(extent))) == True{} : Bool}:
  Shape{+outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  Held.nat_le_right_moved(U32.to_nat(extent),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(U32.to_nat(extent),U32.to_nat(outer)),RowMajor.multiply_commutative(U32.to_nat(outer),U32.to_nat(extent)),RowMajor.product_covers(U32.to_nat(extent),U32.to_nat(outer),outer_positive))

def image_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(U32.to_nat(extent),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),extent_inside(limits,outer,extent,unit_cells,block,per,shape)),shape_room(limits,outer,extent,unit_cells,block,per,shape))

def extent_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(extent),U32.to_nat(unit_cells),shape_unit(limits,outer,extent,unit_cells,block,per,shape)),
    image_room(limits,outer,extent,unit_cells,block,per,shape))

# The pieces of an outer index: at least one, at most the extent.
def per_below(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(per),U32.to_nat(extent)) == True{} : Bool}:
  Held.nat_le_moved(U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(extent),shape_per(limits,outer,extent,unit_cells,block,per,shape),
    NatCeiling.bounded(U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape)))

def per_positive(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(per)) == True{} : Bool}:
  +exact = shape_per(limits,outer,extent,unit_cells,block,per,shape)
  +covers = NatCeiling.covers(U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape))
  true_unless(Nat.is_lt(0n,U32.to_nat(per)),outside =>
    Held.lt_le_absurd(0n,U32.to_nat(extent),extent_positive(limits,outer,extent,unit_cells,block,per,shape),
      Held.nat_le_right_moved(U32.to_nat(extent),0n,Nat.mul(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(block)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),0n,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),
          Equal.trans(Nat,0n,U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),
            Order.antisymmetric(0n,U32.to_nat(per),Order.zero_le(U32.to_nat(per)),Held.lt_false_le(0n,U32.to_nat(per),outside)),exact)),
        covers)))

def per_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(per),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(per),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),per_below(limits,outer,extent,unit_cells,block,per,shape),extent_room(limits,outer,extent,unit_cells,block,per,shape))

# Every piece of the grid has a number below the capacity.
def pieces_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Held.nat_le_moved(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(per),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits)),RowMajor.multiply_commutative(U32.to_nat(outer),U32.to_nat(per)),
    Order.transitive(Nat.mul(U32.to_nat(per),U32.to_nat(outer)),Nat.mul(U32.to_nat(extent),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(U32.to_nat(per),U32.to_nat(extent),U32.to_nat(outer),per_below(limits,outer,extent,unit_cells,block,per,shape)),
      Held.nat_le_moved(Nat.mul(U32.to_nat(extent),U32.to_nat(outer)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(Machine.capacity(limits)),RowMajor.multiply_commutative(U32.to_nat(extent),U32.to_nat(outer)),
        Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
          RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),shape_unit(limits,outer,extent,unit_cells,block,per,shape)),
          shape_room(limits,outer,extent,unit_cells,block,per,shape)))))

# ---- The piece of a cell ----

# A cell value = q * image + r with r = j * cells + s lies in piece j of outer
# index q: at or after its place, before the next piece and the next outer index.
def place_le(+value: Nat,+q: Nat,+r: Nat,+j: Nat,+s: Nat,+image: Nat,+cells: Nat,+outer_exact: {value == Nat.add(Nat.mul(q,image),r) : Nat},+inner_exact: {r == Nat.add(Nat.mul(j,cells),s) : Nat}) -> {Nat.is_le(place(q,j,image,cells),value) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(q,image),Nat.add(Nat.mul(j,cells),s)),Equal.trans(Nat,value,Nat.add(Nat.mul(q,image),r),Nat.add(Nat.mul(q,image),Nat.add(Nat.mul(j,cells),s)),outer_exact,Equal.cong(Nat,Nat,rest => Nat.add(Nat.mul(q,image),rest),r,Nat.add(Nat.mul(j,cells),s),inner_exact))) :
    {Nat.is_le(place(q,j,image,cells),_) == True{} : Bool}
  Order.offset_le(Nat.mul(q,image),Nat.mul(j,cells),Nat.add(Nat.mul(j,cells),s),Order.le_add(Nat.mul(j,cells),s))

def below_next_inner(+value: Nat,+q: Nat,+r: Nat,+j: Nat,+s: Nat,+image: Nat,+cells: Nat,+outer_exact: {value == Nat.add(Nat.mul(q,image),r) : Nat},+inner_exact: {r == Nat.add(Nat.mul(j,cells),s) : Nat},+small: {Nat.is_lt(s,cells) == True{} : Bool}) -> {Nat.is_lt(value,place(q,1n+j,image,cells)) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(q,image),Nat.add(Nat.mul(j,cells),s)),Equal.trans(Nat,value,Nat.add(Nat.mul(q,image),r),Nat.add(Nat.mul(q,image),Nat.add(Nat.mul(j,cells),s)),outer_exact,Equal.cong(Nat,Nat,rest => Nat.add(Nat.mul(q,image),rest),r,Nat.add(Nat.mul(j,cells),s),inner_exact))) :
    {Nat.is_lt(_,place(q,1n+j,image,cells)) == True{} : Bool}
  Order.offset_lt(Nat.mul(q,image),Nat.add(Nat.mul(j,cells),s),Nat.mul(1n+j,cells),
    {%next_outer(j,cells) : {Nat.is_lt(Nat.add(Nat.mul(j,cells),s),_) == True{} : Bool}
      Order.offset_lt(Nat.mul(j,cells),s,cells,small) : {Nat.is_lt(Nat.add(Nat.mul(j,cells),s),Nat.mul(1n+j,cells)) == True{} : Bool}})

def below_next_outer(+value: Nat,+q: Nat,+r: Nat,+image: Nat,+cells: Nat,+outer_exact: {value == Nat.add(Nat.mul(q,image),r) : Nat},
  +rest_small: {Nat.is_lt(r,image) == True{} : Bool}) -> {Nat.is_lt(value,place(1n+q,0n,image,cells)) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(q,image),r),outer_exact) : {Nat.is_lt(_,place(1n+q,0n,image,cells)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.mul(1n+q,image),0n),Nat.mul(1n+q,image),Addition.zero_right(Nat.mul(1n+q,image))) :
    {Nat.is_lt(Nat.add(Nat.mul(q,image),r),_) == True{} : Bool}
  %next_outer(q,image) : {Nat.is_lt(Nat.add(Nat.mul(q,image),r),_) == True{} : Bool}
  Order.offset_lt(Nat.mul(q,image),r,image,rest_small)

def index_inside_with(+choice: Bool,+value: Nat,+q: Nat,+r: Nat,+j: Nat,+s: Nat,+image: Nat,+cells: Nat,+outers: Nat,+per: Nat,+outer_exact: {value == Nat.add(Nat.mul(q,image),r) : Nat},+inner_exact: {r == Nat.add(Nat.mul(j,cells),s) : Nat},+image_positive: {Nat.is_lt(0n,image) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},+inner_inside: {Nat.is_lt(j,per) == True{} : Bool},+inside: {Nat.is_le(value,Nat.mul(outers,image)) == True{} : Bool},+test: {Nat.is_lt(q,outers) == choice : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(q,per),j),Nat.mul(outers,per)) == True{} : Bool}:
  match choice:
    case True{}:
      Bounds.strict_le(Nat.add(Nat.mul(q,per),j),Nat.mul(outers,per),
        Bounds.nat_trans(Nat.add(Nat.mul(q,per),j),Nat.add(Nat.mul(q,per),per),Nat.mul(outers,per),Order.offset_lt(Nat.mul(q,per),j,per,inner_inside),
          Held.nat_le_moved(Nat.add(Nat.mul(q,per),per),Nat.mul(1n+q,per),Nat.mul(outers,per),next_outer(q,per),
            RowMajor.mul_le(1n+q,outers,per,Order.strict_follow(q,outers,test)))))
    case False{}:
      +front = Held.nat_le_moved(Nat.add(Nat.mul(q,image),r),value,Nat.mul(outers,image),Equal.sym(Nat,value,Nat.add(Nat.mul(q,image),r),outer_exact),inside)
      +upper = RowMajor.cancel_multiply_le(q,outers,image,image_positive,
        Order.transitive(Nat.mul(q,image),Nat.add(Nat.mul(q,image),r),Nat.mul(outers,image),Order.le_add(Nat.mul(q,image),r),front))
      +same = Order.antisymmetric(q,outers,upper,Order.not_less(q,outers,test))
      +rest_empty = Equal.trans(Bool,Nat.is_le(r,0n),Nat.is_le(Nat.add(Nat.mul(outers,image),r),Nat.add(Nat.mul(outers,image),0n)),True{},
        Equal.sym(Bool,Nat.is_le(Nat.add(Nat.mul(outers,image),r),Nat.add(Nat.mul(outers,image),0n)),Nat.is_le(r,0n),Order.le_offset(Nat.mul(outers,image),r,0n)),
        Held.nat_le_right_moved(Nat.add(Nat.mul(outers,image),r),Nat.add(Nat.mul(outers,image),0n),Nat.mul(outers,image),Addition.zero_right(Nat.mul(outers,image)),
          {%same : {Nat.is_le(Nat.add(Nat.mul(_,image),r),Nat.mul(outers,image)) == True{} : Bool} front :
            {Nat.is_le(Nat.add(Nat.mul(outers,image),r),Nat.mul(outers,image)) == True{} : Bool}}))
      +inner_empty = RowMajor.cancel_multiply_le(j,0n,cells,cells_positive,
        Order.transitive(Nat.mul(j,cells),r,0n,
          Held.nat_le_right_moved(Nat.mul(j,cells),r,Nat.add(Nat.mul(j,cells),s),inner_exact,Order.le_add(Nat.mul(j,cells),s)),rest_empty))
      +zero = Order.antisymmetric(j,0n,inner_empty,Order.zero_le(j))
      %Equal.sym(Nat,q,outers,same) : {Nat.is_le(Nat.add(Nat.mul(_,per),j),Nat.mul(outers,per)) == True{} : Bool}
      %Equal.sym(Nat,j,0n,zero) : {Nat.is_le(Nat.add(Nat.mul(outers,per),_),Nat.mul(outers,per)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(Nat.mul(outers,per),0n),Nat.mul(outers,per),Addition.zero_right(Nat.mul(outers,per))) :
        {Nat.is_le(_,Nat.mul(outers,per)) == True{} : Bool}
      Order.reflexive(Nat.mul(outers,per))

# The piece of a cell inside the grid is a piece of the grid or the end.
def index_inside(+value: Nat,+q: Nat,+r: Nat,+j: Nat,+s: Nat,+image: Nat,+cells: Nat,+outers: Nat,+per: Nat,+outer_exact: {value == Nat.add(Nat.mul(q,image),r) : Nat},+inner_exact: {r == Nat.add(Nat.mul(j,cells),s) : Nat},+image_positive: {Nat.is_lt(0n,image) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},+inner_inside: {Nat.is_lt(j,per) == True{} : Bool},+inside: {Nat.is_le(value,Nat.mul(outers,image)) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(q,per),j),Nat.mul(outers,per)) == True{} : Bool}:
  index_inside_with(Nat.is_lt(q,outers),value,q,r,j,s,image,cells,outers,per,outer_exact,inner_exact,image_positive,cells_positive,inner_inside,inside,{==})

def shape_below(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(block),U32.to_nat(extent)) == True{} : Bool}:
  Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  block_below

def image_positive(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}:
  Order.product_positive(U32.to_nat(extent),U32.to_nat(unit_cells),extent_positive(limits,outer,extent,unit_cells,block,per,shape),shape_unit(limits,outer,extent,unit_cells,block,per,shape))

def cells_positive(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == True{} : Bool}:
  Order.product_positive(U32.to_nat(block),U32.to_nat(unit_cells),shape_block(limits,outer,extent,unit_cells,block,per,shape),shape_unit(limits,outer,extent,unit_cells,block,per,shape))

def cells_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(U32.to_nat(block),U32.to_nat(extent),U32.to_nat(unit_cells),shape_below(limits,outer,extent,unit_cells,block,per,shape)),image_room(limits,outer,extent,unit_cells,block,per,shape))

def image_word(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat((extent * unit_cells : U32)) == Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)) : Nat}:
  Multiplication.bounded_multiply(extent,unit_cells,Machine.capacity(limits),LimitBounds.capacity_fits(limits),image_room(limits,outer,extent,unit_cells,block,per,shape))

def cells_word(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat((block * unit_cells : U32)) == Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)) : Nat}:
  Multiplication.bounded_multiply(block,unit_cells,Machine.capacity(limits),LimitBounds.capacity_fits(limits),cells_room(limits,outer,extent,unit_cells,block,per,shape))

# The piece of a cell inside its outer index is one of the pieces of that index.
def inner_below(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+r: Nat,+j: Nat,+s: Nat,+inner_exact: {r == Nat.add(Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),s) : Nat},
  +rest_small: {Nat.is_lt(r,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}) -> {Nat.is_lt(j,U32.to_nat(per)) == True{} : Bool}:
  +front = Held.nat_le_right_moved(Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),r,Nat.add(Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),s),inner_exact,Order.le_add(Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),s))
  +scaled = Order.le_lt_trans(Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),r,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),front,rest_small)
  +last = RowMajor.cancel_multiply_lt(Nat.mul(j,U32.to_nat(block)),U32.to_nat(extent),U32.to_nat(unit_cells),
    {%Equal.sym(Nat,Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells)),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),RowMajor.assoc(j,U32.to_nat(block),U32.to_nat(unit_cells))) : {Nat.is_lt(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool} scaled : {Nat.is_lt(Nat.mul(Nat.mul(j,U32.to_nat(block)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}})
  +covers = Held.nat_le_right_moved(U32.to_nat(extent),Nat.mul(U32.to_nat(per),U32.to_nat(block)),Nat.mul(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(block)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),shape_per(limits,outer,extent,unit_cells,block,per,shape)),
    NatCeiling.covers(U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape)))
  RowMajor.cancel_multiply_lt(j,U32.to_nat(per),U32.to_nat(block),Bounds.nat_trans(Nat.mul(j,U32.to_nat(block)),U32.to_nat(extent),Nat.mul(U32.to_nat(per),U32.to_nat(block)),last,covers))

def word_kept(+word: U32,+value: Nat,+exact: {U32.to_nat(word) == value : Nat},+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {U32.max(word,1) == word : U32}:
  maximum_kept(word,1,Order.positive_one(U32.to_nat(word),{%Equal.sym(Nat,U32.to_nat(word),value,exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}}))

# The piece of a cell inside the grid: q * per + j for q outer indices and j
# pieces before it.
def piece_word(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+pieces: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+index: U32,
  +bounded: {Nat.is_le(U32.to_nat(index),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool}) ->
  {U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},index)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) : Nat}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +image = (extent * unit_cells : U32)
  +cells = (block * unit_cells : U32)
  +image_exact = image_word(limits,outer,extent,unit_cells,block,per,shape)
  +cells_exact = cells_word(limits,outer,extent,unit_cells,block,per,shape)
  +image_small = Held.nat_le_moved(U32.to_nat(image),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(cap),image_exact,image_room(limits,outer,extent,unit_cells,block,per,shape))
  +cells_small = Held.nat_le_moved(U32.to_nat(cells),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(cap),cells_exact,cells_room(limits,outer,extent,unit_cells,block,per,shape))
  +image_pos = {%Equal.sym(Nat,U32.to_nat(image),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_exact) : {Nat.is_lt(0n,_) == True{} : Bool} image_positive(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_lt(0n,U32.to_nat(image)) == True{} : Bool}}
  +cells_pos = {%Equal.sym(Nat,U32.to_nat(cells),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_exact) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool}}
  +whole = (index / image : U32)
  +rest = (index % image : U32)
  +part = (rest / cells : U32)
  +q_exact = Equal.trans(Nat,U32.to_nat(whole),Nat.div(U32.to_nat(index),U32.to_nat(image)),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),word_quotient(limits,index,image,image_small,image_pos),
    Equal.cong(Nat,Nat,value => Nat.div(U32.to_nat(index),value),U32.to_nat(image),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_exact))
  +r_exact = Equal.trans(Nat,U32.to_nat(rest),Nat.mod(U32.to_nat(index),U32.to_nat(image)),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),word_remainder(limits,index,image,image_small,image_pos),
    Equal.cong(Nat,Nat,value => Nat.mod(U32.to_nat(index),value),U32.to_nat(image),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_exact))
  +j_exact = Equal.trans(Nat,U32.to_nat(part),Nat.div(U32.to_nat(rest),U32.to_nat(cells)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),word_quotient(limits,rest,cells,cells_small,cells_pos),
    Equal.trans(Nat,Nat.div(U32.to_nat(rest),U32.to_nat(cells)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(cells)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(cells)),U32.to_nat(rest),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),r_exact),
      Equal.cong(Nat,Nat,value => Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),value),U32.to_nat(cells),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_exact)))
  +outer_exact = DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))
  +inner_exact = DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape))
  +inner = inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),inner_exact,DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)))
  +inside = index_inside(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(outer),U32.to_nat(per),outer_exact,inner_exact,image_positive(limits,outer,extent,unit_cells,block,per,shape),cells_positive(limits,outer,extent,unit_cells,block,per,shape),inner,
    Held.nat_le_right_moved(U32.to_nat(index),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),bounded))
  +parts = Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(per)),U32.to_nat(part)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),U32.to_nat(part)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(per)),U32.to_nat(part)),U32.to_nat(whole),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),q_exact),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),value),U32.to_nat(part),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),j_exact))
  %Equal.sym(U32,U32.max(image,1),image,word_kept(image,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_exact,image_positive(limits,outer,extent,unit_cells,block,per,shape))) :
    {U32.to_nat((index / _ * per + index % _ / U32.max(cells,1) : U32)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) : Nat}
  %Equal.sym(U32,U32.max(cells,1),cells,word_kept(cells,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_exact,cells_positive(limits,outer,extent,unit_cells,block,per,shape))) :
    {U32.to_nat((index / image * per + index % image / _ : U32)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) : Nat}
  Equal.trans(Nat,U32.to_nat((whole * per + part : U32)),Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(per)),U32.to_nat(part)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),
    Arithmetic.multiply_add_exact(whole,per,part,cap,cap_ok,
      Held.nat_le_moved(Nat.add(Nat.mul(U32.to_nat(whole),U32.to_nat(per)),U32.to_nat(part)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),U32.to_nat(cap),parts,
        Order.transitive(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),inside,pieces_room(limits,outer,extent,unit_cells,block,per,shape)))),
    parts)

# ---- Pieces from a cell start at or after it ----

def successor_below_double(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.is_le(1n+value,Nat.double(value)) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_le(1n,0n) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +more: Order.le_double(more)

# The next word after a piece number inside the capacity.
def successor_word(+limits: Machine.Limits,+piece: U32,+bounded: {Nat.is_le(U32.to_nat(piece),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat((piece + 1 : U32)) == 1n+U32.to_nat(piece) : Nat}:
  +capacity_positive = Bounds.nat_trans(0n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),LimitBounds.seven_within_capacity(limits)))
  Equal.trans(Nat,U32.to_nat((piece + 1 : U32)),Nat.add(U32.to_nat(piece),1n),1n+U32.to_nat(piece),
    Arithmetic.add(piece,1,Machine.half_word(limits),LimitBounds.half_word_fits(limits),
      Held.nat_le_moved(Nat.add(U32.to_nat(piece),1n),1n+U32.to_nat(piece),U32.to_nat(Machine.half_word(limits)),Order.commutative(U32.to_nat(piece),1n),
        Order.transitive(1n+U32.to_nat(piece),1n+U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),bounded,
          Held.nat_le_right_moved(1n+U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),Nat.double(U32.to_nat(Machine.capacity(limits))),LimitBounds.doubled_capacity(limits),
            successor_below_double(U32.to_nat(Machine.capacity(limits)),capacity_positive))))),
    Order.commutative(U32.to_nat(piece),1n))

def minimum_kept_with(+choice: Bool,+left: U32,+right: U32,+test: {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == choice : Bool},
  +below: {Nat.is_lt(U32.to_nat(U32.min(left,right)),U32.to_nat(right)) == True{} : Bool}) -> {U32.min(left,right) == left : U32}:
  match choice:
    case True{}:
      Arithmetic.to_nat_injective(U32.min(left,right),left,
        Equal.trans(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),Extrema.minimum(left,right),
          Equal.trans(Nat,Nat.min(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),
            NatExtrema.minimum_choice(U32.to_nat(left),U32.to_nat(right)),
            {%Equal.sym(Bool,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),True{},test) : {Bool.pick(Nat,_,U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(left) : Nat} {==} :
              {Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(left) : Nat}})))
    case False{}:
      +at_right = Equal.trans(Nat,U32.to_nat(U32.min(left,right)),Nat.min(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),Extrema.minimum(left,right),
        Equal.trans(Nat,Nat.min(U32.to_nat(left),U32.to_nat(right)),Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)),U32.to_nat(right),
          NatExtrema.minimum_choice(U32.to_nat(left),U32.to_nat(right)),
          {%Equal.sym(Bool,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),False{},test) : {Bool.pick(Nat,_,U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(right) : Nat} {==} :
            {Bool.pick(Nat,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(right) : Nat}}))
      Empty.absurd({U32.min(left,right) == left : U32},Held.lt_le_absurd(U32.to_nat(right),U32.to_nat(right),
        {%at_right : {Nat.is_lt(_,U32.to_nat(right)) == True{} : Bool} below : {Nat.is_lt(U32.to_nat(right),U32.to_nat(right)) == True{} : Bool}},Order.reflexive(U32.to_nat(right))))

# A minimum below its right side is its left side.
def minimum_kept(+left: U32,+right: U32,+below: {Nat.is_lt(U32.to_nat(U32.min(left,right)),U32.to_nat(right)) == True{} : Bool}) -> {U32.min(left,right) == left : U32}:
  minimum_kept_with(Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),left,right,{==},below)

def pieces_word(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat((outer * per : U32)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}:
  Multiplication.bounded_multiply(outer,per,Machine.capacity(limits),LimitBounds.capacity_fits(limits),pieces_room(limits,outer,extent,unit_cells,block,per,shape))

# The piece after the piece of a cell starts after the cell: in the same outer
# index, or at the next one when the cell's piece is its last.
def next_start_with(+choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+index: U32,+bounded: {Nat.is_le(U32.to_nat(index),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool},+test: {Nat.is_lt(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per)) == choice : Bool},
  +next_exact: {U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) : Nat},+next_inside: {Nat.is_le(U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)))) == True{} : Bool}:
  match choice:
    case True{}:
      +bound = within(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(outer),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),per_positive(limits,outer,extent,unit_cells,block,per,shape),
        inner_last(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape),shape_per(limits,outer,extent,unit_cells,block,per,shape),test),
        Held.nat_le_moved(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Equal.sym(Nat,U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),next_exact),next_inside))
      +bound_cells = Order.transitive(Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(block))),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
        RowMajor.mul_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(block))),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),bound),shape_room(limits,outer,extent,unit_cells,block,per,shape))
      +start_exact = start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),per_positive(limits,outer,extent,unit_cells,block,per,shape),per_room(limits,outer,extent,unit_cells,block,per,shape),
        shape_unit(limits,outer,extent,unit_cells,block,per,shape),next_exact,test,bound_cells)
      Held.nat_le_right_moved(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32))),place(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),start_exact,
        Bounds.strict_le(U32.to_nat(index),place(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
          below_next_inner(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)))))
    case False{}:
      +full = Order.antisymmetric(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),Order.strict_follow(Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)))),Order.not_less(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),test))
      +outer_next = Equal.trans(Nat,U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),next_exact,
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),U32.to_nat(per)),Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),
          Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),value),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),full),
          Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),U32.to_nat(per)),Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),next_outer(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),
            Equal.sym(Nat,Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Addition.zero_right(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)))))))
      +bound = within(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n,U32.to_nat(outer),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),per_positive(limits,outer,extent,unit_cells,block,per,shape),extent_positive(limits,outer,extent,unit_cells,block,per,shape),
        Held.nat_le_moved(Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Equal.sym(Nat,U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),0n),outer_next),next_inside))
      +bound_cells = Order.transitive(Nat.mul(Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(0n,U32.to_nat(block))),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
        RowMajor.mul_le(Nat.add(Nat.mul(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(0n,U32.to_nat(block))),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),bound),shape_room(limits,outer,extent,unit_cells,block,per,shape))
      +start_exact = start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n,per_positive(limits,outer,extent,unit_cells,block,per,shape),per_room(limits,outer,extent,unit_cells,block,per,shape),
        shape_unit(limits,outer,extent,unit_cells,block,per,shape),outer_next,per_positive(limits,outer,extent,unit_cells,block,per,shape),bound_cells)
      Held.nat_le_right_moved(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32))),place(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),start_exact,
        Bounds.strict_le(U32.to_nat(index),place(1n+Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
          below_next_outer(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)))))

def starts_after_with(+choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+index: U32,+bounded: {Nat.is_le(U32.to_nat(index),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool},+test: {U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index) == choice : Bool},
  +inside: {Nat.is_lt(U32.to_nat(U32.min(Bool.pick(U32,choice,(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),(outer * per : U32))),U32.to_nat((outer * per : U32))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.min(Bool.pick(U32,choice,(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),(outer * per : U32))))) == True{} : Bool}:
  match choice:
    case True{}:
      +kept = minimum_kept((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32),inside)
      +next_below = {%kept : {Nat.is_lt(U32.to_nat(_),U32.to_nat((outer * per : U32))) == True{} : Bool} inside : {Nat.is_lt(U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),U32.to_nat((outer * per : U32))) == True{} : Bool}}
      +next_inside = Bounds.strict_le(U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
        {%pieces_word(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_lt(U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),_) == True{} : Bool} next_below : {Nat.is_lt(U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}})
      +piece_exact = piece_word(limits,outer,extent,unit_cells,block,per,count,prefix,(outer * per : U32),shape,index,bounded)
      +piece_room = Held.nat_le_moved(U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),U32.to_nat(Machine.capacity(limits)),piece_exact,
        Order.transitive(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),
          index_inside(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(outer),U32.to_nat(per),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),image_positive(limits,outer,extent,unit_cells,block,per,shape),cells_positive(limits,outer,extent,unit_cells,block,per,shape),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))),Held.nat_le_right_moved(U32.to_nat(index),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),bounded)),
          pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
      +next_exact = Equal.trans(Nat,U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),1n+U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),successor_word(limits,Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),piece_room),
        Equal.trans(Nat,1n+U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),1n+Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),
          Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),piece_exact),
          Equal.sym(Nat,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),1n+Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Addition.successor_right(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))))))
      %Equal.sym(U32,U32.min((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32)),(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),kept) : {Nat.is_le(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_))) == True{} : Bool}
      next_start_with(Nat.is_lt(1n+Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per)),limits,outer,extent,unit_cells,block,per,count,prefix,shape,index,bounded,{==},next_exact,next_inside)
    case False{}:
      +kept = minimum_kept(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),(outer * per : U32),inside)
      %Equal.sym(U32,U32.min(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),(outer * per : U32)),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),kept) : {Nat.is_le(U32.to_nat(index),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_))) == True{} : Bool}
      Order.not_less(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index),
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index)),U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),False{},
          Equal.sym(Bool,U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index)),Comparison.lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index)),test))

# The piece from a cell inside the grid starts at or after it.
def starts_after(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> StartsAfter(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))):
  +index => +bounded => +inside => starts_after_with(U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),limits,outer,extent,unit_cells,block,per,count,prefix,shape,index,bounded,{==},inside)

# ---- Pieces before the piece from a cell start before it ----

# The first cell of a piece of the grid, from its outer index and position.
def piece_start(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == place(Nat.div(U32.to_nat(piece),U32.to_nat(per)),Nat.mod(U32.to_nat(piece),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}:
  +exact = DivisionOrder.quotient_exact(U32.to_nat(piece),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +inner = DivisionOrder.remainder_small(U32.to_nat(piece),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +bound = within(Nat.div(U32.to_nat(piece),U32.to_nat(per)),Nat.mod(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(outer),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),per_positive(limits,outer,extent,unit_cells,block,per,shape),
    inner_last(Nat.mod(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape),shape_per(limits,outer,extent,unit_cells,block,per,shape),inner),
    Held.nat_le_moved(Nat.add(Nat.mul(Nat.div(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(piece),U32.to_nat(per))),U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(Nat.div(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(piece),U32.to_nat(per))),exact),inside))
  start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),piece,Nat.div(U32.to_nat(piece),U32.to_nat(per)),Nat.mod(U32.to_nat(piece),U32.to_nat(per)),per_positive(limits,outer,extent,unit_cells,block,per,shape),per_room(limits,outer,extent,unit_cells,block,per,shape),
    shape_unit(limits,outer,extent,unit_cells,block,per,shape),exact,inner,
    Order.transitive(Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(extent)),Nat.mul(Nat.mod(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(block))),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
      RowMajor.mul_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(extent)),Nat.mul(Nat.mod(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(block))),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),bound),shape_room(limits,outer,extent,unit_cells,block,per,shape)))

# Pieces of the grid start in their order.
def start_order(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+lower: U32,+upper: U32,+upper_inside: {Nat.is_le(U32.to_nat(upper),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(lower),U32.to_nat(upper)) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper))) == True{} : Bool}:
  +lower_inside = Order.transitive(U32.to_nat(lower),U32.to_nat(upper),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Bounds.strict_le(U32.to_nat(lower),U32.to_nat(upper),before),upper_inside)
  +lower_exact = DivisionOrder.quotient_exact(U32.to_nat(lower),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +upper_exact = DivisionOrder.quotient_exact(U32.to_nat(upper),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +lower_inner = DivisionOrder.remainder_small(U32.to_nat(lower),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +last = {%RowMajor.assoc(Nat.mod(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(block),U32.to_nat(unit_cells)) : {Nat.is_lt(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}
    RowMajor.multiply_lt(Nat.mul(Nat.mod(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(block)),U32.to_nat(extent),U32.to_nat(unit_cells),inner_last(Nat.mod(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape),shape_per(limits,outer,extent,unit_cells,block,per,shape),lower_inner),shape_unit(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_lt(Nat.mul(Nat.mod(U32.to_nat(lower),U32.to_nat(per)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}}
  +ordered = {%upper_exact : {Nat.is_lt(Nat.add(Nat.mul(Nat.div(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per))),_) == True{} : Bool}
    {%lower_exact : {Nat.is_lt(_,U32.to_nat(upper)) == True{} : Bool} before : {Nat.is_lt(Nat.add(Nat.mul(Nat.div(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per))),U32.to_nat(upper)) == True{} : Bool}} :
    {Nat.is_lt(Nat.add(Nat.mul(Nat.div(U32.to_nat(lower),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per))),Nat.add(Nat.mul(Nat.div(U32.to_nat(upper),U32.to_nat(per)),U32.to_nat(per)),Nat.mod(U32.to_nat(upper),U32.to_nat(per)))) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),place(Nat.div(U32.to_nat(lower),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,lower,lower_inside)) :
    {Nat.is_lt(_,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper)),place(Nat.div(U32.to_nat(upper),U32.to_nat(per)),Nat.mod(U32.to_nat(upper),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,upper,upper_inside)) :
    {Nat.is_lt(place(Nat.div(U32.to_nat(lower),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),_) == True{} : Bool}
  place_before(Nat.div(U32.to_nat(lower),U32.to_nat(per)),Nat.mod(U32.to_nat(lower),U32.to_nat(per)),Nat.div(U32.to_nat(upper),U32.to_nat(per)),Nat.mod(U32.to_nat(upper),U32.to_nat(per)),U32.to_nat(per),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape),last,
    DivisionOrder.remainder_small(U32.to_nat(upper),U32.to_nat(per),per_positive(limits,outer,extent,unit_cells,block,per,shape)),ordered)

def start_monotone_with(+choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+lower: U32,+upper: U32,+test: {Nat.is_lt(U32.to_nat(lower),U32.to_nat(upper)) == choice : Bool},
  +upper_inside: {Nat.is_le(U32.to_nat(upper),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(lower),U32.to_nat(upper)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper))) == True{} : Bool}:
  match choice:
    case True{}: Bounds.strict_le(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper)),start_order(limits,outer,extent,unit_cells,block,per,count,prefix,shape,lower,upper,upper_inside,test))
    case False{}:
      +same = Arithmetic.to_nat_injective(lower,upper,Order.antisymmetric(U32.to_nat(lower),U32.to_nat(upper),ordered,Order.not_less(U32.to_nat(lower),U32.to_nat(upper),test)))
      %same : {Nat.is_le(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_))) == True{} : Bool}
      Order.reflexive(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)))

def start_monotone(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+lower: U32,+upper: U32,+upper_inside: {Nat.is_le(U32.to_nat(upper),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(lower),U32.to_nat(upper)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper))) == True{} : Bool}:
  start_monotone_with(Nat.is_lt(U32.to_nat(lower),U32.to_nat(upper)),limits,outer,extent,unit_cells,block,per,count,prefix,shape,lower,upper,{==},upper_inside,ordered)

def starts_before_with(+choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+index: U32,+bounded: {Nat.is_le(U32.to_nat(index),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool},+test: {U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index) == choice : Bool},+piece: U32,
  +below: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(U32.min(Bool.pick(U32,choice,(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),(outer * per : U32)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(index)) == True{} : Bool}:
  match choice:
    case True{}:
      +piece_exact = piece_word(limits,outer,extent,unit_cells,block,per,count,prefix,(outer * per : U32),shape,index,bounded)
      +own_inside = Held.nat_le_moved(U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),piece_exact,index_inside(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(outer),U32.to_nat(per),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),image_positive(limits,outer,extent,unit_cells,block,per,shape),cells_positive(limits,outer,extent,unit_cells,block,per,shape),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))),Held.nat_le_right_moved(U32.to_nat(index),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),bounded)))
      +piece_room = Order.transitive(U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),own_inside,pieces_room(limits,outer,extent,unit_cells,block,per,shape))
      +next_le = Held.nat_le_right_moved(U32.to_nat(U32.min((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32))),1n+U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),Equal.sym(Nat,U32.to_nat((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32)),1n+U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),successor_word(limits,Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),piece_room)),
        Bounds.as_nat_le(U32.min((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32)),(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),Extrema.minimum_left((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32))))
      +piece_le = Division.lt_succ_le(U32.to_nat(piece),U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Bounds.nat_trans(U32.to_nat(piece),U32.to_nat(U32.min((Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index) + 1 : U32),(outer * per : U32))),1n+U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),below,next_le))
      Order.le_lt_trans(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index),start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),own_inside,piece_le),
        Equal.trans(Bool,Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index)),U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),True{},
          Equal.sym(Bool,U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index)),Comparison.lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index)),test))
    case False{}:
      +piece_exact = piece_word(limits,outer,extent,unit_cells,block,per,count,prefix,(outer * per : U32),shape,index,bounded)
      +own_inside = Held.nat_le_moved(U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),piece_exact,index_inside(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(outer),U32.to_nat(per),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),image_positive(limits,outer,extent,unit_cells,block,per,shape),cells_positive(limits,outer,extent,unit_cells,block,per,shape),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))),Held.nat_le_right_moved(U32.to_nat(index),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),bounded)))
      +piece_lt = Bounds.nat_trans(U32.to_nat(piece),U32.to_nat(U32.min(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),(outer * per : U32))),U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),below,
        Bounds.as_nat_le(U32.min(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),(outer * per : U32)),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),Extrema.minimum_left(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),(outer * per : U32))))
      +bound = within(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(outer),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),per_positive(limits,outer,extent,unit_cells,block,per,shape),inner_last(Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),shape_block(limits,outer,extent,unit_cells,block,per,shape),shape_per(limits,outer,extent,unit_cells,block,per,shape),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)))),index_inside(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(outer),U32.to_nat(per),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),image_positive(limits,outer,extent,unit_cells,block,per,shape),cells_positive(limits,outer,extent,unit_cells,block,per,shape),inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))),Held.nat_le_right_moved(U32.to_nat(index),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),bounded)))
      +own_start = start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),per_positive(limits,outer,extent,unit_cells,block,per,shape),per_room(limits,outer,extent,unit_cells,block,per,shape),
        shape_unit(limits,outer,extent,unit_cells,block,per,shape),piece_exact,inner_below(limits,outer,extent,unit_cells,block,per,shape,Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.remainder_small(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape))),
        Order.transitive(Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(block))),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
          RowMajor.mul_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(extent)),Nat.mul(Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(block))),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),bound),shape_room(limits,outer,extent,unit_cells,block,per,shape)))
      Bounds.nat_trans(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),U32.to_nat(index),start_order(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index),own_inside,piece_lt),
        Held.nat_le_moved(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index))),place(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),U32.to_nat(index),own_start,
          place_le(U32.to_nat(index),Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),DivisionOrder.quotient_exact(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive(limits,outer,extent,unit_cells,block,per,shape)),DivisionOrder.quotient_exact(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),cells_positive(limits,outer,extent,unit_cells,block,per,shape)))))

# Every piece before the piece from a cell inside the grid starts before it.
def starts_before(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> StartsBefore(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))):
  +index => +bounded => +piece => +below => starts_before_with(U32.is_lt(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),index),limits,outer,extent,unit_cells,block,per,count,prefix,shape,index,bounded,{==},piece,below)

# The piece from a cell moves forward with the cell: a piece from upper before
# the piece from lower would start before lower and at or after upper.
def monotone_from(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>,+lower: U32,+upper: U32,+bounded: {Nat.is_le(U32.to_nat(upper),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(lower),U32.to_nat(upper)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Halo.piece_from(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)),U32.to_nat(Halo.piece_from(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper))) == True{} : Bool}:
  +low = Halo.piece_from(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower)
  +high = Halo.piece_from(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},upper)
  Held.lt_false_le(U32.to_nat(high),U32.to_nat(low),
    false_unless(Nat.is_lt(U32.to_nat(high),U32.to_nat(low)),+less =>
      Held.lt_le_absurd(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},high)),U32.to_nat(lower),
        starts_before(limits,outer,extent,unit_cells,block,per,count,prefix,shape)(lower,Order.transitive(U32.to_nat(lower),U32.to_nat(upper),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),ordered,bounded),high,less),
        Order.transitive(U32.to_nat(lower),U32.to_nat(upper),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},high)),ordered,
          starts_after(limits,outer,extent,unit_cells,block,per,count,prefix,shape)(upper,bounded,Bounds.nat_trans(U32.to_nat(high),U32.to_nat(low),U32.to_nat((outer * per : U32)),less,
            Bounds.as_nat_le(low,(outer * per : U32),from_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lower))))))))

def from_monotone(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Shape<limits,outer,extent,unit_cells,block,per>) -> FromMonotone(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))):
  +lower => +upper => +bounded => +ordered => monotone_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,lower,upper,bounded,ordered)

# ---- The laws of a grid ----

# A grid made of pieces of units that fit the machine, and its cells.
type GridFacts<-limits: Machine.Limits,-grid: Grid.Grid> is Data:
  GridFacts{count: U32,prefix: U32,outer: U32,extent: U32,unit_cells: U32,block: U32,per: U32,
    shape: Shape<limits,outer,extent,unit_cells,block,per>,exact: {grid == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
    cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}}

def grid_cells(+grid: Grid.Grid) -> Nat:
  U32.to_nat(Grid.count_of(grid))

def grid_after(+limits: Machine.Limits,+grid: Grid.Grid,+facts: GridFacts<limits,grid>) -> StartsAfter(grid,grid_cells(grid)):
  GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : StartsAfter(_,grid_cells(_))
  %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells) : StartsAfter(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)
  starts_after(limits,outer,extent,unit_cells,block,per,count,prefix,shape)

def grid_before(+limits: Machine.Limits,+grid: Grid.Grid,+facts: GridFacts<limits,grid>) -> StartsBefore(grid,grid_cells(grid)):
  GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : StartsBefore(_,grid_cells(_))
  %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells) : StartsBefore(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)
  starts_before(limits,outer,extent,unit_cells,block,per,count,prefix,shape)

def grid_monotone(+limits: Machine.Limits,+grid: Grid.Grid,+facts: GridFacts<limits,grid>) -> FromMonotone(grid,grid_cells(grid)):
  GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : FromMonotone(_,grid_cells(_))
  %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells) : FromMonotone(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)
  from_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape)
