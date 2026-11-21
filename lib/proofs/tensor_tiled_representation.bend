# The representation of a tiled tensor. A geometry names the outer indices of
# the axes but the lane axis, the values after the lane axis, the lanes and
# the groups, and where the groups lie. Lane l of the cell of group g at outer
# index o holds the value of lane l + 8 * g of the lane axis at o, and +0 past
# the lanes. Operations on tiles state their contract with the values of this
# reader, and the values of the cells read back the logical values.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled_cells.bend as Cells
import ./binary_radix.bend as Radix
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_full_range.bend as FullRange
import ./word_encoding.bend as WordEncoding

# The F32 values of a logical reader at a context, in their geometry.
type Lifted<-Logical: Data> is Data:
  Lifted{logical: Logical,geometry: Cells.Cells}

# A count as a divisor: one when there is none.
def least_one(value: Nat) -> Nat:
  match value:
    case 0n: 1n
    case 1n+rest: 1n+rest

def outer_count(+geometry: Cells.Cells) -> Nat:
  Cells.Cells{+outer,after,lanes,groups,position} = geometry
  U32.to_nat(outer)

def after_count(+geometry: Cells.Cells) -> Nat:
  Cells.Cells{outer,+after,lanes,groups,position} = geometry
  least_one(U32.to_nat(after))

def lane_count(+geometry: Cells.Cells) -> Nat:
  Cells.Cells{outer,after,+lanes,groups,position} = geometry
  U32.to_nat(lanes)

def group_count(+geometry: Cells.Cells) -> Nat:
  Cells.Cells{outer,after,lanes,+groups,position} = geometry
  U32.to_nat(groups)

# The logical index of lane lane of the lane axis at an outer index: the outer
# index names the values before the lane axis and those after it.
def logical_index(+geometry: Cells.Cells,+outer: Nat,+lane: Nat) -> Nat:
  +after = after_count(geometry)
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,after),lane_count(geometry)),lane),after),Nat.mod(outer,after))

# The logical index of an outer index below blocks blocks of the after count
# and a lane below the lane count lies below blocks blocks of lanes by after.
def logical_below(+geometry: Cells.Cells,+outer: Nat,+lane: Nat,+blocks: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(blocks,after_count(geometry))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,lane_count(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(logical_index(geometry,outer,lane),Nat.mul(Nat.mul(blocks,lane_count(geometry)),after_count(geometry))) == True{} : Bool}:
  +after = after_count(geometry)
  +positive = Order.le_lt_trans(0n,outer,Nat.mul(blocks,after),Order.zero_le(outer),outer_inside)
  +after_positive = DivisionOrder.right_factor_positive(blocks,after,positive)
  RowMajor.radix(Nat.add(Nat.mul(Nat.div(outer,after),lane_count(geometry)),lane),Nat.mul(blocks,lane_count(geometry)),Nat.mod(outer,after),after,
    RowMajor.radix(Nat.div(outer,after),blocks,lane,lane_count(geometry),DivisionOrder.quotient_below(outer,blocks,after,after_positive,outer_inside),lane_inside),
    DivisionOrder.remainder_small(outer,after,after_positive))

# Lane lane of the cell of group group at an outer index.
def lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat) -> F32:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Bool.pick(F32,Nat.is_lt(channel,lane_count(geometry)),original(logical,U32.from_nat(logical_index(geometry,outer,channel))),0.0)

def cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat) -> Tile.Tile8:
  Tile.Tile{lane(~Logical,~original,logical,geometry,outer,group,0n),lane(~Logical,~original,logical,geometry,outer,group,1n),
    lane(~Logical,~original,logical,geometry,outer,group,2n),lane(~Logical,~original,logical,geometry,outer,group,3n),
    lane(~Logical,~original,logical,geometry,outer,group,4n),lane(~Logical,~original,logical,geometry,outer,group,5n),
    lane(~Logical,~original,logical,geometry,outer,group,6n),lane(~Logical,~original,logical,geometry,outer,group,7n)}

# The lane of a cell is the lane function there.
def cell_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+selected: Nat,
  +small: {Nat.is_le(selected,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(selected,cell(~Logical,~original,logical,geometry,outer,group)) == lane(~Logical,~original,logical,geometry,outer,group,selected) : F32}:
  match selected:
    case 0n: {==}
    case 1n+ +next1:
      match next1:
        case 0n: {==}
        case 1n+ +next2:
          match next2:
            case 0n: {==}
            case 1n+ +next3:
              match next3:
                case 0n: {==}
                case 1n+ +next4:
                  match next4:
                    case 0n: {==}
                    case 1n+ +next5:
                      match next5:
                        case 0n: {==}
                        case 1n+ +next6:
                          match next6:
                            case 0n: {==}
                            case 1n+ +next7:
                              match next7:
                                case 0n: {==}
                                case 1n+beyond:
                                  Empty.absurd({TileLaws.select_lane(8n+beyond,cell(~Logical,~original,logical,geometry,outer,group)) ==
                                    lane(~Logical,~original,logical,geometry,outer,group,8n+beyond) : F32},WordEncoding.false_true(small))

# ---- Where a cell lies ----

def outer_at(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: Nat.div(index,least_one(groups))
    case Grid.Outermost{}: Nat.mod(index,least_one(outer))

def group_at(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: Nat.mod(index,least_one(groups))
    case Grid.Outermost{}: Nat.div(index,least_one(outer))

def index_at(position: Grid.Position,+outer: Nat,+groups: Nat,+at: Nat,+group: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: Nat.add(Nat.mul(at,groups),group)
    case Grid.Outermost{}: Nat.add(Nat.mul(group,outer),at)

# The outer index and the group of a cell index.
def cell_outer(+geometry: Cells.Cells,+index: Nat) -> Nat:
  Cells.Cells{+outer,after,lanes,+groups,position} = geometry
  outer_at(position,U32.to_nat(outer),U32.to_nat(groups),index)

def cell_group(+geometry: Cells.Cells,+index: Nat) -> Nat:
  Cells.Cells{+outer,after,lanes,+groups,position} = geometry
  group_at(position,U32.to_nat(outer),U32.to_nat(groups),index)

# The cell index of an outer index and a group.
def cell_index(+geometry: Cells.Cells,+outer: Nat,+group: Nat) -> Nat:
  Cells.Cells{+outer_extent,after,lanes,+groups,position} = geometry
  index_at(position,U32.to_nat(outer_extent),U32.to_nat(groups),outer,group)

# The cell at a cell index.
def lifted(~Logical: Data,~original: Logical -> U32 -> F32,context: Lifted<Logical>,index: U32) -> Tile.Tile8:
  Lifted{+logical,+geometry} = context
  +number = U32.to_nat(index)
  cell(~Logical,~original,logical,geometry,cell_outer(geometry,number),cell_group(geometry,number))

def least_positive(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {least_one(value) == value : Nat}:
  match value:
    case 0n: Empty.absurd({least_one(0n) == 0n : Nat},WordEncoding.false_true(positive))
    case 1n+rest: {==}

def below_positive(+value: Nat,+bound: Nat,+below: {Nat.is_lt(value,bound) == True{} : Bool}) -> {Nat.is_lt(0n,bound) == True{} : Bool}:
  Order.le_lt_trans(0n,value,bound,Order.zero_le(value),below)

# Division of a whole number of steps and a remainder below the step.
def split_at(+quotient: Nat,+remainder: Nat,+divisor: Nat,+small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) ->
  {Nat.divmod(Nat.add(Nat.mul(quotient,divisor),remainder),least_one(divisor)) == (quotient,remainder) : Nat & Nat}:
  +positive = below_positive(remainder,divisor,small)
  %Equal.sym(Nat,least_one(divisor),divisor,least_positive(divisor,positive)) :
    {Nat.divmod(Nat.add(Nat.mul(quotient,divisor),remainder),_) == (quotient,remainder) : Nat & Nat}
  Division.unique(Nat.add(Nat.mul(quotient,divisor),remainder),divisor,quotient,remainder,{==},positive,small)

# A count as a divisor is positive.
def least_one_positive(+value: Nat) -> {Nat.is_lt(0n,least_one(value)) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def after_positive(+geometry: Cells.Cells) -> {Nat.is_lt(0n,after_count(geometry)) == True{} : Bool}:
  match geometry:
    case Cells.Cells{+outer,+after,+lanes,+groups,+position}: least_one_positive(U32.to_nat(after))

def placed_outer(position: Grid.Position,+outer: Nat,+groups: Nat,+at: Nat,+group: Nat,+at_inside: {Nat.is_lt(at,outer) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) -> {outer_at(position,outer,groups,index_at(position,outer,groups,at,group)) == at : Nat}:
  match position:
    case Grid.Innermost{}:
      Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(at,groups),group),least_one(groups)),(at,group),split_at(at,group,groups,group_inside))
    case Grid.Outermost{}:
      Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(group,outer),at),least_one(outer)),(group,at),split_at(group,at,outer,at_inside))

def placed_group(position: Grid.Position,+outer: Nat,+groups: Nat,+at: Nat,+group: Nat,+at_inside: {Nat.is_lt(at,outer) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) -> {group_at(position,outer,groups,index_at(position,outer,groups,at,group)) == group : Nat}:
  match position:
    case Grid.Innermost{}:
      Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(at,groups),group),least_one(groups)),(at,group),split_at(at,group,groups,group_inside))
    case Grid.Outermost{}:
      Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(group,outer),at),least_one(outer)),(group,at),split_at(group,at,outer,at_inside))

# A cell index below the cells names an outer index and a group below theirs.
def outer_inside(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool},
  +outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool}) ->
  {Nat.is_lt(outer_at(position,outer,groups,index),outer) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      %Equal.sym(Nat,least_one(groups),groups,least_positive(groups,groups_positive)) : {Nat.is_lt(Nat.div(index,_),outer) == True{} : Bool}
      DivisionOrder.quotient_below(index,outer,groups,groups_positive,%RowMajor.multiply_commutative(groups,outer) : {Nat.is_lt(index,_) == True{} : Bool} inside)
    case Grid.Outermost{}:
      %Equal.sym(Nat,least_one(outer),outer,least_positive(outer,outer_positive)) : {Nat.is_lt(Nat.mod(index,_),outer) == True{} : Bool}
      DivisionOrder.remainder_small(index,outer,outer_positive)

def group_inside(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool},
  +outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool}) ->
  {Nat.is_lt(group_at(position,outer,groups,index),groups) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      %Equal.sym(Nat,least_one(groups),groups,least_positive(groups,groups_positive)) : {Nat.is_lt(Nat.mod(index,_),groups) == True{} : Bool}
      DivisionOrder.remainder_small(index,groups,groups_positive)
    case Grid.Outermost{}:
      %Equal.sym(Nat,least_one(outer),outer,least_positive(outer,outer_positive)) : {Nat.is_lt(Nat.div(index,_),groups) == True{} : Bool}
      DivisionOrder.quotient_below(index,groups,outer,outer_positive,inside)

# A cell index of an outer index and a group below theirs lies below the cells.
def index_inside(position: Grid.Position,+outer: Nat,+groups: Nat,+at: Nat,+group: Nat,+at_inside: {Nat.is_lt(at,outer) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) -> {Nat.is_lt(index_at(position,outer,groups,at,group),Nat.mul(groups,outer)) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      %RowMajor.multiply_commutative(outer,groups) : {Nat.is_lt(Nat.add(Nat.mul(at,groups),group),_) == True{} : Bool}
      RowMajor.radix(at,outer,group,groups,at_inside,group_inside)
    case Grid.Outermost{}: RowMajor.radix(group,groups,at,outer,group_inside,at_inside)

# A cell index of the geometry decodes to its outer index and group.
def decoded(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+word: U32,
  +exact: {U32.to_nat(word) == cell_index(geometry,outer,group) : Nat},
  +outer_inside: {Nat.is_lt(outer,outer_count(geometry)) == True{} : Bool},+group_inside: {Nat.is_lt(group,group_count(geometry)) == True{} : Bool}) ->
  {lifted(~Logical,~original,Lifted{logical,geometry},word) == cell(~Logical,~original,logical,geometry,outer,group) : Tile.Tile8}:
  Cells.Cells{+outer_extent,+after,+lanes,+groups,+position} = geometry
  +extent = U32.to_nat(outer_extent)
  +count = U32.to_nat(groups)
  +at_exact = placed_outer(position,extent,count,outer,group,outer_inside,group_inside)
  +group_exact = placed_group(position,extent,count,outer,group,outer_inside,group_inside)
  +shape = {Cells.Cells{outer_extent,after,lanes,groups,position} : Cells.Cells}
  +index = index_at(position,extent,count,outer,group)
  +target = cell(~Logical,~original,logical,shape,outer,group)
  %Equal.sym(Nat,U32.to_nat(word),index,exact) :
    {cell(~Logical,~original,logical,shape,outer_at(position,extent,count,_),group_at(position,extent,count,U32.to_nat(word))) == target : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat(word),index,exact) :
    {cell(~Logical,~original,logical,shape,outer_at(position,extent,count,index),group_at(position,extent,count,_)) == target : Tile.Tile8}
  %Equal.sym(Nat,outer_at(position,extent,count,index),outer,at_exact) :
    {cell(~Logical,~original,logical,shape,_,group_at(position,extent,count,index)) == target : Tile.Tile8}
  %Equal.sym(Nat,group_at(position,extent,count,index),group,group_exact) : {cell(~Logical,~original,logical,shape,outer,_) == target : Tile.Tile8}
  {==}

# A valid lane of a cell index is the reader at the corresponding logical index.
def decoded_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+selected: Nat,+word: U32,
  +exact: {U32.to_nat(word) == cell_index(geometry,outer,group) : Nat},
  +outer_inside: {Nat.is_lt(outer,outer_count(geometry)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,group_count(geometry)) == True{} : Bool},
  +lane_small: {Nat.is_le(selected,7n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(selected,Nat.mul(group,8n)),lane_count(geometry)) == True{} : Bool}) ->
  {TileLaws.select_lane(selected,lifted(~Logical,~original,Lifted{logical,geometry},word)) ==
    original(logical,U32.from_nat(logical_index(geometry,outer,Nat.add(selected,Nat.mul(group,8n))))) : F32}:
  +value = original(logical,U32.from_nat(logical_index(geometry,outer,Nat.add(selected,Nat.mul(group,8n)))))
  +actual = lifted(~Logical,~original,Lifted{logical,geometry},word)
  +target = cell(~Logical,~original,logical,geometry,outer,group)
  %Equal.sym(Tile.Tile8,actual,target,decoded(~Logical,~original,logical,geometry,outer,group,word,exact,outer_inside,group_inside)) :
    {TileLaws.select_lane(selected,_) == value : F32}
  Equal.trans(F32,TileLaws.select_lane(selected,target),lane(~Logical,~original,logical,geometry,outer,group,selected),value,
    cell_lane(~Logical,~original,logical,geometry,outer,group,selected,lane_small),%Equal.sym(Bool,Nat.is_lt(Nat.add(selected,Nat.mul(group,8n)),lane_count(geometry)),True{},active) :
      {Bool.pick(F32,_,value,0.0) == value : F32}
    {==})

def logical_radix(+geometry: Cells.Cells,+prefix: Nat,+suffix: Nat,+selected: Nat,
  +small: {Nat.is_lt(suffix,after_count(geometry)) == True{} : Bool}) ->
  {logical_index(geometry,Nat.add(Nat.mul(prefix,after_count(geometry)),suffix),selected) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,lane_count(geometry)),selected),after_count(geometry)),suffix) : Nat}:
  +after = after_count(geometry)
  +outer = Nat.add(Nat.mul(prefix,after),suffix)
  +lanes = lane_count(geometry)
  +positive = after_positive(geometry)
  %Equal.sym(Nat,Nat.div(outer,after),prefix,Radix.quotient(prefix,suffix,after,positive,small)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,lanes),selected),after),Nat.mod(outer,after)) ==
      Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,lanes),selected),after),suffix) : Nat}
  %Equal.sym(Nat,Nat.mod(outer,after),suffix,Radix.remainder(prefix,suffix,after,positive,small)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,lanes),selected),after),_) ==
      Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,lanes),selected),after),suffix) : Nat}
  {==}

def last_axis_index(+geometry: Cells.Cells,+outer: Nat,+selected: Nat,+after_is_one: {after_count(geometry) == 1n : Nat}) ->
  {logical_index(geometry,outer,selected) == Nat.add(Nat.mul(outer,lane_count(geometry)),selected) : Nat}:
  +after = after_count(geometry)
  +lanes = lane_count(geometry)
  +col = Nat.add(Nat.mul(outer,lanes),selected)
  +other = Nat.add(Nat.mul(outer,after),0n)
  +exact = {%Equal.sym(Nat,after,1n,after_is_one) : {Nat.add(Nat.mul(outer,_),0n) == outer : Nat}
    %Equal.sym(Nat,Nat.add(Nat.mul(outer,1n),0n),Nat.mul(outer,1n),Addition.zero_right(Nat.mul(outer,1n))) : {_ == outer : Nat}
    Order.times_one(outer) : {other == outer : Nat}}
  %exact : {logical_index(geometry,_,selected) == col : Nat}
  Equal.trans(Nat,logical_index(geometry,other,selected),Nat.add(Nat.mul(col,after),0n),col,
    logical_radix(geometry,outer,0n,selected,%Equal.sym(Nat,after,1n,after_is_one) : {Nat.is_lt(0n,_) == True{} : Bool} {==}),
    %Equal.sym(Nat,after,1n,after_is_one) : {Nat.add(Nat.mul(col,_),0n) == col : Nat}
    %Equal.sym(Nat,Nat.add(Nat.mul(col,1n),0n),Nat.mul(col,1n),Addition.zero_right(Nat.mul(col,1n))) : {_ == col : Nat}
    Order.times_one(col))

# ---- Values read from cells ----

# The channel of a logical index and its outer index.
def value_channel(+geometry: Cells.Cells,+index: Nat) -> Nat:
  Nat.mod(Nat.div(index,after_count(geometry)),least_one(lane_count(geometry)))

def value_outer(+geometry: Cells.Cells,+index: Nat) -> Nat:
  +after = after_count(geometry)
  Nat.add(Nat.mul(Nat.div(Nat.div(index,after),least_one(lane_count(geometry))),after),Nat.mod(index,after))

# The channel of the logical index of a lane below the lane count is the lane.
def lane_channel(+geometry: Cells.Cells,+outer: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,lane_count(geometry)) == True{} : Bool}) ->
  {value_channel(geometry,logical_index(geometry,outer,lane)) == lane : Nat}:
  +positive = after_positive(geometry)
  +quotient = Nat.add(Nat.mul(Nat.div(outer,after_count(geometry)),lane_count(geometry)),lane)
  +divided = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(logical_index(geometry,outer,lane),after_count(geometry)),(quotient,Nat.mod(outer,after_count(geometry))),
    Division.unique(logical_index(geometry,outer,lane),after_count(geometry),quotient,Nat.mod(outer,after_count(geometry)),{==},positive,
      DivisionOrder.remainder_small(outer,after_count(geometry),positive)))
  Equal.trans(Nat,value_channel(geometry,logical_index(geometry,outer,lane)),Nat.mod(quotient,least_one(lane_count(geometry))),lane,
    Equal.cong(Nat,Nat,number => Nat.mod(number,least_one(lane_count(geometry))),Nat.div(logical_index(geometry,outer,lane),after_count(geometry)),quotient,divided),
    Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(quotient,least_one(lane_count(geometry))),(Nat.div(outer,after_count(geometry)),lane),
      split_at(Nat.div(outer,after_count(geometry)),lane,lane_count(geometry),lane_inside)))
