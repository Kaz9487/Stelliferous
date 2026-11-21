# The value of a target cell of a split: its spot reads the source cell of
# its first lane and, when a lane inside the width needs it, the next one;
# both joined at the shift and cut at the width give the target cell at its
# index.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_split_cell.bend as SplitCell
import ./tensor_split_spot.bend as SplitSpot
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as Ceiling
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic

# ---- The words of a spot ----

# The source cell a spot reads first: the cell of its first lane in the
# source groups from its base on.
def spot_read(+width: U32,+spot: Tiled.Spot) -> U32:
  Tiled.Spot{+anchor,+group,+base,+step} = spot
  +first = (anchor * width + group * 8 : U32)
  +cell = (first / 8 : U32)
  (base + cell * step : U32)

# The source cell a spot may read next, one step after the first.
def spot_high(+width: U32,+spot: Tiled.Spot) -> U32:
  Tiled.Spot{+anchor,+group,+base,+step} = spot
  +first = (anchor * width + group * 8 : U32)
  +cell = (first / 8 : U32)
  +source = (base + cell * step : U32)
  (source + 1 * step : U32)

# Whether a spot reads the next source cell: its lanes shift and another
# source cell follows.
def spot_further(+width: U32,+source_groups: U32,+spot: Tiled.Spot) -> Bool:
  Tiled.Spot{+anchor,+group,base,step} = spot
  +first = (anchor * width + group * 8 : U32)
  U32.is_lt(0,(first % 8 : U32)) && U32.is_lt(((first / 8 : U32) + 1 : U32),source_groups)

# Whether a lane of a spot inside the width needs the next source cell.
def spot_needed(+width: U32,+spot: Tiled.Spot) -> Bool:
  Tiled.Spot{+anchor,+group,base,step} = spot
  +first = (anchor * width + group * 8 : U32)
  SplitCell.needed(U32.to_nat(width),U32.to_nat(group),U32.to_nat((first % 8 : U32)))

# The cell a spot gives from the source cell it reads first and the next:
# both joined at the shift of its first lane, cut at the width.
def spot_cell(+width: U32,+spot: Tiled.Spot,+low: Tile.Tile8,+high: Tile.Tile8) -> Tile.Tile8:
  Tiled.Spot{+anchor,+group,base,step} = spot
  +first = (anchor * width + group * 8 : U32)
  Cells.tail(Cells.Tail{(group * 8 : U32),width},Tiled.joined((first % 8 : U32),low,high))

# What the next source cell of a spot that needs it is: read, below the
# source groups, at the source cell after the first.
type Next<-width: U32,-lanes: U32,-source_outer: U32,-position: Grid.Position,-spot: Tiled.Spot,-outer_index: Nat,-cell: Nat> is Data:
  Next{further: {spot_further(width,Tiled.groups(lanes),spot) == True{} : Bool},next_inside: {Nat.is_lt(1n+cell,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
    high_exact: {U32.to_nat(spot_high(width,spot)) == Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Tiled.groups(lanes)),outer_index,1n+cell) : Nat}}

# What the words of a spot read: the source cell of its first lane at an
# outer index and a group inside the source cells, and the next one when a
# lane needs it.
type Words<-width: U32,-lanes: U32,-source_outer: U32,-position: Grid.Position,-spot: Tiled.Spot,-outer_index: Nat,-cell: Nat> is Data:
  Words{read_exact: {U32.to_nat(spot_read(width,spot)) == Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Tiled.groups(lanes)),outer_index,cell) : Nat},
    outer_inside: {Nat.is_lt(outer_index,U32.to_nat(source_outer)) == True{} : Bool},cell_inside: {Nat.is_lt(cell,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
    next: Boolean.When(spot_needed(width,spot),Next<width,lanes,source_outer,position,spot,outer_index,cell>)}

# ---- The source cells as words ----

# The source cell of group 0 and the step between source groups, with the
# groups innermost or outermost.
def spot_base(position: Grid.Position,+outer_index: Nat,+source_groups: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: Nat.mul(outer_index,source_groups)
    case Grid.Outermost{}: outer_index

def spot_step(position: Grid.Position,+outers: Nat) -> U32:
  match position:
    case Grid.Innermost{}: 1
    case Grid.Outermost{}: U32.from_nat(outers)

def step_count(position: Grid.Position,+outers: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: 1n
    case Grid.Outermost{}: outers

# The source cells hold no more words than half a word.
def count_rooms(+limits: Machine.Limits,+outer_index: Nat,+outers: Nat,+source_groups: Nat,+group: Nat,
  +outer_inside: {Nat.is_lt(outer_index,outers) == True{} : Bool},+group_inside: {Nat.is_lt(group,source_groups) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(outer_index,source_groups),group),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(outer_index,source_groups),group),Nat.mul(outers,source_groups),U32.to_nat(Machine.half_word(limits)),
    Bounds.strict_le(Nat.add(Nat.mul(outer_index,source_groups),group),Nat.mul(outers,source_groups),RowMajor.radix(outer_index,outers,group,source_groups,outer_inside,group_inside)),
    %RowMajor.multiply_commutative(source_groups,outers) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool} count_room)

# The source cell a target cell reads first, as its word.
def source_word(position: Grid.Position,+limits: Machine.Limits,+base: U32,+step: U32,+cell_word: U32,+outer_index: Nat,+outers: Nat,+source_groups: Nat,+cell: Nat,
  +base_exact: {U32.to_nat(base) == spot_base(position,outer_index,source_groups) : Nat},+step_exact: {U32.to_nat(step) == step_count(position,outers) : Nat},
  +cell_exact: {U32.to_nat(cell_word) == cell : Nat},+outer_inside: {Nat.is_lt(outer_index,outers) == True{} : Bool},
  +cell_inside: {Nat.is_lt(cell,source_groups) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat((base + cell_word * step : U32)) == Representation.index_at(position,outers,source_groups,outer_index,cell) : Nat}:
  match position:
    case Grid.Innermost{}:
      +half = U32.to_nat(Machine.half_word(limits))
      +front = Nat.mul(outer_index,source_groups)
      +room = count_rooms(limits,outer_index,outers,source_groups,cell,outer_inside,cell_inside,count_room)
      +once = Order.times_one(cell)
      +summed_room = {%Equal.sym(Nat,Nat.mul(cell,1n),cell,once) : {Nat.is_le(Nat.add(front,_),half) == True{} : Bool} room :
        {Nat.is_le(Nat.add(front,Nat.mul(cell,1n)),half) == True{} : Bool}}
      +product = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),cell_word,step,cell,1n,cell_exact,step_exact,
        Order.transitive(Nat.mul(cell,1n),Nat.add(front,Nat.mul(cell,1n)),half,Order.right_le(front,Nat.mul(cell,1n)),summed_room))
      Equal.trans(Nat,U32.to_nat((base + cell_word * step : U32)),Nat.add(front,Nat.mul(cell,1n)),Nat.add(front,cell),
        Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),base,(cell_word * step : U32),front,Nat.mul(cell,1n),base_exact,product,summed_room),
        Equal.cong(Nat,Nat,value => Nat.add(front,value),Nat.mul(cell,1n),cell,once))
    case Grid.Outermost{}:
      +half = U32.to_nat(Machine.half_word(limits))
      +taken = Nat.mul(cell,outers)
      +room = Bounds.strict_le(Nat.add(taken,outer_index),Nat.mul(source_groups,outers),RowMajor.radix(cell,source_groups,outer_index,outers,cell_inside,outer_inside))
      +sum_room = Order.transitive(Nat.add(outer_index,taken),Nat.mul(source_groups,outers),half,
        %Order.commutative(taken,outer_index) : {Nat.is_le(_,Nat.mul(source_groups,outers)) == True{} : Bool} room,count_room)
      +product = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),cell_word,step,cell,outers,cell_exact,step_exact,
        Order.transitive(taken,Nat.add(outer_index,taken),half,Order.right_le(outer_index,taken),sum_room))
      Equal.trans(Nat,U32.to_nat((base + cell_word * step : U32)),Nat.add(outer_index,taken),Nat.add(taken,outer_index),
        Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),base,(cell_word * step : U32),outer_index,taken,base_exact,product,sum_room),
        Order.commutative(outer_index,taken))

# The source cell a target cell reads next, one step after the first.
def high_word(position: Grid.Position,+limits: Machine.Limits,+read: U32,+step: U32,+outer_index: Nat,+outers: Nat,+source_groups: Nat,+cell: Nat,
  +read_exact: {U32.to_nat(read) == Representation.index_at(position,outers,source_groups,outer_index,cell) : Nat},
  +step_exact: {U32.to_nat(step) == step_count(position,outers) : Nat},+outer_inside: {Nat.is_lt(outer_index,outers) == True{} : Bool},
  +next_inside: {Nat.is_lt(1n+cell,source_groups) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat((read + 1 * step : U32)) == Representation.index_at(position,outers,source_groups,outer_index,1n+cell) : Nat}:
  match position:
    case Grid.Innermost{}:
      +half = U32.to_nat(Machine.half_word(limits))
      +front = Nat.mul(outer_index,source_groups)
      +moved = Equal.trans(Nat,Nat.add(Nat.add(front,cell),1n),Nat.add(front,Nat.add(cell,1n)),Nat.add(front,1n+cell),Addition.associative(front,cell,1n),
        Equal.cong(Nat,Nat,value => Nat.add(front,value),Nat.add(cell,1n),Nat.add(1n,cell),Order.commutative(cell,1n)))
      +room = {%Equal.sym(Nat,Nat.add(Nat.add(front,cell),1n),Nat.add(front,1n+cell),moved) : {Nat.is_le(_,half) == True{} : Bool}
        count_rooms(limits,outer_index,outers,source_groups,1n+cell,outer_inside,next_inside,count_room) : {Nat.is_le(Nat.add(Nat.add(front,cell),1n),half) == True{} : Bool}}
      Equal.trans(Nat,U32.to_nat((read + 1 * step : U32)),Nat.add(Nat.add(front,cell),1n),Nat.add(front,1n+cell),
        Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),read,(1 * step : U32),Nat.add(front,cell),1n,read_exact,
          Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),1,step,1n,1n,{==},step_exact,
            Order.transitive(1n,Nat.add(Nat.add(front,cell),1n),half,Order.right_le(Nat.add(front,cell),1n),room)),room),
        moved)
    case Grid.Outermost{}:
      +half = U32.to_nat(Machine.half_word(limits))
      +taken = Nat.mul(cell,outers)
      +once = Addition.zero_right(outers)
      +moved = Equal.trans(Nat,Nat.add(Nat.add(taken,outer_index),Nat.mul(1n,outers)),Nat.add(Nat.add(taken,outer_index),outers),Nat.add(Nat.add(outers,taken),outer_index),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.add(taken,outer_index),value),Nat.mul(1n,outers),outers,once),
        Equal.trans(Nat,Nat.add(Nat.add(taken,outer_index),outers),Nat.add(taken,Nat.add(outer_index,outers)),Nat.add(Nat.add(outers,taken),outer_index),
          Addition.associative(taken,outer_index,outers),
          Equal.trans(Nat,Nat.add(taken,Nat.add(outer_index,outers)),Nat.add(taken,Nat.add(outers,outer_index)),Nat.add(Nat.add(outers,taken),outer_index),
            Equal.cong(Nat,Nat,value => Nat.add(taken,value),Nat.add(outer_index,outers),Nat.add(outers,outer_index),Order.commutative(outer_index,outers)),
            Equal.trans(Nat,Nat.add(taken,Nat.add(outers,outer_index)),Nat.add(Nat.add(taken,outers),outer_index),Nat.add(Nat.add(outers,taken),outer_index),
              Equal.sym(Nat,Nat.add(Nat.add(taken,outers),outer_index),Nat.add(taken,Nat.add(outers,outer_index)),Addition.associative(taken,outers,outer_index)),
              Equal.cong(Nat,Nat,value => Nat.add(value,outer_index),Nat.add(taken,outers),Nat.add(outers,taken),Order.commutative(taken,outers))))))
      +next_room = Order.transitive(Nat.add(Nat.mul(1n+cell,outers),outer_index),Nat.mul(source_groups,outers),half,
        Bounds.strict_le(Nat.add(Nat.mul(1n+cell,outers),outer_index),Nat.mul(source_groups,outers),RowMajor.radix(1n+cell,source_groups,outer_index,outers,next_inside,outer_inside)),
        count_room)
      +room = {%Equal.sym(Nat,Nat.add(Nat.add(taken,outer_index),Nat.mul(1n,outers)),Nat.add(Nat.mul(1n+cell,outers),outer_index),moved) : {Nat.is_le(_,half) == True{} : Bool}
        next_room : {Nat.is_le(Nat.add(Nat.add(taken,outer_index),Nat.mul(1n,outers)),half) == True{} : Bool}}
      Equal.trans(Nat,U32.to_nat((read + 1 * step : U32)),Nat.add(Nat.add(taken,outer_index),Nat.mul(1n,outers)),Nat.add(Nat.mul(1n+cell,outers),outer_index),
        Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),read,(1 * step : U32),Nat.add(taken,outer_index),Nat.mul(1n,outers),read_exact,
          Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),1,step,1n,outers,{==},step_exact,
            Order.transitive(Nat.mul(1n,outers),Nat.add(Nat.add(taken,outer_index),Nat.mul(1n,outers)),half,Order.right_le(Nat.add(taken,outer_index),Nat.mul(1n,outers)),room)),room),
        moved)

# ---- What a split needs ----

# The geometries of a split: the target rows hold the anchors of a source
# row, both hold the elements after the lanes, the source lanes are the
# anchors times the width, both groups and the source cells fit the words.
type Facts<-limits: Machine.Limits,-outer: U32,-anchors: U32,-width: U32,-after: U32,-lanes: U32,-target_outer: U32,-target_after: U32,-source_outer: U32,
  -source_after: U32> is Data:
  Facts{target_outer_exact: {U32.to_nat(target_outer) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)) : Nat},
    target_after_exact: {U32.to_nat(target_after) == U32.to_nat(after) : Nat},
    source_outer_exact: {U32.to_nat(source_outer) == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat},
    source_after_exact: {U32.to_nat(source_after) == U32.to_nat(after) : Nat},
    lanes_exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat},
    width_groups: TiledValues.Groups<limits,width>,lane_groups: TiledValues.Groups<limits,lanes>,
    outer_room: {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    anchors_room: {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    after_room: {Nat.is_le(U32.to_nat(after),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    groups_room: {Nat.is_le(U32.to_nat(Tiled.groups(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    source_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

# ---- The spot of a target cell ----

# Group group outermost at row column of the target: the digits of the
# rows before the lanes run over the groups first.
def outer_digits(+group: Nat,+outer: Nat,+anchors: Nat,+after: Nat,+above: Nat,+anchor: Nat,+below: Nat) ->
  {Nat.add(Nat.mul(group,Nat.mul(Nat.mul(outer,anchors),after)),Nat.add(Nat.mul(Nat.add(Nat.mul(above,anchors),anchor),after),below)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),below) : Nat}:
  +front = Nat.mul(Nat.mul(Nat.mul(group,outer),anchors),after)
  +row = Nat.mul(Nat.add(Nat.mul(above,anchors),anchor),after)
  +whole = Equal.trans(Nat,Nat.mul(group,Nat.mul(Nat.mul(outer,anchors),after)),Nat.mul(Nat.mul(group,Nat.mul(outer,anchors)),after),front,
    Equal.sym(Nat,Nat.mul(Nat.mul(group,Nat.mul(outer,anchors)),after),Nat.mul(group,Nat.mul(Nat.mul(outer,anchors),after)),RowMajor.assoc(group,Nat.mul(outer,anchors),after)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,after),Nat.mul(group,Nat.mul(outer,anchors)),Nat.mul(Nat.mul(group,outer),anchors),
      Equal.sym(Nat,Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(group,Nat.mul(outer,anchors)),RowMajor.assoc(group,outer,anchors))))
  +spread = Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),
    Nat.mul(Nat.add(Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(above,anchors)),anchor),after),Nat.add(front,row),
    Equal.cong(Nat,Nat,value => Nat.mul(Nat.add(value,anchor),after),Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),
      Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(above,anchors)),RowMajor.distrib(Nat.mul(group,outer),above,anchors)),
    Equal.trans(Nat,Nat.mul(Nat.add(Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(above,anchors)),anchor),after),
      Nat.mul(Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.add(Nat.mul(above,anchors),anchor)),after),Nat.add(front,row),
      Equal.cong(Nat,Nat,value => Nat.mul(value,after),Nat.add(Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(above,anchors)),anchor),
        Nat.add(Nat.mul(Nat.mul(group,outer),anchors),Nat.add(Nat.mul(above,anchors),anchor)),
        Addition.associative(Nat.mul(Nat.mul(group,outer),anchors),Nat.mul(above,anchors),anchor)),
      RowMajor.distrib(Nat.mul(Nat.mul(group,outer),anchors),Nat.add(Nat.mul(above,anchors),anchor),after)))
  Equal.trans(Nat,Nat.add(Nat.mul(group,Nat.mul(Nat.mul(outer,anchors),after)),Nat.add(row,below)),Nat.add(front,Nat.add(row,below)),
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),below),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.add(row,below)),Nat.mul(group,Nat.mul(Nat.mul(outer,anchors),after)),front,whole),
    Equal.trans(Nat,Nat.add(front,Nat.add(row,below)),Nat.add(Nat.add(front,row),below),
      Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),below),
      Equal.sym(Nat,Nat.add(Nat.add(front,row),below),Nat.add(front,Nat.add(row,below)),Addition.associative(front,row,below)),
      Equal.cong(Nat,Nat,value => Nat.add(value,below),Nat.add(front,row),Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),
        Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,outer),above),anchors),anchor),after),Nat.add(front,row),spread))))

# The spot of a target cell of digits above, anchor, below and group.
def spot_of(position: Grid.Position,+limits: Machine.Limits,+outer: U32,+anchors: U32,+after: U32,+groups: U32,+source_groups: U32,+outers: Nat,+index: U32,
  +above: Nat,+anchor: Nat,+below: Nat,+group: Nat,
  +above_inside: {Nat.is_lt(above,U32.to_nat(outer)) == True{} : Bool},+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +below_inside: {Nat.is_lt(below,U32.to_nat(after)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool},
  +placed: {U32.to_nat(index) == Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(groups),
    Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),below),group) : Nat},
  +outers_exact: {outers == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat},
  +outer_room: {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +anchors_room: {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +after_room: {Nat.is_le(U32.to_nat(after),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,U32.to_nat(source_groups)) == True{} : Bool},
  +outer_inside: {Nat.is_lt(Nat.add(Nat.mul(above,U32.to_nat(after)),below),outers) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(U32.to_nat(source_groups),outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Tiled.split_spot(position,outer,anchors,after,groups,source_groups,index) ==
    Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(spot_base(position,Nat.add(Nat.mul(above,U32.to_nat(after)),below),U32.to_nat(source_groups))),
      spot_step(position,outers)} : Tiled.Spot}:
  match position:
    case Grid.Innermost{}:
      +source_outer = Nat.add(Nat.mul(above,U32.to_nat(after)),below)
      +half = U32.to_nat(Machine.half_word(limits))
      +base_room = Order.transitive(Nat.mul(source_outer,U32.to_nat(source_groups)),Nat.mul(outers,U32.to_nat(source_groups)),half,
        RowMajor.mul_le(source_outer,outers,U32.to_nat(source_groups),Bounds.strict_le(source_outer,outers,outer_inside)),
        %RowMajor.multiply_commutative(U32.to_nat(source_groups),outers) : {Nat.is_le(_,half) == True{} : Bool} count_room)
      SplitSpot.inner_spot(limits,outer,anchors,after,groups,source_groups,index,above,anchor,below,group,anchor_inside,below_inside,group_inside,placed,
        anchors_room,after_room,groups_room,source_positive,base_room)
    case Grid.Outermost{}:
      +source_outer = Nat.add(Nat.mul(above,U32.to_nat(after)),below)
      +half = U32.to_nat(Machine.half_word(limits))
      +outers_room = Order.transitive(outers,Nat.mul(U32.to_nat(source_groups),outers),half,
        %RowMajor.multiply_commutative(outers,U32.to_nat(source_groups)) : {Nat.is_le(outers,_) == True{} : Bool}
        RowMajor.product_covers(outers,U32.to_nat(source_groups),source_positive),count_room)
      %Equal.sym(Nat,outers,Nat.mul(U32.to_nat(outer),U32.to_nat(after)),outers_exact) :
        {Tiled.split_spot(Grid.Outermost{},outer,anchors,after,groups,source_groups,index) ==
          Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(source_outer),U32.from_nat(_)} : Tiled.Spot}
      SplitSpot.outer_spot(limits,outer,anchors,after,groups,source_groups,index,above,anchor,below,group,above_inside,anchor_inside,below_inside,
        Equal.trans(Nat,U32.to_nat(index),
          Nat.add(Nat.mul(group,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after))),
            Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),below)),
          Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,U32.to_nat(outer)),above),U32.to_nat(anchors)),anchor),U32.to_nat(after)),below),
          placed,outer_digits(group,U32.to_nat(outer),U32.to_nat(anchors),U32.to_nat(after),above,anchor,below)),
        outer_room,anchors_room,after_room,
        Order.transitive(source_outer,outers,half,Bounds.strict_le(source_outer,outers,outer_inside),outers_room),
        %outers_exact : {Nat.is_le(_,half) == True{} : Bool} outers_room)

# A cell index below the cells is the index of its outer index and group.
def index_placed(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {index == Representation.index_at(position,outer,groups,Representation.outer_at(position,outer,groups,index),Representation.group_at(position,outer,groups,index)) : Nat}:
  match position:
    case Grid.Innermost{}:
      +positive = RowMajor.left_factor_positive(groups,outer,Order.positive_bound(index,Nat.mul(groups,outer),inside))
      %Representation.least_positive(groups,positive) :
        {index == Nat.add(Nat.mul(Nat.div(index,Representation.least_one(groups)),_),Nat.mod(index,Representation.least_one(groups))) : Nat}
      DivisionOrder.quotient_exact(index,Representation.least_one(groups),Representation.least_one_positive(groups))
    case Grid.Outermost{}:
      +positive = DivisionOrder.right_factor_positive(groups,outer,Order.positive_bound(index,Nat.mul(groups,outer),inside))
      %Representation.least_positive(outer,positive) :
        {index == Nat.add(Nat.mul(Nat.div(index,Representation.least_one(outer)),_),Nat.mod(index,Representation.least_one(outer))) : Nat}
      DivisionOrder.quotient_exact(index,Representation.least_one(outer),Representation.least_one_positive(outer))

# ---- Room for the words ----

def outers_room(+limits: Machine.Limits,+outers: Nat,+source_groups: Nat,+positive: {Nat.is_lt(0n,source_groups) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(outers,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(outers,Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits)),
    %RowMajor.multiply_commutative(outers,source_groups) : {Nat.is_le(outers,_) == True{} : Bool} RowMajor.product_covers(outers,source_groups,positive),count_room)

# The source cell of group 0 fits half a word.
def base_room(position: Grid.Position,+limits: Machine.Limits,+outer_index: Nat,+outers: Nat,+source_groups: Nat,
  +outer_inside: {Nat.is_lt(outer_index,outers) == True{} : Bool},+positive: {Nat.is_lt(0n,source_groups) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.mul(source_groups,outers),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(spot_base(position,outer_index,source_groups),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      Order.transitive(Nat.mul(outer_index,source_groups),Nat.mul(outers,source_groups),U32.to_nat(Machine.half_word(limits)),
        RowMajor.mul_le(outer_index,outers,source_groups,Bounds.strict_le(outer_index,outers,outer_inside)),
        %RowMajor.multiply_commutative(source_groups,outers) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool} count_room)
    case Grid.Outermost{}:
      Order.transitive(outer_index,outers,U32.to_nat(Machine.half_word(limits)),Bounds.strict_le(outer_index,outers,outer_inside),
        outers_room(limits,outers,source_groups,positive,count_room))

def step_exact(position: Grid.Position,+limits: Machine.Limits,+outers: Nat,+room: {Nat.is_le(outers,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat(spot_step(position,outers)) == step_count(position,outers) : Nat}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: FullRange.roundtrip(outers,Machine.half_word(limits),room)

# A positive count has a positive number of groups.
def groups_positive(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.is_lt(0n,Ceiling.ceiling(value,8n)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(0n,Ceiling.ceiling(value,8n)),Nat.is_lt(Nat.mul(0n,8n),value),True{},Ceiling.upper_comparison(value,8n,0n,{==}),positive)

# ---- The lanes of a spot ----

# The first lane of anchor anchor and group group of the width.
def first_lane(+anchor: Nat,+width: U32,+group: Nat) -> Nat:
  Nat.add(Nat.mul(anchor,U32.to_nat(width)),Nat.mul(group,8n))

# The lane words of a spot of anchor anchor and group group as their values:
# the first lane of the group, eight times the group, the source cell of the
# first lane, the shift into it and the next cell.
type Lanes<-limits: Machine.Limits,-anchors: U32,-width: U32,-lanes: U32,-anchor: Nat,-group: Nat> is Data:
  Lanes{group_exact: {U32.to_nat(U32.from_nat(group)) == group : Nat},eights_exact: {U32.to_nat((U32.from_nat(group) * 8 : U32)) == Nat.mul(group,8n) : Nat},
    first_exact: {U32.to_nat((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32)) == first_lane(anchor,width,group) : Nat},
    cell_exact: {U32.to_nat(((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) / 8 : U32)) == Nat.div(first_lane(anchor,width,group),8n) : Nat},
    shift_exact: {U32.to_nat(((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) % 8 : U32)) == Nat.mod(first_lane(anchor,width,group),8n) : Nat},
    next_exact: {U32.to_nat((((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) / 8 : U32) + 1 : U32)) ==
      Nat.add(Nat.div(first_lane(anchor,width,group),8n),1n) : Nat},
    cell_inside: {Nat.is_lt(Nat.div(first_lane(anchor,width,group),8n),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
    bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
    covered: {Nat.is_le(Nat.mul(U32.to_nat(anchors),U32.to_nat(width)),Nat.mul(U32.to_nat(Tiled.groups(lanes)),8n)) == True{} : Bool}}

# The lane words of a spot of an anchor and a group inside theirs.
def spot_lanes(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,
  +width_groups: TiledValues.Groups<limits,width>,+lane_groups: TiledValues.Groups<limits,lanes>,
  +lanes_exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat},
  +anchors_room: {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(Tiled.groups(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(width))) == True{} : Bool}) ->
  Lanes<limits,anchors,width,lanes,anchor,group>:
  TiledValues.Groups{+width_exact,+width_fewer,+width_fit} = width_groups
  TiledValues.Groups{+lane_exact,+lane_fewer,+lane_fit} = lane_groups
  +half_word = Machine.half_word(limits)
  +half = U32.to_nat(half_word)
  +fits = Limits.half_word_fits(limits)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +count = U32.to_nat(anchors)
  +span = U32.to_nat(width)
  +sources = U32.to_nat(Tiled.groups(lanes))
  +first = first_lane(anchor,width,group)
  +cell = Nat.div(first,8n)
  +group_ceiled = {%width_exact : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,Ceiling.ceiling(span,8n)) == True{} : Bool}}
  +eights_below = Equal.trans(Bool,Nat.is_lt(Nat.mul(group,8n),span),Nat.is_lt(group,Ceiling.ceiling(span,8n)),True{},
    Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(span,8n)),Nat.is_lt(Nat.mul(group,8n),span),Ceiling.upper_comparison(span,8n,group,{==})),group_ceiled)
  +first_below = RowMajor.radix(anchor,count,Nat.mul(group,8n),span,anchor_inside,eights_below)
  +covered = {%lanes_exact : {Nat.is_le(_,Nat.mul(sources,8n)) == True{} : Bool}
    {%Equal.sym(Nat,sources,Ceiling.ceiling(U32.to_nat(lanes),8n),lane_exact) : {Nat.is_le(U32.to_nat(lanes),Nat.mul(_,8n)) == True{} : Bool}
      Ceiling.covers(U32.to_nat(lanes),8n,{==}) : {Nat.is_le(U32.to_nat(lanes),Nat.mul(sources,8n)) == True{} : Bool}} :
    {Nat.is_le(Nat.mul(count,span),Nat.mul(sources,8n)) == True{} : Bool}}
  +first_room = Order.transitive(first,Nat.mul(sources,8n),half,Bounds.strict_le(first,Nat.mul(sources,8n),Order.lt_le(first,Nat.mul(count,span),Nat.mul(sources,8n),first_below,covered)),
    lane_fit)
  +cell_inside = RowMajor.cancel_multiply_lt(cell,sources,8n,Order.le_lt_trans(Nat.mul(cell,8n),first,Nat.mul(sources,8n),DivisionOrder.product_below(first,8n,{==}),
    Order.lt_le(first,Nat.mul(count,span),Nat.mul(sources,8n),first_below,covered)))
  +sources_room = Order.transitive(sources,Nat.mul(sources,8n),half,RowMajor.product_covers(sources,8n,{==}),lane_fit)
  +anchor_word = FullRange.roundtrip(anchor,Machine.capacity(limits),Order.transitive(anchor,count,capacity,Bounds.strict_le(anchor,count,anchor_inside),anchors_room))
  +group_word = FullRange.roundtrip(group,Machine.capacity(limits),
    Order.transitive(group,U32.to_nat(Tiled.groups(width)),capacity,Bounds.strict_le(group,U32.to_nat(Tiled.groups(width)),group_inside),groups_room))
  +eights_room = Order.transitive(Nat.mul(group,8n),first,half,Order.right_le(Nat.mul(anchor,span),Nat.mul(group,8n)),first_room)
  +eights_exact = Arithmetic.product_exact(half_word,fits,U32.from_nat(group),8,group,8n,group_word,{==},eights_room)
  +first_word = (U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32)
  +first_exact = Arithmetic.sum_exact(half_word,fits,(U32.from_nat(anchor) * width : U32),(U32.from_nat(group) * 8 : U32),Nat.mul(anchor,span),Nat.mul(group,8n),
    Arithmetic.product_exact(half_word,fits,U32.from_nat(anchor),width,anchor,span,anchor_word,{==},SplitSpot.left_room(Nat.mul(anchor,span),Nat.mul(group,8n),half,first_room)),
    eights_exact,first_room)
  +sixteen = Bounds.as_nat_le(16,half_word,Limits.sixteen_within_half(limits))
  +cell_exact = U32Division.quotient_exact(half_word,fits,first_word,8,first,8n,first_exact,{==},sixteen,{==})
  +next_room = {%Equal.sym(Nat,Nat.add(cell,1n),Nat.add(1n,cell),Order.commutative(cell,1n)) : {Nat.is_le(_,half) == True{} : Bool}
    Order.transitive(1n+cell,sources,half,Order.strict_follow(cell,sources,cell_inside),sources_room) : {Nat.is_le(Nat.add(cell,1n),half) == True{} : Bool}}
  +bound = Order.transitive(Nat.add(Nat.mul(group,8n),7n),Nat.mul(U32.to_nat(Tiled.groups(width)),8n),half,
    Bounds.strict_le(Nat.add(Nat.mul(group,8n),7n),Nat.mul(U32.to_nat(Tiled.groups(width)),8n),
      Bounds.nat_trans(Nat.add(Nat.mul(group,8n),7n),Nat.mul(1n+group,8n),Nat.mul(U32.to_nat(Tiled.groups(width)),8n),
        %Order.commutative(7n,Nat.mul(group,8n)) : {Nat.is_lt(_,Nat.mul(1n+group,8n)) == True{} : Bool} PaddedCells.step_below(group),
        RowMajor.mul_le(1n+group,U32.to_nat(Tiled.groups(width)),8n,Order.strict_follow(group,U32.to_nat(Tiled.groups(width)),group_inside)))),
    width_fit)
  Lanes{group_word,eights_exact,first_exact,cell_exact,U32Division.remainder_exact(half_word,fits,first_word,8,first,8n,first_exact,{==},sixteen,{==}),
    Arithmetic.sum_exact(half_word,fits,(first_word / 8 : U32),1,cell,1n,cell_exact,{==},next_room),cell_inside,bound,covered}

# A lane inside the width that needs the next source cell finds it inside
# the source groups.
def needed_next(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,
  +anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +need: {SplitCell.needed(U32.to_nat(width),group,Nat.mod(first_lane(anchor,width,group),8n)) == True{} : Bool}) ->
  {Nat.is_lt(1n+Nat.div(first_lane(anchor,width,group),8n),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}:
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  +span = U32.to_nat(width)
  +count = U32.to_nat(anchors)
  +first = first_lane(anchor,width,group)
  +cell = Nat.div(first,8n)
  +shift = Nat.mod(first,8n)
  +start = Nat.mul(anchor,span)
  +rest = Nat.sub(8n,shift)
  +eighths = Nat.mul(cell,8n)
  +inside = Boolean.right(Nat.is_lt(0n,shift),Nat.is_lt(Nat.add(Nat.mul(group,8n),rest),span),need)
  +eight = Order.restore_shift(shift,8n,Order.successor(shift,7n,Division.lt_succ_le(shift,7n,DivisionOrder.remainder_small(first,8n,{==}))))
  +key = DivisionOrder.quotient_exact(first,8n,{==})
  +top = Equal.trans(Nat,Nat.mul(1n+cell,8n),Nat.add(eighths,8n),Nat.add(first,rest),Order.commutative(8n,eighths),
    Equal.trans(Nat,Nat.add(eighths,8n),Nat.add(eighths,Nat.add(shift,rest)),Nat.add(first,rest),
      Equal.cong(Nat,Nat,value => Nat.add(eighths,value),8n,Nat.add(shift,rest),Equal.sym(Nat,Nat.add(shift,rest),8n,eight)),
      Equal.trans(Nat,Nat.add(eighths,Nat.add(shift,rest)),Nat.add(Nat.add(eighths,shift),rest),Nat.add(first,rest),
        Equal.sym(Nat,Nat.add(Nat.add(eighths,shift),rest),Nat.add(eighths,Nat.add(shift,rest)),Addition.associative(eighths,shift,rest)),
        Equal.cong(Nat,Nat,value => Nat.add(value,rest),Nat.add(eighths,shift),first,Equal.sym(Nat,first,Nat.add(eighths,shift),key)))))
  +reach = Equal.trans(Nat,Nat.mul(1n+cell,8n),Nat.add(first,rest),Nat.add(start,Nat.add(Nat.mul(group,8n),rest)),top,
    Addition.associative(start,Nat.mul(group,8n),rest))
  +moved = {%Equal.sym(Nat,Nat.mul(1n+cell,8n),Nat.add(start,Nat.add(Nat.mul(group,8n),rest)),reach) : {Nat.is_lt(_,Nat.add(start,span)) == True{} : Bool}
    Order.offset_lt(start,Nat.add(Nat.mul(group,8n),rest),span,inside) : {Nat.is_lt(Nat.mul(1n+cell,8n),Nat.add(start,span)) == True{} : Bool}}
  +anchored = {%Order.commutative(span,start) : {Nat.is_le(_,Nat.mul(count,span)) == True{} : Bool}
    RowMajor.mul_le(1n+anchor,count,span,Order.strict_follow(anchor,count,anchor_inside)) : {Nat.is_le(Nat.add(start,span),Nat.mul(count,span)) == True{} : Bool}}
  RowMajor.cancel_multiply_lt(1n+cell,U32.to_nat(Tiled.groups(lanes)),8n,
    Order.lt_le(Nat.mul(1n+cell,8n),Nat.mul(count,span),Nat.mul(U32.to_nat(Tiled.groups(lanes)),8n),
      Order.lt_le(Nat.mul(1n+cell,8n),Nat.add(start,span),Nat.mul(count,span),moved,anchored),covered))

# The spot of an anchor and a group, from a source row, as words.
def spot_of_lanes(+position: Grid.Position,+anchor: Nat,+group: Nat,+outer_index: Nat,+lanes: U32,+source_outer: U32) -> Tiled.Spot:
  Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(spot_base(position,outer_index,U32.to_nat(Tiled.groups(lanes)))),
    spot_step(position,U32.to_nat(source_outer))}

# A lane of the spot needs the next source cell exactly when a lane of its
# group does.
def need_spot(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+anchor: Nat,+group: Nat,+outer_index: Nat,
  +lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,
  +need: {SplitCell.needed(U32.to_nat(width),group,Nat.mod(first_lane(anchor,width,group),8n)) == True{} : Bool}) ->
  {spot_needed(width,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer)) == True{} : Bool}:
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  +shift_word = ((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) % 8 : U32)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(group)),group,group_exact) : {SplitCell.needed(U32.to_nat(width),_,U32.to_nat(shift_word)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(shift_word),Nat.mod(first_lane(anchor,width,group),8n),shift_exact) : {SplitCell.needed(U32.to_nat(width),group,_) == True{} : Bool}
  need

def need_nat(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+anchor: Nat,+group: Nat,+outer_index: Nat,
  +lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,
  +need: {spot_needed(width,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer)) == True{} : Bool}) ->
  {SplitCell.needed(U32.to_nat(width),group,Nat.mod(first_lane(anchor,width,group),8n)) == True{} : Bool}:
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  +shift_word = ((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) % 8 : U32)
  %group_exact : {SplitCell.needed(U32.to_nat(width),_,Nat.mod(first_lane(anchor,width,group),8n)) == True{} : Bool}
  %shift_exact : {SplitCell.needed(U32.to_nat(width),U32.to_nat(U32.from_nat(group)),_) == True{} : Bool}
  need

# The next source cell of a spot that needs it.
def spot_next(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+anchor: Nat,+group: Nat,+outer_index: Nat,
  +lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +read_exact: {U32.to_nat(spot_read(width,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer))) ==
    Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Tiled.groups(lanes)),outer_index,Nat.div(first_lane(anchor,width,group),8n)) : Nat},
  +steps: {U32.to_nat(spot_step(position,U32.to_nat(source_outer))) == step_count(position,U32.to_nat(source_outer)) : Nat},
  +outer_inside: {Nat.is_lt(outer_index,U32.to_nat(source_outer)) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +need: {spot_needed(width,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer)) == True{} : Bool}) ->
  Next<width,lanes,source_outer,position,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer),outer_index,Nat.div(first_lane(anchor,width,group),8n)>:
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  +sources = U32.to_nat(Tiled.groups(lanes))
  +first = first_lane(anchor,width,group)
  +cell = Nat.div(first,8n)
  +shift = Nat.mod(first,8n)
  +first_word = (U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32)
  +plain = need_nat(limits,anchors,width,lanes,source_outer,position,anchor,group,outer_index,Lanes{group_exact,eights_exact,first_exact,cell_exact,shift_exact,next_exact,cell_inside,bound,covered},need)
  +next_inside = needed_next(limits,anchors,width,lanes,anchor,group,Lanes{group_exact,eights_exact,first_exact,cell_exact,shift_exact,next_exact,cell_inside,bound,covered},anchor_inside,plain)
  +shifted = Equal.trans(Bool,U32.is_lt(0,(first_word % 8 : U32)),Nat.is_lt(0n,U32.to_nat((first_word % 8 : U32))),True{},Comparison.lt(0,(first_word % 8 : U32)),
    {%Equal.sym(Nat,U32.to_nat((first_word % 8 : U32)),shift,shift_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      Boolean.left(Nat.is_lt(0n,shift),Nat.is_lt(Nat.add(Nat.mul(group,8n),Nat.sub(8n,shift)),U32.to_nat(width)),plain) :
      {Nat.is_lt(0n,U32.to_nat((first_word % 8 : U32))) == True{} : Bool}})
  +follows = Equal.trans(Bool,U32.is_lt(((first_word / 8 : U32) + 1 : U32),Tiled.groups(lanes)),Nat.is_lt(U32.to_nat(((first_word / 8 : U32) + 1 : U32)),sources),True{},
    Comparison.lt(((first_word / 8 : U32) + 1 : U32),Tiled.groups(lanes)),
    {%Equal.sym(Nat,U32.to_nat(((first_word / 8 : U32) + 1 : U32)),Nat.add(cell,1n),next_exact) : {Nat.is_lt(_,sources) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(cell,1n),Nat.add(1n,cell),Order.commutative(cell,1n)) : {Nat.is_lt(_,sources) == True{} : Bool}
      next_inside : {Nat.is_lt(U32.to_nat(((first_word / 8 : U32) + 1 : U32)),sources) == True{} : Bool}})
  Next{Boolean.and_true(U32.is_lt(0,(first_word % 8 : U32)),U32.is_lt(((first_word / 8 : U32) + 1 : U32),Tiled.groups(lanes)),shifted,follows),next_inside,
    high_word(position,limits,spot_read(width,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer)),spot_step(position,U32.to_nat(source_outer)),
      outer_index,U32.to_nat(source_outer),sources,cell,read_exact,steps,outer_inside,next_inside,source_room)}

# The words of the spot of an anchor and a group at a source row inside the
# source cells.
def spot_words(+position: Grid.Position,+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+source_outer: U32,+anchor: Nat,+group: Nat,+outer_index: Nat,
  +lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +outer_inside: {Nat.is_lt(outer_index,U32.to_nat(source_outer)) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Words<width,lanes,source_outer,position,spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer),outer_index,Nat.div(first_lane(anchor,width,group),8n)>:
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  +outers = U32.to_nat(source_outer)
  +sources = U32.to_nat(Tiled.groups(lanes))
  +cell = Nat.div(first_lane(anchor,width,group),8n)
  +sources_positive = Representation.below_positive(cell,sources,cell_inside)
  +base_exact = FullRange.roundtrip(spot_base(position,outer_index,sources),Machine.half_word(limits),base_room(position,limits,outer_index,outers,sources,outer_inside,sources_positive,source_room))
  +steps = step_exact(position,limits,outers,outers_room(limits,outers,sources,sources_positive,source_room))
  +spot = spot_of_lanes(position,anchor,group,outer_index,lanes,source_outer)
  +read_exact = source_word(position,limits,U32.from_nat(spot_base(position,outer_index,sources)),spot_step(position,outers),
    ((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) / 8 : U32),outer_index,outers,sources,cell,base_exact,steps,cell_exact,outer_inside,cell_inside,source_room)
  Words{read_exact,outer_inside,cell_inside,
    Boolean.when_enabled(Next<width,lanes,source_outer,position,spot,outer_index,cell>,spot_needed(width,spot),
      need => spot_next(limits,anchors,width,lanes,source_outer,position,anchor,group,outer_index,
        Lanes{group_exact,eights_exact,first_exact,cell_exact,shift_exact,next_exact,cell_inside,bound,covered},anchor_inside,read_exact,steps,outer_inside,source_room,need))}

# ---- The value of the cell of a spot ----

# The next source cell a spot needs holds the source cell after the first.
def next_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+width: U32,+lanes: U32,+source_outer: U32,+source_after: U32,+position: Grid.Position,
  +spot: Tiled.Spot,+outer_index: Nat,+cell: Nat,+next: Next<width,lanes,source_outer,position,spot,outer_index,cell>,
  +outer_inside: {Nat.is_lt(outer_index,U32.to_nat(source_outer)) == True{} : Bool},+high: Tile.Tile8,
  +hit: {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}},spot_high(width,spot)) : Tile.Tile8}) ->
  {high == Representation.cell(~Logical,~original,logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position},outer_index,1n+cell) : Tile.Tile8}:
  Next{+further,+next_inside,+high_exact} = next
  +source = {Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  Equal.trans(Tile.Tile8,high,Representation.lifted(~Logical,~original,Representation.Lifted{logical,source},spot_high(width,spot)),
    Representation.cell(~Logical,~original,logical,source,outer_index,1n+cell),hit,
    Representation.decoded(~Logical,~original,logical,source,outer_index,1n+cell,spot_high(width,spot),high_exact,outer_inside,next_inside))

# The next cell joins as the source cell after the first when a lane needs
# it; otherwise it does not matter.
def high_cases(need: Bool,+limits: Machine.Limits,+first_lane_word: U32,+width: U32,+group: Nat,+shift: Nat,+low: Tile.Tile8,+high: Tile.Tile8,+next: Tile.Tile8,
  +first_exact: {U32.to_nat(first_lane_word) == Nat.mul(group,8n) : Nat},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+classified: {SplitCell.needed(U32.to_nat(width),group,shift) == need : Bool},
  hit: @+needed: {SplitCell.needed(U32.to_nat(width),group,shift) == True{} : Bool} -> {high == next : Tile.Tile8}) ->
  {Cells.tail(Cells.Tail{first_lane_word,width},Tiled.joined(U32.from_nat(shift),low,high)) == Cells.tail(Cells.Tail{first_lane_word,width},Tiled.joined(U32.from_nat(shift),low,next)) : Tile.Tile8}:
  match need:
    case True{}:
      Equal.cong(Tile.Tile8,Tile.Tile8,cell => Cells.tail(Cells.Tail{first_lane_word,width},Tiled.joined(U32.from_nat(shift),low,cell)),high,next,hit(classified))
    case False{}:
      SplitCell.next_unused(limits,first_lane_word,width,group,shift,low,high,next,first_exact,bound,shift_small,classified)

# A target cell of digits above, anchor, below and group: the cell its spot
# gives from the source cell it reads first and the next, when a lane needs
# it, holds the target cell.
def spotted_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +position: Grid.Position,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,
  +target_index: Nat,+above: Nat,+anchor: Nat,+below: Nat,+group: Nat,
  +lanes_fact: Lanes<limits,anchors,width,lanes,anchor,group>,
  +words: Words<width,lanes,source_outer,position,spot_of_lanes(position,anchor,group,Nat.add(Nat.mul(above,U32.to_nat(after)),below),lanes,source_outer),
    Nat.add(Nat.mul(above,U32.to_nat(after)),below),Nat.div(first_lane(anchor,width,group),8n)>,
  +anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},+below_inside: {Nat.is_lt(below,U32.to_nat(after)) == True{} : Bool},
  +above_exact: {Nat.div(target_index,U32.to_nat(after)) == Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor) : Nat},
  +below_exact: {Nat.mod(target_index,U32.to_nat(after)) == below : Nat},+low: Tile.Tile8,+high: Tile.Tile8,
  +low_exact: {low == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}},
    spot_read(width,spot_of_lanes(position,anchor,group,Nat.add(Nat.mul(above,U32.to_nat(after)),below),lanes,source_outer))) : Tile.Tile8},
  high_exact: @+need: {spot_needed(width,spot_of_lanes(position,anchor,group,Nat.add(Nat.mul(above,U32.to_nat(after)),below),lanes,source_outer)) == True{} : Bool} ->
    {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}},
      spot_high(width,spot_of_lanes(position,anchor,group,Nat.add(Nat.mul(above,U32.to_nat(after)),below),lanes,source_outer))) : Tile.Tile8}) ->
  {spot_cell(width,spot_of_lanes(position,anchor,group,Nat.add(Nat.mul(above,U32.to_nat(after)),below),lanes,source_outer),low,high) ==
    Representation.cell(~Logical,~original,logical,Cells.Cells{target_outer,target_after,width,Tiled.groups(width),position},target_index,group) : Tile.Tile8}:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  Words{+read_exact,+outer_inside,+words_inside,+next} = words
  +source = {Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +target = {Cells.Cells{target_outer,target_after,width,Tiled.groups(width),position} : Cells.Cells}
  +count = U32.to_nat(anchors)
  +span = U32.to_nat(width)
  +rows_after = U32.to_nat(after)
  +source_index = Nat.add(Nat.mul(above,rows_after),below)
  +first = first_lane(anchor,width,group)
  +cell = Nat.div(first,8n)
  +shift = Nat.mod(first,8n)
  +spot = spot_of_lanes(position,anchor,group,source_index,lanes,source_outer)
  +shift_word = ((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) % 8 : U32)
  +after_positive = Representation.below_positive(below,rows_after,below_inside)
  +key = Equal.sym(Nat,first,Nat.add(Nat.mul(cell,8n),shift),DivisionOrder.quotient_exact(first,8n,{==}))
  +shift_small = Division.lt_succ_le(shift,7n,DivisionOrder.remainder_small(first,8n,{==}))
  +low_cell = Representation.cell(~Logical,~original,logical,source,source_index,cell)
  +next_cell = Representation.cell(~Logical,~original,logical,source,source_index,1n+cell)
  +low_value = Equal.trans(Tile.Tile8,low,Representation.lifted(~Logical,~original,Representation.Lifted{logical,source},spot_read(width,spot)),low_cell,low_exact,
    Representation.decoded(~Logical,~original,logical,source,source_index,cell,spot_read(width,spot),read_exact,outer_inside,cell_inside))
  +lanes_again = {Lanes{group_exact,eights_exact,first_exact,cell_exact,shift_exact,next_exact,cell_inside,bound,covered} : Lanes<limits,anchors,width,lanes,anchor,group>}
  +source_after_count = Equal.trans(Nat,Representation.after_count(source),Representation.least_one(rows_after),rows_after,
    Equal.cong(Nat,Nat,value => Representation.least_one(value),U32.to_nat(source_after),rows_after,source_after_exact),Representation.least_positive(rows_after,after_positive))
  +place = {SplitCell.Place{lanes_exact,{==},
    Equal.cong(Nat,Nat,value => Representation.least_one(value),U32.to_nat(target_after),U32.to_nat(source_after),
      Equal.trans(Nat,U32.to_nat(target_after),rows_after,U32.to_nat(source_after),target_after_exact,Equal.sym(Nat,U32.to_nat(source_after),rows_after,source_after_exact))),
    {%Equal.sym(Nat,Representation.after_count(source),rows_after,source_after_count) : {Nat.div(target_index,_) == Nat.add(Nat.mul(above,count),anchor) : Nat} above_exact :
      {Nat.div(target_index,Representation.after_count(source)) == Nat.add(Nat.mul(above,count),anchor) : Nat}},
    {%Equal.sym(Nat,Representation.after_count(source),rows_after,source_after_count) : {Nat.div(source_index,_) == above : Nat}
      SplitSpot.divided(source_index,rows_after,above,below,{==},below_inside) : {Nat.div(source_index,Representation.after_count(source)) == above : Nat}},
    {%Equal.sym(Nat,Representation.after_count(source),rows_after,source_after_count) : {Nat.mod(target_index,_) == Nat.mod(source_index,_) : Nat}
      Equal.trans(Nat,Nat.mod(target_index,rows_after),below,Nat.mod(source_index,rows_after),below_exact,
        Equal.sym(Nat,Nat.mod(source_index,rows_after),below,SplitSpot.remaindered(source_index,rows_after,above,below,{==},below_inside))) :
      {Nat.mod(target_index,Representation.after_count(source)) == Nat.mod(source_index,Representation.after_count(source)) : Nat}},
    anchor_inside} : SplitCell.Place<source,target,source_index,target_index,above,count,anchor,span>}
  +tail = {Cells.Tail{(U32.from_nat(group) * 8 : U32),width} : Cells.Tail}
  +cut = SplitCell.split_cell(~Logical,~original,logical,limits,source,target,source_index,target_index,above,count,anchor,width,group,(U32.from_nat(group) * 8 : U32),cell,shift,
    place,key,shift_small,eights_exact,bound)
  +needed_high = high_cases(SplitCell.needed(span,group,shift),limits,(U32.from_nat(group) * 8 : U32),width,group,shift,low_cell,high,next_cell,eights_exact,bound,shift_small,{==},
    hit => next_value(~Logical,~original,logical,width,lanes,source_outer,source_after,position,spot,source_index,cell,
      Boolean.required(Next<width,lanes,source_outer,position,spot,source_index,cell>,spot_needed(width,spot),next,
        need_spot(limits,anchors,width,lanes,source_outer,position,anchor,group,source_index,lanes_again,hit)),
      outer_inside,high,high_exact(need_spot(limits,anchors,width,lanes,source_outer,position,anchor,group,source_index,lanes_again,hit))))
  %Arithmetic.from_nat_exact(shift_word,shift,shift_exact) :
    {Cells.tail(tail,Tiled.joined(_,low,high)) == Representation.cell(~Logical,~original,logical,target,target_index,group) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Cells.tail(tail,Tiled.joined(U32.from_nat(shift),low,high)),Cells.tail(tail,Tiled.joined(U32.from_nat(shift),low_cell,high)),
    Representation.cell(~Logical,~original,logical,target,target_index,group),
    Equal.cong(Tile.Tile8,Tile.Tile8,value => Cells.tail(tail,Tiled.joined(U32.from_nat(shift),value,high)),low,low_cell,low_value),
    Equal.trans(Tile.Tile8,Cells.tail(tail,Tiled.joined(U32.from_nat(shift),low_cell,high)),Cells.tail(tail,Tiled.joined(U32.from_nat(shift),low_cell,next_cell)),
      Representation.cell(~Logical,~original,logical,target,target_index,group),needed_high,cut))

# ---- The digits of a target cell index ----

# The row of a target cell index, its group, the row of the elements before
# the lanes, its anchor, the index after them and the source row.
def split_row(position: Grid.Position,+target_outer: U32,+width: U32,+index: U32) -> Nat:
  Representation.outer_at(position,U32.to_nat(target_outer),U32.to_nat(Tiled.groups(width)),U32.to_nat(index))

def split_group(position: Grid.Position,+target_outer: U32,+width: U32,+index: U32) -> Nat:
  Representation.group_at(position,U32.to_nat(target_outer),U32.to_nat(Tiled.groups(width)),U32.to_nat(index))

def row_above(+row: Nat,+anchors: U32,+after: U32) -> Nat:
  Nat.div(Nat.div(row,U32.to_nat(after)),U32.to_nat(anchors))

def row_anchor(+row: Nat,+anchors: U32,+after: U32) -> Nat:
  Nat.mod(Nat.div(row,U32.to_nat(after)),U32.to_nat(anchors))

def row_below(+row: Nat,+after: U32) -> Nat:
  Nat.mod(row,U32.to_nat(after))

def source_row(+row: Nat,+anchors: U32,+after: U32) -> Nat:
  Nat.add(Nat.mul(row_above(row,anchors,after),U32.to_nat(after)),row_below(row,after))

# The digits of a target cell index below the target cells lie inside their
# extents, and the index and the source row are their sums.
type Digits<-outer: U32,-anchors: U32,-width: U32,-after: U32,-lanes: U32,-target_outer: U32,-source_outer: U32,-position: Grid.Position,-index: U32> is Data:
  Digits{above_inside: {Nat.is_lt(row_above(split_row(position,target_outer,width,index),anchors,after),U32.to_nat(outer)) == True{} : Bool},
    anchor_inside: {Nat.is_lt(row_anchor(split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)) == True{} : Bool},
    below_inside: {Nat.is_lt(row_below(split_row(position,target_outer,width,index),after),U32.to_nat(after)) == True{} : Bool},
    group_inside: {Nat.is_lt(split_group(position,target_outer,width,index),U32.to_nat(Tiled.groups(width))) == True{} : Bool},
    above_exact: {Nat.div(split_row(position,target_outer,width,index),U32.to_nat(after)) ==
      Nat.add(Nat.mul(row_above(split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)),row_anchor(split_row(position,target_outer,width,index),anchors,after)) : Nat},
    placed: {U32.to_nat(index) == Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(Tiled.groups(width)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(row_above(split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)),
        row_anchor(split_row(position,target_outer,width,index),anchors,after)),U32.to_nat(after)),row_below(split_row(position,target_outer,width,index),after)),
      split_group(position,target_outer,width,index)) : Nat},
    source_inside: {Nat.is_lt(source_row(split_row(position,target_outer,width,index),anchors,after),U32.to_nat(source_outer)) == True{} : Bool},
    source_positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}}

# The digits of a target cell index below the target cells.
def split_digits(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position,+target_outer: U32,+target_after: U32,
  +source_outer: U32,+source_after: U32,+facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,TiledValues.Groups{+width_exact,+width_fewer,+width_fit},
    TiledValues.Groups{+lane_exact,+lane_fewer,+lane_fit},+outer_room,+anchors_room,+after_room,+groups_room,+source_room} = facts
  +outers = U32.to_nat(target_outer)
  +groups = U32.to_nat(Tiled.groups(width))
  +number = U32.to_nat(index)
  +rows_before = U32.to_nat(outer)
  +count = U32.to_nat(anchors)
  +span = U32.to_nat(width)
  +rows_after = U32.to_nat(after)
  +rows = Nat.mul(Nat.mul(rows_before,count),rows_after)
  +target_index = split_row(position,target_outer,width,index)
  +group = split_group(position,target_outer,width,index)
  +cells_positive = Order.positive_bound(number,Nat.mul(groups,outers),inside)
  +outers_positive = DivisionOrder.right_factor_positive(groups,outers,cells_positive)
  +target_groups_positive = RowMajor.left_factor_positive(groups,outers,cells_positive)
  +outer_inside = Representation.outer_inside(position,outers,groups,number,inside,outers_positive,target_groups_positive)
  +group_inside = Representation.group_inside(position,outers,groups,number,inside,outers_positive,target_groups_positive)
  +row_inside = {%target_outer_exact : {Nat.is_lt(target_index,_) == True{} : Bool} outer_inside : {Nat.is_lt(target_index,rows) == True{} : Bool}}
  +rows_positive = Order.positive_bound(target_index,rows,row_inside)
  +after_positive = DivisionOrder.right_factor_positive(Nat.mul(rows_before,count),rows_after,rows_positive)
  +count_positive = DivisionOrder.right_factor_positive(rows_before,count,RowMajor.left_factor_positive(Nat.mul(rows_before,count),rows_after,rows_positive))
  +quotient = Nat.div(target_index,rows_after)
  +above = Nat.div(quotient,count)
  +anchor = Nat.mod(quotient,count)
  +below = Nat.mod(target_index,rows_after)
  +above_exact = DivisionOrder.quotient_exact(quotient,count,count_positive)
  +digits = Equal.trans(Nat,target_index,Nat.add(Nat.mul(quotient,rows_after),below),Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),rows_after),below),
    DivisionOrder.quotient_exact(target_index,rows_after,after_positive),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,rows_after),below),quotient,Nat.add(Nat.mul(above,count),anchor),above_exact))
  +above_inside = DivisionOrder.quotient_below(quotient,rows_before,count,count_positive,
    DivisionOrder.quotient_below(target_index,Nat.mul(rows_before,count),rows_after,after_positive,row_inside))
  +below_inside = DivisionOrder.remainder_small(target_index,rows_after,after_positive)
  +placed = {%digits : {number == Representation.index_at(position,rows,groups,_,group) : Nat}
    %target_outer_exact : {number == Representation.index_at(position,_,groups,target_index,group) : Nat}
    index_placed(position,outers,groups,number,inside) :
    {number == Representation.index_at(position,rows,groups,Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),rows_after),below),group) : Nat}}
  +source_positive = {%Equal.sym(Nat,U32.to_nat(Tiled.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),lane_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    groups_positive(U32.to_nat(lanes),{%Equal.sym(Nat,U32.to_nat(lanes),Nat.mul(count,span),lanes_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      Order.product_positive(count,span,count_positive,Order.lt_le(0n,groups,span,target_groups_positive,width_fewer)) : {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}}) :
    {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}}
  +source_inside = {%Equal.sym(Nat,U32.to_nat(source_outer),Nat.mul(rows_before,rows_after),source_outer_exact) :
      {Nat.is_lt(Nat.add(Nat.mul(above,rows_after),below),_) == True{} : Bool}
    RowMajor.radix(above,rows_before,below,rows_after,above_inside,below_inside) : {Nat.is_lt(Nat.add(Nat.mul(above,rows_after),below),U32.to_nat(source_outer)) == True{} : Bool}}
  Digits{above_inside,DivisionOrder.remainder_small(quotient,count,count_positive),below_inside,group_inside,above_exact,placed,source_inside,source_positive}

# ---- The value of a target cell ----

# The spot of a target cell index: its anchor, its group and its source row.
def split_spot_exact(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position,+target_outer: U32,+target_after: U32,
  +source_outer: U32,+source_after: U32,+facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +digits: Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  {Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index) ==
    spot_of_lanes(position,row_anchor(split_row(position,target_outer,width,index),anchors,after),split_group(position,target_outer,width,index),
      source_row(split_row(position,target_outer,width,index),anchors,after),lanes,source_outer) : Tiled.Spot}:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  +row = split_row(position,target_outer,width,index)
  spot_of(position,limits,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),U32.to_nat(source_outer),index,row_above(row,anchors,after),row_anchor(row,anchors,after),
    row_below(row,after),split_group(position,target_outer,width,index),above_inside,anchor_inside,below_inside,group_inside,placed,source_outer_exact,outer_room,anchors_room,
    after_room,groups_room,source_positive,source_inside,source_room)

# The lane words of a target cell index.
def split_lanes_of(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position,+target_outer: U32,+target_after: U32,
  +source_outer: U32,+source_after: U32,+facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +digits: Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  Lanes<limits,anchors,width,lanes,row_anchor(split_row(position,target_outer,width,index),anchors,after),split_group(position,target_outer,width,index)>:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  spot_lanes(limits,anchors,width,lanes,row_anchor(split_row(position,target_outer,width,index),anchors,after),split_group(position,target_outer,width,index),
    width_groups,lane_groups,lanes_exact,anchors_room,groups_room,anchor_inside,group_inside)

# The words of the spot of a target cell index.
def split_words(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position,+target_outer: U32,+target_after: U32,
  +source_outer: U32,+source_after: U32,+facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +digits: Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  Words<width,lanes,source_outer,position,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),
    source_row(split_row(position,target_outer,width,index),anchors,after),
    Nat.div(first_lane(row_anchor(split_row(position,target_outer,width,index),anchors,after),width,split_group(position,target_outer,width,index)),8n)>:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  +row = split_row(position,target_outer,width,index)
  +anchor = row_anchor(row,anchors,after)
  +group = split_group(position,target_outer,width,index)
  +source_index = source_row(row,anchors,after)
  +rebuilt = {Facts{target_outer_exact,target_after_exact,source_outer_exact,source_after_exact,lanes_exact,width_groups,lane_groups,outer_room,anchors_room,after_room,
    groups_room,source_room} : Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>}
  +again = {Digits{above_inside,anchor_inside,below_inside,group_inside,above_exact,placed,source_inside,source_positive} : Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>}
  %Equal.sym(Tiled.Spot,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),spot_of_lanes(position,anchor,group,source_index,lanes,source_outer),
      split_spot_exact(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again)) :
    Words<width,lanes,source_outer,position,_,source_index,Nat.div(first_lane(anchor,width,group),8n)>
  spot_words(position,limits,anchors,width,lanes,source_outer,anchor,group,source_index,
    split_lanes_of(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again),anchor_inside,source_inside,source_room)

# A lane of the spot of a target cell index needs the next source cell
# exactly when a lane of its group inside the width does.
def split_needed(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position,+target_outer: U32,+target_after: U32,
  +source_outer: U32,+source_after: U32,+facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +digits: Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>,
  +need: {spot_needed(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)) == True{} : Bool}) ->
  {SplitCell.needed(U32.to_nat(width),split_group(position,target_outer,width,index),
    Nat.mod(first_lane(row_anchor(split_row(position,target_outer,width,index),anchors,after),width,split_group(position,target_outer,width,index)),8n)) == True{} : Bool}:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  +row = split_row(position,target_outer,width,index)
  +anchor = row_anchor(row,anchors,after)
  +group = split_group(position,target_outer,width,index)
  +source_index = source_row(row,anchors,after)
  +rebuilt = {Facts{target_outer_exact,target_after_exact,source_outer_exact,source_after_exact,lanes_exact,width_groups,lane_groups,outer_room,anchors_room,after_room,
    groups_room,source_room} : Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>}
  +again = {Digits{above_inside,anchor_inside,below_inside,group_inside,above_exact,placed,source_inside,source_positive} : Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>}
  +exact = split_spot_exact(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again)
  need_nat(limits,anchors,width,lanes,source_outer,position,anchor,group,source_index,
    split_lanes_of(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again),
    {%exact : {spot_needed(width,_) == True{} : Bool} need : {spot_needed(width,spot_of_lanes(position,anchor,group,source_index,lanes,source_outer)) == True{} : Bool}})

# A target cell index below the target cells: the cell its spot gives from
# the source cell it reads first and the next, when a lane needs it, holds
# the target cell at the index.
def split_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +position: Grid.Position,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+index: U32,
  +digits: Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>,+low: Tile.Tile8,+high: Tile.Tile8,
  +low_exact: {low == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}},
    spot_read(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index))) : Tile.Tile8},
  high_exact: @+need: {spot_needed(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)) == True{} : Bool} ->
    {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}},
      spot_high(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index))) : Tile.Tile8}) ->
  {spot_cell(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),low,high) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{target_outer,target_after,width,Tiled.groups(width),position}},index) : Tile.Tile8}:
  Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  +row = split_row(position,target_outer,width,index)
  +above = row_above(row,anchors,after)
  +anchor = row_anchor(row,anchors,after)
  +below = row_below(row,after)
  +group = split_group(position,target_outer,width,index)
  +source_index = source_row(row,anchors,after)
  +source_context = {Representation.Lifted{logical,Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}} : Representation.Lifted<Logical>}
  +split = Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)
  +spot = spot_of_lanes(position,anchor,group,source_index,lanes,source_outer)
  +rebuilt = {Facts{target_outer_exact,target_after_exact,source_outer_exact,source_after_exact,lanes_exact,width_groups,lane_groups,outer_room,anchors_room,after_room,
    groups_room,source_room} : Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>}
  +again = {Digits{above_inside,anchor_inside,below_inside,group_inside,above_exact,placed,source_inside,source_positive} : Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>}
  +exact = split_spot_exact(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again)
  +lanes_fact = split_lanes_of(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,index,again)
  %Equal.sym(Tiled.Spot,split,spot,exact) :
    {spot_cell(width,_,low,high) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{target_outer,target_after,width,Tiled.groups(width),position}},index) : Tile.Tile8}
  spotted_value(~Logical,~original,logical,limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,rebuilt,row,above,anchor,below,group,
    lanes_fact,spot_words(position,limits,anchors,width,lanes,source_outer,anchor,group,source_index,lanes_fact,anchor_inside,source_inside,source_room),
    anchor_inside,below_inside,above_exact,{==},low,high,
    {%exact : {low == Representation.lifted(~Logical,~original,source_context,spot_read(width,_)) : Tile.Tile8} low_exact :
      {low == Representation.lifted(~Logical,~original,source_context,spot_read(width,spot)) : Tile.Tile8}},
    need => {%exact : {high == Representation.lifted(~Logical,~original,source_context,spot_high(width,_)) : Tile.Tile8}
      high_exact({%Equal.sym(Tiled.Spot,split,spot,exact) : {spot_needed(width,_) == True{} : Bool} need : {spot_needed(width,split) == True{} : Bool}}) :
      {high == Representation.lifted(~Logical,~original,source_context,spot_high(width,spot)) : Tile.Tile8}})
