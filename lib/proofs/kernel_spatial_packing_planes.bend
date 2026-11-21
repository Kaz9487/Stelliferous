# The planes the transpose writes. Plane (phase, channel) holds, at element
# e, the pair of tile e and tile e + 1 of the plane; a tile reads eight input
# columns stride apart, the lane of the channel in the cells of its group, +0
# outside the input. Tile (row, column) of a plane reads plane row row at
# tile column column.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_samples.bend as PackingSamples
import ../kernel_tile.bend as Tile
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_channel_lanes.bend as Lanes
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./binary_radix.bend as Radix
import ./boolean_logic.bend as Boolean
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding
import ./word_multiplication.bend as Multiplication
import ./u32_division.bend as WordDivision

# ---- Samples ----

# Lane index of a block: the lane the bits name of the cell at the pixel, +0
# outside the input.
def cell_value(+x: Tree.Tree<Tile.Tile8>,+lanes: Lanes.Bits,+block: Packing.PixelBlock,index: Nat) -> F32:
  Packing.PixelBlock{+row,+column,+channel_base,+height,+width,+stride,+groups} = block
  +ix = (column + U32.from_nat(index) * stride : U32)
  PackingSamples.masked(Lanes.lane_at(lanes,Tree.read(Tile.Tile8,x,(channel_base + (row * width + ix) * groups : U32))),U32.is_lt(row,height) && U32.is_lt(ix,width))

def cells_tile(+x: Tree.Tree<Tile.Tile8>,+lanes: Lanes.Bits,+block: Packing.PixelBlock) -> Tile.Tile8:
  Tile.Tile{cell_value(x,lanes,block,0n),cell_value(x,lanes,block,1n),cell_value(x,lanes,block,2n),cell_value(x,lanes,block,3n),cell_value(x,lanes,block,4n),
    cell_value(x,lanes,block,5n),cell_value(x,lanes,block,6n),cell_value(x,lanes,block,7n)}

# ---- Indexes ----

# Naturals up to a cap are equal exactly when their words are.
def encoded_equal(+left: Nat,+right: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +left_bound: {Nat.is_le(left,U32.to_nat(cap)) == True{} : Bool},+right_bound: {Nat.is_le(right,U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == Nat.is_eq(left,right) : Bool}:
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(left),U32.from_nat(right)),Nat.is_eq(U32.to_nat(U32.from_nat(left)),U32.to_nat(U32.from_nat(right))),
    Comparison.eq(U32.from_nat(left),U32.from_nat(right))) : {_ == Nat.is_eq(left,right) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,Bounds.roundtrip(left,cap,cap_ok,left_bound)) :
    {Nat.is_eq(_,U32.to_nat(U32.from_nat(right))) == Nat.is_eq(left,right) : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(right)),right,Bounds.roundtrip(right,cap,cap_ok,right_bound)) : {Nat.is_eq(left,_) == Nat.is_eq(left,right) : Bool}
  {==}

# Index (major * middles + middle) * minors + minor.
def radix(+major: Nat,+middle: Nat,+minor: Nat,+middles: Nat,+minors: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(major,middles),middle),minors),minor)

# The pair (major, minor) as one number.
def outer(+major: Nat,+minor: Nat,+minors: Nat) -> Nat:
  Nat.add(Nat.mul(major,minors),minor)

type Digits<-middle: Nat,-minor: Nat,-middles: Nat,-minors: Nat> is Data:
  Digits{middles_positive: {Nat.is_lt(0n,middles) == True{} : Bool},minors_positive: {Nat.is_lt(0n,minors) == True{} : Bool},
    middle_inside: {Nat.is_lt(middle,middles) == True{} : Bool},minor_inside: {Nat.is_lt(minor,minors) == True{} : Bool}}

def radix_forward(+major: Nat,+middle: Nat,+minor: Nat,+other_major: Nat,+other_middle: Nat,+other_minor: Nat,+middles: Nat,+minors: Nat,
  +digits: Digits<middle,minor,middles,minors>,+other_digits: Digits<other_middle,other_minor,middles,minors>,
  same: {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(other_major,other_middle,other_minor,middles,minors)) == True{} : Bool}) ->
  {Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)) && Nat.is_eq(middle,other_middle) == True{} : Bool}:
  Digits{+middles_positive,+minors_positive,+middle_inside,+minor_inside} = digits
  Digits{other_middles_positive,other_minors_positive,+other_middle_inside,+other_minor_inside} = other_digits
  +equal = Order.equal_of_is_eq(radix(major,middle,minor,middles,minors),radix(other_major,other_middle,other_minor,middles,minors),same)
  +rows = Radix.row_unique(Nat.add(Nat.mul(major,middles),middle),Nat.add(Nat.mul(other_major,middles),other_middle),minors,minor,other_minor,minors_positive,minor_inside,
    other_minor_inside,equal)
  +minor_same = Radix.col_unique(Nat.add(Nat.mul(major,middles),middle),Nat.add(Nat.mul(other_major,middles),other_middle),minors,minor,other_minor,minors_positive,minor_inside,
    other_minor_inside,equal)
  +major_same = Radix.row_unique(major,other_major,middles,middle,other_middle,middles_positive,middle_inside,other_middle_inside,rows)
  +middle_same = Radix.col_unique(major,other_major,middles,middle,other_middle,middles_positive,middle_inside,other_middle_inside,rows)
  %major_same : {Nat.is_eq(outer(major,minor,minors),outer(_,other_minor,minors)) && Nat.is_eq(middle,other_middle) == True{} : Bool}
  %minor_same : {Nat.is_eq(outer(major,minor,minors),outer(major,_,minors)) && Nat.is_eq(middle,other_middle) == True{} : Bool}
  %middle_same : {Nat.is_eq(outer(major,minor,minors),outer(major,minor,minors)) && Nat.is_eq(middle,_) == True{} : Bool}
  Boolean.and_true(Nat.is_eq(outer(major,minor,minors),outer(major,minor,minors)),Nat.is_eq(middle,middle),Order.equal_reflexive(outer(major,minor,minors)),
    Order.equal_reflexive(middle))

def radix_backward(+major: Nat,+middle: Nat,+minor: Nat,+other_major: Nat,+other_middle: Nat,+other_minor: Nat,+middles: Nat,+minors: Nat,
  +digits: Digits<middle,minor,middles,minors>,+other_digits: Digits<other_middle,other_minor,middles,minors>,
  +both: {Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)) && Nat.is_eq(middle,other_middle) == True{} : Bool}) ->
  {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(other_major,other_middle,other_minor,middles,minors)) == True{} : Bool}:
  Digits{+middles_positive,+minors_positive,+middle_inside,+minor_inside} = digits
  Digits{other_middles_positive,other_minors_positive,+other_middle_inside,+other_minor_inside} = other_digits
  +outer_same = Order.equal_of_is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors),
    Boolean.and_true_left(Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)),Nat.is_eq(middle,other_middle),both))
  +middle_same = Order.equal_of_is_eq(middle,other_middle,
    Boolean.and_true_right(Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)),Nat.is_eq(middle,other_middle),both))
  +major_same = Radix.row_unique(major,other_major,minors,minor,other_minor,minors_positive,minor_inside,other_minor_inside,outer_same)
  +minor_same = Radix.col_unique(major,other_major,minors,minor,other_minor,minors_positive,minor_inside,other_minor_inside,outer_same)
  %major_same : {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(_,other_middle,other_minor,middles,minors)) == True{} : Bool}
  %middle_same : {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(major,_,other_minor,middles,minors)) == True{} : Bool}
  %minor_same : {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(major,middle,_,middles,minors)) == True{} : Bool}
  Order.equal_reflexive(radix(major,middle,minor,middles,minors))

# Two indexes agree when their outer pairs and their middles agree.
def radix_equal(+major: Nat,+middle: Nat,+minor: Nat,+other_major: Nat,+other_middle: Nat,+other_minor: Nat,+middles: Nat,+minors: Nat,
  +digits: Digits<middle,minor,middles,minors>,+other_digits: Digits<other_middle,other_minor,middles,minors>) ->
  {Nat.is_eq(radix(major,middle,minor,middles,minors),radix(other_major,other_middle,other_minor,middles,minors)) ==
    Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)) && Nat.is_eq(middle,other_middle) : Bool}:
  Boolean.equivalent(Nat.is_eq(radix(major,middle,minor,middles,minors),radix(other_major,other_middle,other_minor,middles,minors)),
    Nat.is_eq(outer(major,minor,minors),outer(other_major,other_minor,minors)) && Nat.is_eq(middle,other_middle),
    same => radix_forward(major,middle,minor,other_major,other_middle,other_minor,middles,minors,digits,other_digits,same),
    both => radix_backward(major,middle,minor,other_major,other_middle,other_minor,middles,minors,digits,other_digits,both))

# ---- Rows of a plane ----

# What the rows of a plane read: the input row of plane row zero, the input
# column of tile column zero, the tiles of a plane row, the cells of the
# channel's group from channel_base on, the image, the stride, the groups
# and the channels.
type Row is Data:
  Row{input: Tree.Tree<Tile.Tile8>,first_row: U32,first_column: U32,row_tiles: U32,channel_base: U32,height: U32,width: U32,stride: U32,groups: U32,channels: U32}

# A tile of a plane: plane row row, tile column column.
type Spot is Data:
  Spot{row: Nat,column: Nat}

def row_of(+spot: Spot) -> Nat:
  Spot{+row,column} = spot
  row

def column_of(+spot: Spot) -> Nat:
  Spot{row,+column} = spot
  column

def next_spot(+spot: Spot,+tiles: Nat) -> Spot:
  Spot{+row,+column} = spot
  Bool.pick(Spot,Nat.is_lt(1n+column,tiles),Spot{row,1n+column},Spot{1n+row,0n})

def row_tiles(+context: Row) -> Nat:
  Row{input,first_row,first_column,+row_tiles,channel_base,height,width,stride,groups,channels} = context
  U32.to_nat(row_tiles)

def channels_of(+context: Row) -> U32:
  Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,+channels} = context
  channels

# The tile at a spot, read with lane.
def tile(~lane: Tile.Tile8 -> F32,+context: Row,+spot: Spot) -> Tile.Tile8:
  Row{+input,+first_row,+first_column,row_tiles,+channel_base,+height,+width,+stride,+groups,channels} = context
  Model.row_tile(~lane,input,(first_row + U32.from_nat(row_of(spot)) : U32),(first_column + U32.from_nat(column_of(spot)) * (8 * stride : U32) : U32),channel_base,height,width,
    stride,groups)

# The same tile, the lane named by a pattern of bits.
def lanes_tile(+context: Row,+lanes: Lanes.Bits,+spot: Spot) -> Tile.Tile8:
  Row{+input,+first_row,+first_column,row_tiles,+channel_base,+height,+width,+stride,+groups,channels} = context
  cells_tile(input,lanes,Packing.PixelBlock{(first_row + U32.from_nat(row_of(spot)) : U32),(first_column + U32.from_nat(column_of(spot)) * (8 * stride : U32) : U32),channel_base,
    height,width,stride,groups})

# The rows about to read the tile at a spot, the tile before it kept.
def rowing_pairs(state: Model.Rowing) -> Tree.Tree<Tile.Tile16>:
  Model.Rowing{input,pairs,previous,column,write,row,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels} = state
  pairs

def ready(pairs: Tree.Tree<Tile.Tile16>,previous: Tile.Tile8,+context: Row,+spot: Spot,write: U32) -> Model.Rowing:
  Row{input,+first_row,+first_column,+row_tiles,channel_base,height,width,+stride,groups,channels} = context
  Model.Rowing{input,pairs,previous,(first_column + U32.from_nat(column_of(spot)) * (8 * stride : U32) : U32),write,(first_row + U32.from_nat(row_of(spot)) : U32),
    U32.from_nat(column_of(spot)),row_tiles,first_column,channel_base,height,width,stride,groups,channels}

def ready_pairs(+pairs: Tree.Tree<Tile.Tile16>,+previous: Tile.Tile8,+context: Row,+spot: Spot,+write: U32) ->
  {rowing_pairs(ready(pairs,previous,context,spot,write)) == pairs : Tree.Tree<Tile.Tile16>}:
  Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  {==}

# The rows past the tile at a spot, the plane row ended when wrap holds.
def advanced(+wrap: Bool,pairs: Tree.Tree<Tile.Tile16>,tile: Tile.Tile8,+context: Row,+spot: Spot,write: U32) -> Model.Rowing:
  Row{input,+first_row,+first_column,+row_tiles,channel_base,height,width,+stride,groups,channels} = context
  Spot{+row,+column} = spot
  +column_word = (first_column + U32.from_nat(column) * (8 * stride : U32) : U32)
  +row_word = (first_row + U32.from_nat(row) : U32)
  Model.Rowing{input,pairs,tile,Bool.pick(U32,wrap,first_column,(column_word + 8 * stride : U32)),write,Bool.pick(U32,wrap,(row_word + 1 : U32),row_word),
    Bool.pick(U32,wrap,0,(U32.from_nat(column) + 1 : U32)),row_tiles,first_column,channel_base,height,width,stride,groups,channels}

def wrapped(+context: Row,+spot: Spot) -> Bool:
  Row{input,first_row,first_column,+row_tiles,channel_base,height,width,stride,groups,channels} = context
  Spot{row,+column} = spot
  U32.is_eq((U32.from_nat(column) + 1 : U32),row_tiles)

# The row loop past the tile at a spot, holding tile.
def moved_from(pairs: Tree.Tree<Tile.Tile16>,tile: Tile.Tile8,+context: Row,+spot: Spot,write: U32) -> Model.Rowing:
  Row{input,+first_row,+first_column,+row_tiles,channel_base,height,width,+stride,groups,channels} = context
  Spot{+row,+column} = spot
  Model.moved(input,pairs,tile,(first_column + U32.from_nat(column) * (8 * stride : U32) : U32),write,(first_row + U32.from_nat(row) : U32),U32.from_nat(column),row_tiles,
    first_column,channel_base,height,width,stride,groups,channels)

def moved_advanced(+pairs: Tree.Tree<Tile.Tile16>,+tile: Tile.Tile8,+context: Row,+spot: Spot,+write: U32) ->
  {moved_from(pairs,tile,context,spot,write) == advanced(wrapped(context,spot),pairs,tile,context,spot,write) : Model.Rowing}:
  Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  Spot{+row,+column} = spot
  {==}

# One tile column on: column + 8 * stride.
def column_step(+first_column: U32,+column: Nat,+stride: U32) ->
  {((first_column + U32.from_nat(column) * (8 * stride : U32) : U32) + 8 * stride : U32) == (first_column + U32.from_nat(1n+column) * (8 * stride : U32) : U32) : U32}:
  +span = (8 * stride : U32)
  Equal.trans(U32,((first_column + U32.from_nat(column) * span : U32) + span : U32),(first_column + (U32.from_nat(column) * span + span : U32) : U32),
    (first_column + U32.from_nat(1n+column) * span : U32),Arithmetic.add_associative(first_column,(U32.from_nat(column) * span : U32),span),
    Equal.cong(U32,U32,tail => (first_column + tail : U32),(U32.from_nat(column) * span + span : U32),(U32.from_nat(1n+column) * span : U32),
      Equal.trans(U32,(U32.from_nat(column) * span + span : U32),((U32.from_nat(column) + 1 : U32) * span : U32),(U32.from_nat(1n+column) * span : U32),
        Equal.sym(U32,((U32.from_nat(column) + 1 : U32) * span : U32),(U32.from_nat(column) * span + span : U32),Arithmetic.multiply_successor(U32.from_nat(column),span)),
        Equal.cong(U32,U32,index => (index * span : U32),(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)))))

# Tile column zero.
def column_zero(+first_column: U32,+stride: U32) -> {first_column == (first_column + U32.from_nat(0n) * (8 * stride : U32) : U32) : U32}:
  Equal.sym(U32,(first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),first_column,
    Equal.trans(U32,(first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),(first_column + 0 : U32),first_column,
      Equal.cong(U32,U32,tail => (first_column + tail : U32),(U32.from_nat(0n) * (8 * stride : U32) : U32),0,Arithmetic.multiply_zero_left((8 * stride : U32))),
      Arithmetic.add_zero(first_column)))

# One plane row on.
def row_step(+first_row: U32,+row: Nat) -> {((first_row + U32.from_nat(row) : U32) + 1 : U32) == (first_row + U32.from_nat(1n+row) : U32) : U32}:
  Equal.trans(U32,((first_row + U32.from_nat(row) : U32) + 1 : U32),(first_row + (U32.from_nat(row) + 1 : U32) : U32),(first_row + U32.from_nat(1n+row) : U32),
    Arithmetic.add_associative(first_row,U32.from_nat(row),1),
    Equal.cong(U32,U32,tail => (first_row + tail : U32),(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)))

# The tiles of a plane row: at least one, below a cap.
type RowRoom<-context: Row> is Data:
  RowRoom{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},tiles_bound: {Nat.is_le(row_tiles(context),U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,row_tiles(context)) == True{} : Bool}}

# Below the limit the successor is not the limit; at the limit it is.
def inside_not_equal(+value: Nat,+limit: Nat,+inside: Bool,+exact: {Nat.is_lt(value,limit) == inside : Bool},+bounded: {Nat.is_le(value,limit) == True{} : Bool}) ->
  {Nat.is_eq(value,limit) == Bool.not(inside) : Bool}:
  match inside:
    case True{}: Order.less_not_equal(value,limit,exact)
    case False{}:
      %Order.antisymmetric(value,limit,bounded,Order.not_less(value,limit,exact)) : {Nat.is_eq(value,_) == True{} : Bool}
      Order.equal_reflexive(value)

# Wrapping is the next column falling outside the row.
def wrap_exact(+context: Row,+spot: Spot,+room: RowRoom<context>,+column_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool}) ->
  {wrapped(context,spot) == Bool.not(Nat.is_lt(1n+column_of(spot),row_tiles(context))) : Bool}:
  Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  Spot{+row,+column} = spot
  RowRoom{+cap,+cap_ok,+tiles_bound,+positive} = room
  +tiles = U32.to_nat(row_tiles)
  %Equal.sym(U32,(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)) : {U32.is_eq(_,row_tiles) == Bool.not(Nat.is_lt(1n+column,tiles)) : Bool}
  %Arithmetic.native_roundtrip(row_tiles) : {U32.is_eq(U32.from_nat(1n+column),_) == Bool.not(Nat.is_lt(1n+column,tiles)) : Bool}
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+column),U32.from_nat(tiles)),Nat.is_eq(1n+column,tiles),
    encoded_equal(1n+column,tiles,cap,cap_ok,Order.transitive(1n+column,tiles,U32.to_nat(cap),Order.strict_follow(column,tiles,column_inside),tiles_bound),tiles_bound)) :
    {_ == Bool.not(Nat.is_lt(1n+column,tiles)) : Bool}
  inside_not_equal(1n+column,tiles,Nat.is_lt(1n+column,tiles),{==},Order.strict_follow(column,tiles,column_inside))

# The rows with the given column, row and in-row words.
def placed(pairs: Tree.Tree<Tile.Tile16>,tile: Tile.Tile8,+context: Row,column_word: U32,write: U32,row_word: U32,in_row: U32) -> Model.Rowing:
  Row{input,first_row,+first_column,+row_tiles,channel_base,height,width,stride,groups,channels} = context
  Model.Rowing{input,pairs,tile,column_word,write,row_word,in_row,row_tiles,first_column,channel_base,height,width,stride,groups,channels}

def next_within(+spot: Spot,+tiles: Nat,+exact: {Nat.is_lt(1n+column_of(spot),tiles) == True{} : Bool}) -> {next_spot(spot,tiles) == Spot{row_of(spot),1n+column_of(spot)} : Spot}:
  Spot{+row,+column} = spot
  %Equal.sym(Bool,Nat.is_lt(1n+column,tiles),True{},exact) : {Bool.pick(Spot,_,Spot{row,1n+column},Spot{1n+row,0n}) == Spot{row,1n+column} : Spot}
  {==}

def next_wrapped(+spot: Spot,+tiles: Nat,+exact: {Nat.is_lt(1n+column_of(spot),tiles) == False{} : Bool}) -> {next_spot(spot,tiles) == Spot{1n+row_of(spot),0n} : Spot}:
  Spot{+row,+column} = spot
  %Equal.sym(Bool,Nat.is_lt(1n+column,tiles),False{},exact) : {Bool.pick(Spot,_,Spot{row,1n+column},Spot{1n+row,0n}) == Spot{1n+row,0n} : Spot}
  {==}

# Advancing past a spot readies the next spot.
def advanced_ready(+inside: Bool,+pairs: Tree.Tree<Tile.Tile16>,+tile: Tile.Tile8,+context: Row,+spot: Spot,+write: U32,
  +exact: {Nat.is_lt(1n+column_of(spot),row_tiles(context)) == inside : Bool}) ->
  {advanced(Bool.not(inside),pairs,tile,context,spot,write) == ready(pairs,tile,context,next_spot(spot,row_tiles(context)),write) : Model.Rowing}:
  match inside:
    case True{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      Spot{+row,+column} = spot
      +context = {Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels} : Row}
      +row_word = (first_row + U32.from_nat(row) : U32)
      %Equal.sym(Spot,next_spot(Spot{row,column},U32.to_nat(row_tiles)),Spot{row,1n+column},next_within(Spot{row,column},U32.to_nat(row_tiles),exact)) :
        {advanced(False{},pairs,tile,context,Spot{row,column},write) == ready(pairs,tile,context,_,write) : Model.Rowing}
      Equal.trans(Model.Rowing,placed(pairs,tile,context,((first_column + U32.from_nat(column) * (8 * stride : U32) : U32) + 8 * stride : U32),write,row_word,
          (U32.from_nat(column) + 1 : U32)),
        placed(pairs,tile,context,(first_column + U32.from_nat(1n+column) * (8 * stride : U32) : U32),write,row_word,(U32.from_nat(column) + 1 : U32)),
        placed(pairs,tile,context,(first_column + U32.from_nat(1n+column) * (8 * stride : U32) : U32),write,row_word,U32.from_nat(1n+column)),
        Equal.cong(U32,Model.Rowing,word => placed(pairs,tile,context,word,write,row_word,(U32.from_nat(column) + 1 : U32)),
          ((first_column + U32.from_nat(column) * (8 * stride : U32) : U32) + 8 * stride : U32),(first_column + U32.from_nat(1n+column) * (8 * stride : U32) : U32),
          column_step(first_column,column,stride)),
        Equal.cong(U32,Model.Rowing,word => placed(pairs,tile,context,(first_column + U32.from_nat(1n+column) * (8 * stride : U32) : U32),write,row_word,word),
          (U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)))
    case False{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      Spot{+row,+column} = spot
      +context = {Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels} : Row}
      %Equal.sym(Spot,next_spot(Spot{row,column},U32.to_nat(row_tiles)),Spot{1n+row,0n},next_wrapped(Spot{row,column},U32.to_nat(row_tiles),exact)) :
        {advanced(True{},pairs,tile,context,Spot{row,column},write) == ready(pairs,tile,context,_,write) : Model.Rowing}
      Equal.trans(Model.Rowing,placed(pairs,tile,context,first_column,write,((first_row + U32.from_nat(row) : U32) + 1 : U32),0),
        placed(pairs,tile,context,(first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),write,((first_row + U32.from_nat(row) : U32) + 1 : U32),0),
        placed(pairs,tile,context,(first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),write,(first_row + U32.from_nat(1n+row) : U32),0),
        Equal.cong(U32,Model.Rowing,word => placed(pairs,tile,context,word,write,((first_row + U32.from_nat(row) : U32) + 1 : U32),0),first_column,
          (first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),column_zero(first_column,stride)),
        Equal.cong(U32,Model.Rowing,word => placed(pairs,tile,context,(first_column + U32.from_nat(0n) * (8 * stride : U32) : U32),write,word,0),
          ((first_row + U32.from_nat(row) : U32) + 1 : U32),(first_row + U32.from_nat(1n+row) : U32),row_step(first_row,row)))

# The row loop past the tile at a spot readies the next spot.
def moved_ready(+pairs: Tree.Tree<Tile.Tile16>,+tile: Tile.Tile8,+context: Row,+spot: Spot,+write: U32,+room: RowRoom<context>,
  +column_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool}) ->
  {moved_from(pairs,tile,context,spot,write) == ready(pairs,tile,context,next_spot(spot,row_tiles(context)),write) : Model.Rowing}:
  +inside = Nat.is_lt(1n+column_of(spot),row_tiles(context))
  Equal.trans(Model.Rowing,moved_from(pairs,tile,context,spot,write),advanced(Bool.not(inside),pairs,tile,context,spot,write),
    ready(pairs,tile,context,next_spot(spot,row_tiles(context)),write),
    Equal.trans(Model.Rowing,moved_from(pairs,tile,context,spot,write),advanced(wrapped(context,spot),pairs,tile,context,spot,write),
      advanced(Bool.not(inside),pairs,tile,context,spot,write),moved_advanced(pairs,tile,context,spot,write),
      Equal.cong(Bool,Model.Rowing,wrap => advanced(wrap,pairs,tile,context,spot,write),wrapped(context,spot),Bool.not(inside),wrap_exact(context,spot,room,column_inside))),
    advanced_ready(inside,pairs,tile,context,spot,write,{==}))

# The first read of a row loop, and every later one.
def first_ready(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+previous: Tile.Tile8,+context: Row,+spot: Spot,+write: U32,+room: RowRoom<context>,
  +column_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool}) ->
  {Model.row_first(~lane,ready(pairs,previous,context,spot,write)) == ready(pairs,tile(~lane,context,spot),context,next_spot(spot,row_tiles(context)),write) : Model.Rowing}:
  Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  Spot{+row,+column} = spot
  +context = {Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels} : Row}
  +spot = {Spot{row,column} : Spot}
  moved_ready(pairs,tile(~lane,context,spot),context,spot,write,room,column_inside)

def step_ready(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+previous: Tile.Tile8,+context: Row,+spot: Spot,+write: U32,+room: RowRoom<context>,
  +column_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool}) ->
  {Model.row_step(~lane,ready(pairs,previous,context,spot,write)) ==
    ready(Tree.write(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile(~lane,context,spot)}),tile(~lane,context,spot),context,next_spot(spot,row_tiles(context)),
      (write + channels_of(context) : U32)) : Model.Rowing}:
  Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
  Spot{+row,+column} = spot
  +context = {Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels} : Row}
  +spot = {Spot{row,column} : Spot}
  moved_ready(Tree.write(Tile.Tile16,pairs,write,Tile.Tiles{previous,tile(~lane,context,spot)}),tile(~lane,context,spot),context,spot,(write + channels : U32),room,
    column_inside)

# ---- Elements of a plane ----

# A plane by its phase and channel.
type Plane is Data:
  Plane{phase: Nat,channel: Nat}

def phase_of(+plane: Plane) -> Nat:
  Plane{+phase,channel} = plane
  phase

def channel_of(+plane: Plane) -> Nat:
  Plane{phase,+channel} = plane
  channel

# Plane phase * channels + channel, in the order the transpose takes them.
def plane_number(+plane: Plane,+channels: Nat) -> Nat:
  Plane{+phase,+channel} = plane
  outer(phase,channel,channels)

# Element row * tiles + column of a plane.
def element(+spot: Spot,+tiles: Nat) -> Nat:
  Spot{+row,+column} = spot
  outer(row,column,tiles)

# The pair of an element of a plane: (phase * elements + element) * channels
# + channel.
def plane_word(+plane: Plane,+elements: Nat,+channels: Nat,+index: Nat) -> U32:
  Plane{+phase,+channel} = plane
  U32.from_nat(radix(phase,index,channel,elements,channels))

# The pairs fit a tree of depth depth whose capacity lies within a cap.
type Room<-depth: Nat,-phases: Nat,-elements: Nat,-channels: Nat,-cap: U32> is Data:
  Room{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},limit: {Nat.is_le(depth,31n) == True{} : Bool},
    fits: {Nat.is_le(Nat.mul(Nat.mul(phases,elements),channels),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
    capped: {Nat.is_le(U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap)) == True{} : Bool},
    elements_positive: {Nat.is_lt(0n,elements) == True{} : Bool},channels_positive: {Nat.is_lt(0n,channels) == True{} : Bool}}

type PlaneInside<-plane: Plane,-phases: Nat,-channels: Nat> is Data:
  PlaneInside{phase_inside: {Nat.is_lt(phase_of(plane),phases) == True{} : Bool},channel_inside: {Nat.is_lt(channel_of(plane),channels) == True{} : Bool}}

def index_below(+plane: Plane,+phases: Nat,+elements: Nat,+channels: Nat,+index: Nat,+inside: PlaneInside<plane,phases,channels>,
  +index_inside: {Nat.is_lt(index,elements) == True{} : Bool}) ->
  {Nat.is_lt(radix(phase_of(plane),index,channel_of(plane),elements,channels),Nat.mul(Nat.mul(phases,elements),channels)) == True{} : Bool}:
  Plane{+phase,+channel} = plane
  PlaneInside{+phase_inside,+channel_inside} = inside
  RowMajor.radix(Nat.add(Nat.mul(phase,elements),index),Nat.mul(phases,elements),channel,channels,RowMajor.radix(phase,phases,index,elements,phase_inside,index_inside),
    channel_inside)

# An index below the planes lies inside the tree and within the cap.
def word_inside(+depth: Nat,+phases: Nat,+elements: Nat,+channels: Nat,+cap: U32,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),
  +room: Room<depth,phases,elements,channels,cap>,+index: Nat,+below: {Nat.is_lt(index,Nat.mul(Nat.mul(phases,elements),channels)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(index),Tree.capacity(Tile.Tile16,tree)) == True{} : Bool}:
  Room{+cap_ok,+limit,+fits,+capped,+elements_positive,+channels_positive} = room
  +capacity = U32.to_nat(U32.shln(1,depth))
  +under = Order.lt_le(index,Nat.mul(Nat.mul(phases,elements),channels),capacity,below,fits)
  %Equal.sym(U32,Tree.capacity(Tile.Tile16,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile16,depth,tree,balanced)) : {U32.is_lt(U32.from_nat(index),_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),U32.shln(1,depth)),Nat.is_lt(U32.to_nat(U32.from_nat(index)),capacity),Comparison.lt(U32.from_nat(index),U32.shln(1,depth))) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,
    Bounds.roundtrip(index,cap,cap_ok,Order.transitive(index,capacity,U32.to_nat(cap),Bounds.strict_le(index,capacity,under),capped))) : {Nat.is_lt(_,capacity) == True{} : Bool}
  under

def index_capped(+depth: Nat,+phases: Nat,+elements: Nat,+channels: Nat,+cap: U32,+room: Room<depth,phases,elements,channels,cap>,+index: Nat,
  +below: {Nat.is_lt(index,Nat.mul(Nat.mul(phases,elements),channels)) == True{} : Bool}) -> {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}:
  Room{+cap_ok,+limit,+fits,+capped,+elements_positive,+channels_positive} = room
  +capacity = U32.to_nat(U32.shln(1,depth))
  Order.transitive(index,capacity,U32.to_nat(cap),Bounds.strict_le(index,capacity,Order.lt_le(index,Nat.mul(Nat.mul(phases,elements),channels),capacity,below,fits)),capped)

# Two pairs of planes are one exactly when their planes and elements are.
def index_equal(+depth: Nat,+phases: Nat,+elements: Nat,+channels: Nat,+cap: U32,+plane: Plane,+other: Plane,+first: Nat,+second: Nat,
  +room: Room<depth,phases,elements,channels,cap>,+inside: PlaneInside<plane,phases,channels>,+other_inside: PlaneInside<other,phases,channels>,
  +first_inside: {Nat.is_lt(first,elements) == True{} : Bool},+second_inside: {Nat.is_lt(second,elements) == True{} : Bool}) ->
  {U32.is_eq(plane_word(plane,elements,channels,first),plane_word(other,elements,channels,second)) ==
    Nat.is_eq(plane_number(plane,channels),plane_number(other,channels)) && Nat.is_eq(first,second) : Bool}:
  Plane{+phase,+channel} = plane
  Plane{+other_phase,+other_channel} = other
  Room{+cap_ok,+limit,+fits,+capped,+elements_positive,+channels_positive} = room
  PlaneInside{+phase_inside,+channel_inside} = inside
  PlaneInside{+other_phase_inside,+other_channel_inside} = other_inside
  +plane = {Plane{phase,channel} : Plane}
  +other = {Plane{other_phase,other_channel} : Plane}
  +room = {Room{cap_ok,limit,fits,capped,elements_positive,channels_positive} : Room<depth,phases,elements,channels,cap>}
  +inside = {PlaneInside{phase_inside,channel_inside} : PlaneInside<plane,phases,channels>}
  +other_inside = {PlaneInside{other_phase_inside,other_channel_inside} : PlaneInside<other,phases,channels>}
  +first_index = radix(phase,first,channel,elements,channels)
  +second_index = radix(other_phase,second,other_channel,elements,channels)
  +first_below = index_below(plane,phases,elements,channels,first,inside,first_inside)
  +second_below = index_below(other,phases,elements,channels,second,other_inside,second_inside)
  Equal.trans(Bool,U32.is_eq(U32.from_nat(first_index),U32.from_nat(second_index)),Nat.is_eq(first_index,second_index),
    Nat.is_eq(outer(phase,channel,channels),outer(other_phase,other_channel,channels)) && Nat.is_eq(first,second),
    encoded_equal(first_index,second_index,cap,cap_ok,index_capped(depth,phases,elements,channels,cap,room,first_index,first_below),
      index_capped(depth,phases,elements,channels,cap,room,second_index,second_below)),
    radix_equal(phase,first,channel,other_phase,second,other_channel,elements,channels,Digits{elements_positive,channels_positive,first_inside,channel_inside},
      Digits{elements_positive,channels_positive,second_inside,other_channel_inside}))

# The next element of a plane lies channels pairs further.
def radix_next(+major: Nat,+middle: Nat,+minor: Nat,+middles: Nat,+minors: Nat) ->
  {radix(major,1n+middle,minor,middles,minors) == Nat.add(radix(major,middle,minor,middles,minors),minors) : Nat}:
  +head = Nat.mul(Nat.add(Nat.mul(major,middles),middle),minors)
  %Equal.sym(Nat,Nat.add(Nat.mul(major,middles),1n+middle),1n+Nat.add(Nat.mul(major,middles),middle),Addition.successor_right(Nat.mul(major,middles),middle)) :
    {Nat.add(Nat.mul(_,minors),minor) == Nat.add(radix(major,middle,minor,middles,minors),minors) : Nat}
  Equal.trans(Nat,Nat.add(Nat.add(minors,head),minor),Nat.add(minors,Nat.add(head,minor)),Nat.add(Nat.add(head,minor),minors),Addition.associative(minors,head,minor),
    Order.commutative(minors,Nat.add(head,minor)))

def write_next(+plane: Plane,+elements: Nat,+channels: U32,+index: Nat) ->
  {(plane_word(plane,elements,U32.to_nat(channels),index) + channels : U32) == plane_word(plane,elements,U32.to_nat(channels),1n+index) : U32}:
  Plane{+phase,+channel} = plane
  +count = U32.to_nat(channels)
  +here = radix(phase,index,channel,elements,count)
  Equal.trans(U32,(U32.from_nat(here) + channels : U32),U32.from_nat(Nat.add(here,count)),U32.from_nat(radix(phase,1n+index,channel,elements,count)),
    %Arithmetic.native_roundtrip(channels) : {(U32.from_nat(here) + _ : U32) == U32.from_nat(Nat.add(here,count)) : U32}
    WordArithmetic.native_add(here,count),
    Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(here,count),radix(phase,1n+index,channel,elements,count),
      Equal.sym(Nat,radix(phase,1n+index,channel,elements,count),Nat.add(here,count),radix_next(phase,index,channel,elements,count))))

def next_column(+row: Nat,+column: Nat,+tiles: Nat,+inside: Bool,+exact: {Nat.is_lt(1n+column,tiles) == inside : Bool},+positive: {Nat.is_lt(0n,tiles) == True{} : Bool}) ->
  {Nat.is_lt(column_of(Bool.pick(Spot,inside,Spot{row,1n+column},Spot{1n+row,0n})),tiles) == True{} : Bool}:
  match inside:
    case True{}: exact
    case False{}: positive

# The next spot lies inside the row.
def next_column_inside(+spot: Spot,+tiles: Nat,+positive: {Nat.is_lt(0n,tiles) == True{} : Bool}) -> {Nat.is_lt(column_of(next_spot(spot,tiles)),tiles) == True{} : Bool}:
  Spot{+row,+column} = spot
  next_column(row,column,tiles,Nat.is_lt(1n+column,tiles),{==},positive)

def next_count(+row: Nat,+column: Nat,+tiles: Nat,+inside: Bool,+exact: {Nat.is_lt(1n+column,tiles) == inside : Bool},+column_inside: {Nat.is_lt(column,tiles) == True{} : Bool}) ->
  {element(Bool.pick(Spot,inside,Spot{row,1n+column},Spot{1n+row,0n}),tiles) == 1n+outer(row,column,tiles) : Nat}:
  match inside:
    case True{}: Addition.successor_right(Nat.mul(row,tiles),column)
    case False{}:
      +full = DivisionOrder.not_below(column,tiles,column_inside,exact)
      +head = Nat.mul(row,tiles)
      Equal.trans(Nat,Nat.add(Nat.add(tiles,head),0n),Nat.add(head,tiles),1n+Nat.add(head,column),
        Equal.trans(Nat,Nat.add(Nat.add(tiles,head),0n),Nat.add(tiles,head),Nat.add(head,tiles),Addition.zero_right(Nat.add(tiles,head)),Order.commutative(tiles,head)),
        %full : {Nat.add(head,_) == 1n+Nat.add(head,column) : Nat}
        Addition.successor_right(head,column))

# The next spot is the next element.
def next_element(+spot: Spot,+tiles: Nat,+column_inside: {Nat.is_lt(column_of(spot),tiles) == True{} : Bool}) ->
  {element(next_spot(spot,tiles),tiles) == 1n+element(spot,tiles) : Nat}:
  Spot{+row,+column} = spot
  next_count(row,column,tiles,Nat.is_lt(1n+column,tiles),{==},column_inside)

# Spots of the same element are the same spot.
def spots_equal(+first: Spot,+second: Spot,+tiles: Nat,+positive: {Nat.is_lt(0n,tiles) == True{} : Bool},
  +first_inside: {Nat.is_lt(column_of(first),tiles) == True{} : Bool},+second_inside: {Nat.is_lt(column_of(second),tiles) == True{} : Bool},
  +same: {element(first,tiles) == element(second,tiles) : Nat}) -> {first == second : Spot}:
  Spot{+row,+column} = first
  Spot{+other_row,+other_column} = second
  %Radix.row_unique(row,other_row,tiles,column,other_column,positive,first_inside,second_inside,same) : {Spot{row,column} == Spot{_,other_column} : Spot}
  %Radix.col_unique(row,other_row,tiles,column,other_column,positive,first_inside,second_inside,same) : {Spot{row,column} == Spot{row,_} : Spot}
  {==}

# One write of a pair, read back at any pair of the planes.
def read_written(+depth: Nat,+phases: Nat,+elements: Nat,+channels: Nat,+cap: U32,+pairs: Tree.Tree<Tile.Tile16>,+plane: Plane,+other: Plane,+first: Nat,+second: Nat,
  +value: Tile.Tile16,+room: Room<depth,phases,elements,channels,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+inside: PlaneInside<plane,phases,channels>,
  +other_inside: PlaneInside<other,phases,channels>,+first_inside: {Nat.is_lt(first,elements) == True{} : Bool},+second_inside: {Nat.is_lt(second,elements) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Tree.write(Tile.Tile16,pairs,plane_word(plane,elements,channels,first),value),plane_word(other,elements,channels,second)) ==
    Bool.pick(Tile.Tile16,Nat.is_eq(plane_number(plane,channels),plane_number(other,channels)) && Nat.is_eq(first,second),value,
      Tree.read(Tile.Tile16,pairs,plane_word(other,elements,channels,second))) : Tile.Tile16}:
  Plane{+phase,+channel} = plane
  Plane{+other_phase,+other_channel} = other
  Room{+cap_ok,+limit,+fits,+capped,+elements_positive,+channels_positive} = room
  +plane = {Plane{phase,channel} : Plane}
  +other = {Plane{other_phase,other_channel} : Plane}
  +room = {Room{cap_ok,limit,fits,capped,elements_positive,channels_positive} : Room<depth,phases,elements,channels,cap>}
  +here = plane_word(plane,elements,channels,first)
  +there = plane_word(other,elements,channels,second)
  +old = Tree.read(Tile.Tile16,pairs,there)
  Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,Tree.write(Tile.Tile16,pairs,here,value),there),Bool.pick(Tile.Tile16,U32.is_eq(here,there),value,old),
    Bool.pick(Tile.Tile16,Nat.is_eq(outer(phase,channel,channels),outer(other_phase,other_channel,channels)) && Nat.is_eq(first,second),value,old),
    Readback.one_full(~Tile.Tile16,depth,pairs,here,there,value,balanced,limit,
      word_inside(depth,phases,elements,channels,cap,pairs,balanced,room,radix(phase,first,channel,elements,channels),index_below(plane,phases,elements,channels,first,inside,first_inside)),
      word_inside(depth,phases,elements,channels,cap,pairs,balanced,room,radix(other_phase,second,other_channel,elements,channels),
        index_below(other,phases,elements,channels,second,other_inside,second_inside))),
    Equal.cong(Bool,Tile.Tile16,chosen => Bool.pick(Tile.Tile16,chosen,value,old),U32.is_eq(here,there),
      Nat.is_eq(outer(phase,channel,channels),outer(other_phase,other_channel,channels)) && Nat.is_eq(first,second),
      index_equal(depth,phases,elements,channels,cap,plane,other,first,second,room,inside,other_inside,first_inside,second_inside)))

# ---- The row loop of a plane ----

# The pair the row loop writes for the element of a spot.
def pair(~lane: Tile.Tile8 -> F32,+context: Row,+spot: Spot) -> Tile.Tile16:
  Tile.Tiles{tile(~lane,context,spot),tile(~lane,context,next_spot(spot,row_tiles(context)))}

# The row loop of a plane past the tile at previous: the element of previous
# is the next one written.
def walking(~lane: Tile.Tile8 -> F32,pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+previous: Spot) -> Model.Rowing:
  +tiles = row_tiles(context)
  ready(pairs,tile(~lane,context,previous),context,next_spot(previous,tiles),plane_word(plane,elements,U32.to_nat(channels_of(context)),element(previous,tiles)))

def room_positive(+context: Row,+room: RowRoom<context>) -> {Nat.is_lt(0n,row_tiles(context)) == True{} : Bool}:
  RowRoom{cap,cap_ok,tiles_bound,+positive} = room
  positive

def walk_step(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+previous: Spot,+row_room: RowRoom<context>,
  +column_inside: {Nat.is_lt(column_of(previous),row_tiles(context)) == True{} : Bool}) ->
  {Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)) ==
    walking(~lane,Tree.write(Tile.Tile16,pairs,plane_word(plane,elements,U32.to_nat(channels_of(context)),element(previous,row_tiles(context))),pair(~lane,context,previous)),
      context,plane,elements,next_spot(previous,row_tiles(context))) : Model.Rowing}:
  +tiles = row_tiles(context)
  +count = U32.to_nat(channels_of(context))
  +next = next_spot(previous,tiles)
  +here = element(previous,tiles)
  +word = plane_word(plane,elements,count,here)
  +written = Tree.write(Tile.Tile16,pairs,word,pair(~lane,context,previous))
  Equal.trans(Model.Rowing,Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)),
    ready(written,tile(~lane,context,next),context,next_spot(next,tiles),(word + channels_of(context) : U32)),walking(~lane,written,context,plane,elements,next),
    step_ready(~lane,pairs,tile(~lane,context,previous),context,next,word,row_room,next_column_inside(previous,tiles,room_positive(context,row_room))),
    Equal.cong(U32,Model.Rowing,place => ready(written,tile(~lane,context,next),context,next_spot(next,tiles),place),(word + channels_of(context) : U32),
      plane_word(plane,elements,count,element(next,tiles)),
      Equal.trans(U32,(word + channels_of(context) : U32),plane_word(plane,elements,count,1n+here),plane_word(plane,elements,count,element(next,tiles)),
        write_next(plane,elements,channels_of(context),here),
        Equal.cong(Nat,U32,index => plane_word(plane,elements,count,index),1n+here,element(next,tiles),
          Equal.sym(Nat,element(next,tiles),1n+here,next_element(previous,tiles,column_inside))))))

# A false order excludes equality and the next order.
def not_equal_of_not_le(+left: Nat,+right: Nat,+value: Bool,+exact: {Nat.is_eq(left,right) == value : Bool},+missed: {Nat.is_le(left,right) == False{} : Bool}) ->
  {value == False{} : Bool}:
  match value:
    case False{}: {==}
    case True{}:
      +same = Order.equal_of_is_eq(left,right,exact)
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(left,right),True{},Equal.sym(Bool,Nat.is_le(left,right),False{},missed),
        %same : {Nat.is_le(left,_) == True{} : Bool} Order.reflexive(left))))

def not_le_next(+left: Nat,+right: Nat,+value: Bool,+exact: {Nat.is_le(1n+left,right) == value : Bool},+missed: {Nat.is_le(left,right) == False{} : Bool}) ->
  {value == False{} : Bool}:
  match value:
    case False{}: {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(left,right),True{},Equal.sym(Bool,Nat.is_le(left,right),False{},missed),
        Order.transitive(left,1n+left,right,Order.successor(left,left,Order.reflexive(left)),exact))))

def missed_now(+same: Bool,+left: Nat,+right: Nat,+missed: {same && Nat.is_le(left,right) == False{} : Bool}) -> {same && Nat.is_eq(left,right) == False{} : Bool}:
  match same:
    case False{}: {==}
    case True{}: not_equal_of_not_le(left,right,Nat.is_eq(left,right),{==},missed)

def missed_next(+same: Bool,+left: Nat,+right: Nat,+missed: {same && Nat.is_le(left,right) == False{} : Bool}) -> {same && Nat.is_le(1n+left,right) == False{} : Bool}:
  match same:
    case False{}: {==}
    case True{}: not_le_next(left,right,Nat.is_le(1n+left,right),{==},missed)

def not_le_successor(+value: Nat) -> {Nat.is_le(1n+value,value) == False{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+ +previous: not_le_successor(previous)

def below_of_unequal(+left: Nat,+right: Nat,+ordered: {Nat.is_le(left,right) == True{} : Bool},+apart: {Nat.is_eq(left,right) == False{} : Bool}) ->
  {Nat.is_lt(left,right) == True{} : Bool}:
  match left right:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case 0n 1n+right: {==}
    case 1n+left 0n: Empty.absurd({Nat.is_lt(1n+left,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+ +left 1n+ +right: below_of_unequal(left,right,ordered,apart)

# Writes of another plane, or of later elements of the same plane, leave a
# pair as it was.
def rows_missed(~lane: Tile.Tile8 -> F32,n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+previous: Spot,+queried: Nat,
  +depth: Nat,+phases: Nat,+elements: Nat,+cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,
  +other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,+queried_inside: {Nat.is_lt(queried,elements) == True{} : Bool},
  +column_inside: {Nat.is_lt(column_of(previous),row_tiles(context)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(element(previous,row_tiles(context)),n),elements) == True{} : Bool},
  +missed: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) &&
    Nat.is_le(element(previous,row_tiles(context)),queried) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),n,walking(~lane,pairs,context,plane,elements,previous))),
    plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) == Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) :
    Tile.Tile16}:
  match n:
    case 0n:
      Equal.cong(Tree.Tree<Tile.Tile16>,Tile.Tile16,tree => Tree.read(Tile.Tile16,tree,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)),
        rowing_pairs(walking(~lane,pairs,context,plane,elements,previous)),pairs,
        ready_pairs(pairs,tile(~lane,context,previous),context,next_spot(previous,row_tiles(context)),
          plane_word(plane,elements,U32.to_nat(channels_of(context)),element(previous,row_tiles(context)))))
    case 1n+ +rest:
      +tiles = row_tiles(context)
      +count = U32.to_nat(channels_of(context))
      +here = element(previous,tiles)
      +query = plane_word(other,elements,count,queried)
      +value = pair(~lane,context,previous)
      +word = plane_word(plane,elements,count,here)
      +written = Tree.write(Tile.Tile16,pairs,word,value)
      +next = next_spot(previous,tiles)
      +same = Nat.is_eq(plane_number(plane,count),plane_number(other,count))
      +advanced = next_element(previous,tiles,column_inside)
      %Equal.sym(Model.Rowing,Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)),walking(~lane,written,context,plane,elements,next),
        walk_step(~lane,pairs,context,plane,elements,previous,row_room,column_inside)) :
        {Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,_)),query) == Tree.read(Tile.Tile16,pairs,query) : Tile.Tile16}
      Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,walking(~lane,written,context,plane,elements,next))),query),
        Tree.read(Tile.Tile16,written,query),Tree.read(Tile.Tile16,pairs,query),
        rows_missed(~lane,rest,written,context,plane,other,next,queried,depth,phases,elements,cap,room,row_room,Shape.write(Tile.Tile16,depth,pairs,word,value,balanced),inside,
          other_inside,queried_inside,next_column_inside(previous,tiles,room_positive(context,row_room)),
          %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),elements) == True{} : Bool} Order.advance_room(here,rest,elements,remaining),
          %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {same && Nat.is_le(_,queried) == False{} : Bool} missed_next(same,here,queried,missed)),
        Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,written,query),Bool.pick(Tile.Tile16,same && Nat.is_eq(here,queried),value,Tree.read(Tile.Tile16,pairs,query)),
          Tree.read(Tile.Tile16,pairs,query),
          read_written(depth,phases,elements,count,cap,pairs,plane,other,here,queried,value,room,balanced,inside,other_inside,Order.room_index(here,rest,elements,remaining),
            queried_inside),
          Equal.cong(Bool,Tile.Tile16,chosen => Bool.pick(Tile.Tile16,chosen,value,Tree.read(Tile.Tile16,pairs,query)),same && Nat.is_eq(here,queried),False{},
            missed_now(same,here,queried,missed))))

# The pair of the queried spot, written by the row loop of its plane.
def rows_hit(~lane: Tile.Tile8 -> F32,n: Nat,hit: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+previous: Spot,+spot: Spot,
  +depth: Nat,+phases: Nat,+elements: Nat,+cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,
  +other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,+spot_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool},
  +queried_inside: {Nat.is_lt(element(spot,row_tiles(context)),elements) == True{} : Bool},+column_inside: {Nat.is_lt(column_of(previous),row_tiles(context)) == True{} : Bool},
  +same: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == True{} : Bool},
  +exact: {Nat.is_eq(element(previous,row_tiles(context)),element(spot,row_tiles(context))) == hit : Bool},
  +reached: {Nat.is_le(element(previous,row_tiles(context)),element(spot,row_tiles(context))) == True{} : Bool},
  +within: {Nat.is_lt(element(spot,row_tiles(context)),Nat.add(element(previous,row_tiles(context)),n)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(element(previous,row_tiles(context)),n),elements) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),n,walking(~lane,pairs,context,plane,elements,previous))),
    plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == pair(~lane,context,spot) : Tile.Tile16}:
  match n:
    case 0n:
      +here = element(previous,row_tiles(context))
      +before = Order.le_lt_trans(here,element(spot,row_tiles(context)),here,reached,%Addition.zero_right(here) : {Nat.is_lt(element(spot,row_tiles(context)),_) == True{} : Bool} within)
      Empty.absurd({Tree.read(Tile.Tile16,rowing_pairs(walking(~lane,pairs,context,plane,elements,previous)),
          plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == pair(~lane,context,spot) : Tile.Tile16},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(here,here),True{},Equal.sym(Bool,Nat.is_eq(here,here),False{},Order.less_not_equal(here,here,before)),
          Order.equal_reflexive(here))))
    case 1n+ +rest:
      match hit:
        case True{}:
          +tiles = row_tiles(context)
          +count = U32.to_nat(channels_of(context))
          +here = element(previous,tiles)
          +queried = element(spot,tiles)
          +query = plane_word(other,elements,count,queried)
          +value = pair(~lane,context,previous)
          +word = plane_word(plane,elements,count,here)
          +written = Tree.write(Tile.Tile16,pairs,word,value)
          +next = next_spot(previous,tiles)
          +advanced = next_element(previous,tiles,column_inside)
          +equal = Order.equal_of_is_eq(here,queried,exact)
          %Equal.sym(Model.Rowing,Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)),walking(~lane,written,context,plane,elements,next),
            walk_step(~lane,pairs,context,plane,elements,previous,row_room,column_inside)) :
            {Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,_)),query) == pair(~lane,context,spot) : Tile.Tile16}
          Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,walking(~lane,written,context,plane,elements,next))),query),
            Tree.read(Tile.Tile16,written,query),pair(~lane,context,spot),
            rows_missed(~lane,rest,written,context,plane,other,next,queried,depth,phases,elements,cap,room,row_room,Shape.write(Tile.Tile16,depth,pairs,word,value,balanced),inside,
              other_inside,queried_inside,next_column_inside(previous,tiles,room_positive(context,row_room)),
              %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),elements) == True{} : Bool} Order.advance_room(here,rest,elements,remaining),
              %Equal.sym(Bool,Nat.is_eq(plane_number(plane,count),plane_number(other,count)),True{},same) : {_ && Nat.is_le(element(next,tiles),queried) == False{} : Bool}
              %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_le(_,queried) == False{} : Bool}
              %equal : {Nat.is_le(1n+here,_) == False{} : Bool}
              not_le_successor(here)),
            Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,written,query),
              Bool.pick(Tile.Tile16,Nat.is_eq(plane_number(plane,count),plane_number(other,count)) && Nat.is_eq(here,queried),value,Tree.read(Tile.Tile16,pairs,query)),
              pair(~lane,context,spot),
              read_written(depth,phases,elements,count,cap,pairs,plane,other,here,queried,value,room,balanced,inside,other_inside,
                Order.room_index(here,rest,elements,remaining),queried_inside),
              %Equal.sym(Bool,Nat.is_eq(plane_number(plane,count),plane_number(other,count)),True{},same) :
                {Bool.pick(Tile.Tile16,_ && Nat.is_eq(here,queried),value,Tree.read(Tile.Tile16,pairs,query)) == pair(~lane,context,spot) : Tile.Tile16}
              %Equal.sym(Bool,Nat.is_eq(here,queried),True{},exact) : {Bool.pick(Tile.Tile16,True{} && _,value,Tree.read(Tile.Tile16,pairs,query)) == pair(~lane,context,spot) : Tile.Tile16}
              Equal.cong(Spot,Tile.Tile16,place => pair(~lane,context,place),previous,spot,
                spots_equal(previous,spot,tiles,room_positive(context,row_room),column_inside,spot_inside,equal))))
        case False{}:
          +tiles = row_tiles(context)
          +count = U32.to_nat(channels_of(context))
          +here = element(previous,tiles)
          +queried = element(spot,tiles)
          +query = plane_word(other,elements,count,queried)
          +value = pair(~lane,context,previous)
          +word = plane_word(plane,elements,count,here)
          +written = Tree.write(Tile.Tile16,pairs,word,value)
          +next = next_spot(previous,tiles)
          +advanced = next_element(previous,tiles,column_inside)
          %Equal.sym(Model.Rowing,Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)),walking(~lane,written,context,plane,elements,next),
            walk_step(~lane,pairs,context,plane,elements,previous,row_room,column_inside)) :
            {Tree.read(Tile.Tile16,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,_)),query) == pair(~lane,context,spot) : Tile.Tile16}
          rows_hit(~lane,rest,Nat.is_eq(element(next,tiles),queried),written,context,plane,other,next,spot,depth,phases,elements,cap,room,row_room,
            Shape.write(Tile.Tile16,depth,pairs,word,value,balanced),inside,other_inside,spot_inside,queried_inside,next_column_inside(previous,tiles,room_positive(context,row_room)),
            same,{==},
            %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_le(_,queried) == True{} : Bool}
              Order.strict_follow(here,queried,below_of_unequal(here,queried,reached,exact)),
            %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_lt(queried,Nat.add(_,rest)) == True{} : Bool}
              %Addition.successor_right(here,rest) : {Nat.is_lt(queried,_) == True{} : Bool} within,
            %Equal.sym(Nat,element(next,tiles),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),elements) == True{} : Bool} Order.advance_room(here,rest,elements,remaining))

# A row loop keeps its pairs a tree of the same depth.
def rows_balanced(~lane: Tile.Tile8 -> F32,n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+previous: Spot,+depth: Nat,
  +row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+column_inside: {Nat.is_lt(column_of(previous),row_tiles(context)) == True{} : Bool}) ->
  Shape.Balanced(Tile.Tile16,depth,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),n,walking(~lane,pairs,context,plane,elements,previous)))):
  match n:
    case 0n:
      %Equal.sym(Tree.Tree<Tile.Tile16>,rowing_pairs(walking(~lane,pairs,context,plane,elements,previous)),pairs,
        ready_pairs(pairs,tile(~lane,context,previous),context,next_spot(previous,row_tiles(context)),
          plane_word(plane,elements,U32.to_nat(channels_of(context)),element(previous,row_tiles(context))))) : Shape.Balanced(Tile.Tile16,depth,_)
      balanced
    case 1n+ +rest:
      +tiles = row_tiles(context)
      +word = plane_word(plane,elements,U32.to_nat(channels_of(context)),element(previous,tiles))
      +written = Tree.write(Tile.Tile16,pairs,word,pair(~lane,context,previous))
      %Equal.sym(Model.Rowing,Model.row_step(~lane,walking(~lane,pairs,context,plane,elements,previous)),walking(~lane,written,context,plane,elements,next_spot(previous,tiles)),
        walk_step(~lane,pairs,context,plane,elements,previous,row_room,column_inside)) :
        Shape.Balanced(Tile.Tile16,depth,rowing_pairs(Loop.run(~Model.Rowing,~Model.row_step(~lane),rest,_)))
      rows_balanced(~lane,rest,written,context,plane,elements,next_spot(previous,tiles),depth,row_room,Shape.write(Tile.Tile16,depth,pairs,word,pair(~lane,context,previous),balanced),
        next_column_inside(previous,tiles,room_positive(context,row_room)))

# ---- A plane from its first tile ----

# The row loop of a plane: the first read readies tile zero, then elements
# reads.
def started(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat) -> Model.Rowing:
  Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,
    Model.row_first(~lane,ready(pairs,Tile.filled(0.0),context,Spot{0n,0n},plane_word(plane,elements,U32.to_nat(channels_of(context)),0n))))

def started_walking(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+row_room: RowRoom<context>) ->
  {started(~lane,pairs,context,plane,elements) == Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,walking(~lane,pairs,context,plane,elements,Spot{0n,0n})) : Model.Rowing}:
  Equal.cong(Model.Rowing,Model.Rowing,state => Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,state),
    Model.row_first(~lane,ready(pairs,Tile.filled(0.0),context,Spot{0n,0n},plane_word(plane,elements,U32.to_nat(channels_of(context)),0n))),
    walking(~lane,pairs,context,plane,elements,Spot{0n,0n}),
    first_ready(~lane,pairs,Tile.filled(0.0),context,Spot{0n,0n},plane_word(plane,elements,U32.to_nat(channels_of(context)),0n),row_room,room_positive(context,row_room)))

# The plane of the query writes the pair of its spot.
def started_hit(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+spot: Spot,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +spot_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,row_tiles(context)),elements) == True{} : Bool},
  +same: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(started(~lane,pairs,context,plane,elements)),plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) ==
    pair(~lane,context,spot) : Tile.Tile16}:
  %Equal.sym(Model.Rowing,started(~lane,pairs,context,plane,elements),
    Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,walking(~lane,pairs,context,plane,elements,Spot{0n,0n})),started_walking(~lane,pairs,context,plane,elements,row_room)) :
    {Tree.read(Tile.Tile16,rowing_pairs(_),plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == pair(~lane,context,spot) : Tile.Tile16}
  rows_hit(~lane,elements,Nat.is_eq(0n,element(spot,row_tiles(context))),pairs,context,plane,other,Spot{0n,0n},spot,depth,phases,elements,cap,room,row_room,balanced,inside,
    other_inside,spot_inside,queried_inside,room_positive(context,row_room),same,{==},Order.zero_le(element(spot,row_tiles(context))),queried_inside,Order.reflexive(elements))

# Another plane leaves the pair of the query.
def started_missed(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+queried: Nat,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +queried_inside: {Nat.is_lt(queried,elements) == True{} : Bool},
  +apart: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(started(~lane,pairs,context,plane,elements)),plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) : Tile.Tile16}:
  %Equal.sym(Model.Rowing,started(~lane,pairs,context,plane,elements),
    Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,walking(~lane,pairs,context,plane,elements,Spot{0n,0n})),started_walking(~lane,pairs,context,plane,elements,row_room)) :
    {Tree.read(Tile.Tile16,rowing_pairs(_),plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) ==
      Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) : Tile.Tile16}
  rows_missed(~lane,elements,pairs,context,plane,other,Spot{0n,0n},queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,
    room_positive(context,row_room),Order.reflexive(elements),
    %Equal.sym(Bool,Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))),False{},apart) :
      {_ && Nat.is_le(0n,queried) == False{} : Bool}
    {==})

def started_balanced(~lane: Tile.Tile8 -> F32,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+depth: Nat,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) -> Shape.Balanced(Tile.Tile16,depth,rowing_pairs(started(~lane,pairs,context,plane,elements))):
  %Equal.sym(Model.Rowing,started(~lane,pairs,context,plane,elements),
    Loop.run(~Model.Rowing,~Model.row_step(~lane),elements,walking(~lane,pairs,context,plane,elements,Spot{0n,0n})),started_walking(~lane,pairs,context,plane,elements,row_room)) :
    Shape.Balanced(Tile.Tile16,depth,rowing_pairs(_))
  rows_balanced(~lane,elements,pairs,context,plane,elements,Spot{0n,0n},depth,row_room,balanced,room_positive(context,row_room))

# The pair of a spot, the lane named by a pattern of bits.
def lanes_pair(+context: Row,+lanes: Lanes.Bits,+spot: Spot) -> Tile.Tile16:
  Tile.Tiles{lanes_tile(context,lanes,spot),lanes_tile(context,lanes,next_spot(spot,row_tiles(context)))}

# The row loop of a plane, its lane chosen by the bits of its channel.
def lane_rows(+half: Bool,+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat) -> Model.Rowing:
  Model.row_lane(half,low,first,elements,ready(pairs,Tile.filled(0.0),context,Spot{0n,0n},plane_word(plane,elements,U32.to_nat(channels_of(context)),0n)))

def lanes_missed_low(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+queried: Nat,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +queried_inside: {Nat.is_lt(queried,elements) == True{} : Bool},
  +apart: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(True{},low,first,pairs,context,plane,elements)),plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) : Tile.Tile16}:
  match low first:
    case True{} True{}:
      started_missed(~Tile.lane0,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case True{} False{}:
      started_missed(~Tile.lane1,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case False{} True{}:
      started_missed(~Tile.lane2,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case False{} False{}:
      started_missed(~Tile.lane3,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)

def lanes_missed_high(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+queried: Nat,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +queried_inside: {Nat.is_lt(queried,elements) == True{} : Bool},
  +apart: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(False{},low,first,pairs,context,plane,elements)),plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) : Tile.Tile16}:
  match low first:
    case True{} True{}:
      started_missed(~Tile.lane4,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case True{} False{}:
      started_missed(~Tile.lane5,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case False{} True{}:
      started_missed(~Tile.lane6,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case False{} False{}:
      started_missed(~Tile.lane7,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)

# Another plane leaves the pair of the query, whatever lane its channel reads.
def lanes_missed(+half: Bool,+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+queried: Nat,+depth: Nat,+phases: Nat,
  +elements: Nat,+cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,
  +other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,+queried_inside: {Nat.is_lt(queried,elements) == True{} : Bool},
  +apart: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(half,low,first,pairs,context,plane,elements)),plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(other,elements,U32.to_nat(channels_of(context)),queried)) : Tile.Tile16}:
  match half:
    case True{}: lanes_missed_low(low,first,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)
    case False{}: lanes_missed_high(low,first,pairs,context,plane,other,queried,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,queried_inside,apart)

def lanes_balanced_low(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+depth: Nat,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) -> Shape.Balanced(Tile.Tile16,depth,rowing_pairs(lane_rows(True{},low,first,pairs,context,plane,elements))):
  match low first:
    case True{} True{}: started_balanced(~Tile.lane0,pairs,context,plane,elements,depth,row_room,balanced)
    case True{} False{}: started_balanced(~Tile.lane1,pairs,context,plane,elements,depth,row_room,balanced)
    case False{} True{}: started_balanced(~Tile.lane2,pairs,context,plane,elements,depth,row_room,balanced)
    case False{} False{}: started_balanced(~Tile.lane3,pairs,context,plane,elements,depth,row_room,balanced)

def lanes_balanced_high(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+depth: Nat,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) -> Shape.Balanced(Tile.Tile16,depth,rowing_pairs(lane_rows(False{},low,first,pairs,context,plane,elements))):
  match low first:
    case True{} True{}: started_balanced(~Tile.lane4,pairs,context,plane,elements,depth,row_room,balanced)
    case True{} False{}: started_balanced(~Tile.lane5,pairs,context,plane,elements,depth,row_room,balanced)
    case False{} True{}: started_balanced(~Tile.lane6,pairs,context,plane,elements,depth,row_room,balanced)
    case False{} False{}: started_balanced(~Tile.lane7,pairs,context,plane,elements,depth,row_room,balanced)

def lanes_balanced(+half: Bool,+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+elements: Nat,+depth: Nat,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) -> Shape.Balanced(Tile.Tile16,depth,rowing_pairs(lane_rows(half,low,first,pairs,context,plane,elements))):
  match half:
    case True{}: lanes_balanced_low(low,first,pairs,context,plane,elements,depth,row_room,balanced)
    case False{}: lanes_balanced_high(low,first,pairs,context,plane,elements,depth,row_room,balanced)

def lanes_hit_low(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+spot: Spot,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +spot_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,row_tiles(context)),elements) == True{} : Bool},
  +same: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(True{},low,first,pairs,context,plane,elements)),
    plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == lanes_pair(context,Lanes.Bits{True{},low,first},spot) : Tile.Tile16}:
  match low first:
    case True{} True{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane0,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case True{} False{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane1,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case False{} True{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane2,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case False{} False{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane3,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)

def lanes_hit_high(+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+spot: Spot,+depth: Nat,+phases: Nat,+elements: Nat,
  +cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,+other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,
  +spot_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,row_tiles(context)),elements) == True{} : Bool},
  +same: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(False{},low,first,pairs,context,plane,elements)),
    plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == lanes_pair(context,Lanes.Bits{False{},low,first},spot) : Tile.Tile16}:
  match low first:
    case True{} True{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane4,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case True{} False{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane5,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case False{} True{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane6,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case False{} False{}:
      Row{+input,+first_row,+first_column,+row_tiles,+channel_base,+height,+width,+stride,+groups,+channels} = context
      started_hit(~Tile.lane7,pairs,Row{input,first_row,first_column,row_tiles,channel_base,height,width,stride,groups,channels},plane,other,spot,depth,phases,elements,cap,room,
        row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)

# The plane of the query writes the pair of its spot, the lane its channel's
# bits name.
def lanes_hit(+half: Bool,+low: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+context: Row,+plane: Plane,+other: Plane,+spot: Spot,+depth: Nat,+phases: Nat,
  +elements: Nat,+cap: U32,+room: Room<depth,phases,elements,U32.to_nat(channels_of(context)),cap>,+row_room: RowRoom<context>,
  +balanced: Shape.Balanced(Tile.Tile16,depth,pairs),+inside: PlaneInside<plane,phases,U32.to_nat(channels_of(context))>,
  +other_inside: PlaneInside<other,phases,U32.to_nat(channels_of(context))>,+spot_inside: {Nat.is_lt(column_of(spot),row_tiles(context)) == True{} : Bool},
  +queried_inside: {Nat.is_lt(element(spot,row_tiles(context)),elements) == True{} : Bool},
  +same: {Nat.is_eq(plane_number(plane,U32.to_nat(channels_of(context))),plane_number(other,U32.to_nat(channels_of(context)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(lane_rows(half,low,first,pairs,context,plane,elements)),
    plane_word(other,elements,U32.to_nat(channels_of(context)),element(spot,row_tiles(context)))) == lanes_pair(context,Lanes.Bits{half,low,first},spot) : Tile.Tile16}:
  match half:
    case True{}:
      lanes_hit_low(low,first,pairs,context,plane,other,spot,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)
    case False{}:
      lanes_hit_high(low,first,pairs,context,plane,other,spot,depth,phases,elements,cap,room,row_room,balanced,inside,other_inside,spot_inside,queried_inside,same)

# ---- The planes ----

def phases_of(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat(stride)

def channel_count(+window: Packing.Stencil) -> Nat:
  U32.to_nat(Packing.channels_of(window))

# The planes the transpose writes: the stride's phases times the channels.
def plane_count(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat((stride * Packing.channels_of(window) : U32))

def elements_of(+planes: Packing.Planes) -> Nat:
  Packing.Planes{first_output_row,rows,row_width,+elements} = planes
  U32.to_nat(elements)

def tiles_of(+planes: Packing.Planes) -> Nat:
  Packing.Planes{first_output_row,rows,+row_width,elements} = planes
  U32.to_nat((row_width / 8 : U32))

# The rows of plane number plane: phase plane / channels, channel plane %
# channels.
def context_of(+window: Packing.Stencil,+planes: Packing.Planes,+input: Tree.Tree<Tile.Tile8>,+plane: U32) -> Row:
  Packing.Stencil{+base,+height,+width,tap_rows,tap_columns,+stride,+row_origin,+column_origin,output_width,inner,+groups} = window
  Packing.Planes{+first_output_row,rows,+row_width,elements} = planes
  +channels = Packing.channels_of(window)
  +channel = (plane % channels : U32)
  +phase = (plane / channels : U32)
  Row{input,(first_output_row * stride + row_origin : U32),(phase + column_origin : U32),(row_width / 8 : U32),(base + channel / 8 : U32),height,width,stride,groups,channels}

def plane_of(+number: Nat,+channels: Nat) -> Plane:
  Plane{Nat.div(number,channels),Nat.mod(number,channels)}

# What the planes need: the pairs of all planes fit the tree and the cap, a
# plane row has at least one tile and the planes count the stride's phases
# times the channels.
type Geometry<-window: Packing.Stencil,-planes: Packing.Planes,-depth: Nat,-cap: U32> is Data:
  Geometry{room: Room<depth,phases_of(window),elements_of(planes),channel_count(window),cap>,
    tiles_bound: {Nat.is_le(tiles_of(planes),U32.to_nat(cap)) == True{} : Bool},tiles_positive: {Nat.is_lt(0n,tiles_of(planes)) == True{} : Bool},
    divisor_bound: {Nat.is_le(Nat.double(channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
    planes_bound: {Nat.is_le(Nat.mul(phases_of(window),channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
    planes_exact: {plane_count(window) == Nat.mul(phases_of(window),channel_count(window)) : Nat}}

# The word of plane (major, 0, minor).
def encoded_plane_word(+major: Nat,+elements: Nat,+channels: Nat,+minor: Nat) ->
  {(U32.from_nat(major) * U32.from_nat(elements) * U32.from_nat(channels) + U32.from_nat(minor) : U32) == U32.from_nat(radix(major,0n,minor,elements,channels)) : U32}:
  +product = Nat.mul(major,elements)
  %Equal.sym(U32,(U32.from_nat(major) * U32.from_nat(elements) : U32),U32.from_nat(product),Multiplication.native_encoded_multiply(major,elements)) :
    {(_ * U32.from_nat(channels) + U32.from_nat(minor) : U32) == U32.from_nat(radix(major,0n,minor,elements,channels)) : U32}
  %Equal.sym(U32,(U32.from_nat(product) * U32.from_nat(channels) : U32),U32.from_nat(Nat.mul(product,channels)),Multiplication.native_encoded_multiply(product,channels)) :
    {(_ + U32.from_nat(minor) : U32) == U32.from_nat(radix(major,0n,minor,elements,channels)) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.mul(product,channels)) + U32.from_nat(minor) : U32),U32.from_nat(Nat.add(Nat.mul(product,channels),minor)),
    WordArithmetic.native_add(Nat.mul(product,channels),minor)) : {_ == U32.from_nat(radix(major,0n,minor,elements,channels)) : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(Nat.mul(product,channels),minor),radix(major,0n,minor,elements,channels),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,channels),minor),product,Nat.add(product,0n),Equal.sym(Nat,Nat.add(product,0n),product,Addition.zero_right(product))))

# The first pair a plane writes.
def start_word(+number: Nat,+channels: U32,+elements: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +number_room: {Nat.is_le(number,U32.to_nat(cap)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(channels)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}) ->
  {((U32.from_nat(number) / channels : U32) * elements * channels + (U32.from_nat(number) % channels : U32) : U32) ==
    plane_word(plane_of(number,U32.to_nat(channels)),U32.to_nat(elements),U32.to_nat(channels),0n) : U32}:
  +count = U32.to_nat(channels)
  +size = U32.to_nat(elements)
  +word = U32.from_nat(number)
  +count_room = Order.transitive(count,Nat.double(count),U32.to_nat(cap),Order.le_double(count),divisor_bound)
  Equal.trans(U32,((word / channels : U32) * elements * channels + (word % channels : U32) : U32),
    ((word / U32.from_nat(count) : U32) * U32.from_nat(size) * U32.from_nat(count) + (word % U32.from_nat(count) : U32) : U32),
    plane_word(plane_of(number,count),size,count,0n),
    Equal.trans(U32,((word / channels : U32) * elements * channels + (word % channels : U32) : U32),
      ((word / U32.from_nat(count) : U32) * elements * U32.from_nat(count) + (word % U32.from_nat(count) : U32) : U32),
      ((word / U32.from_nat(count) : U32) * U32.from_nat(size) * U32.from_nat(count) + (word % U32.from_nat(count) : U32) : U32),
      Equal.cong(U32,U32,divisor => ((word / divisor : U32) * elements * divisor + (word % divisor : U32) : U32),channels,U32.from_nat(count),
        Equal.sym(U32,U32.from_nat(count),channels,Arithmetic.native_roundtrip(channels))),
      Equal.cong(U32,U32,length => ((word / U32.from_nat(count) : U32) * length * U32.from_nat(count) + (word % U32.from_nat(count) : U32) : U32),elements,U32.from_nat(size),
        Equal.sym(U32,U32.from_nat(size),elements,Arithmetic.native_roundtrip(elements)))),
    %Equal.sym(U32,(word / U32.from_nat(count) : U32),U32.from_nat(Nat.div(number,count)),WordDivision.encoded_divide(number,count,cap,cap_ok,number_room,count_room,divisor_bound,
      positive)) :
      {(_ * U32.from_nat(size) * U32.from_nat(count) + (word % U32.from_nat(count) : U32) : U32) == plane_word(plane_of(number,count),size,count,0n) : U32}
    %Equal.sym(U32,(word % U32.from_nat(count) : U32),U32.from_nat(Nat.mod(number,count)),WordDivision.encoded_modulo(number,count,cap,cap_ok,number_room,count_room,divisor_bound,
      positive)) :
      {(U32.from_nat(Nat.div(number,count)) * U32.from_nat(size) * U32.from_nat(count) + _ : U32) == plane_word(plane_of(number,count),size,count,0n) : U32}
    encoded_plane_word(Nat.div(number,count),size,count,Nat.mod(number,count)))

# A plane starts ready to read its tile zero.
def start_ready(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+depth: Nat,+cap: U32,
  +geometry: Geometry<window,planes,depth,cap>,+number_inside: {Nat.is_lt(number,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool}) ->
  {Model.plane_start(input,pairs,U32.from_nat(number),window,planes) ==
    ready(pairs,Tile.filled(0.0),context_of(window,planes,input,U32.from_nat(number)),Spot{0n,0n},
      plane_word(plane_of(number,channel_count(window)),elements_of(planes),channel_count(window),0n)) : Model.Rowing}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  Geometry{+room,+tiles_bound,+tiles_positive,+divisor_bound,+planes_bound,+planes_exact} = geometry
  Room{+cap_ok,+limit,+fits,+capped,+elements_positive,+channels_positive} = room
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +channels = Packing.channels_of(window)
  +word = U32.from_nat(number)
  +context = context_of(window,planes,input,word)
  +phase = (word / channels : U32)
  +column_zero_word = (phase + column_origin : U32)
  +row_zero = (first_output_row * stride + row_origin : U32)
  +write = (phase * elements * channels + (word % channels : U32) : U32)
  +target = plane_word(plane_of(number,U32.to_nat(channels)),U32.to_nat(elements),U32.to_nat(channels),0n)
  +shifted = (column_zero_word + U32.from_nat(0n) * (8 * stride : U32) : U32)
  Equal.trans(Model.Rowing,placed(pairs,Tile.filled(0.0),context,column_zero_word,write,row_zero,0),placed(pairs,Tile.filled(0.0),context,shifted,write,row_zero,0),
    placed(pairs,Tile.filled(0.0),context,shifted,target,(row_zero + U32.from_nat(0n) : U32),U32.from_nat(0n)),
    Equal.cong(U32,Model.Rowing,place => placed(pairs,Tile.filled(0.0),context,place,write,row_zero,0),column_zero_word,shifted,column_zero(column_zero_word,stride)),
    Equal.trans(Model.Rowing,placed(pairs,Tile.filled(0.0),context,shifted,write,row_zero,0),placed(pairs,Tile.filled(0.0),context,shifted,target,row_zero,0),
      placed(pairs,Tile.filled(0.0),context,shifted,target,(row_zero + U32.from_nat(0n) : U32),U32.from_nat(0n)),
      Equal.cong(U32,Model.Rowing,place => placed(pairs,Tile.filled(0.0),context,shifted,place,row_zero,0),write,target,
        start_word(number,channels,elements,cap,cap_ok,
          Order.transitive(number,Nat.mul(U32.to_nat(stride),U32.to_nat(channels)),U32.to_nat(cap),Bounds.strict_le(number,Nat.mul(U32.to_nat(stride),U32.to_nat(channels)),number_inside),
            planes_bound),divisor_bound,channels_positive)),
      Equal.cong(U32,Model.Rowing,place => placed(pairs,Tile.filled(0.0),context,shifted,target,place,0),row_zero,(row_zero + U32.from_nat(0n) : U32),
        Equal.sym(U32,(row_zero + U32.from_nat(0n) : U32),row_zero,Arithmetic.add_zero(row_zero)))))

def geometry_room(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>) ->
  Room<depth,phases_of(window),elements_of(planes),channel_count(window),cap>:
  Geometry{+room,tiles_bound,tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  room

def geometry_positive(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>) ->
  {Nat.is_lt(0n,channel_count(window)) == True{} : Bool}:
  Geometry{+room,tiles_bound,tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  Room{cap_ok,limit,fits,capped,elements_positive,+channels_positive} = room
  channels_positive

def geometry_planes(+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>) ->
  {plane_count(window) == Nat.mul(phases_of(window),channel_count(window)) : Nat}:
  Geometry{room,tiles_bound,tiles_positive,divisor_bound,planes_bound,+planes_exact} = geometry
  planes_exact

def row_room_of(+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+word: U32,+depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>) ->
  RowRoom<context_of(window,planes,input,word)>:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  Geometry{+room,+tiles_bound,+tiles_positive,divisor_bound,planes_bound,planes_exact} = geometry
  Room{+cap_ok,limit,fits,capped,elements_positive,channels_positive} = room
  RowRoom{cap,cap_ok,tiles_bound,tiles_positive}

def plane_inside(+number: Nat,+phases: Nat,+channels: Nat,+positive: {Nat.is_lt(0n,channels) == True{} : Bool},
  +inside: {Nat.is_lt(number,Nat.mul(phases,channels)) == True{} : Bool}) -> PlaneInside<plane_of(number,channels),phases,channels>:
  PlaneInside{DivisionOrder.quotient_below(number,phases,channels,positive,inside),DivisionOrder.remainder_small(number,channels,positive)}

def number_of_plane(+number: Nat,+channels: Nat,+positive: {Nat.is_lt(0n,channels) == True{} : Bool}) -> {plane_number(plane_of(number,channels),channels) == number : Nat}:
  Equal.sym(Nat,number,Nat.add(Nat.mul(Nat.div(number,channels),channels),Nat.mod(number,channels)),DivisionOrder.quotient_exact(number,channels,positive))

# Two plane numbers compare as their planes.
def planes_compared(+number: Nat,+other: Nat,+channels: Nat,+positive: {Nat.is_lt(0n,channels) == True{} : Bool}) ->
  {Nat.is_eq(plane_number(plane_of(number,channels),channels),plane_number(plane_of(other,channels),channels)) == Nat.is_eq(number,other) : Bool}:
  %Equal.sym(Nat,plane_number(plane_of(number,channels),channels),number,number_of_plane(number,channels,positive)) :
    {Nat.is_eq(_,plane_number(plane_of(other,channels),channels)) == Nat.is_eq(number,other) : Bool}
  %Equal.sym(Nat,plane_number(plane_of(other,channels),channels),other,number_of_plane(other,channels,positive)) : {Nat.is_eq(number,_) == Nat.is_eq(number,other) : Bool}
  {==}

def done_form(+plane: U32,+window: Packing.Stencil,+planes: Packing.Planes,+done: Model.Rowing) ->
  {Model.plane_done(plane,window,planes,done) == Model.Transposing{Model.rowing_input(done),rowing_pairs(done),(plane + 1 : U32),window,planes} : Model.Transposing}:
  Model.Rowing{+input,+pairs,+previous,+column,+write,+row,+in_row,+row_tiles,+first_column,+channel_base,+height,+width,+stride,+groups,+channels} = done
  {==}

# The rows plane number number writes, its lane chosen by its channel.
def plane_rows_of(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat) -> Model.Rowing:
  +word = U32.from_nat(number)
  +lane = (word % Packing.channels_of(window) % 8 : U32)
  lane_rows(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),pairs,context_of(window,planes,input,word),plane_of(number,channel_count(window)),
    elements_of(planes))

# The pair of a spot of plane number number.
def plane_pair(+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+spot: Spot) -> Tile.Tile16:
  +word = U32.from_nat(number)
  lanes_pair(context_of(window,planes,input,word),Lanes.bits((word % Packing.channels_of(window) : U32)),spot)

# A plane step writes the rows of its plane and moves to the next.
def plane_advance(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+depth: Nat,+cap: U32,
  +geometry: Geometry<window,planes,depth,cap>,+number_inside: {Nat.is_lt(number,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool}) ->
  {Model.plane_step(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}) ==
    Model.Transposing{input,rowing_pairs(plane_rows_of(input,pairs,window,planes,number)),U32.from_nat(1n+number),window,planes} : Model.Transposing}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +word = U32.from_nat(number)
  +lane = (word % Packing.channels_of(window) % 8 : U32)
  +half = U32.is_lt(lane,4)
  +low = U32.is_lt((lane % 4 : U32),2)
  +first = U32.is_zero((lane % 2 : U32))
  +start = Model.plane_start(input,pairs,word,window,planes)
  +began = ready(pairs,Tile.filled(0.0),context_of(window,planes,input,word),Spot{0n,0n},
    plane_word(plane_of(number,channel_count(window)),elements_of(planes),channel_count(window),0n))
  +done = Model.row_lane(half,low,first,U32.to_nat(elements),began)
  +target = {Model.Transposing{input,rowing_pairs(plane_rows_of(input,pairs,window,planes,number)),U32.from_nat(1n+number),window,planes} : Model.Transposing}
  %Equal.sym(Model.Rowing,start,began,start_ready(input,pairs,window,planes,number,depth,cap,geometry,number_inside)) :
    {Model.plane_done(word,window,planes,Model.row_lane(half,low,first,U32.to_nat(elements),_)) == target : Model.Transposing}
  Equal.trans(Model.Transposing,Model.plane_done(word,window,planes,done),Model.Transposing{Model.rowing_input(done),rowing_pairs(done),(word + 1 : U32),window,planes},target,
    done_form(word,window,planes,done),
    %Equal.sym(Tree.Tree<Tile.Tile8>,Model.rowing_input(done),Model.rowing_input(began),Model.row_lane_input(half,low,first,U32.to_nat(elements),began)) :
      {Model.Transposing{_,rowing_pairs(done),(word + 1 : U32),window,planes} == target : Model.Transposing}
    %Equal.sym(U32,(word + 1 : U32),U32.from_nat(1n+number),Bounds.increment_encoded(number)) :
      {Model.Transposing{Model.rowing_input(began),rowing_pairs(done),_,window,planes} == target : Model.Transposing}
    {==})

# Another plane leaves the pair of the query.
def plane_missed(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+other: Nat,+queried: Nat,
  +depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +number_inside: {Nat.is_lt(number,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},
  +other_inside: {Nat.is_lt(other,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},+queried_inside: {Nat.is_lt(queried,elements_of(planes)) == True{} : Bool},
  +apart: {Nat.is_eq(number,other) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(plane_rows_of(input,pairs,window,planes,number)),
    plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)) : Tile.Tile16}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +count = channel_count(window)
  +phases = phases_of(window)
  +positive = geometry_positive(window,planes,depth,cap,geometry)
  +word = U32.from_nat(number)
  +lane = (word % Packing.channels_of(window) % 8 : U32)
  lanes_missed(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),pairs,context_of(window,planes,input,word),plane_of(number,count),
    plane_of(other,count),queried,depth,phases,elements_of(planes),cap,geometry_room(window,planes,depth,cap,geometry),row_room_of(input,window,planes,word,depth,cap,geometry),
    balanced,plane_inside(number,phases,count,positive,number_inside),plane_inside(other,phases,count,positive,other_inside),queried_inside,
    %Equal.sym(Bool,Nat.is_eq(plane_number(plane_of(number,count),count),plane_number(plane_of(other,count),count)),Nat.is_eq(number,other),
      planes_compared(number,other,count,positive)) : {_ == False{} : Bool}
    apart)

# The plane of the query writes the pair of its spot.
def plane_hit(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+other: Nat,+spot: Spot,
  +depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +number_inside: {Nat.is_lt(number,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},
  +other_inside: {Nat.is_lt(other,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},
  +spot_inside: {Nat.is_lt(column_of(spot),tiles_of(planes)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,tiles_of(planes)),elements_of(planes)) == True{} : Bool},
  +same: {Nat.is_eq(number,other) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,rowing_pairs(plane_rows_of(input,pairs,window,planes,number)),
    plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))) ==
    plane_pair(input,window,planes,number,spot) : Tile.Tile16}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +count = channel_count(window)
  +phases = phases_of(window)
  +positive = geometry_positive(window,planes,depth,cap,geometry)
  +word = U32.from_nat(number)
  +lane = (word % Packing.channels_of(window) % 8 : U32)
  lanes_hit(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),pairs,context_of(window,planes,input,word),plane_of(number,count),
    plane_of(other,count),spot,depth,phases,elements_of(planes),cap,geometry_room(window,planes,depth,cap,geometry),row_room_of(input,window,planes,word,depth,cap,geometry),
    balanced,plane_inside(number,phases,count,positive,number_inside),plane_inside(other,phases,count,positive,other_inside),spot_inside,queried_inside,
    %Equal.sym(Bool,Nat.is_eq(plane_number(plane_of(number,count),count),plane_number(plane_of(other,count),count)),Nat.is_eq(number,other),
      planes_compared(number,other,count,positive)) : {_ == True{} : Bool}
    same)

def plane_balanced(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+depth: Nat,+cap: U32,
  +geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) ->
  Shape.Balanced(Tile.Tile16,depth,rowing_pairs(plane_rows_of(input,pairs,window,planes,number))):
  +word = U32.from_nat(number)
  +lane = (word % Packing.channels_of(window) % 8 : U32)
  lanes_balanced(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),pairs,context_of(window,planes,input,word),plane_of(number,channel_count(window)),
    elements_of(planes),depth,row_room_of(input,window,planes,word,depth,cap,geometry),balanced)

# ---- The plane loop ----

# Planes past the query leave its pair.
def planes_missed(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+other: Nat,+queried: Nat,
  +depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +other_inside: {Nat.is_lt(other,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},+queried_inside: {Nat.is_lt(queried,elements_of(planes)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(number,n),Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},+missed: {Nat.is_le(number,other) == False{} : Bool}) ->
  {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,n,Model.Transposing{input,pairs,U32.from_nat(number),window,planes})),
    plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)) ==
    Tree.read(Tile.Tile16,pairs,plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)) : Tile.Tile16}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +total = Nat.mul(phases_of(window),channel_count(window))
      +query = plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)
      +written = rowing_pairs(plane_rows_of(input,pairs,window,planes,number))
      +number_inside = Order.room_index(number,rest,total,remaining)
      %Equal.sym(Model.Transposing,Model.plane_step(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}),
        Model.Transposing{input,written,U32.from_nat(1n+number),window,planes},plane_advance(input,pairs,window,planes,number,depth,cap,geometry,number_inside)) :
        {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,_)),query) == Tree.read(Tile.Tile16,pairs,query) : Tile.Tile16}
      Equal.trans(Tile.Tile16,
        Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,Model.Transposing{input,written,U32.from_nat(1n+number),window,planes})),
          query),
        Tree.read(Tile.Tile16,written,query),Tree.read(Tile.Tile16,pairs,query),
        planes_missed(rest,written,input,window,planes,1n+number,other,queried,depth,cap,geometry,plane_balanced(input,pairs,window,planes,number,depth,cap,geometry,balanced),
          other_inside,queried_inside,Order.advance_room(number,rest,total,remaining),not_le_next(number,other,Nat.is_le(1n+number,other),{==},missed)),
        plane_missed(input,pairs,window,planes,number,other,queried,depth,cap,geometry,balanced,number_inside,other_inside,queried_inside,
          not_equal_of_not_le(number,other,Nat.is_eq(number,other),{==},missed)))

# The pair of the queried spot, written by the plane of the query.
def planes_hit(n: Nat,hit: Bool,+pairs: Tree.Tree<Tile.Tile16>,+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+other: Nat,
  +spot: Spot,+depth: Nat,+cap: U32,+geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +other_inside: {Nat.is_lt(other,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},
  +spot_inside: {Nat.is_lt(column_of(spot),tiles_of(planes)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,tiles_of(planes)),elements_of(planes)) == True{} : Bool},
  +exact: {Nat.is_eq(number,other) == hit : Bool},+reached: {Nat.is_le(number,other) == True{} : Bool},+within: {Nat.is_lt(other,Nat.add(number,n)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(number,n),Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,n,Model.Transposing{input,pairs,U32.from_nat(number),window,planes})),
    plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))) == plane_pair(input,window,planes,other,spot) :
    Tile.Tile16}:
  match n:
    case 0n:
      +before = Order.le_lt_trans(number,other,number,reached,%Addition.zero_right(number) : {Nat.is_lt(other,_) == True{} : Bool} within)
      Empty.absurd({Tree.read(Tile.Tile16,Model.transposing_pairs(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}),
          plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))) == plane_pair(input,window,planes,other,spot) :
          Tile.Tile16},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(number,number),True{},
          Equal.sym(Bool,Nat.is_eq(number,number),False{},Order.less_not_equal(number,number,before)),Order.equal_reflexive(number))))
    case 1n+ +rest:
      match hit:
        case True{}:
          +total = Nat.mul(phases_of(window),channel_count(window))
          +queried = element(spot,tiles_of(planes))
          +query = plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),queried)
          +written = rowing_pairs(plane_rows_of(input,pairs,window,planes,number))
          +number_inside = Order.room_index(number,rest,total,remaining)
          +equal = Order.equal_of_is_eq(number,other,exact)
          %Equal.sym(Model.Transposing,Model.plane_step(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}),
            Model.Transposing{input,written,U32.from_nat(1n+number),window,planes},plane_advance(input,pairs,window,planes,number,depth,cap,geometry,number_inside)) :
            {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,_)),query) == plane_pair(input,window,planes,other,spot) :
              Tile.Tile16}
          Equal.trans(Tile.Tile16,
            Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,Model.Transposing{input,written,U32.from_nat(1n+number),window,planes})),
              query),
            Tree.read(Tile.Tile16,written,query),plane_pair(input,window,planes,other,spot),
            planes_missed(rest,written,input,window,planes,1n+number,other,queried,depth,cap,geometry,plane_balanced(input,pairs,window,planes,number,depth,cap,geometry,balanced),
              other_inside,queried_inside,Order.advance_room(number,rest,total,remaining),%equal : {Nat.is_le(1n+number,_) == False{} : Bool} not_le_successor(number)),
            Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,written,query),plane_pair(input,window,planes,number,spot),plane_pair(input,window,planes,other,spot),
              plane_hit(input,pairs,window,planes,number,other,spot,depth,cap,geometry,balanced,number_inside,other_inside,spot_inside,queried_inside,exact),
              Equal.cong(Nat,Tile.Tile16,chosen => plane_pair(input,window,planes,chosen,spot),number,other,equal)))
        case False{}:
          +total = Nat.mul(phases_of(window),channel_count(window))
          +query = plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))
          +written = rowing_pairs(plane_rows_of(input,pairs,window,planes,number))
          +number_inside = Order.room_index(number,rest,total,remaining)
          %Equal.sym(Model.Transposing,Model.plane_step(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}),
            Model.Transposing{input,written,U32.from_nat(1n+number),window,planes},plane_advance(input,pairs,window,planes,number,depth,cap,geometry,number_inside)) :
            {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,_)),query) == plane_pair(input,window,planes,other,spot) :
              Tile.Tile16}
          planes_hit(rest,Nat.is_eq(1n+number,other),written,input,window,planes,1n+number,other,spot,depth,cap,geometry,
            plane_balanced(input,pairs,window,planes,number,depth,cap,geometry,balanced),other_inside,spot_inside,queried_inside,{==},
            Order.strict_follow(number,other,below_of_unequal(number,other,reached,exact)),
            %Addition.successor_right(number,rest) : {Nat.is_lt(other,_) == True{} : Bool} within,
            Order.advance_room(number,rest,total,remaining))

# The pair the transpose leaves at a spot of plane number other.
def transposed_pair(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+other: Nat,+spot: Spot,+depth: Nat,
  +cap: U32,+geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +other_inside: {Nat.is_lt(other,Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool},
  +spot_inside: {Nat.is_lt(column_of(spot),tiles_of(planes)) == True{} : Bool},+queried_inside: {Nat.is_lt(element(spot,tiles_of(planes)),elements_of(planes)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Model.transposed(window,planes,input,pairs),
    plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))) == plane_pair(input,window,planes,other,spot) :
    Tile.Tile16}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +total = Nat.mul(phases_of(window),channel_count(window))
  +query = plane_word(plane_of(other,channel_count(window)),elements_of(planes),channel_count(window),element(spot,tiles_of(planes)))
  %Equal.sym(Nat,plane_count(window),total,geometry_planes(window,planes,depth,cap,geometry)) :
    {Tree.read(Tile.Tile16,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,_,Model.Transposing{input,pairs,U32.from_nat(0n),window,planes})),query) ==
      plane_pair(input,window,planes,other,spot) : Tile.Tile16}
  planes_hit(total,Nat.is_eq(0n,other),pairs,input,window,planes,0n,other,spot,depth,cap,geometry,balanced,other_inside,spot_inside,queried_inside,{==},Order.zero_le(other),
    other_inside,Order.reflexive(total))

# The transpose keeps its pairs a tree of the same depth.
def planes_balanced(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+input: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+number: Nat,+depth: Nat,+cap: U32,
  +geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +remaining: {Nat.is_le(Nat.add(number,n),Nat.mul(phases_of(window),channel_count(window))) == True{} : Bool}) ->
  Shape.Balanced(Tile.Tile16,depth,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,n,Model.Transposing{input,pairs,U32.from_nat(number),window,planes}))):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      +total = Nat.mul(phases_of(window),channel_count(window))
      +written = rowing_pairs(plane_rows_of(input,pairs,window,planes,number))
      %Equal.sym(Model.Transposing,Model.plane_step(Model.Transposing{input,pairs,U32.from_nat(number),window,planes}),
        Model.Transposing{input,written,U32.from_nat(1n+number),window,planes},
        plane_advance(input,pairs,window,planes,number,depth,cap,geometry,Order.room_index(number,rest,total,remaining))) :
        Shape.Balanced(Tile.Tile16,depth,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,rest,_)))
      planes_balanced(rest,written,input,window,planes,1n+number,depth,cap,geometry,plane_balanced(input,pairs,window,planes,number,depth,cap,geometry,balanced),
        Order.advance_room(number,rest,total,remaining))

# The pairs after the transpose are a tree of the same depth.
def transposed_balanced(+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+cap: U32,
  +geometry: Geometry<window,planes,depth,cap>,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs)) -> Shape.Balanced(Tile.Tile16,depth,Model.transposed(window,planes,input,pairs)):
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +total = Nat.mul(phases_of(window),channel_count(window))
  %Equal.sym(Nat,plane_count(window),total,geometry_planes(window,planes,depth,cap,geometry)) :
    Shape.Balanced(Tile.Tile16,depth,Model.transposing_pairs(Loop.run(~Model.Transposing,~Model.plane_step,_,Model.Transposing{input,pairs,U32.from_nat(0n),window,planes})))
  planes_balanced(total,pairs,input,window,planes,0n,depth,cap,geometry,balanced,Order.reflexive(total))
