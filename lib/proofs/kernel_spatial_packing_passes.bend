# The passes of the pack read back pointwise: each tile of a kernel column
# holds, lane by lane, the shifted pair of the last pass that reaches it.
import Base
import ../traversal_loop.bend as Loop
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_spatial_packing_planes.bend as Planes
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./word_subtraction.bend as Subtraction
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./nat_division_order.bend as DivisionOrder

# ---- A pass over the entries of a kernel column ----

# A pass: the pair it reads first, the tile it writes first, the lane its
# tiles take from and the taps.
type Pass is Data:
  Pass{read_origin: Nat,write_origin: Nat,lane: U32,tap_rows: U32,tap_columns: U32,row_step: U32}

def taps(+at: Pass) -> Nat:
  Pass{read_origin,write_origin,lane,+tap_rows,tap_columns,row_step} = at
  U32.to_nat(tap_rows)

def columns_of(+at: Pass) -> Nat:
  Pass{read_origin,write_origin,lane,tap_rows,+tap_columns,row_step} = at
  U32.to_nat(tap_columns)

def lane_of(+at: Pass) -> U32:
  Pass{read_origin,write_origin,+lane,tap_rows,tap_columns,row_step} = at
  lane

def origin_of(+at: Pass) -> Nat:
  Pass{read_origin,+write_origin,lane,tap_rows,tap_columns,row_step} = at
  write_origin

# The tile of entry index: one tap column apart.
def slot(+at: Pass,+index: Nat) -> Nat:
  Nat.add(origin_of(at),Nat.mul(index,columns_of(at)))

# The pair of the entry at a spot: channel row_of(spot), kernel row
# column_of(spot).
def pair_number(+at: Pass,+spot: Planes.Spot) -> Nat:
  Pass{+read_origin,write_origin,lane,tap_rows,tap_columns,+row_step} = at
  Nat.add(Nat.add(read_origin,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),U32.to_nat(row_step)))

def entry_tile(+at: Pass,+spot: Planes.Spot) -> U32:
  U32.from_nat(slot(at,Planes.element(spot,taps(at))))

def entry_pair(+at: Pass,+spot: Planes.Spot) -> U32:
  U32.from_nat(pair_number(at,spot))

# The pass about to read the entry at a spot.
def entry_ready(pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+at: Pass,+spot: Planes.Spot) -> Model.Passing:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  Model.Passing{pairs,tiles,U32.from_nat(Nat.add(read_origin,Planes.row_of(spot))),
    U32.from_nat(Nat.add(Nat.add(read_origin,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),U32.to_nat(row_step)))),U32.from_nat(Planes.column_of(spot)),
    U32.from_nat(Nat.add(write_origin,Nat.mul(Planes.element(spot,U32.to_nat(tap_rows)),U32.to_nat(tap_columns)))),lane,tap_rows,tap_columns,row_step}

def passing_tiles(state: Model.Passing) -> Tree.Tree<Tile.Tile8>:
  Model.Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step} = state
  tiles

def entry_tiles(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+at: Pass,+spot: Planes.Spot) ->
  {passing_tiles(entry_ready(pairs,tiles,at,spot)) == tiles : Tree.Tree<Tile.Tile8>}:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  {==}

# The pass with the given start, read, kernel row and write words.
def placed_entry(pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+at: Pass,start: U32,read: U32,kernel_row: U32,write: U32) -> Model.Passing:
  Pass{read_origin,write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  Model.Passing{pairs,tiles,start,read,kernel_row,write,lane,tap_rows,tap_columns,row_step}

def placed_equal(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+at: Pass,+start: U32,+other_start: U32,+read: U32,+other_read: U32,+kernel_row: U32,
  +other_kernel_row: U32,+write: U32,+other_write: U32,+same_start: {start == other_start : U32},+same_read: {read == other_read : U32},
  +same_kernel_row: {kernel_row == other_kernel_row : U32},+same_write: {write == other_write : U32}) ->
  {placed_entry(pairs,tiles,at,start,read,kernel_row,write) == placed_entry(pairs,tiles,at,other_start,other_read,other_kernel_row,other_write) : Model.Passing}:
  %same_start : {placed_entry(pairs,tiles,at,start,read,kernel_row,write) == placed_entry(pairs,tiles,at,_,other_read,other_kernel_row,other_write) : Model.Passing}
  %same_read : {placed_entry(pairs,tiles,at,start,read,kernel_row,write) == placed_entry(pairs,tiles,at,start,_,other_kernel_row,other_write) : Model.Passing}
  %same_kernel_row : {placed_entry(pairs,tiles,at,start,read,kernel_row,write) == placed_entry(pairs,tiles,at,start,read,_,other_write) : Model.Passing}
  %same_write : {placed_entry(pairs,tiles,at,start,read,kernel_row,write) == placed_entry(pairs,tiles,at,start,read,kernel_row,_) : Model.Passing}
  {==}

# The kernel row after the entry at a spot reaches the taps.
def wrap_of(+at: Pass,+spot: Planes.Spot) -> Bool:
  Pass{read_origin,write_origin,lane,+tap_rows,tap_columns,row_step} = at
  U32.is_eq((U32.from_nat(Planes.column_of(spot)) + 1 : U32),tap_rows)

# The pass past the entry at a spot, its next words picked by wrap.
def moved_entry(+wrap: Bool,pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>,+at: Pass,+spot: Planes.Spot) -> Model.Passing:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  +start = U32.from_nat(Nat.add(read_origin,Planes.row_of(spot)))
  +read = U32.from_nat(Nat.add(Nat.add(read_origin,Planes.row_of(spot)),Nat.mul(Planes.column_of(spot),U32.to_nat(row_step))))
  +kernel_row = U32.from_nat(Planes.column_of(spot))
  +write = U32.from_nat(Nat.add(write_origin,Nat.mul(Planes.element(spot,U32.to_nat(tap_rows)),U32.to_nat(tap_columns))))
  Model.Passing{pairs,tiles,Bool.pick(U32,wrap,(start + 1 : U32),start),Bool.pick(U32,wrap,(start + 1 : U32),(read + row_step : U32)),
    Bool.pick(U32,wrap,0,(kernel_row + 1 : U32)),Bool.pick(U32,wrap,(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),(write + tap_columns : U32)),lane,
    tap_rows,tap_columns,row_step}

# What a pass needs: taps and tap columns at least one, the taps within a
# cap, the tiles of its entries inside a tree of depth depth.
type PassRoom<-at: Pass,-entries: Nat,-depth: Nat> is Data:
  PassRoom{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},limit: {Nat.is_le(depth,31n) == True{} : Bool},
    capped: {Nat.is_le(U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap)) == True{} : Bool},taps_bound: {Nat.is_le(taps(at),U32.to_nat(cap)) == True{} : Bool},
    taps_positive: {Nat.is_lt(0n,taps(at)) == True{} : Bool},columns_positive: {Nat.is_lt(0n,columns_of(at)) == True{} : Bool},
    fits: {Nat.is_lt(slot(at,entries),Nat.add(columns_of(at),U32.to_nat(U32.shln(1,depth)))) == True{} : Bool}}

def room_taps(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>) -> {Nat.is_lt(0n,taps(at)) == True{} : Bool}:
  PassRoom{cap,cap_ok,limit,capped,taps_bound,+taps_positive,columns_positive,fits} = room
  taps_positive

def room_columns(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>) -> {Nat.is_lt(0n,columns_of(at)) == True{} : Bool}:
  PassRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,+columns_positive,fits} = room
  columns_positive

def room_limit(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>) -> {Nat.is_le(depth,31n) == True{} : Bool}:
  PassRoom{cap,cap_ok,+limit,capped,taps_bound,taps_positive,columns_positive,fits} = room
  limit

# ---- One entry ----

# One more of a step: origin + (1 + index) * step.
def step_number(+origin: Nat,+index: Nat,+word: U32) ->
  {(U32.from_nat(Nat.add(origin,Nat.mul(index,U32.to_nat(word)))) + word : U32) == U32.from_nat(Nat.add(origin,Nat.mul(1n+index,U32.to_nat(word)))) : U32}:
  +size = U32.to_nat(word)
  +part = Nat.mul(index,size)
  %Arithmetic.native_roundtrip(word) : {(U32.from_nat(Nat.add(origin,part)) + _ : U32) == U32.from_nat(Nat.add(origin,Nat.mul(1n+index,size))) : U32}
  Equal.trans(U32,(U32.from_nat(Nat.add(origin,part)) + U32.from_nat(size) : U32),U32.from_nat(Nat.add(Nat.add(origin,part),size)),
    U32.from_nat(Nat.add(origin,Nat.mul(1n+index,size))),WordArithmetic.native_add(Nat.add(origin,part),size),
    Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(Nat.add(origin,part),size),Nat.add(origin,Nat.add(size,part)),
      Equal.trans(Nat,Nat.add(Nat.add(origin,part),size),Nat.add(origin,Nat.add(part,size)),Nat.add(origin,Nat.add(size,part)),Addition.associative(origin,part,size),
        Equal.cong(Nat,Nat,tail => Nat.add(origin,tail),Nat.add(part,size),Nat.add(size,part),Order.commutative(part,size)))))

# The next channel's first pair.
def start_bump(+origin: Nat,+row: Nat) -> {(U32.from_nat(Nat.add(origin,row)) + 1 : U32) == U32.from_nat(Nat.add(origin,1n+row)) : U32}:
  Equal.trans(U32,(U32.from_nat(Nat.add(origin,row)) + 1 : U32),U32.from_nat(1n+Nat.add(origin,row)),U32.from_nat(Nat.add(origin,1n+row)),
    Bounds.increment_encoded(Nat.add(origin,row)),
    Equal.cong(Nat,U32,value => U32.from_nat(value),1n+Nat.add(origin,row),Nat.add(origin,1n+row),
      Equal.sym(Nat,Nat.add(origin,1n+row),Nat.add(1n+origin,row),Addition.successor_right(origin,row))))

# A wrap moves the write past the channel's taps: one tap column on.
def wrap_write(+write: U32,+rows: U32,+columns: U32) -> {((write - (rows - 1) * columns : U32) + rows * columns : U32) == (write + columns : U32) : U32}:
  +full = (rows * columns : U32)
  +less = Equal.trans(U32,((rows - 1 : U32) * columns : U32),(full - 1 * columns : U32),(full - columns : U32),Arithmetic.multiply_subtractive(rows,1,columns),
    Equal.cong(U32,U32,part => (full - part : U32),(1 * columns : U32),columns,
      Equal.trans(U32,(1 * columns : U32),(columns * 1 : U32),columns,Arithmetic.multiply_commutative(1,columns),Arithmetic.multiply_one(columns))))
  %Equal.sym(U32,((rows - 1 : U32) * columns : U32),(full - columns : U32),less) : {((write - _ : U32) + full : U32) == (write + columns : U32) : U32}
  %Arithmetic.subtract_difference(write,full,columns) : {(_ + full : U32) == (write + columns : U32) : U32}
  %Equal.sym(U32,(((write - full : U32) + columns : U32) + full : U32),(((write - full : U32) + full : U32) + columns : U32),
    Arithmetic.add_swap((write - full : U32),columns,full)) : {_ == (write + columns : U32) : U32}
  %Equal.sym(U32,((write - full : U32) + full : U32),write,Subtraction.add_subtract(write,full)) : {(_ + columns : U32) == (write + columns : U32) : U32}
  {==}

# The kernel row wraps exactly when the next one falls outside the taps.
def entry_wrap(+at: Pass,+spot: Planes.Spot,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool}) -> {wrap_of(at,spot) == Bool.not(Nat.is_lt(1n+Planes.column_of(spot),taps(at))) : Bool}:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  PassRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+fits} = room
  +rows = U32.to_nat(tap_rows)
  +column = Planes.column_of(spot)
  %Equal.sym(U32,(U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)) :
    {U32.is_eq(_,tap_rows) == Bool.not(Nat.is_lt(1n+column,rows)) : Bool}
  %Arithmetic.native_roundtrip(tap_rows) : {U32.is_eq(U32.from_nat(1n+column),_) == Bool.not(Nat.is_lt(1n+column,rows)) : Bool}
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+column),U32.from_nat(rows)),Nat.is_eq(1n+column,rows),
    Planes.encoded_equal(1n+column,rows,cap,cap_ok,Order.transitive(1n+column,rows,U32.to_nat(cap),Order.strict_follow(column,rows,column_inside),taps_bound),taps_bound)) :
    {_ == Bool.not(Nat.is_lt(1n+column,rows)) : Bool}
  Planes.inside_not_equal(1n+column,rows,Nat.is_lt(1n+column,rows),{==},Order.strict_follow(column,rows,column_inside))

# Past the entry at a spot the pass is ready at the next spot.
def entry_moved(+inside: Bool,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+at: Pass,+spot: Planes.Spot,
  +exact: {Nat.is_lt(1n+Planes.column_of(spot),taps(at)) == inside : Bool},+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool}) ->
  {moved_entry(Bool.not(inside),pairs,tiles,at,spot) == entry_ready(pairs,tiles,at,Planes.next_spot(spot,taps(at))) : Model.Passing}:
  match inside:
    case True{}:
      Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
      Planes.Spot{+row,+column} = spot
      +at = {Pass{read_origin,write_origin,lane,tap_rows,tap_columns,row_step} : Pass}
      +rows = U32.to_nat(tap_rows)
      +start = U32.from_nat(Nat.add(read_origin,row))
      +read = U32.from_nat(Nat.add(Nat.add(read_origin,row),Nat.mul(column,U32.to_nat(row_step))))
      +index = Planes.outer(row,column,rows)
      +write = U32.from_nat(Nat.add(write_origin,Nat.mul(index,U32.to_nat(tap_columns))))
      +next = {Planes.Spot{row,1n+column} : Planes.Spot}
      +target = U32.from_nat(Nat.add(write_origin,Nat.mul(Planes.element(next,rows),U32.to_nat(tap_columns))))
      %Equal.sym(Planes.Spot,Planes.next_spot(Planes.Spot{row,column},rows),next,Planes.next_within(Planes.Spot{row,column},rows,exact)) :
        {moved_entry(False{},pairs,tiles,at,Planes.Spot{row,column}) == entry_ready(pairs,tiles,at,_) : Model.Passing}
      placed_equal(pairs,tiles,at,start,start,(read + row_step : U32),U32.from_nat(Nat.add(Nat.add(read_origin,row),Nat.mul(1n+column,U32.to_nat(row_step)))),
        (U32.from_nat(column) + 1 : U32),U32.from_nat(1n+column),(write + tap_columns : U32),target,{==},step_number(Nat.add(read_origin,row),column,row_step),
        Bounds.increment_encoded(column),
        Equal.trans(U32,(write + tap_columns : U32),U32.from_nat(Nat.add(write_origin,Nat.mul(1n+index,U32.to_nat(tap_columns)))),target,
          step_number(write_origin,index,tap_columns),
          Equal.cong(Nat,U32,position => U32.from_nat(Nat.add(write_origin,Nat.mul(position,U32.to_nat(tap_columns)))),1n+index,Planes.element(next,rows),
            Equal.sym(Nat,Planes.element(next,rows),1n+index,Planes.next_count(row,column,rows,True{},exact,column_inside)))))
    case False{}:
      Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
      Planes.Spot{+row,+column} = spot
      +at = {Pass{read_origin,write_origin,lane,tap_rows,tap_columns,row_step} : Pass}
      +rows = U32.to_nat(tap_rows)
      +start = U32.from_nat(Nat.add(read_origin,row))
      +index = Planes.outer(row,column,rows)
      +write = U32.from_nat(Nat.add(write_origin,Nat.mul(index,U32.to_nat(tap_columns))))
      +next = {Planes.Spot{1n+row,0n} : Planes.Spot}
      +target = U32.from_nat(Nat.add(write_origin,Nat.mul(Planes.element(next,rows),U32.to_nat(tap_columns))))
      +bumped = U32.from_nat(Nat.add(read_origin,1n+row))
      %Equal.sym(Planes.Spot,Planes.next_spot(Planes.Spot{row,column},rows),next,Planes.next_wrapped(Planes.Spot{row,column},rows,exact)) :
        {moved_entry(True{},pairs,tiles,at,Planes.Spot{row,column}) == entry_ready(pairs,tiles,at,_) : Model.Passing}
      placed_equal(pairs,tiles,at,(start + 1 : U32),bumped,(start + 1 : U32),U32.from_nat(Nat.add(Nat.add(read_origin,1n+row),Nat.mul(0n,U32.to_nat(row_step)))),0,
        U32.from_nat(0n),(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),target,start_bump(read_origin,row),
        Equal.trans(U32,(start + 1 : U32),bumped,U32.from_nat(Nat.add(Nat.add(read_origin,1n+row),Nat.mul(0n,U32.to_nat(row_step)))),start_bump(read_origin,row),
          Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(read_origin,1n+row),Nat.add(Nat.add(read_origin,1n+row),0n),
            Equal.sym(Nat,Nat.add(Nat.add(read_origin,1n+row),0n),Nat.add(read_origin,1n+row),Addition.zero_right(Nat.add(read_origin,1n+row))))),
        {==},
        Equal.trans(U32,(write - (tap_rows - 1) * tap_columns + tap_rows * tap_columns : U32),(write + tap_columns : U32),target,wrap_write(write,tap_rows,tap_columns),
          Equal.trans(U32,(write + tap_columns : U32),U32.from_nat(Nat.add(write_origin,Nat.mul(1n+index,U32.to_nat(tap_columns)))),target,
            step_number(write_origin,index,tap_columns),
            Equal.cong(Nat,U32,position => U32.from_nat(Nat.add(write_origin,Nat.mul(position,U32.to_nat(tap_columns)))),1n+index,Planes.element(next,rows),
              Equal.sym(Nat,Planes.element(next,rows),1n+index,Planes.next_count(row,column,rows,False{},exact,column_inside))))))

# The tile a first pass writes, and the tile a later pass merges.
def first_value(~shift: Tile.Tile16 -> Tile.Tile8,+lane: U32,+kept: Tile.Tile8,+pair: Tile.Tile16) -> Tile.Tile8:
  shift(pair)

def merge_value(~shift: Tile.Tile16 -> Tile.Tile8,+lane: U32,+kept: Tile.Tile8,+pair: Tile.Tile16) -> Tile.Tile8:
  Packing.merged(lane,kept,shift(pair))

# A step from the entry at a spot writes its tile and readies the next spot.
def Moves(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8) -> Type:
  @+pairs: Tree.Tree<Tile.Tile16> -> @+at: Pass -> @+entries: Nat -> @+depth: Nat -> @+room: PassRoom<at,entries,depth> -> @+tiles: Tree.Tree<Tile.Tile8> ->
    @+spot: Planes.Spot -> @+inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool} ->
    {step(entry_ready(pairs,tiles,at,spot)) ==
      entry_ready(pairs,Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))),at,
        Planes.next_spot(spot,taps(at))) : Model.Passing}

def first_moves(~shift: Tile.Tile16 -> Tile.Tile8,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,+inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool}) ->
  {Model.first_step(~shift,entry_ready(pairs,tiles,at,spot)) ==
    entry_ready(pairs,Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),
      first_value(~shift,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))),at,Planes.next_spot(spot,taps(at))) : Model.Passing}:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  +at = {Pass{read_origin,write_origin,lane,tap_rows,tap_columns,row_step} : Pass}
  +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),shift(Tree.read(Tile.Tile16,pairs,entry_pair(at,spot))))
  +ahead = Nat.is_lt(1n+Planes.column_of(spot),taps(at))
  Equal.trans(Model.Passing,moved_entry(wrap_of(at,spot),pairs,written,at,spot),moved_entry(Bool.not(ahead),pairs,written,at,spot),
    entry_ready(pairs,written,at,Planes.next_spot(spot,taps(at))),
    Equal.cong(Bool,Model.Passing,wrap => moved_entry(wrap,pairs,written,at,spot),wrap_of(at,spot),Bool.not(ahead),entry_wrap(at,spot,entries,depth,room,inside)),
    entry_moved(ahead,pairs,written,at,spot,{==},inside))

def merge_moves(~shift: Tile.Tile16 -> Tile.Tile8,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,+inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool}) ->
  {Model.merge_step(~shift,entry_ready(pairs,tiles,at,spot)) ==
    entry_ready(pairs,Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),
      merge_value(~shift,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))),at,Planes.next_spot(spot,taps(at))) :
    Model.Passing}:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  +at = {Pass{read_origin,write_origin,lane,tap_rows,tap_columns,row_step} : Pass}
  +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),
    Packing.merged(lane,Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),shift(Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))))
  +ahead = Nat.is_lt(1n+Planes.column_of(spot),taps(at))
  Equal.trans(Model.Passing,moved_entry(wrap_of(at,spot),pairs,written,at,spot),moved_entry(Bool.not(ahead),pairs,written,at,spot),
    entry_ready(pairs,written,at,Planes.next_spot(spot,taps(at))),
    Equal.cong(Bool,Model.Passing,wrap => moved_entry(wrap,pairs,written,at,spot),wrap_of(at,spot),Bool.not(ahead),entry_wrap(at,spot,entries,depth,room,inside)),
    entry_moved(ahead,pairs,written,at,spot,{==},inside))

# ---- Tiles of entries ----

def slot_less(+at: Pass,+first: Nat,+second: Nat,+positive: {Nat.is_lt(0n,columns_of(at)) == True{} : Bool},+less: {Nat.is_lt(first,second) == True{} : Bool}) ->
  {Nat.is_lt(slot(at,first),slot(at,second)) == True{} : Bool}:
  Pass{+read_origin,+write_origin,+lane,+tap_rows,+tap_columns,+row_step} = at
  +columns = U32.to_nat(tap_columns)
  +low = Nat.mul(first,columns)
  +high = Nat.mul(second,columns)
  +below = Order.lt_le(low,Nat.mul(1n+first,columns),high,Order.add_lt(0n,columns,low,positive),
    RowMajor.mul_le(1n+first,second,columns,Order.strict_follow(first,second,less)))
  %Equal.sym(Nat,Nat.add(write_origin,low),Nat.add(low,write_origin),Order.commutative(write_origin,low)) : {Nat.is_lt(_,Nat.add(write_origin,high)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(write_origin,high),Nat.add(high,write_origin),Order.commutative(write_origin,high)) : {Nat.is_lt(Nat.add(low,write_origin),_) == True{} : Bool}
  Order.add_lt(low,high,write_origin,below)

def slots_ordered(+at: Pass,+first: Nat,+second: Nat,+positive: {Nat.is_lt(0n,columns_of(at)) == True{} : Bool},+below: Bool,
  +below_exact: {Nat.is_lt(first,second) == below : Bool},+above: Bool,+above_exact: {Nat.is_lt(second,first) == above : Bool}) ->
  {Nat.is_eq(slot(at,first),slot(at,second)) == Nat.is_eq(first,second) : Bool}:
  match below:
    case True{}:
      Equal.trans(Bool,Nat.is_eq(slot(at,first),slot(at,second)),False{},Nat.is_eq(first,second),
        Order.less_not_equal(slot(at,first),slot(at,second),slot_less(at,first,second,positive,below_exact)),
        Equal.sym(Bool,Nat.is_eq(first,second),False{},Order.less_not_equal(first,second,below_exact)))
    case False{}:
      match above:
        case True{}:
          Equal.trans(Bool,Nat.is_eq(slot(at,first),slot(at,second)),Nat.is_eq(slot(at,second),slot(at,first)),Nat.is_eq(first,second),
            Order.equal_symmetric(slot(at,first),slot(at,second)),
            Equal.trans(Bool,Nat.is_eq(slot(at,second),slot(at,first)),False{},Nat.is_eq(first,second),
              Order.less_not_equal(slot(at,second),slot(at,first),slot_less(at,second,first,positive,above_exact)),
              Equal.sym(Bool,Nat.is_eq(first,second),False{},
                Equal.trans(Bool,Nat.is_eq(first,second),Nat.is_eq(second,first),False{},Order.equal_symmetric(first,second),
                  Order.less_not_equal(second,first,above_exact)))))
        case False{}:
          +same = Order.antisymmetric(first,second,Order.not_less(second,first,above_exact),Order.not_less(first,second,below_exact))
          Equal.trans(Bool,Nat.is_eq(slot(at,first),slot(at,second)),True{},Nat.is_eq(first,second),
            %same : {Nat.is_eq(slot(at,first),slot(at,_)) == True{} : Bool} Order.equal_reflexive(slot(at,first)),
            Equal.sym(Bool,Nat.is_eq(first,second),True{},%same : {Nat.is_eq(first,_) == True{} : Bool} Order.equal_reflexive(first)))

# Tiles of entries are one exactly when the entries are.
def slots_equal(+at: Pass,+first: Nat,+second: Nat,+positive: {Nat.is_lt(0n,columns_of(at)) == True{} : Bool}) ->
  {Nat.is_eq(slot(at,first),slot(at,second)) == Nat.is_eq(first,second) : Bool}:
  slots_ordered(at,first,second,positive,Nat.is_lt(first,second),{==},Nat.is_lt(second,first),{==})

# A tap column past an entry's tile is at most the tile one past the last.
def slot_reach(+at: Pass,+entries: Nat,+index: Nat,+below: {Nat.is_lt(index,entries) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(columns_of(at),slot(at,index)),slot(at,entries)) == True{} : Bool}:
  +columns = columns_of(at)
  +origin = origin_of(at)
  +part = Nat.mul(index,columns)
  +shifted = Equal.trans(Nat,Nat.add(columns,Nat.add(origin,part)),Nat.add(Nat.add(columns,origin),part),Nat.add(origin,Nat.add(columns,part)),
    Equal.sym(Nat,Nat.add(Nat.add(columns,origin),part),Nat.add(columns,Nat.add(origin,part)),Addition.associative(columns,origin,part)),
    Equal.trans(Nat,Nat.add(Nat.add(columns,origin),part),Nat.add(Nat.add(origin,columns),part),Nat.add(origin,Nat.add(columns,part)),
      Equal.cong(Nat,Nat,head => Nat.add(head,part),Nat.add(columns,origin),Nat.add(origin,columns),Order.commutative(columns,origin)),
      Addition.associative(origin,columns,part)))
  %Equal.sym(Nat,Nat.add(columns,Nat.add(origin,part)),Nat.add(origin,Nat.add(columns,part)),shifted) : {Nat.is_le(_,slot(at,entries)) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(Nat.add(origin,Nat.add(columns,part)),Nat.add(origin,Nat.mul(entries,columns))),Nat.is_le(Nat.add(columns,part),Nat.mul(entries,columns)),
    Order.le_offset(origin,Nat.add(columns,part),Nat.mul(entries,columns))) : {_ == True{} : Bool}
  RowMajor.mul_le(1n+index,entries,columns,Order.strict_follow(index,entries,below))

def slot_inside(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,+index: Nat,+below: {Nat.is_lt(index,entries) == True{} : Bool}) ->
  {Nat.is_lt(slot(at,index),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}:
  PassRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+fits} = room
  +size = U32.to_nat(U32.shln(1,depth))
  Order.cancel_offset_lt(columns_of(at),slot(at,index),size,
    Order.le_lt_trans(Nat.add(columns_of(at),slot(at,index)),slot(at,entries),Nat.add(columns_of(at),size),slot_reach(at,entries,index,below),fits))

# A number inside the tree is a word inside it.
def tile_word_inside(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+index: Nat,+below: {Nat.is_lt(index,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(index),Tree.capacity(Tile.Tile8,tiles)) == True{} : Bool}:
  PassRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+fits} = room
  +capacity = U32.to_nat(U32.shln(1,depth))
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,tiles),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tiles,balanced)) : {U32.is_lt(U32.from_nat(index),_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),U32.shln(1,depth)),Nat.is_lt(U32.to_nat(U32.from_nat(index)),capacity),Comparison.lt(U32.from_nat(index),U32.shln(1,depth))) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,
    Bounds.roundtrip(index,cap,cap_ok,Order.transitive(index,capacity,U32.to_nat(cap),Bounds.strict_le(index,capacity,below),capped))) :
    {Nat.is_lt(_,capacity) == True{} : Bool}
  below

def tiles_equal(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,+first: Nat,+second: Nat,
  +first_below: {Nat.is_lt(first,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+second_below: {Nat.is_lt(second,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(first),U32.from_nat(second)) == Nat.is_eq(first,second) : Bool}:
  PassRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+fits} = room
  +capacity = U32.to_nat(U32.shln(1,depth))
  Planes.encoded_equal(first,second,cap,cap_ok,Order.transitive(first,capacity,U32.to_nat(cap),Bounds.strict_le(first,capacity,first_below),capped),
    Order.transitive(second,capacity,U32.to_nat(cap),Bounds.strict_le(second,capacity,second_below),capped))

# Whether a number is the tile of one of count entries from start on, the
# tiles origin + index * columns.
def any_from(count: Nat,+origin: Nat,+columns: Nat,+start: Nat,+query: Nat) -> Bool:
  match count:
    case 0n: False{}
    case 1n+ +rest: Nat.is_eq(Nat.add(origin,Nat.mul(start,columns)),query) || any_from(rest,origin,columns,1n+start,query)

# A number no entry's tile is differs from each of them.
def slot_apart(count: Nat,same: Bool,+origin: Nat,+columns: Nat,+start: Nat,+query: Nat,+index: Nat,
  +outside: {any_from(count,origin,columns,start,query) == False{} : Bool},
  +exact: {Nat.is_eq(start,index) == same : Bool},+reached: {Nat.is_le(start,index) == True{} : Bool},+within: {Nat.is_lt(index,Nat.add(start,count)) == True{} : Bool}) ->
  {Nat.is_eq(Nat.add(origin,Nat.mul(index,columns)),query) == False{} : Bool}:
  match count:
    case 0n:
      +before = Order.le_lt_trans(start,index,start,reached,%Addition.zero_right(start) : {Nat.is_lt(index,_) == True{} : Bool} within)
      Empty.absurd({Nat.is_eq(Nat.add(origin,Nat.mul(index,columns)),query) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(start,start),True{},Equal.sym(Bool,Nat.is_eq(start,start),False{},Order.less_not_equal(start,start,before)),
          Order.equal_reflexive(start))))
    case 1n+ +rest:
      match same:
        case True{}:
          %Order.equal_of_is_eq(start,index,exact) : {Nat.is_eq(Nat.add(origin,Nat.mul(_,columns)),query) == False{} : Bool}
          Boolean.or_false_left(Nat.is_eq(Nat.add(origin,Nat.mul(start,columns)),query),any_from(rest,origin,columns,1n+start,query),outside)
        case False{}:
          slot_apart(rest,Nat.is_eq(1n+start,index),origin,columns,1n+start,query,index,
            Boolean.or_false_right(Nat.is_eq(Nat.add(origin,Nat.mul(start,columns)),query),any_from(rest,origin,columns,1n+start,query),outside),
            {==},Order.strict_follow(start,index,Planes.below_of_unequal(start,index,reached,exact)),
            %Addition.successor_right(start,rest) : {Nat.is_lt(index,_) == True{} : Bool} within)

# One write of a tile, read back at a number inside the tree.
def tile_written(+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,+tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +index: Nat,+query: Nat,+value: Tile.Tile8,+index_below: {Nat.is_lt(index,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +query_below: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tiles,U32.from_nat(index),value),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile8,Nat.is_eq(index,query),value,Tree.read(Tile.Tile8,tiles,U32.from_nat(query))) : Tile.Tile8}:
  +old = Tree.read(Tile.Tile8,tiles,U32.from_nat(query))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Tree.write(Tile.Tile8,tiles,U32.from_nat(index),value),U32.from_nat(query)),
    Bool.pick(Tile.Tile8,U32.is_eq(U32.from_nat(index),U32.from_nat(query)),value,old),Bool.pick(Tile.Tile8,Nat.is_eq(index,query),value,old),
    Readback.one_full(~Tile.Tile8,depth,tiles,U32.from_nat(index),U32.from_nat(query),value,balanced,room_limit(at,entries,depth,room),
      tile_word_inside(at,entries,depth,room,tiles,balanced,index,index_below),tile_word_inside(at,entries,depth,room,tiles,balanced,query,query_below)),
    Equal.cong(Bool,Tile.Tile8,chosen => Bool.pick(Tile.Tile8,chosen,value,old),U32.is_eq(U32.from_nat(index),U32.from_nat(query)),Nat.is_eq(index,query),
      tiles_equal(at,entries,depth,room,index,query,index_below,query_below)))

# ---- The entries of a pass ----

# Entries past a tile leave it.
def walk_after(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8,
  ~moves: Moves(~step,~value),+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,n: Nat,+tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,+passed: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(Planes.element(spot,taps(at)),n),entries) == True{} : Bool},
  +behind: {Nat.is_le(Planes.element(spot,taps(at)),passed) == False{} : Bool},+passed_inside: {Nat.is_lt(passed,entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,n,entry_ready(pairs,tiles,at,spot))),U32.from_nat(slot(at,passed))) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(slot(at,passed))) : Tile.Tile8}:
  match n:
    case 0n:
      Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,U32.from_nat(slot(at,passed))),passing_tiles(entry_ready(pairs,tiles,at,spot)),tiles,
        entry_tiles(pairs,tiles,at,spot))
    case 1n+ +rest:
      +rows = taps(at)
      +here = Planes.element(spot,rows)
      +query = U32.from_nat(slot(at,passed))
      +written_value = value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))
      +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),written_value)
      +next = Planes.next_spot(spot,rows)
      +advanced = Planes.next_element(spot,rows,column_inside)
      %Equal.sym(Model.Passing,step(entry_ready(pairs,tiles,at,spot)),entry_ready(pairs,written,at,next),moves(pairs,at,entries,depth,room,tiles,spot,column_inside)) :
        {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,_)),query) == Tree.read(Tile.Tile8,tiles,query) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,entry_ready(pairs,written,at,next))),query),
        Tree.read(Tile.Tile8,written,query),Tree.read(Tile.Tile8,tiles,query),
        walk_after(~step,~value,~moves,pairs,at,entries,depth,room,rest,written,next,passed,Shape.write(Tile.Tile8,depth,tiles,entry_tile(at,spot),written_value,balanced),
          Planes.next_column_inside(spot,rows,room_taps(at,entries,depth,room)),
          %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),entries) == True{} : Bool} Order.advance_room(here,rest,entries,remaining),
          %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(_,passed) == False{} : Bool}
            Planes.not_le_next(here,passed,Nat.is_le(1n+here,passed),{==},behind),
          passed_inside),
        Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,written,query),Bool.pick(Tile.Tile8,Nat.is_eq(slot(at,here),slot(at,passed)),written_value,Tree.read(Tile.Tile8,tiles,query)),
          Tree.read(Tile.Tile8,tiles,query),
          tile_written(at,entries,depth,room,tiles,balanced,slot(at,here),slot(at,passed),written_value,
            slot_inside(at,entries,depth,room,here,Order.room_index(here,rest,entries,remaining)),slot_inside(at,entries,depth,room,passed,passed_inside)),
          Equal.cong(Bool,Tile.Tile8,chosen => Bool.pick(Tile.Tile8,chosen,written_value,Tree.read(Tile.Tile8,tiles,query)),Nat.is_eq(slot(at,here),slot(at,passed)),False{},
            Equal.trans(Bool,Nat.is_eq(slot(at,here),slot(at,passed)),Nat.is_eq(here,passed),False{},slots_equal(at,here,passed,room_columns(at,entries,depth,room)),
              Planes.not_equal_of_not_le(here,passed,Nat.is_eq(here,passed),{==},behind)))))

# Entries whose tiles all differ from a number leave it.
def walk_missed(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8,
  ~moves: Moves(~step,~value),+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,n: Nat,+tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,+query: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(Planes.element(spot,taps(at)),n),entries) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +outside: {any_from(entries,origin_of(at),columns_of(at),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,n,entry_ready(pairs,tiles,at,spot))),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match n:
    case 0n:
      Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,U32.from_nat(query)),passing_tiles(entry_ready(pairs,tiles,at,spot)),tiles,
        entry_tiles(pairs,tiles,at,spot))
    case 1n+ +rest:
      +rows = taps(at)
      +here = Planes.element(spot,rows)
      +word = U32.from_nat(query)
      +written_value = value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))
      +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),written_value)
      +next = Planes.next_spot(spot,rows)
      +advanced = Planes.next_element(spot,rows,column_inside)
      +here_inside = Order.room_index(here,rest,entries,remaining)
      %Equal.sym(Model.Passing,step(entry_ready(pairs,tiles,at,spot)),entry_ready(pairs,written,at,next),moves(pairs,at,entries,depth,room,tiles,spot,column_inside)) :
        {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,_)),word) == Tree.read(Tile.Tile8,tiles,word) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,entry_ready(pairs,written,at,next))),word),
        Tree.read(Tile.Tile8,written,word),Tree.read(Tile.Tile8,tiles,word),
        walk_missed(~step,~value,~moves,pairs,at,entries,depth,room,rest,written,next,query,Shape.write(Tile.Tile8,depth,tiles,entry_tile(at,spot),written_value,balanced),
          Planes.next_column_inside(spot,rows,room_taps(at,entries,depth,room)),
          %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),entries) == True{} : Bool} Order.advance_room(here,rest,entries,remaining),
          query_inside,outside),
        Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,written,word),Bool.pick(Tile.Tile8,Nat.is_eq(slot(at,here),query),written_value,Tree.read(Tile.Tile8,tiles,word)),
          Tree.read(Tile.Tile8,tiles,word),
          tile_written(at,entries,depth,room,tiles,balanced,slot(at,here),query,written_value,slot_inside(at,entries,depth,room,here,here_inside),query_inside),
          Equal.cong(Bool,Tile.Tile8,chosen => Bool.pick(Tile.Tile8,chosen,written_value,Tree.read(Tile.Tile8,tiles,word)),Nat.is_eq(slot(at,here),query),False{},
            slot_apart(entries,Nat.is_eq(0n,here),origin_of(at),columns_of(at),0n,query,here,outside,{==},Order.zero_le(here),here_inside))))

# The entry at a spot writes its tile from the tile it finds, and the later
# entries leave it.
def walk_here(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8,
  ~moves: Moves(~step,~value),+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,+rest: Nat,+tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(Planes.element(spot,taps(at)),1n+rest),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,1n+rest,entry_ready(pairs,tiles,at,spot))),entry_tile(at,spot)) ==
    value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot))) : Tile.Tile8}:
  +rows = taps(at)
  +here = Planes.element(spot,rows)
  +query = entry_tile(at,spot)
  +written_value = value(lane_of(at),Tree.read(Tile.Tile8,tiles,query),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))
  +written = Tree.write(Tile.Tile8,tiles,query,written_value)
  +next = Planes.next_spot(spot,rows)
  +advanced = Planes.next_element(spot,rows,column_inside)
  +here_inside = Order.room_index(here,rest,entries,remaining)
  +slot_below = slot_inside(at,entries,depth,room,here,here_inside)
  %Equal.sym(Model.Passing,step(entry_ready(pairs,tiles,at,spot)),entry_ready(pairs,written,at,next),moves(pairs,at,entries,depth,room,tiles,spot,column_inside)) :
    {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,_)),query) == written_value : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,entry_ready(pairs,written,at,next))),query),
    Tree.read(Tile.Tile8,written,query),written_value,
    walk_after(~step,~value,~moves,pairs,at,entries,depth,room,rest,written,next,here,Shape.write(Tile.Tile8,depth,tiles,query,written_value,balanced),
      Planes.next_column_inside(spot,rows,room_taps(at,entries,depth,room)),
      %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),entries) == True{} : Bool} Order.advance_room(here,rest,entries,remaining),
      %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(_,here) == False{} : Bool} Planes.not_le_successor(here),here_inside),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,written,query),Bool.pick(Tile.Tile8,Nat.is_eq(slot(at,here),slot(at,here)),written_value,Tree.read(Tile.Tile8,tiles,query)),
      written_value,tile_written(at,entries,depth,room,tiles,balanced,slot(at,here),slot(at,here),written_value,slot_below,slot_below),
      Equal.cong(Bool,Tile.Tile8,chosen => Bool.pick(Tile.Tile8,chosen,written_value,Tree.read(Tile.Tile8,tiles,query)),Nat.is_eq(slot(at,here),slot(at,here)),True{},
        Order.equal_reflexive(slot(at,here)))))

# The entry of a target spot writes its tile from the tile it finds.
def walk_hit(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8,
  ~moves: Moves(~step,~value),+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,n: Nat,hit: Bool,+tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,
  +target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool},
  +target_inside: {Nat.is_lt(Planes.column_of(target),taps(at)) == True{} : Bool},
  +exact: {Nat.is_eq(Planes.element(spot,taps(at)),Planes.element(target,taps(at))) == hit : Bool},
  +reached: {Nat.is_le(Planes.element(spot,taps(at)),Planes.element(target,taps(at))) == True{} : Bool},
  +within: {Nat.is_lt(Planes.element(target,taps(at)),Nat.add(Planes.element(spot,taps(at)),n)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(Planes.element(spot,taps(at)),n),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,n,entry_ready(pairs,tiles,at,spot))),entry_tile(at,target)) ==
    value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),Tree.read(Tile.Tile16,pairs,entry_pair(at,target))) : Tile.Tile8}:
  match n:
    case 0n:
      +here = Planes.element(spot,taps(at))
      +before = Order.le_lt_trans(here,Planes.element(target,taps(at)),here,reached,
        %Addition.zero_right(here) : {Nat.is_lt(Planes.element(target,taps(at)),_) == True{} : Bool} within)
      Empty.absurd({Tree.read(Tile.Tile8,passing_tiles(entry_ready(pairs,tiles,at,spot)),entry_tile(at,target)) ==
          value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),Tree.read(Tile.Tile16,pairs,entry_pair(at,target))) : Tile.Tile8},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(here,here),True{},Equal.sym(Bool,Nat.is_eq(here,here),False{},Order.less_not_equal(here,here,before)),
          Order.equal_reflexive(here))))
    case 1n+ +rest:
      match hit:
        case True{}:
          +rows = taps(at)
          +same = Planes.spots_equal(spot,target,rows,room_taps(at,entries,depth,room),column_inside,target_inside,
            Order.equal_of_is_eq(Planes.element(spot,rows),Planes.element(target,rows),exact))
          %same : {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,1n+rest,entry_ready(pairs,tiles,at,spot))),entry_tile(at,_)) ==
            value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,_)),Tree.read(Tile.Tile16,pairs,entry_pair(at,_))) : Tile.Tile8}
          walk_here(~step,~value,~moves,pairs,at,entries,depth,room,rest,tiles,spot,balanced,column_inside,remaining)
        case False{}:
          +rows = taps(at)
          +here = Planes.element(spot,rows)
          +aim = Planes.element(target,rows)
          +query = entry_tile(at,target)
          +pair = Tree.read(Tile.Tile16,pairs,entry_pair(at,target))
          +written_value = value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))
          +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),written_value)
          +next = Planes.next_spot(spot,rows)
          +advanced = Planes.next_element(spot,rows,column_inside)
          +aim_inside = Order.lt_le(aim,Nat.add(here,1n+rest),entries,within,remaining)
          %Equal.sym(Model.Passing,step(entry_ready(pairs,tiles,at,spot)),entry_ready(pairs,written,at,next),moves(pairs,at,entries,depth,room,tiles,spot,column_inside)) :
            {Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,_)),query) == value(lane_of(at),Tree.read(Tile.Tile8,tiles,query),pair) : Tile.Tile8}
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,passing_tiles(Loop.run(~Model.Passing,~step,rest,entry_ready(pairs,written,at,next))),query),
            value(lane_of(at),Tree.read(Tile.Tile8,written,query),pair),value(lane_of(at),Tree.read(Tile.Tile8,tiles,query),pair),
            walk_hit(~step,~value,~moves,pairs,at,entries,depth,room,rest,Nat.is_eq(Planes.element(next,rows),aim),written,next,target,
              Shape.write(Tile.Tile8,depth,tiles,entry_tile(at,spot),written_value,balanced),Planes.next_column_inside(spot,rows,room_taps(at,entries,depth,room)),target_inside,
              {==},
              %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(_,aim) == True{} : Bool}
                Order.strict_follow(here,aim,Planes.below_of_unequal(here,aim,reached,exact)),
              %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_lt(aim,Nat.add(_,rest)) == True{} : Bool}
                %Addition.successor_right(here,rest) : {Nat.is_lt(aim,_) == True{} : Bool} within,
              %Equal.sym(Nat,Planes.element(next,rows),1n+here,advanced) : {Nat.is_le(Nat.add(_,rest),entries) == True{} : Bool} Order.advance_room(here,rest,entries,remaining)),
            Equal.cong(Tile.Tile8,Tile.Tile8,kept => value(lane_of(at),kept,pair),Tree.read(Tile.Tile8,written,query),Tree.read(Tile.Tile8,tiles,query),
              Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,written,query),Bool.pick(Tile.Tile8,Nat.is_eq(slot(at,here),slot(at,aim)),written_value,Tree.read(Tile.Tile8,tiles,query)),
                Tree.read(Tile.Tile8,tiles,query),
                tile_written(at,entries,depth,room,tiles,balanced,slot(at,here),slot(at,aim),written_value,
                  slot_inside(at,entries,depth,room,here,Order.room_index(here,rest,entries,remaining)),slot_inside(at,entries,depth,room,aim,aim_inside)),
                Equal.cong(Bool,Tile.Tile8,chosen => Bool.pick(Tile.Tile8,chosen,written_value,Tree.read(Tile.Tile8,tiles,query)),Nat.is_eq(slot(at,here),slot(at,aim)),False{},
                  Equal.trans(Bool,Nat.is_eq(slot(at,here),slot(at,aim)),Nat.is_eq(here,aim),False{},slots_equal(at,here,aim,room_columns(at,entries,depth,room)),exact)))))

# The entries keep the tiles a tree of the same depth.
def walk_balanced(~step: Model.Passing -> Model.Passing,~value: @+lane: U32 -> @+kept: Tile.Tile8 -> @+pair: Tile.Tile16 -> Tile.Tile8,
  ~moves: Moves(~step,~value),+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,n: Nat,+tiles: Tree.Tree<Tile.Tile8>,+spot: Planes.Spot,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+column_inside: {Nat.is_lt(Planes.column_of(spot),taps(at)) == True{} : Bool}) ->
  Shape.Balanced(Tile.Tile8,depth,passing_tiles(Loop.run(~Model.Passing,~step,n,entry_ready(pairs,tiles,at,spot)))):
  match n:
    case 0n:
      %Equal.sym(Tree.Tree<Tile.Tile8>,passing_tiles(entry_ready(pairs,tiles,at,spot)),tiles,entry_tiles(pairs,tiles,at,spot)) : Shape.Balanced(Tile.Tile8,depth,_)
      balanced
    case 1n+ +rest:
      +rows = taps(at)
      +written_value = value(lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,spot)),Tree.read(Tile.Tile16,pairs,entry_pair(at,spot)))
      +written = Tree.write(Tile.Tile8,tiles,entry_tile(at,spot),written_value)
      %Equal.sym(Model.Passing,step(entry_ready(pairs,tiles,at,spot)),entry_ready(pairs,written,at,Planes.next_spot(spot,rows)),moves(pairs,at,entries,depth,room,tiles,spot,column_inside)) :
        Shape.Balanced(Tile.Tile8,depth,passing_tiles(Loop.run(~Model.Passing,~step,rest,_)))
      walk_balanced(~step,~value,~moves,pairs,at,entries,depth,room,rest,written,Planes.next_spot(spot,rows),
        Shape.write(Tile.Tile8,depth,tiles,entry_tile(at,spot),written_value,balanced),Planes.next_column_inside(spot,rows,room_taps(at,entries,depth,room)))

# ---- One pass ----

# The tile an entry writes: its shifted pair, merged from the pass's lane on
# after the first pass.
def entry_value(+first: Bool,+lane: U32,kept: Tile.Tile8,shifted: Tile.Tile8) -> Tile.Tile8:
  match first:
    case True{}: shifted
    case False{}: Packing.merged(lane,kept,shifted)

def run_hit(~shift: Tile.Tile16 -> Tile.Tile8,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +target_inside: {Nat.is_lt(Planes.column_of(target),taps(at)) == True{} : Bool},+target_below: {Nat.is_lt(Planes.element(target,taps(at)),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_run(~shift,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),entry_tile(at,target)) ==
    entry_value(first,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),shift(Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))) : Tile.Tile8}:
  match first:
    case True{}:
      walk_hit(~Model.first_step(~shift),~first_value(~shift),~first_moves(~shift),pairs,at,entries,depth,room,entries,Nat.is_eq(0n,Planes.element(target,taps(at))),tiles,
        Planes.Spot{0n,0n},target,balanced,room_taps(at,entries,depth,room),target_inside,{==},Order.zero_le(Planes.element(target,taps(at))),target_below,
        Order.reflexive(entries))
    case False{}:
      walk_hit(~Model.merge_step(~shift),~merge_value(~shift),~merge_moves(~shift),pairs,at,entries,depth,room,entries,Nat.is_eq(0n,Planes.element(target,taps(at))),tiles,
        Planes.Spot{0n,0n},target,balanced,room_taps(at,entries,depth,room),target_inside,{==},Order.zero_le(Planes.element(target,taps(at))),target_below,
        Order.reflexive(entries))

def run_missed(~shift: Tile.Tile16 -> Tile.Tile8,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+outside: {any_from(entries,origin_of(at),columns_of(at),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_run(~shift,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match first:
    case True{}:
      walk_missed(~Model.first_step(~shift),~first_value(~shift),~first_moves(~shift),pairs,at,entries,depth,room,entries,tiles,Planes.Spot{0n,0n},query,balanced,
        room_taps(at,entries,depth,room),Order.reflexive(entries),query_inside,outside)
    case False{}:
      walk_missed(~Model.merge_step(~shift),~merge_value(~shift),~merge_moves(~shift),pairs,at,entries,depth,room,entries,tiles,Planes.Spot{0n,0n},query,balanced,
        room_taps(at,entries,depth,room),Order.reflexive(entries),query_inside,outside)

def run_balanced(~shift: Tile.Tile16 -> Tile.Tile8,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,passing_tiles(Model.pass_run(~shift,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n})))):
  match first:
    case True{}:
      walk_balanced(~Model.first_step(~shift),~first_value(~shift),~first_moves(~shift),pairs,at,entries,depth,room,entries,tiles,Planes.Spot{0n,0n},balanced,
        room_taps(at,entries,depth,room))
    case False{}:
      walk_balanced(~Model.merge_step(~shift),~merge_value(~shift),~merge_moves(~shift),pairs,at,entries,depth,room,entries,tiles,Planes.Spot{0n,0n},balanced,
        room_taps(at,entries,depth,room))

# The tile of a pair from a lane on, the lane named by its bits.
def shifted_at(half: Bool,low: Bool,odd: Bool,pair: Tile.Tile16) -> Tile.Tile8:
  match half low odd:
    case True{} True{} False{}: Packing.shifted0(pair)
    case True{} True{} True{}: Packing.shifted1(pair)
    case True{} False{} False{}: Packing.shifted2(pair)
    case True{} False{} True{}: Packing.shifted3(pair)
    case False{} True{} False{}: Packing.shifted4(pair)
    case False{} True{} True{}: Packing.shifted5(pair)
    case False{} False{} False{}: Packing.shifted6(pair)
    case False{} False{} True{}: Packing.shifted7(pair)

def low_hit(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +target_inside: {Nat.is_lt(Planes.column_of(target),taps(at)) == True{} : Bool},+target_below: {Nat.is_lt(Planes.element(target,taps(at)),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_low(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),entry_tile(at,target)) ==
    entry_value(first,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),shifted_at(True{},low,odd,Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))) : Tile.Tile8}:
  match low odd:
    case True{} False{}: run_hit(~Packing.shifted0,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case True{} True{}: run_hit(~Packing.shifted1,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case False{} False{}: run_hit(~Packing.shifted2,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case False{} True{}: run_hit(~Packing.shifted3,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)

def high_hit(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +target_inside: {Nat.is_lt(Planes.column_of(target),taps(at)) == True{} : Bool},+target_below: {Nat.is_lt(Planes.element(target,taps(at)),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_high(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),entry_tile(at,target)) ==
    entry_value(first,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),shifted_at(False{},low,odd,Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))) : Tile.Tile8}:
  match low odd:
    case True{} False{}: run_hit(~Packing.shifted4,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case True{} True{}: run_hit(~Packing.shifted5,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case False{} False{}: run_hit(~Packing.shifted6,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case False{} True{}: run_hit(~Packing.shifted7,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)

# A pass writes the tile of a target entry from the tile it finds.
def shift_hit(+half: Bool,+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +target_inside: {Nat.is_lt(Planes.column_of(target),taps(at)) == True{} : Bool},+target_below: {Nat.is_lt(Planes.element(target,taps(at)),entries) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_shift(half,low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),entry_tile(at,target)) ==
    entry_value(first,lane_of(at),Tree.read(Tile.Tile8,tiles,entry_tile(at,target)),shifted_at(half,low,odd,Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))) : Tile.Tile8}:
  match half:
    case True{}: low_hit(low,odd,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)
    case False{}: high_hit(low,odd,first,pairs,at,entries,depth,room,tiles,target,balanced,target_inside,target_below)

def low_missed(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+outside: {any_from(entries,origin_of(at),columns_of(at),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_low(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match low odd:
    case True{} False{}: run_missed(~Packing.shifted0,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case True{} True{}: run_missed(~Packing.shifted1,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case False{} False{}: run_missed(~Packing.shifted2,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case False{} True{}: run_missed(~Packing.shifted3,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)

def high_missed(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+outside: {any_from(entries,origin_of(at),columns_of(at),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_high(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match low odd:
    case True{} False{}: run_missed(~Packing.shifted4,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case True{} True{}: run_missed(~Packing.shifted5,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case False{} False{}: run_missed(~Packing.shifted6,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case False{} True{}: run_missed(~Packing.shifted7,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)

# A pass leaves a tile no entry of it writes.
def shift_missed(+half: Bool,+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+outside: {any_from(entries,origin_of(at),columns_of(at),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passing_tiles(Model.pass_shift(half,low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n}))),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match half:
    case True{}: low_missed(low,odd,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)
    case False{}: high_missed(low,odd,first,pairs,at,entries,depth,room,tiles,query,balanced,query_inside,outside)

def low_balanced(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,passing_tiles(Model.pass_low(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n})))):
  match low odd:
    case True{} False{}: run_balanced(~Packing.shifted0,first,pairs,at,entries,depth,room,tiles,balanced)
    case True{} True{}: run_balanced(~Packing.shifted1,first,pairs,at,entries,depth,room,tiles,balanced)
    case False{} False{}: run_balanced(~Packing.shifted2,first,pairs,at,entries,depth,room,tiles,balanced)
    case False{} True{}: run_balanced(~Packing.shifted3,first,pairs,at,entries,depth,room,tiles,balanced)

def high_balanced(+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,passing_tiles(Model.pass_high(low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n})))):
  match low odd:
    case True{} False{}: run_balanced(~Packing.shifted4,first,pairs,at,entries,depth,room,tiles,balanced)
    case True{} True{}: run_balanced(~Packing.shifted5,first,pairs,at,entries,depth,room,tiles,balanced)
    case False{} False{}: run_balanced(~Packing.shifted6,first,pairs,at,entries,depth,room,tiles,balanced)
    case False{} True{}: run_balanced(~Packing.shifted7,first,pairs,at,entries,depth,room,tiles,balanced)

def shift_balanced(+half: Bool,+low: Bool,+odd: Bool,+first: Bool,+pairs: Tree.Tree<Tile.Tile16>,+at: Pass,+entries: Nat,+depth: Nat,+room: PassRoom<at,entries,depth>,
  +tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,passing_tiles(Model.pass_shift(half,low,odd,first,entries,entry_ready(pairs,tiles,at,Planes.Spot{0n,0n})))):
  match half:
    case True{}: low_balanced(low,odd,first,pairs,at,entries,depth,room,tiles,balanced)
    case False{}: high_balanced(low,odd,first,pairs,at,entries,depth,room,tiles,balanced)

# ---- The passes of a kernel column ----

def tap_rows_of(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,+tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat(tap_rows)

def tap_columns_of(+window: Packing.Stencil) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,+tap_columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  U32.to_nat(tap_columns)

# Where the tiles of a kernel column of a unit start.
def column_origin_of(+window: Packing.Stencil,+kernel_column: U32,+unit: U32) -> Nat:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  U32.to_nat((unit * inner + kernel_column : U32))

# A pass of a kernel column of a unit, its lane and plane offset given.
def pass_at(+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes) -> Pass:
  Packing.Stencil{base_cell,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,output_width,+inner,groups} = window
  Packing.Planes{first_output_row,rows,+row_width,+elements} = planes
  +channels = Packing.channels_of(window)
  Pass{U32.to_nat((((kernel_column % stride) * elements + base / 8) * channels : U32)),U32.to_nat((unit * inner + kernel_column : U32)),lane,tap_rows,tap_columns,
    (row_width / 8 * channels : U32)}

# The tile of a target entry of a kernel column of a unit.
def column_slot(+window: Packing.Stencil,+kernel_column: U32,+unit: U32,+target: Planes.Spot) -> U32:
  U32.from_nat(Nat.add(column_origin_of(window,kernel_column,unit),Nat.mul(Planes.element(target,tap_rows_of(window)),tap_columns_of(window))))

def origin_word(+word: U32) -> {word == U32.from_nat(Nat.add(U32.to_nat(word),0n)) : U32}:
  Equal.sym(U32,U32.from_nat(Nat.add(U32.to_nat(word),0n)),word,
    Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(word),0n)),U32.from_nat(U32.to_nat(word)),word,
      Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.add(U32.to_nat(word),0n),U32.to_nat(word),Addition.zero_right(U32.to_nat(word))),
      Arithmetic.native_roundtrip(word)))

# A pass starts ready at its first entry.
def start_entry(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes) ->
  {Model.pass_start(pairs,tiles,lane,base,kernel_column,unit,window,planes) == entry_ready(pairs,tiles,pass_at(lane,base,kernel_column,unit,window,planes),Planes.Spot{0n,0n}) :
    Model.Passing}:
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +read = ((((kernel_column % stride) * elements + base / 8) * Packing.channels_of(window)) : U32)
  +write = (unit * inner + kernel_column : U32)
  +origin = Nat.add(U32.to_nat(read),0n)
  placed_equal(pairs,tiles,pass_at(lane,base,kernel_column,unit,window,planes),read,U32.from_nat(origin),read,U32.from_nat(Nat.add(origin,0n)),0,U32.from_nat(0n),write,
    U32.from_nat(Nat.add(U32.to_nat(write),0n)),origin_word(read),
    Equal.trans(U32,read,U32.from_nat(origin),U32.from_nat(Nat.add(origin,0n)),origin_word(read),
      Equal.cong(Nat,U32,value => U32.from_nat(value),origin,Nat.add(origin,0n),Equal.sym(Nat,Nat.add(origin,0n),origin,Addition.zero_right(origin)))),
    {==},origin_word(write))

def passes_tiles(state: Model.Passes) -> Tree.Tree<Tile.Tile8>:
  Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes} = state
  tiles

def pass_done_form(+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,
  +done: Model.Passing) ->
  {Model.pass_done(pass,lane,base,second_lane,kernel_column,unit,window,planes,done) ==
    Model.Passes{Model.passing_pairs(done),passing_tiles(done),pass,lane,base,second_lane,kernel_column,unit,window,planes} : Model.Passes}:
  Model.Passing{+pairs,+tiles,+start,+read,+kernel_row,+write,+passing_lane,+tap_rows,+tap_columns,+row_step} = done
  {==}

# The lane and plane offset of the next pass.
def next_lane(+pass: U32,+lane: U32,+second_lane: U32,+window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32))

def next_base(+base: U32,+window: Packing.Stencil,+planes: Packing.Planes) -> U32:
  Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  Packing.Planes{first_output_row,rows,+row_width,elements} = planes
  (base + stride * row_width - output_width : U32)

# The tiles after one pass.
def pass_tiles(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+pass: U32,+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes) -> Tree.Tree<Tile.Tile8>:
  +shift = (base % 8 : U32)
  passing_tiles(Model.pass_shift(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),Model.pass_entries(window),
    entry_ready(pairs,tiles,pass_at(lane,base,kernel_column,unit,window,planes),Planes.Spot{0n,0n})))

# A pass step runs one pass and moves to the next.
def pass_advance(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,
  +window: Packing.Stencil,+planes: Packing.Planes) ->
  {Model.pass_step(Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}) ==
    Model.Passes{pairs,pass_tiles(pairs,tiles,pass,lane,base,kernel_column,unit,window,planes),(pass + 1 : U32),next_lane(pass,lane,second_lane,window),
      next_base(base,window,planes),second_lane,kernel_column,unit,window,planes} : Model.Passes}:
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  +window = {Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
  +shift = (base % 8 : U32)
  +half = U32.is_lt(shift,4)
  +low = U32.is_lt((shift % 4 : U32),2)
  +odd = U32.is_eq((shift % 2 : U32),1)
  +start = Model.pass_start(pairs,tiles,lane,base,kernel_column,unit,window,planes)
  +began = entry_ready(pairs,tiles,pass_at(lane,base,kernel_column,unit,window,planes),Planes.Spot{0n,0n})
  +done = Model.pass_shift(half,low,odd,U32.is_zero(pass),Model.pass_entries(window),began)
  +lane_next = Bool.pick(U32,U32.is_zero(pass),second_lane,(lane + output_width : U32))
  +base_next = (base + stride * row_width - output_width : U32)
  +target = {Model.Passes{pairs,passing_tiles(done),(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes} : Model.Passes}
  %Equal.sym(Model.Passing,start,began,start_entry(pairs,tiles,lane,base,kernel_column,unit,window,planes)) :
    {Model.pass_done((pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,Model.pass_shift(half,low,odd,U32.is_zero(pass),Model.pass_entries(window),_)) ==
      target : Model.Passes}
  Equal.trans(Model.Passes,Model.pass_done((pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,done),
    Model.Passes{Model.passing_pairs(done),passing_tiles(done),(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes},target,
    pass_done_form((pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,done),
    %Equal.sym(Tree.Tree<Tile.Tile16>,Model.passing_pairs(done),Model.passing_pairs(began),Model.pass_shift_pairs(half,low,odd,U32.is_zero(pass),Model.pass_entries(window),began)) :
      {Model.Passes{_,passing_tiles(done),(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes} == target : Model.Passes}
    {==})

# What the passes of a kernel column of a unit need: a pass's room, for every
# pass.
type ColumnRoom<-window: Packing.Stencil,-kernel_column: U32,-unit: U32,-depth: Nat> is Data:
  ColumnRoom{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},limit: {Nat.is_le(depth,31n) == True{} : Bool},
    capped: {Nat.is_le(U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap)) == True{} : Bool},taps_bound: {Nat.is_le(tap_rows_of(window),U32.to_nat(cap)) == True{} : Bool},
    taps_positive: {Nat.is_lt(0n,tap_rows_of(window)) == True{} : Bool},columns_positive: {Nat.is_lt(0n,tap_columns_of(window)) == True{} : Bool},
    fits: {Nat.is_lt(Nat.add(column_origin_of(window,kernel_column,unit),Nat.mul(Model.pass_entries(window),tap_columns_of(window))),
      Nat.add(tap_columns_of(window),U32.to_nat(U32.shln(1,depth)))) == True{} : Bool}}

def pass_room(+lane: U32,+base: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,
  +room: ColumnRoom<window,kernel_column,unit,depth>) -> PassRoom<pass_at(lane,base,kernel_column,unit,window,planes),Model.pass_entries(window),depth>:
  Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
  ColumnRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+fits} = room
  PassRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,fits}

# The tile of a target entry after n passes from a pass on.
def passes_tile(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,+window: Packing.Stencil,
  +planes: Packing.Planes,+target: Planes.Spot,kept: Tile.Tile8) -> Tile.Tile8:
  match n:
    case 0n: kept
    case 1n+ +rest:
      +shift = (base % 8 : U32)
      +at = pass_at(lane,base,kernel_column,unit,window,planes)
      passes_tile(rest,pairs,(pass + 1 : U32),next_lane(pass,lane,second_lane,window),next_base(base,window,planes),second_lane,kernel_column,unit,window,planes,target,
        entry_value(U32.is_zero(pass),lane_of(at),kept,shifted_at(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),
          Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))))

# The passes write the tile of a target entry pass by pass.
def passes_hit(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,
  +window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+target: Planes.Spot,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+target_inside: {Nat.is_lt(Planes.column_of(target),tap_rows_of(window)) == True{} : Bool},
  +target_below: {Nat.is_lt(Planes.element(target,tap_rows_of(window)),Model.pass_entries(window)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,n,Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes})),
    column_slot(window,kernel_column,unit,target)) ==
    passes_tile(n,pairs,pass,lane,base,second_lane,kernel_column,unit,window,planes,target,Tree.read(Tile.Tile8,tiles,column_slot(window,kernel_column,unit,target))) :
    Tile.Tile8}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
      +window = {Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
      +slot_word = column_slot(window,kernel_column,unit,target)
      +at = pass_at(lane,base,kernel_column,unit,window,planes)
      +written = pass_tiles(pairs,tiles,pass,lane,base,kernel_column,unit,window,planes)
      +lane_next = next_lane(pass,lane,second_lane,window)
      +base_next = next_base(base,window,planes)
      +shift = (base % 8 : U32)
      +half = U32.is_lt(shift,4)
      +low = U32.is_lt((shift % 4 : U32),2)
      +odd = U32.is_eq((shift % 2 : U32),1)
      +here = pass_room(lane,base,kernel_column,unit,window,planes,depth,room)
      +entries = Model.pass_entries(window)
      %Equal.sym(Model.Passes,Model.pass_step(Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}),
        Model.Passes{pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes},
        pass_advance(pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes)) :
        {Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,rest,_)),slot_word) ==
          passes_tile(1n+rest,pairs,pass,lane,base,second_lane,kernel_column,unit,window,planes,target,Tree.read(Tile.Tile8,tiles,slot_word)) : Tile.Tile8}
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,rest,
          Model.Passes{pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes})),slot_word),
        passes_tile(rest,pairs,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,target,Tree.read(Tile.Tile8,written,slot_word)),
        passes_tile(1n+rest,pairs,pass,lane,base,second_lane,kernel_column,unit,window,planes,target,Tree.read(Tile.Tile8,tiles,slot_word)),
        passes_hit(rest,pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,depth,room,target,
          shift_balanced(half,low,odd,U32.is_zero(pass),pairs,at,entries,depth,here,tiles,balanced),target_inside,target_below),
        Equal.cong(Tile.Tile8,Tile.Tile8,kept => passes_tile(rest,pairs,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,target,kept),
          Tree.read(Tile.Tile8,written,slot_word),
          entry_value(U32.is_zero(pass),lane_of(at),Tree.read(Tile.Tile8,tiles,slot_word),shifted_at(half,low,odd,Tree.read(Tile.Tile16,pairs,entry_pair(at,target)))),
          shift_hit(half,low,odd,U32.is_zero(pass),pairs,at,entries,depth,here,tiles,target,balanced,target_inside,target_below)))

# The passes leave a tile no entry of the column writes.
def passes_missed(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,
  +window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+query: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +outside: {any_from(Model.pass_entries(window),column_origin_of(window,kernel_column,unit),tap_columns_of(window),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,n,Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes})),
    U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
      +window = {Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
      +word = U32.from_nat(query)
      +at = pass_at(lane,base,kernel_column,unit,window,planes)
      +written = pass_tiles(pairs,tiles,pass,lane,base,kernel_column,unit,window,planes)
      +lane_next = next_lane(pass,lane,second_lane,window)
      +base_next = next_base(base,window,planes)
      +shift = (base % 8 : U32)
      +half = U32.is_lt(shift,4)
      +low = U32.is_lt((shift % 4 : U32),2)
      +odd = U32.is_eq((shift % 2 : U32),1)
      +here = pass_room(lane,base,kernel_column,unit,window,planes,depth,room)
      +entries = Model.pass_entries(window)
      %Equal.sym(Model.Passes,Model.pass_step(Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}),
        Model.Passes{pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes},
        pass_advance(pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes)) :
        {Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,rest,_)),word) == Tree.read(Tile.Tile8,tiles,word) : Tile.Tile8}
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,rest,
          Model.Passes{pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes})),word),
        Tree.read(Tile.Tile8,written,word),Tree.read(Tile.Tile8,tiles,word),
        passes_missed(rest,pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,depth,room,query,
          shift_balanced(half,low,odd,U32.is_zero(pass),pairs,at,entries,depth,here,tiles,balanced),query_inside,outside),
        shift_missed(half,low,odd,U32.is_zero(pass),pairs,at,entries,depth,here,tiles,query,balanced,query_inside,outside))

# The passes keep the tiles a tree of the same depth.
def passes_balanced(n: Nat,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+pass: U32,+lane: U32,+base: U32,+second_lane: U32,+kernel_column: U32,+unit: U32,
  +window: Packing.Stencil,+planes: Packing.Planes,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,n,Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}))):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      Packing.Stencil{+base_cell,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      Packing.Planes{+first_output_row,+rows,+row_width,+elements} = planes
      +window = {Packing.Stencil{base_cell,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      +planes = {Packing.Planes{first_output_row,rows,row_width,elements} : Packing.Planes}
      +at = pass_at(lane,base,kernel_column,unit,window,planes)
      +written = pass_tiles(pairs,tiles,pass,lane,base,kernel_column,unit,window,planes)
      +lane_next = next_lane(pass,lane,second_lane,window)
      +base_next = next_base(base,window,planes)
      +shift = (base % 8 : U32)
      %Equal.sym(Model.Passes,Model.pass_step(Model.Passes{pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes}),
        Model.Passes{pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes},
        pass_advance(pairs,tiles,pass,lane,base,second_lane,kernel_column,unit,window,planes)) :
        Shape.Balanced(Tile.Tile8,depth,passes_tiles(Loop.run(~Model.Passes,~Model.pass_step,rest,_)))
      passes_balanced(rest,pairs,written,(pass + 1 : U32),lane_next,base_next,second_lane,kernel_column,unit,window,planes,depth,room,
        shift_balanced(U32.is_lt(shift,4),U32.is_lt((shift % 4 : U32),2),U32.is_eq((shift % 2 : U32),1),U32.is_zero(pass),pairs,at,Model.pass_entries(window),depth,
          pass_room(lane,base,kernel_column,unit,window,planes,depth,room),tiles,balanced))

# ---- Tile numbers of the pack ----

def inner_of(+window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,+inner,groups} = window
  inner

# Tile number (unit * entries + entry) * columns + column.
def number(+unit: Nat,+column: Nat,+entry: Nat,+entries: Nat,+columns: Nat) -> Nat:
  Planes.radix(unit,entry,column,entries,columns)

# The unit and the kernel column of a tile number.
def unit_of(+query: Nat,+entries: Nat,+columns: Nat) -> Nat:
  Nat.div(Nat.div(query,columns),entries)

def column_index(+query: Nat,+columns: Nat) -> Nat:
  Nat.mod(query,columns)

def number_column(+unit: Nat,+column: Nat,+entry: Nat,+entries: Nat,+columns: Nat,+positive: {Nat.is_lt(0n,columns) == True{} : Bool},
  +inside: {Nat.is_lt(column,columns) == True{} : Bool}) -> {column_index(number(unit,column,entry,entries,columns),columns) == column : Nat}:
  +major = Nat.add(Nat.mul(unit,entries),entry)
  %Equal.sym(Nat,Nat.add(Nat.mul(major,columns),column),Nat.add(column,Nat.mul(major,columns)),Order.commutative(Nat.mul(major,columns),column)) :
    {Nat.mod(_,columns) == column : Nat}
  DivisionOrder.shifted_small_remainder(column,columns,major,positive,inside)

def number_unit(+unit: Nat,+column: Nat,+entry: Nat,+entries: Nat,+columns: Nat,+columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},
  +entries_positive: {Nat.is_lt(0n,entries) == True{} : Bool},+column_inside: {Nat.is_lt(column,columns) == True{} : Bool},
  +entry_inside: {Nat.is_lt(entry,entries) == True{} : Bool}) -> {unit_of(number(unit,column,entry,entries,columns),entries,columns) == unit : Nat}:
  +major = Nat.add(Nat.mul(unit,entries),entry)
  +quotient = Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(major,columns),column),columns),Nat.div(Nat.add(column,Nat.mul(major,columns)),columns),major,
    Equal.cong(Nat,Nat,value => Nat.div(value,columns),Nat.add(Nat.mul(major,columns),column),Nat.add(column,Nat.mul(major,columns)),
      Order.commutative(Nat.mul(major,columns),column)),
    DivisionOrder.shifted_small_quotient(column,columns,major,columns_positive,column_inside))
  %Equal.sym(Nat,Nat.div(number(unit,column,entry,entries,columns),columns),major,quotient) : {Nat.div(_,entries) == unit : Nat}
  Equal.trans(Nat,Nat.div(major,entries),Nat.div(Nat.add(entry,Nat.mul(unit,entries)),entries),unit,
    Equal.cong(Nat,Nat,value => Nat.div(value,entries),major,Nat.add(entry,Nat.mul(unit,entries)),Order.commutative(Nat.mul(unit,entries),entry)),
    DivisionOrder.shifted_small_quotient(entry,entries,unit,entries_positive,entry_inside))

# The tile of an entry from the first tile of its kernel column.
def slot_number(+origin: Nat,+unit: Nat,+column: Nat,+entry: Nat,+entries: Nat,+columns: Nat,
  +origin_eq: {origin == Nat.add(Nat.mul(unit,Nat.mul(entries,columns)),column) : Nat}) ->
  {Nat.add(origin,Nat.mul(entry,columns)) == number(unit,column,entry,entries,columns) : Nat}:
  +head = Nat.mul(unit,Nat.mul(entries,columns))
  +part = Nat.mul(entry,columns)
  +spread = Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(unit,entries),entry),columns),Nat.add(Nat.mul(Nat.mul(unit,entries),columns),part),Nat.add(head,part),
    RowMajor.distrib(Nat.mul(unit,entries),entry,columns),
    Equal.cong(Nat,Nat,value => Nat.add(value,part),Nat.mul(Nat.mul(unit,entries),columns),head,RowMajor.assoc(unit,entries,columns)))
  Equal.trans(Nat,Nat.add(origin,part),Nat.add(Nat.add(head,column),part),number(unit,column,entry,entries,columns),
    Equal.cong(Nat,Nat,value => Nat.add(value,part),origin,Nat.add(head,column),origin_eq),
    Equal.trans(Nat,Nat.add(Nat.add(head,column),part),Nat.add(head,Nat.add(column,part)),number(unit,column,entry,entries,columns),
      Addition.associative(head,column,part),
      Equal.trans(Nat,Nat.add(head,Nat.add(column,part)),Nat.add(head,Nat.add(part,column)),number(unit,column,entry,entries,columns),
        Equal.cong(Nat,Nat,value => Nat.add(head,value),Nat.add(column,part),Nat.add(part,column),Order.commutative(column,part)),
        Equal.trans(Nat,Nat.add(head,Nat.add(part,column)),Nat.add(Nat.add(head,part),column),number(unit,column,entry,entries,columns),
          Equal.sym(Nat,Nat.add(Nat.add(head,part),column),Nat.add(head,Nat.add(part,column)),Addition.associative(head,part,column)),
          Equal.cong(Nat,Nat,value => Nat.add(value,column),Nat.add(head,part),Nat.mul(Nat.add(Nat.mul(unit,entries),entry),columns),
            Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(unit,entries),entry),columns),Nat.add(head,part),spread))))))

# The tile of an entry of a kernel column is not a number of another unit
# or kernel column.
def term_apart(same: Bool,+origin: Nat,+unit: Nat,+column: Nat,+entry: Nat,+entries: Nat,+columns: Nat,+query: Nat,
  +origin_eq: {origin == Nat.add(Nat.mul(unit,Nat.mul(entries,columns)),column) : Nat},+columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},
  +entries_positive: {Nat.is_lt(0n,entries) == True{} : Bool},+column_inside: {Nat.is_lt(column,columns) == True{} : Bool},
  +entry_inside: {Nat.is_lt(entry,entries) == True{} : Bool},
  +apart: {Nat.is_eq(unit_of(query,entries,columns),unit) && Nat.is_eq(column,column_index(query,columns)) == False{} : Bool},
  +exact: {Nat.is_eq(Nat.add(origin,Nat.mul(entry,columns)),query) == same : Bool}) -> {Nat.is_eq(Nat.add(origin,Nat.mul(entry,columns)),query) == False{} : Bool}:
  match same:
    case False{}: exact
    case True{}:
      +value = Nat.add(origin,Nat.mul(entry,columns))
      +tile = number(unit,column,entry,entries,columns)
      +at_number = Equal.trans(Nat,query,value,tile,Equal.sym(Nat,value,query,Order.equal_of_is_eq(value,query,exact)),
        slot_number(origin,unit,column,entry,entries,columns,origin_eq))
      Empty.absurd({Nat.is_eq(value,query) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(unit_of(query,entries,columns),unit) && Nat.is_eq(column,column_index(query,columns)),True{},
          Equal.sym(Bool,Nat.is_eq(unit_of(query,entries,columns),unit) && Nat.is_eq(column,column_index(query,columns)),False{},apart),
          %Equal.sym(Nat,query,tile,at_number) : {Nat.is_eq(unit_of(_,entries,columns),unit) && Nat.is_eq(column,column_index(_,columns)) == True{} : Bool}
          %Equal.sym(Nat,unit_of(tile,entries,columns),unit,number_unit(unit,column,entry,entries,columns,columns_positive,entries_positive,column_inside,entry_inside)) :
            {Nat.is_eq(_,unit) && Nat.is_eq(column,column_index(tile,columns)) == True{} : Bool}
          %Equal.sym(Nat,column_index(tile,columns),column,number_column(unit,column,entry,entries,columns,columns_positive,column_inside)) :
            {Nat.is_eq(unit,unit) && Nat.is_eq(column,_) == True{} : Bool}
          Boolean.and_true(Nat.is_eq(unit,unit),Nat.is_eq(column,column),Order.equal_reflexive(unit),Order.equal_reflexive(column)))))

# No entry of a kernel column has a number of another unit or kernel column.
def any_none(n: Nat,+origin: Nat,+unit: Nat,+column: Nat,+start: Nat,+entries: Nat,+columns: Nat,+query: Nat,
  +origin_eq: {origin == Nat.add(Nat.mul(unit,Nat.mul(entries,columns)),column) : Nat},+columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},
  +entries_positive: {Nat.is_lt(0n,entries) == True{} : Bool},+column_inside: {Nat.is_lt(column,columns) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(start,n),entries) == True{} : Bool},
  +apart: {Nat.is_eq(unit_of(query,entries,columns),unit) && Nat.is_eq(column,column_index(query,columns)) == False{} : Bool}) ->
  {any_from(n,origin,columns,start,query) == False{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +term = term_apart(Nat.is_eq(Nat.add(origin,Nat.mul(start,columns)),query),origin,unit,column,start,entries,columns,query,origin_eq,columns_positive,entries_positive,
        column_inside,Order.room_index(start,rest,entries,remaining),apart,{==})
      %Equal.sym(Bool,Nat.is_eq(Nat.add(origin,Nat.mul(start,columns)),query),False{},term) : {_ || any_from(rest,origin,columns,1n+start,query) == False{} : Bool}
      %Equal.sym(Bool,any_from(rest,origin,columns,1n+start,query),False{},
        any_none(rest,origin,unit,column,1n+start,entries,columns,query,origin_eq,columns_positive,entries_positive,column_inside,
          Order.advance_room(start,rest,entries,remaining),apart)) : {False{} || _ == False{} : Bool}
      {==}

# What the pack needs: a pass's room for every kernel column of every unit,
# the inner extent the entries times the tap columns, and the tiles of the
# units inside a tree of depth depth.
type PackRoom<-window: Packing.Stencil,-units: Nat,-depth: Nat> is Data:
  PackRoom{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},limit: {Nat.is_le(depth,31n) == True{} : Bool},
    capped: {Nat.is_le(U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap)) == True{} : Bool},taps_bound: {Nat.is_le(tap_rows_of(window),U32.to_nat(cap)) == True{} : Bool},
    taps_positive: {Nat.is_lt(0n,tap_rows_of(window)) == True{} : Bool},columns_positive: {Nat.is_lt(0n,tap_columns_of(window)) == True{} : Bool},
    entries_positive: {Nat.is_lt(0n,Model.pass_entries(window)) == True{} : Bool},
    inner_exact: {U32.to_nat(inner_of(window)) == Nat.mul(Model.pass_entries(window),tap_columns_of(window)) : Nat},
    fits: {Nat.is_le(Nat.mul(units,Nat.mul(Model.pass_entries(window),tap_columns_of(window))),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}

def pack_columns(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>) -> {Nat.is_lt(0n,tap_columns_of(window)) == True{} : Bool}:
  PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,+columns_positive,entries_positive,inner_exact,fits} = room
  columns_positive

def pack_entries(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>) -> {Nat.is_lt(0n,Model.pass_entries(window)) == True{} : Bool}:
  PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,+entries_positive,inner_exact,fits} = room
  entries_positive

def pack_fits(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>) ->
  {Nat.is_le(Nat.mul(units,Nat.mul(Model.pass_entries(window),tap_columns_of(window))),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}:
  PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,entries_positive,inner_exact,+fits} = room
  fits

def pack_taps(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>) -> {Nat.is_lt(0n,tap_rows_of(window)) == True{} : Bool}:
  PackRoom{cap,cap_ok,limit,capped,taps_bound,+taps_positive,columns_positive,entries_positive,inner_exact,fits} = room
  taps_positive

def pack_inner(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>) ->
  {U32.to_nat(inner_of(window)) == Nat.mul(Model.pass_entries(window),tap_columns_of(window)) : Nat}:
  PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,entries_positive,+inner_exact,fits} = room
  inner_exact

# The first tile of a unit's block and one past it fit the tree.
def block_end(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,+unit: Nat,+unit_inside: {Nat.is_lt(unit,units) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+unit,Nat.mul(Model.pass_entries(window),tap_columns_of(window))),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}:
  PackRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+entries_positive,+inner_exact,+fits} = room
  +block = Nat.mul(Model.pass_entries(window),tap_columns_of(window))
  Order.transitive(Nat.mul(1n+unit,block),Nat.mul(units,block),U32.to_nat(U32.shln(1,depth)),
    RowMajor.mul_le(1n+unit,units,block,Order.strict_follow(unit,units,unit_inside)),fits)

# A kernel column's first tile lies before the end of its unit's block.
def origin_below(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,+unit: Nat,+column: Nat,
  +column_inside: {Nat.is_lt(column,tap_columns_of(window)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(unit,Nat.mul(Model.pass_entries(window),tap_columns_of(window))),column),
    Nat.mul(1n+unit,Nat.mul(Model.pass_entries(window),tap_columns_of(window)))) == True{} : Bool}:
  +block = Nat.mul(Model.pass_entries(window),tap_columns_of(window))
  +head = Nat.mul(unit,block)
  %Equal.sym(Nat,Nat.add(head,column),Nat.add(column,head),Order.commutative(head,column)) : {Nat.is_lt(_,Nat.mul(1n+unit,block)) == True{} : Bool}
  Order.add_lt(column,block,head,Order.lt_le(column,tap_columns_of(window),block,column_inside,
    DivisionOrder.le_multiple(Model.pass_entries(window),tap_columns_of(window),pack_entries(window,units,depth,room))))

# The first tile of a kernel column of a unit is unit * block + column.
def column_origin(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,+unit: Nat,+column: Nat,
  +unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+column_inside: {Nat.is_lt(column,tap_columns_of(window)) == True{} : Bool}) ->
  {column_origin_of(window,U32.from_nat(column),U32.from_nat(unit)) == Nat.add(Nat.mul(unit,Nat.mul(Model.pass_entries(window),tap_columns_of(window))),column) : Nat}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  PackRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+entries_positive,+inner_exact,+fits} = room
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +room = {PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,entries_positive,inner_exact,fits} : PackRoom<window,units,depth>}
  +size = U32.to_nat(U32.shln(1,depth))
  +block = Nat.mul(Model.pass_entries(window),tap_columns_of(window))
  +inner_n = U32.to_nat(inner)
  +value = Nat.add(Nat.mul(unit,block),column)
  +value_room = Order.transitive(value,size,U32.to_nat(cap),
    Bounds.strict_le(value,size,Order.lt_le(value,Nat.mul(1n+unit,block),size,origin_below(window,units,depth,room,unit,column,column_inside),
      block_end(window,units,depth,room,unit,unit_inside))),capped)
  +unit_room = Order.transitive(unit,value,U32.to_nat(cap),
    Order.transitive(unit,Nat.mul(unit,block),value,
      %Equal.sym(Nat,Nat.mul(unit,block),Nat.mul(block,unit),RowMajor.multiply_commutative(unit,block)) : {Nat.is_le(unit,_) == True{} : Bool}
        DivisionOrder.le_multiple(block,unit,Order.lt_le(0n,tap_columns_of(window),block,columns_positive,
          DivisionOrder.le_multiple(Model.pass_entries(window),tap_columns_of(window),entries_positive))),
      Order.le_add(Nat.mul(unit,block),column)),value_room)
  +column_room = Order.transitive(column,value,U32.to_nat(cap),
    %Equal.sym(Nat,Nat.add(Nat.mul(unit,block),column),Nat.add(column,Nat.mul(unit,block)),Order.commutative(Nat.mul(unit,block),column)) : {Nat.is_le(column,_) == True{} : Bool} Order.le_add(column,Nat.mul(unit,block)),value_room)
  +unit_back = Bounds.roundtrip(unit,cap,cap_ok,unit_room)
  +column_back = Bounds.roundtrip(column,cap,cap_ok,column_room)
  +unit_word = U32.from_nat(unit)
  +column_word = U32.from_nat(column)
  Equal.trans(Nat,U32.to_nat((unit_word * inner + column_word : U32)),Nat.add(Nat.mul(U32.to_nat(unit_word),inner_n),U32.to_nat(column_word)),value,
    Arithmetic.multiply_add_exact(unit_word,inner,column_word,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(unit_word),unit,unit_back) : {Nat.is_le(Nat.add(Nat.mul(_,inner_n),U32.to_nat(column_word)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(column_word),column,column_back) : {Nat.is_le(Nat.add(Nat.mul(unit,inner_n),_),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,inner_n,block,inner_exact) : {Nat.is_le(Nat.add(Nat.mul(unit,_),column),U32.to_nat(cap)) == True{} : Bool}
      value_room),
    Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(unit_word),inner_n),U32.to_nat(column_word)),Nat.add(Nat.mul(unit,inner_n),column),value,
      Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(unit_word),inner_n),U32.to_nat(column_word)),Nat.add(Nat.mul(unit,inner_n),U32.to_nat(column_word)),
        Nat.add(Nat.mul(unit,inner_n),column),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,inner_n),U32.to_nat(column_word)),U32.to_nat(unit_word),unit,unit_back),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(unit,inner_n),value),U32.to_nat(column_word),column,column_back)),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(unit,value),column),inner_n,block,inner_exact)))

# The tile word of a target entry is its number.
def slot_word(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,+unit: Nat,+column: Nat,+target: Planes.Spot,
  +unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+column_inside: {Nat.is_lt(column,tap_columns_of(window)) == True{} : Bool}) ->
  {column_slot(window,U32.from_nat(column),U32.from_nat(unit),target) ==
    U32.from_nat(number(unit,column,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window))) : U32}:
  Equal.cong(Nat,U32,value => U32.from_nat(value),
    Nat.add(column_origin_of(window,U32.from_nat(column),U32.from_nat(unit)),Nat.mul(Planes.element(target,tap_rows_of(window)),tap_columns_of(window))),
    number(unit,column,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window)),
    slot_number(column_origin_of(window,U32.from_nat(column),U32.from_nat(unit)),unit,column,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),
      tap_columns_of(window),column_origin(window,units,depth,room,unit,column,unit_inside,column_inside)))

# The room of a kernel column of a unit.
def column_room(+window: Packing.Stencil,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,+unit: Nat,+column: Nat,
  +unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+column_inside: {Nat.is_lt(column,tap_columns_of(window)) == True{} : Bool}) ->
  ColumnRoom<window,U32.from_nat(column),U32.from_nat(unit),depth>:
  PackRoom{+cap,+cap_ok,+limit,+capped,+taps_bound,+taps_positive,+columns_positive,+entries_positive,+inner_exact,+fits} = room
  +room = {PackRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,entries_positive,inner_exact,fits} : PackRoom<window,units,depth>}
  +origin = column_origin(window,units,depth,room,unit,column,unit_inside,column_inside)
  +ends = block_end(window,units,depth,room,unit,unit_inside)
  +size = U32.to_nat(U32.shln(1,depth))
  +columns = tap_columns_of(window)
  +block = Nat.mul(Model.pass_entries(window),columns)
  +head = Nat.mul(unit,block)
  +reach = Nat.add(block,head)
  +shuffled = Equal.trans(Nat,Nat.add(Nat.add(head,column),block),Nat.add(Nat.add(column,head),block),Nat.add(column,reach),
    Equal.cong(Nat,Nat,value => Nat.add(value,block),Nat.add(head,column),Nat.add(column,head),Order.commutative(head,column)),
    Equal.trans(Nat,Nat.add(Nat.add(column,head),block),Nat.add(column,Nat.add(head,block)),Nat.add(column,reach),Addition.associative(column,head,block),
      Equal.cong(Nat,Nat,value => Nat.add(column,value),Nat.add(head,block),reach,Order.commutative(head,block))))
  ColumnRoom{cap,cap_ok,limit,capped,taps_bound,taps_positive,columns_positive,
    %Equal.sym(Nat,column_origin_of(window,U32.from_nat(column),U32.from_nat(unit)),Nat.add(head,column),origin) :
      {Nat.is_lt(Nat.add(_,block),Nat.add(columns,size)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(head,column),block),Nat.add(column,reach),shuffled) : {Nat.is_lt(_,Nat.add(columns,size)) == True{} : Bool}
    Order.lt_le(Nat.add(column,reach),Nat.add(columns,reach),Nat.add(columns,size),Order.add_lt(column,columns,reach,column_inside),
      %Equal.sym(Bool,Nat.is_le(Nat.add(columns,reach),Nat.add(columns,size)),Nat.is_le(reach,size),Order.le_offset(columns,reach,size)) : {_ == True{} : Bool} ends)}

# ---- The kernel columns of a unit ----

def second_lane_of(+window: Packing.Stencil,+position: U32) -> U32:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  (output_width - position % output_width : U32)

# The tile of a target entry of a kernel column, from the tile it finds.
def column_tile(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+target: Planes.Spot) -> Tile.Tile8:
  passes_tile(Model.pass_count(window,position),pairs,0,0,Model.first_base(window,planes,position,kernel_column),second_lane_of(window,position),kernel_column,unit,window,
    planes,target,Tree.read(Tile.Tile8,tiles,column_slot(window,kernel_column,unit,target)))

def column_tiles(+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+pairs: Tree.Tree<Tile.Tile16>,
  +tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Model.arrays_tiles(Model.kernel_column_step(kernel_column,window,planes,position,unit,Model.Arrays{pairs,tiles}))

def arrays_form(+arrays: Model.Arrays) -> {arrays == Model.Arrays{Model.arrays_pairs(arrays),Model.arrays_tiles(arrays)} : Model.Arrays}:
  Model.Arrays{+pairs,+tiles} = arrays
  {==}

def tiles_done(+state: Model.Passes) -> {Model.arrays_tiles(Model.passes_done(state)) == passes_tiles(state) : Tree.Tree<Tile.Tile8>}:
  Model.Passes{+pairs,+tiles,+pass,+lane,+base,+second_lane,+kernel_column,+unit,+window,+planes} = state
  {==}

# A kernel column keeps the pairs and writes its tiles.
def column_form(+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+pairs: Tree.Tree<Tile.Tile16>,
  +tiles: Tree.Tree<Tile.Tile8>) ->
  {Model.kernel_column_step(kernel_column,window,planes,position,unit,Model.Arrays{pairs,tiles}) ==
    Model.Arrays{pairs,column_tiles(kernel_column,window,planes,position,unit,pairs,tiles)} : Model.Arrays}:
  +step = Model.kernel_column_step(kernel_column,window,planes,position,unit,Model.Arrays{pairs,tiles})
  Equal.trans(Model.Arrays,step,Model.Arrays{Model.arrays_pairs(step),Model.arrays_tiles(step)},Model.Arrays{pairs,Model.arrays_tiles(step)},arrays_form(step),
    Equal.cong(Tree.Tree<Tile.Tile16>,Model.Arrays,kept => Model.Arrays{kept,Model.arrays_tiles(step)},Model.arrays_pairs(step),pairs,
      Model.kernel_column_pairs(kernel_column,window,planes,position,unit,Model.Arrays{pairs,tiles})))

def column_hit(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +target_inside: {Nat.is_lt(Planes.column_of(target),tap_rows_of(window)) == True{} : Bool},
  +target_below: {Nat.is_lt(Planes.element(target,tap_rows_of(window)),Model.pass_entries(window)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,column_tiles(kernel_column,window,planes,position,unit,pairs,tiles),column_slot(window,kernel_column,unit,target)) ==
    column_tile(pairs,tiles,kernel_column,window,planes,position,unit,target) : Tile.Tile8}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +run = Loop.run(~Model.Passes,~Model.pass_step,Model.pass_count(window,position),Model.column_passes(pairs,tiles,kernel_column,window,planes,position,unit))
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.arrays_tiles(Model.passes_done(run)),passes_tiles(run),tiles_done(run)) :
    {Tree.read(Tile.Tile8,_,column_slot(window,kernel_column,unit,target)) == column_tile(pairs,tiles,kernel_column,window,planes,position,unit,target) : Tile.Tile8}
  passes_hit(Model.pass_count(window,position),pairs,tiles,0,0,Model.first_base(window,planes,position,kernel_column),(output_width - position % output_width : U32),
    kernel_column,unit,window,planes,depth,room,target,balanced,target_inside,target_below)

def column_missed(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +outside: {any_from(Model.pass_entries(window),column_origin_of(window,kernel_column,unit),tap_columns_of(window),0n,query) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,column_tiles(kernel_column,window,planes,position,unit,pairs,tiles),U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) :
    Tile.Tile8}:
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +run = Loop.run(~Model.Passes,~Model.pass_step,Model.pass_count(window,position),Model.column_passes(pairs,tiles,kernel_column,window,planes,position,unit))
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.arrays_tiles(Model.passes_done(run)),passes_tiles(run),tiles_done(run)) :
    {Tree.read(Tile.Tile8,_,U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}
  passes_missed(Model.pass_count(window,position),pairs,tiles,0,0,Model.first_base(window,planes,position,kernel_column),(output_width - position % output_width : U32),
    kernel_column,unit,window,planes,depth,room,query,balanced,query_inside,outside)

def column_balanced(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+kernel_column: U32,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,
  +unit: U32,+depth: Nat,+room: ColumnRoom<window,kernel_column,unit,depth>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,column_tiles(kernel_column,window,planes,position,unit,pairs,tiles)):
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
  +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
  +run = Loop.run(~Model.Passes,~Model.pass_step,Model.pass_count(window,position),Model.column_passes(pairs,tiles,kernel_column,window,planes,position,unit))
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.arrays_tiles(Model.passes_done(run)),passes_tiles(run),tiles_done(run)) : Shape.Balanced(Tile.Tile8,depth,_)
  passes_balanced(Model.pass_count(window,position),pairs,tiles,0,0,Model.first_base(window,planes,position,kernel_column),(output_width - position % output_width : U32),
    kernel_column,unit,window,planes,depth,room,balanced)

# The tiles after the kernel columns from kernel on.
def columns_tiles(n: Nat,+kernel: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+pairs: Tree.Tree<Tile.Tile16>,
  tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match n:
    case 0n: tiles
    case 1n+ +rest:
      columns_tiles(rest,1n+kernel,window,planes,position,unit,pairs,column_tiles(U32.from_nat(kernel),window,planes,position,unit,pairs,tiles))

def columns_form(n: Nat,+kernel: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: U32,+pairs: Tree.Tree<Tile.Tile16>,
  +tiles: Tree.Tree<Tile.Tile8>) ->
  {Model.kernel_columns(n,U32.from_nat(kernel),window,planes,position,unit,Model.Arrays{pairs,tiles}) ==
    Model.Arrays{pairs,columns_tiles(n,kernel,window,planes,position,unit,pairs,tiles)} : Model.Arrays}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +written = column_tiles(U32.from_nat(kernel),window,planes,position,unit,pairs,tiles)
      %Equal.sym(Model.Arrays,Model.kernel_column_step(U32.from_nat(kernel),window,planes,position,unit,Model.Arrays{pairs,tiles}),Model.Arrays{pairs,written},
        column_form(U32.from_nat(kernel),window,planes,position,unit,pairs,tiles)) :
        {Model.kernel_columns(rest,(U32.from_nat(kernel) + 1 : U32),window,planes,position,unit,_) ==
          Model.Arrays{pairs,columns_tiles(rest,1n+kernel,window,planes,position,unit,pairs,written)} : Model.Arrays}
      %Equal.sym(U32,(U32.from_nat(kernel) + 1 : U32),U32.from_nat(1n+kernel),Bounds.increment_encoded(kernel)) :
        {Model.kernel_columns(rest,_,window,planes,position,unit,Model.Arrays{pairs,written}) ==
          Model.Arrays{pairs,columns_tiles(rest,1n+kernel,window,planes,position,unit,pairs,written)} : Model.Arrays}
      columns_form(rest,1n+kernel,window,planes,position,unit,pairs,written)

# Kernel columns from kernel on leave the tile of a number of another unit or
# of a kernel column before kernel.
def columns_missed(n: Nat,+kernel: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: Nat,+units: Nat,+depth: Nat,
  +room: PackRoom<window,units,depth>,+unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+query: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(kernel,n),tap_columns_of(window)) == True{} : Bool},
  +missed: {Nat.is_eq(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),unit) &&
    Nat.is_le(kernel,column_index(query,tap_columns_of(window))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,columns_tiles(n,kernel,window,planes,position,U32.from_nat(unit),pairs,tiles),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +columns = tap_columns_of(window)
      +entries = Model.pass_entries(window)
      +word = U32.from_nat(query)
      +same = Nat.is_eq(unit_of(query,entries,columns),unit)
      +kernel_inside = Order.room_index(kernel,rest,columns,remaining)
      +here = column_room(window,units,depth,room,unit,kernel,unit_inside,kernel_inside)
      +written = column_tiles(U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),pairs,tiles)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,columns_tiles(rest,1n+kernel,window,planes,position,U32.from_nat(unit),pairs,written),word),
        Tree.read(Tile.Tile8,written,word),Tree.read(Tile.Tile8,tiles,word),
        columns_missed(rest,1n+kernel,window,planes,position,unit,units,depth,room,unit_inside,pairs,written,query,
          column_balanced(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,balanced),query_inside,
          Order.advance_room(kernel,rest,columns,remaining),Planes.missed_next(same,kernel,column_index(query,columns),missed)),
        column_missed(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,query,balanced,query_inside,
          any_none(entries,column_origin_of(window,U32.from_nat(kernel),U32.from_nat(unit)),unit,kernel,0n,entries,columns,query,
            column_origin(window,units,depth,room,unit,kernel,unit_inside,kernel_inside),pack_columns(window,units,depth,room),pack_entries(window,units,depth,room),
            kernel_inside,Order.reflexive(entries),Planes.missed_now(same,kernel,column_index(query,columns),missed))))

# The kernel columns write the tile of a target entry of a kernel column.
def columns_hit(n: Nat,hit: Bool,+kernel: Nat,+aim: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: Nat,+units: Nat,+depth: Nat,
  +room: PackRoom<window,units,depth>,+unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,
  +target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+target_inside: {Nat.is_lt(Planes.column_of(target),tap_rows_of(window)) == True{} : Bool},
  +target_below: {Nat.is_lt(Planes.element(target,tap_rows_of(window)),Model.pass_entries(window)) == True{} : Bool},+exact: {Nat.is_eq(kernel,aim) == hit : Bool},
  +reached: {Nat.is_le(kernel,aim) == True{} : Bool},+within: {Nat.is_lt(aim,Nat.add(kernel,n)) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(kernel,n),tap_columns_of(window)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,columns_tiles(n,kernel,window,planes,position,U32.from_nat(unit),pairs,tiles),
    U32.from_nat(number(unit,aim,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window)))) ==
    column_tile(pairs,tiles,U32.from_nat(aim),window,planes,position,U32.from_nat(unit),target) : Tile.Tile8}:
  match n:
    case 0n:
      +before = Order.le_lt_trans(kernel,aim,kernel,reached,%Addition.zero_right(kernel) : {Nat.is_lt(aim,_) == True{} : Bool} within)
      Empty.absurd({Tree.read(Tile.Tile8,tiles,U32.from_nat(number(unit,aim,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window)))) ==
          column_tile(pairs,tiles,U32.from_nat(aim),window,planes,position,U32.from_nat(unit),target) : Tile.Tile8},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(kernel,kernel),True{},Equal.sym(Bool,Nat.is_eq(kernel,kernel),False{},
          Order.less_not_equal(kernel,kernel,before)),Order.equal_reflexive(kernel))))
    case 1n+ +rest:
      match hit:
        case True{}:
          +columns = tap_columns_of(window)
          +entries = Model.pass_entries(window)
          +entry = Planes.element(target,tap_rows_of(window))
          +aim_inside = Order.lt_le(aim,Nat.add(kernel,1n+rest),columns,within,remaining)
          +kernel_inside = Order.room_index(kernel,rest,columns,remaining)
          +here = column_room(window,units,depth,room,unit,kernel,unit_inside,kernel_inside)
          +written = column_tiles(U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),pairs,tiles)
          +query = number(unit,aim,entry,entries,columns)
          +columns_positive = pack_columns(window,units,depth,room)
          +entries_positive = pack_entries(window,units,depth,room)
          +query_inside = Order.lt_le(query,Nat.mul(Nat.mul(units,entries),columns),U32.to_nat(U32.shln(1,depth)),
            Planes.index_below(Planes.Plane{unit,aim},units,entries,columns,entry,Planes.PlaneInside{unit_inside,aim_inside},target_below),
            %Equal.sym(Nat,Nat.mul(Nat.mul(units,entries),columns),Nat.mul(units,Nat.mul(entries,columns)),RowMajor.assoc(units,entries,columns)) :
              {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
              pack_fits(window,units,depth,room))
          +equal = Order.equal_of_is_eq(kernel,aim,exact)
          %equal : {Tree.read(Tile.Tile8,columns_tiles(1n+rest,kernel,window,planes,position,U32.from_nat(unit),pairs,tiles),
            U32.from_nat(number(unit,_,entry,entries,columns))) == column_tile(pairs,tiles,U32.from_nat(_),window,planes,position,U32.from_nat(unit),target) : Tile.Tile8}
          +own = number(unit,kernel,entry,entries,columns)
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,columns_tiles(rest,1n+kernel,window,planes,position,U32.from_nat(unit),pairs,written),U32.from_nat(own)),
            Tree.read(Tile.Tile8,written,U32.from_nat(own)),column_tile(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),target),
            columns_missed(rest,1n+kernel,window,planes,position,unit,units,depth,room,unit_inside,pairs,written,own,
              column_balanced(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,balanced),
              %Equal.sym(Nat,kernel,aim,equal) : {Nat.is_lt(number(unit,_,entry,entries,columns),U32.to_nat(U32.shln(1,depth))) == True{} : Bool} query_inside,
              Order.advance_room(kernel,rest,columns,remaining),
              %Equal.sym(Nat,unit_of(own,entries,columns),unit,number_unit(unit,kernel,entry,entries,columns,columns_positive,entries_positive,kernel_inside,target_below)) :
                {Nat.is_eq(_,unit) && Nat.is_le(1n+kernel,column_index(own,columns)) == False{} : Bool}
              %Equal.sym(Nat,column_index(own,columns),kernel,number_column(unit,kernel,entry,entries,columns,columns_positive,kernel_inside)) :
                {Nat.is_eq(unit,unit) && Nat.is_le(1n+kernel,_) == False{} : Bool}
              %Equal.sym(Bool,Nat.is_eq(unit,unit),True{},Order.equal_reflexive(unit)) : {_ && Nat.is_le(1n+kernel,kernel) == False{} : Bool}
              Planes.not_le_successor(kernel)),
            %slot_word(window,units,depth,room,unit,kernel,target,unit_inside,kernel_inside) :
              {Tree.read(Tile.Tile8,written,_) == column_tile(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),target) : Tile.Tile8}
            column_hit(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,target,balanced,target_inside,target_below))
        case False{}:
          +columns = tap_columns_of(window)
          +entries = Model.pass_entries(window)
          +entry = Planes.element(target,tap_rows_of(window))
          +aim_inside = Order.lt_le(aim,Nat.add(kernel,1n+rest),columns,within,remaining)
          +kernel_inside = Order.room_index(kernel,rest,columns,remaining)
          +here = column_room(window,units,depth,room,unit,kernel,unit_inside,kernel_inside)
          +written = column_tiles(U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),pairs,tiles)
          +query = number(unit,aim,entry,entries,columns)
          +columns_positive = pack_columns(window,units,depth,room)
          +entries_positive = pack_entries(window,units,depth,room)
          +query_inside = Order.lt_le(query,Nat.mul(Nat.mul(units,entries),columns),U32.to_nat(U32.shln(1,depth)),
            Planes.index_below(Planes.Plane{unit,aim},units,entries,columns,entry,Planes.PlaneInside{unit_inside,aim_inside},target_below),
            %Equal.sym(Nat,Nat.mul(Nat.mul(units,entries),columns),Nat.mul(units,Nat.mul(entries,columns)),RowMajor.assoc(units,entries,columns)) :
              {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
              pack_fits(window,units,depth,room))
          +word = U32.from_nat(query)
          +slot_aim = column_slot(window,U32.from_nat(aim),U32.from_nat(unit),target)
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,columns_tiles(rest,1n+kernel,window,planes,position,U32.from_nat(unit),pairs,written),word),
            column_tile(pairs,written,U32.from_nat(aim),window,planes,position,U32.from_nat(unit),target),
            column_tile(pairs,tiles,U32.from_nat(aim),window,planes,position,U32.from_nat(unit),target),
            columns_hit(rest,Nat.is_eq(1n+kernel,aim),1n+kernel,aim,window,planes,position,unit,units,depth,room,unit_inside,pairs,written,target,
              column_balanced(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,balanced),target_inside,target_below,{==},
              Order.strict_follow(kernel,aim,Planes.below_of_unequal(kernel,aim,reached,exact)),
              %Addition.successor_right(kernel,rest) : {Nat.is_lt(aim,_) == True{} : Bool} within,Order.advance_room(kernel,rest,columns,remaining)),
            Equal.cong(Tile.Tile8,Tile.Tile8,kept => passes_tile(Model.pass_count(window,position),pairs,0,0,Model.first_base(window,planes,position,U32.from_nat(aim)),
                second_lane_of(window,position),U32.from_nat(aim),U32.from_nat(unit),window,planes,target,kept),
              Tree.read(Tile.Tile8,written,slot_aim),Tree.read(Tile.Tile8,tiles,slot_aim),
              %Equal.sym(U32,slot_aim,word,slot_word(window,units,depth,room,unit,aim,target,unit_inside,aim_inside)) :
                {Tree.read(Tile.Tile8,written,_) == Tree.read(Tile.Tile8,tiles,_) : Tile.Tile8}
              column_missed(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,here,query,balanced,query_inside,
                any_none(entries,column_origin_of(window,U32.from_nat(kernel),U32.from_nat(unit)),unit,kernel,0n,entries,columns,query,
                  column_origin(window,units,depth,room,unit,kernel,unit_inside,kernel_inside),columns_positive,entries_positive,kernel_inside,Order.reflexive(entries),
                  %Equal.sym(Nat,unit_of(query,entries,columns),unit,number_unit(unit,aim,entry,entries,columns,columns_positive,entries_positive,aim_inside,target_below)) :
                    {Nat.is_eq(_,unit) && Nat.is_eq(kernel,column_index(query,columns)) == False{} : Bool}
                  %Equal.sym(Nat,column_index(query,columns),aim,number_column(unit,aim,entry,entries,columns,columns_positive,aim_inside)) :
                    {Nat.is_eq(unit,unit) && Nat.is_eq(kernel,_) == False{} : Bool}
                  %Equal.sym(Bool,Nat.is_eq(unit,unit),True{},Order.equal_reflexive(unit)) : {_ && Nat.is_eq(kernel,aim) == False{} : Bool}
                  exact))))

# The kernel columns keep the tiles a tree of the same depth.
def columns_balanced(n: Nat,+kernel: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+position: U32,+unit: Nat,+units: Nat,+depth: Nat,
  +room: PackRoom<window,units,depth>,+unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+remaining: {Nat.is_le(Nat.add(kernel,n),tap_columns_of(window)) == True{} : Bool}) ->
  Shape.Balanced(Tile.Tile8,depth,columns_tiles(n,kernel,window,planes,position,U32.from_nat(unit),pairs,tiles)):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      +kernel_inside = Order.room_index(kernel,rest,tap_columns_of(window),remaining)
      columns_balanced(rest,1n+kernel,window,planes,position,unit,units,depth,room,unit_inside,pairs,
        column_tiles(U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),pairs,tiles),
        column_balanced(pairs,tiles,U32.from_nat(kernel),window,planes,position,U32.from_nat(unit),depth,
          column_room(window,units,depth,room,unit,kernel,unit_inside,kernel_inside),balanced),Order.advance_room(kernel,rest,tap_columns_of(window),remaining))

# ---- The units ----

# The position of a unit's first output.
def position_of(+start: U32,+unit: Nat) -> U32:
  (start + U32.from_nat(unit) * 8 : U32)

def position_step(+start: U32,+unit: Nat) -> {(position_of(start,unit) + 8 : U32) == position_of(start,1n+unit) : U32}:
  Equal.trans(U32,((start + U32.from_nat(unit) * 8 : U32) + 8 : U32),(start + (U32.from_nat(unit) * 8 + 8 : U32) : U32),position_of(start,1n+unit),
    Arithmetic.add_associative(start,(U32.from_nat(unit) * 8 : U32),8),
    Equal.cong(U32,U32,tail => (start + tail : U32),(U32.from_nat(unit) * 8 + 8 : U32),(U32.from_nat(1n+unit) * 8 : U32),
      Equal.trans(U32,(U32.from_nat(unit) * 8 + 8 : U32),((U32.from_nat(unit) + 1 : U32) * 8 : U32),(U32.from_nat(1n+unit) * 8 : U32),
        Equal.sym(U32,((U32.from_nat(unit) + 1 : U32) * 8 : U32),(U32.from_nat(unit) * 8 + 8 : U32),Arithmetic.multiply_successor(U32.from_nat(unit),8)),
        Equal.cong(U32,U32,index => (index * 8 : U32),(U32.from_nat(unit) + 1 : U32),U32.from_nat(1n+unit),Bounds.increment_encoded(unit)))))

def position_zero(+start: U32) -> {start == position_of(start,0n) : U32}:
  Equal.sym(U32,position_of(start,0n),start,
    Equal.trans(U32,(start + U32.from_nat(0n) * 8 : U32),(start + 0 : U32),start,
      Equal.cong(U32,U32,tail => (start + tail : U32),(U32.from_nat(0n) * 8 : U32),0,Arithmetic.multiply_zero_left(8)),Arithmetic.add_zero(start)))

# The tiles after the units from unit on.
def units_tiles(n: Nat,+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+pairs: Tree.Tree<Tile.Tile16>,tiles: Tree.Tree<Tile.Tile8>) ->
  Tree.Tree<Tile.Tile8>:
  match n:
    case 0n: tiles
    case 1n+ +rest:
      units_tiles(rest,1n+unit,window,planes,start,pairs,columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles))

def units_form(n: Nat,+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {Model.units(n,window,planes,position_of(start,unit),U32.from_nat(unit),Model.Arrays{pairs,tiles}) ==
    Model.Arrays{pairs,units_tiles(n,unit,window,planes,start,pairs,tiles)} : Model.Arrays}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner,+groups} = window
      +window = {Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width,inner,groups} : Packing.Stencil}
      +position = position_of(start,unit)
      +written = columns_tiles(tap_columns_of(window),0n,window,planes,position,U32.from_nat(unit),pairs,tiles)
      %Equal.sym(Model.Arrays,Model.kernel_columns(tap_columns_of(window),U32.from_nat(0n),window,planes,position,U32.from_nat(unit),Model.Arrays{pairs,tiles}),
        Model.Arrays{pairs,written},columns_form(tap_columns_of(window),0n,window,planes,position,U32.from_nat(unit),pairs,tiles)) :
        {Model.units(rest,window,planes,(position + 8 : U32),(U32.from_nat(unit) + 1 : U32),_) ==
          Model.Arrays{pairs,units_tiles(rest,1n+unit,window,planes,start,pairs,written)} : Model.Arrays}
      %Equal.sym(U32,(position + 8 : U32),position_of(start,1n+unit),position_step(start,unit)) :
        {Model.units(rest,window,planes,_,(U32.from_nat(unit) + 1 : U32),Model.Arrays{pairs,written}) ==
          Model.Arrays{pairs,units_tiles(rest,1n+unit,window,planes,start,pairs,written)} : Model.Arrays}
      %Equal.sym(U32,(U32.from_nat(unit) + 1 : U32),U32.from_nat(1n+unit),Bounds.increment_encoded(unit)) :
        {Model.units(rest,window,planes,position_of(start,1n+unit),_,Model.Arrays{pairs,written}) ==
          Model.Arrays{pairs,units_tiles(rest,1n+unit,window,planes,start,pairs,written)} : Model.Arrays}
      units_form(rest,1n+unit,window,planes,start,pairs,written)

# A unit leaves the tile of a number of another unit.
def unit_missed(+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+query: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +apart: {Nat.is_eq(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),unit) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles),U32.from_nat(query)) ==
    Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  columns_missed(tap_columns_of(window),0n,window,planes,position_of(start,unit),unit,units,depth,room,unit_inside,pairs,tiles,query,balanced,query_inside,
    Order.reflexive(tap_columns_of(window)),
    %Equal.sym(Bool,Nat.is_eq(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),unit),False{},apart) :
      {_ && Nat.is_le(0n,column_index(query,tap_columns_of(window))) == False{} : Bool}
    {==})

def unit_balanced(+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +unit_inside: {Nat.is_lt(unit,units) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles)):
  columns_balanced(tap_columns_of(window),0n,window,planes,position_of(start,unit),unit,units,depth,room,unit_inside,pairs,tiles,balanced,Order.reflexive(tap_columns_of(window)))

# Units past a number's unit leave its tile.
def units_after(n: Nat,+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+remaining: {Nat.is_le(Nat.add(unit,n),units) == True{} : Bool},
  +behind: {Nat.is_le(unit,unit_of(query,Model.pass_entries(window),tap_columns_of(window))) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,units_tiles(n,unit,window,planes,start,pairs,tiles),U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +owner = unit_of(query,Model.pass_entries(window),tap_columns_of(window))
      +unit_inside = Order.room_index(unit,rest,units,remaining)
      +written = columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,units_tiles(rest,1n+unit,window,planes,start,pairs,written),U32.from_nat(query)),
        Tree.read(Tile.Tile8,written,U32.from_nat(query)),Tree.read(Tile.Tile8,tiles,U32.from_nat(query)),
        units_after(rest,1n+unit,window,planes,start,units,depth,room,pairs,written,query,
          unit_balanced(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,balanced),query_inside,Order.advance_room(unit,rest,units,remaining),
          Planes.not_le_next(unit,owner,Nat.is_le(1n+unit,owner),{==},behind)),
        unit_missed(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,query,balanced,query_inside,
          Equal.trans(Bool,Nat.is_eq(owner,unit),Nat.is_eq(unit,owner),False{},Order.equal_symmetric(owner,unit),
            Planes.not_equal_of_not_le(unit,owner,Nat.is_eq(unit,owner),{==},behind))))

# Units before a number's unit leave its tile.
def units_before(n: Nat,+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+query: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},+remaining: {Nat.is_le(Nat.add(unit,n),units) == True{} : Bool},
  +ahead: {Nat.is_le(Nat.add(unit,n),unit_of(query,Model.pass_entries(window),tap_columns_of(window))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,units_tiles(n,unit,window,planes,start,pairs,tiles),U32.from_nat(query)) == Tree.read(Tile.Tile8,tiles,U32.from_nat(query)) : Tile.Tile8}:
  match n:
    case 0n: {==}
    case 1n+ +rest:
      +owner = unit_of(query,Model.pass_entries(window),tap_columns_of(window))
      +unit_inside = Order.room_index(unit,rest,units,remaining)
      +written = columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles)
      +below = Order.lt_le(unit,Nat.add(unit,1n+rest),owner,Order.room_index(unit,rest,Nat.add(unit,1n+rest),Order.reflexive(Nat.add(unit,1n+rest))),ahead)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,units_tiles(rest,1n+unit,window,planes,start,pairs,written),U32.from_nat(query)),
        Tree.read(Tile.Tile8,written,U32.from_nat(query)),Tree.read(Tile.Tile8,tiles,U32.from_nat(query)),
        units_before(rest,1n+unit,window,planes,start,units,depth,room,pairs,written,query,
          unit_balanced(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,balanced),query_inside,Order.advance_room(unit,rest,units,remaining),
          %Addition.successor_right(unit,rest) : {Nat.is_le(_,owner) == True{} : Bool} ahead),
        unit_missed(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,query,balanced,query_inside,
          Equal.trans(Bool,Nat.is_eq(owner,unit),Nat.is_eq(unit,owner),False{},Order.equal_symmetric(owner,unit),Order.less_not_equal(unit,owner,below))))

# The units write the tile of a target entry of a kernel column of a unit.
def units_hit(n: Nat,hit: Bool,+unit: Nat,+aim: Nat,+column: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,
  +room: PackRoom<window,units,depth>,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+target: Planes.Spot,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +column_inside: {Nat.is_lt(column,tap_columns_of(window)) == True{} : Bool},+target_inside: {Nat.is_lt(Planes.column_of(target),tap_rows_of(window)) == True{} : Bool},
  +target_below: {Nat.is_lt(Planes.element(target,tap_rows_of(window)),Model.pass_entries(window)) == True{} : Bool},+exact: {Nat.is_eq(unit,aim) == hit : Bool},
  +reached: {Nat.is_le(unit,aim) == True{} : Bool},+within: {Nat.is_lt(aim,Nat.add(unit,n)) == True{} : Bool},+remaining: {Nat.is_le(Nat.add(unit,n),units) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,units_tiles(n,unit,window,planes,start,pairs,tiles),
    U32.from_nat(number(aim,column,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window)))) ==
    column_tile(pairs,tiles,U32.from_nat(column),window,planes,position_of(start,aim),U32.from_nat(aim),target) : Tile.Tile8}:
  match n:
    case 0n:
      +before = Order.le_lt_trans(unit,aim,unit,reached,%Addition.zero_right(unit) : {Nat.is_lt(aim,_) == True{} : Bool} within)
      Empty.absurd({Tree.read(Tile.Tile8,tiles,U32.from_nat(number(aim,column,Planes.element(target,tap_rows_of(window)),Model.pass_entries(window),tap_columns_of(window)))) ==
          column_tile(pairs,tiles,U32.from_nat(column),window,planes,position_of(start,aim),U32.from_nat(aim),target) : Tile.Tile8},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(unit,unit),True{},Equal.sym(Bool,Nat.is_eq(unit,unit),False{},Order.less_not_equal(unit,unit,before)),
          Order.equal_reflexive(unit))))
    case 1n+ +rest:
      match hit:
        case True{}:
          +columns = tap_columns_of(window)
          +entries = Model.pass_entries(window)
          +entry = Planes.element(target,tap_rows_of(window))
          +equal = Order.equal_of_is_eq(unit,aim,exact)
          +unit_inside = Order.room_index(unit,rest,units,remaining)
          +written = columns_tiles(columns,0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles)
          +own = number(unit,column,entry,entries,columns)
          +own_inside = Order.lt_le(own,Nat.mul(Nat.mul(units,entries),columns),U32.to_nat(U32.shln(1,depth)),
            Planes.index_below(Planes.Plane{unit,column},units,entries,columns,entry,Planes.PlaneInside{unit_inside,column_inside},target_below),
            %Equal.sym(Nat,Nat.mul(Nat.mul(units,entries),columns),Nat.mul(units,Nat.mul(entries,columns)),RowMajor.assoc(units,entries,columns)) :
              {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
              pack_fits(window,units,depth,room))
          %equal : {Tree.read(Tile.Tile8,units_tiles(1n+rest,unit,window,planes,start,pairs,tiles),U32.from_nat(number(_,column,entry,entries,columns))) ==
            column_tile(pairs,tiles,U32.from_nat(column),window,planes,position_of(start,_),U32.from_nat(_),target) : Tile.Tile8}
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,units_tiles(rest,1n+unit,window,planes,start,pairs,written),U32.from_nat(own)),
            Tree.read(Tile.Tile8,written,U32.from_nat(own)),column_tile(pairs,tiles,U32.from_nat(column),window,planes,position_of(start,unit),U32.from_nat(unit),target),
            units_after(rest,1n+unit,window,planes,start,units,depth,room,pairs,written,own,
              unit_balanced(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,balanced),own_inside,Order.advance_room(unit,rest,units,remaining),
              %Equal.sym(Nat,unit_of(own,entries,columns),unit,
                number_unit(unit,column,entry,entries,columns,pack_columns(window,units,depth,room),pack_entries(window,units,depth,room),column_inside,target_below)) :
                {Nat.is_le(1n+unit,_) == False{} : Bool}
              Planes.not_le_successor(unit)),
            columns_hit(columns,Nat.is_eq(0n,column),0n,column,window,planes,position_of(start,unit),unit,units,depth,room,unit_inside,pairs,tiles,target,balanced,
              target_inside,target_below,{==},Order.zero_le(column),column_inside,Order.reflexive(columns)))
        case False{}:
          +columns = tap_columns_of(window)
          +entries = Model.pass_entries(window)
          +entry = Planes.element(target,tap_rows_of(window))
          +unit_inside = Order.room_index(unit,rest,units,remaining)
          +aim_inside = Order.lt_le(aim,Nat.add(unit,1n+rest),units,within,remaining)
          +written = columns_tiles(columns,0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles)
          +query = number(aim,column,entry,entries,columns)
          +word = U32.from_nat(query)
          +slot_aim = column_slot(window,U32.from_nat(column),U32.from_nat(aim),target)
          +position = position_of(start,aim)
          +query_inside = Order.lt_le(query,Nat.mul(Nat.mul(units,entries),columns),U32.to_nat(U32.shln(1,depth)),
            Planes.index_below(Planes.Plane{aim,column},units,entries,columns,entry,Planes.PlaneInside{aim_inside,column_inside},target_below),
            %Equal.sym(Nat,Nat.mul(Nat.mul(units,entries),columns),Nat.mul(units,Nat.mul(entries,columns)),RowMajor.assoc(units,entries,columns)) :
              {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
              pack_fits(window,units,depth,room))
          Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,units_tiles(rest,1n+unit,window,planes,start,pairs,written),word),
            column_tile(pairs,written,U32.from_nat(column),window,planes,position,U32.from_nat(aim),target),
            column_tile(pairs,tiles,U32.from_nat(column),window,planes,position,U32.from_nat(aim),target),
            units_hit(rest,Nat.is_eq(1n+unit,aim),1n+unit,aim,column,window,planes,start,units,depth,room,pairs,written,target,
              unit_balanced(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,balanced),column_inside,target_inside,target_below,{==},
              Order.strict_follow(unit,aim,Planes.below_of_unequal(unit,aim,reached,exact)),
              %Addition.successor_right(unit,rest) : {Nat.is_lt(aim,_) == True{} : Bool} within,Order.advance_room(unit,rest,units,remaining)),
            Equal.cong(Tile.Tile8,Tile.Tile8,kept => passes_tile(Model.pass_count(window,position),pairs,0,0,Model.first_base(window,planes,position,U32.from_nat(column)),
                second_lane_of(window,position),U32.from_nat(column),U32.from_nat(aim),window,planes,target,kept),
              Tree.read(Tile.Tile8,written,slot_aim),Tree.read(Tile.Tile8,tiles,slot_aim),
              %Equal.sym(U32,slot_aim,word,slot_word(window,units,depth,room,aim,column,target,aim_inside,column_inside)) :
                {Tree.read(Tile.Tile8,written,_) == Tree.read(Tile.Tile8,tiles,_) : Tile.Tile8}
              unit_missed(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,query,balanced,query_inside,
                %Equal.sym(Nat,unit_of(query,entries,columns),aim,
                  number_unit(aim,column,entry,entries,columns,pack_columns(window,units,depth,room),pack_entries(window,units,depth,room),column_inside,target_below)) :
                  {Nat.is_eq(_,unit) == False{} : Bool}
                Equal.trans(Bool,Nat.is_eq(aim,unit),Nat.is_eq(unit,aim),False{},Order.equal_symmetric(aim,unit),exact))))

def units_balanced(n: Nat,+unit: Nat,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +remaining: {Nat.is_le(Nat.add(unit,n),units) == True{} : Bool}) -> Shape.Balanced(Tile.Tile8,depth,units_tiles(n,unit,window,planes,start,pairs,tiles)):
  match n:
    case 0n: balanced
    case 1n+ +rest:
      +unit_inside = Order.room_index(unit,rest,units,remaining)
      units_balanced(rest,1n+unit,window,planes,start,units,depth,room,pairs,
        columns_tiles(tap_columns_of(window),0n,window,planes,position_of(start,unit),U32.from_nat(unit),pairs,tiles),
        unit_balanced(unit,window,planes,start,units,depth,room,unit_inside,pairs,tiles,balanced),Order.advance_room(unit,rest,units,remaining))

# ---- The pack ----

def packed_form(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>) ->
  {Model.packed_tiles(window,planes,count,start,pairs,tiles) == units_tiles(Model.unit_count(window,count),0n,window,planes,start,pairs,tiles) : Tree.Tree<Tile.Tile8>}:
  +units = Model.unit_count(window,count)
  +origin = {Model.Arrays{pairs,tiles} : Model.Arrays}
  Equal.trans(Tree.Tree<Tile.Tile8>,Model.arrays_tiles(Model.units(units,window,planes,start,0,origin)),
    Model.arrays_tiles(Model.units(units,window,planes,position_of(start,0n),U32.from_nat(0n),origin)),units_tiles(units,0n,window,planes,start,pairs,tiles),
    Equal.cong(U32,Tree.Tree<Tile.Tile8>,first => Model.arrays_tiles(Model.units(units,window,planes,first,0,origin)),start,position_of(start,0n),position_zero(start)),
    Equal.cong(Model.Arrays,Tree.Tree<Tile.Tile8>,arrays => Model.arrays_tiles(arrays),Model.units(units,window,planes,position_of(start,0n),U32.from_nat(0n),origin),
      Model.Arrays{pairs,units_tiles(units,0n,window,planes,start,pairs,tiles)},units_form(units,0n,window,planes,start,pairs,tiles)))

# The entry and the spot of a tile number.
def entry_of(+query: Nat,+entries: Nat,+columns: Nat) -> Nat:
  Nat.mod(Nat.div(query,columns),entries)

def spot_of(+query: Nat,+entries: Nat,+columns: Nat,+rows: Nat) -> Planes.Spot:
  +entry = entry_of(query,entries,columns)
  Planes.Spot{Nat.div(entry,rows),Nat.mod(entry,rows)}

# Every number is the number of its unit, kernel column and entry.
def number_of(+query: Nat,+entries: Nat,+columns: Nat,+entries_positive: {Nat.is_lt(0n,entries) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool}) ->
  {number(unit_of(query,entries,columns),column_index(query,columns),entry_of(query,entries,columns),entries,columns) == query : Nat}:
  +major = Nat.div(query,columns)
  Equal.trans(Nat,number(unit_of(query,entries,columns),column_index(query,columns),entry_of(query,entries,columns),entries,columns),
    Nat.add(Nat.mul(major,columns),Nat.mod(query,columns)),query,
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,columns),Nat.mod(query,columns)),Nat.add(Nat.mul(Nat.div(major,entries),entries),Nat.mod(major,entries)),major,
      Equal.sym(Nat,major,Nat.add(Nat.mul(Nat.div(major,entries),entries),Nat.mod(major,entries)),DivisionOrder.quotient_exact(major,entries,entries_positive))),
    Equal.sym(Nat,query,Nat.add(Nat.mul(major,columns),Nat.mod(query,columns)),DivisionOrder.quotient_exact(query,columns,columns_positive)))

def spot_entry(+query: Nat,+entries: Nat,+columns: Nat,+rows: Nat,+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool}) ->
  {Planes.element(spot_of(query,entries,columns,rows),rows) == entry_of(query,entries,columns) : Nat}:
  +entry = entry_of(query,entries,columns)
  Equal.sym(Nat,entry,Nat.add(Nat.mul(Nat.div(entry,rows),rows),Nat.mod(entry,rows)),DivisionOrder.quotient_exact(entry,rows,rows_positive))

# The tile the pack writes at a number of a unit: its kernel column's tile of
# its entry.
def pack_value(+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+query: Nat) -> Tile.Tile8:
  +entries = Model.pass_entries(window)
  +columns = tap_columns_of(window)
  +owner = unit_of(query,entries,columns)
  column_tile(pairs,tiles,U32.from_nat(column_index(query,columns)),window,planes,position_of(start,owner),U32.from_nat(owner),
    spot_of(query,entries,columns,tap_rows_of(window)))

def pack_at(inside: Bool,+window: Packing.Stencil,+planes: Packing.Planes,+start: U32,+units: Nat,+depth: Nat,+room: PackRoom<window,units,depth>,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),+query: Nat,
  +query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +exact: {Nat.is_lt(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),units) == inside : Bool}) ->
  {Tree.read(Tile.Tile8,units_tiles(units,0n,window,planes,start,pairs,tiles),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile8,inside,pack_value(pairs,tiles,window,planes,start,query),Tree.read(Tile.Tile8,tiles,U32.from_nat(query))) : Tile.Tile8}:
  match inside:
    case True{}:
      +entries = Model.pass_entries(window)
      +columns = tap_columns_of(window)
      +rows = tap_rows_of(window)
      +owner = unit_of(query,entries,columns)
      +column = column_index(query,columns)
      +entry = entry_of(query,entries,columns)
      +target = spot_of(query,entries,columns,rows)
      +columns_positive = pack_columns(window,units,depth,room)
      +entries_positive = pack_entries(window,units,depth,room)
      +rows_positive = pack_taps(window,units,depth,room)
      +element_eq = spot_entry(query,entries,columns,rows,rows_positive)
      +number_eq = Equal.trans(Nat,number(owner,column,Planes.element(target,rows),entries,columns),number(owner,column,entry,entries,columns),query,
        Equal.cong(Nat,Nat,value => number(owner,column,value,entries,columns),Planes.element(target,rows),entry,element_eq),
        number_of(query,entries,columns,entries_positive,columns_positive))
      %number_eq : {Tree.read(Tile.Tile8,units_tiles(units,0n,window,planes,start,pairs,tiles),U32.from_nat(_)) == pack_value(pairs,tiles,window,planes,start,query) : Tile.Tile8}
      units_hit(units,Nat.is_eq(0n,owner),0n,owner,column,window,planes,start,units,depth,room,pairs,tiles,target,balanced,
        DivisionOrder.remainder_small(query,columns,columns_positive),DivisionOrder.remainder_small(entry,rows,rows_positive),
        %Equal.sym(Nat,Planes.element(target,rows),entry,element_eq) : {Nat.is_lt(_,entries) == True{} : Bool}
          DivisionOrder.remainder_small(Nat.div(query,columns),entries,entries_positive),
        {==},Order.zero_le(owner),exact,Order.reflexive(units))
    case False{}:
      units_before(units,0n,window,planes,start,units,depth,room,pairs,tiles,query,balanced,query_inside,Order.reflexive(units),
        Order.not_less(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),units,exact))

# The pack reads back pointwise: a number of a unit holds its kernel column's
# tile of its entry, every other number its old tile.
def pack_read(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+units: Nat,
  +depth: Nat,+room: PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},+balanced: Shape.Balanced(Tile.Tile8,depth,tiles),
  +query: Nat,+query_inside: {Nat.is_lt(query,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Model.packed_tiles(window,planes,count,start,pairs,tiles),U32.from_nat(query)) ==
    Bool.pick(Tile.Tile8,Nat.is_lt(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),units),pack_value(pairs,tiles,window,planes,start,query),
      Tree.read(Tile.Tile8,tiles,U32.from_nat(query))) : Tile.Tile8}:
  +packed = Model.packed_tiles(window,planes,count,start,pairs,tiles)
  +laid = units_tiles(units,0n,window,planes,start,pairs,tiles)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,packed,U32.from_nat(query)),Tree.read(Tile.Tile8,laid,U32.from_nat(query)),
    Bool.pick(Tile.Tile8,Nat.is_lt(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),units),pack_value(pairs,tiles,window,planes,start,query),
      Tree.read(Tile.Tile8,tiles,U32.from_nat(query))),
    Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,U32.from_nat(query)),packed,laid,
      Equal.trans(Tree.Tree<Tile.Tile8>,packed,units_tiles(Model.unit_count(window,count),0n,window,planes,start,pairs,tiles),laid,
        packed_form(window,planes,count,start,pairs,tiles),
        Equal.cong(Nat,Tree.Tree<Tile.Tile8>,total => units_tiles(total,0n,window,planes,start,pairs,tiles),Model.unit_count(window,count),units,units_exact))),
    pack_at(Nat.is_lt(unit_of(query,Model.pass_entries(window),tap_columns_of(window)),units),window,planes,start,units,depth,room,pairs,tiles,balanced,query,query_inside,
      {==}))

# The pack keeps the tiles a tree of the same depth.
def pack_balanced(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+units: Nat,
  +depth: Nat,+room: PackRoom<window,units,depth>,+units_exact: {Model.unit_count(window,count) == units : Nat},+balanced: Shape.Balanced(Tile.Tile8,depth,tiles)) ->
  Shape.Balanced(Tile.Tile8,depth,Model.packed_tiles(window,planes,count,start,pairs,tiles)):
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.packed_tiles(window,planes,count,start,pairs,tiles),units_tiles(units,0n,window,planes,start,pairs,tiles),
    Equal.trans(Tree.Tree<Tile.Tile8>,Model.packed_tiles(window,planes,count,start,pairs,tiles),units_tiles(Model.unit_count(window,count),0n,window,planes,start,pairs,tiles),
      units_tiles(units,0n,window,planes,start,pairs,tiles),packed_form(window,planes,count,start,pairs,tiles),
      Equal.cong(Nat,Tree.Tree<Tile.Tile8>,total => units_tiles(total,0n,window,planes,start,pairs,tiles),Model.unit_count(window,count),units,units_exact))) :
    Shape.Balanced(Tile.Tile8,depth,_)
  units_balanced(units,0n,window,planes,start,units,depth,room,pairs,tiles,balanced,Order.reflexive(units))
