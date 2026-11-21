# Cells of values [n,A,B,1] with their lanes on A and cells of values [n,B,A]
# with their lanes on A, the groups innermost, are the same cells: lane l of
# group g at outer index o holds the value at (o / B, l + 8 g, o % B) of the
# first and at (o / B, o % B, l + 8 g) of the second. A reader of one holds the
# cells of a reader of the other that agrees at the exchanged indices.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding

# ---- The exchanged index ----

# The index of [n,B,A] at the index of [n,A,B] with lanes A and rows B.
def swapped(+index: Nat,+lanes: Nat,+rows: Nat) -> Nat:
  +line = Nat.div(index,rows)
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(line,lanes),rows),Nat.mod(index,rows)),lanes),Nat.mod(line,lanes))

def swap_exact(+image: Nat,+lane: Nat,+row: Nat,+lanes: Nat,+rows: Nat,+lanes_positive: {Nat.is_lt(0n,lanes) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,lanes) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {swapped(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row),lanes,rows) == Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),lanes),lane) : Nat}:
  +line = Nat.add(Nat.mul(image,lanes),lane)
  +index = Nat.add(Nat.mul(line,rows),row)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),lanes),lane)
  %Equal.sym(Nat & Nat,Nat.divmod(index,rows),(line,row),Division.quotient_remainder(line,row,rows,rows_positive,row_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Pair.fst(Nat,Nat,_),lanes),rows),Pair.snd(Nat,Nat,_)),lanes),Nat.mod(Pair.fst(Nat,Nat,_),lanes)) == target : Nat}
  %Equal.sym(Nat & Nat,Nat.divmod(line,lanes),(image,lane),Division.quotient_remainder(image,lane,lanes,lanes_positive,lane_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),rows),row),lanes),Pair.snd(Nat,Nat,_)) == target : Nat}
  {==}

# ---- The two geometries ----

def middle_geometry(+count: U32,+lanes: U32,+rows: U32) -> Cells.Cells:
  Public.geometry_of([count,lanes,rows,1],1n,Grid.Innermost{})

def last_geometry(+count: U32,+lanes: U32,+rows: U32) -> Cells.Cells:
  Public.geometry_of([count,rows,lanes],2n,Grid.Innermost{})

# The index of an outer index and a lane in the middle geometry.
def middle_index(+count: U32,+lanes: U32,+rows: U32,+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+outer: Nat,+channel: Nat) ->
  {Representation.logical_index(middle_geometry(count,lanes,rows),outer,channel) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,U32.to_nat(rows)),U32.to_nat(lanes)),channel),U32.to_nat(rows)),Nat.mod(outer,U32.to_nat(rows))) : Nat}:
  +after = Representation.least_one(U32.to_nat((rows * 1 : U32)))
  +rows_count = U32.to_nat(rows)
  +same = {Equal.trans(Nat,after,Representation.least_one(rows_count),rows_count,
      Equal.cong(U32,Nat,word => Representation.least_one(U32.to_nat(word)),(rows * 1 : U32),rows,Arithmetic.multiply_one(rows)),
      Representation.least_positive(rows_count,rows_positive)) : {after == rows_count : Nat}}
  %Equal.sym(Nat,after,rows_count,same) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,_),U32.to_nat(lanes)),channel),_),Nat.mod(outer,_)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,rows_count),U32.to_nat(lanes)),channel),rows_count),Nat.mod(outer,rows_count)) : Nat}
  {==}

def by_one(+value: Nat) -> {Nat.divmod(value,1n) == (value,0n) : Nat & Nat}:
  Division.unique(value,1n,value,0n,
    Equal.sym(Nat,Nat.add(Nat.mul(value,1n),0n),value,
      Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,Addition.zero_right(Nat.mul(value,1n)),Order.times_one(value))),{==},{==})

# The index of an outer index and a lane in the last geometry.
def last_index(+count: U32,+lanes: U32,+rows: U32,+outer: Nat,+channel: Nat) ->
  {Representation.logical_index(last_geometry(count,lanes,rows),outer,channel) == Nat.add(Nat.mul(outer,U32.to_nat(lanes)),channel) : Nat}:
  +line = Nat.add(Nat.mul(outer,U32.to_nat(lanes)),channel)
  %Equal.sym(Nat & Nat,Nat.divmod(outer,1n),(outer,0n),by_one(outer)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),U32.to_nat(lanes)),channel),1n),Pair.snd(Nat,Nat,_)) == line : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(line,1n),0n),Nat.mul(line,1n),line,Addition.zero_right(Nat.mul(line,1n)),Order.times_one(line))

# ---- Readers that agree at the exchanged indices ----

def Agreement(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,
  ~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data) -> Type:
  @+middle: Middle -> @+last: Last -> @+count: U32 -> @+lanes: U32 -> @+rows: U32 -> @+related: Related(middle,last,count,lanes,rows) ->
    @+image: Nat -> @+lane: Nat -> @+row: Nat -> @+image_inside: {Nat.is_lt(image,U32.to_nat(count)) == True{} : Bool} ->
    @+lane_inside: {Nat.is_lt(lane,U32.to_nat(lanes)) == True{} : Bool} -> @+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool} ->
    {middle_value(middle,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),lane),U32.to_nat(rows)),row))) ==
      last_value(last,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(lanes)),lane))) : F32}

def lane_read(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data,
  ~agreement: Agreement(~Middle,~middle_value,~Last,~last_value,~Related),+middle: Middle,+last: Last,
  +count: U32,+lanes: U32,+rows: U32,+related: Related(middle,last,count,lanes,rows),
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+outer: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool},+channel: Nat,
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool}) ->
  {middle_value(middle,U32.from_nat(Representation.logical_index(middle_geometry(count,lanes,rows),outer,channel))) ==
    last_value(last,U32.from_nat(Representation.logical_index(last_geometry(count,lanes,rows),outer,channel))) : F32}:
  +rows_count = U32.to_nat(rows)
  +lanes_count = U32.to_nat(lanes)
  +image = Nat.div(outer,rows_count)
  +row = Nat.mod(outer,rows_count)
  %Equal.sym(Nat,Representation.logical_index(middle_geometry(count,lanes,rows),outer,channel),
      Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes_count),channel),rows_count),row),middle_index(count,lanes,rows,rows_positive,outer,channel)) :
    {middle_value(middle,U32.from_nat(_)) == last_value(last,U32.from_nat(Representation.logical_index(last_geometry(count,lanes,rows),outer,channel))) : F32}
  %Equal.sym(Nat,Representation.logical_index(last_geometry(count,lanes,rows),outer,channel),Nat.add(Nat.mul(outer,lanes_count),channel),last_index(count,lanes,rows,outer,channel)) :
    {middle_value(middle,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes_count),channel),rows_count),row))) ==
      last_value(last,U32.from_nat(_)) : F32}
  %Equal.sym(Nat,outer,Nat.add(Nat.mul(image,rows_count),row),DivisionOrder.quotient_exact(outer,rows_count,rows_positive)) :
    {middle_value(middle,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes_count),channel),rows_count),row))) ==
      last_value(last,U32.from_nat(Nat.add(Nat.mul(_,lanes_count),channel))) : F32}
  agreement(middle,last,count,lanes,rows,related,image,channel,row,DivisionOrder.quotient_below(outer,U32.to_nat(count),rows_count,rows_positive,outer_inside),channel_inside,
    DivisionOrder.remainder_small(outer,rows_count,rows_positive))

def lane_cases(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data,
  ~agreement: Agreement(~Middle,~middle_value,~Last,~last_value,~Related),inside: Bool,+middle: Middle,+last: Last,
  +count: U32,+lanes: U32,+rows: U32,+related: Related(middle,last,count,lanes,rows),
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+outer: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool},+channel: Nat,
  +classified: {Nat.is_lt(channel,U32.to_nat(lanes)) == inside : Bool}) ->
  {Bool.pick(F32,inside,middle_value(middle,U32.from_nat(Representation.logical_index(middle_geometry(count,lanes,rows),outer,channel))),0.0) ==
    Bool.pick(F32,inside,last_value(last,U32.from_nat(Representation.logical_index(last_geometry(count,lanes,rows),outer,channel))),0.0) : F32}:
  match inside:
    case True{}: lane_read(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,channel,classified)
    case False{}: {==}

def lane_agrees(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data,
  ~agreement: Agreement(~Middle,~middle_value,~Last,~last_value,~Related),+middle: Middle,+last: Last,
  +count: U32,+lanes: U32,+rows: U32,+related: Related(middle,last,count,lanes,rows),
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+outer: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool},+group: Nat,+lane: Nat) ->
  {Representation.lane(~Middle,~middle_value,middle,middle_geometry(count,lanes,rows),outer,group,lane) ==
    Representation.lane(~Last,~last_value,last,last_geometry(count,lanes,rows),outer,group,lane) : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  lane_cases(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,Nat.is_lt(channel,U32.to_nat(lanes)),middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,channel,{==})

def cell_agrees(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data,
  ~agreement: Agreement(~Middle,~middle_value,~Last,~last_value,~Related),+middle: Middle,+last: Last,
  +count: U32,+lanes: U32,+rows: U32,+related: Related(middle,last,count,lanes,rows),
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+outer: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool},+group: Nat) ->
  {Representation.cell(~Middle,~middle_value,middle,middle_geometry(count,lanes,rows),outer,group) ==
    Representation.cell(~Last,~last_value,last,last_geometry(count,lanes,rows),outer,group) : Tile.Tile8}:
  TileLaws.tiles_equal(Representation.cell(~Middle,~middle_value,middle,middle_geometry(count,lanes,rows),outer,group),
    Representation.cell(~Last,~last_value,last,last_geometry(count,lanes,rows),outer,group),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,0n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,1n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,2n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,3n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,4n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,5n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,6n),
    lane_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,outer,outer_inside,group,7n))

# The cell count of both geometries.
def cell_count(+count: U32,+lanes: U32,+rows: U32) -> Nat:
  Nat.mul(Nat.mul(U32.to_nat(count),U32.to_nat(rows)),U32.to_nat(Tiled.groups(lanes)))

# Below the cell count the lifted readers agree.
def lifted_agrees(~Middle: Data,~middle_value: Middle -> U32 -> F32,~Last: Data,~last_value: Last -> U32 -> F32,~Related: Middle -> Last -> U32 -> U32 -> U32 -> Data,
  ~agreement: Agreement(~Middle,~middle_value,~Last,~last_value,~Related),+middle: Middle,+last: Last,
  +count: U32,+lanes: U32,+rows: U32,+related: Related(middle,last,count,lanes,rows),
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),cell_count(count,lanes,rows)) == True{} : Bool}) ->
  {Representation.lifted(~Middle,~middle_value,Representation.Lifted{middle,middle_geometry(count,lanes,rows)},index) ==
    Representation.lifted(~Last,~last_value,Representation.Lifted{last,last_geometry(count,lanes,rows)},index) : Tile.Tile8}:
  +groups = U32.to_nat(Tiled.groups(lanes))
  +number = U32.to_nat(index)
  +least = Representation.least_positive(groups,groups_positive)
  +outer_inside = {%Equal.sym(Nat,Representation.least_one(groups),groups,least) : {Nat.is_lt(Nat.div(number,_),Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool}
    DivisionOrder.quotient_below(number,Nat.mul(U32.to_nat(count),U32.to_nat(rows)),groups,groups_positive,inside) :
    {Nat.is_lt(Nat.div(number,Representation.least_one(groups)),Nat.mul(U32.to_nat(count),U32.to_nat(rows))) == True{} : Bool}}
  cell_agrees(~Middle,~middle_value,~Last,~last_value,~Related,~agreement,middle,last,count,lanes,rows,related,rows_positive,Nat.div(number,Representation.least_one(groups)),outer_inside,
    Nat.mod(number,Representation.least_one(groups)))

# ---- The exchanged reader ----

# A reader read at the exchanged index.
type Swapped<-Logical: Data> is Data:
  Swapped{logical: Logical,lanes: Nat,rows: Nat}

def swapped_value(~Logical: Data,~original: Logical -> U32 -> F32,context: Swapped<Logical>,index: U32) -> F32:
  Swapped{+logical,+lanes,+rows} = context
  original(logical,U32.from_nat(swapped(U32.to_nat(index),lanes,rows)))

# The values of [n,A,B] within a capacity.
def index_room(+image: Nat,+lane: Nat,+row: Nat,+count: Nat,+lanes: Nat,+rows: Nat,+cap: U32,
  +image_inside: {Nat.is_lt(image,count) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,lanes) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.mul(count,lanes),rows),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row),U32.to_nat(cap)) == True{} : Bool}:
  Bounds.strict_le(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row),U32.to_nat(cap),
    Bounds.nat_trans(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row),Nat.mul(Nat.mul(count,lanes),rows),U32.to_nat(cap),
      RowMajor.radix(Nat.add(Nat.mul(image,lanes),lane),Nat.mul(count,lanes),row,rows,RowMajor.radix(image,count,lane,lanes,image_inside,lane_inside),row_inside),room))

# The exchanged reader of a reader in the last geometry, in the middle.
type Into<-Logical: Data,-middle: Swapped<Logical>,-last: Logical,-count: U32,-lanes: U32,-rows: U32> is Data:
  Into{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(count),U32.to_nat(lanes)),U32.to_nat(rows)),U32.to_nat(cap)) == True{} : Bool},
    lanes_positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    same: {middle == Swapped{last,U32.to_nat(lanes),U32.to_nat(rows)} : Swapped<Logical>}}

def into_values(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+count: Nat,+lanes: Nat,+rows: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(count,lanes),rows),U32.to_nat(cap)) == True{} : Bool},
  +lanes_positive: {Nat.is_lt(0n,lanes) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+image: Nat,+lane: Nat,+row: Nat,
  +image_inside: {Nat.is_lt(image,count) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,lanes) == True{} : Bool},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {swapped_value(~Logical,~original,Swapped{logical,lanes,rows},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row))) ==
    original(logical,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),lanes),lane))) : F32}:
  +index = Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),lane),rows),row)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),lanes),lane)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,index_room(image,lane,row,count,lanes,rows,cap,image_inside,lane_inside,row_inside,room))) :
    {original(logical,U32.from_nat(swapped(_,lanes,rows))) == original(logical,U32.from_nat(target)) : F32}
  %Equal.sym(Nat,swapped(index,lanes,rows),target,swap_exact(image,lane,row,lanes,rows,lanes_positive,rows_positive,lane_inside,row_inside)) :
    {original(logical,U32.from_nat(_)) == original(logical,U32.from_nat(target)) : F32}
  {==}

def into(~Logical: Data,~original: Logical -> U32 -> F32,+middle: Swapped<Logical>,+last: Logical,+count: U32,+lanes: U32,+rows: U32,
  +related: Into<Logical,middle,last,count,lanes,rows>,+image: Nat,+lane: Nat,+row: Nat,+image_inside: {Nat.is_lt(image,U32.to_nat(count)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,U32.to_nat(lanes)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool}) ->
  {swapped_value(~Logical,~original,middle,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),lane),U32.to_nat(rows)),row))) ==
    original(last,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(lanes)),lane))) : F32}:
  Into{+cap,+cap_ok,+room,+lanes_positive,+rows_positive,+same} = related
  %Equal.sym(Swapped<Logical>,middle,Swapped{last,U32.to_nat(lanes),U32.to_nat(rows)},same) :
    {swapped_value(~Logical,~original,_,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),lane),U32.to_nat(rows)),row))) ==
      original(last,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(rows)),row),U32.to_nat(lanes)),lane))) : F32}
  into_values(~Logical,~original,last,U32.to_nat(count),U32.to_nat(lanes),U32.to_nat(rows),cap,cap_ok,room,lanes_positive,rows_positive,image,lane,row,
    image_inside,lane_inside,row_inside)
