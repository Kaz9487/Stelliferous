# What the cell conversions leave. The tiled run writes at every cell the tile
# of its lane reads and the untiled run writes at every value the lane of its
# cell. Within the machine their block runs write what one flat pass of the
# reads writes, and for one image with its groups innermost the cell of pixel
# o and group g holds the values of channels 8 * g to 8 * g + 7 at o, the
# value of channel c at o is lane c % 8 of the cell of group c / 8 at o, and
# the values of the cells of the values are the values.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_blocks.bend as Blocks
import ../tensor_grid.bend as Grid
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_blocks_refinement.bend as BlocksLaws
import ./traversal_indexed_values.bend as IndexedValues
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_cursor_refinement.bend as Cursor
import ./encoded_coordinates.bend as Encoded
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as Ceiling
import ./u32_division.bend as NativeDivision
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./kernel_tile_refinement.bend as TileLaws
import ./binary_radix.bend as Radix
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_tiled.bend as Tiled

# ---- Groups ----

# The groups of a channel count within the machine capacity: the ceiling of
# eighths, no more than the channels, and their lanes within half the word.
type Groups<-limits: Machine.Limits,-channels: U32> is Data:
  Groups{exact: {U32.to_nat(Grid.groups(channels)) == Ceiling.ceiling(U32.to_nat(channels),8n) : Nat},
    fewer: {Nat.is_le(U32.to_nat(Grid.groups(channels)),U32.to_nat(channels)) == True{} : Bool},
    lanes: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(channels)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

def grouped(+limits: Machine.Limits,+channels: U32,+bound: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Groups<limits,channels>:
  +count = U32.to_nat(channels)
  +exact = Encoded.eight_blocks_exact(limits,channels,bound)
  Groups{exact,
    %Equal.sym(Nat,U32.to_nat(Grid.groups(channels)),Ceiling.ceiling(count,8n),exact) : {Nat.is_le(_,count) == True{} : Bool}
    Ceiling.bounded(count,8n,{==}),
    %Equal.sym(Nat,U32.to_nat(Grid.groups(channels)),Ceiling.ceiling(count,8n),exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    %Ceiling.shifted_formula(count,8n,{==}) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Order.transitive(Nat.mul(Nat.div(Nat.add(count,7n),8n),8n),Nat.add(count,7n),U32.to_nat(Machine.half_word(limits)),
      DivisionOrder.product_below(Nat.add(count,7n),8n,{==}),Encoded.add_seven_bound(limits,count,bound))}

# Every lane of a group lies within half the word.
def channel_bound(+limits: Machine.Limits,+channels: U32,+group: Nat,+lane: Nat,+facts: Groups<limits,channels>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(channels))) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Groups{exact,fewer,lanes} = facts
  +groups = U32.to_nat(Grid.groups(channels))
  Bounds.strict_le(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Machine.half_word(limits)),
    Bounds.nat_trans(Nat.add(lane,Nat.mul(group,8n)),Nat.mul(groups,8n),U32.to_nat(Machine.half_word(limits)),
      %Order.commutative(Nat.mul(group,8n),lane) : {Nat.is_lt(_,Nat.mul(groups,8n)) == True{} : Bool}
      RowMajor.radix(group,groups,lane,8n,group_inside,lane_inside),lanes))

def fewer_of(+limits: Machine.Limits,+channels: U32,facts: Groups<limits,channels>) -> {Nat.is_le(U32.to_nat(Grid.groups(channels)),U32.to_nat(channels)) == True{} : Bool}:
  Groups{exact,fewer,lanes} = facts
  fewer

# The group of a channel lies below the groups.
def group_inside(+limits: Machine.Limits,+channels: U32,+channel: Nat,+facts: Groups<limits,channels>,
  +inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool}) -> {Nat.is_lt(Nat.div(channel,8n),U32.to_nat(Grid.groups(channels))) == True{} : Bool}:
  Groups{exact,fewer,lanes} = facts
  +count = U32.to_nat(channels)
  %Equal.sym(Nat,U32.to_nat(Grid.groups(channels)),Ceiling.ceiling(count,8n),exact) : {Nat.is_lt(Nat.div(channel,8n),_) == True{} : Bool}
  RowMajor.cancel_multiply_lt(Nat.div(channel,8n),Ceiling.ceiling(count,8n),8n,
    Order.le_lt_trans(Nat.mul(Nat.div(channel,8n),8n),channel,Nat.mul(Ceiling.ceiling(count,8n),8n),DivisionOrder.product_below(channel,8n,{==}),
      Bounds.nat_trans(channel,count,Nat.mul(Ceiling.ceiling(count,8n),8n),inside,Ceiling.covers(count,8n,{==}))))

def group_positive(+limits: Machine.Limits,+channels: U32,+facts: Groups<limits,channels>,
  +positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Grid.groups(channels))) == True{} : Bool}:
  group_inside(limits,channels,0n,facts,positive)

# A channel is its lane past eight times its group.
def channel_split(+channel: Nat) -> {Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)) == channel : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)),Nat.add(Nat.mul(Nat.div(channel,8n),8n),Nat.mod(channel,8n)),channel,
    Order.commutative(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)),
    Equal.sym(Nat,channel,Nat.add(Nat.mul(Nat.div(channel,8n),8n),Nat.mod(channel,8n)),DivisionOrder.quotient_exact(channel,8n,{==})))

# A channel below half the machine word compares as its number.
def channel_compare(+limits: Machine.Limits,+channel: Nat,+lanes: U32,bound: {Nat.is_le(channel,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(channel),lanes) == Nat.is_lt(channel,U32.to_nat(lanes)) : Bool}:
  Equal.trans(Bool,U32.is_lt(U32.from_nat(channel),lanes),Nat.is_lt(U32.to_nat(U32.from_nat(channel)),U32.to_nat(lanes)),Nat.is_lt(channel,U32.to_nat(lanes)),
    Comparison.lt(U32.from_nat(channel),lanes),
    Equal.cong(Nat,Bool,value => Nat.is_lt(value,U32.to_nat(lanes)),U32.to_nat(U32.from_nat(channel)),channel,
      Bounds.roundtrip(channel,Machine.half_word(limits),Limits.half_word_fits(limits),bound)))

# A lane past the true channel count reads +0 on both sides; a lane inside
# reads the same value.
def picked(+valid: Bool,+read: F32,+value: F32,equal: @valid_true: {valid == True{} : Bool} -> {read == value : F32}) ->
  {Bool.pick(F32,valid,read,0.0) == Bool.pick(F32,valid,value,0.0) : F32}:
  match valid:
    case True{}: equal({==})
    case False{}: {==}

def tiles_equal(+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,+h: F32,+a2: F32,+b2: F32,+c2: F32,+d2: F32,+e2: F32,+f2: F32,+g2: F32,+h2: F32,
  ea: {a == a2 : F32},eb: {b == b2 : F32},ec: {c == c2 : F32},ed: {d == d2 : F32},ee: {e == e2 : F32},ef: {f == f2 : F32},eg: {g == g2 : F32},eh: {h == h2 : F32}) ->
  {Tile.Tile{a,b,c,d,e,f,g,h} == Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2} : Tile.Tile8}:
  +goal = {Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2} : Tile.Tile8}
  %Equal.sym(F32,a,a2,ea) : {Tile.Tile{_,b,c,d,e,f,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,b,b2,eb) : {Tile.Tile{a2,_,c,d,e,f,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,c,c2,ec) : {Tile.Tile{a2,b2,_,d,e,f,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,d,d2,ed) : {Tile.Tile{a2,b2,c2,_,e,f,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,e,e2,ee) : {Tile.Tile{a2,b2,c2,d2,_,f,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,f,f2,ef) : {Tile.Tile{a2,b2,c2,d2,e2,_,g,h} == goal : Tile.Tile8}
  %Equal.sym(F32,g,g2,eg) : {Tile.Tile{a2,b2,c2,d2,e2,f2,_,h} == goal : Tile.Tile8}
  %Equal.sym(F32,h,h2,eh) : {Tile.Tile{a2,b2,c2,d2,e2,f2,g2,_} == goal : Tile.Tile8}
  {==}

# The lane read of a tile is its selected lane.
def lane_selected(+tile: Tile.Tile8,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Cells.lane_of(tile,U32.from_nat(lane)) == TileLaws.select_lane(lane,tile) : F32}:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  match lane:
    case 0n: {==}
    case 1n+ +l1:
      match l1:
        case 0n: {==}
        case 1n+ +l2:
          match l2:
            case 0n: {==}
            case 1n+ +l3:
              match l3:
                case 0n: {==}
                case 1n+ +l4:
                  match l4:
                    case 0n: {==}
                    case 1n+ +l5:
                      match l5:
                        case 0n: {==}
                        case 1n+ +l6:
                          match l6:
                            case 0n: {==}
                            case 1n+ +l7:
                              match l7:
                                case 0n: {==}
                                case 1n+ +rest:
                                  Empty.absurd({Cells.lane_of(Tile.Tile{a,b,c,d,e,f,g,h},U32.from_nat(8n+rest)) == TileLaws.select_lane(8n+rest,Tile.Tile{a,b,c,d,e,f,g,h}) : F32},
                                    WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def swap_last(+a: Nat,+b: Nat,+c: Nat) -> {Nat.add(Nat.add(a,b),c) == Nat.add(Nat.add(c,a),b) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(c,a),b),Addition.associative(a,b,c),
    Equal.trans(Nat,Nat.add(a,Nat.add(b,c)),Nat.add(a,Nat.add(c,b)),Nat.add(Nat.add(c,a),b),
      Equal.cong(Nat,Nat,value => Nat.add(a,value),Nat.add(b,c),Nat.add(c,b),Order.commutative(b,c)),
      Equal.trans(Nat,Nat.add(a,Nat.add(c,b)),Nat.add(Nat.add(a,c),b),Nat.add(Nat.add(c,a),b),Equal.sym(Nat,Nat.add(Nat.add(a,c),b),Nat.add(a,Nat.add(c,b)),Addition.associative(a,c,b)),
        Equal.cong(Nat,Nat,value => Nat.add(value,b),Nat.add(a,c),Nat.add(c,a),Order.commutative(a,c)))))

# ---- Positions in blocks ----

# A position of channel block at offset below the size size decodes to that
# channel and offset, when the count is within the machine capacity.
type Decoded<-size: U32,-block: Nat,-offset: Nat> is Data:
  Decoded{quotient: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) / size : U32) == U32.from_nat(block) : U32},
    remainder: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) % size : U32) == U32.from_nat(offset) : U32},
    local: {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) - U32.from_nat(Nat.mul(block,U32.to_nat(size))) : U32) == U32.from_nat(offset) : U32}}

def decoded(+limits: Machine.Limits,+total: U32,+size: U32,+block: Nat,+offset: Nat,
  +inside: {Nat.is_lt(offset,U32.to_nat(size)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size)),U32.to_nat(size)),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) -> Decoded<size,block,offset>:
  +positive = Order.le_lt_trans(0n,offset,U32.to_nat(size),Order.zero_le(offset),inside)
  +within = Bounds.as_nat_le(total,Machine.capacity(limits),bounded)
  +below = Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(size)),offset),U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(size)),offset),Nat.add(Nat.mul(block,U32.to_nat(size)),U32.to_nat(size)),U32.to_nat(total),Order.offset_lt(Nat.mul(block,U32.to_nat(size)),offset,U32.to_nat(size),inside),room),within)
  +exact = Bounds.roundtrip(Nat.add(Nat.mul(block,U32.to_nat(size)),offset),Machine.half_word(limits),Limits.half_word_fits(limits),
    Bounds.strict_le(Nat.add(Nat.mul(block,U32.to_nat(size)),offset),U32.to_nat(Machine.half_word(limits)),Bounds.nat_trans(Nat.add(Nat.mul(block,U32.to_nat(size)),offset),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),below,
      Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))))
  +divisor = Limits.double_bound(limits,U32.to_nat(size),Order.transitive(U32.to_nat(size),U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),Order.drop_left_le(Nat.mul(block,U32.to_nat(size)),U32.to_nat(size),U32.to_nat(total),room),within))
  +pair = Division.unique(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset))),U32.to_nat(size),block,offset,exact,positive,inside)
  Decoded{
    Encoded.encoding((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) / size : U32),block,
      Equal.trans(Nat,U32.to_nat((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) / size : U32)),Nat.div(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset))),U32.to_nat(size)),block,
        NativeDivision.divide(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)),size,Machine.half_word(limits),Limits.half_word_fits(limits),divisor,positive),
        Equal.cong(Nat & Nat,Nat,result => Pair.fst(Nat,Nat,result),Nat.divmod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset))),U32.to_nat(size)),(block,offset),pair))),
    Encoded.encoding((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) % size : U32),offset,
      Equal.trans(Nat,U32.to_nat((U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) % size : U32)),Nat.mod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset))),U32.to_nat(size)),offset,
        NativeDivision.modulo(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)),size,Machine.half_word(limits),Limits.half_word_fits(limits),divisor,positive),
        Equal.cong(Nat & Nat,Nat,result => Pair.snd(Nat,Nat,result),Nat.divmod(U32.to_nat(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset))),U32.to_nat(size)),(block,offset),pair))),
    Cursor.subtract_front(Nat.mul(block,U32.to_nat(size)),offset)}

# A flat pass writes where it reads.
def in_place_identity(~Inner: Data,+context: Inner,+position: U32) -> {Blocks.in_place(~Inner,context,position) == position : U32}:
  {==}

def fresh_sized(~Element: Data,+total: U32,+fill: Element,ready: Depth.Sized<total>) ->
  Buffer.Sized<Element,Buffer.fresh(~Element,Storage.depth_for(total),fill,U32.to_nat(total))>:
  Depth.Sized{+depth_ok,+room} = ready
  Buffer.fresh_ready(~Element,Storage.depth_for(total),fill,U32.to_nat(total),depth_ok,room)

# A flat pass over a fresh tree stores its sample at every position it covers.
def model_stored(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+total: U32,+fill: Element,+context: Context,+reader: Reader,
  +position: Nat,ready: Buffer.Sized<Element,Buffer.fresh(~Element,Storage.depth_for(total),fill,U32.to_nat(total))>,
  +selected: {Nat.is_lt(position,U32.to_nat(total)) == True{} : Bool}) ->
  {Tree.read(Element,IndexedValues.run(~Reader,~Element,~Context,~sample,~Blocks.in_place(~Context),U32.to_nat(total),0,context,reader,Tree.fresh(Element,Storage.depth_for(total),fill)),
      U32.from_nat(position)) == sample(reader,context,U32.from_nat(position)) : Element}:
  Buffer.Sized{+depth_ok,+room} = ready
  +model = Buffer.fresh(~Element,Storage.depth_for(total),fill,U32.to_nat(total))
  IndexedValues.stored(~Reader,~Element,~Context,~sample,~Blocks.in_place(~Context),~in_place_identity(~Context),U32.to_nat(total),context,reader,Buffer.tree(~Element,model),
    Buffer.depth(~Element,model),Tree.capacity(Element,Buffer.tree(~Element,model)),position,{==},Buffer.balanced(~Element,model),depth_ok,
    Buffer.capacity_bound(~Element,model,depth_ok),room,selected)

def fresh_stored(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+total: U32,+fill: Element,+context: Context,+reader: Reader,
  +position: Nat,+within: {U32.is_le(total,1073741824) == True{} : Bool},+selected: {Nat.is_lt(position,U32.to_nat(total)) == True{} : Bool}) ->
  {Tree.read(Element,IndexedValues.run(~Reader,~Element,~Context,~sample,~Blocks.in_place(~Context),U32.to_nat(total),0,context,reader,Tree.fresh(Element,Storage.depth_for(total),fill)),
      U32.from_nat(position)) == sample(reader,context,U32.from_nat(position)) : Element}:
  model_stored(~Reader,~Element,~Context,~sample,total,fill,context,reader,position,fresh_sized(~Element,total,fill,Depth.bounded(total,within)),selected)

# A count within the machine can be allocated.
def allocatable(+limits: Machine.Limits,+total: U32,+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) -> {U32.is_le(total,1073741824) == True{} : Bool}:
  Encoded.native_le_transitive(total,Machine.capacity(limits),1073741824,bounded,Limits.capacity_allocatable(limits))

# A position inside rows of a size lies in their count.
def inside_rows(+row: Nat,+rows: Nat,+offset: Nat,+size: Nat,+total: Nat,+exact: {Nat.mul(rows,size) == total : Nat},
  +row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,size) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(row,size),offset),total) == True{} : Bool}:
  %exact : {Nat.is_lt(Nat.add(Nat.mul(row,size),offset),_) == True{} : Bool}
  RowMajor.radix(row,rows,offset,size,row_inside,offset_inside)

# The row of a position inside rows of a size ends within their count.
def row_room(+row: Nat,+rows: Nat,+size: Nat,+total: Nat,+exact: {Nat.mul(rows,size) == total : Nat},+row_inside: {Nat.is_lt(row,rows) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(row,size),size),total) == True{} : Bool}:
  %exact : {Nat.is_le(Nat.add(Nat.mul(row,size),size),_) == True{} : Bool}
  %Order.commutative(size,Nat.mul(row,size)) : {Nat.is_le(_,Nat.mul(rows,size)) == True{} : Bool}
  RowMajor.mul_le(1n+row,rows,size,Order.strict_follow(row,rows,row_inside))

# ---- To cells ----

# The spot count lanes after spot.
def steps(spot: Cells.Spot,count: Nat) -> Cells.Spot:
  match count:
    case 0n: spot
    case 1n+rest: Cells.next(steps(spot,rest))

def spot_value(tree: Tree.Tree<F32>,spot: Cells.Spot) -> F32:
  Cells.Spot{+channel,index,plane,+lanes} = spot
  Bool.pick(F32,U32.is_lt(channel,lanes),Tree.read(F32,tree,index),0.0)

# The tile of the eight lanes from spot on a tree.
def spot_cell(+tree: Tree.Tree<F32>,+spot: Cells.Spot) -> Tile.Tile8:
  Tile.Tile{spot_value(tree,steps(spot,0n)),spot_value(tree,steps(spot,1n)),spot_value(tree,steps(spot,2n)),spot_value(tree,steps(spot,3n)),
    spot_value(tree,steps(spot,4n)),spot_value(tree,steps(spot,5n)),spot_value(tree,steps(spot,6n)),spot_value(tree,steps(spot,7n))}

# The tile the cell read of a run gives at a position on a tree.
def run_value(tree: Tree.Tree<F32>,run: Cells.Run,position: U32) -> Tile.Tile8:
  spot_cell(tree,Cells.run_spot(run,position))

def spot_read(+tree: Tree.Tree<F32>,+spot: Cells.Spot) ->
  {Cells.read_spot(Tree.pack(F32,tree),spot) == (Tree.pack(F32,tree),spot_value(tree,spot)) : Array<F32> & F32}:
  Cells.Spot{+channel,+index,+plane,+lanes} = spot
  StorageLaws.guarded_read(~F32,0.0,U32.is_lt(channel,lanes),tree,index)

# The eight lanes read from a spot on a tree are the cell of the spot.
def spot_lanes(+tree: Tree.Tree<F32>,+spot: Cells.Spot) ->
  {Cells.lane1(spot,Cells.read_spot(Tree.pack(F32,tree),spot)) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}:
  +s1 = Cells.next(spot)
  +s2 = Cells.next(s1)
  +s3 = Cells.next(s2)
  +s4 = Cells.next(s3)
  +s5 = Cells.next(s4)
  +s6 = Cells.next(s5)
  +s7 = Cells.next(s6)
  +a = spot_value(tree,spot)
  +b = spot_value(tree,s1)
  +c = spot_value(tree,s2)
  +d = spot_value(tree,s3)
  +e = spot_value(tree,s4)
  +f = spot_value(tree,s5)
  +g = spot_value(tree,s6)
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),spot),(Tree.pack(F32,tree),a),spot_read(tree,spot)) : {Cells.lane1(spot,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s1),(Tree.pack(F32,tree),b),spot_read(tree,s1)) : {Cells.lane2(s1,a,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s2),(Tree.pack(F32,tree),c),spot_read(tree,s2)) : {Cells.lane3(s2,a,b,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s3),(Tree.pack(F32,tree),d),spot_read(tree,s3)) : {Cells.lane4(s3,a,b,c,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s4),(Tree.pack(F32,tree),e),spot_read(tree,s4)) : {Cells.lane5(s4,a,b,c,d,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s5),(Tree.pack(F32,tree),f),spot_read(tree,s5)) : {Cells.lane6(s5,a,b,c,d,e,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s6),(Tree.pack(F32,tree),g),spot_read(tree,s6)) : {Cells.lane7(s6,a,b,c,d,e,f,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Cells.read_spot(Tree.pack(F32,tree),s7),(Tree.pack(F32,tree),spot_value(tree,s7)),spot_read(tree,s7)) : {Cells.lane8(a,b,c,d,e,f,g,_) == (Tree.pack(F32,tree),spot_cell(tree,spot)) : Array<F32> & Tile.Tile8}
  {==}

def cell_read(+tree: Tree.Tree<F32>,+run: Cells.Run,+position: U32) ->
  {Cells.cell_read(Tree.pack(F32,tree),run,position) == (Tree.pack(F32,tree),run_value(tree,run,position)) : Array<F32> & Tile.Tile8}:
  spot_lanes(tree,Cells.run_spot(run,position))

# One more plane past rest planes from base.
def plane_step(+rest: Nat,+base: Nat,+plane: U32) ->
  {(U32.from_nat(Nat.add(Nat.mul(rest,U32.to_nat(plane)),base)) + plane : U32) == U32.from_nat(Nat.add(Nat.mul(1n+rest,U32.to_nat(plane)),base)) : U32}:
  +step = U32.to_nat(plane)
  +done = Nat.add(Nat.mul(rest,step),base)
  %Arithmetic.native_roundtrip(plane) : {(U32.from_nat(done) + _ : U32) == U32.from_nat(Nat.add(Nat.add(step,Nat.mul(rest,step)),base)) : U32}
  Equal.trans(U32,(U32.from_nat(done) + U32.from_nat(step) : U32),U32.from_nat(Nat.add(done,step)),U32.from_nat(Nat.add(Nat.add(step,Nat.mul(rest,step)),base)),
    WordArithmetic.native_add(done,step),Equal.cong(Nat,U32,U32.from_nat,Nat.add(done,step),Nat.add(Nat.add(step,Nat.mul(rest,step)),base),swap_last(Nat.mul(rest,step),base,step)))

# count steps from a spot add count to its channel and count planes to its index.
def stepped(+channel: Nat,+base: Nat,+plane: U32,+lanes: U32,count: Nat) ->
  {steps(Cells.Spot{U32.from_nat(channel),U32.from_nat(base),plane,lanes},count) ==
    Cells.Spot{U32.from_nat(Nat.add(count,channel)),U32.from_nat(Nat.add(Nat.mul(count,U32.to_nat(plane)),base)),plane,lanes} : Cells.Spot}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +step = U32.to_nat(plane)
      +done = Nat.add(Nat.mul(rest,step),base)
      +goal = {Cells.Spot{U32.from_nat(1n+Nat.add(rest,channel)),U32.from_nat(Nat.add(Nat.add(step,Nat.mul(rest,step)),base)),plane,lanes} : Cells.Spot}
      %Equal.sym(Cells.Spot,steps(Cells.Spot{U32.from_nat(channel),U32.from_nat(base),plane,lanes},rest),
        Cells.Spot{U32.from_nat(Nat.add(rest,channel)),U32.from_nat(done),plane,lanes},stepped(channel,base,plane,lanes,rest)) :
        {Cells.next(_) == goal : Cells.Spot}
      %Equal.sym(U32,(U32.from_nat(Nat.add(rest,channel)) + 1 : U32),U32.from_nat(1n+Nat.add(rest,channel)),Bounds.increment_encoded(Nat.add(rest,channel))) :
        {Cells.Spot{_,(U32.from_nat(done) + plane : U32),plane,lanes} == goal : Cells.Spot}
      %Equal.sym(U32,(U32.from_nat(done) + plane : U32),U32.from_nat(Nat.add(Nat.add(step,Nat.mul(rest,step)),base)),plane_step(rest,base,plane)) :
        {Cells.Spot{U32.from_nat(1n+Nat.add(rest,channel)),_,plane,lanes} == goal : Cells.Spot}
      {==}

def tiling_groups(context: Cells.Tiling) -> U32:
  Cells.Tiling{after,lanes,+groups,first} = context
  groups

# The cell at a position on a tree: the run of its outer index.
def tiled_cell(tree: Tree.Tree<F32>,context: Cells.Tiling,position: U32) -> Tile.Tile8:
  Cells.Tiling{after,lanes,+groups,first} = context
  +position = position
  +block = (position / groups : U32)
  run_value(tree,Cells.group_run(Cells.Tiling{after,lanes,groups,first},block,(block * groups : U32)),position)

# The model of the tiled run: blocks outer indices of groups cells.
def tiled_blocks(+blocks: U32,context: Cells.Tiling,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>:
  Cells.Tiling{after,lanes,+groups,first} = context
  BlocksLaws.blocked(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~Cells.group_run,~run_value,~Blocks.in_place(~Cells.Run),U32.to_nat(blocks),groups,
    Cells.Tiling{after,lanes,groups,first},input,output)

def tiled_run(+blocks: U32,+context: Cells.Tiling,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Cells.tiled(blocks,context,Tree.pack(F32,input),Tree.pack(Tile.Tile8,output)) ==
    AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),tiled_blocks(blocks,context,input,output)) : Array<F32> & Array<Tile.Tile8>}:
  Cells.Tiling{+after,+lanes,+groups,+first} = context
  BlocksLaws.represented(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~ArrayLaws.pack_tree(~F32),~Cells.group_run,~Cells.cell_read,~run_value,
    ~Blocks.in_place(~Cells.Run),~cell_read,U32.to_nat(blocks),groups,Cells.Tiling{after,lanes,groups,first},input,output)

# The run of a block reads at every position of it what the run of the block
# the position decodes to reads.
def run_decoded(+tree: Tree.Tree<F32>,+after: U32,+lanes: U32,+groups: U32,+first: U32,+block: Nat,+offset: Nat,facts: Decoded<groups,block,offset>) ->
  {run_value(tree,Cells.group_run(Cells.Tiling{after,lanes,groups,first},U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(groups)))),
      U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(groups)),offset))) ==
    tiled_cell(tree,Cells.Tiling{after,lanes,groups,first},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(groups)),offset))) : Tile.Tile8}:
  Decoded{quotient,remainder,local} = facts
  +context = {Cells.Tiling{after,lanes,groups,first} : Cells.Tiling}
  +position = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(groups)),offset))
  +index = U32.from_nat(block)
  %Equal.sym(U32,(position / groups : U32),index,quotient) :
    {run_value(tree,Cells.group_run(context,index,U32.from_nat(Nat.mul(block,U32.to_nat(groups)))),position) ==
      run_value(tree,Cells.group_run(context,_,(_ * groups : U32)),position) : Tile.Tile8}
  Equal.cong(U32,Tile.Tile8,start => run_value(tree,Cells.group_run(context,index,start),position),U32.from_nat(Nat.mul(block,U32.to_nat(groups))),(index * groups : U32),
    Equal.sym(U32,(index * groups : U32),U32.from_nat(Nat.mul(block,U32.to_nat(groups))),Encoded.scale_encoded(block,groups)))

# Inside a block within the machine the run writes the cell the flat pass
# writes at the same position.
def tiled_same(+reader: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,+context: Cells.Tiling,+limits: Machine.Limits,+total: U32,+block: Nat,+offset: Nat,
  inside: {Nat.is_lt(offset,U32.to_nat(tiling_groups(context))) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(tiling_groups(context))),U32.to_nat(tiling_groups(context))),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {Tree.write(Tile.Tile8,output,
      Blocks.in_place(~Cells.Run,Cells.group_run(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(tiling_groups(context))))),
        U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(tiling_groups(context))),offset))),
      run_value(reader,Cells.group_run(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(tiling_groups(context))))),
        U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(tiling_groups(context))),offset)))) ==
    Tree.write(Tile.Tile8,output,Blocks.in_place(~Cells.Tiling,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(tiling_groups(context))),offset))),
      tiled_cell(reader,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(tiling_groups(context))),offset)))) : Tree.Tree<Tile.Tile8>}:
  Cells.Tiling{+after,+lanes,+groups,+first} = context
  +position = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(groups)),offset))
  Equal.cong(Tile.Tile8,Tree.Tree<Tile.Tile8>,value => Tree.write(Tile.Tile8,output,position,value),
    run_value(reader,Cells.group_run(Cells.Tiling{after,lanes,groups,first},U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(groups)))),position),
    tiled_cell(reader,Cells.Tiling{after,lanes,groups,first},position),
    run_decoded(reader,after,lanes,groups,first,block,offset,decoded(limits,total,groups,block,offset,inside,room,bounded)))

# Within the machine the tiled run over blocks outer indices is one flat pass.
def tiled_flat(+limits: Machine.Limits,+total: U32,+blocks: U32,+context: Cells.Tiling,+source: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>,
  +exact: {Nat.mul(U32.to_nat(blocks),U32.to_nat(tiling_groups(context))) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {tiled_blocks(blocks,context,source,output) ==
    AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Tiling>,~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~tiled_cell,~Blocks.in_place(~Cells.Tiling)),
        U32.to_nat(total),Traversal.State{source,output,0,context})) : AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>}:
  Cells.Tiling{+after,+lanes,+groups,+first} = context
  BlocksLaws.flat(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~Cells.group_run,~run_value,~Blocks.in_place(~Cells.Run),~tiling_groups,~tiled_cell,~Blocks.in_place(~Cells.Tiling),
    ~tiled_same,U32.to_nat(blocks),limits,total,Cells.Tiling{after,lanes,groups,first},source,output,exact,bounded)

# The offset of a decoded position past the start of its block.
def local_of(+size: U32,+block: Nat,+offset: Nat,facts: Decoded<size,block,offset>) ->
  {(U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size)),offset)) - U32.from_nat(Nat.mul(block,U32.to_nat(size))) : U32) == U32.from_nat(offset) : U32}:
  Decoded{quotient,remainder,local} = facts
  local

# The spot of group group at pixel pixel of one image.
def image_spot(+after: U32,+lanes: U32,+pixel: Nat,+group: Nat,+position: U32,+start: U32,+local: {(position - start : U32) == U32.from_nat(group) : U32}) ->
  {Cells.Spot{(0 + (position - start) * 8 : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + (position - start) * 8 * after : U32),after,lanes} ==
    Cells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(Nat.add(Nat.mul(Nat.mul(group,8n),U32.to_nat(after)),pixel)),after,lanes} : Cells.Spot}:
  +scaled = Nat.mul(group,8n)
  +rows = Nat.mul(scaled,U32.to_nat(after))
  +goal = {Cells.Spot{U32.from_nat(scaled),U32.from_nat(Nat.add(rows,pixel)),after,lanes} : Cells.Spot}
  %Equal.sym(U32,(position - start : U32),U32.from_nat(group),local) :
    {Cells.Spot{(0 + _ * 8 : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + _ * 8 * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(U32.from_nat(group) * 8 : U32),U32.from_nat(scaled),Encoded.scale_encoded(group,8)) :
    {Cells.Spot{(0 + _ : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + _ * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(0 + U32.from_nat(scaled) : U32),U32.from_nat(scaled),Arithmetic.add_zero_left(U32.from_nat(scaled))) :
    {Cells.Spot{_,((0 * lanes * after + U32.from_nat(pixel) : U32) + U32.from_nat(scaled) * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(0 * lanes : U32),0,Arithmetic.multiply_zero_left(lanes)) :
    {Cells.Spot{U32.from_nat(scaled),((_ * after + U32.from_nat(pixel) : U32) + U32.from_nat(scaled) * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(0 * after : U32),0,Arithmetic.multiply_zero_left(after)) :
    {Cells.Spot{U32.from_nat(scaled),((_ + U32.from_nat(pixel) : U32) + U32.from_nat(scaled) * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(0 + U32.from_nat(pixel) : U32),U32.from_nat(pixel),Arithmetic.add_zero_left(U32.from_nat(pixel))) :
    {Cells.Spot{U32.from_nat(scaled),(_ + U32.from_nat(scaled) * after : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(U32.from_nat(scaled) * after : U32),U32.from_nat(rows),Encoded.scale_encoded(scaled,after)) :
    {Cells.Spot{U32.from_nat(scaled),(U32.from_nat(pixel) + _ : U32),after,lanes} == goal : Cells.Spot}
  %Equal.sym(U32,(U32.from_nat(pixel) + U32.from_nat(rows) : U32),U32.from_nat(Nat.add(pixel,rows)),WordArithmetic.native_add(pixel,rows)) :
    {Cells.Spot{U32.from_nat(scaled),_,after,lanes} == goal : Cells.Spot}
  Equal.cong(Nat,Cells.Spot,value => Cells.Spot{U32.from_nat(scaled),U32.from_nat(value),after,lanes},Nat.add(pixel,rows),Nat.add(rows,pixel),Order.commutative(pixel,rows))

# The run of pixel pixel of one image starts at its first channel.
def image_run(+tree: Tree.Tree<F32>,+after: U32,+lanes: U32,+groups: U32,+pixel: Nat,+start: U32,+position: U32,facts: Decoded<after,0n,pixel>,
  +positive: {Nat.is_lt(0n,U32.to_nat(after)) == True{} : Bool}) ->
  {run_value(tree,Cells.group_run(Cells.Tiling{after,lanes,groups,0},U32.from_nat(pixel),start),position) ==
    spot_cell(tree,Cells.Spot{(0 + (position - start) * 8 : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + (position - start) * 8 * after : U32),after,lanes}) : Tile.Tile8}:
  Decoded{quotient,remainder,local} = facts
  +index = U32.from_nat(pixel)
  +group = (position - start : U32)
  +goal = {spot_cell(tree,Cells.Spot{(0 + group * 8 : U32),((0 * lanes * after + index : U32) + group * 8 * after : U32),after,lanes}) : Tile.Tile8}
  %Equal.sym(U32,(0 + index : U32),index,Arithmetic.add_zero_left(index)) :
    {spot_cell(tree,Cells.Spot{(0 + group * 8 : U32),((_ / U32.max(after,1) * lanes * after + _ % U32.max(after,1) : U32) + group * 8 * after : U32),after,lanes}) == goal : Tile.Tile8}
  %Equal.sym(U32,U32.max(after,1),after,Depth.positive_max(after,positive)) :
    {spot_cell(tree,Cells.Spot{(0 + group * 8 : U32),((index / _ * lanes * after + index % _ : U32) + group * 8 * after : U32),after,lanes}) == goal : Tile.Tile8}
  %Equal.sym(U32,(index / after : U32),0,quotient) :
    {spot_cell(tree,Cells.Spot{(0 + group * 8 : U32),((_ * lanes * after + index % after : U32) + group * 8 * after : U32),after,lanes}) == goal : Tile.Tile8}
  %Equal.sym(U32,(index % after : U32),index,remainder) :
    {spot_cell(tree,Cells.Spot{(0 + group * 8 : U32),((0 * lanes * after + _ : U32) + group * 8 * after : U32),after,lanes}) == goal : Tile.Tile8}
  {==}

# The cell of pixel and group of one image holds the eight lanes from channel
# 8 * group at pixel on.
def tiled_read(+limits: Machine.Limits,+total: U32,+after: U32,+lanes: U32,+groups: U32,+fill: Tile.Tile8,+source: Tree.Tree<F32>,+pixel: Nat,+group: Nat,
  +exact: {Nat.mul(U32.to_nat(after),U32.to_nat(groups)) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool},
  +pixel_inside: {Nat.is_lt(pixel,U32.to_nat(after)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,tiled_blocks(after,Cells.Tiling{after,lanes,groups,0},source,Tree.fresh(Tile.Tile8,Storage.depth_for(total),fill))),
      U32.from_nat(Nat.add(Nat.mul(pixel,U32.to_nat(groups)),group))) ==
    spot_cell(source,Cells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(Nat.add(Nat.mul(Nat.mul(group,8n),U32.to_nat(after)),pixel)),after,lanes}) : Tile.Tile8}:
  +context = {Cells.Tiling{after,lanes,groups,0} : Cells.Tiling}
  +g = U32.to_nat(groups)
  +p = U32.to_nat(after)
  +position = Nat.add(Nat.mul(pixel,g),group)
  +start = U32.from_nat(Nat.mul(pixel,g))
  +fresh = Tree.fresh(Tile.Tile8,Storage.depth_for(total),fill)
  +flat = tiled_flat(limits,total,after,context,source,fresh,exact,bounded)
  +selected = inside_rows(pixel,p,group,g,U32.to_nat(total),exact,pixel_inside,group_inside)
  +positive = Order.le_lt_trans(0n,pixel,p,Order.zero_le(pixel),pixel_inside)
  +room = row_room(pixel,p,g,U32.to_nat(total),exact,pixel_inside)
  +local_facts = decoded(limits,total,groups,pixel,group,group_inside,room,bounded)
  +image_facts = decoded(limits,total,after,0n,pixel,pixel_inside,
    Order.transitive(p,Nat.mul(p,g),U32.to_nat(total),RowMajor.product_covers(p,g,Order.le_lt_trans(0n,group,g,Order.zero_le(group),group_inside)),
      %exact : {Nat.is_le(Nat.mul(p,g),_) == True{} : Bool} Order.reflexive(Nat.mul(p,g))),bounded)
  +spot = {Cells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(Nat.add(Nat.mul(Nat.mul(group,8n),p),pixel)),after,lanes} : Cells.Spot}
  %Equal.sym(AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>,tiled_blocks(after,context,source,fresh),
    AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Tiling>,~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~tiled_cell,~Blocks.in_place(~Cells.Tiling)),
        U32.to_nat(total),Traversal.State{source,fresh,0,context})),flat) :
    {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,_),U32.from_nat(position)) == spot_cell(source,spot) : Tile.Tile8}
  Equal.trans(Tile.Tile8,
    Tree.read(Tile.Tile8,IndexedValues.run(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~tiled_cell,~Blocks.in_place(~Cells.Tiling),U32.to_nat(total),0,context,source,fresh),U32.from_nat(position)),
    tiled_cell(source,context,U32.from_nat(position)),spot_cell(source,spot),
    fresh_stored(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~tiled_cell,total,fill,context,source,position,allocatable(limits,total,bounded),selected),
    Equal.trans(Tile.Tile8,tiled_cell(source,context,U32.from_nat(position)),
      run_value(source,Cells.group_run(context,U32.from_nat(pixel),start),U32.from_nat(position)),spot_cell(source,spot),
      Equal.sym(Tile.Tile8,run_value(source,Cells.group_run(context,U32.from_nat(pixel),start),U32.from_nat(position)),tiled_cell(source,context,U32.from_nat(position)),
        run_decoded(source,after,lanes,groups,0,pixel,group,local_facts)),
      Equal.trans(Tile.Tile8,run_value(source,Cells.group_run(context,U32.from_nat(pixel),start),U32.from_nat(position)),
        spot_cell(source,Cells.Spot{(0 + (U32.from_nat(position) - start) * 8 : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + (U32.from_nat(position) - start) * 8 * after : U32),after,lanes}),
        spot_cell(source,spot),
        image_run(source,after,lanes,groups,pixel,start,U32.from_nat(position),image_facts,positive),
        Equal.cong(Cells.Spot,Tile.Tile8,value => spot_cell(source,value),
          Cells.Spot{(0 + (U32.from_nat(position) - start) * 8 : U32),((0 * lanes * after + U32.from_nat(pixel) : U32) + (U32.from_nat(position) - start) * 8 * after : U32),after,lanes},spot,
          image_spot(after,lanes,pixel,group,U32.from_nat(position),start,local_of(groups,pixel,group,local_facts))))))

# ---- From cells ----

def cells_after(context: Cells.Cells) -> U32:
  Cells.Cells{outer,+after,lanes,groups,position} = context
  after

# The value the lane read of a pick gives at a position on a tree.
def pick_value(tree: Tree.Tree<Tile.Tile8>,pick: Cells.Pick,position: U32) -> F32:
  Cells.Pick{lane,base,step,start} = pick
  Cells.lane_of(Tree.read(Tile.Tile8,tree,(base + (position - start) * step : U32)),lane)

def value_read(+tree: Tree.Tree<Tile.Tile8>,+pick: Cells.Pick,+position: U32) ->
  {Cells.value_read(Tree.pack(Tile.Tile8,tree),pick,position) == (Tree.pack(Tile.Tile8,tree),pick_value(tree,pick,position)) : Array<Tile.Tile8> & F32}:
  Cells.Pick{+lane,+base,+step,+start} = pick
  +index = (base + (position - start) * step : U32)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,tree),index),(Tree.pack(Tile.Tile8,tree),Tree.read(Tile.Tile8,tree,index)),StorageLaws.read(Tile.Tile8,tree,index)) :
    {Cells.lane_read(lane,_) == (Tree.pack(Tile.Tile8,tree),pick_value(tree,Cells.Pick{lane,base,step,start},position)) : Array<Tile.Tile8> & F32}
  {==}

# The value at a position on a tree of cells: the pick of its block.
def value_value(tree: Tree.Tree<Tile.Tile8>,context: Cells.Cells,position: U32) -> F32:
  Cells.Cells{outer,+after,lanes,groups,arrangement} = context
  +position = position
  +block = (position / after : U32)
  pick_value(tree,Cells.channel_pick(Cells.Cells{outer,after,lanes,groups,arrangement},block,(block * after : U32)),position)

# The model of the untiled run: blocks blocks of after values.
def untiled_blocks(+blocks: U32,context: Cells.Cells,input: Tree.Tree<Tile.Tile8>,output: Tree.Tree<F32>) -> AccessLaws.ModelResult<Tree.Tree<Tile.Tile8>,F32>:
  Cells.Cells{outer,+after,lanes,groups,arrangement} = context
  BlocksLaws.blocked(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~Cells.Pick,~Cells.channel_pick,~pick_value,~Blocks.in_place(~Cells.Pick),U32.to_nat(blocks),after,
    Cells.Cells{outer,after,lanes,groups,arrangement},input,output)

def untiled_run(+blocks: U32,+context: Cells.Cells,+input: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>) ->
  {Cells.untiled(blocks,context,Tree.pack(Tile.Tile8,input),Tree.pack(F32,output)) ==
    AccessLaws.pack_result(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~F32,~ArrayLaws.pack_tree(~Tile.Tile8),untiled_blocks(blocks,context,input,output)) : Array<Tile.Tile8> & Array<F32>}:
  Cells.Cells{+outer,+after,+lanes,+groups,+arrangement} = context
  BlocksLaws.represented(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~Cells.Pick,~ArrayLaws.pack_tree(~Tile.Tile8),~Cells.channel_pick,~Cells.value_read,~pick_value,
    ~Blocks.in_place(~Cells.Pick),~value_read,U32.to_nat(blocks),after,Cells.Cells{outer,after,lanes,groups,arrangement},input,output)

# The pick of a block reads at every position of it what the pick of the block
# the position decodes to reads.
def pick_decoded(+tree: Tree.Tree<Tile.Tile8>,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+arrangement: Grid.Position,+block: Nat,+offset: Nat,
  facts: Decoded<after,block,offset>) ->
  {pick_value(tree,Cells.channel_pick(Cells.Cells{outer,after,lanes,groups,arrangement},U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(after)))),
      U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(after)),offset))) ==
    value_value(tree,Cells.Cells{outer,after,lanes,groups,arrangement},U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(after)),offset))) : F32}:
  Decoded{quotient,remainder,local} = facts
  +context = {Cells.Cells{outer,after,lanes,groups,arrangement} : Cells.Cells}
  +position = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(after)),offset))
  +index = U32.from_nat(block)
  %Equal.sym(U32,(position / after : U32),index,quotient) :
    {pick_value(tree,Cells.channel_pick(context,index,U32.from_nat(Nat.mul(block,U32.to_nat(after)))),position) ==
      pick_value(tree,Cells.channel_pick(context,_,(_ * after : U32)),position) : F32}
  Equal.cong(U32,F32,start => pick_value(tree,Cells.channel_pick(context,index,start),position),U32.from_nat(Nat.mul(block,U32.to_nat(after))),(index * after : U32),
    Equal.sym(U32,(index * after : U32),U32.from_nat(Nat.mul(block,U32.to_nat(after))),Encoded.scale_encoded(block,after)))

# Inside a block within the machine the run writes the value the flat pass
# writes at the same position.
def untiled_same(+reader: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,+context: Cells.Cells,+limits: Machine.Limits,+total: U32,+block: Nat,+offset: Nat,
  inside: {Nat.is_lt(offset,U32.to_nat(cells_after(context))) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(cells_after(context))),U32.to_nat(cells_after(context))),U32.to_nat(total)) == True{} : Bool},
  bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {Tree.write(F32,output,
      Blocks.in_place(~Cells.Pick,Cells.channel_pick(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(cells_after(context))))),
        U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(cells_after(context))),offset))),
      pick_value(reader,Cells.channel_pick(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(cells_after(context))))),
        U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(cells_after(context))),offset)))) ==
    Tree.write(F32,output,Blocks.in_place(~Cells.Cells,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(cells_after(context))),offset))),
      value_value(reader,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(cells_after(context))),offset)))) : Tree.Tree<F32>}:
  Cells.Cells{+outer,+after,+lanes,+groups,+arrangement} = context
  +position = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(after)),offset))
  Equal.cong(F32,Tree.Tree<F32>,value => Tree.write(F32,output,position,value),
    pick_value(reader,Cells.channel_pick(Cells.Cells{outer,after,lanes,groups,arrangement},U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(after)))),position),
    value_value(reader,Cells.Cells{outer,after,lanes,groups,arrangement},position),
    pick_decoded(reader,outer,after,lanes,groups,arrangement,block,offset,decoded(limits,total,after,block,offset,inside,room,bounded)))

# Within the machine the untiled run over blocks blocks is one flat pass.
def untiled_flat(+limits: Machine.Limits,+total: U32,+blocks: U32,+context: Cells.Cells,+source: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<F32>,
  +exact: {Nat.mul(U32.to_nat(blocks),U32.to_nat(cells_after(context))) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {untiled_blocks(blocks,context,source,output) ==
    AccessLaws.finish_model(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<F32>,Cells.Cells>,~ArrayLaws.model_step(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~value_value,~Blocks.in_place(~Cells.Cells)),
        U32.to_nat(total),Traversal.State{source,output,0,context})) : AccessLaws.ModelResult<Tree.Tree<Tile.Tile8>,F32>}:
  Cells.Cells{+outer,+after,+lanes,+groups,+arrangement} = context
  BlocksLaws.flat(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~Cells.Pick,~Cells.channel_pick,~pick_value,~Blocks.in_place(~Cells.Pick),~cells_after,~value_value,~Blocks.in_place(~Cells.Cells),
    ~untiled_same,U32.to_nat(blocks),limits,total,Cells.Cells{outer,after,lanes,groups,arrangement},source,output,exact,bounded)

def image_pick(+tree: Tree.Tree<Tile.Tile8>,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+channel: Nat,+start: U32,+position: U32,facts: Decoded<lanes,0n,channel>,
  +positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) ->
  {pick_value(tree,Cells.channel_pick(Cells.Cells{outer,after,lanes,groups,Grid.Innermost{}},U32.from_nat(channel),start),position) ==
    Cells.lane_of(Tree.read(Tile.Tile8,tree,((0 * after * groups + U32.from_nat(channel) / 8 : U32) + (position - start) * groups : U32)),(U32.from_nat(channel) % 8 : U32)) : F32}:
  Decoded{quotient,remainder,local} = facts
  +index = U32.from_nat(channel)
  +goal = {Cells.lane_of(Tree.read(Tile.Tile8,tree,((0 * after * groups + index / 8 : U32) + (position - start) * groups : U32)),(index % 8 : U32)) : F32}
  %Equal.sym(U32,U32.max(lanes,1),lanes,Depth.positive_max(lanes,positive)) :
    {Cells.lane_of(Tree.read(Tile.Tile8,tree,((index / _ * after * groups + index % _ / 8 : U32) + (position - start) * groups : U32)),(index % _ % 8 : U32)) == goal : F32}
  %Equal.sym(U32,(index / lanes : U32),0,quotient) :
    {Cells.lane_of(Tree.read(Tile.Tile8,tree,((_ * after * groups + index % lanes / 8 : U32) + (position - start) * groups : U32)),(index % lanes % 8 : U32)) == goal : F32}
  %Equal.sym(U32,(index % lanes : U32),index,remainder) :
    {Cells.lane_of(Tree.read(Tile.Tile8,tree,((0 * after * groups + _ / 8 : U32) + (position - start) * groups : U32)),(_ % 8 : U32)) == goal : F32}
  {==}

# The cell index of the pick of a channel at an offset of one image.
def image_cell(+limits: Machine.Limits,+after: U32,+groups: U32,+channel: Nat,+offset: Nat,+position: U32,+start: U32,
  +local: {(position - start : U32) == U32.from_nat(offset) : U32},+bound: {Nat.is_le(channel,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {((0 * after * groups + U32.from_nat(channel) / 8 : U32) + (position - start) * groups : U32) ==
    U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),Nat.div(channel,8n))) : U32}:
  +half = Machine.half_word(limits)
  +encoded = Bounds.roundtrip(channel,half,Limits.half_word_fits(limits),bound)
  +sixteen = Bounds.as_nat_le(16,half,Limits.sixteen_within_half(limits))
  +group = Nat.div(channel,8n)
  +divided = Arithmetic.from_nat_exact((U32.from_nat(channel) / 8 : U32),group,
    %encoded : {U32.to_nat((U32.from_nat(channel) / 8 : U32)) == Nat.div(_,8n) : Nat}
    NativeDivision.divide(U32.from_nat(channel),8,half,Limits.half_word_fits(limits),sixteen,{==}))
  +goal = U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),group))
  %Equal.sym(U32,(position - start : U32),U32.from_nat(offset),local) : {((0 * after * groups + U32.from_nat(channel) / 8 : U32) + _ * groups : U32) == goal : U32}
  %Equal.sym(U32,(0 * after : U32),0,Arithmetic.multiply_zero_left(after)) : {((_ * groups + U32.from_nat(channel) / 8 : U32) + U32.from_nat(offset) * groups : U32) == goal : U32}
  %Equal.sym(U32,(0 * groups : U32),0,Arithmetic.multiply_zero_left(groups)) : {((_ + U32.from_nat(channel) / 8 : U32) + U32.from_nat(offset) * groups : U32) == goal : U32}
  %Equal.sym(U32,(0 + (U32.from_nat(channel) / 8 : U32) : U32),(U32.from_nat(channel) / 8 : U32),Arithmetic.add_zero_left((U32.from_nat(channel) / 8 : U32))) :
    {(_ + U32.from_nat(offset) * groups : U32) == goal : U32}
  %divided : {(_ + U32.from_nat(offset) * groups : U32) == goal : U32}
  %Arithmetic.native_roundtrip(groups) : {(U32.from_nat(group) + U32.from_nat(offset) * _ : U32) == goal : U32}
  Equal.trans(U32,(U32.from_nat(group) + U32.from_nat(offset) * U32.from_nat(U32.to_nat(groups)) : U32),
    (U32.from_nat(offset) * U32.from_nat(U32.to_nat(groups)) + U32.from_nat(group) : U32),goal,
    Arithmetic.add_commutative(U32.from_nat(group),(U32.from_nat(offset) * U32.from_nat(U32.to_nat(groups)) : U32)),
    Encoded.encoded_madd(offset,U32.to_nat(groups),group))

# The lane of a channel below half the word is its number modulo eight.
def image_lane(+limits: Machine.Limits,+channel: Nat,+bound: {Nat.is_le(channel,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.from_nat(Nat.mod(channel,8n)) == (U32.from_nat(channel) % 8 : U32) : U32}:
  +half = Machine.half_word(limits)
  Arithmetic.from_nat_exact((U32.from_nat(channel) % 8 : U32),Nat.mod(channel,8n),
    %Bounds.roundtrip(channel,half,Limits.half_word_fits(limits),bound) : {U32.to_nat((U32.from_nat(channel) % 8 : U32)) == Nat.mod(_,8n) : Nat}
    NativeDivision.modulo(U32.from_nat(channel),8,half,Limits.half_word_fits(limits),Bounds.as_nat_le(16,half,Limits.sixteen_within_half(limits)),{==}))

# The value of channel channel at offset of one image whose groups lie
# innermost is lane channel % 8 of the cell of group channel / 8 at offset.
def untiled_read(+limits: Machine.Limits,+total: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+source: Tree.Tree<Tile.Tile8>,+channel: Nat,+offset: Nat,
  +exact: {Nat.mul(U32.to_nat(lanes),U32.to_nat(after)) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,U32.to_nat(after)) == True{} : Bool}) ->
  {Tree.read(F32,IndexedValues.output_tree(~Tree.Tree<Tile.Tile8>,~F32,
      untiled_blocks(lanes,Cells.Cells{outer,after,lanes,groups,Grid.Innermost{}},source,Tree.fresh(F32,Storage.depth_for(total),0.0))),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(after)),offset))) ==
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,source,U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),Nat.div(channel,8n))))) : F32}:
  +context = {Cells.Cells{outer,after,lanes,groups,Grid.Innermost{}} : Cells.Cells}
  +p = U32.to_nat(after)
  +l = U32.to_nat(lanes)
  +position = Nat.add(Nat.mul(channel,p),offset)
  +start = U32.from_nat(Nat.mul(channel,p))
  +fresh = Tree.fresh(F32,Storage.depth_for(total),0.0)
  +flat = untiled_flat(limits,total,lanes,context,source,fresh,exact,bounded)
  +selected = inside_rows(channel,l,offset,p,U32.to_nat(total),exact,channel_inside,offset_inside)
  +room = row_room(channel,l,p,U32.to_nat(total),exact,channel_inside)
  +positive = Order.le_lt_trans(0n,channel,l,Order.zero_le(channel),channel_inside)
  +facts = decoded(limits,total,after,channel,offset,offset_inside,room,bounded)
  +lanes_total = Order.transitive(l,Nat.mul(l,p),U32.to_nat(total),RowMajor.product_covers(l,p,Order.le_lt_trans(0n,offset,p,Order.zero_le(offset),offset_inside)),
    %exact : {Nat.is_le(Nat.mul(l,p),_) == True{} : Bool} Order.reflexive(Nat.mul(l,p)))
  +image_facts = decoded(limits,total,lanes,0n,channel,channel_inside,lanes_total,bounded)
  +bound = Bounds.strict_le(channel,U32.to_nat(Machine.half_word(limits)),Bounds.nat_trans(channel,l,U32.to_nat(Machine.half_word(limits)),channel_inside,
    Order.transitive(l,U32.to_nat(total),U32.to_nat(Machine.half_word(limits)),lanes_total,
      Order.transitive(U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),Bounds.as_nat_le(total,Machine.capacity(limits),bounded),
        Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits))))))
  +read = Tree.read(Tile.Tile8,source,U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),Nat.div(channel,8n))))
  +picked_read = Tree.read(Tile.Tile8,source,((0 * after * groups + U32.from_nat(channel) / 8 : U32) + (U32.from_nat(position) - start) * groups : U32))
  +local = local_of(after,channel,offset,facts)
  %Equal.sym(AccessLaws.ModelResult<Tree.Tree<Tile.Tile8>,F32>,untiled_blocks(lanes,context,source,fresh),
    AccessLaws.finish_model(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<F32>,Cells.Cells>,~ArrayLaws.model_step(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~value_value,~Blocks.in_place(~Cells.Cells)),
        U32.to_nat(total),Traversal.State{source,fresh,0,context})),flat) :
    {Tree.read(F32,IndexedValues.output_tree(~Tree.Tree<Tile.Tile8>,~F32,_),U32.from_nat(position)) == TileLaws.select_lane(Nat.mod(channel,8n),read) : F32}
  Equal.trans(F32,
    Tree.read(F32,IndexedValues.run(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~value_value,~Blocks.in_place(~Cells.Cells),U32.to_nat(total),0,context,source,fresh),U32.from_nat(position)),
    value_value(source,context,U32.from_nat(position)),TileLaws.select_lane(Nat.mod(channel,8n),read),
    fresh_stored(~Tree.Tree<Tile.Tile8>,~F32,~Cells.Cells,~value_value,total,0.0,context,source,position,allocatable(limits,total,bounded),selected),
    Equal.trans(F32,value_value(source,context,U32.from_nat(position)),pick_value(source,Cells.channel_pick(context,U32.from_nat(channel),start),U32.from_nat(position)),
      TileLaws.select_lane(Nat.mod(channel,8n),read),
      Equal.sym(F32,pick_value(source,Cells.channel_pick(context,U32.from_nat(channel),start),U32.from_nat(position)),value_value(source,context,U32.from_nat(position)),
        pick_decoded(source,outer,after,lanes,groups,Grid.Innermost{},channel,offset,facts)),
      Equal.trans(F32,pick_value(source,Cells.channel_pick(context,U32.from_nat(channel),start),U32.from_nat(position)),
        Cells.lane_of(picked_read,(U32.from_nat(channel) % 8 : U32)),TileLaws.select_lane(Nat.mod(channel,8n),read),
        image_pick(source,outer,after,lanes,groups,channel,start,U32.from_nat(position),image_facts,positive),
        %Equal.sym(U32,((0 * after * groups + U32.from_nat(channel) / 8 : U32) + (U32.from_nat(position) - start) * groups : U32),
          U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),Nat.div(channel,8n))),image_cell(limits,after,groups,channel,offset,U32.from_nat(position),start,local,bound)) :
          {Cells.lane_of(Tree.read(Tile.Tile8,source,_),(U32.from_nat(channel) % 8 : U32)) == TileLaws.select_lane(Nat.mod(channel,8n),read) : F32}
        %image_lane(limits,channel,bound) : {Cells.lane_of(read,_) == TileLaws.select_lane(Nat.mod(channel,8n),read) : F32}
        lane_selected(read,Nat.mod(channel,8n),DivisionOrder.remainder_small(channel,8n,{==})))))

# The values of the channels of one image of plane positions whose groups lie
# innermost.
def image_values(+lanes: U32,+plane: U32,cells: Tree.Tree<Tile.Tile8>) -> Tree.Tree<F32>:
  IndexedValues.output_tree(~Tree.Tree<Tile.Tile8>,~F32,
    untiled_blocks(lanes,Cells.Cells{plane,plane,lanes,Grid.groups(lanes),Grid.Innermost{}},cells,Tree.fresh(F32,Storage.depth_for((lanes * plane : U32)),0.0)))

# ---- The values of the cells of values ----

# Lane lane of the cell of a spot is the value at the spot lane lanes on.
def spot_lane(+tree: Tree.Tree<F32>,+spot: Cells.Spot,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,spot_cell(tree,spot)) == spot_value(tree,steps(spot,lane)) : F32}:
  match lane:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+ +rest:
      Empty.absurd({TileLaws.select_lane(8n+rest,spot_cell(tree,spot)) == spot_value(tree,steps(spot,8n+rest)) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

# Lane channel % 8 past the planes of the channels before group channel / 8 is
# the position of channel at offset.
def lane_position(+channel: Nat,+size: Nat,+offset: Nat) ->
  {Nat.add(Nat.mul(Nat.mod(channel,8n),size),Nat.add(Nat.mul(Nat.mul(Nat.div(channel,8n),8n),size),offset)) == Nat.add(Nat.mul(channel,size),offset) : Nat}:
  +lane = Nat.mod(channel,8n)
  +base = Nat.mul(Nat.div(channel,8n),8n)
  Equal.trans(Nat,Nat.add(Nat.mul(lane,size),Nat.add(Nat.mul(base,size),offset)),Nat.add(Nat.add(Nat.mul(lane,size),Nat.mul(base,size)),offset),Nat.add(Nat.mul(channel,size),offset),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(lane,size),Nat.mul(base,size)),offset),Nat.add(Nat.mul(lane,size),Nat.add(Nat.mul(base,size),offset)),
      Addition.associative(Nat.mul(lane,size),Nat.mul(base,size),offset)),
    Equal.cong(Nat,Nat,value => Nat.add(value,offset),Nat.add(Nat.mul(lane,size),Nat.mul(base,size)),Nat.mul(channel,size),
      Equal.trans(Nat,Nat.add(Nat.mul(lane,size),Nat.mul(base,size)),Nat.mul(Nat.add(lane,base),size),Nat.mul(channel,size),
        Equal.sym(Nat,Nat.mul(Nat.add(lane,base),size),Nat.add(Nat.mul(lane,size),Nat.mul(base,size)),RowMajor.distrib(lane,base,size)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,size),Nat.add(lane,base),channel,channel_split(channel)))))

# Lane channel % 8 of the cell of offset and group channel / 8 of one image is
# the value of channel at offset.
def tiled_lane(+limits: Machine.Limits,+cells: U32,+after: U32,+lanes: U32,+fill: Tile.Tile8,+source: Tree.Tree<F32>,+channel: Nat,+offset: Nat,
  +cells_exact: {Nat.mul(U32.to_nat(after),U32.to_nat(Grid.groups(lanes))) == U32.to_nat(cells) : Nat},
  +cells_bounded: {U32.is_le(cells,Machine.capacity(limits)) == True{} : Bool},
  +lanes_bounded: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,U32.to_nat(after)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,
      IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,tiled_blocks(after,Cells.Tiling{after,lanes,Grid.groups(lanes),0},source,Tree.fresh(Tile.Tile8,Storage.depth_for(cells),fill))),
      U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(Grid.groups(lanes))),Nat.div(channel,8n))))) ==
    Tree.read(F32,source,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(after)),offset))) : F32}:
  +groups = Grid.groups(lanes)
  +tiles = IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,tiled_blocks(after,Cells.Tiling{after,lanes,groups,0},source,Tree.fresh(Tile.Tile8,Storage.depth_for(cells),fill)))
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +p = U32.to_nat(after)
  +l = U32.to_nat(lanes)
  +base = Nat.add(Nat.mul(Nat.mul(group,8n),p),offset)
  +position = Nat.add(Nat.mul(channel,p),offset)
  +facts = grouped(limits,lanes,lanes_bounded)
  +bound = channel_bound(limits,lanes,group,lane,facts,group_inside(limits,lanes,channel,facts,channel_inside),DivisionOrder.remainder_small(channel,8n,{==}))
  +goal = Tree.read(F32,source,U32.from_nat(position))
  +spot = {Cells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(base),after,lanes} : Cells.Spot}
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,tiles,U32.from_nat(Nat.add(Nat.mul(offset,U32.to_nat(groups)),group))),spot_cell(source,spot),
    tiled_read(limits,cells,after,lanes,groups,fill,source,offset,group,cells_exact,cells_bounded,offset_inside,group_inside(limits,lanes,channel,facts,channel_inside))) :
    {TileLaws.select_lane(lane,_) == goal : F32}
  %Equal.sym(F32,TileLaws.select_lane(lane,spot_cell(source,spot)),spot_value(source,steps(spot,lane)),spot_lane(source,spot,lane,DivisionOrder.remainder_small(channel,8n,{==}))) :
    {_ == goal : F32}
  %Equal.sym(Cells.Spot,steps(spot,lane),Cells.Spot{U32.from_nat(Nat.add(lane,Nat.mul(group,8n))),U32.from_nat(Nat.add(Nat.mul(lane,p),base)),after,lanes},
    stepped(Nat.mul(group,8n),base,after,lanes,lane)) : {spot_value(source,_) == goal : F32}
  %Equal.sym(Nat,Nat.add(Nat.mul(lane,p),base),position,lane_position(channel,p,offset)) :
    {Bool.pick(F32,U32.is_lt(U32.from_nat(Nat.add(lane,Nat.mul(group,8n))),lanes),Tree.read(F32,source,U32.from_nat(_)),0.0) == goal : F32}
  %Equal.sym(Nat,Nat.add(lane,Nat.mul(group,8n)),channel,channel_split(channel)) : {Bool.pick(F32,U32.is_lt(U32.from_nat(_),lanes),goal,0.0) == goal : F32}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(channel),lanes),Nat.is_lt(channel,l),
    channel_compare(limits,channel,lanes,%channel_split(channel) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool} bound)) :
    {Bool.pick(F32,_,goal,0.0) == goal : F32}
  %Equal.sym(Bool,Nat.is_lt(channel,l),True{},channel_inside) : {Bool.pick(F32,_,goal,0.0) == goal : F32}
  {==}

def cells_address(position: Grid.Position,+outer: U32,+after: U32,+groups: U32,+before: Nat,+channel: Nat,+offset: Nat) -> U32:
  match position:
    case Grid.Innermost{}: ((U32.from_nat(before) * after * groups + U32.from_nat(channel) / 8 : U32) + U32.from_nat(offset) * groups : U32)
    case Grid.Outermost{}: ((U32.from_nat(channel) / 8 * outer + U32.from_nat(before) * after : U32) + U32.from_nat(offset) * 1 : U32)

def cells_pick(+tree: Tree.Tree<Tile.Tile8>,+outer: U32,+after: U32,+groups: U32,position: Grid.Position,+before: Nat,+channel: Nat,+offset: Nat,+start: U32,+at: U32,
  +local: {(at - start : U32) == U32.from_nat(offset) : U32}) ->
  {pick_value(tree,Cells.picked(position,outer,after,groups,U32.from_nat(before),U32.from_nat(channel),start),at) ==
    Cells.lane_of(Tree.read(Tile.Tile8,tree,cells_address(position,outer,after,groups,before,channel,offset)),(U32.from_nat(channel) % 8 : U32)) : F32}:
  match position:
    case Grid.Innermost{}:
      +base = (U32.from_nat(before) * after * groups + U32.from_nat(channel) / 8 : U32)
      %Equal.sym(U32,(at - start : U32),U32.from_nat(offset),local) :
        {Cells.lane_of(Tree.read(Tile.Tile8,tree,(base + _ * groups : U32)),(U32.from_nat(channel) % 8 : U32)) ==
          Cells.lane_of(Tree.read(Tile.Tile8,tree,(base + U32.from_nat(offset) * groups : U32)),(U32.from_nat(channel) % 8 : U32)) : F32}
      {==}
    case Grid.Outermost{}:
      +base = (U32.from_nat(channel) / 8 * outer + U32.from_nat(before) * after : U32)
      %Equal.sym(U32,(at - start : U32),U32.from_nat(offset),local) :
        {Cells.lane_of(Tree.read(Tile.Tile8,tree,(base + _ * 1 : U32)),(U32.from_nat(channel) % 8 : U32)) ==
          Cells.lane_of(Tree.read(Tile.Tile8,tree,(base + U32.from_nat(offset) * 1 : U32)),(U32.from_nat(channel) % 8 : U32)) : F32}
      {==}

# The value at ((before * lanes + channel) * after + offset) is lane
# channel % 8 of the cell the pick of that channel and before index reads.
def cells_lane(+tree: Tree.Tree<Tile.Tile8>,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+position: Grid.Position,+before: Nat,+channel: Nat,+offset: Nat,
  after_facts: Decoded<after,Nat.add(Nat.mul(before,U32.to_nat(lanes)),channel),offset>,lane_facts: Decoded<lanes,before,channel>,
  +positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) ->
  {value_value(tree,Cells.Cells{outer,after,lanes,groups,position},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(before,U32.to_nat(lanes)),channel),U32.to_nat(after)),offset))) ==
    Cells.lane_of(Tree.read(Tile.Tile8,tree,cells_address(position,outer,after,groups,before,channel,offset)),(U32.from_nat(channel) % 8 : U32)) : F32}:
  Decoded{+after_quotient,+after_remainder,+local} = after_facts
  Decoded{+quotient,+remainder,lane_local} = lane_facts
  +block = Nat.add(Nat.mul(before,U32.to_nat(lanes)),channel)
  +at = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(after)),offset))
  +start = U32.from_nat(Nat.mul(block,U32.to_nat(after)))
  +context = {Cells.Cells{outer,after,lanes,groups,position} : Cells.Cells}
  +wanted = Cells.lane_of(Tree.read(Tile.Tile8,tree,cells_address(position,outer,after,groups,before,channel,offset)),(U32.from_nat(channel) % 8 : U32))
  Equal.trans(F32,value_value(tree,context,at),pick_value(tree,Cells.channel_pick(context,U32.from_nat(block),start),at),wanted,
    Equal.sym(F32,pick_value(tree,Cells.channel_pick(context,U32.from_nat(block),start),at),value_value(tree,context,at),
      pick_decoded(tree,outer,after,lanes,groups,position,block,offset,Decoded{after_quotient,after_remainder,local})),
    %Equal.sym(U32,U32.max(lanes,1),lanes,Depth.positive_max(lanes,positive)) :
      {pick_value(tree,Cells.picked(position,outer,after,groups,(U32.from_nat(block) / _ : U32),(U32.from_nat(block) % _ : U32),start),at) == wanted : F32}
    %Equal.sym(U32,(U32.from_nat(block) / lanes : U32),U32.from_nat(before),quotient) :
      {pick_value(tree,Cells.picked(position,outer,after,groups,_,(U32.from_nat(block) % lanes : U32),start),at) == wanted : F32}
    %Equal.sym(U32,(U32.from_nat(block) % lanes : U32),U32.from_nat(channel),remainder) :
      {pick_value(tree,Cells.picked(position,outer,after,groups,U32.from_nat(before),_,start),at) == wanted : F32}
    cells_pick(tree,outer,after,groups,position,before,channel,offset,start,at,local))

# The same for a before index below images of values that fit the capacity.
def cells_lane_within(+limits: Machine.Limits,+tree: Tree.Tree<Tile.Tile8>,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+position: Grid.Position,
  +images: Nat,+before: Nat,+channel: Nat,+offset: Nat,
  +room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),U32.to_nat(after)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +before_inside: {Nat.is_lt(before,images) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(after)) == True{} : Bool}) ->
  {value_value(tree,Cells.Cells{outer,after,lanes,groups,position},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(before,U32.to_nat(lanes)),channel),U32.to_nat(after)),offset))) ==
    Cells.lane_of(Tree.read(Tile.Tile8,tree,cells_address(position,outer,after,groups,before,channel,offset)),(U32.from_nat(channel) % 8 : U32)) : F32}:
  +l = U32.to_nat(lanes)
  +a = U32.to_nat(after)
  +capacity = Machine.capacity(limits)
  +total = Nat.mul(Nat.mul(images,l),a)
  +block = Nat.add(Nat.mul(before,l),channel)
  +block_inside = RowMajor.radix(before,images,channel,l,before_inside,channel_inside)
  +after_room = Order.transitive(Nat.add(Nat.mul(block,a),a),total,U32.to_nat(capacity),row_room(block,Nat.mul(images,l),a,total,{==},block_inside),room)
  +after_positive = Order.le_lt_trans(0n,offset,a,Order.zero_le(offset),offset_inside)
  +lane_room = Order.transitive(Nat.add(Nat.mul(before,l),l),Nat.mul(images,l),U32.to_nat(capacity),row_room(before,images,l,Nat.mul(images,l),{==},before_inside),
    Order.transitive(Nat.mul(images,l),total,U32.to_nat(capacity),RowMajor.product_covers(Nat.mul(images,l),a,after_positive),room))
  +positive = Order.le_lt_trans(0n,channel,l,Order.zero_le(channel),channel_inside)
  +bounded = {%Equal.sym(Bool,U32.is_le(capacity,capacity),Nat.is_le(U32.to_nat(capacity),U32.to_nat(capacity)),Comparison.le(capacity,capacity)) : {_ == True{} : Bool}
    Order.reflexive(U32.to_nat(capacity)) : {U32.is_le(capacity,capacity) == True{} : Bool}}
  cells_lane(tree,outer,after,lanes,groups,position,before,channel,offset,
    decoded(limits,capacity,after,block,offset,offset_inside,after_room,bounded),decoded(limits,capacity,lanes,before,channel,channel_inside,lane_room,bounded),positive)

# ---- Cells of lifted values ----

# The cell a value pick reads, as a word: the cell index of the outer index
# and the group of the value.
def address_word(position: Grid.Position,+outer: U32,+after: U32,+groups: U32,+before: Nat,+block: Nat,+channel: Nat,+offset: Nat,
  +block_word: {U32.from_nat(block) == (U32.from_nat(channel) / 8 : U32) : U32}) ->
  {cells_address(position,outer,after,groups,before,channel,offset) ==
    U32.from_nat(Representation.index_at(position,U32.to_nat(outer),U32.to_nat(groups),Nat.add(Nat.mul(before,U32.to_nat(after)),offset),block)) : U32}:
  match position:
    case Grid.Innermost{}:
      +a = U32.to_nat(after)
      +g = U32.to_nat(groups)
      +x = Nat.mul(Nat.mul(before,a),g)
      +z = Nat.mul(offset,g)
      +target = Nat.add(Nat.mul(Nat.add(Nat.mul(before,a),offset),g),block)
      +sums = {%Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(before,a),offset),g),Nat.add(x,z),RowMajor.distrib(Nat.mul(before,a),offset,g)) :
          {Nat.add(Nat.add(x,block),z) == Nat.add(_,block) : Nat}
        Equal.trans(Nat,Nat.add(Nat.add(x,block),z),Nat.add(x,Nat.add(block,z)),Nat.add(Nat.add(x,z),block),
          Addition.associative(x,block,z),
          Equal.trans(Nat,Nat.add(x,Nat.add(block,z)),Nat.add(x,Nat.add(z,block)),Nat.add(Nat.add(x,z),block),
            Equal.cong(Nat,Nat,n => Nat.add(x,n),Nat.add(block,z),Nat.add(z,block),Order.commutative(block,z)),
            Equal.sym(Nat,Nat.add(Nat.add(x,z),block),Nat.add(x,Nat.add(z,block)),Addition.associative(x,z,block)))) :
        {Nat.add(Nat.add(x,block),z) == target : Nat}}
      %block_word : {((U32.from_nat(before) * after * groups + _ : U32) + U32.from_nat(offset) * groups : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(before) * after : U32),U32.from_nat(Nat.mul(before,a)),Encoded.scale_encoded(before,after)) :
        {((_ * groups + U32.from_nat(block) : U32) + U32.from_nat(offset) * groups : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.mul(before,a)) * groups : U32),U32.from_nat(x),Encoded.scale_encoded(Nat.mul(before,a),groups)) :
        {((_ + U32.from_nat(block) : U32) + U32.from_nat(offset) * groups : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(offset) * groups : U32),U32.from_nat(z),Encoded.scale_encoded(offset,groups)) :
        {((U32.from_nat(x) + U32.from_nat(block) : U32) + _ : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(x) + U32.from_nat(block) : U32),U32.from_nat(Nat.add(x,block)),WordArithmetic.native_add(x,block)) :
        {(_ + U32.from_nat(z) : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.add(x,block)) + U32.from_nat(z) : U32),U32.from_nat(Nat.add(Nat.add(x,block),z)),WordArithmetic.native_add(Nat.add(x,block),z)) :
        {_ == U32.from_nat(target) : U32}
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(x,block),z),target,sums)
    case Grid.Outermost{}:
      +a = U32.to_nat(after)
      +o = U32.to_nat(outer)
      +x = Nat.mul(block,o)
      +y = Nat.mul(before,a)
      +target = Nat.add(x,Nat.add(y,offset))
      %block_word : {((_ * outer + U32.from_nat(before) * after : U32) + U32.from_nat(offset) * 1 : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(block) * outer : U32),U32.from_nat(x),Encoded.scale_encoded(block,outer)) :
        {((_ + U32.from_nat(before) * after : U32) + U32.from_nat(offset) * 1 : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(before) * after : U32),U32.from_nat(y),Encoded.scale_encoded(before,after)) :
        {((U32.from_nat(x) + _ : U32) + U32.from_nat(offset) * 1 : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(offset) * 1 : U32),U32.from_nat(offset),Arithmetic.multiply_one(U32.from_nat(offset))) :
        {((U32.from_nat(x) + U32.from_nat(y) : U32) + _ : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(x) + U32.from_nat(y) : U32),U32.from_nat(Nat.add(x,y)),WordArithmetic.native_add(x,y)) :
        {(_ + U32.from_nat(offset) : U32) == U32.from_nat(target) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.add(x,y)) + U32.from_nat(offset) : U32),U32.from_nat(Nat.add(Nat.add(x,y),offset)),WordArithmetic.native_add(Nat.add(x,y),offset)) :
        {_ == U32.from_nat(target) : U32}
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.add(x,y),offset),target,Addition.associative(x,y,offset))

# Cells that hold the lifted values of a reader give, at a value index, the
# value of the reader there.
def lifted_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+tree: Tree.Tree<Tile.Tile8>,
  +outer: U32,+after: U32,+lanes: U32,+groups: U32,+position: Grid.Position,+images: Nat,+before: Nat,+block: Nat,+lane: Nat,+offset: Nat,
  +room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),U32.to_nat(after)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(images,U32.to_nat(after)) : Nat},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_cover: {Nat.is_le(U32.to_nat(lanes),Nat.mul(U32.to_nat(groups),8n)) == True{} : Bool},
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool} ->
    {Tree.read(Tile.Tile8,tree,word) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,lanes,groups,position}},word) : Tile.Tile8},
  +before_inside: {Nat.is_lt(before,images) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),U32.to_nat(lanes)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(after)) == True{} : Bool}) ->
  {value_value(tree,Cells.Cells{outer,after,lanes,groups,position},
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(before,U32.to_nat(lanes)),Nat.add(Nat.mul(block,8n),lane)),U32.to_nat(after)),offset))) ==
    original(logical,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(before,U32.to_nat(lanes)),Nat.add(Nat.mul(block,8n),lane)),U32.to_nat(after)),offset))) : F32}:
  +l = U32.to_nat(lanes)
  +a = U32.to_nat(after)
  +g = U32.to_nat(groups)
  +o = U32.to_nat(outer)
  +cap = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +channel = Nat.add(Nat.mul(block,8n),lane)
  +turned = Nat.add(lane,Nat.mul(block,8n))
  +at = Nat.add(Nat.mul(before,a),offset)
  +geometry = {Cells.Cells{outer,after,lanes,groups,position} : Cells.Cells}
  +index = Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),channel),a),offset)
  +wanted = original(logical,U32.from_nat(index))
  +after_positive = Order.le_lt_trans(0n,offset,a,Order.zero_le(offset),offset_inside)
  +images_positive = Order.le_lt_trans(0n,before,images,Order.zero_le(before),before_inside)
  +at_inside = {%Equal.sym(Nat,o,Nat.mul(images,a),outer_exact) : {Nat.is_lt(at,_) == True{} : Bool}
    RowMajor.radix(before,images,offset,a,before_inside,offset_inside) : {Nat.is_lt(at,o) == True{} : Bool}}
  +group_inside = {%Radix.quotient(block,lane,8n,{==},lane_inside) : {Nat.is_lt(_,g) == True{} : Bool}
    DivisionOrder.quotient_below(channel,g,8n,{==},Bounds.nat_trans(channel,l,Nat.mul(g,8n),channel_inside,groups_cover)) : {Nat.is_lt(block,g) == True{} : Bool}}
  +lanes_room = Order.transitive(l,Nat.mul(images,l),U32.to_nat(cap),DivisionOrder.le_multiple(images,l,images_positive),
    Order.transitive(Nat.mul(images,l),Nat.mul(Nat.mul(images,l),a),U32.to_nat(cap),RowMajor.product_covers(Nat.mul(images,l),a,after_positive),room))
  +channel_exact = Bounds.roundtrip(channel,cap,fits,Order.transitive(channel,l,U32.to_nat(cap),Bounds.strict_le(channel,l,channel_inside),lanes_room))
  +channel_word = U32.from_nat(channel)
  +block_nat = Equal.trans(Nat,U32.to_nat((channel_word / 8 : U32)),Nat.div(U32.to_nat(channel_word),8n),block,
    NativeDivision.divide(channel_word,8,16,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(channel_word),channel,channel_exact) : {Nat.div(_,8n) == block : Nat} Radix.quotient(block,lane,8n,{==},lane_inside))
  +lane_nat = Equal.trans(Nat,U32.to_nat((channel_word % 8 : U32)),Nat.mod(U32.to_nat(channel_word),8n),lane,
    NativeDivision.modulo(channel_word,8,16,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(channel_word),channel,channel_exact) : {Nat.mod(_,8n) == lane : Nat} Radix.remainder(block,lane,8n,{==},lane_inside))
  +block_word = Arithmetic.from_nat_exact((channel_word / 8 : U32),block,block_nat)
  +lane_word = Arithmetic.from_nat_exact((channel_word % 8 : U32),lane,lane_nat)
  +address = cells_address(position,outer,after,groups,before,channel,offset)
  +cell_number = Representation.index_at(position,o,g,at,block)
  +number_inside = Representation.index_inside(position,o,g,at,block,at_inside,group_inside)
  +address_exact = Equal.trans(Nat,U32.to_nat(address),U32.to_nat(U32.from_nat(cell_number)),cell_number,
    Equal.cong(U32,Nat,word => U32.to_nat(word),address,U32.from_nat(cell_number),address_word(position,outer,after,groups,before,block,channel,offset,block_word)),
    Bounds.roundtrip(cell_number,cap,fits,Order.transitive(cell_number,Nat.mul(g,o),U32.to_nat(cap),Bounds.strict_le(cell_number,Nat.mul(g,o),number_inside),cells_room)))
  +address_inside = {%Equal.sym(Nat,U32.to_nat(address),cell_number,address_exact) : {Nat.is_lt(_,Nat.mul(g,o)) == True{} : Bool}
    number_inside : {Nat.is_lt(U32.to_nat(address),Nat.mul(g,o)) == True{} : Bool}}
  +turned_inside = {%Order.commutative(Nat.mul(block,8n),lane) : {Nat.is_lt(_,l) == True{} : Bool} channel_inside : {Nat.is_lt(turned,l) == True{} : Bool}}
  +index_exact = {%Equal.sym(Nat,Representation.least_one(a),a,Representation.least_positive(a,after_positive)) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(at,_),l),turned),_),Nat.mod(at,_)) == index : Nat}
    %Equal.sym(Nat,Nat.div(at,a),before,Radix.quotient(before,offset,a,after_positive,offset_inside)) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(_,l),turned),a),Nat.mod(at,a)) == index : Nat}
    %Equal.sym(Nat,Nat.mod(at,a),offset,Radix.remainder(before,offset,a,after_positive,offset_inside)) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),turned),a),_) == index : Nat}
    %Equal.sym(Nat,turned,channel,Order.commutative(lane,Nat.mul(block,8n))) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),_),a),offset) == index : Nat}
    {==} : {Representation.logical_index(geometry,at,turned) == index : Nat}}
  +lane_value = {%Equal.sym(Bool,Nat.is_lt(turned,l),True{},turned_inside) :
      {Bool.pick(F32,_,original(logical,U32.from_nat(Representation.logical_index(geometry,at,turned))),0.0) == wanted : F32}
    Equal.cong(Nat,F32,number => original(logical,U32.from_nat(number)),Representation.logical_index(geometry,at,turned),index,index_exact) :
    {Representation.lane(~Logical,~original,logical,geometry,at,block,lane) == wanted : F32}}
  +cell_value = {%Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,tree,address),Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},address),reads(address,address_inside)) :
      {TileLaws.select_lane(lane,_) == wanted : F32}
    %Equal.sym(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},address),Representation.cell(~Logical,~original,logical,geometry,at,block),
        Representation.decoded(~Logical,~original,logical,geometry,at,block,address,address_exact,at_inside,group_inside)) :
      {TileLaws.select_lane(lane,_) == wanted : F32}
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,geometry,at,block)),Representation.lane(~Logical,~original,logical,geometry,at,block,lane),wanted,
      Representation.cell_lane(~Logical,~original,logical,geometry,at,block,lane,Division.lt_succ_le(lane,7n,lane_inside)),lane_value) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,address)) == wanted : F32}}
  Equal.trans(F32,value_value(tree,geometry,U32.from_nat(index)),Cells.lane_of(Tree.read(Tile.Tile8,tree,address),(channel_word % 8 : U32)),wanted,
    cells_lane_within(limits,tree,outer,after,lanes,groups,position,images,before,channel,offset,room,before_inside,channel_inside,offset_inside),
    {%lane_word : {Cells.lane_of(Tree.read(Tile.Tile8,tree,address),_) == wanted : F32}
      Equal.trans(F32,Cells.lane_of(Tree.read(Tile.Tile8,tree,address),U32.from_nat(lane)),TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,address)),wanted,
        lane_selected(Tree.read(Tile.Tile8,tree,address),lane,lane_inside),cell_value) :
      {Cells.lane_of(Tree.read(Tile.Tile8,tree,address),(channel_word % 8 : U32)) == wanted : F32}})

# Cells that hold the lifted values of a reader give at every value index
# below the values the value of the reader there.
def lifted_value_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+tree: Tree.Tree<Tile.Tile8>,
  +outer: U32,+after: U32,+lanes: U32,+groups: U32,+position: Grid.Position,+images: Nat,
  +room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),U32.to_nat(after)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(images,U32.to_nat(after)) : Nat},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_cover: {Nat.is_le(U32.to_nat(lanes),Nat.mul(U32.to_nat(groups),8n)) == True{} : Bool},
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool} ->
    {Tree.read(Tile.Tile8,tree,word) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,lanes,groups,position}},word) : Tile.Tile8},
  +index: Nat,+inside: {Nat.is_lt(index,Nat.mul(Nat.mul(images,U32.to_nat(lanes)),U32.to_nat(after))) == True{} : Bool}) ->
  {value_value(tree,Cells.Cells{outer,after,lanes,groups,position},U32.from_nat(index)) == original(logical,U32.from_nat(index)) : F32}:
  +l = U32.to_nat(lanes)
  +a = U32.to_nat(after)
  +total_positive = Order.le_lt_trans(0n,index,Nat.mul(Nat.mul(images,l),a),Order.zero_le(index),inside)
  +after_positive = DivisionOrder.right_factor_positive(Nat.mul(images,l),a,total_positive)
  +lanes_positive = DivisionOrder.right_factor_positive(images,l,RowMajor.left_factor_positive(Nat.mul(images,l),a,total_positive))
  +rows = Nat.div(index,a)
  +offset = Nat.mod(index,a)
  +before = Nat.div(rows,l)
  +channel = Nat.mod(rows,l)
  +block = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +parts = {%DivisionOrder.quotient_exact(channel,8n,{==}) : {index == Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),_),a),offset) : Nat}
    %DivisionOrder.quotient_exact(rows,l,lanes_positive) : {index == Nat.add(Nat.mul(_,a),offset) : Nat}
    DivisionOrder.quotient_exact(index,a,after_positive) :
    {index == Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),Nat.add(Nat.mul(block,8n),lane)),a),offset) : Nat}}
  +rows_inside = DivisionOrder.quotient_below(index,Nat.mul(images,l),a,after_positive,inside)
  +channel_inside = {%DivisionOrder.quotient_exact(channel,8n,{==}) : {Nat.is_lt(_,l) == True{} : Bool}
    DivisionOrder.remainder_small(rows,l,lanes_positive) : {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),l) == True{} : Bool}}
  %Equal.sym(Nat,index,Nat.add(Nat.mul(Nat.add(Nat.mul(before,l),Nat.add(Nat.mul(block,8n),lane)),a),offset),parts) :
    {value_value(tree,Cells.Cells{outer,after,lanes,groups,position},U32.from_nat(_)) == original(logical,U32.from_nat(_)) : F32}
  lifted_value(~Logical,~original,logical,limits,tree,outer,after,lanes,groups,position,images,before,block,lane,offset,room,outer_exact,cells_room,groups_cover,reads,
    DivisionOrder.quotient_below(rows,images,l,lanes_positive,rows_inside),DivisionOrder.remainder_small(channel,8n,{==}),channel_inside,
    DivisionOrder.remainder_small(index,a,after_positive))

# ---- Groups between images and planes ----

# The groups of channels channels.
def groups_of(+channels: Nat) -> Nat:
  Ceiling.ceiling(channels,8n)

# Cells [images][groups][plane] of channels channels: value (image * channels
# + channel) * plane + offset is lane channel % 8 of the cell of group
# channel / 8 at offset.
type CellSource is Data:
  CellSource{cells: Tree.Tree<Tile.Tile8>,channels: Nat,plane: Nat}

def cell_value(source: CellSource,index: Nat) -> F32:
  CellSource{cells,+channels,+plane} = source
  +index = index
  +block = Nat.div(index,plane)
  +channel = Nat.mod(block,channels)
  TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,
    U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(block,channels),groups_of(channels)),Nat.div(channel,8n)),plane),Nat.mod(index,plane)))))

# Lane lane of the cell of group block at offset is the value of channel
# 8 * block + lane there.
def cell_lane_at(+cells: Tree.Tree<Tile.Tile8>,+channels: Nat,+plane: Nat,+image: Nat,+block: Nat,+lane: Nat,+offset: Nat,
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+channel_inside: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),channels) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups_of(channels)),block),plane),offset)))) ==
    cell_value(CellSource{cells,channels,plane},Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),Nat.add(Nat.mul(block,8n),lane)),plane),offset)) : F32}:
  +channel = Nat.add(Nat.mul(block,8n),lane)
  +groups = groups_of(channels)
  +prefix = Nat.add(Nat.mul(image,channels),channel)
  +plane_positive = Order.le_lt_trans(0n,offset,plane,Order.zero_le(offset),offset_inside)
  +channels_positive = Order.le_lt_trans(0n,channel,channels,Order.zero_le(channel),channel_inside)
  +eight_positive = {{==} : {Nat.is_lt(0n,8n) == True{} : Bool}}
  +wanted = TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups),block),plane),offset))))
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(prefix,plane),offset),plane),prefix,Radix.quotient(prefix,offset,plane,plane_positive,offset_inside)) :
    {wanted == TileLaws.select_lane(Nat.mod(Nat.mod(_,channels),8n),Tree.read(Tile.Tile8,cells,
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(_,channels),groups),Nat.div(Nat.mod(_,channels),8n)),plane),Nat.mod(Nat.add(Nat.mul(prefix,plane),offset),plane))))) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(prefix,plane),offset),plane),offset,Radix.remainder(prefix,offset,plane,plane_positive,offset_inside)) :
    {wanted == TileLaws.select_lane(Nat.mod(Nat.mod(prefix,channels),8n),Tree.read(Tile.Tile8,cells,
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(prefix,channels),groups),Nat.div(Nat.mod(prefix,channels),8n)),plane),_)))) : F32}
  %Equal.sym(Nat,Nat.div(prefix,channels),image,Radix.quotient(image,channel,channels,channels_positive,channel_inside)) :
    {wanted == TileLaws.select_lane(Nat.mod(Nat.mod(prefix,channels),8n),Tree.read(Tile.Tile8,cells,
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(_,groups),Nat.div(Nat.mod(prefix,channels),8n)),plane),offset)))) : F32}
  %Equal.sym(Nat,Nat.mod(prefix,channels),channel,Radix.remainder(image,channel,channels,channels_positive,channel_inside)) :
    {wanted == TileLaws.select_lane(Nat.mod(_,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups),Nat.div(_,8n)),plane),offset)))) : F32}
  %Equal.sym(Nat,Nat.div(channel,8n),block,Radix.quotient(block,lane,8n,eight_positive,small)) :
    {wanted == TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups),_),plane),offset)))) : F32}
  %Equal.sym(Nat,Nat.mod(channel,8n),lane,Radix.remainder(block,lane,8n,eight_positive,small)) :
    {wanted == TileLaws.select_lane(_,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups),block),plane),offset)))) : F32}
  {==}

def cell_channel_at(+cells: Tree.Tree<Tile.Tile8>,+channels: Nat,+plane: Nat,+image: Nat,+channel: Nat,+offset: Nat,
  +channel_inside: {Nat.is_lt(channel,channels) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups_of(channels)),Nat.div(channel,8n)),plane),offset)))) ==
    cell_value(CellSource{cells,channels,plane},Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),offset)) : F32}:
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +split = DivisionOrder.quotient_exact(channel,8n,{==})
  %Equal.sym(Nat,channel,Nat.add(Nat.mul(group,8n),lane),split) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,groups_of(channels)),group),plane),offset)))) ==
      cell_value(CellSource{cells,channels,plane},Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),_),plane),offset)) : F32}
  cell_lane_at(cells,channels,plane,image,group,lane,offset,DivisionOrder.remainder_small(channel,8n,{==}),
    %split : {Nat.is_lt(_,channels) == True{} : Bool} channel_inside,offset_inside)

# Cells of the geometry of values whose split and cover are known, that read
# the lifted values of a reader, give at every value index the value of the
# reader.
def covered_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+tree: Tree.Tree<Tile.Tile8>,
  +values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +split: Public.Split<limits,values,axis,Grid.lanes_at(values,axis)>,
  +cover: Public.Cover<limits,Grid.lanes_at(values,axis),Grid.outer_count(Tiled.cell_shape(values,axis,position),position),position,
    (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),position,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool} ->
    {Tree.read(Tile.Tile8,tree,word) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,
      Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
        Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position}},word) : Tile.Tile8},
  +index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(values)) == True{} : Bool}) ->
  {value_value(tree,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
    Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position},U32.from_nat(index)) == original(logical,U32.from_nat(index)) : F32}:
  Public.Split{+blocks,+outer_exact,+elements_exact,+lanes_within,+after_positive} = split
  Public.Cover{+counted,+cells_exact,+padded_cells,+placement} = cover
  +lanes = Grid.lanes_at(values,axis)
  +after = Tiled.product(Public.lane_after(values,axis))
  +outer = (Public.lane_prefix(values,axis,1) * after : U32)
  +groups = Tiled.groups(lanes)
  +l = U32.to_nat(lanes)
  +a = U32.to_nat(after)
  +g = U32.to_nat(groups)
  +o = U32.to_nat(outer)
  +cap = U32.to_nat(Machine.capacity(limits))
  +least = Representation.least_positive(a,after_positive)
  +outer_nat = {%least : {o == Nat.mul(blocks,_) : Nat} outer_exact : {o == Nat.mul(blocks,a) : Nat}}
  +elements_nat = {%least : {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,l),_) : Nat} elements_exact : {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,l),a) : Nat}}
  +total_room = {%elements_nat : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.mul(Nat.mul(blocks,l),a),cap) == True{} : Bool}}
  +index_inside = {%elements_nat : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,Nat.mul(Nat.mul(blocks,l),a)) == True{} : Bool}}
  +total_positive = {%elements_nat : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,Nat.mul(Nat.mul(blocks,l),a)) == True{} : Bool}}
  +lanes_positive = DivisionOrder.right_factor_positive(blocks,l,RowMajor.left_factor_positive(Nat.mul(blocks,l),a,total_positive))
  +groups_exact = Encoded.eight_blocks_exact(limits,lanes,lanes_within)
  +groups_cover = {%Equal.sym(Nat,g,Ceiling.ceiling(l,8n),groups_exact) : {Nat.is_le(l,Nat.mul(_,8n)) == True{} : Bool} Ceiling.covers(l,8n,{==}) :
    {Nat.is_le(l,Nat.mul(g,8n)) == True{} : Bool}}
  +groups_below = {%Equal.sym(Nat,g,Ceiling.ceiling(l,8n),groups_exact) : {Nat.is_le(_,l) == True{} : Bool} Ceiling.minimal(l,8n,l,{==},RowMajor.product_covers(l,8n,{==})) :
    {Nat.is_le(g,l) == True{} : Bool}}
  +cells_room = Order.transitive(Nat.mul(g,o),Nat.mul(l,o),cap,RowMajor.mul_le(g,l,o,groups_below),counted)
  lifted_value_at(~Logical,~original,logical,limits,tree,outer,after,lanes,groups,position,blocks,total_room,outer_nat,cells_room,groups_cover,
    word => inside_word => reads(word,%cells_exact : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} inside_word),index,index_inside)

# Cells of the geometry of counted values that read the lifted values of a
# reader give at every value index the value of the reader.
def geometry_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+tree: Tree.Tree<Tile.Tile8>,
  +values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool} ->
    {Tree.read(Tile.Tile8,tree,word) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word) : Tile.Tile8},
  +index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(values)) == True{} : Bool}) ->
  {value_value(tree,Public.geometry_of(values,axis,position),U32.from_nat(index)) == original(logical,U32.from_nat(index)) : F32}:
  +lanes = Grid.lanes_at(values,axis)
  +cells = {Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
    lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +exact = Public.geometry_exact(values,axis,position,Bounds.strict_le(axis,List.length(&2,U32,values),within))
  +count = Global.count(values,counted)
  +count_exact = Global.whole_counted(values,count,Global.accepted(values,counted))
  +capacity = Machine.capacity(limits)
  +bounded = {%Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)),Comparison.le(count,capacity)) : {_ == True{} : Bool}
    %count_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room : {U32.is_le(count,capacity) == True{} : Bool}}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),cells,exact) :
    {value_value(tree,_,U32.from_nat(index)) == original(logical,U32.from_nat(index)) : F32}
  covered_value(~Logical,~original,logical,limits,tree,values,axis,position,Public.values_split(limits,values,axis,lanes,{==},room,nonempty),
    Public.along_cover(limits,values,axis,lanes,position,count,count_exact,bounded,within,{==}),room,nonempty,
    word => inside_word => %exact : {Tree.read(Tile.Tile8,tree,word) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},word) : Tile.Tile8} reads(word,inside_word),
    index,inside)
