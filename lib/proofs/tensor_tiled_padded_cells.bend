# The cells of tiled values after padding. A leaf puts +0 back in the lanes
# past the lane count; a cell whose lanes agree with a target below the count,
# the target holding +0 past it, then holds the target.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_map.bend as TiledMap
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../kernel_gemm.bend as KernelGemm
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_content.bend as Content
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./kernel_tile_refinement.bend as TileLaws
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as Ceiling
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as WordMultiplication
import ./binary_radix.bend as Radix
import ./word_encoding.bend as WordEncoding

# ---- Lanes past the count ----

# Lane lane of a cell of group group: its value below the lane count, +0 past
# it.
def kept(+lanes: Nat,+group: Nat,+lane: Nat,value: F32) -> F32:
  Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes),value,0.0)

# The cell of group group whose lanes below the lane count hold the value at
# their index, and +0 past it.
def kept_tile(~value: Nat -> F32,+lanes: Nat,+group: Nat) -> Tile.Tile8:
  Tile.Tile{kept(lanes,group,0n,value(Nat.add(0n,Nat.mul(group,8n)))),kept(lanes,group,1n,value(Nat.add(1n,Nat.mul(group,8n)))),
    kept(lanes,group,2n,value(Nat.add(2n,Nat.mul(group,8n)))),kept(lanes,group,3n,value(Nat.add(3n,Nat.mul(group,8n)))),
    kept(lanes,group,4n,value(Nat.add(4n,Nat.mul(group,8n)))),kept(lanes,group,5n,value(Nat.add(5n,Nat.mul(group,8n)))),
    kept(lanes,group,6n,value(Nat.add(6n,Nat.mul(group,8n)))),kept(lanes,group,7n,value(Nat.add(7n,Nat.mul(group,8n))))}

def lane_inside(+lanes: Nat,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +full: {Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),lanes) == True{} : Bool}) -> {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes) == True{} : Bool}:
  Order.le_lt_trans(Nat.add(lane,Nat.mul(group,8n)),Nat.add(7n,Nat.mul(group,8n)),lanes,Order.add_le(lane,7n,Nat.mul(group,8n),small),full)

# Two values kept on a lane agree when they agree below the lane count.
def kept_agree(+lanes: Nat,+group: Nat,+lane: Nat,+left: F32,+right: F32,below: Bool,
  +below_exact: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes) == below : Bool},
  same: @+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes) == True{} : Bool} -> {left == right : F32}) ->
  {kept(lanes,group,lane,left) == kept(lanes,group,lane,right) : F32}:
  match below:
    case True{}:
      Equal.cong(F32,F32,value => kept(lanes,group,lane,value),left,right,same(below_exact))
    case False{}:
      Equal.trans(F32,kept(lanes,group,lane,left),0.0,kept(lanes,group,lane,right),
        Equal.cong(Bool,F32,state => Bool.pick(F32,state,left,0.0),Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes),False{},below_exact),
        Equal.cong(Bool,F32,state => Bool.pick(F32,state,right,0.0),False{},Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes),
          Equal.sym(Bool,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes),False{},below_exact)))

# Below the lane count two kept values agree when the values do.
def kept_inside(+lanes: Nat,+group: Nat,+lane: Nat,+left: F32,+right: F32,
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes) == True{} : Bool},
  +agree: {kept(lanes,group,lane,left) == kept(lanes,group,lane,right) : F32}) -> {left == right : F32}:
  +condition = Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),lanes)
  Equal.trans(F32,left,Bool.pick(F32,condition,left,0.0),right,
    Equal.cong(Bool,F32,state => Bool.pick(F32,state,left,0.0),True{},condition,Equal.sym(Bool,condition,True{},inside)),
    Equal.trans(F32,Bool.pick(F32,condition,left,0.0),Bool.pick(F32,condition,right,0.0),right,agree,
      Equal.cong(Bool,F32,state => Bool.pick(F32,state,right,0.0),condition,True{},inside)))

# The runtime compares the lane past the first lane of its group as a word;
# below half the word it compares as its number.
def lane_compare(+limits: Machine.Limits,+first: U32,+lanes: U32,+group: Nat,+lane: Nat,
  +first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),lane),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.is_lt((first + U32.from_nat(lane) : U32),lanes) == Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) : Bool}:
  +sum = Nat.add(Nat.mul(group,8n),lane)
  +lane_word = Bounds.roundtrip(lane,Machine.half_word(limits),Limits.half_word_fits(limits),
    Order.transitive(lane,sum,U32.to_nat(Machine.half_word(limits)),%Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_le(lane,_) == True{} : Bool} Order.le_add(lane,Nat.mul(group,8n)),bound))
  +added = Arithmetic.add(first,U32.from_nat(lane),Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(group,8n),first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(lane))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_word) : {Nat.is_le(Nat.add(Nat.mul(group,8n),_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} bound)
  Equal.trans(Bool,U32.is_lt((first + U32.from_nat(lane) : U32),lanes),Nat.is_lt(U32.to_nat((first + U32.from_nat(lane) : U32)),U32.to_nat(lanes)),
    Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)),Comparison.lt((first + U32.from_nat(lane) : U32),lanes),
    Equal.cong(Nat,Bool,value => Nat.is_lt(value,U32.to_nat(lanes)),U32.to_nat((first + U32.from_nat(lane) : U32)),Nat.add(lane,Nat.mul(group,8n)),
      Equal.trans(Nat,U32.to_nat((first + U32.from_nat(lane) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(U32.from_nat(lane))),Nat.add(lane,Nat.mul(group,8n)),added,
        %Equal.sym(Nat,U32.to_nat(first),Nat.mul(group,8n),first_exact) : {Nat.add(_,U32.to_nat(U32.from_nat(lane))) == Nat.add(lane,Nat.mul(group,8n)) : Nat}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_word) : {Nat.add(Nat.mul(group,8n),_) == Nat.add(lane,Nat.mul(group,8n)) : Nat}
        Order.commutative(Nat.mul(group,8n),lane))))

def tail_lane(+value: F32,+limits: Machine.Limits,+first: U32,+lanes: U32,+group: Nat,+lane: Nat,
  +first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),lane),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Cells.tail_lane(first,lanes,U32.from_nat(lane),value) == kept(U32.to_nat(lanes),group,lane,value) : F32}:
  Equal.cong(Bool,F32,valid => Bool.pick(F32,valid,value,0.0),U32.is_lt((first + U32.from_nat(lane) : U32),lanes),Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)),
    lane_compare(limits,first,lanes,group,lane,first_exact,small,bound))

def lane_bound(+limits: Machine.Limits,+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(group,8n),lane),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(group,8n),lane),Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits)),
    %Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_le(_,Nat.add(Nat.mul(group,8n),7n)) == True{} : Bool}
    %Order.commutative(7n,Nat.mul(group,8n)) : {Nat.is_le(Nat.add(lane,Nat.mul(group,8n)),_) == True{} : Bool}
    Order.add_le(lane,7n,Nat.mul(group,8n),small),bound)

# ---- Groups ----

# A lane count that is not a multiple of eight is positive.
def unaligned_selected(zero: Bool,+lanes: U32,+classified: {U32.is_zero(lanes) == zero : Bool},+unaligned: {U32.is_zero((lanes % 8 : U32)) == False{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}:
  match zero:
    case True{}:
      +empty = Arithmetic.word_zero(lanes,classified)
      Empty.absurd({Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},{%empty : {U32.is_zero((_ % 8 : U32)) == False{} : Bool} unaligned : {U32.is_zero((0 % 8 : U32)) == False{} : Bool}})))
    case False{}: Arithmetic.word_nonzero_positive(lanes,classified)

def unaligned_positive(+lanes: U32,+unaligned: {U32.is_zero((lanes % 8 : U32)) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}:
  unaligned_selected(U32.is_zero(lanes),lanes,{==},unaligned)

# The first lane of the last group lies below a positive count.
def last_below(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n),8n),U32.to_nat(lanes)) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes_fit} = facts
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),exact) : {Nat.is_lt(Nat.mul(Nat.sub(_,1n),8n),U32.to_nat(lanes)) == True{} : Bool}
  Ceiling.previous_below(U32.to_nat(lanes),8n,positive,{==})

def step_below(+group: Nat) -> {Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),Nat.mul(1n+group,8n)) == True{} : Bool}:
  Order.lt_succ(Nat.add(7n,Nat.mul(group,8n)),Nat.add(7n,Nat.mul(group,8n)),Order.reflexive(Nat.add(7n,Nat.mul(group,8n))))

# Every lane of a group before the last lies below the count.
def inner_full(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},
  +group: Nat,+below: {Nat.is_lt(group,Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool}:
  +last = Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)
  Bounds.nat_trans(Nat.add(7n,Nat.mul(group,8n)),Nat.mul(1n+group,8n),U32.to_nat(lanes),step_below(group),
    Order.transitive(Nat.mul(1n+group,8n),Nat.mul(last,8n),U32.to_nat(lanes),RowMajor.mul_le(1n+group,last,8n,Order.strict_follow(group,last,below)),
      Bounds.strict_le(Nat.mul(last,8n),U32.to_nat(lanes),last_below(limits,lanes,facts,positive))))

# Every lane of every group lies below a count that is a multiple of eight.
def aligned_full(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+aligned: {U32.is_zero((lanes % 8 : U32)) == True{} : Bool},
  +group: Nat,+inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes_fit} = facts
  +count = U32.to_nat(lanes)
  +quotient = U32.to_nat((lanes / 8 : U32))
  +multiple = Radix.divide_eight_exact_multiple(lanes,aligned)
  +bound = {%Equal.sym(Nat,Nat.mul(quotient,8n),count,multiple) : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count) : {Nat.is_le(count,Nat.mul(quotient,8n)) == True{} : Bool}}
  +most = {%Equal.sym(Nat,U32.to_nat(Tiled.groups(lanes)),Ceiling.ceiling(count,8n),exact) : {Nat.is_le(_,quotient) == True{} : Bool}
    Ceiling.minimal(count,8n,quotient,{==},bound) : {Nat.is_le(U32.to_nat(Tiled.groups(lanes)),quotient) == True{} : Bool}}
  +below = Order.le_lt_trans(group,group,quotient,Order.reflexive(group),Bounds.nat_trans(group,U32.to_nat(Tiled.groups(lanes)),quotient,inside,most))
  Bounds.nat_trans(Nat.add(7n,Nat.mul(group,8n)),Nat.mul(1n+group,8n),count,step_below(group),
    %multiple : {Nat.is_le(Nat.mul(1n+group,8n),_) == True{} : Bool}
    RowMajor.mul_le(1n+group,quotient,8n,Order.strict_follow(group,quotient,below)))

# A group inside the groups that is not before the last is the last.
def last_group(+group: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+inside: {Nat.is_lt(group,groups) == True{} : Bool},
  +outer: {Nat.is_lt(group,Nat.sub(groups,1n)) == False{} : Bool}) -> {group == Nat.sub(groups,1n) : Nat}:
  +last = Nat.sub(groups,1n)
  +below = {%Equal.sym(Nat,1n+last,groups,Order.positive_has_last(groups,positive)) : {Nat.is_lt(group,_) == True{} : Bool} inside :
    {Nat.is_lt(group,1n+last) == True{} : Bool}}
  Order.antisymmetric(group,last,Order.cancel_offset_le(1n,group,last,Order.strict_follow(group,1n+last,below)),Order.not_less(group,last,outer))

def below_cases(below: Bool,+group: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+inside: {Nat.is_lt(group,groups) == True{} : Bool},
  +other: {Nat.is_eq(group,Nat.sub(groups,1n)) == False{} : Bool},+classified: {Nat.is_lt(group,Nat.sub(groups,1n)) == below : Bool}) ->
  {Nat.is_lt(group,Nat.sub(groups,1n)) == True{} : Bool}:
  match below:
    case True{}: classified
    case False{}:
      +last = Nat.sub(groups,1n)
      +same = last_group(group,groups,positive,inside,classified)
      Empty.absurd({Nat.is_lt(group,last) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(last,last),True{},
        Equal.sym(Bool,Nat.is_eq(last,last),False{},%same : {Nat.is_eq(_,last) == False{} : Bool} other),Order.equal_reflexive(last))))

# A group inside the groups that is not the last lies before it.
def before_last(+group: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+inside: {Nat.is_lt(group,groups) == True{} : Bool},
  +other: {Nat.is_eq(group,Nat.sub(groups,1n)) == False{} : Bool}) -> {Nat.is_lt(group,Nat.sub(groups,1n)) == True{} : Bool}:
  below_cases(Nat.is_lt(group,Nat.sub(groups,1n)),group,groups,positive,inside,other,{==})

# The words of the last group: its index and its first lane.
def last_word(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}) ->
  {U32.to_nat((Tiled.groups(lanes) - 1 : U32)) == Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n) : Nat}:
  TiledValues.Groups{+exact,+fewer,+lanes_fit} = facts
  +groups = U32.to_nat(Tiled.groups(lanes))
  +half = U32.to_nat(Machine.half_word(limits))
  Arithmetic.subtract(Tiled.groups(lanes),1,Machine.half_word(limits),Limits.half_word_fits(limits),Order.strict_follow(0n,groups,positive),
    Order.transitive(Nat.sub(groups,1n),groups,half,Order.sub_le(groups,1n),Order.transitive(groups,Nat.mul(groups,8n),half,RowMajor.product_covers(groups,8n,{==}),lanes_fit)))

def first_lane_word(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}) ->
  {U32.to_nat(((Tiled.groups(lanes) - 1) * 8 : U32)) == Nat.mul(Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n),8n) : Nat}:
  TiledValues.Groups{+exact,+fewer,+lanes_fit} = facts
  +groups = U32.to_nat(Tiled.groups(lanes))
  +last = Nat.sub(groups,1n)
  +index = last_word(limits,lanes,TiledValues.Groups{exact,fewer,lanes_fit},positive)
  %index : {U32.to_nat(((Tiled.groups(lanes) - 1) * 8 : U32)) == Nat.mul(_,8n) : Nat}
  WordMultiplication.bounded_multiply((Tiled.groups(lanes) - 1 : U32),8,Machine.half_word(limits),Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat((Tiled.groups(lanes) - 1 : U32)),last,index) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Order.transitive(Nat.mul(last,8n),Nat.mul(groups,8n),U32.to_nat(Machine.half_word(limits)),RowMajor.mul_le(last,groups,8n,Order.sub_le(groups,1n)),lanes_fit))

def first_lane_bound(+limits: Machine.Limits,+lanes: U32,+facts: TiledValues.Groups<limits,lanes>,+positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n),8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes_fit} = facts
  +groups = U32.to_nat(Tiled.groups(lanes))
  +last = Nat.sub(groups,1n)
  Bounds.strict_le(Nat.add(Nat.mul(last,8n),7n),U32.to_nat(Machine.half_word(limits)),
    Bounds.nat_trans(Nat.add(Nat.mul(last,8n),7n),Nat.mul(groups,8n),U32.to_nat(Machine.half_word(limits)),
      %Order.positive_has_last(groups,positive) : {Nat.is_lt(Nat.add(Nat.mul(last,8n),7n),Nat.mul(_,8n)) == True{} : Bool}
      %Order.commutative(7n,Nat.mul(last,8n)) : {Nat.is_lt(_,Nat.mul(1n+last,8n)) == True{} : Bool}
      step_below(last),lanes_fit))

# ---- A padded cell ----

# The lanes of a tile that agree with a target below the lane count, the
# target holding +0 past it.
type Valid<-lanes: Nat,-group: Nat,-tile: Tile.Tile8,-target: Tile.Tile8> is Data:
  Valid{lane0: {kept(lanes,group,0n,Tile.lane0(tile)) == Tile.lane0(target) : F32},lane1: {kept(lanes,group,1n,Tile.lane1(tile)) == Tile.lane1(target) : F32},
    lane2: {kept(lanes,group,2n,Tile.lane2(tile)) == Tile.lane2(target) : F32},lane3: {kept(lanes,group,3n,Tile.lane3(tile)) == Tile.lane3(target) : F32},
    lane4: {kept(lanes,group,4n,Tile.lane4(tile)) == Tile.lane4(target) : F32},lane5: {kept(lanes,group,5n,Tile.lane5(tile)) == Tile.lane5(target) : F32},
    lane6: {kept(lanes,group,6n,Tile.lane6(tile)) == Tile.lane6(target) : F32},lane7: {kept(lanes,group,7n,Tile.lane7(tile)) == Tile.lane7(target) : F32}}

# A group whose lanes lie below the count holds the target.
def full_tile(+lanes: Nat,+group: Nat,+tile: Tile.Tile8,+target: Tile.Tile8,+valid: Valid<lanes,group,tile,target>,
  +full: {Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),lanes) == True{} : Bool}) -> {tile == target : Tile.Tile8}:
  Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = valid
  TileLaws.tiles_equal(tile,target,
    %lane_inside(lanes,group,0n,{==},full) : {Bool.pick(F32,_,Tile.lane0(tile),0.0) == Tile.lane0(target) : F32} lane0,
    %lane_inside(lanes,group,1n,{==},full) : {Bool.pick(F32,_,Tile.lane1(tile),0.0) == Tile.lane1(target) : F32} lane1,
    %lane_inside(lanes,group,2n,{==},full) : {Bool.pick(F32,_,Tile.lane2(tile),0.0) == Tile.lane2(target) : F32} lane2,
    %lane_inside(lanes,group,3n,{==},full) : {Bool.pick(F32,_,Tile.lane3(tile),0.0) == Tile.lane3(target) : F32} lane3,
    %lane_inside(lanes,group,4n,{==},full) : {Bool.pick(F32,_,Tile.lane4(tile),0.0) == Tile.lane4(target) : F32} lane4,
    %lane_inside(lanes,group,5n,{==},full) : {Bool.pick(F32,_,Tile.lane5(tile),0.0) == Tile.lane5(target) : F32} lane5,
    %lane_inside(lanes,group,6n,{==},full) : {Bool.pick(F32,_,Tile.lane6(tile),0.0) == Tile.lane6(target) : F32} lane6,
    %lane_inside(lanes,group,7n,{==},full) : {Bool.pick(F32,_,Tile.lane7(tile),0.0) == Tile.lane7(target) : F32} lane7)

# The tail of the runtime leaves in a cell of the last group the target.
def tail_tile(+limits: Machine.Limits,+lanes: U32,+tile: Tile.Tile8,+target: Tile.Tile8,+facts: TiledValues.Groups<limits,lanes>,
  +positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +valid: Valid<U32.to_nat(lanes),Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n),tile,target>) ->
  {Cells.tail(Cells.Tail{((Grid.groups(lanes) - 1) * 8 : U32),lanes},tile) == target : Tile.Tile8}:
  Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = valid
  +first = ((Grid.groups(lanes) - 1) * 8 : U32)
  +last = Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)
  +count = U32.to_nat(lanes)
  +exact = first_lane_word(limits,lanes,facts,positive)
  +bound = first_lane_bound(limits,lanes,facts,positive)
  +parts = {Tile.Tile{Tile.lane0(tile),Tile.lane1(tile),Tile.lane2(tile),Tile.lane3(tile),Tile.lane4(tile),Tile.lane5(tile),Tile.lane6(tile),Tile.lane7(tile)} : Tile.Tile8}
  %Equal.sym(Tile.Tile8,tile,parts,TileLaws.tile_eta(tile)) : {Cells.tail(Cells.Tail{first,lanes},_) == target : Tile.Tile8}
  TileLaws.tiles_equal(Cells.tail(Cells.Tail{first,lanes},parts),target,
    Equal.trans(F32,Cells.tail_lane(first,lanes,0,Tile.lane0(tile)),kept(count,last,0n,Tile.lane0(tile)),Tile.lane0(target),
      tail_lane(Tile.lane0(tile),limits,first,lanes,last,0n,exact,{==},lane_bound(limits,last,0n,{==},bound)),lane0),
    Equal.trans(F32,Cells.tail_lane(first,lanes,1,Tile.lane1(tile)),kept(count,last,1n,Tile.lane1(tile)),Tile.lane1(target),
      tail_lane(Tile.lane1(tile),limits,first,lanes,last,1n,exact,{==},lane_bound(limits,last,1n,{==},bound)),lane1),
    Equal.trans(F32,Cells.tail_lane(first,lanes,2,Tile.lane2(tile)),kept(count,last,2n,Tile.lane2(tile)),Tile.lane2(target),
      tail_lane(Tile.lane2(tile),limits,first,lanes,last,2n,exact,{==},lane_bound(limits,last,2n,{==},bound)),lane2),
    Equal.trans(F32,Cells.tail_lane(first,lanes,3,Tile.lane3(tile)),kept(count,last,3n,Tile.lane3(tile)),Tile.lane3(target),
      tail_lane(Tile.lane3(tile),limits,first,lanes,last,3n,exact,{==},lane_bound(limits,last,3n,{==},bound)),lane3),
    Equal.trans(F32,Cells.tail_lane(first,lanes,4,Tile.lane4(tile)),kept(count,last,4n,Tile.lane4(tile)),Tile.lane4(target),
      tail_lane(Tile.lane4(tile),limits,first,lanes,last,4n,exact,{==},lane_bound(limits,last,4n,{==},bound)),lane4),
    Equal.trans(F32,Cells.tail_lane(first,lanes,5,Tile.lane5(tile)),kept(count,last,5n,Tile.lane5(tile)),Tile.lane5(target),
      tail_lane(Tile.lane5(tile),limits,first,lanes,last,5n,exact,{==},lane_bound(limits,last,5n,{==},bound)),lane5),
    Equal.trans(F32,Cells.tail_lane(first,lanes,6,Tile.lane6(tile)),kept(count,last,6n,Tile.lane6(tile)),Tile.lane6(target),
      tail_lane(Tile.lane6(tile),limits,first,lanes,last,6n,exact,{==},lane_bound(limits,last,6n,{==},bound)),lane6),
    Equal.trans(F32,Cells.tail_lane(first,lanes,7,Tile.lane7(tile)),kept(count,last,7n,Tile.lane7(tile)),Tile.lane7(target),
      tail_lane(Tile.lane7(tile),limits,first,lanes,last,7n,exact,{==},lane_bound(limits,last,7n,{==},bound)),lane7))

# The cell of the last group or of another one, past an unaligned count.
def last_cases(last: Bool,+limits: Machine.Limits,+lanes: U32,+group: Nat,+tile: Tile.Tile8,+target: Tile.Tile8,+facts: TiledValues.Groups<limits,lanes>,
  +inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},+unaligned: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool},
  +valid: Valid<U32.to_nat(lanes),group,tile,target>,+classified: {Nat.is_eq(group,Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)) == last : Bool}) ->
  {Padding.kept(last,Cells.Tail{((Grid.groups(lanes) - 1) * 8 : U32),lanes},tile) == target : Tile.Tile8}:
  match last:
    case True{}:
      +groups = U32.to_nat(Tiled.groups(lanes))
      +same = Order.equal_of_is_eq(group,Nat.sub(groups,1n),classified)
      tail_tile(limits,lanes,tile,target,facts,Order.positive_bound(group,groups,inside),%same : Valid<U32.to_nat(lanes),_,tile,target> valid)
    case False{}:
      +groups = U32.to_nat(Tiled.groups(lanes))
      full_tile(U32.to_nat(lanes),group,tile,target,valid,
        inner_full(limits,lanes,facts,unaligned,group,before_last(group,groups,Order.positive_bound(group,groups,inside),inside,classified)))

# The cell of an aligned count or of an unaligned one.
def aligned_cases(aligned: Bool,+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+group: Nat,+index: Nat,+tile: Tile.Tile8,
  +target: Tile.Tile8,+facts: TiledValues.Groups<limits,lanes>,+inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +last_exact: {Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index) == Nat.is_eq(group,Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)) : Bool},
  +valid: Valid<U32.to_nat(lanes),group,tile,target>,+classified: {U32.is_zero((lanes % 8 : U32)) == aligned : Bool}) ->
  {Bool.pick(Tile.Tile8,aligned,tile,Padding.kept(Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index),
    Cells.Tail{((Grid.groups(lanes) - 1) * 8 : U32),lanes},tile)) == target : Tile.Tile8}:
  match aligned:
    case True{}: full_tile(U32.to_nat(lanes),group,tile,target,valid,aligned_full(limits,lanes,facts,classified,group,inside))
    case False{}:
      +final = Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)
      %Equal.sym(Bool,Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index),Nat.is_eq(group,final),last_exact) :
        {Padding.kept(_,Cells.Tail{((Grid.groups(lanes) - 1) * 8 : U32),lanes},tile) == target : Tile.Tile8}
      last_cases(Nat.is_eq(group,final),limits,lanes,group,tile,target,facts,inside,unaligned_positive(lanes,classified),valid,{==})

# Padding leaves the target in a cell of a group whose lanes agree with it.
def padded_group(+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+group: Nat,+index: Nat,+tile: Tile.Tile8,+target: Tile.Tile8,
  +facts: TiledValues.Groups<limits,lanes>,+inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +last_exact: {Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index) == Nat.is_eq(group,Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)) : Bool},
  +valid: Valid<U32.to_nat(lanes),group,tile,target>) ->
  {Padding.cell(TiledMap.Padding{lanes,Grid.groups(lanes),padded_outer,padded_position},index,tile) == target : Tile.Tile8}:
  aligned_cases(U32.is_zero((lanes % 8 : U32)),limits,lanes,padded_outer,padded_position,group,index,tile,target,facts,inside,last_exact,valid,{==})

# ---- Where a cell lies ----

# The group of a cell lies below the groups.
def group_below(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Nat.is_lt(Representation.group_at(position,outer,groups,index),groups) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      +positive = RowMajor.left_factor_positive(groups,outer,Order.positive_bound(index,Nat.mul(groups,outer),inside))
      %Equal.sym(Nat,Representation.least_one(groups),groups,Representation.least_positive(groups,positive)) : {Nat.is_lt(Nat.mod(index,_),groups) == True{} : Bool}
      DivisionOrder.remainder_small(index,groups,positive)
    case Grid.Outermost{}:
      +positive = DivisionOrder.right_factor_positive(groups,outer,Order.positive_bound(index,Nat.mul(groups,outer),inside))
      %Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,positive)) : {Nat.is_lt(Nat.div(index,_),groups) == True{} : Bool}
      DivisionOrder.quotient_below(index,groups,outer,positive,inside)

def and_true(left: Bool) -> {left && True{} == left : Bool}:
  match left:
    case True{}: {==}
    case False{}: {==}

# With the groups outermost a cell is in the last group when its quotient is.
def outer_last(final: Bool,+outer: Nat,+groups: Nat,+index: Nat,+positive: {Nat.is_lt(0n,outer) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +below: {Nat.is_lt(Nat.div(index,outer),groups) == True{} : Bool},+classified: {Nat.is_eq(Nat.div(index,outer),Nat.sub(groups,1n)) == final : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.sub(groups,1n),outer),index) == final : Bool}:
  match final:
    case True{}:
      +same = Order.equal_of_is_eq(Nat.div(index,outer),Nat.sub(groups,1n),classified)
      %same : {Nat.is_le(Nat.mul(_,outer),index) == True{} : Bool}
      DivisionOrder.product_below(index,outer,positive)
    case False{}:
      +quotient = Nat.div(index,outer)
      +last = Nat.sub(groups,1n)
      +before = before_last(quotient,groups,groups_positive,below,classified)
      Padding.below_not_le(index,Nat.mul(last,outer),
        Bounds.nat_trans(index,Nat.mul(1n+quotient,outer),Nat.mul(last,outer),DivisionOrder.next_above(index,outer,positive),
          RowMajor.mul_le(1n+quotient,last,outer,Order.strict_follow(quotient,last,before))))

# Along the lane axis the padding finds the last group where the cells have it.
def last_along(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Padding.last(position,groups,outer,index) == Nat.is_eq(Representation.group_at(position,outer,groups,index),Nat.sub(groups,1n)) : Bool}:
  match position:
    case Grid.Innermost{}:
      +positive = RowMajor.left_factor_positive(groups,outer,Order.positive_bound(index,Nat.mul(groups,outer),inside))
      %Equal.sym(Nat,Representation.least_one(groups),groups,Representation.least_positive(groups,positive)) :
        {Nat.is_eq(Nat.mod(index,groups),Nat.sub(groups,1n)) == Nat.is_eq(Nat.mod(index,_),Nat.sub(groups,1n)) : Bool}
      {==}
    case Grid.Outermost{}:
      +product_positive = Order.positive_bound(index,Nat.mul(groups,outer),inside)
      +positive = DivisionOrder.right_factor_positive(groups,outer,product_positive)
      +last = Nat.sub(groups,1n)
      %Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,positive)) :
        {Nat.is_le(Nat.mul(last,outer),index) && Nat.is_lt(index,Nat.mul(groups,outer)) == Nat.is_eq(Nat.div(index,_),last) : Bool}
      %Equal.sym(Bool,Nat.is_lt(index,Nat.mul(groups,outer)),True{},inside) : {Nat.is_le(Nat.mul(last,outer),index) && _ == Nat.is_eq(Nat.div(index,outer),last) : Bool}
      %Equal.sym(Bool,Nat.is_le(Nat.mul(last,outer),index) && True{},Nat.is_le(Nat.mul(last,outer),index),and_true(Nat.is_le(Nat.mul(last,outer),index))) :
        {_ == Nat.is_eq(Nat.div(index,outer),last) : Bool}
      outer_last(Nat.is_eq(Nat.div(index,outer),last),outer,groups,index,positive,RowMajor.left_factor_positive(groups,outer,product_positive),
        DivisionOrder.quotient_below(index,groups,outer,positive,inside),{==})

def mod_one(+index: Nat) -> {Nat.mod(index,1n) == 0n : Nat}:
  Order.equal_of_is_eq(Nat.mod(index,1n),0n,Padding.below_one(Nat.mod(index,1n),DivisionOrder.remainder_small(index,1n,{==})))

# With one group every cell is in the last group.
def single_padded(+index: Nat,+groups: Nat,+one: {groups == 1n : Nat}) -> {Nat.is_eq(Nat.mod(index,groups),Nat.sub(groups,1n)) == True{} : Bool}:
  %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Nat.mod(index,_),Nat.sub(groups,1n)) == True{} : Bool}
  %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Nat.mod(index,1n),Nat.sub(_,1n)) == True{} : Bool}
  %Equal.sym(Nat,Nat.mod(index,1n),0n,mod_one(index)) : {Nat.is_eq(_,0n) == True{} : Bool}
  {==}

def single_group(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+one: {groups == 1n : Nat},+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Nat.is_eq(Representation.group_at(position,outer,groups,index),Nat.sub(groups,1n)) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Representation.group_at(Grid.Innermost{},outer,_,index),Nat.sub(groups,1n)) == True{} : Bool}
      %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Representation.group_at(Grid.Innermost{},outer,1n,index),Nat.sub(_,1n)) == True{} : Bool}
      %Equal.sym(Nat,Nat.mod(index,1n),0n,mod_one(index)) : {Nat.is_eq(_,0n) == True{} : Bool}
      {==}
    case Grid.Outermost{}:
      +small = {%Addition.zero_right(outer) : {Nat.is_lt(index,_) == True{} : Bool} %one : {Nat.is_lt(index,Nat.mul(_,outer)) == True{} : Bool} inside :
        {Nat.is_lt(index,outer) == True{} : Bool}}
      +positive = Order.positive_bound(index,outer,small)
      +zero = {Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(index,outer),(0n,index),Division.small(index,outer,small)) : {Nat.div(index,outer) == 0n : Nat}}
      %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Nat.div(index,Representation.least_one(outer)),Nat.sub(_,1n)) == True{} : Bool}
      %Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,positive)) : {Nat.is_eq(Nat.div(index,_),0n) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(index,outer),0n,zero) : {Nat.is_eq(_,0n) == True{} : Bool}
      {==}

def last_single(position: Grid.Position,+outer: Nat,+groups: Nat,+index: Nat,+one: {groups == 1n : Nat},+inside: {Nat.is_lt(index,Nat.mul(groups,outer)) == True{} : Bool}) ->
  {Nat.is_eq(Nat.mod(index,groups),Nat.sub(groups,1n)) == Nat.is_eq(Representation.group_at(position,outer,groups,index),Nat.sub(groups,1n)) : Bool}:
  Equal.trans(Bool,Nat.is_eq(Nat.mod(index,groups),Nat.sub(groups,1n)),True{},Nat.is_eq(Representation.group_at(position,outer,groups,index),Nat.sub(groups,1n)),
    single_padded(index,groups,one),
    Equal.sym(Bool,Nat.is_eq(Representation.group_at(position,outer,groups,index),Nat.sub(groups,1n)),True{},single_group(position,outer,groups,index,one,inside)))

# The padding finds the last group of a cell where the values have it.
def last_placed(+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+outer: U32,+position: Grid.Position,+index: Nat,
  placement: Public.Placement<lanes,padded_outer,padded_position,outer,position>,
  +inside: {Nat.is_lt(index,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool}) ->
  {Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index) ==
    Nat.is_eq(Representation.group_at(position,U32.to_nat(outer),U32.to_nat(Tiled.groups(lanes)),index),Nat.sub(U32.to_nat(Tiled.groups(lanes)),1n)) : Bool}:
  match placement:
    case Public.Along{+outer_same,+position_same}:
      +groups = U32.to_nat(Grid.groups(lanes))
      +found = Nat.is_eq(Representation.group_at(position,U32.to_nat(outer),groups,index),Nat.sub(groups,1n))
      %Equal.sym(U32,padded_outer,outer,outer_same) : {Padding.last(padded_position,groups,U32.to_nat(_),index) == found : Bool}
      %Equal.sym(Grid.Position,padded_position,position,position_same) : {Padding.last(_,groups,U32.to_nat(outer),index) == found : Bool}
      last_along(position,U32.to_nat(outer),groups,index,inside)
    case Public.Single{+one,+inner}:
      +groups = U32.to_nat(Grid.groups(lanes))
      +found = Nat.is_eq(Representation.group_at(position,U32.to_nat(outer),groups,index),Nat.sub(groups,1n))
      %Equal.sym(Grid.Position,padded_position,Grid.Innermost{},inner) : {Padding.last(_,groups,U32.to_nat(padded_outer),index) == found : Bool}
      last_single(position,U32.to_nat(outer),groups,index,Equal.cong(U32,Nat,word => U32.to_nat(Grid.groups(word)),lanes,1,one),inside)

# The cell at a cell index of the values holds the target when its lanes agree
# with it.
def padded_index(+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+outer: U32,+after: U32,+position: Grid.Position,+total: Nat,
  +index: Nat,+tile: Tile.Tile8,+target: Tile.Tile8,+facts: TiledValues.Groups<limits,lanes>,+cells: {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)) == total : Nat},
  +placement: Public.Placement<lanes,padded_outer,padded_position,outer,position>,+total_half: {Nat.is_le(total,U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(index,total) == True{} : Bool},
  +valid: Valid<U32.to_nat(lanes),Representation.cell_group(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(U32.from_nat(index))),tile,target>) ->
  {Padding.cell(TiledMap.Padding{lanes,Grid.groups(lanes),padded_outer,padded_position},index,tile) == target : Tile.Tile8}:
  +geometry = {Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +groups = U32.to_nat(Tiled.groups(lanes))
  +product = Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))
  +number = U32.to_nat(U32.from_nat(index))
  +round = {Bounds.roundtrip(index,Machine.half_word(limits),Limits.half_word_fits(limits),
      Order.transitive(index,total,U32.to_nat(Machine.half_word(limits)),Bounds.strict_le(index,total,inside),total_half)) : {number == index : Nat}}
  +product_inside = {%Equal.sym(Nat,product,total,cells) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,product) == True{} : Bool}}
  +number_inside = {%Equal.sym(Nat,number,index,round) : {Nat.is_lt(_,product) == True{} : Bool} product_inside : {Nat.is_lt(number,product) == True{} : Bool}}
  +last_index = last_placed(lanes,padded_outer,padded_position,outer,position,index,placement,product_inside)
  +last_number = {%Equal.sym(Nat,number,index,round) :
      {Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index) ==
        Nat.is_eq(Representation.group_at(position,U32.to_nat(outer),groups,_),Nat.sub(groups,1n)) : Bool} last_index :
    {Padding.last(padded_position,U32.to_nat(Grid.groups(lanes)),U32.to_nat(padded_outer),index) ==
      Nat.is_eq(Representation.group_at(position,U32.to_nat(outer),groups,number),Nat.sub(groups,1n)) : Bool}}
  padded_group(limits,lanes,padded_outer,padded_position,Representation.cell_group(geometry,number),index,tile,target,facts,
    group_below(position,U32.to_nat(outer),groups,number,number_inside),last_number,valid)

# ---- The layout of the cells ----

# The lanes, the padding and the geometry of the cells of values, and their
# cover.
type Layout is Data:
  Layout{limits: Machine.Limits,lanes: U32,padded_outer: U32,padded_position: Grid.Position,outer: U32,after: U32,position: Grid.Position,total: Nat,
    cover: Public.Cover<limits,lanes,padded_outer,padded_position,outer,position,total>}

def layout_padding(+layout: Layout) -> TiledMap.Padding:
  Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  TiledMap.Padding{lanes,Grid.groups(lanes),padded_outer,padded_position}

def layout_geometry(+layout: Layout) -> Cells.Cells:
  Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position}

def layout_total(+layout: Layout) -> Nat:
  Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  total

def fits_when(aligned: Bool,+cap: U32,+groups: Nat,+outer: Nat,+first: Nat,+count: Nat,+unaligned: Padding.Unaligned(cap,groups,outer,first,count)) ->
  Padding.Fits(cap,aligned,groups,outer,first,count):
  match aligned:
    case True{}: Unit{}
    case False{}: unaligned

# A cell of a piece read back after padding holds the target when the cell
# read before it agrees with the target.
def padded_read(+layout: Layout,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +length_exact: {length == U32.to_nat(count) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),layout_total(layout)) == True{} : Bool},
  +index: Nat,+below: {Nat.is_lt(index,length) == True{} : Bool},+tile: Tile.Tile8,+read_exact: {Tree.read(Tile.Tile8,tree,U32.from_nat(index)) == tile : Tile.Tile8},
  +target: Tile.Tile8,
  valid: @+inside: {Nat.is_lt(U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(first),index))),layout_total(layout)) == True{} : Bool} ->
    Valid<Representation.lane_count(layout_geometry(layout)),Representation.cell_group(layout_geometry(layout),U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(first),index)))),
      tile,target>) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Padding.step(layout_padding(layout),first,count,Buffer.Content{tree,length})),U32.from_nat(index)) == target : Tile.Tile8}:
  Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  Public.Cover{+counted,+cells_count,+padded_cells,+placement} = cover
  +padding = {TiledMap.Padding{lanes,Grid.groups(lanes),padded_outer,padded_position} : TiledMap.Padding}
  +count_inside = {%length_exact : {Nat.is_lt(index,_) == True{} : Bool} below : {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  +number = Nat.add(U32.to_nat(first),index)
  +span = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +number_inside = {Bounds.nat_trans(number,span,total,Order.offset_lt(U32.to_nat(first),index,U32.to_nat(count),count_inside),inside) : {Nat.is_lt(number,total) == True{} : Bool}}
  +groups = U32.to_nat(Grid.groups(lanes))
  +outer_count = U32.to_nat(outer)
  +lane_count = U32.to_nat(lanes)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +half = U32.to_nat(Machine.half_word(limits))
  +product_positive = {%Equal.sym(Nat,Nat.mul(groups,outer_count),total,cells_count) : {Nat.is_lt(0n,_) == True{} : Bool} Order.positive_bound(number,total,number_inside) :
      {Nat.is_lt(0n,Nat.mul(groups,outer_count)) == True{} : Bool}}
  +outer_positive = DivisionOrder.right_factor_positive(groups,outer_count,product_positive)
  +groups_positive = RowMajor.left_factor_positive(groups,outer_count,product_positive)
  +lanes_bound = Order.transitive(lane_count,Nat.mul(lane_count,outer_count),capacity,RowMajor.product_covers(lane_count,outer_count,outer_positive),counted)
  +facts = TiledValues.grouped(limits,lanes,lanes_bound)
  +total_bound = {%cells_count : {Nat.is_le(_,capacity) == True{} : Bool}
      Order.transitive(Nat.mul(groups,outer_count),Nat.mul(lane_count,outer_count),capacity,RowMajor.mul_le(groups,lane_count,outer_count,TiledValues.fewer_of(limits,lanes,facts)),counted) :
      {Nat.is_le(total,capacity) == True{} : Bool}}
  +total_half = Order.transitive(total,capacity,half,total_bound,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))
  +span_half = Order.transitive(span,total,half,inside,total_half)
  +groups_bound = Order.transitive(groups,total,capacity,
    %cells_count : {Nat.is_le(groups,_) == True{} : Bool} RowMajor.product_covers(groups,outer_count,outer_positive),total_bound)
  +fits = fits_when(U32.is_zero((lanes % 8 : U32)),Machine.half_word(limits),groups,U32.to_nat(padded_outer),U32.to_nat(first),U32.to_nat(count),
    KernelGemm.Both{groups_positive,KernelGemm.Both{Limits.double_bound(limits,groups,groups_bound),
      KernelGemm.Both{%Equal.sym(Nat,Nat.mul(groups,U32.to_nat(padded_outer)),total,padded_cells) : {Nat.is_le(_,half) == True{} : Bool} total_half,span_half}}})
  +bounded = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(_,half) == True{} : Bool}
      Order.transitive(U32.to_nat(count),span,half,Order.right_le(U32.to_nat(first),U32.to_nat(count)),span_half) : {Nat.is_le(length,half) == True{} : Bool}}
  +room_count = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} Order.reflexive(U32.to_nat(count)) :
      {Nat.is_le(U32.to_nat(count),length) == True{} : Bool}}
  +read_padded = Padding.read(Machine.half_word(limits),Limits.half_word_fits(limits),padding,first,count,tree,length,ready,fits,bounded,room_count,index,count_inside)
  +word_exact = FullRange.roundtrip(number,Machine.capacity(limits),Bounds.strict_le(number,capacity,Bounds.nat_trans(number,total,capacity,number_inside,total_bound)))
  +word_inside = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,word_exact) : {Nat.is_lt(_,total) == True{} : Bool} number_inside :
    {Nat.is_lt(U32.to_nat(U32.from_nat(number)),total) == True{} : Bool}}
  +cell = padded_index(limits,lanes,padded_outer,padded_position,outer,after,position,total,number,tile,target,facts,cells_count,placement,total_half,number_inside,valid(word_inside))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})),U32.from_nat(index)),
    Padding.cell(padding,number,Tree.read(Tile.Tile8,tree,U32.from_nat(index))),target,read_padded,
    %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,tree,U32.from_nat(index)),tile,read_exact) : {Padding.cell(padding,number,_) == target : Tile.Tile8} cell)

# ---- The layout of tiled values ----

def count_bounded(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  bounded

# The layout of the cells of values with their padding, their geometry and
# their count.
type Placed<-shape: List<&2,U32>,-values: List<&2,U32>,-axis: Nat,-lanes: U32,-position: Grid.Position,-within: Bool> is Data:
  Placed{layout: Layout,padding_exact: {layout_padding(layout) == TiledMap.padding_when(within,shape,lanes,position) : TiledMap.Padding},
    geometry_exact: {layout_geometry(layout) == Public.geometry_of(values,axis,position) : Cells.Cells},total_exact: {layout_total(layout) == ShapeLaws.elements(shape) : Nat}}

# The layout of laid values with the lane axis inside them or past them.
def placed(within: Bool,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+shape: List<&2,U32>,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool},+laid: Public.Laid<shape,axis,lanes,position,values>) ->
  Placed<shape,values,axis,lanes,position,within>:
  match within:
    case True{}:
      Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
      Global.Counted{+count,+accepted} = counted
      +limits = Default.limits()
      +outer = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
      +after = Tiled.product(Public.lane_after(values,axis))
      +exact = Count.count_exact(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +bounded = count_bounded(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +within_values = {Equal.trans(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),Nat.is_lt(axis,List.length(&2,U32,shape)),True{},
          Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),Nat.is_lt(axis,List.length(&2,U32,values)),
            %shaped : {Nat.is_lt(axis,List.length(&2,U32,_)) == Nat.is_lt(axis,List.length(&2,U32,values)) : Bool} Public.cells_within(values,axis,position)),classified) :
        {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}}
      +geometry_exact = {%pinned : {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,_,Tiled.groups(lanes),position} : Cells.Cells}
          %pinned : {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,Grid.lanes_at(values,axis),Tiled.groups(_),position} : Cells.Cells}
          Public.geometry_exact(values,axis,position,inside) :
        {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position} : Cells.Cells}}
      +shape_cells = Tiled.cell_shape(values,axis,position)
      +cover = {%shaped : Public.Cover<limits,lanes,Grid.outer_count(_,position),position,outer,position,ShapeLaws.elements(shape)>
          %shaped : Public.Cover<limits,lanes,Grid.outer_count(shape_cells,position),position,outer,position,ShapeLaws.elements(_)>
          Public.along_cover(limits,values,axis,lanes,position,count,exact,bounded,within_values,pinned) :
        Public.Cover<limits,lanes,Grid.outer_count(shape,position),position,outer,position,ShapeLaws.elements(shape)>}
      Placed{Layout{limits,lanes,Grid.outer_count(shape,position),position,outer,after,position,ShapeLaws.elements(shape),cover},{==},
        Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},geometry_exact),{==}}
    case False{}:
      Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
      Global.Counted{+count,+accepted} = counted
      +limits = Default.limits()
      +outer = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
      +after = Tiled.product(Public.lane_after(values,axis))
      +exact = Count.count_exact(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +bounded = count_bounded(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
      +past = {Equal.trans(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),Nat.is_lt(axis,List.length(&2,U32,shape)),False{},
          Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),Nat.is_lt(axis,List.length(&2,U32,values)),
            %shaped : {Nat.is_lt(axis,List.length(&2,U32,_)) == Nat.is_lt(axis,List.length(&2,U32,values)) : Bool} Public.cells_within(values,axis,position)),classified) :
        {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool}}
      +one = Public.lanes_past(values,axis,past)
      +geometry_exact = {%one : {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,_,Tiled.groups(1),position} : Cells.Cells}
          %one : {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,Grid.lanes_at(values,axis),Tiled.groups(_),position} : Cells.Cells}
          Public.geometry_exact(values,axis,position,inside) :
        {Public.geometry_of(values,axis,position) == Cells.Cells{outer,after,1,Tiled.groups(1),position} : Cells.Cells}}
      +shape_cells = Tiled.cell_shape(values,axis,position)
      +cover = {%shaped : Public.Cover<limits,1,Grid.product(_),Grid.Innermost{},outer,position,ShapeLaws.elements(shape)>
          %shaped : Public.Cover<limits,1,Grid.product(shape_cells),Grid.Innermost{},outer,position,ShapeLaws.elements(_)>
          Public.single_cover(limits,values,axis,position,count,exact,bounded,past) :
        Public.Cover<limits,1,Grid.product(shape),Grid.Innermost{},outer,position,ShapeLaws.elements(shape)>}
      Placed{Layout{limits,1,Grid.product(shape),Grid.Innermost{},outer,after,position,ShapeLaws.elements(shape),cover},{==},
        Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),Cells.Cells{outer,after,1,Tiled.groups(1),position},geometry_exact),{==}}

# ---- A padded piece ----

# What every cell of a padded piece reads: its layout, a reader of its tiles
# and a target reader whose cells agree with those tiles lane by lane.
def Agreement(~Scope: Data,~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,
  ~layout_of: @+scope: Scope -> Layout,~reader_of: @+scope: Scope -> Reader,~target_of: @+scope: Scope -> Target) -> Type:
  @+scope: Scope -> @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),layout_total(layout_of(scope))) == True{} : Bool} ->
    Valid<Representation.lane_count(layout_geometry(layout_of(scope))),Representation.cell_group(layout_geometry(layout_of(scope)),U32.to_nat(word)),
      read(reader_of(scope),word),target_read(target_of(scope),word)>

# What one piece supplies to every one of its cells.
type CellScope<-Scope: Data,-layout_of: @+scope: Scope -> Layout> is Data:
  CellScope{scope: Scope,first: U32,count: U32,tree: Tree.Tree<Tile.Tile8>,length: Nat,ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
    length_exact: {length == U32.to_nat(count) : Nat},inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),layout_total(layout_of(scope))) == True{} : Bool},
    same: {Buffer.content_length(~Tile.Tile8,Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})) == length : Nat}}

def Before(~Scope: Data,~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~layout_of: @+scope: Scope -> Layout,~reader_of: @+scope: Scope -> Reader,
  +cells: CellScope<Scope,layout_of>,+index: Nat) -> Data:
  CellScope{+scope,+first,count,+tree,+length,ready,length_exact,inside,same} = cells
  Content.At(~Tile.Tile8,~Reader,~read,reader_of(scope),first,tree,length,index)

def After(~Scope: Data,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,~layout_of: @+scope: Scope -> Layout,~target_of: @+scope: Scope -> Target,
  +cells: CellScope<Scope,layout_of>,+index: Nat) -> Data:
  CellScope{+scope,+first,+count,+tree,+length,ready,length_exact,inside,same} = cells
  +output = Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})
  Content.At(~Tile.Tile8,~Target,~target_read,target_of(scope),first,Buffer.content_tree(~Tile.Tile8,output),Buffer.content_length(~Tile.Tile8,output),index)

# A cell of a piece holding the tile of the reader holds after padding the
# tile of the target.
def cell_at(~Scope: Data,~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,
  ~layout_of: @+scope: Scope -> Layout,~reader_of: @+scope: Scope -> Reader,~target_of: @+scope: Scope -> Target,
  ~agree: Agreement(~Scope,~Reader,~read,~Target,~target_read,~layout_of,~reader_of,~target_of),
  +cells: CellScope<Scope,layout_of>,+index: Nat,+before: Before(~Scope,~Reader,~read,~layout_of,~reader_of,cells,index)) ->
  After(~Scope,~Target,~target_read,~layout_of,~target_of,cells,index):
  CellScope{+scope,+first,+count,+tree,+length,+ready,+length_exact,+inside,+same} = cells
  KernelGemm.Both{+below,+equal} = before
  +layout = layout_of(scope)
  +word = U32.from_nat(Nat.add(U32.to_nat(first),index))
  +output = Padding.step(layout_padding(layout),first,count,Buffer.Content{tree,length})
  KernelGemm.Both{%Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,output),length,same) : {Nat.is_lt(index,_) == True{} : Bool} below,
    padded_read(layout,first,count,tree,length,ready,length_exact,inside,index,below,read(reader_of(scope),word),equal,target_read(target_of(scope),word),agree(scope,word))}

def kept_piece(~Scope: Data,~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,
  ~layout_of: @+scope: Scope -> Layout,~reader_of: @+scope: Scope -> Reader,~target_of: @+scope: Scope -> Target,
  ~agree: Agreement(~Scope,~Reader,~read,~Target,~target_read,~layout_of,~reader_of,~target_of),
  +scope: Scope,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +length_exact: {length == U32.to_nat(count) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),layout_total(layout_of(scope))) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Reader,~read,reader_of(scope),first,count,Buffer.Content{tree,length}),
  kept: KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})) == length : Nat}>) ->
  PiecesMap.Stepped(~Tile.Tile8,~Target,~target_read,target_of(scope),first,count,Buffer.Content{tree,length},
    Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})):
  KernelGemm.Both{+out_ready,+same} = kept
  +output = Padding.step(layout_padding(layout_of(scope)),first,count,Buffer.Content{tree,length})
  KernelGemm.Both{out_ready,KernelGemm.Both{same,
    %Equal.sym(Buffer.Content<Tile.Tile8>,output,Buffer.Content{Buffer.content_tree(~Tile.Tile8,output),Buffer.content_length(~Tile.Tile8,output)},Buffer.content_eta(~Tile.Tile8,output)) :
      Content.piece_values(~Tile.Tile8,~Target,~target_read,target_of(scope),first,count,_)
    Members.transform_all(~Nat,~CellScope<Scope,layout_of>,~Before(~Scope,~Reader,~read,~layout_of,~reader_of),~After(~Scope,~Target,~target_read,~layout_of,~target_of),
      ~cell_at(~Scope,~Reader,~read,~Target,~target_read,~layout_of,~reader_of,~target_of,~agree),Ranges.range(U32.to_nat(count),0n),
      CellScope{scope,first,count,tree,length,ready,length_exact,inside,same},values)}}

# A piece holding the tiles of a reader holds after padding the tiles of a
# target that agrees with them lane by lane.
def padded_piece(~Scope: Data,~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,
  ~layout_of: @+scope: Scope -> Layout,~reader_of: @+scope: Scope -> Reader,~target_of: @+scope: Scope -> Target,
  ~agree: Agreement(~Scope,~Reader,~read,~Target,~target_read,~layout_of,~reader_of,~target_of),
  +scope: Scope,+first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),layout_total(layout_of(scope))) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Reader,~read,reader_of(scope),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Target,~target_read,target_of(scope),first,count,input,Padding.step(layout_padding(layout_of(scope)),first,count,input)):
  Buffer.Content{+tree,+length} = input
  kept_piece(~Scope,~Reader,~read,~Target,~target_read,~layout_of,~reader_of,~target_of,~agree,scope,first,count,tree,length,ready,length_exact,inside,values,
    Padding.kept_ready(layout_padding(layout_of(scope)),first,count,tree,length,ready))
