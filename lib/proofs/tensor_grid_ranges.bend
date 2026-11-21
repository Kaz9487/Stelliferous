# The ranges of a grid: one range for each piece, in order. The left half of a
# split holds the largest power of two below its count.
import Base
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ../storage_buffer.bend as Storage
import ./u32_bit_length.bend as Bits
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ./u32_shift.bend as U32Shift
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Coordinates
import ./u32_comparison.bend as Comparison
import ./row_major_index.bend as RowMajor
import ./word_multiplication.bend as Multiplication
import ./tensor_grid_pieces.bend as Laws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as TiledPublic
import ./tensor_halo_windows.bend as Windows
import ./tensor_task_runs.bend as Runs
import ./tensor_placed.bend as Placed
import ./tensor_pieces_model.bend as PiecesModel
import ../traversal_partition.bend as Partition
import ../tensor_tasks.bend as Tasks
import ../kernel_cost.bend as Cost
import ../default_kernel_cost.bend as DefaultCost
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./binary_radix.bend as Radix
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as NatCeiling
import ./storage_buffer_model.bend as Buffer
import ./traversal_partition_refinement.bend as PartitionLaws
import ../tensor_move.bend as Move
import ../tensor_halo.bend as Halo
import ./tensor_pieces_grouped_with.bend as Grouped
import ../tensor.bend as Tensors
import ./tensor_global_shape_count.bend as Global
import ../kernel_gemm.bend as Gemm

def native_le_lt(+first: U32,+middle: U32,+last: U32,+before: {U32.is_le(first,middle) == True{} : Bool},+after: {U32.is_lt(middle,last) == True{} : Bool}) ->
  {U32.is_lt(first,last) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(first,last),Nat.is_lt(U32.to_nat(first),U32.to_nat(last)),True{},Comparison.lt(first,last),
    Order.le_lt_trans(U32.to_nat(first),U32.to_nat(middle),U32.to_nat(last),Bounds.as_nat_le(first,middle,before),Bounds.as_nat_lt(middle,last,after)))

# One shifted left depth times is two to the depth.
def shifted_power(+depth: Nat,+small: {Nat.is_le(depth,30n) == True{} : Bool}) -> {U32.to_nat(U32.shln(1,depth)) == Bits.power(depth) : Nat}:
  match depth:
    case 0n: {==}
    case 1n+ +rest:
      +rest_small = Order.transitive(rest,1n+rest,30n,Order.successor(rest,rest,Order.reflexive(rest)),small)
      Equal.trans(Nat,U32.to_nat(U32.shl(U32.shln(1,rest))),Nat.double(U32.to_nat(U32.shln(1,rest))),Nat.double(Bits.power(rest)),
        U32Shift.double(U32.shln(1,rest),native_le_lt(U32.shln(1,rest),1073741824,2147483648,Power.native_allocatable(rest,rest_small),{==})),
        Equal.cong(Nat,Nat,value => Nat.double(value),U32.to_nat(U32.shln(1,rest)),Bits.power(rest),shifted_power(rest,rest_small)))

# The bits of a count less one, but the top bit.
def half_depth(+count: U32) -> Nat:
  Storage.bit_length(31n,U32.shr((count - 1 : U32)))

def count_positive(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}:
  Bounds.nat_trans(0n,1n,U32.to_nat(count),{==},Bounds.strict_le(1n,U32.to_nat(count),more))

def rest_exact(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {U32.to_nat((count - 1 : U32)) == Nat.sub(U32.to_nat(count),1n) : Nat}:
  Arithmetic.subtract(count,1,count,Coordinates.native_le_transitive(count,1073741824,2147483648,within,{==}),
    Order.positive_one(U32.to_nat(count),count_positive(count,more,within)),Order.sub_le(U32.to_nat(count),1n))

def rest_positive(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat((count - 1 : U32))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),rest_exact(count,more,within)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.positive_difference(U32.to_nat(count),1n,more)

# The bits of a count less one are one more than its half depth.
def length_split(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {Storage.bit_length(32n,(count - 1 : U32)) == 1n+half_depth(count) : Nat}:
  Equal.cong(Bool,Nat,zero => Bool.pick(Nat,zero,0n,1n+half_depth(count)),U32.is_zero((count - 1 : U32)),False{},Bits.nonzero_word((count - 1 : U32),rest_positive(count,more,within)))

def depth_split(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {Storage.depth_for(count) == 1n+half_depth(count) : Nat}:
  Equal.trans(Nat,Storage.depth_for(count),Storage.allocation_depth(count),1n+half_depth(count),Depth.positive_depth(count,count_positive(count,more,within)),length_split(count,more,within))

# The left half of a split of a count of at least two: two to its half depth,
# below the count and at least half of it.
type Below<-count: U32> is Data:
  Below{exact: {U32.to_nat(Pieces.below(count)) == Bits.power(half_depth(count)) : Nat},
    under: {Nat.is_lt(Bits.power(half_depth(count)),U32.to_nat(count)) == True{} : Bool},
    cover: {Nat.is_le(U32.to_nat(count),Nat.double(Bits.power(half_depth(count)))) == True{} : Bool}}

def below_exact(+count: U32,+halves: Below<count>) -> {U32.to_nat(Pieces.below(count)) == Bits.power(half_depth(count)) : Nat}:
  Below{exact,under,cover} = halves
  exact

def below_under(+count: U32,+halves: Below<count>) -> {Nat.is_lt(Bits.power(half_depth(count)),U32.to_nat(count)) == True{} : Bool}:
  Below{exact,under,cover} = halves
  under

def below_cover(+count: U32,+halves: Below<count>) -> {Nat.is_le(U32.to_nat(count),Nat.double(Bits.power(half_depth(count)))) == True{} : Bool}:
  Below{exact,under,cover} = halves
  cover

def below_facts(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Below<count>:
  +split = depth_split(count,more,within)
  +sized = Depth.bounded(count,within)
  +depth_small = {%split : {Nat.is_le(_,30n) == True{} : Bool} Depth.allocation_depth(count,sized) : {Nat.is_le(1n+half_depth(count),30n) == True{} : Bool}}
  +half_small = Order.transitive(half_depth(count),1n+half_depth(count),30n,Order.successor(half_depth(count),half_depth(count),Order.reflexive(half_depth(count))),depth_small)
  +lower = {%length_split(count,more,within) : {Nat.is_le(Bits.power(Bits.before_one(_)),U32.to_nat((count - 1 : U32))) == True{} : Bool}
    Bits.length_lower(32n,(count - 1 : U32),rest_positive(count,more,within)) : {Nat.is_le(Bits.power(half_depth(count)),U32.to_nat((count - 1 : U32))) == True{} : Bool}}
  +rest_below = {%Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),rest_exact(count,more,within)) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
    Order.last_below(U32.to_nat(count),count_positive(count,more,within)) : {Nat.is_lt(U32.to_nat((count - 1 : U32)),U32.to_nat(count)) == True{} : Bool}}
  +room = {%split : {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,_))) == True{} : Bool} Depth.allocation_room(count,sized) :
    {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,1n+half_depth(count)))) == True{} : Bool}}
  Below{
    {%Equal.sym(Nat,Storage.depth_for(count),1n+half_depth(count),split) : {U32.to_nat(U32.shln(1,Nat.sub(_,1n))) == Bits.power(half_depth(count)) : Nat}
      {%Equal.sym(Nat,Nat.sub(half_depth(count),0n),half_depth(count),Order.subtract_zero(half_depth(count))) : {U32.to_nat(U32.shln(1,_)) == Bits.power(half_depth(count)) : Nat}
        shifted_power(half_depth(count),half_small) : {U32.to_nat(U32.shln(1,Nat.sub(half_depth(count),0n))) == Bits.power(half_depth(count)) : Nat}} :
      {U32.to_nat(U32.shln(1,Nat.sub(Storage.depth_for(count),1n))) == Bits.power(half_depth(count)) : Nat}},
    Order.le_lt_trans(Bits.power(half_depth(count)),U32.to_nat((count - 1 : U32)),U32.to_nat(count),lower,rest_below),
    {%shifted_power(1n+half_depth(count),depth_small) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
      room : {Nat.is_le(U32.to_nat(count),Bits.power(1n+half_depth(count))) == True{} : Bool}}}

# ---- One range for each piece ----

# The ranges of count pieces from piece on, then rest.
def piece_spans(+grid: Grid.Grid,count: Nat,+piece: Nat,+rest: Held.Spans) -> Held.Spans:
  match count:
    case 0n: rest
    case 1n+more: Held.Spanned{Grid.start(grid,U32.from_nat(piece)),Pieces.cells_of(grid,U32.from_nat(piece),1),piece_spans(grid,more,1n+piece,rest)}

def spans_append(+grid: Grid.Grid,+left: Nat,+piece: Nat,+right: Nat,+rest: Held.Spans) ->
  {piece_spans(grid,left,piece,piece_spans(grid,right,Nat.add(piece,left),rest)) == piece_spans(grid,Nat.add(left,right),piece,rest) : Held.Spans}:
  match left:
    case 0n:
      Equal.cong(Nat,Held.Spans,start => piece_spans(grid,right,start,rest),Nat.add(piece,0n),piece,Addition.zero_right(piece))
    case 1n+ +more:
      +inner = Equal.trans(Held.Spans,piece_spans(grid,more,1n+piece,piece_spans(grid,right,Nat.add(piece,1n+more),rest)),
        piece_spans(grid,more,1n+piece,piece_spans(grid,right,Nat.add(1n+piece,more),rest)),piece_spans(grid,Nat.add(more,right),1n+piece,rest),
        Equal.cong(Nat,Held.Spans,start => piece_spans(grid,more,1n+piece,piece_spans(grid,right,start,rest)),Nat.add(piece,1n+more),1n+Nat.add(piece,more),
          Addition.successor_right(piece,more)),
        spans_append(grid,more,1n+piece,right,rest))
      Equal.cong(Held.Spans,Held.Spans,spans => Held.Spanned{Grid.start(grid,U32.from_nat(piece)),Pieces.cells_of(grid,U32.from_nat(piece),1),spans},
        piece_spans(grid,more,1n+piece,piece_spans(grid,right,Nat.add(piece,1n+more),rest)),piece_spans(grid,Nat.add(more,right),1n+piece,rest),inner)

def power_positive(+exponent: Nat) -> {Nat.is_lt(0n,Bits.power(exponent)) == True{} : Bool}:
  match exponent:
    case 0n: {==}
    case 1n+ +rest: Bounds.nat_trans(0n,Bits.power(rest),Nat.double(Bits.power(rest)),power_positive(rest),Order.le_double(Bits.power(rest)))

def double_add(+value: Nat) -> {Nat.double(value) == Nat.add(value,value) : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(Nat,2n+Nat.double(rest),2n+Nat.add(rest,rest),1n+Nat.add(rest,1n+rest),
        Equal.cong(Nat,Nat,sum => 2n+sum,Nat.double(rest),Nat.add(rest,rest),double_add(rest)),
        Equal.cong(Nat,Nat,sum => 1n+sum,1n+Nat.add(rest,rest),Nat.add(rest,1n+rest),Equal.sym(Nat,Nat.add(rest,1n+rest),1n+Nat.add(rest,rest),Addition.successor_right(rest,rest))))

# What lies past half of at most twice a value is at most the value.
def past_half(+whole: Nat,+half: Nat,+ordered: {Nat.is_le(half,whole) == True{} : Bool},+cover: {Nat.is_le(whole,Nat.double(half)) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(whole,half),half) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(Nat.sub(whole,half),half),Nat.is_le(Nat.add(half,Nat.sub(whole,half)),Nat.add(half,half)),True{},
    Equal.sym(Bool,Nat.is_le(Nat.add(half,Nat.sub(whole,half)),Nat.add(half,half)),Nat.is_le(Nat.sub(whole,half),half),Order.le_offset(half,Nat.sub(whole,half),half)),
    Held.nat_le_moved(Nat.add(half,Nat.sub(whole,half)),whole,Nat.add(half,half),Order.restore_shift(half,whole,ordered),
      Held.nat_le_right_moved(whole,Nat.add(half,half),Nat.double(half),Equal.sym(Nat,Nat.double(half),Nat.add(half,half),double_add(half)),cover)))

def more_than_one(+count: U32,+test: {U32.is_le(count,1) == False{} : Bool}) -> {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}:
  Bits.true_unless(Nat.is_lt(1n,U32.to_nat(count)),outside => WordEncoding.false_true(
    Equal.trans(Bool,False{},U32.is_le(count,1),True{},Equal.sym(Bool,U32.is_le(count,1),False{},test),
      Equal.trans(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),1n),True{},Comparison.le(count,1),Order.not_less(1n,U32.to_nat(count),outside)))))

# A range of one piece is that piece.
def leaf_spans(+grid: Grid.Grid,+first: U32,+count: U32,+rest: Held.Spans,+test: {U32.is_le(count,1) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {Held.layout_spans(Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)},rest) == piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest) : Held.Spans}:
  +one = Arithmetic.to_nat_injective(count,1,Order.antisymmetric(U32.to_nat(count),1n,Bounds.as_nat_le(count,1,test),Order.positive_one(U32.to_nat(count),positive)))
  %Equal.sym(U32,count,1,one) : {Held.layout_spans(Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,_)},rest) == piece_spans(grid,U32.to_nat(_),U32.to_nat(first),rest) : Held.Spans}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)) :
    {Held.Spanned{Grid.start(grid,first),Pieces.cells_of(grid,first,1),rest} == Held.Spanned{Grid.start(grid,_),Pieces.cells_of(grid,_,1),rest} : Held.Spans}
  {==}

# The halves of a split of a count of at least two and at most two to one
# more than depth: both positive and at most two to depth, the left at most
# the count.
def SplitHalves(+count: U32,+depth: Nat) -> Data:
  Held.Both<{Nat.is_lt(0n,U32.to_nat(Pieces.below(count))) == True{} : Bool},Held.Both<{Nat.is_le(U32.to_nat(Pieces.below(count)),Bits.power(depth)) == True{} : Bool},
    Held.Both<{Nat.is_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool},Held.Both<{U32.to_nat((count - Pieces.below(count) : U32)) == Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))) : Nat},
      Held.Both<{Nat.is_lt(0n,U32.to_nat((count - Pieces.below(count) : U32))) == True{} : Bool},{Nat.is_le(U32.to_nat((count - Pieces.below(count) : U32)),Bits.power(depth)) == True{} : Bool}>>>>>

def halves_of(+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+more: Nat,+count: U32,+test: {U32.is_le(count,1) == False{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(count),Bits.power(1n+more)) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool}) -> SplitHalves(count,more):
  +several = more_than_one(count,test)
  +within = Coordinates.native_le_transitive(count,cap,1073741824,Held.u32_le(count,cap,inside),cap_ok)
  +halves = below_facts(count,several,within)
  +exact = below_exact(count,halves)
  +under = below_under(count,halves)
  +cover = below_cover(count,halves)
  +left_le = Bounds.strict_le(Bits.power(half_depth(count)),U32.to_nat(count),under)
  +half_le = {%length_split(count,several,within) : {Nat.is_le(_,1n+more) == True{} : Bool}
    Bits.length_upper(32n,(count - 1 : U32),1n+more,
      {%Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),rest_exact(count,several,within)) : {Nat.is_lt(_,Bits.power(1n+more)) == True{} : Bool}
        Bounds.nat_trans(Nat.sub(U32.to_nat(count),1n),U32.to_nat(count),Bits.power(1n+more),Order.last_below(U32.to_nat(count),positive),bounded) :
        {Nat.is_lt(U32.to_nat((count - 1 : U32)),Bits.power(1n+more)) == True{} : Bool}}) :
    {Nat.is_le(half_depth(count),more) == True{} : Bool}}
  +left_bound = Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(half_depth(count)),Bits.power(more),exact,Bits.power_monotone(half_depth(count),more,half_le))
  +left_positive = {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(half_depth(count)),exact) : {Nat.is_lt(0n,_) == True{} : Bool} power_positive(half_depth(count)) :
    {Nat.is_lt(0n,U32.to_nat(Pieces.below(count))) == True{} : Bool}}
  +left_count = Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(half_depth(count)),U32.to_nat(count),exact,left_le)
  +right_exact = Arithmetic.subtract(count,Pieces.below(count),cap,Coordinates.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),left_count,Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat(count),U32.to_nat(cap),Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count))),inside))
  +right_positive = {%Equal.sym(Nat,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_difference(U32.to_nat(count),U32.to_nat(Pieces.below(count)),{%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(half_depth(count)),exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} under :
      {Nat.is_lt(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}}) : {Nat.is_lt(0n,U32.to_nat((count - Pieces.below(count) : U32))) == True{} : Bool}}
  +right_bound = Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(more),right_exact,
    Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat(Pieces.below(count)),Bits.power(more),
      past_half(U32.to_nat(count),U32.to_nat(Pieces.below(count)),left_count,{%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(half_depth(count)),exact) : {Nat.is_le(U32.to_nat(count),Nat.double(_)) == True{} : Bool} cover :
        {Nat.is_le(U32.to_nat(count),Nat.double(U32.to_nat(Pieces.below(count)))) == True{} : Bool}}),left_bound))
  Held.Both{left_positive,Held.Both{left_bound,Held.Both{left_count,Held.Both{right_exact,Held.Both{right_positive,right_bound}}}}}

def halves_left_positive(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {Nat.is_lt(0n,U32.to_nat(Pieces.below(count))) == True{} : Bool}:
  Held.Both{+left_positive,+rest0} = halves
  left_positive

def halves_left_bound(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {Nat.is_le(U32.to_nat(Pieces.below(count)),Bits.power(depth)) == True{} : Bool}:
  Held.Both{+left_positive,+rest0} = halves
  Held.Both{+left_bound,+rest1} = rest0
  left_bound

def halves_left_count(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {Nat.is_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}:
  Held.Both{+left_positive,+rest0} = halves
  Held.Both{+left_bound,+rest1} = rest0
  Held.Both{+left_count,+rest2} = rest1
  left_count

def halves_right_exact(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {U32.to_nat((count - Pieces.below(count) : U32)) == Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))) : Nat}:
  Held.Both{+left_positive,+rest0} = halves
  Held.Both{+left_bound,+rest1} = rest0
  Held.Both{+left_count,+rest2} = rest1
  Held.Both{+right_exact,+rest3} = rest2
  right_exact

def halves_right_positive(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {Nat.is_lt(0n,U32.to_nat((count - Pieces.below(count) : U32))) == True{} : Bool}:
  Held.Both{+left_positive,+rest0} = halves
  Held.Both{+left_bound,+rest1} = rest0
  Held.Both{+left_count,+rest2} = rest1
  Held.Both{+right_exact,+rest3} = rest2
  Held.Both{+right_positive,+rest4} = rest3
  right_positive

def halves_right_bound(+count: U32,+depth: Nat,+halves: SplitHalves(count,depth)) -> {Nat.is_le(U32.to_nat((count - Pieces.below(count) : U32)),Bits.power(depth)) == True{} : Bool}:
  Held.Both{+left_positive,+rest0} = halves
  Held.Both{+left_bound,+rest1} = rest0
  Held.Both{+left_count,+rest2} = rest1
  Held.Both{+right_exact,+rest3} = rest2
  Held.Both{+right_positive,+rest4} = rest3
  rest4

# The right half of a split starts where the left one ends.
def split_sum(+first: U32,+count: U32,
  +first_exact: {U32.to_nat((first + Pieces.below(count) : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))) : Nat},
  +right_exact: {U32.to_nat((count - Pieces.below(count) : U32)) == Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))) : Nat},
  +left_count: {Nat.is_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Equal.trans(Nat,Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),first_exact),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),value),U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right_exact)),
    Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(Pieces.below(count)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))))),Nat.add(U32.to_nat(first),U32.to_nat(count)),
      Addition.associative(U32.to_nat(first),U32.to_nat(Pieces.below(count)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),
      Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(Pieces.below(count)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),U32.to_nat(count),Order.restore_shift(U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count))))

# The ranges of count pieces from first, one for each piece in order.
def inner_spans(+grid: Grid.Grid,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+depth: Nat,+single: Bool,+first: U32,+count: U32,+rest: Held.Spans,+test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}) -> {Held.layout_spans(Pieces.inner_ranges(1n+depth,single,grid,first,count),rest) == piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest) : Held.Spans}:
  match depth single:
    case 0n True{}: leaf_spans(grid,first,count,rest,test,positive)
    case 0n False{}:
      Empty.absurd({Held.layout_spans(Pieces.inner_ranges(1n,False{},grid,first,count),rest) == piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest) : Held.Spans},
        Held.lt_le_absurd(1n,U32.to_nat(count),more_than_one(count,test),bounded))
    case 1n+ +more True{}: leaf_spans(grid,first,count,rest,test,positive)
    case 1n+ +more False{}:
      +inside = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.right_le(U32.to_nat(first),U32.to_nat(count)),room)
      +halves = halves_of(cap,cap_ok,more,count,test,positive,bounded,inside)
      +left_positive = halves_left_positive(count,more,halves)
      +left_bound = halves_left_bound(count,more,halves)
      +left_count = halves_left_count(count,more,halves)
      +right_exact = halves_right_exact(count,more,halves)
      +right_positive = halves_right_positive(count,more,halves)
      +right_bound = halves_right_bound(count,more,halves)
      +left_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.offset_le(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count),room)
      +first_exact = Arithmetic.add(first,Pieces.below(count),cap,Coordinates.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),left_room)
      +right_sum = split_sum(first,count,first_exact,right_exact,left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),right_sum,room)
      +right_spans = inner_spans(grid,cap,cap_ok,more,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),rest,{==},right_positive,right_bound,right_room)
      +left_spans = inner_spans(grid,cap,cap_ok,more,U32.is_le(Pieces.below(count),1),first,Pieces.below(count),Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest),{==},left_positive,left_bound,left_room)
      Equal.trans(Held.Spans,Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest),left_spans,
        Equal.trans(Held.Spans,piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat((first + Pieces.below(count) : U32)),rest)),piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest),
          Equal.cong(Held.Spans,Held.Spans,spans => piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),spans),Held.layout_spans(Pieces.inner_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat((first + Pieces.below(count) : U32)),rest),right_spans),
          Equal.trans(Held.Spans,piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat((first + Pieces.below(count) : U32)),rest)),piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),rest)),piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest),
            Equal.cong(Nat,Held.Spans,start => piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),start,rest)),U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),first_exact),
            Equal.trans(Held.Spans,piece_spans(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),piece_spans(grid,U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),rest)),piece_spans(grid,Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(first),rest),piece_spans(grid,U32.to_nat(count),U32.to_nat(first),rest),
              spans_append(grid,U32.to_nat(Pieces.below(count)),U32.to_nat(first),U32.to_nat((count - Pieces.below(count) : U32)),rest),
              Equal.cong(Nat,Held.Spans,total => piece_spans(grid,total,U32.to_nat(first),rest),Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(count),
                Equal.trans(Nat,Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(Pieces.below(count)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),U32.to_nat(count),
                  Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(Pieces.below(count)),value),U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right_exact),Order.restore_shift(U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count)))))))

# The pieces of one outer index: per of them from first * per.
def outer_leaf(+grid: Grid.Grid,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+per: U32,+per_positive: {Nat.is_lt(0n,U32.to_nat(per)) == True{} : Bool},
  +first: U32,+count: U32,+rest: Held.Spans,+test: {U32.is_le(count,1) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Held.layout_spans(Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),rest) == piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest) : Held.Spans}:
  +one = Arithmetic.to_nat_injective(count,1,Order.antisymmetric(U32.to_nat(count),1n,Bounds.as_nat_le(count,1,test),Order.positive_one(U32.to_nat(count),positive)))
  +spread = Equal.trans(Nat,Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(1n,U32.to_nat(per))),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),RowMajor.distrib(U32.to_nat(first),1n,U32.to_nat(per)),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),value),Nat.add(U32.to_nat(per),0n),U32.to_nat(per),Addition.zero_right(U32.to_nat(per))))
  +one_room = {%one : {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(_)),U32.to_nat(per)),U32.to_nat(cap)) == True{} : Bool} room :
    {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),U32.to_nat(cap)) == True{} : Bool}}
  +product_room = Order.transitive(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),U32.to_nat(cap),Order.le_add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),Held.nat_le_moved(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),U32.to_nat(cap),Equal.sym(Nat,Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),spread),one_room))
  +start_exact = Multiplication.bounded_multiply(first,per,cap,Coordinates.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),product_room)
  +inner_room = Held.nat_le_moved(Nat.add(U32.to_nat((first * per : U32)),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),U32.to_nat(cap),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(per)),U32.to_nat((first * per : U32)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),start_exact),
    Held.nat_le_moved(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),U32.to_nat(cap),Equal.sym(Nat,Nat.mul(Nat.add(U32.to_nat(first),1n),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),U32.to_nat(per)),spread),one_room))
  +per_cap = Order.transitive(U32.to_nat(per),Nat.add(U32.to_nat((first * per : U32)),U32.to_nat(per)),U32.to_nat(cap),Order.right_le(U32.to_nat((first * per : U32)),U32.to_nat(per)),inner_room)
  +per_bound = Order.transitive(U32.to_nat(per),Bits.power(30n),Bits.power(31n),
    Held.nat_le_right_moved(U32.to_nat(per),Bits.power(30n),U32.to_nat(U32.shln(1,30n)),Equal.sym(Nat,U32.to_nat(U32.shln(1,30n)),Bits.power(30n),shifted_power(30n,{==})),
      Order.transitive(U32.to_nat(per),U32.to_nat(cap),U32.to_nat(U32.shln(1,30n)),per_cap,Bounds.as_nat_le(cap,U32.shln(1,30n),cap_ok))),
    Bits.power_monotone(30n,31n,{==}))
  +spans = inner_spans(grid,cap,cap_ok,31n,U32.is_le(per,1),(first * per : U32),per,rest,{==},per_positive,per_bound,inner_room)
  %Equal.sym(U32,count,1,one) : {Held.layout_spans(Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),rest) == piece_spans(grid,Nat.mul(U32.to_nat(_),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest) : Held.Spans}
  Equal.trans(Held.Spans,Held.layout_spans(Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),rest),piece_spans(grid,U32.to_nat(per),U32.to_nat((first * per : U32)),rest),
    piece_spans(grid,Nat.add(U32.to_nat(per),0n),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),spans,
    Equal.trans(Held.Spans,piece_spans(grid,U32.to_nat(per),U32.to_nat((first * per : U32)),rest),piece_spans(grid,U32.to_nat(per),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),piece_spans(grid,Nat.add(U32.to_nat(per),0n),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),
      Equal.cong(Nat,Held.Spans,start => piece_spans(grid,U32.to_nat(per),start,rest),U32.to_nat((first * per : U32)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),start_exact),
      Equal.cong(Nat,Held.Spans,total => piece_spans(grid,total,Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),U32.to_nat(per),Nat.add(U32.to_nat(per),0n),Equal.sym(Nat,Nat.add(U32.to_nat(per),0n),U32.to_nat(per),Addition.zero_right(U32.to_nat(per))))))

# The ranges of count outer indices from first, one for each piece in order.
def outer_spans(+grid: Grid.Grid,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+per: U32,+per_positive: {Nat.is_lt(0n,U32.to_nat(per)) == True{} : Bool},+depth: Nat,+single: Bool,+first: U32,+count: U32,+rest: Held.Spans,+test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap)) == True{} : Bool}) -> {Held.layout_spans(Pieces.outer_ranges(1n+depth,single,grid,per,first,count),rest) == piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest) : Held.Spans}:
  match depth single:
    case 0n True{}: outer_leaf(grid,cap,cap_ok,per,per_positive,first,count,rest,test,positive,room)
    case 0n False{}:
      Empty.absurd({Held.layout_spans(Pieces.outer_ranges(1n,False{},grid,per,first,count),rest) == piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest) : Held.Spans},
        Held.lt_le_absurd(1n,U32.to_nat(count),more_than_one(count,test),bounded))
    case 1n+ +more True{}: outer_leaf(grid,cap,cap_ok,per,per_positive,first,count,rest,test,positive,room)
    case 1n+ +more False{}:
      +inside = Order.transitive(U32.to_nat(count),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap),
        Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),Order.right_le(U32.to_nat(first),U32.to_nat(count)),RowMajor.product_covers(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per),per_positive)),room)
      +halves = halves_of(cap,cap_ok,more,count,test,positive,bounded,inside)
      +left_positive = halves_left_positive(count,more,halves)
      +left_bound = halves_left_bound(count,more,halves)
      +left_count = halves_left_count(count,more,halves)
      +right_exact = halves_right_exact(count,more,halves)
      +right_positive = halves_right_positive(count,more,halves)
      +right_bound = halves_right_bound(count,more,halves)
      +sum_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),Order.offset_le(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count),
        RowMajor.product_covers(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per),per_positive))
      +left_room = Order.transitive(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat(per)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap),RowMajor.mul_le(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per),Order.offset_le(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count)),room)
      +first_exact = Arithmetic.add(first,Pieces.below(count),cap,Coordinates.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap),sum_room,room))
      +right_sum = split_sum(first,count,first_exact,right_exact,left_count)
      +right_room = Held.nat_le_moved(Nat.mul(Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(per)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(per)),U32.to_nat(cap),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(per)),Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count)),right_sum),room)
      +right_spans = outer_spans(grid,cap,cap_ok,per,per_positive,more,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),rest,{==},right_positive,right_bound,right_room)
      +left_spans = outer_spans(grid,cap,cap_ok,per,per_positive,more,U32.is_le(Pieces.below(count),1),first,Pieces.below(count),Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest),{==},left_positive,left_bound,left_room)
      +start_split = Equal.trans(Nat,Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat(per)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per))),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(per)),U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),first_exact),RowMajor.distrib(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(per)))
      +count_join = Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per))),Nat.mul(Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(per)),Nat.mul(U32.to_nat(count),U32.to_nat(per)),
        Equal.sym(Nat,Nat.mul(Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per))),RowMajor.distrib(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per))),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(per)),Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat(count),
          Equal.trans(Nat,Nat.add(U32.to_nat(Pieces.below(count)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(Pieces.below(count)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),U32.to_nat(count),
            Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(Pieces.below(count)),value),U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right_exact),Order.restore_shift(U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count))))
      Equal.trans(Held.Spans,Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le(Pieces.below(count),1),grid,per,first,Pieces.below(count)),Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),left_spans,
        Equal.trans(Held.Spans,piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest)),piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat(per)),rest)),piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),
          Equal.cong(Held.Spans,Held.Spans,spans => piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),spans),Held.layout_spans(Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),rest),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat(per)),rest),right_spans),
          Equal.trans(Held.Spans,piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat(per)),rest)),piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per))),rest)),piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),
            Equal.cong(Nat,Held.Spans,start => piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),start,rest)),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per))),start_split),
            Equal.trans(Held.Spans,piece_spans(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),piece_spans(grid,Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per))),rest)),piece_spans(grid,Nat.add(Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per))),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),piece_spans(grid,Nat.mul(U32.to_nat(count),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),
              spans_append(grid,Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat(first),U32.to_nat(per)),Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per)),rest),
              Equal.cong(Nat,Held.Spans,total => piece_spans(grid,total,Nat.mul(U32.to_nat(first),U32.to_nat(per)),rest),Nat.add(Nat.mul(U32.to_nat(Pieces.below(count)),U32.to_nat(per)),Nat.mul(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(per))),Nat.mul(U32.to_nat(count),U32.to_nat(per)),count_join)))))

# ---- The ranges of a grid ----

def divided_by_one(+limits: Machine.Limits,+value: U32) -> {(value / 1 : U32) == value : U32}:
  +capacity_positive = Bounds.nat_trans(0n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),LimitBounds.seven_within_capacity(limits)))
  Arithmetic.to_nat_injective((value / 1 : U32),value,
    Equal.trans(Nat,U32.to_nat((value / 1 : U32)),Nat.div(U32.to_nat(value),1n),U32.to_nat(value),Laws.word_quotient(limits,value,1,Order.positive_one(U32.to_nat(Machine.capacity(limits)),capacity_positive),{==}),
      Equal.trans(Nat,Nat.div(U32.to_nat(value),1n),Nat.div(Nat.add(Nat.mul(U32.to_nat(value),1n),0n),1n),U32.to_nat(value),
        Equal.cong(Nat,Nat,whole => Nat.div(whole,1n),U32.to_nat(value),Nat.add(Nat.mul(U32.to_nat(value),1n),0n),
          Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(value),1n),0n),U32.to_nat(value),
            Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(value),1n),0n),Nat.mul(U32.to_nat(value),1n),U32.to_nat(value),Addition.zero_right(Nat.mul(U32.to_nat(value),1n)),Order.times_one(U32.to_nat(value))))),
        Radix.quotient(U32.to_nat(value),0n,1n,{==},{==}))))

def capacity_power(+limits: Machine.Limits,+value: Nat,+inside: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(value,Bits.power(31n)) == True{} : Bool}:
  Order.transitive(value,Bits.power(30n),Bits.power(31n),
    Held.nat_le_right_moved(value,Bits.power(30n),U32.to_nat(U32.shln(1,30n)),Equal.sym(Nat,U32.to_nat(U32.shln(1,30n)),Bits.power(30n),shifted_power(30n,{==})),
      Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(U32.shln(1,30n)),inside,Bounds.as_nat_le(Machine.capacity(limits),U32.shln(1,30n),LimitBounds.capacity_allocatable(limits)))),
    Bits.power_monotone(30n,31n,{==}))

# The ranges of a grid of outer indices with per pieces each: one for each piece.
def grid_spans(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>) -> {Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) == piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}) : Held.Spans}:
  Laws.Shape{+outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +pieces_room = Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
  +outer_room = Order.transitive(U32.to_nat(outer),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(outer),U32.to_nat(per),per_positive),pieces_room)
  +groups = Equal.trans(U32,U32.max((outer / 1 : U32),1),U32.max(outer,1),outer,
    Equal.cong(U32,U32,whole => U32.max(whole,1),(outer / 1 : U32),outer,divided_by_one(limits,outer)),
    Laws.maximum_kept(outer,1,Order.positive_one(U32.to_nat(outer),outer_positive)))
  +each = Equal.trans(U32,(U32.max(per,1) * 1 : U32),U32.max(per,1),per,Arithmetic.multiply_one(U32.max(per,1)),
    Laws.maximum_kept(per,1,Order.positive_one(U32.to_nat(per),per_positive)))
  %Equal.sym(U32,U32.max((outer / 1 : U32),1),outer,groups) :
    {Held.layout_spans(Pieces.outer_ranges(32n,U32.is_le(_,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.max(per,1) * 1 : U32),0,_),Held.Ended{}) == piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}) : Held.Spans}
  %Equal.sym(U32,(U32.max(per,1) * 1 : U32),per,each) :
    {Held.layout_spans(Pieces.outer_ranges(32n,U32.is_le(outer,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_,0,outer),Held.Ended{}) == piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}) : Held.Spans}
  outer_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Machine.capacity(limits),LimitBounds.capacity_allocatable(limits),per,per_positive,31n,U32.is_le(outer,1),0,outer,Held.Ended{},{==},outer_positive,
    capacity_power(limits,U32.to_nat(outer),outer_room),pieces_room)

# ---- The layout of a grid ----

# The last piece of the grid starts at its cells.
def end_start(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}:
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +exact = Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
  +whole = Equal.trans(Nat,Nat.div(U32.to_nat((outer * per : U32)),U32.to_nat(per)),Nat.div(Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),U32.to_nat(per)),U32.to_nat(outer),
    Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(per)),U32.to_nat((outer * per : U32)),Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),Equal.trans(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),exact,Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Addition.zero_right(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))))),
    Radix.quotient(U32.to_nat(outer),0n,U32.to_nat(per),per_positive,per_positive))
  +part = Equal.trans(Nat,Nat.mod(U32.to_nat((outer * per : U32)),U32.to_nat(per)),Nat.mod(Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),U32.to_nat(per)),0n,
    Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(per)),U32.to_nat((outer * per : U32)),Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),Equal.trans(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),exact,Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Addition.zero_right(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))))),
    Radix.remainder(U32.to_nat(outer),0n,U32.to_nat(per),per_positive,per_positive))
  +placed = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,(outer * per : U32),Held.nat_le_moved(U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),exact,Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))))
  %Equal.sym(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),Laws.place(Nat.div(U32.to_nat((outer * per : U32)),U32.to_nat(per)),Nat.mod(U32.to_nat((outer * per : U32)),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),placed) : {_ == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}
  %Equal.sym(Nat,Nat.div(U32.to_nat((outer * per : U32)),U32.to_nat(per)),U32.to_nat(outer),whole) : {Laws.place(_,Nat.mod(U32.to_nat((outer * per : U32)),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}
  %Equal.sym(Nat,Nat.mod(U32.to_nat((outer * per : U32)),U32.to_nat(per)),0n,part) : {Laws.place(U32.to_nat(outer),_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Addition.zero_right(Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))))

# A piece of the grid starts at or before its cells.
def start_cells(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(count)) == True{} : Bool}:
  +exact = Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
  Held.nat_le_right_moved(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells,
    Held.nat_le_right_moved(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),Equal.sym(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape)),
      Laws.start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,(outer * per : U32),Held.nat_le_moved(U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),exact,Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))),
        Held.nat_le_right_moved(U32.to_nat(piece),U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),exact,inside))))

def minimum_start(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) -> {U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),count)) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) : Nat}:
  Equal.trans(Nat,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),count)),Nat.min(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Extrema.minimum(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),count),
    NatExtrema.minimum_when_ordered(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(count),start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)))

# The cells of piece k end where piece k + 1 starts.
def piece_cells(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))) : Nat}:
  +room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +next_inside = Order.strict_follow(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside)
  +here_inside = Bounds.strict_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside)
  +here = Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),here_inside,room))
  +next = Bounds.roundtrip(1n+piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),next_inside,room))
  +here_word = Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),here,here_inside)
  +next_word = Held.nat_le_moved(U32.to_nat(U32.from_nat(1n+piece)),1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),next,next_inside)
  +ordered = Laws.start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(piece),U32.from_nat(1n+piece),next_word,
    Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,U32.to_nat(U32.from_nat(1n+piece)),here,
      Held.nat_le_right_moved(piece,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next,Order.successor(piece,piece,Order.reflexive(piece)))))
  +low = minimum_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(piece),here_word)
  +high = Equal.trans(Nat,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),
    Equal.cong(U32,Nat,word => U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},word),count)),(U32.from_nat(piece) + 1 : U32),U32.from_nat(1n+piece),Bounds.increment_encoded(piece)),
    minimum_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(1n+piece),next_word))
  +ends = Held.nat_le_moved(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),low,
    Held.nat_le_right_moved(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),high,ordered))
  +difference = Arithmetic.subtract(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count),U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count),Machine.capacity(limits),LimitBounds.capacity_fits(limits),ends,
    Order.transitive(Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),
      U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(Machine.capacity(limits)),Order.sub_le(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),
      Held.nat_le_moved(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(Machine.capacity(limits)),high,
        Order.transitive(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(1n+piece),next_word),
          Held.nat_le_moved(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),cells,Laws.shape_room(limits,outer,extent,unit_cells,block,per,shape))))))
  +difference_nat = Equal.trans(Nat,U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1)),
    Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),
    Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))),difference,
    Equal.trans(Nat,Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),
      Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),
      Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))),
      Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.from_nat(piece) + 1 : U32)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),high),
      Equal.cong(Nat,Nat,value => Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),value),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),low)))
  Equal.trans(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),
    Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),value),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1)),
      Nat.sub(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))),difference_nat),
    Order.restore_shift(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),ordered))

# The ranges of the pieces from piece k on start where their pieces start and
# end at the cells of the grid.
def spans_along(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+remaining: Nat,+piece: Nat,+full: {Nat.add(piece,remaining) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),piece,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))):
  match remaining:
    case 0n:
      +last = Equal.trans(Nat,piece,Nat.add(piece,0n),U32.to_nat((outer * per : U32)),Equal.sym(Nat,Nat.add(piece,0n),piece,Addition.zero_right(piece)),
        Equal.trans(Nat,Nat.add(piece,0n),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((outer * per : U32)),full,Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape))))
      +word = Arithmetic.from_nat_exact((outer * per : U32),piece,Equal.sym(Nat,piece,U32.to_nat((outer * per : U32)),last))
      Held.Both{last,
        Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),U32.to_nat(count),Equal.cong(U32,Nat,word => U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},word)),U32.from_nat(piece),(outer * per : U32),word),
          Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape),Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells)))}
    case 1n+ +more:
      +inside = {%full : {Nat.is_lt(piece,_) == True{} : Bool} Order.below_sum(piece,more) : {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
      +later = spans_along(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,more,1n+piece,Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
        Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),full))
      Held.Both{{%Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(piece,_) == True{} : Bool} inside : {Nat.is_lt(piece,U32.to_nat((outer * per : U32))) == True{} : Bool}},
        Held.Both{{==},
          {%Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
            Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},more,1n+piece,Held.Ended{}),1n+piece,_)
            later : Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},more,1n+piece,Held.Ended{}),1n+piece,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))))}}}

# The layout of the ranges of a grid is the layout of its pieces.
def grid_along(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) -> Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),0n,0n):
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +first = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,0,Order.zero_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))
  +first_zero = Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0)),Laws.place(Nat.div(0n,U32.to_nat(per)),Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),0n,first,
    Equal.trans(Nat,Laws.place(Nat.div(0n,U32.to_nat(per)),Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Laws.place(0n,Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),0n,
      Equal.cong(Nat,Nat,whole => Laws.place(whole,Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.div(0n,U32.to_nat(per)),0n,Radix.quotient(0n,0n,U32.to_nat(per),per_positive,per_positive)),
      Equal.cong(Nat,Nat,part => Laws.place(0n,part,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mod(0n,U32.to_nat(per)),0n,Radix.remainder(0n,0n,U32.to_nat(per),per_positive,per_positive))))
  %Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)) : Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_,0n,0n)
  %first_zero : Windows.Along(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),0n,_)
  spans_along(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,{==})

# ---- The grid of cell axes ----

# The units of cell axes: positive outer indices, extent and unit cells whose
# product is the cells.
type UnitsFacts<-limits: Machine.Limits,-found: Grid.Units,-total: Nat> is Data:
  UnitsFacts{prefix: U32,outer: U32,extent: U32,unit_cells: U32,exact: {found == Grid.Units{prefix,outer,extent,unit_cells} : Grid.Units},
    outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},
    cells: {Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) == total : Nat}}

def prepended(+head: U32,+tail: List<&2,U32>) -> List<&2,U32>:
  head <> tail

def right_factor_positive(+left: Nat,+right: Nat,+positive: {Nat.is_lt(0n,Nat.mul(left,right)) == True{} : Bool}) -> {Nat.is_lt(0n,right) == True{} : Bool}:
  RowMajor.left_factor_positive(right,left,{%RowMajor.multiply_commutative(left,right) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,Nat.mul(right,left)) == True{} : Bool}})

def units_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,
  +room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> UnitsFacts<limits,Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents))>:
  match extents fits:
    case extents True{}:
      +elements_room = Order.transitive(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits)),
        Held.nat_le_right_moved(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),Nat.mul(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent))),RowMajor.multiply_commutative(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),
          RowMajor.product_covers(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive))),room)
      +exact = TiledPublic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extents,elements_room)
      UnitsFacts{prefix,outer,extent,Grid.product(extents),
        Laws.units_fit(extents,prefix,outer,extent,target),
        outer_positive,extent_positive,
        {%Equal.sym(Nat,U32.to_nat(Grid.product(extents)),ShapeLaws.elements(extents),exact) : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,U32.to_nat(Grid.product(extents))) == True{} : Bool}},
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),value),U32.to_nat(Grid.product(extents)),ShapeLaws.elements(extents),exact)}
    case Nil{} False{}: UnitsFacts{prefix,outer,extent,1,{==},outer_positive,extent_positive,{==},{==}}
    case Con{+axis,Nil{}} False{}:
      UnitsFacts{prefix,outer,extent,axis,{==},outer_positive,extent_positive,
        RowMajor.left_factor_positive(U32.to_nat(axis),1n,nonempty),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),value),U32.to_nat(axis),Nat.mul(U32.to_nat(axis),1n),Equal.sym(Nat,Nat.mul(U32.to_nat(axis),1n),U32.to_nat(axis),Order.times_one(U32.to_nat(axis))))}
    case Con{+axis,Con{+next,+tail}} False{}:
      +others = prepended(next,tail)
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(others),nonempty)
      +others_positive = right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(others),nonempty)
      +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,others}),nonempty),room)
      +outer_exact = Multiplication.bounded_multiply(outer,extent,Machine.capacity(limits),LimitBounds.capacity_fits(limits),outer_room)
      +regrouped = Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(others)),Nat.mul(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis)),ShapeLaws.elements(others)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(value,U32.to_nat(axis)),ShapeLaws.elements(others)),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_exact),
        RowMajor.assoc(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis),ShapeLaws.elements(others)))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,others}),Con{Grid.product(others),Grid.strides(others)},Laws.strides_con(axis,others)) :
        UnitsFacts<limits,Grid.units_from(Con{axis,others},_,False{},prefix,outer,extent,target),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents))>
      %regrouped : UnitsFacts<limits,Grid.units_from(others,Grid.strides(others),U32.is_le(Grid.product(others),target),(prefix + 1 : U32),(outer * extent : U32),axis,target),_>
      units_facts(limits,Con{next,tail},U32.is_le(Grid.product(others),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,
        Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(others)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits)),regrouped,room),
        {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
          Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive) : {Nat.is_lt(0n,U32.to_nat((outer * extent : U32))) == True{} : Bool}},
        axis_positive,others_positive)

def block_below_of(+units: U32,+extent: U32,+block: Laws.Block<units,extent>) -> {Nat.is_le(U32.to_nat(units),U32.to_nat(U32.max(extent,1))) == True{} : Bool}:
  Laws.Block{positive,below} = block
  below

# The shape of the grid of units with cells inside the machine.
def units_shape(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Laws.Shape<limits,outer,extent,unit_cells,Grid.piece_units(unit_cells,extent,target),Grid.divided_up(extent,Grid.piece_units(unit_cells,extent,target))>:
  +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),unit_positive),room)
  +extent_room = Order.transitive(U32.to_nat(extent),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(Machine.capacity(limits)),
    Held.nat_le_right_moved(U32.to_nat(extent),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(U32.to_nat(extent),U32.to_nat(outer)),RowMajor.multiply_commutative(U32.to_nat(outer),U32.to_nat(extent)),RowMajor.product_covers(U32.to_nat(extent),U32.to_nat(outer),outer_positive)),outer_room)
  +blocks = Laws.grid_block(unit_cells,extent,target)
  +block_positive = Laws.block_positive(Grid.piece_units(unit_cells,extent,target),extent,blocks)
  +block_below = Held.nat_le_right_moved(U32.to_nat(Grid.piece_units(unit_cells,extent,target)),U32.to_nat(extent),U32.to_nat(U32.max(extent,1)),
    Equal.cong(U32,Nat,word => U32.to_nat(word),extent,U32.max(extent,1),Equal.sym(U32,U32.max(extent,1),extent,Laws.maximum_kept(extent,1,Order.positive_one(U32.to_nat(extent),extent_positive)))),
    block_below_of(Grid.piece_units(unit_cells,extent,target),extent,blocks))
  +per_exact = Laws.divided_exact(limits,extent,Grid.piece_units(unit_cells,extent,target),extent_room,Order.transitive(U32.to_nat(Grid.piece_units(unit_cells,extent,target)),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),block_below,extent_room),block_positive)
  Laws.Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room}

# The cells of the grid of units as a word.
def units_word(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat((outer * extent * unit_cells : U32)) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}:
  +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),unit_positive),room)
  +outer_exact = Multiplication.bounded_multiply(outer,extent,Machine.capacity(limits),LimitBounds.capacity_fits(limits),outer_room)
  Equal.trans(Nat,U32.to_nat((outer * extent * unit_cells : U32)),Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),
    Multiplication.bounded_multiply((outer * extent : U32),unit_cells,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
      Held.nat_le_moved(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(unit_cells)),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_exact),room)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(unit_cells)),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_exact))

def grid_of_facts(+limits: Machine.Limits,+found: Grid.Units,+total: Nat,+target: U32,+facts: UnitsFacts<limits,found,total>,
  +total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Laws.GridFacts<limits,Grid.grid_of_units(found,target)>:
  UnitsFacts{+prefix,+outer,+extent,+unit_cells,+exact,+outer_positive,+extent_positive,+unit_positive,+cells} = facts
  +room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),total,U32.to_nat(Machine.capacity(limits)),cells,total_room)
  %Equal.sym(Grid.Units,found,Grid.Units{prefix,outer,extent,unit_cells},exact) : Laws.GridFacts<limits,Grid.grid_of_units(_,target)>
  Laws.GridFacts{(outer * extent * unit_cells : U32),prefix,outer,extent,unit_cells,Grid.piece_units(unit_cells,extent,target),Grid.divided_up(extent,Grid.piece_units(unit_cells,extent,target)),
    units_shape(limits,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),{==},units_word(limits,outer,extent,unit_cells,unit_positive,room)}

# The grid of a shape with cells inside the machine.
def grid_facts(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool}) -> Laws.GridFacts<limits,Grid.grid(cells,target)>:
  +total_room = Held.nat_le_moved(Nat.add(ShapeLaws.elements(cells),0n),ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits)),Addition.zero_right(ShapeLaws.elements(cells)),room)
  %Equal.sym(Grid.Grid,Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),
      Laws.grid_units(cells,target)) : Laws.GridFacts<limits,_>
  grid_of_facts(limits,Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),Nat.add(ShapeLaws.elements(cells),0n),target,
    units_facts(limits,cells,U32.is_le(Grid.product(cells),target),0,1,1,target,total_room,{==},{==},nonempty),total_room)

# A grid with its facts, laid out by its own ranges: what the halo walk needs.
def grid_laid(+limits: Machine.Limits,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>) ->
  Windows.Laid<limits,grid,Held.layout_spans(Pieces.ranges_of_grid(grid),Held.Ended{}),Laws.grid_cells(grid)>:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : Windows.Laid<limits,_,Held.layout_spans(Pieces.ranges_of_grid(_),Held.Ended{}),Laws.grid_cells(_)>
  Windows.Laid{Laws.GridFacts{count,prefix,outer,extent,unit_cells,block,per,shape,{==},cells},{==},
    grid_along(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells)}

# ---- Runs of whole units ----

def scaled_units(+outer: Nat,+inner: Nat,+cells: Nat,+units: Nat,+unit: Nat,+exact: {cells == Nat.mul(units,unit) : Nat}) ->
  {Nat.mul(outer,Nat.mul(inner,cells)) == Nat.mul(Nat.mul(Nat.mul(outer,inner),units),unit) : Nat}:
  Equal.trans(Nat,Nat.mul(outer,Nat.mul(inner,cells)),Nat.mul(Nat.mul(outer,inner),cells),Nat.mul(Nat.mul(Nat.mul(outer,inner),units),unit),
    Equal.sym(Nat,Nat.mul(Nat.mul(outer,inner),cells),Nat.mul(outer,Nat.mul(inner,cells)),RowMajor.assoc(outer,inner,cells)),
    Equal.trans(Nat,Nat.mul(Nat.mul(outer,inner),cells),Nat.mul(Nat.mul(outer,inner),Nat.mul(units,unit)),Nat.mul(Nat.mul(Nat.mul(outer,inner),units),unit),
      Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(outer,inner),value),cells,Nat.mul(units,unit),exact),
      Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(outer,inner),units),unit),Nat.mul(Nat.mul(outer,inner),Nat.mul(units,unit)),RowMajor.assoc(Nat.mul(outer,inner),units,unit))))

# A piece starts at a whole number of units when the unit cells are whole units.
def start_units(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) -> {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Nat.mul(Nat.add(Nat.mul(Nat.mul(Nat.div(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(extent)),units),Nat.mul(Nat.mul(Nat.mod(U32.to_nat(piece),U32.to_nat(per)),U32.to_nat(block)),units)),unit) : Nat}:
  +q = Nat.div(U32.to_nat(piece),U32.to_nat(per))
  +j = Nat.mod(U32.to_nat(piece),U32.to_nat(per))
  +placed = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside)
  +x = Nat.mul(Nat.mul(q,U32.to_nat(extent)),units)
  +y = Nat.mul(Nat.mul(j,U32.to_nat(block)),units)
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(Nat.add(x,y),unit),placed,
    Equal.trans(Nat,Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(x,unit),Nat.mul(y,unit)),Nat.mul(Nat.add(x,y),unit),
      Equal.trans(Nat,Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(x,unit),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.add(Nat.mul(x,unit),Nat.mul(y,unit)),
        Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(x,unit),scaled_units(q,U32.to_nat(extent),U32.to_nat(unit_cells),units,unit,unit_exact)),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(x,unit),value),Nat.mul(j,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(y,unit),scaled_units(j,U32.to_nat(block),U32.to_nat(unit_cells),units,unit,unit_exact))),
      Equal.sym(Nat,Nat.mul(Nat.add(x,y),unit),Nat.add(Nat.mul(x,unit),Nat.mul(y,unit)),RowMajor.distrib(x,y,unit))))

# The pieces from piece on chain from its start, each a run of whole units.
def piece_chain(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat},+remaining: Nat,+piece: Nat,+full: {Nat.add(piece,remaining) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),unit,U32.to_nat(count)):
  match remaining:
    case 0n: {==}
    case 1n+ +more:
      +inside = {%full : {Nat.is_lt(piece,_) == True{} : Bool} Order.below_sum(piece,more) : {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
      +room = Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
      +next_inside = Order.strict_follow(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside)
      +here_inside = Bounds.strict_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside)
      +here = Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),here_inside,room))
      +next = Bounds.roundtrip(1n+piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),next_inside,room))
      +here_bound = Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),here,here_inside)
      +next_bound = Held.nat_le_moved(U32.to_nat(U32.from_nat(1n+piece)),1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),next,next_inside)
      +ends = piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)
      +before = Held.nat_lt_left(U32.to_nat(U32.from_nat(piece)),piece,U32.to_nat(U32.from_nat(1n+piece)),here,Held.nat_lt_moved(piece,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next,Order.lt_succ(piece,piece,Order.reflexive(piece))))
      +less = Held.nat_lt_moved(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),ends,Laws.start_order(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(piece),U32.from_nat(1n+piece),next_bound,before))
      +inside_count = Held.nat_le_moved(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),U32.to_nat(count),ends,start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(1n+piece),next_bound))
      +high = Equal.trans(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),Nat.mul(Nat.add(Nat.mul(Nat.mul(Nat.div(U32.to_nat(U32.from_nat(1n+piece)),U32.to_nat(per)),U32.to_nat(extent)),units),Nat.mul(Nat.mul(Nat.mod(U32.to_nat(U32.from_nat(1n+piece)),U32.to_nat(per)),U32.to_nat(block)),units)),unit),ends,start_units(limits,outer,extent,unit_cells,block,per,count,prefix,shape,unit,units,unit_exact,U32.from_nat(1n+piece),next_bound))
      +later = piece_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact,more,1n+piece,Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
        Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),full))
      Held.Both{Runs.Fits{Nat.add(Nat.mul(Nat.mul(Nat.div(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),U32.to_nat(extent)),units),Nat.mul(Nat.mul(Nat.mod(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),U32.to_nat(block)),units)),Nat.add(Nat.mul(Nat.mul(Nat.div(U32.to_nat(U32.from_nat(1n+piece)),U32.to_nat(per)),U32.to_nat(extent)),units),Nat.mul(Nat.mul(Nat.mod(U32.to_nat(U32.from_nat(1n+piece)),U32.to_nat(per)),U32.to_nat(block)),units)),start_units(limits,outer,extent,unit_cells,block,per,count,prefix,shape,unit,units,unit_exact,U32.from_nat(piece),here_bound),high,less,inside_count},Held.Both{{==},
        {%Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),ends) : Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},more,1n+piece,Held.Ended{}),_,unit,U32.to_nat(count))
          later : Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},more,1n+piece,Held.Ended{}),Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),1))),unit,U32.to_nat(count))}}}

# The ranges of a grid whose unit cells are whole units are a run of them.
# The ranges of a grid whose unit cells are whole units are a chain of them from its first cell.
def grid_chain(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat}) -> Runs.Chain(Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(0n))),unit,U32.to_nat(count)):
  +low = U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(0n)))
  {%Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
      grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)) : Runs.Chain(_,low,unit,U32.to_nat(count))
    piece_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,{==}) : Runs.Chain(Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),low,unit,U32.to_nat(count))}

def grid_run(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat}) -> Runs.Run(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),unit,U32.to_nat(count)):
  +low = U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(0n)))
  +chain = grid_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact)
  Runs.part_run(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{},low,unit,U32.to_nat(count),Runs.tree_run(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{},low,unit,U32.to_nat(count),chain))

# ---- The pieces of the starts ----

# The cells of a piece are the cells from its first cell to the next piece.
def span_cells(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+piece: U32) ->
  {Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) ==
    Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece,1) : U32}:
  {==}

# The first cell of a piece lies in that piece.
def piece_at_start(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,
  +inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == piece : U32}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +image = Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))
  +size = Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))
  +q = Nat.div(U32.to_nat(piece),U32.to_nat(per))
  +j = Nat.mod(U32.to_nat(piece),U32.to_nat(per))
  +size_positive = Laws.cells_positive(limits,outer,extent,unit_cells,block,per,shape)
  +image_positive = Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape)
  +inner = {%RowMajor.assoc(j,U32.to_nat(block),U32.to_nat(unit_cells)) : {Nat.is_lt(_,image) == True{} : Bool}
    RowMajor.multiply_lt(Nat.mul(j,U32.to_nat(block)),U32.to_nat(extent),U32.to_nat(unit_cells),
      Laws.inner_last(j,U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),
        Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape),DivisionOrder.remainder_small(U32.to_nat(piece),U32.to_nat(per),per_positive)),
      Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(Nat.mul(j,size),image) == True{} : Bool}}
  +bounded = Held.nat_le_right_moved(U32.to_nat(Grid.start(grid,piece)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),
    Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells),
    start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside))
  +located = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside)
  +whole = Radix.quotient(q,Nat.mul(j,size),image,image_positive,inner)
  +part = Radix.remainder(q,Nat.mul(j,size),image,image_positive,inner)
  +lane = Equal.trans(Nat,Nat.div(Nat.mul(j,size),size),Nat.div(Nat.add(Nat.mul(j,size),0n),size),j,
    Equal.cong(Nat,Nat,value => Nat.div(value,size),Nat.mul(j,size),Nat.add(Nat.mul(j,size),0n),Equal.sym(Nat,Nat.add(Nat.mul(j,size),0n),Nat.mul(j,size),Addition.zero_right(Nat.mul(j,size)))),
    Radix.quotient(j,0n,size,size_positive,size_positive))
  Arithmetic.to_nat_injective(Grid.piece_of(grid,Grid.start(grid,piece)),piece,
    {%Equal.sym(Nat,U32.to_nat(Grid.piece_of(grid,Grid.start(grid,piece))),
        Nat.add(Nat.mul(Nat.div(U32.to_nat(Grid.start(grid,piece)),image),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(Grid.start(grid,piece)),image),size)),
        Laws.piece_word(limits,outer,extent,unit_cells,block,per,count,prefix,(outer * per : U32),shape,Grid.start(grid,piece),bounded)) :
        {_ == U32.to_nat(piece) : Nat}
      %Equal.sym(Nat,U32.to_nat(Grid.start(grid,piece)),Laws.place(q,j,image,size),located) :
        {Nat.add(Nat.mul(Nat.div(_,image),U32.to_nat(per)),Nat.div(Nat.mod(_,image),size)) == U32.to_nat(piece) : Nat}
      %Equal.sym(Nat,Nat.div(Laws.place(q,j,image,size),image),q,whole) :
        {Nat.add(Nat.mul(_,U32.to_nat(per)),Nat.div(Nat.mod(Laws.place(q,j,image,size),image),size)) == U32.to_nat(piece) : Nat}
      %Equal.sym(Nat,Nat.mod(Laws.place(q,j,image,size),image),Nat.mul(j,size),part) :
        {Nat.add(Nat.mul(q,U32.to_nat(per)),Nat.div(_,size)) == U32.to_nat(piece) : Nat}
      %Equal.sym(Nat,Nat.div(Nat.mul(j,size),size),j,lane) :
        {Nat.add(Nat.mul(q,U32.to_nat(per)),_) == U32.to_nat(piece) : Nat}
      Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(q,U32.to_nat(per)),j),DivisionOrder.quotient_exact(U32.to_nat(piece),U32.to_nat(per),per_positive)) :
      {U32.to_nat(Grid.piece_of(grid,Grid.start(grid,piece))) == U32.to_nat(piece) : Nat}})

# A product with one more row is the product and one more row.
def succ_times(+rows: Nat,+width: Nat) -> {Nat.mul(1n+rows,width) == Nat.add(Nat.mul(rows,width),width) : Nat}:
  %Order.commutative(width,Nat.mul(rows,width)) : {Nat.mul(1n+rows,width) == _ : Nat}
  %RowMajor.multiply_commutative(width,rows) : {Nat.mul(1n+rows,width) == Nat.add(width,_) : Nat}
  %RowMajor.multiply_successor_right(width,rows) : {Nat.mul(1n+rows,width) == _ : Nat}
  RowMajor.multiply_commutative(1n+rows,width)

# A value from lane rows of size on and before the next row is in row lane.
def lane_between(+value: Nat,+lane: Nat,+size: Nat,+positive: {Nat.is_lt(0n,size) == True{} : Bool},+low: {Nat.is_le(Nat.mul(lane,size),value) == True{} : Bool},
  +high: {Nat.is_lt(value,Nat.add(Nat.mul(lane,size),size)) == True{} : Bool}) -> {Nat.div(value,size) == lane : Nat}:
  %Order.restore_shift(Nat.mul(lane,size),value,low) : {Nat.div(_,size) == lane : Nat}
  Radix.quotient(lane,Nat.sub(value,Nat.mul(lane,size)),size,positive,Order.sub_lt(value,Nat.mul(lane,size),size,low,high))

# A cell row images on and within cells into its image is in piece row * per
# plus the pieces of its image before it.
def piece_located(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+index: U32,+bounded: {Nat.is_le(U32.to_nat(index),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells))) == True{} : Bool},
  +row: Nat,+within: Nat,+lane: Nat,+exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within) : Nat},+small: {Nat.is_lt(within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool},
  +lane_exact: {Nat.div(within,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == lane : Nat}) -> {U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}:
  +image_positive = Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape)
  %Equal.sym(Nat,U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(U32.to_nat(index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))),
      Laws.piece_word(limits,outer,extent,unit_cells,block,per,count,prefix,(outer * per : U32),shape,index,bounded)) :
    {_ == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}
  %Equal.sym(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),exact) :
    {Nat.add(Nat.mul(Nat.div(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(per)),Nat.div(Nat.mod(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),row,Radix.quotient(row,within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive,small)) :
    {Nat.add(Nat.mul(_,U32.to_nat(per)),Nat.div(Nat.mod(Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within,Radix.remainder(row,within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive,small)) :
    {Nat.add(Nat.mul(row,U32.to_nat(per)),Nat.div(_,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}
  %Equal.sym(Nat,Nat.div(within,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),lane,lane_exact) : {Nat.add(Nat.mul(row,U32.to_nat(per)),_) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}
  {==}

# The cell before a piece that starts one cell past within cells of row
# images is in the piece of that cell.
def last_from(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+row: Nat,+within: Nat,+lane: Nat,
  +split: {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within) : Nat},+small: {Nat.is_lt(within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool},
  +lane_exact: {Nat.div(within,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == lane : Nat}) -> {U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) - 1 : U32))) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +start = U32.to_nat(Grid.start(grid,piece))
  +start_room = Held.nat_le_right_moved(start,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells),
    start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside))
  +start_cap = Order.transitive(start,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(cap),start_room,Laws.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  +positive = {%Equal.sym(Nat,start,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),split) : {Nat.is_le(1n,_) == True{} : Bool}
    {%Equal.sym(Nat,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),1n+Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),Addition.successor_right(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within)) :
        {Nat.is_le(1n,_) == True{} : Bool}
      Order.le_add(1n,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within)) : {Nat.is_le(1n,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within)) == True{} : Bool}} :
    {Nat.is_le(1n,start) == True{} : Bool}}
  +sub_word = Arithmetic.subtract(Grid.start(grid,piece),1,cap,cap_ok,positive,Order.transitive(Nat.sub(start,1n),start,U32.to_nat(cap),Order.sub_le(start,1n),start_cap))
  +exact = Equal.trans(Nat,U32.to_nat((Grid.start(grid,piece) - 1 : U32)),Nat.sub(start,1n),Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),sub_word,
    Equal.trans(Nat,Nat.sub(start,1n),Nat.sub(Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),1n),Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within),
      Equal.cong(Nat,Nat,value => Nat.sub(value,1n),start,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),split),
      Order.subtract_one_complement(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),within)))
  +bounded = Order.transitive(U32.to_nat((Grid.start(grid,piece) - 1 : U32)),start,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),
    {%Equal.sym(Nat,U32.to_nat((Grid.start(grid,piece) - 1 : U32)),Nat.sub(start,1n),sub_word) : {Nat.is_le(_,start) == True{} : Bool}
      Order.sub_le(start,1n) : {Nat.is_le(U32.to_nat((Grid.start(grid,piece) - 1 : U32)),start) == True{} : Bool}},start_room)
  piece_located(limits,outer,extent,unit_cells,block,per,count,prefix,shape,(Grid.start(grid,piece) - 1 : U32),bounded,row,within,lane,exact,small,lane_exact)

# The cell before a piece one past row images and lane pieces: inside the
# image when the piece is not the first of its image, else its last cell.
def last_located_with(+choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+row: Nat,+lane: Nat,
  +test: {Nat.is_lt(1n+lane,U32.to_nat(per)) == choice : Bool},+lane_small: {Nat.is_lt(lane,U32.to_nat(per)) == True{} : Bool},
  +moved: {U32.to_nat(piece) == Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane) : Nat}) -> {U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) - 1 : U32))) == Nat.add(Nat.mul(row,U32.to_nat(per)),lane) : Nat}:
  match choice:
    case True{}:
      +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
      +size_positive = Laws.cells_positive(limits,outer,extent,unit_cells,block,per,shape)
      +image_positive = Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape)
      +located = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside)
      +piece_row = {%Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane),moved) : {Nat.div(_,U32.to_nat(per)) == row : Nat}
        Radix.quotient(row,1n+lane,U32.to_nat(per),per_positive,test) : {Nat.div(U32.to_nat(piece),U32.to_nat(per)) == row : Nat}}
      +piece_lane = {%Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane),moved) : {Nat.mod(_,U32.to_nat(per)) == 1n+lane : Nat}
        Radix.remainder(row,1n+lane,U32.to_nat(per),per_positive,test) : {Nat.mod(U32.to_nat(piece),U32.to_nat(per)) == 1n+lane : Nat}}
      +within = Nat.add(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n))
      +size_last = Order.positive_has_last(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),size_positive)
      +block_exact = {%Addition.successor_right(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n)) : {Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == _ : Nat}
        {%Equal.sym(Nat,1n+Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),size_last) : {Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.add(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),_) : Nat}
          succ_times(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : {Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.add(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),1n+Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n)) : Nat}} : {Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == 1n+within : Nat}}
      +start_split = Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Laws.place(row,1n+lane,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),
        {%piece_row : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(_,1n+lane,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}
          {%piece_lane : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(Nat.div(U32.to_nat(piece),U32.to_nat(per)),_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}
            located : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(Nat.div(U32.to_nat(piece),U32.to_nat(per)),1n+lane,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}} :
          {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(row,1n+lane,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}},
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),value),Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),1n+within,block_exact))
      +block_lt = {%RowMajor.assoc(1n+lane,U32.to_nat(block),U32.to_nat(unit_cells)) : {Nat.is_lt(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}
        RowMajor.multiply_lt(Nat.mul(1n+lane,U32.to_nat(block)),U32.to_nat(extent),U32.to_nat(unit_cells),
          Laws.inner_last(1n+lane,U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape),test),Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape)) :
        {Nat.is_lt(Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}}
      +small = Order.le_lt_trans(within,Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),
        {%Equal.sym(Nat,Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),1n+within,block_exact) : {Nat.is_le(within,_) == True{} : Bool}
          Order.successor(within,within,Order.reflexive(within)) : {Nat.is_le(within,Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}},block_lt)
      +below = {%size_last : {Nat.is_lt(Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),_) == True{} : Bool}
        Order.lt_succ(Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),Order.reflexive(Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n))) : {Nat.is_lt(Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == True{} : Bool}}
      last_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,row,within,lane,start_split,small,Radix.quotient(lane,Nat.sub(Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),1n),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),size_positive,below))
    case False{}:
      +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
      +size_positive = Laws.cells_positive(limits,outer,extent,unit_cells,block,per,shape)
      +image_positive = Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape)
      +located = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside)
      +full = DivisionOrder.not_below(lane,U32.to_nat(per),lane_small,test)
      +piece_rows = {%Equal.sym(Nat,Nat.add(Nat.mul(1n+row,U32.to_nat(per)),0n),Nat.mul(1n+row,U32.to_nat(per)),Addition.zero_right(Nat.mul(1n+row,U32.to_nat(per)))) : {U32.to_nat(piece) == _ : Nat}
        {%Equal.sym(Nat,Nat.mul(1n+row,U32.to_nat(per)),Nat.add(Nat.mul(row,U32.to_nat(per)),U32.to_nat(per)),succ_times(row,U32.to_nat(per))) : {U32.to_nat(piece) == _ : Nat}
          {%full : {U32.to_nat(piece) == Nat.add(Nat.mul(row,U32.to_nat(per)),_) : Nat}
            moved : {U32.to_nat(piece) == Nat.add(Nat.mul(row,U32.to_nat(per)),U32.to_nat(per)) : Nat}} : {U32.to_nat(piece) == Nat.mul(1n+row,U32.to_nat(per)) : Nat}} :
        {U32.to_nat(piece) == Nat.add(Nat.mul(1n+row,U32.to_nat(per)),0n) : Nat}}
      +piece_row = {%Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(1n+row,U32.to_nat(per)),0n),piece_rows) : {Nat.div(_,U32.to_nat(per)) == 1n+row : Nat}
        Radix.quotient(1n+row,0n,U32.to_nat(per),per_positive,per_positive) : {Nat.div(U32.to_nat(piece),U32.to_nat(per)) == 1n+row : Nat}}
      +piece_lane = {%Equal.sym(Nat,U32.to_nat(piece),Nat.add(Nat.mul(1n+row,U32.to_nat(per)),0n),piece_rows) : {Nat.mod(_,U32.to_nat(per)) == 0n : Nat}
        Radix.remainder(1n+row,0n,U32.to_nat(per),per_positive,per_positive) : {Nat.mod(U32.to_nat(piece),U32.to_nat(per)) == 0n : Nat}}
      +within = Nat.sub(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),1n)
      +image_last = Order.positive_has_last(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_positive)
      +image_exact = {%Equal.sym(Nat,1n+within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_last) : {Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),_) : Nat}
        {%succ_times(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : {Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == _ : Nat}
          Addition.zero_right(Nat.mul(1n+row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))) : {Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat}} :
        {Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within) : Nat}}
      +start_split = Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),1n+within),
        {%piece_row : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(_,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}
          {%piece_lane : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(Nat.div(U32.to_nat(piece),U32.to_nat(per)),_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}
            located : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(Nat.div(U32.to_nat(piece),U32.to_nat(per)),0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}} :
          {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Laws.place(1n+row,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}},image_exact)
      +small = {%image_last : {Nat.is_lt(within,_) == True{} : Bool}
        Order.lt_succ(within,within,Order.reflexive(within)) : {Nat.is_lt(within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}}
      +lane_lt = {%RowMajor.assoc(lane,U32.to_nat(block),U32.to_nat(unit_cells)) : {Nat.is_lt(_,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}
        RowMajor.multiply_lt(Nat.mul(lane,U32.to_nat(block)),U32.to_nat(extent),U32.to_nat(unit_cells),
          Laws.inner_last(lane,U32.to_nat(per),U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape),lane_small),Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape)) :
        {Nat.is_lt(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}}
      +low = {%Equal.sym(Nat,1n+within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),image_last) : {Nat.is_le(1n+Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),_) == True{} : Bool}
        Order.strict_follow(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),lane_lt) : {Nat.is_le(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),within) == True{} : Bool}}
      +image_cover = {%RowMajor.assoc(U32.to_nat(per),U32.to_nat(block),U32.to_nat(unit_cells)) : {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),_) == True{} : Bool}
        RowMajor.mul_le(U32.to_nat(extent),Nat.mul(U32.to_nat(per),U32.to_nat(block)),U32.to_nat(unit_cells),
          {%Equal.sym(Nat,U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_le(U32.to_nat(extent),Nat.mul(_,U32.to_nat(block))) == True{} : Bool}
            NatCeiling.covers(U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape)) :
            {Nat.is_le(U32.to_nat(extent),Nat.mul(U32.to_nat(per),U32.to_nat(block))) == True{} : Bool}}) :
        {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(per),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}}
      +high = {%succ_times(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : {Nat.is_lt(within,_) == True{} : Bool}
        {%Equal.sym(Nat,1n+lane,U32.to_nat(per),full) : {Nat.is_lt(within,Nat.mul(_,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}
          Order.lt_le(within,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(per),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),small,image_cover) : {Nat.is_lt(within,Nat.mul(1n+lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}} :
        {Nat.is_lt(within,Nat.add(Nat.mul(lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}}
      last_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,row,within,lane,start_split,small,lane_between(within,lane,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),size_positive,low,high))

# The cell before a piece after the first is in the piece before it.
def piece_last(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+inside: {Nat.is_le(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(piece)) == True{} : Bool}) ->
  {Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) - 1 : U32)) == (piece - 1 : U32) : U32}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +one_le = Order.positive_one(U32.to_nat(piece),positive)
  +before = Nat.sub(U32.to_nat(piece),1n)
  +before_word = Arithmetic.subtract(piece,1,cap,cap_ok,one_le,
    Order.transitive(before,U32.to_nat(piece),U32.to_nat(cap),Order.sub_le(U32.to_nat(piece),1n),
      Order.transitive(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),inside,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))))
  +row = Nat.div(before,U32.to_nat(per))
  +lane = Nat.mod(before,U32.to_nat(per))
  +before_exact = DivisionOrder.quotient_exact(before,U32.to_nat(per),per_positive)
  +moved = Equal.trans(Nat,U32.to_nat(piece),1n+before,Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane),
    Equal.sym(Nat,1n+before,U32.to_nat(piece),Order.restore_shift(1n,U32.to_nat(piece),one_le)),
    {%Equal.sym(Nat,Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane),1n+Nat.add(Nat.mul(row,U32.to_nat(per)),lane),Addition.successor_right(Nat.mul(row,U32.to_nat(per)),lane)) : {1n+before == _ : Nat}
      Equal.cong(Nat,Nat,value => 1n+value,before,Nat.add(Nat.mul(row,U32.to_nat(per)),lane),before_exact) : {1n+before == Nat.add(Nat.mul(row,U32.to_nat(per)),1n+lane) : Nat}})
  Arithmetic.to_nat_injective(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) - 1 : U32)),(piece - 1 : U32),
    Equal.trans(Nat,U32.to_nat(Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) - 1 : U32))),Nat.add(Nat.mul(row,U32.to_nat(per)),lane),U32.to_nat((piece - 1 : U32)),
      last_located_with(Nat.is_lt(1n+lane,U32.to_nat(per)),limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,row,lane,{==},DivisionOrder.remainder_small(before,U32.to_nat(per),per_positive),moved),
      Equal.trans(Nat,Nat.add(Nat.mul(row,U32.to_nat(per)),lane),before,U32.to_nat((piece - 1 : U32)),Equal.sym(Nat,before,Nat.add(Nat.mul(row,U32.to_nat(per)),lane),before_exact),
        Equal.sym(Nat,U32.to_nat((piece - 1 : U32)),before,before_word))))

# ---- Unit cells of a grid ----

def unit_cells_of(+found: Grid.Units) -> U32:
  Grid.Units{prefix,outer,extent,unit_cells} = found
  unit_cells

def grid_unit_cells(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces} = grid
  unit_cells

def unit_cells_kept(+found: Grid.Units,+target: U32) -> {grid_unit_cells(Grid.grid_of_units(found,target)) == unit_cells_of(found) : U32}:
  Grid.Units{+prefix,+outer,+extent,+unit_cells} = found
  {==}

# The extent of the last axis.
def last_extent(extents: List<&2,U32>) -> Nat:
  match extents:
    case Nil{}: 1n
    case Con{+axis,Nil{}}: U32.to_nat(axis)
    case Con{+axis,Con{+next,+tail}}: last_extent(Con{next,tail})

# A value that is a whole number of units.
type Multiple<-value: Nat,-unit: Nat> is Data:
  Multiple{times: Nat,exact: {value == Nat.mul(times,unit) : Nat}}

def multiple_times(+value: Nat,+unit: Nat,+multiple: Multiple<value,unit>) -> Nat:
  Multiple{+times,+exact} = multiple
  times

def multiple_exact(+value: Nat,+unit: Nat,+multiple: Multiple<value,unit>) -> {value == Nat.mul(multiple_times(value,unit,multiple),unit) : Nat}:
  Multiple{+times,+exact} = multiple
  exact

# A multiple of a multiple of a unit is a multiple of the unit.
def multiple_through(+value: Nat,+middle: Nat,+unit: Nat,+first: Multiple<value,middle>,+second: Multiple<middle,unit>) -> Multiple<value,unit>:
  +a = multiple_times(value,middle,first)
  +b = multiple_times(middle,unit,second)
  Multiple{Nat.mul(a,b),
    Equal.trans(Nat,value,Nat.mul(a,middle),Nat.mul(Nat.mul(a,b),unit),multiple_exact(value,middle,first),
      Equal.trans(Nat,Nat.mul(a,middle),Nat.mul(a,Nat.mul(b,unit)),Nat.mul(Nat.mul(a,b),unit),
        Equal.cong(Nat,Nat,factor => Nat.mul(a,factor),middle,Nat.mul(b,unit),multiple_exact(middle,unit,second)),
        Equal.sym(Nat,Nat.mul(Nat.mul(a,b),unit),Nat.mul(a,Nat.mul(b,unit)),RowMajor.assoc(a,b,unit))))}

def multiple_moved(+from: Nat,+to: Nat,+unit: Nat,+equal: {from == to : Nat},+multiple: Multiple<to,unit>) -> Multiple<from,unit>:
  Multiple{+times,+exact} = multiple
  Multiple{times,Equal.trans(Nat,from,to,Nat.mul(times,unit),equal,exact)}

# The larger of two multiples of a unit is a multiple of it.
def picked_multiple(less: Bool,+left: Nat,+right: Nat,+unit: Nat,+left_units: Multiple<left,unit>,+right_units: Multiple<right,unit>) ->
  Multiple<Bool.pick(Nat,less,right,left),unit>:
  match less:
    case True{}: right_units
    case False{}: left_units

def max_multiple(+left: Nat,+right: Nat,+unit: Nat,+left_units: Multiple<left,unit>,+right_units: Multiple<right,unit>) -> Multiple<Nat.max(left,right),unit>:
  %Equal.sym(Nat,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),NatExtrema.maximum_choice(left,right)) : Multiple<_,unit>
  picked_multiple(Nat.is_lt(left,right),left,right,unit,left_units,right_units)

# The whole cells a decision keeps together: one cell, or whole units of a
# unit inside the capacity.
type WholeCells<-limits: Machine.Limits,-whole: U32,-unit: Nat> is Data:
  SingleCell{single: {U32.to_nat(whole) == 1n : Nat}}
  WholeUnits{units: Multiple<U32.to_nat(whole),unit>,room: {Nat.is_le(U32.to_nat(whole),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The cut unit of a grid whose unit cells are whole units of a unit, keeping
# such whole cells together, is the larger of the two.
def cut_larger(+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32,+whole: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool}) -> {U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)) == Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)) : Nat}:
  +one_le = Order.transitive(1n,U32.to_nat(unit_cells),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),Order.positive_one(U32.to_nat(unit_cells),positive),NatExtrema.maximum_left(U32.to_nat(unit_cells),U32.to_nat(whole)))
  Equal.trans(Nat,U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),Nat.max(U32.to_nat(U32.max(unit_cells,whole)),1n),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),Extrema.maximum(U32.max(unit_cells,whole),1),
    Equal.trans(Nat,Nat.max(U32.to_nat(U32.max(unit_cells,whole)),1n),Nat.max(Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),1n),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),
      Equal.cong(Nat,Nat,larger => Nat.max(larger,1n),U32.to_nat(U32.max(unit_cells,whole)),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),Extrema.maximum(unit_cells,whole)),
      NatExtrema.maximum_when_above(Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),1n,one_le)))

# The cut unit of a grid whose unit cells are whole units of a unit, keeping
# whole cells of that unit together, is whole units of it inside the
# capacity.
def cut_whole(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32,+whole: U32,+unit: Nat,
  +units: Multiple<U32.to_nat(unit_cells),unit>,+positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(unit_cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells: WholeCells<limits,whole,unit>) ->
  Held.Both<Multiple<U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),unit>,{Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>:
  match cells:
    case SingleCell{+single}:
      +larger = cut_larger(count,prefix,outer,extent,unit_cells,block,per,pieces,whole,positive)
      +below = Held.nat_le_moved(U32.to_nat(whole),1n,U32.to_nat(unit_cells),single,Order.positive_one(U32.to_nat(unit_cells),positive))
      +same = Equal.trans(Nat,U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),U32.to_nat(unit_cells),larger,NatExtrema.maximum_when_above(U32.to_nat(unit_cells),U32.to_nat(whole),below))
      Held.Both{multiple_moved(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),U32.to_nat(unit_cells),unit,same,units),Held.nat_le_moved(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),U32.to_nat(unit_cells),U32.to_nat(Machine.capacity(limits)),same,room)}
    case WholeUnits{+whole_units,+whole_room}:
      +larger = cut_larger(count,prefix,outer,extent,unit_cells,block,per,pieces,whole,positive)
      Held.Both{multiple_moved(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),unit,larger,max_multiple(U32.to_nat(unit_cells),U32.to_nat(whole),unit,units,whole_units)),
        Held.nat_le_moved(U32.to_nat(Runs.cut_unit_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},whole)),Nat.max(U32.to_nat(unit_cells),U32.to_nat(whole)),U32.to_nat(Machine.capacity(limits)),larger,NatExtrema.maximum_upper(U32.to_nat(unit_cells),U32.to_nat(whole),U32.to_nat(Machine.capacity(limits)),room,whole_room))}

# The cells of a shape are whole numbers of its last axis.
def elements_last(extents: List<&2,U32>) -> Multiple<ShapeLaws.elements(extents),last_extent(extents)>:
  match extents:
    case Nil{}: Multiple{1n,{==}}
    case Con{+axis,Nil{}}:
      Multiple{1n,Equal.trans(Nat,Nat.mul(U32.to_nat(axis),1n),U32.to_nat(axis),Nat.add(U32.to_nat(axis),0n),Order.times_one(U32.to_nat(axis)),
        Equal.sym(Nat,Nat.add(U32.to_nat(axis),0n),U32.to_nat(axis),Addition.zero_right(U32.to_nat(axis))))}
    case Con{+axis,Con{+next,+tail}}:
      +others = prepended(next,tail)
      +rest = elements_last(Con{next,tail})
      +times = multiple_times(ShapeLaws.elements(others),last_extent(others),rest)
      Multiple{Nat.mul(U32.to_nat(axis),times),Equal.trans(Nat,Nat.mul(U32.to_nat(axis),ShapeLaws.elements(others)),Nat.mul(U32.to_nat(axis),Nat.mul(times,last_extent(others))),
        Nat.mul(Nat.mul(U32.to_nat(axis),times),last_extent(others)),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(axis),value),ShapeLaws.elements(others),Nat.mul(times,last_extent(others)),multiple_exact(ShapeLaws.elements(others),last_extent(others),rest)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(axis),times),last_extent(others)),Nat.mul(U32.to_nat(axis),Nat.mul(times,last_extent(others))),RowMajor.assoc(U32.to_nat(axis),times,last_extent(others))))}

# The unit cells of a shape keep its last axis whole.
def units_last(+limits: Machine.Limits,+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Multiple<U32.to_nat(unit_cells_of(Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target))),last_extent(extents)>:
  match extents fits:
    case extents True{}:
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},
          Laws.units_fit(extents,prefix,outer,extent,target)) : Multiple<U32.to_nat(unit_cells_of(_)),last_extent(extents)>
      multiple_moved(U32.to_nat(Grid.product(extents)),ShapeLaws.elements(extents),last_extent(extents),
        TiledPublic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extents,room),elements_last(extents))
    case Nil{} False{}: Multiple{1n,{==}}
    case Con{+axis,Nil{}} False{}: Multiple{1n,Equal.sym(Nat,Nat.add(U32.to_nat(axis),0n),U32.to_nat(axis),Addition.zero_right(U32.to_nat(axis)))}
    case Con{+axis,Con{+next,+tail}} False{}:
      +others = prepended(next,tail)
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(others),nonempty)
      +others_room = Order.transitive(ShapeLaws.elements(others),Nat.mul(U32.to_nat(axis),ShapeLaws.elements(others)),U32.to_nat(Machine.capacity(limits)),
        Held.nat_le_right_moved(ShapeLaws.elements(others),Nat.mul(U32.to_nat(axis),ShapeLaws.elements(others)),Nat.mul(ShapeLaws.elements(others),U32.to_nat(axis)),RowMajor.multiply_commutative(U32.to_nat(axis),ShapeLaws.elements(others)),
          RowMajor.product_covers(ShapeLaws.elements(others),U32.to_nat(axis),axis_positive)),room)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,others}),Con{Grid.product(others),Grid.strides(others)},Laws.strides_con(axis,others)) :
        Multiple<U32.to_nat(unit_cells_of(Grid.units_from(Con{axis,others},_,False{},prefix,outer,extent,target))),last_extent(extents)>
      units_last(limits,Con{next,tail},U32.is_le(Grid.product(others),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,others_room,
        right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(others),nonempty))

# The cut unit of the grid of a shape, given the facts of that grid and whole
# units in its unit cells, keeping whole cells together.
def cut_facts(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+unit: Nat,+facts: Laws.GridFacts<limits,Grid.grid(cells,target)>,
  +units: Multiple<U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit>,+whole: U32,+whole_cells: WholeCells<limits,whole,unit>) ->
  Held.Both<Multiple<U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,target),whole)),unit>,{Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,target),whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  +kept = Equal.trans(U32,unit_cells,grid_unit_cells(Grid.grid(cells,target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
    Equal.sym(U32,grid_unit_cells(Grid.grid(cells,target)),unit_cells,Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact)),
    Equal.trans(U32,grid_unit_cells(Grid.grid(cells,target)),grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
      Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),Laws.grid_units(cells,target)),
      unit_cells_kept(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)))
  +unit_whole = multiple_moved(U32.to_nat(unit_cells),U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit,Equal.cong(U32,Nat,word => U32.to_nat(word),unit_cells,unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),kept),units)
  +unit_room = Order.transitive(U32.to_nat(unit_cells),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.le_multiple(U32.to_nat(extent),U32.to_nat(unit_cells),Laws.extent_positive(limits,outer,extent,unit_cells,block,per,shape)),Laws.image_room(limits,outer,extent,unit_cells,block,per,shape))
  %Equal.sym(Grid.Grid,Grid.grid(cells,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : Held.Both<Multiple<U32.to_nat(Runs.cut_unit_of(_,whole)),unit>,{Nat.is_le(U32.to_nat(Runs.cut_unit_of(_,whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>
  cut_whole(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),whole,unit,unit_whole,Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape),unit_room,whole_cells)

# The cut unit of the grid of a shape whose last axis is whole units of a
# unit, keeping whole cells of that unit together, is whole units of it
# inside the capacity.
def shape_cut(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: Multiple<last_extent(cells),unit>,+whole: U32,+whole_cells: WholeCells<limits,whole,unit>) ->
  Held.Both<Multiple<U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,target),whole)),unit>,{Nat.is_le(U32.to_nat(Runs.cut_unit_of(Grid.grid(cells,target),whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>:
  cut_facts(limits,cells,target,unit,grid_facts(limits,cells,target,room,nonempty),
    multiple_through(U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),last_extent(cells),unit,units_last(limits,cells,U32.is_le(Grid.product(cells),target),0,1,1,target,room,nonempty),divides),
    whole,whole_cells)

def grid_run_facts(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+unit: Nat,+facts: Laws.GridFacts<limits,Grid.grid(cells,target)>,
  +units: Multiple<U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit>) -> Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,U32.to_nat(Grid.count_of(Grid.grid(cells,target)))):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  +kept = Equal.trans(U32,unit_cells,grid_unit_cells(Grid.grid(cells,target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
    Equal.sym(U32,grid_unit_cells(Grid.grid(cells,target)),unit_cells,Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact)),
    Equal.trans(U32,grid_unit_cells(Grid.grid(cells,target)),grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
      Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),Laws.grid_units(cells,target)),
      unit_cells_kept(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)))
  +whole = multiple_moved(U32.to_nat(unit_cells),U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit,Equal.cong(U32,Nat,word => U32.to_nat(word),unit_cells,unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),kept),units)
  %Equal.sym(Grid.Grid,Grid.grid(cells,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : Runs.Run(Pieces.ranges_of_grid(_),unit,U32.to_nat(Grid.count_of(_)))
  grid_run(limits,outer,extent,unit_cells,block,per,count,prefix,shape,total,unit,multiple_times(U32.to_nat(unit_cells),unit,whole),multiple_exact(U32.to_nat(unit_cells),unit,whole))

# The ranges of the grid of a shape are a run of any unit that divides its last axis.
def grid_run_of(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: Multiple<last_extent(cells),unit>) ->
  Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,U32.to_nat(Grid.count_of(Grid.grid(cells,target)))):
  +facts = grid_facts(limits,cells,target,room,nonempty)
  +whole = units_last(limits,cells,U32.is_le(Grid.product(cells),target),0,1,1,target,room,nonempty)
  grid_run_facts(limits,cells,target,unit,facts,multiple_through(U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),last_extent(cells),unit,whole,divides))

# The grid of units holds the cells the units hold.
def units_count(+limits: Machine.Limits,+found: Grid.Units,+total: Nat,+target: U32,+facts: UnitsFacts<limits,found,total>,
  +total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(Grid.count_of(Grid.grid_of_units(found,target))) == total : Nat}:
  UnitsFacts{+prefix,+outer,+extent,+unit_cells,+exact,+outer_positive,+extent_positive,+unit_positive,+cells} = facts
  +room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),total,U32.to_nat(Machine.capacity(limits)),cells,total_room)
  %Equal.sym(Grid.Units,found,Grid.Units{prefix,outer,extent,unit_cells},exact) : {U32.to_nat(Grid.count_of(Grid.grid_of_units(_,target))) == total : Nat}
  Equal.trans(Nat,U32.to_nat((outer * extent * unit_cells : U32)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),total,units_word(limits,outer,extent,unit_cells,unit_positive,room),cells)

# The grid of a shape holds its elements.
def grid_count(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool}) -> {U32.to_nat(Grid.count_of(Grid.grid(cells,target))) == ShapeLaws.elements(cells) : Nat}:
  +total_room = Held.nat_le_moved(Nat.add(ShapeLaws.elements(cells),0n),ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits)),Addition.zero_right(ShapeLaws.elements(cells)),room)
  %Equal.sym(Grid.Grid,Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),Laws.grid_units(cells,target)) :
    {U32.to_nat(Grid.count_of(_)) == ShapeLaws.elements(cells) : Nat}
  Equal.trans(Nat,U32.to_nat(Grid.count_of(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target))),Nat.add(ShapeLaws.elements(cells),0n),ShapeLaws.elements(cells),
    units_count(limits,Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),Nat.add(ShapeLaws.elements(cells),0n),target,
      units_facts(limits,cells,U32.is_le(Grid.product(cells),target),0,1,1,target,total_room,{==},{==},nonempty),total_room),
    Addition.zero_right(ShapeLaws.elements(cells)))

# ---- Tasks of whole pieces ----

# A cell where a piece of the grid starts, the end of the cells counted as the
# start of the piece past the last.
type Boundary<-grid: Grid.Grid,-cell: Nat> is Data:
  Boundary{piece: U32,inside: {Nat.is_le(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool},exact: {U32.to_nat(Grid.start(grid,piece)) == cell : Nat}}

# Every leaf of a layout starts and ends where pieces start.
def Bounded(+grid: Grid.Grid,layout: Pieces.Ranges) -> Data:
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both<Boundary<grid,U32.to_nat(first)>,Boundary<grid,Nat.add(U32.to_nat(first),U32.to_nat(count))>>
    case Pieces.Split{left,right}:
      Held.Both<Bounded(grid,left),Bounded(grid,right)>

def low_bounded(+grid: Grid.Grid,layout: Pieces.Ranges,+bounded: Bounded(grid,layout)) -> Boundary<grid,Runs.range_low(layout)>:
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+low,+high} = bounded
      low
    case Pieces.Split{+left,+right}:
      Held.Both{+left_bounded,+right_bounded} = bounded
      low_bounded(grid,left,left_bounded)

def high_bounded(+grid: Grid.Grid,layout: Pieces.Ranges,+bounded: Bounded(grid,layout)) -> Boundary<grid,Runs.range_high(layout)>:
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+low,+high} = bounded
      high
    case Pieces.Split{+left,+right}:
      Held.Both{+left_bounded,+right_bounded} = bounded
      high_bounded(grid,right,right_bounded)

def boundary_moved(+grid: Grid.Grid,+from: Nat,+to: Nat,+equal: {from == to : Nat},+boundary: Boundary<grid,from>) -> Boundary<grid,to>:
  %equal : Boundary<grid,_>
  boundary

# The ends of a group run.
def group_first(+first: U32,+count: U32,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(Tasks.Group{first,count,layout},low,high,unit,bound)) ->
  {U32.to_nat(first) == low : Nat}:
  Held.Both{+first_low,+rest} = facts
  first_low

def group_end(+first: U32,+count: U32,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(Tasks.Group{first,count,layout},low,high,unit,bound)) ->
  {Nat.add(U32.to_nat(first),U32.to_nat(count)) == high : Nat}:
  Held.Both{+first_low,+rest} = facts
  Held.Both{+ended,+more} = rest
  ended

def group_low(+first: U32,+count: U32,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(Tasks.Group{first,count,layout},low,high,unit,bound)) ->
  {Runs.range_low(layout) == low : Nat}:
  Held.Both{+first_low,+rest} = facts
  Held.Both{+ended,+more} = rest
  Held.Both{+run,+ends} = more
  Held.Both{+layout_low,+layout_high} = ends
  layout_low

def group_high(+first: U32,+count: U32,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(Tasks.Group{first,count,layout},low,high,unit,bound)) ->
  {Runs.range_high(layout) == high : Nat}:
  Held.Both{+first_low,+rest} = facts
  Held.Both{+ended,+more} = rest
  Held.Both{+run,+ends} = more
  Held.Both{+layout_low,+layout_high} = ends
  layout_high

def bounded_choice(choice: Bool,+grid: Grid.Grid,+first: U32,+count: U32,+left: Pieces.Ranges,+right: Pieces.Ranges,+low: Boundary<grid,U32.to_nat(first)>,
  +high: Boundary<grid,Nat.add(U32.to_nat(first),U32.to_nat(count))>,+left_bounded: Bounded(grid,left),+right_bounded: Bounded(grid,right)) ->
  Bounded(grid,Bool.pick(Pieces.Ranges,choice,Pieces.Range{first,count},Pieces.Split{left,right})):
  match choice:
    case True{}: Held.Both{low,high}
    case False{}: Held.Both{left_bounded,right_bounded}

# Joining two adjacent groups keeps their leaves on piece starts.
def bounded_joined(+limits: Machine.Limits,+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,+left: Tasks.Group,+right: Tasks.Group,+low: Nat,+middle: Nat,
  +high: Nat,+unit: Nat,+bound: Nat,+left_facts: Runs.GroupRun(left,low,middle,unit,bound),+right_facts: Runs.GroupRun(right,middle,high,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+grid: Grid.Grid,+left_bounded: Bounded(grid,Tasks.group_layout(left)),
  +right_bounded: Bounded(grid,Tasks.group_layout(right))) -> Bounded(grid,Tasks.group_layout(Tasks.group_join(limit,left_ranges,right_ranges,left,right))):
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}:
      +count = (left_count + right_count : U32)
      +joined = Runs.joined(limits,limit,left_ranges,right_ranges,Tasks.Group{left_first,left_count,left_layout},Tasks.Group{right_first,right_count,right_layout},
        low,middle,high,unit,bound,left_facts,right_facts,room)
      +ended = group_end(left_first,count,Bool.pick(Pieces.Ranges,U32.is_le(count,limit),Pieces.Range{left_first,count},Pieces.Split{left_layout,right_layout}),low,high,unit,bound,joined)
      +starts = Equal.trans(Nat,Runs.range_low(left_layout),low,U32.to_nat(left_first),group_low(left_first,left_count,left_layout,low,middle,unit,bound,left_facts),
        Equal.sym(Nat,U32.to_nat(left_first),low,group_first(left_first,left_count,left_layout,low,middle,unit,bound,left_facts)))
      +ends = Equal.trans(Nat,Runs.range_high(right_layout),high,Nat.add(U32.to_nat(left_first),U32.to_nat(count)),group_high(right_first,right_count,right_layout,middle,high,unit,bound,right_facts),
        Equal.sym(Nat,Nat.add(U32.to_nat(left_first),U32.to_nat(count)),high,ended))
      bounded_choice(U32.is_le(count,limit),grid,left_first,count,left_layout,right_layout,
        boundary_moved(grid,Runs.range_low(left_layout),U32.to_nat(left_first),starts,low_bounded(grid,left_layout,left_bounded)),
        boundary_moved(grid,Runs.range_high(right_layout),Nat.add(U32.to_nat(left_first),U32.to_nat(count)),ends,high_bounded(grid,right_layout,right_bounded)),
        left_bounded,right_bounded)

# Folding a run into groups keeps its leaves on piece starts.
def bounded_grouped(+limits: Machine.Limits,+limit: U32,ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(ranges,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+grid: Grid.Grid,+bounded: Bounded(grid,ranges)) ->
  Bounded(grid,Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,ranges,limit))):
  match ranges:
    case Pieces.Range{+first,+count}: bounded
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{+left_bounded,+right_bounded} = bounded
      +left_group = Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,left,limit)
      +right_group = Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,right,limit)
      bounded_joined(limits,limit,left,right,left_group,right_group,Runs.range_low(left),Runs.range_high(left),Runs.range_high(right),unit,bound,
        Runs.grouped(limits,limit,left,unit,bound,left_run,room),
        {%Equal.sym(Nat,Runs.range_high(left),Runs.range_low(right),glue) : Runs.GroupRun(right_group,_,Runs.range_high(right),unit,bound)
          Runs.grouped(limits,limit,right,unit,bound,right_run,room) : Runs.GroupRun(right_group,Runs.range_high(left),Runs.range_high(right),unit,bound)},room,grid,
        bounded_grouped(limits,limit,left,unit,bound,left_run,room,grid,left_bounded),bounded_grouped(limits,limit,right,unit,bound,right_run,room,grid,right_bounded))

# The tasks of ranges sized by their count keep their leaves on piece starts.
def sized_bounded(+limits: Machine.Limits,+ranges: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size,+unit: Nat,+bound: Nat,+run: Runs.Run(ranges,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+grid: Grid.Grid,+bounded: Bounded(grid,ranges)) ->
  Bounded(grid,Tasks.tasks_sized(ranges,count,pool,whole,found)):
  match found:
    case Tasks.Size{+listed,+largest}:
      bounded_grouped(limits,U32.max(U32.max(Tasks.divided_up(count,pool),whole),(Tasks.divided_up(listed,pool) * largest : U32)),ranges,unit,bound,run,room,grid,bounded)

# The tasks an operation decides over the ranges of a grid hold whole pieces.
def bounded_planned(+limits: Machine.Limits,+parallel: Bool,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+grid: Grid.Grid,+bounded: Bounded(grid,layout)) ->
  Bounded(grid,Runs.decision_tasks(Tasks.planned(parallel,layout))):
  match parallel:
    case True{}: bounded
    case False{}:
      Held.Both{boundary_moved(grid,Runs.range_low(layout),U32.to_nat(Pieces.first(layout)),Equal.sym(Nat,U32.to_nat(Pieces.first(layout)),Runs.range_low(layout),Runs.first_low(layout)),
          low_bounded(grid,layout,bounded)),
        boundary_moved(grid,Runs.range_high(layout),Nat.add(U32.to_nat(Pieces.first(layout)),U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),
          Equal.sym(Nat,Nat.add(U32.to_nat(Pieces.first(layout)),U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),Runs.range_high(layout),Runs.whole_end(limits,layout,unit,bound,run,room)),
          high_bounded(grid,layout,bounded))}

def bounded_decided(+limits: Machine.Limits,+grid: Grid.Grid,+workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,+unit: Nat,+bound: Nat,
  +run: Runs.Run(Pieces.ranges_of_grid(grid),unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +bounded: Bounded(grid,Pieces.ranges_of_grid(grid))) ->
  Bounded(grid,Runs.decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra))):
  Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  +ranges = Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces})
  +layout = Tasks.tasks_waved(ranges,count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra)
  bounded_planned(limits,Tasks.worth(workers,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,count,Tasks.size(layout)),
    layout,unit,bound,Runs.waved_run(limits,ranges,count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,unit,bound,run,room),room,Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},
    sized_bounded(limits,ranges,count,Tasks.waved_pool(workers,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,count,whole,Tasks.size(ranges)),whole,Tasks.size(ranges),unit,bound,run,room,Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},
      bounded))

# The leaves of the ranges of a grid start and end where pieces start.
def BoundedSpans(+grid: Grid.Grid,spans: Held.Spans) -> Data:
  match spans:
    case Held.Ended{}:
      Unit
    case Held.Spanned{+first,+count,+rest}:
      Held.Both<Held.Both<Boundary<grid,U32.to_nat(first)>,Boundary<grid,Nat.add(U32.to_nat(first),U32.to_nat(count))>>,BoundedSpans(grid,rest)>

def bounded_part(+grid: Grid.Grid,+layout: Pieces.Ranges,+rest: Held.Spans,+facts: Held.Both<Bounded(grid,layout),BoundedSpans(grid,rest)>) -> Bounded(grid,layout):
  Held.Both{+here,+after} = facts
  here

def bounded_rest(+grid: Grid.Grid,+layout: Pieces.Ranges,+rest: Held.Spans,+facts: Held.Both<Bounded(grid,layout),BoundedSpans(grid,rest)>) -> BoundedSpans(grid,rest):
  Held.Both{+here,+after} = facts
  after

def spans_bounded(+grid: Grid.Grid,layout: Pieces.Ranges,+rest: Held.Spans,+facts: BoundedSpans(grid,Held.layout_spans(layout,rest))) ->
  Held.Both<Bounded(grid,layout),BoundedSpans(grid,rest)>:
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+here,+after} = facts
      Held.Both{here,after}
    case Pieces.Split{+left,+right}:
      +left_facts = spans_bounded(grid,left,Held.layout_spans(right,rest),facts)
      +right_facts = spans_bounded(grid,right,rest,bounded_rest(grid,left,Held.layout_spans(right,rest),left_facts))
      Held.Both{Held.Both{bounded_part(grid,left,Held.layout_spans(right,rest),left_facts),bounded_part(grid,right,rest,right_facts)},bounded_rest(grid,right,rest,right_facts)}

def pieces_bounded(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},remaining: Nat,+piece: Nat,
  +full: {Nat.add(piece,remaining) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  BoundedSpans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{})):
  match remaining:
    case 0n: Unit{}
    case 1n+ +more:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +inside = {%full : {Nat.is_lt(piece,_) == True{} : Bool} Order.below_sum(piece,more) : {Nat.is_lt(piece,total) == True{} : Bool}}
      +room = Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
      +words = Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
      +next_inside = Order.strict_follow(piece,total,inside)
      +here_inside = Bounds.strict_le(piece,total,inside)
      +here = Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(piece,total,U32.to_nat(Machine.capacity(limits)),here_inside,room))
      +next = Bounds.roundtrip(1n+piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(1n+piece,total,U32.to_nat(Machine.capacity(limits)),next_inside,room))
      +here_bound = Held.nat_le_right_moved(U32.to_nat(U32.from_nat(piece)),U32.to_nat((outer * per : U32)),total,words,Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,total,here,here_inside))
      +next_bound = Held.nat_le_right_moved(U32.to_nat(U32.from_nat(1n+piece)),U32.to_nat((outer * per : U32)),total,words,Held.nat_le_moved(U32.to_nat(U32.from_nat(1n+piece)),1n+piece,total,next,next_inside))
      +ends = piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)
      Held.Both{Held.Both{Boundary{U32.from_nat(piece),here_bound,{==}},
          Boundary{U32.from_nat(1n+piece),next_bound,Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(grid,U32.from_nat(piece),1))),
            U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece))),ends)}},
        pieces_bounded(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,more,1n+piece,Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),total,
          Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),full))}

# Every leaf of the ranges of a grid starts and ends where pieces start.
def grid_bounded(+limits: Machine.Limits,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>) -> Bounded(grid,Pieces.ranges_of_grid(grid)):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  +literal = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(Grid.Grid,grid,literal,exact) : Bounded(_,Pieces.ranges_of_grid(_))
  bounded_part(literal,Pieces.ranges_of_grid(literal),Held.Ended{},spans_bounded(literal,Pieces.ranges_of_grid(literal),Held.Ended{},
    {%Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(literal),Held.Ended{}),piece_spans(literal,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
        grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)) : BoundedSpans(literal,_)
      pieces_bounded(limits,outer,extent,unit_cells,block,per,count,prefix,shape,total,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,{==}) :
      BoundedSpans(literal,Held.layout_spans(Pieces.ranges_of_grid(literal),Held.Ended{}))}))

# ---- Every leaf of the ranges of a grid is one of its pieces ----

type PieceAt<-grid: Grid.Grid,-first: U32,-count: U32> is Data:
  PieceAt{piece: U32,inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool},
    start: {Grid.start(grid,piece) == first : U32},span: {Grid.span(grid,piece) == count : U32}}

def PiecesAt(+grid: Grid.Grid,layout: Pieces.Ranges) -> Data:
  match layout:
    case Pieces.Range{+first,+count}: PieceAt<grid,first,count>
    case Pieces.Split{left,right}: Held.Both<PiecesAt(grid,left),PiecesAt(grid,right)>

def SpansAt(+grid: Grid.Grid,spans: Held.Spans) -> Data:
  match spans:
    case Held.Ended{}: Unit
    case Held.Spanned{+first,+count,+rest}: Held.Both<PieceAt<grid,first,count>,SpansAt(grid,rest)>

def at_part(+grid: Grid.Grid,+layout: Pieces.Ranges,+rest: Held.Spans,+facts: Held.Both<PiecesAt(grid,layout),SpansAt(grid,rest)>) -> PiecesAt(grid,layout):
  Held.Both{+here,+after} = facts
  here

def at_rest(+grid: Grid.Grid,+layout: Pieces.Ranges,+rest: Held.Spans,+facts: Held.Both<PiecesAt(grid,layout),SpansAt(grid,rest)>) -> SpansAt(grid,rest):
  Held.Both{+here,+after} = facts
  after

def spans_at(+grid: Grid.Grid,layout: Pieces.Ranges,+rest: Held.Spans,+facts: SpansAt(grid,Held.layout_spans(layout,rest))) ->
  Held.Both<PiecesAt(grid,layout),SpansAt(grid,rest)>:
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+here,+after} = facts
      Held.Both{here,after}
    case Pieces.Split{+left,+right}:
      +left_facts = spans_at(grid,left,Held.layout_spans(right,rest),facts)
      +right_facts = spans_at(grid,right,rest,at_rest(grid,left,Held.layout_spans(right,rest),left_facts))
      Held.Both{Held.Both{at_part(grid,left,Held.layout_spans(right,rest),left_facts),at_part(grid,right,rest,right_facts)},at_rest(grid,right,rest,right_facts)}

def pieces_at(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,remaining: Nat,+piece: Nat,
  +full: {Nat.add(piece,remaining) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  SpansAt(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{})):
  match remaining:
    case 0n: Unit{}
    case 1n+ +more:
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +inside = {%full : {Nat.is_lt(piece,_) == True{} : Bool} Order.below_sum(piece,more) : {Nat.is_lt(piece,total) == True{} : Bool}}
      +room = Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
      +here = Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
        Order.transitive(piece,total,U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(piece,total,inside),room))
      +word_inside = {%Equal.sym(Nat,U32.to_nat((outer * per : U32)),total,Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(U32.to_nat(U32.from_nat(piece)),_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(piece)),piece,here) : {Nat.is_lt(_,total) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(U32.from_nat(piece)),U32.to_nat((outer * per : U32))) == True{} : Bool}}
      Held.Both{PieceAt{U32.from_nat(piece),word_inside,{==},span_cells(outer,extent,unit_cells,block,per,count,prefix,U32.from_nat(piece))},
        pieces_at(limits,outer,extent,unit_cells,block,per,count,prefix,shape,more,1n+piece,Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),total,
          Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),full))}

# Every leaf of the ranges of a grid is one of its pieces.
def grid_pieces_at(+limits: Machine.Limits,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>) -> PiecesAt(grid,Pieces.ranges_of_grid(grid)):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  +literal = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(Grid.Grid,grid,literal,exact) : PiecesAt(_,Pieces.ranges_of_grid(_))
  at_part(literal,Pieces.ranges_of_grid(literal),Held.Ended{},spans_at(literal,Pieces.ranges_of_grid(literal),Held.Ended{},
    {%Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(literal),Held.Ended{}),piece_spans(literal,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
        grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)) : SpansAt(literal,_)
      pieces_at(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,{==}) :
      SpansAt(literal,Held.layout_spans(Pieces.ranges_of_grid(literal),Held.Ended{}))}))

# ---- The pieces of the tasks ----

# The ranges of the pieces from piece on, remaining more after it, as a task
# writes them.
def pieces_layout(+grid: Grid.Grid,remaining: Nat,+piece: U32) -> Pieces.Ranges:
  match remaining:
    case 0n:
      Pieces.Range{Grid.start(grid,piece),Grid.span(grid,piece)}
    case 1n+rest:
      Pieces.Split{Pieces.Range{Grid.start(grid,piece),Grid.span(grid,piece)},pieces_layout(grid,rest,(piece + 1 : U32))}

def laid_spans(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  remaining: Nat,+piece: Nat,+rest: Held.Spans,+inside: {Nat.is_le(Nat.add(piece,1n+remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Held.layout_spans(pieces_layout(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,U32.from_nat(piece)),rest) ==
    piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1n+remaining,piece,rest) : Held.Spans}:
  match remaining:
    case 0n:
      {==}
    case 1n+ +more:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +cap = Machine.capacity(limits)
      +room = Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
      +reach = Nat.add(piece,1n+(1n+more))
      +reach_room = Order.transitive(reach,total,U32.to_nat(cap),inside,room)
      +here = Bounds.roundtrip(piece,cap,LimitBounds.capacity_fits(limits),Order.transitive(piece,reach,U32.to_nat(cap),Order.le_add(piece,1n+(1n+more)),reach_room))
      +stepped = Addition.successor_right(piece,1n+more)
      +next_le = {%Equal.sym(Nat,reach,1n+Nat.add(piece,1n+more),stepped) : {Nat.is_le(1n+piece,_) == True{} : Bool}
        Order.le_add(piece,1n+more) : {Nat.is_le(1n+piece,reach) == True{} : Bool}}
      +next = Bounds.roundtrip(1n+piece,cap,LimitBounds.capacity_fits(limits),Order.transitive(1n+piece,reach,U32.to_nat(cap),next_le,reach_room))
      +word = Arithmetic.to_nat_injective((U32.from_nat(piece) + 1 : U32),U32.from_nat(1n+piece),
        Equal.trans(Nat,U32.to_nat((U32.from_nat(piece) + 1 : U32)),1n+U32.to_nat(U32.from_nat(piece)),U32.to_nat(U32.from_nat(1n+piece)),
          Laws.successor_word(limits,U32.from_nat(piece),Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,U32.to_nat(cap),here,
            Order.transitive(piece,reach,U32.to_nat(cap),Order.le_add(piece,1n+(1n+more)),reach_room))),
          Equal.trans(Nat,1n+U32.to_nat(U32.from_nat(piece)),1n+piece,U32.to_nat(U32.from_nat(1n+piece)),
            Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(U32.from_nat(piece)),piece,here),Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next))))
      +later_inside = {%stepped : {Nat.is_le(_,total) == True{} : Bool} inside : {Nat.is_le(1n+Nat.add(piece,1n+more),total) == True{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(piece) + 1 : U32),U32.from_nat(1n+piece),word) :
        {Held.Spanned{Grid.start(grid,U32.from_nat(piece)),Grid.span(grid,U32.from_nat(piece)),Held.layout_spans(pieces_layout(grid,more,_),rest)} ==
          piece_spans(grid,1n+(1n+more),piece,rest) : Held.Spans}
      Equal.cong(Held.Spans,Held.Spans,spans => Held.Spanned{Grid.start(grid,U32.from_nat(piece)),Grid.span(grid,U32.from_nat(piece)),spans},
        Held.layout_spans(pieces_layout(grid,more,U32.from_nat(1n+piece)),rest),piece_spans(grid,1n+more,1n+piece,rest),
        laid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape,more,1n+piece,rest,later_inside))

# Two pieces that start at the same cell are the same piece.
def same_piece(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first: U32,+second: U32,
  +first_inside: {Nat.is_le(U32.to_nat(first),U32.to_nat((outer * per : U32))) == True{} : Bool},+second_inside: {Nat.is_le(U32.to_nat(second),U32.to_nat((outer * per : U32))) == True{} : Bool},
  +same: {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},first)) ==
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},second)) : Nat}) -> {first == second : U32}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +words = Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape))
  +first_room = Held.nat_le_right_moved(U32.to_nat(first),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((outer * per : U32)),words,first_inside)
  +second_room = Held.nat_le_right_moved(U32.to_nat(second),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((outer * per : U32)),words,second_inside)
  Equal.trans(U32,first,Grid.piece_of(grid,Grid.start(grid,first)),second,
    Equal.sym(U32,Grid.piece_of(grid,Grid.start(grid,first)),first,piece_at_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first,first_room)),
    Equal.trans(U32,Grid.piece_of(grid,Grid.start(grid,first)),Grid.piece_of(grid,Grid.start(grid,second)),second,
      Equal.cong(U32,U32,cell => Grid.piece_of(grid,cell),Grid.start(grid,first),Grid.start(grid,second),Arithmetic.to_nat_injective(Grid.start(grid,first),Grid.start(grid,second),same)),
      piece_at_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,second,second_room)))

# The piece at a boundary and what it says.
def boundary_piece(+grid: Grid.Grid,+cell: Nat,+boundary: Boundary<grid,cell>) -> U32:
  Boundary{+piece,+inside,+exact} = boundary
  piece

def boundary_inside(+grid: Grid.Grid,+cell: Nat,+boundary: Boundary<grid,cell>) ->
  {Nat.is_le(U32.to_nat(boundary_piece(grid,cell,boundary)),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}:
  Boundary{+piece,+inside,+exact} = boundary
  inside

def boundary_exact(+grid: Grid.Grid,+cell: Nat,+boundary: Boundary<grid,cell>) -> {U32.to_nat(Grid.start(grid,boundary_piece(grid,cell,boundary))) == cell : Nat}:
  Boundary{+piece,+inside,+exact} = boundary
  exact

# A boundary before another is at an earlier piece.
def boundary_less(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +low_cell: Nat,+high_cell: Nat,+low: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},low_cell>,
  +high: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},high_cell>,+less: {Nat.is_lt(low_cell,high_cell) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},low_cell,low)),
    U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},high_cell,high))) == True{} : Bool}:
  Boundary{+p,+p_inside,+p_exact} = low
  Boundary{+q,+q_inside,+q_exact} = high
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +words = Equal.sym(Nat,U32.to_nat((outer * per : U32)),total,Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape))
  +p_room = Held.nat_le_right_moved(U32.to_nat(p),total,U32.to_nat((outer * per : U32)),words,p_inside)
  Laws.true_unless(Nat.is_lt(U32.to_nat(p),U32.to_nat(q)),absurd => Held.lt_le_absurd(low_cell,high_cell,less,
    {%p_exact : {Nat.is_le(high_cell,_) == True{} : Bool}
      {%q_exact : {Nat.is_le(_,U32.to_nat(Grid.start(grid,p))) == True{} : Bool}
        Laws.start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,q,p,p_room,Held.lt_false_le(U32.to_nat(p),U32.to_nat(q),absurd)) :
        {Nat.is_le(high_cell,U32.to_nat(Grid.start(grid,p))) == True{} : Bool}} :
      {Nat.is_le(high_cell,low_cell) == True{} : Bool}}))

# A piece at a boundary is a piece of the grid or the end of its pieces.
def boundary_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cell: Nat,
  +boundary: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},cell>) ->
  {Nat.is_le(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},cell,boundary)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  Held.nat_le_right_moved(U32.to_nat(boundary_piece(grid,cell,boundary)),total,U32.to_nat((outer * per : U32)),
    Equal.sym(Nat,U32.to_nat((outer * per : U32)),total,Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)),boundary_inside(grid,cell,boundary))

# The pieces of a task of whole pieces: its first cell is in the piece at its
# low boundary, and it has one piece more than its later pieces up to the
# piece at its high boundary.
type TaskBounds<-grid: Grid.Grid,-first: U32,-count: U32,-low_piece: Nat,-high_piece: Nat> is Data:
  TaskBounds{first_piece: {Grid.piece_of(grid,first) == U32.from_nat(low_piece) : U32},
    pieces: {1n+Grid.later_pieces(grid,first,count) == Nat.sub(high_piece,low_piece) : Nat},
    ordered: {Nat.is_le(low_piece,high_piece) == True{} : Bool}}

def bounds_first_piece(+grid: Grid.Grid,+first: U32,+count: U32,+low_piece: Nat,+high_piece: Nat,+bounds: TaskBounds<grid,first,count,low_piece,high_piece>) ->
  {Grid.piece_of(grid,first) == U32.from_nat(low_piece) : U32}:
  TaskBounds{+first_piece,+pieces,+ordered} = bounds
  first_piece

def bounds_pieces(+grid: Grid.Grid,+first: U32,+count: U32,+low_piece: Nat,+high_piece: Nat,+bounds: TaskBounds<grid,first,count,low_piece,high_piece>) ->
  {1n+Grid.later_pieces(grid,first,count) == Nat.sub(high_piece,low_piece) : Nat}:
  TaskBounds{+first_piece,+pieces,+ordered} = bounds
  pieces

def bounds_ordered(+grid: Grid.Grid,+first: U32,+count: U32,+low_piece: Nat,+high_piece: Nat,+bounds: TaskBounds<grid,first,count,low_piece,high_piece>) ->
  {Nat.is_le(low_piece,high_piece) == True{} : Bool}:
  TaskBounds{+first_piece,+pieces,+ordered} = bounds
  ordered

def task_bounds(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+task_first: U32,+task_count: U32,+unit: Nat,+bound: Nat,+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: Runs.Fits<unit,U32.to_nat(task_first),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),bound>,
  +low: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first)>,
  +high: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))>) ->
  TaskBounds<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first,task_count,U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low)),U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),high))>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +low_cell = U32.to_nat(task_first)
  +high_cell = Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))
  +p = boundary_piece(grid,low_cell,low)
  +q = boundary_piece(grid,high_cell,high)
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +p_room = boundary_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,low_cell,low)
  +q_room = boundary_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,high_cell,high)
  +less = Runs.fits_less(unit,low_cell,high_cell,bound,fits)
  +end_room = Order.transitive(high_cell,bound,U32.to_nat(cap),Runs.fits_inside(unit,low_cell,high_cell,bound,fits),room)
  +sum = Arithmetic.add(task_first,task_count,cap,cap_ok,end_room)
  +start_p = Arithmetic.to_nat_injective(Grid.start(grid,p),task_first,boundary_exact(grid,low_cell,low))
  +start_q = Arithmetic.to_nat_injective(Grid.start(grid,q),(task_first + task_count : U32),
    Equal.trans(Nat,U32.to_nat(Grid.start(grid,q)),high_cell,U32.to_nat((task_first + task_count : U32)),boundary_exact(grid,high_cell,high),
      Equal.sym(Nat,U32.to_nat((task_first + task_count : U32)),high_cell,sum)))
  +at_p = Equal.trans(U32,Grid.piece_of(grid,task_first),Grid.piece_of(grid,Grid.start(grid,p)),p,
    Equal.cong(U32,U32,cell => Grid.piece_of(grid,cell),task_first,Grid.start(grid,p),Equal.sym(U32,Grid.start(grid,p),task_first,start_p)),
    piece_at_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,p,p_room))
  +ordered = boundary_less(limits,outer,extent,unit_cells,block,per,count,prefix,shape,low_cell,high_cell,low,high,less)
  +p_le_q = Bounds.strict_le(U32.to_nat(p),U32.to_nat(q),ordered)
  +d = Nat.sub(U32.to_nat(q),U32.to_nat(p))
  +restored = Order.restore_shift(U32.to_nat(p),U32.to_nat(q),p_le_q)
  +positive = Order.cancel_offset_lt(U32.to_nat(p),0n,d,
    {%Equal.sym(Nat,Nat.add(U32.to_nat(p),0n),U32.to_nat(p),Addition.zero_right(U32.to_nat(p))) : {Nat.is_lt(_,Nat.add(U32.to_nat(p),d)) == True{} : Bool}
      {%Equal.sym(Nat,Nat.add(U32.to_nat(p),d),U32.to_nat(q),restored) : {Nat.is_lt(U32.to_nat(p),_) == True{} : Bool}
        ordered : {Nat.is_lt(U32.to_nat(p),Nat.add(U32.to_nat(p),d)) == True{} : Bool}} :
      {Nat.is_lt(Nat.add(U32.to_nat(p),0n),Nat.add(U32.to_nat(p),d)) == True{} : Bool}})
  +last = Order.positive_has_last(d,positive)
  +rest = Nat.sub(d,1n)
  +q_positive = Order.le_lt_trans(0n,U32.to_nat(p),U32.to_nat(q),Order.zero_le(U32.to_nat(p)),ordered)
  +q_cap = Order.transitive(U32.to_nat(q),total,U32.to_nat(cap),q_room,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +q_split = Equal.trans(Nat,U32.to_nat(q),Nat.add(U32.to_nat(p),d),Nat.add(U32.to_nat(p),1n+rest),Equal.sym(Nat,Nat.add(U32.to_nat(p),d),U32.to_nat(q),restored),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(p),value),d,1n+rest,Equal.sym(Nat,1n+rest,d,last)))
  +q_word = Arithmetic.subtract(q,1,cap,cap_ok,Order.positive_one(U32.to_nat(q),q_positive),
    Order.transitive(Nat.sub(U32.to_nat(q),1n),U32.to_nat(q),U32.to_nat(cap),Order.sub_le(U32.to_nat(q),1n),q_cap))
  +q_less = Equal.trans(Nat,U32.to_nat((q - 1 : U32)),Nat.sub(U32.to_nat(q),1n),Nat.add(U32.to_nat(p),rest),q_word,
    Equal.trans(Nat,Nat.sub(U32.to_nat(q),1n),Nat.sub(Nat.add(U32.to_nat(p),1n+rest),1n),Nat.add(U32.to_nat(p),rest),
      Equal.cong(Nat,Nat,value => Nat.sub(value,1n),U32.to_nat(q),Nat.add(U32.to_nat(p),1n+rest),q_split),Order.subtract_one_complement(U32.to_nat(p),rest)))
  +less_cap = {%Equal.sym(Nat,U32.to_nat((q - 1 : U32)),Nat.sub(U32.to_nat(q),1n),q_word) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.sub(U32.to_nat(q),1n),U32.to_nat(q),U32.to_nat(cap),Order.sub_le(U32.to_nat(q),1n),q_cap) :
    {Nat.is_le(U32.to_nat((q - 1 : U32)),U32.to_nat(cap)) == True{} : Bool}}
  +p_below = {%Equal.sym(Nat,U32.to_nat((q - 1 : U32)),Nat.add(U32.to_nat(p),rest),q_less) : {Nat.is_le(U32.to_nat(p),_) == True{} : Bool}
    Order.le_add(U32.to_nat(p),rest) : {Nat.is_le(U32.to_nat(p),U32.to_nat((q - 1 : U32))) == True{} : Bool}}
  +later_word = Equal.trans(Nat,U32.to_nat(((q - 1 : U32) - p : U32)),Nat.sub(U32.to_nat((q - 1 : U32)),U32.to_nat(p)),rest,
    Arithmetic.subtract((q - 1 : U32),p,cap,cap_ok,p_below,
      Order.transitive(Nat.sub(U32.to_nat((q - 1 : U32)),U32.to_nat(p)),U32.to_nat((q - 1 : U32)),U32.to_nat(cap),Order.sub_le(U32.to_nat((q - 1 : U32)),U32.to_nat(p)),less_cap)),
    Equal.trans(Nat,Nat.sub(U32.to_nat((q - 1 : U32)),U32.to_nat(p)),Nat.sub(Nat.add(U32.to_nat(p),rest),U32.to_nat(p)),rest,
      Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(p)),U32.to_nat((q - 1 : U32)),Nat.add(U32.to_nat(p),rest),q_less),Order.subtract_shift(U32.to_nat(p),rest)))
  +last_piece = piece_last(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,q,q_room,q_positive)
  +later_exact = {%start_q : {U32.to_nat((Grid.piece_of(grid,(_ - 1 : U32)) - Grid.piece_of(grid,task_first) : U32)) == rest : Nat}
    {%Equal.sym(U32,Grid.piece_of(grid,task_first),p,at_p) : {U32.to_nat((Grid.piece_of(grid,(Grid.start(grid,q) - 1 : U32)) - _ : U32)) == rest : Nat}
      {%Equal.sym(U32,Grid.piece_of(grid,(Grid.start(grid,q) - 1 : U32)),(q - 1 : U32),last_piece) : {U32.to_nat((_ - p : U32)) == rest : Nat}
        later_word : {U32.to_nat((Grid.piece_of(grid,(Grid.start(grid,q) - 1 : U32)) - p : U32)) == rest : Nat}} :
      {U32.to_nat((Grid.piece_of(grid,(Grid.start(grid,q) - 1 : U32)) - Grid.piece_of(grid,task_first) : U32)) == rest : Nat}} :
    {Grid.later_pieces(grid,task_first,task_count) == Nat.sub(d,1n) : Nat}}
  TaskBounds{Equal.trans(U32,Grid.piece_of(grid,task_first),p,U32.from_nat(U32.to_nat(p)),at_p,Equal.sym(U32,U32.from_nat(U32.to_nat(p)),p,Arithmetic.native_roundtrip(p))),
    Equal.trans(Nat,1n+Grid.later_pieces(grid,task_first,task_count),1n+Nat.sub(d,1n),d,
      Equal.cong(Nat,Nat,value => 1n+value,Grid.later_pieces(grid,task_first,task_count),Nat.sub(d,1n),later_exact),last),
    p_le_q}

# The pieces of a task of whole pieces end inside the grid.
def task_inside(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+task_first: U32,+task_count: U32,+unit: Nat,+bound: Nat,+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: Runs.Fits<unit,U32.to_nat(task_first),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),bound>,
  +low: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first)>,
  +high: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))>,
  +bounds: TaskBounds<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first,task_count,U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low)),U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),high))>) ->
  {Nat.is_le(Nat.add(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low)),1n+Grid.later_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first,task_count)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +high_cell = Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))
  +p = U32.to_nat(boundary_piece(grid,U32.to_nat(task_first),low))
  +q = U32.to_nat(boundary_piece(grid,high_cell,high))
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +later = Grid.later_pieces(grid,task_first,task_count)
  +d = Nat.sub(q,p)
  {%Equal.sym(Nat,1n+later,d,bounds_pieces(grid,task_first,task_count,p,q,bounds)) : {Nat.is_le(Nat.add(p,_),total) == True{} : Bool}
    {%Equal.sym(Nat,Nat.add(p,d),q,Order.restore_shift(p,q,bounds_ordered(grid,task_first,task_count,p,q,bounds))) : {Nat.is_le(_,total) == True{} : Bool}
      boundary_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,high_cell,high) : {Nat.is_le(Nat.add(p,d),total) == True{} : Bool}} :
    {Nat.is_le(Nat.add(p,1n+later),total) == True{} : Bool}}

# The pieces of a task of whole pieces are the pieces between its boundaries.
def task_tiled(~Element: Data,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+task_first: U32,+task_count: U32,+pieces: PiecesModel.Model<Element>,
  +rest: Held.Spans,+unit: Nat,+bound: Nat,+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: Runs.Fits<unit,U32.to_nat(task_first),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),bound>,
  +low: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first)>,
  +high: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))>,
  +laid: {PiecesModel.layout(~Element,pieces) == pieces_layout(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.later_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first,task_count),Grid.piece_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first)) : Pieces.Ranges}) ->
  {Held.layout_spans(PiecesModel.layout(~Element,pieces),rest) ==
    piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.sub(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),high)),U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low))),U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low)),rest) : Held.Spans}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +p = U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low))
  +q = U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),high))
  +later = Grid.later_pieces(grid,task_first,task_count)
  +bounds = task_bounds(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,task_first,task_count,unit,bound,room,fits,low,high)
  %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,pieces),pieces_layout(grid,later,Grid.piece_of(grid,task_first)),laid) :
    {Held.layout_spans(_,rest) == piece_spans(grid,Nat.sub(q,p),p,rest) : Held.Spans}
  %Equal.sym(U32,Grid.piece_of(grid,task_first),U32.from_nat(p),bounds_first_piece(grid,task_first,task_count,p,q,bounds)) :
    {Held.layout_spans(pieces_layout(grid,later,_),rest) == piece_spans(grid,Nat.sub(q,p),p,rest) : Held.Spans}
  %bounds_pieces(grid,task_first,task_count,p,q,bounds) :
    {Held.layout_spans(pieces_layout(grid,later,U32.from_nat(p)),rest) == piece_spans(grid,_,p,rest) : Held.Spans}
  laid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape,later,p,rest,task_inside(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,task_first,task_count,unit,bound,room,fits,low,high,bounds))

# The output pieces of every task, each laid as a task of whole pieces writes them.
def TasksLaid(~Element: Data,+grid: Grid.Grid,tasks: Pieces.Ranges,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) -> Data:
  match tasks:
    case Pieces.Range{+first,+count}:
      match results:
        case Partition.Leaf{pieces,offset,span,details}:
          {PiecesModel.layout(~Element,pieces) == pieces_layout(grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first)) : Pieces.Ranges}
        case Partition.Fork{left,right}:
          Empty
    case Pieces.Split{left,right}:
      match results:
        case Partition.Leaf{pieces,offset,span,details}:
          Empty
        case Partition.Fork{left_results,right_results}:
          Held.Both<TasksLaid(~Element,grid,left,left_results),TasksLaid(~Element,grid,right,right_results)>

def counts_joined(+low: Nat,+middle: Nat,+high: Nat,+lower: {Nat.is_le(low,middle) == True{} : Bool},+upper: {Nat.is_le(middle,high) == True{} : Bool}) ->
  {Nat.add(Nat.sub(middle,low),Nat.sub(high,middle)) == Nat.sub(high,low) : Nat}:
  +a = Nat.sub(middle,low)
  +b = Nat.sub(high,middle)
  +whole = Equal.trans(Nat,Nat.add(low,Nat.add(a,b)),Nat.add(Nat.add(low,a),b),high,
    Equal.sym(Nat,Nat.add(Nat.add(low,a),b),Nat.add(low,Nat.add(a,b)),Addition.associative(low,a,b)),
    Equal.trans(Nat,Nat.add(Nat.add(low,a),b),Nat.add(middle,b),high,
      Equal.cong(Nat,Nat,value => Nat.add(value,b),Nat.add(low,a),middle,Order.restore_shift(low,middle,lower)),Order.restore_shift(middle,high,upper)))
  Equal.trans(Nat,Nat.add(a,b),Nat.sub(Nat.add(Nat.add(a,b),low),low),Nat.sub(high,low),
    Equal.sym(Nat,Nat.sub(Nat.add(Nat.add(a,b),low),low),Nat.add(a,b),DivisionOrder.subtract_sum(Nat.add(a,b),low)),
    Equal.cong(Nat,Nat,value => Nat.sub(value,low),Nat.add(Nat.add(a,b),low),high,
      Equal.trans(Nat,Nat.add(Nat.add(a,b),low),Nat.add(low,Nat.add(a,b)),high,Order.commutative(Nat.add(a,b),low),whole)))

# The pieces of the tasks in order are the pieces between the boundaries of
# their ranges.
def tiled(~Element: Data,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},tasks: Pieces.Ranges,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,
  +rest: Held.Spans,+unit: Nat,+bound: Nat,+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+run: Runs.Run(tasks,unit,bound),
  +bounded: Bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks),
  +laid: TasksLaid(~Element,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,results)) ->
  {Placed.tasks_spans(~Element,results,rest) ==
    piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
      Nat.sub(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_high(tasks),
          high_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,bounded))),
        U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(tasks),
          low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,bounded)))),
      U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(tasks),
        low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,bounded))),rest) : Held.Spans}:
  match tasks:
    case Pieces.Range{+task_first,+task_count}:
      match results:
        case Partition.Leaf{+pieces,+offset,+span,+details}:
          Held.Both{+low,+high} = bounded
          task_tiled(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,task_first,task_count,pieces,rest,unit,bound,room,run,low,high,laid)
        case Partition.Fork{+left_results,+right_results}:
          Empty.absurd({Placed.tasks_spans(~Element,Partition.Fork{left_results,right_results},rest) ==
            piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
              Nat.sub(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_high(Pieces.Range{task_first,task_count}),
                  high_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Range{task_first,task_count},bounded))),
                U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(Pieces.Range{task_first,task_count}),
                  low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Range{task_first,task_count},bounded)))),
              U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(Pieces.Range{task_first,task_count}),
                low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Range{task_first,task_count},bounded))),rest) : Held.Spans},laid)
    case Pieces.Split{+left,+right}:
      match results:
        case Partition.Leaf{+pieces,+offset,+span,+details}:
          Empty.absurd({Placed.tasks_spans(~Element,Partition.Leaf{pieces,offset,span,details},rest) ==
            piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
              Nat.sub(U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_high(Pieces.Split{left,right}),
                  high_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Split{left,right},bounded))),
                U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(Pieces.Split{left,right}),
                  low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Split{left,right},bounded)))),
              U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Runs.range_low(Pieces.Split{left,right}),
                low_bounded(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Pieces.Split{left,right},bounded))),rest) : Held.Spans},laid)
        case Partition.Fork{+left_results,+right_results}:
          Held.Both{+left_run,+run_rest} = run
          Held.Both{+right_run,+glue} = run_rest
          Held.Both{+left_bounded,+right_bounded} = bounded
          Held.Both{+left_laid,+right_laid} = laid
          +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
          +left_low = low_bounded(grid,left,left_bounded)
          +left_high = high_bounded(grid,left,left_bounded)
          +right_low = low_bounded(grid,right,right_bounded)
          +right_high = high_bounded(grid,right,right_bounded)
          +pl = U32.to_nat(boundary_piece(grid,Runs.range_low(left),left_low))
          +ql = U32.to_nat(boundary_piece(grid,Runs.range_high(left),left_high))
          +pr = U32.to_nat(boundary_piece(grid,Runs.range_low(right),right_low))
          +qr = U32.to_nat(boundary_piece(grid,Runs.range_high(right),right_high))
          +a = Nat.sub(ql,pl)
          +b = Nat.sub(qr,pr)
          +left_order = Bounds.strict_le(pl,ql,boundary_less(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Runs.range_low(left),Runs.range_high(left),left_low,left_high,
            Runs.fits_less(unit,Runs.range_low(left),Runs.range_high(left),bound,Runs.run_fits(left,unit,bound,left_run))))
          +right_order = Bounds.strict_le(pr,qr,boundary_less(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Runs.range_low(right),Runs.range_high(right),right_low,right_high,
            Runs.fits_less(unit,Runs.range_low(right),Runs.range_high(right),bound,Runs.run_fits(right,unit,bound,right_run))))
          +meet_word = same_piece(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,boundary_piece(grid,Runs.range_high(left),left_high),boundary_piece(grid,Runs.range_low(right),right_low),
            boundary_inside(grid,Runs.range_high(left),left_high),boundary_inside(grid,Runs.range_low(right),right_low),
            Equal.trans(Nat,U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_high(left),left_high))),Runs.range_high(left),U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_low(right),right_low))),
              boundary_exact(grid,Runs.range_high(left),left_high),
              Equal.trans(Nat,Runs.range_high(left),Runs.range_low(right),U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_low(right),right_low))),glue,
                Equal.sym(Nat,U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_low(right),right_low))),Runs.range_low(right),boundary_exact(grid,Runs.range_low(right),right_low)))))
          +meet = Equal.cong(U32,Nat,word => U32.to_nat(word),boundary_piece(grid,Runs.range_high(left),left_high),boundary_piece(grid,Runs.range_low(right),right_low),meet_word)
          +shifted = Equal.trans(Nat,Nat.add(pl,a),ql,pr,Order.restore_shift(pl,ql,left_order),meet)
          +joined = counts_joined(pl,ql,qr,left_order,{%Equal.sym(Nat,ql,pr,meet) : {Nat.is_le(_,qr) == True{} : Bool} right_order : {Nat.is_le(ql,qr) == True{} : Bool}})
          +left_tiled = tiled(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,left,left_results,Placed.tasks_spans(~Element,right_results,rest),unit,bound,room,left_run,left_bounded,left_laid)
          +right_tiled = tiled(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,right,right_results,rest,unit,bound,room,right_run,right_bounded,right_laid)
          Equal.trans(Held.Spans,Placed.tasks_spans(~Element,left_results,Placed.tasks_spans(~Element,right_results,rest)),piece_spans(grid,a,pl,Placed.tasks_spans(~Element,right_results,rest)),
            piece_spans(grid,Nat.sub(qr,pl),pl,rest),left_tiled,
            Equal.trans(Held.Spans,piece_spans(grid,a,pl,Placed.tasks_spans(~Element,right_results,rest)),piece_spans(grid,a,pl,piece_spans(grid,b,pr,rest)),piece_spans(grid,Nat.sub(qr,pl),pl,rest),
              Equal.cong(Held.Spans,Held.Spans,spans => piece_spans(grid,a,pl,spans),Placed.tasks_spans(~Element,right_results,rest),piece_spans(grid,b,pr,rest),right_tiled),
              Equal.trans(Held.Spans,piece_spans(grid,a,pl,piece_spans(grid,b,pr,rest)),piece_spans(grid,a,pl,piece_spans(grid,b,Nat.add(pl,a),rest)),piece_spans(grid,Nat.sub(qr,pl),pl,rest),
                Equal.cong(Nat,Held.Spans,start => piece_spans(grid,a,pl,piece_spans(grid,b,start,rest)),pr,Nat.add(pl,a),Equal.sym(Nat,Nat.add(pl,a),pr,shifted)),
                Equal.trans(Held.Spans,piece_spans(grid,a,pl,piece_spans(grid,b,Nat.add(pl,a),rest)),piece_spans(grid,Nat.add(a,b),pl,rest),piece_spans(grid,Nat.sub(qr,pl),pl,rest),
                  spans_append(grid,a,pl,b,rest),
                  Equal.cong(Nat,Held.Spans,number => piece_spans(grid,number,pl,rest),Nat.add(a,b),Nat.sub(qr,pl),
                    {%meet : {Nat.add(a,Nat.sub(qr,_)) == Nat.sub(qr,pl) : Nat} joined : {Nat.add(a,b) == Nat.sub(qr,pl) : Nat}})))))

# ---- The tasks cover the grid ----

# The chain of the ranges of the pieces from piece on ends where the pieces end.
def chain_spans_end(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},remaining: Nat,+piece: Nat,
  +full: {Nat.add(piece,remaining) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  {Placed.chain_end(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))) ==
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))) : Nat}:
  match remaining:
    case 0n:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +reached = Equal.trans(Nat,piece,Nat.add(piece,0n),total,Equal.sym(Nat,Nat.add(piece,0n),piece,Addition.zero_right(piece)),full)
      +room = Held.nat_le_moved(piece,total,U32.to_nat(Machine.capacity(limits)),reached,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
      +at_end = Arithmetic.to_nat_injective(U32.from_nat(piece),(outer * per : U32),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(piece)),piece,U32.to_nat((outer * per : U32)),Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),room),
          Equal.trans(Nat,piece,total,U32.to_nat((outer * per : U32)),reached,Equal.sym(Nat,U32.to_nat((outer * per : U32)),total,Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)))))
      Equal.cong(U32,Nat,word => U32.to_nat(Grid.start(grid,word)),U32.from_nat(piece),(outer * per : U32),at_end)
    case 1n+ +more:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +inside = {%full : {Nat.is_lt(piece,_) == True{} : Bool} Order.below_sum(piece,more) : {Nat.is_lt(piece,total) == True{} : Bool}}
      +ends = piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)
      %Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(grid,U32.from_nat(piece),1))),U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece))),ends) :
        {Placed.chain_end(piece_spans(grid,more,1n+piece,Held.Ended{}),_) == U32.to_nat(Grid.start(grid,(outer * per : U32))) : Nat}
      chain_spans_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,more,1n+piece,
        Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),total,Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),full))

# The ranges of a grid end at its cells.
def grid_high(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) ->
  {Runs.range_high(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})) == U32.to_nat(count) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +low = U32.to_nat(Grid.start(grid,U32.from_nat(0n)))
  Equal.trans(Nat,Runs.range_high(layout),Placed.chain_end(Held.layout_spans(layout,Held.Ended{}),low),U32.to_nat(count),
    Equal.sym(Nat,Placed.chain_end(Held.layout_spans(layout,Held.Ended{}),low),Runs.range_high(layout),Placed.layout_chain_end(layout,Held.Ended{},low)),
    Equal.trans(Nat,Placed.chain_end(Held.layout_spans(layout,Held.Ended{}),low),Placed.chain_end(piece_spans(grid,total,0n,Held.Ended{}),low),U32.to_nat(count),
      Equal.cong(Held.Spans,Nat,spans => Placed.chain_end(spans,low),Held.layout_spans(layout,Held.Ended{}),piece_spans(grid,total,0n,Held.Ended{}),
        grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)),
      Equal.trans(Nat,Placed.chain_end(piece_spans(grid,total,0n,Held.Ended{}),low),U32.to_nat(Grid.start(grid,(outer * per : U32))),U32.to_nat(count),
        chain_spans_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,total,0n,{==}),
        Equal.trans(Nat,U32.to_nat(Grid.start(grid,(outer * per : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),
          end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape),
          Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells)))))

# The tasks of a decision over a run start and end where the run does.
def group_ends(+group: Tasks.Group,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(group,low,high,unit,bound)) ->
  Held.Both<{Runs.range_low(Tasks.group_layout(group)) == low : Nat},{Runs.range_high(Tasks.group_layout(group)) == high : Nat}>:
  match group:
    case Tasks.Group{+first,+count,+layout}:
      Held.Both{group_low(first,count,layout,low,high,unit,bound,facts),group_high(first,count,layout,low,high,unit,bound,facts)}

def planned_ends(+limits: Machine.Limits,+parallel: Bool,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ends: Held.Both<{Runs.range_low(layout) == low : Nat},{Runs.range_high(layout) == high : Nat}>) ->
  Held.Both<{Runs.range_low(Runs.decision_tasks(Tasks.planned(parallel,layout))) == low : Nat},{Runs.range_high(Runs.decision_tasks(Tasks.planned(parallel,layout))) == high : Nat}>:
  match parallel:
    case True{}: ends
    case False{}:
      Held.Both{+low_exact,+high_exact} = ends
      Held.Both{Equal.trans(Nat,U32.to_nat(Pieces.first(layout)),Runs.range_low(layout),low,Runs.first_low(layout),low_exact),
        Equal.trans(Nat,Nat.add(U32.to_nat(Pieces.first(layout)),U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),Runs.range_high(layout),high,
          Runs.whole_end(limits,layout,unit,bound,run,room),high_exact)}

# The tasks of ranges sized by their count start and end where the ranges do.
def sized_ends(+limits: Machine.Limits,+ranges: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size,+unit: Nat,+bound: Nat,+run: Runs.Run(ranges,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<{Runs.range_low(Tasks.tasks_sized(ranges,count,pool,whole,found)) == Runs.range_low(ranges) : Nat},
    {Runs.range_high(Tasks.tasks_sized(ranges,count,pool,whole,found)) == Runs.range_high(ranges) : Nat}>:
  match found:
    case Tasks.Size{+listed,+largest}:
      +limit = U32.max(U32.max(Tasks.divided_up(count,pool),whole),(Tasks.divided_up(listed,pool) * largest : U32))
      group_ends(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,ranges,limit),Runs.range_low(ranges),Runs.range_high(ranges),unit,bound,
        Runs.grouped(limits,limit,ranges,unit,bound,run,room))

def decided_ends(+limits: Machine.Limits,+grid: Grid.Grid,+workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,+unit: Nat,+bound: Nat,
  +run: Runs.Run(Pieces.ranges_of_grid(grid),unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<{Runs.range_low(Runs.decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra))) == Runs.range_low(Pieces.ranges_of_grid(grid)) : Nat},
    {Runs.range_high(Runs.decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra))) == Runs.range_high(Pieces.ranges_of_grid(grid)) : Nat}>:
  Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  +ranges = Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces})
  +layout = Tasks.tasks_waved(ranges,count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra)
  planned_ends(limits,Tasks.worth(workers,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,count,Tasks.size(layout)),
    layout,Runs.range_low(ranges),Runs.range_high(ranges),unit,bound,Runs.waved_run(limits,ranges,count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,unit,bound,run,room),room,
    sized_ends(limits,ranges,count,Tasks.waved_pool(workers,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,count,whole,Tasks.size(ranges)),whole,Tasks.size(ranges),unit,bound,run,room))

# The pieces the tasks of a decision over a grid write, in order, are the
# ranges of the grid.
def tasks_cover(~Element: Data,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat},
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,
  +laid: TasksLaid(~Element,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Runs.decision_tasks(Tasks.decide_whole(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},workers,whole,bytes_per_cell,operations,extra)),results)) ->
  {Placed.tasks_spans(~Element,results,Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) : Held.Spans}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +tasks = Runs.decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra))
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +bound = U32.to_nat(count)
  +cap = Machine.capacity(limits)
  +room_cap = Held.nat_le_moved(bound,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(cap),cells,Laws.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  +low = U32.to_nat(Grid.start(grid,U32.from_nat(0n)))
  +chain = grid_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact)
  +run_facts = Runs.tree_run(layout,Held.Ended{},low,unit,bound,chain)
  +run = Runs.part_run(layout,Held.Ended{},low,unit,bound,run_facts)
  +task_run = Runs.decided_run(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,bound,run,room_cap)
  +bounded = bounded_decided(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,bound,run,room_cap,
    grid_bounded(limits,grid,Laws.GridFacts{count,prefix,outer,extent,unit_cells,block,per,shape,{==},cells}))
  +ends = decided_ends(limits,grid,workers,whole,bytes_per_cell,operations,extra,unit,bound,run,room_cap)
  +task_low = low_bounded(grid,tasks,bounded)
  +task_high = high_bounded(grid,tasks,bounded)
  +words = Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
  +zero_word = Bounds.roundtrip(0n,cap,LimitBounds.capacity_fits(limits),Order.zero_le(U32.to_nat(cap)))
  +first_piece = same_piece(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,boundary_piece(grid,Runs.range_low(tasks),task_low),U32.from_nat(0n),
    boundary_inside(grid,Runs.range_low(tasks),task_low),
    Held.nat_le_moved(U32.to_nat(U32.from_nat(0n)),0n,U32.to_nat((outer * per : U32)),zero_word,Order.zero_le(U32.to_nat((outer * per : U32)))),
    Equal.trans(Nat,U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_low(tasks),task_low))),Runs.range_low(tasks),low,boundary_exact(grid,Runs.range_low(tasks),task_low),
      Equal.trans(Nat,Runs.range_low(tasks),Runs.range_low(layout),low,Held.first_of({Runs.range_low(tasks) == Runs.range_low(layout) : Nat},{Runs.range_high(tasks) == Runs.range_high(layout) : Nat},ends),
        Runs.part_low(layout,Held.Ended{},low,unit,bound,run_facts))))
  +last_piece = same_piece(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,boundary_piece(grid,Runs.range_high(tasks),task_high),(outer * per : U32),
    boundary_inside(grid,Runs.range_high(tasks),task_high),Order.reflexive(U32.to_nat((outer * per : U32))),
    Equal.trans(Nat,U32.to_nat(Grid.start(grid,boundary_piece(grid,Runs.range_high(tasks),task_high))),Runs.range_high(tasks),U32.to_nat(Grid.start(grid,(outer * per : U32))),
      boundary_exact(grid,Runs.range_high(tasks),task_high),
      Equal.trans(Nat,Runs.range_high(tasks),Runs.range_high(layout),U32.to_nat(Grid.start(grid,(outer * per : U32))),
        Held.second_of({Runs.range_low(tasks) == Runs.range_low(layout) : Nat},{Runs.range_high(tasks) == Runs.range_high(layout) : Nat},ends),
        Equal.trans(Nat,Runs.range_high(layout),bound,U32.to_nat(Grid.start(grid,(outer * per : U32))),grid_high(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells),
          Equal.trans(Nat,bound,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Grid.start(grid,(outer * per : U32))),cells,
            Equal.sym(Nat,U32.to_nat(Grid.start(grid,(outer * per : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape)))))))
  +first_nat = Equal.trans(Nat,U32.to_nat(boundary_piece(grid,Runs.range_low(tasks),task_low)),U32.to_nat(U32.from_nat(0n)),0n,
    Equal.cong(U32,Nat,word => U32.to_nat(word),boundary_piece(grid,Runs.range_low(tasks),task_low),U32.from_nat(0n),first_piece),zero_word)
  +last_nat = Equal.trans(Nat,U32.to_nat(boundary_piece(grid,Runs.range_high(tasks),task_high)),U32.to_nat((outer * per : U32)),total,
    Equal.cong(U32,Nat,word => U32.to_nat(word),boundary_piece(grid,Runs.range_high(tasks),task_high),(outer * per : U32),last_piece),words)
  %Equal.sym(Held.Spans,Held.layout_spans(layout,Held.Ended{}),piece_spans(grid,total,0n,Held.Ended{}),grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape)) :
    {Placed.tasks_spans(~Element,results,Held.Ended{}) == _ : Held.Spans}
  %Order.subtract_zero(total) : {Placed.tasks_spans(~Element,results,Held.Ended{}) == piece_spans(grid,_,0n,Held.Ended{}) : Held.Spans}
  %first_nat : {Placed.tasks_spans(~Element,results,Held.Ended{}) == piece_spans(grid,Nat.sub(total,_),_,Held.Ended{}) : Held.Spans}
  %last_nat : {Placed.tasks_spans(~Element,results,Held.Ended{}) == piece_spans(grid,Nat.sub(_,U32.to_nat(boundary_piece(grid,Runs.range_low(tasks),task_low))),U32.to_nat(boundary_piece(grid,Runs.range_low(tasks),task_low)),Held.Ended{}) : Held.Spans}
  tiled(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,tasks,results,Held.Ended{},unit,bound,room_cap,task_run,bounded,laid)

# ---- The chain of the pieces of a task ----

# The first pieces of a chain are a chain.
def chain_prefix(+grid: Grid.Grid,count: Nat,+piece: Nat,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,
  +chain: Runs.Chain(piece_spans(grid,count,piece,rest),low,unit,bound)) -> Runs.Chain(piece_spans(grid,count,piece,Held.Ended{}),low,unit,bound):
  match count:
    case 0n: {==}
    case 1n+ +more:
      Held.Both{+fits,+facts} = chain
      Held.Both{+starts,+next} = facts
      Held.Both{fits,Held.Both{starts,chain_prefix(grid,more,1n+piece,rest,
        Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(grid,U32.from_nat(piece),1))),unit,bound,next)}}

# A chain ends at or after where it starts.
def chain_end_after(spans: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(spans,low,unit,bound)) ->
  {Nat.is_le(low,Placed.chain_end(spans,low)) == True{} : Bool}:
  match spans:
    case Held.Ended{}: Order.reflexive(low)
    case Held.Spanned{+first,+count,+rest}:
      Held.Both{+fits,+facts} = chain
      Held.Both{+starts,+next} = facts
      +high = Nat.add(U32.to_nat(first),U32.to_nat(count))
      %starts : {Nat.is_le(_,Placed.chain_end(rest,high)) == True{} : Bool}
      Order.transitive(U32.to_nat(first),high,Placed.chain_end(rest,high),Bounds.strict_le(U32.to_nat(first),high,Runs.fits_less(unit,U32.to_nat(first),high,bound,fits)),
        chain_end_after(rest,high,unit,bound,next))

def fits_bounded(+unit: Nat,+low: Nat,+high: Nat,+bound: Nat,+wider: Nat,+fits: Runs.Fits<unit,low,high,bound>,+inside: {Nat.is_le(high,wider) == True{} : Bool}) ->
  Runs.Fits<unit,low,high,wider>:
  Runs.Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+old} = fits
  Runs.Fits{low_units,high_units,low_exact,high_exact,less,inside}

# A chain fits before its own end.
def narrow(spans: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+chain: Runs.Chain(spans,low,unit,bound)) -> Runs.Chain(spans,low,unit,Placed.chain_end(spans,low)):
  match spans:
    case Held.Ended{}: {==}
    case Held.Spanned{+first,+count,+rest}:
      Held.Both{+fits,+facts} = chain
      Held.Both{+starts,+next} = facts
      +high = Nat.add(U32.to_nat(first),U32.to_nat(count))
      Held.Both{fits_bounded(unit,U32.to_nat(first),high,bound,Placed.chain_end(rest,high),fits,chain_end_after(rest,high,unit,bound,next)),
        Held.Both{starts,narrow(rest,high,unit,bound,next)}}

# The pieces from piece on end where the piece after them starts.
def spans_end(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},remaining: Nat,+piece: Nat,
  +inside: {Nat.is_le(Nat.add(piece,remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Placed.chain_end(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))) ==
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(Nat.add(piece,remaining)))) : Nat}:
  match remaining:
    case 0n:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      Equal.cong(Nat,Nat,value => U32.to_nat(Grid.start(grid,U32.from_nat(value))),piece,Nat.add(piece,0n),Equal.sym(Nat,Nat.add(piece,0n),piece,Addition.zero_right(piece)))
    case 1n+ +more:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +below = Bounds.nat_trans(piece,Nat.add(piece,1n+more),total,Order.below_sum(piece,more),inside)
      +ends = piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,below)
      %Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),U32.to_nat(Pieces.cells_of(grid,U32.from_nat(piece),1))),U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece))),ends) :
        {Placed.chain_end(piece_spans(grid,more,1n+piece,Held.Ended{}),_) == U32.to_nat(Grid.start(grid,U32.from_nat(Nat.add(piece,1n+more)))) : Nat}
      %Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)) :
        {Placed.chain_end(piece_spans(grid,more,1n+piece,Held.Ended{}),U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece)))) == U32.to_nat(Grid.start(grid,U32.from_nat(_))) : Nat}
      spans_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,more,1n+piece,
        {%Addition.successor_right(piece,more) : {Nat.is_le(_,total) == True{} : Bool} inside : {Nat.is_le(1n+Nat.add(piece,more),total) == True{} : Bool}})

# The pieces from the piece of the first cell of a task to the piece of the
# first cell after it are a chain from one to the other.
def task_chain(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat},
  +low_piece: Nat,+high_piece: Nat,+ordered: {Nat.is_le(low_piece,high_piece) == True{} : Bool},+inside: {Nat.is_le(high_piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.sub(high_piece,low_piece),low_piece,Held.Ended{}),
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(low_piece))),unit,
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(high_piece)))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +low = U32.to_nat(Grid.start(grid,U32.from_nat(low_piece)))
  +a = Nat.sub(high_piece,low_piece)
  +b = Nat.sub(total,high_piece)
  +low_inside = Order.transitive(low_piece,high_piece,total,ordered,inside)
  +restored = Order.restore_shift(low_piece,high_piece,ordered)
  +joined = counts_joined(low_piece,high_piece,total,ordered,inside)
  +whole = piece_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact,Nat.sub(total,low_piece),low_piece,Order.restore_shift(low_piece,total,low_inside))
  +split = spans_append(grid,a,low_piece,b,Held.Ended{})
  +full = {%restored : Runs.Chain(piece_spans(grid,a,low_piece,piece_spans(grid,b,_,Held.Ended{})),low,unit,U32.to_nat(count))
    {%Equal.sym(Held.Spans,piece_spans(grid,a,low_piece,piece_spans(grid,b,Nat.add(low_piece,a),Held.Ended{})),piece_spans(grid,Nat.add(a,b),low_piece,Held.Ended{}),split) :
        Runs.Chain(_,low,unit,U32.to_nat(count))
      {%Equal.sym(Nat,Nat.add(a,b),Nat.sub(total,low_piece),joined) : Runs.Chain(piece_spans(grid,_,low_piece,Held.Ended{}),low,unit,U32.to_nat(count))
        whole : Runs.Chain(piece_spans(grid,Nat.add(a,b),low_piece,Held.Ended{}),low,unit,U32.to_nat(count))} :
      Runs.Chain(piece_spans(grid,a,low_piece,piece_spans(grid,b,Nat.add(low_piece,a),Held.Ended{})),low,unit,U32.to_nat(count))} :
    Runs.Chain(piece_spans(grid,a,low_piece,piece_spans(grid,b,high_piece,Held.Ended{})),low,unit,U32.to_nat(count))}
  +narrowed = narrow(piece_spans(grid,a,low_piece,Held.Ended{}),low,unit,U32.to_nat(count),chain_prefix(grid,a,low_piece,piece_spans(grid,b,high_piece,Held.Ended{}),low,unit,U32.to_nat(count),full))
  {%Equal.trans(Nat,Placed.chain_end(piece_spans(grid,a,low_piece,Held.Ended{}),low),U32.to_nat(Grid.start(grid,U32.from_nat(Nat.add(low_piece,a)))),U32.to_nat(Grid.start(grid,U32.from_nat(high_piece))),
      spans_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,a,low_piece,{%Equal.sym(Nat,Nat.add(low_piece,a),high_piece,restored) : {Nat.is_le(_,total) == True{} : Bool} inside :
        {Nat.is_le(Nat.add(low_piece,a),total) == True{} : Bool}}),
      Equal.cong(Nat,Nat,value => U32.to_nat(Grid.start(grid,U32.from_nat(value))),Nat.add(low_piece,a),high_piece,restored)) :
    Runs.Chain(piece_spans(grid,a,low_piece,Held.Ended{}),low,unit,_)
    narrowed : Runs.Chain(piece_spans(grid,a,low_piece,Held.Ended{}),low,unit,U32.to_nat(Grid.start(grid,U32.from_nat(high_piece))))}

# The pieces of a task of whole pieces are a chain from its first cell to the
# cell after it.
def task_piece_chain(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+task_first: U32,+task_count: U32,+unit: Nat,+bound: Nat,+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: Runs.Fits<unit,U32.to_nat(task_first),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count)),bound>,
  +low: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first)>,
  +high: Boundary<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))>,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat}) ->
  Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},1n+Grid.later_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_first,task_count),U32.to_nat(boundary_piece(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.to_nat(task_first),low)),Held.Ended{}),U32.to_nat(task_first),unit,Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +low_cell = U32.to_nat(task_first)
  +high_cell = Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))
  +p_word = boundary_piece(grid,low_cell,low)
  +q_word = boundary_piece(grid,high_cell,high)
  +p = U32.to_nat(p_word)
  +q = U32.to_nat(q_word)
  +d = Nat.sub(q,p)
  +bounds = task_bounds(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,task_first,task_count,unit,bound,room,fits,low,high)
  +chain = task_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact,p,q,bounds_ordered(grid,task_first,task_count,p,q,bounds),boundary_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,high_cell,high))
  +low_start = Equal.trans(Nat,U32.to_nat(Grid.start(grid,U32.from_nat(p))),U32.to_nat(Grid.start(grid,p_word)),low_cell,
    Equal.cong(U32,Nat,word => U32.to_nat(Grid.start(grid,word)),U32.from_nat(p),p_word,Arithmetic.native_roundtrip(p_word)),boundary_exact(grid,low_cell,low))
  +high_start = Equal.trans(Nat,U32.to_nat(Grid.start(grid,U32.from_nat(q))),U32.to_nat(Grid.start(grid,q_word)),high_cell,
    Equal.cong(U32,Nat,word => U32.to_nat(Grid.start(grid,word)),U32.from_nat(q),q_word,Arithmetic.native_roundtrip(q_word)),boundary_exact(grid,high_cell,high))
  {%Equal.sym(Nat,1n+Grid.later_pieces(grid,task_first,task_count),d,bounds_pieces(grid,task_first,task_count,p,q,bounds)) :
      Runs.Chain(piece_spans(grid,_,p,Held.Ended{}),low_cell,unit,high_cell)
    {%low_start : Runs.Chain(piece_spans(grid,d,p,Held.Ended{}),_,unit,high_cell)
      {%high_start : Runs.Chain(piece_spans(grid,d,p,Held.Ended{}),U32.to_nat(Grid.start(grid,U32.from_nat(p))),unit,_)
        chain : Runs.Chain(piece_spans(grid,d,p,Held.Ended{}),U32.to_nat(Grid.start(grid,U32.from_nat(p))),unit,high_cell)} :
      Runs.Chain(piece_spans(grid,d,p,Held.Ended{}),low_cell,unit,high_cell)} :
    Runs.Chain(piece_spans(grid,1n+Grid.later_pieces(grid,task_first,task_count),p,Held.Ended{}),low_cell,unit,high_cell)}

# ---- The results of tasks of whole pieces ----

# What a task of whole pieces gives: its pieces laid as the pieces of its
# cells, every piece holding the property.
def LaidHolds(~Cell: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Cell> -> Data,+grid: Grid.Grid,+first: U32,+count: U32,
  +result: Partition.Tree<&2,PiecesModel.Model<Cell>,Unit>) -> Data:
  match result:
    case Partition.Leaf{pieces,offset,span,details}:
      Held.Both<{PiecesModel.layout(~Cell,pieces) == pieces_layout(grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first)) : Pieces.Ranges},
        PiecesModel.ContentAll(~Cell,~Property,PiecesModel.content(~Cell,pieces))>
    case Partition.Fork{left,right}:
      Empty

# The result of a task held for its window is laid for its range.
def leaf_laid(~Cell: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Cell> -> Data,+grid: Grid.Grid,+first: U32,+count: U32,
  +task_first: U32,+task_count: U32,+result: Partition.Tree<&2,PiecesModel.Model<Cell>,Unit>,+same_first: {task_first == first : U32},+same_count: {task_count == count : U32},
  +holds: LaidHolds(~Cell,~Property,grid,task_first,task_count,result)) ->
  Held.Both<TasksLaid(~Cell,grid,Pieces.Range{first,count},result),Placed.TasksHold(~Cell,~Property,result)>:
  match result:
    case Partition.Leaf{+pieces,+offset,+span,+details}:
      Held.Both{+laid,+values} = holds
      Held.Both{
        {%same_first : {PiecesModel.layout(~Cell,pieces) == pieces_layout(grid,Grid.later_pieces(grid,_,count),Grid.piece_of(grid,_)) : Pieces.Ranges}
          {%same_count : {PiecesModel.layout(~Cell,pieces) == pieces_layout(grid,Grid.later_pieces(grid,task_first,_),Grid.piece_of(grid,task_first)) : Pieces.Ranges}
            laid : {PiecesModel.layout(~Cell,pieces) == pieces_layout(grid,Grid.later_pieces(grid,task_first,count),Grid.piece_of(grid,task_first)) : Pieces.Ranges}} :
          {PiecesModel.layout(~Cell,pieces) == pieces_layout(grid,Grid.later_pieces(grid,first,count),Grid.piece_of(grid,first)) : Pieces.Ranges}},
        values}
    case Partition.Fork{+left,+right}:
      Empty.absurd(Held.Both<TasksLaid(~Cell,grid,Pieces.Range{first,count},Partition.Fork{left,right}),Placed.TasksHold(~Cell,~Property,Partition.Fork{left,right})>,holds)

# The results of tasks of whole pieces, each held for its window, are laid as
# their tasks and every piece holds the property.
def laid_results(~Element: Data,~Cell: Data,~Output: Type,~Resource: Type,~Context: Data,~RModel: Data,~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Move.Holding<Element> -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,~pack_o: PiecesModel.Model<Cell> -> Output,~pack_r: RModel -> Resource,
  ~split_r: RModel -> PartitionLaws.Split<RModel>,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Cell> -> Data,
  ~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+source: Source,+task: Task,+total: Nat,
  +grid: Grid.Grid,+context: Context,tasks: Pieces.Ranges,+windows: Windows.WindowModel<Element>,+resource: RModel,
  +facts: Grouped.TaskFacts(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,tasks,windows,resource),
  +ok: Windows.WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,tasks,windows)) ->
  Held.Both<TasksLaid(~Cell,grid,tasks,Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,tasks,windows,resource,facts)),
    Placed.TasksHold(~Cell,~Property,Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,tasks,windows,resource,facts))>:
  match tasks:
    case Pieces.Range{+first,+count}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Grouped.TaskFact{+result,+rest,+exact,+holds} = facts
          Held.Both{+same_first,+window} = ok
          Held.Both{+same_count,+within} = window
          leaf_laid(~Cell,~Property,grid,first,count,task_first,task_count,result,same_first,same_count,holds)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Empty.absurd(Held.Both<TasksLaid(~Cell,grid,Pieces.Range{first,count},Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,Pieces.Range{first,count},Windows.WindowFork{left_windows,right_windows},resource,facts)),
            Placed.TasksHold(~Cell,~Property,Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,Pieces.Range{first,count},Windows.WindowFork{left_windows,right_windows},resource,facts))>,facts)
    case Pieces.Split{+left,+right}:
      match windows:
        case Windows.WindowLeaf{+holding,+task_first,+task_count}:
          Empty.absurd(Held.Both<TasksLaid(~Cell,grid,Pieces.Split{left,right},Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,Pieces.Split{left,right},Windows.WindowLeaf{holding,task_first,task_count},resource,facts)),
            Placed.TasksHold(~Cell,~Property,Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,Pieces.Split{left,right},Windows.WindowLeaf{holding,task_first,task_count},resource,facts))>,facts)
        case Windows.WindowFork{+left_windows,+right_windows}:
          Held.Both{+left_facts,+right_facts} = facts
          Held.Both{+left_ok,+right_ok} = ok
          +left_resource = PartitionLaws.first(~RModel,split_r(resource))
          +right_resource = PartitionLaws.second(~RModel,split_r(resource))
          +left_results = Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,left,left_windows,left_resource,left_facts)
          +right_results = Grouped.results(~Element,~Output,~Resource,~Context,~PiecesModel.Model<Cell>,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Grid.Grid,~LaidHolds(~Cell,~Property),grid,context,right,right_windows,right_resource,right_facts)
          +left_laid = laid_results(~Element,~Cell,~Output,~Resource,~Context,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Property,~Source,~value,~Task,~reach,source,task,total,grid,context,left,left_windows,left_resource,left_facts,left_ok)
          +right_laid = laid_results(~Element,~Cell,~Output,~Resource,~Context,~RModel,~share,~leaf,~pack_o,~pack_r,~split_r,~Property,~Source,~value,~Task,~reach,source,task,total,grid,context,right,right_windows,right_resource,right_facts,right_ok)
          Held.Both{
            Held.Both{Held.first_of(TasksLaid(~Cell,grid,left,left_results),Placed.TasksHold(~Cell,~Property,left_results),left_laid),
              Held.first_of(TasksLaid(~Cell,grid,right,right_results),Placed.TasksHold(~Cell,~Property,right_results),right_laid)},
            Held.Both{Held.second_of(TasksLaid(~Cell,grid,left,left_results),Placed.TasksHold(~Cell,~Property,left_results),left_laid),
              Held.second_of(TasksLaid(~Cell,grid,right,right_results),Placed.TasksHold(~Cell,~Property,right_results),right_laid)}}

# ---- The layout of the grid of a shape ----

# The ranges of a grid start at cell 0.
def grid_low(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+unit: Nat,+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,unit) : Nat}) ->
  {Runs.range_low(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})) == 0n : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +low = U32.to_nat(Grid.start(grid,U32.from_nat(0n)))
  +zero_pair = DivisionOrder.observed_pair(0n,U32.to_nat(per),DivisionOrder.witness(0n,U32.to_nat(per),Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)),Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +chain = grid_chain(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,unit,units,unit_exact)
  Equal.trans(Nat,Runs.range_low(layout),low,0n,Runs.part_low(layout,Held.Ended{},low,unit,U32.to_nat(count),Runs.tree_run(layout,Held.Ended{},low,unit,U32.to_nat(count),chain)),
    Equal.trans(Nat,low,Laws.place(Nat.div(0n,U32.to_nat(per)),Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),0n,
      Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(0n),Order.zero_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))),
      {%Equal.sym(Nat & Nat,Nat.divmod(0n,U32.to_nat(per)),(0n,0n),zero_pair) : {Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Pair.snd(Nat,Nat,_),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == 0n : Nat}
        {==} : {Laws.place(Nat.div(0n,U32.to_nat(per)),Nat.mod(0n,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == 0n : Nat}}))

# The ranges of the grid of a shape: a run of a unit that divides its last
# axis, from cell 0 to the cell past its last element.
type GridLayout<-cells: List<&2,U32>,-target: U32,-unit: Nat> is Data:
  GridLayout{run: Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,ShapeLaws.elements(cells)),
    low: {Runs.range_low(Pieces.ranges_of_grid(Grid.grid(cells,target))) == 0n : Nat},
    high: {Runs.range_high(Pieces.ranges_of_grid(Grid.grid(cells,target))) == ShapeLaws.elements(cells) : Nat}}

def grid_layout_facts(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+unit: Nat,+facts: Laws.GridFacts<limits,Grid.grid(cells,target)>,
  +units: Multiple<U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit>,
  +counted: {U32.to_nat(Grid.count_of(Grid.grid(cells,target))) == ShapeLaws.elements(cells) : Nat}) -> GridLayout<cells,target,unit>:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  +lit = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +kept = Equal.trans(U32,unit_cells,grid_unit_cells(Grid.grid(cells,target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
    Equal.sym(U32,grid_unit_cells(Grid.grid(cells,target)),unit_cells,Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),lit,exact)),
    Equal.trans(U32,grid_unit_cells(Grid.grid(cells,target)),grid_unit_cells(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)),unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),
      Equal.cong(Grid.Grid,U32,grid => grid_unit_cells(grid),Grid.grid(cells,target),Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target),Laws.grid_units(cells,target)),
      unit_cells_kept(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target),target)))
  +whole = multiple_moved(U32.to_nat(unit_cells),U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),unit,
    Equal.cong(U32,Nat,word => U32.to_nat(word),unit_cells,unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target)),kept),units)
  +run = {%Equal.sym(Grid.Grid,Grid.grid(cells,target),lit,exact) : Runs.Run(Pieces.ranges_of_grid(_),unit,U32.to_nat(Grid.count_of(_)))
    grid_run(limits,outer,extent,unit_cells,block,per,count,prefix,shape,total,unit,multiple_times(U32.to_nat(unit_cells),unit,whole),multiple_exact(U32.to_nat(unit_cells),unit,whole)) :
    Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,U32.to_nat(Grid.count_of(Grid.grid(cells,target))))}
  +low = {%Equal.sym(Grid.Grid,Grid.grid(cells,target),lit,exact) : {Runs.range_low(Pieces.ranges_of_grid(_)) == 0n : Nat}
    grid_low(limits,outer,extent,unit_cells,block,per,count,prefix,shape,total,unit,multiple_times(U32.to_nat(unit_cells),unit,whole),multiple_exact(U32.to_nat(unit_cells),unit,whole)) :
    {Runs.range_low(Pieces.ranges_of_grid(Grid.grid(cells,target))) == 0n : Nat}}
  +high = {%Equal.sym(Grid.Grid,Grid.grid(cells,target),lit,exact) : {Runs.range_high(Pieces.ranges_of_grid(_)) == U32.to_nat(Grid.count_of(_)) : Nat}
    grid_high(limits,outer,extent,unit_cells,block,per,count,prefix,shape,total) :
    {Runs.range_high(Pieces.ranges_of_grid(Grid.grid(cells,target))) == U32.to_nat(Grid.count_of(Grid.grid(cells,target))) : Nat}}
  GridLayout{%counted : Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,_) run,low,
    Equal.trans(Nat,Runs.range_high(Pieces.ranges_of_grid(Grid.grid(cells,target))),U32.to_nat(Grid.count_of(Grid.grid(cells,target))),ShapeLaws.elements(cells),high,counted)}

def grid_layout(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+unit: Nat,+divides: Multiple<last_extent(cells),unit>) -> GridLayout<cells,target,unit>:
  +whole = units_last(limits,cells,U32.is_le(Grid.product(cells),target),0,1,1,target,room,nonempty)
  grid_layout_facts(limits,cells,target,unit,grid_facts(limits,cells,target,room,nonempty),
    multiple_through(U32.to_nat(unit_cells_of(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),target),0,1,1,target))),last_extent(cells),unit,whole,divides),
    grid_count(limits,cells,target,room,nonempty))

# Pieces placed in the grid layout of a counted shape, each holding the
# values of a reader, are a tensor of the contract of that reader.
def layout_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),+unit: Nat,+counted: Global.Counted<shape>,+layout: GridLayout<shape,Grid.piece_target(),unit>,
  +placed: Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Grid.grid(shape,Grid.piece_target())),pieces>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  GridLayout{+run,+low,+high} = layout
  Placed.placed_contract(~Element,~Logical,~original,logical,shape,fill,workers,pieces,unit,counted,run,low,high,placed)

def grid_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+unit: Nat,+divides: Multiple<last_extent(shape),unit>,+counted: Global.Counted<shape>,
  +placed: Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Grid.grid(shape,Grid.piece_target())),pieces>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  layout_contract(~Element,~Logical,~original,logical,shape,fill,workers,pieces,unit,counted,grid_layout(limits,shape,Grid.piece_target(),room,nonempty,unit,divides),placed)

# Pieces whose contents lie in the grid layout of a counted shape, each
# holding the values of a reader, are a tensor of the contract of that reader.
def content_layout_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),+unit: Nat,+counted: Global.Counted<shape>,+layout: GridLayout<shape,Grid.piece_target(),unit>,
  +content: PiecesModel.Content<Element>,+contents_exact: {PiecesModel.contents(~Element,pieces) == content : PiecesModel.Content<Element>},
  +same_layout: {PiecesModel.content_layout(~Element,content) == Pieces.ranges_of_grid(Grid.grid(shape,Grid.piece_target())) : Pieces.Ranges},
  +holds: PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),content)) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  GridLayout{+run,+low,+high} = layout
  Placed.content_contract(~Element,~Logical,~original,logical,shape,fill,workers,pieces,unit,counted,run,low,high,content,contents_exact,same_layout,holds)

# The ranges of a grid layout are a run of its unit up to the count of the
# cells.
def layout_run(+cells: List<&2,U32>,+target: U32,+unit: Nat,+layout: GridLayout<cells,target,unit>) ->
  Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,ShapeLaws.elements(cells)):
  GridLayout{+run,+low,+high} = layout
  run

# ---- The pieces of the grid of a shape ----

# The grid of a shape has no more pieces than the capacity.
def grid_pieces_facts(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(cells,target)>) ->
  {Nat.is_le(U32.to_nat(Grid.pieces_of(Grid.grid(cells,target))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+total} = facts
  %Equal.sym(Grid.Grid,Grid.grid(cells,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : {Nat.is_le(U32.to_nat(Grid.pieces_of(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)

def grid_pieces_word(+limits: Machine.Limits,+cells: List<&2,U32>,+target: U32,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool}) -> {U32.is_le(Grid.pieces_of(Grid.grid(cells,target)),2147483648) == True{} : Bool}:
  +pieces = Grid.pieces_of(Grid.grid(cells,target))
  +within = {%Equal.sym(Bool,U32.is_le(pieces,Machine.capacity(limits)),Nat.is_le(U32.to_nat(pieces),U32.to_nat(Machine.capacity(limits))),Comparison.le(pieces,Machine.capacity(limits))) :
      {_ == True{} : Bool}
    grid_pieces_facts(limits,cells,target,grid_facts(limits,cells,target,room,nonempty)) : {U32.is_le(pieces,Machine.capacity(limits)) == True{} : Bool}}
  Order.word_le(pieces,Machine.capacity(limits),2147483648,within,LimitBounds.capacity_fits(limits))

# ---- The pieces of a task ----

# A multiple of a unit divided by the unit is its count of units.
def multiple_quotient(+units: Nat,+unit: Nat,+positive: {Nat.is_lt(0n,unit) == True{} : Bool}) -> {Nat.div(Nat.mul(units,unit),unit) == units : Nat}:
  %Addition.zero_right(Nat.mul(units,unit)) : {Nat.div(_,unit) == units : Nat}
  Radix.quotient(units,0n,unit,positive,positive)
# The pieces of a task from piece on, remaining of them: each a run of whole
# units at or after low, ending at or before bound.
def PieceRuns(remaining: Nat,+grid: Grid.Grid,+piece: U32,+low: Nat,+unit: Nat,+bound: Nat) -> Data:
  match remaining:
    case 0n:
      Unit
    case 1n+rest:
      Gemm.Both<Runs.Fits<unit,U32.to_nat(Grid.start(grid,piece)),Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece))),bound>,
        Gemm.Both<{Nat.is_le(low,U32.to_nat(Grid.start(grid,piece))) == True{} : Bool},
          PieceRuns(rest,grid,(piece + 1 : U32),Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece))),unit,bound)>>
# The next piece word after a piece inside the grid.
def next_piece_word(+limits: Machine.Limits,+outer: U32,+per: U32,+piece: Nat,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_le(1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) -> {(U32.from_nat(piece) + 1 : U32) == U32.from_nat(1n+piece) : U32}:
  +cap = Machine.capacity(limits)
  +next_room = Order.transitive(1n+piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),inside,room)
  +here_room = Order.transitive(piece,1n+piece,U32.to_nat(cap),Bounds.strict_le(piece,1n+piece,Order.lt_succ(piece,piece,Order.reflexive(piece))),next_room)
  +here = Bounds.roundtrip(piece,cap,LimitBounds.capacity_fits(limits),here_room)
  +next = Bounds.roundtrip(1n+piece,cap,LimitBounds.capacity_fits(limits),next_room)
  Arithmetic.to_nat_injective((U32.from_nat(piece) + 1 : U32),U32.from_nat(1n+piece),
    Equal.trans(Nat,U32.to_nat((U32.from_nat(piece) + 1 : U32)),1n+U32.to_nat(U32.from_nat(piece)),U32.to_nat(U32.from_nat(1n+piece)),
      Laws.successor_word(limits,U32.from_nat(piece),Held.nat_le_moved(U32.to_nat(U32.from_nat(piece)),piece,U32.to_nat(cap),here,here_room)),
      Equal.trans(Nat,1n+U32.to_nat(U32.from_nat(piece)),1n+piece,U32.to_nat(U32.from_nat(1n+piece)),
        Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(U32.from_nat(piece)),piece,here),Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next))))
# A chain of the pieces of a grid is a run of the pieces of a task.
def runs_of_chain(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},remaining: Nat,+piece: Nat,+low: Nat,+unit: Nat,+bound: Nat,
  +inside: {Nat.is_le(Nat.add(piece,remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +chain: Runs.Chain(piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),low,unit,bound)) ->
  PieceRuns(remaining,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece),low,unit,bound):
  match remaining:
    case 0n: Unit{}
    case 1n+ +more:
      Held.Both{+fits,+facts} = chain
      Held.Both{+starts,+next} = facts
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +start = U32.to_nat(Grid.start(grid,U32.from_nat(piece)))
      +stepped = Addition.successor_right(piece,more)
      +word = next_piece_word(limits,outer,per,piece,room,Order.transitive(1n+piece,Nat.add(piece,1n+more),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
        {%Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),stepped) : {Nat.is_le(1n+piece,_) == True{} : Bool}
          Order.le_add(piece,more) : {Nat.is_le(1n+piece,Nat.add(piece,1n+more)) == True{} : Bool}},inside))
      Gemm.Both{fits,Gemm.Both{{%starts : {Nat.is_le(_,start) == True{} : Bool} Order.reflexive(start) : {Nat.is_le(low,start) == True{} : Bool}},
        {%Equal.sym(U32,(U32.from_nat(piece) + 1 : U32),U32.from_nat(1n+piece),word) :
            PieceRuns(more,grid,_,Nat.add(start,U32.to_nat(Grid.span(grid,U32.from_nat(piece)))),unit,bound)
          runs_of_chain(limits,outer,extent,unit_cells,block,per,count,prefix,room,more,1n+piece,Nat.add(start,U32.to_nat(Grid.span(grid,U32.from_nat(piece)))),unit,bound,
            {%stepped : {Nat.is_le(_,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool} inside :
              {Nat.is_le(1n+Nat.add(piece,more),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}},next) :
          PieceRuns(more,grid,(U32.from_nat(piece) + 1 : U32),Nat.add(start,U32.to_nat(Grid.span(grid,U32.from_nat(piece)))),unit,bound)}}}
