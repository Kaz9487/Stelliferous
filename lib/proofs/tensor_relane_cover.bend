# The blocks of a relane piece cover it: every cell of the piece is the cell
# some block writes at some lane.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tasks.bend as Tasks
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_relane_writes.bend as Writes
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_target_cells.bend as Targets
import ./tensor_relane_block.bend as Block
import ./tensor_relane_blocks.bend as Blocks
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Logic
import ./word_arithmetic.bend as WordArithmetic
import ./word_subtraction.bend as Subtraction

# ---- The digits of a row-major number ----

# The digit of every axis of a number: the number over the cells of the later
# axes, modulo the extent of the axis.
def Decoded(counts: List<&2,U32>,+index: Nat,digits: List<&2,Nat>) -> Data:
  match counts digits:
    case Con{+count,+rest} Con{+digit,more}:
      Held.Both<{Nat.mod(Nat.div(index,ShapeLaws.elements(rest)),U32.to_nat(count)) == digit : Nat},Decoded(rest,index,more)>
    case Nil{} Nil{}: Unit
    case counts digits: Empty

# A number of whole rows before the digits keeps their digits.
def decoded(counts: List<&2,U32>,digits: List<&2,Nat>,+carry: Nat,+below: Coordinates.DigitsBelow(counts,digits)) ->
  Decoded(counts,Nat.add(Nat.mul(carry,ShapeLaws.elements(counts)),ShapeLaws.digits_value(counts,digits)),digits):
  match counts digits:
    case Con{+count,+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +inner = ShapeLaws.elements(rest)
      +tail = ShapeLaws.digits_value(rest,more)
      +size = U32.to_nat(count)
      +lead = Nat.add(Nat.mul(carry,size),digit)
      +tail_below = Coordinates.digits_below(rest,more,later)
      +positive = Order.le_lt_trans(0n,tail,inner,Order.zero_le(tail),tail_below)
      +size_positive = Order.le_lt_trans(0n,digit,size,Order.zero_le(digit),here)
      +number = Nat.add(Nat.mul(carry,Nat.mul(size,inner)),Nat.add(Nat.mul(digit,inner),tail))
      +front = Nat.add(Nat.mul(carry,Nat.mul(size,inner)),Nat.mul(digit,inner))
      +regrouped = Equal.trans(Nat,number,Nat.add(front,tail),Nat.add(Nat.mul(lead,inner),tail),
        Equal.sym(Nat,Nat.add(front,tail),number,Addition.associative(Nat.mul(carry,Nat.mul(size,inner)),Nat.mul(digit,inner),tail)),
        Equal.cong(Nat,Nat,value => Nat.add(value,tail),front,Nat.mul(lead,inner),
          Equal.trans(Nat,front,Nat.add(Nat.mul(Nat.mul(carry,size),inner),Nat.mul(digit,inner)),Nat.mul(lead,inner),
            Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(digit,inner)),Nat.mul(carry,Nat.mul(size,inner)),Nat.mul(Nat.mul(carry,size),inner),
              Equal.sym(Nat,Nat.mul(Nat.mul(carry,size),inner),Nat.mul(carry,Nat.mul(size,inner)),RowMajor.assoc(carry,size,inner))),
            Equal.sym(Nat,Nat.mul(lead,inner),Nat.add(Nat.mul(Nat.mul(carry,size),inner),Nat.mul(digit,inner)),RowMajor.distrib(Nat.mul(carry,size),digit,inner)))))
      +quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(lead,inner),tail),inner),(lead,tail),
        Division.quotient_remainder(lead,tail,inner,positive,tail_below))
      +remainder = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(lead,size),(carry,digit),
        Division.quotient_remainder(carry,digit,size,size_positive,here))
      Held.Both{
        %Equal.sym(Nat,number,Nat.add(Nat.mul(lead,inner),tail),regrouped) : {Nat.mod(Nat.div(_,inner),size) == digit : Nat}
        %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(lead,inner),tail),inner),lead,quotient) : {Nat.mod(_,size) == digit : Nat}
        remainder,
        {%Equal.sym(Nat,number,Nat.add(Nat.mul(lead,inner),tail),regrouped) : Decoded(rest,_,more) decoded(rest,more,lead,later) : Decoded(rest,number,more)}}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+digit,+more}:
      Empty.absurd(Decoded(Nil{},Nat.add(Nat.mul(carry,ShapeLaws.elements(Nil{})),ShapeLaws.digits_value(Nil{},Con{digit,more})),Con{digit,more}),below)
    case Con{+count,+rest} Nil{}:
      Empty.absurd(Decoded(Con{count,rest},Nat.add(Nat.mul(carry,ShapeLaws.elements(Con{count,rest})),ShapeLaws.digits_value(Con{count,rest},Nil{})),Nil{}),below)

# ---- The block coordinates of a block index ----

def counts_of(axes: List<&2,Tiled.Ranged>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{blocked,low,extent,+count,inner,target},rest}: count <> counts_of(rest)

# Every axis counts the blocks of the later axes as its inner blocks.
def Inners(axes: List<&2,Tiled.Ranged>) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Tiled.Ranged{blocked,low,extent,count,+inner,target},+rest}: Held.Both<{inner == Grid.product(counts_of(rest)) : U32},Inners(rest)>

def head_inner(axes: List<&2,Tiled.Ranged>,+inners: Inners(axes)) -> {Tiled.ranged_head_count(axes) == Grid.product(counts_of(axes)) : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Tiled.Ranged{+blocked,+low,+extent,+count,+inner,+target},+rest}:
      Held.Both{+here,+later} = inners
      Equal.cong(U32,U32,word => (count * word : U32),inner,Grid.product(counts_of(rest)),here)

def ranged_inners(axes: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,targets: List<&2,U32>) -> Inners(Tiled.ranged_axes(axes,lows,extents,targets)):
  match axes lows extents targets:
    case Con{+blocked,+rest} Con{+low,+more} Con{+extent,+others} Con{+target,+later}:
      +inners = ranged_inners(rest,more,others,later)
      Held.Both{head_inner(Tiled.ranged_axes(rest,more,others,later),inners),inners}
    case Nil{} +lows +extents +targets: Unit{}
    case Con{+blocked,+rest} Nil{} +extents +targets: Unit{}
    case Con{+blocked,+rest} Con{+low,+more} Nil{} +targets: Unit{}
    case Con{+blocked,+rest} Con{+low,+more} Con{+extent,+others} Nil{}: Unit{}

# The index of an axis of a block: its number over the inner blocks, modulo
# the blocks of the axis.
def axis_index(+index: U32,+inner: U32,+count: U32,+inner_value: Nat,+count_value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inner_exact: {U32.to_nat(inner) == inner_value : Nat},+count_exact: {U32.to_nat(count) == count_value : Nat},
  +inner_positive: {Nat.is_lt(0n,inner_value) == True{} : Bool},+count_positive: {Nat.is_lt(0n,count_value) == True{} : Bool},
  +inner_room: {Nat.is_le(Nat.double(inner_value),U32.to_nat(cap)) == True{} : Bool},+count_room: {Nat.is_le(Nat.double(count_value),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((index / U32.max(inner,1) % U32.max(count,1) : U32)) == Nat.mod(Nat.div(U32.to_nat(index),inner_value),count_value) : Nat}:
  +inner_one = U32.max(inner,1)
  +count_one = U32.max(count,1)
  +inner_max = Extrema.max_one_exact(inner,inner_value,inner_exact,inner_positive)
  +count_max = Extrema.max_one_exact(count,count_value,count_exact,count_positive)
  +inner_back = Equal.sym(Nat,U32.to_nat(inner_one),inner_value,inner_max)
  +count_back = Equal.sym(Nat,U32.to_nat(count_one),count_value,count_max)
  +quotient = (index / inner_one : U32)
  +divided = Equal.trans(Nat,U32.to_nat(quotient),Nat.div(U32.to_nat(index),U32.to_nat(inner_one)),Nat.div(U32.to_nat(index),inner_value),
    U32Division.divide(index,inner_one,cap,cap_ok,{%inner_back : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} inner_room :
        {Nat.is_le(Nat.double(U32.to_nat(inner_one)),U32.to_nat(cap)) == True{} : Bool}},
      {%inner_back : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,U32.to_nat(inner_one)) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.div(U32.to_nat(index),value),U32.to_nat(inner_one),inner_value,inner_max))
  Equal.trans(Nat,U32.to_nat((quotient % count_one : U32)),Nat.mod(U32.to_nat(quotient),U32.to_nat(count_one)),Nat.mod(Nat.div(U32.to_nat(index),inner_value),count_value),
    U32Division.modulo(quotient,count_one,cap,cap_ok,{%count_back : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} count_room :
        {Nat.is_le(Nat.double(U32.to_nat(count_one)),U32.to_nat(cap)) == True{} : Bool}},
      {%count_back : {Nat.is_lt(0n,_) == True{} : Bool} count_positive : {Nat.is_lt(0n,U32.to_nat(count_one)) == True{} : Bool}}),
    Equal.trans(Nat,Nat.mod(U32.to_nat(quotient),U32.to_nat(count_one)),Nat.mod(Nat.div(U32.to_nat(index),inner_value),U32.to_nat(count_one)),
      Nat.mod(Nat.div(U32.to_nat(index),inner_value),count_value),
      Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(count_one)),U32.to_nat(quotient),Nat.div(U32.to_nat(index),inner_value),divided),
      Equal.cong(Nat,Nat,value => Nat.mod(Nat.div(U32.to_nat(index),inner_value),value),U32.to_nat(count_one),count_value,count_max)))

# The index of every axis of a block.
def AxesIndex(axes: List<&2,Tiled.Ranged>,+index: U32,coords: List<&2,Nat>) -> Data:
  match axes coords:
    case Con{Tiled.Ranged{blocked,low,extent,+count,+inner,target},rest} Con{+coord,more}:
      Held.Both<{U32.to_nat((index / U32.max(inner,1) % U32.max(count,1) : U32)) == coord : Nat},AxesIndex(rest,index,more)>
    case Nil{} Nil{}: Unit
    case axes coords: Empty

# A block numbered by coordinates below the block counts has them as its
# axis indices.
def axes_index(axes: List<&2,Tiled.Ranged>,coords: List<&2,Nat>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inners: Inners(axes),+room: {Nat.is_le(Nat.double(ShapeLaws.elements(counts_of(axes))),U32.to_nat(cap)) == True{} : Bool},
  +decoded: Decoded(counts_of(axes),U32.to_nat(index),coords),+below: Coordinates.DigitsBelow(counts_of(axes),coords)) -> AxesIndex(axes,index,coords):
  match axes coords:
    case Con{Tiled.Ranged{+blocked,+low,+extent,+count,+inner,+target},+rest} Con{+coord,+more}:
      Held.Both{+inner_same,+inners_rest} = inners
      Held.Both{+here,+later} = decoded
      KernelGemm.Both{+coord_below,+below_rest} = below
      +rest_counts = counts_of(rest)
      +inner_value = ShapeLaws.elements(rest_counts)
      +size = U32.to_nat(count)
      +whole = Nat.mul(size,inner_value)
      +size_positive = Order.le_lt_trans(0n,coord,size,Order.zero_le(coord),coord_below)
      +inner_positive = Order.le_lt_trans(0n,ShapeLaws.digits_value(rest_counts,more),inner_value,Order.zero_le(ShapeLaws.digits_value(rest_counts,more)),
        Coordinates.digits_below(rest_counts,more,below_rest))
      +inner_le = {%RowMajor.multiply_commutative(inner_value,size) : {Nat.is_le(inner_value,_) == True{} : Bool}
        RowMajor.product_covers(inner_value,size,size_positive) : {Nat.is_le(inner_value,whole) == True{} : Bool}}
      +rest_room = Order.transitive(Nat.double(inner_value),Nat.double(whole),U32.to_nat(cap),Order.double_le(inner_value,whole,inner_le),room)
      +size_room = Order.transitive(Nat.double(size),Nat.double(whole),U32.to_nat(cap),Order.double_le(size,whole,RowMajor.product_covers(size,inner_value,inner_positive)),
        room)
      +inner_exact = Equal.trans(Nat,U32.to_nat(inner),U32.to_nat(Grid.product(rest_counts)),inner_value,
        Equal.cong(U32,Nat,word => U32.to_nat(word),inner,Grid.product(rest_counts),inner_same),
        Public.product_exact(cap,cap_ok,rest_counts,Order.transitive(inner_value,Nat.double(inner_value),U32.to_nat(cap),Order.le_double(inner_value),rest_room)))
      Held.Both{Equal.trans(Nat,U32.to_nat((index / U32.max(inner,1) % U32.max(count,1) : U32)),Nat.mod(Nat.div(U32.to_nat(index),inner_value),size),coord,
          axis_index(index,inner,count,inner_value,size,cap,cap_ok,inner_exact,{==},inner_positive,size_positive,rest_room,size_room),here),
        axes_index(rest,more,index,cap,cap_ok,inners_rest,rest_room,later,below_rest)}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+coord,+more}: Empty.absurd(AxesIndex(Nil{},index,Con{coord,more}),decoded)
    case Con{Tiled.Ranged{+blocked,+low,+extent,+count,+inner,+target},+rest} Nil{}:
      Empty.absurd(AxesIndex(Con{Tiled.Ranged{blocked,low,extent,count,inner,target},rest},index,Nil{}),decoded)

# ---- The block and lane of a box cell ----

# The block coordinate of the digit of an axis: the source lane axis counts
# its blocks of eight from the block of its first index.
def cover_coord(role: Tiled.Role,+low: U32,+digit: Nat) -> Nat:
  match role:
    case Tiled.Lanes{}: Nat.sub(Nat.div(Nat.add(U32.to_nat(low),digit),8n),Nat.div(U32.to_nat(low),8n))
    case Tiled.Shared{}: digit
    case Tiled.Groups{}: digit

def cover_coords(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>) -> List<&2,Nat>:
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,extent,count,inner,target},rest} Con{+digit,more}: cover_coord(role,low,digit) <> cover_coords(rest,more)
    case axes digits: []

# The lane of the digit of the source lane axis in its block of eight.
def cover_lane(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>) -> Nat:
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},+low,extent,count,inner,target},rest} Con{+digit,more}: Nat.mod(Nat.add(U32.to_nat(low),digit),8n)
    case Con{axis,rest} Con{digit,more}: cover_lane(rest,more)
    case axes digits: 0n

def LaneAt(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>,+lane: Nat) -> Data:
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},+low,extent,count,inner,target},rest} Con{+digit,more}:
      {lane == Nat.mod(Nat.add(U32.to_nat(low),digit),8n) : Nat}
    case Con{axis,rest} Con{digit,more}: LaneAt(rest,more,lane)
    case axes digits: Unit

def lane_at(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>) -> LaneAt(axes,digits,cover_lane(axes,digits)):
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: lane_at(rest,more)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: lane_at(rest,more)
    case Nil{} +digits: Unit{}
    case Con{+axis,+rest} Nil{}: Unit{}

def no_lane_at(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>,+lane: Nat,+none: Geometry.NoLanes(Geometry.blocked_of(axes))) -> LaneAt(axes,digits,lane):
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      Empty.absurd({lane == Nat.mod(Nat.add(U32.to_nat(low),digit),8n) : Nat},none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: no_lane_at(rest,more,lane,none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: no_lane_at(rest,more,lane,none)
    case Nil{} +digits: Unit{}
    case Con{+axis,+rest} Nil{}: Unit{}

# The group of the block of a lane digit is the eighth of the digit.
def lane_group(+low: U32,+digit: Nat,+at: U32,
  +at_value: {U32.to_nat(at) == Nat.sub(Nat.div(Nat.add(U32.to_nat(low),digit),8n),Nat.div(U32.to_nat(low),8n)) : Nat}) ->
  {Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)) == Nat.div(Nat.add(U32.to_nat(low),digit),8n) : Nat}:
  +l = U32.to_nat(low)
  +whole = Nat.div(Nat.add(l,digit),8n)
  +start = Nat.div(l,8n)
  %Equal.sym(Nat,U32.to_nat((low / 8 : U32)),start,U32Division.divide(low,8,2147483648,{==},{==},{==})) : {Nat.add(_,U32.to_nat(at)) == whole : Nat}
  %Equal.sym(Nat,U32.to_nat(at),Nat.sub(whole,start),at_value) : {Nat.add(start,_) == whole : Nat}
  Order.restore_shift(start,whole,DivisionOrder.quotient_le(l,Nat.add(l,digit),8n,{==},Order.le_add(l,digit)))

def added_less(+low: Nat,+digit: Nat) -> {Nat.sub(Nat.add(low,digit),low) == digit : Nat}:
  Equal.trans(Nat,Nat.sub(Nat.add(low,digit),low),Nat.sub(Nat.add(digit,low),low),digit,
    Equal.cong(Nat,Nat,value => Nat.sub(value,low),Nat.add(low,digit),Nat.add(digit,low),Order.commutative(low,digit)),DivisionOrder.subtract_sum(digit,low))

# The group of a digit past the group of the first index lies at most at the
# group of the last index.
def group_le(+low: Nat,+extent: Nat,+digit: Nat,+digit_below: {Nat.is_lt(digit,extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(Nat.div(Nat.add(low,digit),8n),Nat.div(low,8n)),Nat.sub(Nat.div(Nat.sub(Nat.add(low,extent),1n),8n),Nat.div(low,8n))) == True{} : Bool}:
  +sum = Nat.add(low,extent)
  +reach = Nat.add(low,digit)
  +reach_last = {%Order.subtract_zero(reach) : {Nat.is_le(_,Nat.sub(sum,1n)) == True{} : Bool}
    DivisionOrder.subtract_le(1n+reach,sum,1n,Order.strict_follow(reach,sum,Order.offset_lt(low,digit,extent,digit_below))) :
    {Nat.is_le(reach,Nat.sub(sum,1n)) == True{} : Bool}}
  DivisionOrder.subtract_le(Nat.div(reach,8n),Nat.div(Nat.sub(sum,1n),8n),Nat.div(low,8n),DivisionOrder.quotient_le(reach,Nat.sub(sum,1n),8n,{==},reach_last))

# The lane block at the block of a lane digit writes from at most its lane to
# past it.
def lane_cover_with(+low: U32,+extent: U32,+at: U32,+lane: U32,+digit: Nat,+digit_below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +at_value: {U32.to_nat(at) == Nat.sub(Nat.div(Nat.add(U32.to_nat(low),digit),8n),Nat.div(U32.to_nat(low),8n)) : Nat},
  +lane_value: {U32.to_nat(lane) == Nat.mod(Nat.add(U32.to_nat(low),digit),8n) : Nat},+words: Writes.LaneWords(low,extent,at)) ->
  Held.Both<{Nat.is_le(U32.to_nat(Writes.lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(lane),U32.to_nat(Writes.lane_high(low,extent,at))) == True{} : Bool}>:
  Held.Both{+low_word,Held.Both{+high_word,+first_below}} = words
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +sum = Nat.add(l,e)
  +reach = Nat.add(l,digit)
  +f = Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n)
  +g = Nat.mul(Nat.div(reach,8n),8n)
  +m = Nat.mod(reach,8n)
  +same = Equal.cong(Nat,Nat,value => Nat.mul(value,8n),Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),Nat.div(reach,8n),lane_group(low,digit,at,at_value))
  +split = Equal.sym(Nat,reach,Nat.add(g,m),DivisionOrder.quotient_exact(reach,8n,{==}))
  +g_reach = DivisionOrder.product_below(reach,8n,{==})
  +reach_sum = Order.offset_lt(l,digit,e,digit_below)
  +g_sum = Order.transitive(g,reach,sum,g_reach,Bounds.strict_le(reach,sum,reach_sum))
  +rest = Equal.trans(Nat,Nat.sub(reach,g),Nat.sub(Nat.add(g,m),g),m,
    Equal.cong(Nat,Nat,value => Nat.sub(value,g),reach,Nat.add(g,m),Equal.sym(Nat,Nat.add(g,m),reach,split)),added_less(g,m))
  +from_g = {%rest : {Nat.is_le(Nat.sub(Nat.max(l,g),g),_) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.max(l,g),reach,g,NatExtrema.maximum_upper(l,g,reach,Order.le_add(l,digit),g_reach)) :
    {Nat.is_le(Nat.sub(Nat.max(l,g),g),m) == True{} : Bool}}
  +below_reach = Equal.trans(Bool,Nat.is_lt(m,Nat.sub(sum,g)),Nat.is_lt(Nat.add(g,m),Nat.add(g,Nat.sub(sum,g))),True{},
    Equal.sym(Bool,Nat.is_lt(Nat.add(g,m),Nat.add(g,Nat.sub(sum,g))),Nat.is_lt(m,Nat.sub(sum,g)),Order.lt_offset(g,m,Nat.sub(sum,g))),
    %Equal.sym(Nat,Nat.add(g,Nat.sub(sum,g)),sum,Order.restore_shift(g,sum,g_sum)) : {Nat.is_lt(Nat.add(g,m),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(g,m),reach,split) : {Nat.is_lt(_,sum) == True{} : Bool}
    reach_sum)
  +to_g = {%Equal.sym(Bool,Nat.is_lt(m,Nat.min(Nat.sub(sum,g),8n)),Nat.is_lt(m,Nat.sub(sum,g)) && Nat.is_lt(m,8n),NatExtrema.below_minimum(m,Nat.sub(sum,g),8n)) :
      {_ == True{} : Bool}
    %Equal.sym(Bool,Nat.is_lt(m,Nat.sub(sum,g)),True{},below_reach) : {_ && Nat.is_lt(m,8n) == True{} : Bool}
    DivisionOrder.remainder_small(reach,8n,{==}) : {Nat.is_lt(m,Nat.min(Nat.sub(sum,g),8n)) == True{} : Bool}}
  +moved = Equal.cong(Nat,Nat,value => Nat.sub(Nat.max(l,value),value),f,g,same)
  +reached = Equal.cong(Nat,Nat,value => Nat.min(Nat.sub(sum,value),8n),f,g,same)
  Held.Both{
    %Equal.sym(Nat,U32.to_nat(Writes.lane_low(low,at)),Nat.sub(Nat.max(l,f),f),low_word) : {Nat.is_le(_,U32.to_nat(lane)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(lane),m,lane_value) : {Nat.is_le(Nat.sub(Nat.max(l,f),f),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.sub(Nat.max(l,f),f),Nat.sub(Nat.max(l,g),g),moved) : {Nat.is_le(_,m) == True{} : Bool}
    from_g,
    %Equal.sym(Nat,U32.to_nat(Writes.lane_high(low,extent,at)),Nat.min(Nat.sub(sum,f),8n),high_word) : {Nat.is_lt(U32.to_nat(lane),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(lane),m,lane_value) : {Nat.is_lt(_,Nat.min(Nat.sub(sum,f),8n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.min(Nat.sub(sum,f),8n),Nat.min(Nat.sub(sum,g),8n),reached) : {Nat.is_lt(m,_) == True{} : Bool}
    to_g}

def lane_cover(+low: U32,+extent: U32,+at: U32,+lane: U32,+digit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +digit_below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +at_value: {U32.to_nat(at) == Nat.sub(Nat.div(Nat.add(U32.to_nat(low),digit),8n),Nat.div(U32.to_nat(low),8n)) : Nat},
  +lane_value: {U32.to_nat(lane) == Nat.mod(Nat.add(U32.to_nat(low),digit),8n) : Nat}) ->
  Held.Both<{Nat.is_le(U32.to_nat(Writes.lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(lane),U32.to_nat(Writes.lane_high(low,extent,at))) == True{} : Bool}>:
  +l = U32.to_nat(low)
  +sum = Nat.add(l,U32.to_nat(extent))
  +reach = Nat.add(l,digit)
  +at_le = {%Equal.sym(Nat,U32.to_nat(at),Nat.sub(Nat.div(reach,8n),Nat.div(l,8n)),at_value) : {Nat.is_le(_,Nat.sub(Nat.div(Nat.sub(sum,1n),8n),Nat.div(l,8n))) == True{} : Bool}
    group_le(l,U32.to_nat(extent),digit,digit_below) : {Nat.is_le(U32.to_nat(at),Nat.sub(Nat.div(Nat.sub(sum,1n),8n),Nat.div(l,8n))) == True{} : Bool}}
  lane_cover_with(low,extent,at,lane,digit,digit_below,at_value,lane_value,Writes.lane_words(low,extent,at,cap,cap_ok,positive,room,at_le))

# ---- The block coordinates of box digits ----

def lanes_below(+low: U32,+extent: U32,+count: U32,+digit: Nat,+digit_below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +counts: Writes.LaneCount(low,extent,count)) -> {Nat.is_lt(cover_coord(Tiled.Lanes{},low,digit),U32.to_nat(count)) == True{} : Bool}:
  Held.Both{+count_word,+count_le} = counts
  +l = U32.to_nat(low)
  +difference = Nat.sub(Nat.div(Nat.sub(Nat.add(l,U32.to_nat(extent)),1n),8n),Nat.div(l,8n))
  %Equal.sym(Nat,U32.to_nat(count),Nat.add(difference,1n),count_word) : {Nat.is_lt(cover_coord(Tiled.Lanes{},low,digit),_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(difference,1n),1n+difference,Addition.successor_width(difference)) : {Nat.is_lt(cover_coord(Tiled.Lanes{},low,digit),_) == True{} : Bool}
  Order.lt_succ(cover_coord(Tiled.Lanes{},low,digit),difference,group_le(l,U32.to_nat(extent),digit,digit_below))

def lanes_le(+low: U32,+extent: U32,+count: U32,+counts: Writes.LaneCount(low,extent,count)) -> {Nat.is_le(U32.to_nat(count),U32.to_nat(extent)) == True{} : Bool}:
  Held.Both{+count_word,+count_le} = counts
  +l = U32.to_nat(low)
  +difference = Nat.sub(Nat.div(Nat.sub(Nat.add(l,U32.to_nat(extent)),1n),8n),Nat.div(l,8n))
  %Equal.sym(Nat,U32.to_nat(count),Nat.add(difference,1n),count_word) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
  count_le

# The block coordinate of a digit below the extent lies below the blocks of
# the axis, and those number at most the extent.
def coord_below(role: Tiled.Role,+low: U32,+extent: U32,+count: U32,+digit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +counted: {count == Tiled.block_count(role,low,extent) : U32},+digit_below: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_lt(cover_coord(role,low,digit),U32.to_nat(count)) == True{} : Bool}:
  match role:
    case Tiled.Lanes{}: lanes_below(low,extent,count,digit,digit_below,Writes.lane_count(low,extent,count,cap,cap_ok,positive,room,counted))
    case Tiled.Shared{}:
      %Equal.cong(U32,Nat,word => U32.to_nat(word),extent,count,Equal.sym(U32,count,extent,counted)) : {Nat.is_lt(digit,_) == True{} : Bool}
      digit_below
    case Tiled.Groups{}:
      %Equal.cong(U32,Nat,word => U32.to_nat(word),extent,count,Equal.sym(U32,count,extent,counted)) : {Nat.is_lt(digit,_) == True{} : Bool}
      digit_below

def count_le(role: Tiled.Role,+low: U32,+extent: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +counted: {count == Tiled.block_count(role,low,extent) : U32}) -> {Nat.is_le(U32.to_nat(count),U32.to_nat(extent)) == True{} : Bool}:
  match role:
    case Tiled.Lanes{}: lanes_le(low,extent,count,Writes.lane_count(low,extent,count,cap,cap_ok,positive,room,counted))
    case Tiled.Shared{}:
      %Equal.cong(U32,Nat,word => U32.to_nat(word),extent,count,Equal.sym(U32,count,extent,counted)) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
      Order.reflexive(U32.to_nat(extent))
    case Tiled.Groups{}:
      %Equal.cong(U32,Nat,word => U32.to_nat(word),extent,count,Equal.sym(U32,count,extent,counted)) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
      Order.reflexive(U32.to_nat(extent))

def cover_below(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ok: Writes.RangedOk(axes,cap),
  +below: Coordinates.DigitsBelow(Writes.extents_of(axes),digits)) -> Coordinates.DigitsBelow(counts_of(axes),cover_coords(axes,digits)):
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      KernelGemm.Both{+positive,+tail} = ok
      KernelGemm.Both{+room,+others} = tail
      KernelGemm.Both{+counted,+later} = others
      KernelGemm.Both{+here,+after} = below
      KernelGemm.Both{coord_below(role,low,extent,count,digit,cap,cap_ok,positive,room,counted,here),cover_below(rest,more,cap,cap_ok,later,after)}
    case Nil{} +digits: Unit{}
    case Con{+axis,+rest} Nil{}: below

def counts_le(axes: List<&2,Tiled.Ranged>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ok: Writes.RangedOk(axes,cap)) ->
  {Nat.is_le(ShapeLaws.elements(counts_of(axes)),ShapeLaws.elements(Writes.extents_of(axes))) == True{} : Bool}:
  match axes:
    case Nil{}: Order.reflexive(1n)
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{+positive,+tail} = ok
      KernelGemm.Both{+room,+others} = tail
      KernelGemm.Both{+counted,+later} = others
      +inner_counts = ShapeLaws.elements(counts_of(rest))
      Order.transitive(Nat.mul(U32.to_nat(count),inner_counts),Nat.mul(U32.to_nat(extent),inner_counts),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Writes.extents_of(rest))),
        RowMajor.mul_le(U32.to_nat(count),U32.to_nat(extent),inner_counts,count_le(role,low,extent,count,cap,cap_ok,positive,room,counted)),
        DivisionOrder.multiply_left_le(U32.to_nat(extent),inner_counts,ShapeLaws.elements(Writes.extents_of(rest)),counts_le(rest,cap,cap_ok,later)))

def cover_lane_small(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>) -> {Nat.is_lt(cover_lane(axes,digits),8n) == True{} : Bool}:
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      DivisionOrder.remainder_small(Nat.add(U32.to_nat(low),digit),8n,{==})
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: cover_lane_small(rest,more)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}: cover_lane_small(rest,more)
    case Nil{} +digits: {==}
    case Con{+axis,+rest} Nil{}: {==}

# ---- The block at the coordinates of box digits ----

def shared_digit(+low: U32,+digit: Nat,+at: U32,+at_value: {U32.to_nat(at) == digit : Nat}) ->
  {Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(at)),U32.to_nat(low)) == digit : Nat}:
  %Equal.sym(Nat,U32.to_nat(at),digit,at_value) : {Nat.sub(Nat.add(U32.to_nat(low),_),U32.to_nat(low)) == digit : Nat}
  added_less(U32.to_nat(low),digit)

def lanes_digit(+low: U32,+digit: Nat,+at: U32,+lane: Nat,
  +at_value: {U32.to_nat(at) == Nat.sub(Nat.div(Nat.add(U32.to_nat(low),digit),8n),Nat.div(U32.to_nat(low),8n)) : Nat},
  +lane_value: {lane == Nat.mod(Nat.add(U32.to_nat(low),digit),8n) : Nat}) ->
  {Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,at,lane),U32.to_nat(low)) == digit : Nat}:
  +l = U32.to_nat(low)
  +reach = Nat.add(l,digit)
  %Equal.sym(Nat,Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),Nat.div(reach,8n),lane_group(low,digit,at,at_value)) :
    {Nat.sub(Nat.add(Nat.mul(_,8n),lane),l) == digit : Nat}
  %Equal.sym(Nat,lane,Nat.mod(reach,8n),lane_value) : {Nat.sub(Nat.add(Nat.mul(Nat.div(reach,8n),8n),_),l) == digit : Nat}
  %DivisionOrder.quotient_exact(reach,8n,{==}) : {Nat.sub(_,l) == digit : Nat}
  added_less(l,digit)

# The block numbered by the block coordinates of box digits, at the lane of
# their source lane digit, has them as its box digits.
def cover_box(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>,+index: U32,+lane: Nat,+most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),
  +indices: AxesIndex(axes,index,cover_coords(axes,digits)),+placed: LaneAt(axes,digits,lane),
  +below: Coordinates.DigitsBelow(Writes.extents_of(axes),digits)) -> {Writes.box_digits(axes,index,lane) == digits : List<&2,Nat>}:
  match axes digits:
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      Held.Both{+here,+later} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      %Equal.sym(Nat,Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,at,lane),U32.to_nat(low)),digit,lanes_digit(low,digit,at,lane,here,placed)) :
        {Con{_,Writes.box_digits(rest,index,lane)} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Writes.box_digits(rest,index,lane),more,
          cover_box(rest,more,index,lane,Geometry.no_lanes_most(Geometry.blocked_of(rest),most),later,no_lane_at(rest,more,lane,most),more_below)) :
        {Con{digit,_} == Con{digit,more} : List<&2,Nat>}
      {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      Held.Both{+here,+later} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(at)),U32.to_nat(low)),digit,shared_digit(low,digit,at,here)) :
        {Con{_,Writes.box_digits(rest,index,lane)} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Writes.box_digits(rest,index,lane),more,cover_box(rest,more,index,lane,most,later,placed,more_below)) :
        {Con{digit,_} == Con{digit,more} : List<&2,Nat>}
      {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}:
      Held.Both{+here,+later} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(at)),U32.to_nat(low)),digit,shared_digit(low,digit,at,here)) :
        {Con{_,Writes.box_digits(rest,index,lane)} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Writes.box_digits(rest,index,lane),more,cover_box(rest,more,index,lane,most,later,placed,more_below)) :
        {Con{digit,_} == Con{digit,more} : List<&2,Nat>}
      {==}
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({Writes.box_digits(Nil{},index,lane) == Con{digit,more} : List<&2,Nat>},below)
    case Con{+axis,+rest} Nil{}: Empty.absurd({Writes.box_digits(Con{axis,rest},index,lane) == Nil{} : List<&2,Nat>},below)

# ---- The lanes the block writes ----

def KeptRange(+block: Tiled.Block,+lane: U32) -> Data:
  Held.Both<{Nat.is_le(U32.to_nat(Writes.low_of(block)),U32.to_nat(lane)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(lane),U32.to_nat(Writes.high_of(block))) == True{} : Bool}>

def kept_range(+final: Tiled.Block,+next: Tiled.Block,+lane: U32,+kept: {Writes.writes_of(final) == Writes.writes_of(next) : U32 & (U32 & U32)},
  +range: KeptRange(next,lane)) -> KeptRange(final,lane):
  Held.Both{+from_low,+to_high} = range
  +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),Writes.writes_of(final),Writes.writes_of(next),kept)
  +high_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),Writes.writes_of(final),Writes.writes_of(next),kept)
  Held.Both{
    %Equal.sym(U32,Writes.low_of(final),Writes.low_of(next),low_kept) : {Nat.is_le(U32.to_nat(_),U32.to_nat(lane)) == True{} : Bool}
    from_low,
    %Equal.sym(U32,Writes.high_of(final),Writes.high_of(next),high_kept) : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool}
    to_high}

def range_from(+block: Tiled.Block,+lane: U32,+range: KeptRange(block,lane)) -> {Nat.is_le(U32.to_nat(Writes.low_of(block)),U32.to_nat(lane)) == True{} : Bool}:
  Held.Both{+from_low,+to_high} = range
  from_low

def range_to(+block: Tiled.Block,+lane: U32,+range: KeptRange(block,lane)) -> {Nat.is_lt(U32.to_nat(lane),U32.to_nat(Writes.high_of(block))) == True{} : Bool}:
  Held.Both{+from_low,+to_high} = range
  to_high

# The block at the block coordinates of box digits writes the lane of their
# source lane digit.
def cover_range(axes: List<&2,Tiled.Ranged>,digits: List<&2,Nat>,block: Tiled.Block,+index: U32,+lane: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),+ok: Writes.RangedOk(axes,cap),
  +indices: AxesIndex(axes,index,cover_coords(axes,digits)),+below: Coordinates.DigitsBelow(Writes.extents_of(axes),digits),
  +lane_value: {U32.to_nat(lane) == cover_lane(axes,digits) : Nat},+low_zero: {Writes.low_of(block) == 0 : U32},+high_one: {Writes.high_of(block) == 1 : U32}) ->
  KeptRange(Tiled.block_at(axes,index,block),lane):
  match axes digits block:
    case Nil{} Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      Held.Both{
        %Equal.sym(U32,block_low,0,low_zero) : {Nat.is_le(U32.to_nat(_),U32.to_nat(lane)) == True{} : Bool}
        Order.zero_le(U32.to_nat(lane)),
        %Equal.sym(U32,block_high,1,high_one) : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(lane),0n,lane_value) : {Nat.is_lt(_,1n) == True{} : Bool}
        {==}}
    case Nil{} Con{+digit,+more} +block: Empty.absurd(KeptRange(Tiled.block_at(Nil{},index,block),lane),below)
    case Con{+axis,+rest} Nil{} +block: Empty.absurd(KeptRange(Tiled.block_at(Con{axis,rest},index,block),lane),below)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}
      Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+tail} = ok
      KernelGemm.Both{+room,+others} = tail
      KernelGemm.Both{+counted,+later} = others
      Held.Both{+here,+after} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      cover_range(rest,more,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,after,more_below,lane_value,low_zero,high_one)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}
      Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+tail} = ok
      KernelGemm.Both{+room,+others} = tail
      KernelGemm.Both{+counted,+later} = others
      Held.Both{+here,+after} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      cover_range(rest,more,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,after,more_below,lane_value,low_zero,high_one)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Con{+digit,+more}
      Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+tail} = ok
      KernelGemm.Both{+room,+others} = tail
      KernelGemm.Both{+counted,+later} = others
      Held.Both{+here,+after} = indices
      KernelGemm.Both{+digit_below,+more_below} = below
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      kept_range(Tiled.block_at(rest,index,next),next,lane,Writes.kept_writes(rest,next,index,most),
        lane_cover(low,extent,at,lane,digit,cap,cap_ok,positive,room,digit_below,here,lane_value))

# ---- Some write hits the cell ----

def successor_word(+count: Nat) -> {U32.from_nat(1n+count) == (U32.from_nat(count) + 1 : U32) : U32}:
  Equal.trans(U32,U32.from_nat(1n+count),U32.from_nat(Nat.add(count,1n)),(U32.from_nat(count) + 1 : U32),
    Equal.cong(Nat,U32,value => U32.from_nat(value),1n+count,Nat.add(count,1n),Equal.sym(Nat,Nat.add(count,1n),1n+count,Addition.successor_width(count))),
    Equal.sym(U32,(U32.from_nat(count) + U32.from_nat(1n) : U32),U32.from_nat(Nat.add(count,1n)),WordArithmetic.native_add(count,1n)))

# A write one stride further and one step closer lands on the same cell.
def moved_lands(+target: U32,+stride: U32,+count: Nat) ->
  {((target + stride : U32) + (U32.from_nat(count) * stride : U32) : U32) == (target + (U32.from_nat(1n+count) * stride : U32) : U32) : U32}:
  +step = (U32.from_nat(count) * stride : U32)
  Equal.trans(U32,((target + stride : U32) + step : U32),(target + (stride + step : U32) : U32),(target + (U32.from_nat(1n+count) * stride : U32) : U32),
    Arithmetic.add_associative(target,stride,step),
    Equal.cong(U32,U32,word => (target + word : U32),(stride + step : U32),(U32.from_nat(1n+count) * stride : U32),
      Equal.trans(U32,(stride + step : U32),(step + stride : U32),(U32.from_nat(1n+count) * stride : U32),Arithmetic.add_commutative(stride,step),
        Equal.trans(U32,(step + stride : U32),((U32.from_nat(count) + 1 : U32) * stride : U32),(U32.from_nat(1n+count) * stride : U32),
          Equal.sym(U32,((U32.from_nat(count) + 1 : U32) * stride : U32),(step + stride : U32),Arithmetic.multiply_successor(U32.from_nat(count),stride)),
          Equal.cong(U32,U32,word => (word * stride : U32),(U32.from_nat(count) + 1 : U32),U32.from_nat(1n+count),
            Equal.sym(U32,U32.from_nat(1n+count),(U32.from_nat(count) + 1 : U32),successor_word(count)))))))

# The writes of a block from a lane on hit a cell that the write a number of
# steps further lands on, while that write lies before the end of the block.
def writes_reach(fuel: Nat,+index: U32,+high: U32,+target: U32,+stride: U32,+query: U32,+steps: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+steps_fuel: {Nat.is_lt(steps,fuel) == True{} : Bool},
  +reach: {Nat.is_lt(Nat.add(U32.to_nat(index),steps),U32.to_nat(high)) == True{} : Bool},
  +small: {Nat.is_lt(Nat.add(U32.to_nat(index),steps),U32.to_nat(cap)) == True{} : Bool},
  +lands: {(target + (U32.from_nat(steps) * stride : U32) : U32) == query : U32}) ->
  {Block.writes_hit(fuel,U32.is_lt(index,high),index,high,target,stride,query) == True{} : Bool}:
  match fuel steps:
    case 0n +steps: Empty.absurd({Block.writes_hit(0n,U32.is_lt(index,high),index,high,target,stride,query) == True{} : Bool},Order.below_zero(steps,steps_fuel))
    case 1n+ +rest 0n:
      +i = U32.to_nat(index)
      +more = Equal.trans(Bool,U32.is_lt(index,high),Nat.is_lt(i,U32.to_nat(high)),True{},Comparison.lt(index,high),
        {%Addition.zero_right(i) : {Nat.is_lt(_,U32.to_nat(high)) == True{} : Bool} reach : {Nat.is_lt(i,U32.to_nat(high)) == True{} : Bool}})
      +same = Equal.trans(U32,target,(target + (0 * stride : U32) : U32),query,
        Equal.sym(U32,(target + (0 * stride : U32) : U32),target,
          Equal.trans(U32,(target + (0 * stride : U32) : U32),(target + 0 : U32),target,
            Equal.cong(U32,U32,word => (target + word : U32),(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)),Arithmetic.add_zero(target))),
        lands)
      +later = Block.writes_hit(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,query)
      %Equal.sym(Bool,U32.is_lt(index,high),True{},more) : {Block.writes_hit(1n+rest,_,index,high,target,stride,query) == True{} : Bool}
      %same : {U32.is_eq(target,_) || later == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(target,target),True{},Arithmetic.equal_reflexive(target)) : {_ || later == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +count:
      +i = U32.to_nat(index)
      +far = Nat.add(i,1n+count)
      +i_below = Order.le_lt_trans(i,far,U32.to_nat(high),Order.le_add(i,1n+count),reach)
      +more = Equal.trans(Bool,U32.is_lt(index,high),Nat.is_lt(i,U32.to_nat(high)),True{},Comparison.lt(index,high),i_below)
      +one_le = Order.offset_le(i,1n,1n+count,Order.zero_le(count))
      +next_word = Arithmetic.sum_exact(cap,cap_ok,index,1,i,1n,{==},{==},
        Order.transitive(Nat.add(i,1n),far,U32.to_nat(cap),one_le,Bounds.strict_le(far,U32.to_nat(cap),small)))
      +shift = Addition.associative(i,1n,count)
      +back = Equal.sym(Nat,U32.to_nat((index + 1 : U32)),Nat.add(i,1n),next_word)
      +later_reach = {%back : {Nat.is_lt(Nat.add(_,count),U32.to_nat(high)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(i,1n),count),far,shift) : {Nat.is_lt(_,U32.to_nat(high)) == True{} : Bool}
        reach : {Nat.is_lt(Nat.add(U32.to_nat((index + 1 : U32)),count),U32.to_nat(high)) == True{} : Bool}}
      +later_small = {%back : {Nat.is_lt(Nat.add(_,count),U32.to_nat(cap)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(i,1n),count),far,shift) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
        small : {Nat.is_lt(Nat.add(U32.to_nat((index + 1 : U32)),count),U32.to_nat(cap)) == True{} : Bool}}
      +later_lands = Equal.trans(U32,((target + stride : U32) + (U32.from_nat(count) * stride : U32) : U32),(target + (U32.from_nat(1n+count) * stride : U32) : U32),query,
        moved_lands(target,stride,count),lands)
      +later = Block.writes_hit(rest,U32.is_lt((index + 1 : U32),high),(index + 1 : U32),high,(target + stride : U32),stride,query)
      %Equal.sym(Bool,U32.is_lt(index,high),True{},more) : {Block.writes_hit(1n+rest,_,index,high,target,stride,query) == True{} : Bool}
      %Equal.sym(Bool,later,True{},writes_reach(rest,(index + 1 : U32),high,(target + stride : U32),stride,query,count,cap,cap_ok,steps_fuel,later_reach,later_small,later_lands)) :
        {U32.is_eq(target,query) || _ == True{} : Bool}
      Logic.or_true_right(U32.is_eq(target,query))

# The blocks from an index on hit a cell that the block a number of steps
# further hits.
def blocks_reach(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+query: U32,+steps: Nat,+steps_fuel: {Nat.is_lt(steps,fuel) == True{} : Bool},
  +hit: {Blocks.block_hit(Blocks.block_of(axes,(index + U32.from_nat(steps) : U32)),query) == True{} : Bool}) ->
  {Blocks.blocks_hit(fuel,index,axes,query) == True{} : Bool}:
  match fuel steps:
    case 0n +steps: Empty.absurd({Blocks.blocks_hit(0n,index,axes,query) == True{} : Bool},Order.below_zero(steps,steps_fuel))
    case 1n+ +rest 0n:
      +here = Equal.trans(Bool,Blocks.block_hit(Blocks.block_of(axes,index),query),Blocks.block_hit(Blocks.block_of(axes,(index + 0 : U32)),query),True{},
        Equal.cong(U32,Bool,word => Blocks.block_hit(Blocks.block_of(axes,word),query),index,(index + 0 : U32),Equal.sym(U32,(index + 0 : U32),index,Arithmetic.add_zero(index))),
        hit)
      %Equal.sym(Bool,Blocks.block_hit(Blocks.block_of(axes,index),query),True{},here) : {_ || Blocks.blocks_hit(rest,(index + 1 : U32),axes,query) == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +count:
      +moved = Equal.trans(U32,((index + 1 : U32) + U32.from_nat(count) : U32),(index + (1 + U32.from_nat(count) : U32) : U32),(index + U32.from_nat(1n+count) : U32),
        Arithmetic.add_associative(index,1,U32.from_nat(count)),
        Equal.cong(U32,U32,word => (index + word : U32),(1 + U32.from_nat(count) : U32),U32.from_nat(1n+count),
          Equal.trans(U32,(1 + U32.from_nat(count) : U32),(U32.from_nat(count) + 1 : U32),U32.from_nat(1n+count),Arithmetic.add_commutative(1,U32.from_nat(count)),
            Equal.sym(U32,U32.from_nat(1n+count),(U32.from_nat(count) + 1 : U32),successor_word(count)))))
      +later_hit = Equal.trans(Bool,Blocks.block_hit(Blocks.block_of(axes,((index + 1 : U32) + U32.from_nat(count) : U32)),query),
        Blocks.block_hit(Blocks.block_of(axes,(index + U32.from_nat(1n+count) : U32)),query),True{},
        Equal.cong(U32,Bool,word => Blocks.block_hit(Blocks.block_of(axes,word),query),((index + 1 : U32) + U32.from_nat(count) : U32),(index + U32.from_nat(1n+count) : U32),moved),
        hit)
      %Equal.sym(Bool,Blocks.blocks_hit(rest,(index + 1 : U32),axes,query),True{},blocks_reach(rest,(index + 1 : U32),axes,query,count,steps_fuel,later_hit)) :
        {Blocks.block_hit(Blocks.block_of(axes,index),query) || _ == True{} : Bool}
      Logic.or_true_right(Blocks.block_hit(Blocks.block_of(axes,index),query))

def block_fields(+block: Tiled.Block,+query: U32) ->
  {Blocks.block_hit(block,query) == Block.writes_hit(8n,U32.is_lt(Writes.low_of(block),Writes.high_of(block)),Writes.low_of(block),Writes.high_of(block),
    Writes.target_of(block),Writes.stride_of(block),query) : Bool}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}: {==}

# A block hits a cell that its write at a lane it writes lands on.
def block_cover(+axes: List<&2,Tiled.Ranged>,+index: U32,+lane: U32,+query: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(Tiled.block_at(axes,index,Writes.first_block()))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(Writes.high_of(Tiled.block_at(axes,index,Writes.first_block())))) == True{} : Bool},
  +small: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(cap)) == True{} : Bool},+eight: {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool},
  +address: {Writes.write_address(axes,index,lane) == query : U32}) -> {Blocks.block_hit(Blocks.block_of(axes,index),query) == True{} : Bool}:
  +block = Tiled.block_at(axes,index,Writes.first_block())
  +low = Writes.low_of(block)
  +high = Writes.high_of(block)
  +target = Writes.target_of(block)
  +stride = Writes.stride_of(block)
  +i = U32.to_nat(lane)
  +l = U32.to_nat(low)
  +steps = Nat.sub(i,l)
  +restored = Equal.sym(Nat,Nat.add(l,steps),i,Order.restore_shift(l,i,from_low))
  +difference = Equal.sym(U32,(U32.from_nat(i) - U32.from_nat(l) : U32),U32.from_nat(steps),Subtraction.encoded_subtract(i,l,from_low))
  +steps_word = {%Arithmetic.native_roundtrip(lane) : {U32.from_nat(steps) == (_ - low : U32) : U32}
    %Arithmetic.native_roundtrip(low) : {U32.from_nat(steps) == (U32.from_nat(i) - _ : U32) : U32}
    difference : {U32.from_nat(steps) == (lane - low : U32) : U32}}
  +lands = Equal.trans(U32,(target + (U32.from_nat(steps) * stride : U32) : U32),Writes.write_address(axes,index,lane),query,
    Equal.cong(U32,U32,word => (target + (word * stride : U32) : U32),U32.from_nat(steps),(lane - low : U32),steps_word),address)
  Equal.trans(Bool,Blocks.block_hit(Blocks.block_of(axes,index),query),Block.writes_hit(8n,U32.is_lt(low,high),low,high,target,stride,query),True{},
    block_fields(block,query),
    writes_reach(8n,low,high,target,stride,query,steps,cap,cap_ok,Order.le_lt_trans(steps,i,8n,Order.sub_le(i,l),eight),
      {%restored : {Nat.is_lt(_,U32.to_nat(high)) == True{} : Bool} to_high : {Nat.is_lt(Nat.add(l,steps),U32.to_nat(high)) == True{} : Bool}},
      {%restored : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool} small : {Nat.is_lt(Nat.add(l,steps),U32.to_nat(cap)) == True{} : Bool}},
      lands))

def facts_address(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: U32,+facts: Writes.WriteFacts(blocked,lows,boxes,index,lane)) ->
  {Writes.write_address(Writes.piece_axes(blocked,lows,boxes),index,lane) ==
    U32.from_nat(ShapeLaws.digits_value(boxes,Writes.box_digits(Writes.piece_axes(blocked,lows,boxes),index,U32.to_nat(lane)))) : U32}:
  Held.Both{+address,+rest} = facts
  address

# Every cell of the box of a piece is hit by the blocks of the piece.
def cover_hit(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,boxes) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +most: Geometry.LanesAtMost(blocked),+positive: BoxDigits.Positive(boxes),+rooms: BoxDigits.Rooms(BoxDigits.words_nat(lows),boxes,cap),
  +eight_room: {Nat.is_le(8n,U32.to_nat(cap)) == True{} : Bool},+boxes_room: {Nat.is_le(Nat.double(ShapeLaws.elements(boxes)),U32.to_nat(cap)) == True{} : Bool},
  +query: U32,+query_below: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(boxes)) == True{} : Bool}) ->
  {Blocks.blocks_hit(U32.to_nat(Tiled.ranged_head_count(Writes.piece_axes(blocked,lows,boxes))),0,Writes.piece_axes(blocked,lows,boxes),query) == True{} : Bool}:
  +axes = Writes.piece_axes(blocked,lows,boxes)
  +limit = U32.to_nat(cap)
  +extents_same = Writes.ranged_extents(blocked,lows,boxes,same_blocked,same_lows)
  +ok = Writes.ranged_ok(blocked,lows,boxes,cap,positive,rooms)
  +axes_most = {%Equal.sym(List<&2,Tiled.Blocked>,Geometry.blocked_of(axes),blocked,Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)) :
      Geometry.LanesAtMost(_)
    most : Geometry.LanesAtMost(Geometry.blocked_of(axes))}
  +q = U32.to_nat(query)
  +digits = Targets.cell_coordinates(boxes,q)
  +digits_value = BoxDigits.value_coordinates(boxes,q,query_below)
  +digits_below = {%Equal.sym(List<&2,U32>,Writes.extents_of(axes),boxes,extents_same) : Coordinates.DigitsBelow(_,digits)
    BoxDigits.coordinates_below(boxes,q,query_below) : Coordinates.DigitsBelow(Writes.extents_of(axes),digits)}
  +counts = counts_of(axes)
  +coords = cover_coords(axes,digits)
  +coords_below = cover_below(axes,digits,cap,cap_ok,ok,digits_below)
  +counts_room = Order.transitive(Nat.double(ShapeLaws.elements(counts)),Nat.double(ShapeLaws.elements(boxes)),limit,
    Order.double_le(ShapeLaws.elements(counts),ShapeLaws.elements(boxes),
      {%extents_same : {Nat.is_le(ShapeLaws.elements(counts),ShapeLaws.elements(_)) == True{} : Bool}
        counts_le(axes,cap,cap_ok,ok) : {Nat.is_le(ShapeLaws.elements(counts),ShapeLaws.elements(boxes)) == True{} : Bool}}),
    boxes_room)
  +counts_cap = Order.transitive(ShapeLaws.elements(counts),Nat.double(ShapeLaws.elements(counts)),limit,Order.le_double(ShapeLaws.elements(counts)),counts_room)
  +number = ShapeLaws.digits_value(counts,coords)
  +number_below = Coordinates.digits_below(counts,coords,coords_below)
  +index = U32.from_nat(number)
  +index_value = Bounds.roundtrip(number,cap,cap_ok,Order.transitive(number,ShapeLaws.elements(counts),limit,Bounds.strict_le(number,ShapeLaws.elements(counts),number_below),counts_cap))
  +indices = axes_index(axes,coords,index,cap,cap_ok,ranged_inners(blocked,lows,boxes,Tasks.strides_of(boxes)),counts_room,
    {%Equal.sym(Nat,U32.to_nat(index),number,index_value) : Decoded(counts,_,coords) decoded(counts,coords,0n,coords_below) : Decoded(counts,U32.to_nat(index),coords)},
    coords_below)
  +lane_number = cover_lane(axes,digits)
  +lane_small = cover_lane_small(axes,digits)
  +lane = U32.from_nat(lane_number)
  +lane_value = Bounds.roundtrip(lane_number,cap,cap_ok,Order.transitive(lane_number,8n,limit,Bounds.strict_le(lane_number,8n,lane_small),eight_room))
  +lane_back = Equal.sym(Nat,U32.to_nat(lane),lane_number,lane_value)
  +box_same = {%lane_back : {Writes.box_digits(axes,index,_) == digits : List<&2,Nat>}
    cover_box(axes,digits,index,lane_number,axes_most,indices,lane_at(axes,digits),digits_below) :
    {Writes.box_digits(axes,index,U32.to_nat(lane)) == digits : List<&2,Nat>}}
  +range = cover_range(axes,digits,Writes.first_block(),index,lane,cap,cap_ok,axes_most,ok,indices,digits_below,lane_value,{==},{==})
  +block = Tiled.block_at(axes,index,Writes.first_block())
  +from_low = range_from(block,lane,range)
  +to_high = range_to(block,lane,range)
  +written = facts_address(blocked,lows,boxes,index,lane,Writes.piece_write(blocked,lows,boxes,index,lane,cap,cap_ok,same_blocked,same_lows,most,positive,rooms,from_low,to_high))
  +address = Equal.trans(U32,Writes.write_address(axes,index,lane),U32.from_nat(ShapeLaws.digits_value(boxes,Writes.box_digits(axes,index,U32.to_nat(lane)))),query,written,
    Equal.trans(U32,U32.from_nat(ShapeLaws.digits_value(boxes,Writes.box_digits(axes,index,U32.to_nat(lane)))),U32.from_nat(ShapeLaws.digits_value(boxes,digits)),query,
      Equal.cong(List<&2,Nat>,U32,list => U32.from_nat(ShapeLaws.digits_value(boxes,list)),Writes.box_digits(axes,index,U32.to_nat(lane)),digits,box_same),
      Equal.trans(U32,U32.from_nat(ShapeLaws.digits_value(boxes,digits)),U32.from_nat(q),query,
        Equal.cong(Nat,U32,value => U32.from_nat(value),ShapeLaws.digits_value(boxes,digits),q,digits_value),Arithmetic.native_roundtrip(query))))
  +lane_cap = {%lane_back : {Nat.is_lt(_,limit) == True{} : Bool}
    BoxDigits.follow_strict(lane_number,limit,Order.transitive(1n+lane_number,8n,limit,Order.strict_follow(lane_number,8n,lane_small),eight_room)) :
    {Nat.is_lt(U32.to_nat(lane),limit) == True{} : Bool}}
  +hit = block_cover(axes,index,lane,query,cap,cap_ok,from_low,to_high,lane_cap,
    {%lane_back : {Nat.is_lt(_,8n) == True{} : Bool} lane_small : {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}},address)
  +head = Equal.trans(Nat,U32.to_nat(Tiled.ranged_head_count(axes)),U32.to_nat(Grid.product(counts)),ShapeLaws.elements(counts),
    Equal.cong(U32,Nat,word => U32.to_nat(word),Tiled.ranged_head_count(axes),Grid.product(counts),head_inner(axes,ranged_inners(blocked,lows,boxes,Tasks.strides_of(boxes)))),
    Public.product_exact(cap,cap_ok,counts,counts_cap))
  blocks_reach(U32.to_nat(Tiled.ranged_head_count(axes)),0,axes,query,number,
    {%Equal.sym(Nat,U32.to_nat(Tiled.ranged_head_count(axes)),ShapeLaws.elements(counts),head) : {Nat.is_lt(number,_) == True{} : Bool}
      number_below : {Nat.is_lt(number,U32.to_nat(Tiled.ranged_head_count(axes))) == True{} : Bool}},
    {%Equal.sym(U32,(0 + U32.from_nat(number) : U32),index,Arithmetic.add_zero_left(index)) : {Blocks.block_hit(Blocks.block_of(axes,_),query) == True{} : Bool}
      hit : {Blocks.block_hit(Blocks.block_of(axes,(0 + U32.from_nat(number) : U32)),query) == True{} : Bool}})
