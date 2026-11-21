# Tasks.components_of: the components of a moving operation read disjoint
# source pieces and together write every target piece.
import Base
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./u32_division.bend as U32Division
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Logic
import ./nat_ceiling.bend as NatCeiling
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ../tensor.bend as Tensors
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_grid_ranges.bend as GridRanges
import ./binary_radix.bend as Radix
import ./u32_bit_length.bend as Bits
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_pieces.bend as Pieces
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./storage_buffer_model.bend as Buffer

# ---- The classes of one target axis ----

# A footprint that reads blocks runs forward and ends below cap.
def Formed(found: Tasks.Footprint,+cap: U32) -> Data:
  match found:
    case Tasks.Footprint{+first,+last,True{}}:
      Unit
    case Tasks.Footprint{+first,+last,False{}}:
      Held.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},{Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool}>

# The source blocks of a class reach from the first of a footprint past its
# last, below cap.
def Covers(current: Tasks.Class,+first: U32,+last: U32,+cap: U32) -> Data:
  match current:
    case Tasks.Class{target_first,target_count,+source_first,+source_count}:
      Held.Both<{Nat.is_lt(0n,U32.to_nat(source_count)) == True{} : Bool},
        Held.Both<{Nat.is_le(U32.to_nat(source_first),U32.to_nat(first)) == True{} : Bool},
          Held.Both<{Nat.is_lt(U32.to_nat(last),Nat.add(U32.to_nat(source_first),U32.to_nat(source_count))) == True{} : Bool},
            {Nat.is_le(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),U32.to_nat(cap)) == True{} : Bool}>>>

def Hollow(current: Tasks.Class) -> Data:
  match current:
    case Tasks.Class{target_first,target_count,source_first,+source_count}:
      {source_count == 0 : U32}

# The current class of a walk holds no block after an empty footprint and
# covers a footprint that reads blocks.
def WalkHull(+current: Tasks.Class,previous: Tasks.Footprint,+cap: U32) -> Data:
  match previous:
    case Tasks.Footprint{+first,+last,True{}}:
      Hollow(current)
    case Tasks.Footprint{+first,+last,False{}}:
      Covers(current,first,last,cap)

# Two footprints share no source block.
def Apart(left: Tasks.Footprint,right: Tasks.Footprint) -> Data:
  match left right:
    case Tasks.Footprint{+left_first,+left_last,+left_empty} Tasks.Footprint{+right_first,+right_last,+right_empty}:
      Either<&2,&2,{left_empty == True{} : Bool},Either<&2,&2,{right_empty == True{} : Bool},
        Either<&2,&2,{Nat.is_lt(U32.to_nat(left_last),U32.to_nat(right_first)) == True{} : Bool},
          {Nat.is_lt(U32.to_nat(right_last),U32.to_nat(left_first)) == True{} : Bool}>>>

# A footprint from first to last ends one past last.
def span_end(+first: Nat,+last: Nat,+ordered: {Nat.is_le(first,last) == True{} : Bool}) -> {Nat.add(first,Nat.add(Nat.sub(last,first),1n)) == 1n+last : Nat}:
  Equal.trans(Nat,Nat.add(first,Nat.add(Nat.sub(last,first),1n)),Nat.add(Nat.add(first,Nat.sub(last,first)),1n),1n+last,
    Equal.sym(Nat,Nat.add(Nat.add(first,Nat.sub(last,first)),1n),Nat.add(first,Nat.add(Nat.sub(last,first),1n)),Addition.associative(first,Nat.sub(last,first),1n)),
    %Equal.sym(Nat,Nat.add(first,Nat.sub(last,first)),last,Order.restore_shift(first,last,ordered)) : {Nat.add(_,1n) == 1n+last : Nat}
    Addition.successor_width(last))

# The blocks from first to last, counted below cap.
def span_count(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+last: U32,
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+below: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((last - first + 1 : U32)) == Nat.add(Nat.sub(U32.to_nat(last),U32.to_nat(first)),1n) : Nat}:
  +low = U32.to_nat(first)
  +high = U32.to_nat(last)
  +room = Order.strict_follow(high,U32.to_nat(cap),below)
  Arithmetic.sum_exact(cap,cap_ok,(last - first : U32),1,Nat.sub(high,low),1n,
    Arithmetic.difference_exact(cap,cap_ok,last,first,high,low,{==},{==},ordered,Order.transitive(high,1n+high,U32.to_nat(cap),Order.successor(high,high,Order.reflexive(high)),room)),
    {==},
    Order.transitive(Nat.add(Nat.sub(high,low),1n),Nat.add(low,Nat.add(Nat.sub(high,low),1n)),U32.to_nat(cap),
      Order.right_le(low,Nat.add(Nat.sub(high,low),1n)),
      %Equal.sym(Nat,Nat.add(low,Nat.add(Nat.sub(high,low),1n)),1n+high,span_end(low,high,ordered)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
      room))

# A class started at a footprint covers it.
def started_hull(+index: U32,+found: Tasks.Footprint,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+formed: Formed(found,cap)) ->
  WalkHull(Tasks.started(index,found),found,cap):
  match found:
    case Tasks.Footprint{+first,+last,True{}}:
      {==}
    case Tasks.Footprint{+first,+last,False{}}:
      Held.Both{+ordered,+below} = formed
      +low = U32.to_nat(first)
      +high = U32.to_nat(last)
      +count = Nat.add(Nat.sub(high,low),1n)
      %Equal.sym(Nat,U32.to_nat((last - first + 1 : U32)),count,span_count(cap,cap_ok,first,last,ordered,below)) :
        Held.Both<{Nat.is_lt(0n,_) == True{} : Bool},
          Held.Both<{Nat.is_le(low,low) == True{} : Bool},
            Held.Both<{Nat.is_lt(high,Nat.add(low,_)) == True{} : Bool},{Nat.is_le(Nat.add(low,_),U32.to_nat(cap)) == True{} : Bool}>>>
      %Equal.sym(Nat,Nat.add(low,count),1n+high,span_end(low,high,ordered)) :
        Held.Both<{Nat.is_lt(0n,count) == True{} : Bool},
          Held.Both<{Nat.is_le(low,low) == True{} : Bool},
            Held.Both<{Nat.is_lt(high,_) == True{} : Bool},{Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}>>>
      %Equal.sym(Nat,Nat.add(Nat.sub(high,low),1n),1n+Nat.sub(high,low),Addition.successor_width(Nat.sub(high,low))) :
        Held.Both<{Nat.is_lt(0n,_) == True{} : Bool},
          Held.Both<{Nat.is_le(low,low) == True{} : Bool},
            Held.Both<{Nat.is_lt(high,1n+high) == True{} : Bool},{Nat.is_le(1n+high,U32.to_nat(cap)) == True{} : Bool}>>>
      Held.Both{{==},Held.Both{Order.reflexive(low),Held.Both{Order.lt_succ(high,high,Order.reflexive(high)),Order.strict_follow(high,U32.to_nat(cap),below)}}}

# ---- Order helpers ----

def pred_le(+a: Nat,+b: Nat,+h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(a,Nat.sub(b,1n)) == True{} : Bool}:
  match b:
    case 0n:
      Empty.absurd({Nat.is_le(a,Nat.sub(0n,1n)) == True{} : Bool},Order.below_zero(a,h))
    case 1n+ +p:
      %Equal.sym(Nat,Nat.sub(p,0n),p,Order.subtract_zero(p)) : {Nat.is_le(a,_) == True{} : Bool}
      Division.lt_succ_le(a,p,h)

def pred_lt(+b: Nat,+positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(b,1n),b) == True{} : Bool}:
  match b:
    case 0n:
      Empty.absurd({Nat.is_lt(Nat.sub(0n,1n),0n) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +p:
      %Equal.sym(Nat,Nat.sub(p,0n),p,Order.subtract_zero(p)) : {Nat.is_lt(_,1n+p) == True{} : Bool}
      Order.lt_succ(p,p,Order.reflexive(p))

def max_lt(+a: Nat,+b: Nat,+c: Nat,+a_below: {Nat.is_lt(a,c) == True{} : Bool},+b_below: {Nat.is_lt(b,c) == True{} : Bool}) -> {Nat.is_lt(Nat.max(a,b),c) == True{} : Bool}:
  Order.succ_le_lt(Nat.max(a,b),c,NatExtrema.maximum_upper(1n+a,1n+b,c,Order.strict_follow(a,c,a_below),Order.strict_follow(b,c,b_below)))

def max_above_left(+a: U32,+b: U32) -> {Nat.is_le(U32.to_nat(a),U32.to_nat(U32.max(a,b))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(a,b)),Nat.max(U32.to_nat(a),U32.to_nat(b)),Extrema.maximum(a,b)) : {Nat.is_le(U32.to_nat(a),_) == True{} : Bool}
  NatExtrema.maximum_left(U32.to_nat(a),U32.to_nat(b))

def max_below(+a: U32,+b: U32,+cap: U32,+a_below: {Nat.is_lt(U32.to_nat(a),U32.to_nat(cap)) == True{} : Bool},
  +b_below: {Nat.is_lt(U32.to_nat(b),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(U32.max(a,b)),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(a,b)),Nat.max(U32.to_nat(a),U32.to_nat(b)),Extrema.maximum(a,b)) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
  max_lt(U32.to_nat(a),U32.to_nat(b),U32.to_nat(cap),a_below,b_below)

# The last source block of a class.
def class_end(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+source_first: U32,+source_count: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(source_count)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((source_first + source_count - 1 : U32)) == Nat.sub(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),1n) : Nat}:
  +total = Nat.add(U32.to_nat(source_first),U32.to_nat(source_count))
  Arithmetic.difference_exact(cap,cap_ok,(source_first + source_count : U32),1,total,1n,
    Arithmetic.sum_exact(cap,cap_ok,source_first,source_count,U32.to_nat(source_first),U32.to_nat(source_count),{==},{==},room),{==},
    Order.strict_follow(0n,total,Order.lt_le(0n,U32.to_nat(source_count),total,positive,Order.right_le(U32.to_nat(source_first),U32.to_nat(source_count)))),room)

def end_below(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+source_first: U32,+source_count: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(source_count)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((source_first + source_count - 1 : U32)),U32.to_nat(cap)) == True{} : Bool}:
  +total = Nat.add(U32.to_nat(source_first),U32.to_nat(source_count))
  %Equal.sym(Nat,U32.to_nat((source_first + source_count - 1 : U32)),Nat.sub(total,1n),class_end(cap,cap_ok,source_first,source_count,positive,room)) :
    {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
  Order.lt_le(Nat.sub(total,1n),total,U32.to_nat(cap),pred_lt(total,Order.lt_le(0n,U32.to_nat(source_count),total,positive,Order.right_le(U32.to_nat(source_first),U32.to_nat(source_count)))),room)

# ---- One step of the walk ----

def CurrentRoom(current: Tasks.Class,+cap: U32) -> Data:
  match current:
    case Tasks.Class{target_first,target_count,+source_first,+source_count}:
      Held.Both<{Nat.is_lt(0n,U32.to_nat(source_count)) == True{} : Bool},{Nat.is_le(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),U32.to_nat(cap)) == True{} : Bool}>

# Joining a footprint to a class covers it.
def joined_hull(+first: U32,+last: U32,+current: Tasks.Class,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+below: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +room: CurrentRoom(current,cap)) -> Covers(Tasks.joined(Tasks.Footprint{first,last,False{}},current),first,last,cap):
  match current:
    case Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      Held.Both{+positive,+fits} = room
      +end = (source_first + source_count - 1 : U32)
      +low = U32.min(first,source_first)
      +high = U32.max(last,end)
      +low_value = U32.to_nat(low)
      +high_value = U32.to_nat(high)
      +high_below = max_below(last,end,cap,below,end_below(cap,cap_ok,source_first,source_count,positive,fits))
      +ordered_ends = Order.transitive(low_value,U32.to_nat(first),high_value,Extrema.minimum_below(first,source_first),
        Order.transitive(U32.to_nat(first),U32.to_nat(last),high_value,ordered,max_above_left(last,end)))
      +count = Nat.add(Nat.sub(high_value,low_value),1n)
      %Equal.sym(Nat,U32.to_nat((high - low + 1 : U32)),count,span_count(cap,cap_ok,low,high,ordered_ends,high_below)) :
        Held.Both<{Nat.is_lt(0n,_) == True{} : Bool},
          Held.Both<{Nat.is_le(low_value,U32.to_nat(first)) == True{} : Bool},
            Held.Both<{Nat.is_lt(U32.to_nat(last),Nat.add(low_value,_)) == True{} : Bool},{Nat.is_le(Nat.add(low_value,_),U32.to_nat(cap)) == True{} : Bool}>>>
      %Equal.sym(Nat,Nat.add(low_value,count),1n+high_value,span_end(low_value,high_value,ordered_ends)) :
        Held.Both<{Nat.is_lt(0n,count) == True{} : Bool},
          Held.Both<{Nat.is_le(low_value,U32.to_nat(first)) == True{} : Bool},
            Held.Both<{Nat.is_lt(U32.to_nat(last),_) == True{} : Bool},{Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}>>>
      %Equal.sym(Nat,count,1n+Nat.sub(high_value,low_value),Addition.successor_width(Nat.sub(high_value,low_value))) :
        Held.Both<{Nat.is_lt(0n,_) == True{} : Bool},
          Held.Both<{Nat.is_le(low_value,U32.to_nat(first)) == True{} : Bool},
            Held.Both<{Nat.is_lt(U32.to_nat(last),1n+high_value) == True{} : Bool},{Nat.is_le(1n+high_value,U32.to_nat(cap)) == True{} : Bool}>>>
      Held.Both{{==},Held.Both{Extrema.minimum_below(first,source_first),
        Held.Both{Order.lt_succ(U32.to_nat(last),high_value,max_above_left(last,end)),Order.strict_follow(high_value,U32.to_nat(cap),high_below)}}}

# The overlap test fails at one of its three conditions.
def gap(+p: Bool,+q: Bool,+r: Bool,+h: {True{} && p && q && r == False{} : Bool}) ->
  Either<&2,&2,{p == False{} : Bool},Either<&2,&2,{q == False{} : Bool},{r == False{} : Bool}>>:
  match p q r:
    case False{} +q +r:
      Inl{{==}}
    case True{} False{} +r:
      Inr{Inl{{==}}}
    case True{} True{} False{}:
      Inr{Inr{{==}}}
    case True{} True{} True{}:
      Empty.absurd(Either<&2,&2,{True{} == False{} : Bool},Either<&2,&2,{True{} == False{} : Bool},{True{} == False{} : Bool}>>,
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))

def gap_apart(+previous_first: U32,+previous_last: U32,+first: U32,+last: U32,+source_first: U32,+source_count: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +covers: Covers(Tasks.Class{0,0,source_first,source_count},previous_first,previous_last,cap),
  +which: Either<&2,&2,{U32.is_lt(0,source_count) == False{} : Bool},Either<&2,&2,{U32.is_le(first,(source_first + source_count - 1 : U32)) == False{} : Bool},
    {U32.is_le(source_first,last) == False{} : Bool}>>) ->
  Apart(Tasks.Footprint{previous_first,previous_last,False{}},Tasks.Footprint{first,last,False{}}):
  Held.Both{+positive,+rest} = covers
  Held.Both{+starts,+more} = rest
  Held.Both{+ends,+fits} = more
  match which:
    case Inl{+zero}:
      Empty.absurd(Apart(Tasks.Footprint{previous_first,previous_last,False{}},Tasks.Footprint{first,last,False{}}),
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_lt(0,source_count),True{},Equal.sym(Bool,U32.is_lt(0,source_count),False{},zero),
          Equal.trans(Bool,U32.is_lt(0,source_count),Nat.is_lt(U32.to_nat(0),U32.to_nat(source_count)),True{},Comparison.lt(0,source_count),positive))))
    case Inr{+later}:
      match later:
        case Inl{+after}:
          +end = (source_first + source_count - 1 : U32)
          +end_value = Nat.sub(Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),1n)
          +beyond = Held.le_false_lt(U32.to_nat(first),U32.to_nat(end),
            Equal.trans(Bool,Nat.is_le(U32.to_nat(first),U32.to_nat(end)),U32.is_le(first,end),False{},Equal.sym(Bool,U32.is_le(first,end),Nat.is_le(U32.to_nat(first),U32.to_nat(end)),Comparison.le(first,end)),after))
          +reach = {%Equal.sym(Nat,U32.to_nat(end),end_value,class_end(cap,cap_ok,source_first,source_count,positive,fits)) :
              {Nat.is_le(U32.to_nat(previous_last),_) == True{} : Bool}
            pred_le(U32.to_nat(previous_last),Nat.add(U32.to_nat(source_first),U32.to_nat(source_count)),ends) :
            {Nat.is_le(U32.to_nat(previous_last),U32.to_nat(end)) == True{} : Bool}}
          Inr{Inr{Inl{Order.le_lt_trans(U32.to_nat(previous_last),U32.to_nat(end),U32.to_nat(first),reach,beyond)}}}
        case Inr{+before}:
          +short = Held.le_false_lt(U32.to_nat(source_first),U32.to_nat(last),
            Equal.trans(Bool,Nat.is_le(U32.to_nat(source_first),U32.to_nat(last)),U32.is_le(source_first,last),False{},
              Equal.sym(Bool,U32.is_le(source_first,last),Nat.is_le(U32.to_nat(source_first),U32.to_nat(last)),Comparison.le(source_first,last)),before))
          Inr{Inr{Inr{Order.lt_le(U32.to_nat(last),U32.to_nat(source_first),U32.to_nat(previous_first),short,starts)}}}

# ---- The walk over the blocks of a target axis ----

# A class that starts past the first block reads blocks apart from the block
# before it.
def ClassApart(+read: Tasks.Source,+extent: U32,+block: U32,class: Tasks.Class) -> Data:
  match class:
    case Tasks.Class{+target_first,target_count,source_first,source_count}:
      Either<&2,&2,{target_first == 0 : U32},
        Apart(Tasks.block_footprint(read,extent,block,(target_first - 1 : U32)),Tasks.block_footprint(read,extent,block,target_first))>

def StartsApart(+read: Tasks.Source,+extent: U32,+block: U32,classes: List<&2,Tasks.Class>) -> Data:
  match classes:
    case Nil{}:
      Unit
    case Con{class,rest}:
      Held.Both<ClassApart(read,extent,block,class),StartsApart(read,extent,block,rest)>

def WalkOk(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,walk: Tasks.Walk,+previous: Tasks.Footprint) -> Data:
  match walk:
    case Tasks.Walk{+current,done}:
      Held.Both<WalkHull(current,previous,cap),StartsApart(read,extent,block,Con{current,done})>

def joined_walk(+found: Tasks.Footprint,+previous: Tasks.Footprint,+current: Tasks.Class,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +formed: Formed(found,cap),+hull: WalkHull(current,previous,cap),+overlap: {Tasks.overlapping(found,current) == True{} : Bool}) ->
  WalkHull(Tasks.joined(found,current),found,cap):
  match found previous current:
    case Tasks.Footprint{+first,+last,True{}} +previous Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      Empty.absurd(WalkHull(Tasks.joined(Tasks.Footprint{first,last,True{}},Tasks.Class{target_first,target_count,source_first,source_count}),
        Tasks.Footprint{first,last,True{}},cap),WordEncoding.false_true(overlap))
    case Tasks.Footprint{+first,+last,False{}} Tasks.Footprint{+previous_first,+previous_last,True{}} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      +hollow = {%hull : {True{} && U32.is_lt(0,_) && U32.is_le(first,(source_first + _ - 1 : U32)) && U32.is_le(source_first,last) == True{} : Bool}
        overlap : {True{} && U32.is_lt(0,0) && U32.is_le(first,(source_first + 0 - 1 : U32)) && U32.is_le(source_first,last) == True{} : Bool}}
      Empty.absurd(Covers(Tasks.joined(Tasks.Footprint{first,last,False{}},Tasks.Class{target_first,target_count,source_first,source_count}),first,last,cap),
        WordEncoding.false_true(hollow))
    case Tasks.Footprint{+first,+last,False{}} Tasks.Footprint{+previous_first,+previous_last,False{}} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      Held.Both{+ordered,+below} = formed
      Held.Both{+positive,+rest} = hull
      Held.Both{+starts,+more} = rest
      Held.Both{+ends,+fits} = more
      joined_hull(first,last,Tasks.Class{target_first,target_count,source_first,source_count},cap,cap_ok,ordered,below,Held.Both{positive,fits})

def apart_walk(+found: Tasks.Footprint,+previous: Tasks.Footprint,+current: Tasks.Class,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +hull: WalkHull(current,previous,cap),+missed: {Tasks.overlapping(found,current) == False{} : Bool}) -> Apart(previous,found):
  match found previous current:
    case Tasks.Footprint{+first,+last,True{}} Tasks.Footprint{+previous_first,+previous_last,+previous_empty} +current:
      Inr{Inl{{==}}}
    case Tasks.Footprint{+first,+last,False{}} Tasks.Footprint{+previous_first,+previous_last,True{}} +current:
      Inl{{==}}
    case Tasks.Footprint{+first,+last,False{}} Tasks.Footprint{+previous_first,+previous_last,False{}} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      gap_apart(previous_first,previous_last,first,last,source_first,source_count,cap,cap_ok,hull,
        gap(U32.is_lt(0,source_count),U32.is_le(first,(source_first + source_count - 1 : U32)),U32.is_le(source_first,last),missed))

def joined_starts(+read: Tasks.Source,+extent: U32,+block: U32,+found: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,
  +starts: StartsApart(read,extent,block,Con{current,done})) -> StartsApart(read,extent,block,Con{Tasks.joined(found,current),done}):
  match found current:
    case Tasks.Footprint{+first,+last,+empty} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      starts

def started_starts(+read: Tasks.Source,+extent: U32,+block: U32,+index: U32,+found: Tasks.Footprint,+rest: List<&2,Tasks.Class>,
  +apart: Either<&2,&2,{index == 0 : U32},Apart(Tasks.block_footprint(read,extent,block,(index - 1 : U32)),Tasks.block_footprint(read,extent,block,index))>,
  +starts: StartsApart(read,extent,block,rest)) -> StartsApart(read,extent,block,Con{Tasks.started(index,found),rest}):
  match found:
    case Tasks.Footprint{+first,+last,+empty}:
      Held.Both{apart,starts}

def walk_joined(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,
  +previous: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,+formed: Formed(Tasks.block_footprint(read,extent,block,index),cap),
  +hull: WalkHull(current,previous,cap),+starts: StartsApart(read,extent,block,Con{current,done}),
  +overlap: {Tasks.overlapping(Tasks.block_footprint(read,extent,block,index),current) == True{} : Bool}) ->
  WalkOk(read,extent,block,cap,Tasks.stepped(Tasks.overlapping(Tasks.block_footprint(read,extent,block,index),current),index,
    Tasks.block_footprint(read,extent,block,index),current,done),Tasks.block_footprint(read,extent,block,index)):
  +found = Tasks.block_footprint(read,extent,block,index)
  %Equal.sym(Bool,Tasks.overlapping(found,current),True{},overlap) : WalkOk(read,extent,block,cap,Tasks.stepped(_,index,found,current,done),found)
  Held.Both{joined_walk(found,previous,current,cap,cap_ok,formed,hull,overlap),joined_starts(read,extent,block,found,current,done,starts)}

def walk_started(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,
  +current: Tasks.Class,+done: List<&2,Tasks.Class>,+formed: Formed(Tasks.block_footprint(read,extent,block,index),cap),
  +hull: WalkHull(current,Tasks.block_footprint(read,extent,block,(index - 1 : U32)),cap),+starts: StartsApart(read,extent,block,Con{current,done}),
  +missed: {Tasks.overlapping(Tasks.block_footprint(read,extent,block,index),current) == False{} : Bool}) ->
  WalkOk(read,extent,block,cap,Tasks.stepped(Tasks.overlapping(Tasks.block_footprint(read,extent,block,index),current),index,
    Tasks.block_footprint(read,extent,block,index),current,done),Tasks.block_footprint(read,extent,block,index)):
  +found = Tasks.block_footprint(read,extent,block,index)
  %Equal.sym(Bool,Tasks.overlapping(found,current),False{},missed) : WalkOk(read,extent,block,cap,Tasks.stepped(_,index,found,current,done),found)
  Held.Both{started_hull(index,found,cap,cap_ok,formed),
    started_starts(read,extent,block,index,found,Con{current,done},
      Inr{apart_walk(found,Tasks.block_footprint(read,extent,block,(index - 1 : U32)),current,cap,cap_ok,hull,missed)},starts)}

def walk_next(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,+walk: Tasks.Walk,
  +ok: WalkOk(read,extent,block,cap,walk,Tasks.block_footprint(read,extent,block,(index - 1 : U32))),
  +formed: Formed(Tasks.block_footprint(read,extent,block,index),cap)) ->
  WalkOk(read,extent,block,cap,Tasks.walk_step(index,Tasks.block_footprint(read,extent,block,index),walk),Tasks.block_footprint(read,extent,block,index)):
  match walk:
    case Tasks.Walk{+current,+done}:
      Held.Both{+hull,+starts} = ok
      +found = Tasks.block_footprint(read,extent,block,index)
      Held.by_cases(WalkOk(read,extent,block,cap,Tasks.stepped(Tasks.overlapping(found,current),index,found,current,done),found),Tasks.overlapping(found,current),
        overlap => walk_joined(read,extent,block,cap,cap_ok,index,Tasks.block_footprint(read,extent,block,(index - 1 : U32)),current,done,formed,hull,starts,overlap),
        missed => walk_started(read,extent,block,cap,cap_ok,index,current,done,formed,hull,starts,missed))

def reversed_apart(+read: Tasks.Source,+extent: U32,+block: U32,classes: List<&2,Tasks.Class>,+done: List<&2,Tasks.Class>,
  +all: StartsApart(read,extent,block,classes),+all_done: StartsApart(read,extent,block,done)) ->
  StartsApart(read,extent,block,Tasks.reversed_classes(classes,done)):
  match classes:
    case Nil{}:
      all_done
    case Con{+class,+rest}:
      Held.Both{+here,+later} = all
      reversed_apart(read,extent,block,rest,Con{class,done},later,Held.Both{here,all_done})

# Every footprint of the blocks from index on is formed.
def FormedFrom(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+index: U32,remaining: Nat) -> Data:
  match remaining:
    case 0n:
      Unit
    case 1n+rest:
      Held.Both<Formed(Tasks.block_footprint(read,extent,block,index),cap),FormedFrom(read,extent,block,cap,(index + 1 : U32),rest)>

def index_back(+index: U32) -> {((index + 1 : U32) - 1 : U32) == index : U32}:
  Equal.trans(U32,((index + 1 : U32) - 1 : U32),(index + (1 - 1 : U32) : U32),index,Arithmetic.subtract_from_sum(index,1,1),Arithmetic.add_zero(index))

def walked_apart(remaining: Nat,+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,
  +walk: Tasks.Walk,+ok: WalkOk(read,extent,block,cap,walk,Tasks.block_footprint(read,extent,block,(index - 1 : U32))),
  +formed: FormedFrom(read,extent,block,cap,index,remaining)) -> StartsApart(read,extent,block,Tasks.walked(remaining,read,extent,block,index,walk)):
  match remaining:
    case 0n:
      match walk:
        case Tasks.Walk{+current,+done}:
          Held.Both{+hull,+starts} = ok
          reversed_apart(read,extent,block,Con{current,done},Nil{},starts,Unit{})
    case 1n+ +rest:
      Held.Both{+here,+later} = formed
      +next = (index + 1 : U32)
      +stepped = Tasks.walk_step(index,Tasks.block_footprint(read,extent,block,index),walk)
      walked_apart(rest,read,extent,block,cap,cap_ok,next,stepped,
        {%Equal.sym(U32,((index + 1 : U32) - 1 : U32),index,index_back(index)) :
            WalkOk(read,extent,block,cap,stepped,Tasks.block_footprint(read,extent,block,_))
          walk_next(read,extent,block,cap,cap_ok,index,walk,ok,here) :
          WalkOk(read,extent,block,cap,stepped,Tasks.block_footprint(read,extent,block,((index + 1 : U32) - 1 : U32)))},
        later)

def classes_none(+read: Tasks.Source,+extent: U32,+block: U32,+none: {U32.is_zero(Tasks.target_blocks(extent,block)) == True{} : Bool}) ->
  StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)):
  %Equal.sym(Bool,U32.is_zero(Tasks.target_blocks(extent,block)),True{},none) :
    StartsApart(read,extent,block,Tasks.read_classes_when(_,read,extent,block))
  Unit{}

def classes_some(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +first: Formed(Tasks.block_footprint(read,extent,block,0),cap),
  +formed: FormedFrom(read,extent,block,cap,1,U32.to_nat((Tasks.target_blocks(extent,block) - 1 : U32))),
  +some: {U32.is_zero(Tasks.target_blocks(extent,block)) == False{} : Bool}) ->
  StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)):
  +start = {Tasks.Walk{Tasks.started(0,Tasks.block_footprint(read,extent,block,0)),[]} : Tasks.Walk}
  %Equal.sym(Bool,U32.is_zero(Tasks.target_blocks(extent,block)),False{},some) :
    StartsApart(read,extent,block,Tasks.read_classes_when(_,read,extent,block))
  walked_apart(U32.to_nat((Tasks.target_blocks(extent,block) - 1 : U32)),read,extent,block,cap,cap_ok,1,start,
    Held.Both{started_hull(0,Tasks.block_footprint(read,extent,block,0),cap,cap_ok,first),
      started_starts(read,extent,block,0,Tasks.block_footprint(read,extent,block,0),Nil{},Inl{{==}},Unit{})},
    formed)

def lt_to_le(+a: Nat,+b: Nat,+h: {Nat.is_lt(a,b) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  Order.transitive(a,1n+a,b,Order.successor(a,a,Order.reflexive(a)),Order.strict_follow(a,b,h))

def positive_max(+value: U32) -> {Nat.is_lt(0n,U32.to_nat(U32.max(value,1))) == True{} : Bool}:
  Order.succ_le_lt(0n,U32.to_nat(U32.max(value,1)),
    %Equal.sym(Nat,U32.to_nat(U32.max(value,1)),Nat.max(U32.to_nat(value),1n),Extrema.maximum(value,1)) : {Nat.is_le(1n,_) == True{} : Bool}
      NatExtrema.maximum_right(U32.to_nat(value),1n))

# A value from shift on, in blocks past shift.
def scaled(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+shift: U32,+source_block: U32,
  +above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(value)) == True{} : Bool},+below: {Nat.is_le(U32.to_nat(value),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(((value - shift) / U32.max(source_block,1) : U32)) == Nat.div(Nat.sub(U32.to_nat(value),U32.to_nat(shift)),U32.to_nat(U32.max(source_block,1))) : Nat}:
  U32Division.quotient_exact(cap,cap_ok,(value - shift : U32),U32.max(source_block,1),Nat.sub(U32.to_nat(value),U32.to_nat(shift)),U32.to_nat(U32.max(source_block,1)),
    Arithmetic.difference_exact(cap,cap_ok,value,shift,U32.to_nat(value),U32.to_nat(shift),{==},{==},above,below),{==},block_room,positive_max(source_block))

# Values from shift to the top of bound cells, in blocks past shift, are
# formed below cap.
def scaled_formed(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+low: U32,+high: U32,+shift: U32,+bound: U32,+source_block: U32,
  +above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(low)) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(high),Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(U32.to_nat(((low - shift) / U32.max(source_block,1) : U32)),U32.to_nat(((high - shift) / U32.max(source_block,1) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),U32.to_nat(cap)) == True{} : Bool}>:
  +blocks = U32.to_nat(U32.max(source_block,1))
  +start = U32.to_nat(shift)
  +low_value = U32.to_nat(low)
  +high_value = U32.to_nat(high)
  +total = Nat.add(start,U32.to_nat(bound))
  +last = Nat.sub(total,1n)
  +last_below = Order.lt_le(last,total,U32.to_nat(cap),pred_lt(total,Order.lt_le(0n,U32.to_nat(bound),total,positive,Order.right_le(start,U32.to_nat(bound)))),fits)
  +high_below = Order.le_lt_trans(high_value,last,U32.to_nat(cap),inside,last_below)
  +high_room = lt_to_le(high_value,U32.to_nat(cap),high_below)
  +low_quotient = Nat.div(Nat.sub(low_value,start),blocks)
  +high_quotient = Nat.div(Nat.sub(high_value,start),blocks)
  +block_positive = positive_max(source_block)
  %Equal.sym(Nat,U32.to_nat(((low - shift) / U32.max(source_block,1) : U32)),low_quotient,
      scaled(cap,cap_ok,low,shift,source_block,above,Order.transitive(low_value,high_value,U32.to_nat(cap),ordered,high_room),block_room)) :
    Held.Both<{Nat.is_le(_,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32))) == True{} : Bool},
      {Nat.is_lt(U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),U32.to_nat(cap)) == True{} : Bool}>
  %Equal.sym(Nat,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),high_quotient,
      scaled(cap,cap_ok,high,shift,source_block,Order.transitive(start,low_value,high_value,above,ordered),high_room,block_room)) :
    Held.Both<{Nat.is_le(low_quotient,_) == True{} : Bool},{Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}>
  Held.Both{DivisionOrder.quotient_le(Nat.sub(low_value,start),Nat.sub(high_value,start),blocks,block_positive,DivisionOrder.subtract_le(low_value,high_value,start,ordered)),
    Order.le_lt_trans(high_quotient,high_value,U32.to_nat(cap),
      Order.transitive(high_quotient,Nat.sub(high_value,start),high_value,DivisionOrder.quotient_at_most(Nat.sub(high_value,start),blocks,block_positive),Order.sub_le(high_value,start)),
      high_below)}

def below_plus(+a: Nat,+b: Nat,+positive: {Nat.is_lt(0n,b) == True{} : Bool}) -> {Nat.is_lt(a,Nat.add(a,b)) == True{} : Bool}:
  %Addition.zero_right(a) : {Nat.is_lt(_,Nat.add(a,b)) == True{} : Bool}
  Order.offset_lt(a,0n,b,positive)

def or3(+a: Bool,+b: Bool,+c: Bool,+h: {a || b || c == False{} : Bool}) ->
  Held.Both<{a == False{} : Bool},Held.Both<{b == False{} : Bool},{c == False{} : Bool}>>:
  match a b c:
    case True{} +b +c:
      Empty.absurd(Held.Both<{True{} == False{} : Bool},Held.Both<{b == False{} : Bool},{c == False{} : Bool}>>,WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case False{} True{} +c:
      Empty.absurd(Held.Both<{False{} == False{} : Bool},Held.Both<{True{} == False{} : Bool},{c == False{} : Bool}>>,WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case False{} False{} True{}:
      Empty.absurd(Held.Both<{False{} == False{} : Bool},Held.Both<{False{} == False{} : Bool},{True{} == False{} : Bool}>>,WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case False{} False{} False{}:
      Held.Both{{==},Held.Both{{==},{==}}}

def nat_lt_false(+a: U32,+b: U32,+h: {U32.is_lt(a,b) == False{} : Bool}) -> {Nat.is_le(U32.to_nat(b),U32.to_nat(a)) == True{} : Bool}:
  Held.lt_false_le(U32.to_nat(a),U32.to_nat(b),Equal.trans(Bool,Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),U32.is_lt(a,b),False{},Equal.sym(Bool,U32.is_lt(a,b),Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),Comparison.lt(a,b)),h))

def max_value(+a: U32,+b: U32) -> {U32.to_nat(U32.max(a,b)) == Nat.max(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  Extrema.maximum(a,b)

def clamp_value(+value: U32,+low: U32,+high: U32) ->
  {U32.to_nat(Tasks.clamp(value,low,high)) == Nat.min(Nat.max(U32.to_nat(value),U32.to_nat(low)),U32.to_nat(high)) : Nat}:
  %Extrema.maximum(value,low) : {U32.to_nat(U32.min(U32.max(value,low),high)) == Nat.min(_,U32.to_nat(high)) : Nat}
  Extrema.minimum(U32.max(value,low),high)

# The top of bound cells from shift.
def top_value(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift: U32,+bound: U32,+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((shift + bound - 1 : U32)) == Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n) : Nat}:
  class_end(cap,cap_ok,shift,bound,positive,fits)

def shift_below_top(+shift: Nat,+bound: Nat,+positive: {Nat.is_lt(0n,bound) == True{} : Bool}) -> {Nat.is_le(shift,Nat.sub(Nat.add(shift,bound),1n)) == True{} : Bool}:
  pred_le(shift,Nat.add(shift,bound),below_plus(shift,bound,positive))

def clamped_formed(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +nonzero: {U32.is_zero(bound) == False{} : Bool}) ->
  Held.Both<{Nat.is_le(U32.to_nat(((Tasks.clamp(first,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)),
      U32.to_nat(((Tasks.clamp(last,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(((Tasks.clamp(last,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)),U32.to_nat(cap)) == True{} : Bool}>:
  +top = (shift + bound - 1 : U32)
  +start = U32.to_nat(shift)
  +positive = Arithmetic.word_nonzero_positive(bound,nonzero)
  +top_nat = Nat.sub(Nat.add(start,U32.to_nat(bound)),1n)
  +first_max = Nat.max(U32.to_nat(first),start)
  +last_max = Nat.max(U32.to_nat(last),start)
  +low = Tasks.clamp(first,shift,top)
  +high = Tasks.clamp(last,shift,top)
  +maxes = NatExtrema.maximum_upper(U32.to_nat(first),start,last_max,
    Order.transitive(U32.to_nat(first),U32.to_nat(last),last_max,ordered,NatExtrema.maximum_left(U32.to_nat(last),start)),NatExtrema.maximum_right(U32.to_nat(last),start))
  scaled_formed(cap,cap_ok,low,high,shift,bound,source_block,
    {%Equal.sym(Nat,U32.to_nat(low),Nat.min(first_max,U32.to_nat(top)),clamp_value(first,shift,top)) : {Nat.is_le(start,_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(start,Nat.min(first_max,_)) == True{} : Bool}
      NatExtrema.minimum_lower(first_max,top_nat,start,NatExtrema.maximum_right(U32.to_nat(first),start),shift_below_top(start,U32.to_nat(bound),positive)) :
      {Nat.is_le(start,U32.to_nat(low)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(low),Nat.min(first_max,U32.to_nat(top)),clamp_value(first,shift,top)) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(high),Nat.min(last_max,U32.to_nat(top)),clamp_value(last,shift,top)) : {Nat.is_le(Nat.min(first_max,U32.to_nat(top)),_) == True{} : Bool}
      NatExtrema.minimum_lower(last_max,U32.to_nat(top),Nat.min(first_max,U32.to_nat(top)),
        Order.transitive(Nat.min(first_max,U32.to_nat(top)),first_max,last_max,NatExtrema.minimum_left(first_max,U32.to_nat(top)),maxes),
        NatExtrema.minimum_right(first_max,U32.to_nat(top))) :
      {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(high),Nat.min(last_max,U32.to_nat(top)),clamp_value(last,shift,top)) : {Nat.is_le(_,top_nat) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(Nat.min(last_max,_),top_nat) == True{} : Bool}
      NatExtrema.minimum_right(last_max,top_nat) :
      {Nat.is_le(U32.to_nat(high),top_nat) == True{} : Bool}},
    positive,fits,block_room)

def clipped_formed(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +parts: Held.Both<{U32.is_zero(bound) == False{} : Bool},Held.Both<{U32.is_lt(last,shift) == False{} : Bool},{U32.is_lt((shift + bound - 1 : U32),first) == False{} : Bool}>>) ->
  Held.Both<{Nat.is_le(U32.to_nat(((U32.max(first,shift) - shift) / U32.max(source_block,1) : U32)),
      U32.to_nat(((U32.min(last,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(((U32.min(last,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)),U32.to_nat(cap)) == True{} : Bool}>:
  Held.Both{+nonzero,+rest} = parts
  Held.Both{+reaches,+starts} = rest
  +top = (shift + bound - 1 : U32)
  +start = U32.to_nat(shift)
  +positive = Arithmetic.word_nonzero_positive(bound,nonzero)
  +top_nat = Nat.sub(Nat.add(start,U32.to_nat(bound)),1n)
  +first_value = U32.to_nat(first)
  +last_value = U32.to_nat(last)
  +shift_last = nat_lt_false(last,shift,reaches)
  +first_top = nat_lt_false(top,first,starts)
  +shift_top = {%Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(start,_) == True{} : Bool}
    shift_below_top(start,U32.to_nat(bound),positive) : {Nat.is_le(start,U32.to_nat(top)) == True{} : Bool}}
  +low = U32.max(first,shift)
  +high = U32.min(last,top)
  scaled_formed(cap,cap_ok,low,high,shift,bound,source_block,
    {%Equal.sym(Nat,U32.to_nat(low),Nat.max(first_value,start),max_value(first,shift)) : {Nat.is_le(start,_) == True{} : Bool}
      NatExtrema.maximum_right(first_value,start) : {Nat.is_le(start,U32.to_nat(low)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(low),Nat.max(first_value,start),max_value(first,shift)) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(high),Nat.min(last_value,U32.to_nat(top)),Extrema.minimum(last,top)) : {Nat.is_le(Nat.max(first_value,start),_) == True{} : Bool}
      NatExtrema.maximum_upper(first_value,start,Nat.min(last_value,U32.to_nat(top)),
        NatExtrema.minimum_lower(last_value,U32.to_nat(top),first_value,ordered,first_top),
        NatExtrema.minimum_lower(last_value,U32.to_nat(top),start,shift_last,shift_top)) :
      {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(high),Nat.min(last_value,U32.to_nat(top)),Extrema.minimum(last,top)) : {Nat.is_le(_,top_nat) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(Nat.min(last_value,_),top_nat) == True{} : Bool}
      NatExtrema.minimum_right(last_value,top_nat) :
      {Nat.is_le(U32.to_nat(high),top_nat) == True{} : Bool}},
    positive,fits,block_room)

def clamped_some(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(bound) == False{} : Bool}) ->
  Formed(Tasks.Footprint{((Tasks.clamp(first,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),
    ((Tasks.clamp(last,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),U32.is_zero(bound)},cap):
  +low = ((Tasks.clamp(first,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)
  +high = ((Tasks.clamp(last,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)
  %Equal.sym(Bool,U32.is_zero(bound),False{},some) : Formed(Tasks.Footprint{low,high,_},cap)
  clamped_formed(first,last,shift,bound,source_block,cap,cap_ok,ordered,fits,block_room,some)

def clipped_some(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt((shift + bound - 1 : U32),first) == False{} : Bool}) ->
  Formed(Tasks.Footprint{((U32.max(first,shift) - shift) / U32.max(source_block,1) : U32),
    ((U32.min(last,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),
    U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt((shift + bound - 1 : U32),first)},cap):
  +top = (shift + bound - 1 : U32)
  +low = ((U32.max(first,shift) - shift) / U32.max(source_block,1) : U32)
  +high = ((U32.min(last,top) - shift) / U32.max(source_block,1) : U32)
  %Equal.sym(Bool,U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first),False{},some) : Formed(Tasks.Footprint{low,high,_},cap)
  clipped_formed(first,last,shift,bound,source_block,cap,cap_ok,ordered,fits,block_room,or3(U32.is_zero(bound),U32.is_lt(last,shift),U32.is_lt(top,first),some))

# The footprint of an ordered virtual range is formed.
def placed_formed(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  Formed(Tasks.placed(clamped,first,last,shift,bound,source_block),cap):
  match clamped:
    case True{}:
      +top = (shift + bound - 1 : U32)
      +blocks = U32.max(source_block,1)
      +low = ((Tasks.clamp(first,shift,top) - shift) / blocks : U32)
      +high = ((Tasks.clamp(last,shift,top) - shift) / blocks : U32)
      Held.by_cases(Formed(Tasks.Footprint{low,high,U32.is_zero(bound)},cap),U32.is_zero(bound),
        none => {%Equal.sym(Bool,U32.is_zero(bound),True{},none) : Formed(Tasks.Footprint{low,high,_},cap) Unit{} : Formed(Tasks.Footprint{low,high,U32.is_zero(bound)},cap)},
        some => clamped_some(first,last,shift,bound,source_block,cap,cap_ok,ordered,fits,block_room,some))
    case False{}:
      +top = (shift + bound - 1 : U32)
      +blocks = U32.max(source_block,1)
      +low = ((U32.max(first,shift) - shift) / blocks : U32)
      +high = ((U32.min(last,top) - shift) / blocks : U32)
      +empty = U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first)
      Held.by_cases(Formed(Tasks.Footprint{low,high,empty},cap),empty,
        none => {%Equal.sym(Bool,empty,True{},none) : Formed(Tasks.Footprint{low,high,_},cap) Unit{} : Formed(Tasks.Footprint{low,high,empty},cap)},
        some => clipped_some(first,last,shift,bound,source_block,cap,cap_ok,ordered,fits,block_room,some))

# ---- The virtual indices a read range reaches ----

def forward_sum(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+index: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),U32.to_nat(index))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((offset + step * index : U32)) == Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),U32.to_nat(index))) : Nat}:
  +product = Nat.mul(U32.to_nat(step),U32.to_nat(index))
  Arithmetic.sum_exact(cap,cap_ok,offset,(step * index : U32),U32.to_nat(offset),product,{==},
    Arithmetic.product_exact(cap,cap_ok,step,index,U32.to_nat(step),U32.to_nat(index),{==},{==},
      Order.transitive(product,Nat.add(U32.to_nat(offset),product),U32.to_nat(cap),Order.right_le(U32.to_nat(offset),product),room)),room)

def backward_difference(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+index: U32,
  +inside: {Nat.is_le(Nat.mul(U32.to_nat(step),U32.to_nat(index)),U32.to_nat(offset)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(offset),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((offset - step * index : U32)) == Nat.sub(U32.to_nat(offset),Nat.mul(U32.to_nat(step),U32.to_nat(index))) : Nat}:
  +product = Nat.mul(U32.to_nat(step),U32.to_nat(index))
  Arithmetic.difference_exact(cap,cap_ok,offset,(step * index : U32),U32.to_nat(offset),product,{==},
    Arithmetic.product_exact(cap,cap_ok,step,index,U32.to_nat(step),U32.to_nat(index),{==},{==},Order.transitive(product,U32.to_nat(offset),U32.to_nat(cap),inside,room)),
    inside,room)

def widened(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+width: U32,+value_nat: Nat,+exact: {U32.to_nat(value) == value_nat : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+room: {Nat.is_le(Nat.add(value_nat,U32.to_nat(width)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((value + width - 1 : U32)) == Nat.sub(Nat.add(value_nat,U32.to_nat(width)),1n) : Nat}:
  +total = Nat.add(value_nat,U32.to_nat(width))
  Arithmetic.difference_exact(cap,cap_ok,(value + width : U32),1,total,1n,Arithmetic.sum_exact(cap,cap_ok,value,width,value_nat,U32.to_nat(width),exact,{==},room),{==},
    Order.strict_follow(0n,total,Order.lt_le(0n,U32.to_nat(width),total,positive,Order.right_le(value_nat,U32.to_nat(width)))),room)

def divided(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+divisor: U32,+value_nat: Nat,+exact: {U32.to_nat(value) == value_nat : Nat},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((value / U32.max(divisor,1) : U32)) == Nat.div(value_nat,U32.to_nat(U32.max(divisor,1))) : Nat}:
  U32Division.quotient_exact(cap,cap_ok,value,U32.max(divisor,1),value_nat,U32.to_nat(U32.max(divisor,1)),exact,{==},divisor_room,positive_max(divisor))

# The virtual indices a read reaches up to target index stay below cap.
def ReachAt(mirrored: Bool,+step: U32,+offset: U32,+width: U32,+index: Nat,+cap: U32) -> Data:
  match mirrored:
    case False{}:
      {Nat.is_le(Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),index)),U32.to_nat(width)),U32.to_nat(cap)) == True{} : Bool}
    case True{}:
      Held.Both<{Nat.is_le(Nat.mul(U32.to_nat(step),index),U32.to_nat(offset)) == True{} : Bool},{Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(cap)) == True{} : Bool}>

def reach_down(mirrored: Bool,+step: U32,+offset: U32,+width: U32,+lower: Nat,+upper: Nat,+cap: U32,+ordered: {Nat.is_le(lower,upper) == True{} : Bool},
  +reach: ReachAt(mirrored,step,offset,width,upper,cap)) -> ReachAt(mirrored,step,offset,width,lower,cap):
  match mirrored:
    case False{}:
      Order.transitive(Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),lower)),U32.to_nat(width)),
        Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),upper)),U32.to_nat(width)),U32.to_nat(cap),
        Order.add_le(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),lower)),Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),upper)),U32.to_nat(width),
          Order.offset_le(U32.to_nat(offset),Nat.mul(U32.to_nat(step),lower),Nat.mul(U32.to_nat(step),upper),DivisionOrder.multiply_left_le(U32.to_nat(step),lower,upper,ordered))),
        reach)
    case True{}:
      Held.Both{+inside,+room} = reach
      Held.Both{Order.transitive(Nat.mul(U32.to_nat(step),lower),Nat.mul(U32.to_nat(step),upper),U32.to_nat(offset),
        DivisionOrder.multiply_left_le(U32.to_nat(step),lower,upper,ordered),inside),room}

# The first virtual index of a range does not pass its last.
def virtual_ordered(mirrored: Bool,+step: U32,+offset: U32,+width: U32,+divisor: U32,+low: U32,+high: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+reach: ReachAt(mirrored,step,offset,width,U32.to_nat(high),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tasks.virtual_first(step,offset,mirrored,divisor,low,high)),U32.to_nat(Tasks.virtual_last(step,offset,mirrored,width,divisor,low,high))) == True{} : Bool}:
  match mirrored:
    case False{}:
      +origin = U32.to_nat(offset)
      +near = Nat.mul(U32.to_nat(step),U32.to_nat(low))
      +far = Nat.mul(U32.to_nat(step),U32.to_nat(high))
      +blocks = U32.to_nat(U32.max(divisor,1))
      +stepped = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(low),U32.to_nat(high),ordered)
      +far_sum = Nat.add(origin,far)
      +near_sum = Nat.add(origin,near)
      +wide = Nat.add(far_sum,U32.to_nat(width))
      +far_room = Order.transitive(far_sum,wide,U32.to_nat(cap),Order.le_add(far_sum,U32.to_nat(width)),reach)
      +near_room = Order.transitive(near_sum,far_sum,U32.to_nat(cap),Order.offset_le(origin,near,far,stepped),far_room)
      +first = (offset + step * low : U32)
      +last = (offset + step * high : U32)
      %Equal.sym(Nat,U32.to_nat((first / U32.max(divisor,1) : U32)),Nat.div(near_sum,blocks),
          divided(cap,cap_ok,first,divisor,near_sum,forward_sum(cap,cap_ok,step,offset,low,near_room),divisor_room)) :
        {Nat.is_le(_,U32.to_nat(((last + width - 1 : U32) / U32.max(divisor,1) : U32))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(((last + width - 1 : U32) / U32.max(divisor,1) : U32)),Nat.div(Nat.sub(wide,1n),blocks),
          divided(cap,cap_ok,(last + width - 1 : U32),divisor,Nat.sub(wide,1n),
            widened(cap,cap_ok,last,width,far_sum,forward_sum(cap,cap_ok,step,offset,high,far_room),positive,reach),divisor_room)) :
        {Nat.is_le(Nat.div(near_sum,blocks),_) == True{} : Bool}
      DivisionOrder.quotient_le(near_sum,Nat.sub(wide,1n),blocks,positive_max(divisor),
        Order.transitive(near_sum,far_sum,Nat.sub(wide,1n),Order.offset_le(origin,near,far,stepped),pred_le(far_sum,wide,below_plus(far_sum,U32.to_nat(width),positive))))
    case True{}:
      Held.Both{+inside,+room} = reach
      +origin = U32.to_nat(offset)
      +near = Nat.mul(U32.to_nat(step),U32.to_nat(low))
      +far = Nat.mul(U32.to_nat(step),U32.to_nat(high))
      +blocks = U32.to_nat(U32.max(divisor,1))
      +stepped = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(low),U32.to_nat(high),ordered)
      +origin_room = Order.transitive(origin,Nat.add(origin,U32.to_nat(width)),U32.to_nat(cap),Order.le_add(origin,U32.to_nat(width)),room)
      +near_inside = Order.transitive(near,far,origin,stepped,inside)
      +back = Nat.sub(origin,near)
      +wide = Nat.add(back,U32.to_nat(width))
      +wide_room = Order.transitive(wide,Nat.add(origin,U32.to_nat(width)),U32.to_nat(cap),Order.add_le(back,origin,U32.to_nat(width),Order.sub_le(origin,near)),room)
      +first = (offset - step * high : U32)
      +last = (offset - step * low : U32)
      %Equal.sym(Nat,U32.to_nat((first / U32.max(divisor,1) : U32)),Nat.div(Nat.sub(origin,far),blocks),
          divided(cap,cap_ok,first,divisor,Nat.sub(origin,far),backward_difference(cap,cap_ok,step,offset,high,inside,origin_room),divisor_room)) :
        {Nat.is_le(_,U32.to_nat(((last + width - 1 : U32) / U32.max(divisor,1) : U32))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(((last + width - 1 : U32) / U32.max(divisor,1) : U32)),Nat.div(Nat.sub(wide,1n),blocks),
          divided(cap,cap_ok,(last + width - 1 : U32),divisor,Nat.sub(wide,1n),
            widened(cap,cap_ok,last,width,back,backward_difference(cap,cap_ok,step,offset,low,near_inside,origin_room),positive,wide_room),divisor_room)) :
        {Nat.is_le(Nat.div(Nat.sub(origin,far),blocks),_) == True{} : Bool}
      DivisionOrder.quotient_le(Nat.sub(origin,far),Nat.sub(wide,1n),blocks,positive_max(divisor),
        Order.transitive(Nat.sub(origin,far),back,Nat.sub(wide,1n),Order.sub_antitone(origin,near,far,stepped),pred_le(back,wide,below_plus(back,U32.to_nat(width),positive))))

# A read whose virtual indices up to target index stay below cap, and whose
# divisor, source blocks and bound fit below cap.
def ReadRoom(read: Tasks.Source,+index: Nat,+cap: U32) -> Data:
  match read:
    case Tasks.Read{axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,clamped}:
      Held.Both<{Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},Held.Both<ReachAt(mirrored,step,offset,width,index,cap),
        Held.Both<{Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool},
          Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
            {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}>>>>
    case read:
      Unit

def room_down(+read: Tasks.Source,+lower: Nat,+upper: Nat,+cap: U32,+ordered: {Nat.is_le(lower,upper) == True{} : Bool},+room: ReadRoom(read,upper,cap)) ->
  ReadRoom(read,lower,cap):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+positive,+rest} = room
      Held.Both{+reach,+more} = rest
      Held.Both{positive,Held.Both{reach_down(mirrored,step,offset,width,lower,upper,cap,ordered,reach),more}}
    case Tasks.Unread{}:
      Unit{}
    case Tasks.Crossing{}:
      Unit{}

# The footprint of an ordered target range of a read is formed.
def range_formed(+read: Tasks.Source,+low: U32,+high: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool},+room: ReadRoom(read,U32.to_nat(high),cap)) ->
  Formed(Tasks.range_footprint(read,low,high),cap):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+positive,+rest} = room
      Held.Both{+reach,+more} = rest
      Held.Both{+divisor_room,+placing} = more
      Held.Both{+fits,+block_room} = placing
      placed_formed(clamped,Tasks.virtual_first(step,offset,mirrored,divisor,low,high),Tasks.virtual_last(step,offset,mirrored,width,divisor,low,high),
        shift,bound,source_block,cap,cap_ok,virtual_ordered(mirrored,step,offset,width,divisor,low,high,cap,cap_ok,ordered,positive,reach,divisor_room),fits,block_room)
    case Tasks.Unread{}:
      Unit{}
    case Tasks.Crossing{}:
      Unit{}

# ---- The footprints of every block of a target axis ----

# The cells of target block index lie in the axis.
def block_formed(+read: Tasks.Source,+extent: U32,+block: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +before: {Nat.is_lt(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(extent)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap)) -> Formed(Tasks.block_footprint(read,extent,block,index),cap):
  +start = Nat.mul(U32.to_nat(index),U32.to_nat(block))
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +low = (index * block : U32)
  +reach = Nat.add(start,size)
  +start_room = Order.transitive(start,total,U32.to_nat(cap),lt_to_le(start,total,before),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  +low_exact = Arithmetic.product_exact(cap,cap_ok,index,block,U32.to_nat(index),size,{==},{==},start_room)
  +reach_room = Order.transitive(reach,Nat.add(total,size),U32.to_nat(cap),Order.add_le(start,total,size,lt_to_le(start,total,before)),room)
  +end = U32.min((low + block : U32),extent)
  +end_value = Nat.min(reach,total)
  +end_exact = {%Arithmetic.sum_exact(cap,cap_ok,low,block,start,size,low_exact,{==},reach_room) :
      {U32.to_nat(end) == Nat.min(_,total) : Nat}
    Extrema.minimum((low + block : U32),extent) : {U32.to_nat(end) == end_value : Nat}}
  +inside = NatExtrema.minimum_strict_lower(reach,total,start,below_plus(start,size,positive),before)
  +end_room = Order.transitive(end_value,total,U32.to_nat(cap),NatExtrema.minimum_right(reach,total),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  +high = (end - 1 : U32)
  +high_value = Nat.sub(end_value,1n)
  +high_exact = Arithmetic.difference_exact(cap,cap_ok,end,1,end_value,1n,end_exact,{==},
    Order.strict_follow(0n,end_value,Order.le_lt_trans(0n,start,end_value,Order.zero_le(start),inside)),end_room)
  range_formed(read,low,high,cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(low),start,low_exact) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(high),high_value,high_exact) : {Nat.is_le(start,_) == True{} : Bool}
      pred_le(start,end_value,inside) : {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}},
    room_down(read,U32.to_nat(high),Nat.sub(total,1n),cap,
      {%Equal.sym(Nat,U32.to_nat(high),high_value,high_exact) : {Nat.is_le(_,Nat.sub(total,1n)) == True{} : Bool}
        DivisionOrder.subtract_le(end_value,total,1n,NatExtrema.minimum_right(reach,total)) : {Nat.is_le(U32.to_nat(high),Nat.sub(total,1n)) == True{} : Bool}},
      read_room))

def formed_from(remaining: Nat,+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,
  +covered: {Nat.is_le(Nat.add(U32.to_nat(index),remaining),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +blocks_room: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap)) -> FormedFrom(read,extent,block,cap,index,remaining):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +rest:
      +blocks = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
      +here = Order.lt_le(U32.to_nat(index),Nat.add(U32.to_nat(index),1n+rest),blocks,below_plus(U32.to_nat(index),1n+rest,{==}),covered)
      +before = Equal.trans(Bool,Nat.is_lt(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(extent)),Nat.is_lt(U32.to_nat(index),blocks),True{},
        Equal.sym(Bool,Nat.is_lt(U32.to_nat(index),blocks),Nat.is_lt(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(extent)),
          NatCeiling.upper_comparison(U32.to_nat(extent),U32.to_nat(block),U32.to_nat(index),positive)),here)
      +next_exact = Arithmetic.sum_exact(cap,cap_ok,index,1,U32.to_nat(index),1n,{==},{==},
        Order.transitive(Nat.add(U32.to_nat(index),1n),blocks,U32.to_nat(cap),
          %Equal.sym(Nat,Nat.add(U32.to_nat(index),1n),1n+U32.to_nat(index),Addition.successor_width(U32.to_nat(index))) : {Nat.is_le(_,blocks) == True{} : Bool} Order.strict_follow(U32.to_nat(index),blocks,here),blocks_room))
      Held.Both{block_formed(read,extent,block,index,cap,cap_ok,before,positive,room,read_room),
        formed_from(rest,read,extent,block,cap,cap_ok,(index + 1 : U32),
          {%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),Nat.add(U32.to_nat(index),1n),next_exact) : {Nat.is_le(Nat.add(_,rest),blocks) == True{} : Bool}
            %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(index),1n),rest),Nat.add(U32.to_nat(index),Nat.add(1n,rest)),Addition.associative(U32.to_nat(index),1n,rest)) : {Nat.is_le(_,blocks) == True{} : Bool}
            covered : {Nat.is_le(Nat.add(U32.to_nat((index + 1 : U32)),rest),blocks) == True{} : Bool}},
          blocks_room,positive,room,read_room)}

def formed_split(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},
  +blocks_room: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap),
  +some: {U32.is_zero(Tasks.target_blocks(extent,block)) == False{} : Bool}) ->
  Held.Both<Formed(Tasks.block_footprint(read,extent,block,0),cap),FormedFrom(read,extent,block,cap,1,U32.to_nat((Tasks.target_blocks(extent,block) - 1 : U32)))>:
  +blocks = Tasks.target_blocks(extent,block)
  +count = U32.to_nat(blocks)
  +ceiling = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +nonempty = Arithmetic.word_nonzero_positive(blocks,some)
  +covered = {%blocks_value : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count) : {Nat.is_le(count,ceiling) == True{} : Bool}}
  +count_room = Order.transitive(count,ceiling,U32.to_nat(cap),covered,blocks_room)
  %Equal.sym(Nat,U32.to_nat((blocks - 1 : U32)),Nat.sub(count,1n),
      Arithmetic.difference_exact(cap,cap_ok,blocks,1,count,1n,{==},{==},Order.strict_follow(0n,count,nonempty),count_room)) :
    Held.Both<Formed(Tasks.block_footprint(read,extent,block,0),cap),FormedFrom(read,extent,block,cap,1,_)>
  {%Equal.sym(Nat,1n+Nat.sub(count,1n),count,Order.positive_has_last(count,nonempty)) : FormedFrom(read,extent,block,cap,0,_)
    formed_from(count,read,extent,block,cap,cap_ok,0,covered,blocks_room,positive,room,read_room) : FormedFrom(read,extent,block,cap,0,1n+Nat.sub(count,1n))}

def split_apart(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +split: Held.Both<Formed(Tasks.block_footprint(read,extent,block,0),cap),FormedFrom(read,extent,block,cap,1,U32.to_nat((Tasks.target_blocks(extent,block) - 1 : U32)))>,
  +some: {U32.is_zero(Tasks.target_blocks(extent,block)) == False{} : Bool}) -> StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)):
  Held.Both{+first,+formed} = split
  classes_some(read,extent,block,cap,cap_ok,first,formed,some)

def classes_split(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},
  +blocks_room: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap),
  +some: {U32.is_zero(Tasks.target_blocks(extent,block)) == False{} : Bool}) -> StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)):
  split_apart(read,extent,block,cap,cap_ok,formed_split(read,extent,block,cap,cap_ok,blocks_value,blocks_room,positive,room,read_room,some),some)

# Every class of a read target axis that starts past the first block reads
# blocks apart from the block before it.
def read_classes_apart(+read: Tasks.Source,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},
  +blocks_room: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap)) -> StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)):
  Held.by_cases(StartsApart(read,extent,block,Tasks.read_classes(read,extent,block)),U32.is_zero(Tasks.target_blocks(extent,block)),
    none => classes_none(read,extent,block,none),
    some => classes_split(read,extent,block,cap,cap_ok,blocks_value,blocks_room,positive,room,read_room,some))

# ---- The classes of every reading ----

def whole_none(+source: Tasks.Source,+extent: U32,+block: U32,+none: {U32.is_zero(extent) == True{} : Bool}) ->
  StartsApart(source,extent,block,Tasks.whole_class(extent,block)):
  %Equal.sym(Bool,U32.is_zero(extent),True{},none) :
    StartsApart(source,extent,block,Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}]))
  Unit{}

def whole_some(+source: Tasks.Source,+extent: U32,+block: U32,+some: {U32.is_zero(extent) == False{} : Bool}) ->
  StartsApart(source,extent,block,Tasks.whole_class(extent,block)):
  %Equal.sym(Bool,U32.is_zero(extent),False{},some) :
    StartsApart(source,extent,block,Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}]))
  Held.Both{Inl{{==}},Unit{}}

# One class from the first block starts nowhere past it.
def whole_apart(+source: Tasks.Source,+extent: U32,+block: U32) -> StartsApart(source,extent,block,Tasks.whole_class(extent,block)):
  Held.by_cases(StartsApart(source,extent,block,Tasks.whole_class(extent,block)),U32.is_zero(extent),
    none => whole_none(source,extent,block,none),
    some => whole_some(source,extent,block,some))

# The room a target axis needs: its blocks fit below cap.
def AxisRoom(+extent: U32,+block: U32,+cap: U32) -> Data:
  Held.Both<{U32.to_nat(Tasks.target_blocks(extent,block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},
    Held.Both<{Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
      Held.Both<{Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
        {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}>>>

# The room a reading needs: the blocks of its target axis fit below cap, and
# so do the virtual indices of a read.
def ReadingRoom(+extent: U32,+block: U32,+source: Tasks.Source,+cap: U32) -> Data:
  Held.Both<AxisRoom(extent,block,cap),ReadRoom(source,Nat.sub(U32.to_nat(extent),1n),cap)>

def stepped_none(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,
  +bound: U32,+clamped: Bool,+none: {U32.is_zero(step) == True{} : Bool}) ->
  StartsApart(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,
    Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})):
  +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  %Equal.sym(Bool,U32.is_zero(step),True{},none) :
    StartsApart(read,extent,block,Tasks.stepped_classes(_,read,extent,block))
  whole_apart(read,extent,block)

def stepped_some(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,
  +bound: U32,+clamped: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},cap),
  +some: {U32.is_zero(step) == False{} : Bool}) ->
  StartsApart(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,
    Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})):
  Held.Both{+axis_facts,+read_room} = room
  Held.Both{+blocks_value,+rest} = axis_facts
  Held.Both{+blocks_room,+more} = rest
  Held.Both{+positive,+axis_room} = more
  +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  %Equal.sym(Bool,U32.is_zero(step),False{},some) :
    StartsApart(read,extent,block,Tasks.stepped_classes(_,read,extent,block))
  read_classes_apart(read,extent,block,cap,cap_ok,blocks_value,blocks_room,positive,axis_room,read_room)

# Every class of a reading that starts past the first block reads blocks
# apart from the block before it.
def reading_apart(+extent: U32,+block: U32,+source: Tasks.Source,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,source,cap)) -> StartsApart(source,extent,block,Tasks.classes(Tasks.Reading{extent,block,source})):
  match source:
    case Tasks.Unread{}:
      whole_apart(Tasks.Unread{},extent,block)
    case Tasks.Crossing{}:
      whole_apart(Tasks.Crossing{},extent,block)
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.by_cases(StartsApart(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,
          Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})),U32.is_zero(step),
        none => stepped_none(extent,block,axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,none),
        some => stepped_some(extent,block,axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,cap,cap_ok,room,some))

# ---- Where classes start ----

# A start of the classes of a reading is the first block or reads blocks
# apart from the block before it.
def StartApart(+source: Tasks.Source,+extent: U32,+block: U32,+start: U32) -> Data:
  Either<&2,&2,{start == 0 : U32},Apart(Tasks.block_footprint(source,extent,block,(start - 1 : U32)),Tasks.block_footprint(source,extent,block,start))>

def start_moved(+source: Tasks.Source,+extent: U32,+block: U32,+from: U32,+to: U32,+same: {from == to : U32},+apart: StartApart(source,extent,block,from)) ->
  StartApart(source,extent,block,to):
  %same : StartApart(source,extent,block,_)
  apart

def starts_member(classes: List<&2,Tasks.Class>,+source: Tasks.Source,+extent: U32,+block: U32,+start: U32,+all: StartsApart(source,extent,block,classes),
  +member: Moved.Member(Tasks.class_starts(classes),start)) -> StartApart(source,extent,block,start):
  match classes:
    case Nil{}:
      Empty.absurd(StartApart(source,extent,block,start),member)
    case Con{Tasks.Class{+target_first,+target_count,+source_first,+source_count},+rest}:
      Held.Both{+here,+later} = all
      match member:
        case Placed.OnLeft{+same}:
          start_moved(source,extent,block,target_first,start,same,here)
        case Placed.OnRight{+inner}:
          starts_member(rest,source,extent,block,start,later,inner)

# ---- Starts shared by every source ----

def holds_member(list: List<&2,U32>,+value: U32,+h: {Tasks.holds(list,value) == True{} : Bool}) -> Moved.Member(list,value):
  match list:
    case Nil{}:
      Empty.absurd(Moved.Member(Nil{},value),WordEncoding.false_true(h))
    case Con{+other,+rest}:
      Held.by_cases(Moved.Member(Con{other,rest},value),U32.is_eq(other,value),
        same => Placed.OnLeft{Arithmetic.equal_of_is_eq(other,value,same)},
        differ => Placed.OnRight{holds_member(rest,value,
          {%differ : {_ || Tasks.holds(rest,value) == True{} : Bool}
            h : {False{} || Tasks.holds(rest,value) == True{} : Bool}})})

# A value every list holds.
def AllMember(lists: List<&2,List<&2,U32>>,+value: U32) -> Data:
  match lists:
    case Nil{}:
      Unit
    case Con{+list,rest}:
      Held.Both<Moved.Member(list,value),AllMember(rest,value)>

def all_member(lists: List<&2,List<&2,U32>>,+value: U32,+h: {Tasks.all_hold(lists,value) == True{} : Bool}) -> AllMember(lists,value):
  match lists:
    case Nil{}:
      Unit{}
    case Con{+list,+rest}:
      Held.Both{holds_member(list,value,Logic.left(Tasks.holds(list,value),Tasks.all_hold(rest,value),h)),
        all_member(rest,value,Logic.right(Tasks.holds(list,value),Tasks.all_hold(rest,value),h))}

def all_moved(+lists: List<&2,List<&2,U32>>,+from: U32,+to: U32,+same: {from == to : U32},+h: {Tasks.all_hold(lists,from) == True{} : Bool}) ->
  {Tasks.all_hold(lists,to) == True{} : Bool}:
  %same : {Tasks.all_hold(lists,_) == True{} : Bool}
  h

def shared_after(+first: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+start: U32,
  +result: Held.Both<Moved.Member(rest,start),AllMember(others,start)>) -> Held.Both<Moved.Member(Con{first,rest},start),AllMember(others,start)>:
  Held.Both{+member,+all} = result
  Held.Both{Placed.OnRight{member},all}

def shared_side(+first: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+start: U32,+kept: {Tasks.all_hold(others,first) == True{} : Bool},
  +member: Moved.Member(Con{first,Tasks.shared_starts(rest,others)},start),
  after: Moved.Member(Tasks.shared_starts(rest,others),start) -> Held.Both<Moved.Member(rest,start),AllMember(others,start)>) ->
  Held.Both<Moved.Member(Con{first,rest},start),AllMember(others,start)>:
  match member:
    case Placed.OnLeft{+same}:
      Held.Both{Placed.OnLeft{same},all_member(others,start,all_moved(others,first,start,same,kept))}
    case Placed.OnRight{+inner}:
      shared_after(first,rest,others,start,after(inner))

def shared_kept(+first: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+start: U32,+kept: {Tasks.all_hold(others,first) == True{} : Bool},
  +member: Moved.Member(Tasks.shared_starts(Con{first,rest},others),start),
  after: Moved.Member(Tasks.shared_starts(rest,others),start) -> Held.Both<Moved.Member(rest,start),AllMember(others,start)>) ->
  Held.Both<Moved.Member(Con{first,rest},start),AllMember(others,start)>:
  +later = Tasks.shared_starts(rest,others)
  shared_side(first,rest,others,start,kept,
    {%kept : Moved.Member(Bool.pick(List<&2,U32>,_,Con{first,later},later),start) member : Moved.Member(Con{first,later},start)},after)

def shared_dropped(+first: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+start: U32,+dropped: {Tasks.all_hold(others,first) == False{} : Bool},
  +member: Moved.Member(Tasks.shared_starts(Con{first,rest},others),start),
  after: Moved.Member(Tasks.shared_starts(rest,others),start) -> Held.Both<Moved.Member(rest,start),AllMember(others,start)>) ->
  Held.Both<Moved.Member(Con{first,rest},start),AllMember(others,start)>:
  +later = Tasks.shared_starts(rest,others)
  shared_after(first,rest,others,start,
    after({%dropped : Moved.Member(Bool.pick(List<&2,U32>,_,Con{first,later},later),start) member : Moved.Member(later,start)}))

# A shared start is a start of the first source and of every other.
def shared_member(starts: List<&2,U32>,+others: List<&2,List<&2,U32>>,+start: U32,+member: Moved.Member(Tasks.shared_starts(starts,others),start)) ->
  Held.Both<Moved.Member(starts,start),AllMember(others,start)>:
  match starts:
    case Nil{}:
      Empty.absurd(Held.Both<Moved.Member(Nil{},start),AllMember(others,start)>,member)
    case Con{+first,+rest}:
      +later = Tasks.shared_starts(rest,others)
      Held.by_cases(Held.Both<Moved.Member(Con{first,rest},start),AllMember(others,start)>,Tasks.all_hold(others,first),
        kept => shared_kept(first,rest,others,start,kept,member,inner => shared_member(rest,others,start,inner)),
        dropped => shared_dropped(first,rest,others,start,dropped,member,inner => shared_member(rest,others,start,inner)))

# ---- Runs of a target axis ----

# Every reading of a target axis has room.
def ReadingsRoom(readings: List<&2,Tasks.Reading>,+cap: U32) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+extent,+block,+source},rest}:
      Held.Both<ReadingRoom(extent,block,source,cap),ReadingsRoom(rest,cap)>

def runs_member(starts: List<&2,U32>,+blocks: U32,+start: U32,+member: Moved.Member(Tasks.class_starts(Tasks.runs(starts,blocks)),start)) ->
  Moved.Member(starts,start):
  match starts:
    case Nil{}:
      Empty.absurd(Moved.Member(Nil{},start),member)
    case Con{+first,Nil{}}:
      match member:
        case Placed.OnLeft{+same}:
          Placed.OnLeft{same}
        case Placed.OnRight{+inner}:
          Empty.absurd(Moved.Member(Con{first,Nil{}},start),inner)
    case Con{+first,Con{+next,+rest}}:
      match member:
        case Placed.OnLeft{+same}:
          Placed.OnLeft{same}
        case Placed.OnRight{+inner}:
          Placed.OnRight{runs_member(Con{next,rest},blocks,start,inner)}

def run_starts(+own: Tasks.Reading,+others: List<&2,Tasks.Reading>,+start: U32,
  +shared: Held.Both<Moved.Member(Tasks.class_starts(Tasks.classes(own)),start),AllMember(Tasks.starts_of(others),start)>) ->
  AllMember(Tasks.starts_of(Con{own,others}),start):
  shared

def clipped_first_monotone(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+left: U32,+right: U32,+shift: U32,+source_block: U32,
  +ordered: {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool},
  +right_room: {Nat.is_le(U32.to_nat(U32.max(right,shift)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(((U32.max(left,shift) - shift) / U32.max(source_block,1) : U32)),U32.to_nat(((U32.max(right,shift) - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}:
  +start = U32.to_nat(shift)
  +low = U32.max(left,shift)
  +high = U32.max(right,shift)
  +low_value = Nat.max(U32.to_nat(left),start)
  +high_value = Nat.max(U32.to_nat(right),start)
  +maxes = NatExtrema.maximum_upper(U32.to_nat(left),start,high_value,
    Order.transitive(U32.to_nat(left),U32.to_nat(right),high_value,ordered,NatExtrema.maximum_left(U32.to_nat(right),start)),NatExtrema.maximum_right(U32.to_nat(right),start))
  +ordered_words = {%Equal.sym(Nat,U32.to_nat(low),low_value,max_value(left,shift)) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(high),high_value,max_value(right,shift)) : {Nat.is_le(low_value,_) == True{} : Bool}
    maxes : {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}}
  +low_above = {%Equal.sym(Nat,U32.to_nat(low),low_value,max_value(left,shift)) : {Nat.is_le(start,_) == True{} : Bool}
    NatExtrema.maximum_right(U32.to_nat(left),start) : {Nat.is_le(start,U32.to_nat(low)) == True{} : Bool}}
  +blocks = U32.to_nat(U32.max(source_block,1))
  %Equal.sym(Nat,U32.to_nat(((low - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(low),start),blocks),
      scaled(cap,cap_ok,low,shift,source_block,low_above,Order.transitive(U32.to_nat(low),U32.to_nat(high),U32.to_nat(cap),ordered_words,right_room),block_room)) :
    {Nat.is_le(_,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(high),start),blocks),
      scaled(cap,cap_ok,high,shift,source_block,Order.transitive(start,U32.to_nat(low),U32.to_nat(high),low_above,ordered_words),right_room,block_room)) :
    {Nat.is_le(Nat.div(Nat.sub(U32.to_nat(low),start),blocks),_) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.sub(U32.to_nat(low),start),Nat.sub(U32.to_nat(high),start),blocks,positive_max(source_block),
    DivisionOrder.subtract_le(U32.to_nat(low),U32.to_nat(high),start,ordered_words))

# Clipping at top from above keeps the order of virtual indices past shift.
def clipped_last_monotone(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+left: U32,+right: U32,+shift: U32,+top: U32,+source_block: U32,
  +ordered: {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool},
  +left_above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(left)) == True{} : Bool},+top_above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(top)) == True{} : Bool},
  +top_room: {Nat.is_le(U32.to_nat(top),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(((U32.min(left,top) - shift) / U32.max(source_block,1) : U32)),U32.to_nat(((U32.min(right,top) - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}:
  +start = U32.to_nat(shift)
  +limit = U32.to_nat(top)
  +low = U32.min(left,top)
  +high = U32.min(right,top)
  +low_value = Nat.min(U32.to_nat(left),limit)
  +high_value = Nat.min(U32.to_nat(right),limit)
  +mins = NatExtrema.minimum_lower(U32.to_nat(right),limit,low_value,
    Order.transitive(low_value,U32.to_nat(left),U32.to_nat(right),NatExtrema.minimum_left(U32.to_nat(left),limit),ordered),NatExtrema.minimum_right(U32.to_nat(left),limit))
  +ordered_words = {%Equal.sym(Nat,U32.to_nat(low),low_value,Extrema.minimum(left,top)) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(high),high_value,Extrema.minimum(right,top)) : {Nat.is_le(low_value,_) == True{} : Bool}
    mins : {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}}
  +low_above = {%Equal.sym(Nat,U32.to_nat(low),low_value,Extrema.minimum(left,top)) : {Nat.is_le(start,_) == True{} : Bool}
    NatExtrema.minimum_lower(U32.to_nat(left),limit,start,left_above,top_above) : {Nat.is_le(start,U32.to_nat(low)) == True{} : Bool}}
  +high_room = {%Equal.sym(Nat,U32.to_nat(high),high_value,Extrema.minimum(right,top)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(high_value,limit,U32.to_nat(cap),NatExtrema.minimum_right(U32.to_nat(right),limit),top_room) : {Nat.is_le(U32.to_nat(high),U32.to_nat(cap)) == True{} : Bool}}
  +blocks = U32.to_nat(U32.max(source_block,1))
  %Equal.sym(Nat,U32.to_nat(((low - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(low),start),blocks),
      scaled(cap,cap_ok,low,shift,source_block,low_above,Order.transitive(U32.to_nat(low),U32.to_nat(high),U32.to_nat(cap),ordered_words,high_room),block_room)) :
    {Nat.is_le(_,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(high),start),blocks),
      scaled(cap,cap_ok,high,shift,source_block,Order.transitive(start,U32.to_nat(low),U32.to_nat(high),low_above,ordered_words),high_room,block_room)) :
    {Nat.is_le(Nat.div(Nat.sub(U32.to_nat(low),start),blocks),_) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.sub(U32.to_nat(low),start),Nat.sub(U32.to_nat(high),start),blocks,positive_max(source_block),
    DivisionOrder.subtract_le(U32.to_nat(low),U32.to_nat(high),start,ordered_words))

# Clamping between shift and top keeps the order of virtual indices.
def clamped_monotone(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+left: U32,+right: U32,+shift: U32,+top: U32,+source_block: U32,
  +ordered: {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool},+top_above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(top)) == True{} : Bool},
  +top_room: {Nat.is_le(U32.to_nat(top),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(((Tasks.clamp(left,shift,top) - shift) / U32.max(source_block,1) : U32)),
    U32.to_nat(((Tasks.clamp(right,shift,top) - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}:
  +start = U32.to_nat(shift)
  +limit = U32.to_nat(top)
  +low = Tasks.clamp(left,shift,top)
  +high = Tasks.clamp(right,shift,top)
  +left_max = Nat.max(U32.to_nat(left),start)
  +right_max = Nat.max(U32.to_nat(right),start)
  +maxes = NatExtrema.maximum_upper(U32.to_nat(left),start,right_max,
    Order.transitive(U32.to_nat(left),U32.to_nat(right),right_max,ordered,NatExtrema.maximum_left(U32.to_nat(right),start)),NatExtrema.maximum_right(U32.to_nat(right),start))
  +low_value = Nat.min(left_max,limit)
  +high_value = Nat.min(right_max,limit)
  +ordered_words = {%Equal.sym(Nat,U32.to_nat(low),low_value,clamp_value(left,shift,top)) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(high),high_value,clamp_value(right,shift,top)) : {Nat.is_le(low_value,_) == True{} : Bool}
    NatExtrema.minimum_lower(right_max,limit,low_value,Order.transitive(low_value,left_max,right_max,NatExtrema.minimum_left(left_max,limit),maxes),
      NatExtrema.minimum_right(left_max,limit)) : {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}}
  +low_above = {%Equal.sym(Nat,U32.to_nat(low),low_value,clamp_value(left,shift,top)) : {Nat.is_le(start,_) == True{} : Bool}
    NatExtrema.minimum_lower(left_max,limit,start,NatExtrema.maximum_right(U32.to_nat(left),start),top_above) : {Nat.is_le(start,U32.to_nat(low)) == True{} : Bool}}
  +high_room = {%Equal.sym(Nat,U32.to_nat(high),high_value,clamp_value(right,shift,top)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(high_value,limit,U32.to_nat(cap),NatExtrema.minimum_right(right_max,limit),top_room) : {Nat.is_le(U32.to_nat(high),U32.to_nat(cap)) == True{} : Bool}}
  +blocks = U32.to_nat(U32.max(source_block,1))
  %Equal.sym(Nat,U32.to_nat(((low - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(low),start),blocks),
      scaled(cap,cap_ok,low,shift,source_block,low_above,Order.transitive(U32.to_nat(low),U32.to_nat(high),U32.to_nat(cap),ordered_words,high_room),block_room)) :
    {Nat.is_le(_,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(((high - shift) / U32.max(source_block,1) : U32)),Nat.div(Nat.sub(U32.to_nat(high),start),blocks),
      scaled(cap,cap_ok,high,shift,source_block,Order.transitive(start,U32.to_nat(low),U32.to_nat(high),low_above,ordered_words),high_room,block_room)) :
    {Nat.is_le(Nat.div(Nat.sub(U32.to_nat(low),start),blocks),_) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.sub(U32.to_nat(low),start),Nat.sub(U32.to_nat(high),start),blocks,positive_max(source_block),
    DivisionOrder.subtract_le(U32.to_nat(low),U32.to_nat(high),start,ordered_words))

# ---- Ranges on both sides of a class start read blocks apart ----

def lt_self(+a: Nat,+h: {Nat.is_lt(a,a) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(a,a),True{},Equal.sym(Bool,Nat.is_lt(a,a),False{},Order.not_below(a,a,Order.reflexive(a))),h))

def word_lt(+a: U32,+b: U32,+h: {U32.is_lt(a,b) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),U32.is_lt(a,b),True{},Equal.sym(Bool,U32.is_lt(a,b),Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),Comparison.lt(a,b)),h)

def nat_lt(+a: U32,+b: U32,+h: {Nat.is_lt(U32.to_nat(a),U32.to_nat(b)) == True{} : Bool}) -> {U32.is_lt(a,b) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(a,b),Nat.is_lt(U32.to_nat(a),U32.to_nat(b)),True{},Comparison.lt(a,b),h)

def or3_true(+a: Bool,+b: Bool,+c: Bool,+h: {a || b || c == True{} : Bool}) ->
  Either<&2,&2,{a == True{} : Bool},Either<&2,&2,{b == True{} : Bool},{c == True{} : Bool}>>:
  match a b c:
    case True{} +b +c:
      Inl{{==}}
    case False{} True{} +c:
      Inr{Inl{{==}}}
    case False{} False{} True{}:
      Inr{Inr{{==}}}
    case False{} False{} False{}:
      Empty.absurd(Either<&2,&2,{False{} == True{} : Bool},Either<&2,&2,{False{} == True{} : Bool},{False{} == True{} : Bool}>>,WordEncoding.false_true(h))

def or3_first(+a: Bool,+b: Bool,+c: Bool,+h: {a == True{} : Bool}) -> {a || b || c == True{} : Bool}:
  match a:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} || b || c == True{} : Bool},WordEncoding.false_true(h))

def or3_second(+a: Bool,+b: Bool,+c: Bool,+h: {b == True{} : Bool}) -> {a || b || c == True{} : Bool}:
  match a b:
    case True{} +b:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      Empty.absurd({False{} || False{} || c == True{} : Bool},WordEncoding.false_true(h))

def or3_third(+a: Bool,+b: Bool,+c: Bool,+h: {c == True{} : Bool}) -> {a || b || c == True{} : Bool}:
  match a b c:
    case True{} +b +c:
      {==}
    case False{} True{} +c:
      {==}
    case False{} False{} True{}:
      {==}
    case False{} False{} False{}:
      Empty.absurd({False{} || False{} || False{} == True{} : Bool},WordEncoding.false_true(h))

def shift_le_top(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift: U32,+bound: U32,+nonzero: {U32.is_zero(bound) == False{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(shift),U32.to_nat((shift + bound - 1 : U32))) == True{} : Bool}:
  +positive = Arithmetic.word_nonzero_positive(bound,nonzero)
  %Equal.sym(Nat,U32.to_nat((shift + bound - 1 : U32)),Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n),top_value(cap,cap_ok,shift,bound,positive,fits)) :
    {Nat.is_le(U32.to_nat(shift),_) == True{} : Bool}
  shift_below_top(U32.to_nat(shift),U32.to_nat(bound),positive)

def top_le_cap(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift: U32,+bound: U32,+nonzero: {U32.is_zero(bound) == False{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((shift + bound - 1 : U32)),U32.to_nat(cap)) == True{} : Bool}:
  +positive = Arithmetic.word_nonzero_positive(bound,nonzero)
  +total = Nat.add(U32.to_nat(shift),U32.to_nat(bound))
  %Equal.sym(Nat,U32.to_nat((shift + bound - 1 : U32)),Nat.sub(total,1n),top_value(cap,cap_ok,shift,bound,positive,fits)) :
    {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(Nat.sub(total,1n),total,U32.to_nat(cap),Order.sub_le(total,1n),fits)

def max_room(+value: U32,+shift: U32,+top: U32,+cap: U32,+value_top: {Nat.is_le(U32.to_nat(value),U32.to_nat(top)) == True{} : Bool},
  +shift_top: {Nat.is_le(U32.to_nat(shift),U32.to_nat(top)) == True{} : Bool},+top_room: {Nat.is_le(U32.to_nat(top),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.max(value,shift)),U32.to_nat(cap)) == True{} : Bool}:
  +highest = Nat.max(U32.to_nat(value),U32.to_nat(shift))
  %Equal.sym(Nat,U32.to_nat(U32.max(value,shift)),highest,max_value(value,shift)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(highest,U32.to_nat(top),U32.to_nat(cap),NatExtrema.maximum_upper(U32.to_nat(value),U32.to_nat(shift),U32.to_nat(top),value_top,shift_top),top_room)

# The source block a clamped virtual index lands in.
def clamped_at(+value: U32,+shift: U32,+bound: U32,+source_block: U32) -> U32:
  ((Tasks.clamp(value,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)

# The source blocks a clipped virtual range starts and ends in, and whether
# it reads none.
def clipped_first_at(+value: U32,+shift: U32,+source_block: U32) -> U32:
  ((U32.max(value,shift) - shift) / U32.max(source_block,1) : U32)

def clipped_last_at(+value: U32,+shift: U32,+bound: U32,+source_block: U32) -> U32:
  ((U32.min(value,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)

def clipped_empty(+first: U32,+last: U32,+shift: U32,+bound: U32) -> Bool:
  U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt((shift + bound - 1 : U32),first)

def clamped_below(+f2: U32,+f3: U32,+l0: U32,+l1: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},+l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},
  +nonzero: {U32.is_zero(bound) == False{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(clamped_at(l1,shift,bound,source_block)),U32.to_nat(clamped_at(f2,shift,bound,source_block))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(clamped_at(l0,shift,bound,source_block)),U32.to_nat(clamped_at(f3,shift,bound,source_block))) == True{} : Bool}:
  +top = (shift + bound - 1 : U32)
  +above = shift_le_top(cap,cap_ok,shift,bound,nonzero,fits)
  +room = top_le_cap(cap,cap_ok,shift,bound,nonzero,fits)
  Order.le_lt_trans(U32.to_nat(clamped_at(l0,shift,bound,source_block)),U32.to_nat(clamped_at(l1,shift,bound,source_block)),U32.to_nat(clamped_at(f3,shift,bound,source_block)),
    clamped_monotone(cap,cap_ok,l0,l1,shift,top,source_block,l01,above,room,block_room),
    Order.lt_le(U32.to_nat(clamped_at(l1,shift,bound,source_block)),U32.to_nat(clamped_at(f2,shift,bound,source_block)),U32.to_nat(clamped_at(f3,shift,bound,source_block)),below,
      clamped_monotone(cap,cap_ok,f2,f3,shift,top,source_block,f23,above,room,block_room)))

def clamped_crossing(+f1: U32,+f2: U32,+l2: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+nonzero: {U32.is_zero(bound) == False{} : Bool},
  +formed: Formed(Tasks.Footprint{clamped_at(f2,shift,bound,source_block),clamped_at(l2,shift,bound,source_block),False{}},cap),
  +crossed: {Nat.is_lt(U32.to_nat(clamped_at(l2,shift,bound,source_block)),U32.to_nat(clamped_at(f1,shift,bound,source_block))) == True{} : Bool}) -> Empty:
  Held.Both{+ordered,+inside} = formed
  +top = (shift + bound - 1 : U32)
  +last = U32.to_nat(clamped_at(l2,shift,bound,source_block))
  +start = U32.to_nat(clamped_at(f1,shift,bound,source_block))
  lt_self(last,Order.lt_le(last,start,last,crossed,
    Order.transitive(start,U32.to_nat(clamped_at(f2,shift,bound,source_block)),last,
      clamped_monotone(cap,cap_ok,f1,f2,shift,top,source_block,f12,shift_le_top(cap,cap_ok,shift,bound,nonzero,fits),top_le_cap(cap,cap_ok,shift,bound,nonzero,fits),block_room),
      ordered)))

def clamped_crossed(+f1: U32,+f2: U32,+l2: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+nonzero: {U32.is_zero(bound) == False{} : Bool},
  +formed: Formed(Tasks.placed(True{},f2,l2,shift,bound,source_block),cap),
  +crossed: {Nat.is_lt(U32.to_nat(clamped_at(l2,shift,bound,source_block)),U32.to_nat(clamped_at(f1,shift,bound,source_block))) == True{} : Bool}) -> Empty:
  +low = clamped_at(f2,shift,bound,source_block)
  +high = clamped_at(l2,shift,bound,source_block)
  clamped_crossing(f1,f2,l2,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,nonzero,
    {%nonzero : Formed(Tasks.Footprint{low,high,_},cap) formed : Formed(Tasks.Footprint{low,high,False{}},cap)},crossed)

# Clamped ranges before and after a pair of ranges that read blocks apart
# read blocks apart.
def clamped_apart(+f0: U32,+f1: U32,+f2: U32,+f3: U32,+l0: U32,+l1: U32,+l2: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},
  +l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},
  +formed: Formed(Tasks.placed(True{},f2,l2,shift,bound,source_block),cap),
  +apart: Apart(Tasks.placed(True{},f1,l1,shift,bound,source_block),Tasks.placed(True{},f2,l2,shift,bound,source_block))) ->
  Apart(Tasks.placed(True{},f0,l0,shift,bound,source_block),Tasks.placed(True{},f3,l3,shift,bound,source_block)):
  match apart:
    case Inl{+empty}:
      Inl{empty}
    case Inr{+later}:
      match later:
        case Inl{+empty}:
          Inl{empty}
        case Inr{+ordered}:
          match ordered:
            case Inl{+below}:
              Held.by_cases(Apart(Tasks.placed(True{},f0,l0,shift,bound,source_block),Tasks.placed(True{},f3,l3,shift,bound,source_block)),U32.is_zero(bound),
                zero => Inl{zero},
                nonzero => Inr{Inr{Inl{clamped_below(f2,f3,l0,l1,shift,bound,source_block,cap,cap_ok,fits,block_room,f23,l01,nonzero,below)}}})
            case Inr{+crossed}:
              Held.by_cases(Apart(Tasks.placed(True{},f0,l0,shift,bound,source_block),Tasks.placed(True{},f3,l3,shift,bound,source_block)),U32.is_zero(bound),
                zero => Inl{zero},
                nonzero => Empty.absurd(Apart(Tasks.placed(True{},f0,l0,shift,bound,source_block),Tasks.placed(True{},f3,l3,shift,bound,source_block)),
                  clamped_crossed(f1,f2,l2,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,nonzero,formed,crossed)))

def clipped_empty_cases(+f0: U32,+fm: U32,+f3: U32,+l0: U32,+lm: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +f_le: {Nat.is_le(U32.to_nat(fm),U32.to_nat(f3)) == True{} : Bool},+l_le: {Nat.is_le(U32.to_nat(l0),U32.to_nat(lm)) == True{} : Bool},
  +which: Either<&2,&2,{U32.is_zero(bound) == True{} : Bool},Either<&2,&2,{U32.is_lt(lm,shift) == True{} : Bool},{U32.is_lt((shift + bound - 1 : U32),fm) == True{} : Bool}>>) ->
  Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)):
  match which:
    case Inl{+zero}:
      Inl{or3_first(U32.is_zero(bound),U32.is_lt(l0,shift),U32.is_lt((shift + bound - 1 : U32),f0),zero)}
    case Inr{+later}:
      match later:
        case Inl{+short}:
          Inl{or3_second(U32.is_zero(bound),U32.is_lt(l0,shift),U32.is_lt((shift + bound - 1 : U32),f0),
            nat_lt(l0,shift,Order.le_lt_trans(U32.to_nat(l0),U32.to_nat(lm),U32.to_nat(shift),l_le,word_lt(lm,shift,short))))}
        case Inr{+past}:
          +top = (shift + bound - 1 : U32)
          Inr{Inl{or3_third(U32.is_zero(bound),U32.is_lt(l3,shift),U32.is_lt(top,f3),
            nat_lt(top,f3,Order.lt_le(U32.to_nat(top),U32.to_nat(fm),U32.to_nat(f3),word_lt(top,fm,past),f_le)))}}

# A range that reads no block makes the ranges outside it read blocks apart.
def clipped_from_empty(+f0: U32,+fm: U32,+f3: U32,+l0: U32,+lm: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +f_le: {Nat.is_le(U32.to_nat(fm),U32.to_nat(f3)) == True{} : Bool},+l_le: {Nat.is_le(U32.to_nat(l0),U32.to_nat(lm)) == True{} : Bool},
  +empty: {clipped_empty(fm,lm,shift,bound) == True{} : Bool}) ->
  Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)):
  clipped_empty_cases(f0,fm,f3,l0,lm,l3,shift,bound,source_block,f_le,l_le,
    or3_true(U32.is_zero(bound),U32.is_lt(lm,shift),U32.is_lt((shift + bound - 1 : U32),fm),empty))

def clipped_below(+f0: U32,+f2: U32,+f3: U32,+l0: U32,+l1: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},+l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},
  +left: Held.Both<{U32.is_zero(bound) == False{} : Bool},Held.Both<{U32.is_lt(l0,shift) == False{} : Bool},{U32.is_lt((shift + bound - 1 : U32),f0) == False{} : Bool}>>,
  +right: Held.Both<{U32.is_zero(bound) == False{} : Bool},Held.Both<{U32.is_lt(l3,shift) == False{} : Bool},{U32.is_lt((shift + bound - 1 : U32),f3) == False{} : Bool}>>,
  +below: {Nat.is_lt(U32.to_nat(clipped_last_at(l1,shift,bound,source_block)),U32.to_nat(clipped_first_at(f2,shift,source_block))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(clipped_last_at(l0,shift,bound,source_block)),U32.to_nat(clipped_first_at(f3,shift,source_block))) == True{} : Bool}:
  Held.Both{+nonzero,+left_rest} = left
  Held.Both{+reaches,+left_starts} = left_rest
  Held.Both{+right_nonzero,+right_rest} = right
  Held.Both{+right_reaches,+starts} = right_rest
  +top = (shift + bound - 1 : U32)
  +above = shift_le_top(cap,cap_ok,shift,bound,nonzero,fits)
  +room = top_le_cap(cap,cap_ok,shift,bound,nonzero,fits)
  Order.le_lt_trans(U32.to_nat(clipped_last_at(l0,shift,bound,source_block)),U32.to_nat(clipped_last_at(l1,shift,bound,source_block)),
    U32.to_nat(clipped_first_at(f3,shift,source_block)),
    clipped_last_monotone(cap,cap_ok,l0,l1,shift,top,source_block,l01,nat_lt_false(l0,shift,reaches),above,room,block_room),
    Order.lt_le(U32.to_nat(clipped_last_at(l1,shift,bound,source_block)),U32.to_nat(clipped_first_at(f2,shift,source_block)),U32.to_nat(clipped_first_at(f3,shift,source_block)),below,
      clipped_first_monotone(cap,cap_ok,f2,f3,shift,source_block,f23,max_room(f3,shift,top,cap,nat_lt_false(top,f3,starts),above,room),block_room)))

def clipped_below_right(+f0: U32,+f2: U32,+f3: U32,+l0: U32,+l1: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},+l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},
  +left: {clipped_empty(f0,l0,shift,bound) == False{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(clipped_last_at(l1,shift,bound,source_block)),U32.to_nat(clipped_first_at(f2,shift,source_block))) == True{} : Bool}) ->
  Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)):
  +top = (shift + bound - 1 : U32)
  Held.by_cases(Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)),clipped_empty(f3,l3,shift,bound),
    empty => Inr{Inl{empty}},
    right => Inr{Inr{Inl{clipped_below(f0,f2,f3,l0,l1,l3,shift,bound,source_block,cap,cap_ok,fits,block_room,f23,l01,
      or3(U32.is_zero(bound),U32.is_lt(l0,shift),U32.is_lt(top,f0),left),or3(U32.is_zero(bound),U32.is_lt(l3,shift),U32.is_lt(top,f3),right),below)}}})

def clipped_crossed(+f1: U32,+f2: U32,+l2: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},
  +parts: Held.Both<{U32.is_zero(bound) == False{} : Bool},Held.Both<{U32.is_lt(l2,shift) == False{} : Bool},{U32.is_lt((shift + bound - 1 : U32),f2) == False{} : Bool}>>,
  +formed: Formed(Tasks.Footprint{clipped_first_at(f2,shift,source_block),clipped_last_at(l2,shift,bound,source_block),False{}},cap),
  +crossed: {Nat.is_lt(U32.to_nat(clipped_last_at(l2,shift,bound,source_block)),U32.to_nat(clipped_first_at(f1,shift,source_block))) == True{} : Bool}) -> Empty:
  Held.Both{+nonzero,+rest} = parts
  Held.Both{+reaches,+starts} = rest
  Held.Both{+ordered,+inside} = formed
  +top = (shift + bound - 1 : U32)
  +above = shift_le_top(cap,cap_ok,shift,bound,nonzero,fits)
  +room = top_le_cap(cap,cap_ok,shift,bound,nonzero,fits)
  +last = U32.to_nat(clipped_last_at(l2,shift,bound,source_block))
  +start = U32.to_nat(clipped_first_at(f1,shift,source_block))
  lt_self(last,Order.lt_le(last,start,last,crossed,
    Order.transitive(start,U32.to_nat(clipped_first_at(f2,shift,source_block)),last,
      clipped_first_monotone(cap,cap_ok,f1,f2,shift,source_block,f12,max_room(f2,shift,top,cap,nat_lt_false(top,f2,starts),above,room),block_room),
      ordered)))

def clipped_crossing(+f1: U32,+f2: U32,+l2: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+some: {clipped_empty(f2,l2,shift,bound) == False{} : Bool},
  +formed: Formed(Tasks.placed(False{},f2,l2,shift,bound,source_block),cap),
  +crossed: {Nat.is_lt(U32.to_nat(clipped_last_at(l2,shift,bound,source_block)),U32.to_nat(clipped_first_at(f1,shift,source_block))) == True{} : Bool}) -> Empty:
  +low = clipped_first_at(f2,shift,source_block)
  +high = clipped_last_at(l2,shift,bound,source_block)
  clipped_crossed(f1,f2,l2,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,
    or3(U32.is_zero(bound),U32.is_lt(l2,shift),U32.is_lt((shift + bound - 1 : U32),f2),some),
    {%some : Formed(Tasks.Footprint{low,high,_},cap) formed : Formed(Tasks.Footprint{low,high,False{}},cap)},crossed)

def clipped_crossed_cases(+f0: U32,+f1: U32,+f2: U32,+f3: U32,+l0: U32,+l2: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},
  +l02: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l2)) == True{} : Bool},
  +formed: Formed(Tasks.placed(False{},f2,l2,shift,bound,source_block),cap),
  +crossed: {Nat.is_lt(U32.to_nat(clipped_last_at(l2,shift,bound,source_block)),U32.to_nat(clipped_first_at(f1,shift,source_block))) == True{} : Bool}) ->
  Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)):
  Held.by_cases(Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)),clipped_empty(f2,l2,shift,bound),
    empty => clipped_from_empty(f0,f2,f3,l0,l2,l3,shift,bound,source_block,f23,l02,empty),
    some => Empty.absurd(Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)),
      clipped_crossing(f1,f2,l2,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,some,formed,crossed)))

# Clipped ranges before and after a pair of ranges that read blocks apart
# read blocks apart.
def clipped_apart(+f0: U32,+f1: U32,+f2: U32,+f3: U32,+l0: U32,+l1: U32,+l2: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},
  +l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},+l12: {Nat.is_le(U32.to_nat(l1),U32.to_nat(l2)) == True{} : Bool},
  +formed: Formed(Tasks.placed(False{},f2,l2,shift,bound,source_block),cap),
  +apart: Apart(Tasks.placed(False{},f1,l1,shift,bound,source_block),Tasks.placed(False{},f2,l2,shift,bound,source_block))) ->
  Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)):
  match apart:
    case Inl{+empty}:
      clipped_from_empty(f0,f1,f3,l0,l1,l3,shift,bound,source_block,Order.transitive(U32.to_nat(f1),U32.to_nat(f2),U32.to_nat(f3),f12,f23),l01,empty)
    case Inr{+later}:
      match later:
        case Inl{+empty}:
          clipped_from_empty(f0,f2,f3,l0,l2,l3,shift,bound,source_block,f23,Order.transitive(U32.to_nat(l0),U32.to_nat(l1),U32.to_nat(l2),l01,l12),empty)
        case Inr{+ordered}:
          match ordered:
            case Inl{+below}:
              Held.by_cases(Apart(Tasks.placed(False{},f0,l0,shift,bound,source_block),Tasks.placed(False{},f3,l3,shift,bound,source_block)),clipped_empty(f0,l0,shift,bound),
                empty => Inl{empty},
                left => clipped_below_right(f0,f2,f3,l0,l1,l3,shift,bound,source_block,cap,cap_ok,fits,block_room,f23,l01,left,below))
            case Inr{+crossed}:
              clipped_crossed_cases(f0,f1,f2,f3,l0,l2,l3,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,f23,
                Order.transitive(U32.to_nat(l0),U32.to_nat(l1),U32.to_nat(l2),l01,l12),formed,crossed)

# The placements of virtual ranges whose first and last indices grow read
# blocks apart before and after a pair that reads blocks apart.
def placed_apart(clamped: Bool,+f0: U32,+f1: U32,+f2: U32,+f3: U32,+l0: U32,+l1: U32,+l2: U32,+l3: U32,+shift: U32,+bound: U32,+source_block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +f12: {Nat.is_le(U32.to_nat(f1),U32.to_nat(f2)) == True{} : Bool},+f23: {Nat.is_le(U32.to_nat(f2),U32.to_nat(f3)) == True{} : Bool},
  +l01: {Nat.is_le(U32.to_nat(l0),U32.to_nat(l1)) == True{} : Bool},+l12: {Nat.is_le(U32.to_nat(l1),U32.to_nat(l2)) == True{} : Bool},
  +formed: Formed(Tasks.placed(clamped,f2,l2,shift,bound,source_block),cap),
  +apart: Apart(Tasks.placed(clamped,f1,l1,shift,bound,source_block),Tasks.placed(clamped,f2,l2,shift,bound,source_block))) ->
  Apart(Tasks.placed(clamped,f0,l0,shift,bound,source_block),Tasks.placed(clamped,f3,l3,shift,bound,source_block)):
  match clamped:
    case True{}:
      clamped_apart(f0,f1,f2,f3,l0,l1,l2,l3,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,f23,l01,formed,apart)
    case False{}:
      clipped_apart(f0,f1,f2,f3,l0,l1,l2,l3,shift,bound,source_block,cap,cap_ok,fits,block_room,f12,f23,l01,l12,formed,apart)

def apart_swap(+left: Tasks.Footprint,+right: Tasks.Footprint,+apart: Apart(left,right)) -> Apart(right,left):
  match left right:
    case Tasks.Footprint{+left_first,+left_last,+left_empty} Tasks.Footprint{+right_first,+right_last,+right_empty}:
      match apart:
        case Inl{+empty}:
          Inr{Inl{empty}}
        case Inr{+later}:
          match later:
            case Inl{+empty}:
              Inl{empty}
            case Inr{+ordered}:
              match ordered:
                case Inl{+below}:
                  Inr{Inr{Inr{below}}}
                case Inr{+above}:
                  Inr{Inr{Inl{above}}}

# A quotient grows with the value divided.
def quotient_grows(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+low: U32,+high: U32,+divisor: U32,+low_nat: Nat,+high_nat: Nat,
  +low_exact: {U32.to_nat(low) == low_nat : Nat},+high_exact: {U32.to_nat(high) == high_nat : Nat},
  +ordered: {Nat.is_le(low_nat,high_nat) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((low / U32.max(divisor,1) : U32)),U32.to_nat((high / U32.max(divisor,1) : U32))) == True{} : Bool}:
  +blocks = U32.to_nat(U32.max(divisor,1))
  %Equal.sym(Nat,U32.to_nat((low / U32.max(divisor,1) : U32)),Nat.div(low_nat,blocks),divided(cap,cap_ok,low,divisor,low_nat,low_exact,divisor_room)) :
    {Nat.is_le(_,U32.to_nat((high / U32.max(divisor,1) : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((high / U32.max(divisor,1) : U32)),Nat.div(high_nat,blocks),divided(cap,cap_ok,high,divisor,high_nat,high_exact,divisor_room)) :
    {Nat.is_le(Nat.div(low_nat,blocks),_) == True{} : Bool}
  DivisionOrder.quotient_le(low_nat,high_nat,blocks,positive_max(divisor),ordered)

# A forward read starts at a later virtual index from a later target index.
def forward_first_grows(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+width: U32,+divisor: U32,
  +near: U32,+near_high: U32,+far: U32,+far_high: U32,+ordered: {Nat.is_le(U32.to_nat(near),U32.to_nat(far)) == True{} : Bool},
  +reach: ReachAt(False{},step,offset,width,U32.to_nat(far),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tasks.virtual_first(step,offset,False{},divisor,near,near_high)),U32.to_nat(Tasks.virtual_first(step,offset,False{},divisor,far,far_high))) == True{} : Bool}:
  +origin = U32.to_nat(offset)
  +near_step = Nat.mul(U32.to_nat(step),U32.to_nat(near))
  +far_step = Nat.mul(U32.to_nat(step),U32.to_nat(far))
  +stepped = Order.offset_le(origin,near_step,far_step,DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(near),U32.to_nat(far),ordered))
  +far_sum = Nat.add(origin,far_step)
  +near_sum = Nat.add(origin,near_step)
  +far_room = Order.transitive(far_sum,Nat.add(far_sum,U32.to_nat(width)),U32.to_nat(cap),Order.le_add(far_sum,U32.to_nat(width)),reach)
  +near_room = Order.transitive(near_sum,far_sum,U32.to_nat(cap),stepped,far_room)
  quotient_grows(cap,cap_ok,(offset + step * near : U32),(offset + step * far : U32),divisor,near_sum,far_sum,
    forward_sum(cap,cap_ok,step,offset,near,near_room),forward_sum(cap,cap_ok,step,offset,far,far_room),stepped,divisor_room)

# A forward read ends at a later virtual index from a later target index.
def forward_last_grows(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+width: U32,+divisor: U32,
  +near_low: U32,+near: U32,+far_low: U32,+far: U32,+ordered: {Nat.is_le(U32.to_nat(near),U32.to_nat(far)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+reach: ReachAt(False{},step,offset,width,U32.to_nat(far),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tasks.virtual_last(step,offset,False{},width,divisor,near_low,near)),U32.to_nat(Tasks.virtual_last(step,offset,False{},width,divisor,far_low,far))) == True{} : Bool}:
  +origin = U32.to_nat(offset)
  +size = U32.to_nat(width)
  +near_step = Nat.mul(U32.to_nat(step),U32.to_nat(near))
  +far_step = Nat.mul(U32.to_nat(step),U32.to_nat(far))
  +stepped = Order.offset_le(origin,near_step,far_step,DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(near),U32.to_nat(far),ordered))
  +far_sum = Nat.add(origin,far_step)
  +near_sum = Nat.add(origin,near_step)
  +widened_le = Order.add_le(near_sum,far_sum,size,stepped)
  +far_room = Order.transitive(far_sum,Nat.add(far_sum,size),U32.to_nat(cap),Order.le_add(far_sum,size),reach)
  +near_room = Order.transitive(near_sum,far_sum,U32.to_nat(cap),stepped,far_room)
  +near_wide_room = Order.transitive(Nat.add(near_sum,size),Nat.add(far_sum,size),U32.to_nat(cap),widened_le,reach)
  +near_first = (offset + step * near : U32)
  +far_first = (offset + step * far : U32)
  quotient_grows(cap,cap_ok,(near_first + width - 1 : U32),(far_first + width - 1 : U32),divisor,Nat.sub(Nat.add(near_sum,size),1n),Nat.sub(Nat.add(far_sum,size),1n),
    widened(cap,cap_ok,near_first,width,near_sum,forward_sum(cap,cap_ok,step,offset,near,near_room),positive,near_wide_room),
    widened(cap,cap_ok,far_first,width,far_sum,forward_sum(cap,cap_ok,step,offset,far,far_room),positive,reach),
    DivisionOrder.subtract_le(Nat.add(near_sum,size),Nat.add(far_sum,size),1n,widened_le),divisor_room)

# A mirrored read starts at an earlier virtual index from a later target
# index.
def backward_first_falls(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+width: U32,+divisor: U32,
  +near_low: U32,+near: U32,+far_low: U32,+far: U32,+ordered: {Nat.is_le(U32.to_nat(near),U32.to_nat(far)) == True{} : Bool},
  +reach: ReachAt(True{},step,offset,width,U32.to_nat(far),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tasks.virtual_first(step,offset,True{},divisor,far_low,far)),U32.to_nat(Tasks.virtual_first(step,offset,True{},divisor,near_low,near))) == True{} : Bool}:
  Held.Both{+inside,+room} = reach
  +origin = U32.to_nat(offset)
  +near_step = Nat.mul(U32.to_nat(step),U32.to_nat(near))
  +far_step = Nat.mul(U32.to_nat(step),U32.to_nat(far))
  +stepped = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(near),U32.to_nat(far),ordered)
  +origin_room = Order.transitive(origin,Nat.add(origin,U32.to_nat(width)),U32.to_nat(cap),Order.le_add(origin,U32.to_nat(width)),room)
  quotient_grows(cap,cap_ok,(offset - step * far : U32),(offset - step * near : U32),divisor,Nat.sub(origin,far_step),Nat.sub(origin,near_step),
    backward_difference(cap,cap_ok,step,offset,far,inside,origin_room),
    backward_difference(cap,cap_ok,step,offset,near,Order.transitive(near_step,far_step,origin,stepped,inside),origin_room),
    Order.sub_antitone(origin,near_step,far_step,stepped),divisor_room)

# A mirrored read ends at an earlier virtual index from a later target index.
def backward_last_falls(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+step: U32,+offset: U32,+width: U32,+divisor: U32,
  +near: U32,+near_high: U32,+far: U32,+far_high: U32,+ordered: {Nat.is_le(U32.to_nat(near),U32.to_nat(far)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+reach: ReachAt(True{},step,offset,width,U32.to_nat(far),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Tasks.virtual_last(step,offset,True{},width,divisor,far,far_high)),U32.to_nat(Tasks.virtual_last(step,offset,True{},width,divisor,near,near_high))) == True{} : Bool}:
  Held.Both{+inside,+room} = reach
  +origin = U32.to_nat(offset)
  +size = U32.to_nat(width)
  +near_step = Nat.mul(U32.to_nat(step),U32.to_nat(near))
  +far_step = Nat.mul(U32.to_nat(step),U32.to_nat(far))
  +stepped = DivisionOrder.multiply_left_le(U32.to_nat(step),U32.to_nat(near),U32.to_nat(far),ordered)
  +origin_room = Order.transitive(origin,Nat.add(origin,size),U32.to_nat(cap),Order.le_add(origin,size),room)
  +near_back = Nat.sub(origin,near_step)
  +far_back = Nat.sub(origin,far_step)
  +backs = Order.sub_antitone(origin,near_step,far_step,stepped)
  +widened_le = Order.add_le(far_back,near_back,size,backs)
  +near_wide_room = Order.transitive(Nat.add(near_back,size),Nat.add(origin,size),U32.to_nat(cap),Order.add_le(near_back,origin,size,Order.sub_le(origin,near_step)),room)
  +far_wide_room = Order.transitive(Nat.add(far_back,size),Nat.add(near_back,size),U32.to_nat(cap),widened_le,near_wide_room)
  +near_first = (offset - step * near : U32)
  +far_first = (offset - step * far : U32)
  quotient_grows(cap,cap_ok,(far_first + width - 1 : U32),(near_first + width - 1 : U32),divisor,Nat.sub(Nat.add(far_back,size),1n),Nat.sub(Nat.add(near_back,size),1n),
    widened(cap,cap_ok,far_first,width,far_back,backward_difference(cap,cap_ok,step,offset,far,inside,origin_room),positive,far_wide_room),
    widened(cap,cap_ok,near_first,width,near_back,
      backward_difference(cap,cap_ok,step,offset,near,Order.transitive(near_step,far_step,origin,stepped,inside),origin_room),positive,near_wide_room),
    DivisionOrder.subtract_le(Nat.add(far_back,size),Nat.add(near_back,size),1n,widened_le),divisor_room)

# Target ranges r, p, q and s in this order, where p and q read blocks
# apart: r and s read blocks apart.
def read_apart(mirrored: Bool,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +r_low: U32,+r_high: U32,+p_low: U32,+p_high: U32,+q_low: U32,+q_high: U32,+s_low: U32,+s_high: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_pq: {Nat.is_le(U32.to_nat(p_low),U32.to_nat(q_low)) == True{} : Bool},+low_qs: {Nat.is_le(U32.to_nat(q_low),U32.to_nat(s_low)) == True{} : Bool},
  +high_rp: {Nat.is_le(U32.to_nat(r_high),U32.to_nat(p_high)) == True{} : Bool},+high_pq: {Nat.is_le(U32.to_nat(p_high),U32.to_nat(q_high)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +reach_q: ReachAt(mirrored,step,offset,width,U32.to_nat(q_high),cap),+reach_s: ReachAt(mirrored,step,offset,width,U32.to_nat(s_low),cap),
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +formed_p: Formed(Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},p_low,p_high),cap),
  +formed_q: Formed(Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},q_low,q_high),cap),
  +apart: Apart(Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},p_low,p_high),
    Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},q_low,q_high))) ->
  Apart(Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},r_low,r_high),
    Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},s_low,s_high)):
  match mirrored:
    case False{}:
      placed_apart(clamped,Tasks.virtual_first(step,offset,False{},divisor,r_low,r_high),Tasks.virtual_first(step,offset,False{},divisor,p_low,p_high),
        Tasks.virtual_first(step,offset,False{},divisor,q_low,q_high),Tasks.virtual_first(step,offset,False{},divisor,s_low,s_high),
        Tasks.virtual_last(step,offset,False{},width,divisor,r_low,r_high),Tasks.virtual_last(step,offset,False{},width,divisor,p_low,p_high),
        Tasks.virtual_last(step,offset,False{},width,divisor,q_low,q_high),Tasks.virtual_last(step,offset,False{},width,divisor,s_low,s_high),
        shift,bound,source_block,cap,cap_ok,fits,block_room,
        forward_first_grows(cap,cap_ok,step,offset,width,divisor,p_low,p_high,q_low,q_high,low_pq,
          reach_down(False{},step,offset,width,U32.to_nat(q_low),U32.to_nat(s_low),cap,low_qs,reach_s),divisor_room),
        forward_first_grows(cap,cap_ok,step,offset,width,divisor,q_low,q_high,s_low,s_high,low_qs,reach_s,divisor_room),
        forward_last_grows(cap,cap_ok,step,offset,width,divisor,r_low,r_high,p_low,p_high,high_rp,positive,
          reach_down(False{},step,offset,width,U32.to_nat(p_high),U32.to_nat(q_high),cap,high_pq,reach_q),divisor_room),
        forward_last_grows(cap,cap_ok,step,offset,width,divisor,p_low,p_high,q_low,q_high,high_pq,positive,reach_q,divisor_room),
        formed_q,apart)
    case True{}:
      +r_found = Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,True{},width,divisor,shift,bound,clamped},r_low,r_high)
      +p_found = Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,True{},width,divisor,shift,bound,clamped},p_low,p_high)
      +q_found = Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,True{},width,divisor,shift,bound,clamped},q_low,q_high)
      +s_found = Tasks.range_footprint(Tasks.Read{axis,source_block,step,offset,True{},width,divisor,shift,bound,clamped},s_low,s_high)
      apart_swap(s_found,r_found,
        placed_apart(clamped,Tasks.virtual_first(step,offset,True{},divisor,s_low,s_high),Tasks.virtual_first(step,offset,True{},divisor,q_low,q_high),
          Tasks.virtual_first(step,offset,True{},divisor,p_low,p_high),Tasks.virtual_first(step,offset,True{},divisor,r_low,r_high),
          Tasks.virtual_last(step,offset,True{},width,divisor,s_low,s_high),Tasks.virtual_last(step,offset,True{},width,divisor,q_low,q_high),
          Tasks.virtual_last(step,offset,True{},width,divisor,p_low,p_high),Tasks.virtual_last(step,offset,True{},width,divisor,r_low,r_high),
          shift,bound,source_block,cap,cap_ok,fits,block_room,
          backward_first_falls(cap,cap_ok,step,offset,width,divisor,p_low,p_high,q_low,q_high,high_pq,reach_q,divisor_room),
          backward_first_falls(cap,cap_ok,step,offset,width,divisor,r_low,r_high,p_low,p_high,high_rp,
            reach_down(True{},step,offset,width,U32.to_nat(p_high),U32.to_nat(q_high),cap,high_pq,reach_q),divisor_room),
          backward_last_falls(cap,cap_ok,step,offset,width,divisor,q_low,q_high,s_low,s_high,low_qs,positive,reach_s,divisor_room),
          backward_last_falls(cap,cap_ok,step,offset,width,divisor,p_low,p_high,q_low,q_high,low_pq,positive,
            reach_down(True{},step,offset,width,U32.to_nat(q_low),U32.to_nat(s_low),cap,low_qs,reach_s),divisor_room),
          formed_p,apart_swap(p_found,q_found,apart)))

# Target ranges r, p, q and s in this order, where p and q read blocks
# apart: r and s read blocks apart.
def range_apart(+read: Tasks.Source,+r_low: U32,+r_high: U32,+p_low: U32,+p_high: U32,+q_low: U32,+q_high: U32,+s_low: U32,+s_high: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_pq: {Nat.is_le(U32.to_nat(p_low),U32.to_nat(q_low)) == True{} : Bool},+low_qs: {Nat.is_le(U32.to_nat(q_low),U32.to_nat(s_low)) == True{} : Bool},
  +high_rp: {Nat.is_le(U32.to_nat(r_high),U32.to_nat(p_high)) == True{} : Bool},+high_pq: {Nat.is_le(U32.to_nat(p_high),U32.to_nat(q_high)) == True{} : Bool},
  +room_q: ReadRoom(read,U32.to_nat(q_high),cap),+room_s: ReadRoom(read,U32.to_nat(s_low),cap),
  +formed_p: Formed(Tasks.range_footprint(read,p_low,p_high),cap),+formed_q: Formed(Tasks.range_footprint(read,q_low,q_high),cap),
  +apart: Apart(Tasks.range_footprint(read,p_low,p_high),Tasks.range_footprint(read,q_low,q_high))) ->
  Apart(Tasks.range_footprint(read,r_low,r_high),Tasks.range_footprint(read,s_low,s_high)):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+positive,+rest_q} = room_q
      Held.Both{+reach_q,+more_q} = rest_q
      Held.Both{+divisor_room,+placing} = more_q
      Held.Both{+fits,+block_room} = placing
      Held.Both{+positive_s,+rest_s} = room_s
      Held.Both{+reach_s,+more_s} = rest_s
      read_apart(mirrored,axis,source_block,step,offset,width,divisor,shift,bound,clamped,r_low,r_high,p_low,p_high,q_low,q_high,s_low,s_high,cap,cap_ok,
        low_pq,low_qs,high_rp,high_pq,positive,reach_q,reach_s,divisor_room,fits,block_room,formed_p,formed_q,apart)
    case Tasks.Unread{}:
      Inl{{==}}
    case Tasks.Crossing{}:
      Inl{{==}}

# ---- Two runs of a target axis read blocks apart ----

# A block of the axis starts before its end.
def block_before(+extent: U32,+block: U32,+index: Nat,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +inside: {Nat.is_lt(index,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(index,U32.to_nat(block)),U32.to_nat(extent)) == True{} : Bool}:
  +blocks = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  Equal.trans(Bool,Nat.is_lt(Nat.mul(index,U32.to_nat(block)),U32.to_nat(extent)),Nat.is_lt(index,blocks),True{},
    Equal.sym(Bool,Nat.is_lt(index,blocks),Nat.is_lt(Nat.mul(index,U32.to_nat(block)),U32.to_nat(extent)),NatCeiling.upper_comparison(U32.to_nat(extent),U32.to_nat(block),index,positive)),
    inside)

def block_low_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+extent: U32,+block: U32,+index: U32,
  +before: {Nat.is_lt(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((index * block : U32)) == Nat.mul(U32.to_nat(index),U32.to_nat(block)) : Nat}:
  +start = Nat.mul(U32.to_nat(index),U32.to_nat(block))
  +total = U32.to_nat(extent)
  Arithmetic.product_exact(cap,cap_ok,index,block,U32.to_nat(index),U32.to_nat(block),{==},{==},
    Order.transitive(start,total,U32.to_nat(cap),lt_to_le(start,total,before),Order.transitive(total,Nat.add(total,U32.to_nat(block)),U32.to_nat(cap),Order.le_add(total,U32.to_nat(block)),room)))

def block_high_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+extent: U32,+block: U32,+index: U32,
  +before: {Nat.is_lt(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(extent)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((U32.min(((index * block : U32) + block : U32),extent) - 1 : U32)) ==
    Nat.sub(Nat.min(Nat.add(Nat.mul(U32.to_nat(index),U32.to_nat(block)),U32.to_nat(block)),U32.to_nat(extent)),1n) : Nat}:
  +start = Nat.mul(U32.to_nat(index),U32.to_nat(block))
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +low = (index * block : U32)
  +reach = Nat.add(start,size)
  +reach_room = Order.transitive(reach,Nat.add(total,size),U32.to_nat(cap),Order.add_le(start,total,size,lt_to_le(start,total,before)),room)
  +end = U32.min((low + block : U32),extent)
  +end_value = Nat.min(reach,total)
  +end_exact = {%Arithmetic.sum_exact(cap,cap_ok,low,block,start,size,block_low_exact(cap,cap_ok,extent,block,index,before,room),{==},reach_room) :
      {U32.to_nat(end) == Nat.min(_,total) : Nat}
    Extrema.minimum((low + block : U32),extent) : {U32.to_nat(end) == end_value : Nat}}
  +inside = NatExtrema.minimum_strict_lower(reach,total,start,below_plus(start,size,positive),before)
  +end_room = Order.transitive(end_value,total,U32.to_nat(cap),NatExtrema.minimum_right(reach,total),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  Arithmetic.difference_exact(cap,cap_ok,end,1,end_value,1n,end_exact,{==},
    Order.strict_follow(0n,end_value,Order.le_lt_trans(0n,start,end_value,Order.zero_le(start),inside)),end_room)

# The run before a class start and a run from a later start read blocks
# apart when the blocks on both sides of the start do.
def run_pair(+read: Tasks.Source,+extent: U32,+block: U32,+first: U32,+count: U32,+later_first: U32,+later_count: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +started: {Nat.is_lt(0n,U32.to_nat((first + count : U32))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(later_first)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(later_first),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap),
  +pair: Apart(Tasks.block_footprint(read,extent,block,((first + count : U32) - 1 : U32)),Tasks.block_footprint(read,extent,block,(first + count : U32)))) ->
  Apart(Tasks.range_footprint(read,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),
    Tasks.range_footprint(read,(later_first * block : U32),(U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32))):
  +start = (first + count : U32)
  +previous = (start - 1 : U32)
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +blocks = NatCeiling.ceiling(total,size)
  +start_nat = U32.to_nat(start)
  +later_nat = U32.to_nat(later_first)
  +start_inside = Order.le_lt_trans(start_nat,later_nat,blocks,ordered,inside)
  +start_before = block_before(extent,block,start_nat,positive,start_inside)
  +later_before = block_before(extent,block,later_nat,positive,inside)
  +total_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +start_room = Order.transitive(start_nat,Nat.mul(start_nat,size),U32.to_nat(cap),RowMajor.product_covers(start_nat,size,positive),
    Order.transitive(Nat.mul(start_nat,size),total,U32.to_nat(cap),lt_to_le(Nat.mul(start_nat,size),total,start_before),total_room))
  +previous_nat = Nat.sub(start_nat,1n)
  +previous_exact = Arithmetic.difference_exact(cap,cap_ok,start,1,start_nat,1n,{==},{==},Order.strict_follow(0n,start_nat,started),start_room)
  +step_nat = {%Order.positive_has_last(start_nat,started) : {Nat.add(Nat.mul(previous_nat,size),size) == Nat.mul(_,size) : Nat}
    Order.commutative(Nat.mul(previous_nat,size),size) : {Nat.add(Nat.mul(previous_nat,size),size) == Nat.mul(start_nat,size) : Nat}}
  +previous_value = U32.to_nat(previous)
  +stepped = {%Equal.sym(Nat,previous_value,previous_nat,previous_exact) : {Nat.add(Nat.mul(_,size),size) == Nat.mul(start_nat,size) : Nat}
    step_nat : {Nat.add(Nat.mul(previous_value,size),size) == Nat.mul(start_nat,size) : Nat}}
  +previous_le = {%Equal.sym(Nat,previous_value,previous_nat,previous_exact) : {Nat.is_le(_,start_nat) == True{} : Bool}
    Order.sub_le(start_nat,1n) : {Nat.is_le(previous_value,start_nat) == True{} : Bool}}
  +previous_before = Order.le_lt_trans(Nat.mul(previous_value,size),Nat.mul(start_nat,size),total,RowMajor.mul_le(previous_value,start_nat,size,previous_le),start_before)
  +p_low = (previous * block : U32)
  +q_low = (start * block : U32)
  +s_low = (later_first * block : U32)
  +p_high = (U32.min((p_low + block : U32),extent) - 1 : U32)
  +q_high = (U32.min((q_low + block : U32),extent) - 1 : U32)
  +p_low_exact = block_low_exact(cap,cap_ok,extent,block,previous,previous_before,room)
  +q_low_exact = block_low_exact(cap,cap_ok,extent,block,start,start_before,room)
  +s_low_exact = block_low_exact(cap,cap_ok,extent,block,later_first,later_before,room)
  +p_high_exact = block_high_exact(cap,cap_ok,extent,block,previous,previous_before,positive,room)
  +q_high_exact = block_high_exact(cap,cap_ok,extent,block,start,start_before,positive,room)
  +p_reach = Nat.add(Nat.mul(previous_value,size),size)
  +q_reach = Nat.add(Nat.mul(start_nat,size),size)
  +p_end = Nat.min(p_reach,total)
  +q_end = Nat.min(q_reach,total)
  +reach_le = Order.add_le(Nat.mul(previous_value,size),Nat.mul(start_nat,size),size,RowMajor.mul_le(previous_value,start_nat,size,previous_le))
  +end_le = NatExtrema.minimum_lower(q_reach,total,p_end,Order.transitive(p_end,p_reach,q_reach,NatExtrema.minimum_left(p_reach,total),reach_le),
    NatExtrema.minimum_right(p_reach,total))
  +same_end = Arithmetic.to_nat_injective((p_low + block : U32),q_low,
    Equal.trans(Nat,U32.to_nat((p_low + block : U32)),p_reach,U32.to_nat(q_low),
      Arithmetic.sum_exact(cap,cap_ok,p_low,block,Nat.mul(previous_value,size),size,p_low_exact,{==},
        {%Equal.sym(Nat,p_reach,Nat.mul(start_nat,size),stepped) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          Order.transitive(Nat.mul(start_nat,size),total,U32.to_nat(cap),lt_to_le(Nat.mul(start_nat,size),total,start_before),total_room) :
          {Nat.is_le(p_reach,U32.to_nat(cap)) == True{} : Bool}}),
      Equal.trans(Nat,p_reach,Nat.mul(start_nat,size),U32.to_nat(q_low),stepped,Equal.sym(Nat,U32.to_nat(q_low),Nat.mul(start_nat,size),q_low_exact))))
  +low_pq = {%Equal.sym(Nat,U32.to_nat(p_low),Nat.mul(previous_value,size),p_low_exact) : {Nat.is_le(_,U32.to_nat(q_low)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(q_low),Nat.mul(start_nat,size),q_low_exact) : {Nat.is_le(Nat.mul(previous_value,size),_) == True{} : Bool}
    RowMajor.mul_le(previous_value,start_nat,size,previous_le) : {Nat.is_le(U32.to_nat(p_low),U32.to_nat(q_low)) == True{} : Bool}}
  +low_qs = {%Equal.sym(Nat,U32.to_nat(q_low),Nat.mul(start_nat,size),q_low_exact) : {Nat.is_le(_,U32.to_nat(s_low)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(s_low),Nat.mul(later_nat,size),s_low_exact) : {Nat.is_le(Nat.mul(start_nat,size),_) == True{} : Bool}
    RowMajor.mul_le(start_nat,later_nat,size,ordered) : {Nat.is_le(U32.to_nat(q_low),U32.to_nat(s_low)) == True{} : Bool}}
  +high_pq = {%Equal.sym(Nat,U32.to_nat(p_high),Nat.sub(p_end,1n),p_high_exact) : {Nat.is_le(_,U32.to_nat(q_high)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(q_high),Nat.sub(q_end,1n),q_high_exact) : {Nat.is_le(Nat.sub(p_end,1n),_) == True{} : Bool}
    DivisionOrder.subtract_le(p_end,q_end,1n,end_le) : {Nat.is_le(U32.to_nat(p_high),U32.to_nat(q_high)) == True{} : Bool}}
  +room_q = room_down(read,U32.to_nat(q_high),Nat.sub(total,1n),cap,
    {%Equal.sym(Nat,U32.to_nat(q_high),Nat.sub(q_end,1n),q_high_exact) : {Nat.is_le(_,Nat.sub(total,1n)) == True{} : Bool}
      DivisionOrder.subtract_le(q_end,total,1n,NatExtrema.minimum_right(q_reach,total)) : {Nat.is_le(U32.to_nat(q_high),Nat.sub(total,1n)) == True{} : Bool}},
    read_room)
  +room_s = room_down(read,U32.to_nat(s_low),Nat.sub(total,1n),cap,
    {%Equal.sym(Nat,U32.to_nat(s_low),Nat.mul(later_nat,size),s_low_exact) : {Nat.is_le(_,Nat.sub(total,1n)) == True{} : Bool}
      pred_le(Nat.mul(later_nat,size),total,later_before) : {Nat.is_le(U32.to_nat(s_low),Nat.sub(total,1n)) == True{} : Bool}},
    read_room)
  +s_high = (U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32)
  %same_end : Apart(Tasks.range_footprint(read,(first * block : U32),(U32.min(_,extent) - 1 : U32)),Tasks.range_footprint(read,s_low,s_high))
  range_apart(read,(first * block : U32),p_high,p_low,p_high,q_low,q_high,s_low,s_high,cap,cap_ok,low_pq,low_qs,Order.reflexive(U32.to_nat(p_high)),high_pq,
    room_q,room_s,block_formed(read,extent,block,previous,cap,cap_ok,previous_before,positive,room,read_room),
    block_formed(read,extent,block,start,cap,cap_ok,start_before,positive,room,read_room),pair)

# The run before a start of the classes of a reading and a run from a later
# start read blocks apart.
def run_apart(+read: Tasks.Source,+extent: U32,+block: U32,+first: U32,+count: U32,+later_first: U32,+later_count: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +started: {Nat.is_lt(0n,U32.to_nat((first + count : U32))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(later_first)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(later_first),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap),
  +apart: StartApart(read,extent,block,(first + count : U32))) ->
  Apart(Tasks.range_footprint(read,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),
    Tasks.range_footprint(read,(later_first * block : U32),(U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32))):
  match apart:
    case Inl{+zero}:
      Empty.absurd(Apart(Tasks.range_footprint(read,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),
          Tasks.range_footprint(read,(later_first * block : U32),(U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32))),
        lt_self(0n,{%zero : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} started : {Nat.is_lt(0n,U32.to_nat(0)) == True{} : Bool}}))
    case Inr{+pair}:
      run_pair(read,extent,block,first,count,later_first,later_count,cap,cap_ok,started,ordered,inside,positive,room,read_room,pair)

# ---- Class starts rise from zero ----

# The values of a list rise strictly from low on and stay below limit.
def Rising(list: List<&2,U32>,+low: Nat,+limit: Nat) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+first,rest}:
      Held.Both<{Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
        Held.Both<{Nat.is_lt(U32.to_nat(first),limit) == True{} : Bool},Rising(rest,1n+U32.to_nat(first),limit)>>

# A list of starts begins at zero, or is empty when there is no block.
def HeadAt(list: List<&2,U32>,+limit: Nat) -> Data:
  match list:
    case Nil{}:
      {limit == 0n : Nat}
    case Con{+first,rest}:
      {first == 0 : U32}

def Starts(+list: List<&2,U32>,+limit: Nat) -> Data:
  Held.Both<HeadAt(list,limit),Rising(list,0n,limit)>

def rising_lower(list: List<&2,U32>,+low: Nat,+lower: Nat,+limit: Nat,+ordered: {Nat.is_le(lower,low) == True{} : Bool},+rising: Rising(list,low,limit)) ->
  Rising(list,lower,limit):
  match list:
    case Nil{}:
      Unit{}
    case Con{+first,+rest}:
      Held.Both{+above,+more} = rising
      Held.Both{Order.transitive(lower,low,U32.to_nat(first),ordered,above),more}

# The values of a stack fall strictly down to a last zero.
def FallingRest(+last: U32,rest: List<&2,U32>) -> Data:
  match rest:
    case Nil{}:
      {last == 0 : U32}
    case Con{+first,more}:
      Held.Both<{Nat.is_lt(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},FallingRest(first,more)>

# The values of a stack fall strictly from below high down to a last zero.
def Falling(list: List<&2,U32>,+high: Nat) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+first,rest}:
      Held.Both<{Nat.is_lt(U32.to_nat(first),high) == True{} : Bool},FallingRest(first,rest)>

def falling_raise(list: List<&2,U32>,+high: Nat,+higher: Nat,+ordered: {Nat.is_le(high,higher) == True{} : Bool},+falling: Falling(list,high)) ->
  Falling(list,higher):
  match list:
    case Nil{}:
      Unit{}
    case Con{+first,+rest}:
      Held.Both{+below,+more} = falling
      Held.Both{Order.lt_le(U32.to_nat(first),high,higher,below,ordered),more}

# Reversing a falling stack of classes onto rising starts gives starts that
# rise from zero.
def reversed_starts(rest: List<&2,Tasks.Class>,+class: Tasks.Class,+done: List<&2,Tasks.Class>,+high: Nat,+limit: Nat,
  +falling: Falling(Tasks.class_starts(Con{class,rest}),high),+rising: Rising(Tasks.class_starts(done),high,limit),
  +below: {Nat.is_le(high,limit) == True{} : Bool}) -> Starts(Tasks.class_starts(Tasks.reversed_classes(Con{class,rest},done)),limit):
  match rest class:
    case Nil{} Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both{+under,+zero} = falling
      Held.Both{zero,Held.Both{Order.zero_le(U32.to_nat(first)),Held.Both{Order.lt_le(U32.to_nat(first),high,limit,under,below),
        rising_lower(Tasks.class_starts(done),high,1n+U32.to_nat(first),limit,Order.strict_follow(U32.to_nat(first),high,under),rising)}}}
    case Con{Tasks.Class{+next,+next_count,+next_source_first,+next_source_count},+more} Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both{+under,+later} = falling
      +inside = Order.lt_le(U32.to_nat(first),high,limit,under,below)
      reversed_starts(more,Tasks.Class{next,next_count,next_source_first,next_source_count},Con{Tasks.Class{first,count,source_first,source_count},done},
        U32.to_nat(first),limit,later,
        Held.Both{Order.reflexive(U32.to_nat(first)),Held.Both{inside,
          rising_lower(Tasks.class_starts(done),high,1n+U32.to_nat(first),limit,Order.strict_follow(U32.to_nat(first),high,under),rising)}},
        lt_to_le(U32.to_nat(first),limit,inside))

def WalkFalling(walk: Tasks.Walk,+high: Nat) -> Data:
  match walk:
    case Tasks.Walk{+current,done}:
      Falling(Tasks.class_starts(Con{current,done}),high)

def joined_falling(+found: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,+high: Nat,
  +falling: Falling(Tasks.class_starts(Con{current,done}),high)) -> Falling(Tasks.class_starts(Con{Tasks.joined(found,current),done}),high):
  match found current:
    case Tasks.Footprint{+first,+last,+empty} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      falling

def started_falling(+index: U32,+found: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,+high: Nat,
  +above: {Nat.is_lt(U32.to_nat(index),high) == True{} : Bool},+falling: Falling(Tasks.class_starts(Con{current,done}),U32.to_nat(index))) ->
  Falling(Tasks.class_starts(Con{Tasks.started(index,found),Con{current,done}}),high):
  match found current:
    case Tasks.Footprint{+first,+last,+empty} Tasks.Class{+target_first,+target_count,+source_first,+source_count}:
      Held.Both{above,falling}

def falling_joined(+index: U32,+found: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,+high: Nat,
  +falling: Falling(Tasks.class_starts(Con{current,done}),high),+overlap: {Tasks.overlapping(found,current) == True{} : Bool}) ->
  WalkFalling(Tasks.stepped(Tasks.overlapping(found,current),index,found,current,done),high):
  %Equal.sym(Bool,Tasks.overlapping(found,current),True{},overlap) : WalkFalling(Tasks.stepped(_,index,found,current,done),high)
  joined_falling(found,current,done,high,falling)

def falling_started(+index: U32,+found: Tasks.Footprint,+current: Tasks.Class,+done: List<&2,Tasks.Class>,+high: Nat,
  +above: {Nat.is_lt(U32.to_nat(index),high) == True{} : Bool},+falling: Falling(Tasks.class_starts(Con{current,done}),U32.to_nat(index)),
  +missed: {Tasks.overlapping(found,current) == False{} : Bool}) ->
  WalkFalling(Tasks.stepped(Tasks.overlapping(found,current),index,found,current,done),high):
  %Equal.sym(Bool,Tasks.overlapping(found,current),False{},missed) : WalkFalling(Tasks.stepped(_,index,found,current,done),high)
  started_falling(index,found,current,done,high,above,falling)

# A step of the walk keeps its stack of class starts falling.
def walk_falling(+index: U32,+found: Tasks.Footprint,+walk: Tasks.Walk,+high: Nat,+above: {Nat.is_lt(U32.to_nat(index),high) == True{} : Bool},
  +falling: WalkFalling(walk,U32.to_nat(index))) -> WalkFalling(Tasks.walk_step(index,found,walk),high):
  match walk:
    case Tasks.Walk{+current,+done}:
      Held.by_cases(WalkFalling(Tasks.stepped(Tasks.overlapping(found,current),index,found,current,done),high),Tasks.overlapping(found,current),
        overlap => falling_joined(index,found,current,done,high,
          falling_raise(Tasks.class_starts(Con{current,done}),U32.to_nat(index),high,lt_to_le(U32.to_nat(index),high,above),falling),overlap),
        missed => falling_started(index,found,current,done,high,above,falling,missed))

def walked_starts(remaining: Nat,+read: Tasks.Source,+extent: U32,+block: U32,+index: U32,+walk: Tasks.Walk,+limit: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+falling: WalkFalling(walk,U32.to_nat(index)),
  +covered: {Nat.is_le(Nat.add(U32.to_nat(index),remaining),limit) == True{} : Bool},+room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.walked(remaining,read,extent,block,index,walk)),limit):
  match remaining:
    case 0n:
      match walk:
        case Tasks.Walk{+current,+done}:
          reversed_starts(done,current,Nil{},U32.to_nat(index),limit,falling,Unit{},
            {%Addition.zero_right(U32.to_nat(index)) : {Nat.is_le(_,limit) == True{} : Bool} covered : {Nat.is_le(U32.to_nat(index),limit) == True{} : Bool}})
    case 1n+ +rest:
      +position = U32.to_nat(index)
      +next = (index + 1 : U32)
      +next_exact = Arithmetic.sum_exact(cap,cap_ok,index,1,position,1n,{==},{==},
        Order.transitive(Nat.add(position,1n),limit,U32.to_nat(cap),
          Order.transitive(Nat.add(position,1n),Nat.add(position,1n+rest),limit,Order.offset_le(position,1n,1n+rest,Order.zero_le(rest)),covered),room))
      +found = Tasks.block_footprint(read,extent,block,index)
      walked_starts(rest,read,extent,block,next,Tasks.walk_step(index,found,walk),limit,cap,cap_ok,
        walk_falling(index,found,walk,U32.to_nat(next),
          {%Equal.sym(Nat,U32.to_nat(next),Nat.add(position,1n),next_exact) : {Nat.is_lt(position,_) == True{} : Bool}
            below_plus(position,1n,{==}) : {Nat.is_lt(position,U32.to_nat(next)) == True{} : Bool}},falling),
        {%Equal.sym(Nat,U32.to_nat(next),Nat.add(position,1n),next_exact) : {Nat.is_le(Nat.add(_,rest),limit) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(Nat.add(position,1n),rest),Nat.add(position,Nat.add(1n,rest)),Addition.associative(position,1n,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
          covered : {Nat.is_le(Nat.add(U32.to_nat(next),rest),limit) == True{} : Bool}},
        room)

def started_zero(+found: Tasks.Footprint) -> Falling(Tasks.class_starts(Con{Tasks.started(0,found),Nil{}}),1n):
  match found:
    case Tasks.Footprint{+first,+last,+empty}:
      Held.Both{{==},{==}}

def read_starts_none(+read: Tasks.Source,+extent: U32,+block: U32,+limit: Nat,
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == limit : Nat},
  +none: {U32.is_zero(Tasks.target_blocks(extent,block)) == True{} : Bool}) -> Starts(Tasks.class_starts(Tasks.read_classes(read,extent,block)),limit):
  +blocks = Tasks.target_blocks(extent,block)
  %Equal.sym(Bool,U32.is_zero(blocks),True{},none) :
    Starts(Tasks.class_starts(Tasks.read_classes_when(_,read,extent,block)),limit)
  Held.Both{Equal.sym(Nat,U32.to_nat(0),limit,{%Arithmetic.word_zero(blocks,none) : {U32.to_nat(_) == limit : Nat} blocks_value : {U32.to_nat(0) == limit : Nat}}),Unit{}}

def read_starts_some(+read: Tasks.Source,+extent: U32,+block: U32,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == limit : Nat},+room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(Tasks.target_blocks(extent,block)) == False{} : Bool}) -> Starts(Tasks.class_starts(Tasks.read_classes(read,extent,block)),limit):
  +blocks = Tasks.target_blocks(extent,block)
  +start = {Tasks.Walk{Tasks.started(0,Tasks.block_footprint(read,extent,block,0)),[]} : Tasks.Walk}
  +positive = {%blocks_value : {Nat.is_lt(0n,_) == True{} : Bool} Arithmetic.word_nonzero_positive(blocks,some) : {Nat.is_lt(0n,limit) == True{} : Bool}}
  +rest_exact = Arithmetic.difference_exact(cap,cap_ok,blocks,1,limit,1n,blocks_value,{==},Order.strict_follow(0n,limit,positive),room)
  %Equal.sym(Bool,U32.is_zero(blocks),False{},some) :
    Starts(Tasks.class_starts(Tasks.read_classes_when(_,read,extent,block)),limit)
  walked_starts(U32.to_nat((blocks - 1 : U32)),read,extent,block,1,start,limit,cap,cap_ok,started_zero(Tasks.block_footprint(read,extent,block,0)),
    {%Equal.sym(Nat,U32.to_nat((blocks - 1 : U32)),Nat.sub(limit,1n),rest_exact) : {Nat.is_le(Nat.add(1n,_),limit) == True{} : Bool}
      %Equal.sym(Nat,1n+Nat.sub(limit,1n),limit,Order.positive_has_last(limit,positive)) : {Nat.is_le(_,limit) == True{} : Bool}
      Order.reflexive(limit) : {Nat.is_le(Nat.add(1n,U32.to_nat((blocks - 1 : U32))),limit) == True{} : Bool}},
    room)

# The classes of a read axis start at zero and rise below the blocks.
def read_starts(+read: Tasks.Source,+extent: U32,+block: U32,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == limit : Nat},+room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.read_classes(read,extent,block)),limit):
  Held.by_cases(Starts(Tasks.class_starts(Tasks.read_classes(read,extent,block)),limit),U32.is_zero(Tasks.target_blocks(extent,block)),
    none => read_starts_none(read,extent,block,limit,blocks_value,none),
    some => read_starts_some(read,extent,block,limit,cap,cap_ok,blocks_value,room,some))

def whole_starts_none(+extent: U32,+block: U32,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +none: {U32.is_zero(extent) == True{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.whole_class(extent,block)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  +blocks = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +empty = {%Equal.sym(U32,extent,0,Arithmetic.word_zero(extent,none)) : {Nat.is_le(NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(block)),0n) == True{} : Bool}
    NatCeiling.minimal(0n,U32.to_nat(block),0n,positive,{==}) : {Nat.is_le(blocks,0n) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_zero(extent),True{},none) :
    Starts(Tasks.class_starts(Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}])),blocks)
  Held.Both{Order.antisymmetric(blocks,0n,empty,Order.zero_le(blocks)),Unit{}}

def whole_starts_some(+extent: U32,+block: U32,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +some: {U32.is_zero(extent) == False{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.whole_class(extent,block)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  +blocks = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +inside = Equal.trans(Bool,Nat.is_lt(0n,blocks),Nat.is_lt(Nat.mul(0n,U32.to_nat(block)),U32.to_nat(extent)),True{},
    NatCeiling.upper_comparison(U32.to_nat(extent),U32.to_nat(block),0n,positive),Arithmetic.word_nonzero_positive(extent,some))
  %Equal.sym(Bool,U32.is_zero(extent),False{},some) :
    Starts(Tasks.class_starts(Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}])),blocks)
  Held.Both{{==},Held.Both{{==},Held.Both{inside,Unit{}}}}

# One class from the first block starts at zero.
def whole_starts(+extent: U32,+block: U32,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.whole_class(extent,block)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  Held.by_cases(Starts(Tasks.class_starts(Tasks.whole_class(extent,block)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),U32.is_zero(extent),
    none => whole_starts_none(extent,block,positive,none),
    some => whole_starts_some(extent,block,positive,some))

def stepped_starts_none(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,
  +bound: U32,+clamped: Bool,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+none: {U32.is_zero(step) == True{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})),
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  %Equal.sym(Bool,U32.is_zero(step),True{},none) :
    Starts(Tasks.class_starts(Tasks.stepped_classes(_,read,extent,block)),
      NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))
  whole_starts(extent,block,positive)

def stepped_starts_some(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,
  +bound: U32,+clamped: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +blocks_value: {U32.to_nat(Tasks.target_blocks(extent,block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) : Nat},
  +blocks_room: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(step) == False{} : Bool}) ->
  Starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})),
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  %Equal.sym(Bool,U32.is_zero(step),False{},some) :
    Starts(Tasks.class_starts(Tasks.stepped_classes(_,read,extent,block)),
      NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))
  read_starts(read,extent,block,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),cap,cap_ok,blocks_value,blocks_room)

# The classes of every reading start at zero and rise below the blocks of
# its target axis.
def reading_starts(+extent: U32,+block: U32,+source: Tasks.Source,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,source,cap)) ->
  Starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  match source:
    case Tasks.Unread{}:
      Held.Both{+axis_facts,+read_room} = room
      Held.Both{+blocks_value,+rest} = axis_facts
      Held.Both{+blocks_room,+more} = rest
      Held.Both{+positive,+axis_room} = more
      whole_starts(extent,block,positive)
    case Tasks.Crossing{}:
      Held.Both{+axis_facts,+read_room} = room
      Held.Both{+blocks_value,+rest} = axis_facts
      Held.Both{+blocks_room,+more} = rest
      Held.Both{+positive,+axis_room} = more
      whole_starts(extent,block,positive)
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+axis_facts,+read_room} = room
      Held.Both{+blocks_value,+rest} = axis_facts
      Held.Both{+blocks_room,+more} = rest
      Held.Both{+positive,+axis_room} = more
      Held.by_cases(Starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}})),
          NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),U32.is_zero(step),
        none => stepped_starts_none(extent,block,axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,positive,none),
        some => stepped_starts_some(extent,block,axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,cap,cap_ok,blocks_value,blocks_room,some))

# ---- Starts shared by every source rise from zero ----

def StartsAll(lists: List<&2,List<&2,U32>>,+limit: Nat) -> Data:
  match lists:
    case Nil{}:
      Unit
    case Con{+list,rest}:
      Held.Both<Starts(list,limit),StartsAll(rest,limit)>

def kept_rising(+start: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+low: Nat,+limit: Nat,
  +kept: {Tasks.all_hold(others,start) == True{} : Bool},+above: {Nat.is_le(low,U32.to_nat(start)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(start),limit) == True{} : Bool},+later: Rising(Tasks.shared_starts(rest,others),1n+U32.to_nat(start),limit)) ->
  Rising(Tasks.shared_starts(Con{start,rest},others),low,limit):
  +shared = Tasks.shared_starts(rest,others)
  %Equal.sym(Bool,Tasks.all_hold(others,start),True{},kept) : Rising(Bool.pick(List<&2,U32>,_,Con{start,shared},shared),low,limit)
  Held.Both{above,Held.Both{inside,later}}

def dropped_rising(+start: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+low: Nat,+limit: Nat,
  +dropped: {Tasks.all_hold(others,start) == False{} : Bool},+above: {Nat.is_le(low,U32.to_nat(start)) == True{} : Bool},
  +later: Rising(Tasks.shared_starts(rest,others),1n+U32.to_nat(start),limit)) ->
  Rising(Tasks.shared_starts(Con{start,rest},others),low,limit):
  +shared = Tasks.shared_starts(rest,others)
  %Equal.sym(Bool,Tasks.all_hold(others,start),False{},dropped) : Rising(Bool.pick(List<&2,U32>,_,Con{start,shared},shared),low,limit)
  rising_lower(shared,1n+U32.to_nat(start),low,limit,Order.successor(low,U32.to_nat(start),above),later)

# The starts every source shares rise as the first source's do.
def shared_rising(own: List<&2,U32>,+others: List<&2,List<&2,U32>>,+low: Nat,+limit: Nat,+rising: Rising(own,low,limit)) ->
  Rising(Tasks.shared_starts(own,others),low,limit):
  match own:
    case Nil{}:
      Unit{}
    case Con{+start,+rest}:
      Held.Both{+above,+more} = rising
      Held.Both{+inside,+later} = more
      Held.by_cases(Rising(Tasks.shared_starts(Con{start,rest},others),low,limit),Tasks.all_hold(others,start),
        kept => kept_rising(start,rest,others,low,limit,kept,above,inside,shared_rising(rest,others,1n+U32.to_nat(start),limit,later)),
        dropped => dropped_rising(start,rest,others,low,limit,dropped,above,shared_rising(rest,others,1n+U32.to_nat(start),limit,later)))

def head_holds(list: List<&2,U32>,+limit: Nat,+positive: {Nat.is_lt(0n,limit) == True{} : Bool},+head: HeadAt(list,limit)) ->
  {Tasks.holds(list,0) == True{} : Bool}:
  match list:
    case Nil{}:
      Empty.absurd({Tasks.holds(Nil{},0) == True{} : Bool},lt_self(0n,{%head : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,0n) == True{} : Bool}}))
    case Con{+first,+rest}:
      %Equal.sym(U32,first,0,head) : {Tasks.holds(Con{_,rest},0) == True{} : Bool}
      {==}

def all_hold_zero(lists: List<&2,List<&2,U32>>,+limit: Nat,+positive: {Nat.is_lt(0n,limit) == True{} : Bool},+all: StartsAll(lists,limit)) ->
  {Tasks.all_hold(lists,0) == True{} : Bool}:
  match lists:
    case Nil{}:
      {==}
    case Con{+list,+rest}:
      Held.Both{+starts,+later} = all
      Held.Both{+head,+rising} = starts
      Logic.and_true(Tasks.holds(list,0),Tasks.all_hold(rest,0),head_holds(list,limit,positive,head),all_hold_zero(rest,limit,positive,later))

def shared_head_zero(+start: U32,+rest: List<&2,U32>,+others: List<&2,List<&2,U32>>,+limit: Nat,+zero: {start == 0 : U32},
  +kept: {Tasks.all_hold(others,0) == True{} : Bool}) -> HeadAt(Tasks.shared_starts(Con{start,rest},others),limit):
  +shared = Tasks.shared_starts(rest,others)
  %Equal.sym(U32,start,0,zero) : HeadAt(Bool.pick(List<&2,U32>,Tasks.all_hold(others,_),Con{_,shared},shared),limit)
  %Equal.sym(Bool,Tasks.all_hold(others,0),True{},kept) : HeadAt(Bool.pick(List<&2,U32>,_,Con{0,shared},shared),limit)
  {==}

def shared_head(own: List<&2,U32>,+others: List<&2,List<&2,U32>>,+limit: Nat,+starts: Starts(own,limit),+all: StartsAll(others,limit)) ->
  HeadAt(Tasks.shared_starts(own,others),limit):
  match own:
    case Nil{}:
      Held.Both{+head,+rising} = starts
      head
    case Con{+start,+rest}:
      Held.Both{+head,+rising} = starts
      Held.Both{+above,+more} = rising
      Held.Both{+inside,+later} = more
      shared_head_zero(start,rest,others,limit,head,
        all_hold_zero(others,limit,Order.le_lt_trans(0n,U32.to_nat(start),limit,Order.zero_le(U32.to_nat(start)),inside),all))

# The starts every source shares start at zero and rise below the blocks.
def shared_starts_rise(+own: List<&2,U32>,+others: List<&2,List<&2,U32>>,+limit: Nat,+starts: Starts(own,limit),+all: StartsAll(others,limit)) ->
  Starts(Tasks.shared_starts(own,others),limit):
  Held.Both{+head,+rising} = starts
  Held.Both{shared_head(own,others,limit,starts,all),shared_rising(own,others,0n,limit,rising)}

# Every reading of a target axis has its extent and block.
def SameAxis(readings: List<&2,Tasks.Reading>,+extent: U32,+block: U32) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+other_extent,+other_block,+source},rest}:
      Held.Both<{other_extent == extent : U32},Held.Both<{other_block == block : U32},SameAxis(rest,extent,block)>>

def others_starts(readings: List<&2,Tasks.Reading>,+extent: U32,+block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rooms: ReadingsRoom(readings,cap),+same: SameAxis(readings,extent,block)) ->
  StartsAll(Tasks.starts_of(readings),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+other_extent,+other_block,+source},+rest}:
      Held.Both{+room,+rest_rooms} = rooms
      Held.Both{+same_extent,+more} = same
      Held.Both{+same_block,+rest_same} = more
      +starts = Tasks.class_starts(Tasks.classes(Tasks.Reading{other_extent,other_block,source}))
      +by_block = {%same_block : Starts(starts,NatCeiling.ceiling(U32.to_nat(other_extent),U32.to_nat(_)))
        reading_starts(other_extent,other_block,source,cap,cap_ok,room) : Starts(starts,NatCeiling.ceiling(U32.to_nat(other_extent),U32.to_nat(block)))}
      Held.Both{{%same_extent : Starts(starts,NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(block)))
          by_block : Starts(starts,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))},
        others_starts(rest,extent,block,cap,cap_ok,rest_rooms,rest_same)}

# The starts every reading of a target axis shares start at zero and rise
# below its blocks.
def axis_starts(+extent: U32,+block: U32,+source: Tasks.Source,+others: List<&2,Tasks.Reading>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rooms: ReadingsRoom(Con{Tasks.Reading{extent,block,source},others},cap),+same: SameAxis(others,extent,block)) ->
  Starts(Tasks.shared_starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),Tasks.starts_of(others)),
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))):
  Held.Both{+room,+rest_rooms} = rooms
  shared_starts_rise(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),Tasks.starts_of(others),
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),reading_starts(extent,block,source,cap,cap_ok,room),
    others_starts(others,extent,block,cap,cap_ok,rest_rooms,same))

# ---- The classes of a target axis by a word index ----

# The class at a word index of a list, the empty class past its end.
def class_word(classes: List<&2,Tasks.Class>,+index: U32) -> Tasks.Class:
  match classes:
    case Nil{}: Tasks.Class{0,0,0,0}
    case Con{class,rest}: Bool.pick(Tasks.Class,U32.is_zero(index),class,class_word(rest,(index - 1 : U32)))

# The readings of a source by a word index, none past the end.
def readings_word(lists: List<&2,List<&2,Tasks.Reading>>,+index: U32) -> List<&2,Tasks.Reading>:
  match lists:
    case Nil{}: []
    case Con{own,rest}: Bool.pick(List<&2,Tasks.Reading>,U32.is_zero(index),own,readings_word(rest,(index - 1 : U32)))

# ---- The runs of a target axis ----

def at_zero(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+index: U32,+zero: {U32.is_zero(index) == True{} : Bool}) ->
  {class_word(Con{class,rest},index) == class : Tasks.Class}:
  %Equal.sym(Bool,U32.is_zero(index),True{},zero) : {Bool.pick(Tasks.Class,_,class,class_word(rest,(index - 1 : U32))) == class : Tasks.Class}
  {==}

def at_later(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+index: U32,+nonzero: {U32.is_zero(index) == False{} : Bool}) ->
  {class_word(Con{class,rest},index) == class_word(rest,(index - 1 : U32)) : Tasks.Class}:
  %Equal.sym(Bool,U32.is_zero(index),False{},nonzero) :
    {Bool.pick(Tasks.Class,_,class,class_word(rest,(index - 1 : U32))) == class_word(rest,(index - 1 : U32)) : Tasks.Class}
  {==}

def index_back_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,+nonzero: {U32.is_zero(index) == False{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((index - 1 : U32)) == Nat.sub(U32.to_nat(index),1n) : Nat}:
  Arithmetic.difference_exact(cap,cap_ok,index,1,U32.to_nat(index),1n,{==},{==},
    Order.strict_follow(0n,U32.to_nat(index),Arithmetic.word_nonzero_positive(index,nonzero)),room)

# An index past zero below one more than a length lies one back below it.
def index_back_inside(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,+length: Nat,
  +nonzero: {U32.is_zero(index) == False{} : Bool},+room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),1n+length) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat((index - 1 : U32)),length) == True{} : Bool}:
  +value = U32.to_nat(index)
  %Equal.sym(Nat,U32.to_nat((index - 1 : U32)),Nat.sub(value,1n),index_back_exact(cap,cap_ok,index,nonzero,room)) : {Nat.is_lt(_,length) == True{} : Bool}
  %Equal.sym(Nat,1n+Nat.sub(value,1n),value,Order.positive_has_last(value,Arithmetic.word_nonzero_positive(index,nonzero))) : {Nat.is_lt(_,1n+length) == True{} : Bool}
  inside

def index_back_room(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+index: U32,+nonzero: {U32.is_zero(index) == False{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat((index - 1 : U32)),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((index - 1 : U32)),Nat.sub(U32.to_nat(index),1n),index_back_exact(cap,cap_ok,index,nonzero,room)) :
    {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(Nat.sub(U32.to_nat(index),1n),U32.to_nat(index),U32.to_nat(cap),Order.sub_le(U32.to_nat(index),1n),room)

# The runtime looks a class up by the number of its word index.
def class_word_with(classes: List<&2,Tasks.Class>,+zero: Bool,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool},+test: {U32.is_zero(index) == zero : Bool}) ->
  {Tasks.class_at(classes,U32.to_nat(index)) == class_word(classes,index) : Tasks.Class}:
  match classes zero:
    case Nil{} zero:
      {==}
    case Con{+class,+rest} True{}:
      %Equal.sym(U32,index,0,Arithmetic.word_zero(index,test)) :
        {Tasks.class_at(Con{class,rest},U32.to_nat(_)) == Bool.pick(Tasks.Class,U32.is_zero(_),class,class_word(rest,(_ - 1 : U32))) : Tasks.Class}
      {==}
    case Con{+class,+rest} False{}:
      +back = Equal.trans(Nat,U32.to_nat(index),1n+Nat.sub(U32.to_nat(index),1n),1n+U32.to_nat((index - 1 : U32)),
        Equal.sym(Nat,1n+Nat.sub(U32.to_nat(index),1n),U32.to_nat(index),Order.positive_has_last(U32.to_nat(index),Arithmetic.word_nonzero_positive(index,test))),
        Equal.cong(Nat,Nat,value => 1n+value,Nat.sub(U32.to_nat(index),1n),U32.to_nat((index - 1 : U32)),
          Equal.sym(Nat,U32.to_nat((index - 1 : U32)),Nat.sub(U32.to_nat(index),1n),index_back_exact(cap,cap_ok,index,test,room))))
      %Equal.sym(Bool,U32.is_zero(index),False{},test) :
        {Tasks.class_at(Con{class,rest},U32.to_nat(index)) == Bool.pick(Tasks.Class,_,class,class_word(rest,(index - 1 : U32))) : Tasks.Class}
      %Equal.sym(Nat,U32.to_nat(index),1n+U32.to_nat((index - 1 : U32)),back) :
        {Tasks.class_at(Con{class,rest},_) == class_word(rest,(index - 1 : U32)) : Tasks.Class}
      class_word_with(rest,U32.is_zero((index - 1 : U32)),(index - 1 : U32),cap,cap_ok,index_back_room(cap,cap_ok,index,test,room),{==})

def class_word_exact(+classes: List<&2,Tasks.Class>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> {Tasks.class_at(classes,U32.to_nat(index)) == class_word(classes,index) : Tasks.Class}:
  class_word_with(classes,U32.is_zero(index),index,cap,cap_ok,room,{==})

# The runtime looks the readings of a source up by the number of its index.
def readings_word_with(lists: List<&2,List<&2,Tasks.Reading>>,+zero: Bool,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool},+test: {U32.is_zero(index) == zero : Bool}) ->
  {Tasks.readings_at(lists,U32.to_nat(index)) == readings_word(lists,index) : List<&2,Tasks.Reading>}:
  match lists zero:
    case Nil{} zero:
      {==}
    case Con{+own,+rest} True{}:
      %Equal.sym(U32,index,0,Arithmetic.word_zero(index,test)) :
        {Tasks.readings_at(Con{own,rest},U32.to_nat(_)) == Bool.pick(List<&2,Tasks.Reading>,U32.is_zero(_),own,readings_word(rest,(_ - 1 : U32))) : List<&2,Tasks.Reading>}
      {==}
    case Con{+own,+rest} False{}:
      +back = Equal.trans(Nat,U32.to_nat(index),1n+Nat.sub(U32.to_nat(index),1n),1n+U32.to_nat((index - 1 : U32)),
        Equal.sym(Nat,1n+Nat.sub(U32.to_nat(index),1n),U32.to_nat(index),Order.positive_has_last(U32.to_nat(index),Arithmetic.word_nonzero_positive(index,test))),
        Equal.cong(Nat,Nat,value => 1n+value,Nat.sub(U32.to_nat(index),1n),U32.to_nat((index - 1 : U32)),
          Equal.sym(Nat,U32.to_nat((index - 1 : U32)),Nat.sub(U32.to_nat(index),1n),index_back_exact(cap,cap_ok,index,test,room))))
      %Equal.sym(Bool,U32.is_zero(index),False{},test) :
        {Tasks.readings_at(Con{own,rest},U32.to_nat(index)) == Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))) : List<&2,Tasks.Reading>}
      %Equal.sym(Nat,U32.to_nat(index),1n+U32.to_nat((index - 1 : U32)),back) :
        {Tasks.readings_at(Con{own,rest},_) == readings_word(rest,(index - 1 : U32)) : List<&2,Tasks.Reading>}
      readings_word_with(rest,U32.is_zero((index - 1 : U32)),(index - 1 : U32),cap,cap_ok,index_back_room(cap,cap_ok,index,test,room),{==})

def readings_word_exact(+lists: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.readings_at(lists,U32.to_nat(index)) == readings_word(lists,index) : List<&2,Tasks.Reading>}:
  readings_word_with(lists,U32.is_zero(index),index,cap,cap_ok,room,{==})

def single_index(+index: U32,+nonzero: {U32.is_zero(index) == False{} : Bool},+inside: {Nat.is_lt(U32.to_nat(index),1n) == True{} : Bool}) -> Empty:
  lt_self(U32.to_nat(index),Order.lt_le(U32.to_nat(index),1n,U32.to_nat(index),inside,Order.strict_follow(0n,U32.to_nat(index),Arithmetic.word_nonzero_positive(index,nonzero))))

# The first block of a class lies from low on below limit.
def FirstIn(class: Tasks.Class,+low: Nat,+limit: Nat) -> Data:
  match class:
    case Tasks.Class{+first,count,source_first,source_count}:
      Held.Both<{Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},{Nat.is_lt(U32.to_nat(first),limit) == True{} : Bool}>

def first_lower(+class: Tasks.Class,+low: Nat,+lower: Nat,+limit: Nat,+ordered: {Nat.is_le(lower,low) == True{} : Bool},+found: FirstIn(class,low,limit)) ->
  FirstIn(class,lower,limit):
  match class:
    case Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both{+above,+inside} = found
      Held.Both{Order.transitive(lower,low,U32.to_nat(first),ordered,above),inside}

def single_first(+start: U32,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+rising: Rising(Con{start,Nil{}},low,limit),
  +inside: {Nat.is_lt(U32.to_nat(index),1n) == True{} : Bool},+zero: {U32.is_zero(index) == True{} : Bool}) ->
  FirstIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),low,limit):
  Held.Both{+above,+more} = rising
  Held.Both{+below,+rest} = more
  +class = {Tasks.Class{start,(blocks - start : U32),0,0} : Tasks.Class}
  %Equal.sym(Tasks.Class,class_word(Con{class,Nil{}},index),class,at_zero(class,Nil{},index,zero)) : FirstIn(_,low,limit)
  Held.Both{above,below}

def head_first(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+rising: Rising(Con{start,Con{next,rest}},low,limit),
  +zero: {U32.is_zero(index) == True{} : Bool}) -> FirstIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),low,limit):
  Held.Both{+above,+more} = rising
  Held.Both{+below,+later} = more
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},index),class,at_zero(class,others,index,zero)) : FirstIn(_,low,limit)
  Held.Both{above,below}

def later_first(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,
  +above: {Nat.is_le(low,U32.to_nat(start)) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  +found: FirstIn(class_word(Tasks.runs(Con{next,rest},blocks),(index - 1 : U32)),1n+U32.to_nat(start),limit)) ->
  FirstIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),low,limit):
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},index),class_word(others,(index - 1 : U32)),at_later(class,others,index,nonzero)) : FirstIn(_,low,limit)
  first_lower(class_word(others,(index - 1 : U32)),1n+U32.to_nat(start),low,limit,Order.successor(low,U32.to_nat(start),above),found)

def later_step(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +above: {Nat.is_le(low,U32.to_nat(start)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),1n+List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat((index - 1 : U32)),U32.to_nat(cap)) == True{} : Bool} ->
    FirstIn(class_word(Tasks.runs(Con{next,rest},blocks),(index - 1 : U32)),1n+U32.to_nat(start),limit)) ->
  FirstIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),low,limit):
  later_first(start,next,rest,blocks,low,limit,index,above,nonzero,
    recurse(index_back_inside(cap,cap_ok,index,List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks)),nonzero,room,inside),
      index_back_room(cap,cap_ok,index,nonzero,room)))

# Every run of rising starts begins at one of them.
def runs_first_in(starts: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rising: Rising(starts,low,limit),+inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,Tasks.Class,Tasks.runs(starts,blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> FirstIn(class_word(Tasks.runs(starts,blocks),index),low,limit):
  match starts:
    case Nil{}:
      Empty.absurd(FirstIn(class_word(Nil{},index),low,limit),Order.below_zero(U32.to_nat(index),inside))
    case Con{+start,Nil{}}:
      Held.by_cases(FirstIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),low,limit),U32.is_zero(index),
        zero => single_first(start,blocks,low,limit,index,rising,inside,zero),
        nonzero => Empty.absurd(FirstIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),low,limit),single_index(index,nonzero,inside)))
    case Con{+start,Con{+next,+rest}}:
      Held.Both{+above,+more} = rising
      Held.Both{+below,+later} = more
      Held.by_cases(FirstIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),low,limit),U32.is_zero(index),
        zero => head_first(start,next,rest,blocks,low,limit,index,rising,zero),
        nonzero => later_step(start,next,rest,blocks,low,limit,index,cap,cap_ok,above,inside,room,nonzero,
          back_inside => back_room => runs_first_in(Con{next,rest},blocks,1n+U32.to_nat(start),limit,(index - 1 : U32),cap,cap_ok,later,back_inside,back_room)))

# The run left ends at a start of the runs past zero, no later than run
# right begins, and right begins below limit.
def RunOrder(left: Tasks.Class,right: Tasks.Class,+starts: List<&2,U32>,+limit: Nat) -> Data:
  match left right:
    case Tasks.Class{+first,+count,source_first,source_count} Tasks.Class{+later_first,later_count,later_source_first,later_source_count}:
      Held.Both<Moved.Member(starts,(first + count : U32)),Held.Both<{Nat.is_lt(0n,U32.to_nat((first + count : U32))) == True{} : Bool},
        Held.Both<{Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(later_first)) == True{} : Bool},{Nat.is_lt(U32.to_nat(later_first),limit) == True{} : Bool}>>>

def order_lift(+left: Tasks.Class,+right: Tasks.Class,+first: U32,+starts: List<&2,U32>,+limit: Nat,+order: RunOrder(left,right,starts,limit)) ->
  RunOrder(left,right,Con{first,starts},limit):
  match left right:
    case Tasks.Class{+first_start,+count,+source_first,+source_count} Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      Held.Both{+member,+rest} = order
      Held.Both{Placed.OnRight{member},rest}

def end_same(+start: U32,+next: U32) -> {next == (start + (next - start : U32) : U32) : U32}:
  Equal.sym(U32,(start + (next - start : U32) : U32),next,
    Equal.trans(U32,(start + (next - start : U32) : U32),((next - start : U32) + start : U32),next,Arithmetic.add_commutative(start,(next - start : U32)),
      Subtraction.add_subtract(next,start)))

def runs_head(+next: U32,rest: List<&2,U32>,+blocks: U32,+value: U32,+same: {next == value : U32}) ->
  Moved.Member(Tasks.class_starts(Tasks.runs(Con{next,rest},blocks)),value):
  match rest:
    case Nil{}:
      Placed.OnLeft{same}
    case Con{+after,+more}:
      Placed.OnLeft{same}

def head_pair(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+limit: Nat,+right: Tasks.Class,
  +found: FirstIn(right,U32.to_nat(next),limit),+positive: {Nat.is_lt(0n,U32.to_nat(next)) == True{} : Bool}) ->
  RunOrder(Tasks.Class{start,(next - start : U32),0,0},right,Con{start,Tasks.class_starts(Tasks.runs(Con{next,rest},blocks))},limit):
  match right:
    case Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      Held.Both{+above,+inside} = found
      +end = (start + (next - start : U32) : U32)
      +same = end_same(start,next)
      Held.Both{Placed.OnRight{runs_head(next,rest,blocks,end,same)},
        Held.Both{{%same : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(end)) == True{} : Bool}},
          Held.Both{{%same : {Nat.is_le(U32.to_nat(_),U32.to_nat(later_first)) == True{} : Bool} above : {Nat.is_le(U32.to_nat(end),U32.to_nat(later_first)) == True{} : Bool}},
            inside}}}

def nonzero_after(+first: U32,+second: U32,+ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool}) -> {U32.is_zero(second) == False{} : Bool}:
  Arithmetic.word_positive_nonzero(second,Order.le_lt_trans(0n,U32.to_nat(first),U32.to_nat(second),Order.zero_le(U32.to_nat(first)),ordered))

def head_order(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+first: U32,+second: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rising: Rising(Con{start,Con{next,rest}},low,limit),
  +ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(second),1n+List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool},+zero: {U32.is_zero(first) == True{} : Bool}) ->
  RunOrder(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),first),class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),second),
    Tasks.class_starts(Tasks.runs(Con{start,Con{next,rest}},blocks)),limit):
  Held.Both{+above,+more} = rising
  Held.Both{+below,+later} = more
  Held.Both{+next_above,+next_more} = later
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  +nonzero = nonzero_after(first,second,ordered)
  +all = {Con{start,Tasks.class_starts(others)} : List<&2,U32>}
  %Equal.sym(Tasks.Class,class_word(Con{class,others},first),class,at_zero(class,others,first,zero)) :
    RunOrder(_,class_word(Con{class,others},second),all,limit)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},second),class_word(others,(second - 1 : U32)),at_later(class,others,second,nonzero)) :
    RunOrder(class,_,all,limit)
  head_pair(start,next,rest,blocks,limit,class_word(others,(second - 1 : U32)),
    runs_first_in(Con{next,rest},blocks,U32.to_nat(next),limit,(second - 1 : U32),cap,cap_ok,Held.Both{Order.reflexive(U32.to_nat(next)),next_more},
      index_back_inside(cap,cap_ok,second,List.length(&2,Tasks.Class,others),nonzero,room,inside),index_back_room(cap,cap_ok,second,nonzero,room)),
    Order.le_lt_trans(0n,U32.to_nat(start),U32.to_nat(next),Order.zero_le(U32.to_nat(start)),Order.succ_le_lt(U32.to_nat(start),U32.to_nat(next),next_above)))

def back_ordered(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+second: U32,
  +first_nonzero: {U32.is_zero(first) == False{} : Bool},+room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool},
  +ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((first - 1 : U32)),U32.to_nat((second - 1 : U32))) == True{} : Bool}:
  +early = U32.to_nat(first)
  +late = U32.to_nat(second)
  +second_nonzero = nonzero_after(first,second,ordered)
  +first_room = Order.transitive(early,late,U32.to_nat(cap),lt_to_le(early,late,ordered),room)
  %Equal.sym(Nat,U32.to_nat((first - 1 : U32)),Nat.sub(early,1n),index_back_exact(cap,cap_ok,first,first_nonzero,first_room)) :
    {Nat.is_lt(_,U32.to_nat((second - 1 : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((second - 1 : U32)),Nat.sub(late,1n),index_back_exact(cap,cap_ok,second,second_nonzero,room)) :
    {Nat.is_lt(Nat.sub(early,1n),_) == True{} : Bool}
  %Equal.sym(Nat,1n+Nat.sub(early,1n),early,Order.positive_has_last(early,Arithmetic.word_nonzero_positive(first,first_nonzero))) :
    {Nat.is_lt(_,1n+Nat.sub(late,1n)) == True{} : Bool}
  %Equal.sym(Nat,1n+Nat.sub(late,1n),late,Order.positive_has_last(late,Arithmetic.word_nonzero_positive(second,second_nonzero))) :
    {Nat.is_lt(early,_) == True{} : Bool}
  ordered

def later_order(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+limit: Nat,+first: U32,+second: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(second),1n+List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool},+nonzero: {U32.is_zero(first) == False{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((first - 1 : U32)),U32.to_nat((second - 1 : U32))) == True{} : Bool} ->
    {Nat.is_lt(U32.to_nat((second - 1 : U32)),List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat((second - 1 : U32)),U32.to_nat(cap)) == True{} : Bool} ->
    RunOrder(class_word(Tasks.runs(Con{next,rest},blocks),(first - 1 : U32)),class_word(Tasks.runs(Con{next,rest},blocks),(second - 1 : U32)),
      Tasks.class_starts(Tasks.runs(Con{next,rest},blocks)),limit)) ->
  RunOrder(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),first),class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),second),
    Tasks.class_starts(Tasks.runs(Con{start,Con{next,rest}},blocks)),limit):
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  +second_nonzero = nonzero_after(first,second,ordered)
  +all = {Con{start,Tasks.class_starts(others)} : List<&2,U32>}
  %Equal.sym(Tasks.Class,class_word(Con{class,others},first),class_word(others,(first - 1 : U32)),at_later(class,others,first,nonzero)) :
    RunOrder(_,class_word(Con{class,others},second),all,limit)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},second),class_word(others,(second - 1 : U32)),at_later(class,others,second,second_nonzero)) :
    RunOrder(class_word(others,(first - 1 : U32)),_,all,limit)
  order_lift(class_word(others,(first - 1 : U32)),class_word(others,(second - 1 : U32)),start,Tasks.class_starts(others),limit,
    recurse(back_ordered(cap,cap_ok,first,second,nonzero,room,ordered),
      index_back_inside(cap,cap_ok,second,List.length(&2,Tasks.Class,others),second_nonzero,room,inside),
      index_back_room(cap,cap_ok,second,second_nonzero,room)))

# An earlier run of rising starts ends at a start of the runs no later than a
# later run begins.
def runs_order(starts: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rising: Rising(starts,low,limit),+ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(second),List.length(&2,Tasks.Class,Tasks.runs(starts,blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool}) ->
  RunOrder(class_word(Tasks.runs(starts,blocks),first),class_word(Tasks.runs(starts,blocks),second),Tasks.class_starts(Tasks.runs(starts,blocks)),limit):
  match starts:
    case Nil{}:
      Empty.absurd(RunOrder(class_word(Nil{},first),class_word(Nil{},second),Nil{},limit),Order.below_zero(U32.to_nat(second),inside))
    case Con{+start,Nil{}}:
      Empty.absurd(RunOrder(class_word(Tasks.runs(Con{start,Nil{}},blocks),first),class_word(Tasks.runs(Con{start,Nil{}},blocks),second),
          Tasks.class_starts(Tasks.runs(Con{start,Nil{}},blocks)),limit),
        single_index(second,nonzero_after(first,second,ordered),inside))
    case Con{+start,Con{+next,+rest}}:
      Held.Both{+above,+more} = rising
      Held.Both{+below,+later} = more
      Held.by_cases(RunOrder(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),first),class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),second),
          Tasks.class_starts(Tasks.runs(Con{start,Con{next,rest}},blocks)),limit),U32.is_zero(first),
        zero => head_order(start,next,rest,blocks,low,limit,first,second,cap,cap_ok,rising,ordered,inside,room,zero),
        nonzero => later_order(start,next,rest,blocks,limit,first,second,cap,cap_ok,ordered,inside,room,nonzero,
          back => back_inside => back_room => runs_order(Con{next,rest},blocks,1n+U32.to_nat(start),limit,(first - 1 : U32),(second - 1 : U32),cap,cap_ok,
            later,back,back_inside,back_room)))

# ---- Two runs of a target axis read blocks apart in every source ----

# The ranges of target classes left and right read blocks of a read source
# apart.
def ClassesApart(+source: Tasks.Source,+extent: U32,+block: U32,left: Tasks.Class,right: Tasks.Class) -> Data:
  match left right:
    case Tasks.Class{+first,+count,source_first,source_count} Tasks.Class{+later_first,+later_count,later_source_first,later_source_count}:
      Apart(Tasks.range_footprint(source,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),
        Tasks.range_footprint(source,(later_first * block : U32),(U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32)))

# A source that reads no axis cannot tell two runs apart.
def ReadApart(source: Tasks.Source,+extent: U32,+block: U32,+left: Tasks.Class,+right: Tasks.Class) -> Data:
  match source:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      ClassesApart(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,left,right)
    case source:
      Empty

def AllReadApart(readings: List<&2,Tasks.Reading>,+left: Tasks.Class,+right: Tasks.Class) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+extent,+block,+source},rest}:
      Held.Both<ReadApart(source,extent,block,left,right),AllReadApart(rest,left,right)>

def zero_member(+start: U32,+started: {Nat.is_lt(0n,U32.to_nat(start)) == True{} : Bool},+member: Moved.Member(Con{0,Nil{}},start)) -> Empty:
  match member:
    case Placed.OnLeft{+same}:
      lt_self(0n,{%Equal.sym(U32,0,start,same) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} started : {Nat.is_lt(0n,U32.to_nat(0)) == True{} : Bool}})
    case Placed.OnRight{+inner}:
      inner

def whole_member_none(+extent: U32,+block: U32,+start: U32,+none: {U32.is_zero(extent) == True{} : Bool},
  +member: Moved.Member(Tasks.class_starts(Tasks.whole_class(extent,block)),start)) -> Empty:
  {%none :
      Moved.Member(Tasks.class_starts(Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}])),start)
    member : Moved.Member(Nil{},start)}

def whole_member_some(+extent: U32,+block: U32,+start: U32,+started: {Nat.is_lt(0n,U32.to_nat(start)) == True{} : Bool},+some: {U32.is_zero(extent) == False{} : Bool},
  +member: Moved.Member(Tasks.class_starts(Tasks.whole_class(extent,block)),start)) -> Empty:
  zero_member(start,started,
    {%some :
        Moved.Member(Tasks.class_starts(Bool.pick(List<&2,Tasks.Class>,_,[],[Tasks.Class{0,Tasks.target_blocks(extent,block),0,0}])),start)
      member : Moved.Member(Con{0,Nil{}},start)})

# A start past zero is no start of one class from the first block.
def whole_member(+extent: U32,+block: U32,+start: U32,+started: {Nat.is_lt(0n,U32.to_nat(start)) == True{} : Bool},
  +member: Moved.Member(Tasks.class_starts(Tasks.whole_class(extent,block)),start)) -> Empty:
  Held.by_cases(Empty,U32.is_zero(extent),
    none => whole_member_none(extent,block,start,none,member),
    some => whole_member_some(extent,block,start,started,some,member))

def run_end(+class: Tasks.Class) -> U32:
  match class:
    case Tasks.Class{+first,+count,source_first,source_count}:
      (first + count : U32)

def class_first(+class: Tasks.Class) -> U32:
  match class:
    case Tasks.Class{+first,count,source_first,source_count}:
      first

def read_run_apart(+read: Tasks.Source,+extent: U32,+block: U32,+left: Tasks.Class,+right: Tasks.Class,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,read,cap),+apart: StartApart(read,extent,block,run_end(left)),
  +started: {Nat.is_lt(0n,U32.to_nat(run_end(left))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(run_end(left)),U32.to_nat(class_first(right))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool}) ->
  ClassesApart(read,extent,block,left,right):
  match left right:
    case Tasks.Class{+first,+count,+source_first,+source_count} Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      Held.Both{+axis_facts,+read_room} = room
      Held.Both{+blocks_value,+rest} = axis_facts
      Held.Both{+blocks_room,+more} = rest
      Held.Both{+positive,+axis_room} = more
      run_apart(read,extent,block,first,count,later_first,later_count,cap,cap_ok,started,ordered,inside,positive,axis_room,read_room,apart)

# A run that ends at a start of the classes of a reading and a later run read
# blocks of a read source apart, and a source that reads no axis has no start
# past zero.
def reading_run_apart(+extent: U32,+block: U32,+source: Tasks.Source,+left: Tasks.Class,+right: Tasks.Class,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,source,cap),+member: Moved.Member(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),run_end(left)),
  +started: {Nat.is_lt(0n,U32.to_nat(run_end(left))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(run_end(left)),U32.to_nat(class_first(right))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool}) ->
  ReadApart(source,extent,block,left,right):
  match source:
    case Tasks.Unread{}:
      whole_member(extent,block,run_end(left),started,member)
    case Tasks.Crossing{}:
      whole_member(extent,block,run_end(left),started,member)
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
      read_run_apart(read,extent,block,left,right,cap,cap_ok,room,
        starts_member(Tasks.classes(Tasks.Reading{extent,block,read}),read,extent,block,run_end(left),reading_apart(extent,block,read,cap,cap_ok,room),member),
        started,ordered,inside)

def all_readings_apart(readings: List<&2,Tasks.Reading>,+left: Tasks.Class,+right: Tasks.Class,+extent: U32,+block: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ReadingsRoom(readings,cap),+same: SameAxis(readings,extent,block),
  +members: AllMember(Tasks.starts_of(readings),run_end(left)),+started: {Nat.is_lt(0n,U32.to_nat(run_end(left))) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(run_end(left)),U32.to_nat(class_first(right))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool}) ->
  AllReadApart(readings,left,right):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+other_extent,+other_block,+source},+rest}:
      Held.Both{+room,+rest_rooms} = rooms
      Held.Both{+same_extent,+more} = same
      Held.Both{+same_block,+rest_same} = more
      Held.Both{+member,+rest_members} = members
      +here = {%Equal.sym(U32,other_extent,extent,same_extent) :
          {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(other_block))) == True{} : Bool}
        %Equal.sym(U32,other_block,block,same_block) :
          {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_))) == True{} : Bool}
        inside : {Nat.is_lt(U32.to_nat(class_first(right)),NatCeiling.ceiling(U32.to_nat(other_extent),U32.to_nat(other_block))) == True{} : Bool}}
      Held.Both{reading_run_apart(other_extent,other_block,source,left,right,cap,cap_ok,room,member,started,ordered,here),
        all_readings_apart(rest,left,right,extent,block,cap,cap_ok,rest_rooms,rest_same,rest_members,started,ordered,inside)}

def order_all_apart(+extent: U32,+block: U32,+source: Tasks.Source,+others: List<&2,Tasks.Reading>,+left: Tasks.Class,+right: Tasks.Class,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ReadingsRoom(Con{Tasks.Reading{extent,block,source},others},cap),
  +same: SameAxis(others,extent,block),
  +order: RunOrder(left,right,Tasks.class_starts(Tasks.axis_classes(Con{Tasks.Reading{extent,block,source},others})),
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))) ->
  AllReadApart(Con{Tasks.Reading{extent,block,source},others},left,right):
  match left right:
    case Tasks.Class{+first,+count,+source_first,+source_count} Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      Held.Both{+member,+rest} = order
      Held.Both{+started,+more} = rest
      Held.Both{+ordered,+inside} = more
      +own = {Tasks.Reading{extent,block,source} : Tasks.Reading}
      +own_starts = Tasks.class_starts(Tasks.classes(own))
      +other_starts = Tasks.starts_of(others)
      +end = (first + count : U32)
      all_readings_apart(Con{own,others},Tasks.Class{first,count,source_first,source_count},Tasks.Class{later_first,later_count,later_source_first,later_source_count},
        extent,block,cap,cap_ok,rooms,Held.Both{{==},Held.Both{{==},same}},
        run_starts(own,others,end,shared_member(own_starts,other_starts,end,
          runs_member(Tasks.shared_starts(own_starts,other_starts),Tasks.target_blocks(extent,block),end,member))),
        started,ordered,inside)

def starts_rising(+list: List<&2,U32>,+limit: Nat,+starts: Starts(list,limit)) -> Rising(list,0n,limit):
  Held.Both{+head,+rising} = starts
  rising

# Two runs of a target axis read blocks apart in every source.
def axis_runs_apart(+extent: U32,+block: U32,+source: Tasks.Source,+others: List<&2,Tasks.Reading>,+first: U32,+second: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ReadingsRoom(Con{Tasks.Reading{extent,block,source},others},cap),
  +same: SameAxis(others,extent,block),+ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(second),List.length(&2,Tasks.Class,Tasks.axis_classes(Con{Tasks.Reading{extent,block,source},others}))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool}) ->
  AllReadApart(Con{Tasks.Reading{extent,block,source},others},class_word(Tasks.axis_classes(Con{Tasks.Reading{extent,block,source},others}),first),
    class_word(Tasks.axis_classes(Con{Tasks.Reading{extent,block,source},others}),second)):
  +own_starts = Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source}))
  +shared = Tasks.shared_starts(own_starts,Tasks.starts_of(others))
  +limit = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +runs = Tasks.runs(shared,Tasks.target_blocks(extent,block))
  order_all_apart(extent,block,source,others,class_word(runs,first),class_word(runs,second),cap,cap_ok,rooms,same,
    runs_order(shared,Tasks.target_blocks(extent,block),0n,limit,first,second,cap,cap_ok,
      starts_rising(shared,limit,axis_starts(extent,block,source,others,cap,cap_ok,rooms,same)),ordered,inside,room))

# ---- Distinct components read blocks apart at some target axis ----

# Some target axis of a source row reads blocks apart for two class lists.
def FirstApart(row: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,others: List<&2,Tasks.Class>) -> Data:
  match row classes others:
    case Con{Tasks.Reading{+extent,+block,+source},row} Con{+left,classes} Con{+right,others}:
      Either<&2,&2,ReadApart(source,extent,block,left,right),FirstApart(row,classes,others)>
    case row classes others:
      Empty

def AllFirstApart(readings: List<&2,List<&2,Tasks.Reading>>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{+row,rest}:
      Held.Both<FirstApart(row,classes,others),AllFirstApart(rest,classes,others)>

# Every source row has one reading for each target axis.
def RowsAt(readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{+row,rest}:
      Held.Both<{List.length(&2,Tasks.Reading,row) == rank : Nat},RowsAt(rest,rank)>

def zero_successor(+rank: Nat,+same: {0n == 1n+rank : Nat}) -> Empty:
  lt_self(0n,{%Equal.sym(Nat,0n,1n+rank,same) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,0n) == True{} : Bool}})

def length_back(+length: Nat,+rank: Nat,+same: {1n+length == 1n+rank : Nat}) -> {length == rank : Nat}:
  Equal.trans(Nat,length,Nat.sub(1n+length,1n),rank,Equal.sym(Nat,Nat.sub(1n+length,1n),length,Order.subtract_zero(length)),
    Equal.trans(Nat,Nat.sub(1n+length,1n),Nat.sub(1n+rank,1n),rank,Equal.cong(Nat,Nat,value => Nat.sub(value,1n),1n+length,1n+rank,same),Order.subtract_zero(rank)))

def rows_tails(readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+shape: RowsAt(readings,1n+rank)) -> RowsAt(Tasks.tails(readings),rank):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Nil{},+rest}:
      Held.Both{+same,+later} = shape
      Empty.absurd(RowsAt(Tasks.tails(Con{Nil{},rest}),rank),zero_successor(rank,same))
    case Con{Con{+head,+tail},+rest}:
      Held.Both{+same,+later} = shape
      Held.Both{length_back(List.length(&2,Tasks.Reading,tail),rank,same),rows_tails(rest,rank,later)}

def all_here(readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+left: Tasks.Class,+right: Tasks.Class,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,
  +shape: RowsAt(readings,1n+rank),+apart: AllReadApart(Tasks.heads(readings),left,right)) -> AllFirstApart(readings,Con{left,classes},Con{right,others}):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Nil{},+rest}:
      Held.Both{+same,+later} = shape
      Empty.absurd(AllFirstApart(Con{Nil{},rest},Con{left,classes},Con{right,others}),zero_successor(rank,same))
    case Con{Con{Tasks.Reading{+extent,+block,+source},+tail},+rest}:
      Held.Both{+same,+later} = shape
      Held.Both{+here,+rest_apart} = apart
      Held.Both{Inl{here},all_here(rest,rank,left,right,classes,others,later,rest_apart)}

def all_there(readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+left: Tasks.Class,+right: Tasks.Class,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,
  +shape: RowsAt(readings,1n+rank),+later: AllFirstApart(Tasks.tails(readings),classes,others)) -> AllFirstApart(readings,Con{left,classes},Con{right,others}):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Nil{},+rest}:
      Held.Both{+same,+rest_shape} = shape
      Empty.absurd(AllFirstApart(Con{Nil{},rest},Con{left,classes},Con{right,others}),zero_successor(rank,same))
    case Con{Con{Tasks.Reading{+extent,+block,+source},+tail},+rest}:
      Held.Both{+same,+rest_shape} = shape
      Held.Both{+here,+rest_later} = later
      Held.Both{Inr{here},all_there(rest,rank,left,right,classes,others,rest_shape,rest_later)}

def classes_apart_swap(+source: Tasks.Source,+extent: U32,+block: U32,+left: Tasks.Class,+right: Tasks.Class,
  +apart: ClassesApart(source,extent,block,left,right)) -> ClassesApart(source,extent,block,right,left):
  match left right:
    case Tasks.Class{+first,+count,+source_first,+source_count} Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      apart_swap(Tasks.range_footprint(source,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),
        Tasks.range_footprint(source,(later_first * block : U32),(U32.min(((later_first + later_count) * block : U32),extent) - 1 : U32)),apart)

def read_apart_swap(source: Tasks.Source,+extent: U32,+block: U32,+left: Tasks.Class,+right: Tasks.Class,+apart: ReadApart(source,extent,block,left,right)) ->
  ReadApart(source,extent,block,right,left):
  match source:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      classes_apart_swap(Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,left,right,apart)
    case Tasks.Unread{}:
      apart
    case Tasks.Crossing{}:
      apart

def all_read_swap(readings: List<&2,Tasks.Reading>,+left: Tasks.Class,+right: Tasks.Class,+apart: AllReadApart(readings,left,right)) ->
  AllReadApart(readings,right,left):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+extent,+block,+source},+rest}:
      Held.Both{+here,+later} = apart
      Held.Both{read_apart_swap(source,extent,block,left,right,here),all_read_swap(rest,left,right,later)}

# Every column of target axes has room and one extent and block.
def ColumnOk(column: List<&2,Tasks.Reading>,+cap: U32) -> Data:
  match column:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+extent,+block,+source},others}:
      Held.Both<ReadingsRoom(Con{Tasks.Reading{extent,block,source},others},cap),SameAxis(others,extent,block)>

def ColumnsOk(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+cap: U32) -> Data:
  match rank:
    case 0n:
      Unit
    case 1n+rest:
      Held.Both<ColumnOk(Tasks.heads(readings),cap),ColumnsOk(rest,Tasks.tails(readings),cap)>

def column_apart(column: List<&2,Tasks.Reading>,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ColumnOk(column,cap),+ordered: {Nat.is_lt(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(second),List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool}) ->
  AllReadApart(column,class_word(Tasks.axis_classes(column),first),class_word(Tasks.axis_classes(column),second)):
  match column:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+extent,+block,+source},+others}:
      Held.Both{+rooms,+same} = ok
      axis_runs_apart(extent,block,source,others,first,second,cap,cap_ok,rooms,same,ordered,inside,room)

# The components the class lists of the target axes count.
def count_nat(lists: List<&2,List<&2,Tasks.Class>>) -> Nat:
  match lists:
    case Nil{}:
      1n
    case Con{+list,rest}:
      Nat.mul(List.length(&2,Tasks.Class,list),count_nat(rest))

# The class lists of the target axes, outermost first, and the runtime's
# lists, each with its count.
def raw_lists(remaining: Nat,+readings: List<&2,List<&2,Tasks.Reading>>) -> List<&2,List<&2,Tasks.Class>>:
  match remaining:
    case 0n: []
    case 1n+rest: Tasks.axis_classes(Tasks.heads(readings)) <> raw_lists(rest,Tasks.tails(readings))

def counted_all(lists: List<&2,List<&2,Tasks.Class>>) -> List<&2,Tasks.Classes>:
  match lists:
    case Nil{}: []
    case Con{+list,rest}: Tasks.counted(list) <> counted_all(rest)

def lists_counted(remaining: Nat,+readings: List<&2,List<&2,Tasks.Reading>>) ->
  {Tasks.class_lists(remaining,readings) == counted_all(raw_lists(remaining,readings)) : List<&2,Tasks.Classes>}:
  match remaining:
    case 0n:
      {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Tasks.Classes>,List<&2,Tasks.Classes>,tail => Tasks.counted(Tasks.axis_classes(Tasks.heads(readings))) <> tail,
        Tasks.class_lists(rest,Tasks.tails(readings)),counted_all(raw_lists(rest,Tasks.tails(readings))),lists_counted(rest,Tasks.tails(readings)))

# The components class lists count, as a word.
def lists_count(lists: List<&2,List<&2,Tasks.Class>>) -> U32:
  match lists:
    case Nil{}: 1
    case Con{list,rest}: (U32.from_nat(List.length(&2,Tasks.Class,list)) * lists_count(rest) : U32)

def count_counted(lists: List<&2,List<&2,Tasks.Class>>) -> {Tasks.count_of(counted_all(lists)) == lists_count(lists) : U32}:
  match lists:
    case Nil{}:
      {==}
    case Con{+list,+rest}:
      Equal.cong(U32,U32,count => (U32.from_nat(List.length(&2,Tasks.Class,list)) * count : U32),Tasks.count_of(counted_all(rest)),lists_count(rest),count_counted(rest))

# The class of every target axis of one component, outermost first: the
# index of the leading axis is the quotient by the components of the rest.
def chosen_word(lists: List<&2,List<&2,Tasks.Class>>,+index: U32) -> List<&2,Tasks.Class>:
  match lists:
    case Nil{}: []
    case Con{list,+rest}:
      +inner = U32.max(lists_count(rest),1)
      class_word(list,(index / inner : U32)) <> chosen_word(rest,(index % inner : U32))

def ListsRoom(lists: List<&2,List<&2,Tasks.Class>>,+cap: U32) -> Data:
  match lists:
    case Nil{}:
      Unit
    case Con{+list,rest}:
      Held.Both<{Nat.is_le(List.length(&2,Tasks.Class,list),U32.to_nat(cap)) == True{} : Bool},ListsRoom(rest,cap)>

def count_exact(lists: List<&2,List<&2,Tasks.Class>>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: ListsRoom(lists,cap),
  +fits: {Nat.is_le(count_nat(lists),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(lists_count(lists)) == count_nat(lists) : Nat}:
  match lists:
    case Nil{}:
      {==}
    case Con{Nil{},+rest}:
      %Equal.sym(U32,(0 * lists_count(rest) : U32),0,Arithmetic.multiply_zero_left(lists_count(rest))) : {U32.to_nat(_) == 0n : Nat}
      {==}
    case Con{Con{+class,+more},+rest}:
      Held.Both{+length_room,+rest_room} = room
      +length = {1n+List.length(&2,Tasks.Class,more) : Nat}
      +rest_count = count_nat(rest)
      Arithmetic.product_exact(cap,cap_ok,U32.from_nat(length),lists_count(rest),length,rest_count,Bounds.roundtrip(length,cap,cap_ok,length_room),
        count_exact(rest,cap,cap_ok,rest_room,Order.transitive(rest_count,Nat.mul(length,rest_count),U32.to_nat(cap),
          Order.le_add(rest_count,Nat.mul(List.length(&2,Tasks.Class,more),rest_count)),fits)),fits)

def same_zero(+first: U32,+second: U32,+first_inside: {Nat.is_lt(U32.to_nat(first),1n) == True{} : Bool},
  +second_inside: {Nat.is_lt(U32.to_nat(second),1n) == True{} : Bool},+differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool}) -> Empty:
  +first_zero = Order.antisymmetric(U32.to_nat(first),0n,Division.lt_succ_le(U32.to_nat(first),0n,first_inside),Order.zero_le(U32.to_nat(first)))
  +second_zero = Order.antisymmetric(U32.to_nat(second),0n,Division.lt_succ_le(U32.to_nat(second),0n,second_inside),Order.zero_le(U32.to_nat(second)))
  WordEncoding.false_true(Equal.sym(Bool,True{},False{},
    {%second_zero : {Nat.is_eq(0n,_) == False{} : Bool}
      {%first_zero : {Nat.is_eq(_,U32.to_nat(second)) == False{} : Bool} differ : {Nat.is_eq(0n,U32.to_nat(second)) == False{} : Bool}} :
      {Nat.is_eq(0n,0n) == False{} : Bool}}))

def strict_absurd(+low: Nat,+high: Nat,+le: {Nat.is_le(low,high) == True{} : Bool},+ne: {Nat.is_eq(high,low) == False{} : Bool},
  +not_lt: {Nat.is_lt(low,high) == False{} : Bool}) -> Empty:
  +same = Order.antisymmetric(high,low,Held.lt_false_le(low,high,not_lt),le)
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(high,low),True{},Equal.sym(Bool,Nat.is_eq(high,low),False{},ne),
    {%Equal.sym(Nat,high,low,same) : {Nat.is_eq(_,low) == True{} : Bool} Order.equal_reflexive(low) : {Nat.is_eq(high,low) == True{} : Bool}}))

def strict_of(+low: Nat,+high: Nat,+le: {Nat.is_le(low,high) == True{} : Bool},+ne: {Nat.is_eq(high,low) == False{} : Bool}) ->
  {Nat.is_lt(low,high) == True{} : Bool}:
  Held.by_cases({Nat.is_lt(low,high) == True{} : Bool},Nat.is_lt(low,high),
    lt => lt,
    not_lt => Empty.absurd({Nat.is_lt(low,high) == True{} : Bool},strict_absurd(low,high,le,ne,not_lt)))

def remainder_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+divisor: U32,+divisor_value: Nat,
  +divisor_exact: {U32.to_nat(divisor) == divisor_value : Nat},+divisor_room: {Nat.is_le(Nat.double(divisor_value),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor_value) == True{} : Bool}) -> {U32.to_nat((value % divisor : U32)) == Nat.mod(U32.to_nat(value),divisor_value) : Nat}:
  %divisor_exact : {U32.to_nat((value % divisor : U32)) == Nat.mod(U32.to_nat(value),_) : Nat}
  U32Division.modulo(value,divisor,cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(divisor),divisor_value,divisor_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} divisor_room :
      {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(cap)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(divisor),divisor_value,divisor_exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}})

def differ_after(+column: List<&2,Tasks.Reading>,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ColumnOk(column,cap),+first_inside: {Nat.is_lt(U32.to_nat(first),List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(cap)) == True{} : Bool},+differ: {U32.is_eq(first,second) == False{} : Bool},
  +after: {U32.is_lt(first,second) == False{} : Bool}) ->
  AllReadApart(column,class_word(Tasks.axis_classes(column),first),class_word(Tasks.axis_classes(column),second)):
  +ne = Equal.trans(Bool,Nat.is_eq(U32.to_nat(first),U32.to_nat(second)),U32.is_eq(first,second),False{},
    Equal.sym(Bool,U32.is_eq(first,second),Nat.is_eq(U32.to_nat(first),U32.to_nat(second)),Comparison.eq(first,second)),differ)
  all_read_swap(column,class_word(Tasks.axis_classes(column),second),class_word(Tasks.axis_classes(column),first),
    column_apart(column,second,first,cap,cap_ok,ok,strict_of(U32.to_nat(second),U32.to_nat(first),nat_lt_false(first,second,after),ne),first_inside,first_room))

# Two classes of a target axis at distinct indices read blocks apart in every
# source.
def differ_apart(+column: List<&2,Tasks.Reading>,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ColumnOk(column,cap),+first_inside: {Nat.is_lt(U32.to_nat(first),List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
  +second_inside: {Nat.is_lt(U32.to_nat(second),List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(cap)) == True{} : Bool},+second_room: {Nat.is_le(U32.to_nat(second),U32.to_nat(cap)) == True{} : Bool},
  +differ: {U32.is_eq(first,second) == False{} : Bool}) ->
  AllReadApart(column,class_word(Tasks.axis_classes(column),first),class_word(Tasks.axis_classes(column),second)):
  Held.by_cases(AllReadApart(column,class_word(Tasks.axis_classes(column),first),class_word(Tasks.axis_classes(column),second)),U32.is_lt(first,second),
    before => column_apart(column,first,second,cap,cap_ok,ok,word_lt(first,second,before),second_inside,second_room),
    after => differ_after(column,first,second,cap,cap_ok,ok,first_inside,first_room,differ,after))

def remainders_differ(+first: U32,+second: U32,+divisor: Nat,+first_quotient: U32,+second_quotient: U32,+first_rest: U32,+second_rest: U32,
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +first_exact: {U32.to_nat(first_quotient) == Nat.div(U32.to_nat(first),divisor) : Nat},
  +second_exact: {U32.to_nat(second_quotient) == Nat.div(U32.to_nat(second),divisor) : Nat},
  +first_rest_exact: {U32.to_nat(first_rest) == Nat.mod(U32.to_nat(first),divisor) : Nat},
  +second_rest_exact: {U32.to_nat(second_rest) == Nat.mod(U32.to_nat(second),divisor) : Nat},
  +same: {first_quotient == second_quotient : U32},+differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool},
  +equal: {Nat.is_eq(U32.to_nat(first_rest),U32.to_nat(second_rest)) == True{} : Bool}) -> Empty:
  +low = U32.to_nat(first)
  +high = U32.to_nat(second)
  +quotients = Equal.trans(Nat,Nat.div(low,divisor),U32.to_nat(first_quotient),Nat.div(high,divisor),Equal.sym(Nat,U32.to_nat(first_quotient),Nat.div(low,divisor),first_exact),
    Equal.trans(Nat,U32.to_nat(first_quotient),U32.to_nat(second_quotient),Nat.div(high,divisor),Equal.cong(U32,Nat,word => U32.to_nat(word),first_quotient,second_quotient,same),second_exact))
  +remainders = Equal.trans(Nat,Nat.mod(low,divisor),U32.to_nat(first_rest),Nat.mod(high,divisor),Equal.sym(Nat,U32.to_nat(first_rest),Nat.mod(low,divisor),first_rest_exact),
    Equal.trans(Nat,U32.to_nat(first_rest),U32.to_nat(second_rest),Nat.mod(high,divisor),Order.equal_of_is_eq(U32.to_nat(first_rest),U32.to_nat(second_rest),equal),second_rest_exact))
  +low_parts = Nat.add(Nat.mul(Nat.div(low,divisor),divisor),Nat.mod(low,divisor))
  +high_parts = Nat.add(Nat.mul(Nat.div(high,divisor),divisor),Nat.mod(high,divisor))
  +parts = {%quotients : {low_parts == Nat.add(Nat.mul(_,divisor),Nat.mod(high,divisor)) : Nat}
    %remainders : {low_parts == Nat.add(Nat.mul(Nat.div(low,divisor),divisor),_) : Nat}
    {==} : {low_parts == high_parts : Nat}}
  +values = Equal.trans(Nat,low,low_parts,high,DivisionOrder.quotient_exact(low,divisor,positive),
    Equal.trans(Nat,low_parts,high_parts,high,parts,Equal.sym(Nat,high,high_parts,DivisionOrder.quotient_exact(high,divisor,positive))))
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(low,high),True{},Equal.sym(Bool,Nat.is_eq(low,high),False{},differ),
    {%values : {Nat.is_eq(low,_) == True{} : Bool} Order.equal_reflexive(low) : {Nat.is_eq(low,high) == True{} : Bool}}))

# The divisor that picks the class of the first target axis of a component.
def inner_of(+lists: List<&2,List<&2,Tasks.Class>>) -> U32:
  U32.max(lists_count(lists),1)

# The runtime picks the classes innermost first: the index left after the
# inner axes is the quotient by their components.
def word_over_one(+value: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+two: {Nat.is_le(2n,U32.to_nat(cap)) == True{} : Bool}) ->
  {(value / 1 : U32) == value : U32}:
  Arithmetic.to_nat_injective((value / 1 : U32),value,
    Equal.trans(Nat,U32.to_nat((value / 1 : U32)),Nat.div(U32.to_nat(value),1n),U32.to_nat(value),
      U32Division.quotient_exact(cap,cap_ok,value,1,U32.to_nat(value),1n,{==},{==},two,{==}),
      {%Equal.sym(Nat,U32.to_nat(value),Nat.add(Nat.mul(U32.to_nat(value),1n),0n),
          Equal.trans(Nat,U32.to_nat(value),Nat.mul(U32.to_nat(value),1n),Nat.add(Nat.mul(U32.to_nat(value),1n),0n),
            Equal.sym(Nat,Nat.mul(U32.to_nat(value),1n),U32.to_nat(value),Order.times_one(U32.to_nat(value))),
            Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(value),1n),0n),Nat.mul(U32.to_nat(value),1n),Addition.zero_right(Nat.mul(U32.to_nat(value),1n))))) :
          {Nat.div(_,1n) == U32.to_nat(value) : Nat}
        Radix.quotient(U32.to_nat(value),0n,1n,{==},{==}) : {Nat.div(U32.to_nat(value),1n) == U32.to_nat(value) : Nat}}))

def chosen_pair(+left: List<&2,Tasks.Class>,+right: List<&2,Tasks.Class>,+first: U32,+second: U32,
  +classes_same: {left == right : List<&2,Tasks.Class>},+index_same: {first == second : U32}) ->
  {Tasks.Chosen{left,first} == Tasks.Chosen{right,second} : Tasks.Chosen}:
  %classes_same : {Tasks.Chosen{left,first} == Tasks.Chosen{_,second} : Tasks.Chosen}
  %index_same : {Tasks.Chosen{left,first} == Tasks.Chosen{left,_} : Tasks.Chosen}
  {==}

def chosen_from_exact(lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ListsRoom(lists,cap),
  +room: {Nat.is_le(Nat.double(count_nat(lists)),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,count_nat(lists)) == True{} : Bool}) ->
  {Tasks.chosen_from(counted_all(lists),index) == Tasks.Chosen{chosen_word(lists,(index % inner_of(lists) : U32)),(index / inner_of(lists) : U32)} : Tasks.Chosen}:
  match lists:
    case Nil{}:
      %Equal.sym(U32,(index / 1 : U32),index,word_over_one(index,cap,cap_ok,room)) : {Tasks.Chosen{[],index} == Tasks.Chosen{[],_} : Tasks.Chosen}
      {==}
    case Con{+list,+rest}:
      Held.Both{+length_room,+rest_rooms} = rooms
      +value = U32.to_nat(index)
      +length = List.length(&2,Tasks.Class,list)
      +divisor = count_nat(rest)
      +total = Nat.mul(length,divisor)
      +positive = DivisionOrder.right_factor_positive(length,divisor,filled)
      +length_positive = RowMajor.left_factor_positive(length,divisor,filled)
      +divisor_le = {%RowMajor.multiply_commutative(divisor,length) : {Nat.is_le(divisor,_) == True{} : Bool}
        RowMajor.product_covers(divisor,length,length_positive) : {Nat.is_le(divisor,total) == True{} : Bool}}
      +length_le = RowMajor.product_covers(length,divisor,positive)
      +total_room = Order.transitive(total,Nat.double(total),U32.to_nat(cap),Order.le_double(total),room)
      +rest_room = Order.transitive(Nat.double(divisor),Nat.double(total),U32.to_nat(cap),Order.double_le(divisor,total,divisor_le),room)
      +length_double = Order.transitive(Nat.double(length),Nat.double(total),U32.to_nat(cap),Order.double_le(length,total,length_le),room)
      +inner = inner_of(rest)
      +outer = U32.max(U32.from_nat(length),1)
      +whole = inner_of(Con{list,rest})
      +inner_value = Extrema.max_one_exact(lists_count(rest),divisor,
        count_exact(rest,cap,cap_ok,rest_rooms,Order.transitive(divisor,total,U32.to_nat(cap),divisor_le,total_room)),positive)
      +outer_value = Extrema.max_one_exact(U32.from_nat(length),length,Bounds.roundtrip(length,cap,cap_ok,length_room),length_positive)
      +whole_value = Extrema.max_one_exact(lists_count(Con{list,rest}),total,count_exact(Con{list,rest},cap,cap_ok,rooms,total_room),filled)
      +below = U32Division.quotient_exact(cap,cap_ok,index,inner,value,divisor,{==},inner_value,rest_room,positive)
      +picked = Equal.trans(Nat,U32.to_nat(((index / inner : U32) % outer : U32)),Nat.mod(U32.to_nat((index / inner : U32)),length),Nat.mod(Nat.div(value,divisor),length),
        remainder_exact(cap,cap_ok,(index / inner : U32),outer,length,outer_value,length_double,length_positive),
        Equal.cong(Nat,Nat,number => Nat.mod(number,length),U32.to_nat((index / inner : U32)),Nat.div(value,divisor),below))
      +within = remainder_exact(cap,cap_ok,index,whole,total,whole_value,room,filled)
      +lead = Equal.trans(Nat,U32.to_nat(((index % whole : U32) / inner : U32)),Nat.div(Nat.mod(value,total),divisor),Nat.mod(Nat.div(value,divisor),length),
        U32Division.quotient_exact(cap,cap_ok,(index % whole : U32),inner,Nat.mod(value,total),divisor,within,inner_value,rest_room,positive),
        DivisionOrder.remainder_quotient(value,length,divisor,length_positive,positive))
      +lead_room = Order.transitive(U32.to_nat(((index % whole : U32) / inner : U32)),length,U32.to_nat(cap),
        lt_to_le(U32.to_nat(((index % whole : U32) / inner : U32)),length,
          {%Equal.sym(Nat,U32.to_nat(((index % whole : U32) / inner : U32)),Nat.mod(Nat.div(value,divisor),length),lead) : {Nat.is_lt(_,length) == True{} : Bool}
            DivisionOrder.remainder_small(Nat.div(value,divisor),length,length_positive) : {Nat.is_lt(U32.to_nat(((index % whole : U32) / inner : U32)),length) == True{} : Bool}}),
        length_room)
      +class_same = Equal.trans(Tasks.Class,Tasks.class_at(list,U32.to_nat(((index / inner : U32) % outer : U32))),Tasks.class_at(list,U32.to_nat(((index % whole : U32) / inner : U32))),
        class_word(list,((index % whole : U32) / inner : U32)),
        Equal.cong(Nat,Tasks.Class,number => Tasks.class_at(list,number),U32.to_nat(((index / inner : U32) % outer : U32)),U32.to_nat(((index % whole : U32) / inner : U32)),
          Equal.trans(Nat,U32.to_nat(((index / inner : U32) % outer : U32)),Nat.mod(Nat.div(value,divisor),length),U32.to_nat(((index % whole : U32) / inner : U32)),picked,
            Equal.sym(Nat,U32.to_nat(((index % whole : U32) / inner : U32)),Nat.mod(Nat.div(value,divisor),length),lead))),
        class_word_exact(list,((index % whole : U32) / inner : U32),cap,cap_ok,lead_room))
      +rest_same = Arithmetic.to_nat_injective((index % inner : U32),((index % whole : U32) % inner : U32),
        Equal.trans(Nat,U32.to_nat((index % inner : U32)),Nat.mod(value,divisor),U32.to_nat(((index % whole : U32) % inner : U32)),
          remainder_exact(cap,cap_ok,index,inner,divisor,inner_value,rest_room,positive),
          Equal.sym(Nat,U32.to_nat(((index % whole : U32) % inner : U32)),Nat.mod(value,divisor),
            Equal.trans(Nat,U32.to_nat(((index % whole : U32) % inner : U32)),Nat.mod(U32.to_nat((index % whole : U32)),divisor),Nat.mod(value,divisor),
              remainder_exact(cap,cap_ok,(index % whole : U32),inner,divisor,inner_value,rest_room,positive),
              Equal.trans(Nat,Nat.mod(U32.to_nat((index % whole : U32)),divisor),Nat.mod(Nat.mod(value,total),divisor),Nat.mod(value,divisor),
                Equal.cong(Nat,Nat,number => Nat.mod(number,divisor),U32.to_nat((index % whole : U32)),Nat.mod(value,total),within),
                DivisionOrder.remainder_remainder(value,length,divisor,length_positive,positive))))))
      +index_same = Arithmetic.to_nat_injective(((index / inner : U32) / outer : U32),(index / whole : U32),
        Equal.trans(Nat,U32.to_nat(((index / inner : U32) / outer : U32)),Nat.div(Nat.div(value,divisor),length),U32.to_nat((index / whole : U32)),
          U32Division.quotient_exact(cap,cap_ok,(index / inner : U32),outer,Nat.div(value,divisor),length,below,outer_value,length_double,length_positive),
          Equal.sym(Nat,U32.to_nat((index / whole : U32)),Nat.div(Nat.div(value,divisor),length),
            Equal.trans(Nat,U32.to_nat((index / whole : U32)),Nat.div(value,total),Nat.div(Nat.div(value,divisor),length),
              U32Division.quotient_exact(cap,cap_ok,index,whole,value,total,{==},whole_value,room,filled),
              DivisionOrder.product_quotient(value,length,divisor,length_positive,positive)))))
      %Equal.sym(Tasks.Chosen,Tasks.chosen_from(counted_all(rest),index),Tasks.Chosen{chosen_word(rest,(index % inner : U32)),(index / inner : U32)},
          chosen_from_exact(rest,index,cap,cap_ok,rest_rooms,rest_room,positive)) :
        {Tasks.chosen_into(Tasks.counted(list),_) == Tasks.Chosen{chosen_word(Con{list,rest},(index % whole : U32)),(index / whole : U32)} : Tasks.Chosen}
      chosen_pair(Tasks.class_at(list,U32.to_nat(((index / inner : U32) % outer : U32))) <> chosen_word(rest,(index % inner : U32)),
        class_word(list,((index % whole : U32) / inner : U32)) <> chosen_word(rest,((index % whole : U32) % inner : U32)),
        ((index / inner : U32) / outer : U32),(index / whole : U32),
        Equal.trans(List<&2,Tasks.Class>,Tasks.class_at(list,U32.to_nat(((index / inner : U32) % outer : U32))) <> chosen_word(rest,(index % inner : U32)),
          class_word(list,((index % whole : U32) / inner : U32)) <> chosen_word(rest,(index % inner : U32)),
          class_word(list,((index % whole : U32) / inner : U32)) <> chosen_word(rest,((index % whole : U32) % inner : U32)),
          Equal.cong(Tasks.Class,List<&2,Tasks.Class>,head => head <> chosen_word(rest,(index % inner : U32)),
            Tasks.class_at(list,U32.to_nat(((index / inner : U32) % outer : U32))),class_word(list,((index % whole : U32) / inner : U32)),class_same),
          Equal.cong(U32,List<&2,Tasks.Class>,number => class_word(list,((index % whole : U32) / inner : U32)) <> chosen_word(rest,number),
            (index % inner : U32),((index % whole : U32) % inner : U32),rest_same)),
        index_same)

# Inside the components, the runtime's classes of a component are the model's.
def chosen_exact(+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ListsRoom(lists,cap),
  +room: {Nat.is_le(Nat.double(count_nat(lists)),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(index),count_nat(lists)) == True{} : Bool}) ->
  {Tasks.chosen(counted_all(lists),index) == chosen_word(lists,index) : List<&2,Tasks.Class>}:
  +filled = Order.le_lt_trans(0n,U32.to_nat(index),count_nat(lists),Order.zero_le(U32.to_nat(index)),inside)
  +total_room = Order.transitive(count_nat(lists),Nat.double(count_nat(lists)),U32.to_nat(cap),Order.le_double(count_nat(lists)),room)
  +whole_value = Extrema.max_one_exact(lists_count(lists),count_nat(lists),count_exact(lists,cap,cap_ok,rooms,total_room),filled)
  +kept = Arithmetic.to_nat_injective((index % inner_of(lists) : U32),index,
    Equal.trans(Nat,U32.to_nat((index % inner_of(lists) : U32)),Nat.mod(U32.to_nat(index),count_nat(lists)),U32.to_nat(index),
      remainder_exact(cap,cap_ok,index,inner_of(lists),count_nat(lists),whole_value,room,filled),
      DivisionOrder.small_remainder(U32.to_nat(index),count_nat(lists),inside)))
  %Equal.sym(Tasks.Chosen,Tasks.chosen_from(counted_all(lists),index),
      Tasks.Chosen{chosen_word(lists,(index % inner_of(lists) : U32)),(index / inner_of(lists) : U32)},chosen_from_exact(lists,index,cap,cap_ok,rooms,room,filled)) :
    {Tasks.chosen_classes(_) == chosen_word(lists,index) : List<&2,Tasks.Class>}
  %Equal.sym(U32,(index % inner_of(lists) : U32),index,kept) : {chosen_word(lists,_) == chosen_word(lists,index) : List<&2,Tasks.Class>}
  {==}

# The runtime's classes of a component of the class lists of readings.
def classes_chosen(+rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +lists_room: ListsRoom(raw_lists(rank,readings),cap),
  +lists_double: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Tasks.chosen(Tasks.class_lists(rank,readings),index) == chosen_word(raw_lists(rank,readings),index) : List<&2,Tasks.Class>}:
  Equal.trans(List<&2,Tasks.Class>,Tasks.chosen(Tasks.class_lists(rank,readings),index),Tasks.chosen(counted_all(raw_lists(rank,readings)),index),
    chosen_word(raw_lists(rank,readings),index),
    Equal.cong(List<&2,Tasks.Classes>,List<&2,Tasks.Class>,lists => Tasks.chosen(lists,index),Tasks.class_lists(rank,readings),counted_all(raw_lists(rank,readings)),
      lists_counted(rank,readings)),
    chosen_exact(raw_lists(rank,readings),index,cap,cap_ok,lists_room,lists_double,inside))

# The runtime's count of the class lists of readings.
def classes_count(+rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>) ->
  {Tasks.count_of(Tasks.class_lists(rank,readings)) == lists_count(raw_lists(rank,readings)) : U32}:
  Equal.trans(U32,Tasks.count_of(Tasks.class_lists(rank,readings)),Tasks.count_of(counted_all(raw_lists(rank,readings))),lists_count(raw_lists(rank,readings)),
    Equal.cong(List<&2,Tasks.Classes>,U32,lists => Tasks.count_of(lists),Tasks.class_lists(rank,readings),counted_all(raw_lists(rank,readings)),
      lists_counted(rank,readings)),
    count_counted(raw_lists(rank,readings)))

def chosen_same(+readings: List<&2,List<&2,Tasks.Reading>>,+rest: Nat,+first: U32,+second: U32,+divisor: Nat,
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+shape: RowsAt(readings,1n+rest),
  +first_exact: {U32.to_nat((first / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == Nat.div(U32.to_nat(first),divisor) : Nat},
  +second_exact: {U32.to_nat((second / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == Nat.div(U32.to_nat(second),divisor) : Nat},
  +first_rest_exact: {U32.to_nat((first % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == Nat.mod(U32.to_nat(first),divisor) : Nat},
  +second_rest_exact: {U32.to_nat((second % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == Nat.mod(U32.to_nat(second),divisor) : Nat},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool},
  +same: {U32.is_eq((first / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32),(second / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == True{} : Bool},
  recurse: {Nat.is_eq(U32.to_nat((first % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),
      U32.to_nat((second % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32))) == False{} : Bool} ->
    AllFirstApart(Tasks.tails(readings),chosen_word(raw_lists(rest,Tasks.tails(readings)),(first % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),
      chosen_word(raw_lists(rest,Tasks.tails(readings)),(second % inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)))) ->
  AllFirstApart(readings,chosen_word(raw_lists(1n+rest,readings),first),chosen_word(raw_lists(1n+rest,readings),second)):
  +lists = raw_lists(rest,Tasks.tails(readings))
  +inner = inner_of(lists)
  +list = Tasks.axis_classes(Tasks.heads(readings))
  +first_rest = (first % inner : U32)
  +second_rest = (second % inner : U32)
  +rest_differ = Held.by_cases({Nat.is_eq(U32.to_nat(first_rest),U32.to_nat(second_rest)) == False{} : Bool},Nat.is_eq(U32.to_nat(first_rest),U32.to_nat(second_rest)),
    equal => Empty.absurd({Nat.is_eq(U32.to_nat(first_rest),U32.to_nat(second_rest)) == False{} : Bool},
      remainders_differ(first,second,divisor,(first / inner : U32),(second / inner : U32),first_rest,second_rest,positive,first_exact,second_exact,
        first_rest_exact,second_rest_exact,Arithmetic.equal_of_is_eq((first / inner : U32),(second / inner : U32),same),differ,equal)),
    unequal => unequal)
  all_there(readings,rest,class_word(list,(first / inner : U32)),class_word(list,(second / inner : U32)),chosen_word(lists,first_rest),chosen_word(lists,second_rest),
    shape,recurse(rest_differ))

def chosen_differ(+readings: List<&2,List<&2,Tasks.Reading>>,+rest: Nat,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +shape: RowsAt(readings,1n+rest),+ok: ColumnOk(Tasks.heads(readings),cap),
  +first_inside: {Nat.is_lt(U32.to_nat((first / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),
    List.length(&2,Tasks.Class,Tasks.axis_classes(Tasks.heads(readings)))) == True{} : Bool},
  +second_inside: {Nat.is_lt(U32.to_nat((second / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),
    List.length(&2,Tasks.Class,Tasks.axis_classes(Tasks.heads(readings)))) == True{} : Bool},
  +first_room: {Nat.is_le(U32.to_nat((first / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),U32.to_nat(cap)) == True{} : Bool},
  +second_room: {Nat.is_le(U32.to_nat((second / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),U32.to_nat(cap)) == True{} : Bool},
  +differ: {U32.is_eq((first / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32),(second / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)) == False{} : Bool}) ->
  AllFirstApart(readings,chosen_word(raw_lists(1n+rest,readings),first),chosen_word(raw_lists(1n+rest,readings),second)):
  +lists = raw_lists(rest,Tasks.tails(readings))
  +inner = inner_of(lists)
  +column = Tasks.heads(readings)
  +list = Tasks.axis_classes(column)
  all_here(readings,rest,class_word(list,(first / inner : U32)),class_word(list,(second / inner : U32)),
    chosen_word(lists,(first % inner : U32)),chosen_word(lists,(second % inner : U32)),shape,
    differ_apart(column,(first / inner : U32),(second / inner : U32),cap,cap_ok,ok,first_inside,second_inside,first_room,second_room,differ))

# Two distinct components choose classes apart at some target axis, where
# every source reads blocks apart.
def chosen_apart(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+first: U32,+second: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +shape: RowsAt(readings,rank),+columns: ColumnsOk(rank,readings,cap),+lists_room: ListsRoom(raw_lists(rank,readings),cap),
  +room: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
  +first_inside: {Nat.is_lt(U32.to_nat(first),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +second_inside: {Nat.is_lt(U32.to_nat(second),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool}) ->
  AllFirstApart(readings,chosen_word(raw_lists(rank,readings),first),chosen_word(raw_lists(rank,readings),second)):
  match rank:
    case 0n:
      Empty.absurd(AllFirstApart(readings,Nil{},Nil{}),same_zero(first,second,first_inside,second_inside,differ))
    case 1n+ +rest:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +length = List.length(&2,Tasks.Class,list)
      +divisor = count_nat(lists)
      +total = Nat.mul(length,divisor)
      +inner = inner_of(lists)
      +low = U32.to_nat(first)
      +high = U32.to_nat(second)
      +total_positive = Order.le_lt_trans(0n,low,total,Order.zero_le(low),first_inside)
      +positive = DivisionOrder.right_factor_positive(length,divisor,total_positive)
      +length_positive = RowMajor.left_factor_positive(length,divisor,total_positive)
      +divisor_le = {%RowMajor.multiply_commutative(divisor,length) : {Nat.is_le(divisor,_) == True{} : Bool}
        RowMajor.product_covers(divisor,length,length_positive) : {Nat.is_le(divisor,total) == True{} : Bool}}
      +total_room = Order.transitive(total,Nat.double(total),U32.to_nat(cap),Order.le_double(total),room)
      +divisor_room = Order.transitive(divisor,total,U32.to_nat(cap),divisor_le,total_room)
      +double_room = Order.transitive(Nat.double(divisor),Nat.double(total),U32.to_nat(cap),Order.double_le(divisor,total,divisor_le),room)
      +count_value = count_exact(lists,cap,cap_ok,rest_room,divisor_room)
      +inner_value = Extrema.max_one_exact(lists_count(lists),divisor,count_value,positive)
      +first_exact = U32Division.quotient_exact(cap,cap_ok,first,inner,low,divisor,{==},inner_value,double_room,positive)
      +second_exact = U32Division.quotient_exact(cap,cap_ok,second,inner,high,divisor,{==},inner_value,double_room,positive)
      +first_rest_exact = remainder_exact(cap,cap_ok,first,inner,divisor,inner_value,double_room,positive)
      +second_rest_exact = remainder_exact(cap,cap_ok,second,inner,divisor,inner_value,double_room,positive)
      +first_quotient = (first / inner : U32)
      +second_quotient = (second / inner : U32)
      +first_quotient_inside = {%Equal.sym(Nat,U32.to_nat(first_quotient),Nat.div(low,divisor),first_exact) : {Nat.is_lt(_,length) == True{} : Bool}
        DivisionOrder.quotient_below(low,length,divisor,positive,first_inside) : {Nat.is_lt(U32.to_nat(first_quotient),length) == True{} : Bool}}
      +second_quotient_inside = {%Equal.sym(Nat,U32.to_nat(second_quotient),Nat.div(high,divisor),second_exact) : {Nat.is_lt(_,length) == True{} : Bool}
        DivisionOrder.quotient_below(high,length,divisor,positive,second_inside) : {Nat.is_lt(U32.to_nat(second_quotient),length) == True{} : Bool}}
      +first_quotient_room = Order.transitive(U32.to_nat(first_quotient),length,U32.to_nat(cap),lt_to_le(U32.to_nat(first_quotient),length,first_quotient_inside),list_room)
      +second_quotient_room = Order.transitive(U32.to_nat(second_quotient),length,U32.to_nat(cap),lt_to_le(U32.to_nat(second_quotient),length,second_quotient_inside),list_room)
      +first_rest = (first % inner : U32)
      +second_rest = (second % inner : U32)
      +first_rest_inside = {%Equal.sym(Nat,U32.to_nat(first_rest),Nat.mod(low,divisor),first_rest_exact) : {Nat.is_lt(_,divisor) == True{} : Bool}
        DivisionOrder.remainder_small(low,divisor,positive) : {Nat.is_lt(U32.to_nat(first_rest),divisor) == True{} : Bool}}
      +second_rest_inside = {%Equal.sym(Nat,U32.to_nat(second_rest),Nat.mod(high,divisor),second_rest_exact) : {Nat.is_lt(_,divisor) == True{} : Bool}
        DivisionOrder.remainder_small(high,divisor,positive) : {Nat.is_lt(U32.to_nat(second_rest),divisor) == True{} : Bool}}
      +tail_shape = rows_tails(readings,rest,shape)
      Held.by_cases(AllFirstApart(readings,chosen_word(raw_lists(1n+rest,readings),first),chosen_word(raw_lists(1n+rest,readings),second)),
        U32.is_eq(first_quotient,second_quotient),
        same => chosen_same(readings,rest,first,second,divisor,positive,shape,first_exact,second_exact,first_rest_exact,second_rest_exact,differ,same,
          rest_differ => chosen_apart(rest,Tasks.tails(readings),first_rest,second_rest,cap,cap_ok,tail_shape,rest_columns,rest_room,double_room,
            first_rest_inside,second_rest_inside,rest_differ)),
        unequal => chosen_differ(readings,rest,first,second,cap,cap_ok,shape,column_ok,first_quotient_inside,second_quotient_inside,
          first_quotient_room,second_quotient_room,unequal))

# ---- Row-major piece numbers of a box ----

# The positions of a box of spans.
def span_total(spans: List<&2,Tasks.Span>) -> Nat:
  match spans:
    case Nil{}:
      1n
    case Con{Tasks.Span{+first,+count},rest}:
      Nat.mul(U32.to_nat(count),span_total(rest))

# The row-major number Tasks.piece_at gives position index of a box, after
# the value of the axes before it.
def row_value(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat,+value: Nat) -> Nat:
  match spans extents:
    case Con{Tasks.Span{+first,+count},+spans} Con{+extent,extents}:
      +inner = span_total(spans)
      row_value(spans,extents,Nat.mod(index,inner),Nat.add(Nat.add(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(first)),Nat.div(index,inner)))
    case spans extents:
      value

# Every span lies inside its axis.
def SpansIn(spans: List<&2,Tasks.Span>,extents: List<&2,U32>) -> Data:
  match spans extents:
    case Nil{} Nil{}:
      Unit
    case Con{Tasks.Span{+first,+count},spans} Con{+extent,extents}:
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent)) == True{} : Bool},SpansIn(spans,extents)>
    case spans extents:
      Empty

# The value before a box scales by the size of the axes it spans.
def row_split(spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: Nat,+value: Nat,+within: SpansIn(spans,extents)) ->
  {row_value(spans,extents,index,value) == Nat.add(Nat.mul(value,ShapeLaws.elements(extents)),row_value(spans,extents,index,0n)) : Nat}:
  match spans extents:
    case Nil{} Nil{}:
      Equal.sym(Nat,Nat.add(Nat.mul(value,1n),0n),value,Equal.trans(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),value,Addition.zero_right(Nat.mul(value,1n)),Order.times_one(value)))
    case Nil{} Con{+extent,+more}:
      Empty.absurd({row_value(Nil{},Con{extent,more},index,value) == Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,more})),row_value(Nil{},Con{extent,more},index,0n)) : Nat},within)
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      Empty.absurd({row_value(Con{Tasks.Span{first,count},rest},Nil{},index,value) == Nat.add(Nat.mul(value,1n),row_value(Con{Tasks.Span{first,count},rest},Nil{},index,0n)) : Nat},within)
    case Con{Tasks.Span{+first,+count},+rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      +inner = span_total(rest)
      +later = ShapeLaws.elements(more)
      +position = Nat.mod(index,inner)
      +block = Nat.div(index,inner)
      +start = U32.to_nat(first)
      +size = U32.to_nat(extent)
      +scaled = Nat.mul(value,size)
      +coordinate = Nat.add(start,block)
      +tail = row_value(rest,more,position,0n)
      +own = row_split(rest,more,position,Nat.add(Nat.add(scaled,start),block),rest_within)
      +zero = row_split(rest,more,position,coordinate,rest_within)
      +grouped = {%Equal.sym(Nat,Nat.add(Nat.add(scaled,start),block),Nat.add(scaled,coordinate),Addition.associative(scaled,start,block)) :
          {Nat.mul(_,later) == Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later)) : Nat}
        %RowMajor.assoc(value,size,later) : {Nat.mul(Nat.add(scaled,coordinate),later) == Nat.add(_,Nat.mul(coordinate,later)) : Nat}
        RowMajor.distrib(scaled,coordinate,later) :
        {Nat.mul(Nat.add(Nat.add(scaled,start),block),later) == Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later)) : Nat}}
      Equal.trans(Nat,row_value(rest,more,position,Nat.add(Nat.add(scaled,start),block)),
        Nat.add(Nat.mul(Nat.add(Nat.add(scaled,start),block),later),tail),
        Nat.add(Nat.mul(value,Nat.mul(size,later)),row_value(rest,more,position,coordinate)),own,
        Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.add(scaled,start),block),later),tail),
          Nat.add(Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later)),tail),
          Nat.add(Nat.mul(value,Nat.mul(size,later)),row_value(rest,more,position,coordinate)),
          Equal.cong(Nat,Nat,head => Nat.add(head,tail),Nat.mul(Nat.add(Nat.add(scaled,start),block),later),
            Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later)),grouped),
          Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later)),tail),
            Nat.add(Nat.mul(value,Nat.mul(size,later)),Nat.add(Nat.mul(coordinate,later),tail)),
            Nat.add(Nat.mul(value,Nat.mul(size,later)),row_value(rest,more,position,coordinate)),
            Addition.associative(Nat.mul(value,Nat.mul(size,later)),Nat.mul(coordinate,later),tail),
            Equal.cong(Nat,Nat,rest_value => Nat.add(Nat.mul(value,Nat.mul(size,later)),rest_value),Nat.add(Nat.mul(coordinate,later),tail),
              row_value(rest,more,position,coordinate),Equal.sym(Nat,row_value(rest,more,position,coordinate),Nat.add(Nat.mul(coordinate,later),tail),zero)))))

def count_positive(+count: Nat,+inner: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(count,inner)) == True{} : Bool}) -> {Nat.is_lt(0n,count) == True{} : Bool}:
  RowMajor.left_factor_positive(count,inner,Order.le_lt_trans(0n,index,Nat.mul(count,inner),Order.zero_le(index),inside))

def inner_positive(+count: Nat,+inner: Nat,+index: Nat,+inside: {Nat.is_lt(index,Nat.mul(count,inner)) == True{} : Bool}) -> {Nat.is_lt(0n,inner) == True{} : Bool}:
  DivisionOrder.right_factor_positive(count,inner,Order.le_lt_trans(0n,index,Nat.mul(count,inner),Order.zero_le(index),inside))

# The coordinate a position of a span reads lies inside its axis.
def coordinate_inside(+first: Nat,+count: Nat,+extent: Nat,+inner: Nat,+index: Nat,+fits: {Nat.is_le(Nat.add(first,count),extent) == True{} : Bool},
  +inside: {Nat.is_lt(index,Nat.mul(count,inner)) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first,Nat.div(index,inner)),extent) == True{} : Bool}:
  Order.lt_le(Nat.add(first,Nat.div(index,inner)),Nat.add(first,count),extent,
    Order.offset_lt(first,Nat.div(index,inner),count,DivisionOrder.quotient_below(index,count,inner,inner_positive(count,inner,index,inside),inside)),fits)

# A position of a box numbers below the size of its axes.
def row_below(spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: Nat,+within: SpansIn(spans,extents),
  +inside: {Nat.is_lt(index,span_total(spans)) == True{} : Bool}) -> {Nat.is_lt(row_value(spans,extents,index,0n),ShapeLaws.elements(extents)) == True{} : Bool}:
  match spans extents:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+extent,+more}:
      Empty.absurd({Nat.is_lt(row_value(Nil{},Con{extent,more},index,0n),ShapeLaws.elements(Con{extent,more})) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      Empty.absurd({Nat.is_lt(row_value(Con{Tasks.Span{first,count},rest},Nil{},index,0n),1n) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      +inner = span_total(rest)
      +later = ShapeLaws.elements(more)
      +coordinate = Nat.add(U32.to_nat(first),Nat.div(index,inner))
      +position = Nat.mod(index,inner)
      +tail = row_value(rest,more,position,0n)
      +tail_below = row_below(rest,more,position,rest_within,DivisionOrder.remainder_small(index,inner,inner_positive(U32.to_nat(count),inner,index,inside)))
      +scaled_below = {%Order.commutative(Nat.mul(coordinate,later),later) : {Nat.is_lt(Nat.add(Nat.mul(coordinate,later),tail),_) == True{} : Bool}
        Order.offset_lt(Nat.mul(coordinate,later),tail,later,tail_below) :
        {Nat.is_lt(Nat.add(Nat.mul(coordinate,later),tail),Nat.mul(1n+coordinate,later)) == True{} : Bool}}
      %Equal.sym(Nat,row_value(rest,more,position,coordinate),Nat.add(Nat.mul(coordinate,later),tail),row_split(rest,more,position,coordinate,rest_within)) :
        {Nat.is_lt(_,Nat.mul(U32.to_nat(extent),later)) == True{} : Bool}
      Order.lt_le(Nat.add(Nat.mul(coordinate,later),tail),Nat.mul(1n+coordinate,later),Nat.mul(U32.to_nat(extent),later),scaled_below,
        RowMajor.mul_le(1n+coordinate,U32.to_nat(extent),later,
          Order.strict_follow(coordinate,U32.to_nat(extent),coordinate_inside(U32.to_nat(first),U32.to_nat(count),U32.to_nat(extent),inner,index,fits,inside))))

# Values before a box that number one position agree.
def leading_same(+left: Nat,+right: Nat,+left_tail: Nat,+right_tail: Nat,+size: Nat,
  +left_below: {Nat.is_lt(left_tail,size) == True{} : Bool},+right_below: {Nat.is_lt(right_tail,size) == True{} : Bool},
  +same: {Nat.add(Nat.mul(left,size),left_tail) == Nat.add(Nat.mul(right,size),right_tail) : Nat}) -> {left == right : Nat}:
  +positive = Order.le_lt_trans(0n,left_tail,size,Order.zero_le(left_tail),left_below)
  +pairs = Equal.trans(Nat & Nat,(left,left_tail),Nat.divmod(Nat.add(Nat.mul(left,size),left_tail),size),(right,right_tail),
    Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(left,size),left_tail),size),(left,left_tail),
      Division.unique(Nat.add(Nat.mul(left,size),left_tail),size,left,left_tail,{==},positive,left_below)),
    Division.unique(Nat.add(Nat.mul(left,size),left_tail),size,right,right_tail,same,positive,right_below))
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),(left,left_tail),(right,right_tail),pairs)

# Two spans share no index.
def RangeApart(+first: U32,+count: U32,+other_first: U32,+other_count: U32) -> Data:
  Either<&2,&2,{Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(other_first)) == True{} : Bool},
    {Nat.is_le(Nat.add(U32.to_nat(other_first),U32.to_nat(other_count)),U32.to_nat(first)) == True{} : Bool}>

# Two boxes share no index at some axis.
def SpansApart(spans: List<&2,Tasks.Span>,others: List<&2,Tasks.Span>) -> Data:
  match spans others:
    case Con{Tasks.Span{+first,+count},spans} Con{Tasks.Span{+other_first,+other_count},others}:
      Either<&2,&2,RangeApart(first,count,other_first,other_count),SpansApart(spans,others)>
    case spans others:
      Empty

def range_clash(+first: Nat,+count: Nat,+other_first: Nat,+other_count: Nat,+offset: Nat,+other_offset: Nat,
  +below: {Nat.is_lt(offset,count) == True{} : Bool},+other_below: {Nat.is_lt(other_offset,other_count) == True{} : Bool},
  +same: {Nat.add(first,offset) == Nat.add(other_first,other_offset) : Nat},
  +apart: Either<&2,&2,{Nat.is_le(Nat.add(first,count),other_first) == True{} : Bool},{Nat.is_le(Nat.add(other_first,other_count),first) == True{} : Bool}>) -> Empty:
  match apart:
    case Inl{+before}:
      lt_self(Nat.add(other_first,other_offset),{%same : {Nat.is_lt(_,Nat.add(other_first,other_offset)) == True{} : Bool}
        Order.lt_le(Nat.add(first,offset),Nat.add(first,count),Nat.add(other_first,other_offset),Order.offset_lt(first,offset,count,below),
          Order.transitive(Nat.add(first,count),other_first,Nat.add(other_first,other_offset),before,Order.le_add(other_first,other_offset))) :
        {Nat.is_lt(Nat.add(other_first,other_offset),Nat.add(other_first,other_offset)) == True{} : Bool}})
    case Inr{+after}:
      lt_self(Nat.add(other_first,other_offset),{%same : {Nat.is_lt(Nat.add(other_first,other_offset),_) == True{} : Bool}
        Order.lt_le(Nat.add(other_first,other_offset),Nat.add(other_first,other_count),Nat.add(first,offset),Order.offset_lt(other_first,other_offset,other_count,other_below),
          Order.transitive(Nat.add(other_first,other_count),first,Nat.add(first,offset),after,Order.le_add(first,offset))) :
        {Nat.is_lt(Nat.add(other_first,other_offset),Nat.add(other_first,other_offset)) == True{} : Bool}})

def apart_step(+first: U32,+count: U32,+other_first: U32,+other_count: U32,+rest: List<&2,Tasks.Span>,+other_rest: List<&2,Tasks.Span>,+more: List<&2,U32>,
  +position: Nat,+other_position: Nat,+scaled: Nat,+block: Nat,+other_block: Nat,
  +below: {Nat.is_lt(block,U32.to_nat(count)) == True{} : Bool},+other_below: {Nat.is_lt(other_block,U32.to_nat(other_count)) == True{} : Bool},
  +leadings: {Nat.add(Nat.add(scaled,U32.to_nat(first)),block) == Nat.add(Nat.add(scaled,U32.to_nat(other_first)),other_block) : Nat},
  +same: {row_value(rest,more,position,Nat.add(Nat.add(scaled,U32.to_nat(first)),block)) ==
    row_value(other_rest,more,other_position,Nat.add(Nat.add(scaled,U32.to_nat(other_first)),other_block)) : Nat},
  +apart: Either<&2,&2,RangeApart(first,count,other_first,other_count),SpansApart(rest,other_rest)>,
  recurse: SpansApart(rest,other_rest) -> {row_value(rest,more,position,Nat.add(Nat.add(scaled,U32.to_nat(first)),block)) ==
    row_value(other_rest,more,other_position,Nat.add(Nat.add(scaled,U32.to_nat(first)),block)) : Nat} -> Empty) -> Empty:
  match apart:
    case Inl{+here}:
      +start = U32.to_nat(first)
      +other_start = U32.to_nat(other_first)
      range_clash(start,U32.to_nat(count),other_start,U32.to_nat(other_count),block,other_block,below,other_below,
        Order.cancel_sum(scaled,Nat.add(start,block),Nat.add(other_start,other_block),
          Equal.trans(Nat,Nat.add(scaled,Nat.add(start,block)),Nat.add(Nat.add(scaled,start),block),Nat.add(scaled,Nat.add(other_start,other_block)),
            Equal.sym(Nat,Nat.add(Nat.add(scaled,start),block),Nat.add(scaled,Nat.add(start,block)),Addition.associative(scaled,start,block)),
            Equal.trans(Nat,Nat.add(Nat.add(scaled,start),block),Nat.add(Nat.add(scaled,other_start),other_block),Nat.add(scaled,Nat.add(other_start,other_block)),leadings,
              Addition.associative(scaled,other_start,other_block)))),
        here)
    case Inr{+later_apart}:
      +leading = Nat.add(Nat.add(scaled,U32.to_nat(first)),block)
      +other_leading = Nat.add(Nat.add(scaled,U32.to_nat(other_first)),other_block)
      recurse(later_apart,{%Equal.sym(Nat,leading,other_leading,leadings) : {row_value(rest,more,position,leading) == row_value(other_rest,more,other_position,_) : Nat}
        same : {row_value(rest,more,position,leading) == row_value(other_rest,more,other_position,leading) : Nat}})

# Positions of two boxes apart at some axis number differently.
def row_apart(spans: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: Nat,+other_index: Nat,+value: Nat,
  +within: SpansIn(spans,extents),+other_within: SpansIn(others,extents),
  +inside: {Nat.is_lt(index,span_total(spans)) == True{} : Bool},+other_inside: {Nat.is_lt(other_index,span_total(others)) == True{} : Bool},
  +apart: SpansApart(spans,others),+same: {row_value(spans,extents,index,value) == row_value(others,extents,other_index,value) : Nat}) -> Empty:
  match spans others extents:
    case Nil{} +others +extents:
      apart
    case Con{Tasks.Span{+first,+count},+rest} Nil{} +extents:
      apart
    case Con{Tasks.Span{+first,+count},+rest} Con{Tasks.Span{+other_first,+other_count},+other_rest} Nil{}:
      within
    case Con{Tasks.Span{+first,+count},+rest} Con{Tasks.Span{+other_first,+other_count},+other_rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      Held.Both{+other_fits,+other_rest_within} = other_within
      +inner = span_total(rest)
      +other_inner = span_total(other_rest)
      +later = ShapeLaws.elements(more)
      +scaled = Nat.mul(value,U32.to_nat(extent))
      +block = Nat.div(index,inner)
      +other_block = Nat.div(other_index,other_inner)
      +position = Nat.mod(index,inner)
      +other_position = Nat.mod(other_index,other_inner)
      +leading = Nat.add(Nat.add(scaled,U32.to_nat(first)),block)
      +other_leading = Nat.add(Nat.add(scaled,U32.to_nat(other_first)),other_block)
      +inner_ok = inner_positive(U32.to_nat(count),inner,index,inside)
      +other_inner_ok = inner_positive(U32.to_nat(other_count),other_inner,other_index,other_inside)
      +tail_below = row_below(rest,more,position,rest_within,DivisionOrder.remainder_small(index,inner,inner_ok))
      +other_tail_below = row_below(other_rest,more,other_position,other_rest_within,DivisionOrder.remainder_small(other_index,other_inner,other_inner_ok))
      +tail = row_value(rest,more,position,0n)
      +other_tail = row_value(other_rest,more,other_position,0n)
      +leadings = leading_same(leading,other_leading,tail,other_tail,later,tail_below,other_tail_below,
        Equal.trans(Nat,Nat.add(Nat.mul(leading,later),tail),row_value(rest,more,position,leading),Nat.add(Nat.mul(other_leading,later),other_tail),
          Equal.sym(Nat,row_value(rest,more,position,leading),Nat.add(Nat.mul(leading,later),tail),row_split(rest,more,position,leading,rest_within)),
          Equal.trans(Nat,row_value(rest,more,position,leading),row_value(other_rest,more,other_position,other_leading),Nat.add(Nat.mul(other_leading,later),other_tail),
            same,row_split(other_rest,more,other_position,other_leading,other_rest_within))))
      apart_step(first,count,other_first,other_count,rest,other_rest,more,position,other_position,scaled,block,other_block,
        DivisionOrder.quotient_below(index,U32.to_nat(count),inner,inner_ok,inside),
        DivisionOrder.quotient_below(other_index,U32.to_nat(other_count),other_inner,other_inner_ok,other_inside),leadings,same,apart,
        later_apart => later_same => row_apart(rest,other_rest,more,position,other_position,leading,rest_within,other_rest_within,
          DivisionOrder.remainder_small(index,inner,inner_ok),DivisionOrder.remainder_small(other_index,other_inner,other_inner_ok),later_apart,later_same))

# A larger leading value numbers every position past a smaller one.
def leading_below(+leading: Nat,+other_leading: Nat,+tail: Nat,+other_tail: Nat,+size: Nat,
  +less: {Nat.is_lt(leading,other_leading) == True{} : Bool},+tail_below: {Nat.is_lt(tail,size) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(leading,size),tail),Nat.add(Nat.mul(other_leading,size),other_tail)) == True{} : Bool}:
  +scaled_below = {%Order.commutative(Nat.mul(leading,size),size) : {Nat.is_lt(Nat.add(Nat.mul(leading,size),tail),_) == True{} : Bool}
    Order.offset_lt(Nat.mul(leading,size),tail,size,tail_below) :
    {Nat.is_lt(Nat.add(Nat.mul(leading,size),tail),Nat.mul(1n+leading,size)) == True{} : Bool}}
  Order.lt_le(Nat.add(Nat.mul(leading,size),tail),Nat.mul(1n+leading,size),Nat.add(Nat.mul(other_leading,size),other_tail),scaled_below,
    Order.transitive(Nat.mul(1n+leading,size),Nat.mul(other_leading,size),Nat.add(Nat.mul(other_leading,size),other_tail),
      RowMajor.mul_le(1n+leading,other_leading,size,Order.strict_follow(leading,other_leading,less)),Order.le_add(Nat.mul(other_leading,size),other_tail)))

def row_lt_less(+rest: List<&2,Tasks.Span>,+more: List<&2,U32>,+position: Nat,+other_position: Nat,+leading: Nat,+other_leading: Nat,
  +within: SpansIn(rest,more),+inside: {Nat.is_lt(position,span_total(rest)) == True{} : Bool},+less: {Nat.is_lt(leading,other_leading) == True{} : Bool}) ->
  {Nat.is_lt(row_value(rest,more,position,leading),row_value(rest,more,other_position,other_leading)) == True{} : Bool}:
  +later = ShapeLaws.elements(more)
  +tail = row_value(rest,more,position,0n)
  +other_tail = row_value(rest,more,other_position,0n)
  %Equal.sym(Nat,row_value(rest,more,position,leading),Nat.add(Nat.mul(leading,later),tail),row_split(rest,more,position,leading,within)) :
    {Nat.is_lt(_,row_value(rest,more,other_position,other_leading)) == True{} : Bool}
  %Equal.sym(Nat,row_value(rest,more,other_position,other_leading),Nat.add(Nat.mul(other_leading,later),other_tail),row_split(rest,more,other_position,other_leading,within)) :
    {Nat.is_lt(Nat.add(Nat.mul(leading,later),tail),_) == True{} : Bool}
  leading_below(leading,other_leading,tail,other_tail,later,less,row_below(rest,more,position,within,inside))

def row_lt_same(+rest: List<&2,Tasks.Span>,+more: List<&2,U32>,+index: Nat,+other: Nat,+inner: Nat,+head: Nat,
  +inner_ok: {Nat.is_lt(0n,inner) == True{} : Bool},+ordered: {Nat.is_lt(index,other) == True{} : Bool},
  +block_le: {Nat.is_le(Nat.div(index,inner),Nat.div(other,inner)) == True{} : Bool},
  +not_less: {Nat.is_lt(Nat.div(index,inner),Nat.div(other,inner)) == False{} : Bool},
  recurse: {Nat.is_lt(Nat.mod(index,inner),Nat.mod(other,inner)) == True{} : Bool} ->
    {Nat.is_lt(row_value(rest,more,Nat.mod(index,inner),Nat.add(head,Nat.div(index,inner))),
      row_value(rest,more,Nat.mod(other,inner),Nat.add(head,Nat.div(index,inner)))) == True{} : Bool}) ->
  {Nat.is_lt(row_value(rest,more,Nat.mod(index,inner),Nat.add(head,Nat.div(index,inner))),
    row_value(rest,more,Nat.mod(other,inner),Nat.add(head,Nat.div(other,inner)))) == True{} : Bool}:
  +block = Nat.div(index,inner)
  +other_block = Nat.div(other,inner)
  +same = Order.antisymmetric(block,other_block,block_le,Held.lt_false_le(block,other_block,not_less))
  +positions = Order.cancel_offset_lt(Nat.mul(block,inner),Nat.mod(index,inner),Nat.mod(other,inner),
    {%DivisionOrder.quotient_exact(index,inner,inner_ok) : {Nat.is_lt(_,Nat.add(Nat.mul(block,inner),Nat.mod(other,inner))) == True{} : Bool}
      %Equal.sym(Nat,block,other_block,same) : {Nat.is_lt(index,Nat.add(Nat.mul(_,inner),Nat.mod(other,inner))) == True{} : Bool}
      %DivisionOrder.quotient_exact(other,inner,inner_ok) : {Nat.is_lt(index,_) == True{} : Bool}
      ordered : {Nat.is_lt(Nat.add(Nat.mul(block,inner),Nat.mod(index,inner)),Nat.add(Nat.mul(block,inner),Nat.mod(other,inner))) == True{} : Bool}})
  %same : {Nat.is_lt(row_value(rest,more,Nat.mod(index,inner),Nat.add(head,block)),row_value(rest,more,Nat.mod(other,inner),Nat.add(head,_))) == True{} : Bool}
  recurse(positions)

# Later positions of a box number higher.
def row_lt(spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: Nat,+other: Nat,+value: Nat,+within: SpansIn(spans,extents),
  +other_inside: {Nat.is_lt(other,span_total(spans)) == True{} : Bool},+ordered: {Nat.is_lt(index,other) == True{} : Bool}) ->
  {Nat.is_lt(row_value(spans,extents,index,value),row_value(spans,extents,other,value)) == True{} : Bool}:
  match spans extents:
    case Nil{} Nil{}:
      Empty.absurd({Nat.is_lt(row_value(Nil{},Nil{},index,value),row_value(Nil{},Nil{},other,value)) == True{} : Bool},
        Order.below_zero(index,Order.lt_le(index,other,0n,ordered,Division.lt_succ_le(other,0n,other_inside))))
    case Nil{} Con{+extent,+more}:
      Empty.absurd({Nat.is_lt(row_value(Nil{},Con{extent,more},index,value),row_value(Nil{},Con{extent,more},other,value)) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      Empty.absurd({Nat.is_lt(row_value(Con{Tasks.Span{first,count},rest},Nil{},index,value),row_value(Con{Tasks.Span{first,count},rest},Nil{},other,value)) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      +inner = span_total(rest)
      +inner_ok = inner_positive(U32.to_nat(count),inner,other,other_inside)
      +head = Nat.add(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(first))
      +block = Nat.div(index,inner)
      +other_block = Nat.div(other,inner)
      +block_le = DivisionOrder.quotient_le(index,other,inner,inner_ok,lt_to_le(index,other,ordered))
      Held.by_cases({Nat.is_lt(row_value(rest,more,Nat.mod(index,inner),Nat.add(head,block)),row_value(rest,more,Nat.mod(other,inner),Nat.add(head,other_block))) == True{} : Bool},
        Nat.is_lt(block,other_block),
        less => row_lt_less(rest,more,Nat.mod(index,inner),Nat.mod(other,inner),Nat.add(head,block),Nat.add(head,other_block),rest_within,
          DivisionOrder.remainder_small(index,inner,inner_ok),Order.offset_lt(head,block,other_block,less)),
        not_less => row_lt_same(rest,more,index,other,inner,head,inner_ok,ordered,block_le,not_less,
          positions => row_lt(rest,more,Nat.mod(index,inner),Nat.mod(other,inner),Nat.add(head,block),rest_within,
            DivisionOrder.remainder_small(other,inner,inner_ok),positions)))

# A box inside its axes has at most their positions.
def span_total_le(spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+within: SpansIn(spans,extents)) ->
  {Nat.is_le(span_total(spans),ShapeLaws.elements(extents)) == True{} : Bool}:
  match spans extents:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+extent,+more}:
      Empty.absurd({Nat.is_le(span_total(Nil{}),ShapeLaws.elements(Con{extent,more})) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      Empty.absurd({Nat.is_le(span_total(Con{Tasks.Span{first,count},rest}),ShapeLaws.elements(Nil{})) == True{} : Bool},within)
    case Con{Tasks.Span{+first,+count},+rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      +size = U32.to_nat(count)
      +whole = U32.to_nat(extent)
      +inner = span_total(rest)
      Order.transitive(Nat.mul(size,inner),Nat.mul(whole,inner),Nat.mul(whole,ShapeLaws.elements(more)),
        RowMajor.mul_le(size,whole,inner,Order.transitive(size,Nat.add(U32.to_nat(first),size),whole,Order.right_le(U32.to_nat(first),size),fits)),
        DivisionOrder.multiply_left_le(whole,inner,ShapeLaws.elements(more),span_total_le(rest,more,rest_within)))

def factor_covers(+value: Nat,+factor: Nat,+positive: {Nat.is_lt(0n,factor) == True{} : Bool}) -> {Nat.is_le(value,Nat.mul(factor,value)) == True{} : Bool}:
  %RowMajor.multiply_commutative(value,factor) : {Nat.is_le(value,_) == True{} : Bool}
  RowMajor.product_covers(value,factor,positive)

# The positions of a box count as Tasks.span_count counts them.
def span_count_exact(spans: List<&2,Tasks.Span>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,span_total(spans)) == True{} : Bool},+room: {Nat.is_le(span_total(spans),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Tasks.span_count(spans)) == span_total(spans) : Nat}:
  match spans:
    case Nil{}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest}:
      +size = U32.to_nat(count)
      +inner = span_total(rest)
      +inner_ok = DivisionOrder.right_factor_positive(size,inner,positive)
      Arithmetic.product_exact(cap,cap_ok,count,Tasks.span_count(rest),size,inner,{==},
        span_count_exact(rest,cap,cap_ok,inner_ok,
          Order.transitive(inner,Nat.mul(size,inner),U32.to_nat(cap),factor_covers(inner,size,RowMajor.left_factor_positive(size,inner,positive)),room)),
        room)

# Tasks.piece_at numbers a position of a box as row_value does.
def row_exact(spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: U32,+value: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +within: SpansIn(spans,extents),+inside: {Nat.is_lt(U32.to_nat(index),span_total(spans)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(1n+U32.to_nat(value),ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +double_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Tasks.piece_at(spans,extents,index,value)) == row_value(spans,extents,U32.to_nat(index),U32.to_nat(value)) : Nat}:
  match spans extents:
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+extent,+more}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest} Con{+extent,+more}:
      Held.Both{+fits,+rest_within} = within
      +position = U32.to_nat(index)
      +start = U32.to_nat(value)
      +size = U32.to_nat(count)
      +head = U32.to_nat(first)
      +whole = U32.to_nat(extent)
      +inner = span_total(rest)
      +later = ShapeLaws.elements(more)
      +inner_ok = inner_positive(size,inner,position,inside)
      +whole_ok = Order.lt_le(0n,size,whole,count_positive(size,inner,position,inside),
        Order.transitive(size,Nat.add(head,size),whole,Order.right_le(head,size),fits))
      +inner_later = span_total_le(rest,more,rest_within)
      +later_ok = Order.lt_le(0n,inner,later,inner_ok,inner_later)
      +later_double = Order.transitive(Nat.double(later),Nat.double(Nat.mul(whole,later)),U32.to_nat(cap),
        Order.double_le(later,Nat.mul(whole,later),factor_covers(later,whole,whole_ok)),double_room)
      +inner_double = Order.transitive(Nat.double(inner),Nat.double(later),U32.to_nat(cap),Order.double_le(inner,later,inner_later),later_double)
      +inner_room = Order.transitive(inner,Nat.double(inner),U32.to_nat(cap),Order.le_double(inner),inner_double)
      +divisor = U32.max(Tasks.span_count(rest),1)
      +divisor_value = Extrema.max_one_exact(Tasks.span_count(rest),inner,span_count_exact(rest,cap,cap_ok,inner_ok,inner_room),inner_ok)
      +block = Nat.div(position,inner)
      +block_exact = U32Division.quotient_exact(cap,cap_ok,index,divisor,position,inner,{==},divisor_value,inner_double,inner_ok)
      +position_exact = remainder_exact(cap,cap_ok,index,divisor,inner,divisor_value,inner_double,inner_ok)
      +scaled = Nat.mul(start,whole)
      +leading = Nat.add(Nat.add(scaled,head),block)
      +coordinate_below = coordinate_inside(head,size,whole,inner,position,fits,inside)
      +leading_below = {%Equal.sym(Nat,leading,Nat.add(scaled,Nat.add(head,block)),Addition.associative(scaled,head,block)) :
          {Nat.is_lt(_,Nat.mul(1n+start,whole)) == True{} : Bool}
        %Order.commutative(scaled,whole) : {Nat.is_lt(Nat.add(scaled,Nat.add(head,block)),_) == True{} : Bool}
        Order.offset_lt(scaled,Nat.add(head,block),whole,coordinate_below) : {Nat.is_lt(leading,Nat.mul(1n+start,whole)) == True{} : Bool}}
      +outer_room = {%Equal.sym(Nat,Nat.mul(Nat.mul(1n+start,whole),later),Nat.mul(1n+start,Nat.mul(whole,later)),RowMajor.assoc(1n+start,whole,later)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(Nat.mul(Nat.mul(1n+start,whole),later),U32.to_nat(cap)) == True{} : Bool}}
      +leading_room = Order.transitive(leading,Nat.mul(Nat.mul(1n+start,whole),later),U32.to_nat(cap),
        Order.transitive(leading,Nat.mul(1n+start,whole),Nat.mul(Nat.mul(1n+start,whole),later),lt_to_le(leading,Nat.mul(1n+start,whole),leading_below),
          RowMajor.product_covers(Nat.mul(1n+start,whole),later,later_ok)),
        outer_room)
      +sum_room = Order.transitive(Nat.add(scaled,head),leading,U32.to_nat(cap),Order.le_add(Nat.add(scaled,head),block),leading_room)
      +scaled_room = Order.transitive(scaled,Nat.add(scaled,head),U32.to_nat(cap),Order.le_add(scaled,head),sum_room)
      +scaled_word = (value * extent : U32)
      +sum_word = (scaled_word + first : U32)
      +leading_word = (sum_word + (index / divisor : U32) : U32)
      +scaled_exact = Arithmetic.product_exact(cap,cap_ok,value,extent,start,whole,{==},{==},scaled_room)
      +sum_exact = Arithmetic.sum_exact(cap,cap_ok,scaled_word,first,scaled,head,scaled_exact,{==},sum_room)
      +leading_exact = Arithmetic.sum_exact(cap,cap_ok,sum_word,(index / divisor : U32),Nat.add(scaled,head),block,sum_exact,block_exact,leading_room)
      +next_room = Order.transitive(Nat.mul(1n+leading,later),Nat.mul(Nat.mul(1n+start,whole),later),U32.to_nat(cap),
        RowMajor.mul_le(1n+leading,Nat.mul(1n+start,whole),later,Order.strict_follow(leading,Nat.mul(1n+start,whole),leading_below)),outer_room)
      +rest_position = (index % divisor : U32)
      %position_exact : {U32.to_nat(Tasks.piece_at(rest,more,rest_position,leading_word)) == row_value(rest,more,_,leading) : Nat}
      %leading_exact : {U32.to_nat(Tasks.piece_at(rest,more,rest_position,leading_word)) == row_value(rest,more,U32.to_nat(rest_position),_) : Nat}
      row_exact(rest,more,rest_position,leading_word,cap,cap_ok,rest_within,
        {%Equal.sym(Nat,U32.to_nat(rest_position),Nat.mod(position,inner),position_exact) : {Nat.is_lt(_,inner) == True{} : Bool}
          DivisionOrder.remainder_small(position,inner,inner_ok) : {Nat.is_lt(U32.to_nat(rest_position),inner) == True{} : Bool}},
        {%Equal.sym(Nat,U32.to_nat(leading_word),leading,leading_exact) : {Nat.is_le(Nat.mul(1n+_,later),U32.to_nat(cap)) == True{} : Bool}
          next_room : {Nat.is_le(Nat.mul(1n+U32.to_nat(leading_word),later),U32.to_nat(cap)) == True{} : Bool}},
        later_double)

# ---- The pieces of a box in order ----

# The pieces of a list number the positions of a box from position from on.
def ListedFrom(list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+piece,rest}:
      Held.Both<{U32.to_nat(piece) == row_value(spans,extents,from,0n) : Nat},ListedFrom(rest,spans,extents,1n+from)>

def listed_back(+list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+to: Nat,+same: {from == to : Nat},
  +listed: ListedFrom(list,spans,extents,from)) -> ListedFrom(list,spans,extents,to):
  %same : ListedFrom(list,spans,extents,_)
  listed

# Tasks.pieces_listed lists the positions below index, after done.
def listed_from(remaining: Nat,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: U32,+done: List<&2,U32>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within: SpansIn(spans,extents),
  +listed: ListedFrom(done,spans,extents,U32.to_nat(index)),+covered: {Nat.is_le(remaining,U32.to_nat(index)) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(index),span_total(spans)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +double_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  ListedFrom(Tasks.pieces_listed(remaining,spans,extents,index,done),spans,extents,Nat.sub(U32.to_nat(index),remaining)):
  match remaining:
    case 0n:
      listed_back(done,spans,extents,U32.to_nat(index),Nat.sub(U32.to_nat(index),0n),Equal.sym(Nat,Nat.sub(U32.to_nat(index),0n),U32.to_nat(index),Order.subtract_zero(U32.to_nat(index))),listed)
    case 1n+ +rest:
      +position = U32.to_nat(index)
      +previous = (index - 1 : U32)
      +before = Order.succ_le_lt(rest,position,covered)
      +positive = Order.le_lt_trans(0n,rest,position,Order.zero_le(rest),before)
      +index_room = Order.transitive(position,span_total(spans),U32.to_nat(cap),inside,Order.transitive(span_total(spans),ShapeLaws.elements(extents),U32.to_nat(cap),
        span_total_le(spans,extents,within),room))
      +previous_exact = Arithmetic.difference_exact(cap,cap_ok,index,1,position,1n,{==},{==},Order.strict_follow(0n,position,positive),index_room)
      +back = Equal.trans(Nat,1n+U32.to_nat(previous),1n+Nat.sub(position,1n),position,
        Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(previous),Nat.sub(position,1n),previous_exact),Order.positive_has_last(position,positive))
      +previous_inside = {%Equal.sym(Nat,U32.to_nat(previous),Nat.sub(position,1n),previous_exact) : {Nat.is_lt(_,span_total(spans)) == True{} : Bool}
        Order.lt_le(Nat.sub(position,1n),position,span_total(spans),pred_lt(position,positive),inside) :
        {Nat.is_lt(U32.to_nat(previous),span_total(spans)) == True{} : Bool}}
      +piece = Tasks.piece_at(spans,extents,previous,0)
      +here = row_exact(spans,extents,previous,0,cap,cap_ok,within,previous_inside,
        {%Equal.sym(Nat,Nat.add(ShapeLaws.elements(extents),0n),ShapeLaws.elements(extents),Addition.zero_right(ShapeLaws.elements(extents))) :
            {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          room : {Nat.is_le(Nat.mul(1n+U32.to_nat(0),ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool}},
        double_room)
      %back : ListedFrom(Tasks.pieces_listed(1n+rest,spans,extents,index,done),spans,extents,Nat.sub(_,1n+rest))
      listed_from(rest,spans,extents,previous,Con{piece,done},cap,cap_ok,within,
        Held.Both{here,listed_back(done,spans,extents,position,1n+U32.to_nat(previous),Equal.sym(Nat,1n+U32.to_nat(previous),position,back),listed)},
        {%Equal.sym(Nat,U32.to_nat(previous),Nat.sub(position,1n),previous_exact) : {Nat.is_le(rest,_) == True{} : Bool}
          pred_le(rest,position,before) :
          {Nat.is_le(rest,U32.to_nat(previous)) == True{} : Bool}},
        lt_to_le(U32.to_nat(previous),span_total(spans),previous_inside),room,double_room)

# ---- The source axis two components read apart ----

def read_axis(source: Tasks.Source) -> U32:
  match source:
    case Tasks.Read{+axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}:
      axis
    case source:
      0

# The source axis where a target axis of a row reads blocks apart.
def apart_axis(row: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,others: List<&2,Tasks.Class>,+first: FirstApart(row,classes,others)) -> U32:
  match row classes others:
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Con{+left,+lefts} Con{+right,+rights}:
      match first:
        case Inl{+here}:
          read_axis(source)
        case Inr{+there}:
          apart_axis(rest,lefts,rights,there)
    case row classes others:
      0

def or_left_true(+left: Bool,+right: Bool,+truth: {left == True{} : Bool}) -> {left || right == True{} : Bool}:
  match left:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({False{} || right == True{} : Bool},WordEncoding.false_true(truth))

def or_right_true(+left: Bool,+right: Bool,+truth: {right == True{} : Bool}) -> {left || right == True{} : Bool}:
  match left right:
    case True{} +right:
      {==}
    case False{} True{}:
      {==}
    case False{} False{}:
      Empty.absurd({False{} || False{} == True{} : Bool},WordEncoding.false_true(truth))

def held_here(+extent: U32,+block: U32,+source: Tasks.Source,+rest: List<&2,Tasks.Reading>,+left: Tasks.Class,+right: Tasks.Class,
  +here: ReadApart(source,extent,block,left,right)) -> {Tasks.holds(Tasks.read_axes(Con{Tasks.Reading{extent,block,source},rest}),read_axis(source)) == True{} : Bool}:
  match source:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      or_left_true(U32.is_eq(axis,axis),Tasks.holds(Tasks.read_axes(rest),axis),Arithmetic.equal_reflexive(axis))
    case Tasks.Unread{}:
      Empty.absurd({Tasks.holds(Tasks.read_axes(rest),0) == True{} : Bool},here)
    case Tasks.Crossing{}:
      Empty.absurd({Tasks.holds(Tasks.read_axes(rest),0) == True{} : Bool},here)

def held_source(+extent: U32,+block: U32,+source: Tasks.Source,+rest: List<&2,Tasks.Reading>,+axis: U32,+held: {Tasks.holds(Tasks.read_axes(rest),axis) == True{} : Bool}) ->
  {Tasks.holds(Tasks.read_axes(Con{Tasks.Reading{extent,block,source},rest}),axis) == True{} : Bool}:
  match source:
    case Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      or_right_true(U32.is_eq(other,axis),Tasks.holds(Tasks.read_axes(rest),axis),held)
    case Tasks.Unread{}:
      held
    case Tasks.Crossing{}:
      held

def held_there(+reading: Tasks.Reading,+rest: List<&2,Tasks.Reading>,+axis: U32,+held: {Tasks.holds(Tasks.read_axes(rest),axis) == True{} : Bool}) ->
  {Tasks.holds(Tasks.read_axes(Con{reading,rest}),axis) == True{} : Bool}:
  match reading:
    case Tasks.Reading{+extent,+block,+source}:
      held_source(extent,block,source,rest,axis,held)

# The apart axis is read by the row.
def apart_axis_held(row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+first: FirstApart(row,classes,others)) ->
  {Tasks.holds(Tasks.read_axes(row),apart_axis(row,classes,others,first)) == True{} : Bool}:
  match row classes others:
    case Nil{} +classes +others:
      Empty.absurd({Tasks.holds(Tasks.read_axes(Nil{}),apart_axis(Nil{},classes,others,first)) == True{} : Bool},first)
    case Con{+reading,+rest} Nil{} +others:
      Empty.absurd({Tasks.holds(Tasks.read_axes(Con{reading,rest}),apart_axis(Con{reading,rest},Nil{},others,first)) == True{} : Bool},first)
    case Con{+reading,+rest} Con{+left,+lefts} Nil{}:
      Empty.absurd({Tasks.holds(Tasks.read_axes(Con{reading,rest}),apart_axis(Con{reading,rest},Con{left,lefts},Nil{},first)) == True{} : Bool},first)
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Con{+left,+lefts} Con{+right,+rights}:
      match first:
        case Inl{+here}:
          held_here(extent,block,source,rest,left,right,here)
        case Inr{+there}:
          held_there(Tasks.Reading{extent,block,source},rest,apart_axis(rest,lefts,rights,there),apart_axis_held(rest,lefts,rights,there))

# Two found footprints read blocks apart.
def MaybeApart(found: Maybe<&2,Tasks.Footprint>,other: Maybe<&2,Tasks.Footprint>) -> Data:
  match found other:
    case Some{+left} Some{+right}:
      Apart(left,right)
    case found other:
      Empty

def axis_clash(+axis: U32,+wanted: U32,+rest: List<&2,U32>,+fresh: {Tasks.holds(rest,axis) == False{} : Bool},+held: {Tasks.holds(rest,wanted) == True{} : Bool},
  +same: {U32.is_eq(axis,wanted) == True{} : Bool}) -> Empty:
  WordEncoding.false_true(Equal.trans(Bool,False{},Tasks.holds(rest,axis),True{},Equal.sym(Bool,Tasks.holds(rest,axis),False{},fresh),
    {%Equal.sym(U32,axis,wanted,Arithmetic.equal_of_is_eq(axis,wanted,same)) : {Tasks.holds(rest,_) == True{} : Bool} held : {Tasks.holds(rest,axis) == True{} : Bool}}))

def axis_other(+axis: U32,+wanted: U32,+rest: List<&2,U32>,+fresh: {Tasks.holds(rest,axis) == False{} : Bool},+held: {Tasks.holds(rest,wanted) == True{} : Bool}) ->
  {U32.is_eq(axis,wanted) == False{} : Bool}:
  Held.by_cases({U32.is_eq(axis,wanted) == False{} : Bool},U32.is_eq(axis,wanted),
    same => Empty.absurd({U32.is_eq(axis,wanted) == False{} : Bool},axis_clash(axis,wanted,rest,fresh,held,same)),
    other => other)

def rest_distinct(+source: Tasks.Source,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,
  +distinct: {Tasks.distinct(Tasks.read_axes(Con{Tasks.Reading{extent,block,source},rest})) == True{} : Bool}) -> {Tasks.distinct(Tasks.read_axes(rest)) == True{} : Bool}:
  match source:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Logic.right(Bool.not(Tasks.holds(Tasks.read_axes(rest),axis)),Tasks.distinct(Tasks.read_axes(rest)),distinct)
    case Tasks.Unread{}:
      distinct
    case Tasks.Crossing{}:
      distinct

# The footprints of the target axis that reads blocks apart.
def here_apart(+source: Tasks.Source,+left: Tasks.Class,+right: Tasks.Class,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,
  +lefts: List<&2,Tasks.Class>,+rights: List<&2,Tasks.Class>,+here: ReadApart(source,extent,block,left,right)) ->
  MaybeApart(Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Con{left,lefts},read_axis(source)),
    Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Con{right,rights},read_axis(source))):
  match source left right:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped} Tasks.Class{+first,+count,+source_first,+source_count}
      Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      +read = {Tasks.Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
      %Equal.sym(Bool,U32.is_eq(axis,axis),True{},Arithmetic.equal_reflexive(axis)) :
        MaybeApart(Tasks.found_footprint(_,read,extent,block,first,count,Tasks.read_footprint(rest,lefts,axis)),
          Tasks.found_footprint(_,read,extent,block,later_first,later_count,Tasks.read_footprint(rest,rights,axis)))
      here
    case Tasks.Unread{} +left +right:
      Empty.absurd(MaybeApart(Tasks.read_footprint(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{left,lefts},0),
        Tasks.read_footprint(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{right,rights},0)),here)
    case Tasks.Crossing{} +left +right:
      Empty.absurd(MaybeApart(Tasks.read_footprint(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},Con{left,lefts},0),
        Tasks.read_footprint(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},Con{right,rights},0)),here)

# A later target axis reads blocks apart at an axis this one does not read.
def there_apart(+source: Tasks.Source,+left: Tasks.Class,+right: Tasks.Class,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,
  +lefts: List<&2,Tasks.Class>,+rights: List<&2,Tasks.Class>,+axis: U32,
  +inner: MaybeApart(Tasks.read_footprint(rest,lefts,axis),Tasks.read_footprint(rest,rights,axis)),
  +held: {Tasks.holds(Tasks.read_axes(rest),axis) == True{} : Bool},
  +distinct: {Tasks.distinct(Tasks.read_axes(Con{Tasks.Reading{extent,block,source},rest})) == True{} : Bool}) ->
  MaybeApart(Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Con{left,lefts},axis),
    Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Con{right,rights},axis)):
  match source left right:
    case Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped} Tasks.Class{+first,+count,+source_first,+source_count}
      Tasks.Class{+later_first,+later_count,+later_source_first,+later_source_count}:
      +read = {Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
      +fresh = Logic.not_true(Tasks.holds(Tasks.read_axes(rest),other),
        Logic.left(Bool.not(Tasks.holds(Tasks.read_axes(rest),other)),Tasks.distinct(Tasks.read_axes(rest)),distinct))
      +differ = axis_other(other,axis,Tasks.read_axes(rest),fresh,held)
      %Equal.sym(Bool,U32.is_eq(other,axis),False{},differ) :
        MaybeApart(Tasks.found_footprint(_,read,extent,block,first,count,Tasks.read_footprint(rest,lefts,axis)),
          Tasks.found_footprint(_,read,extent,block,later_first,later_count,Tasks.read_footprint(rest,rights,axis)))
      inner
    case Tasks.Unread{} +left +right:
      inner
    case Tasks.Crossing{} +left +right:
      inner

# The source axis where a row reads blocks apart holds footprints apart.
def footprints_apart(row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+first: FirstApart(row,classes,others),
  +distinct: {Tasks.distinct(Tasks.read_axes(row)) == True{} : Bool}) ->
  MaybeApart(Tasks.read_footprint(row,classes,apart_axis(row,classes,others,first)),Tasks.read_footprint(row,others,apart_axis(row,classes,others,first))):
  match row classes others:
    case Nil{} +classes +others:
      Empty.absurd(MaybeApart(Tasks.read_footprint(Nil{},classes,apart_axis(Nil{},classes,others,first)),
        Tasks.read_footprint(Nil{},others,apart_axis(Nil{},classes,others,first))),first)
    case Con{+reading,+rest} Nil{} +others:
      Empty.absurd(MaybeApart(Tasks.read_footprint(Con{reading,rest},Nil{},apart_axis(Con{reading,rest},Nil{},others,first)),
        Tasks.read_footprint(Con{reading,rest},others,apart_axis(Con{reading,rest},Nil{},others,first))),first)
    case Con{+reading,+rest} Con{+left,+lefts} Nil{}:
      Empty.absurd(MaybeApart(Tasks.read_footprint(Con{reading,rest},Con{left,lefts},apart_axis(Con{reading,rest},Con{left,lefts},Nil{},first)),
        Tasks.read_footprint(Con{reading,rest},Nil{},apart_axis(Con{reading,rest},Con{left,lefts},Nil{},first))),first)
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Con{+left,+lefts} Con{+right,+rights}:
      match first:
        case Inl{+here}:
          here_apart(source,left,right,extent,block,rest,lefts,rights,here)
        case Inr{+there}:
          +axis = apart_axis(rest,lefts,rights,there)
          there_apart(source,left,right,extent,block,rest,lefts,rights,axis,
            footprints_apart(rest,lefts,rights,there,rest_distinct(source,extent,block,rest,distinct)),
            apart_axis_held(rest,lefts,rights,there),distinct)

# ---- Classes end inside their axis ----

# A class reaches past its first block and ends by limit.
def EndIn(class: Tasks.Class,+limit: Nat) -> Data:
  match class:
    case Tasks.Class{+first,+count,source_first,source_count}:
      Held.Both<{Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},{Nat.is_le(U32.to_nat((first + count : U32)),limit) == True{} : Bool}>

def class_end_in(+start: U32,+next: U32,+limit: Nat,+before: {Nat.is_lt(U32.to_nat(start),U32.to_nat(next)) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(next),limit) == True{} : Bool}) -> EndIn(Tasks.Class{start,(next - start : U32),0,0},limit):
  %end_same(start,next) : Held.Both<{Nat.is_lt(U32.to_nat(start),U32.to_nat(_)) == True{} : Bool},{Nat.is_le(U32.to_nat(_),limit) == True{} : Bool}>
  Held.Both{before,inside}

def single_end(+start: U32,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+rising: Rising(Con{start,Nil{}},low,limit),
  +blocks_value: {U32.to_nat(blocks) == limit : Nat},+zero: {U32.is_zero(index) == True{} : Bool}) ->
  EndIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),limit):
  Held.Both{+above,+more} = rising
  Held.Both{+below,+rest} = more
  +class = {Tasks.Class{start,(blocks - start : U32),0,0} : Tasks.Class}
  %Equal.sym(Tasks.Class,class_word(Con{class,Nil{}},index),class,at_zero(class,Nil{},index,zero)) : EndIn(_,limit)
  class_end_in(start,blocks,limit,
    {%Equal.sym(Nat,U32.to_nat(blocks),limit,blocks_value) : {Nat.is_lt(U32.to_nat(start),_) == True{} : Bool}
      below : {Nat.is_lt(U32.to_nat(start),U32.to_nat(blocks)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(blocks),limit,blocks_value) : {Nat.is_le(_,limit) == True{} : Bool}
      Order.reflexive(limit) : {Nat.is_le(U32.to_nat(blocks),limit) == True{} : Bool}})

def head_end(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+rising: Rising(Con{start,Con{next,rest}},low,limit),
  +zero: {U32.is_zero(index) == True{} : Bool}) -> EndIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),limit):
  Held.Both{+above,+more} = rising
  Held.Both{+below,+later} = more
  Held.Both{+next_above,+next_more} = later
  Held.Both{+next_below,+after} = next_more
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},index),class,at_zero(class,others,index,zero)) : EndIn(_,limit)
  class_end_in(start,next,limit,Order.succ_le_lt(U32.to_nat(start),U32.to_nat(next),next_above),lt_to_le(U32.to_nat(next),limit,next_below))

def later_end(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+limit: Nat,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),1n+List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,Tasks.Class,Tasks.runs(Con{next,rest},blocks))) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat((index - 1 : U32)),U32.to_nat(cap)) == True{} : Bool} ->
    EndIn(class_word(Tasks.runs(Con{next,rest},blocks),(index - 1 : U32)),limit)) ->
  EndIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),limit):
  +class = {Tasks.Class{start,(next - start : U32),0,0} : Tasks.Class}
  +others = Tasks.runs(Con{next,rest},blocks)
  %Equal.sym(Tasks.Class,class_word(Con{class,others},index),class_word(others,(index - 1 : U32)),at_later(class,others,index,nonzero)) : EndIn(_,limit)
  recurse(index_back_inside(cap,cap_ok,index,List.length(&2,Tasks.Class,others),nonzero,room,inside),index_back_room(cap,cap_ok,index,nonzero,room))

# Every run of rising starts ends past its first block, by limit.
def runs_end_in(starts: List<&2,U32>,+blocks: U32,+low: Nat,+limit: Nat,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rising: Rising(starts,low,limit),+blocks_value: {U32.to_nat(blocks) == limit : Nat},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,Tasks.Class,Tasks.runs(starts,blocks))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> EndIn(class_word(Tasks.runs(starts,blocks),index),limit):
  match starts:
    case Nil{}:
      Empty.absurd(EndIn(class_word(Nil{},index),limit),Order.below_zero(U32.to_nat(index),inside))
    case Con{+start,Nil{}}:
      Held.by_cases(EndIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),limit),U32.is_zero(index),
        zero => single_end(start,blocks,low,limit,index,rising,blocks_value,zero),
        nonzero => Empty.absurd(EndIn(class_word(Tasks.runs(Con{start,Nil{}},blocks),index),limit),single_index(index,nonzero,inside)))
    case Con{+start,Con{+next,+rest}}:
      Held.Both{+above,+more} = rising
      Held.Both{+below,+later} = more
      Held.by_cases(EndIn(class_word(Tasks.runs(Con{start,Con{next,rest}},blocks),index),limit),U32.is_zero(index),
        zero => head_end(start,next,rest,blocks,low,limit,index,rising,zero),
        nonzero => later_end(start,next,rest,blocks,limit,index,cap,cap_ok,inside,room,nonzero,
          back_inside => back_room => runs_end_in(Con{next,rest},blocks,1n+U32.to_nat(start),limit,(index - 1 : U32),cap,cap_ok,later,blocks_value,back_inside,back_room)))

# Every reading of a column takes class to the end of its axis.
def ColumnEnds(column: List<&2,Tasks.Reading>,+class: Tasks.Class) -> Data:
  match column:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+extent,+block,+source},rest}:
      Held.Both<EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),ColumnEnds(rest,class)>

def same_ends(others: List<&2,Tasks.Reading>,+extent: U32,+block: U32,+class: Tasks.Class,+same: SameAxis(others,extent,block),
  +ends: EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))) -> ColumnEnds(others,class):
  match others:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+other_extent,+other_block,+source},+rest}:
      Held.Both{+extent_same,+more} = same
      Held.Both{+block_same,+later} = more
      Held.Both{
        {%Equal.sym(U32,other_extent,extent,extent_same) : EndIn(class,NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(other_block)))
          %Equal.sym(U32,other_block,block,block_same) : EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_)))
          ends : EndIn(class,NatCeiling.ceiling(U32.to_nat(other_extent),U32.to_nat(other_block)))},
        same_ends(rest,extent,block,class,later,ends)}

def column_ends(column: List<&2,Tasks.Reading>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ok: ColumnOk(column,cap),
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> ColumnEnds(column,class_word(Tasks.axis_classes(column),index)):
  match column:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+extent,+block,+source},+others}:
      Held.Both{+rooms,+same} = ok
      Held.Both{+head_room,+other_rooms} = rooms
      Held.Both{+axis_room,+read_room} = head_room
      Held.Both{+blocks_value,+axis_more} = axis_room
      +limit = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
      +starts = Tasks.shared_starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),Tasks.starts_of(others))
      +rising = starts_rising(starts,limit,axis_starts(extent,block,source,others,cap,cap_ok,rooms,same))
      +class = class_word(Tasks.runs(starts,Tasks.target_blocks(extent,block)),index)
      +ends = runs_end_in(starts,Tasks.target_blocks(extent,block),0n,limit,index,cap,cap_ok,rising,blocks_value,inside,room)
      Held.Both{ends,same_ends(others,extent,block,class,same,ends)}

# Every reading of a row takes its class to the end of its axis.
def RowEnds(row: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>) -> Data:
  match row classes:
    case Con{Tasks.Reading{+extent,+block,+source},row} Con{+class,classes}:
      Held.Both<EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),RowEnds(row,classes)>
    case row classes:
      Unit

def AllEnds(readings: List<&2,List<&2,Tasks.Reading>>,+classes: List<&2,Tasks.Class>) -> Data:
  match readings:
    case Nil{}:
      Unit
    case Con{+row,rest}:
      Held.Both<RowEnds(row,classes),AllEnds(rest,classes)>

def row_ends_none(row: List<&2,Tasks.Reading>) -> RowEnds(row,Nil{}):
  match row:
    case Nil{}:
      Unit{}
    case Con{Tasks.Reading{+extent,+block,+source},+rest}:
      Unit{}

def ends_none(readings: List<&2,List<&2,Tasks.Reading>>) -> AllEnds(readings,Nil{}):
  match readings:
    case Nil{}:
      Unit{}
    case Con{+row,+rest}:
      Held.Both{row_ends_none(row),ends_none(rest)}

def ends_cons(readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+class: Tasks.Class,+classes: List<&2,Tasks.Class>,+shape: RowsAt(readings,1n+rank),
  +here: ColumnEnds(Tasks.heads(readings),class),+later: AllEnds(Tasks.tails(readings),classes)) -> AllEnds(readings,Con{class,classes}):
  match readings:
    case Nil{}:
      Unit{}
    case Con{Nil{},+rest}:
      Held.Both{+same,+rest_shape} = shape
      Empty.absurd(AllEnds(Con{Nil{},rest},Con{class,classes}),zero_successor(rank,same))
    case Con{Con{Tasks.Reading{+extent,+block,+source},+tail},+rest}:
      Held.Both{+same,+rest_shape} = shape
      Held.Both{+head,+rest_here} = here
      Held.Both{+tail_ends,+rest_later} = later
      Held.Both{Held.Both{head,tail_ends},ends_cons(rest,rank,class,classes,rest_shape,rest_here,rest_later)}

# Every class a component chooses ends inside its axis.
# The class of the leading axis and the component of the rest that an index
# of the class lists picks lie inside their lists.
type IndexSplit<-list: List<&2,Tasks.Class>,-lists: List<&2,List<&2,Tasks.Class>>,-index: U32,-cap: U32> is Data:
  IndexSplit{quotient_inside: {Nat.is_lt(U32.to_nat((index / inner_of(lists) : U32)),List.length(&2,Tasks.Class,list)) == True{} : Bool},
    quotient_room: {Nat.is_le(U32.to_nat((index / inner_of(lists) : U32)),U32.to_nat(cap)) == True{} : Bool},
    remainder_inside: {Nat.is_lt(U32.to_nat((index % inner_of(lists) : U32)),count_nat(lists)) == True{} : Bool},
    rest_double: {Nat.is_le(Nat.double(count_nat(lists)),U32.to_nat(cap)) == True{} : Bool}}

def split_index(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +list_room: {Nat.is_le(List.length(&2,Tasks.Class,list),U32.to_nat(cap)) == True{} : Bool},+rest_room: ListsRoom(lists,cap),
  +room: {Nat.is_le(Nat.double(count_nat(Con{list,lists})),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),count_nat(Con{list,lists})) == True{} : Bool}) -> IndexSplit<list,lists,index,cap>:
  +length = List.length(&2,Tasks.Class,list)
  +divisor = count_nat(lists)
  +total = Nat.mul(length,divisor)
  +inner = inner_of(lists)
  +value = U32.to_nat(index)
  +total_positive = Order.le_lt_trans(0n,value,total,Order.zero_le(value),inside)
  +positive = DivisionOrder.right_factor_positive(length,divisor,total_positive)
  +length_positive = RowMajor.left_factor_positive(length,divisor,total_positive)
  +divisor_le = {%RowMajor.multiply_commutative(divisor,length) : {Nat.is_le(divisor,_) == True{} : Bool}
    RowMajor.product_covers(divisor,length,length_positive) : {Nat.is_le(divisor,total) == True{} : Bool}}
  +total_room = Order.transitive(total,Nat.double(total),U32.to_nat(cap),Order.le_double(total),room)
  +divisor_room = Order.transitive(divisor,total,U32.to_nat(cap),divisor_le,total_room)
  +double_room = Order.transitive(Nat.double(divisor),Nat.double(total),U32.to_nat(cap),Order.double_le(divisor,total,divisor_le),room)
  +inner_value = Extrema.max_one_exact(lists_count(lists),divisor,count_exact(lists,cap,cap_ok,rest_room,divisor_room),positive)
  +quotient_exact = U32Division.quotient_exact(cap,cap_ok,index,inner,value,divisor,{==},inner_value,double_room,positive)
  +rest_exact = remainder_exact(cap,cap_ok,index,inner,divisor,inner_value,double_room,positive)
  +quotient = (index / inner : U32)
  +quotient_inside = {%Equal.sym(Nat,U32.to_nat(quotient),Nat.div(value,divisor),quotient_exact) : {Nat.is_lt(_,length) == True{} : Bool}
    DivisionOrder.quotient_below(value,length,divisor,positive,inside) : {Nat.is_lt(U32.to_nat(quotient),length) == True{} : Bool}}
  IndexSplit{quotient_inside,Order.transitive(U32.to_nat(quotient),length,U32.to_nat(cap),lt_to_le(U32.to_nat(quotient),length,quotient_inside),list_room),
    {%Equal.sym(Nat,U32.to_nat((index % inner : U32)),Nat.mod(value,divisor),rest_exact) : {Nat.is_lt(_,divisor) == True{} : Bool}
      DivisionOrder.remainder_small(value,divisor,positive) : {Nat.is_lt(U32.to_nat((index % inner : U32)),divisor) == True{} : Bool}},
    double_room}

def quotient_inside_of(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+split: IndexSplit<list,lists,index,cap>) ->
  {Nat.is_lt(U32.to_nat((index / inner_of(lists) : U32)),List.length(&2,Tasks.Class,list)) == True{} : Bool}:
  IndexSplit{+quotient_inside,+quotient_room,+remainder_inside,+rest_double} = split
  quotient_inside

def quotient_room_of(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+split: IndexSplit<list,lists,index,cap>) ->
  {Nat.is_le(U32.to_nat((index / inner_of(lists) : U32)),U32.to_nat(cap)) == True{} : Bool}:
  IndexSplit{+quotient_inside,+quotient_room,+remainder_inside,+rest_double} = split
  quotient_room

def remainder_inside_of(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+split: IndexSplit<list,lists,index,cap>) ->
  {Nat.is_lt(U32.to_nat((index % inner_of(lists) : U32)),count_nat(lists)) == True{} : Bool}:
  IndexSplit{+quotient_inside,+quotient_room,+remainder_inside,+rest_double} = split
  remainder_inside

def rest_double_of(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+index: U32,+cap: U32,+split: IndexSplit<list,lists,index,cap>) ->
  {Nat.is_le(Nat.double(count_nat(lists)),U32.to_nat(cap)) == True{} : Bool}:
  IndexSplit{+quotient_inside,+quotient_room,+remainder_inside,+rest_double} = split
  rest_double

def chosen_ends(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +shape: RowsAt(readings,rank),+columns: ColumnsOk(rank,readings,cap),+lists_room: ListsRoom(raw_lists(rank,readings),cap),
  +room: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  AllEnds(readings,chosen_word(raw_lists(rank,readings),index)):
  match rank:
    case 0n:
      ends_none(readings)
    case 1n+ +rest:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +inner = inner_of(lists)
      +split = split_index(list,lists,index,cap,cap_ok,list_room,rest_room,room,inside)
      ends_cons(readings,rest,class_word(list,(index / inner : U32)),chosen_word(lists,(index % inner : U32)),shape,
        column_ends(Tasks.heads(readings),(index / inner : U32),cap,cap_ok,column_ok,quotient_inside_of(list,lists,index,cap,split),
          quotient_room_of(list,lists,index,cap,split)),
        chosen_ends(rest,Tasks.tails(readings),(index % inner : U32),cap,cap_ok,rows_tails(readings,rest,shape),rest_columns,rest_room,
          rest_double_of(list,lists,index,cap,split),remainder_inside_of(list,lists,index,cap,split)))

# ---- Footprints stay inside their source axis ----

# The last block of a footprint that reads something starts inside bound.
def Bounded(found: Tasks.Footprint,+block: Nat,+bound: Nat) -> Data:
  match found:
    case Tasks.Footprint{+first,+last,True{}}:
      Unit
    case Tasks.Footprint{+first,+last,False{}}:
      {Nat.is_lt(Nat.mul(U32.to_nat(last),block),bound) == True{} : Bool}

def bounded_less(found: Tasks.Footprint,+block: Nat,+smaller: Nat,+bound: Nat,+ordered: {Nat.is_le(smaller,block) == True{} : Bool},
  +bounded: Bounded(found,block,bound)) -> Bounded(found,smaller,bound):
  match found:
    case Tasks.Footprint{+first,+last,True{}}:
      Unit{}
    case Tasks.Footprint{+first,+last,False{}}:
      Order.le_lt_trans(Nat.mul(U32.to_nat(last),smaller),Nat.mul(U32.to_nat(last),block),bound,
        DivisionOrder.multiply_left_le(U32.to_nat(last),smaller,block,ordered),bounded)

# A value from shift to the top of bound cells starts a block inside bound.
def scaled_bounded(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+shift: U32,+bound: U32,+source_block: U32,
  +above: {Nat.is_le(U32.to_nat(shift),U32.to_nat(value)) == True{} : Bool},
  +below: {Nat.is_le(U32.to_nat(value),Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(U32.to_nat(((value - shift) / U32.max(source_block,1) : U32)),U32.to_nat(U32.max(source_block,1))),U32.to_nat(bound)) == True{} : Bool}:
  +blocks = U32.to_nat(U32.max(source_block,1))
  +total = Nat.add(U32.to_nat(shift),U32.to_nat(bound))
  +total_positive = Order.lt_le(0n,U32.to_nat(bound),total,positive,Order.right_le(U32.to_nat(shift),U32.to_nat(bound)))
  +below_total = Order.le_lt_trans(U32.to_nat(value),Nat.sub(total,1n),total,below,pred_lt(total,total_positive))
  +offset = Nat.sub(U32.to_nat(value),U32.to_nat(shift))
  +offset_below = Order.sub_lt(U32.to_nat(value),U32.to_nat(shift),U32.to_nat(bound),above,below_total)
  +value_room = Order.transitive(U32.to_nat(value),total,U32.to_nat(cap),lt_to_le(U32.to_nat(value),total,below_total),fits)
  %Equal.sym(Nat,U32.to_nat(((value - shift) / U32.max(source_block,1) : U32)),Nat.div(offset,blocks),scaled(cap,cap_ok,value,shift,source_block,above,value_room,block_room)) :
    {Nat.is_lt(Nat.mul(_,blocks),U32.to_nat(bound)) == True{} : Bool}
  Order.le_lt_trans(Nat.mul(Nat.div(offset,blocks),blocks),offset,U32.to_nat(bound),DivisionOrder.product_below(offset,blocks,positive_max(source_block)),offset_below)

def clamped_bounded(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(bound) == False{} : Bool}) ->
  Bounded(Tasks.Footprint{((Tasks.clamp(first,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),
    ((Tasks.clamp(last,shift,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),U32.is_zero(bound)},
    U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound)):
  +top = (shift + bound - 1 : U32)
  +start = U32.to_nat(shift)
  +positive = Arithmetic.word_nonzero_positive(bound,some)
  +top_nat = Nat.sub(Nat.add(start,U32.to_nat(bound)),1n)
  +last_max = Nat.max(U32.to_nat(last),start)
  +value = Tasks.clamp(last,shift,top)
  +low = ((Tasks.clamp(first,shift,top) - shift) / U32.max(source_block,1) : U32)
  +high = ((value - shift) / U32.max(source_block,1) : U32)
  %Equal.sym(Bool,U32.is_zero(bound),False{},some) : Bounded(Tasks.Footprint{low,high,_},U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound))
  scaled_bounded(cap,cap_ok,value,shift,bound,source_block,
    {%Equal.sym(Nat,U32.to_nat(value),Nat.min(last_max,U32.to_nat(top)),clamp_value(last,shift,top)) : {Nat.is_le(start,_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(start,Nat.min(last_max,_)) == True{} : Bool}
      NatExtrema.minimum_lower(last_max,top_nat,start,NatExtrema.maximum_right(U32.to_nat(last),start),shift_below_top(start,U32.to_nat(bound),positive)) :
      {Nat.is_le(start,U32.to_nat(value)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(value),Nat.min(last_max,U32.to_nat(top)),clamp_value(last,shift,top)) : {Nat.is_le(_,top_nat) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(Nat.min(last_max,_),top_nat) == True{} : Bool}
      NatExtrema.minimum_right(last_max,top_nat) :
      {Nat.is_le(U32.to_nat(value),top_nat) == True{} : Bool}},
    positive,fits,block_room)

def clipped_high_bounded(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +parts: Held.Both<{U32.is_zero(bound) == False{} : Bool},Held.Both<{U32.is_lt(last,shift) == False{} : Bool},{U32.is_lt((shift + bound - 1 : U32),first) == False{} : Bool}>>) ->
  {Nat.is_lt(Nat.mul(U32.to_nat(((U32.min(last,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32)),U32.to_nat(U32.max(source_block,1))),U32.to_nat(bound)) == True{} : Bool}:
  Held.Both{+nonzero,+rest} = parts
  Held.Both{+reaches,+starts} = rest
  +top = (shift + bound - 1 : U32)
  +start = U32.to_nat(shift)
  +positive = Arithmetic.word_nonzero_positive(bound,nonzero)
  +top_nat = Nat.sub(Nat.add(start,U32.to_nat(bound)),1n)
  +last_value = U32.to_nat(last)
  +shift_last = nat_lt_false(last,shift,reaches)
  +shift_top = {%Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(start,_) == True{} : Bool}
    shift_below_top(start,U32.to_nat(bound),positive) : {Nat.is_le(start,U32.to_nat(top)) == True{} : Bool}}
  +value = U32.min(last,top)
  scaled_bounded(cap,cap_ok,value,shift,bound,source_block,
    {%Equal.sym(Nat,U32.to_nat(value),Nat.min(last_value,U32.to_nat(top)),Extrema.minimum(last,top)) : {Nat.is_le(start,_) == True{} : Bool}
      NatExtrema.minimum_lower(last_value,U32.to_nat(top),start,shift_last,shift_top) : {Nat.is_le(start,U32.to_nat(value)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(value),Nat.min(last_value,U32.to_nat(top)),Extrema.minimum(last,top)) : {Nat.is_le(_,top_nat) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(top),top_nat,top_value(cap,cap_ok,shift,bound,positive,fits)) : {Nat.is_le(Nat.min(last_value,_),top_nat) == True{} : Bool}
      NatExtrema.minimum_right(last_value,top_nat) :
      {Nat.is_le(U32.to_nat(value),top_nat) == True{} : Bool}},
    positive,fits,block_room)

def clipped_bounded(+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt((shift + bound - 1 : U32),first) == False{} : Bool}) ->
  Bounded(Tasks.Footprint{((U32.max(first,shift) - shift) / U32.max(source_block,1) : U32),
    ((U32.min(last,(shift + bound - 1 : U32)) - shift) / U32.max(source_block,1) : U32),
    U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt((shift + bound - 1 : U32),first)},U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound)):
  +top = (shift + bound - 1 : U32)
  +low = ((U32.max(first,shift) - shift) / U32.max(source_block,1) : U32)
  +high = ((U32.min(last,top) - shift) / U32.max(source_block,1) : U32)
  %Equal.sym(Bool,U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first),False{},some) :
    Bounded(Tasks.Footprint{low,high,_},U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound))
  clipped_high_bounded(first,last,shift,bound,source_block,cap,cap_ok,fits,block_room,or3(U32.is_zero(bound),U32.is_lt(last,shift),U32.is_lt(top,first),some))

# The footprint of a virtual range ends inside bound.
def placed_bounded(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  Bounded(Tasks.placed(clamped,first,last,shift,bound,source_block),U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound)):
  match clamped:
    case True{}:
      +top = (shift + bound - 1 : U32)
      +blocks = U32.max(source_block,1)
      +low = ((Tasks.clamp(first,shift,top) - shift) / blocks : U32)
      +high = ((Tasks.clamp(last,shift,top) - shift) / blocks : U32)
      Held.by_cases(Bounded(Tasks.Footprint{low,high,U32.is_zero(bound)},U32.to_nat(blocks),U32.to_nat(bound)),U32.is_zero(bound),
        none => {%Equal.sym(Bool,U32.is_zero(bound),True{},none) : Bounded(Tasks.Footprint{low,high,_},U32.to_nat(blocks),U32.to_nat(bound))
          Unit{} : Bounded(Tasks.Footprint{low,high,U32.is_zero(bound)},U32.to_nat(blocks),U32.to_nat(bound))},
        some => clamped_bounded(first,last,shift,bound,source_block,cap,cap_ok,fits,block_room,some))
    case False{}:
      +top = (shift + bound - 1 : U32)
      +blocks = U32.max(source_block,1)
      +low = ((U32.max(first,shift) - shift) / blocks : U32)
      +high = ((U32.min(last,top) - shift) / blocks : U32)
      +empty = U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first)
      Held.by_cases(Bounded(Tasks.Footprint{low,high,empty},U32.to_nat(blocks),U32.to_nat(bound)),empty,
        none => {%Equal.sym(Bool,empty,True{},none) : Bounded(Tasks.Footprint{low,high,_},U32.to_nat(blocks),U32.to_nat(bound))
          Unit{} : Bounded(Tasks.Footprint{low,high,empty},U32.to_nat(blocks),U32.to_nat(bound))},
        some => clipped_bounded(first,last,shift,bound,source_block,cap,cap_ok,fits,block_room,some))

# The footprint of a read range ends inside the bound of the read.
def RangeBounded(read: Tasks.Source,+found: Tasks.Footprint) -> Data:
  match read:
    case Tasks.Read{axis,+source_block,step,offset,mirrored,width,divisor,shift,+bound,clamped}:
      Bounded(found,U32.to_nat(U32.max(source_block,1)),U32.to_nat(bound))
    case read:
      Unit

def range_bounded(+read: Tasks.Source,+low: U32,+high: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadRoom(read,index,cap)) -> RangeBounded(read,Tasks.range_footprint(read,low,high)):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+positive,+rest} = room
      Held.Both{+reach,+more} = rest
      Held.Both{+divisor_room,+placing} = more
      Held.Both{+fits,+block_room} = placing
      placed_bounded(clamped,Tasks.virtual_first(step,offset,mirrored,divisor,low,high),Tasks.virtual_last(step,offset,mirrored,width,divisor,low,high),
        shift,bound,source_block,cap,cap_ok,fits,block_room)
    case Tasks.Unread{}:
      Unit{}
    case Tasks.Crossing{}:
      Unit{}

# The footprint of a class that ends inside its axis is formed.
def class_formed(+read: Tasks.Source,+extent: U32,+block: U32,+first: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +starts: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +read_room: ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),cap)) ->
  Formed(Tasks.range_footprint(read,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),cap):
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +blocks = NatCeiling.ceiling(total,size)
  +head = U32.to_nat(first)
  +end_word = (first + count : U32)
  +finish = U32.to_nat(end_word)
  +start = Nat.mul(head,size)
  +reach = Nat.mul(finish,size)
  +head_inside = Order.lt_le(head,finish,blocks,starts,ends)
  +before = Equal.trans(Bool,Nat.is_lt(start,total),Nat.is_lt(head,blocks),True{},
    Equal.sym(Bool,Nat.is_lt(head,blocks),Nat.is_lt(start,total),NatCeiling.upper_comparison(total,size,head,positive)),head_inside)
  +finish_positive = Order.le_lt_trans(0n,head,finish,Order.zero_le(head),starts)
  +last_inside = Order.lt_le(Nat.sub(finish,1n),finish,blocks,pred_lt(finish,finish_positive),ends)
  +last_before = Equal.trans(Bool,Nat.is_lt(Nat.mul(Nat.sub(finish,1n),size),total),Nat.is_lt(Nat.sub(finish,1n),blocks),True{},
    Equal.sym(Bool,Nat.is_lt(Nat.sub(finish,1n),blocks),Nat.is_lt(Nat.mul(Nat.sub(finish,1n),size),total),
      NatCeiling.upper_comparison(total,size,Nat.sub(finish,1n),positive)),last_inside)
  +reach_below = {%Order.positive_has_last(finish,finish_positive) : {Nat.is_lt(Nat.mul(_,size),Nat.add(size,total)) == True{} : Bool}
    Order.offset_lt(size,Nat.mul(Nat.sub(finish,1n),size),total,last_before) :
    {Nat.is_lt(reach,Nat.add(size,total)) == True{} : Bool}}
  +reach_room = Order.transitive(reach,Nat.add(size,total),U32.to_nat(cap),lt_to_le(reach,Nat.add(size,total),reach_below),
    {%Order.commutative(total,size) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(size,total),U32.to_nat(cap)) == True{} : Bool}})
  +start_room = Order.transitive(start,total,U32.to_nat(cap),lt_to_le(start,total,before),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  +low = (first * block : U32)
  +low_exact = Arithmetic.product_exact(cap,cap_ok,first,block,head,size,{==},{==},start_room)
  +scaled_end = ((first + count) * block : U32)
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,end_word,block,finish,size,{==},{==},reach_room)
  +end = U32.min(scaled_end,extent)
  +end_value = Nat.min(reach,total)
  +end_exact = {%scaled_exact : {U32.to_nat(end) == Nat.min(_,total) : Nat} Extrema.minimum(scaled_end,extent) : {U32.to_nat(end) == end_value : Nat}}
  +inside = NatExtrema.minimum_strict_lower(reach,total,start,RowMajor.multiply_lt(head,finish,size,starts,positive),before)
  +end_room = Order.transitive(end_value,total,U32.to_nat(cap),NatExtrema.minimum_right(reach,total),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  +high = (end - 1 : U32)
  +high_value = Nat.sub(end_value,1n)
  +high_exact = Arithmetic.difference_exact(cap,cap_ok,end,1,end_value,1n,end_exact,{==},
    Order.strict_follow(0n,end_value,Order.le_lt_trans(0n,start,end_value,Order.zero_le(start),inside)),end_room)
  range_formed(read,low,high,cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(low),start,low_exact) : {Nat.is_le(_,U32.to_nat(high)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(high),high_value,high_exact) : {Nat.is_le(start,_) == True{} : Bool}
      pred_le(start,end_value,inside) : {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}},
    room_down(read,U32.to_nat(high),Nat.sub(total,1n),cap,
      {%Equal.sym(Nat,U32.to_nat(high),high_value,high_exact) : {Nat.is_le(_,Nat.sub(total,1n)) == True{} : Bool}
        DivisionOrder.subtract_le(end_value,total,1n,NatExtrema.minimum_right(reach,total)) : {Nat.is_le(U32.to_nat(high),Nat.sub(total,1n)) == True{} : Bool}},
      read_room))

# ---- The footprint a component reads at one source axis ----

# A footprint found at a source axis is formed and ends inside the axis.
def FoundOk(found: Maybe<&2,Tasks.Footprint>,+block: U32,+extent: U32,+cap: U32) -> Data:
  match found:
    case Some{+footprint}:
      Held.Both<Formed(footprint,cap),Bounded(footprint,U32.to_nat(block),U32.to_nat(extent))>
    case None{}:
      Unit

# A read of axis reads it in the blocks of the source grid, bounded by its
# extent.
def AxisFacts(hit: Bool,+source_block: U32,+bound: U32,+grid: Grid.Grid,+axis: U32,+extent: U32) -> Data:
  match hit:
    case True{}:
      Held.Both<{source_block == Grid.block(grid,axis,extent) : U32},{bound == extent : U32}>
    case False{}:
      Unit

def ReadsAxis(row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis: U32,+extent: U32) -> Data:
  match row:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{reading_extent,reading_block,Tasks.Read{+other,+source_block,step,offset,mirrored,width,divisor,shift,+bound,clamped}},rest}:
      Held.Both<AxisFacts(U32.is_eq(other,axis),source_block,bound,grid,axis,extent),ReadsAxis(rest,grid,axis,extent)>
    case Con{reading,rest}:
      ReadsAxis(rest,grid,axis,extent)

def found_read(+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +extent: U32,+block: U32,+first: U32,+count: U32,+grid: Grid.Grid,+axis: U32,+source_extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},cap),
  +starts: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +facts: Held.Both<{source_block == Grid.block(grid,axis,source_extent) : U32},{bound == source_extent : U32}>) ->
  FoundOk(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},(first * block : U32),
    (U32.min(((first + count) * block : U32),extent) - 1 : U32))},Grid.block(grid,axis,source_extent),source_extent,cap):
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+blocks_value,+more} = axis_room
  Held.Both{+ceiling_room,+rest} = more
  Held.Both{+positive,+fits} = rest
  Held.Both{+block_same,+bound_same} = facts
  +read = {Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  +low = (first * block : U32)
  +high = (U32.min(((first + count) * block : U32),extent) - 1 : U32)
  +footprint = Tasks.range_footprint(read,low,high)
  +narrow = bounded_less(footprint,U32.to_nat(U32.max(source_block,1)),U32.to_nat(source_block),U32.to_nat(bound),
    {%Equal.sym(Nat,U32.to_nat(U32.max(source_block,1)),Nat.max(U32.to_nat(source_block),1n),Extrema.maximum(source_block,1)) :
        {Nat.is_le(U32.to_nat(source_block),_) == True{} : Bool}
      NatExtrema.maximum_left(U32.to_nat(source_block),1n) : {Nat.is_le(U32.to_nat(source_block),U32.to_nat(U32.max(source_block,1))) == True{} : Bool}},
    range_bounded(read,low,high,Nat.sub(U32.to_nat(extent),1n),cap,cap_ok,read_room))
  Held.Both{class_formed(read,extent,block,first,count,cap,cap_ok,starts,ends,positive,fits,read_room),
    {%block_same : Bounded(footprint,U32.to_nat(_),U32.to_nat(source_extent))
      %bound_same : Bounded(footprint,U32.to_nat(source_block),U32.to_nat(_))
      narrow : Bounded(footprint,U32.to_nat(Grid.block(grid,axis,source_extent)),U32.to_nat(source_extent))}}

def found_same(+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +extent: U32,+block: U32,+first: U32,+count: U32,+later: Maybe<&2,Tasks.Footprint>,+grid: Grid.Grid,+axis: U32,+source_extent: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},cap),
  +starts: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +facts: AxisFacts(U32.is_eq(other,axis),source_block,bound,grid,axis,source_extent),+same: {U32.is_eq(other,axis) == True{} : Bool}) ->
  FoundOk(Tasks.found_footprint(U32.is_eq(other,axis),Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,first,count,later),
    Grid.block(grid,axis,source_extent),source_extent,cap):
  +read = {Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
  %Equal.sym(Bool,U32.is_eq(other,axis),True{},same) :
    FoundOk(Tasks.found_footprint(_,read,extent,block,first,count,later),Grid.block(grid,axis,source_extent),source_extent,cap)
  found_read(other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,extent,block,first,count,grid,axis,source_extent,cap,cap_ok,room,starts,ends,
    {%same : AxisFacts(_,source_block,bound,grid,axis,source_extent) facts : AxisFacts(True{},source_block,bound,grid,axis,source_extent)})

def found_other(+found: Bool,+read: Tasks.Source,+extent: U32,+block: U32,+first: U32,+count: U32,+later: Maybe<&2,Tasks.Footprint>,
  +grid: Grid.Grid,+axis: U32,+source_extent: U32,+cap: U32,+other: {found == False{} : Bool},
  +held: FoundOk(later,Grid.block(grid,axis,source_extent),source_extent,cap)) ->
  FoundOk(Tasks.found_footprint(found,read,extent,block,first,count,later),Grid.block(grid,axis,source_extent),source_extent,cap):
  %Equal.sym(Bool,found,False{},other) : FoundOk(Tasks.found_footprint(_,read,extent,block,first,count,later),Grid.block(grid,axis,source_extent),source_extent,cap)
  held

def found_head(+source: Tasks.Source,+class: Tasks.Class,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,
  +grid: Grid.Grid,+axis: U32,+source_extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: ReadingRoom(extent,block,source,cap),+end: EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),
  +reads: ReadsAxis(Con{Tasks.Reading{extent,block,source},rest},grid,axis,source_extent),
  +later: FoundOk(Tasks.read_footprint(rest,classes,axis),Grid.block(grid,axis,source_extent),source_extent,cap)) ->
  FoundOk(Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Con{class,classes},axis),Grid.block(grid,axis,source_extent),source_extent,cap):
  match source class:
    case Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped} Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both{+starts,+ends} = end
      Held.Both{+facts,+rest_reads} = reads
      +read = {Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped} : Tasks.Source}
      +found = Tasks.read_footprint(rest,classes,axis)
      Held.by_cases(FoundOk(Tasks.found_footprint(U32.is_eq(other,axis),read,extent,block,first,count,found),Grid.block(grid,axis,source_extent),source_extent,cap),
        U32.is_eq(other,axis),
        same => found_same(other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,extent,block,first,count,found,grid,axis,source_extent,
          cap,cap_ok,room,starts,ends,facts,same),
        apart => found_other(U32.is_eq(other,axis),read,extent,block,first,count,found,grid,axis,source_extent,cap,apart,later))
    case Tasks.Unread{} +class:
      later
    case Tasks.Crossing{} +class:
      later

def found_none(+source: Tasks.Source,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis: U32,+source_extent: U32,+cap: U32) ->
  FoundOk(Tasks.read_footprint(Con{Tasks.Reading{extent,block,source},rest},Nil{},axis),Grid.block(grid,axis,source_extent),source_extent,cap):
  match source:
    case Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Unit{}
    case Tasks.Unread{}:
      Unit{}
    case Tasks.Crossing{}:
      Unit{}

def reads_rest(+source: Tasks.Source,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis: U32,+source_extent: U32,
  +reads: ReadsAxis(Con{Tasks.Reading{extent,block,source},rest},grid,axis,source_extent)) -> ReadsAxis(rest,grid,axis,source_extent):
  match source:
    case Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+facts,+later} = reads
      later
    case Tasks.Unread{}:
      reads
    case Tasks.Crossing{}:
      reads

# The footprint a row of readings finds at a source axis is formed and ends
# inside the axis.
def found_ok(row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+source_extent: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: ReadingsRoom(row,cap),+ends: RowEnds(row,classes),+reads: ReadsAxis(row,grid,axis,source_extent)) ->
  FoundOk(Tasks.read_footprint(row,classes,axis),Grid.block(grid,axis,source_extent),source_extent,cap):
  match row classes:
    case Nil{} +classes:
      Unit{}
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Nil{}:
      found_none(source,extent,block,rest,grid,axis,source_extent,cap)
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Con{+class,+others}:
      Held.Both{+room,+rest_rooms} = rooms
      Held.Both{+end,+rest_ends} = ends
      found_head(source,class,extent,block,rest,others,grid,axis,source_extent,cap,cap_ok,room,end,reads,
        found_ok(rest,others,grid,axis,source_extent,cap,cap_ok,rest_rooms,rest_ends,reads_rest(source,extent,block,rest,grid,axis,source_extent,reads)))

# ---- The spans of a box ----

# A span lies inside its axis.
def SpanIn(span: Tasks.Span,+extent: U32) -> Data:
  match span:
    case Tasks.Span{+first,+count}:
      {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent)) == True{} : Bool}

# The indices from low to the clipped reach end at the clipped reach.
def span_sum(+low: U32,+reach: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(low),Nat.min(U32.to_nat(reach),U32.to_nat(extent))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(low),U32.to_nat((U32.min(reach,extent) - low : U32))) == Nat.min(U32.to_nat(reach),U32.to_nat(extent)) : Nat}:
  +total = U32.to_nat(extent)
  +end = U32.min(reach,extent)
  +end_value = Nat.min(U32.to_nat(reach),total)
  +end_room = Order.transitive(end_value,total,U32.to_nat(cap),NatExtrema.minimum_right(U32.to_nat(reach),total),room)
  +count_exact = Arithmetic.difference_exact(cap,cap_ok,end,low,end_value,U32.to_nat(low),Extrema.minimum(reach,extent),{==},ordered,end_room)
  %Equal.sym(Nat,U32.to_nat((end - low : U32)),Nat.sub(end_value,U32.to_nat(low)),count_exact) : {Nat.add(U32.to_nat(low),_) == end_value : Nat}
  Order.restore_shift(U32.to_nat(low),end_value,ordered)

def span_close(+low: U32,+reach: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(low),Nat.min(U32.to_nat(reach),U32.to_nat(extent))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) -> SpanIn(Tasks.Span{low,(U32.min(reach,extent) - low : U32)},extent):
  +total = U32.to_nat(extent)
  +sum = Nat.add(U32.to_nat(low),U32.to_nat((U32.min(reach,extent) - low : U32)))
  %Equal.sym(Nat,sum,Nat.min(U32.to_nat(reach),total),span_sum(low,reach,extent,cap,cap_ok,ordered,room)) : {Nat.is_le(_,total) == True{} : Bool}
  NatExtrema.minimum_right(U32.to_nat(reach),total)

# The block around the fixed index of an axis no target axis reads lies inside
# the axis.
def none_span_in(+extent: U32,+block: U32,+coordinate: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) -> SpanIn(Tasks.box_span(None{},extent,block,coordinate),extent):
  +total = U32.to_nat(extent)
  +size = U32.to_nat(block)
  +scaled = (coordinate / U32.max(block,1) * block : U32)
  +low = U32.min(scaled,extent)
  +start = U32.to_nat(low)
  +start_le = {%Equal.sym(Nat,start,Nat.min(U32.to_nat(scaled),total),Extrema.minimum(scaled,extent)) : {Nat.is_le(_,total) == True{} : Bool}
    NatExtrema.minimum_right(U32.to_nat(scaled),total) : {Nat.is_le(start,total) == True{} : Bool}}
  +total_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +reach_exact = Arithmetic.sum_exact(cap,cap_ok,low,block,start,size,{==},{==},
    Order.transitive(Nat.add(start,size),Nat.add(total,size),U32.to_nat(cap),Order.add_le(start,total,size,start_le),room))
  span_close(low,(low + block : U32),extent,cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat((low + block : U32)),Nat.add(start,size),reach_exact) : {Nat.is_le(start,Nat.min(_,total)) == True{} : Bool}
      NatExtrema.minimum_lower(Nat.add(start,size),total,start,Order.le_add(start,size),start_le) :
      {Nat.is_le(start,Nat.min(U32.to_nat((low + block : U32)),total)) == True{} : Bool}},
    total_room)

# The span of a footprint that reads blocks first to last starts at the first
# and ends inside the axis, by the end of the last.
def NonEmptySpan(span: Tasks.Span,+extent: U32,+first: Nat,+last: Nat,+block: Nat) -> Data:
  match span:
    case Tasks.Span{+low,+count}:
      Held.Both<{U32.to_nat(low) == Nat.mul(first,block) : Nat},Held.Both<SpanIn(Tasks.Span{low,count},extent),
        {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(count)),Nat.mul(1n+last,block)) == True{} : Bool}>>

def nonempty_in(span: Tasks.Span,+extent: U32,+first: Nat,+last: Nat,+block: Nat,+facts: NonEmptySpan(span,extent,first,last,block)) -> SpanIn(span,extent):
  match span:
    case Tasks.Span{+low,+count}:
      Held.Both{+start,+rest} = facts
      Held.Both{+inside,+end} = rest
      inside

def full_span(+first: U32,+last: U32,+extent: U32,+block: U32,+coordinate: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+last_room: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(U32.to_nat(last),U32.to_nat(block)),U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) ->
  NonEmptySpan(Tasks.box_span(Some{Tasks.Footprint{first,last,False{}}},extent,block,coordinate),extent,U32.to_nat(first),U32.to_nat(last),U32.to_nat(block)):
  +total = U32.to_nat(extent)
  +size = U32.to_nat(block)
  +head = U32.to_nat(first)
  +tail = U32.to_nat(last)
  +total_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +scaled = (first * block : U32)
  +start_value = Nat.mul(head,size)
  +start_below = Order.le_lt_trans(start_value,Nat.mul(tail,size),total,RowMajor.mul_le(head,tail,size,ordered),bounded)
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,first,block,head,size,{==},{==},
    Order.transitive(start_value,total,U32.to_nat(cap),lt_to_le(start_value,total,start_below),total_room))
  +low = U32.min(scaled,extent)
  +low_exact = Equal.trans(Nat,U32.to_nat(low),Nat.min(U32.to_nat(scaled),total),start_value,Extrema.minimum(scaled,extent),
    {%Equal.sym(Nat,U32.to_nat(scaled),start_value,scaled_exact) : {Nat.min(_,total) == start_value : Nat}
      NatExtrema.minimum_when_ordered(start_value,total,lt_to_le(start_value,total,start_below)) : {Nat.min(U32.to_nat(scaled),total) == start_value : Nat}})
  +next_exact = Equal.trans(Nat,U32.to_nat((last + 1 : U32)),Nat.add(tail,1n),1n+tail,
    Arithmetic.sum_exact(cap,cap_ok,last,1,tail,1n,{==},{==},
      {%Equal.sym(Nat,Nat.add(tail,1n),1n+tail,Addition.successor_width(tail)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.strict_follow(tail,U32.to_nat(cap),last_room) : {Nat.is_le(Nat.add(tail,1n),U32.to_nat(cap)) == True{} : Bool}}),
    Addition.successor_width(tail))
  +reach_value = Nat.mul(1n+tail,size)
  +reach_below = {Order.offset_lt(size,Nat.mul(tail,size),total,bounded) : {Nat.is_lt(reach_value,Nat.add(size,total)) == True{} : Bool}}
  +reach_room = Order.transitive(reach_value,Nat.add(size,total),U32.to_nat(cap),lt_to_le(reach_value,Nat.add(size,total),reach_below),
    {%Order.commutative(total,size) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(size,total),U32.to_nat(cap)) == True{} : Bool}})
  +reach = ((last + 1) * block : U32)
  +reach_exact = Arithmetic.product_exact(cap,cap_ok,(last + 1 : U32),block,1n+tail,size,next_exact,{==},reach_room)
  +ordered_span = {%Equal.sym(Nat,U32.to_nat(low),start_value,low_exact) : {Nat.is_le(_,Nat.min(U32.to_nat(reach),total)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(reach),reach_value,reach_exact) : {Nat.is_le(start_value,Nat.min(_,total)) == True{} : Bool}
    NatExtrema.minimum_lower(reach_value,total,start_value,RowMajor.mul_le(head,1n+tail,size,Order.successor(head,tail,ordered)),lt_to_le(start_value,total,start_below)) :
    {Nat.is_le(U32.to_nat(low),Nat.min(U32.to_nat(reach),total)) == True{} : Bool}}
  +count = (U32.min(reach,extent) - low : U32)
  +sum = span_sum(low,reach,extent,cap,cap_ok,ordered_span,total_room)
  Held.Both{low_exact,Held.Both{span_close(low,reach,extent,cap,cap_ok,ordered_span,total_room),
    {%Equal.sym(Nat,Nat.add(U32.to_nat(low),U32.to_nat(count)),Nat.min(U32.to_nat(reach),total),sum) : {Nat.is_le(_,reach_value) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(reach),reach_value,reach_exact) : {Nat.is_le(Nat.min(_,total),reach_value) == True{} : Bool}
      NatExtrema.minimum_left(reach_value,total) : {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(count)),reach_value) == True{} : Bool}}}}

# The span of every footprint found at an axis lies inside the axis.
def span_in(found: Maybe<&2,Tasks.Footprint>,+extent: U32,+block: U32,+coordinate: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: FoundOk(found,block,extent,cap),+room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) ->
  SpanIn(Tasks.box_span(found,extent,block,coordinate),extent):
  match found:
    case None{}:
      none_span_in(extent,block,coordinate,cap,cap_ok,room)
    case Some{Tasks.Footprint{+first,+last,True{}}}:
      Order.zero_le(U32.to_nat(extent))
    case Some{Tasks.Footprint{+first,+last,False{}}}:
      Held.Both{+formed,+bounded} = ok
      Held.Both{+ordered,+last_room} = formed
      nonempty_in(Tasks.box_span(Some{Tasks.Footprint{first,last,False{}}},extent,block,coordinate),extent,U32.to_nat(first),U32.to_nat(last),U32.to_nat(block),
        full_span(first,last,extent,block,coordinate,cap,cap_ok,ordered,last_room,bounded,room))

# ---- The block span of the last prefix axis ----

def unit_span(+span: Tasks.Span,+units: U32) -> Tasks.Span:
  match span:
    case Tasks.Span{+first,+count}:
      +unit = U32.max(units,1)
      Tasks.Span{(first / unit : U32),(Grid.divided_up((first + count : U32),unit) - first / unit : U32)}

# A rounded up quotient of words is the ceiling of their values.
def divided_value(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count: U32,+size: U32,+size_value: Nat,
  +size_exact: {U32.to_nat(U32.max(size,1)) == size_value : Nat},+positive: {Nat.is_lt(0n,size_value) == True{} : Bool},
  +double_room: {Nat.is_le(Nat.double(size_value),U32.to_nat(cap)) == True{} : Bool},+count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Grid.divided_up(count,size)) == NatCeiling.ceiling(U32.to_nat(count),size_value) : Nat}:
  +divisor = U32.max(size,1)
  +quotient = Nat.div(U32.to_nat(count),size_value)
  +rest = Nat.mod(U32.to_nat(count),size_value)
  +more = Bool.pick(Nat,Nat.is_eq(rest,0n),0n,1n)
  +parts = Equal.sym(Nat,NatCeiling.rounded(quotient,rest),Nat.add(quotient,more),NatCeiling.rounded_choice(quotient,rest))
  Equal.trans(Nat,U32.to_nat(Grid.divided_up(count,size)),Nat.add(quotient,more),NatCeiling.ceiling(U32.to_nat(count),size_value),
    Arithmetic.sum_exact(cap,cap_ok,(count / divisor : U32),Bool.pick(U32,U32.is_zero((count % divisor : U32)),0,1),quotient,more,
      U32Division.quotient_exact(cap,cap_ok,count,divisor,U32.to_nat(count),size_value,{==},size_exact,double_room,positive),
      GridLaws.picked((count % divisor : U32),rest,remainder_exact(cap,cap_ok,count,divisor,size_value,size_exact,double_room,positive)),
      {%Equal.sym(Nat,Nat.add(quotient,more),NatCeiling.ceiling(U32.to_nat(count),size_value),parts) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.transitive(NatCeiling.ceiling(U32.to_nat(count),size_value),U32.to_nat(count),U32.to_nat(cap),NatCeiling.bounded(U32.to_nat(count),size_value,positive),count_room) :
        {Nat.is_le(Nat.add(quotient,more),U32.to_nat(cap)) == True{} : Bool}}),
    parts)

def quotient_below_ceiling(+value: Nat,+divisor: Nat) -> {Nat.is_le(Nat.div(value,divisor),NatCeiling.ceiling(value,divisor)) == True{} : Bool}:
  %Equal.sym(Nat,NatCeiling.rounded(Nat.div(value,divisor),Nat.mod(value,divisor)),
      Nat.add(Nat.div(value,divisor),Bool.pick(Nat,Nat.is_eq(Nat.mod(value,divisor),0n),0n,1n)),NatCeiling.rounded_choice(Nat.div(value,divisor),Nat.mod(value,divisor))) :
    {Nat.is_le(Nat.div(value,divisor),_) == True{} : Bool}
  Order.le_add(Nat.div(value,divisor),Bool.pick(Nat,Nat.is_eq(Nat.mod(value,divisor),0n),0n,1n))

def UnitFacts(span: Tasks.Span,+units: U32) -> Data:
  match span:
    case Tasks.Span{+first,+count}:
      +unit = U32.max(units,1)
      Held.Both<{U32.to_nat((first / unit : U32)) == Nat.div(U32.to_nat(first),U32.to_nat(unit)) : Nat},
        {Nat.add(U32.to_nat((first / unit : U32)),U32.to_nat((Grid.divided_up((first + count : U32),unit) - first / unit : U32))) ==
          NatCeiling.ceiling(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(unit)) : Nat}>

# The blocks of the last prefix axis a span covers, as Tasks.block_spans
# takes them.
def unit_facts(span: Tasks.Span,+extent: U32,+units: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within: SpanIn(span,extent),
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(units,1))),U32.to_nat(cap)) == True{} : Bool}) ->
  UnitFacts(span,units):
  match span:
    case Tasks.Span{+first,+count}:
      +unit = U32.max(units,1)
      +size = U32.to_nat(unit)
      +positive = positive_max(units)
      +head = U32.to_nat(first)
      +reach = Nat.add(head,U32.to_nat(count))
      +reach_room = Order.transitive(reach,U32.to_nat(extent),U32.to_nat(cap),within,room)
      +reach_exact = Arithmetic.sum_exact(cap,cap_ok,first,count,head,U32.to_nat(count),{==},{==},reach_room)
      +quotient_exact = U32Division.quotient_exact(cap,cap_ok,first,unit,head,size,{==},{==},unit_room,positive)
      +ceiling_exact = {%reach_exact : {U32.to_nat(Grid.divided_up((first + count : U32),unit)) == NatCeiling.ceiling(_,size) : Nat}
        divided_value(cap,cap_ok,(first + count : U32),unit,size,Extrema.max_one_exact(unit,size,{==},positive),positive,unit_room,
          {%Equal.sym(Nat,U32.to_nat((first + count : U32)),reach,reach_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} reach_room :
            {Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(cap)) == True{} : Bool}}) :
        {U32.to_nat(Grid.divided_up((first + count : U32),unit)) == NatCeiling.ceiling(reach,size) : Nat}}
      +ceiling_value = NatCeiling.ceiling(reach,size)
      +ordered = Order.transitive(Nat.div(head,size),Nat.div(reach,size),ceiling_value,
        DivisionOrder.quotient_le(head,reach,size,positive,Order.le_add(head,U32.to_nat(count))),quotient_below_ceiling(reach,size))
      +ceiling_room = Order.transitive(ceiling_value,reach,U32.to_nat(cap),NatCeiling.bounded(reach,size,positive),reach_room)
      +count_exact = Arithmetic.difference_exact(cap,cap_ok,Grid.divided_up((first + count : U32),unit),(first / unit : U32),ceiling_value,Nat.div(head,size),
        ceiling_exact,quotient_exact,ordered,ceiling_room)
      Held.Both{quotient_exact,
        {%Equal.sym(Nat,U32.to_nat((first / unit : U32)),Nat.div(head,size),quotient_exact) :
            {Nat.add(_,U32.to_nat((Grid.divided_up((first + count : U32),unit) - first / unit : U32))) == ceiling_value : Nat}
          %Equal.sym(Nat,U32.to_nat((Grid.divided_up((first + count : U32),unit) - first / unit : U32)),Nat.sub(ceiling_value,Nat.div(head,size)),count_exact) :
            {Nat.add(Nat.div(head,size),_) == ceiling_value : Nat}
          Order.restore_shift(Nat.div(head,size),ceiling_value,ordered) :
          {Nat.add(U32.to_nat((first / unit : U32)),U32.to_nat((Grid.divided_up((first + count : U32),unit) - first / unit : U32))) == ceiling_value : Nat}}}

# The block span of the last prefix axis lies inside its pieces.
def unit_bound(span: Tasks.Span,+extent: U32,+units: U32,+per: U32,+within: SpanIn(span,extent),
  +fits: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(U32.max(units,1))),U32.to_nat(per)) == True{} : Bool},
  +facts: UnitFacts(span,units)) -> SpanIn(unit_span(span,units),per):
  match span:
    case Tasks.Span{+first,+count}:
      Held.Both{+start,+total} = facts
      +unit = U32.max(units,1)
      +size = U32.to_nat(unit)
      +reach = Nat.add(U32.to_nat(first),U32.to_nat(count))
      +sum = Nat.add(U32.to_nat((first / unit : U32)),U32.to_nat((Grid.divided_up((first + count : U32),unit) - first / unit : U32)))
      %Equal.sym(Nat,sum,NatCeiling.ceiling(reach,size),total) : {Nat.is_le(_,U32.to_nat(per)) == True{} : Bool}
      Order.transitive(NatCeiling.ceiling(reach,size),NatCeiling.ceiling(U32.to_nat(extent),size),U32.to_nat(per),
        NatCeiling.monotone(reach,U32.to_nat(extent),size,positive_max(units),within),fits)

def unit_in(+span: Tasks.Span,+extent: U32,+units: U32,+per: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within: SpanIn(span,extent),
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(units,1))),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(U32.max(units,1))),U32.to_nat(per)) == True{} : Bool}) ->
  SpanIn(unit_span(span,units),per):
  unit_bound(span,extent,units,per,within,fits,unit_facts(span,extent,units,cap,cap_ok,within,room,unit_room))

# ---- Two boxes of one source ----

def grid_prefix(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,+prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  prefix

def grid_units(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,outer,extent,unit_cells,+piece_units,per_outer,pieces} = grid
  piece_units

def block_value(+grid: Grid.Grid,+axis: U32,+extent: U32) ->
  {Grid.block(grid,axis,extent) == Bool.pick(U32,U32.is_lt((axis + 1 : U32),grid_prefix(grid)),1,
    Bool.pick(U32,U32.is_eq((axis + 1 : U32),grid_prefix(grid)),grid_units(grid),extent)) : U32}:
  Grid.Grid{+count,+prefix,+outer,+last,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  {==}

# The extent of the axis steps past the first.
def extent_nat(extents: List<&2,U32>,steps: Nat) -> U32:
  match extents steps:
    case Nil{} steps:
      0
    case Con{+extent,rest} 0n:
      extent
    case Con{+extent,rest} 1n+later:
      extent_nat(rest,later)

# Both axis lists reach steps axes further.
def Reach(extents: List<&2,U32>,coordinates: List<&2,U32>,steps: Nat) -> Data:
  match extents coordinates steps:
    case Con{extent,extents} Con{coordinate,coordinates} 0n:
      Unit
    case Con{extent,extents} Con{coordinate,coordinates} 1n+later:
      Reach(extents,coordinates,later)
    case extents coordinates steps:
      Empty

def next_word(+at: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((at + 1 : U32)) == 1n+position : Nat}:
  Equal.trans(Nat,U32.to_nat((at + 1 : U32)),Nat.add(position,1n),1n+position,
    Arithmetic.sum_exact(cap,cap_ok,at,1,position,1n,exact,{==},
      {%Equal.sym(Nat,Nat.add(position,1n),1n+position,Addition.successor_width(position)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.add(position,1n),U32.to_nat(cap)) == True{} : Bool}}),
    Addition.successor_width(position))

def steps_next(+position: Nat,+later: Nat,+total: Nat,+distance: {Nat.add(position,1n+later) == total : Nat}) -> {Nat.add(1n+position,later) == total : Nat}:
  Equal.trans(Nat,1n+Nat.add(position,later),Nat.add(position,1n+later),total,
    Equal.sym(Nat,Nat.add(position,1n+later),1n+Nat.add(position,later),Addition.successor_right(position,later)),distance)

def steps_room(+position: Nat,+later: Nat,+total: Nat,+distance: {Nat.add(position,1n+later) == total : Nat}) -> {Nat.is_le(1n+position,total) == True{} : Bool}:
  %distance : {Nat.is_le(1n+position,_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(position,1n+later),1n+Nat.add(position,later),Addition.successor_right(position,later)) : {Nat.is_le(1n+position,_) == True{} : Bool}
  Order.le_add(position,later)

def steps_zero(+position: Nat,+total: Nat,+distance: {Nat.add(position,0n) == total : Nat}) -> {position == total : Nat}:
  Equal.trans(Nat,position,Nat.add(position,0n),total,Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)),distance)

def same_axis(+at: U32,+axis: U32,+position: Nat,+exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,0n) == U32.to_nat(axis) : Nat}) ->
  {at == axis : U32}:
  Arithmetic.to_nat_injective(at,axis,Equal.trans(Nat,U32.to_nat(at),position,U32.to_nat(axis),exact,steps_zero(position,U32.to_nat(axis),distance)))

# The axis after at is still a prefix axis but the last.
def kept_lt(+at: U32,+prefix: U32,+position: Nat,+later: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+(1n+later)) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool}:
  +below = {%distance : {Nat.is_lt(1n+position,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(position,1n+(1n+later)),1n+Nat.add(position,1n+later),Addition.successor_right(position,1n+later)) : {Nat.is_lt(1n+position,_) == True{} : Bool}
    below_plus(position,1n+later,{==}) : {Nat.is_lt(1n+position,U32.to_nat(prefix)) == True{} : Bool}}
  +next = next_word(at,position,cap,cap_ok,exact,Order.transitive(1n+position,U32.to_nat(prefix),U32.to_nat(cap),lt_to_le(1n+position,U32.to_nat(prefix),below),prefix_room))
  Equal.trans(Bool,U32.is_lt((at + 1 : U32),prefix),Nat.is_lt(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)),True{},Comparison.lt((at + 1 : U32),prefix),
    {%Equal.sym(Nat,U32.to_nat((at + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == True{} : Bool} below :
      {Nat.is_lt(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)) == True{} : Bool}})

def kept_next(+at: U32,+prefix: U32,+position: Nat,+later: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+later) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((at + 1 : U32)) == 1n+position : Nat}:
  next_word(at,position,cap,cap_ok,exact,Order.transitive(1n+position,U32.to_nat(prefix),U32.to_nat(cap),steps_room(position,later,U32.to_nat(prefix),distance),prefix_room))

def last_value(+position: Nat,+total: Nat,+distance: {Nat.add(position,1n+0n) == total : Nat}) -> {1n+position == total : Nat}:
  Equal.trans(Nat,1n+position,Nat.add(position,1n),total,Equal.sym(Nat,Nat.add(position,1n),1n+position,Addition.successor_width(position)),distance)

# The axis after at is the last prefix axis.
def last_lt(+at: U32,+prefix: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+0n) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool}:
  +next = kept_next(at,prefix,position,0n,cap,cap_ok,exact,distance,prefix_room)
  +top = last_value(position,U32.to_nat(prefix),distance)
  Equal.trans(Bool,U32.is_lt((at + 1 : U32),prefix),Nat.is_lt(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.lt((at + 1 : U32),prefix),
    {%Equal.sym(Nat,U32.to_nat((at + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == False{} : Bool}
      %top : {Nat.is_lt(1n+position,_) == False{} : Bool}
      Order.not_below(1n+position,1n+position,Order.reflexive(1n+position)) : {Nat.is_lt(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}})

def last_eq(+at: U32,+prefix: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+0n) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool}:
  +next = kept_next(at,prefix,position,0n,cap,cap_ok,exact,distance,prefix_room)
  +top = last_value(position,U32.to_nat(prefix),distance)
  Equal.trans(Bool,U32.is_eq((at + 1 : U32),prefix),Nat.is_eq(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)),True{},Comparison.eq((at + 1 : U32),prefix),
    {%Equal.sym(Nat,U32.to_nat((at + 1 : U32)),1n+position,next) : {Nat.is_eq(_,U32.to_nat(prefix)) == True{} : Bool}
      %top : {Nat.is_eq(1n+position,_) == True{} : Bool}
      Order.equal_reflexive(1n+position) : {Nat.is_eq(U32.to_nat((at + 1 : U32)),U32.to_nat(prefix)) == True{} : Bool}})

# ---- Span lists ----

def span_size(+span: Tasks.Span) -> U32:
  match span:
    case Tasks.Span{+first,+count}:
      count

def count_cons(+head: Tasks.Span,+rest: List<&2,Tasks.Span>) -> {Tasks.span_count(Con{head,rest}) == (span_size(head) * Tasks.span_count(rest) : U32) : U32}:
  match head:
    case Tasks.Span{+first,+count}:
      {==}

def blocks_cons(+head: Tasks.Span,+rest: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+axis: U32) ->
  {Tasks.block_spans(Con{head,rest},prefix,units,axis) ==
    Bool.pick(List<&2,Tasks.Span>,U32.is_lt((axis + 1 : U32),prefix),Con{head,Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
      Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{unit_span(head,units),Nil{}},Nil{})) : List<&2,Tasks.Span>}:
  match head:
    case Tasks.Span{+first,+count}:
      {==}

def HeadApart(span: Tasks.Span,other: Tasks.Span) -> Data:
  match span other:
    case Tasks.Span{+first,+count} Tasks.Span{+other_first,+other_count}:
      RangeApart(first,count,other_first,other_count)

def apart_cons(+head: Tasks.Span,+other: Tasks.Span,+rest: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,
  +apart: Either<&2,&2,HeadApart(head,other),SpansApart(rest,others)>) -> SpansApart(Con{head,rest},Con{other,others}):
  match head other:
    case Tasks.Span{+first,+count} Tasks.Span{+other_first,+other_count}:
      apart

def spans_swap(spans: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+apart: SpansApart(spans,others)) -> SpansApart(others,spans):
  match spans others:
    case Nil{} +others:
      Empty.absurd(SpansApart(others,Nil{}),apart)
    case Con{Tasks.Span{+first,+count},+rest} Nil{}:
      Empty.absurd(SpansApart(Nil{},Con{Tasks.Span{first,count},rest}),apart)
    case Con{Tasks.Span{+first,+count},+rest} Con{Tasks.Span{+other_first,+other_count},+more}:
      match apart:
        case Inl{+here}:
          match here:
            case Inl{+before}:
              Inl{Inr{before}}
            case Inr{+after}:
              Inl{Inl{after}}
        case Inr{+later}:
          Inr{spans_swap(rest,more,later)}

# Footprints of a block apart give spans apart at any block.
def index_apart(+span: Tasks.Span,+other: Tasks.Span,+extent: U32,+first: Nat,+last: Nat,+other_first: Nat,+other_last: Nat,+block: Nat,
  +facts: NonEmptySpan(span,extent,first,last,block),+other_facts: NonEmptySpan(other,extent,other_first,other_last,block),
  +before: {Nat.is_lt(last,other_first) == True{} : Bool}) -> HeadApart(span,other):
  match span other:
    case Tasks.Span{+low,+count} Tasks.Span{+other_low,+other_count}:
      Held.Both{+start,+rest} = facts
      Held.Both{+inside,+end} = rest
      Held.Both{+other_start,+other_rest} = other_facts
      Inl{Order.transitive(Nat.add(U32.to_nat(low),U32.to_nat(count)),Nat.mul(1n+last,block),U32.to_nat(other_low),end,
        {%Equal.sym(Nat,U32.to_nat(other_low),Nat.mul(other_first,block),other_start) : {Nat.is_le(Nat.mul(1n+last,block),_) == True{} : Bool}
          RowMajor.mul_le(1n+last,other_first,block,Order.strict_follow(last,other_first,before)) :
          {Nat.is_le(Nat.mul(1n+last,block),U32.to_nat(other_low)) == True{} : Bool}})}

def unit_apart_with(+span: Tasks.Span,+other: Tasks.Span,+extent: U32,+units: U32,+first: Nat,+last: Nat,+other_first: Nat,+other_last: Nat,
  +facts: NonEmptySpan(span,extent,first,last,U32.to_nat(U32.max(units,1))),
  +other_facts: NonEmptySpan(other,extent,other_first,other_last,U32.to_nat(U32.max(units,1))),
  +unit: UnitFacts(span,units),+other_unit: UnitFacts(other,units),+before: {Nat.is_lt(last,other_first) == True{} : Bool}) ->
  HeadApart(unit_span(span,units),unit_span(other,units)):
  match span other:
    case Tasks.Span{+low,+count} Tasks.Span{+other_low,+other_count}:
      Held.Both{+start,+rest} = facts
      Held.Both{+inside,+end} = rest
      Held.Both{+other_start,+other_rest} = other_facts
      Held.Both{+quotient,+sum} = unit
      Held.Both{+other_quotient,+other_sum} = other_unit
      +divisor = U32.max(units,1)
      +size = U32.to_nat(divisor)
      +positive = positive_max(units)
      +reach = Nat.add(U32.to_nat(low),U32.to_nat(count))
      +total = Nat.add(U32.to_nat((low / divisor : U32)),U32.to_nat((Grid.divided_up((low + count : U32),divisor) - low / divisor : U32)))
      +rounded = Order.transitive(NatCeiling.ceiling(reach,size),NatCeiling.ceiling(Nat.mul(1n+last,size),size),1n+last,
        NatCeiling.monotone(reach,Nat.mul(1n+last,size),size,positive,end),
        NatCeiling.minimal(Nat.mul(1n+last,size),size,1n+last,positive,Order.reflexive(Nat.mul(1n+last,size))))
      Inl{{%Equal.sym(Nat,total,NatCeiling.ceiling(reach,size),sum) : {Nat.is_le(_,U32.to_nat((other_low / divisor : U32))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat((other_low / divisor : U32)),Nat.div(U32.to_nat(other_low),size),other_quotient) :
          {Nat.is_le(NatCeiling.ceiling(reach,size),_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(other_low),Nat.mul(other_first,size),other_start) : {Nat.is_le(NatCeiling.ceiling(reach,size),Nat.div(_,size)) == True{} : Bool}
        %Equal.sym(Nat,Nat.div(Nat.mul(other_first,size),size),other_first,Division.exact_quotient(other_first,size,positive)) :
          {Nat.is_le(NatCeiling.ceiling(reach,size),_) == True{} : Bool}
        Order.transitive(NatCeiling.ceiling(reach,size),1n+last,other_first,rounded,Order.strict_follow(last,other_first,before)) :
        {Nat.is_le(total,U32.to_nat((other_low / divisor : U32))) == True{} : Bool}}}

# Past the prefix axes a block is the whole axis, so two footprints that read
# something read the same block.
def past_clash(+last: Nat,+other_first: Nat,+other_last: Nat,+extent: Nat,+ordered: {Nat.is_le(other_first,other_last) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(other_last,extent),extent) == True{} : Bool},+before: {Nat.is_lt(last,other_first) == True{} : Bool}) -> Empty:
  +positive = Order.lt_le(0n,other_first,other_last,Order.le_lt_trans(0n,last,other_first,Order.zero_le(last),before),ordered)
  lt_self(extent,Order.le_lt_trans(extent,Nat.mul(other_last,extent),extent,factor_covers(extent,other_last,positive),bounded))

# A box whose footprint at some axis reads nothing holds no position.
def empty_box(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,coordinates: List<&2,U32>,+at: U32,+position: Nat,
  steps: Nat,+axis: U32,+first: U32,+last: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,steps) == U32.to_nat(axis) : Nat},
  +room: {Nat.is_le(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},+reach: Reach(extents,coordinates,steps),
  +found: {Tasks.read_footprint(row,classes,axis) == Some{Tasks.Footprint{first,last,True{}}} : Maybe<&2,Tasks.Footprint>}) ->
  {Tasks.span_count(Tasks.boxed(row,classes,grid,extents,coordinates,at)) == 0 : U32}:
  match extents coordinates steps:
    case Con{+extent,+rest} Con{+coordinate,+more} 0n:
      +head = Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate)
      +tail = Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32))
      +same = same_axis(at,axis,position,exact,distance)
      %Equal.sym(U32,Tasks.span_count(Con{head,tail}),(span_size(head) * Tasks.span_count(tail) : U32),count_cons(head,tail)) : {_ == 0 : U32}
      %Equal.sym(U32,at,axis,same) :
        {(span_size(Tasks.box_span(Tasks.read_footprint(row,classes,_),extent,Grid.block(grid,at,extent),coordinate)) * Tasks.span_count(tail) : U32) == 0 : U32}
      %Equal.sym(Maybe<&2,Tasks.Footprint>,Tasks.read_footprint(row,classes,axis),Some{Tasks.Footprint{first,last,True{}}},found) :
        {(span_size(Tasks.box_span(_,extent,Grid.block(grid,at,extent),coordinate)) * Tasks.span_count(tail) : U32) == 0 : U32}
      Arithmetic.multiply_zero_left(Tasks.span_count(tail))
    case Con{+extent,+rest} Con{+coordinate,+more} 1n+ +later:
      +head = Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate)
      +tail = Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32))
      +next = next_word(at,position,cap,cap_ok,exact,Order.transitive(1n+position,U32.to_nat(axis),U32.to_nat(cap),steps_room(position,later,U32.to_nat(axis),distance),room))
      +empty = empty_box(row,classes,grid,rest,more,(at + 1 : U32),1n+position,later,axis,first,last,cap,cap_ok,next,steps_next(position,later,U32.to_nat(axis),distance),
        room,reach,found)
      %Equal.sym(U32,Tasks.span_count(Con{head,tail}),(span_size(head) * Tasks.span_count(tail) : U32),count_cons(head,tail)) : {_ == 0 : U32}
      %Equal.sym(U32,Tasks.span_count(tail),0,empty) : {(span_size(head) * _ : U32) == 0 : U32}
      Arithmetic.multiply_zero_right(span_size(head))
    case Nil{} +coordinates +steps:
      Empty.absurd({Tasks.span_count(Tasks.boxed(row,classes,grid,Nil{},coordinates,at)) == 0 : U32},reach)
    case Con{+extent,+rest} Nil{} +steps:
      Empty.absurd({Tasks.span_count(Tasks.boxed(row,classes,grid,Con{extent,rest},Nil{},at)) == 0 : U32},reach)

# Spans apart at a prefix axis but the last keep the block spans apart.
def apart_kept(+head: Tasks.Span,+other: Tasks.Span,+tail: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+at: U32,
  +kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool},
  +apart: Either<&2,&2,HeadApart(head,other),SpansApart(Tasks.block_spans(tail,prefix,units,(at + 1 : U32)),Tasks.block_spans(others,prefix,units,(at + 1 : U32)))>) ->
  SpansApart(Tasks.block_spans(Con{head,tail},prefix,units,at),Tasks.block_spans(Con{other,others},prefix,units,at)):
  +rest = Tasks.block_spans(tail,prefix,units,(at + 1 : U32))
  +more = Tasks.block_spans(others,prefix,units,(at + 1 : U32))
  +last = U32.is_eq((at + 1 : U32),prefix)
  %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),
      Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},Bool.pick(List<&2,Tasks.Span>,last,Con{unit_span(head,units),Nil{}},Nil{})),
      blocks_cons(head,tail,prefix,units,at)) : SpansApart(_,Tasks.block_spans(Con{other,others},prefix,units,at))
  %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{other,others},prefix,units,at),
      Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{other,more},Bool.pick(List<&2,Tasks.Span>,last,Con{unit_span(other,units),Nil{}},Nil{})),
      blocks_cons(other,others,prefix,units,at)) :
    SpansApart(Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},Bool.pick(List<&2,Tasks.Span>,last,Con{unit_span(head,units),Nil{}},Nil{})),_)
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},kept) :
    SpansApart(Bool.pick(List<&2,Tasks.Span>,_,Con{head,rest},Bool.pick(List<&2,Tasks.Span>,last,Con{unit_span(head,units),Nil{}},Nil{})),
      Bool.pick(List<&2,Tasks.Span>,_,Con{other,more},Bool.pick(List<&2,Tasks.Span>,last,Con{unit_span(other,units),Nil{}},Nil{})))
  apart_cons(head,other,rest,more,apart)

# Spans apart at the last prefix axis keep its block spans apart.
def apart_last(+head: Tasks.Span,+other: Tasks.Span,+tail: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+at: U32,
  +kept: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},+last: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool},
  +apart: HeadApart(unit_span(head,units),unit_span(other,units))) ->
  SpansApart(Tasks.block_spans(Con{head,tail},prefix,units,at),Tasks.block_spans(Con{other,others},prefix,units,at)):
  +rest = Tasks.block_spans(tail,prefix,units,(at + 1 : U32))
  +more = Tasks.block_spans(others,prefix,units,(at + 1 : U32))
  %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),
      Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},
        Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(head,units),Nil{}},Nil{})),
      blocks_cons(head,tail,prefix,units,at)) : SpansApart(_,Tasks.block_spans(Con{other,others},prefix,units,at))
  %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{other,others},prefix,units,at),
      Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{other,more},
        Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(other,units),Nil{}},Nil{})),
      blocks_cons(other,others,prefix,units,at)) :
    SpansApart(Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},
      Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(head,units),Nil{}},Nil{})),_)
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},kept) :
    SpansApart(Bool.pick(List<&2,Tasks.Span>,_,Con{head,rest},Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(head,units),Nil{}},Nil{})),
      Bool.pick(List<&2,Tasks.Span>,_,Con{other,more},Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(other,units),Nil{}},Nil{})))
  %Equal.sym(Bool,U32.is_eq((at + 1 : U32),prefix),True{},last) :
    SpansApart(Bool.pick(List<&2,Tasks.Span>,False{},Con{head,rest},Bool.pick(List<&2,Tasks.Span>,_,Con{unit_span(head,units),Nil{}},Nil{})),
      Bool.pick(List<&2,Tasks.Span>,False{},Con{other,more},Bool.pick(List<&2,Tasks.Span>,_,Con{unit_span(other,units),Nil{}},Nil{})))
  apart_cons(unit_span(head,units),unit_span(other,units),Nil{},Nil{},Inl{apart})

# The span of a footprint that reads blocks first to last at the axis a box
# walk reached.
def head_facts(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+extent: U32,+coordinate: U32,+at: U32,+axis: U32,
  +first: U32,+last: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+same: {at == axis : U32},
  +found: {Tasks.read_footprint(row,classes,axis) == Some{Tasks.Footprint{first,last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+last_room: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(U32.to_nat(last),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(extent)) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool}) ->
  NonEmptySpan(Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate),extent,U32.to_nat(first),U32.to_nat(last),
    U32.to_nat(Grid.block(grid,at,extent))):
  %Equal.sym(U32,at,axis,same) :
    NonEmptySpan(Tasks.box_span(Tasks.read_footprint(row,classes,_),extent,Grid.block(grid,_,extent),coordinate),extent,U32.to_nat(first),U32.to_nat(last),
      U32.to_nat(Grid.block(grid,_,extent)))
  %Equal.sym(Maybe<&2,Tasks.Footprint>,Tasks.read_footprint(row,classes,axis),Some{Tasks.Footprint{first,last,False{}}},found) :
    NonEmptySpan(Tasks.box_span(_,extent,Grid.block(grid,axis,extent),coordinate),extent,U32.to_nat(first),U32.to_nat(last),U32.to_nat(Grid.block(grid,axis,extent)))
  full_span(first,last,extent,Grid.block(grid,axis,extent),coordinate,cap,cap_ok,ordered,last_room,bounded,axis_room)

def unit_block(+grid: Grid.Grid,+at: U32,+extent: U32,+kept: {U32.is_lt((at + 1 : U32),grid_prefix(grid)) == False{} : Bool},
  +last: {U32.is_eq((at + 1 : U32),grid_prefix(grid)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool}) ->
  {U32.to_nat(Grid.block(grid,at,extent)) == U32.to_nat(U32.max(grid_units(grid),1)) : Nat}:
  +units = grid_units(grid)
  +same = Equal.trans(U32,Grid.block(grid,at,extent),
    Bool.pick(U32,U32.is_lt((at + 1 : U32),grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),grid_prefix(grid)),units,extent)),units,
    block_value(grid,at,extent),
    {%Equal.sym(Bool,U32.is_lt((at + 1 : U32),grid_prefix(grid)),False{},kept) :
        {Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),grid_prefix(grid)),units,extent)) == units : U32}
      %Equal.sym(Bool,U32.is_eq((at + 1 : U32),grid_prefix(grid)),True{},last) : {Bool.pick(U32,False{},1,Bool.pick(U32,_,units,extent)) == units : U32}
      {==} : {Bool.pick(U32,U32.is_lt((at + 1 : U32),grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),grid_prefix(grid)),units,extent)) == units : U32}})
  Equal.trans(Nat,U32.to_nat(Grid.block(grid,at,extent)),U32.to_nat(units),U32.to_nat(U32.max(units,1)),
    Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.block(grid,at,extent),units,same),
    Equal.sym(Nat,U32.to_nat(U32.max(units,1)),U32.to_nat(units),Extrema.max_one_exact(units,U32.to_nat(units),{==},positive)))

# Two boxes of one source whose footprints read blocks apart at axis keep
# their block spans apart.
def boxes_walk(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,
  coordinates: List<&2,U32>,+at: U32,+position: Nat,steps: Nat,remaining: Nat,+axis: U32,+first: U32,+last: U32,+other_first: U32,+other_last: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,steps) == U32.to_nat(axis) : Nat},
  +prefix_distance: {Nat.add(position,1n+remaining) == U32.to_nat(grid_prefix(grid)) : Nat},+within: {Nat.is_le(steps,remaining) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},+reach: Reach(extents,coordinates,steps),
  +found: {Tasks.read_footprint(row,classes,axis) == Some{Tasks.Footprint{first,last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +other_found: {Tasks.read_footprint(row,others,axis) == Some{Tasks.Footprint{other_first,other_last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+last_room: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +other_ordered: {Nat.is_le(U32.to_nat(other_first),U32.to_nat(other_last)) == True{} : Bool},
  +other_last_room: {Nat.is_lt(U32.to_nat(other_last),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(U32.to_nat(last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,steps)))),U32.to_nat(extent_nat(extents,steps))) == True{} : Bool},
  +other_bounded: {Nat.is_lt(Nat.mul(U32.to_nat(other_last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,steps)))),U32.to_nat(extent_nat(extents,steps))) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(U32.to_nat(extent_nat(extents,steps)),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,steps)))),U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(last),U32.to_nat(other_first)) == True{} : Bool}) ->
  SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,coordinates,at),grid_prefix(grid),grid_units(grid),at),
    Tasks.block_spans(Tasks.boxed(row,others,grid,extents,coordinates,at),grid_prefix(grid),grid_units(grid),at)):
  match extents coordinates steps remaining:
    case Con{+extent,+rest} Con{+coordinate,+more} 0n 0n:
      +prefix = grid_prefix(grid)
      +units = grid_units(grid)
      +same = same_axis(at,axis,position,exact,distance)
      +kept = last_lt(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +last_axis = last_eq(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +head = Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate)
      +other_head = Tasks.box_span(Tasks.read_footprint(row,others,at),extent,Grid.block(grid,at,extent),coordinate)
      +size = U32.to_nat(U32.max(units,1))
      +size_same = unit_block(grid,at,extent,kept,last_axis,units_positive)
      +facts = {%size_same : NonEmptySpan(head,extent,U32.to_nat(first),U32.to_nat(last),_)
        head_facts(row,classes,grid,extent,coordinate,at,axis,first,last,cap,cap_ok,same,found,ordered,last_room,bounded,axis_room) :
        NonEmptySpan(head,extent,U32.to_nat(first),U32.to_nat(last),size)}
      +other_facts = {%size_same : NonEmptySpan(other_head,extent,U32.to_nat(other_first),U32.to_nat(other_last),_)
        head_facts(row,others,grid,extent,coordinate,at,axis,other_first,other_last,cap,cap_ok,same,other_found,other_ordered,other_last_room,other_bounded,axis_room) :
        NonEmptySpan(other_head,extent,U32.to_nat(other_first),U32.to_nat(other_last),size)}
      +extent_room = Order.transitive(U32.to_nat(extent),Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap),
        Order.le_add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),axis_room)
      apart_last(head,other_head,Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32)),Tasks.boxed(row,others,grid,rest,more,(at + 1 : U32)),prefix,units,at,kept,last_axis,
        unit_apart_with(head,other_head,extent,units,U32.to_nat(first),U32.to_nat(last),U32.to_nat(other_first),U32.to_nat(other_last),facts,other_facts,
          unit_facts(head,extent,units,cap,cap_ok,nonempty_in(head,extent,U32.to_nat(first),U32.to_nat(last),size,facts),extent_room,unit_room),
          unit_facts(other_head,extent,units,cap,cap_ok,nonempty_in(other_head,extent,U32.to_nat(other_first),U32.to_nat(other_last),size,other_facts),extent_room,unit_room),
          before))
    case Con{+extent,+rest} Con{+coordinate,+more} 0n 1n+ +later:
      +prefix = grid_prefix(grid)
      +units = grid_units(grid)
      +same = same_axis(at,axis,position,exact,distance)
      +kept = kept_lt(at,prefix,position,later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +head = Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate)
      +other_head = Tasks.box_span(Tasks.read_footprint(row,others,at),extent,Grid.block(grid,at,extent),coordinate)
      +size = U32.to_nat(Grid.block(grid,at,extent))
      apart_kept(head,other_head,Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32)),Tasks.boxed(row,others,grid,rest,more,(at + 1 : U32)),prefix,units,at,kept,
        Inl{index_apart(head,other_head,extent,U32.to_nat(first),U32.to_nat(last),U32.to_nat(other_first),U32.to_nat(other_last),size,
          head_facts(row,classes,grid,extent,coordinate,at,axis,first,last,cap,cap_ok,same,found,ordered,last_room,bounded,axis_room),
          head_facts(row,others,grid,extent,coordinate,at,axis,other_first,other_last,cap,cap_ok,same,other_found,other_ordered,other_last_room,other_bounded,axis_room),
          before)})
    case Con{+extent,+rest} Con{+coordinate,+more} 1n+ +later 0n:
      Empty.absurd(SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,Con{extent,rest},Con{coordinate,more},at),grid_prefix(grid),grid_units(grid),at),
        Tasks.block_spans(Tasks.boxed(row,others,grid,Con{extent,rest},Con{coordinate,more},at),grid_prefix(grid),grid_units(grid),at)),WordEncoding.false_true(within))
    case Con{+extent,+rest} Con{+coordinate,+more} 1n+ +later 1n+ +left:
      +prefix = grid_prefix(grid)
      +units = grid_units(grid)
      +kept = kept_lt(at,prefix,position,left,cap,cap_ok,exact,prefix_distance,prefix_room)
      +next = kept_next(at,prefix,position,1n+left,cap,cap_ok,exact,prefix_distance,prefix_room)
      +head = Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate)
      +other_head = Tasks.box_span(Tasks.read_footprint(row,others,at),extent,Grid.block(grid,at,extent),coordinate)
      apart_kept(head,other_head,Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32)),Tasks.boxed(row,others,grid,rest,more,(at + 1 : U32)),prefix,units,at,kept,
        Inr{boxes_walk(row,classes,others,grid,rest,more,(at + 1 : U32),1n+position,later,left,axis,first,last,other_first,other_last,cap,cap_ok,next,
          steps_next(position,later,U32.to_nat(axis),distance),steps_next(position,1n+left,U32.to_nat(prefix),prefix_distance),within,prefix_room,reach,
          found,other_found,ordered,last_room,other_ordered,other_last_room,bounded,other_bounded,axis_room,units_positive,unit_room,before)})
    case Nil{} +coordinates +steps +remaining:
      Empty.absurd(SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,Nil{},coordinates,at),grid_prefix(grid),grid_units(grid),at),
        Tasks.block_spans(Tasks.boxed(row,others,grid,Nil{},coordinates,at),grid_prefix(grid),grid_units(grid),at)),reach)
    case Con{+extent,+rest} Nil{} +steps +remaining:
      Empty.absurd(SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,Con{extent,rest},Nil{},at),grid_prefix(grid),grid_units(grid),at),
        Tasks.block_spans(Tasks.boxed(row,others,grid,Con{extent,rest},Nil{},at),grid_prefix(grid),grid_units(grid),at)),reach)

# ---- The boxes of two components of one source ----

# Every axis a row reads lies below limit.
def AxesBelow(axes: List<&2,U32>,+limit: Nat) -> Data:
  match axes:
    case Nil{}:
      Unit
    case Con{+axis,rest}:
      Held.Both<{Nat.is_lt(U32.to_nat(axis),limit) == True{} : Bool},AxesBelow(rest,limit)>

def held_below(axes: List<&2,U32>,+axis: U32,+limit: Nat,+below: AxesBelow(axes,limit),+held: {Tasks.holds(axes,axis) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(axis),limit) == True{} : Bool}:
  match axes:
    case Nil{}:
      Empty.absurd({Nat.is_lt(U32.to_nat(axis),limit) == True{} : Bool},WordEncoding.false_true(held))
    case Con{+other,+rest}:
      Held.Both{+here,+later} = below
      Held.by_cases({Nat.is_lt(U32.to_nat(axis),limit) == True{} : Bool},U32.is_eq(other,axis),
        same => {%Arithmetic.equal_of_is_eq(other,axis,same) : {Nat.is_lt(U32.to_nat(_),limit) == True{} : Bool} here :
          {Nat.is_lt(U32.to_nat(axis),limit) == True{} : Bool}},
        apart => held_below(rest,axis,limit,later,{%apart : {(_ || Tasks.holds(rest,axis)) == True{} : Bool} held : {Tasks.holds(rest,axis) == True{} : Bool}}))

def reach_coordinates(extents: List<&2,U32>,+offset: U32,steps: Nat,+inside: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool}) ->
  Reach(extents,GridLaws.coordinates_model(extents,offset),steps):
  match extents steps:
    case Nil{} +steps:
      Empty.absurd(Reach(Nil{},GridLaws.coordinates_model(Nil{},offset),steps),Order.below_zero(steps,inside))
    case Con{+extent,+rest} 0n:
      Unit{}
    case Con{+extent,+rest} 1n+ +later:
      reach_coordinates(rest,(offset % U32.max(Grid.product(rest),1) : U32),later,inside)

# A source axis is read in the blocks of the grid and its blocks fit cap.
def AxisOk(+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis: U32,+extent: U32,+cap: U32) -> Data:
  Held.Both<ReadsAxis(row,grid,axis,extent),{Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool}>

def AxesOk(+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,extents: List<&2,U32>,+at: U32,+cap: U32) -> Data:
  match extents:
    case Nil{}:
      Unit
    case Con{+extent,rest}:
      Held.Both<AxisOk(row,grid,at,extent,cap),AxesOk(row,grid,rest,(at + 1 : U32),cap)>

def axes_at(+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,extents: List<&2,U32>,+at: U32,+position: Nat,steps: Nat,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,steps) == U32.to_nat(axis) : Nat},
  +room: {Nat.is_le(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool},
  +axes: AxesOk(row,grid,extents,at,cap)) -> AxisOk(row,grid,axis,extent_nat(extents,steps),cap):
  match extents steps:
    case Nil{} +steps:
      Empty.absurd(AxisOk(row,grid,axis,extent_nat(Nil{},steps),cap),Order.below_zero(steps,inside))
    case Con{+extent,+rest} 0n:
      Held.Both{+here,+later} = axes
      {%same_axis(at,axis,position,exact,distance) : AxisOk(row,grid,_,extent,cap) here : AxisOk(row,grid,axis,extent,cap)}
    case Con{+extent,+rest} 1n+ +later:
      Held.Both{+here,+more} = axes
      axes_at(row,grid,rest,(at + 1 : U32),1n+position,later,axis,cap,cap_ok,
        next_word(at,position,cap,cap_ok,exact,Order.transitive(1n+position,U32.to_nat(axis),U32.to_nat(cap),steps_room(position,later,U32.to_nat(axis),distance),room)),
        steps_next(position,later,U32.to_nat(axis),distance),room,inside,more)

def BoxesApart(+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,+prefix: U32,+units: U32) -> Data:
  Either<&2,&2,{Tasks.span_count(box) == 0 : U32},Either<&2,&2,{Tasks.span_count(other) == 0 : U32},
    SpansApart(Tasks.block_spans(box,prefix,units,0),Tasks.block_spans(other,prefix,units,0))>>

def past_block(+grid: Grid.Grid,+axis: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +past: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(axis)) == True{} : Bool},+room: {Nat.is_lt(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool}) ->
  {Grid.block(grid,axis,extent) == extent : U32}:
  +prefix = grid_prefix(grid)
  +value = U32.to_nat(axis)
  +next = next_word(axis,value,cap,cap_ok,{==},Order.strict_follow(value,U32.to_nat(cap),room))
  Equal.trans(U32,Grid.block(grid,axis,extent),
    Bool.pick(U32,U32.is_lt((axis + 1 : U32),prefix),1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),grid_units(grid),extent)),extent,block_value(grid,axis,extent),
    {%Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},
        Equal.trans(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.lt((axis + 1 : U32),prefix),
          {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+value,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == False{} : Bool}
            Order.not_below(U32.to_nat(prefix),1n+value,Order.successor(U32.to_nat(prefix),value,past)) :
            {Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}})) :
        {Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),grid_units(grid),extent)) == extent : U32}
      %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),False{},
        Equal.trans(Bool,U32.is_eq((axis + 1 : U32),prefix),Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.eq((axis + 1 : U32),prefix),
          {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+value,next) : {Nat.is_eq(_,U32.to_nat(prefix)) == False{} : Bool}
            %Order.equal_symmetric(U32.to_nat(prefix),1n+value) : {_ == False{} : Bool}
            Order.less_not_equal(U32.to_nat(prefix),1n+value,Order.succ_le_lt(U32.to_nat(prefix),1n+value,past)) :
            {Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}})) :
        {Bool.pick(U32,False{},1,Bool.pick(U32,_,grid_units(grid),extent)) == extent : U32}
      {==} : {Bool.pick(U32,U32.is_lt((axis + 1 : U32),prefix),1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),grid_units(grid),extent)) == extent : U32}})

def boxes_from(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,
  +axis: U32,+first: U32,+last: U32,+other_first: U32,+other_last: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,extents)) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},
  +found: {Tasks.read_footprint(row,classes,axis) == Some{Tasks.Footprint{first,last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +other_found: {Tasks.read_footprint(row,others,axis) == Some{Tasks.Footprint{other_first,other_last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+last_room: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +other_ordered: {Nat.is_le(U32.to_nat(other_first),U32.to_nat(other_last)) == True{} : Bool},
  +other_last_room: {Nat.is_lt(U32.to_nat(other_last),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(U32.to_nat(last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(extent_nat(extents,U32.to_nat(axis)))) == True{} : Bool},
  +other_bounded: {Nat.is_lt(Nat.mul(U32.to_nat(other_last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(extent_nat(extents,U32.to_nat(axis)))) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(U32.to_nat(extent_nat(extents,U32.to_nat(axis))),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(last),U32.to_nat(other_first)) == True{} : Bool},
  +within: {Nat.is_lt(U32.to_nat(axis),U32.to_nat(grid_prefix(grid))) == True{} : Bool}) ->
  SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
    Tasks.block_spans(Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0)):
  +value = U32.to_nat(axis)
  +limit = U32.to_nat(grid_prefix(grid))
  boxes_walk(row,classes,others,grid,extents,GridLaws.coordinates_model(extents,offset),0,0n,value,Nat.sub(limit,1n),axis,first,last,other_first,other_last,cap,cap_ok,
    {==},{==},Order.positive_has_last(limit,Order.le_lt_trans(0n,value,limit,Order.zero_le(value),within)),pred_le(value,limit,within),prefix_room,
    reach_coordinates(extents,offset,value,inside),found,other_found,ordered,last_room,other_ordered,other_last_room,bounded,other_bounded,axis_room,
    units_positive,unit_room,before)

# Two footprints that read blocks apart at axis, the first before the second,
# keep the block spans of their boxes apart.
def ordered_apart(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,
  +axis: U32,+first: U32,+last: U32,+other_first: U32,+other_last: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,extents)) == True{} : Bool},+room: {Nat.is_lt(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},
  +found: {Tasks.read_footprint(row,classes,axis) == Some{Tasks.Footprint{first,last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +other_found: {Tasks.read_footprint(row,others,axis) == Some{Tasks.Footprint{other_first,other_last,False{}}} : Maybe<&2,Tasks.Footprint>},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},+last_room: {Nat.is_lt(U32.to_nat(last),U32.to_nat(cap)) == True{} : Bool},
  +other_ordered: {Nat.is_le(U32.to_nat(other_first),U32.to_nat(other_last)) == True{} : Bool},
  +other_last_room: {Nat.is_lt(U32.to_nat(other_last),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_lt(Nat.mul(U32.to_nat(last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(extent_nat(extents,U32.to_nat(axis)))) == True{} : Bool},
  +other_bounded: {Nat.is_lt(Nat.mul(U32.to_nat(other_last),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(extent_nat(extents,U32.to_nat(axis)))) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(U32.to_nat(extent_nat(extents,U32.to_nat(axis))),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(last),U32.to_nat(other_first)) == True{} : Bool}) ->
  SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
    Tasks.block_spans(Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0)):
  +value = U32.to_nat(axis)
  +limit = U32.to_nat(grid_prefix(grid))
  +extent = extent_nat(extents,value)
  Held.by_cases(SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
    Tasks.block_spans(Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0)),Nat.is_lt(value,limit),
    within => boxes_from(row,classes,others,grid,extents,offset,axis,first,last,other_first,other_last,cap,cap_ok,inside,prefix_room,found,other_found,
      ordered,last_room,other_ordered,other_last_room,bounded,other_bounded,axis_room,units_positive,unit_room,before,within),
    past => Empty.absurd(SpansApart(Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
    Tasks.block_spans(Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0)),past_clash(U32.to_nat(last),U32.to_nat(other_first),U32.to_nat(other_last),U32.to_nat(extent),other_ordered,
      {%past_block(grid,axis,extent,cap,cap_ok,Held.lt_false_le(value,limit,past),room) :
          {Nat.is_lt(Nat.mul(U32.to_nat(other_last),U32.to_nat(_)),U32.to_nat(extent)) == True{} : Bool}
        other_bounded : {Nat.is_lt(Nat.mul(U32.to_nat(other_last),U32.to_nat(extent)),U32.to_nat(extent)) == True{} : Bool}},before)))

def boxes_cases(found: Maybe<&2,Tasks.Footprint>,other: Maybe<&2,Tasks.Footprint>,+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,
  +others: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +found_same: {Tasks.read_footprint(row,classes,axis) == found : Maybe<&2,Tasks.Footprint>},
  +other_same: {Tasks.read_footprint(row,others,axis) == other : Maybe<&2,Tasks.Footprint>},+maybe: MaybeApart(found,other),
  +ok: FoundOk(found,Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))),extent_nat(extents,U32.to_nat(axis)),cap),
  +other_ok: FoundOk(other,Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))),extent_nat(extents,U32.to_nat(axis)),cap),
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,extents)) == True{} : Bool},+room: {Nat.is_lt(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(U32.to_nat(extent_nat(extents,U32.to_nat(axis))),U32.to_nat(Grid.block(grid,axis,extent_nat(extents,U32.to_nat(axis))))),
    U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool}) ->
  BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),
    grid_prefix(grid),grid_units(grid)):
  match found other:
    case None{} +other:
      Empty.absurd(BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),
        Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid)),maybe)
    case Some{Tasks.Footprint{+first,+last,True{}}} +other:
      Inl{empty_box(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0,0n,U32.to_nat(axis),axis,first,last,cap,cap_ok,{==},{==},
        lt_to_le(U32.to_nat(axis),U32.to_nat(cap),room),reach_coordinates(extents,offset,U32.to_nat(axis),inside),found_same)}
    case Some{Tasks.Footprint{+first,+last,False{}}} None{}:
      Empty.absurd(BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),
        Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid)),maybe)
    case Some{Tasks.Footprint{+first,+last,False{}}} Some{Tasks.Footprint{+other_first,+other_last,True{}}}:
      Inr{Inl{empty_box(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0,0n,U32.to_nat(axis),axis,other_first,other_last,cap,cap_ok,{==},{==},
        lt_to_le(U32.to_nat(axis),U32.to_nat(cap),room),reach_coordinates(extents,offset,U32.to_nat(axis),inside),other_same)}}
    case Some{Tasks.Footprint{+first,+last,False{}}} Some{Tasks.Footprint{+other_first,+other_last,False{}}}:
      match maybe:
        case Inl{+no}:
          Empty.absurd(BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),
            Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid)),WordEncoding.false_true(no))
        case Inr{Inl{+no}}:
          Empty.absurd(BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),
            Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid)),WordEncoding.false_true(no))
        case Inr{Inr{Inl{+before}}}:
          Held.Both{+formed,+bounded} = ok
          Held.Both{+ordered,+last_room} = formed
          Held.Both{+other_formed,+other_bounded} = other_ok
          Held.Both{+other_ordered,+other_last_room} = other_formed
          Inr{Inr{ordered_apart(row,classes,others,grid,extents,offset,axis,first,last,other_first,other_last,cap,cap_ok,inside,room,prefix_room,found_same,other_same,
            ordered,last_room,other_ordered,other_last_room,bounded,other_bounded,axis_room,units_positive,unit_room,before)}}
        case Inr{Inr{Inr{+after}}}:
          Held.Both{+formed,+bounded} = ok
          Held.Both{+ordered,+last_room} = formed
          Held.Both{+other_formed,+other_bounded} = other_ok
          Held.Both{+other_ordered,+other_last_room} = other_formed
          Inr{Inr{spans_swap(Tasks.block_spans(Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
            Tasks.block_spans(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),grid_prefix(grid),grid_units(grid),0),
            ordered_apart(row,others,classes,grid,extents,offset,axis,other_first,other_last,first,last,cap,cap_ok,inside,room,prefix_room,other_same,found_same,
              other_ordered,other_last_room,ordered,last_room,other_bounded,bounded,axis_room,units_positive,unit_room,after))}}

def boxes_at(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,
  +axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +maybe: MaybeApart(Tasks.read_footprint(row,classes,axis),Tasks.read_footprint(row,others,axis)),
  +at_ok: AxisOk(row,grid,axis,extent_nat(extents,U32.to_nat(axis)),cap),+rooms: ReadingsRoom(row,cap),+ends: RowEnds(row,classes),+other_ends: RowEnds(row,others),
  +inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,extents)) == True{} : Bool},+room: {Nat.is_lt(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool}) ->
  BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),
    grid_prefix(grid),grid_units(grid)):
  Held.Both{+reads,+axis_room} = at_ok
  +extent = extent_nat(extents,U32.to_nat(axis))
  boxes_cases(Tasks.read_footprint(row,classes,axis),Tasks.read_footprint(row,others,axis),row,classes,others,grid,extents,offset,axis,cap,cap_ok,{==},{==},maybe,
    found_ok(row,classes,grid,axis,extent,cap,cap_ok,rooms,ends,reads),found_ok(row,others,grid,axis,extent,cap,cap_ok,rooms,other_ends,reads),
    inside,room,prefix_room,axis_room,units_positive,unit_room)

# Two components that read blocks apart at some target axis read boxes of a
# source with no block in common, or one of them reads an empty box.
def boxes_apart(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: FirstApart(row,classes,others),
  +distinct: {Tasks.distinct(Tasks.read_axes(row)) == True{} : Bool},+rooms: ReadingsRoom(row,cap),+ends: RowEnds(row,classes),+other_ends: RowEnds(row,others),
  +axes: AxesOk(row,grid,extents,0,cap),+below: AxesBelow(Tasks.read_axes(row),List.length(&2,U32,extents)),
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +prefix_room: {Nat.is_le(U32.to_nat(grid_prefix(grid)),U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(grid_units(grid))) == True{} : Bool},
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(grid_units(grid),1))),U32.to_nat(cap)) == True{} : Bool}) ->
  BoxesApart(Tasks.boxed(row,classes,grid,extents,GridLaws.coordinates_model(extents,offset),0),Tasks.boxed(row,others,grid,extents,GridLaws.coordinates_model(extents,offset),0),
    grid_prefix(grid),grid_units(grid)):
  +axis = apart_axis(row,classes,others,first)
  +inside = held_below(Tasks.read_axes(row),axis,List.length(&2,U32,extents),below,apart_axis_held(row,classes,others,first))
  +room = Order.lt_le(U32.to_nat(axis),List.length(&2,U32,extents),U32.to_nat(cap),inside,rank_room)
  boxes_at(row,classes,others,grid,extents,offset,axis,cap,cap_ok,footprints_apart(row,classes,others,first,distinct),
    axes_at(row,grid,extents,0,0n,U32.to_nat(axis),axis,cap,cap_ok,{==},{==},lt_to_le(U32.to_nat(axis),U32.to_nat(cap),room),inside,axes),
    rooms,ends,other_ends,inside,room,prefix_room,units_positive,unit_room)

# ---- The block spans of a box lie inside the pieces ----

def spans_in_cons(+span: Tasks.Span,+rest: List<&2,Tasks.Span>,+extent: U32,+extents: List<&2,U32>,+inside: SpanIn(span,extent),+later: SpansIn(rest,extents)) ->
  SpansIn(Con{span,rest},Con{extent,extents}):
  match span:
    case Tasks.Span{+first,+count}:
      Held.Both{inside,later}

def kept_extents(+extent: U32,+rest: List<&2,U32>,+prefix: U32,+per: U32,+at: U32,+kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool}) ->
  {Tasks.piece_extents(Con{extent,rest},prefix,per,at) == Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))} : List<&2,U32>}:
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},kept) :
    {Bool.pick(List<&2,U32>,_,Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))},Con{per,Nil{}}) ==
      Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))} : List<&2,U32>}
  {==}

def last_extents(+extent: U32,+rest: List<&2,U32>,+prefix: U32,+per: U32,+at: U32,+kept: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool}) ->
  {Tasks.piece_extents(Con{extent,rest},prefix,per,at) == Con{per,Nil{}} : List<&2,U32>}:
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},kept) :
    {Bool.pick(List<&2,U32>,_,Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))},Con{per,Nil{}}) == Con{per,Nil{}} : List<&2,U32>}
  {==}

def kept_blocks(+head: Tasks.Span,+tail: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+at: U32,+kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool}) ->
  {Tasks.block_spans(Con{head,tail},prefix,units,at) == Con{head,Tasks.block_spans(tail,prefix,units,(at + 1 : U32))} : List<&2,Tasks.Span>}:
  +rest = Tasks.block_spans(tail,prefix,units,(at + 1 : U32))
  +last = Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),Con{unit_span(head,units),Nil{}},Nil{})
  Equal.trans(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},last),
    Con{head,rest},blocks_cons(head,tail,prefix,units,at),
    {%Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},kept) : {Bool.pick(List<&2,Tasks.Span>,_,Con{head,rest},last) == Con{head,rest} : List<&2,Tasks.Span>}
      {==} : {Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},last) == Con{head,rest} : List<&2,Tasks.Span>}})

def last_blocks(+head: Tasks.Span,+tail: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+at: U32,+kept: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},
  +last: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool}) ->
  {Tasks.block_spans(Con{head,tail},prefix,units,at) == Con{unit_span(head,units),Nil{}} : List<&2,Tasks.Span>}:
  +rest = Tasks.block_spans(tail,prefix,units,(at + 1 : U32))
  +unit = {Con{unit_span(head,units),Nil{}} : List<&2,Tasks.Span>}
  Equal.trans(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),
    Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),unit,Nil{})),unit,
    blocks_cons(head,tail,prefix,units,at),
    {%Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},kept) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{head,rest},Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),unit,Nil{})) == unit : List<&2,Tasks.Span>}
      %Equal.sym(Bool,U32.is_eq((at + 1 : U32),prefix),True{},last) :
        {Bool.pick(List<&2,Tasks.Span>,False{},Con{head,rest},Bool.pick(List<&2,Tasks.Span>,_,unit,Nil{})) == unit : List<&2,Tasks.Span>}
      {==} : {Bool.pick(List<&2,Tasks.Span>,U32.is_lt((at + 1 : U32),prefix),Con{head,rest},Bool.pick(List<&2,Tasks.Span>,U32.is_eq((at + 1 : U32),prefix),unit,Nil{})) ==
        unit : List<&2,Tasks.Span>}})

# The spans of a box up to the last prefix axis lie inside their axes, and the
# axes fit cap.
def SpansAlong(box: List<&2,Tasks.Span>,extents: List<&2,U32>,steps: Nat,+cap: U32) -> Data:
  match box extents steps:
    case Con{+span,box} Con{+extent,extents} 0n:
      Held.Both<SpanIn(span,extent),{Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}>
    case Con{+span,box} Con{+extent,extents} 1n+later:
      Held.Both<SpanIn(span,extent),SpansAlong(box,extents,later,cap)>
    case box extents steps:
      Empty

# The block spans of a box along the prefix axes lie inside the pieces.
def blocks_within(box: List<&2,Tasks.Span>,extents: List<&2,U32>,+prefix: U32,+units: U32,+at: U32,+position: Nat,remaining: Nat,+per: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +prefix_distance: {Nat.add(position,1n+remaining) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+along: SpansAlong(box,extents,remaining,cap),
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(units,1))),U32.to_nat(cap)) == True{} : Bool},
  +fits: {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent_nat(extents,remaining)),U32.to_nat(U32.max(units,1))),U32.to_nat(per)) == True{} : Bool}) ->
  SpansIn(Tasks.block_spans(box,prefix,units,at),Tasks.piece_extents(extents,prefix,per,at)):
  match box extents remaining:
    case Con{+head,+tail} Con{+extent,+rest} 0n:
      Held.Both{+inside,+extent_room} = along
      +kept = last_lt(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +last_axis = last_eq(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),Con{unit_span(head,units),Nil{}},last_blocks(head,tail,prefix,units,at,kept,last_axis)) :
        SpansIn(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at))
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{per,Nil{}},last_extents(extent,rest,prefix,per,at,kept)) :
        SpansIn(Con{unit_span(head,units),Nil{}},_)
      spans_in_cons(unit_span(head,units),Nil{},per,Nil{},unit_in(head,extent,units,per,cap,cap_ok,inside,extent_room,unit_room,fits),Unit{})
    case Con{+head,+tail} Con{+extent,+rest} 1n+ +later:
      Held.Both{+inside,+others} = along
      +kept = kept_lt(at,prefix,position,later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +next = kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,prefix_distance,prefix_room)
      %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{head,tail},prefix,units,at),Con{head,Tasks.block_spans(tail,prefix,units,(at + 1 : U32))},
          kept_blocks(head,tail,prefix,units,at,kept)) :
        SpansIn(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at))
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))},
          kept_extents(extent,rest,prefix,per,at,kept)) :
        SpansIn(Con{head,Tasks.block_spans(tail,prefix,units,(at + 1 : U32))},_)
      spans_in_cons(head,Tasks.block_spans(tail,prefix,units,(at + 1 : U32)),extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32)),inside,
        blocks_within(tail,rest,prefix,units,(at + 1 : U32),1n+position,later,per,cap,cap_ok,next,steps_next(position,1n+later,U32.to_nat(prefix),prefix_distance),
          prefix_room,others,unit_room,fits))
    case Nil{} +extents +remaining:
      Empty.absurd(SpansIn(Tasks.block_spans(Nil{},prefix,units,at),Tasks.piece_extents(extents,prefix,per,at)),along)
    case Con{+head,+tail} Nil{} +remaining:
      Empty.absurd(SpansIn(Tasks.block_spans(Con{head,tail},prefix,units,at),Tasks.piece_extents(Nil{},prefix,per,at)),along)

# The box a component reads lies inside the source axes up to the last prefix
# axis.
def boxed_along(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,coordinates: List<&2,U32>,+at: U32,
  remaining: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+reach: Reach(extents,coordinates,remaining),
  +rooms: ReadingsRoom(row,cap),+ends: RowEnds(row,classes),+axes: AxesOk(row,grid,extents,at,cap)) ->
  SpansAlong(Tasks.boxed(row,classes,grid,extents,coordinates,at),extents,remaining,cap):
  match extents coordinates remaining:
    case Con{+extent,+rest} Con{+coordinate,+more} 0n:
      Held.Both{+here,+later} = axes
      Held.Both{+reads,+axis_room} = here
      +block = Grid.block(grid,at,extent)
      Held.Both{span_in(Tasks.read_footprint(row,classes,at),extent,block,coordinate,cap,cap_ok,found_ok(row,classes,grid,at,extent,cap,cap_ok,rooms,ends,reads),axis_room),
        Order.transitive(U32.to_nat(extent),Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap),Order.le_add(U32.to_nat(extent),U32.to_nat(block)),axis_room)}
    case Con{+extent,+rest} Con{+coordinate,+more} 1n+ +later:
      Held.Both{+here,+others} = axes
      Held.Both{+reads,+axis_room} = here
      Held.Both{span_in(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate,cap,cap_ok,found_ok(row,classes,grid,at,extent,cap,cap_ok,rooms,ends,reads),
          axis_room),
        boxed_along(row,classes,grid,rest,more,(at + 1 : U32),later,cap,cap_ok,reach,rooms,ends,others)}
    case Nil{} +coordinates +remaining:
      Empty.absurd(SpansAlong(Tasks.boxed(row,classes,grid,Nil{},coordinates,at),Nil{},remaining,cap),reach)
    case Con{+extent,+rest} Nil{} +remaining:
      Empty.absurd(SpansAlong(Tasks.boxed(row,classes,grid,Con{extent,rest},Nil{},at),Con{extent,rest},remaining,cap),reach)

# ---- The pieces a box lists ----

# The pieces listed from a position on rise strictly below the pieces of the
# extents.
def listed_rising(list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+low: Nat,+within: SpansIn(spans,extents),
  +listed: ListedFrom(list,spans,extents,from),+inside: {Nat.is_le(Nat.add(from,List.length(&2,U32,list)),span_total(spans)) == True{} : Bool},
  +above: {Nat.is_le(low,row_value(spans,extents,from,0n)) == True{} : Bool}) -> Rising(list,low,ShapeLaws.elements(extents)):
  match list:
    case Nil{}:
      Unit{}
    case Con{+piece,Nil{}}:
      Held.Both{+value,+later} = listed
      +back = Equal.sym(Nat,U32.to_nat(piece),row_value(spans,extents,from,0n),value)
      +here = Order.lt_le(from,Nat.add(from,1n),span_total(spans),below_plus(from,1n,{==}),inside)
      Held.Both{{%back : {Nat.is_le(low,_) == True{} : Bool} above : {Nat.is_le(low,U32.to_nat(piece)) == True{} : Bool}},
        Held.Both{{%back : {Nat.is_lt(_,ShapeLaws.elements(extents)) == True{} : Bool} row_below(spans,extents,from,within,here) :
          {Nat.is_lt(U32.to_nat(piece),ShapeLaws.elements(extents)) == True{} : Bool}},Unit{}}}
    case Con{+piece,Con{+next,+more}}:
      Held.Both{+value,+later} = listed
      +back = Equal.sym(Nat,U32.to_nat(piece),row_value(spans,extents,from,0n),value)
      +length = List.length(&2,U32,more)
      +total = span_total(spans)
      +here = Order.lt_le(from,Nat.add(from,1n+(1n+length)),total,below_plus(from,1n+(1n+length),{==}),inside)
      +later_inside = {%Addition.successor_right(from,1n+length) : {Nat.is_le(_,total) == True{} : Bool}
        inside : {Nat.is_le(Nat.add(1n+from,1n+length),total) == True{} : Bool}}
      +next_here = Order.lt_le(1n+from,Nat.add(1n+from,1n+length),total,below_plus(1n+from,1n+length,{==}),later_inside)
      +step = row_lt(spans,extents,from,1n+from,0n,within,next_here,Order.succ_le_lt(from,1n+from,Order.reflexive(1n+from)))
      Held.Both{{%back : {Nat.is_le(low,_) == True{} : Bool} above : {Nat.is_le(low,U32.to_nat(piece)) == True{} : Bool}},
        Held.Both{{%back : {Nat.is_lt(_,ShapeLaws.elements(extents)) == True{} : Bool} row_below(spans,extents,from,within,here) :
            {Nat.is_lt(U32.to_nat(piece),ShapeLaws.elements(extents)) == True{} : Bool}},
          listed_rising(Con{next,more},spans,extents,1n+from,1n+U32.to_nat(piece),within,later,later_inside,
            {%back : {Nat.is_le(1n+_,row_value(spans,extents,1n+from,0n)) == True{} : Bool}
              Order.strict_follow(row_value(spans,extents,from,0n),row_value(spans,extents,1n+from,0n),step) :
              {Nat.is_le(1n+U32.to_nat(piece),row_value(spans,extents,1n+from,0n)) == True{} : Bool}})}}

# No piece of a list is value.
def NotIn(+value: U32,list: List<&2,U32>) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+other,rest}:
      Held.Both<{Nat.is_eq(U32.to_nat(value),U32.to_nat(other)) == False{} : Bool},NotIn(value,rest)>

# No piece of one list is a piece of the other.
def ListsApart(list: List<&2,U32>,+others: List<&2,U32>) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+value,rest}:
      Held.Both<NotIn(value,others),ListsApart(rest,others)>

def value_clash(+value: U32,+other: U32,+spans: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+extents: List<&2,U32>,+position: Nat,+other_position: Nat,
  +within: SpansIn(spans,extents),+other_within: SpansIn(others,extents),
  +inside: {Nat.is_lt(position,span_total(spans)) == True{} : Bool},+other_inside: {Nat.is_lt(other_position,span_total(others)) == True{} : Bool},
  +apart: SpansApart(spans,others),+value_exact: {U32.to_nat(value) == row_value(spans,extents,position,0n) : Nat},
  +other_exact: {U32.to_nat(other) == row_value(others,extents,other_position,0n) : Nat},
  +equal: {Nat.is_eq(U32.to_nat(value),U32.to_nat(other)) == True{} : Bool}) -> Empty:
  row_apart(spans,others,extents,position,other_position,0n,within,other_within,inside,other_inside,apart,
    Equal.trans(Nat,row_value(spans,extents,position,0n),U32.to_nat(value),row_value(others,extents,other_position,0n),Equal.sym(Nat,U32.to_nat(value),
      row_value(spans,extents,position,0n),value_exact),
      Equal.trans(Nat,U32.to_nat(value),U32.to_nat(other),row_value(others,extents,other_position,0n),
        Order.equal_of_is_eq(U32.to_nat(value),U32.to_nat(other),equal),other_exact)))

def not_in(+value: U32,list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+others: List<&2,Tasks.Span>,+extents: List<&2,U32>,+position: Nat,+from: Nat,
  +within: SpansIn(spans,extents),+other_within: SpansIn(others,extents),+inside: {Nat.is_lt(position,span_total(spans)) == True{} : Bool},
  +apart: SpansApart(spans,others),+value_exact: {U32.to_nat(value) == row_value(spans,extents,position,0n) : Nat},
  +listed: ListedFrom(list,others,extents,from),+list_inside: {Nat.is_le(Nat.add(from,List.length(&2,U32,list)),span_total(others)) == True{} : Bool}) ->
  NotIn(value,list):
  match list:
    case Nil{}:
      Unit{}
    case Con{+other,+rest}:
      Held.Both{+other_exact,+later} = listed
      +length = List.length(&2,U32,rest)
      +total = span_total(others)
      +here = Order.lt_le(from,Nat.add(from,1n+length),total,below_plus(from,1n+length,{==}),list_inside)
      +later_inside = {%Addition.successor_right(from,length) : {Nat.is_le(_,total) == True{} : Bool}
        list_inside : {Nat.is_le(Nat.add(1n+from,length),total) == True{} : Bool}}
      Held.Both{Held.by_cases({Nat.is_eq(U32.to_nat(value),U32.to_nat(other)) == False{} : Bool},Nat.is_eq(U32.to_nat(value),U32.to_nat(other)),
          equal => Empty.absurd({Nat.is_eq(U32.to_nat(value),U32.to_nat(other)) == False{} : Bool},
            value_clash(value,other,spans,others,extents,position,from,within,other_within,inside,here,apart,value_exact,other_exact,equal)),
          unequal => unequal),
        not_in(value,rest,spans,others,extents,position,1n+from,within,other_within,inside,apart,value_exact,later,later_inside)}

# Two boxes apart list no piece in common.
def lists_apart(list: List<&2,U32>,+others: List<&2,U32>,+spans: List<&2,Tasks.Span>,+other_spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,
  +from: Nat,+other_from: Nat,+within: SpansIn(spans,extents),+other_within: SpansIn(other_spans,extents),+apart: SpansApart(spans,other_spans),
  +listed: ListedFrom(list,spans,extents,from),+inside: {Nat.is_le(Nat.add(from,List.length(&2,U32,list)),span_total(spans)) == True{} : Bool},
  +other_listed: ListedFrom(others,other_spans,extents,other_from),
  +other_inside: {Nat.is_le(Nat.add(other_from,List.length(&2,U32,others)),span_total(other_spans)) == True{} : Bool}) -> ListsApart(list,others):
  match list:
    case Nil{}:
      Unit{}
    case Con{+value,+rest}:
      Held.Both{+value_exact,+later} = listed
      +length = List.length(&2,U32,rest)
      +total = span_total(spans)
      +here = Order.lt_le(from,Nat.add(from,1n+length),total,below_plus(from,1n+length,{==}),inside)
      +later_inside = {%Addition.successor_right(from,length) : {Nat.is_le(_,total) == True{} : Bool}
        inside : {Nat.is_le(Nat.add(1n+from,length),total) == True{} : Bool}}
      Held.Both{not_in(value,others,spans,other_spans,extents,from,other_from,within,other_within,here,apart,value_exact,other_listed,other_inside),
        lists_apart(rest,others,spans,other_spans,extents,1n+from,other_from,within,other_within,apart,later,later_inside,other_listed,other_inside)}

def count_zero(+first: U32,+count: U32,+rest: List<&2,Tasks.Span>,+zero: {U32.is_zero(count) == True{} : Bool}) ->
  {U32.to_nat(Tasks.span_count(Con{Tasks.Span{first,count},rest})) == span_total(Con{Tasks.Span{first,count},rest}) : Nat}:
  %Equal.sym(U32,count,0,Arithmetic.word_zero(count,zero)) :
    {U32.to_nat((_ * Tasks.span_count(rest) : U32)) == Nat.mul(U32.to_nat(_),span_total(rest)) : Nat}
  %Equal.sym(U32,(0 * Tasks.span_count(rest) : U32),0,Arithmetic.multiply_zero_left(Tasks.span_count(rest))) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(0),span_total(rest)) : Nat}
  {==}

def count_some(+first: U32,+count: U32,+rest: List<&2,Tasks.Span>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(count),span_total(rest)),U32.to_nat(cap)) == True{} : Bool},+some: {U32.is_zero(count) == False{} : Bool},
  recurse: {Nat.is_le(span_total(rest),U32.to_nat(cap)) == True{} : Bool} -> {U32.to_nat(Tasks.span_count(rest)) == span_total(rest) : Nat}) ->
  {U32.to_nat(Tasks.span_count(Con{Tasks.Span{first,count},rest})) == span_total(Con{Tasks.Span{first,count},rest}) : Nat}:
  Arithmetic.product_exact(cap,cap_ok,count,Tasks.span_count(rest),U32.to_nat(count),span_total(rest),{==},
    recurse(Order.transitive(span_total(rest),Nat.mul(U32.to_nat(count),span_total(rest)),U32.to_nat(cap),
      factor_covers(span_total(rest),U32.to_nat(count),Arithmetic.word_nonzero_positive(count,some)),room)),room)

# The positions of a box count as Tasks.span_count counts them.
def span_count_value(spans: List<&2,Tasks.Span>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(span_total(spans),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(Tasks.span_count(spans)) == span_total(spans) : Nat}:
  match spans:
    case Nil{}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest}:
      Held.by_cases({U32.to_nat(Tasks.span_count(Con{Tasks.Span{first,count},rest})) == span_total(Con{Tasks.Span{first,count},rest}) : Nat},U32.is_zero(count),
        zero => count_zero(first,count,rest,zero),
        some => count_some(first,count,rest,cap,cap_ok,room,some,rest_room => span_count_value(rest,cap,cap_ok,rest_room)))

# Tasks.pieces_listed lists remaining positions after done.
def pieces_length(remaining: Nat,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+index: U32,+done: List<&2,U32>) ->
  {List.length(&2,U32,Tasks.pieces_listed(remaining,spans,extents,index,done)) == Nat.add(remaining,List.length(&2,U32,done)) : Nat}:
  match remaining:
    case 0n:
      {==}
    case 1n+ +rest:
      +previous = (index - 1 : U32)
      +piece = Tasks.piece_at(spans,extents,previous,0)
      Equal.trans(Nat,List.length(&2,U32,Tasks.pieces_listed(rest,spans,extents,previous,Con{piece,done})),Nat.add(rest,1n+List.length(&2,U32,done)),
        1n+Nat.add(rest,List.length(&2,U32,done)),pieces_length(rest,spans,extents,previous,Con{piece,done}),
        Addition.successor_right(rest,List.length(&2,U32,done)))

# The positions of a box in order: each position numbered as the row-major
# model numbers it, one for each position.
def BoxListed(list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>) -> Data:
  Held.Both<ListedFrom(list,spans,extents,0n),{Nat.is_le(Nat.add(0n,List.length(&2,U32,list)),span_total(spans)) == True{} : Bool}>

def box_listed(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within: SpansIn(spans,extents),
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +double_room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  BoxListed(Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{}),spans,extents):
  +total = Tasks.span_count(spans)
  +count = U32.to_nat(total)
  +total_le = span_total_le(spans,extents,within)
  +exact = span_count_value(spans,cap,cap_ok,Order.transitive(span_total(spans),ShapeLaws.elements(extents),U32.to_nat(cap),total_le,room))
  +inside = {%exact : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count) :
    {Nat.is_le(count,span_total(spans)) == True{} : Bool}}
  +list = Tasks.pieces_listed(count,spans,extents,total,Nil{})
  Held.Both{{%Order.subtract_below(count,count,Order.reflexive(count)) : ListedFrom(list,spans,extents,_)
      listed_from(count,spans,extents,total,Nil{},cap,cap_ok,within,Unit{},Order.reflexive(count),inside,room,double_room) :
      ListedFrom(list,spans,extents,0n)},
    {%Equal.sym(Nat,List.length(&2,U32,list),Nat.add(count,0n),pieces_length(count,spans,extents,total,Nil{})) : {Nat.is_le(Nat.add(0n,_),span_total(spans)) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(count,0n),count,Addition.zero_right(count)) : {Nat.is_le(_,span_total(spans)) == True{} : Bool}
      inside : {Nat.is_le(Nat.add(0n,List.length(&2,U32,list)),span_total(spans)) == True{} : Bool}}}

# ---- The prefix axes of a source grid ----

def units_prefix(+found: Grid.Units) -> U32:
  Grid.Units{+prefix,outer,extent,unit_cells} = found
  prefix

def units_outer(+found: Grid.Units) -> U32:
  Grid.Units{prefix,+outer,extent,unit_cells} = found
  outer

def units_extent(+found: Grid.Units) -> U32:
  Grid.Units{prefix,outer,+extent,unit_cells} = found
  extent

def self_not_below(+value: U32) -> {U32.is_lt(value,value) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(value,value),Nat.is_lt(U32.to_nat(value),U32.to_nat(value)),False{},Comparison.lt(value,value),
    Order.not_below(U32.to_nat(value),U32.to_nat(value),Order.reflexive(U32.to_nat(value))))

# The pieces of the last prefix axis: one entry past the prefix.
def last_piece_extents(+extent: U32,+rest: List<&2,U32>,+per: U32,+axis: U32) ->
  {ShapeLaws.elements(Tasks.piece_extents(Con{extent,rest},(axis + 1 : U32),per,axis)) == Nat.mul(U32.to_nat(per),1n) : Nat}:
  %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),(axis + 1 : U32)),False{},self_not_below((axis + 1 : U32))) :
    {ShapeLaws.elements(Bool.pick(List<&2,U32>,_,Con{extent,Tasks.piece_extents(rest,(axis + 1 : U32),per,(axis + 1 : U32))},Con{per,Nil{}})) ==
      Nat.mul(U32.to_nat(per),1n) : Nat}
  {==}

def last_extent_at(+extent: U32,+rest: List<&2,U32>,+axis: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(axis) == position : Nat},+room: {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool}) ->
  {extent_nat(Con{extent,rest},Nat.sub(U32.to_nat((axis + 1 : U32)),1n+position)) == extent : U32}:
  %Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next_word(axis,position,cap,cap_ok,exact,room)) :
    {extent_nat(Con{extent,rest},Nat.sub(_,1n+position)) == extent : U32}
  %Equal.sym(Nat,Nat.sub(1n+position,1n+position),0n,Order.subtract_below(1n+position,1n+position,Order.reflexive(1n+position))) :
    {extent_nat(Con{extent,rest},_) == extent : U32}
  {==}

def sub_step(+total: Nat,+value: Nat,+below: {Nat.is_lt(value,total) == True{} : Bool}) -> {Nat.sub(total,value) == 1n+Nat.sub(total,1n+value) : Nat}:
  match total value:
    case 0n +value:
      Empty.absurd({Nat.sub(0n,value) == 1n+Nat.sub(0n,1n+value) : Nat},Order.below_zero(value,below))
    case 1n+ +rest 0n:
      Equal.sym(Nat,1n+Nat.sub(rest,0n),1n+rest,Equal.cong(Nat,Nat,value => 1n+value,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)))
    case 1n+ +rest 1n+ +smaller:
      sub_step(rest,smaller,below)

# The room the products of the prefix axes need.
def AxesRoom(+outer: U32,+extent: U32,+rest: List<&2,U32>,+position: Nat,+cap: U32) -> Data:
  Held.Both<{Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool},
    Held.Both<{Nat.is_lt(0n,ShapeLaws.elements(rest)) == True{} : Bool},{Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}>>

def room_axes(+outer: U32,+extent: U32,+rest: List<&2,U32>,+position: Nat,+cap: U32,+room: AxesRoom(outer,extent,rest,position,cap)) ->
  {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}:
  Held.Both{+products,+more} = room
  Held.Both{+positive,+axes} = more
  axes

def room_one(+position: Nat,+rest: List<&2,U32>,+cap: U32,+room: {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(1n+position,Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap),Order.le_add(1n+position,List.length(&2,U32,rest)),room)

def room_next(+position: Nat,+other: U32,+rest: List<&2,U32>,+cap: U32,
  +room: {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,Con{other,rest})),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+(1n+position),List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}:
  %Addition.successor_right(position,List.length(&2,U32,rest)) : {Nat.is_le(1n+_,U32.to_nat(cap)) == True{} : Bool}
  room

def step_kept(+axis: U32,+prefix: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},
  +room: {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool},+reached: {Nat.is_le(1n+(1n+position),U32.to_nat(prefix)) == True{} : Bool}) ->
  {U32.is_lt((axis + 1 : U32),prefix) == True{} : Bool}:
  +next = next_word(axis,position,cap,cap_ok,exact,room)
  Equal.trans(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),True{},Comparison.lt((axis + 1 : U32),prefix),
    {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == True{} : Bool}
      Order.succ_le_lt(1n+position,U32.to_nat(prefix),reached) : {Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == True{} : Bool}})

def step_exact(+outer: U32,+extent: U32,+other: U32,+rest: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +products: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{other,rest})),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Con{other,rest})) == True{} : Bool}) -> {U32.to_nat((outer * extent : U32)) == Nat.mul(U32.to_nat(outer),U32.to_nat(extent)) : Nat}:
  Arithmetic.product_exact(cap,cap_ok,outer,extent,U32.to_nat(outer),U32.to_nat(extent),{==},{==},
    Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{other,rest})),U32.to_nat(cap),
      RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{other,rest}),positive),products))

def step_room(+outer: U32,+extent: U32,+other: U32,+rest: List<&2,U32>,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: AxesRoom(outer,extent,Con{other,rest},position,cap)) -> {U32.to_nat((outer * extent : U32)) == Nat.mul(U32.to_nat(outer),U32.to_nat(extent)) : Nat}:
  Held.Both{+products,+more} = room
  Held.Both{+positive,+axes} = more
  step_exact(outer,extent,other,rest,cap,cap_ok,products,positive)

def next_room(+outer: U32,+extent: U32,+other: U32,+rest: List<&2,U32>,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: AxesRoom(outer,extent,Con{other,rest},position,cap)) -> AxesRoom((outer * extent : U32),other,rest,1n+position,cap):
  Held.Both{+products,+more} = room
  Held.Both{+positive,+axes} = more
  +exact = step_exact(outer,extent,other,rest,cap,cap_ok,products,positive)
  +scaled = Nat.mul(U32.to_nat(outer),U32.to_nat(extent))
  Held.Both{{%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),scaled,exact) : {Nat.is_le(Nat.mul(Nat.mul(_,U32.to_nat(other)),ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,Nat.mul(Nat.mul(scaled,U32.to_nat(other)),ShapeLaws.elements(rest)),Nat.mul(scaled,Nat.mul(U32.to_nat(other),ShapeLaws.elements(rest))),
          RowMajor.assoc(scaled,U32.to_nat(other),ShapeLaws.elements(rest))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
      products : {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(other)),ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool}},
    Held.Both{DivisionOrder.right_factor_positive(U32.to_nat(other),ShapeLaws.elements(rest),positive),room_next(position,other,rest,cap,axes)}}

def prefix_here(+axis: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},
  +room: {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(1n+position,U32.to_nat((axis + 1 : U32))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next_word(axis,position,cap,cap_ok,exact,room)) : {Nat.is_le(1n+position,_) == True{} : Bool}
  Order.reflexive(1n+position)

def prefix_there(+axis: U32,+position: Nat,+rest: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},
  +room: {Nat.is_le(1n+position,U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((axis + 1 : U32)),Nat.add(1n+position,List.length(&2,U32,rest))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next_word(axis,position,cap,cap_ok,exact,room)) :
    {Nat.is_le(_,Nat.add(1n+position,List.length(&2,U32,rest))) == True{} : Bool}
  Order.le_add(1n+position,List.length(&2,U32,rest))

# The grid prefix lies past every axis taken.
def prefix_reached(rest: List<&2,U32>,fits: Bool,+axis: U32,+outer: U32,+extent: U32,+target: U32,+position: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},
  +room: {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(1n+position,U32.to_nat(units_prefix(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target)))) == True{} : Bool}:
  match rest fits:
    case +others True{}:
      prefix_here(axis,position,cap,cap_ok,exact,room_one(position,others,cap,room))
    case Nil{} False{}:
      prefix_here(axis,position,cap,cap_ok,exact,room_one(position,Nil{},cap,room))
    case Con{+last,Nil{}} False{}:
      prefix_here(axis,position,cap,cap_ok,exact,room_one(position,Con{last,Nil{}},cap,room))
    case Con{+other,Con{+next_axis,+tail}} False{}:
      +later = {Con{next_axis,tail} : List<&2,U32>}
      +next = next_word(axis,position,cap,cap_ok,exact,room_one(position,Con{other,later},cap,room))
      Order.transitive(1n+position,1n+(1n+position),U32.to_nat(units_prefix(GridLaws.units_model(later,U32.is_le(Grid.product(later),target),
          ((axis + 1 : U32) + 1 : U32),(outer * extent : U32),other,target))),Order.successor(1n+position,1n+position,Order.reflexive(1n+position)),
        prefix_reached(later,U32.is_le(Grid.product(later),target),(axis + 1 : U32),(outer * extent : U32),other,target,1n+position,cap,cap_ok,next,
          room_next(position,other,later,cap,room)))

# The grid prefix lies within the axes.
def prefix_within(rest: List<&2,U32>,fits: Bool,+axis: U32,+outer: U32,+extent: U32,+target: U32,+position: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},
  +room: {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(units_prefix(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target))),Nat.add(1n+position,List.length(&2,U32,rest))) == True{} : Bool}:
  match rest fits:
    case +others True{}:
      prefix_there(axis,position,others,cap,cap_ok,exact,room_one(position,others,cap,room))
    case Nil{} False{}:
      prefix_there(axis,position,Nil{},cap,cap_ok,exact,room_one(position,Nil{},cap,room))
    case Con{+last,Nil{}} False{}:
      prefix_there(axis,position,Con{last,Nil{}},cap,cap_ok,exact,room_one(position,Con{last,Nil{}},cap,room))
    case Con{+other,Con{+next_axis,+tail}} False{}:
      +later = {Con{next_axis,tail} : List<&2,U32>}
      +next = next_word(axis,position,cap,cap_ok,exact,room_one(position,Con{other,later},cap,room))
      %Equal.sym(Nat,Nat.add(position,1n+List.length(&2,U32,later)),1n+Nat.add(position,List.length(&2,U32,later)),Addition.successor_right(position,List.length(&2,U32,later))) :
        {Nat.is_le(U32.to_nat(units_prefix(GridLaws.units_model(later,U32.is_le(Grid.product(later),target),((axis + 1 : U32) + 1 : U32),(outer * extent : U32),other,target))),
          1n+_) == True{} : Bool}
      prefix_within(later,U32.is_le(Grid.product(later),target),(axis + 1 : U32),(outer * extent : U32),other,target,1n+position,cap,cap_ok,next,
        room_next(position,other,later,cap,room))

def total_last(+outer: U32,+extent: U32,+others: List<&2,U32>,+per: U32,+axis: U32) ->
  {Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Tasks.piece_extents(Con{extent,others},(axis + 1 : U32),per,axis))) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}:
  %Equal.sym(Nat,ShapeLaws.elements(Tasks.piece_extents(Con{extent,others},(axis + 1 : U32),per,axis)),Nat.mul(U32.to_nat(per),1n),last_piece_extents(extent,others,per,axis)) :
    {Nat.mul(U32.to_nat(outer),_) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}
  %Equal.sym(Nat,Nat.mul(U32.to_nat(per),1n),U32.to_nat(per),Order.times_one(U32.to_nat(per))) : {Nat.mul(U32.to_nat(outer),_) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}
  {==}

def total_step(+outer: U32,+extent: U32,+other: U32,+later: List<&2,U32>,+per: U32,+axis: U32,+prefix: U32,+found: Nat,
  +kept: {U32.is_lt((axis + 1 : U32),prefix) == True{} : Bool},
  +exact: {U32.to_nat((outer * extent : U32)) == Nat.mul(U32.to_nat(outer),U32.to_nat(extent)) : Nat},
  +inner: {Nat.mul(U32.to_nat((outer * extent : U32)),ShapeLaws.elements(Tasks.piece_extents(Con{other,later},prefix,per,(axis + 1 : U32)))) == found : Nat}) ->
  {Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Tasks.piece_extents(Con{extent,Con{other,later}},prefix,per,axis))) == found : Nat}:
  +rest = Tasks.piece_extents(Con{other,later},prefix,per,(axis + 1 : U32))
  %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),True{},kept) :
    {Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Bool.pick(List<&2,U32>,_,Con{extent,rest},Con{per,Nil{}}))) == found : Nat}
  %RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),ShapeLaws.elements(rest)) : {_ == found : Nat}
  %exact : {Nat.mul(_,ShapeLaws.elements(rest)) == found : Nat}
  inner

# The pieces of the prefix axes number the outer indices times the pieces of
# the last.
def units_total(rest: List<&2,U32>,fits: Bool,+axis: U32,+outer: U32,+extent: U32,+target: U32,+per: U32,+position: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},+room: AxesRoom(outer,extent,rest,position,cap)) ->
  {Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Tasks.piece_extents(Con{extent,rest},units_prefix(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target)),per,axis))) ==
    Nat.mul(U32.to_nat(units_outer(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target))),U32.to_nat(per)) : Nat}:
  match rest fits:
    case +others True{}:
      total_last(outer,extent,others,per,axis)
    case Nil{} False{}:
      total_last(outer,extent,Nil{},per,axis)
    case Con{+last,Nil{}} False{}:
      total_last(outer,extent,Con{last,Nil{}},per,axis)
    case Con{+other,Con{+next_axis,+tail}} False{}:
      +axes = room_axes(outer,extent,Con{other,Con{next_axis,tail}},position,cap,room)
      +later = {Con{next_axis,tail} : List<&2,U32>}
      +next_fits = U32.is_le(Grid.product(later),target)
      +found = GridLaws.units_model(later,next_fits,((axis + 1 : U32) + 1 : U32),(outer * extent : U32),other,target)
      +next = next_word(axis,position,cap,cap_ok,exact,room_one(position,Con{other,later},cap,axes))
      +reached = prefix_reached(later,next_fits,(axis + 1 : U32),(outer * extent : U32),other,target,1n+position,cap,cap_ok,next,room_next(position,other,later,cap,axes))
      total_step(outer,extent,other,later,per,axis,units_prefix(found),Nat.mul(U32.to_nat(units_outer(found)),U32.to_nat(per)),
        step_kept(axis,units_prefix(found),position,cap,cap_ok,exact,room_one(position,Con{other,later},cap,axes),reached),
        step_room(outer,extent,other,later,position,cap,cap_ok,room),
        units_total(later,next_fits,(axis + 1 : U32),(outer * extent : U32),other,target,per,1n+position,cap,cap_ok,next,
          next_room(outer,extent,other,later,position,cap,cap_ok,room)))

# The last prefix axis is the extent of the grid.
def units_extent_at(rest: List<&2,U32>,fits: Bool,+axis: U32,+outer: U32,+extent: U32,+target: U32,+position: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(axis) == position : Nat},+room: AxesRoom(outer,extent,rest,position,cap)) ->
  {extent_nat(Con{extent,rest},Nat.sub(U32.to_nat(units_prefix(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target))),1n+position)) ==
    units_extent(GridLaws.units_model(rest,fits,(axis + 1 : U32),outer,extent,target)) : U32}:
  match rest fits:
    case +others True{}:
      +axes = room_axes(outer,extent,others,position,cap,room)
      last_extent_at(extent,others,axis,position,cap,cap_ok,exact,room_one(position,others,cap,axes))
    case Nil{} False{}:
      +axes = room_axes(outer,extent,Nil{},position,cap,room)
      last_extent_at(extent,Nil{},axis,position,cap,cap_ok,exact,room_one(position,Nil{},cap,axes))
    case Con{+last,Nil{}} False{}:
      +axes = room_axes(outer,extent,Con{last,Nil{}},position,cap,room)
      last_extent_at(extent,Con{last,Nil{}},axis,position,cap,cap_ok,exact,room_one(position,Con{last,Nil{}},cap,axes))
    case Con{+other,Con{+next_axis,+tail}} False{}:
      +axes = room_axes(outer,extent,Con{other,Con{next_axis,tail}},position,cap,room)
      +later = {Con{next_axis,tail} : List<&2,U32>}
      +next_fits = U32.is_le(Grid.product(later),target)
      +found = GridLaws.units_model(later,next_fits,((axis + 1 : U32) + 1 : U32),(outer * extent : U32),other,target)
      +next = next_word(axis,position,cap,cap_ok,exact,room_one(position,Con{other,later},cap,axes))
      +reached = prefix_reached(later,next_fits,(axis + 1 : U32),(outer * extent : U32),other,target,1n+position,cap,cap_ok,next,room_next(position,other,later,cap,axes))
      %Equal.sym(Nat,Nat.sub(U32.to_nat(units_prefix(found)),1n+position),1n+Nat.sub(U32.to_nat(units_prefix(found)),1n+(1n+position)),
          sub_step(U32.to_nat(units_prefix(found)),1n+position,Order.succ_le_lt(1n+position,U32.to_nat(units_prefix(found)),reached))) :
        {extent_nat(Con{extent,Con{other,later}},_) == units_extent(found) : U32}
      units_extent_at(later,next_fits,(axis + 1 : U32),(outer * extent : U32),other,target,1n+position,cap,cap_ok,next,
        next_room(outer,extent,other,later,position,cap,cap_ok,room))

# ---- Taking the pieces of a source ----

# Two pieces of a grid apart start apart.
def starts_clash(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first: U32,+second: U32,
  +first_in: {Nat.is_le(U32.to_nat(first),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +second_in: {Nat.is_le(U32.to_nat(second),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool},
  +equal: {Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},first) ==
    Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},second) : U32}) -> Empty:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +firsts = GridRanges.piece_at_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first,first_in)
  +seconds = GridRanges.piece_at_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,second,second_in)
  +words = Equal.trans(U32,first,Grid.piece_of(grid,Grid.start(grid,first)),second,Equal.sym(U32,Grid.piece_of(grid,Grid.start(grid,first)),first,firsts),
    Equal.trans(U32,Grid.piece_of(grid,Grid.start(grid,first)),Grid.piece_of(grid,Grid.start(grid,second)),second,
      Equal.cong(U32,U32,cell => Grid.piece_of(grid,cell),Grid.start(grid,first),Grid.start(grid,second),equal),seconds))
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(U32.to_nat(first),U32.to_nat(second)),True{},
    Equal.sym(Bool,Nat.is_eq(U32.to_nat(first),U32.to_nat(second)),False{},differ),
    {%words : {Nat.is_eq(U32.to_nat(first),U32.to_nat(_)) == True{} : Bool} Order.equal_reflexive(U32.to_nat(first)) :
      {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == True{} : Bool}}))

def starts_apart(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first: U32,+second: U32,
  +first_in: {Nat.is_lt(U32.to_nat(first),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +second_in: {Nat.is_lt(U32.to_nat(second),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool}) ->
  {U32.is_eq(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},first),
    Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},second)) == False{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  Held.by_cases({U32.is_eq(Grid.start(grid,first),Grid.start(grid,second)) == False{} : Bool},U32.is_eq(Grid.start(grid,first),Grid.start(grid,second)),
    same => Empty.absurd({U32.is_eq(Grid.start(grid,first),Grid.start(grid,second)) == False{} : Bool},
      starts_clash(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first,second,
        lt_to_le(U32.to_nat(first),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),first_in),lt_to_le(U32.to_nat(second),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),second_in),
        differ,Arithmetic.equal_of_is_eq(Grid.start(grid,first),Grid.start(grid,second),same))),
    other => other)

# The ranges from piece start on: the steps-th of them is a leaf with the cells
# of its piece.
def spans_lookup(+layout: Pieces.Ranges,+grid: Grid.Grid,total: Nat,steps: Nat,+start: Nat,+rest: Held.Spans,
  +at: Placed.SpansAt(layout,GridRanges.piece_spans(grid,total,start,rest)),+inside: {Nat.is_lt(steps,total) == True{} : Bool}) ->
  Held.Both<Placed.LeafAt(layout,Grid.start(grid,U32.from_nat(Nat.add(start,steps)))),
    {Placed.rows_of(layout,Grid.start(grid,U32.from_nat(Nat.add(start,steps)))) == Pieces.cells_of(grid,U32.from_nat(Nat.add(start,steps)),1) : U32}>:
  match total steps:
    case 0n +steps:
      Empty.absurd(Held.Both<Placed.LeafAt(layout,Grid.start(grid,U32.from_nat(Nat.add(start,steps)))),
        {Placed.rows_of(layout,Grid.start(grid,U32.from_nat(Nat.add(start,steps)))) == Pieces.cells_of(grid,U32.from_nat(Nat.add(start,steps)),1) : U32}>,
        Order.below_zero(steps,inside))
    case 1n+ +more 0n:
      Held.Both{+leaf,+others} = at
      Held.Both{+rows,+later} = others
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        Held.Both<Placed.LeafAt(layout,Grid.start(grid,U32.from_nat(_))),{Placed.rows_of(layout,Grid.start(grid,U32.from_nat(_))) == Pieces.cells_of(grid,U32.from_nat(_),1) : U32}>
      Held.Both{leaf,rows}
    case 1n+ +more 1n+ +step:
      Held.Both{+leaf,+others} = at
      Held.Both{+rows,+later} = others
      %Equal.sym(Nat,Nat.add(start,1n+step),1n+Nat.add(start,step),Addition.successor_right(start,step)) :
        Held.Both<Placed.LeafAt(layout,Grid.start(grid,U32.from_nat(_))),{Placed.rows_of(layout,Grid.start(grid,U32.from_nat(_))) == Pieces.cells_of(grid,U32.from_nat(_),1) : U32}>
      spans_lookup(layout,grid,more,step,1n+start,rest,later,inside)

# A piece of a grid laid out by its ranges starts a leaf with its cells.
def piece_leaf(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Held.Both<Placed.LeafAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
      Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),
    {Placed.rows_of(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
      Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) ==
      Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece) : U32}>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +all = Placed.leaves_spans(layout,layout,Held.Ended{},Placed.own_leaves(layout,ordered),Unit{})
  +listed = {%GridRanges.grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape) : Placed.SpansAt(layout,_)
    all : Placed.SpansAt(layout,GridRanges.piece_spans(grid,total,0n,Held.Ended{}))}
  +room = Order.transitive(U32.to_nat(piece),total,U32.to_nat(Machine.capacity(limits)),lt_to_le(U32.to_nat(piece),total,inside),
    GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +word = Arithmetic.to_nat_injective(U32.from_nat(U32.to_nat(piece)),piece,
    Bounds.roundtrip(U32.to_nat(piece),Machine.capacity(limits),LimitBounds.capacity_fits(limits),room))
  %Equal.sym(U32,Grid.span(grid,piece),Pieces.cells_of(grid,piece,1),GridRanges.span_cells(outer,extent,unit_cells,block,per,count,prefix,piece)) :
    Held.Both<Placed.LeafAt(layout,Grid.start(grid,piece)),{Placed.rows_of(layout,Grid.start(grid,piece)) == _ : U32}>
  %word : Held.Both<Placed.LeafAt(layout,Grid.start(grid,_)),{Placed.rows_of(layout,Grid.start(grid,_)) == Pieces.cells_of(grid,_,1) : U32}>
  spans_lookup(layout,grid,total,U32.to_nat(piece),0n,Held.Ended{},listed,inside)

# Every listed piece starts none of the spans gone.
def FreshAll(+gone: Held.Spans,+grid: Grid.Grid,list: List<&2,U32>) -> Data:
  match list:
    case Nil{}:
      Unit
    case Con{+piece,rest}:
      Held.Both<Placed.Fresh(gone,Grid.start(grid,piece)),FreshAll(gone,grid,rest)>

def fresh_ended(+grid: Grid.Grid,list: List<&2,U32>) -> FreshAll(Held.Ended{},grid,list):
  match list:
    case Nil{}:
      Unit{}
    case Con{+piece,+rest}:
      Held.Both{Unit{},fresh_ended(grid,rest)}

# The pieces after a piece start apart from it.
def fresh_after(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+head: U32,+gone: Held.Spans,
  list: List<&2,U32>,+rising: Rising(list,1n+U32.to_nat(head),Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
  +head_in: {Nat.is_lt(U32.to_nat(head),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +fresh: FreshAll(gone,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list)) ->
  FreshAll(Held.Spanned{Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},head),
    Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},head),gone},
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list):
  match list:
    case Nil{}:
      Unit{}
    case Con{+piece,+rest}:
      Held.Both{+below,+more} = rising
      Held.Both{+inside,+later} = more
      Held.Both{+here,+others} = fresh
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +differ = Order.less_not_equal(U32.to_nat(head),U32.to_nat(piece),Order.succ_le_lt(U32.to_nat(head),U32.to_nat(piece),below))
      Held.Both{Held.Both{starts_apart(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,head,piece,head_in,inside,differ),here},
        fresh_after(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,head,gone,rest,
          rising_lower(rest,1n+U32.to_nat(piece),1n+U32.to_nat(head),total,Order.successor(1n+U32.to_nat(head),U32.to_nat(piece),below),later),head_in,others)}

def take_one(-Later: Data,+layout: Pieces.Ranges,+cell: U32,+rows: U32,+gone: Held.Spans,
  +pair: Held.Both<Placed.LeafAt(layout,cell),{Placed.rows_of(layout,cell) == rows : U32}>,+here: Placed.Fresh(gone,cell),+later: Later) ->
  Held.Both<Placed.LeafAt(layout,cell),Held.Both<{Placed.rows_of(layout,cell) == rows : U32},Held.Both<Placed.Fresh(gone,cell),Later>>>:
  Held.Both{+leaf,+same} = pair
  Held.Both{leaf,Held.Both{same,Held.Both{here,later}}}

# Rising pieces, none starting a span gone, are taken as Moved.TakeFacts
# needs.
def take_facts(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  list: List<&2,U32>,+gone: Held.Spans,+low: Nat,+rising: Rising(list,low,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
  +fresh: FreshAll(gone,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list)) ->
  Moved.TakeFacts(list,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),gone):
  match list:
    case Nil{}:
      Unit{}
    case Con{+piece,+rest}:
      Held.Both{+at,+more} = rising
      Held.Both{+inside,+later} = more
      Held.Both{+here,+others} = fresh
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +taken = {Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),gone} : Held.Spans}
      take_one(Moved.TakeFacts(rest,grid,Pieces.ranges_of_grid(grid),Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),gone}),
        Pieces.ranges_of_grid(grid),Grid.start(grid,piece),Grid.span(grid,piece),gone,
        piece_leaf(limits,outer,extent,unit_cells,block,per,count,prefix,shape,ordered,piece,inside),here,
        take_facts(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,rest,taken,1n+U32.to_nat(piece),later,
          fresh_after(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,gone,rest,later,inside,others)))

def head_differ(+value: U32,+other: U32,+rest: List<&2,U32>,+apart: NotIn(value,Con{other,rest})) ->
  {Nat.is_eq(U32.to_nat(other),U32.to_nat(value)) == False{} : Bool}:
  Held.Both{+differ,+later} = apart
  Equal.trans(Bool,Nat.is_eq(U32.to_nat(other),U32.to_nat(value)),Nat.is_eq(U32.to_nat(value),U32.to_nat(other)),False{},
    Order.equal_symmetric(U32.to_nat(other),U32.to_nat(value)),differ)

def not_rest(+value: U32,+other: U32,+rest: List<&2,U32>,+apart: NotIn(value,Con{other,rest})) -> NotIn(value,rest):
  Held.Both{+differ,+later} = apart
  later

def apart_tail(list: List<&2,U32>,+other: U32,+rest: List<&2,U32>,+apart: ListsApart(list,Con{other,rest})) -> ListsApart(list,rest):
  match list:
    case Nil{}:
      Unit{}
    case Con{+value,+others}:
      Held.Both{+here,+later} = apart
      Held.Both{not_rest(value,other,rest,here),apart_tail(others,other,rest,later)}

# A piece no listed piece equals starts apart from them.
def avoid_one(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+other: U32,+rest: List<&2,U32>,
  +other_in: {Nat.is_lt(U32.to_nat(other),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+gone: Held.Spans,
  list: List<&2,U32>,+low: Nat,+rising: Rising(list,low,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),+apart: ListsApart(list,Con{other,rest}),
  +fresh: FreshAll(gone,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list)) ->
  FreshAll(Held.Spanned{Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},other),
    Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},other),gone},
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list):
  match list:
    case Nil{}:
      Unit{}
    case Con{+value,+others}:
      Held.Both{+at,+more} = rising
      Held.Both{+inside,+later} = more
      Held.Both{+here,+apart_rest} = apart
      Held.Both{+fresh_here,+fresh_rest} = fresh
      Held.Both{Held.Both{starts_apart(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,other,value,other_in,inside,head_differ(value,other,rest,here)),
          fresh_here},
        avoid_one(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,other,rest,other_in,gone,others,1n+U32.to_nat(value),later,apart_rest,fresh_rest)}

# Pieces apart from the pieces taken before start none of their spans.
def fresh_taken(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  previous: List<&2,U32>,+gone: Held.Spans,+previous_low: Nat,+previous_rising: Rising(previous,previous_low,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
  +list: List<&2,U32>,+low: Nat,+rising: Rising(list,low,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),+apart: ListsApart(list,previous),
  +fresh: FreshAll(gone,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list)) ->
  FreshAll(Moved.taken_gone(previous,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},gone),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list):
  match previous:
    case Nil{}:
      fresh
    case Con{+other,+rest}:
      Held.Both{+at,+more} = previous_rising
      Held.Both{+inside,+later} = more
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      fresh_taken(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,rest,Held.Spanned{Grid.start(grid,other),Grid.span(grid,other),gone},
        1n+U32.to_nat(other),later,list,low,rising,apart_tail(list,other,rest,apart),
        avoid_one(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,other,rest,inside,gone,list,low,rising,apart,fresh))

# ---- The pieces of a source box ----

def empty_box_list(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +zero: {U32.is_zero(Tasks.span_count(box)) == True{} : Bool}) ->
  {Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box) == Nil{} : List<&2,U32>}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(box)),True{},zero) :
    {Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,axes,Tasks.block_spans(box,prefix,block,0))) == Nil{} : List<&2,U32>}
  {==}

def listed_box_list(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+some: {U32.is_zero(prefix) == False{} : Bool}) ->
  {Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box) ==
    Tasks.pieces_listed(U32.to_nat(Tasks.span_count(Tasks.block_spans(box,prefix,block,0))),Tasks.block_spans(box,prefix,block,0),
      Tasks.piece_extents(axes,prefix,U32.max(per,1),0),Tasks.span_count(Tasks.block_spans(box,prefix,block,0)),Nil{}) : List<&2,U32>}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +spans = Tasks.block_spans(box,prefix,block,0)
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),Tasks.span_count(spans),Nil{})
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(box)),False{},nonzero) :
    {Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,axes,spans)) == listed : List<&2,U32>}
  %Equal.sym(Bool,U32.is_zero(prefix),False{},some) :
    {Bool.pick(List<&2,U32>,False{},Nil{},Bool.pick(List<&2,U32>,_,Con{0,Nil{}},listed)) == listed : List<&2,U32>}
  {==}

def whole_box_list(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  {Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box) == Con{0,Nil{}} : List<&2,U32>}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +spans = Tasks.block_spans(box,prefix,block,0)
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),Tasks.span_count(spans),Nil{})
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(box)),False{},nonzero) :
    {Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,axes,spans)) == Con{0,Nil{}} : List<&2,U32>}
  %Equal.sym(Bool,U32.is_zero(prefix),True{},none) :
    {Bool.pick(List<&2,U32>,False{},Nil{},Bool.pick(List<&2,U32>,_,Con{0,Nil{}},listed)) == Con{0,Nil{}} : List<&2,U32>}
  {==}

def rising_listed(+list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+within: SpansIn(spans,extents),+listed: BoxListed(list,spans,extents)) ->
  Rising(list,0n,ShapeLaws.elements(extents)):
  Held.Both{+from,+inside} = listed
  listed_rising(list,spans,extents,0n,0n,within,from,inside,Order.zero_le(row_value(spans,extents,0n,0n)))

# The pieces of a grid fit half a word, and so do twice them.
def pieces_half(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),
    GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape),
    Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),LimitBounds.capacity_within_half(limits)))

# The pieces a box of spans inside the pieces of a grid lists are listed in
# row-major order.
def pieces_box_listed(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+prefix: U32,+spans: List<&2,Tasks.Span>,
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) ->
  BoxListed(Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),Tasks.span_count(spans),Nil{}),spans,
    Tasks.piece_extents(axes,prefix,U32.max(per,1),0)):
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +half = Machine.half_word(limits)
  +back = Equal.sym(Nat,ShapeLaws.elements(extents),total,total_exact)
  +room = {%back : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} pieces_half(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(half)) == True{} : Bool}}
  +double_room = {%back : {Nat.is_le(Nat.double(_),U32.to_nat(half)) == True{} : Bool}
    LimitBounds.double_bound(limits,total,GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(half)) == True{} : Bool}}
  box_listed(spans,extents,half,LimitBounds.half_word_fits(limits),within,room,double_room)

# A box of spans inside the pieces of a grid counts at most half a word.
def spans_half(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+prefix: U32,+spans: List<&2,Tasks.Span>,
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) ->
  {Nat.is_le(span_total(spans),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  Order.transitive(span_total(spans),total,U32.to_nat(Machine.half_word(limits)),
    {%total_exact : {Nat.is_le(span_total(spans),_) == True{} : Bool} span_total_le(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),within) :
      {Nat.is_le(span_total(spans),total) == True{} : Bool}},
    pieces_half(limits,outer,extent,unit_cells,block,per,shape))

# The pieces of a box list one for each position.
def listed_length(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(span_total(spans),U32.to_nat(cap)) == True{} : Bool}) ->
  {List.length(&2,U32,Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})) == span_total(spans) : Nat}:
  Equal.trans(Nat,List.length(&2,U32,Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})),
    Nat.add(U32.to_nat(Tasks.span_count(spans)),0n),span_total(spans),
    pieces_length(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{}),
    {%Equal.sym(Nat,Nat.add(U32.to_nat(Tasks.span_count(spans)),0n),U32.to_nat(Tasks.span_count(spans)),Addition.zero_right(U32.to_nat(Tasks.span_count(spans)))) :
        {_ == span_total(spans) : Nat}
      span_count_value(spans,cap,cap_ok,room) : {Nat.add(U32.to_nat(Tasks.span_count(spans)),0n) == span_total(spans) : Nat}})

def box_rising_empty(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +zero: {U32.is_zero(Tasks.span_count(box)) == True{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),Nil{},
      empty_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,box,zero)) :
    Rising(_,0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  Unit{}

def box_rising_listed(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  +spans = Tasks.block_spans(box,prefix,block,0)
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),listed,
      listed_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,box,nonzero,some)) :
    Rising(_,0n,total)
  %total_exact : Rising(listed,0n,_)
  rising_listed(listed,spans,extents,within,pieces_box_listed(limits,outer,extent,unit_cells,block,per,shape,axes,prefix,spans,total_exact,within))

def box_rising_one(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  GridLaws.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+room} = shape
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),Con{0,Nil{}},
      whole_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,box,nonzero,none)) :
    Rising(_,0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  Held.Both{{==},Held.Both{Order.product_positive(U32.to_nat(outer),U32.to_nat(per),outer_positive,
    GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,GridLaws.Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,room})),Unit{}}}

# The pieces of a box rise through the pieces of the grid.
def box_rising(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  Held.by_cases(Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
    U32.is_zero(Tasks.span_count(box)),
    zero => box_rising_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,zero),
    nonzero => box_rising_listed(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,nonzero,some,total_exact,within))

# Without prefix axes a box holds the one piece or none.
def box_rising_whole(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,
  +none: {U32.is_zero(prefix) == True{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  Held.by_cases(Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
    U32.is_zero(Tasks.span_count(box)),
    zero => box_rising_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,zero),
    nonzero => box_rising_one(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,nonzero,none))

def apart_nil(list: List<&2,U32>) -> ListsApart(list,Nil{}):
  match list:
    case Nil{}:
      Unit{}
    case Con{+value,+rest}:
      Held.Both{Unit{},apart_nil(rest)}

def zero_count(+box: List<&2,Tasks.Span>,+zero: {Tasks.span_count(box) == 0 : U32}) -> {U32.is_zero(Tasks.span_count(box)) == True{} : Bool}:
  %Equal.sym(U32,Tasks.span_count(box),0,zero) : {U32.is_zero(_) == True{} : Bool}
  {==}

def apart_first_empty(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,
  +box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,+zero: {U32.is_zero(Tasks.span_count(box)) == True{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(grid,axes,box),Nil{},empty_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,box,zero)) :
    ListsApart(_,Tasks.box_pieces(grid,axes,other))
  Unit{}

def apart_second_empty(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,
  +box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,+zero: {U32.is_zero(Tasks.span_count(other)) == True{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(grid,axes,other),Nil{},empty_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,other,zero)) :
    ListsApart(Tasks.box_pieces(grid,axes,box),_)
  apart_nil(Tasks.box_pieces(grid,axes,box))

def apart_listed(+list: List<&2,U32>,+others: List<&2,U32>,+spans: List<&2,Tasks.Span>,+other_spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,
  +within: SpansIn(spans,extents),+other_within: SpansIn(other_spans,extents),+apart: SpansApart(spans,other_spans),
  +listed: BoxListed(list,spans,extents),+other_listed: BoxListed(others,other_spans,extents)) -> ListsApart(list,others):
  Held.Both{+from,+inside} = listed
  Held.Both{+other_from,+other_inside} = other_listed
  lists_apart(list,others,spans,other_spans,extents,0n,0n,within,other_within,apart,from,inside,other_from,other_inside)

def apart_both_listed(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +other_within: SpansIn(Tasks.block_spans(other,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +apart: SpansApart(Tasks.block_spans(box,prefix,block,0),Tasks.block_spans(other,prefix,block,0)),
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+other_nonzero: {U32.is_zero(Tasks.span_count(other)) == False{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +spans = Tasks.block_spans(box,prefix,block,0)
  +other_spans = Tasks.block_spans(other,prefix,block,0)
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +half = Machine.half_word(limits)
  +back = Equal.sym(Nat,ShapeLaws.elements(extents),total,total_exact)
  +room = {%back : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} pieces_half(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(half)) == True{} : Bool}}
  +double_room = {%back : {Nat.is_le(Nat.double(_),U32.to_nat(half)) == True{} : Bool}
    LimitBounds.double_bound(limits,total,GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(half)) == True{} : Bool}}
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})
  +other_listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(other_spans)),other_spans,extents,Tasks.span_count(other_spans),Nil{})
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(grid,axes,box),listed,listed_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,box,nonzero,some)) :
    ListsApart(_,Tasks.box_pieces(grid,axes,other))
  %Equal.sym(List<&2,U32>,Tasks.box_pieces(grid,axes,other),other_listed,listed_box_list(outer,extent,unit_cells,block,per,count,prefix,axes,other,other_nonzero,some)) :
    ListsApart(listed,_)
  apart_listed(listed,other_listed,spans,other_spans,extents,within,other_within,apart,
    box_listed(spans,extents,half,LimitBounds.half_word_fits(limits),within,room,double_room),
    box_listed(other_spans,extents,half,LimitBounds.half_word_fits(limits),other_within,room,double_room))

def apart_spans_cases(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +other_within: SpansIn(Tasks.block_spans(other,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +apart: SpansApart(Tasks.block_spans(box,prefix,block,0),Tasks.block_spans(other,prefix,block,0)),
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  Held.by_cases(ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
      Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)),
    U32.is_zero(Tasks.span_count(other)),
    zero => apart_second_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero),
    other_nonzero => apart_both_listed(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,other,some,total_exact,within,other_within,apart,
      nonzero,other_nonzero))

def apart_spans(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +other_within: SpansIn(Tasks.block_spans(other,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +apart: SpansApart(Tasks.block_spans(box,prefix,block,0),Tasks.block_spans(other,prefix,block,0))) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  Held.by_cases(ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
      Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)),
    U32.is_zero(Tasks.span_count(box)),
    zero => apart_first_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero),
    nonzero => apart_spans_cases(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,other,some,total_exact,within,other_within,apart,nonzero))

# Boxes apart list pieces apart.
def box_apart(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +other_within: SpansIn(Tasks.block_spans(other,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
  +apart: BoxesApart(box,other,prefix,block)) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  match apart:
    case Inl{+zero}:
      apart_first_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero_count(box,zero))
    case Inr{+later}:
      match later:
        case Inl{+zero}:
          apart_second_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero_count(other,zero))
        case Inr{+spans}:
          apart_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,other,some,total_exact,within,other_within,spans)

def no_prefix_blocks(box: List<&2,Tasks.Span>,+units: U32) -> {Tasks.block_spans(box,0,units,0) == Nil{} : List<&2,Tasks.Span>}:
  match box:
    case Nil{}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest}:
      {==}

def no_prefix_apart(+box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,+prefix: U32,+units: U32,+none: {U32.is_zero(prefix) == True{} : Bool},
  +apart: SpansApart(Tasks.block_spans(box,prefix,units,0),Tasks.block_spans(other,prefix,units,0))) -> Empty:
  +word = Arithmetic.word_zero(prefix,none)
  +zeroed = {%word : SpansApart(Tasks.block_spans(box,_,units,0),Tasks.block_spans(other,_,units,0)) apart :
    SpansApart(Tasks.block_spans(box,0,units,0),Tasks.block_spans(other,0,units,0))}
  +first = {%no_prefix_blocks(box,units) : SpansApart(_,Tasks.block_spans(other,0,units,0)) zeroed : SpansApart(Nil{},Tasks.block_spans(other,0,units,0))}
  {%no_prefix_blocks(other,units) : SpansApart(Nil{},_) first : SpansApart(Nil{},Nil{})}

# Without prefix axes, boxes apart have one empty.
def box_apart_whole(+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+axes: List<&2,U32>,
  +box: List<&2,Tasks.Span>,+other: List<&2,Tasks.Span>,+none: {U32.is_zero(prefix) == True{} : Bool},+apart: BoxesApart(box,other,prefix,block)) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)):
  match apart:
    case Inl{+zero}:
      apart_first_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero_count(box,zero))
    case Inr{+later}:
      match later:
        case Inl{+zero}:
          apart_second_empty(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,zero_count(other,zero))
        case Inr{+spans}:
          Empty.absurd(ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,box),
            Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,other)),
            no_prefix_apart(box,other,prefix,block,none,spans))

# ---- The pieces of the grid of a shape ----

def grid_outer(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,+outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  outer

def grid_extent(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,outer,+extent,unit_cells,piece_units,per_outer,pieces} = grid
  extent

def units_grid_prefix(+found: Grid.Units,+target: U32) -> {grid_prefix(Grid.grid_of_units(found,target)) == units_prefix(found) : U32}:
  Grid.Units{+prefix,+outer,+extent,+unit_cells} = found
  {==}

def units_grid_outer(+found: Grid.Units,+target: U32) -> {grid_outer(Grid.grid_of_units(found,target)) == units_outer(found) : U32}:
  Grid.Units{+prefix,+outer,+extent,+unit_cells} = found
  {==}

def units_grid_extent(+found: Grid.Units,+target: U32) -> {grid_extent(Grid.grid_of_units(found,target)) == units_extent(found) : U32}:
  Grid.Units{+prefix,+outer,+extent,+unit_cells} = found
  {==}

def one_times_nat(+value: Nat) -> {value == Nat.mul(U32.to_nat(1),value) : Nat}:
  Equal.sym(Nat,Nat.add(value,0n),value,Addition.zero_right(value))

def top_room(+first: U32,+rest: List<&2,U32>,+cap: U32,
  +room: {Nat.is_le(ShapeLaws.elements(Con{first,rest}),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Con{first,rest})) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Con{first,rest}),U32.to_nat(cap)) == True{} : Bool}) -> AxesRoom(1,first,rest,0n,cap):
  Held.Both{{%Equal.sym(Nat,Nat.add(U32.to_nat(first),0n),U32.to_nat(first),Addition.zero_right(U32.to_nat(first))) :
        {Nat.is_le(Nat.mul(_,ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool}
      room : {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(1),U32.to_nat(first)),ShapeLaws.elements(rest)),U32.to_nat(cap)) == True{} : Bool}},
    Held.Both{DivisionOrder.right_factor_positive(U32.to_nat(first),ShapeLaws.elements(rest),positive),rank_room}}

def top_total(+first: U32,+rest: List<&2,U32>,+prefix: U32,+per: U32,+outer: U32,
  +total: {Nat.mul(U32.to_nat(1),ShapeLaws.elements(Tasks.piece_extents(Con{first,rest},prefix,per,0))) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  {ShapeLaws.elements(Tasks.piece_extents(Con{first,rest},prefix,per,0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}:
  +inner = ShapeLaws.elements(Tasks.piece_extents(Con{first,rest},prefix,per,0))
  Equal.trans(Nat,inner,Nat.mul(U32.to_nat(1),inner),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),one_times_nat(inner),total)

def no_prefix(-Goal: Data,+some: {U32.is_zero(0) == False{} : Bool}) -> Goal:
  Empty.absurd(Goal,WordEncoding.false_true(Equal.sym(Bool,True{},False{},some)))

# With prefix axes the first step of the units walk was taken: the pieces
# number the outer indices times per, and the last prefix axis is the extent.
def top_pieces(axes: List<&2,U32>,fits: Bool,+target: U32,+per: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(cap)) == True{} : Bool},
  +some: {U32.is_zero(units_prefix(GridLaws.units_model(axes,fits,0,1,1,target))) == False{} : Bool}) ->
  Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,units_prefix(GridLaws.units_model(axes,fits,0,1,1,target)),per,0)) ==
      Nat.mul(U32.to_nat(units_outer(GridLaws.units_model(axes,fits,0,1,1,target))),U32.to_nat(per)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(units_prefix(GridLaws.units_model(axes,fits,0,1,1,target))),1n)) == units_extent(GridLaws.units_model(axes,fits,0,1,1,target)) : U32}>:
  match axes fits:
    case +axes True{}:
      no_prefix(Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,0,per,0)) == Nat.mul(U32.to_nat(1),U32.to_nat(per)) : Nat},
        {extent_nat(axes,Nat.sub(U32.to_nat(0),1n)) == 1 : U32}>,some)
    case Nil{} False{}:
      no_prefix(Held.Both<{ShapeLaws.elements(Tasks.piece_extents(Nil{},0,per,0)) == Nat.mul(U32.to_nat(1),U32.to_nat(per)) : Nat},
        {extent_nat(Nil{},Nat.sub(U32.to_nat(0),1n)) == 1 : U32}>,some)
    case Con{+last,Nil{}} False{}:
      no_prefix(Held.Both<{ShapeLaws.elements(Tasks.piece_extents(Con{last,Nil{}},0,per,0)) == Nat.mul(U32.to_nat(1),U32.to_nat(per)) : Nat},
        {extent_nat(Con{last,Nil{}},Nat.sub(U32.to_nat(0),1n)) == 1 : U32}>,some)
    case Con{+first,Con{+next,+tail}} False{}:
      +rest = {Con{next,tail} : List<&2,U32>}
      +next_fits = U32.is_le(Grid.product(rest),target)
      +found = GridLaws.units_model(rest,next_fits,(0 + 1 : U32),1,first,target)
      +axes_room = top_room(first,rest,cap,room,positive,rank_room)
      Held.Both{top_total(first,rest,units_prefix(found),per,units_outer(found),
          units_total(rest,next_fits,0,1,first,target,per,0n,cap,cap_ok,{==},axes_room)),
        units_extent_at(rest,next_fits,0,1,first,target,0n,cap,cap_ok,{==},axes_room)}

def shape_grid_prefix(+axes: List<&2,U32>,+target: U32) -> {grid_prefix(Grid.grid(axes,target)) == units_prefix(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) : U32}:
  Equal.trans(U32,grid_prefix(Grid.grid(axes,target)),grid_prefix(Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target)),units_prefix(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)),
    Equal.cong(Grid.Grid,U32,whole => grid_prefix(whole),Grid.grid(axes,target),Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target),GridLaws.grid_model(axes,target)),
    units_grid_prefix(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target))

def shape_grid_outer(+axes: List<&2,U32>,+target: U32) -> {grid_outer(Grid.grid(axes,target)) == units_outer(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) : U32}:
  Equal.trans(U32,grid_outer(Grid.grid(axes,target)),grid_outer(Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target)),units_outer(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)),
    Equal.cong(Grid.Grid,U32,whole => grid_outer(whole),Grid.grid(axes,target),Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target),GridLaws.grid_model(axes,target)),
    units_grid_outer(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target))

def shape_grid_extent(+axes: List<&2,U32>,+target: U32) -> {grid_extent(Grid.grid(axes,target)) == units_extent(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) : U32}:
  Equal.trans(U32,grid_extent(Grid.grid(axes,target)),grid_extent(Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target)),units_extent(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)),
    Equal.cong(Grid.Grid,U32,whole => grid_extent(whole),Grid.grid(axes,target),Grid.grid_of_units(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target),GridLaws.grid_model(axes,target)),
    units_grid_extent(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target),target))

def shape_prefix_eq(+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}) ->
  {units_prefix(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) == prefix : U32}:
  +found = GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)
  Equal.trans(U32,units_prefix(found),grid_prefix(Grid.grid(axes,target)),prefix,Equal.sym(U32,grid_prefix(Grid.grid(axes,target)),units_prefix(found),
    shape_grid_prefix(axes,target)),Equal.cong(Grid.Grid,U32,whole => grid_prefix(whole),Grid.grid(axes,target),
      Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact))

# The grid of a shape with prefix axes: its pieces as an index number its outer
# indices times per, and its last prefix axis is its extent.
def shape_pieces(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)) == extent : U32}>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +fits = U32.is_le(Grid.product(axes),target)
  +found = GridLaws.units_model(axes,fits,0,1,1,target)
  +prefix_eq = shape_prefix_eq(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact)
  +outer_eq = Equal.trans(U32,units_outer(found),grid_outer(Grid.grid(axes,target)),outer,Equal.sym(U32,grid_outer(Grid.grid(axes,target)),units_outer(found),
    shape_grid_outer(axes,target)),Equal.cong(Grid.Grid,U32,whole => grid_outer(whole),Grid.grid(axes,target),grid,exact))
  +extent_eq = Equal.trans(U32,units_extent(found),grid_extent(Grid.grid(axes,target)),extent,Equal.sym(U32,grid_extent(Grid.grid(axes,target)),units_extent(found),
    shape_grid_extent(axes,target)),Equal.cong(Grid.Grid,U32,whole => grid_extent(whole),Grid.grid(axes,target),grid,exact))
  +per_kept = GridLaws.maximum_kept(per,1,Order.positive_one(U32.to_nat(per),GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape)))
  +found_some = {%Equal.sym(U32,units_prefix(found),prefix,prefix_eq) : {U32.is_zero(_) == False{} : Bool} some : {U32.is_zero(units_prefix(found)) == False{} : Bool}}
  %prefix_eq : Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,_,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(_),1n)) == extent : U32}>
  %outer_eq : Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,units_prefix(found),U32.max(per,1),0)) == Nat.mul(U32.to_nat(_),U32.to_nat(per)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(units_prefix(found)),1n)) == extent : U32}>
  %extent_eq : Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,units_prefix(found),U32.max(per,1),0)) == Nat.mul(U32.to_nat(units_outer(found)),U32.to_nat(per)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(units_prefix(found)),1n)) == _ : U32}>
  %per_kept : Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,units_prefix(found),U32.max(per,1),0)) == Nat.mul(U32.to_nat(units_outer(found)),U32.to_nat(_)) : Nat},
    {extent_nat(axes,Nat.sub(U32.to_nat(units_prefix(found)),1n)) == units_extent(found) : U32}>
  top_pieces(axes,fits,target,U32.max(per,1),Machine.capacity(limits),LimitBounds.capacity_fits(limits),room,positive,rank_room,found_some)

def top_within(axes: List<&2,U32>,fits: Bool,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(units_prefix(GridLaws.units_model(axes,fits,0,1,1,target))),List.length(&2,U32,axes)) == True{} : Bool}:
  match axes fits:
    case +axes True{}:
      Order.zero_le(List.length(&2,U32,axes))
    case Nil{} False{}:
      {==}
    case Con{+last,Nil{}} False{}:
      Order.zero_le(1n)
    case Con{+first,Con{+next,+tail}} False{}:
      +rest = {Con{next,tail} : List<&2,U32>}
      prefix_within(rest,U32.is_le(Grid.product(rest),target),0,1,first,target,0n,cap,cap_ok,{==},rank_room)

# The prefix axes of the grid of a shape are axes of the shape.
def shape_within(+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(prefix),List.length(&2,U32,axes)) == True{} : Bool}:
  %shape_prefix_eq(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact) : {Nat.is_le(U32.to_nat(_),List.length(&2,U32,axes)) == True{} : Bool}
  top_within(axes,U32.is_le(Grid.product(axes),target),target,cap,cap_ok,rank_room)

def capacity_half(+limits: Machine.Limits) -> {Nat.is_le(U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),LimitBounds.capacity_within_half(limits))

def unit_half(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>) ->
  {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +kept = GridLaws.maximum_kept(block,1,Order.positive_one(U32.to_nat(block),GridLaws.shape_block(limits,outer,extent,unit_cells,block,per,shape)))
  %Equal.sym(U32,U32.max(block,1),block,kept) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  LimitBounds.double_bound(limits,U32.to_nat(block),Order.transitive(U32.to_nat(block),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),
    GridLaws.shape_below(limits,outer,extent,unit_cells,block,per,shape),GridLaws.extent_room(limits,outer,extent,unit_cells,block,per,shape)))

def pieces_total(+axes: List<&2,U32>,+prefix: U32,+per: U32,+outer: U32,+extent: U32,+facts: Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},{extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)) == extent : U32}>) -> {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}:
  Held.Both{+total,+last} = facts
  total

def pieces_last(+axes: List<&2,U32>,+prefix: U32,+per: U32,+outer: U32,+extent: U32,+facts: Held.Both<{ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},{extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)) == extent : U32}>) -> {extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)) == extent : U32}:
  Held.Both{+total,+last} = facts
  last

# The last prefix axis fits the pieces of one outer index.
def last_fits(+limits: Machine.Limits,+axes: List<&2,U32>,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +last: {extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)) == extent : U32}) ->
  {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n))),U32.to_nat(U32.max(block,1))),U32.to_nat(U32.max(per,1))) == True{} : Bool}:
  +block_kept = GridLaws.maximum_kept(block,1,Order.positive_one(U32.to_nat(block),GridLaws.shape_block(limits,outer,extent,unit_cells,block,per,shape)))
  +per_kept = GridLaws.maximum_kept(per,1,Order.positive_one(U32.to_nat(per),GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape)))
  %Equal.sym(U32,extent_nat(axes,Nat.sub(U32.to_nat(prefix),1n)),extent,last) : {Nat.is_le(NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(U32.max(block,1))),U32.to_nat(U32.max(per,1))) == True{} : Bool}
  %Equal.sym(U32,U32.max(block,1),block,block_kept) : {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_)),U32.to_nat(U32.max(per,1))) == True{} : Bool}
  %Equal.sym(U32,U32.max(per,1),per,per_kept) : {Nat.is_le(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(_)) == True{} : Bool}
  %GridLaws.shape_per(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_le(_,U32.to_nat(per)) == True{} : Bool}
  Order.reflexive(U32.to_nat(per))

def source_total(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}:
  pieces_total(axes,prefix,per,outer,extent,shape_pieces(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some))

def rank_half(+limits: Machine.Limits,+axes: List<&2,U32>,
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),rank_room,capacity_half(limits))

# The last prefix axis is an axis of the shape.
def prefix_steps_in(+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(cap)) == True{} : Bool},+some: {U32.is_zero(prefix) == False{} : Bool}) ->
  {Nat.is_lt(Nat.sub(U32.to_nat(prefix),1n),List.length(&2,U32,axes)) == True{} : Bool}:
  Order.lt_le(Nat.sub(U32.to_nat(prefix),1n),U32.to_nat(prefix),List.length(&2,U32,axes),pred_lt(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some)),
    shape_within(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,cap_ok,rank_room))

# The block spans of a box inside the source axes up to the last prefix axis
# lie inside the pieces of its grid.
def source_blocks(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool},+box: List<&2,Tasks.Span>,
  +along: SpansAlong(box,axes,Nat.sub(U32.to_nat(prefix),1n),Machine.half_word(limits))) ->
  SpansIn(Tasks.block_spans(box,prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)):
  +half = Machine.half_word(limits)
  +prefix_positive = Arithmetic.word_nonzero_positive(prefix,some)
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,axes),U32.to_nat(half),
    shape_within(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact,Machine.capacity(limits),LimitBounds.capacity_fits(limits),rank_room),
    rank_half(limits,axes,rank_room))
  +last = pieces_last(axes,prefix,per,outer,extent,shape_pieces(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some))
  blocks_within(box,axes,prefix,block,0,0n,Nat.sub(U32.to_nat(prefix),1n),U32.max(per,1),half,LimitBounds.half_word_fits(limits),{==},
    Order.positive_has_last(U32.to_nat(prefix),prefix_positive),prefix_room,along,unit_half(limits,outer,extent,unit_cells,block,per,shape),
    last_fits(limits,axes,outer,extent,unit_cells,block,per,prefix,shape,last))

# The block spans of the box a component reads of a source lie inside the
# pieces of its grid.
def source_within(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool},+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +rooms: ReadingsRoom(row,Machine.half_word(limits)),+ends: RowEnds(row,classes),
  +axes_ok: AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,0,Machine.half_word(limits))) ->
  SpansIn(Tasks.block_spans(Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),
      prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +steps_in = prefix_steps_in(axes,target,count,prefix,outer,extent,unit_cells,block,per,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,rank_room,some)
  source_blocks(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some,
    Tasks.boxed(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0),
    boxed_along(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0,steps,Machine.half_word(limits),LimitBounds.half_word_fits(limits),
      reach_coordinates(axes,offset,steps,steps_in),rooms,ends,axes_ok))

# Whole spans lie inside their axes.
def whole_along(extents: List<&2,U32>,remaining: Nat,+cap: U32,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},+inside: {Nat.is_lt(remaining,List.length(&2,U32,extents)) == True{} : Bool}) ->
  SpansAlong(Tasks.whole_spans(extents),extents,remaining,cap):
  match extents remaining:
    case Nil{} +remaining:
      Empty.absurd(SpansAlong(Tasks.whole_spans(Nil{}),Nil{},remaining,cap),Order.below_zero(remaining,inside))
    case Con{+extent,+rest} 0n:
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      Held.Both{Order.reflexive(U32.to_nat(extent)),Order.transitive(U32.to_nat(extent),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),
        RowMajor.product_covers(U32.to_nat(extent),ShapeLaws.elements(rest),rest_positive),room)}
    case Con{+extent,+rest} 1n+ +later:
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      +rest_room = Order.transitive(ShapeLaws.elements(rest),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),
        Held.nat_le_right_moved(ShapeLaws.elements(rest),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),Nat.mul(ShapeLaws.elements(rest),U32.to_nat(extent)),
          RowMajor.multiply_commutative(U32.to_nat(extent),ShapeLaws.elements(rest)),RowMajor.product_covers(ShapeLaws.elements(rest),U32.to_nat(extent),extent_positive)),room)
      Held.Both{Order.reflexive(U32.to_nat(extent)),whole_along(rest,later,cap,rest_room,rest_positive,inside)}

# The block spans of a whole source lie inside the pieces of its grid.
def source_whole_within(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  SpansIn(Tasks.block_spans(Tasks.whole_spans(axes),prefix,block,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)):
  +half = Machine.half_word(limits)
  source_blocks(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some,Tasks.whole_spans(axes),
    whole_along(axes,Nat.sub(U32.to_nat(prefix),1n),half,Order.transitive(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),room,
      capacity_half(limits)),positive,
      prefix_steps_in(axes,target,count,prefix,outer,extent,unit_cells,block,per,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,rank_room,some)))

def source_rising_some(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +rooms: ReadingsRoom(row,Machine.half_word(limits)),+ends: RowEnds(row,classes),
  +axes_ok: AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,0,Machine.half_word(limits)),
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0)),
    0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  box_rising(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,Tasks.boxed(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0),some,
    source_total(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some),
    source_within(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some,row,classes,offset,rooms,ends,axes_ok))

# The pieces a component lists of a source rise through its pieces.
def source_rising(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +rooms: ReadingsRoom(row,Machine.half_word(limits)),+ends: RowEnds(row,classes),
  +axes_ok: AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,0,Machine.half_word(limits))) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0)),
    0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +box = Tasks.boxed(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0)
  Held.by_cases(Rising(Tasks.box_pieces(grid,axes,box),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),U32.is_zero(prefix),
    none => box_rising_whole(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,box,none),
    some => source_rising_some(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,row,classes,offset,rooms,ends,
      axes_ok,some))

def source_apart_some(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+offset: U32,
  +rooms: ReadingsRoom(row,Machine.half_word(limits)),+ends: RowEnds(row,classes),+other_ends: RowEnds(row,others),
  +axes_ok: AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,0,Machine.half_word(limits)),
  +apart: BoxesApart(Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),
    Tasks.boxed(row,others,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),prefix,block),
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0)),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,others,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  box_apart(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,Tasks.boxed(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0),
    Tasks.boxed(row,others,grid,axes,GridLaws.coordinates_model(axes,offset),0),some,
    source_total(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some),
    source_within(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some,row,classes,offset,rooms,ends,axes_ok),
    source_within(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some,row,others,offset,rooms,other_ends,axes_ok),
    apart)

# Two components whose classes read apart list pieces apart of a source.
def source_apart(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,+offset: U32,
  +first: FirstApart(row,classes,others),+distinct: {Tasks.distinct(Tasks.read_axes(row)) == True{} : Bool},
  +rooms: ReadingsRoom(row,Machine.half_word(limits)),+ends: RowEnds(row,classes),+other_ends: RowEnds(row,others),
  +axes_ok: AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,0,Machine.half_word(limits)),
  +below: AxesBelow(Tasks.read_axes(row),List.length(&2,U32,axes))) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0)),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(row,others,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +box = Tasks.boxed(row,classes,grid,axes,GridLaws.coordinates_model(axes,offset),0)
  +other = Tasks.boxed(row,others,grid,axes,GridLaws.coordinates_model(axes,offset),0)
  +half = Machine.half_word(limits)
  +ranks = rank_half(limits,axes,rank_room)
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,axes),U32.to_nat(half),
    shape_within(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact,Machine.capacity(limits),LimitBounds.capacity_fits(limits),rank_room),ranks)
  +apart = boxes_apart(row,classes,others,grid,axes,offset,half,LimitBounds.half_word_fits(limits),first,distinct,rooms,ends,other_ends,axes_ok,below,ranks,prefix_room,
    GridLaws.shape_block(limits,outer,extent,unit_cells,block,per,shape),unit_half(limits,outer,extent,unit_cells,block,per,shape))
  Held.by_cases(ListsApart(Tasks.box_pieces(grid,axes,box),Tasks.box_pieces(grid,axes,other)),U32.is_zero(prefix),
    none => box_apart_whole(outer,extent,unit_cells,block,per,count,prefix,axes,box,other,none,apart),
    some => source_apart_some(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,row,classes,others,offset,
      rooms,ends,other_ends,axes_ok,apart,some))

# ---- The row of one source ----

def index_step(+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  +room: {Nat.is_le(U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((index - 1 : U32)) == Nat.sub(U32.to_nat(index),1n) : Nat}:
  Arithmetic.difference_exact(cap,cap_ok,index,1,U32.to_nat(index),1n,{==},{==},Order.positive_one(U32.to_nat(index),Arithmetic.word_nonzero_positive(index,nonzero)),room)

def pred_below(value: Nat,+limit: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool},+below: {Nat.is_lt(value,1n+limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(value,1n),limit) == True{} : Bool}:
  match value:
    case 0n:
      Empty.absurd({Nat.is_lt(Nat.sub(0n,1n),limit) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +smaller:
      %Equal.sym(Nat,Nat.sub(smaller,0n),smaller,Order.subtract_zero(smaller)) : {Nat.is_lt(_,limit) == True{} : Bool}
      below

# The index of the next row: one less, inside the rest.
def next_row(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +nonzero: {U32.is_zero(index) == False{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest})) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,List<&2,Tasks.Reading>,rest)) == True{} : Bool}:
  +index_room = Order.transitive(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap),
    lt_to_le(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),inside),room)
  %Equal.sym(Nat,U32.to_nat((index - 1 : U32)),Nat.sub(U32.to_nat(index),1n),index_step(index,cap,cap_ok,nonzero,index_room)) :
    {Nat.is_lt(_,List.length(&2,List<&2,Tasks.Reading>,rest)) == True{} : Bool}
  pred_below(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,rest),Arithmetic.word_nonzero_positive(index,nonzero),inside)

def rest_room(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+cap: U32,
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,rest),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(List.length(&2,List<&2,Tasks.Reading>,rest),1n+List.length(&2,List<&2,Tasks.Reading>,rest),U32.to_nat(cap),
    Order.successor(List.length(&2,List<&2,Tasks.Reading>,rest),List.length(&2,List<&2,Tasks.Reading>,rest),Order.reflexive(List.length(&2,List<&2,Tasks.Reading>,rest))),room)

def first_here(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,
  +zero: {U32.is_zero(index) == True{} : Bool},+here: FirstApart(own,classes,others)) -> FirstApart(readings_word(Con{own,rest},index),classes,others):
  %Equal.sym(Bool,U32.is_zero(index),True{},zero) : FirstApart(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))),classes,others)
  here

def first_there(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest})) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap)) == True{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,List<&2,Tasks.Reading>,rest)) == True{} : Bool} ->
    FirstApart(readings_word(rest,(index - 1 : U32)),classes,others)) -> FirstApart(readings_word(Con{own,rest},index),classes,others):
  %Equal.sym(Bool,U32.is_zero(index),False{},nonzero) : FirstApart(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))),classes,others)
  recurse(next_row(own,rest,index,cap,cap_ok,nonzero,inside,room))

# Every row of the readings reads first apart, so does the row of a source.
def first_at(readings: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,+others: List<&2,Tasks.Class>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+all: AllFirstApart(readings,classes,others),
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}) ->
  FirstApart(readings_word(readings,index),classes,others):
  match readings:
    case Nil{}:
      Empty.absurd(FirstApart(readings_word(Nil{},index),classes,others),Order.below_zero(U32.to_nat(index),inside))
    case Con{+own,+rest}:
      Held.Both{+here,+later} = all
      Held.by_cases(FirstApart(readings_word(Con{own,rest},index),classes,others),U32.is_zero(index),
        zero => first_here(own,rest,index,classes,others,zero,here),
        nonzero => first_there(own,rest,index,classes,others,cap,cap_ok,nonzero,inside,room,
          step => first_at(rest,(index - 1 : U32),classes,others,cap,cap_ok,later,step,rest_room(own,rest,cap,room))))

def ends_here(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,
  +zero: {U32.is_zero(index) == True{} : Bool},+here: RowEnds(own,classes)) -> RowEnds(readings_word(Con{own,rest},index),classes):
  %Equal.sym(Bool,U32.is_zero(index),True{},zero) : RowEnds(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))),classes)
  here

def ends_there(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+nonzero: {U32.is_zero(index) == False{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest})) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap)) == True{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,List<&2,Tasks.Reading>,rest)) == True{} : Bool} ->
    RowEnds(readings_word(rest,(index - 1 : U32)),classes)) -> RowEnds(readings_word(Con{own,rest},index),classes):
  %Equal.sym(Bool,U32.is_zero(index),False{},nonzero) : RowEnds(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))),classes)
  recurse(next_row(own,rest,index,cap,cap_ok,nonzero,inside,room))

def ends_at(readings: List<&2,List<&2,Tasks.Reading>>,+index: U32,+classes: List<&2,Tasks.Class>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+all: AllEnds(readings,classes),
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}) ->
  RowEnds(readings_word(readings,index),classes):
  match readings:
    case Nil{}:
      Empty.absurd(RowEnds(readings_word(Nil{},index),classes),Order.below_zero(U32.to_nat(index),inside))
    case Con{+own,+rest}:
      Held.Both{+here,+later} = all
      Held.by_cases(RowEnds(readings_word(Con{own,rest},index),classes),U32.is_zero(index),
        zero => ends_here(own,rest,index,classes,zero,here),
        nonzero => ends_there(own,rest,index,classes,cap,cap_ok,nonzero,inside,room,
          step => ends_at(rest,(index - 1 : U32),classes,cap,cap_ok,later,step,rest_room(own,rest,cap,room))))

def separate_own(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+separate: {Tasks.separable(Con{own,rest}) == True{} : Bool}) ->
  {Tasks.distinct(Tasks.read_axes(own)) == True{} : Bool}:
  Logic.left(Tasks.distinct(Tasks.read_axes(own)),Bool.not(Tasks.crossing(own)),
    Logic.left(Tasks.distinct(Tasks.read_axes(own)) && Bool.not(Tasks.crossing(own)),Tasks.separable(rest),separate))

def separate_rest(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+separate: {Tasks.separable(Con{own,rest}) == True{} : Bool}) ->
  {Tasks.separable(rest) == True{} : Bool}:
  Logic.right(Tasks.distinct(Tasks.read_axes(own)) && Bool.not(Tasks.crossing(own)),Tasks.separable(rest),separate)

def distinct_here(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+zero: {U32.is_zero(index) == True{} : Bool},
  +here: {Tasks.distinct(Tasks.read_axes(own)) == True{} : Bool}) -> {Tasks.distinct(Tasks.read_axes(readings_word(Con{own,rest},index))) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(index),True{},zero) :
    {Tasks.distinct(Tasks.read_axes(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))))) == True{} : Bool}
  here

def distinct_there(+own: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +nonzero: {U32.is_zero(index) == False{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,Con{own,rest})) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Con{own,rest}),U32.to_nat(cap)) == True{} : Bool},
  recurse: {Nat.is_lt(U32.to_nat((index - 1 : U32)),List.length(&2,List<&2,Tasks.Reading>,rest)) == True{} : Bool} ->
    {Tasks.distinct(Tasks.read_axes(readings_word(rest,(index - 1 : U32)))) == True{} : Bool}) ->
  {Tasks.distinct(Tasks.read_axes(readings_word(Con{own,rest},index))) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(index),False{},nonzero) :
    {Tasks.distinct(Tasks.read_axes(Bool.pick(List<&2,Tasks.Reading>,_,own,readings_word(rest,(index - 1 : U32))))) == True{} : Bool}
  recurse(next_row(own,rest,index,cap,cap_ok,nonzero,inside,room))

# Separable readings read every axis of a row once.
def distinct_at(readings: List<&2,List<&2,Tasks.Reading>>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +separate: {Tasks.separable(readings) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.distinct(Tasks.read_axes(readings_word(readings,index))) == True{} : Bool}:
  match readings:
    case Nil{}:
      Empty.absurd({Tasks.distinct(Tasks.read_axes(readings_word(Nil{},index))) == True{} : Bool},Order.below_zero(U32.to_nat(index),inside))
    case Con{+own,+rest}:
      Held.by_cases({Tasks.distinct(Tasks.read_axes(readings_word(Con{own,rest},index))) == True{} : Bool},U32.is_zero(index),
        zero => distinct_here(own,rest,index,zero,separate_own(own,rest,separate)),
        nonzero => distinct_there(own,rest,index,cap,cap_ok,nonzero,inside,room,
          step => distinct_at(rest,(index - 1 : U32),cap,cap_ok,separate_rest(own,rest,separate),step,rest_room(own,rest,cap,room))))

# ---- Taking the pieces of every source ----

# The spans a source has given after the components below steps took their
# pieces, the last taken first.
def gone_upto(+components: Tasks.Components,+source: U32,+origin: Tensors.Origin,steps: Nat) -> Held.Spans:
  match steps:
    case 0n:
      Held.Ended{}
    case 1n+ +before:
      Moved.taken_gone(Moved.source_pieces(components,source,U32.from_nat(before),origin),Moved.origin_grid(origin),gone_upto(components,source,origin,before))

def gones_upto(origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+steps: Nat) -> Moved.Gones:
  match origins:
    case Nil{}:
      Moved.NoGones{}
    case Con{+origin,origins}:
      Moved.Gones{gone_upto(components,source,origin,steps),gones_upto(origins,components,(source + 1 : U32),steps)}

# One more component takes its pieces of every source.
def after_upto(origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+steps: Nat) ->
  {Moved.gones_after(origins,components,source,U32.from_nat(steps),gones_upto(origins,components,source,steps)) == gones_upto(origins,components,source,1n+steps) : Moved.Gones}:
  match origins:
    case Nil{}:
      {==}
    case Con{+origin,+rest}:
      Equal.cong(Moved.Gones,Moved.Gones,later => Moved.Gones{gone_upto(components,source,origin,1n+steps),later},
        Moved.gones_after(rest,components,(source + 1 : U32),U32.from_nat(steps),gones_upto(rest,components,(source + 1 : U32),steps)),
        gones_upto(rest,components,(source + 1 : U32),1n+steps),after_upto(rest,components,(source + 1 : U32),steps))

# The facts the class lists of the readings need.
def ComponentsOk(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32) -> Data:
  Held.Both<RowsAt(readings,rank),Held.Both<ColumnsOk(rank,readings,cap),Held.Both<ListsRoom(raw_lists(rank,readings),cap),
    Held.Both<{Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
      {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}>>>>

def components_ends(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ComponentsOk(readings,rank,cap),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  AllEnds(readings,chosen_word(raw_lists(rank,readings),index)):
  Held.Both{+rows,+more} = ok
  Held.Both{+columns,+lists} = more
  Held.Both{+lists_room,+rooms} = lists
  Held.Both{+room,+rows_room} = rooms
  chosen_ends(rank,readings,index,cap,cap_ok,rows,columns,lists_room,room,inside)

# The classes the runtime chooses for a component are its chosen classes.
def components_chosen(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ComponentsOk(readings,rank,cap),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Tasks.chosen(Tasks.class_lists(rank,readings),index) == chosen_word(raw_lists(rank,readings),index) : List<&2,Tasks.Class>}:
  Held.Both{+rows,Held.Both{+columns,Held.Both{+lists_room,Held.Both{+lists_double,+rows_room}}}} = ok
  classes_chosen(rank,readings,index,cap,cap_ok,lists_room,lists_double,inside)

# Every row ends in the classes the runtime chooses for a component.
def runtime_ends(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ComponentsOk(readings,rank,cap),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  AllEnds(readings,Tasks.chosen(Tasks.class_lists(rank,readings),index)):
  %Equal.sym(List<&2,Tasks.Class>,Tasks.chosen(Tasks.class_lists(rank,readings),index),chosen_word(raw_lists(rank,readings),index),components_chosen(readings,rank,cap,cap_ok,ok,index,inside)) : AllEnds(readings,_)
  components_ends(readings,rank,cap,cap_ok,ok,index,inside)

def components_first(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ok: ComponentsOk(readings,rank,cap),+first: U32,+second: U32,
  +first_in: {Nat.is_lt(U32.to_nat(first),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +second_in: {Nat.is_lt(U32.to_nat(second),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool}) ->
  AllFirstApart(readings,chosen_word(raw_lists(rank,readings),first),chosen_word(raw_lists(rank,readings),second)):
  Held.Both{+rows,+more} = ok
  Held.Both{+columns,+lists} = more
  Held.Both{+lists_room,+rooms} = lists
  Held.Both{+room,+rows_room} = rooms
  chosen_apart(rank,readings,first,second,cap,cap_ok,rows,columns,lists_room,room,first_in,second_in,differ)

def components_rows(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+ok: ComponentsOk(readings,rank,cap)) ->
  {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}:
  Held.Both{+rows,+more} = ok
  Held.Both{+columns,+lists} = more
  Held.Both{+lists_room,+rooms} = lists
  Held.Both{+room,+rows_room} = rooms
  rows_room

def components_count(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+ok: ComponentsOk(readings,rank,cap)) ->
  {Nat.is_le(count_nat(raw_lists(rank,readings)),U32.to_nat(cap)) == True{} : Bool}:
  Held.Both{+rows,+more} = ok
  Held.Both{+columns,+lists} = more
  Held.Both{+lists_room,+rooms} = lists
  Held.Both{+room,+rows_room} = rooms
  Order.transitive(count_nat(raw_lists(rank,readings)),Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap),
    Order.le_double(count_nat(raw_lists(rank,readings))),room)

# Twice a number at most twice another: the number is at most the other.
def halved_le(+a: Nat,+b: Nat,ordered: {Nat.is_le(Nat.double(a),Nat.double(b)) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  match a b:
    case 0n b: Order.zero_le(b)
    case 1n+a 0n: Empty.absurd({Nat.is_le(1n+a,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+a 1n+b: halved_le(a,b,ordered)

# The components that fit half a word twice over fit the capacity.
def components_within(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,+ok: ComponentsOk(readings,rank,Machine.half_word(limits))) ->
  {Nat.is_le(count_nat(raw_lists(rank,readings)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Held.Both{+rows,Held.Both{+columns,Held.Both{+lists_room,Held.Both{+room,+rows_room}}}} = ok
  +count = count_nat(raw_lists(rank,readings))
  halved_le(count,U32.to_nat(Machine.capacity(limits)),
    {%LimitBounds.doubled_capacity(limits) : {Nat.is_le(Nat.double(count),_) == True{} : Bool} room :
      {Nat.is_le(Nat.double(count),Nat.double(U32.to_nat(Machine.capacity(limits)))) == True{} : Bool}})

# The runtime box of a source for a component of separate sources is the
# model's box.
def source_box_model(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+count: U32,+source: U32,+grid: Grid.Grid,+axes: List<&2,U32>,+offset: U32,+index: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ok: ComponentsOk(readings,rank,cap),
  +source_in: {Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),count,True{}},source,grid,axes,offset,index) ==
    Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),grid,axes,GridLaws.coordinates_model(axes,offset),0) : List<&2,Tasks.Span>}:
  Held.Both{+rows,Held.Both{+columns,Held.Both{+lists_room,Held.Both{+room,+rows_room}}}} = ok
  +source_room = Order.transitive(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap),
    lt_to_le(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings),source_in),rows_room)
  %Equal.sym(List<&2,Tasks.Reading>,Tasks.readings_at(readings,U32.to_nat(source)),readings_word(readings,source),readings_word_exact(readings,source,cap,cap_ok,source_room)) :
    {Tasks.boxed(_,Tasks.chosen(counted_all(raw_lists(rank,readings)),index),grid,axes,Grid.coordinates(axes,offset),0) ==
      Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),grid,axes,GridLaws.coordinates_model(axes,offset),0) : List<&2,Tasks.Span>}
  %Equal.sym(List<&2,Tasks.Class>,Tasks.chosen(counted_all(raw_lists(rank,readings)),index),chosen_word(raw_lists(rank,readings),index),
      chosen_exact(raw_lists(rank,readings),index,cap,cap_ok,lists_room,room,index_in)) :
    {Tasks.boxed(readings_word(readings,source),_,grid,axes,Grid.coordinates(axes,offset),0) ==
      Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),grid,axes,GridLaws.coordinates_model(axes,offset),0) : List<&2,Tasks.Span>}
  %Equal.sym(List<&2,U32>,Grid.coordinates(axes,offset),GridLaws.coordinates_model(axes,offset),GridLaws.coordinates_exact(axes,offset)) :
    {Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),grid,axes,_,0) ==
      Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),grid,axes,GridLaws.coordinates_model(axes,offset),0) : List<&2,Tasks.Span>}
  {==}

# The cells of a source shape fit the machine, and so does its rank.
def ShapeOk(+limits: Machine.Limits,+shape: List<&2,U32>) -> Data:
  Held.Both<{Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    Held.Both<{Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},{Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}>>

# The readings of a source row fit cap a word and read its axes.
def RowOk(+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+shape: List<&2,U32>) -> Data:
  Held.Both<ReadingsRoom(row,Machine.half_word(limits)),Held.Both<AxesOk(row,grid,shape,0,Machine.half_word(limits)),
    AxesBelow(Tasks.read_axes(row),List.length(&2,U32,shape))>>

def source_rising_at(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},+shape_ok: ShapeOk(limits,axes),
  +readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+source: U32,+offset: U32,+index: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +row_ok: RowOk(limits,readings_word(readings,source),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes),
  +source_in: {Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,index)),
    0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  Held.Both{+room,+more} = shape_ok
  Held.Both{+positive,+rank_room} = more
  Held.Both{+rooms,+row_more} = row_ok
  Held.Both{+axes_ok,+below} = row_more
  +cap = Machine.half_word(limits)
  +cap_ok = LimitBounds.half_word_fits(limits)
  +classes = chosen_word(raw_lists(rank,readings),index)
  %Equal.sym(List<&2,Tasks.Span>,Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,index),Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),index),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),
      source_box_model(readings,rank,0,source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,index,Machine.half_word(limits),LimitBounds.half_word_fits(limits),components_ok,source_in,index_in)) :
    Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      _),
    0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  source_rising(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,readings_word(readings,source),classes,offset,
    rooms,ends_at(readings,source,classes,cap,cap_ok,components_ends(readings,rank,cap,cap_ok,components_ok,index,index_in),source_in,
      components_rows(readings,rank,cap,components_ok)),axes_ok)

def source_apart_at(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},+shape_ok: ShapeOk(limits,axes),
  +readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+source: U32,+offset: U32,+first: U32,+second: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +row_ok: RowOk(limits,readings_word(readings,source),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes),
  +separate: {Tasks.separable(readings) == True{} : Bool},
  +source_in: {Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +first_in: {Nat.is_lt(U32.to_nat(first),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +second_in: {Nat.is_lt(U32.to_nat(second),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +differ: {Nat.is_eq(U32.to_nat(first),U32.to_nat(second)) == False{} : Bool}) ->
  ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,first)),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,second))):
  Held.Both{+room,+more} = shape_ok
  Held.Both{+positive,+rank_room} = more
  Held.Both{+rooms,+row_more} = row_ok
  Held.Both{+axes_ok,+below} = row_more
  +cap = Machine.half_word(limits)
  +cap_ok = LimitBounds.half_word_fits(limits)
  +rows_room = components_rows(readings,rank,cap,components_ok)
  +classes = chosen_word(raw_lists(rank,readings),first)
  +others = chosen_word(raw_lists(rank,readings),second)
  %Equal.sym(List<&2,Tasks.Span>,Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,first),Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),first),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),
      source_box_model(readings,rank,0,source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,first,Machine.half_word(limits),LimitBounds.half_word_fits(limits),components_ok,source_in,first_in)) :
    ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      _),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,second)))
  %Equal.sym(List<&2,Tasks.Span>,Tasks.source_box(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),0,True{}},source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,second),Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),second),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0),
      source_box_model(readings,rank,0,source,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,offset,second,Machine.half_word(limits),LimitBounds.half_word_fits(limits),components_ok,source_in,second_in)) :
    ListsApart(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      Tasks.boxed(readings_word(readings,source),chosen_word(raw_lists(rank,readings),first),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,GridLaws.coordinates_model(axes,offset),0)),
    Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,
      _))
  source_apart(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,readings_word(readings,source),classes,others,offset,
    first_at(readings,source,classes,others,cap,cap_ok,components_first(readings,rank,cap,cap_ok,components_ok,first,second,first_in,second_in,differ),source_in,rows_room),
    distinct_at(readings,source,cap,cap_ok,separate,source_in,rows_room),rooms,
    ends_at(readings,source,classes,cap,cap_ok,components_ends(readings,rank,cap,cap_ok,components_ok,first,first_in),source_in,rows_room),
    ends_at(readings,source,others,cap,cap_ok,components_ends(readings,rank,cap,cap_ok,components_ok,second,second_in),source_in,rows_room),axes_ok,below)

# A component index taken after steps others starts none of their spans.
def fresh_upto(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},+shape_ok: ShapeOk(limits,axes),
  +readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,+source: U32,+ranges: Pieces.Ranges,+offset: U32,+index: U32,steps: Nat,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +row_ok: RowOk(limits,readings_word(readings,source),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes),
  +separate: {Tasks.separable(readings) == True{} : Bool},
  +source_in: {Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +below: {Nat.is_le(steps,U32.to_nat(index)) == True{} : Bool}) ->
  FreshAll(gone_upto(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,
      Tensors.Origin{axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},ranges,offset},steps),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Moved.source_pieces(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,index,
      Tensors.Origin{axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},ranges,offset})):
  match steps:
    case 0n:
      fresh_ended(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
        Moved.source_pieces(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,index,
          Tensors.Origin{axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},ranges,offset}))
    case 1n+ +before:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +components = {Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}} : Tasks.Components}
      +origin = {Tensors.Origin{axes,grid,ranges,offset} : Tensors.Origin}
      +cap = Machine.half_word(limits)
      +lists = count_nat(raw_lists(rank,readings))
      +earlier = Order.succ_le_lt(before,U32.to_nat(index),below)
      +before_in = Order.lt_le(before,U32.to_nat(index),lists,earlier,lt_to_le(U32.to_nat(index),lists,index_in))
      +word = Bounds.roundtrip(before,cap,LimitBounds.half_word_fits(limits),
        Order.transitive(before,lists,U32.to_nat(cap),lt_to_le(before,lists,before_in),components_count(readings,rank,cap,components_ok)))
      +back = Equal.sym(Nat,U32.to_nat(U32.from_nat(before)),before,word)
      +other_in = {%back : {Nat.is_lt(_,lists) == True{} : Bool} before_in : {Nat.is_lt(U32.to_nat(U32.from_nat(before)),lists) == True{} : Bool}}
      +differ = {%back : {Nat.is_eq(U32.to_nat(index),_) == False{} : Bool}
        Equal.trans(Bool,Nat.is_eq(U32.to_nat(index),before),Nat.is_eq(before,U32.to_nat(index)),False{},Order.equal_symmetric(U32.to_nat(index),before),
          Order.less_not_equal(before,U32.to_nat(index),earlier)) : {Nat.is_eq(U32.to_nat(index),U32.to_nat(U32.from_nat(before))) == False{} : Bool}}
      fresh_taken(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,Moved.source_pieces(components,source,U32.from_nat(before),origin),
        gone_upto(components,source,origin,before),0n,
        source_rising_at(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok,readings,rank,source,offset,U32.from_nat(before),
          components_ok,row_ok,source_in,other_in),
        Moved.source_pieces(components,source,index,origin),0n,
        source_rising_at(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok,readings,rank,source,offset,index,
          components_ok,row_ok,source_in,index_in),
        source_apart_at(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok,readings,rank,source,offset,index,U32.from_nat(before),
          components_ok,row_ok,separate,source_in,index_in,other_in,differ),
        fresh_upto(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,shape_ok,readings,rank,tasks,source,ranges,offset,index,before,
          components_ok,row_ok,separate,source_in,index_in,lt_to_le(before,U32.to_nat(index),earlier)))

def origin_axes(+origin: Tensors.Origin) -> List<&2,U32>:
  Tensors.Origin{+shape,+grid,+ranges,+offset} = origin
  shape

# A source laid by the ranges of the grid of its shape, in order.
# A shape laid in the pieces of its grid by the grid's own ranges.
def LaidOk(+limits: Machine.Limits,+shape: List<&2,U32>,+grid: Grid.Grid,+ranges: Pieces.Ranges) -> Data:
  Held.Both<{grid == Grid.grid(shape,Grid.piece_target()) : Grid.Grid},Held.Both<{ranges == Pieces.ranges_of_grid(grid) : Pieces.Ranges},ShapeOk(limits,shape)>>

def OriginOk(+limits: Machine.Limits,+origin: Tensors.Origin) -> Data:
  match origin:
    case Tensors.Origin{+shape,+grid,+ranges,+offset}:
      LaidOk(limits,shape,grid,ranges)

# Every source from source on is laid as OriginOk says, and when the
# components are separate, has a row of readings that RowOk holds for.
def OriginsOk(+limits: Machine.Limits,origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+source: U32,+separate: Bool) -> Data:
  match origins:
    case Nil{}:
      Unit
    case Con{+origin,origins}:
      Held.Both<OriginOk(limits,origin),Held.Both<Logic.When(separate,Held.Both<{Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
        RowOk(limits,readings_word(readings,source),Moved.origin_grid(origin),origin_axes(origin))>),OriginsOk(limits,origins,readings,(source + 1 : U32),separate)>>

def shape_room_of(+limits: Machine.Limits,+shape: List<&2,U32>,+ok: ShapeOk(limits,shape)) ->
  {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Held.Both{+room,+more} = ok
  room

def shape_positive_of(+limits: Machine.Limits,+shape: List<&2,U32>,+ok: ShapeOk(limits,shape)) -> {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}:
  Held.Both{+room,+more} = ok
  Held.Both{+positive,+rank_room} = more
  positive

# The ranges of a grid are ordered.
def grid_ordered(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) ->
  Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})):
  Placed.run_ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),1n,U32.to_nat(count),
    GridRanges.grid_run(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,1n,U32.to_nat(unit_cells),
      Equal.sym(Nat,Nat.mul(U32.to_nat(unit_cells),1n),U32.to_nat(unit_cells),Order.times_one(U32.to_nat(unit_cells)))))

# The pieces component index lists of a source laid by its grid start fresh
# leaves of its ranges.
def take_grid(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,+source: U32,+axes: List<&2,U32>,+grid: Grid.Grid,
  +ranges: Pieces.Ranges,+offset: U32,+index: U32,+components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +separate: {Tasks.separable(readings) == True{} : Bool},+grid_eq: {grid == Grid.grid(axes,Grid.piece_target()) : Grid.Grid},
  +ranges_eq: {ranges == Pieces.ranges_of_grid(grid) : Pieces.Ranges},+shape_ok: ShapeOk(limits,axes),
  +source_in: {Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
  +row_ok: RowOk(limits,readings_word(readings,source),grid,axes),
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +facts: GridLaws.GridFacts<limits,Grid.grid(axes,Grid.piece_target())>) ->
  Moved.TakeFacts(Moved.source_pieces(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,index,Tensors.Origin{axes,grid,ranges,offset}),
    grid,ranges,gone_upto(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,Tensors.Origin{axes,grid,ranges,offset},U32.to_nat(index))):
  GridLaws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +whole = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +components = {Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}} : Tasks.Components}
  +target = Grid.piece_target()
  +same = Equal.trans(Grid.Grid,grid,Grid.grid(axes,target),whole,grid_eq,exact)
  +laid = Pieces.ranges_of_grid(whole)
  +same_ranges = Equal.trans(Pieces.Ranges,ranges,Pieces.ranges_of_grid(grid),laid,ranges_eq,
    Equal.cong(Grid.Grid,Pieces.Ranges,part => Pieces.ranges_of_grid(part),grid,whole,same))
  +ordered_laid = grid_ordered(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells)
  +row_laid = {%same : RowOk(limits,readings_word(readings,source),_,axes) row_ok : RowOk(limits,readings_word(readings,source),whole,axes)}
  %Equal.sym(Grid.Grid,grid,whole,same) :
    Moved.TakeFacts(Moved.source_pieces(components,source,index,Tensors.Origin{axes,_,ranges,offset}),_,ranges,
      gone_upto(components,source,Tensors.Origin{axes,_,ranges,offset},U32.to_nat(index)))
  %Equal.sym(Pieces.Ranges,ranges,laid,same_ranges) :
    Moved.TakeFacts(Moved.source_pieces(components,source,index,Tensors.Origin{axes,whole,_,offset}),whole,_,
      gone_upto(components,source,Tensors.Origin{axes,whole,_,offset},U32.to_nat(index)))
  take_facts(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered_laid,
    Moved.source_pieces(components,source,index,Tensors.Origin{axes,whole,laid,offset}),
    gone_upto(components,source,Tensors.Origin{axes,whole,laid,offset},U32.to_nat(index)),0n,
    source_rising_at(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok,readings,rank,source,offset,index,components_ok,row_laid,
      source_in,index_in),
    fresh_upto(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,shape_ok,readings,rank,tasks,source,laid,offset,index,U32.to_nat(index),
      components_ok,row_laid,separate,source_in,index_in,Order.reflexive(U32.to_nat(index))))

def take_origin(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,+source: U32,origin: Tensors.Origin,+index: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separate: {Tasks.separable(readings) == True{} : Bool},
  +origin_ok: OriginOk(limits,origin),
  +row: Held.Both<{Nat.is_lt(U32.to_nat(source),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},
    RowOk(limits,readings_word(readings,source),Moved.origin_grid(origin),origin_axes(origin))>,
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  Moved.TakeFacts(Moved.source_pieces(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,index,origin),
    Moved.origin_grid(origin),Moved.origin_ranges(origin),gone_upto(Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,origin,U32.to_nat(index))):
  match origin:
    case Tensors.Origin{+axes,+grid,+ranges,+offset}:
      Held.Both{+grid_eq,+more} = origin_ok
      Held.Both{+ranges_eq,+shape_ok} = more
      Held.Both{+source_in,+row_ok} = row
      take_grid(limits,readings,rank,tasks,source,axes,grid,ranges,offset,index,components_ok,separate,grid_eq,ranges_eq,shape_ok,source_in,row_ok,index_in,
        GridRanges.grid_facts(limits,axes,Grid.piece_target(),shape_room_of(limits,axes,shape_ok),shape_positive_of(limits,axes,shape_ok)))

# Every source gives component index the pieces no earlier component took.
def take_all(+limits: Machine.Limits,origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,+source: U32,+index: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separate: {Tasks.separable(readings) == True{} : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,source,True{}),
  +index_in: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  Moved.TakeAll(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,index,
    gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},source,U32.to_nat(index))):
  match origins:
    case Nil{}:
      Unit{}
    case Con{+origin,+rest}:
      Held.Both{+origin_ok,+more} = origins_ok
      Held.Both{+row,+later} = more
      Held.Both{take_origin(limits,readings,rank,tasks,source,origin,index,components_ok,separate,origin_ok,row,index_in),
        take_all(limits,rest,readings,rank,tasks,(source + 1 : U32),index,components_ok,separate,later,index_in)}

def one_of_test(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+test: {U32.is_le(count,1) == True{} : Bool}) -> {count == 1 : U32}:
  Arithmetic.to_nat_injective(count,1,Order.antisymmetric(U32.to_nat(count),1n,Bounds.as_nat_le(count,1,test),Order.positive_one(U32.to_nat(count),positive)))

def one_of_bound(+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(count),Bits.power(0n)) == True{} : Bool}) ->
  {count == 1 : U32}:
  Arithmetic.to_nat_injective(count,1,Order.antisymmetric(U32.to_nat(count),1n,bounded,Order.positive_one(U32.to_nat(count),positive)))

# A leaf of one component index takes the pieces of that index.
def leaf_gones(+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+first: U32,+count: U32,+one: {count == 1 : U32}) ->
  {Moved.layout_gones(Pieces.Range{first,count},origins,components,gones_upto(origins,components,0,U32.to_nat(first))) ==
    gones_upto(origins,components,0,Nat.add(U32.to_nat(first),U32.to_nat(count))) : Moved.Gones}:
  %Equal.sym(U32,count,1,one) :
    {Moved.layout_gones(Pieces.Range{first,_},origins,components,gones_upto(origins,components,0,U32.to_nat(first))) ==
      gones_upto(origins,components,0,Nat.add(U32.to_nat(first),U32.to_nat(_))) : Moved.Gones}
  %Equal.sym(Nat,Nat.add(U32.to_nat(first),1n),1n+U32.to_nat(first),Addition.successor_width(U32.to_nat(first))) :
    {Moved.gones_after(origins,components,0,first,gones_upto(origins,components,0,U32.to_nat(first))) == gones_upto(origins,components,0,_) : Moved.Gones}
  %Arithmetic.native_roundtrip(first) :
    {Moved.gones_after(origins,components,0,_,gones_upto(origins,components,0,U32.to_nat(first))) == gones_upto(origins,components,0,1n+U32.to_nat(first)) : Moved.Gones}
  after_upto(origins,components,0,U32.to_nat(first))

# The components of a tree of indices take their pieces one index after the
# other.
def ready_gones(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,depth: Nat,single: Bool,+first: U32,+count: U32,
  +test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Moved.layout_gones(Tasks.index_ranges(depth,single,first,count),origins,components,gones_upto(origins,components,0,U32.to_nat(first))) ==
    gones_upto(origins,components,0,Nat.add(U32.to_nat(first),U32.to_nat(count))) : Moved.Gones}:
  match depth single:
    case 0n +single:
      leaf_gones(origins,components,first,count,one_of_bound(count,positive,bounded))
    case 1n+ +more True{}:
      leaf_gones(origins,components,first,count,one_of_test(count,positive,test))
    case 1n+ +more False{}:
      +cap = Machine.capacity(limits)
      +start = U32.to_nat(first)
      +total = U32.to_nat(count)
      +inside = Order.transitive(total,Nat.add(start,total),U32.to_nat(cap),Order.right_le(start,total),room)
      +halves = GridRanges.halves_of(cap,LimitBounds.capacity_allocatable(limits),more,count,test,positive,bounded,inside)
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +left_count = GridRanges.halves_left_count(count,more,halves)
      +left_room = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,U32.to_nat(left),total,left_count),room)
      +first_exact = Arithmetic.add(first,left,cap,LimitBounds.capacity_fits(limits),left_room)
      +right_sum = GridRanges.split_sum(first,count,first_exact,GridRanges.halves_right_exact(count,more,halves),left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),U32.to_nat(cap),right_sum,room)
      +left_tree = Tasks.index_ranges(more,U32.is_le(left,1),first,left)
      +right_tree = Tasks.index_ranges(more,U32.is_le(right,1),(first + left : U32),right)
      %Equal.sym(Moved.Gones,Moved.layout_gones(left_tree,origins,components,gones_upto(origins,components,0,start)),
          gones_upto(origins,components,0,Nat.add(start,U32.to_nat(left))),
          ready_gones(limits,origins,components,more,U32.is_le(left,1),first,left,{==},GridRanges.halves_left_positive(count,more,halves),
            GridRanges.halves_left_bound(count,more,halves),left_room)) :
        {Moved.layout_gones(right_tree,origins,components,_) == gones_upto(origins,components,0,Nat.add(start,total)) : Moved.Gones}
      %first_exact : {Moved.layout_gones(right_tree,origins,components,gones_upto(origins,components,0,_)) == gones_upto(origins,components,0,Nat.add(start,total)) : Moved.Gones}
      %right_sum :
        {Moved.layout_gones(right_tree,origins,components,gones_upto(origins,components,0,U32.to_nat((first + left : U32)))) == gones_upto(origins,components,0,_) : Moved.Gones}
      ready_gones(limits,origins,components,more,U32.is_le(right,1),(first + left : U32),right,{==},GridRanges.halves_right_positive(count,more,halves),
        GridRanges.halves_right_bound(count,more,halves),right_room)

def first_inside(+start: Nat,+count: Nat,+total: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(start,count),total) == True{} : Bool}) -> {Nat.is_lt(start,total) == True{} : Bool}:
  Order.lt_le(start,Nat.add(start,count),total,{%Addition.zero_right(start) : {Nat.is_lt(_,Nat.add(start,count)) == True{} : Bool}
    Order.offset_lt(start,0n,count,positive) : {Nat.is_lt(start,Nat.add(start,count)) == True{} : Bool}},within)

# Every component of a tree of indices lists pieces no earlier one took.
def ready_tree(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,depth: Nat,single: Bool,
  +first: U32,+count: U32,+test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separate: {Tasks.separable(readings) == True{} : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,True{}),
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +below: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(tasks)) == True{} : Bool}) ->
  Moved.BuildReady(Tasks.index_ranges(depth,single,first,count),origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},
    gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},0,U32.to_nat(first))):
  match depth single:
    case 0n +single:
      Held.Both{take_all(limits,origins,readings,rank,tasks,0,first,components_ok,separate,origins_ok,
          first_inside(U32.to_nat(first),U32.to_nat(count),count_nat(raw_lists(rank,readings)),positive,within)),
        first_inside(U32.to_nat(first),U32.to_nat(count),U32.to_nat(tasks),positive,below)}
    case 1n+ +more True{}:
      Held.Both{take_all(limits,origins,readings,rank,tasks,0,first,components_ok,separate,origins_ok,
          first_inside(U32.to_nat(first),U32.to_nat(count),count_nat(raw_lists(rank,readings)),positive,within)),
        first_inside(U32.to_nat(first),U32.to_nat(count),U32.to_nat(tasks),positive,below)}
    case 1n+ +more False{}:
      +components = {Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}} : Tasks.Components}
      +cap = Machine.capacity(limits)
      +start = U32.to_nat(first)
      +total = U32.to_nat(count)
      +lists = count_nat(raw_lists(rank,readings))
      +inside = Order.transitive(total,Nat.add(start,total),U32.to_nat(cap),Order.right_le(start,total),room)
      +halves = GridRanges.halves_of(cap,LimitBounds.capacity_allocatable(limits),more,count,test,positive,bounded,inside)
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +left_count = GridRanges.halves_left_count(count,more,halves)
      +left_room = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,U32.to_nat(left),total,left_count),room)
      +left_within = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),lists,Order.offset_le(start,U32.to_nat(left),total,left_count),within)
      +left_below = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),U32.to_nat(tasks),Order.offset_le(start,U32.to_nat(left),total,left_count),below)
      +first_exact = Arithmetic.add(first,left,cap,LimitBounds.capacity_fits(limits),left_room)
      +right_sum = GridRanges.split_sum(first,count,first_exact,GridRanges.halves_right_exact(count,more,halves),left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),U32.to_nat(cap),right_sum,room)
      +right_within = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),lists,right_sum,within)
      +right_below = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),U32.to_nat(tasks),right_sum,below)
      +left_tree = Tasks.index_ranges(more,U32.is_le(left,1),first,left)
      +right_tree = Tasks.index_ranges(more,U32.is_le(right,1),(first + left : U32),right)
      +left_positive = GridRanges.halves_left_positive(count,more,halves)
      +left_bound = GridRanges.halves_left_bound(count,more,halves)
      Held.Both{ready_tree(limits,origins,readings,rank,tasks,more,U32.is_le(left,1),first,left,{==},left_positive,left_bound,left_room,components_ok,separate,origins_ok,
          left_within,left_below),
        {%Equal.sym(Moved.Gones,Moved.layout_gones(left_tree,origins,components,gones_upto(origins,components,0,start)),
              gones_upto(origins,components,0,Nat.add(start,U32.to_nat(left))),
              ready_gones(limits,origins,components,more,U32.is_le(left,1),first,left,{==},left_positive,left_bound,left_room)) :
            Moved.BuildReady(right_tree,origins,components,_)
          %first_exact : Moved.BuildReady(right_tree,origins,components,gones_upto(origins,components,0,_))
          ready_tree(limits,origins,readings,rank,tasks,more,U32.is_le(right,1),(first + left : U32),right,{==},GridRanges.halves_right_positive(count,more,halves),
            GridRanges.halves_right_bound(count,more,halves),right_room,components_ok,separate,origins_ok,right_within,right_below) :
          Moved.BuildReady(right_tree,origins,components,Moved.layout_gones(left_tree,origins,components,gones_upto(origins,components,0,start)))}}

# Separate components list pieces no earlier component took.
def build_ready_separate(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separate: {Tasks.separable(readings) == True{} : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,True{}),+positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},
  +within: {Nat.is_le(U32.to_nat(tasks),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  Moved.BuildReady(Tasks.component_ranges(tasks),origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},
    gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},0,0n)):
  +room = Order.transitive(U32.to_nat(tasks),count_nat(raw_lists(rank,readings)),U32.to_nat(Machine.capacity(limits)),within,
    components_within(readings,rank,limits,components_ok))
  ready_tree(limits,origins,readings,rank,tasks,32n,U32.is_le(tasks,1),0,tasks,{==},positive,
    Order.transitive(U32.to_nat(tasks),Bits.power(31n),Bits.power(32n),GridRanges.capacity_power(limits,U32.to_nat(tasks),room),Bits.power_monotone(31n,32n,{==})),
    room,components_ok,separate,origins_ok,within,Order.reflexive(U32.to_nat(tasks)))

def whole_rising_some(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,Tasks.whole_spans(axes)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  box_rising(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,Tasks.whole_spans(axes),some,
    source_total(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some),
    source_whole_within(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some))

# The pieces of a whole source rise through its pieces.
def whole_rising(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},+shape_ok: ShapeOk(limits,axes)) ->
  Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,Tasks.whole_spans(axes)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  Held.Both{+room,+more} = shape_ok
  Held.Both{+positive,+rank_room} = more
  Held.by_cases(Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,Tasks.whole_spans(axes)),0n,
      Nat.mul(U32.to_nat(outer),U32.to_nat(per))),U32.is_zero(prefix),
    none => box_rising_whole(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,Tasks.whole_spans(axes),none),
    some => whole_rising_some(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some))

# The one component of an inseparable operation takes every piece of a source.
def take_grid_whole(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+source: U32,
  +axes: List<&2,U32>,+grid: Grid.Grid,+ranges: Pieces.Ranges,+offset: U32,+index: U32,+grid_eq: {grid == Grid.grid(axes,Grid.piece_target()) : Grid.Grid},
  +ranges_eq: {ranges == Pieces.ranges_of_grid(grid) : Pieces.Ranges},+shape_ok: ShapeOk(limits,axes),
  +facts: GridLaws.GridFacts<limits,Grid.grid(axes,Grid.piece_target())>) ->
  Moved.TakeFacts(Moved.source_pieces(Tasks.Components{readings,counted_all(lists),tasks,False{}},source,index,Tensors.Origin{axes,grid,ranges,offset}),grid,ranges,Held.Ended{}):
  GridLaws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +whole = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +components = {Tasks.Components{readings,counted_all(lists),tasks,False{}} : Tasks.Components}
  +same = Equal.trans(Grid.Grid,grid,Grid.grid(axes,Grid.piece_target()),whole,grid_eq,exact)
  +laid = Pieces.ranges_of_grid(whole)
  +same_ranges = Equal.trans(Pieces.Ranges,ranges,Pieces.ranges_of_grid(grid),laid,ranges_eq,
    Equal.cong(Grid.Grid,Pieces.Ranges,part => Pieces.ranges_of_grid(part),grid,whole,same))
  +ordered_laid = grid_ordered(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells)
  %Equal.sym(Grid.Grid,grid,whole,same) :
    Moved.TakeFacts(Moved.source_pieces(components,source,index,Tensors.Origin{axes,_,ranges,offset}),_,ranges,Held.Ended{})
  %Equal.sym(Pieces.Ranges,ranges,laid,same_ranges) :
    Moved.TakeFacts(Moved.source_pieces(components,source,index,Tensors.Origin{axes,whole,_,offset}),whole,_,Held.Ended{})
  +list = Tasks.box_pieces(whole,axes,Tasks.whole_spans(axes))
  take_facts(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered_laid,list,Held.Ended{},0n,
    whole_rising(limits,axes,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok),fresh_ended(whole,list))

def take_origin_whole(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+source: U32,
  origin: Tensors.Origin,+index: U32,+origin_ok: OriginOk(limits,origin)) ->
  Moved.TakeFacts(Moved.source_pieces(Tasks.Components{readings,counted_all(lists),tasks,False{}},source,index,origin),Moved.origin_grid(origin),Moved.origin_ranges(origin),Held.Ended{}):
  match origin:
    case Tensors.Origin{+axes,+grid,+ranges,+offset}:
      Held.Both{+grid_eq,+more} = origin_ok
      Held.Both{+ranges_eq,+shape_ok} = more
      take_grid_whole(limits,readings,lists,tasks,source,axes,grid,ranges,offset,index,grid_eq,ranges_eq,shape_ok,
        GridRanges.grid_facts(limits,axes,Grid.piece_target(),shape_room_of(limits,axes,shape_ok),shape_positive_of(limits,axes,shape_ok)))

def take_all_whole(+limits: Machine.Limits,origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,
  +tasks: U32,+source: U32,+index: U32,+origins_ok: OriginsOk(limits,origins,readings,source,False{})) ->
  Moved.TakeAll(origins,Tasks.Components{readings,counted_all(lists),tasks,False{}},source,index,gones_upto(origins,Tasks.Components{readings,counted_all(lists),tasks,False{}},source,0n)):
  match origins:
    case Nil{}:
      Unit{}
    case Con{+origin,+rest}:
      Held.Both{+origin_ok,+more} = origins_ok
      Held.Both{+row,+later} = more
      Held.Both{take_origin_whole(limits,readings,lists,tasks,source,origin,index,origin_ok),
        take_all_whole(limits,rest,readings,lists,tasks,(source + 1 : U32),index,later)}

# The one component of an inseparable operation is ready to build.
def build_ready_whole(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,
  +tasks: U32,+origins_ok: OriginsOk(limits,origins,readings,0,False{}),+one: {U32.is_le(tasks,1) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool}) ->
  Moved.BuildReady(Tasks.component_ranges(tasks),origins,Tasks.Components{readings,counted_all(lists),tasks,False{}},gones_upto(origins,Tasks.Components{readings,counted_all(lists),tasks,False{}},0,0n)):
  +components = {Tasks.Components{readings,counted_all(lists),tasks,False{}} : Tasks.Components}
  %Equal.sym(Bool,U32.is_le(tasks,1),True{},one) : Moved.BuildReady(Tasks.index_ranges(32n,_,0,tasks),origins,components,gones_upto(origins,components,0,0n))
  Held.Both{take_all_whole(limits,origins,readings,lists,tasks,0,0,origins_ok),positive}

def separate_tasks(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,+tasks: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +tasks_eq: {tasks == lists_count(raw_lists(rank,readings)) : U32}) -> {U32.to_nat(tasks) == count_nat(raw_lists(rank,readings)) : Nat}:
  Held.Both{+rows,+more} = components_ok
  Held.Both{+columns,+lists} = more
  Held.Both{+lists_room,+rooms} = lists
  Held.Both{+room,+rows_room} = rooms
  +cap = Machine.half_word(limits)
  %Equal.sym(U32,tasks,lists_count(raw_lists(rank,readings)),tasks_eq) : {U32.to_nat(_) == count_nat(raw_lists(rank,readings)) : Nat}
  count_exact(raw_lists(rank,readings),cap,LimitBounds.half_word_fits(limits),lists_room,
    Order.transitive(count_nat(raw_lists(rank,readings)),Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap),
      Order.le_double(count_nat(raw_lists(rank,readings))),room))

def ready_separate(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+tasks: U32,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separate: {Tasks.separable(readings) == True{} : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,True{}),+lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +tasks_eq: {tasks == lists_count(raw_lists(rank,readings)) : U32}) ->
  Moved.BuildReady(Tasks.component_ranges(tasks),origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},
    gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,True{}},0,0n)):
  +exact = separate_tasks(readings,rank,limits,tasks,components_ok,tasks_eq)
  +back = Equal.sym(Nat,U32.to_nat(tasks),count_nat(raw_lists(rank,readings)),exact)
  build_ready_separate(limits,origins,readings,rank,tasks,components_ok,separate,origins_ok,
    {%back : {Nat.is_lt(0n,_) == True{} : Bool} lists_positive : {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool}},
    {%exact : {Nat.is_le(U32.to_nat(tasks),_) == True{} : Bool} Order.reflexive(U32.to_nat(tasks)) :
      {Nat.is_le(U32.to_nat(tasks),count_nat(raw_lists(rank,readings))) == True{} : Bool}})

# Every component of a moving operation lists pieces of every source that no
# earlier component took, each starting a fresh leaf of the source ranges.
def build_ready(+limits: Machine.Limits,+origins: List<&2,Tensors.Origin>,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,separate: Bool,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separable: {Tasks.separable(readings) == separate : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,separate),+lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  Moved.BuildReady(Tasks.component_ranges(Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1))),
    origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),
      U32.min(lists_count(raw_lists(rank,readings)),1)),separate},
    gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),
      U32.min(lists_count(raw_lists(rank,readings)),1)),separate},0,0n)):
  match separate:
    case True{}:
      ready_separate(limits,origins,readings,rank,lists_count(raw_lists(rank,readings)),components_ok,separable,origins_ok,lists_positive,{==})
    case False{}:
      +lists = raw_lists(rank,readings)
      +exact = separate_tasks(readings,rank,limits,lists_count(lists),components_ok,{==})
      build_ready_whole(limits,origins,readings,lists,U32.min(lists_count(lists),1),origins_ok,Extrema.minimum_right(lists_count(lists),1),
        {%Equal.sym(Nat,U32.to_nat(U32.min(lists_count(lists),1)),Nat.min(U32.to_nat(lists_count(lists)),1n),Extrema.minimum(lists_count(lists),1)) :
            {Nat.is_lt(0n,_) == True{} : Bool}
          NatExtrema.minimum_strict_lower(U32.to_nat(lists_count(lists)),1n,0n,
            {%Equal.sym(Nat,U32.to_nat(lists_count(lists)),count_nat(lists),exact) : {Nat.is_lt(0n,_) == True{} : Bool} lists_positive :
              {Nat.is_lt(0n,U32.to_nat(lists_count(lists))) == True{} : Bool}},{==}) :
          {Nat.is_lt(0n,U32.to_nat(U32.min(lists_count(lists),1))) == True{} : Bool}})

# ---- A piece inside a box is listed ----

# Every row-major coordinate of index over the extents lies in its span.
def InBox(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat) -> Data:
  match spans extents:
    case Con{Tasks.Span{+first,+count},spans} Con{+extent,extents}:
      Held.Both<{Nat.is_le(U32.to_nat(first),Nat.div(index,ShapeLaws.elements(extents))) == True{} : Bool},
        Held.Both<{Nat.is_lt(Nat.div(index,ShapeLaws.elements(extents)),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},
          InBox(spans,extents,Nat.mod(index,ShapeLaws.elements(extents)))>>
    case Nil{} Nil{}:
      Unit
    case spans extents:
      Empty

# The position in the box of an index inside it.
def box_position(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat) -> Nat:
  match spans extents:
    case Con{Tasks.Span{+first,+count},+spans} Con{+extent,+extents}:
      Nat.add(Nat.mul(Nat.sub(Nat.div(index,ShapeLaws.elements(extents)),U32.to_nat(first)),span_total(spans)),
        box_position(spans,extents,Nat.mod(index,ShapeLaws.elements(extents))))
    case spans extents:
      0n

def box_position_below(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat,+inside: InBox(spans,extents,index),
  +below: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) -> {Nat.is_lt(box_position(spans,extents,index),span_total(spans)) == True{} : Bool}:
  match spans extents:
    case Con{Tasks.Span{+first,+count},+spans} Con{+extent,+extents}:
      Held.Both{+low,+more} = inside
      Held.Both{+high,+later} = more
      +inner = ShapeLaws.elements(extents)
      +positive = inner_positive(U32.to_nat(extent),ShapeLaws.elements(extents),index,below)
      RowMajor.radix(Nat.sub(Nat.div(index,inner),U32.to_nat(first)),U32.to_nat(count),box_position(spans,extents,Nat.mod(index,inner)),span_total(spans),
        Order.sub_lt(Nat.div(index,inner),U32.to_nat(first),U32.to_nat(count),low,high),
        box_position_below(spans,extents,Nat.mod(index,inner),later,DivisionOrder.remainder_small(index,inner,positive)))
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+extent,+extents}:
      Empty.absurd({Nat.is_lt(box_position(Nil{},Con{extent,extents},index),span_total(Nil{})) == True{} : Bool},inside)
    case Con{+span,+spans} Nil{}:
      Empty.absurd({Nat.is_lt(box_position(Con{span,spans},Nil{},index),span_total(Con{span,spans})) == True{} : Bool},inside)

def only_index(+value: Nat,index: Nat,+below: {Nat.is_lt(index,1n) == True{} : Bool}) -> {value == Nat.add(value,index) : Nat}:
  match index:
    case 0n:
      Equal.sym(Nat,Nat.add(value,0n),value,Addition.zero_right(value))
    case 1n+ +more:
      Empty.absurd({value == Nat.add(value,1n+more) : Nat},Order.below_zero(more,below))

# The row-major value at the position of an index inside a box is the index.
def box_value_step(+value: Nat,+extent: Nat,+quotient: Nat,+inner: Nat,+rest: Nat,+result: Nat,
  +later: {result == Nat.add(Nat.mul(Nat.add(Nat.mul(value,extent),quotient),inner),rest) : Nat}) ->
  {result == Nat.add(Nat.mul(value,Nat.mul(extent,inner)),Nat.add(Nat.mul(quotient,inner),rest)) : Nat}:
  %Equal.sym(Nat,result,Nat.add(Nat.mul(Nat.add(Nat.mul(value,extent),quotient),inner),rest),later) :
    {_ == Nat.add(Nat.mul(value,Nat.mul(extent,inner)),Nat.add(Nat.mul(quotient,inner),rest)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(value,extent),quotient),inner),Nat.add(Nat.mul(Nat.mul(value,extent),inner),Nat.mul(quotient,inner)),
      RowMajor.distrib(Nat.mul(value,extent),quotient,inner)) :
    {Nat.add(_,rest) == Nat.add(Nat.mul(value,Nat.mul(extent,inner)),Nat.add(Nat.mul(quotient,inner),rest)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(value,extent),inner),Nat.mul(value,Nat.mul(extent,inner)),RowMajor.assoc(value,extent,inner)) :
    {Nat.add(Nat.add(_,Nat.mul(quotient,inner)),rest) == Nat.add(Nat.mul(value,Nat.mul(extent,inner)),Nat.add(Nat.mul(quotient,inner),rest)) : Nat}
  Addition.associative(Nat.mul(value,Nat.mul(extent,inner)),Nat.mul(quotient,inner),rest)

def box_value(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat,+value: Nat,+inside: InBox(spans,extents,index),
  +below: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {row_value(spans,extents,box_position(spans,extents,index),value) == Nat.add(Nat.mul(value,ShapeLaws.elements(extents)),index) : Nat}:
  match spans extents:
    case Con{Tasks.Span{+first,+count},+spans} Con{+extent,+extents}:
      Held.Both{+low,+more} = inside
      Held.Both{+high,+later} = more
      +inner = ShapeLaws.elements(extents)
      +positive = inner_positive(U32.to_nat(extent),ShapeLaws.elements(extents),index,below)
      +quotient = Nat.div(index,inner)
      +rest = Nat.mod(index,inner)
      +offset = Nat.sub(quotient,U32.to_nat(first))
      +position = box_position(spans,extents,rest)
      +total = span_total(spans)
      +position_below = box_position_below(spans,extents,rest,later,DivisionOrder.remainder_small(index,inner,positive))
      +total_positive = Order.le_lt_trans(0n,position,total,Order.zero_le(position),position_below)
      +start = Nat.add(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(first))
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(offset,total),position),total),offset,Radix.quotient(offset,position,total,total_positive,position_below)) :
        {row_value(spans,extents,Nat.mod(Nat.add(Nat.mul(offset,total),position),total),Nat.add(start,_)) ==
          Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),index) : Nat}
      %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(offset,total),position),total),position,Radix.remainder(offset,position,total,total_positive,position_below)) :
        {row_value(spans,extents,_,Nat.add(start,offset)) == Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),index) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(first)),offset),Nat.add(Nat.mul(value,U32.to_nat(extent)),Nat.add(U32.to_nat(first),offset)),
          Addition.associative(Nat.mul(value,U32.to_nat(extent)),U32.to_nat(first),offset)) :
        {row_value(spans,extents,position,_) == Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),index) : Nat}
      %Equal.sym(Nat,Nat.add(U32.to_nat(first),offset),quotient,Order.restore_shift(U32.to_nat(first),quotient,low)) :
        {row_value(spans,extents,position,Nat.add(Nat.mul(value,U32.to_nat(extent)),_)) == Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),index) : Nat}
      %Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,inner),rest),DivisionOrder.quotient_exact(index,inner,positive)) :
        {row_value(spans,extents,position,Nat.add(Nat.mul(value,U32.to_nat(extent)),quotient)) == Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),_) : Nat}
      box_value_step(value,U32.to_nat(extent),quotient,inner,rest,row_value(spans,extents,position,Nat.add(Nat.mul(value,U32.to_nat(extent)),quotient)),
        box_value(spans,extents,rest,Nat.add(Nat.mul(value,U32.to_nat(extent)),quotient),later,DivisionOrder.remainder_small(index,inner,positive)))
    case Nil{} Nil{}:
      %Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value)) : {value == Nat.add(_,index) : Nat}
      only_index(value,index,below)
    case Nil{} Con{+extent,+extents}:
      Empty.absurd({row_value(Nil{},Con{extent,extents},box_position(Nil{},Con{extent,extents},index),value) ==
        Nat.add(Nat.mul(value,ShapeLaws.elements(Con{extent,extents})),index) : Nat},inside)
    case Con{+span,+spans} Nil{}:
      Empty.absurd({row_value(Con{span,spans},Nil{},box_position(Con{span,spans},Nil{},index),value) == Nat.add(Nat.mul(value,ShapeLaws.elements(Nil{})),index) : Nat},inside)

def listed_here(+piece: U32,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+position: Nat,+target: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(piece) == row_value(spans,extents,from,0n) : Nat},
  +value: {row_value(spans,extents,position,0n) == target : Nat},+room: {Nat.is_le(target,U32.to_nat(cap)) == True{} : Bool},
  +same: {Nat.is_eq(position,from) == True{} : Bool}) -> {piece == U32.from_nat(target) : U32}:
  +equal = Order.equal_of_is_eq(position,from,same)
  +reached = {%equal : {row_value(spans,extents,_,0n) == target : Nat} value : {row_value(spans,extents,from,0n) == target : Nat}}
  Arithmetic.to_nat_injective(piece,U32.from_nat(target),Equal.trans(Nat,U32.to_nat(piece),target,U32.to_nat(U32.from_nat(target)),
    Equal.trans(Nat,U32.to_nat(piece),row_value(spans,extents,from,0n),target,exact,reached),
    Equal.sym(Nat,U32.to_nat(U32.from_nat(target)),target,Bounds.roundtrip(target,cap,cap_ok,room))))

def listed_later_member(+piece: U32,rest: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+position: Nat,+target: Nat,
  +after: {Nat.is_le(from,position) == True{} : Bool},+other: {Nat.is_eq(position,from) == False{} : Bool},
  recurse: {Nat.is_le(1n+from,position) == True{} : Bool} -> Moved.Member(rest,U32.from_nat(target))) -> Moved.Member(Con{piece,rest},U32.from_nat(target)):
  Placed.OnRight{recurse(Order.strict_follow(from,position,strict_of(from,position,after,other)))}

# A position listed from from on is a member of the list with its row-major
# value.
def listed_member(list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+listed: ListedFrom(list,spans,extents,from),+position: Nat,
  +after: {Nat.is_le(from,position) == True{} : Bool},+before: {Nat.is_lt(position,Nat.add(from,List.length(&2,U32,list))) == True{} : Bool},+target: Nat,
  +value: {row_value(spans,extents,position,0n) == target : Nat},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(target,U32.to_nat(cap)) == True{} : Bool}) -> Moved.Member(list,U32.from_nat(target)):
  match list:
    case Nil{}:
      Empty.absurd(Moved.Member(Nil{},U32.from_nat(target)),Held.lt_le_absurd(position,from,
        {%Addition.zero_right(from) : {Nat.is_lt(position,_) == True{} : Bool} before : {Nat.is_lt(position,from) == True{} : Bool}},after))
    case Con{+piece,+rest}:
      Held.Both{+exact,+later} = listed
      Held.by_cases(Moved.Member(Con{piece,rest},U32.from_nat(target)),Nat.is_eq(position,from),
        same => Placed.OnLeft{listed_here(piece,spans,extents,from,position,target,cap,cap_ok,exact,value,room,same)},
        other => listed_later_member(piece,rest,spans,extents,from,position,target,after,other,
          next => listed_member(rest,spans,extents,1n+from,later,position,next,
            {%Addition.successor_right(from,List.length(&2,U32,rest)) : {Nat.is_lt(position,_) == True{} : Bool} before :
              {Nat.is_lt(position,Nat.add(1n+from,List.length(&2,U32,rest))) == True{} : Bool}},target,value,cap,cap_ok,room)))

# A member of a list of target pieces starts one of their spans.
def in_targets(list: List<&2,U32>,+grid: Grid.Grid,+rest: Held.Spans,+piece: U32,+member: Moved.Member(list,piece)) ->
  Placed.Listed(Moved.target_spans(list,grid,rest),Grid.start(grid,piece)):
  match list:
    case Nil{}:
      Empty.absurd(Placed.Listed(Moved.target_spans(Nil{},grid,rest),Grid.start(grid,piece)),member)
    case Con{+first,Nil{}}:
      match member:
        case Placed.OnLeft{+same}:
          Placed.OnLeft{Equal.cong(U32,U32,cell => Grid.start(grid,cell),first,piece,same)}
        case Placed.OnRight{+none}:
          Empty.absurd(Placed.Listed(Moved.target_spans(Con{first,Nil{}},grid,rest),Grid.start(grid,piece)),none)
    case Con{+first,Con{+next,+more}}:
      match member:
        case Placed.OnLeft{+same}:
          Placed.OnLeft{Equal.cong(U32,U32,cell => Grid.start(grid,cell),first,piece,same)}
        case Placed.OnRight{+later}:
          Placed.OnRight{in_targets(Con{next,more},grid,rest,piece,later)}

def member_listed(+list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+listed: BoxListed(list,spans,extents),
  +length: {List.length(&2,U32,list) == span_total(spans) : Nat},+within: SpansIn(spans,extents),+index: Nat,+inside: InBox(spans,extents,index),
  +below: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) -> Moved.Member(list,U32.from_nat(index)):
  Held.Both{+from,+bounded} = listed
  +position = box_position(spans,extents,index)
  listed_member(list,spans,extents,0n,from,position,Order.zero_le(position),
    {%Equal.sym(Nat,List.length(&2,U32,list),span_total(spans),length) : {Nat.is_lt(position,_) == True{} : Bool}
      box_position_below(spans,extents,index,inside,below) : {Nat.is_lt(position,List.length(&2,U32,list)) == True{} : Bool}},
    index,box_value(spans,extents,index,0n,inside,below),cap,cap_ok,room)

# A piece inside the block spans of a grid with prefix axes is one of the
# pieces they list.
def box_member(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+axes: List<&2,U32>,+spans: List<&2,Tasks.Span>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),+index: Nat,
  +inside: InBox(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),index),
  +below: {Nat.is_lt(index,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,spans),U32.from_nat(index)):
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +half = Machine.half_word(limits)
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})
  +index_room = Order.transitive(index,total,U32.to_nat(half),lt_to_le(index,total,below),pieces_half(limits,outer,extent,unit_cells,block,per,shape))
  %Equal.sym(Bool,U32.is_zero(prefix),False{},some) :
    Moved.Member(Bool.pick(List<&2,U32>,_,Con{0,Nil{}},listed),U32.from_nat(index))
  member_listed(listed,spans,extents,pieces_box_listed(limits,outer,extent,unit_cells,block,per,shape,axes,prefix,spans,total_exact,within),
    listed_length(spans,extents,half,LimitBounds.half_word_fits(limits),spans_half(limits,outer,extent,unit_cells,block,per,shape,axes,prefix,spans,total_exact,within)),
    within,index,inside,{%Equal.sym(Nat,ShapeLaws.elements(extents),total,total_exact) : {Nat.is_lt(index,_) == True{} : Bool} below :
      {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}},half,LimitBounds.half_word_fits(limits),index_room)

# ---- Every target block lies in a class ----

def class_stop(+class: Tasks.Class) -> Nat:
  Tasks.Class{+first,+count,+source_first,+source_count} = class
  Nat.add(U32.to_nat(first),U32.to_nat(count))

# A class of the list that holds value, found at least low positions on.
type ClassFor<-classes: List<&2,Tasks.Class>,-value: Nat,-low: Nat> is Data:
  ClassFor{index: U32,inside: {Nat.is_lt(U32.to_nat(index),List.length(&2,Tasks.Class,classes)) == True{} : Bool},
    after: {Nat.is_le(U32.to_nat(class_first(class_word(classes,index))),value) == True{} : Bool},
    before: {Nat.is_lt(value,class_stop(class_word(classes,index))) == True{} : Bool},
    early: {Nat.is_le(Nat.add(low,U32.to_nat(index)),value) == True{} : Bool}}

def class_next(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(1n+U32.to_nat(index),U32.to_nat(cap)) == True{} : Bool}) ->
  {class_word(Con{class,rest},(index + 1 : U32)) == class_word(rest,index) : Tasks.Class}:
  +sum = next_word(index,U32.to_nat(index),cap,cap_ok,{==},room)
  +nonzero = Arithmetic.word_positive_nonzero((index + 1 : U32),{%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),sum) : {Nat.is_lt(0n,_) == True{} : Bool}
    {==} : {Nat.is_lt(0n,U32.to_nat((index + 1 : U32))) == True{} : Bool}})
  %Equal.sym(Bool,U32.is_zero((index + 1 : U32)),False{},nonzero) :
    {Bool.pick(Tasks.Class,_,class,class_word(rest,((index + 1 : U32) - 1 : U32))) == class_word(rest,index) : Tasks.Class}
  %Equal.sym(U32,((index + 1 : U32) - 1 : U32),index,index_back(index)) :
    {Bool.pick(Tasks.Class,False{},class,class_word(rest,_)) == class_word(rest,index) : Tasks.Class}
  {==}

# The class a later class list holds value in, one position further.
def class_later(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+value: Nat,+low: Nat,+start: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_start: {Nat.is_le(low,start) == True{} : Bool},+room: {Nat.is_lt(value,U32.to_nat(cap)) == True{} : Bool},+found: ClassFor<rest,value,1n+start>) ->
  ClassFor<Con{class,rest},value,low>:
  ClassFor{+index,+inside,+after,+before,+early} = found
  +index_room = Order.transitive(1n+U32.to_nat(index),Nat.add(1n+start,U32.to_nat(index)),U32.to_nat(cap),
    Order.right_le(start,U32.to_nat(index)),
    Order.transitive(Nat.add(1n+start,U32.to_nat(index)),value,U32.to_nat(cap),early,lt_to_le(value,U32.to_nat(cap),room)))
  +same = class_next(class,rest,index,cap,cap_ok,index_room)
  +next = next_word(index,U32.to_nat(index),cap,cap_ok,{==},index_room)
  +back = Equal.sym(Tasks.Class,class_word(Con{class,rest},(index + 1 : U32)),class_word(rest,index),same)
  ClassFor{(index + 1 : U32),
    {%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_lt(_,1n+List.length(&2,Tasks.Class,rest)) == True{} : Bool}
      inside : {Nat.is_lt(U32.to_nat((index + 1 : U32)),List.length(&2,Tasks.Class,Con{class,rest})) == True{} : Bool}},
    {%back : {Nat.is_le(U32.to_nat(class_first(_)),value) == True{} : Bool} after : {Nat.is_le(U32.to_nat(class_first(class_word(Con{class,rest},(index + 1 : U32)))),value) == True{} : Bool}},
    {%back : {Nat.is_lt(value,class_stop(_)) == True{} : Bool} before : {Nat.is_lt(value,class_stop(class_word(Con{class,rest},(index + 1 : U32)))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((index + 1 : U32)),1n+U32.to_nat(index),next) : {Nat.is_le(Nat.add(low,_),value) == True{} : Bool}
      Order.transitive(Nat.add(low,1n+U32.to_nat(index)),Nat.add(1n+start,U32.to_nat(index)),value,
        {%Equal.sym(Nat,Nat.add(low,1n+U32.to_nat(index)),1n+Nat.add(low,U32.to_nat(index)),Addition.successor_right(low,U32.to_nat(index))) :
            {Nat.is_le(_,Nat.add(1n+start,U32.to_nat(index))) == True{} : Bool}
          Order.add_le(low,start,U32.to_nat(index),low_start) : {Nat.is_le(Nat.add(low,1n+U32.to_nat(index)),Nat.add(1n+start,U32.to_nat(index))) == True{} : Bool}},
        early) : {Nat.is_le(Nat.add(low,U32.to_nat((index + 1 : U32))),value) == True{} : Bool}}}

# The first start lies at or below value.
def HeadUpTo(list: List<&2,U32>,+value: Nat) -> Data:
  match list:
    case Nil{}:
      Empty
    case Con{+first,rest}:
      {Nat.is_le(U32.to_nat(first),value) == True{} : Bool}

def next_above(+next: U32,+rest: List<&2,U32>,+low: Nat,+limit: Nat,+rising: Rising(Con{next,rest},low,limit)) -> {Nat.is_le(low,U32.to_nat(next)) == True{} : Bool}:
  Held.Both{+above,+more} = rising
  above

def next_below(+next: U32,+rest: List<&2,U32>,+low: Nat,+limit: Nat,+rising: Rising(Con{next,rest},low,limit)) -> {Nat.is_lt(U32.to_nat(next),limit) == True{} : Bool}:
  Held.Both{+above,+more} = rising
  Held.Both{+below,+later} = more
  below

def early_here(+low: Nat,+start: Nat,+value: Nat,+low_start: {Nat.is_le(low,start) == True{} : Bool},+head: {Nat.is_le(start,value) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(low,U32.to_nat(0)),value) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(low,0n),low,Addition.zero_right(low)) : {Nat.is_le(_,value) == True{} : Bool}
  Order.transitive(low,start,value,low_start,head)

def run_ends(+start: U32,+next: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(start),U32.to_nat(next)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(next),U32.to_nat(cap)) == True{} : Bool},
  +below: {Nat.is_lt(value,U32.to_nat(next)) == True{} : Bool}) -> {Nat.is_lt(value,Nat.add(U32.to_nat(start),U32.to_nat((next - start : U32)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((next - start : U32)),Nat.sub(U32.to_nat(next),U32.to_nat(start)),
      Arithmetic.difference_exact(cap,cap_ok,next,start,U32.to_nat(next),U32.to_nat(start),{==},{==},ordered,room)) :
    {Nat.is_lt(value,Nat.add(U32.to_nat(start),_)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(next),U32.to_nat(start))),U32.to_nat(next),Order.restore_shift(U32.to_nat(start),U32.to_nat(next),ordered)) :
    {Nat.is_lt(value,_) == True{} : Bool}
  below

def runs_here(+start: U32,+next: U32,+rest: List<&2,U32>,+blocks: U32,+value: Nat,+low: Nat,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low_start: {Nat.is_le(low,U32.to_nat(start)) == True{} : Bool},+head: {Nat.is_le(U32.to_nat(start),value) == True{} : Bool},
  +later: Rising(Con{next,rest},1n+U32.to_nat(start),limit),+room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(value,U32.to_nat(next)) == True{} : Bool}) -> ClassFor<Tasks.runs(Con{start,Con{next,rest}},blocks),value,low>:
  +ordered = lt_to_le(U32.to_nat(start),U32.to_nat(next),Order.succ_le_lt(U32.to_nat(start),U32.to_nat(next),next_above(next,rest,1n+U32.to_nat(start),limit,later)))
  +next_room = Order.transitive(U32.to_nat(next),limit,U32.to_nat(cap),lt_to_le(U32.to_nat(next),limit,next_below(next,rest,1n+U32.to_nat(start),limit,later)),room)
  ClassFor{0,{==},head,run_ends(start,next,value,cap,cap_ok,ordered,next_room,inside),early_here(low,U32.to_nat(start),value,low_start,head)}

# Runs from a start at or below value to blocks hold value in one of them.
def runs_cover(starts: List<&2,U32>,+blocks: U32,+value: Nat,+low: Nat,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rising: Rising(starts,low,limit),+head: HeadUpTo(starts,value),+blocks_value: {U32.to_nat(blocks) == limit : Nat},+below: {Nat.is_lt(value,limit) == True{} : Bool},
  +room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool}) -> ClassFor<Tasks.runs(starts,blocks),value,low>:
  match starts:
    case Nil{}:
      Empty.absurd(ClassFor<Tasks.runs(Nil{},blocks),value,low>,head)
    case Con{+start,Nil{}}:
      Held.Both{+low_start,+more} = rising
      Held.Both{+start_in,+later} = more
      +ordered = lt_to_le(U32.to_nat(start),limit,start_in)
      ClassFor{0,{==},head,
        {%Equal.sym(Nat,U32.to_nat((blocks - start : U32)),Nat.sub(limit,U32.to_nat(start)),
            Arithmetic.difference_exact(cap,cap_ok,blocks,start,limit,U32.to_nat(start),blocks_value,{==},ordered,room)) :
            {Nat.is_lt(value,Nat.add(U32.to_nat(start),_)) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(U32.to_nat(start),Nat.sub(limit,U32.to_nat(start))),limit,Order.restore_shift(U32.to_nat(start),limit,ordered)) :
            {Nat.is_lt(value,_) == True{} : Bool}
          below : {Nat.is_lt(value,Nat.add(U32.to_nat(start),U32.to_nat((blocks - start : U32)))) == True{} : Bool}},
        early_here(low,U32.to_nat(start),value,low_start,head)}
    case Con{+start,Con{+next,+rest}}:
      Held.Both{+low_start,+more} = rising
      Held.Both{+start_in,+later} = more
      Held.by_cases(ClassFor<Tasks.runs(Con{start,Con{next,rest}},blocks),value,low>,Nat.is_lt(value,U32.to_nat(next)),
        inside => runs_here(start,next,rest,blocks,value,low,limit,cap,cap_ok,low_start,head,later,room,inside),
        outside => class_later(Tasks.Class{start,(next - start : U32),0,0},Tasks.runs(Con{next,rest},blocks),value,low,U32.to_nat(start),cap,cap_ok,low_start,
          Order.lt_le(value,limit,U32.to_nat(cap),below,room),
          runs_cover(Con{next,rest},blocks,value,1n+U32.to_nat(start),limit,cap,cap_ok,later,Order.not_less(value,U32.to_nat(next),outside),blocks_value,below,room)))

def column_limit(column: List<&2,Tasks.Reading>) -> Nat:
  match column:
    case Nil{}:
      0n
    case Con{Tasks.Reading{+extent,+block,source},others}:
      NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))

def head_up(starts: List<&2,U32>,+value: Nat,+limit: Nat,+head: HeadAt(starts,limit),+below: {Nat.is_lt(value,limit) == True{} : Bool}) -> HeadUpTo(starts,value):
  match starts:
    case Nil{}:
      Order.below_zero(value,{%head : {Nat.is_lt(value,_) == True{} : Bool} below : {Nat.is_lt(value,0n) == True{} : Bool}})
    case Con{+first,+rest}:
      %Equal.sym(U32,first,0,head) : {Nat.is_le(U32.to_nat(_),value) == True{} : Bool}
      Order.zero_le(value)

def cover_starts(+starts: List<&2,U32>,+blocks: U32,+value: Nat,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +facts: Starts(starts,limit),+blocks_value: {U32.to_nat(blocks) == limit : Nat},+below: {Nat.is_lt(value,limit) == True{} : Bool},
  +room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool}) -> ClassFor<Tasks.runs(starts,blocks),value,0n>:
  Held.Both{+head,+rising} = facts
  runs_cover(starts,blocks,value,0n,limit,cap,cap_ok,rising,head_up(starts,value,limit,head,below),blocks_value,below,room)

# Every block of a target axis lies in one of its classes.
def column_cover(column: List<&2,Tasks.Reading>,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ok: ColumnOk(column,cap),
  +below: {Nat.is_lt(value,column_limit(column)) == True{} : Bool}) -> ClassFor<Tasks.axis_classes(column),value,0n>:
  match column:
    case Nil{}:
      Empty.absurd(ClassFor<Tasks.axis_classes(Nil{}),value,0n>,Order.below_zero(value,below))
    case Con{Tasks.Reading{+extent,+block,+source},+others}:
      Held.Both{+rooms,+same} = ok
      Held.Both{+head_room,+other_rooms} = rooms
      Held.Both{+axis_room,+read_room} = head_room
      Held.Both{+blocks_value,+axis_more} = axis_room
      Held.Both{+limit_room,+block_more} = axis_more
      +limit = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
      +starts = Tasks.shared_starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,source})),Tasks.starts_of(others))
      cover_starts(starts,Tasks.target_blocks(extent,block),value,limit,cap,cap_ok,axis_starts(extent,block,source,others,cap,cap_ok,rooms,same),blocks_value,below,limit_room)

# ---- The component holding each target piece ----

# The readings of every target axis count the blocks the grid gives it.
def LimitsAt(+readings: List<&2,List<&2,Tasks.Reading>>,rank: Nat,cells: List<&2,U32>,+grid: Grid.Grid,+at: U32) -> Data:
  match rank cells:
    case 1n+rest Con{+extent,more}:
      Held.Both<{column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))) : Nat},
        LimitsAt(Tasks.tails(readings),rest,more,grid,(at + 1 : U32))>
    case 0n Nil{}:
      Unit
    case rank cells:
      Empty

# A component of the class lists whose target spans from axis at hold index.
type Chosen<-lists: List<&2,List<&2,Tasks.Class>>,-prefix: U32,-at: U32,-extents: List<&2,U32>,-index: Nat> is Data:
  Chosen{component: U32,inside: {Nat.is_lt(U32.to_nat(component),count_nat(lists)) == True{} : Bool},
    box: InBox(Tasks.target_spans(chosen_word(lists,component),prefix,at),extents,index)}

def at_kept(+at: U32,+prefix: U32,+position: Nat,+remaining: Nat,+exact: {U32.to_nat(at) == position : Nat},
  +distance: {Nat.add(position,1n+remaining) == U32.to_nat(prefix) : Nat}) -> {U32.is_lt(at,prefix) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(at,prefix),Nat.is_lt(U32.to_nat(at),U32.to_nat(prefix)),True{},Comparison.lt(at,prefix),
    {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_lt(_,U32.to_nat(prefix)) == True{} : Bool}
      %distance : {Nat.is_lt(position,_) == True{} : Bool}
      below_plus(position,1n+remaining,{==}) : {Nat.is_lt(U32.to_nat(at),U32.to_nat(prefix)) == True{} : Bool}})

def spans_end(classes: List<&2,Tasks.Class>,+prefix: U32,+at: U32,+stop: {U32.is_lt(at,prefix) == False{} : Bool}) ->
  {Tasks.target_spans(classes,prefix,at) == Nil{} : List<&2,Tasks.Span>}:
  match classes:
    case Nil{}:
      {==}
    case Con{Tasks.Class{+first,+count,+source_first,+source_count},+rest}:
      %Equal.sym(Bool,U32.is_lt(at,prefix),False{},stop) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(rest,prefix,(at + 1 : U32))},Nil{}) == Nil{} : List<&2,Tasks.Span>}
      {==}

# Past the prefix axes any component holds the index.
def chosen_beyond(+lists: List<&2,List<&2,Tasks.Class>>,+prefix: U32,+at: U32,+index: Nat,+stop: {U32.is_lt(at,prefix) == False{} : Bool},
  +positive: {Nat.is_lt(0n,count_nat(lists)) == True{} : Bool}) -> Chosen<lists,prefix,at,Nil{},index>:
  Chosen{0,positive,
    {%Equal.sym(List<&2,Tasks.Span>,Tasks.target_spans(chosen_word(lists,0),prefix,at),Nil{},spans_end(chosen_word(lists,0),prefix,at,stop)) :
        InBox(_,Nil{},index)
      Unit{} : InBox(Tasks.target_spans(chosen_word(lists,0),prefix,at),Nil{},index)}}

# A class holding the leading coordinate of index spans it.
def class_box(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+prefix: U32,+at: U32,+extent: U32,+extents: List<&2,U32>,+index: Nat,
  +kept: {U32.is_lt(at,prefix) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(class_first(class)),Nat.div(index,ShapeLaws.elements(extents))) == True{} : Bool},
  +before: {Nat.is_lt(Nat.div(index,ShapeLaws.elements(extents)),class_stop(class)) == True{} : Bool},
  +later: InBox(Tasks.target_spans(rest,prefix,(at + 1 : U32)),extents,Nat.mod(index,ShapeLaws.elements(extents)))) ->
  InBox(Tasks.target_spans(Con{class,rest},prefix,at),Con{extent,extents},index):
  match class:
    case Tasks.Class{+first,+count,+source_first,+source_count}:
      %Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept) :
        InBox(Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(rest,prefix,(at + 1 : U32))},Nil{}),Con{extent,extents},index)
      Held.Both{after,Held.Both{before,later}}

# The class of the leading coordinate and the component of the rest make one
# component, numbered with the last axis fastest.
def chosen_step(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+prefix: U32,+at: U32,+extent: U32,+extents: List<&2,U32>,+index: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+kept: {U32.is_lt(at,prefix) == True{} : Bool},+rest_room: ListsRoom(lists,cap),
  +room: {Nat.is_le(Nat.double(count_nat(Con{list,lists})),U32.to_nat(cap)) == True{} : Bool},
  +found: ClassFor<list,Nat.div(index,ShapeLaws.elements(extents)),0n>,
  +later: Chosen<lists,prefix,(at + 1 : U32),extents,Nat.mod(index,ShapeLaws.elements(extents))>) ->
  Chosen<Con{list,lists},prefix,at,Con{extent,extents},index>:
  ClassFor{+row,+row_inside,+after,+before,+early} = found
  Chosen{+column,+column_inside,+box} = later
  +length = List.length(&2,Tasks.Class,list)
  +divisor = count_nat(lists)
  +total = Nat.mul(length,divisor)
  +inner = inner_of(lists)
  +low = Nat.mul(U32.to_nat(row),divisor)
  +value = Nat.add(low,U32.to_nat(column))
  +positive = Order.le_lt_trans(0n,U32.to_nat(column),divisor,Order.zero_le(U32.to_nat(column)),column_inside)
  +length_positive = Order.le_lt_trans(0n,U32.to_nat(row),length,Order.zero_le(U32.to_nat(row)),row_inside)
  +value_inside = RowMajor.radix(U32.to_nat(row),length,U32.to_nat(column),divisor,row_inside,column_inside)
  +total_room = Order.transitive(total,Nat.double(total),U32.to_nat(cap),Order.le_double(total),room)
  +divisor_le = {%RowMajor.multiply_commutative(divisor,length) : {Nat.is_le(divisor,_) == True{} : Bool}
    RowMajor.product_covers(divisor,length,length_positive) : {Nat.is_le(divisor,total) == True{} : Bool}}
  +divisor_room = Order.transitive(divisor,total,U32.to_nat(cap),divisor_le,total_room)
  +double_room = Order.transitive(Nat.double(divisor),Nat.double(total),U32.to_nat(cap),Order.double_le(divisor,total,divisor_le),room)
  +inner_value = Extrema.max_one_exact(lists_count(lists),divisor,count_exact(lists,cap,cap_ok,rest_room,divisor_room),positive)
  +value_room = Order.transitive(value,total,U32.to_nat(cap),lt_to_le(value,total,value_inside),total_room)
  +product = Arithmetic.product_exact(cap,cap_ok,row,inner,U32.to_nat(row),divisor,{==},inner_value,
    Order.transitive(low,value,U32.to_nat(cap),Order.le_add(low,U32.to_nat(column)),value_room))
  +component = ((row * inner : U32) + column : U32)
  +sum = Arithmetic.sum_exact(cap,cap_ok,(row * inner : U32),column,low,U32.to_nat(column),product,{==},value_room)
  +row_same = Arithmetic.to_nat_injective((component / inner : U32),row,
    Radix.quotient_word(component,inner,U32.to_nat(row),U32.to_nat(column),divisor,cap,cap_ok,sum,inner_value,positive,column_inside,double_room))
  +column_same = Arithmetic.to_nat_injective((component % inner : U32),column,
    Radix.remainder_word(component,inner,U32.to_nat(row),U32.to_nat(column),divisor,cap,cap_ok,sum,inner_value,positive,column_inside,double_room))
  Chosen{component,
    {%Equal.sym(Nat,U32.to_nat(component),value,sum) : {Nat.is_lt(_,total) == True{} : Bool}
      value_inside : {Nat.is_lt(U32.to_nat(component),count_nat(Con{list,lists})) == True{} : Bool}},
    {%Equal.sym(U32,(component / inner : U32),row,row_same) :
        InBox(Tasks.target_spans(Con{class_word(list,_),chosen_word(lists,(component % inner : U32))},prefix,at),Con{extent,extents},index)
      %Equal.sym(U32,(component % inner : U32),column,column_same) :
        InBox(Tasks.target_spans(Con{class_word(list,row),chosen_word(lists,_)},prefix,at),Con{extent,extents},index)
      class_box(class_word(list,row),chosen_word(lists,column),prefix,at,extent,extents,index,kept,after,before,box) :
      InBox(Tasks.target_spans(chosen_word(Con{list,lists},component),prefix,at),Con{extent,extents},index)}}

# The leading coordinate of the last prefix axis lies below the blocks its
# readings count.
def last_limit(+readings: List<&2,List<&2,Tasks.Reading>>,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +at: U32,+value: Nat,+stop: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},+last_axis: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool},
  +limit: {column_limit(Tasks.heads(readings)) ==
    NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))) : Nat},
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)) : Nat},
  +below: {Nat.is_lt(value,U32.to_nat(U32.max(per,1))) == True{} : Bool}) -> {Nat.is_lt(value,column_limit(Tasks.heads(readings))) == True{} : Bool}:
  %Equal.sym(Nat,column_limit(Tasks.heads(readings)),
      NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))),limit) :
    {Nat.is_lt(value,_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},stop) :
    {Nat.is_lt(value,NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head))))) == True{} : Bool}
  %Equal.sym(Bool,U32.is_eq((at + 1 : U32),prefix),True{},last_axis) :
    {Nat.is_lt(value,NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,block,head)))) == True{} : Bool}
  %last : {Nat.is_lt(value,_) == True{} : Bool}
  below

# The leading coordinate of a kept prefix axis lies below the blocks its
# readings count.
def kept_limit(+readings: List<&2,List<&2,Tasks.Reading>>,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +at: U32,+value: Nat,+kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool},
  +limit: {column_limit(Tasks.heads(readings)) ==
    NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))) : Nat},
  +below: {Nat.is_lt(value,U32.to_nat(head)) == True{} : Bool}) -> {Nat.is_lt(value,column_limit(Tasks.heads(readings))) == True{} : Bool}:
  %Equal.sym(Nat,column_limit(Tasks.heads(readings)),
      NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))),limit) :
    {Nat.is_lt(value,_) == True{} : Bool}
  %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},kept) :
    {Nat.is_lt(value,NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head))))) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(value,NatCeiling.ceiling(U32.to_nat(head),1n)),Nat.is_lt(Nat.mul(value,1n),U32.to_nat(head)),
      NatCeiling.upper_comparison(U32.to_nat(head),1n,value,{==})) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value)) : {Nat.is_lt(_,U32.to_nat(head)) == True{} : Bool}
  below

# Every piece of a grid with prefix axes lies in the target spans of one
# component.
def chosen_walk(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,cells: List<&2,U32>,remaining: Nat,+count: U32,+prefix: U32,+outer: U32,+extent: U32,
  +unit_cells: U32,+block: U32,+per: U32,+at: U32,+position: Nat,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+prefix_distance: {Nat.add(position,1n+remaining) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+reach: {Nat.is_lt(remaining,List.length(&2,U32,cells)) == True{} : Bool},
  +columns: ColumnsOk(rank,readings,cap),+lists_room: ListsRoom(raw_lists(rank,readings),cap),
  +room: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +limits: LimitsAt(readings,rank,cells,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at),
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(cells,remaining)),U32.to_nat(block)) : Nat},
  +below: {Nat.is_lt(index,ShapeLaws.elements(Tasks.piece_extents(cells,prefix,U32.max(per,1),at))) == True{} : Bool}) ->
  Chosen<raw_lists(rank,readings),prefix,at,Tasks.piece_extents(cells,prefix,U32.max(per,1),at),index>:
  match rank cells remaining:
    case 1n+ +rest Con{+head,+more} 0n:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      Held.Both{+limit,+rest_limits} = limits
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +size = U32.max(per,1)
      +stop = last_lt(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +last_axis = last_eq(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,0n,exact,prefix_distance)
      +shape = last_extents(head,more,prefix,size,at,stop)
      +inside = {%shape : {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool} below :
        {Nat.is_lt(index,ShapeLaws.elements(Con{size,Nil{}})) == True{} : Bool}}
      +value = Nat.div(index,ShapeLaws.elements(Nil{}))
      +value_below = DivisionOrder.quotient_below(index,U32.to_nat(size),ShapeLaws.elements(Nil{}),{==},inside)
      +found = column_cover(Tasks.heads(readings),value,cap,cap_ok,column_ok,
        last_limit(readings,head,count,prefix,outer,extent,unit_cells,block,per,at,value,stop,last_axis,limit,last,value_below))
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{size,Nil{}},shape) :
        Chosen<raw_lists(1n+rest,readings),prefix,at,_,index>
      chosen_step(list,lists,prefix,at,size,Nil{},index,cap,cap_ok,kept,rest_room,room,found,
        chosen_beyond(lists,prefix,(at + 1 : U32),Nat.mod(index,ShapeLaws.elements(Nil{})),stop,
          DivisionOrder.right_factor_positive(List.length(&2,Tasks.Class,list),count_nat(lists),positive)))
    case 1n+ +rest Con{+head,+more} 1n+ +later:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      Held.Both{+limit,+rest_limits} = limits
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +size = U32.max(per,1)
      +length = List.length(&2,Tasks.Class,list)
      +divisor = count_nat(lists)
      +total = Nat.mul(length,divisor)
      +keeps = kept_lt(at,prefix,position,later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,1n+later,exact,prefix_distance)
      +next = kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +inner_extents = Tasks.piece_extents(more,prefix,size,(at + 1 : U32))
      +inner = ShapeLaws.elements(inner_extents)
      +shape = kept_extents(head,more,prefix,size,at,keeps)
      +inside = {%shape : {Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool} below :
        {Nat.is_lt(index,ShapeLaws.elements(Con{head,inner_extents})) == True{} : Bool}}
      +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(U32.to_nat(head),inner),Order.zero_le(index),inside)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(head),inner,whole_positive)
      +value = Nat.div(index,inner)
      +found = column_cover(Tasks.heads(readings),value,cap,cap_ok,column_ok,
        kept_limit(readings,head,count,prefix,outer,extent,unit_cells,block,per,at,value,keeps,limit,
          DivisionOrder.quotient_below(index,U32.to_nat(head),inner,inner_positive,inside)))
      +length_positive = RowMajor.left_factor_positive(length,divisor,positive)
      +rest_positive = DivisionOrder.right_factor_positive(length,divisor,positive)
      +divisor_le = {%RowMajor.multiply_commutative(divisor,length) : {Nat.is_le(divisor,_) == True{} : Bool}
        RowMajor.product_covers(divisor,length,length_positive) : {Nat.is_le(divisor,total) == True{} : Bool}}
      +rest_double = Order.transitive(Nat.double(divisor),Nat.double(total),U32.to_nat(cap),Order.double_le(divisor,total,divisor_le),room)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{head,inner_extents},shape) :
        Chosen<raw_lists(1n+rest,readings),prefix,at,_,index>
      chosen_step(list,lists,prefix,at,head,inner_extents,index,cap,cap_ok,kept,rest_room,room,found,
        chosen_walk(rest,Tasks.tails(readings),more,later,count,prefix,outer,extent,unit_cells,block,per,(at + 1 : U32),1n+position,Nat.mod(index,inner),cap,cap_ok,
          next,steps_next(position,1n+later,U32.to_nat(prefix),prefix_distance),prefix_room,reach,rest_columns,rest_room,rest_double,rest_positive,rest_limits,last,
          DivisionOrder.remainder_small(index,inner,inner_positive)))
    case 0n Nil{} +remaining:
      Empty.absurd(Chosen<raw_lists(0n,readings),prefix,at,Tasks.piece_extents(Nil{},prefix,U32.max(per,1),at),index>,Order.below_zero(remaining,reach))
    case 0n Con{+head,+more} +remaining:
      Empty.absurd(Chosen<raw_lists(0n,readings),prefix,at,Tasks.piece_extents(Con{head,more},prefix,U32.max(per,1),at),index>,limits)
    case 1n+ +rest Nil{} +remaining:
      Empty.absurd(Chosen<raw_lists(1n+rest,readings),prefix,at,Tasks.piece_extents(Nil{},prefix,U32.max(per,1),at),index>,Order.below_zero(remaining,reach))

# ---- The target spans of every component ----

# Every span counts at least one index.
def Counted(spans: List<&2,Tasks.Span>) -> Data:
  match spans:
    case Nil{}:
      Unit
    case Con{Tasks.Span{+first,+count},rest}:
      Held.Both<{Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},Counted(rest)>

def counted_total(spans: List<&2,Tasks.Span>,+counted: Counted(spans)) -> {Nat.is_lt(0n,span_total(spans)) == True{} : Bool}:
  match spans:
    case Nil{}:
      {==}
    case Con{Tasks.Span{+first,+count},+rest}:
      Held.Both{+positive,+later} = counted
      Order.product_positive(U32.to_nat(count),span_total(rest),positive,counted_total(rest,later))

def positive_offset(+low: Nat,offset: Nat,+above: {Nat.is_lt(low,Nat.add(low,offset)) == True{} : Bool}) -> {Nat.is_lt(0n,offset) == True{} : Bool}:
  match offset:
    case 0n:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},lt_self(low,{%Addition.zero_right(low) : {Nat.is_lt(low,_) == True{} : Bool} above :
        {Nat.is_lt(low,low) == True{} : Bool}}))
    case 1n+ +more:
      {==}

# A class that ends by a limit in the cap ends at its first block plus its
# count.
def end_exact(+first: U32,+count: U32,+limit: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +starts: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),limit) == True{} : Bool},+room: {Nat.is_le(limit,U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(first),U32.to_nat(count)) == U32.to_nat((first + count : U32)) : Nat}:
  +ordered = lt_to_le(U32.to_nat(first),U32.to_nat((first + count : U32)),starts)
  +difference = Arithmetic.difference_exact(cap,cap_ok,(first + count : U32),first,U32.to_nat((first + count : U32)),U32.to_nat(first),{==},{==},ordered,
    Order.transitive(U32.to_nat((first + count : U32)),limit,U32.to_nat(cap),ends,room))
  %Arithmetic.subtract_first(first,count) : {Nat.add(U32.to_nat(first),U32.to_nat(_)) == U32.to_nat((first + count : U32)) : Nat}
  %Equal.sym(Nat,U32.to_nat(((first + count : U32) - first : U32)),Nat.sub(U32.to_nat((first + count : U32)),U32.to_nat(first)),difference) :
    {Nat.add(U32.to_nat(first),_) == U32.to_nat((first + count : U32)) : Nat}
  Order.restore_shift(U32.to_nat(first),U32.to_nat((first + count : U32)),ordered)

# A class ending by the extent of an axis spans indices inside it.
def class_in(+class: Tasks.Class,+rest: List<&2,Tasks.Class>,+prefix: U32,+at: U32,+extent: U32,+extents: List<&2,U32>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+kept: {U32.is_lt(at,prefix) == True{} : Bool},+ends: EndIn(class,U32.to_nat(extent)),
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},
  +later: Held.Both<SpansIn(Tasks.target_spans(rest,prefix,(at + 1 : U32)),extents),Counted(Tasks.target_spans(rest,prefix,(at + 1 : U32)))>) ->
  Held.Both<SpansIn(Tasks.target_spans(Con{class,rest},prefix,at),Con{extent,extents}),Counted(Tasks.target_spans(Con{class,rest},prefix,at))>:
  match class:
    case Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both{+starts,+stops} = ends
      Held.Both{+within,+counted} = later
      +exact = end_exact(first,count,U32.to_nat(extent),cap,cap_ok,starts,stops,room)
      %Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept) :
        Held.Both<SpansIn(Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(rest,prefix,(at + 1 : U32))},Nil{}),Con{extent,extents}),
          Counted(Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(rest,prefix,(at + 1 : U32))},Nil{}))>
      Held.Both{Held.Both{{%Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat((first + count : U32)),exact) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool} stops :
            {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent)) == True{} : Bool}},within},
        Held.Both{positive_offset(U32.to_nat(first),U32.to_nat(count),{%Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat((first + count : U32)),exact) : {Nat.is_lt(U32.to_nat(first),_) == True{} : Bool} starts :
            {Nat.is_lt(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}),counted}}

# The class of a column ends by the blocks its readings count.
def column_end(column: List<&2,Tasks.Reading>,+class: Tasks.Class,+ends: ColumnEnds(column,class),
  +positive: {Nat.is_lt(0n,column_limit(column)) == True{} : Bool}) -> EndIn(class,column_limit(column)):
  match column:
    case Nil{}:
      Empty.absurd(EndIn(class,column_limit(Nil{})),lt_self(0n,positive))
    case Con{Tasks.Reading{+extent,+block,+source},+others}:
      Held.Both{+here,+rest} = ends
      here

def ceiling_one(+value: Nat) -> {NatCeiling.ceiling(value,1n) == value : Nat}:
  Order.antisymmetric(NatCeiling.ceiling(value,1n),value,NatCeiling.bounded(value,1n,{==}),
    {%Order.times_one(NatCeiling.ceiling(value,1n)) :
        {Nat.is_le(value,_) == True{} : Bool}
      NatCeiling.covers(value,1n,{==}) : {Nat.is_le(value,NatCeiling.ceiling(value,1n)) == True{} : Bool}})

# The blocks the readings of a kept prefix axis count are its extent.
def kept_blocks_count(+readings: List<&2,List<&2,Tasks.Reading>>,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +at: U32,+kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool},
  +limit: {column_limit(Tasks.heads(readings)) ==
    NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))) : Nat}) ->
  {column_limit(Tasks.heads(readings)) == U32.to_nat(head) : Nat}:
  +reduced = {%kept :
      {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head)))) : Nat}
    limit : {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),1n) : Nat}}
  Equal.trans(Nat,column_limit(Tasks.heads(readings)),NatCeiling.ceiling(U32.to_nat(head),1n),U32.to_nat(head),reduced,ceiling_one(U32.to_nat(head)))

# The blocks the readings of the last prefix axis count are the pieces of an
# outer index.
def last_blocks_count(+readings: List<&2,List<&2,Tasks.Reading>>,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +at: U32,+stop: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},+last_axis: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool},
  +limit: {column_limit(Tasks.heads(readings)) ==
    NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head))) : Nat},
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)) : Nat}) ->
  {column_limit(Tasks.heads(readings)) == U32.to_nat(U32.max(per,1)) : Nat}:
  +passed = {%stop :
      {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head)))) : Nat}
    limit : {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head))) : Nat}}
  +reduced = {%last_axis : {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(Bool.pick(U32,_,block,head))) : Nat}
    passed : {column_limit(Tasks.heads(readings)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)) : Nat}}
  Equal.trans(Nat,column_limit(Tasks.heads(readings)),NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)),U32.to_nat(U32.max(per,1)),reduced,
    Equal.sym(Nat,U32.to_nat(U32.max(per,1)),NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)),last))

# The class an index picks on an axis ends by the blocks of the axis.
def picked_end(+readings: List<&2,List<&2,Tasks.Reading>>,+rest: Nat,+index: U32,+size: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +column_ok: ColumnOk(Tasks.heads(readings),cap),
  +split: IndexSplit<Tasks.axis_classes(Tasks.heads(readings)),raw_lists(rest,Tasks.tails(readings)),index,cap>,
  +blocks: {column_limit(Tasks.heads(readings)) == U32.to_nat(size) : Nat},+positive: {Nat.is_lt(0n,U32.to_nat(size)) == True{} : Bool}) ->
  EndIn(class_word(Tasks.axis_classes(Tasks.heads(readings)),(index / inner_of(raw_lists(rest,Tasks.tails(readings))) : U32)),U32.to_nat(size)):
  +list = Tasks.axis_classes(Tasks.heads(readings))
  +lists = raw_lists(rest,Tasks.tails(readings))
  +quotient = (index / inner_of(lists) : U32)
  %blocks : EndIn(class_word(list,quotient),_)
  column_end(Tasks.heads(readings),class_word(list,quotient),
    column_ends(Tasks.heads(readings),quotient,cap,cap_ok,column_ok,quotient_inside_of(list,lists,index,cap,split),quotient_room_of(list,lists,index,cap,split)),
    {%Equal.sym(Nat,column_limit(Tasks.heads(readings)),U32.to_nat(size),blocks) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,column_limit(Tasks.heads(readings))) == True{} : Bool}})

# The target spans of every component lie inside the pieces of a grid with
# prefix axes, and none is empty.
def chosen_within(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,cells: List<&2,U32>,remaining: Nat,+count: U32,+prefix: U32,+outer: U32,+extent: U32,
  +unit_cells: U32,+block: U32,+per: U32,+at: U32,+position: Nat,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+prefix_distance: {Nat.add(position,1n+remaining) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+reach: {Nat.is_lt(remaining,List.length(&2,U32,cells)) == True{} : Bool},
  +columns: ColumnsOk(rank,readings,cap),+lists_room: ListsRoom(raw_lists(rank,readings),cap),
  +room: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +limits: LimitsAt(readings,rank,cells,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at),
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(cells,remaining)),U32.to_nat(block)) : Nat},
  +filled: {Nat.is_lt(0n,ShapeLaws.elements(Tasks.piece_extents(cells,prefix,U32.max(per,1),at))) == True{} : Bool},
  +fits: {Nat.is_le(ShapeLaws.elements(Tasks.piece_extents(cells,prefix,U32.max(per,1),at)),U32.to_nat(cap)) == True{} : Bool}) ->
  Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,at),Tasks.piece_extents(cells,prefix,U32.max(per,1),at)),
    Counted(Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,at))>:
  match rank cells remaining:
    case 1n+ +rest Con{+head,+more} 0n:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      Held.Both{+limit,+rest_limits} = limits
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +inner = inner_of(lists)
      +size = U32.max(per,1)
      +stop = last_lt(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +last_axis = last_eq(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,0n,exact,prefix_distance)
      +shape = last_extents(head,more,prefix,size,at,stop)
      +size_filled = {%shape : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} filled : {Nat.is_lt(0n,Nat.mul(U32.to_nat(size),1n)) == True{} : Bool}}
      +size_fits = {%shape : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(cap)) == True{} : Bool} fits : {Nat.is_le(Nat.mul(U32.to_nat(size),1n),U32.to_nat(cap)) == True{} : Bool}}
      +split = split_index(list,lists,index,cap,cap_ok,list_room,rest_room,room,inside)
      +ends = picked_end(readings,rest,index,size,cap,cap_ok,column_ok,split,
        last_blocks_count(readings,head,count,prefix,outer,extent,unit_cells,block,per,at,stop,last_axis,limit,last),
        RowMajor.left_factor_positive(U32.to_nat(size),1n,size_filled))
      +remainder = (index % inner : U32)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{size,Nil{}},shape) :
        Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at),_),
          Counted(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at))>
      class_in(class_word(list,(index / inner : U32)),chosen_word(lists,remainder),prefix,at,size,Nil{},cap,cap_ok,kept,ends,
        {%Order.times_one(U32.to_nat(size)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          size_fits : {Nat.is_le(U32.to_nat(size),U32.to_nat(cap)) == True{} : Bool}},
        {%Equal.sym(List<&2,Tasks.Span>,Tasks.target_spans(chosen_word(lists,remainder),prefix,(at + 1 : U32)),Nil{},
            spans_end(chosen_word(lists,remainder),prefix,(at + 1 : U32),stop)) :
            Held.Both<SpansIn(_,Nil{}),Counted(_)>
          Held.Both{Unit{},Unit{}} :
          Held.Both<SpansIn(Tasks.target_spans(chosen_word(lists,remainder),prefix,(at + 1 : U32)),Nil{}),
            Counted(Tasks.target_spans(chosen_word(lists,remainder),prefix,(at + 1 : U32)))>})
    case 1n+ +rest Con{+head,+more} 1n+ +later:
      Held.Both{+column_ok,+rest_columns} = columns
      Held.Both{+list_room,+rest_room} = lists_room
      Held.Both{+limit,+rest_limits} = limits
      +list = Tasks.axis_classes(Tasks.heads(readings))
      +lists = raw_lists(rest,Tasks.tails(readings))
      +inner = inner_of(lists)
      +size = U32.max(per,1)
      +keeps = kept_lt(at,prefix,position,later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,1n+later,exact,prefix_distance)
      +next = kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +inner_extents = Tasks.piece_extents(more,prefix,size,(at + 1 : U32))
      +cells_inner = ShapeLaws.elements(inner_extents)
      +shape = kept_extents(head,more,prefix,size,at,keeps)
      +head_filled = {%shape : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} filled :
        {Nat.is_lt(0n,Nat.mul(U32.to_nat(head),cells_inner)) == True{} : Bool}}
      +head_fits = {%shape : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(cap)) == True{} : Bool} fits :
        {Nat.is_le(Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap)) == True{} : Bool}}
      +head_positive = RowMajor.left_factor_positive(U32.to_nat(head),cells_inner,head_filled)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(head),cells_inner,head_filled)
      +inner_le = Held.nat_le_right_moved(cells_inner,Nat.mul(U32.to_nat(head),cells_inner),Nat.mul(cells_inner,U32.to_nat(head)),
        RowMajor.multiply_commutative(U32.to_nat(head),cells_inner),RowMajor.product_covers(cells_inner,U32.to_nat(head),head_positive))
      +split = split_index(list,lists,index,cap,cap_ok,list_room,rest_room,room,inside)
      +ends = picked_end(readings,rest,index,head,cap,cap_ok,column_ok,split,
        kept_blocks_count(readings,head,count,prefix,outer,extent,unit_cells,block,per,at,keeps,limit),head_positive)
      +remainder = (index % inner : U32)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{head,inner_extents},shape) :
        Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at),_),
          Counted(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at))>
      class_in(class_word(list,(index / inner : U32)),chosen_word(lists,remainder),prefix,at,head,inner_extents,cap,cap_ok,kept,ends,
        Order.transitive(U32.to_nat(head),Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap),
          RowMajor.product_covers(U32.to_nat(head),cells_inner,inner_positive),head_fits),
        chosen_within(rest,Tasks.tails(readings),more,later,count,prefix,outer,extent,unit_cells,block,per,(at + 1 : U32),1n+position,remainder,cap,cap_ok,
          next,steps_next(position,1n+later,U32.to_nat(prefix),prefix_distance),prefix_room,reach,rest_columns,rest_room,
          rest_double_of(list,lists,index,cap,split),remainder_inside_of(list,lists,index,cap,split),rest_limits,last,inner_positive,
          Order.transitive(cells_inner,Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap),inner_le,head_fits)))
    case 0n Nil{} +remaining:
      Empty.absurd(Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(0n,readings),index),prefix,at),Tasks.piece_extents(Nil{},prefix,U32.max(per,1),at)),
        Counted(Tasks.target_spans(chosen_word(raw_lists(0n,readings),index),prefix,at))>,Order.below_zero(remaining,reach))
    case 0n Con{+head,+more} +remaining:
      Empty.absurd(Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(0n,readings),index),prefix,at),
        Tasks.piece_extents(Con{head,more},prefix,U32.max(per,1),at)),Counted(Tasks.target_spans(chosen_word(raw_lists(0n,readings),index),prefix,at))>,limits)
    case 1n+ +rest Nil{} +remaining:
      Empty.absurd(Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at),Tasks.piece_extents(Nil{},prefix,U32.max(per,1),at)),
        Counted(Tasks.target_spans(chosen_word(raw_lists(1n+rest,readings),index),prefix,at))>,Order.below_zero(remaining,reach))

# ---- Target pieces are leaves of the target layout ----

# A piece of a grid without cells would start where the next one does.
def span_empty(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +zero: {U32.is_zero(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece,1)) == True{} : Bool}) -> Empty:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +value = U32.to_nat(piece)
  +next = U32.from_nat(1n+value)
  +next_le = Order.strict_follow(value,total,inside)
  +next_value = Bounds.roundtrip(1n+value,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
    Order.transitive(1n+value,total,U32.to_nat(Machine.capacity(limits)),next_le,GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +sum = GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,value,inside)
  +own = {%Arithmetic.native_roundtrip(piece) :
      {Nat.add(U32.to_nat(Grid.start(grid,_)),U32.to_nat(Pieces.cells_of(grid,_,1))) == U32.to_nat(Grid.start(grid,next)) : Nat}
    sum : {Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Pieces.cells_of(grid,piece,1))) == U32.to_nat(Grid.start(grid,next)) : Nat}}
  +empty = {%Arithmetic.word_zero(Pieces.cells_of(grid,piece,1),zero) :
      {Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(_)) == U32.to_nat(Grid.start(grid,next)) : Nat}
    own : {Nat.add(U32.to_nat(Grid.start(grid,piece)),0n) == U32.to_nat(Grid.start(grid,next)) : Nat}}
  +same = Arithmetic.to_nat_injective(Grid.start(grid,piece),Grid.start(grid,next),
    Equal.trans(Nat,U32.to_nat(Grid.start(grid,piece)),Nat.add(U32.to_nat(Grid.start(grid,piece)),0n),U32.to_nat(Grid.start(grid,next)),
      Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(grid,piece)),0n),U32.to_nat(Grid.start(grid,piece)),Addition.zero_right(U32.to_nat(Grid.start(grid,piece)))),empty))
  +back = Equal.sym(Nat,U32.to_nat(next),1n+value,next_value)
  starts_clash(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,next,lt_to_le(value,total,inside),
    {%back : {Nat.is_le(_,total) == True{} : Bool} next_le : {Nat.is_le(U32.to_nat(next),total) == True{} : Bool}},
    {%back : {Nat.is_eq(value,_) == False{} : Bool} Order.less_not_equal(value,1n+value,Order.succ_le_lt(value,1n+value,Order.reflexive(1n+value))) :
      {Nat.is_eq(value,U32.to_nat(next)) == False{} : Bool}},
    same)

# Every piece of a grid holds cells.
def span_nonzero(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.is_zero(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == False{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  Held.by_cases({U32.is_zero(Grid.span(grid,piece)) == False{} : Bool},U32.is_zero(Pieces.cells_of(grid,piece,1)),
    zero => Empty.absurd({U32.is_zero(Grid.span(grid,piece)) == False{} : Bool},span_empty(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,zero)),
    nonzero => nonzero)

def target_pair(+layout: Pieces.Ranges,+grid: Grid.Grid,+piece: U32,
  +pair: Held.Both<Placed.LeafAt(layout,Grid.start(grid,piece)),{Placed.rows_of(layout,Grid.start(grid,piece)) == Grid.span(grid,piece) : U32}>,
  +nonzero: {U32.is_zero(Grid.span(grid,piece)) == False{} : Bool}) -> Moved.TargetAt(layout,grid,piece):
  Held.Both{+leaf,+rows} = pair
  Held.Both{leaf,Held.Both{rows,nonzero}}

# A piece of a grid starts a leaf of its layout with its cells.
def target_at(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Moved.TargetAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  target_pair(Pieces.ranges_of_grid(grid),grid,piece,piece_leaf(limits,outer,extent,unit_cells,block,per,count,prefix,shape,ordered,piece,inside),
    span_nonzero(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside))

# A piece of a grid ends where the piece after it starts, at or before the
# start of any later piece.
def piece_before(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+later: U32,
  +inside: {Nat.is_lt(U32.to_nat(later),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +after: {Nat.is_le(1n+U32.to_nat(piece),U32.to_nat(later)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},later))) == True{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +value = U32.to_nat(piece)
  +next = U32.from_nat(1n+value)
  +later_le = Bounds.strict_le(U32.to_nat(later),total,inside)
  +piece_inside = Bounds.nat_trans(value,U32.to_nat(later),total,Order.succ_le_lt(value,U32.to_nat(later),after),later_le)
  +next_value = Bounds.roundtrip(1n+value,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
    Order.transitive(1n+value,total,U32.to_nat(Machine.capacity(limits)),Order.strict_follow(value,total,piece_inside),
      GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +sum = GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,value,piece_inside)
  +own = {%Arithmetic.native_roundtrip(piece) :
      {Nat.add(U32.to_nat(Grid.start(grid,_)),U32.to_nat(Pieces.cells_of(grid,_,1))) == U32.to_nat(Grid.start(grid,next)) : Nat}
    sum : {Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Pieces.cells_of(grid,piece,1))) == U32.to_nat(Grid.start(grid,next)) : Nat}}
  +monotone = GridLaws.start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,next,later,later_le,
    {%Equal.sym(Nat,U32.to_nat(next),1n+value,next_value) : {Nat.is_le(_,U32.to_nat(later)) == True{} : Bool}
      after : {Nat.is_le(U32.to_nat(next),U32.to_nat(later)) == True{} : Bool}})
  %Equal.sym(U32,Grid.span(grid,piece),Pieces.cells_of(grid,piece,1),GridRanges.span_cells(outer,extent,unit_cells,block,per,count,prefix,piece)) :
    {Nat.is_le(Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(_)),U32.to_nat(Grid.start(grid,later))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Pieces.cells_of(grid,piece,1))),U32.to_nat(Grid.start(grid,next)),own) :
    {Nat.is_le(_,U32.to_nat(Grid.start(grid,later))) == True{} : Bool}
  monotone

# Rising pieces of a grid, at least one, are targets of its layout.
def targets_rising(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  list: List<&2,U32>,+low: Nat,+rising: Rising(list,low,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
  +nonempty: {Nat.is_lt(0n,List.length(&2,U32,list)) == True{} : Bool}) ->
  Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},list):
  match list:
    case Nil{}:
      Empty.absurd(Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
        Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nil{}),lt_self(0n,nonempty))
    case Con{+piece,Nil{}}:
      Held.Both{+above,+more} = rising
      Held.Both{+inside,+later} = more
      target_at(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,piece,inside)
    case Con{+piece,Con{+next,+rest}}:
      Held.Both{+above,+more} = rising
      Held.Both{+inside,+later} = more
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      Held.Both{target_at(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,piece,inside),
        Held.Both{piece_before(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,next,next_below(next,rest,1n+U32.to_nat(piece),total,later),
            next_above(next,rest,1n+U32.to_nat(piece),total,later)),
          targets_rising(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,Con{next,rest},1n+U32.to_nat(piece),later,{==})}}

# The pieces of spans inside the pieces of a grid with prefix axes, none
# empty, are targets of its layout.
def targets_in(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +axes: List<&2,U32>,+spans: List<&2,Tasks.Span>,+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +within: SpansIn(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),+counted: Counted(spans)) ->
  Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,spans)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,extents,Tasks.span_count(spans),Nil{})
  +length = listed_length(spans,extents,Machine.half_word(limits),LimitBounds.half_word_fits(limits),
    spans_half(limits,outer,extent,unit_cells,block,per,shape,axes,prefix,spans,total_exact,within))
  %Equal.sym(Bool,U32.is_zero(prefix),False{},some) :
    Moved.TargetsAt(Pieces.ranges_of_grid(grid),grid,Bool.pick(List<&2,U32>,_,Con{0,Nil{}},listed))
  targets_rising(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,listed,0n,
    {%total_exact : Rising(listed,0n,_)
      rising_listed(listed,spans,extents,within,pieces_box_listed(limits,outer,extent,unit_cells,block,per,shape,axes,prefix,spans,total_exact,within)) :
      Rising(listed,0n,total)},
    {%Equal.sym(Nat,List.length(&2,U32,listed),span_total(spans),length) : {Nat.is_lt(0n,_) == True{} : Bool}
      counted_total(spans,counted) : {Nat.is_lt(0n,List.length(&2,U32,listed)) == True{} : Bool}})

# ---- Every target piece is listed by its component ----

def targets_later(pieces: List<&2,U32>,+grid: Grid.Grid,+rest: Held.Spans,+cell: U32,+listed: Placed.Listed(rest,cell)) ->
  Placed.Listed(Moved.target_spans(pieces,grid,rest),cell):
  match pieces:
    case Nil{}:
      Placed.OnRight{listed}
    case Con{+piece,Nil{}}:
      Placed.OnRight{listed}
    case Con{+piece,Con{+next,+more}}:
      Placed.OnRight{targets_later(Con{next,more},grid,rest,cell,listed)}

# A cell listed after the spans of a tree of components stays listed.
def ranges_later(ranges: Pieces.Ranges,+components: Tasks.Components,+target: List<&2,U32>,+grid: Grid.Grid,+rest: Held.Spans,+cell: U32,
  +listed: Placed.Listed(rest,cell)) -> Placed.Listed(Moved.ranges_spans(ranges,components,target,grid,rest),cell):
  match ranges:
    case Pieces.Range{+first,+count}:
      targets_later(Tasks.target_pieces(components,grid,target,first),grid,rest,cell,listed)
    case Pieces.Split{+left,+right}:
      ranges_later(left,components,target,grid,Moved.ranges_spans(right,components,target,grid,rest),cell,
        ranges_later(right,components,target,grid,rest,cell,listed))

# The one index a leaf of at most one index holds is its first.
def leaf_index(+first: U32,+count: U32,+index: U32,+low: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},
  +high: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},
  +one: {Nat.is_le(U32.to_nat(count),1n) == True{} : Bool}) -> {index == first : U32}:
  +bound = Order.lt_le(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(first),1n),high,
    Order.offset_le(U32.to_nat(first),U32.to_nat(count),1n,one))
  +below = {%Addition.successor_width(U32.to_nat(first)) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} bound :
    {Nat.is_lt(U32.to_nat(index),1n+U32.to_nat(first)) == True{} : Bool}}
  Arithmetic.to_nat_injective(index,first,Order.antisymmetric(U32.to_nat(index),U32.to_nat(first),Division.lt_succ_le(U32.to_nat(index),U32.to_nat(first),below),low))

def single_one(+count: U32,+test: {U32.is_le(count,1) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(count),1n) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(U32.to_nat(count),1n),U32.is_le(count,1),True{},Equal.sym(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),1n),Comparison.le(count,1)),test)

def right_low(+first: U32,+left: U32,+index: U32,+first_exact: {U32.to_nat((first + left : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(left)) : Nat},
  +later: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(left))) == False{} : Bool}) ->
  {Nat.is_le(U32.to_nat((first + left : U32)),U32.to_nat(index)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((first + left : U32)),Nat.add(U32.to_nat(first),U32.to_nat(left)),first_exact) : {Nat.is_le(_,U32.to_nat(index)) == True{} : Bool}
  Order.not_less(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(left)),later)

# A piece of the component at index is listed by the spans of a tree of
# components holding index.
def listed_tree(+limits: Machine.Limits,+components: Tasks.Components,+target: List<&2,U32>,+grid: Grid.Grid,depth: Nat,single: Bool,+first: U32,+count: U32,
  +test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+index: U32,
  +low: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},
  +high: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},+piece: U32,+rest: Held.Spans,
  +member: Moved.Member(Tasks.target_pieces(components,grid,target,index),piece)) ->
  Placed.Listed(Moved.ranges_spans(Tasks.index_ranges(depth,single,first,count),components,target,grid,rest),Grid.start(grid,piece)):
  match depth single:
    case 0n +single:
      %leaf_index(first,count,index,low,high,bounded) : Placed.Listed(Moved.target_spans(Tasks.target_pieces(components,grid,target,_),grid,rest),Grid.start(grid,piece))
      in_targets(Tasks.target_pieces(components,grid,target,index),grid,rest,piece,member)
    case 1n+ +more True{}:
      %leaf_index(first,count,index,low,high,single_one(count,test)) :
        Placed.Listed(Moved.target_spans(Tasks.target_pieces(components,grid,target,_),grid,rest),Grid.start(grid,piece))
      in_targets(Tasks.target_pieces(components,grid,target,index),grid,rest,piece,member)
    case 1n+ +more False{}:
      +cap = Machine.capacity(limits)
      +start = U32.to_nat(first)
      +total = U32.to_nat(count)
      +inside = Order.transitive(total,Nat.add(start,total),U32.to_nat(cap),Order.right_le(start,total),room)
      +halves = GridRanges.halves_of(cap,LimitBounds.capacity_allocatable(limits),more,count,test,positive,bounded,inside)
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +left_count = GridRanges.halves_left_count(count,more,halves)
      +left_room = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,U32.to_nat(left),total,left_count),room)
      +first_exact = Arithmetic.add(first,left,cap,LimitBounds.capacity_fits(limits),left_room)
      +right_sum = GridRanges.split_sum(first,count,first_exact,GridRanges.halves_right_exact(count,more,halves),left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),U32.to_nat(cap),right_sum,room)
      +left_tree = Tasks.index_ranges(more,U32.is_le(left,1),first,left)
      +right_tree = Tasks.index_ranges(more,U32.is_le(right,1),(first + left : U32),right)
      +after = Moved.ranges_spans(right_tree,components,target,grid,rest)
      Held.by_cases(Placed.Listed(Moved.ranges_spans(Pieces.Split{left_tree,right_tree},components,target,grid,rest),Grid.start(grid,piece)),
        Nat.is_lt(U32.to_nat(index),Nat.add(start,U32.to_nat(left))),
        before => listed_tree(limits,components,target,grid,more,U32.is_le(left,1),first,left,{==},GridRanges.halves_left_positive(count,more,halves),
          GridRanges.halves_left_bound(count,more,halves),left_room,index,low,before,piece,after,member),
        later => ranges_later(left_tree,components,target,grid,after,Grid.start(grid,piece),
          listed_tree(limits,components,target,grid,more,U32.is_le(right,1),(first + left : U32),right,{==},GridRanges.halves_right_positive(count,more,halves),
            GridRanges.halves_right_bound(count,more,halves),right_room,index,right_low(first,left,index,first_exact,later),
            {%Equal.sym(Nat,Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),right_sum) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
              high : {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right))) == True{} : Bool}},piece,rest,member)))

# ---- A grid without prefix axes is one piece ----

def units_zero(axes: List<&2,U32>,fits: Bool,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(cap)) == True{} : Bool},
  +none: {U32.is_zero(units_prefix(GridLaws.units_model(axes,fits,0,1,1,target))) == True{} : Bool}) ->
  Held.Both<{units_outer(GridLaws.units_model(axes,fits,0,1,1,target)) == 1 : U32},{units_extent(GridLaws.units_model(axes,fits,0,1,1,target)) == 1 : U32}>:
  match axes fits:
    case +axes True{}:
      Held.Both{{==},{==}}
    case Nil{} False{}:
      Held.Both{{==},{==}}
    case Con{+last,Nil{}} False{}:
      Held.Both{{==},{==}}
    case Con{+first,Con{+next,+tail}} False{}:
      +rest = {Con{next,tail} : List<&2,U32>}
      +next_fits = U32.is_le(Grid.product(rest),target)
      +found = GridLaws.units_model(rest,next_fits,(0 + 1 : U32),1,first,target)
      +reached = prefix_reached(rest,next_fits,0,1,first,target,0n,cap,cap_ok,{==},room_axes(1,first,rest,0n,cap,top_room(first,rest,cap,room,positive,rank_room)))
      Empty.absurd(Held.Both<{units_outer(found) == 1 : U32},{units_extent(found) == 1 : U32}>,
        WordEncoding.false_true({%Arithmetic.word_zero(units_prefix(found),none) : {Nat.is_le(1n+0n,U32.to_nat(_)) == True{} : Bool} reached :
          {Nat.is_le(1n+0n,U32.to_nat(0)) == True{} : Bool}}))

def shape_outer_eq(+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}) ->
  {units_outer(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) == outer : U32}:
  +found = GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)
  Equal.trans(U32,units_outer(found),grid_outer(Grid.grid(axes,target)),outer,Equal.sym(U32,grid_outer(Grid.grid(axes,target)),units_outer(found),
    shape_grid_outer(axes,target)),Equal.cong(Grid.Grid,U32,whole => grid_outer(whole),Grid.grid(axes,target),
      Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact))

def shape_extent_eq(+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}) ->
  {units_extent(GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)) == extent : U32}:
  +found = GridLaws.units_model(axes,U32.is_le(Grid.product(axes),target),0,1,1,target)
  Equal.trans(U32,units_extent(found),grid_extent(Grid.grid(axes,target)),extent,Equal.sym(U32,grid_extent(Grid.grid(axes,target)),units_extent(found),
    shape_grid_extent(axes,target)),Equal.cong(Grid.Grid,U32,whole => grid_extent(whole),Grid.grid(axes,target),
      Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact))

# A grid of one outer index and one index of its last prefix axis is one
# piece.
def one_piece(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+outer_one: {outer == 1 : U32},+extent_one: {extent == 1 : U32}) ->
  {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == 1n : Nat}:
  GridLaws.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+room} = shape
  +block_one = Order.antisymmetric(U32.to_nat(block),1n,{%extent_one : {Nat.is_le(U32.to_nat(block),U32.to_nat(_)) == True{} : Bool} block_below :
      {Nat.is_le(U32.to_nat(block),1n) == True{} : Bool}},Order.strict_follow(0n,U32.to_nat(block),block_positive))
  +per_one = Equal.trans(Nat,U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),1n,per_exact,
    {%Equal.sym(U32,extent,1,extent_one) : {NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(block)) == 1n : Nat}
      %Equal.sym(Nat,U32.to_nat(block),1n,block_one) : {NatCeiling.ceiling(U32.to_nat(1),_) == 1n : Nat}
      ceiling_one(1n) : {NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)) == 1n : Nat}})
  %Equal.sym(U32,outer,1,outer_one) : {Nat.mul(U32.to_nat(_),U32.to_nat(per)) == 1n : Nat}
  %Equal.sym(Nat,U32.to_nat(per),1n,per_one) : {Nat.mul(U32.to_nat(1),_) == 1n : Nat}
  {==}

def piece_from_units(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+found: Grid.Units,+outer_eq: {units_outer(found) == outer : U32},
  +extent_eq: {units_extent(found) == extent : U32},+units: Held.Both<{units_outer(found) == 1 : U32},{units_extent(found) == 1 : U32}>) ->
  {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == 1n : Nat}:
  Held.Both{+outer_found,+extent_found} = units
  one_piece(limits,outer,extent,unit_cells,block,per,shape,Equal.trans(U32,outer,units_outer(found),1,Equal.sym(U32,units_outer(found),outer,outer_eq),outer_found),
    Equal.trans(U32,extent,units_extent(found),1,Equal.sym(U32,units_extent(found),extent,extent_eq),extent_found))

# The grid of a shape without prefix axes is one piece.
def prefix_zero_total(+limits: Machine.Limits,+axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +none: {U32.is_zero(prefix) == True{} : Bool}) -> {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == 1n : Nat}:
  +fits = U32.is_le(Grid.product(axes),target)
  +found = GridLaws.units_model(axes,fits,0,1,1,target)
  +zero = {%Equal.sym(U32,units_prefix(found),prefix,shape_prefix_eq(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact)) :
      {U32.is_zero(_) == True{} : Bool}
    none : {U32.is_zero(units_prefix(found)) == True{} : Bool}}
  piece_from_units(limits,outer,extent,unit_cells,block,per,shape,found,shape_outer_eq(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact),
    shape_extent_eq(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact),
    units_zero(axes,fits,target,Machine.capacity(limits),LimitBounds.capacity_fits(limits),room,positive,rank_room,zero))

# ---- The whole spans of the prefix axes ----

def whole_end(extents: List<&2,U32>,+grid: Grid.Grid,+prefix: U32,+at: U32,+stop: {U32.is_lt(at,prefix) == False{} : Bool}) ->
  {Tasks.whole_blocks(extents,grid,prefix,at) == Nil{} : List<&2,Tasks.Span>}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      %Equal.sym(Bool,U32.is_lt(at,prefix),False{},stop) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(extent,Grid.block(grid,at,extent))},Tasks.whole_blocks(rest,grid,prefix,(at + 1 : U32))},Nil{}) ==
          Nil{} : List<&2,Tasks.Span>}
      {==}

def kept_whole_count(+limits: Machine.Limits,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+at: U32,
  +keeps: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool},+head_room: {Nat.is_le(U32.to_nat(head),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +head_positive: {Nat.is_lt(0n,U32.to_nat(head)) == True{} : Bool}) ->
  {Tasks.target_blocks(head,Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head)) == head : U32}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  Arithmetic.to_nat_injective(Tasks.target_blocks(head,Grid.block(grid,at,head)),head,
    {%Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},keeps) :
        {U32.to_nat(Grid.divided_up(head,Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head)))) == U32.to_nat(head) : Nat}
      Equal.trans(Nat,U32.to_nat(Grid.divided_up(head,1)),NatCeiling.ceiling(U32.to_nat(head),1n),U32.to_nat(head),
        GridLaws.divided_exact(limits,head,1,head_room,
          Order.transitive(1n,U32.to_nat(head),U32.to_nat(Machine.capacity(limits)),Order.strict_follow(0n,U32.to_nat(head),head_positive),head_room),{==}),
        ceiling_one(U32.to_nat(head))) :
      {U32.to_nat(Tasks.target_blocks(head,Grid.block(grid,at,head))) == U32.to_nat(head) : Nat}})

def last_whole_count(+limits: Machine.Limits,+head: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+at: U32,
  +stop: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},+last_axis: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool},
  +head_room: {Nat.is_le(U32.to_nat(head),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_room: {Nat.is_le(U32.to_nat(block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)) : Nat}) ->
  {Tasks.target_blocks(head,Grid.block(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},at,head)) == U32.max(per,1) : U32}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  Arithmetic.to_nat_injective(Tasks.target_blocks(head,Grid.block(grid,at,head)),U32.max(per,1),
    {%Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},stop) :
        {U32.to_nat(Grid.divided_up(head,Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,head)))) == U32.to_nat(U32.max(per,1)) : Nat}
      %Equal.sym(Bool,U32.is_eq((at + 1 : U32),prefix),True{},last_axis) :
        {U32.to_nat(Grid.divided_up(head,Bool.pick(U32,_,block,head))) == U32.to_nat(U32.max(per,1)) : Nat}
      Equal.trans(Nat,U32.to_nat(Grid.divided_up(head,block)),NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)),U32.to_nat(U32.max(per,1)),
        GridLaws.divided_exact(limits,head,block,head_room,block_room,block_positive),
        Equal.sym(Nat,U32.to_nat(U32.max(per,1)),NatCeiling.ceiling(U32.to_nat(head),U32.to_nat(block)),last)) :
      {U32.to_nat(Tasks.target_blocks(head,Grid.block(grid,at,head))) == U32.to_nat(U32.max(per,1)) : Nat}})

# The whole blocks of the prefix axes are the whole spans of the pieces.
def whole_blocks_spans(cells: List<&2,U32>,remaining: Nat,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,
  +per: U32,+at: U32,+position: Nat,+exact: {U32.to_nat(at) == position : Nat},+prefix_distance: {Nat.add(position,1n+remaining) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +reach: {Nat.is_lt(remaining,List.length(&2,U32,cells)) == True{} : Bool},
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(cells,remaining)),U32.to_nat(block)) : Nat},
  +last_room: {Nat.is_le(U32.to_nat(extent_nat(cells,remaining)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_room: {Nat.is_le(U32.to_nat(block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +filled: {Nat.is_lt(0n,ShapeLaws.elements(Tasks.piece_extents(cells,prefix,U32.max(per,1),at))) == True{} : Bool},
  +fits: {Nat.is_le(ShapeLaws.elements(Tasks.piece_extents(cells,prefix,U32.max(per,1),at)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Tasks.whole_blocks(cells,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,at) ==
    Tasks.whole_spans(Tasks.piece_extents(cells,prefix,U32.max(per,1),at)) : List<&2,Tasks.Span>}:
  match cells remaining:
    case Con{+head,+more} 0n:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +cap = Machine.capacity(limits)
      +cap_ok = LimitBounds.capacity_fits(limits)
      +size = U32.max(per,1)
      +stop = last_lt(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +last_axis = last_eq(at,prefix,position,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,0n,exact,prefix_distance)
      +shape = last_extents(head,more,prefix,size,at,stop)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{size,Nil{}},shape) :
        {Tasks.whole_blocks(Con{head,more},grid,prefix,at) == Tasks.whole_spans(_) : List<&2,Tasks.Span>}
      %Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(head,Grid.block(grid,at,head))},Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32))},Nil{}) ==
          Tasks.whole_spans(Con{size,Nil{}}) : List<&2,Tasks.Span>}
      %Equal.sym(U32,Tasks.target_blocks(head,Grid.block(grid,at,head)),size,
          last_whole_count(limits,head,count,prefix,outer,extent,unit_cells,block,per,at,stop,last_axis,last_room,block_room,block_positive,last)) :
        {Con{Tasks.Span{0,_},Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32))} == Tasks.whole_spans(Con{size,Nil{}}) : List<&2,Tasks.Span>}
      %Equal.sym(List<&2,Tasks.Span>,Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32)),Nil{},whole_end(more,grid,prefix,(at + 1 : U32),stop)) :
        {Con{Tasks.Span{0,size},_} == Tasks.whole_spans(Con{size,Nil{}}) : List<&2,Tasks.Span>}
      {==}
    case Con{+head,+more} 1n+ +later:
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      +cap = Machine.capacity(limits)
      +cap_ok = LimitBounds.capacity_fits(limits)
      +size = U32.max(per,1)
      +keeps = kept_lt(at,prefix,position,later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +kept = at_kept(at,prefix,position,1n+later,exact,prefix_distance)
      +next = kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,prefix_distance,prefix_room)
      +inner_extents = Tasks.piece_extents(more,prefix,size,(at + 1 : U32))
      +cells_inner = ShapeLaws.elements(inner_extents)
      +shape = kept_extents(head,more,prefix,size,at,keeps)
      +head_filled = {%shape : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} filled :
        {Nat.is_lt(0n,Nat.mul(U32.to_nat(head),cells_inner)) == True{} : Bool}}
      +head_fits = {%shape : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(cap)) == True{} : Bool} fits :
        {Nat.is_le(Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap)) == True{} : Bool}}
      +head_positive = RowMajor.left_factor_positive(U32.to_nat(head),cells_inner,head_filled)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(head),cells_inner,head_filled)
      +inner_le = Held.nat_le_right_moved(cells_inner,Nat.mul(U32.to_nat(head),cells_inner),Nat.mul(cells_inner,U32.to_nat(head)),
        RowMajor.multiply_commutative(U32.to_nat(head),cells_inner),RowMajor.product_covers(cells_inner,U32.to_nat(head),head_positive))
      +head_room = Order.transitive(U32.to_nat(head),Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap),
        RowMajor.product_covers(U32.to_nat(head),cells_inner,inner_positive),head_fits)
      +rest = whole_blocks_spans(more,later,limits,count,prefix,outer,extent,unit_cells,block,per,(at + 1 : U32),1n+position,next,
        steps_next(position,1n+later,U32.to_nat(prefix),prefix_distance),prefix_room,reach,last,last_room,block_room,block_positive,inner_positive,
        Order.transitive(cells_inner,Nat.mul(U32.to_nat(head),cells_inner),U32.to_nat(cap),inner_le,head_fits))
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{head,more},prefix,size,at),Con{head,inner_extents},shape) :
        {Tasks.whole_blocks(Con{head,more},grid,prefix,at) == Tasks.whole_spans(_) : List<&2,Tasks.Span>}
      %Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(head,Grid.block(grid,at,head))},Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32))},Nil{}) ==
          Tasks.whole_spans(Con{head,inner_extents}) : List<&2,Tasks.Span>}
      %Equal.sym(U32,Tasks.target_blocks(head,Grid.block(grid,at,head)),head,
          kept_whole_count(limits,head,count,prefix,outer,extent,unit_cells,block,per,at,keeps,head_room,head_positive)) :
        {Con{Tasks.Span{0,_},Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32))} == Tasks.whole_spans(Con{head,inner_extents}) : List<&2,Tasks.Span>}
      %Equal.sym(List<&2,Tasks.Span>,Tasks.whole_blocks(more,grid,prefix,(at + 1 : U32)),Tasks.whole_spans(inner_extents),rest) :
        {Con{Tasks.Span{0,head},_} == Tasks.whole_spans(Con{head,inner_extents}) : List<&2,Tasks.Span>}
      {==}
    case Nil{} +remaining:
      Empty.absurd({Tasks.whole_blocks(Nil{},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,at) ==
        Tasks.whole_spans(Tasks.piece_extents(Nil{},prefix,U32.max(per,1),at)) : List<&2,Tasks.Span>},Order.below_zero(remaining,reach))

def whole_in(extents: List<&2,U32>) -> SpansIn(Tasks.whole_spans(extents),extents):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      Held.Both{Order.reflexive(U32.to_nat(extent)),whole_in(rest)}

def whole_counted(extents: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> Counted(Tasks.whole_spans(extents)):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      Held.Both{RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive),
        whole_counted(rest,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))}

def whole_box(extents: List<&2,U32>,+index: Nat,+below: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  InBox(Tasks.whole_spans(extents),extents,index):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,
        Order.le_lt_trans(0n,index,Nat.mul(U32.to_nat(extent),inner),Order.zero_le(index),below))
      Held.Both{Order.zero_le(Nat.div(index,inner)),Held.Both{DivisionOrder.quotient_below(index,U32.to_nat(extent),inner,inner_positive,below),
        whole_box(rest,Nat.mod(index,inner),DivisionOrder.remainder_small(index,inner,inner_positive))}}

# ---- The target cover ----

# Every component of a tree lists its target pieces as leaves of the target
# layout, and every leaf of the layout starts a span the components list.
def TargetCover(+tasks: U32,+components: Tasks.Components,+cells: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges) -> Data:
  Held.Both<Moved.RangesTargets(Tasks.component_ranges(tasks),layout,components,cells,grid),
    Placed.LeavesListed(layout,Moved.ranges_spans(Tasks.component_ranges(tasks),components,cells,grid,Held.Ended{}))>

# A component that lists a piece among its target pieces.
type Holder<-components: Tasks.Components,-grid: Grid.Grid,-cells: List<&2,U32>,-limit: Nat,-piece: U32> is Data:
  Holder{index: U32,inside: {Nat.is_lt(U32.to_nat(index),limit) == True{} : Bool},member: Moved.Member(Tasks.target_pieces(components,grid,cells,index),piece)}

def tasks_bound(+limits: Machine.Limits,+tasks: U32,+room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(tasks),Bits.power(32n)) == True{} : Bool}:
  Order.transitive(U32.to_nat(tasks),Bits.power(31n),Bits.power(32n),GridRanges.capacity_power(limits,U32.to_nat(tasks),room),Bits.power_monotone(31n,32n,{==}))

# A piece a component holds is listed by the target spans of the components.
def held_listed(+limits: Machine.Limits,+components: Tasks.Components,+cells: List<&2,U32>,+grid: Grid.Grid,+tasks: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +limit: Nat,+limit_le: {Nat.is_le(limit,U32.to_nat(tasks)) == True{} : Bool},+piece: U32,+holder: Holder<components,grid,cells,limit,piece>) ->
  Placed.Listed(Moved.ranges_spans(Tasks.component_ranges(tasks),components,cells,grid,Held.Ended{}),Grid.start(grid,piece)):
  Holder{+index,+inside,+member} = holder
  listed_tree(limits,components,cells,grid,32n,U32.is_le(tasks,1),0,tasks,{==},positive,tasks_bound(limits,tasks,room),room,index,
    Order.zero_le(U32.to_nat(index)),Order.lt_le(U32.to_nat(index),limit,U32.to_nat(tasks),inside,limit_le),piece,Held.Ended{},member)

# The target pieces of every component below count are leaves of the layout.
def TargetsBelow(+layout: Pieces.Ranges,+grid: Grid.Grid,+components: Tasks.Components,+cells: List<&2,U32>,count: Nat) -> Data:
  match count:
    case 0n:
      Unit
    case 1n+more:
      Held.Both<Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(more))),TargetsBelow(layout,grid,components,cells,more)>

def earlier_index(+index: Nat,+more: Nat,+inside: {Nat.is_lt(index,1n+more) == True{} : Bool},+other: {Nat.is_eq(index,more) == False{} : Bool}) ->
  {Nat.is_lt(index,more) == True{} : Bool}:
  strict_of(index,more,Division.lt_succ_le(index,more,inside),
    Equal.trans(Bool,Nat.is_eq(more,index),Nat.is_eq(index,more),False{},Order.equal_symmetric(more,index),other))

def targets_here(+layout: Pieces.Ranges,+grid: Grid.Grid,+components: Tasks.Components,+cells: List<&2,U32>,+more: Nat,+index: Nat,
  +same: {Nat.is_eq(index,more) == True{} : Bool},+here: Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(more)))) ->
  Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(index))):
  %Equal.sym(Nat,index,more,Order.equal_of_is_eq(index,more,same)) : Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(_)))
  here

def targets_lookup(+layout: Pieces.Ranges,+grid: Grid.Grid,+components: Tasks.Components,+cells: List<&2,U32>,count: Nat,+index: Nat,
  +inside: {Nat.is_lt(index,count) == True{} : Bool},+below: TargetsBelow(layout,grid,components,cells,count)) ->
  Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(index))):
  match count:
    case 0n:
      Empty.absurd(Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(index))),Order.below_zero(index,inside))
    case 1n+ +more:
      Held.Both{+here,+rest} = below
      Held.by_cases(Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,U32.from_nat(index))),Nat.is_eq(index,more),
        same => targets_here(layout,grid,components,cells,more,index,same,here),
        other => targets_lookup(layout,grid,components,cells,more,index,earlier_index(index,more,inside,other),rest))

# Every component of a tree of indices below limit lists target pieces that
# are leaves of the layout.
def ranges_targets(+limits: Machine.Limits,+components: Tasks.Components,+cells: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,depth: Nat,single: Bool,
  +first: U32,+count: U32,+test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+limit: Nat,
  +within: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),limit) == True{} : Bool},+below: TargetsBelow(layout,grid,components,cells,limit)) ->
  Moved.RangesTargets(Tasks.index_ranges(depth,single,first,count),layout,components,cells,grid):
  match depth single:
    case 0n +single:
      %Arithmetic.native_roundtrip(first) : Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,_))
      targets_lookup(layout,grid,components,cells,limit,U32.to_nat(first),first_inside(U32.to_nat(first),U32.to_nat(count),limit,positive,within),below)
    case 1n+ +more True{}:
      %Arithmetic.native_roundtrip(first) : Moved.TargetsAt(layout,grid,Tasks.target_pieces(components,grid,cells,_))
      targets_lookup(layout,grid,components,cells,limit,U32.to_nat(first),first_inside(U32.to_nat(first),U32.to_nat(count),limit,positive,within),below)
    case 1n+ +more False{}:
      +cap = Machine.capacity(limits)
      +start = U32.to_nat(first)
      +total = U32.to_nat(count)
      +inside = Order.transitive(total,Nat.add(start,total),U32.to_nat(cap),Order.right_le(start,total),room)
      +halves = GridRanges.halves_of(cap,LimitBounds.capacity_allocatable(limits),more,count,test,positive,bounded,inside)
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +left_count = GridRanges.halves_left_count(count,more,halves)
      +left_room = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,U32.to_nat(left),total,left_count),room)
      +left_within = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,total),limit,Order.offset_le(start,U32.to_nat(left),total,left_count),within)
      +first_exact = Arithmetic.add(first,left,cap,LimitBounds.capacity_fits(limits),left_room)
      +right_sum = GridRanges.split_sum(first,count,first_exact,GridRanges.halves_right_exact(count,more,halves),left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),U32.to_nat(cap),right_sum,room)
      +right_within = Held.nat_le_moved(Nat.add(U32.to_nat((first + left : U32)),U32.to_nat(right)),Nat.add(start,total),limit,right_sum,within)
      Held.Both{ranges_targets(limits,components,cells,grid,layout,more,U32.is_le(left,1),first,left,{==},GridRanges.halves_left_positive(count,more,halves),
          GridRanges.halves_left_bound(count,more,halves),left_room,limit,left_within,below),
        ranges_targets(limits,components,cells,grid,layout,more,U32.is_le(right,1),(first + left : U32),right,{==},GridRanges.halves_right_positive(count,more,halves),
          GridRanges.halves_right_bound(count,more,halves),right_room,limit,right_within,below)}

def cover_tree(+limits: Machine.Limits,+components: Tasks.Components,+cells: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+tasks: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +limit: Nat,+within: {Nat.is_le(U32.to_nat(tasks),limit) == True{} : Bool},+below: TargetsBelow(layout,grid,components,cells,limit)) ->
  Moved.RangesTargets(Tasks.component_ranges(tasks),layout,components,cells,grid):
  ranges_targets(limits,components,cells,grid,layout,32n,U32.is_le(tasks,1),0,tasks,{==},positive,tasks_bound(limits,tasks,room),room,limit,within,below)

def start_inside(+piece: Nat,+more: Nat,+limit: Nat,+room: {Nat.is_le(Nat.add(piece,1n+more),limit) == True{} : Bool}) -> {Nat.is_lt(piece,limit) == True{} : Bool}:
  Order.lt_le(piece,Nat.add(piece,1n+more),limit,below_plus(piece,1n+more,{==}),room)

def start_room(+piece: Nat,+more: Nat,+limit: Nat,+room: {Nat.is_le(Nat.add(piece,1n+more),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+piece,more),limit) == True{} : Bool}:
  %Addition.successor_right(piece,more) : {Nat.is_le(_,limit) == True{} : Bool}
  room

# The leaves of a grid's layout are listed when the start of every piece is.
def leaves_from(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+to: Held.Spans,
  +starts: Placed.StartsListed(GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),to)) ->
  Placed.LeavesListed(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),to):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +own = {%GridRanges.grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape) : Placed.LeavesListed(layout,_)
    Placed.own_listed(layout,Held.Ended{}) : Placed.LeavesListed(layout,GridRanges.piece_spans(grid,total,0n,Held.Ended{}))}
  Placed.leaves_moved(layout,GridRanges.piece_spans(grid,total,0n,Held.Ended{}),to,starts,own)

# ---- Components and their tasks ----

def separate_positive(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(lists_count(raw_lists(rank,readings)))) == True{} : Bool}:
  +lists = raw_lists(rank,readings)
  %Equal.sym(Nat,U32.to_nat(lists_count(lists)),count_nat(lists),separate_tasks(readings,rank,limits,lists_count(lists),components_ok,{==})) :
    {Nat.is_lt(0n,_) == True{} : Bool}
  lists_positive

def separate_room(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits))) ->
  {Nat.is_le(U32.to_nat(lists_count(raw_lists(rank,readings))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +lists = raw_lists(rank,readings)
  %Equal.sym(Nat,U32.to_nat(lists_count(lists)),count_nat(lists),separate_tasks(readings,rank,limits,lists_count(lists),components_ok,{==})) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  components_within(readings,rank,limits,components_ok)

def whole_positive(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(U32.min(lists_count(raw_lists(rank,readings)),1))) == True{} : Bool}:
  +whole = lists_count(raw_lists(rank,readings))
  %Equal.sym(Nat,U32.to_nat(U32.min(whole,1)),Nat.min(U32.to_nat(whole),1n),Extrema.minimum(whole,1)) : {Nat.is_lt(0n,_) == True{} : Bool}
  NatExtrema.minimum_strict_lower(U32.to_nat(whole),1n,0n,separate_positive(readings,rank,limits,components_ok,lists_positive),{==})

def whole_room(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+limits: Machine.Limits,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits))) ->
  {Nat.is_le(U32.to_nat(U32.min(lists_count(raw_lists(rank,readings)),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +whole = lists_count(raw_lists(rank,readings))
  Order.transitive(U32.to_nat(U32.min(whole,1)),U32.to_nat(whole),U32.to_nat(Machine.capacity(limits)),Extrema.minimum_below(whole,1),
    separate_room(readings,rank,limits,components_ok))

# ---- Without prefix axes ----

def none_pieces(+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+cells: List<&2,U32>,+spans: List<&2,Tasks.Span>,
  +none: {U32.is_zero(prefix) == True{} : Bool}) ->
  {Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},cells,spans) == Con{0,Nil{}} : List<&2,U32>}:
  %Equal.sym(Bool,U32.is_zero(prefix),True{},none) :
    {Bool.pick(List<&2,U32>,_,Con{0,Nil{}},Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,Tasks.piece_extents(cells,prefix,U32.max(per,1),0),
      Tasks.span_count(spans),Nil{})) == Con{0,Nil{}} : List<&2,U32>}
  {==}

def none_pieces_of(separate: Bool,+classes: List<&2,Tasks.Class>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +cells: List<&2,U32>,+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  {Tasks.target_when(separate,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},cells,prefix) == Con{0,Nil{}} : List<&2,U32>}:
  match separate:
    case True{}:
      none_pieces(count,prefix,outer,extent,unit_cells,block,per,cells,Tasks.target_spans(classes,prefix,0),none)
    case False{}:
      none_pieces(count,prefix,outer,extent,unit_cells,block,per,cells,
        Tasks.whole_blocks(cells,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,0),none)

def none_targets(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+separate: Bool,+axes: List<&2,U32>,
  +total_positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool},+index: U32) ->
  Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Tasks.target_pieces(Tasks.Components{readings,counted_all(lists),tasks,separate},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,index)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(List<&2,U32>,Tasks.target_when(separate,Tasks.chosen(counted_all(lists),index),grid,axes,prefix),Con{0,Nil{}},
      none_pieces_of(separate,Tasks.chosen(counted_all(lists),index),count,prefix,outer,extent,unit_cells,block,per,axes,none)) :
    Moved.TargetsAt(Pieces.ranges_of_grid(grid),grid,_)
  target_at(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,0,total_positive)

def none_holder(+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+separate: Bool,+axes: List<&2,U32>,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +one: {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == 1n : Nat},+none: {U32.is_zero(prefix) == True{} : Bool},+piece: Nat,
  +inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Holder<Tasks.Components{readings,counted_all(lists),tasks,separate},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,1n,U32.from_nat(piece)>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +below = {%one : {Nat.is_lt(piece,_) == True{} : Bool} inside : {Nat.is_lt(piece,1n) == True{} : Bool}}
  +zero = Order.antisymmetric(piece,0n,Division.lt_succ_le(piece,0n,below),Order.zero_le(piece))
  Holder{0,{==},
    {%Equal.sym(List<&2,U32>,Tasks.target_when(separate,Tasks.chosen(counted_all(lists),0),grid,axes,prefix),Con{0,Nil{}},
        none_pieces_of(separate,Tasks.chosen(counted_all(lists),0),count,prefix,outer,extent,unit_cells,block,per,axes,none)) :
        Moved.Member(_,U32.from_nat(piece))
      Placed.OnLeft{{%Equal.sym(Nat,piece,0n,zero) : {0 == U32.from_nat(_) : U32} {==} : {0 == U32.from_nat(piece) : U32}}} :
      Moved.Member(Tasks.target_pieces(Tasks.Components{readings,counted_all(lists),tasks,separate},grid,axes,0),U32.from_nat(piece))}}

def none_below(remaining: Nat,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+separate: Bool,+axes: List<&2,U32>,
  +total_positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  TargetsBelow(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.Components{readings,counted_all(lists),tasks,separate},axes,remaining):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      Held.Both{none_targets(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,readings,lists,tasks,separate,axes,total_positive,none,
          U32.from_nat(more)),
        none_below(more,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,readings,lists,tasks,separate,axes,total_positive,none)}

def none_leaves(remaining: Nat,+piece: Nat,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,
  +separate: Bool,+axes: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+one: {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == 1n : Nat},+none: {U32.is_zero(prefix) == True{} : Bool},
  +tasks_positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},
  +tasks_room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(piece,remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Placed.StartsListed(GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),
    Moved.ranges_spans(Tasks.component_ranges(tasks),Tasks.Components{readings,counted_all(lists),tasks,separate},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Held.Ended{})):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      Held.Both{held_listed(limits,Tasks.Components{readings,counted_all(lists),tasks,separate},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,tasks_positive,tasks_room,1n,
          Order.strict_follow(0n,U32.to_nat(tasks),tasks_positive),U32.from_nat(piece),
          none_holder(readings,lists,tasks,separate,axes,count,prefix,outer,extent,unit_cells,block,per,one,none,piece,start_inside(piece,more,total,room))),
        none_leaves(more,1n+piece,limits,readings,lists,tasks,separate,axes,count,prefix,outer,extent,unit_cells,block,per,one,none,tasks_positive,tasks_room,
          start_room(piece,more,total,room))}

# Without prefix axes the grid is one piece every component lists.
def cover_none(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+separate: Bool,
  +axes: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +tasks_positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},
  +tasks_room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  TargetCover(tasks,Tasks.Components{readings,counted_all(lists),tasks,separate},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +components = {Tasks.Components{readings,counted_all(lists),tasks,separate} : Tasks.Components}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +one = prefix_zero_total(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,none)
  +total_positive = {%Equal.sym(Nat,total,1n,one) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,total) == True{} : Bool}}
  Held.Both{cover_tree(limits,components,axes,grid,layout,tasks,tasks_positive,tasks_room,U32.to_nat(tasks),Order.reflexive(U32.to_nat(tasks)),
      none_below(U32.to_nat(tasks),limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,readings,lists,tasks,separate,axes,total_positive,none)),
    leaves_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Moved.ranges_spans(Tasks.component_ranges(tasks),components,axes,grid,Held.Ended{}),
      none_leaves(total,0n,limits,readings,lists,tasks,separate,axes,count,prefix,outer,extent,unit_cells,block,per,one,none,tasks_positive,tasks_room,
        Order.reflexive(total)))}

# ---- With prefix axes ----

def shape_outer(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool}:
  GridLaws.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+room} = shape
  outer_positive

def per_last(+limits: Machine.Limits,+axes: List<&2,U32>,+steps: Nat,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+last: {extent_nat(axes,steps) == extent : U32}) ->
  {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(axes,steps)),U32.to_nat(block)) : Nat}:
  +per_kept = GridLaws.maximum_kept(per,1,Order.positive_one(U32.to_nat(per),GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape)))
  %Equal.sym(U32,U32.max(per,1),per,per_kept) : {U32.to_nat(_) == NatCeiling.ceiling(U32.to_nat(extent_nat(axes,steps)),U32.to_nat(block)) : Nat}
  %Equal.sym(U32,extent_nat(axes,steps),extent,last) : {U32.to_nat(per) == NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(block)) : Nat}
  GridLaws.shape_per(limits,outer,extent,unit_cells,block,per,shape)

def below_extents(+axes: List<&2,U32>,+prefix: U32,+per: U32,+outer: U32,+piece: Nat,
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.is_lt(piece,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) == True{} : Bool}:
  %Equal.sym(Nat,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),total_exact) :
    {Nat.is_lt(piece,_) == True{} : Bool}
  inside

def within_of(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+both: Held.Both<SpansIn(spans,extents),Counted(spans)>) -> SpansIn(spans,extents):
  Held.Both{+within,+counted} = both
  within

def targets_both(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +axes: List<&2,U32>,+spans: List<&2,Tasks.Span>,+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +both: Held.Both<SpansIn(spans,Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),Counted(spans)>) ->
  Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,spans)):
  Held.Both{+within,+counted} = both
  targets_in(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,axes,spans,some,total_exact,within,counted)

# The runtime's classes of a component inside the components, at half a word.
def chosen_bridge(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+index: U32,
  +lists_room: ListsRoom(raw_lists(rank,readings),Machine.half_word(limits)),
  +lists_double: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +index_inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  {Tasks.chosen(counted_all(raw_lists(rank,readings)),index) == chosen_word(raw_lists(rank,readings),index) : List<&2,Tasks.Class>}:
  chosen_exact(raw_lists(rank,readings),index,Machine.half_word(limits),LimitBounds.half_word_fits(limits),lists_room,lists_double,index_inside)

def holder_of_chosen(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  within: @+index: U32 -> @+index_inside: {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool} ->
    Held.Both<SpansIn(Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,0),Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),
      Counted(Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,0))>,
  +chosen: Chosen<raw_lists(rank,readings),prefix,0,Tasks.piece_extents(axes,prefix,U32.max(per,1),0),piece>,
  +lists_room: ListsRoom(raw_lists(rank,readings),Machine.half_word(limits)),
  +lists_double: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Holder<Tasks.Components{readings,counted_all(raw_lists(rank,readings)),lists_count(raw_lists(rank,readings)),True{}},
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,count_nat(raw_lists(rank,readings)),U32.from_nat(piece)>:
  Chosen{+index,+index_inside,+box} = chosen
  +spans = Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,0)
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  Holder{index,index_inside,
    {%Equal.sym(List<&2,Tasks.Class>,Tasks.chosen(counted_all(raw_lists(rank,readings)),index),chosen_word(raw_lists(rank,readings),index),
        chosen_exact(raw_lists(rank,readings),index,Machine.half_word(limits),LimitBounds.half_word_fits(limits),lists_room,lists_double,index_inside)) :
      Moved.Member(Tasks.target_when(True{},_,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,prefix),U32.from_nat(piece))
    box_member(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,spans,some,total_exact,
      within_of(spans,extents,within(index,index_inside)),piece,box,inside) :
      Moved.Member(Tasks.target_when(True{},Tasks.chosen(counted_all(raw_lists(rank,readings)),index),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,prefix),U32.from_nat(piece))}}

def whole_holder(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+axes: List<&2,U32>,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +whole_eq: {Tasks.whole_blocks(axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,0) ==
    Tasks.whole_spans(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) : List<&2,Tasks.Span>},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Holder<Tasks.Components{readings,counted_all(lists),tasks,False{}},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,1n,U32.from_nat(piece)>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  Holder{0,{==},
    {%Equal.sym(List<&2,Tasks.Span>,Tasks.whole_blocks(axes,grid,prefix,0),Tasks.whole_spans(extents),whole_eq) : Moved.Member(Tasks.pieces_in(grid,axes,_),U32.from_nat(piece))
      box_member(limits,outer,extent,unit_cells,block,per,count,prefix,shape,axes,Tasks.whole_spans(extents),some,total_exact,whole_in(extents),piece,
        whole_box(extents,piece,below_extents(axes,prefix,per,outer,piece,total_exact,inside)),inside) :
      Moved.Member(Tasks.target_pieces(Tasks.Components{readings,counted_all(lists),tasks,False{}},grid,axes,0),U32.from_nat(piece))}}

def separate_below(remaining: Nat,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +steps: Nat,+distance: {Nat.add(0n,1n+steps) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +reach: {Nat.is_lt(steps,List.length(&2,U32,axes)) == True{} : Bool},+columns: ColumnsOk(rank,readings,Machine.half_word(limits)),
  +lists_room: ListsRoom(raw_lists(rank,readings),Machine.half_word(limits)),
  +lists_double: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +limits_at: LimitsAt(readings,rank,axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(axes,steps)),U32.to_nat(block)) : Nat},
  +filled: {Nat.is_lt(0n,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) == True{} : Bool},
  +fits: {Nat.is_le(ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(remaining,count_nat(raw_lists(rank,readings))) == True{} : Bool}) ->
  TargetsBelow(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.Components{readings,counted_all(raw_lists(rank,readings)),lists_count(raw_lists(rank,readings)),True{}},axes,remaining):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      +cap = Machine.half_word(limits)
      +inside = Order.succ_le_lt(more,count_nat(raw_lists(rank,readings)),room)
      +more_le = lt_to_le(more,count_nat(raw_lists(rank,readings)),inside)
      +word = Bounds.roundtrip(more,cap,LimitBounds.half_word_fits(limits),Order.transitive(more,count_nat(raw_lists(rank,readings)),U32.to_nat(cap),more_le,
        Order.transitive(count_nat(raw_lists(rank,readings)),Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(cap),Order.le_double(count_nat(raw_lists(rank,readings))),lists_double)))
      +index = U32.from_nat(more)
      +index_inside = {%Equal.sym(Nat,U32.to_nat(index),more,word) : {Nat.is_lt(_,count_nat(raw_lists(rank,readings))) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(index),count_nat(raw_lists(rank,readings))) == True{} : Bool}}
      Held.Both{{%Equal.sym(List<&2,Tasks.Class>,Tasks.chosen(counted_all(raw_lists(rank,readings)),index),chosen_word(raw_lists(rank,readings),index),chosen_bridge(limits,readings,rank,index,lists_room,lists_double,index_inside)) :
          Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.target_when(True{},_,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,prefix))
          targets_both(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,axes,Tasks.target_spans(chosen_word(raw_lists(rank,readings),index),prefix,0),
          some,total_exact,chosen_within(rank,readings,axes,steps,count,prefix,outer,extent,unit_cells,block,per,0,0n,index,Machine.half_word(limits),LimitBounds.half_word_fits(limits),{==},distance,LimitBounds.within_half(limits,U32.to_nat(prefix),prefix_room),reach,columns,lists_room,lists_double,index_inside,limits_at,last,filled,LimitBounds.within_half(limits,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),fits))) :
          Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.target_when(True{},Tasks.chosen(counted_all(raw_lists(rank,readings)),index),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,prefix))},
        separate_below(more,limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,cells,ordered,some,total_exact,steps,distance,prefix_room,reach,columns,lists_room,lists_double,limits_at,last,filled,fits,more_le)}

def separate_leaves(remaining: Nat,+piece: Nat,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +steps: Nat,+distance: {Nat.add(0n,1n+steps) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +reach: {Nat.is_lt(steps,List.length(&2,U32,axes)) == True{} : Bool},+columns: ColumnsOk(rank,readings,Machine.half_word(limits)),
  +lists_room: ListsRoom(raw_lists(rank,readings),Machine.half_word(limits)),
  +lists_double: {Nat.is_le(Nat.double(count_nat(raw_lists(rank,readings))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +limits_at: LimitsAt(readings,rank,axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),
  +last: {U32.to_nat(U32.max(per,1)) == NatCeiling.ceiling(U32.to_nat(extent_nat(axes,steps)),U32.to_nat(block)) : Nat},
  +filled: {Nat.is_lt(0n,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0))) == True{} : Bool},
  +fits: {Nat.is_le(ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +tasks_positive: {Nat.is_lt(0n,U32.to_nat(lists_count(raw_lists(rank,readings)))) == True{} : Bool},
  +tasks_room: {Nat.is_le(U32.to_nat(lists_count(raw_lists(rank,readings))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lists_le: {Nat.is_le(count_nat(raw_lists(rank,readings)),U32.to_nat(lists_count(raw_lists(rank,readings)))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(piece,remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Placed.StartsListed(GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),
    Moved.ranges_spans(Tasks.component_ranges(lists_count(raw_lists(rank,readings))),Tasks.Components{readings,counted_all(raw_lists(rank,readings)),lists_count(raw_lists(rank,readings)),True{}},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Held.Ended{})):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +inside = start_inside(piece,more,total,room)
      Held.Both{held_listed(limits,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),lists_count(raw_lists(rank,readings)),True{}},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},lists_count(raw_lists(rank,readings)),tasks_positive,tasks_room,count_nat(raw_lists(rank,readings)),lists_le,U32.from_nat(piece),
          holder_of_chosen(limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,some,total_exact,piece,inside,
            +index => +index_inside => chosen_within(rank,readings,axes,steps,count,prefix,outer,extent,unit_cells,block,per,0,0n,index,Machine.half_word(limits),LimitBounds.half_word_fits(limits),{==},distance,LimitBounds.within_half(limits,U32.to_nat(prefix),prefix_room),reach,columns,lists_room,lists_double,index_inside,limits_at,last,filled,LimitBounds.within_half(limits,ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)),fits)),
            chosen_walk(rank,readings,axes,steps,count,prefix,outer,extent,unit_cells,block,per,0,0n,piece,Machine.half_word(limits),LimitBounds.half_word_fits(limits),{==},
              distance,LimitBounds.within_half(limits,U32.to_nat(prefix),prefix_room),reach,columns,lists_room,lists_double,lists_positive,limits_at,last,below_extents(axes,prefix,per,outer,piece,total_exact,inside)),
            lists_room,lists_double)),
        separate_leaves(more,1n+piece,limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,cells,ordered,some,total_exact,steps,distance,prefix_room,reach,columns,lists_room,lists_double,limits_at,last,filled,fits,lists_positive,tasks_positive,tasks_room,lists_le,start_room(piece,more,total,room))}

# Separate components cover the pieces of a grid with prefix axes.
def cover_separate(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+target: U32,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +limits_at: LimitsAt(readings,rank,axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  TargetCover(lists_count(raw_lists(rank,readings)),Tasks.Components{readings,counted_all(raw_lists(rank,readings)),lists_count(raw_lists(rank,readings)),True{}},
    axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})):
  Held.Both{+rows,+more} = components_ok
  Held.Both{+columns,+lists_more} = more
  Held.Both{+lists_room,+rooms} = lists_more
  Held.Both{+lists_double,+rows_room} = rooms
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +lists = raw_lists(rank,readings)
  +tasks = lists_count(lists)
  +components = {Tasks.Components{readings,counted_all(lists),tasks,True{}} : Tasks.Components}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +distance = Order.positive_has_last(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some))
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,axes),U32.to_nat(cap),
    shape_within(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,cap_ok,rank_room),rank_room)
  +reach = prefix_steps_in(axes,target,count,prefix,outer,extent,unit_cells,block,per,cap,cap_ok,exact,rank_room,some)
  +pieces = shape_pieces(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some)
  +total_exact = pieces_total(axes,prefix,per,outer,extent,pieces)
  +last = per_last(limits,axes,steps,outer,extent,unit_cells,block,per,shape,pieces_last(axes,prefix,per,outer,extent,pieces))
  +back = Equal.sym(Nat,ShapeLaws.elements(extents),total,total_exact)
  +filled = {%back : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(outer),U32.to_nat(per),shape_outer(limits,outer,extent,unit_cells,block,per,shape),
      GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}}
  +fits = {%back : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool}}
  +tasks_exact = separate_tasks(readings,rank,limits,tasks,components_ok,{==})
  +tasks_le = {%Equal.sym(Nat,U32.to_nat(tasks),count_nat(lists),tasks_exact) : {Nat.is_le(_,count_nat(lists)) == True{} : Bool} Order.reflexive(count_nat(lists)) :
    {Nat.is_le(U32.to_nat(tasks),count_nat(lists)) == True{} : Bool}}
  +lists_le = {%Equal.sym(Nat,U32.to_nat(tasks),count_nat(lists),tasks_exact) : {Nat.is_le(count_nat(lists),_) == True{} : Bool} Order.reflexive(count_nat(lists)) :
    {Nat.is_le(count_nat(lists),U32.to_nat(tasks)) == True{} : Bool}}
  +tasks_positive = separate_positive(readings,rank,limits,components_ok,lists_positive)
  +tasks_room = separate_room(readings,rank,limits,components_ok)
  Held.Both{cover_tree(limits,components,axes,grid,layout,tasks,tasks_positive,tasks_room,count_nat(lists),tasks_le,
      separate_below(count_nat(lists),limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,cells,ordered,some,total_exact,steps,distance,prefix_room,reach,columns,lists_room,lists_double,limits_at,last,filled,fits,Order.reflexive(count_nat(lists)))),
    leaves_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Moved.ranges_spans(Tasks.component_ranges(tasks),components,axes,grid,Held.Ended{}),
      separate_leaves(total,0n,limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,cells,ordered,some,total_exact,steps,distance,prefix_room,reach,columns,lists_room,lists_double,limits_at,last,filled,fits,lists_positive,tasks_positive,tasks_room,lists_le,Order.reflexive(total)))}

def whole_below(remaining: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,+axes: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +whole_targets: Moved.TargetsAt(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},axes,Tasks.whole_blocks(axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,0)))) ->
  TargetsBelow(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Tasks.Components{readings,counted_all(lists),tasks,False{}},axes,remaining):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      Held.Both{whole_targets,whole_below(more,readings,lists,tasks,axes,count,prefix,outer,extent,unit_cells,block,per,whole_targets)}

def whole_leaves(remaining: Nat,+piece: Nat,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+lists: List<&2,List<&2,Tasks.Class>>,+tasks: U32,
  +axes: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+some: {U32.is_zero(prefix) == False{} : Bool},
  +total_exact: {ShapeLaws.elements(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat},
  +whole_eq: {Tasks.whole_blocks(axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},prefix,0) == Tasks.whole_spans(Tasks.piece_extents(axes,prefix,U32.max(per,1),0)) : List<&2,Tasks.Span>},
  +tasks_positive: {Nat.is_lt(0n,U32.to_nat(tasks)) == True{} : Bool},
  +tasks_room: {Nat.is_le(U32.to_nat(tasks),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(piece,remaining),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  Placed.StartsListed(GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},remaining,piece,Held.Ended{}),Moved.ranges_spans(Tasks.component_ranges(tasks),Tasks.Components{readings,counted_all(lists),tasks,False{}},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Held.Ended{})):
  match remaining:
    case 0n:
      Unit{}
    case 1n+ +more:
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      Held.Both{held_listed(limits,Tasks.Components{readings,counted_all(lists),tasks,False{}},axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},tasks,tasks_positive,tasks_room,1n,Order.strict_follow(0n,U32.to_nat(tasks),tasks_positive),U32.from_nat(piece),
          whole_holder(limits,readings,lists,tasks,axes,count,prefix,outer,extent,unit_cells,block,per,shape,some,total_exact,whole_eq,piece,
            start_inside(piece,more,total,room))),
        whole_leaves(more,1n+piece,limits,readings,lists,tasks,axes,count,prefix,outer,extent,unit_cells,block,per,shape,some,total_exact,whole_eq,
          tasks_positive,tasks_room,start_room(piece,more,total,room))}

# One component of everything covers the pieces of a grid with prefix axes.
def cover_whole(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+target: U32,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +exact: {Grid.grid(axes,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  TargetCover(U32.min(lists_count(raw_lists(rank,readings)),1),
    Tasks.Components{readings,counted_all(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1),False{}},
    axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +layout = Pieces.ranges_of_grid(grid)
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +lists = raw_lists(rank,readings)
  +tasks = U32.min(lists_count(lists),1)
  +components = {Tasks.Components{readings,counted_all(lists),tasks,False{}} : Tasks.Components}
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +extents = Tasks.piece_extents(axes,prefix,U32.max(per,1),0)
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +distance = Order.positive_has_last(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some))
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,axes),U32.to_nat(cap),
    shape_within(axes,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,cap_ok,rank_room),rank_room)
  +reach = prefix_steps_in(axes,target,count,prefix,outer,extent,unit_cells,block,per,cap,cap_ok,exact,rank_room,some)
  +pieces = shape_pieces(limits,axes,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,some)
  +total_exact = pieces_total(axes,prefix,per,outer,extent,pieces)
  +last_extent = pieces_last(axes,prefix,per,outer,extent,pieces)
  +last = per_last(limits,axes,steps,outer,extent,unit_cells,block,per,shape,last_extent)
  +last_room = {%Equal.sym(U32,extent_nat(axes,steps),extent,last_extent) : {Nat.is_le(U32.to_nat(_),U32.to_nat(cap)) == True{} : Bool}
    GridLaws.extent_room(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_le(U32.to_nat(extent_nat(axes,steps)),U32.to_nat(cap)) == True{} : Bool}}
  +block_room = Order.transitive(U32.to_nat(block),U32.to_nat(extent),U32.to_nat(cap),GridLaws.shape_below(limits,outer,extent,unit_cells,block,per,shape),
    GridLaws.extent_room(limits,outer,extent,unit_cells,block,per,shape))
  +back = Equal.sym(Nat,ShapeLaws.elements(extents),total,total_exact)
  +filled = {%back : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(outer),U32.to_nat(per),shape_outer(limits,outer,extent,unit_cells,block,per,shape),
      GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}}
  +fits = {%back : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool}}
  +tasks_positive = whole_positive(readings,rank,limits,components_ok,lists_positive)
  +tasks_room = whole_room(readings,rank,limits,components_ok)
  +whole_eq = whole_blocks_spans(axes,steps,limits,count,prefix,outer,extent,unit_cells,block,per,0,0n,{==},distance,prefix_room,reach,last,last_room,block_room,
    GridLaws.shape_block(limits,outer,extent,unit_cells,block,per,shape),filled,fits)
  +whole_targets = {%Equal.sym(List<&2,Tasks.Span>,Tasks.whole_blocks(axes,grid,prefix,0),Tasks.whole_spans(extents),whole_eq) :
      Moved.TargetsAt(layout,grid,Tasks.pieces_in(grid,axes,_))
    targets_in(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,ordered,axes,Tasks.whole_spans(extents),some,total_exact,whole_in(extents),
      whole_counted(extents,filled)) :
    Moved.TargetsAt(layout,grid,Tasks.pieces_in(grid,axes,Tasks.whole_blocks(axes,grid,prefix,0)))}
  Held.Both{cover_tree(limits,components,axes,grid,layout,tasks,tasks_positive,tasks_room,U32.to_nat(tasks),Order.reflexive(U32.to_nat(tasks)),
      whole_below(U32.to_nat(tasks),readings,lists,tasks,axes,count,prefix,outer,extent,unit_cells,block,per,whole_targets)),
    leaves_from(limits,outer,extent,unit_cells,block,per,count,prefix,shape,Moved.ranges_spans(Tasks.component_ranges(tasks),components,axes,grid,Held.Ended{}),
      whole_leaves(total,0n,limits,readings,lists,tasks,axes,count,prefix,outer,extent,unit_cells,block,per,shape,some,total_exact,whole_eq,tasks_positive,tasks_room,
        Order.reflexive(total)))}

# The components of a moving operation cover the pieces of a grid.
def cover_grid(separate: Bool,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +exact: {Grid.grid(axes,Grid.piece_target()) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ordered: Placed.Ordered(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})),
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +limits_at: LimitsAt(readings,rank,axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0)) ->
  TargetCover(Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),
    Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),
      U32.min(lists_count(raw_lists(rank,readings)),1)),separate},
    axes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
    Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})):
  match separate:
    case True{}:
      +lists = raw_lists(rank,readings)
      +tasks = lists_count(lists)
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      Held.by_cases(TargetCover(tasks,Tasks.Components{readings,counted_all(lists),tasks,True{}},axes,grid,Pieces.ranges_of_grid(grid)),U32.is_zero(prefix),
        none => cover_none(limits,readings,lists,tasks,True{},axes,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,room,positive,
          rank_room,ordered,separate_positive(readings,rank,limits,components_ok,lists_positive),separate_room(readings,rank,limits,components_ok),none),
        some => cover_separate(limits,readings,rank,axes,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,room,positive,rank_room,
          ordered,components_ok,lists_positive,limits_at,some))
    case False{}:
      +lists = raw_lists(rank,readings)
      +tasks = U32.min(lists_count(lists),1)
      +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
      Held.by_cases(TargetCover(tasks,Tasks.Components{readings,counted_all(lists),tasks,False{}},axes,grid,Pieces.ranges_of_grid(grid)),U32.is_zero(prefix),
        none => cover_none(limits,readings,lists,tasks,False{},axes,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,room,positive,
          rank_room,ordered,whole_positive(readings,rank,limits,components_ok,lists_positive),whole_room(readings,rank,limits,components_ok),none),
        some => cover_whole(limits,readings,rank,axes,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,room,positive,rank_room,
          ordered,components_ok,lists_positive,some))

def cover_facts(+separate: Bool,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+grid: Grid.Grid,
  +layout: Pieces.Ranges,+grid_eq: {grid == Grid.grid(axes,Grid.piece_target()) : Grid.Grid},+layout_eq: {layout == Pieces.ranges_of_grid(grid) : Pieces.Ranges},
  +room: {Nat.is_le(ShapeLaws.elements(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(axes)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},+limits_at: LimitsAt(readings,rank,axes,grid,0),
  +facts: GridLaws.GridFacts<limits,Grid.grid(axes,Grid.piece_target())>) ->
  TargetCover(Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),
    Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),
      U32.min(lists_count(raw_lists(rank,readings)),1)),separate},axes,grid,layout):
  GridLaws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +whole = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +tasks = Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1))
  +components = {Tasks.Components{readings,counted_all(raw_lists(rank,readings)),tasks,separate} : Tasks.Components}
  +same = Equal.trans(Grid.Grid,grid,Grid.grid(axes,Grid.piece_target()),whole,grid_eq,exact)
  %Equal.sym(Pieces.Ranges,layout,Pieces.ranges_of_grid(grid),layout_eq) : TargetCover(tasks,components,axes,grid,_)
  %Equal.sym(Grid.Grid,grid,whole,same) : TargetCover(tasks,components,axes,_,Pieces.ranges_of_grid(_))
  cover_grid(separate,limits,readings,rank,axes,count,prefix,outer,extent,unit_cells,block,per,shape,cells,exact,room,positive,rank_room,
    grid_ordered(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells),components_ok,lists_positive,
    {%same : LimitsAt(readings,rank,axes,_,0) limits_at : LimitsAt(readings,rank,axes,whole,0)})

# The components of a moving operation list every target piece as a leaf of
# the target layout, and every leaf of the layout starts a span they list.
def target_cover(+separate: Bool,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+axes: List<&2,U32>,+grid: Grid.Grid,
  +layout: Pieces.Ranges,+laid: LaidOk(limits,axes,grid,layout),+components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},+limits_at: LimitsAt(readings,rank,axes,grid,0)) ->
  TargetCover(Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),
    Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),
      U32.min(lists_count(raw_lists(rank,readings)),1)),separate},axes,grid,layout):
  Held.Both{+grid_eq,+rest} = laid
  Held.Both{+layout_eq,+shape_ok} = rest
  Held.Both{+room,+sizes} = shape_ok
  Held.Both{+positive,+rank_room} = sizes
  cover_facts(separate,limits,readings,rank,axes,grid,layout,grid_eq,layout_eq,room,positive,rank_room,components_ok,lists_positive,limits_at,
    GridRanges.grid_facts(limits,axes,Grid.piece_target(),room,positive))

# ---- A moving operation ----

def facts_ordered(+limits: Machine.Limits,+shape: List<&2,U32>,+facts: GridLaws.GridFacts<limits,Grid.grid(shape,Grid.piece_target())>) ->
  Placed.Ordered(Pieces.ranges_of_grid(Grid.grid(shape,Grid.piece_target()))):
  GridLaws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+grid_shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,Grid.grid(shape,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) :
    Placed.Ordered(Pieces.ranges_of_grid(_))
  grid_ordered(limits,outer,extent,unit_cells,block,per,count,prefix,grid_shape,cells)

# The ranges of the grid of a shape are ordered.
def shape_ordered(+limits: Machine.Limits,+shape: List<&2,U32>,+ok: ShapeOk(limits,shape)) -> Placed.Ordered(Pieces.ranges_of_grid(Grid.grid(shape,Grid.piece_target()))):
  facts_ordered(limits,shape,GridRanges.grid_facts(limits,shape,Grid.piece_target(),shape_room_of(limits,shape,ok),shape_positive_of(limits,shape,ok)))

def moved_any(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Moved.Models<Element> -> @+holdings: Moved.Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: Moved.KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: Moved.KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,
  +agrees: Agrees(context,components,origins,target,target_grid,fill),decision: Tasks.Decision,+count: U32,+layout: Pieces.Ranges,+models: Moved.Models<Element>,+gones: Moved.Gones,
  +static: Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones),+good: Moved.SourcesGood(~Element,~Property,origins,0,models,gones),
  +ready: Moved.BuildReady(Tasks.component_ranges(count),origins,components,gones),+cover: TargetCover(count,components,target,target_grid,layout),
  +ordered: Placed.Ordered(layout)) ->
  Placed.PlacedModel<Element,Target,layout,Tensors.moved_decided(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},
    decision,count,layout,fill,Moved.pack_sources(~Element,models))>:
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      Held.Both{+targets,+listed} = cover
      Moved.kept_model(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,components,origins,target,target_grid,fill,context,agrees,parallel,tasks,count,layout,
        models,gones,static,good,ready,targets,ordered,listed)

# The pieces a moving operation writes hold the target property: every
# component takes its pieces of every source and gathers its target pieces.
def moved_with_counted(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Moved.Models<Element> -> @+holdings: Moved.Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: Moved.KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: Moved.KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +separate: Bool,+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,
  +fill: Element,+context: Context,
  +agrees: Agrees(context,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),
    Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},origins,target,
    Tensors.grid_of(target),fill),
  +bytes_per_cell: F32,+operations: F32,+workers: U32,+models: Moved.Models<Element>,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),+separable: {Tasks.separable(readings) == separate : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,separate),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +target_ok: ShapeOk(limits,target),+limits_at: LimitsAt(readings,rank,target,Tensors.grid_of(target),0),
  +static: Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),
    Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},0,0n)),
  +good: Moved.SourcesGood(~Element,~Property,origins,0,models,gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),
    Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},0,0n))) ->
  Placed.PlacedModel<Element,Target,Pieces.ranges_of_grid(Tensors.grid_of(target)),
    Tensors.moved_with(~Element,~Context,~gather,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),
      Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},
      context,bytes_per_cell,operations,Moved.pack_sources(~Element,models),origins,target,fill,workers)>:
  +lists = raw_lists(rank,readings)
  +count = Bool.pick(U32,separate,lists_count(lists),U32.min(lists_count(lists),1))
  +components = {Tasks.Components{readings,counted_all(lists),count,separate} : Tasks.Components}
  +grid = Tensors.grid_of(target)
  +layout = Pieces.ranges_of_grid(grid)
  moved_any(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,components,origins,target,grid,fill,context,agrees,
    Tasks.decide_components(Tensors.count_or_zero(target),components,workers,bytes_per_cell,operations),count,layout,models,
    gones_upto(origins,components,0,0n),static,good,build_ready(limits,origins,readings,rank,separate,components_ok,separable,origins_ok,lists_positive),
    target_cover(separate,limits,readings,rank,target,grid,layout,Held.Both{{==},Held.Both{{==},target_ok}},components_ok,lists_positive,limits_at),
    shape_ordered(limits,target,target_ok))
# The components the runtime counts are the counted model components.
def runtime_components(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+separate: Bool) -> {Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate} == Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate} : Tasks.Components}:
  Equal.trans(Tasks.Components,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,Tasks.count_of(counted_all(raw_lists(rank,readings))),U32.min(Tasks.count_of(counted_all(raw_lists(rank,readings))),1)),separate},Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},
    Equal.cong(List<&2,Tasks.Classes>,Tasks.Components,lists => Tasks.Components{readings,lists,Bool.pick(U32,separate,Tasks.count_of(lists),U32.min(Tasks.count_of(lists),1)),separate},
      Tasks.class_lists(rank,readings),counted_all(raw_lists(rank,readings)),lists_counted(rank,readings)),
    Equal.cong(U32,Tasks.Components,count => Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,count,U32.min(count,1)),separate},
      Tasks.count_of(counted_all(raw_lists(rank,readings))),lists_count(raw_lists(rank,readings)),count_counted(raw_lists(rank,readings))))

# A moving operation whose gathers keep Kept from one target piece to the
# next writes pieces with the target property.
def kept_with_model(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,~Kept: @+grid: Grid.Grid -> @+models: Moved.Models<Element> -> @+holdings: Moved.Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: Moved.KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: Moved.KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +separate: Bool,
  +limits: Machine.Limits,
  +readings: List<&2,List<&2,Tasks.Reading>>,
  +rank: Nat,
  +origins: List<&2,Tensors.Origin>,
  +target: List<&2,U32>,
  +fill: Element,
  +context: Context,
  +agrees: Agrees(context,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},origins,target,Tensors.grid_of(target),fill),
  +bytes_per_cell: F32,
  +operations: F32,
  +workers: U32,
  +models: Moved.Models<Element>,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +separable: {Tasks.separable(readings) == separate : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,separate),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +target_ok: ShapeOk(limits,target),
  +limits_at: LimitsAt(readings,rank,target,Tensors.grid_of(target),0),
  +static: Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones_upto(origins,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},0,0n)),
  +good: Moved.SourcesGood(~Element,~Property,origins,0,models,gones_upto(origins,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},0,0n))) ->
  Placed.PlacedModel<Element,Target,Pieces.ranges_of_grid(Tensors.grid_of(target)),Tensors.moved_with(~Element,~Context,~gather,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},context,bytes_per_cell,operations,Moved.pack_sources(~Element,models),origins,target,fill,workers)>:
  +same = runtime_components(readings,rank,separate)
  %Equal.sym(Tasks.Components,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},same) :
    Placed.PlacedModel<Element,Target,Pieces.ranges_of_grid(Tensors.grid_of(target)),
      Tensors.moved_with(~Element,~Context,~gather,_,context,bytes_per_cell,operations,Moved.pack_sources(~Element,models),origins,target,fill,workers)>
  moved_with_counted(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,separate,limits,readings,rank,origins,target,fill,context,{%same : Agrees(context,_,origins,target,Tensors.grid_of(target),fill) agrees : Agrees(context,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},origins,target,Tensors.grid_of(target),fill)},bytes_per_cell,operations,workers,models,components_ok,separable,origins_ok,lists_positive,target_ok,limits_at,{%same : Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones_upto(origins,_,0,0n)) static : Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},0,0n))},{%same : Moved.SourcesGood(~Element,~Property,origins,0,models,gones_upto(origins,_,0,0n)) good : Moved.SourcesGood(~Element,~Property,origins,0,models,gones_upto(origins,Tasks.Components{readings,counted_all(raw_lists(rank,readings)),Bool.pick(U32,separate,lists_count(raw_lists(rank,readings)),U32.min(lists_count(raw_lists(rank,readings)),1)),separate},0,0n))})

# A moving operation whose gathers keep the holdings in order.
def moved_with_model(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,~gathers: Moved.GatherLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees),
  +separate: Bool,
  +limits: Machine.Limits,
  +readings: List<&2,List<&2,Tasks.Reading>>,
  +rank: Nat,
  +origins: List<&2,Tensors.Origin>,
  +target: List<&2,U32>,
  +fill: Element,
  +context: Context,
  +agrees: Agrees(context,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},origins,target,Tensors.grid_of(target),fill),
  +bytes_per_cell: F32,
  +operations: F32,
  +workers: U32,
  +models: Moved.Models<Element>,
  +components_ok: ComponentsOk(readings,rank,Machine.half_word(limits)),
  +separable: {Tasks.separable(readings) == separate : Bool},
  +origins_ok: OriginsOk(limits,origins,readings,0,separate),
  +lists_positive: {Nat.is_lt(0n,count_nat(raw_lists(rank,readings))) == True{} : Bool},
  +target_ok: ShapeOk(limits,target),
  +limits_at: LimitsAt(readings,rank,target,Tensors.grid_of(target),0),
  +static: Moved.SourcesStatic(~Element,~Property,origins,0,fill,models,gones_upto(origins,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},0,0n)),
  +good: Moved.SourcesGood(~Element,~Property,origins,0,models,gones_upto(origins,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},0,0n))) ->
  Placed.PlacedModel<Element,Target,Pieces.ranges_of_grid(Tensors.grid_of(target)),Tensors.moved_with(~Element,~Context,~gather,Tasks.Components{readings,Tasks.class_lists(rank,readings),Bool.pick(U32,separate,Tasks.count_of(Tasks.class_lists(rank,readings)),U32.min(Tasks.count_of(Tasks.class_lists(rank,readings)),1)),separate},context,bytes_per_cell,operations,Moved.pack_sources(~Element,models),origins,target,fill,workers)>:
  kept_with_model(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Moved.ordered_kept(~Element),
    ~Moved.ordered_law(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,gathers),~Moved.ordered_begin(~Element,~Context,~Property,~Agrees),
    separate,limits,readings,rank,origins,target,fill,context,agrees,bytes_per_cell,operations,workers,models,components_ok,separable,origins_ok,lists_positive,
    target_ok,limits_at,static,good)

