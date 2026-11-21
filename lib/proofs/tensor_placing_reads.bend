# The source cells the parts of a placing read lie in the pieces its
# component holds: every segment of an axis reads the source index its reading
# keeps for the target index, inside the footprint of the class of the target
# index.

import Base
import ../tensor_tasks.bend as Tasks
import ./tensor_task_reads.bend as TaskReads
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./nat_extrema.bend as NatExtrema
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./u32_extrema.bend as Extrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_moved.bend as Moved
import ./tensor_box_cells.bend as BoxCells
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_view_reads.bend as ViewReads
import ./tensor_target_cells.bend as Targets
import ./tensor_components.bend as Components
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_grid.bend as Grid

# ---- The index a segment reads ----

# How a segment of an axis reads a source its reading keeps from shift on,
# below bound: in the middle it reads every target index less the shift;
# clamped, before the shift it repeats the first index and after the kept
# indices the last; an empty segment reads nothing.
type SegmentRead<-first: U32,-count: U32,-source: U32,-step: U32,-shift: U32,-bound: U32,-clamped: Bool> is Data:
  Middle{step_one: {step == 1 : U32},placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},
    inside: {Nat.is_le(Nat.add(U32.to_nat(source),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool}}
  Before{clamp: {clamped == True{} : Bool},step_zero: {step == 0 : U32},source_zero: {source == 0 : U32},positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
    ends: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),1n+U32.to_nat(shift)) == True{} : Bool}}
  After{clamp: {clamped == True{} : Bool},step_zero: {step == 0 : U32},source_last: {Nat.add(U32.to_nat(source),1n) == U32.to_nat(bound) : Nat},
    starts: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n+U32.to_nat(first)) == True{} : Bool}}
  Unused{none: {count == 0 : U32}}

# A word clamped to the indices a read keeps.
def clamp_exact(+word: U32,+shift: U32,+bound: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool}) ->
  {U32.to_nat(Tasks.clamp(word,shift,(shift + bound - 1 : U32))) == Nat.min(Nat.max(U32.to_nat(word),U32.to_nat(shift)),Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n)) : Nat}:
  +top = (shift + bound - 1 : U32)
  +last = Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n)
  +widest = Nat.max(U32.to_nat(word),U32.to_nat(shift))
  Equal.trans(Nat,U32.to_nat(Tasks.clamp(word,shift,top)),Nat.min(U32.to_nat(U32.max(word,shift)),U32.to_nat(top)),Nat.min(widest,last),Extrema.minimum(U32.max(word,shift),top),
    Equal.trans(Nat,Nat.min(U32.to_nat(U32.max(word,shift)),U32.to_nat(top)),Nat.min(widest,U32.to_nat(top)),Nat.min(widest,last),
      Equal.cong(Nat,Nat,found => Nat.min(found,U32.to_nat(top)),U32.to_nat(U32.max(word,shift)),widest,Extrema.maximum(word,shift)),
      Equal.cong(Nat,Nat,found => Nat.min(widest,found),U32.to_nat(top),last,TaskReads.top_exact(shift,bound,cap,cap_ok,shift_room,positive))))

# The end of a middle segment lies inside cap.
def middle_end(+first: U32,+count: U32,+source: U32,+shift: U32,+bound: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(source),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}:
  +s = U32.to_nat(shift)
  +o = U32.to_nat(source)
  +c = U32.to_nat(count)
  +order = Equal.trans(Nat,Nat.add(Nat.add(o,s),c),Nat.add(Nat.add(s,o),c),Nat.add(s,Nat.add(o,c)),
    Equal.cong(Nat,Nat,sum => Nat.add(sum,c),Nat.add(o,s),Nat.add(s,o),Order.commutative(o,s)),Addition.associative(s,o,c))
  %Equal.sym(Nat,U32.to_nat(first),Nat.add(o,s),placed) : {Nat.is_le(Nat.add(_,c),U32.to_nat(cap)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(o,s),c),Nat.add(s,Nat.add(o,c)),order) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Order.transitive(Nat.add(s,Nat.add(o,c)),Nat.add(s,U32.to_nat(bound)),U32.to_nat(cap),Order.offset_le(s,Nat.add(o,c),U32.to_nat(bound),inside),shift_room)

# In the middle a segment reads the source index of the target index less its
# first index past its source index.
def middle_word(+first: U32,+count: U32,+source: U32,+shift: U32,+bound: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(source),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) ->
  {U32.to_nat((source + (index - first) * 1 : U32)) == Nat.add(U32.to_nat(source),Nat.sub(U32.to_nat(index),U32.to_nat(first))) : Nat}:
  +b = U32.to_nat(bound)
  +f = U32.to_nat(first)
  +c = U32.to_nat(count)
  +o = U32.to_nat(source)
  +i = U32.to_nat(index)
  +d = Nat.sub(i,f)
  +v = Nat.add(o,d)
  +bound_room = Order.transitive(b,Nat.add(U32.to_nat(shift),b),U32.to_nat(cap),
    {%Order.commutative(b,U32.to_nat(shift)) : {Nat.is_le(b,_) == True{} : Bool} Order.le_add(b,U32.to_nat(shift)) : {Nat.is_le(b,Nat.add(U32.to_nat(shift),b)) == True{} : Bool}},shift_room)
  +v_below = Bounds.nat_trans(v,Nat.add(o,c),b,Order.offset_lt(o,d,c,TaskReads.below_after(f,i,c,after,before)),inside)
  +v_room = Order.transitive(v,b,U32.to_nat(cap),Bounds.strict_le(v,b,v_below),bound_room)
  +i_room = Order.transitive(i,Nat.add(f,c),U32.to_nat(cap),Bounds.strict_le(i,Nat.add(f,c),before),middle_end(first,count,source,shift,bound,cap,cap_ok,shift_room,placed,inside))
  +difference = Arithmetic.difference_exact(cap,cap_ok,index,first,i,f,{==},{==},after,i_room)
  +once = Equal.trans(Nat,U32.to_nat(((index - first : U32) * 1 : U32)),U32.to_nat((index - first : U32)),d,
    Equal.cong(U32,Nat,word => U32.to_nat(word),((index - first : U32) * 1 : U32),(index - first : U32),Arithmetic.multiply_one((index - first : U32))),difference)
  Arithmetic.sum_exact(cap,cap_ok,source,((index - first : U32) * 1 : U32),o,d,{==},once,v_room)

# In the middle a segment reads below the bound.
def middle_below(+first: U32,+count: U32,+source: U32,+shift: U32,+bound: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(source),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat((source + (index - first) * 1 : U32)),U32.to_nat(bound)) == True{} : Bool}:
  +f = U32.to_nat(first)
  +o = U32.to_nat(source)
  +c = U32.to_nat(count)
  +d = Nat.sub(U32.to_nat(index),f)
  %Equal.sym(Nat,U32.to_nat((source + (index - first) * 1 : U32)),Nat.add(o,d),middle_word(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)) : {Nat.is_lt(_,U32.to_nat(bound)) == True{} : Bool}
  Bounds.nat_trans(Nat.add(o,d),Nat.add(o,c),U32.to_nat(bound),Order.offset_lt(o,d,c,TaskReads.below_after(f,U32.to_nat(index),c,after,before)),inside)

# In the middle the shift plus the source index a segment reads is the target
# index.
def middle_sum(+first: U32,+count: U32,+source: U32,+shift: U32,+bound: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(source),U32.to_nat(count)),U32.to_nat(bound)) == True{} : Bool},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) -> {Nat.add(U32.to_nat(shift),U32.to_nat((source + (index - first) * 1 : U32))) == U32.to_nat(index) : Nat}:
  +s = U32.to_nat(shift)
  +f = U32.to_nat(first)
  +o = U32.to_nat(source)
  +i = U32.to_nat(index)
  +d = Nat.sub(i,f)
  Equal.trans(Nat,Nat.add(s,U32.to_nat((source + (index - first) * 1 : U32))),Nat.add(s,Nat.add(o,d)),i,
    Equal.cong(Nat,Nat,value => Nat.add(s,value),U32.to_nat((source + (index - first) * 1 : U32)),Nat.add(o,d),middle_word(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)),
    Equal.trans(Nat,Nat.add(s,Nat.add(o,d)),Nat.add(Nat.add(s,o),d),i,Equal.sym(Nat,Nat.add(Nat.add(s,o),d),Nat.add(s,Nat.add(o,d)),Addition.associative(s,o,d)),
      Equal.trans(Nat,Nat.add(Nat.add(s,o),d),Nat.add(f,d),i,
        Equal.cong(Nat,Nat,sum => Nat.add(sum,d),Nat.add(s,o),f,Equal.trans(Nat,Nat.add(s,o),Nat.add(o,s),f,Order.commutative(s,o),Equal.sym(Nat,f,Nat.add(o,s),placed))),
        Order.restore_shift(f,i,after))))

# The shift lies at or before every index of a middle segment.
def middle_after(+first: U32,+source: U32,+shift: U32,+index: U32,+placed: {U32.to_nat(first) == Nat.add(U32.to_nat(source),U32.to_nat(shift)) : Nat},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(shift),U32.to_nat(index)) == True{} : Bool}:
  +s = U32.to_nat(shift)
  +o = U32.to_nat(source)
  Order.transitive(s,U32.to_nat(first),U32.to_nat(index),
    {%Equal.sym(Nat,U32.to_nat(first),Nat.add(o,s),placed) : {Nat.is_le(s,_) == True{} : Bool}
      %Order.commutative(s,o) : {Nat.is_le(s,_) == True{} : Bool} Order.le_add(s,o) : {Nat.is_le(s,U32.to_nat(first)) == True{} : Bool}},after)

# An empty segment holds no index.
def unused_absurd(+first: U32,+count: U32,+index: U32,+none: {count == 0 : U32},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+before: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) -> Empty:
  +f = U32.to_nat(first)
  +i = U32.to_nat(index)
  +below = {%none : {Nat.is_lt(i,Nat.add(f,U32.to_nat(_))) == True{} : Bool} before : {Nat.is_lt(i,Nat.add(f,0n)) == True{} : Bool}}
  WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(i,f),True{},Equal.sym(Bool,Nat.is_lt(i,f),False{},Order.not_below(f,i,after)),
    {%Addition.zero_right(f) : {Nat.is_lt(i,_) == True{} : Bool} below : {Nat.is_lt(i,f) == True{} : Bool}}))

# A word plus nothing stepped is the word.
def unstepped(+source: U32,+index: U32,+first: U32) -> {(source + (index - first) * 0 : U32) == source : U32}:
  Equal.trans(U32,(source + (index - first) * 0 : U32),(source + 0 : U32),source,
    Equal.cong(U32,U32,word => (source + word : U32),((index - first : U32) * 0 : U32),0,Arithmetic.multiply_zero_right((index - first : U32))),Arithmetic.add_zero(source))

# ---- The source index a read keeps ----

# The kept index of a coordinate clamped to the indices from shift on, below
# bound.
def clamped_index(+shift: Nat,+bound: Nat,+coordinate: Nat) -> Nat:
  Nat.min(Nat.max(coordinate,shift),Nat.sub(Nat.add(shift,bound),1n))

# The source index a read of shift and bound keeps for a target coordinate:
# the coordinate less the shift, clamped first when the read clamps.
def kept_index(clamped: Bool,+shift: Nat,+bound: Nat,+coordinate: Nat) -> Nat:
  match clamped:
    case False{}: Nat.sub(coordinate,shift)
    case True{}: Nat.sub(clamped_index(shift,bound,coordinate),shift)

# A read keeps a target coordinate: unclamped, the coordinate lies from the
# shift on below the shift plus the bound, clamped, the bound is not empty.
def KeptAt(clamped: Bool,+shift: U32,+bound: U32,+index: U32) -> Data:
  match clamped:
    case False{}: Held.Both<{Nat.is_le(U32.to_nat(shift),U32.to_nat(index)) == True{} : Bool},{Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(shift),U32.to_nat(bound))) == True{} : Bool}>
    case True{}: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool}

# A clamped index lies from the shift on.
def clamped_after(+shift: Nat,+bound: Nat,+coordinate: Nat,+positive: {Nat.is_lt(0n,bound) == True{} : Bool}) -> {Nat.is_le(shift,clamped_index(shift,bound,coordinate)) == True{} : Bool}:
  +last_after = {%Addition.zero_right(shift) : {Nat.is_le(_,Nat.sub(Nat.add(shift,bound),1n)) == True{} : Bool} TaskReads.value_top(shift,0n,bound,positive) : {Nat.is_le(shift,Nat.sub(Nat.add(shift,bound),1n)) == True{} : Bool}}
  NatExtrema.minimum_lower(Nat.max(coordinate,shift),Nat.sub(Nat.add(shift,bound),1n),shift,NatExtrema.maximum_right(coordinate,shift),last_after)

# A clamped index lies less the shift below the bound.
def clamped_below(+shift: Nat,+bound: Nat,+coordinate: Nat,+positive: {Nat.is_lt(0n,bound) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(clamped_index(shift,bound,coordinate),shift),bound) == True{} : Bool}:
  +sum_positive = Bounds.nat_trans(0n,bound,Nat.add(shift,bound),positive,Order.right_le(shift,bound))
  TaskReads.below_after(shift,clamped_index(shift,bound,coordinate),bound,clamped_after(shift,bound,coordinate,positive),
    Order.le_lt_trans(clamped_index(shift,bound,coordinate),Nat.sub(Nat.add(shift,bound),1n),Nat.add(shift,bound),NatExtrema.minimum_right(Nat.max(coordinate,shift),Nat.sub(Nat.add(shift,bound),1n)),
      Order.last_below(Nat.add(shift,bound),sum_positive)))

# Clamping keeps the order of coordinates.
def clamped_monotone(+shift: Nat,+bound: Nat,+left: Nat,+right: Nat,+ordered: {Nat.is_le(left,right) == True{} : Bool}) -> {Nat.is_le(clamped_index(shift,bound,left),clamped_index(shift,bound,right)) == True{} : Bool}:
  +widest = Nat.max(right,shift)
  NatExtrema.minimum_lower(widest,Nat.sub(Nat.add(shift,bound),1n),clamped_index(shift,bound,left),
    Order.transitive(clamped_index(shift,bound,left),Nat.max(left,shift),widest,NatExtrema.minimum_left(Nat.max(left,shift),Nat.sub(Nat.add(shift,bound),1n)),
      NatExtrema.maximum_upper(left,shift,widest,Order.transitive(left,right,widest,ordered,NatExtrema.maximum_left(right,shift)),NatExtrema.maximum_right(right,shift))),
    NatExtrema.minimum_right(Nat.max(left,shift),Nat.sub(Nat.add(shift,bound),1n)))

# A read that keeps a target coordinate between the virtual indices first and
# last keeps its kept index.
def kept_at(clamped: Bool,+low: U32,+high: U32,+shift: U32,+bound: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +low_le: {Nat.is_le(U32.to_nat(low),U32.to_nat(index)) == True{} : Bool},+high_ge: {Nat.is_le(U32.to_nat(index),U32.to_nat(high)) == True{} : Bool},+at: KeptAt(clamped,shift,bound,index)) ->
  TaskReads.KeptIn(clamped,low,high,shift,bound,kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index))):
  match clamped:
    case False{}:
      Held.Both{+after,+below} = at
      +s = U32.to_nat(shift)
      +b = U32.to_nat(bound)
      +i = U32.to_nat(index)
      +lo = U32.to_nat(low)
      +hi = U32.to_nat(high)
      +back = Equal.sym(Nat,Nat.add(s,Nat.sub(i,s)),i,Order.restore_shift(s,i,after))
      Held.Both{{%back : {Nat.is_le(lo,_) == True{} : Bool} low_le : {Nat.is_le(lo,Nat.add(s,Nat.sub(i,s))) == True{} : Bool}},
        Held.Both{{%back : {Nat.is_le(_,hi) == True{} : Bool} high_ge : {Nat.is_le(Nat.add(s,Nat.sub(i,s)),hi) == True{} : Bool}},TaskReads.below_after(s,i,b,after,below)}}
    case True{}:
      +s = U32.to_nat(shift)
      +b = U32.to_nat(bound)
      +i = U32.to_nat(index)
      +lo = U32.to_nat(low)
      +hi = U32.to_nat(high)
      +m = clamped_index(s,b,i)
      +back = Equal.sym(Nat,Nat.add(s,Nat.sub(m,s)),m,Order.restore_shift(s,m,clamped_after(s,b,i,at)))
      +top = (shift + bound - 1 : U32)
      +low_m = {%Equal.sym(Nat,U32.to_nat(Tasks.clamp(low,shift,top)),clamped_index(s,b,lo),clamp_exact(low,shift,bound,cap,cap_ok,shift_room,at)) :
          {Nat.is_le(_,Nat.add(s,Nat.sub(m,s))) == True{} : Bool}
        %back : {Nat.is_le(clamped_index(s,b,lo),_) == True{} : Bool}
        clamped_monotone(s,b,lo,i,low_le) : {Nat.is_le(U32.to_nat(Tasks.clamp(low,shift,top)),Nat.add(s,Nat.sub(m,s))) == True{} : Bool}}
      +m_high = {%Equal.sym(Nat,U32.to_nat(Tasks.clamp(high,shift,top)),clamped_index(s,b,hi),clamp_exact(high,shift,bound,cap,cap_ok,shift_room,at)) :
          {Nat.is_le(Nat.add(s,Nat.sub(m,s)),_) == True{} : Bool}
        %back : {Nat.is_le(_,clamped_index(s,b,hi)) == True{} : Bool}
        clamped_monotone(s,b,i,hi,high_ge) : {Nat.is_le(Nat.add(s,Nat.sub(m,s)),U32.to_nat(Tasks.clamp(high,shift,top))) == True{} : Bool}}
      Held.Both{low_m,Held.Both{m_high,clamped_below(s,b,i,at)}}

def kept_virtual(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +first: U32,+count: U32,+source_first: U32,+source_count: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+at: KeptAt(clamped,shift,bound,index),
  +virtual: Held.Both<{Nat.is_le(U32.to_nat(Tasks.virtual_first(1,0,False{},1,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32))),U32.to_nat(index)) == True{} : Bool},{Nat.is_le(U32.to_nat(index),U32.to_nat(Tasks.virtual_last(1,0,False{},1,1,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)))) == True{} : Bool}>) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,1,0,False{},1,1,shift,bound,clamped}},Tasks.Class{first,count,source_first,source_count},kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)),cap):
  Held.Both{+low_le,+high_ge} = virtual
  Held.Both{shift_room,kept_at(clamped,Tasks.virtual_first(1,0,False{},1,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),Tasks.virtual_last(1,0,False{},1,1,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)),shift,bound,index,cap,cap_ok,shift_room,low_le,high_ge,at)}

# A target axis that reads its source axis with unit step from a shift keeps
# at a target coordinate it keeps the kept index.
def kept_placed(+extent: U32,+block: U32,+axis: U32,+source_block: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +first: U32,+count: U32,+source_first: U32,+source_count: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: Components.ReadingRoom(extent,block,Tasks.Read{axis,source_block,1,0,False{},1,1,shift,bound,clamped},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,U32.to_nat(index)),+at: KeptAt(clamped,shift,bound,index)) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{axis,source_block,1,0,False{},1,1,shift,bound,clamped}},Tasks.Class{first,count,source_first,source_count},kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)),cap):
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+positive,Held.Both{+reach,Held.Both{+divisor_room,Held.Both{+shift_room,+block_rest}}}} = read_room
  +coordinate = U32.to_nat(index)
  kept_virtual(extent,block,axis,source_block,shift,bound,clamped,first,count,source_first,source_count,index,cap,cap_ok,shift_room,at,
    TaskReads.read_virtual(extent,block,axis,source_block,1,0,1,1,shift,bound,clamped,first,count,source_first,source_count,coordinate,coordinate,cap,cap_ok,room,
      class_in,TaskReads.unit_above(coordinate),TaskReads.unit_under(coordinate)))

# ---- The index a segment reads is the index its read keeps ----

# A segment that holds a target coordinate and reads its source as the read
# keeps it reads the kept index, which the read keeps.
def segment_kept(clamped: Bool,+first: U32,+count: U32,+source: U32,+step: U32,+shift: U32,+bound: U32,+read: SegmentRead<first,count,source,step,shift,bound,clamped>,
  +index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) ->
  Held.Both<KeptAt(clamped,shift,bound,index),{U32.to_nat((source + (index - first) * step : U32)) == kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>:
  match clamped:
    case False{}:
      match read:
        case Middle{+step_one,+placed,+inside}:
          +s = U32.to_nat(shift)
          +b = U32.to_nat(bound)
          +i = U32.to_nat(index)
          +w = U32.to_nat((source + (index - first) * 1 : U32))
          +sum = middle_sum(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)
          +index_below = {%sum : {Nat.is_lt(_,Nat.add(s,b)) == True{} : Bool} Order.offset_lt(s,w,b,middle_below(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)) : {Nat.is_lt(i,Nat.add(s,b)) == True{} : Bool}}
          %Equal.sym(U32,step,1,step_one) : Held.Both<KeptAt(False{},shift,bound,index),{U32.to_nat((source + (index - first) * _ : U32)) == kept_index(False{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          Held.Both{Held.Both{middle_after(first,source,shift,index,placed,after),index_below},
            Equal.trans(Nat,w,Nat.sub(Nat.add(s,w),s),Nat.sub(i,s),Equal.sym(Nat,Nat.sub(Nat.add(s,w),s),w,Order.subtract_shift(s,w)),
              Equal.cong(Nat,Nat,value => Nat.sub(value,s),Nat.add(s,w),i,sum))}
        case Before{+clamp,+step_zero,+source_zero,+positive,+ends}:
          Empty.absurd(Held.Both<KeptAt(False{},shift,bound,index),{U32.to_nat((source + (index - first) * step : U32)) == kept_index(False{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>,WordEncoding.false_true(clamp))
        case After{+clamp,+step_zero,+source_last,+starts}:
          Empty.absurd(Held.Both<KeptAt(False{},shift,bound,index),{U32.to_nat((source + (index - first) * step : U32)) == kept_index(False{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>,WordEncoding.false_true(clamp))
        case Unused{+none}:
          Empty.absurd(Held.Both<KeptAt(False{},shift,bound,index),{U32.to_nat((source + (index - first) * step : U32)) == kept_index(False{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>,unused_absurd(first,count,index,none,after,before))
    case True{}:
      match read:
        case Middle{+step_one,+placed,+inside}:
          +s = U32.to_nat(shift)
          +b = U32.to_nat(bound)
          +i = U32.to_nat(index)
          +w = U32.to_nat((source + (index - first) * 1 : U32))
          +m = clamped_index(s,b,i)
          +sum = middle_sum(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)
          +below = middle_below(first,count,source,shift,bound,index,cap,cap_ok,shift_room,placed,inside,after,before)
          +shift_index = middle_after(first,source,shift,index,placed,after)
          +index_last = {%sum : {Nat.is_le(_,Nat.sub(Nat.add(s,b),1n)) == True{} : Bool} TaskReads.value_top(s,w,b,below) : {Nat.is_le(i,Nat.sub(Nat.add(s,b),1n)) == True{} : Bool}}
          +same = Order.antisymmetric(m,i,Order.transitive(m,Nat.max(i,s),i,NatExtrema.minimum_left(Nat.max(i,s),Nat.sub(Nat.add(s,b),1n)),
              NatExtrema.maximum_upper(i,s,i,Order.reflexive(i),shift_index)),
            NatExtrema.minimum_lower(Nat.max(i,s),Nat.sub(Nat.add(s,b),1n),i,NatExtrema.maximum_left(i,s),index_last))
          %Equal.sym(U32,step,1,step_one) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat((source + (index - first) * _ : U32)) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          Held.Both{Order.positive_bound(w,b,below),
            Equal.trans(Nat,w,Nat.sub(i,s),Nat.sub(m,s),
              Equal.trans(Nat,w,Nat.sub(Nat.add(s,w),s),Nat.sub(i,s),Equal.sym(Nat,Nat.sub(Nat.add(s,w),s),w,Order.subtract_shift(s,w)),
                Equal.cong(Nat,Nat,value => Nat.sub(value,s),Nat.add(s,w),i,sum)),
              Equal.cong(Nat,Nat,value => Nat.sub(value,s),i,m,Equal.sym(Nat,m,i,same)))}
        case Before{+clamp,+step_zero,+source_zero,+positive,+ends}:
          +s = U32.to_nat(shift)
          +b = U32.to_nat(bound)
          +i = U32.to_nat(index)
          +m = clamped_index(s,b,i)
          +index_shift = Division.lt_succ_le(i,s,Bounds.nat_trans(i,Nat.add(U32.to_nat(first),U32.to_nat(count)),1n+s,before,ends))
          +m_shift = Order.transitive(m,Nat.max(i,s),s,NatExtrema.minimum_left(Nat.max(i,s),Nat.sub(Nat.add(s,b),1n)),NatExtrema.maximum_upper(i,s,s,index_shift,Order.reflexive(s)))
          %Equal.sym(U32,step,0,step_zero) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat((source + (index - first) * _ : U32)) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          %Equal.sym(U32,(source + (index - first) * 0 : U32),source,unstepped(source,index,first)) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat(_) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          %Equal.sym(U32,source,0,source_zero) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat(_) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          Held.Both{positive,Equal.sym(Nat,Nat.sub(m,s),0n,Order.subtract_below(m,s,m_shift))}
        case After{+clamp,+step_zero,+source_last,+starts}:
          +s = U32.to_nat(shift)
          +b = U32.to_nat(bound)
          +i = U32.to_nat(index)
          +f = U32.to_nat(first)
          +o = U32.to_nat(source)
          +last = Nat.sub(Nat.add(s,b),1n)
          +m = clamped_index(s,b,i)
          +source_below = Order.lt_succ(o,o,Order.reflexive(o))
          +source_less = {%source_last : {Nat.is_lt(o,_) == True{} : Bool}
            %Equal.sym(Nat,Nat.add(o,1n),1n+o,Addition.successor_width(o)) : {Nat.is_lt(o,_) == True{} : Bool} source_below : {Nat.is_lt(o,b) == True{} : Bool}}
          +same_last = Equal.trans(Nat,last,Nat.sub(Nat.add(Nat.add(s,o),1n),1n),Nat.add(s,o),
            Equal.cong(Nat,Nat,sum => Nat.sub(sum,1n),Nat.add(s,b),Nat.add(Nat.add(s,o),1n),
              Equal.trans(Nat,Nat.add(s,b),Nat.add(s,Nat.add(o,1n)),Nat.add(Nat.add(s,o),1n),
                Equal.cong(Nat,Nat,value => Nat.add(s,value),b,Nat.add(o,1n),Equal.sym(Nat,Nat.add(o,1n),b,source_last)),
                Equal.sym(Nat,Nat.add(Nat.add(s,o),1n),Nat.add(s,Nat.add(o,1n)),Addition.associative(s,o,1n)))),
            DivisionOrder.subtract_sum(Nat.add(s,o),1n))
          +last_first = Order.transitive(last,Nat.sub(1n+f,1n),f,DivisionOrder.subtract_le(Nat.add(s,b),1n+f,1n,starts),
            {%Equal.sym(Nat,Nat.sub(f,0n),f,Order.subtract_zero(f)) : {Nat.is_le(_,f) == True{} : Bool} Order.reflexive(f) : {Nat.is_le(Nat.sub(1n+f,1n),f) == True{} : Bool}})
          +last_index = Order.transitive(last,f,i,last_first,after)
          +same = Order.antisymmetric(m,last,NatExtrema.minimum_right(Nat.max(i,s),last),
            NatExtrema.minimum_lower(Nat.max(i,s),last,last,Order.transitive(last,i,Nat.max(i,s),last_index,NatExtrema.maximum_left(i,s)),Order.reflexive(last)))
          +kept = Equal.trans(Nat,Nat.sub(m,s),Nat.sub(Nat.add(s,o),s),o,
            Equal.cong(Nat,Nat,value => Nat.sub(value,s),m,Nat.add(s,o),Equal.trans(Nat,m,last,Nat.add(s,o),same,same_last)),Order.subtract_shift(s,o))
          %Equal.sym(U32,step,0,step_zero) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat((source + (index - first) * _ : U32)) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          %Equal.sym(U32,(source + (index - first) * 0 : U32),source,unstepped(source,index,first)) : Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat(_) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>
          Held.Both{Order.positive_bound(o,b,source_less),Equal.sym(Nat,Nat.sub(m,s),o,kept)}
        case Unused{+none}:
          Empty.absurd(Held.Both<KeptAt(True{},shift,bound,index),{U32.to_nat((source + (index - first) * step : U32)) == kept_index(True{},U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) : Nat}>,unused_absurd(first,count,index,none,after,before))

# ---- The source cell a target cell reads ----

# The kept indices of the coordinates of a target cell.
def kept_indices(coordinates: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+clamped: Bool) -> List<&2,Nat>:
  match coordinates source shifts:
    case Con{+index,coordinates} Con{+bound,source} Con{+shift,shifts}:
      kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(index)) <> kept_indices(coordinates,source,shifts,clamped)
    case coordinates source shifts:
      []

# Every read keeps the coordinate of its axis.
def KeptAll(coordinates: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+clamped: Bool) -> Data:
  match coordinates source shifts:
    case Con{+index,coordinates} Con{+bound,source} Con{+shift,shifts}:
      Held.Both<KeptAt(clamped,shift,bound,index),KeptAll(coordinates,source,shifts,clamped)>
    case Nil{} Nil{} Nil{}:
      Unit
    case coordinates source shifts:
      Empty

# Every target axis of a placed source keeps the coordinate of a target cell
# it keeps at its kept index.
def kept_row(target: List<&2,U32>,classes: List<&2,Tasks.Class>,coordinates: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,
  +target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: Components.ReadingsRoom(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis),cap),
  +classes_in: Targets.ClassesIn(classes,target_grid,target,BoxDigits.words_nat(coordinates),axis),+kept: KeptAll(coordinates,source,shifts,clamped)) ->
  TaskReads.PlacedRow(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis),classes,kept_indices(coordinates,source,shifts,clamped),cap):
  match target classes coordinates source shifts:
    case Con{+extent,+target} Con{Tasks.Class{+first,+count,+source_first,+source_count},+classes} Con{+index,+coordinates} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+room,+rooms_rest} = rooms
      Held.Both{+class_in,+classes_rest} = classes_in
      Held.Both{+at,+kept_rest} = kept
      Held.Both{kept_placed(extent,Grid.block(target_grid,axis,extent),axis,Grid.block(source_grid,axis,bound),shift,bound,clamped,first,count,source_first,source_count,
          index,cap,cap_ok,room,class_in,at),
        kept_row(target,classes,coordinates,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32),cap,cap_ok,rooms_rest,classes_rest,kept_rest)}
    case Nil{} Nil{} Nil{} source shifts:
      Unit{}
    case Nil{} Con{+class,+classes} coordinates source shifts:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Nil{},source,shifts,target_grid,source_grid,clamped,axis),Con{class,classes},kept_indices(coordinates,source,shifts,clamped),cap),classes_in)
    case Nil{} Nil{} Con{+index,+coordinates} source shifts:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Nil{},source,shifts,target_grid,source_grid,clamped,axis),Nil{},kept_indices(Con{index,coordinates},source,shifts,clamped),cap),classes_in)
    case Con{+extent,+target} Nil{} coordinates source shifts:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Con{extent,target},source,shifts,target_grid,source_grid,clamped,axis),Nil{},kept_indices(coordinates,source,shifts,clamped),cap),classes_in)
    case Con{+extent,+target} Con{+class,+classes} Nil{} source shifts:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Con{extent,target},source,shifts,target_grid,source_grid,clamped,axis),Con{class,classes},kept_indices(Nil{},source,shifts,clamped),cap),classes_in)
    case Con{+extent,+target} Con{+class,+classes} Con{+index,+coordinates} Nil{} shifts:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Con{extent,target},Nil{},shifts,target_grid,source_grid,clamped,axis),Con{class,classes},kept_indices(Con{index,coordinates},Nil{},shifts,clamped),cap),kept)
    case Con{+extent,+target} Con{+class,+classes} Con{+index,+coordinates} Con{+bound,+source} Nil{}:
      Empty.absurd(TaskReads.PlacedRow(Layout.placed_readings(Con{extent,target},Con{bound,source},Nil{},target_grid,source_grid,clamped,axis),Con{class,classes},kept_indices(Con{index,coordinates},Con{bound,source},Nil{},clamped),cap),kept)

# A target cell of a piece of a component of a placing whose coordinates every
# read of a source keeps reads a source cell the component takes.
def placing_held(+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,
  +readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,+number: Nat,+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +number_in: {Nat.is_lt(number,U32.to_nat(Grid.span(Tensors.grid_of(target),piece))) == True{} : Bool},+coordinates: List<&2,U32>,
  +cells_at: {BoxDigits.words_nat(coordinates) == Targets.cell_coordinates(target,Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(target),piece)),number)) : List<&2,Nat>},
  +kept: KeptAll(coordinates,source,shifts,clamped),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),
  +apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),
    Tensors.grid_of(source)),U32.to_nat(TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),kept_indices(coordinates,source,shifts,clamped),Tasks.strides_of(source)))):
  Held.Both{+rooms,Held.Both{+axes_ok,+axes_below}} = row_ok
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +capacity = Machine.capacity(limits)
  +cap_half = Bounds.as_nat_le(capacity,cap,Limits.capacity_within_half(limits))
  +target_grid = Tensors.grid_of(target)
  +source_grid = Tensors.grid_of(source)
  +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)
  +row = Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)
  +start = Nat.add(U32.to_nat(Grid.start(target_grid,piece)),number)
  +classes_in = {%Equal.sym(List<&2,Nat>,BoxDigits.words_nat(coordinates),Targets.cell_coordinates(target,start),cells_at) : Targets.ClassesIn(classes,target_grid,target,_,0)
    Targets.target_classes(limits,target,Grid.piece_target(),room,nonempty,rank_room,readings,component,separate,classes_ok,piece,member,number,number_in) :
    Targets.ClassesIn(classes,target_grid,target,BoxDigits.words_nat(coordinates),0)}
  +values = kept_indices(coordinates,source,shifts,clamped)
  +placed_row = kept_row(target,classes,coordinates,source,shifts,target_grid,source_grid,clamped,0,cap,cap_ok,rooms,classes_in,kept)
  +fixed = Grid.coordinates(source,0)
  +source_double = Limits.double_bound(limits,ShapeLaws.elements(source),source_room)
  +found = TaskReads.digits_placed(row,classes,values,source_grid,source,fixed,0,cap,cap_ok,placed_row,axes_ok,blocks,
    ViewReads.coordinates_in(source,0,cap,cap_ok,source_double,source_nonempty))
  +placed = TaskReads.placed_digits(row,values,source,fixed,0)
  +number_word = TaskReads.placed_number(row,values,source,fixed,cap,cap_ok,TaskReads.zero_offset(source,cap,cap_ok,source_double,source_nonempty),apart,axes_below,
    TaskReads.row_paired(row,classes,values,cap,placed_row),BoxDigits.coordinates_count(source,0),
    Order.transitive(List.length(&2,U32,source),U32.to_nat(capacity),U32.to_nat(cap),source_rank,cap_half))
  +word = TaskReads.placed_word(row,values,Tasks.strides_of(source))
  +box = Tasks.boxed(row,classes,source_grid,source,fixed,0)
  %Equal.sym(Bool,Tasks.separable(readings),True{},separate) :
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(source_grid,source,Tasks.box_when(_,row,classes,source_grid,source,0)),source_grid),U32.to_nat(word))
  TaskReads.held_of(limits,source,box,placed,word,source_room,source_nonempty,source_rank,
    BoxCells.boxed_coordinates_in(row,classes,source_grid,source,0,cap,cap_ok,rooms,ends,axes_ok),ViewReads.digits_box(box,source,placed,found),
    Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(source,placed)),word,number_word))

# ---- A placed source reads every axis once ----

# A placed source reads no axis across.
def placed_uncrossed(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32) ->
  {Tasks.crossing(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis)) == False{} : Bool}:
  match target source shifts:
    case Con{+extent,+target} Con{+bound,+source} Con{+shift,+shifts}: placed_uncrossed(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))
    case Nil{} source shifts: {==}
    case Con{+extent,+target} Nil{} shifts: {==}
    case Con{+extent,+target} Con{+bound,+source} Nil{}: {==}

# The axis after an axis that leaves room.
def next_axis(+axis: U32,+rest: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(axis),1n+rest),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat}:
  +a = U32.to_nat(axis)
  Equal.trans(Nat,U32.to_nat((axis + 1 : U32)),Nat.add(a,1n),1n+a,
    Arithmetic.sum_exact(cap,cap_ok,axis,1,a,1n,{==},{==},Order.transitive(Nat.add(a,1n),Nat.add(a,1n+rest),U32.to_nat(cap),Order.offset_le(a,1n,1n+rest,Order.le_add(1n,rest)),room)),
    Addition.successor_width(a))

# A placed source read from axis on reads no axis below it.
def placed_holds(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32,+other: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: {Nat.is_lt(U32.to_nat(other),U32.to_nat(axis)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,target)),U32.to_nat(cap)) == True{} : Bool}) -> {Tasks.holds(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis)),other) == False{} : Bool}:
  match target source shifts:
    case Con{+extent,+target} Con{+bound,+source} Con{+shift,+shifts}:
      +a = U32.to_nat(axis)
      +n = List.length(&2,U32,target)
      +exact = next_axis(axis,n,cap,cap_ok,room)
      +apart = Equal.trans(Bool,U32.is_eq(axis,other),Nat.is_eq(a,U32.to_nat(other)),False{},Comparison.eq(axis,other),
        Equal.trans(Bool,Nat.is_eq(a,U32.to_nat(other)),Nat.is_eq(U32.to_nat(other),a),False{},Order.equal_symmetric(a,U32.to_nat(other)),Order.less_not_equal(U32.to_nat(other),a,below)))
      +later_below = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+a,exact) : {Nat.is_lt(U32.to_nat(other),_) == True{} : Bool} Order.lt_succ(U32.to_nat(other),a,Bounds.strict_le(U32.to_nat(other),a,below)) :
        {Nat.is_lt(U32.to_nat(other),U32.to_nat((axis + 1 : U32))) == True{} : Bool}}
      +later_room = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+a,exact) : {Nat.is_le(Nat.add(_,n),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(a,n,U32.to_nat(cap),room) :
        {Nat.is_le(Nat.add(U32.to_nat((axis + 1 : U32)),n),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_eq(axis,other),False{},apart) : {(_ || Tasks.holds(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))),other)) == False{} : Bool}
      placed_holds(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32),other,cap,cap_ok,later_below,later_room)
    case Nil{} source shifts: {==}
    case Con{+extent,+target} Nil{} shifts: {==}
    case Con{+extent,+target} Con{+bound,+source} Nil{}: {==}

# A placed source reads every source axis from one target axis at most.
def placed_distinct(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,target)),U32.to_nat(cap)) == True{} : Bool}) -> {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis))) == True{} : Bool}:
  match target source shifts:
    case Con{+extent,+target} Con{+bound,+source} Con{+shift,+shifts}:
      +a = U32.to_nat(axis)
      +n = List.length(&2,U32,target)
      +exact = next_axis(axis,n,cap,cap_ok,room)
      +later_below = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+a,exact) : {Nat.is_lt(a,_) == True{} : Bool} Order.lt_succ(a,a,Order.reflexive(a)) : {Nat.is_lt(a,U32.to_nat((axis + 1 : U32))) == True{} : Bool}}
      +later_room = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+a,exact) : {Nat.is_le(Nat.add(_,n),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(a,n,U32.to_nat(cap),room) :
        {Nat.is_le(Nat.add(U32.to_nat((axis + 1 : U32)),n),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(Bool,Tasks.holds(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))),axis),False{},placed_holds(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32),axis,cap,cap_ok,later_below,
        later_room)) : {(Bool.not(_) && Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))))) == True{} : Bool}
      placed_distinct(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32),cap,cap_ok,later_room)
    case Nil{} source shifts: {==}
    case Con{+extent,+target} Nil{} shifts: {==}
    case Con{+extent,+target} Con{+bound,+source} Nil{}: {==}

# A placed source is separable.
def placed_separable(+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.separable([Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,0)]) == True{} : Bool}:
  %Equal.sym(Bool,Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,0))),True{},placed_distinct(target,source,shifts,target_grid,source_grid,clamped,0,cap,cap_ok,room)) :
    {((_ && Bool.not(Tasks.crossing(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,0)))) && Tasks.separable([])) == True{} : Bool}
  %Equal.sym(Bool,Tasks.crossing(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,0)),False{},placed_uncrossed(target,source,shifts,target_grid,source_grid,clamped,0)) :
    {((True{} && Bool.not(_)) && Tasks.separable([])) == True{} : Bool}
  {==}
