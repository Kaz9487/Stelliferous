# The source cell a target cell of a component reads lies in the box the
# component takes of the source: every target axis that reads a source axis
# reads it at a value inside the footprint of its class, and the digits the
# values make number the sum of every value times the stride of its axis.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./u32_comparison.bend as Comparison
import ./row_major_index.bend as RowMajor
import ./tensor_components.bend as Components
import ./tensor_move_readback.bend as Held
import ./tensor_view_reads.bend as ViewReads
import ./tensor_box_digits.bend as BoxDigits
import ./nat_division.bend as Division
import ./tensor_moved.bend as Moved
import ./tensor_box_cells.bend as BoxCells
import ./machine_limit_bounds.bend as LimitBounds
import ../machine_limits.bend as Machine
import ./u32_extrema.bend as WordExtrema
import ./tensor_target_cells.bend as Targets
import ./word_encoding.bend as WordEncoding
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as Extrema
import ./tensor_grid_pieces.bend as GridLaws

# ---- The digits read at values ----

# The digit of source axis axis: the value of the first target axis that
# reads it, or the fixed digit.
def placed_digit(readings: List<&2,Tasks.Reading>,values: List<&2,Nat>,+axis: U32,+fixed: Nat) -> Nat:
  match readings values:
    case Con{Tasks.Reading{extent,block,Tasks.Read{+other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},+rest} Con{+value,+more}:
      Bool.pick(Nat,U32.is_eq(other,axis),value,placed_digit(rest,more,axis,fixed))
    case Con{reading,+rest} Con{value,+more}:
      placed_digit(rest,more,axis,fixed)
    case readings values:
      fixed

def placed_digits(+readings: List<&2,Tasks.Reading>,+values: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+axis: U32) -> List<&2,Nat>:
  match extents fixed:
    case Con{extent,+rest} Con{+index,+more}:
      Con{placed_digit(readings,values,axis,U32.to_nat(index)),placed_digits(readings,values,rest,more,(axis + 1 : U32))}
    case extents fixed:
      Nil{}

# The source index value of the virtual indices from first to last that a read
# keeps from shift on, below bound: unclamped, shift plus the value lies
# between first and last; clamped, between first and last clamped to the
# kept indices.
def KeptIn(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+value: Nat) -> Data:
  match clamped:
    case False{}:
      Held.Both<{Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(shift),value)) == True{} : Bool},
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),value),U32.to_nat(last)) == True{} : Bool},{Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool}>>
    case True{}:
      Held.Both<{Nat.is_le(U32.to_nat(Tasks.clamp(first,shift,(shift + bound - 1 : U32))),Nat.add(U32.to_nat(shift),value)) == True{} : Bool},
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),value),U32.to_nat(Tasks.clamp(last,shift,(shift + bound - 1 : U32)))) == True{} : Bool},
          {Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool}>>

# A target axis that reads a source axis reads it at a value its class keeps,
# with its shift and bound inside cap.
def PlacedIn(reading: Tasks.Reading,class: Tasks.Class,+value: Nat,+cap: U32) -> Data:
  match reading class:
    case Tasks.Reading{+extent,+block,Tasks.Read{other,source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,clamped}} Tasks.Class{+first,+count,source_first,source_count}:
      +low = (first * block : U32)
      +high = (U32.min(((first + count) * block : U32),extent) - 1 : U32)
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
        KeptIn(clamped,Tasks.virtual_first(step,offset,mirrored,divisor,low,high),Tasks.virtual_last(step,offset,mirrored,width,divisor,low,high),shift,bound,value)>
    case reading class:
      Unit

def PlacedRow(readings: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,values: List<&2,Nat>,+cap: U32) -> Data:
  match readings classes values:
    case Con{+reading,readings} Con{+class,classes} Con{+value,values}:
      Held.Both<PlacedIn(reading,class,value,cap),PlacedRow(readings,classes,values,cap)>
    case Nil{} Nil{} Nil{}:
      Unit
    case readings classes values:
      Empty

def one_times(+value: Nat) -> {Nat.mul(1n,value) == value : Nat}:
  Equal.trans(Nat,Nat.mul(1n,value),Nat.mul(value,1n),value,RowMajor.multiply_commutative(1n,value),Order.times_one(value))

def one_quotient(+value: Nat) -> {Nat.div(value,1n) == value : Nat}:
  Equal.trans(Nat,Nat.div(value,1n),Nat.div(Nat.mul(value,1n),1n),value,
    Equal.cong(Nat,Nat,number => Nat.div(number,1n),value,Nat.mul(value,1n),Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value))),
    Division.exact_quotient(value,1n,{==}))

# A read of unit step, width and divisor starts at its coordinate.
def unit_above(+coordinate: Nat) -> {Nat.is_le(Nat.div(Nat.mul(1n,coordinate),1n),coordinate) == True{} : Bool}:
  +same = Equal.trans(Nat,Nat.div(Nat.mul(1n,coordinate),1n),Nat.div(coordinate,1n),coordinate,
    Equal.cong(Nat,Nat,number => Nat.div(number,1n),Nat.mul(1n,coordinate),coordinate,one_times(coordinate)),one_quotient(coordinate))
  %Equal.sym(Nat,Nat.div(Nat.mul(1n,coordinate),1n),coordinate,same) : {Nat.is_le(_,coordinate) == True{} : Bool}
  Order.reflexive(coordinate)

# A read of unit step, width and divisor ends at its coordinate.
def unit_under(+coordinate: Nat) -> {Nat.is_le(coordinate,Nat.div(Nat.sub(Nat.add(Nat.mul(1n,coordinate),1n),1n),1n)) == True{} : Bool}:
  +same = Equal.trans(Nat,Nat.div(Nat.mul(1n,coordinate),1n),Nat.div(coordinate,1n),coordinate,
    Equal.cong(Nat,Nat,number => Nat.div(number,1n),Nat.mul(1n,coordinate),coordinate,one_times(coordinate)),one_quotient(coordinate))
  %Equal.sym(Nat,Nat.sub(Nat.add(Nat.mul(1n,coordinate),1n),1n),Nat.mul(1n,coordinate),DivisionOrder.subtract_sum(Nat.mul(1n,coordinate),1n)) :
    {Nat.is_le(coordinate,Nat.div(_,1n)) == True{} : Bool}
  %Equal.sym(Nat,Nat.div(Nat.mul(1n,coordinate),1n),coordinate,same) : {Nat.is_le(coordinate,_) == True{} : Bool}
  Order.reflexive(coordinate)

# A placed row pairs every reading with one value.
def row_paired(readings: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,values: List<&2,Nat>,+cap: U32,+row: PlacedRow(readings,classes,values,cap)) ->
  {List.length(&2,Tasks.Reading,readings) == List.length(&2,Nat,values) : Nat}:
  match readings classes values:
    case Con{+reading,+readings} Con{+class,+classes} Con{+value,+values}:
      Held.Both{+here,+later} = row
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tasks.Reading,readings),List.length(&2,Nat,values),row_paired(readings,classes,values,cap,later))
    case Nil{} Nil{} Nil{}:
      {==}
    case Nil{} Nil{} Con{+value,+values}:
      Empty.absurd({List.length(&2,Tasks.Reading,Nil{}) == List.length(&2,Nat,Con{value,values}) : Nat},row)
    case Nil{} Con{+class,+classes} values:
      Empty.absurd({List.length(&2,Tasks.Reading,Nil{}) == List.length(&2,Nat,values) : Nat},row)
    case Con{+reading,+readings} Nil{} values:
      Empty.absurd({List.length(&2,Tasks.Reading,Con{reading,readings}) == List.length(&2,Nat,values) : Nat},row)
    case Con{+reading,+readings} Con{+class,+classes} Nil{}:
      Empty.absurd({List.length(&2,Tasks.Reading,Con{reading,readings}) == List.length(&2,Nat,Nil{}) : Nat},row)

# A target axis that reads its source axis unmirrored reads at its coordinate
# the indices from its offset plus its step times the coordinate on, divided:
# a value among those lies between the virtual indices of its class.
def read_virtual(+extent: U32,+block: U32,+other: U32,+source_block: U32,+step: U32,+offset: U32,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +first: U32,+count: U32,+source_first: U32,+source_count: U32,+coordinate: Nat,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,step,offset,False{},width,divisor,shift,bound,clamped},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate),
  +above: {Nat.is_le(Nat.div(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),U32.to_nat(U32.max(divisor,1))),value) == True{} : Bool},
  +under: {Nat.is_le(value,Nat.div(Nat.sub(Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),U32.to_nat(width)),1n),U32.to_nat(U32.max(divisor,1)))) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(U32.to_nat(Tasks.virtual_first(step,offset,False{},divisor,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32))),value) == True{} : Bool},
    {Nat.is_le(value,U32.to_nat(Tasks.virtual_last(step,offset,False{},width,divisor,(first * block : U32),(U32.min(((first + count) * block : U32),extent) - 1 : U32)))) == True{} : Bool}>:
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+blocks_exact,Held.Both{+blocks_room,Held.Both{+block_positive,+extent_room}}} = axis_room
  Held.Both{+positive,Held.Both{+reach,Held.Both{+divisor_room,+rest_room}}} = read_room
  Held.Both{+low_le,+top_above} = class_in
  +low = (first * block : U32)
  +ends = ((first + count) * block : U32)
  +top = U32.min(ends,extent)
  +high = (top - 1 : U32)
  +step_n = U32.to_nat(step)
  +width_n = U32.to_nat(width)
  +blocks = U32.to_nat(U32.max(divisor,1))
  +extent_n = U32.to_nat(extent)
  +top_n = U32.to_nat(top)
  +low_n = U32.to_nat(low)
  +high_n = U32.to_nat(high)
  +scaled = Nat.mul(step_n,coordinate)
  +offset_n = U32.to_nat(offset)
  +placed = Nat.add(offset_n,scaled)
  +top_extent = {%Equal.sym(Nat,top_n,Nat.min(U32.to_nat(ends),extent_n),WordExtrema.minimum(ends,extent)) : {Nat.is_le(_,extent_n) == True{} : Bool}
    NatExtrema.minimum_right(U32.to_nat(ends),extent_n) : {Nat.is_le(top_n,extent_n) == True{} : Bool}}
  +extent_cap = Order.transitive(extent_n,Nat.add(extent_n,U32.to_nat(block)),U32.to_nat(cap),Order.le_add(extent_n,U32.to_nat(block)),extent_room)
  +top_positive = Order.le_lt_trans(0n,coordinate,top_n,Order.zero_le(coordinate),top_above)
  +high_exact = Arithmetic.difference_exact(cap,cap_ok,top,1,top_n,1n,{==},{==},Order.positive_one(top_n,top_positive),
    Order.transitive(top_n,extent_n,U32.to_nat(cap),top_extent,extent_cap))
  +back = Equal.sym(Nat,high_n,Nat.sub(top_n,1n),high_exact)
  +coordinate_high = {%back : {Nat.is_le(coordinate,_) == True{} : Bool} Components.pred_le(coordinate,top_n,top_above) : {Nat.is_le(coordinate,high_n) == True{} : Bool}}
  +high_last = {%back : {Nat.is_le(_,Nat.sub(extent_n,1n)) == True{} : Bool} DivisionOrder.subtract_le(top_n,extent_n,1n,top_extent) :
    {Nat.is_le(high_n,Nat.sub(extent_n,1n)) == True{} : Bool}}
  +reach_high = Components.reach_down(False{},step,offset,width,high_n,Nat.sub(extent_n,1n),cap,high_last,reach)
  +far = Nat.mul(step_n,high_n)
  +near = Nat.mul(step_n,low_n)
  +far_sum = Nat.add(offset_n,far)
  +near_sum = Nat.add(offset_n,near)
  +far_room = Order.transitive(far_sum,Nat.add(far_sum,width_n),U32.to_nat(cap),Order.le_add(far_sum,width_n),reach_high)
  +low_high = Order.transitive(low_n,coordinate,high_n,low_le,coordinate_high)
  +near_room = Order.transitive(near_sum,far_sum,U32.to_nat(cap),Order.offset_le(offset_n,near,far,DivisionOrder.multiply_left_le(step_n,low_n,high_n,low_high)),far_room)
  +first_exact = Components.divided(cap,cap_ok,(offset + step * low : U32),divisor,near_sum,Components.forward_sum(cap,cap_ok,step,offset,low,near_room),divisor_room)
  +last_exact = Components.divided(cap,cap_ok,((offset + step * high : U32) + width - 1 : U32),divisor,Nat.sub(Nat.add(far_sum,width_n),1n),
    Components.widened(cap,cap_ok,(offset + step * high : U32),width,far_sum,Components.forward_sum(cap,cap_ok,step,offset,high,far_room),positive,reach_high),divisor_room)
  +first_le = Order.transitive(Nat.div(near_sum,blocks),Nat.div(placed,blocks),value,
    DivisionOrder.quotient_le(near_sum,placed,blocks,Components.positive_max(divisor),
      Order.offset_le(offset_n,near,scaled,DivisionOrder.multiply_left_le(step_n,low_n,coordinate,low_le))),above)
  +last_ge = Order.transitive(value,Nat.div(Nat.sub(Nat.add(placed,width_n),1n),blocks),Nat.div(Nat.sub(Nat.add(far_sum,width_n),1n),blocks),under,
    DivisionOrder.quotient_le(Nat.sub(Nat.add(placed,width_n),1n),Nat.sub(Nat.add(far_sum,width_n),1n),blocks,Components.positive_max(divisor),
      DivisionOrder.subtract_le(Nat.add(placed,width_n),Nat.add(far_sum,width_n),1n,Order.add_le(placed,far_sum,width_n,
        Order.offset_le(offset_n,scaled,far,DivisionOrder.multiply_left_le(step_n,coordinate,high_n,coordinate_high))))))
  Held.Both{
    {%Equal.sym(Nat,U32.to_nat(((offset + step * low : U32) / U32.max(divisor,1) : U32)),Nat.div(near_sum,blocks),first_exact) : {Nat.is_le(_,value) == True{} : Bool}
      first_le : {Nat.is_le(U32.to_nat(((offset + step * low : U32) / U32.max(divisor,1) : U32)),value) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((((offset + step * high : U32) + width - 1 : U32) / U32.max(divisor,1) : U32)),Nat.div(Nat.sub(Nat.add(far_sum,width_n),1n),blocks),last_exact) :
        {Nat.is_le(value,_) == True{} : Bool}
      last_ge : {Nat.is_le(value,U32.to_nat((((offset + step * high : U32) + width - 1 : U32) / U32.max(divisor,1) : U32))) == True{} : Bool}}}

# A read keeps a value its class reaches, from its shift on, below its bound.
def kept_shifted(+value: Nat,+shift: U32,+bound: U32,+cap: U32,+first: U32,+last: U32,
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +virtual: Held.Both<{Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(shift),value)) == True{} : Bool},{Nat.is_le(Nat.add(U32.to_nat(shift),value),U32.to_nat(last)) == True{} : Bool}>,
  +value_below: {Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},KeptIn(False{},first,last,shift,bound,value)>:
  Held.Both{+first_le,+last_ge} = virtual
  Held.Both{shift_room,Held.Both{first_le,Held.Both{last_ge,value_below}}}

# A target axis that reads its source axis unmirrored and unclamped reads at
# its coordinate the indices from its offset plus its step times the
# coordinate on, divided: a value among those, past the shift, lies between
# the virtual indices of its class.
def read_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+step: U32,+offset: U32,+width: U32,+divisor: U32,+shift: U32,+bound: U32,
  +first: U32,+count: U32,+source_first: U32,+source_count: U32,+coordinate: Nat,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,step,offset,False{},width,divisor,shift,bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate),
  +above: {Nat.is_le(Nat.div(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),U32.to_nat(U32.max(divisor,1))),Nat.add(U32.to_nat(shift),value)) == True{} : Bool},
  +under: {Nat.is_le(Nat.add(U32.to_nat(shift),value),
    Nat.div(Nat.sub(Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(U32.to_nat(step),coordinate)),U32.to_nat(width)),1n),U32.to_nat(U32.max(divisor,1)))) == True{} : Bool},
  +value_below: {Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool}) ->
  PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,False{},width,divisor,shift,bound,False{}}},
    Tasks.Class{first,count,source_first,source_count},value,cap):
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+positive,Held.Both{+reach,Held.Both{+divisor_room,Held.Both{+shift_room,+block_rest}}}} = read_room
  +low = (first * block : U32)
  +high = (U32.min(((first + count) * block : U32),extent) - 1 : U32)
  kept_shifted(value,shift,bound,cap,Tasks.virtual_first(step,offset,False{},divisor,low,high),Tasks.virtual_last(step,offset,False{},width,divisor,low,high),shift_room,
    read_virtual(extent,block,other,source_block,step,offset,width,divisor,shift,bound,False{},first,count,source_first,source_count,coordinate,
      Nat.add(U32.to_nat(shift),value),cap,cap_ok,room,class_in,above,under),value_below)

# An axis both sides share reads its coordinate.
def shared_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+bound: U32,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +coordinate: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,1,0,False{},1,1,0,bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate),
  +value_below: {Nat.is_lt(coordinate,U32.to_nat(bound)) == True{} : Bool}) ->
  PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,1,0,False{},1,1,0,bound,False{}}},Tasks.Class{first,count,source_first,source_count},coordinate,cap):
  read_placed(extent,block,other,source_block,1,0,1,1,0,bound,first,count,source_first,source_count,coordinate,coordinate,cap,cap_ok,room,class_in,
    unit_above(coordinate),unit_under(coordinate),value_below)

# ---- A read keeps its value ----

# A word at most cap and a Nat at most the word subtract exactly.
def kept_difference(+word: U32,+shift: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(shift),U32.to_nat(word)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(word),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((word - shift : U32)) == Nat.sub(U32.to_nat(word),U32.to_nat(shift)) : Nat}:
  Arithmetic.difference_exact(cap,cap_ok,word,shift,U32.to_nat(word),U32.to_nat(shift),{==},{==},ordered,bounded)

# What lies below shift plus bound lies below bound after shift.
def below_after(+shift: Nat,+value: Nat,+bound: Nat,+after: {Nat.is_le(shift,value) == True{} : Bool},+below: {Nat.is_lt(value,Nat.add(shift,bound)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(value,shift),bound) == True{} : Bool}:
  match shift value:
    case 0n +value:
      %Equal.sym(Nat,Nat.sub(value,0n),value,Order.subtract_zero(value)) : {Nat.is_lt(_,bound) == True{} : Bool}
      below
    case 1n+ +rest 0n:
      Empty.absurd({Nat.is_lt(Nat.sub(0n,1n+rest),bound) == True{} : Bool},WordEncoding.false_true(after))
    case 1n+ +rest 1n+ +more:
      below_after(rest,more,bound,after,below)

# Shift plus a value at most a word leaves the value after shift.
def kept_value(+shift: Nat,+value: Nat,+word: Nat,+ordered: {Nat.is_le(word,Nat.add(shift,value)) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(word,shift),value) == True{} : Bool}:
  %Order.subtract_shift(shift,value) : {Nat.is_le(Nat.sub(word,shift),_) == True{} : Bool}
  DivisionOrder.subtract_le(word,Nat.add(shift,value),shift,ordered)

def value_kept(+shift: Nat,+value: Nat,+word: Nat,+ordered: {Nat.is_le(Nat.add(shift,value),word) == True{} : Bool}) ->
  {Nat.is_le(value,Nat.sub(word,shift)) == True{} : Bool}:
  %Order.subtract_shift(shift,value) : {Nat.is_le(_,Nat.sub(word,shift)) == True{} : Bool}
  DivisionOrder.subtract_le(Nat.add(shift,value),word,shift,ordered)

# The last index a read keeps, shift plus bound less one.
def top_exact(+shift: U32,+bound: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool}) ->
  {U32.to_nat((shift + bound - 1 : U32)) == Nat.sub(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),1n) : Nat}:
  +sum = Nat.add(U32.to_nat(shift),U32.to_nat(bound))
  Arithmetic.difference_exact(cap,cap_ok,(shift + bound : U32),1,sum,1n,Arithmetic.add(shift,bound,cap,cap_ok,shift_room),{==},
    Order.transitive(1n,U32.to_nat(bound),sum,Order.positive_one(U32.to_nat(bound),positive),
      {%Order.commutative(U32.to_nat(bound),U32.to_nat(shift)) : {Nat.is_le(U32.to_nat(bound),_) == True{} : Bool} Order.le_add(U32.to_nat(bound),U32.to_nat(shift)) :
        {Nat.is_le(U32.to_nat(bound),sum) == True{} : Bool}}),
    shift_room)

# Shift plus a value below bound lies at most at the last kept index.
def value_top(+shift: Nat,+value: Nat,+bound: Nat,+below: {Nat.is_lt(value,bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(shift,value),Nat.sub(Nat.add(shift,bound),1n)) == True{} : Bool}:
  Components.pred_le(Nat.add(shift,value),Nat.add(shift,bound),Order.offset_lt(shift,value,bound,below))

def kept_below(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+value: Nat,+kept: KeptIn(clamped,first,last,shift,bound,value)) ->
  {Nat.is_lt(value,U32.to_nat(bound)) == True{} : Bool}:
  match clamped:
    case False{}:
      Held.Both{+low_le,Held.Both{+high_ge,+below}} = kept
      below
    case True{}:
      Held.Both{+low_le,Held.Both{+high_ge,+below}} = kept
      below

# A value a read keeps lies in the span of the footprint of the read.
def kept_in(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32,+coordinate: U32,+value: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +kept: KeptIn(clamped,first,last,shift,bound,value),
  +room: {Nat.is_le(Nat.add(U32.to_nat(bound),U32.to_nat(source_block)),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(source_block)) == True{} : Bool}) ->
  ViewReads.InSpan(Tasks.box_span(Some{Tasks.placed(clamped,first,last,shift,bound,source_block)},bound,source_block,coordinate),value):
  match clamped:
    case False{}:
      Held.Both{+low_le,Held.Both{+high_ge,+value_below}} = kept
      +before = U32.to_nat(shift)
      +total = U32.to_nat(bound)
      +sum = Nat.add(before,total)
      +reached = Nat.add(before,value)
      +positive = Order.le_lt_trans(0n,value,total,Order.zero_le(value),value_below)
      +top = (shift + bound - 1 : U32)
      +last_n = Nat.sub(sum,1n)
      +top_value = top_exact(shift,bound,cap,cap_ok,shift_room,positive)
      +reached_top = value_top(before,value,total,value_below)
      +reached_room = Order.transitive(reached,last_n,U32.to_nat(cap),reached_top,Order.transitive(last_n,sum,U32.to_nat(cap),Order.sub_le(sum,1n),shift_room))
      +after = Order.le_add(before,value)
      +lower = U32.max(first,shift)
      +lower_value = Extrema.maximum(first,shift)
      +lower_reached = {%Equal.sym(Nat,U32.to_nat(lower),Nat.max(U32.to_nat(first),before),lower_value) : {Nat.is_le(_,reached) == True{} : Bool}
        NatExtrema.maximum_upper(U32.to_nat(first),before,reached,low_le,after) : {Nat.is_le(U32.to_nat(lower),reached) == True{} : Bool}}
      +lower_after = {%Equal.sym(Nat,U32.to_nat(lower),Nat.max(U32.to_nat(first),before),lower_value) : {Nat.is_le(before,_) == True{} : Bool}
        NatExtrema.maximum_right(U32.to_nat(first),before) : {Nat.is_le(before,U32.to_nat(lower)) == True{} : Bool}}
      +upper = U32.min(last,top)
      +upper_value = Equal.trans(Nat,U32.to_nat(upper),Nat.min(U32.to_nat(last),U32.to_nat(top)),Nat.min(U32.to_nat(last),last_n),Extrema.minimum(last,top),
        Equal.cong(Nat,Nat,found => Nat.min(U32.to_nat(last),found),U32.to_nat(top),last_n,top_value))
      +upper_reached = {%Equal.sym(Nat,U32.to_nat(upper),Nat.min(U32.to_nat(last),last_n),upper_value) : {Nat.is_le(reached,_) == True{} : Bool}
        NatExtrema.minimum_lower(U32.to_nat(last),last_n,reached,high_ge,reached_top) : {Nat.is_le(reached,U32.to_nat(upper)) == True{} : Bool}}
      +upper_top = {%Equal.sym(Nat,U32.to_nat(upper),Nat.min(U32.to_nat(last),last_n),upper_value) : {Nat.is_le(_,last_n) == True{} : Bool}
        NatExtrema.minimum_right(U32.to_nat(last),last_n) : {Nat.is_le(U32.to_nat(upper),last_n) == True{} : Bool}}
      +upper_room = Order.transitive(U32.to_nat(upper),last_n,U32.to_nat(cap),upper_top,Order.transitive(last_n,sum,U32.to_nat(cap),Order.sub_le(sum,1n),shift_room))
      +sum_positive = Order.le_lt_trans(0n,value,sum,Order.zero_le(value),
        Order.le_lt_trans(value,reached,sum,{%Order.commutative(value,before) : {Nat.is_le(value,_) == True{} : Bool} Order.le_add(value,before) :
          {Nat.is_le(value,reached) == True{} : Bool}},Order.offset_lt(before,value,total,value_below)))
      +upper_below = Order.le_lt_trans(U32.to_nat(upper),last_n,sum,upper_top,Order.last_below(sum,sum_positive))
      +start = Nat.sub(U32.to_nat(lower),before)
      +end = Nat.sub(U32.to_nat(upper),before)
      +empty_false = {%Equal.sym(Bool,U32.is_zero(bound),False{},Arithmetic.word_positive_nonzero(bound,positive)) :
          {(_ || U32.is_lt(last,shift) || U32.is_lt(top,first)) == False{} : Bool}
        %Equal.sym(Bool,U32.is_lt(last,shift),False{},Equal.trans(Bool,U32.is_lt(last,shift),Nat.is_lt(U32.to_nat(last),before),False{},Comparison.lt(last,shift),
            Order.not_below(before,U32.to_nat(last),Order.transitive(before,reached,U32.to_nat(last),after,high_ge)))) :
          {(False{} || _ || U32.is_lt(top,first)) == False{} : Bool}
        %Equal.sym(Bool,U32.is_lt(top,first),False{},Equal.trans(Bool,U32.is_lt(top,first),Nat.is_lt(U32.to_nat(top),U32.to_nat(first)),False{},Comparison.lt(top,first),
            Order.not_below(U32.to_nat(first),U32.to_nat(top),{%Equal.sym(Nat,U32.to_nat(top),last_n,top_value) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
              Order.transitive(U32.to_nat(first),reached,last_n,low_le,reached_top) : {Nat.is_le(U32.to_nat(first),U32.to_nat(top)) == True{} : Bool}}))) :
          {(False{} || False{} || _) == False{} : Bool}
        {==} : {(U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first)) == False{} : Bool}}
      +low = ((lower - shift : U32) / U32.max(source_block,1) : U32)
      +high = ((upper - shift : U32) / U32.max(source_block,1) : U32)
      %Equal.sym(Tasks.Footprint,Tasks.placed(False{},first,last,shift,bound,source_block),Tasks.Footprint{low,high,False{}},
          Equal.cong(Bool,Tasks.Footprint,empty => Tasks.Footprint{low,high,empty},(U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first)),False{},empty_false)) :
        ViewReads.InSpan(Tasks.box_span(Some{_},bound,source_block,coordinate),value)
      ViewReads.footprint_in((lower - shift : U32),(upper - shift : U32),bound,source_block,coordinate,value,cap,cap_ok,start,end,
        kept_difference(lower,shift,cap,cap_ok,lower_after,Order.transitive(U32.to_nat(lower),reached,U32.to_nat(cap),lower_reached,reached_room)),
        kept_difference(upper,shift,cap,cap_ok,Order.transitive(before,reached,U32.to_nat(upper),after,upper_reached),upper_room),
        kept_value(before,value,U32.to_nat(lower),lower_reached),value_kept(before,value,U32.to_nat(upper),upper_reached),
        below_after(before,U32.to_nat(upper),total,Order.transitive(before,reached,U32.to_nat(upper),after,upper_reached),upper_below),room,block_room,block_positive)
    case True{}:
      Held.Both{+low_le,Held.Both{+high_ge,+value_below}} = kept
      +before = U32.to_nat(shift)
      +total = U32.to_nat(bound)
      +sum = Nat.add(before,total)
      +reached = Nat.add(before,value)
      +positive = Order.le_lt_trans(0n,value,total,Order.zero_le(value),value_below)
      +top = (shift + bound - 1 : U32)
      +last_n = Nat.sub(sum,1n)
      +top_value = top_exact(shift,bound,cap,cap_ok,shift_room,positive)
      +reached_top = value_top(before,value,total,value_below)
      +reached_room = Order.transitive(reached,last_n,U32.to_nat(cap),reached_top,Order.transitive(last_n,sum,U32.to_nat(cap),Order.sub_le(sum,1n),shift_room))
      +after = Order.le_add(before,value)
      +lower = Tasks.clamp(first,shift,top)
      +upper = Tasks.clamp(last,shift,top)
      +lower_after = {%Equal.sym(Nat,U32.to_nat(lower),Nat.min(U32.to_nat(U32.max(first,shift)),U32.to_nat(top)),Extrema.minimum(U32.max(first,shift),top)) :
          {Nat.is_le(before,_) == True{} : Bool}
        NatExtrema.minimum_lower(U32.to_nat(U32.max(first,shift)),U32.to_nat(top),before,
          {%Equal.sym(Nat,U32.to_nat(U32.max(first,shift)),Nat.max(U32.to_nat(first),before),Extrema.maximum(first,shift)) : {Nat.is_le(before,_) == True{} : Bool}
            NatExtrema.maximum_right(U32.to_nat(first),before) : {Nat.is_le(before,U32.to_nat(U32.max(first,shift))) == True{} : Bool}},
          {%Equal.sym(Nat,U32.to_nat(top),last_n,top_value) : {Nat.is_le(before,_) == True{} : Bool}
            Order.transitive(before,reached,last_n,after,reached_top) : {Nat.is_le(before,U32.to_nat(top)) == True{} : Bool}}) :
        {Nat.is_le(before,U32.to_nat(lower)) == True{} : Bool}}
      +upper_top = {%Equal.sym(Nat,U32.to_nat(upper),Nat.min(U32.to_nat(U32.max(last,shift)),U32.to_nat(top)),Extrema.minimum(U32.max(last,shift),top)) :
          {Nat.is_le(_,last_n) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(top),last_n,top_value) : {Nat.is_le(Nat.min(U32.to_nat(U32.max(last,shift)),_),last_n) == True{} : Bool}
        NatExtrema.minimum_right(U32.to_nat(U32.max(last,shift)),last_n) : {Nat.is_le(U32.to_nat(upper),last_n) == True{} : Bool}}
      +upper_room = Order.transitive(U32.to_nat(upper),last_n,U32.to_nat(cap),upper_top,Order.transitive(last_n,sum,U32.to_nat(cap),Order.sub_le(sum,1n),shift_room))
      +sum_positive = Order.le_lt_trans(0n,value,sum,Order.zero_le(value),
        Order.le_lt_trans(value,reached,sum,{%Order.commutative(value,before) : {Nat.is_le(value,_) == True{} : Bool} Order.le_add(value,before) :
          {Nat.is_le(value,reached) == True{} : Bool}},Order.offset_lt(before,value,total,value_below)))
      +upper_below = Order.le_lt_trans(U32.to_nat(upper),last_n,sum,upper_top,Order.last_below(sum,sum_positive))
      +start = Nat.sub(U32.to_nat(lower),before)
      +end = Nat.sub(U32.to_nat(upper),before)
      +low = ((lower - shift : U32) / U32.max(source_block,1) : U32)
      +high = ((upper - shift : U32) / U32.max(source_block,1) : U32)
      %Equal.sym(Tasks.Footprint,Tasks.placed(True{},first,last,shift,bound,source_block),Tasks.Footprint{low,high,False{}},
          Equal.cong(Bool,Tasks.Footprint,empty => Tasks.Footprint{low,high,empty},U32.is_zero(bound),False{},Arithmetic.word_positive_nonzero(bound,positive))) :
        ViewReads.InSpan(Tasks.box_span(Some{_},bound,source_block,coordinate),value)
      ViewReads.footprint_in((lower - shift : U32),(upper - shift : U32),bound,source_block,coordinate,value,cap,cap_ok,start,end,
        kept_difference(lower,shift,cap,cap_ok,lower_after,Order.transitive(U32.to_nat(lower),reached,U32.to_nat(cap),low_le,reached_room)),
        kept_difference(upper,shift,cap,cap_ok,Order.transitive(before,reached,U32.to_nat(upper),after,high_ge),upper_room),
        kept_value(before,value,U32.to_nat(lower),low_le),value_kept(before,value,U32.to_nat(upper),high_ge),
        below_after(before,U32.to_nat(upper),total,Order.transitive(before,reached,U32.to_nat(upper),after,high_ge),upper_below),room,block_room,block_positive)

# ---- The digits lie in the box ----

# The source axis a target axis reads, when it reads it, holds its value.
def placed_found(hit: Bool,+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +target_extent: U32,+target_block: U32,+first: U32,+count: U32,+source_first: U32,+source_count: U32,+later_found: Maybe<&2,Tasks.Footprint>,+later_digit: Nat,
  +value: Nat,+grid: Grid.Grid,+axis: U32,+extent: U32,+fixed: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +here: PlacedIn(Tasks.Reading{target_extent,target_block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},
    Tasks.Class{first,count,source_first,source_count},value,cap),
  +facts: Components.AxisFacts(hit,source_block,bound,grid,axis,extent),
  +later: ViewReads.AxisIn(Tasks.box_span(later_found,extent,Grid.block(grid,axis,extent),fixed),later_digit,extent),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,extent),1))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,axis,extent))) == True{} : Bool}) ->
  ViewReads.AxisIn(Tasks.box_span(Tasks.found_footprint(hit,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},target_extent,target_block,
      first,count,later_found),extent,Grid.block(grid,axis,extent),fixed),Bool.pick(Nat,hit,value,later_digit),extent):
  match hit:
    case True{}:
      Held.Both{+shift_room,+kept} = here
      Held.Both{+same_block,+same_bound} = facts
      +block = Grid.block(grid,axis,extent)
      +low = (first * target_block : U32)
      +high = (U32.min(((first + count) * target_block : U32),target_extent) - 1 : U32)
      +start = Tasks.virtual_first(step,offset,mirrored,divisor,low,high)
      +finish = Tasks.virtual_last(step,offset,mirrored,width,divisor,low,high)
      +kept_extent = {%same_bound : KeptIn(clamped,start,finish,shift,_,value) kept : KeptIn(clamped,start,finish,shift,extent,value)}
      +spanned = kept_in(clamped,start,finish,shift,extent,block,fixed,value,cap,cap_ok,
        {%same_bound : {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(_)),U32.to_nat(cap)) == True{} : Bool} shift_room :
          {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}},kept_extent,room,block_room,positive)
      Held.Both{
        {%Equal.sym(U32,source_block,block,same_block) : ViewReads.InSpan(Tasks.box_span(Some{Tasks.placed(clamped,start,finish,shift,bound,_)},extent,block,fixed),value)
          %Equal.sym(U32,bound,extent,same_bound) : ViewReads.InSpan(Tasks.box_span(Some{Tasks.placed(clamped,start,finish,shift,_,block)},extent,block,fixed),value)
          spanned :
          ViewReads.InSpan(Tasks.box_span(Some{Tasks.range_footprint(Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},low,high)},extent,
            block,fixed),value)},
        kept_below(clamped,start,finish,shift,extent,value,kept_extent)}
    case False{}:
      later

# The source axis axis holds the digit its reading target axis gives it.
def axis_placed(readings: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,values: List<&2,Nat>,+grid: Grid.Grid,+axis: U32,+extent: U32,+fixed: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+row: PlacedRow(readings,classes,values,cap),+reads: Components.ReadsAxis(readings,grid,axis,extent),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
  +block_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,extent),1))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,axis,extent))) == True{} : Bool},
  +fixed_in: {Nat.is_lt(U32.to_nat(fixed),U32.to_nat(extent)) == True{} : Bool}) ->
  ViewReads.AxisIn(Tasks.box_span(Tasks.read_footprint(readings,classes,axis),extent,Grid.block(grid,axis,extent),fixed),placed_digit(readings,values,axis,U32.to_nat(fixed)),
    extent):
  match readings classes values:
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest}
        Con{Tasks.Class{+first,+count,+source_first,+source_count},+others} Con{+value,+more}:
      Held.Both{+here,+later} = row
      Held.Both{+facts,+reads_rest} = reads
      placed_found(U32.is_eq(other,axis),other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,target_extent,target_block,first,count,source_first,
        source_count,Tasks.read_footprint(rest,others,axis),placed_digit(rest,more,axis,U32.to_nat(fixed)),value,grid,axis,extent,fixed,cap,cap_ok,here,facts,
        axis_placed(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads_rest,room,block_room,positive,fixed_in),room,block_room,positive)
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Unread{}},+rest} Con{+class,+others} Con{+value,+more}:
      Held.Both{+here,+later} = row
      axis_placed(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads,room,block_room,positive,fixed_in)
    case Con{Tasks.Reading{+target_extent,+target_block,Tasks.Crossing{}},+rest} Con{+class,+others} Con{+value,+more}:
      Held.Both{+here,+later} = row
      axis_placed(rest,others,more,grid,axis,extent,fixed,cap,cap_ok,later,reads,room,block_room,positive,fixed_in)
    case Nil{} Nil{} Nil{}:
      Held.Both{ViewReads.none_in(extent,Grid.block(grid,axis,extent),fixed,cap,cap_ok,room,block_room,positive,fixed_in),fixed_in}
    case Nil{} Nil{} Con{+value,+more}:
      Empty.absurd(ViewReads.AxisIn(Tasks.box_span(Tasks.read_footprint(Nil{},Nil{},axis),extent,Grid.block(grid,axis,extent),fixed),
        placed_digit(Nil{},Con{value,more},axis,U32.to_nat(fixed)),extent),row)
    case Nil{} Con{+class,+others} +values:
      Empty.absurd(ViewReads.AxisIn(Tasks.box_span(Tasks.read_footprint(Nil{},Con{class,others},axis),extent,Grid.block(grid,axis,extent),fixed),
        placed_digit(Nil{},values,axis,U32.to_nat(fixed)),extent),row)
    case Con{+reading,+rest} Nil{} +values:
      Empty.absurd(ViewReads.AxisIn(Tasks.box_span(Tasks.read_footprint(Con{reading,rest},Nil{},axis),extent,Grid.block(grid,axis,extent),fixed),
        placed_digit(Con{reading,rest},values,axis,U32.to_nat(fixed)),extent),row)
    case Con{+reading,+rest} Con{+class,+others} Nil{}:
      Empty.absurd(ViewReads.AxisIn(Tasks.box_span(Tasks.read_footprint(Con{reading,rest},Con{class,others},axis),extent,Grid.block(grid,axis,extent),fixed),
        placed_digit(Con{reading,rest},Nil{},axis,U32.to_nat(fixed)),extent),row)

# Every source axis holds the digit its reading target axis gives it.
def digits_placed(+readings: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+values: List<&2,Nat>,+grid: Grid.Grid,extents: List<&2,U32>,fixed: List<&2,U32>,
  +at: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+row: PlacedRow(readings,classes,values,cap),
  +axes: Components.AxesOk(readings,grid,extents,at,cap),+blocks: ViewReads.BlocksIn(grid,extents,at,cap),+fixed_in: ViewReads.FixedIn(fixed,extents)) ->
  ViewReads.DigitsIn(Tasks.boxed(readings,classes,grid,extents,fixed,at),extents,placed_digits(readings,values,extents,fixed,at)):
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      Held.Both{+here,+later} = axes
      Held.Both{+reads,+room} = here
      Held.Both{+positive,+blocks_rest} = blocks
      Held.Both{+block_room,+blocks_more} = blocks_rest
      Held.Both{+index_in,+fixed_rest} = fixed_in
      Held.Both{axis_placed(readings,classes,values,grid,at,extent,index,cap,cap_ok,row,reads,room,block_room,positive,index_in),
        digits_placed(readings,classes,values,grid,rest,more,(at + 1 : U32),cap,cap_ok,row,later,blocks_more,fixed_rest)}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+index,+more}:
      Empty.absurd(ViewReads.DigitsIn(Tasks.boxed(readings,classes,grid,Nil{},Con{index,more},at),Nil{},placed_digits(readings,values,Nil{},Con{index,more},at)),fixed_in)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(ViewReads.DigitsIn(Tasks.boxed(readings,classes,grid,Con{extent,rest},Nil{},at),Con{extent,rest},placed_digits(readings,values,Con{extent,rest},Nil{},at)),
        fixed_in)

# ---- The number of the digits ----

# Every fixed digit is zero.
def Zeros(fixed: List<&2,U32>) -> Data:
  match fixed:
    case Nil{}:
      Unit
    case Con{+index,rest}:
      Held.Both<{U32.to_nat(index) == 0n : Nat},Zeros(rest)>

# Every value times the stride of the source axis its target axis reads.
def placed_word(readings: List<&2,Tasks.Reading>,values: List<&2,Nat>,+strides: List<&2,U32>) -> U32:
  match readings values:
    case Con{Tasks.Reading{extent,block,Tasks.Read{+other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},+rest} Con{+value,+more}:
      ((U32.from_nat(value) * Tasks.nth(strides,other) : U32) + placed_word(rest,more,strides) : U32)
    case Con{reading,+rest} Con{value,+more}:
      placed_word(rest,more,strides)
    case readings values:
      0

# A source axis no target axis reads keeps its fixed digit.
def placed_unheld(readings: List<&2,Tasks.Reading>,values: List<&2,Nat>,+axis: U32,+fixed: Nat,
  +unheld: {Tasks.holds(Tasks.read_axes(readings),axis) == False{} : Bool}) -> {placed_digit(readings,values,axis,fixed) == fixed : Nat}:
  match readings values:
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Con{+value,+more}:
      +others = Tasks.holds(Tasks.read_axes(rest),axis)
      +missed = ViewReads.left_false(U32.is_eq(other,axis),others,unheld)
      %Equal.sym(Bool,U32.is_eq(other,axis),False{},missed) : {Bool.pick(Nat,_,value,placed_digit(rest,more,axis,fixed)) == fixed : Nat}
      placed_unheld(rest,more,axis,fixed,ViewReads.right_false(U32.is_eq(other,axis),others,unheld))
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} Con{+value,+more}:
      placed_unheld(rest,more,axis,fixed,unheld)
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} Con{+value,+more}:
      placed_unheld(rest,more,axis,fixed,unheld)
    case Nil{} +values:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Nil{}:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} Nil{}:
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} Nil{}:
      {==}

# A target axis that reads source axis other puts its value there.
def placed_put(+extent: U32,+block: U32,+other: U32,+source_block: U32,+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+shift: U32,+bound: U32,
  +clamped: Bool,+rest: List<&2,Tasks.Reading>,+value: Nat,+more: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {placed_digits(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},Con{value,more},extents,fixed,at) ==
    ViewReads.put(placed_digits(rest,more,extents,fixed,at),at,other,value) : List<&2,Nat>}:
  match extents fixed:
    case Con{+size,+others} Con{+index,+indices}:
      +head = Bool.pick(Nat,U32.is_eq(other,at),value,placed_digit(rest,more,at,U32.to_nat(index)))
      +next = (at + 1 : U32)
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{head,tail},
        placed_digits(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},Con{value,more},others,indices,
          next),ViewReads.put(placed_digits(rest,more,others,indices,next),next,other,value),
        placed_put(extent,block,other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,rest,value,more,others,indices,next))
    case Nil{} +fixed:
      {==}
    case Con{+size,+others} Nil{}:
      {==}

# A target axis that reads no source axis keeps every digit.
def placed_unread(+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+value: Nat,+more: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {placed_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{value,more},extents,fixed,at) == placed_digits(rest,more,extents,fixed,at) : List<&2,Nat>}:
  match extents fixed:
    case Con{+size,+others} Con{+index,+indices}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{placed_digit(rest,more,at,U32.to_nat(index)),tail},
        placed_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{value,more},others,indices,(at + 1 : U32)),placed_digits(rest,more,others,indices,(at + 1 : U32)),
        placed_unread(extent,block,rest,value,more,others,indices,(at + 1 : U32)))
    case Nil{} +fixed:
      {==}
    case Con{+size,+others} Nil{}:
      {==}

def placed_crossing(+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+value: Nat,+more: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32) ->
  {placed_digits(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},Con{value,more},extents,fixed,at) == placed_digits(rest,more,extents,fixed,at) : List<&2,Nat>}:
  match extents fixed:
    case Con{+size,+others} Con{+index,+indices}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Con{placed_digit(rest,more,at,U32.to_nat(index)),tail},
        placed_digits(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},Con{value,more},others,indices,(at + 1 : U32)),placed_digits(rest,more,others,indices,(at + 1 : U32)),
        placed_crossing(extent,block,rest,value,more,others,indices,(at + 1 : U32)))
    case Nil{} +fixed:
      {==}
    case Con{+size,+others} Nil{}:
      {==}

# The digit of a source axis no target axis reads is zero.
def zero_found(hit: Bool,+readings: List<&2,Tasks.Reading>,+values: List<&2,Nat>,+index: U32,+at: U32,+axis: U32,+same: {U32.is_eq(axis,at) == hit : Bool},
  +unheld: {Tasks.holds(Tasks.read_axes(readings),axis) == False{} : Bool},+zero: {U32.to_nat(index) == 0n : Nat},+later_digit: Nat,+later: {later_digit == 0n : Nat}) ->
  {Bool.pick(Nat,hit,placed_digit(readings,values,at,U32.to_nat(index)),later_digit) == 0n : Nat}:
  match hit:
    case True{}:
      %Arithmetic.equal_of_is_eq(axis,at,same) : {placed_digit(readings,values,_,U32.to_nat(index)) == 0n : Nat}
      %Equal.sym(Nat,U32.to_nat(index),0n,zero) : {placed_digit(readings,values,axis,_) == 0n : Nat}
      placed_unheld(readings,values,axis,0n,unheld)
    case False{}:
      later

def placed_zero_at(+readings: List<&2,Tasks.Reading>,+values: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32,+axis: U32,
  +unheld: {Tasks.holds(Tasks.read_axes(readings),axis) == False{} : Bool},+zeros: Zeros(fixed)) ->
  {ViewReads.digit_at(placed_digits(readings,values,extents,fixed,at),at,axis) == 0n : Nat}:
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      Held.Both{+zero,+later} = zeros
      +next = (at + 1 : U32)
      zero_found(U32.is_eq(axis,at),readings,values,index,at,axis,{==},unheld,zero,ViewReads.digit_at(placed_digits(readings,values,rest,more,next),next,axis),
        placed_zero_at(readings,values,rest,more,next,axis,unheld,later))
    case Nil{} +fixed:
      {==}
    case Con{+extent,+rest} Nil{}:
      {==}

# The digits number zero when no target axis reads.
def fixed_value(extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32,+zeros: Zeros(fixed)) ->
  {ShapeLaws.digits_value(extents,placed_digits(Nil{},Nil{},extents,fixed,at)) == 0n : Nat}:
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      Held.Both{+zero,+later} = zeros
      %Equal.sym(Nat,U32.to_nat(index),0n,zero) :
        {Nat.add(Nat.mul(_,ShapeLaws.elements(rest)),ShapeLaws.digits_value(rest,placed_digits(Nil{},Nil{},rest,more,(at + 1 : U32)))) == 0n : Nat}
      fixed_value(rest,more,(at + 1 : U32),later)
    case Nil{} +fixed:
      {==}
    case Con{+extent,+rest} Nil{}:
      {==}

def placed_length(+readings: List<&2,Tasks.Reading>,+values: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+at: U32,
  +same: {List.length(&2,U32,fixed) == List.length(&2,U32,extents) : Nat}) ->
  {List.length(&2,Nat,placed_digits(readings,values,extents,fixed,at)) == List.length(&2,U32,extents) : Nat}:
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,placed_digits(readings,values,rest,more,(at + 1 : U32))),List.length(&2,U32,rest),
        placed_length(readings,values,rest,more,(at + 1 : U32),
          Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,U32,more),1n+List.length(&2,U32,rest),same)))
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+index,+more}:
      Empty.absurd({List.length(&2,Nat,placed_digits(readings,values,Nil{},Con{index,more},at)) == 0n : Nat},
        BoxDigits.zero_successor(List.length(&2,U32,more),Equal.sym(Nat,1n+List.length(&2,U32,more),0n,same)))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({List.length(&2,Nat,placed_digits(readings,values,Con{extent,rest},Nil{},at)) == 1n+List.length(&2,U32,rest) : Nat},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same))

# The weight of a source axis a distance past the first axis is its stride.
def weight_stride(extents: List<&2,U32>,digits: List<&2,Nat>,+at: U32,+position: Nat,+distance: Nat,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},+reach: {Nat.add(position,distance) == U32.to_nat(axis) : Nat},
  +axis_room: {Nat.is_le(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool},+same: {List.length(&2,Nat,digits) == List.length(&2,U32,extents) : Nat}) ->
  {U32.from_nat(ViewReads.axis_weight(extents,digits,at,axis)) == Tasks.nth(Tasks.strides_of(extents),U32.from_nat(distance)) : U32}:
  match extents digits distance:
    case Con{+extent,+rest} Con{+digit,+more} 0n:
      +hit = Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),True{},Comparison.eq(axis,at),
        {%reach : {Nat.is_eq(_,U32.to_nat(at)) == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(Nat.add(position,0n),_) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {Nat.is_eq(_,position) == True{} : Bool}
          Order.equal_reflexive(position) : {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == True{} : Bool}})
      %Equal.sym(Bool,U32.is_eq(axis,at),True{},hit) :
        {U32.from_nat(Bool.pick(Nat,_,ShapeLaws.elements(rest),ViewReads.axis_weight(rest,more,(at + 1 : U32),axis))) ==
          Tasks.nth(Tasks.strides_of(Con{extent,rest}),U32.from_nat(0n)) : U32}
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {U32.from_nat(ShapeLaws.elements(rest)) == Tasks.nth(_,0) : U32}
      BoxDigits.product_from(rest)
    case Con{+extent,+rest} Con{+digit,+more} 1n+ +further:
      +limit = U32.to_nat(cap)
      +far = Nat.add(position,1n+further)
      +far_room = {%Equal.sym(Nat,far,U32.to_nat(axis),reach) : {Nat.is_le(_,limit) == True{} : Bool} axis_room : {Nat.is_le(far,limit) == True{} : Bool}}
      +word = U32.from_nat(1n+further)
      +word_exact = Bounds.roundtrip(1n+further,cap,cap_ok,Order.transitive(1n+further,far,limit,Order.right_le(position,1n+further),far_room))
      +nonzero = Arithmetic.word_positive_nonzero(word,{%Equal.sym(Nat,U32.to_nat(word),1n+further,word_exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
        {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}})
      +step = Equal.trans(U32,word,U32.from_nat(Nat.add(further,1n)),(U32.from_nat(further) + 1 : U32),
        Equal.cong(Nat,U32,value => U32.from_nat(value),1n+further,Nat.add(further,1n),Equal.sym(Nat,Nat.add(further,1n),1n+further,Addition.successor_width(further))),
        Equal.sym(U32,(U32.from_nat(further) + U32.from_nat(1n) : U32),U32.from_nat(Nat.add(further,1n)),WordArithmetic.native_add(further,1n)))
      +less = Equal.trans(U32,(word - 1 : U32),((U32.from_nat(further) + 1 : U32) - 1 : U32),U32.from_nat(further),
        Equal.cong(U32,U32,value => (value - 1 : U32),word,(U32.from_nat(further) + 1 : U32),step),Subtraction.subtract_add(U32.from_nat(further),1))
      +miss = Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),False{},Comparison.eq(axis,at),
        {%reach : {Nat.is_eq(_,U32.to_nat(at)) == False{} : Bool}
          %Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(far,_) == False{} : Bool}
          Equal.trans(Bool,Nat.is_eq(far,position),Nat.is_eq(position,far),False{},Order.equal_symmetric(far,position),
            Order.less_not_equal(position,far,Order.below_sum(position,further))) :
          {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == False{} : Bool}})
      +next_exact = Components.next_word(at,position,cap,cap_ok,exact,
        Order.transitive(1n+position,far,limit,
          {%Equal.sym(Nat,far,1n+Nat.add(position,further),Addition.successor_right(position,further)) : {Nat.is_le(1n+position,_) == True{} : Bool}
            Order.le_add(1n+position,further) : {Nat.is_le(1n+position,far) == True{} : Bool}},far_room))
      +later = weight_stride(rest,more,(at + 1 : U32),1n+position,further,axis,cap,cap_ok,next_exact,
        Equal.trans(Nat,Nat.add(1n+position,further),far,U32.to_nat(axis),Equal.sym(Nat,far,1n+Nat.add(position,further),Addition.successor_right(position,further)),reach),
        axis_room,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,U32,rest),same))
      %Equal.sym(Bool,U32.is_eq(axis,at),False{},miss) :
        {U32.from_nat(Bool.pick(Nat,_,ShapeLaws.elements(rest),ViewReads.axis_weight(rest,more,(at + 1 : U32),axis))) ==
          Tasks.nth(Tasks.strides_of(Con{extent,rest}),word) : U32}
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {U32.from_nat(ViewReads.axis_weight(rest,more,(at + 1 : U32),axis)) == Tasks.nth(_,word) : U32}
      %Equal.sym(Bool,U32.is_zero(word),False{},nonzero) :
        {U32.from_nat(ViewReads.axis_weight(rest,more,(at + 1 : U32),axis)) == Bool.pick(U32,_,Grid.product(rest),Tasks.nth(Tasks.strides_of(rest),(word - 1 : U32))) : U32}
      %Equal.sym(U32,(word - 1 : U32),U32.from_nat(further),less) :
        {U32.from_nat(ViewReads.axis_weight(rest,more,(at + 1 : U32),axis)) == Tasks.nth(Tasks.strides_of(rest),_) : U32}
      later
    case Nil{} +digits +distance:
      {==}
    case Con{+extent,+rest} Nil{} +distance:
      Empty.absurd({U32.from_nat(ViewReads.axis_weight(Con{extent,rest},Nil{},at,axis)) == Tasks.nth(Tasks.strides_of(Con{extent,rest}),U32.from_nat(distance)) : U32},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same))

# The digits the values give number the sum of every value times the stride
# of its axis, when no two target axes read one source axis.
def placed_number(readings: List<&2,Tasks.Reading>,values: List<&2,Nat>,+extents: List<&2,U32>,+fixed: List<&2,U32>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+zeros: Zeros(fixed),+apart: {Tasks.distinct(Tasks.read_axes(readings)) == True{} : Bool},
  +below: Components.AxesBelow(Tasks.read_axes(readings),List.length(&2,U32,extents)),
  +paired: {List.length(&2,Tasks.Reading,readings) == List.length(&2,Nat,values) : Nat},
  +same: {List.length(&2,U32,fixed) == List.length(&2,U32,extents) : Nat},+room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.from_nat(ShapeLaws.digits_value(extents,placed_digits(readings,values,extents,fixed,0))) == placed_word(readings,values,Tasks.strides_of(extents)) : U32}:
  match readings values:
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Con{+value,+more}:
      Held.Both{+other_below,+below_rest} = below
      +strides = Tasks.strides_of(extents)
      +limit = U32.to_nat(cap)
      +count = List.length(&2,U32,extents)
      +digits = placed_digits(rest,more,extents,fixed,0)
      +others = Tasks.holds(Tasks.read_axes(rest),other)
      +unheld = ViewReads.not_true(others,ViewReads.and_left(Bool.not(others),Tasks.distinct(Tasks.read_axes(rest)),apart))
      +rest_apart = ViewReads.and_right(Bool.not(others),Tasks.distinct(Tasks.read_axes(rest)),apart)
      +length = placed_length(rest,more,extents,fixed,0,same)
      +weight = ViewReads.axis_weight(extents,digits,0,other)
      +moved = ViewReads.put_value(extents,digits,0,0n,other,value,cap,cap_ok,{==},
        {%Equal.sym(Nat,List.length(&2,Nat,digits),count,length) : {Nat.is_le(_,limit) == True{} : Bool} room : {Nat.is_le(List.length(&2,Nat,digits),limit) == True{} : Bool}})
      +changed = ShapeLaws.digits_value(extents,ViewReads.put(digits,0,other,value))
      +kept = ShapeLaws.digits_value(extents,digits)
      +flat = {%placed_zero_at(rest,more,extents,fixed,0,other,unheld,zeros) : {Nat.add(changed,Nat.mul(_,weight)) == Nat.add(kept,Nat.mul(value,weight)) : Nat}
        moved : {Nat.add(changed,Nat.mul(0n,weight)) == Nat.add(kept,Nat.mul(value,weight)) : Nat}}
      +equation = Equal.trans(Nat,changed,Nat.add(changed,0n),Nat.add(kept,Nat.mul(value,weight)),Equal.sym(Nat,Nat.add(changed,0n),changed,Addition.zero_right(changed)),flat)
      +stride = Equal.trans(U32,U32.from_nat(weight),Tasks.nth(strides,U32.from_nat(U32.to_nat(other))),Tasks.nth(strides,other),
        weight_stride(extents,digits,0,0n,U32.to_nat(other),other,cap,cap_ok,{==},{==},
          Order.transitive(U32.to_nat(other),count,limit,Bounds.strict_le(U32.to_nat(other),count,other_below),room),length),
        Equal.cong(U32,U32,word => Tasks.nth(strides,word),U32.from_nat(U32.to_nat(other)),other,Arithmetic.native_roundtrip(other)))
      +later = placed_number(rest,more,extents,fixed,cap,cap_ok,zeros,rest_apart,below_rest,
        Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Tasks.Reading,rest),1n+List.length(&2,Nat,more),paired),same,room)
      +rest_word = placed_word(rest,more,strides)
      +term = (U32.from_nat(value) * Tasks.nth(strides,other) : U32)
      +sum_eq = {%Equal.sym(U32,U32.from_nat(kept),rest_word,later) : {(_ + U32.from_nat(Nat.mul(value,weight)) : U32) == (rest_word + term : U32) : U32}
        %Multiplication.native_encoded_multiply(value,weight) : {(rest_word + _ : U32) == (rest_word + term : U32) : U32}
        %Equal.sym(U32,U32.from_nat(weight),Tasks.nth(strides,other),stride) : {(rest_word + (U32.from_nat(value) * _ : U32) : U32) == (rest_word + term : U32) : U32}
        {==} : {(U32.from_nat(kept) + U32.from_nat(Nat.mul(value,weight)) : U32) == (rest_word + term : U32) : U32}}
      %Equal.sym(List<&2,Nat>,placed_digits(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},
          Con{value,more},extents,fixed,0),ViewReads.put(digits,0,other,value),
          placed_put(extent,block,other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped,rest,value,more,extents,fixed,0)) :
        {U32.from_nat(ShapeLaws.digits_value(extents,_)) == (term + rest_word : U32) : U32}
      Equal.trans(U32,U32.from_nat(changed),(U32.from_nat(kept) + U32.from_nat(Nat.mul(value,weight)) : U32),(term + rest_word : U32),
        Equal.trans(U32,U32.from_nat(changed),U32.from_nat(Nat.add(kept,Nat.mul(value,weight))),(U32.from_nat(kept) + U32.from_nat(Nat.mul(value,weight)) : U32),
          Equal.cong(Nat,U32,number => U32.from_nat(number),changed,Nat.add(kept,Nat.mul(value,weight)),equation),
          Equal.sym(U32,(U32.from_nat(kept) + U32.from_nat(Nat.mul(value,weight)) : U32),U32.from_nat(Nat.add(kept,Nat.mul(value,weight))),
            WordArithmetic.native_add(kept,Nat.mul(value,weight)))),
        Equal.trans(U32,(U32.from_nat(kept) + U32.from_nat(Nat.mul(value,weight)) : U32),(rest_word + term : U32),(term + rest_word : U32),sum_eq,
          Arithmetic.add_commutative(rest_word,term)))
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} Con{+value,+more}:
      %Equal.sym(List<&2,Nat>,placed_digits(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},Con{value,more},extents,fixed,0),placed_digits(rest,more,extents,fixed,0),
          placed_unread(extent,block,rest,value,more,extents,fixed,0)) :
        {U32.from_nat(ShapeLaws.digits_value(extents,_)) == placed_word(rest,more,Tasks.strides_of(extents)) : U32}
      placed_number(rest,more,extents,fixed,cap,cap_ok,zeros,apart,below,
        Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Tasks.Reading,rest),1n+List.length(&2,Nat,more),paired),same,room)
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} Con{+value,+more}:
      %Equal.sym(List<&2,Nat>,placed_digits(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},Con{value,more},extents,fixed,0),placed_digits(rest,more,extents,fixed,0),
          placed_crossing(extent,block,rest,value,more,extents,fixed,0)) :
        {U32.from_nat(ShapeLaws.digits_value(extents,_)) == placed_word(rest,more,Tasks.strides_of(extents)) : U32}
      placed_number(rest,more,extents,fixed,cap,cap_ok,zeros,apart,below,
        Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Tasks.Reading,rest),1n+List.length(&2,Nat,more),paired),same,room)
    case Nil{} Nil{}:
      Equal.cong(Nat,U32,number => U32.from_nat(number),ShapeLaws.digits_value(extents,placed_digits(Nil{},Nil{},extents,fixed,0)),0n,fixed_value(extents,fixed,0,zeros))
    case Nil{} Con{+value,+more}:
      Empty.absurd({U32.from_nat(ShapeLaws.digits_value(extents,placed_digits(Nil{},Con{value,more},extents,fixed,0))) ==
          placed_word(Nil{},Con{value,more},Tasks.strides_of(extents)) : U32},BoxDigits.zero_successor(List.length(&2,Nat,more),paired))
    case Con{+reading,+rest} Nil{}:
      Empty.absurd({U32.from_nat(ShapeLaws.digits_value(extents,placed_digits(Con{reading,rest},Nil{},extents,fixed,0))) ==
          placed_word(Con{reading,rest},Nil{},Tasks.strides_of(extents)) : U32},
        BoxDigits.zero_successor(List.length(&2,Tasks.Reading,rest),Equal.sym(Nat,1n+List.length(&2,Tasks.Reading,rest),0n,paired)))

# ---- The source cell of digits a component holds ----

# Every number is zero.
def AllZero(numbers: List<&2,Nat>) -> Data:
  match numbers:
    case Nil{}: Unit
    case Con{+number,rest}: Held.Both<{number == 0n : Nat},AllZero(rest)>

def zero_cells(extents: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> AllZero(Targets.cell_coordinates(extents,0n)):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +split = Division.small(0n,inner,inner_positive)
      +quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(0n,inner),(0n,0n),split)
      +remainder = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(0n,inner),(0n,0n),split)
      Held.Both{quotient,{%Equal.sym(Nat,Nat.mod(0n,inner),0n,remainder) : AllZero(Targets.cell_coordinates(rest,_))
        zero_cells(rest,inner_positive) : AllZero(Targets.cell_coordinates(rest,Nat.mod(0n,inner)))}}

def zeros_from(words: List<&2,U32>,+zero: AllZero(BoxDigits.words_nat(words))) -> Zeros(words):
  match words:
    case Nil{}: Unit{}
    case Con{+word,+rest}:
      Held.Both{+here,+later} = zero
      Held.Both{here,zeros_from(rest,later)}

# The coordinates of the first source cell are zero.
def zero_offset(+extents: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) -> Zeros(Grid.coordinates(extents,0)):
  zeros_from(Grid.coordinates(extents,0),
    {%Equal.sym(List<&2,Nat>,BoxDigits.words_nat(Grid.coordinates(extents,0)),Targets.cell_coordinates(extents,U32.to_nat(0)),
        BoxDigits.coordinates_exact(extents,0,cap,cap_ok,room,positive)) : AllZero(_)
      zero_cells(extents,positive) : AllZero(BoxDigits.words_nat(Grid.coordinates(extents,0)))})

# A source cell inside a box of the source cells, at a word, lies in a
# piece the box takes.
def held_of(+limits: Machine.Limits,+extents: List<&2,U32>,+box: List<&2,Tasks.Span>,+digits: List<&2,Nat>,+word: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+within: Components.SpansIn(box,extents),
  +found: ViewReads.BoxDigits(box,extents,digits),+exact: {word == U32.from_nat(ShapeLaws.digits_value(extents,digits)) : U32}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(extents,Grid.piece_target()),extents,box),Grid.grid(extents,Grid.piece_target())),U32.to_nat(word)):
  Held.Both{+inside,+below} = found
  +number = ShapeLaws.digits_value(extents,digits)
  +capacity = Machine.capacity(limits)
  +value = Equal.trans(Nat,U32.to_nat(word),U32.to_nat(U32.from_nat(number)),number,Equal.cong(U32,Nat,cell => U32.to_nat(cell),word,U32.from_nat(number),exact),
    Bounds.roundtrip(number,capacity,LimitBounds.capacity_fits(limits),
      Order.transitive(number,ShapeLaws.elements(extents),U32.to_nat(capacity),Bounds.strict_le(number,ShapeLaws.elements(extents),below),room)))
  %Equal.sym(Nat,U32.to_nat(word),number,value) :
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(extents,Grid.piece_target()),extents,box),Grid.grid(extents,Grid.piece_target())),_)
  BoxCells.cell_held(limits,extents,Grid.piece_target(),room,nonempty,rank_room,box,number,within,inside,below)
