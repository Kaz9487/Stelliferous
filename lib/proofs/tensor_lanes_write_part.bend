# The part reads of a write along the lane axis: every target cell the part
# reaches reads part cells its component takes.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as U32Division
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_components.bend as Components
import ./tensor_task_reads.bend as TaskReads
import ./tensor_target_cells.bend as Targets
import ./tensor_move_readback.bend as Held
import ./tensor_lanes_write_gather.bend as Gather
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_tile_coordinates.bend as Coordinates
import ../kernel_gemm.bend as KernelGemm
import ./tensor_tiled_representation.bend as Representation
import ./tensor_moved.bend as Moved
import ./tensor_view_reads.bend as ViewReads
import ./tensor_box_cells.bend as BoxCells
import ./tensor_shape_refinement.bend as ShapeLaws
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./boolean_logic.bend as Logic
import ./tensor_tiled_public.bend as Public

# ---- The part groups a target group reads ----

# A value at least count divisors holds count divisors.
def multiple_quotient(+count: Nat,+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +ordered: {Nat.is_le(Nat.mul(count,divisor),value) == True{} : Bool}) -> {Nat.is_le(count,Nat.div(value,divisor)) == True{} : Bool}:
  Division.lt_succ_le(count,Nat.div(value,divisor),
    RowMajor.cancel_multiply_lt(count,1n+Nat.div(value,divisor),divisor,
      Order.le_lt_trans(Nat.mul(count,divisor),value,Nat.mul(1n+Nat.div(value,divisor),divisor),ordered,DivisionOrder.next_above(value,divisor,positive))))

# The first lane of the part group a target group starts at: the lanes the
# part skips in its first group, then the target group, are the whole groups
# of start and the lanes the target group reaches past start.
def placed_sum(+start: Nat,+group: Nat,+inside: {Nat.is_le(start,Nat.add(Nat.mul(8n,group),8n)) == True{} : Bool}) ->
  {Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group)) == Nat.add(Nat.mul(Nat.div(start,8n),8n),Nat.sub(Nat.add(Nat.mul(8n,group),8n),start)) : Nat}:
  +lanes = Nat.mul(8n,group)
  +rest = Nat.mod(start,8n)
  +whole = Nat.mul(Nat.div(start,8n),8n)
  +end = Nat.add(lanes,8n)
  +reach = Nat.sub(end,start)
  +rest_le = Bounds.strict_le(rest,8n,DivisionOrder.remainder_small(start,8n,{==}))
  +left = Equal.trans(Nat,Nat.add(rest,Nat.add(Nat.sub(8n,rest),lanes)),Nat.add(Nat.add(rest,Nat.sub(8n,rest)),lanes),end,
    Equal.sym(Nat,Nat.add(Nat.add(rest,Nat.sub(8n,rest)),lanes),Nat.add(rest,Nat.add(Nat.sub(8n,rest),lanes)),Addition.associative(rest,Nat.sub(8n,rest),lanes)),
    {%Equal.sym(Nat,Nat.add(rest,Nat.sub(8n,rest)),8n,Order.restore_shift(rest,8n,rest_le)) : {Nat.add(_,lanes) == end : Nat}
      Order.commutative(8n,lanes) : {Nat.add(Nat.add(rest,Nat.sub(8n,rest)),lanes) == end : Nat}})
  +right = Equal.trans(Nat,Nat.add(rest,Nat.add(whole,reach)),Nat.add(Nat.add(rest,whole),reach),end,
    Equal.sym(Nat,Nat.add(Nat.add(rest,whole),reach),Nat.add(rest,Nat.add(whole,reach)),Addition.associative(rest,whole,reach)),
    {%Equal.trans(Nat,start,Nat.add(whole,rest),Nat.add(rest,whole),DivisionOrder.quotient_exact(start,8n,{==}),Order.commutative(whole,rest)) :
        {Nat.add(_,reach) == end : Nat}
      Order.restore_shift(start,end,inside) : {Nat.add(Nat.add(rest,whole),reach) == end : Nat}})
  Order.cancel_sum(rest,Nat.add(Nat.sub(8n,rest),lanes),Nat.add(whole,reach),
    Equal.trans(Nat,Nat.add(rest,Nat.add(Nat.sub(8n,rest),lanes)),end,Nat.add(rest,Nat.add(whole,reach)),left,
      Equal.sym(Nat,Nat.add(rest,Nat.add(whole,reach)),end,right)))

# The first virtual index of a target group: the whole groups of start and
# the part groups the reach covers.
def first_value(+start: Nat,+group: Nat,+inside: {Nat.is_le(start,Nat.add(Nat.mul(8n,group),8n)) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group)),8n) == Nat.add(Nat.div(start,8n),Nat.div(Nat.sub(Nat.add(Nat.mul(8n,group),8n),start),8n)) : Nat}:
  +reach = Nat.sub(Nat.add(Nat.mul(8n,group),8n),start)
  Equal.trans(Nat,Nat.div(Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group)),8n),Nat.div(Nat.add(Nat.mul(Nat.div(start,8n),8n),reach),8n),
    Nat.add(Nat.div(start,8n),Nat.div(reach,8n)),
    Equal.cong(Nat,Nat,value => Nat.div(value,8n),Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group)),Nat.add(Nat.mul(Nat.div(start,8n),8n),reach),
      placed_sum(start,group,inside)),
    DivisionOrder.quotient_shift(reach,8n,Nat.div(start,8n),{==}))

# The last virtual index of a target group.
def last_value(+start: Nat,+group: Nat,+inside: {Nat.is_le(start,Nat.add(Nat.mul(8n,group),8n)) == True{} : Bool}) ->
  {Nat.div(Nat.sub(Nat.add(Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group)),8n),1n),8n) ==
    Nat.add(Nat.div(start,8n),Nat.div(Nat.add(Nat.sub(Nat.add(Nat.mul(8n,group),8n),start),7n),8n)) : Nat}:
  +placed = Nat.add(Nat.sub(8n,Nat.mod(start,8n)),Nat.mul(8n,group))
  +whole = Nat.mul(Nat.div(start,8n),8n)
  +reach = Nat.sub(Nat.add(Nat.mul(8n,group),8n),start)
  +moved = Equal.trans(Nat,Nat.sub(Nat.add(placed,8n),1n),Nat.add(placed,7n),Nat.add(whole,Nat.add(reach,7n)),Order.subtract_one_complement(placed,7n),
    Equal.trans(Nat,Nat.add(placed,7n),Nat.add(Nat.add(whole,reach),7n),Nat.add(whole,Nat.add(reach,7n)),
      Equal.cong(Nat,Nat,value => Nat.add(value,7n),placed,Nat.add(whole,reach),placed_sum(start,group,inside)),
      Addition.associative(whole,reach,7n)))
  Equal.trans(Nat,Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n),Nat.div(Nat.add(whole,Nat.add(reach,7n)),8n),Nat.add(Nat.div(start,8n),Nat.div(Nat.add(reach,7n),8n)),
    Equal.cong(Nat,Nat,value => Nat.div(value,8n),Nat.sub(Nat.add(placed,8n),1n),Nat.add(whole,Nat.add(reach,7n)),moved),
    DivisionOrder.quotient_shift(Nat.add(reach,7n),8n,Nat.div(start,8n),{==}))

# A target group from the first group of start on starts inside its lanes.
def start_inside(+start: U32,+group: U32,+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(start),Nat.add(Nat.mul(8n,U32.to_nat(group)),8n)) == True{} : Bool}:
  +value = U32.to_nat(start)
  +whole = Nat.div(value,8n)
  +rest = Nat.mod(value,8n)
  %Equal.sym(Nat,value,Nat.add(Nat.mul(whole,8n),rest),DivisionOrder.quotient_exact(value,8n,{==})) :
    {Nat.is_le(_,Nat.add(Nat.mul(8n,U32.to_nat(group)),8n)) == True{} : Bool}
  %RowMajor.multiply_commutative(8n,whole) : {Nat.is_le(Nat.add(_,rest),Nat.add(Nat.mul(8n,U32.to_nat(group)),8n)) == True{} : Bool}
  Order.transitive(Nat.add(Nat.mul(8n,whole),rest),Nat.add(Nat.mul(8n,whole),8n),Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),
    Order.offset_le(Nat.mul(8n,whole),rest,8n,Bounds.strict_le(rest,8n,DivisionOrder.remainder_small(value,8n,{==}))),
    Order.add_le(Nat.mul(8n,whole),Nat.mul(8n,U32.to_nat(group)),8n,DivisionOrder.multiply_left_le(8n,whole,U32.to_nat(group),from)))

# The lanes a target group reaches past start.
def reach_exact(+start: U32,+group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((8 * group + 8 - start : U32)) == Nat.sub(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(start)) : Nat}:
  +lanes = Nat.mul(8n,U32.to_nat(group))
  +end = Nat.add(lanes,8n)
  +product = Arithmetic.product_exact(cap,cap_ok,8,group,8n,U32.to_nat(group),{==},{==},Order.transitive(lanes,end,U32.to_nat(cap),Order.le_add(lanes,8n),end_room))
  +sum = Arithmetic.sum_exact(cap,cap_ok,(8 * group : U32),8,lanes,8n,product,{==},end_room)
  Arithmetic.difference_exact(cap,cap_ok,(8 * group + 8 : U32),start,end,U32.to_nat(start),sum,{==},start_inside(start,group,from),end_room)

# The lanes a part skips in its first group.
def offset_exact(+start: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((8 - start % 8 : U32)) == Nat.sub(8n,Nat.mod(U32.to_nat(start),8n)) : Nat}:
  +rest = Nat.mod(U32.to_nat(start),8n)
  Arithmetic.difference_exact(cap,cap_ok,8,(start % 8 : U32),8n,rest,{==},
    U32Division.remainder_exact(cap,cap_ok,start,8,U32.to_nat(start),8n,{==},{==},divisor_room,{==}),
    Bounds.strict_le(rest,8n,DivisionOrder.remainder_small(U32.to_nat(start),8n,{==})),
    Order.transitive(8n,Nat.double(8n),U32.to_nat(cap),Order.le_double(8n),divisor_room))

# The part groups start skips, and one more.
def shift_exact(+start: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(8n),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.div(U32.to_nat(start),8n),1n),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((start / 8 + 1 : U32)) == Nat.add(Nat.div(U32.to_nat(start),8n),1n) : Nat}:
  Arithmetic.sum_exact(cap,cap_ok,(start / 8 : U32),1,Nat.div(U32.to_nat(start),8n),1n,
    U32Division.quotient_exact(cap,cap_ok,start,8,U32.to_nat(start),8n,{==},{==},divisor_room,{==}),{==},room)

# A target group from the first group of start on, below the end of the
# word, reaches part groups below the word.
def shift_cap(+start: U32,+group: U32,+cap: U32,+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.div(U32.to_nat(start),8n),1n),U32.to_nat(cap)) == True{} : Bool}:
  +g = U32.to_nat(group)
  Order.transitive(Nat.add(Nat.div(U32.to_nat(start),8n),1n),Nat.add(g,1n),U32.to_nat(cap),Order.add_le(Nat.div(U32.to_nat(start),8n),g,1n,from),
    Order.transitive(Nat.add(g,1n),Nat.add(Nat.mul(8n,g),1n),U32.to_nat(cap),Order.add_le(g,Nat.mul(8n,g),1n,Components.factor_covers(g,8n,{==})),
      Order.transitive(Nat.add(Nat.mul(8n,g),1n),Nat.add(Nat.mul(8n,g),8n),U32.to_nat(cap),Order.offset_le(Nat.mul(8n,g),1n,8n,{==}),end_room)))

# A target group whose reach covers a part group reads the part group before
# the one its reach ends in.
def above_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+start: U32,+bound: U32,+first: U32,+count: U32,+source_first: U32,
  +source_count: U32,+group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,U32.to_nat(group)),
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool},
  +reaches: {U32.is_le(8,(8 * group + 8 - start : U32)) == True{} : Bool},
  +within: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}}},
    Tasks.Class{first,count,source_first,source_count},U32.to_nat((((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32)),cap):
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+positive,Held.Both{+reach_at,Held.Both{+divisor_room,Held.Both{+shift_room,+block_room}}}} = read_room
  +st = U32.to_nat(start)
  +g = U32.to_nat(group)
  +whole = Nat.div(st,8n)
  +reach_word = (8 * group + 8 - start : U32)
  +above_word = (reach_word / 8 : U32)
  +value_word = (above_word - 1 : U32)
  +offset_word = (8 - start % 8 : U32)
  +shift_word = (start / 8 + 1 : U32)
  +end = Nat.add(Nat.mul(8n,g),8n)
  +reach = Nat.sub(end,st)
  +above = Nat.div(reach,8n)
  +last = Nat.div(Nat.add(reach,7n),8n)
  +inside = start_inside(start,group,from)
  +reach_value = reach_exact(start,group,cap,cap_ok,from,end_room)
  +reach_room = Order.transitive(reach,end,U32.to_nat(cap),Order.sub_le(end,st),end_room)
  +above_value = U32Division.quotient_exact(cap,cap_ok,reach_word,8,reach,8n,reach_value,{==},divisor_room,{==})
  +eight = {%reach_value : {Nat.is_le(8n,_) == True{} : Bool}
    Equal.trans(Bool,Nat.is_le(8n,U32.to_nat(reach_word)),U32.is_le(8,reach_word),True{},
      Equal.sym(Bool,U32.is_le(8,reach_word),Nat.is_le(8n,U32.to_nat(reach_word)),Comparison.le(8,reach_word)),reaches) :
    {Nat.is_le(8n,reach) == True{} : Bool}}
  +above_positive = multiple_quotient(1n,reach,8n,{==},eight)
  +above_room = Order.transitive(above,reach,U32.to_nat(cap),DivisionOrder.quotient_at_most(reach,8n,{==}),reach_room)
  +value_value = Arithmetic.difference_exact(cap,cap_ok,above_word,1,above,1n,above_value,{==},above_positive,above_room)
  +offset_value = offset_exact(start,cap,cap_ok,divisor_room)
  +shift_value = shift_exact(start,cap,cap_ok,divisor_room,shift_cap(start,group,cap,from,end_room))
  +first_at = first_value(st,g,inside)
  +last_at = last_value(st,g,inside)
  +sum_back = Equal.trans(Nat,Nat.add(Nat.add(whole,1n),Nat.sub(above,1n)),Nat.add(whole,Nat.add(1n,Nat.sub(above,1n))),Nat.add(whole,above),
    Addition.associative(whole,1n,Nat.sub(above,1n)),
    Equal.cong(Nat,Nat,value => Nat.add(whole,value),Nat.add(1n,Nat.sub(above,1n)),above,Order.restore_shift(1n,above,above_positive)))
  +placed = Nat.add(U32.to_nat(offset_word),Nat.mul(8n,g))
  TaskReads.read_placed(extent,block,other,source_block,8,offset_word,8,8,shift_word,bound,first,count,source_first,source_count,g,U32.to_nat(value_word),cap,cap_ok,
    room,class_in,
    {%Equal.sym(Nat,U32.to_nat(value_word),Nat.sub(above,1n),value_value) : {Nat.is_le(Nat.div(placed,8n),Nat.add(U32.to_nat(shift_word),_)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(shift_word),Nat.add(whole,1n),shift_value) : {Nat.is_le(Nat.div(placed,8n),Nat.add(_,Nat.sub(above,1n))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(offset_word),Nat.sub(8n,Nat.mod(st,8n)),offset_value) :
        {Nat.is_le(Nat.div(Nat.add(_,Nat.mul(8n,g)),8n),Nat.add(Nat.add(whole,1n),Nat.sub(above,1n))) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.sub(8n,Nat.mod(st,8n)),Nat.mul(8n,g)),8n),Nat.add(whole,above),first_at) :
        {Nat.is_le(_,Nat.add(Nat.add(whole,1n),Nat.sub(above,1n))) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(Nat.add(whole,1n),Nat.sub(above,1n)),Nat.add(whole,above),sum_back) : {Nat.is_le(Nat.add(whole,above),_) == True{} : Bool}
      Order.reflexive(Nat.add(whole,above)) : {Nat.is_le(Nat.div(placed,8n),Nat.add(U32.to_nat(shift_word),U32.to_nat(value_word))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(value_word),Nat.sub(above,1n),value_value) :
        {Nat.is_le(Nat.add(U32.to_nat(shift_word),_),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(shift_word),Nat.add(whole,1n),shift_value) :
        {Nat.is_le(Nat.add(_,Nat.sub(above,1n)),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(offset_word),Nat.sub(8n,Nat.mod(st,8n)),offset_value) :
        {Nat.is_le(Nat.add(Nat.add(whole,1n),Nat.sub(above,1n)),Nat.div(Nat.sub(Nat.add(Nat.add(_,Nat.mul(8n,g)),8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.sub(Nat.add(Nat.add(Nat.sub(8n,Nat.mod(st,8n)),Nat.mul(8n,g)),8n),1n),8n),Nat.add(whole,last),last_at) :
        {Nat.is_le(Nat.add(Nat.add(whole,1n),Nat.sub(above,1n)),_) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(Nat.add(whole,1n),Nat.sub(above,1n)),Nat.add(whole,above),sum_back) : {Nat.is_le(_,Nat.add(whole,last)) == True{} : Bool}
      Order.offset_le(whole,above,last,DivisionOrder.quotient_le(reach,Nat.add(reach,7n),8n,{==},Order.le_add(reach,7n))) :
        {Nat.is_le(Nat.add(U32.to_nat(shift_word),U32.to_nat(value_word)),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(value_word),Nat.sub(above,1n),value_value) : {Nat.is_lt(_,U32.to_nat(bound)) == True{} : Bool}
      Order.lt_le(Nat.sub(above,1n),above,U32.to_nat(bound),Components.pred_lt(above,Order.succ_le_lt(0n,above,above_positive)),
        {%above_value : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool} within : {Nat.is_le(above,U32.to_nat(bound)) == True{} : Bool}}) :
      {Nat.is_lt(U32.to_nat(value_word),U32.to_nat(bound)) == True{} : Bool}})

# A target group whose reach ends inside a part group below the bound reads
# that part group.
def beside_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+start: U32,+bound: U32,+first: U32,+count: U32,+source_first: U32,
  +source_count: U32,+group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,U32.to_nat(group)),
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool},
  +shifted: {U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool},
  +below: {U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound) == True{} : Bool}) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}}},
    Tasks.Class{first,count,source_first,source_count},U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),cap):
  Held.Both{+axis_room,+read_room} = room
  Held.Both{+positive,Held.Both{+reach_at,Held.Both{+divisor_room,Held.Both{+shift_room,+block_room}}}} = read_room
  +st = U32.to_nat(start)
  +g = U32.to_nat(group)
  +whole = Nat.div(st,8n)
  +reach_word = (8 * group + 8 - start : U32)
  +above_word = (reach_word / 8 : U32)
  +offset_word = (8 - start % 8 : U32)
  +shift_word = (start / 8 + 1 : U32)
  +end = Nat.add(Nat.mul(8n,g),8n)
  +reach = Nat.sub(end,st)
  +above = Nat.div(reach,8n)
  +rest = Nat.mod(reach,8n)
  +last = Nat.div(Nat.add(reach,7n),8n)
  +lanes = Nat.mul(above,8n)
  +inside = start_inside(start,group,from)
  +reach_value = reach_exact(start,group,cap,cap_ok,from,end_room)
  +above_value = U32Division.quotient_exact(cap,cap_ok,reach_word,8,reach,8n,reach_value,{==},divisor_room,{==})
  +rest_value = U32Division.remainder_exact(cap,cap_ok,reach_word,8,reach,8n,reach_value,{==},divisor_room,{==})
  +rest_positive = {%rest_value : {Nat.is_lt(0n,_) == True{} : Bool}
    Equal.trans(Bool,Nat.is_lt(0n,U32.to_nat((reach_word % 8 : U32))),U32.is_lt(0,(reach_word % 8 : U32)),True{},
      Equal.sym(Bool,U32.is_lt(0,(reach_word % 8 : U32)),Nat.is_lt(0n,U32.to_nat((reach_word % 8 : U32))),Comparison.lt(0,(reach_word % 8 : U32))),shifted) :
    {Nat.is_lt(0n,rest) == True{} : Bool}}
  +offset_value = offset_exact(start,cap,cap_ok,divisor_room)
  +shift_value = shift_exact(start,cap,cap_ok,divisor_room,shift_cap(start,group,cap,from,end_room))
  +first_at = first_value(st,g,inside)
  +last_at = last_value(st,g,inside)
  +grouped = Equal.sym(Nat,Nat.add(Nat.add(whole,1n),above),Nat.add(whole,Nat.add(1n,above)),Addition.associative(whole,1n,above))
  +covered = {%Equal.sym(Nat,reach,Nat.add(lanes,rest),DivisionOrder.quotient_exact(reach,8n,{==})) : {Nat.is_le(Nat.add(8n,lanes),Nat.add(_,7n)) == True{} : Bool}
    %Order.commutative(lanes,8n) : {Nat.is_le(_,Nat.add(Nat.add(lanes,rest),7n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(lanes,rest),7n),Nat.add(lanes,Nat.add(rest,7n)),Addition.associative(lanes,rest,7n)) :
      {Nat.is_le(Nat.add(lanes,8n),_) == True{} : Bool}
    Order.offset_le(lanes,8n,Nat.add(rest,7n),Order.add_le(1n,rest,7n,Order.positive_one(rest,rest_positive))) :
    {Nat.is_le(Nat.mul(Nat.add(1n,above),8n),Nat.add(reach,7n)) == True{} : Bool}}
  +placed = Nat.add(U32.to_nat(offset_word),Nat.mul(8n,g))
  TaskReads.read_placed(extent,block,other,source_block,8,offset_word,8,8,shift_word,bound,first,count,source_first,source_count,g,U32.to_nat(above_word),cap,cap_ok,
    room,class_in,
    {%Equal.sym(Nat,U32.to_nat(above_word),above,above_value) : {Nat.is_le(Nat.div(placed,8n),Nat.add(U32.to_nat(shift_word),_)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(shift_word),Nat.add(whole,1n),shift_value) : {Nat.is_le(Nat.div(placed,8n),Nat.add(_,above)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(offset_word),Nat.sub(8n,Nat.mod(st,8n)),offset_value) :
        {Nat.is_le(Nat.div(Nat.add(_,Nat.mul(8n,g)),8n),Nat.add(Nat.add(whole,1n),above)) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.sub(8n,Nat.mod(st,8n)),Nat.mul(8n,g)),8n),Nat.add(whole,above),first_at) :
        {Nat.is_le(_,Nat.add(Nat.add(whole,1n),above)) == True{} : Bool}
      %grouped : {Nat.is_le(Nat.add(whole,above),_) == True{} : Bool}
      Order.offset_le(whole,above,Nat.add(1n,above),Order.right_le(1n,above)) :
        {Nat.is_le(Nat.div(placed,8n),Nat.add(U32.to_nat(shift_word),U32.to_nat(above_word))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(above_word),above,above_value) :
        {Nat.is_le(Nat.add(U32.to_nat(shift_word),_),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(shift_word),Nat.add(whole,1n),shift_value) :
        {Nat.is_le(Nat.add(_,above),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(offset_word),Nat.sub(8n,Nat.mod(st,8n)),offset_value) :
        {Nat.is_le(Nat.add(Nat.add(whole,1n),above),Nat.div(Nat.sub(Nat.add(Nat.add(_,Nat.mul(8n,g)),8n),1n),8n)) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.sub(Nat.add(Nat.add(Nat.sub(8n,Nat.mod(st,8n)),Nat.mul(8n,g)),8n),1n),8n),Nat.add(whole,last),last_at) :
        {Nat.is_le(Nat.add(Nat.add(whole,1n),above),_) == True{} : Bool}
      %grouped : {Nat.is_le(_,Nat.add(whole,last)) == True{} : Bool}
      Order.offset_le(whole,Nat.add(1n,above),last,multiple_quotient(Nat.add(1n,above),Nat.add(reach,7n),8n,{==},covered)) :
        {Nat.is_le(Nat.add(U32.to_nat(shift_word),U32.to_nat(above_word)),Nat.div(Nat.sub(Nat.add(placed,8n),1n),8n)) == True{} : Bool}},
    Equal.trans(Bool,Nat.is_lt(U32.to_nat(above_word),U32.to_nat(bound)),U32.is_lt(above_word,bound),True{},
      Equal.sym(Bool,U32.is_lt(above_word,bound),Nat.is_lt(U32.to_nat(above_word),U32.to_nat(bound)),Comparison.lt(above_word,bound)),below))

# ---- The part row places the values of a target cell ----

# The part group a target group reads: the one before the group its reach
# ends in, or that group.
def GroupValue(late: Bool,+start: U32,+bound: U32,+group: U32,+value: Nat) -> Data:
  match late:
    case False{}:
      Held.Both<{value == U32.to_nat((((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32)) : Nat},
        Held.Both<{U32.is_le(8,(8 * group + 8 - start : U32)) == True{} : Bool},
          {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}>>
    case True{}:
      Held.Both<{value == U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)) : Nat},
        Held.Both<{U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool},
          {U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound) == True{} : Bool}>>

# The value a part axis reads at a digit of the target cell: the part group
# on the groups axis, the digit on every other.
def ValueAt(same: Bool,+extent: U32,+digit: Nat,+value: Nat,+start: U32,+bound: U32,+late: Bool,+group: U32,+cap: U32) -> Data:
  match same:
    case True{}:
      Held.Both<{digit == U32.to_nat(group) : Nat},Held.Both<{Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
        Held.Both<{Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool},GroupValue(late,start,bound,group,value)>>>
    case False{}:
      Held.Both<{value == digit : Nat},{Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}>

def PartValues(extents: List<&2,U32>,digits: List<&2,Nat>,values: List<&2,Nat>,+groups_axis: U32,+axis: U32,+start: U32,+bound: U32,+late: Bool,+group: U32,
  +cap: U32) -> Data:
  match extents digits values:
    case Con{+extent,extents} Con{+digit,digits} Con{+value,values}:
      Held.Both<ValueAt(U32.is_eq(axis,groups_axis),extent,digit,value,start,bound,late,group,cap),
        PartValues(extents,digits,values,groups_axis,(axis + 1 : U32),start,bound,late,group,cap)>
    case Nil{} Nil{} Nil{}:
      Unit
    case extents digits values:
      Empty

# A part axis places the value it reads at the digit of a target cell in the
# class of the digit.
def part_axis(same: Bool,late: Bool,class: Tasks.Class,+extent: U32,+digit: Nat,+value: Nat,+part_grid: Grid.Grid,+grid: Grid.Grid,+start: U32,+bound: U32,
  +group: U32,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +class_in: Targets.ClassIn(class,Grid.block(grid,axis,extent),extent,digit),
  +room: Components.ReadingRoom(extent,Grid.block(grid,axis,extent),Bool.pick(Tasks.Source,same,
    Tasks.Read{axis,Grid.block(part_grid,axis,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},
    Tiled.read_at(part_grid,axis,extent,1,1,1)),cap),
  +at: ValueAt(same,extent,digit,value,start,bound,late,group,cap)) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,Grid.block(grid,axis,extent),Bool.pick(Tasks.Source,same,
    Tasks.Read{axis,Grid.block(part_grid,axis,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},
    Tiled.read_at(part_grid,axis,extent,1,1,1))},class,value,cap):
  match same:
    case True{}:
      match late:
        case False{}:
          match class:
            case Tasks.Class{+first,+count,+source_first,+source_count}:
              Held.Both{+digit_same,Held.Both{+from,Held.Both{+end_room,Held.Both{+value_same,Held.Both{+reaches,+within}}}}} = at
              +block = Grid.block(grid,axis,extent)
              %Equal.sym(Nat,value,U32.to_nat((((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32)),value_same) :
                TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{axis,Grid.block(part_grid,axis,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}}},
                  Tasks.Class{first,count,source_first,source_count},_,cap)
              above_placed(extent,block,axis,Grid.block(part_grid,axis,bound),start,bound,first,count,source_first,source_count,group,cap,cap_ok,room,
                {%digit_same : Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,_)
                  class_in : Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,U32.to_nat(group))},from,end_room,reaches,within)
        case True{}:
          match class:
            case Tasks.Class{+first,+count,+source_first,+source_count}:
              Held.Both{+digit_same,Held.Both{+from,Held.Both{+end_room,Held.Both{+value_same,Held.Both{+shifted,+below}}}}} = at
              +block = Grid.block(grid,axis,extent)
              %Equal.sym(Nat,value,U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),value_same) :
                TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{axis,Grid.block(part_grid,axis,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}}},
                  Tasks.Class{first,count,source_first,source_count},_,cap)
              beside_placed(extent,block,axis,Grid.block(part_grid,axis,bound),start,bound,first,count,source_first,source_count,group,cap,cap_ok,room,
                {%digit_same : Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,_)
                  class_in : Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,U32.to_nat(group))},from,end_room,shifted,below)
    case False{}:
      match class:
        case Tasks.Class{+first,+count,+source_first,+source_count}:
          Held.Both{+value_same,+below} = at
          +block = Grid.block(grid,axis,extent)
          %Equal.sym(Nat,value,digit,value_same) :
            TaskReads.PlacedIn(Tasks.Reading{extent,block,Tiled.read_at(part_grid,axis,extent,1,1,1)},Tasks.Class{first,count,source_first,source_count},_,cap)
          TaskReads.shared_placed(extent,block,axis,Grid.block(part_grid,axis,extent),extent,first,count,source_first,source_count,digit,cap,cap_ok,room,class_in,below)

# The part row places the values a target cell reads.
def part_row(extents: List<&2,U32>,classes: List<&2,Tasks.Class>,digits: List<&2,Nat>,values: List<&2,Nat>,+part_grid: Grid.Grid,+grid: Grid.Grid,
  +groups_axis: U32,+start: U32,+bound: U32,+late: Bool,+group: U32,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +classes_in: Targets.ClassesIn(classes,grid,extents,digits,axis),
  +rooms: Components.ReadingsRoom(Tiled.part_readings(extents,part_grid,grid,groups_axis,start,bound,axis),cap),
  +values_ok: PartValues(extents,digits,values,groups_axis,axis,start,bound,late,group,cap)) ->
  TaskReads.PlacedRow(Tiled.part_readings(extents,part_grid,grid,groups_axis,start,bound,axis),classes,values,cap):
  match extents classes digits values:
    case Con{+extent,+rest} Con{+class,+more} Con{+digit,+later} Con{+value,+others}:
      Held.Both{+class_in,+classes_rest} = classes_in
      Held.Both{+room,+rooms_rest} = rooms
      Held.Both{+at,+values_rest} = values_ok
      Held.Both{part_axis(U32.is_eq(axis,groups_axis),late,class,extent,digit,value,part_grid,grid,start,bound,group,axis,cap,cap_ok,class_in,room,at),
        part_row(rest,more,later,others,part_grid,grid,groups_axis,start,bound,late,group,(axis + 1 : U32),cap,cap_ok,classes_rest,rooms_rest,values_rest)}
    case Con{+extent,+rest} Con{+class,+more} Con{+digit,+later} Nil{}:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Con{extent,rest},part_grid,grid,groups_axis,start,bound,axis),Con{class,more},Nil{},cap),values_ok)
    case Con{+extent,+rest} Con{+class,+more} Nil{} +values:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Con{extent,rest},part_grid,grid,groups_axis,start,bound,axis),Con{class,more},values,cap),classes_in)
    case Con{+extent,+rest} Nil{} +digits +values:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Con{extent,rest},part_grid,grid,groups_axis,start,bound,axis),Nil{},values,cap),classes_in)
    case Nil{} Nil{} Nil{} Nil{}:
      Unit{}
    case Nil{} Nil{} Nil{} Con{+value,+others}:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Nil{},part_grid,grid,groups_axis,start,bound,axis),Nil{},Con{value,others},cap),values_ok)
    case Nil{} Nil{} Con{+digit,+later} +values:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Nil{},part_grid,grid,groups_axis,start,bound,axis),Nil{},values,cap),classes_in)
    case Nil{} Con{+class,+more} +digits +values:
      Empty.absurd(TaskReads.PlacedRow(Tiled.part_readings(Nil{},part_grid,grid,groups_axis,start,bound,axis),Con{class,more},values,cap),classes_in)

# ---- The part row reads every part axis at its own index ----

# Every reading of a row reads the source axis of its index.
def ReadsSelf(readings: List<&2,Tasks.Reading>,+axis: U32) -> Data:
  match readings:
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},rest}:
      Held.Both<{other == axis : U32},ReadsSelf(rest,(axis + 1 : U32))>
    case Con{reading,rest}:
      Empty
    case Nil{}:
      Unit

def self_head(same: Bool,+extent: U32,+rest: List<&2,Tasks.Reading>,+part_grid: Grid.Grid,+grid: Grid.Grid,+start: U32,+bound: U32,+axis: U32,
  +later: ReadsSelf(rest,(axis + 1 : U32))) ->
  ReadsSelf(Con{Tasks.Reading{extent,Grid.block(grid,axis,extent),Bool.pick(Tasks.Source,same,
    Tasks.Read{axis,Grid.block(part_grid,axis,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},
    Tiled.read_at(part_grid,axis,extent,1,1,1))},rest},axis):
  match same:
    case True{}:
      Held.Both{{==},later}
    case False{}:
      Held.Both{{==},later}

def part_self(extents: List<&2,U32>,+part_grid: Grid.Grid,+grid: Grid.Grid,+groups_axis: U32,+start: U32,+bound: U32,+axis: U32) ->
  ReadsSelf(Tiled.part_readings(extents,part_grid,grid,groups_axis,start,bound,axis),axis):
  match extents:
    case Con{+extent,+rest}:
      self_head(U32.is_eq(axis,groups_axis),extent,Tiled.part_readings(rest,part_grid,grid,groups_axis,start,bound,(axis + 1 : U32)),part_grid,grid,start,bound,axis,
        part_self(rest,part_grid,grid,groups_axis,start,bound,(axis + 1 : U32)))
    case Nil{}:
      Unit{}

# The values from skip on.
def digits_from(skip: Nat,values: List<&2,Nat>) -> List<&2,Nat>:
  match skip values:
    case 0n values:
      values
    case 1n+next Con{value,rest}:
      digits_from(next,rest)
    case 1n+next Nil{}:
      Nil{}

def from_split(values: List<&2,Nat>,skip: Nat,+inside: {Nat.is_lt(skip,List.length(&2,Nat,values)) == True{} : Bool}) ->
  {digits_from(skip,values) == Con{BoxDigits.digit_at(values,skip),digits_from(1n+skip,values)} : List<&2,Nat>}:
  match values skip:
    case Con{+value,+rest} 0n:
      {==}
    case Con{+value,+rest} 1n+ +next:
      from_split(rest,next,inside)
    case Nil{} +skip:
      Empty.absurd({digits_from(skip,Nil{}) == Con{BoxDigits.digit_at(Nil{},skip),digits_from(1n+skip,Nil{})} : List<&2,Nat>},Order.below_zero(skip,inside))

def from_end(values: List<&2,Nat>) -> {digits_from(List.length(&2,Nat,values),values) == Nil{} : List<&2,Nat>}:
  match values:
    case Nil{}:
      {==}
    case Con{+value,+rest}:
      from_end(rest)

# A row whose readings read their own axes places at a source axis the value
# of the reading of that axis.
def digit_scan(readings: List<&2,Tasks.Reading>,values: List<&2,Nat>,+axis: U32,+target: U32,skip: Nat,+fixed: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+self: ReadsSelf(readings,axis),
  +paired: {List.length(&2,Tasks.Reading,readings) == List.length(&2,Nat,values) : Nat},
  +exact: {U32.to_nat(target) == Nat.add(U32.to_nat(axis),skip) : Nat},+target_room: {Nat.is_lt(U32.to_nat(target),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(skip,List.length(&2,Nat,values)) == True{} : Bool}) ->
  {TaskReads.placed_digit(readings,values,target,fixed) == BoxDigits.digit_at(values,skip) : Nat}:
  match readings values skip:
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Con{+value,+more} 0n:
      Held.Both{+same,+later} = self
      +later_digit = TaskReads.placed_digit(rest,more,target,fixed)
      +at = Arithmetic.to_nat_injective(target,axis,Equal.trans(Nat,U32.to_nat(target),Nat.add(U32.to_nat(axis),0n),U32.to_nat(axis),exact,
        Addition.zero_right(U32.to_nat(axis))))
      %Equal.sym(U32,other,axis,same) : {Bool.pick(Nat,U32.is_eq(_,target),value,later_digit) == value : Nat}
      %Equal.sym(U32,target,axis,at) : {Bool.pick(Nat,U32.is_eq(axis,_),value,later_digit) == value : Nat}
      %Equal.sym(Bool,U32.is_eq(axis,axis),True{},Arithmetic.equal_reflexive(axis)) : {Bool.pick(Nat,_,value,later_digit) == value : Nat}
      {==}
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Con{+value,+more} 1n+ +next:
      Held.Both{+same,+later} = self
      +later_digit = TaskReads.placed_digit(rest,more,target,fixed)
      +start = U32.to_nat(axis)
      +below = {%Equal.sym(Nat,U32.to_nat(target),Nat.add(start,1n+next),exact) : {Nat.is_lt(start,_) == True{} : Bool}
        Components.below_plus(start,1n+next,{==}) : {Nat.is_lt(start,U32.to_nat(target)) == True{} : Bool}}
      +apart = Equal.trans(Bool,U32.is_eq(axis,target),Nat.is_eq(start,U32.to_nat(target)),False{},Comparison.eq(axis,target),
        Order.less_not_equal(start,U32.to_nat(target),below))
      +next_word = Gather.word_next(axis,cap,cap_ok,Bounds.nat_trans(start,U32.to_nat(target),U32.to_nat(cap),below,Bounds.strict_le(U32.to_nat(target),U32.to_nat(cap),target_room)))
      %Equal.sym(U32,other,axis,same) : {Bool.pick(Nat,U32.is_eq(_,target),value,later_digit) == BoxDigits.digit_at(more,next) : Nat}
      %Equal.sym(Bool,U32.is_eq(axis,target),False{},apart) : {Bool.pick(Nat,_,value,later_digit) == BoxDigits.digit_at(more,next) : Nat}
      digit_scan(rest,more,(axis + 1 : U32),target,next,fixed,cap,cap_ok,later,
        Components.length_back(List.length(&2,Tasks.Reading,rest),List.length(&2,Nat,more),paired),
        {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+start,next_word) : {U32.to_nat(target) == Nat.add(_,next) : Nat}
          Equal.trans(Nat,U32.to_nat(target),Nat.add(start,1n+next),1n+Nat.add(start,next),exact,Addition.successor_right(start,next)) :
          {U32.to_nat(target) == Nat.add(U32.to_nat((axis + 1 : U32)),next) : Nat}},target_room,inside)
    case Con{Tasks.Reading{+extent,+block,Tasks.Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},+rest} Nil{} +skip:
      Empty.absurd({TaskReads.placed_digit(Con{Tasks.Reading{extent,block,Tasks.Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest},
        Nil{},target,fixed) == BoxDigits.digit_at(Nil{},skip) : Nat},Order.below_zero(skip,inside))
    case Con{Tasks.Reading{+extent,+block,Tasks.Unread{}},+rest} +values +skip:
      Empty.absurd({TaskReads.placed_digit(Con{Tasks.Reading{extent,block,Tasks.Unread{}},rest},values,target,fixed) == BoxDigits.digit_at(values,skip) : Nat},self)
    case Con{Tasks.Reading{+extent,+block,Tasks.Crossing{}},+rest} +values +skip:
      Empty.absurd({TaskReads.placed_digit(Con{Tasks.Reading{extent,block,Tasks.Crossing{}},rest},values,target,fixed) == BoxDigits.digit_at(values,skip) : Nat},self)
    case Nil{} Nil{} +skip:
      Empty.absurd({TaskReads.placed_digit(Nil{},Nil{},target,fixed) == BoxDigits.digit_at(Nil{},skip) : Nat},Order.below_zero(skip,inside))
    case Nil{} Con{+value,+more} +skip:
      Empty.absurd({TaskReads.placed_digit(Nil{},Con{value,more},target,fixed) == BoxDigits.digit_at(Con{value,more},skip) : Nat},
        Components.zero_successor(List.length(&2,Nat,more),paired))

# A row whose readings read their own axes places its values.
def placed_self(+readings: List<&2,Tasks.Reading>,+values: List<&2,Nat>,extents: List<&2,U32>,fixed: List<&2,U32>,+axis: U32,+skip: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+self: ReadsSelf(readings,0),
  +paired: {List.length(&2,Tasks.Reading,readings) == List.length(&2,Nat,values) : Nat},
  +fixed_count: {List.length(&2,U32,fixed) == List.length(&2,U32,extents) : Nat},+exact: {U32.to_nat(axis) == skip : Nat},
  +remaining: {Nat.add(skip,List.length(&2,U32,extents)) == List.length(&2,Nat,values) : Nat},
  +room: {Nat.is_lt(List.length(&2,Nat,values),U32.to_nat(cap)) == True{} : Bool}) ->
  {TaskReads.placed_digits(readings,values,extents,fixed,axis) == digits_from(skip,values) : List<&2,Nat>}:
  match extents fixed:
    case Con{+extent,+rest} Con{+index,+more}:
      +count = List.length(&2,Nat,values)
      +inside = {%remaining : {Nat.is_lt(skip,_) == True{} : Bool}
        Components.below_plus(skip,1n+List.length(&2,U32,rest),{==}) : {Nat.is_lt(skip,count) == True{} : Bool}}
      +target_room = {%Equal.sym(Nat,U32.to_nat(axis),skip,exact) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
        Bounds.nat_trans(skip,count,U32.to_nat(cap),inside,Bounds.strict_le(count,U32.to_nat(cap),room)) : {Nat.is_lt(U32.to_nat(axis),U32.to_nat(cap)) == True{} : Bool}}
      +head = digit_scan(readings,values,0,axis,skip,U32.to_nat(index),cap,cap_ok,self,paired,exact,target_room,inside)
      +next_word = Gather.word_next(axis,cap,cap_ok,target_room)
      +tail = placed_self(readings,values,rest,more,(axis + 1 : U32),1n+skip,cap,cap_ok,self,paired,
        Components.length_back(List.length(&2,U32,more),List.length(&2,U32,rest),fixed_count),
        Equal.trans(Nat,U32.to_nat((axis + 1 : U32)),1n+U32.to_nat(axis),1n+skip,next_word,Equal.cong(Nat,Nat,value => 1n+value,U32.to_nat(axis),skip,exact)),
        Equal.trans(Nat,Nat.add(1n+skip,List.length(&2,U32,rest)),Nat.add(skip,1n+List.length(&2,U32,rest)),count,
          Equal.sym(Nat,Nat.add(skip,1n+List.length(&2,U32,rest)),1n+Nat.add(skip,List.length(&2,U32,rest)),Addition.successor_right(skip,List.length(&2,U32,rest))),
          remaining),room)
      +first = TaskReads.placed_digit(readings,values,axis,U32.to_nat(index))
      +later = TaskReads.placed_digits(readings,values,rest,more,(axis + 1 : U32))
      Equal.trans(List<&2,Nat>,Con{first,later},Con{BoxDigits.digit_at(values,skip),digits_from(1n+skip,values)},digits_from(skip,values),
        Equal.trans(List<&2,Nat>,Con{first,later},Con{BoxDigits.digit_at(values,skip),later},Con{BoxDigits.digit_at(values,skip),digits_from(1n+skip,values)},
          Equal.cong(Nat,List<&2,Nat>,digit => Con{digit,later},first,BoxDigits.digit_at(values,skip),head),
          Equal.cong(List<&2,Nat>,List<&2,Nat>,tail_digits => Con{BoxDigits.digit_at(values,skip),tail_digits},later,digits_from(1n+skip,values),tail)),
        Equal.sym(List<&2,Nat>,digits_from(skip,values),Con{BoxDigits.digit_at(values,skip),digits_from(1n+skip,values)},from_split(values,skip,inside)))
    case Nil{} Nil{}:
      +ended = Equal.trans(Nat,skip,Nat.add(skip,0n),List.length(&2,Nat,values),Equal.sym(Nat,Nat.add(skip,0n),skip,Addition.zero_right(skip)),remaining)
      %Equal.sym(Nat,skip,List.length(&2,Nat,values),ended) : {Nil{} == digits_from(_,values) : List<&2,Nat>}
      Equal.sym(List<&2,Nat>,digits_from(List.length(&2,Nat,values),values),Nil{},from_end(values))
    case Nil{} Con{+index,+more}:
      Empty.absurd({TaskReads.placed_digits(readings,values,Nil{},Con{index,more},axis) == digits_from(skip,values) : List<&2,Nat>},
        Components.zero_successor(List.length(&2,U32,more),Equal.sym(Nat,1n+List.length(&2,U32,more),0n,fixed_count)))
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({TaskReads.placed_digits(readings,values,Con{extent,rest},Nil{},axis) == digits_from(skip,values) : List<&2,Nat>},
        Components.zero_successor(List.length(&2,U32,rest),fixed_count))

# ---- The values a target cell reads ----

# The other axes before the groups axis read their digits, the groups axis
# last reads the part group.
def inner_values(others: List<&2,U32>,digits: List<&2,Nat>,+groups: U32,+digit: Nat,+value: Nat,+groups_axis: U32,+axis: U32,+start: U32,+bound: U32,
  +late: Bool,+group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: Coordinates.DigitsBelow(others,digits),
  +at: {Nat.add(U32.to_nat(axis),List.length(&2,U32,others)) == U32.to_nat(groups_axis) : Nat},
  +axis_room: {Nat.is_lt(U32.to_nat(groups_axis),U32.to_nat(cap)) == True{} : Bool},+last: ValueAt(True{},groups,digit,value,start,bound,late,group,cap)) ->
  PartValues(List.append(&2,U32,others,[groups]),List.append(&2,Nat,digits,[digit]),List.append(&2,Nat,digits,[value]),groups_axis,axis,start,bound,late,group,cap):
  match others digits:
    case Con{+extent,+rest} Con{+head,+more}:
      KernelGemm.Both{+here,+later} = below
      +position = U32.to_nat(axis)
      +before = {%at : {Nat.is_lt(position,_) == True{} : Bool}
        Components.below_plus(position,1n+List.length(&2,U32,rest),{==}) : {Nat.is_lt(position,U32.to_nat(groups_axis)) == True{} : Bool}}
      +apart = Equal.trans(Bool,U32.is_eq(axis,groups_axis),Nat.is_eq(position,U32.to_nat(groups_axis)),False{},Comparison.eq(axis,groups_axis),
        Order.less_not_equal(position,U32.to_nat(groups_axis),before))
      +next = Gather.word_next(axis,cap,cap_ok,Bounds.nat_trans(position,U32.to_nat(groups_axis),U32.to_nat(cap),before,Bounds.strict_le(U32.to_nat(groups_axis),U32.to_nat(cap),axis_room)))
      Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,groups_axis),False{},apart) : ValueAt(_,extent,head,head,start,bound,late,group,cap)
          Held.Both{{==},here} : ValueAt(U32.is_eq(axis,groups_axis),extent,head,head,start,bound,late,group,cap)},
        inner_values(rest,more,groups,digit,value,groups_axis,(axis + 1 : U32),start,bound,late,group,cap,cap_ok,later,
          {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.add(_,List.length(&2,U32,rest)) == U32.to_nat(groups_axis) : Nat}
            Equal.trans(Nat,1n+Nat.add(position,List.length(&2,U32,rest)),Nat.add(position,1n+List.length(&2,U32,rest)),U32.to_nat(groups_axis),
              Equal.sym(Nat,Nat.add(position,1n+List.length(&2,U32,rest)),1n+Nat.add(position,List.length(&2,U32,rest)),Addition.successor_right(position,List.length(&2,U32,rest))),
              at) : {Nat.add(U32.to_nat((axis + 1 : U32)),List.length(&2,U32,rest)) == U32.to_nat(groups_axis) : Nat}},axis_room,last)}
    case Nil{} Nil{}:
      +same = Arithmetic.to_nat_injective(axis,groups_axis,Equal.trans(Nat,U32.to_nat(axis),Nat.add(U32.to_nat(axis),0n),U32.to_nat(groups_axis),
        Equal.sym(Nat,Nat.add(U32.to_nat(axis),0n),U32.to_nat(axis),Addition.zero_right(U32.to_nat(axis))),at))
      +hit = {%same : {U32.is_eq(axis,_) == True{} : Bool} Arithmetic.equal_reflexive(axis) : {U32.is_eq(axis,groups_axis) == True{} : Bool}}
      Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,groups_axis),True{},hit) : ValueAt(_,groups,digit,value,start,bound,late,group,cap)
          last : ValueAt(U32.is_eq(axis,groups_axis),groups,digit,value,start,bound,late,group,cap)},Unit{}}
    case Nil{} Con{+head,+more}:
      Empty.absurd(PartValues(List.append(&2,U32,Nil{},[groups]),List.append(&2,Nat,Con{head,more},[digit]),List.append(&2,Nat,Con{head,more},[value]),groups_axis,axis,
        start,bound,late,group,cap),below)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(PartValues(List.append(&2,U32,Con{extent,rest},[groups]),List.append(&2,Nat,Nil{},[digit]),List.append(&2,Nat,Nil{},[value]),groups_axis,axis,
        start,bound,late,group,cap),below)

# The other axes after the groups axis read their digits.
def outer_values(others: List<&2,U32>,digits: List<&2,Nat>,+groups_axis: U32,+axis: U32,+start: U32,+bound: U32,+late: Bool,+group: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: Coordinates.DigitsBelow(others,digits),
  +after: {Nat.is_lt(U32.to_nat(groups_axis),U32.to_nat(axis)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,others)),U32.to_nat(cap)) == True{} : Bool}) ->
  PartValues(others,digits,digits,groups_axis,axis,start,bound,late,group,cap):
  match others digits:
    case Con{+extent,+rest} Con{+head,+more}:
      KernelGemm.Both{+here,+later} = below
      +position = U32.to_nat(axis)
      +apart = Equal.trans(Bool,U32.is_eq(axis,groups_axis),Nat.is_eq(position,U32.to_nat(groups_axis)),False{},Comparison.eq(axis,groups_axis),
        Equal.trans(Bool,Nat.is_eq(position,U32.to_nat(groups_axis)),Nat.is_eq(U32.to_nat(groups_axis),position),False{},
          Order.equal_symmetric(position,U32.to_nat(groups_axis)),Order.less_not_equal(U32.to_nat(groups_axis),position,after)))
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(position,List.length(&2,U32,rest),U32.to_nat(cap),room))
      Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,groups_axis),False{},apart) : ValueAt(_,extent,head,head,start,bound,late,group,cap)
          Held.Both{{==},here} : ValueAt(U32.is_eq(axis,groups_axis),extent,head,head,start,bound,late,group,cap)},
        outer_values(rest,more,groups_axis,(axis + 1 : U32),start,bound,late,group,cap,cap_ok,later,
          {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_lt(U32.to_nat(groups_axis),_) == True{} : Bool}
            Order.lt_succ(U32.to_nat(groups_axis),position,Bounds.strict_le(U32.to_nat(groups_axis),position,after)) :
            {Nat.is_lt(U32.to_nat(groups_axis),U32.to_nat((axis + 1 : U32))) == True{} : Bool}},
          Gather.range_rest(axis,List.length(&2,U32,rest),U32.to_nat(cap),next,room))}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+head,+more}:
      Empty.absurd(PartValues(Nil{},Con{head,more},Con{head,more},groups_axis,axis,start,bound,late,group,cap),below)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(PartValues(Con{extent,rest},Nil{},Nil{},groups_axis,axis,start,bound,late,group,cap),below)

# A target cell reads its other digits and the part group at the groups axis.
def cell_values(position: Grid.Position,+others: List<&2,U32>,+digits: List<&2,Nat>,+groups: U32,+digit: Nat,+value: Nat,+start: U32,+bound: U32,+late: Bool,
  +group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+below: Coordinates.DigitsBelow(others,digits),
  +rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(cap)) == True{} : Bool},+last: ValueAt(True{},groups,digit,value,start,bound,late,group,cap)) ->
  PartValues(Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,digits,digit),Coordinates.cell_digits(position,digits,value),
    Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),0,start,bound,late,group,cap):
  match position:
    case Grid.Innermost{}:
      +cells = List.append(&2,U32,others,[groups])
      +count = {1n+List.length(&2,U32,others) : Nat}
      +counted = Coordinates.placed_count(Grid.Innermost{},others,groups)
      +count_room = Bounds.strict_le(count,U32.to_nat(cap),rank_room)
      +word = Equal.trans(Nat,U32.to_nat(U32.from_nat(List.length(&2,U32,cells))),List.length(&2,U32,cells),count,
        Bounds.roundtrip(List.length(&2,U32,cells),cap,cap_ok,{%Equal.sym(Nat,List.length(&2,U32,cells),count,counted) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} count_room :
          {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool}}),counted)
      +last_axis = Equal.trans(Nat,U32.to_nat((U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),Nat.sub(count,1n),List.length(&2,U32,others),
        Arithmetic.difference_exact(cap,cap_ok,U32.from_nat(List.length(&2,U32,cells)),1,count,1n,word,{==},Order.zero_le(List.length(&2,U32,others)),count_room),
        Order.subtract_zero(List.length(&2,U32,others)))
      inner_values(others,digits,groups,digit,value,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32),0,start,bound,late,group,cap,cap_ok,below,
        Equal.sym(Nat,U32.to_nat((U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),List.length(&2,U32,others),last_axis),
        {%Equal.sym(Nat,U32.to_nat((U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),List.length(&2,U32,others),last_axis) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
          Bounds.nat_trans(List.length(&2,U32,others),count,U32.to_nat(cap),Order.lt_succ(List.length(&2,U32,others),List.length(&2,U32,others),
            Order.reflexive(List.length(&2,U32,others))),count_room) :
          {Nat.is_lt(U32.to_nat((U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),U32.to_nat(cap)) == True{} : Bool}},last)
    case Grid.Outermost{}:
      Held.Both{last,outer_values(others,digits,0,1,start,bound,late,group,cap,cap_ok,below,{==},Bounds.strict_le(1n+List.length(&2,U32,others),U32.to_nat(cap),rank_room))}

# ---- A target cell reads part cells its component takes ----

# The part row has a reading for each target cell axis.
def part_count(extents: List<&2,U32>,+part_grid: Grid.Grid,+grid: Grid.Grid,+groups_axis: U32,+start: U32,+bound: U32,+axis: U32) ->
  {List.length(&2,Tasks.Reading,Tiled.part_readings(extents,part_grid,grid,groups_axis,start,bound,axis)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tasks.Reading,Tiled.part_readings(rest,part_grid,grid,groups_axis,start,bound,(axis + 1 : U32))),
        List.length(&2,U32,rest),part_count(rest,part_grid,grid,groups_axis,start,bound,(axis + 1 : U32)))

# A target cell of a separable lane write reads, in the part group of a read,
# a part cell the boxes of its component hold.
def part_cell(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+late: Bool,+group: U32,+other: U32,+read: U32,
  +rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),
  +blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),
  +part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},
  +part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+other_inside: {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool},
  +classes_in: Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,Targets.cell_coordinates(others,U32.to_nat(other)),U32.to_nat(group)),0),
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +value: GroupValue(late,start,bound,group,U32.to_nat(read))) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),
    U32.to_nat(Tiled.cell_of(position,other,read,bound,count))):
  +cells = Grid.placed(True{},position,others,groups)
  +part_cells = Grid.placed(True{},position,others,bound)
  +grid = Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target())
  +part_grid = Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())
  +axis = Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32))
  +row = Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +capacity = Machine.capacity(limits)
  +capacity_ok = Limits.capacity_fits(limits)
  +own = Targets.cell_coordinates(others,U32.to_nat(other))
  +below = BoxDigits.coordinates_below(others,U32.to_nat(other),other_inside)
  +digits = Coordinates.cell_digits(position,own,U32.to_nat(group))
  +values = Coordinates.cell_digits(position,own,U32.to_nat(read))
  +values_ok = cell_values(position,others,own,groups,U32.to_nat(group),U32.to_nat(read),start,bound,late,group,cap,cap_ok,below,rank_room,
    Held.Both{{==},Held.Both{from,Held.Both{end_room,value}}})
  +placed_row = part_row(cells,classes,digits,values,part_grid,grid,axis,start,bound,late,group,0,cap,cap_ok,classes_in,rooms,values_ok)
  +fixed = Grid.coordinates(part_cells,0)
  +found = TaskReads.digits_placed(row,classes,values,part_grid,part_cells,fixed,0,cap,cap_ok,placed_row,axes,blocks,
    ViewReads.coordinates_in(part_cells,0,cap,cap_ok,Limits.double_bound(limits,ShapeLaws.elements(part_cells),part_room),part_nonempty))
  +paired = TaskReads.row_paired(row,classes,values,cap,placed_row)
  +same_count = Equal.trans(Nat,List.length(&2,U32,part_cells),1n+List.length(&2,U32,others),List.length(&2,U32,cells),
    Coordinates.placed_count(position,others,bound),Equal.sym(Nat,List.length(&2,U32,cells),1n+List.length(&2,U32,others),Coordinates.placed_count(position,others,groups)))
  +remaining = Equal.trans(Nat,List.length(&2,U32,part_cells),List.length(&2,U32,cells),List.length(&2,Nat,values),same_count,
    Equal.trans(Nat,List.length(&2,U32,cells),List.length(&2,Tasks.Reading,row),List.length(&2,Nat,values),
      Equal.sym(Nat,List.length(&2,Tasks.Reading,row),List.length(&2,U32,cells),part_count(cells,part_grid,grid,axis,start,bound,0)),paired))
  +values_room = {%remaining : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,U32,part_cells),1n+List.length(&2,U32,others),Coordinates.placed_count(position,others,bound)) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool}
    rank_room : {Nat.is_lt(List.length(&2,Nat,values),U32.to_nat(cap)) == True{} : Bool}}
  +same_digits = placed_self(row,values,part_cells,fixed,0,0n,cap,cap_ok,part_self(cells,part_grid,grid,axis,start,bound,0),paired,
    BoxDigits.coordinates_count(part_cells,0),{==},remaining,values_room)
  +box = Tasks.boxed(row,classes,part_grid,part_cells,fixed,0)
  +inside = {%same_digits : ViewReads.DigitsIn(box,part_cells,_) found : ViewReads.DigitsIn(box,part_cells,values)}
  +boxed_digits = ViewReads.digits_box(box,part_cells,values,inside)
  +number = ShapeLaws.digits_value(part_cells,values)
  +number_below = Held.second_of(Components.InBox(box,part_cells,number),{Nat.is_lt(number,ShapeLaws.elements(part_cells)) == True{} : Bool},boxed_digits)
  +number_room = Order.transitive(number,ShapeLaws.elements(part_cells),U32.to_nat(capacity),Bounds.strict_le(number,ShapeLaws.elements(part_cells),number_below),part_room)
  +index = Representation.index_at(position,U32.to_nat(count),U32.to_nat(bound),U32.to_nat(other),U32.to_nat(read))
  +number_index = Equal.trans(Nat,number,
    Representation.index_at(position,ShapeLaws.elements(others),U32.to_nat(bound),ShapeLaws.digits_value(others,own),U32.to_nat(read)),index,
    Coordinates.cell_number(position,own,U32.to_nat(read),others,bound,below),
    Equal.trans(Nat,Representation.index_at(position,ShapeLaws.elements(others),U32.to_nat(bound),ShapeLaws.digits_value(others,own),U32.to_nat(read)),
      Representation.index_at(position,U32.to_nat(count),U32.to_nat(bound),ShapeLaws.digits_value(others,own),U32.to_nat(read)),index,
      Equal.cong(Nat,Nat,cells_of => Representation.index_at(position,cells_of,U32.to_nat(bound),ShapeLaws.digits_value(others,own),U32.to_nat(read)),
        ShapeLaws.elements(others),U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(others),count_exact)),
      Equal.cong(Nat,Nat,at => Representation.index_at(position,U32.to_nat(count),U32.to_nat(bound),at,U32.to_nat(read)),
        ShapeLaws.digits_value(others,own),U32.to_nat(other),BoxDigits.value_coordinates(others,U32.to_nat(other),other_inside))))
  +word_exact = Gather.word_by_position(position,bound,count,capacity,capacity_ok,other,read,
    {%number_index : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} number_room : {Nat.is_le(index,U32.to_nat(capacity)) == True{} : Bool}})
  +exact = Arithmetic.to_nat_injective(Tiled.cell_of(position,other,read,bound,count),U32.from_nat(number),
    Equal.trans(Nat,U32.to_nat(Tiled.cell_of(position,other,read,bound,count)),index,U32.to_nat(U32.from_nat(number)),word_exact,
      Equal.trans(Nat,index,number,U32.to_nat(U32.from_nat(number)),Equal.sym(Nat,number,index,number_index),
        Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,Bounds.roundtrip(number,capacity,capacity_ok,number_room)))))
  TaskReads.held_of(limits,part_cells,box,values,Tiled.cell_of(position,other,read,bound,count),part_room,part_nonempty,part_rank,
    BoxCells.boxed_coordinates_in(row,classes,part_grid,part_cells,0,cap,cap_ok,rooms,ends,axes),boxed_digits,exact)

# ---- Every part cell a target piece reads ----

# A target cell of an other index and a group in the target piece has the
# digits of its other index and its group, inside the classes of its
# component.
def cell_classes(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +group: U32,+other: U32,+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate: {Tasks.separable(readings) == True{} : Bool},+ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits)),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},
  +cell_room: {Nat.is_le(Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Tiled.cell_of(position,other,group,groups,count))) == True{} : Bool},+high: {Nat.is_lt(U32.to_nat(Tiled.cell_of(position,other,group,groups,count)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)))) == True{} : Bool}) ->
  Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,Targets.cell_coordinates(others,U32.to_nat(other)),U32.to_nat(group)),0):
  +cell = U32.to_nat(Tiled.cell_of(position,other,group,groups,count))
  +begin = U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece))
  +inside_others = {%count_exact : {Nat.is_lt(U32.to_nat(other),_) == True{} : Bool} other_inside : {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool}}
  +own = Targets.cell_coordinates(others,U32.to_nat(other))
  +below = BoxDigits.coordinates_below(others,U32.to_nat(other),inside_others)
  +digits = Coordinates.cell_digits(position,own,U32.to_nat(group))
  +digits_below = {%Coordinates.placed_cell(position,own,U32.to_nat(group)) : Coordinates.DigitsBelow(Grid.placed(True{},position,others,groups),_)
    Coordinates.placed_below(True{},position,others,groups,own,U32.to_nat(group),below,group_inside) : Coordinates.DigitsBelow(Grid.placed(True{},position,others,groups),digits)}
  +index = Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group))
  +numbered = Representation.index_at(position,ShapeLaws.elements(others),U32.to_nat(groups),ShapeLaws.digits_value(others,own),U32.to_nat(group))
  +value = Equal.trans(Nat,ShapeLaws.digits_value(Grid.placed(True{},position,others,groups),digits),numbered,cell,Coordinates.cell_number(position,own,U32.to_nat(group),others,groups,below),
    Equal.trans(Nat,numbered,index,cell,
      Equal.trans(Nat,numbered,Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),ShapeLaws.digits_value(others,own),U32.to_nat(group)),index,
        Equal.cong(Nat,Nat,cells_of => Representation.index_at(position,cells_of,U32.to_nat(groups),ShapeLaws.digits_value(others,own),U32.to_nat(group)),
          ShapeLaws.elements(others),U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(others),count_exact)),
        Equal.cong(Nat,Nat,at => Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),at,U32.to_nat(group)),
          ShapeLaws.digits_value(others,own),U32.to_nat(other),BoxDigits.value_coordinates(others,U32.to_nat(other),inside_others))),
      Equal.sym(Nat,cell,index,Gather.word_by_position(position,groups,count,Machine.capacity(limits),Limits.capacity_fits(limits),other,group,cell_room))))
  +restored = Order.restore_shift(begin,cell,low)
  +same = Equal.trans(List<&2,Nat>,Targets.cell_coordinates(Grid.placed(True{},position,others,groups),Nat.add(begin,Nat.sub(cell,begin))),Targets.cell_coordinates(Grid.placed(True{},position,others,groups),cell),digits,
    Equal.cong(Nat,List<&2,Nat>,number => Targets.cell_coordinates(Grid.placed(True{},position,others,groups),number),Nat.add(begin,Nat.sub(cell,begin)),cell,restored),
    {%value : {Targets.cell_coordinates(Grid.placed(True{},position,others,groups),_) == digits : List<&2,Nat>}
      BoxDigits.coordinates_value(Grid.placed(True{},position,others,groups),digits,digits_below) : {Targets.cell_coordinates(Grid.placed(True{},position,others,groups),cell) == digits : List<&2,Nat>}})
  +index_in = Order.cancel_offset_lt(begin,Nat.sub(cell,begin),U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),
    {%Equal.sym(Nat,Nat.add(begin,Nat.sub(cell,begin)),cell,restored) : {Nat.is_lt(_,Nat.add(begin,U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)))) == True{} : Bool}
      high : {Nat.is_lt(Nat.add(begin,Nat.sub(cell,begin)),Nat.add(begin,U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)))) == True{} : Bool}})
  %same : Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),_,0)
  %chosen : Targets.ClassesIn(_,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Targets.cell_coordinates(Grid.placed(True{},position,others,groups),Nat.add(begin,Nat.sub(cell,begin))),0)
  Targets.target_classes(limits,Grid.placed(True{},position,others,groups),Grid.piece_target(),target_room,target_nonempty,target_rank,readings,component,separate,ok,piece,member,Nat.sub(cell,begin),
    index_in)

# A target cell whose reach passes a whole part group reads the part cell
# before the group its reach ends in.
def before_held(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+other_inside: {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool},+classes_in: Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,Targets.cell_coordinates(others,U32.to_nat(other)),U32.to_nat(group)),0),+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},+end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool},
  +hit: {U32.is_le(8,(8 * group + 8 - start : U32)) == True{} : Bool}) -> Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))):
  %Equal.sym(Bool,U32.is_le(8,(8 * group + 8 - start : U32)),True{},hit) : Logic.When(_,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(U32,(Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 : U32),Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count),Arithmetic.add_zero(Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count))) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat(_))
  part_cell(position,limits,others,groups,bound,count,classes,start,False{},group,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,other_inside,classes_in,from,end_room,Held.Both{{==},Held.Both{hit,top}})

# A target cell whose reach ends inside a part group below the bound reads
# the part cell of that group.
def inside_held(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+other_inside: {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool},+classes_in: Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,Targets.cell_coordinates(others,U32.to_nat(other)),U32.to_nat(group)),0),+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},+end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+shifted: {U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool},+below: {U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound) == True{} : Bool}) ->
  Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))):
  %Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),True{},shifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(Bool,U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),True{},below) : Logic.When(True{} && _,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(U32,(Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 : U32),Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count),Arithmetic.add_zero(Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count))) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat(_))
  part_cell(position,limits,others,groups,bound,count,classes,start,True{},group,other,((8 * group + 8 - start : U32) / 8 : U32),rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,other_inside,classes_in,from,end_room,Held.Both{{==},Held.Both{shifted,below}})

# A target cell whose reach ends inside a part group of the part reads the
# part cell of that group.
def at_held(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+other_inside: {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool},+classes_in: Targets.ClassesIn(classes,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Coordinates.cell_digits(position,Targets.cell_coordinates(others,U32.to_nat(other)),U32.to_nat(group)),0),+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},+end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+shifted: {U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool}) -> Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))):
  Held.by_cases(Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))),U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),
    below => inside_held(position,limits,others,groups,bound,count,classes,start,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,other_inside,classes_in,from,end_room,shifted,below),
    beyond => {%Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),True{},shifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
      %Equal.sym(Bool,U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),False{},beyond) : Logic.When(True{} && _,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
      Unit{} : Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))})

# A target cell of the target piece reads the part cells of its group.
def cell_reads(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate: {Tasks.separable(readings) == True{} : Bool},+ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits)),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},
  +cell_room: {Nat.is_le(Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Tiled.cell_of(position,other,group,groups,count))) == True{} : Bool},+high: {Nat.is_lt(U32.to_nat(Tiled.cell_of(position,other,group,groups,count)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)))) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}) ->
  Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other):
  +classes_in = cell_classes(position,limits,others,groups,count,classes,group,other,count_exact,target_room,target_nonempty,target_rank,readings,component,separate,ok,chosen,piece,member,group_inside,other_inside,cell_room,low,high)
  +inside_others = {%count_exact : {Nat.is_lt(U32.to_nat(other),_) == True{} : Bool} other_inside : {Nat.is_lt(U32.to_nat(other),ShapeLaws.elements(others)) == True{} : Bool}}
  Held.Both{
    Held.by_cases(Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))),U32.is_le(8,(8 * group + 8 - start : U32)),
      hit => before_held(position,limits,others,groups,bound,count,classes,start,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,inside_others,classes_in,from,end_room,top,hit),
      miss => {%Equal.sym(Bool,U32.is_le(8,(8 * group + 8 - start : U32)),False{},miss) : Logic.When(_,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))) Unit{} : Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32))))}),
    Held.by_cases(Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))),U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),
      shifted => at_held(position,limits,others,groups,bound,count,classes,start,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,inside_others,classes_in,from,end_room,shifted),
      unshifted => {%Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),False{},unshifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
        Unit{} : Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.boxed(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Grid.coordinates(Grid.placed(True{},position,others,bound),0),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))})}


# ---- Every target cell of the target piece reads part cells it holds ----

# The other count of a cell grid fits its cap.
def count_cap(+groups: U32,+count: U32,+grid_cap: U32,+first: U32,+span: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,first,span>) ->
  {Nat.is_le(U32.to_nat(count),U32.to_nat(grid_cap)) == True{} : Bool}:
  Gather.PieceIn{Gather.CellGrid{+cap_ok,+groups_positive,+others_positive,+room,+wide,+wide_ok,+doubled},+inside} = piece_in
  Order.transitive(U32.to_nat(count),Nat.mul(U32.to_nat(groups),U32.to_nat(count)),U32.to_nat(grid_cap),
    {%RowMajor.multiply_commutative(U32.to_nat(count),U32.to_nat(groups)) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
      RowMajor.product_covers(U32.to_nat(count),U32.to_nat(groups),groups_positive) : {Nat.is_le(U32.to_nat(count),Nat.mul(U32.to_nat(groups),U32.to_nat(count))) == True{} : Bool}},
    room)

# The cell of an other index and a group below theirs lies inside the
# capacity.
def cell_room_of(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+count: U32,+group: U32,+other: U32,
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},
  +target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_le(Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +index = Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group))
  +below = Representation.index_inside(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group),other_inside,group_inside)
  +total = Equal.trans(Nat,Nat.mul(U32.to_nat(groups),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)),ShapeLaws.elements(Grid.placed(True{},position,others,groups)),
    Equal.cong(Nat,Nat,number => Nat.mul(U32.to_nat(groups),number),U32.to_nat(count),ShapeLaws.elements(others),count_exact),
    Equal.sym(Nat,ShapeLaws.elements(Grid.placed(True{},position,others,groups)),Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)),Public.placed_cells(position,others,groups)))
  Order.transitive(index,ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits)),
    Bounds.strict_le(index,ShapeLaws.elements(Grid.placed(True{},position,others,groups)),{%total : {Nat.is_lt(index,_) == True{} : Bool}
      below : {Nat.is_lt(index,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool}}),target_room)

# A cell inside the target piece starts at or after the piece.
def inside_low(+first: U32,+span: U32,+index: U32,+inside: {U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(U32.to_nat(first),U32.to_nat(index)),U32.is_le(first,index),True{},
    Equal.sym(Bool,U32.is_le(first,index),Nat.is_le(U32.to_nat(first),U32.to_nat(index)),Comparison.le(first,index)),
    Logic.left(U32.is_le(first,index),U32.is_lt(index,(first + span : U32)),inside))

# A cell inside the target piece lies below its end.
def inside_high(+first: U32,+span: U32,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}:
  +below_word = Equal.trans(Bool,Nat.is_lt(U32.to_nat(index),U32.to_nat((first + span : U32))),U32.is_lt(index,(first + span : U32)),True{},
    Equal.sym(Bool,U32.is_lt(index,(first + span : U32)),Nat.is_lt(U32.to_nat(index),U32.to_nat((first + span : U32))),Comparison.lt(index,(first + span : U32))),
    Logic.right(U32.is_le(first,index),U32.is_lt(index,(first + span : U32)),inside))
  %Arithmetic.add(first,span,cap,cap_ok,room) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
  below_word

# A part cell of an other index and a part group below theirs lies inside the
# part, all of which an inseparable component takes.
def whole_cell(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+bound: U32,+count: U32,+other: U32,+read: U32,
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},
  +part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},
  +part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},+read_inside: {Nat.is_lt(U32.to_nat(read),U32.to_nat(bound)) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat(Tiled.cell_of(position,other,read,bound,count))):
  +cell = Tiled.cell_of(position,other,read,bound,count)
  +index = Representation.index_at(position,U32.to_nat(count),U32.to_nat(bound),U32.to_nat(other),U32.to_nat(read))
  +total = Equal.trans(Nat,Nat.mul(U32.to_nat(bound),U32.to_nat(count)),Nat.mul(U32.to_nat(bound),ShapeLaws.elements(others)),ShapeLaws.elements(Grid.placed(True{},position,others,bound)),
    Equal.cong(Nat,Nat,number => Nat.mul(U32.to_nat(bound),number),U32.to_nat(count),ShapeLaws.elements(others),count_exact),
    Equal.sym(Nat,ShapeLaws.elements(Grid.placed(True{},position,others,bound)),Nat.mul(U32.to_nat(bound),ShapeLaws.elements(others)),Public.placed_cells(position,others,bound)))
  +index_below = {%total : {Nat.is_lt(index,_) == True{} : Bool}
    Representation.index_inside(position,U32.to_nat(count),U32.to_nat(bound),U32.to_nat(other),U32.to_nat(read),other_inside,read_inside) :
    {Nat.is_lt(index,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool}}
  +word = Gather.word_by_position(position,bound,count,Machine.capacity(limits),Limits.capacity_fits(limits),other,read,
    Order.transitive(index,ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(index,ShapeLaws.elements(Grid.placed(True{},position,others,bound)),index_below),part_room))
  +below = {%Equal.sym(Nat,U32.to_nat(cell),index,word) : {Nat.is_lt(_,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool} index_below :
    {Nat.is_lt(U32.to_nat(cell),ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool}}
  BoxCells.cell_held(limits,Grid.placed(True{},position,others,bound),Grid.piece_target(),part_room,part_nonempty,part_rank,Tasks.whole_spans(Grid.placed(True{},position,others,bound)),U32.to_nat(cell),Components.whole_in(Grid.placed(True{},position,others,bound)),
    ViewReads.whole_in(Grid.placed(True{},position,others,bound),U32.to_nat(cell),below),below)

# A target group whose reach passes a whole part group reads, before the
# group its reach ends in, a part group below the part groups.
def before_below(+start: U32,+group: U32,+bound: U32,+limits: Machine.Limits,
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +hit: {U32.is_le(8,(8 * group + 8 - start : U32)) == True{} : Bool},+top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32)),U32.to_nat(bound)) == True{} : Bool}:
  +g = U32.to_nat(group)
  +reach = Nat.sub(Nat.add(Nat.mul(8n,g),8n),U32.to_nat(start))
  +hw_ok = Limits.half_word_fits(limits)
  +reach_word = reach_exact(start,group,Machine.half_word(limits),hw_ok,from,end_room)
  +quotient = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,(8 * group + 8 - start : U32),8,reach,8n,reach_word,{==},{Equal.trans(Bool,Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),U32.is_le(16,Machine.half_word(limits)),True{},Equal.sym(Bool,U32.is_le(16,Machine.half_word(limits)),Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),Comparison.le(16,Machine.half_word(limits))),Limits.sixteen_within_half(limits)) : {Nat.is_le(Nat.double(8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},{==})
  +above = U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32))
  +eight = {%reach_word : {Nat.is_le(8n,_) == True{} : Bool}
    Equal.trans(Bool,Nat.is_le(8n,U32.to_nat((8 * group + 8 - start : U32))),U32.is_le(8,(8 * group + 8 - start : U32)),True{},Equal.sym(Bool,U32.is_le(8,(8 * group + 8 - start : U32)),Nat.is_le(8n,U32.to_nat((8 * group + 8 - start : U32))),Comparison.le(8,(8 * group + 8 - start : U32))),hit) :
    {Nat.is_le(8n,reach) == True{} : Bool}}
  +one = {%Equal.sym(Nat,above,Nat.div(reach,8n),quotient) : {Nat.is_le(1n,_) == True{} : Bool} multiple_quotient(1n,reach,8n,{==},eight) :
    {Nat.is_le(1n,above) == True{} : Bool}}
  +above_room = {%Equal.sym(Nat,above,Nat.div(reach,8n),quotient) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Order.transitive(Nat.div(reach,8n),reach,U32.to_nat(Machine.half_word(limits)),
      Order.transitive(Nat.div(reach,8n),Nat.mul(Nat.div(reach,8n),8n),reach,RowMajor.product_covers(Nat.div(reach,8n),8n,{==}),DivisionOrder.product_below(reach,8n,{==})),
      Order.transitive(reach,Nat.add(Nat.mul(8n,g),8n),U32.to_nat(Machine.half_word(limits)),Order.sub_le(Nat.add(Nat.mul(8n,g),8n),U32.to_nat(start)),end_room)) :
    {Nat.is_le(above,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +back = Arithmetic.difference_exact(Machine.half_word(limits),hw_ok,((8 * group + 8 - start : U32) / 8 : U32),1,above,1n,{==},{==},one,above_room)
  %Equal.sym(Nat,U32.to_nat((((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32)),Nat.sub(above,1n),back) : {Nat.is_lt(_,U32.to_nat(bound)) == True{} : Bool}
  Order.lt_le(Nat.sub(above,1n),above,U32.to_nat(bound),
    {%Order.restore_shift(1n,above,one) : {Nat.is_lt(Nat.sub(above,1n),_) == True{} : Bool}
      Order.lt_succ(Nat.sub(above,1n),Nat.sub(above,1n),Order.reflexive(Nat.sub(above,1n))) : {Nat.is_lt(Nat.sub(above,1n),above) == True{} : Bool}},top)

# A target cell of an inseparable component whose reach passes a whole part
# group reads the part cell before the group its reach ends in.
def whole_before(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+bound: U32,+count: U32,+start: U32,+group: U32,+other: U32,+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool},+hit: {U32.is_le(8,(8 * group + 8 - start : U32)) == True{} : Bool}) -> Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))):
  +read_inside = before_below(start,group,bound,limits,from,end_room,hit,top)
  %Equal.sym(Bool,U32.is_le(8,(8 * group + 8 - start : U32)),True{},hit) : Logic.When(_,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(U32,(Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 : U32),Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count),Arithmetic.add_zero(Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count))) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat(_))
  whole_cell(position,limits,others,bound,count,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),count_exact,part_room,part_nonempty,part_rank,other_inside,read_inside)

# A target cell of an inseparable component whose reach ends inside a part
# group below the bound reads the part cell of that group.
def whole_inside(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+bound: U32,+count: U32,+start: U32,+group: U32,+other: U32,+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},+shifted: {U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool},+below: {U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound) == True{} : Bool}) -> Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))):
  +read_inside = Equal.trans(Bool,Nat.is_lt(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)),U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),True{},
    Equal.sym(Bool,U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Nat.is_lt(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)),Comparison.lt(((8 * group + 8 - start : U32) / 8 : U32),bound)),below)
  %Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),True{},shifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(Bool,U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),True{},below) : Logic.When(True{} && _,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
  %Equal.sym(U32,(Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 : U32),Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count),Arithmetic.add_zero(Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count))) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat(_))
  whole_cell(position,limits,others,bound,count,other,((8 * group + 8 - start : U32) / 8 : U32),count_exact,part_room,part_nonempty,part_rank,other_inside,read_inside)

# A target cell of an inseparable component whose reach ends inside a part
# group of the part reads the part cell of that group.
def whole_at(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+bound: U32,+count: U32,+start: U32,+group: U32,+other: U32,+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},+shifted: {U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) == True{} : Bool}) -> Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))):
  Held.by_cases(Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))),U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),
    below => whole_inside(position,limits,others,bound,count,start,group,other,count_exact,part_room,part_nonempty,part_rank,other_inside,shifted,below),
    beyond => {%Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),True{},shifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
      %Equal.sym(Bool,U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),False{},beyond) : Logic.When(True{} && _,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
      Unit{} : Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))})

# A target cell of an inseparable component reads part cells the component
# takes.
def whole_reads(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+bound: U32,+count: U32,+start: U32,+group: U32,+other: U32,+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},+from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}) -> Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other):
  Held.Both{
    Held.by_cases(Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))),U32.is_le(8,(8 * group + 8 - start : U32)),
      hit => whole_before(position,limits,others,bound,count,start,group,other,count_exact,part_room,part_nonempty,part_rank,other_inside,from,end_room,top,hit),
      miss => {%Equal.sym(Bool,U32.is_le(8,(8 * group + 8 - start : U32)),False{},miss) : Logic.When(_,Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32)))) Unit{} : Logic.When(U32.is_le(8,(8 * group + 8 - start : U32)),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,(((8 * group + 8 - start : U32) / 8 : U32) - 1 : U32),bound,count) + 0 * 0 : U32))))}),
    Held.by_cases(Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32)))),U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),
      shifted => whole_at(position,limits,others,bound,count,start,group,other,count_exact,part_room,part_nonempty,part_rank,other_inside,shifted),
      unshifted => {%Equal.sym(Bool,U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)),False{},unshifted) : Logic.When(_ && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))
        Unit{} : Logic.When(U32.is_lt(0,((8 * group + 8 - start : U32) % 8 : U32)) && U32.is_lt(((8 * group + 8 - start : U32) / 8 : U32),bound),Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.whole_spans(Grid.placed(True{},position,others,bound))),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),U32.to_nat((Tiled.cell_of(position,other,((8 * group + 8 - start : U32) / 8 : U32),bound,count) + 0 * 0 : U32))))})}

# A target cell of a component reads part cells the component takes, the
# component separable or not.
def held_cell(separate: Bool,+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,
  +start: U32,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},
  +cell_room: {Nat.is_le(Representation.index_at(position,U32.to_nat(count),U32.to_nat(groups),U32.to_nat(other),U32.to_nat(group)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Tiled.cell_of(position,other,group,groups,count))) == True{} : Bool},+high: {Nat.is_lt(U32.to_nat(Tiled.cell_of(position,other,group,groups,count)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)))) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}) ->
  Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other):
  match separate:
    case True{}:
      cell_reads(position,limits,others,groups,bound,count,classes,start,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,
        chosen,piece,member,group_inside,other_inside,cell_room,low,high,from,end_room,top)
    case False{}:
      whole_reads(position,limits,others,bound,count,start,group,other,count_exact,part_room,part_nonempty,part_rank,other_inside,from,end_room,top)

# A target cell inside the target piece reads part cells its component
# holds.
def piece_cell(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},+other_inside: {Nat.is_lt(U32.to_nat(other),U32.to_nat(count)) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool},+inside: {U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)) == True{} : Bool}) -> Logic.When(U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other)):
  +cap_ok = Gather.cap_fits(groups,count,grid_cap,Gather.piece_grid(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in))
  +room = Gather.piece_room(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in)
  %Equal.sym(Bool,U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),True{},inside) : Logic.When(_,Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other))
  held_cell(separate,position,limits,others,groups,bound,count,classes,start,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,group_inside,other_inside,
    cell_room_of(position,limits,others,groups,count,group,other,count_exact,target_room,group_inside,other_inside),
    inside_low(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count),inside),inside_high(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count),grid_cap,cap_ok,room,inside),from,end_room,top)

# The target cells of a group from an other index on read part cells their
# component holds.
def others_held(fuel: Nat,+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+group: U32,+other: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool},
  +other_end: {Nat.is_le(Nat.add(U32.to_nat(other),fuel),U32.to_nat(count)) == True{} : Bool}) ->
  Gather.OthersHeld(fuel,Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),group,other):
  match fuel:
    case 0n:
      Unit{}
    case 1n+ +rest:
      +other_inside = Order.room_index(U32.to_nat(other),rest,U32.to_nat(count),other_end)
      +next = Gather.word_next(other,grid_cap,Gather.cap_fits(groups,count,grid_cap,Gather.piece_grid(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in)),
        Bounds.nat_trans(U32.to_nat(other),U32.to_nat(count),U32.to_nat(grid_cap),other_inside,count_cap(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in)))
      Held.Both{
        Held.by_cases(Logic.When(U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other)),U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),
          inside => piece_cell(position,limits,others,groups,bound,count,classes,start,separate,group,other,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,group_inside,other_inside,from,end_room,top,inside),
          outside => {%Equal.sym(Bool,U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),False{},outside) : Logic.When(_,Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other)) Unit{} : Logic.When(U32.is_le(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Tiled.cell_of(position,other,group,groups,count)) && U32.is_lt(Tiled.cell_of(position,other,group,groups,count),(Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) + Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece) : U32)),Gather.ReadsHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,bound,count,position,group,other))}),
        others_held(rest,position,limits,others,groups,bound,count,classes,start,separate,group,(other + 1 : U32),rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,group_inside,from,end_room,top,Gather.range_rest(other,rest,U32.to_nat(count),next,other_end))}

# The target cells of a group in the range of other indices of the target
# piece read part cells their component holds.
def range_held(+found: Tiled.Others,+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+group: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,
  +group_inside: {Nat.is_lt(U32.to_nat(group),U32.to_nat(groups)) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +top: {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Gather.range_low(found)),Gather.range_fuel(found)),U32.to_nat(count)) == True{} : Bool}) ->
  Gather.RangeHeld(found,Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),group):
  Tiled.Others{+low,+high} = found
  others_held(U32.to_nat((U32.max(high,low) - low : U32)),position,limits,others,groups,bound,count,classes,start,separate,group,low,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,group_inside,from,end_room,top,room)

# A group below a last group ends its lanes within eight times the last.
def group_after(+group: Nat,+last: Nat,+limit: Nat,+below: {Nat.is_lt(group,last) == True{} : Bool},
  +last_room: {Nat.is_le(Nat.mul(8n,last),limit) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(8n,group),8n),limit) == True{} : Bool}:
  +expanded = Equal.trans(Nat,Nat.mul(8n,1n+group),Nat.add(8n,Nat.mul(8n,group)),Nat.add(Nat.mul(8n,group),8n),RowMajor.multiply_successor_right(8n,group),
    Order.commutative(8n,Nat.mul(8n,group)))
  Order.transitive(Nat.add(Nat.mul(8n,group),8n),Nat.mul(8n,last),limit,
    {%expanded : {Nat.is_le(_,Nat.mul(8n,last)) == True{} : Bool}
      DivisionOrder.multiply_left_le(8n,1n+group,last,Order.strict_follow(group,last,below)) : {Nat.is_le(Nat.add(Nat.mul(8n,group),8n),Nat.mul(8n,last)) == True{} : Bool}},
    last_room)

# A group whose lanes end inside the word lies below the word.
def group_word(+group: Nat,+cap: Nat,+end_room: {Nat.is_le(Nat.add(Nat.mul(8n,group),8n),cap) == True{} : Bool}) -> {Nat.is_lt(group,cap) == True{} : Bool}:
  Order.room_index(group,0n,cap,Order.transitive(Nat.add(group,1n),Nat.add(Nat.mul(8n,group),1n),cap,Order.add_le(group,Nat.mul(8n,group),1n,Components.factor_covers(group,8n,{==})),
    Order.transitive(Nat.add(Nat.mul(8n,group),1n),Nat.add(Nat.mul(8n,group),8n),cap,Order.offset_le(Nat.mul(8n,group),1n,8n,{==}),end_room)))

# A target group whose lanes end within the part reads part groups below the
# part groups.
def group_top(+start: U32,+group: U32,+width: U32,+bound: U32,+limits: Machine.Limits,
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},
  +lanes: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),U32.to_nat(bound)) == True{} : Bool}:
  +st = U32.to_nat(start)
  +w = U32.to_nat(width)
  +reach = Nat.sub(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),st)
  +hw_ok = Limits.half_word_fits(limits)
  +divisor_room = {Equal.trans(Bool,Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),U32.is_le(16,Machine.half_word(limits)),True{},Equal.sym(Bool,U32.is_le(16,Machine.half_word(limits)),Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),Comparison.le(16,Machine.half_word(limits))),
    Limits.sixteen_within_half(limits)) : {Nat.is_le(Nat.double(8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +quotient = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,(8 * group + 8 - start : U32),8,reach,8n,reach_exact(start,group,Machine.half_word(limits),hw_ok,from,end_room),{==},divisor_room,{==})
  +moved = Equal.trans(Nat,Nat.sub(Nat.add(Nat.add(st,w),7n),st),Nat.sub(Nat.add(Nat.add(w,7n),st),st),Nat.add(w,7n),
    Equal.cong(Nat,Nat,value => Nat.sub(value,st),Nat.add(Nat.add(st,w),7n),Nat.add(Nat.add(w,7n),st),
      Equal.trans(Nat,Nat.add(Nat.add(st,w),7n),Nat.add(st,Nat.add(w,7n)),Nat.add(Nat.add(w,7n),st),Addition.associative(st,w,7n),Order.commutative(st,Nat.add(w,7n)))),
    DivisionOrder.subtract_sum(Nat.add(w,7n),st))
  +ordered = {%moved : {Nat.is_le(reach,_) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),Nat.add(Nat.add(st,w),7n),st,lanes) :
    {Nat.is_le(reach,Nat.add(w,7n)) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(((8 * group + 8 - start : U32) / 8 : U32)),Nat.div(reach,8n),quotient) : {Nat.is_le(_,U32.to_nat(bound)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(bound),Nat.div(Nat.add(w,7n),8n),bound_exact) : {Nat.is_le(Nat.div(reach,8n),_) == True{} : Bool}
  DivisionOrder.quotient_le(reach,Nat.add(w,7n),8n,{==},ordered)

# The target groups from a group on below a last group read part cells their
# component holds, for every target cell of the target piece.
def groups_held(fuel: Nat,+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+group: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,+span_positive: {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece))) == True{} : Bool},+width: U32,
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},
  +groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+last: Nat,
  +last_groups: {Nat.is_le(last,U32.to_nat(groups)) == True{} : Bool},
  +last_lanes: {Nat.is_le(Nat.mul(8n,last),Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n)) == True{} : Bool},
  +from: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +group_end: {Nat.is_le(Nat.add(U32.to_nat(group),fuel),last) == True{} : Bool}) ->
  Gather.GroupsHeld(fuel,Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),group):
  match fuel:
    case 0n:
      Unit{}
    case 1n+ +rest:
      +g = U32.to_nat(group)
      +below = Order.room_index(g,rest,last,group_end)
      +group_inside = Order.lt_le(g,last,U32.to_nat(groups),below,last_groups)
      +end_room = group_after(g,U32.to_nat(groups),U32.to_nat(Machine.half_word(limits)),group_inside,groups_room)
      +top = group_top(start,group,width,bound,limits,from,end_room,bound_exact,group_after(g,last,Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),below,last_lanes))
      +next = Gather.word_next(group,Machine.half_word(limits),Limits.half_word_fits(limits),group_word(g,U32.to_nat(Machine.half_word(limits)),end_room))
      +found = Tiled.others_of(position,groups,count,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),group)
      +cap_ok = Gather.cap_fits(groups,count,grid_cap,Gather.piece_grid(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in))
      +room = Gather.range_room(found,count,grid_cap,cap_ok,count_cap(groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),piece_in),
        Gather.others_fit(position,groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),group,piece_in,group_inside,span_positive))
      Held.Both{range_held(found,position,limits,others,groups,bound,count,classes,start,separate,group,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,group_inside,from,end_room,top,room),
        groups_held(rest,position,limits,others,groups,bound,count,classes,start,separate,(group + 1 : U32),rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,span_positive,width,bound_exact,groups_room,last,last_groups,last_lanes,
          {%Equal.sym(Nat,U32.to_nat((group + 1 : U32)),1n+g,next) : {Nat.is_le(Nat.div(U32.to_nat(start),8n),_) == True{} : Bool}
            Order.successor(Nat.div(U32.to_nat(start),8n),g,from) : {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat((group + 1 : U32))) == True{} : Bool}},
          Gather.range_rest(group,rest,last,next,group_end))}

# ---- The groups a separable lane write reaches ----

# A part of some width ends at least one lane past zero.
def lanes_positive(+start: U32,+width: U32,+some: {U32.is_zero(width) == False{} : Bool}) -> {Nat.is_le(1n,Nat.add(U32.to_nat(start),U32.to_nat(width))) == True{} : Bool}:
  +w = U32.to_nat(width)
  Order.transitive(1n,w,Nat.add(U32.to_nat(start),U32.to_nat(width)),Order.strict_follow(0n,w,Arithmetic.word_nonzero_positive(width,some)),
    {%Order.commutative(w,U32.to_nat(start)) : {Nat.is_le(w,_) == True{} : Bool} Order.le_add(w,U32.to_nat(start)) : {Nat.is_le(w,Nat.add(U32.to_nat(start),U32.to_nat(width))) == True{} : Bool}})

# The last lane of a part of some width lies at or after its start.
def start_last(+start: U32,+width: U32,+some: {U32.is_zero(width) == False{} : Bool}) -> {Nat.is_le(U32.to_nat(start),Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n)) == True{} : Bool}:
  +st = U32.to_nat(start)
  %DivisionOrder.subtract_sum(st,1n) : {Nat.is_le(_,Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n)) == True{} : Bool}
  DivisionOrder.subtract_le(Nat.add(st,1n),Nat.add(U32.to_nat(start),U32.to_nat(width)),1n,Order.offset_le(st,1n,U32.to_nat(width),Order.strict_follow(0n,U32.to_nat(width),Arithmetic.word_nonzero_positive(width,some))))

# The group after the last lane of a part is the group the lanes of the part
# end in, rounded up.
def last_exact(+start: U32,+width: U32,+some: {U32.is_zero(width) == False{} : Bool}) -> {Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n) == 1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n) : Nat}:
  +y = Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n)
  +restored = Order.restore_shift(1n,Nat.add(U32.to_nat(start),U32.to_nat(width)),lanes_positive(start,width,some))
  +shifted = Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),Nat.add(Nat.add(1n,y),7n),Nat.add(8n,y),
    Equal.cong(Nat,Nat,value => Nat.add(value,7n),Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.add(1n,y),Equal.sym(Nat,Nat.add(1n,y),Nat.add(U32.to_nat(start),U32.to_nat(width)),restored)),
    Equal.cong(Nat,Nat,value => 1n+value,Nat.add(y,7n),Nat.add(7n,y),Order.commutative(y,7n)))
  Equal.trans(Nat,Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),Nat.div(Nat.add(8n,y),8n),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Equal.cong(Nat,Nat,value => Nat.div(value,8n),Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),Nat.add(8n,y),shifted),
    DivisionOrder.quotient_shift(y,8n,1n,{==}))

# The groups a part reaches end within eight lanes past its end.
def last_lanes(+start: U32,+width: U32) -> {Nat.is_le(Nat.mul(8n,Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n)),Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n)) == True{} : Bool}:
  %RowMajor.multiply_commutative(Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),8n) : {Nat.is_le(_,Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n)) == True{} : Bool}
  DivisionOrder.product_below(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n,{==})

# The groups a part reaches lie inside the target groups.
def last_groups(+start: U32,+width: U32,+groups: U32,+lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool}) -> {Nat.is_le(Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),U32.to_nat(groups)) == True{} : Bool}:
  +g = U32.to_nat(groups)
  +lanes = Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n)
  +swapped = Equal.trans(Nat,Nat.add(Nat.mul(8n,g),7n),Nat.add(7n,Nat.mul(8n,g)),Nat.add(7n,Nat.mul(g,8n)),Order.commutative(Nat.mul(8n,g),7n),
    Equal.cong(Nat,Nat,value => Nat.add(7n,value),Nat.mul(8n,g),Nat.mul(g,8n),RowMajor.multiply_commutative(8n,g)))
  +within = {%swapped : {Nat.is_le(lanes,_) == True{} : Bool} Order.add_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,g),7n,lanes_room) : {Nat.is_le(lanes,Nat.add(7n,Nat.mul(g,8n))) == True{} : Bool}}
  Division.lt_succ_le(Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),g,DivisionOrder.quotient_below(lanes,1n+g,8n,{==},
    {Order.lt_succ(lanes,Nat.add(7n,Nat.mul(g,8n)),within) : {Nat.is_lt(lanes,Nat.mul(1n+g,8n)) == True{} : Bool}}))

# The number of groups a part of some width reaches.
def reached_exact(+start: U32,+width: U32,+groups: U32,+limits: Machine.Limits,+some: {U32.is_zero(width) == False{} : Bool},+lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat(((start + width - 1) / 8 - start / 8 + 1 : U32)) == Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n) : Nat}:
  +st = U32.to_nat(start)
  +y = Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n)
  +hw_ok = Limits.half_word_fits(limits)
  +sum_room = Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits)),lanes_room,groups_room)
  +one = lanes_positive(start,width,some)
  +sum = Arithmetic.sum_exact(Machine.half_word(limits),hw_ok,start,width,st,U32.to_nat(width),{==},{==},sum_room)
  +last = Arithmetic.difference_exact(Machine.half_word(limits),hw_ok,(start + width : U32),1,Nat.add(U32.to_nat(start),U32.to_nat(width)),1n,sum,{==},one,sum_room)
  +divisor_room = {Equal.trans(Bool,Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),U32.is_le(16,Machine.half_word(limits)),True{},Equal.sym(Bool,U32.is_le(16,Machine.half_word(limits)),Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),Comparison.le(16,Machine.half_word(limits))),Limits.sixteen_within_half(limits)) : {Nat.is_le(Nat.double(8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +last_group = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,(start + width - 1 : U32),8,y,8n,last,{==},divisor_room,{==})
  +first_group = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,start,8,st,8n,{==},{==},divisor_room,{==})
  +last_le = Order.transitive(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.mul(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),8n),y,RowMajor.product_covers(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),8n,{==}),DivisionOrder.product_below(y,8n,{==}))
  +ordered = DivisionOrder.quotient_le(st,y,8n,{==},start_last(start,width,some))
  +y_room = Order.transitive(y,Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.half_word(limits)),Order.sub_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),sum_room)
  +last_room = Order.transitive(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),y,U32.to_nat(Machine.half_word(limits)),last_le,y_room)
  +apart = Arithmetic.difference_exact(Machine.half_word(limits),hw_ok,((start + width - 1) / 8 : U32),(start / 8 : U32),Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n),last_group,first_group,ordered,last_room)
  +back = Equal.trans(Nat,Nat.add(y,1n),Nat.add(1n,y),Nat.add(U32.to_nat(start),U32.to_nat(width)),Order.commutative(y,1n),Order.restore_shift(1n,Nat.add(U32.to_nat(start),U32.to_nat(width)),one))
  +next_room = Order.transitive(Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n),Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.half_word(limits)),
    {%back : {Nat.is_le(Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n),_) == True{} : Bool} Order.add_le(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),y,1n,last_le) : {Nat.is_le(Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n),Nat.add(U32.to_nat(start),U32.to_nat(width))) == True{} : Bool}},sum_room)
  Arithmetic.sum_exact(Machine.half_word(limits),hw_ok,((start + width - 1) / 8 - start / 8 : U32),1,Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n,apart,{==},
    Order.transitive(Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n),Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n),U32.to_nat(Machine.half_word(limits)),Order.add_le(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n,Order.sub_le(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n))),next_room))

# The groups a part reaches from the group of its start end at the group
# after its last lane.
def reached_end(+start: U32,+width: U32,+groups: U32,+limits: Machine.Limits,+some: {U32.is_zero(width) == False{} : Bool},+lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat((start / 8 : U32)),U32.to_nat(((start + width - 1) / 8 - start / 8 + 1 : U32))),Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n)) == True{} : Bool}:
  +hw_ok = Limits.half_word_fits(limits)
  +first_group = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,start,8,U32.to_nat(start),8n,{==},{==},{Equal.trans(Bool,Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),U32.is_le(16,Machine.half_word(limits)),True{},Equal.sym(Bool,U32.is_le(16,Machine.half_word(limits)),Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),Comparison.le(16,Machine.half_word(limits))),Limits.sixteen_within_half(limits)) : {Nat.is_le(Nat.double(8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},{==})
  +ordered = DivisionOrder.quotient_le(U32.to_nat(start),Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n,{==},start_last(start,width,some))
  %Equal.sym(Nat,U32.to_nat((start / 8 : U32)),Nat.div(U32.to_nat(start),8n),first_group) : {Nat.is_le(Nat.add(_,U32.to_nat(((start + width - 1) / 8 - start / 8 + 1 : U32))),Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(((start + width - 1) / 8 - start / 8 + 1 : U32)),Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n),reached_exact(start,width,groups,limits,some,lanes_room,groups_room)) :
    {Nat.is_le(Nat.add(Nat.div(U32.to_nat(start),8n),_),Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n)) == True{} : Bool}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),last_exact(start,width,some)) : {Nat.is_le(Nat.add(Nat.div(U32.to_nat(start),8n),Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n)),_) == True{} : Bool}
  %Addition.associative(Nat.div(U32.to_nat(start),8n),Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n)),1n) : {Nat.is_le(_,1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.div(U32.to_nat(start),8n),Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Nat.div(U32.to_nat(start),8n))),Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),Order.restore_shift(Nat.div(U32.to_nat(start),8n),Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),ordered)) : {Nat.is_le(Nat.add(_,1n),1n+Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n)) == True{} : Bool}
  %Order.commutative(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n) : {Nat.is_le(Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n),_) == True{} : Bool}
  Order.reflexive(Nat.add(Nat.div(Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(width)),1n),8n),1n))

# The groups a part of some width reaches read, for every target cell of the
# target piece, part cells their component holds.
def written_held(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+width: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,+span_positive: {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece))) == True{} : Bool},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool},+some: {U32.is_zero(width) == False{} : Bool}) ->
  Gather.GroupsHeld(U32.to_nat(Gather.reached_groups(start,width)),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32)):
  +hw_ok = Limits.half_word_fits(limits)
  +first_group = U32Division.quotient_exact(Machine.half_word(limits),hw_ok,start,8,U32.to_nat(start),8n,{==},{==},{Equal.trans(Bool,Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),U32.is_le(16,Machine.half_word(limits)),True{},Equal.sym(Bool,U32.is_le(16,Machine.half_word(limits)),Nat.is_le(16n,U32.to_nat(Machine.half_word(limits))),Comparison.le(16,Machine.half_word(limits))),Limits.sixteen_within_half(limits)) : {Nat.is_le(Nat.double(8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},{==})
  %Equal.sym(Bool,U32.is_zero(width),False{},some) :
    Gather.GroupsHeld(U32.to_nat(Bool.pick(U32,_,0,((start + width - 1) / 8 - start / 8 + 1 : U32))),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32))
  groups_held(U32.to_nat(((start + width - 1) / 8 - start / 8 + 1 : U32)),position,limits,others,groups,bound,count,classes,start,separate,(start / 8 : U32),rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,span_positive,width,bound_exact,groups_room,Nat.div(Nat.add(Nat.add(U32.to_nat(start),U32.to_nat(width)),7n),8n),last_groups(start,width,groups,lanes_room),
    last_lanes(start,width),
    {%Equal.sym(Nat,U32.to_nat((start / 8 : U32)),Nat.div(U32.to_nat(start),8n),first_group) : {Nat.is_le(Nat.div(U32.to_nat(start),8n),_) == True{} : Bool} Order.reflexive(Nat.div(U32.to_nat(start),8n)) :
      {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat((start / 8 : U32))) == True{} : Bool}},
    reached_end(start,width,groups,limits,some,lanes_room,groups_room))

# The groups a separable lane write reaches read, for every target cell of
# the target piece, part cells their component holds.
def lanes_held(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+classes: List<&2,Tasks.Class>,+start: U32,+separate: Bool,+width: U32,+rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes),+axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rank_room: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate_exact: {Tasks.separable(readings) == separate : Bool},+ok: Logic.When(separate,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),Machine.half_word(limits))),+chosen: {Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component) == classes : List<&2,Tasks.Class>},+piece: U32,+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),component),piece),+grid_cap: U32,+piece_in: Gather.PieceIn<groups,count,grid_cap,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece)>,+span_positive: {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece))) == True{} : Bool},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool}) ->
  Gather.GroupsHeld(U32.to_nat(Gather.reached_groups(start,width)),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32)):
  Held.by_cases(Gather.GroupsHeld(U32.to_nat(Gather.reached_groups(start,width)),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32)),
    U32.is_zero(width),
    none => {%Equal.sym(Bool,U32.is_zero(width),True{},none) :
        Gather.GroupsHeld(U32.to_nat(Bool.pick(U32,_,0,((start + width - 1) / 8 - start / 8 + 1 : U32))),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32))
      Unit{} : Gather.GroupsHeld(U32.to_nat(Gather.reached_groups(start,width)),Moved.taken_spans(Tasks.box_pieces(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),Tasks.box_when(separate,Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),classes,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0)),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target())),start,groups,bound,count,position,Grid.start(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),Grid.span(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),piece),(start / 8 : U32))},
    some => written_held(position,limits,others,groups,bound,count,classes,start,separate,width,rooms,ends,axes,blocks,part_room,part_nonempty,part_rank,rank_room,count_exact,target_room,target_nonempty,target_rank,readings,component,separate_exact,ok,chosen,piece,member,grid_cap,piece_in,span_positive,bound_exact,groups_room,lanes_room,some))
