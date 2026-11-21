# Where a target cell of a split lies: the words a split computes for a
# target cell index give its anchor, its group, the source cell of group 0
# with the same elements before and after the lanes, and the step between
# source groups. The index is written in its digits: the row of the elements
# before the lanes, the anchor, the index after the lanes and the group.
import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_representation.bend as Representation
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./bounded_u32_arithmetic.bend as Arithmetic

# ---- Division by a word of at least one ----

# Whole steps and a remainder below the step, divided by the step.
def divided(+value: Nat,+divisor: Nat,+quotient: Nat,+remainder: Nat,+exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},
  +small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) -> {Nat.div(value,divisor) == quotient : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,divisor),(quotient,remainder),
    Division.unique(value,divisor,quotient,remainder,exact,Representation.below_positive(remainder,divisor,small),small))

def remaindered(+value: Nat,+divisor: Nat,+quotient: Nat,+remainder: Nat,+exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},
  +small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) -> {Nat.mod(value,divisor) == remainder : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,divisor),(quotient,remainder),
    Division.unique(value,divisor,quotient,remainder,exact,Representation.below_positive(remainder,divisor,small),small))

# A word of whole steps of a divisor word and a remainder below it, divided
# by the divisor or one: the steps, and the remainder.
def word_quotient(+limits: Machine.Limits,+value: U32,+divisor: U32,+quotient: Nat,+remainder: Nat,
  +exact: {U32.to_nat(value) == Nat.add(Nat.mul(quotient,U32.to_nat(divisor)),remainder) : Nat},+small: {Nat.is_lt(remainder,U32.to_nat(divisor)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(divisor),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat((value / U32.max(divisor,1) : U32)) == quotient : Nat}:
  +steps = U32.to_nat(divisor)
  +positive = Representation.below_positive(remainder,steps,small)
  Equal.trans(Nat,U32.to_nat((value / U32.max(divisor,1) : U32)),Nat.div(U32.to_nat(value),steps),quotient,
    U32Division.quotient_exact(Machine.half_word(limits),Limits.half_word_fits(limits),value,U32.max(divisor,1),U32.to_nat(value),steps,{==},
      Extrema.max_one_exact(divisor,steps,{==},positive),Limits.double_bound(limits,steps,room),positive),
    divided(U32.to_nat(value),steps,quotient,remainder,exact,small))

def word_remainder(+limits: Machine.Limits,+value: U32,+divisor: U32,+quotient: Nat,+remainder: Nat,
  +exact: {U32.to_nat(value) == Nat.add(Nat.mul(quotient,U32.to_nat(divisor)),remainder) : Nat},+small: {Nat.is_lt(remainder,U32.to_nat(divisor)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(divisor),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat((value % U32.max(divisor,1) : U32)) == remainder : Nat}:
  +steps = U32.to_nat(divisor)
  +positive = Representation.below_positive(remainder,steps,small)
  Equal.trans(Nat,U32.to_nat((value % U32.max(divisor,1) : U32)),Nat.mod(U32.to_nat(value),steps),remainder,
    U32Division.remainder_exact(Machine.half_word(limits),Limits.half_word_fits(limits),value,U32.max(divisor,1),U32.to_nat(value),steps,{==},
      Extrema.max_one_exact(divisor,steps,{==},positive),Limits.double_bound(limits,steps,room),positive),
    remaindered(U32.to_nat(value),steps,quotient,remainder,exact,small))

# A sum below a bound has both terms below it.
def left_room(+left: Nat,+right: Nat,+bound: Nat,+room: {Nat.is_le(Nat.add(left,right),bound) == True{} : Bool}) -> {Nat.is_le(left,bound) == True{} : Bool}:
  Order.transitive(left,Nat.add(left,right),bound,Order.le_add(left,right),room)

def factor_room(+left: Nat,+right: Nat,+bound: Nat,+positive: {Nat.is_lt(0n,right) == True{} : Bool},+room: {Nat.is_le(Nat.mul(left,right),bound) == True{} : Bool}) ->
  {Nat.is_le(left,bound) == True{} : Bool}:
  Order.transitive(left,Nat.mul(left,right),bound,RowMajor.product_covers(left,right,positive),room)

# ---- The groups innermost ----

# Target cell index ((above * anchors + anchor) * after + below) * groups +
# group: its anchor and group, and the source cell of group 0 at outer index
# above * after + below, the source groups one apart.
def inner_spot(+limits: Machine.Limits,+outer: U32,+anchors: U32,+after: U32,+groups: U32,+source_groups: U32,+index: U32,
  +above: Nat,+anchor: Nat,+below: Nat,+group: Nat,
  +anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},+below_inside: {Nat.is_lt(below,U32.to_nat(after)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool},
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),below),U32.to_nat(groups)),group) : Nat},
  +anchors_room: {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +after_room: {Nat.is_le(U32.to_nat(after),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,U32.to_nat(source_groups)) == True{} : Bool},
  +base_room: {Nat.is_le(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(after)),below),U32.to_nat(source_groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Tiled.split_spot(Grid.Innermost{},outer,anchors,after,groups,source_groups,index) ==
    Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(after)),below),U32.to_nat(source_groups))),1} : Tiled.Spot}:
  +lanes_after = U32.to_nat(after)
  +half = U32.to_nat(Machine.half_word(limits))
  +row = Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor)
  +source_outer = Nat.add(Nat.mul(above,lanes_after),below)
  +base = Nat.mul(source_outer,U32.to_nat(source_groups))
  +rest = (index / U32.max(groups,1) : U32)
  +inner = (rest / U32.max(after,1) : U32)
  +taken = (inner / U32.max(anchors,1) * after : U32)
  +summed = ((inner / U32.max(anchors,1) * after : U32) + (rest % U32.max(after,1) : U32) : U32)
  +rest_exact = word_quotient(limits,index,groups,Nat.add(Nat.mul(row,lanes_after),below),group,exact,group_inside,groups_room)
  +inner_exact = word_quotient(limits,rest,after,row,below,rest_exact,below_inside,after_room)
  +outer_room = factor_room(source_outer,U32.to_nat(source_groups),half,source_positive,base_room)
  +taken_exact = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),(inner / U32.max(anchors,1) : U32),after,above,lanes_after,
    word_quotient(limits,inner,anchors,above,anchor,inner_exact,anchor_inside,anchors_room),{==},left_room(Nat.mul(above,lanes_after),below,half,outer_room))
  +summed_exact = Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),taken,(rest % U32.max(after,1) : U32),Nat.mul(above,lanes_after),below,
    taken_exact,word_remainder(limits,rest,after,row,below,rest_exact,below_inside,after_room),outer_room)
  +base_exact = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),summed,source_groups,source_outer,U32.to_nat(source_groups),
    summed_exact,{==},base_room)
  %Arithmetic.from_nat_exact((inner % U32.max(anchors,1) : U32),anchor,word_remainder(limits,inner,anchors,above,anchor,inner_exact,anchor_inside,anchors_room)) :
    {Tiled.Spot{_,(index % U32.max(groups,1) : U32),(summed * source_groups : U32),1} == Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),1} : Tiled.Spot}
  %Arithmetic.from_nat_exact((index % U32.max(groups,1) : U32),group,
    word_remainder(limits,index,groups,Nat.add(Nat.mul(row,lanes_after),below),group,exact,group_inside,groups_room)) :
    {Tiled.Spot{U32.from_nat(anchor),_,(summed * source_groups : U32),1} == Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),1} : Tiled.Spot}
  %Arithmetic.from_nat_exact((summed * source_groups : U32),base,base_exact) :
    {Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),_,1} == Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),1} : Tiled.Spot}
  {==}

# ---- The groups outermost ----

# Target cell index (((group * outer + above) * anchors + anchor) * after +
# below: its anchor and group, and the source cell of group 0 at outer index
# above * after + below, the source groups outer * after apart.
def outer_spot(+limits: Machine.Limits,+outer: U32,+anchors: U32,+after: U32,+groups: U32,+source_groups: U32,+index: U32,
  +above: Nat,+anchor: Nat,+below: Nat,+group: Nat,
  +above_inside: {Nat.is_lt(above,U32.to_nat(outer)) == True{} : Bool},+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +below_inside: {Nat.is_lt(below,U32.to_nat(after)) == True{} : Bool},
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(group,U32.to_nat(outer)),above),U32.to_nat(anchors)),anchor),U32.to_nat(after)),below) : Nat},
  +outer_room: {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +anchors_room: {Nat.is_le(U32.to_nat(anchors),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +after_room: {Nat.is_le(U32.to_nat(after),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +base_room: {Nat.is_le(Nat.add(Nat.mul(above,U32.to_nat(after)),below),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +step_room: {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(after)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Tiled.split_spot(Grid.Outermost{},outer,anchors,after,groups,source_groups,index) ==
    Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(Nat.add(Nat.mul(above,U32.to_nat(after)),below)),
      U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(after)))} : Tiled.Spot}:
  +lanes_after = U32.to_nat(after)
  +half = U32.to_nat(Machine.half_word(limits))
  +column = Nat.add(Nat.mul(group,U32.to_nat(outer)),above)
  +row = Nat.add(Nat.mul(column,U32.to_nat(anchors)),anchor)
  +base = Nat.add(Nat.mul(above,lanes_after),below)
  +step = Nat.mul(U32.to_nat(outer),lanes_after)
  +rest = (index / U32.max(after,1) : U32)
  +inner = (rest / U32.max(anchors,1) : U32)
  +taken = (inner % U32.max(outer,1) * after : U32)
  +rest_exact = word_quotient(limits,index,after,row,below,exact,below_inside,after_room)
  +inner_exact = word_quotient(limits,rest,anchors,column,anchor,rest_exact,anchor_inside,anchors_room)
  +taken_exact = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),(inner % U32.max(outer,1) : U32),after,above,lanes_after,
    word_remainder(limits,inner,outer,group,above,inner_exact,above_inside,outer_room),{==},left_room(Nat.mul(above,lanes_after),below,half,base_room))
  +base_exact = Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),taken,(index % U32.max(after,1) : U32),Nat.mul(above,lanes_after),below,
    taken_exact,word_remainder(limits,index,after,row,below,exact,below_inside,after_room),base_room)
  +step_exact = Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),outer,after,U32.to_nat(outer),lanes_after,{==},{==},step_room)
  %Arithmetic.from_nat_exact((rest % U32.max(anchors,1) : U32),anchor,word_remainder(limits,rest,anchors,column,anchor,rest_exact,anchor_inside,anchors_room)) :
    {Tiled.Spot{_,(inner / U32.max(outer,1) : U32),(taken + (index % U32.max(after,1) : U32) : U32),(outer * after : U32)} ==
      Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),U32.from_nat(step)} : Tiled.Spot}
  %Arithmetic.from_nat_exact((inner / U32.max(outer,1) : U32),group,word_quotient(limits,inner,outer,group,above,inner_exact,above_inside,outer_room)) :
    {Tiled.Spot{U32.from_nat(anchor),_,(taken + (index % U32.max(after,1) : U32) : U32),(outer * after : U32)} ==
      Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),U32.from_nat(step)} : Tiled.Spot}
  %Arithmetic.from_nat_exact((taken + (index % U32.max(after,1) : U32) : U32),base,base_exact) :
    {Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),_,(outer * after : U32)} == Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),U32.from_nat(step)} : Tiled.Spot}
  %Arithmetic.from_nat_exact((outer * after : U32),step,step_exact) :
    {Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),_} == Tiled.Spot{U32.from_nat(anchor),U32.from_nat(group),U32.from_nat(base),U32.from_nat(step)} : Tiled.Spot}
  {==}
