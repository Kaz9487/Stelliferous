# Whole eight-lane output groups, including a possibly short final group.
import Base
import ../kernel_gemm.bend as Gemm
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./kernel_gemm_point_readback.bend as Readback
import ./machine_limit_bounds.bend as Limits
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as NatExtrema
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_extrema.bend as Extrema
import ./u32_division.bend as U32Division
import ./word_encoding.bend as WordEncoding
import ./word_multiplication.bend as Multiply

def groups_fit(+limits: Machine.Limits,+channels: U32,+room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +lanes = U32.to_nat(channels)
  +cap = U32.to_nat(Machine.capacity(limits))
  %Equal.sym(Nat,U32.to_nat(Gemm.blocks(channels)),Ceiling.ceiling(lanes,8n),Encoded.eight_blocks_exact(limits,channels,room)) :
    {Nat.is_le(_,cap) == True{} : Bool}
  Order.transitive(Ceiling.ceiling(lanes,8n),lanes,cap,Ceiling.bounded(lanes,8n,{==}),room)

def pairs_exact(+limits: Machine.Limits,+channels: U32,+groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Gemm.pairs_of(channels)) == Nat.div(Nat.add(U32.to_nat(Gemm.blocks(channels)),1n),2n) : Nat}:
  +groups = Gemm.blocks(channels)
  +count = U32.to_nat(groups)
  +cap = U32.to_nat(Machine.capacity(limits))
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +twice = {%Equal.sym(Nat,Nat.add(cap,cap),Nat.double(cap),Addition.double(cap)) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
    Limits.double_bound(limits,cap,Order.reflexive(cap)) : {Nat.is_le(Nat.add(cap,cap),U32.to_nat(half)) == True{} : Bool}}
  +one_ok = Order.transitive(1n,16n,cap,{==},Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))
  +sum_room = Order.transitive(Nat.add(count,1n),Nat.add(cap,1n),U32.to_nat(half),Order.add_le(count,cap,1n,groups_room),
    Order.transitive(Nat.add(cap,1n),Nat.add(cap,cap),U32.to_nat(half),Order.offset_le(cap,1n,cap,one_ok),twice))
  U32Division.quotient_exact(half,half_ok,(groups + 1 : U32),2,Nat.add(count,1n),2n,
    Arithmetic.add(groups,1,half,half_ok,sum_room),{==},
    Limits.within_half(limits,4n,Order.transitive(4n,16n,cap,{==},Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))),{==})

def pairs_fewer(+limits: Machine.Limits,+channels: U32,+groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}:
  +groups = U32.to_nat(Gemm.blocks(channels))
  %Equal.sym(Nat,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(Nat.add(groups,1n),2n),pairs_exact(limits,channels,groups_room)) :
    {Nat.is_le(_,groups) == True{} : Bool}
  Readback.pairs_below(groups)

def groups_positive(+limits: Machine.Limits,+channels: U32,+room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}:
  +lanes = U32.to_nat(channels)
  +groups = Ceiling.ceiling(lanes,8n)
  %Equal.sym(Nat,U32.to_nat(Gemm.blocks(channels)),groups,Encoded.eight_blocks_exact(limits,channels,room)) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(groups,8n,Order.lt_le(0n,lanes,Nat.mul(groups,8n),positive,Ceiling.covers(lanes,8n,{==})))

def group_span(+limits: Machine.Limits,+channels: U32,+room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(Gemm.blocks(channels)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +lanes = U32.to_nat(channels)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +cap = U32.to_nat(Machine.capacity(limits))
  +padded = Nat.add(lanes,7n)
  +full_groups = Nat.mul(groups,8n)
  +lanes_room = {%Equal.sym(Nat,groups,Ceiling.ceiling(lanes,8n),Encoded.eight_blocks_exact(limits,channels,room)) :
    {Nat.is_le(Nat.mul(_,8n),padded) == True{} : Bool}
    %Ceiling.shifted_formula(lanes,8n,{==}) :
    {Nat.is_le(Nat.mul(_,8n),padded) == True{} : Bool}
    DivisionOrder.product_below(padded,8n,{==}) : {Nat.is_le(full_groups,padded) == True{} : Bool}}
  +four = {%Order.twice(cap) : {Nat.is_le(_,Nat.mul(cap,4n)) == True{} : Bool}
    %RowMajor.multiply_commutative(cap,2n) : {Nat.is_le(_,Nat.mul(cap,4n)) == True{} : Bool}
    RowMajor.mul_le_left(cap,2n,4n,{==}) : {Nat.is_le(Nat.add(cap,cap),Nat.mul(cap,4n)) == True{} : Bool}}
  +extra = Order.transitive(15n,16n,cap,{==},Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))
  +padded_room = {%Equal.sym(Nat,Nat.add(padded,8n),Nat.add(lanes,15n),Addition.associative(lanes,7n,8n)) : {Nat.is_le(_,Nat.add(cap,cap)) == True{} : Bool}
    Order.transitive(Nat.add(lanes,15n),Nat.add(cap,15n),Nat.add(cap,cap),Order.add_le(lanes,cap,15n,room),Order.offset_le(cap,15n,cap,extra)) :
    {Nat.is_le(Nat.add(padded,8n),Nat.add(cap,cap)) == True{} : Bool}}
  +scaled = Nat.add(Nat.mul(groups,2n),2n)
  RowMajor.cancel_multiply_le(scaled,cap,4n,{==},
    %Equal.sym(Nat,Nat.mul(scaled,4n),Nat.add(Nat.mul(Nat.mul(groups,2n),4n),8n),RowMajor.distrib(Nat.mul(groups,2n),2n,4n)) :
      {Nat.is_le(_,Nat.mul(cap,4n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(groups,2n),4n),full_groups,RowMajor.assoc(groups,2n,4n)) :
      {Nat.is_le(Nat.add(_,8n),Nat.mul(cap,4n)) == True{} : Bool}
    Order.transitive(Nat.add(full_groups,8n),Nat.add(padded,8n),Nat.mul(cap,4n),Order.add_le(full_groups,padded,8n,lanes_room),
      Order.transitive(Nat.add(padded,8n),Nat.add(cap,cap),Nat.mul(cap,4n),padded_room,four)))

# A part of the output blocks [first, first + rows) of total outputs has
# min(rows * 8, total - first * 8) outputs: as many blocks as rows, at least
# one, and none past the outputs of the whole call.
type PartOutputs<-first: Nat,-rows: Nat,-total: Nat,-outputs: Nat> is Data:
  PartOutputs{blocks: {Ceiling.ceiling(outputs,8n) == rows : Nat},positive: {Nat.is_lt(0n,outputs) == True{} : Bool},
    end: {Nat.is_le(Nat.add(Nat.mul(first,8n),outputs),total) == True{} : Bool},start: {Nat.is_le(Nat.mul(first,8n),total) == True{} : Bool},
    last: {Nat.is_lt(Nat.mul(Nat.sub(rows,1n),8n),total) == True{} : Bool}}

def clipped_groups(+first: Nat,+rows: Nat,+total: Nat,+outputs: Nat,+facts: PartOutputs<first,rows,total,outputs>) ->
  {Ceiling.ceiling(outputs,8n) == rows : Nat}:
  PartOutputs{blocks,positive,end,start,last} = facts
  blocks

def total_positive(+total: Nat,+blocks_positive: {Nat.is_lt(0n,Ceiling.ceiling(total,8n)) == True{} : Bool}) -> {Nat.is_lt(0n,total) == True{} : Bool}:
  match total:
    case 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(blocks_positive))
    case 1n+ +rest: {==}

def outputs_split(+first: Nat,+rows: Nat,+total: Nat,+rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(first,rows),Ceiling.ceiling(total,8n)) == True{} : Bool}) ->
  PartOutputs<first,rows,total,Nat.min(Nat.mul(rows,8n),Nat.sub(total,Nat.mul(first,8n)))>:
  +s = Nat.sub(rows,1n)
  +blocks = Ceiling.ceiling(total,8n)
  +r_eq = Order.positive_has_last(rows,rows_positive)
  +blocks_positive = Bounds.nat_trans(0n,rows,blocks,rows_positive,Order.transitive(rows,Nat.add(first,rows),blocks,Order.right_le(first,rows),within))
  +b_eq = Order.positive_has_last(blocks,blocks_positive)
  +within_prev = {%Equal.sym(Nat,1n+Nat.sub(blocks,1n),blocks,b_eq) : {Nat.is_le(1n+Nat.add(first,s),_) == True{} : Bool}
    %Addition.successor_right(first,s) : {Nat.is_le(_,blocks) == True{} : Bool}
    %Equal.sym(Nat,1n+s,rows,r_eq) : {Nat.is_le(Nat.add(first,_),blocks) == True{} : Bool}
    within : {Nat.is_le(Nat.add(first,s),Nat.sub(blocks,1n)) == True{} : Bool}}
  +low = Order.le_lt_trans(Nat.mul(Nat.add(first,s),8n),Nat.mul(Nat.sub(blocks,1n),8n),total,RowMajor.mul_le(Nat.add(first,s),Nat.sub(blocks,1n),8n,within_prev),
    Ceiling.previous_below(total,8n,total_positive(total,blocks_positive),{==}))
  +f8 = Nat.mul(first,8n)
  +s8 = Nat.mul(s,8n)
  +low_split = {%RowMajor.distrib(first,s,8n) : {Nat.is_lt(_,total) == True{} : Bool} low : {Nat.is_lt(Nat.add(f8,s8),total) == True{} : Bool}}
  +start = Order.transitive(f8,Nat.add(f8,s8),total,Order.le_add(f8,s8),Bounds.strict_le(Nat.add(f8,s8),total,low_split))
  +d = Nat.sub(total,f8)
  +restored = Order.restore_shift(f8,total,start)
  +s_below_d = Order.cancel_offset_lt(f8,s8,d,{%Equal.sym(Nat,Nat.add(f8,d),total,restored) : {Nat.is_lt(Nat.add(f8,s8),_) == True{} : Bool} low_split :
    {Nat.is_lt(Nat.add(f8,s8),Nat.add(f8,d)) == True{} : Bool}})
  +s_below_r = {%r_eq : {Nat.is_lt(s8,Nat.mul(_,8n)) == True{} : Bool} Order.add_lt(0n,8n,s8,{==}) : {Nat.is_lt(s8,Nat.mul(rows,8n)) == True{} : Bool}}
  +outputs = Nat.min(Nat.mul(rows,8n),d)
  +lower = NatExtrema.minimum_strict_lower(Nat.mul(rows,8n),d,s8,s_below_r,s_below_d)
  +ceiling_low = {%r_eq : {Nat.is_le(_,Ceiling.ceiling(outputs,8n)) == True{} : Bool}
    Order.strict_follow(s,Ceiling.ceiling(outputs,8n),RowMajor.cancel_multiply_lt(s,Ceiling.ceiling(outputs,8n),8n,
      Bounds.nat_trans(s8,outputs,Nat.mul(Ceiling.ceiling(outputs,8n),8n),lower,Ceiling.covers(outputs,8n,{==})))) :
    {Nat.is_le(rows,Ceiling.ceiling(outputs,8n)) == True{} : Bool}}
  PartOutputs{Order.antisymmetric(Ceiling.ceiling(outputs,8n),rows,Ceiling.minimal(outputs,8n,rows,{==},NatExtrema.minimum_left(Nat.mul(rows,8n),d)),ceiling_low),
    Order.le_lt_trans(0n,s8,outputs,Order.zero_le(s8),lower),
    {%restored : {Nat.is_le(Nat.add(f8,outputs),_) == True{} : Bool} Order.offset_le(f8,outputs,d,NatExtrema.minimum_right(Nat.mul(rows,8n),d)) :
      {Nat.is_le(Nat.add(f8,outputs),total) == True{} : Bool}},
    start,
    Order.le_lt_trans(s8,Nat.add(f8,s8),total,Order.right_le(f8,s8),low_split)}

# The machine-word clipping of a part of the output blocks.
def part_outputs_exact(+outputs: U32,+first: U32,+rows: U32,+limits: Machine.Limits,+total: Nat,
  +outputs_exact: {U32.to_nat(outputs) == total : Nat},+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +facts: PartOutputs<U32.to_nat(first),U32.to_nat(rows),total,Nat.min(Nat.mul(U32.to_nat(rows),8n),Nat.sub(total,Nat.mul(U32.to_nat(first),8n)))>) ->
  {U32.to_nat(U32.min((rows * 8 : U32),(outputs - first * 8 : U32))) == Nat.min(Nat.mul(U32.to_nat(rows),8n),Nat.sub(total,Nat.mul(U32.to_nat(first),8n))) : Nat}:
  PartOutputs{+blocks,+positive,+end,+start,+last} = facts
  +f = U32.to_nat(first)
  +r = U32.to_nat(rows)
  +s = Nat.sub(r,1n)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +f8 = Nat.mul(f,8n)
  +s8 = Nat.mul(s,8n)
  +f8_room = Order.transitive(f8,total,cap,start,total_room)
  +first_word = Multiply.bounded_multiply(first,8,capacity,Limits.capacity_fits(limits),f8_room)
  +one_room = Order.transitive(1n+s8,total,cap,Order.strict_follow(s8,total,last),total_room)
  +seven = Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits))
  +r8_room = {%Order.positive_has_last(r,rows_positive) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(cap),Limits.doubled_capacity(limits)) : {Nat.is_le(Nat.mul(1n+s,8n),_) == True{} : Bool}
    %Addition.double(cap) : {Nat.is_le(Nat.mul(1n+s,8n),_) == True{} : Bool}
    Order.transitive(Nat.add(7n,1n+s8),Nat.add(7n,cap),Nat.add(cap,cap),Order.offset_le(7n,1n+s8,cap,one_room),Order.add_le(7n,cap,cap,seven)) :
    {Nat.is_le(Nat.mul(r,8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +rows_word = Multiply.bounded_multiply(rows,8,Machine.half_word(limits),Limits.half_word_fits(limits),r8_room)
  +ordered = {%Equal.sym(Nat,U32.to_nat((first * 8 : U32)),f8,first_word) : {Nat.is_le(_,U32.to_nat(outputs)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(outputs),total,outputs_exact) : {Nat.is_le(f8,_) == True{} : Bool}
    start : {Nat.is_le(U32.to_nat((first * 8 : U32)),U32.to_nat(outputs)) == True{} : Bool}}
  +difference_room = Order.transitive(Nat.sub(U32.to_nat(outputs),U32.to_nat((first * 8 : U32))),U32.to_nat(outputs),cap,
    Order.sub_le(U32.to_nat(outputs),U32.to_nat((first * 8 : U32))),{%Equal.sym(Nat,U32.to_nat(outputs),total,outputs_exact) :
      {Nat.is_le(_,cap) == True{} : Bool} total_room : {Nat.is_le(U32.to_nat(outputs),cap) == True{} : Bool}})
  +difference_word = Arithmetic.subtract(outputs,(first * 8 : U32),capacity,Limits.capacity_fits(limits),ordered,difference_room)
  Equal.trans(Nat,U32.to_nat(U32.min((rows * 8 : U32),(outputs - first * 8 : U32))),Nat.min(U32.to_nat((rows * 8 : U32)),U32.to_nat((outputs - first * 8 : U32))),
    Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)),Extrema.minimum((rows * 8 : U32),(outputs - first * 8 : U32)),
    {%Equal.sym(Nat,U32.to_nat((rows * 8 : U32)),Nat.mul(r,8n),rows_word) : {Nat.min(_,U32.to_nat((outputs - first * 8 : U32))) == Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)) : Nat}
      %Equal.sym(Nat,U32.to_nat((outputs - first * 8 : U32)),Nat.sub(U32.to_nat(outputs),U32.to_nat((first * 8 : U32))),difference_word) :
        {Nat.min(Nat.mul(r,8n),_) == Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)) : Nat}
      %Equal.sym(Nat,U32.to_nat((first * 8 : U32)),f8,first_word) : {Nat.min(Nat.mul(r,8n),Nat.sub(U32.to_nat(outputs),_)) == Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)) : Nat}
      %Equal.sym(Nat,U32.to_nat(outputs),total,outputs_exact) : {Nat.min(Nat.mul(r,8n),Nat.sub(_,f8)) == Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)) : Nat}
      {==} :
      {Nat.min(U32.to_nat((rows * 8 : U32)),U32.to_nat((outputs - first * 8 : U32))) == Nat.min(Nat.mul(r,8n),Nat.sub(total,f8)) : Nat}})
