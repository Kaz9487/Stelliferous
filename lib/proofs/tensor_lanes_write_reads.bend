# The reads of a write along the lane axis: the target reads every target
# piece of a component from itself, so its pieces are the target pieces.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./nat_extrema.bend as NatExtrema
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_ceiling.bend as NatCeiling
import ./nat_division.bend as NatDivision
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./tensor_lanes_write_gather.bend as Gather
import ./tensor_components.bend as Components
import ./word_encoding.bend as WordEncoding
import ./tensor_box_digits.bend as Digits
import ./tensor_shape_refinement.bend as ShapeLaws

# ---- The footprint of an axis read in place ----

def word_one_times(+word: U32) -> {(1 * word : U32) == word : U32}:
  Equal.trans(U32,(1 * word : U32),(word * 1 : U32),word,Arithmetic.multiply_commutative(1,word),Arithmetic.multiply_one(word))

def word_over_one(+word: U32) -> {(word / U32.max(1,1) : U32) == word : U32}:
  Arithmetic.to_nat_injective((word / U32.max(1,1) : U32),word,U32Division.divided_by_one(word))

# The first index an index read in place reaches.
def word_read(+word: U32) -> {((0 + 1 * word : U32) / U32.max(1,1) : U32) == word : U32}:
  %Equal.sym(U32,(1 * word : U32),word,word_one_times(word)) : {((0 + _ : U32) / U32.max(1,1) : U32) == word : U32}
  %Equal.sym(U32,(0 + word : U32),word,Arithmetic.add_zero_left(word)) : {(_ / U32.max(1,1) : U32) == word : U32}
  word_over_one(word)

# The last index an index read in place one wide reaches.
def word_last(+word: U32) -> {((((0 + 1 * word : U32) + 1 : U32) - 1 : U32) / U32.max(1,1) : U32) == word : U32}:
  %Equal.sym(U32,(1 * word : U32),word,word_one_times(word)) : {((((0 + _ : U32) + 1 : U32) - 1 : U32) / U32.max(1,1) : U32) == word : U32}
  %Equal.sym(U32,(0 + word : U32),word,Arithmetic.add_zero_left(word)) : {(((_ + 1 : U32) - 1 : U32) / U32.max(1,1) : U32) == word : U32}
  %Equal.sym(U32,((word + 1 : U32) - 1 : U32),word,
      Equal.trans(U32,((word + 1 : U32) - 1 : U32),(word + (1 - 1 : U32) : U32),word,Arithmetic.subtract_from_sum(word,1,1),Arithmetic.add_zero(word))) :
    {(_ / U32.max(1,1) : U32) == word : U32}
  word_over_one(word)

def word_raised(+word: U32) -> {(U32.max(word,0) - 0 : U32) == word : U32}:
  Equal.trans(U32,(U32.max(word,0) - 0 : U32),U32.max(word,0),word,Arithmetic.subtract_zero(U32.max(word,0)),
    Arithmetic.to_nat_injective(U32.max(word,0),word,
      Equal.trans(Nat,U32.to_nat(U32.max(word,0)),Nat.max(U32.to_nat(word),0n),U32.to_nat(word),Extrema.maximum(word,0),Held.max_zero(U32.to_nat(word)))))

# A footprint of an axis read in place, before its words are simplified.
def placed_footprint(+low: U32,+high: U32,+top: U32,+extent: U32,+block: U32) -> Tasks.Footprint:
  Tasks.Footprint{((U32.max(low,0) - 0 : U32) / U32.max(block,1) : U32),((U32.min(high,top) - 0 : U32) / U32.max(block,1) : U32),
    U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt(top,low)}

# An axis read in place reaches the blocks of the indices it reads.
def own_footprint(+grid: Grid.Grid,+axis: U32,+extent: U32,+low: U32,+high: U32) ->
  {Tasks.range_footprint(Tiled.read_at(grid,axis,extent,1,1,1),low,high) ==
    Tasks.Footprint{(low / U32.max(Grid.block(grid,axis,extent),1) : U32),(U32.min(high,(extent - 1 : U32)) / U32.max(Grid.block(grid,axis,extent),1) : U32),
      U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)} : Tasks.Footprint}:
  +block = Grid.block(grid,axis,extent)
  +first = ((0 + 1 * low : U32) / U32.max(1,1) : U32)
  +last = ((((0 + 1 * high : U32) + 1 : U32) - 1 : U32) / U32.max(1,1) : U32)
  +top = ((0 + extent : U32) - 1 : U32)
  Equal.trans(Tasks.Footprint,placed_footprint(first,last,top,extent,block),placed_footprint(low,last,top,extent,block),
    Tasks.Footprint{(low / U32.max(block,1) : U32),(U32.min(high,(extent - 1 : U32)) / U32.max(block,1) : U32),U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)},
    Equal.cong(U32,Tasks.Footprint,word => placed_footprint(word,last,top,extent,block),first,low,word_read(low)),
    Equal.trans(Tasks.Footprint,placed_footprint(low,last,top,extent,block),placed_footprint(low,high,top,extent,block),
      Tasks.Footprint{(low / U32.max(block,1) : U32),(U32.min(high,(extent - 1 : U32)) / U32.max(block,1) : U32),U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)},
      Equal.cong(U32,Tasks.Footprint,word => placed_footprint(low,word,top,extent,block),last,high,word_last(high)),
      Equal.trans(Tasks.Footprint,placed_footprint(low,high,top,extent,block),placed_footprint(low,high,(extent - 1 : U32),extent,block),
        Tasks.Footprint{(low / U32.max(block,1) : U32),(U32.min(high,(extent - 1 : U32)) / U32.max(block,1) : U32),U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)},
        Equal.cong(U32,Tasks.Footprint,word => placed_footprint(low,high,(word - 1 : U32),extent,block),(0 + extent : U32),extent,Arithmetic.add_zero_left(extent)),
        %word_raised(low) : {placed_footprint(low,high,(extent - 1 : U32),extent,block) ==
          Tasks.Footprint{(_ / U32.max(block,1) : U32),(U32.min(high,(extent - 1 : U32)) / U32.max(block,1) : U32),U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)} : Tasks.Footprint}
        %Arithmetic.subtract_zero(U32.min(high,(extent - 1 : U32))) : {placed_footprint(low,high,(extent - 1 : U32),extent,block) ==
          Tasks.Footprint{((U32.max(low,0) - 0 : U32) / U32.max(block,1) : U32),(_ / U32.max(block,1) : U32),U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)} : Tasks.Footprint}
        {==})))

# ---- The end of the blocks a class reaches ----

# The blocks from the one of the last index of a nonempty range cover it.
def reach_cover(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+nonempty: {Nat.is_lt(0n,value) == True{} : Bool}) ->
  {Nat.is_le(value,Nat.mul(1n+Nat.div(Nat.sub(value,1n),divisor),divisor)) == True{} : Bool}:
  %Order.positive_has_last(value,nonempty) : {Nat.is_le(_,Nat.mul(1n+Nat.div(Nat.sub(value,1n),divisor),divisor)) == True{} : Bool}
  Order.strict_follow(Nat.sub(value,1n),Nat.mul(1n+Nat.div(Nat.sub(value,1n),divisor),divisor),DivisionOrder.next_above(Nat.sub(value,1n),divisor,positive))

# The blocks up to whole blocks end there.
def reach_tight(+blocks: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,Nat.mul(blocks,divisor)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+Nat.div(Nat.sub(Nat.mul(blocks,divisor),1n),divisor),divisor),Nat.mul(blocks,divisor)) == True{} : Bool}:
  +value = Nat.mul(blocks,divisor)
  +last = Nat.div(Nat.sub(value,1n),divisor)
  RowMajor.mul_le(1n+last,blocks,divisor,
    Order.strict_follow(last,blocks,RowMajor.cancel_multiply_lt(last,blocks,divisor,
      Order.le_lt_trans(Nat.mul(last,divisor),Nat.sub(value,1n),value,DivisionOrder.product_below(Nat.sub(value,1n),divisor,positive),Order.last_below(value,nonempty)))))

# The blocks a class reaches end where its indices end, clipped to the axis.
def reach_end(choice: Bool,+blocks: Nat,+divisor: Nat,+extent: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,Nat.min(Nat.mul(blocks,divisor),extent)) == True{} : Bool},+classified: {Nat.is_le(Nat.mul(blocks,divisor),extent) == choice : Bool}) ->
  {Nat.min(Nat.mul(1n+Nat.div(Nat.sub(Nat.min(Nat.mul(blocks,divisor),extent),1n),divisor),divisor),extent) == Nat.min(Nat.mul(blocks,divisor),extent) : Nat}:
  match choice:
    case True{}:
      +value = Nat.mul(blocks,divisor)
      +clipped = NatExtrema.minimum_when_ordered(value,extent,classified)
      +filled = {%clipped : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,value) == True{} : Bool}}
      %Equal.sym(Nat,Nat.min(value,extent),value,clipped) :
        {Nat.min(Nat.mul(1n+Nat.div(Nat.sub(_,1n),divisor),divisor),extent) == _ : Nat}
      %Equal.sym(Nat,Nat.mul(1n+Nat.div(Nat.sub(value,1n),divisor),divisor),value,Order.antisymmetric(Nat.mul(1n+Nat.div(Nat.sub(value,1n),divisor),divisor),value,reach_tight(blocks,divisor,positive,filled),reach_cover(value,divisor,positive,filled))) :
        {Nat.min(_,extent) == value : Nat}
      clipped
    case False{}:
      +value = Nat.mul(blocks,divisor)
      +below = Bounds.strict_le(extent,value,Order.ge_false(extent,value,Equal.trans(Bool,Nat.is_ge(extent,value),Nat.is_le(value,extent),False{},Order.ge_le(extent,value),classified)))
      +clipped = Equal.trans(Nat,Nat.min(value,extent),Nat.min(extent,value),extent,NatExtrema.minimum_commutative(value,extent),NatExtrema.minimum_when_ordered(extent,value,below))
      +filled = {%clipped : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,extent) == True{} : Bool}}
      %Equal.sym(Nat,Nat.min(value,extent),extent,clipped) :
        {Nat.min(Nat.mul(1n+Nat.div(Nat.sub(_,1n),divisor),divisor),extent) == _ : Nat}
      Equal.trans(Nat,Nat.min(Nat.mul(1n+Nat.div(Nat.sub(extent,1n),divisor),divisor),extent),Nat.min(extent,Nat.mul(1n+Nat.div(Nat.sub(extent,1n),divisor),divisor)),extent,
        NatExtrema.minimum_commutative(Nat.mul(1n+Nat.div(Nat.sub(extent,1n),divisor),divisor),extent),
        NatExtrema.minimum_when_ordered(extent,Nat.mul(1n+Nat.div(Nat.sub(extent,1n),divisor),divisor),reach_cover(extent,divisor,positive,filled)))

# ---- The box of an axis read in place ----

def below_more(+value: Nat,+more: Nat,+positive: {Nat.is_lt(0n,more) == True{} : Bool}) -> {Nat.is_lt(value,Nat.add(more,value)) == True{} : Bool}:
  %Order.commutative(value,more) : {Nat.is_lt(value,_) == True{} : Bool}
  %Addition.zero_right(value) : {Nat.is_lt(_,Nat.add(value,more)) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.add(value,0n),Nat.add(value,more)),Nat.is_lt(0n,more),Order.lt_offset(value,0n,more)) : {_ == True{} : Bool}
  positive

# The words of a class read in place.
def class_low(+first: U32,+block: U32) -> U32:
  (first * block : U32)

def class_clip(+first: U32,+count: U32,+block: U32,+extent: U32) -> U32:
  U32.min(((first + count) * block : U32),extent)

# A class of an axis read in place reads the indices of its blocks, clipped
# to the axis.
def own_span(+grid: Grid.Grid,+axis: U32,+extent: U32,+first: U32,+count: U32,+coordinate: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,extent),1))),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,axis,extent))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
  +begins: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},+ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent)))) == True{} : Bool},
  +reach_room: {Nat.is_le(Nat.mul(U32.to_nat((first + count : U32)),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.box_span(Some{Tasks.range_footprint(Tiled.read_at(grid,axis,extent,1,1,1),class_low(first,Grid.block(grid,axis,extent)),(class_clip(first,count,Grid.block(grid,axis,extent),extent) - 1 : U32))},extent,Grid.block(grid,axis,extent),coordinate) ==
    Tasks.Span{class_low(first,Grid.block(grid,axis,extent)),(class_clip(first,count,Grid.block(grid,axis,extent),extent) - class_low(first,Grid.block(grid,axis,extent)) : U32)} : Tasks.Span}:
  +block = Grid.block(grid,axis,extent)
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +start = U32.to_nat(first)
  +reach = U32.to_nat((first + count : U32))
  +low = class_low(first,block)
  +clip = class_clip(first,count,block,extent)
  +high = (clip - 1 : U32)
  +extent_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +unit_exact = Held.stride_divisor(block,Arithmetic.word_positive_nonzero(block,block_positive))
  +divisor_bound = {%unit_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} divisor_room :
    {Nat.is_le(Nat.double(size),U32.to_nat(cap)) == True{} : Bool}}
  +low_value = Nat.mul(start,size)
  +low_below = Equal.trans(Bool,Nat.is_lt(low_value,total),Nat.is_lt(start,NatCeiling.ceiling(total,size)),True{},
    Equal.sym(Bool,Nat.is_lt(start,NatCeiling.ceiling(total,size)),Nat.is_lt(low_value,total),NatCeiling.upper_comparison(total,size,start,block_positive)),
    Order.lt_le(start,reach,NatCeiling.ceiling(total,size),begins,ends))
  +low_room = Order.transitive(low_value,total,U32.to_nat(cap),Bounds.strict_le(low_value,total,low_below),extent_room)
  +low_exact = Arithmetic.product_exact(cap,cap_ok,first,block,start,size,{==},{==},low_room)
  +top_exact = Arithmetic.product_exact(cap,cap_ok,(first + count : U32),block,reach,size,{==},{==},reach_room)
  +clipped = Nat.min(Nat.mul(reach,size),total)
  +clip_exact = Equal.trans(Nat,U32.to_nat(clip),Nat.min(U32.to_nat(((first + count) * block : U32)),total),clipped,Extrema.minimum(((first + count) * block : U32),extent),
    Equal.cong(Nat,Nat,value => Nat.min(value,total),U32.to_nat(((first + count) * block : U32)),Nat.mul(reach,size),top_exact))
  +more = Order.lt_le(low_value,Nat.add(size,low_value),Nat.mul(reach,size),below_more(low_value,size,block_positive),
    RowMajor.mul_le(1n+start,reach,size,Order.strict_follow(start,reach,begins)))
  +clip_above = Gather.min_above(Nat.mul(reach,size),total,low_value,more,low_below)
  +clip_positive = Order.le_lt_trans(0n,low_value,clipped,Order.zero_le(low_value),clip_above)
  +clip_room = Order.transitive(clipped,total,U32.to_nat(cap),NatExtrema.minimum_right(Nat.mul(reach,size),total),extent_room)
  +high_exact = Arithmetic.difference_exact(cap,cap_ok,clip,1,clipped,1n,clip_exact,{==},Order.strict_follow(0n,clipped,clip_positive),clip_room)
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,extent,1,total,1n,{==},{==},Order.strict_follow(0n,total,extent_positive),extent_room)
  +high_le = DivisionOrder.subtract_le(clipped,total,1n,NatExtrema.minimum_right(Nat.mul(reach,size),total))
  +top_exact_min = Equal.trans(Nat,U32.to_nat(U32.min(high,(extent - 1 : U32))),Nat.min(U32.to_nat(high),U32.to_nat((extent - 1 : U32))),Nat.sub(clipped,1n),
    Extrema.minimum(high,(extent - 1 : U32)),
    %Equal.sym(Nat,U32.to_nat(high),Nat.sub(clipped,1n),high_exact) : {Nat.min(_,U32.to_nat((extent - 1 : U32))) == Nat.sub(clipped,1n) : Nat}
    %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(total,1n),last_exact) : {Nat.min(Nat.sub(clipped,1n),_) == Nat.sub(clipped,1n) : Nat}
    NatExtrema.minimum_when_ordered(Nat.sub(clipped,1n),Nat.sub(total,1n),high_le))
  +blocks_first = (low / U32.max(block,1) : U32)
  +blocks_last = (U32.min(high,(extent - 1 : U32)) / U32.max(block,1) : U32)
  +first_exact = Equal.trans(Nat,U32.to_nat(blocks_first),Nat.div(low_value,size),start,
    U32Division.quotient_exact(cap,cap_ok,low,U32.max(block,1),low_value,size,low_exact,unit_exact,divisor_bound,block_positive),
    NatDivision.exact_quotient(start,size,block_positive))
  +last_value = Nat.div(Nat.sub(clipped,1n),size)
  +last_word = U32Division.quotient_exact(cap,cap_ok,U32.min(high,(extent - 1 : U32)),U32.max(block,1),Nat.sub(clipped,1n),size,top_exact_min,unit_exact,divisor_bound,block_positive)
  +last_room = Order.transitive(Nat.mul(1n+last_value,size),Nat.add(size,Nat.sub(clipped,1n)),U32.to_nat(cap),
    Order.offset_le(size,Nat.mul(last_value,size),Nat.sub(clipped,1n),DivisionOrder.product_below(Nat.sub(clipped,1n),size,block_positive)),
    Order.transitive(Nat.add(size,Nat.sub(clipped,1n)),Nat.add(size,total),U32.to_nat(cap),
      Order.offset_le(size,Nat.sub(clipped,1n),total,Order.transitive(Nat.sub(clipped,1n),clipped,total,Order.sub_le(clipped,1n),NatExtrema.minimum_right(Nat.mul(reach,size),total))),
      {%Order.commutative(total,size) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(size,total),U32.to_nat(cap)) == True{} : Bool}}))
  +next_room = Order.transitive(Nat.add(last_value,1n),Nat.mul(1n+last_value,size),U32.to_nat(cap),
    {%Equal.sym(Nat,Nat.add(last_value,1n),1n+last_value,Order.commutative(last_value,1n)) : {Nat.is_le(_,Nat.mul(1n+last_value,size)) == True{} : Bool}
      RowMajor.product_covers(1n+last_value,size,block_positive) : {Nat.is_le(Nat.add(last_value,1n),Nat.mul(1n+last_value,size)) == True{} : Bool}},last_room)
  +next_exact = Equal.trans(Nat,U32.to_nat((blocks_last + 1 : U32)),Nat.add(last_value,1n),1n+last_value,
    Arithmetic.sum_exact(cap,cap_ok,blocks_last,1,last_value,1n,last_word,{==},next_room),Order.commutative(last_value,1n))
  +end_exact = Arithmetic.product_exact(cap,cap_ok,(blocks_last + 1 : U32),block,1n+last_value,size,next_exact,{==},last_room)
  +low_again = Arithmetic.to_nat_injective(U32.min((blocks_first * block : U32),extent),low,
    Equal.trans(Nat,U32.to_nat(U32.min((blocks_first * block : U32),extent)),Nat.min(low_value,total),U32.to_nat(low),
      Equal.trans(Nat,U32.to_nat(U32.min((blocks_first * block : U32),extent)),Nat.min(U32.to_nat((blocks_first * block : U32)),total),Nat.min(low_value,total),
        Extrema.minimum((blocks_first * block : U32),extent),
        Equal.cong(Nat,Nat,value => Nat.min(value,total),U32.to_nat((blocks_first * block : U32)),low_value,
          Arithmetic.product_exact(cap,cap_ok,blocks_first,block,start,size,first_exact,{==},low_room))),
      Equal.trans(Nat,Nat.min(low_value,total),low_value,U32.to_nat(low),NatExtrema.minimum_when_ordered(low_value,total,Bounds.strict_le(low_value,total,low_below)),
        Equal.sym(Nat,U32.to_nat(low),low_value,low_exact))))
  +end_again = Arithmetic.to_nat_injective(U32.min(((blocks_last + 1) * block : U32),extent),clip,
    Equal.trans(Nat,U32.to_nat(U32.min(((blocks_last + 1) * block : U32),extent)),Nat.min(Nat.mul(1n+last_value,size),total),U32.to_nat(clip),
      Equal.trans(Nat,U32.to_nat(U32.min(((blocks_last + 1) * block : U32),extent)),Nat.min(U32.to_nat(((blocks_last + 1) * block : U32)),total),
        Nat.min(Nat.mul(1n+last_value,size),total),Extrema.minimum(((blocks_last + 1) * block : U32),extent),
        Equal.cong(Nat,Nat,value => Nat.min(value,total),U32.to_nat(((blocks_last + 1) * block : U32)),Nat.mul(1n+last_value,size),end_exact)),
      Equal.trans(Nat,Nat.min(Nat.mul(1n+last_value,size),total),clipped,U32.to_nat(clip),
        reach_end(Nat.is_le(Nat.mul(reach,size),total),reach,size,total,block_positive,clip_positive,{==}),Equal.sym(Nat,U32.to_nat(clip),clipped,clip_exact))))
  +below_low = Order.not_below(low_value,Nat.sub(total,1n),Order.before_last(low_value,total,low_below))
  +last_low = Equal.trans(Bool,U32.is_lt((extent - 1 : U32),low),Nat.is_lt(U32.to_nat((extent - 1 : U32)),U32.to_nat(low)),False{},Comparison.lt((extent - 1 : U32),low),
    {%Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(total,1n),last_exact) : {Nat.is_lt(_,U32.to_nat(low)) == False{} : Bool}
      %Equal.sym(Nat,U32.to_nat(low),low_value,low_exact) : {Nat.is_lt(Nat.sub(total,1n),_) == False{} : Bool}
      below_low : {Nat.is_lt(U32.to_nat((extent - 1 : U32)),U32.to_nat(low)) == False{} : Bool}})
  +high_zero = Equal.trans(Bool,U32.is_lt(high,0),Nat.is_lt(U32.to_nat(high),0n),False{},Comparison.lt(high,0),Order.not_below(0n,U32.to_nat(high),Order.zero_le(U32.to_nat(high))))
  +empty = Equal.trans(Bool,U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low),False{} || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low),False{},
    Equal.cong(Bool,Bool,flag => flag || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low),U32.is_zero(extent),False{},Arithmetic.word_positive_nonzero(extent,extent_positive)),
    Equal.trans(Bool,False{} || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low),False{} || False{} || U32.is_lt((extent - 1 : U32),low),False{},
      Equal.cong(Bool,Bool,flag => False{} || flag || U32.is_lt((extent - 1 : U32),low),U32.is_lt(high,0),False{},high_zero),
      Equal.cong(Bool,Bool,flag => False{} || False{} || flag,U32.is_lt((extent - 1 : U32),low),False{},last_low)))
  %Equal.sym(Tasks.Footprint,Tasks.range_footprint(Tiled.read_at(grid,axis,extent,1,1,1),low,high),
      Tasks.Footprint{blocks_first,blocks_last,U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low)},own_footprint(grid,axis,extent,low,high)) :
    {Tasks.box_span(Some{_},extent,block,coordinate) == Tasks.Span{low,(clip - low : U32)} : Tasks.Span}
  %Equal.sym(Bool,U32.is_zero(extent) || U32.is_lt(high,0) || U32.is_lt((extent - 1 : U32),low),False{},empty) :
    {Tasks.box_span(Some{Tasks.Footprint{blocks_first,blocks_last,_}},extent,block,coordinate) == Tasks.Span{low,(clip - low : U32)} : Tasks.Span}
  %Equal.sym(U32,U32.min((blocks_first * block : U32),extent),low,low_again) :
    {Tasks.Span{_,(U32.min(((blocks_last + 1) * block : U32),extent) - _ : U32)} == Tasks.Span{low,(clip - low : U32)} : Tasks.Span}
  %Equal.sym(U32,U32.min(((blocks_last + 1) * block : U32),extent),clip,end_again) :
    {Tasks.Span{low,(_ - low : U32)} == Tasks.Span{low,(clip - low : U32)} : Tasks.Span}
  {==}

# A class of an axis of unit blocks inside the axis reads its own indices.
def unit_class(+first: U32,+count: U32,+extent: U32,+inside: {Nat.is_le(U32.to_nat((first + count : U32)),U32.to_nat(extent)) == True{} : Bool}) ->
  {Tasks.Span{class_low(first,1),(class_clip(first,count,1,extent) - class_low(first,1) : U32)} == Tasks.Span{first,count} : Tasks.Span}:
  +clipped = Arithmetic.to_nat_injective(U32.min((first + count : U32),extent),(first + count : U32),
    Equal.trans(Nat,U32.to_nat(U32.min((first + count : U32),extent)),Nat.min(U32.to_nat((first + count : U32)),U32.to_nat(extent)),U32.to_nat((first + count : U32)),
      Extrema.minimum((first + count : U32),extent),NatExtrema.minimum_when_ordered(U32.to_nat((first + count : U32)),U32.to_nat(extent),inside)))
  %Equal.sym(U32,(first * 1 : U32),first,Arithmetic.multiply_one(first)) :
    {Tasks.Span{_,(U32.min(((first + count) * 1 : U32),extent) - _ : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,((first + count) * 1 : U32),(first + count : U32),Arithmetic.multiply_one((first + count : U32))) :
    {Tasks.Span{first,(U32.min(_,extent) - first : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,U32.min((first + count : U32),extent),(first + count : U32),clipped) :
    {Tasks.Span{first,(_ - first : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,((first + count : U32) - first : U32),count,Arithmetic.subtract_first(first,count)) :
    {Tasks.Span{first,_} == Tasks.Span{first,count} : Tasks.Span}
  {==}

# The blocks of a class clipped to its axis are the class's blocks.
def clip_ceiling(+reach: Nat,+size: Nat,+total: Nat,+positive: {Nat.is_lt(0n,size) == True{} : Bool},+begun: {Nat.is_lt(0n,reach) == True{} : Bool},
  +ends: {Nat.is_le(reach,NatCeiling.ceiling(total,size)) == True{} : Bool}) ->
  {NatCeiling.ceiling(Nat.min(Nat.mul(reach,size),total),size) == reach : Nat}:
  +clipped = Nat.min(Nat.mul(reach,size),total)
  +before = Nat.sub(reach,1n)
  +whole = Order.positive_has_last(reach,begun)
  +upper = Equal.trans(Bool,Nat.is_le(NatCeiling.ceiling(clipped,size),reach),Nat.is_le(clipped,Nat.mul(reach,size)),True{},
    NatCeiling.lower_comparison(clipped,size,reach,positive),NatExtrema.minimum_left(Nat.mul(reach,size),total))
  +more = {%whole : {Nat.is_lt(Nat.mul(before,size),Nat.mul(_,size)) == True{} : Bool}
    below_more(Nat.mul(before,size),size,positive) : {Nat.is_lt(Nat.mul(before,size),Nat.mul(reach,size)) == True{} : Bool}}
  +inside = Equal.trans(Bool,Nat.is_lt(Nat.mul(before,size),total),Nat.is_lt(before,NatCeiling.ceiling(total,size)),True{},
    Equal.sym(Bool,Nat.is_lt(before,NatCeiling.ceiling(total,size)),Nat.is_lt(Nat.mul(before,size),total),NatCeiling.upper_comparison(total,size,before,positive)),
    Order.succ_le_lt(before,NatCeiling.ceiling(total,size),
      {%Equal.sym(Nat,1n+before,reach,whole) : {Nat.is_le(_,NatCeiling.ceiling(total,size)) == True{} : Bool}
        ends : {Nat.is_le(1n+before,NatCeiling.ceiling(total,size)) == True{} : Bool}}))
  +below = Equal.trans(Bool,Nat.is_lt(before,NatCeiling.ceiling(clipped,size)),Nat.is_lt(Nat.mul(before,size),clipped),True{},
    NatCeiling.upper_comparison(clipped,size,before,positive),Gather.min_above(Nat.mul(reach,size),total,Nat.mul(before,size),more,inside))
  +lower = {%whole : {Nat.is_le(_,NatCeiling.ceiling(clipped,size)) == True{} : Bool}
    Order.strict_follow(before,NatCeiling.ceiling(clipped,size),below) : {Nat.is_le(reach,NatCeiling.ceiling(clipped,size)) == True{} : Bool}}
  Order.antisymmetric(NatCeiling.ceiling(clipped,size),reach,upper,lower)

# A class of the last axis of the pieces, read in place, covers the piece
# blocks of the class.
def piece_class(+first: U32,+count: U32,+extent: U32,+units: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(units,1))),U32.to_nat(cap)) == True{} : Bool},
  +units_positive: {Nat.is_lt(0n,U32.to_nat(units)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(units)),U32.to_nat(cap)) == True{} : Bool},
  +begins: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(units))) == True{} : Bool},
  +reach_room: {Nat.is_le(Nat.mul(U32.to_nat((first + count : U32)),U32.to_nat(units)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
    (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
      class_low(first,units) / U32.max(units,1) : U32)} == Tasks.Span{first,count} : Tasks.Span}:
  +size = U32.to_nat(units)
  +total = U32.to_nat(extent)
  +start = U32.to_nat(first)
  +reach = U32.to_nat((first + count : U32))
  +low = class_low(first,units)
  +clip = class_clip(first,count,units,extent)
  +extent_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +unit_exact = Held.stride_divisor(units,Arithmetic.word_positive_nonzero(units,units_positive))
  +divisor_bound = {%unit_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} divisor_room :
    {Nat.is_le(Nat.double(size),U32.to_nat(cap)) == True{} : Bool}}
  +low_value = Nat.mul(start,size)
  +low_below = Equal.trans(Bool,Nat.is_lt(low_value,total),Nat.is_lt(start,NatCeiling.ceiling(total,size)),True{},
    Equal.sym(Bool,Nat.is_lt(start,NatCeiling.ceiling(total,size)),Nat.is_lt(low_value,total),NatCeiling.upper_comparison(total,size,start,units_positive)),
    Order.lt_le(start,reach,NatCeiling.ceiling(total,size),begins,ends))
  +low_room = Order.transitive(low_value,total,U32.to_nat(cap),Bounds.strict_le(low_value,total,low_below),extent_room)
  +low_exact = Arithmetic.product_exact(cap,cap_ok,first,units,start,size,{==},{==},low_room)
  +top_exact = Arithmetic.product_exact(cap,cap_ok,(first + count : U32),units,reach,size,{==},{==},reach_room)
  +clipped = Nat.min(Nat.mul(reach,size),total)
  +clip_exact = Equal.trans(Nat,U32.to_nat(clip),Nat.min(U32.to_nat(((first + count) * units : U32)),total),clipped,Extrema.minimum(((first + count) * units : U32),extent),
    Equal.cong(Nat,Nat,value => Nat.min(value,total),U32.to_nat(((first + count) * units : U32)),Nat.mul(reach,size),top_exact))
  +more = Order.lt_le(low_value,Nat.add(size,low_value),Nat.mul(reach,size),below_more(low_value,size,units_positive),
    RowMajor.mul_le(1n+start,reach,size,Order.strict_follow(start,reach,begins)))
  +clip_above = Gather.min_above(Nat.mul(reach,size),total,low_value,more,low_below)
  +ordered = Bounds.strict_le(low_value,clipped,clip_above)
  +clip_room = Order.transitive(clipped,total,U32.to_nat(cap),NatExtrema.minimum_right(Nat.mul(reach,size),total),extent_room)
  +low_div = Arithmetic.to_nat_injective((low / U32.max(units,1) : U32),first,
    Equal.trans(Nat,U32.to_nat((low / U32.max(units,1) : U32)),Nat.div(low_value,size),start,
      U32Division.quotient_exact(cap,cap_ok,low,U32.max(units,1),low_value,size,low_exact,unit_exact,divisor_bound,units_positive),
      NatDivision.exact_quotient(start,size,units_positive)))
  +gap_exact = Arithmetic.difference_exact(cap,cap_ok,clip,low,clipped,low_value,clip_exact,low_exact,ordered,clip_room)
  +restored = Order.restore_shift(low_value,clipped,ordered)
  +sum_back = Arithmetic.to_nat_injective((low + (clip - low : U32) : U32),clip,
    Equal.trans(Nat,U32.to_nat((low + (clip - low : U32) : U32)),Nat.add(low_value,Nat.sub(clipped,low_value)),U32.to_nat(clip),
      Arithmetic.sum_exact(cap,cap_ok,low,(clip - low : U32),low_value,Nat.sub(clipped,low_value),low_exact,gap_exact,
        {%Equal.sym(Nat,Nat.add(low_value,Nat.sub(clipped,low_value)),clipped,restored) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          clip_room : {Nat.is_le(Nat.add(low_value,Nat.sub(clipped,low_value)),U32.to_nat(cap)) == True{} : Bool}}),
      Equal.trans(Nat,Nat.add(low_value,Nat.sub(clipped,low_value)),clipped,U32.to_nat(clip),restored,Equal.sym(Nat,U32.to_nat(clip),clipped,clip_exact))))
  +unit_once = Equal.trans(Nat,U32.to_nat(U32.max(U32.max(units,1),1)),U32.to_nat(U32.max(units,1)),size,
    Held.stride_divisor(U32.max(units,1),Arithmetic.word_positive_nonzero(U32.max(units,1),
      {%Equal.sym(Nat,U32.to_nat(U32.max(units,1)),size,unit_exact) : {Nat.is_lt(0n,_) == True{} : Bool} units_positive :
        {Nat.is_lt(0n,U32.to_nat(U32.max(units,1))) == True{} : Bool}})),unit_exact)
  +clip_cap = {%Equal.sym(Nat,U32.to_nat(clip),clipped,clip_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} clip_room :
    {Nat.is_le(U32.to_nat(clip),U32.to_nat(cap)) == True{} : Bool}}
  +up_exact = Components.divided_value(cap,cap_ok,clip,U32.max(units,1),size,unit_once,units_positive,divisor_bound,clip_cap)
  +up_word = Arithmetic.to_nat_injective(Grid.divided_up(clip,U32.max(units,1)),(first + count : U32),
    Equal.trans(Nat,U32.to_nat(Grid.divided_up(clip,U32.max(units,1))),NatCeiling.ceiling(U32.to_nat(clip),size),reach,up_exact,
      {%Equal.sym(Nat,U32.to_nat(clip),clipped,clip_exact) : {NatCeiling.ceiling(_,size) == reach : Nat}
        clip_ceiling(reach,size,total,units_positive,Order.le_lt_trans(0n,start,reach,Order.zero_le(start),begins),ends) :
        {NatCeiling.ceiling(U32.to_nat(clip),size) == reach : Nat}}))
  %Equal.sym(U32,(low / U32.max(units,1) : U32),first,low_div) :
    {Tasks.Span{_,(Grid.divided_up((low + (clip - low : U32) : U32),U32.max(units,1)) - _ : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,(low + (clip - low : U32) : U32),clip,sum_back) :
    {Tasks.Span{first,(Grid.divided_up(_,U32.max(units,1)) - first : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,Grid.divided_up(clip,U32.max(units,1)),(first + count : U32),up_word) :
    {Tasks.Span{first,(_ - first : U32)} == Tasks.Span{first,count} : Tasks.Span}
  %Equal.sym(U32,((first + count : U32) - first : U32),count,Arithmetic.subtract_first(first,count)) :
    {Tasks.Span{first,_} == Tasks.Span{first,count} : Tasks.Span}
  {==}

# ---- The boxes of the axes read in place ----

# An axis read in place is not found at another axis.
def own_skip(+head: U32,+grid: Grid.Grid,+index: U32,+rest: List<&2,Tasks.Reading>,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +classes: List<&2,Tasks.Class>,+axis: U32,+apart: {U32.is_eq(index,axis) == False{} : Bool}) ->
  {Tasks.read_footprint(Con{Tasks.Reading{head,Grid.block(grid,index,head),Tiled.read_at(grid,index,head,1,1,1)},rest},Con{Tasks.Class{first,count,source_first,source_count},classes},axis) ==
    Tasks.read_footprint(rest,classes,axis) : Maybe<&2,Tasks.Footprint>}:
  %Equal.sym(Bool,U32.is_eq(index,axis),False{},apart) :
    {Tasks.found_footprint(_,Tiled.read_at(grid,index,head,1,1,1),head,Grid.block(grid,index,head),first,count,Tasks.read_footprint(rest,classes,axis)) ==
      Tasks.read_footprint(rest,classes,axis) : Maybe<&2,Tasks.Footprint>}
  {==}

# An axis read in place is found at its own axis, with the indices of its
# class.
def own_found(+head: U32,+grid: Grid.Grid,+index: U32,+rest: List<&2,Tasks.Reading>,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +classes: List<&2,Tasks.Class>,+axis: U32,+same: {U32.is_eq(index,axis) == True{} : Bool}) ->
  {Tasks.read_footprint(Con{Tasks.Reading{head,Grid.block(grid,index,head),Tiled.read_at(grid,index,head,1,1,1)},rest},Con{Tasks.Class{first,count,source_first,source_count},classes},axis) ==
    Some{Tasks.range_footprint(Tiled.read_at(grid,index,head,1,1,1),class_low(first,Grid.block(grid,index,head)),(class_clip(first,count,Grid.block(grid,index,head),head) - 1 : U32))} :
    Maybe<&2,Tasks.Footprint>}:
  %Equal.sym(Bool,U32.is_eq(index,axis),True{},same) :
    {Tasks.found_footprint(_,Tiled.read_at(grid,index,head,1,1,1),head,Grid.block(grid,index,head),first,count,Tasks.read_footprint(rest,classes,axis)) ==
      Some{Tasks.range_footprint(Tiled.read_at(grid,index,head,1,1,1),class_low(first,Grid.block(grid,index,head)),(class_clip(first,count,Grid.block(grid,index,head),head) - 1 : U32))} :
      Maybe<&2,Tasks.Footprint>}
  {==}

def word_apart(+left: U32,+right: U32,+below: {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}) -> {U32.is_eq(left,right) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(left,right),Nat.is_eq(U32.to_nat(left),U32.to_nat(right)),False{},Comparison.eq(left,right),
    Order.less_not_equal(U32.to_nat(left),U32.to_nat(right),below))

# The boxes of the axes from axis on do not see an axis read in place before
# them.
def boxed_drop(extents: List<&2,U32>,coordinates: List<&2,U32>,+head: U32,+grid: Grid.Grid,+index: U32,+rest: List<&2,Tasks.Reading>,+first: U32,+count: U32,
  +source_first: U32,+source_count: U32,+classes: List<&2,Tasks.Class>,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(index),U32.to_nat(axis)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.boxed(Con{Tasks.Reading{head,Grid.block(grid,index,head),Tiled.read_at(grid,index,head,1,1,1)},rest},Con{Tasks.Class{first,count,source_first,source_count},classes},grid,
    extents,coordinates,axis) == Tasks.boxed(rest,classes,grid,extents,coordinates,axis) : List<&2,Tasks.Span>}:
  match extents coordinates:
    case Con{+extent,+more} Con{+coordinate,+coords}:
      +readings = {Con{Tasks.Reading{head,Grid.block(grid,index,head),Tiled.read_at(grid,index,head,1,1,1)},rest} : List<&2,Tasks.Reading>}
      +all = {Con{Tasks.Class{first,count,source_first,source_count},classes} : List<&2,Tasks.Class>}
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(U32.to_nat(axis),List.length(&2,U32,more),U32.to_nat(cap),room))
      +later = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+U32.to_nat(axis),next) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
        Order.lt_succ(U32.to_nat(index),U32.to_nat(axis),Bounds.strict_le(U32.to_nat(index),U32.to_nat(axis),below)) :
        {Nat.is_lt(U32.to_nat(index),U32.to_nat((axis + 1 : U32))) == True{} : Bool}}
      %Equal.sym(Maybe<&2,Tasks.Footprint>,Tasks.read_footprint(readings,all,axis),Tasks.read_footprint(rest,classes,axis),
          own_skip(head,grid,index,rest,first,count,source_first,source_count,classes,axis,word_apart(index,axis,below))) :
        {Con{Tasks.box_span(_,extent,Grid.block(grid,axis,extent),coordinate),Tasks.boxed(readings,all,grid,more,coords,(axis + 1 : U32))} ==
          Con{Tasks.box_span(Tasks.read_footprint(rest,classes,axis),extent,Grid.block(grid,axis,extent),coordinate),Tasks.boxed(rest,classes,grid,more,coords,(axis + 1 : U32))} :
          List<&2,Tasks.Span>}
      Equal.cong(List<&2,Tasks.Span>,List<&2,Tasks.Span>,tail => Con{Tasks.box_span(Tasks.read_footprint(rest,classes,axis),extent,Grid.block(grid,axis,extent),coordinate),tail},
        Tasks.boxed(readings,all,grid,more,coords,(axis + 1 : U32)),Tasks.boxed(rest,classes,grid,more,coords,(axis + 1 : U32)),
        boxed_drop(more,coords,head,grid,index,rest,first,count,source_first,source_count,classes,(axis + 1 : U32),cap,cap_ok,later,
          Gather.range_rest(axis,List.length(&2,U32,more),U32.to_nat(cap),next,room)))
    case Con{+extent,+more} Nil{}:
      {==}
    case Nil{} +coordinates:
      {==}

# The blocks of a class inside its axis end within the axis and one block.
def reach_within(+reach: Nat,+size: Nat,+total: Nat,+cap: Nat,+positive: {Nat.is_lt(0n,size) == True{} : Bool},+begun: {Nat.is_lt(0n,reach) == True{} : Bool},
  +ends: {Nat.is_le(reach,NatCeiling.ceiling(total,size)) == True{} : Bool},+room: {Nat.is_le(Nat.add(total,size),cap) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(reach,size),cap) == True{} : Bool}:
  +before = Nat.sub(reach,1n)
  +whole = Order.positive_has_last(reach,begun)
  +below = Equal.trans(Bool,Nat.is_lt(Nat.mul(before,size),total),Nat.is_lt(before,NatCeiling.ceiling(total,size)),True{},
    Equal.sym(Bool,Nat.is_lt(before,NatCeiling.ceiling(total,size)),Nat.is_lt(Nat.mul(before,size),total),NatCeiling.upper_comparison(total,size,before,positive)),
    Order.succ_le_lt(before,NatCeiling.ceiling(total,size),
      {%Equal.sym(Nat,1n+before,reach,whole) : {Nat.is_le(_,NatCeiling.ceiling(total,size)) == True{} : Bool}
        ends : {Nat.is_le(1n+before,NatCeiling.ceiling(total,size)) == True{} : Bool}}))
  %whole : {Nat.is_le(Nat.mul(_,size),cap) == True{} : Bool}
  Order.transitive(Nat.mul(1n+before,size),Nat.add(size,total),cap,Order.offset_le(size,Nat.mul(before,size),total,Bounds.strict_le(Nat.mul(before,size),total,below)),
    {%Order.commutative(total,size) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.add(size,total),cap) == True{} : Bool}})

# What a class of an axis read in place needs: it begins and ends inside the
# blocks of the axis, its blocks are not empty, nor is the axis, and the axis
# and a block fit the word.
def ClassFits(class: Tasks.Class,+extent: U32,+block: U32,+cap: U32) -> Data:
  match class:
    case Tasks.Class{+first,+count,+source_first,+source_count}:
      Held.Both<{Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
        Held.Both<{Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
          Held.Both<{Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},Held.Both<{Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
            Held.Both<{Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
              {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(cap)) == True{} : Bool}>>>>>

def OwnFits(extents: List<&2,U32>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+cap: U32) -> Data:
  match extents classes:
    case Con{+extent,rest} Con{+class,more}:
      Held.Both<ClassFits(class,extent,Grid.block(grid,axis,extent),cap),OwnFits(rest,more,grid,(axis + 1 : U32),cap)>
    case Con{+extent,rest} Nil{}:
      Empty
    case Nil{} Nil{}:
      Unit
    case Nil{} Con{+class,more}:
      Empty

# The box of a class of an axis read in place.
def own_box(class: Tasks.Class,+extent: U32,+block: U32) -> Tasks.Span:
  match class:
    case Tasks.Class{+first,+count,source_first,source_count}:
      Tasks.Span{class_low(first,block),(class_clip(first,count,block,extent) - class_low(first,block) : U32)}

def own_boxes(extents: List<&2,U32>,coordinates: List<&2,U32>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32) -> List<&2,Tasks.Span>:
  match extents coordinates classes:
    case Con{+extent,rest} Con{coordinate,coords} Con{class,more}:
      Con{own_box(class,extent,Grid.block(grid,axis,extent)),own_boxes(rest,coords,more,grid,(axis + 1 : U32))}
    case extents coordinates classes:
      []

# The axes read in place box their classes.
def boxed_own(extents: List<&2,U32>,coordinates: List<&2,U32>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},+fits: OwnFits(extents,classes,grid,axis,cap)) ->
  {Tasks.boxed(Tiled.own_readings(extents,grid,axis),classes,grid,extents,coordinates,axis) == own_boxes(extents,coordinates,classes,grid,axis) : List<&2,Tasks.Span>}:
  match extents coordinates classes:
    case Con{+extent,+rest} Con{+coordinate,+coords} Con{Tasks.Class{+first,+count,+source_first,+source_count},+more}:
      Held.Both{Held.Both{+begins,Held.Both{+ends,Held.Both{+block_positive,Held.Both{+extent_positive,Held.Both{+axis_room,+divisor_room}}}}},+later} = fits
      +block = Grid.block(grid,axis,extent)
      +own = Tiled.own_readings(rest,grid,(axis + 1 : U32))
      +all = {Con{Tasks.Class{first,count,source_first,source_count},more} : List<&2,Tasks.Class>}
      +readings = {Con{Tasks.Reading{extent,block,Tiled.read_at(grid,axis,extent,1,1,1)},own} : List<&2,Tasks.Reading>}
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(U32.to_nat(axis),List.length(&2,U32,rest),U32.to_nat(cap),room))
      +after = Gather.range_rest(axis,List.length(&2,U32,rest),U32.to_nat(cap),next,room)
      +head = Equal.trans(Tasks.Span,Tasks.box_span(Tasks.read_footprint(readings,all,axis),extent,block,coordinate),
        Tasks.box_span(Some{Tasks.range_footprint(Tiled.read_at(grid,axis,extent,1,1,1),class_low(first,block),(class_clip(first,count,block,extent) - 1 : U32))},extent,block,coordinate),
        own_box(Tasks.Class{first,count,source_first,source_count},extent,block),
        Equal.cong(Maybe<&2,Tasks.Footprint>,Tasks.Span,found => Tasks.box_span(found,extent,block,coordinate),Tasks.read_footprint(readings,all,axis),
          Some{Tasks.range_footprint(Tiled.read_at(grid,axis,extent,1,1,1),class_low(first,block),(class_clip(first,count,block,extent) - 1 : U32))},
          own_found(extent,grid,axis,own,first,count,source_first,source_count,more,axis,Arithmetic.equal_reflexive(axis))),
        own_span(grid,axis,extent,first,count,coordinate,cap,cap_ok,divisor_room,block_positive,extent_positive,axis_room,begins,ends,
          reach_within(U32.to_nat((first + count : U32)),U32.to_nat(block),U32.to_nat(extent),U32.to_nat(cap),block_positive,
            Order.le_lt_trans(0n,U32.to_nat(first),U32.to_nat((first + count : U32)),Order.zero_le(U32.to_nat(first)),begins),ends,axis_room)))
      +tail = Equal.trans(List<&2,Tasks.Span>,Tasks.boxed(readings,all,grid,rest,coords,(axis + 1 : U32)),Tasks.boxed(own,more,grid,rest,coords,(axis + 1 : U32)),
        own_boxes(rest,coords,more,grid,(axis + 1 : U32)),
        boxed_drop(rest,coords,extent,grid,axis,own,first,count,source_first,source_count,more,(axis + 1 : U32),cap,cap_ok,
          {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+U32.to_nat(axis),next) : {Nat.is_lt(U32.to_nat(axis),_) == True{} : Bool}
            Order.lt_succ(U32.to_nat(axis),U32.to_nat(axis),Order.reflexive(U32.to_nat(axis))) : {Nat.is_lt(U32.to_nat(axis),U32.to_nat((axis + 1 : U32))) == True{} : Bool}},after),
        boxed_own(rest,coords,more,grid,(axis + 1 : U32),cap,cap_ok,after,later))
      %Equal.sym(Tasks.Span,Tasks.box_span(Tasks.read_footprint(readings,all,axis),extent,block,coordinate),own_box(Tasks.Class{first,count,source_first,source_count},extent,block),head) :
        {Con{_,Tasks.boxed(readings,all,grid,rest,coords,(axis + 1 : U32))} == Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,block),
        own_boxes(rest,coords,more,grid,(axis + 1 : U32))} : List<&2,Tasks.Span>}
      Equal.cong(List<&2,Tasks.Span>,List<&2,Tasks.Span>,later_boxes => Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,block),later_boxes},
        Tasks.boxed(readings,all,grid,rest,coords,(axis + 1 : U32)),own_boxes(rest,coords,more,grid,(axis + 1 : U32)),tail)
    case Con{+extent,+rest} Con{+coordinate,+coords} Nil{}:
      Empty.absurd({Tasks.boxed(Tiled.own_readings(Con{extent,rest},grid,axis),Nil{},grid,Con{extent,rest},Con{coordinate,coords},axis) ==
        own_boxes(Con{extent,rest},Con{coordinate,coords},Nil{},grid,axis) : List<&2,Tasks.Span>},fits)
    case Con{+extent,+rest} Nil{} +classes:
      {==}
    case Nil{} +coordinates +classes:
      {==}

# The target spans past the prefix axes are none.
def target_stop(classes: List<&2,Tasks.Class>,+prefix: U32,+axis: U32,+stop: {U32.is_lt(axis,prefix) == False{} : Bool}) ->
  {Tasks.target_spans(classes,prefix,axis) == Nil{} : List<&2,Tasks.Span>}:
  match classes:
    case Nil{}:
      {==}
    case Con{Tasks.Class{+target_first,+target_count,+source_first,+source_count},+rest}:
      %Equal.sym(Bool,U32.is_lt(axis,prefix),False{},stop) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{target_first,target_count},Tasks.target_spans(rest,prefix,(axis + 1 : U32))},Nil{}) == Nil{} : List<&2,Tasks.Span>}
      {==}

# An axis before the last prefix axis, or the last, lies among the prefix
# axes, and an axis past them does not.
def lower_then(+axis: U32,+prefix: U32,+next: {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat},+lower: {U32.is_lt((axis + 1 : U32),prefix) == True{} : Bool}) ->
  {U32.is_lt(axis,prefix) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(axis,prefix),Nat.is_lt(U32.to_nat(axis),U32.to_nat(prefix)),True{},Comparison.lt(axis,prefix),
    Order.le_lt_trans(U32.to_nat(axis),1n+U32.to_nat(axis),U32.to_nat(prefix),Order.successor(U32.to_nat(axis),U32.to_nat(axis),Order.reflexive(U32.to_nat(axis))),
      {%next : {Nat.is_lt(_,U32.to_nat(prefix)) == True{} : Bool}
        Equal.trans(Bool,Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),U32.is_lt((axis + 1 : U32),prefix),True{},
          Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),Comparison.lt((axis + 1 : U32),prefix)),lower) :
        {Nat.is_lt(1n+U32.to_nat(axis),U32.to_nat(prefix)) == True{} : Bool}}))

def equal_then(+axis: U32,+prefix: U32,+next: {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat},+equal: {U32.is_eq((axis + 1 : U32),prefix) == True{} : Bool}) ->
  {U32.is_lt(axis,prefix) == True{} : Bool}:
  +same = Arithmetic.equal_of_is_eq((axis + 1 : U32),prefix,equal)
  Equal.trans(Bool,U32.is_lt(axis,prefix),Nat.is_lt(U32.to_nat(axis),U32.to_nat(prefix)),True{},Comparison.lt(axis,prefix),
    {%same : {Nat.is_lt(U32.to_nat(axis),U32.to_nat(_)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+U32.to_nat(axis),next) : {Nat.is_lt(U32.to_nat(axis),_) == True{} : Bool}
      Order.lt_succ(U32.to_nat(axis),U32.to_nat(axis),Order.reflexive(U32.to_nat(axis))) : {Nat.is_lt(U32.to_nat(axis),U32.to_nat(prefix)) == True{} : Bool}})

def neither_then(+axis: U32,+prefix: U32,+next: {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat},+lower: {U32.is_lt((axis + 1 : U32),prefix) == False{} : Bool},
  +equal: {U32.is_eq((axis + 1 : U32),prefix) == False{} : Bool}) -> {U32.is_lt(axis,prefix) == False{} : Bool}:
  +after = U32.to_nat((axis + 1 : U32))
  +not_below = Order.not_less(after,U32.to_nat(prefix),Equal.trans(Bool,Nat.is_lt(after,U32.to_nat(prefix)),U32.is_lt((axis + 1 : U32),prefix),False{},
    Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(after,U32.to_nat(prefix)),Comparison.lt((axis + 1 : U32),prefix)),lower))
  +apart = Equal.trans(Bool,Nat.is_eq(after,U32.to_nat(prefix)),U32.is_eq((axis + 1 : U32),prefix),False{},
    Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),Nat.is_eq(after,U32.to_nat(prefix)),Comparison.eq((axis + 1 : U32),prefix)),equal)
  +before = {%next : {Nat.is_lt(U32.to_nat(prefix),_) == True{} : Bool} Gather.le_apart_lt(U32.to_nat(prefix),after,not_below,apart) :
    {Nat.is_lt(U32.to_nat(prefix),1n+U32.to_nat(axis)) == True{} : Bool}}
  Equal.trans(Bool,U32.is_lt(axis,prefix),Nat.is_lt(U32.to_nat(axis),U32.to_nat(prefix)),False{},Comparison.lt(axis,prefix),
    Order.not_below(U32.to_nat(prefix),U32.to_nat(axis),
      {Order.strict_follow(U32.to_nat(prefix),1n+U32.to_nat(axis),before) : {Nat.is_le(U32.to_nat(prefix),U32.to_nat(axis)) == True{} : Bool}}))

# ---- The block spans of the boxes read in place ----

def ceiling_one(+value: Nat) -> {Nat.is_le(NatCeiling.ceiling(value,1n),value) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(NatCeiling.ceiling(value,1n),value),Nat.is_le(value,Nat.mul(value,1n)),True{},NatCeiling.lower_comparison(value,1n,value,{==}),
    %Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value)) : {Nat.is_le(value,_) == True{} : Bool} Order.reflexive(value))

# The block span of the box of one axis read in place is the target span of
# its class.
def spans_head(lower: Bool,equal: Bool,+extent: U32,+rest: List<&2,Tasks.Span>,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +more: List<&2,Tasks.Class>,+prefix: U32,+units: U32,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +next: {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat},+lower_is: {U32.is_lt((axis + 1 : U32),prefix) == lower : Bool},
  +equal_is: {U32.is_eq((axis + 1 : U32),prefix) == equal : Bool},
  +fits: ClassFits(Tasks.Class{first,count,source_first,source_count},extent,Bool.pick(U32,lower,1,Bool.pick(U32,equal,units,extent)),cap),
  +later: {Tasks.block_spans(rest,prefix,units,(axis + 1 : U32)) == Tasks.target_spans(more,prefix,(axis + 1 : U32)) : List<&2,Tasks.Span>}) ->
  {Tasks.block_spans(Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,Bool.pick(U32,lower,1,Bool.pick(U32,equal,units,extent))),rest},prefix,units,axis) ==
    Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}:
  match lower:
    case True{}:
      Held.Both{+begins,Held.Both{+ends,Held.Both{+positive,Held.Both{+extent_positive,Held.Both{+axis_room,+divisor_room}}}}} = fits
      +inside = Order.transitive(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),1n),U32.to_nat(extent),ends,ceiling_one(U32.to_nat(extent)))
      %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),True{},lower_is) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,1),Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
          Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(class_low(first,1) / U32.max(units,1) : U32),
            (Grid.divided_up((class_low(first,1) + (class_clip(first,count,1,extent) - class_low(first,1) : U32) : U32),U32.max(units,1)) - class_low(first,1) / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
          Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}
      %Equal.sym(Bool,U32.is_lt(axis,prefix),True{},lower_then(axis,prefix,next,lower_is)) :
        {Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,1),Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))} ==
          Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(more,prefix,(axis + 1 : U32))},Nil{}) : List<&2,Tasks.Span>}
      Equal.trans(List<&2,Tasks.Span>,Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,1),Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
        Con{Tasks.Span{first,count},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},Con{Tasks.Span{first,count},Tasks.target_spans(more,prefix,(axis + 1 : U32))},
        Equal.cong(Tasks.Span,List<&2,Tasks.Span>,span => Con{span,Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
          own_box(Tasks.Class{first,count,source_first,source_count},extent,1),Tasks.Span{first,count},unit_class(first,count,extent,inside)),
        Equal.cong(List<&2,Tasks.Span>,List<&2,Tasks.Span>,tail => Con{Tasks.Span{first,count},tail},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32)),
          Tasks.target_spans(more,prefix,(axis + 1 : U32)),later))
    case False{}:
      match equal:
        case True{}:
          Held.Both{+begins,Held.Both{+ends,Held.Both{+positive,Held.Both{+extent_positive,Held.Both{+axis_room,+divisor_room}}}}} = fits
          %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lower_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,units),Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
              Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
                  class_low(first,units) / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
              Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),True{},equal_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
                  class_low(first,units) / U32.max(units,1) : U32)},Nil{}},Nil{}) ==
              Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_lt(axis,prefix),True{},equal_then(axis,prefix,next,equal_is)) :
            {Con{Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
                  class_low(first,units) / U32.max(units,1) : U32)},Nil{}} ==
              Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(more,prefix,(axis + 1 : U32))},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(List<&2,Tasks.Span>,Tasks.target_spans(more,prefix,(axis + 1 : U32)),Nil{},target_stop(more,prefix,(axis + 1 : U32),lower_is)) :
            {Con{Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
                  class_low(first,units) / U32.max(units,1) : U32)},Nil{}} == Con{Tasks.Span{first,count},_} : List<&2,Tasks.Span>}
          Equal.cong(Tasks.Span,List<&2,Tasks.Span>,span => Con{span,Nil{}},
            Tasks.Span{(class_low(first,units) / U32.max(units,1) : U32),
              (Grid.divided_up((class_low(first,units) + (class_clip(first,count,units,extent) - class_low(first,units) : U32) : U32),U32.max(units,1)) -
                class_low(first,units) / U32.max(units,1) : U32)},Tasks.Span{first,count},
            piece_class(first,count,extent,units,cap,cap_ok,divisor_room,positive,extent_positive,axis_room,begins,ends,
              reach_within(U32.to_nat((first + count : U32)),U32.to_nat(units),U32.to_nat(extent),U32.to_nat(cap),positive,
                Order.le_lt_trans(0n,U32.to_nat(first),U32.to_nat((first + count : U32)),Order.zero_le(U32.to_nat(first)),begins),ends,axis_room)))
        case False{}:
          %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lower_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{own_box(Tasks.Class{first,count,source_first,source_count},extent,extent),Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
              Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(class_low(first,extent) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,extent) + (class_clip(first,count,extent,extent) - class_low(first,extent) : U32) : U32),U32.max(units,1)) -
                  class_low(first,extent) / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
              Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),False{},equal_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{(class_low(first,extent) / U32.max(units,1) : U32),
                (Grid.divided_up((class_low(first,extent) + (class_clip(first,count,extent,extent) - class_low(first,extent) : U32) : U32),U32.max(units,1)) -
                  class_low(first,extent) / U32.max(units,1) : U32)},Nil{}},Nil{}) ==
              Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},more},prefix,axis) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_lt(axis,prefix),False{},neither_then(axis,prefix,next,lower_is,equal_is)) :
            {Nil{} == Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},Tasks.target_spans(more,prefix,(axis + 1 : U32))},Nil{}) : List<&2,Tasks.Span>}
          {==}

# The block spans of the boxes of the axes read in place are the target spans
# of their classes.
def spans_own(extents: List<&2,U32>,coordinates: List<&2,U32>,classes: List<&2,Tasks.Class>,+count: U32,+prefix: U32,+outer: U32,+last: U32,+unit_cells: U32,
  +units: U32,+per: U32,+pieces: U32,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +paired: {Nat.is_eq(List.length(&2,U32,coordinates),List.length(&2,U32,extents)) == True{} : Bool},
  +fits: OwnFits(extents,classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis,cap)) ->
  {Tasks.block_spans(own_boxes(extents,coordinates,classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis),prefix,units,axis) ==
    Tasks.target_spans(classes,prefix,axis) : List<&2,Tasks.Span>}:
  match extents coordinates classes:
    case Con{+extent,+rest} Con{+coordinate,+coords} Con{Tasks.Class{+first,+number,+source_first,+source_count},+more}:
      Held.Both{+head,+later} = fits
      +grid = {Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces} : Grid.Grid}
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(U32.to_nat(axis),List.length(&2,U32,rest),U32.to_nat(cap),room))
      spans_head(U32.is_lt((axis + 1 : U32),prefix),U32.is_eq((axis + 1 : U32),prefix),extent,own_boxes(rest,coords,more,grid,(axis + 1 : U32)),first,number,
        source_first,source_count,more,prefix,units,axis,cap,cap_ok,next,{==},{==},head,
        spans_own(rest,coords,more,count,prefix,outer,last,unit_cells,units,per,pieces,(axis + 1 : U32),cap,cap_ok,
          Gather.range_rest(axis,List.length(&2,U32,rest),U32.to_nat(cap),next,room),paired,later))
    case Con{+extent,+rest} Con{+coordinate,+coords} Nil{}:
      Empty.absurd({Tasks.block_spans(own_boxes(Con{extent,rest},Con{coordinate,coords},Nil{},Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis),prefix,units,axis) ==
        Tasks.target_spans(Nil{},prefix,axis) : List<&2,Tasks.Span>},fits)
    case Con{+extent,+rest} Nil{} +classes:
      Empty.absurd({Tasks.block_spans(own_boxes(Con{extent,rest},Nil{},classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis),prefix,units,axis) ==
        Tasks.target_spans(classes,prefix,axis) : List<&2,Tasks.Span>},WordEncoding.false_true(paired))
    case Nil{} +coordinates Nil{}:
      {==}
    case Nil{} +coordinates Con{+class,+more}:
      Empty.absurd({Tasks.block_spans(own_boxes(Nil{},coordinates,Con{class,more},Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis),prefix,units,axis) ==
        Tasks.target_spans(Con{class,more},prefix,axis) : List<&2,Tasks.Span>},fits)

# ---- The pieces of the target read in place ----

# The box of a class of an axis read in place lies inside the axis and is not
# empty.
def own_box_fits(+block: U32,+extent: U32,+first: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +begins: {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool},
  +ends: {Nat.is_le(U32.to_nat((first + count : U32)),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))) == True{} : Bool},
  +reach_room: {Nat.is_le(Nat.mul(U32.to_nat((first + count : U32)),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(Nat.add(U32.to_nat(class_low(first,block)),U32.to_nat((class_clip(first,count,block,extent) - class_low(first,block) : U32))),U32.to_nat(extent)) == True{} : Bool},
    {Nat.is_lt(0n,U32.to_nat((class_clip(first,count,block,extent) - class_low(first,block) : U32))) == True{} : Bool}>:
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +start = U32.to_nat(first)
  +reach = U32.to_nat((first + count : U32))
  +low = class_low(first,block)
  +clip = class_clip(first,count,block,extent)
  +extent_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +low_value = Nat.mul(start,size)
  +low_below = Equal.trans(Bool,Nat.is_lt(low_value,total),Nat.is_lt(start,NatCeiling.ceiling(total,size)),True{},
    Equal.sym(Bool,Nat.is_lt(start,NatCeiling.ceiling(total,size)),Nat.is_lt(low_value,total),NatCeiling.upper_comparison(total,size,start,block_positive)),
    Order.lt_le(start,reach,NatCeiling.ceiling(total,size),begins,ends))
  +low_room = Order.transitive(low_value,total,U32.to_nat(cap),Bounds.strict_le(low_value,total,low_below),extent_room)
  +low_exact = Arithmetic.product_exact(cap,cap_ok,first,block,start,size,{==},{==},low_room)
  +top_exact = Arithmetic.product_exact(cap,cap_ok,(first + count : U32),block,reach,size,{==},{==},reach_room)
  +clipped = Nat.min(Nat.mul(reach,size),total)
  +clip_exact = Equal.trans(Nat,U32.to_nat(clip),Nat.min(U32.to_nat(((first + count) * block : U32)),total),clipped,Extrema.minimum(((first + count) * block : U32),extent),
    Equal.cong(Nat,Nat,value => Nat.min(value,total),U32.to_nat(((first + count) * block : U32)),Nat.mul(reach,size),top_exact))
  +more = Order.lt_le(low_value,Nat.add(size,low_value),Nat.mul(reach,size),below_more(low_value,size,block_positive),
    RowMajor.mul_le(1n+start,reach,size,Order.strict_follow(start,reach,begins)))
  +clip_above = Gather.min_above(Nat.mul(reach,size),total,low_value,more,low_below)
  +clip_inside = NatExtrema.minimum_right(Nat.mul(reach,size),total)
  +clip_room = Order.transitive(clipped,total,U32.to_nat(cap),clip_inside,extent_room)
  +ordered = Bounds.strict_le(low_value,clipped,clip_above)
  +width_exact = Arithmetic.difference_exact(cap,cap_ok,clip,low,clipped,low_value,clip_exact,low_exact,ordered,clip_room)
  Held.Both{
    {%Equal.sym(Nat,U32.to_nat(low),low_value,low_exact) : {Nat.is_le(Nat.add(_,U32.to_nat((clip - low : U32))),total) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat((clip - low : U32)),Nat.sub(clipped,low_value),width_exact) : {Nat.is_le(Nat.add(low_value,_),total) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(low_value,Nat.sub(clipped,low_value)),clipped,Order.restore_shift(low_value,clipped,ordered)) : {Nat.is_le(_,total) == True{} : Bool}
      clip_inside : {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat((clip - low : U32))),total) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((clip - low : U32)),Nat.sub(clipped,low_value),width_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      Order.positive_difference(clipped,low_value,clip_above) : {Nat.is_lt(0n,U32.to_nat((clip - low : U32))) == True{} : Bool}}}

# The boxes of the axes read in place lie inside the axes and none is empty.
def own_fits(extents: List<&2,U32>,coordinates: List<&2,U32>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +paired: {Nat.is_eq(List.length(&2,U32,coordinates),List.length(&2,U32,extents)) == True{} : Bool},+fits: OwnFits(extents,classes,grid,axis,cap)) ->
  Held.Both<Components.SpansIn(own_boxes(extents,coordinates,classes,grid,axis),extents),
    {Nat.is_lt(0n,Components.span_total(own_boxes(extents,coordinates,classes,grid,axis))) == True{} : Bool}>:
  match extents coordinates classes:
    case Con{+extent,+rest} Con{+coordinate,+coords} Con{Tasks.Class{+first,+count,+source_first,+source_count},+more}:
      Held.Both{Held.Both{+begins,Held.Both{+ends,Held.Both{+block_positive,Held.Both{+extent_positive,Held.Both{+axis_room,+divisor_room}}}}},+later} = fits
      +block = Grid.block(grid,axis,extent)
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(U32.to_nat(axis),List.length(&2,U32,rest),U32.to_nat(cap),room))
      +after = Gather.range_rest(axis,List.length(&2,U32,rest),U32.to_nat(cap),next,room)
      +boxes = own_boxes(rest,coords,more,grid,(axis + 1 : U32))
      +width = U32.to_nat((class_clip(first,count,block,extent) - class_low(first,block) : U32))
      +head = own_box_fits(block,extent,first,count,cap,cap_ok,block_positive,extent_positive,axis_room,begins,ends,
        reach_within(U32.to_nat((first + count : U32)),U32.to_nat(block),U32.to_nat(extent),U32.to_nat(cap),block_positive,
          Order.le_lt_trans(0n,U32.to_nat(first),U32.to_nat((first + count : U32)),Order.zero_le(U32.to_nat(first)),begins),ends,axis_room))
      +tail = own_fits(rest,coords,more,grid,(axis + 1 : U32),cap,cap_ok,after,paired,later)
      Held.Both{Held.Both{Held.first_of({Nat.is_le(Nat.add(U32.to_nat(class_low(first,block)),width),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(0n,width) == True{} : Bool},head),Held.first_of(Components.SpansIn(boxes,rest),{Nat.is_lt(0n,Components.span_total(boxes)) == True{} : Bool},tail)},
        Order.product_positive(width,Components.span_total(boxes),Held.second_of({Nat.is_le(Nat.add(U32.to_nat(class_low(first,block)),width),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(0n,width) == True{} : Bool},head),Held.second_of(Components.SpansIn(boxes,rest),{Nat.is_lt(0n,Components.span_total(boxes)) == True{} : Bool},tail))}
    case Con{+extent,+rest} Con{+coordinate,+coords} Nil{}:
      Empty.absurd(Held.Both<Components.SpansIn(own_boxes(Con{extent,rest},Con{coordinate,coords},Nil{},grid,axis),Con{extent,rest}),
        {Nat.is_lt(0n,Components.span_total(own_boxes(Con{extent,rest},Con{coordinate,coords},Nil{},grid,axis))) == True{} : Bool}>,fits)
    case Con{+extent,+rest} Nil{} +classes:
      Empty.absurd(Held.Both<Components.SpansIn(own_boxes(Con{extent,rest},Nil{},classes,grid,axis),Con{extent,rest}),
        {Nat.is_lt(0n,Components.span_total(own_boxes(Con{extent,rest},Nil{},classes,grid,axis))) == True{} : Bool}>,WordEncoding.false_true(paired))
    case Nil{} +coordinates +classes:
      Held.Both{Unit{},{==}}

# The box of the axes read in place is not empty.
def own_nonzero(+extents: List<&2,U32>,+coordinates: List<&2,U32>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +paired: {Nat.is_eq(List.length(&2,U32,coordinates),List.length(&2,U32,extents)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},+fits: OwnFits(extents,classes,grid,axis,cap)) ->
  {U32.is_zero(Tasks.span_count(own_boxes(extents,coordinates,classes,grid,axis))) == False{} : Bool}:
  +boxes = own_boxes(extents,coordinates,classes,grid,axis)
  +fitted = own_fits(extents,coordinates,classes,grid,axis,cap,cap_ok,room,paired,fits)
  +exact = Components.span_count_value(boxes,cap,cap_ok,
    Order.transitive(Components.span_total(boxes),ShapeLaws.elements(extents),U32.to_nat(cap),Components.span_total_le(boxes,extents,Held.first_of(Components.SpansIn(boxes,extents),{Nat.is_lt(0n,Components.span_total(boxes)) == True{} : Bool},fitted)),cells_room))
  Arithmetic.word_positive_nonzero(Tasks.span_count(boxes),
    {%Equal.sym(Nat,U32.to_nat(Tasks.span_count(boxes)),Components.span_total(boxes),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      Held.second_of(Components.SpansIn(boxes,extents),{Nat.is_lt(0n,Components.span_total(boxes)) == True{} : Bool},fitted) : {Nat.is_lt(0n,U32.to_nat(Tasks.span_count(boxes))) == True{} : Bool}})

# A separable component reads, of the target, exactly its target pieces.
def own_pieces(+cells: List<&2,U32>,+offset: U32,+classes: List<&2,Tasks.Class>,+count: U32,+prefix: U32,+outer: U32,+last: U32,+unit_cells: U32,
  +units: U32,+per: U32,+pieces: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(0),List.length(&2,U32,cells)),U32.to_nat(cap)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(cap)) == True{} : Bool},
  +fits: OwnFits(cells,classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},0,cap)) ->
  {Tasks.box_pieces(Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},cells,
    Tasks.boxed(Tiled.own_readings(cells,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},0),classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},
      cells,Grid.coordinates(cells,offset),0)) ==
    Tasks.target_when(True{},classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},cells,prefix) : List<&2,U32>}:
  +grid = {Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces} : Grid.Grid}
  +coordinates = Grid.coordinates(cells,offset)
  +paired = {%Digits.coordinates_count(cells,offset) : {Nat.is_eq(List.length(&2,U32,coordinates),_) == True{} : Bool}
    Order.equal_reflexive(List.length(&2,U32,coordinates)) : {Nat.is_eq(List.length(&2,U32,coordinates),List.length(&2,U32,cells)) == True{} : Bool}}
  +boxes = own_boxes(cells,coordinates,classes,grid,0)
  +same = boxed_own(cells,coordinates,classes,grid,0,cap,cap_ok,room,fits)
  +nonzero = own_nonzero(cells,coordinates,classes,grid,0,cap,cap_ok,room,paired,cells_room,fits)
  +spans = spans_own(cells,coordinates,classes,count,prefix,outer,last,unit_cells,units,per,pieces,0,cap,cap_ok,room,paired,fits)
  %Equal.sym(List<&2,Tasks.Span>,Tasks.boxed(Tiled.own_readings(cells,grid,0),classes,grid,cells,coordinates,0),boxes,same) :
    {Tasks.box_pieces(grid,cells,_) == Tasks.pieces_in(grid,cells,Tasks.target_spans(classes,prefix,0)) : List<&2,U32>}
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(boxes)),False{},nonzero) :
    {Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,cells,Tasks.block_spans(boxes,prefix,units,0))) ==
      Tasks.pieces_in(grid,cells,Tasks.target_spans(classes,prefix,0)) : List<&2,U32>}
  Equal.cong(List<&2,Tasks.Span>,List<&2,U32>,listed => Tasks.pieces_in(grid,cells,listed),Tasks.block_spans(boxes,prefix,units,0),
    Tasks.target_spans(classes,prefix,0),spans)

# ---- The pieces of the whole target ----

# What a whole axis needs: its blocks are not empty and the axis and a block
# fit the word.
def WholeAxis(+extent: U32,+block: U32,+cap: U32) -> Data:
  Held.Both<{Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},Held.Both<{Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
    {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(cap)) == True{} : Bool}>>

def WholeFits(extents: List<&2,U32>,+grid: Grid.Grid,+axis: U32,+cap: U32) -> Data:
  match extents:
    case Con{+extent,rest}:
      Held.Both<WholeAxis(extent,Grid.block(grid,axis,extent),cap),WholeFits(rest,grid,(axis + 1 : U32),cap)>
    case Nil{}:
      Unit

def ceiling_by_one(+value: Nat) -> {NatCeiling.ceiling(value,1n) == value : Nat}:
  Order.antisymmetric(NatCeiling.ceiling(value,1n),value,ceiling_one(value),
    {%Order.times_one(NatCeiling.ceiling(value,1n)) : {Nat.is_le(value,_) == True{} : Bool}
      NatCeiling.covers(value,1n,{==}) : {Nat.is_le(value,NatCeiling.ceiling(value,1n)) == True{} : Bool}})

# An axis of unit blocks has a block for each index.
def unit_blocks(+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +two_cap: {Nat.is_le(Nat.double(1n),U32.to_nat(cap)) == True{} : Bool},+extent_cap: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.target_blocks(extent,1) == extent : U32}:
  Arithmetic.to_nat_injective(Tasks.target_blocks(extent,1),extent,
    Equal.trans(Nat,U32.to_nat(Tasks.target_blocks(extent,1)),NatCeiling.ceiling(U32.to_nat(extent),1n),U32.to_nat(extent),
      Components.divided_value(cap,cap_ok,extent,1,1n,{==},{==},two_cap,extent_cap),ceiling_by_one(U32.to_nat(extent))))

# The last prefix axis read whole covers all its piece blocks.
def last_whole(+extent: U32,+units: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(units,1))),U32.to_nat(cap)) == True{} : Bool},+units_positive: {Nat.is_lt(0n,U32.to_nat(units)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(units)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.Span{(0 / U32.max(units,1) : U32),(Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)} ==
    Tasks.Span{0,Tasks.target_blocks(extent,units)} : Tasks.Span}:
  +size = U32.to_nat(units)
  +total = U32.to_nat(extent)
  +unit_exact = Held.stride_divisor(units,Arithmetic.word_positive_nonzero(units,units_positive))
  +divisor_bound = {%unit_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} divisor_room :
    {Nat.is_le(Nat.double(size),U32.to_nat(cap)) == True{} : Bool}}
  +extent_cap = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +zero_div = Arithmetic.to_nat_injective((0 / U32.max(units,1) : U32),0,
    Equal.trans(Nat,U32.to_nat((0 / U32.max(units,1) : U32)),Nat.div(Nat.mul(0n,size),size),0n,
      U32Division.quotient_exact(cap,cap_ok,0,U32.max(units,1),Nat.mul(0n,size),size,{==},unit_exact,divisor_bound,units_positive),
      NatDivision.exact_quotient(0n,size,units_positive)))
  +unit_once = Equal.trans(Nat,U32.to_nat(U32.max(U32.max(units,1),1)),U32.to_nat(U32.max(units,1)),size,
    Held.stride_divisor(U32.max(units,1),Arithmetic.word_positive_nonzero(U32.max(units,1),
      {%Equal.sym(Nat,U32.to_nat(U32.max(units,1)),size,unit_exact) : {Nat.is_lt(0n,_) == True{} : Bool} units_positive :
        {Nat.is_lt(0n,U32.to_nat(U32.max(units,1))) == True{} : Bool}})),unit_exact)
  +up_same = Arithmetic.to_nat_injective(Grid.divided_up(extent,U32.max(units,1)),Grid.divided_up(extent,units),
    Equal.trans(Nat,U32.to_nat(Grid.divided_up(extent,U32.max(units,1))),NatCeiling.ceiling(total,size),U32.to_nat(Grid.divided_up(extent,units)),
      Components.divided_value(cap,cap_ok,extent,U32.max(units,1),size,unit_once,units_positive,divisor_bound,extent_cap),
      Equal.sym(Nat,U32.to_nat(Grid.divided_up(extent,units)),NatCeiling.ceiling(total,size),
        Components.divided_value(cap,cap_ok,extent,units,size,unit_exact,units_positive,divisor_bound,extent_cap))))
  %Equal.sym(U32,(0 / U32.max(units,1) : U32),0,zero_div) :
    {Tasks.Span{_,(Grid.divided_up((0 + extent : U32),U32.max(units,1)) - _ : U32)} == Tasks.Span{0,Tasks.target_blocks(extent,units)} : Tasks.Span}
  %Equal.sym(U32,(0 + extent : U32),extent,Arithmetic.add_zero_left(extent)) :
    {Tasks.Span{0,(Grid.divided_up(_,U32.max(units,1)) - 0 : U32)} == Tasks.Span{0,Tasks.target_blocks(extent,units)} : Tasks.Span}
  %Equal.sym(U32,Grid.divided_up(extent,U32.max(units,1)),Grid.divided_up(extent,units),up_same) :
    {Tasks.Span{0,(_ - 0 : U32)} == Tasks.Span{0,Tasks.target_blocks(extent,units)} : Tasks.Span}
  %Equal.sym(U32,(Grid.divided_up(extent,units) - 0 : U32),Grid.divided_up(extent,units),Arithmetic.subtract_zero(Grid.divided_up(extent,units))) :
    {Tasks.Span{0,_} == Tasks.Span{0,Tasks.target_blocks(extent,units)} : Tasks.Span}
  {==}

# The block span of one axis read whole is its whole block span.
def whole_head(lower: Bool,equal: Bool,+extent: U32,+rest: List<&2,Tasks.Span>,+more: List<&2,U32>,+grid: Grid.Grid,+prefix: U32,+units: U32,+axis: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +next: {U32.to_nat((axis + 1 : U32)) == 1n+U32.to_nat(axis) : Nat},+lower_is: {U32.is_lt((axis + 1 : U32),prefix) == lower : Bool},
  +equal_is: {U32.is_eq((axis + 1 : U32),prefix) == equal : Bool},+fits: WholeAxis(extent,Bool.pick(U32,lower,1,Bool.pick(U32,equal,units,extent)),cap),
  +later: {Tasks.block_spans(rest,prefix,units,(axis + 1 : U32)) == Tasks.whole_blocks(more,grid,prefix,(axis + 1 : U32)) : List<&2,Tasks.Span>}) ->
  {Tasks.block_spans(Con{Tasks.Span{0,extent},rest},prefix,units,axis) ==
    Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,Bool.pick(U32,lower,1,Bool.pick(U32,equal,units,extent)))},
      Tasks.whole_blocks(more,grid,prefix,(axis + 1 : U32))},Nil{}) : List<&2,Tasks.Span>}:
  match lower:
    case True{}:
      Held.Both{+positive,Held.Both{+room,+divisor_room}} = fits
      +total = U32.to_nat(extent)
      +two_cap = {divisor_room : {Nat.is_le(Nat.double(1n),U32.to_nat(cap)) == True{} : Bool}}
      +extent_cap = Order.transitive(total,Nat.add(total,1n),U32.to_nat(cap),Order.le_add(total,1n),room)
      +blocks = Tasks.whole_blocks(more,grid,prefix,(axis + 1 : U32))
      %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),True{},lower_is) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,extent},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
          Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(0 / U32.max(units,1) : U32),
            (Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
          Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,1)},blocks},Nil{}) : List<&2,Tasks.Span>}
      %Equal.sym(Bool,U32.is_lt(axis,prefix),True{},lower_then(axis,prefix,next,lower_is)) :
        {Con{Tasks.Span{0,extent},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))} ==
          Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(extent,1)},blocks},Nil{}) : List<&2,Tasks.Span>}
      Equal.trans(List<&2,Tasks.Span>,Con{Tasks.Span{0,extent},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},Con{Tasks.Span{0,extent},blocks},
        Con{Tasks.Span{0,Tasks.target_blocks(extent,1)},blocks},
        Equal.cong(List<&2,Tasks.Span>,List<&2,Tasks.Span>,tail => Con{Tasks.Span{0,extent},tail},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32)),blocks,later),
        Equal.cong(U32,List<&2,Tasks.Span>,word => Con{Tasks.Span{0,word},blocks},extent,Tasks.target_blocks(extent,1),
          Equal.sym(U32,Tasks.target_blocks(extent,1),extent,unit_blocks(extent,cap,cap_ok,two_cap,extent_cap))))
    case False{}:
      match equal:
        case True{}:
          Held.Both{+positive,Held.Both{+room,+divisor_room}} = fits
          +blocks = Tasks.whole_blocks(more,grid,prefix,(axis + 1 : U32))
          %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lower_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,extent},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
              Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(0 / U32.max(units,1) : U32),
                (Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
              Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,units)},blocks},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),True{},equal_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{(0 / U32.max(units,1) : U32),
                (Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}},Nil{}) ==
              Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,units)},blocks},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_lt(axis,prefix),True{},equal_then(axis,prefix,next,equal_is)) :
            {Con{Tasks.Span{(0 / U32.max(units,1) : U32),(Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}} ==
              Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(extent,units)},blocks},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(List<&2,Tasks.Span>,blocks,Nil{},Components.whole_end(more,grid,prefix,(axis + 1 : U32),lower_is)) :
            {Con{Tasks.Span{(0 / U32.max(units,1) : U32),(Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}} ==
              Con{Tasks.Span{0,Tasks.target_blocks(extent,units)},_} : List<&2,Tasks.Span>}
          Equal.cong(Tasks.Span,List<&2,Tasks.Span>,span => Con{span,Nil{}},
            Tasks.Span{(0 / U32.max(units,1) : U32),(Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},
            Tasks.Span{0,Tasks.target_blocks(extent,units)},last_whole(extent,units,cap,cap_ok,divisor_room,positive,room))
        case False{}:
          +blocks = Tasks.whole_blocks(more,grid,prefix,(axis + 1 : U32))
          %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lower_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,extent},Tasks.block_spans(rest,prefix,units,(axis + 1 : U32))},
              Bool.pick(List<&2,Tasks.Span>,U32.is_eq((axis + 1 : U32),prefix),Con{Tasks.Span{(0 / U32.max(units,1) : U32),
                (Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}},Nil{})) ==
              Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,extent)},blocks},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),False{},equal_is) :
            {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{(0 / U32.max(units,1) : U32),
                (Grid.divided_up((0 + extent : U32),U32.max(units,1)) - 0 / U32.max(units,1) : U32)},Nil{}},Nil{}) ==
              Bool.pick(List<&2,Tasks.Span>,U32.is_lt(axis,prefix),Con{Tasks.Span{0,Tasks.target_blocks(extent,extent)},blocks},Nil{}) : List<&2,Tasks.Span>}
          %Equal.sym(Bool,U32.is_lt(axis,prefix),False{},neither_then(axis,prefix,next,lower_is,equal_is)) :
            {Nil{} == Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{0,Tasks.target_blocks(extent,extent)},blocks},Nil{}) : List<&2,Tasks.Span>}
          {==}

# The block spans of the whole axes are the whole blocks.
def whole_own(extents: List<&2,U32>,+count: U32,+prefix: U32,+outer: U32,+last: U32,+unit_cells: U32,+units: U32,+per: U32,+pieces: U32,+axis: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +fits: WholeFits(extents,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},axis,cap)) ->
  {Tasks.block_spans(Tasks.whole_spans(extents),prefix,units,axis) == Tasks.whole_blocks(extents,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},prefix,axis) :
    List<&2,Tasks.Span>}:
  match extents:
    case Con{+extent,+rest}:
      Held.Both{+head,+later} = fits
      +grid = {Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces} : Grid.Grid}
      +next = Gather.word_next(axis,cap,cap_ok,Order.room_index(U32.to_nat(axis),List.length(&2,U32,rest),U32.to_nat(cap),room))
      whole_head(U32.is_lt((axis + 1 : U32),prefix),U32.is_eq((axis + 1 : U32),prefix),extent,Tasks.whole_spans(rest),rest,grid,prefix,units,axis,cap,cap_ok,
        next,{==},{==},head,
        whole_own(rest,count,prefix,outer,last,unit_cells,units,per,pieces,(axis + 1 : U32),cap,cap_ok,
          Gather.range_rest(axis,List.length(&2,U32,rest),U32.to_nat(cap),next,room),later))
    case Nil{}:
      {==}

# The positions of a whole box are the cells.
def whole_total(extents: List<&2,U32>) -> {Components.span_total(Tasks.whole_spans(extents)) == ShapeLaws.elements(extents) : Nat}:
  match extents:
    case Nil{}:
      {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,inner => Nat.mul(U32.to_nat(extent),inner),Components.span_total(Tasks.whole_spans(rest)),ShapeLaws.elements(rest),whole_total(rest))

# An inseparable component reads the whole target, and its target pieces are
# all of them.
def whole_pieces(+cells: List<&2,U32>,+classes: List<&2,Tasks.Class>,+count: U32,+prefix: U32,+outer: U32,+last: U32,+unit_cells: U32,
  +units: U32,+per: U32,+pieces: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(0),List.length(&2,U32,cells)),U32.to_nat(cap)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +fits: WholeFits(cells,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},0,cap)) ->
  {Tasks.box_pieces(Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},cells,Tasks.whole_spans(cells)) ==
    Tasks.target_when(False{},classes,Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces},cells,prefix) : List<&2,U32>}:
  +grid = {Grid.Grid{count,prefix,outer,last,unit_cells,units,per,pieces} : Grid.Grid}
  +boxes = Tasks.whole_spans(cells)
  +total = whole_total(cells)
  +exact = Components.span_count_value(boxes,cap,cap_ok,
    {%Equal.sym(Nat,Components.span_total(boxes),ShapeLaws.elements(cells),total) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
      cells_room : {Nat.is_le(Components.span_total(boxes),U32.to_nat(cap)) == True{} : Bool}})
  +nonzero = Arithmetic.word_positive_nonzero(Tasks.span_count(boxes),
    {%Equal.sym(Nat,U32.to_nat(Tasks.span_count(boxes)),Components.span_total(boxes),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
      %Equal.sym(Nat,Components.span_total(boxes),ShapeLaws.elements(cells),total) : {Nat.is_lt(0n,_) == True{} : Bool}
      filled : {Nat.is_lt(0n,U32.to_nat(Tasks.span_count(boxes))) == True{} : Bool}})
  +spans = whole_own(cells,count,prefix,outer,last,unit_cells,units,per,pieces,0,cap,cap_ok,room,fits)
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(boxes)),False{},nonzero) :
    {Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,cells,Tasks.block_spans(boxes,prefix,units,0))) ==
      Tasks.pieces_in(grid,cells,Tasks.whole_blocks(cells,grid,prefix,0)) : List<&2,U32>}
  Equal.cong(List<&2,Tasks.Span>,List<&2,U32>,listed => Tasks.pieces_in(grid,cells,listed),Tasks.block_spans(boxes,prefix,units,0),
    Tasks.whole_blocks(cells,grid,prefix,0),spans)
