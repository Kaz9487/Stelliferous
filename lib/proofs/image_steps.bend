# Steps over positions grouped in images of a fixed size, from a position
# to an end: one step for each image the positions touch, each ending at the
# next image or at the end.
import Base
import ./nat_order.bend as Order
import ./nat_extrema.bend as NatExtrema
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./binary_radix.bend as Radix
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./u32_division.bend as U32Division
import ./tensor_cursor_refinement.bend as Cursor

# The calls of the positions [position, end) of images of plane positions.
def calls_count(less: Bool,+plane: Nat,+position: Nat,+end: Nat) -> Nat:
  match less:
    case True{}:
      Nat.add(Nat.sub(Nat.div(Nat.sub(end,1n),plane),Nat.div(position,plane)),1n)
    case False{}:
      0n

def calls_with(less: Bool,+limits: Machine.Limits,+plane: U32,+position: U32,+end: U32,+positive: {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool},
  +plane_room: {Nat.is_le(U32.to_nat(plane),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +evidence: {Nat.is_lt(U32.to_nat(position),U32.to_nat(end)) == less : Bool}) ->
  {U32.to_nat(Bool.pick(U32,less,((end - 1) / plane - position / plane + 1 : U32),0)) == calls_count(less,U32.to_nat(plane),U32.to_nat(position),U32.to_nat(end)) : Nat}:
  match less:
    case True{}:
      +cap = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +half = Machine.half_word(limits)
      +divisor = Limits.double_bound(limits,U32.to_nat(plane),plane_room)
      +pl = U32.to_nat(plane)
      +p = U32.to_nat(position)
      +e = U32.to_nat(end)
      +before = evidence
      +e_positive = Order.le_lt_trans(0n,p,e,Order.zero_le(p),before)
      +one_le = Order.strict_follow(0n,e,e_positive)
      +e_last = Order.positive_has_last(e,e_positive)
      +p_le = Division.lt_succ_le(p,Nat.sub(e,1n),{%Equal.sym(Nat,1n+Nat.sub(e,1n),e,e_last) : {Nat.is_lt(p,_) == True{} : Bool} before : {Nat.is_lt(p,1n+Nat.sub(e,1n)) == True{} : Bool}})
      +last_word = (end - 1 : U32)
      +last_exact = Arithmetic.difference_exact(cap,fits,end,1,e,1n,{==},{==},one_le,end_room)
      +x = Nat.div(Nat.sub(e,1n),pl)
      +d = Nat.div(p,pl)
      +x_exact = {%last_exact : {U32.to_nat((last_word / plane : U32)) == Nat.div(_,pl) : Nat}
        U32Division.divide(last_word,plane,half,Limits.half_word_fits(limits),divisor,positive) : {U32.to_nat((last_word / plane : U32)) == x : Nat}}
      +d_exact = U32Division.divide(position,plane,half,Limits.half_word_fits(limits),divisor,positive)
      +ordered = DivisionOrder.quotient_le(p,Nat.sub(e,1n),pl,positive,p_le)
      +x_le = Order.transitive(x,Nat.mul(x,pl),Nat.sub(e,1n),RowMajor.product_covers(x,pl,positive),DivisionOrder.product_below(Nat.sub(e,1n),pl,positive))
      +x_room = Order.transitive(x,Nat.sub(e,1n),U32.to_nat(cap),x_le,Order.transitive(Nat.sub(e,1n),e,U32.to_nat(cap),Order.sub_le(e,1n),end_room))
      +difference = Arithmetic.difference_exact(cap,fits,(last_word / plane : U32),(position / plane : U32),x,d,x_exact,d_exact,ordered,x_room)
      +e_eq = Equal.trans(Nat,Nat.add(Nat.sub(e,1n),1n),Nat.add(1n,Nat.sub(e,1n)),e,Order.commutative(Nat.sub(e,1n),1n),e_last)
      +sum_room = Order.transitive(Nat.add(Nat.sub(x,d),1n),Nat.add(Nat.sub(e,1n),1n),U32.to_nat(cap),
        Order.add_le(Nat.sub(x,d),Nat.sub(e,1n),1n,Order.transitive(Nat.sub(x,d),x,Nat.sub(e,1n),Order.sub_le(x,d),x_le)),
        {%Equal.sym(Nat,Nat.add(Nat.sub(e,1n),1n),e,e_eq) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} end_room : {Nat.is_le(Nat.add(Nat.sub(e,1n),1n),U32.to_nat(cap)) == True{} : Bool}})
      Arithmetic.sum_exact(cap,fits,((last_word / plane : U32) - (position / plane : U32) : U32),1,Nat.sub(x,d),1n,difference,{==},sum_room)
    case False{}:
      {==}

# A pick below its right side picks its left side.
def minimum_below(choice: Bool,+left: Nat,+right: Nat,+below: {Nat.is_lt(Bool.pick(Nat,choice,left,right),right) == True{} : Bool}) ->
  {Bool.pick(Nat,choice,left,right) == left : Nat}:
  match choice:
    case True{}:
      {==}
    case False{}:
      Empty.absurd({right == left : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(right,right),True{},
        Equal.sym(Bool,Nat.is_lt(right,right),False{},Cursor.less_reflexive(right)),below)))

# One past a subtracted amount leaves one less.
def subtract_successor(+value: Nat,+amount: Nat,+ordered: {Nat.is_le(1n+amount,value) == True{} : Bool}) ->
  {Nat.sub(value,amount) == 1n+Nat.sub(value,1n+amount) : Nat}:
  match value amount:
    case 0n amount:
      Empty.absurd({Nat.sub(0n,amount) == 1n+Nat.sub(0n,1n+amount) : Nat},WordEncoding.false_true(ordered))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.sub(1n+rest,0n),1n+rest,Order.subtract_zero(1n+rest)) : {_ == 1n+Nat.sub(1n+rest,1n) : Nat}
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {1n+rest == 1n+_ : Nat}
      {==}
    case 1n+rest 1n+smaller:
      subtract_successor(rest,smaller,ordered)

# The positions of an image from a position up to the next image.
def next_image(+plane: Nat,+position: Nat,+positive: {Nat.is_lt(0n,plane) == True{} : Bool}) ->
  {Nat.add(position,Nat.sub(plane,Nat.mod(position,plane))) == Nat.mul(1n+Nat.div(position,plane),plane) : Nat}:
  +base = Nat.mul(Nat.div(position,plane),plane)
  +rest = Nat.mod(position,plane)
  %Equal.sym(Nat,position,Nat.add(base,rest),DivisionOrder.quotient_exact(position,plane,positive)) :
    {Nat.add(_,Nat.sub(plane,rest)) == Nat.mul(1n+Nat.div(position,plane),plane) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(base,rest),Nat.sub(plane,rest)),Nat.add(base,Nat.add(rest,Nat.sub(plane,rest))),Addition.associative(base,rest,Nat.sub(plane,rest))) :
    {_ == Nat.mul(1n+Nat.div(position,plane),plane) : Nat}
  %Equal.sym(Nat,Nat.add(rest,Nat.sub(plane,rest)),plane,Order.restore_shift(rest,plane,Bounds.strict_le(rest,plane,DivisionOrder.remainder_small(position,plane,positive)))) :
    {Nat.add(base,_) == Nat.mul(1n+Nat.div(position,plane),plane) : Nat}
  Order.commutative(base,plane)

# A call leaves one call less: it ends at the next image or at the end.
def calls_step(less: Bool,+plane: Nat,+position: Nat,+end: Nat,+count: Nat,+positive: {Nat.is_lt(0n,plane) == True{} : Bool},
  +before: {Nat.is_lt(position,end) == True{} : Bool},+count_exact: {count == Nat.min(Nat.sub(plane,Nat.mod(position,plane)),Nat.sub(end,position)) : Nat},
  +next_le: {Nat.is_le(Nat.add(position,count),end) == True{} : Bool},+evidence: {Nat.is_lt(Nat.add(position,count),end) == less : Bool}) ->
  {calls_count(True{},plane,position,end) == 1n+calls_count(less,plane,Nat.add(position,count),end) : Nat}:
  match less:
    case True{}:
      +gap = Nat.sub(plane,Nat.mod(position,plane))
      +left = Nat.sub(end,position)
      +image = Nat.div(position,plane)
      +x = Nat.div(Nat.sub(end,1n),plane)
      +e_positive = Order.le_lt_trans(0n,position,end,Order.zero_le(position),before)
      +e_last = Order.positive_has_last(end,e_positive)
      +next_exact = next_image(plane,position,positive)
      +count_gap = {%Equal.sym(Nat,count,Nat.min(gap,left),count_exact) : {Nat.is_le(_,gap) == True{} : Bool} NatExtrema.minimum_left(gap,left) : {Nat.is_le(count,gap) == True{} : Bool}}
      +whole = Order.restore_shift(position,end,Bounds.strict_le(position,end,before))
      +count_left = Order.cancel_offset_lt(position,count,left,
        {%Equal.sym(Nat,Nat.add(position,left),end,whole) : {Nat.is_lt(Nat.add(position,count),_) == True{} : Bool} evidence : {Nat.is_lt(Nat.add(position,count),Nat.add(position,left)) == True{} : Bool}})
      +count_is_gap = Equal.trans(Nat,count,Nat.min(gap,left),gap,count_exact,
        Equal.trans(Nat,Nat.min(gap,left),Bool.pick(Nat,Nat.is_lt(gap,left),gap,left),gap,NatExtrema.minimum_choice(gap,left),
          minimum_below(Nat.is_lt(gap,left),gap,left,
            {%NatExtrema.minimum_choice(gap,left) : {Nat.is_lt(_,left) == True{} : Bool}
              %count_exact : {Nat.is_lt(_,left) == True{} : Bool}
              count_left : {Nat.is_lt(Bool.pick(Nat,Nat.is_lt(gap,left),gap,left),left) == True{} : Bool}})))
      +next_start = Equal.trans(Nat,Nat.add(position,count),Nat.add(position,gap),Nat.mul(1n+image,plane),
        Equal.cong(Nat,Nat,value => Nat.add(position,value),count,gap,count_is_gap),next_exact)
      +next_image_exact = {%Equal.sym(Nat,Nat.add(position,count),Nat.mul(1n+image,plane),next_start) : {Nat.div(_,plane) == 1n+image : Nat}
        %Addition.zero_right(Nat.mul(1n+image,plane)) : {Nat.div(_,plane) == 1n+image : Nat}
        Radix.quotient(1n+image,0n,plane,positive,positive) : {Nat.div(Nat.add(position,count),plane) == 1n+image : Nat}}
      +next_last = Division.lt_succ_le(Nat.add(position,count),Nat.sub(end,1n),
        {%Equal.sym(Nat,1n+Nat.sub(end,1n),end,e_last) : {Nat.is_lt(Nat.add(position,count),_) == True{} : Bool} evidence :
          {Nat.is_lt(Nat.add(position,count),1n+Nat.sub(end,1n)) == True{} : Bool}})
      +image_le = {%next_image_exact : {Nat.is_le(_,x) == True{} : Bool}
        DivisionOrder.quotient_le(Nat.add(position,count),Nat.sub(end,1n),plane,positive,next_last) : {Nat.is_le(1n+image,x) == True{} : Bool}}
      %Equal.sym(Nat,Nat.div(Nat.add(position,count),plane),1n+image,next_image_exact) : {calls_count(True{},plane,position,end) == 1n+Nat.add(Nat.sub(x,_),1n) : Nat}
      %Equal.sym(Nat,Nat.sub(x,image),1n+Nat.sub(x,1n+image),subtract_successor(x,image,image_le)) : {Nat.add(_,1n) == 1n+Nat.add(Nat.sub(x,1n+image),1n) : Nat}
      {==}
    case False{}:
      +gap = Nat.sub(plane,Nat.mod(position,plane))
      +left = Nat.sub(end,position)
      +image = Nat.div(position,plane)
      +x = Nat.div(Nat.sub(end,1n),plane)
      +e_positive = Order.le_lt_trans(0n,position,end,Order.zero_le(position),before)
      +e_last = Order.positive_has_last(end,e_positive)
      +next_exact = next_image(plane,position,positive)
      +count_gap = {%Equal.sym(Nat,count,Nat.min(gap,left),count_exact) : {Nat.is_le(_,gap) == True{} : Bool} NatExtrema.minimum_left(gap,left) : {Nat.is_le(count,gap) == True{} : Bool}}
      +end_le = Order.not_less(Nat.add(position,count),end,evidence)
      +end_next = Order.transitive(end,Nat.add(position,count),Nat.mul(1n+image,plane),end_le,
        {%next_exact : {Nat.is_le(Nat.add(position,count),_) == True{} : Bool} Order.offset_le(position,count,gap,count_gap) :
          {Nat.is_le(Nat.add(position,count),Nat.mul(1n+image,plane)) == True{} : Bool}})
      +last_below = Bounds.nat_trans(Nat.sub(end,1n),end,Nat.mul(1n+image,plane),
        {%e_last : {Nat.is_lt(Nat.sub(end,1n),_) == True{} : Bool}
          Order.lt_succ(Nat.sub(end,1n),Nat.sub(end,1n),Order.reflexive(Nat.sub(end,1n))) : {Nat.is_lt(Nat.sub(end,1n),end) == True{} : Bool}},end_next)
      +x_le = Division.lt_succ_le(x,image,DivisionOrder.quotient_below(Nat.sub(end,1n),1n+image,plane,positive,last_below))
      %Equal.sym(Nat,Nat.sub(x,image),0n,Order.subtract_below(x,image,x_le)) : {Nat.add(_,1n) == 1n+0n : Nat}
      {==}

# No calls left: the position is the end.
def none_with(less: Bool,+plane: Nat,+position: Nat,+end: Nat,+evidence: {Nat.is_lt(position,end) == less : Bool},
  +ordered: {Nat.is_le(position,end) == True{} : Bool},+none: {calls_count(less,plane,position,end) == 0n : Nat}) -> {position == end : Nat}:
  match less:
    case True{}:
      +rest = Nat.sub(Nat.div(Nat.sub(end,1n),plane),Nat.div(position,plane))
      Empty.absurd({position == end : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},
        Equal.cong(Nat,Bool,value => Nat.is_lt(0n,value),1n+rest,0n,
          Equal.trans(Nat,1n+rest,Nat.add(rest,1n),0n,Equal.sym(Nat,Nat.add(rest,1n),Nat.add(1n,rest),Order.commutative(rest,1n)),none)))))
    case False{}:
      Order.antisymmetric(position,end,ordered,Order.not_less(position,end,evidence))

# Calls left: the position lies before the end.
def some_with(less: Bool,+plane: Nat,+position: Nat,+end: Nat,+rest: Nat,+evidence: {Nat.is_lt(position,end) == less : Bool},
  +some: {calls_count(less,plane,position,end) == 1n+rest : Nat}) -> {Nat.is_lt(position,end) == True{} : Bool}:
  match less:
    case True{}:
      evidence
    case False{}:
      Empty.absurd({Nat.is_lt(position,end) == True{} : Bool},WordEncoding.false_true(Equal.cong(Nat,Bool,value => Nat.is_lt(0n,value),0n,1n+rest,some)))

# ---- The facts of the calls ----
