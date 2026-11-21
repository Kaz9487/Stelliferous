# Where a block of a relane reads: the first source cell of a block, summed
# over the target cell axes in their order, is the sum over the logical axes
# in theirs, and so is the number of the source digits.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_shape_refinement.bend as ShapeLaws
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ../tensor_tasks.bend as Tasks
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division
import ./tensor_tile_coordinates.bend as Coordinates
import ../kernel_gemm.bend as KernelGemm
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_grid_pieces.bend as GridLaws

# ---- Digits ----

# The axes from index on, remaining of them.
def every_axis(remaining: Nat,+index: Nat) -> List<&2,Tiled.CellAxis>:
  match remaining:
    case 0n: []
    case 1n+rest: Tiled.OfAxis{index} <> every_axis(rest,1n+index)

def removed_at(digits: List<&2,Nat>,axis: Nat) -> List<&2,Nat>:
  match digits axis:
    case Nil{} axis: []
    case Con{digit,rest} 0n: rest
    case Con{+digit,rest} 1n+next: digit <> removed_at(rest,next)

# The axes from index on never reach an axis before index.
def never_hit(remaining: Nat,+index: Nat,+skip: Nat,+below: {Nat.is_lt(skip,index) == True{} : Bool}) ->
  {Tiled.axes_from(remaining,index,skip) == every_axis(remaining,index) : List<&2,Tiled.CellAxis>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +differ = Equal.trans(Bool,Nat.is_eq(index,skip),Nat.is_eq(skip,index),False{},Order.equal_symmetric(index,skip),Order.less_not_equal(skip,index,below))
      +later = never_hit(rest,1n+index,skip,Order.lt_succ(skip,index,Bounds.strict_le(skip,index,below)))
      %Equal.sym(Bool,Nat.is_eq(index,skip),False{},differ) :
        {Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)) ==
          every_axis(1n+rest,index) : List<&2,Tiled.CellAxis>}
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rest,1n+index,skip),every_axis(rest,1n+index),later) :
        {Tiled.OfAxis{index} <> _ == Tiled.OfAxis{index} <> every_axis(rest,1n+index) : List<&2,Tiled.CellAxis>}
      {==}

# ---- Sums over axes ----

# Digit digit of an axis times its source step: the source lane axis in
# blocks of eight.
def blocked_term(+blocked: Tiled.Blocked,+digit: Nat) -> U32:
  match blocked:
    case Tiled.Blocked{Tiled.Lanes{},+source,lanes}: (U32.from_nat(Nat.div(digit,8n)) * source : U32)
    case Tiled.Blocked{Tiled.Shared{},+source,lanes}: (U32.from_nat(digit) * source : U32)
    case Tiled.Blocked{Tiled.Groups{},+source,lanes}: (U32.from_nat(Nat.mul(digit,8n)) * source : U32)

def blocked_sum(digits: List<&2,Nat>,blocked: List<&2,Tiled.Blocked>) -> U32:
  match digits blocked:
    case Con{+digit,more} Con{+head,rest}: (blocked_term(head,digit) + blocked_sum(more,rest) : U32)
    case digits blocked: 0

def term_zero(+blocked: Tiled.Blocked) -> {blocked_term(blocked,0n) == 0 : U32}:
  match blocked:
    case Tiled.Blocked{Tiled.Lanes{},+source,+lanes}: Arithmetic.multiply_zero_left(source)
    case Tiled.Blocked{Tiled.Shared{},+source,+lanes}: Arithmetic.multiply_zero_left(source)
    case Tiled.Blocked{Tiled.Groups{},+source,+lanes}: Arithmetic.multiply_zero_left(source)

def empty_sum(digits: List<&2,Nat>) -> {blocked_sum(digits,[]) == 0 : U32}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+more}: {==}

# The term of an axis the axes skip, when they skip one.
def skipped_term(hit: Bool,+blocked: Tiled.Blocked,+digit: Nat) -> U32:
  match hit:
    case True{}: blocked_term(blocked,digit)
    case False{}: 0

def skipped_zero(hit: Bool,+blocked: Tiled.Blocked) -> {skipped_term(hit,blocked,0n) == 0 : U32}:
  match hit:
    case True{}: term_zero(blocked)
    case False{}: {==}

# The axes but one, with its digit removed, and its term make the sum over
# every axis.
def skipped_sum(remaining: Nat,gap: Nat,+index: Nat,+digits: List<&2,Nat>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  {(blocked_sum(removed_at(digits,gap),Tiled.blocked_list(Tiled.axes_from(remaining,index,Nat.add(index,gap)),source,source_within,source_position,target,strides,shape,groups_axis)) +
    skipped_term(Nat.is_lt(gap,remaining),Tiled.blocked_axis(Tiled.OfAxis{Nat.add(index,gap)},source,source_within,source_position,target,strides,shape,groups_axis),
      BoxDigits.digit_at(digits,gap)) : U32) ==
    blocked_sum(digits,Tiled.blocked_list(every_axis(remaining,index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}:
  match remaining gap digits:
    case 0n +gap +digits:
      %Equal.sym(U32,blocked_sum(removed_at(digits,gap),[]),0,empty_sum(removed_at(digits,gap))) :
        {(_ + skipped_term(Nat.is_lt(gap,0n),Tiled.blocked_axis(Tiled.OfAxis{Nat.add(index,gap)},source,source_within,source_position,target,strides,shape,groups_axis),
          BoxDigits.digit_at(digits,gap)) : U32) == blocked_sum(digits,[]) : U32}
      %Equal.sym(U32,blocked_sum(digits,[]),0,empty_sum(digits)) :
        {(0 + skipped_term(Nat.is_lt(gap,0n),Tiled.blocked_axis(Tiled.OfAxis{Nat.add(index,gap)},source,source_within,source_position,target,strides,shape,groups_axis),
          BoxDigits.digit_at(digits,gap)) : U32) == _ : U32}
      %Equal.sym(Bool,Nat.is_lt(gap,0n),False{},Order.not_below(0n,gap,Order.zero_le(gap))) :
        {(0 + skipped_term(_,Tiled.blocked_axis(Tiled.OfAxis{Nat.add(index,gap)},source,source_within,source_position,target,strides,shape,groups_axis),
          BoxDigits.digit_at(digits,gap)) : U32) == 0 : U32}
      Arithmetic.add_zero(0)
    case 1n+ +rest +gap Nil{}:
      +skipped = Tiled.blocked_axis(Tiled.OfAxis{Nat.add(index,gap)},source,source_within,source_position,target,strides,shape,groups_axis)
      %Equal.sym(U32,skipped_term(Nat.is_lt(gap,1n+rest),skipped,0n),0,skipped_zero(Nat.is_lt(gap,1n+rest),skipped)) :
        {(0 + _ : U32) == 0 : U32}
      Arithmetic.add_zero(0)
    case 1n+ +rest 0n Con{+digit,+more}:
      +head = Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis)
      +later = never_hit(rest,1n+index,index,Order.lt_succ(index,index,Order.reflexive(index)))
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {(blocked_sum(removed_at(Con{digit,more},0n),Tiled.blocked_list(Tiled.axes_from(1n+rest,index,_),source,source_within,source_position,target,strides,shape,groups_axis)) +
          skipped_term(True{},Tiled.blocked_axis(Tiled.OfAxis{_},source,source_within,source_position,target,strides,shape,groups_axis),BoxDigits.digit_at(Con{digit,more},0n)) : U32) ==
          blocked_sum(Con{digit,more},Tiled.blocked_list(every_axis(1n+rest,index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}
      %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) :
        {(blocked_sum(removed_at(Con{digit,more},0n),Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,index),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,index)),
              source,source_within,source_position,target,strides,shape,groups_axis)) +
          blocked_term(head,BoxDigits.digit_at(Con{digit,more},0n)) : U32) ==
          blocked_sum(Con{digit,more},head <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rest,1n+index,index),every_axis(rest,1n+index),later) :
        {(blocked_sum(removed_at(Con{digit,more},0n),Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis)) +
          blocked_term(head,BoxDigits.digit_at(Con{digit,more},0n)) : U32) ==
          blocked_sum(Con{digit,more},head <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}
      Arithmetic.add_commutative(blocked_sum(more,Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)),
        blocked_term(head,digit))
    case 1n+ +rest 1n+ +next Con{+digit,+more}:
      +head = Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis)
      +skip = Nat.add(1n+index,next)
      +differ = Order.less_not_equal(index,Nat.add(index,1n+next),Order.below_sum(index,next))
      %Equal.sym(Nat,Nat.add(index,1n+next),skip,Addition.successor_right(index,next)) :
        {(blocked_sum(removed_at(Con{digit,more},1n+next),Tiled.blocked_list(Tiled.axes_from(1n+rest,index,_),source,source_within,source_position,target,strides,shape,groups_axis)) +
          skipped_term(Nat.is_lt(next,rest),Tiled.blocked_axis(Tiled.OfAxis{_},source,source_within,source_position,target,strides,shape,groups_axis),BoxDigits.digit_at(Con{digit,more},1n+next)) : U32) ==
          blocked_sum(Con{digit,more},head <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}
      %Equal.sym(Bool,Nat.is_eq(index,skip),False{},{%Addition.successor_right(index,next) : {Nat.is_eq(index,_) == False{} : Bool} differ : {Nat.is_eq(index,skip) == False{} : Bool}}) :
        {(blocked_sum(removed_at(Con{digit,more},1n+next),Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)),
              source,source_within,source_position,target,strides,shape,groups_axis)) +
          skipped_term(Nat.is_lt(next,rest),Tiled.blocked_axis(Tiled.OfAxis{skip},source,source_within,source_position,target,strides,shape,groups_axis),BoxDigits.digit_at(Con{digit,more},1n+next)) : U32) ==
          blocked_sum(Con{digit,more},head <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32}
      +later = Tiled.blocked_list(Tiled.axes_from(rest,1n+index,skip),source,source_within,source_position,target,strides,shape,groups_axis)
      +skipped = Tiled.blocked_axis(Tiled.OfAxis{skip},source,source_within,source_position,target,strides,shape,groups_axis)
      +inner = skipped_sum(rest,next,1n+index,more,source,source_within,source_position,target,strides,shape,groups_axis)
      Equal.trans(U32,((blocked_term(head,digit) + blocked_sum(removed_at(more,next),later) : U32) + skipped_term(Nat.is_lt(next,rest),skipped,BoxDigits.digit_at(more,next)) : U32),
        (blocked_term(head,digit) + (blocked_sum(removed_at(more,next),later) + skipped_term(Nat.is_lt(next,rest),skipped,BoxDigits.digit_at(more,next)) : U32) : U32),
        (blocked_term(head,digit) + blocked_sum(more,Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)) : U32),
        Arithmetic.add_associative(blocked_term(head,digit),blocked_sum(removed_at(more,next),later),skipped_term(Nat.is_lt(next,rest),skipped,BoxDigits.digit_at(more,next))),
        Equal.cong(U32,U32,sum => (blocked_term(head,digit) + sum : U32),
          (blocked_sum(removed_at(more,next),later) + skipped_term(Nat.is_lt(next,rest),skipped,BoxDigits.digit_at(more,next)) : U32),
          blocked_sum(more,Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis)),inner))

# ---- The number of digits as a sum ----

# Every digit times the stride of its axis, the first digit on axis axis.
def strided_at(digits: List<&2,Nat>,+strides: List<&2,U32>,+axis: Nat) -> U32:
  match digits:
    case Nil{}: 0
    case Con{+digit,more}: (U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) + strided_at(more,strides,1n+axis) : U32)

def strided_none(digits: List<&2,Nat>,+axis: Nat) -> {strided_at(digits,[],axis) == 0 : U32}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+more}:
      %Equal.sym(U32,strided_at(more,[],1n+axis),0,strided_none(more,1n+axis)) : {((U32.from_nat(digit) * 0 : U32) + _ : U32) == 0 : U32}
      %Equal.sym(U32,(U32.from_nat(digit) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(digit))) : {(_ + 0 : U32) == 0 : U32}
      Arithmetic.add_zero(0)

# The stride after the first.
def nth_next(+first: U32,+strides: List<&2,U32>,+axis: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(1n+axis,U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.nth(first <> strides,U32.from_nat(1n+axis)) == Tasks.nth(strides,U32.from_nat(axis)) : U32}:
  +word = U32.from_nat(1n+axis)
  +positive = {%Equal.sym(Nat,U32.to_nat(word),1n+axis,Bounds.roundtrip(1n+axis,cap,cap_ok,room)) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
    {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}}
  +step = Equal.trans(U32,word,U32.from_nat(Nat.add(axis,1n)),(U32.from_nat(axis) + 1 : U32),
    Equal.cong(Nat,U32,U32.from_nat,1n+axis,Nat.add(axis,1n),Equal.sym(Nat,Nat.add(axis,1n),1n+axis,Addition.successor_width(axis))),
    Equal.sym(U32,(U32.from_nat(axis) + 1 : U32),U32.from_nat(Nat.add(axis,1n)),WordArithmetic.native_add(axis,1n)))
  %Equal.sym(Bool,U32.is_zero(word),False{},Arithmetic.word_positive_nonzero(word,positive)) :
    {Bool.pick(U32,_,first,Tasks.nth(strides,(word - 1 : U32))) == Tasks.nth(strides,U32.from_nat(axis)) : U32}
  %Equal.sym(U32,word,(U32.from_nat(axis) + 1 : U32),step) : {Tasks.nth(strides,(_ - 1 : U32)) == Tasks.nth(strides,U32.from_nat(axis)) : U32}
  %Equal.sym(U32,((U32.from_nat(axis) + 1 : U32) - 1 : U32),U32.from_nat(axis),Subtraction.subtract_add(U32.from_nat(axis),1)) :
    {Tasks.nth(strides,_) == Tasks.nth(strides,U32.from_nat(axis)) : U32}
  {==}

# Digits from axis one on read the strides after the first.
def strided_shift(digits: List<&2,Nat>,+first: U32,+strides: List<&2,U32>,+axis: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(1n+axis,List.length(&2,Nat,digits)),U32.to_nat(cap)) == True{} : Bool}) ->
  {strided_at(digits,first <> strides,1n+axis) == strided_at(digits,strides,axis) : U32}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+more}:
      +here = nth_next(first,strides,axis,cap,cap_ok,Order.transitive(1n+axis,Nat.add(1n+axis,1n+List.length(&2,Nat,more)),U32.to_nat(cap),
        Order.le_add(1n+axis,1n+List.length(&2,Nat,more)),room))
      +later = strided_shift(more,first,strides,1n+axis,cap,cap_ok,
        {%Addition.successor_right(1n+axis,List.length(&2,Nat,more)) :
          {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(1n+axis,List.length(&2,Nat,more)),U32.to_nat(cap)) == True{} : Bool}})
      %Equal.sym(U32,Tasks.nth(first <> strides,U32.from_nat(1n+axis)),Tasks.nth(strides,U32.from_nat(axis)),here) :
        {((U32.from_nat(digit) * _ : U32) + strided_at(more,first <> strides,2n+axis) : U32) ==
          ((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) + strided_at(more,strides,1n+axis) : U32) : U32}
      %Equal.sym(U32,strided_at(more,first <> strides,2n+axis),strided_at(more,strides,1n+axis),later) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) + _ : U32) ==
          ((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) + strided_at(more,strides,1n+axis) : U32) : U32}
      {==}

# The number of digits is the sum of every digit times the stride of its
# axis.
def digits_strided(extents: List<&2,U32>,digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,Nat,digits),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.from_nat(ShapeLaws.digits_value(extents,digits)) == strided_at(digits,Tasks.strides_of(extents),0n) : U32}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      +inner = ShapeLaws.elements(rest)
      +value = ShapeLaws.digits_value(rest,more)
      +shift = strided_shift(more,Grid.product(rest),Tasks.strides_of(rest),0n,cap,cap_ok,room)
      +later = digits_strided(rest,more,cap,cap_ok,Order.transitive(List.length(&2,Nat,more),1n+List.length(&2,Nat,more),U32.to_nat(cap),
        Order.successor(List.length(&2,Nat,more),List.length(&2,Nat,more),Order.reflexive(List.length(&2,Nat,more))),room))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {U32.from_nat(ShapeLaws.digits_value(Con{extent,rest},Con{digit,more})) == strided_at(Con{digit,more},_,0n) : U32}
      %WordArithmetic.native_add(Nat.mul(digit,inner),value) :
        {_ == ((U32.from_nat(digit) * Grid.product(rest) : U32) + strided_at(more,Grid.product(rest) <> Tasks.strides_of(rest),1n) : U32) : U32}
      %Multiplication.native_encoded_multiply(digit,inner) :
        {(_ + U32.from_nat(value) : U32) == ((U32.from_nat(digit) * Grid.product(rest) : U32) + strided_at(more,Grid.product(rest) <> Tasks.strides_of(rest),1n) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(inner),Grid.product(rest),BoxDigits.product_from(rest)) :
        {((U32.from_nat(digit) * _ : U32) + U32.from_nat(value) : U32) ==
          ((U32.from_nat(digit) * Grid.product(rest) : U32) + strided_at(more,Grid.product(rest) <> Tasks.strides_of(rest),1n) : U32) : U32}
      %Equal.sym(U32,U32.from_nat(value),strided_at(more,Tasks.strides_of(rest),0n),later) :
        {((U32.from_nat(digit) * Grid.product(rest) : U32) + _ : U32) ==
          ((U32.from_nat(digit) * Grid.product(rest) : U32) + strided_at(more,Grid.product(rest) <> Tasks.strides_of(rest),1n) : U32) : U32}
      %Equal.sym(U32,strided_at(more,Grid.product(rest) <> Tasks.strides_of(rest),1n),strided_at(more,Tasks.strides_of(rest),0n),shift) :
        {((U32.from_nat(digit) * Grid.product(rest) : U32) + strided_at(more,Tasks.strides_of(rest),0n) : U32) ==
          ((U32.from_nat(digit) * Grid.product(rest) : U32) + _ : U32) : U32}
      {==}
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Nil{}: {==}
    case Nil{} Con{+digit,+more}:
      %Equal.sym(U32,strided_at(more,[],1n),0,strided_none(more,1n)) : {0 == ((U32.from_nat(digit) * 0 : U32) + _ : U32) : U32}
      %Equal.sym(U32,(U32.from_nat(digit) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(digit))) : {0 == (_ + 0 : U32) : U32}
      Equal.sym(U32,(0 + 0 : U32),0,Arithmetic.add_zero(0))

# ---- The source steps of the logical axes ----

def and_false(+value: Bool) -> {Bool.and(value,False{}) == False{} : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

def pick_next(outer: Bool,+index: Nat) -> {1n+Bool.pick(Nat,outer,1n+index,index) == Bool.pick(Nat,outer,2n+index,1n+index) : Nat}:
  match outer:
    case True{}: {==}
    case False{}: {==}

# A logical axis before the source lane axis reads the source cell axis of
# its own index, after the groups when they come first.
def before_tag(+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +outer: Bool,+same_outer: {Bool.and(source_within,Grid.is_outermost(source_position)) == outer : Bool},+below: {Nat.is_lt(index,source) == True{} : Bool}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}:
  +differ = Order.less_not_equal(index,source,below)
  +not_after = Order.not_below(index,source,Bounds.strict_le(index,source,below))
  %Equal.sym(Bool,Nat.is_eq(index,source),False{},differ) :
    {Bool.pick(Tiled.Blocked,Bool.and(source_within,_),Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Bool.and(source_within,False{}),False{},and_false(source_within)) :
    {Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Nat.is_lt(source,index),False{},not_after) :
    {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,Bool.and(source_within,Grid.is_outermost(source_position)),
        1n+Bool.pick(Nat,Bool.and(source_within,_),Nat.sub(index,1n),index),Bool.pick(Nat,Bool.and(source_within,_),Nat.sub(index,1n),index)))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Bool.and(source_within,False{}),False{},and_false(source_within)) :
    {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,Bool.and(source_within,Grid.is_outermost(source_position)),
        1n+Bool.pick(Nat,_,Nat.sub(index,1n),index),Bool.pick(Nat,_,Nat.sub(index,1n),index)))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Bool.and(source_within,Grid.is_outermost(source_position)),outer,same_outer) : {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,_,1n+index,index))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1} : Tiled.Blocked}
  {==}

# A logical axis after the source lane axis reads the source cell axis of
# the index before it, after the groups when they come first.
def after_tag(+previous: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +outer: Bool,+same_outer: {Grid.is_outermost(source_position) == outer : Bool},+within: {source_within == True{} : Bool},
  +past: {Nat.is_lt(source,1n+previous) == True{} : Bool}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{1n+previous},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}:
  +differ = Equal.trans(Bool,Nat.is_eq(1n+previous,source),Nat.is_eq(source,1n+previous),False{},Order.equal_symmetric(1n+previous,source),
    Order.less_not_equal(source,1n+previous,past))
  %Equal.sym(Bool,source_within,True{},within) :
    {Tiled.blocked_axis(Tiled.OfAxis{1n+previous},source,_,source_position,target,strides,shape,groups_axis) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Nat.is_eq(1n+previous,source),False{},differ) :
    {Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,True{},source_position,1n+previous)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Nat.is_lt(source,1n+previous),True{},past) :
    {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,Grid.is_outermost(source_position),
        1n+Bool.pick(Nat,_,Nat.sub(1n+previous,1n),1n+previous),Bool.pick(Nat,_,Nat.sub(1n+previous,1n),1n+previous)))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}
  %Equal.sym(Nat,Nat.sub(previous,0n),previous,Order.subtract_zero(previous)) :
    {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,Grid.is_outermost(source_position),1n+_,_))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}
  %Equal.sym(Bool,Grid.is_outermost(source_position),outer,same_outer) : {Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,_,1n+previous,previous))),1} ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1} : Tiled.Blocked}
  {==}

# The logical axes after the source lane axis read the source cell axes in
# order.
def after_strided(digits: List<&2,Nat>,+previous: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,
  +shape: List<&2,U32>,+groups_axis: U32,+outer: Bool,+same_outer: {Grid.is_outermost(source_position) == outer : Bool},+within: {source_within == True{} : Bool},
  +past: {Nat.is_lt(source,1n+previous) == True{} : Bool}) ->
  {blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),1n+previous),source,source_within,source_position,target,strides,shape,groups_axis)) ==
    strided_at(digits,strides,Bool.pick(Nat,outer,1n+previous,previous)) : U32}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+more}:
      +here = after_tag(previous,source,source_within,source_position,target,strides,shape,groups_axis,outer,same_outer,within,past)
      +later = after_strided(more,1n+previous,source,source_within,source_position,target,strides,shape,groups_axis,outer,same_outer,within,
        Order.lt_succ(source,1n+previous,Bounds.strict_le(source,1n+previous,past)))
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{1n+previous},source,source_within,source_position,target,strides,shape,groups_axis),
          Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))),1},here) :
        {(blocked_term(_,digit) +
            blocked_sum(more,Tiled.blocked_list(every_axis(List.length(&2,Nat,more),2n+previous),source,source_within,source_position,target,strides,shape,groups_axis)) : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+previous,previous)) : U32}
      %Equal.sym(U32,blocked_sum(more,Tiled.blocked_list(every_axis(List.length(&2,Nat,more),2n+previous),source,source_within,source_position,target,strides,shape,groups_axis)),
          strided_at(more,strides,Bool.pick(Nat,outer,2n+previous,1n+previous)),later) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))) : U32) + _ : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+previous,previous)) : U32}
      %pick_next(outer,previous) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+previous,previous))) : U32) + strided_at(more,strides,_) : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+previous,previous)) : U32}
      {==}

# The logical axes but the source lane axis read the source cell axes in
# order.
def before_strided(digits: List<&2,Nat>,gap: Nat,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,
  +shape: List<&2,U32>,+groups_axis: U32,+outer: Bool,+same_outer: {Bool.and(source_within,Grid.is_outermost(source_position)) == outer : Bool},
  +within: {source_within == True{} : Bool},+skip: {Nat.add(index,gap) == source : Nat}) ->
  {blocked_sum(digits,Tiled.blocked_list(Tiled.axes_from(1n+List.length(&2,Nat,digits),index,source),source,source_within,source_position,target,strides,shape,groups_axis)) ==
    strided_at(digits,strides,Bool.pick(Nat,outer,1n+index,index)) : U32}:
  match digits gap:
    case Nil{} +gap: {==}
    case Con{+digit,+more} 0n:
      +same = Equal.trans(Nat,index,Nat.add(index,0n),source,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),skip)
      +hit = {%same : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,source) == True{} : Bool}}
      +past = {%same : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) : {Nat.is_lt(source,1n+index) == True{} : Bool}}
      +outer_now = {%within : {Bool.and(_,Grid.is_outermost(source_position)) == outer : Bool} same_outer : {Bool.and(True{},Grid.is_outermost(source_position)) == outer : Bool}}
      +later = never_hit(1n+List.length(&2,Nat,more),1n+index,source,past)
      %Equal.sym(Bool,Nat.is_eq(index,source),True{},hit) :
        {blocked_sum(Con{digit,more},Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source),
            Tiled.OfAxis{index} <> Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source)),source,source_within,source_position,target,strides,shape,groups_axis)) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source),every_axis(1n+List.length(&2,Nat,more),1n+index),later) :
        {blocked_sum(Con{digit,more},Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis)) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      after_strided(Con{digit,more},index,source,source_within,source_position,target,strides,shape,groups_axis,outer,outer_now,within,past)
    case Con{+digit,+more} 1n+ +next:
      +below = {%skip : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,next) : {Nat.is_lt(index,source) == True{} : Bool}}
      +differ = Order.less_not_equal(index,source,below)
      +here = before_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,outer,same_outer,below)
      +later = before_strided(more,next,1n+index,source,source_within,source_position,target,strides,shape,groups_axis,outer,same_outer,within,
        Equal.trans(Nat,Nat.add(1n+index,next),Nat.add(index,1n+next),source,Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)),skip))
      %Equal.sym(Bool,Nat.is_eq(index,source),False{},differ) :
        {blocked_sum(Con{digit,more},Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source),
            Tiled.OfAxis{index} <> Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source)),source,source_within,source_position,target,strides,shape,groups_axis)) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis),
          Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))),1},here) :
        {(blocked_term(_,digit) +
            blocked_sum(more,Tiled.blocked_list(Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source),source,source_within,source_position,target,strides,shape,groups_axis)) : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      %Equal.sym(U32,blocked_sum(more,Tiled.blocked_list(Tiled.axes_from(1n+List.length(&2,Nat,more),1n+index,source),source,source_within,source_position,target,strides,shape,groups_axis)),
          strided_at(more,strides,Bool.pick(Nat,outer,2n+index,1n+index)),later) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))) : U32) + _ : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      %pick_next(outer,index) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Bool.pick(Nat,outer,1n+index,index))) : U32) + strided_at(more,strides,_) : U32) ==
          strided_at(Con{digit,more},strides,Bool.pick(Nat,outer,1n+index,index)) : U32}
      {==}

# ---- The number of the source digits ----

def length_last(digits: List<&2,Nat>,+digit: Nat) -> {List.length(&2,Nat,List.append(&2,Nat,digits,[digit])) == 1n+List.length(&2,Nat,digits) : Nat}:
  match digits:
    case Nil{}: {==}
    case Con{+head,+rest}: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Nat,List.append(&2,Nat,rest,[digit])),1n+List.length(&2,Nat,rest),length_last(rest,digit))

def length_removed(digits: List<&2,Nat>,axis: Nat,+below: {Nat.is_lt(axis,List.length(&2,Nat,digits)) == True{} : Bool}) ->
  {1n+List.length(&2,Nat,removed_at(digits,axis)) == List.length(&2,Nat,digits) : Nat}:
  match digits axis:
    case Nil{} +axis: Empty.absurd({1n+List.length(&2,Nat,removed_at(Nil{},axis)) == 0n : Nat},Order.below_zero(axis,below))
    case Con{+digit,+rest} 0n: {==}
    case Con{+digit,+rest} 1n+ +next:
      Equal.cong(Nat,Nat,length => 1n+length,1n+List.length(&2,Nat,removed_at(rest,next)),List.length(&2,Nat,rest),length_removed(rest,next,below))

def removed_beyond(digits: List<&2,Nat>,axis: Nat,+past: {Nat.is_le(List.length(&2,Nat,digits),axis) == True{} : Bool}) -> {removed_at(digits,axis) == digits : List<&2,Nat>}:
  match digits axis:
    case Nil{} +axis: {==}
    case Con{+digit,+rest} 0n: Empty.absurd({rest == Con{digit,rest} : List<&2,Nat>},WordEncoding.false_true(past))
    case Con{+digit,+rest} 1n+ +next: Equal.cong(List<&2,Nat>,List<&2,Nat>,list => digit <> list,removed_at(rest,next),rest,removed_beyond(rest,next,past))

# The last index of a shape of one more axis.
def less_one(+value: Nat) -> {(U32.from_nat(1n+value) - 1 : U32) == U32.from_nat(value) : U32}:
  %Addition.successor_width(value) : {(U32.from_nat(_) - 1 : U32) == U32.from_nat(value) : U32}
  %WordArithmetic.native_add(value,1n) : {(_ - 1 : U32) == U32.from_nat(value) : U32}
  Subtraction.subtract_add(U32.from_nat(value),1)

# A digit after others reads the stride after theirs.
def strided_append(digits: List<&2,Nat>,+digit: Nat,+strides: List<&2,U32>,+axis: Nat) ->
  {strided_at(List.append(&2,Nat,digits,[digit]),strides,axis) ==
    (strided_at(digits,strides,axis) + U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Nat.add(axis,List.length(&2,Nat,digits)))) : U32) : U32}:
  match digits:
    case Nil{}:
      %Equal.sym(Nat,Nat.add(axis,0n),axis,Addition.zero_right(axis)) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) + 0 : U32) ==
          (0 + U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(_)) : U32) : U32}
      %Equal.sym(U32,((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) + 0 : U32),(U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32),Arithmetic.add_zero((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32))) :
        {_ == (0 + U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32) : U32}
      Equal.sym(U32,(0 + U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32),(U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32),
        Arithmetic.add_zero_left((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(axis)) : U32)))
    case Con{+head,+rest}:
      +first = (U32.from_nat(head) * Tasks.nth(strides,U32.from_nat(axis)) : U32)
      +last = (U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(Nat.add(1n+axis,List.length(&2,Nat,rest)))) : U32)
      %Equal.sym(Nat,Nat.add(axis,1n+List.length(&2,Nat,rest)),Nat.add(1n+axis,List.length(&2,Nat,rest)),Addition.successor_right(axis,List.length(&2,Nat,rest))) :
        {(first + strided_at(List.append(&2,Nat,rest,[digit]),strides,1n+axis) : U32) ==
          ((first + strided_at(rest,strides,1n+axis) : U32) + U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(_)) : U32) : U32}
      %Equal.sym(U32,strided_at(List.append(&2,Nat,rest,[digit]),strides,1n+axis),(strided_at(rest,strides,1n+axis) + last : U32),
          strided_append(rest,digit,strides,1n+axis)) :
        {(first + _ : U32) == ((first + strided_at(rest,strides,1n+axis) : U32) + last : U32) : U32}
      Equal.sym(U32,((first + strided_at(rest,strides,1n+axis) : U32) + last : U32),(first + (strided_at(rest,strides,1n+axis) + last : U32) : U32),
        Arithmetic.add_associative(first,strided_at(rest,strides,1n+axis),last))

# Lanes on the unit axis: every logical axis reads its own source axis.
def plain_strided(digits: List<&2,Nat>,+index: Nat,+source: Nat,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  {blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),index),source,False{},source_position,target,strides,shape,groups_axis)) ==
    strided_at(digits,strides,index) : U32}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+more}:
      %Equal.sym(U32,blocked_sum(more,Tiled.blocked_list(every_axis(List.length(&2,Nat,more),1n+index),source,False{},source_position,target,strides,shape,groups_axis)),
          strided_at(more,strides,1n+index),plain_strided(more,1n+index,source,source_position,target,strides,shape,groups_axis)) :
        {((U32.from_nat(digit) * Tasks.nth(strides,U32.from_nat(index)) : U32) + _ : U32) == strided_at(Con{digit,more},strides,index) : U32}
      {==}

# The source lane axis reads the groups.
def lane_tag(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +within: {source_within == True{} : Bool}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{source},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)} : Tiled.Blocked}:
  %Equal.sym(Bool,source_within,True{},within) :
    {Tiled.blocked_axis(Tiled.OfAxis{source},source,_,source_position,target,strides,shape,groups_axis) ==
      Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)} : Tiled.Blocked}
  %Equal.sym(Bool,Nat.is_eq(source,source),True{},Order.equal_reflexive(source)) :
    {Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,True{},source_position,source)),1}) ==
      Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)} : Tiled.Blocked}
  {==}

# The number of the source digits of logical digits: the sum of every
# logical digit times the source step of its axis, the source lane axis in
# groups of eight.
def source_placed(within: Bool,position: Grid.Position,+digits: List<&2,Nat>,+source: Nat,+target: Nat,+cells: List<&2,U32>,+shape: List<&2,U32>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within_value: {Nat.is_lt(source,List.length(&2,Nat,digits)) == within : Bool},
  +cells_length: {List.length(&2,U32,cells) == List.length(&2,Nat,digits) : Nat},+room: {Nat.is_le(List.length(&2,Nat,digits),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.from_nat(ShapeLaws.digits_value(cells,Coordinates.placed_digits(within,position,removed_at(digits,source),Nat.div(BoxDigits.digit_at(digits,source),8n)))) ==
    blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,within,position,target,Tasks.strides_of(cells),shape,
      Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)))) : U32}:
  match within position:
    case True{} Grid.Innermost{}:
      +strides = Tasks.strides_of(cells)
      +others = removed_at(digits,source)
      +group = Nat.div(BoxDigits.digit_at(digits,source),8n)
      +groups_axis = (U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)
      +counted = length_removed(digits,source,within_value)
      +placed = List.append(&2,Nat,others,[group])
      +placed_room = {%Equal.sym(Nat,List.length(&2,Nat,placed),1n+List.length(&2,Nat,others),length_last(others,group)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        {%Equal.sym(Nat,1n+List.length(&2,Nat,others),List.length(&2,Nat,digits),counted) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
          {Nat.is_le(1n+List.length(&2,Nat,others),U32.to_nat(cap)) == True{} : Bool}} : {Nat.is_le(List.length(&2,Nat,placed),U32.to_nat(cap)) == True{} : Bool}}
      +last_axis = Equal.trans(U32,groups_axis,(U32.from_nat(1n+List.length(&2,Nat,others)) - 1 : U32),U32.from_nat(List.length(&2,Nat,others)),
        Equal.cong(Nat,U32,length => (U32.from_nat(length) - 1 : U32),List.length(&2,U32,cells),1n+List.length(&2,Nat,others),
          Equal.trans(Nat,List.length(&2,U32,cells),List.length(&2,Nat,digits),1n+List.length(&2,Nat,others),cells_length,Equal.sym(Nat,1n+List.length(&2,Nat,others),List.length(&2,Nat,digits),counted))),
        less_one(List.length(&2,Nat,others)))
      +skipped = skipped_sum(List.length(&2,Nat,digits),source,0n,digits,source,True{},Grid.Innermost{},target,strides,shape,groups_axis)
      +tags = before_strided(others,source,0n,source,True{},Grid.Innermost{},target,strides,shape,groups_axis,False{},{==},{==},{==})
      %Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(cells,placed)),strided_at(placed,strides,0n),digits_strided(cells,placed,cap,cap_ok,placed_room)) : {_ == blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,True{},Grid.Innermost{},
        target,strides,shape,groups_axis)) : U32}
      %Equal.sym(U32,strided_at(placed,strides,0n),(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32),
          strided_append(others,group,strides,0n)) :
        {_ == blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,True{},Grid.Innermost{},target,strides,shape,groups_axis)) : U32}
      %skipped : {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) == _ : U32}
      %Equal.sym(Bool,Nat.is_lt(source,List.length(&2,Nat,digits)),True{},within_value) : {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) ==
        (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(List.length(&2,Nat,digits),0n,source),source,True{},Grid.Innermost{},target,strides,shape,groups_axis)) +
          skipped_term(_,Tiled.blocked_axis(Tiled.OfAxis{source},source,True{},Grid.Innermost{},target,strides,shape,groups_axis),BoxDigits.digit_at(digits,source)) : U32) : U32}
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{source},source,True{},Grid.Innermost{},target,strides,shape,groups_axis),Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},lane_tag(source,True{},Grid.Innermost{},target,strides,shape,groups_axis,{==})) :
        {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) ==
          (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(List.length(&2,Nat,digits),0n,source),source,True{},Grid.Innermost{},target,strides,shape,groups_axis)) +
            blocked_term(_,BoxDigits.digit_at(digits,source)) : U32) : U32}
      %counted : {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) ==
          (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(_,0n,source),source,True{},Grid.Innermost{},target,strides,shape,groups_axis)) +
            (U32.from_nat(group) * Tasks.nth(strides,groups_axis) : U32) : U32) : U32}
      %Equal.sym(U32,blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(1n+List.length(&2,Nat,others),0n,source),source,True{},Grid.Innermost{},target,strides,shape,groups_axis)),strided_at(others,strides,0n),tags) : {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) ==
          (_ + (U32.from_nat(group) * Tasks.nth(strides,groups_axis) : U32) : U32) : U32}
      %Equal.sym(U32,groups_axis,U32.from_nat(List.length(&2,Nat,others)),last_axis) : {(strided_at(others,strides,0n) + U32.from_nat(group) * Tasks.nth(strides,U32.from_nat(List.length(&2,Nat,others))) : U32) ==
          (strided_at(others,strides,0n) + (U32.from_nat(group) * Tasks.nth(strides,_) : U32) : U32) : U32}
      {==}
    case True{} Grid.Outermost{}:
      +strides = Tasks.strides_of(cells)
      +others = removed_at(digits,source)
      +group = Nat.div(BoxDigits.digit_at(digits,source),8n)
      +counted = length_removed(digits,source,within_value)
      +placed_room = {%Equal.sym(Nat,1n+List.length(&2,Nat,others),List.length(&2,Nat,digits),counted) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(1n+List.length(&2,Nat,others),U32.to_nat(cap)) == True{} : Bool}}
      +skipped = skipped_sum(List.length(&2,Nat,digits),source,0n,digits,source,True{},Grid.Outermost{},target,strides,shape,0)
      +tags = before_strided(others,source,0n,source,True{},Grid.Outermost{},target,strides,shape,0,True{},{==},{==},{==})
      %Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(cells,group <> others)),strided_at(group <> others,strides,0n),digits_strided(cells,group <> others,cap,cap_ok,placed_room)) :
        {_ == blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,True{},Grid.Outermost{},target,strides,shape,0)) : U32}
      %skipped : {((U32.from_nat(group) * Tasks.nth(strides,0) : U32) + strided_at(others,strides,1n) : U32) == _ : U32}
      %Equal.sym(Bool,Nat.is_lt(source,List.length(&2,Nat,digits)),True{},within_value) : {((U32.from_nat(group) * Tasks.nth(strides,0) : U32) + strided_at(others,strides,1n) : U32) ==
        (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(List.length(&2,Nat,digits),0n,source),source,True{},Grid.Outermost{},target,strides,shape,0)) +
          skipped_term(_,Tiled.blocked_axis(Tiled.OfAxis{source},source,True{},Grid.Outermost{},target,strides,shape,0),BoxDigits.digit_at(digits,source)) : U32) : U32}
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{source},source,True{},Grid.Outermost{},target,strides,shape,0),Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,0),Grid.lanes_at(shape,source)},lane_tag(source,True{},Grid.Outermost{},target,strides,shape,0,{==})) :
        {((U32.from_nat(group) * Tasks.nth(strides,0) : U32) + strided_at(others,strides,1n) : U32) ==
          (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(List.length(&2,Nat,digits),0n,source),source,True{},Grid.Outermost{},target,strides,shape,0)) +
            blocked_term(_,BoxDigits.digit_at(digits,source)) : U32) : U32}
      %counted : {((U32.from_nat(group) * Tasks.nth(strides,0) : U32) + strided_at(others,strides,1n) : U32) ==
          (blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(_,0n,source),source,True{},Grid.Outermost{},target,strides,shape,0)) +
            (U32.from_nat(group) * Tasks.nth(strides,0) : U32) : U32) : U32}
      %Equal.sym(U32,blocked_sum(others,Tiled.blocked_list(Tiled.axes_from(1n+List.length(&2,Nat,others),0n,source),source,True{},Grid.Outermost{},target,strides,shape,0)),strided_at(others,strides,1n),tags) : {((U32.from_nat(group) * Tasks.nth(strides,0) : U32) + strided_at(others,strides,1n) : U32) == (_ + (U32.from_nat(group) * Tasks.nth(strides,0) : U32) : U32) : U32}
      Arithmetic.add_commutative((U32.from_nat(group) * Tasks.nth(strides,0) : U32),strided_at(others,strides,1n))
    case False{} +position:
      +strides = Tasks.strides_of(cells)
      +others = removed_at(digits,source)
      +group = Nat.div(BoxDigits.digit_at(digits,source),8n)
      +groups_axis = Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32))
      +same = removed_beyond(digits,source,Order.not_less(source,List.length(&2,Nat,digits),within_value))
      %Equal.sym(List<&2,Nat>,removed_at(digits,source),digits,same) :
        {U32.from_nat(ShapeLaws.digits_value(cells,_)) ==
          blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,False{},position,target,strides,shape,groups_axis)) : U32}
      %Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(cells,digits)),strided_at(digits,strides,0n),digits_strided(cells,digits,cap,cap_ok,room)) :
        {_ == blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,False{},position,target,strides,shape,groups_axis)) : U32}
      Equal.sym(U32,blocked_sum(digits,Tiled.blocked_list(every_axis(List.length(&2,Nat,digits),0n),source,False{},position,target,strides,shape,groups_axis)),
        strided_at(digits,strides,0n),plain_strided(digits,0n,source,position,target,strides,shape,groups_axis))

# ---- The first source cell of a block ----

def blocked_of(axes: List<&2,Tiled.Ranged>) -> List<&2,Tiled.Blocked>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{+blocked,low,extent,count,inner,target},rest}: blocked <> blocked_of(rest)

# The target digit of a block on an axis of its role, at index index of the
# blocks of the axis: the source lane axis at lane lane of the block.
def role_digit(role: Tiled.Role,+low: U32,+index: U32,+lane: Nat) -> Nat:
  match role:
    case Tiled.Shared{}: Nat.add(U32.to_nat(low),U32.to_nat(index))
    case Tiled.Groups{}: Nat.add(U32.to_nat(low),U32.to_nat(index))
    case Tiled.Lanes{}: Nat.add(Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index)),8n),lane)

# The target digits of block index at lane lane.
def block_digits(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat) -> List<&2,Nat>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,extent,+count,+inner,target},rest}:
      role_digit(role,low,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane) <> block_digits(rest,index,lane)

def source_of(+block: Tiled.Block) -> U32:
  Tiled.Block{+source,target,stride,low,high,reads,step} = block
  source

# What an axis of each role adds to the first source cell of a block.
def role_word(role: Tiled.Role,+low: U32,+index: U32,+source: U32) -> U32:
  match role:
    case Tiled.Shared{}: ((low + index : U32) * source : U32)
    case Tiled.Groups{}: ((low + index : U32) * 8 * source : U32)
    case Tiled.Lanes{}: ((low / 8 + index : U32) * source : U32)

def axis_source(role: Tiled.Role,+first: U32,+start: U32,+stride: U32,+block_low: U32,+block_high: U32,+reads: U32,+step: U32,+index: U32,+low: U32,+extent: U32,
  +source: U32,+lanes: U32,+target: U32) ->
  {source_of(Tiled.block_axis(role,Tiled.Block{first,start,stride,block_low,block_high,reads,step},index,low,extent,source,lanes,target)) ==
    (first + role_word(role,low,index,source) : U32) : U32}:
  match role:
    case Tiled.Shared{}: {==}
    case Tiled.Groups{}: {==}
    case Tiled.Lanes{}: {==}

def lane_group(+group: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.div(Nat.add(Nat.mul(group,8n),lane),8n) == group : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(group,8n),lane),8n),(group,lane),
    Division.quotient_remainder(group,lane,8n,{==},small))

# The term of the target digit of an axis is what the axis adds.
def role_term(role: Tiled.Role,+low: U32,+index: U32,+lane: Nat,+source: U32,+lanes: U32,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {blocked_term(Tiled.Blocked{role,source,lanes},role_digit(role,low,index,lane)) == role_word(role,low,index,source) : U32}:
  match role:
    case Tiled.Shared{}:
      %Arithmetic.native_add(low,index) : {(_ * source : U32) == ((low + index : U32) * source : U32) : U32}
      {==}
    case Tiled.Groups{}:
      %Multiplication.native_encoded_multiply(Nat.add(U32.to_nat(low),U32.to_nat(index)),8n) : {(_ * source : U32) == ((low + index : U32) * 8 * source : U32) : U32}
      %Arithmetic.native_add(low,index) : {((_ * 8 : U32) * source : U32) == ((low + index : U32) * 8 * source : U32) : U32}
      {==}
    case Tiled.Lanes{}:
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index)),8n),lane),8n),Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index)),
          lane_group(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index)),lane,small)) :
        {(U32.from_nat(_) * source : U32) == ((low / 8 + index : U32) * source : U32) : U32}
      %Arithmetic.native_add((low / 8 : U32),index) : {(_ * source : U32) == ((low / 8 + index : U32) * source : U32) : U32}
      {==}

# The first source cell of a block is the sum of its target digits times
# their source steps.
def block_source(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {source_of(Tiled.block_at(axes,index,block)) == (source_of(block) + blocked_sum(block_digits(axes,index,lane),blocked_of(axes)) : U32) : U32}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: Equal.sym(U32,(first + 0 : U32),first,Arithmetic.add_zero(first))
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(role,Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +later = blocked_sum(block_digits(rest,index,lane),blocked_of(rest))
      +word = role_word(role,low,at,source)
      +term = blocked_term(Tiled.Blocked{role,source,lanes},role_digit(role,low,at,lane))
      Equal.trans(U32,source_of(Tiled.block_at(rest,index,next)),(source_of(next) + later : U32),(first + (term + later : U32) : U32),
        block_source(rest,next,index,lane,small),
        Equal.trans(U32,(source_of(next) + later : U32),((first + word : U32) + later : U32),(first + (term + later : U32) : U32),
          Equal.cong(U32,U32,value => (value + later : U32),source_of(next),(first + word : U32),
            axis_source(role,first,start,stride,block_low,block_high,reads,step,at,low,extent,source,lanes,target)),
          Equal.trans(U32,((first + word : U32) + later : U32),(first + (word + later : U32) : U32),(first + (term + later : U32) : U32),
            Arithmetic.add_associative(first,word,later),
            Equal.cong(U32,U32,value => (first + (value + later : U32) : U32),word,term,Equal.sym(U32,term,word,role_term(role,low,at,lane,source,lanes,small))))))

# ---- The target groups of a block ----

# The digits but the last, and the last.
def digits_front(digits: List<&2,Nat>) -> List<&2,Nat>:
  match digits:
    case Nil{}: []
    case Con{+digit,Nil{}}: []
    case Con{+digit,rest}: digit <> digits_front(rest)

def digits_back(digits: List<&2,Nat>) -> Nat:
  match digits:
    case Nil{}: 0n
    case Con{+digit,Nil{}}: digit
    case Con{digit,rest}: digits_back(rest)

def front_back(+digits: List<&2,Nat>,+digit: Nat) ->
  {List.append(&2,Nat,digits_front(digit <> digits),[digits_back(digit <> digits)]) == digit <> digits : List<&2,Nat>}:
  match digits:
    case Nil{}: {==}
    case Con{+next,+rest}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => digit <> list,List.append(&2,Nat,digits_front(next <> rest),[digits_back(next <> rest)]),next <> rest,front_back(rest,next))

def blocked_append(axes: List<&2,Tiled.CellAxis>,+axis: Tiled.CellAxis,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  {Tiled.blocked_list(List.append(&2,Tiled.CellAxis,axes,[axis]),source,source_within,source_position,target,strides,shape,groups_axis) ==
    List.append(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),
      [Tiled.blocked_axis(axis,source,source_within,source_position,target,strides,shape,groups_axis)]) : List<&2,Tiled.Blocked>}:
  match axes:
    case Nil{}: {==}
    case Con{+head,+rest}:
      Equal.cong(List<&2,Tiled.Blocked>,List<&2,Tiled.Blocked>,list => Tiled.blocked_axis(head,source,source_within,source_position,target,strides,shape,groups_axis) <> list,
        Tiled.blocked_list(List.append(&2,Tiled.CellAxis,rest,[axis]),source,source_within,source_position,target,strides,shape,groups_axis),
        List.append(&2,Tiled.Blocked,Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis),
          [Tiled.blocked_axis(axis,source,source_within,source_position,target,strides,shape,groups_axis)]),
        blocked_append(rest,axis,source,source_within,source_position,target,strides,shape,groups_axis))

# The sum over axes with one more at the end: the last digit takes its
# term.
def sum_last(digits: List<&2,Nat>,blocked: List<&2,Tiled.Blocked>,+last: Tiled.Blocked,
  +length: {List.length(&2,Nat,digits) == 1n+List.length(&2,Tiled.Blocked,blocked) : Nat}) ->
  {blocked_sum(digits,List.append(&2,Tiled.Blocked,blocked,[last])) ==
    (blocked_sum(digits_front(digits),blocked) + blocked_term(last,digits_back(digits)) : U32) : U32}:
  match digits blocked:
    case Nil{} +blocked: Empty.absurd({0 == (blocked_sum([],blocked) + blocked_term(last,0n) : U32) : U32},
      BoxDigits.zero_successor(List.length(&2,Tiled.Blocked,blocked),length))
    case Con{+digit,Nil{}} Nil{}:
      Equal.trans(U32,(blocked_term(last,digit) + 0 : U32),blocked_term(last,digit),(0 + blocked_term(last,digit) : U32),
        Arithmetic.add_zero(blocked_term(last,digit)),Equal.sym(U32,(0 + blocked_term(last,digit) : U32),blocked_term(last,digit),Arithmetic.add_zero_left(blocked_term(last,digit))))
    case Con{+digit,Con{+next,+rest}} Nil{}:
      Empty.absurd({blocked_sum(digit <> next <> rest,[last]) == (blocked_sum(digits_front(digit <> next <> rest),[]) + blocked_term(last,digits_back(digit <> next <> rest)) : U32) : U32},
        BoxDigits.zero_successor(List.length(&2,Nat,rest),Equal.sym(Nat,1n+List.length(&2,Nat,rest),0n,Equal.cong(Nat,Nat,number => BoxDigits.before(number),2n+List.length(&2,Nat,rest),1n,length))))
    case Con{+digit,Nil{}} Con{+head,+tail}:
      Empty.absurd({blocked_sum([digit],head <> List.append(&2,Tiled.Blocked,tail,[last])) == (blocked_sum([],head <> tail) + blocked_term(last,digit) : U32) : U32},
        BoxDigits.zero_successor(List.length(&2,Tiled.Blocked,tail),Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n,2n+List.length(&2,Tiled.Blocked,tail),length)))
    case Con{+digit,Con{+next,+rest}} Con{+head,+tail}:
      +term = blocked_term(head,digit)
      +inner = sum_last(next <> rest,tail,last,Equal.cong(Nat,Nat,number => BoxDigits.before(number),2n+List.length(&2,Nat,rest),2n+List.length(&2,Tiled.Blocked,tail),length))
      Equal.trans(U32,(term + blocked_sum(next <> rest,List.append(&2,Tiled.Blocked,tail,[last])) : U32),
        (term + (blocked_sum(digits_front(next <> rest),tail) + blocked_term(last,digits_back(next <> rest)) : U32) : U32),
        ((term + blocked_sum(digits_front(next <> rest),tail) : U32) + blocked_term(last,digits_back(next <> rest)) : U32),
        Equal.cong(U32,U32,sum => (term + sum : U32),blocked_sum(next <> rest,List.append(&2,Tiled.Blocked,tail,[last])),
          (blocked_sum(digits_front(next <> rest),tail) + blocked_term(last,digits_back(next <> rest)) : U32),inner),
        Equal.sym(U32,((term + blocked_sum(digits_front(next <> rest),tail) : U32) + blocked_term(last,digits_back(next <> rest)) : U32),
          (term + (blocked_sum(digits_front(next <> rest),tail) + blocked_term(last,digits_back(next <> rest)) : U32) : U32),
          Arithmetic.add_associative(term,blocked_sum(digits_front(next <> rest),tail),blocked_term(last,digits_back(next <> rest)))))

# The step a block reads its source cells with: the source step of the
# target groups, or the step before when it has none.
def role_step(role: Tiled.Role,+source: U32,+step: U32) -> U32:
  match role:
    case Tiled.Groups{}: source
    case Tiled.Shared{}: step
    case Tiled.Lanes{}: step

def blocked_step(blocked: List<&2,Tiled.Blocked>,+step: U32) -> U32:
  match blocked:
    case Nil{}: step
    case Con{Tiled.Blocked{role,+source,lanes},rest}: blocked_step(rest,role_step(role,source,step))

def step_of(+block: Tiled.Block) -> U32:
  Tiled.Block{source,target,stride,low,high,reads,+step} = block
  step

def axis_step(role: Tiled.Role,+first: U32,+start: U32,+stride: U32,+block_low: U32,+block_high: U32,+reads: U32,+step: U32,+index: U32,+low: U32,+extent: U32,
  +source: U32,+lanes: U32,+target: U32) ->
  {step_of(Tiled.block_axis(role,Tiled.Block{first,start,stride,block_low,block_high,reads,step},index,low,extent,source,lanes,target)) == role_step(role,source,step) : U32}:
  match role:
    case Tiled.Shared{}: {==}
    case Tiled.Groups{}: {==}
    case Tiled.Lanes{}: {==}

def block_step(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32) ->
  {step_of(Tiled.block_at(axes,index,block)) == blocked_step(blocked_of(axes),step_of(block)) : U32}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(role,Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      Equal.trans(U32,step_of(Tiled.block_at(rest,index,next)),blocked_step(blocked_of(rest),step_of(next)),blocked_step(blocked_of(rest),role_step(role,source,step)),
        block_step(rest,next,index),
        Equal.cong(U32,U32,value => blocked_step(blocked_of(rest),value),step_of(next),role_step(role,source,step),
          axis_step(role,first,start,stride,block_low,block_high,reads,step,at,low,extent,source,lanes,target)))

def blocked_step_append(blocked: List<&2,Tiled.Blocked>,+last: Tiled.Blocked,+step: U32) ->
  {blocked_step(List.append(&2,Tiled.Blocked,blocked,[last]),step) == blocked_step([last],blocked_step(blocked,step)) : U32}:
  match blocked:
    case Nil{}: {==}
    case Con{Tiled.Blocked{+role,+source,+lanes},+rest}: blocked_step_append(rest,last,role_step(role,source,step))

# Every source a list of blocked axes names steps at most half a word.
def SourcesIn(blocked: List<&2,Tiled.Blocked>,+word: U32) -> Data:
  match blocked:
    case Nil{}: Unit
    case Con{Tiled.Blocked{role,+source,lanes},rest}: KernelGemm.Both<{Nat.is_le(Nat.double(U32.to_nat(source)),U32.to_nat(word)) == True{} : Bool},SourcesIn(rest,word)>

# A blocked axis that is the source lane axis or shared names a stride.
def blocked_in(choice: Bool,+lane_source: U32,+lane_lanes: U32,+source: U32,+rest: List<&2,Tiled.Blocked>,+word: U32,
  +lane_in: {Nat.is_le(Nat.double(U32.to_nat(lane_source)),U32.to_nat(word)) == True{} : Bool},
  +source_in: {Nat.is_le(Nat.double(U32.to_nat(source)),U32.to_nat(word)) == True{} : Bool},+rest_in: SourcesIn(rest,word)) ->
  SourcesIn(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lane_lanes},Tiled.Blocked{Tiled.Shared{},source,1}) <> rest,word):
  match choice:
    case True{}: KernelGemm.Both{lane_in,rest_in}
    case False{}: KernelGemm.Both{source_in,rest_in}

# Every blocked axis names a source stride.
def listed_in(axes: List<&2,Tiled.CellAxis>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,
  +groups_axis: U32,+word: U32,+bounded: BoxDigits.StridesIn(strides,word)) ->
  SourcesIn(Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),word):
  match axes:
    case Nil{}: Unit{}
    case Con{Tiled.OfGroups{},+rest}:
      KernelGemm.Both{BoxDigits.nth_in(strides,Tiled.source_axis(source,source_within,source_position,target),word,bounded),
        listed_in(rest,source,source_within,source_position,target,strides,shape,groups_axis,word,bounded)}
    case Con{Tiled.OfAxis{+logical},+rest}:
      blocked_in(Bool.and(source_within,Nat.is_eq(logical,source)),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,logical)),Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis),word,
        BoxDigits.nth_in(strides,groups_axis,word,bounded),BoxDigits.nth_in(strides,Tiled.source_axis(source,source_within,source_position,logical),word,bounded),
        listed_in(rest,source,source_within,source_position,target,strides,shape,groups_axis,word,bounded))

# The ranged axes keep the sources of their blocked axes.
def ranged_in(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,targets: List<&2,U32>,+word: U32,+bounded: SourcesIn(blocked,word)) ->
  SourcesIn(blocked_of(Tiled.ranged_axes(blocked,lows,extents,targets)),word):
  match blocked:
    case Nil{}: Unit{}
    case Con{Tiled.Blocked{+role,+source,+lanes},+rest}:
      match lows:
        case Nil{}: Unit{}
        case Con{+low,+more_lows}:
          match extents:
            case Nil{}: Unit{}
            case Con{+extent,+more_extents}:
              match targets:
                case Nil{}: Unit{}
                case Con{+target,+more_targets}:
                  KernelGemm.Both{+here,+later} = bounded
                  KernelGemm.Both{here,ranged_in(rest,more_lows,more_extents,more_targets,word,later)}

def role_in(role: Tiled.Role,+source: U32,+step: U32,+word: U32,+source_in: {Nat.is_le(Nat.double(U32.to_nat(source)),U32.to_nat(word)) == True{} : Bool},
  +step_in: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(role_step(role,source,step))),U32.to_nat(word)) == True{} : Bool}:
  match role:
    case Tiled.Groups{}: source_in
    case Tiled.Shared{}: step_in
    case Tiled.Lanes{}: step_in

# A block reads with a step its blocked axes name, or the step it starts with.
def step_in(blocked: List<&2,Tiled.Blocked>,+step: U32,+word: U32,+bounded: SourcesIn(blocked,word),
  +step_ok: {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(word)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(blocked_step(blocked,step))),U32.to_nat(word)) == True{} : Bool}:
  match blocked:
    case Nil{}: step_ok
    case Con{Tiled.Blocked{+role,+source,+lanes},+rest}:
      KernelGemm.Both{+here,+later} = bounded
      step_in(rest,role_step(role,source,step),word,later,role_in(role,source,step,word,here,step_ok))

# An axis that is the source lane axis or shared keeps the step.
def shared_step(choice: Bool,+lane_source: U32,+lane_lanes: U32,+source: U32,+rest: List<&2,Tiled.Blocked>,+step: U32) ->
  {blocked_step(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lane_lanes},Tiled.Blocked{Tiled.Shared{},source,1}) <> rest,step) ==
    blocked_step(rest,step) : U32}:
  match choice:
    case True{}: {==}
    case False{}: {==}

def step_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+step: U32,
  +inner: {blocked_step(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),step) == step : U32}) ->
  {blocked_step(Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),source,source_within,source_position,target,strides,shape,
    groups_axis),step) == step : U32}:
  match choice:
    case True{}: inner
    case False{}:
      Equal.trans(U32,blocked_step(Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis) <>
          Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),step),
        blocked_step(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),step),step,
        shared_step(Bool.and(source_within,Nat.is_eq(index,source)),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
          Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),step),
        inner)

# The axes but one hold no target groups: they keep the step.
def axes_step(remaining: Nat,+index: Nat,+skip: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+step: U32) ->
  {blocked_step(Tiled.blocked_list(Tiled.axes_from(remaining,index,skip),source,source_within,source_position,target,strides,shape,groups_axis),step) == step : U32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      step_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,source,source_within,source_position,target,strides,shape,groups_axis,step,
        axes_step(rest,1n+index,skip,source,source_within,source_position,target,strides,shape,groups_axis,step))

# ---- The first source cell of a block over the logical axes ----

def blocked_count(axes: List<&2,Tiled.CellAxis>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,
  +shape: List<&2,U32>,+groups_axis: U32) ->
  {List.length(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis)) == List.length(&2,Tiled.CellAxis,axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{+head,+rest}:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tiled.Blocked,Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis)),
        List.length(&2,Tiled.CellAxis,rest),blocked_count(rest,source,source_within,source_position,target,strides,shape,groups_axis))

def every_count(remaining: Nat,+index: Nat) -> {List.length(&2,Tiled.CellAxis,every_axis(remaining,index)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tiled.CellAxis,every_axis(rest,1n+index)),rest,every_count(rest,1n+index))

def skipped_count(hit: Bool) -> Nat:
  match hit:
    case True{}: 1n
    case False{}: 0n

# The axes but one count one fewer when the one lies among them.
def axes_count(remaining: Nat,gap: Nat,+index: Nat) ->
  {Nat.add(List.length(&2,Tiled.CellAxis,Tiled.axes_from(remaining,index,Nat.add(index,gap))),skipped_count(Nat.is_lt(gap,remaining))) == remaining : Nat}:
  match remaining gap:
    case 0n +gap:
      %Equal.sym(Bool,Nat.is_lt(gap,0n),False{},Order.not_below(0n,gap,Order.zero_le(gap))) : {Nat.add(0n,skipped_count(_)) == 0n : Nat}
      {==}
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {Nat.add(List.length(&2,Tiled.CellAxis,Tiled.axes_from(1n+rest,index,_)),1n) == 1n+rest : Nat}
      %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) :
        {Nat.add(List.length(&2,Tiled.CellAxis,Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,index),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,index))),1n) ==
          1n+rest : Nat}
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rest,1n+index,index),every_axis(rest,1n+index),never_hit(rest,1n+index,index,Order.lt_succ(index,index,Order.reflexive(index)))) :
        {Nat.add(List.length(&2,Tiled.CellAxis,_),1n) == 1n+rest : Nat}
      %Equal.sym(Nat,List.length(&2,Tiled.CellAxis,every_axis(rest,1n+index)),rest,every_count(rest,1n+index)) : {Nat.add(_,1n) == 1n+rest : Nat}
      Addition.successor_width(rest)
    case 1n+ +rest 1n+ +next:
      +skip = Nat.add(1n+index,next)
      +differ = Order.less_not_equal(index,Nat.add(index,1n+next),Order.below_sum(index,next))
      %Equal.sym(Nat,Nat.add(index,1n+next),skip,Addition.successor_right(index,next)) :
        {Nat.add(List.length(&2,Tiled.CellAxis,Tiled.axes_from(1n+rest,index,_)),skipped_count(Nat.is_lt(next,rest))) == 1n+rest : Nat}
      %Equal.sym(Bool,Nat.is_eq(index,skip),False{},{%Addition.successor_right(index,next) : {Nat.is_eq(index,_) == False{} : Bool} differ : {Nat.is_eq(index,skip) == False{} : Bool}}) :
        {Nat.add(List.length(&2,Tiled.CellAxis,Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip))),
          skipped_count(Nat.is_lt(next,rest))) == 1n+rest : Nat}
      Equal.cong(Nat,Nat,count => 1n+count,Nat.add(List.length(&2,Tiled.CellAxis,Tiled.axes_from(rest,1n+index,skip)),skipped_count(Nat.is_lt(next,rest))),rest,
        axes_count(rest,next,1n+index))

def removed_inserted(axis: Nat,digits: List<&2,Nat>,+value: Nat,+room: {Nat.is_le(axis,List.length(&2,Nat,digits)) == True{} : Bool}) ->
  {removed_at(Coordinates.inserted_digit(axis,digits,value),axis) == digits : List<&2,Nat>}:
  match axis digits:
    case 0n +digits: {==}
    case 1n+ +next Nil{}: Empty.absurd({removed_at(Coordinates.inserted_digit(1n+next,Nil{},value),1n+next) == Nil{} : List<&2,Nat>},WordEncoding.false_true(room))
    case 1n+ +next Con{+digit,+rest}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => digit <> list,removed_at(Coordinates.inserted_digit(next,rest,value),next),rest,removed_inserted(next,rest,value,room))

def digit_inserted(axis: Nat,digits: List<&2,Nat>,+value: Nat,+room: {Nat.is_le(axis,List.length(&2,Nat,digits)) == True{} : Bool}) ->
  {BoxDigits.digit_at(Coordinates.inserted_digit(axis,digits,value),axis) == value : Nat}:
  match axis digits:
    case 0n +digits: {==}
    case 1n+ +next Nil{}: Empty.absurd({BoxDigits.digit_at(Coordinates.inserted_digit(1n+next,Nil{},value),1n+next) == value : Nat},WordEncoding.false_true(room))
    case 1n+ +next Con{+digit,+rest}: digit_inserted(next,rest,value,room)

def front_count(rest: List<&2,Nat>,+digit: Nat) -> {1n+List.length(&2,Nat,digits_front(digit <> rest)) == 1n+List.length(&2,Nat,rest) : Nat}:
  match rest:
    case Nil{}: {==}
    case Con{+next,+more}: Equal.cong(Nat,Nat,length => 1n+length,1n+List.length(&2,Nat,digits_front(next <> more)),1n+List.length(&2,Nat,more),front_count(more,next))

def digits_head(digits: List<&2,Nat>) -> Nat:
  match digits:
    case Nil{}: 0n
    case Con{+digit,rest}: digit

def digits_tail(digits: List<&2,Nat>) -> List<&2,Nat>:
  match digits:
    case Nil{}: []
    case Con{digit,+rest}: rest

# The logical digits of target digits at lane lane: the lane put back in its
# group at the target lane axis.
def target_logical(within: Bool,position: Grid.Position,+target: Nat,+digits: List<&2,Nat>,+lane: Nat) -> List<&2,Nat>:
  match within position:
    case False{} position: digits
    case True{} Grid.Innermost{}: Coordinates.inserted_digit(target,digits_front(digits),Nat.add(lane,Nat.mul(digits_back(digits),8n)))
    case True{} Grid.Outermost{}: Coordinates.inserted_digit(target,digits_tail(digits),Nat.add(lane,Nat.mul(digits_head(digits),8n)))

# The source step of the target groups taken group times eight and lane
# more times.
def groups_merge(+sum: U32,+group: Nat,+lane: Nat,+step: U32) ->
  {((sum + U32.from_nat(Nat.mul(group,8n)) * step : U32) + U32.from_nat(lane) * step : U32) == (sum + U32.from_nat(Nat.add(lane,Nat.mul(group,8n))) * step : U32) : U32}:
  +eight = U32.from_nat(Nat.mul(group,8n))
  %Equal.sym(U32,((sum + eight * step : U32) + U32.from_nat(lane) * step : U32),(sum + (eight * step + U32.from_nat(lane) * step : U32) : U32),
      Arithmetic.add_associative(sum,(eight * step : U32),(U32.from_nat(lane) * step : U32))) :
    {_ == (sum + U32.from_nat(Nat.add(lane,Nat.mul(group,8n))) * step : U32) : U32}
  %Arithmetic.multiply_additive(eight,U32.from_nat(lane),step) :
    {(sum + _ : U32) == (sum + U32.from_nat(Nat.add(lane,Nat.mul(group,8n))) * step : U32) : U32}
  %Equal.sym(U32,(eight + U32.from_nat(lane) : U32),U32.from_nat(Nat.add(Nat.mul(group,8n),lane)),WordArithmetic.native_add(Nat.mul(group,8n),lane)) :
    {(sum + _ * step : U32) == (sum + U32.from_nat(Nat.add(lane,Nat.mul(group,8n))) * step : U32) : U32}
  %Equal.sym(Nat,Nat.add(Nat.mul(group,8n),lane),Nat.add(lane,Nat.mul(group,8n)),Order.commutative(Nat.mul(group,8n),lane)) :
    {(sum + U32.from_nat(_) * step : U32) == (sum + U32.from_nat(Nat.add(lane,Nat.mul(group,8n))) * step : U32) : U32}
  {==}

# The axis of the target lanes read through the target groups is shared.
def target_tag(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{target},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target)),1} : Tiled.Blocked}:
  %Equal.sym(Bool,Bool.and(source_within,Nat.is_eq(target,source)),False{},apart) :
    {Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target)),1} : Tiled.Blocked}
  {==}

# The first source cell of a block and lane lane of the target groups: the
# sum over the logical axes of the logical digits at the lane.
def logical_sum(within: Bool,position: Grid.Position,digits: List<&2,Nat>,+rank: Nat,+lane: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+within_value: {Nat.is_lt(target,rank) == within : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},+length: {List.length(&2,Nat,digits) == rank : Nat}) ->
  {(blocked_sum(digits,Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,strides,shape,groups_axis)) +
      U32.from_nat(lane) * blocked_step(Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,strides,shape,
        groups_axis),0) : U32) ==
    blocked_sum(target_logical(within,position,target,digits,lane),Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,
      groups_axis)) : U32}:
  match within position digits:
    case False{} +position +digits:
      +axes = Tiled.blocked_list(Tiled.axes_from(rank,0n,rank),source,source_within,source_position,target,strides,shape,groups_axis)
      +every = Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,groups_axis)
      +skipped = Tiled.blocked_axis(Tiled.OfAxis{rank},source,source_within,source_position,target,strides,shape,groups_axis)
      +kept = axes_step(rank,0n,rank,source,source_within,source_position,target,strides,shape,groups_axis,0)
      +same = removed_beyond(digits,rank,{%Equal.sym(Nat,List.length(&2,Nat,digits),rank,length) : {Nat.is_le(_,rank) == True{} : Bool} Order.reflexive(rank) :
        {Nat.is_le(List.length(&2,Nat,digits),rank) == True{} : Bool}})
      +all = skipped_sum(rank,rank,0n,digits,source,source_within,source_position,target,strides,shape,groups_axis)
      %Equal.sym(U32,blocked_step(axes,0),0,kept) : {(blocked_sum(digits,axes) + U32.from_nat(lane) * _ : U32) == blocked_sum(digits,every) : U32}
      %Equal.sym(U32,(U32.from_nat(lane) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(lane))) : {(blocked_sum(digits,axes) + _ : U32) == blocked_sum(digits,every) : U32}
      %Equal.sym(U32,(blocked_sum(digits,axes) + 0 : U32),blocked_sum(digits,axes),Arithmetic.add_zero(blocked_sum(digits,axes))) : {_ == blocked_sum(digits,every) : U32}
      %same : {blocked_sum(_,axes) == blocked_sum(digits,every) : U32}
      Equal.trans(U32,blocked_sum(removed_at(digits,rank),axes),(blocked_sum(removed_at(digits,rank),axes) + 0 : U32),blocked_sum(digits,every),
        Equal.sym(U32,(blocked_sum(removed_at(digits,rank),axes) + 0 : U32),blocked_sum(removed_at(digits,rank),axes),Arithmetic.add_zero(blocked_sum(removed_at(digits,rank),axes))),
        {%Order.not_below(rank,rank,Order.reflexive(rank)) : {(blocked_sum(removed_at(digits,rank),axes) + skipped_term(_,skipped,BoxDigits.digit_at(digits,rank)) : U32) == blocked_sum(digits,every) : U32}
          all : {(blocked_sum(removed_at(digits,rank),axes) + 0 : U32) == blocked_sum(digits,every) : U32}})
    case True{} Grid.Innermost{} Nil{}:
      Empty.absurd({(blocked_sum(Nil{},Tiled.blocked_list(Tiled.target_axes(True{},Grid.Innermost{},target,rank),source,source_within,source_position,target,strides,shape,groups_axis)) +
          U32.from_nat(lane) * blocked_step(Tiled.blocked_list(Tiled.target_axes(True{},Grid.Innermost{},target,rank),source,source_within,source_position,target,strides,shape,
            groups_axis),0) : U32) ==
        blocked_sum(target_logical(True{},Grid.Innermost{},target,Nil{},lane),Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,
          groups_axis)) : U32},
        Order.below_zero(target,{%Equal.sym(Nat,0n,rank,length) : {Nat.is_lt(target,_) == True{} : Bool} within_value : {Nat.is_lt(target,0n) == True{} : Bool}}))
    case True{} Grid.Outermost{} Nil{}:
      Empty.absurd({(blocked_sum(Nil{},Tiled.blocked_list(Tiled.target_axes(True{},Grid.Outermost{},target,rank),source,source_within,source_position,target,strides,shape,groups_axis)) +
          U32.from_nat(lane) * blocked_step(Tiled.blocked_list(Tiled.target_axes(True{},Grid.Outermost{},target,rank),source,source_within,source_position,target,strides,shape,
            groups_axis),0) : U32) ==
        blocked_sum(target_logical(True{},Grid.Outermost{},target,Nil{},lane),Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,
          groups_axis)) : U32},
        Order.below_zero(target,{%Equal.sym(Nat,0n,rank,length) : {Nat.is_lt(target,_) == True{} : Bool} within_value : {Nat.is_lt(target,0n) == True{} : Bool}}))
    case True{} Grid.Innermost{} Con{+first,+rest}:
      +digits = {first <> rest : List<&2,Nat>}
      +others = Tiled.axes_from(rank,0n,target)
      +axes = Tiled.blocked_list(others,source,source_within,source_position,target,strides,shape,groups_axis)
      +every = Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,groups_axis)
      +step = Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target))
      +groups = Tiled.blocked_axis(Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)
      +value = Nat.add(lane,Nat.mul(digits_back(digits),8n))
      +logical = Coordinates.inserted_digit(target,digits_front(digits),value)
      +counted = {%within_value : {Nat.add(List.length(&2,Tiled.CellAxis,others),skipped_count(_)) == rank : Nat} axes_count(rank,target,0n) :
        {Nat.add(List.length(&2,Tiled.CellAxis,others),1n) == rank : Nat}}
      +others_count = Equal.trans(Nat,1n+List.length(&2,Tiled.CellAxis,others),Nat.add(List.length(&2,Tiled.CellAxis,others),1n),rank,
        Equal.sym(Nat,Nat.add(List.length(&2,Tiled.CellAxis,others),1n),1n+List.length(&2,Tiled.CellAxis,others),Addition.successor_width(List.length(&2,Tiled.CellAxis,others))),counted)
      +digits_count = Equal.trans(Nat,List.length(&2,Nat,digits),rank,1n+List.length(&2,Tiled.Blocked,axes),length,
        Equal.trans(Nat,rank,1n+List.length(&2,Tiled.CellAxis,others),1n+List.length(&2,Tiled.Blocked,axes),Equal.sym(Nat,1n+List.length(&2,Tiled.CellAxis,others),rank,others_count),
          Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Tiled.CellAxis,others),List.length(&2,Tiled.Blocked,axes),
            Equal.sym(Nat,List.length(&2,Tiled.Blocked,axes),List.length(&2,Tiled.CellAxis,others),blocked_count(others,source,source_within,source_position,target,strides,shape,groups_axis)))))
      +room = Division.lt_succ_le(target,List.length(&2,Nat,digits_front(digits)),
        {%Equal.sym(Nat,1n+List.length(&2,Nat,digits_front(digits)),rank,Equal.trans(Nat,1n+List.length(&2,Nat,digits_front(digits)),1n+List.length(&2,Nat,rest),rank,front_count(rest,first),length)) :
          {Nat.is_lt(target,_) == True{} : Bool} within_value : {Nat.is_lt(target,1n+List.length(&2,Nat,digits_front(digits))) == True{} : Bool}})
      +all = skipped_sum(rank,target,0n,logical,source,source_within,source_position,target,strides,shape,groups_axis)
      %Equal.sym(List<&2,Tiled.Blocked>,Tiled.blocked_list(List.append(&2,Tiled.CellAxis,others,[Tiled.OfGroups{}]),source,source_within,source_position,target,strides,shape,groups_axis),
          List.append(&2,Tiled.Blocked,axes,[groups]),blocked_append(others,Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)) :
        {(blocked_sum(digits,_) + U32.from_nat(lane) * blocked_step(_,0) : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,blocked_step(List.append(&2,Tiled.Blocked,axes,[groups]),0),blocked_step([groups],blocked_step(axes,0)),blocked_step_append(axes,groups,0)) :
        {(blocked_sum(digits,List.append(&2,Tiled.Blocked,axes,[groups])) + U32.from_nat(lane) * _ : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,blocked_step(axes,0),0,axes_step(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis,0)) :
        {(blocked_sum(digits,List.append(&2,Tiled.Blocked,axes,[groups])) + U32.from_nat(lane) * blocked_step([groups],_) : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,blocked_sum(digits,List.append(&2,Tiled.Blocked,axes,[groups])),(blocked_sum(digits_front(digits),axes) + blocked_term(groups,digits_back(digits)) : U32),
          sum_last(digits,axes,groups,digits_count)) :
        {(_ + U32.from_nat(lane) * step : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,((blocked_sum(digits_front(digits),axes) + U32.from_nat(Nat.mul(digits_back(digits),8n)) * step : U32) + U32.from_nat(lane) * step : U32),
          (blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32),groups_merge(blocked_sum(digits_front(digits),axes),digits_back(digits),lane,step)) :
        {_ == blocked_sum(logical,every) : U32}
      %all : {(blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32) == _ : U32}
      %Equal.sym(Bool,Nat.is_lt(target,rank),True{},within_value) : {(blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32) ==
          (blocked_sum(removed_at(logical,target),axes) + skipped_term(_,Tiled.blocked_axis(Tiled.OfAxis{target},source,source_within,source_position,target,strides,shape,groups_axis),
            BoxDigits.digit_at(logical,target)) : U32) : U32}
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{target},source,source_within,source_position,target,strides,shape,groups_axis),Tiled.Blocked{Tiled.Shared{},step,1},
          target_tag(source,source_within,source_position,target,strides,shape,groups_axis,apart)) :
        {(blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32) == (blocked_sum(removed_at(logical,target),axes) + blocked_term(_,BoxDigits.digit_at(logical,target)) : U32) : U32}
      %Equal.sym(List<&2,Nat>,removed_at(logical,target),digits_front(digits),removed_inserted(target,digits_front(digits),value,room)) :
        {(blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32) == (blocked_sum(_,axes) + U32.from_nat(BoxDigits.digit_at(logical,target)) * step : U32) : U32}
      %Equal.sym(Nat,BoxDigits.digit_at(logical,target),value,digit_inserted(target,digits_front(digits),value,room)) :
        {(blocked_sum(digits_front(digits),axes) + U32.from_nat(value) * step : U32) == (blocked_sum(digits_front(digits),axes) + U32.from_nat(_) * step : U32) : U32}
      {==}
    case True{} Grid.Outermost{} Con{+first,+rest}:
      +others = Tiled.axes_from(rank,0n,target)
      +axes = Tiled.blocked_list(others,source,source_within,source_position,target,strides,shape,groups_axis)
      +every = Tiled.blocked_list(every_axis(rank,0n),source,source_within,source_position,target,strides,shape,groups_axis)
      +step = Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target))
      +value = Nat.add(lane,Nat.mul(first,8n))
      +logical = Coordinates.inserted_digit(target,rest,value)
      +room = Division.lt_succ_le(target,List.length(&2,Nat,rest),{%Equal.sym(Nat,1n+List.length(&2,Nat,rest),rank,length) : {Nat.is_lt(target,_) == True{} : Bool} within_value : {Nat.is_lt(target,1n+List.length(&2,Nat,rest)) == True{} : Bool}})
      +all = skipped_sum(rank,target,0n,logical,source,source_within,source_position,target,strides,shape,groups_axis)
      %Equal.sym(U32,blocked_step(axes,step),step,axes_step(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis,step)) :
        {((U32.from_nat(Nat.mul(first,8n)) * step + blocked_sum(rest,axes) : U32) + U32.from_nat(lane) * _ : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,(U32.from_nat(Nat.mul(first,8n)) * step + blocked_sum(rest,axes) : U32),(blocked_sum(rest,axes) + U32.from_nat(Nat.mul(first,8n)) * step : U32),
          Arithmetic.add_commutative((U32.from_nat(Nat.mul(first,8n)) * step : U32),blocked_sum(rest,axes))) :
        {(_ + U32.from_nat(lane) * step : U32) == blocked_sum(logical,every) : U32}
      %Equal.sym(U32,((blocked_sum(rest,axes) + U32.from_nat(Nat.mul(first,8n)) * step : U32) + U32.from_nat(lane) * step : U32),
          (blocked_sum(rest,axes) + U32.from_nat(value) * step : U32),groups_merge(blocked_sum(rest,axes),first,lane,step)) :
        {_ == blocked_sum(logical,every) : U32}
      %all : {(blocked_sum(rest,axes) + U32.from_nat(value) * step : U32) == _ : U32}
      %Equal.sym(Bool,Nat.is_lt(target,rank),True{},within_value) : {(blocked_sum(rest,axes) + U32.from_nat(value) * step : U32) ==
          (blocked_sum(removed_at(logical,target),axes) + skipped_term(_,Tiled.blocked_axis(Tiled.OfAxis{target},source,source_within,source_position,target,strides,shape,groups_axis),
            BoxDigits.digit_at(logical,target)) : U32) : U32}
      %Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{target},source,source_within,source_position,target,strides,shape,groups_axis),Tiled.Blocked{Tiled.Shared{},step,1},
          target_tag(source,source_within,source_position,target,strides,shape,groups_axis,apart)) :
        {(blocked_sum(rest,axes) + U32.from_nat(value) * step : U32) == (blocked_sum(removed_at(logical,target),axes) + blocked_term(_,BoxDigits.digit_at(logical,target)) : U32) : U32}
      %Equal.sym(List<&2,Nat>,removed_at(logical,target),rest,removed_inserted(target,rest,value,room)) :
        {(blocked_sum(rest,axes) + U32.from_nat(value) * step : U32) == (blocked_sum(_,axes) + U32.from_nat(BoxDigits.digit_at(logical,target)) * step : U32) : U32}
      %Equal.sym(Nat,BoxDigits.digit_at(logical,target),value,digit_inserted(target,rest,value,room)) :
        {(blocked_sum(rest,axes) + U32.from_nat(value) * step : U32) == (blocked_sum(rest,axes) + U32.from_nat(_) * step : U32) : U32}
      {==}

# ---- The source lane axis among the target cell axes ----

# The axes from index on never reach an axis past them.
def never_reach(remaining: Nat,+index: Nat,+skip: Nat,+past: {Nat.is_le(Nat.add(index,remaining),skip) == True{} : Bool}) ->
  {Tiled.axes_from(remaining,index,skip) == every_axis(remaining,index) : List<&2,Tiled.CellAxis>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +below = Order.le_lt_trans(index,Nat.add(index,rest),skip,Order.le_add(index,rest),
        BoxDigits.follow_strict(Nat.add(index,rest),skip,{%Addition.successor_right(index,rest) : {Nat.is_le(_,skip) == True{} : Bool} past :
          {Nat.is_le(1n+Nat.add(index,rest),skip) == True{} : Bool}}))
      +differ = Order.less_not_equal(index,skip,below)
      +later = never_reach(rest,1n+index,skip,{%Addition.successor_right(index,rest) : {Nat.is_le(_,skip) == True{} : Bool} past :
        {Nat.is_le(1n+Nat.add(index,rest),skip) == True{} : Bool}})
      %Equal.sym(Bool,Nat.is_eq(index,skip),False{},differ) :
        {Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)) ==
          every_axis(1n+rest,index) : List<&2,Tiled.CellAxis>}
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rest,1n+index,skip),every_axis(rest,1n+index),later) :
        {Tiled.OfAxis{index} <> _ == Tiled.OfAxis{index} <> every_axis(rest,1n+index) : List<&2,Tiled.CellAxis>}
      {==}

# The digit of the source lane axis is lane lane of its block.
def role_lane(role: Tiled.Role,+digit: Nat,+lane: Nat) -> Data:
  match role:
    case Tiled.Lanes{}: {Nat.mod(digit,8n) == lane : Nat}
    case Tiled.Shared{}: Unit
    case Tiled.Groups{}: Unit

def LaneDigits(blocked: List<&2,Tiled.Blocked>,digits: List<&2,Nat>,+lane: Nat) -> Data:
  match blocked digits:
    case Con{Tiled.Blocked{role,source,lanes},rest} Con{+digit,more}: KernelGemm.Both<role_lane(role,digit,lane),LaneDigits(rest,more,lane)>
    case Nil{} Nil{}: Unit
    case blocked digits: Empty

def lane_rest(+group: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.mod(Nat.add(Nat.mul(group,8n),lane),8n) == lane : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(group,8n),lane),8n),(group,lane),
    Division.quotient_remainder(group,lane,8n,{==},small))

def role_lane_digit(role: Tiled.Role,+low: U32,+index: U32,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> role_lane(role,role_digit(role,low,index,lane),lane):
  match role:
    case Tiled.Lanes{}: lane_rest(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index)),lane,small)
    case Tiled.Shared{}: Unit{}
    case Tiled.Groups{}: Unit{}

# The source lane axis of a block holds lane lane of its block.
def block_lanes(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  LaneDigits(blocked_of(axes),block_digits(axes,index,lane),lane):
  match axes:
    case Nil{}: Unit{}
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{role_lane_digit(role,low,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane,small),block_lanes(rest,index,lane,small)}

def appended_empty(blocked: List<&2,Tiled.Blocked>,+last: Tiled.Blocked,+lane: Nat,+lanes: LaneDigits(List.append(&2,Tiled.Blocked,blocked,[last]),Nil{},lane)) -> Empty:
  match blocked:
    case Nil{}:
      match last:
        case Tiled.Blocked{+role,+source,+count}: lanes
    case Con{Tiled.Blocked{+role,+source,+count},+tail}: lanes

def lanes_front(digits: List<&2,Nat>,blocked: List<&2,Tiled.Blocked>,+last: Tiled.Blocked,+lane: Nat,
  +lanes: LaneDigits(List.append(&2,Tiled.Blocked,blocked,[last]),digits,lane)) -> LaneDigits(blocked,digits_front(digits),lane):
  match digits blocked:
    case Nil{} Nil{}: Empty.absurd(Unit,appended_empty(Nil{},last,lane,lanes))
    case Con{+digit,Nil{}} Nil{}: Unit{}
    case Con{+digit,Con{+next,+rest}} Nil{}:
      match last:
        case Tiled.Blocked{+role,+source,+count}:
          KernelGemm.Both{+here,+later} = lanes
          Empty.absurd(LaneDigits(Nil{},digits_front(digit <> next <> rest),lane),later)
    case Nil{} Con{Tiled.Blocked{+role,+source,+count},+tail}: Empty.absurd(LaneDigits(Tiled.Blocked{role,source,count} <> tail,Nil{},lane),lanes)
    case Con{+digit,Nil{}} Con{Tiled.Blocked{+role,+source,+count},+tail}:
      KernelGemm.Both{+here,+later} = lanes
      Empty.absurd(LaneDigits(Tiled.Blocked{role,source,count} <> tail,Nil{},lane),appended_empty(tail,last,lane,later))
    case Con{+digit,Con{+next,+rest}} Con{Tiled.Blocked{+role,+source,+count},+tail}:
      KernelGemm.Both{+here,+later} = lanes
      KernelGemm.Both{here,lanes_front(next <> rest,tail,last,lane,later)}

# The axes from index on when the skipped axis comes later, and when it is
# index itself.
def tags_later(+rest: Nat,+index: Nat,+gap: Nat) ->
  {Tiled.axes_from(1n+rest,index,Nat.add(index,1n+gap)) == Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,Nat.add(1n+index,gap)) : List<&2,Tiled.CellAxis>}:
  +skip = Nat.add(1n+index,gap)
  +differ = Order.less_not_equal(index,Nat.add(index,1n+gap),Order.below_sum(index,gap))
  %Equal.sym(Nat,Nat.add(index,1n+gap),skip,Addition.successor_right(index,gap)) :
    {Tiled.axes_from(1n+rest,index,_) == Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip) : List<&2,Tiled.CellAxis>}
  %Equal.sym(Bool,Nat.is_eq(index,skip),False{},{%Addition.successor_right(index,gap) : {Nat.is_eq(index,_) == False{} : Bool} differ : {Nat.is_eq(index,skip) == False{} : Bool}}) :
    {Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)) ==
      Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip) : List<&2,Tiled.CellAxis>}
  {==}

def tags_here(+rest: Nat,+index: Nat) -> {Tiled.axes_from(1n+rest,index,Nat.add(index,0n)) == every_axis(rest,1n+index) : List<&2,Tiled.CellAxis>}:
  %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) : {Tiled.axes_from(1n+rest,index,_) == every_axis(rest,1n+index) : List<&2,Tiled.CellAxis>}
  %Equal.sym(Bool,Nat.is_eq(index,index),True{},Order.equal_reflexive(index)) :
    {Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,index),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,index)) == every_axis(rest,1n+index) :
      List<&2,Tiled.CellAxis>}
  never_hit(rest,1n+index,index,Order.lt_succ(index,index,Order.reflexive(index)))

# A logical axis other than the source lane axis is shared.
def shared_tag(+axis: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +differ: {Nat.is_eq(axis,source) == False{} : Bool}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{axis},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,axis)),1} : Tiled.Blocked}:
  %Equal.sym(Bool,Nat.is_eq(axis,source),False{},differ) :
    {Bool.pick(Tiled.Blocked,Bool.and(source_within,_),Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,axis)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,axis)),1} : Tiled.Blocked}
  %Equal.sym(Bool,Bool.and(source_within,False{}),False{},and_false(source_within)) :
    {Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,axis)),1}) ==
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,axis)),1} : Tiled.Blocked}
  {==}

def lane_first(+digit: Nat,+more: List<&2,Nat>,+rest: List<&2,Tiled.Blocked>,+lane: Nat,
  +both: KernelGemm.Both<{Nat.mod(digit,8n) == lane : Nat},LaneDigits(rest,more,lane)>) -> {Nat.mod(digit,8n) == lane : Nat}:
  KernelGemm.Both{+here,+later} = both
  here

def lane_later(+source: U32,+digit: Nat,+more: List<&2,Nat>,+rest: List<&2,Tiled.Blocked>,+lane: Nat,
  +both: LaneDigits(Tiled.Blocked{Tiled.Shared{},source,1} <> rest,digit <> more,lane)) -> LaneDigits(rest,more,lane):
  KernelGemm.Both{+here,+later} = both
  later

# The source lane axis among every axis from index on holds lane lane.
def every_lane(remaining: Nat,offset: Nat,digits: List<&2,Nat>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+lane: Nat,+within: {source_within == True{} : Bool},+place: {Nat.add(index,offset) == source : Nat},
  +inside: {Nat.is_lt(offset,remaining) == True{} : Bool},
  +lanes: LaneDigits(Tiled.blocked_list(every_axis(remaining,index),source,source_within,source_position,target,strides,shape,groups_axis),digits,lane)) ->
  {Nat.mod(BoxDigits.digit_at(digits,offset),8n) == lane : Nat}:
  match remaining offset digits:
    case 0n +offset +digits: Empty.absurd({Nat.mod(BoxDigits.digit_at(digits,offset),8n) == lane : Nat},Order.below_zero(offset,inside))
    case 1n+ +rest 0n Nil{}:
      +same = Equal.trans(Nat,index,Nat.add(index,0n),source,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),place)
      +hit = {%same : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,source) == True{} : Bool}}
      +lane_at = {%Equal.sym(Bool,source_within,True{},within) : {Bool.and(_,Nat.is_eq(index,source)) == True{} : Bool} hit :
        {Bool.and(source_within,Nat.is_eq(index,source)) == True{} : Bool}}
      Empty.absurd({Nat.mod(0n,8n) == lane : Nat},
        {%lane_at : LaneDigits(Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
            Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1}) <>
            Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
          lanes : Empty})
    case 1n+ +rest 0n Con{+digit,+more}:
      +same = Equal.trans(Nat,index,Nat.add(index,0n),source,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),place)
      +hit = {%same : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,source) == True{} : Bool}}
      +lane_at = {%Equal.sym(Bool,source_within,True{},within) : {Bool.and(_,Nat.is_eq(index,source)) == True{} : Bool} hit :
        {Bool.and(source_within,Nat.is_eq(index,source)) == True{} : Bool}}
      +found = {%lane_at : LaneDigits(Bool.pick(Tiled.Blocked,_,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
            Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1}) <>
            Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),digit <> more,lane)
          lanes : KernelGemm.Both<{Nat.mod(digit,8n) == lane : Nat},LaneDigits(Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,
            shape,groups_axis),more,lane)>}
      lane_first(digit,more,Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),lane,found)
    case 1n+ +rest 1n+ +next Nil{}:
      +differ = {%place : {Nat.is_eq(index,_) == False{} : Bool} Order.less_not_equal(index,Nat.add(index,1n+next),Order.below_sum(index,next)) :
        {Nat.is_eq(index,source) == False{} : Bool}}
      +shared = {%shared_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,differ) :
          LaneDigits(_ <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
        lanes : LaneDigits(Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1} <>
          Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)}
      Empty.absurd({Nat.mod(0n,8n) == lane : Nat},shared)
    case 1n+ +rest 1n+ +next Con{+digit,+more}:
      +differ = {%place : {Nat.is_eq(index,_) == False{} : Bool} Order.less_not_equal(index,Nat.add(index,1n+next),Order.below_sum(index,next)) :
        {Nat.is_eq(index,source) == False{} : Bool}}
      +shared = {%shared_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,differ) :
          LaneDigits(_ <> Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),digit <> more,lane)
        lanes : LaneDigits(Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1} <>
          Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),digit <> more,lane)}
      every_lane(rest,next,more,1n+index,source,source_within,source_position,target,strides,shape,groups_axis,lane,within,
        Equal.trans(Nat,Nat.add(1n+index,next),Nat.add(index,1n+next),source,Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)),place),
        inside,lane_later(Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),digit,more,
          Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),lane,shared))

def lane_tag_at(+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,
  +within: {source_within == True{} : Bool},+same: {index == source : Nat}) ->
  {Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis) ==
    Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)} : Tiled.Blocked}:
  %Equal.sym(Nat,index,source,same) :
    {Tiled.blocked_axis(Tiled.OfAxis{_},source,source_within,source_position,target,strides,shape,groups_axis) ==
      Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)} : Tiled.Blocked}
  lane_tag(source,source_within,source_position,target,strides,shape,groups_axis,within)

# The logical digit of the source lane axis is lane lane of its block, the
# target groups put back at their axis.
def logical_lane(remaining: Nat,gap: Nat,offset: Nat,digits: List<&2,Nat>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+value: Nat,+lane: Nat,+within: {source_within == True{} : Bool},
  +place: {Nat.add(index,offset) == source : Nat},+apart: {Nat.is_eq(offset,gap) == False{} : Bool},+inside: {Nat.is_lt(offset,remaining) == True{} : Bool},
  +lanes: LaneDigits(Tiled.blocked_list(Tiled.axes_from(remaining,index,Nat.add(index,gap)),source,source_within,source_position,target,strides,shape,groups_axis),digits,lane)) ->
  {Nat.mod(BoxDigits.digit_at(Coordinates.inserted_digit(gap,digits,value),offset),8n) == lane : Nat}:
  match remaining gap offset digits:
    case 0n +gap +offset +digits:
      Empty.absurd({Nat.mod(BoxDigits.digit_at(Coordinates.inserted_digit(gap,digits,value),offset),8n) == lane : Nat},Order.below_zero(offset,inside))
    case 1n+ +rest 0n 0n +digits: Empty.absurd({Nat.mod(BoxDigits.digit_at(Coordinates.inserted_digit(0n,digits,value),0n),8n) == lane : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case 1n+ +rest 0n 1n+ +next +digits:
      every_lane(rest,next,digits,1n+index,source,source_within,source_position,target,strides,shape,groups_axis,lane,within,
        Equal.trans(Nat,Nat.add(1n+index,next),Nat.add(index,1n+next),source,Equal.sym(Nat,Nat.add(index,1n+next),1n+Nat.add(index,next),Addition.successor_right(index,next)),place),
        inside,{%tags_here(rest,index) : LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis),digits,lane) lanes :
          LaneDigits(Tiled.blocked_list(every_axis(rest,1n+index),source,source_within,source_position,target,strides,shape,groups_axis),digits,lane)})
    case 1n+ +rest 1n+ +next 0n Nil{}:
      +same = Equal.trans(Nat,index,Nat.add(index,0n),source,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),place)
      Empty.absurd({Nat.mod(value,8n) == lane : Nat},
        {%lane_tag_at(index,source,source_within,source_position,target,strides,shape,groups_axis,within,same) :
            LaneDigits(_ <> Tiled.blocked_list(Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
          {%tags_later(rest,index,next) : LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
            lanes : LaneDigits(Tiled.blocked_list(Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,
              groups_axis),Nil{},lane)} : Empty})
    case 1n+ +rest 1n+ +next 0n Con{+digit,+more}:
      +same = Equal.trans(Nat,index,Nat.add(index,0n),source,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),place)
      +later = Tiled.blocked_list(Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,groups_axis)
      lane_first(digit,more,later,lane,
        {%lane_tag_at(index,source,source_within,source_position,target,strides,shape,groups_axis,within,same) : LaneDigits(_ <> later,digit <> more,lane)
          {%tags_later(rest,index,next) : LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis),digit <> more,lane)
            lanes : LaneDigits(Tiled.blocked_list(Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,
              groups_axis),digit <> more,lane)} : KernelGemm.Both<{Nat.mod(digit,8n) == lane : Nat},LaneDigits(later,more,lane)>})
    case 1n+ +rest 1n+ +next 1n+ +after Nil{}:
      +differ = {%place : {Nat.is_eq(index,_) == False{} : Bool} Order.less_not_equal(index,Nat.add(index,1n+after),Order.below_sum(index,after)) :
        {Nat.is_eq(index,source) == False{} : Bool}}
      Empty.absurd({Nat.mod(0n,8n) == lane : Nat},
        {%shared_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,differ) :
            LaneDigits(_ <> Tiled.blocked_list(Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
          {%tags_later(rest,index,next) : LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis),Nil{},lane)
            lanes : LaneDigits(Tiled.blocked_list(Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,
              groups_axis),Nil{},lane)} : Empty})
    case 1n+ +rest 1n+ +next 1n+ +after Con{+digit,+more}:
      +differ = {%place : {Nat.is_eq(index,_) == False{} : Bool} Order.less_not_equal(index,Nat.add(index,1n+after),Order.below_sum(index,after)) :
        {Nat.is_eq(index,source) == False{} : Bool}}
      +step = Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index))
      +later = Tiled.blocked_list(Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,groups_axis)
      logical_lane(rest,next,after,more,1n+index,source,source_within,source_position,target,strides,shape,groups_axis,value,lane,within,
        Equal.trans(Nat,Nat.add(1n+index,after),Nat.add(index,1n+after),source,Equal.sym(Nat,Nat.add(index,1n+after),1n+Nat.add(index,after),Addition.successor_right(index,after)),place),
        apart,inside,
        lane_later(step,digit,more,later,lane,
          {%shared_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,differ) : LaneDigits(_ <> later,digit <> more,lane)
            {%tags_later(rest,index,next) : LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,groups_axis),digit <> more,lane)
              lanes : LaneDigits(Tiled.blocked_list(Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,Nat.add(1n+index,next)),source,source_within,source_position,target,strides,shape,
                groups_axis),digit <> more,lane)} : LaneDigits(Tiled.Blocked{Tiled.Shared{},step,1} <> later,digit <> more,lane)}))

# ---- The source lane axis appears once ----

def NoLanes(blocked: List<&2,Tiled.Blocked>) -> Data:
  match blocked:
    case Nil{}: Unit
    case Con{Tiled.Blocked{Tiled.Lanes{},source,lanes},rest}: Empty
    case Con{Tiled.Blocked{role,source,lanes},rest}: NoLanes(rest)

def LanesAtMost(blocked: List<&2,Tiled.Blocked>) -> Data:
  match blocked:
    case Nil{}: Unit
    case Con{Tiled.Blocked{Tiled.Lanes{},source,lanes},rest}: NoLanes(rest)
    case Con{Tiled.Blocked{role,source,lanes},rest}: LanesAtMost(rest)

def no_lanes_most(blocked: List<&2,Tiled.Blocked>,+none: NoLanes(blocked)) -> LanesAtMost(blocked):
  match blocked:
    case Nil{}: Unit{}
    case Con{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+rest}: Empty.absurd(NoLanes(rest),none)
    case Con{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+rest}: no_lanes_most(rest,none)
    case Con{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+rest}: no_lanes_most(rest,none)

# A shared axis before axes with no source lane axis.
def shared_none(choice: Bool,+lane_source: U32,+lane_lanes: U32,+source: U32,+rest: List<&2,Tiled.Blocked>,+differ: {choice == False{} : Bool},+none: NoLanes(rest)) ->
  NoLanes(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lane_lanes},Tiled.Blocked{Tiled.Shared{},source,1}) <> rest):
  match choice:
    case True{}: Empty.absurd(Empty,WordEncoding.false_true(Equal.sym(Bool,True{},False{},differ)))
    case False{}: none

def none_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+below: {Nat.is_lt(source,index) == True{} : Bool},
  +none: NoLanes(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))) ->
  NoLanes(Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),source,source_within,source_position,target,strides,shape,groups_axis)):
  match choice:
    case True{}: none
    case False{}:
      +differ = Equal.trans(Bool,Nat.is_eq(index,source),Nat.is_eq(source,index),False{},Order.equal_symmetric(index,source),Order.less_not_equal(source,index,below))
      shared_none(Bool.and(source_within,Nat.is_eq(index,source)),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),
        {%Equal.sym(Bool,Nat.is_eq(index,source),False{},differ) : {Bool.and(source_within,_) == False{} : Bool} and_false(source_within) :
          {Bool.and(source_within,Nat.is_eq(index,source)) == False{} : Bool}},none)

# The axes after the source lane axis hold none of it.
def no_lanes_after(remaining: Nat,+index: Nat,+skip: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+below: {Nat.is_lt(source,index) == True{} : Bool}) ->
  NoLanes(Tiled.blocked_list(Tiled.axes_from(remaining,index,skip),source,source_within,source_position,target,strides,shape,groups_axis)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      none_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,source,source_within,source_position,target,strides,shape,groups_axis,below,
        no_lanes_after(rest,1n+index,skip,source,source_within,source_position,target,strides,shape,groups_axis,
          Order.lt_succ(source,index,Bounds.strict_le(source,index,below))))

# An axis that may be the source lane axis, before axes after it.
def once_pick_head(choice: Bool,+lane_source: U32,+lane_lanes: U32,+source: U32,+rest: List<&2,Tiled.Blocked>,+none: NoLanes(rest)) ->
  LanesAtMost(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lane_lanes},Tiled.Blocked{Tiled.Shared{},source,1}) <> rest):
  match choice:
    case True{}: none
    case False{}: no_lanes_most(rest,none)

def once_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+same: {Nat.is_eq(index,source) == True{} : Bool},
  +none: NoLanes(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))) ->
  LanesAtMost(Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),source,source_within,source_position,target,strides,shape,groups_axis)):
  match choice:
    case True{}: no_lanes_most(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),none)
    case False{}:
      once_pick_head(Bool.and(source_within,Nat.is_eq(index,source)),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),none)

def most_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+differ: {Nat.is_eq(index,source) == False{} : Bool},
  +most: LanesAtMost(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))) ->
  LanesAtMost(Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),source,source_within,source_position,target,strides,shape,groups_axis)):
  match choice:
    case True{}: most
    case False{}:
      {%Equal.sym(Tiled.Blocked,Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis),
          Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),1},
          shared_tag(index,source,source_within,source_position,target,strides,shape,groups_axis,differ)) :
        LanesAtMost(_ <> Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))
        most : LanesAtMost(Tiled.blocked_axis(Tiled.OfAxis{index},source,source_within,source_position,target,strides,shape,groups_axis) <>
          Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))}

def most_case(hit: Bool,+rest: Nat,+index: Nat,+skip: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+same: {Nat.is_eq(index,source) == hit : Bool},
  +most: LanesAtMost(Tiled.blocked_list(Tiled.axes_from(rest,1n+index,skip),source,source_within,source_position,target,strides,shape,groups_axis))) ->
  LanesAtMost(Tiled.blocked_list(Tiled.axes_from(1n+rest,index,skip),source,source_within,source_position,target,strides,shape,groups_axis)):
  match hit:
    case True{}:
      +equal = Order.equal_of_is_eq(index,source,same)
      once_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,source,source_within,source_position,target,strides,shape,groups_axis,same,
        no_lanes_after(rest,1n+index,skip,source,source_within,source_position,target,strides,shape,groups_axis,
          {%equal : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) : {Nat.is_lt(source,1n+index) == True{} : Bool}}))
    case False{}:
      most_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,source,source_within,source_position,target,strides,shape,groups_axis,same,most)

# The axes but one hold the source lane axis at most once.
def lanes_at_most(remaining: Nat,+index: Nat,+skip: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  LanesAtMost(Tiled.blocked_list(Tiled.axes_from(remaining,index,skip),source,source_within,source_position,target,strides,shape,groups_axis)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest: most_case(Nat.is_eq(index,source),rest,index,skip,source,source_within,source_position,target,strides,shape,groups_axis,{==},
      lanes_at_most(rest,1n+index,skip,source,source_within,source_position,target,strides,shape,groups_axis))

# Target groups after axes that hold the source lane axis at most once.
def none_groups(blocked: List<&2,Tiled.Blocked>,+source: U32,+lanes: U32,+none: NoLanes(blocked)) ->
  NoLanes(List.append(&2,Tiled.Blocked,blocked,[Tiled.Blocked{Tiled.Groups{},source,lanes}])):
  match blocked:
    case Nil{}: Unit{}
    case Con{Tiled.Blocked{Tiled.Lanes{},+step,+extent},+rest}:
      Empty.absurd(NoLanes(List.append(&2,Tiled.Blocked,Con{Tiled.Blocked{Tiled.Lanes{},step,extent},rest},[Tiled.Blocked{Tiled.Groups{},source,lanes}])),none)
    case Con{Tiled.Blocked{Tiled.Shared{},+step,+extent},+rest}: none_groups(rest,source,lanes,none)
    case Con{Tiled.Blocked{Tiled.Groups{},+step,+extent},+rest}: none_groups(rest,source,lanes,none)

def most_groups(blocked: List<&2,Tiled.Blocked>,+source: U32,+lanes: U32,+most: LanesAtMost(blocked)) ->
  LanesAtMost(List.append(&2,Tiled.Blocked,blocked,[Tiled.Blocked{Tiled.Groups{},source,lanes}])):
  match blocked:
    case Nil{}: Unit{}
    case Con{Tiled.Blocked{Tiled.Lanes{},+step,+extent},+rest}: none_groups(rest,source,lanes,most)
    case Con{Tiled.Blocked{Tiled.Shared{},+step,+extent},+rest}: most_groups(rest,source,lanes,most)
    case Con{Tiled.Blocked{Tiled.Groups{},+step,+extent},+rest}: most_groups(rest,source,lanes,most)

# The target cell axes hold the source lane axis at most once.
def target_most(within: Bool,position: Grid.Position,+target: Nat,+rank: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  LanesAtMost(Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,strides,shape,groups_axis)):
  match within position:
    case False{} +position: lanes_at_most(rank,0n,rank,source,source_within,source_position,target,strides,shape,groups_axis)
    case True{} Grid.Innermost{}:
      +axes = Tiled.axes_from(rank,0n,target)
      %Equal.sym(List<&2,Tiled.Blocked>,Tiled.blocked_list(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),source,source_within,source_position,target,strides,shape,groups_axis),
          List.append(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),
            [Tiled.blocked_axis(Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)]),
          blocked_append(axes,Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)) :
        LanesAtMost(_)
      most_groups(Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target)),Grid.lanes_at(shape,target),
        lanes_at_most(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis))
    case True{} Grid.Outermost{}: lanes_at_most(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis)
