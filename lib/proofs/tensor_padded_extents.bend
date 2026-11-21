# The extents of a padded tensor: every axis of the tensor fits its padded
# axis from the padding in front on, without wrapping a word.

import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ./tensor_placing_facts.bend as Facts
import ./tensor_move_readback.bend as Held
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./word_encoding.bend as Words

# Two words that the runtime adds without wrapping sum within a word.
def word_room(+left: U32,+right: U32,+top: U32,+top_exact: {top == 4294967295 : U32},+classified: {U32.is_le(right,(top - left : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(top)) == True{} : Bool}:
  +l = U32.to_nat(left)
  +most = U32.to_nat(top)
  +ordered = Bounds.as_nat_le(left,top,{%Equal.sym(U32,top,4294967295,top_exact) : {U32.is_le(left,_) == True{} : Bool} FullRange.maximum(left) : {U32.is_le(left,top) == True{} : Bool}})
  +exact = FullRange.subtract(top,left,top,ordered,Order.sub_le(most,l))
  +narrow = {%exact : {Nat.is_le(U32.to_nat(right),_) == True{} : Bool} Bounds.as_nat_le(right,(top - left : U32),classified) :
    {Nat.is_le(U32.to_nat(right),Nat.sub(most,l)) == True{} : Bool}}
  %Order.restore_shift(l,most,ordered) : {Nat.is_le(Nat.add(l,U32.to_nat(right)),_) == True{} : Bool}
  Equal.trans(Bool,Nat.is_le(Nat.add(l,U32.to_nat(right)),Nat.add(l,Nat.sub(most,l))),Nat.is_le(U32.to_nat(right),Nat.sub(most,l)),True{},
    Order.le_offset(l,U32.to_nat(right),Nat.sub(most,l)),narrow)

def present(found: Maybe<&2,U32>) -> Bool:
  match found:
    case Some{value}: True{}
    case None{}: False{}

def known(found: Maybe<&2,U32>) -> U32:
  match found:
    case Some{+value}: value
    case None{}: 0

# A checked sum that is present is the sum without wrapping.
def checked_sum(valid: Bool,+left: U32,+right: U32,+total: U32,+classified: {U32.is_le(right,(4294967295 - left : U32)) == valid : Bool},
  found: {Shapes.sum_when(valid,(left + right : U32)) == Some{total} : Maybe<&2,U32>}) ->
  {Nat.add(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(total) : Nat}:
  match valid:
    case False{}:
      Empty.absurd({Nat.add(U32.to_nat(left),U32.to_nat(right)) == U32.to_nat(total) : Nat},
        Words.false_true(Equal.cong(Maybe<&2,U32>,Bool,value => present(value),None{},Some{total},found)))
    case True{}:
      +value = Equal.cong(Maybe<&2,U32>,U32,value => known(value),Some{(left + right : U32)},Some{total},found)
      Equal.trans(Nat,Nat.add(U32.to_nat(left),U32.to_nat(right)),U32.to_nat((left + right : U32)),U32.to_nat(total),
        Equal.sym(Nat,U32.to_nat((left + right : U32)),Nat.add(U32.to_nat(left),U32.to_nat(right)),FullRange.add(left,right,4294967295,word_room(left,right,4294967295,{==},classified))),
        Equal.cong(U32,Nat,word => U32.to_nat(word),(left + right : U32),total,value))

# An axis with front elements before it and back behind it, as the runtime
# checks it.
def widened_from(first: Maybe<&2,U32>,+front: U32,+extent: U32,+back: U32,+total: U32,+first_exact: {Shapes.checked_add(front,extent) == first : Maybe<&2,U32>},
  computed: {Shapes.checked_add_to(first,back) == Some{total} : Maybe<&2,U32>}) ->
  {Nat.add(Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(back)) == U32.to_nat(total) : Nat}:
  match first:
    case None{}:
      Empty.absurd({Nat.add(Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(back)) == U32.to_nat(total) : Nat},
        Words.false_true(Equal.cong(Maybe<&2,U32>,Bool,value => present(value),None{},Some{total},computed)))
    case Some{+sum}:
      +inner = checked_sum(U32.is_le(extent,(4294967295 - front : U32)),front,extent,sum,{==},first_exact)
      +outer = checked_sum(U32.is_le(back,(4294967295 - sum : U32)),sum,back,total,{==},computed)
      Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(back)),Nat.add(U32.to_nat(sum),U32.to_nat(back)),U32.to_nat(total),
        Equal.cong(Nat,Nat,number => Nat.add(number,U32.to_nat(back)),Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(sum),inner),outer)

def present_extents(found: Maybe<&2,List<&2,U32>>) -> Bool:
  match found:
    case Some{extents}: True{}
    case None{}: False{}

def known_extents(found: Maybe<&2,List<&2,U32>>) -> List<&2,U32>:
  match found:
    case Some{+extents}: extents
    case None{}: []

# The padded extents of an axis and the axes after it.
type Prepended<-front: U32,-extent: U32,-back: U32,-extents: List<&2,U32>,-before: List<&2,U32>,-after: List<&2,U32>,-padded: List<&2,U32>> is Data:
  Prepended{total: U32,tail: List<&2,U32>,
    widened: {Nat.add(Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(back)) == U32.to_nat(total) : Nat},
    tail_exact: {Shapes.padded(extents,before,after) == Some{tail} : Maybe<&2,List<&2,U32>>},padded_exact: {total <> tail == padded : List<&2,U32>}}

def prepended(total_found: Maybe<&2,U32>,tail_found: Maybe<&2,List<&2,U32>>,+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,
  +after: List<&2,U32>,+padded: List<&2,U32>,+total_exact: {Shapes.checked_add_to(Shapes.checked_add(front,extent),back) == total_found : Maybe<&2,U32>},
  +tail_exact: {Shapes.padded(extents,before,after) == tail_found : Maybe<&2,List<&2,U32>>},
  +padded_exact: {Shapes.prepend_extent(total_found,tail_found) == Some{padded} : Maybe<&2,List<&2,U32>>}) -> Prepended<front,extent,back,extents,before,after,padded>:
  match total_found:
    case None{}:
      Empty.absurd(Prepended<front,extent,back,extents,before,after,padded>,
        Words.false_true(Equal.cong(Maybe<&2,List<&2,U32>>,Bool,found => present_extents(found),None{},Some{padded},padded_exact)))
    case Some{+total}:
      match tail_found:
        case None{}:
          Empty.absurd(Prepended<front,extent,back,extents,before,after,padded>,
            Words.false_true(Equal.cong(Maybe<&2,List<&2,U32>>,Bool,found => present_extents(found),None{},Some{padded},padded_exact)))
        case Some{+tail}:
          Prepended{total,tail,widened_from(Shapes.checked_add(front,extent),front,extent,back,total,{==},total_exact),tail_exact,
            Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => known_extents(found),Some{total <> tail},Some{padded},padded_exact)}

def prepended_total(+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,
  +found: Prepended<front,extent,back,extents,before,after,padded>) -> U32:
  Prepended{+total,+tail,+widened,+tail_exact,+padded_exact} = found
  total

def prepended_tail(+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,
  +found: Prepended<front,extent,back,extents,before,after,padded>) -> List<&2,U32>:
  Prepended{+total,+tail,+widened,+tail_exact,+padded_exact} = found
  tail

def prepended_widened(+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,
  +found: Prepended<front,extent,back,extents,before,after,padded>) ->
  {Nat.add(Nat.add(U32.to_nat(front),U32.to_nat(extent)),U32.to_nat(back)) == U32.to_nat(prepended_total(front,extent,back,extents,before,after,padded,found)) : Nat}:
  Prepended{+total,+tail,+widened,+tail_exact,+padded_exact} = found
  widened

def prepended_tail_exact(+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,
  +found: Prepended<front,extent,back,extents,before,after,padded>) ->
  {Shapes.padded(extents,before,after) == Some{prepended_tail(front,extent,back,extents,before,after,padded,found)} : Maybe<&2,List<&2,U32>>}:
  Prepended{+total,+tail,+widened,+tail_exact,+padded_exact} = found
  tail_exact

def prepended_exact(+front: U32,+extent: U32,+back: U32,+extents: List<&2,U32>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>,
  +found: Prepended<front,extent,back,extents,before,after,padded>) ->
  {prepended_total(front,extent,back,extents,before,after,padded,found) <> prepended_tail(front,extent,back,extents,before,after,padded,found) == padded : List<&2,U32>}:
  Prepended{+total,+tail,+widened,+tail_exact,+padded_exact} = found
  padded_exact

def unpadded(+padded: List<&2,U32>,exact: {None{} == Some{padded} : Maybe<&2,List<&2,U32>>}) -> Empty:
  Words.false_true(Equal.cong(Maybe<&2,List<&2,U32>>,Bool,found => present_extents(found),None{},Some{padded},exact))

# Every axis of a tensor fits its padded axis from the padding in front on.
def padded_in(extents: List<&2,U32>,before: List<&2,U32>,after: List<&2,U32>,+padded: List<&2,U32>,
  +exact: {Shapes.padded(extents,before,after) == Some{padded} : Maybe<&2,List<&2,U32>>}) -> Facts.PlacedIn(padded,extents,before):
  match extents before after:
    case Nil{} Nil{} Nil{}:
      %Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,found => known_extents(found),Some{[]},Some{padded},exact) : Facts.PlacedIn(_,[],[])
      Unit{}
    case Con{+extent,+rest} Con{+front,+fronts} Con{+back,+backs}:
      +found = prepended(Shapes.checked_add_to(Shapes.checked_add(front,extent),back),Shapes.padded(rest,fronts,backs),front,extent,back,rest,fronts,backs,padded,
        {==},{==},exact)
      +total = prepended_total(front,extent,back,rest,fronts,backs,padded,found)
      +tail = prepended_tail(front,extent,back,rest,fronts,backs,padded,found)
      +sum = Nat.add(U32.to_nat(front),U32.to_nat(extent))
      %prepended_exact(front,extent,back,rest,fronts,backs,padded,found) : Facts.PlacedIn(_,Con{extent,rest},Con{front,fronts})
      Held.Both{{%prepended_widened(front,extent,back,rest,fronts,backs,padded,found) : {Nat.is_le(sum,_) == True{} : Bool}
          Order.le_add(sum,U32.to_nat(back)) : {Nat.is_le(sum,U32.to_nat(total)) == True{} : Bool}},
        padded_in(rest,fronts,backs,tail,prepended_tail_exact(front,extent,back,rest,fronts,backs,padded,found))}
    case Con{+extent,+rest} Nil{} +after:
      Empty.absurd(Facts.PlacedIn(padded,Con{extent,rest},Nil{}),unpadded(padded,exact))
    case Con{+extent,+rest} Con{+front,+fronts} Nil{}:
      Empty.absurd(Facts.PlacedIn(padded,Con{extent,rest},Con{front,fronts}),unpadded(padded,exact))
    case Nil{} Con{+front,+fronts} +after:
      Empty.absurd(Facts.PlacedIn(padded,Nil{},Con{front,fronts}),unpadded(padded,exact))
    case Nil{} Nil{} Con{+back,+backs}:
      Empty.absurd(Facts.PlacedIn(padded,Nil{},Nil{}),unpadded(padded,exact))
