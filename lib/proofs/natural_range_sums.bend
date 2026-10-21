# Finite Nat sums use the existing range, map and list-fold skeletons.
import Base
import ./ordered_ranges.bend as Ranges
import ./ordered_reduction_split.bend as Ordered
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ../traversal_array.bend as Traversal
import ./word_arithmetic.bend as WordArithmetic

def fold(values: List<&2,Nat>) -> Nat:
  List.foldl(~&2,~Nat,~Nat,~Nat.add,values,0n)

def fold_shift(values: List<&2,Nat>,+first: Nat,+second: Nat) ->
  {List.foldl(~&2,~Nat,~Nat,~Nat.add,values,Nat.add(first,second)) == Nat.add(first,List.foldl(~&2,~Nat,~Nat,~Nat.add,values,second)) : Nat}:
  match values:
    case Nil{}: {==}
    case +head <> rest:
      %Equal.sym(Nat,Nat.add(Nat.add(first,second),head),Nat.add(first,Nat.add(second,head)),Addition.associative(first,second,head)) :
        {List.foldl(~&2,~Nat,~Nat,~Nat.add,rest,_) == Nat.add(first,List.foldl(~&2,~Nat,~Nat,~Nat.add,rest,Nat.add(second,head))) : Nat}
      fold_shift(rest,first,Nat.add(second,head))

def fold_cons(+head: Nat,+tail: List<&2,Nat>) -> {fold(head <> tail) == Nat.add(head,fold(tail)) : Nat}:
  %Addition.zero_right(head) :
    {List.foldl(~&2,~Nat,~Nat,~Nat.add,tail,_) == Nat.add(head,fold(tail)) : Nat}
  fold_shift(tail,head,0n)

def sum(~Context: Data,~sample: Context -> Nat -> Nat,count: Nat,start: Nat,context: Context) -> Nat:
  fold(Ordered.map_keys(~Nat,~Nat,~Context,~sample,Ranges.range(count,start),context))

def successor(~Context: Data,~sample: Context -> Nat -> Nat,+count: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~sample,1n+count,start,context) == Nat.add(sample(context,start),sum(~Context,~sample,count,1n+start,context)) : Nat}:
  fold_cons(sample(context,start),Ordered.map_keys(~Nat,~Nat,~Context,~sample,Ranges.range(count,1n+start),context))

def split(~Context: Data,~sample: Context -> Nat -> Nat,first: Nat,+second: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~sample,Nat.add(first,second),start,context) ==
    Nat.add(sum(~Context,~sample,first,start,context),sum(~Context,~sample,second,Nat.add(start,first),context)) : Nat}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {sum(~Context,~sample,second,start,context) == sum(~Context,~sample,second,_,context) : Nat}
      {==}
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {sum(~Context,~sample,1n+Nat.add(rest,second),start,context) == Nat.add(sum(~Context,~sample,1n+rest,start,context),sum(~Context,~sample,second,_,context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~sample,1n+Nat.add(rest,second),start,context),Nat.add(sample(context,start),sum(~Context,~sample,Nat.add(rest,second),1n+start,context)),successor(~Context,~sample,Nat.add(rest,second),start,context)) :
        {_ == Nat.add(sum(~Context,~sample,1n+rest,start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~sample,Nat.add(rest,second),1n+start,context),
        Nat.add(sum(~Context,~sample,rest,1n+start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context)),split(~Context,~sample,rest,second,1n+start,context)) :
        {Nat.add(sample(context,start),_) == Nat.add(sum(~Context,~sample,1n+rest,start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~sample,1n+rest,start,context),Nat.add(sample(context,start),sum(~Context,~sample,rest,1n+start,context)),successor(~Context,~sample,rest,start,context)) :
        {Nat.add(sample(context,start),Nat.add(sum(~Context,~sample,rest,1n+start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context))) == Nat.add(_,sum(~Context,~sample,second,1n+Nat.add(start,rest),context)) : Nat}
      Equal.sym(Nat,Nat.add(Nat.add(sample(context,start),sum(~Context,~sample,rest,1n+start,context)),sum(~Context,~sample,second,1n+Nat.add(start,rest),context)),
        Nat.add(sample(context,start),Nat.add(sum(~Context,~sample,rest,1n+start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context))),
        Addition.associative(sample(context,start),sum(~Context,~sample,rest,1n+start,context),sum(~Context,~sample,second,1n+Nat.add(start,rest),context)))

def agrees(~Context: Data,~limit: Context -> Nat,~left: Context -> Nat -> Nat,~right: Context -> Nat -> Nat,
  ~same: @+context: Context -> @+index: Nat -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> {left(context,index) == right(context,index) : Nat},
  remaining: Nat,+start: Nat,+context: Context,+room: {Nat.is_le(Nat.add(start,remaining),limit(context)) == True{} : Bool}) ->
  {sum(~Context,~left,remaining,start,context) == sum(~Context,~right,remaining,start,context) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +left_tail = sum(~Context,~left,rest,1n+start,context)
      +right_tail = sum(~Context,~right,rest,1n+start,context)
      %Equal.sym(Nat,sum(~Context,~left,1n+rest,start,context),Nat.add(left(context,start),left_tail),successor(~Context,~left,rest,start,context)) :
        {_ == sum(~Context,~right,1n+rest,start,context) : Nat}
      %Equal.sym(Nat,sum(~Context,~right,1n+rest,start,context),Nat.add(right(context,start),right_tail),successor(~Context,~right,rest,start,context)) :
        {Nat.add(left(context,start),left_tail) == _ : Nat}
      Equal.trans(Nat,Nat.add(left(context,start),left_tail),Nat.add(right(context,start),left_tail),Nat.add(right(context,start),right_tail),
        Equal.cong(Nat,Nat,head => Nat.add(head,left_tail),left(context,start),right(context,start),same(context,start,Order.room_index(start,rest,limit(context),room))),
        Equal.cong(Nat,Nat,tail => Nat.add(right(context,start),tail),left_tail,right_tail,agrees(~Context,~limit,~left,~right,~same,rest,1n+start,context,Order.advance_room(start,rest,limit(context),room))))

def added(~Context: Data,~left: Context -> Nat -> Nat,~right: Context -> Nat -> Nat,context: Context,index: Nat) -> Nat:
  +context = context
  +index = index
  Nat.add(left(context,index),right(context,index))

def add_four(+a: Nat,+b: Nat,+c: Nat,+d: Nat) -> {Nat.add(Nat.add(a,b),Nat.add(c,d)) == Nat.add(Nat.add(a,c),Nat.add(b,d)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(a,b),Nat.add(c,d)),Nat.add(a,Nat.add(b,Nat.add(c,d))),Nat.add(Nat.add(a,c),Nat.add(b,d)),
    Addition.associative(a,b,Nat.add(c,d)),
    Equal.trans(Nat,Nat.add(a,Nat.add(b,Nat.add(c,d))),Nat.add(a,Nat.add(c,Nat.add(b,d))),Nat.add(Nat.add(a,c),Nat.add(b,d)),
      Equal.cong(Nat,Nat,tail => Nat.add(a,tail),Nat.add(b,Nat.add(c,d)),Nat.add(c,Nat.add(b,d)),RowMajor.add_swap(b,c,d)),
      Equal.sym(Nat,Nat.add(Nat.add(a,c),Nat.add(b,d)),Nat.add(a,Nat.add(c,Nat.add(b,d))),Addition.associative(a,c,Nat.add(b,d)))))

def addition(~Context: Data,~left: Context -> Nat -> Nat,~right: Context -> Nat -> Nat,remaining: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~added(~Context,~left,~right),remaining,start,context) == Nat.add(sum(~Context,~left,remaining,start,context),sum(~Context,~right,remaining,start,context)) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +left_tail = sum(~Context,~left,rest,1n+start,context)
      +right_tail = sum(~Context,~right,rest,1n+start,context)
      %Equal.sym(Nat,sum(~Context,~added(~Context,~left,~right),1n+rest,start,context),
        Nat.add(Nat.add(left(context,start),right(context,start)),sum(~Context,~added(~Context,~left,~right),rest,1n+start,context)),successor(~Context,~added(~Context,~left,~right),rest,start,context)) :
        {_ == Nat.add(sum(~Context,~left,1n+rest,start,context),sum(~Context,~right,1n+rest,start,context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~left,1n+rest,start,context),Nat.add(left(context,start),left_tail),successor(~Context,~left,rest,start,context)) :
        {Nat.add(Nat.add(left(context,start),right(context,start)),sum(~Context,~added(~Context,~left,~right),rest,1n+start,context)) == Nat.add(_,sum(~Context,~right,1n+rest,start,context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~right,1n+rest,start,context),Nat.add(right(context,start),right_tail),successor(~Context,~right,rest,start,context)) :
        {Nat.add(Nat.add(left(context,start),right(context,start)),sum(~Context,~added(~Context,~left,~right),rest,1n+start,context)) == Nat.add(Nat.add(left(context,start),left_tail),_) : Nat}
      %Equal.sym(Nat,sum(~Context,~added(~Context,~left,~right),rest,1n+start,context),Nat.add(left_tail,right_tail),addition(~Context,~left,~right,rest,1n+start,context)) :
        {Nat.add(Nat.add(left(context,start),right(context,start)),_) == Nat.add(Nat.add(left(context,start),left_tail),Nat.add(right(context,start),right_tail)) : Nat}
      add_four(left(context,start),right(context,start),left_tail,right_tail)

def constant(~Context: Data,~value: Context -> Nat,context: Context,index: Nat) -> Nat:
  value(context)

def repeated(~Context: Data,~value: Context -> Nat,remaining: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~constant(~Context,~value),remaining,start,context) == Nat.mul(remaining,value(context)) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Nat,sum(~Context,~constant(~Context,~value),1n+rest,start,context),Nat.add(value(context),sum(~Context,~constant(~Context,~value),rest,1n+start,context)),
        successor(~Context,~constant(~Context,~value),rest,start,context)) : {_ == Nat.add(value(context),Nat.mul(rest,value(context))) : Nat}
      Equal.cong(Nat,Nat,tail => Nat.add(value(context),tail),sum(~Context,~constant(~Context,~value),rest,1n+start,context),Nat.mul(rest,value(context)),repeated(~Context,~value,rest,1n+start,context))

def scaled(~Context: Data,~sample: Context -> Nat -> Nat,~factor: Context -> Nat,context: Context,index: Nat) -> Nat:
  +context = context
  Nat.mul(factor(context),sample(context,index))

def scaling(~Context: Data,~sample: Context -> Nat -> Nat,~factor: Context -> Nat,remaining: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~scaled(~Context,~sample,~factor),remaining,start,context) == Nat.mul(factor(context),sum(~Context,~sample,remaining,start,context)) : Nat}:
  match remaining:
    case 0n: Equal.sym(Nat,Nat.mul(factor(context),0n),0n,RowMajor.multiply_zero_right(factor(context)))
    case 1n+ +rest:
      +tail = sum(~Context,~sample,rest,1n+start,context)
      %Equal.sym(Nat,sum(~Context,~scaled(~Context,~sample,~factor),1n+rest,start,context),
        Nat.add(Nat.mul(factor(context),sample(context,start)),sum(~Context,~scaled(~Context,~sample,~factor),rest,1n+start,context)),successor(~Context,~scaled(~Context,~sample,~factor),rest,start,context)) :
        {_ == Nat.mul(factor(context),sum(~Context,~sample,1n+rest,start,context)) : Nat}
      %Equal.sym(Nat,sum(~Context,~sample,1n+rest,start,context),Nat.add(sample(context,start),tail),successor(~Context,~sample,rest,start,context)) :
        {Nat.add(Nat.mul(factor(context),sample(context,start)),sum(~Context,~scaled(~Context,~sample,~factor),rest,1n+start,context)) == Nat.mul(factor(context),_) : Nat}
      %Equal.sym(Nat,sum(~Context,~scaled(~Context,~sample,~factor),rest,1n+start,context),Nat.mul(factor(context),tail),scaling(~Context,~sample,~factor,rest,1n+start,context)) :
        {Nat.add(Nat.mul(factor(context),sample(context,start)),_) == Nat.mul(factor(context),Nat.add(sample(context,start),tail)) : Nat}
      Equal.sym(Nat,Nat.mul(factor(context),Nat.add(sample(context,start),tail)),Nat.add(Nat.mul(factor(context),sample(context,start)),Nat.mul(factor(context),tail)),RowMajor.right_distrib(factor(context),sample(context,start),tail))

def prefix_bounded(~Context: Data,~sample: Context -> Nat -> Nat,+first: Nat,+whole: Nat,+start: Nat,+context: Context,
  room: {Nat.is_le(first,whole) == True{} : Bool}) ->
  {Nat.is_le(sum(~Context,~sample,first,start,context),sum(~Context,~sample,whole,start,context)) == True{} : Bool}:
  %Order.restore_shift(first,whole,room) : {Nat.is_le(sum(~Context,~sample,first,start,context),sum(~Context,~sample,_,start,context)) == True{} : Bool}
  %Equal.sym(Nat,sum(~Context,~sample,Nat.add(first,Nat.sub(whole,first)),start,context),
    Nat.add(sum(~Context,~sample,first,start,context),sum(~Context,~sample,Nat.sub(whole,first),Nat.add(start,first),context)),split(~Context,~sample,first,Nat.sub(whole,first),start,context)) :
    {Nat.is_le(sum(~Context,~sample,first,start,context),_) == True{} : Bool}
  Order.le_add(sum(~Context,~sample,first,start,context),sum(~Context,~sample,Nat.sub(whole,first),Nat.add(start,first),context))

def shifted(~Context: Data,~sample: Context -> Nat -> Nat,~offset: Context -> Nat,context: Context,index: Nat) -> Nat:
  +context = context
  sample(context,Nat.add(offset(context),index))

def shifting(~Context: Data,~sample: Context -> Nat -> Nat,~offset: Context -> Nat,remaining: Nat,+start: Nat,+context: Context) ->
  {sum(~Context,~shifted(~Context,~sample,~offset),remaining,start,context) == sum(~Context,~sample,remaining,Nat.add(offset(context),start),context) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +first = Nat.add(offset(context),start)
      %Equal.sym(Nat,sum(~Context,~shifted(~Context,~sample,~offset),1n+rest,start,context),
        Nat.add(sample(context,first),sum(~Context,~shifted(~Context,~sample,~offset),rest,1n+start,context)),successor(~Context,~shifted(~Context,~sample,~offset),rest,start,context)) :
        {_ == sum(~Context,~sample,1n+rest,first,context) : Nat}
      %Equal.sym(Nat,sum(~Context,~sample,1n+rest,first,context),Nat.add(sample(context,first),sum(~Context,~sample,rest,1n+first,context)),successor(~Context,~sample,rest,first,context)) :
        {Nat.add(sample(context,first),sum(~Context,~shifted(~Context,~sample,~offset),rest,1n+start,context)) == _ : Nat}
      %Addition.successor_right(offset(context),start) :
        {Nat.add(sample(context,first),sum(~Context,~shifted(~Context,~sample,~offset),rest,1n+start,context)) == Nat.add(sample(context,first),sum(~Context,~sample,rest,_,context)) : Nat}
      Equal.cong(Nat,Nat,tail => Nat.add(sample(context,first),tail),sum(~Context,~shifted(~Context,~sample,~offset),rest,1n+start,context),sum(~Context,~sample,rest,Nat.add(offset(context),1n+start),context),shifting(~Context,~sample,~offset,rest,1n+start,context))

def project_context(~Outer: Data,~Inner: Data,~project: Outer -> Inner,~left: Outer -> Nat -> Nat,~right: Inner -> Nat -> Nat,
  ~same: @+context: Outer -> @+index: Nat -> {left(context,index) == right(project(context),index) : Nat},
  remaining: Nat,+start: Nat,+context: Outer) ->
  {sum(~Outer,~left,remaining,start,context) == sum(~Inner,~right,remaining,start,project(context)) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +left_tail = sum(~Outer,~left,rest,1n+start,context)
      +right_tail = sum(~Inner,~right,rest,1n+start,project(context))
      %Equal.sym(Nat,sum(~Outer,~left,1n+rest,start,context),Nat.add(left(context,start),left_tail),successor(~Outer,~left,rest,start,context)) :
        {_ == sum(~Inner,~right,1n+rest,start,project(context)) : Nat}
      %Equal.sym(Nat,sum(~Inner,~right,1n+rest,start,project(context)),Nat.add(right(project(context),start),right_tail),successor(~Inner,~right,rest,start,project(context))) :
        {Nat.add(left(context,start),left_tail) == _ : Nat}
      Equal.trans(Nat,Nat.add(left(context,start),left_tail),Nat.add(right(project(context),start),left_tail),Nat.add(right(project(context),start),right_tail),
        Equal.cong(Nat,Nat,head => Nat.add(head,left_tail),left(context,start),right(project(context),start),same(context,start)),
        Equal.cong(Nat,Nat,tail => Nat.add(right(project(context),start),tail),left_tail,right_tail,project_context(~Outer,~Inner,~project,~left,~right,~same,rest,1n+start,context)))

# Native addition follows the same sum modulo U32. Exact totals use roundtrip.
def encoded_update(~Context: Data,~native: Context -> U32 -> U32,position: U32,context: Context,value: U32) -> U32:
  (value + native(context,position) : U32)

def encoded(~Context: Data,~limit: Context -> Nat,~sample: Context -> Nat -> Nat,~native: Context -> U32 -> U32,
  ~same: @+context: Context -> @+index: Nat -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> {native(context,U32.from_nat(index)) == U32.from_nat(sample(context,index)) : U32},
  remaining: Nat,+start: Nat,+context: Context,+initial: Nat,+room: {Nat.is_le(Nat.add(start,remaining),limit(context)) == True{} : Bool}) ->
  {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),remaining,U32.from_nat(start),context,U32.from_nat(initial)) ==
    U32.from_nat(Nat.add(initial,sum(~Context,~sample,remaining,start,context))) : U32}:
  match remaining:
    case 0n:
      %Equal.sym(Nat,Nat.add(initial,0n),initial,Addition.zero_right(initial)) : {U32.from_nat(initial) == U32.from_nat(_) : U32}
      {==}
    case 1n+ +rest:
      +value = sample(context,start)
      +tail = sum(~Context,~sample,rest,1n+start,context)
      %Equal.sym(U32,native(context,U32.from_nat(start)),U32.from_nat(value),same(context,start,Order.room_index(start,rest,limit(context),room))) :
        {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),rest,(U32.from_nat(start) + 1 : U32),context,(U32.from_nat(initial) + _ : U32)) == U32.from_nat(Nat.add(initial,sum(~Context,~sample,1n+rest,start,context))) : U32}
      %Equal.sym(U32,(U32.from_nat(initial) + U32.from_nat(value) : U32),U32.from_nat(Nat.add(initial,value)),WordArithmetic.native_add(initial,value)) :
        {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),rest,(U32.from_nat(start) + 1 : U32),context,_) == U32.from_nat(Nat.add(initial,sum(~Context,~sample,1n+rest,start,context))) : U32}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),rest,_,context,U32.from_nat(Nat.add(initial,value))) == U32.from_nat(Nat.add(initial,sum(~Context,~sample,1n+rest,start,context))) : U32}
      %Equal.sym(Nat,sum(~Context,~sample,1n+rest,start,context),Nat.add(value,tail),successor(~Context,~sample,rest,start,context)) :
        {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),rest,U32.from_nat(1n+start),context,U32.from_nat(Nat.add(initial,value))) == U32.from_nat(Nat.add(initial,_)) : U32}
      %Addition.associative(initial,value,tail) :
        {Traversal.indexed_fold(~U32,~Context,~encoded_update(~Context,~native),rest,U32.from_nat(1n+start),context,U32.from_nat(Nat.add(initial,value))) == U32.from_nat(_) : U32}
      encoded(~Context,~limit,~sample,~native,~same,rest,1n+start,context,Nat.add(initial,value),Order.advance_room(start,rest,limit(context),room))
