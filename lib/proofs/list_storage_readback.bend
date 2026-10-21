# Finite list writes preserve earlier addresses and read back every list item.
# Allocation bounds come from the real public count, never example shapes.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as Readback
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as Words
import ./nat_interval.bend as Interval
import ./nat_division.bend as Division
import ./loop_projection.bend as Projection
import ./bounded_u32_arithmetic.bend as NativeArithmetic

def end_position(~Element: Data,values: List<&2,Element>,offset: U32) -> U32:
  match values:
    case Nil{}: offset
    case head <> rest: end_position(~Element,rest,(offset + 1 : U32))

def end_position_length(~Element: Data,values: List<&2,Element>,+offset: U32) ->
  {end_position(~Element,values,offset) == (offset + U32.from_nat(List.length(&2,Element,values)) : U32) : U32}:
  match values:
    case Nil{}: Equal.sym(U32,(offset + 0 : U32),offset,NativeArithmetic.add_zero(offset))
    case head <> +rest:
      Equal.trans(U32,end_position(~Element,rest,(offset + 1 : U32)),((offset + 1 : U32) + U32.from_nat(List.length(&2,Element,rest)) : U32),
        (offset + U32.from_nat(1n+List.length(&2,Element,rest)) : U32),end_position_length(~Element,rest,(offset + 1 : U32)),
        Equal.sym(U32,(offset + U32.from_nat(1n+List.length(&2,Element,rest)) : U32),((offset + 1 : U32) + U32.from_nat(List.length(&2,Element,rest)) : U32),
          Projection.indexed_successor(offset,List.length(&2,Element,rest))))

def append_length(~Element: Data,left: List<&2,Element>,+right: List<&2,Element>) ->
  {List.length(&2,Element,List.append(&2,Element,left,right)) == Nat.add(List.length(&2,Element,left),List.length(&2,Element,right)) : Nat}:
  match left:
    case Nil{}: {==}
    case head <> +rest:
      %append_length(~Element,rest,right) : {1n+List.length(&2,Element,List.append(&2,Element,rest,right)) == 1n+_ : Nat}
      {==}

def concatenated(~Element: Data,+first: List<&2,Element>,+second: List<&2,Element>,+offset: U32,+tree: Tree.Tree<Element>) ->
  {Tree.write_sequence(~Element,List.append(&2,Element,first,second),offset,tree) ==
    Tree.write_sequence(~Element,second,end_position(~Element,first,offset),Tree.write_sequence(~Element,first,offset,tree)) : Tree.Tree<Element>}:
  match first:
    case Nil{}: {==}
    case +head <> +rest: concatenated(~Element,rest,second,(offset + 1 : U32),Tree.write(Element,tree,offset,head))

def at(~Element: Data,+fallback: Element,index: Nat,values: List<&2,Element>) -> Element:
  match index values:
    case 0n Nil{}: fallback
    case 0n head <> tail: head
    case 1n+rest Nil{}: fallback
    case 1n+rest head <> tail: at(~Element,fallback,rest,tail)

# Generic ordered enumeration; the parameter bridge keeps its F32 wrappers.
def enumerate(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,remaining: Nat,+first: Nat,+context: Context) -> List<&2,Element>:
  match remaining:
    case 0n: []
    case 1n+rest: value(context,first) <> enumerate(~Element,~Context,~value,rest,1n+first,context)

def enumerate_length(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,remaining: Nat,+first: Nat,+context: Context) -> {List.length(&2,Element,enumerate(~Element,~Context,~value,remaining,first,context)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %enumerate_length(~Element,~Context,~value,rest,1n+first,context) : {1n+List.length(&2,Element,enumerate(~Element,~Context,~value,rest,1n+first,context)) == 1n+_ : Nat}
      {==}

def enumerate_at(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,+fallback: Element,remaining: Nat,index: Nat,+first: Nat,+context: Context,within: {Nat.is_lt(index,remaining) == True{} : Bool}) ->
  {at(~Element,fallback,index,enumerate(~Element,~Context,~value,remaining,first,context)) == value(context,Nat.add(first,index)) : Element}:
  match remaining index:
    case 0n 0n: Empty.absurd({fallback == value(context,Nat.add(first,0n)) : Element},Words.false_true(within))
    case 0n 1n+index: Empty.absurd({fallback == value(context,Nat.add(first,1n+index)) : Element},Words.false_true(within))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : {value(context,first) == value(context,_) : Element}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(first,1n+previous),1n+Nat.add(first,previous),Addition.successor_right(first,previous)) : {at(~Element,fallback,previous,enumerate(~Element,~Context,~value,rest,1n+first,context)) == value(context,_) : Element}
      enumerate_at(~Element,~Context,~value,fallback,rest,previous,1n+first,context,within)

def inside(+index: Nat,+capacity: U32,+bound: {Nat.is_lt(index,U32.to_nat(capacity)) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(index),capacity) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),capacity),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(capacity)),Comparison.lt(U32.from_nat(index),capacity)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),bound))) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  bound

def distinct(+first: Nat,+second: Nat,+capacity: U32,+first_bound: {Nat.is_lt(first,U32.to_nat(capacity)) == True{} : Bool},+second_bound: {Nat.is_lt(second,U32.to_nat(capacity)) == True{} : Bool},ordered: {Nat.is_lt(second,first) == True{} : Bool}) -> {U32.is_eq(U32.from_nat(first),U32.from_nat(second)) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(first),U32.from_nat(second)),Nat.is_eq(U32.to_nat(U32.from_nat(first)),U32.to_nat(U32.from_nat(second))),Comparison.eq(U32.from_nat(first),U32.from_nat(second))) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(first)),first,FullRange.roundtrip(first,capacity,Bounds.strict_le(first,U32.to_nat(capacity),first_bound))) : {Nat.is_eq(_,U32.to_nat(U32.from_nat(second))) == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(second)),second,FullRange.roundtrip(second,capacity,Bounds.strict_le(second,U32.to_nat(capacity),second_bound))) : {Nat.is_eq(first,_) == False{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(first,second),Nat.is_eq(second,first),Order.equal_symmetric(first,second)) : {_ == False{} : Bool}
  Order.less_not_equal(second,first,ordered)

def preserved(~Element: Data,values: List<&2,Element>,+offset: Nat,+query: Nat,+depth: Nat,+tree: Tree.Tree<Element>,+balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(offset,List.length(&2,Element,values)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},+before: {Nat.is_lt(query,offset) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write_sequence(~Element,values,U32.from_nat(offset),tree),U32.from_nat(query)) == Tree.read(Element,tree,U32.from_nat(query)) : Element}:
  match values:
    case []: {==}
    case +head <> +rest:
      +index_inside = Order.room_index(offset,List.length(&2,Element,rest),U32.to_nat(Tree.capacity(Element,tree)),room)
      +written = Tree.write(Element,tree,U32.from_nat(offset),head)
      +capacity = Readback.write_capacity(~Element,tree,U32.from_nat(offset),head)
      %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.increment_encoded(offset)) :
        {Tree.read(Element,Tree.write_sequence(~Element,rest,_,written),U32.from_nat(query)) == Tree.read(Element,tree,U32.from_nat(query)) : Element}
      Equal.trans(Element,Tree.read(Element,Tree.write_sequence(~Element,rest,U32.from_nat(1n+offset),written),U32.from_nat(query)),Tree.read(Element,written,U32.from_nat(query)),Tree.read(Element,tree,U32.from_nat(query)),
        preserved(~Element,rest,1n+offset,query,depth,written,Shape.write(Element,depth,tree,U32.from_nat(offset),head,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity) : {Nat.is_le(Nat.add(1n+offset,List.length(&2,Element,rest)),U32.to_nat(_)) == True{} : Bool}
          Order.advance_room(offset,List.length(&2,Element,rest),U32.to_nat(Tree.capacity(Element,tree)),room),
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity) : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool} query_inside,
          Order.lt_succ(query,offset,Bounds.strict_le(query,offset,before))),
        Readback.read_preserved(~Element,depth,tree,U32.from_nat(offset),U32.from_nat(query),head,balanced,limit,inside(offset,Tree.capacity(Element,tree),index_inside),inside(query,Tree.capacity(Element,tree),query_inside),distinct(offset,query,Tree.capacity(Element,tree),index_inside,query_inside,before)))

def outside(~Element: Data,values: List<&2,Element>,+offset: Nat,+query: Nat,+depth: Nat,+tree: Tree.Tree<Element>,+balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(offset,List.length(&2,Element,values)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +separate: Interval.Outside<offset,List.length(&2,Element,values),query>) ->
  {Tree.read(Element,Tree.write_sequence(~Element,values,U32.from_nat(offset),tree),U32.from_nat(query)) == Tree.read(Element,tree,U32.from_nat(query)) : Element}:
  match values:
    case Nil{}: {==}
    case +head <> +rest:
      +capacity = Tree.capacity(Element,tree)
      +index_inside = Order.room_index(offset,List.length(&2,Element,rest),U32.to_nat(capacity),room)
      +written = Tree.write(Element,tree,U32.from_nat(offset),head)
      +same_capacity = Readback.write_capacity(~Element,tree,U32.from_nat(offset),head)
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(offset),U32.from_nat(query)),Nat.is_eq(U32.to_nat(U32.from_nat(offset)),U32.to_nat(U32.from_nat(query))),Comparison.eq(U32.from_nat(offset),U32.from_nat(query))) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(offset)),offset,FullRange.roundtrip(offset,capacity,Bounds.strict_le(offset,U32.to_nat(capacity),index_inside))) : {Nat.is_eq(_,U32.to_nat(U32.from_nat(query))) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(query)),query,FullRange.roundtrip(query,capacity,Bounds.strict_le(query,U32.to_nat(capacity),query_inside))) : {Nat.is_eq(offset,_) == False{} : Bool}
        %Order.equal_symmetric(query,offset) : {_ == False{} : Bool}
        Interval.outside_head(offset,List.length(&2,Element,rest),query,separate) : {U32.is_eq(U32.from_nat(offset),U32.from_nat(query)) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.increment_encoded(offset)) :
        {Tree.read(Element,Tree.write_sequence(~Element,rest,_,written),U32.from_nat(query)) == Tree.read(Element,tree,U32.from_nat(query)) : Element}
      Equal.trans(Element,Tree.read(Element,Tree.write_sequence(~Element,rest,U32.from_nat(1n+offset),written),U32.from_nat(query)),Tree.read(Element,written,U32.from_nat(query)),Tree.read(Element,tree,U32.from_nat(query)),
        outside(~Element,rest,1n+offset,query,depth,written,Shape.write(Element,depth,tree,U32.from_nat(offset),head,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),capacity,same_capacity) : {Nat.is_le(Nat.add(1n+offset,List.length(&2,Element,rest)),U32.to_nat(_)) == True{} : Bool}
          Order.advance_room(offset,List.length(&2,Element,rest),U32.to_nat(capacity),room),
          %Equal.sym(U32,Tree.capacity(Element,written),capacity,same_capacity) : {Nat.is_lt(query,U32.to_nat(_)) == True{} : Bool} query_inside,
          Interval.outside_tail(offset,List.length(&2,Element,rest),query,separate)),
        Readback.read_preserved(~Element,depth,tree,U32.from_nat(offset),U32.from_nat(query),head,balanced,limit,inside(offset,capacity,index_inside),inside(query,capacity,query_inside),different))

def listed(~Element: Data,+values: List<&2,Element>,index: Nat,+offset: Nat,+depth: Nat,+tree: Tree.Tree<Element>,+fallback: Element,+balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(offset,List.length(&2,Element,values)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  within: {Nat.is_lt(index,List.length(&2,Element,values)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write_sequence(~Element,values,U32.from_nat(offset),tree),U32.from_nat(Nat.add(offset,index))) == at(~Element,fallback,index,values) : Element}:
  match values index:
    case [] 0n: Empty.absurd({Tree.read(Element,tree,U32.from_nat(Nat.add(offset,0n))) == fallback : Element},Words.false_true(within))
    case [] 1n+rest: Empty.absurd({Tree.read(Element,tree,U32.from_nat(Nat.add(offset,1n+rest))) == fallback : Element},Words.false_true(within))
    case +head <> +rest 0n:
      +written = Tree.write(Element,tree,U32.from_nat(offset),head)
      +capacity = Readback.write_capacity(~Element,tree,U32.from_nat(offset),head)
      %Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset)) : {Tree.read(Element,Tree.write_sequence(~Element,rest,(U32.from_nat(offset) + 1 : U32),written),U32.from_nat(_)) == head : Element}
      %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.increment_encoded(offset)) : {Tree.read(Element,Tree.write_sequence(~Element,rest,_,written),U32.from_nat(offset)) == head : Element}
      Equal.trans(Element,Tree.read(Element,Tree.write_sequence(~Element,rest,U32.from_nat(1n+offset),written),U32.from_nat(offset)),Tree.read(Element,written,U32.from_nat(offset)),head,
        preserved(~Element,rest,1n+offset,offset,depth,written,Shape.write(Element,depth,tree,U32.from_nat(offset),head,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity) : {Nat.is_le(Nat.add(1n+offset,List.length(&2,Element,rest)),U32.to_nat(_)) == True{} : Bool}
          Order.advance_room(offset,List.length(&2,Element,rest),U32.to_nat(Tree.capacity(Element,tree)),room),
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity) : {Nat.is_lt(offset,U32.to_nat(_)) == True{} : Bool}
          Order.room_index(offset,List.length(&2,Element,rest),U32.to_nat(Tree.capacity(Element,tree)),room),Order.lt_succ(offset,offset,Order.reflexive(offset))),
        Readback.read_written(~Element,tree,U32.from_nat(offset),head))
    case +head <> +rest 1n+ +previous:
      +written = Tree.write(Element,tree,U32.from_nat(offset),head)
      +capacity = Readback.write_capacity(~Element,tree,U32.from_nat(offset),head)
      %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.increment_encoded(offset)) : {Tree.read(Element,Tree.write_sequence(~Element,rest,_,written),U32.from_nat(Nat.add(offset,1n+previous))) == at(~Element,fallback,previous,rest) : Element}
      %Equal.sym(Nat,Nat.add(offset,1n+previous),1n+Nat.add(offset,previous),Addition.successor_right(offset,previous)) : {Tree.read(Element,Tree.write_sequence(~Element,rest,U32.from_nat(1n+offset),written),U32.from_nat(_)) == at(~Element,fallback,previous,rest) : Element}
      listed(~Element,rest,previous,1n+offset,depth,written,fallback,Shape.write(Element,depth,tree,U32.from_nat(offset),head,balanced),limit,
        %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity) : {Nat.is_le(Nat.add(1n+offset,List.length(&2,Element,rest)),U32.to_nat(_)) == True{} : Bool}
        Order.advance_room(offset,List.length(&2,Element,rest),U32.to_nat(Tree.capacity(Element,tree)),room),within)

# Readback from concatenated lists preserves each rectangular row's order.
def appended_before(~Element: Data,+fallback: Element,+left: List<&2,Element>,+right: List<&2,Element>,index: Nat,
  inside: {Nat.is_lt(index,List.length(&2,Element,left)) == True{} : Bool}) ->
  {at(~Element,fallback,index,List.append(&2,Element,left,right)) == at(~Element,fallback,index,left) : Element}:
  match left index:
    case Nil{} index:
      Empty.absurd({at(~Element,fallback,index,right) == at(~Element,fallback,index,[]) : Element},
        Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),inside)))
    case head <> rest 0n: {==}
    case head <> +rest 1n+ +previous: appended_before(~Element,fallback,rest,right,previous,inside)

def appended_after(~Element: Data,+fallback: Element,left: List<&2,Element>,+right: List<&2,Element>,+index: Nat) ->
  {at(~Element,fallback,Nat.add(List.length(&2,Element,left),index),List.append(&2,Element,left,right)) == at(~Element,fallback,index,right) : Element}:
  match left:
    case Nil{}: {==}
    case head <> rest: appended_after(~Element,fallback,rest,right,index)

def concat_enumerate_at(~Element: Data,~Context: Data,~row: Context -> Nat -> List<&2,Element>,~width: Context -> Nat,
  ~row_length: @+context: Context -> @+index: Nat -> {List.length(&2,Element,row(context,index)) == width(context) : Nat},
  +fallback: Element,+rows: Nat,+first: Nat,+row_index: Nat,+column_index: Nat,+context: Context,
  row_inside: {Nat.is_lt(row_index,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column_index,width(context)) == True{} : Bool}) ->
  {at(~Element,fallback,Nat.add(Nat.mul(row_index,width(context)),column_index),List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rows,first,context))) ==
    at(~Element,fallback,column_index,row(context,Nat.add(first,row_index))) : Element}:
  match rows row_index:
    case 0n row_index:
      Empty.absurd({at(~Element,fallback,Nat.add(Nat.mul(row_index,width(context)),column_index),[]) == at(~Element,fallback,column_index,row(context,Nat.add(first,row_index))) : Element},
        Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(row_index,0n),True{},Equal.sym(Bool,Nat.is_lt(row_index,0n),False{},Division.lt_zero(row_index)),row_inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) :
        {at(~Element,fallback,column_index,List.append(&2,Element,row(context,first),List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rest,1n+first,context)))) == at(~Element,fallback,column_index,row(context,_)) : Element}
      appended_before(~Element,fallback,row(context,first),List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rest,1n+first,context)),column_index,
        %Equal.sym(Nat,List.length(&2,Element,row(context,first)),width(context),row_length(context,first)) : {Nat.is_lt(column_index,_) == True{} : Bool} column_inside)
    case 1n+ +rest 1n+ +previous:
      +tail_index = Nat.add(Nat.mul(previous,width(context)),column_index)
      +tail = List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rest,1n+first,context))
      %Equal.sym(Nat,Nat.add(first,1n+previous),1n+Nat.add(first,previous),Addition.successor_right(first,previous)) :
        {at(~Element,fallback,Nat.add(Nat.mul(1n+previous,width(context)),column_index),List.append(&2,Element,row(context,first),tail)) == at(~Element,fallback,column_index,row(context,_)) : Element}
      %Equal.sym(Nat,Nat.add(Nat.add(width(context),Nat.mul(previous,width(context))),column_index),Nat.add(width(context),tail_index),Addition.associative(width(context),Nat.mul(previous,width(context)),column_index)) :
        {at(~Element,fallback,_,List.append(&2,Element,row(context,first),tail)) == at(~Element,fallback,column_index,row(context,1n+Nat.add(first,previous))) : Element}
      %row_length(context,first) :
        {at(~Element,fallback,Nat.add(_,tail_index),List.append(&2,Element,row(context,first),tail)) == at(~Element,fallback,column_index,row(context,1n+Nat.add(first,previous))) : Element}
      Equal.trans(Element,at(~Element,fallback,Nat.add(List.length(&2,Element,row(context,first)),tail_index),List.append(&2,Element,row(context,first),tail)),at(~Element,fallback,tail_index,tail),
        at(~Element,fallback,column_index,row(context,1n+Nat.add(first,previous))),appended_after(~Element,fallback,row(context,first),tail,tail_index),
        concat_enumerate_at(~Element,~Context,~row,~width,~row_length,fallback,rest,1n+first,previous,column_index,context,row_inside,column_inside))

def concat_enumerate_length(~Element: Data,~Context: Data,~row: Context -> Nat -> List<&2,Element>,~width: Context -> Nat,
  ~row_length: @+context: Context -> @+index: Nat -> {List.length(&2,Element,row(context,index)) == width(context) : Nat},
  rows: Nat,+first: Nat,+context: Context) ->
  {List.length(&2,Element,List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rows,first,context))) == Nat.mul(rows,width(context)) : Nat}:
  match rows:
    case 0n: {==}
    case 1n+ +rest:
      +head = row(context,first)
      +tail = List.concat(&2,Element,enumerate(~List<&2,Element>,~Context,~row,rest,1n+first,context))
      %Equal.sym(Nat,List.length(&2,Element,List.append(&2,Element,head,tail)),Nat.add(List.length(&2,Element,head),List.length(&2,Element,tail)),append_length(~Element,head,tail)) :
        {_ == Nat.add(width(context),Nat.mul(rest,width(context))) : Nat}
      %Equal.sym(Nat,List.length(&2,Element,head),width(context),row_length(context,first)) :
        {Nat.add(_,List.length(&2,Element,tail)) == Nat.add(width(context),Nat.mul(rest,width(context))) : Nat}
      %Equal.sym(Nat,List.length(&2,Element,tail),Nat.mul(rest,width(context)),concat_enumerate_length(~Element,~Context,~row,~width,~row_length,rest,1n+first,context)) :
        {Nat.add(width(context),_) == Nat.add(width(context),Nat.mul(rest,width(context))) : Nat}
      {==}
