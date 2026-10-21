# Readback of any bounded native store program. The observed value is the
# last matching assignment; writes at other valid addresses preserve it.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./write_program.bend as Writes
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order

def Inside(~Element: Data,+capacity: U32,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<{U32.is_lt(index,capacity) == True{} : Bool},Inside(~Element,capacity,rest)>

def selected(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+query: U32,previous: Element) -> Element:
  match writes:
    case Nil{}: previous
    case Writes.Put{index,value} <> rest:
      selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))

def inside_append(~Element: Data,+capacity: U32,first: List<&2,Writes.Assignment<Element>>,+second: List<&2,Writes.Assignment<Element>>,
  first_inside: Inside(~Element,capacity,first),+second_inside: Inside(~Element,capacity,second)) -> Inside(~Element,capacity,List.append(&2,Writes.Assignment<Element>,first,second)):
  match first:
    case Nil{}: second_inside
    case Writes.Put{index,value} <> rest:
      Shape.Both{head,tail} = first_inside
      Shape.Both{head,inside_append(~Element,capacity,rest,second,tail,second_inside)}

def selected_append(~Element: Data,first: List<&2,Writes.Assignment<Element>>,+second: List<&2,Writes.Assignment<Element>>,+query: U32,+previous: Element) ->
  {selected(~Element,List.append(&2,Writes.Assignment<Element>,first,second),query,previous) == selected(~Element,second,query,selected(~Element,first,query,previous)) : Element}:
  match first:
    case Nil{}: {==}
    case Writes.Put{index,value} <> rest:
      selected_append(~Element,rest,second,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))

def one_compared_full(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,equal: Bool,
  compared: {U32.is_eq(index,query) == equal : Bool},balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,31n) == True{} : Bool},
  index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Bool.pick(Element,equal,value,Tree.read(Element,tree,query)) : Element}:
  match equal:
    case True{}:
      %Arithmetic.equal_of_is_eq(index,query,compared) : {Tree.read(Element,Tree.write(Element,tree,index,value),_) == value : Element}
      StorageReadback.read_written(~Element,tree,index,value)
    case False{}: StorageReadback.read_preserved_full(~Element,depth,tree,index,query,value,balanced,limit,index_inside,query_inside,compared)

def one_full(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,
  balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,31n) == True{} : Bool},
  index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)) : Element}:
  one_compared_full(~Element,depth,tree,index,query,value,U32.is_eq(index,query),{==},balanced,limit,index_inside,query_inside)

def program_full(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,
  +balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},inside: Inside(~Element,Tree.capacity(Element,tree),writes)) ->
  {Tree.read(Element,Writes.pure(Element,writes,tree),query) == selected(~Element,writes,query,Tree.read(Element,tree,query)) : Element}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{index_inside,rest_inside} = inside
      +written = Tree.write(Element,tree,index,value)
      +capacity_equal = StorageReadback.write_capacity(~Element,tree,index,value)
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,rest,written),query),selected(~Element,rest,query,Tree.read(Element,written,query)),
        selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query))),
        program_full(~Element,rest,depth,written,query,Shape.write(Element,depth,tree,index,value,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : Inside(~Element,_,rest) rest_inside),
        Equal.cong(Element,Element,previous => selected(~Element,rest,query,previous),Tree.read(Element,written,query),
          Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)),one_full(~Element,depth,tree,index,query,value,balanced,limit,index_inside,query_inside)))

def one(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,
  balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)) : Element}:
  one_full(~Element,depth,tree,index,query,value,balanced,Order.successor(depth,30n,limit),index_inside,query_inside)

def program(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,
  +balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},inside: Inside(~Element,Tree.capacity(Element,tree),writes)) ->
  {Tree.read(Element,Writes.pure(Element,writes,tree),query) == selected(~Element,writes,query,Tree.read(Element,tree,query)) : Element}:
  program_full(~Element,writes,depth,tree,query,balanced,Order.successor(depth,30n,limit),query_inside,inside)

# A producer supplies matching values and complete coverage independently.
# Later writes need not be disjoint if they agree at the queried address.
def Matching(~Element: Data,+query: U32,+wanted: Element,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case []: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<Boolean.When(U32.is_eq(index,query),{value == wanted : Element}),Matching(~Element,query,wanted,rest)>

def contains(~Element: Data,+query: U32,writes: List<&2,Writes.Assignment<Element>>) -> Bool:
  match writes:
    case []: False{}
    case Writes.Put{index,value} <> rest: U32.is_eq(index,query) || contains(~Element,query,rest)

def value_matches(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+destination: U32,+query: U32,+actual: Element,equal: Bool,
  compared: {U32.is_eq(destination,query) == equal : Bool},correct: {actual == value(context,destination) : Element}) ->
  Boolean.When(equal,{actual == value(context,query) : Element}):
  match equal:
    case False{}: Unit{}
    case True{}:
      Equal.trans(Element,actual,value(context,destination),value(context,query),correct,
        Equal.cong(U32,Element,index => value(context,index),destination,query,Arithmetic.equal_of_is_eq(destination,query,compared)))

def selected_matching(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+query: U32,+wanted: Element,+previous: Element,matching: Matching(~Element,query,wanted,writes)) ->
  {selected(~Element,writes,query,previous) == Bool.pick(Element,contains(~Element,query,writes),wanted,previous) : Element}:
  match writes:
    case []: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{head,tail} = matching
      %Equal.sym(Element,selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous)),
        Bool.pick(Element,contains(~Element,query,rest),wanted,Bool.pick(Element,U32.is_eq(index,query),value,previous)),
        selected_matching(~Element,rest,query,wanted,Bool.pick(Element,U32.is_eq(index,query),value,previous),tail)) :
        {_ == Bool.pick(Element,U32.is_eq(index,query) || contains(~Element,query,rest),wanted,previous) : Element}
      Boolean.choose_matching(Element,U32.is_eq(index,query),contains(~Element,query,rest),value,wanted,previous,head)

def matching_append(~Element: Data,+query: U32,+wanted: Element,left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  left_matching: Matching(~Element,query,wanted,left),right_matching: Matching(~Element,query,wanted,right)) ->
  Matching(~Element,query,wanted,List.append(&2,Writes.Assignment<Element>,left,right)):
  match left:
    case []: right_matching
    case Writes.Put{index,value} <> rest:
      Shape.Both{head,tail} = left_matching
      Shape.Both{head,matching_append(~Element,query,wanted,rest,right,tail,right_matching)}

def contains_append(~Element: Data,+query: U32,left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>) ->
  {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == contains(~Element,query,left) || contains(~Element,query,right) : Bool}:
  match left:
    case []: {==}
    case Writes.Put{+index,value} <> +rest:
      %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,rest,right)),contains(~Element,query,rest) || contains(~Element,query,right),contains_append(~Element,query,rest,right)) :
        {U32.is_eq(index,query) || _ == (U32.is_eq(index,query) || contains(~Element,query,rest)) || contains(~Element,query,right) : Bool}
      Boolean.or_associative(U32.is_eq(index,query),contains(~Element,query,rest),contains(~Element,query,right))

def covered_left(~Element: Data,+query: U32,+left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  covered: {contains(~Element,query,left) == True{} : Bool}) -> {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == True{} : Bool}:
  %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)),contains(~Element,query,left) || contains(~Element,query,right),contains_append(~Element,query,left,right)) : {_ == True{} : Bool}
  %Equal.sym(Bool,contains(~Element,query,left),True{},covered) : {_ || contains(~Element,query,right) == True{} : Bool}
  {==}

def covered_right(~Element: Data,+query: U32,+left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  covered: {contains(~Element,query,right) == True{} : Bool}) -> {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == True{} : Bool}:
  %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)),contains(~Element,query,left) || contains(~Element,query,right),contains_append(~Element,query,left,right)) : {_ == True{} : Bool}
  %Equal.sym(Bool,contains(~Element,query,right),True{},covered) : {contains(~Element,query,left) || _ == True{} : Bool}
  Boolean.or_true_right(contains(~Element,query,left))

def covered(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,+wanted: Element,
  balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},inside: Inside(~Element,Tree.capacity(Element,tree),writes),
  matching: Matching(~Element,query,wanted,writes),coverage: {contains(~Element,query,writes) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes,tree),query) == wanted : Element}:
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,writes,tree),query),selected(~Element,writes,query,Tree.read(Element,tree,query)),wanted,
    program(~Element,writes,depth,tree,query,balanced,limit,query_inside,inside),
    %coverage : {selected(~Element,writes,query,Tree.read(Element,tree,query)) == Bool.pick(Element,_,wanted,Tree.read(Element,tree,query)) : Element}
    selected_matching(~Element,writes,query,wanted,Tree.read(Element,tree,query),matching))

# Natural-address receipts share the native store-program semantics.
type NaturalWrite is Data:
  NaturalWrite{index: Nat,value: F32}

def natural_contains(+query: Nat,writes: List<&2,NaturalWrite>) -> Bool:
  match writes:
    case Nil{}:
      False{}
    case NaturalWrite{i,v} <> rest:
      Nat.is_eq(query,i) || natural_contains(query,rest)

def Matches(ok: Bool,+actual: F32,+wanted: F32) -> Data:
  Boolean.When(ok,{actual == wanted : F32})

def MatchingWritesHaveValue(+query: Nat,+value: F32,writes: List<&2,NaturalWrite>) -> Data:
  match writes:
    case Nil{}:
      Unit
    case NaturalWrite{i,v} <> rest:
      Shape.Both<Matches(Nat.is_eq(query,i),v,value),MatchingWritesHaveValue(query,value,rest)>

def natural_write_addresses(writes: List<&2,Writes.Assignment<F32>>) -> List<&2,NaturalWrite>:
  match writes:
    case Nil{}:
      []
    case Writes.Put{i,v} <> rest:
      NaturalWrite{U32.to_nat(i),v} <> natural_write_addresses(rest)

type WriteEvidence<-query: Nat,-value: F32,-capacity: U32,-writes: List<&2,Writes.Assignment<F32>>> is Data:
  WriteEvidence{bounds: Inside(~F32,capacity,writes),values: MatchingWritesHaveValue(query,value,natural_write_addresses(writes))}

def prepend_evidence(+query: Nat,+value: F32,+capacity: U32,+address: U32,+actual: F32,+rest: List<&2,Writes.Assignment<F32>>,
  inside: {U32.is_lt(address,capacity) == True{} : Bool},same: Matches(Nat.is_eq(query,U32.to_nat(address)),actual,value),
  tail: WriteEvidence<query,value,capacity,rest>) -> WriteEvidence<query,value,capacity,Writes.Put{address,actual} <> rest>:
  WriteEvidence{bounds,values} = tail
  WriteEvidence{Shape.Both{inside,bounds},Shape.Both{same,values}}

def concatenated_evidence(+query: Nat,+value: F32,+capacity: U32,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,
  first: WriteEvidence<query,value,capacity,left>,second: WriteEvidence<query,value,capacity,right>) ->
  WriteEvidence<query,value,capacity,List.append(&2,Writes.Assignment<F32>,left,right)>:
  match left:
    case Nil{}: second
    case Writes.Put{+address,+actual} <> +rest:
      WriteEvidence{bounds,values} = first
      Shape.Both{inside,tail_bounds} = bounds
      Shape.Both{same,tail_values} = values
      prepend_evidence(query,value,capacity,address,actual,List.append(&2,Writes.Assignment<F32>,rest,right),inside,same,
        concatenated_evidence(query,value,capacity,rest,right,WriteEvidence{tail_bounds,tail_values},second))

def natural_matching(+writes: List<&2,Writes.Assignment<F32>>,+query: U32,+wanted: F32,matching: MatchingWritesHaveValue(U32.to_nat(query),wanted,natural_write_addresses(writes))) -> Matching(~F32,query,wanted,writes):
  match writes:
    case []: Unit{}
    case Writes.Put{+index,value} <> +rest:
      Shape.Both{head,tail} = matching
      Shape.Both{
        %Equal.sym(Bool,U32.is_eq(index,query),Nat.is_eq(U32.to_nat(index),U32.to_nat(query)),Comparison.eq(index,query)) : Boolean.When(_,{value == wanted : F32})
        %Equal.sym(Bool,Nat.is_eq(U32.to_nat(index),U32.to_nat(query)),Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),Order.equal_symmetric(U32.to_nat(index),U32.to_nat(query))) : Boolean.When(_,{value == wanted : F32})
        head,
        natural_matching(rest,query,wanted,tail)}

def natural_coverage(+writes: List<&2,Writes.Assignment<F32>>,+query: U32) ->
  {contains(~F32,query,writes) == natural_contains(U32.to_nat(query),natural_write_addresses(writes)) : Bool}:
  match writes:
    case []: {==}
    case Writes.Put{+index,value} <> +rest:
      %Equal.sym(Bool,U32.is_eq(index,query),Nat.is_eq(U32.to_nat(index),U32.to_nat(query)),Comparison.eq(index,query)) : {_ || contains(~F32,query,rest) == natural_contains(U32.to_nat(query),natural_write_addresses(writes)) : Bool}
      %Equal.sym(Bool,Nat.is_eq(U32.to_nat(index),U32.to_nat(query)),Nat.is_eq(U32.to_nat(query),U32.to_nat(index)),Order.equal_symmetric(U32.to_nat(index),U32.to_nat(query))) : {_ || contains(~F32,query,rest) == natural_contains(U32.to_nat(query),natural_write_addresses(writes)) : Bool}
      %Equal.sym(Bool,contains(~F32,query,rest),natural_contains(U32.to_nat(query),natural_write_addresses(rest)),natural_coverage(rest,query)) : {Nat.is_eq(U32.to_nat(query),U32.to_nat(index)) || _ == natural_contains(U32.to_nat(query),natural_write_addresses(writes)) : Bool}
      {==}

def readback_from_matching_covered_writes(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+i: U32,+wanted: F32,
  shape: Shape.Balanced(F32,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},bounds: Inside(~F32,Tree.capacity(F32,tree),writes),inside: {U32.is_lt(i,Tree.capacity(F32,tree)) == True{} : Bool},
  correct: MatchingWritesHaveValue(U32.to_nat(i),wanted,natural_write_addresses(writes)),coverage: {natural_contains(U32.to_nat(i),natural_write_addresses(writes)) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,writes,tree),i) == wanted : F32}:
  covered(~F32,writes,depth,tree,i,wanted,shape,limit,inside,bounds,natural_matching(writes,i,wanted,correct),
    %Equal.sym(Bool,contains(~F32,i,writes),natural_contains(U32.to_nat(i),natural_write_addresses(writes)),natural_coverage(writes,i)) : {_ == True{} : Bool} coverage)

def natural_contains_append(+query: Nat,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>) ->
  {natural_contains(query,natural_write_addresses(List.append(&2,Writes.Assignment<F32>,left,right))) == natural_contains(query,natural_write_addresses(left)) || natural_contains(query,natural_write_addresses(right)) : Bool}:
  match left:
    case []: {==}
    case Writes.Put{+index,value} <> +rest:
      %Equal.sym(Bool,natural_contains(query,natural_write_addresses(List.append(&2,Writes.Assignment<F32>,rest,right))),natural_contains(query,natural_write_addresses(rest)) || natural_contains(query,natural_write_addresses(right)),natural_contains_append(query,rest,right)) :
        {Nat.is_eq(query,U32.to_nat(index)) || _ == (Nat.is_eq(query,U32.to_nat(index)) || natural_contains(query,natural_write_addresses(rest))) || natural_contains(query,natural_write_addresses(right)) : Bool}
      Boolean.or_associative(Nat.is_eq(query,U32.to_nat(index)),natural_contains(query,natural_write_addresses(rest)),natural_contains(query,natural_write_addresses(right)))

def concatenation_covered(+query: Nat,+left: List<&2,Writes.Assignment<F32>>,+right: List<&2,Writes.Assignment<F32>>,
  choice: Either<&2,&2,{natural_contains(query,natural_write_addresses(left)) == True{} : Bool},{natural_contains(query,natural_write_addresses(right)) == True{} : Bool}>) ->
  {natural_contains(query,natural_write_addresses(List.append(&2,Writes.Assignment<F32>,left,right))) == True{} : Bool}:
  match choice:
    case Inl{proof}:
      %Equal.sym(Bool,natural_contains(query,natural_write_addresses(List.append(&2,Writes.Assignment<F32>,left,right))),natural_contains(query,natural_write_addresses(left)) || natural_contains(query,natural_write_addresses(right)),natural_contains_append(query,left,right)) : {_ == True{} : Bool}
      %Equal.sym(Bool,natural_contains(query,natural_write_addresses(left)),True{},proof) : {_ || natural_contains(query,natural_write_addresses(right)) == True{} : Bool}
      {==}
    case Inr{proof}:
      %Equal.sym(Bool,natural_contains(query,natural_write_addresses(List.append(&2,Writes.Assignment<F32>,left,right))),natural_contains(query,natural_write_addresses(left)) || natural_contains(query,natural_write_addresses(right)),natural_contains_append(query,left,right)) : {_ == True{} : Bool}
      %Equal.sym(Bool,natural_contains(query,natural_write_addresses(right)),True{},proof) : {natural_contains(query,natural_write_addresses(left)) || _ == True{} : Bool}
      Boolean.or_true_right(natural_contains(query,natural_write_addresses(left)))

def tail_bounds(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+i: U32,+v: F32,balanced: Shape.Balanced(F32,depth,tree),bounds: Inside(~F32,Tree.capacity(F32,tree),writes)) -> Inside(~F32,Tree.capacity(F32,Tree.write(F32,tree,i,v)),writes):
  %Equal.sym(U32,Tree.capacity(F32,Tree.write(F32,tree,i,v)),Tree.capacity(F32,tree),StorageReadback.write_capacity(~F32,tree,i,v)) : Inside(~F32,_,writes)
  bounds

def Excludes(+query: U32,writes: List<&2,Writes.Assignment<F32>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<{U32.is_eq(index,query) == False{} : Bool},Excludes(query,rest)>

def excluded_readback(+writes: List<&2,Writes.Assignment<F32>>,+tree: Tree.Tree<F32>,+depth: Nat,+query: U32,
  +balanced: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  bounds: Inside(~F32,Tree.capacity(F32,tree),writes),+inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},
  excludes: Excludes(query,writes)) -> {Tree.read(F32,Writes.pure(F32,writes,tree),query) == Tree.read(F32,tree,query) : F32}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{+index_inside,rest_inside} = bounds
      Shape.Both{different,rest_excludes} = excludes
      +next = Tree.write(F32,tree,index,value)
      Equal.trans(F32,Tree.read(F32,Writes.pure(F32,rest,next),query),Tree.read(F32,next,query),Tree.read(F32,tree,query),
        excluded_readback(rest,next,depth,query,Shape.write(F32,depth,tree,index,value,balanced),depth_ok,
          tail_bounds(rest,tree,depth,index,value,balanced,rest_inside),
          %Equal.sym(U32,Tree.capacity(F32,next),Tree.capacity(F32,tree),StorageReadback.write_capacity(~F32,tree,index,value)) : {U32.is_lt(query,_) == True{} : Bool} inside,
          rest_excludes),
        StorageReadback.read_preserved(~F32,depth,tree,index,query,value,balanced,depth_ok,index_inside,inside,different))

def contains_native(query: Nat,writes: List<&2,Writes.Assignment<F32>>) -> Bool:
  natural_contains(query,natural_write_addresses(writes))
