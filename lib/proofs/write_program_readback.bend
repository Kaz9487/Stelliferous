# Readback of any bounded native store program. The observed value is the
# last matching assignment; writes at other valid addresses preserve it.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./write_program.bend as Writes
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ../kernel_gemm.bend as KernelGemm

def Inside(~Element: Data,+capacity: U32,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      KernelGemm.Both<{U32.is_lt(index,capacity) == True{} : Bool},Inside(~Element,capacity,rest)>

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
      KernelGemm.Both{head,tail} = first_inside
      KernelGemm.Both{head,inside_append(~Element,capacity,rest,second,tail,second_inside)}

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
      KernelGemm.Both{index_inside,rest_inside} = inside
      +written = Tree.write(Element,tree,index,value)
      +capacity_equal = StorageReadback.write_capacity(~Element,tree,index,value)
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,rest,written),query),selected(~Element,rest,query,Tree.read(Element,written,query)),
        selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query))),
        program_full(~Element,rest,depth,written,query,Shape.write(Element,depth,tree,index,value,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : Inside(~Element,_,rest) rest_inside),
        Equal.cong(Element,Element,previous => selected(~Element,rest,query,previous),Tree.read(Element,written,query),
          Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)),one_full(~Element,depth,tree,index,query,value,balanced,limit,index_inside,query_inside)))

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
      KernelGemm.Both<Boolean.When(U32.is_eq(index,query),{value == wanted : Element}),Matching(~Element,query,wanted,rest)>

def contains(~Element: Data,+query: U32,writes: List<&2,Writes.Assignment<Element>>) -> Bool:
  match writes:
    case []: False{}
    case Writes.Put{index,value} <> rest: U32.is_eq(index,query) || contains(~Element,query,rest)

def selected_matching(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+query: U32,+wanted: Element,+previous: Element,matching: Matching(~Element,query,wanted,writes)) ->
  {selected(~Element,writes,query,previous) == Bool.pick(Element,contains(~Element,query,writes),wanted,previous) : Element}:
  match writes:
    case []: {==}
    case Writes.Put{+index,+value} <> +rest:
      KernelGemm.Both{head,tail} = matching
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
      KernelGemm.Both{head,tail} = left_matching
      KernelGemm.Both{head,matching_append(~Element,query,wanted,rest,right,tail,right_matching)}

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

def tail_bounds(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,+tree: Tree.Tree<Element>,+depth: Nat,+i: U32,+v: Element,
  balanced: Shape.Balanced(Element,depth,tree),bounds: Inside(~Element,Tree.capacity(Element,tree),writes)) -> Inside(~Element,Tree.capacity(Element,Tree.write(Element,tree,i,v)),writes):
  %Equal.sym(U32,Tree.capacity(Element,Tree.write(Element,tree,i,v)),Tree.capacity(Element,tree),StorageReadback.write_capacity(~Element,tree,i,v)) : Inside(~Element,_,writes)
  bounds

def Excludes(~Element: Data,+query: U32,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      KernelGemm.Both<{U32.is_eq(index,query) == False{} : Bool},Excludes(~Element,query,rest)>

def excluded_readback(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,+tree: Tree.Tree<Element>,+depth: Nat,+query: U32,
  +balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  bounds: Inside(~Element,Tree.capacity(Element,tree),writes),+inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  excludes: Excludes(~Element,query,writes)) -> {Tree.read(Element,Writes.pure(Element,writes,tree),query) == Tree.read(Element,tree,query) : Element}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      KernelGemm.Both{+index_inside,rest_inside} = bounds
      KernelGemm.Both{different,rest_excludes} = excludes
      +next = Tree.write(Element,tree,index,value)
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,rest,next),query),Tree.read(Element,next,query),Tree.read(Element,tree,query),
        excluded_readback(~Element,rest,next,depth,query,Shape.write(Element,depth,tree,index,value,balanced),depth_ok,
          tail_bounds(~Element,rest,tree,depth,index,value,balanced,rest_inside),
          %Equal.sym(U32,Tree.capacity(Element,next),Tree.capacity(Element,tree),StorageReadback.write_capacity(~Element,tree,index,value)) : {U32.is_lt(query,_) == True{} : Bool} inside,
          rest_excludes),
        StorageReadback.read_preserved(~Element,depth,tree,index,query,value,balanced,depth_ok,index_inside,inside,different))

