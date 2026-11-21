# A balanced tree is the tree of its reads, so balanced trees whose reads
# agree are equal.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_address_bounds.bend as Address
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ../kernel_gemm.bend as KernelGemm

# The tree of a depth whose leaves are the reads of a tree from an offset on.
def built(-T: Data,depth: Nat,+tree: Tree.Tree<T>,+capacity: U32,+offset: U32) -> Tree.Tree<T>:
  match depth:
    case 0n: Tree.Leaf{Tree.fetch(T,tree,capacity,offset)}
    case 1n+ +rest: Tree.Node{built(T,rest,tree,capacity,offset),built(T,rest,tree,capacity,(offset + Address.capacity(rest) : U32))}

# An offset moved by the capacity of a size, inside the capacity of a depth,
# has the moved value.
def step_value(+depth: Nat,+size: Nat,+offset: U32,+limit: {Nat.is_le(depth,31n) == True{} : Bool},+size_limit: {Nat.is_le(size,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Shape.extent(size)),Shape.extent(depth)) == True{} : Bool}) ->
  {U32.to_nat((offset + Address.capacity(size) : U32)) == Nat.add(U32.to_nat(offset),Shape.extent(size)) : Nat}:
  +sum = Nat.add(U32.to_nat(offset),Shape.extent(size))
  +word = {
    %Address.capacity_value(size,size_limit) : {(offset + Address.capacity(size) : U32) == U32.from_nat(Nat.add(U32.to_nat(offset),_)) : U32}
    Arithmetic.native_add(offset,Address.capacity(size)) : {(offset + Address.capacity(size) : U32) == U32.from_nat(sum) : U32}}
  Equal.trans(Nat,U32.to_nat((offset + Address.capacity(size) : U32)),U32.to_nat(U32.from_nat(sum)),sum,
    Equal.cong(U32,Nat,U32.to_nat,(offset + Address.capacity(size) : U32),U32.from_nat(sum),word),
    Bounds.roundtrip(sum,Address.capacity(depth),Power.native_power_bound(depth,limit),
      %Equal.sym(Nat,U32.to_nat(Address.capacity(depth)),Shape.extent(depth),Address.capacity_value(depth,limit)) : {Nat.is_le(sum,_) == True{} : Bool}
      room))

# An offset with room for one read is inside the capacity of a depth.
def offset_inside(+depth: Nat,+offset: U32,+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),1n),Shape.extent(depth)) == True{} : Bool}) -> {U32.is_lt(offset,Address.capacity(depth)) == True{} : Bool}:
  +value = U32.to_nat(offset)
  +below = Order.succ_le_lt(value,Shape.extent(depth),{
    %Order.commutative(value,1n) : {Nat.is_le(_,Shape.extent(depth)) == True{} : Bool}
    room : {Nat.is_le(1n+value,Shape.extent(depth)) == True{} : Bool}})
  Equal.trans(Bool,U32.is_lt(offset,Address.capacity(depth)),Nat.is_lt(value,U32.to_nat(Address.capacity(depth))),True{},Comparison.lt(offset,Address.capacity(depth)),
    %Equal.sym(Nat,U32.to_nat(Address.capacity(depth)),Shape.extent(depth),Address.capacity_value(depth,limit)) : {Nat.is_lt(value,_) == True{} : Bool}
    below)

# The halves of a run of twice a size from an offset fit where the run fits.
def first_half(+offset: Nat,+size: Nat,+bound: Nat,+room: {Nat.is_le(Nat.add(offset,Nat.double(Shape.extent(size))),bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(offset,Shape.extent(size)),bound) == True{} : Bool}:
  +extent = Shape.extent(size)
  Order.transitive(Nat.add(offset,extent),Nat.add(offset,Nat.double(extent)),bound,Order.offset_le(offset,extent,Nat.double(extent),Order.le_double(extent)),room)

def second_half(+offset: Nat,+size: Nat,+bound: Nat,+room: {Nat.is_le(Nat.add(offset,Nat.double(Shape.extent(size))),bound) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(offset,Shape.extent(size)),Shape.extent(size)),bound) == True{} : Bool}:
  +extent = Shape.extent(size)
  %Equal.sym(Nat,Nat.add(Nat.add(offset,extent),extent),Nat.add(offset,Nat.add(extent,extent)),Addition.associative(offset,extent,extent)) : {Nat.is_le(_,bound) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(extent,extent),Nat.double(extent),Addition.double(extent)) : {Nat.is_le(Nat.add(offset,_),bound) == True{} : Bool}
  room

# The second half from an offset moved by the first half's capacity fits.
def moved_half(+depth: Nat,+size: Nat,+offset: U32,+limit: {Nat.is_le(depth,31n) == True{} : Bool},+size_limit: {Nat.is_le(size,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.double(Shape.extent(size))),Shape.extent(depth)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat((offset + Address.capacity(size) : U32)),Shape.extent(size)),Shape.extent(depth)) == True{} : Bool}:
  +value = U32.to_nat(offset)
  %Equal.sym(Nat,U32.to_nat((offset + Address.capacity(size) : U32)),Nat.add(value,Shape.extent(size)),
    step_value(depth,size,offset,limit,size_limit,first_half(value,size,Shape.extent(depth),room))) :
    {Nat.is_le(Nat.add(_,Shape.extent(size)),Shape.extent(depth)) == True{} : Bool}
  second_half(value,size,Shape.extent(depth),room)

# Reads of a node from an offset inside its left half are reads of the left
# half.
def built_left(~Element: Data,+depth: Nat,+rest: Nat,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+offset: U32,
  +limit: {Nat.is_le(1n+rest,31n) == True{} : Bool},+depth_limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Shape.extent(depth)),Shape.extent(rest)) == True{} : Bool}) ->
  {built(Element,depth,Tree.Node{left,right},Address.capacity(1n+rest),offset) == built(Element,depth,left,Address.capacity(rest),offset) : Tree.Tree<Element>}:
  match depth:
    case 0n:
      +half = Power.native_half_power(rest,Bounds.predecessor_strict(rest,31n,limit))
      +inside = offset_inside(rest,offset,Address.child_limit_full(rest,limit),room)
      %Equal.sym(U32,U32.shr(Address.capacity(1n+rest)),Address.capacity(rest),half) :
        {Tree.Leaf{Bool.pick(Element,U32.is_lt(offset,_),Tree.fetch(Element,left,_,offset),Tree.fetch(Element,right,_,(offset - _ : U32)))} ==
          Tree.Leaf{Tree.fetch(Element,left,Address.capacity(rest),offset)} : Tree.Tree<Element>}
      %Equal.sym(Bool,U32.is_lt(offset,Address.capacity(rest)),True{},inside) :
        {Tree.Leaf{Bool.pick(Element,_,Tree.fetch(Element,left,Address.capacity(rest),offset),Tree.fetch(Element,right,Address.capacity(rest),(offset - Address.capacity(rest) : U32)))} ==
          Tree.Leaf{Tree.fetch(Element,left,Address.capacity(rest),offset)} : Tree.Tree<Element>}
      {==}
    case 1n+ +size:
      +rest_limit = Address.child_limit_full(rest,limit)
      +size_limit = Address.child_limit_full(size,depth_limit)
      +node = {Tree.Node{left,right} : Tree.Tree<Element>}
      +whole = Address.capacity(1n+rest)
      +half = Address.capacity(rest)
      +moved = (offset + Address.capacity(size) : U32)
      %built_left(~Element,size,rest,left,right,offset,limit,size_limit,first_half(U32.to_nat(offset),size,Shape.extent(rest),room)) :
        {Tree.Node{built(Element,size,node,whole,offset),built(Element,size,node,whole,moved)} == Tree.Node{_,built(Element,size,left,half,moved)} : Tree.Tree<Element>}
      %built_left(~Element,size,rest,left,right,moved,limit,size_limit,moved_half(rest,size,offset,rest_limit,size_limit,room)) :
        {Tree.Node{built(Element,size,node,whole,offset),built(Element,size,node,whole,moved)} == Tree.Node{built(Element,size,node,whole,offset),_} : Tree.Tree<Element>}
      {==}

# Reads of a node from its left half's capacity on are reads of its right
# half.
def built_right(~Element: Data,+depth: Nat,+rest: Nat,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+offset: U32,
  +limit: {Nat.is_le(1n+rest,31n) == True{} : Bool},+depth_limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Shape.extent(depth)),Shape.extent(rest)) == True{} : Bool}) ->
  {built(Element,depth,Tree.Node{left,right},Address.capacity(1n+rest),(Address.capacity(rest) + offset : U32)) == built(Element,depth,right,Address.capacity(rest),offset) :
    Tree.Tree<Element>}:
  match depth:
    case 0n:
      +half = Power.native_half_power(rest,Bounds.predecessor_strict(rest,31n,limit))
      +rest_limit = Address.child_limit_full(rest,limit)
      +step = Address.capacity(rest)
      +extent = Shape.extent(rest)
      +value = U32.to_nat(offset)
      +shifted = (step + offset : U32)
      +value_room = Order.transitive(value,Nat.add(value,1n),extent,Order.le_add(value,1n),room)
      +sum_room = {
        %Addition.double(extent) : {Nat.is_le(Nat.add(value,extent),_) == True{} : Bool}
        Order.add_le(value,extent,extent,value_room) : {Nat.is_le(Nat.add(value,extent),Shape.extent(1n+rest)) == True{} : Bool}}
      +shifted_value = {
        %Arithmetic.add_commutative(offset,step) : {U32.to_nat(_) == Nat.add(value,extent) : Nat}
        step_value(1n+rest,rest,offset,limit,rest_limit,sum_room) : {U32.to_nat(shifted) == Nat.add(value,extent) : Nat}}
      +outside = Equal.trans(Bool,U32.is_lt(shifted,step),Nat.is_lt(U32.to_nat(shifted),U32.to_nat(step)),False{},Comparison.lt(shifted,step),
        %Equal.sym(Nat,U32.to_nat(shifted),Nat.add(value,extent),shifted_value) : {Nat.is_lt(_,U32.to_nat(step)) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(step),extent,Address.capacity_value(rest,rest_limit)) : {Nat.is_lt(Nat.add(value,extent),_) == False{} : Bool}
        Order.not_below(extent,Nat.add(value,extent),{
          %Order.commutative(extent,value) : {Nat.is_le(extent,_) == True{} : Bool}
          Order.le_add(extent,value) : {Nat.is_le(extent,Nat.add(value,extent)) == True{} : Bool}}))
      +back = Equal.trans(U32,(shifted - step : U32),((offset + step : U32) - step : U32),offset,
        Equal.cong(U32,U32,word => (word - step : U32),shifted,(offset + step : U32),Arithmetic.add_commutative(step,offset)),Subtraction.subtract_add(offset,step))
      %Equal.sym(U32,U32.shr(Address.capacity(1n+rest)),step,half) :
        {Tree.Leaf{Bool.pick(Element,U32.is_lt(shifted,_),Tree.fetch(Element,left,_,shifted),Tree.fetch(Element,right,_,(shifted - _ : U32)))} ==
          Tree.Leaf{Tree.fetch(Element,right,step,offset)} : Tree.Tree<Element>}
      %Equal.sym(Bool,U32.is_lt(shifted,step),False{},outside) :
        {Tree.Leaf{Bool.pick(Element,_,Tree.fetch(Element,left,step,shifted),Tree.fetch(Element,right,step,(shifted - step : U32)))} ==
          Tree.Leaf{Tree.fetch(Element,right,step,offset)} : Tree.Tree<Element>}
      %Equal.sym(U32,(shifted - step : U32),offset,back) : {Tree.Leaf{Tree.fetch(Element,right,step,_)} == Tree.Leaf{Tree.fetch(Element,right,step,offset)} : Tree.Tree<Element>}
      {==}
    case 1n+ +size:
      +rest_limit = Address.child_limit_full(rest,limit)
      +size_limit = Address.child_limit_full(size,depth_limit)
      +node = {Tree.Node{left,right} : Tree.Tree<Element>}
      +whole = Address.capacity(1n+rest)
      +half = Address.capacity(rest)
      +step = Address.capacity(size)
      +moved = (offset + step : U32)
      %built_right(~Element,size,rest,left,right,offset,limit,size_limit,first_half(U32.to_nat(offset),size,Shape.extent(rest),room)) :
        {Tree.Node{built(Element,size,node,whole,(half + offset : U32)),built(Element,size,node,whole,((half + offset : U32) + step : U32))} ==
          Tree.Node{_,built(Element,size,right,half,moved)} : Tree.Tree<Element>}
      %Equal.sym(U32,((half + offset : U32) + step : U32),(half + moved : U32),Arithmetic.add_associative(half,offset,step)) :
        {Tree.Node{built(Element,size,node,whole,(half + offset : U32)),built(Element,size,node,whole,_)} ==
          Tree.Node{built(Element,size,node,whole,(half + offset : U32)),built(Element,size,right,half,moved)} : Tree.Tree<Element>}
      %built_right(~Element,size,rest,left,right,moved,limit,size_limit,moved_half(rest,size,offset,rest_limit,size_limit,room)) :
        {Tree.Node{built(Element,size,node,whole,(half + offset : U32)),built(Element,size,node,whole,(half + moved : U32))} ==
          Tree.Node{built(Element,size,node,whole,(half + offset : U32)),_} : Tree.Tree<Element>}
      {==}

# A balanced tree is the tree of its reads.
def built_tree(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool}) ->
  {built(Element,depth,tree,Address.capacity(depth),0) == tree : Tree.Tree<Element>}:
  match depth tree:
    case 0n Tree.Leaf{value}: {==}
    case 0n Tree.Node{left,right}: match balanced:
    case 1n+rest Tree.Leaf{value}: match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      KernelGemm.Both{+left_balanced,+right_balanced} = balanced
      +rest_limit = Address.child_limit_full(rest,limit)
      +step = Address.capacity(rest)
      +node = {Tree.Node{left,right} : Tree.Tree<Element>}
      +whole = Address.capacity(1n+rest)
      +room = Order.reflexive(Shape.extent(rest))
      +start = Equal.trans(U32,(step + 0 : U32),step,(0 + step : U32),Arithmetic.add_zero(step),Equal.sym(U32,(0 + step : U32),step,Arithmetic.add_zero_left(step)))
      +first = built(Element,rest,node,whole,0)
      +second = built(Element,rest,node,whole,(0 + step : U32))
      +left_built = built(Element,rest,left,step,0)
      +right_built = built(Element,rest,right,step,0)
      %Equal.sym(Tree.Tree<Element>,first,left_built,built_left(~Element,rest,rest,left,right,0,limit,rest_limit,room)) :
        {Tree.Node{_,second} == Tree.Node{left,right} : Tree.Tree<Element>}
      %Equal.sym(Tree.Tree<Element>,left_built,left,built_tree(~Element,rest,left,left_balanced,rest_limit)) : {Tree.Node{_,second} == Tree.Node{left,right} : Tree.Tree<Element>}
      %start : {Tree.Node{left,built(Element,rest,node,whole,_)} == Tree.Node{left,right} : Tree.Tree<Element>}
      %Equal.sym(Tree.Tree<Element>,built(Element,rest,node,whole,(step + 0 : U32)),right_built,built_right(~Element,rest,rest,left,right,0,limit,rest_limit,room)) :
        {Tree.Node{left,_} == Tree.Node{left,right} : Tree.Tree<Element>}
      %Equal.sym(Tree.Tree<Element>,right_built,right,built_tree(~Element,rest,right,right_balanced,rest_limit)) : {Tree.Node{left,_} == Tree.Node{left,right} : Tree.Tree<Element>}
      {==}

# Trees of a context whose reads agree inside the capacity of its depth build
# the same tree from every offset with room.
def built_same(~Element: Data,~Context: Data,~first: Context -> Tree.Tree<Element>,~second: Context -> Tree.Tree<Element>,~depth_of: Context -> Nat,
  ~same: @+context: Context -> @+index: U32 -> @+inside: {U32.is_lt(index,Address.capacity(depth_of(context))) == True{} : Bool} ->
    {Tree.fetch(Element,first(context),Address.capacity(depth_of(context)),index) == Tree.fetch(Element,second(context),Address.capacity(depth_of(context)),index) : Element},
  +context: Context,+depth: Nat,+offset: U32,+limit: {Nat.is_le(depth_of(context),31n) == True{} : Bool},+depth_limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),Shape.extent(depth)),Shape.extent(depth_of(context))) == True{} : Bool}) ->
  {built(Element,depth,first(context),Address.capacity(depth_of(context)),offset) == built(Element,depth,second(context),Address.capacity(depth_of(context)),offset) :
    Tree.Tree<Element>}:
  match depth:
    case 0n:
      +whole = Address.capacity(depth_of(context))
      Equal.cong(Element,Tree.Tree<Element>,value => Tree.Leaf{value},Tree.fetch(Element,first(context),whole,offset),Tree.fetch(Element,second(context),whole,offset),
        same(context,offset,offset_inside(depth_of(context),offset,limit,room)))
    case 1n+ +size:
      +size_limit = Address.child_limit_full(size,depth_limit)
      +whole = Address.capacity(depth_of(context))
      +moved = (offset + Address.capacity(size) : U32)
      %built_same(~Element,~Context,~first,~second,~depth_of,~same,context,size,offset,limit,size_limit,first_half(U32.to_nat(offset),size,Shape.extent(depth_of(context)),room)) :
        {Tree.Node{built(Element,size,first(context),whole,offset),built(Element,size,first(context),whole,moved)} ==
          Tree.Node{_,built(Element,size,second(context),whole,moved)} : Tree.Tree<Element>}
      %built_same(~Element,~Context,~first,~second,~depth_of,~same,context,size,moved,limit,size_limit,moved_half(depth_of(context),size,offset,limit,size_limit,room)) :
        {Tree.Node{built(Element,size,first(context),whole,offset),built(Element,size,first(context),whole,moved)} ==
          Tree.Node{built(Element,size,first(context),whole,offset),_} : Tree.Tree<Element>}
      {==}

# Balanced trees of a context whose reads agree inside the capacity of its
# depth are equal.
def same_trees(~Element: Data,~Context: Data,~first: Context -> Tree.Tree<Element>,~second: Context -> Tree.Tree<Element>,~depth_of: Context -> Nat,
  ~same: @+context: Context -> @+index: U32 -> @+inside: {U32.is_lt(index,Address.capacity(depth_of(context))) == True{} : Bool} ->
    {Tree.fetch(Element,first(context),Address.capacity(depth_of(context)),index) == Tree.fetch(Element,second(context),Address.capacity(depth_of(context)),index) : Element},
  +context: Context,+first_balanced: Shape.Balanced(Element,depth_of(context),first(context)),+second_balanced: Shape.Balanced(Element,depth_of(context),second(context)),
  +limit: {Nat.is_le(depth_of(context),31n) == True{} : Bool}) -> {first(context) == second(context) : Tree.Tree<Element>}:
  +depth = depth_of(context)
  +whole = Address.capacity(depth)
  +first_built = built(Element,depth,first(context),whole,0)
  +second_built = built(Element,depth,second(context),whole,0)
  Equal.trans(Tree.Tree<Element>,first(context),first_built,second(context),
    Equal.sym(Tree.Tree<Element>,first_built,first(context),built_tree(~Element,depth,first(context),first_balanced,limit)),
    Equal.trans(Tree.Tree<Element>,first_built,second_built,second(context),
      built_same(~Element,~Context,~first,~second,~depth_of,~same,context,depth,0,limit,limit,Order.reflexive(Shape.extent(depth))),
      built_tree(~Element,depth,second(context),second_balanced,limit)))

# A read inside a balanced tree fetches at its capacity.
def read_fetch(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+balanced: Shape.Balanced(Element,depth,tree),
  +inside: {U32.is_lt(index,Address.capacity(depth)) == True{} : Bool}) ->
  {Tree.read(Element,tree,index) == Tree.fetch(Element,tree,Address.capacity(depth),index) : Element}:
  +capacity_equal = Shape.capacity(Element,depth,tree,balanced)
  %Equal.sym(U32,Tree.capacity(Element,tree),Address.capacity(depth),capacity_equal) : {Tree.fetch(Element,tree,_,U32.and(index,U32.sub(_,1))) == Tree.fetch(Element,tree,Address.capacity(depth),index) : Element}
  %Equal.sym(U32,U32.and(index,U32.sub(Address.capacity(depth),1)),index,Power.native_mask_preserves_bounded(depth,index)(inside)) : {Tree.fetch(Element,tree,Address.capacity(depth),_) == Tree.fetch(Element,tree,Address.capacity(depth),index) : Element}
  {==}
