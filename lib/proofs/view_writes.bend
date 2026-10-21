# A write through a view: element k of a part goes to the address of the k-th
# row-major coordinates of the view, elements in order. A scatter and the
# tile stores of the kernels are this list.
import Base
import ../tensor_view.bend as Views
import ./write_program.bend as Writes
import ./write_program_every.bend as Every
import ./write_program_readback.bend as Readback
import ./consecutive_writes.bend as Consecutive
import ./tensor_copy_values.bend as Copy
import ./tensor_row_major_coordinates.bend as Digits
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as U32Extrema
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./boolean_logic.bend as Boolean
import ./row_major_index.bend as RowMajor
import ../tensor_shape.bend as Shapes
import ./tensor_shape_refinement.bend as ShapeLaws

# The writes of count elements from position on.
def writes_from(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+position: Nat,+view: Views.View,+part: Part) ->
  List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{Copy.address(view,position),value(part,position)} <> writes_from(~Element,~Part,~value,rest,1n+position,view,part)

def writes(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+view: Views.View,+part: Part) -> List<&2,Writes.Assignment<Element>>:
  writes_from(~Element,~Part,~value,count,0n,view,part)

# The writes of first and then second elements are the writes of both.
def split(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,first: Nat,+second: Nat,+position: Nat,+view: Views.View,+part: Part) ->
  {writes_from(~Element,~Part,~value,Nat.add(first,second),position,view,part) ==
    List.append(&2,Writes.Assignment<Element>,writes_from(~Element,~Part,~value,first,position,view,part),
      writes_from(~Element,~Part,~value,second,Nat.add(position,first),view,part)) : List<&2,Writes.Assignment<Element>>}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {writes_from(~Element,~Part,~value,second,position,view,part) == writes_from(~Element,~Part,~value,second,_,view,part) : List<&2,Writes.Assignment<Element>>}
      {==}
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)) :
        {writes_from(~Element,~Part,~value,Nat.add(1n+rest,second),position,view,part) ==
          List.append(&2,Writes.Assignment<Element>,writes_from(~Element,~Part,~value,1n+rest,position,view,part),writes_from(~Element,~Part,~value,second,_,view,part)) : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,
        tail => Writes.Put{Copy.address(view,position),value(part,position)} <> tail,
        writes_from(~Element,~Part,~value,Nat.add(rest,second),1n+position,view,part),
        List.append(&2,Writes.Assignment<Element>,writes_from(~Element,~Part,~value,rest,1n+position,view,part),
          writes_from(~Element,~Part,~value,second,Nat.add(1n+position,rest),view,part)),
        split(~Element,~Part,~value,rest,second,1n+position,view,part))

# The writes from position contain the address of the element gap further on.
def covered_from(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,gap: Nat,+position: Nat,+view: Views.View,+part: Part,
  +inside: {Nat.is_lt(gap,count) == True{} : Bool}) ->
  {Readback.contains(~Element,Copy.address(view,Nat.add(position,gap)),writes_from(~Element,~Part,~value,count,position,view,part)) == True{} : Bool}:
  match count gap:
    case 0n gap: Empty.absurd({Readback.contains(~Element,Copy.address(view,Nat.add(position,gap)),[]) == True{} : Bool},Order.below_zero(gap,inside))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Readback.contains(~Element,Copy.address(view,_),writes_from(~Element,~Part,~value,1n+rest,position,view,part)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(Copy.address(view,position),Copy.address(view,position)),True{},Arithmetic.equal_reflexive(Copy.address(view,position))) :
        {_ || Readback.contains(~Element,Copy.address(view,position),writes_from(~Element,~Part,~value,rest,1n+position,view,part)) == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +more:
      +later = covered_from(~Element,~Part,~value,rest,more,1n+position,view,part,inside)
      %Equal.sym(Nat,Nat.add(position,1n+more),1n+Nat.add(position,more),Addition.successor_right(position,more)) :
        {Readback.contains(~Element,Copy.address(view,_),writes_from(~Element,~Part,~value,1n+rest,position,view,part)) == True{} : Bool}
      Readback.covered_right(~Element,Copy.address(view,Nat.add(1n+position,more)),
        [Writes.Put{Copy.address(view,position),value(part,position)}],writes_from(~Element,~Part,~value,rest,1n+position,view,part),later)

def covered(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+count: Nat,+index: Nat,+view: Views.View,+part: Part,
  +inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Readback.contains(~Element,Copy.address(view,index),writes(~Element,~Part,~value,count,view,part)) == True{} : Bool}:
  covered_from(~Element,~Part,~value,count,index,0n,view,part,inside)

# The word of base plus index times stride.
def encoded_line(+base: Nat,+index: Nat,+stride: U32) ->
  {U32.from_nat(Nat.add(Nat.mul(index,U32.to_nat(stride)),base)) == (U32.from_nat(base) + U32.from_nat(index) * stride : U32) : U32}:
  +product = Nat.mul(index,U32.to_nat(stride))
  Equal.trans(U32,U32.from_nat(Nat.add(product,base)),(U32.from_nat(product) + U32.from_nat(base) : U32),(U32.from_nat(base) + U32.from_nat(index) * stride : U32),
    Equal.sym(U32,(U32.from_nat(product) + U32.from_nat(base) : U32),U32.from_nat(Nat.add(product,base)),WordArithmetic.native_add(product,base)),
    Equal.trans(U32,(U32.from_nat(product) + U32.from_nat(base) : U32),(U32.from_nat(base) + U32.from_nat(product) : U32),(U32.from_nat(base) + U32.from_nat(index) * stride : U32),
      Arithmetic.add_commutative(U32.from_nat(product),U32.from_nat(base)),
      Equal.cong(U32,U32,word => (U32.from_nat(base) + word : U32),U32.from_nat(product),(U32.from_nat(index) * stride : U32),
        Equal.trans(U32,U32.from_nat(product),(U32.from_nat(index) * U32.from_nat(U32.to_nat(stride)) : U32),(U32.from_nat(index) * stride : U32),
          Equal.sym(U32,(U32.from_nat(index) * U32.from_nat(U32.to_nat(stride)) : U32),U32.from_nat(product),Multiplication.native_encoded_multiply(index,U32.to_nat(stride))),
          Equal.cong(U32,U32,word => (U32.from_nat(index) * word : U32),U32.from_nat(U32.to_nat(stride)),stride,Arithmetic.native_roundtrip(stride))))))
# The address of element (row, column) of a view with two axes, as a number
# and as a word.
def plane_digits(+target: U32,+rows: U32,+plane: U32,+row: Nat,+columns: U32,+step: U32,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool}) ->
  {Copy.address(Views.View{[Views.Axis{rows,plane},Views.Axis{columns,step}],target},Nat.add(Nat.mul(row,U32.to_nat(columns)),column)) ==
    U32.from_nat(Nat.add(Nat.mul(column,U32.to_nat(step)),Nat.add(Nat.mul(row,U32.to_nat(plane)),U32.to_nat(target)))) : U32}:
  +points = {[Digits.Point{rows,row,plane,False{}},Digits.Point{columns,column,step,False{}}] : List<&2,Digits.Point>}
  +valid = {Shape.Both{row_inside,Shape.Both{column_inside,Unit{}}} : Digits.Valid(points)}
  Digits.copied(points,target,valid,Digits.inside(points,valid))
def plane_address(+target: U32,+rows: U32,+plane: U32,+row: Nat,+columns: U32,+step: U32,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool}) ->
  {Copy.address(Views.View{[Views.Axis{rows,plane},Views.Axis{columns,step}],target},Nat.add(Nat.mul(row,U32.to_nat(columns)),column)) ==
    (target + U32.from_nat(row) * plane + U32.from_nat(column) * step : U32) : U32}:
  +inner = Nat.add(Nat.mul(row,U32.to_nat(plane)),U32.to_nat(target))
  Equal.trans(U32,Copy.address(Views.View{[Views.Axis{rows,plane},Views.Axis{columns,step}],target},Nat.add(Nat.mul(row,U32.to_nat(columns)),column)),
    U32.from_nat(Nat.add(Nat.mul(column,U32.to_nat(step)),inner)),(target + U32.from_nat(row) * plane + U32.from_nat(column) * step : U32),
    plane_digits(target,rows,plane,row,columns,step,column,row_inside,column_inside),
    Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(column,U32.to_nat(step)),inner)),(U32.from_nat(inner) + U32.from_nat(column) * step : U32),
      (target + U32.from_nat(row) * plane + U32.from_nat(column) * step : U32),
      encoded_line(inner,column,step),
      Equal.cong(U32,U32,word => (word + U32.from_nat(column) * step : U32),U32.from_nat(inner),(target + U32.from_nat(row) * plane : U32),
        Equal.trans(U32,U32.from_nat(inner),(U32.from_nat(U32.to_nat(target)) + U32.from_nat(row) * plane : U32),(target + U32.from_nat(row) * plane : U32),
          encoded_line(U32.to_nat(target),row,plane),
          Equal.cong(U32,U32,word => (word + U32.from_nat(row) * plane : U32),U32.from_nat(U32.to_nat(target)),target,Arithmetic.native_roundtrip(target))))))

# A tile store writes its parts one after another. Each part is empty or a
# run of the writes of a view.
def parts(~Element: Data,~Context: Data,~part: @+context: Context -> @+index: Nat -> List<&2,Writes.Assignment<Element>>,+context: Context,count: Nat) ->
  List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+ +rest: List.append(&2,Writes.Assignment<Element>,parts(~Element,~Context,~part,context,rest),part(context,rest))

def append_empty(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>) ->
  {List.append(&2,Writes.Assignment<Element>,writes,Nil{}) == writes : List<&2,Writes.Assignment<Element>>}:
  match writes:
    case Nil{}: {==}
    case Con{+head,+tail}:
      %append_empty(~Element,tail) : {head <> List.append(&2,Writes.Assignment<Element>,tail,Nil{}) == head <> _ : List<&2,Writes.Assignment<Element>>}
      {==}

# Parts below used that are each size writes, part index from base plus index
# times size, make the writes of count times size elements from base.
def grouped(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,
  ~part: @+context: Context -> @+index: Nat -> List<&2,Writes.Assignment<Element>>,
  ~view: @+context: Context -> Views.View,~source: @+context: Context -> Part,~base: @+context: Context -> Nat,~size: @+context: Context -> Nat,
  ~used: @+context: Context -> Nat,
  ~present: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,used(context)) == True{} : Bool} ->
    {part(context,index) == writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(index,size(context))),view(context),source(context)) : List<&2,Writes.Assignment<Element>>},
  +context: Context,count: Nat,+gap: Nat,+room: {Nat.add(count,gap) == used(context) : Nat}) ->
  {parts(~Element,~Context,~part,context,count) ==
    writes_from(~Element,~Part,~value,Nat.mul(count,size(context)),base(context),view(context),source(context)) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +width = size(context)
      +first = base(context)
      +done = Nat.mul(rest,width)
      +moved = Equal.trans(Nat,Nat.add(rest,1n+gap),1n+Nat.add(rest,gap),used(context),Addition.successor_right(rest,gap),room)
      +earlier = grouped(~Element,~Part,~value,~Context,~part,~view,~source,~base,~size,~used,~present,context,rest,1n+gap,moved)
      +here = present(context,rest,Order.below_complement(rest,gap,used(context),moved))
      +earlier_writes = writes_from(~Element,~Part,~value,done,first,view(context),source(context))
      +later_writes = writes_from(~Element,~Part,~value,width,Nat.add(first,done),view(context),source(context))
      +joined = List.append(&2,Writes.Assignment<Element>,earlier_writes,later_writes)
      %Order.commutative(done,width) :
        {parts(~Element,~Context,~part,context,1n+rest) == writes_from(~Element,~Part,~value,_,first,view(context),source(context)) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,writes_from(~Element,~Part,~value,Nat.add(done,width),first,view(context),source(context)),joined,
        split(~Element,~Part,~value,done,width,first,view(context),source(context))) :
        {parts(~Element,~Context,~part,context,1n+rest) == _ : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,parts(~Element,~Context,~part,context,rest),earlier_writes,earlier) :
        {List.append(&2,Writes.Assignment<Element>,_,part(context,rest)) == joined : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,part(context,rest),later_writes,here) :
        {List.append(&2,Writes.Assignment<Element>,earlier_writes,_) == joined : List<&2,Writes.Assignment<Element>>}
      {==}

# Empty parts after the used ones leave the list of the used ones.
def padded(~Element: Data,~Context: Data,~part: @+context: Context -> @+index: Nat -> List<&2,Writes.Assignment<Element>>,
  ~used: @+context: Context -> Nat,~limit: @+context: Context -> Nat,
  ~absent: @+context: Context -> @+extra: Nat -> @+inside: {Nat.is_lt(Nat.add(used(context),extra),limit(context)) == True{} : Bool} ->
    {part(context,Nat.add(used(context),extra)) == Nil{} : List<&2,Writes.Assignment<Element>>},
  +context: Context,extra: Nat,+gap: Nat,+room: {Nat.add(Nat.add(used(context),extra),gap) == limit(context) : Nat}) ->
  {parts(~Element,~Context,~part,context,Nat.add(used(context),extra)) == parts(~Element,~Context,~part,context,used(context)) : List<&2,Writes.Assignment<Element>>}:
  match extra:
    case 0n:
      %Equal.sym(Nat,Nat.add(used(context),0n),used(context),Addition.zero_right(used(context))) :
        {parts(~Element,~Context,~part,context,_) == parts(~Element,~Context,~part,context,used(context)) : List<&2,Writes.Assignment<Element>>}
      {==}
    case 1n+ +more:
      +first = used(context)
      +position = Nat.add(first,more)
      +shifted = {
        %Addition.successor_right(first,more) : {Nat.add(_,gap) == limit(context) : Nat}
        room : {Nat.add(1n+position,gap) == limit(context) : Nat}}
      +moved = Equal.trans(Nat,Nat.add(position,1n+gap),1n+Nat.add(position,gap),limit(context),Addition.successor_right(position,gap),shifted)
      +empty = absent(context,more,Order.below_complement(position,gap,limit(context),moved))
      +earlier = padded(~Element,~Context,~part,~used,~limit,~absent,context,more,1n+gap,moved)
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+position,Addition.successor_right(first,more)) :
        {parts(~Element,~Context,~part,context,_) == parts(~Element,~Context,~part,context,first) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,part(context,position),Nil{},empty) :
        {List.append(&2,Writes.Assignment<Element>,parts(~Element,~Context,~part,context,position),_) == parts(~Element,~Context,~part,context,first) : List<&2,Writes.Assignment<Element>>}
      Equal.trans(List<&2,Writes.Assignment<Element>>,List.append(&2,Writes.Assignment<Element>,parts(~Element,~Context,~part,context,position),Nil{}),
        parts(~Element,~Context,~part,context,position),parts(~Element,~Context,~part,context,first),
        append_empty(~Element,parts(~Element,~Context,~part,context,position)),earlier)

# Limit parts whose first used ones are size writes each, from base on, and
# whose others are empty, are the writes of used times size elements.
def masked(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Context: Data,
  ~part: @+context: Context -> @+index: Nat -> List<&2,Writes.Assignment<Element>>,
  ~view: @+context: Context -> Views.View,~source: @+context: Context -> Part,~base: @+context: Context -> Nat,~size: @+context: Context -> Nat,
  ~used: @+context: Context -> Nat,~limit: @+context: Context -> Nat,
  ~present: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,used(context)) == True{} : Bool} ->
    {part(context,index) == writes_from(~Element,~Part,~value,size(context),Nat.add(base(context),Nat.mul(index,size(context))),view(context),source(context)) : List<&2,Writes.Assignment<Element>>},
  ~absent: @+context: Context -> @+extra: Nat -> @+inside: {Nat.is_lt(Nat.add(used(context),extra),limit(context)) == True{} : Bool} ->
    {part(context,Nat.add(used(context),extra)) == Nil{} : List<&2,Writes.Assignment<Element>>},
  +context: Context,+ordered: {Nat.is_le(used(context),limit(context)) == True{} : Bool}) ->
  {parts(~Element,~Context,~part,context,limit(context)) ==
    writes_from(~Element,~Part,~value,Nat.mul(used(context),size(context)),base(context),view(context),source(context)) : List<&2,Writes.Assignment<Element>>}:
  +first = used(context)
  +gap = Nat.sub(limit(context),first)
  +whole = Order.restore_shift(first,limit(context),ordered)
  +written = writes_from(~Element,~Part,~value,Nat.mul(first,size(context)),base(context),view(context),source(context))
  %whole : {parts(~Element,~Context,~part,context,_) == written : List<&2,Writes.Assignment<Element>>}
  Equal.trans(List<&2,Writes.Assignment<Element>>,parts(~Element,~Context,~part,context,Nat.add(first,gap)),parts(~Element,~Context,~part,context,first),written,
    padded(~Element,~Context,~part,~used,~limit,~absent,context,gap,0n,
      Equal.trans(Nat,Nat.add(Nat.add(first,gap),0n),Nat.add(first,gap),limit(context),Addition.zero_right(Nat.add(first,gap)),whole)),
    grouped(~Element,~Part,~value,~Context,~part,~view,~source,~base,~size,~used,~present,context,first,0n,Addition.zero_right(first)))

# A count clipped to a limit is at most the limit.
def limited(+limit: U32,+count: U32) -> {Nat.is_le(U32.to_nat(U32.min(limit,count)),U32.to_nat(limit)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.min(limit,count)),Nat.min(U32.to_nat(limit),U32.to_nat(count)),U32Extrema.minimum(limit,count)) :
    {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  NatExtrema.minimum_left(U32.to_nat(limit),U32.to_nat(count))

# An index is below a count clipped to a limit exactly when it is below both.
def clipped_both(+index: Nat,+limit: U32,+count: U32) ->
  {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)) : Bool}:
  +bound = U32.to_nat(limit)
  +total = U32.to_nat(count)
  Equal.trans(Bool,Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),Nat.is_lt(index,Nat.min(bound,total)),Nat.is_lt(index,bound) && Nat.is_lt(index,total),
    Equal.cong(Nat,Bool,end => Nat.is_lt(index,end),U32.to_nat(U32.min(limit,count)),Nat.min(bound,total),U32Extrema.minimum(limit,count)),
    NatExtrema.below_minimum(index,bound,total))
def clipped_small(+index: Nat,+limit: U32,+count: U32,+below: {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == True{} : Bool}) ->
  {Nat.is_lt(index,U32.to_nat(limit)) == True{} : Bool}:
  Boolean.left(Nat.is_lt(index,U32.to_nat(limit)),Nat.is_lt(index,U32.to_nat(count)),
    Equal.trans(Bool,Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)),Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),True{},
      Equal.sym(Bool,Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)),clipped_both(index,limit,count)),below))
def clipped_below(+index: Nat,+limit: U32,+count: U32,+below: {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == True{} : Bool}) ->
  {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}:
  Boolean.right(Nat.is_lt(index,U32.to_nat(limit)),Nat.is_lt(index,U32.to_nat(count)),
    Equal.trans(Bool,Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)),Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),True{},
      Equal.sym(Bool,Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)),clipped_both(index,limit,count)),below))
def clipped(+index: Nat,+limit: U32,+count: U32,+small: {Nat.is_lt(index,U32.to_nat(limit)) == True{} : Bool},+below: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),Nat.is_lt(index,U32.to_nat(limit)) && Nat.is_lt(index,U32.to_nat(count)),True{},
    clipped_both(index,limit,count),Boolean.and_true(Nat.is_lt(index,U32.to_nat(limit)),Nat.is_lt(index,U32.to_nat(count)),small,below))

# An index at most a word limit compares with a word as its number does.
def word_compared(+index: Nat,+limit: U32,+count: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(index,U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(index),count) == Nat.is_lt(index,U32.to_nat(count)) : Bool}:
  Equal.trans(Bool,U32.is_lt(U32.from_nat(index),count),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(count)),Nat.is_lt(index,U32.to_nat(count)),
    Comparison.lt(U32.from_nat(index),count),
    Equal.cong(Nat,Bool,value => Nat.is_lt(value,U32.to_nat(count)),U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,limit,limit_ok,small)))

# An index below the limit but not below the clipped count is not below the
# count.
def clipped_beyond(+index: Nat,+limit: U32,+count: U32,+small: {Nat.is_lt(index,U32.to_nat(limit)) == True{} : Bool},
  +outside: {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == False{} : Bool}) -> {Nat.is_lt(index,U32.to_nat(count)) == False{} : Bool}:
  +bound = U32.to_nat(limit)
  +total = U32.to_nat(count)
  Equal.trans(Bool,Nat.is_lt(index,total),Nat.is_lt(index,bound) && Nat.is_lt(index,total),False{},
    Equal.cong(Bool,Bool,first => first && Nat.is_lt(index,total),True{},Nat.is_lt(index,bound),Equal.sym(Bool,Nat.is_lt(index,bound),True{},small)),
    Equal.trans(Bool,Nat.is_lt(index,bound) && Nat.is_lt(index,total),Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),False{},
      Equal.sym(Bool,Nat.is_lt(index,U32.to_nat(U32.min(limit,count))),Nat.is_lt(index,bound) && Nat.is_lt(index,total),clipped_both(index,limit,count)),outside))

# An index below a count clipped to a limit is below the count as a word.
def clipped_inside(+index: Nat,+limit: U32,+count: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +below: {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(index),count) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(U32.from_nat(index),count),Nat.is_lt(index,U32.to_nat(count)),True{},
    word_compared(index,limit,count,limit_ok,Bounds.strict_le(index,U32.to_nat(limit),clipped_small(index,limit,count,below))),
    clipped_below(index,limit,count,below))

# An index below the limit but not below the clipped count is not below the
# count as a word.
def clipped_outside(+index: Nat,+limit: U32,+count: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_lt(index,U32.to_nat(limit)) == True{} : Bool},
  +outside: {Nat.is_lt(index,U32.to_nat(U32.min(limit,count))) == False{} : Bool}) -> {U32.is_lt(U32.from_nat(index),count) == False{} : Bool}:
  Equal.trans(Bool,U32.is_lt(U32.from_nat(index),count),Nat.is_lt(index,U32.to_nat(count)),False{},
    word_compared(index,limit,count,limit_ok,Bounds.strict_le(index,U32.to_nat(limit),small)),
    clipped_beyond(index,limit,count,small,outside))

# A row counted from a first word is below a count exactly when it is below
# the rows the count leaves after first.
def offset_compared(+first: U32,+row: Nat,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(count)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),row),U32.to_nat(cap)) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_lt((first + U32.from_nat(row) : U32),count) == Nat.is_lt(row,U32.to_nat((count - first : U32))) : Bool}:
  +start = U32.to_nat(first)
  +total = U32.to_nat(count)
  +sum = (first + U32.from_nat(row) : U32)
  +left = Nat.sub(total,start)
  +row_small = Order.transitive(row,Nat.add(start,row),U32.to_nat(cap),
    %Order.commutative(row,start) : {Nat.is_le(row,_) == True{} : Bool} Order.le_add(row,start),room)
  +row_exact = Bounds.roundtrip(row,cap,cap_ok,row_small)
  +sum_exact = Equal.trans(Nat,U32.to_nat(sum),Nat.add(start,U32.to_nat(U32.from_nat(row))),Nat.add(start,row),
    Arithmetic.add(first,U32.from_nat(row),cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,row_exact) : {Nat.is_le(Nat.add(start,_),U32.to_nat(cap)) == True{} : Bool} room),
    Equal.cong(Nat,Nat,value => Nat.add(start,value),U32.to_nat(U32.from_nat(row)),row,row_exact))
  +difference = Arithmetic.subtract(count,first,cap,cap_ok,ordered,Order.transitive(left,total,U32.to_nat(cap),Order.sub_le(total,start),count_room))
  %Equal.sym(Nat,U32.to_nat((count - first : U32)),left,difference) : {U32.is_lt(sum,count) == Nat.is_lt(row,_) : Bool}
  %Equal.sym(Bool,U32.is_lt(sum,count),Nat.is_lt(U32.to_nat(sum),total),Comparison.lt(sum,count)) : {_ == Nat.is_lt(row,left) : Bool}
  %Equal.sym(Nat,U32.to_nat(sum),Nat.add(start,row),sum_exact) : {Nat.is_lt(_,total) == Nat.is_lt(row,left) : Bool}
  %Order.restore_shift(start,total,ordered) : {Nat.is_lt(Nat.add(start,row),_) == Nat.is_lt(row,left) : Bool}
  Order.lt_offset(start,row,left)

# Every write of a view of rows by columns elements has a property when each
# cell has it at its address. The cells read an environment passed at run
# time.
def every_columns(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~view: @+environment: Environment -> Views.View,~part: @+environment: Environment -> Part,
  ~rows: @+environment: Environment -> Nat,~columns: @+environment: Environment -> Nat,
  ~cell: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,rows(environment)) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,columns(environment)) == True{} : Bool} ->
    Holds(target(environment),Copy.address(view(environment),Nat.add(Nat.mul(row,columns(environment)),column)),value(part(environment),Nat.add(Nat.mul(row,columns(environment)),column))),
  +environment: Environment,+row: Nat,+row_inside: {Nat.is_lt(row,rows(environment)) == True{} : Bool},rest: Nat,+column: Nat,+room: {Nat.add(column,rest) == columns(environment) : Nat}) ->
  Every.Every(~Element,~Target,~Holds,target(environment),writes_from(~Element,~Part,~value,rest,Nat.add(Nat.mul(row,columns(environment)),column),view(environment),part(environment))):
  match rest:
    case 0n: Unit{}
    case 1n+ +more:
      +first = Nat.mul(row,columns(environment))
      +inside = {
        %room : {Nat.is_lt(column,_) == True{} : Bool}
        Order.below_sum(column,more) : {Nat.is_lt(column,columns(environment)) == True{} : Bool}}
      +later = every_columns(~Element,~Part,~value,~Environment,~Target,~target,~Holds,~view,~part,~rows,~columns,~cell,environment,row,row_inside,more,1n+column,
        %Addition.successor_right(column,more) : {_ == columns(environment) : Nat} room)
      Shape.Both{cell(environment,row,column,row_inside,inside),
        %Addition.successor_right(first,column) : Every.Every(~Element,~Target,~Holds,target(environment),writes_from(~Element,~Part,~value,more,_,view(environment),part(environment))) later}
def every_rows(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~view: @+environment: Environment -> Views.View,~part: @+environment: Environment -> Part,
  ~rows: @+environment: Environment -> Nat,~columns: @+environment: Environment -> Nat,
  ~cell: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,rows(environment)) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,columns(environment)) == True{} : Bool} ->
    Holds(target(environment),Copy.address(view(environment),Nat.add(Nat.mul(row,columns(environment)),column)),value(part(environment),Nat.add(Nat.mul(row,columns(environment)),column))),
  +environment: Environment,rest: Nat,+row: Nat,+room: {Nat.add(row,rest) == rows(environment) : Nat}) ->
  Every.Every(~Element,~Target,~Holds,target(environment),
    writes_from(~Element,~Part,~value,Nat.mul(rest,columns(environment)),Nat.mul(row,columns(environment)),view(environment),part(environment))):
  match rest:
    case 0n: Unit{}
    case 1n+ +more:
      +width = columns(environment)
      +first = Nat.mul(row,width)
      +row_inside = {
        %room : {Nat.is_lt(row,_) == True{} : Bool}
        Order.below_sum(row,more) : {Nat.is_lt(row,rows(environment)) == True{} : Bool}}
      +line = every_columns(~Element,~Part,~value,~Environment,~Target,~target,~Holds,~view,~part,~rows,~columns,~cell,environment,row,row_inside,width,0n,{==})
      +later = every_rows(~Element,~Part,~value,~Environment,~Target,~target,~Holds,~view,~part,~rows,~columns,~cell,environment,more,1n+row,
        %Addition.successor_right(row,more) : {_ == rows(environment) : Nat} room)
      +here = writes_from(~Element,~Part,~value,width,first,view(environment),part(environment))
      +rest_writes = writes_from(~Element,~Part,~value,Nat.mul(more,width),Nat.add(first,width),view(environment),part(environment))
      %Equal.sym(List<&2,Writes.Assignment<Element>>,writes_from(~Element,~Part,~value,Nat.add(width,Nat.mul(more,width)),first,view(environment),part(environment)),
        List.append(&2,Writes.Assignment<Element>,here,rest_writes),split(~Element,~Part,~value,width,Nat.mul(more,width),first,view(environment),part(environment))) :
        Every.Every(~Element,~Target,~Holds,target(environment),_)
      Every.appended(~Element,~Target,~Holds,target(environment),here,rest_writes,
        %Addition.zero_right(first) : Every.Every(~Element,~Target,~Holds,target(environment),writes_from(~Element,~Part,~value,width,_,view(environment),part(environment))) line,
        %Order.commutative(width,first) :
          Every.Every(~Element,~Target,~Holds,target(environment),writes_from(~Element,~Part,~value,Nat.mul(more,width),_,view(environment),part(environment))) later)
def every_plane(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,~view: @+environment: Environment -> Views.View,~part: @+environment: Environment -> Part,
  ~rows: @+environment: Environment -> Nat,~columns: @+environment: Environment -> Nat,
  ~cell: @+environment: Environment -> @+row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(row,rows(environment)) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,columns(environment)) == True{} : Bool} ->
    Holds(target(environment),Copy.address(view(environment),Nat.add(Nat.mul(row,columns(environment)),column)),value(part(environment),Nat.add(Nat.mul(row,columns(environment)),column))),
  +environment: Environment) ->
  Every.Every(~Element,~Target,~Holds,target(environment),writes(~Element,~Part,~value,Nat.mul(rows(environment),columns(environment)),view(environment),part(environment))):
  every_rows(~Element,~Part,~value,~Environment,~Target,~target,~Holds,~view,~part,~rows,~columns,~cell,environment,rows(environment),0n,{==})

# The writes of a view of rows by columns elements contain the address of
# each cell.
def covered_plane(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+rows: Nat,+columns: Nat,+row: Nat,+column: Nat,+view: Views.View,+part: Part,
  +row_inside: {Nat.is_lt(row,rows) == True{} : Bool},+column_inside: {Nat.is_lt(column,columns) == True{} : Bool}) ->
  {Readback.contains(~Element,Copy.address(view,Nat.add(Nat.mul(row,columns),column)),writes(~Element,~Part,~value,Nat.mul(rows,columns),view,part)) == True{} : Bool}:
  covered(~Element,~Part,~value,Nat.mul(rows,columns),Nat.add(Nat.mul(row,columns),column),view,part,RowMajor.radix(row,rows,column,columns,row_inside,column_inside))

# ---- Views at consecutive addresses ----

# A view whose elements lie at consecutive addresses writes the consecutive
# list.
def on_line(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+position: Nat,+view: Views.View,+part: Part,
  +base: Nat,line: Copy.Line(view,base,count,position)) ->
  {writes_from(~Element,~Part,~value,count,position,view,part) == Consecutive.writes(~Element,~Part,~value,count,base,position,part) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Shape.Both{here,later} = line
      +start = U32.from_nat(Nat.add(base,position))
      %Equal.sym(U32,Copy.address(view,position),start,here) :
        {Writes.Put{_,value(part,position)} <> writes_from(~Element,~Part,~value,rest,1n+position,view,part) ==
          Writes.Put{start,value(part,position)} <> Consecutive.writes(~Element,~Part,~value,rest,base,1n+position,part) : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => Writes.Put{start,value(part,position)} <> tail,
        writes_from(~Element,~Part,~value,rest,1n+position,view,part),Consecutive.writes(~Element,~Part,~value,rest,base,1n+position,part),
        on_line(~Element,~Part,~value,rest,1n+position,view,part,base,later))

# The writes of a contiguous view at an address of the view store the
# element there; before the view and after it the previous element stays.
def contiguous_hit(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+count: Nat,+axes: List<&2,Views.Axis>,+offset: U32,+part: Part,
  +previous: Element,index: Nat,+length: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(axes)) == count : Nat},+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(U32.to_nat(offset),count),U32.to_nat(length)) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,Views.View{axes,offset},part),U32.from_nat(Nat.add(U32.to_nat(offset),index)),previous) == value(part,index) : Element}:
  +view = {Views.View{axes,offset} : Views.View}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,writes_from(~Element,~Part,~value,count,0n,view,part),Consecutive.writes(~Element,~Part,~value,count,U32.to_nat(offset),0n,part),
    on_line(~Element,~Part,~value,count,0n,view,part,U32.to_nat(offset),Copy.contiguous_start(axes,offset,count,nonempty,contiguous,exact))) :
    {Readback.selected(~Element,_,U32.from_nat(Nat.add(U32.to_nat(offset),index)),previous) == value(part,index) : Element}
  Consecutive.hit(~Element,~Part,~value,count,U32.to_nat(offset),0n,part,previous,index,length,bound,inside)
def contiguous_before(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+count: Nat,+axes: List<&2,Views.Axis>,+offset: U32,+part: Part,
  +previous: Element,+query: Nat,+length: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(axes)) == count : Nat},+before: {Nat.is_lt(query,U32.to_nat(offset)) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(U32.to_nat(offset),count),U32.to_nat(length)) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,Views.View{axes,offset},part),U32.from_nat(query),previous) == previous : Element}:
  +view = {Views.View{axes,offset} : Views.View}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,writes_from(~Element,~Part,~value,count,0n,view,part),Consecutive.writes(~Element,~Part,~value,count,U32.to_nat(offset),0n,part),
    on_line(~Element,~Part,~value,count,0n,view,part,U32.to_nat(offset),Copy.contiguous_start(axes,offset,count,nonempty,contiguous,exact))) :
    {Readback.selected(~Element,_,U32.from_nat(query),previous) == previous : Element}
  Consecutive.before(~Element,~Part,~value,count,U32.to_nat(offset),0n,part,previous,query,length,bound,
    %Equal.sym(Nat,Nat.add(U32.to_nat(offset),0n),U32.to_nat(offset),Addition.zero_right(U32.to_nat(offset))) : {Nat.is_lt(query,_) == True{} : Bool} before)
def contiguous_after(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+count: Nat,+axes: List<&2,Views.Axis>,+offset: U32,+part: Part,
  +previous: Element,+query: Nat,+length: U32,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(axes)) == count : Nat},+beyond: {Nat.is_le(Nat.add(U32.to_nat(offset),count),query) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(length)) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,Views.View{axes,offset},part),U32.from_nat(query),previous) == previous : Element}:
  +view = {Views.View{axes,offset} : Views.View}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,writes_from(~Element,~Part,~value,count,0n,view,part),Consecutive.writes(~Element,~Part,~value,count,U32.to_nat(offset),0n,part),
    on_line(~Element,~Part,~value,count,0n,view,part,U32.to_nat(offset),Copy.contiguous_start(axes,offset,count,nonempty,contiguous,exact))) :
    {Readback.selected(~Element,_,U32.from_nat(query),previous) == previous : Element}
  Consecutive.after(~Element,~Part,~value,count,U32.to_nat(offset),0n,part,previous,query,length,beyond,query_inside)

# Every write of a view that fits a length lies below a capacity of at least
# that length.
def inside_from(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+position: Nat,+view: Views.View,+part: Part,+length: U32,+capacity: U32,
  +fits: {Views.fits_nonempty(False{},view,length) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(length),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,writes_from(~Element,~Part,~value,count,position,view,part)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +address = Copy.address(view,position)
      Shape.Both{
        %Equal.sym(Bool,U32.is_lt(address,capacity),Nat.is_lt(U32.to_nat(address),U32.to_nat(capacity)),Comparison.lt(address,capacity)) : {_ == True{} : Bool}
        Bounds.nat_trans(U32.to_nat(address),U32.to_nat(length),U32.to_nat(capacity),Copy.address_inside(view,position,length,fits),room),
        inside_from(~Element,~Part,~value,rest,1n+position,view,part,length,capacity,fits,room)}

# An axis of extent one adds nothing to the addresses of a view.
def unit_axis(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+position: Nat,+stride: U32,+axes: List<&2,Views.Axis>,+offset: U32,
  +part: Part) ->
  {writes_from(~Element,~Part,~value,count,position,Views.View{Views.Axis{1,stride} <> axes,offset},part) ==
    writes_from(~Element,~Part,~value,count,position,Views.View{axes,offset},part) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,
        tail => Writes.Put{Copy.address(Views.View{axes,offset},position),value(part,position)} <> tail,
        writes_from(~Element,~Part,~value,rest,1n+position,Views.View{Views.Axis{1,stride} <> axes,offset},part),
        writes_from(~Element,~Part,~value,rest,1n+position,Views.View{axes,offset},part),
        unit_axis(~Element,~Part,~value,rest,1n+position,stride,axes,offset,part))
