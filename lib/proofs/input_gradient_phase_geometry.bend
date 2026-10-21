# The native phase axes enumerate the bounded congruent input positions and taps.
import Base
import ../kernel_input_gradient.bend as Kernel
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as DivisionOrder
import ./u32_division.bend as NativeDivision
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_comparison.bend as Comparison
import ./word_subtraction.bend as Subtraction
import ./nat_division.bend as Division
import ./binary_radix.bend as Radix

def axis_count(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  count

def axis_taps(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  taps

def axis_tap(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  tap

def axis_source(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  source

def tail(+size: U32,+first: U32) -> Nat:
  Nat.sub(Nat.sub(U32.to_nat(size),1n),U32.to_nat(first))

def tail_exact(+size: U32,+first: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool}) ->
  {U32.to_nat((size - 1 - first : U32)) == tail(size,first) : Nat}:
  +positive = Order.positive_bound(U32.to_nat(first),U32.to_nat(size),inside)
  +last = Nat.sub(U32.to_nat(size),1n)
  +last_room = Order.transitive(last,U32.to_nat(size),U32.to_nat(capacity),Order.sub_le(U32.to_nat(size),1n),size_room)
  +last_exact = Arithmetic.subtract(size,1,capacity,capacity_ok,Order.positive_one(U32.to_nat(size),positive),last_room)
  %last_exact : {U32.to_nat((size - 1 - first : U32)) == Nat.sub(_,U32.to_nat(first)) : Nat}
  Arithmetic.subtract((size - 1 : U32),first,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat((size - 1 : U32)),last,last_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
    Order.before_last(U32.to_nat(first),U32.to_nat(size),inside),
    %Equal.sym(Nat,U32.to_nat((size - 1 : U32)),last,last_exact) : {Nat.is_le(Nat.sub(_,U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(tail(size,first),last,U32.to_nat(capacity),Order.sub_le(last,U32.to_nat(first)),last_room))

def quotient_room(+value: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(value,stride),value) == True{} : Bool}:
  Order.transitive(Nat.div(value,stride),Nat.mul(Nat.div(value,stride),stride),value,
    RowMajor.product_covers(Nat.div(value,stride),stride,positive),DivisionOrder.product_below(value,stride,positive))

def count_room(+size: U32,+first: U32,+stride: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool}) ->
  {Nat.is_le(1n+Nat.div(tail(size,first),U32.to_nat(stride)),U32.to_nat(size)) == True{} : Bool}:
  +last = Nat.sub(U32.to_nat(size),1n)
  +below = Order.transitive(Nat.div(tail(size,first),U32.to_nat(stride)),tail(size,first),last,
    quotient_room(tail(size,first),U32.to_nat(stride),positive),Order.sub_le(last,U32.to_nat(first)))
  %Order.restore_shift(1n,U32.to_nat(size),Order.positive_one(U32.to_nat(size),Order.positive_bound(U32.to_nat(first),U32.to_nat(size),inside))) :
    {Nat.is_le(1n+Nat.div(tail(size,first),U32.to_nat(stride)),_) == True{} : Bool}
  %Order.commutative(Nat.div(tail(size,first),U32.to_nat(stride)),1n) :
    {Nat.is_le(_,1n+last) == True{} : Bool}
  %Order.commutative(last,1n) : {Nat.is_le(Nat.add(Nat.div(tail(size,first),U32.to_nat(stride)),1n),_) == True{} : Bool}
  Order.add_le(Nat.div(tail(size,first),U32.to_nat(stride)),last,1n,below)

def stepped_count(+size: U32,+first: U32,+stride: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool}) ->
  {U32.to_nat(((size - 1 - first) / stride + 1 : U32)) == 1n+Nat.div(tail(size,first),U32.to_nat(stride)) : Nat}:
  +tail_exact = tail_exact(size,first,capacity,capacity_ok,size_room,inside)
  +divided = Equal.trans(Nat,U32.to_nat(((size - 1 - first) / stride : U32)),Nat.div(U32.to_nat((size - 1 - first : U32)),U32.to_nat(stride)),
    Nat.div(tail(size,first),U32.to_nat(stride)),NativeDivision.divide((size - 1 - first : U32),stride,capacity,capacity_ok,stride_room,positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(stride)),U32.to_nat((size - 1 - first : U32)),tail(size,first),tail_exact))
  +room = Order.transitive(1n+Nat.div(tail(size,first),U32.to_nat(stride)),U32.to_nat(size),U32.to_nat(capacity),count_room(size,first,stride,positive,inside),size_room)
  %Order.commutative(Nat.div(tail(size,first),U32.to_nat(stride)),1n) :
    {U32.to_nat(((size - 1 - first) / stride + 1 : U32)) == _ : Nat}
  %divided : {U32.to_nat(((size - 1 - first) / stride + 1 : U32)) == Nat.add(_,1n) : Nat}
  Arithmetic.add(((size - 1 - first) / stride : U32),1,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(((size - 1 - first) / stride : U32)),Nat.div(tail(size,first),U32.to_nat(stride)),divided) :
      {Nat.is_le(Nat.add(_,1n),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.div(tail(size,first),U32.to_nat(stride)),1n),1n+Nat.div(tail(size,first),U32.to_nat(stride)),Order.commutative(Nat.div(tail(size,first),U32.to_nat(stride)),1n)) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)

def count_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool}) ->
  {U32.to_nat(axis_count(Kernel.axis(first,size,kernel,stride,padding))) == 1n+Nat.div(tail(size,first),U32.to_nat(stride)) : Nat}:
  stepped_count(size,first,stride,capacity,capacity_ok,size_room,stride_room,positive,inside)

def position_inside(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +first_inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool},
  +inside: {Nat.is_lt(position,U32.to_nat(axis_count(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(size)) == True{} : Bool}:
  +last = Nat.sub(U32.to_nat(size),1n)
  +q = Nat.div(tail(size,first),U32.to_nat(stride))
  +within = Division.lt_succ_le(position,q,
    %count_exact(first,size,kernel,stride,padding,capacity,capacity_ok,size_room,stride_room,positive,first_inside) : {Nat.is_lt(position,_) == True{} : Bool} inside)
  +step_below = Order.transitive(Nat.mul(position,U32.to_nat(stride)),Nat.mul(q,U32.to_nat(stride)),tail(size,first),
    RowMajor.mul_le(position,q,U32.to_nat(stride),within),DivisionOrder.product_below(tail(size,first),U32.to_nat(stride),positive))
  +point_below = {
    %Order.restore_shift(U32.to_nat(first),last,Order.before_last(U32.to_nat(first),U32.to_nat(size),first_inside)) :
      {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),_) == True{} : Bool}
    %Order.commutative(Nat.mul(position,U32.to_nat(stride)),U32.to_nat(first)) : {Nat.is_le(_,Nat.add(U32.to_nat(first),tail(size,first))) == True{} : Bool}
    %Order.commutative(tail(size,first),U32.to_nat(first)) : {Nat.is_le(Nat.add(Nat.mul(position,U32.to_nat(stride)),U32.to_nat(first)),_) == True{} : Bool}
    Order.add_le(Nat.mul(position,U32.to_nat(stride)),tail(size,first),U32.to_nat(first),step_below) :
    {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),last) == True{} : Bool}}
  Order.le_lt_trans(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),last,U32.to_nat(size),point_below,
    Order.last_below(U32.to_nat(size),Order.positive_bound(U32.to_nat(first),U32.to_nat(size),first_inside)))

def position_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +first_inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(position),U32.to_nat(axis_count(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.to_nat((first + position * stride : U32)) == Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))) : Nat}:
  %Arithmetic.add_commutative((position * stride : U32),first) : {U32.to_nat(_) == Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))) : Nat}
  %Order.commutative(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(first)) : {U32.to_nat((position * stride + first : U32)) == _ : Nat}
  Arithmetic.multiply_add_exact(position,stride,first,capacity,capacity_ok,
    %Order.commutative(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(size),U32.to_nat(capacity),
      Bounds.strict_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(size),position_inside(first,size,kernel,stride,padding,U32.to_nat(position),capacity,capacity_ok,size_room,stride_room,positive,first_inside,inside)),size_room))

def residue(+first: U32,+stride: U32,+padding: U32) -> U32:
  ((first + padding) % stride : U32)

def residue_exact(+first: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(residue(first,stride,padding)) == Nat.mod(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(stride)) : Nat}:
  %Arithmetic.add(first,padding,capacity,capacity_ok,padded_room) :
    {U32.to_nat(residue(first,stride,padding)) == Nat.mod(_,U32.to_nat(stride)) : Nat}
  NativeDivision.modulo((first + padding : U32),stride,capacity,capacity_ok,stride_room,positive)

def tap_count(+kernel: U32,+residue: U32,+stride: U32) -> Nat:
  Bool.pick(Nat,Nat.is_lt(U32.to_nat(residue),U32.to_nat(kernel)),1n+Nat.div(tail(kernel,residue),U32.to_nat(stride)),0n)

def chosen_count(choice: Bool,+kernel: U32,+residue: U32,+stride: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  compared: {U32.is_lt(residue,kernel) == choice : Bool}) ->
  {U32.to_nat(Bool.pick(U32,choice,((kernel - 1 - residue) / stride + 1 : U32),0)) == Bool.pick(Nat,choice,1n+Nat.div(tail(kernel,residue),U32.to_nat(stride)),0n) : Nat}:
  match choice:
    case False{}: {==}
    case True{}: stepped_count(kernel,residue,stride,capacity,capacity_ok,kernel_room,stride_room,positive,Bounds.as_nat_lt(residue,kernel,compared))

def taps_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding))) == tap_count(kernel,residue(first,stride,padding),stride) : Nat}:
  %Comparison.lt(residue(first,stride,padding),kernel) :
    {U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding))) == Bool.pick(Nat,_,1n+Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),0n) : Nat}
  chosen_count(U32.is_lt(residue(first,stride,padding),kernel),kernel,residue(first,stride,padding),stride,capacity,capacity_ok,kernel_room,stride_room,positive,{==})

def last_tap(+kernel: U32,+residue: U32,+stride: U32) -> Nat:
  Nat.add(U32.to_nat(residue),Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)))

def last_tap_room(+kernel: U32,+residue: U32,+stride: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(residue),U32.to_nat(kernel)) == True{} : Bool}) ->
  {Nat.is_le(last_tap(kernel,residue,stride),Nat.sub(U32.to_nat(kernel),1n)) == True{} : Bool}:
  +last = Nat.sub(U32.to_nat(kernel),1n)
  %Order.restore_shift(U32.to_nat(residue),last,Order.before_last(U32.to_nat(residue),U32.to_nat(kernel),inside)) :
    {Nat.is_le(last_tap(kernel,residue,stride),_) == True{} : Bool}
  %Order.commutative(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),U32.to_nat(residue)) :
    {Nat.is_le(_,Nat.add(U32.to_nat(residue),tail(kernel,residue))) == True{} : Bool}
  %Order.commutative(tail(kernel,residue),U32.to_nat(residue)) :
    {Nat.is_le(Nat.add(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),U32.to_nat(residue)),_) == True{} : Bool}
  Order.add_le(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),tail(kernel,residue),U32.to_nat(residue),
    DivisionOrder.product_below(tail(kernel,residue),U32.to_nat(stride),positive))

def last_tap_exact(+kernel: U32,+residue: U32,+stride: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(residue),U32.to_nat(kernel)) == True{} : Bool}) ->
  {U32.to_nat((residue + (((kernel - 1 - residue) / stride + 1 - 1) * stride) : U32)) == last_tap(kernel,residue,stride) : Nat}:
  +division = Equal.trans(Nat,U32.to_nat(((kernel - 1 - residue) / stride : U32)),Nat.div(U32.to_nat((kernel - 1 - residue : U32)),U32.to_nat(stride)),
    Nat.div(tail(kernel,residue),U32.to_nat(stride)),NativeDivision.divide((kernel - 1 - residue : U32),stride,capacity,capacity_ok,stride_room,positive),
    Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(stride)),U32.to_nat((kernel - 1 - residue : U32)),tail(kernel,residue),tail_exact(kernel,residue,capacity,capacity_ok,kernel_room,inside)))
  +last_room = Order.transitive(Nat.sub(U32.to_nat(kernel),1n),U32.to_nat(kernel),U32.to_nat(capacity),Order.sub_le(U32.to_nat(kernel),1n),kernel_room)
  +room = Order.transitive(last_tap(kernel,residue,stride),Nat.sub(U32.to_nat(kernel),1n),U32.to_nat(capacity),last_tap_room(kernel,residue,stride,positive,inside),last_room)
  %Equal.sym(U32,(((kernel - 1 - residue) / stride + 1 - 1) : U32),((kernel - 1 - residue) / stride : U32),Subtraction.subtract_add(((kernel - 1 - residue) / stride : U32),1)) :
    {U32.to_nat((residue + (_ * stride) : U32)) == last_tap(kernel,residue,stride) : Nat}
  %Equal.sym(U32,(residue + (((kernel - 1 - residue) / stride) * stride) : U32),((((kernel - 1 - residue) / stride) * stride + residue) : U32),Arithmetic.add_commutative(residue,(((kernel - 1 - residue) / stride) * stride : U32))) :
    {U32.to_nat(_) == last_tap(kernel,residue,stride) : Nat}
  %Order.commutative(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),U32.to_nat(residue)) :
    {U32.to_nat((((kernel - 1 - residue) / stride) * stride + residue : U32)) == _ : Nat}
  %division :
    {U32.to_nat((((kernel - 1 - residue) / stride) * stride + residue : U32)) == Nat.add(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(residue)) : Nat}
  Arithmetic.multiply_add_exact(((kernel - 1 - residue) / stride : U32),stride,residue,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(((kernel - 1 - residue) / stride : U32)),Nat.div(tail(kernel,residue),U32.to_nat(stride)),division) :
      {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(residue)),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),U32.to_nat(residue)),last_tap(kernel,residue,stride),Order.commutative(Nat.mul(Nat.div(tail(kernel,residue),U32.to_nat(stride)),U32.to_nat(stride)),U32.to_nat(residue))) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)

def axis_tap_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {U32.to_nat(axis_tap(Kernel.axis(first,size,kernel,stride,padding))) == last_tap(kernel,residue(first,stride,padding),stride) : Nat}:
  %Equal.sym(Bool,U32.is_lt(residue(first,stride,padding),kernel),True{},inside) :
    {U32.to_nat((residue(first,stride,padding) + (Bool.pick(U32,_,((kernel - 1 - residue(first,stride,padding)) / stride + 1 : U32),0) - 1) * stride : U32)) == last_tap(kernel,residue(first,stride,padding),stride) : Nat}
  last_tap_exact(kernel,residue(first,stride,padding),stride,capacity,capacity_ok,kernel_room,stride_room,positive,Bounds.as_nat_lt(residue(first,stride,padding),kernel,inside))

def source_offset(+first_quotient: U32,+last_quotient: U32,+position: U32,+offset: U32) ->
  {((first_quotient - last_quotient : U32) + position + offset : U32) == ((first_quotient + position : U32) - (last_quotient - offset : U32) : U32) : U32}:
  %Equal.sym(U32,((first_quotient - last_quotient : U32) + position : U32),((first_quotient + position : U32) - last_quotient : U32),Arithmetic.add_after_subtract(first_quotient,last_quotient,position)) :
    {(_ + offset : U32) == ((first_quotient + position : U32) - (last_quotient - offset : U32) : U32) : U32}
  Arithmetic.subtract_difference((first_quotient + position : U32),last_quotient,offset)

def descending_tap(+value: U32,+residue: U32,+stride: U32,+quotient: Nat,+offset: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(value) == Nat.add(U32.to_nat(residue),Nat.mul(quotient,U32.to_nat(stride))) : Nat},
  +room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool},
  +inside: {Nat.is_le(offset,quotient) == True{} : Bool}) ->
  {U32.to_nat((value - U32.from_nat(Nat.mul(offset,U32.to_nat(stride))) : U32)) == Nat.add(U32.to_nat(residue),Nat.mul(Nat.sub(quotient,offset),U32.to_nat(stride))) : Nat}:
  +step = Nat.mul(offset,U32.to_nat(stride))
  +product = Nat.mul(quotient,U32.to_nat(stride))
  +product_order = RowMajor.mul_le(offset,quotient,U32.to_nat(stride),inside)
  +step_order = Order.transitive(step,product,U32.to_nat(value),product_order,
    %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(residue),product),exact) : {Nat.is_le(product,_) == True{} : Bool}
    %Order.commutative(product,U32.to_nat(residue)) : {Nat.is_le(product,_) == True{} : Bool} Order.le_add(product,U32.to_nat(residue)))
  +step_room = Order.transitive(step,U32.to_nat(value),U32.to_nat(capacity),step_order,room)
  +step_exact = Bounds.roundtrip(step,capacity,capacity_ok,step_room)
  +difference_exact = Arithmetic.subtract(value,U32.from_nat(step),capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(step)),step,step_exact) : {Nat.is_le(_,U32.to_nat(value)) == True{} : Bool} step_order,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(step)),step,step_exact) : {Nat.is_le(Nat.sub(U32.to_nat(value),_),U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.sub(U32.to_nat(value),step),U32.to_nat(value),U32.to_nat(capacity),Order.sub_le(U32.to_nat(value),step),room))
  %Equal.sym(Nat,U32.to_nat((value - U32.from_nat(step) : U32)),Nat.sub(U32.to_nat(value),U32.to_nat(U32.from_nat(step))),difference_exact) :
    {_ == Nat.add(U32.to_nat(residue),Nat.mul(Nat.sub(quotient,offset),U32.to_nat(stride))) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(step)),step,step_exact) :
    {Nat.sub(U32.to_nat(value),_) == Nat.add(U32.to_nat(residue),Nat.mul(Nat.sub(quotient,offset),U32.to_nat(stride))) : Nat}
  %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(residue),product),exact) :
    {Nat.sub(_,step) == Nat.add(U32.to_nat(residue),Nat.mul(Nat.sub(quotient,offset),U32.to_nat(stride))) : Nat}
  %RowMajor.subtract_products(quotient,offset,U32.to_nat(stride)) :
    {Nat.sub(Nat.add(U32.to_nat(residue),product),step) == Nat.add(U32.to_nat(residue),_) : Nat}
  Order.subtract_from_sum(U32.to_nat(residue),product,step,product_order)

def active_taps_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding))) == 1n+Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)) : Nat}:
  %Equal.sym(Bool,U32.is_lt(residue(first,stride,padding),kernel),True{},inside) :
    {U32.to_nat(Bool.pick(U32,_,((kernel - 1 - residue(first,stride,padding)) / stride + 1 : U32),0)) == 1n+Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)) : Nat}
  stepped_count(kernel,residue(first,stride,padding),stride,capacity,capacity_ok,kernel_room,stride_room,positive,Bounds.as_nat_lt(residue(first,stride,padding),kernel,inside))

def axis_offset_exact(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool},
  +inside: {Nat.is_lt(offset,U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.to_nat((axis_tap(Kernel.axis(first,size,kernel,stride,padding)) - U32.from_nat(Nat.mul(offset,U32.to_nat(stride))) : U32)) ==
    Nat.add(U32.to_nat(residue(first,stride,padding)),Nat.mul(Nat.sub(Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),offset),U32.to_nat(stride))) : Nat}:
  +tap_exact = axis_tap_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)
  +tap_room = {
    %Equal.sym(Nat,U32.to_nat(axis_tap(Kernel.axis(first,size,kernel,stride,padding))),last_tap(kernel,residue(first,stride,padding),stride),tap_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(last_tap(kernel,residue(first,stride,padding),stride),Nat.sub(U32.to_nat(kernel),1n),U32.to_nat(capacity),last_tap_room(kernel,residue(first,stride,padding),stride,positive,Bounds.as_nat_lt(residue(first,stride,padding),kernel,active)),Order.transitive(Nat.sub(U32.to_nat(kernel),1n),U32.to_nat(kernel),U32.to_nat(capacity),Order.sub_le(U32.to_nat(kernel),1n),kernel_room)) :
    {Nat.is_le(U32.to_nat(axis_tap(Kernel.axis(first,size,kernel,stride,padding))),U32.to_nat(capacity)) == True{} : Bool}}
  descending_tap(axis_tap(Kernel.axis(first,size,kernel,stride,padding)),residue(first,stride,padding),stride,Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),offset,capacity,capacity_ok,tap_exact,tap_room,
    Division.lt_succ_le(offset,Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),
      %active_taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active) : {Nat.is_lt(offset,_) == True{} : Bool} inside))

def shifted_quotient(+base: U32,+stride: U32,+shift: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(shift),U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(((base + shift * stride) / stride : U32)) == Nat.add(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride))) : Nat}:
  +address = (base + shift * stride : U32)
  +address_exact = Equal.trans(Nat,U32.to_nat(address),U32.to_nat((shift * stride + base : U32)),Nat.add(Nat.mul(U32.to_nat(shift),U32.to_nat(stride)),U32.to_nat(base)),
    Equal.cong(U32,Nat,U32.to_nat,address,(shift * stride + base : U32),Arithmetic.add_commutative(base,(shift * stride : U32))),
    Arithmetic.multiply_add_exact(shift,stride,base,capacity,capacity_ok,
      %Order.commutative(U32.to_nat(base),Nat.mul(U32.to_nat(shift),U32.to_nat(stride))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room))
  %Equal.sym(Nat,U32.to_nat((address / stride : U32)),Nat.div(U32.to_nat(address),U32.to_nat(stride)),NativeDivision.divide(address,stride,capacity,capacity_ok,stride_room,positive)) :
    {_ == Nat.add(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride))) : Nat}
  %Equal.sym(Nat,U32.to_nat(address),Nat.add(Nat.mul(U32.to_nat(shift),U32.to_nat(stride)),U32.to_nat(base)),address_exact) :
    {Nat.div(_,U32.to_nat(stride)) == Nat.add(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride))) : Nat}
  DivisionOrder.quotient_shift(U32.to_nat(base),U32.to_nat(stride),U32.to_nat(shift),positive)

def shifted_quotient_word(+base: U32,+stride: U32,+shift: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(base),Nat.mul(U32.to_nat(shift),U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {((base + shift * stride) / stride : U32) == (base / stride + shift : U32) : U32}:
  %Equal.sym(U32,(base / stride + shift : U32),U32.from_nat(Nat.add(U32.to_nat((base / stride : U32)),U32.to_nat(shift))),Arithmetic.native_add((base / stride : U32),shift)) :
    {((base + shift * stride) / stride : U32) == _ : U32}
  %Equal.sym(Nat,U32.to_nat((base / stride : U32)),Nat.div(U32.to_nat(base),U32.to_nat(stride)),NativeDivision.divide(base,stride,capacity,capacity_ok,stride_room,positive)) :
    {((base + shift * stride) / stride : U32) == U32.from_nat(Nat.add(_,U32.to_nat(shift))) : U32}
  %Order.commutative(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride))) :
    {((base + shift * stride) / stride : U32) == U32.from_nat(_) : U32}
  Equal.sym(U32,U32.from_nat(Nat.add(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride)))),((base + shift * stride) / stride : U32),
    Arithmetic.from_nat_exact(((base + shift * stride) / stride : U32),Nat.add(U32.to_nat(shift),Nat.div(U32.to_nat(base),U32.to_nat(stride))),shifted_quotient(base,stride,shift,capacity,capacity_ok,stride_room,positive,room)))

def residue_inside(+first: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(residue(first,stride,padding)),U32.to_nat(stride)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(residue(first,stride,padding)),Nat.mod(U32.to_nat((first + padding : U32)),U32.to_nat(stride)),NativeDivision.modulo((first + padding : U32),stride,capacity,capacity_ok,stride_room,positive)) :
    {Nat.is_lt(_,U32.to_nat(stride)) == True{} : Bool}
  DivisionOrder.remainder_small(U32.to_nat((first + padding : U32)),U32.to_nat(stride),positive)

def axis_offset_quotient(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool},
  +inside: {Nat.is_lt(offset,U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.to_nat(((axis_tap(Kernel.axis(first,size,kernel,stride,padding)) - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32)) ==
    Nat.sub(Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),offset) : Nat}:
  +tap = (axis_tap(Kernel.axis(first,size,kernel,stride,padding)) - U32.from_nat(Nat.mul(offset,U32.to_nat(stride))) : U32)
  +q = Nat.sub(Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)),offset)
  +r = U32.to_nat(residue(first,stride,padding))
  %Equal.sym(Nat,U32.to_nat((tap / stride : U32)),Nat.div(U32.to_nat(tap),U32.to_nat(stride)),NativeDivision.divide(tap,stride,capacity,capacity_ok,stride_room,positive)) : {_ == q : Nat}
  %Equal.sym(Nat,U32.to_nat(tap),Nat.add(r,Nat.mul(q,U32.to_nat(stride))),axis_offset_exact(first,size,kernel,stride,padding,offset,capacity,capacity_ok,kernel_room,stride_room,positive,active,inside)) :
    {Nat.div(_,U32.to_nat(stride)) == q : Nat}
  %Order.commutative(Nat.mul(q,U32.to_nat(stride)),r) : {Nat.div(_,U32.to_nat(stride)) == q : Nat}
  Radix.quotient(q,r,U32.to_nat(stride),positive,residue_inside(first,stride,padding,capacity,capacity_ok,stride_room,positive))

def axis_pixel_radix(+first: U32,+stride: U32,+padding: U32,+position: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat((first + padding : U32)),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(padding)) ==
    Nat.add(Nat.mul(U32.to_nat(((first + padding) / stride + position : U32)),U32.to_nat(stride)),U32.to_nat(residue(first,stride,padding))) : Nat}:
  +base = (first + padding : U32)
  +q = Nat.div(U32.to_nat(base),U32.to_nat(stride))
  +r = Nat.mod(U32.to_nat(base),U32.to_nat(stride))
  +native_q = Equal.trans(Nat,U32.to_nat((base / stride + position : U32)),U32.to_nat(((base + position * stride) / stride : U32)),Nat.add(U32.to_nat(position),q),
    Equal.cong(U32,Nat,U32.to_nat,(base / stride + position : U32),((base + position * stride) / stride : U32),Equal.sym(U32,((base + position * stride) / stride : U32),(base / stride + position : U32),shifted_quotient_word(base,stride,position,capacity,capacity_ok,stride_room,positive,shift_room))),
    shifted_quotient(base,stride,position,capacity,capacity_ok,stride_room,positive,shift_room))
  %Equal.sym(Nat,U32.to_nat((base / stride + position : U32)),Nat.add(U32.to_nat(position),q),native_q) :
    {Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(padding)) == Nat.add(Nat.mul(_,U32.to_nat(stride)),U32.to_nat(residue(first,stride,padding))) : Nat}
  %Equal.sym(Nat,U32.to_nat(residue(first,stride,padding)),r,NativeDivision.modulo(base,stride,capacity,capacity_ok,stride_room,positive)) :
    {Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(padding)) == Nat.add(Nat.mul(Nat.add(U32.to_nat(position),q),U32.to_nat(stride)),_) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(padding)),Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),Nat.add(U32.to_nat(first),U32.to_nat(padding))),
    Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(padding)),Nat.add(Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(first)),U32.to_nat(padding)),Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),Nat.add(U32.to_nat(first),U32.to_nat(padding))),
      Equal.cong(Nat,Nat,n => Nat.add(n,U32.to_nat(padding)),Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(first)),Order.commutative(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))),
      Addition.associative(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(first),U32.to_nat(padding)))) :
    {_ == Nat.add(Nat.mul(Nat.add(U32.to_nat(position),q),U32.to_nat(stride)),r) : Nat}
  %Arithmetic.add(first,padding,capacity,capacity_ok,padded_room) :
    {Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),_) == Nat.add(Nat.mul(Nat.add(U32.to_nat(position),q),U32.to_nat(stride)),r) : Nat}
  DivisionOrder.shifted_exact(U32.to_nat(base),U32.to_nat(stride),U32.to_nat(position),positive)

def axis_tap_quotient(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {U32.to_nat((axis_tap(Kernel.axis(first,size,kernel,stride,padding)) / stride : U32)) == Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride)) : Nat}:
  +tap = axis_tap(Kernel.axis(first,size,kernel,stride,padding))
  +q = Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride))
  +r = U32.to_nat(residue(first,stride,padding))
  %Equal.sym(Nat,U32.to_nat((tap / stride : U32)),Nat.div(U32.to_nat(tap),U32.to_nat(stride)),NativeDivision.divide(tap,stride,capacity,capacity_ok,stride_room,positive)) : {_ == q : Nat}
  %Equal.sym(Nat,U32.to_nat(tap),Nat.add(r,Nat.mul(q,U32.to_nat(stride))),axis_tap_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)) :
    {Nat.div(_,U32.to_nat(stride)) == q : Nat}
  %Order.commutative(Nat.mul(q,U32.to_nat(stride)),r) : {Nat.div(_,U32.to_nat(stride)) == q : Nat}
  Radix.quotient(q,r,U32.to_nat(stride),positive,residue_inside(first,stride,padding,capacity,capacity_ok,stride_room,positive))

def axis_source_counter(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: U32,+offset: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(residue(first,stride,padding),kernel) == True{} : Bool},
  +inside: {Nat.is_lt(offset,U32.to_nat(axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {((axis_source(Kernel.axis(first,size,kernel,stride,padding)) + position : U32) + U32.from_nat(offset) : U32) ==
    (((first + padding) / stride + position : U32) - ((axis_tap(Kernel.axis(first,size,kernel,stride,padding)) - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32) : U32) : U32}:
  +tap = axis_tap(Kernel.axis(first,size,kernel,stride,padding))
  +q = Nat.div(tail(kernel,residue(first,stride,padding)),U32.to_nat(stride))
  +before = Division.lt_succ_le(offset,q,
    %active_taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active) : {Nat.is_lt(offset,_) == True{} : Bool} inside)
  +tap_q = axis_tap_quotient(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)
  +next_q = axis_offset_quotient(first,size,kernel,stride,padding,offset,capacity,capacity_ok,kernel_room,stride_room,positive,active,inside)
  +word_q = Equal.trans(U32,((tap - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32),U32.from_nat(Nat.sub(q,offset)),((tap / stride : U32) - U32.from_nat(offset) : U32),
    Equal.sym(U32,U32.from_nat(Nat.sub(q,offset)),((tap - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32),Arithmetic.from_nat_exact(((tap - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32),Nat.sub(q,offset),next_q)),
    %Arithmetic.from_nat_exact((tap / stride : U32),q,tap_q) : {U32.from_nat(Nat.sub(q,offset)) == (_ - U32.from_nat(offset) : U32) : U32}
    Equal.sym(U32,(U32.from_nat(q) - U32.from_nat(offset) : U32),U32.from_nat(Nat.sub(q,offset)),Subtraction.encoded_subtract(q,offset,before)))
  %Equal.sym(U32,((tap - U32.from_nat(Nat.mul(offset,U32.to_nat(stride)))) / stride : U32),((tap / stride : U32) - U32.from_nat(offset) : U32),word_q) :
    {((axis_source(Kernel.axis(first,size,kernel,stride,padding)) + position : U32) + U32.from_nat(offset) : U32) == (((first + padding) / stride + position : U32) - _ : U32) : U32}
  source_offset(((first + padding) / stride : U32),(tap / stride : U32),position,U32.from_nat(offset))
