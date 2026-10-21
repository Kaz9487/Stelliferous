# The padded group layout preserves the order and bounds of row-major jobs.
import Base
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as NativeDivision

def group_position(+index: Nat,+span: Nat,+groups: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(index,span),groups),Nat.mod(index,span))

def group_expanded(+index: Nat,+span: Nat,+groups: Nat,
  positive: {Nat.is_lt(0n,span) == True{} : Bool},ordered: {Nat.is_le(span,groups) == True{} : Bool}) ->
  {group_position(index,span,groups) == Nat.add(index,Nat.mul(Nat.div(index,span),Nat.sub(groups,span))) : Nat}:
  +row = Nat.div(index,span)
  +column = Nat.mod(index,span)
  +first = Nat.mul(row,span)
  +gap = Nat.mul(row,Nat.sub(groups,span))
  %Order.restore_shift(span,groups,ordered) : {Nat.add(Nat.mul(row,_),column) == Nat.add(index,gap) : Nat}
  %Equal.sym(Nat,Nat.mul(row,Nat.add(span,Nat.sub(groups,span))),Nat.add(first,gap),RowMajor.right_distrib(row,span,Nat.sub(groups,span))) :
    {Nat.add(_,column) == Nat.add(index,gap) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(first,gap),column),Nat.add(first,Nat.add(gap,column)),Addition.associative(first,gap,column)) :
    {_ == Nat.add(index,gap) : Nat}
  %Order.commutative(column,gap) : {Nat.add(first,_) == Nat.add(index,gap) : Nat}
  %Addition.associative(first,column,gap) : {_ == Nat.add(index,gap) : Nat}
  %Division.quotient_exact(index,span,positive) : {Nat.add(_,gap) == Nat.add(index,gap) : Nat}
  {==}

def group_increasing(+first: Nat,+second: Nat,+span: Nat,+groups: Nat,
  +positive: {Nat.is_lt(0n,span) == True{} : Bool},+ordered: {Nat.is_le(span,groups) == True{} : Bool},
  +before: {Nat.is_lt(first,second) == True{} : Bool}) ->
  {Nat.is_lt(group_position(first,span,groups),group_position(second,span,groups)) == True{} : Bool}:
  %Equal.sym(Nat,group_position(first,span,groups),Nat.add(first,Nat.mul(Nat.div(first,span),Nat.sub(groups,span))),group_expanded(first,span,groups,positive,ordered)) : {Nat.is_lt(_,group_position(second,span,groups)) == True{} : Bool}
  %Equal.sym(Nat,group_position(second,span,groups),Nat.add(second,Nat.mul(Nat.div(second,span),Nat.sub(groups,span))),group_expanded(second,span,groups,positive,ordered)) :
    {Nat.is_lt(Nat.add(first,Nat.mul(Nat.div(first,span),Nat.sub(groups,span))),_) == True{} : Bool}
  Bounds.nat_trans(Nat.add(first,Nat.mul(Nat.div(first,span),Nat.sub(groups,span))),Nat.add(second,Nat.mul(Nat.div(first,span),Nat.sub(groups,span))),Nat.add(second,Nat.mul(Nat.div(second,span),Nat.sub(groups,span))),
    Order.add_lt(first,second,Nat.mul(Nat.div(first,span),Nat.sub(groups,span)),before),
    Order.offset_le(second,Nat.mul(Nat.div(first,span),Nat.sub(groups,span)),Nat.mul(Nat.div(second,span),Nat.sub(groups,span)),
      RowMajor.mul_le(Nat.div(first,span),Nat.div(second,span),Nat.sub(groups,span),Division.quotient_le(first,second,span,positive,Bounds.strict_le(first,second,before)))))

def group_inside(+index: Nat,+rows: Nat,+span: Nat,+groups: Nat,
  +positive: {Nat.is_lt(0n,span) == True{} : Bool},ordered: {Nat.is_le(span,groups) == True{} : Bool},
  inside: {Nat.is_lt(index,Nat.mul(rows,span)) == True{} : Bool}) ->
  {Nat.is_lt(group_position(index,span,groups),Nat.mul(rows,groups)) == True{} : Bool}:
  RowMajor.radix(Nat.div(index,span),rows,Nat.mod(index,span),groups,
    Division.quotient_below(index,rows,span,positive,inside),
    Bounds.nat_trans(Nat.mod(index,span),span,groups,Division.remainder_small(index,span,positive),ordered))

# The native group counter has a bounded row and column at the padded stride.
def group_address_exact(+index: Nat,+span: U32,+groups: U32,+block: U32,+rows: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(span),U32.to_nat(groups)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +double_span: {Nat.is_le(Nat.double(U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.mul(rows,U32.to_nat(groups)),U32.to_nat(block)),U32.to_nat(capacity)) == True{} : Bool},
  +inside: {Nat.is_lt(index,Nat.mul(rows,U32.to_nat(span))) == True{} : Bool}) ->
  {U32.to_nat(((U32.from_nat(index) / span * groups + U32.from_nat(index) % span) * block : U32)) ==
    Nat.mul(group_position(index,U32.to_nat(span),U32.to_nat(groups)),U32.to_nat(block)) : Nat}:
  +width = U32.to_nat(span)
  +group_width = U32.to_nat(groups)
  +block_width = U32.to_nat(block)
  +group_count = Nat.mul(rows,group_width)
  +group_count_room = Order.transitive(group_count,Nat.mul(group_count,block_width),U32.to_nat(capacity),RowMajor.product_covers(group_count,block_width,block_positive),room)
  +count_room = Order.transitive(Nat.mul(rows,width),group_count,U32.to_nat(capacity),Division.multiply_left_le(rows,width,group_width,ordered),group_count_room)
  +index_exact = Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,Nat.mul(rows,width),U32.to_nat(capacity),inside,count_room)))
  +row = Nat.div(index,width)
  +column = Nat.mod(index,width)
  +row_word = (U32.from_nat(index) / span : U32)
  +column_word = (U32.from_nat(index) % span : U32)
  +row_exact = {
    %index_exact : {U32.to_nat(row_word) == Nat.div(_,width) : Nat}
    NativeDivision.divide(U32.from_nat(index),span,capacity,capacity_ok,double_span,span_positive) : {U32.to_nat(row_word) == row : Nat}}
  +column_exact = {
    %index_exact : {U32.to_nat(column_word) == Nat.mod(_,width) : Nat}
    NativeDivision.modulo(U32.from_nat(index),span,capacity,capacity_ok,double_span,span_positive) : {U32.to_nat(column_word) == column : Nat}}
  +position = group_position(index,width,group_width)
  +position_inside = group_inside(index,rows,width,group_width,span_positive,ordered,inside)
  +position_room = Bounds.strict_le(position,U32.to_nat(capacity),Bounds.nat_trans(position,group_count,U32.to_nat(capacity),position_inside,group_count_room))
  +position_word = (row_word * groups + column_word : U32)
  +position_exact = {
    %row_exact : {U32.to_nat(position_word) == Nat.add(Nat.mul(_,group_width),column) : Nat}
    %column_exact : {U32.to_nat(position_word) == Nat.add(Nat.mul(U32.to_nat(row_word),group_width),_) : Nat}
    Arithmetic.multiply_add_exact(row_word,groups,column_word,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(row_word),row,row_exact) : {Nat.is_le(Nat.add(Nat.mul(_,group_width),U32.to_nat(column_word)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(column_word),column,column_exact) : {Nat.is_le(Nat.add(Nat.mul(row,group_width),_),U32.to_nat(capacity)) == True{} : Bool}
      position_room) : {U32.to_nat(position_word) == position : Nat}}
  %position_exact : {U32.to_nat((position_word * block : U32)) == Nat.mul(_,block_width) : Nat}
  Arithmetic.multiply(position_word,block,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(position_word),position,position_exact) : {Nat.is_le(Nat.mul(_,block_width),U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mul(position,block_width),Nat.mul(group_count,block_width),U32.to_nat(capacity),RowMajor.mul_le(position,group_count,block_width,Bounds.strict_le(position,group_count,position_inside)),room))
