# Every vector of count cells has its descriptor: its count, contiguity and
# the address of its last cell inside it.
import Base
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ./tensor_vector_view_refinement.bend as Vector
import ./boolean_logic.bend as Boolean
import ./checked_word_unit_products.bend as UnitProducts
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./u32_full_range.bend as FullRange
import ./nat_order.bend as Order

def product_single(+start: U32,+count: U32,+output: U32,accepted: {Shapes.checked_multiply(start,count) == Some{output} : Maybe<&2,U32>}) ->
  {Shapes.product([count],Some{start}) == Some{output} : Maybe<&2,U32>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(start,count),Some{output},accepted) : {Shapes.product([],_) == Some{output} : Maybe<&2,U32>}
  {==}

def counted_when(+count: U32,empty: Bool,classified: {U32.is_zero(count) == empty : Bool},
  one_left: {Shapes.checked_multiply(1,count) == Some{count} : Maybe<&2,U32>}) ->
  {Shapes.count_nonempty(empty,[count]) == Some{count} : Maybe<&2,U32>}:
  match empty:
    case True{}:
      %Equal.sym(U32,count,0,Arithmetic.equal_of_is_eq(count,0,classified)) : {Some{0} == Some{_} : Maybe<&2,U32>}
      {==}
    case False{}: product_single(1,count,count,one_left)

def counted(+count: U32,one_left: {Shapes.checked_multiply(1,count) == Some{count} : Maybe<&2,U32>}) -> {Views.numel(Vector.vector(count)) == Some{count} : Maybe<&2,U32>}:
  %Equal.sym(Bool,Shapes.has_zero([count]),U32.is_zero(count),Vector.zero_flag(count)) :
    {Shapes.count_nonempty(_,[count]) == Some{count} : Maybe<&2,U32>}
  counted_when(count,U32.is_zero(count),{==},one_left)

def contiguous_when(+count: U32,empty: Bool,one_right: {Shapes.checked_multiply(count,1) == Some{count} : Maybe<&2,U32>}) ->
  {Views.contiguous_nonempty(empty,[Views.Axis{count,1}]) == True{} : Bool}:
  match empty:
    case True{}: {==}
    case False{}:
      %Equal.sym(Bool,U32.is_le(count,1) || True{},True{},Boolean.or_true_right(U32.is_le(count,1))) :
        {Views.has_count(Views.contiguous_product(_,count,1)) == True{} : Bool}
      %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(count,1),Some{count},one_right) :
        {Views.has_count(_) == True{} : Bool}
      {==}

def contiguous(+count: U32,one_right: {Shapes.checked_multiply(count,1) == Some{count} : Maybe<&2,U32>}) -> {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}:
  contiguous_when(count,Shapes.has_zero([count]),one_right)

def native_positive(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool}) -> {U32.is_lt(0,count) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(0,count),Nat.is_lt(0n,U32.to_nat(count)),Comparison.lt(0,count)) : {_ == True{} : Bool}
  Arithmetic.word_nonzero_positive(count,nonzero)

def predecessor_inside(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool}) -> {U32.is_lt((count - 1 : U32),count) == True{} : Bool}:
  +lower = Arithmetic.word_nonzero_positive(count,nonzero)
  +ordered = Order.strict_follow(0n,U32.to_nat(count),lower)
  %Equal.sym(Bool,U32.is_lt((count - 1 : U32),count),Nat.is_lt(U32.to_nat((count - 1 : U32)),U32.to_nat(count)),Comparison.lt((count - 1 : U32),count)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((count - 1 : U32)),Nat.sub(U32.to_nat(count),1n),
    FullRange.subtract(count,1,count,ordered,Order.sub_le(U32.to_nat(count),1n))) :
    {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
  Order.sub_lt(U32.to_nat(count),1n,U32.to_nat(count),ordered,Order.lt_succ(U32.to_nat(count),U32.to_nat(count),Order.reflexive(U32.to_nat(count))))

def native_reflexive(+word: U32) -> {U32.is_le(word,word) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(word,word),Nat.is_le(U32.to_nat(word),U32.to_nat(word)),Comparison.le(word,word)) : {_ == True{} : Bool}
  Order.reflexive(U32.to_nat(word))

def advance_last(+count: U32,nonzero: {U32.is_zero(count) == False{} : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.advance(0,(count - 1 : U32),1,count) == Some{(count - 1 : U32)} : Maybe<&2,U32>}:
  +last = (count - 1 : U32)
  %Equal.sym(Bool,U32.is_lt(0,count),True{},native_positive(count,nonzero)) :
    {Views.advance_inside(_,0,last,1,count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(last,1),Some{last},one_last) :
    {Views.advance_product(_,0,(last - 0 : U32)) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(U32,(last - 0 : U32),last,Arithmetic.subtract_zero(last)) :
    {Views.advance_product(Some{last},0,_) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_le(last,last),True{},native_reflexive(last)) :
    {Views.address_when(_,0,last) == Some{last} : Maybe<&2,U32>}
  Equal.cong(U32,Maybe<&2,U32>,word => Some{word},(0 + last : U32),last,Arithmetic.add_zero_left(last))

def last_address(+count: U32,+nonzero: {U32.is_zero(count) == False{} : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.address(Vector.vector(count),[(count - 1 : U32)],count) == Some{(count - 1 : U32)} : Maybe<&2,U32>}:
  +last = (count - 1 : U32)
  %Equal.sym(Bool,U32.is_lt(last,count),True{},predecessor_inside(count,nonzero)) :
    {Views.locate([],[],Views.advance_inside(_ && U32.is_lt(0,count),0,last,1,count),count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Maybe<&2,U32>,Views.advance(0,last,1,count),Some{last},advance_last(count,nonzero,one_last)) :
    {Views.locate([],[],_,count) == Some{last} : Maybe<&2,U32>}
  %Equal.sym(Bool,U32.is_lt(last,count),True{},predecessor_inside(count,nonzero)) :
    {Views.finish_address(_,last) == Some{last} : Maybe<&2,U32>}
  {==}

def fits_when(+count: U32,empty: Bool,classified: {U32.is_zero(count) == empty : Bool},
  one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) ->
  {Views.fits_nonempty(empty,Vector.vector(count),count) == True{} : Bool}:
  match empty:
    case True{}:
      %Equal.sym(Bool,U32.is_le(0,count),Nat.is_le(0n,U32.to_nat(count)),Comparison.le(0,count)) : {_ == True{} : Bool}
      Order.zero_le(U32.to_nat(count))
    case False{}:
      Equal.cong(Maybe<&2,U32>,Bool,result => Views.has_count(result),Views.address(Vector.vector(count),[(count - 1 : U32)],count),Some{(count - 1 : U32)},last_address(count,classified,one_last))

def fits(+count: U32,one_last: {Shapes.checked_multiply((count - 1 : U32),1) == Some{(count - 1 : U32)} : Maybe<&2,U32>}) -> {Views.fits(Vector.vector(count),count) == True{} : Bool}:
  %Equal.sym(Bool,Shapes.has_zero([count]),U32.is_zero(count),Vector.zero_flag(count)) :
    {Views.fits_nonempty(_,Vector.vector(count),count) == True{} : Bool}
  fits_when(count,U32.is_zero(count),{==},one_last)

def descriptor(+count: U32) -> Vector.Descriptor<count>:
  Vector.Descriptor{counted(count,UnitProducts.one_left(count)),contiguous(count,UnitProducts.one_right(count)),fits(count,UnitProducts.one_right((count - 1 : U32)))}
