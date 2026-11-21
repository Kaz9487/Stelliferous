# The row planner: coverage of its interval, bounds of every leaf, and monotone
# endpoint maps between input and output rows.
import Base
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ../tensor_pieces.bend as Pieces
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./u32_division.bend as U32Division
import ../kernel_gemm.bend as KernelGemm
import ./word_multiplication.bend as WordMultiplication

# The real U32 row splitter covers its original interval exactly. Forks join
# consecutive intervals, so their leaves neither omit nor duplicate a row.
law Coverage:
  for plan: Pieces.Ranges
  for first: Nat
  for rows: Nat
  Data

type LeafCoverage<-native_first: U32,-native_rows: U32,-first: Nat,-rows: Nat> is Data:
  LeafCoverage{first_equal: {U32.to_nat(native_first) == first : Nat},rows_equal: {U32.to_nat(native_rows) == rows : Nat}}

type ForkCoverage<-left: Pieces.Ranges,-right: Pieces.Ranges,-first: Nat,-rows: Nat> is Data:
  ForkCoverage{left_rows: Nat,right_rows: Nat,rows_equal: {rows == Nat.add(left_rows,right_rows) : Nat},
    left_coverage: Coverage(left,first,left_rows),right_coverage: Coverage(right,Nat.add(first,left_rows),right_rows)}

def Coverage(plan,first,rows):
  match plan:
    case Pieces.Range{native_first,native_rows}: LeafCoverage<native_first,native_rows,first,rows>
    case Pieces.Split{left,right}: ForkCoverage<left,right,first,rows>

def rows_bound(+first: U32,+rows: U32,+capacity: U32,room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(rows),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(capacity),
    %Order.commutative(U32.to_nat(rows),U32.to_nat(first)) : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool}
    Order.le_add(U32.to_nat(rows),U32.to_nat(first)),room)

# The first half of a unit plan is a whole number of units no larger than the count.
def unit_half_bound(+unit: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +divisor_cap: U32,+divisor_cap_ok: {U32.is_le(divisor_cap,2147483648) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat((unit * 2 : U32))),U32.to_nat(divisor_cap)) == True{} : Bool},
  +doubled: {U32.to_nat((unit * 2 : U32)) == Nat.mul(U32.to_nat(unit),2n) : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat((unit * 2 : U32))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((count / (unit * 2) * unit : U32)),U32.to_nat(count)) == True{} : Bool}:
  +divisor = (unit * 2 : U32)
  +word_quotient = (count / divisor : U32)
  +quotient = Nat.div(U32.to_nat(count),U32.to_nat(divisor))
  +product = Nat.mul(quotient,U32.to_nat(unit))
  +quotient_exact = U32Division.divide(count,divisor,divisor_cap,divisor_cap_ok,divisor_bound,positive)
  +below = {
    %Equal.sym(Nat,U32.to_nat(count),Nat.add(Nat.mul(quotient,U32.to_nat(divisor)),Nat.mod(U32.to_nat(count),U32.to_nat(divisor))),
        Radix.division_identity(count,U32.to_nat(divisor),positive)) :
      {Nat.is_le(product,_) == True{} : Bool}
    Order.transitive(product,Nat.mul(quotient,U32.to_nat(divisor)),Nat.add(Nat.mul(quotient,U32.to_nat(divisor)),Nat.mod(U32.to_nat(count),U32.to_nat(divisor))),
      %Equal.sym(Nat,U32.to_nat(divisor),Nat.mul(U32.to_nat(unit),2n),doubled) : {Nat.is_le(product,Nat.mul(quotient,_)) == True{} : Bool}
      %RowMajor.assoc(quotient,U32.to_nat(unit),2n) : {Nat.is_le(product,_) == True{} : Bool}
      RowMajor.product_covers(product,2n,{==}),
      Order.le_add(Nat.mul(quotient,U32.to_nat(divisor)),Nat.mod(U32.to_nat(count),U32.to_nat(divisor)))) :
    {Nat.is_le(product,U32.to_nat(count)) == True{} : Bool}}
  +exact = {
    %quotient_exact : {U32.to_nat((word_quotient * unit : U32)) == Nat.mul(_,U32.to_nat(unit)) : Nat}
    WordMultiplication.bounded_multiply(word_quotient,unit,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(word_quotient),quotient,quotient_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(unit)),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(product,U32.to_nat(count),U32.to_nat(capacity),below,inside)) :
    {U32.to_nat((word_quotient * unit : U32)) == product : Nat}}
  %Equal.sym(Nat,U32.to_nat((word_quotient * unit : U32)),product,exact) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
  below

def refinement_unit_ranges(depth: Nat,+unit: U32,+first: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +divisor_cap: U32,+divisor_cap_ok: {U32.is_le(divisor_cap,2147483648) == True{} : Bool},
  +divisor_bound: {Nat.is_le(Nat.double(U32.to_nat((unit * 2 : U32))),U32.to_nat(divisor_cap)) == True{} : Bool},
  +doubled: {U32.to_nat((unit * 2 : U32)) == Nat.mul(U32.to_nat(unit),2n) : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat((unit * 2 : U32))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Coverage(Pieces.unit_ranges(depth,unit,first,count),U32.to_nat(first),U32.to_nat(count)):
  match depth:
    case 0n: LeafCoverage{{==},{==}}
    case 1n+ +rest:
      +half = (count / (unit * 2) * unit : U32)
      +remainder = (count - half : U32)
      +next = (first + half : U32)
      +half_inside = unit_half_bound(unit,count,capacity,capacity_ok,divisor_cap,divisor_cap_ok,divisor_bound,doubled,positive,rows_bound(first,count,capacity,room))
      +left_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(half)),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),
        Order.offset_le(U32.to_nat(first),U32.to_nat(half),U32.to_nat(count),half_inside),room)
      +next_exact = Arithmetic.add(first,half,capacity,capacity_ok,left_room)
      +remainder_exact = Arithmetic.subtract(count,half,capacity,capacity_ok,half_inside,
        Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(half)),U32.to_nat(count),U32.to_nat(capacity),Order.sub_le(U32.to_nat(count),U32.to_nat(half)),rows_bound(first,count,capacity,room)))
      +joined = {
        %Equal.sym(Nat,U32.to_nat(remainder),Nat.sub(U32.to_nat(count),U32.to_nat(half)),remainder_exact) : {Nat.add(U32.to_nat(half),_) == U32.to_nat(count) : Nat}
        Order.restore_shift(U32.to_nat(half),U32.to_nat(count),half_inside) : {Nat.add(U32.to_nat(half),U32.to_nat(remainder)) == U32.to_nat(count) : Nat}}
      +right_room = {
        %Equal.sym(Nat,U32.to_nat(next),Nat.add(U32.to_nat(first),U32.to_nat(half)),next_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(half)),U32.to_nat(remainder)),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(half),U32.to_nat(remainder))),
          Addition.associative(U32.to_nat(first),U32.to_nat(half),U32.to_nat(remainder))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(U32.to_nat(half),U32.to_nat(remainder)),U32.to_nat(count),joined) : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(capacity)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(U32.to_nat(next),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}}
      ForkCoverage{U32.to_nat(half),U32.to_nat(remainder),
        Equal.sym(Nat,Nat.add(U32.to_nat(half),U32.to_nat(remainder)),U32.to_nat(count),joined),
        refinement_unit_ranges(rest,unit,first,half,capacity,capacity_ok,divisor_cap,divisor_cap_ok,divisor_bound,doubled,positive,left_room),
        %next_exact : Coverage(Pieces.unit_ranges(rest,unit,next,remainder),_,U32.to_nat(remainder))
        refinement_unit_ranges(rest,unit,next,remainder,capacity,capacity_ok,divisor_cap,divisor_cap_ok,divisor_bound,doubled,positive,right_room)}

# Coverage of a subpartition also bounds every leaf. Applying this once per
# actual image plan proves that the U32 row splitter never crosses an image.
type LeafWithin<-first: U32,-rows: U32,-lower: Nat,-upper: Nat> is Data:
  LeafWithin{lower: {Nat.is_le(lower,U32.to_nat(first)) == True{} : Bool},
    upper: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),upper) == True{} : Bool}}

def Within(plan: Pieces.Ranges,+lower: Nat,+upper: Nat) -> Data:
  match plan:
    case Pieces.Range{first,rows}: LeafWithin<first,rows,lower,upper>
    case Pieces.Split{left,right}: KernelGemm.Both<Within(left,lower,upper),Within(right,lower,upper)>

def covered(plan: Pieces.Ranges,+first: Nat,+rows: Nat,+lower: Nat,+upper: Nat,coverage: Coverage(plan,first,rows),
  +lower_bound: {Nat.is_le(lower,first) == True{} : Bool},+upper_bound: {Nat.is_le(Nat.add(first,rows),upper) == True{} : Bool}) -> Within(plan,lower,upper):
  match plan:
    case Pieces.Range{+native_first,+native_rows}:
      LeafCoverage{+first_equal,rows_equal} = coverage
      LeafWithin{
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(lower,_) == True{} : Bool} lower_bound,
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(native_rows)),upper) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound}
    case Pieces.Split{+left,+right}:
      ForkCoverage{+left_rows,+right_rows,+rows_equal,left_coverage,right_coverage} = coverage
      KernelGemm.Both{
        covered(left,first,left_rows,lower,upper,left_coverage,lower_bound,
          Order.transitive(Nat.add(first,left_rows),Nat.add(first,Nat.add(left_rows,right_rows)),upper,
            Order.offset_le(first,left_rows,Nat.add(left_rows,right_rows),Order.le_add(left_rows,right_rows)),
            %rows_equal : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound)),
        covered(right,Nat.add(first,left_rows),right_rows,lower,upper,right_coverage,
          Order.transitive(lower,first,Nat.add(first,left_rows),lower_bound,Order.le_add(first,left_rows)),
          %Equal.sym(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Addition.associative(first,left_rows,right_rows)) :
            {Nat.is_le(_,upper) == True{} : Bool}
          %rows_equal : {Nat.is_le(Nat.add(first,_),upper) == True{} : Bool} upper_bound)}
