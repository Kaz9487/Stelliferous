# A phase clipped to a leaf band covers exactly its rows.
import Base
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_extrema.bend as Extrema
import ./nat_division_order.bend as Division
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./input_gradient_tiled_geometry.bend as Geometry

def clipped_lower(+phase: Nat,+first: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(first,Nat.add(phase,Nat.mul(Ceiling.ceiling(Nat.sub(first,phase),stride),stride))) == True{} : Bool}:
  %Order.subtraction_le(first,phase,Nat.mul(Ceiling.ceiling(Nat.sub(first,phase),stride),stride)) : {_ == True{} : Bool}
  Ceiling.covers(Nat.sub(first,phase),stride,positive)

def clipped_upper(+phase: Nat,+last: Nat,+stride: Nat,+high: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +nonzero: {Nat.is_lt(0n,high) == True{} : Bool},+bounded: {Nat.is_le(high,Ceiling.ceiling(Nat.sub(last,phase),stride)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(phase,Nat.mul(Nat.sub(high,1n),stride)),last) == True{} : Bool}:
  match high:
    case 0n: Empty.absurd({Nat.is_lt(Nat.add(phase,Nat.mul(Nat.sub(0n,1n),stride)),last) == True{} : Bool},WordEncoding.false_true(nonzero))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.is_lt(Nat.add(phase,Nat.mul(_,stride)),last) == True{} : Bool}
      %Order.subtraction_lt(last,phase,Nat.mul(rest,stride)) : {_ == True{} : Bool}
      %Ceiling.upper_comparison(Nat.sub(last,phase),stride,rest,positive) : {_ == True{} : Bool}
      Bounds.predecessor_strict(rest,Ceiling.ceiling(Nat.sub(last,phase),stride),bounded)

type Band is Data:
  Band{phase: Nat,phase_count: Nat,stride: Nat,first: Nat,last: Nat,positive: {Nat.is_lt(0n,stride) == True{} : Bool}}
def phase(band: Band) -> Nat:
  Band{phase,count,stride,first,last,positive} = band
  phase
def stride(band: Band) -> Nat:
  Band{phase,count,stride,first,last,positive} = band
  stride
def first(band: Band) -> Nat:
  Band{phase,count,stride,first,last,positive} = band
  first
def last(band: Band) -> Nat:
  Band{phase,count,stride,first,last,positive} = band
  last
def phase_count(band: Band) -> Nat:
  Band{phase,count,stride,first,last,positive} = band
  count
def positive(+band: Band) -> {Nat.is_lt(0n,stride(band)) == True{} : Bool}:
  Band{phase,count,stride,first,last,positive} = band
  positive
def low(+band: Band) -> Nat:
  Ceiling.ceiling(Nat.sub(first(band),phase(band)),stride(band))
def high(+band: Band) -> Nat:
  Nat.min(phase_count(band),Ceiling.ceiling(Nat.sub(last(band),phase(band)),stride(band)))
def position(+band: Band,+index: Nat) -> Nat:
  Nat.add(phase(band),Nat.mul(index,stride(band)))
def start_point(+band: Band) -> Nat:
  position(band,low(band))
def last_point(+band: Band) -> Nat:
  position(band,Nat.sub(high(band),1n))

def start_inside(+band: Band) -> {Nat.is_le(first(band),start_point(band)) == True{} : Bool}:
  clipped_lower(phase(band),first(band),stride(band),positive(band))
def end_inside(+band: Band,+active: {Nat.is_lt(low(band),high(band)) == True{} : Bool}) ->
  {Nat.is_lt(last_point(band),last(band)) == True{} : Bool}:
  clipped_upper(phase(band),last(band),stride(band),high(band),positive(band),Order.positive_bound(low(band),high(band),active),
    Extrema.minimum_right(phase_count(band),Ceiling.ceiling(Nat.sub(last(band),phase(band)),stride(band))))
def ordered_points(+band: Band,+active: {Nat.is_lt(low(band),high(band)) == True{} : Bool}) ->
  {Nat.is_le(start_point(band),last_point(band)) == True{} : Bool}:
  Order.offset_le(phase(band),Nat.mul(low(band),stride(band)),Nat.mul(Nat.sub(high(band),1n),stride(band)),
    RowMajor.mul_le(low(band),Nat.sub(high(band),1n),stride(band),Order.before_last(low(band),high(band),active)))

# A row in the band and phase occurs in precisely this clipped job range.
def member_lower(+band: Band,+index: Nat,+in_band: {Nat.is_le(first(band),position(band,index)) == True{} : Bool}) ->
  {Nat.is_le(low(band),index) == True{} : Bool}:
  Ceiling.minimal(Nat.sub(first(band),phase(band)),stride(band),index,positive(band),
    %Equal.sym(Bool,Nat.is_le(Nat.sub(first(band),phase(band)),Nat.mul(index,stride(band))),Nat.is_le(first(band),position(band,index)),Order.subtraction_le(first(band),phase(band),Nat.mul(index,stride(band)))) : {_ == True{} : Bool} in_band)
def member_upper(+band: Band,+index: Nat,+in_phase: {Nat.is_lt(index,phase_count(band)) == True{} : Bool},
  +in_band: {Nat.is_lt(position(band,index),last(band)) == True{} : Bool}) ->
  {Nat.is_lt(index,high(band)) == True{} : Bool}:
  Extrema.minimum_strict_lower(phase_count(band),Ceiling.ceiling(Nat.sub(last(band),phase(band)),stride(band)),index,in_phase,
    %Equal.sym(Bool,Nat.is_lt(index,Ceiling.ceiling(Nat.sub(last(band),phase(band)),stride(band))),Nat.is_lt(Nat.mul(index,stride(band)),Nat.sub(last(band),phase(band))),Ceiling.upper_comparison(Nat.sub(last(band),phase(band)),stride(band),index,positive(band))) : {_ == True{} : Bool}
    %Equal.sym(Bool,Nat.is_lt(Nat.mul(index,stride(band)),Nat.sub(last(band),phase(band))),Nat.is_lt(position(band,index),last(band)),Order.subtraction_lt(last(band),phase(band),Nat.mul(index,stride(band)))) : {_ == True{} : Bool} in_band)

def quotient_increment(+value: Nat,+distance: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.add(value,distance),stride),Nat.add(distance,Nat.div(value,stride))) == True{} : Bool}:
  %Division.quotient_shift(value,stride,distance,positive) : {Nat.is_le(Nat.div(Nat.add(value,distance),stride),_) == True{} : Bool}
  Division.quotient_le(Nat.add(value,distance),Nat.add(Nat.mul(distance,stride),value),stride,positive,
    %Order.commutative(value,Nat.mul(distance,stride)) : {Nat.is_le(Nat.add(value,distance),_) == True{} : Bool}
    Order.offset_le(value,distance,Nat.mul(distance,stride),RowMajor.product_covers(distance,stride,positive)))

def row_origin_upper(+first: Nat,+row: Nat,+padding: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +lower: {Nat.is_le(first,row) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.add(row,padding),stride),Nat.add(Nat.sub(row,first),Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}:
  +distance = Nat.sub(row,first)
  %Equal.trans(Nat,Nat.add(Nat.add(first,padding),distance),Nat.add(first,Nat.add(padding,distance)),Nat.add(row,padding),Addition.associative(first,padding,distance),
      Equal.trans(Nat,Nat.add(first,Nat.add(padding,distance)),Nat.add(first,Nat.add(distance,padding)),Nat.add(row,padding),Equal.cong(Nat,Nat,value => Nat.add(first,value),Nat.add(padding,distance),Nat.add(distance,padding),Order.commutative(padding,distance)),
        Equal.trans(Nat,Nat.add(first,Nat.add(distance,padding)),Nat.add(Nat.add(first,distance),padding),Nat.add(row,padding),Equal.sym(Nat,Nat.add(Nat.add(first,distance),padding),Nat.add(first,Nat.add(distance,padding)),Addition.associative(first,distance,padding)),
          Equal.cong(Nat,Nat,value => Nat.add(value,padding),Nat.add(first,distance),row,Order.restore_shift(first,row,lower))))) :
      {Nat.is_le(Nat.div(_,stride),Nat.add(distance,Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}
  quotient_increment(Nat.add(first,padding),distance,stride,positive)

def origin(+band: Band,+padding: Nat,+kernel: Nat) -> Nat:
  Nat.add(Nat.div(Nat.add(first(band),padding),stride(band)),1n+kernel)
def sample_low(+band: Band,+padding: Nat,+kernel: Nat,+taps: Nat) -> Nat:
  Nat.sub(Nat.add(Nat.div(Nat.add(start_point(band),padding),stride(band)),1n+kernel),taps)
def sample_high(+band: Band,+padding: Nat,+kernel: Nat) -> Nat:
  Nat.add(Nat.div(Nat.add(last_point(band),padding),stride(band)),1n+kernel)

def halo_inside_origin(+band: Band,+padding: Nat,+kernel: Nat,+halo: Nat,+bounded: {Nat.is_le(halo,kernel) == True{} : Bool}) ->
  {Nat.is_le(halo,origin(band,padding,kernel)) == True{} : Bool}:
  +quotient = Nat.div(Nat.add(first(band),padding),stride(band))
  Order.transitive(halo,kernel,origin(band,padding,kernel),bounded,
    Order.transitive(kernel,1n+kernel,origin(band,padding,kernel),Bounds.strict_le(kernel,1n+kernel,Order.lt_succ(kernel,kernel,Order.reflexive(kernel))),
      %Order.commutative(1n+kernel,quotient) : {Nat.is_le(1n+kernel,_) == True{} : Bool} Order.le_add(1n+kernel,quotient)))

def sample_lower(+band: Band,+padding: Nat,+kernel: Nat,+taps: Nat,+halo: Nat,
  +taps_bounded: {Nat.is_le(taps,halo) == True{} : Bool},+halo_bounded: {Nat.is_le(halo,kernel) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(origin(band,padding,kernel),halo),sample_low(band,padding,kernel,taps)) == True{} : Bool}:
  +high = Nat.add(Nat.div(Nat.add(start_point(band),padding),stride(band)),1n+kernel)
  +lower = Order.add_le(Nat.div(Nat.add(first(band),padding),stride(band)),Nat.div(Nat.add(start_point(band),padding),stride(band)),1n+kernel,
    Division.quotient_le(Nat.add(first(band),padding),Nat.add(start_point(band),padding),stride(band),positive(band),Order.add_le(first(band),start_point(band),padding,start_inside(band))))
  +inside = Order.transitive(taps,halo,high,taps_bounded,
    Order.transitive(halo,origin(band,padding,kernel),high,halo_inside_origin(band,padding,kernel,halo,halo_bounded),lower))
  Geometry.low_from_envelope(origin(band,padding,kernel),high,taps,halo,lower,taps_bounded,inside)

def sample_ordered(+band: Band,+padding: Nat,+kernel: Nat,+taps: Nat,+active: {Nat.is_lt(low(band),high(band)) == True{} : Bool}) ->
  {Nat.is_le(sample_low(band,padding,kernel,taps),sample_high(band,padding,kernel)) == True{} : Bool}:
  +start = Nat.add(Nat.div(Nat.add(start_point(band),padding),stride(band)),1n+kernel)
  Order.transitive(sample_low(band,padding,kernel,taps),start,sample_high(band,padding,kernel),Order.sub_le(start,taps),
    Order.add_le(Nat.div(Nat.add(start_point(band),padding),stride(band)),Nat.div(Nat.add(last_point(band),padding),stride(band)),1n+kernel,
      Division.quotient_le(Nat.add(start_point(band),padding),Nat.add(last_point(band),padding),stride(band),positive(band),Order.add_le(start_point(band),last_point(band),padding,ordered_points(band,active)))))
def end_origin(+band: Band,+padding: Nat,+kernel: Nat) -> Nat:
  Nat.add(Nat.div(Nat.add(Nat.sub(last(band),1n),padding),stride(band)),1n+kernel)
def sample_upper(+band: Band,+padding: Nat,+kernel: Nat,+active: {Nat.is_lt(low(band),high(band)) == True{} : Bool}) ->
  {Nat.is_le(sample_high(band,padding,kernel),end_origin(band,padding,kernel)) == True{} : Bool}:
  Order.add_le(Nat.div(Nat.add(last_point(band),padding),stride(band)),Nat.div(Nat.add(Nat.sub(last(band),1n),padding),stride(band)),1n+kernel,
    Division.quotient_le(Nat.add(last_point(band),padding),Nat.add(Nat.sub(last(band),1n),padding),stride(band),positive(band),
      Order.add_le(last_point(band),Nat.sub(last(band),1n),padding,Order.before_last(last_point(band),last(band),end_inside(band,active)))))

# The outputs below first + count read quotients at most count - 1 past the
# quotient of first.
def end_quotient(+first: Nat,+count: Nat,+padding: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.add(Nat.sub(Nat.add(first,count),1n),padding),stride),Nat.add(Nat.sub(count,1n),Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) :
        {Nat.is_le(Nat.div(Nat.add(Nat.sub(_,1n),padding),stride),Nat.div(Nat.add(first,padding),stride)) == True{} : Bool}
      Division.quotient_le(Nat.add(Nat.sub(first,1n),padding),Nat.add(first,padding),stride,positive,Order.add_le(Nat.sub(first,1n),first,padding,Order.sub_le(first,1n)))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(Nat.add(first,1n+rest),1n),Nat.add(first,rest),Order.subtract_one_complement(first,rest)) :
        {Nat.is_le(Nat.div(Nat.add(_,padding),stride),Nat.add(Nat.sub(rest,0n),Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) :
        {Nat.is_le(Nat.div(Nat.add(Nat.add(first,rest),padding),stride),Nat.add(_,Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}
      %Order.subtract_shift(first,rest) :
        {Nat.is_le(Nat.div(Nat.add(Nat.add(first,rest),padding),stride),Nat.add(_,Nat.div(Nat.add(first,padding),stride))) == True{} : Bool}
      row_origin_upper(first,Nat.add(first,rest),padding,stride,positive,Order.le_add(first,rest))

# The clipped phase indices describe physical rows inside the band.
def position_lower(+band: Band,+index: Nat,+lower: {Nat.is_le(low(band),index) == True{} : Bool}) -> {Nat.is_le(first(band),position(band,index)) == True{} : Bool}:
  Order.transitive(first(band),start_point(band),position(band,index),start_inside(band),
    Order.offset_le(phase(band),Nat.mul(low(band),stride(band)),Nat.mul(index,stride(band)),RowMajor.mul_le(low(band),index,stride(band),lower)))
def position_upper(+band: Band,+index: Nat,+lower: {Nat.is_le(low(band),index) == True{} : Bool},+upper: {Nat.is_lt(index,high(band)) == True{} : Bool}) -> {Nat.is_lt(position(band,index),last(band)) == True{} : Bool}:
  +active = Order.le_lt_trans(low(band),index,high(band),lower,upper)
  Order.le_lt_trans(position(band,index),last_point(band),last(band),
    Order.offset_le(phase(band),Nat.mul(index,stride(band)),Nat.mul(Nat.sub(high(band),1n),stride(band)),RowMajor.mul_le(index,Nat.sub(high(band),1n),stride(band),Order.before_last(index,high(band),upper))),end_inside(band,active))
