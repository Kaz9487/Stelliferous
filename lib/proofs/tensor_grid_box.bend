# A piece of a grid is a box of the cell axes: cell first + index of the
# piece has the coordinates of first plus the coordinates of index over the
# box. The axes before the last prefix axis hold one index of the piece, the
# last prefix axis its units, every later axis all of its indices.
import Base
import ../tensor_grid.bend as Grid
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as GridRanges
import ./nat_ceiling.bend as NatCeiling
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Logic
import ./tensor_tiled_public.bend as TiledPublic
import ./tensor_move_readback.bend as Held

# ---- Sums over coordinates ----

# The row-major coordinates of index over extents, each times the step of its
# axis, summed.
def stepped_sum(extents: List<&2,U32>,steps: List<&2,U32>,+index: Nat) -> U32:
  match extents steps:
    case Con{extent,+rest} Con{+step,later}:
      (U32.from_nat(Nat.div(index,ShapeLaws.elements(rest))) * step + stepped_sum(rest,later,Nat.mod(index,ShapeLaws.elements(rest))) : U32)
    case extents steps: 0

def zero_parts(+divisor: Nat) -> {Nat.divmod(0n,divisor) == (0n,0n) : Nat & Nat}:
  match divisor:
    case 0n: {==}
    case 1n+rest: {==}

def stepped_zero(+extents: List<&2,U32>,+steps: List<&2,U32>) -> {stepped_sum(extents,steps,0n) == 0 : U32}:
  match extents steps:
    case Nil{} +steps: {==}
    case Con{+extent,+rest} Nil{}: {==}
    case Con{+extent,+rest} Con{+step,+later}:
      %Equal.sym(Nat & Nat,Nat.divmod(0n,ShapeLaws.elements(rest)),(0n,0n),zero_parts(ShapeLaws.elements(rest))) :
        {(U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(rest,later,Pair.snd(Nat,Nat,_)) : U32) == 0 : U32}
      %Equal.sym(U32,stepped_sum(rest,later,0n),0,stepped_zero(rest,later)) : {(U32.from_nat(0n) * step + _ : U32) == 0 : U32}
      %Equal.sym(U32,(0 * step : U32),0,Arithmetic.multiply_zero_left(step)) : {(_ + 0 : U32) == 0 : U32}
      Arithmetic.add_zero(0)

def exchange(+first: U32,+second: U32,+third: U32) -> {(first + (second + third : U32) : U32) == (second + (first + third : U32) : U32) : U32}:
  Equal.trans(U32,(first + (second + third : U32) : U32),((first + second : U32) + third : U32),(second + (first + third : U32) : U32),
    Equal.sym(U32,((first + second : U32) + third : U32),(first + (second + third : U32) : U32),Arithmetic.add_associative(first,second,third)),
    Equal.trans(U32,((first + second : U32) + third : U32),((second + first : U32) + third : U32),(second + (first + third : U32) : U32),
      Equal.cong(U32,U32,word => (word + third : U32),(first + second : U32),(second + first : U32),Arithmetic.add_commutative(first,second)),
      Arithmetic.add_associative(second,first,third)))

def rotate(+first: U32,+second: U32,+third: U32) -> {((second + third : U32) + first : U32) == ((first + second : U32) + third : U32) : U32}:
  Equal.trans(U32,((second + third : U32) + first : U32),(first + (second + third : U32) : U32),((first + second : U32) + third : U32),
    Arithmetic.add_commutative((second + third : U32),first),
    Equal.sym(U32,((first + second : U32) + third : U32),(first + (second + third : U32) : U32),Arithmetic.add_associative(first,second,third)))

# ---- The box of a piece ----

# Cell axes with before axes ahead of the last prefix axis: the outer indices
# of the before axes, the extent of the last prefix axis, the cells of a unit,
# and the box of a piece of units indices of the last prefix axis.
def lead(extents: List<&2,U32>,before: Nat) -> Nat:
  match extents before:
    case Nil{} before: 1n
    case Con{extent,rest} 0n: 1n
    case Con{+extent,rest} 1n+later: Nat.mul(U32.to_nat(extent),lead(rest,later))

def prefix_extent(extents: List<&2,U32>,before: Nat) -> Nat:
  match extents before:
    case Nil{} before: 1n
    case Con{+extent,rest} 0n: U32.to_nat(extent)
    case Con{extent,rest} 1n+later: prefix_extent(rest,later)

def unit_of(extents: List<&2,U32>,before: Nat) -> Nat:
  match extents before:
    case Nil{} before: 1n
    case Con{extent,rest} 0n: ShapeLaws.elements(rest)
    case Con{extent,rest} 1n+later: unit_of(rest,later)

def box_from(extents: List<&2,U32>,before: Nat,+units: U32) -> List<&2,U32>:
  match extents before:
    case Nil{} before: []
    case Con{extent,rest} 0n: units <> rest
    case Con{extent,rest} 1n+later: 1 <> box_from(rest,later,units)

# The first cell of piece inner of outer index outer, in units of the last
# prefix axis.
def first_of(+extents: List<&2,U32>,+before: Nat,+outer: Nat,+inner: Nat) -> Nat:
  Nat.add(Nat.mul(outer,Nat.mul(prefix_extent(extents,before),unit_of(extents,before))),Nat.mul(inner,unit_of(extents,before)))

def lead_split(+extents: List<&2,U32>,+before: Nat) ->
  {ShapeLaws.elements(extents) == Nat.mul(lead(extents,before),Nat.mul(prefix_extent(extents,before),unit_of(extents,before))) : Nat}:
  match extents before:
    case Nil{} +before: {==}
    case Con{+extent,+rest} 0n:
      Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),0n),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),
        Addition.zero_right(Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest))))
    case Con{+extent,+rest} 1n+ +later:
      +whole = Nat.mul(prefix_extent(rest,later),unit_of(rest,later))
      %Equal.sym(Nat,ShapeLaws.elements(rest),Nat.mul(lead(rest,later),whole),lead_split(rest,later)) :
        {Nat.mul(U32.to_nat(extent),_) == Nat.mul(Nat.mul(U32.to_nat(extent),lead(rest,later)),whole) : Nat}
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),lead(rest,later)),whole),Nat.mul(U32.to_nat(extent),Nat.mul(lead(rest,later),whole)),
        RowMajor.assoc(U32.to_nat(extent),lead(rest,later),whole))

def lt_le(+low: Nat,+middle: Nat,+high: Nat,+below: {Nat.is_lt(low,middle) == True{} : Bool},+within: {Nat.is_le(middle,high) == True{} : Bool}) ->
  {Nat.is_lt(low,high) == True{} : Bool}:
  Bounds.predecessor_strict(low,high,Order.transitive(1n+low,middle,high,Order.strict_follow(low,middle,below),within))

# On the last prefix axis a piece starts at a whole unit and its cells carry
# into that axis.
def last_shift(+extent: U32,+rest: List<&2,U32>,+step: U32,+later: List<&2,U32>,+units: U32,+inner: Nat,+index: Nat,
  +room: {Nat.is_lt(Nat.add(Nat.mul(inner,ShapeLaws.elements(rest)),index),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest))) == True{} : Bool}) ->
  {(stepped_sum(Con{units,rest},Con{step,later},index) + stepped_sum(Con{extent,rest},Con{step,later},Nat.mul(inner,ShapeLaws.elements(rest))) : U32) ==
    stepped_sum(Con{extent,rest},Con{step,later},Nat.add(Nat.mul(inner,ShapeLaws.elements(rest)),index)) : U32}:
  +cells = ShapeLaws.elements(rest)
  +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),cells,Order.positive_bound(Nat.add(Nat.mul(inner,cells),index),Nat.mul(U32.to_nat(extent),cells),room))
  +quotient = Nat.div(index,cells)
  +tail = stepped_sum(rest,later,Nat.mod(index,cells))
  +shifted = (U32.from_nat(quotient) * step : U32)
  +start = (U32.from_nat(inner) * step : U32)
  +first_pair = Division.unique(Nat.mul(inner,cells),cells,inner,0n,Equal.sym(Nat,Nat.add(Nat.mul(inner,cells),0n),Nat.mul(inner,cells),Addition.zero_right(Nat.mul(inner,cells))),
    positive,positive)
  +sum_pair = DivisionOrder.shifted_pair(index,cells,inner,positive)
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.mul(inner,cells),cells),(inner,0n),first_pair) :
    {((shifted + tail : U32) + (U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(rest,later,Pair.snd(Nat,Nat,_)) : U32) : U32) ==
      (U32.from_nat(Nat.div(Nat.add(Nat.mul(inner,cells),index),cells)) * step + stepped_sum(rest,later,Nat.mod(Nat.add(Nat.mul(inner,cells),index),cells)) : U32) : U32}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(inner,cells),index),cells),(Nat.add(inner,quotient),Nat.mod(index,cells)),sum_pair) :
    {((shifted + tail : U32) + (start + stepped_sum(rest,later,0n) : U32) : U32) ==
      (U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(rest,later,Pair.snd(Nat,Nat,_)) : U32) : U32}
  %Equal.sym(U32,stepped_sum(rest,later,0n),0,stepped_zero(rest,later)) :
    {((shifted + tail : U32) + (start + _ : U32) : U32) == (U32.from_nat(Nat.add(inner,quotient)) * step + tail : U32) : U32}
  %Equal.sym(U32,(start + 0 : U32),start,Arithmetic.add_zero(start)) :
    {((shifted + tail : U32) + _ : U32) == (U32.from_nat(Nat.add(inner,quotient)) * step + tail : U32) : U32}
  %WordArithmetic.native_add(inner,quotient) : {((shifted + tail : U32) + start : U32) == (_ * step + tail : U32) : U32}
  %Equal.sym(U32,((U32.from_nat(inner) + U32.from_nat(quotient) : U32) * step : U32),(start + shifted : U32),Arithmetic.multiply_additive(U32.from_nat(inner),U32.from_nat(quotient),step)) :
    {((shifted + tail : U32) + start : U32) == (_ + tail : U32) : U32}
  rotate(start,shifted,tail)

# On an axis before the last prefix axis a piece holds one index: the cells of
# the piece stay inside it and the rest of the axes carry the shift.
def before_shift(+extent: U32,+rest: List<&2,U32>,+step: U32,+later: List<&2,U32>,+before: Nat,+units: U32,+outer: Nat,+inner: Nat,+index: Nat,
  +room: {Nat.is_lt(Nat.add(Nat.mul(inner,unit_of(rest,before)),index),Nat.mul(prefix_extent(rest,before),unit_of(rest,before))) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(box_from(rest,before,units))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,lead(rest,before)) == True{} : Bool},
  +shifted: {(stepped_sum(box_from(rest,before,units),later,index) + stepped_sum(rest,later,first_of(rest,before,Nat.mod(outer,lead(rest,before)),inner)) : U32) ==
    stepped_sum(rest,later,Nat.add(first_of(rest,before,Nat.mod(outer,lead(rest,before)),inner),index)) : U32}) ->
  {(stepped_sum(box_from(Con{extent,rest},1n+before,units),Con{step,later},index) + stepped_sum(Con{extent,rest},Con{step,later},first_of(Con{extent,rest},1n+before,outer,inner)) : U32) ==
    stepped_sum(Con{extent,rest},Con{step,later},Nat.add(first_of(Con{extent,rest},1n+before,outer,inner),index)) : U32}:
  +ahead = lead(rest,before)
  +whole = Nat.mul(prefix_extent(rest,before),unit_of(rest,before))
  +cells = ShapeLaws.elements(rest)
  +boxed = ShapeLaws.elements(box_from(rest,before,units))
  +split = lead_split(rest,before)
  +quotient = Nat.div(outer,ahead)
  +remaining = Nat.mod(outer,ahead)
  +offset = Nat.mul(inner,unit_of(rest,before))
  +start = Nat.add(Nat.mul(outer,whole),offset)
  +later_start = Nat.add(Nat.mul(remaining,whole),offset)
  +start_exact = Equal.trans(Nat,start,Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,ahead),remaining),whole),offset),Nat.add(Nat.mul(quotient,cells),later_start),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,whole),offset),outer,Nat.add(Nat.mul(quotient,ahead),remaining),DivisionOrder.quotient_exact(outer,ahead,positive)),
    {%Equal.sym(Nat,cells,Nat.mul(ahead,whole),split) :
        {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,ahead),remaining),whole),offset) == Nat.add(Nat.mul(quotient,_),later_start) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,ahead),remaining),whole),Nat.add(Nat.mul(Nat.mul(quotient,ahead),whole),Nat.mul(remaining,whole)),
          RowMajor.distrib(Nat.mul(quotient,ahead),remaining,whole)) :
        {Nat.add(_,offset) == Nat.add(Nat.mul(quotient,Nat.mul(ahead,whole)),later_start) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.mul(quotient,ahead),whole),Nat.mul(quotient,Nat.mul(ahead,whole)),RowMajor.assoc(quotient,ahead,whole)) :
        {Nat.add(Nat.add(_,Nat.mul(remaining,whole)),offset) == Nat.add(Nat.mul(quotient,Nat.mul(ahead,whole)),later_start) : Nat}
      Addition.associative(Nat.mul(quotient,Nat.mul(ahead,whole)),Nat.mul(remaining,whole),offset) :
      {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,ahead),remaining),whole),offset) == Nat.add(Nat.mul(quotient,cells),later_start) : Nat}})
  +remaining_small = DivisionOrder.remainder_small(outer,ahead,positive)
  +below = {%Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(remaining,whole),offset),index),Nat.add(Nat.mul(remaining,whole),Nat.add(offset,index)),
        Addition.associative(Nat.mul(remaining,whole),offset,index)) : {Nat.is_lt(_,cells) == True{} : Bool}
      lt_le(Nat.add(Nat.mul(remaining,whole),Nat.add(offset,index)),Nat.add(Nat.mul(remaining,whole),whole),cells,
        Order.offset_lt(Nat.mul(remaining,whole),Nat.add(offset,index),whole,room),
        {%Equal.sym(Nat,cells,Nat.mul(ahead,whole),split) : {Nat.is_le(Nat.add(Nat.mul(remaining,whole),whole),_) == True{} : Bool}
          %Order.commutative(whole,Nat.mul(remaining,whole)) : {Nat.is_le(_,Nat.mul(ahead,whole)) == True{} : Bool}
          RowMajor.mul_le(1n+remaining,ahead,whole,Order.strict_follow(remaining,ahead,remaining_small)) :
          {Nat.is_le(Nat.add(Nat.mul(remaining,whole),whole),cells) == True{} : Bool}}) :
    {Nat.is_lt(Nat.add(later_start,index),cells) == True{} : Bool}}
  +cells_positive = Order.positive_bound(Nat.add(later_start,index),cells,below)
  +start_pair = Division.unique(start,cells,quotient,later_start,start_exact,cells_positive,
    Order.le_lt_trans(later_start,Nat.add(later_start,index),cells,Order.le_add(later_start,index),below))
  +sum_pair = Division.unique(Nat.add(start,index),cells,quotient,Nat.add(later_start,index),
    Equal.trans(Nat,Nat.add(start,index),Nat.add(Nat.add(Nat.mul(quotient,cells),later_start),index),Nat.add(Nat.mul(quotient,cells),Nat.add(later_start,index)),
      Equal.cong(Nat,Nat,value => Nat.add(value,index),start,Nat.add(Nat.mul(quotient,cells),later_start),start_exact),
      Addition.associative(Nat.mul(quotient,cells),later_start,index)),
    cells_positive,below)
  +box_pair = Division.small(index,boxed,inside)
  +head = (U32.from_nat(quotient) * step : U32)
  +boxed_sum = stepped_sum(box_from(rest,before,units),later,index)
  +start_sum = stepped_sum(rest,later,later_start)
  %Equal.sym(Nat & Nat,Nat.divmod(index,boxed),(0n,index),box_pair) :
    {((U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(box_from(rest,before,units),later,Pair.snd(Nat,Nat,_)) : U32) +
        (U32.from_nat(Nat.div(start,cells)) * step + stepped_sum(rest,later,Nat.mod(start,cells)) : U32) : U32) ==
      (U32.from_nat(Nat.div(Nat.add(start,index),cells)) * step + stepped_sum(rest,later,Nat.mod(Nat.add(start,index),cells)) : U32) : U32}
  %Equal.sym(Nat & Nat,Nat.divmod(start,cells),(quotient,later_start),start_pair) :
    {((U32.from_nat(0n) * step + boxed_sum : U32) + (U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(rest,later,Pair.snd(Nat,Nat,_)) : U32) : U32) ==
      (U32.from_nat(Nat.div(Nat.add(start,index),cells)) * step + stepped_sum(rest,later,Nat.mod(Nat.add(start,index),cells)) : U32) : U32}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(start,index),cells),(quotient,Nat.add(later_start,index)),sum_pair) :
    {((U32.from_nat(0n) * step + boxed_sum : U32) + (head + start_sum : U32) : U32) ==
      (U32.from_nat(Pair.fst(Nat,Nat,_)) * step + stepped_sum(rest,later,Pair.snd(Nat,Nat,_)) : U32) : U32}
  %shifted : {((U32.from_nat(0n) * step + boxed_sum : U32) + (head + start_sum : U32) : U32) == (head + _ : U32) : U32}
  %Equal.sym(U32,(0 * step : U32),0,Arithmetic.multiply_zero_left(step)) : {((_ + boxed_sum : U32) + (head + start_sum : U32) : U32) == (head + (boxed_sum + start_sum : U32) : U32) : U32}
  %Equal.sym(U32,(0 + boxed_sum : U32),boxed_sum,Arithmetic.add_zero_left(boxed_sum)) : {(_ + (head + start_sum : U32) : U32) == (head + (boxed_sum + start_sum : U32) : U32) : U32}
  exchange(boxed_sum,head,start_sum)

# The coordinates of the cell first + index of a piece are those of first plus
# those of index over the box, weighted by any steps.
def box_shift(+extents: List<&2,U32>,+steps: List<&2,U32>,+before: Nat,+units: U32,+outer: Nat,+inner: Nat,+index: Nat,
  +outer_in: {Nat.is_lt(outer,lead(extents,before)) == True{} : Bool},
  +room: {Nat.is_lt(Nat.add(Nat.mul(inner,unit_of(extents,before)),index),Nat.mul(prefix_extent(extents,before),unit_of(extents,before))) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(box_from(extents,before,units))) == True{} : Bool}) ->
  {(stepped_sum(box_from(extents,before,units),steps,index) + stepped_sum(extents,steps,first_of(extents,before,outer,inner)) : U32) ==
    stepped_sum(extents,steps,Nat.add(first_of(extents,before,outer,inner),index)) : U32}:
  match extents steps before:
    case Nil{} +steps +before:
      Arithmetic.add_zero(0)
    case Con{+extent,+rest} Nil{} 0n:
      Arithmetic.add_zero(0)
    case Con{+extent,+rest} Nil{} 1n+ +later:
      Arithmetic.add_zero(0)
    case Con{+extent,+rest} Con{+step,+later} 0n:
      match outer:
        case 0n:
          last_shift(extent,rest,step,later,units,inner,index,room)
        case 1n+ +more:
          Empty.absurd({(stepped_sum(box_from(Con{extent,rest},0n,units),Con{step,later},index) +
              stepped_sum(Con{extent,rest},Con{step,later},first_of(Con{extent,rest},0n,1n+more,inner)) : U32) ==
            stepped_sum(Con{extent,rest},Con{step,later},Nat.add(first_of(Con{extent,rest},0n,1n+more,inner),index)) : U32},Order.below_zero(more,outer_in))
    case Con{+extent,+rest} Con{+step,+later} 1n+ +below:
      +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),lead(rest,below),Order.positive_bound(outer,Nat.mul(U32.to_nat(extent),lead(rest,below)),outer_in))
      +boxed = ShapeLaws.elements(box_from(rest,below,units))
      +inside_rest = {%Addition.zero_right(boxed) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,boxed) == True{} : Bool}}
      before_shift(extent,rest,step,later,below,units,outer,inner,index,room,inside_rest,positive,
        box_shift(rest,later,below,units,Nat.mod(outer,lead(rest,below)),inner,index,DivisionOrder.remainder_small(outer,lead(rest,below),positive),room,inside_rest))

def box_elements(+extents: List<&2,U32>,+before: Nat,+units: U32,+long: {Nat.is_lt(before,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {ShapeLaws.elements(box_from(extents,before,units)) == Nat.mul(U32.to_nat(units),unit_of(extents,before)) : Nat}:
  match extents before:
    case Nil{} +before:
      Empty.absurd({ShapeLaws.elements(box_from(Nil{},before,units)) == Nat.mul(U32.to_nat(units),unit_of(Nil{},before)) : Nat},Order.below_zero(before,long))
    case Con{+extent,+rest} 0n:
      {==}
    case Con{+extent,+rest} 1n+ +later:
      +boxed = ShapeLaws.elements(box_from(rest,later,units))
      Equal.trans(Nat,Nat.add(boxed,0n),boxed,Nat.mul(U32.to_nat(units),unit_of(rest,later)),Addition.zero_right(boxed),box_elements(rest,later,units,long))

# ---- The pieces of a grid ----

# The first unit of piece inner of outer index outer along the last prefix
# axis, and where the next piece starts.
def piece_base(+outer: Nat,+inner: Nat,+extent: Nat,+block: Nat) -> Nat:
  Nat.add(Nat.mul(outer,extent),Nat.mul(inner,block))

def next_outer(choice: Bool,+outer: Nat) -> Nat:
  match choice:
    case True{}: outer
    case False{}: 1n+outer

def next_inner(choice: Bool,+inner: Nat) -> Nat:
  match choice:
    case True{}: 1n+inner
    case False{}: 0n

def place_units(+outer: Nat,+inner: Nat,+extent: Nat,+block: Nat,+unit: Nat) ->
  {Laws.place(outer,inner,Nat.mul(extent,unit),Nat.mul(block,unit)) == Nat.mul(piece_base(outer,inner,extent,block),unit) : Nat}:
  %Equal.sym(Nat,Nat.mul(piece_base(outer,inner,extent,block),unit),Nat.add(Nat.mul(Nat.mul(outer,extent),unit),Nat.mul(Nat.mul(inner,block),unit)),
      RowMajor.distrib(Nat.mul(outer,extent),Nat.mul(inner,block),unit)) :
    {Laws.place(outer,inner,Nat.mul(extent,unit),Nat.mul(block,unit)) == _ : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(outer,extent),unit),Nat.mul(outer,Nat.mul(extent,unit)),RowMajor.assoc(outer,extent,unit)) :
    {Laws.place(outer,inner,Nat.mul(extent,unit),Nat.mul(block,unit)) == Nat.add(_,Nat.mul(Nat.mul(inner,block),unit)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(inner,block),unit),Nat.mul(inner,Nat.mul(block,unit)),RowMajor.assoc(inner,block,unit)) :
    {Laws.place(outer,inner,Nat.mul(extent,unit),Nat.mul(block,unit)) == Nat.add(Nat.mul(outer,Nat.mul(extent,unit)),_) : Nat}
  {==}

def start_base(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))) ==
    Nat.mul(piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)) : Nat}:
  +pieces = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +image = Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))
  +size = Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))
  +room = Order.transitive(piece,pieces,U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(piece,pieces,inside),Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +back = Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),room)
  +located = Laws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(piece),
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(piece)),piece,back) : {Nat.is_le(_,pieces) == True{} : Bool} Bounds.strict_le(piece,pieces,inside) :
      {Nat.is_le(U32.to_nat(U32.from_nat(piece)),pieces) == True{} : Bool}})
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),
    Laws.place(Nat.div(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),Nat.mod(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),image,size),
    Nat.mul(piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)),located,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(piece)),piece,back) :
        {Laws.place(Nat.div(_,U32.to_nat(per)),Nat.mod(_,U32.to_nat(per)),image,size) ==
          Nat.mul(piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)) : Nat}
      place_units(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block),U32.to_nat(unit_cells)) :
      {Laws.place(Nat.div(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),Nat.mod(U32.to_nat(U32.from_nat(piece)),U32.to_nat(per)),image,size) ==
        Nat.mul(piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)) : Nat}})

def next_start_within(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +test: {Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))) ==
    Nat.mul(piece_base(Nat.div(piece,U32.to_nat(per)),1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)) : Nat}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +per_count = U32.to_nat(per)
  +pieces = Nat.mul(U32.to_nat(outer),per_count)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +exact = DivisionOrder.quotient_exact(piece,per_count,per_positive)
  +next_room = Order.strict_follow(piece,pieces,inside)
  +back = Bounds.roundtrip(1n+piece,cap,cap_ok,Order.transitive(1n+piece,pieces,U32.to_nat(cap),next_room,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +stepped = Equal.trans(Nat,1n+piece,1n+Nat.add(Nat.mul(quotient,per_count),inner),Nat.add(Nat.mul(quotient,per_count),1n+inner),
    Equal.cong(Nat,Nat,value => 1n+value,piece,Nat.add(Nat.mul(quotient,per_count),inner),exact),
    Equal.sym(Nat,Nat.add(Nat.mul(quotient,per_count),1n+inner),1n+Nat.add(Nat.mul(quotient,per_count),inner),Addition.successor_right(Nat.mul(quotient,per_count),inner)))
  +last = Laws.inner_last(1n+inner,per_count,U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),
    Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape),test)
  +inside_next = {%stepped : {Nat.is_le(_,pieces) == True{} : Bool} next_room : {Nat.is_le(Nat.add(Nat.mul(quotient,per_count),1n+inner),pieces) == True{} : Bool}}
  +base = piece_base(quotient,1n+inner,U32.to_nat(extent),U32.to_nat(block))
  +bound = Order.transitive(Nat.mul(base,U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(cap),
    RowMajor.mul_le(base,Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),
      Laws.within(quotient,1n+inner,U32.to_nat(outer),per_count,U32.to_nat(extent),U32.to_nat(block),per_positive,last,inside_next)),
    Laws.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),
    Laws.place(quotient,1n+inner,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(base,U32.to_nat(unit_cells)),
    Laws.start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),U32.from_nat(1n+piece),quotient,1n+inner,per_positive,
      Laws.per_room(limits,outer,extent,unit_cells,block,per,shape),Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape),
      Equal.trans(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,Nat.add(Nat.mul(quotient,per_count),1n+inner),back,stepped),test,bound),
    place_units(quotient,1n+inner,U32.to_nat(extent),U32.to_nat(block),U32.to_nat(unit_cells)))

def next_start_wrapped(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +test: {Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)) == False{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))) ==
    Nat.mul(piece_base(1n+Nat.div(piece,U32.to_nat(per)),0n,U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)) : Nat}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +per_count = U32.to_nat(per)
  +pieces = Nat.mul(U32.to_nat(outer),per_count)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +exact = DivisionOrder.quotient_exact(piece,per_count,per_positive)
  +next_room = Order.strict_follow(piece,pieces,inside)
  +back = Bounds.roundtrip(1n+piece,cap,cap_ok,Order.transitive(1n+piece,pieces,U32.to_nat(cap),next_room,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +stepped = Equal.trans(Nat,1n+piece,1n+Nat.add(Nat.mul(quotient,per_count),inner),Nat.add(Nat.mul(quotient,per_count),1n+inner),
    Equal.cong(Nat,Nat,value => 1n+value,piece,Nat.add(Nat.mul(quotient,per_count),inner),exact),
    Equal.sym(Nat,Nat.add(Nat.mul(quotient,per_count),1n+inner),1n+Nat.add(Nat.mul(quotient,per_count),inner),Addition.successor_right(Nat.mul(quotient,per_count),inner)))
  +full = Order.antisymmetric(1n+inner,per_count,Order.strict_follow(inner,per_count,DivisionOrder.remainder_small(piece,per_count,per_positive)),
    Order.not_less(1n+inner,per_count,test))
  +wrapped = Equal.trans(Nat,1n+piece,Nat.add(Nat.mul(quotient,per_count),1n+inner),Nat.add(Nat.mul(1n+quotient,per_count),0n),stepped,
    Equal.trans(Nat,Nat.add(Nat.mul(quotient,per_count),1n+inner),Nat.add(Nat.mul(quotient,per_count),per_count),Nat.add(Nat.mul(1n+quotient,per_count),0n),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(quotient,per_count),value),1n+inner,per_count,full),
      Equal.trans(Nat,Nat.add(Nat.mul(quotient,per_count),per_count),Nat.mul(1n+quotient,per_count),Nat.add(Nat.mul(1n+quotient,per_count),0n),
        Order.commutative(Nat.mul(quotient,per_count),per_count),
        Equal.sym(Nat,Nat.add(Nat.mul(1n+quotient,per_count),0n),Nat.mul(1n+quotient,per_count),Addition.zero_right(Nat.mul(1n+quotient,per_count))))))
  +inside_next = {%wrapped : {Nat.is_le(_,pieces) == True{} : Bool} next_room : {Nat.is_le(Nat.add(Nat.mul(1n+quotient,per_count),0n),pieces) == True{} : Bool}}
  +base = piece_base(1n+quotient,0n,U32.to_nat(extent),U32.to_nat(block))
  +bound = Order.transitive(Nat.mul(base,U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(cap),
    RowMajor.mul_le(base,Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),
      Laws.within(1n+quotient,0n,U32.to_nat(outer),per_count,U32.to_nat(extent),U32.to_nat(block),per_positive,
        Laws.extent_positive(limits,outer,extent,unit_cells,block,per,shape),inside_next)),
    Laws.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))),
    Laws.place(1n+quotient,0n,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(base,U32.to_nat(unit_cells)),
    Laws.start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),U32.from_nat(1n+piece),1n+quotient,0n,per_positive,
      Laws.per_room(limits,outer,extent,unit_cells,block,per,shape),Laws.shape_unit(limits,outer,extent,unit_cells,block,per,shape),
      Equal.trans(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,Nat.add(Nat.mul(1n+quotient,per_count),0n),back,wrapped),per_positive,bound),
    place_units(1n+quotient,0n,U32.to_nat(extent),U32.to_nat(block),U32.to_nat(unit_cells)))

def next_start_with(choice: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +test: {Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)) == choice : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))) ==
    Nat.mul(piece_base(next_outer(choice,Nat.div(piece,U32.to_nat(per))),next_inner(choice,Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
      U32.to_nat(unit_cells)) : Nat}:
  match choice:
    case True{}: next_start_within(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside,test)
    case False{}: next_start_wrapped(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside,test)

def base_before_next(choice: Bool,+outer: Nat,+inner: Nat,+per: Nat,+extent: Nat,+block: Nat,+positive: {Nat.is_lt(0n,block) == True{} : Bool},
  +per_exact: {per == NatCeiling.ceiling(extent,block) : Nat},+inner_in: {Nat.is_lt(inner,per) == True{} : Bool},+test: {Nat.is_lt(1n+inner,per) == choice : Bool}) ->
  {Nat.is_le(piece_base(outer,inner,extent,block),piece_base(next_outer(choice,outer),next_inner(choice,inner),extent,block)) == True{} : Bool}:
  match choice:
    case True{}:
      Order.offset_le(Nat.mul(outer,extent),Nat.mul(inner,block),Nat.mul(1n+inner,block),Order.right_le(block,Nat.mul(inner,block)))
    case False{}:
      +last = Laws.inner_last(inner,per,extent,block,positive,per_exact,inner_in)
      %Equal.sym(Nat,Nat.add(Nat.add(extent,Nat.mul(outer,extent)),0n),Nat.add(extent,Nat.mul(outer,extent)),Addition.zero_right(Nat.add(extent,Nat.mul(outer,extent)))) :
        {Nat.is_le(piece_base(outer,inner,extent,block),_) == True{} : Bool}
      %Order.commutative(Nat.mul(outer,extent),extent) : {Nat.is_le(piece_base(outer,inner,extent,block),_) == True{} : Bool}
      Order.offset_le(Nat.mul(outer,extent),Nat.mul(inner,block),extent,Bounds.strict_le(Nat.mul(inner,block),extent,last))

def next_before_end(choice: Bool,+outer: Nat,+inner: Nat,+per: Nat,+extent: Nat,+block: Nat,+positive: {Nat.is_lt(0n,block) == True{} : Bool},
  +per_exact: {per == NatCeiling.ceiling(extent,block) : Nat},+test: {Nat.is_lt(1n+inner,per) == choice : Bool}) ->
  {Nat.is_le(piece_base(next_outer(choice,outer),next_inner(choice,inner),extent,block),Nat.add(Nat.mul(outer,extent),extent)) == True{} : Bool}:
  match choice:
    case True{}:
      Order.offset_le(Nat.mul(outer,extent),Nat.mul(1n+inner,block),extent,
        Bounds.strict_le(Nat.mul(1n+inner,block),extent,Laws.inner_last(1n+inner,per,extent,block,positive,per_exact,test)))
    case False{}:
      %Equal.sym(Nat,Nat.add(Nat.add(extent,Nat.mul(outer,extent)),0n),Nat.add(extent,Nat.mul(outer,extent)),Addition.zero_right(Nat.add(extent,Nat.mul(outer,extent)))) :
        {Nat.is_le(_,Nat.add(Nat.mul(outer,extent),extent)) == True{} : Bool}
      %Order.commutative(Nat.mul(outer,extent),extent) : {Nat.is_le(_,Nat.add(Nat.mul(outer,extent),extent)) == True{} : Bool}
      Order.reflexive(Nat.add(Nat.mul(outer,extent),extent))

# A piece holds whole units: its cells are its units of the last prefix axis
# times the cells of a unit.
def piece_span(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))) ==
    Nat.mul(Nat.sub(piece_base(next_outer(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.div(piece,U32.to_nat(per))),
        next_inner(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
      piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block))),U32.to_nat(unit_cells)) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +per_count = U32.to_nat(per)
  +choice = Nat.is_lt(1n+Nat.mod(piece,per_count),per_count)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +unit = U32.to_nat(unit_cells)
  +base = piece_base(quotient,inner,U32.to_nat(extent),U32.to_nat(block))
  +next = piece_base(next_outer(choice,quotient),next_inner(choice,inner),U32.to_nat(extent),U32.to_nat(block))
  +difference = Nat.sub(next,base)
  +span = U32.to_nat(Grid.span(grid,U32.from_nat(piece)))
  +here = start_base(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside)
  +there = next_start_with(choice,limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside,{==})
  +ordered = base_before_next(choice,quotient,inner,per_count,U32.to_nat(extent),U32.to_nat(block),Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),
    Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape),DivisionOrder.remainder_small(piece,per_count,Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)),{==})
  +grown = Equal.trans(Nat,Nat.mul(next,unit),Nat.mul(Nat.add(base,difference),unit),Nat.add(Nat.mul(base,unit),Nat.mul(difference,unit)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,unit),next,Nat.add(base,difference),Equal.sym(Nat,Nat.add(base,difference),next,Order.restore_shift(base,next,ordered))),
    RowMajor.distrib(base,difference,unit))
  +placed = GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)
  Order.cancel_sum(Nat.mul(base,unit),span,Nat.mul(difference,unit),
    Equal.trans(Nat,Nat.add(Nat.mul(base,unit),span),Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),span),Nat.add(Nat.mul(base,unit),Nat.mul(difference,unit)),
      Equal.cong(Nat,Nat,value => Nat.add(value,span),Nat.mul(base,unit),U32.to_nat(Grid.start(grid,U32.from_nat(piece))),Equal.sym(Nat,U32.to_nat(Grid.start(grid,U32.from_nat(piece))),Nat.mul(base,unit),here)),
      Equal.trans(Nat,Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(piece))),span),U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece))),Nat.add(Nat.mul(base,unit),Nat.mul(difference,unit)),
        placed,Equal.trans(Nat,U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece))),Nat.mul(next,unit),Nat.add(Nat.mul(base,unit),Nat.mul(difference,unit)),there,grown))))

# The units of a piece along the last prefix axis.
def piece_units(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat((Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)) / U32.max(unit_cells,1) : U32)) ==
    Nat.sub(piece_base(next_outer(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.div(piece,U32.to_nat(per))),
        next_inner(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
      piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block))) : Nat}:
  Laws.Shape{+outer_positive,+unit_positive,+block_positive,+block_below,+per_exact,+room} = shape
  +span = Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))
  +unit = U32.to_nat(unit_cells)
  +difference = Nat.sub(piece_base(next_outer(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.div(piece,U32.to_nat(per))),
      next_inner(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
    piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)))
  +images = Nat.mul(U32.to_nat(outer),U32.to_nat(extent))
  +small = Order.transitive(unit,Nat.mul(images,unit),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.le_multiple(images,unit,Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,Laws.extent_positive(limits,outer,extent,unit_cells,block,per,shape))),room)
  %Equal.sym(U32,U32.max(unit_cells,1),unit_cells,Laws.maximum_kept(unit_cells,1,Order.positive_one(unit,unit_positive))) : {U32.to_nat((span / _ : U32)) == difference : Nat}
  Equal.trans(Nat,U32.to_nat((span / unit_cells : U32)),Nat.div(U32.to_nat(span),unit),difference,Laws.word_quotient(limits,span,unit_cells,small,unit_positive),
    Equal.trans(Nat,Nat.div(U32.to_nat(span),unit),Nat.div(Nat.mul(difference,unit),unit),difference,
      Equal.cong(Nat,Nat,value => Nat.div(value,unit),U32.to_nat(span),Nat.mul(difference,unit),piece_span(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)),
      Division.exact_quotient(difference,unit,unit_positive)))

# The cells of a piece from its first unit stay inside its outer index.
def piece_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.mul(Nat.mod(piece,U32.to_nat(per)),U32.to_nat(block)),U32.to_nat(unit_cells)),index),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) == True{} : Bool}:
  +per_count = U32.to_nat(per)
  +choice = Nat.is_lt(1n+Nat.mod(piece,per_count),per_count)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +unit = U32.to_nat(unit_cells)
  +extent_count = U32.to_nat(extent)
  +lead_units = Nat.mul(inner,U32.to_nat(block))
  +base = piece_base(quotient,inner,extent_count,U32.to_nat(block))
  +next = piece_base(next_outer(choice,quotient),next_inner(choice,inner),extent_count,U32.to_nat(block))
  +difference = Nat.sub(next,base)
  +positive = Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape)
  +per_exact = Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape)
  +ordered = base_before_next(choice,quotient,inner,per_count,extent_count,U32.to_nat(block),positive,per_exact,
    DivisionOrder.remainder_small(piece,per_count,Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)),{==})
  +ending = next_before_end(choice,quotient,inner,per_count,extent_count,U32.to_nat(block),positive,per_exact,{==})
  +regrouped = Equal.trans(Nat,next,Nat.add(base,difference),Nat.add(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference)),
    Equal.sym(Nat,Nat.add(base,difference),next,Order.restore_shift(base,next,ordered)),Addition.associative(Nat.mul(quotient,extent_count),lead_units,difference))
  +fits = Order.cancel_offset_le(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference),extent_count,
    {%regrouped : {Nat.is_le(_,Nat.add(Nat.mul(quotient,extent_count),extent_count)) == True{} : Bool} ending :
      {Nat.is_le(Nat.add(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference)),Nat.add(Nat.mul(quotient,extent_count),extent_count)) == True{} : Bool}})
  +below = {%piece_span(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside) : {Nat.is_lt(index,_) == True{} : Bool} index_in :
    {Nat.is_lt(index,Nat.mul(difference,unit)) == True{} : Bool}}
  lt_le(Nat.add(Nat.mul(lead_units,unit),index),Nat.add(Nat.mul(lead_units,unit),Nat.mul(difference,unit)),Nat.mul(extent_count,unit),
    Order.offset_lt(Nat.mul(lead_units,unit),index,Nat.mul(difference,unit),below),
    {%RowMajor.distrib(lead_units,difference,unit) : {Nat.is_le(_,Nat.mul(extent_count,unit)) == True{} : Bool}
      RowMajor.mul_le(Nat.add(lead_units,difference),extent_count,unit,fits) :
      {Nat.is_le(Nat.add(Nat.mul(lead_units,unit),Nat.mul(difference,unit)),Nat.mul(extent_count,unit)) == True{} : Bool}})

# ---- The prefix of a grid ----

# The axes past the first that the unit walk takes into its prefix.
def walked(extents: List<&2,U32>,fits: Bool,+target: U32) -> Nat:
  match extents fits:
    case extents True{}: 0n
    case Nil{} False{}: 0n
    case Con{last,Nil{}} False{}: 0n
    case Con{axis,+rest} False{}: 1n+walked(rest,U32.is_le(Grid.product(rest),target),target)

def units_prefix_of(+found: Grid.Units) -> U32:
  Grid.Units{+prefix,outer,extent,unit_cells} = found
  prefix

def units_outer_of(+found: Grid.Units) -> U32:
  Grid.Units{prefix,+outer,extent,unit_cells} = found
  outer

def units_extent_of(+found: Grid.Units) -> U32:
  Grid.Units{prefix,outer,+extent,unit_cells} = found
  extent

def grid_prefix(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,+prefix,outer,extent,unit_cells,block,per,pieces} = grid
  prefix

def grid_outer(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,+outer,extent,unit_cells,block,per,pieces} = grid
  outer

def grid_extent(+grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,outer,+extent,unit_cells,block,per,pieces} = grid
  extent

def prefix_kept(+found: Grid.Units,+target: U32) -> {grid_prefix(Grid.grid_of_units(found,target)) == units_prefix_of(found) : U32}:
  match found:
    case Grid.Units{+prefix,+outer,+extent,+unit_cells}: {==}

def outer_kept(+found: Grid.Units,+target: U32) -> {grid_outer(Grid.grid_of_units(found,target)) == units_outer_of(found) : U32}:
  match found:
    case Grid.Units{+prefix,+outer,+extent,+unit_cells}: {==}

def extent_kept(+found: Grid.Units,+target: U32) -> {grid_extent(Grid.grid_of_units(found,target)) == units_extent_of(found) : U32}:
  match found:
    case Grid.Units{+prefix,+outer,+extent,+unit_cells}: {==}

def next_index(+at: U32,+position: Nat,+tail: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(Nat.add(position,1n+tail),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((at + 1 : U32)) == 1n+position : Nat}:
  +bounded = {%Equal.sym(Nat,Nat.add(position,1n),1n+position,Addition.successor_width(position)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.strict_follow(position,U32.to_nat(cap),Order.room_index(position,tail,U32.to_nat(cap),room)) : {Nat.is_le(Nat.add(position,1n),U32.to_nat(cap)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((at + 1 : U32)),Nat.add(position,1n),1n+position,Arithmetic.sum_exact(cap,cap_ok,at,1,position,1n,exact,{==},bounded),
    Addition.successor_width(position))

def walk_prefix(+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(prefix),List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(units_prefix_of(Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target))) == Nat.add(U32.to_nat(prefix),walked(extents,fits,target)) : Nat}:
  match extents fits:
    case +extents True{}:
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},
          Laws.units_fit(extents,prefix,outer,extent,target)) :
        {U32.to_nat(units_prefix_of(_)) == Nat.add(U32.to_nat(prefix),walked(extents,True{},target)) : Nat}
      Equal.sym(Nat,Nat.add(U32.to_nat(prefix),0n),U32.to_nat(prefix),Addition.zero_right(U32.to_nat(prefix)))
    case Nil{} False{}: Equal.sym(Nat,Nat.add(U32.to_nat(prefix),0n),U32.to_nat(prefix),Addition.zero_right(U32.to_nat(prefix)))
    case Con{+last,Nil{}} False{}: Equal.sym(Nat,Nat.add(U32.to_nat(prefix),0n),U32.to_nat(prefix),Addition.zero_right(U32.to_nat(prefix)))
    case Con{+axis,Con{+next,+tail}} False{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},Laws.strides_con(axis,Con{next,tail})) :
        {U32.to_nat(units_prefix_of(Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target))) == Nat.add(U32.to_nat(prefix),walked(Con{axis,Con{next,tail}},False{},target)) : Nat}
      +later = {Con{next,tail} : List<&2,U32>}
      +start = U32.to_nat(prefix)
      +steps = walked(later,U32.is_le(Grid.product(later),target),target)
      +step = next_index(prefix,start,List.length(&2,U32,later),cap,cap_ok,{==},room)
      +room_next = {%Equal.sym(Nat,U32.to_nat((prefix + 1 : U32)),1n+start,step) : {Nat.is_le(Nat.add(_,List.length(&2,U32,later)),U32.to_nat(cap)) == True{} : Bool}
        Order.advance_room(start,List.length(&2,U32,later),U32.to_nat(cap),room) :
        {Nat.is_le(Nat.add(U32.to_nat((prefix + 1 : U32)),List.length(&2,U32,later)),U32.to_nat(cap)) == True{} : Bool}}
      Equal.trans(Nat,U32.to_nat(units_prefix_of(Grid.units_from(later,Grid.strides(later),U32.is_le(Grid.product(later),target),(prefix + 1 : U32),(outer * extent : U32),axis,target))),
        Nat.add(U32.to_nat((prefix + 1 : U32)),steps),Nat.add(start,1n+steps),
        walk_prefix(later,U32.is_le(Grid.product(later),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,cap,cap_ok,room_next),
        Equal.trans(Nat,Nat.add(U32.to_nat((prefix + 1 : U32)),steps),Nat.add(1n+start,steps),Nat.add(start,1n+steps),
          Equal.cong(Nat,Nat,value => Nat.add(value,steps),U32.to_nat((prefix + 1 : U32)),1n+start,step),
          Equal.sym(Nat,Nat.add(start,1n+steps),1n+Nat.add(start,steps),Addition.successor_right(start,steps))))

def walk_outer(+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Con{extent,extents})),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(outer),ShapeLaws.elements(Con{extent,extents}))) == True{} : Bool}) ->
  {U32.to_nat(units_outer_of(Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target))) == Nat.mul(U32.to_nat(outer),lead(Con{extent,extents},walked(extents,fits,target))) : Nat}:
  match extents fits:
    case +extents True{}:
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},
          Laws.units_fit(extents,prefix,outer,extent,target)) :
        {U32.to_nat(units_outer_of(_)) == Nat.mul(U32.to_nat(outer),lead(Con{extent,extents},walked(extents,True{},target))) : Nat}
      Equal.sym(Nat,Nat.mul(U32.to_nat(outer),1n),U32.to_nat(outer),Order.times_one(U32.to_nat(outer)))
    case Nil{} False{}: Equal.sym(Nat,Nat.mul(U32.to_nat(outer),1n),U32.to_nat(outer),Order.times_one(U32.to_nat(outer)))
    case Con{+last,Nil{}} False{}: Equal.sym(Nat,Nat.mul(U32.to_nat(outer),1n),U32.to_nat(outer),Order.times_one(U32.to_nat(outer)))
    case Con{+axis,Con{+next,+tail}} False{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},Laws.strides_con(axis,Con{next,tail})) :
        {U32.to_nat(units_outer_of(Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target))) == Nat.mul(U32.to_nat(outer),lead(Con{extent,Con{axis,Con{next,tail}}},walked(Con{axis,Con{next,tail}},False{},target))) : Nat}
      +later = {Con{next,tail} : List<&2,U32>}
      +steps = walked(later,U32.is_le(Grid.product(later),target),target)
      +rest_cells = ShapeLaws.elements(Con{axis,later})
      +product = Nat.mul(U32.to_nat(outer),U32.to_nat(extent))
      +whole = Nat.mul(U32.to_nat(extent),rest_cells)
      +regroup = Equal.sym(Nat,Nat.mul(product,rest_cells),Nat.mul(U32.to_nat(outer),whole),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),rest_cells))
      +grouped_room = {%regroup : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.mul(product,rest_cells),U32.to_nat(cap)) == True{} : Bool}}
      +grouped_positive = {%regroup : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(product,rest_cells)) == True{} : Bool}}
      +rest_positive = DivisionOrder.right_factor_positive(product,rest_cells,grouped_positive)
      +exact = Arithmetic.product_exact(cap,cap_ok,outer,extent,U32.to_nat(outer),U32.to_nat(extent),{==},{==},
        Order.transitive(product,Nat.mul(product,rest_cells),U32.to_nat(cap),RowMajor.product_covers(product,rest_cells,rest_positive),grouped_room))
      +room_next = {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),product,exact) : {Nat.is_le(Nat.mul(_,rest_cells),U32.to_nat(cap)) == True{} : Bool} grouped_room :
        {Nat.is_le(Nat.mul(U32.to_nat((outer * extent : U32)),rest_cells),U32.to_nat(cap)) == True{} : Bool}}
      +positive_next = {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),product,exact) : {Nat.is_lt(0n,Nat.mul(_,rest_cells)) == True{} : Bool} grouped_positive :
        {Nat.is_lt(0n,Nat.mul(U32.to_nat((outer * extent : U32)),rest_cells)) == True{} : Bool}}
      +ahead = lead(Con{axis,later},steps)
      Equal.trans(Nat,U32.to_nat(units_outer_of(Grid.units_from(later,Grid.strides(later),U32.is_le(Grid.product(later),target),(prefix + 1 : U32),(outer * extent : U32),axis,target))),
        Nat.mul(U32.to_nat((outer * extent : U32)),ahead),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),ahead)),
        walk_outer(later,U32.is_le(Grid.product(later),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,cap,cap_ok,room_next,positive_next),
        Equal.trans(Nat,Nat.mul(U32.to_nat((outer * extent : U32)),ahead),Nat.mul(product,ahead),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),ahead)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,ahead),U32.to_nat((outer * extent : U32)),product,exact),
          RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),ahead)))

def walk_extent(+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32) ->
  {U32.to_nat(units_extent_of(Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target))) == prefix_extent(Con{extent,extents},walked(extents,fits,target)) : Nat}:
  match extents fits:
    case +extents True{}:
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},
          Laws.units_fit(extents,prefix,outer,extent,target)) :
        {U32.to_nat(units_extent_of(_)) == prefix_extent(Con{extent,extents},walked(extents,True{},target)) : Nat}
      {==}
    case Nil{} False{}: {==}
    case Con{+last,Nil{}} False{}: {==}
    case Con{+axis,Con{+next,+tail}} False{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},Laws.strides_con(axis,Con{next,tail})) :
        {U32.to_nat(units_extent_of(Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target))) == prefix_extent(Con{extent,Con{axis,Con{next,tail}}},walked(Con{axis,Con{next,tail}},False{},target)) : Nat}
      walk_extent(Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target)

def walk_unit(+extents: List<&2,U32>,+fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target))) == unit_of(Con{extent,extents},walked(extents,fits,target)) : Nat}:
  match extents fits:
    case +extents True{}:
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},
          Laws.units_fit(extents,prefix,outer,extent,target)) :
        {U32.to_nat(GridRanges.unit_cells_of(_)) == unit_of(Con{extent,extents},walked(extents,True{},target)) : Nat}
      TiledPublic.product_exact(cap,cap_ok,extents,room)
    case Nil{} False{}: {==}
    case Con{+last,Nil{}} False{}: Equal.sym(Nat,Nat.mul(U32.to_nat(last),1n),U32.to_nat(last),Order.times_one(U32.to_nat(last)))
    case Con{+axis,Con{+next,+tail}} False{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},Laws.strides_con(axis,Con{next,tail})) :
        {U32.to_nat(GridRanges.unit_cells_of(Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target))) == unit_of(Con{extent,Con{axis,Con{next,tail}}},walked(Con{axis,Con{next,tail}},False{},target)) : Nat}
      +later = {Con{next,tail} : List<&2,U32>}
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(later),positive)
      walk_unit(later,U32.is_le(Grid.product(later),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,cap,cap_ok,
        Order.transitive(ShapeLaws.elements(later),ShapeLaws.elements(Con{axis,later}),U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(axis),ShapeLaws.elements(later),axis_positive),room),
        DivisionOrder.right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(later),positive))

# ---- The box of a piece over the cell axes ----

def box_rest(+extents: List<&2,U32>,+axis: U32,+prefix: U32,+units: U32,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(axis) == position : Nat},+past: {Nat.is_le(U32.to_nat(prefix),position) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool}) -> {Grid.box(extents,axis,prefix,units) == extents : List<&2,U32>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      +next = next_index(axis,position,List.length(&2,U32,rest),cap,cap_ok,exact,room)
      +bigger = Order.successor(U32.to_nat(prefix),position,past)
      +lt_nat = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == False{} : Bool}
        Order.not_below(U32.to_nat(prefix),1n+position,bigger) : {Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}}
      +lt_false = Equal.trans(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.lt((axis + 1 : U32),prefix),lt_nat)
      +eq_nat = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_eq(_,U32.to_nat(prefix)) == False{} : Bool}
        Equal.trans(Bool,Nat.is_eq(1n+position,U32.to_nat(prefix)),Nat.is_eq(U32.to_nat(prefix),1n+position),False{},Order.equal_symmetric(1n+position,U32.to_nat(prefix)),
          Order.less_not_equal(U32.to_nat(prefix),1n+position,Order.lt_succ(U32.to_nat(prefix),position,past))) :
        {Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}}
      +eq_false = Equal.trans(Bool,U32.is_eq((axis + 1 : U32),prefix),Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.eq((axis + 1 : U32),prefix),eq_nat)
      %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lt_false) :
        {Con{Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),units,extent)),Grid.box(rest,(axis + 1 : U32),prefix,units)} == Con{extent,rest} : List<&2,U32>}
      %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),False{},eq_false) :
        {Con{Bool.pick(U32,_,units,extent),Grid.box(rest,(axis + 1 : U32),prefix,units)} == Con{extent,rest} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => Con{extent,later},Grid.box(rest,(axis + 1 : U32),prefix,units),rest,
        box_rest(rest,(axis + 1 : U32),prefix,units,1n+position,cap,cap_ok,next,bigger,Order.advance_room(position,List.length(&2,U32,rest),U32.to_nat(cap),room)))

def box_at(+extents: List<&2,U32>,+axis: U32,+prefix: U32,+units: U32,+before: Nat,+position: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(axis) == position : Nat},+at: {U32.to_nat(prefix) == 1n+Nat.add(position,before) : Nat},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Grid.box(extents,axis,prefix,units) == box_from(extents,before,units) : List<&2,U32>}:
  match extents before:
    case Nil{} +before: {==}
    case Con{+extent,+rest} 0n:
      +next = next_index(axis,position,List.length(&2,U32,rest),cap,cap_ok,exact,room)
      +here = Equal.trans(Nat,U32.to_nat(prefix),1n+Nat.add(position,0n),1n+position,at,Equal.cong(Nat,Nat,value => 1n+value,Nat.add(position,0n),position,Addition.zero_right(position)))
      +lt_nat = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(prefix),1n+position,here) : {Nat.is_lt(1n+position,_) == False{} : Bool}
        Order.not_below(1n+position,1n+position,Order.reflexive(1n+position)) : {Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == False{} : Bool}}
      +lt_false = Equal.trans(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),False{},Comparison.lt((axis + 1 : U32),prefix),lt_nat)
      +eq_nat = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_eq(_,U32.to_nat(prefix)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(prefix),1n+position,here) : {Nat.is_eq(1n+position,_) == True{} : Bool}
        Order.equal_reflexive(1n+position) : {Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == True{} : Bool}}
      +eq_true = Equal.trans(Bool,U32.is_eq((axis + 1 : U32),prefix),Nat.is_eq(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),True{},Comparison.eq((axis + 1 : U32),prefix),eq_nat)
      +past = {%Equal.sym(Nat,U32.to_nat(prefix),1n+position,here) : {Nat.is_le(_,1n+position) == True{} : Bool} Order.reflexive(1n+position) :
        {Nat.is_le(U32.to_nat(prefix),1n+position) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),False{},lt_false) :
        {Con{Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),units,extent)),Grid.box(rest,(axis + 1 : U32),prefix,units)} == Con{units,rest} : List<&2,U32>}
      %Equal.sym(Bool,U32.is_eq((axis + 1 : U32),prefix),True{},eq_true) :
        {Con{Bool.pick(U32,_,units,extent),Grid.box(rest,(axis + 1 : U32),prefix,units)} == Con{units,rest} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => Con{units,later},Grid.box(rest,(axis + 1 : U32),prefix,units),rest,
        box_rest(rest,(axis + 1 : U32),prefix,units,1n+position,cap,cap_ok,next,past,Order.advance_room(position,List.length(&2,U32,rest),U32.to_nat(cap),room)))
    case Con{+extent,+rest} 1n+ +later:
      +next = next_index(axis,position,List.length(&2,U32,rest),cap,cap_ok,exact,room)
      +lt_nat = {%Equal.sym(Nat,U32.to_nat((axis + 1 : U32)),1n+position,next) : {Nat.is_lt(_,U32.to_nat(prefix)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(prefix),1n+Nat.add(position,1n+later),at) : {Nat.is_lt(1n+position,_) == True{} : Bool}
        Order.below_sum(position,later) : {Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)) == True{} : Bool}}
      +lt_true = Equal.trans(Bool,U32.is_lt((axis + 1 : U32),prefix),Nat.is_lt(U32.to_nat((axis + 1 : U32)),U32.to_nat(prefix)),True{},Comparison.lt((axis + 1 : U32),prefix),lt_nat)
      +at_next = Equal.trans(Nat,U32.to_nat(prefix),1n+Nat.add(position,1n+later),1n+Nat.add(1n+position,later),at,
        Equal.cong(Nat,Nat,value => 1n+value,Nat.add(position,1n+later),1n+Nat.add(position,later),Addition.successor_right(position,later)))
      %Equal.sym(Bool,U32.is_lt((axis + 1 : U32),prefix),True{},lt_true) :
        {Con{Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((axis + 1 : U32),prefix),units,extent)),Grid.box(rest,(axis + 1 : U32),prefix,units)} ==
          Con{1,box_from(rest,later,units)} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Con{1,tail},Grid.box(rest,(axis + 1 : U32),prefix,units),box_from(rest,later,units),
        box_at(rest,(axis + 1 : U32),prefix,units,later,1n+position,cap,cap_ok,next,at_next,Order.advance_room(position,List.length(&2,U32,rest),U32.to_nat(cap),room)))

# ---- A piece of a grid is its box ----

# A leading axis of one index and step zero changes no sum below its cells.
def fake_axis(+extent: U32,+extents: List<&2,U32>,+steps: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {stepped_sum(Con{extent,extents},Con{0,steps},index) == stepped_sum(extents,steps,index) : U32}:
  %Equal.sym(Nat & Nat,Nat.divmod(index,ShapeLaws.elements(extents)),(0n,index),Division.small(index,ShapeLaws.elements(extents),inside)) :
    {(U32.from_nat(Pair.fst(Nat,Nat,_)) * 0 + stepped_sum(extents,steps,Pair.snd(Nat,Nat,_)) : U32) == stepped_sum(extents,steps,index) : U32}
  %Equal.sym(U32,(0 * 0 : U32),0,Arithmetic.multiply_zero_left(0)) : {(_ + stepped_sum(extents,steps,index) : U32) == stepped_sum(extents,steps,index) : U32}
  Arithmetic.add_zero_left(stepped_sum(extents,steps,index))

def walked_le(+extents: List<&2,U32>,+fits: Bool,+target: U32) -> {Nat.is_le(walked(extents,fits,target),List.length(&2,U32,extents)) == True{} : Bool}:
  match extents fits:
    case +extents True{}: Order.zero_le(List.length(&2,U32,extents))
    case Nil{} False{}: {==}
    case Con{+last,Nil{}} False{}: {==}
    case Con{+axis,Con{+next,+tail}} False{}: walked_le(Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),target)

def doubled_zero(+value: Nat) -> {Nat.add(Nat.add(value,0n),0n) == value : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(value,0n),0n),Nat.add(value,0n),value,Addition.zero_right(Nat.add(value,0n)),Addition.zero_right(value))

# The box of the walk over the cells behind a leading axis of one index is the
# box of the grid.
def box_part(+walk: Nat,+extents: List<&2,U32>,+steps: List<&2,U32>,+prefix: U32,+units: U32,+inner: Nat,+index: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+prefix_nat: {U32.to_nat(prefix) == walk : Nat},
  +length_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(box_from(Con{1,extents},walk,units))) == True{} : Bool},
  +room: {Nat.is_lt(Nat.add(Nat.mul(inner,unit_of(Con{1,extents},walk)),index),Nat.mul(prefix_extent(Con{1,extents},walk),unit_of(Con{1,extents},walk))) == True{} : Bool}) ->
  {stepped_sum(box_from(Con{1,extents},walk,units),Con{0,steps},index) == stepped_sum(Grid.box(extents,0,prefix,units),steps,index) : U32}:
  match walk:
    case 0n:
      +cells = ShapeLaws.elements(extents)
      +below = {%Addition.zero_right(cells) : {Nat.is_lt(index,_) == True{} : Bool}
        Order.le_lt_trans(index,Nat.add(Nat.mul(inner,cells),index),Nat.add(cells,0n),Order.right_le(Nat.mul(inner,cells),index),room) :
        {Nat.is_lt(index,cells) == True{} : Bool}}
      +past = {%Equal.sym(Nat,U32.to_nat(prefix),0n,prefix_nat) : {Nat.is_le(_,0n) == True{} : Bool} Order.reflexive(0n) : {Nat.is_le(U32.to_nat(prefix),0n) == True{} : Bool}}
      Equal.trans(U32,stepped_sum(Con{units,extents},Con{0,steps},index),stepped_sum(extents,steps,index),stepped_sum(Grid.box(extents,0,prefix,units),steps,index),
        fake_axis(units,extents,steps,index,below),
        Equal.cong(List<&2,U32>,U32,boxes => stepped_sum(boxes,steps,index),extents,Grid.box(extents,0,prefix,units),
          Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),extents,box_rest(extents,0,prefix,units,0n,cap,cap_ok,{==},past,length_room))))
    case 1n+ +before:
      +boxes = box_from(extents,before,units)
      +below = {%Addition.zero_right(ShapeLaws.elements(boxes)) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,ShapeLaws.elements(boxes)) == True{} : Bool}}
      Equal.trans(U32,stepped_sum(Con{1,boxes},Con{0,steps},index),stepped_sum(boxes,steps,index),stepped_sum(Grid.box(extents,0,prefix,units),steps,index),
        fake_axis(1,boxes,steps,index,below),
        Equal.cong(List<&2,U32>,U32,found => stepped_sum(found,steps,index),boxes,Grid.box(extents,0,prefix,units),
          Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),boxes,box_at(extents,0,prefix,units,before,0n,cap,cap_ok,{==},prefix_nat,length_room))))

# The shift of a piece over its box, and the box inside the cells.
def PieceBox(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat) -> Data:
  Held.Both<{(stepped_sum(boxes,steps,index) + stepped_sum(extents,steps,start) : U32) == stepped_sum(extents,steps,Nat.add(start,index)) : U32},
    Held.Both<{Nat.is_lt(index,ShapeLaws.elements(boxes)) == True{} : Bool},
      Held.Both<{Nat.is_le(ShapeLaws.elements(boxes),ShapeLaws.elements(extents)) == True{} : Bool},
        Held.Both<{Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool},{ShapeLaws.elements(boxes) == cells : Nat}>>>>

# The box of a piece holds as many cells as the piece.
def piece_box_cells(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,
  +boxed: PieceBox(extents,steps,boxes,start,cells,index)) -> {ShapeLaws.elements(boxes) == cells : Nat}:
  Held.Both{+shift,Held.Both{+index_box,Held.Both{+box_le,Held.Both{+ending,+box_cells}}}} = boxed
  box_cells

def box_size(+walk: Nat,+extents: List<&2,U32>,+prefix: U32,+units: U32,+index: Nat,+span: Nat,+start: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +prefix_nat: {U32.to_nat(prefix) == walk : Nat},+length_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(cap)) == True{} : Bool},
  +cells: {ShapeLaws.elements(box_from(Con{1,extents},walk,units)) == span : Nat},+span_le: {Nat.is_le(span,ShapeLaws.elements(extents)) == True{} : Bool},
  +index_in: {Nat.is_lt(index,span) == True{} : Bool},+ending: {Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Held.Both<{Nat.is_lt(index,ShapeLaws.elements(Grid.box(extents,0,prefix,units))) == True{} : Bool},
    Held.Both<{Nat.is_le(ShapeLaws.elements(Grid.box(extents,0,prefix,units)),ShapeLaws.elements(extents)) == True{} : Bool},
      Held.Both<{Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool},{ShapeLaws.elements(Grid.box(extents,0,prefix,units)) == span : Nat}>>>:
  match walk:
    case 0n:
      +past = {%Equal.sym(Nat,U32.to_nat(prefix),0n,prefix_nat) : {Nat.is_le(_,0n) == True{} : Bool} Order.reflexive(0n) : {Nat.is_le(U32.to_nat(prefix),0n) == True{} : Bool}}
      +whole = ShapeLaws.elements(extents)
      +units_count = U32.to_nat(units)
      +span_positive = Order.positive_bound(index,span,index_in)
      +product_positive = {%Equal.sym(Nat,Nat.mul(units_count,whole),span,cells) : {Nat.is_lt(0n,_) == True{} : Bool} span_positive : {Nat.is_lt(0n,Nat.mul(units_count,whole)) == True{} : Bool}}
      +covered = {%RowMajor.multiply_commutative(whole,units_count) : {Nat.is_le(whole,_) == True{} : Bool}
        RowMajor.product_covers(whole,units_count,RowMajor.left_factor_positive(units_count,whole,product_positive)) :
        {Nat.is_le(whole,Nat.mul(units_count,whole)) == True{} : Bool}}
      +whole_span = Order.antisymmetric(whole,span,{%cells : {Nat.is_le(whole,_) == True{} : Bool} covered : {Nat.is_le(whole,span) == True{} : Bool}},span_le)
      %Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),extents,box_rest(extents,0,prefix,units,0n,cap,cap_ok,{==},past,length_room)) :
        Held.Both<{Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool},
          Held.Both<{Nat.is_le(ShapeLaws.elements(_),ShapeLaws.elements(extents)) == True{} : Bool},
            Held.Both<{Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool},{ShapeLaws.elements(_) == span : Nat}>>>
      Held.Both{lt_le(index,span,ShapeLaws.elements(extents),index_in,span_le),Held.Both{Order.reflexive(ShapeLaws.elements(extents)),Held.Both{ending,whole_span}}}
    case 1n+ +before:
      +boxes = box_from(extents,before,units)
      +exact = Equal.trans(Nat,ShapeLaws.elements(boxes),Nat.add(ShapeLaws.elements(boxes),0n),span,
        Equal.sym(Nat,Nat.add(ShapeLaws.elements(boxes),0n),ShapeLaws.elements(boxes),Addition.zero_right(ShapeLaws.elements(boxes))),cells)
      %Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),boxes,box_at(extents,0,prefix,units,before,0n,cap,cap_ok,{==},prefix_nat,length_room)) :
        Held.Both<{Nat.is_lt(index,ShapeLaws.elements(_)) == True{} : Bool},
          Held.Both<{Nat.is_le(ShapeLaws.elements(_),ShapeLaws.elements(extents)) == True{} : Bool},
            Held.Both<{Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool},{ShapeLaws.elements(_) == span : Nat}>>>
      %Equal.sym(Nat,ShapeLaws.elements(boxes),span,exact) :
        Held.Both<{Nat.is_lt(index,_) == True{} : Bool},
          Held.Both<{Nat.is_le(_,ShapeLaws.elements(extents)) == True{} : Bool},
            Held.Both<{Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool},{_ == span : Nat}>>>
      Held.Both{index_in,Held.Both{span_le,Held.Both{ending,{==}}}}

# The fields of the grid of a shape, walked over a leading axis of one index:
# its prefix axes, the cells of the axes before the last of them, that last
# extent and the cells of one of its indices.
def field_prefix(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(prefix) == walked(extents,U32.is_le(Grid.product(extents),target),target) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +fits = U32.is_le(Grid.product(extents),target)
  +found = Grid.units_from(extents,Grid.strides(extents),fits,0,1,1,target)
  +walk = walked(extents,fits,target)
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  Equal.trans(Nat,U32.to_nat(prefix),U32.to_nat(units_prefix_of(found)),walk,
    Equal.cong(U32,Nat,word => U32.to_nat(word),prefix,units_prefix_of(found),
      Equal.trans(U32,prefix,grid_prefix(Grid.grid(extents,target)),units_prefix_of(found),Equal.cong(Grid.Grid,U32,value => grid_prefix(value),grid,Grid.grid(extents,target),back),
        Equal.trans(U32,grid_prefix(Grid.grid(extents,target)),grid_prefix(Grid.grid_of_units(found,target)),units_prefix_of(found),
          Equal.cong(Grid.Grid,U32,value => grid_prefix(value),Grid.grid(extents,target),Grid.grid_of_units(found,target),Laws.grid_units(extents,target)),prefix_kept(found,target)))),
    walk_prefix(extents,fits,0,1,1,target,cap,cap_ok,rank_room))

def field_outer(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {U32.to_nat(outer) == lead(Con{1,extents},walked(extents,U32.is_le(Grid.product(extents),target),target)) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +fits = U32.is_le(Grid.product(extents),target)
  +found = Grid.units_from(extents,Grid.strides(extents),fits,0,1,1,target)
  +walk = walked(extents,fits,target)
  +wide = {Con{1,extents} : List<&2,U32>}
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  Equal.trans(Nat,U32.to_nat(outer),Nat.mul(U32.to_nat(1),lead(wide,walk)),lead(wide,walk),
    Equal.trans(Nat,U32.to_nat(outer),U32.to_nat(units_outer_of(found)),Nat.mul(U32.to_nat(1),lead(wide,walk)),
      Equal.cong(U32,Nat,word => U32.to_nat(word),outer,units_outer_of(found),
        Equal.trans(U32,outer,grid_outer(Grid.grid(extents,target)),units_outer_of(found),Equal.cong(Grid.Grid,U32,value => grid_outer(value),grid,Grid.grid(extents,target),back),
          Equal.trans(U32,grid_outer(Grid.grid(extents,target)),grid_outer(Grid.grid_of_units(found,target)),units_outer_of(found),
          Equal.cong(Grid.Grid,U32,value => grid_outer(value),Grid.grid(extents,target),Grid.grid_of_units(found,target),Laws.grid_units(extents,target)),outer_kept(found,target)))),
      walk_outer(extents,fits,0,1,1,target,cap,cap_ok,
        {%Equal.sym(Nat,Nat.add(Nat.add(ShapeLaws.elements(extents),0n),0n),ShapeLaws.elements(extents),doubled_zero(ShapeLaws.elements(extents))) :
            {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
          {Nat.is_le(Nat.mul(U32.to_nat(1),ShapeLaws.elements(wide)),U32.to_nat(cap)) == True{} : Bool}},
        {%Equal.sym(Nat,Nat.add(Nat.add(ShapeLaws.elements(extents),0n),0n),ShapeLaws.elements(extents),doubled_zero(ShapeLaws.elements(extents))) :
            {Nat.is_lt(0n,_) == True{} : Bool} nonempty :
          {Nat.is_lt(0n,Nat.mul(U32.to_nat(1),ShapeLaws.elements(wide))) == True{} : Bool}})),
    Addition.zero_right(lead(wide,walk)))

def field_extent(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}) ->
  {U32.to_nat(extent) == prefix_extent(Con{1,extents},walked(extents,U32.is_le(Grid.product(extents),target),target)) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +fits = U32.is_le(Grid.product(extents),target)
  +found = Grid.units_from(extents,Grid.strides(extents),fits,0,1,1,target)
  +walk = walked(extents,fits,target)
  +wide = {Con{1,extents} : List<&2,U32>}
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  Equal.trans(Nat,U32.to_nat(extent),U32.to_nat(units_extent_of(found)),prefix_extent(wide,walk),
    Equal.cong(U32,Nat,word => U32.to_nat(word),extent,units_extent_of(found),
      Equal.trans(U32,extent,grid_extent(Grid.grid(extents,target)),units_extent_of(found),Equal.cong(Grid.Grid,U32,value => grid_extent(value),grid,Grid.grid(extents,target),back),
        Equal.trans(U32,grid_extent(Grid.grid(extents,target)),grid_extent(Grid.grid_of_units(found,target)),units_extent_of(found),
          Equal.cong(Grid.Grid,U32,value => grid_extent(value),Grid.grid(extents,target),Grid.grid_of_units(found,target),Laws.grid_units(extents,target)),extent_kept(found,target)))),
    walk_extent(extents,fits,0,1,1,target))

def field_unit(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {U32.to_nat(unit_cells) == unit_of(Con{1,extents},walked(extents,U32.is_le(Grid.product(extents),target),target)) : Nat}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +fits = U32.is_le(Grid.product(extents),target)
  +found = Grid.units_from(extents,Grid.strides(extents),fits,0,1,1,target)
  +walk = walked(extents,fits,target)
  +wide = {Con{1,extents} : List<&2,U32>}
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  Equal.trans(Nat,U32.to_nat(unit_cells),U32.to_nat(GridRanges.unit_cells_of(found)),unit_of(wide,walk),
    Equal.cong(U32,Nat,word => U32.to_nat(word),unit_cells,GridRanges.unit_cells_of(found),
      Equal.trans(U32,unit_cells,GridRanges.grid_unit_cells(Grid.grid(extents,target)),GridRanges.unit_cells_of(found),
        Equal.cong(Grid.Grid,U32,value => GridRanges.grid_unit_cells(value),grid,Grid.grid(extents,target),back),Equal.trans(U32,GridRanges.grid_unit_cells(Grid.grid(extents,target)),GridRanges.grid_unit_cells(Grid.grid_of_units(found,target)),GridRanges.unit_cells_of(found),
          Equal.cong(Grid.Grid,U32,value => GridRanges.grid_unit_cells(value),Grid.grid(extents,target),Grid.grid_of_units(found,target),Laws.grid_units(extents,target)),
          GridRanges.unit_cells_kept(found,target)))),
    walk_unit(extents,fits,0,1,1,target,cap,cap_ok,room,nonempty))

# Cell first + index of a piece: the steps over the cell coordinates of first
# plus the steps over the box coordinates of index.
def piece_box(+limits: Machine.Limits,+extents: List<&2,U32>,+steps: List<&2,U32>,+target: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))) == True{} : Bool}) ->
  PieceBox(extents,steps,Grid.box_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),
    U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),
    U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),index):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +fits = U32.is_le(Grid.product(extents),target)
  +walk = walked(extents,fits,target)
  +wide = {Con{1,extents} : List<&2,U32>}
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  +prefix_nat = field_prefix(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same,rank_room)
  +outer_nat = field_outer(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same,room,nonempty)
  +extent_nat = field_extent(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same)
  +unit_nat = field_unit(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same,room,nonempty)
  +per_count = U32.to_nat(per)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +lead_units = Nat.mul(inner,U32.to_nat(block))
  +span = Grid.span(grid,U32.from_nat(piece))
  +units = (span / U32.max(unit_cells,1) : U32)
  +whole = prefix_extent(wide,walk)
  +unit = unit_of(wide,walk)
  +outer_in = {%outer_nat : {Nat.is_lt(quotient,_) == True{} : Bool}
    DivisionOrder.quotient_below(piece,U32.to_nat(outer),per_count,Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape),inside) :
    {Nat.is_lt(quotient,lead(wide,walk)) == True{} : Bool}}
  +room_units = {%unit_nat : {Nat.is_lt(Nat.add(Nat.mul(lead_units,_),index),Nat.mul(U32.to_nat(extent),_)) == True{} : Bool}
    piece_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,index,index_in) :
    {Nat.is_lt(Nat.add(Nat.mul(lead_units,unit),index),Nat.mul(U32.to_nat(extent),unit)) == True{} : Bool}}
  +room_box = {%extent_nat : {Nat.is_lt(Nat.add(Nat.mul(lead_units,unit),index),Nat.mul(_,unit)) == True{} : Bool} room_units :
    {Nat.is_lt(Nat.add(Nat.mul(lead_units,unit),index),Nat.mul(whole,unit)) == True{} : Bool}}
  +difference = Nat.sub(piece_base(next_outer(Nat.is_lt(1n+inner,per_count),quotient),next_inner(Nat.is_lt(1n+inner,per_count),inner),U32.to_nat(extent),U32.to_nat(block)),
    piece_base(quotient,inner,U32.to_nat(extent),U32.to_nat(block)))
  +box_cells = Equal.trans(Nat,ShapeLaws.elements(box_from(wide,walk,units)),Nat.mul(U32.to_nat(units),unit),U32.to_nat(span),
    box_elements(wide,walk,units,Order.lt_succ(walk,List.length(&2,U32,extents),walked_le(extents,fits,target))),
    Equal.trans(Nat,Nat.mul(U32.to_nat(units),unit),Nat.mul(difference,U32.to_nat(unit_cells)),U32.to_nat(span),
      {%Equal.sym(Nat,U32.to_nat(units),difference,piece_units(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
          {Nat.mul(_,unit) == Nat.mul(difference,U32.to_nat(unit_cells)) : Nat}
        %Equal.sym(Nat,U32.to_nat(unit_cells),unit,unit_nat) : {Nat.mul(difference,unit) == Nat.mul(difference,_) : Nat}
        {==} : {Nat.mul(U32.to_nat(units),unit) == Nat.mul(difference,U32.to_nat(unit_cells)) : Nat}},
      Equal.sym(Nat,U32.to_nat(span),Nat.mul(difference,U32.to_nat(unit_cells)),piece_span(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside))))
  +inside_box = {%Equal.sym(Nat,ShapeLaws.elements(box_from(wide,walk,units)),U32.to_nat(span),box_cells) : {Nat.is_lt(index,_) == True{} : Bool} index_in :
    {Nat.is_lt(index,ShapeLaws.elements(box_from(wide,walk,units))) == True{} : Bool}}
  +first = first_of(wide,walk,quotient,lead_units)
  +start = U32.to_nat(Grid.start(grid,U32.from_nat(piece)))
  +start_exact = Equal.trans(Nat,start,Nat.mul(piece_base(quotient,inner,U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)),first,
    start_base(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside),
    Equal.trans(Nat,Nat.mul(piece_base(quotient,inner,U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(unit_cells)),
      Nat.add(Nat.mul(Nat.mul(quotient,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(lead_units,U32.to_nat(unit_cells))),first,
      RowMajor.distrib(Nat.mul(quotient,U32.to_nat(extent)),lead_units,U32.to_nat(unit_cells)),
      {%Equal.sym(Nat,Nat.mul(Nat.mul(quotient,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(quotient,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
            RowMajor.assoc(quotient,U32.to_nat(extent),U32.to_nat(unit_cells))) :
          {Nat.add(_,Nat.mul(lead_units,U32.to_nat(unit_cells))) == first : Nat}
        %extent_nat : {Nat.add(Nat.mul(quotient,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(lead_units,U32.to_nat(unit_cells))) ==
          Nat.add(Nat.mul(quotient,Nat.mul(_,unit)),Nat.mul(lead_units,unit)) : Nat}
        %unit_nat : {Nat.add(Nat.mul(quotient,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(lead_units,U32.to_nat(unit_cells))) ==
          Nat.add(Nat.mul(quotient,Nat.mul(U32.to_nat(extent),_)),Nat.mul(lead_units,_)) : Nat}
        {==} : {Nat.add(Nat.mul(Nat.mul(quotient,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(lead_units,U32.to_nat(unit_cells))) == first : Nat}}))
  +total = ShapeLaws.elements(extents)
  +count_nat = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(extents,target))),total,
    Equal.cong(Grid.Grid,Nat,value => U32.to_nat(Grid.count_of(value)),grid,Grid.grid(extents,target),back),GridRanges.grid_count(limits,extents,target,room,nonempty))
  +next_inside = Order.strict_follow(piece,Nat.mul(U32.to_nat(outer),per_count),inside)
  +next_back = Bounds.roundtrip(1n+piece,cap,cap_ok,Order.transitive(1n+piece,Nat.mul(U32.to_nat(outer),per_count),U32.to_nat(cap),next_inside,
    Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +next_start = U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece)))
  +next_end = {%count_nat : {Nat.is_le(next_start,_) == True{} : Bool}
    GridRanges.start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(1n+piece),
      {%Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next_back) : {Nat.is_le(_,Nat.mul(U32.to_nat(outer),per_count)) == True{} : Bool} next_inside :
        {Nat.is_le(U32.to_nat(U32.from_nat(1n+piece)),Nat.mul(U32.to_nat(outer),per_count)) == True{} : Bool}}) :
    {Nat.is_le(next_start,total) == True{} : Bool}}
  +end_here = lt_le(Nat.add(start,index),Nat.add(start,U32.to_nat(span)),total,Order.offset_lt(start,index,U32.to_nat(span),index_in),
    {%Equal.sym(Nat,Nat.add(start,U32.to_nat(span)),next_start,GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
        {Nat.is_le(_,total) == True{} : Bool} next_end :
      {Nat.is_le(Nat.add(start,U32.to_nat(span)),total) == True{} : Bool}})
  +end_start = {%start_exact : {Nat.is_lt(Nat.add(_,index),total) == True{} : Bool} end_here : {Nat.is_lt(Nat.add(first,index),total) == True{} : Bool}}
  +first_in = Order.le_lt_trans(first,Nat.add(first,index),total,Order.le_add(first,index),end_start)
  +span_le = Order.transitive(U32.to_nat(span),Nat.add(start,U32.to_nat(span)),total,Order.right_le(start,U32.to_nat(span)),
    {%Equal.sym(Nat,Nat.add(start,U32.to_nat(span)),next_start,GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
        {Nat.is_le(_,total) == True{} : Bool} next_end : {Nat.is_le(Nat.add(start,U32.to_nat(span)),total) == True{} : Bool}})
  Held.Both{{
      %Equal.sym(Nat,start,first,start_exact) :
        {(stepped_sum(Grid.box(extents,0,prefix,units),steps,index) + stepped_sum(extents,steps,_) : U32) == stepped_sum(extents,steps,Nat.add(_,index)) : U32}
      %fake_axis(1,extents,steps,first,first_in) :
        {(stepped_sum(Grid.box(extents,0,prefix,units),steps,index) + _ : U32) == stepped_sum(extents,steps,Nat.add(first,index)) : U32}
      %fake_axis(1,extents,steps,Nat.add(first,index),end_start) :
        {(stepped_sum(Grid.box(extents,0,prefix,units),steps,index) + stepped_sum(wide,Con{0,steps},first) : U32) == _ : U32}
      %box_part(walk,extents,steps,prefix,units,lead_units,index,cap,cap_ok,prefix_nat,rank_room,inside_box,room_box) :
        {(_ + stepped_sum(wide,Con{0,steps},first) : U32) == stepped_sum(wide,Con{0,steps},Nat.add(first,index)) : U32}
      box_shift(wide,Con{0,steps},walk,units,quotient,lead_units,index,outer_in,room_box,inside_box)
    : {(stepped_sum(Grid.box(extents,0,prefix,units),steps,index) + stepped_sum(extents,steps,start) : U32) == stepped_sum(extents,steps,Nat.add(start,index)) : U32}},
    box_size(walk,extents,prefix,units,index,U32.to_nat(span),start,cap,cap_ok,prefix_nat,rank_room,box_cells,span_le,index_in,end_here)}

def piece_box_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+steps: List<&2,U32>,+target: U32,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>,
  +same: {grid == Grid.grid(extents,target) : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(grid,piece))) == True{} : Bool}) ->
  PieceBox(extents,steps,Grid.box_of(grid,extents,Grid.span(grid,piece)),U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece)),index):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +flat = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +inside_flat = {%exact : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(_))) == True{} : Bool} inside :
    {Nat.is_lt(U32.to_nat(piece),U32.to_nat((outer * per : U32))) == True{} : Bool}}
  +inside_nat = {%Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool} inside_flat :
    {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
  +index_flat = {%exact : {Nat.is_lt(index,U32.to_nat(Grid.span(_,piece))) == True{} : Bool} index_in :
    {Nat.is_lt(index,U32.to_nat(Grid.span(flat,piece))) == True{} : Bool}}
  +index_word = {%Equal.sym(U32,U32.from_nat(U32.to_nat(piece)),piece,Arithmetic.native_roundtrip(piece)) : {Nat.is_lt(index,U32.to_nat(Grid.span(flat,_))) == True{} : Bool}
    index_flat : {Nat.is_lt(index,U32.to_nat(Grid.span(flat,U32.from_nat(U32.to_nat(piece))))) == True{} : Bool}}
  %Equal.sym(Grid.Grid,grid,flat,exact) :
    PieceBox(extents,steps,Grid.box_of(_,extents,Grid.span(_,piece)),U32.to_nat(Grid.start(_,piece)),U32.to_nat(Grid.span(_,piece)),index)
  %Arithmetic.native_roundtrip(piece) :
    PieceBox(extents,steps,Grid.box_of(flat,extents,Grid.span(flat,_)),U32.to_nat(Grid.start(flat,_)),U32.to_nat(Grid.span(flat,_)),index)
  piece_box(limits,extents,steps,target,outer,extent,unit_cells,block,per,count,prefix,shape,cells,
    Equal.trans(Grid.Grid,Grid.grid(extents,target),grid,flat,Equal.sym(Grid.Grid,grid,Grid.grid(extents,target),same),exact),room,nonempty,rank_room,
    U32.to_nat(piece),inside_nat,index,index_word)

# A piece of the grid of a shape of cells inside the machine is its box.
def grid_piece_box(+limits: Machine.Limits,+extents: List<&2,U32>,+steps: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool}) ->
  PieceBox(extents,steps,Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece)),
    U32.to_nat(Grid.start(Grid.grid(extents,target),piece)),U32.to_nat(Grid.span(Grid.grid(extents,target),piece)),index):
  piece_box_facts(limits,extents,steps,target,Grid.grid(extents,target),GridRanges.grid_facts(limits,extents,target,room,nonempty),{==},room,nonempty,rank_room,
    piece,inside,index,index_in)

# ---- Every axis of a piece box spans no more than its axis ----

# Every axis of a box spans no more than its axis.
def BoxWithin(boxes: List<&2,U32>,extents: List<&2,U32>) -> Data:
  match boxes extents:
    case Con{+box,boxes} Con{+extent,extents}:
      Held.Both<{Nat.is_le(U32.to_nat(box),U32.to_nat(extent)) == True{} : Bool},BoxWithin(boxes,extents)>
    case boxes extents:
      Unit

# An axis of a box spans more than one index only where its axis does.

def wider_one(+box: U32,+extent: U32,+fits: {Nat.is_le(U32.to_nat(box),U32.to_nat(extent)) == True{} : Bool}) ->
  Logic.When(U32.is_lt(1,box),{U32.is_lt(1,extent) == True{} : Bool}):
  Held.by_cases(Logic.When(U32.is_lt(1,box),{U32.is_lt(1,extent) == True{} : Bool}),U32.is_lt(1,box),
    +wide => {%Equal.sym(Bool,U32.is_lt(1,box),True{},wide) : Logic.When(_,{U32.is_lt(1,extent) == True{} : Bool})
      %Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(U32.to_nat(1),U32.to_nat(extent)),Comparison.lt(1,extent)) : {_ == True{} : Bool}
      lt_le(U32.to_nat(1),U32.to_nat(box),U32.to_nat(extent),{%Comparison.lt(1,box) : {_ == True{} : Bool} wide : {Nat.is_lt(U32.to_nat(1),U32.to_nat(box)) == True{} : Bool}},fits) :
      Logic.When(U32.is_lt(1,box),{U32.is_lt(1,extent) == True{} : Bool})},
    +narrow => {%Equal.sym(Bool,U32.is_lt(1,box),False{},narrow) : Logic.When(_,{U32.is_lt(1,extent) == True{} : Bool}) Unit{} :
      Logic.When(U32.is_lt(1,box),{U32.is_lt(1,extent) == True{} : Bool})})

def within_same(extents: List<&2,U32>) -> BoxWithin(extents,extents):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}: Held.Both{Order.reflexive(U32.to_nat(extent)),within_same(rest)}

def within_from(extents: List<&2,U32>,before: Nat,+units: U32,+fits: {Nat.is_le(U32.to_nat(units),prefix_extent(extents,before)) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  BoxWithin(box_from(extents,before,units),extents):
  match extents before:
    case Nil{} +before: Unit{}
    case Con{+extent,+rest} 0n: Held.Both{fits,within_same(rest)}
    case Con{+extent,+rest} 1n+ +later:
      +size = U32.to_nat(extent)
      Held.Both{Order.strict_follow(0n,size,RowMajor.left_factor_positive(size,ShapeLaws.elements(rest),nonempty)),
        within_from(rest,later,units,fits,DivisionOrder.right_factor_positive(size,ShapeLaws.elements(rest),nonempty))}

# The units of a piece stay inside its axis.
def units_within(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(piece_base(next_outer(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.div(piece,U32.to_nat(per))),
        next_inner(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
      piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block))),U32.to_nat(extent)) == True{} : Bool}:
  +per_count = U32.to_nat(per)
  +choice = Nat.is_lt(1n+Nat.mod(piece,per_count),per_count)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +extent_count = U32.to_nat(extent)
  +lead_units = Nat.mul(inner,U32.to_nat(block))
  +base = piece_base(quotient,inner,extent_count,U32.to_nat(block))
  +next = piece_base(next_outer(choice,quotient),next_inner(choice,inner),extent_count,U32.to_nat(block))
  +difference = Nat.sub(next,base)
  +positive = Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape)
  +per_exact = Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape)
  +ordered = base_before_next(choice,quotient,inner,per_count,extent_count,U32.to_nat(block),positive,per_exact,
    DivisionOrder.remainder_small(piece,per_count,Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)),{==})
  +ending = next_before_end(choice,quotient,inner,per_count,extent_count,U32.to_nat(block),positive,per_exact,{==})
  +regrouped = Equal.trans(Nat,next,Nat.add(base,difference),Nat.add(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference)),
    Equal.sym(Nat,Nat.add(base,difference),next,Order.restore_shift(base,next,ordered)),Addition.associative(Nat.mul(quotient,extent_count),lead_units,difference))
  +fits = Order.cancel_offset_le(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference),extent_count,
    {%regrouped : {Nat.is_le(_,Nat.add(Nat.mul(quotient,extent_count),extent_count)) == True{} : Bool} ending :
      {Nat.is_le(Nat.add(Nat.mul(quotient,extent_count),Nat.add(lead_units,difference)),Nat.add(Nat.mul(quotient,extent_count),extent_count)) == True{} : Bool}})
  Order.transitive(difference,Nat.add(lead_units,difference),extent_count,Order.right_le(lead_units,difference),fits)

def within_none(+limits: Machine.Limits,+extents: List<&2,U32>,+prefix: U32,+units: U32,
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+none: {U32.is_zero(prefix) == True{} : Bool}) ->
  BoxWithin(Grid.box(extents,0,prefix,units),extents):
  +past = {%Equal.sym(U32,prefix,0,Arithmetic.word_zero(prefix,none)) : {Nat.is_le(U32.to_nat(_),0n) == True{} : Bool} {==} :
    {Nat.is_le(U32.to_nat(prefix),0n) == True{} : Bool}}
  %Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),extents,
      box_rest(extents,0,prefix,units,0n,Machine.capacity(limits),LimitBounds.capacity_fits(limits),{==},past,rank_room)) :
    BoxWithin(_,extents)
  within_same(extents)

def within_some(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+some: {U32.is_zero(prefix) == False{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  BoxWithin(Grid.box(extents,0,prefix,(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)) / U32.max(unit_cells,1) : U32)),extents):
  +units = (Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)) / U32.max(unit_cells,1) : U32)
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +distance = Order.positive_has_last(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some))
  +walk = walked(extents,U32.is_le(Grid.product(extents),target),target)
  +prefix_nat = field_prefix(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same,rank_room)
  +walk_steps = Equal.trans(Nat,walk,U32.to_nat(prefix),1n+steps,Equal.sym(Nat,U32.to_nat(prefix),walk,prefix_nat),Equal.sym(Nat,Nat.add(0n,1n+steps),U32.to_nat(prefix),distance))
  +extent_eq = Equal.trans(Nat,U32.to_nat(extent),prefix_extent(Con{1,extents},walk),prefix_extent(extents,steps),
    field_extent(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,same),
    Equal.cong(Nat,Nat,found => prefix_extent(Con{1,extents},found),walk,1n+steps,walk_steps))
  +difference = Nat.sub(piece_base(next_outer(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.div(piece,U32.to_nat(per))),
      next_inner(Nat.is_lt(1n+Nat.mod(piece,U32.to_nat(per)),U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per))),U32.to_nat(extent),U32.to_nat(block)),
    piece_base(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),U32.to_nat(extent),U32.to_nat(block)))
  +fits = {%extent_eq : {Nat.is_le(U32.to_nat(units),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(units),difference,piece_units(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
      {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
    units_within(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside) : {Nat.is_le(U32.to_nat(units),prefix_extent(extents,steps)) == True{} : Bool}}
  %Equal.sym(List<&2,U32>,Grid.box(extents,0,prefix,units),box_from(extents,steps,units),
      box_at(extents,0,prefix,units,steps,0n,Machine.capacity(limits),LimitBounds.capacity_fits(limits),{==},Equal.sym(Nat,Nat.add(0n,1n+steps),U32.to_nat(prefix),distance),
        rank_room)) :
    BoxWithin(_,extents)
  within_from(extents,steps,units,fits,nonempty)

def piece_within(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  BoxWithin(Grid.box_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),extents):
  Held.by_cases(BoxWithin(Grid.box_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),extents),U32.is_zero(prefix),
    +none => within_none(limits,extents,prefix,(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)) / U32.max(unit_cells,1) : U32),rank_room,none),
    +some => within_some(limits,extents,target,outer,extent,unit_cells,block,per,count,prefix,shape,cells,same,rank_room,piece,inside,some,nonempty))

def grid_within_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>,
  +same: {grid == Grid.grid(extents,target) : Grid.Grid},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  BoxWithin(Grid.box_of(grid,extents,Grid.span(grid,piece)),extents):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +inside_flat = {%exact : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(_))) == True{} : Bool} inside :
    {Nat.is_lt(U32.to_nat(piece),U32.to_nat((outer * per : U32))) == True{} : Bool}}
  +inside_nat = {%Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool} inside_flat :
    {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : BoxWithin(Grid.box_of(_,extents,Grid.span(_,piece)),extents)
  %Arithmetic.native_roundtrip(piece) : BoxWithin(Grid.box_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)),extents)
  piece_within(limits,extents,target,outer,extent,unit_cells,block,per,count,prefix,shape,cells,
    Equal.trans(Grid.Grid,Grid.grid(extents,target),grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Equal.sym(Grid.Grid,grid,Grid.grid(extents,target),same),exact),rank_room,U32.to_nat(piece),inside_nat,nonempty)

# Every axis of a piece box of the grid of a shape spans no more than its
# axis of the shape.
def grid_within(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  BoxWithin(Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece)),extents):
  grid_within_facts(limits,extents,target,Grid.grid(extents,target),GridRanges.grid_facts(limits,extents,target,room,nonempty),{==},rank_room,piece,inside,nonempty)

# ---- A piece holds no more cells than its grid ----

def span_within(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +count: U32,+prefix: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +same: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),ShapeLaws.elements(extents)) == True{} : Bool}:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +per_count = U32.to_nat(per)
  +span = Grid.span(grid,U32.from_nat(piece))
  +start = U32.to_nat(Grid.start(grid,U32.from_nat(piece)))
  +total = ShapeLaws.elements(extents)
  +back = Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,same)
  +count_nat = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(extents,target))),total,
    Equal.cong(Grid.Grid,Nat,value => U32.to_nat(Grid.count_of(value)),grid,Grid.grid(extents,target),back),GridRanges.grid_count(limits,extents,target,room,nonempty))
  +next_inside = Order.strict_follow(piece,Nat.mul(U32.to_nat(outer),per_count),inside)
  +next_back = Bounds.roundtrip(1n+piece,cap,cap_ok,Order.transitive(1n+piece,Nat.mul(U32.to_nat(outer),per_count),U32.to_nat(cap),next_inside,
    Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +next_start = U32.to_nat(Grid.start(grid,U32.from_nat(1n+piece)))
  +next_end = {%count_nat : {Nat.is_le(next_start,_) == True{} : Bool}
    GridRanges.start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(1n+piece),
      {%Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+piece)),1n+piece,next_back) : {Nat.is_le(_,Nat.mul(U32.to_nat(outer),per_count)) == True{} : Bool} next_inside :
        {Nat.is_le(U32.to_nat(U32.from_nat(1n+piece)),Nat.mul(U32.to_nat(outer),per_count)) == True{} : Bool}}) :
    {Nat.is_le(next_start,total) == True{} : Bool}}
  Order.transitive(U32.to_nat(span),Nat.add(start,U32.to_nat(span)),total,Order.right_le(start,U32.to_nat(span)),
    {%Equal.sym(Nat,Nat.add(start,U32.to_nat(span)),next_start,GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
        {Nat.is_le(_,total) == True{} : Bool} next_end : {Nat.is_le(Nat.add(start,U32.to_nat(span)),total) == True{} : Bool}})

def span_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>,
  +same: {grid == Grid.grid(extents,target) : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.span(grid,piece)),ShapeLaws.elements(extents)) == True{} : Bool}:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +inside_flat = {%exact : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(_))) == True{} : Bool} inside :
    {Nat.is_lt(U32.to_nat(piece),U32.to_nat((outer * per : U32))) == True{} : Bool}}
  +inside_nat = {%Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool} inside_flat :
    {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : {Nat.is_le(U32.to_nat(Grid.span(_,piece)),ShapeLaws.elements(extents)) == True{} : Bool}
  %Arithmetic.native_roundtrip(piece) : {Nat.is_le(U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)),ShapeLaws.elements(extents)) == True{} : Bool}
  span_within(limits,extents,target,outer,extent,unit_cells,block,per,count,prefix,shape,cells,
    Equal.trans(Grid.Grid,Grid.grid(extents,target),grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Equal.sym(Grid.Grid,grid,Grid.grid(extents,target),same),exact),room,nonempty,U32.to_nat(piece),inside_nat)

# A piece of the grid of a shape holds no more cells than the shape.
def grid_span_room(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.span(Grid.grid(extents,target),piece)),ShapeLaws.elements(extents)) == True{} : Bool}:
  span_facts(limits,extents,target,Grid.grid(extents,target),GridRanges.grid_facts(limits,extents,target,room,nonempty),{==},room,nonempty,piece,inside)

# A cell of a piece box lies inside the shape.
def box_end_of(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,
  +boxed: PieceBox(extents,steps,boxes,start,cells,index)) -> {Nat.is_lt(Nat.add(start,index),ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  ending

# A nonempty piece of a grid ends inside its cells.
def piece_end(+limits: Machine.Limits,+cells: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(cells,Grid.piece_target())))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece))) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(cells,Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece))),ShapeLaws.elements(cells)) == True{} : Bool}:
  +span = U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece))
  +begin = U32.to_nat(Grid.start(Grid.grid(cells,Grid.piece_target()),piece))
  +last = Nat.sub(span,1n)
  +whole = Order.restore_shift(1n,span,Order.strict_follow(0n,span,positive))
  +last_in = {%whole : {Nat.is_lt(last,_) == True{} : Bool} Order.lt_succ(last,last,Order.reflexive(last)) : {Nat.is_lt(last,span) == True{} : Bool}}
  +steps = Grid.strides(cells)
  +boxed = grid_piece_box(limits,cells,steps,Grid.piece_target(),room,nonempty,rank_room,piece,inside,last,last_in)
  +ending = box_end_of(cells,steps,Grid.box_of(Grid.grid(cells,Grid.piece_target()),cells,Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),begin,span,last,boxed)
  %whole : {Nat.is_le(Nat.add(begin,_),ShapeLaws.elements(cells)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(begin,1n+last),1n+Nat.add(begin,last),Addition.successor_right(begin,last)) : {Nat.is_le(_,ShapeLaws.elements(cells)) == True{} : Bool}
  Order.strict_follow(Nat.add(begin,last),ShapeLaws.elements(cells),ending)

# The cells of a shape with its groups placed are the groups times the cells
# of the other axes.
def placed_elements(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)) : Nat}:
  match position:
    case Grid.Innermost{}:
      Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,others,[groups])),Nat.mul(ShapeLaws.elements(others),Nat.mul(U32.to_nat(groups),1n)),
        Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)),ShapeLaws.elements_appended(others,[groups]),
        {%Equal.sym(Nat,Nat.mul(U32.to_nat(groups),1n),U32.to_nat(groups),Order.times_one(U32.to_nat(groups))) :
            {Nat.mul(ShapeLaws.elements(others),_) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)) : Nat}
          RowMajor.multiply_commutative(ShapeLaws.elements(others),U32.to_nat(groups)) :
          {Nat.mul(ShapeLaws.elements(others),Nat.mul(U32.to_nat(groups),1n)) == Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)) : Nat}})
    case Grid.Outermost{}: {==}
