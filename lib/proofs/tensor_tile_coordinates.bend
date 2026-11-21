# The cells of a tiled tensor by coordinates: the cell at the digits of its
# other axes and its group holds, in lane l, the value at the logical digits
# with 8 * group + l put back at the lane axis, and +0 past the lanes.

import Base
import ../tensor_grid.bend as Grid
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ../kernel_gemm.bend as KernelGemm
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_representation.bend as Representation
import ../tensor_tiled_cells.bend as Cells
import ../kernel_tile.bend as Tile
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_tiled_values.bend as TiledValues
import ./nat_ceiling.bend as Ceiling
import ./list_storage_readback.bend as ListReadback

# ---- Digits ----

# Every digit lies below its extent.
def DigitsBelow(extents: List<&2,U32>,digits: List<&2,Nat>) -> Data:
  match extents digits:
    case Con{+extent,extents} Con{+digit,digits}: KernelGemm.Both<{Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},DigitsBelow(extents,digits)>
    case Nil{} Nil{}: Unit
    case extents digits: Empty

# Digits below their extents number less than the cells.
def digits_below(extents: List<&2,U32>,digits: List<&2,Nat>,+below: DigitsBelow(extents,digits)) ->
  {Nat.is_lt(ShapeLaws.digits_value(extents,digits),ShapeLaws.elements(extents)) == True{} : Bool}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      RowMajor.radix(digit,U32.to_nat(extent),ShapeLaws.digits_value(rest,more),ShapeLaws.elements(rest),here,digits_below(rest,more,later))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({Nat.is_lt(0n,1n) == True{} : Bool},below)
    case Con{+extent,+rest} Nil{}: Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(Con{extent,rest})) == True{} : Bool},below)

def digits_positive(+extents: List<&2,U32>,+digits: List<&2,Nat>,+below: DigitsBelow(extents,digits)) -> {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}:
  Order.le_lt_trans(0n,ShapeLaws.digits_value(extents,digits),ShapeLaws.elements(extents),Order.zero_le(ShapeLaws.digits_value(extents,digits)),digits_below(extents,digits,below))

# The number of digits after digits.
def digits_appended(extents: List<&2,U32>,digits: List<&2,Nat>,+more_extents: List<&2,U32>,+more: List<&2,Nat>,+below: DigitsBelow(extents,digits)) ->
  {ShapeLaws.digits_value(List.append(&2,U32,extents,more_extents),List.append(&2,Nat,digits,more)) ==
    Nat.add(Nat.mul(ShapeLaws.digits_value(extents,digits),ShapeLaws.elements(more_extents)),ShapeLaws.digits_value(more_extents,more)) : Nat}:
  match extents digits:
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Con{+digit,+tail}:
      KernelGemm.Both{here,+later} = below
      +cells = ShapeLaws.elements(rest)
      +size = ShapeLaws.elements(more_extents)
      +value = ShapeLaws.digits_value(rest,tail)
      +moved = ShapeLaws.digits_value(more_extents,more)
      %Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,rest,more_extents)),Nat.mul(cells,size),ShapeLaws.elements_appended(rest,more_extents)) :
        {Nat.add(Nat.mul(digit,_),ShapeLaws.digits_value(List.append(&2,U32,rest,more_extents),List.append(&2,Nat,tail,more))) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,cells),value),size),moved) : Nat}
      %Equal.sym(Nat,ShapeLaws.digits_value(List.append(&2,U32,rest,more_extents),List.append(&2,Nat,tail,more)),Nat.add(Nat.mul(value,size),moved),digits_appended(rest,tail,more_extents,more,later)) :
        {Nat.add(Nat.mul(digit,Nat.mul(cells,size)),_) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,cells),value),size),moved) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(digit,cells),value),size),Nat.add(Nat.mul(Nat.mul(digit,cells),size),Nat.mul(value,size)),RowMajor.distrib(Nat.mul(digit,cells),value,size)) :
        {Nat.add(Nat.mul(digit,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),moved)) == Nat.add(_,moved) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(digit,cells),size),Nat.mul(value,size)),moved),
          Nat.add(Nat.mul(Nat.mul(digit,cells),size),Nat.add(Nat.mul(value,size),moved)),Addition.associative(Nat.mul(Nat.mul(digit,cells),size),Nat.mul(value,size),moved)) :
        {Nat.add(Nat.mul(digit,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),moved)) == _ : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.mul(digit,cells),size),Nat.mul(digit,Nat.mul(cells,size)),RowMajor.assoc(digit,cells,size)) :
        {Nat.add(Nat.mul(digit,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),moved)) == Nat.add(_,Nat.add(Nat.mul(value,size),moved)) : Nat}
      {==}
    case Nil{} Con{+digit,+tail}:
      Empty.absurd({ShapeLaws.digits_value(List.append(&2,U32,[],more_extents),List.append(&2,Nat,digit <> tail,more)) ==
        Nat.add(Nat.mul(ShapeLaws.digits_value([],digit <> tail),ShapeLaws.elements(more_extents)),ShapeLaws.digits_value(more_extents,more)) : Nat},below)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({ShapeLaws.digits_value(List.append(&2,U32,extent <> rest,more_extents),List.append(&2,Nat,[],more)) ==
        Nat.add(Nat.mul(ShapeLaws.digits_value(extent <> rest,[]),ShapeLaws.elements(more_extents)),ShapeLaws.digits_value(more_extents,more)) : Nat},below)

# Digits below their extents, after digits below theirs.
def digits_joined(extents: List<&2,U32>,digits: List<&2,Nat>,+more_extents: List<&2,U32>,+more: List<&2,Nat>,+below: DigitsBelow(extents,digits),+more_below: DigitsBelow(more_extents,more)) ->
  DigitsBelow(List.append(&2,U32,extents,more_extents),List.append(&2,Nat,digits,more)):
  match extents digits:
    case Nil{} Nil{}: more_below
    case Con{+extent,+rest} Con{+digit,+tail}:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,digits_joined(rest,tail,more_extents,more,later,more_below)}
    case Nil{} Con{+digit,+tail}: Empty.absurd(DigitsBelow(List.append(&2,U32,[],more_extents),List.append(&2,Nat,digit <> tail,more)),below)
    case Con{+extent,+rest} Nil{}: Empty.absurd(DigitsBelow(List.append(&2,U32,extent <> rest,more_extents),List.append(&2,Nat,[],more)),below)

# The digits with value put at an axis, after the digits when the axis lies
# past them.
def inserted_digit(axis: Nat,digits: List<&2,Nat>,+value: Nat) -> List<&2,Nat>:
  match axis:
    case 0n: Con{value,digits}
    case 1n+next:
      match digits:
        case Nil{}: Con{value,Nil{}}
        case Con{digit,rest}: Con{digit,inserted_digit(next,rest,value)}

# The cells of the axes before an axis.
def before_cells(extents: List<&2,U32>,axis: Nat) -> Nat:
  match extents axis:
    case Nil{} axis: 1n
    case Con{extent,rest} 0n: 1n
    case Con{+extent,rest} 1n+next: Nat.mul(U32.to_nat(extent),before_cells(rest,next))

# ---- The logical number of a channel ----

# The logical number of channel channel at an outer index: the outer index
# names the values before the lane axis and those after it.
def channel_number(+outer: Nat,+channel: Nat,+lanes: Nat,+after: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,after),lanes),channel),after),Nat.mod(outer,after))

# An outer index whole steps of the values after the lane axis further holds
# the channel as many steps of the lanes further.
def channel_shift(+steps: Nat,+outer: Nat,+channel: Nat,+lanes: Nat,+after: Nat,+positive: {Nat.is_lt(0n,after) == True{} : Bool}) ->
  {channel_number(Nat.add(Nat.mul(steps,after),outer),channel,lanes,after) ==
    Nat.add(Nat.mul(Nat.mul(steps,lanes),after),channel_number(outer,channel,lanes,after)) : Nat}:
  +quotient = Nat.div(outer,after)
  +remainder = Nat.mod(outer,after)
  +inner = Nat.add(Nat.mul(quotient,lanes),channel)
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(steps,after),outer),after),Nat.add(steps,quotient),DivisionOrder.quotient_shift(outer,after,steps,positive)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,lanes),channel),after),Nat.mod(Nat.add(Nat.mul(steps,after),outer),after)) ==
      Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.add(Nat.mul(inner,after),remainder)) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(steps,after),outer),after),remainder,DivisionOrder.remainder_shift(outer,after,steps,positive)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(steps,quotient),lanes),channel),after),_) ==
      Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.add(Nat.mul(inner,after),remainder)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(steps,quotient),lanes),Nat.add(Nat.mul(steps,lanes),Nat.mul(quotient,lanes)),RowMajor.distrib(steps,quotient,lanes)) :
    {Nat.add(Nat.mul(Nat.add(_,channel),after),remainder) == Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.add(Nat.mul(inner,after),remainder)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(steps,lanes),Nat.mul(quotient,lanes)),channel),Nat.add(Nat.mul(steps,lanes),inner),
      Addition.associative(Nat.mul(steps,lanes),Nat.mul(quotient,lanes),channel)) :
    {Nat.add(Nat.mul(_,after),remainder) == Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.add(Nat.mul(inner,after),remainder)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(steps,lanes),inner),after),Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.mul(inner,after)),
      RowMajor.distrib(Nat.mul(steps,lanes),inner,after)) :
    {Nat.add(_,remainder) == Nat.add(Nat.mul(Nat.mul(steps,lanes),after),Nat.add(Nat.mul(inner,after),remainder)) : Nat}
  Addition.associative(Nat.mul(Nat.mul(steps,lanes),after),Nat.mul(inner,after),remainder)

# The cells of the axes but one are those before it times those after it.
def removed_cells(extents: List<&2,U32>,axis: Nat) ->
  {ShapeLaws.elements(Grid.removed(extents,axis)) == Nat.mul(before_cells(extents,axis),ShapeLaws.elements(Public.lane_after(extents,axis))) : Nat}:
  match extents axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n: Equal.sym(Nat,Nat.add(ShapeLaws.elements(rest),0n),ShapeLaws.elements(rest),Addition.zero_right(ShapeLaws.elements(rest)))
    case Con{+extent,+rest} 1n+ +next:
      Equal.trans(Nat,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Grid.removed(rest,next))),
        Nat.mul(U32.to_nat(extent),Nat.mul(before_cells(rest,next),ShapeLaws.elements(Public.lane_after(rest,next)))),
        Nat.mul(Nat.mul(U32.to_nat(extent),before_cells(rest,next)),ShapeLaws.elements(Public.lane_after(rest,next))),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),ShapeLaws.elements(Grid.removed(rest,next)),
          Nat.mul(before_cells(rest,next),ShapeLaws.elements(Public.lane_after(rest,next))),removed_cells(rest,next)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),before_cells(rest,next)),ShapeLaws.elements(Public.lane_after(rest,next))),
          Nat.mul(U32.to_nat(extent),Nat.mul(before_cells(rest,next),ShapeLaws.elements(Public.lane_after(rest,next)))),
          RowMajor.assoc(U32.to_nat(extent),before_cells(rest,next),ShapeLaws.elements(Public.lane_after(rest,next)))))

# The cells of the axes are those before the lane axis, its lanes and those
# after it.
def lane_cells(extents: List<&2,U32>,axis: Nat) ->
  {ShapeLaws.elements(extents) ==
    Nat.mul(Nat.mul(before_cells(extents,axis),U32.to_nat(Grid.lanes_at(extents,axis))),ShapeLaws.elements(Public.lane_after(extents,axis))) : Nat}:
  match extents axis:
    case Nil{} axis: {==}
    case Con{+extent,+rest} 0n:
      Equal.cong(Nat,Nat,value => Nat.mul(value,ShapeLaws.elements(rest)),U32.to_nat(extent),Nat.add(U32.to_nat(extent),0n),
        Equal.sym(Nat,Nat.add(U32.to_nat(extent),0n),U32.to_nat(extent),Addition.zero_right(U32.to_nat(extent))))
    case Con{+extent,+rest} 1n+ +next:
      +lanes = U32.to_nat(Grid.lanes_at(rest,next))
      +after = ShapeLaws.elements(Public.lane_after(rest,next))
      +head = U32.to_nat(extent)
      +before = before_cells(rest,next)
      Equal.trans(Nat,Nat.mul(head,ShapeLaws.elements(rest)),Nat.mul(head,Nat.mul(Nat.mul(before,lanes),after)),Nat.mul(Nat.mul(Nat.mul(head,before),lanes),after),
        Equal.cong(Nat,Nat,value => Nat.mul(head,value),ShapeLaws.elements(rest),Nat.mul(Nat.mul(before,lanes),after),lane_cells(rest,next)),
        Equal.trans(Nat,Nat.mul(head,Nat.mul(Nat.mul(before,lanes),after)),Nat.mul(Nat.mul(head,Nat.mul(before,lanes)),after),Nat.mul(Nat.mul(Nat.mul(head,before),lanes),after),
          Equal.sym(Nat,Nat.mul(Nat.mul(head,Nat.mul(before,lanes)),after),Nat.mul(head,Nat.mul(Nat.mul(before,lanes),after)),RowMajor.assoc(head,Nat.mul(before,lanes),after)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,after),Nat.mul(head,Nat.mul(before,lanes)),Nat.mul(Nat.mul(head,before),lanes),
            Equal.sym(Nat,Nat.mul(Nat.mul(head,before),lanes),Nat.mul(head,Nat.mul(before,lanes)),RowMajor.assoc(head,before,lanes)))))

# The logical number of a channel at the outer index of the digits of the
# axes but the lane axis is the number of the digits with the channel put
# back at the lane axis.
def channel_digits(+extents: List<&2,U32>,axis: Nat,+others: List<&2,Nat>,+channel: Nat,+below: DigitsBelow(Grid.removed(extents,axis),others),
  +inside: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(extents,axis))) == True{} : Bool}) ->
  {channel_number(ShapeLaws.digits_value(Grid.removed(extents,axis),others),channel,U32.to_nat(Grid.lanes_at(extents,axis)),ShapeLaws.elements(Public.lane_after(extents,axis))) ==
    ShapeLaws.digits_value(extents,inserted_digit(axis,others,channel)) : Nat}:
  match extents axis:
    case Nil{} +axis:
      match others:
        case Nil{}:
          %Equal.sym(Nat,channel,0n,Order.zero_below(channel,Division.lt_succ_le(channel,0n,inside))) :
            {channel_number(0n,_,1n,1n) == ShapeLaws.digits_value(Nil{},inserted_digit(axis,Nil{},_)) : Nat}
          {==}
        case Con{+digit,+more}:
          Empty.absurd({channel_number(ShapeLaws.digits_value(Nil{},Con{digit,more}),channel,1n,1n) ==
            ShapeLaws.digits_value(Nil{},inserted_digit(axis,Con{digit,more},channel)) : Nat},below)
    case Con{+extent,+rest} 0n:
      +outer = ShapeLaws.digits_value(rest,others)
      +after = ShapeLaws.elements(rest)
      %Equal.sym(Nat & Nat,Nat.divmod(outer,after),(0n,outer),Division.small(outer,after,digits_below(rest,others,below))) :
        {Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),U32.to_nat(extent)),channel),after),Pair.snd(Nat,Nat,_)) ==
          Nat.add(Nat.mul(channel,after),outer) : Nat}
      {==}
    case Con{+extent,+rest} 1n+ +next:
      match others:
        case Con{+digit,+more}:
          KernelGemm.Both{+here,+later} = below
          +removed = Grid.removed(rest,next)
          +before = before_cells(rest,next)
          +lanes = U32.to_nat(Grid.lanes_at(rest,next))
          +after = ShapeLaws.elements(Public.lane_after(rest,next))
          +outer = ShapeLaws.digits_value(removed,more)
          +removed_exact = removed_cells(rest,next)
          +positive = DivisionOrder.right_factor_positive(before,after,
            %removed_exact : {Nat.is_lt(0n,_) == True{} : Bool} digits_positive(removed,more,later))
          +inserted = inserted_digit(next,more,channel)
          %Equal.sym(Nat,ShapeLaws.elements(removed),Nat.mul(before,after),removed_exact) :
            {channel_number(Nat.add(Nat.mul(digit,_),outer),channel,lanes,after) == Nat.add(Nat.mul(digit,ShapeLaws.elements(rest)),ShapeLaws.digits_value(rest,inserted)) : Nat}
          %Equal.sym(Nat,ShapeLaws.elements(rest),Nat.mul(Nat.mul(before,lanes),after),lane_cells(rest,next)) :
            {channel_number(Nat.add(Nat.mul(digit,Nat.mul(before,after)),outer),channel,lanes,after) == Nat.add(Nat.mul(digit,_),ShapeLaws.digits_value(rest,inserted)) : Nat}
          %RowMajor.assoc(digit,before,after) :
            {channel_number(Nat.add(_,outer),channel,lanes,after) == Nat.add(Nat.mul(digit,Nat.mul(Nat.mul(before,lanes),after)),ShapeLaws.digits_value(rest,inserted)) : Nat}
          %RowMajor.assoc(digit,Nat.mul(before,lanes),after) :
            {channel_number(Nat.add(Nat.mul(Nat.mul(digit,before),after),outer),channel,lanes,after) == Nat.add(_,ShapeLaws.digits_value(rest,inserted)) : Nat}
          %RowMajor.assoc(digit,before,lanes) :
            {channel_number(Nat.add(Nat.mul(Nat.mul(digit,before),after),outer),channel,lanes,after) == Nat.add(Nat.mul(_,after),ShapeLaws.digits_value(rest,inserted)) : Nat}
          Equal.trans(Nat,channel_number(Nat.add(Nat.mul(Nat.mul(digit,before),after),outer),channel,lanes,after),
            Nat.add(Nat.mul(Nat.mul(Nat.mul(digit,before),lanes),after),channel_number(outer,channel,lanes,after)),
            Nat.add(Nat.mul(Nat.mul(Nat.mul(digit,before),lanes),after),ShapeLaws.digits_value(rest,inserted)),
            channel_shift(Nat.mul(digit,before),outer,channel,lanes,after,positive),
            Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.mul(Nat.mul(digit,before),lanes),after),value),channel_number(outer,channel,lanes,after),
              ShapeLaws.digits_value(rest,inserted),channel_digits(rest,next,more,channel,later,inside)))
        case Nil{}:
          Empty.absurd({channel_number(ShapeLaws.digits_value(Grid.removed(Con{extent,rest},1n+next),Nil{}),channel,U32.to_nat(Grid.lanes_at(rest,next)),
              ShapeLaws.elements(Public.lane_after(rest,next))) ==
            ShapeLaws.digits_value(Con{extent,rest},inserted_digit(1n+next,Nil{},channel)) : Nat},below)

# ---- The cell at the digits of its axes ----

# The digits of a cell: the digits of the other axes and the group, the group
# at its position.
def cell_digits(position: Grid.Position,+others: List<&2,Nat>,+group: Nat) -> List<&2,Nat>:
  match position:
    case Grid.Innermost{}: List.append(&2,Nat,others,[group])
    case Grid.Outermost{}: Con{group,others}

# The number of digits with one more digit after them.
def digits_last(extents: List<&2,U32>,digits: List<&2,Nat>,+extent: U32,+digit: Nat,+below: DigitsBelow(extents,digits)) ->
  {ShapeLaws.digits_value(List.append(&2,U32,extents,[extent]),List.append(&2,Nat,digits,[digit])) ==
    Nat.add(Nat.mul(ShapeLaws.digits_value(extents,digits),U32.to_nat(extent)),digit) : Nat}:
  match extents digits:
    case Nil{} Nil{}:
      Equal.trans(Nat,Nat.add(Nat.mul(digit,1n),0n),Nat.mul(digit,1n),digit,Addition.zero_right(Nat.mul(digit,1n)),Order.times_one(digit))
    case Con{+head,+rest} Con{+first,+more}:
      KernelGemm.Both{+here,+later} = below
      +cells = ShapeLaws.elements(rest)
      +value = ShapeLaws.digits_value(rest,more)
      +size = U32.to_nat(extent)
      %Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,rest,[extent])),Nat.mul(cells,Nat.mul(size,1n)),ShapeLaws.elements_appended(rest,[extent])) :
        {Nat.add(Nat.mul(first,_),ShapeLaws.digits_value(List.append(&2,U32,rest,[extent]),List.append(&2,Nat,more,[digit]))) ==
          Nat.add(Nat.mul(Nat.add(Nat.mul(first,cells),value),size),digit) : Nat}
      %Equal.sym(Nat,Nat.mul(size,1n),size,Order.times_one(size)) :
        {Nat.add(Nat.mul(first,Nat.mul(cells,_)),ShapeLaws.digits_value(List.append(&2,U32,rest,[extent]),List.append(&2,Nat,more,[digit]))) ==
          Nat.add(Nat.mul(Nat.add(Nat.mul(first,cells),value),size),digit) : Nat}
      %Equal.sym(Nat,ShapeLaws.digits_value(List.append(&2,U32,rest,[extent]),List.append(&2,Nat,more,[digit])),Nat.add(Nat.mul(value,size),digit),
          digits_last(rest,more,extent,digit,later)) :
        {Nat.add(Nat.mul(first,Nat.mul(cells,size)),_) == Nat.add(Nat.mul(Nat.add(Nat.mul(first,cells),value),size),digit) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(first,cells),value),size),Nat.add(Nat.mul(Nat.mul(first,cells),size),Nat.mul(value,size)),
          RowMajor.distrib(Nat.mul(first,cells),value,size)) :
        {Nat.add(Nat.mul(first,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),digit)) == Nat.add(_,digit) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(first,cells),size),Nat.mul(value,size)),digit),
          Nat.add(Nat.mul(Nat.mul(first,cells),size),Nat.add(Nat.mul(value,size),digit)),Addition.associative(Nat.mul(Nat.mul(first,cells),size),Nat.mul(value,size),digit)) :
        {Nat.add(Nat.mul(first,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),digit)) == _ : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.mul(first,cells),size),Nat.mul(first,Nat.mul(cells,size)),RowMajor.assoc(first,cells,size)) :
        {Nat.add(Nat.mul(first,Nat.mul(cells,size)),Nat.add(Nat.mul(value,size),digit)) == Nat.add(_,Nat.add(Nat.mul(value,size),digit)) : Nat}
      {==}
    case Nil{} Con{+first,+more}:
      Empty.absurd({ShapeLaws.digits_value(List.append(&2,U32,Nil{},[extent]),List.append(&2,Nat,Con{first,more},[digit])) ==
        Nat.add(Nat.mul(ShapeLaws.digits_value(Nil{},Con{first,more}),U32.to_nat(extent)),digit) : Nat},below)
    case Con{+head,+rest} Nil{}:
      Empty.absurd({ShapeLaws.digits_value(List.append(&2,U32,Con{head,rest},[extent]),List.append(&2,Nat,Nil{},[digit])) ==
        Nat.add(Nat.mul(ShapeLaws.digits_value(Con{head,rest},Nil{}),U32.to_nat(extent)),digit) : Nat},below)

# The number of the digits of a cell is its index from its outer index and
# its group.
def cell_number(position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+extents: List<&2,U32>,+groups: U32,+below: DigitsBelow(extents,others)) ->
  {ShapeLaws.digits_value(Grid.placed(True{},position,extents,groups),cell_digits(position,others,group)) ==
    Representation.index_at(position,ShapeLaws.elements(extents),U32.to_nat(groups),ShapeLaws.digits_value(extents,others),group) : Nat}:
  match position:
    case Grid.Innermost{}: digits_last(extents,others,groups,group,below)
    case Grid.Outermost{}: {==}

# The cells of a counted shape within a capacity number the cells of the
# shape, as words.
def product_word(+limits: Machine.Limits,+extents: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(extents)) == ShapeLaws.elements(extents) : Nat}:
  Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),extents,room)

# The outer count of the geometry of values is the cells of the axes but the
# lane axis.
def outer_word(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)) == ShapeLaws.elements(Grid.removed(values,axis)) : Nat}:
  +removed = Grid.removed(values,axis)
  Equal.trans(Nat,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Grid.product(removed)),
    ShapeLaws.elements(removed),
    Equal.cong(U32,Nat,U32.to_nat,(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Grid.product(removed),
      Equal.trans(U32,(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),(1 * Grid.product(removed) : U32),Grid.product(removed),
        Public.prefix_after(values,axis,1),Public.one_times(Grid.product(removed)))),
    product_word(limits,removed,room))

# The geometry of values with their lane axis inside them.
def geometry_inside(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> Cells.Cells:
  +lanes = Grid.lanes_at(values,axis)
  Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),lanes,Tiled.groups(lanes),
    position}

def geometry_same(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Public.geometry_of(values,axis,position) == geometry_inside(values,axis,position) : Cells.Cells}:
  Public.geometry_exact(values,axis,position,Bounds.strict_le(axis,List.length(&2,U32,values),within))

# The cell at the digits of its other axes and its group holds the cell of
# the reader at that outer index and group.
def cell_lifted(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+word: U32,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),cell_digits(position,others,group)) : Nat}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word) ==
    Representation.cell(~Logical,~original,logical,Public.geometry_of(values,axis,position),ShapeLaws.digits_value(Grid.removed(values,axis),others),group) : Tile.Tile8}:
  +removed = Grid.removed(values,axis)
  +outer = ShapeLaws.digits_value(removed,others)
  +lanes = Grid.lanes_at(values,axis)
  +geometry = geometry_inside(values,axis,position)
  +outer_count = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
  +outer_exact = outer_word(limits,values,axis,room)
  +placed = {%within : {U32.to_nat(word) == ShapeLaws.digits_value(Grid.placed(_,position,removed,Grid.groups(lanes)),cell_digits(position,others,group)) : Nat}
    exact : {U32.to_nat(word) == ShapeLaws.digits_value(Grid.placed(True{},position,removed,Grid.groups(lanes)),cell_digits(position,others,group)) : Nat}}
  +numbered = Equal.trans(Nat,U32.to_nat(word),ShapeLaws.digits_value(Grid.placed(True{},position,removed,Grid.groups(lanes)),cell_digits(position,others,group)),
    Representation.index_at(position,ShapeLaws.elements(removed),U32.to_nat(Grid.groups(lanes)),outer,group),placed,
    cell_number(position,others,group,removed,Grid.groups(lanes),below))
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),geometry,geometry_same(values,axis,position,within)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},word) == Representation.cell(~Logical,~original,logical,_,outer,group) : Tile.Tile8}
  Representation.decoded(~Logical,~original,logical,geometry,outer,group,word,
    {%Equal.sym(Nat,U32.to_nat(outer_count),ShapeLaws.elements(removed),outer_exact) :
        {U32.to_nat(word) == Representation.index_at(position,_,U32.to_nat(Grid.groups(lanes)),outer,group) : Nat}
      numbered : {U32.to_nat(word) == Representation.index_at(position,U32.to_nat(outer_count),U32.to_nat(Grid.groups(lanes)),outer,group) : Nat}},
    {%Equal.sym(Nat,U32.to_nat(outer_count),ShapeLaws.elements(removed),outer_exact) : {Nat.is_lt(outer,_) == True{} : Bool}
      digits_below(removed,others,below) : {Nat.is_lt(outer,U32.to_nat(outer_count)) == True{} : Bool}},
    group_inside)

# A channel inside the lanes reads the digits with the channel put back.
def channel_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,valid: Bool,+values: List<&2,U32>,+axis: Nat,+others: List<&2,Nat>,+channel: Nat,
  +below: DigitsBelow(Grid.removed(values,axis),others),+classified: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool}) ->
  {Bool.pick(F32,valid,original(logical,U32.from_nat(channel_number(ShapeLaws.digits_value(Grid.removed(values,axis),others),channel,U32.to_nat(Grid.lanes_at(values,axis)),
      ShapeLaws.elements(Public.lane_after(values,axis))))),0.0) ==
    Bool.pick(F32,valid,original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,channel)))),0.0) : F32}:
  match valid:
    case True{}:
      Equal.cong(Nat,F32,number => original(logical,U32.from_nat(number)),
        channel_number(ShapeLaws.digits_value(Grid.removed(values,axis),others),channel,U32.to_nat(Grid.lanes_at(values,axis)),ShapeLaws.elements(Public.lane_after(values,axis))),
        ShapeLaws.digits_value(values,inserted_digit(axis,others,channel)),channel_digits(values,axis,others,channel,below,classified))
    case False{}: {==}

# Lane lane of group group at the outer index of the digits of the other axes
# holds the value at the logical digits with 8 * group + lane put back at the
# lane axis, and +0 past the lanes.
def lane_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+lane: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others)) ->
  {Representation.lane(~Logical,~original,logical,Public.geometry_of(values,axis,position),ShapeLaws.digits_value(Grid.removed(values,axis),others),group,lane) ==
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),
      original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32}:
  +removed = Grid.removed(values,axis)
  +outer = ShapeLaws.digits_value(removed,others)
  +after = Public.lane_after(values,axis)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +geometry = geometry_inside(values,axis,position)
  +removed_positive = {%removed_cells(values,axis) : {Nat.is_lt(0n,_) == True{} : Bool}
    digits_positive(removed,others,below) : {Nat.is_lt(0n,Nat.mul(before_cells(values,axis),ShapeLaws.elements(after))) == True{} : Bool}}
  +after_positive = DivisionOrder.right_factor_positive(before_cells(values,axis),ShapeLaws.elements(after),removed_positive)
  +after_room = Order.transitive(ShapeLaws.elements(after),Nat.mul(before_cells(values,axis),ShapeLaws.elements(after)),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.le_multiple(before_cells(values,axis),ShapeLaws.elements(after),RowMajor.left_factor_positive(before_cells(values,axis),ShapeLaws.elements(after),removed_positive)),
    %removed_cells(values,axis) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room)
  +after_exact = Equal.trans(Nat,Representation.least_one(U32.to_nat(Tiled.product(after))),U32.to_nat(Tiled.product(after)),ShapeLaws.elements(after),
    Representation.least_positive(U32.to_nat(Tiled.product(after)),
      %Equal.sym(Nat,U32.to_nat(Tiled.product(after)),ShapeLaws.elements(after),product_word(limits,after,after_room)) : {Nat.is_lt(0n,_) == True{} : Bool} after_positive),
    product_word(limits,after,after_room))
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),geometry,geometry_same(values,axis,position,within)) :
    {Representation.lane(~Logical,~original,logical,_,outer,group,lane) ==
      Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,channel)))),0.0) :
      F32}
  %Equal.sym(Nat,Representation.least_one(U32.to_nat(Tiled.product(after))),ShapeLaws.elements(after),after_exact) :
    {Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),
        original(logical,U32.from_nat(channel_number(outer,channel,U32.to_nat(Grid.lanes_at(values,axis)),_))),0.0) ==
      Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,channel)))),0.0) :
      F32}
  channel_value(~Logical,~original,logical,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),values,axis,others,channel,below,{==})

# Lane lane of the cell at the digits of its other axes and its group holds
# the value at the logical digits with 8 * group + lane put back at the lane
# axis, and +0 past the lanes.
def lane_digits(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +others: List<&2,Nat>,+group: Nat,+word: U32,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),cell_digits(position,others,group)) : Nat},
  +lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),
      original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32}:
  +outer = ShapeLaws.digits_value(Grid.removed(values,axis),others)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)),
    Representation.lane(~Logical,~original,logical,Public.geometry_of(values,axis,position),outer,group,lane),
    Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,inserted_digit(axis,others,channel)))),0.0),
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)),
      TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Public.geometry_of(values,axis,position),outer,group)),
      Representation.lane(~Logical,~original,logical,Public.geometry_of(values,axis,position),outer,group,lane),
      Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(lane,tile),
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word),
        Representation.cell(~Logical,~original,logical,Public.geometry_of(values,axis,position),outer,group),
        cell_lifted(~Logical,~original,logical,limits,values,axis,position,others,group,word,within,room,below,group_inside,exact)),
      Representation.cell_lane(~Logical,~original,logical,Public.geometry_of(values,axis,position),outer,group,lane,small)),
    lane_value(~Logical,~original,logical,limits,values,axis,position,others,group,lane,within,room,below))

# A reader of cells made from values and a geometry agrees with the values
# lifted there on every lane below the lanes of a cell of the values.
def CellsAgree(~Logical: Data,~original: Logical -> U32 -> F32,~Context: Data,~read: Context -> U32 -> Tile.Tile8,~context_at: Logical -> Cells.Cells -> Context) -> Type:
  @+logical: Logical -> @+limits: Machine.Limits -> @+values: List<&2,U32> -> @+axis: Nat -> @+position: Grid.Position -> @+others: List<&2,Nat> -> @+group: Nat -> @+word: U32 ->
  @+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool} -> @+room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ->
  @+below: DigitsBelow(Grid.removed(values,axis),others) -> @+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool} ->
  @+exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),cell_digits(position,others,group)) : Nat} -> @+lane: Nat -> @+small: {Nat.is_le(lane,7n) == True{} : Bool} ->
  @+valid: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == True{} : Bool} ->
  {TileLaws.select_lane(lane,read(context_at(logical,Public.geometry_of(values,axis,position)),word)) ==
    TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) : F32}

# The values lifted agree with themselves.
def lifted_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+word: U32,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),cell_digits(position,others,group)) : Nat},+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +valid: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Public.lifted_at(~Logical,logical,Public.geometry_of(values,axis,position)),word)) ==
    TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) : F32}:
  {==}

# ---- The digits of a cell, the lanes inside or on the unit axis ----

# The digits of the axes but the lane axis and the group, the group at its
# position; the digits alone for lanes on the unit axis.
def placed_digits(lane: Bool,position: Grid.Position,+others: List<&2,Nat>,+group: Nat) -> List<&2,Nat>:
  match lane position:
    case False{} position: others
    case True{} Grid.Innermost{}: List.append(&2,Nat,others,[group])
    case True{} Grid.Outermost{}: group <> others

def placed_cell(position: Grid.Position,+others: List<&2,Nat>,+group: Nat) -> {placed_digits(True{},position,others,group) == cell_digits(position,others,group) : List<&2,Nat>}:
  match position:
    case Grid.Innermost{}: {==}
    case Grid.Outermost{}: {==}

# The cell axes are the other axes and the groups.
def placed_count(position: Grid.Position,+extents: List<&2,U32>,+groups: U32) -> {List.length(&2,U32,Grid.placed(True{},position,extents,groups)) == 1n+List.length(&2,U32,extents) : Nat}:
  match position:
    case Grid.Innermost{}:
      Equal.trans(Nat,List.length(&2,U32,List.append(&2,U32,extents,[groups])),Nat.add(List.length(&2,U32,extents),1n),1n+List.length(&2,U32,extents),
        ListReadback.append_length(~U32,extents,[groups]),Addition.successor_width(List.length(&2,U32,extents)))
    case Grid.Outermost{}: {==}

# Within a group of its own.
def GroupBelow(within: Bool,+group: Nat,+groups: U32) -> Data:
  match within:
    case True{}: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}
    case False{}: Unit

# Digits of the other axes and a group below their extents are the digits of
# a cell below the cell extents.
def below_append(extents: List<&2,U32>,digits: List<&2,Nat>,+last: U32,+digit: Nat,+below: DigitsBelow(extents,digits),
  +last_below: {Nat.is_lt(digit,U32.to_nat(last)) == True{} : Bool}) -> DigitsBelow(List.append(&2,U32,extents,[last]),List.append(&2,Nat,digits,[digit])):
  match extents digits:
    case Nil{} Nil{}: KernelGemm.Both{last_below,Unit{}}
    case Con{+extent,+rest} Con{+first,+more}:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,below_append(rest,more,last,digit,later,last_below)}
    case Nil{} Con{+first,+more}:
      Empty.absurd(DigitsBelow(List.append(&2,U32,Nil{},[last]),List.append(&2,Nat,Con{first,more},[digit])),below)
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(DigitsBelow(List.append(&2,U32,Con{extent,rest},[last]),List.append(&2,Nat,Nil{},[digit])),below)

def placed_below(within: Bool,position: Grid.Position,+others: List<&2,U32>,+groups: U32,+digits: List<&2,Nat>,+group: Nat,
  +below: DigitsBelow(others,digits),+inside: GroupBelow(within,group,groups)) ->
  DigitsBelow(Grid.placed(within,position,others,groups),placed_digits(within,position,digits,group)):
  match within position:
    case False{} +position: below
    case True{} Grid.Innermost{}: below_append(others,digits,groups,group,below,inside)
    case True{} Grid.Outermost{}: KernelGemm.Both{inside,below}

# The logical digits of a channel of a cell: the channel put back at the lane
# axis, or the digits alone for lanes on the unit axis.
def logical_digits(within: Bool,+axis: Nat,+others: List<&2,Nat>,+channel: Nat) -> List<&2,Nat>:
  match within:
    case True{}: inserted_digit(axis,others,channel)
    case False{}: others

def unit_index(position: Grid.Position,+outer: Nat,+word: Nat) -> {word == Representation.index_at(position,outer,1n,word,0n) : Nat}:
  match position:
    case Grid.Innermost{}: Equal.sym(Nat,Nat.add(Nat.mul(word,1n),0n),word,Equal.trans(Nat,Nat.add(Nat.mul(word,1n),0n),Nat.mul(word,1n),word,
      Addition.zero_right(Nat.mul(word,1n)),Order.times_one(word)))
    case Grid.Outermost{}: {==}

def unit_number(+word: Nat) -> {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(word,1n),1n),0n),1n),Nat.mod(word,1n)) == word : Nat}:
  +parts = Equal.trans(Nat & Nat,Nat.divmod(word,1n),Nat.divmod(Nat.add(Nat.mul(word,1n),0n),1n),(word,0n),
    Equal.cong(Nat,Nat & Nat,number => Nat.divmod(number,1n),word,Nat.add(Nat.mul(word,1n),0n),
      Equal.sym(Nat,Nat.add(Nat.mul(word,1n),0n),word,Equal.trans(Nat,Nat.add(Nat.mul(word,1n),0n),Nat.mul(word,1n),word,Addition.zero_right(Nat.mul(word,1n)),Order.times_one(word)))),
    Division.quotient_remainder(word,0n,1n,{==},{==}))
  %Equal.sym(Nat & Nat,Nat.divmod(word,1n),(word,0n),parts) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),1n),0n),1n),Pair.snd(Nat,Nat,_)) == word : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(word,1n),0n),1n),0n),Nat.mul(Nat.add(Nat.mul(word,1n),0n),1n),word,
    Addition.zero_right(Nat.mul(Nat.add(Nat.mul(word,1n),0n),1n)),
    Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(word,1n),0n),1n),Nat.add(Nat.mul(word,1n),0n),word,Order.times_one(Nat.add(Nat.mul(word,1n),0n)),
      Equal.trans(Nat,Nat.add(Nat.mul(word,1n),0n),Nat.mul(word,1n),word,Addition.zero_right(Nat.mul(word,1n)),Order.times_one(word))))

# Lane lane of the unit lane cell at an index: the value at the index in the
# first lane, +0 after it.
def unit_cell_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+position: Grid.Position,+word: U32,lane: Nat) ->
  {Representation.lane(~Logical,~original,logical,Cells.Cells{outer,Tiled.product([]),1,Tiled.groups(1),position},U32.to_nat(word),0n,lane) ==
    Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,word),0.0) : F32}:
  match lane:
    case 0n:
      %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(word),1n),1n),0n),1n),Nat.mod(U32.to_nat(word),1n)),U32.to_nat(word),unit_number(U32.to_nat(word))) :
        {original(logical,U32.from_nat(_)) == original(logical,word) : F32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)) : {original(logical,_) == original(logical,word) : F32}
      {==}
    case 1n+ +more:
      +value = original(logical,U32.from_nat(Representation.logical_index(Cells.Cells{outer,Tiled.product([]),1,Tiled.groups(1),position},U32.to_nat(word),Nat.add(1n+more,0n))))
      %Equal.sym(Bool,Nat.is_lt(Nat.add(more,0n),0n),False{},Division.lt_zero(Nat.add(more,0n))) :
        {Bool.pick(F32,_,value,0.0) == Bool.pick(F32,Nat.is_lt(more,0n),original(logical,word),0.0) : F32}
      %Equal.sym(Bool,Nat.is_lt(more,0n),False{},Division.lt_zero(more)) : {Bool.pick(F32,False{},value,0.0) == Bool.pick(F32,_,original(logical,word),0.0) : F32}
      {==}

# Lanes on the unit axis: lane lane of the cell at a word holds the value at
# the word in the first lane and +0 after it.
def unit_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+position: Grid.Position,+word: U32,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(values)) == True{} : Bool},+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,List.length(&2,U32,values),position)},word)) ==
    Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,word),0.0) : F32}:
  +rank = List.length(&2,U32,values)
  +past = Order.not_below(rank,rank,Order.reflexive(rank))
  +outer = (Public.lane_prefix(values,rank,1) * Tiled.product(Public.lane_after(values,rank)) : U32)
  +removed_same = Public.removed_past(values,rank,past)
  +outer_exact = Equal.trans(Nat,U32.to_nat(outer),ShapeLaws.elements(Grid.removed(values,rank)),ShapeLaws.elements(values),
    outer_word(limits,values,rank,{%Equal.sym(List<&2,U32>,Grid.removed(values,rank),values,removed_same) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      room : {Nat.is_le(ShapeLaws.elements(Grid.removed(values,rank)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}),
    Equal.cong(List<&2,U32>,Nat,shape => ShapeLaws.elements(shape),Grid.removed(values,rank),values,removed_same))
  +outer_new = (Public.lane_prefix(values,rank,1) * Tiled.product([]) : U32)
  +outer_new_exact = {%Public.after_past(values,rank,past) : {U32.to_nat((Public.lane_prefix(values,rank,1) * Tiled.product(_) : U32)) == ShapeLaws.elements(values) : Nat}
    outer_exact : {U32.to_nat(outer_new) == ShapeLaws.elements(values) : Nat}}
  +geometry = {Cells.Cells{outer_new,Tiled.product([]),1,Tiled.groups(1),position} : Cells.Cells}
  +same = Equal.trans(Cells.Cells,Public.geometry_of(values,rank,position),
    Cells.Cells{outer,Tiled.product(Public.lane_after(values,rank)),Grid.lanes_at(values,rank),Tiled.groups(Grid.lanes_at(values,rank)),position},geometry,
    Public.geometry_exact(values,rank,position,Order.reflexive(rank)),
    %Equal.sym(U32,Grid.lanes_at(values,rank),1,Public.lanes_past(values,rank,past)) :
      {Cells.Cells{outer,Tiled.product(Public.lane_after(values,rank)),_,Tiled.groups(_),position} == geometry : Cells.Cells}
    %Equal.sym(List<&2,U32>,Public.lane_after(values,rank),[],Public.after_past(values,rank,past)) :
      {Cells.Cells{(Public.lane_prefix(values,rank,1) * Tiled.product(_) : U32),Tiled.product(_),1,Tiled.groups(1),position} == geometry : Cells.Cells}
    {==})
  +number = U32.to_nat(word)
  %Equal.sym(Cells.Cells,Public.geometry_of(values,rank,position),geometry,same) :
    {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},word)) == Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,word),0.0) : F32}
  Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)),
    Representation.lane(~Logical,~original,logical,geometry,number,0n,lane),Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,word),0.0),
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)),
      TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,geometry,number,0n)),Representation.lane(~Logical,~original,logical,geometry,number,0n,lane),
      Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(lane,tile),Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word),
        Representation.cell(~Logical,~original,logical,geometry,number,0n),
        Representation.decoded(~Logical,~original,logical,geometry,number,0n,word,unit_index(position,U32.to_nat(outer_new),number),
          {%Equal.sym(Nat,U32.to_nat(outer_new),ShapeLaws.elements(values),outer_new_exact) : {Nat.is_lt(number,_) == True{} : Bool} inside :
            {Nat.is_lt(number,U32.to_nat(outer_new)) == True{} : Bool}},{==})),
      Representation.cell_lane(~Logical,~original,logical,geometry,number,0n,lane,small)),
    unit_cell_lane(~Logical,~original,logical,outer_new,position,word,lane))

# Lane lane of the cell at the digits of its other axes and its group holds
# the value at the logical digits of the channel 8 * group + lane, and +0 past
# the lanes, whether the lanes lie inside the values or on the unit axis.
def lane_values(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  within: Bool,+others: List<&2,Nat>,+group: Nat,+word: U32,+within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),placed_digits(within,position,others,group)) : Nat},
  +lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,axis))),
      original(logical,U32.from_nat(ShapeLaws.digits_value(values,logical_digits(within,axis,others,Nat.add(lane,Nat.mul(group,8n)))))),0.0) : F32}:
  match within:
    case True{}:
      lane_digits(~Logical,~original,logical,limits,values,axis,position,others,group,word,within_value,room,below,group_inside,
        {%placed_cell(position,others,group) : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),_) : Nat} exact :
          {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),cell_digits(position,others,group)) : Nat}},lane,small)
    case False{}:
      +rank = List.length(&2,U32,values)
      +equal = Order.antisymmetric(axis,rank,inside,Order.not_less(axis,rank,within_value))
      +group_zero = Order.zero_below(group,Division.lt_succ_le(group,0n,
        {%Public.lanes_past(values,axis,within_value) : {Nat.is_lt(group,U32.to_nat(Grid.groups(_))) == True{} : Bool} group_inside :
          {Nat.is_lt(group,1n) == True{} : Bool}}))
      +removed_same = Public.removed_past(values,axis,within_value)
      +placed_exact = {%within_value :
          {U32.to_nat(word) == ShapeLaws.digits_value(Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),others) : Nat}
        exact : {U32.to_nat(word) == ShapeLaws.digits_value(Grid.removed(values,axis),others) : Nat}}
      +number = {%removed_same : {U32.to_nat(word) == ShapeLaws.digits_value(_,others) : Nat} placed_exact : {U32.to_nat(word) == ShapeLaws.digits_value(values,others) : Nat}}
      +word_back = Equal.trans(U32,U32.from_nat(ShapeLaws.digits_value(values,others)),U32.from_nat(U32.to_nat(word)),word,
        Equal.cong(Nat,U32,U32.from_nat,ShapeLaws.digits_value(values,others),U32.to_nat(word),Equal.sym(Nat,U32.to_nat(word),ShapeLaws.digits_value(values,others),number)),
        Arithmetic.native_roundtrip(word))
      +digits_inside = {%removed_same : {Nat.is_lt(ShapeLaws.digits_value(_,others),ShapeLaws.elements(_)) == True{} : Bool}
        digits_below(Grid.removed(values,axis),others,below) : {Nat.is_lt(ShapeLaws.digits_value(values,others),ShapeLaws.elements(values)) == True{} : Bool}}
      +word_inside = {%Equal.sym(Nat,U32.to_nat(word),ShapeLaws.digits_value(values,others),number) : {Nat.is_lt(_,ShapeLaws.elements(values)) == True{} : Bool}
        digits_inside : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(values)) == True{} : Bool}}
      +values_room = {%removed_same : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
        {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      %Equal.sym(Nat,group,0n,group_zero) :
        {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
          Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(_,8n)),U32.to_nat(Grid.lanes_at(values,axis))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,others))),0.0) : F32}
      %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,within_value)) :
        {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
          Bool.pick(F32,Nat.is_lt(Nat.add(lane,0n),U32.to_nat(_)),original(logical,U32.from_nat(ShapeLaws.digits_value(values,others))),0.0) : F32}
      %Equal.sym(Nat,Nat.add(lane,0n),lane,Addition.zero_right(lane)) :
        {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
          Bool.pick(F32,Nat.is_lt(_,1n),original(logical,U32.from_nat(ShapeLaws.digits_value(values,others))),0.0) : F32}
      %Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(values,others)),word,word_back) :
        {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) ==
          Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,_),0.0) : F32}
      %Equal.sym(Nat,axis,rank,equal) :
        {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,_,position)},word)) ==
          Bool.pick(F32,Nat.is_lt(lane,1n),original(logical,word),0.0) : F32}
      unit_lane(~Logical,~original,logical,limits,values,position,word,values_room,word_inside,lane,small)

# ---- The cells of values within a capacity ----

def groups_fewer(+limits: Machine.Limits,+channels: U32,+facts: TiledValues.Groups<limits,channels>) ->
  {Nat.is_le(U32.to_nat(Grid.groups(channels)),U32.to_nat(channels)) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes} = facts
  fewer
def groups_positive(+limits: Machine.Limits,+channels: U32,+facts: TiledValues.Groups<limits,channels>,+positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Grid.groups(channels))) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes} = facts
  %Equal.sym(Nat,U32.to_nat(Grid.groups(channels)),Ceiling.ceiling(U32.to_nat(channels),8n),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(Ceiling.ceiling(U32.to_nat(channels),8n),8n,
    Bounds.nat_trans(0n,U32.to_nat(channels),Nat.mul(Ceiling.ceiling(U32.to_nat(channels),8n),8n),positive,Ceiling.covers(U32.to_nat(channels),8n,{==})))
# The other axes, the lanes and the cells of values within a capacity fit it,
# and values with at least one value have at least one cell.
def axis_cells(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}>>>:
  match within:
    case True{}:
      +lanes = U32.to_nat(Grid.lanes_at(values,axis))
      +others = ShapeLaws.elements(Grid.removed(values,axis))
      +whole = Nat.mul(lanes,others)
      +split = Public.elements_split(values,axis,within_value)
      +whole_room = {%split : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(whole,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +whole_positive = {%split : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,whole) == True{} : Bool}}
      +lanes_positive = RowMajor.left_factor_positive(lanes,others,whole_positive)
      +others_positive = DivisionOrder.right_factor_positive(lanes,others,whole_positive)
      +lanes_room = Order.transitive(lanes,whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(lanes,others,others_positive),whole_room)
      +groups = U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))
      +grouped = TiledValues.grouped(limits,Grid.lanes_at(values,axis),lanes_room)
      +cells_split = Equal.sym(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),Nat.mul(groups,others),Public.cells_split(values,axis,position,within_value))
      KernelGemm.Both{Order.transitive(others,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(lanes,others,lanes_positive),whole_room),
        KernelGemm.Both{lanes_room,
          KernelGemm.Both{{%cells_split : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
              Order.transitive(Nat.mul(groups,others),whole,U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(groups,lanes,others,groups_fewer(limits,Grid.lanes_at(values,axis),grouped)),whole_room) :
              {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
            {%cells_split : {Nat.is_lt(0n,_) == True{} : Bool}
              Order.product_positive(groups,others,groups_positive(limits,Grid.lanes_at(values,axis),grouped,lanes_positive),others_positive) :
              {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}}}}}
    case False{}:
      +same = Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,Public.removed_past(values,axis,within_value))
      +others_room = {%same : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +others_positive = {%same : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} positive : {Nat.is_lt(0n,ShapeLaws.elements(Grid.removed(values,axis))) == True{} : Bool}}
      +past = Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),False{},within_value)
      KernelGemm.Both{others_room,
        KernelGemm.Both{{%Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,within_value)) : {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            Order.transitive(1n,ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits)),Order.positive_one(ShapeLaws.elements(values),positive),room) : {Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
          KernelGemm.Both{{%past : {Nat.is_le(ShapeLaws.elements(Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} others_room :
              {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
            {%past : {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool} others_positive :
              {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}}}}}

# ---- The geometry of a cell of digits ----

# The group of the cell at the digits of its other axes and its group.
def group_within(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+group: Nat,+word: U32,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: DigitsBelow(Grid.removed(values,axis),others),+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),placed_digits(True{},position,others,group)) : Nat}) ->
  {Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)) == group : Nat}:
  +removed = Grid.removed(values,axis)
  +outer = ShapeLaws.digits_value(removed,others)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +outer_count = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
  +placed = {%within : {U32.to_nat(word) == ShapeLaws.digits_value(Grid.placed(_,position,removed,groups),cell_digits(position,others,group)) : Nat}
    %placed_cell(position,others,group) : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,position),_) : Nat}
    exact : {U32.to_nat(word) == ShapeLaws.digits_value(Grid.placed(True{},position,removed,groups),cell_digits(position,others,group)) : Nat}}
  +numbered = Equal.trans(Nat,U32.to_nat(word),ShapeLaws.digits_value(Grid.placed(True{},position,removed,groups),cell_digits(position,others,group)),
    Representation.index_at(position,ShapeLaws.elements(removed),U32.to_nat(groups),outer,group),placed,cell_number(position,others,group,removed,groups,below))
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),geometry_inside(values,axis,position),geometry_same(values,axis,position,within)) :
    {Representation.cell_group(_,U32.to_nat(word)) == group : Nat}
  %Equal.sym(Nat,U32.to_nat(word),Representation.index_at(position,ShapeLaws.elements(removed),U32.to_nat(groups),outer,group),numbered) :
    {Representation.group_at(position,U32.to_nat(outer_count),U32.to_nat(groups),_) == group : Nat}
  %Equal.sym(Nat,U32.to_nat(outer_count),ShapeLaws.elements(removed),outer_word(limits,values,axis,room)) :
    {Representation.group_at(position,_,U32.to_nat(groups),Representation.index_at(position,ShapeLaws.elements(removed),U32.to_nat(groups),outer,group)) == group : Nat}
  Representation.placed_group(position,ShapeLaws.elements(removed),U32.to_nat(groups),outer,group,digits_below(removed,others,below),group_inside)

# The lane count of the geometry of values is the extent of the lane axis.
def geometry_lanes(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Representation.lane_count(Public.geometry_of(values,axis,position)) == U32.to_nat(Grid.lanes_at(values,axis)) : Nat}:
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
      Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position},Public.geometry_exact(values,axis,position,inside)) :
    {Representation.lane_count(_) == U32.to_nat(Grid.lanes_at(values,axis)) : Nat}
  {==}
