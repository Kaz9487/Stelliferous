# How many cells values laid out on a lane axis have, against the capacity
# of the machine: their rank, their count and the count of their other axes.
import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_tiled_public.bend as Public
import ./nat_ceiling.bend as Ceiling
import ./encoded_coordinates.bend as Encoded
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./tensor_shape_refinement.bend as ShapeLaws

def rank_cases(within: Bool,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool}) ->
  {List.length(&2,U32,Grid.placed(within,position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis)))) == List.length(&2,U32,shape) : Nat}:
  match within:
    case True{}:
      Equal.trans(Nat,List.length(&2,U32,Grid.placed(True{},position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis)))),1n+List.length(&2,U32,Grid.removed(shape,axis)),List.length(&2,U32,shape),
        Public.placed_length(position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis))),Public.removed_length(shape,axis,classified))
    case False{}:
      Equal.cong(List<&2,U32>,Nat,cells => List.length(&2,U32,cells),Grid.removed(shape,axis),shape,Public.removed_past(shape,axis,classified))

# The cells of a shape have its rank.
def cell_rank(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> {List.length(&2,U32,Tiled.cell_shape(shape,axis,position)) == List.length(&2,U32,shape) : Nat}:
  rank_cases(Nat.is_lt(axis,List.length(&2,U32,shape)),shape,axis,position,{==})

# The lane axis of a shape with values within the capacity holds at most the
# capacity, and the other axes too.
def lane_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.lanes_at(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +lanes = U32.to_nat(Grid.lanes_at(shape,axis))
  +others = ShapeLaws.elements(Grid.removed(shape,axis))
  +split = Public.elements_split(shape,axis,within)
  +whole = {%split : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(lanes,others)) == True{} : Bool}}
  Order.transitive(lanes,Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(lanes,others,DivisionOrder.right_factor_positive(lanes,others,whole)),
    {%split : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

def removed_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +lanes = U32.to_nat(Grid.lanes_at(shape,axis))
  +others = ShapeLaws.elements(Grid.removed(shape,axis))
  +split = Public.elements_split(shape,axis,within)
  +whole = {%split : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(lanes,others)) == True{} : Bool}}
  Order.transitive(others,Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(lanes,others,RowMajor.left_factor_positive(lanes,others,whole)),
    {%split : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

def groups_positive(+limits: Machine.Limits,+lanes: U32,+room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Grid.groups(lanes))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),Encoded.eight_blocks_exact(limits,lanes,room)) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(Ceiling.ceiling(U32.to_nat(lanes),8n),8n,
    Bounds.nat_trans(0n,U32.to_nat(lanes),Nat.mul(Ceiling.ceiling(U32.to_nat(lanes),8n),8n),positive,Ceiling.covers(U32.to_nat(lanes),8n,{==})))

def positive_cases(within: Bool,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,shape)) == within : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(within,position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis))))) == True{} : Bool}:
  match within:
    case True{}:
      +lanes = U32.to_nat(Grid.lanes_at(shape,axis))
      +others = ShapeLaws.elements(Grid.removed(shape,axis))
      +whole = {%Public.elements_split(shape,axis,classified) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(lanes,others)) == True{} : Bool}}
      %Equal.sym(Nat,ShapeLaws.elements(Grid.placed(True{},position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis)))),
          Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),others),Public.placed_cells(position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis)))) :
        {Nat.is_lt(0n,_) == True{} : Bool}
      Order.product_positive(U32.to_nat(Grid.groups(Grid.lanes_at(shape,axis))),others,
        groups_positive(limits,Grid.lanes_at(shape,axis),lane_room(limits,shape,axis,classified,positive,room),RowMajor.left_factor_positive(lanes,others,whole)),
        DivisionOrder.right_factor_positive(lanes,others,whole))
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),shape,Public.removed_past(shape,axis,classified)) : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool}
      positive

# The cells of values with values have cells.
def cells_positive(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(shape,axis,position))) == True{} : Bool}:
  positive_cases(Nat.is_lt(axis,List.length(&2,U32,shape)),limits,shape,axis,position,{==},positive,room)

def room_cases(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  match within:
    case True{}:
      +lanes = U32.to_nat(Grid.lanes_at(values,axis))
      +others = ShapeLaws.elements(Grid.removed(values,axis))
      +groups = Ceiling.ceiling(lanes,8n)
      +fewer = Ceiling.minimal(lanes,8n,lanes,{==},RowMajor.product_covers(lanes,8n,{==}))
      +cells_room = Order.transitive(Nat.mul(groups,others),Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(groups,lanes,others,fewer),
        {%Public.elements_split(values,axis,classified) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
          {Nat.is_le(Nat.mul(lanes,others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
      +placed = ShapeLaws.elements(Grid.placed(True{},position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))
      +split = {%classified : {ShapeLaws.elements(Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),others) : Nat}
        Public.cells_split(values,axis,position,classified) : {placed == Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),others) : Nat}}
      %Equal.sym(Nat,placed,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),others),split) :
        {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),groups,Encoded.eight_blocks_exact(limits,Grid.lanes_at(values,axis),lane_room(limits,values,axis,classified,positive,room))) :
        {Nat.is_le(Nat.mul(_,others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      cells_room
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,Public.removed_past(values,axis,classified)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      room

# The cells of values within the capacity are within it.
def cells_room(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  room_cases(Nat.is_lt(axis,List.length(&2,U32,values)),limits,values,axis,position,{==},positive,room)
