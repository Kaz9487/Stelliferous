# What a relane of a shape gives the gather of the relane: the target axes,
# the blocks of the source and the rows and classes of the readings of the
# relane fit the words.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_relane_reads.bend as Reads
import ./tensor_task_reads.bend as TaskReads
import ./tensor_relane_piece.bend as Piece
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_view_reads.bend as ViewReads
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Logic
import ./word_encoding.bend as WordEncoding
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_relane_gather.bend as Gather
import ./tensor_tile_coordinates.bend as Coordinates
import ../kernel_gemm.bend as KernelGemm
import ./tensor_tiled_public.bend as Public
import ./nat_ceiling.bend as NatCeiling

# ---- The readings of a relane by axis ----

# Every target cell axis of a relane is limited by its blocks in the target
# grid.
def relane_limits(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+sized: Reads.AxesExtents(axes,extents,shape,target)) ->
  Components.LimitsAt([Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index)],List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index)),extents,target_grid,index):
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Held.Both{{==},relane_limits(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later)}
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Held.Both{{==},relane_limits(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later)}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+extent,+more}:
      Empty.absurd(Components.LimitsAt([Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Nil{},Con{extent,more},index)],List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Nil{},Con{extent,more},index)),Con{extent,more},target_grid,index),sized)
    case Con{Tiled.OfAxis{+axis},+rest} Nil{}:
      Empty.absurd(Components.LimitsAt([Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfAxis{axis},rest},Nil{},index)],List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfAxis{axis},rest},Nil{},index)),Nil{},target_grid,index),sized)
    case Con{Tiled.OfGroups{},+rest} Nil{}:
      Empty.absurd(Components.LimitsAt([Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfGroups{},rest},Nil{},index)],List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfGroups{},rest},Nil{},index)),Nil{},target_grid,index),sized)

# The target cells of the readings of a relane are the cells of its target
# cell axes.
def relane_row_cells(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+sized: Reads.AxesExtents(axes,extents,shape,target)) ->
  {ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index)) == ShapeLaws.elements(extents) : Nat}:
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))),ShapeLaws.elements(more),
        relane_row_cells(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later))
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))),ShapeLaws.elements(more),
        relane_row_cells(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later))
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+extent,+more}:
      Empty.absurd({ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Nil{},Con{extent,more},index)) == ShapeLaws.elements(Con{extent,more}) : Nat},sized)
    case Con{Tiled.OfAxis{+axis},+rest} Nil{}:
      Empty.absurd({ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfAxis{axis},rest},Nil{},index)) == ShapeLaws.elements(Nil{}) : Nat},sized)
    case Con{Tiled.OfGroups{},+rest} Nil{}:
      Empty.absurd({ViewFacts.RowCells(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfGroups{},rest},Nil{},index)) == ShapeLaws.elements(Nil{}) : Nat},sized)

# ---- Where a source axis lies among the source cell axes ----

# Two numbers apart, one no larger than the other, are ordered.
def le_apart(a: Nat,b: Nat,+le: {Nat.is_le(a,b) == True{} : Bool},+apart: {Nat.is_eq(a,b) == False{} : Bool}) -> {Nat.is_lt(a,b) == True{} : Bool}:
  match a b:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case 0n 1n+ +rest: {==}
    case 1n+ +rest 0n: Empty.absurd({Nat.is_lt(1n+rest,0n) == True{} : Bool},WordEncoding.false_true(le))
    case 1n+ +left 1n+ +right: le_apart(left,right,le,apart)

# An axis of the extents below the one removed keeps its extent.
def removed_below(values: List<&2,U32>,removed: Nat,index: Nat,+below: {Nat.is_lt(index,removed) == True{} : Bool}) ->
  {Grid.lanes_at(Grid.removed(values,removed),index) == Grid.lanes_at(values,index) : U32}:
  match values removed index:
    case Nil{} +removed +index: {==}
    case Con{+extent,+rest} 0n +index:
      Empty.absurd({Grid.lanes_at(Grid.removed(Con{extent,rest},0n),index) == Grid.lanes_at(Con{extent,rest},index) : U32},Order.below_zero(index,below))
    case Con{+extent,+rest} 1n+ +next 0n: {==}
    case Con{+extent,+rest} 1n+ +next 1n+ +later: removed_below(rest,next,later,below)

# An axis of the extents from the one removed on holds the extent after it.
def removed_above(values: List<&2,U32>,removed: Nat,index: Nat,+above: {Nat.is_le(removed,index) == True{} : Bool}) ->
  {Grid.lanes_at(Grid.removed(values,removed),index) == Grid.lanes_at(values,1n+index) : U32}:
  match values removed index:
    case Nil{} +removed +index: {==}
    case Con{+extent,+rest} 0n +index: {==}
    case Con{+extent,+rest} 1n+ +next 0n:
      Empty.absurd({Grid.lanes_at(Grid.removed(Con{extent,rest},1n+next),0n) == Grid.lanes_at(Con{extent,rest},1n) : U32},WordEncoding.false_true(above))
    case Con{+extent,+rest} 1n+ +next 1n+ +later: removed_above(rest,next,later,above)

# Removing an axis inside the extents leaves one fewer.
def removed_length(values: List<&2,U32>,removed: Nat,+inside: {Nat.is_lt(removed,List.length(&2,U32,values)) == True{} : Bool}) ->
  {1n+List.length(&2,U32,Grid.removed(values,removed)) == List.length(&2,U32,values) : Nat}:
  match values removed:
    case Nil{} +removed:
      Empty.absurd({1n+List.length(&2,U32,Grid.removed(Nil{},removed)) == List.length(&2,U32,Nil{}) : Nat},Order.below_zero(removed,inside))
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next:
      Equal.cong(Nat,Nat,count => 1n+count,1n+List.length(&2,U32,Grid.removed(rest,next)),List.length(&2,U32,rest),removed_length(rest,next,inside))

# An axis of extents keeps its extent when one more is appended.
def appended_lanes(values: List<&2,U32>,index: Nat,+last: U32,+inside: {Nat.is_lt(index,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Grid.lanes_at(List.append(&2,U32,values,[last]),index) == Grid.lanes_at(values,index) : U32}:
  match values index:
    case Nil{} +index:
      Empty.absurd({Grid.lanes_at(List.append(&2,U32,Nil{},[last]),index) == Grid.lanes_at(Nil{},index) : U32},Order.below_zero(index,inside))
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next: appended_lanes(rest,next,last,inside)

# The place among the cell axes of a shape with its lanes on source of a
# logical axis other than source, without and with the groups.
def inner_place(+source: Nat,+source_within: Bool,+axis: Nat) -> Nat:
  Bool.pick(Nat,source_within && Nat.is_lt(source,axis),Nat.sub(axis,1n),axis)

def cell_place(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat) -> Nat:
  Bool.pick(Nat,source_within && Grid.is_outermost(source_position),1n+inner_place(source,source_within,axis),inner_place(source,source_within,axis))

def inner_after(axis: Nat,+source: Nat,+values: List<&2,U32>,+after: {Nat.is_lt(source,axis) == True{} : Bool},
  +axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  Held.Both<{Grid.lanes_at(Grid.removed(values,source),Nat.sub(axis,1n)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(1n+Nat.sub(axis,1n),List.length(&2,U32,values)) == True{} : Bool}>:
  match axis:
    case 0n:
      Empty.absurd(Held.Both<{Grid.lanes_at(Grid.removed(values,source),Nat.sub(0n,1n)) == Grid.lanes_at(values,0n) : U32},
        {Nat.is_lt(1n+Nat.sub(0n,1n),List.length(&2,U32,values)) == True{} : Bool}>,Order.below_zero(source,after))
    case 1n+ +previous:
      {%Equal.sym(Nat,Nat.sub(1n+previous,1n),previous,Order.subtract_zero(previous)) :
          Held.Both<{Grid.lanes_at(Grid.removed(values,source),_) == Grid.lanes_at(values,1n+previous) : U32},
            {Nat.is_lt(1n+_,List.length(&2,U32,values)) == True{} : Bool}>
        Held.Both{removed_above(values,source,previous,Order.strict_follow(source,1n+previous,after)),axis_in} :
        Held.Both<{Grid.lanes_at(Grid.removed(values,source),Nat.sub(1n+previous,1n)) == Grid.lanes_at(values,1n+previous) : U32},
          {Nat.is_lt(1n+Nat.sub(1n+previous,1n),List.length(&2,U32,values)) == True{} : Bool}>}

# A logical axis other than the source lane axis keeps its extent among the
# extents without the source axis, at its inner place.
def inner_lanes(+source: Nat,+axis: Nat,+values: List<&2,U32>,+apart: {Nat.is_eq(axis,source) == False{} : Bool},
  +axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+source_in: {Nat.is_lt(source,List.length(&2,U32,values)) == True{} : Bool}) ->
  Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,values)) == True{} : Bool}>:
  Held.by_cases(Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
      {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,values)) == True{} : Bool}>,Nat.is_lt(source,axis),
    +after => {%Equal.sym(Bool,Nat.is_lt(source,axis),True{},after) :
        Held.Both<{Grid.lanes_at(Grid.removed(values,source),Bool.pick(Nat,_,Nat.sub(axis,1n),axis)) == Grid.lanes_at(values,axis) : U32},
          {Nat.is_lt(1n+Bool.pick(Nat,_,Nat.sub(axis,1n),axis),List.length(&2,U32,values)) == True{} : Bool}>
      inner_after(axis,source,values,after,axis_in) :
      Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
        {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,values)) == True{} : Bool}>},
    +before => {%Equal.sym(Bool,Nat.is_lt(source,axis),False{},before) :
        Held.Both<{Grid.lanes_at(Grid.removed(values,source),Bool.pick(Nat,_,Nat.sub(axis,1n),axis)) == Grid.lanes_at(values,axis) : U32},
          {Nat.is_lt(1n+Bool.pick(Nat,_,Nat.sub(axis,1n),axis),List.length(&2,U32,values)) == True{} : Bool}>
      Held.Both{removed_below(values,source,axis,le_apart(axis,source,Order.not_less(source,axis,before),apart)),
        Order.le_lt_trans(1n+axis,source,List.length(&2,U32,values),Order.strict_follow(axis,source,le_apart(axis,source,Order.not_less(source,axis,before),apart)),source_in)} :
      Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
        {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,values)) == True{} : Bool}>})

def inner_both(+source: Nat,+axis: Nat,+values: List<&2,U32>,
  +inner: Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,values)) == True{} : Bool}>,
  +source_in: {Nat.is_lt(source,List.length(&2,U32,values)) == True{} : Bool}) ->
  Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(inner_place(source,True{},axis),List.length(&2,U32,Grid.removed(values,source))) == True{} : Bool}>:
  Held.Both{+same,+room} = inner
  Held.Both{same,{%Equal.sym(Nat,1n+List.length(&2,U32,Grid.removed(values,source)),List.length(&2,U32,values),removed_length(values,source,source_in)) :
      {Nat.is_lt(1n+inner_place(source,True{},axis),_) == True{} : Bool}
    room : {Nat.is_lt(1n+inner_place(source,True{},axis),1n+List.length(&2,U32,Grid.removed(values,source))) == True{} : Bool}}}

def innermost_lanes(+source: Nat,+axis: Nat,+values: List<&2,U32>,+groups: U32,
  +inner: Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(inner_place(source,True{},axis),List.length(&2,U32,Grid.removed(values,source))) == True{} : Bool}>) ->
  Held.Both<{Grid.lanes_at(List.append(&2,U32,Grid.removed(values,source),[groups]),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(inner_place(source,True{},axis),List.length(&2,U32,List.append(&2,U32,Grid.removed(values,source),[groups]))) == True{} : Bool}>:
  Held.Both{+same,+below} = inner
  +others = Grid.removed(values,source)
  +place = inner_place(source,True{},axis)
  Held.Both{Equal.trans(U32,Grid.lanes_at(List.append(&2,U32,others,[groups]),place),Grid.lanes_at(others,place),Grid.lanes_at(values,axis),
      appended_lanes(others,place,groups,below),same),
    {%Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,others,[groups])),1n+List.length(&2,U32,others),Piece.appended_count(~U32,others,groups)) :
        {Nat.is_lt(place,_) == True{} : Bool}
      Order.lt_succ(place,List.length(&2,U32,others),Bounds.strict_le(place,List.length(&2,U32,others),below)) :
      {Nat.is_lt(place,List.length(&2,U32,List.append(&2,U32,others,[groups]))) == True{} : Bool}}}

def outermost_lanes(+source: Nat,+axis: Nat,+values: List<&2,U32>,+groups: U32,
  +inner: Held.Both<{Grid.lanes_at(Grid.removed(values,source),inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(inner_place(source,True{},axis),List.length(&2,U32,Grid.removed(values,source))) == True{} : Bool}>) ->
  Held.Both<{Grid.lanes_at(Con{groups,Grid.removed(values,source)},1n+inner_place(source,True{},axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(1n+inner_place(source,True{},axis),List.length(&2,U32,Con{groups,Grid.removed(values,source)})) == True{} : Bool}>:
  Held.Both{+same,+below} = inner
  Held.Both{same,below}

# A logical axis other than the source lane axis holds its extent at its
# place among the cell axes of the source.
def placed_lanes(within: Bool,position: Grid.Position,+values: List<&2,U32>,+source: Nat,+axis: Nat,+groups: U32,
  +within_value: {Nat.is_lt(source,List.length(&2,U32,values)) == within : Bool},+apart: {Bool.and(within,Nat.is_eq(axis,source)) == False{} : Bool},
  +axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  Held.Both<{Grid.lanes_at(Grid.placed(within,position,Grid.removed(values,source),groups),cell_place(source,within,position,axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(cell_place(source,within,position,axis),List.length(&2,U32,Grid.placed(within,position,Grid.removed(values,source),groups))) == True{} : Bool}>:
  match within position:
    case False{} +position:
      +past = Order.not_less(source,List.length(&2,U32,values),within_value)
      Held.Both{removed_below(values,source,axis,Bounds.nat_trans(axis,List.length(&2,U32,values),source,axis_in,past)),
        {%Equal.sym(List<&2,U32>,Grid.removed(values,source),values,Reads.removed_past(values,source,past)) : {Nat.is_lt(axis,List.length(&2,U32,_)) == True{} : Bool}
          axis_in : {Nat.is_lt(axis,List.length(&2,U32,Grid.removed(values,source))) == True{} : Bool}}}
    case True{} Grid.Innermost{}:
      innermost_lanes(source,axis,values,groups,inner_both(source,axis,values,inner_lanes(source,axis,values,apart,axis_in,within_value),within_value))
    case True{} Grid.Outermost{}:
      outermost_lanes(source,axis,values,groups,inner_both(source,axis,values,inner_lanes(source,axis,values,apart,axis_in,within_value),within_value))

# The same, on the cell shape of the source.
def cell_lanes(+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+apart: {Bool.and(source_within,Nat.is_eq(axis,source)) == False{} : Bool},
  +axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>:
  +groups = Grid.groups(Grid.lanes_at(values,source))
  {%Equal.sym(Bool,Nat.is_lt(source,List.length(&2,U32,values)),source_within,source_value) :
      Held.Both<{Grid.lanes_at(Grid.placed(_,source_position,Grid.removed(values,source),groups),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
        {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Grid.placed(_,source_position,Grid.removed(values,source),groups))) == True{} : Bool}>
    placed_lanes(source_within,source_position,values,source,axis,groups,source_value,apart,axis_in) :
    Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
      {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>}

# The block of the axis at index index of extents walked from at fits half a
# word twice over.
def block_index(extents: List<&2,U32>,index: Nat,+grid: Grid.Grid,+at: U32,+position: Nat,+limits: Machine.Limits,+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(position,index),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,extents)) == True{} : Bool},+blocks: ViewReads.BlocksIn(grid,extents,at,Machine.half_word(limits))) ->
  {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(Nat.add(position,index)),Grid.lanes_at(extents,index)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  match extents index:
    case Con{+head,+rest} 0n:
      Held.Both{+positive,Held.Both{+double,+later}} = blocks
      +word = Equal.trans(U32,U32.from_nat(Nat.add(position,0n)),U32.from_nat(position),at,
        Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.add(position,0n),position,Addition.zero_right(position)),
        Equal.trans(U32,U32.from_nat(position),U32.from_nat(U32.to_nat(at)),at,
          Equal.cong(Nat,U32,number => U32.from_nat(number),position,U32.to_nat(at),Equal.sym(Nat,U32.to_nat(at),position,exact)),Arithmetic.native_roundtrip(at)))
      {%Equal.sym(U32,U32.from_nat(Nat.add(position,0n)),at,word) :
          {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,_,head),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
        double : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(Nat.add(position,0n)),head),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
    case Con{+head,+rest} 1n+ +k:
      Held.Both{+positive,Held.Both{+double,+later}} = blocks
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(position,length),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),Limits.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),cap,Order.le_add(1n+position,length),shifted))
      {%Equal.sym(Nat,Nat.add(position,1n+k),1n+Nat.add(position,k),Addition.successor_right(position,k)) :
          {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(_),Grid.lanes_at(rest,k)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
        block_index(rest,k,grid,(at + 1 : U32),1n+position,limits,next,shifted,
          {%Addition.successor_right(position,k) : {Nat.is_le(_,cap) == True{} : Bool} axis_room : {Nat.is_le(1n+Nat.add(position,k),cap) == True{} : Bool}},inside,later) :
        {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(Nat.add(position,1n+k)),Grid.lanes_at(rest,k)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
    case Nil{} +index:
      Empty.absurd({Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(Nat.add(position,index)),Grid.lanes_at(Nil{},index)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
        Order.below_zero(index,inside))

# A read of a source axis with no offset, unmirrored and unshifted, has room
# at half a word and reads its axis.
def read_at_ok(+limits: Machine.Limits,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+extent: U32,+block: U32,+axis: U32,+bound: U32,+step: U32,+width: U32,
  +divisor: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +reach: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n)),U32.to_nat(width)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +divisor_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(divisor,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(bound),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(source_grid,axis,bound),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +extent_at: ViewFacts.ExtentAt(source_cells,0,axis,bound),+axis_below: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,source_cells)) == True{} : Bool}) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(source_grid,axis,bound,step,width,divisor),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(source_grid,axis,bound,step,width,divisor),source_cells,source_grid)>:
  Held.Both{Held.Both{room,Held.Both{width_positive,Held.Both{reach,Held.Both{divisor_double,Held.Both{bound_room,block_double}}}}},
    Held.Both{{==},Held.Both{extent_at,axis_below}}}

def minus_one_plus(value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.add(Nat.sub(value,1n),1n) == value : Nat}:
  match value:
    case 0n: Empty.absurd({Nat.add(Nat.sub(0n,1n),1n) == 0n : Nat},Order.below_zero(0n,positive))
    case 1n+ +rest:
      Equal.trans(Nat,Nat.add(Nat.sub(1n+rest,1n),1n),Nat.add(rest,1n),1n+rest,
        Equal.cong(Nat,Nat,number => Nat.add(number,1n),Nat.sub(1n+rest,1n),rest,Order.subtract_zero(rest)),Addition.successor_width(rest))

# A read of unit step and width reaches as far as its extent.
def unit_reach(+limits: Machine.Limits,+extent: U32,+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(1n,Nat.sub(U32.to_nat(extent),1n)),1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +before = Nat.sub(U32.to_nat(extent),1n)
  {%Equal.sym(Nat,Nat.mul(1n,before),before,TaskReads.one_times(before)) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(before,1n),U32.to_nat(extent),minus_one_plus(U32.to_nat(extent),extent_positive)) :
      {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Limits.within_half(limits,U32.to_nat(extent),extent_room) :
    {Nat.is_le(Nat.add(Nat.mul(1n,before),1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

# ---- Every target cell axis of a relane names an axis of the shape ----

def AxesIn(axes: List<&2,Tiled.CellAxis>,+values: List<&2,U32>,+target: Nat) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Tiled.OfGroups{},rest}: Held.Both<{Nat.is_lt(target,List.length(&2,U32,values)) == True{} : Bool},AxesIn(rest,values,target)>
    case Con{Tiled.OfAxis{+axis},rest}: Held.Both<{Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},AxesIn(rest,values,target)>

def from_choice(choice: Bool,+index: Nat,+later: List<&2,Tiled.CellAxis>,+values: List<&2,U32>,+target: Nat,
  +index_in: {Nat.is_lt(index,List.length(&2,U32,values)) == True{} : Bool},+later_in: AxesIn(later,values,target)) ->
  AxesIn(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),values,target):
  match choice:
    case True{}: later_in
    case False{}: Held.Both{index_in,later_in}

def from_in(remaining: Nat,+index: Nat,+skipped: Nat,+values: List<&2,U32>,+target: Nat,
  +room: {Nat.is_le(Nat.add(index,remaining),List.length(&2,U32,values)) == True{} : Bool}) -> AxesIn(Tiled.axes_from(remaining,index,skipped),values,target):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +length = List.length(&2,U32,values)
      +shifted = {%Addition.successor_right(index,rest) : {Nat.is_le(_,length) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(index,rest),length) == True{} : Bool}}
      from_choice(Nat.is_eq(index,skipped),index,Tiled.axes_from(rest,1n+index,skipped),values,target,
        Bounds.nat_trans(index,1n+Nat.add(index,rest),length,Order.lt_succ(index,Nat.add(index,rest),Order.le_add(index,rest)),shifted),
        from_in(rest,1n+index,skipped,values,target,shifted))

def appended_in(axes: List<&2,Tiled.CellAxis>,+values: List<&2,U32>,+target: Nat,+axes_in: AxesIn(axes,values,target),
  +target_in: {Nat.is_lt(target,List.length(&2,U32,values)) == True{} : Bool}) -> AxesIn(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),values,target):
  match axes:
    case Nil{}: Held.Both{target_in,Unit{}}
    case Con{Tiled.OfGroups{},+rest}:
      Held.Both{+here,+later} = axes_in
      Held.Both{here,appended_in(rest,values,target,later,target_in)}
    case Con{Tiled.OfAxis{+axis},+rest}:
      Held.Both{+here,+later} = axes_in
      Held.Both{here,appended_in(rest,values,target,later,target_in)}

# Every target cell axis of a relane is the lanes target or an axis of the
# shape.
def axes_in(within: Bool,position: Grid.Position,+target: Nat,+values: List<&2,U32>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool}) ->
  AxesIn(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),values,target):
  match within position:
    case False{} +position: from_in(List.length(&2,U32,values),0n,List.length(&2,U32,values),values,target,Order.reflexive(List.length(&2,U32,values)))
    case True{} Grid.Innermost{}:
      appended_in(Tiled.axes_from(List.length(&2,U32,values),0n,target),values,target,from_in(List.length(&2,U32,values),0n,target,values,target,Order.reflexive(List.length(&2,U32,values))),within_value)
    case True{} Grid.Outermost{}:
      Held.Both{within_value,from_in(List.length(&2,U32,values),0n,target,values,target,Order.reflexive(List.length(&2,U32,values)))}

# ---- The groups axis of the source cells ----

# The extent a word index picks is the extent at that index.
def nth_lanes(list: List<&2,U32>,index: Nat,+limits: Machine.Limits,
  +room: {Nat.is_le(List.length(&2,U32,list),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,list)) == True{} : Bool}) -> {Tasks.nth(list,U32.from_nat(index)) == Grid.lanes_at(list,index) : U32}:
  match list index:
    case Con{+head,+rest} 0n: {==}
    case Con{+head,+rest} 1n+ +k:
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,rest)
      +word = U32.from_nat(1n+k)
      +back = Bounds.roundtrip(1n+k,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(1n+k,cap,Bounds.nat_trans(1n+k,1n+length,cap,inside,room)))
      +nonzero = Arithmetic.word_positive_nonzero(word,{%Equal.sym(Nat,U32.to_nat(word),1n+k,back) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
        {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}})
      +less = Equal.trans(U32,(word - 1 : U32),U32.from_nat(Nat.sub(U32.to_nat(word),1n)),U32.from_nat(k),
        Arithmetic.native_subtract(word,1,{%Equal.sym(Nat,U32.to_nat(word),1n+k,back) : {Nat.is_le(1n,_) == True{} : Bool} Order.zero_le(k) :
          {Nat.is_le(1n,U32.to_nat(word)) == True{} : Bool}}),
        Equal.trans(U32,U32.from_nat(Nat.sub(U32.to_nat(word),1n)),U32.from_nat(Nat.sub(1n+k,1n)),U32.from_nat(k),
          Equal.cong(Nat,U32,number => U32.from_nat(Nat.sub(number,1n)),U32.to_nat(word),1n+k,back),
          Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.sub(1n+k,1n),k,Order.subtract_zero(k))))
      +rest_room = Bounds.strict_le(length,cap,Bounds.nat_trans(length,1n+length,cap,Order.lt_succ(length,length,Order.reflexive(length)),room))
      %Equal.sym(Bool,U32.is_zero(word),False{},nonzero) : {Bool.pick(U32,_,head,Tasks.nth(rest,(word - 1 : U32))) == Grid.lanes_at(rest,k) : U32}
      %Equal.sym(U32,(word - 1 : U32),U32.from_nat(k),less) : {Tasks.nth(rest,_) == Grid.lanes_at(rest,k) : U32}
      nth_lanes(rest,k,limits,rest_room,inside)
    case Nil{} +index:
      Empty.absurd({Tasks.nth(Nil{},U32.from_nat(index)) == Grid.lanes_at(Nil{},index) : U32},Order.below_zero(index,inside))

# The last of extents appended with one more is that one.
def appended_last(values: List<&2,U32>,+last: U32) -> {Grid.lanes_at(List.append(&2,U32,values,[last]),List.length(&2,U32,values)) == last : U32}:
  match values:
    case Nil{}: {==}
    case Con{+extent,+rest}: appended_last(rest,last)

# The place of the groups among the source cell axes.
def groups_place(+position: Grid.Position,+cells: List<&2,U32>) -> Nat:
  Bool.pick(Nat,Grid.is_outermost(position),0n,Nat.sub(List.length(&2,U32,cells),1n))

def groups_word(position: Grid.Position,+cells: List<&2,U32>,+limits: Machine.Limits,
  +room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_positive: {Nat.is_lt(0n,List.length(&2,U32,cells)) == True{} : Bool}) ->
  {Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)) == U32.from_nat(groups_place(position,cells)) : U32}:
  match position:
    case Grid.Outermost{}: {==}
    case Grid.Innermost{}:
      +length = List.length(&2,U32,cells)
      +back = Bounds.roundtrip(length,Machine.capacity(limits),Limits.capacity_fits(limits),room)
      Equal.trans(U32,(U32.from_nat(length) - 1 : U32),U32.from_nat(Nat.sub(U32.to_nat(U32.from_nat(length)),1n)),U32.from_nat(Nat.sub(length,1n)),
        Arithmetic.native_subtract(U32.from_nat(length),1,{%Equal.sym(Nat,U32.to_nat(U32.from_nat(length)),length,back) : {Nat.is_le(1n,_) == True{} : Bool}
          Order.positive_one(length,rank_positive) : {Nat.is_le(1n,U32.to_nat(U32.from_nat(length))) == True{} : Bool}}),
        Equal.cong(Nat,U32,number => U32.from_nat(Nat.sub(number,1n)),U32.to_nat(U32.from_nat(length)),length,back))

# The groups of placed cell axes lie at the groups place.
def placed_groups(position: Grid.Position,+others: List<&2,U32>,+groups: U32) ->
  Held.Both<{Grid.lanes_at(Grid.placed(True{},position,others,groups),groups_place(position,Grid.placed(True{},position,others,groups))) == groups : U32},
    Held.Both<{Nat.is_lt(groups_place(position,Grid.placed(True{},position,others,groups)),List.length(&2,U32,Grid.placed(True{},position,others,groups))) == True{} : Bool},
      {Nat.is_lt(0n,List.length(&2,U32,Grid.placed(True{},position,others,groups))) == True{} : Bool}>>:
  match position:
    case Grid.Outermost{}: Held.Both{{==},Held.Both{{==},{==}}}
    case Grid.Innermost{}:
      +list = List.append(&2,U32,others,[groups])
      +length = List.length(&2,U32,others)
      +count = Piece.appended_count(~U32,others,groups)
      +place_eq = Equal.trans(Nat,Nat.sub(List.length(&2,U32,list),1n),Nat.sub(1n+length,1n),length,
        Equal.cong(Nat,Nat,number => Nat.sub(number,1n),List.length(&2,U32,list),1n+length,count),Order.subtract_zero(length))
      Held.Both{{%Equal.sym(Nat,Nat.sub(List.length(&2,U32,list),1n),length,place_eq) : {Grid.lanes_at(list,_) == groups : U32}
          appended_last(others,groups) : {Grid.lanes_at(list,Nat.sub(List.length(&2,U32,list),1n)) == groups : U32}},
        Held.Both{{%Equal.sym(Nat,Nat.sub(List.length(&2,U32,list),1n),length,place_eq) : {Nat.is_lt(_,List.length(&2,U32,list)) == True{} : Bool}
            %Equal.sym(Nat,List.length(&2,U32,list),1n+length,count) : {Nat.is_lt(length,_) == True{} : Bool}
            Order.lt_succ(length,length,Order.reflexive(length)) :
            {Nat.is_lt(Nat.sub(List.length(&2,U32,list),1n),List.length(&2,U32,list)) == True{} : Bool}},
          {%Equal.sym(Nat,List.length(&2,U32,list),1n+length,count) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
            {Nat.is_lt(0n,List.length(&2,U32,list)) == True{} : Bool}}}}

# An extent of a shape of cells is at most its cells.
def lanes_le_elements(list: List<&2,U32>,index: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(list)) == True{} : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,list)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.lanes_at(list,index)),ShapeLaws.elements(list)) == True{} : Bool}:
  match list index:
    case Con{+extent,+rest} 0n:
      RowMajor.product_covers(U32.to_nat(extent),ShapeLaws.elements(rest),DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive))
    case Con{+extent,+rest} 1n+ +k:
      +inner = ShapeLaws.elements(rest)
      Order.transitive(U32.to_nat(Grid.lanes_at(rest,k)),inner,Nat.mul(U32.to_nat(extent),inner),
        lanes_le_elements(rest,k,DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive),inside),
        DivisionOrder.le_multiple(U32.to_nat(extent),inner,RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)))
    case Nil{} +index:
      Empty.absurd({Nat.is_le(U32.to_nat(Grid.lanes_at(Nil{},index)),ShapeLaws.elements(Nil{})) == True{} : Bool},Order.below_zero(index,inside))

# The read of the source lane axis reads the source groups: it has room at
# half a word and reads the groups axis.
def lane_read_at(+limits: Machine.Limits,+cells: List<&2,U32>,+place: Nat,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Grid.grid(cells,Grid.piece_target()),cells,0,Machine.half_word(limits)),
  +below: {Nat.is_lt(place,List.length(&2,U32,cells)) == True{} : Bool}) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(cells,Grid.piece_target()),U32.from_nat(place),Tasks.nth(cells,U32.from_nat(place)),1,1,8),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Grid.grid(cells,Grid.piece_target()),U32.from_nat(place),Tasks.nth(cells,U32.from_nat(place)),1,1,8),cells,Grid.grid(cells,Grid.piece_target()))>:
  +grid = Grid.grid(cells,Grid.piece_target())
  +cap = U32.to_nat(Machine.capacity(limits))
  +place_room = Bounds.strict_le(place,cap,Bounds.nat_trans(place,List.length(&2,U32,cells),cap,below,cells_rank))
  +picked = nth_lanes(cells,place,limits,cells_rank,below)
  +bound_room = {%Equal.sym(U32,Tasks.nth(cells,U32.from_nat(place)),Grid.lanes_at(cells,place),picked) :
      {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    Limits.within_half(limits,U32.to_nat(Grid.lanes_at(cells,place)),
      Order.transitive(U32.to_nat(Grid.lanes_at(cells,place)),ShapeLaws.elements(cells),cap,lanes_le_elements(cells,place,cells_positive,below),cells_room)) :
    {Nat.is_le(U32.to_nat(Tasks.nth(cells,U32.from_nat(place))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +extent_at = {%Equal.sym(U32,Tasks.nth(cells,U32.from_nat(place)),Grid.lanes_at(cells,place),picked) : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),_)
    ViewFacts.extent_index(cells,place,0,0n,limits,{==},cells_rank,place_room,below) : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),Tasks.nth(cells,U32.from_nat(place)))}
  +block_double = {%Equal.sym(U32,Tasks.nth(cells,U32.from_nat(place)),Grid.lanes_at(cells,place),picked) :
      {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),_),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    block_index(cells,place,grid,0,0n,limits,{==},cells_rank,place_room,below,blocks) :
    {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),Tasks.nth(cells,U32.from_nat(place))),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +axis_below = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(place)),place,Bounds.roundtrip(place,Machine.capacity(limits),Limits.capacity_fits(limits),place_room)) :
      {Nat.is_lt(_,List.length(&2,U32,cells)) == True{} : Bool}
    below : {Nat.is_lt(U32.to_nat(U32.from_nat(place)),List.length(&2,U32,cells)) == True{} : Bool}}
  read_at_ok(limits,cells,grid,extent,block,U32.from_nat(place),Tasks.nth(cells,U32.from_nat(place)),1,1,8,room,{==},unit_reach(limits,extent,extent_room,extent_positive),
    Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),bound_room,block_double,extent_at,axis_below)

# The read of a logical axis at its place among the source cell axes, with
# no offset and a divisor of one, has room at half a word and reads its axis.
def place_read_ok(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,+extent: U32,+block: U32,+step: U32,+width: U32,
  +room: Components.AxisRoom(extent,block,Machine.half_word(limits)),+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +reach: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n)),U32.to_nat(width)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +bound_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +placed: Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),step,width,1),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),step,width,1),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  Held.Both{+lanes,+below} = placed
  +place = cell_place(source,source_within,source_position,axis)
  +cells = Tiled.cell_shape(values,source,source_position)
  +grid = Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target())
  +cap = U32.to_nat(Machine.capacity(limits))
  +place_room = Bounds.strict_le(place,cap,Bounds.nat_trans(place,List.length(&2,U32,cells),cap,below,cells_rank))
  +extent_at = {%lanes : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),_)
    ViewFacts.extent_index(cells,place,0,0n,limits,{==},cells_rank,place_room,below) : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),Grid.lanes_at(values,axis))}
  +block_double = {%lanes : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),_),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    block_index(cells,place,grid,0,0n,limits,{==},cells_rank,place_room,below,blocks) :
    {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),Grid.lanes_at(values,axis)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +axis_below = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(place)),place,Bounds.roundtrip(place,Machine.capacity(limits),Limits.capacity_fits(limits),place_room)) :
      {Nat.is_lt(_,List.length(&2,U32,cells)) == True{} : Bool}
    below : {Nat.is_lt(U32.to_nat(U32.from_nat(place)),List.length(&2,U32,cells)) == True{} : Bool}}
  read_at_ok(limits,cells,grid,extent,block,U32.from_nat(place),Grid.lanes_at(values,axis),step,width,1,room,width_positive,reach,
    ViewFacts.two_half(limits),Limits.within_half(limits,U32.to_nat(Grid.lanes_at(values,axis)),bound_room),block_double,extent_at,axis_below)

# The read of a logical axis other than the source lane axis has room at
# half a word and reads its source cell axis.
def shared_read_ok(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +sized: {extent == Grid.lanes_at(values,axis) : U32},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +placed: Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),1,1,1),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),1,1,1),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  place_read_ok(limits,values,source,source_within,source_position,axis,extent,block,1,1,room,{==},unit_reach(limits,extent,extent_room,extent_positive),
    {%sized : {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} extent_room :
      {Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},cells_rank,blocks,placed)

def eight_reach(groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {Nat.add(Nat.mul(8n,Nat.sub(groups,1n)),8n) == Nat.mul(groups,8n) : Nat}:
  match groups:
    case 0n: Empty.absurd({Nat.add(Nat.mul(8n,Nat.sub(0n,1n)),8n) == Nat.mul(0n,8n) : Nat},Order.below_zero(0n,positive))
    case 1n+ +rest:
      Equal.trans(Nat,Nat.add(Nat.mul(8n,Nat.sub(1n+rest,1n)),8n),Nat.add(Nat.mul(8n,rest),8n),Nat.mul(1n+rest,8n),
        Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(8n,number),8n),Nat.sub(1n+rest,1n),rest,Order.subtract_zero(rest)),
        Equal.trans(Nat,Nat.add(Nat.mul(8n,rest),8n),Nat.add(Nat.mul(rest,8n),8n),Nat.mul(1n+rest,8n),
          Equal.cong(Nat,Nat,number => Nat.add(number,8n),Nat.mul(8n,rest),Nat.mul(rest,8n),RowMajor.multiply_commutative(8n,rest)),
          Order.commutative(Nat.mul(rest,8n),8n)))

def groups_lanes(+limits: Machine.Limits,+channels: U32,+facts: TiledValues.Groups<limits,channels>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(channels)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+lanes} = facts
  lanes

# The read of the target lanes axis of a relane, eight lanes a group, has
# room at half a word and reads its source cell axis.
def groups_read_ok(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+sized: {extent == Grid.groups(Grid.lanes_at(values,axis)) : U32},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +placed: Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),cell_place(source,source_within,source_position,axis)) == Grid.lanes_at(values,axis) : U32},
    {Nat.is_lt(cell_place(source,source_within,source_position,axis),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),8,8,1),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),8,8,1),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  +half = U32.to_nat(Machine.half_word(limits))
  +groups = U32.to_nat(extent)
  +reach = {%Equal.sym(Nat,Nat.add(Nat.mul(8n,Nat.sub(groups,1n)),8n),Nat.mul(groups,8n),eight_reach(groups,extent_positive)) : {Nat.is_le(_,half) == True{} : Bool}
    %Equal.sym(U32,extent,Grid.groups(Grid.lanes_at(values,axis)),sized) : {Nat.is_le(Nat.mul(U32.to_nat(_),8n),half) == True{} : Bool}
    groups_lanes(limits,Grid.lanes_at(values,axis),TiledValues.grouped(limits,Grid.lanes_at(values,axis),lanes_room)) :
    {Nat.is_le(Nat.add(Nat.mul(8n,Nat.sub(groups,1n)),8n),half) == True{} : Bool}}
  place_read_ok(limits,values,source,source_within,source_position,axis,extent,block,8,8,room,{==},reach,lanes_room,cells_rank,blocks,placed)

# ---- The readings of a relane ----

def and_left(left: Bool,+right: Bool,+both: {Bool.and(left,right) == True{} : Bool}) -> {left == True{} : Bool}:
  match left:
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(both))
    case True{}: {==}

def lane_read_with(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +groups_facts: Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),groups_place(source_position,Tiled.cell_shape(values,source,source_position))) == Grid.groups(Grid.lanes_at(values,source)) : U32},
    Held.Both<{Nat.is_lt(groups_place(source_position,Tiled.cell_shape(values,source,source_position)),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
      {Nat.is_lt(0n,List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>>) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Machine.half_word(limits)),ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  Held.Both{+lanes,Held.Both{+below,+rank_positive}} = groups_facts
  %Equal.sym(U32,Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),U32.from_nat(groups_place(source_position,Tiled.cell_shape(values,source,source_position))),groups_word(source_position,Tiled.cell_shape(values,source,source_position),limits,cells_rank,rank_positive)) :
    Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),_,Tasks.nth(Tiled.cell_shape(values,source,source_position),_),1,1,8),Machine.half_word(limits)),ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),_,Tasks.nth(Tiled.cell_shape(values,source,source_position),_),1,1,8),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>
  lane_read_at(limits,Tiled.cell_shape(values,source,source_position),groups_place(source_position,Tiled.cell_shape(values,source,source_position)),extent,block,room,extent_room,extent_positive,cells_room,cells_positive,cells_rank,blocks,below)

# The read of the source lane axis of a relane within the shape reads the
# source groups.
def lane_read_ok(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+within_true: {source_within == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits))) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Machine.half_word(limits)),ViewFacts.ReadOk(Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  +inside = Equal.trans(Bool,Nat.is_lt(source,List.length(&2,U32,values)),source_within,True{},source_value,within_true)
  lane_read_with(limits,values,source,source_within,source_position,extent,block,room,extent_room,extent_positive,cells_room,cells_positive,cells_rank,blocks,
    {%Equal.sym(Bool,Nat.is_lt(source,List.length(&2,U32,values)),True{},inside) : Held.Both<{Grid.lanes_at(Grid.placed(_,source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))),groups_place(source_position,Grid.placed(_,source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))))) == Grid.groups(Grid.lanes_at(values,source)) : U32},
    Held.Both<{Nat.is_lt(groups_place(source_position,Grid.placed(_,source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source)))),List.length(&2,U32,Grid.placed(_,source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))))) == True{} : Bool},
      {Nat.is_lt(0n,List.length(&2,U32,Grid.placed(_,source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))))) == True{} : Bool}>>
      placed_groups(source_position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))) : Held.Both<{Grid.lanes_at(Tiled.cell_shape(values,source,source_position),groups_place(source_position,Tiled.cell_shape(values,source,source_position))) == Grid.groups(Grid.lanes_at(values,source)) : U32},
    Held.Both<{Nat.is_lt(groups_place(source_position,Tiled.cell_shape(values,source,source_position)),List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
      {Nat.is_lt(0n,List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>>})

def some_choice(choice: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +sized: {extent == Grid.lanes_at(values,axis) : U32},+axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +choice_value: {Bool.and(source_within,Nat.is_eq(axis,source)) == choice : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits))) ->
  Held.Both<Components.ReadingRoom(extent,block,Bool.pick(Tasks.Source,choice,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),1,1,1)),Machine.half_word(limits)),ViewFacts.ReadOk(Bool.pick(Tasks.Source,choice,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32)),Tasks.nth(Tiled.cell_shape(values,source,source_position),Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,Tiled.cell_shape(values,source,source_position))) - 1 : U32))),1,1,8),Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,axis)),Grid.lanes_at(values,axis),1,1,1)),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  match choice:
    case True{}:
      lane_read_ok(limits,values,source,source_within,source_position,extent,block,room,extent_room,extent_positive,source_value,and_left(source_within,Nat.is_eq(axis,source),choice_value),cells_room,cells_positive,cells_rank,blocks)
    case False{}:
      shared_read_ok(limits,values,source,source_within,source_position,axis,extent,block,room,extent_room,extent_positive,sized,cells_rank,blocks,
        cell_lanes(values,source,source_within,source_position,axis,source_value,choice_value,axis_in))

# The read of a logical axis of a relane has room at half a word and reads
# its source cell axis.
def some_read(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat,+extent: U32,+block: U32,+room: Components.AxisRoom(extent,block,Machine.half_word(limits)),
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +sized: {extent == Grid.lanes_at(values,axis) : U32},+axis_in: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits))) ->
  Held.Both<Components.ReadingRoom(extent,block,Tiled.logical_reading(source,source_within,source_position,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),axis,Grid.lanes_at(values,axis)),Machine.half_word(limits)),ViewFacts.ReadOk(Tiled.logical_reading(source,source_within,source_position,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),axis,Grid.lanes_at(values,axis)),Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()))>:
  some_choice(Bool.and(source_within,Nat.is_eq(axis,source)),limits,values,source,source_within,source_position,axis,extent,block,room,extent_room,extent_positive,sized,axis_in,source_value,{==},cells_room,cells_positive,cells_rank,blocks)

# The readings of a relane have room at half a word and read the source
# cells.
def relane_reads(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +target: Nat,+target_grid: Grid.Grid,+index: U32,
  +sized: Reads.AxesExtents(axes,extents,values,target),+inside: AxesIn(axes,values,target),
  +target_blocks: ViewReads.BlocksIn(target_grid,extents,index,Machine.half_word(limits)),
  +extents_room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+extents_positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +target_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits))) ->
  ViewFacts.ReadsOk(Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,axes,extents,index),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target())):
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Held.Both{+axis_in,+later_in} = inside
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = target_blocks
      +inner = ShapeLaws.elements(more)
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,extents_positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,extents_positive)
      +extent_room = Order.transitive(U32.to_nat(extent),whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),extents_room)
      +block = Grid.block(target_grid,index,extent)
      +room = ViewFacts.axis_room(limits,extent,block,extent_room,block_positive,ViewFacts.block_within(limits,block,block_positive,block_double))
      ViewFacts.reads_cons(extent,block,Tiled.logical_reading(source,source_within,source_position,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),axis,Grid.lanes_at(values,axis)),
        Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,rest,more,(index + 1 : U32)),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),
        some_read(limits,values,source,source_within,source_position,axis,extent,block,room,extent_room,extent_positive,here,axis_in,source_value,cells_room,cells_positive,cells_rank,blocks),
        relane_reads(rest,more,limits,values,source,source_within,source_position,target,target_grid,(index + 1 : U32),later,later_in,blocks_rest,
          Order.transitive(inner,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),extents_room),inner_positive,
          source_value,apart,target_room,cells_room,cells_positive,cells_rank,blocks))
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Held.Both{+target_in,+later_in} = inside
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = target_blocks
      +inner = ShapeLaws.elements(more)
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,extents_positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,extents_positive)
      +extent_room = Order.transitive(U32.to_nat(extent),whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),extents_room)
      +block = Grid.block(target_grid,index,extent)
      +room = ViewFacts.axis_room(limits,extent,block,extent_room,block_positive,ViewFacts.block_within(limits,block,block_positive,block_double))
      ViewFacts.reads_cons(extent,block,Tiled.read_at(Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),U32.from_nat(cell_place(source,source_within,source_position,target)),Grid.lanes_at(values,target),8,8,1),
        Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,rest,more,(index + 1 : U32)),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),
        groups_read_ok(limits,values,source,source_within,source_position,target,extent,block,room,extent_positive,here,target_room,cells_rank,blocks,
          cell_lanes(values,source,source_within,source_position,target,source_value,apart,target_in)),
        relane_reads(rest,more,limits,values,source,source_within,source_position,target,target_grid,(index + 1 : U32),later,later_in,blocks_rest,
          Order.transitive(inner,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),extents_room),inner_positive,
          source_value,apart,target_room,cells_room,cells_positive,cells_rank,blocks))
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+extent,+more}:
      Empty.absurd(ViewFacts.ReadsOk(Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,Nil{},Con{extent,more},index),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target())),sized)
    case Con{Tiled.OfAxis{+axis},+rest} Nil{}:
      Empty.absurd(ViewFacts.ReadsOk(Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,Con{Tiled.OfAxis{axis},rest},Nil{},index),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target())),sized)
    case Con{Tiled.OfGroups{},+rest} Nil{}:
      Empty.absurd(ViewFacts.ReadsOk(Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),target_grid,values,Con{Tiled.OfGroups{},rest},Nil{},index),limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target())),sized)

# ---- What a relane the runtime accepts gives its gather ----

# A relane whose shape, source and target cells fit the machine gives its
# gather every fact it needs.
def relane_facts(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  Gather.RelaneFacts(limits,values,source,source_position,source_within,target,position,within):
  +source_rank = Piece.cells_rank_room(limits,values,source,source_position,source_within,source_value,rank_room)
  +target_rank = Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room)
  +blocks = ViewFacts.grid_blocks(limits,Tiled.cell_shape(values,source,source_position),Grid.piece_target(),Held.Both{source_cells_room,Held.Both{source_nonempty,source_rank}})
  +sized = Reads.target_extents(within,position,target,values,within_value)
  +ok = relane_reads(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),Tiled.cell_shape(values,target,position),limits,values,source,source_within,source_position,target,Grid.grid(Tiled.cell_shape(values,target,position),Grid.piece_target()),0,sized,axes_in(within,position,target,values,within_value),
    ViewFacts.grid_blocks(limits,Tiled.cell_shape(values,target,position),Grid.piece_target(),Held.Both{cells_room,Held.Both{nonempty,target_rank}}),cells_room,nonempty,
    source_value,apart,lanes_room,source_cells_room,source_nonempty,source_rank,blocks)
  +row = Tiled.relane_readings(source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Grid.grid(Tiled.cell_shape(values,target,position),Grid.piece_target()),values,Tiled.target_axes(within,position,target,List.length(&2,U32,values)),Tiled.cell_shape(values,target,position),0)
  +cells_same = relane_row_cells(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),Tiled.cell_shape(values,target,position),source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Grid.grid(Tiled.cell_shape(values,target,position),Grid.piece_target()),values,0,sized)
  +row_room = {%Equal.sym(Nat,ViewFacts.RowCells(row),ShapeLaws.elements(Tiled.cell_shape(values,target,position)),cells_same) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} cells_room :
    {Nat.is_le(ViewFacts.RowCells(row),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +row_positive = {%Equal.sym(Nat,ViewFacts.RowCells(row),ShapeLaws.elements(Tiled.cell_shape(values,target,position)),cells_same) : {Nat.is_lt(0n,_) == True{} : Bool} nonempty :
    {Nat.is_lt(0n,ViewFacts.RowCells(row)) == True{} : Bool}}
  +components = ViewFacts.row_components(row,limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),ok,ViewFacts.row_classes(row,limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),ok,row_room,row_positive),row_room)
  +row_ok = {Held.Both{ViewFacts.reads_room(row,limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),ok),Held.Both{ViewFacts.reads_axes(Tiled.cell_shape(values,source,source_position),row,limits,Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),0,ViewFacts.reads_from(row,limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),ok),blocks,source_cells_room,source_nonempty),
      ViewFacts.reads_below(row,limits,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),ok)}} : Components.RowOk(limits,row,Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position))}
  Held.Both{within_value,Held.Both{inside,Held.Both{room,Held.Both{lanes_room,Held.Both{cells_room,Held.Both{nonempty,Held.Both{source_value,Held.Both{source_inside,Held.Both{source_room,
    Held.Both{source_cells_room,Held.Both{source_lanes_room,Held.Both{rank_room,Held.Both{apart,Held.Both{source_nonempty,Held.Both{blocks,
    Held.Both{ViewFacts.components_part(row,limits,components),Held.Both{ViewFacts.positive_part(row,limits,components),
    Held.Both{relane_limits(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),Tiled.cell_shape(values,target,position),source,source_within,source_position,target,Tiled.cell_shape(values,source,source_position),Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Grid.grid(Tiled.cell_shape(values,target,position),Grid.piece_target()),values,0,sized),
      ViewFacts.row_when(Tasks.separable([row]),limits,row,Grid.grid(Tiled.cell_shape(values,source,source_position),Grid.piece_target()),Tiled.cell_shape(values,source,source_position),row_ok)}}}}}}}}}}}}}}}}}}

# ---- What counted values give a relane ----

def and_false(left: Bool) -> {Bool.and(left,False{}) == False{} : Bool}:
  match left:
    case False{}: {==}
    case True{}: {==}

def accepted_parts(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+target: Nat,+position: Grid.Position,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +apart: {Bool.and(Nat.is_lt(source,List.length(&2,U32,values)),Nat.is_eq(target,source)) == False{} : Bool},
  +source_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}>>>,
  +target_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool}>>>) ->
  Gather.RelaneFacts(limits,values,source,source_position,Nat.is_lt(source,List.length(&2,U32,values)),target,position,Nat.is_lt(target,List.length(&2,U32,values))):
  KernelGemm.Both{+source_room,KernelGemm.Both{+source_lanes_room,KernelGemm.Both{+source_cells_room,+source_nonempty}}} = source_bounds
  KernelGemm.Both{+room,KernelGemm.Both{+lanes_room,KernelGemm.Both{+cells_room,+nonempty}}} = target_bounds
  relane_facts(limits,values,source,source_position,Nat.is_lt(source,List.length(&2,U32,values)),target,position,Nat.is_lt(target,List.length(&2,U32,values)),{==},inside,room,lanes_room,cells_room,nonempty,{==},source_inside,
    source_room,source_cells_room,source_lanes_room,rank_room,apart,source_nonempty)

# A relane of values within a capacity with at least one value and a rank
# within it, onto an axis other than its own and at most the rank, gives its
# gather every fact it needs.
def relane_accepted(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+target: Nat,+position: Grid.Position,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +apart: {Nat.is_eq(target,source) == False{} : Bool}) ->
  Gather.RelaneFacts(limits,values,source,source_position,Nat.is_lt(source,List.length(&2,U32,values)),target,position,Nat.is_lt(target,List.length(&2,U32,values))):
  accepted_parts(limits,values,source,source_position,target,position,rank_room,source_inside,inside,
    {%Equal.sym(Bool,Nat.is_eq(target,source),False{},apart) : {Bool.and(Nat.is_lt(source,List.length(&2,U32,values)),_) == False{} : Bool} and_false(Nat.is_lt(source,List.length(&2,U32,values))) :
      {Bool.and(Nat.is_lt(source,List.length(&2,U32,values)),Nat.is_eq(target,source)) == False{} : Bool}},
    Coordinates.axis_cells(Nat.is_lt(source,List.length(&2,U32,values)),limits,values,source,source_position,{==},room,positive),
    Coordinates.axis_cells(Nat.is_lt(target,List.length(&2,U32,values)),limits,values,target,position,{==},room,positive))
