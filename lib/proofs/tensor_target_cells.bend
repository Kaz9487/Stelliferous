# A cell of a target piece of a component lies inside the classes of the
# component: on every axis its coordinate lies in the target cells of the
# class the component takes there.

import Base
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_ceiling.bend as NatCeiling
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as Extrema
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Logic
import ./tensor_components.bend as Components
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_box.bend as Box
import ./tensor_box_cells.bend as BoxCells
import ./tensor_grid_ranges.bend as GridRanges
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds

# ---- The classes a target cell lies in ----

# The row-major coordinates of a target index.
def cell_coordinates(extents: List<&2,U32>,+index: Nat) -> List<&2,Nat>:
  match extents:
    case Nil{}: Nil{}
    case Con{extent,+rest}:
      Con{Nat.div(index,ShapeLaws.elements(rest)),cell_coordinates(rest,Nat.mod(index,ShapeLaws.elements(rest)))}

# A target coordinate inside the cells of a class.
def ClassIn(class: Tasks.Class,+block: U32,+extent: U32,+coordinate: Nat) -> Data:
  match class:
    case Tasks.Class{+first,+count,source_first,source_count}:
      Held.Both<{Nat.is_le(U32.to_nat((first * block : U32)),coordinate) == True{} : Bool},
        {Nat.is_lt(coordinate,U32.to_nat(U32.min(((first + count) * block : U32),extent))) == True{} : Bool}>

# Every target coordinate lies inside the cells of the class of its axis.
def ClassesIn(classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,cells: List<&2,Nat>,+axis: U32) -> Data:
  match classes extents cells:
    case Con{+class,classes} Con{+extent,extents} Con{+cell,cells}:
      Held.Both<ClassIn(class,Grid.block(grid,axis,extent),extent,cell),ClassesIn(classes,grid,extents,cells,(axis + 1 : U32))>
    case Nil{} Nil{} Nil{}:
      Unit
    case classes extents cells:
      Empty

# Every class ends inside the blocks of its axis, and the axis and its block
# fit the word.
def ClassEnds(classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,+axis: U32,+cap: U32) -> Data:
  match classes extents:
    case Con{+class,classes} Con{+extent,extents}:
      Held.Both<Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent)))),
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,axis,extent))),U32.to_nat(cap)) == True{} : Bool},
          ClassEnds(classes,grid,extents,(axis + 1 : U32),cap)>>
    case Nil{} Nil{}:
      Unit
    case classes extents:
      Empty

# The blocks that cover an axis fit the word with one more block.
def ceiling_room(+extent: Nat,+block: Nat,+blocks: Nat,+cap: Nat,+exact: {NatCeiling.ceiling(extent,block) == blocks : Nat},
  +extent_positive: {Nat.is_lt(0n,extent) == True{} : Bool},+block_positive: {Nat.is_lt(0n,block) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(extent,block),cap) == True{} : Bool}) -> {Nat.is_le(Nat.mul(blocks,block),cap) == True{} : Bool}:
  match blocks:
    case 0n:
      Order.zero_le(cap)
    case 1n+ +before:
      +previous = {%exact : {Nat.is_lt(Nat.mul(Nat.sub(_,1n),block),extent) == True{} : Bool}
        NatCeiling.previous_below(extent,block,extent_positive,block_positive) :
        {Nat.is_lt(Nat.mul(Nat.sub(1n+before,1n),block),extent) == True{} : Bool}}
      +below = {%Order.subtract_zero(before) : {Nat.is_lt(Nat.mul(_,block),extent) == True{} : Bool} previous :
        {Nat.is_lt(Nat.mul(before,block),extent) == True{} : Bool}}
      Order.transitive(Nat.add(block,Nat.mul(before,block)),Nat.add(block,extent),cap,
        Order.offset_le(block,Nat.mul(before,block),extent,Bounds.strict_le(Nat.mul(before,block),extent,below)),
        {%Order.commutative(extent,block) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.add(block,extent),cap) == True{} : Bool}})

# A coordinate whose block lies in the blocks of a class lies in its cells.
def class_block(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+block: U32,+extent: U32,+coordinate: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(first),Nat.div(coordinate,U32.to_nat(block))) == True{} : Bool},
  +high: {Nat.is_lt(Nat.div(coordinate,U32.to_nat(block)),U32.to_nat((first + count : U32))) == True{} : Bool}) ->
  ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate):
  Held.Both{+starts,+ending} = ends
  +size = U32.to_nat(block)
  +total = U32.to_nat(extent)
  +start = U32.to_nat(first)
  +reach = U32.to_nat((first + count : U32))
  +quotient = Nat.div(coordinate,size)
  +total_room = Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room)
  +low_cell = Order.transitive(Nat.mul(start,size),Nat.mul(quotient,size),coordinate,RowMajor.mul_le(start,quotient,size,low),
    DivisionOrder.product_below(coordinate,size,block_positive))
  +low_exact = Arithmetic.product_exact(cap,cap_ok,first,block,start,size,{==},{==},
    Order.transitive(Nat.mul(start,size),coordinate,U32.to_nat(cap),low_cell,Order.transitive(coordinate,total,U32.to_nat(cap),Bounds.strict_le(coordinate,total,inside),total_room)))
  +extent_positive = Order.le_lt_trans(0n,coordinate,total,Order.zero_le(coordinate),inside)
  +reach_room = Order.transitive(Nat.mul(reach,size),Nat.mul(NatCeiling.ceiling(total,size),size),U32.to_nat(cap),RowMajor.mul_le(reach,NatCeiling.ceiling(total,size),size,ending),
    ceiling_room(total,size,NatCeiling.ceiling(total,size),U32.to_nat(cap),{==},extent_positive,block_positive,room))
  +reach_exact = Arithmetic.product_exact(cap,cap_ok,(first + count : U32),block,reach,size,{==},{==},reach_room)
  +high_cell = Bounds.nat_trans(coordinate,Nat.add(size,Nat.mul(quotient,size)),Nat.mul(reach,size),DivisionOrder.next_above(coordinate,size,block_positive),
    RowMajor.mul_le(1n+quotient,reach,size,Order.strict_follow(quotient,reach,high)))
  +limit = ((first + count) * block : U32)
  Held.Both{{%Equal.sym(Nat,U32.to_nat((first * block : U32)),Nat.mul(start,size),low_exact) : {Nat.is_le(_,coordinate) == True{} : Bool} low_cell :
      {Nat.is_le(U32.to_nat((first * block : U32)),coordinate) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.min(limit,extent)),Nat.min(U32.to_nat(limit),total),Extrema.minimum(limit,extent)) : {Nat.is_lt(coordinate,_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(limit),Nat.mul(reach,size),reach_exact) : {Nat.is_lt(coordinate,Nat.min(_,total)) == True{} : Bool}
      NatExtrema.minimum_strict_lower(Nat.mul(reach,size),total,coordinate,high_cell,inside) :
      {Nat.is_lt(coordinate,U32.to_nat(U32.min(limit,extent))) == True{} : Bool}}}

def ends_starts(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+limit: Nat,
  +ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},limit)) -> {Nat.is_lt(U32.to_nat(first),U32.to_nat((first + count : U32))) == True{} : Bool}:
  Held.Both{+starts,+ending} = ends
  starts

def ends_reach(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+limit: Nat,
  +ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},limit)) -> {Nat.is_le(U32.to_nat((first + count : U32)),limit) == True{} : Bool}:
  Held.Both{+starts,+ending} = ends
  ending

# A start below an end at most one is zero.
def start_zero(+start: Nat,+reach: Nat,+limit: Nat,+starts: {Nat.is_lt(start,reach) == True{} : Bool},+ending: {Nat.is_le(reach,limit) == True{} : Bool},
  +one: {Nat.is_le(limit,1n) == True{} : Bool}) -> {Nat.is_le(start,0n) == True{} : Bool}:
  Division.lt_succ_le(start,0n,Bounds.nat_trans(start,reach,1n,starts,Order.transitive(reach,limit,1n,ending,one)))

# A class of an axis that is one block covers the whole axis.
def class_whole(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+extent: U32,+coordinate: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(extent))),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) -> ClassIn(Tasks.Class{first,count,source_first,source_count},extent,extent,coordinate):
  +total = U32.to_nat(extent)
  +positive = Order.le_lt_trans(0n,coordinate,total,Order.zero_le(coordinate),inside)
  +one = NatCeiling.minimal(total,total,1n,positive,Order.le_add(total,0n))
  +zero = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(coordinate,total),(0n,coordinate),Division.small(coordinate,total,inside))
  class_block(first,count,source_first,source_count,extent,extent,coordinate,cap,cap_ok,ends,room,positive,inside,
    {%Equal.sym(Nat,Nat.div(coordinate,total),0n,zero) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
      start_zero(U32.to_nat(first),U32.to_nat((first + count : U32)),NatCeiling.ceiling(total,total),ends_starts(first,count,source_first,source_count,
        NatCeiling.ceiling(total,total),ends),ends_reach(first,count,source_first,source_count,NatCeiling.ceiling(total,total),ends),one) :
      {Nat.is_le(U32.to_nat(first),Nat.div(coordinate,total)) == True{} : Bool}},
    {%Equal.sym(Nat,Nat.div(coordinate,total),0n,zero) : {Nat.is_lt(_,U32.to_nat((first + count : U32))) == True{} : Bool}
      Order.le_lt_trans(0n,U32.to_nat(first),U32.to_nat((first + count : U32)),Order.zero_le(U32.to_nat(first)),
        ends_starts(first,count,source_first,source_count,NatCeiling.ceiling(total,total),ends)) :
      {Nat.is_lt(Nat.div(coordinate,total),U32.to_nat((first + count : U32))) == True{} : Bool}})

# ---- The pieces a box lists lie inside it ----

# An index inside a box of extents and below their cells.
def InExtents(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+value: Nat) -> Data:
  Held.Both<Components.InBox(spans,extents,value),{Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}>

# One more axis whose digit lies in its span.
def row_step(+first: U32,+count: U32,+rest_spans: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+digit: Nat,+tail: Nat,
  +low: {Nat.is_le(U32.to_nat(first),digit) == True{} : Bool},+high: {Nat.is_lt(digit,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},
  +small: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+later: InExtents(rest_spans,rest,tail)) ->
  InExtents(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},Nat.add(Nat.mul(digit,ShapeLaws.elements(rest)),tail)):
  Held.Both{+box,+below} = later
  +inner = ShapeLaws.elements(rest)
  +inner_positive = Order.le_lt_trans(0n,tail,inner,Order.zero_le(tail),below)
  +whole = Nat.add(Nat.mul(digit,inner),tail)
  +pair = Division.unique(whole,inner,digit,tail,{==},inner_positive,below)
  +quotient = Equal.cong(Nat & Nat,Nat,found => Pair.fst(Nat,Nat,found),Nat.divmod(whole,inner),(digit,tail),pair)
  +remainder = Equal.cong(Nat & Nat,Nat,found => Pair.snd(Nat,Nat,found),Nat.divmod(whole,inner),(digit,tail),pair)
  Held.Both{
    Held.Both{{%Equal.sym(Nat,Nat.div(whole,inner),digit,quotient) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} low :
        {Nat.is_le(U32.to_nat(first),Nat.div(whole,inner)) == True{} : Bool}},
      Held.Both{{%Equal.sym(Nat,Nat.div(whole,inner),digit,quotient) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool} high :
          {Nat.is_lt(Nat.div(whole,inner),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}},
        {%Equal.sym(Nat,Nat.mod(whole,inner),tail,remainder) : Components.InBox(rest_spans,rest,_) box : Components.InBox(rest_spans,rest,Nat.mod(whole,inner))}}},
    RowMajor.radix(digit,U32.to_nat(extent),tail,inner,small,below)}

# The row-major value at a position of a box lies inside the box.
def row_inside(spans: List<&2,Tasks.Span>,extents: List<&2,U32>,+index: Nat,+within: Components.SpansIn(spans,extents),
  +below: {Nat.is_lt(index,Components.span_total(spans)) == True{} : Bool}) -> InExtents(spans,extents,Components.row_value(spans,extents,index,0n)):
  match spans extents:
    case Con{Tasks.Span{+first,+count},+rest_spans} Con{+extent,+rest}:
      Held.Both{+fits,+rest_within} = within
      +inner = Components.span_total(rest_spans)
      +inner_positive = Components.inner_positive(U32.to_nat(count),inner,index,below)
      +quotient = Nat.div(index,inner)
      +digit = Nat.add(U32.to_nat(first),quotient)
      +remainder = Nat.mod(index,inner)
      +quotient_below = DivisionOrder.quotient_below(index,U32.to_nat(count),inner,inner_positive,below)
      +high = Order.offset_lt(U32.to_nat(first),quotient,U32.to_nat(count),quotient_below)
      +later = row_inside(rest_spans,rest,remainder,rest_within,DivisionOrder.remainder_small(index,inner,inner_positive))
      %Equal.sym(Nat,Components.row_value(rest_spans,rest,remainder,digit),Nat.add(Nat.mul(digit,ShapeLaws.elements(rest)),Components.row_value(rest_spans,rest,remainder,0n)),
          Components.row_split(rest_spans,rest,remainder,digit,rest_within)) : InExtents(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},_)
      row_step(first,count,rest_spans,extent,rest,digit,Components.row_value(rest_spans,rest,remainder,0n),Order.le_add(U32.to_nat(first),quotient),high,
        Bounds.nat_trans(digit,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent),high,fits),later)
    case Nil{} Nil{}:
      Held.Both{Unit{},{==}}
    case Nil{} Con{+extent,+rest}:
      Empty.absurd(InExtents(Nil{},Con{extent,rest},Components.row_value(Nil{},Con{extent,rest},index,0n)),within)
    case Con{Tasks.Span{+first,+count},+rest_spans} Nil{}:
      Empty.absurd(InExtents(Con{Tasks.Span{first,count},rest_spans},Nil{},Components.row_value(Con{Tasks.Span{first,count},rest_spans},Nil{},index,0n)),within)

# A member of the pieces a box lists lies inside the box.
def member_in(list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+from: Nat,+piece: U32,
  +listed: Components.ListedFrom(list,spans,extents,from),
  +bound: {Nat.is_le(Nat.add(from,List.length(&2,U32,list)),Components.span_total(spans)) == True{} : Bool},
  +within: Components.SpansIn(spans,extents),+member: Moved.Member(list,piece)) -> InExtents(spans,extents,U32.to_nat(piece)):
  match list:
    case Nil{}:
      Empty.absurd(InExtents(spans,extents,U32.to_nat(piece)),member)
    case Con{+first,+rest}:
      Held.Both{+value,+later} = listed
      match member:
        case Placed.OnLeft{+same}:
          +length = List.length(&2,U32,rest)
          +shifted = {%Addition.successor_right(from,length) : {Nat.is_le(_,Components.span_total(spans)) == True{} : Bool}
            bound : {Nat.is_le(1n+Nat.add(from,length),Components.span_total(spans)) == True{} : Bool}}
          %same : InExtents(spans,extents,U32.to_nat(_))
          %Equal.sym(Nat,U32.to_nat(first),Components.row_value(spans,extents,from,0n),value) : InExtents(spans,extents,_)
          row_inside(spans,extents,from,within,Order.succ_le_lt(from,Components.span_total(spans),
            Order.transitive(1n+from,1n+Nat.add(from,length),Components.span_total(spans),Order.le_add(1n+from,length),shifted)))
        case Placed.OnRight{+others}:
          +length = List.length(&2,U32,rest)
          +shifted = {%Addition.successor_right(from,length) : {Nat.is_le(_,Components.span_total(spans)) == True{} : Bool}
            bound : {Nat.is_le(1n+Nat.add(from,length),Components.span_total(spans)) == True{} : Bool}}
          member_in(rest,spans,extents,1n+from,piece,later,shifted,within,others)

# ---- The piece of a cell of a piece ----

# The piece after one starts at most one block after it.
def next_within(choice: Bool,+outer: Nat,+inner: Nat,+per: Nat,+extent: Nat,+block: Nat,+per_exact: {per == NatCeiling.ceiling(extent,block) : Nat},
  +positive: {Nat.is_lt(0n,block) == True{} : Bool},+test: {Nat.is_lt(1n+inner,per) == choice : Bool}) ->
  {Nat.is_le(Box.piece_base(Box.next_outer(choice,outer),Box.next_inner(choice,inner),extent,block),Nat.add(block,Box.piece_base(outer,inner,extent,block))) == True{} : Bool}:
  match choice:
    case True{}:
      +lead = Nat.mul(outer,extent)
      +part = Nat.mul(inner,block)
      %Equal.sym(Nat,Nat.add(lead,Nat.add(block,part)),Nat.add(block,Nat.add(lead,part)),RowMajor.add_swap(lead,block,part)) :
        {Nat.is_le(_,Nat.add(block,Nat.add(lead,part))) == True{} : Bool}
      Order.reflexive(Nat.add(block,Nat.add(lead,part)))
    case False{}:
      +lead = Nat.mul(outer,extent)
      +part = Nat.mul(inner,block)
      +target = Nat.add(block,Nat.add(lead,part))
      +per_le = Order.not_less(1n+inner,per,test)
      +covered = {%Equal.sym(Nat,per,NatCeiling.ceiling(extent,block),per_exact) : {Nat.is_le(extent,Nat.mul(_,block)) == True{} : Bool}
        NatCeiling.covers(extent,block,positive) : {Nat.is_le(extent,Nat.mul(per,block)) == True{} : Bool}}
      +extent_le = Order.transitive(extent,Nat.mul(per,block),Nat.add(block,part),covered,RowMajor.mul_le(per,1n+inner,block,per_le))
      +regrouped = Equal.trans(Nat,Nat.add(Nat.add(block,part),lead),Nat.add(block,Nat.add(part,lead)),target,Addition.associative(block,part,lead),
        Equal.cong(Nat,Nat,sum => Nat.add(block,sum),Nat.add(part,lead),Nat.add(lead,part),Order.commutative(part,lead)))
      %Equal.sym(Nat,Nat.add(Nat.add(extent,lead),0n),Nat.add(extent,lead),Addition.zero_right(Nat.add(extent,lead))) : {Nat.is_le(_,target) == True{} : Bool}
      %regrouped : {Nat.is_le(Nat.add(extent,lead),_) == True{} : Bool}
      Order.add_le(extent,Nat.add(block,part),lead,extent_le)

# A cell of a piece lies in that piece and among the cells of the grid.
def cell_of_piece(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))) == True{} : Bool}) ->
  Held.Both<{BoxCells.cell_piece(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),index),
      Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per)) == piece : Nat},
    {Nat.is_lt(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),index),U32.to_nat(count)) == True{} : Bool}>:
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +size = U32.to_nat(extent)
  +units = U32.to_nat(unit_cells)
  +blocks = U32.to_nat(block)
  +per_count = U32.to_nat(per)
  +image = Nat.mul(size,units)
  +piece_cells = Nat.mul(blocks,units)
  +image_positive = Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape)
  +cells_positive = Laws.cells_positive(limits,outer,extent,unit_cells,block,per,shape)
  +block_positive = Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape)
  +per_positive = Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)
  +per_exact = Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape)
  +quotient = Nat.div(piece,per_count)
  +inner = Nat.mod(piece,per_count)
  +choice = Nat.is_lt(1n+inner,per_count)
  +base = Box.piece_base(quotient,inner,size,blocks)
  +next = Box.piece_base(Box.next_outer(choice,quotient),Box.next_inner(choice,inner),size,blocks)
  +start = U32.to_nat(Grid.start(grid,U32.from_nat(piece)))
  +start_place = Equal.trans(Nat,start,Nat.mul(base,units),Laws.place(quotient,inner,image,piece_cells),
    Box.start_base(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,inside),
    Equal.sym(Nat,Laws.place(quotient,inner,image,piece_cells),Nat.mul(base,units),Box.place_units(quotient,inner,size,blocks,units)))
  +room = {%RowMajor.assoc(inner,blocks,units) : {Nat.is_lt(Nat.add(_,index),image) == True{} : Bool}
    Box.piece_room(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside,index,index_in) :
    {Nat.is_lt(Nat.add(Nat.mul(inner,piece_cells),index),image) == True{} : Bool}}
  +difference_le = Order.transitive(Nat.sub(next,base),Nat.sub(Nat.add(blocks,base),base),blocks,
    DivisionOrder.subtract_le(next,Nat.add(blocks,base),base,
      next_within(choice,quotient,inner,per_count,size,blocks,per_exact,block_positive,{==})),
    {%Equal.sym(Nat,Nat.sub(Nat.add(blocks,base),base),blocks,DivisionOrder.subtract_sum(blocks,base)) : {Nat.is_le(_,blocks) == True{} : Bool} Order.reflexive(blocks) :
      {Nat.is_le(Nat.sub(Nat.add(blocks,base),base),blocks) == True{} : Bool}})
  +index_small = {%Box.piece_span(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside) : {Nat.is_lt(index,_) == True{} : Bool}
    index_in : {Nat.is_lt(index,Nat.mul(Nat.sub(next,base),units)) == True{} : Bool}}
  +index_cells = Bounds.nat_trans(index,Nat.mul(Nat.sub(next,base),units),piece_cells,index_small,RowMajor.mul_le(Nat.sub(next,base),blocks,units,difference_le))
  +rest = Nat.add(Nat.mul(inner,piece_cells),index)
  +value = Nat.add(start,index)
  +value_exact = Equal.trans(Nat,value,Nat.add(Laws.place(quotient,inner,image,piece_cells),index),Nat.add(Nat.mul(quotient,image),rest),
    Equal.cong(Nat,Nat,head => Nat.add(head,index),start,Laws.place(quotient,inner,image,piece_cells),start_place),
    Addition.associative(Nat.mul(quotient,image),Nat.mul(inner,piece_cells),index))
  +outer_pair = Division.unique(value,image,quotient,rest,value_exact,image_positive,room)
  +inner_pair = Division.unique(rest,piece_cells,inner,index,{==},cells_positive,index_cells)
  +same = {%Equal.sym(Nat,Nat.div(value,image),quotient,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,image),(quotient,rest),outer_pair)) :
      {Nat.add(Nat.mul(_,per_count),Nat.div(Nat.mod(value,image),piece_cells)) == piece : Nat}
    %Equal.sym(Nat,Nat.mod(value,image),rest,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,image),(quotient,rest),outer_pair)) :
      {Nat.add(Nat.mul(quotient,per_count),Nat.div(_,piece_cells)) == piece : Nat}
    %Equal.sym(Nat,Nat.div(rest,piece_cells),inner,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(rest,piece_cells),(inner,index),inner_pair)) :
      {Nat.add(Nat.mul(quotient,per_count),_) == piece : Nat}
    Equal.sym(Nat,piece,Nat.add(Nat.mul(quotient,per_count),inner),DivisionOrder.quotient_exact(piece,per_count,per_positive)) :
    {BoxCells.cell_piece(value,image,piece_cells,per_count) == piece : Nat}}
  +outers = U32.to_nat(outer)
  +below = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(outers,size),units),cells) : {Nat.is_lt(value,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(outers,size),units),Nat.mul(outers,image),RowMajor.assoc(outers,size,units)) : {Nat.is_lt(value,_) == True{} : Bool}
    %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,image),rest),value_exact) : {Nat.is_lt(_,Nat.mul(outers,image)) == True{} : Bool}
    RowMajor.radix(quotient,outers,rest,image,DivisionOrder.quotient_below(piece,outers,per_count,per_positive,inside),room) :
    {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool}}
  Held.Both{same,below}

# ---- The classes of the cells of a piece ----

# A class of an axis that is its own block holds every coordinate.
def class_axis_whole(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+grid: Grid.Grid,+at: U32,+extent: U32,+coordinate: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+same_block: {Grid.block(grid,at,extent) == extent : U32},
  +ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent)))),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}) ->
  ClassIn(Tasks.Class{first,count,source_first,source_count},Grid.block(grid,at,extent),extent,coordinate):
  %Equal.sym(U32,Grid.block(grid,at,extent),extent,same_block) : ClassIn(Tasks.Class{first,count,source_first,source_count},_,extent,coordinate)
  class_whole(first,count,source_first,source_count,extent,coordinate,cap,cap_ok,
    {%same_block : Components.EndIn(Tasks.Class{first,count,source_first,source_count},NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_))) ends :
      Components.EndIn(Tasks.Class{first,count,source_first,source_count},NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(extent)))},
    {%same_block : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(_)),U32.to_nat(cap)) == True{} : Bool} room :
      {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}},
    inside)

# Every axis past the prefix holds a cell whole.
def past_classes(classes: List<&2,Tasks.Class>,extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+position: Nat,+value: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +past: {Nat.is_le(U32.to_nat(Components.grid_prefix(grid)),position) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +ends: ClassEnds(classes,grid,extents,at,cap),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  ClassesIn(classes,grid,extents,cell_coordinates(extents,value),at):
  match classes extents:
    case Con{Tasks.Class{+first,+count,+source_first,+source_count},+rest_classes} Con{+extent,+rest}:
      Held.Both{+end_in,+more} = ends
      Held.Both{+axis_room,+later_ends} = more
      +inner = ShapeLaws.elements(rest)
      +inner_positive = Components.inner_positive(U32.to_nat(extent),inner,value,below)
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
      +at_room = Order.succ_le_lt(position,U32.to_nat(cap),Order.transitive(1n+position,1n+Nat.add(position,length),U32.to_nat(cap),Order.le_add(1n+position,length),shifted))
      +next_exact = Components.next_word(at,position,cap,cap_ok,exact,Order.strict_follow(position,U32.to_nat(cap),at_room))
      +same_block = Components.past_block(grid,at,extent,cap,cap_ok,{%Equal.sym(Nat,U32.to_nat(at),position,exact) :
          {Nat.is_le(U32.to_nat(Components.grid_prefix(grid)),_) == True{} : Bool} past : {Nat.is_le(U32.to_nat(Components.grid_prefix(grid)),U32.to_nat(at)) == True{} : Bool}},
        {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool} at_room : {Nat.is_lt(U32.to_nat(at),U32.to_nat(cap)) == True{} : Bool}})
      Held.Both{class_axis_whole(first,count,source_first,source_count,grid,at,extent,Nat.div(value,inner),cap,cap_ok,same_block,end_in,axis_room,
          DivisionOrder.quotient_below(value,U32.to_nat(extent),inner,inner_positive,below)),
        past_classes(rest_classes,rest,grid,(at + 1 : U32),1n+position,Nat.mod(value,inner),cap,cap_ok,next_exact,
          Order.transitive(U32.to_nat(Components.grid_prefix(grid)),position,1n+position,past,Bounds.strict_le(position,1n+position,Order.lt_succ(position,position,Order.reflexive(position)))),
          shifted,later_ends,DivisionOrder.remainder_small(value,inner,inner_positive))}
    case Nil{} Nil{}:
      Unit{}
    case Nil{} Con{+extent,+rest}:
      Empty.absurd(ClassesIn(Nil{},grid,Con{extent,rest},cell_coordinates(Con{extent,rest},value),at),ends)
    case Con{+class,+rest_classes} Nil{}:
      Empty.absurd(ClassesIn(Con{class,rest_classes},grid,Nil{},cell_coordinates(Nil{},value),at),ends)

# The parts of an index inside a box whose first axis is spanned.
def box_low(+first: U32,+count: U32,+rest_spans: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+index: Nat,
  +inside: Components.InBox(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},index)) ->
  {Nat.is_le(U32.to_nat(first),Nat.div(index,ShapeLaws.elements(rest))) == True{} : Bool}:
  Held.Both{+low,+more} = inside
  low

def box_high(+first: U32,+count: U32,+rest_spans: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+index: Nat,
  +inside: Components.InBox(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},index)) ->
  {Nat.is_lt(Nat.div(index,ShapeLaws.elements(rest)),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}:
  Held.Both{+low,+more} = inside
  Held.Both{+high,+later} = more
  high

def box_rest(+first: U32,+count: U32,+rest_spans: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+index: Nat,
  +inside: Components.InBox(Con{Tasks.Span{first,count},rest_spans},Con{extent,rest},index)) ->
  Components.InBox(rest_spans,rest,Nat.mod(index,ShapeLaws.elements(rest))):
  Held.Both{+low,+more} = inside
  Held.Both{+high,+later} = more
  later

# The block of a prefix axis before the last is one index.
def kept_block(+grid: Grid.Grid,+at: U32,+extent: U32,+prefix: U32,+same_prefix: {Components.grid_prefix(grid) == prefix : U32},
  +kept: {U32.is_lt((at + 1 : U32),prefix) == True{} : Bool}) -> {Grid.block(grid,at,extent) == 1 : U32}:
  +units = Components.grid_units(grid)
  Equal.trans(U32,Grid.block(grid,at,extent),
    Bool.pick(U32,U32.is_lt((at + 1 : U32),Components.grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),Components.grid_prefix(grid)),units,extent)),1,
    Components.block_value(grid,at,extent),
    {%Equal.sym(U32,Components.grid_prefix(grid),prefix,same_prefix) :
        {Bool.pick(U32,U32.is_lt((at + 1 : U32),_),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),_),units,extent)) == 1 : U32}
      %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),True{},kept) : {Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),units,extent)) == 1 : U32}
      {==} :
      {Bool.pick(U32,U32.is_lt((at + 1 : U32),Components.grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),Components.grid_prefix(grid)),units,extent)) == 1 : U32}})

# The block of the last prefix axis is the units of the grid.
def last_block(+grid: Grid.Grid,+at: U32,+extent: U32,+prefix: U32,+units: U32,+same_prefix: {Components.grid_prefix(grid) == prefix : U32},
  +same_units: {Components.grid_units(grid) == units : U32},+stop: {U32.is_lt((at + 1 : U32),prefix) == False{} : Bool},
  +last: {U32.is_eq((at + 1 : U32),prefix) == True{} : Bool}) -> {Grid.block(grid,at,extent) == units : U32}:
  Equal.trans(U32,Grid.block(grid,at,extent),
    Bool.pick(U32,U32.is_lt((at + 1 : U32),Components.grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),Components.grid_prefix(grid)),Components.grid_units(grid),extent)),
    units,Components.block_value(grid,at,extent),
    {%Equal.sym(U32,Components.grid_prefix(grid),prefix,same_prefix) :
        {Bool.pick(U32,U32.is_lt((at + 1 : U32),_),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),_),Components.grid_units(grid),extent)) == units : U32}
      %Equal.sym(U32,Components.grid_units(grid),units,same_units) :
        {Bool.pick(U32,U32.is_lt((at + 1 : U32),prefix),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),_,extent)) == units : U32}
      %Equal.sym(Bool,U32.is_lt((at + 1 : U32),prefix),False{},stop) : {Bool.pick(U32,_,1,Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),units,extent)) == units : U32}
      %Equal.sym(Bool,U32.is_eq((at + 1 : U32),prefix),True{},last) : {Bool.pick(U32,False{},1,Bool.pick(U32,_,units,extent)) == units : U32}
      {==} :
      {Bool.pick(U32,U32.is_lt((at + 1 : U32),Components.grid_prefix(grid)),1,Bool.pick(U32,U32.is_eq((at + 1 : U32),Components.grid_prefix(grid)),Components.grid_units(grid),extent)) ==
        units : U32}})

# A coordinate whose block lies in a class's blocks, given as a sum, lies in
# the class's cells.
def class_sum(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+block: U32,+extent: U32,+coordinate: Nat,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+ends: Components.EndIn(Tasks.Class{first,count,source_first,source_count},
    NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),
  +room: {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+inside: {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},
  +low: {Nat.is_le(U32.to_nat(first),Nat.div(coordinate,U32.to_nat(block))) == True{} : Bool},
  +high: {Nat.is_lt(Nat.div(coordinate,U32.to_nat(block)),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) ->
  ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate):
  +total = U32.to_nat(extent)
  +size = U32.to_nat(block)
  +limit = NatCeiling.ceiling(total,size)
  +limit_room = Order.transitive(limit,total,U32.to_nat(cap),NatCeiling.bounded(total,size,block_positive),
    Order.transitive(total,Nat.add(total,size),U32.to_nat(cap),Order.le_add(total,size),room))
  +sum = Components.end_exact(first,count,limit,cap,cap_ok,ends_starts(first,count,source_first,source_count,limit,ends),
    ends_reach(first,count,source_first,source_count,limit,ends),limit_room)
  class_block(first,count,source_first,source_count,block,extent,coordinate,cap,cap_ok,ends,room,block_positive,inside,low,
    {%sum : {Nat.is_lt(Nat.div(coordinate,size),_) == True{} : Bool} high : {Nat.is_lt(Nat.div(coordinate,size),U32.to_nat((first + count : U32))) == True{} : Bool}})

# The last prefix axis: the block of the cell there is the index of its
# piece, which lies in the span of the class.
def last_class(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+rest_classes: List<&2,Tasks.Class>,+extent: U32,+rest: List<&2,U32>,
  +grid: Grid.Grid,+at: U32,+position: Nat,+prefix: U32,+block: U32,+per: U32,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+0n) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +same_prefix: {Components.grid_prefix(grid) == prefix : U32},+same_units: {Components.grid_units(grid) == block : U32},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,Con{extent,rest})),U32.to_nat(cap)) == True{} : Bool},
  +ends: ClassEnds(Con{Tasks.Class{first,count,source_first,source_count},rest_classes},grid,Con{extent,rest},at,cap),
  +inside: Components.InBox(Tasks.target_spans(Con{Tasks.Class{first,count,source_first,source_count},rest_classes},prefix,at),
    Tasks.piece_extents(Con{extent,rest},prefix,per,at),
    BoxCells.cell_piece(value,BoxCells.image_of(Con{extent,rest},0n),BoxCells.cells_of(Con{extent,rest},0n,block),U32.to_nat(per))),
  +below: {Nat.is_lt(value,ShapeLaws.elements(Con{extent,rest})) == True{} : Bool}) ->
  ClassesIn(Con{Tasks.Class{first,count,source_first,source_count},rest_classes},grid,Con{extent,rest},cell_coordinates(Con{extent,rest},value),at):
  Held.Both{+end_in,+more} = ends
  Held.Both{+axis_room,+later_ends} = more
  +class = {Tasks.Class{first,count,source_first,source_count} : Tasks.Class}
  +stop = Components.last_lt(at,prefix,position,cap,cap_ok,exact,distance,prefix_room)
  +kept_at = Components.at_kept(at,prefix,position,0n,exact,distance)
  +later_spans = Tasks.target_spans(rest_classes,prefix,(at + 1 : U32))
  +spans = Equal.trans(List<&2,Tasks.Span>,Tasks.target_spans(Con{class,rest_classes},prefix,at),Con{Tasks.Span{first,count},later_spans},Con{Tasks.Span{first,count},Nil{}},
    {%Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept_at) :
        {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},later_spans},Nil{}) == Con{Tasks.Span{first,count},later_spans} : List<&2,Tasks.Span>}
      {==} :
      {Tasks.target_spans(Con{class,rest_classes},prefix,at) == Con{Tasks.Span{first,count},later_spans} : List<&2,Tasks.Span>}},
    Equal.cong(List<&2,Tasks.Span>,List<&2,Tasks.Span>,tail => Con{Tasks.Span{first,count},tail},later_spans,Nil{},
      Components.spans_end(rest_classes,prefix,(at + 1 : U32),stop)))
  +image = BoxCells.image_of(Con{extent,rest},0n)
  +cells = BoxCells.cells_of(Con{extent,rest},0n,block)
  +piece = BoxCells.cell_piece(value,image,cells,U32.to_nat(per))
  +held = {%Components.last_extents(extent,rest,prefix,per,at,stop) : Components.InBox(Con{Tasks.Span{first,count},Nil{}},_,piece)
    %spans : Components.InBox(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at),piece)
    inside : Components.InBox(Con{Tasks.Span{first,count},Nil{}},Con{per,Nil{}},piece)}
  +unit = ShapeLaws.elements(rest)
  +size = U32.to_nat(block)
  +unit_positive = Components.inner_positive(U32.to_nat(extent),unit,value,below)
  +coordinate = Nat.div(value,unit)
  +small = Division.small(value,image,below)
  +piece_exact = {%Equal.sym(Nat,Nat.div(value,image),0n,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,image),(0n,value),small)) :
      {Nat.add(Nat.mul(_,U32.to_nat(per)),Nat.div(Nat.mod(value,image),cells)) == Nat.div(coordinate,size) : Nat}
    %Equal.sym(Nat,Nat.mod(value,image),value,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,image),(0n,value),small)) :
      {Nat.add(Nat.mul(0n,U32.to_nat(per)),Nat.div(_,cells)) == Nat.div(coordinate,size) : Nat}
    BoxCells.unit_quotient(value,size,unit,block_positive,unit_positive) : {piece == Nat.div(coordinate,size) : Nat}}
  +one = Equal.trans(Nat,Nat.div(piece,1n),piece,Nat.div(coordinate,size),
    Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(piece,1n),(piece,0n),BoxCells.by_one(piece)),piece_exact)
  +low = {%one : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} box_low(first,count,Nil{},per,Nil{},piece,held) :
    {Nat.is_le(U32.to_nat(first),Nat.div(coordinate,size)) == True{} : Bool}}
  +high = {%one : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool} box_high(first,count,Nil{},per,Nil{},piece,held) :
    {Nat.is_lt(Nat.div(coordinate,size),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}
  +same_block = last_block(grid,at,extent,prefix,block,same_prefix,same_units,stop,Components.last_eq(at,prefix,position,cap,cap_ok,exact,distance,prefix_room))
  +coordinate_in = DivisionOrder.quotient_below(value,U32.to_nat(extent),unit,unit_positive,below)
  +length = List.length(&2,U32,rest)
  +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
  +next_exact = Components.kept_next(at,prefix,position,0n,cap,cap_ok,exact,distance,prefix_room)
  +reached = Equal.trans(Nat,U32.to_nat(Components.grid_prefix(grid)),U32.to_nat(prefix),1n+position,
    Equal.cong(U32,Nat,word => U32.to_nat(word),Components.grid_prefix(grid),prefix,same_prefix),
    Equal.trans(Nat,U32.to_nat(prefix),Nat.add(position,1n),1n+position,Equal.sym(Nat,Nat.add(position,1n+0n),U32.to_nat(prefix),distance),Addition.successor_width(position)))
  Held.Both{
    {%Equal.sym(U32,Grid.block(grid,at,extent),block,same_block) : ClassIn(class,_,extent,coordinate)
      class_sum(first,count,source_first,source_count,block,extent,coordinate,cap,cap_ok,
        {%same_block : Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_))) end_in :
          Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))},
        {%same_block : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(_)),U32.to_nat(cap)) == True{} : Bool} axis_room :
          {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}},
        block_positive,coordinate_in,low,high) :
      ClassIn(class,Grid.block(grid,at,extent),extent,coordinate)},
    past_classes(rest_classes,rest,grid,(at + 1 : U32),1n+position,Nat.mod(value,unit),cap,cap_ok,next_exact,
      {%Equal.sym(Nat,U32.to_nat(Components.grid_prefix(grid)),1n+position,reached) : {Nat.is_le(_,1n+position) == True{} : Bool} Order.reflexive(1n+position) :
        {Nat.is_le(U32.to_nat(Components.grid_prefix(grid)),1n+position) == True{} : Bool}},
      shifted,later_ends,DivisionOrder.remainder_small(value,unit,unit_positive))}

# Every axis of a cell whose piece lies in the target spans of a component
# lies in the class of the component there: the prefix axes before the last
# by the index of the piece, the last by its block, the rest whole.
def piece_classes(classes: List<&2,Tasks.Class>,extents: List<&2,U32>,steps: Nat,+grid: Grid.Grid,+at: U32,+position: Nat,+prefix: U32,+block: U32,+per: U32,
  +value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+steps) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+reach: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +fits: {Nat.is_le(BoxCells.image_of(extents,steps),Nat.mul(U32.to_nat(per),BoxCells.cells_of(extents,steps,block))) == True{} : Bool},
  +same_prefix: {Components.grid_prefix(grid) == prefix : U32},+same_units: {Components.grid_units(grid) == block : U32},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(cap)) == True{} : Bool},
  +ends: ClassEnds(classes,grid,extents,at,cap),
  +inside: Components.InBox(Tasks.target_spans(classes,prefix,at),Tasks.piece_extents(extents,prefix,per,at),
    BoxCells.cell_piece(value,BoxCells.image_of(extents,steps),BoxCells.cells_of(extents,steps,block),U32.to_nat(per))),
  +below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  ClassesIn(classes,grid,extents,cell_coordinates(extents,value),at):
  match classes extents steps:
    case Con{Tasks.Class{+first,+count,+source_first,+source_count},+rest_classes} Con{+extent,+rest} 0n:
      last_class(first,count,source_first,source_count,rest_classes,extent,rest,grid,at,position,prefix,block,per,value,cap,cap_ok,exact,distance,prefix_room,
        block_positive,same_prefix,same_units,room,ends,inside,below)
    case Con{Tasks.Class{+first,+count,+source_first,+source_count},+rest_classes} Con{+extent,+rest} 1n+ +later:
      Held.Both{+end_in,+more} = ends
      Held.Both{+axis_room,+later_ends} = more
      +class = {Tasks.Class{first,count,source_first,source_count} : Tasks.Class}
      +kept = Components.kept_lt(at,prefix,position,later,cap,cap_ok,exact,distance,prefix_room)
      +kept_at = Components.at_kept(at,prefix,position,1n+later,exact,distance)
      +later_spans = Tasks.target_spans(rest_classes,prefix,(at + 1 : U32))
      +later_pieces = Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))
      +image = BoxCells.image_of(rest,later)
      +cells = BoxCells.cells_of(rest,later,block)
      +per_count = U32.to_nat(per)
      +piece = BoxCells.cell_piece(value,image,cells,per_count)
      +spans = {%Equal.sym(Bool,U32.is_lt(at,prefix),True{},kept_at) :
          {Bool.pick(List<&2,Tasks.Span>,_,Con{Tasks.Span{first,count},later_spans},Nil{}) == Con{Tasks.Span{first,count},later_spans} : List<&2,Tasks.Span>}
        {==} :
        {Tasks.target_spans(Con{class,rest_classes},prefix,at) == Con{Tasks.Span{first,count},later_spans} : List<&2,Tasks.Span>}}
      +held = {%Components.kept_extents(extent,rest,prefix,per,at,kept) : Components.InBox(Con{Tasks.Span{first,count},later_spans},_,piece)
        %spans : Components.InBox(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at),piece)
        inside : Components.InBox(Con{Tasks.Span{first,count},later_spans},Con{extent,later_pieces},piece)}
      +next_exact = Components.kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,distance,prefix_room)
      +next_distance = Components.steps_next(position,1n+later,U32.to_nat(prefix),distance)
      +lead = Box.lead(rest,later)
      +inner = ShapeLaws.elements(rest)
      +inner_positive = Components.inner_positive(U32.to_nat(extent),inner,value,below)
      +split = Box.lead_split(rest,later)
      +product_positive = {%split : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,Nat.mul(lead,image)) == True{} : Bool}}
      +lead_positive = RowMajor.left_factor_positive(lead,image,product_positive)
      +image_positive = DivisionOrder.right_factor_positive(lead,image,product_positive)
      +unit_positive = DivisionOrder.right_factor_positive(Box.prefix_extent(rest,later),Box.unit_of(rest,later),image_positive)
      +cells_positive = RowMajor.multiply_lt(0n,U32.to_nat(block),Box.unit_of(rest,later),block_positive,unit_positive)
      +pieces = BoxCells.pieces_lead(rest,later,(at + 1 : U32),1n+position,prefix,per,cap,cap_ok,next_exact,next_distance,prefix_room,reach)
      +pair = BoxCells.piece_pair(value,lead,image,cells,per_count,image_positive,cells_positive,lead_positive,fits)
      +inner_pieces = ShapeLaws.elements(later_pieces)
      +rest_piece = BoxCells.cell_piece(Nat.mod(value,Nat.mul(lead,image)),image,cells,per_count)
      +same_quotient = Equal.trans(Nat,Nat.div(piece,inner_pieces),Nat.div(piece,Nat.mul(lead,per_count)),Nat.div(value,inner),
        Equal.cong(Nat,Nat,divisor => Nat.div(piece,divisor),inner_pieces,Nat.mul(lead,per_count),pieces),
        Equal.trans(Nat,Nat.div(piece,Nat.mul(lead,per_count)),Nat.div(value,Nat.mul(lead,image)),Nat.div(value,inner),
          Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(piece,Nat.mul(lead,per_count)),(Nat.div(value,Nat.mul(lead,image)),rest_piece),pair),
          Equal.cong(Nat,Nat,divisor => Nat.div(value,divisor),Nat.mul(lead,image),inner,Equal.sym(Nat,inner,Nat.mul(lead,image),split))))
      +same_rest = Equal.trans(Nat,Nat.mod(piece,inner_pieces),Nat.mod(piece,Nat.mul(lead,per_count)),BoxCells.cell_piece(Nat.mod(value,inner),image,cells,per_count),
        Equal.cong(Nat,Nat,divisor => Nat.mod(piece,divisor),inner_pieces,Nat.mul(lead,per_count),pieces),
        Equal.trans(Nat,Nat.mod(piece,Nat.mul(lead,per_count)),rest_piece,BoxCells.cell_piece(Nat.mod(value,inner),image,cells,per_count),
          Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(piece,Nat.mul(lead,per_count)),(Nat.div(value,Nat.mul(lead,image)),rest_piece),pair),
          Equal.cong(Nat,Nat,divisor => BoxCells.cell_piece(Nat.mod(value,divisor),image,cells,per_count),Nat.mul(lead,image),inner,Equal.sym(Nat,inner,Nat.mul(lead,image),split))))
      +coordinate = Nat.div(value,inner)
      +one = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(coordinate,1n),(coordinate,0n),BoxCells.by_one(coordinate))
      +low = {%Equal.sym(Nat,Nat.div(coordinate,1n),coordinate,one) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
        %same_quotient : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
        box_low(first,count,later_spans,extent,later_pieces,piece,held) : {Nat.is_le(U32.to_nat(first),Nat.div(coordinate,1n)) == True{} : Bool}}
      +high = {%Equal.sym(Nat,Nat.div(coordinate,1n),coordinate,one) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}
        %same_quotient : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}
        box_high(first,count,later_spans,extent,later_pieces,piece,held) :
        {Nat.is_lt(Nat.div(coordinate,1n),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}
      +later_box = {%same_rest : Components.InBox(later_spans,later_pieces,_) box_rest(first,count,later_spans,extent,later_pieces,piece,held) :
        Components.InBox(later_spans,later_pieces,BoxCells.cell_piece(Nat.mod(value,inner),image,cells,per_count))}
      +same_block = kept_block(grid,at,extent,prefix,same_prefix,kept)
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(1n+Nat.add(position,length),U32.to_nat(cap)) == True{} : Bool}}
      Held.Both{
        {%Equal.sym(U32,Grid.block(grid,at,extent),1,same_block) : ClassIn(class,_,extent,coordinate)
          class_sum(first,count,source_first,source_count,1,extent,coordinate,cap,cap_ok,
            {%same_block : Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_))) end_in :
              Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(1)))},
            {%same_block : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(_)),U32.to_nat(cap)) == True{} : Bool} axis_room :
              {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(1)),U32.to_nat(cap)) == True{} : Bool}},
            {==},DivisionOrder.quotient_below(value,U32.to_nat(extent),inner,inner_positive,below),low,high) :
          ClassIn(class,Grid.block(grid,at,extent),extent,coordinate)},
        piece_classes(rest_classes,rest,later,grid,(at + 1 : U32),1n+position,prefix,block,per,Nat.mod(value,inner),cap,cap_ok,next_exact,next_distance,prefix_room,
          reach,block_positive,fits,same_prefix,same_units,shifted,later_ends,later_box,DivisionOrder.remainder_small(value,inner,inner_positive))}
    case Nil{} Con{+extent,+rest} +steps:
      Empty.absurd(ClassesIn(Nil{},grid,Con{extent,rest},cell_coordinates(Con{extent,rest},value),at),ends)
    case Nil{} Nil{} +steps:
      Empty.absurd(ClassesIn(Nil{},grid,Nil{},cell_coordinates(Nil{},value),at),Order.below_zero(steps,reach))
    case Con{+class,+rest_classes} Nil{} +steps:
      Empty.absurd(ClassesIn(Con{class,rest_classes},grid,Nil{},cell_coordinates(Nil{},value),at),Order.below_zero(steps,reach))

# ---- The cells of the target pieces of a component ----

def piece_same(+value: Nat,+image: Nat,+cells: Nat,+per: Nat,+piece: Nat,+count: Nat,
  +parts: Held.Both<{BoxCells.cell_piece(value,image,cells,per) == piece : Nat},{Nat.is_lt(value,count) == True{} : Bool}>) ->
  {BoxCells.cell_piece(value,image,cells,per) == piece : Nat}:
  Held.Both{+same,+below} = parts
  same

def piece_below(+value: Nat,+image: Nat,+cells: Nat,+per: Nat,+piece: Nat,+count: Nat,
  +parts: Held.Both<{BoxCells.cell_piece(value,image,cells,per) == piece : Nat},{Nat.is_lt(value,count) == True{} : Bool}>) ->
  {Nat.is_lt(value,count) == True{} : Bool}:
  Held.Both{+same,+below} = parts
  below

def extents_box(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+value: Nat,+found: InExtents(spans,extents,value)) -> Components.InBox(spans,extents,value):
  Held.Both{+box,+below} = found
  box

def extents_below(+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+value: Nat,+found: InExtents(spans,extents,value)) ->
  {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+box,+below} = found
  below

# A member of the pieces a box lists, from the first position, lies inside
# the box.
def listed_in(+list: List<&2,U32>,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+piece: U32,+listed: Components.BoxListed(list,spans,extents),
  +within: Components.SpansIn(spans,extents),+member: Moved.Member(list,piece)) -> InExtents(spans,extents,U32.to_nat(piece)):
  Held.Both{+from_zero,+bound} = listed
  member_in(list,spans,extents,0n,piece,from_zero,bound,within,member)

# The one piece of a grid without prefix axes is the first.
def first_piece(+piece: U32,+member: Moved.Member(Con{0,Nil{}},piece)) -> {U32.to_nat(piece) == 0n : Nat}:
  match member:
    case Placed.OnLeft{+same}:
      %same : {U32.to_nat(_) == 0n : Nat}
      {==}
    case Placed.OnRight{+others}:
      Empty.absurd({U32.to_nat(piece) == 0n : Nat},others)

# Without prefix axes every class of a cell holds it whole.
def target_classes_none(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classes: List<&2,Tasks.Class>,
  +ends: ClassEnds(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,0,Machine.half_word(limits)),
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(extents) : Nat},+piece: U32,
  +member: Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.target_spans(classes,prefix,0)),piece),
  +index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(U32.to_nat(piece))))) == True{} : Bool},
  +none: {U32.is_zero(prefix) == True{} : Bool}) ->
  ClassesIn(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(U32.to_nat(piece)))),index)),0):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +number = U32.to_nat(piece)
  +value = Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(number))),index)
  +first = first_piece(piece,{%Components.none_pieces(count,prefix,outer,extent,unit_cells,block,per,extents,Tasks.target_spans(classes,prefix,0),none) :
      Moved.Member(_,piece) member : Moved.Member(Con{0,Nil{}},piece)})
  +inside = {%Equal.sym(Nat,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,
      Components.prefix_zero_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,none)) :
      {Nat.is_lt(number,_) == True{} : Bool}
    %Equal.sym(Nat,number,0n,first) : {Nat.is_lt(_,1n) == True{} : Bool}
    {==} : {Nat.is_lt(number,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
  +count_below = piece_below(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per),number,
    U32.to_nat(count),cell_of_piece(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,number,inside,index,index_in))
  +below = {%count_exact : {Nat.is_lt(value,_) == True{} : Bool} count_below : {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}}
  +past = {%Equal.sym(U32,prefix,0,Arithmetic.word_zero(prefix,none)) : {Nat.is_le(U32.to_nat(_),0n) == True{} : Bool} {==} :
    {Nat.is_le(U32.to_nat(Components.grid_prefix(grid)),0n) == True{} : Bool}}
  past_classes(classes,extents,grid,0,0n,value,Machine.half_word(limits),LimitBounds.half_word_fits(limits),{==},past,Components.rank_half(limits,extents,rank_room),
    ends,below)

# With prefix axes the piece of a cell lies in the target spans, so every
# axis of the cell lies in its class.
def target_classes_some(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classes: List<&2,Tasks.Class>,
  +within: Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)),
  +ends: ClassEnds(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,0,Machine.half_word(limits)),
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(extents) : Nat},+piece: U32,
  +member: Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.target_spans(classes,prefix,0)),piece),
  +index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(U32.to_nat(piece))))) == True{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  ClassesIn(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(U32.to_nat(piece)))),index)),0):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +half = Machine.half_word(limits)
  +cap = Machine.capacity(limits)
  +number = U32.to_nat(piece)
  +value = Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(number))),index)
  +size = U32.to_nat(extent)
  +units = U32.to_nat(unit_cells)
  +blocks = U32.to_nat(block)
  +per_count = U32.to_nat(per)
  +image = Nat.mul(size,units)
  +piece_cells = Nat.mul(blocks,units)
  +spans = Tasks.target_spans(classes,prefix,0)
  +pieces = Tasks.piece_extents(extents,prefix,U32.max(per,1),0)
  +total = Tasks.span_count(spans)
  +list = Tasks.pieces_listed(U32.to_nat(total),spans,pieces,total,Nil{})
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +prefix_positive = Arithmetic.word_nonzero_positive(prefix,some)
  +prefix_steps = Order.restore_shift(1n,U32.to_nat(prefix),Order.strict_follow(0n,U32.to_nat(prefix),prefix_positive))
  +walk = Box.walked(extents,U32.is_le(Grid.product(extents),target),target)
  +walk_steps = Equal.trans(Nat,walk,U32.to_nat(prefix),1n+steps,
    Equal.sym(Nat,U32.to_nat(prefix),walk,Box.field_prefix(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,exact,rank_room)),
    Equal.sym(Nat,1n+steps,U32.to_nat(prefix),prefix_steps))
  +extent_eq = Equal.trans(Nat,size,Box.prefix_extent(Con{1,extents},walk),Box.prefix_extent(extents,steps),
    Box.field_extent(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,exact),
    Equal.cong(Nat,Nat,found => Box.prefix_extent(Con{1,extents},found),walk,1n+steps,walk_steps))
  +unit_eq = Equal.trans(Nat,units,Box.unit_of(Con{1,extents},walk),Box.unit_of(extents,steps),
    Box.field_unit(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,exact,room,nonempty),
    Equal.cong(Nat,Nat,found => Box.unit_of(Con{1,extents},found),walk,1n+steps,walk_steps))
  +per_eq = Extrema.max_one_exact(per,per_count,{==},Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape))
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,extents),U32.to_nat(half),
    Components.shape_within(extents,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,LimitBounds.capacity_fits(limits),rank_room),
    Components.rank_half(limits,extents,rank_room))
  +reach = Components.prefix_steps_in(extents,target,count,prefix,outer,extent,unit_cells,block,per,cap,LimitBounds.capacity_fits(limits),exact,rank_room,some)
  +fits = {%extent_eq :
      {Nat.is_le(Nat.mul(_,Box.unit_of(extents,steps)),Nat.mul(U32.to_nat(U32.max(per,1)),Nat.mul(blocks,Box.unit_of(extents,steps)))) == True{} : Bool}
    %unit_eq :
      {Nat.is_le(Nat.mul(size,_),Nat.mul(U32.to_nat(U32.max(per,1)),Nat.mul(blocks,_))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.max(per,1)),per_count,per_eq) : {Nat.is_le(image,Nat.mul(_,piece_cells)) == True{} : Bool}
    BoxCells.grid_fits(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(BoxCells.image_of(extents,steps),Nat.mul(U32.to_nat(U32.max(per,1)),BoxCells.cells_of(extents,steps,block))) == True{} : Bool}}
  +total_exact = Components.source_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some)
  +pieces_room = {%Equal.sym(Nat,ShapeLaws.elements(pieces),Nat.mul(U32.to_nat(outer),per_count),total_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_le(ShapeLaws.elements(pieces),U32.to_nat(cap)) == True{} : Bool}}
  +listed = Components.box_listed(spans,pieces,half,LimitBounds.half_word_fits(limits),within,
    Order.transitive(ShapeLaws.elements(pieces),U32.to_nat(cap),U32.to_nat(half),pieces_room,Components.capacity_half(limits)),
    LimitBounds.double_bound(limits,ShapeLaws.elements(pieces),pieces_room))
  +listed_member = {%some : Moved.Member(Bool.pick(List<&2,U32>,_,Con{0,Nil{}},list),piece) member : Moved.Member(list,piece)}
  +found = listed_in(list,spans,pieces,piece,listed,within,listed_member)
  +box = extents_box(spans,pieces,number,found)
  +number_below = extents_below(spans,pieces,number,found)
  +inside = {%total_exact : {Nat.is_lt(number,_) == True{} : Bool} number_below : {Nat.is_lt(number,Nat.mul(U32.to_nat(outer),per_count)) == True{} : Bool}}
  +parts = cell_of_piece(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,number,inside,index,index_in)
  +same = piece_same(value,image,piece_cells,per_count,number,U32.to_nat(count),parts)
  +count_below = piece_below(value,image,piece_cells,per_count,number,U32.to_nat(count),parts)
  +below = {%count_exact : {Nat.is_lt(value,_) == True{} : Bool} count_below : {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}}
  +held = {%Equal.sym(Nat,U32.to_nat(U32.max(per,1)),per_count,per_eq) :
      Components.InBox(spans,pieces,BoxCells.cell_piece(value,Nat.mul(Box.prefix_extent(extents,steps),Box.unit_of(extents,steps)),
        Nat.mul(blocks,Box.unit_of(extents,steps)),_))
    %unit_eq : Components.InBox(spans,pieces,BoxCells.cell_piece(value,Nat.mul(Box.prefix_extent(extents,steps),_),Nat.mul(blocks,_),per_count))
    %extent_eq : Components.InBox(spans,pieces,BoxCells.cell_piece(value,Nat.mul(_,units),piece_cells,per_count))
    %Equal.sym(Nat,BoxCells.cell_piece(value,image,piece_cells,per_count),number,same) : Components.InBox(spans,pieces,_)
    box : Components.InBox(spans,pieces,BoxCells.cell_piece(value,BoxCells.image_of(extents,steps),BoxCells.cells_of(extents,steps,block),U32.to_nat(U32.max(per,1))))}
  piece_classes(classes,extents,steps,grid,0,0n,prefix,block,U32.max(per,1),value,half,LimitBounds.half_word_fits(limits),{==},prefix_steps,prefix_room,reach,
    Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),fits,{==},{==},Components.rank_half(limits,extents,rank_room),ends,held,below)

# A cell of a target piece of a separable component lies in the classes of
# the component.
def target_classes_flat(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classes: List<&2,Tasks.Class>,
  +within: Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))),
  +ends: ClassEnds(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,0,Machine.half_word(limits)),+piece: U32,
  +member: Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.target_spans(classes,prefix,0)),piece),
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))) == True{} : Bool}) ->
  ClassesIn(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),index)),0):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +back = Arithmetic.native_roundtrip(piece)
  +count_exact = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(extents,target))),ShapeLaws.elements(extents),
    Equal.cong(Grid.Grid,Nat,found => U32.to_nat(Grid.count_of(found)),grid,Grid.grid(extents,target),Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,exact)),
    GridRanges.grid_count(limits,extents,target,room,nonempty))
  +index_back = {%Equal.sym(U32,U32.from_nat(U32.to_nat(piece)),piece,back) : {Nat.is_lt(index,U32.to_nat(Grid.span(grid,_))) == True{} : Bool}
    index_in : {Nat.is_lt(index,U32.to_nat(Grid.span(grid,U32.from_nat(U32.to_nat(piece))))) == True{} : Bool}}
  %back : ClassesIn(classes,grid,extents,cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(grid,_)),index)),0)
  Held.by_cases(ClassesIn(classes,grid,extents,cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(grid,U32.from_nat(U32.to_nat(piece)))),index)),0),U32.is_zero(prefix),
    none => target_classes_none(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,cells,room,nonempty,rank_room,classes,ends,
      count_exact,piece,member,index,index_back,none),
    +some => target_classes_some(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,cells,room,nonempty,rank_room,classes,
      {%some : Logic.When(Bool.not(_),Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))) within : Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))},
      ends,count_exact,piece,member,index,index_back,some))

# The classes of a component cover the pieces of the target grid and end
# inside its blocks.
def TargetClassesOk(+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+cap: U32) -> Data:
  match grid:
    case Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+pieces}:
      Held.Both<Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))),
        ClassEnds(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},extents,0,cap)>

def target_classes_with(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+classes: List<&2,Tasks.Class>,
  +ok: Held.Both<Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))),
    ClassEnds(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,0,Machine.half_word(limits))>,+piece: U32,
  +member: Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.target_spans(classes,prefix,0)),piece),
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))) == True{} : Bool}) ->
  ClassesIn(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),index)),0):
  Held.Both{+within,+ends} = ok
  target_classes_flat(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,cells,room,nonempty,rank_room,classes,within,ends,piece,member,
    index,index_in)

def target_classes_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(extents,target)>,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate: {Tasks.separable(readings) == True{} : Bool},
  +ok: TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits)),+piece: U32,
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(extents,target),extents,component),piece),
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool}) ->
  ClassesIn(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.grid(extents,target),piece)),index)),0):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +flat = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)
  +flat_member = {%exact : Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),_,extents,component),piece) member :
    Moved.Member(Tasks.target_when(Tasks.separable(readings),classes,flat,extents,prefix),piece)}
  %Equal.sym(Grid.Grid,Grid.grid(extents,target),flat,exact) :
    ClassesIn(classes,_,extents,cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(_,piece)),index)),0)
  target_classes_with(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,cells,room,nonempty,rank_room,classes,
    {%exact : TargetClassesOk(classes,_,extents,Machine.half_word(limits)) ok : TargetClassesOk(classes,flat,extents,Machine.half_word(limits))},piece,
    {%separate : Moved.Member(Tasks.target_when(_,classes,flat,extents,prefix),piece) flat_member : Moved.Member(Tasks.pieces_in(flat,extents,Tasks.target_spans(classes,prefix,0)),piece)},
    index,{%exact : {Nat.is_lt(index,U32.to_nat(Grid.span(_,piece))) == True{} : Bool} index_in : {Nat.is_lt(index,U32.to_nat(Grid.span(flat,piece))) == True{} : Bool}})

# A cell of a target piece of a separable component lies in the classes of
# the component.
def target_classes(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+separate: {Tasks.separable(readings) == True{} : Bool},
  +ok: TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits)),+piece: U32,
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(extents,target),extents,component),piece),
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool}) ->
  ClassesIn(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,
    cell_coordinates(extents,Nat.add(U32.to_nat(Grid.start(Grid.grid(extents,target),piece)),index)),0):
  target_classes_facts(limits,extents,target,GridRanges.grid_facts(limits,extents,target,room,nonempty),room,nonempty,rank_room,readings,component,separate,ok,piece,
    member,index,index_in)

# ---- A target piece of a component is a piece of the grid ----

# A piece listed by spans inside the pieces of a grid with prefix axes is a
# piece of the grid.
def listed_below(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+spans: List<&2,Tasks.Span>,
  +within: Components.SpansIn(spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0)),+some: {U32.is_zero(prefix) == False{} : Bool},+piece: U32,
  +member: Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,spans),piece)) -> {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +half = Machine.half_word(limits)
  +cap = Machine.capacity(limits)
  +pieces = Tasks.piece_extents(extents,prefix,U32.max(per,1),0)
  +total = Tasks.span_count(spans)
  +list = Tasks.pieces_listed(U32.to_nat(total),spans,pieces,total,Nil{})
  +total_exact = Components.source_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some)
  +pieces_room = {%Equal.sym(Nat,ShapeLaws.elements(pieces),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),total_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_le(ShapeLaws.elements(pieces),U32.to_nat(cap)) == True{} : Bool}}
  +listed = Components.box_listed(spans,pieces,half,LimitBounds.half_word_fits(limits),within,
    Order.transitive(ShapeLaws.elements(pieces),U32.to_nat(cap),U32.to_nat(half),pieces_room,Components.capacity_half(limits)),
    LimitBounds.double_bound(limits,ShapeLaws.elements(pieces),pieces_room))
  +listed_member = {%some : Moved.Member(Bool.pick(List<&2,U32>,_,Con{0,Nil{}},list),piece) member : Moved.Member(list,piece)}
  +found = listed_in(list,spans,pieces,piece,listed,within,listed_member)
  %total_exact : {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool}
  extents_below(spans,pieces,U32.to_nat(piece),found)

def classes_within(+classes: List<&2,Tasks.Class>,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+extents: List<&2,U32>,
  +cap: U32,+ok: TargetClassesOk(classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,cap),
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)):
  Held.Both{+within,+ends} = ok
  %some : Logic.When(Bool.not(_),Components.SpansIn(Tasks.target_spans(classes,prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)))
  within

def piece_separate(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +some: {U32.is_zero(prefix) == False{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +ok: Logic.When(Tasks.separable(readings),TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Machine.half_word(limits))),+piece: U32,
  +member: Moved.Member(Tasks.target_when(Tasks.separable(readings),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece)) ->
  {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  listed_below(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),
    classes_within(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),count,prefix,outer,extent,unit_cells,block,per,extents,Machine.half_word(limits),
      {%separate : Logic.When(_,TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Machine.half_word(limits))) ok : TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Machine.half_word(limits))},some),
    some,piece,{%separate : Moved.Member(Tasks.target_when(_,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece) member : Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0)),piece)})

def piece_whole(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +some: {U32.is_zero(prefix) == False{} : Bool},+whole: {Tasks.separable(readings) == False{} : Bool},+piece: U32,
  +member: Moved.Member(Tasks.target_when(Tasks.separable(readings),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece)) ->
  {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +pieces = Tasks.piece_extents(extents,prefix,U32.max(per,1),0)
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +distance = Order.positive_has_last(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some))
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,extents),U32.to_nat(cap),
    Components.shape_within(extents,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,cap_ok,rank_room),rank_room)
  +reach = Components.prefix_steps_in(extents,target,count,prefix,outer,extent,unit_cells,block,per,cap,cap_ok,exact,rank_room,some)
  +shaped = Components.shape_pieces(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some)
  +total_exact = Components.pieces_total(extents,prefix,per,outer,extent,shaped)
  +last_extent = Components.pieces_last(extents,prefix,per,outer,extent,shaped)
  +last = Components.per_last(limits,extents,steps,outer,extent,unit_cells,block,per,shape,last_extent)
  +last_room = {%Equal.sym(U32,Components.extent_nat(extents,steps),extent,last_extent) : {Nat.is_le(U32.to_nat(_),U32.to_nat(cap)) == True{} : Bool}
    Laws.extent_room(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_le(U32.to_nat(Components.extent_nat(extents,steps)),U32.to_nat(cap)) == True{} : Bool}}
  +block_room = Order.transitive(U32.to_nat(block),U32.to_nat(extent),U32.to_nat(cap),Laws.shape_below(limits,outer,extent,unit_cells,block,per,shape),
    Laws.extent_room(limits,outer,extent,unit_cells,block,per,shape))
  +back = Equal.sym(Nat,ShapeLaws.elements(pieces),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),total_exact)
  +filled = {%back : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(outer),U32.to_nat(per),Components.shape_outer(limits,outer,extent,unit_cells,block,per,shape),
      Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(0n,ShapeLaws.elements(pieces)) == True{} : Bool}}
  +fits = {%back : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(pieces),U32.to_nat(cap)) == True{} : Bool}}
  +whole_eq = Components.whole_blocks_spans(extents,steps,limits,count,prefix,outer,extent,unit_cells,block,per,0,0n,{==},distance,prefix_room,reach,last,last_room,
    block_room,Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),filled,fits)
  listed_below(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,Tasks.whole_spans(pieces),
    Components.whole_in(pieces),some,piece,
    {%whole_eq : Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,_),piece)
      %whole : Moved.Member(Tasks.target_when(_,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece)
      member : Moved.Member(Tasks.pieces_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Tasks.whole_spans(pieces)),piece)})

def piece_alone(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +none: {U32.is_zero(prefix) == True{} : Bool},+piece: U32,+member: Moved.Member(Tasks.target_when(Tasks.separable(readings),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece)) ->
  {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +first = first_piece(piece,{%Components.none_pieces_of(Tasks.separable(readings),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),count,prefix,outer,extent,unit_cells,block,per,extents,none) :
      Moved.Member(_,piece) member : Moved.Member(Con{0,Nil{}},piece)})
  %Equal.sym(Nat,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n,
      Components.prefix_zero_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,none)) :
    {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(piece),0n,first) : {Nat.is_lt(_,1n) == True{} : Bool}
  {==}

def target_piece_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(extents,target)>,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +ok: Logic.When(Tasks.separable(readings),TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits))),+piece: U32,
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(extents,target),extents,component),piece)) ->
  {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +flat_member = {%exact : Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),_,extents,component),piece) member :
    Moved.Member(Tasks.target_when(Tasks.separable(readings),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,prefix),piece)}
  +flat_ok = {%exact : Logic.When(Tasks.separable(readings),TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),_,extents,Machine.half_word(limits))) ok :
    Logic.When(Tasks.separable(readings),TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,Machine.half_word(limits)))}
  %Equal.sym(Grid.Grid,Grid.grid(extents,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(_))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool}
  Held.by_cases({Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},U32.is_zero(prefix),
    +none => piece_alone(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,readings,component,none,piece,flat_member),
    +some => Held.by_cases({Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},Tasks.separable(readings),
      +yes => piece_separate(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,readings,component,some,yes,
        flat_ok,piece,flat_member),
      +no => piece_whole(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,readings,component,some,no,piece,
        flat_member)))

# A target piece of a component is a piece of the grid.
def target_piece_below(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +ok: Logic.When(Tasks.separable(readings),TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits))),+piece: U32,
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Grid.grid(extents,target),extents,component),piece)) ->
  {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}:
  target_piece_facts(limits,extents,target,GridRanges.grid_facts(limits,extents,target,room,nonempty),room,nonempty,rank_room,readings,component,ok,piece,member)

# What a separable operation needs of a component, in its last part.
def when_third(~First: Data,~Second: Data,~Third: Data,separate: Bool,+facts: Logic.When(separate,Held.Both<First,Held.Both<Second,Third>>)) ->
  Logic.When(separate,Third):
  match separate:
    case True{}:
      Held.Both{+first,Held.Both{+second,+third}} = facts
      third
    case False{}: Unit{}

# A piece of a grid below the piece count holds at least one cell.
def span_positive_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(extents,target)>,+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool}:
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +grid_inside = {%exact : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(_))) == True{} : Bool}
    inside : {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}}
  +flat_inside = {%Laws.pieces_word(limits,outer,extent,unit_cells,block,per,shape) : {Nat.is_lt(U32.to_nat(piece),_) == True{} : Bool}
    grid_inside : {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
  %Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,exact) : {Nat.is_lt(0n,U32.to_nat(Grid.span(_,piece))) == True{} : Bool}
  Arithmetic.word_nonzero_positive(Grid.span(grid,piece),Components.span_nonzero(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,flat_inside))

def grid_span_positive(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Grid.span(Grid.grid(extents,target),piece))) == True{} : Bool}:
  span_positive_facts(limits,extents,target,GridRanges.grid_facts(limits,extents,target,room,nonempty),piece,inside)

# The box of a piece of a grid holds as many cells as the piece.
def grid_box_cells(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(extents,target)))) == True{} : Bool}) ->
  {ShapeLaws.elements(Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece))) == U32.to_nat(Grid.span(Grid.grid(extents,target),piece)) : Nat}:
  Box.piece_box_cells(extents,Tasks.strides_of(extents),Grid.box_of(Grid.grid(extents,target),extents,Grid.span(Grid.grid(extents,target),piece)),
    U32.to_nat(Grid.start(Grid.grid(extents,target),piece)),U32.to_nat(Grid.span(Grid.grid(extents,target),piece)),0n,
    Box.grid_piece_box(limits,extents,Tasks.strides_of(extents),target,room,nonempty,rank_room,piece,inside,0n,grid_span_positive(limits,extents,target,room,nonempty,piece,inside)))

# ---- Lengths of class lists ----

# A component chooses one class of every list.
def into_length(axis: Tasks.Classes,inner: Tasks.Chosen) ->
  {List.length(&2,Tasks.Class,Tasks.chosen_classes(Tasks.chosen_into(axis,inner))) == 1n+List.length(&2,Tasks.Class,Tasks.chosen_classes(inner)) : Nat}:
  match axis:
    case Tasks.Classes{+count,+list}:
      match inner:
        case Tasks.Chosen{+classes,+index}: {==}

def chosen_length(lists: List<&2,Tasks.Classes>,+index: U32) ->
  {List.length(&2,Tasks.Class,Tasks.chosen(lists,index)) == List.length(&2,Tasks.Classes,lists) : Nat}:
  match lists:
    case Nil{}: {==}
    case Con{+list,+rest}:
      Equal.trans(Nat,List.length(&2,Tasks.Class,Tasks.chosen(Con{list,rest},index)),1n+List.length(&2,Tasks.Class,Tasks.chosen(rest,index)),
        1n+List.length(&2,Tasks.Classes,rest),
        into_length(list,Tasks.chosen_from(rest,index)),
        Equal.cong(Nat,Nat,value => 1n+value,List.length(&2,Tasks.Class,Tasks.chosen(rest,index)),List.length(&2,Tasks.Classes,rest),
          chosen_length(rest,index)))

# The class lists of rank axes are rank lists.
def lists_length(rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>) -> {List.length(&2,Tasks.Classes,Tasks.class_lists(rank,readings)) == rank : Nat}:
  match rank:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,value => 1n+value,List.length(&2,Tasks.Classes,Tasks.class_lists(rest,Tasks.tails(readings))),rest,lists_length(rest,Tasks.tails(readings)))

# A component of rank axes has a class of every axis.
def chosen_rank(+rank: Nat,+readings: List<&2,List<&2,Tasks.Reading>>,+index: U32) ->
  {List.length(&2,Tasks.Class,Tasks.chosen(Tasks.class_lists(rank,readings),index)) == rank : Nat}:
  Equal.trans(Nat,List.length(&2,Tasks.Class,Tasks.chosen(Tasks.class_lists(rank,readings),index)),List.length(&2,Tasks.Classes,Tasks.class_lists(rank,readings)),rank,
    chosen_length(Tasks.class_lists(rank,readings),index),lists_length(rank,readings))

# The target spans of a component lie inside the pieces of a grid with
# prefix axes.
def classes_spans(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,
  +block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,+exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +columns: Components.ColumnsOk(Tasks.target_rank(readings),readings,Machine.half_word(limits)),
  +lists_room: Components.ListsRoom(Components.raw_lists(Tasks.target_rank(readings),readings),Machine.half_word(limits)),
  +lists_double: {Nat.is_le(Nat.double(Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))) == True{} : Bool},
  +limits_at: Components.LimitsAt(readings,Tasks.target_rank(readings),extents,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},0),+some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)):
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +pieces = Tasks.piece_extents(extents,prefix,U32.max(per,1),0)
  +steps = Nat.sub(U32.to_nat(prefix),1n)
  +distance = Order.positive_has_last(U32.to_nat(prefix),Arithmetic.word_nonzero_positive(prefix,some))
  +prefix_room = Order.transitive(U32.to_nat(prefix),List.length(&2,U32,extents),U32.to_nat(cap),
    Components.shape_within(extents,target,count,prefix,outer,extent,unit_cells,block,per,exact,cap,cap_ok,rank_room),rank_room)
  +reach = Components.prefix_steps_in(extents,target,count,prefix,outer,extent,unit_cells,block,per,cap,cap_ok,exact,rank_room,some)
  +shaped = Components.shape_pieces(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some)
  +total_exact = Components.pieces_total(extents,prefix,per,outer,extent,shaped)
  +last_extent = Components.pieces_last(extents,prefix,per,outer,extent,shaped)
  +last = Components.per_last(limits,extents,steps,outer,extent,unit_cells,block,per,shape,last_extent)
  +back = Equal.sym(Nat,ShapeLaws.elements(pieces),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),total_exact)
  +filled = {%back : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(outer),U32.to_nat(per),Components.shape_outer(limits,outer,extent,unit_cells,block,per,shape),
      Laws.per_positive(limits,outer,extent,unit_cells,block,per,shape)) : {Nat.is_lt(0n,ShapeLaws.elements(pieces)) == True{} : Bool}}
  +fits = {%back : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(ShapeLaws.elements(pieces),U32.to_nat(cap)) == True{} : Bool}}
  %Equal.sym(List<&2,Tasks.Class>,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Components.chosen_word(Components.raw_lists(Tasks.target_rank(readings),readings),component),
      Components.classes_chosen(Tasks.target_rank(readings),readings,component,Machine.half_word(limits),LimitBounds.half_word_fits(limits),lists_room,lists_double,
        inside)) :
    Components.SpansIn(Tasks.target_spans(_,prefix,0),pieces)
  Components.within_of(Tasks.target_spans(Components.chosen_word(Components.raw_lists(Tasks.target_rank(readings),readings),component),prefix,0),pieces,
    Components.chosen_within(Tasks.target_rank(readings),readings,extents,steps,count,prefix,outer,extent,unit_cells,block,per,0,0n,component,Machine.half_word(limits),LimitBounds.half_word_fits(limits),{==},distance,
      LimitBounds.within_half(limits,U32.to_nat(prefix),prefix_room),reach,columns,lists_room,lists_double,inside,limits_at,last,filled,
      LimitBounds.within_half(limits,ShapeLaws.elements(pieces),fits)))

# A component below the count of the components of readings is below the
# count of their class lists.
def component_lists(+limits: Machine.Limits,+readings: List<&2,List<&2,Tasks.Reading>>,
  +components_ok: Components.ComponentsOk(readings,Tasks.target_rank(readings),Machine.half_word(limits)),
  +lists_positive: {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))) == True{} : Bool},+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components_of(readings)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(index),Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))) == True{} : Bool}:
  +lists = Tasks.class_lists(Tasks.target_rank(readings),readings)
  +exact = Components.separate_tasks(readings,Tasks.target_rank(readings),limits,Tasks.count_of(lists),components_ok,
    Components.classes_count(Tasks.target_rank(readings),readings))
  %exact : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
  Held.by_cases({Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.count_of(lists))) == True{} : Bool},Tasks.separable(readings),
    +yes => {%yes : {Nat.is_lt(U32.to_nat(index),U32.to_nat(Bool.pick(U32,_,Tasks.count_of(lists),U32.min(Tasks.count_of(lists),1)))) == True{} : Bool} inside :
      {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.count_of(lists))) == True{} : Bool}},
    +no => Order.lt_le(U32.to_nat(index),U32.to_nat(U32.min(Tasks.count_of(lists),1)),U32.to_nat(Tasks.count_of(lists)),
      {%no : {Nat.is_lt(U32.to_nat(index),U32.to_nat(Bool.pick(U32,_,Tasks.count_of(lists),U32.min(Tasks.count_of(lists),1)))) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.min(Tasks.count_of(lists),1))) == True{} : Bool}},Extrema.minimum_below(Tasks.count_of(lists),1)))

# The classes of a component cover the target pieces, and they end inside
# their blocks when the readings of the component end inside their axes.
def classes_ok_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(extents,target)>,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +components_ok: Components.ComponentsOk(readings,Tasks.target_rank(readings),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt(readings,Tasks.target_rank(readings),extents,Grid.grid(extents,target),0),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))) == True{} : Bool},
  +ends: ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,0,Machine.half_word(limits))) ->
  TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits)):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  Held.Both{+rows,Held.Both{+columns,Held.Both{+lists_room,Held.Both{+lists_double,+rows_room}}}} = components_ok
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +flat_limits = {%exact : Components.LimitsAt(readings,Tasks.target_rank(readings),extents,_,0)
    limits_at : Components.LimitsAt(readings,Tasks.target_rank(readings),extents,grid,0)}
  +flat_ends = {%exact : ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),_,extents,0,Machine.half_word(limits)) ends : ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),grid,extents,0,Machine.half_word(limits))}
  +within = Held.by_cases(Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0))),U32.is_zero(prefix),
    +none => {%Equal.sym(Bool,U32.is_zero(prefix),True{},none) : Logic.When(Bool.not(_),Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)))
      Unit{} : Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)))},
    +some => {%Equal.sym(Bool,U32.is_zero(prefix),False{},some) : Logic.When(Bool.not(_),Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)))
      classes_spans(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,readings,component,columns,lists_room,
        lists_double,inside,flat_limits,some) :
      Logic.When(Bool.not(U32.is_zero(prefix)),Components.SpansIn(Tasks.target_spans(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),prefix,0),Tasks.piece_extents(extents,prefix,U32.max(per,1),0)))})
  %Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,exact) : TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),_,extents,Machine.half_word(limits))
  Held.Both{within,flat_ends}

def classes_ok(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,
  +components_ok: Components.ComponentsOk(readings,Tasks.target_rank(readings),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt(readings,Tasks.target_rank(readings),extents,Grid.grid(extents,target),0),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank(readings),readings))) == True{} : Bool},
  +ends: ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,0,Machine.half_word(limits))) ->
  TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Grid.grid(extents,target),extents,Machine.half_word(limits)):
  classes_ok_facts(limits,extents,target,GridRanges.grid_facts(limits,extents,target,room,nonempty),room,nonempty,rank_room,readings,component,components_ok,
    limits_at,inside,ends)
