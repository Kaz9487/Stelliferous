# A source cell inside the box of a component lies in a piece the component
# takes: its piece is listed by the block spans of the box and starts at or
# before the cell, and the next piece starts after it.

import Base
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_ceiling.bend as NatCeiling
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./u32_extrema.bend as Extrema
import ./tensor_components.bend as Components
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_box.bend as Box
import ./tensor_shape_refinement.bend as ShapeLaws
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds

# ---- A member of the pieces taken ----

# A cell inside the cells of a member of a list of pieces lies in one of their
# spans.
def member_held(list: List<&2,U32>,+grid: Grid.Grid,+piece: U32,+index: Nat,+member: Moved.Member(list,piece),
  +within: Held.Within<Grid.start(grid,piece),Grid.span(grid,piece),index>) -> Held.SpanHeld(Moved.taken_spans(list,grid),index):
  match list:
    case Nil{}:
      Empty.absurd(Held.SpanHeld(Moved.taken_spans(Nil{},grid),index),member)
    case Con{+first,Nil{}}:
      match member:
        case Placed.OnLeft{+same}:
          Inl{%Equal.sym(U32,first,piece,same) : Held.Within<Grid.start(grid,_),Grid.span(grid,_),index> within}
        case Placed.OnRight{+none}:
          Empty.absurd(Held.SpanHeld(Moved.taken_spans(Con{first,Nil{}},grid),index),none)
    case Con{+first,Con{+next,+more}}:
      match member:
        case Placed.OnLeft{+same}:
          Inl{%Equal.sym(U32,first,piece,same) : Held.Within<Grid.start(grid,_),Grid.span(grid,_),index> within}
        case Placed.OnRight{+later}:
          Inr{member_held(Con{next,more},grid,piece,index,later,within)}

# ---- The piece of a cell ----

# The piece of cell value: the pieces of the whole images before it, then the
# pieces before it in its image.
def cell_piece(+value: Nat,+image: Nat,+cells: Nat,+per: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.div(value,image),per),Nat.div(Nat.mod(value,image),cells))

# A cell of an image lies in one of its pieces.
def inner_below(+rest: Nat,+image: Nat,+cells: Nat,+per: Nat,+cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},
  +fits: {Nat.is_le(image,Nat.mul(per,cells)) == True{} : Bool},+small: {Nat.is_lt(rest,image) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(rest,cells),per) == True{} : Bool}:
  DivisionOrder.quotient_below(rest,per,cells,cells_positive,Order.lt_le(rest,image,Nat.mul(per,cells),small,fits))

# A cell of whole images lies in one of their pieces.
def piece_below(+value: Nat,+outer: Nat,+image: Nat,+cells: Nat,+per: Nat,+image_positive: {Nat.is_lt(0n,image) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},+fits: {Nat.is_le(image,Nat.mul(per,cells)) == True{} : Bool},
  +below: {Nat.is_lt(value,Nat.mul(outer,image)) == True{} : Bool}) -> {Nat.is_lt(cell_piece(value,image,cells,per),Nat.mul(outer,per)) == True{} : Bool}:
  RowMajor.radix(Nat.div(value,image),outer,Nat.div(Nat.mod(value,image),cells),per,DivisionOrder.quotient_below(value,outer,image,image_positive,below),
    inner_below(Nat.mod(value,image),image,cells,per,cells_positive,fits,DivisionOrder.remainder_small(value,image,image_positive)))

def piece_regrouped(+rows: Nat,+quotient: Nat,+inner: Nat,+whole: Nat,+outer: Nat,+per: Nat,+rows_exact: {rows == Nat.mul(whole,outer) : Nat}) ->
  {Nat.add(Nat.mul(Nat.add(rows,quotient),per),inner) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),Nat.add(Nat.mul(quotient,per),inner)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(rows,quotient),per),Nat.add(Nat.mul(rows,per),Nat.mul(quotient,per)),RowMajor.distrib(rows,quotient,per)) :
    {Nat.add(_,inner) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),Nat.add(Nat.mul(quotient,per),inner)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(rows,per),Nat.mul(quotient,per)),inner),Nat.add(Nat.mul(rows,per),Nat.add(Nat.mul(quotient,per),inner)),
      Addition.associative(Nat.mul(rows,per),Nat.mul(quotient,per),inner)) :
    {_ == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),Nat.add(Nat.mul(quotient,per),inner)) : Nat}
  %Equal.sym(Nat,rows,Nat.mul(whole,outer),rows_exact) :
    {Nat.add(Nat.mul(_,per),Nat.add(Nat.mul(quotient,per),inner)) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),Nat.add(Nat.mul(quotient,per),inner)) : Nat}
  Equal.cong(Nat,Nat,value => Nat.add(value,Nat.add(Nat.mul(quotient,per),inner)),Nat.mul(Nat.mul(whole,outer),per),Nat.mul(whole,Nat.mul(outer,per)),
    RowMajor.assoc(whole,outer,per))

# The piece of a cell past whole blocks of outer images is past as many
# blocks of outer pieces.
def piece_shift(+value: Nat,+outer: Nat,+image: Nat,+cells: Nat,+per: Nat,+image_positive: {Nat.is_lt(0n,image) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,Nat.mul(outer,image)) == True{} : Bool}) ->
  {cell_piece(value,image,cells,per) ==
    Nat.add(Nat.mul(Nat.div(value,Nat.mul(outer,image)),Nat.mul(outer,per)),cell_piece(Nat.mod(value,Nat.mul(outer,image)),image,cells,per)) : Nat}:
  +block = Nat.mul(outer,image)
  +whole = Nat.div(value,block)
  +rest = Nat.mod(value,block)
  +rows = Nat.mul(whole,outer)
  +moved = Nat.add(Nat.mul(rows,image),rest)
  +same = Equal.trans(Nat,value,Nat.add(Nat.mul(whole,block),rest),moved,DivisionOrder.quotient_exact(value,block,block_positive),
    Equal.cong(Nat,Nat,part => Nat.add(part,rest),Nat.mul(whole,block),Nat.mul(rows,image),Equal.sym(Nat,Nat.mul(rows,image),Nat.mul(whole,block),RowMajor.assoc(whole,outer,image))))
  %Equal.sym(Nat,value,moved,same) :
    {Nat.add(Nat.mul(Nat.div(_,image),per),Nat.div(Nat.mod(_,image),cells)) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),cell_piece(rest,image,cells,per)) : Nat}
  %Equal.sym(Nat,Nat.div(moved,image),Nat.add(rows,Nat.div(rest,image)),DivisionOrder.quotient_shift(rest,image,rows,image_positive)) :
    {Nat.add(Nat.mul(_,per),Nat.div(Nat.mod(moved,image),cells)) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),cell_piece(rest,image,cells,per)) : Nat}
  %Equal.sym(Nat,Nat.mod(moved,image),Nat.mod(rest,image),DivisionOrder.remainder_shift(rest,image,rows,image_positive)) :
    {Nat.add(Nat.mul(Nat.add(rows,Nat.div(rest,image)),per),Nat.div(_,cells)) == Nat.add(Nat.mul(whole,Nat.mul(outer,per)),cell_piece(rest,image,cells,per)) : Nat}
  piece_regrouped(rows,Nat.div(rest,image),Nat.div(Nat.mod(rest,image),cells),whole,outer,per,{==})

# The leading block of a cell and of its piece agree, and so do the rest.
def piece_pair(+value: Nat,+outer: Nat,+image: Nat,+cells: Nat,+per: Nat,+image_positive: {Nat.is_lt(0n,image) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,cells) == True{} : Bool},+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},
  +fits: {Nat.is_le(image,Nat.mul(per,cells)) == True{} : Bool}) ->
  {Nat.divmod(cell_piece(value,image,cells,per),Nat.mul(outer,per)) ==
    (Nat.div(value,Nat.mul(outer,image)),cell_piece(Nat.mod(value,Nat.mul(outer,image)),image,cells,per)) : Nat & Nat}:
  +block_positive = RowMajor.multiply_lt(0n,outer,image,outer_positive,image_positive)
  +rest = Nat.mod(value,Nat.mul(outer,image))
  +per_positive = Order.le_lt_trans(0n,Nat.div(Nat.mod(rest,image),cells),per,Order.zero_le(Nat.div(Nat.mod(rest,image),cells)),
    inner_below(Nat.mod(rest,image),image,cells,per,cells_positive,fits,DivisionOrder.remainder_small(rest,image,image_positive)))
  Division.unique(cell_piece(value,image,cells,per),Nat.mul(outer,per),Nat.div(value,Nat.mul(outer,image)),cell_piece(rest,image,cells,per),
    piece_shift(value,outer,image,cells,per,image_positive,block_positive),
    RowMajor.multiply_lt(0n,outer,per,outer_positive,per_positive),
    piece_below(rest,outer,image,cells,per,image_positive,cells_positive,fits,DivisionOrder.remainder_small(value,Nat.mul(outer,image),block_positive)))

# A value divided by one is itself.
def by_one(+value: Nat) -> {Nat.divmod(value,1n) == (value,0n) : Nat & Nat}:
  Division.unique(value,1n,value,0n,
    Equal.trans(Nat,value,Nat.mul(value,1n),Nat.add(Nat.mul(value,1n),0n),Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value)),
      Equal.sym(Nat,Nat.add(Nat.mul(value,1n),0n),Nat.mul(value,1n),Addition.zero_right(Nat.mul(value,1n)))),{==},{==})

# A cell over blocks of units of cells is its unit over blocks of units.
def unit_quotient(+value: Nat,+block: Nat,+unit: Nat,+block_positive: {Nat.is_lt(0n,block) == True{} : Bool},
  +unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool}) -> {Nat.div(value,Nat.mul(block,unit)) == Nat.div(Nat.div(value,unit),block) : Nat}:
  +outer = Nat.div(value,unit)
  +inner = Nat.mod(value,unit)
  +whole = Nat.div(outer,block)
  +part = Nat.mod(outer,block)
  +exact = Equal.trans(Nat,value,Nat.add(Nat.mul(outer,unit),inner),Nat.add(Nat.mul(Nat.add(Nat.mul(whole,block),part),unit),inner),
    DivisionOrder.quotient_exact(value,unit,unit_positive),
    Equal.cong(Nat,Nat,head => Nat.add(Nat.mul(head,unit),inner),outer,Nat.add(Nat.mul(whole,block),part),DivisionOrder.quotient_exact(outer,block,block_positive)))
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,Nat.mul(block,unit)),(whole,Nat.add(Nat.mul(part,unit),inner)),
    Division.unique(value,Nat.mul(block,unit),whole,Nat.add(Nat.mul(part,unit),inner),
      Equal.trans(Nat,value,Nat.add(Nat.mul(Nat.add(Nat.mul(whole,block),part),unit),inner),Nat.add(Nat.mul(whole,Nat.mul(block,unit)),Nat.add(Nat.mul(part,unit),inner)),
        exact,piece_regrouped(Nat.mul(whole,block),part,inner,whole,block,unit,{==})),
      RowMajor.multiply_lt(0n,block,unit,block_positive,unit_positive),
      RowMajor.radix(part,block,inner,unit,DivisionOrder.remainder_small(outer,block,block_positive),DivisionOrder.remainder_small(value,unit,unit_positive))))

# The pieces of the axes before the last prefix axis, then the pieces of one
# of their indices.
def pieces_lead(extents: List<&2,U32>,steps: Nat,+at: U32,+position: Nat,+prefix: U32,+per: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+steps) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+reach: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {ShapeLaws.elements(Tasks.piece_extents(extents,prefix,per,at)) == Nat.mul(Box.lead(extents,steps),U32.to_nat(per)) : Nat}:
  match extents steps:
    case Con{+extent,+rest} 0n:
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{per,Nil{}},
          Components.last_extents(extent,rest,prefix,per,at,Components.last_lt(at,prefix,position,cap,cap_ok,exact,distance,prefix_room))) :
        {ShapeLaws.elements(_) == Nat.mul(Box.lead(Con{extent,rest},0n),U32.to_nat(per)) : Nat}
      Equal.trans(Nat,Nat.mul(U32.to_nat(per),1n),U32.to_nat(per),Nat.add(U32.to_nat(per),0n),Order.times_one(U32.to_nat(per)),
        Equal.sym(Nat,Nat.add(U32.to_nat(per),0n),U32.to_nat(per),Addition.zero_right(U32.to_nat(per))))
    case Con{+extent,+rest} 1n+ +later:
      +kept = Components.kept_lt(at,prefix,position,later,cap,cap_ok,exact,distance,prefix_room)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{extent,Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))},
          Components.kept_extents(extent,rest,prefix,per,at,kept)) :
        {ShapeLaws.elements(_) == Nat.mul(Box.lead(Con{extent,rest},1n+later),U32.to_nat(per)) : Nat}
      Equal.trans(Nat,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Tasks.piece_extents(rest,prefix,per,(at + 1 : U32)))),
        Nat.mul(U32.to_nat(extent),Nat.mul(Box.lead(rest,later),U32.to_nat(per))),Nat.mul(Nat.mul(U32.to_nat(extent),Box.lead(rest,later)),U32.to_nat(per)),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),ShapeLaws.elements(Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))),
          Nat.mul(Box.lead(rest,later),U32.to_nat(per)),
          pieces_lead(rest,later,(at + 1 : U32),1n+position,prefix,per,cap,cap_ok,
            Components.kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,distance,prefix_room),Components.steps_next(position,1n+later,U32.to_nat(prefix),distance),
            prefix_room,reach)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),Box.lead(rest,later)),U32.to_nat(per)),Nat.mul(U32.to_nat(extent),Nat.mul(Box.lead(rest,later),U32.to_nat(per))),
          RowMajor.assoc(U32.to_nat(extent),Box.lead(rest,later),U32.to_nat(per))))
    case Nil{} +steps:
      Empty.absurd({ShapeLaws.elements(Tasks.piece_extents(Nil{},prefix,per,at)) == Nat.mul(Box.lead(Nil{},steps),U32.to_nat(per)) : Nat},Order.below_zero(steps,reach))

# The image and the cells of a piece at the last prefix axis of extents.
def image_of(+extents: List<&2,U32>,+steps: Nat) -> Nat:
  Nat.mul(Box.prefix_extent(extents,steps),Box.unit_of(extents,steps))

def cells_of(+extents: List<&2,U32>,+steps: Nat,+block: U32) -> Nat:
  Nat.mul(U32.to_nat(block),Box.unit_of(extents,steps))

# The last prefix axis: the cell's index lies in the units of the span, and
# its unit is its piece.
def last_blocks(+first: U32,+count: U32,+rest_box: List<&2,Tasks.Span>,+extent: U32,+rest: List<&2,U32>,+at: U32,+position: Nat,+prefix: U32,+block: U32,+per: U32,
  +value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+0n) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +facts: Components.UnitFacts(Tasks.Span{first,count},block),
  +inside: Components.InBox(Con{Tasks.Span{first,count},rest_box},Con{extent,rest},value),
  +below: {Nat.is_lt(value,ShapeLaws.elements(Con{extent,rest})) == True{} : Bool}) ->
  Components.InBox(Tasks.block_spans(Con{Tasks.Span{first,count},rest_box},prefix,block,at),Tasks.piece_extents(Con{extent,rest},prefix,per,at),
    cell_piece(value,image_of(Con{extent,rest},0n),cells_of(Con{extent,rest},0n,block),U32.to_nat(per))):
  Held.Both{+first_exact,+end_exact} = facts
  Held.Both{+low,+more} = inside
  Held.Both{+high,+later} = more
  +stop = Components.last_lt(at,prefix,position,cap,cap_ok,exact,distance,prefix_room)
  +image = image_of(Con{extent,rest},0n)
  +unit = ShapeLaws.elements(rest)
  +cells = cells_of(Con{extent,rest},0n,block)
  +size = U32.to_nat(block)
  +unit_positive = Components.inner_positive(U32.to_nat(extent),unit,value,below)
  +coordinate = Nat.div(value,unit)
  +piece = cell_piece(value,image,cells,U32.to_nat(per))
  +small = Division.small(value,image,below)
  +piece_exact = {%Equal.sym(Nat,Nat.div(value,image),0n,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,image),(0n,value),small)) :
      {Nat.add(Nat.mul(_,U32.to_nat(per)),Nat.div(Nat.mod(value,image),cells)) == Nat.div(coordinate,size) : Nat}
    %Equal.sym(Nat,Nat.mod(value,image),value,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,image),(0n,value),small)) :
      {Nat.add(Nat.mul(0n,U32.to_nat(per)),Nat.div(_,cells)) == Nat.div(coordinate,size) : Nat}
    unit_quotient(value,size,unit,block_positive,unit_positive) : {piece == Nat.div(coordinate,size) : Nat}}
  +one = Equal.trans(Nat,Nat.div(piece,1n),piece,Nat.div(coordinate,size),Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(piece,1n),(piece,0n),by_one(piece)),
    piece_exact)
  +unit_exact = Extrema.max_one_exact(block,size,{==},block_positive)
  +unit_word = U32.max(block,1)
  +low_word = (first / unit_word : U32)
  +count_word = (Grid.divided_up((first + count : U32),unit_word) - first / unit_word : U32)
  +reach = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +first_value = Equal.trans(Nat,U32.to_nat(low_word),Nat.div(U32.to_nat(first),U32.to_nat(unit_word)),Nat.div(U32.to_nat(first),size),
    first_exact,Equal.cong(Nat,Nat,divisor => Nat.div(U32.to_nat(first),divisor),U32.to_nat(unit_word),size,unit_exact))
  +reach_value = Equal.trans(Nat,Nat.add(U32.to_nat(low_word),U32.to_nat(count_word)),NatCeiling.ceiling(reach,U32.to_nat(unit_word)),NatCeiling.ceiling(reach,size),
    end_exact,Equal.cong(Nat,Nat,divisor => NatCeiling.ceiling(reach,divisor),U32.to_nat(unit_word),size,unit_exact))
  +start_le = {%Equal.sym(Nat,U32.to_nat(low_word),Nat.div(U32.to_nat(first),size),first_value) : {Nat.is_le(_,Nat.div(piece,1n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.div(piece,1n),Nat.div(coordinate,size),one) : {Nat.is_le(Nat.div(U32.to_nat(first),size),_) == True{} : Bool}
    DivisionOrder.quotient_le(U32.to_nat(first),coordinate,size,block_positive,low) :
    {Nat.is_le(U32.to_nat(low_word),Nat.div(piece,1n)) == True{} : Bool}}
  +end_lt = {%Equal.sym(Nat,Nat.add(U32.to_nat(low_word),U32.to_nat(count_word)),NatCeiling.ceiling(reach,size),reach_value) :
      {Nat.is_lt(Nat.div(piece,1n),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.div(piece,1n),Nat.div(coordinate,size),one) : {Nat.is_lt(_,NatCeiling.ceiling(reach,size)) == True{} : Bool}
    Order.succ_le_lt(Nat.div(coordinate,size),NatCeiling.ceiling(reach,size),
      %NatCeiling.successor_formula(coordinate,size,block_positive) : {Nat.is_le(_,NatCeiling.ceiling(reach,size)) == True{} : Bool}
      NatCeiling.monotone(1n+coordinate,reach,size,block_positive,Order.strict_follow(coordinate,reach,high))) :
    {Nat.is_lt(Nat.div(piece,1n),Nat.add(U32.to_nat(low_word),U32.to_nat(count_word))) == True{} : Bool}}
  %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{Tasks.Span{first,count},rest_box},prefix,block,at),Con{Components.unit_span(Tasks.Span{first,count},block),Nil{}},
      Components.last_blocks(Tasks.Span{first,count},rest_box,prefix,block,at,stop,Components.last_eq(at,prefix,position,cap,cap_ok,exact,distance,prefix_room))) :
    Components.InBox(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at),piece)
  %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{per,Nil{}},Components.last_extents(extent,rest,prefix,per,at,stop)) :
    Components.InBox(Con{Components.unit_span(Tasks.Span{first,count},block),Nil{}},_,piece)
  Held.Both{start_le,Held.Both{end_lt,Unit{}}}

# Every prefix axis: the index of a cell inside a box lies in the span of its
# axis, and its piece has the same index at the axes before the last prefix
# axis and its unit at the last, inside the block spans.
def cell_blocks(box: List<&2,Tasks.Span>,extents: List<&2,U32>,steps: Nat,+at: U32,+position: Nat,+prefix: U32,+block: U32,+per: U32,+value: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+distance: {Nat.add(position,1n+steps) == U32.to_nat(prefix) : Nat},
  +prefix_room: {Nat.is_le(U32.to_nat(prefix),U32.to_nat(cap)) == True{} : Bool},+reach: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +fits: {Nat.is_le(image_of(extents,steps),Nat.mul(U32.to_nat(per),cells_of(extents,steps,block))) == True{} : Bool},
  +along: Components.SpansAlong(box,extents,steps,cap),
  +unit_room: {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(cap)) == True{} : Bool},
  +inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Components.InBox(Tasks.block_spans(box,prefix,block,at),Tasks.piece_extents(extents,prefix,per,at),
    cell_piece(value,image_of(extents,steps),cells_of(extents,steps,block),U32.to_nat(per))):
  match box extents steps:
    case Con{Tasks.Span{+first,+count},+rest_box} Con{+extent,+rest} 0n:
      Held.Both{+span_in,+extent_room} = along
      last_blocks(first,count,rest_box,extent,rest,at,position,prefix,block,per,value,cap,cap_ok,exact,distance,prefix_room,block_positive,
        Components.unit_facts(Tasks.Span{first,count},extent,block,cap,cap_ok,span_in,extent_room,unit_room),inside,below)
    case Con{Tasks.Span{+first,+count},+rest_box} Con{+extent,+rest} 1n+ +later:
      Held.Both{+span_in,+later_along} = along
      Held.Both{+low,+more} = inside
      Held.Both{+high,+later_box} = more
      +kept = Components.kept_lt(at,prefix,position,later,cap,cap_ok,exact,distance,prefix_room)
      +next_exact = Components.kept_next(at,prefix,position,1n+later,cap,cap_ok,exact,distance,prefix_room)
      +next_distance = Components.steps_next(position,1n+later,U32.to_nat(prefix),distance)
      +image = image_of(rest,later)
      +cells = cells_of(rest,later,block)
      +lead = Box.lead(rest,later)
      +inner = ShapeLaws.elements(rest)
      +per_count = U32.to_nat(per)
      +inner_positive = Components.inner_positive(U32.to_nat(extent),inner,value,below)
      +split = Box.lead_split(rest,later)
      +product_positive = {%split : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,Nat.mul(lead,image)) == True{} : Bool}}
      +lead_positive = RowMajor.left_factor_positive(lead,image,product_positive)
      +image_positive = DivisionOrder.right_factor_positive(lead,image,product_positive)
      +unit_positive = DivisionOrder.right_factor_positive(Box.prefix_extent(rest,later),Box.unit_of(rest,later),image_positive)
      +cells_positive = RowMajor.multiply_lt(0n,U32.to_nat(block),Box.unit_of(rest,later),block_positive,unit_positive)
      +pieces = pieces_lead(rest,later,(at + 1 : U32),1n+position,prefix,per,cap,cap_ok,next_exact,next_distance,prefix_room,reach)
      +piece = cell_piece(value,image,cells,per_count)
      +pair = piece_pair(value,lead,image,cells,per_count,image_positive,cells_positive,lead_positive,fits)
      +inner_pieces = ShapeLaws.elements(Tasks.piece_extents(rest,prefix,per,(at + 1 : U32)))
      +rest_piece = cell_piece(Nat.mod(value,Nat.mul(lead,image)),image,cells,per_count)
      +same_quotient = Equal.trans(Nat,Nat.div(piece,inner_pieces),Nat.div(piece,Nat.mul(lead,per_count)),Nat.div(value,inner),
        Equal.cong(Nat,Nat,divisor => Nat.div(piece,divisor),inner_pieces,Nat.mul(lead,per_count),pieces),
        Equal.trans(Nat,Nat.div(piece,Nat.mul(lead,per_count)),Nat.div(value,Nat.mul(lead,image)),Nat.div(value,inner),
          Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(piece,Nat.mul(lead,per_count)),(Nat.div(value,Nat.mul(lead,image)),rest_piece),pair),
          Equal.cong(Nat,Nat,divisor => Nat.div(value,divisor),Nat.mul(lead,image),inner,Equal.sym(Nat,inner,Nat.mul(lead,image),split))))
      +same_rest = Equal.trans(Nat,Nat.mod(piece,inner_pieces),Nat.mod(piece,Nat.mul(lead,per_count)),cell_piece(Nat.mod(value,inner),image,cells,per_count),
        Equal.cong(Nat,Nat,divisor => Nat.mod(piece,divisor),inner_pieces,Nat.mul(lead,per_count),pieces),
        Equal.trans(Nat,Nat.mod(piece,Nat.mul(lead,per_count)),rest_piece,cell_piece(Nat.mod(value,inner),image,cells,per_count),
          Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(piece,Nat.mul(lead,per_count)),(Nat.div(value,Nat.mul(lead,image)),rest_piece),pair),
          Equal.cong(Nat,Nat,divisor => cell_piece(Nat.mod(value,divisor),image,cells,per_count),Nat.mul(lead,image),inner,Equal.sym(Nat,inner,Nat.mul(lead,image),split))))
      +recursive = cell_blocks(rest_box,rest,later,(at + 1 : U32),1n+position,prefix,block,per,Nat.mod(value,inner),cap,cap_ok,next_exact,next_distance,prefix_room,reach,
        block_positive,fits,later_along,unit_room,later_box,DivisionOrder.remainder_small(value,inner,inner_positive))
      +later_blocks = Tasks.block_spans(rest_box,prefix,block,(at + 1 : U32))
      +later_pieces = Tasks.piece_extents(rest,prefix,per,(at + 1 : U32))
      %Equal.sym(List<&2,Tasks.Span>,Tasks.block_spans(Con{Tasks.Span{first,count},rest_box},prefix,block,at),Con{Tasks.Span{first,count},later_blocks},
          Components.kept_blocks(Tasks.Span{first,count},rest_box,prefix,block,at,kept)) :
        Components.InBox(_,Tasks.piece_extents(Con{extent,rest},prefix,per,at),piece)
      %Equal.sym(List<&2,U32>,Tasks.piece_extents(Con{extent,rest},prefix,per,at),Con{extent,later_pieces},Components.kept_extents(extent,rest,prefix,per,at,kept)) :
        Components.InBox(Con{Tasks.Span{first,count},later_blocks},_,piece)
      Held.Both{{%Equal.sym(Nat,Nat.div(piece,inner_pieces),Nat.div(value,inner),same_quotient) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} low :
          {Nat.is_le(U32.to_nat(first),Nat.div(piece,inner_pieces)) == True{} : Bool}},
        Held.Both{{%Equal.sym(Nat,Nat.div(piece,inner_pieces),Nat.div(value,inner),same_quotient) :
              {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool} high :
            {Nat.is_lt(Nat.div(piece,inner_pieces),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}},
          {%Equal.sym(Nat,Nat.mod(piece,inner_pieces),cell_piece(Nat.mod(value,inner),image,cells,per_count),same_rest) :
              Components.InBox(later_blocks,later_pieces,_) recursive :
            Components.InBox(later_blocks,later_pieces,Nat.mod(piece,inner_pieces))}}}
    case Nil{} Con{+extent,+rest} +steps:
      Empty.absurd(Components.InBox(Tasks.block_spans(Nil{},prefix,block,at),Tasks.piece_extents(Con{extent,rest},prefix,per,at),
        cell_piece(value,image_of(Con{extent,rest},steps),cells_of(Con{extent,rest},steps,block),U32.to_nat(per))),inside)
    case +box Nil{} +steps:
      Empty.absurd(Components.InBox(Tasks.block_spans(box,prefix,block,at),Tasks.piece_extents(Nil{},prefix,per,at),
        cell_piece(value,image_of(Nil{},steps),cells_of(Nil{},steps,block),U32.to_nat(per))),Order.below_zero(steps,reach))

# The spans of a box inside its extents lie in their axes, and the axes up to
# the last prefix axis fit cap.
def along_of(box: List<&2,Tasks.Span>,extents: List<&2,U32>,steps: Nat,+cap: U32,+within: Components.SpansIn(box,extents),
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +reach: {Nat.is_lt(steps,List.length(&2,U32,extents)) == True{} : Bool}) -> Components.SpansAlong(box,extents,steps,cap):
  match box extents steps:
    case Con{Tasks.Span{+first,+count},+rest_box} Con{+extent,+rest} 0n:
      Held.Both{+fits,+later} = within
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      Held.Both{fits,Order.transitive(U32.to_nat(extent),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),
        RowMajor.product_covers(U32.to_nat(extent),ShapeLaws.elements(rest),rest_positive),room)}
    case Con{Tasks.Span{+first,+count},+rest_box} Con{+extent,+rest} 1n+ +later:
      Held.Both{+fits,+inner} = within
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
      Held.Both{fits,along_of(rest_box,rest,later,cap,inner,
        Order.transitive(ShapeLaws.elements(rest),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),
          DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),extent_positive),room),rest_positive,reach)}
    case Nil{} Con{+extent,+rest} +steps:
      Empty.absurd(Components.SpansAlong(Nil{},Con{extent,rest},steps,cap),within)
    case +box Nil{} +steps:
      Empty.absurd(Components.SpansAlong(box,Nil{},steps,cap),Order.below_zero(steps,reach))

# A cell lies before the start of the piece after its own.
def below_next(choice: Bool,+value: Nat,+outer: Nat,+rest: Nat,+inner: Nat,+cell: Nat,+image: Nat,+cells: Nat,
  +outer_exact: {value == Nat.add(Nat.mul(outer,image),rest) : Nat},+inner_exact: {rest == Nat.add(Nat.mul(inner,cells),cell) : Nat},
  +cell_small: {Nat.is_lt(cell,cells) == True{} : Bool},+rest_small: {Nat.is_lt(rest,image) == True{} : Bool}) ->
  {Nat.is_lt(value,Laws.place(Box.next_outer(choice,outer),Box.next_inner(choice,inner),image,cells)) == True{} : Bool}:
  match choice:
    case True{}: Laws.below_next_inner(value,outer,rest,inner,cell,image,cells,outer_exact,inner_exact,cell_small)
    case False{}: Laws.below_next_outer(value,outer,rest,image,cells,outer_exact,rest_small)

# A cell of a grid lies in the piece cell_piece names.
def cell_within(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+value: Nat,
  +below: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool}) ->
  Held.Within<Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
      U32.from_nat(cell_piece(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per)))),
    Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},
      U32.from_nat(cell_piece(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per)))),value>:
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
  +ceiling = NatCeiling.ceiling(size,blocks)
  +fits = {%Equal.sym(Nat,per_count,ceiling,per_exact) : {Nat.is_le(image,Nat.mul(_,piece_cells)) == True{} : Bool}
    Held.nat_le_right_moved(image,Nat.mul(ceiling,piece_cells),Nat.mul(Nat.mul(ceiling,blocks),units),Equal.sym(Nat,Nat.mul(Nat.mul(ceiling,blocks),units),
        Nat.mul(ceiling,piece_cells),RowMajor.assoc(ceiling,blocks,units)),
      RowMajor.mul_le(size,Nat.mul(ceiling,blocks),units,NatCeiling.covers(size,blocks,block_positive))) :
    {Nat.is_le(image,Nat.mul(per_count,piece_cells)) == True{} : Bool}}
  +whole_below = {%RowMajor.assoc(U32.to_nat(outer),size,units) : {Nat.is_lt(value,_) == True{} : Bool}
    %cells : {Nat.is_lt(value,_) == True{} : Bool} below : {Nat.is_lt(value,Nat.mul(U32.to_nat(outer),image)) == True{} : Bool}}
  +piece = cell_piece(value,image,piece_cells,per_count)
  +piece_inside = piece_below(value,U32.to_nat(outer),image,piece_cells,per_count,image_positive,cells_positive,fits,whole_below)
  +quotient = Nat.div(value,image)
  +rest = Nat.mod(value,image)
  +inner = Nat.div(rest,piece_cells)
  +cell = Nat.mod(rest,piece_cells)
  +rest_small = DivisionOrder.remainder_small(value,image,image_positive)
  +inner_inside = inner_below(rest,image,piece_cells,per_count,cells_positive,fits,rest_small)
  +pair = Division.unique(piece,per_count,quotient,inner,{==},per_positive,inner_inside)
  +quotient_exact = Equal.cong(Nat & Nat,Nat,found => Pair.fst(Nat,Nat,found),Nat.divmod(piece,per_count),(quotient,inner),pair)
  +inner_exact = Equal.cong(Nat & Nat,Nat,found => Pair.snd(Nat,Nat,found),Nat.divmod(piece,per_count),(quotient,inner),pair)
  +outer_exact = DivisionOrder.quotient_exact(value,image,image_positive)
  +cell_exact = DivisionOrder.quotient_exact(rest,piece_cells,cells_positive)
  +cell_small = DivisionOrder.remainder_small(rest,piece_cells,cells_positive)
  +word = U32.from_nat(piece)
  +start = U32.to_nat(Grid.start(grid,word))
  +span = U32.to_nat(Grid.span(grid,word))
  +low = Nat.div(piece,per_count)
  +high = Nat.mod(piece,per_count)
  +choice = Nat.is_lt(1n+high,per_count)
  +base = Box.piece_base(low,high,size,blocks)
  +next = Box.piece_base(Box.next_outer(choice,low),Box.next_inner(choice,high),size,blocks)
  +start_exact = Box.start_base(limits,outer,extent,unit_cells,block,per,count,prefix,shape,piece,piece_inside)
  +span_exact = Box.piece_span(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,piece_inside)
  +start_place = Equal.trans(Nat,start,Nat.mul(base,units),Nat.mul(Box.piece_base(quotient,inner,size,blocks),units),start_exact,
    Equal.trans(Nat,Nat.mul(base,units),Nat.mul(Box.piece_base(quotient,high,size,blocks),units),Nat.mul(Box.piece_base(quotient,inner,size,blocks),units),
      Equal.cong(Nat,Nat,found => Nat.mul(Box.piece_base(found,high,size,blocks),units),low,quotient,quotient_exact),
      Equal.cong(Nat,Nat,found => Nat.mul(Box.piece_base(quotient,found,size,blocks),units),high,inner,inner_exact)))
  +after = {%Equal.sym(Nat,start,Nat.mul(Box.piece_base(quotient,inner,size,blocks),units),start_place) : {Nat.is_le(_,value) == True{} : Bool}
    %Box.place_units(quotient,inner,size,blocks,units) : {Nat.is_le(_,value) == True{} : Bool}
    Laws.place_le(value,quotient,rest,inner,cell,image,piece_cells,outer_exact,cell_exact) : {Nat.is_le(start,value) == True{} : Bool}}
  +ordered = RowMajor.mul_le(base,next,units,Box.base_before_next(choice,low,high,per_count,size,blocks,block_positive,per_exact,
    DivisionOrder.remainder_small(piece,per_count,per_positive),{==}))
  +end_exact = {%Equal.sym(Nat,start,Nat.mul(base,units),start_exact) : {Nat.add(_,span) == Nat.mul(next,units) : Nat}
    %Equal.sym(Nat,span,Nat.mul(Nat.sub(next,base),units),span_exact) : {Nat.add(Nat.mul(base,units),_) == Nat.mul(next,units) : Nat}
    %RowMajor.subtract_products(next,base,units) : {Nat.add(Nat.mul(base,units),_) == Nat.mul(next,units) : Nat}
    Order.restore_shift(Nat.mul(base,units),Nat.mul(next,units),ordered) : {Nat.add(start,span) == Nat.mul(next,units) : Nat}}
  +before = {%Equal.sym(Nat,Nat.add(start,span),Nat.mul(next,units),end_exact) : {Nat.is_lt(value,_) == True{} : Bool}
    %Box.place_units(Box.next_outer(choice,low),Box.next_inner(choice,high),size,blocks,units) : {Nat.is_lt(value,_) == True{} : Bool}
    %Equal.sym(Nat,low,quotient,quotient_exact) : {Nat.is_lt(value,Laws.place(Box.next_outer(choice,_),Box.next_inner(choice,high),image,piece_cells)) == True{} : Bool}
    %Equal.sym(Nat,high,inner,inner_exact) : {Nat.is_lt(value,Laws.place(Box.next_outer(choice,quotient),Box.next_inner(choice,_),image,piece_cells)) == True{} : Bool}
    below_next(choice,value,quotient,rest,inner,cell,image,piece_cells,outer_exact,cell_exact,cell_small,rest_small) :
    {Nat.is_lt(value,Nat.add(start,span)) == True{} : Bool}}
  Held.Within{after,before}

# ---- The pieces a box lists hold its cells ----

def zero_below_one(n: Nat,+below: {Nat.is_lt(n,1n) == True{} : Bool}) -> {n == 0n : Nat}:
  match n:
    case 0n: {==}
    case 1n+ +more: Empty.absurd({1n+more == 0n : Nat},Order.below_zero(more,below))

# A box that holds a cell has an index on every axis.
def box_listed(+box: List<&2,Tasks.Span>,+extents: List<&2,U32>,+value: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +within: Components.SpansIn(box,extents),+inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_zero(Tasks.span_count(box)) == False{} : Bool}:
  +position = Components.box_position(box,extents,value)
  +positive = Order.le_lt_trans(0n,position,Components.span_total(box),Order.zero_le(position),Components.box_position_below(box,extents,value,inside,below))
  +total_room = Order.transitive(Components.span_total(box),ShapeLaws.elements(extents),U32.to_nat(cap),Components.span_total_le(box,extents,within),room)
  Arithmetic.word_positive_nonzero(Tasks.span_count(box),
    {%Equal.sym(Nat,U32.to_nat(Tasks.span_count(box)),Components.span_total(box),Components.span_count_exact(box,cap,cap_ok,positive,total_room)) :
        {Nat.is_lt(0n,_) == True{} : Bool} positive :
      {Nat.is_lt(0n,U32.to_nat(Tasks.span_count(box))) == True{} : Bool}})

# The cells of an image fit the pieces of one of its outer indices.
def grid_fits(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(per),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)))) == True{} : Bool}:
  +size = U32.to_nat(extent)
  +units = U32.to_nat(unit_cells)
  +blocks = U32.to_nat(block)
  +ceiling = NatCeiling.ceiling(size,blocks)
  %Equal.sym(Nat,U32.to_nat(per),ceiling,Laws.shape_per(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_le(Nat.mul(size,units),Nat.mul(_,Nat.mul(blocks,units))) == True{} : Bool}
  Held.nat_le_right_moved(Nat.mul(size,units),Nat.mul(ceiling,Nat.mul(blocks,units)),Nat.mul(Nat.mul(ceiling,blocks),units),
    Equal.sym(Nat,Nat.mul(Nat.mul(ceiling,blocks),units),Nat.mul(ceiling,Nat.mul(blocks,units)),RowMajor.assoc(ceiling,blocks,units)),
    RowMajor.mul_le(size,Nat.mul(ceiling,blocks),units,NatCeiling.covers(size,blocks,Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape))))

# The piece of a cell of a grid is one of its pieces.
def cell_piece_inside(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+value: Nat,
  +below: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(cell_piece(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per)),
    Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  +image = Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))
  piece_below(value,U32.to_nat(outer),image,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per),
    Laws.image_positive(limits,outer,extent,unit_cells,block,per,shape),Laws.cells_positive(limits,outer,extent,unit_cells,block,per,shape),
    grid_fits(limits,outer,extent,unit_cells,block,per,shape),
    {%RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells)) : {Nat.is_lt(value,_) == True{} : Bool}
      %cells : {Nat.is_lt(value,_) == True{} : Bool} below : {Nat.is_lt(value,Nat.mul(U32.to_nat(outer),image)) == True{} : Bool}})

# Without prefix axes the grid is one piece, the one a box lists.
def cell_member_none(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,
  +per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+piece: Nat,
  +nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +none: {U32.is_zero(prefix) == True{} : Bool}) ->
  Moved.Member(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,box),U32.from_nat(piece)):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +spans = Tasks.block_spans(box,prefix,block,0)
  +listed = Tasks.pieces_listed(U32.to_nat(Tasks.span_count(spans)),spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0),Tasks.span_count(spans),Nil{})
  +zero = zero_below_one(piece,{%Components.prefix_zero_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,none) :
      {Nat.is_lt(piece,_) == True{} : Bool} inside : {Nat.is_lt(piece,1n) == True{} : Bool}})
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(box)),False{},nonzero) :
    Moved.Member(Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,extents,spans)),U32.from_nat(piece))
  %Equal.sym(Bool,U32.is_zero(prefix),True{},none) : Moved.Member(Bool.pick(List<&2,U32>,_,Con{0,Nil{}},listed),U32.from_nat(piece))
  Placed.OnLeft{%Equal.sym(Nat,piece,0n,zero) : {0 == U32.from_nat(_) : U32} {==}}

# With prefix axes the piece of a cell inside a box lies in the block spans
# of the box, so the box lists it.
def cell_member_some(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,
  +per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+value: Nat,
  +within: Components.SpansIn(box,extents),+inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool},
  +count_below: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool},+nonzero: {U32.is_zero(Tasks.span_count(box)) == False{} : Bool},
  +some: {U32.is_zero(prefix) == False{} : Bool}) ->
  Moved.Member(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,box),
    U32.from_nat(cell_piece(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per)))):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +half = Machine.half_word(limits)
  +cap = Machine.capacity(limits)
  +size = U32.to_nat(extent)
  +units = U32.to_nat(unit_cells)
  +blocks = U32.to_nat(block)
  +per_count = U32.to_nat(per)
  +image = Nat.mul(size,units)
  +piece_cells = Nat.mul(blocks,units)
  +piece = cell_piece(value,image,piece_cells,per_count)
  +spans = Tasks.block_spans(box,prefix,block,0)
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
  +half_room = Order.transitive(ShapeLaws.elements(extents),U32.to_nat(cap),U32.to_nat(half),room,Components.capacity_half(limits))
  +along = along_of(box,extents,steps,half,within,half_room,nonempty,reach)
  +fits = {%extent_eq :
      {Nat.is_le(Nat.mul(_,Box.unit_of(extents,steps)),Nat.mul(U32.to_nat(U32.max(per,1)),Nat.mul(blocks,Box.unit_of(extents,steps)))) == True{} : Bool}
    %unit_eq :
      {Nat.is_le(Nat.mul(size,_),Nat.mul(U32.to_nat(U32.max(per,1)),Nat.mul(blocks,_))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.max(per,1)),per_count,per_eq) : {Nat.is_le(image,Nat.mul(_,piece_cells)) == True{} : Bool}
    grid_fits(limits,outer,extent,unit_cells,block,per,shape) :
    {Nat.is_le(image_of(extents,steps),Nat.mul(U32.to_nat(U32.max(per,1)),cells_of(extents,steps,block))) == True{} : Bool}}
  +blocks_in = cell_blocks(box,extents,steps,0,0n,prefix,block,U32.max(per,1),value,half,LimitBounds.half_word_fits(limits),{==},prefix_steps,prefix_room,reach,
    Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape),fits,along,Components.unit_half(limits,outer,extent,unit_cells,block,per,shape),inside,below)
  +piece_in = {%per_eq : Components.InBox(spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0),cell_piece(value,image,piece_cells,_))
    %Equal.sym(Nat,units,Box.unit_of(extents,steps),unit_eq) : Components.InBox(spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0),cell_piece(value,Nat.mul(size,_),Nat.mul(blocks,_),U32.to_nat(U32.max(per,1))))
    %Equal.sym(Nat,size,Box.prefix_extent(extents,steps),extent_eq) : Components.InBox(spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0),
      cell_piece(value,Nat.mul(_,Box.unit_of(extents,steps)),Nat.mul(blocks,Box.unit_of(extents,steps)),U32.to_nat(U32.max(per,1))))
    blocks_in : Components.InBox(spans,Tasks.piece_extents(extents,prefix,U32.max(per,1),0),piece)}
  %Equal.sym(Bool,U32.is_zero(Tasks.span_count(box)),False{},nonzero) :
    Moved.Member(Bool.pick(List<&2,U32>,_,Nil{},Tasks.pieces_in(grid,extents,spans)),U32.from_nat(piece))
  Components.box_member(limits,outer,extent,unit_cells,block,per,count,prefix,shape,extents,spans,some,
    Components.source_total(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some),
    Components.source_blocks(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,some,box,along),
    piece,piece_in,cell_piece_inside(limits,outer,extent,unit_cells,block,per,count,shape,cells,value,count_below))

def cell_held_flat(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,
  +per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(extents,target) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+value: Nat,
  +within: Components.SpansIn(box,extents),+inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,box),
    Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),value):
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +piece = cell_piece(value,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(per))
  +count_nat = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(extents,target))),ShapeLaws.elements(extents),
    Equal.cong(Grid.Grid,Nat,found => U32.to_nat(Grid.count_of(found)),grid,Grid.grid(extents,target),Equal.sym(Grid.Grid,Grid.grid(extents,target),grid,exact)),
    GridRanges.grid_count(limits,extents,target,room,nonempty))
  +count_below = {%Equal.sym(Nat,U32.to_nat(count),ShapeLaws.elements(extents),count_nat) : {Nat.is_lt(value,_) == True{} : Bool} below :
    {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool}}
  +nonzero = box_listed(box,extents,value,Machine.capacity(limits),LimitBounds.capacity_fits(limits),within,inside,below,room)
  +member = Held.by_cases(Moved.Member(Tasks.box_pieces(grid,extents,box),U32.from_nat(piece)),U32.is_zero(prefix),
    none => cell_member_none(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,nonempty,rank_room,box,piece,nonzero,
      cell_piece_inside(limits,outer,extent,unit_cells,block,per,count,shape,cells,value,count_below),none),
    some => cell_member_some(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,exact,cells,room,nonempty,rank_room,box,value,within,inside,below,
      count_below,nonzero,some))
  member_held(Tasks.box_pieces(grid,extents,box),grid,U32.from_nat(piece),value,member,
    cell_within(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,value,count_below))

def cell_held_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+grid: Grid.Grid,+facts: Laws.GridFacts<limits,grid>,
  +same: {grid == Grid.grid(extents,target) : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+value: Nat,
  +within: Components.SpansIn(box,extents),+inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(grid,extents,box),grid),value):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +flat = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  %Equal.sym(Grid.Grid,grid,flat,exact) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(_,extents,box),_),value)
  cell_held_flat(limits,extents,target,count,prefix,outer,extent,unit_cells,block,per,shape,
    Equal.trans(Grid.Grid,Grid.grid(extents,target),grid,flat,Equal.sym(Grid.Grid,grid,Grid.grid(extents,target),same),exact),cells,room,nonempty,rank_room,
    box,value,within,inside,below)

# A cell inside a box of a source lies in a piece the box lists.
def cell_held(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+value: Nat,
  +within: Components.SpansIn(box,extents),+inside: Components.InBox(box,extents,value),+below: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(extents,target),extents,box),Grid.grid(extents,target)),value):
  cell_held_facts(limits,extents,target,Grid.grid(extents,target),GridRanges.grid_facts(limits,extents,target,room,nonempty),{==},room,nonempty,rank_room,box,value,
    within,inside,below)

# ---- A source box lies inside its source ----

# A span inside its axis before spans inside the later axes.
def span_then(+span: Tasks.Span,+extent: U32,+spans: List<&2,Tasks.Span>,+extents: List<&2,U32>,+here: Components.SpanIn(span,extent),
  +later: Components.SpansIn(spans,extents)) -> Components.SpansIn(Con{span,spans},Con{extent,extents}):
  match span:
    case Tasks.Span{+first,+count}:
      Held.Both{here,later}

# A box that lies along every axis lies inside the axes.
def along_in(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,coordinates: List<&2,U32>,+at: U32,steps: Nat,+cap: U32,
  +along: Components.SpansAlong(Tasks.boxed(row,classes,grid,extents,coordinates,at),extents,steps,cap),
  +last: {Nat.is_le(List.length(&2,U32,extents),1n+steps) == True{} : Bool}) ->
  Components.SpansIn(Tasks.boxed(row,classes,grid,extents,coordinates,at),extents):
  match extents coordinates steps:
    case Con{+extent,Nil{}} Con{+coordinate,+more} 0n:
      Held.Both{+here,+room} = along
      span_then(Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate),extent,Nil{},Nil{},here,Unit{})
    case Con{+extent,Con{+next,+rest}} Con{+coordinate,+more} 0n:
      Empty.absurd(Components.SpansIn(Tasks.boxed(row,classes,grid,Con{extent,Con{next,rest}},Con{coordinate,more},at),Con{extent,Con{next,rest}}),
        WordEncoding.false_true(last))
    case Con{+extent,+rest} Con{+coordinate,+more} 1n+later:
      Held.Both{+here,+tail} = along
      span_then(Tasks.box_span(Tasks.read_footprint(row,classes,at),extent,Grid.block(grid,at,extent),coordinate),extent,
        Tasks.boxed(row,classes,grid,rest,more,(at + 1 : U32)),rest,here,along_in(row,classes,grid,rest,more,(at + 1 : U32),later,cap,tail,last))
    case Nil{} +coordinates +steps:
      Unit{}
    case Con{+extent,+rest} Nil{} +steps:
      Empty.absurd(Components.SpansIn(Tasks.boxed(row,classes,grid,Con{extent,rest},Nil{},at),Con{extent,rest}),along)

# The box a separable row reads of a source lies inside the source.
def boxed_in(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,extents: List<&2,U32>,+offset: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: Components.ReadingsRoom(row,cap),+ends: Components.RowEnds(row,classes),
  +axes: Components.AxesOk(row,grid,extents,0,cap)) ->
  Components.SpansIn(Tasks.boxed(row,classes,grid,extents,Laws.coordinates_model(extents,offset),0),extents):
  match extents:
    case Nil{}:
      Unit{}
    case Con{+extent,+rest}:
      +steps = List.length(&2,U32,rest)
      +reach = Components.reach_coordinates(Con{extent,rest},offset,steps,Order.lt_succ(steps,steps,Order.reflexive(steps)))
      along_in(row,classes,grid,Con{extent,rest},Laws.coordinates_model(Con{extent,rest},offset),0,steps,cap,
        Components.boxed_along(row,classes,grid,Con{extent,rest},Laws.coordinates_model(Con{extent,rest},offset),0,steps,cap,cap_ok,reach,rooms,ends,axes),
        Order.reflexive(1n+steps))
# The box a separable row reads of a source from the coordinates of an
# offset lies inside the source.
def boxed_coordinates_in(+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+rooms: Components.ReadingsRoom(row,cap),+ends: Components.RowEnds(row,classes),
  +axes: Components.AxesOk(row,grid,extents,0,cap)) ->
  Components.SpansIn(Tasks.boxed(row,classes,grid,extents,Grid.coordinates(extents,offset),0),extents):
  %Equal.sym(List<&2,U32>,Grid.coordinates(extents,offset),Laws.coordinates_model(extents,offset),Laws.coordinates_exact(extents,offset)) :
    Components.SpansIn(Tasks.boxed(row,classes,grid,extents,_,0),extents)
  boxed_in(row,classes,grid,extents,offset,cap,cap_ok,rooms,ends,axes)

