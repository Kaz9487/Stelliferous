# The tiled form with its groups at the other end: a view of the cells reads
# every cell from where the other end puts it.

import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_view.bend as Views
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./tensor_view_reads.bend as ViewReads
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_view_gather.bend as ViewGather
import ./tensor_grid_box.bend as Box
import ./tensor_view_part.bend as ViewPart
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../default_machine_limits.bend as Default
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_view_bounds.bend as ViewBounds
import ../tensor_pieces.bend as Pieces
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./natural_addition.bend as Addition
import ./tensor_grid_pieces.bend as GridLaws

# ---- The axes of the other cells ----

def inner_extents(extents: List<&2,U32>,+stride: U32) -> {Views.axis_extents(Tiled.inner_axes(extents,stride)) == extents : List<&2,U32>}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {Views.axis_extents(Tiled.scaled_axes(Con{extent,rest},_,stride)) == Con{extent,rest} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => extent <> later,Views.axis_extents(Tiled.inner_axes(rest,stride)),rest,inner_extents(rest,stride))

# The other cells read an index at the index times the stride.
def inner_sum(extents: List<&2,U32>,index: Nat,+stride: U32,+limits: Machine.Limits,
  +inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool},
  +extents_room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(extents),U32.to_nat(stride)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {ViewReads.view_sum(Tiled.inner_axes(extents,stride),[],index) == Nat.mul(index,U32.to_nat(stride)) : Nat}:
  match extents index:
    case Nil{} 0n: {==}
    case Nil{} 1n+ +rest:
      Empty.absurd({ViewReads.view_sum(Tiled.inner_axes(Nil{},stride),[],1n+rest) == Nat.mul(1n+rest,U32.to_nat(stride)) : Nat},Order.below_zero(rest,inside))
    case Con{+extent,+rest} +index:
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +step = U32.to_nat(stride)
      +cap = U32.to_nat(Machine.capacity(limits))
      +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(size,inner),Order.zero_le(index),inside)
      +size_positive = RowMajor.left_factor_positive(size,inner,whole_positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,whole_positive)
      +inner_room = Order.transitive(inner,Nat.mul(size,inner),cap,DivisionOrder.le_multiple(size,inner,size_positive),extents_room)
      +rest_room = Order.transitive(Nat.mul(inner,step),Nat.mul(size,Nat.mul(inner,step)),cap,DivisionOrder.le_multiple(size,Nat.mul(inner,step),size_positive),
        {%RowMajor.assoc(size,inner,step) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.mul(size,Nat.mul(inner,step)),cap) == True{} : Bool}})
      +word = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),Grid.product(rest),stride,inner,step,
        Coordinates.product_word(limits,rest,inner_room),{==},rest_room)
      +high = Nat.div(index,inner)
      +low = Nat.mod(index,inner)
      +later = inner_sum(rest,low,stride,limits,DivisionOrder.remainder_small(index,inner,inner_positive),inner_room,rest_room)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {ViewReads.view_sum(Tiled.scaled_axes(Con{extent,rest},_,stride),[],index) == Nat.mul(index,U32.to_nat(stride)) : Nat}
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(rest,stride)))),U32.to_nat((Grid.product(rest) * stride : U32))),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],Nat.mod(index,ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(rest,stride)))))),Nat.add(Nat.mul(high,U32.to_nat((Grid.product(rest) * stride : U32))),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low)),Nat.mul(index,step),Equal.cong(List<&2,U32>,Nat,extents => Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(extents)),U32.to_nat((Grid.product(rest) * stride : U32))),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],Nat.mod(index,ShapeLaws.elements(extents)))),Views.axis_extents(Tiled.inner_axes(rest,stride)),rest,inner_extents(rest,stride)),
      Equal.trans(Nat,Nat.add(Nat.mul(high,U32.to_nat((Grid.product(rest) * stride : U32))),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low)),Nat.add(Nat.mul(high,Nat.mul(inner,step)),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low)),Nat.mul(index,step),Equal.cong(Nat,Nat,word => Nat.add(Nat.mul(high,word),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low)),U32.to_nat((Grid.product(rest) * stride : U32)),Nat.mul(inner,step),word),
      Equal.trans(Nat,Nat.add(Nat.mul(high,Nat.mul(inner,step)),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low)),Nat.add(Nat.mul(high,Nat.mul(inner,step)),Nat.mul(low,step)),Nat.mul(index,step),Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(high,Nat.mul(inner,step)),sum),ViewReads.view_sum(Tiled.inner_axes(rest,stride),[],low),Nat.mul(low,step),later),
      Equal.trans(Nat,Nat.add(Nat.mul(high,Nat.mul(inner,step)),Nat.mul(low,step)),Nat.add(Nat.mul(Nat.mul(high,inner),step),Nat.mul(low,step)),Nat.mul(index,step),Equal.cong(Nat,Nat,term => Nat.add(term,Nat.mul(low,step)),Nat.mul(high,Nat.mul(inner,step)),Nat.mul(Nat.mul(high,inner),step),Equal.sym(Nat,Nat.mul(Nat.mul(high,inner),step),Nat.mul(high,Nat.mul(inner,step)),RowMajor.assoc(high,inner,step))),
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.mul(high,inner),step),Nat.mul(low,step)),Nat.mul(Nat.add(Nat.mul(high,inner),low),step),Nat.mul(index,step),Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(high,inner),low),step),Nat.add(Nat.mul(Nat.mul(high,inner),step),Nat.mul(low,step)),RowMajor.distrib(Nat.mul(high,inner),low,step)),
      Equal.cong(Nat,Nat,number => Nat.mul(number,step),Nat.add(Nat.mul(high,inner),low),index,Equal.sym(Nat,index,Nat.add(Nat.mul(high,inner),low),DivisionOrder.quotient_exact(index,inner,inner_positive))))))))

def last_after(rest: List<&2,U32>,+head: U32,+last: U32) -> {Grid.last_of(Con{head,List.append(&2,U32,rest,[last])}) == last : U32}:
  match rest:
    case Nil{}: {==}
    case Con{+next,+more}: last_after(more,next,last)

def last_appended(others: List<&2,U32>,+last: U32) -> {Grid.last_of(List.append(&2,U32,others,[last])) == last : U32}:
  match others:
    case Nil{}: {==}
    case Con{+head,+rest}: last_after(rest,head,last)

# The view that puts the groups of cells with them innermost outermost.
def outer_view(+others: List<&2,U32>,+groups: U32) ->
  {Tiled.repositioned_view(Grid.Outermost{},List.append(&2,U32,others,[groups])) == Con{Views.Axis{groups,1},Tiled.inner_axes(others,groups)} : List<&2,Views.Axis>}:
  +cells = List.append(&2,U32,others,[groups])
  %Equal.sym(List<&2,U32>,Grid.without_last(cells),others,Public.without_last_appended(others,groups)) :
    {Con{Views.Axis{Grid.last_of(cells),1},Tiled.inner_axes(_,Grid.last_of(cells))} == Con{Views.Axis{groups,1},Tiled.inner_axes(others,groups)} : List<&2,Views.Axis>}
  %Equal.sym(U32,Grid.last_of(cells),groups,last_appended(others,groups)) :
    {Con{Views.Axis{_,1},Tiled.inner_axes(others,_)} == Con{Views.Axis{groups,1},Tiled.inner_axes(others,groups)} : List<&2,Views.Axis>}
  {==}

# The view with the groups outermost reads an index of the other cells at its
# quotient by them, and its remainder by the groups.
def outer_address(+others: List<&2,U32>,+groups: U32,+limits: Machine.Limits,+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others))) == True{} : Bool},+others_room: {Nat.is_le(ShapeLaws.elements(others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(others),U32.to_nat(groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {ViewReads.view_sum(Con{Views.Axis{groups,1},Tiled.inner_axes(others,groups)},[],index) == Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(others)),1n),Nat.mul(Nat.mod(index,ShapeLaws.elements(others)),U32.to_nat(groups))) : Nat}:
  +others_positive = DivisionOrder.right_factor_positive(U32.to_nat(groups),ShapeLaws.elements(others),Order.le_lt_trans(0n,index,Nat.mul(U32.to_nat(groups),ShapeLaws.elements(others)),Order.zero_le(index),inside))
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(others,groups)))),1n),ViewReads.view_sum(Tiled.inner_axes(others,groups),[],Nat.mod(index,ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(others,groups)))))),Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(others)),1n),ViewReads.view_sum(Tiled.inner_axes(others,groups),[],Nat.mod(index,ShapeLaws.elements(others)))),Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(others)),1n),Nat.mul(Nat.mod(index,ShapeLaws.elements(others)),U32.to_nat(groups))),Equal.cong(List<&2,U32>,Nat,extents => Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(extents)),1n),ViewReads.view_sum(Tiled.inner_axes(others,groups),[],Nat.mod(index,ShapeLaws.elements(extents)))),Views.axis_extents(Tiled.inner_axes(others,groups)),others,inner_extents(others,groups)),
    Equal.cong(Nat,Nat,sum => Nat.add(Nat.mul(Nat.div(index,ShapeLaws.elements(others)),1n),sum),ViewReads.view_sum(Tiled.inner_axes(others,groups),[],Nat.mod(index,ShapeLaws.elements(others))),Nat.mul(Nat.mod(index,ShapeLaws.elements(others)),U32.to_nat(groups)),
      inner_sum(others,Nat.mod(index,ShapeLaws.elements(others)),groups,limits,DivisionOrder.remainder_small(index,ShapeLaws.elements(others),others_positive),others_room,room)))

# The view with the groups innermost reads an index at its remainder by the
# groups, and its quotient by them on the other cells.
def inner_address(+others: List<&2,U32>,+groups: U32,+limits: Machine.Limits,+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(ShapeLaws.elements(others),U32.to_nat(groups))) == True{} : Bool},+others_room: {Nat.is_le(ShapeLaws.elements(others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool}) ->
  {ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(others,1),[Views.Axis{groups,Grid.product(others)}]),[],index) == Nat.add(Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),Nat.mul(Nat.mod(index,U32.to_nat(groups)),ShapeLaws.elements(others))) : Nat}:
  +unit_room = {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements(others),1n),ShapeLaws.elements(others),Order.times_one(ShapeLaws.elements(others))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} others_room :
    {Nat.is_le(Nat.mul(ShapeLaws.elements(others),1n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +appended_inside = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(others,1),[Views.Axis{groups,Grid.product(others)}]))),Nat.mul(ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(others,1))),U32.to_nat(groups)),
        ViewReads.appended_count(Tiled.inner_axes(others,1),groups,Grid.product(others))) : {Nat.is_lt(index,_) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents(Tiled.inner_axes(others,1)),others,inner_extents(others,1)) : {Nat.is_lt(index,Nat.mul(ShapeLaws.elements(_),U32.to_nat(groups))) == True{} : Bool}
    inside : {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(others,1),[Views.Axis{groups,Grid.product(others)}])))) == True{} : Bool}}
  Equal.trans(Nat,ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(others,1),[Views.Axis{groups,Grid.product(others)}]),[],index),Nat.add(ViewReads.view_sum(Tiled.inner_axes(others,1),[],Nat.div(index,U32.to_nat(groups))),Nat.mul(Nat.mod(index,U32.to_nat(groups)),U32.to_nat(Grid.product(others)))),Nat.add(Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),Nat.mul(Nat.mod(index,U32.to_nat(groups)),ShapeLaws.elements(others))),ViewReads.appended_sum(Tiled.inner_axes(others,1),groups,Grid.product(others),index,appended_inside,positive),
    Equal.trans(Nat,Nat.add(ViewReads.view_sum(Tiled.inner_axes(others,1),[],Nat.div(index,U32.to_nat(groups))),Nat.mul(Nat.mod(index,U32.to_nat(groups)),U32.to_nat(Grid.product(others)))),Nat.add(Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),Nat.mul(Nat.mod(index,U32.to_nat(groups)),U32.to_nat(Grid.product(others)))),Nat.add(Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),Nat.mul(Nat.mod(index,U32.to_nat(groups)),ShapeLaws.elements(others))),Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.mul(Nat.mod(index,U32.to_nat(groups)),U32.to_nat(Grid.product(others)))),ViewReads.view_sum(Tiled.inner_axes(others,1),[],Nat.div(index,U32.to_nat(groups))),Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),
      inner_sum(others,Nat.div(index,U32.to_nat(groups)),1,limits,DivisionOrder.quotient_below(index,ShapeLaws.elements(others),U32.to_nat(groups),positive,inside),others_room,unit_room)),
    Equal.cong(Nat,Nat,count => Nat.add(Nat.mul(Nat.div(index,U32.to_nat(groups)),1n),Nat.mul(Nat.mod(index,U32.to_nat(groups)),count)),U32.to_nat(Grid.product(others)),ShapeLaws.elements(others),Coordinates.product_word(limits,others,others_room))))

# ---- Where a cell lies at the other end ----

def swapped_split(+index: Nat,+inner: Nat,+outer: Nat) -> {Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)) == Nat.add(Nat.mul(Nat.mod(index,inner),outer),Nat.div(index,inner)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Nat.add(Nat.div(index,inner),Nat.mul(Nat.mod(index,inner),outer)),Nat.add(Nat.mul(Nat.mod(index,inner),outer),Nat.div(index,inner)),
    Equal.cong(Nat,Nat,number => Nat.add(number,Nat.mul(Nat.mod(index,inner),outer)),Nat.mul(Nat.div(index,inner),1n),Nat.div(index,inner),Order.times_one(Nat.div(index,inner))),
    Order.commutative(Nat.div(index,inner),Nat.mul(Nat.mod(index,inner),outer)))

# An index read with its digits swapped divides back into them.
def swapped_divmod(+index: Nat,+inner: Nat,+outer: Nat,+inside: {Nat.is_lt(index,Nat.mul(outer,inner)) == True{} : Bool}) ->
  {Nat.divmod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),outer) == (Nat.mod(index,inner),Nat.div(index,inner)) : Nat & Nat}:
  +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(outer,inner),Order.zero_le(index),inside)
  Division.unique(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),outer,Nat.mod(index,inner),Nat.div(index,inner),swapped_split(index,inner,outer),RowMajor.left_factor_positive(outer,inner,whole_positive),
    DivisionOrder.quotient_below(index,outer,inner,DivisionOrder.right_factor_positive(outer,inner,whole_positive),inside))

def swapped_below(+index: Nat,+inner: Nat,+outer: Nat,+inside: {Nat.is_lt(index,Nat.mul(outer,inner)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Nat.mul(inner,outer)) == True{} : Bool}:
  +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(outer,inner),Order.zero_le(index),inside)
  +inner_positive = DivisionOrder.right_factor_positive(outer,inner,whole_positive)
  %Equal.sym(Nat,Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Nat.add(Nat.mul(Nat.mod(index,inner),outer),Nat.div(index,inner)),swapped_split(index,inner,outer)) : {Nat.is_lt(_,Nat.mul(inner,outer)) == True{} : Bool}
  RowMajor.radix(Nat.mod(index,inner),inner,Nat.div(index,inner),outer,DivisionOrder.remainder_small(index,inner,inner_positive),DivisionOrder.quotient_below(index,outer,inner,inner_positive,inside))

def swapped_facts(+index: Nat,+inner: Nat,+outer: Nat,+inside: {Nat.is_lt(index,Nat.mul(outer,inner)) == True{} : Bool}) ->
  KernelGemm.Both<{Nat.div(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Representation.least_one(outer)) == Nat.mod(index,Representation.least_one(inner)) : Nat},
    {Nat.mod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Representation.least_one(outer)) == Nat.div(index,Representation.least_one(inner)) : Nat}>:
  +whole_positive = Order.le_lt_trans(0n,index,Nat.mul(outer,inner),Order.zero_le(index),inside)
  +outer_least = Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,RowMajor.left_factor_positive(outer,inner,whole_positive)))
  +inner_least = Equal.sym(Nat,Representation.least_one(inner),inner,Representation.least_positive(inner,DivisionOrder.right_factor_positive(outer,inner,whole_positive)))
  +pair = Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),outer),(Nat.mod(index,inner),Nat.div(index,inner)),swapped_divmod(index,inner,outer,inside))
  KernelGemm.Both{
    {%outer_least : {Nat.div(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),_) == Nat.mod(index,Representation.least_one(inner)) : Nat}
      %inner_least : {Nat.div(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),outer) == Nat.mod(index,_) : Nat}
      %pair : {Pair.fst(Nat,Nat,_) == Nat.mod(index,inner) : Nat}
      {==} : {Nat.div(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Representation.least_one(outer)) == Nat.mod(index,Representation.least_one(inner)) : Nat}},
    {%outer_least : {Nat.mod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),_) == Nat.div(index,Representation.least_one(inner)) : Nat}
      %inner_least : {Nat.mod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),outer) == Nat.div(index,_) : Nat}
      %pair : {Pair.snd(Nat,Nat,_) == Nat.div(index,inner) : Nat}
      {==} : {Nat.mod(Nat.add(Nat.mul(Nat.div(index,inner),1n),Nat.mul(Nat.mod(index,inner),outer)),Representation.least_one(outer)) == Nat.div(index,Representation.least_one(inner)) : Nat}}}

def first_of(+left: Nat,+right: Nat,+other_left: Nat,+other_right: Nat,+both: KernelGemm.Both<{left == right : Nat},{other_left == other_right : Nat}>) -> {left == right : Nat}:
  KernelGemm.Both{+first,+second} = both
  first

def second_of(+left: Nat,+right: Nat,+other_left: Nat,+other_right: Nat,+both: KernelGemm.Both<{left == right : Nat},{other_left == other_right : Nat}>) -> {other_left == other_right : Nat}:
  KernelGemm.Both{+first,+second} = both
  second

# A cell is the same cell of the geometry with the groups at the other end
# at the same outer index and group.
def cell_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+from: Grid.Position,+to: Grid.Position,
  +first_outer: Nat,+first_group: Nat,+second_outer: Nat,+second_group: Nat,+same_outer: {first_outer == second_outer : Nat},+same_group: {first_group == second_group : Nat}) ->
  {Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,lanes,groups,from},first_outer,first_group) ==
    Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,lanes,groups,to},second_outer,second_group) : Tile.Tile8}:
  +geometry = {Cells.Cells{outer,after,lanes,groups,from} : Cells.Cells}
  Equal.trans(Tile.Tile8,Representation.cell(~Logical,~original,logical,geometry,first_outer,first_group),Representation.cell(~Logical,~original,logical,geometry,second_outer,first_group),
    Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,lanes,groups,to},second_outer,second_group),
    Equal.cong(Nat,Tile.Tile8,number => Representation.cell(~Logical,~original,logical,geometry,number,first_group),first_outer,second_outer,same_outer),
    Equal.cong(Nat,Tile.Tile8,number => Representation.cell(~Logical,~original,logical,geometry,second_outer,number),first_group,second_group,same_group))

# ---- The view reads the cells at the other end ----

# The view of cells with the groups innermost that puts them outermost reads,
# at every index, the cell with the groups outermost.
def outer_agrees(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+limits: Machine.Limits,+index: U32,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +others_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{}))) == True{} : Bool}) ->
  {ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},0,[]},index) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},index) : Tile.Tile8}:
  +inside_axis = Bounds.strict_le(axis,List.length(&2,U32,values),within)
  +below_cells = {%within : {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Grid.placed(_,Grid.Outermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool} below :
    {Nat.is_lt(U32.to_nat(index),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}}
  +view_inside = {%Equal.sym(List<&2,U32>,Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) :
      {Nat.is_lt(U32.to_nat(index),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(_))) == True{} : Bool}
    below_cells : {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}))) == True{} : Bool}}
  +cells_count = Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))
  +address_below = swapped_below(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),below_cells)
  +back = Bounds.roundtrip(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),cells_count,U32.to_nat(Machine.capacity(limits)),address_below,cells_room)))
  +facts = swapped_facts(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),below_cells)
  %Equal.sym(U32,(Views.mirrored_offset(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],0) + Box.stepped_sum(Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),ViewPart.view_steps(Views.mirrored_axes(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[])),U32.to_nat(index)) : U32),U32.from_nat(Nat.add(U32.to_nat(0),ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],U32.to_nat(index)))),ViewReads.view_address(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],0,U32.to_nat(index),view_inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},_) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},index) : Tile.Tile8}
  %Equal.sym(Nat,ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],U32.to_nat(index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),outer_address(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)),limits,U32.to_nat(index),below_cells,others_room,cells_room)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},U32.from_nat(Nat.add(U32.to_nat(0),_))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},index) : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),back) :
    {Representation.cell(~Logical,~original,logical,Public.geometry_of(values,axis,Grid.Innermost{}),Representation.cell_outer(Public.geometry_of(values,axis,Grid.Innermost{}),_),Representation.cell_group(Public.geometry_of(values,axis,Grid.Innermost{}),_)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},index) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Innermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Public.geometry_exact(values,axis,Grid.Innermost{},inside_axis)) :
    {Representation.cell(~Logical,~original,logical,_,Representation.cell_outer(_,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))))),Representation.cell_group(_,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},index) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Outermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Public.geometry_exact(values,axis,Grid.Outermost{},inside_axis)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Representation.cell_outer(Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))))),Representation.cell_group(Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))))) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},index) : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),ShapeLaws.elements(Grid.removed(values,axis)),Coordinates.outer_word(limits,values,axis,others_room)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Representation.outer_at(Grid.Innermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))))),Representation.group_at(Grid.Innermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))))) ==
      Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Representation.outer_at(Grid.Outermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),Representation.group_at(Grid.Outermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index))) : Tile.Tile8}
  cell_moved(~Logical,~original,logical,(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{},Grid.Outermost{},
    Representation.outer_at(Grid.Innermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))))),Representation.group_at(Grid.Innermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))))),
    Representation.outer_at(Grid.Outermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),Representation.group_at(Grid.Outermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),
    first_of(Nat.div(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.mod(U32.to_nat(index),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.mod(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.div(U32.to_nat(index),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),facts),second_of(Nat.div(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.mod(U32.to_nat(index),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.mod(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(U32.to_nat(index),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.div(U32.to_nat(index),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),facts))

# The view of cells with the groups outermost that puts them innermost reads,
# at every index, the cell with the groups innermost.
def inner_agrees(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+limits: Machine.Limits,+index: U32,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +others_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool}) ->
  {ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),0,[]},index) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},index) : Tile.Tile8}:
  +inside_axis = Bounds.strict_le(axis,List.length(&2,U32,values),within)
  +placed_below = {%within : {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Grid.placed(_,Grid.Innermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool} below :
    {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Grid.placed(True{},Grid.Innermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}}
  +below_cells = {%RowMajor.multiply_commutative(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
    %Public.placed_cells(Grid.Innermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} placed_below :
    {Nat.is_lt(U32.to_nat(index),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}}
  +positive = RowMajor.left_factor_positive(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)),{%RowMajor.multiply_commutative(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,U32.to_nat(index),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Order.zero_le(U32.to_nat(index)),below_cells) : {Nat.is_lt(0n,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}})
  +view_inside = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]))),Nat.mul(ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),
        ViewReads.appended_count(Tiled.inner_axes(Grid.removed(values,axis),1),Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis)))) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1)),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),1)) : {Nat.is_lt(U32.to_nat(index),Nat.mul(ShapeLaws.elements(_),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}
    below_cells : {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])))) == True{} : Bool}}
  +cells_count = Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))
  +address_below = swapped_below(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)),below_cells)
  +cells_room = {%RowMajor.multiply_commutative(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} cells_room : {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +back = Bounds.roundtrip(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),cells_count,U32.to_nat(Machine.capacity(limits)),address_below,cells_room)))
  +facts = swapped_facts(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)),below_cells)
  %Equal.sym(U32,(Views.mirrored_offset(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],0) + Box.stepped_sum(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),ViewPart.view_steps(Views.mirrored_axes(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[])),U32.to_nat(index)) : U32),U32.from_nat(Nat.add(U32.to_nat(0),ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],U32.to_nat(index)))),ViewReads.view_address(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],0,U32.to_nat(index),view_inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},_) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},index) : Tile.Tile8}
  %Equal.sym(Nat,ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],U32.to_nat(index)),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),inner_address(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)),limits,U32.to_nat(index),below_cells,others_room,positive)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},U32.from_nat(Nat.add(U32.to_nat(0),_))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},index) : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),back) :
    {Representation.cell(~Logical,~original,logical,Public.geometry_of(values,axis,Grid.Outermost{}),Representation.cell_outer(Public.geometry_of(values,axis,Grid.Outermost{}),_),Representation.cell_group(Public.geometry_of(values,axis,Grid.Outermost{}),_)) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},index) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Outermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Public.geometry_exact(values,axis,Grid.Outermost{},inside_axis)) :
    {Representation.cell(~Logical,~original,logical,_,Representation.cell_outer(_,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis))))),Representation.cell_group(_,Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},index) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Innermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Public.geometry_exact(values,axis,Grid.Innermost{},inside_axis)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Representation.cell_outer(Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis))))),Representation.cell_group(Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))))) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},index) : Tile.Tile8}
  %Equal.sym(Nat,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),ShapeLaws.elements(Grid.removed(values,axis)),Coordinates.outer_word(limits,values,axis,others_room)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Representation.outer_at(Grid.Outermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis))))),Representation.group_at(Grid.Outermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))))) ==
      Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Representation.outer_at(Grid.Innermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),Representation.group_at(Grid.Innermost{},_,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index))) : Tile.Tile8}
  cell_moved(~Logical,~original,logical,(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{},Grid.Innermost{},
    Representation.outer_at(Grid.Outermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis))))),Representation.group_at(Grid.Outermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis))))),
    Representation.outer_at(Grid.Innermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),Representation.group_at(Grid.Innermost{},ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),U32.to_nat(index)),
    second_of(Nat.div(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.mod(U32.to_nat(index),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.mod(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.div(U32.to_nat(index),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),facts),first_of(Nat.div(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.mod(U32.to_nat(index),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),Nat.mod(Nat.add(Nat.mul(Nat.div(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(U32.to_nat(index),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),Representation.least_one(ShapeLaws.elements(Grid.removed(values,axis)))),Nat.div(U32.to_nat(index),Representation.least_one(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),facts))

# ---- The tiled form with its groups at the other end ----

def cells_length(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {List.length(&2,U32,Tiled.cell_shape(values,axis,position)) == List.length(&2,U32,values) : Nat}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),True{},within) : {List.length(&2,U32,Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == List.length(&2,U32,values) : Nat}
  Equal.trans(Nat,List.length(&2,U32,Grid.placed(True{},position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),1n+List.length(&2,U32,Grid.removed(values,axis)),List.length(&2,U32,values),
    Public.placed_length(position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Public.removed_length(values,axis,within))

# A view of cells of the contract with the groups at one end, that reads every
# cell from where the other end puts it, gives cells of the contract with the
# groups at the other end.
def moved_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+from: Grid.Position,+to: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,+axes: List<&2,Views.Axis>,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,from)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,from),fill,Tensors.Usable{},workers}),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +extents: {Views.axis_extents(axes) == Tiled.cell_shape(values,axis,to) : List<&2,U32>},
  +source_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,from)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,to)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,to)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,to))) == True{} : Bool},
  +fits: {Views.fits(Views.View{axes,0},Tensors.count_or_zero(Tiled.cell_shape(values,axis,from))) == True{} : Bool},
  agrees: @+index: U32 -> @+below: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,axis,to))) == True{} : Bool} ->
    {ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,axis,from)},axes,0,[]},index) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,to)},index) : Tile.Tile8}) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,from),fill,Tensors.Usable{},workers},axes,0,[]),axis,lanes,to}):
  +target = Tiled.cell_shape(values,axis,to)
  +target_counted = StorageCount.counted_within(limits,target,cells_room,default_capacity)
  +count = Global.count(target,target_counted)
  +accepted = Global.accepted(target,target_counted)
  +nonzero = Arithmetic.word_positive_nonzero(count,{%Global.whole_counted(target,count,accepted) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive :
    {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}})
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(axes),target,extents)
  +axes_accepted = {%back : {Tensors.count_of(_) == Some{count} : Maybe<&2,U32>} accepted : {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>}}
  +axes_rank = {%back : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} target_rank : {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +room = {%back : {Nat.is_le(ShapeLaws.elements(_),ShapeLaws.elements(target)) == True{} : Bool} Order.reflexive(ShapeLaws.elements(target)) :
    {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),ShapeLaws.elements(target)) == True{} : Bool}}
  +viewed = Global.restated_below(~Tile.Tile8,~ViewGather.Viewing<Representation.Lifted<Logical>>,~Representation.Lifted<Logical>,~ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Representation.lifted(~Logical,~original),
    ~ShapeLaws.elements(target),ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(values,axis,from)},axes,0,[]},Representation.Lifted{logical,Public.geometry_of(values,axis,to)},agrees,Views.axis_extents(axes),Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Tile.Tile8),~Tensors.unchanged(~Tile.Tile8)),
      Tasks.components(Tiled.cell_shape(values,axis,from),Tensors.grid_of(Tiled.cell_shape(values,axis,from)),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,0},[]),
      Tensors.Then{Tensors.Viewed{Tiled.cell_shape(values,axis,from),Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,0,[]},Unit{}},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,axis,from),0)],Views.axis_extents(axes),fill,workers),Views.axis_extents(axes),fill,Tensors.Usable{},workers},
    KernelGemm.Both{Global.usable(~Tile.Tile8,Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Tile.Tile8),~Tensors.unchanged(~Tile.Tile8)),
      Tasks.components(Tiled.cell_shape(values,axis,from),Tensors.grid_of(Tiled.cell_shape(values,axis,from)),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,0},[]),
      Tensors.Then{Tensors.Viewed{Tiled.cell_shape(values,axis,from),Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,0,[]},Unit{}},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,axis,from),0)],Views.axis_extents(axes),fill,workers),Views.axis_extents(axes),fill,Tensors.Usable{},workers},{==}),
      ViewMoved.view_contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,from)},limits,model,Tiled.cell_shape(values,axis,from),axes,0,[],fill,workers,global,
        default_capacity,source_rank,axes_rank,count,axes_accepted,nonzero,fits)},
    room,Machine.capacity(limits),cells_room)
  %Equal.sym(Bool,Views.fits(Views.View{axes,0},Tensors.count_or_zero(Tiled.cell_shape(values,axis,from))),True{},fits) :
    Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved_status(~Tile.Tile8,~Unit,~Tensors.unchanged(~Tile.Tile8),Unit{},"repositioned",Tensors.Usable{},_,Tensors.count_of(Views.axis_extents(axes)),PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,from),Views.axis_extents(axes),axes,0,[],fill,workers),axis,lanes,to})
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Views.axis_extents(axes)),Some{count},axes_accepted) :
    Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved_status(~Tile.Tile8,~Unit,~Tensors.unchanged(~Tile.Tile8),Unit{},"repositioned",Tensors.Usable{},True{},_,PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,from),Views.axis_extents(axes),axes,0,[],fill,workers),axis,lanes,to})
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) :
    Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved_counted(~Tile.Tile8,~Unit,~Tensors.unchanged(~Tile.Tile8),Unit{},_,PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,from),
      Views.axis_extents(axes),axes,0,[],fill,workers),axis,lanes,to})
  KernelGemm.Both{%extents : Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,to)},_,
      Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,Unit>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~Unit,~Tensors.view_gather(~Tile.Tile8),~Tensors.unchanged(~Tile.Tile8)),
      Tasks.components(Tiled.cell_shape(values,axis,from),Tensors.grid_of(Tiled.cell_shape(values,axis,from)),Tensors.grid_of(Views.axis_extents(axes)),Views.View{axes,0},[]),
      Tensors.Then{Tensors.Viewed{Tiled.cell_shape(values,axis,from),Tensors.grid_of(Views.axis_extents(axes)),Views.axis_extents(axes),axes,0,[]},Unit{}},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(Tiled.cell_shape(values,axis,from),0)],Views.axis_extents(axes),fill,workers),Views.axis_extents(axes),fill,Tensors.Usable{},workers})
      viewed,
    Public.Laid{inside,{==},
      {%pinned : {Tiled.logical_shape(target,axis,_,to) == values : List<&2,U32>}
        Public.cells_restored(values,axis,to) : {Tiled.logical_shape(target,axis,lanes,to) == values : List<&2,U32>}},
      counted,pinned}}

# The view that puts the groups outermost reaches below the cells.
def outer_span(+values: List<&2,U32>,+axis: Nat,+limits: Machine.Limits,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +others_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}) ->
  {Nat.is_lt(ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}:
  +count = ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}))
  +same = Equal.cong(List<&2,U32>,Nat,extents => Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(extents)),Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))))
  +axes_positive = {%Equal.sym(Nat,count,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),same) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,count) == True{} : Bool}}
  +last = Nat.sub(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),1n)
  +inside_last = Order.last_below(Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),positive)
  +span = Equal.trans(Nat,ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],Nat.sub(count,1n)),ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],last),
    Equal.sym(Nat,ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],Nat.sub(count,1n)),ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),ViewReads.last_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},axes_positive)),
    Equal.cong(Nat,Nat,whole => ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],Nat.sub(whole,1n)),count,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),same))
  +address = outer_address(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)),limits,last,inside_last,others_room,cells_room)
  %Equal.sym(Nat,ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],last),span) : {Nat.is_lt(_,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}
  %Equal.sym(Nat,ViewReads.view_sum(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},[],last),Nat.add(Nat.mul(Nat.div(last,ShapeLaws.elements(Grid.removed(values,axis))),1n),Nat.mul(Nat.mod(last,ShapeLaws.elements(Grid.removed(values,axis))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))),address) : {Nat.is_lt(_,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}
  swapped_below(last,ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),inside_last)

# The view that puts the groups innermost reaches below the cells.
def inner_span(+values: List<&2,U32>,+axis: Nat,+limits: Machine.Limits,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +others_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}) ->
  {Nat.is_lt(ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}:
  +count = ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])))
  +same = Equal.trans(Nat,count,Nat.mul(ShapeLaws.elements(Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1))),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),
    ViewReads.appended_count(Tiled.inner_axes(Grid.removed(values,axis),1),Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))),
    Equal.cong(List<&2,U32>,Nat,extents => Nat.mul(ShapeLaws.elements(extents),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1)),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),1)))
  +axes_positive = {%Equal.sym(Nat,count,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),same) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,count) == True{} : Bool}}
  +last = Nat.sub(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n)
  +inside_last = Order.last_below(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),positive)
  +span = Equal.trans(Nat,ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],Nat.sub(count,1n)),ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],last),
    Equal.sym(Nat,ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],Nat.sub(count,1n)),ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),ViewReads.last_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),axes_positive)),
    Equal.cong(Nat,Nat,whole => ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],Nat.sub(whole,1n)),count,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),same))
  +address = inner_address(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)),limits,last,inside_last,others_room,DivisionOrder.right_factor_positive(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),positive))
  %Equal.sym(Nat,ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],last),span) : {Nat.is_lt(_,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}
  %Equal.sym(Nat,ViewReads.view_sum(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),[],last),Nat.add(Nat.mul(Nat.div(last,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),1n),Nat.mul(Nat.mod(last,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),ShapeLaws.elements(Grid.removed(values,axis)))),address) : {Nat.is_lt(_,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}
  swapped_below(last,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)),inside_last)

# Cells of the contract with the groups innermost, viewed with them outermost.
def outer_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Innermost{}),fill,Tensors.Usable{},workers}),
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +from_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool}>>>,
  +to_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{}))) == True{} : Bool}>>>) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Innermost{}),fill,Tensors.Usable{},workers},Tiled.repositioned_view(Grid.Outermost{},Tiled.cell_shape(values,axis,Grid.Innermost{})),0,[]),axis,lanes,Grid.Outermost{}}):
  KernelGemm.Both{+others_room,KernelGemm.Both{+lanes_room,KernelGemm.Both{+from_room,+from_positive}}} = from_bounds
  KernelGemm.Both{+to_others_room,KernelGemm.Both{+to_lanes_room,KernelGemm.Both{+to_room,+to_positive}}} = to_bounds
  +within_back = Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),True{},within)
  +from_cells = Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Public.cells_split(values,axis,Grid.Innermost{},within),RowMajor.multiply_commutative(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))))
  +cells_room = {%from_cells : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} from_room : {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +positive = {%Public.cells_split(values,axis,Grid.Outermost{},within) : {Nat.is_lt(0n,_) == True{} : Bool} to_positive : {Nat.is_lt(0n,Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) == True{} : Bool}}
  +extents = {%within_back : {Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}) == Grid.placed(_,Grid.Outermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) : List<&2,U32>}
    Equal.cong(List<&2,U32>,List<&2,U32>,later => Grid.groups(Grid.lanes_at(values,axis)) <> later,Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) :
    {Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}) == Tiled.cell_shape(values,axis,Grid.Outermost{}) : List<&2,U32>}}
  +source_counted = StorageCount.counted_within(limits,Tiled.cell_shape(values,axis,Grid.Innermost{}),from_room,default_capacity)
  +source_count = Global.count(Tiled.cell_shape(values,axis,Grid.Innermost{}),source_counted)
  +bound = {%Equal.sym(Nat,U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Innermost{}))),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),
        Equal.trans(Nat,U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Innermost{}))),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),
          ViewMoved.counted_cells(Tiled.cell_shape(values,axis,Grid.Innermost{}),source_count,Global.accepted(Tiled.cell_shape(values,axis,Grid.Innermost{}),source_counted)),from_cells)) :
      {Nat.is_lt(Nat.add(U32.to_nat(0),ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))})),_) == True{} : Bool}
    outer_span(values,axis,limits,within,others_room,cells_room,positive) : {Nat.is_lt(Nat.add(U32.to_nat(0),ViewBounds.last_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))})),U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Innermost{})))) == True{} : Bool}}
  +nonempty_axes = ViewMoved.none_zero(Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),{%Equal.sym(List<&2,U32>,Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}),Tiled.cell_shape(values,axis,Grid.Outermost{}),extents) :
      {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} to_positive : {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))}))) == True{} : Bool}})
  %Equal.sym(List<&2,Views.Axis>,Tiled.repositioned_view(Grid.Outermost{},Tiled.cell_shape(values,axis,Grid.Innermost{})),Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},{%within_back : {Tiled.repositioned_view(Grid.Outermost{},Grid.placed(_,Grid.Innermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))} : List<&2,Views.Axis>} outer_view(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) :
      {Tiled.repositioned_view(Grid.Outermost{},Tiled.cell_shape(values,axis,Grid.Innermost{})) == Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))} : List<&2,Views.Axis>}}) :
    Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Innermost{}),fill,Tensors.Usable{},workers},_,0,[]),axis,lanes,Grid.Outermost{}})
  moved_contract(~Logical,~original,logical,values,axis,lanes,Grid.Innermost{},Grid.Outermost{},limits,default_capacity,model,fill,workers,Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},global,inside,counted,pinned,extents,
    {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})),List.length(&2,U32,values),cells_length(values,axis,Grid.Innermost{},within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Outermost{})),List.length(&2,U32,values),cells_length(values,axis,Grid.Outermost{},within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Outermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    to_room,to_positive,ViewBounds.fits_span(Con{Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),1},Tiled.inner_axes(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))},0,Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Innermost{})),nonempty_axes,bound),
    index => below => outer_agrees(~Logical,~original,logical,values,axis,limits,index,within,others_room,cells_room,below))

# Cells of the contract with the groups outermost, viewed with them innermost.
def inner_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Outermost{}),fill,Tensors.Usable{},workers}),
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +from_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{}))) == True{} : Bool}>>>,
  +to_bounds: KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(Grid.lanes_at(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},{Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool}>>>) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Outermost{}),fill,Tensors.Usable{},workers},Tiled.repositioned_view(Grid.Innermost{},Tiled.cell_shape(values,axis,Grid.Outermost{})),0,[]),axis,lanes,Grid.Innermost{}}):
  KernelGemm.Both{+others_room,KernelGemm.Both{+lanes_room,KernelGemm.Both{+from_room,+from_positive}}} = from_bounds
  KernelGemm.Both{+to_others_room,KernelGemm.Both{+to_lanes_room,KernelGemm.Both{+to_room,+to_positive}}} = to_bounds
  +within_back = Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),True{},within)
  +from_cells = Public.cells_split(values,axis,Grid.Outermost{},within)
  +cells_room = {%Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Public.cells_split(values,axis,Grid.Outermost{},within),RowMajor.multiply_commutative(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} from_room : {Nat.is_le(Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +positive = {%Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))),Public.cells_split(values,axis,Grid.Innermost{},within),RowMajor.multiply_commutative(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis)))) :
      {Nat.is_lt(0n,_) == True{} : Bool} to_positive : {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))))) == True{} : Bool}}
  +extents = {%within_back : {Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])) == Grid.placed(_,Grid.Innermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) : List<&2,U32>}
    Equal.trans(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),List.append(&2,U32,Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1)),[Grid.groups(Grid.lanes_at(values,axis))]),List.append(&2,U32,Grid.removed(values,axis),[Grid.groups(Grid.lanes_at(values,axis))]),
      ViewReads.appended_extents(Tiled.inner_axes(Grid.removed(values,axis),1),Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))),
      Equal.cong(List<&2,U32>,List<&2,U32>,front => List.append(&2,U32,front,[Grid.groups(Grid.lanes_at(values,axis))]),Views.axis_extents(Tiled.inner_axes(Grid.removed(values,axis),1)),Grid.removed(values,axis),inner_extents(Grid.removed(values,axis),1))) :
    {Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])) == Tiled.cell_shape(values,axis,Grid.Innermost{}) : List<&2,U32>}}
  +source_counted = StorageCount.counted_within(limits,Tiled.cell_shape(values,axis,Grid.Outermost{}),from_room,default_capacity)
  +source_count = Global.count(Tiled.cell_shape(values,axis,Grid.Outermost{}),source_counted)
  +bound = {%Equal.sym(Nat,U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Outermost{}))),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),
        Equal.trans(Nat,U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Outermost{}))),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Outermost{})),Nat.mul(U32.to_nat(Grid.groups(Grid.lanes_at(values,axis))),ShapeLaws.elements(Grid.removed(values,axis))),
          ViewMoved.counted_cells(Tiled.cell_shape(values,axis,Grid.Outermost{}),source_count,Global.accepted(Tiled.cell_shape(values,axis,Grid.Outermost{}),source_counted)),from_cells)) :
      {Nat.is_lt(Nat.add(U32.to_nat(0),ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]))),_) == True{} : Bool}
    inner_span(values,axis,limits,within,others_room,cells_room,positive) : {Nat.is_lt(Nat.add(U32.to_nat(0),ViewBounds.last_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]))),U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Outermost{})))) == True{} : Bool}}
  +nonempty_axes = ViewMoved.none_zero(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),{%Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])),Tiled.cell_shape(values,axis,Grid.Innermost{}),extents) :
      {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} to_positive : {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}])))) == True{} : Bool}})
  %Equal.sym(List<&2,Views.Axis>,Tiled.repositioned_view(Grid.Innermost{},Tiled.cell_shape(values,axis,Grid.Outermost{})),List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),{%within_back : {Tiled.repositioned_view(Grid.Innermost{},Grid.placed(_,Grid.Outermost{},Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]) : List<&2,Views.Axis>} {==} :
      {Tiled.repositioned_view(Grid.Innermost{},Tiled.cell_shape(values,axis,Grid.Outermost{})) == List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]) : List<&2,Views.Axis>}}) :
    Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Outermost{}),fill,Tensors.Usable{},workers},_,0,[]),axis,lanes,Grid.Innermost{}})
  moved_contract(~Logical,~original,logical,values,axis,lanes,Grid.Outermost{},Grid.Innermost{},limits,default_capacity,model,fill,workers,List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),global,inside,counted,pinned,extents,
    {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Outermost{})),List.length(&2,U32,values),cells_length(values,axis,Grid.Outermost{},within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Outermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})),List.length(&2,U32,values),cells_length(values,axis,Grid.Innermost{},within)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
      {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    to_room,to_positive,ViewBounds.fits_span(List.append(&2,Views.Axis,Tiled.inner_axes(Grid.removed(values,axis),1),[Views.Axis{Grid.groups(Grid.lanes_at(values,axis)),Grid.product(Grid.removed(values,axis))}]),0,Tensors.count_or_zero(Tiled.cell_shape(values,axis,Grid.Outermost{})),nonempty_axes,bound),
    index => below => inner_agrees(~Logical,~original,logical,values,axis,limits,index,within,others_room,cells_room,below))

def repositioned_moved(~Logical: Data,~original: Logical -> U32 -> F32,own: Grid.Position,position: Grid.Position,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+workers: U32,
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,own),fill,Tensors.Usable{},workers}),
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,+pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+different: {Grid.same_position(own,position) == False{} : Bool}) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,own),fill,Tensors.Usable{},workers},Tiled.repositioned_view(position,Tiled.cell_shape(values,axis,own)),0,[]),axis,lanes,position}):
  match own position:
    case Grid.Innermost{} Grid.Outermost{}:
      outer_moved(~Logical,~original,logical,values,axis,lanes,limits,default_capacity,model,fill,workers,global,within,inside,counted,pinned,rank_room,Coordinates.axis_cells(True{},limits,values,axis,Grid.Innermost{},within,room,nonempty),Coordinates.axis_cells(True{},limits,values,axis,Grid.Outermost{},within,room,nonempty))
    case Grid.Outermost{} Grid.Innermost{}:
      inner_moved(~Logical,~original,logical,values,axis,lanes,limits,default_capacity,model,fill,workers,global,within,inside,counted,pinned,rank_room,Coordinates.axis_cells(True{},limits,values,axis,Grid.Outermost{},within,room,nonempty),Coordinates.axis_cells(True{},limits,values,axis,Grid.Innermost{},within,room,nonempty))
    case Grid.Innermost{} Grid.Innermost{}:
      Empty.absurd(Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Innermost{}),fill,Tensors.Usable{},workers},Tiled.repositioned_view(Grid.Innermost{},Tiled.cell_shape(values,axis,Grid.Innermost{})),0,[]),axis,lanes,Grid.Innermost{}}),
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},different)))
    case Grid.Outermost{} Grid.Outermost{}:
      Empty.absurd(Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,Grid.Outermost{}),fill,Tensors.Usable{},workers},Tiled.repositioned_view(Grid.Outermost{},Tiled.cell_shape(values,axis,Grid.Outermost{})),0,[]),axis,lanes,Grid.Outermost{}}),
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},different)))

def not_false(value: Bool,+evidence: {Bool.not(value) == False{} : Bool}) -> {value == True{} : Bool}:
  match value:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},evidence)))

def repositioned_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+own: Grid.Position,+position: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +different: {Grid.same_position(own,position) == False{} : Bool},+cells_within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+shaped: {Tiled.cell_shape(values,axis,own) == shape : List<&2,U32>},+counted: Global.Counted<values>,
  +pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>) ->
  Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},Tiled.repositioned_view(position,shape),0,[]),axis,lanes,position}):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  +moved_global = {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},_)
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers})}
  +cells_global = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,own),shape,shaped) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers})
    moved_global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(values,axis,own),model_fill,Tensors.Usable{},model_workers})}
  +own_within = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,own),shape,shaped) : {Nat.is_lt(axis,List.length(&2,U32,_)) == True{} : Bool} cells_within :
    {Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,own))) == True{} : Bool}}
  +within = Equal.trans(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,own))),True{},
    Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,own))),Nat.is_lt(axis,List.length(&2,U32,values)),Public.cells_within(values,axis,own)),own_within)
  %equation : Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",_,Tiled.repositioned_view(position,shape),0,[]),axis,lanes,position})
  %shaped : Public.ShapedLaid(~Logical,~original,logical,values,Tiled.Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers},Tiled.repositioned_view(position,_),0,[]),axis,lanes,position})
  repositioned_moved(~Logical,~original,own,position,logical,values,axis,lanes,limits,default_capacity,model,model_fill,model_workers,cells_global,within,inside,counted,pinned,rank_room,
    StorageCount.counted_room(limits,values,counted,default_capacity),nonempty,different)

# ---- The unit axis at either position ----

# A cell of one group at either position: its outer index is the index and
# its group 0, when the index lies below the outer indices.
def one_group(position: Grid.Position,+outer: Nat,+index: Nat,+inside: {Nat.is_lt(index,outer) == True{} : Bool}) ->
  KernelGemm.Both<{Representation.outer_at(position,outer,1n,index) == index : Nat},{Representation.group_at(position,outer,1n,index) == 0n : Nat}>:
  match position:
    case Grid.Innermost{}:
      +exact = Equal.trans(Nat,Nat.add(Nat.mul(index,1n),0n),Nat.mul(index,1n),index,Addition.zero_right(Nat.mul(index,1n)),Order.times_one(index))
      +split = Division.unique(index,1n,index,0n,Equal.sym(Nat,Nat.add(Nat.mul(index,1n),0n),index,exact),{==},{==})
      KernelGemm.Both{%Equal.sym(Nat & Nat,Nat.divmod(index,1n),(index,0n),split) : {Pair.fst(Nat,Nat,_) == index : Nat} {==},
        %Equal.sym(Nat & Nat,Nat.divmod(index,1n),(index,0n),split) : {Pair.snd(Nat,Nat,_) == 0n : Nat} {==}}
    case Grid.Outermost{}:
      +positive = Order.le_lt_trans(0n,index,outer,Order.zero_le(index),inside)
      +split = Division.small(index,outer,inside)
      %Equal.sym(Nat,Representation.least_one(outer),outer,Representation.least_positive(outer,positive)) :
        KernelGemm.Both<{Nat.mod(index,_) == index : Nat},{Nat.div(index,_) == 0n : Nat}>
      KernelGemm.Both{%Equal.sym(Nat & Nat,Nat.divmod(index,outer),(0n,index),split) : {Pair.snd(Nat,Nat,_) == index : Nat} {==},
        %Equal.sym(Nat & Nat,Nat.divmod(index,outer),(0n,index),split) : {Pair.fst(Nat,Nat,_) == 0n : Nat} {==}}

# A lifted cell of one lane and one group at a position is the cell of its
# index and group 0, below the outer indices.
def one_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+outer: U32,+after: U32,+position: Grid.Position,+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(outer)) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,1,1,position}},index) ==
    Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,Grid.Innermost{}},U32.to_nat(index),0n) : Tile.Tile8}:
  +number = U32.to_nat(index)
  +count = U32.to_nat(outer)
  +split = one_group(position,count,number,inside)
  %Equal.sym(Nat,Representation.outer_at(position,count,1n,number),number,
      first_of(Representation.outer_at(position,count,1n,number),number,Representation.group_at(position,count,1n,number),0n,split)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,position},_,Representation.group_at(position,count,1n,number)) ==
      Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,Grid.Innermost{}},number,0n) : Tile.Tile8}
  %Equal.sym(Nat,Representation.group_at(position,count,1n,number),0n,
      second_of(Representation.outer_at(position,count,1n,number),number,Representation.group_at(position,count,1n,number),0n,split)) :
    {Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,position},number,_) ==
      Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,Grid.Innermost{}},number,0n) : Tile.Tile8}
  {==}

# On the unit axis the lifted cells of values read alike at either position
# below the values.
def unit_lifted(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+own: Grid.Position,+position: Grid.Position,+limits: Machine.Limits,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +index: U32,+below: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(values)) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,own)},index) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},index) : Tile.Tile8}:
  +after = Tiled.product(Public.lane_after(values,axis))
  +outer = (Public.lane_prefix(values,axis,1) * after : U32)
  +lanes = Grid.lanes_at(values,axis)
  +others = Grid.product(Grid.removed(values,axis))
  +word = Equal.trans(U32,outer,(1 * others : U32),Grid.product(values),Public.prefix_after(values,axis,1),
    Equal.trans(U32,(1 * others : U32),others,Grid.product(values),Public.one_times(others),
      Equal.cong(List<&2,U32>,U32,extents => Grid.product(extents),Grid.removed(values,axis),values,Public.removed_past(values,axis,past))))
  +exact = Equal.trans(Nat,U32.to_nat(outer),U32.to_nat(Grid.product(values)),ShapeLaws.elements(values),Equal.cong(U32,Nat,U32.to_nat,outer,Grid.product(values),word),
    Public.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),values,room))
  +inside_outer = {%Equal.sym(Nat,U32.to_nat(outer),ShapeLaws.elements(values),exact) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} below :
    {Nat.is_lt(U32.to_nat(index),U32.to_nat(outer)) == True{} : Bool}}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,own),Cells.Cells{outer,after,lanes,Tiled.groups(lanes),own},Public.geometry_exact(values,axis,own,inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},index) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},index) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},Public.geometry_exact(values,axis,position,inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,lanes,Tiled.groups(lanes),own}},index) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},index) : Tile.Tile8}
  %Equal.sym(U32,lanes,1,Public.lanes_past(values,axis,past)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,_,Tiled.groups(_),own}},index) ==
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,_,Tiled.groups(_),position}},index) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,1,1,own}},index),
    Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,Grid.Innermost{}},U32.to_nat(index),0n),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,1,1,position}},index),
    one_cell(~Logical,~original,logical,outer,after,own,index,inside_outer),
    Equal.sym(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{outer,after,1,1,position}},index),
      Representation.cell(~Logical,~original,logical,Cells.Cells{outer,after,1,1,Grid.Innermost{}},U32.to_nat(index),0n),
      one_cell(~Logical,~original,logical,outer,after,position,index,inside_outer)))

# The cells of values with their lanes on the unit axis are the values.
def unit_cells(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool}) ->
  {Tiled.cell_shape(values,axis,position) == values : List<&2,U32>}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),False{},past) :
    {Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))) == values : List<&2,U32>}
  Public.removed_past(values,axis,past)

# A cell of values with their lanes on the unit axis is of group 0.
def group_unit(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+word: U32,+past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) -> {Representation.cell_group(Public.geometry_of(values,axis,position),U32.to_nat(word)) == 0n : Nat}:
  +outer_count = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
  +removed = Grid.removed(values,axis)
  +word_inside = {%unit_cells(values,axis,position,past) : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(_)) == True{} : Bool} inside_word :
    {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(values)) == True{} : Bool}}
  +removed_inside = {%Equal.sym(List<&2,U32>,removed,values,Public.removed_past(values,axis,past)) : {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(_)) == True{} : Bool} word_inside :
    {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(removed)) == True{} : Bool}}
  +outer_inside = {%Equal.sym(Nat,U32.to_nat(outer_count),ShapeLaws.elements(removed),Coordinates.outer_word(limits,values,axis,room)) : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool}
    removed_inside : {Nat.is_lt(U32.to_nat(word),U32.to_nat(outer_count)) == True{} : Bool}}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),Cells.Cells{outer_count,Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position},Public.geometry_exact(values,axis,position,inside)) :
    {Representation.cell_group(_,U32.to_nat(word)) == 0n : Nat}
  %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,past)) :
    {Representation.cell_group(Cells.Cells{outer_count,Tiled.product(Public.lane_after(values,axis)),_,Tiled.groups(_),position},U32.to_nat(word)) == 0n : Nat}
  second_of(Representation.outer_at(position,U32.to_nat(outer_count),1n,U32.to_nat(word)),U32.to_nat(word),Representation.group_at(position,U32.to_nat(outer_count),1n,U32.to_nat(word)),0n,
    one_group(position,U32.to_nat(outer_count),U32.to_nat(word),outer_inside))

# Tiled values with their lanes on the unit axis hold the same values with
# their groups at another position.
def unit_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,+axis: Nat,+lanes: U32,+own: Grid.Position,+position: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,own})) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}):
  match cells:
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
      +room = StorageCount.counted_room(limits,values,counted,default_capacity)
      +values_shape = Equal.trans(List<&2,U32>,values,Tiled.cell_shape(values,axis,own),shape,Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,own),values,unit_cells(values,axis,own,past)),shaped)
      +moved_shaped = Equal.trans(List<&2,U32>,Tiled.cell_shape(values,axis,position),values,shape,unit_cells(values,axis,position,past),values_shape)
      +moved_restored = {%moved_shaped : {Tiled.logical_shape(_,axis,lanes,position) == values : List<&2,U32>}
        %pinned : {Tiled.logical_shape(Tiled.cell_shape(values,axis,position),axis,_,position) == values : List<&2,U32>}
        Public.cells_restored(values,axis,position) : {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>}}
      +shape_room = {%values_shape : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
        {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +shaped_cells = Global.shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,own)},shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==},{==},global)
      +moved = Global.restated_below(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.lifted(~Logical,~original),~ShapeLaws.elements(shape),
        Representation.Lifted{logical,Public.geometry_of(values,axis,own)},Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
        index => below => unit_lifted(~Logical,~original,logical,values,axis,own,position,limits,room,past,inside,index,
          {%Equal.sym(List<&2,U32>,values,shape,values_shape) : {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(_)) == True{} : Bool} below :
            {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(values)) == True{} : Bool}}),
        shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},shaped_cells,Order.reflexive(ShapeLaws.elements(shape)),Machine.capacity(limits),shape_room)
      KernelGemm.Both{Global.contract_of(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,position)},shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},moved),
        Public.Laid{inside,moved_shaped,moved_restored,counted,pinned}}
    case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)

# Tiled values hold the same values with their groups at another position
# when the positions agree or the lanes lie on the unit axis.
def position_free(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,cells: Tensors.Tensor<Tile.Tile8>,+axis: Nat,+lanes: U32,own: Grid.Position,position: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +free: {Grid.same_position(own,position) || Bool.not(Nat.is_lt(axis,List.length(&2,U32,values))) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,own})) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{cells,axis,lanes,position}):
  match own position:
    case Grid.Innermost{} Grid.Innermost{}: contract
    case Grid.Outermost{} Grid.Outermost{}: contract
    case Grid.Innermost{} Grid.Outermost{}:
      unit_moved(~Logical,~original,logical,values,cells,axis,lanes,Grid.Innermost{},Grid.Outermost{},limits,default_capacity,Boolean.not_true(Nat.is_lt(axis,List.length(&2,U32,values)),free),contract)
    case Grid.Outermost{} Grid.Innermost{}:
      unit_moved(~Logical,~original,logical,values,cells,axis,lanes,Grid.Outermost{},Grid.Innermost{},limits,default_capacity,Boolean.not_true(Nat.is_lt(axis,List.length(&2,U32,values)),free),contract)

# The cells of the contract have the lane axis inside them when the values do.
def cells_rank(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,
  +workers: U32,+axis: Nat,+lanes: U32,+own: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,own})) ->
  {Nat.is_lt(axis,List.length(&2,U32,shape)) == Nat.is_lt(axis,List.length(&2,U32,values)) : Bool}:
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  Equal.trans(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,own))),Nat.is_lt(axis,List.length(&2,U32,values)),
    Equal.cong(List<&2,U32>,Bool,cells => Nat.is_lt(axis,List.length(&2,U32,cells)),shape,Tiled.cell_shape(values,axis,own),Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,own),shape,shaped)),
    Public.cells_within(values,axis,own))
