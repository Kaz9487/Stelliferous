# What a view that the runtime accepts gives the gather of the view: its
# strides, the axes it steps, the blocks of its source and the rows and
# classes of its readings fit the words.

import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_tasks.bend as Tasks
import ./u32_division.bend as U32Division
import ./natural_addition.bend as Addition
import ./nat_ceiling.bend as NatCeiling
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_view_reads.bend as ViewReads
import ./u32_extrema.bend as Extrema
import ../tensor_grid.bend as Grid
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_view_gather.bend as ViewGather
import ./tensor_components.bend as Components
import ./tensor_move_readback.bend as Held
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Logic
import ./nat_to_u32_bounds.bend as Bounds
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds

# ---- Strides ----

def less_one(+value: Nat,+above: {Nat.is_lt(1n,value) == True{} : Bool}) -> {Nat.is_lt(0n,Nat.sub(value,1n)) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_lt(0n,Nat.sub(0n,1n)) == True{} : Bool},WordEncoding.false_true(above))
    case 1n+ +rest: {%Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.is_lt(0n,_) == True{} : Bool} above : {Nat.is_lt(0n,Nat.sub(rest,0n)) == True{} : Bool}}

# A stride of an axis that spans more than one index fits the span of the
# last coordinates.
def stride_within(+limits: Machine.Limits,+extent: U32,+stride: U32,+later: Nat,
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +span_le: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),later),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}):
  +cap = Machine.capacity(limits)
  +minus = Arithmetic.difference_exact(cap,LimitBounds.capacity_fits(limits),extent,1,U32.to_nat(extent),1n,{==},{==},Order.positive_one(U32.to_nat(extent),positive),room)
  +span = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))
  Held.by_cases(Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}),U32.is_lt(1,extent),
    +wide => {%Equal.sym(Bool,U32.is_lt(1,extent),True{},wide) : Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool})
      LimitBounds.double_bound(limits,U32.to_nat(stride),
        Order.transitive(U32.to_nat(stride),span,U32.to_nat(cap),
          DivisionOrder.le_multiple(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride),
            {%Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),minus) : {Nat.is_lt(0n,_) == True{} : Bool}
              less_one(U32.to_nat(extent),{%Comparison.lt(1,extent) : {_ == True{} : Bool} wide : {Nat.is_lt(U32.to_nat(1),U32.to_nat(extent)) == True{} : Bool}}) :
              {Nat.is_lt(0n,U32.to_nat((extent - 1 : U32))) == True{} : Bool}}),
          Order.transitive(span,Nat.add(span,later),U32.to_nat(cap),Order.le_add(span,later),span_le))) : Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool})},
    +narrow => {%Equal.sym(Bool,U32.is_lt(1,extent),False{},narrow) : Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) Unit{} : Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool})})

def strides_from_span(+limits: Machine.Limits,axes: List<&2,Views.Axis>,
  +span_le: {Nat.is_le(ViewBounds.last_span(axes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  ViewGather.ViewStrides(axes,Machine.half_word(limits)):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +span = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))
      Held.Both{stride_within(limits,extent,stride,ViewBounds.last_span(rest),
          Order.transitive(U32.to_nat(extent),whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),room),
          extent_positive,span_le),
        strides_from_span(limits,rest,
          Order.transitive(ViewBounds.last_span(rest),Nat.add(span,ViewBounds.last_span(rest)),U32.to_nat(Machine.capacity(limits)),
            Order.right_le(span,ViewBounds.last_span(rest)),span_le),
          Order.transitive(inner,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room),inner_positive)}

# Every axis of a view that fits a source within the machine steps at most
# half a word where it spans more than one index.
def view_strides(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +length_room: {Nat.is_le(U32.to_nat(length),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_ok: Components.ShapeOk(limits,Views.axis_extents(axes))) -> ViewGather.ViewStrides(axes,Machine.half_word(limits)):
  Held.Both{+room,Held.Both{+nonempty,+rank_room}} = target_ok
  +base = U32.to_nat(offset)
  +last = {%ViewBounds.last_location(axes,base) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} ViewBounds.last_address(axes,offset,length,fits) :
    {Nat.is_lt(Nat.add(base,ViewBounds.last_span(axes)),U32.to_nat(length)) == True{} : Bool}}
  strides_from_span(limits,axes,
    Order.transitive(ViewBounds.last_span(axes),U32.to_nat(length),U32.to_nat(Machine.capacity(limits)),
      Order.transitive(ViewBounds.last_span(axes),Nat.add(base,ViewBounds.last_span(axes)),U32.to_nat(length),Order.right_le(base,ViewBounds.last_span(axes)),
        Bounds.strict_le(Nat.add(base,ViewBounds.last_span(axes)),U32.to_nat(length),last)),length_room),
    room,nonempty)

# ---- The blocks of the source ----

# A block that holds at least one index and fits the machine fits half a
# word twice over.
def block_double(+limits: Machine.Limits,+block: U32,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  %Equal.sym(U32,U32.max(block,1),block,Laws.maximum_kept(block,1,Order.positive_one(U32.to_nat(block),positive))) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  LimitBounds.double_bound(limits,U32.to_nat(block),room)

def pick_positive(choice: Bool,+left: U32,+right: U32,+left_ok: {Nat.is_lt(0n,U32.to_nat(left)) == True{} : Bool},
  +right_ok: {Nat.is_lt(0n,U32.to_nat(right)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Bool.pick(U32,choice,left,right))) == True{} : Bool}:
  match choice:
    case True{}: left_ok
    case False{}: right_ok

def pick_room(choice: Bool,+left: U32,+right: U32,+cap: U32,+left_ok: {Nat.is_le(U32.to_nat(left),U32.to_nat(cap)) == True{} : Bool},
  +right_ok: {Nat.is_le(U32.to_nat(right),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(Bool.pick(U32,choice,left,right)),U32.to_nat(cap)) == True{} : Bool}:
  match choice:
    case True{}: left_ok
    case False{}: right_ok

def blocks_flat(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,extents: List<&2,U32>,+at: U32,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  ViewReads.BlocksIn(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},extents,at,Machine.half_word(limits)):
  match extents:
    case Nil{}: Unit{}
    case Con{+axis_extent,+rest}:
      +cap = Machine.capacity(limits)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(U32.to_nat(axis_extent),inner)
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis_extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(axis_extent),inner,positive)
      +axis_room = Order.transitive(U32.to_nat(axis_extent),whole,U32.to_nat(cap),RowMajor.product_covers(U32.to_nat(axis_extent),inner,inner_positive),room)
      +one_positive = Order.le_lt_trans(0n,0n,U32.to_nat(axis_extent),{==},axis_positive)
      +block_positive = Laws.shape_block(limits,outer,extent,unit_cells,block,per,shape)
      +block_room = Order.transitive(U32.to_nat(block),U32.to_nat(extent),U32.to_nat(cap),Laws.shape_below(limits,outer,extent,unit_cells,block,per,shape),
        Laws.extent_room(limits,outer,extent,unit_cells,block,per,shape))
      +one_room = Order.transitive(U32.to_nat(1),U32.to_nat(axis_extent),U32.to_nat(cap),Order.positive_one(U32.to_nat(axis_extent),axis_positive),axis_room)
      +inner_pick = Bool.pick(U32,U32.is_eq((at + 1 : U32),prefix),block,axis_extent)
      +picked = Bool.pick(U32,U32.is_lt((at + 1 : U32),prefix),1,inner_pick)
      +picked_positive = pick_positive(U32.is_lt((at + 1 : U32),prefix),1,inner_pick,{==},
        pick_positive(U32.is_eq((at + 1 : U32),prefix),block,axis_extent,block_positive,axis_positive))
      +picked_room = pick_room(U32.is_lt((at + 1 : U32),prefix),1,inner_pick,cap,one_room,
        pick_room(U32.is_eq((at + 1 : U32),prefix),block,axis_extent,cap,block_room,axis_room))
      Held.Both{picked_positive,Held.Both{block_double(limits,picked,picked_positive,picked_room),
        blocks_flat(limits,count,prefix,outer,extent,unit_cells,block,per,shape,rest,(at + 1 : U32),
          Order.transitive(inner,whole,U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(axis_extent),inner,axis_positive),room),inner_positive)}}

def blocks_facts(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+facts: Laws.GridFacts<limits,Grid.grid(extents,target)>,
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  ViewReads.BlocksIn(Grid.grid(extents,target),extents,0,Machine.half_word(limits)):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  %Equal.sym(Grid.Grid,Grid.grid(extents,target),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) : ViewReads.BlocksIn(_,extents,0,Machine.half_word(limits))
  blocks_flat(limits,count,prefix,outer,extent,unit_cells,block,per,shape,extents,0,room,positive)

# The blocks of the grid of a shape hold indices and fit half a word.
def grid_blocks(+limits: Machine.Limits,+extents: List<&2,U32>,+target: U32,+shape_ok: Components.ShapeOk(limits,extents)) ->
  ViewReads.BlocksIn(Grid.grid(extents,target),extents,0,Machine.half_word(limits)):
  Held.Both{+room,Held.Both{+positive,+rank_room}} = shape_ok
  blocks_facts(limits,extents,target,GridRanges.grid_facts(limits,extents,target,room,positive),room,positive)

# A block that holds an index and fits half a word twice over fits the
# capacity.
def block_within(+limits: Machine.Limits,+block: U32,+positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Laws.maximum_kept(block,1,Order.positive_one(U32.to_nat(block),positive)) : {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Components.halved_le(U32.to_nat(U32.max(block,1)),U32.to_nat(Machine.capacity(limits)),
    {%LimitBounds.doubled_capacity(limits) : {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),_) == True{} : Bool} double :
      {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),Nat.double(U32.to_nat(Machine.capacity(limits)))) == True{} : Bool}})

# Two numbers within the capacity sum within half a word.
def pair_half(+limits: Machine.Limits,+left: Nat,+right: Nat,+left_room: {Nat.is_le(left,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(left,right),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +cap = U32.to_nat(Machine.capacity(limits))
  +sum = Order.transitive(Nat.add(left,right),Nat.add(cap,right),Nat.add(cap,cap),Order.add_le(left,cap,right,left_room),Order.offset_le(cap,right,cap,right_room))
  +doubled = {%GridRanges.double_add(cap) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
    LimitBounds.double_bound(limits,cap,Order.reflexive(cap)) : {Nat.is_le(Nat.add(cap,cap),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  Order.transitive(Nat.add(left,right),Nat.add(cap,cap),U32.to_nat(Machine.half_word(limits)),sum,doubled)

# ---- The axes a view steps ----

# The extent a stepped source axis has at its place, walked from at.
def ExtentAt(extents: List<&2,U32>,+at: U32,+axis: U32,+extent: U32) -> Data:
  match extents:
    case Nil{}:
      Unit
    case Con{+head,rest}:
      Held.Both<Logic.When(U32.is_eq(axis,at),{extent == head : U32}),ExtentAt(rest,(at + 1 : U32),axis,extent)>

# What a stepped source axis tells: a fixed coordinate, an extent and a block
# within the machine, a step at most the stride, and its place among the
# extents walked from at.
def Found(found: Tasks.Stepped,+extents: List<&2,U32>,+stride: U32,+limits: Machine.Limits,+grid: Grid.Grid,+at: U32,+position: Nat) -> Data:
  match found:
    case Tasks.Unstepped{}:
      Unit
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      Held.Both<{Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
        Held.Both<{Nat.is_le(U32.to_nat(step),U32.to_nat(stride)) == True{} : Bool},
          Held.Both<{Nat.is_le(U32.to_nat(source_extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
            Held.Both<{Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,axis,source_extent),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
              Held.Both<ExtentAt(extents,at,axis,source_extent),
                Held.Both<{Nat.is_le(position,U32.to_nat(axis)) == True{} : Bool},
                  {Nat.is_lt(U32.to_nat(axis),Nat.add(position,List.length(&2,U32,extents))) == True{} : Bool}>>>>>>

# What the axis a nonzero stride steps tells, and a source of one cell when
# none is found.
def StepFit(found: Tasks.Stepped,+extents: List<&2,U32>,+stride: U32,+limits: Machine.Limits,+grid: Grid.Grid,+at: U32,+position: Nat) -> Data:
  match found:
    case Tasks.Unstepped{}:
      {Nat.is_le(ShapeLaws.elements(extents),1n) == True{} : Bool}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      Found(Tasks.Stepped{axis,step,source_extent,first},extents,stride,limits,grid,at,position)

def found_of(found: Tasks.Stepped,+extents: List<&2,U32>,+stride: U32,+limits: Machine.Limits,+grid: Grid.Grid,+at: U32,+position: Nat,
  +facts: StepFit(found,extents,stride,limits,grid,at,position)) -> Found(found,extents,stride,limits,grid,at,position):
  match found:
    case Tasks.Unstepped{}: Unit{}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}: facts

def one_of(+value: Nat,+small: {Nat.is_le(value,1n) == True{} : Bool},+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {value == 1n : Nat}:
  Order.antisymmetric(value,1n,small,Order.positive_one(value,positive))

def and_trues(a: Bool) -> {(a && True{} && True{} && True{}) == a : Bool}:
  match a:
    case True{}: {==}
    case False{}: {==}

# The test of an axis whose later tests hold is whether it spans more than
# one index.
def wide_test(+a: Bool,+b: Bool,+c: Bool,+d: Bool,+b_true: {b == True{} : Bool},+c_true: {c == True{} : Bool},+d_true: {d == True{} : Bool}) ->
  {(a && b && c && d) == a : Bool}:
  %Equal.sym(Bool,b,True{},b_true) : {(a && _ && c && d) == a : Bool}
  %Equal.sym(Bool,c,True{},c_true) : {(a && True{} && _ && d) == a : Bool}
  %Equal.sym(Bool,d,True{},d_true) : {(a && True{} && True{} && _) == a : Bool}
  and_trues(a)

# A word below the position of another differs from it.
def word_before(+axis: U32,+at: U32,+position: Nat,+exact: {U32.to_nat(at) == position : Nat},+before: {Nat.is_lt(U32.to_nat(axis),position) == True{} : Bool}) ->
  {U32.is_eq(axis,at) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),False{},Comparison.eq(axis,at),
    {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(U32.to_nat(axis),_) == False{} : Bool}
      Order.less_not_equal(U32.to_nat(axis),position,before) : {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == False{} : Bool}})

# An axis before the place walked from claims no extent there or later.
def extent_past(extents: List<&2,U32>,+at: U32,+position: Nat,+axis: U32,+extent: U32,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == position : Nat},+before: {Nat.is_lt(U32.to_nat(axis),position) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ExtentAt(extents,at,axis,extent):
  match extents:
    case Nil{}: Unit{}
    case Con{+head,+rest}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(position,length),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),cap,Order.le_add(1n+position,length),shifted))
      Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,at),False{},word_before(axis,at,position,exact,before)) : Logic.When(_,{extent == head : U32})
          Unit{} : Logic.When(U32.is_eq(axis,at),{extent == head : U32})},
        extent_past(rest,(at + 1 : U32),1n+position,axis,extent,limits,next,
          Order.lt_succ(U32.to_nat(axis),position,Bounds.strict_le(U32.to_nat(axis),position,before)),shifted)}

def when_same(enabled: Bool,+value: U32) -> Logic.When(enabled,{value == value : U32}):
  match enabled:
    case False{}: Unit{}
    case True{}: {==}

# The extent at index index of a list of extents walked from at is its
# extent there.
def extent_index(extents: List<&2,U32>,index: Nat,+at: U32,+position: Nat,+limits: Machine.Limits,+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +axis_room: {Nat.is_le(Nat.add(position,index),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,extents)) == True{} : Bool}) ->
  ExtentAt(extents,at,U32.from_nat(Nat.add(position,index)),Grid.lanes_at(extents,index)):
  match extents index:
    case Con{+head,+rest} 0n:
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,rest)
      +axis = U32.from_nat(Nat.add(position,0n))
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(position,length),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),cap,Order.le_add(1n+position,length),shifted))
      +back = Equal.trans(Nat,U32.to_nat(axis),Nat.add(position,0n),position,Bounds.roundtrip(Nat.add(position,0n),Machine.capacity(limits),LimitBounds.capacity_fits(limits),axis_room),
        Addition.zero_right(position))
      +before = {%Equal.sym(Nat,U32.to_nat(axis),position,back) : {Nat.is_lt(_,1n+position) == True{} : Bool}
        Order.lt_succ(position,position,Order.reflexive(position)) : {Nat.is_lt(U32.to_nat(axis),1n+position) == True{} : Bool}}
      Held.Both{when_same(U32.is_eq(axis,at),head),extent_past(rest,(at + 1 : U32),1n+position,axis,head,limits,next,before,shifted)}
    case Con{+head,+rest} 1n+ +k:
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,rest)
      +axis = U32.from_nat(Nat.add(position,1n+k))
      +value = {1n+Nat.add(position,k) : Nat}
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(position,length),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),cap,Order.le_add(1n+position,length),shifted))
      +axis_exact = Equal.trans(Nat,U32.to_nat(axis),Nat.add(position,1n+k),value,
        Bounds.roundtrip(Nat.add(position,1n+k),Machine.capacity(limits),LimitBounds.capacity_fits(limits),axis_room),Addition.successor_right(position,k))
      +apart = Equal.trans(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),False{},Comparison.eq(axis,at),
        {%Equal.sym(Nat,U32.to_nat(axis),value,axis_exact) : {Nat.is_eq(_,U32.to_nat(at)) == False{} : Bool}
          %Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(value,_) == False{} : Bool}
          Equal.trans(Bool,Nat.is_eq(value,position),Nat.is_eq(position,value),False{},Order.equal_symmetric(value,position),
            Order.less_not_equal(position,value,Order.lt_succ(position,Nat.add(position,k),Order.le_add(position,k)))) :
          {Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)) == False{} : Bool}})
      +later = extent_index(rest,k,(at + 1 : U32),1n+position,limits,next,shifted,
        {%Addition.successor_right(position,k) : {Nat.is_le(_,cap) == True{} : Bool} axis_room : {Nat.is_le(value,cap) == True{} : Bool}},inside)
      Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,at),False{},apart) : Logic.When(_,{Grid.lanes_at(rest,k) == head : U32})
          Unit{} : Logic.When(U32.is_eq(axis,at),{Grid.lanes_at(rest,k) == head : U32})},
        {%Equal.sym(Nat,Nat.add(position,1n+k),value,Addition.successor_right(position,k)) :
            ExtentAt(rest,(at + 1 : U32),U32.from_nat(_),Grid.lanes_at(rest,k))
          later : ExtentAt(rest,(at + 1 : U32),axis,Grid.lanes_at(rest,k))}}
    case Nil{} +index:
      Empty.absurd(ExtentAt(Nil{},at,U32.from_nat(Nat.add(position,index)),Grid.lanes_at(Nil{},index)),Order.below_zero(index,inside))

# An axis the test skips keeps what the later axes tell, and when none of
# them steps the axis spans at most one index.
def fit_down(found: Tasks.Stepped,+extent: U32,+rest: List<&2,U32>,+stride: U32,+limits: Machine.Limits,+grid: Grid.Grid,+at: U32,+position: Nat,
  +exact: {U32.to_nat(at) == position : Nat},
  +test: {(U32.is_lt(1,extent) && U32.is_lt(0,Grid.product(rest)) && U32.is_le(Grid.product(rest),stride) && U32.is_zero((stride % U32.max(Grid.product(rest),1) : U32))) == False{} : Bool},
  +inner_exact: {U32.to_nat(Grid.product(rest)) == ShapeLaws.elements(rest) : Nat},
  +inner_room: {Nat.is_le(ShapeLaws.elements(rest),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,ShapeLaws.elements(rest)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +later: StepFit(found,rest,stride,limits,grid,(at + 1 : U32),1n+position)) -> StepFit(found,Con{extent,rest},stride,limits,grid,at,position):
  match found:
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      Held.Both{+first_room,Held.Both{+step_le,Held.Both{+extent_room,Held.Both{+block_room,Held.Both{+extent_at,Held.Both{+after,+below}}}}}} = later
      +length = List.length(&2,U32,rest)
      +apart = ViewReads.word_apart(axis,at,position,exact,Order.succ_le_lt(position,U32.to_nat(axis),after))
      Held.Both{first_room,Held.Both{step_le,Held.Both{extent_room,Held.Both{block_room,
        Held.Both{Held.Both{{%Equal.sym(Bool,U32.is_eq(axis,at),False{},apart) : Logic.When(_,{source_extent == extent : U32})
              Unit{} : Logic.When(U32.is_eq(axis,at),{source_extent == extent : U32})},extent_at},
          Held.Both{Order.transitive(position,1n+position,U32.to_nat(axis),Order.successor(position,position,Order.reflexive(position)),after),
            {%Equal.sym(Nat,Nat.add(position,1n+length),1n+Nat.add(position,length),Addition.successor_right(position,length)) :
                {Nat.is_lt(U32.to_nat(axis),_) == True{} : Bool}
              below : {Nat.is_lt(U32.to_nat(axis),Nat.add(position,1n+length)) == True{} : Bool}}}}}}}}
    case Tasks.Unstepped{}:
      +inner = ShapeLaws.elements(rest)
      +size = U32.to_nat(extent)
      +value = U32.to_nat(stride)
      +inner_one = one_of(inner,later,inner_positive)
      +own_one = Arithmetic.to_nat_injective(Grid.product(rest),1,Equal.trans(Nat,U32.to_nat(Grid.product(rest)),inner,1n,inner_exact,inner_one))
      +test_one = {%own_one : {(U32.is_lt(1,extent) && U32.is_lt(0,_) && U32.is_le(_,stride) && U32.is_zero((stride % U32.max(_,1) : U32))) == False{} : Bool}
        test : {(U32.is_lt(1,extent) && U32.is_lt(0,1) && U32.is_le(1,stride) && U32.is_zero((stride % 1 : U32))) == False{} : Bool}}
      +le_one = Equal.trans(Bool,U32.is_le(1,stride),Nat.is_le(U32.to_nat(1),value),True{},Comparison.le(1,stride),Order.positive_one(value,stride_positive))
      +one_room = {%inner_one : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} inner_room : {Nat.is_le(1n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +remainder = U32Division.modulo(stride,1,Machine.half_word(limits),LimitBounds.half_word_fits(limits),LimitBounds.double_bound(limits,1n,one_room),{==})
      +remainder_zero = ViewReads.below_one(Nat.mod(value,1n),DivisionOrder.remainder_small(value,1n,{==}))
      +word_zero = Arithmetic.to_nat_injective((stride % 1 : U32),0,
        Equal.trans(Nat,U32.to_nat((stride % 1 : U32)),Nat.mod(value,1n),0n,remainder,Equal.sym(Nat,0n,Nat.mod(value,1n),remainder_zero)))
      +whole = {%Equal.sym(U32,(stride % 1 : U32),0,word_zero) : {U32.is_zero(_) == True{} : Bool} {==} : {U32.is_zero((stride % 1 : U32)) == True{} : Bool}}
      +cond = (U32.is_lt(1,extent) && U32.is_lt(0,1) && U32.is_le(1,stride) && U32.is_zero((stride % 1 : U32)))
      +same = wide_test(U32.is_lt(1,extent),U32.is_lt(0,1),U32.is_le(1,stride),U32.is_zero((stride % 1 : U32)),{==},le_one,whole)
      +narrow = Equal.trans(Bool,U32.is_lt(1,extent),cond,False{},Equal.sym(Bool,cond,U32.is_lt(1,extent),same),test_one)
      +narrow_value = Equal.trans(Bool,Nat.is_lt(U32.to_nat(1),size),U32.is_lt(1,extent),False{},Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(U32.to_nat(1),size),Comparison.lt(1,extent)),narrow)
      %Equal.sym(Nat,inner,1n,inner_one) : {Nat.is_le(Nat.mul(size,_),1n) == True{} : Bool}
      %Equal.sym(Nat,Nat.mul(size,1n),size,Order.times_one(size)) : {Nat.is_le(_,1n) == True{} : Bool}
      Order.not_less(1n,size,narrow_value)

# The axis the test picks, or what a later axis tells.
def fit_choice(choice: Bool,+extent: U32,+rest: List<&2,U32>,+first: U32,+more: List<&2,U32>,+at: U32,+position: Nat,+stride: U32,+limits: Machine.Limits,+grid: Grid.Grid,
  +test: {Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride) == choice : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+next: {U32.to_nat((at + 1 : U32)) == 1n+position : Nat},
  +shifted: {Nat.is_le(Nat.add(1n+position,List.length(&2,U32,rest)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first_below: {Nat.is_lt(U32.to_nat(first),U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,at,extent),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inner_exact: {U32.to_nat(Grid.product(rest)) == ShapeLaws.elements(rest) : Nat},
  +inner_double: {Nat.is_le(Nat.double(ShapeLaws.elements(rest)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inner_room: {Nat.is_le(ShapeLaws.elements(rest),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,ShapeLaws.elements(rest)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +later: StepFit(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),rest,stride,limits,grid,(at + 1 : U32),1n+position)) ->
  StepFit(Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Con{first,more},choice,stride,at),Con{extent,rest},stride,limits,grid,at,position):
  match choice:
    case True{}:
      +own = U32.max(Grid.product(rest),1)
      +inner = ShapeLaws.elements(rest)
      +length = List.length(&2,U32,rest)
      +whole = ViewReads.and_right((U32.is_lt(1,extent) && U32.is_lt(0,Grid.product(rest)) && U32.is_le(Grid.product(rest),stride)),U32.is_zero((stride % own : U32)),test)
      +step = (stride / own : U32)
      +step_exact = ViewReads.step_found(stride,own,inner,Machine.half_word(limits),LimitBounds.half_word_fits(limits),Extrema.max_one_exact(Grid.product(rest),inner,inner_exact,inner_positive),
        inner_double,inner_positive,whole)
      +back = Equal.sym(Nat,U32.to_nat(at),position,exact)
      Held.Both{Order.transitive(U32.to_nat(first),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(U32.to_nat(first),U32.to_nat(extent),first_below),extent_room),
        Held.Both{{%step_exact : {Nat.is_le(U32.to_nat(step),_) == True{} : Bool} RowMajor.product_covers(U32.to_nat(step),inner,inner_positive) :
            {Nat.is_le(U32.to_nat(step),U32.to_nat(stride)) == True{} : Bool}},
          Held.Both{extent_room,Held.Both{block_double,
            Held.Both{Held.Both{{%Equal.sym(Bool,U32.is_eq(at,at),True{},ViewReads.word_same(at)) : Logic.When(_,{extent == extent : U32})
                  {==} : Logic.When(U32.is_eq(at,at),{extent == extent : U32})},
                extent_past(rest,(at + 1 : U32),1n+position,at,extent,limits,next,
                  {%back : {Nat.is_lt(_,1n+position) == True{} : Bool} Order.lt_succ(position,position,Order.reflexive(position)) :
                    {Nat.is_lt(U32.to_nat(at),1n+position) == True{} : Bool}},shifted)},
              Held.Both{{%back : {Nat.is_le(position,_) == True{} : Bool} Order.reflexive(position) : {Nat.is_le(position,U32.to_nat(at)) == True{} : Bool}},
                {%back : {Nat.is_lt(_,Nat.add(position,1n+length)) == True{} : Bool} Order.below_sum(position,length) :
                  {Nat.is_lt(U32.to_nat(at),Nat.add(position,1n+length)) == True{} : Bool}}}}}}}}
    case False{}:
      fit_down(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),extent,rest,stride,limits,grid,at,position,exact,test,inner_exact,inner_room,inner_positive,stride_positive,later)

# What the axis a nonzero stride steps tells, walked with fixed coordinates
# inside the source.
def step_fit(extents: List<&2,U32>,fixed: List<&2,U32>,+stride: U32,+at: U32,+position: Nat,+limits: Machine.Limits,+grid: Grid.Grid,
  +fixed_in: ViewReads.FixedIn(fixed,extents),+blocks: ViewReads.BlocksIn(grid,extents,at,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+exact: {U32.to_nat(at) == position : Nat},
  +length_room: {Nat.is_le(Nat.add(position,List.length(&2,U32,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  StepFit(Tasks.stepped_axis(extents,fixed,stride,at),extents,stride,limits,grid,at,position):
  match extents fixed:
    case Con{+extent,+rest} Con{+first,+more}:
      Held.Both{+first_below,+more_in} = fixed_in
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = blocks
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(size,inner)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +inner_room = Order.transitive(inner,whole,cap,ViewReads.inner_le(size,inner,positive),room)
      +extent_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +length = List.length(&2,U32,rest)
      +shifted = {%Addition.successor_right(position,length) : {Nat.is_le(_,cap) == True{} : Bool} length_room : {Nat.is_le(1n+Nat.add(position,length),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,length),cap,Order.le_add(1n+position,length),shifted))
      %Equal.sym(Tasks.Stepped,Tasks.stepped_axis(Con{extent,rest},Con{first,more},stride,at),
          Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Con{first,more},Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),stride,at),
          ViewReads.stepped_con(extent,rest,Con{first,more},stride,at)) :
        StepFit(_,Con{extent,rest},stride,limits,grid,at,position)
      fit_choice(Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),extent,rest,first,more,at,position,stride,limits,grid,{==},
        exact,next,shifted,first_below,extent_room,block_double,ViewReads.product_value(rest,Machine.capacity(limits),LimitBounds.capacity_fits(limits),inner_room,inner_positive),
        LimitBounds.double_bound(limits,inner,inner_room),inner_room,inner_positive,stride_positive,
        step_fit(rest,more,stride,(at + 1 : U32),1n+position,limits,grid,more_in,blocks_rest,inner_room,inner_positive,stride_positive,next,shifted))
    case Nil{} +fixed:
      {==}
    case Con{+extent,+rest} Nil{}:
      %Equal.sym(Tasks.Stepped,Tasks.stepped_axis(Con{extent,rest},Nil{},stride,at),
          Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Nil{},Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),stride,at),
          ViewReads.stepped_con(extent,rest,Nil{},stride,at)) :
        StepFit(_,Con{extent,rest},stride,limits,grid,at,position)
      Empty.absurd(StepFit(Tasks.Unstepped{},Con{extent,rest},stride,limits,grid,at,position),fixed_in)

# An axis a view steps reaches its last index within half a word, and one it
# does not step spans nothing.
def axis_reach(found: Tasks.Stepped,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+extent: U32,+stride: U32,
  +facts: StepFit(found,source,stride,limits,grid,0,0n),
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span: {Nat.is_lt(Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)),ShapeLaws.elements(source)) == True{} : Bool}) ->
  ViewReads.AxisFit(found,extent,stride,Machine.half_word(limits)):
  match found:
    case Tasks.Unstepped{}:
      +value = U32.to_nat(stride)
      +elements = ShapeLaws.elements(source)
      +before = Nat.sub(U32.to_nat(extent),1n)
      Equal.sym(Nat,0n,Nat.mul(before,value),ViewReads.below_one(Nat.mul(before,value),Order.lt_le(Nat.mul(before,value),elements,1n,span,facts)))
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      Held.Both{+first_room,Held.Both{+step_le,+others}} = facts
      +cap = U32.to_nat(Machine.capacity(limits))
      +value = U32.to_nat(stride)
      +elements = ShapeLaws.elements(source)
      +before = Nat.sub(U32.to_nat(extent),1n)
      +reach = Nat.mul(U32.to_nat(step),before)
      +moved = RowMajor.mul_le(U32.to_nat(step),value,before,step_le)
      +span_room = {%RowMajor.multiply_commutative(before,value) : {Nat.is_le(_,cap) == True{} : Bool}
        Order.transitive(Nat.mul(before,value),elements,cap,Bounds.strict_le(Nat.mul(before,value),elements,span),source_room) :
        {Nat.is_le(Nat.mul(value,before),cap) == True{} : Bool}}
      pair_half(limits,U32.to_nat(first),reach,first_room,Order.transitive(reach,Nat.mul(value,before),cap,moved,span_room))

# The same, with the span of the axis counted in words.
def axis_fit(found: Tasks.Stepped,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+extent: U32,+stride: U32,
  +facts: StepFit(found,source,stride,limits,grid,0,0n),+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_below: {Nat.is_lt(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ShapeLaws.elements(source)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ViewReads.AxisFit(found,extent,stride,Machine.half_word(limits)):
  +size = U32.to_nat(extent)
  +value = U32.to_nat(stride)
  +elements = ShapeLaws.elements(source)
  +before = Nat.sub(size,1n)
  +minus = Arithmetic.difference_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extent,1,size,1n,{==},{==},Order.positive_one(size,extent_positive),extent_room)
  +span = {%minus : {Nat.is_lt(Nat.mul(_,value),elements) == True{} : Bool} span_below : {Nat.is_lt(Nat.mul(before,value),elements) == True{} : Bool}}
  axis_reach(found,limits,source,grid,extent,stride,facts,source_room,span)

# What a view axis tells: how far it reaches, and what the source axis it
# steps tells.
def AxisFound(found: Tasks.Stepped,+extent: U32,+stride: U32,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid) -> Data:
  Held.Both<ViewReads.AxisFit(found,extent,stride,Machine.half_word(limits)),Found(found,source,stride,limits,grid,0,0n)>

# An unread axis adds nothing: its stride is zero or it has one index.
def unread_span(zero: Bool,+extent: U32,+stride: U32,+classified: {U32.is_zero(stride) == zero : Bool},+unread: {zero || U32.is_eq(extent,1) == True{} : Bool}) ->
  {Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride)) == 0n : Nat}:
  match zero:
    case True{}:
      %Equal.sym(U32,stride,0,Arithmetic.word_zero(stride,classified)) : {Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(_)) == 0n : Nat}
      RowMajor.multiply_zero_right(Nat.sub(U32.to_nat(extent),1n))
    case False{}:
      %Equal.sym(U32,extent,1,Arithmetic.equal_of_is_eq(extent,1,unread)) : {Nat.mul(Nat.sub(U32.to_nat(_),1n),U32.to_nat(stride)) == 0n : Nat}
      {==}

def axis_step(zero: Bool,+limits: Machine.Limits,+source: List<&2,U32>,+fixed: List<&2,U32>,+grid: Grid.Grid,+extent: U32,+stride: U32,
  +classified: {Tasks.unread(extent,stride) == zero : Bool},+fixed_in: ViewReads.FixedIn(fixed,source),+blocks: ViewReads.BlocksIn(grid,source,0,Machine.half_word(limits)),
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_below: {Nat.is_lt(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ShapeLaws.elements(source)) == True{} : Bool}) ->
  AxisFound(Bool.pick(Tasks.Stepped,zero,Tasks.Unstepped{},Tasks.stepped_axis(source,fixed,stride,0)),extent,stride,limits,source,grid):
  match zero:
    case True{}:
      Held.Both{unread_span(U32.is_zero(stride),extent,stride,{==},classified),Unit{}}
    case False{}:
      +found = Tasks.stepped_axis(source,fixed,stride,0)
      +facts = step_fit(source,fixed,stride,0,0n,limits,grid,fixed_in,blocks,source_room,source_positive,Arithmetic.word_nonzero_positive(stride,Logic.or_false_left(U32.is_zero(stride),U32.is_eq(extent,1),classified)),{==},source_rank)
      Held.Both{axis_fit(found,limits,source,grid,extent,stride,facts,extent_positive,extent_room,span_below,source_room),
        found_of(found,source,stride,limits,grid,0,0n,facts)}

def fit_part(found: Tasks.Stepped,+extent: U32,+stride: U32,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,
  +both: AxisFound(found,extent,stride,limits,source,grid)) -> ViewReads.AxisFit(found,extent,stride,Machine.half_word(limits)):
  Held.Both{+fit,+found_ok} = both
  fit

def found_part(found: Tasks.Stepped,+extent: U32,+stride: U32,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,
  +both: AxisFound(found,extent,stride,limits,source,grid)) -> Found(found,source,stride,limits,grid,0,0n):
  Held.Both{+fit,+found_ok} = both
  found_ok

# The premises a view walk needs at every axis.
def AxesPremises(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>) -> Data:
  Held.Both<{Nat.is_lt(ViewBounds.last_span(axes),ShapeLaws.elements(source)) == True{} : Bool},
    Held.Both<{Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
      {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}>>

# Every axis of a view whose last address lies inside its source steps
# within half a word, with the fixed coordinates of the source.
def axes_fit(+limits: Machine.Limits,+source: List<&2,U32>,+fixed: List<&2,U32>,+grid: Grid.Grid,axes: List<&2,Views.Axis>,+fixed_in: ViewReads.FixedIn(fixed,source),
  +blocks: ViewReads.BlocksIn(grid,source,0,Machine.half_word(limits)),+source_ok: Components.ShapeOk(limits,source),+premises: AxesPremises(limits,source,axes)) ->
  ViewReads.AxesFit(source,fixed,axes,Machine.half_word(limits)):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = source_ok
      Held.Both{+span_below,Held.Both{+room,+positive}} = premises
      +cap = U32.to_nat(Machine.capacity(limits))
      +elements = ShapeLaws.elements(source)
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +span = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))
      +later = ViewBounds.last_span(rest)
      +both = axis_step(Tasks.unread(extent,stride),limits,source,fixed,grid,extent,stride,{==},fixed_in,blocks,source_room,source_positive,source_rank,extent_positive,
        Order.transitive(U32.to_nat(extent),whole,cap,RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),room),
        Order.le_lt_trans(span,Nat.add(span,later),elements,Order.le_add(span,later),span_below))
      Held.Both{fit_part(Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(source,fixed,stride,0)),extent,stride,limits,source,grid,both),
        axes_fit(limits,source,fixed,grid,rest,fixed_in,blocks,Held.Both{source_room,Held.Both{source_positive,source_rank}},
          Held.Both{Order.le_lt_trans(later,Nat.add(span,later),elements,Order.right_le(span,later),span_below),
            Held.Both{Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room),inner_positive}})}

# ---- The readings of a view ----

# What a read of a source axis tells: its block is the block of its axis,
# its extent the extent at its place, and its place a source axis.
def ReadOk(read: Tasks.Source,+source: List<&2,U32>,+grid: Grid.Grid) -> Data:
  match read:
    case Tasks.Read{+axis,+source_block,step,offset,mirrored,width,divisor,shift,+bound,clamped}:
      Held.Both<{source_block == Grid.block(grid,axis,bound) : U32},
        Held.Both<ExtentAt(source,0,axis,bound),{Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,source)) == True{} : Bool}>>
    case Tasks.Unread{}:
      Unit
    case Tasks.Crossing{}:
      Unit

# Every reading of a row has room at half a word, and what its read tells.
def ReadsOk(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid) -> Data:
  match row:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{+extent,+block,+read},rest}:
      Held.Both<Components.ReadingRoom(extent,block,read,Machine.half_word(limits)),Held.Both<ReadOk(read,source,grid),ReadsOk(rest,limits,source,grid)>>

def reads_cons(+extent: U32,+block: U32,+read: Tasks.Source,+rest: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,
  +head: Held.Both<Components.ReadingRoom(extent,block,read,Machine.half_word(limits)),ReadOk(read,source,grid)>,+tail: ReadsOk(rest,limits,source,grid)) ->
  ReadsOk(Con{Tasks.Reading{extent,block,read},rest},limits,source,grid):
  Held.Both{+room,+read_ok} = head
  Held.Both{room,Held.Both{read_ok,tail}}

# The room a target axis of a view needs at half a word.
def axis_room(+limits: Machine.Limits,+extent: U32,+block: U32,
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +block_room: {Nat.is_le(U32.to_nat(block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Components.AxisRoom(extent,block,Machine.half_word(limits)):
  +ceiling = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  Held.Both{Laws.divided_exact(limits,extent,block,extent_room,block_room,block_positive),
    Held.Both{LimitBounds.within_half(limits,ceiling,Order.transitive(ceiling,U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),
        NatCeiling.bounded(U32.to_nat(extent),U32.to_nat(block),block_positive),extent_room)),
      Held.Both{block_positive,pair_half(limits,U32.to_nat(extent),U32.to_nat(block),extent_room,block_room)}}}

def plus_one(+value: Nat) -> {Nat.add(value,1n) == 1n+value : Nat}:
  Equal.trans(Nat,Nat.add(value,1n),1n+Nat.add(value,0n),1n+value,Addition.successor_right(value,0n),
    Equal.cong(Nat,Nat,sum => 1n+sum,Nat.add(value,0n),value,Addition.zero_right(value)))

def last_plus(+limits: Machine.Limits,+last: U32,+last_room: {Nat.is_le(1n+U32.to_nat(last),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(last),1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(last),1n),1n+U32.to_nat(last),plus_one(U32.to_nat(last))) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
  last_room

# A read reaches its last index within half a word, whichever way it runs.
def reach_of(flag: Bool,+step: U32,+first: U32,+last: U32,+before: Nat,+limits: Machine.Limits,
  +last_exact: {U32.to_nat(last) == Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),before)) : Nat},
  +last_room: {Nat.is_le(1n+U32.to_nat(last),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Components.ReachAt(flag,step,Bool.pick(U32,flag,last,first),1,before,Machine.half_word(limits)):
  match flag:
    case False{}:
      {%last_exact : {Nat.is_le(Nat.add(_,1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} last_plus(limits,last,last_room) :
        {Nat.is_le(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),before)),1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
    case True{}:
      Held.Both{{%Equal.sym(Nat,U32.to_nat(last),Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),before)),last_exact) :
            {Nat.is_le(Nat.mul(U32.to_nat(step),before),_) == True{} : Bool}
          Order.right_le(U32.to_nat(first),Nat.mul(U32.to_nat(step),before)) : {Nat.is_le(Nat.mul(U32.to_nat(step),before),U32.to_nat(last)) == True{} : Bool}},
        last_plus(limits,last,last_room)}

def two_half(+limits: Machine.Limits) -> {Nat.is_le(Nat.double(U32.to_nat(U32.max(1,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(Nat.double(U32.to_nat(U32.max(1,1))),16n,U32.to_nat(Machine.half_word(limits)),{==},
    Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits)))

# A read of a stepped axis that stays inside it has room at half a word and
# reads its axis.
def stepped_ok(fits: Bool,+limits: Machine.Limits,+source: List<&2,U32>,+sgrid: Grid.Grid,+extent: U32,+block: U32,+flag: Bool,
  +axis: U32,+step: U32,+source_extent: U32,+first: U32,
  +test: {U32.is_lt((first + step * (U32.max(extent,1) - 1 : U32) : U32),source_extent) == fits : Bool},
  +room: Components.AxisRoom(extent,block,Machine.half_word(limits)),+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(step),Nat.sub(U32.to_nat(extent),1n))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(U32.to_nat(source_extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(sgrid,axis,source_extent),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +extent_at: ExtentAt(source,0,axis,source_extent),+axis_below: {Nat.is_lt(U32.to_nat(axis),Nat.add(0n,List.length(&2,U32,source))) == True{} : Bool}) ->
  Held.Both<Components.ReadingRoom(extent,block,ViewReads.read_of(ViewReads.stepped_reading(fits,axis,Grid.block(sgrid,axis,source_extent),step,
      Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block)),Machine.half_word(limits)),
    ReadOk(ViewReads.read_of(ViewReads.stepped_reading(fits,axis,Grid.block(sgrid,axis,source_extent),step,
      Bool.pick(U32,flag,(first + step * (U32.max(extent,1) - 1 : U32) : U32),first),flag,source_extent,extent,block)),source,sgrid)>:
  match fits:
    case False{}:
      Held.Both{Held.Both{room,Unit{}},Unit{}}
    case True{}:
      +last = (first + step * (U32.max(extent,1) - 1 : U32) : U32)
      +half = Machine.half_word(limits)
      +before = Nat.sub(U32.to_nat(extent),1n)
      +reach = Nat.mul(U32.to_nat(step),before)
      +kept = Laws.maximum_kept(extent,1,Order.positive_one(U32.to_nat(extent),extent_positive))
      +minus = Arithmetic.difference_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extent,1,U32.to_nat(extent),1n,{==},{==},
        Order.positive_one(U32.to_nat(extent),extent_positive),extent_room)
      +product = Arithmetic.product_exact(half,LimitBounds.half_word_fits(limits),step,(extent - 1 : U32),U32.to_nat(step),before,{==},minus,
        Order.transitive(reach,Nat.add(U32.to_nat(first),reach),U32.to_nat(half),Order.right_le(U32.to_nat(first),reach),fit))
      +sum = Arithmetic.sum_exact(half,LimitBounds.half_word_fits(limits),first,(step * (extent - 1 : U32) : U32),U32.to_nat(first),reach,{==},product,fit)
      +last_exact = {%Equal.sym(U32,U32.max(extent,1),extent,kept) : {U32.to_nat((first + step * (_ - 1 : U32) : U32)) == Nat.add(U32.to_nat(first),reach) : Nat}
        sum : {U32.to_nat(last) == Nat.add(U32.to_nat(first),reach) : Nat}}
      +below = Equal.trans(Bool,Nat.is_lt(U32.to_nat(last),U32.to_nat(source_extent)),U32.is_lt(last,source_extent),True{},
        Equal.sym(Bool,U32.is_lt(last,source_extent),Nat.is_lt(U32.to_nat(last),U32.to_nat(source_extent)),Comparison.lt(last,source_extent)),test)
      +last_room = LimitBounds.within_half(limits,1n+U32.to_nat(last),
        Order.transitive(1n+U32.to_nat(last),U32.to_nat(source_extent),U32.to_nat(Machine.capacity(limits)),Order.strict_follow(U32.to_nat(last),U32.to_nat(source_extent),below),source_room))
      Held.Both{Held.Both{room,Held.Both{{==},Held.Both{reach_of(flag,step,first,last,before,limits,last_exact,last_room),
          Held.Both{two_half(limits),Held.Both{LimitBounds.within_half(limits,U32.to_nat(source_extent),source_room),block_double}}}}},
        Held.Both{{==},Held.Both{extent_at,axis_below}}}

# The reading of a view axis has room at half a word and reads its source
# axis.
def reading_found(found: Tasks.Stepped,+limits: Machine.Limits,+source: List<&2,U32>,+sgrid: Grid.Grid,+extent: U32,+block: U32,+flag: Bool,+stride: U32,
  +room: Components.AxisRoom(extent,block,Machine.half_word(limits)),+fit: ViewReads.AxisFit(found,extent,stride,Machine.half_word(limits)),
  +found_ok: Found(found,source,stride,limits,sgrid,0,0n),+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<Components.ReadingRoom(extent,block,ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found)),Machine.half_word(limits)),
    ReadOk(ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found)),source,sgrid)>:
  match found:
    case Tasks.Unstepped{}:
      Held.Both{Held.Both{room,Unit{}},Unit{}}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      Held.Both{+first_room,Held.Both{+step_le,Held.Both{+source_room,Held.Both{+block_double,Held.Both{+extent_at,Held.Both{+after,+axis_below}}}}}} = found_ok
      stepped_ok(U32.is_lt((first + step * (U32.max(extent,1) - 1 : U32) : U32),source_extent),limits,source,sgrid,extent,block,flag,axis,step,source_extent,first,{==},
        room,extent_positive,extent_room,fit,source_room,block_double,extent_at,axis_below)

# Every reading of a view has room at half a word and reads its source axis.
def view_reads(+limits: Machine.Limits,+source: List<&2,U32>,+sgrid: Grid.Grid,+tgrid: Grid.Grid,+fixed: List<&2,U32>,axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+at: U32,
  +fixed_in: ViewReads.FixedIn(fixed,source),+blocks: ViewReads.BlocksIn(sgrid,source,0,Machine.half_word(limits)),
  +target_blocks: ViewReads.BlocksIn(tgrid,Views.axis_extents(axes),at,Machine.half_word(limits)),
  +source_ok: Components.ShapeOk(limits,source),+premises: AxesPremises(limits,source,axes)) ->
  ReadsOk(Tasks.readings(source,sgrid,tgrid,fixed,axes,flags,at),limits,source,sgrid):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = target_blocks
      Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = source_ok
      Held.Both{+span_below,Held.Both{+room,+positive}} = premises
      +cap = U32.to_nat(Machine.capacity(limits))
      +elements = ShapeLaws.elements(source)
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +extent_room = Order.transitive(U32.to_nat(extent),whole,cap,RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),room)
      +span = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride))
      +later = ViewBounds.last_span(rest)
      +block = Grid.block(tgrid,at,extent)
      +flag = Views.flag(flags)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(source,fixed,stride,0))
      +both = axis_step(Tasks.unread(extent,stride),limits,source,fixed,sgrid,extent,stride,{==},fixed_in,blocks,source_room,source_positive,source_rank,extent_positive,extent_room,
        Order.le_lt_trans(span,Nat.add(span,later),elements,Order.le_add(span,later),span_below))
      +head = reading_found(found,limits,source,sgrid,extent,block,flag,stride,
        axis_room(limits,extent,block,extent_room,block_positive,block_within(limits,block,block_positive,block_double)),
        fit_part(found,extent,stride,limits,source,sgrid,both),found_part(found,extent,stride,limits,source,sgrid,both),extent_positive,extent_room)
      +tail = view_reads(limits,source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32),fixed_in,blocks,blocks_rest,
        Held.Both{source_room,Held.Both{source_positive,source_rank}},
        Held.Both{Order.le_lt_trans(later,Nat.add(span,later),elements,Order.right_le(span,later),span_below),
          Held.Both{Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room),inner_positive}})
      %Equal.sym(Tasks.Reading,Tasks.source_reading(sgrid,extent,block,flag,found),
          Tasks.Reading{extent,block,ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found))},ViewReads.reading_parts(found,sgrid,extent,block,flag)) :
        ReadsOk(Con{_,Tasks.readings(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32))},limits,source,sgrid)
      reads_cons(extent,block,ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found)),
        Tasks.readings(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32)),limits,source,sgrid,head,tail)

# ---- What the readings of a row give the components ----

# The readings of a row have room at half a word.
def reads_room(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid)) ->
  Components.ReadingsRoom(row,Machine.half_word(limits)):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = ok
      Held.Both{room,reads_room(rest,limits,source,grid,later)}

# Each reading of a row alone makes a column with room at half a word.
def reads_columns(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid)) ->
  Components.ColumnsOk(List.length(&2,Tasks.Reading,row),[row],Machine.half_word(limits)):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = ok
      Held.Both{Held.Both{Held.Both{room,Unit{}},Unit{}},reads_columns(rest,limits,source,grid,later)}

# The axes a row reads are source axes.
def read_below(read: Tasks.Source,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadOk(read,source,grid),
  +later: Components.AxesBelow(Tasks.read_axes(rest),List.length(&2,U32,source))) ->
  Components.AxesBelow(Tasks.read_axes(Con{Tasks.Reading{extent,block,read},rest}),List.length(&2,U32,source)):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+same,Held.Both{+extent_at,+below}} = ok
      Held.Both{below,later}
    case Tasks.Unread{}: later
    case Tasks.Crossing{}: later

def reads_below(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid)) ->
  Components.AxesBelow(Tasks.read_axes(row),List.length(&2,U32,source)):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = ok
      read_below(read,extent,block,rest,source,grid,read_ok,reads_below(rest,limits,source,grid,later))

# What the reads of a row tell about the source axes walked from at.
def ReadFrom(read: Tasks.Source,+extents: List<&2,U32>,+at: U32,+grid: Grid.Grid) -> Data:
  match read:
    case Tasks.Read{+axis,+source_block,step,offset,mirrored,width,divisor,shift,+bound,clamped}:
      Held.Both<{source_block == Grid.block(grid,axis,bound) : U32},ExtentAt(extents,at,axis,bound)>
    case Tasks.Unread{}:
      Unit
    case Tasks.Crossing{}:
      Unit

def ReadsFrom(row: List<&2,Tasks.Reading>,+extents: List<&2,U32>,+at: U32,+grid: Grid.Grid) -> Data:
  match row:
    case Nil{}:
      Unit
    case Con{Tasks.Reading{extent,block,+read},rest}:
      Held.Both<ReadFrom(read,extents,at,grid),ReadsFrom(rest,extents,at,grid)>

def read_from(read: Tasks.Source,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadOk(read,source,grid)) -> ReadFrom(read,source,0,grid):
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+same,Held.Both{+extent_at,+below}} = ok
      Held.Both{same,extent_at}
    case Tasks.Unread{}: Unit{}
    case Tasks.Crossing{}: Unit{}

def reads_from(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid)) ->
  ReadsFrom(row,source,0,grid):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = ok
      Held.Both{read_from(read,source,grid,read_ok),reads_from(rest,limits,source,grid,later)}

# A read that hits a source axis has the block and the extent of that axis.
def hit_facts(hit: Bool,+axis: U32,+at: U32,+source_block: U32,+bound: U32,+grid: Grid.Grid,+extent: U32,
  +classified: {U32.is_eq(axis,at) == hit : Bool},+same: {source_block == Grid.block(grid,axis,bound) : U32},+when: Logic.When(hit,{bound == extent : U32})) ->
  Components.AxisFacts(hit,source_block,bound,grid,at,extent):
  match hit:
    case False{}: Unit{}
    case True{}:
      +equal = Arithmetic.to_nat_injective(axis,at,Order.equal_of_is_eq(U32.to_nat(axis),U32.to_nat(at),
        Equal.trans(Bool,Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),U32.is_eq(axis,at),True{},
          Equal.sym(Bool,U32.is_eq(axis,at),Nat.is_eq(U32.to_nat(axis),U32.to_nat(at)),Comparison.eq(axis,at)),classified)))
      Held.Both{{%equal : {source_block == Grid.block(grid,_,extent) : U32} %when : {source_block == Grid.block(grid,axis,_) : U32} same :
          {source_block == Grid.block(grid,at,extent) : U32}},when}

# What a reading reads of one source axis, and what it keeps for the later
# axes.
def read_axis(read: Tasks.Source,+extent: U32,+block: U32,+rest: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis_extent: U32,+more: List<&2,U32>,+at: U32,
  +from: ReadFrom(read,Con{axis_extent,more},at,grid),
  +later: Held.Both<Components.ReadsAxis(rest,grid,at,axis_extent),ReadsFrom(rest,more,(at + 1 : U32),grid)>) ->
  Held.Both<Components.ReadsAxis(Con{Tasks.Reading{extent,block,read},rest},grid,at,axis_extent),ReadsFrom(Con{Tasks.Reading{extent,block,read},rest},more,(at + 1 : U32),grid)>:
  match read:
    case Tasks.Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      Held.Both{+same,Held.Both{+when,+extent_rest}} = from
      Held.Both{+reads,+froms} = later
      Held.Both{Held.Both{hit_facts(U32.is_eq(axis,at),axis,at,source_block,bound,grid,axis_extent,{==},same,when),reads},
        Held.Both{Held.Both{same,extent_rest},froms}}
    case Tasks.Unread{}:
      Held.Both{+reads,+froms} = later
      Held.Both{reads,Held.Both{Unit{},froms}}
    case Tasks.Crossing{}:
      Held.Both{+reads,+froms} = later
      Held.Both{reads,Held.Both{Unit{},froms}}

def reads_axis(row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis_extent: U32,+more: List<&2,U32>,+at: U32,+from: ReadsFrom(row,Con{axis_extent,more},at,grid)) ->
  Held.Both<Components.ReadsAxis(row,grid,at,axis_extent),ReadsFrom(row,more,(at + 1 : U32),grid)>:
  match row:
    case Nil{}: Held.Both{Unit{},Unit{}}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+head,+tail} = from
      read_axis(read,extent,block,rest,grid,axis_extent,more,at,head,reads_axis(rest,grid,axis_extent,more,at,tail))

def axis_reads_of(+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis_extent: U32,+more: List<&2,U32>,+at: U32,
  +pair: Held.Both<Components.ReadsAxis(row,grid,at,axis_extent),ReadsFrom(row,more,(at + 1 : U32),grid)>) -> Components.ReadsAxis(row,grid,at,axis_extent):
  Held.Both{+reads,+froms} = pair
  reads

def later_reads_of(+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+axis_extent: U32,+more: List<&2,U32>,+at: U32,
  +pair: Held.Both<Components.ReadsAxis(row,grid,at,axis_extent),ReadsFrom(row,more,(at + 1 : U32),grid)>) -> ReadsFrom(row,more,(at + 1 : U32),grid):
  Held.Both{+reads,+froms} = pair
  froms

# Every source axis is read with its block and its extent.
def reads_axes(extents: List<&2,U32>,+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+grid: Grid.Grid,+at: U32,+from: ReadsFrom(row,extents,at,grid),
  +blocks: ViewReads.BlocksIn(grid,extents,at,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Components.AxesOk(row,grid,extents,at,Machine.half_word(limits)):
  match extents:
    case Nil{}: Unit{}
    case Con{+axis_extent,+more}:
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = blocks
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(axis_extent)
      +inner = ShapeLaws.elements(more)
      +whole = Nat.mul(size,inner)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +block = Grid.block(grid,at,axis_extent)
      +pair = reads_axis(row,grid,axis_extent,more,at,from)
      Held.Both{Held.Both{axis_reads_of(row,grid,axis_extent,more,at,pair),
          pair_half(limits,size,U32.to_nat(block),size_room,block_within(limits,block,block_positive,block_double))},
        reads_axes(more,row,limits,grid,(at + 1 : U32),later_reads_of(row,grid,axis_extent,more,at,pair),blocks_rest,
          Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive)}

# ---- The classes of the readings of a row ----

def shared_none(list: List<&2,U32>) -> {Tasks.shared_starts(list,[]) == list : List<&2,U32>}:
  match list:
    case Nil{}: {==}
    case Con{+start,+rest}: Equal.cong(List<&2,U32>,List<&2,U32>,later => start <> later,Tasks.shared_starts(rest,[]),rest,shared_none(rest))

def runs_after(rest: List<&2,U32>,+start: U32,+blocks: U32) ->
  {List.length(&2,Tasks.Class,Tasks.runs(Con{start,rest},blocks)) == 1n+List.length(&2,U32,rest) : Nat}:
  match rest:
    case Nil{}: {==}
    case Con{+next,+more}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Tasks.Class,Tasks.runs(Con{next,more},blocks)),1n+List.length(&2,U32,more),runs_after(more,next,blocks))

def runs_length(starts: List<&2,U32>,+blocks: U32) -> {List.length(&2,Tasks.Class,Tasks.runs(starts,blocks)) == List.length(&2,U32,starts) : Nat}:
  match starts:
    case Nil{}: {==}
    case Con{+start,+rest}: runs_after(rest,start,blocks)

# A rising list of starts from low below limit holds at most the room
# between them.
def rising_count(list: List<&2,U32>,+low: Nat,+limit: Nat,+rising: Components.Rising(list,low,limit),+bound: {Nat.is_le(low,limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(low,List.length(&2,U32,list)),limit) == True{} : Bool}:
  match list:
    case Nil{}:
      %Equal.sym(Nat,Nat.add(low,0n),low,Addition.zero_right(low)) : {Nat.is_le(_,limit) == True{} : Bool}
      bound
    case Con{+first,+rest}:
      Held.Both{+above,Held.Both{+below,+later}} = rising
      +length = List.length(&2,U32,rest)
      +next = rising_count(rest,1n+U32.to_nat(first),limit,later,Order.strict_follow(U32.to_nat(first),limit,below))
      %Equal.sym(Nat,Nat.add(low,1n+length),1n+Nat.add(low,length),Addition.successor_right(low,length)) : {Nat.is_le(_,limit) == True{} : Bool}
      Order.transitive(1n+Nat.add(low,length),1n+Nat.add(U32.to_nat(first),length),limit,Order.add_le(low,U32.to_nat(first),length,above),next)

# Starts below a positive limit hold at least one.
def head_nonempty(list: List<&2,U32>,+limit: Nat,+head: Components.HeadAt(list,limit),+positive: {Nat.is_lt(0n,limit) == True{} : Bool}) ->
  {Nat.is_lt(0n,List.length(&2,U32,list)) == True{} : Bool}:
  match list:
    case Nil{}:
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true({%head : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,0n) == True{} : Bool}}))
    case Con{+first,+rest}: {==}

def ceiling_positive(+value: Nat,+divisor: Nat,+value_positive: {Nat.is_lt(0n,value) == True{} : Bool},+divisor_positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(0n,NatCeiling.ceiling(value,divisor)) == True{} : Bool}:
  RowMajor.left_factor_positive(NatCeiling.ceiling(value,divisor),divisor,
    Order.lt_le(0n,value,Nat.mul(NatCeiling.ceiling(value,divisor),divisor),value_positive,NatCeiling.covers(value,divisor,divisor_positive)))

# The classes of a reading alone number at least one and at most its
# extent.
def reading_classes(+extent: U32,+block: U32,+read: Tasks.Source,+limits: Machine.Limits,
  +room: Components.ReadingRoom(extent,block,read,Machine.half_word(limits)),+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +starts_ok: Components.Starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,read})),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)))) ->
  Held.Both<{Nat.is_lt(0n,List.length(&2,Tasks.Class,Tasks.axis_classes([Tasks.Reading{extent,block,read}]))) == True{} : Bool},
    {Nat.is_le(List.length(&2,Tasks.Class,Tasks.axis_classes([Tasks.Reading{extent,block,read}])),U32.to_nat(extent)) == True{} : Bool}>:
  Held.Both{+axis_facts,+read_room} = room
  Held.Both{+blocks_value,Held.Both{+blocks_room,Held.Both{+block_positive,+sum_room}}} = axis_facts
  Held.Both{+head,+rising} = starts_ok
  +starts = Tasks.class_starts(Tasks.classes(Tasks.Reading{extent,block,read}))
  +limit = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +classes = Tasks.axis_classes([Tasks.Reading{extent,block,read}])
  +same = Equal.trans(Nat,List.length(&2,Tasks.Class,classes),List.length(&2,U32,Tasks.shared_starts(starts,[])),List.length(&2,U32,starts),
    runs_length(Tasks.shared_starts(starts,[]),Tasks.target_blocks(extent,block)),
    Equal.cong(List<&2,U32>,Nat,list => List.length(&2,U32,list),Tasks.shared_starts(starts,[]),starts,shared_none(starts)))
  +back = Equal.sym(Nat,List.length(&2,Tasks.Class,classes),List.length(&2,U32,starts),same)
  Held.Both{{%back : {Nat.is_lt(0n,_) == True{} : Bool}
      head_nonempty(starts,limit,head,ceiling_positive(U32.to_nat(extent),U32.to_nat(block),extent_positive,block_positive)) :
      {Nat.is_lt(0n,List.length(&2,Tasks.Class,classes)) == True{} : Bool}},
    {%back : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
      Order.transitive(List.length(&2,U32,starts),limit,U32.to_nat(extent),rising_count(starts,0n,limit,rising,Order.zero_le(limit)),
        NatCeiling.bounded(U32.to_nat(extent),U32.to_nat(block),block_positive)) :
      {Nat.is_le(List.length(&2,Tasks.Class,classes),U32.to_nat(extent)) == True{} : Bool}}}

# The cells of the target axes of a row.
def RowCells(row: List<&2,Tasks.Reading>) -> Nat:
  match row:
    case Nil{}: 1n
    case Con{Tasks.Reading{+extent,block,read},rest}: Nat.mul(U32.to_nat(extent),RowCells(rest))

# The class lists of a row have room, count no more than its cells and at
# least one.
def ClassesOk(+lists: List<&2,List<&2,Tasks.Class>>,+cells: Nat,+cap: U32) -> Data:
  Held.Both<Components.ListsRoom(lists,cap),
    Held.Both<{Nat.is_le(Components.count_nat(lists),cells) == True{} : Bool},{Nat.is_lt(0n,Components.count_nat(lists)) == True{} : Bool}>>

def classes_cons(+list: List<&2,Tasks.Class>,+lists: List<&2,List<&2,Tasks.Class>>,+size: Nat,+cells: Nat,+limits: Machine.Limits,
  +own: Held.Both<{Nat.is_lt(0n,List.length(&2,Tasks.Class,list)) == True{} : Bool},{Nat.is_le(List.length(&2,Tasks.Class,list),size) == True{} : Bool}>,
  +rest: ClassesOk(lists,cells,Machine.half_word(limits)),+size_room: {Nat.is_le(size,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ClassesOk(Con{list,lists},Nat.mul(size,cells),Machine.half_word(limits)):
  Held.Both{+positive,+within} = own
  Held.Both{+lists_room,Held.Both{+count_le,+count_positive}} = rest
  +length = List.length(&2,Tasks.Class,list)
  +count = Components.count_nat(lists)
  Held.Both{Held.Both{LimitBounds.within_half(limits,length,Order.transitive(length,size,U32.to_nat(Machine.capacity(limits)),within,size_room)),lists_room},
    Held.Both{Order.transitive(Nat.mul(length,count),Nat.mul(size,count),Nat.mul(size,cells),RowMajor.mul_le(length,size,count,within),
        DivisionOrder.multiply_left_le(size,count,cells,count_le)),
      Order.product_positive(length,count,positive,count_positive)}}

def row_classes(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid),
  +cells_room: {Nat.is_le(RowCells(row),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,RowCells(row)) == True{} : Bool}) ->
  ClassesOk(Components.raw_lists(List.length(&2,Tasks.Reading,row),[row]),RowCells(row),Machine.half_word(limits)):
  match row:
    case Nil{}: Held.Both{Unit{},Held.Both{{==},{==}}}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = ok
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(extent)
      +inner = RowCells(rest)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,cells_positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,cells_positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),cells_room)
      classes_cons(Tasks.axis_classes([Tasks.Reading{extent,block,read}]),Components.raw_lists(List.length(&2,Tasks.Reading,rest),[rest]),size,inner,limits,
        reading_classes(extent,block,read,limits,room,size_positive,Components.reading_starts(extent,block,read,Machine.half_word(limits),LimitBounds.half_word_fits(limits),room)),
        row_classes(rest,limits,source,grid,later,Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),cells_room),inner_positive),size_room)

# ---- The readings of a view by axis ----

# The target cells of the readings of a view are the cells of its extents.
def view_cells(+source: List<&2,U32>,+sgrid: Grid.Grid,+tgrid: Grid.Grid,+fixed: List<&2,U32>,axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+at: U32) ->
  {RowCells(Tasks.readings(source,sgrid,tgrid,fixed,axes,flags,at)) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +block = Grid.block(tgrid,at,extent)
      +flag = Views.flag(flags)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(source,fixed,stride,0))
      +later = Tasks.readings(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32))
      %Equal.sym(Tasks.Reading,Tasks.source_reading(sgrid,extent,block,flag,found),
          Tasks.Reading{extent,block,ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found))},ViewReads.reading_parts(found,sgrid,extent,block,flag)) :
        {RowCells(Con{_,later}) == ShapeLaws.elements(Views.axis_extents(Con{Views.Axis{extent,stride},rest})) : Nat}
      Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),RowCells(later),ShapeLaws.elements(Views.axis_extents(rest)),
        view_cells(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32)))

# Every target axis of a view is limited by its blocks in the target grid.
def view_limits(+source: List<&2,U32>,+sgrid: Grid.Grid,+tgrid: Grid.Grid,+fixed: List<&2,U32>,axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+at: U32) ->
  Components.LimitsAt([Tasks.readings(source,sgrid,tgrid,fixed,axes,flags,at)],List.length(&2,Tasks.Reading,Tasks.readings(source,sgrid,tgrid,fixed,axes,flags,at)),
    Views.axis_extents(axes),tgrid,at):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +block = Grid.block(tgrid,at,extent)
      +flag = Views.flag(flags)
      +found = Bool.pick(Tasks.Stepped,Tasks.unread(extent,stride),Tasks.Unstepped{},Tasks.stepped_axis(source,fixed,stride,0))
      +later = Tasks.readings(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32))
      %Equal.sym(Tasks.Reading,Tasks.source_reading(sgrid,extent,block,flag,found),
          Tasks.Reading{extent,block,ViewReads.read_of(Tasks.source_reading(sgrid,extent,block,flag,found))},ViewReads.reading_parts(found,sgrid,extent,block,flag)) :
        Components.LimitsAt([Con{_,later}],List.length(&2,Tasks.Reading,Con{_,later}),Con{extent,Views.axis_extents(rest)},tgrid,at)
      Held.Both{{==},view_limits(source,sgrid,tgrid,fixed,rest,Views.drop_flags(1n,flags),(at + 1 : U32))}

# ---- What a view the runtime accepts gives its gather ----

def one_half(+limits: Machine.Limits) -> {Nat.is_le(1n,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(1n,16n,U32.to_nat(Machine.half_word(limits)),{==},Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits)))

# The components of a row of readings alone: one row, its columns, and its
# class lists with room at half a word.
def row_components(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,+ok: ReadsOk(row,limits,source,grid),
  +classes: ClassesOk(Components.raw_lists(List.length(&2,Tasks.Reading,row),[row]),RowCells(row),Machine.half_word(limits)),
  +cells_room: {Nat.is_le(RowCells(row),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool}>:
  Held.Both{+lists_room,Held.Both{+count_le,+count_positive}} = classes
  +count = Components.count_nat(Components.raw_lists(List.length(&2,Tasks.Reading,row),[row]))
  Held.Both{Held.Both{Held.Both{{==},Unit{}},Held.Both{reads_columns(row,limits,source,grid,ok),Held.Both{lists_room,
      Held.Both{LimitBounds.double_bound(limits,count,Order.transitive(count,RowCells(row),U32.to_nat(Machine.capacity(limits)),count_le,cells_room)),one_half(limits)}}}},
    count_positive}

def components_part(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,
  +both: Held.Both<Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool}>) ->
  Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)):
  Held.Both{+ok,+positive} = both
  ok

def positive_part(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,
  +both: Held.Both<Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool}>) ->
  {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool}:
  Held.Both{+ok,+positive} = both
  positive

def row_when(enabled: Bool,+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+shape: List<&2,U32>,+ok: Components.RowOk(limits,row,grid,shape)) ->
  Logic.When(enabled,Components.RowOk(limits,row,grid,shape)):
  match enabled:
    case False{}: Unit{}
    case True{}: ok

# A view whose source and target fit the machine and whose last address lies
# below the count of its source gives its gather every fact it needs.
def view_facts(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+target: U32,
  +source_ok: Components.ShapeOk(limits,source),+target_ok: Components.ShapeOk(limits,Views.axis_extents(axes)),
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool},
  +count_le: {Nat.is_le(U32.to_nat(Tensors.count_or_zero(source)),ShapeLaws.elements(source)) == True{} : Bool}) ->
  ViewGather.ViewFacts(limits,source,axes,offset,flags,target):
  Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = source_ok
  Held.Both{+room,Held.Both{+positive,+rank_room}} = target_ok
  +cap = U32.to_nat(Machine.capacity(limits))
  +half = Machine.half_word(limits)
  +extents = Views.axis_extents(axes)
  +sgrid = Grid.grid(source,Grid.piece_target())
  +tgrid = Grid.grid(extents,target)
  +fixed = Grid.coordinates(source,offset)
  +row = Tasks.readings(source,sgrid,tgrid,fixed,axes,flags,0)
  +length = Tensors.count_or_zero(source)
  +elements = ShapeLaws.elements(source)
  +base = U32.to_nat(offset)
  +span = ViewBounds.last_span(axes)
  +shape_ok = {Held.Both{source_room,Held.Both{source_positive,source_rank}} : Components.ShapeOk(limits,source)}
  +view_ok = {Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,extents)}
  +last = {%ViewBounds.last_location(axes,base) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} ViewBounds.last_address(axes,offset,length,fits) :
    {Nat.is_lt(Nat.add(base,span),U32.to_nat(length)) == True{} : Bool}}
  +span_below = Order.lt_le(span,U32.to_nat(length),elements,Order.le_lt_trans(span,Nat.add(base,span),U32.to_nat(length),Order.right_le(base,span),last),count_le)
  +premises = {Held.Both{span_below,Held.Both{room,positive}} : AxesPremises(limits,source,axes)}
  +fixed_in = ViewReads.coordinates_in(source,offset,half,LimitBounds.half_word_fits(limits),LimitBounds.double_bound(limits,elements,source_room),source_positive)
  +blocks = grid_blocks(limits,source,Grid.piece_target(),shape_ok)
  +ok = view_reads(limits,source,sgrid,tgrid,fixed,axes,flags,0,fixed_in,blocks,grid_blocks(limits,extents,target,view_ok),shape_ok,premises)
  +cells_same = view_cells(source,sgrid,tgrid,fixed,axes,flags,0)
  +cells_room = {%Equal.sym(Nat,RowCells(row),ShapeLaws.elements(extents),cells_same) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(RowCells(row),cap) == True{} : Bool}}
  +cells_positive = {%Equal.sym(Nat,RowCells(row),ShapeLaws.elements(extents),cells_same) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,RowCells(row)) == True{} : Bool}}
  +components = row_components(row,limits,source,sgrid,ok,row_classes(row,limits,source,sgrid,ok,cells_room,cells_positive),cells_room)
  +row_ok = {Held.Both{reads_room(row,limits,source,sgrid,ok),Held.Both{reads_axes(source,row,limits,sgrid,0,reads_from(row,limits,source,sgrid,ok),blocks,source_room,source_positive),
      reads_below(row,limits,source,sgrid,ok)}} : Components.RowOk(limits,row,sgrid,source)}
  Held.Both{shape_ok,Held.Both{view_ok,Held.Both{fits,Held.Both{count_le,
    Held.Both{axes_fit(limits,source,fixed,sgrid,axes,fixed_in,blocks,shape_ok,premises),Held.Both{blocks,
    Held.Both{view_strides(limits,axes,offset,length,fits,Order.transitive(U32.to_nat(length),elements,cap,count_le,source_room),view_ok),
    Held.Both{components_part(row,limits,components),Held.Both{positive_part(row,limits,components),
    Held.Both{view_limits(source,sgrid,tgrid,fixed,axes,flags,0),
      row_when(Tasks.separable([row]),limits,row,sgrid,source,row_ok)}}}}}}}}}}
