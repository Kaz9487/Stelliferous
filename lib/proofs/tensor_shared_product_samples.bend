# The read span of a shared rectangle is the corresponding output groups.
import Base
import ../kernel_gemm.bend as Gemm
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ../tensor_point_products.bend as Products
import ../tensor_tiled_cells.bend as Cells
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./machine_limit_bounds.bend as Limits
import ./nat_extrema.bend as Extrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as Index
import ./tensor_point_product_samples.bend as Samples
import ./tensor_point_product_fragments.bend as Fragments
import ./tensor_point_product_tasks.bend as Shared
import ./tensor_point_products.bend as Point
import ./tensor_tiled_representation.bend as Representation

def start_group(+rows: U32,+first: U32) -> Nat:
  U32.to_nat((first / rows : U32))

def group_count(+rows: U32,+count: U32) -> Nat:
  U32.to_nat((count / rows : U32))

def start_word(+rows: U32,+inner: U32,+first: U32) -> U32:
  (first / rows * inner : U32)

def span_word(+rows: U32,+inner: U32,+count: U32) -> U32:
  (count / rows * inner : U32)

def weight_cells(+inner: U32,+channels: U32) -> Nat:
  Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))

type Span<-rows: U32,-inner: U32,-channels: U32,-first: U32,-count: U32> is Data:
  Span{groups: {Nat.is_le(Nat.add(start_group(rows,first),group_count(rows,count)),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
    start: {U32.to_nat(start_word(rows,inner,first)) == Nat.mul(start_group(rows,first),U32.to_nat(inner)) : Nat},
    length: {U32.to_nat(span_word(rows,inner,count)) == Nat.mul(group_count(rows,count),U32.to_nat(inner)) : Nat},
    within: {Nat.is_le(Nat.add(U32.to_nat(start_word(rows,inner,first)),U32.to_nat(span_word(rows,inner,count))),weight_cells(inner,channels)) == True{} : Bool}}

def weighted_span(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+first: U32,+count: U32,
  +room: {Nat.is_le(weight_cells(inner,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +interval: Shared.GroupInterval<first,count,rows,channels>) -> Span<rows,inner,channels,first,count>:
  Shared.GroupInterval{+lower,+length,+first_exact,first_aligned,+groups_exact,+aligned,+nonempty,+inside} = interval
  +start = start_group(rows,first)
  +counted = group_count(rows,count)
  +width = U32.to_nat(inner)
  +first_cell = Nat.mul(start,width)
  +length_cells = Nat.mul(counted,width)
  +whole = weight_cells(inner,channels)
  +total = Nat.add(first_cell,length_cells)
  +cap = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +end = Nat.add(start,counted)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +fits = {%Equal.sym(Nat,start,lower,first_exact) : {Nat.is_le(Nat.add(_,counted),groups) == True{} : Bool}
    %Equal.sym(Nat,counted,length,groups_exact) : {Nat.is_le(Nat.add(lower,_),groups) == True{} : Bool}
    inside : {Nat.is_le(end,groups) == True{} : Bool}}
  +bounded = {%Index.distrib(start,counted,width) : {Nat.is_le(_,whole) == True{} : Bool}
    Index.mul_le(end,groups,width,fits) : {Nat.is_le(total,whole) == True{} : Bool}}
  +total_room = Order.transitive(total,whole,U32.to_nat(cap),bounded,room)
  +base_exact = Arithmetic.product_exact(cap,cap_ok,(first / rows : U32),inner,start,width,{==},{==},
    Order.transitive(first_cell,total,U32.to_nat(cap),Order.le_add(first_cell,length_cells),total_room))
  +count_exact = Arithmetic.product_exact(cap,cap_ok,(count / rows : U32),inner,counted,width,{==},{==},
    Order.transitive(length_cells,total,U32.to_nat(cap),Order.right_le(first_cell,length_cells),total_room))
  Span{fits,base_exact,count_exact,
    %Equal.sym(Nat,U32.to_nat(start_word(rows,inner,first)),first_cell,base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(span_word(rows,inner,count))),whole) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(span_word(rows,inner,count)),length_cells,count_exact) : {Nat.is_le(Nat.add(first_cell,_),whole) == True{} : Bool}
    bounded}

def active_in_clip(+limits: Machine.Limits,+rows: U32,+channels: U32,+first: U32,+count: U32,+group: Nat,+lane: Nat,
  +clip: Shared.Clipped<limits,rows,channels,first,count>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Shared.clipped_extent(rows,channels,first,count)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +global_active: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.add(start_group(rows,first),group),8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Shared.clipped_extent(rows,channels,first,count))) == True{} : Bool}:
  Shared.Clipped{+exact,+room,+positive,+group_exact} = clip
  +lower = start_group(rows,first)
  +length = group_count(rows,count)
  +offset = Nat.mul(lower,8n)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +capacity = U32.to_nat(channels)
  +within = {%Order.commutative(Nat.mul(group,8n),lane) : {Nat.is_lt(_,Nat.mul(length,8n)) == True{} : Bool}
    Index.radix(group,length,lane,8n,%group_exact : {Nat.is_lt(group,_) == True{} : Bool} group_inside,lane_inside) :
    {Nat.is_lt(channel,Nat.mul(length,8n)) == True{} : Bool}}
  +flattened = {%Equal.sym(Nat,Nat.add(offset,channel),Nat.add(lane,Nat.mul(Nat.add(lower,group),8n)),Samples.offset_channel(lower,group,lane)) :
    {Nat.is_lt(_,capacity) == True{} : Bool}
    global_active : {Nat.is_lt(Nat.add(offset,channel),capacity) == True{} : Bool}}
  +from = Order.le_lt_trans(offset,Nat.add(offset,channel),capacity,Order.le_add(offset,channel),flattened)
  +before = {Equal.trans(Bool,Nat.is_lt(channel,Nat.sub(capacity,offset)),Nat.is_lt(Nat.add(offset,channel),capacity),True{},
    Order.below_difference(offset,channel,capacity,Bounds.strict_le(offset,capacity,from)),flattened) :
    {Nat.is_lt(channel,Nat.sub(capacity,offset)) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(Shared.clipped_extent(rows,channels,first,count)),Nat.min(Nat.mul(length,8n),Nat.sub(capacity,offset)),exact) :
    {Nat.is_lt(channel,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(channel,Nat.min(Nat.mul(length,8n),Nat.sub(capacity,offset))),
    Nat.is_lt(channel,Nat.mul(length,8n)) && Nat.is_lt(channel,Nat.sub(capacity,offset)),
    Extrema.below_minimum(channel,Nat.mul(length,8n),Nat.sub(capacity,offset))) : {_ == True{} : Bool}
  Boolean.and_true(Nat.is_lt(channel,Nat.mul(length,8n)),Nat.is_lt(channel,Nat.sub(capacity,offset)),within,before)

def output_geometry(+rows: U32,+channels: U32) -> Cells.Cells:
  Cells.Cells{rows,1,channels,Gemm.blocks(channels),Grid.Outermost{}}

def output_index(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,+word: U32,
  +aligned: {U32.to_nat(first) == Nat.mul(start_group(rows,first),U32.to_nat(rows)) : Nat},
  +local: Fragments.Coordinate<Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),word>) ->
  {U32.to_nat(word) == Representation.cell_index(output_geometry(rows,channels),
    Fragments.rectangle_position(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),word),
    Nat.add(start_group(rows,first),Fragments.rectangle_group(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),word))) : Nat}:
  Fragments.Coordinate{+position_inside,+group_inside,+exact} = local
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  +position = Fragments.rectangle_position(rectangle,word)
  +group = Fragments.rectangle_group(rectangle,word)
  +start = start_group(rows,first)
  +width = U32.to_nat(rows)
  +local_index = Nat.add(Nat.mul(group,width),position)
  +wanted = Nat.add(Nat.mul(Nat.add(start,group),width),position)
  Equal.trans(Nat,U32.to_nat(word),Nat.add(U32.to_nat(first),local_index),wanted,exact,
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(start,width),aligned) :
      {Nat.add(_,local_index) == wanted : Nat}
    Samples.joined_rows(start,group,width,position))
