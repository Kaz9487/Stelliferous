# A written fragment exposes the position and group of each staged pair.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../kernel_window_pipeline.bend as Pipeline
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ./binary_radix.bend as Radix
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_pipeline_converted.bend as Converted
import ./kernel_gemm_read_into.bend as ReadInto
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./ordered_ranges.bend as Ranges
import ./row_major_index.bend as Index
import ./storage_shape.bend as StorageShape
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./sequence_membership.bend as Members
import ./tensor_allocation_depth.bend as Depth
import ./tensor_clipped_pieces.bend as Clipped
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_model.bend as Model
import ./tensor_placed.bend as Placement
import ./tensor_placed_parts.bend as Placed
import ./tensor_point_product_tasks.bend as TaskProofs
import ./tensor_point_product_values.bend as Values
import ./tensor_point_products.bend as Point
import ./word_power_of_two.bend as Power
import ./traversal_update_readback.bend as Updates

def ordered_index(group_major: Bool,+positions: Nat,+groups: Nat,+position: Nat,+group: Nat) -> Nat:
  match group_major:
    case True{}: Values.slot(group,positions,position)
    case False{}: Values.slot(position,groups,group)

def slice_order(group_major: Bool,+limits: Machine.Limits,+positions: U32,+groups: U32,+first: U32,+index: U32,+position: Nat,+group: Nat,
  +positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_positive: {Nat.is_lt(0n,U32.to_nat(positions)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,U32.to_nat(positions)) == True{} : Bool},
  +cell_room: {Nat.is_le(Values.slot(position,U32.to_nat(groups),group),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +exact: {U32.to_nat((first + index : U32)) == ordered_index(group_major,U32.to_nat(positions),U32.to_nat(groups),position,group) : Nat}) ->
  {Gemm.staged_slice_index(Gemm.StagedSlice{groups,positions,first,group_major},index) == U32.from_nat(Values.slot(position,U32.to_nat(groups),group)) : U32}:
  match group_major:
    case False{}:
      Equal.sym(U32,U32.from_nat(Values.slot(position,U32.to_nat(groups),group)),(first + index : U32),Arithmetic.from_nat_exact((first + index : U32),Values.slot(position,U32.to_nat(groups),group),exact))
    case True{}:
      +cell = (first + index : U32)
      +capacity = Machine.capacity(limits)
      +half = Machine.half_word(limits)
      +half_ok = Limits.half_word_fits(limits)
      +double = Limits.double_bound(limits,U32.to_nat(positions),positions_room)
      +row_exact = Radix.remainder_word(cell,positions,group,position,U32.to_nat(positions),half,half_ok,exact,{==},positions_positive,position_inside,double)
      +group_exact = Radix.quotient_word(cell,positions,group,position,U32.to_nat(positions),half,half_ok,exact,{==},positions_positive,position_inside,double)
      +row_word = (cell % positions : U32)
      +group_word = (cell / positions : U32)
      +product = Nat.mul(position,U32.to_nat(groups))
      +target = Values.slot(position,U32.to_nat(groups),group)
      +product_room = Order.transitive(product,target,U32.to_nat(capacity),Order.le_add(product,group),cell_room)
      +product_exact = Arithmetic.product_exact(capacity,Limits.capacity_fits(limits),row_word,groups,position,U32.to_nat(groups),row_exact,{==},product_room)
      +sum = (row_word * groups + group_word : U32)
      +sum_exact = Arithmetic.sum_exact(capacity,Limits.capacity_fits(limits),(row_word * groups : U32),group_word,product,group,product_exact,group_exact,cell_room)
      Equal.sym(U32,U32.from_nat(target),sum,Arithmetic.from_nat_exact(sum,target,sum_exact))

def staged_group(+limits: Machine.Limits,+channels: U32,+position: Nat,+group: Nat,+staged: Tree.Tree<Tile.Tile16>,
  +groups_positive: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fewer: {Nat.is_le(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +cell_room: {Nat.is_le(Values.slot(position,U32.to_nat(Gemm.blocks(channels)),group),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {PointBuffer.staged_tile(~Unit,~Gemm.unmapped_tile,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},
      U32.from_nat(Values.slot(position,U32.to_nat(Gemm.blocks(channels)),group))) ==
    Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Values.slot(position,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(group,2n)))),
      U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1)) : Tile.Tile8}:
  +groups = U32.to_nat(Gemm.blocks(channels))
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +index = Values.slot(position,groups,group)
  +half = Machine.half_word(limits)
  %Radix.quotient(position,group,groups,groups_positive,group_inside) :
    {PointBuffer.staged_tile(~Unit,~Gemm.unmapped_tile,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},U32.from_nat(index)) ==
      Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Values.slot(_,pairs,Nat.div(group,2n)))),U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1)) : Tile.Tile8}
  %Radix.remainder(position,group,groups,groups_positive,group_inside) :
    {PointBuffer.staged_tile(~Unit,~Gemm.unmapped_tile,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},U32.from_nat(index)) ==
      Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Values.slot(Nat.div(index,groups),pairs,Nat.div(_,2n)))),U32.is_eq(U32.from_nat(Nat.mod(_,2n)),1)) : Tile.Tile8}
  Converted.staged_cell(~Unit,~Gemm.unmapped_tile,Unit{},channels,staged,index,half,Limits.half_word_fits(limits),groups,{==},pairs,{==},groups_positive,fewer,
    Limits.double_bound(limits,groups,groups_room),Limits.within_half(limits,4n,Values.pair_room(limits)),Limits.within_half(limits,index,cell_room))

def fresh_slice(+slice: Gemm.StagedSlice,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+index: Nat,
  +bounded: {U32.is_le(count,1073741824) == True{} : Bool},+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,PointBuffer.slice_loop(slice,count,staged,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),U32.from_nat(index)) ==
    PointBuffer.slice_sample(staged,slice,U32.from_nat(index)) : Tile.Tile8}:
  +depth = Tensors.depth_for(count)
  +allocated = Depth.bounded(count,bounded)
  +output = Tree.fresh(Tile.Tile8,depth,Products.zero())
  +balanced = StorageShape.fresh(Tile.Tile8,depth,Products.zero())
  +limit = Order.successor(depth,30n,Depth.allocation_depth(count,allocated))
  +capacity = Tree.capacity(Tile.Tile8,output)
  +exact = StorageShape.capacity(Tile.Tile8,depth,output,balanced)
  PointBuffer.slice_at(slice,count,staged,output,depth,balanced,limit,
    %Equal.sym(U32,capacity,U32.shln(1,depth),exact) : {U32.is_le(_,2147483648) == True{} : Bool} Power.native_power_bound(depth,limit),
    %Equal.sym(U32,capacity,U32.shln(1,depth),exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,allocated),index,inside)

def fragment_cell(+limits: Machine.Limits,+group_major: Bool,+channels: U32,+positions: U32,+first: U32,+count: U32,
  +index: Nat,+position: Nat,+group: Nat,+staged: Tree.Tree<Tile.Tile16>,
  +bounded: {U32.is_le(count,1073741824) == True{} : Bool},+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},
  +positions_room: {Nat.is_le(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positions_positive: {Nat.is_lt(0n,U32.to_nat(positions)) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,U32.to_nat(positions)) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +fewer: {Nat.is_le(U32.to_nat(Gemm.pairs_of(channels)),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +cell_room: {Nat.is_le(Values.slot(position,U32.to_nat(Gemm.blocks(channels)),group),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +exact: {U32.to_nat((first + U32.from_nat(index) : U32)) == ordered_index(group_major,U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels)),position,group) : Nat}) ->
  {Tree.read(Tile.Tile8,PointBuffer.slice_loop(Gemm.StagedSlice{Gemm.blocks(channels),positions,first,group_major},count,staged,
      Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),U32.from_nat(index)) ==
    Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Values.slot(position,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(group,2n)))),
      U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1)) : Tile.Tile8}:
  +groups = Gemm.blocks(channels)
  +slice = {Gemm.StagedSlice{groups,positions,first,group_major} : Gemm.StagedSlice}
  +plan = {Gemm.StagedPlan{groups,Gemm.pairs_of(channels),Unit{}} : Gemm.StagedPlan<Unit>}
  +word = U32.from_nat(index)
  +positioned = Gemm.staged_slice_index(slice,word)
  +wanted = Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Values.slot(position,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(group,2n)))),
    U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1))
  +mapped = PointBuffer.slice_sample(staged,slice,word)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,PointBuffer.slice_loop(slice,count,staged,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),word),mapped,wanted,
    fresh_slice(slice,count,staged,index,bounded,inside),
    %Equal.sym(U32,positioned,U32.from_nat(Values.slot(position,U32.to_nat(groups),group)),
      slice_order(group_major,limits,positions,groups,first,word,position,group,positions_room,positions_positive,position_inside,cell_room,exact)) :
      {PointBuffer.staged_tile(~Unit,~Gemm.unmapped_tile,staged,plan,_) == wanted : Tile.Tile8}
    staged_group(limits,channels,position,group,staged,groups_positive,groups_room,fewer,group_inside,cell_room))

def slice_ready(+slice: Gemm.StagedSlice,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+bounded: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.Content{PointBuffer.slice_loop(slice,count,staged,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),U32.to_nat(count)}>:
  +depth = Tensors.depth_for(count)
  +allocated = Depth.bounded(count,bounded)
  +output = Tree.fresh(Tile.Tile8,depth,Products.zero())
  +new_output = PointBuffer.slice_loop(slice,count,staged,output)
  +balance = StorageShape.fresh(Tile.Tile8,depth,Products.zero())
  +kept = Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedSlice,~PointBuffer.slice_sample,
    ~Gemm.replaced(~Tile.Tile8),~Gemm.staged_slice_destination,U32.to_nat(count),0,staged,slice,output,depth,balance)
  +cap = Tree.capacity(Tile.Tile8,new_output)
  Buffer.ContentReady{depth,kept,Depth.allocation_depth(count,allocated),
    %Equal.sym(U32,cap,U32.shln(1,depth),StorageShape.capacity(Tile.Tile8,depth,new_output,kept)) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,allocated)}

# Every fragment of a laid range is inside that range.
def Enclosed(+lower: Nat,+upper: Nat,ranges: Pieces.Ranges) -> Data:
  match ranges:
    case Pieces.Range{+first,+count}: Held.Both<{Nat.is_le(lower,U32.to_nat(first)) == True{} : Bool},
      {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),upper) == True{} : Bool}>
    case Pieces.Split{+left,+right}: Held.Both<Enclosed(lower,upper,left),Enclosed(lower,upper,right)>

def enclosed(+target: Pieces.Ranges,ranges: Pieces.Ranges,+position: Nat,+lower: Nat,+upper: Nat,
  +laid: Placed.LaidParts(target,ranges,position),+low: {Nat.is_le(lower,position) == True{} : Bool},
  +high: {Nat.is_le(Placed.ranges_end(ranges,position),upper) == True{} : Bool}) -> Enclosed(lower,upper,ranges):
  match ranges:
    case Pieces.Range{+first,+count}:
      Gemm.Both{+same,rest} = laid
      Held.Both{%Equal.sym(Nat,U32.to_nat(first),position,same) : {Nat.is_le(lower,_) == True{} : Bool} low,
        %Equal.sym(Nat,U32.to_nat(first),position,same) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),upper) == True{} : Bool} high}
    case Pieces.Split{+left,+right}:
      Gemm.Both{+left_laid,+right_laid} = laid
      +middle = Placed.ranges_end(left,position)
      +last = Placed.ranges_end(right,middle)
      Held.Both{enclosed(target,left,position,lower,upper,left_laid,low,
        Order.transitive(middle,last,upper,Placed.ranges_end_after(right,middle),high)),
        enclosed(target,right,middle,lower,upper,right_laid,Order.transitive(lower,position,middle,low,Placed.ranges_end_after(left,position)),high)}

def position_of(group_major: Bool,+positions: Nat,+groups: Nat,+index: Nat) -> Nat:
  match group_major:
    case True{}: Nat.mod(index,positions)
    case False{}: Nat.div(index,groups)

def group_of(group_major: Bool,+positions: Nat,+groups: Nat,+index: Nat) -> Nat:
  match group_major:
    case True{}: Nat.div(index,positions)
    case False{}: Nat.mod(index,groups)

type Located<-group_major: Bool,-positions: Nat,-groups: Nat,-index: Nat> is Data:
  Located{position: {Nat.is_lt(position_of(group_major,positions,groups,index),positions) == True{} : Bool},
    group: {Nat.is_lt(group_of(group_major,positions,groups,index),groups) == True{} : Bool},
    exact: {index == ordered_index(group_major,positions,groups,position_of(group_major,positions,groups,index),group_of(group_major,positions,groups,index)) : Nat}}

def located(group_major: Bool,+positions: Nat,+groups: Nat,+index: Nat,
  +positions_positive: {Nat.is_lt(0n,positions) == True{} : Bool},+groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +inside: {Nat.is_lt(index,Nat.mul(positions,groups)) == True{} : Bool}) -> Located<group_major,positions,groups,index>:
  match group_major:
    case False{}:
      Located{DivisionOrder.quotient_below(index,positions,groups,groups_positive,inside),
        DivisionOrder.remainder_small(index,groups,groups_positive),DivisionOrder.quotient_exact(index,groups,groups_positive)}
    case True{}:
      Located{DivisionOrder.remainder_small(index,positions,positions_positive),
        DivisionOrder.quotient_below(index,groups,positions,positions_positive,
          %Index.multiply_commutative(positions,groups) : {Nat.is_lt(index,_) == True{} : Bool} inside),
        DivisionOrder.quotient_exact(index,positions,positions_positive)}

type RectangleRead is Data:
  RectangleRead{rectangle: Products.Rectangle,stage: Tree.Tree<Tile.Tile16>}

def staged_at(reader: RectangleRead,+position: Nat,+group: Nat) -> Tile.Tile8:
  RectangleRead{Products.Rectangle{first,positions,inner,+channels,group_major,grid},+stage} = reader
  Gemm.staged_half(Tree.read(Tile.Tile16,stage,U32.from_nat(Values.slot(position,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(group,2n)))),
    U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1))

def cell_value(reader: RectangleRead,index: U32) -> Tile.Tile8:
  RectangleRead{Products.Rectangle{+first,+positions,inner,+channels,+group_major,grid},+stage} = reader
  +groups = U32.to_nat(Gemm.blocks(channels))
  +local = Nat.sub(U32.to_nat(index),U32.to_nat(first))
  +position = position_of(group_major,U32.to_nat(positions),groups,local)
  +group = group_of(group_major,U32.to_nat(positions),groups,local)
  staged_at(reader,position,group)

def rectangle_start(+rectangle: Products.Rectangle) -> U32:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  first

def rectangle_cells(+rectangle: Products.Rectangle) -> Nat:
  Nat.mul(U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))))

def group_major_of(+rectangle: Products.Rectangle) -> Bool:
  Products.Rectangle{first,positions,inner,channels,group_major,grid} = rectangle
  group_major

def rectangle_offset(+rectangle: Products.Rectangle,+index: U32) -> Nat:
  Nat.sub(U32.to_nat(index),U32.to_nat(rectangle_start(rectangle)))

def rectangle_position(+rectangle: Products.Rectangle,+index: U32) -> Nat:
  Products.Rectangle{first,+positions,inner,+channels,+group_major,grid} = rectangle
  position_of(group_major,U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels)),rectangle_offset(rectangle,index))

def rectangle_group(+rectangle: Products.Rectangle,+index: U32) -> Nat:
  Products.Rectangle{first,+positions,inner,+channels,+group_major,grid} = rectangle
  group_of(group_major,U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels)),rectangle_offset(rectangle,index))

type RectangleReadRoom<-limits: Machine.Limits,-rectangle: Products.Rectangle> is Data:
  RectangleReadRoom{positions_positive: {Nat.is_lt(0n,U32.to_nat(Values.rectangle_positions(rectangle))) == True{} : Bool},
    positions_room: {Nat.is_le(U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    groups_positive: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
    groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    fewer: {Nat.is_le(U32.to_nat(Gemm.pairs_of(Values.rectangle_channels(rectangle))),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
    inside: {Nat.is_le(Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

type Coordinate<-rectangle: Products.Rectangle,-index: U32> is Data:
  Coordinate{position_inside: {Nat.is_lt(rectangle_position(rectangle,index),U32.to_nat(Values.rectangle_positions(rectangle))) == True{} : Bool},
    group_inside: {Nat.is_lt(rectangle_group(rectangle,index),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
    exact: {U32.to_nat(index) == Nat.add(U32.to_nat(rectangle_start(rectangle)),ordered_index(group_major_of(rectangle),U32.to_nat(Values.rectangle_positions(rectangle)),
      U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))),rectangle_position(rectangle,index),rectangle_group(rectangle,index))) : Nat}}

def sliced(+rectangle: Products.Rectangle,+first: U32) -> Gemm.StagedSlice:
  Products.Rectangle{+start,+positions,inner,+channels,+group_major,grid} = rectangle
  Gemm.StagedSlice{Gemm.blocks(channels),positions,(first - start : U32),group_major}

def fragment_tree(+rectangle: Products.Rectangle,+first: U32,+count: U32,+stage: Tree.Tree<Tile.Tile16>) -> Tree.Tree<Tile.Tile8>:
  PointBuffer.slice_loop(sliced(rectangle,first),count,stage,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero()))

def at_position(+rectangle: Products.Rectangle,+index: U32,
  +starts: {Nat.is_le(U32.to_nat(rectangle_start(rectangle)),U32.to_nat(index)) == True{} : Bool},
  +location: Located<group_major_of(rectangle),U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))),rectangle_offset(rectangle,index)>) ->
  Coordinate<rectangle,index>:
  Products.Rectangle{+first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  Located{+position_inside,+group_inside,+recombined} = location
  +start = U32.to_nat(first)
  +cell = U32.to_nat(index)
  +offset = Nat.sub(cell,start)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +rows = U32.to_nat(positions)
  +position = position_of(group_major,rows,groups,offset)
  +group = group_of(group_major,rows,groups,offset)
  Coordinate{position_inside,group_inside,
    Equal.trans(Nat,cell,Nat.add(start,offset),Nat.add(start,ordered_index(group_major,rows,groups,position,group)),
      Equal.sym(Nat,Nat.add(start,offset),cell,Order.restore_shift(start,cell,starts)),
      Equal.cong(Nat,Nat,relative => Nat.add(start,relative),offset,ordered_index(group_major,rows,groups,position,group),recombined))}

def rectangle_coordinate(+limits: Machine.Limits,+rectangle: Products.Rectangle,+index: U32,+space: RectangleReadRoom<limits,rectangle>,
  +starts: {Nat.is_le(U32.to_nat(rectangle_start(rectangle)),U32.to_nat(index)) == True{} : Bool},
  +ends: {Nat.is_lt(U32.to_nat(index),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))) == True{} : Bool}) -> Coordinate<rectangle,index>:
  Products.Rectangle{+first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  RectangleReadRoom{+positions_positive,+positions_room,+groups_positive,+groups_room,+fewer,+inside} = space
  at_position(rectangle,index,starts,
    located(group_major,U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels)),rectangle_offset(rectangle,index),positions_positive,groups_positive,
      Order.sub_lt(U32.to_nat(index),U32.to_nat(first),rectangle_cells(rectangle),starts,ends)))

def fragment_value(+rectangle: Products.Rectangle,+stage: Tree.Tree<Tile.Tile16>,+offset: Nat) -> Tile.Tile8:
  Products.Rectangle{first,+positions,inner,+channels,+group_major,grid} = rectangle
  +groups = U32.to_nat(Gemm.blocks(channels))
  +position = position_of(group_major,U32.to_nat(positions),groups,offset)
  +group = group_of(group_major,U32.to_nat(positions),groups,offset)
  Gemm.staged_half(Tree.read(Tile.Tile16,stage,U32.from_nat(Values.slot(position,U32.to_nat(Gemm.pairs_of(channels)),Nat.div(group,2n)))),
    U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1))

def at_coordinate(+limits: Machine.Limits,+rectangle: Products.Rectangle,+first: U32,+count: U32,+stage: Tree.Tree<Tile.Tile16>,+local: Nat,+offset: Nat,
  +space: RectangleReadRoom<limits,rectangle>,+bounded: {U32.is_le(count,1073741824) == True{} : Bool},+inside: {Nat.is_lt(local,U32.to_nat(count)) == True{} : Bool},
  +sum_exact: {U32.to_nat((first - rectangle_start(rectangle) + U32.from_nat(local) : U32)) == offset : Nat},
  +coords: Located<group_major_of(rectangle),U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle))),offset>) ->
  {Tree.read(Tile.Tile8,fragment_tree(rectangle,first,count,stage),U32.from_nat(local)) == fragment_value(rectangle,stage,offset) : Tile.Tile8}:
  Products.Rectangle{+origin,+positions,+inner,+channels,+group_major,+grid} = rectangle
  RectangleReadRoom{+pos_positive,+pos_room,+go_positive,+go_room,+fewer,+room} = space
  Located{+position_inside,+group_inside,+order_exact} = coords
  +groups = U32.to_nat(Gemm.blocks(channels))
  +position = position_of(group_major,U32.to_nat(positions),groups,offset)
  +group = group_of(group_major,U32.to_nat(positions),groups,offset)
  +used = Values.slot(position,groups,group)
  +total = rectangle_cells(rectangle)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +cell_room = Order.transitive(used,total,capacity,
    Bounds.strict_le(used,total,Index.radix(position,U32.to_nat(positions),group,groups,position_inside,group_inside)),
    Order.transitive(total,Nat.add(U32.to_nat(origin),total),capacity,Order.right_le(U32.to_nat(origin),total),room))
  fragment_cell(limits,group_major,channels,positions,(first - origin : U32),count,local,position,group,stage,bounded,inside,pos_room,pos_positive,position_inside,
    go_room,go_positive,fewer,group_inside,cell_room,
    Equal.trans(Nat,U32.to_nat((first - origin + U32.from_nat(local) : U32)),offset,ordered_index(group_major,U32.to_nat(positions),groups,position,group),sum_exact,order_exact))

def rectangle_cell(+limits: Machine.Limits,+rectangle: Products.Rectangle,+first: U32,+count: U32,+stage: Tree.Tree<Tile.Tile16>,+local: Nat,
  +space: RectangleReadRoom<limits,rectangle>,
  +starts: {Nat.is_le(U32.to_nat(rectangle_start(rectangle)),U32.to_nat(first)) == True{} : Bool},
  +ends: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))) == True{} : Bool},
  +inside: {Nat.is_lt(local,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,fragment_tree(rectangle,first,count,stage),U32.from_nat(local)) ==
    cell_value(RectangleRead{rectangle,stage},U32.from_nat(Nat.add(U32.to_nat(first),local))) : Tile.Tile8}:
  Products.Rectangle{+origin,+positions,+inner,+channels,+group_major,+grid} = rectangle
  RectangleReadRoom{+pos_positive,+pos_room,+go_positive,+go_room,+fewer,+room} = space
  +capacity = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +first_nat = U32.to_nat(first)
  +origin_nat = U32.to_nat(origin)
  +end = Nat.add(first_nat,U32.to_nat(count))
  +limit = Nat.add(origin_nat,rectangle_cells(rectangle))
  +end_room = Order.transitive(end,limit,U32.to_nat(capacity),ends,room)
  +first_room = Order.transitive(first_nat,end,U32.to_nat(capacity),Order.le_add(first_nat,U32.to_nat(count)),end_room)
  +count_room = Order.transitive(U32.to_nat(count),end,U32.to_nat(capacity),Order.right_le(first_nat,U32.to_nat(count)),end_room)
  +global = Nat.add(first_nat,local)
  +global_inside = Order.lt_le(global,end,limit,Order.offset_lt(first_nat,local,U32.to_nat(count),inside),ends)
  +global_room = Order.transitive(global,limit,U32.to_nat(capacity),Bounds.strict_le(global,limit,global_inside),room)
  +relative_first = Nat.sub(first_nat,origin_nat)
  +local_offset = Nat.add(relative_first,local)
  +relative_word = (first - origin : U32)
  +relative_exact = Arithmetic.subtract(first,origin,capacity,cap_ok,starts,
    Order.transitive(relative_first,first_nat,U32.to_nat(capacity),Order.sub_le(first_nat,origin_nat),first_room))
  +origin_sum = Equal.trans(Nat,Nat.add(origin_nat,local_offset),Nat.add(Nat.add(origin_nat,relative_first),local),global,
    Equal.sym(Nat,Nat.add(Nat.add(origin_nat,relative_first),local),Nat.add(origin_nat,local_offset),Addition.associative(origin_nat,relative_first,local)),
    Equal.cong(Nat,Nat,base => Nat.add(base,local),Nat.add(origin_nat,relative_first),first_nat,Order.restore_shift(origin_nat,first_nat,starts)))
  +offset_inside = Order.cancel_offset_lt(origin_nat,local_offset,rectangle_cells(rectangle),
    %Equal.sym(Nat,Nat.add(origin_nat,local_offset),global,origin_sum) : {Nat.is_lt(_,limit) == True{} : Bool} global_inside)
  +offset_room = Order.transitive(local_offset,rectangle_cells(rectangle),U32.to_nat(capacity),
    Bounds.strict_le(local_offset,rectangle_cells(rectangle),offset_inside),
    Order.transitive(rectangle_cells(rectangle),limit,U32.to_nat(capacity),Order.right_le(origin_nat,rectangle_cells(rectangle)),room))
  +local_exact = Bounds.roundtrip(local,capacity,cap_ok,Order.transitive(local,U32.to_nat(count),U32.to_nat(capacity),Bounds.strict_le(local,U32.to_nat(count),inside),count_room))
  +sum_exact = Arithmetic.sum_exact(capacity,cap_ok,relative_word,U32.from_nat(local),relative_first,local,relative_exact,local_exact,offset_room)
  +global_exact = Bounds.roundtrip(global,capacity,cap_ok,global_room)
  +offset_exact = Equal.trans(Nat,Nat.sub(U32.to_nat(U32.from_nat(global)),origin_nat),Nat.sub(global,origin_nat),local_offset,
    Equal.cong(Nat,Nat,cell => Nat.sub(cell,origin_nat),U32.to_nat(U32.from_nat(global)),global,global_exact),
    %origin_sum : {Nat.sub(_,origin_nat) == local_offset : Nat} Order.subtract_shift(origin_nat,local_offset))
  %Equal.sym(Nat,Nat.sub(U32.to_nat(U32.from_nat(global)),origin_nat),local_offset,offset_exact) :
    {Tree.read(Tile.Tile8,fragment_tree(rectangle,first,count,stage),U32.from_nat(local)) == fragment_value(rectangle,stage,_) : Tile.Tile8}
  at_coordinate(limits,rectangle,first,count,stage,local,local_offset,space,Limits.count_allocatable(limits,count,count_room),inside,sum_exact,
    located(group_major,U32.to_nat(positions),U32.to_nat(Gemm.blocks(channels)),local_offset,pos_positive,go_positive,offset_inside))

type FragmentRead is Data:
  FragmentRead{limits: Machine.Limits,rectangle: Products.Rectangle,first: U32,count: U32,stage: Tree.Tree<Tile.Tile16>,
    space: RectangleReadRoom<limits,rectangle>,
    starts: {Nat.is_le(U32.to_nat(rectangle_start(rectangle)),U32.to_nat(first)) == True{} : Bool},
    ends: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))) == True{} : Bool}}

def fragment_length(fragment: FragmentRead) -> Nat:
  FragmentRead{limits,rectangle,first,count,stage,space,starts,ends} = fragment
  U32.to_nat(count)

def AtFragment(+fragment: FragmentRead,+index: Nat) -> Data:
  FragmentRead{+limits,+rectangle,+first,+count,+stage,space,starts,ends} = fragment
  Content.At(~Tile.Tile8,~RectangleRead,~cell_value,RectangleRead{rectangle,stage},first,fragment_tree(rectangle,first,count,stage),U32.to_nat(count),index)

def fragment_at(+fragment: FragmentRead,+index: Nat,+inside: {Nat.is_lt(index,fragment_length(fragment)) == True{} : Bool}) ->
  AtFragment(fragment,index):
  FragmentRead{+limits,+rectangle,+first,+count,+stage,+space,+starts,+ends} = fragment
  Gemm.Both{inside,rectangle_cell(limits,rectangle,first,count,stage,index,space,starts,ends,inside)}

def rectangle_room(+limits: Machine.Limits,+rectangle: Products.Rectangle,+space: RectangleReadRoom<limits,rectangle>) ->
  {Nat.is_le(Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  RectangleReadRoom{pos_positive,positions_room,groups_positive,groups_room,fewer,inside} = space
  inside

def fragment_holds(+limits: Machine.Limits,+rectangle: Products.Rectangle,+first: U32,+count: U32,+stage: Tree.Tree<Tile.Tile16>,
  +space: RectangleReadRoom<limits,rectangle>,
  +starts: {Nat.is_le(U32.to_nat(rectangle_start(rectangle)),U32.to_nat(first)) == True{} : Bool},
  +ends: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))) == True{} : Bool}) ->
  Placement.PieceHolds(~Tile.Tile8,~RectangleRead,~cell_value,RectangleRead{rectangle,stage},first,count,Buffer.Content{fragment_tree(rectangle,first,count,stage),U32.to_nat(count)}):
  +end = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +bound = Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))
  +cap = U32.to_nat(Machine.capacity(limits))
  +count_room = Order.transitive(U32.to_nat(count),end,cap,Order.right_le(U32.to_nat(first),U32.to_nat(count)),
    Order.transitive(end,bound,cap,ends,rectangle_room(limits,rectangle,space)))
  Gemm.Both{Members.range_all(~FragmentRead,~AtFragment,~fragment_length,~fragment_at,U32.to_nat(count),0n,FragmentRead{limits,rectangle,first,count,stage,space,starts,ends},
      Order.reflexive(U32.to_nat(count))),
    Gemm.Both{{==},slice_ready(sliced(rectangle,first),count,stage,Limits.count_allocatable(limits,count,count_room))}}

def ReadPieces(+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>,contents: Model.Content<Tile.Tile8>) -> Data:
  match contents:
    case Model.PieceContent{+buffer,+first,+count}:
      Placement.PieceHolds(~Tile.Tile8,~RectangleRead,~cell_value,RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},first,count,buffer)
    case Model.Joined{+left,+right}: Gemm.Both<ReadPieces(rectangle,stage,left),ReadPieces(rectangle,stage,right)>

def emitted_contents(remaining: Nat,+rectangle: Products.Rectangle,+end: U32,+piece: U32,+stage: Buffer.Content<Tile.Tile16>) -> Model.Content<Tile.Tile8>:
  Products.Rectangle{+first,+positions,inner,+channels,+group_major,+grid} = rectangle
  Point.emitted_tree(remaining,grid,piece,first,end,Gemm.blocks(channels),positions,group_major,stage)

def emitted_holds(remaining: Nat,+limits: Machine.Limits,+rectangle: Products.Rectangle,+end: U32,+piece: U32,+stage: Buffer.Content<Tile.Tile16>,
  +space: RectangleReadRoom<limits,rectangle>,
  +inside: Enclosed(U32.to_nat(rectangle_start(rectangle)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)),
    Model.content_layout(~Tile.Tile8,emitted_contents(remaining,rectangle,end,piece,stage)))) ->
  ReadPieces(rectangle,stage,emitted_contents(remaining,rectangle,end,piece,stage)):
  match remaining:
    case 0n:
      Products.Rectangle{+origin,+positions,+inner,+channels,+group_major,+grid} = rectangle
      Buffer.Content{+tree,+length} = stage
      Held.Both{+starts,+ends} = inside
      +low = U32.max(origin,Grid.start(grid,piece))
      +count = (U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - low : U32)
      fragment_holds(limits,rectangle,low,count,tree,space,starts,ends)
    case 1n+ +rest:
      Products.Rectangle{+origin,+positions,+inner,+channels,+group_major,+grid} = rectangle
      Buffer.Content{+tree,+length} = stage
      Held.Both{+head,+tail} = inside
      Held.Both{+starts,+ends} = head
      +low = U32.max(origin,Grid.start(grid,piece))
      +count = (U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - low : U32)
      Gemm.Both{fragment_holds(limits,rectangle,low,count,tree,space,starts,ends),emitted_holds(rest,limits,rectangle,end,(piece + 1 : U32),stage,space,tail)}

def reader_holds(+limits: Machine.Limits,+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>,
  +target: Pieces.Ranges,+space: RectangleReadRoom<limits,rectangle>,
  +laid: TaskProofs.Laid(target,Point.clipped_rectangle(rectangle),U32.to_nat(rectangle_start(rectangle)),
    Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)))) ->
  ReadPieces(rectangle,stage,Point.rectangle_model(rectangle,stage)):
  Products.Rectangle{+first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  Held.Both{+parts,+end_exact} = laid
  +groups = Gemm.blocks(channels)
  +count = (positions * groups : U32)
  +end = (first + count : U32)
  +piece = Grid.piece_of(grid,first)
  +remaining = Grid.later_pieces(grid,first,count)
  +low = U32.to_nat(first)
  +high = Nat.add(low,rectangle_cells(rectangle))
  +clipped = Point.clipped_rectangle(rectangle)
  +emitted = emitted_contents(remaining,rectangle,end,piece,stage)
  +within = enclosed(target,clipped,low,low,high,parts,Order.reflexive(low),
    %Equal.sym(Nat,Placed.ranges_end(clipped,low),high,end_exact) : {Nat.is_le(_,high) == True{} : Bool} Order.reflexive(high))
  %Equal.sym(Point.Emitted,Point.emit_model(remaining,grid,piece,first,end,groups,positions,group_major,Point.EmitInput{Point.Empty{},stage}),
    Point.Emitted{emitted,Point.Empty{},stage},Point.emission_tree(remaining,grid,piece,first,end,groups,positions,group_major,Point.Empty{},stage)) :
    ReadPieces(rectangle,stage,Point.output_content(_))
  emitted_holds(remaining,limits,rectangle,end,piece,stage,space,
    %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,emitted),clipped,Point.emitted_tree_layout(remaining,grid,piece,first,end,groups,positions,group_major,stage)) :
      Enclosed(low,high,_)
    within)

def lower_of(+source: RectangleRead) -> Nat:
  RectangleRead{+rectangle,stage} = source
  U32.to_nat(rectangle_start(rectangle))

def upper_of(+source: RectangleRead) -> Nat:
  RectangleRead{+rectangle,stage} = source
  Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle))

def Agreement(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data) -> Type:
  @+source: RectangleRead -> @+target: Target -> @+related: Related(source,target) -> @+index: U32 ->
    @+starts: {Nat.is_le(lower_of(source),U32.to_nat(index)) == True{} : Bool} ->
    @+ends: {Nat.is_lt(U32.to_nat(index),upper_of(source)) == True{} : Bool} ->
    {cell_value(source,index) == target_value(target,index) : Tile.Tile8}

type Restate<-Target: Data,-Related: RectangleRead -> Target -> Data> is Data:
  Restate{source: RectangleRead,target: Target,related: Related(source,target),first: U32,count: U32,tree: Tree.Tree<Tile.Tile8>,length: Nat,
    limits: Machine.Limits,begin: {Nat.is_le(lower_of(source),U32.to_nat(first)) == True{} : Bool},
    end: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),upper_of(source)) == True{} : Bool},
    room: {Nat.is_le(upper_of(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    length_exact: {length == U32.to_nat(count) : Nat}}

def prior_cell(~Target: Data,~Related: RectangleRead -> Target -> Data,+scope: Restate<Target,Related>,+index: Nat) -> Data:
  Restate{source,target,related,first,count,tree,length,limits,begin,end,room,exact} = scope
  Content.At(~Tile.Tile8,~RectangleRead,~cell_value,source,first,tree,length,index)

def later_cell(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data,+scope: Restate<Target,Related>,+index: Nat) -> Data:
  Restate{source,target,related,first,count,tree,length,limits,begin,end,room,exact} = scope
  Content.At(~Tile.Tile8,~Target,~target_value,target,first,tree,length,index)

def cell_restate(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data,
  ~agrees: Agreement(~Target,~target_value,~Related),+scope: Restate<Target,Related>,+index: Nat,
  +prior: prior_cell(~Target,~Related,scope,index)) -> later_cell(~Target,~target_value,~Related,scope,index):
  Restate{+source,+target,+related,+first,+count,+tree,+length,+limits,+begin,+end,+room,+exact} = scope
  Gemm.Both{+inside,+read} = prior
  +position = Nat.add(U32.to_nat(first),index)
  +bound = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +start = Order.transitive(lower_of(source),U32.to_nat(first),position,begin,Order.le_add(U32.to_nat(first),index))
  +last = Order.lt_le(position,bound,upper_of(source),
    Order.offset_lt(U32.to_nat(first),index,U32.to_nat(count),%exact : {Nat.is_lt(index,_) == True{} : Bool} inside),end)
  +capacity = Machine.capacity(limits)
  +word = U32.from_nat(position)
  +word_exact = Bounds.roundtrip(position,capacity,Limits.capacity_fits(limits),
    Bounds.strict_le(position,U32.to_nat(capacity),Order.lt_le(position,upper_of(source),U32.to_nat(capacity),last,room)))
  Gemm.Both{inside,Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,U32.from_nat(index)),cell_value(source,word),target_value(target,word),read,
    agrees(source,target,related,word,
      %Equal.sym(Nat,U32.to_nat(word),position,word_exact) : {Nat.is_le(lower_of(source),_) == True{} : Bool} start,
      %Equal.sym(Nat,U32.to_nat(word),position,word_exact) : {Nat.is_lt(_,upper_of(source)) == True{} : Bool} last))}

def restated_piece(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data,
  ~agrees: Agreement(~Target,~target_value,~Related),+limits: Machine.Limits,+source: RectangleRead,+target: Target,+related: Related(source,target),
  +first: U32,+count: U32,buffer: Buffer.Content<Tile.Tile8>,
  +begin: {Nat.is_le(lower_of(source),U32.to_nat(first)) == True{} : Bool},
  +end: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),upper_of(source)) == True{} : Bool},
  +room: {Nat.is_le(upper_of(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +holds: Placement.PieceHolds(~Tile.Tile8,~RectangleRead,~cell_value,source,first,count,buffer)) ->
  Placement.PieceHolds(~Tile.Tile8,~Target,~target_value,target,first,count,buffer):
  Buffer.Content{+tree,+length} = buffer
  Gemm.Both{+values,+filled} = holds
  Gemm.Both{+exact,+ready} = filled
  Gemm.Both{Members.transform_all(~Nat,~Restate<Target,Related>,~prior_cell(~Target,~Related),~later_cell(~Target,~target_value,~Related),
    ~cell_restate(~Target,~target_value,~Related,~agrees),Ranges.range(U32.to_nat(count),0n),
    Restate{source,target,related,first,count,tree,length,limits,begin,end,room,exact},values),filled}

def RestatedPieces(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,+target: Target,contents: Model.Content<Tile.Tile8>) -> Data:
  match contents:
    case Model.PieceContent{+buffer,+first,+count}: Placement.PieceHolds(~Tile.Tile8,~Target,~target_value,target,first,count,buffer)
    case Model.Joined{+left,+right}: Gemm.Both<RestatedPieces(~Target,~target_value,target,left),RestatedPieces(~Target,~target_value,target,right)>

def restated_tree(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data,
  ~agrees: Agreement(~Target,~target_value,~Related),+limits: Machine.Limits,+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>,
  +target: Target,+related: Related(RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},target),
  contents: Model.Content<Tile.Tile8>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  enclosed: Enclosed(U32.to_nat(rectangle_start(rectangle)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)),Model.content_layout(~Tile.Tile8,contents)),
  holds: ReadPieces(rectangle,stage,contents)) -> RestatedPieces(~Target,~target_value,target,contents):
  match contents:
    case Model.PieceContent{+buffer,+first,+count}:
      Held.Both{+begin,+end} = enclosed
      restated_piece(~Target,~target_value,~Related,~agrees,limits,RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},target,related,
        first,count,buffer,begin,end,room,holds)
    case Model.Joined{+left,+right}:
      Held.Both{+left_enclosed,+right_enclosed} = enclosed
      Gemm.Both{+left_holds,+right_holds} = holds
      Gemm.Both{restated_tree(~Target,~target_value,~Related,~agrees,limits,rectangle,stage,target,related,left,room,left_enclosed,left_holds),
        restated_tree(~Target,~target_value,~Related,~agrees,limits,rectangle,stage,target,related,right,room,right_enclosed,right_holds)}

def rectangle_restated(~Target: Data,~target_value: Target -> U32 -> Tile.Tile8,~Related: RectangleRead -> Target -> Data,
  ~agrees: Agreement(~Target,~target_value,~Related),+limits: Machine.Limits,+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>,
  +target: Target,+related: Related(RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},target),
  +layout: Pieces.Ranges,+space: RectangleReadRoom<limits,rectangle>,
  +laid: TaskProofs.Laid(layout,Point.clipped_rectangle(rectangle),U32.to_nat(rectangle_start(rectangle)),Nat.add(U32.to_nat(rectangle_start(rectangle)),rectangle_cells(rectangle)))) ->
  RestatedPieces(~Target,~target_value,target,Point.rectangle_model(rectangle,stage)):
  Held.Both{+parts,+exact} = laid
  +content = Point.rectangle_model(rectangle,stage)
  +low = U32.to_nat(rectangle_start(rectangle))
  +high = Nat.add(low,rectangle_cells(rectangle))
  +region = Point.clipped_rectangle(rectangle)
  +read_layout = Model.content_layout(~Tile.Tile8,content)
  +sorted = enclosed(layout,region,low,low,high,parts,Order.reflexive(low),
    %Equal.sym(Nat,Placed.ranges_end(region,low),high,exact) : {Nat.is_le(_,high) == True{} : Bool}
    Order.reflexive(high))
  restated_tree(~Target,~target_value,~Related,~agrees,limits,rectangle,stage,target,related,content,rectangle_room(limits,rectangle,space),
    %Equal.sym(Pieces.Ranges,read_layout,region,Point.rectangle_layout(rectangle,stage)) : Enclosed(low,high,_)
    sorted,reader_holds(limits,rectangle,stage,layout,space,laid))
