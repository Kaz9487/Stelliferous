# Task trees of point products, with their actual buffer owners and writeback ranges.
import Base
import ../default_kernel_cost.bend as DefaultCost
import ../kernel_cost.bend as Cost
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../kernel_window_pipeline.bend as Pipeline
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_halo.bend as Halo
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../tensor_tasks.bend as Tasks
import ../traversal_partition.bend as Partition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./kernel_gemm_output_groups.bend as OutputGroups
import ./kernel_window_windows.bend as KernelWindows
import ./machine_limit_bounds.bend as Limits
import ./nat_division.bend as Division
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./tensor_clipped_pieces.bend as Clipped
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_grouped_with.bend as Grouped
import ./tensor_pieces_model.bend as Model
import ./tensor_placed_parts.bend as Placed
import ./tensor_point_products.bend as Point
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_task_runs.bend as Runs
import ./traversal_partition_refinement.bend as PartitionProofs
import ./u32_division.bend as U32Division
import ./word_multiplication.bend as Multiply

def SpanFits(+start: U32,+length: U32,+total: Nat) -> Data:
  Held.Both<{Nat.is_le(Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat((start + length : U32)),U32.to_nat(start))),total) == True{} : Bool},
    {Nat.is_le(U32.to_nat(start),U32.to_nat((start + length : U32))) == True{} : Bool}>

# Scale a nonwrapping run of pieces into a contiguous operand span.
def scaled_span(+first: U32,+count: U32,+width: U32,+total: Nat,+capacity: U32,
  +cap_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inside: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(width)),total) == True{} : Bool},
  +room: {Nat.is_le(total,U32.to_nat(capacity)) == True{} : Bool}) ->
  SpanFits((first * width : U32),(count * width : U32),total):
  +start = Nat.mul(U32.to_nat(first),U32.to_nat(width))
  +length = Nat.mul(U32.to_nat(count),U32.to_nat(width))
  +end = Nat.add(start,length)
  +end_inside = Held.nat_le_moved(end,Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(width)),total,
    Equal.sym(Nat,Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(width)),end,Index.distrib(U32.to_nat(first),U32.to_nat(count),U32.to_nat(width))),inside)
  +end_room = Order.transitive(end,total,U32.to_nat(capacity),end_inside,room)
  +start_word = (first * width : U32)
  +length_word = (count * width : U32)
  +start_exact = Multiply.bounded_multiply(first,width,capacity,cap_ok,
    Order.transitive(start,end,U32.to_nat(capacity),Order.le_add(start,length),end_room))
  +length_exact = Multiply.bounded_multiply(count,width,capacity,cap_ok,
    Order.transitive(length,end,U32.to_nat(capacity),
      %Order.commutative(length,start) : {Nat.is_le(length,_) == True{} : Bool} Order.le_add(length,start),end_room))
  +end_exact = {%start_exact : {U32.to_nat((start_word + length_word : U32)) == Nat.add(_,length) : Nat}
    %length_exact : {U32.to_nat((start_word + length_word : U32)) == Nat.add(U32.to_nat(start_word),_) : Nat}
    Arithmetic.add(start_word,length_word,capacity,cap_ok,
      %Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(length_word)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(length_word),length,length_exact) : {Nat.is_le(Nat.add(start,_),U32.to_nat(capacity)) == True{} : Bool}
      end_room) : {U32.to_nat((start_word + length_word : U32)) == end : Nat}}
  %Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : Held.Both<{Nat.is_le(Nat.add(_,Nat.sub(U32.to_nat((start_word + length_word : U32)),_)),total) == True{} : Bool},
    {Nat.is_le(_,U32.to_nat((start_word + length_word : U32))) == True{} : Bool}>
  %Equal.sym(Nat,U32.to_nat((start_word + length_word : U32)),end,end_exact) :
    Held.Both<{Nat.is_le(Nat.add(start,Nat.sub(_,start)),total) == True{} : Bool},{Nat.is_le(start,_) == True{} : Bool}>
  %Equal.sym(Nat,Nat.sub(end,start),length,Order.subtract_shift(start,length)) :
    Held.Both<{Nat.is_le(Nat.add(start,_),total) == True{} : Bool},{Nat.is_le(start,end) == True{} : Bool}>
  Held.Both{end_inside,Order.le_add(start,length)}

type GroupInterval<-first: U32,-count: U32,-rows: U32,-channels: U32> is Data:
  GroupInterval{lower: Nat,length: Nat,first_group: {U32.to_nat((first / rows : U32)) == lower : Nat},
    first_aligned: {U32.to_nat(first) == Nat.mul(U32.to_nat((first / rows : U32)),U32.to_nat(rows)) : Nat},
    groups: {U32.to_nat((count / rows : U32)) == length : Nat},aligned: {U32.to_nat(count) == Nat.mul(length,U32.to_nat(rows)) : Nat},
    nonempty: {Nat.is_lt(0n,length) == True{} : Bool},inside: {Nat.is_le(Nat.add(lower,length),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}}

def group_interval(+rows: U32,+channels: U32,+capacity: U32,
  +pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+rows_room: {Nat.is_le(Nat.double(U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool},
  +cap_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +first: U32,+count: U32,
  +fits: Runs.Fits<U32.to_nat(rows),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))>) ->
  GroupInterval<first,count,rows,channels>:
  Runs.Fits{+lower,+upper,+low_exact,+high_exact,+ordered,+inside} = fits
  +unit = U32.to_nat(rows)
  +low = U32.to_nat(first)
  +high = Nat.add(low,U32.to_nat(count))
  +length = Nat.sub(upper,lower)
  +units_lt = Index.cancel_multiply_lt(lower,upper,unit,
    %low_exact : {Nat.is_lt(_,Nat.mul(upper,unit)) == True{} : Bool}
    %high_exact : {Nat.is_lt(low,_) == True{} : Bool}
    ordered)
  +units_le = Bounds.strict_le(lower,upper,units_lt)
  +upper_inside = Index.cancel_multiply_le(upper,U32.to_nat(Gemm.blocks(channels)),unit,pos,
    %high_exact : {Nat.is_le(_,Nat.mul(U32.to_nat(Gemm.blocks(channels)),unit)) == True{} : Bool} inside)
  +length_exact = Equal.trans(Nat,U32.to_nat(count),Nat.sub(high,low),Nat.mul(length,unit),
    Equal.sym(Nat,Nat.sub(high,low),U32.to_nat(count),Order.subtract_shift(low,U32.to_nat(count))),
    %Equal.sym(Nat,low,Nat.mul(lower,unit),low_exact) : {Nat.sub(high,_) == Nat.mul(length,unit) : Nat}
    %Equal.sym(Nat,high,Nat.mul(upper,unit),high_exact) : {Nat.sub(_,Nat.mul(lower,unit)) == Nat.mul(length,unit) : Nat}
    Index.subtract_products(upper,lower,unit))
  +first_group = (first / rows : U32)
  +groups = (count / rows : U32)
  +first_exact = Equal.trans(Nat,U32.to_nat(first_group),Nat.div(Nat.mul(lower,unit),unit),lower,
    U32Division.quotient_exact(capacity,cap_ok,first,rows,Nat.mul(lower,unit),unit,low_exact,{==},rows_room,pos),
    Division.exact_quotient(lower,unit,pos))
  +groups_exact = Equal.trans(Nat,U32.to_nat(groups),Nat.div(Nat.mul(length,unit),unit),length,
    U32Division.quotient_exact(capacity,cap_ok,count,rows,Nat.mul(length,unit),unit,length_exact,{==},rows_room,pos),
    Division.exact_quotient(length,unit,pos))
  GroupInterval{lower,length,first_exact,
    %Equal.sym(Nat,U32.to_nat(first_group),lower,first_exact) : {low == Nat.mul(_,unit) : Nat} low_exact,
    groups_exact,length_exact,Order.positive_difference(upper,lower,units_lt),
    %Equal.sym(Nat,Nat.add(lower,length),upper,Order.restore_shift(lower,upper,units_le)) :
      {Nat.is_le(_,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}
    upper_inside}

def group_span(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+capacity: U32,
  +cap_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +weights_room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(capacity)) == True{} : Bool},
  +first: U32,+count: U32,+interval: GroupInterval<first,count,rows,channels>) ->
  Windows.Reaches(~Products.Shape,~Products.shared_reach,Products.Shape{rows,inner,channels,grid},Pieces.Range{first,count},
    Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))):
  GroupInterval{+lower,+length,+first_exact,first_aligned,+groups_exact,+aligned,+nonempty,+inside} = interval
  +first_group = (first / rows : U32)
  +groups = (count / rows : U32)
  scaled_span(first_group,groups,inner,Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),capacity,cap_ok,
    %Equal.sym(Nat,U32.to_nat(first_group),lower,first_exact) :
      {Nat.is_le(Nat.mul(Nat.add(_,U32.to_nat(groups)),U32.to_nat(inner)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(groups),length,groups_exact) :
      {Nat.is_le(Nat.mul(Nat.add(lower,_),U32.to_nat(inner)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))) == True{} : Bool}
    Index.mul_le(Nat.add(lower,length),U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),inside),weights_room)

def clipped_extent(+rows: U32,+channels: U32,+first: U32,+count: U32) -> U32:
  U32.min((count / rows * 8 : U32),(channels - first / rows * 8 : U32))

def clipped_positive(+lower: Nat,+length: Nat,+total: Nat,+local: Nat,parts: OutputGroups.PartOutputs<lower,length,total,local>) ->
  {Nat.is_lt(0n,local) == True{} : Bool}:
  OutputGroups.PartOutputs{ceiling,positive,disjoint,starts,last} = parts
  positive
type Clipped<-limits: Machine.Limits,-rows: U32,-channels: U32,-first: U32,-count: U32> is Data:
  Clipped{extent_exact: {U32.to_nat(clipped_extent(rows,channels,first,count)) ==
      Nat.min(Nat.mul(U32.to_nat((count / rows : U32)),8n),Nat.sub(U32.to_nat(channels),Nat.mul(U32.to_nat((first / rows : U32)),8n))) : Nat},
    room: {Nat.is_le(U32.to_nat(clipped_extent(rows,channels,first,count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(clipped_extent(rows,channels,first,count))) == True{} : Bool},
    groups: {U32.to_nat(Gemm.blocks(clipped_extent(rows,channels,first,count))) == U32.to_nat((count / rows : U32)) : Nat}}

def clipped_facts(+limits: Machine.Limits,+rows: U32,+channels: U32,+first: U32,+count: U32,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +interval: GroupInterval<first,count,rows,channels>) -> Clipped<limits,rows,channels,first,count>:
  GroupInterval{+lower,+length,+first_exact,first_aligned,+groups_exact,+aligned,+nonempty,+inside} = interval
  +total = U32.to_nat(channels)
  +lower_word = (first / rows : U32)
  +length_word = (count / rows : U32)
  +local = Nat.min(Nat.mul(length,8n),Nat.sub(total,Nat.mul(lower,8n)))
  +extent = clipped_extent(rows,channels,first,count)
  +parts = OutputGroups.outputs_split(lower,length,total,nonempty,
    %Encoded.eight_blocks_exact(limits,channels,channel_room) : {Nat.is_le(Nat.add(lower,length),_) == True{} : Bool} inside)
  +length_positive = {%Equal.sym(Nat,U32.to_nat(length_word),length,groups_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    nonempty : {Nat.is_lt(0n,U32.to_nat(length_word)) == True{} : Bool}}
  +encoded_parts = {%Equal.sym(Nat,U32.to_nat(lower_word),lower,first_exact) :
      OutputGroups.PartOutputs<_,U32.to_nat(length_word),total,Nat.min(Nat.mul(U32.to_nat(length_word),8n),Nat.sub(total,Nat.mul(_,8n)))>
    %Equal.sym(Nat,U32.to_nat(length_word),length,groups_exact) :
      OutputGroups.PartOutputs<lower,_,total,Nat.min(Nat.mul(_,8n),Nat.sub(total,Nat.mul(lower,8n)))>
    parts : OutputGroups.PartOutputs<U32.to_nat(lower_word),U32.to_nat(length_word),total,Nat.min(Nat.mul(U32.to_nat(length_word),8n),Nat.sub(total,Nat.mul(U32.to_nat(lower_word),8n)))>}
  +extent_exact = {%first_exact : {U32.to_nat(extent) == Nat.min(Nat.mul(length,8n),Nat.sub(total,Nat.mul(_,8n))) : Nat}
    %groups_exact : {U32.to_nat(extent) == Nat.min(Nat.mul(_,8n),Nat.sub(total,Nat.mul(U32.to_nat(lower_word),8n))) : Nat}
    OutputGroups.part_outputs_exact(channels,lower_word,length_word,limits,total,{==},channel_room,length_positive,encoded_parts)
    : {U32.to_nat(extent) == local : Nat}}
  +local_room = Order.transitive(local,Nat.sub(total,Nat.mul(lower,8n)),U32.to_nat(Machine.capacity(limits)),
    Extrema.minimum_right(Nat.mul(length,8n),Nat.sub(total,Nat.mul(lower,8n))),
    Order.transitive(Nat.sub(total,Nat.mul(lower,8n)),total,U32.to_nat(Machine.capacity(limits)),Order.sub_le(total,Nat.mul(lower,8n)),channel_room))
  +extent_room = Held.nat_le_moved(U32.to_nat(extent),local,U32.to_nat(Machine.capacity(limits)),extent_exact,local_room)
  +group_word = Gemm.blocks(extent)
  +ceiling_exact = OutputGroups.clipped_groups(lower,length,total,local,parts)
  +result_groups = Equal.trans(Nat,U32.to_nat(group_word),Ceiling.ceiling(U32.to_nat(extent),8n),length,
    Encoded.eight_blocks_exact(limits,extent,extent_room),
    %Equal.sym(Nat,U32.to_nat(extent),local,extent_exact) : {Ceiling.ceiling(_,8n) == length : Nat} ceiling_exact)
  Clipped{
    %Equal.sym(Nat,U32.to_nat(lower_word),lower,first_exact) :
      {U32.to_nat(extent) == Nat.min(Nat.mul(U32.to_nat(length_word),8n),Nat.sub(total,Nat.mul(_,8n))) : Nat}
    %Equal.sym(Nat,U32.to_nat(length_word),length,groups_exact) :
      {U32.to_nat(extent) == Nat.min(Nat.mul(_,8n),Nat.sub(total,Nat.mul(lower,8n))) : Nat}
    extent_exact,
    extent_room,
    %Equal.sym(Nat,U32.to_nat(extent),local,extent_exact) : {Nat.is_lt(0n,_) == True{} : Bool} clipped_positive(lower,length,total,local,parts),
    Equal.trans(Nat,U32.to_nat(group_word),length,U32.to_nat(length_word),result_groups,Equal.sym(Nat,U32.to_nat(length_word),length,groups_exact))}

def clipped_cells(+limits: Machine.Limits,+rows: U32,+channels: U32,+first: U32,+count: U32,
  +interval: GroupInterval<first,count,rows,channels>,+clip: Clipped<limits,rows,channels,first,count>) ->
  {U32.to_nat(count) == Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(clipped_extent(rows,channels,first,count)))) : Nat}:
  GroupInterval{+lower,+length,+first_exact,first_aligned,+groups_exact,+aligned,+nonempty,+inside} = interval
  Clipped{+extent_exact,+room,+positive,+groups} = clip
  +group_word = Gemm.blocks(clipped_extent(rows,channels,first,count))
  %Equal.sym(Nat,U32.to_nat(group_word),U32.to_nat((count / rows : U32)),groups) : {U32.to_nat(count) == Nat.mul(U32.to_nat(rows),_) : Nat}
  %Equal.sym(Nat,U32.to_nat((count / rows : U32)),length,groups_exact) : {U32.to_nat(count) == Nat.mul(U32.to_nat(rows),_) : Nat}
  Equal.trans(Nat,U32.to_nat(count),Nat.mul(length,U32.to_nat(rows)),Nat.mul(U32.to_nat(rows),length),aligned,Index.multiply_commutative(length,U32.to_nat(rows)))

def clipped_count(+limits: Machine.Limits,+rows: U32,+channels: U32,+first: U32,+count: U32,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +interval: GroupInterval<first,count,rows,channels>) ->
  {(rows * Gemm.blocks(clipped_extent(rows,channels,first,count)) : U32) == count : U32}:
  +extent = clipped_extent(rows,channels,first,count)
  +groups = Gemm.blocks(extent)
  +natural = Nat.mul(U32.to_nat(rows),U32.to_nat(groups))
  +cells = clipped_cells(limits,rows,channels,first,count,interval,clipped_facts(limits,rows,channels,first,count,channel_room,interval))
  +exact = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),rows,groups,U32.to_nat(rows),U32.to_nat(groups),{==},{==},
    %cells : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} count_room)
  Arithmetic.to_nat_injective((rows * groups : U32),count,
    Equal.trans(Nat,U32.to_nat((rows * groups : U32)),natural,U32.to_nat(count),exact,Equal.sym(Nat,U32.to_nat(count),natural,cells)))

def shared_reaches(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+capacity: U32,
  +pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+rows_room: {Nat.is_le(Nat.double(U32.to_nat(rows)),U32.to_nat(capacity)) == True{} : Bool},
  +cap_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +weights_room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(capacity)) == True{} : Bool},
  tasks: Pieces.Ranges,+run: Runs.Run(tasks,U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)))) ->
  Windows.Reaches(~Products.Shape,~Products.shared_reach,Products.Shape{rows,inner,channels,grid},tasks,
    Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))):
  match tasks:
    case Pieces.Range{+first,+count}:
      group_span(rows,inner,channels,grid,capacity,cap_ok,weights_room,first,count,
        group_interval(rows,channels,capacity,pos,rows_room,cap_ok,first,count,run))
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,rest} = run
      Held.Both{+right_run,meets} = rest
      Held.Both{shared_reaches(rows,inner,channels,grid,capacity,pos,rows_room,cap_ok,weights_room,left,left_run),
        shared_reaches(rows,inner,channels,grid,capacity,pos,rows_room,cap_ok,weights_room,right,right_run)}

def shared_grid(+rows: U32,+outputs: U32) -> Grid.Grid:
  Tensors.grid_of([Gemm.blocks(outputs),1,rows])

def shared_extra(+rows: U32,+inner: U32) -> Tasks.Extra:
  Tasks.Extra{U32.to_f32((rows * Gemm.blocks(inner) * 32 : U32)),0.0,0.0}

def shared_cut_ns() -> F32:
  Cost.copy_ns(32.0,DefaultCost.configuration())

def shared_decision(+rows: U32,+inner: U32,+outputs: U32,+workers: U32) -> Tasks.Decision:
  Tasks.decide_cut(shared_grid(rows,outputs),workers,rows,U32.to_f32(inner),0.0,shared_cut_ns(),shared_extra(rows,inner))

def shared_cells(+rows: U32,+outputs: U32) -> {ShapeLaws.elements([Gemm.blocks(outputs),1,rows]) == Nat.mul(U32.to_nat(Gemm.blocks(outputs)),U32.to_nat(rows)) : Nat}:
  +groups = U32.to_nat(Gemm.blocks(outputs))
  +unit = U32.to_nat(rows)
  Equal.trans(Nat,ShapeLaws.elements([Gemm.blocks(outputs),1,rows]),Nat.mul(groups,Nat.mul(1n,unit)),Nat.mul(groups,unit),
    ShapeLaws.elements_three(Gemm.blocks(outputs),1,rows),
    Equal.cong(Nat,Nat,value => Nat.mul(groups,value),Nat.mul(1n,unit),unit,Addition.zero_right(unit)))

# The cut decision used by shared() covers the entire output in whole groups.
def shared_run(+limits: Machine.Limits,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(outputs),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(outputs),1,rows])) == True{} : Bool}) ->
  Runs.RunOf(Runs.decision_tasks(shared_decision(rows,inner,outputs,workers)),Pieces.ranges_of_grid(shared_grid(rows,outputs)),
    U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(outputs)),U32.to_nat(rows))):
  +shape = {[Gemm.blocks(outputs),1,rows] : List<&2,U32>}
  +unit = U32.to_nat(rows)
  +bounds = ShapeLaws.elements(shape)
  +grid = shared_grid(rows,outputs)
  +divides = {GridRanges.Multiple{1n,Equal.sym(Nat,Nat.add(unit,0n),unit,Addition.zero_right(unit))} : GridRanges.Multiple<unit,unit>}
  +layout = GridRanges.grid_layout(limits,shape,Grid.piece_target(),room,nonempty,unit,divides)
  +cut_shape = GridRanges.shape_cut(limits,shape,Grid.piece_target(),room,nonempty,unit,divides,rows,GridRanges.WholeUnits{divides,rows_room})
  +cut_word = U32.to_nat(Runs.cut_unit_of(grid,rows))
  +cut = Held.first_of(GridRanges.Multiple<cut_word,unit>,{Nat.is_le(cut_word,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},cut_shape)
  +cut_room = Held.second_of(GridRanges.Multiple<cut_word,unit>,{Nat.is_le(cut_word,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},cut_shape)
  %shared_cells(rows,outputs) : Runs.RunOf(Runs.decision_tasks(shared_decision(rows,inner,outputs,workers)),Pieces.ranges_of_grid(grid),unit,_)
  Runs.decided_cut_ends(limits,grid,workers,rows,U32.to_f32(inner),0.0,shared_cut_ns(),shared_extra(rows,inner),
    unit,bounds,GridRanges.multiple_times(cut_word,unit,cut),GridRanges.multiple_exact(cut_word,unit,cut),cut_room,
    GridRanges.layout_run(shape,Grid.piece_target(),unit,layout),room)

def actual_shared_reaches(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +weights_room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Windows.Reaches(~Products.Shape,~Products.shared_reach,Products.Shape{rows,inner,channels,shared_grid(rows,channels)},
    Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))):
  +grid = shared_grid(rows,channels)
  +tasks = Runs.decision_tasks(shared_decision(rows,inner,channels,workers))
  +unit = U32.to_nat(rows)
  +bound = Nat.mul(U32.to_nat(Gemm.blocks(channels)),unit)
  +planned = shared_run(limits,rows,inner,channels,workers,rows_room,room,nonempty)
  +run = Runs.fold_run_of(Pieces.ranges_of_grid(grid),tasks,unit,bound,planned)
  shared_reaches(rows,inner,channels,shared_grid(rows,channels),Machine.half_word(limits),pos,
    Limits.double_bound(limits,U32.to_nat(rows),rows_room),Limits.half_word_fits(limits),
    Limits.within_half(limits,Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),weights_room),
    Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),run)

def weight_shape(+inner: U32,+channels: U32) -> List<&2,U32>:
  [Gemm.blocks(channels),inner]

def weight_cells(+inner: U32,+channels: U32) ->
  {ShapeLaws.elements(weight_shape(inner,channels)) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)) : Nat}:
  Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(Gemm.blocks(channels)),value),Nat.mul(U32.to_nat(inner),1n),U32.to_nat(inner),Order.times_one(U32.to_nat(inner)))

# The actual shared task's reads are windows of the caller's weight tensor.
def actual_shared_windows(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,
  +rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +weights_room: {Nat.is_le(ShapeLaws.elements(weight_shape(inner,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(weight_shape(inner,channels))) == True{} : Bool},weights: Pieces.Pieces(~Tile.Tile8),
  +covered: Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,weights),0n,ShapeLaws.elements(weight_shape(inner,channels)),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(weight_shape(inner,channels))),Held.Ended{})>) ->
  Windows.Windowed<Tile.Tile8,Source,value,source,Products.Shape,Products.shared_reach,Products.Shape{rows,inner,channels,shared_grid(rows,channels)},
    ShapeLaws.elements(weight_shape(inner,channels)),Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),
    Halo.windows(~Tile.Tile8,~Products.Shape,~Products.shared_reach,Products.Shape{rows,inner,channels,shared_grid(rows,channels)},Tensors.grid_of(weight_shape(inner,channels)),
      Products.zero(),Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),weights)>:
  +shape = weight_shape(inner,channels)
  +total = ShapeLaws.elements(shape)
  +grid = Tensors.grid_of(shape)
  +work = {Products.Shape{rows,inner,channels,shared_grid(rows,channels)} : Products.Shape}
  +tasks = Runs.decision_tasks(shared_decision(rows,inner,channels,workers))
  +reaches = {%Equal.sym(Nat,total,Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),weight_cells(inner,channels)) :
    Windows.Reaches(~Products.Shape,~Products.shared_reach,work,tasks,_)
    actual_shared_reaches(limits,rows,inner,channels,workers,pos,rows_room,out_room,out_nonempty,
      %weight_cells(inner,channels) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} weights_room)
    : Windows.Reaches(~Products.Shape,~Products.shared_reach,work,tasks,total)}
  Windows.covered_windows(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,limits,grid,Pieces.ranges_of_grid(grid),total,
    KernelWindows.input_laid(limits,shape,weights_room,weights_nonempty),work,source,Products.zero(),Machine.capacity(limits),Limits.capacity_allocatable(limits),weights_room,
    GridRanges.grid_pieces_word(limits,shape,Grid.piece_target(),weights_room,weights_nonempty),tasks,reaches,weights,covered)

def TaskModel() -> Data:
  Partition.Tree<&2,Model.Model<Tile.Tile8>,Unit>

def pack_task(model: TaskModel()) -> Point.TaskOutput():
  PartitionProofs.carry(~&2,~&1,~Model.Model<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Unit,~Model.pack(~Tile.Tile8),model)

type ObservedTask<-native: Point.TaskOutput()> is Data:
  ObservedTask{model: TaskModel(),exact: {pack_task(model) == native : Point.TaskOutput()}}

def observed_leaf(+first: U32,+count: U32,-pieces: Pieces.Pieces(~Tile.Tile8),observed: Model.Observation<Tile.Tile8,pieces>) ->
  ObservedTask<Partition.Leaf{pieces,first,count,Unit{}}>:
  Model.Observation{+model,exact} = observed
  ObservedTask{Partition.Leaf{model,first,count,Unit{}},
    %exact : {Partition.Leaf{Model.pack(~Tile.Tile8,model),first,count,Unit{}} == Partition.Leaf{_,first,count,Unit{}} : Point.TaskOutput()}
    {==}}

def observed_fork(-left: Point.TaskOutput(),-right: Point.TaskOutput(),left_observed: ObservedTask<left>,right_observed: ObservedTask<right>) ->
  ObservedTask<Partition.Fork{left,right}>:
  ObservedTask{+left_model,left_same} = left_observed
  ObservedTask{+right_model,right_same} = right_observed
  ObservedTask{Partition.Fork{left_model,right_model},
    %left_same : {Partition.Fork{pack_task(left_model),pack_task(right_model)} == Partition.Fork{_,right} : Point.TaskOutput()}
    %right_same : {Partition.Fork{pack_task(left_model),pack_task(right_model)} == Partition.Fork{pack_task(left_model),_} : Point.TaskOutput()}
    {==}}

def observe_task(task: Point.TaskOutput()) -> ObservedTask<task>:
  match task:
    case Partition.Leaf{pieces,+first,+count,Unit{}}: observed_leaf(first,count,pieces,Model.observe(~Tile.Tile8,pieces))
    case Partition.Fork{left,right}: observed_fork(left,right,observe_task(left),observe_task(right))

def split_models(-source: Point.SharedResource,copies: Point.SharedCopies<source>) -> PartitionProofs.Split<Point.SharedResource>:
  Point.SharedCopies{left,right,exact,left_content,right_content} = copies
  PartitionProofs.Split{left,right}

def split_resources(source: Point.SharedResource) -> PartitionProofs.Split<Point.SharedResource>:
  +source = source
  split_models(source,Point.shared_copies(source))

def split_observed(+source: Point.SharedResource,+copies: Point.SharedCopies<source>) ->
  {Products.shared_resources(Point.pack_resource(source)) ==
    (Point.pack_resource(PartitionProofs.first(~Point.SharedResource,split_models(source,copies))),
      Point.pack_resource(PartitionProofs.second(~Point.SharedResource,split_models(source,copies)))) : Point.SplitResources()}:
  Point.SharedCopies{+left,+right,exact,left_content,right_content} = copies
  Equal.sym(Point.SplitResources(),(Point.pack_resource(left),Point.pack_resource(right)),Products.shared_resources(Point.pack_resource(source)),exact)

def resources_match(+source: Point.SharedResource) ->
  {Products.shared_resources(Point.pack_resource(source)) ==
    (Point.pack_resource(PartitionProofs.first(~Point.SharedResource,split_resources(source))),
      Point.pack_resource(PartitionProofs.second(~Point.SharedResource,split_resources(source)))) : Point.SplitResources()}:
  split_observed(source,Point.shared_copies(source))

def SharedRegions(+work: Products.Shape,+first: U32,+count: U32,+model: TaskModel()) -> Data:
  {Point.task_regions(pack_task(model)) == Point.clipped_rectangle(Point.shared_geometry(work,first,count)) : Pieces.Ranges}

def SharedLeaf(+work: Products.Shape,+holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+resource: Point.SharedResource) -> Data:
  Grouped.TaskFact<Tile.Tile8,Pieces.Pieces(~Tile.Tile8),Point.Resources(),Products.Shape,Model.Model<Tile.Tile8>,Point.SharedResource,
    Products.shared_resources,Products.shared_leaf,Model.pack(~Tile.Tile8),Point.pack_resource,split_resources,Products.Shape,SharedRegions,
    work,work,holding,first,count,resource>

type ObservedShared<-native: Point.TaskOutput() & Point.Resources()> is Data:
  ObservedShared{model: TaskModel(),rest: Point.SharedResource,exact: {(pack_task(model),Point.pack_resource(rest)) == native : Point.TaskOutput() & Point.Resources()}}

def observed_shared(-task: Point.TaskOutput(),-input: Products.Cells(),-bias: Products.Cells(),observed: ObservedTask<task>,
  seen_input: Buffer.Observation<Tile.Tile8,input>,seen_bias: Buffer.Observation<Tile.Tile8,bias>) -> ObservedShared<(task,(input,bias))>:
  ObservedTask{+model,task_same} = observed
  Buffer.Observation{+input_model,input_same} = seen_input
  Buffer.Observation{+bias_model,bias_same} = seen_bias
  ObservedShared{model,Point.SharedResource{input_model,bias_model},
    %task_same : {(pack_task(model),(Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,bias_model))) == (_,(input,bias)) : Point.TaskOutput() & Point.Resources()}
    %input_same : {(pack_task(model),(Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,bias_model))) == (pack_task(model),(_,bias)) : Point.TaskOutput() & Point.Resources()}
    %bias_same : {(pack_task(model),(Buffer.pack(~Tile.Tile8,input_model),Buffer.pack(~Tile.Tile8,bias_model))) ==
      (pack_task(model),(Buffer.pack(~Tile.Tile8,input_model),_)) : Point.TaskOutput() & Point.Resources()}
    {==}}

def observe_shared(result: Point.TaskOutput() & Point.Resources()) -> ObservedShared<result>:
  (task,(input,bias)) = result
  observed_shared(task,input,bias,observe_task(task),Buffer.observe(~Tile.Tile8,input),Buffer.observe(~Tile.Tile8,bias))

def shared_fact(+work: Products.Shape,+holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+resource: Point.SharedResource,
  seen: ObservedShared<Products.shared_leaf(work,first,count,Held.pack(~Tile.Tile8,holding),Point.pack_resource(resource))>) ->
  SharedLeaf(work,holding,first,count,resource):
  ObservedShared{+model,+rest,+exact} = seen
  -native = Products.shared_leaf(work,first,count,Held.pack(~Tile.Tile8,holding),Point.pack_resource(resource))
  -packed = {(pack_task(model),Point.pack_resource(rest)) : Point.TaskOutput() & Point.Resources()}
  Grouped.TaskFact{model,rest,Equal.sym(Point.TaskOutput() & Point.Resources(),packed,native,exact),
    %Equal.sym(Point.TaskOutput() & Point.Resources(),packed,native,exact) :
      {Point.result_regions(~Point.Resources(),_) == Point.clipped_rectangle(Point.shared_geometry(work,first,count)) : Pieces.Ranges}
    Point.shared_leaf_layout(work,first,count,holding,resource)}

def shared_leaf(+work: Products.Shape,+holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+resource: Point.SharedResource) ->
  SharedLeaf(work,holding,first,count,resource):
  shared_fact(work,holding,first,count,resource,observe_shared(Products.shared_leaf(work,first,count,Held.pack(~Tile.Tile8,holding),Point.pack_resource(resource))))

def SharedTree(+work: Products.Shape,+tasks: Pieces.Ranges,+windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource) -> Data:
  Grouped.TaskFacts(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Model.Model<Tile.Tile8>,~Point.SharedResource,
    ~Products.shared_resources,~Products.shared_leaf,~Model.pack(~Tile.Tile8),~Point.pack_resource,~split_resources,~Products.Shape,~SharedRegions,
    work,work,tasks,windows,resource)

def shared_tree(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+work: Products.Shape,+total: Nat,
  tasks: Pieces.Ranges,windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  reaches: Windows.WindowsOk(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,source,work,total,tasks,windows)) ->
  SharedTree(work,tasks,windows,resource):
  match tasks windows:
    case Pieces.Range{first,count} Windows.WindowLeaf{holding,task_first,task_count}: shared_leaf(work,holding,task_first,task_count,resource)
    case Pieces.Split{+left,+right} Windows.WindowFork{+left_windows,+right_windows}:
      Held.Both{left_reaches,right_reaches} = reaches
      +split = split_resources(resource)
      Held.Both{shared_tree(~Source,~value,source,work,total,left,left_windows,PartitionProofs.first(~Point.SharedResource,split),left_reaches),
        shared_tree(~Source,~value,source,work,total,right,right_windows,PartitionProofs.second(~Point.SharedResource,split),right_reaches)}
    case Pieces.Range{first,count} Windows.WindowFork{left_windows,right_windows}: reaches
    case Pieces.Split{left,right} Windows.WindowLeaf{holding,task_first,task_count}: reaches

def shared_results(+work: Products.Shape,+tasks: Pieces.Ranges,+windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  facts: SharedTree(work,tasks,windows,resource)) -> TaskModel():
  Grouped.results(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Model.Model<Tile.Tile8>,~Point.SharedResource,
    ~Products.shared_resources,~Products.shared_leaf,~Model.pack(~Tile.Tile8),~Point.pack_resource,~split_resources,~Products.Shape,~SharedRegions,
    work,work,tasks,windows,resource,facts)

def grouped_shared(+work: Products.Shape,+tasks: Pieces.Ranges,+parallel: Bool,+windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  +facts: SharedTree(work,tasks,windows,resource)) ->
  {Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Products.shared_resources,~Products.shared_leaf,
    tasks,parallel,work,Windows.pack_windows(~Tile.Tile8,windows),Point.pack_resource(resource)) ==
    pack_task(shared_results(work,tasks,windows,resource,facts)) : Point.TaskOutput()}:
  Grouped.grouped(~Tile.Tile8,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Model.Model<Tile.Tile8>,~Point.SharedResource,
    ~Products.shared_resources,~Products.shared_leaf,~Model.pack(~Tile.Tile8),~Point.pack_resource,~split_resources,~Products.Shape,~SharedRegions,~resources_match,
    tasks,parallel,work,work,windows,resource,facts)

def clipped_region(+grid: Grid.Grid,+first: U32,+count: U32) -> Pieces.Ranges:
  +piece = Grid.piece_of(grid,first)
  +end = (first + count : U32)
  Clipped.clipped_layout(Grid.later_pieces(grid,first,count),grid,piece,first,end,Pipeline.clipped_first(grid,piece,first),Pipeline.clipped_count(grid,piece,first,end))

def Laid(+target: Pieces.Ranges,+region: Pieces.Ranges,+first: Nat,+last: Nat) -> Data:
  Held.Both<Placed.LaidParts(target,region,first),{Placed.ranges_end(region,first) == last : Nat}>

def clipped_laid(+limits: Machine.Limits,+grid: Grid.Grid,+grid_facts: GridLaws.GridFacts<limits,grid>,+unit: Nat,+bound: Nat,
  +run: Runs.Run(Pieces.ranges_of_grid(grid),unit,bound),+first: U32,+count: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),GridLaws.grid_cells(grid)) == True{} : Bool}) ->
  Laid(Pieces.ranges_of_grid(grid),clipped_region(grid,first,count),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))):
  GridLaws.GridFacts{+cells,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+grid_exact,+counted} = grid_facts
  +literal = {Grid.Grid{cells,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +moved = Equal.sym(Grid.Grid,grid,literal,grid_exact)
  %moved : Laid(Pieces.ranges_of_grid(_),clipped_region(_,first,count),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)))
  Held.first_of(Laid(Pieces.ranges_of_grid(literal),clipped_region(literal,first,count),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))),
    {Nat.is_lt(Nat.add(Grid.later_pieces(literal,first,count),U32.to_nat(Grid.piece_of(literal,first))),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
    Clipped.task_laid(limits,outer,extent,unit_cells,block,per,cells,prefix,shape,counted,unit,bound,
      %grid_exact : Runs.Run(Pieces.ranges_of_grid(_),unit,bound) run,
      first,count,positive,%grid_exact : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),GridLaws.grid_cells(_)) == True{} : Bool} inside))

def shared_leaf_laid(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+grid_facts: GridLaws.GridFacts<limits,grid>,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +grid_cells: {GridLaws.grid_cells(grid) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)) : Nat},
  +bound_room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +run: Runs.Run(Pieces.ranges_of_grid(grid),U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))),
  +first: U32,+count: U32,+fits: Runs.Fits<U32.to_nat(rows),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))>) ->
  Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))):
  Runs.Fits{+low,+high,+low_exact,+high_exact,+less,+inside} = fits
  +start = U32.to_nat(first)
  +length = U32.to_nat(count)
  +end = Nat.add(start,length)
  +bound = Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))
  +interval = group_interval(rows,channels,Machine.half_word(limits),rows_pos,Limits.double_bound(limits,U32.to_nat(rows),rows_room),Limits.half_word_fits(limits),
    first,count,Runs.Fits{low,high,low_exact,high_exact,less,inside})
  +count_room = Order.transitive(length,end,U32.to_nat(Machine.capacity(limits)),Order.right_le(start,length),Order.transitive(end,bound,U32.to_nat(Machine.capacity(limits)),inside,bound_room))
  +positive = {%Order.subtract_shift(start,length) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_difference(end,start,less) : {Nat.is_lt(0n,length) == True{} : Bool}}
  %Equal.sym(U32,(rows * Gemm.blocks(clipped_extent(rows,channels,first,count)) : U32),count,
    clipped_count(limits,rows,channels,first,count,channel_room,count_room,interval)) :
    Laid(Pieces.ranges_of_grid(grid),clipped_region(grid,first,_),start,end)
  clipped_laid(limits,grid,grid_facts,U32.to_nat(rows),bound,run,first,count,positive,
    %Equal.sym(Nat,GridLaws.grid_cells(grid),bound,grid_cells) : {Nat.is_le(end,_) == True{} : Bool} inside)

def joined_laid(+target: Pieces.Ranges,+left: Pieces.Ranges,+right: Pieces.Ranges,+first: Nat,+end_left: Nat,+start_right: Nat,+last: Nat,
  +glue: {end_left == start_right : Nat},+left_laid: Laid(target,left,first,end_left),+right_laid: Laid(target,right,start_right,last)) ->
  Laid(target,Pieces.Split{left,right},first,last):
  Held.Both{+left_parts,+left_end} = left_laid
  Held.Both{+right_parts,+right_end} = right_laid
  +end_of_left = Placed.ranges_end(left,first)
  +moved = Equal.sym(Nat,end_of_left,start_right,Equal.trans(Nat,end_of_left,end_left,start_right,left_end,glue))
  Held.Both{Gemm.Both{left_parts,%moved : Placed.LaidParts(target,right,_) right_parts},
    %moved : {Placed.ranges_end(right,_) == last : Nat} right_end}

def observed_leaf_laid(+work: Products.Shape,+grid: Grid.Grid,+first: U32,+count: U32,+window_first: U32,+window_count: U32,
  +first_same: {window_first == first : U32},+count_same: {window_count == count : U32},+model: TaskModel(),
  +regions: SharedRegions(work,window_first,window_count,model),
  +clipped: Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(Point.shared_geometry(work,first,count)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)))) ->
  Laid(Pieces.ranges_of_grid(grid),Point.task_regions(pack_task(model)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))):
  %Equal.sym(Pieces.Ranges,Point.task_regions(pack_task(model)),Point.clipped_rectangle(Point.shared_geometry(work,window_first,window_count)),regions) :
    Laid(Pieces.ranges_of_grid(grid),_,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)))
  %Equal.sym(U32,window_first,first,first_same) :
    Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(Point.shared_geometry(work,_,window_count)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)))
  %Equal.sym(U32,window_count,count,count_same) :
    Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(Point.shared_geometry(work,first,_)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)))
  clipped

# Every fragment written by the actual shared task tree is in its output slot.
def shared_tree_laid(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,
  +grid: Grid.Grid,+grid_facts: GridLaws.GridFacts<limits,grid>,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells: {GridLaws.grid_cells(grid) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(grid),U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))),
  tasks: Pieces.Ranges,windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  +run: Runs.Run(tasks,U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))),
  ok: Windows.WindowsOk(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,source,Products.Shape{rows,inner,channels,grid},total,tasks,windows),
  facts: SharedTree(Products.Shape{rows,inner,channels,grid},tasks,windows,resource)) ->
  Laid(Pieces.ranges_of_grid(grid),Point.task_regions(pack_task(shared_results(Products.Shape{rows,inner,channels,grid},tasks,windows,resource,facts))),
    Runs.range_low(tasks),Runs.range_high(tasks)):
  match tasks windows:
    case Pieces.Range{+first,+count} Windows.WindowLeaf{+holding,+window_first,+window_count}:
      Held.Both{+first_same,rest} = ok
      Held.Both{+count_same,window_ok} = rest
      Grouped.TaskFact{+model,+kept,+exact,+regions} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      observed_leaf_laid(work,grid,first,count,window_first,window_count,first_same,count_same,model,regions,
        shared_leaf_laid(limits,rows,inner,channels,grid,grid_facts,channel_room,rows_pos,rows_room,cells,room,output_run,first,count,run))
    case Pieces.Split{+left,+right} Windows.WindowFork{+left_windows,+right_windows}:
      Held.Both{+left_run,rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{left_ok,right_ok} = ok
      Held.Both{+left_facts,+right_facts} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      +split = split_resources(resource)
      +left_resource = PartitionProofs.first(~Point.SharedResource,split)
      +right_resource = PartitionProofs.second(~Point.SharedResource,split)
      joined_laid(Pieces.ranges_of_grid(grid),
        Point.task_regions(pack_task(shared_results(work,left,left_windows,left_resource,left_facts))),
        Point.task_regions(pack_task(shared_results(work,right,right_windows,right_resource,right_facts))),
        Runs.range_low(left),Runs.range_high(left),Runs.range_low(right),Runs.range_high(right),glue,
        shared_tree_laid(~Source,~value,source,total,limits,rows,inner,channels,grid,grid_facts,channel_room,rows_pos,rows_room,cells,room,output_run,
          left,left_windows,left_resource,left_run,left_ok,left_facts),
        shared_tree_laid(~Source,~value,source,total,limits,rows,inner,channels,grid,grid_facts,channel_room,rows_pos,rows_room,cells,room,output_run,
          right,right_windows,right_resource,right_run,right_ok,right_facts))
    case Pieces.Range{+first,+count} Windows.WindowFork{+left_windows,+right_windows}:
      Empty.absurd(Laid(Pieces.ranges_of_grid(grid),
        Point.task_regions(pack_task(shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Range{first,count},Windows.WindowFork{left_windows,right_windows},resource,facts))),
        U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))),ok)
    case Pieces.Split{+left,+right} Windows.WindowLeaf{+holding,+window_first,+window_count}:
      Empty.absurd(Laid(Pieces.ranges_of_grid(grid),
        Point.task_regions(pack_task(shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Split{left,right},Windows.WindowLeaf{holding,window_first,window_count},resource,facts))),
        Runs.range_low(left),Runs.range_high(right)),ok)

def work_grid(+work: Products.Shape) -> Grid.Grid:
  Products.Shape{positions,inner,channels,grid} = work
  grid

def shared_native(+work: Products.Shape,+weight_grid: Grid.Grid,+tasks: Pieces.Ranges,+parallel: Bool,+resource: Point.SharedResource,
  weights: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  Point.SharedResource{input,bias} = resource
  Products.shared_tasks(work,weight_grid,Tasks.Decision{parallel,tasks},Buffer.pack(~Tile.Tile8,input),weights,Buffer.pack(~Tile.Tile8,bias))

type ScheduledShared<-Source: Data,-value: Source -> U32 -> Tile.Tile8,-source: Source,-total: Nat,
  -work: Products.Shape,-tasks: Pieces.Ranges,-resource: Point.SharedResource,-grid: Grid.Grid,
  -native: Pieces.Pieces(~Tile.Tile8)> is Data:
  ScheduledShared{windows: Windows.WindowModel<Tile.Tile8>,
    reaches: Windows.WindowsOk(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,source,work,total,tasks,windows),
    facts: SharedTree(work,tasks,windows,resource),
    placed: {native == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),pack_task(shared_results(work,tasks,windows,resource,facts))) :
      Pieces.Pieces(~Tile.Tile8)}}

# The shared execution places the real task tree observed from its windows.
def shared_scheduled(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+work: Products.Shape,+weight_grid: Grid.Grid,
  +tasks: Pieces.Ranges,+parallel: Bool,+resource: Point.SharedResource,-weights: Pieces.Pieces(~Tile.Tile8),
  +windowed: Windows.Windowed<Tile.Tile8,Source,value,source,Products.Shape,Products.shared_reach,work,total,tasks,
    Halo.windows(~Tile.Tile8,~Products.Shape,~Products.shared_reach,work,weight_grid,Products.zero(),tasks,weights)>) ->
  ScheduledShared<Source,value,source,total,work,tasks,resource,work_grid(work),shared_native(work,weight_grid,tasks,parallel,resource,weights)>:
  Products.Shape{+positions,+inner,+channels,+grid} = work
  Point.SharedResource{+input,+bias} = resource
  Windows.Windowed{+windows,+packed,+reaches} = windowed
  +facts = shared_tree(~Source,~value,source,work,total,tasks,windows,resource,reaches)
  -results = pack_task(shared_results(work,tasks,windows,resource,facts))
  -halo = Halo.windows(~Tile.Tile8,~Products.Shape,~Products.shared_reach,work,weight_grid,Products.zero(),tasks,weights)
  -by_windows = Windows.pack_windows(~Tile.Tile8,windows)
  ScheduledShared{windows,reaches,facts,
    %Equal.sym(Partition.Tree<&1,Move.Holding<Tile.Tile8>,Unit>,halo,by_windows,packed) :
      {Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),
        Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Products.shared_resources,~Products.shared_leaf,tasks,parallel,work,_,Point.pack_resource(resource))) ==
        Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),results) : Pieces.Pieces(~Tile.Tile8)}
    %Equal.sym(Point.TaskOutput(),Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Point.Resources(),~Products.Shape,~Products.shared_resources,~Products.shared_leaf,
      tasks,parallel,work,by_windows,Point.pack_resource(resource)),results,grouped_shared(work,tasks,parallel,windows,resource,facts)) :
      {Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),_) == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),results) : Pieces.Pieces(~Tile.Tile8)}
    {==}}

type SharedCoverage<-Source: Data,-value: Source -> U32 -> Tile.Tile8,-source: Source,-total: Nat,
  -work: Products.Shape,-tasks: Pieces.Ranges,-grid: Grid.Grid,-native: Pieces.Pieces(~Tile.Tile8)> is Data:
  SharedCoverage{windows: Windows.WindowModel<Tile.Tile8>,
    reads: Windows.WindowsOk(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,source,work,total,tasks,windows),
    model: TaskModel(),placed: {native == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),pack_task(model)) : Pieces.Pieces(~Tile.Tile8)},
    writes: Laid(Pieces.ranges_of_grid(grid),Point.task_regions(pack_task(model)),Runs.range_low(Pieces.ranges_of_grid(grid)),Runs.range_high(Pieces.ranges_of_grid(grid)))}

def shared_coverage(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,
  +grid: Grid.Grid,+grid_facts: GridLaws.GridFacts<limits,grid>,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells: {GridLaws.grid_cells(grid) == Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(grid),U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))),
  +tasks: Pieces.Ranges,+resource: Point.SharedResource,-native: Pieces.Pieces(~Tile.Tile8),
  +plan: Runs.RunOf(tasks,Pieces.ranges_of_grid(grid),U32.to_nat(rows),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))),
  +scheduled: ScheduledShared<Source,value,source,total,Products.Shape{rows,inner,channels,grid},tasks,resource,grid,native>) ->
  SharedCoverage<Source,value,source,total,Products.Shape{rows,inner,channels,grid},tasks,grid,native>:
  Held.Both{+run,+ends} = plan
  Held.Both{+low_same,+high_same} = ends
  ScheduledShared{+windows,+reaches,+facts,+placed} = scheduled
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +model = shared_results(work,tasks,windows,resource,facts)
  SharedCoverage{windows,reaches,model,placed,
    %low_same : Laid(Pieces.ranges_of_grid(grid),Point.task_regions(pack_task(model)),_,Runs.range_high(Pieces.ranges_of_grid(grid)))
    %high_same : Laid(Pieces.ranges_of_grid(grid),Point.task_regions(pack_task(model)),Runs.range_low(tasks),_)
    shared_tree_laid(~Source,~value,source,total,limits,rows,inner,channels,grid,grid_facts,channel_room,rows_pos,rows_room,cells,room,output_run,tasks,windows,resource,run,reaches,facts)}

def shared_execution(+work: Products.Shape,+weight_grid: Grid.Grid,decision: Tasks.Decision,+resource: Point.SharedResource,
  weights: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  Tasks.Decision{+parallel,+tasks} = decision
  shared_native(work,weight_grid,tasks,parallel,resource,weights)

def executed(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+weight_grid: Grid.Grid,decision: Tasks.Decision,
  +input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,-weights: Pieces.Pieces(~Tile.Tile8)) ->
  {Products.shared_tasks(Products.Shape{rows,inner,channels,grid},weight_grid,decision,Buffer.pack(~Tile.Tile8,input),weights,Buffer.pack(~Tile.Tile8,bias)) ==
    shared_execution(Products.Shape{rows,inner,channels,grid},weight_grid,decision,Point.SharedResource{input,bias},weights) : Pieces.Pieces(~Tile.Tile8)}:
  Tasks.Decision{+parallel,+tasks} = decision
  {==}

def scheduled_decision(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+work: Products.Shape,+weight_grid: Grid.Grid,
  decision: Tasks.Decision,+resource: Point.SharedResource,-weights: Pieces.Pieces(~Tile.Tile8),
  +windowed: Windows.Windowed<Tile.Tile8,Source,value,source,Products.Shape,Products.shared_reach,work,total,Runs.decision_tasks(decision),
    Halo.windows(~Tile.Tile8,~Products.Shape,~Products.shared_reach,work,weight_grid,Products.zero(),Runs.decision_tasks(decision),weights)>) ->
  ScheduledShared<Source,value,source,total,work,Runs.decision_tasks(decision),resource,work_grid(work),shared_execution(work,weight_grid,decision,resource,weights)>:
  Tasks.Decision{+parallel,+tasks} = decision
  shared_scheduled(~Source,~value,source,total,work,weight_grid,tasks,parallel,resource,weights,windowed)

def actual_shared_coverage(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,
  +rows: U32,+inner: U32,+channels: U32,+workers: U32,+resource: Point.SharedResource,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +weights_room: {Nat.is_le(ShapeLaws.elements(weight_shape(inner,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(weight_shape(inner,channels))) == True{} : Bool},weights: Pieces.Pieces(~Tile.Tile8),
  +covered: Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,weights),0n,ShapeLaws.elements(weight_shape(inner,channels)),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(weight_shape(inner,channels))),Held.Ended{})>) ->
  SharedCoverage<Source,value,source,ShapeLaws.elements(weight_shape(inner,channels)),Products.Shape{rows,inner,channels,shared_grid(rows,channels)},
    Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),shared_grid(rows,channels),
    shared_execution(Products.Shape{rows,inner,channels,shared_grid(rows,channels)},Tensors.grid_of(weight_shape(inner,channels)),shared_decision(rows,inner,channels,workers),resource,weights)>:
  +shape = {[Gemm.blocks(channels),1,rows] : List<&2,U32>}
  +grid = shared_grid(rows,channels)
  +target = Grid.piece_target()
  +unit = U32.to_nat(rows)
  +bound = Nat.mul(U32.to_nat(Gemm.blocks(channels)),unit)
  +total = ShapeLaws.elements(weight_shape(inner,channels))
  +weight_grid = Tensors.grid_of(weight_shape(inner,channels))
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +decision = shared_decision(rows,inner,channels,workers)
  +tasks = Runs.decision_tasks(decision)
  +exact = shared_cells(rows,channels)
  +plan = shared_run(limits,rows,inner,channels,workers,rows_room,out_room,out_nonempty)
  +divides = {GridRanges.Multiple{1n,Equal.sym(Nat,Nat.add(unit,0n),unit,Addition.zero_right(unit))} : GridRanges.Multiple<unit,unit>}
  +grid_facts = GridRanges.grid_facts(limits,shape,target,out_room,out_nonempty)
  +layout = GridRanges.grid_layout(limits,shape,target,out_room,out_nonempty,unit,divides)
  +run = {%exact : Runs.Run(Pieces.ranges_of_grid(grid),unit,_) GridRanges.layout_run(shape,target,unit,layout) : Runs.Run(Pieces.ranges_of_grid(grid),unit,bound)}
  +reads = actual_shared_windows(~Source,~value,source,limits,rows,inner,channels,workers,rows_pos,rows_room,out_room,out_nonempty,weights_room,weights_nonempty,weights,covered)
  shared_coverage(~Source,~value,source,total,limits,rows,inner,channels,grid,grid_facts,channel_room,rows_pos,rows_room,
    Equal.trans(Nat,GridLaws.grid_cells(grid),ShapeLaws.elements(shape),bound,GridRanges.grid_count(limits,shape,target,out_room,out_nonempty),exact),
    %exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} out_room,
    run,tasks,resource,shared_execution(work,weight_grid,decision,resource,weights),plan,
    scheduled_decision(~Source,~value,source,total,work,weight_grid,decision,resource,weights,reads))

# The whole library entry places the exact shared task tree.
def shared_product_coverage(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,
  +rows: U32,+inner: U32,+channels: U32,+workers: U32,+input: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,-left: Pieces.Pieces(~Tile.Tile8),
  +left_exact: {Tensors.joined(~Tile.Tile8,(rows * Gemm.blocks(inner) : U32),Products.zero(),left) == Buffer.pack(~Tile.Tile8,input) : Products.Cells()},
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +weights_room: {Nat.is_le(ShapeLaws.elements(weight_shape(inner,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(weight_shape(inner,channels))) == True{} : Bool},weights: Pieces.Pieces(~Tile.Tile8),
  +covered: Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,weights),0n,ShapeLaws.elements(weight_shape(inner,channels)),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(weight_shape(inner,channels))),Held.Ended{})>) ->
  SharedCoverage<Source,value,source,ShapeLaws.elements(weight_shape(inner,channels)),Products.Shape{rows,inner,channels,shared_grid(rows,channels)},
    Runs.decision_tasks(shared_decision(rows,inner,channels,workers)),shared_grid(rows,channels),
    Products.shared(rows,inner,channels,workers,left,weights,Buffer.pack(~Tile.Tile8,bias))>:
  +grid = shared_grid(rows,channels)
  +total = ShapeLaws.elements(weight_shape(inner,channels))
  +weight_grid = Tensors.grid_of(weight_shape(inner,channels))
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +decision = shared_decision(rows,inner,channels,workers)
  +tasks = Runs.decision_tasks(decision)
  -joined = Tensors.joined(~Tile.Tile8,(rows * Gemm.blocks(inner) : U32),Products.zero(),left)
  -input_owner = Buffer.pack(~Tile.Tile8,input)
  -bias_owner = Buffer.pack(~Tile.Tile8,bias)
  %Equal.sym(Products.Cells(),joined,input_owner,left_exact) :
    SharedCoverage<Source,value,source,total,work,tasks,grid,Products.shared_tasks(work,weight_grid,decision,_,weights,bias_owner)>
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),Products.shared_tasks(work,weight_grid,decision,input_owner,weights,bias_owner),
    shared_execution(work,weight_grid,decision,Point.SharedResource{input,bias},weights),executed(rows,inner,channels,grid,weight_grid,decision,input,bias,weights)) :
    SharedCoverage<Source,value,source,total,work,tasks,grid,_>
  actual_shared_coverage(~Source,~value,source,limits,rows,inner,channels,workers,Point.SharedResource{input,bias},channel_room,rows_pos,rows_room,
    out_room,out_nonempty,weights_room,weights_nonempty,weights,covered)
