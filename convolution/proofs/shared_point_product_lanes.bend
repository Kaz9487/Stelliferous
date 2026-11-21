# The actual operands of a shared rectangle refine the public ordered sum.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_point_products.bend as Products
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/kernel_gemm_point_dot.bend as Dot
import ../../lib/proofs/kernel_gemm_point_jobs.bend as Jobs
import ../../lib/proofs/kernel_gemm_point_taps.bend as Taps
import ../../lib/proofs/kernel_gemm_output_groups.bend as Groups
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_halo_windows.bend as Halo
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_point_product_evaluation.bend as Evaluate
import ../../lib/proofs/tensor_point_product_operands.bend as Operands
import ../../lib/proofs/tensor_point_product_pairs.bend as Pairs
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_lanes.bend as Lanes
import ../../lib/proofs/tensor_point_product_tasks.bend as Shared
import ../../lib/proofs/tensor_point_product_values.bend as Values
import ../../lib/proofs/tensor_point_product_windows.bend as Windows
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/tensor_shared_product_samples.bend as Span
import ../../lib/proofs/tensor_shared_product_placement.bend as Placement
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ./point_product_values.bend as PublicValues
import ./product_values.bend as Logical

type WholeOperands<-Left: Data,-left_value: Left -> U32 -> F32,-Bias: Data,-bias_value: Bias -> U32 -> F32,-left: Left,-bias: Bias,
  -rows: U32,-inner: U32,-channels: U32> is Data:
  WholeOperands{input: Buffer.Model<Tile.Tile8>,offset: Buffer.Model<Tile.Tile8>,
    input_length: {Buffer.length(~Tile.Tile8,input) == Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))) : Nat},
    input_values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},Buffer.content(~Tile.Tile8,input)),
    offset_length: {Buffer.length(~Tile.Tile8,offset) == U32.to_nat(Gemm.blocks(channels)) : Nat},
    offset_values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},Buffer.content(~Tile.Tile8,offset))}

type LaneOperands<-Left: Data,-left_value: Left -> U32 -> F32,-Weight: Data,-weight_value: Weight -> U32 -> F32,-Bias: Data,-bias_value: Bias -> U32 -> F32> is Data:
  LaneOperands{limits: Machine.Limits,left: Left,weight: Weight,bias: Bias,rows: U32,inner: U32,channels: U32,grid: Grid.Grid,
    first: U32,count: U32,holding: Held.Holding<Tile.Tile8>,whole: WholeOperands<Left,left_value,Bias,bias_value,left,bias,rows,inner,channels>,
    span: Span.Span<rows,inner,channels,first,count>,clip: Shared.Clipped<limits,rows,channels,first,count>,
    window: Halo.TaskWindow<Tile.Tile8,Representation.Lifted<Weight>,Representation.lifted(~Weight,~weight_value),
      Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},holding,Span.start_word(rows,inner,first),
      (Span.start_word(rows,inner,first) + Span.span_word(rows,inner,count) : U32),Span.weight_cells(inner,channels)>,
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weight_room: {Nat.is_le(Span.weight_cells(inner,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    bias_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    input_groups: TiledValues.Groups<limits,inner>,
    position: Nat,group: Nat,lane: Nat,
    position_inside: {Nat.is_lt(position,U32.to_nat(rows)) == True{} : Bool},
    group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Shared.clipped_extent(rows,channels,first,count)))) == True{} : Bool},
    lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
    active: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.add(Span.start_group(rows,first),group),8n)),U32.to_nat(channels)) == True{} : Bool}}

def selected_group(+first: Nat,+groups: Nat,+group: Nat,+total: Nat,
  +inside: {Nat.is_lt(group,groups) == True{} : Bool},+covered: {Nat.is_le(Nat.add(first,groups),total) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(first,group),total) == True{} : Bool}:
  Order.lt_le(Nat.add(first,group),Nat.add(first,groups),total,Order.offset_lt(first,group,groups,inside),covered)

def global_channel(+rows: U32,+first: U32,+group: Nat,+lane: Nat) -> Nat:
  Nat.add(lane,Nat.mul(Nat.add(Span.start_group(rows,first),group),8n))

def whole_cells(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Evaluate.Cells:
  LaneOperands{limits,left,weight,bias,+rows,+inner,+channels,+grid,+first,+count,+holding,WholeOperands{+input,+offset,input_length,input_values,offset_length,offset_values},
    span,clip,window,input_room,weight_room,bias_room,input_groups,position,group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  +local = Windows.copy_model(holding,Span.start_word(rows,inner,first),Span.span_word(rows,inner,count))
  Evaluate.Cells{rectangle,0,Buffer.content(~Tile.Tile8,input),Evaluate.SharedWeights{local},Buffer.content(~Tile.Tile8,Point.shared_bias_at(Point.SharedResource{input,offset},rows,first,count)),lane}

def tap_extent(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Nat:
  LaneOperands{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,WholeOperands{input,offset,input_length,input_values,offset_length,offset_values},span,clip,window,input_room,weight_room,bias_room,input_groups,
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  U32.to_nat(inner)

def cell_extent(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) ->
  {U32.to_nat(Values.input_inner(Evaluate.cells_rectangle(whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)))) ==
    tap_extent(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope) : Nat}:
  LaneOperands{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,WholeOperands{input,offset,input_length,input_values,offset_length,offset_values},
    span,clip,window,input_room,weight_room,bias_room,input_groups,position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  {==}

def Tap(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,+tap: Nat) -> Data:
  LaneOperands{limits,+left,+weight,bias,+rows,+inner,channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +cells = whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)
  +source = {PublicValues.Readers{left,weight,U32.to_nat(inner)} : PublicValues.Readers<Left,Weight>}
  Gemm.Both<{Evaluate.local_input(cells,position,tap) == PublicValues.input(~Left,~left_value,~Weight,source,position,tap) : F32},
    {Evaluate.local_weight(cells,group,tap) == PublicValues.weight(~Left,~Weight,~weight_value,source,global_channel(rows,first,group,lane),tap) : F32}>

def tap(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,+key: Nat,
  +inside: {Nat.is_lt(key,tap_extent(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)) == True{} : Bool}) ->
  Tap(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope,key):
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,
    WholeOperands{+input,+offset,+input_length,+input_values,offset_length,offset_values},
    Span.Span{+group_cover,+first_exact,+length_exact,+window_inside},Shared.Clipped{clipped,clipped_room,clipped_positive,+groups_exact},+window,
    +input_room,+weight_room,+bias_room,+input_groups,+position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  +lower = Span.start_group(rows,first)
  +groups = Span.group_count(rows,count)
  +local_extent = Shared.clipped_extent(rows,channels,first,count)
  +local_group = {%groups_exact : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,groups) == True{} : Bool}}
  +global_group = selected_group(lower,groups,group,U32.to_nat(Gemm.blocks(channels)),local_group,group_cover)
  +content = Buffer.content(~Tile.Tile8,input)
  +size = Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)))
  Gemm.Both{Samples.joined_input_sample(~Left,~left_value,left,limits,rows,inner,content,position,key,input_room,
      Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,content),Buffer.length(~Tile.Tile8,input),size,Buffer.content_length_of(~Tile.Tile8,input),input_length),
      input_values,input_groups,position_inside,inside),
    Samples.copied_weight_sample(~Weight,~weight_value,weight,limits,inner,channels,holding,Span.start_word(rows,inner,first),Span.span_word(rows,inner,count),
      lower,groups,first_exact,length_exact,weight_room,window_inside,window,key,group,lane,inside,local_group,global_group,lane_inside,active)}

def BiasAgrees(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  LaneOperands{limits,left,weight,+bias,+rows,inner,channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  {Evaluate.local_bias(whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope),group) ==
    bias_value(bias,U32.from_nat(global_channel(rows,first,group,lane))) : F32}

def bias_sample(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) ->
  BiasAgrees(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope):
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,
    WholeOperands{+input,+offset,+input_length,+input_values,+offset_length,+offset_values},
    Span.Span{+group_cover,first_exact,length_exact,window_inside},Shared.Clipped{clipped,clipped_room,clipped_positive,+groups_exact},+window,
    +input_room,+weight_room,+bias_room,+input_groups,+position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  +first_group = (first / rows : U32)
  +group_count = (count / rows : U32)
  +low = U32.to_nat(first_group)
  +length = U32.to_nat(group_count)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +inside = {%groups_exact : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,length) == True{} : Bool}}
  +extent_room = Order.transitive(length,Nat.add(low,length),U32.to_nat(Machine.capacity(limits)),Order.right_le(low,length),
    Order.transitive(Nat.add(low,length),groups,U32.to_nat(Machine.capacity(limits)),group_cover,bias_room))
  Samples.copied_bias_sample(~Bias,~bias_value,bias,limits,channels,offset,first_group,group_count,group,lane,bias_room,offset_length,offset_values,
    Limits.count_allocatable(limits,group_count,extent_room),inside,selected_group(low,length,group,groups,inside,group_cover),lane_inside,active)

def SharedDot(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  LaneOperands{limits,+left,+weight,+bias,+rows,+inner,channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +channel = global_channel(rows,first,group,lane)
  {Evaluate.local_value(whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope),position,group) ==
    (Logical.ordered_sum(~Left,~left_value,~Weight,~weight_value,U32.to_nat(inner),0n,left,Nat.mul(position,U32.to_nat(inner)),
      weight,Nat.mul(channel,U32.to_nat(inner)),0.0) + bias_value(bias,U32.from_nat(channel)) : F32) : F32}

def shared_dot(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) ->
  SharedDot(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope):
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,WholeOperands{+input,+offset_buffer,+input_len,+input_values,+offset_len,+offset_values},+span,+clip,+window,+input_room,+weight_room,+bias_room,+input_groups,
    +position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  +cells = whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)
  +source = {PublicValues.Readers{left,weight,U32.to_nat(inner)} : PublicValues.Readers<Left,Weight>}
  +channel = global_channel(rows,first,group,lane)
  +offset = bias_value(bias,U32.from_nat(channel))
  +width = U32.to_nat(inner)
  +expected = Logical.ordered_sum(~Left,~left_value,~Weight,~weight_value,width,0n,left,Nat.mul(position,width),weight,Nat.mul(channel,width),0.0)
  +dotted = Dot.logical_dot(~PublicValues.Readers<Left,Weight>,~PublicValues.input(~Left,~left_value,~Weight),~PublicValues.weight(~Left,~Weight,~weight_value),source,position,channel,width)
  Equal.trans(F32,Evaluate.local_value(cells,position,group),(dotted + offset : F32),(expected + offset : F32),
    Evaluate.local_agrees(~PublicValues.Readers<Left,Weight>,~PublicValues.input(~Left,~left_value,~Weight),~PublicValues.weight(~Left,~Weight,~weight_value),
      cells,source,position,group,position,channel,offset,
      Members.range_all(~LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,~Tap(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
        ~tap_extent(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),~tap(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),width,0n,scope,Order.reflexive(width)),
      bias_sample(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)),
    Equal.cong(F32,F32,total => (total + offset : F32),dotted,expected,
      PublicValues.ordered(~Left,~left_value,~Weight,~weight_value,width,0n,source,position,channel,0.0)))

type RunRoom<-limits: Machine.Limits,-rows: U32,-inner: U32,-extent: U32> is Data:
  RunRoom{row_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(extent))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    pairs: Operands.PairSpace<limits,extent,inner>,span: Values.Span<limits,inner>}

def ordered_value(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> F32:
  LaneOperands{limits,+left,+weight,+bias,+rows,+inner,channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +channel = global_channel(rows,first,group,lane)
  (Logical.ordered_sum(~Left,~left_value,~Weight,~weight_value,U32.to_nat(inner),0n,left,Nat.mul(position,U32.to_nat(inner)),
    weight,Nat.mul(channel,U32.to_nat(inner)),0.0) + bias_value(bias,U32.from_nat(channel)) : F32)

def OutputRoom(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  LaneOperands{+limits,left,weight,bias,+rows,inner,+channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  {Nat.is_le(Nat.mul(Nat.mul(1n,U32.to_nat(rows)),U32.to_nat(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}

def AtLogicalOutput(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,+value: F32) -> Data:
  LaneOperands{limits,+left,+weight,+bias,+rows,+inner,+channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +channel = global_channel(rows,first,group,lane)
  {value == Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},
    U32.from_nat(Nat.add(Nat.mul(position,U32.to_nat(channels)),channel))) : F32}

def shared_value(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,+value: F32,
  +computed: {value == ordered_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope) : F32},
  +room: OutputRoom(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)) ->
  AtLogicalOutput(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope,value):
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,+position_inside,group_inside,lane_inside,+active} = scope
  +channel = global_channel(rows,first,group,lane)
  +middle = ordered_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)
  +wanted = Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},
    U32.from_nat(Nat.add(Nat.mul(position,U32.to_nat(channels)),channel)))
  Equal.trans(F32,value,middle,wanted,computed,
    Equal.sym(F32,wanted,middle,Logical.shared_read(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,1n,U32.to_nat(inner),U32.to_nat(channels),U32.to_nat(rows),weight,bias,
      Machine.capacity(limits),Limits.capacity_fits(limits),room,Order.positive_bound(channel,U32.to_nat(channels),active),0n,channel,position,{==},active,position_inside)))

def width_of(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> U32:
  LaneOperands{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  inner

def NativeRunRoom(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  LaneOperands{+limits,left,weight,bias,+rows,+inner,+channels,grid,+first,+count,holding,whole,span,clip,window,input_room,weight_room,bias_room,input_groups,
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  RunRoom<limits,rows,inner,Shared.clipped_extent(rows,channels,first,count)>

def staged(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>) -> F32:
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,
    WholeOperands{+input,+offset,input_length,input_values,offset_length,offset_values},span,clip,window,input_room,weight_room,bias_room,input_groups,
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  Lanes.stored_lane(~Jobs.run,rectangle,0,Buffer.content(~Tile.Tile8,input),Point.shared_held_pairs(rectangle,Span.start_word(rows,inner,first),holding),
    Buffer.content(~Tile.Tile8,Point.shared_bias_at(Point.SharedResource{input,offset},rows,first,count)),position,group,lane)

def staged_dot(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +scope: LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>,
  +room: NativeRunRoom(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope),
  +whole: Evaluate.Whole(width_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope))) ->
  {staged(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope) == ordered_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope) : F32}:
  LaneOperands{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,
    WholeOperands{+input,+offset,+input_length,+input_values,+offset_length,+offset_values},
    Span.Span{+group_cover,+first_exact,+length_exact,+window_inside},Shared.Clipped{+clipped,+extent_room,+extent_positive,+groups_exact},+window,
    +input_room,+weight_room,+bias_room,+input_groups,+position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  RunRoom{+rows_room,+cells_room,+pairs_room,+steps} = room
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  +extent = Shared.clipped_extent(rows,channels,first,count)
  +groups_word = Arithmetic.to_nat_injective(Gemm.blocks(extent),(count / rows : U32),groups_exact)
  +base = Span.start_word(rows,inner,first)
  +span = Span.span_word(rows,inner,count)
  +local_bias = Buffer.content(~Tile.Tile8,Point.shared_bias_at(Point.SharedResource{input,offset},rows,first,count))
  +source = Buffer.content(~Tile.Tile8,input)
  +wanted = ordered_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)
  +cells = whole_cells(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope)
  +copied = Pairs.held_pairs_copied(~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),
    Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},limits,first,rows,inner,extent,True{},grid,base,holding,span,Span.weight_cells(inner,channels),
    %Equal.sym(U32,Gemm.blocks(extent),(count / rows : U32),groups_word) : {U32.to_nat(span) == Nat.mul(U32.to_nat(_),U32.to_nat(inner)) : Nat} length_exact,
    weight_room,window_inside,window,pairs_room)
  %Equal.sym(Buffer.Content<Tile.Tile16>,Point.shared_held_pairs(rectangle,base,holding),Point.shared_pair_model(rectangle,Windows.copy_model(holding,base,span)),copied) :
    {Lanes.stored_lane(~Jobs.run,rectangle,0,source,_,local_bias,position,group,lane) == wanted : F32}
  Equal.trans(F32,Evaluate.stored_lane(~Jobs.run,cells,position,group),Evaluate.local_value(cells,position,group),wanted,
    Evaluate.rectangle_lane(~Jobs.run,~Jobs.job_accumulators_with(~Gemm.row_times,~Jobs.whole_inner),~Values.whole_frame,~Evaluate.Whole,~Evaluate.whole_order,
      cells,limits,position,group,pairs_room,whole,extent_room,extent_positive,rows_room,cells_room,steps,position_inside,group_inside,lane_inside),
    shared_dot(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,scope))
