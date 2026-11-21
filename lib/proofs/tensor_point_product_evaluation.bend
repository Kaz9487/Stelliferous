# Ordered evaluation of a rectangle from its actual operand cells.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../ordered_matrix.bend as Matrix
import ../tensor_point_products.bend as Products
import ./kernel_gemm_output_groups.bend as Groups
import ./kernel_gemm_point_dot.bend as Dot
import ./kernel_gemm_point_jobs.bend as Jobs
import ./kernel_gemm_point_readback.bend as Readback
import ./kernel_gemm_point_refinement.bend as PointLaws
import ./kernel_gemm_point_spec.bend as Spec
import ./kernel_gemm_point_taps.bend as Taps
import ./kernel_tile_refinement.bend as TileLaws
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./ordered_ranges.bend as Ranges
import ./ordered_reduction_split.bend as Ordered
import ./sequence_membership.bend as Members
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_point_product_lanes.bend as Lanes
import ./tensor_point_product_operands.bend as Operands
import ./tensor_point_product_values.bend as Values
import ./tensor_point_products.bend as Point

def KernelOrder(~accumulate: Jobs.Accumulate(),~Ready: U32 -> Data) -> Type:
  @+limits: Machine.Limits -> @+target: Readback.Target -> @+pair: Nat -> @+unit: Nat -> @+row: Nat -> @+lane: Nat ->
    @+ready: Ready(Taps.inner_of(Readback.target_matrix(target))) -> Taps.OrderedLane<accumulate,target,pair,unit,row,lane>

def Whole(inner: U32) -> Data:
  {U32.is_zero((inner % 16 : U32)) == True{} : Bool}

def Tail(inner: U32) -> Data:
  Unit

def remainder_of(target: Readback.Target) ->
  {Gemm.point_remainder(Readback.target_matrix(target)) == (Taps.inner_of(Readback.target_matrix(target)) % 16 : U32) : U32}:
  Readback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,first,end},base,tiles},readers} = target
  {==}

def whole_order(+limits: Machine.Limits,+target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,
  +whole: Whole(Taps.inner_of(Readback.target_matrix(target)))) ->
  Taps.OrderedLane<Jobs.job_accumulators_with(~Gemm.row_times,~Jobs.whole_inner),target,pair,unit,row,lane>:
  Taps.ordered_lane(~Jobs.job_accumulators_with(~Gemm.row_times,~Jobs.whole_inner),~Taps.job_whole_taps,~Taps.whole_dot,
    target,pair,unit,row,lane,Taps.whole_order(limits,Readback.target_matrix(target),pair,unit,Taps.input_of(target),Taps.pairs_of(target),
      %Equal.sym(U32,Gemm.point_remainder(Readback.target_matrix(target)),(Taps.inner_of(Readback.target_matrix(target)) % 16 : U32),remainder_of(target)) :
        {U32.is_zero(_) == True{} : Bool} whole))

def tail_order(+limits: Machine.Limits,+target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,+ready: Unit) ->
  Taps.OrderedLane<Jobs.job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced),target,pair,unit,row,lane>:
  Taps.ordered_lane(~Jobs.job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced),~Taps.job_tail_taps,~Taps.tail_dot,
    target,pair,unit,row,lane,Taps.tail_order(limits,Readback.target_matrix(target),pair,unit,Taps.input_of(target),Taps.pairs_of(target)))

type Query is Data:
  Query{rectangle: Products.Rectangle,base: U32,first: U32,span: U32,readers: Jobs.Readers,position: Nat,group: Nat,lane: Nat}

def query_target(query: Query) -> Readback.Target:
  Query{+rectangle,+base,+first,+span,+readers,position,group,lane} = query
  Readback.Target{Values.matrix_at(rectangle,base,first,span),readers}

def query_position(query: Query) -> Nat:
  Query{rectangle,base,first,span,readers,position,group,lane} = query
  position

def query_group(query: Query) -> Nat:
  Query{rectangle,base,first,span,readers,position,group,lane} = query
  group

def query_pair(query: Query) -> Nat:
  Lanes.pair_of(query_group(query))

def query_lane(query: Query) -> Nat:
  Query{rectangle,base,first,span,readers,position,+group,+lane} = query
  Lanes.selected_lane(Lanes.upper(group),lane)

def query_unit(query: Query) -> Nat:
  Query{rectangle,base,+first,span,readers,+position,group,lane} = query
  Operands.unit_of(first,position)

def query_row(query: Query) -> Nat:
  Query{rectangle,base,+first,span,readers,+position,group,lane} = query
  Operands.row_of(first,position)

def query_width(query: Query) -> Nat:
  U32.to_nat(Taps.inner_of(Readback.target_matrix(query_target(query))))

def input_scope(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,+query: Query,+reader: Reader,+key: Nat) -> Data:
  {Taps.physical_input(query_target(query),query_unit(query),query_row(query),key) == input(reader,query_position(query),key) : F32}

def weight_scope(~Reader: Data,~weight: Reader -> Nat -> Nat -> F32,+query: Query,+reader: Reader,+key: Nat) -> Data:
  {Taps.physical_weight(query_target(query),query_pair(query),query_lane(query),key) == weight(reader,query_group(query),key) : F32}

def tap_scope(~Context: Data,~Reader: Data,~reader_of: Context -> Reader,~query_of: Context -> Query,
  ~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,+context: Context,+key: Nat) -> Data:
  +reader = reader_of(context)
  +query = query_of(context)
  Gemm.Both<input_scope(~Reader,~input,query,reader,key),weight_scope(~Reader,~weight,query,reader,key)>

def scope_width(~Context: Data,~query_of: Context -> Query,context: Context) -> Nat:
  query_width(query_of(context))

def TapRule(~Context: Data,~Reader: Data,~reader_of: Context -> Reader,~query_of: Context -> Query,
  ~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32) -> Type:
  @+context: Context -> @+key: Nat -> @+inside: {Nat.is_lt(key,scope_width(~Context,~query_of,context)) == True{} : Bool} ->
    tap_scope(~Context,~Reader,~reader_of,~query_of,~input,~weight,context,key)

def dot_value(~Reader: Data,~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  +query: Query,+reader: Reader,+bias: F32) -> F32:
  (Dot.logical_dot(~Reader,~input,~weight,reader,query_position(query),query_group(query),query_width(query)) + bias : F32)

def evaluated(~accumulate: Jobs.Accumulate(),
  ~Context: Data,~Reader: Data,~reader_of: Context -> Reader,~query_of: Context -> Query,
  ~input: Reader -> Nat -> Nat -> F32,~weight: Reader -> Nat -> Nat -> F32,
  ~tap: TapRule(~Context,~Reader,~reader_of,~query_of,~input,~weight),
  +context: Context,+bias: F32,
  +small: {Nat.is_le(query_row(query_of(context)),3n) == True{} : Bool},
  +bias_agrees: {Taps.bias_lane(query_target(query_of(context)),query_pair(query_of(context)),query_lane(query_of(context))) == bias : F32},
  +ordered: Taps.OrderedLane<accumulate,query_target(query_of(context)),query_pair(query_of(context)),query_unit(query_of(context)),query_row(query_of(context)),query_lane(query_of(context))>) ->
  {Spec.pair_lane(query_lane(query_of(context)),
    Readback.stored_value_by(~accumulate,query_target(query_of(context)),U32.from_nat(query_pair(query_of(context))),U32.from_nat(query_unit(query_of(context))),query_row(query_of(context)))) ==
    dot_value(~Reader,~input,~weight,query_of(context),reader_of(context),bias) : F32}:
  +query = query_of(context)
  +width = query_width(query)
  Dot.logical_lane(~accumulate,~Reader,~input,~weight,query_target(query),query_pair(query),query_unit(query),query_row(query),query_lane(query),reader_of(context),
    query_position(query),query_group(query),bias,small,
    Members.range_all(~Context,~tap_scope(~Context,~Reader,~reader_of,~query_of,~input,~weight),~scope_width(~Context,~query_of),~tap,
      width,0n,context,Order.reflexive(width)),
    bias_agrees,ordered)

type WeightCells is Data:
  SharedWeights{cells: Buffer.Model<Tile.Tile8>}
  PanelWeights{base: U32,cells: Buffer.Model<Tile.Tile8>}

def packed_weights(+rectangle: Products.Rectangle,weights: WeightCells) -> Buffer.Content<Tile.Tile16>:
  match weights:
    case SharedWeights{cells}: Point.shared_pair_model(rectangle,cells)
    case PanelWeights{base,cells}: Point.panel_pair_model(rectangle,base,cells)

def weight_cell(+rectangle: Products.Rectangle,weights: WeightCells,+group: Nat,+lane: Nat,+tap: Nat) -> F32:
  match weights:
    case SharedWeights{+cells}:
      TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,cells),Operands.shared_address(Gemm.Both{U32.to_nat(Values.input_inner(rectangle)),tap},group)))
    case PanelWeights{+base,+cells}:
      +start = Nat.add(U32.to_nat(base),Nat.mul(tap,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))))
      TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,cells),Operands.panel_cell_address(start,group)))

def weight_matches(+limits: Machine.Limits,+rectangle: Products.Rectangle,weights: WeightCells,+group: Nat,+lane: Nat,+tap: Nat,
  +space: Operands.PairSpace<limits,Values.rectangle_channels(rectangle),Values.input_inner(rectangle)>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+tap_inside: {Nat.is_lt(tap,U32.to_nat(Values.input_inner(rectangle))) == True{} : Bool}) ->
  {Spec.pair_lane(Lanes.selected_lane(Lanes.upper(group),lane),Tree.read(Tile.Tile16,Buffer.content_tree(~Tile.Tile16,packed_weights(rectangle,weights)),
      U32.from_nat(Nat.add(Nat.mul(Lanes.pair_of(group),U32.to_nat(Values.input_inner(rectangle))),tap)))) ==
    weight_cell(rectangle,weights,group,lane,tap) : F32}:
  match weights:
    case SharedWeights{+cells}:
      Operands.shared_weight_cell(limits,rectangle,cells,group,lane,tap,space,
        Operands.packed_tap(limits,Values.rectangle_channels(rectangle),Values.input_inner(rectangle),group,tap,space,group_inside,tap_inside),group_inside,lane_inside,tap_inside)
    case PanelWeights{+base,+cells}:
      Operands.panel_weight_cell(limits,rectangle,base,cells,group,lane,tap,space,
        Operands.packed_tap(limits,Values.rectangle_channels(rectangle),Values.input_inner(rectangle),group,tap,space,group_inside,tap_inside),group_inside,lane_inside,tap_inside)

type Cells is Data:
  Cells{rectangle: Products.Rectangle,base: U32,input: Buffer.Content<Tile.Tile8>,weights: WeightCells,bias: Buffer.Content<Tile.Tile8>,lane: Nat}

def cells_rectangle(+cells: Cells) -> Products.Rectangle:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  rectangle

def local_input(cells: Cells,position: Nat,tap: Nat) -> F32:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  Operands.input_value(Buffer.content_tree(~Tile.Tile8,input),U32.to_nat(base),U32.to_nat(Gemm.blocks(Values.input_inner(rectangle))),position,tap)

def local_weight(cells: Cells,group: Nat,tap: Nat) -> F32:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  weight_cell(rectangle,weights,group,lane,tap)

def local_bias(+cells: Cells,+group: Nat) -> F32:
  Cells{rectangle,base,input,weights,bias,+lane} = cells
  TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,bias),U32.from_nat(group)))

def local_query(+cells: Cells,+first: U32,+span: U32,+position: Nat,+group: Nat) -> Query:
  Cells{+rectangle,+base,+input,+weights,+bias,+lane} = cells
  Query{rectangle,base,first,span,Values.rectangle_input(input,packed_weights(rectangle,weights),bias),position,group,lane}

def local_lane(+cells: Cells) -> Nat:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  lane

def cells_base(+cells: Cells) -> U32:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  base

def cells_readers(+cells: Cells) -> Jobs.Readers:
  Cells{rectangle,base,input,weights,bias,lane} = cells
  Values.rectangle_input(input,packed_weights(rectangle,weights),bias)

type Local is Data:
  Local{cells: Cells,limits: Machine.Limits,first: U32,span: U32,position: Nat,group: Nat,
    space: Operands.PairSpace<limits,Values.rectangle_channels(cells_rectangle(cells)),Values.input_inner(cells_rectangle(cells))>,
    after: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},
    group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(cells_rectangle(cells))))) == True{} : Bool},
    lane_inside: {Nat.is_lt(local_lane(cells),8n) == True{} : Bool}}

def local_reader(context: Local) -> Cells:
  Local{cells,limits,first,span,position,group,space,after,group_inside,lane_inside} = context
  cells

def local_target(context: Local) -> Query:
  Local{cells,limits,first,span,position,group,space,after,group_inside,lane_inside} = context
  local_query(cells,first,span,position,group)

def local_tap(+context: Local,+tap: Nat,+inside: {Nat.is_lt(tap,scope_width(~Local,~local_target,context)) == True{} : Bool}) ->
  tap_scope(~Local,~Cells,~local_reader,~local_target,~local_input,~local_weight,context,tap):
  Local{Cells{Products.Rectangle{+out_first,+positions,+inner,+channels,+major,+grid},+base,+input,+weights,+bias,+lane},
    +limits,+first,+span,+position,+group,+space,+after,+group_inside,+lane_inside} = context
  +rectangle = {Products.Rectangle{out_first,positions,inner,channels,major,grid} : Products.Rectangle}
  +readers = Values.rectangle_input(input,packed_weights(rectangle,weights),bias)
  Gemm.Both{Operands.input_at(rectangle,base,first,span,readers,position,tap,after),
    weight_matches(limits,rectangle,weights,group,lane,tap,space,group_inside,lane_inside,inside)}

def local_bias_matches(+context: Local) ->
  {Taps.bias_lane(query_target(local_target(context)),query_pair(local_target(context)),query_lane(local_target(context))) ==
    local_bias(local_reader(context),query_group(local_target(context))) : F32}:
  Local{Cells{Products.Rectangle{+out_first,+positions,+inner,+channels,+major,+grid},+base,+input,+weights,+bias,+lane},
    +limits,+first,+span,+position,+group,Operands.PairSpace{+groups_room,+width_room,+weights_room},
    +after,+group_inside,+lane_inside} = context
  Operands.bias_at(Buffer.content_tree(~Tile.Tile8,bias),channels,group,lane,limits,groups_room,group_inside,lane_inside)

def local_value(+cells: Cells,+position: Nat,+group: Nat) -> F32:
  (Dot.logical_dot(~Cells,~local_input,~local_weight,cells,position,group,U32.to_nat(Values.input_inner(cells_rectangle(cells)))) + local_bias(cells,group) : F32)

def evaluated_hit(~accumulate: Jobs.Accumulate(),~Ready: U32 -> Data,~order: KernelOrder(~accumulate,~Ready),
  +cells: Cells,+limits: Machine.Limits,+span: U32,+position: Nat,+group: Nat,
  +space: Operands.PairSpace<limits,Values.rectangle_channels(cells_rectangle(cells)),Values.input_inner(cells_rectangle(cells))>,
  +ready: Ready(Values.input_inner(cells_rectangle(cells))),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(cells_rectangle(cells))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(local_lane(cells),8n) == True{} : Bool},
  +value: F32,
  +hit: Lanes.LaneHit<accumulate,cells_rectangle(cells),cells_base(cells),span,cells_readers(cells),position,group,local_lane(cells),value>) ->
  {value == local_value(cells,position,group) : F32}:
  Cells{Products.Rectangle{+out_first,+positions,+inner,+channels,+major,+grid},+input_base,+input,+weights,+bias,+lane} = cells
  Lanes.LaneHit{+first,+after,+before,+exact} = hit
  +context = {Local{cells,limits,first,span,position,group,space,after,group_inside,lane_inside} : Local}
  +query = local_target(context)
  +target = query_target(query)
  +pair = query_pair(query)
  +unit = query_unit(query)
  +row = query_row(query)
  +selected = query_lane(query)
  +stored = Spec.pair_lane(selected,Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(unit),row))
  Equal.trans(F32,value,stored,local_value(cells,position,group),exact,
    evaluated(~accumulate,~Local,~Cells,~local_reader,~local_target,~local_input,~local_weight,~local_tap,context,local_bias(cells,group),
      Operands.row_small(first,position),local_bias_matches(context),order(limits,target,pair,unit,row,selected,ready)))

def stored_lane(~run: Point.Run(),+cells: Cells,+position: Nat,+group: Nat) -> F32:
  Cells{+rectangle,+base,+input,+weights,+bias,+lane} = cells
  Lanes.stored_lane(~run,rectangle,base,input,packed_weights(rectangle,weights),bias,position,group,lane)

def rectangle_lane(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),
  ~Ready: U32 -> Data,~order: KernelOrder(~accumulate,~Ready),
  +cells: Cells,+limits: Machine.Limits,+position: Nat,+group: Nat,
  +space: Operands.PairSpace<limits,Values.rectangle_channels(cells_rectangle(cells)),Values.input_inner(cells_rectangle(cells))>,
  +ready: Ready(Values.input_inner(cells_rectangle(cells))),
  +channel_room: {Nat.is_le(U32.to_nat(Values.rectangle_channels(cells_rectangle(cells))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel_positive: {Nat.is_lt(0n,U32.to_nat(Values.rectangle_channels(cells_rectangle(cells)))) == True{} : Bool},
  +positions_room: {Nat.is_le(U32.to_nat(Values.rectangle_positions(cells_rectangle(cells))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Values.rectangle_positions(cells_rectangle(cells))),U32.to_nat(Gemm.blocks(Values.rectangle_channels(cells_rectangle(cells))))),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span: Values.Span<limits,Values.input_inner(cells_rectangle(cells))>,
  +position_inside: {Nat.is_lt(position,U32.to_nat(Values.rectangle_positions(cells_rectangle(cells)))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(cells_rectangle(cells))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(local_lane(cells),8n) == True{} : Bool}) ->
  {stored_lane(~run,cells,position,group) == local_value(cells,position,group) : F32}:
  Cells{+rectangle,+base,+input,+weights,+bias,+lane} = cells
  +width = Products.block_positions(Values.input_inner(rectangle))
  evaluated_hit(~accumulate,~Ready,~order,cells,limits,width,position,group,space,ready,group_inside,lane_inside,stored_lane(~run,cells,position,group),
    Lanes.rectangle_lane(~run,~accumulate,~write,limits,rectangle,base,input,packed_weights(rectangle,weights),bias,channel_room,channel_positive,positions_room,
      cells_room,span,position,group,lane,position_inside,group_inside,lane_inside))

def LogicalSamples(~Source: Data,~input: Source -> Nat -> Nat -> F32,~weight: Source -> Nat -> Nat -> F32,+cells: Cells,
  +source: Source,+position: Nat,+group: Nat,+source_position: Nat,+source_channel: Nat) -> Data:
  Ordered.MatchingSamples(~Nat,~Nat,~Nat,~Cells,~Nat,~Nat,~Source,~local_input,~local_weight,~input,~weight,
    Ranges.range(U32.to_nat(Values.input_inner(cells_rectangle(cells))),0n),group,position,cells,source_channel,source_position,source)

def local_agrees(~Source: Data,~input: Source -> Nat -> Nat -> F32,~weight: Source -> Nat -> Nat -> F32,+cells: Cells,
  +source: Source,+position: Nat,+group: Nat,+source_position: Nat,+source_channel: Nat,+bias: F32,
  +samples: LogicalSamples(~Source,~input,~weight,cells,source,position,group,source_position,source_channel),
  +bias_agrees: {local_bias(cells,group) == bias : F32}) ->
  {local_value(cells,position,group) == (Dot.logical_dot(~Source,~input,~weight,source,source_position,source_channel,U32.to_nat(Values.input_inner(cells_rectangle(cells)))) + bias : F32) : F32}:
  +width = U32.to_nat(Values.input_inner(cells_rectangle(cells)))
  +left = Dot.logical_dot(~Cells,~local_input,~local_weight,cells,position,group,width)
  +right = Dot.logical_dot(~Source,~input,~weight,source,source_position,source_channel,width)
  +offset = local_bias(cells,group)
  %bias_agrees : {(left + offset : F32) == (right + _ : F32) : F32}
  Equal.cong(F32,F32,dot => (dot + offset : F32),left,right,
    Ordered.dot_congruence(~Nat,~Nat,~Nat,~Cells,~Nat,~Nat,~Source,~local_input,~local_weight,~input,~weight,
      Ranges.range(width,0n),group,position,cells,source_channel,source_position,source,0.0,samples))
