# Valid output lanes of a staged rectangle read the chosen block's ordered dot.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_point_products.bend as Products
import ./kernel_gemm_output_groups.bend as Groups
import ./kernel_gemm_point_jobs.bend as Jobs
import ./kernel_gemm_point_spec.bend as Spec
import ./kernel_tile_refinement.bend as Lanes
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_point_product_values.bend as Values
import ./tensor_point_products.bend as Point

def pair_of(+group: Nat) -> Nat:
  Nat.div(group,2n)

def upper(+group: Nat) -> Bool:
  U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1)

def selected_lane(high: Bool,+lane: Nat) -> Nat:
  match high:
    case False{}: lane
    case True{}: Nat.add(8n,lane)

def half_lane(high: Bool,+value: Tile.Tile16,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Lanes.select_lane(lane,Gemm.staged_half(value,high)) == Spec.pair_lane(selected_lane(high,lane),value) : F32}:
  match high:
    case False{}:
      Tile.Tiles{+low,+upper} = value
      %Equal.sym(Bool,Nat.is_lt(lane,8n),True{},inside) : {Lanes.select_lane(lane,low) == Spec.half_lane(_,lane,low,upper) : F32}
      {==}
    case True{}:
      Tile.Tiles{+low,+upper} = value
      +later = Nat.add(8n,lane)
      +outside = Order.not_below(8n,later,Order.le_add(8n,lane))
      %Equal.sym(Bool,Nat.is_lt(later,8n),False{},outside) : {Lanes.select_lane(lane,upper) == Spec.half_lane(_,later,low,upper) : F32}
      %Equal.sym(Nat,Nat.sub(later,8n),lane,Order.subtract_shift(8n,lane)) : {Lanes.select_lane(lane,upper) == Lanes.select_lane(_,upper) : F32}
      {==}

def pair_inside(+limits: Machine.Limits,+channels: U32,+group: Nat,
  +groups_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}) ->
  {Nat.is_lt(pair_of(group),U32.to_nat(Gemm.pairs_of(channels))) == True{} : Bool}:
  +groups = U32.to_nat(Gemm.blocks(channels))
  +pairs = U32.to_nat(Gemm.pairs_of(channels))
  +exact = Equal.trans(Nat,pairs,Nat.div(Nat.add(groups,1n),2n),Ceiling.ceiling(groups,2n),
    Groups.pairs_exact(limits,channels,groups_room),
    Ceiling.shifted_formula(groups,2n,{==}))
  DivisionOrder.quotient_below(group,pairs,2n,{==},
    Order.lt_le(group,groups,Nat.mul(pairs,2n),inside,
      %Equal.sym(Nat,pairs,Ceiling.ceiling(groups,2n),exact) : {Nat.is_le(groups,Nat.mul(_,2n)) == True{} : Bool}
      Ceiling.covers(groups,2n,{==})))

def stored_lane(~run: Point.Run(),+rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Content<Tile.Tile8>,+pairs: Buffer.Content<Tile.Tile16>,
  +bias: Buffer.Content<Tile.Tile8>,+position: Nat,+group: Nat,+lane: Nat) -> F32:
  +stage = Buffer.content_tree(~Tile.Tile16,Point.computed_staged(~run,rectangle,input_base,input,pairs,bias))
  +pair = pair_of(group)
  Lanes.select_lane(lane,Gemm.staged_half(Tree.read(Tile.Tile16,stage,U32.from_nat(Values.slot(position,Values.stage_width(rectangle),pair))),upper(group)))

type LaneHit<-accumulate: Jobs.Accumulate(),-rectangle: Products.Rectangle,-base: U32,-span: U32,-readers: Jobs.Readers,-position: Nat,-group: Nat,-lane: Nat,-value: F32> is Data:
  LaneHit{first: U32,after: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},
    before: {Nat.is_lt(position,U32.to_nat(Values.block_end(Values.rectangle_positions(rectangle),first,span))) == True{} : Bool},
    exact: {value == Spec.pair_lane(selected_lane(upper(group),lane),Values.block_value(~accumulate,rectangle,base,first,span,readers,position,pair_of(group))) : F32}}

def from_pair(~accumulate: Jobs.Accumulate(),+rectangle: Products.Rectangle,+base: U32,+span: U32,+readers: Jobs.Readers,
  +position: Nat,+group: Nat,+lane: Nat,+value: Tile.Tile16,+inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +hit: Values.BlockHit<accumulate,rectangle,base,span,readers,position,pair_of(group),value>) ->
  LaneHit<accumulate,rectangle,base,span,readers,position,group,lane,Lanes.select_lane(lane,Gemm.staged_half(value,upper(group)))>:
  Values.BlockHit{+first,+after,+before,+exact} = hit
  +high = upper(group)
  +index = selected_lane(high,lane)
  +wanted = Values.block_value(~accumulate,rectangle,base,first,span,readers,position,pair_of(group))
  LaneHit{first,after,before,Equal.trans(F32,Lanes.select_lane(lane,Gemm.staged_half(value,high)),Spec.pair_lane(index,value),Spec.pair_lane(index,wanted),
    half_lane(high,value,lane,inside),Equal.cong(Tile.Tile16,F32,cell => Spec.pair_lane(index,cell),value,wanted,exact))}

def rectangle_lane(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),
  +limits: Machine.Limits,+rectangle: Products.Rectangle,+input_base: U32,+input: Buffer.Content<Tile.Tile8>,+pairs: Buffer.Content<Tile.Tile16>,+bias: Buffer.Content<Tile.Tile8>,
  +channels_room: {Nat.is_le(U32.to_nat(Values.rectangle_channels(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(Values.rectangle_channels(rectangle))) == True{} : Bool},
  +positions_room: {Nat.is_le(U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rectangle_room: {Nat.is_le(Nat.mul(U32.to_nat(Values.rectangle_positions(rectangle)),U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span: Values.Span<limits,Values.input_inner(rectangle)>,+position: Nat,+group: Nat,+lane: Nat,
  +position_inside: {Nat.is_lt(position,U32.to_nat(Values.rectangle_positions(rectangle))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(Values.rectangle_channels(rectangle)))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  LaneHit<accumulate,rectangle,input_base,Products.block_positions(Values.input_inner(rectangle)),Values.rectangle_input(input,pairs,bias),position,group,lane,
    stored_lane(~run,rectangle,input_base,input,pairs,bias,position,group,lane)>:
  Products.Rectangle{+first,+positions,+inner,+channels,+group_major,+grid} = rectangle
  Values.Span{+span_positive,+span_room} = span
  +groups_room = Groups.groups_fit(limits,channels,channels_room)
  +rows = U32.to_nat(positions)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +width = Values.stage_width(rectangle)
  +cells = Nat.mul(rows,groups)
  +space = {Values.RectangleSpace{Groups.groups_positive(limits,channels,channels_room,channels_positive),Groups.group_span(limits,channels,channels_room),rectangle_room} :
    Values.RectangleSpace<limits,positions,channels>}
  +stage_room = Order.transitive(Values.stage_cells(rectangle),cells,U32.to_nat(Machine.capacity(limits)),
    Index.mul_le_left(rows,width,groups,Groups.pairs_fewer(limits,channels,groups_room)),rectangle_room)
  +packed = Values.rectangle_input(input,pairs,bias)
  +stage = Buffer.content_tree(~Tile.Tile16,Point.computed_staged(~run,rectangle,input_base,input,pairs,bias))
  +pair = pair_of(group)
  +block = Values.computed_hit(~run,~accumulate,~write,limits,rectangle,input_base,input,pairs,bias,space,positions_room,span_room,span_positive,stage_room,
    position,pair,position_inside,pair_inside(limits,channels,group,groups_room,group_inside))
  from_pair(~accumulate,rectangle,input_base,Products.block_positions(inner),packed,position,group,lane,
    Tree.read(Tile.Tile16,stage,U32.from_nat(Values.slot(position,width,pair))),lane_inside,block)
