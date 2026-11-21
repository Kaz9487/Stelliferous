# Native panel dot products use the rows and columns of their own image.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_move.bend as Move
import ../../lib/tensor_point_products.bend as Products
import ../../lib/proofs/kernel_gemm_point_dot.bend as Dot
import ../../lib/proofs/kernel_gemm_point_jobs.bend as Jobs
import ../../lib/proofs/kernel_gemm_point_refinement.bend as Kernel
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_division_order.bend as Division
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as Index
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_panel_product_samples.bend as Panels
import ../../lib/proofs/tensor_point_product_evaluation.bend as Evaluate
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_operands.bend as Operands
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_point_product_values.bend as Values
import ../../lib/proofs/tensor_point_product_windows.bend as Bands
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ./point_product_values.bend as Finite
import ./product_values.bend as Logical

def zero_tile(source: Unit,index: U32) -> Tile.Tile8:
  Products.zero()

type Bias<-buffer: Buffer.Content<Tile.Tile8>,-channels: U32> is Data:
  Bias{count: {Buffer.content_length(~Tile.Tile8,buffer) == U32.to_nat(Gemm.blocks(channels)) : Nat},
    values: Contract.Values(~Tile.Tile8,~Unit,~zero_tile,Unit{},buffer)}

type Domain<-limits: Machine.Limits,-images: U32,-rows: U32,-inner: U32,-channels: U32> is Data:
  Domain{rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
    row_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    left_size: Panels.Size<limits,images,rows,inner>,right_size: Panels.Size<limits,images,inner,channels>,
    transpose_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(Panels.row_total(images,rows),U32.to_nat(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    span: Values.Span<limits,inner>,groups: TiledValues.Groups<limits,inner>,pairs: Operands.PairSpace<limits,channels,inner>,
    work_room: {Nat.is_le(Panels.image_size(rows,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

type PanelImage<-Left: Data,-left_value: Left -> U32 -> F32,-Right: Data,-right_value: Right -> U32 -> F32> is Data:
  PanelImage{limits: Machine.Limits,left: Left,right: Right,images: U32,rows: U32,inner: U32,channels: U32,grid: Grid.Grid,first: U32,
    input: Held.Holding<Tile.Tile8>,weights: Held.Holding<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>,
    domain: Domain<limits,images,rows,inner,channels>,image_inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(images)) == True{} : Bool},
    left_window: Panels.ImageWindow(~Left,~left_value,left,images,rows,inner,first,input),
    right_window: Panels.ImageWindow(~Right,~right_value,right,images,inner,channels,first,weights),bias_values: Bias<Buffer.content(~Tile.Tile8,bias),channels>}

def image_rows(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,image: PanelImage<Left,left_value,Right,right_value>) -> U32:
  PanelImage{limits,left,right,images,rows,inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values} = image
  rows

def image_channels(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,image: PanelImage<Left,left_value,Right,right_value>) -> U32:
  PanelImage{limits,left,right,images,rows,inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values} = image
  channels

type Lane<-Left: Data,-left_value: Left -> U32 -> F32,-Right: Data,-right_value: Right -> U32 -> F32> is Data:
  Lane{image: PanelImage<Left,left_value,Right,right_value>,position: Nat,group: Nat,lane: Nat,
    position_inside: {Nat.is_lt(position,U32.to_nat(image_rows(~Left,~left_value,~Right,~right_value,image))) == True{} : Bool},
    group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(image_channels(~Left,~left_value,~Right,~right_value,image)))) == True{} : Bool},lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
    active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(image_channels(~Left,~left_value,~Right,~right_value,image))) == True{} : Bool}}

def band_model_content(+holding: Held.Holding<Tile.Tile8>,+start: U32,+width: U32) ->
  {Buffer.content(~Tile.Tile8,Bands.band_model(holding,start,width)) ==
    Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Bands.band(holding,start,width))) : Buffer.Content<Tile.Tile8>}:
  Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,Bands.band(holding,start,width)))

def cells_for(+rectangle: Products.Rectangle,+input: Held.Holding<Tile.Tile8>,+weights: Held.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  +image: U32,+rows: U32,+inner: U32,+channels: U32,+lane: Nat) -> Evaluate.Cells:
  +input_start = Panels.band_start(image,rows,inner)
  +input_count = Panels.cell_width(rows,inner)
  +weight_start = Panels.band_start(image,inner,channels)
  +weight_count = Panels.cell_width(inner,channels)
  Evaluate.Cells{rectangle,Bands.band_base(input,input_start,input_count),Buffer.content(~Tile.Tile8,Bands.band_model(input,input_start,input_count)),
    Evaluate.PanelWeights{Bands.band_base(weights,weight_start,weight_count),Bands.band_model(weights,weight_start,weight_count)},Buffer.content(~Tile.Tile8,bias),lane}

def stage_views(~run: Point.Run(),+rectangle: Products.Rectangle,left: Move.Band<Tile.Tile8>,right: Move.Band<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) ->
  {Point.band_stage(~run,rectangle,left,right,bias) == Point.computed_staged(~run,rectangle,Held.band_offset(~Tile.Tile8,left),
    Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,left))),
    Point.panel_pair_model(rectangle,Held.band_offset(~Tile.Tile8,right),Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,right))),Buffer.content(~Tile.Tile8,bias)) : Buffer.Content<Tile.Tile16>}:
  Move.Band{left_loan,input,+input_base} = left
  Move.Band{right_loan,weights,+weight_base} = right
  Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile16>,cells => Point.computed_staged(~run,rectangle,input_base,cells,
    Point.panel_pair_model(rectangle,weight_base,Buffer.remodel(~Tile.Tile8,weights)),Buffer.content(~Tile.Tile8,bias)),
    Buffer.contents(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,input)),
    Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,input)),Buffer.contents(~Tile.Tile8,input),Buffer.remodel_content(~Tile.Tile8,input)))

def transposed_at(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+limits: Machine.Limits,+image: Nat,+inner: Nat,+outputs: Nat,
  +channel: Nat,+tap: Nat,+channel_inside: {Nat.is_lt(channel,outputs) == True{} : Bool},+tap_inside: {Nat.is_lt(tap,inner) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(outputs,inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,inner),tap),outputs),channel))) ==
    Logical.transposed_value(~Right,~right_value,Logical.Transposed{right,image,inner,outputs},U32.from_nat(Nat.add(Nat.mul(channel,inner),tap))) : F32}:
  +word = Nat.add(Nat.mul(channel,inner),tap)
  +start = Nat.mul(image,inner)
  +target = right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(start,tap),outputs),channel)))
  +inner_positive = Representation.below_positive(tap,inner,tap_inside)
  +bound = Machine.capacity(limits)
  +area = Nat.mul(outputs,inner)
  +fits = Order.transitive(word,area,U32.to_nat(bound),Bounds.strict_le(word,area,Index.radix(channel,outputs,tap,inner,channel_inside,tap_inside)),room)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(word)),word,Bounds.roundtrip(word,bound,Limits.capacity_fits(limits),fits)) :
    {target == right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(start,Nat.mod(_,inner)),outputs),Nat.div(_,inner)))) : F32}
  %Equal.sym(Nat,Nat.mod(word,inner),tap,Radix.remainder(channel,tap,inner,inner_positive,tap_inside)) :
    {target == right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(start,_),outputs),Nat.div(word,inner)))) : F32}
  %Equal.sym(Nat,Nat.div(word,inner),channel,Radix.quotient(channel,tap,inner,inner_positive,tap_inside)) :
    {target == right_value(right,U32.from_nat(Nat.add(Nat.mul(Nat.add(start,tap),outputs),_))) : F32}
  {==}

def lane_cells(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  scope: Lane<Left,left_value,Right,right_value>) -> Evaluate.Cells:
  Lane{PanelImage{limits,left,right,images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,domain,image_inside,left_window,right_window,bias_values},
    position,group,+lane,position_inside,group_inside,lane_inside,active} = scope
  cells_for(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first),input,weights,bias,first,rows,inner,channels,lane)

def readers_of(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  scope: Lane<Left,left_value,Right,right_value>) -> Finite.Readers<Left,Logical.Transposed<Right>>:
  Lane{PanelImage{limits,left,right,images,rows,+inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values},
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  Finite.Readers{left,Logical.Transposed{right,U32.to_nat(first),U32.to_nat(inner),U32.to_nat(channels)},U32.to_nat(inner)}

def tap_extent(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  scope: Lane<Left,left_value,Right,right_value>) -> Nat:
  Lane{PanelImage{limits,left,right,images,rows,inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values},
    position,group,lane,position_inside,group_inside,lane_inside,active} = scope
  U32.to_nat(inner)

def Tap(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +scope: Lane<Left,left_value,Right,right_value>,+key: Nat) -> Data:
  Lane{PanelImage{limits,left,right,images,+rows,+inner,channels,grid,+first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values},
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  +cells = lane_cells(~Left,~left_value,~Right,~right_value,scope)
  +source = readers_of(~Left,~left_value,~Right,~right_value,scope)
  +global = Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Gemm.Both<{Evaluate.local_input(cells,position,key) == Finite.input(~Left,~left_value,~Logical.Transposed<Right>,source,global,key) : F32},
    {Evaluate.local_weight(cells,group,key) == Finite.weight(~Left,~Logical.Transposed<Right>,~Logical.transposed_value(~Right,~right_value),source,channel,key) : F32}>

def tap(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +scope: Lane<Left,left_value,Right,right_value>,+key: Nat,+inside: {Nat.is_lt(key,tap_extent(~Left,~left_value,~Right,~right_value,scope)) == True{} : Bool}) ->
  Tap(~Left,~left_value,~Right,~right_value,scope,key):
  Lane{PanelImage{+limits,+left,+right,+images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,
    Domain{rows_pos,channels_pos,rows_room,channels_room,+left_size,+right_size,+transpose_room,output_room,span,+groups,pairs,work_room},
    +image_inside,+left_window,+right_window,bias_values},+position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  +in_group = Nat.div(key,8n)
  +in_lane = Nat.mod(key,8n)
  +in_width = Panels.cell_width(rows,inner)
  +in_first = Panels.band_start(first,rows,inner)
  +in_model = Bands.band_model(input,in_first,in_width)
  +in_base = U32.to_nat(Bands.band_base(input,in_first,in_width))
  +in_word = U32.from_nat(Nat.add(in_base,Nat.add(Nat.mul(position,U32.to_nat(Gemm.blocks(inner))),in_group)))
  +in_index = Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position),U32.to_nat(inner)),key)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +right_width = Panels.cell_width(inner,channels)
  +right_start = Panels.band_start(first,inner,channels)
  +right_model = Bands.band_model(weights,right_start,right_width)
  +right_base = U32.to_nat(Bands.band_base(weights,right_start,right_width))
  +right_offset = Nat.mul(key,U32.to_nat(Gemm.blocks(channels)))
  +right_at = Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(inner)),key),U32.to_nat(channels)),channel)
  +right_sample = right_value(right,U32.from_nat(right_at))
  +transpose = {Logical.Transposed{right,U32.to_nat(first),U32.to_nat(inner),U32.to_nat(channels)} : Logical.Transposed<Right>}
  +target = Logical.transposed_value(~Right,~right_value,transpose,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(inner)),key)))
  Gemm.Both{
    %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,in_model)),Buffer.tree(~Tile.Tile8,in_model),Buffer.content_tree_of(~Tile.Tile8,in_model)) :
      {TileLaws.select_lane(in_lane,Tree.read(Tile.Tile8,_,in_word)) == left_value(left,U32.from_nat(in_index)) : F32}
    %TiledValues.channel_split(key) : {TileLaws.select_lane(in_lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,in_model),in_word)) ==
      left_value(left,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position),U32.to_nat(inner)),_))) : F32}
    Panels.sample(~Left,~left_value,left,limits,images,rows,inner,first,input,image_inside,left_size,left_window,position,in_group,in_lane,
      position_inside,TiledValues.group_inside(limits,inner,key,groups,inside),Division.remainder_small(key,8n,{==}),
      %Equal.sym(Nat,Nat.add(in_lane,Nat.mul(in_group,8n)),key,TiledValues.channel_split(key)) : {Nat.is_lt(_,U32.to_nat(inner)) == True{} : Bool} inside),
    %Equal.sym(Nat,Nat.add(Nat.add(right_base,right_offset),group),Nat.add(right_base,Nat.add(right_offset,group)),Addition.associative(right_base,right_offset,group)) :
      {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,right_model),U32.from_nat(_))) == target : F32}
    Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,right_model),U32.from_nat(Nat.add(right_base,Nat.add(right_offset,group))))),right_sample,target,
      Panels.sample(~Right,~right_value,right,limits,images,inner,channels,first,weights,image_inside,right_size,right_window,key,group,lane,inside,group_inside,lane_inside,active),
      transposed_at(~Right,~right_value,right,limits,U32.to_nat(first),U32.to_nat(inner),U32.to_nat(channels),channel,key,active,inside,transpose_room))}

def bias_zero(+channels: U32,+bias: Buffer.Content<Tile.Tile8>,+group: Nat,+lane: Nat,values: Bias<bias,channels>,
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,bias),U32.from_nat(group))) == 0.0 : F32}:
  Buffer.Content{+tree,+length} = bias
  Bias{+count,+read} = values
  +index = U32.from_nat(group)
  %Equal.sym(Tile.Tile8,Buffer.value(~Tile.Tile8,index,bias),Products.zero(),Contract.read(~Tile.Tile8,~Unit,~zero_tile,Unit{},bias,group,read,
    %Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,bias),U32.to_nat(Gemm.blocks(channels)),count) : {Nat.is_lt(group,_) == True{} : Bool} group_inside)) :
    {TileLaws.select_lane(lane,_) == 0.0 : F32}
  TileLaws.filled_lane(lane,0.0)

def dot_index(+image: U32,+rows: U32,+outputs: U32,+position: Nat,+channel: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(rows)),position),U32.to_nat(outputs)),channel)

def public_lane(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  scope: Lane<Left,left_value,Right,right_value>) -> F32:
  Lane{PanelImage{limits,+left,+right,images,+rows,+inner,+channels,grid,+first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values},
    +position,+group,+lane,position_inside,group_inside,lane_inside,active} = scope
  Logical.batched_value(~Left,~left_value,~Right,~right_value,Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},
    U32.from_nat(dot_index(first,rows,channels,position,Nat.add(lane,Nat.mul(group,8n)))))

def rectangle_value(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  scope: Lane<Left,left_value,Right,right_value>) -> F32:
  Lane{image,+position,+group,lane,position_inside,group_inside,lane_inside,active} = scope
  Evaluate.local_value(lane_cells(~Left,~left_value,~Right,~right_value,scope),position,group)

def local_result(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +scope: Lane<Left,left_value,Right,right_value>) ->
  {rectangle_value(~Left,~left_value,~Right,~right_value,scope) == public_lane(~Left,~left_value,~Right,~right_value,scope) : F32}:
  Lane{PanelImage{+limits,+left,+right,+images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,
    Domain{+rows_pos,+channels_pos,rows_room,channels_room,+left_size,+right_size,+transpose_room,+out_room,span,groups,pairs,work_room},
    +image_inside,+left_window,+right_window,+bias_values},+position,+group,+lane,+position_inside,+group_inside,+lane_inside,+active} = scope
  +cells = lane_cells(~Left,~left_value,~Right,~right_value,scope)
  +source = readers_of(~Left,~left_value,~Right,~right_value,scope)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +width = U32.to_nat(inner)
  +output_width = U32.to_nat(channels)
  +global = Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position)
  +dotted = Dot.logical_dot(~Finite.Readers<Left,Logical.Transposed<Right>>,~Finite.input(~Left,~left_value,~Logical.Transposed<Right>),
    ~Finite.weight(~Left,~Logical.Transposed<Right>,~Logical.transposed_value(~Right,~right_value)),source,global,channel,width)
  +target = Logical.ordered_sum(~Left,~left_value,~Logical.Transposed<Right>,~Logical.transposed_value(~Right,~right_value),width,0n,left,Nat.mul(global,width),
    Logical.Transposed{right,U32.to_nat(first),width,output_width},Nat.mul(channel,width),0.0)
  +expected = public_lane(~Left,~left_value,~Right,~right_value,scope)
  Equal.trans(F32,Evaluate.local_value(cells,position,group),(dotted + 0.0 : F32),expected,
    Evaluate.local_agrees(~Finite.Readers<Left,Logical.Transposed<Right>>,~Finite.input(~Left,~left_value,~Logical.Transposed<Right>),
      ~Finite.weight(~Left,~Logical.Transposed<Right>,~Logical.transposed_value(~Right,~right_value)),cells,source,position,group,global,channel,0.0,
      Members.range_all(~Lane<Left,left_value,Right,right_value>,~Tap(~Left,~left_value,~Right,~right_value),
        ~tap_extent(~Left,~left_value,~Right,~right_value),~tap(~Left,~left_value,~Right,~right_value),width,0n,scope,Order.reflexive(width)),
      bias_zero(channels,Buffer.content(~Tile.Tile8,bias),group,lane,bias_values,group_inside)),
    Equal.trans(F32,(dotted + 0.0 : F32),(target + 0.0 : F32),expected,
      Equal.cong(F32,F32,total => (total + 0.0 : F32),dotted,target,Finite.ordered(~Left,~left_value,~Logical.Transposed<Right>,~Logical.transposed_value(~Right,~right_value),
        width,0n,source,global,channel,0.0)),
      Equal.sym(F32,expected,(target + 0.0 : F32),Logical.batched_read(~Left,~left_value,~Right,~right_value,left,right,U32.to_nat(images),width,output_width,U32.to_nat(rows),
        Machine.capacity(limits),Limits.capacity_fits(limits),out_room,channels_pos,rows_pos,U32.to_nat(first),channel,position,image_inside,active,position_inside))))

def image_source(~run: Point.Run(),+work: Products.Shape,+first: U32,+input: Held.Holding<Tile.Tile8>,+weights: Held.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>) -> Fragments.RectangleRead:
  Fragments.RectangleRead{Point.panel_geometry(work,first),Buffer.content_tree(~Tile.Tile16,Point.panel_stage_model(~run,work,first,Held.pack(~Tile.Tile8,input),Held.pack(~Tile.Tile8,weights),bias))}

def stage_lane(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),
  ~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  +limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,
  +input: Held.Holding<Tile.Tile8>,+weights: Held.Holding<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+domain: Domain<limits,images,rows,inner,channels>,
  +ready: Ready(inner),+position: Nat,+group: Nat,+lane: Nat,
  +position_inside: {Nat.is_lt(position,U32.to_nat(rows)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Fragments.staged_at(image_source(~run,Products.Shape{rows,inner,channels,grid},first,input,weights,bias),position,group)) ==
    Evaluate.local_value(cells_for(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first),input,weights,bias,first,rows,inner,channels,lane),position,group) : F32}:
  Domain{+rows_pos,+channels_pos,+rows_room,+channel_room,left_size,right_size,transpose_room,output_room,+span,groups,+pairs,+work_room} = domain
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.panel_geometry(work,first)
  +cells = cells_for(rectangle,input,weights,bias,first,rows,inner,channels,lane)
  +in_width = Panels.cell_width(rows,inner)
  +out_width = Panels.cell_width(inner,channels)
  +input_start = Panels.band_start(first,rows,inner)
  +weights_start = Panels.band_start(first,inner,channels)
  +want = Evaluate.local_value(cells,position,group)
  -native = Point.panel_stage_model(~run,work,first,Held.pack(~Tile.Tile8,input),Held.pack(~Tile.Tile8,weights),bias)
  +moved = Point.computed_staged(~run,rectangle,Bands.band_base(input,input_start,in_width),Buffer.content(~Tile.Tile8,Bands.band_model(input,input_start,in_width)),
    Point.panel_pair_model(rectangle,Bands.band_base(weights,weights_start,out_width),Bands.band_model(weights,weights_start,out_width)),Buffer.content(~Tile.Tile8,bias))
  %Equal.sym(Buffer.Content<Tile.Tile16>,native,moved,stage_views(~run,rectangle,Bands.band(input,input_start,in_width),Bands.band(weights,weights_start,out_width),bias)) :
    {TileLaws.select_lane(lane,Fragments.staged_at(Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,_)},position,group)) == want : F32}
  Evaluate.rectangle_lane(~run,~accumulate,~write,~Ready,~order,cells,limits,position,group,pairs,ready,channel_room,channels_pos,rows_room,work_room,span,
    position_inside,group_inside,lane_inside)
