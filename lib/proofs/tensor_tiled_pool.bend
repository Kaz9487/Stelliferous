# Pooling over the last two axes of tiled values. The value of the pool at an
# index of [before.., channels, out_height, out_width] is the window of its
# image, channel and output row and column over the logical input, every
# kernel row folded with the operation from pad and the rows folded from pad.
# Tiled values of the contract of rank three or more with a valid pool that
# fits the machine hold that pool after the call; a valid pool of rank two
# holds the pool of its values with a leading axis of extent 1.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_pool.bend as TiledPool
import ../tensor_window.bend as Window
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ../tensor_access.bend as Access
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ../tensor_tiled_layout.bend as TiledLayout
import ../tensor_f32.bend as F
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./tensor_window_values.bend as Values
import ./tensor_window_rows.bend as Rows
import ./tensor_window_spec.bend as Spec
import ./tensor_window_public.bend as WindowPublic
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_reshape.bend as TiledReshape
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./boolean_logic.bend as Boolean

# ---- The pool of logical values ----

# Pooling reads no weights and no bias.
def no_weight(parameters: Unit,channel: Nat,kernel_row: U32,column: U32) -> F32:
  0.0

def no_bias(parameters: Unit,channel: Nat) -> F32:
  0.0

# The pool of logical values of channels channels over the extents of a pool,
# from pad.
def pool_spec(~Logical: Data,logical: Logical,+channels: U32,pool: Access.Pool,+pad: F32) -> Spec.Spec<Logical,Unit>:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  Spec.Spec{Spec.Input{logical,channels,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(channels)},out_height,out_width}

# The shape a pool reads: a shape of rank two gains a leading axis of extent
# 1, its one image of one channel.
def raised(+shape: List<&2,U32>) -> List<&2,U32>:
  Bool.pick(List<&2,U32>,Nat.is_le(3n,List.length(&2,U32,shape)),shape,Con{1,shape})

# The channels of a pool: the lanes of the shape it reads.
def pool_channels(+shape: List<&2,U32>) -> U32:
  Grid.lanes_at(raised(shape),Tiled.lane_axis(List.length(&2,U32,raised(shape))))

# What a window operation needs of the machine for the images, the channels and
# the extents of a pool.
def pool_fits(+limits: Machine.Limits,+images: Nat,+lanes: U32,pool: Access.Pool) -> Data:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  WindowPublic.Fits<limits,images,lanes,height,width,kernel,stride,padding,out_height,out_width>

# What the pool of a shape needs of the machine: the axes before the channels
# of the shape it reads are its images.
def PoolFits(+limits: Machine.Limits,+shape: List<&2,U32>,+kernel: U32,+stride: U32,+padding: U32) -> Data:
  pool_fits(limits,ShapeLaws.elements(List.take(&2,U32,raised(shape),Tiled.lane_axis(List.length(&2,U32,raised(shape))))),pool_channels(shape),
    Window.geometry(raised(shape),kernel,stride,padding))

# Tiled values that hold the pool of the logical values of a shape.
def Pooled(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,
  +padding: U32,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),pool_spec(~Logical,logical,pool_channels(shape),Window.geometry(shape,kernel,stride,padding),pad),
    Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding)),tiled)

# ---- Pooling reads no parameters ----

def pooled_tap(~operation: F32 -> F32 -> F32,+pad: F32,+folded: Values.Folded<F32>,+value: F32,+first_weight: F32,+second_weight: F32) ->
  {Values.tapped(~F32,~operation,~Values.lane_update,Values.Pooling{pad},folded,value,first_weight) ==
    Values.tapped(~F32,~operation,~Values.lane_update,Values.Pooling{pad},folded,value,second_weight) : Values.Folded<F32>}:
  match folded:
    case Values.Folded{+output,+line}: {==}

def pooled_columns(~Source: Data,~read: Source -> Nat -> Nat -> U32 -> U32 -> F32,~First: Data,~first_weight: First -> Nat -> U32 -> U32 -> F32,~first_bias: First -> Nat -> F32,~Second: Data,~second_weight: Second -> Nat -> U32 -> U32 -> F32,~second_bias: Second -> Nat -> F32,~operation: F32 -> F32 -> F32,count: Nat,+column: U32,+source: Source,+first: First,+second: Second,+pad: F32,+taps: Window.Taps,+cell: Values.Cell,+row: U32,
  +kernel_row: U32,+folded: Values.Folded<F32>) -> {Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,count,column,source,first,Values.Pooling{pad},taps,cell,row,kernel_row,folded) == Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,count,column,source,second,Values.Pooling{pad},taps,cell,row,kernel_row,folded) : Values.Folded<F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = Values.sample(~F32,~Source,~read,source,taps,cell,row,column,Values.start(~F32,Values.Pooling{pad}))
      +first_tap = Values.tapped(~F32,~operation,~Values.lane_update,Values.Pooling{pad},folded,value,first_weight(first,Values.channel_of(cell),kernel_row,column))
      +second_tap = Values.tapped(~F32,~operation,~Values.lane_update,Values.Pooling{pad},folded,value,second_weight(second,Values.channel_of(cell),kernel_row,column))
      Equal.trans(Values.Folded<F32>,Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(column + 1 : U32),source,first,Values.Pooling{pad},taps,cell,row,kernel_row,first_tap),Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(column + 1 : U32),source,first,Values.Pooling{pad},taps,cell,row,kernel_row,second_tap),Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,rest,(column + 1 : U32),source,second,Values.Pooling{pad},taps,cell,row,kernel_row,second_tap),
        Equal.cong(Values.Folded<F32>,Values.Folded<F32>,state => Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(column + 1 : U32),source,first,Values.Pooling{pad},taps,cell,row,kernel_row,state),first_tap,second_tap,
          pooled_tap(~operation,pad,folded,value,first_weight(first,Values.channel_of(cell),kernel_row,column),second_weight(second,Values.channel_of(cell),kernel_row,column))),
        pooled_columns(~Source,~read,~First,~first_weight,~first_bias,~Second,~second_weight,~second_bias,~operation,rest,(column + 1 : U32),source,first,second,pad,taps,cell,row,kernel_row,second_tap))

def pooled_rows(~Source: Data,~read: Source -> Nat -> Nat -> U32 -> U32 -> F32,~First: Data,~first_weight: First -> Nat -> U32 -> U32 -> F32,~first_bias: First -> Nat -> F32,~Second: Data,~second_weight: Second -> Nat -> U32 -> U32 -> F32,~second_bias: Second -> Nat -> F32,~operation: F32 -> F32 -> F32,count: Nat,+kernel_row: U32,+source: Source,+first: First,+second: Second,+pad: F32,+taps: Window.Taps,+cell: Values.Cell,
  +accumulator: F32) -> {Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,count,kernel_row,source,first,Values.Pooling{pad},taps,cell,accumulator) == Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,count,kernel_row,source,second,Values.Pooling{pad},taps,cell,accumulator) : F32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +first_row = Values.row_value(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,source,first,Values.Pooling{pad},taps,cell,kernel_row,accumulator)
      +second_row = Values.row_value(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,source,second,Values.Pooling{pad},taps,cell,kernel_row,accumulator)
      Equal.trans(F32,Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(kernel_row + 1 : U32),source,first,Values.Pooling{pad},taps,cell,first_row),Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(kernel_row + 1 : U32),source,first,Values.Pooling{pad},taps,cell,second_row),Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,rest,(kernel_row + 1 : U32),source,second,Values.Pooling{pad},taps,cell,second_row),
        Equal.cong(F32,F32,value => Values.rows(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,rest,(kernel_row + 1 : U32),source,first,Values.Pooling{pad},taps,cell,value),first_row,second_row,
          Equal.cong(Values.Folded<F32>,F32,folded => Values.closed(~F32,~operation,Values.Pooling{pad},folded),Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~First,~first_weight,U32.to_nat(Values.kernel_of(taps)),0,source,first,Values.Pooling{pad},taps,cell,Values.row_of(taps,cell,kernel_row),kernel_row,Values.Folded{accumulator,Values.start(~F32,Values.Pooling{pad})}),Values.columns(~F32,~operation,~Values.lane_update,~Source,~read,~Second,~second_weight,U32.to_nat(Values.kernel_of(taps)),0,source,second,Values.Pooling{pad},taps,cell,Values.row_of(taps,cell,kernel_row),kernel_row,Values.Folded{accumulator,Values.start(~F32,Values.Pooling{pad})}),
            pooled_columns(~Source,~read,~First,~first_weight,~first_bias,~Second,~second_weight,~second_bias,~operation,U32.to_nat(Values.kernel_of(taps)),0,source,first,second,pad,taps,cell,Values.row_of(taps,cell,kernel_row),kernel_row,
              Values.Folded{accumulator,Values.start(~F32,Values.Pooling{pad})}))),
        pooled_rows(~Source,~read,~First,~first_weight,~first_bias,~Second,~second_weight,~second_bias,~operation,rest,(kernel_row + 1 : U32),source,first,second,pad,taps,cell,second_row))

# The pool of a window is the same under any parameters.
def pooled_at(~Source: Data,~read: Source -> Nat -> Nat -> U32 -> U32 -> F32,~First: Data,~first_weight: First -> Nat -> U32 -> U32 -> F32,~first_bias: First -> Nat -> F32,~Second: Data,~second_weight: Second -> Nat -> U32 -> U32 -> F32,~second_bias: Second -> Nat -> F32,~operation: F32 -> F32 -> F32,+source: Source,+first: First,+second: Second,+pad: F32,+taps: Window.Taps,+cell: Values.Cell) -> {Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Source,~read,~First,~first_weight,~first_bias,source,first,Values.Pooling{pad},taps,cell) == Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Source,~read,~Second,~second_weight,~second_bias,source,second,Values.Pooling{pad},taps,cell) : F32}:
  pooled_rows(~Source,~read,~First,~first_weight,~first_bias,~Second,~second_weight,~second_bias,~operation,U32.to_nat(Values.kernel_of(taps)),0,source,first,second,pad,taps,cell,Values.start(~F32,Values.Pooling{pad}))

# ---- The parameter lanes of pooling ----

# The weights and the bias of a channel over parameter cells: its lane of the
# cell of its group.
def split_weight(source: Rows.Weights,split: Nat & Nat,kernel_row: U32,column: U32) -> F32:
  (group,lane) = split
  TileLaws.select_lane(lane,Rows.weight_read(source,group,kernel_row,column))

def lane_weight(source: Rows.Weights,channel: Nat,kernel_row: U32,column: U32) -> F32:
  split_weight(source,Nat.divmod(channel,8n),kernel_row,column)

def split_bias(source: Rows.Weights,split: Nat & Nat) -> F32:
  (group,lane) = split
  TileLaws.select_lane(lane,Rows.bias_read(source,group))

def lane_bias(source: Rows.Weights,channel: Nat) -> F32:
  split_bias(source,Nat.divmod(channel,8n))

def Same(cells: Rows.Weights,parameters: Rows.Weights,channels: Nat) -> Data:
  {cells == parameters : Rows.Weights}

def lane_split(+group: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) -> {Nat.divmod(Nat.add(lane,Nat.mul(group,8n)),8n) == (group,lane) : Nat & Nat}:
  Equal.trans(Nat & Nat,Nat.divmod(Nat.add(lane,Nat.mul(group,8n)),8n),Nat.divmod(Nat.add(Nat.mul(group,8n),lane),8n),(group,lane),
    Equal.cong(Nat,Nat & Nat,value => Nat.divmod(value,8n),Nat.add(lane,Nat.mul(group,8n)),Nat.add(Nat.mul(group,8n),lane),Order.commutative(lane,Nat.mul(group,8n))),
    Division.unique(Nat.add(Nat.mul(group,8n),lane),8n,group,lane,{==},{==},Order.lt_succ(lane,7n,small)))

def lane_weight_at(+cells: Rows.Weights,+parameters: Rows.Weights,+channels: Nat,+same: Same(cells,parameters,channels),+group: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),channels) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Spec.weights_groups(cells))) == True{} : Bool},+kernel_row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(kernel_row,U32.to_nat(Spec.weights_kernel(cells))) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(Spec.weights_kernel(cells))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Rows.weight_read(cells,group,U32.from_nat(kernel_row),U32.from_nat(column))) == lane_weight(parameters,Nat.add(lane,Nat.mul(group,8n)),U32.from_nat(kernel_row),U32.from_nat(column)) : F32}:
  %same : {TileLaws.select_lane(lane,Rows.weight_read(cells,group,U32.from_nat(kernel_row),U32.from_nat(column))) == lane_weight(_,Nat.add(lane,Nat.mul(group,8n)),U32.from_nat(kernel_row),U32.from_nat(column)) : F32}
  Equal.sym(F32,lane_weight(cells,Nat.add(lane,Nat.mul(group,8n)),U32.from_nat(kernel_row),U32.from_nat(column)),TileLaws.select_lane(lane,Rows.weight_read(cells,group,U32.from_nat(kernel_row),U32.from_nat(column))),
    Equal.cong(Nat & Nat,F32,split => split_weight(cells,split,U32.from_nat(kernel_row),U32.from_nat(column)),Nat.divmod(Nat.add(lane,Nat.mul(group,8n)),8n),(group,lane),lane_split(group,lane,small)))

def lane_bias_at(+cells: Rows.Weights,+parameters: Rows.Weights,+channels: Nat,+same: Same(cells,parameters,channels),+group: Nat,+lane: Nat,
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),channels) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Spec.weights_groups(cells))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Rows.bias_read(cells,group)) == lane_bias(parameters,Nat.add(lane,Nat.mul(group,8n))) : F32}:
  %same : {TileLaws.select_lane(lane,Rows.bias_read(cells,group)) == lane_bias(_,Nat.add(lane,Nat.mul(group,8n))) : F32}
  Equal.sym(F32,lane_bias(cells,Nat.add(lane,Nat.mul(group,8n))),TileLaws.select_lane(lane,Rows.bias_read(cells,group)),
    Equal.cong(Nat & Nat,F32,split => split_bias(cells,split),Nat.divmod(Nat.add(lane,Nat.mul(group,8n)),8n),(group,lane),lane_split(group,lane,small)))

# The pool over parameter lanes is the pool without parameters.
def pooled_spec(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32 -> F32,+source: Spec.Input<Logical>,+cells: Rows.Weights,+pad: F32,
  +taps: Window.Taps,+out_height: U32,+out_width: U32,+index: U32) ->
  {Spec.spec_value(~Logical,~original,~Rows.Weights,~lane_weight,~lane_bias,~operation,Spec.Spec{source,cells,Values.Pooling{pad},taps,out_height,out_width},index) ==
    Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation,Spec.Spec{source,Unit{},Values.Pooling{pad},taps,out_height,out_width},index) : F32}:
  pooled_at(~Spec.Input<Logical>,~Spec.input_read(~Logical,~original),~Rows.Weights,~lane_weight,~lane_bias,~Unit,~no_weight,~no_bias,~operation,source,cells,Unit{},pad,taps,
    Spec.output_of(U32.to_nat(Spec.channels_of(~Logical,source)),Spec.stride_of(taps),U32.to_nat(out_height),U32.to_nat(out_width),U32.to_nat(index)))

# ---- The pool of a window shape ----

# Over tiled values of [before.., lanes, height, width] with their lanes on
# the channels and the groups innermost, the pool holds the pool values.
def pooled_window(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+before: List<&2,U32>,+lanes: U32,+height: U32,
  +width: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+pad: F32,+kernel: U32,+stride: U32,
  +padding: U32,+out_height: U32,+out_width: U32,+fits: WindowPublic.Fits<limits,ShapeLaws.elements(before),lanes,height,width,kernel,stride,padding,out_height,out_width>,+workers: U32,
  +observation: Buffer.Observation<Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),0)>,+model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+in_workers: U32,
  +contract: Public.Contract(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),fill,Tensors.Usable{},in_workers},List.length(&2,U32,before),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Grid.Innermost{}})) ->
  Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},List.append(&2,U32,before,[lanes,out_height,out_width]),
    Tiled.Tiled{TiledPool.windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),workers,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{}),fill,Tensors.Usable{},in_workers}),List.length(&2,U32,before),lanes,Grid.Innermost{}}):
  Buffer.Observation{+parameter_model,+equation} = observation
  %equation : Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},List.append(&2,U32,before,[lanes,out_height,out_width]),
    Tiled.Tiled{Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{Window.Pooling{Tile.filled(pad)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},Tensors.grid_of(Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{})),Tasks.decide_whole(Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target()),workers,1,(U32.to_f32(kernel) * U32.to_f32(kernel) * 2.0 + U32.to_f32(kernel) + 1.0 : F32),(U32.to_f32(kernel) * U32.to_f32(kernel) * 13.0 + U32.to_f32(kernel) * 12.0 + 20.0 : F32),Window.extra_of(Window.Windowing{Window.Pooling{Tile.filled(pad)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},0)),PiecesModel.pack(~Tile.Tile8,model),_),Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Tiled.zero_tile(),Tensors.Usable{},workers},List.length(&2,U32,before),lanes,Grid.Innermost{}})
  Public.reread(~Spec.Spec<Logical,Rows.Weights>,~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Rows.Weights,~lane_weight,~lane_bias,~operation),~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),~(left => right => Global.Fixed(~Spec.Spec<Logical,Rows.Weights>,~Spec.Spec<Logical,Unit>,Spec.Spec{Spec.Input{logical,lanes,height,width},Rows.Weights{Buffer.tree(~Tile.Tile8,parameter_model),kernel,Tiled.groups(lanes)},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},left,right)),
    ~(left => right => fixed => index => Global.fixed_agrees(~F32,~Spec.Spec<Logical,Rows.Weights>,~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Rows.Weights,~lane_weight,~lane_bias,~operation),~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},Rows.Weights{Buffer.tree(~Tile.Tile8,parameter_model),kernel,Tiled.groups(lanes)},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},
      cell => pooled_spec(~Logical,~original,~operation,Spec.Input{logical,lanes,height,width},Rows.Weights{Buffer.tree(~Tile.Tile8,parameter_model),kernel,Tiled.groups(lanes)},pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,cell),left,right,fixed,index)),
    Spec.Spec{Spec.Input{logical,lanes,height,width},Rows.Weights{Buffer.tree(~Tile.Tile8,parameter_model),kernel,Tiled.groups(lanes)},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},KernelGemm.Both{{==},{==}},List.append(&2,U32,before,[lanes,out_height,out_width]),
    Tiled.Tiled{Tensors.Tensor{Window.windowed_decided(~TiledMap.pairs(~operation),Window.Windowing{Window.Pooling{Tile.filled(pad)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},Tensors.grid_of(Tiled.cell_shape(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before),Grid.Innermost{})),Tasks.decide_whole(Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target()),workers,1,(U32.to_f32(kernel) * U32.to_f32(kernel) * 2.0 + U32.to_f32(kernel) + 1.0 : F32),(U32.to_f32(kernel) * U32.to_f32(kernel) * 13.0 + U32.to_f32(kernel) * 12.0 + 20.0 : F32),Window.extra_of(Window.Windowing{Window.Pooling{Tile.filled(pad)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},0)),PiecesModel.pack(~Tile.Tile8,model),Buffer.pack(~Tile.Tile8,parameter_model)),Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Tiled.zero_tile(),Tensors.Usable{},workers},List.length(&2,U32,before),lanes,Grid.Innermost{}},
    WindowPublic.window_contract(~Logical,~original,~Rows.Weights,~lane_weight,~lane_bias,~Same,
      ~(cells => parameters => channels => same => group => lane => small => inside => group_inside => kernel_row => column => row_inside => column_inside =>
        lane_weight_at(cells,parameters,channels,same,group,lane,small,inside,group_inside,kernel_row,column,row_inside,column_inside)),
      ~(cells => parameters => channels => same => group => lane => small => inside => group_inside => lane_bias_at(cells,parameters,channels,same,group,lane,small,inside,group_inside)),
      ~operation,logical,before,lanes,height,width,limits,default_capacity,Buffer.tree(~Tile.Tile8,parameter_model),Rows.Weights{Buffer.tree(~Tile.Tile8,parameter_model),kernel,Tiled.groups(lanes)},Values.Pooling{pad},kernel,stride,padding,{==},
      out_height,out_width,Window.Pooling{Tile.filled(pad)},{==},fits,model,fill,in_workers,contract,parameter_model,{==},workers,(U32.to_f32(kernel) * U32.to_f32(kernel) * 2.0 + U32.to_f32(kernel) + 1.0 : F32),(U32.to_f32(kernel) * U32.to_f32(kernel) * 13.0 + U32.to_f32(kernel) * 12.0 + 20.0 : F32),Window.extra_of(Window.Windowing{Window.Pooling{Tile.filled(pad)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,Grid.grid(Tiled.cell_shape(List.append(&2,U32,before,[lanes,out_height,out_width]),List.length(&2,U32,before),Grid.Innermost{}),Grid.piece_target())},0)))

# ---- Shapes of rank three and more ----

# A shape of rank three or more: the axes before its lane axis, the lane axis
# and the two pooled axes.
type Split<-shape: List<&2,U32>> is Data:
  Split{before: List<&2,U32>,lanes: U32,height: U32,width: U32,exact: {List.append(&2,U32,before,[lanes,height,width]) == shape : List<&2,U32>}}

# A split shape with one more axis in front.
def split_prepended(+head: U32,+rest: List<&2,U32>,+split: Split<rest>) -> Split<Con{head,rest}>:
  Split{+before,+lanes,+height,+width,+exact} = split
  Split{Con{head,before},lanes,height,width,Equal.cong(List<&2,U32>,List<&2,U32>,tail => Con{head,tail},List.append(&2,U32,before,[lanes,height,width]),rest,exact)}

def split_of(+shape: List<&2,U32>,+ranked: {Nat.is_le(3n,List.length(&2,U32,shape)) == True{} : Bool}) -> Split<shape>:
  match shape:
    case Nil{}: Empty.absurd(Split<[]>,WordEncoding.false_true(ranked))
    case Con{+lanes,+rest}:
      match rest:
        case Nil{}: Empty.absurd(Split<[lanes]>,WordEncoding.false_true(ranked))
        case Con{+height,+after}:
          match after:
            case Nil{}: Empty.absurd(Split<[lanes,height]>,WordEncoding.false_true(ranked))
            case Con{+width,+last}:
              match last:
                case Nil{}: Split{[],lanes,height,width,{==}}
                case Con{+extra,+tail}:
                  split_prepended(lanes,Con{height,Con{width,Con{extra,tail}}},split_of(Con{height,Con{width,Con{extra,tail}}},Order.zero_le(List.length(&2,U32,tail))))

# The lane axis of a split shape is its axis of lanes.
def split_axis(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {Tiled.lane_axis(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width]))) == List.length(&2,U32,before) : Nat}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),Nat.add(3n,List.length(&2,U32,before)),WindowPublic.split_length(before,lanes,height,width)) : {Tiled.lane_axis(_) == List.length(&2,U32,before) : Nat}
  {==}

# A pool reads a split shape as it is.
def raised_split(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {raised(List.append(&2,U32,before,[lanes,height,width])) == List.append(&2,U32,before,[lanes,height,width]) : List<&2,U32>}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),Nat.add(3n,List.length(&2,U32,before)),WindowPublic.split_length(before,lanes,height,width)) :
    {Bool.pick(List<&2,U32>,Nat.is_le(3n,_),List.append(&2,U32,before,[lanes,height,width]),Con{1,List.append(&2,U32,before,[lanes,height,width])}) == List.append(&2,U32,before,[lanes,height,width]) : List<&2,U32>}
  %Equal.sym(Bool,Nat.is_le(0n,List.length(&2,U32,before)),True{},Order.zero_le(List.length(&2,U32,before))) : {Bool.pick(List<&2,U32>,_,List.append(&2,U32,before,[lanes,height,width]),Con{1,List.append(&2,U32,before,[lanes,height,width])}) == List.append(&2,U32,before,[lanes,height,width]) : List<&2,U32>}
  {==}

def split_take(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)) == before : List<&2,U32>}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}:
      Equal.cong(List<&2,U32>,List<&2,U32>,taken => Con{head,taken},List.take(&2,U32,List.append(&2,U32,tail,[lanes,height,width]),List.length(&2,U32,tail)),tail,
        split_take(tail,lanes,height,width))

def split_leading(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),1n+List.length(&2,U32,before)) == List.append(&2,U32,before,[lanes]) : List<&2,U32>}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}:
      Equal.cong(List<&2,U32>,List<&2,U32>,taken => Con{head,taken},List.take(&2,U32,List.append(&2,U32,tail,[lanes,height,width]),1n+List.length(&2,U32,tail)),
        List.append(&2,U32,tail,[lanes]),split_leading(tail,lanes,height,width))

def split_height(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {Tensors.known_count(List.get(&2,U32,List.append(&2,U32,before,[lanes,height,width]),1n+List.length(&2,U32,before))) == height : U32}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}: split_height(tail,lanes,height,width)

def split_width(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {Tensors.known_count(List.get(&2,U32,List.append(&2,U32,before,[lanes,height,width]),1n+1n+List.length(&2,U32,before))) == width : U32}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}: split_width(tail,lanes,height,width)

def split_rows(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),2n)) == height : U32}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),Nat.add(3n,List.length(&2,U32,before)),WindowPublic.split_length(before,lanes,height,width)) : {Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(_,2n)) == height : U32}
  split_height(before,lanes,height,width)

def split_columns(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32) -> {Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),1n)) == width : U32}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),Nat.add(3n,List.length(&2,U32,before)),WindowPublic.split_length(before,lanes,height,width)) : {Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(_,1n)) == width : U32}
  split_width(before,lanes,height,width)

# The pool of a split shape runs over its last two axes.
def split_pool(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) -> {Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding) == Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)} : Access.Pool}:
  +rank = List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width]))
  +rows = Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(rank,2n))
  +columns = Layout.extent_at(List.append(&2,U32,before,[lanes,height,width]),Nat.sub(rank,1n))
  %Equal.sym(U32,rows,height,split_rows(before,lanes,height,width)) :
    {Access.Pool{_,columns,kernel,stride,padding,Window.out_extent(_,kernel,stride,padding),Window.out_extent(columns,kernel,stride,padding)} == Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)} : Access.Pool}
  %Equal.sym(U32,columns,width,split_columns(before,lanes,height,width)) :
    {Access.Pool{height,_,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(_,kernel,stride,padding)} == Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)} : Access.Pool}
  {==}

def appended_single(+before: List<&2,U32>,+lanes: U32,+rest: List<&2,U32>) ->
  {List.append(&2,U32,List.append(&2,U32,before,[lanes]),rest) == List.append(&2,U32,before,Con{lanes,rest}) : List<&2,U32>}:
  match before:
    case Nil{}: {==}
    case Con{+head,+tail}:
      Equal.cong(List<&2,U32>,List<&2,U32>,joined => Con{head,joined},List.append(&2,U32,List.append(&2,U32,tail,[lanes]),rest),List.append(&2,U32,tail,Con{lanes,rest}),
        appended_single(tail,lanes,rest))

# A pool keeps the axes before the last two and replaces them with the rows
# and columns it leaves.
def split_kept(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+rows: U32,+columns: U32) ->
  {List.append(&2,U32,List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),Nat.sub(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),2n)),[rows,columns]) == List.append(&2,U32,before,[lanes,rows,columns]) : List<&2,U32>}:
  %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),Nat.add(3n,List.length(&2,U32,before)),WindowPublic.split_length(before,lanes,height,width)) :
    {List.append(&2,U32,List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),Nat.sub(_,2n)),[rows,columns]) == List.append(&2,U32,before,[lanes,rows,columns]) : List<&2,U32>}
  %Equal.sym(List<&2,U32>,List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),1n+List.length(&2,U32,before)),List.append(&2,U32,before,[lanes]),split_leading(before,lanes,height,width)) :
    {List.append(&2,U32,_,[rows,columns]) == List.append(&2,U32,before,[lanes,rows,columns]) : List<&2,U32>}
  appended_single(before,lanes,[rows,columns])

def split_output(+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) -> {Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}) == List.append(&2,U32,before,[lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]) : List<&2,U32>}:
  split_kept(before,lanes,height,width,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding))

# A leading axis of extent 1 keeps the count.
def leading_count_when(zero: Bool,+rest: List<&2,U32>,+classified: {Shapes.has_zero(rest) == zero : Bool}) -> {Tensors.count_of(Con{1,rest}) == Tensors.count_of(rest) : Maybe<&2,U32>}:
  match zero:
    case True{}:
      %Equal.sym(Bool,Shapes.has_zero(rest),True{},classified) :
        {Tensors.allocatable(Shapes.count_nonempty(U32.is_zero(1) || _,Con{1,rest})) == Tensors.allocatable(Shapes.count_nonempty(_,rest)) : Maybe<&2,U32>}
      {==}
    case False{}:
      %Equal.sym(Bool,Shapes.has_zero(rest),False{},classified) :
        {Tensors.allocatable(Shapes.count_nonempty(U32.is_zero(1) || _,Con{1,rest})) == Tensors.allocatable(Shapes.count_nonempty(_,rest)) : Maybe<&2,U32>}
      {==}

def leading_count(+rest: List<&2,U32>) -> {Tensors.count_of(Con{1,rest}) == Tensors.count_of(rest) : Maybe<&2,U32>}:
  leading_count_when(Shapes.has_zero(rest),rest,{==})

# ---- The routes of the pool ----

def arranged_cells(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tiled: Tiled.Tiled,+axis: Nat,+position: Grid.Position,
  +arranged: Relane.Arranged(~Logical,~original,logical,values,tiled,axis,position)) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(tiled,axis,position)),axis,Grid.lanes_at(values,axis),position}):
  KernelGemm.Both{+eta,+held} = arranged
  held

# The machine facts of the pool of a split shape.
def fits_split(+limits: Machine.Limits,+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +fits: PoolFits(limits,List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding)) -> WindowPublic.Fits<limits,ShapeLaws.elements(before),lanes,height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)>:
  +own = {%raised_split(before,lanes,height,width) : pool_fits(limits,ShapeLaws.elements(List.take(&2,U32,_,Tiled.lane_axis(List.length(&2,U32,_)))),Grid.lanes_at(_,Tiled.lane_axis(List.length(&2,U32,_))),Window.geometry(_,kernel,stride,padding)) fits : pool_fits(limits,ShapeLaws.elements(List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),Tiled.lane_axis(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width]))))),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),Tiled.lane_axis(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])))),Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))}
  +placed = {%split_axis(before,lanes,height,width) : pool_fits(limits,ShapeLaws.elements(List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),_)),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_),Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))
    own : pool_fits(limits,ShapeLaws.elements(List.take(&2,U32,List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before))),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))}
  +leading = {%split_take(before,lanes,height,width) : pool_fits(limits,ShapeLaws.elements(_),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))
    placed : pool_fits(limits,ShapeLaws.elements(before),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))}
  +channels = {%WindowPublic.split_lanes(before,lanes,[height,width]) : pool_fits(limits,ShapeLaws.elements(before),_,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))
    leading : pool_fits(limits,ShapeLaws.elements(before),lanes,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding))}
  %split_pool(before,lanes,height,width,kernel,stride,padding) : pool_fits(limits,ShapeLaws.elements(before),lanes,_)
  channels

# The pool of [before.., lanes, height, width] runs on its cells with their
# lanes on the channels and the groups innermost.
def split_packed(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+workers: U32,+model: PiecesModel.Model<Tile.Tile8>,+stored: List<&2,U32>,+fill: Tile.Tile8,+in_workers: U32,+axis: Nat,
  +in_lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position}),
  +nonzero: {Shapes.has_zero(List.append(&2,U32,before,[lanes,height,width])) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(List.append(&2,U32,before,[lanes,height,width]))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding)) ->
  Pooled(~operation,~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),pad,kernel,stride,padding,TiledPool.pooled(~operation,pad,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding),List.append(&2,U32,before,[lanes,height,width]),workers,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position})):
  +ranked = {%raised_split(before,lanes,height,width) : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +held = arranged_cells(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},List.length(&2,U32,before),Grid.Innermost{},
    Relane.arranged_held(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},List.length(&2,U32,before),Grid.Innermost{},limits,default_capacity,WindowPublic.split_inside(before,lanes,height,width),CountCompletion.positive_elements(List.append(&2,U32,before,[lanes,height,width]),nonzero),ranked,contract))
  +window_fits = fits_split(limits,before,lanes,height,width,kernel,stride,padding,fits)
  %Equal.sym(Access.Pool,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},split_pool(before,lanes,height,width,kernel,stride,padding)) :
    Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),pool_spec(~Logical,logical,pool_channels(List.append(&2,U32,before,[lanes,height,width])),_,pad),Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),_),TiledPool.pooled(~operation,pad,_,List.append(&2,U32,before,[lanes,height,width]),workers,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position}))
  %Equal.sym(List<&2,U32>,raised(List.append(&2,U32,before,[lanes,height,width])),List.append(&2,U32,before,[lanes,height,width]),raised_split(before,lanes,height,width)) :
    Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),pool_spec(~Logical,logical,Grid.lanes_at(_,Tiled.lane_axis(List.length(&2,U32,_))),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},pad),Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),TiledPool.pooled(~operation,pad,Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},List.append(&2,U32,before,[lanes,height,width]),workers,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position}))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,List.append(&2,U32,before,[lanes,height,width]))),List.length(&2,U32,before),split_axis(before,lanes,height,width)) :
    Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_),height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_))},Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),Tiled.Tiled{TiledPool.windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_))},Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_),Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding),Tiled.cell_shape(Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),_,Grid.Innermost{}),workers,Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},_,Grid.Innermost{}))),_,Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),_),Grid.Innermost{}})
  %Equal.sym(U32,Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),lanes,WindowPublic.split_lanes(before,lanes,[height,width])) :
    Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,_,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(_)},Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),Tiled.Tiled{TiledPool.windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(_)},_,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding),Tiled.cell_shape(Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),List.length(&2,U32,before),Grid.Innermost{}),workers,Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},List.length(&2,U32,before),Grid.Innermost{}))),List.length(&2,U32,before),_,Grid.Innermost{}})
  %Equal.sym(List<&2,U32>,Window.output_shape(List.append(&2,U32,before,[lanes,height,width]),Access.Pool{height,width,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}),List.append(&2,U32,before,[lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),split_output(before,lanes,height,width,kernel,stride,padding)) :
    Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},_,Tiled.Tiled{TiledPool.windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding),Tiled.cell_shape(_,List.length(&2,U32,before),Grid.Innermost{}),workers,Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},List.length(&2,U32,before),Grid.Innermost{}))),List.length(&2,U32,before),lanes,Grid.Innermost{}})
  Relane.modeled(~Logical,~original,(laid => Public.Contract(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),Spec.Spec{Spec.Input{logical,lanes,height,width},Unit{},Values.Pooling{pad},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)},List.append(&2,U32,before,[lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),Tiled.Tiled{TiledPool.windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding),Tiled.cell_shape(List.append(&2,U32,before,[lanes,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),List.length(&2,U32,before),Grid.Innermost{}),workers,Tiled.cells_of(laid)),List.length(&2,U32,before),lanes,Grid.Innermost{}})),
    logical,List.append(&2,U32,before,[lanes,height,width]),Tiled.cells_of(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},in_workers},axis,in_lanes,position},List.length(&2,U32,before),Grid.Innermost{})),List.length(&2,U32,before),Grid.lanes_at(List.append(&2,U32,before,[lanes,height,width]),List.length(&2,U32,before)),Grid.Innermost{},held,
    cell_model => cell_fill => cell_workers => cells => pooled_window(~operation,~Logical,~original,logical,before,lanes,height,width,limits,default_capacity,pad,kernel,stride,padding,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding),window_fits,workers,
      Buffer.observe(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),0)),cell_model,cell_fill,cell_workers,cells))

def split_pooled(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+before: List<&2,U32>,+lanes: U32,+height: U32,+width: U32,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+workers: U32,tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),tiled),
  +nonzero: {Shapes.has_zero(List.append(&2,U32,before,[lanes,height,width])) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(List.append(&2,U32,before,[lanes,height,width]))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding)) ->
  Pooled(~operation,~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),pad,kernel,stride,padding,TiledPool.pooled(~operation,pad,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding),List.append(&2,U32,before,[lanes,height,width]),workers,tiled)):
  Public.by_packed(~Logical,~original,(laid => Pooled(~operation,~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),pad,kernel,stride,padding,TiledPool.pooled(~operation,pad,Window.geometry(List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding),List.append(&2,U32,before,[lanes,height,width]),workers,laid))),
    logical,List.append(&2,U32,before,[lanes,height,width]),tiled,contract,
    packed_model => packed_shape => packed_fill => packed_workers => packed_axis => packed_lanes => packed_position => packed =>
      split_packed(~operation,~Logical,~original,logical,before,lanes,height,width,pad,kernel,stride,padding,limits,default_capacity,workers,packed_model,packed_shape,packed_fill,packed_workers,packed_axis,
        packed_lanes,packed_position,packed,nonzero,rank_room,fits))

# The pool of a shape of rank three or more.
def split_contract(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+workers: U32,
  tiled: Tiled.Tiled,+split: Split<shape>,+contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding)) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.pooled(~operation,pad,Window.geometry(shape,kernel,stride,padding),shape,workers,tiled)):
  Split{+before,+lanes,+height,+width,+exact} = split
  +back = Equal.sym(List<&2,U32>,List.append(&2,U32,before,[lanes,height,width]),shape,exact)
  %exact : Pooled(~operation,~Logical,~original,logical,_,pad,kernel,stride,padding,TiledPool.pooled(~operation,pad,Window.geometry(_,kernel,stride,padding),_,workers,tiled))
  split_pooled(~operation,~Logical,~original,logical,before,lanes,height,width,pad,kernel,stride,padding,limits,default_capacity,workers,tiled,
    {%back : Public.Contract(~Logical,~original,logical,_,tiled) contract : Public.Contract(~Logical,~original,logical,List.append(&2,U32,before,[lanes,height,width]),tiled)},
    {%back : {Shapes.has_zero(_) == False{} : Bool} nonzero : {Shapes.has_zero(List.append(&2,U32,before,[lanes,height,width])) == False{} : Bool}},
    {%back : {Nat.is_le(List.length(&2,U32,raised(_)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,raised(List.append(&2,U32,before,[lanes,height,width]))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%back : PoolFits(limits,_,kernel,stride,padding) fits : PoolFits(limits,List.append(&2,U32,before,[lanes,height,width]),kernel,stride,padding)})

# The route of a shape of rank three or more: its cells, or a failure.
def routed_contract(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,route: Bool,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+workers: U32,tiled: Tiled.Tiled,+ranked: {Nat.is_le(3n,List.length(&2,U32,shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),
  +accepted: {route == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.routed(~operation,name,route,pad,Window.geometry(shape,kernel,stride,padding),shape,workers,tiled)):
  match route:
    case True{}: split_contract(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,limits,default_capacity,workers,tiled,split_of(shape,ranked),contract,nonzero,rank_room,fits)
    case False{}: Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,Tiled.rejected(name,tiled)),WordEncoding.false_true(accepted))

# The pool on the cells of tiled values of rank three or more.
def cells_pool_contract(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},tiled: Tiled.Tiled,+ranked: {Nat.is_le(3n,List.length(&2,U32,shape)) == True{} : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.cells_pool(~operation,name,pad,tiled,kernel,stride,padding)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+stored,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
          %Equal.sym(List<&2,U32>,Tiled.logical_shape(stored,axis,lanes,position),shape,restored) :
            Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.ranked(~operation,name,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{pieces,stored,fill,Tensors.Usable{},workers},axis,lanes,position},_,Tensors.Usable{},workers))
          routed_contract(~operation,~Logical,~original,Tensors.usable(Tensors.Usable{}) && Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) && Nat.is_le(3n,List.length(&2,U32,shape)),name,logical,shape,pad,kernel,stride,padding,limits,default_capacity,workers,Tiled.Tiled{Tensors.Tensor{pieces,stored,fill,Tensors.Usable{},workers},axis,lanes,position},ranked,contract,nonzero,rank_room,fits,
            Boolean.and_true(Window.valid(shape,Window.geometry(shape,kernel,stride,padding)),Nat.is_le(3n,List.length(&2,U32,shape)),window_valid,ranked))
        case Tensors.Tensor{pieces,+stored,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,
            TiledPool.cells_pool(~operation,name,pad,Tiled.Tiled{Tensors.Tensor{pieces,stored,fill,Tensors.Failed{problem},workers},axis,lanes,position},kernel,stride,padding)),contract)

# A valid window of [height,width] is a valid window of [1,height,width].
def planar_valid(+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +valid: {Window.valid([height,width],Window.geometry([height,width],kernel,stride,padding)) == True{} : Bool}) ->
  {Window.valid([1,height,width],Window.geometry([1,height,width],kernel,stride,padding)) == True{} : Bool}:
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of([1,height,Window.out_extent(width,kernel,stride,padding)]),Tensors.count_of([height,Window.out_extent(width,kernel,stride,padding)]),leading_count([height,Window.out_extent(width,kernel,stride,padding)])) :
    {Operations.pool_geometry(Window.geometry([height,width],kernel,stride,padding)) && U32.is_lt(0,height) && U32.is_lt(0,width) && Views.has_count(_) && Views.has_count(Tensors.count_of([1,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)])) == True{} : Bool}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of([1,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),Tensors.count_of([Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),leading_count([Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)])) :
    {Operations.pool_geometry(Window.geometry([height,width],kernel,stride,padding)) && U32.is_lt(0,height) && U32.is_lt(0,width) && Views.has_count(Tensors.count_of([height,Window.out_extent(width,kernel,stride,padding)])) && Views.has_count(_) == True{} : Bool}
  valid

# A valid pool of rank two pools the tensor with a leading axis of extent 1
# and drops that axis from its output.
def planar_contract(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,shape: List<&2,U32>,valid: Bool,+name: String,+logical: Logical,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+model: PiecesModel.Model<Tile.Tile8>,+stored: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position}),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool},
  +below_three: {Nat.is_le(3n,List.length(&2,U32,shape)) == False{} : Bool},+accepted: {valid == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.planar(~operation,name,shape,valid,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})):
  match shape valid:
    case Con{+height,Con{+width,Nil{}}} True{}:
      +lifted = TiledReshape.reshaped(~Logical,~original,logical,limits,[height,width],[1,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},nonzero,leading_count([height,width]),
        Order.transitive(2n,3n,U32.to_nat(Machine.capacity(limits)),{==},rank_room),rank_room,default_capacity,contract)
      +pooled = cells_pool_contract(~operation,~Logical,~original,name,logical,[1,height,width],pad,kernel,stride,padding,limits,default_capacity,TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},[1,height,width]),{==},lifted,
        nonzero,rank_room,fits,planar_valid(height,width,kernel,stride,padding,window_valid))
      TiledReshape.reshaped(~Spec.Spec<Logical,Unit>,~Spec.spec_value(~Logical,~original,~Unit,~no_weight,~no_bias,~operation),pool_spec(~Logical,logical,pool_channels([1,height,width]),Window.geometry([1,height,width],kernel,stride,padding),pad),limits,
        Window.output_shape([1,height,width],Window.geometry([1,height,width],kernel,stride,padding)),Window.output_shape([height,width],Window.geometry([height,width],kernel,stride,padding)),
        TiledPool.cells_pool(~operation,name,pad,TiledLayout.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},[1,height,width]),kernel,stride,padding),out_nonzero,
        Equal.sym(Maybe<&2,U32>,Tensors.count_of([1,Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),Tensors.count_of([Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)]),leading_count([Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)])),rank_room,Order.transitive(2n,3n,U32.to_nat(Machine.capacity(limits)),{==},rank_room),default_capacity,pooled)
    case Con{+height,Con{+width,Nil{}}} False{}:
      Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.planar(~operation,name,shape,valid,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})),WordEncoding.false_true(accepted))
    case Nil{} valid:
      Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.planar(~operation,name,shape,valid,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})),WordEncoding.false_true(two_axes))
    case Con{+first,Nil{}} valid:
      Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.planar(~operation,name,shape,valid,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})),WordEncoding.false_true(two_axes))
    case Con{+first,Con{+second,Con{+third,rest}}} valid:
      Empty.absurd(Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.planar(~operation,name,shape,valid,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})),WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(0n,List.length(&2,U32,rest)),True{},Equal.sym(Bool,Nat.is_le(0n,List.length(&2,U32,rest)),False{},below_three),Order.zero_le(List.length(&2,U32,rest)))))

def dispatched_contract(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,three: Bool,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+model: PiecesModel.Model<Tile.Tile8>,+stored: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,+classified: {Nat.is_le(3n,List.length(&2,U32,shape)) == three : Bool},
  +contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position}),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.dispatched(~operation,name,three,pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},shape,Tensors.Usable{},workers)):
  match three:
    case True{}: routed_contract(~operation,~Logical,~original,Tensors.usable(Tensors.Usable{}) && Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) && Nat.is_le(3n,List.length(&2,U32,shape)),name,logical,shape,pad,kernel,stride,padding,limits,default_capacity,workers,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},classified,contract,nonzero,rank_room,fits,Boolean.and_true(Window.valid(shape,Window.geometry(shape,kernel,stride,padding)),Nat.is_le(3n,List.length(&2,U32,shape)),window_valid,classified))
    case False{}:
      planar_contract(~operation,~Logical,~original,shape,Tensors.usable(Tensors.Usable{}) && Window.valid(shape,Window.geometry(shape,kernel,stride,padding)),name,logical,pad,kernel,stride,padding,limits,default_capacity,model,stored,fill,workers,axis,lanes,position,contract,
        nonzero,rank_room,fits,out_nonzero,window_valid,two_axes,classified,window_valid)

def pool2d_packed(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+model: PiecesModel.Model<Tile.Tile8>,+stored: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,+contract: Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position}),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.pool2d(~operation,name,pad,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},kernel,stride,padding)):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(stored,axis,lanes,position),shape,restored) :
    Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.dispatched(~operation,name,Nat.is_le(3n,List.length(&2,U32,_)),pad,kernel,stride,padding,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},_,Tensors.Usable{},workers))
  dispatched_contract(~operation,~Logical,~original,Nat.is_le(3n,List.length(&2,U32,shape)),name,logical,shape,pad,kernel,stride,padding,limits,default_capacity,model,stored,fill,workers,axis,lanes,position,{==},contract,
    nonzero,rank_room,fits,out_nonzero,window_valid,two_axes)

def pool2d_described(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},-tiled: Tiled.Tiled,+described: Public.Packed<tiled>,+contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.pool2d(~operation,name,pad,tiled,kernel,stride,padding)):
  Public.Packed{+model,+stored,+fill,+workers,+axis,+lanes,+position,+equation} = described
  +back = Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position},tiled,equation)
  %equation : Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.pool2d(~operation,name,pad,_,kernel,stride,padding))
  pool2d_packed(~operation,~Logical,~original,name,logical,shape,pad,kernel,stride,padding,limits,default_capacity,model,stored,fill,workers,axis,lanes,position,
    {%back : Public.Contract(~Logical,~original,logical,shape,_) contract : Public.Contract(~Logical,~original,logical,shape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),stored,fill,Tensors.Usable{},workers},axis,lanes,position})},nonzero,rank_room,fits,out_nonzero,window_valid,two_axes)

# The pool of tiled values of the contract of a nonempty shape of rank two or
# more and a valid window, with a nonempty output that fits the machine: the
# result holds at every index the pool of its window: rows then
# columns in increasing index, folded from pad, of the channel and the image
# of the index.
def pool2d(~operation: F32 -> F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+name: String,+logical: Logical,+shape: List<&2,U32>,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Pooled(~operation,~Logical,~original,logical,shape,pad,kernel,stride,padding,TiledPool.pool2d(~operation,name,pad,tiled,kernel,stride,padding)):
  pool2d_described(~operation,~Logical,~original,name,logical,shape,pad,kernel,stride,padding,limits,default_capacity,tiled,Public.packed(~Logical,~original,logical,shape,tiled,contract),contract,
    nonzero,rank_room,fits,out_nonzero,window_valid,two_axes)

# Max pooling: at every index the largest of its window, from -inf.
def max_pool2d(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,tiled: Tiled.Tiled,+kernel: U32,+stride: U32,+padding: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+contract: Public.Contract(~Logical,~original,logical,shape,tiled),+nonzero: {Shapes.has_zero(shape) == False{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,raised(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: PoolFits(limits,shape,kernel,stride,padding),+out_nonzero: {Shapes.has_zero(Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding))) == False{} : Bool},
  +window_valid: {Window.valid(shape,Window.geometry(shape,kernel,stride,padding)) == True{} : Bool},+two_axes: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Pooled(~F32.max,~Logical,~original,logical,shape,F32.div(F32.neg(1.0),0.0),kernel,stride,padding,F.max_pool2d_with(tiled,kernel,stride,padding)):
  pool2d(~F32.max,~Logical,~original,"max_pool2d",logical,shape,F32.div(F32.neg(1.0),0.0),kernel,stride,padding,limits,default_capacity,tiled,contract,nonzero,rank_room,fits,
    out_nonzero,window_valid,two_axes)
