# The values of a window operation over logical values of [images, channels,
# height, width]: the value at an index of [images, channels, out_height,
# out_width] is the window value of its image, its channel and its output row
# and column. The cells over tiles of a window operation whose input cells
# hold the lifted input hold at every index below the output cells the
# lifted values of the window.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_cells.bend as Cells
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree
import ./tensor_window_values.bend as Values
import ./tensor_window_rows.bend as Rows
import ./tensor_window_piece.bend as Piece
import ./tensor_window_task.bend as Task
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_padding.bend as Padding
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Boolean
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order

# ---- Index arithmetic ----

def parted_quotient(+quotient: Nat,+remainder: Nat,+divisor: Nat,+small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.mul(quotient,divisor),remainder),divisor) == quotient : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(quotient,divisor),remainder),divisor),(quotient,remainder),
    Division.unique(Nat.add(Nat.mul(quotient,divisor),remainder),divisor,quotient,remainder,{==},Representation.below_positive(remainder,divisor,small),small))

def parted_remainder(+quotient: Nat,+remainder: Nat,+divisor: Nat,+small: {Nat.is_lt(remainder,divisor) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.mul(quotient,divisor),remainder),divisor) == remainder : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(quotient,divisor),remainder),divisor),(quotient,remainder),
    Division.unique(Nat.add(Nat.mul(quotient,divisor),remainder),divisor,quotient,remainder,{==},Representation.below_positive(remainder,divisor,small),small))

# A row of a plane after whole planes.
def planar(+outer: Nat,+height: Nat,+row: Nat,+width: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(outer,height),row),width),column) == Nat.add(Nat.mul(outer,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(outer,height),row),width),column),Nat.add(Nat.add(Nat.mul(Nat.mul(outer,height),width),Nat.mul(row,width)),column),Nat.add(Nat.mul(outer,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),
    Equal.cong(Nat,Nat,value => Nat.add(value,column),Nat.mul(Nat.add(Nat.mul(outer,height),row),width),Nat.add(Nat.mul(Nat.mul(outer,height),width),Nat.mul(row,width)),RowMajor.distrib(Nat.mul(outer,height),row,width)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(outer,height),width),Nat.mul(row,width)),column),Nat.add(Nat.mul(Nat.mul(outer,height),width),Nat.add(Nat.mul(row,width),column)),Nat.add(Nat.mul(outer,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),
    Addition.associative(Nat.mul(Nat.mul(outer,height),width),Nat.mul(row,width),column),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.add(Nat.mul(row,width),column)),Nat.mul(Nat.mul(outer,height),width),Nat.mul(outer,Nat.mul(height,width)),RowMajor.assoc(outer,height,width))))

# Positions of whole planes regrouped.
def positions_grouped(+images: Nat,+height: Nat,+width: Nat,+outer: Nat,+inside: {Nat.is_lt(outer,Nat.mul(Nat.mul(images,height),width)) == True{} : Bool}) ->
  {Nat.is_lt(outer,Nat.mul(images,Nat.mul(height,width))) == True{} : Bool}:
  %RowMajor.assoc(images,height,width) : {Nat.is_lt(outer,_) == True{} : Bool}
  inside

# ---- The values ----

# The logical input of [images, channels, height, width].
type Input<-Logical: Data> is Data:
  Input{logical: Logical,channels: U32,height: U32,width: U32}

def input_index(+image: Nat,+channel: Nat,+row: Nat,+column: Nat,+channels: Nat,+height: Nat,+width: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),height),row),width),column)

def input_read(~Logical: Data,~original: Logical -> U32 -> F32,source: Input<Logical>,image: Nat,channel: Nat,row: U32,column: U32) -> F32:
  Input{logical,+channels,+height,+width} = source
  original(logical,U32.from_nat(input_index(image,channel,U32.to_nat(row),U32.to_nat(column),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width))))

def channels_of(~Logical: Data,source: Input<Logical>) -> U32:
  Input{logical,channels,height,width} = source
  channels

def stride_of(taps: Window.Taps) -> U32:
  Window.Taps{kernel,stride,padding,height,width,groups} = taps
  stride

# A window operation over the input with the parameters of its weighting.
type Spec<-Logical: Data,-Parameters: Data> is Data:
  Spec{source: Input<Logical>,parameters: Parameters,kind: Values.Sampling<F32>,taps: Window.Taps,out_height: U32,out_width: U32}

# The window of an index of [images, channels, out_height, out_width].
def output_of(+channels: Nat,+stride: U32,+out_height: Nat,+out_width: Nat,+index: Nat) -> Values.Cell:
  +plane = Nat.mul(out_height,out_width)
  +line = Nat.div(index,plane)
  +position = Nat.mod(index,plane)
  Values.Cell{Nat.div(line,channels),Nat.mod(line,channels),(U32.from_nat(Nat.div(position,out_width)) * stride : U32),
    (U32.from_nat(Nat.mod(position,out_width)) * stride : U32)}

def spec_value(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~operation: F32 -> F32 -> F32,spec: Spec<Logical,Parameters>,index: U32) -> F32:
  Spec{+source,+parameters,+kind,+taps,+out_height,+out_width} = spec
  Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Input<Logical>,~input_read(~Logical,~original),~Parameters,~weight,~bias,source,parameters,kind,taps,
    output_of(U32.to_nat(channels_of(~Logical,source)),stride_of(taps),U32.to_nat(out_height),U32.to_nat(out_width),U32.to_nat(index)))

# Parameter cells related to the parameters hold at the lane of every channel
# its weights and its bias.
# The kernel extent of parameter rows.
def weights_kernel(cells: Rows.Weights) -> U32:
  Rows.Weights{tree,+kernel,groups} = cells
  kernel

# The groups of parameter rows.
def weights_groups(cells: Rows.Weights) -> U32:
  Rows.Weights{tree,kernel,+groups} = cells
  groups

def WeightLanes(~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~Related: Rows.Weights -> Parameters -> Nat -> Data) -> Type:
  @+cells: Rows.Weights -> @+parameters: Parameters -> @+channels: Nat -> @+related: Related(cells,parameters,channels) -> @+group: Nat -> @+lane: Nat -> @+small: {Nat.is_le(lane,7n) == True{} : Bool} -> @+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),channels) == True{} : Bool} ->
    @+group_inside: {Nat.is_lt(group,U32.to_nat(weights_groups(cells))) == True{} : Bool} -> @+kernel_row: Nat -> @+column: Nat -> @+row_inside: {Nat.is_lt(kernel_row,U32.to_nat(weights_kernel(cells))) == True{} : Bool} ->
    @+column_inside: {Nat.is_lt(column,U32.to_nat(weights_kernel(cells))) == True{} : Bool} ->
    {TileLaws.select_lane(lane,Rows.weight_read(cells,group,U32.from_nat(kernel_row),U32.from_nat(column))) == weight(parameters,Nat.add(lane,Nat.mul(group,8n)),U32.from_nat(kernel_row),U32.from_nat(column)) : F32}

def BiasLanes(~Parameters: Data,~bias: Parameters -> Nat -> F32,~Related: Rows.Weights -> Parameters -> Nat -> Data) -> Type:
  @+cells: Rows.Weights -> @+parameters: Parameters -> @+channels: Nat -> @+related: Related(cells,parameters,channels) -> @+group: Nat -> @+lane: Nat -> @+small: {Nat.is_le(lane,7n) == True{} : Bool} -> @+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),channels) == True{} : Bool} ->
    @+group_inside: {Nat.is_lt(group,U32.to_nat(weights_groups(cells))) == True{} : Bool} ->
    {TileLaws.select_lane(lane,Rows.bias_read(cells,group)) == bias(parameters,Nat.add(lane,Nat.mul(group,8n))) : F32}

# ---- One sample at a lane ----

# Inside the input an input cell holds at a lane the value of its channel.
def input_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+image: Nat,+group: Nat,+lane: Nat,+row: Nat,+column: Nat,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(height)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),U32.to_nat(Tiled.groups(lanes))),group)))) ==
    original(logical,U32.from_nat(input_index(image,Nat.add(lane,Nat.mul(group,8n)),row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))) : F32}:
  +number = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),U32.to_nat(Tiled.groups(lanes))),group)
  +least = Representation.least_one(U32.to_nat(Tiled.groups(lanes)))
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +plane = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +offset = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +offset_inside = RowMajor.radix(row,U32.to_nat(height),column,U32.to_nat(width),row_inside,column_inside)
  +at_inside = RowMajor.radix(Nat.add(Nat.mul(image,U32.to_nat(height)),row),Nat.mul(images,U32.to_nat(height)),column,U32.to_nat(width),
    RowMajor.radix(image,images,row,U32.to_nat(height),image_inside,row_inside),column_inside)
  +number_inside = RowMajor.radix(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),group,U32.to_nat(Tiled.groups(lanes)),at_inside,group_inside)
  +number_exact = Bounds.roundtrip(number,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(number,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(number,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits)),number_inside,in_room)))
  +split = Representation.split_at(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group,U32.to_nat(Tiled.groups(lanes)),group_inside)
  +at_exact = planar(image,U32.to_nat(height),row,U32.to_nat(width),column)
  +image_exact = Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),Nat.div(Nat.add(Nat.mul(image,plane),offset),plane),image,
    Equal.cong(Nat,Nat,value => Nat.div(value,plane),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Nat.add(Nat.mul(image,plane),offset),at_exact),parted_quotient(image,offset,plane,offset_inside))
  +offset_exact = Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),Nat.mod(Nat.add(Nat.mul(image,plane),offset),plane),offset,
    Equal.cong(Nat,Nat,value => Nat.mod(value,plane),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Nat.add(Nat.mul(image,plane),offset),at_exact),parted_remainder(image,offset,plane,offset_inside))
  +index_exact = Equal.trans(Nat,Representation.logical_index(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),channel),Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),U32.to_nat(lanes)),channel),plane),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane)),input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)),
    Equal.cong(Nat,Nat,after => Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),after),U32.to_nat(lanes)),channel),after),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),after)),Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}),plane,in_after_exact),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),U32.to_nat(lanes)),channel),plane),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane)),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),plane),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane)),input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.add(Nat.mul(value,U32.to_nat(lanes)),channel),plane),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane)),Nat.div(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),image,image_exact),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),plane),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane)),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),plane),offset),input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),plane),value),Nat.mod(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),plane),offset,offset_exact),
    Equal.sym(Nat,input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),plane),offset),
      planar(Nat.add(Nat.mul(image,U32.to_nat(lanes)),channel),U32.to_nat(height),row,U32.to_nat(width),column)))))
  Equal.trans(F32,TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(U32.to_nat(U32.from_nat(number)),least),Nat.mod(U32.to_nat(U32.from_nat(number)),least))),TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(number,least),Nat.mod(number,least))),original(logical,U32.from_nat(input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))),
    Equal.cong(Nat,F32,value => TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(value,least),Nat.mod(value,least))),U32.to_nat(U32.from_nat(number)),number,number_exact),
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(number,least),Nat.mod(number,least))),TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Nat.mod(number,least))),original(logical,U32.from_nat(input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))),
    Equal.cong(Nat,F32,value => TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},value,Nat.mod(number,least))),Nat.div(number,least),Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(number,least),(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group),split)),
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),Nat.mod(number,least))),TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group)),original(logical,U32.from_nat(input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))),
    Equal.cong(Nat,F32,value => TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),value)),Nat.mod(number,least),group,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(number,least),(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group),split)),
    Equal.trans(F32,TileLaws.select_lane(lane,Representation.cell(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group)),Representation.lane(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group,lane),original(logical,U32.from_nat(input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))),
    Representation.cell_lane(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group,lane,small),
    Equal.trans(F32,Representation.lane(~Logical,~original,logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),group,lane),original(logical,U32.from_nat(Representation.logical_index(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),channel))),original(logical,U32.from_nat(input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)))),
    Equal.cong(Bool,F32,state => Bool.pick(F32,state,original(logical,U32.from_nat(Representation.logical_index(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),channel))),0.0),Nat.is_lt(channel,U32.to_nat(lanes)),True{},inside),
    Equal.cong(Nat,F32,value => original(logical,U32.from_nat(value)),Representation.logical_index(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(height)),row),U32.to_nat(width)),column),channel),input_index(image,channel,row,column,U32.to_nat(lanes),U32.to_nat(height),U32.to_nat(width)),index_exact))))))

# A lane of a choice is the choice of the lanes.
def picked_lane(+lane: Nat,inside: Bool,+cell_read: Tile.Tile8,+value_read: F32,+cell_pad: Tile.Tile8,+value_pad: F32,
  read_lane: @+truth: {inside == True{} : Bool} -> {TileLaws.select_lane(lane,cell_read) == value_read : F32},
  +pad_lane: {TileLaws.select_lane(lane,cell_pad) == value_pad : F32}) ->
  {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,inside,cell_read,cell_pad)) == Bool.pick(F32,inside,value_read,value_pad) : F32}:
  match inside:
    case True{}: read_lane({==})
    case False{}: pad_lane

# A sample inside the input reads at a lane the value of its channel.
def bounded_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+image: Nat,+group: Nat,+lane: Nat,+row: U32,+column: U32,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool},
  +truth: {(U32.is_lt(row,height) && U32.is_lt(column,width)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},image,group,row,column)) == input_read(~Logical,~original,Input{logical,lanes,height,width},image,Nat.add(lane,Nat.mul(group,8n)),row,column) : F32}:
  input_lane(~Logical,~original,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,image,group,lane,U32.to_nat(row),U32.to_nat(column),image_inside,group_inside,small,inside,
    Equal.trans(Bool,Nat.is_lt(U32.to_nat(row),U32.to_nat(height)),U32.is_lt(row,height),True{},Equal.sym(Bool,U32.is_lt(row,height),Nat.is_lt(U32.to_nat(row),U32.to_nat(height)),
      Comparison.lt(row,height)),Boolean.left(U32.is_lt(row,height),U32.is_lt(column,width),truth)),
    Equal.trans(Bool,Nat.is_lt(U32.to_nat(column),U32.to_nat(width)),U32.is_lt(column,width),True{},Equal.sym(Bool,U32.is_lt(column,width),Nat.is_lt(U32.to_nat(column),U32.to_nat(width)),
      Comparison.lt(column,width)),Boolean.right(U32.is_lt(row,height),U32.is_lt(column,width),truth)))

# The sample of a tap over tiles holds at a lane the sample over values.
def sample_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+kernel: U32,+stride: U32,+padding: U32,+kind: Values.Sampling<F32>,
  +image: Nat,+group: Nat,+lane: Nat,+top: U32,+left: U32,+row: U32,+column: U32,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Values.sample(~Tile.Tile8,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Values.Cell{image,group,top,left},row,column,Values.start(~Tile.Tile8,Values.lifted(kind)))) ==
    Values.sample(~F32,~Input<Logical>,~input_read(~Logical,~original),Input{logical,lanes,height,width},Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Values.Cell{image,Nat.add(lane,Nat.mul(group,8n)),top,left},row,column,Values.start(~F32,kind)) : F32}:
  +x = (left + column - padding : U32)
  picked_lane(lane,U32.is_lt(row,height) && U32.is_lt(x,width),Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},image,group,row,x),
    input_read(~Logical,~original,Input{logical,lanes,height,width},image,Nat.add(lane,Nat.mul(group,8n)),row,x),Values.start(~Tile.Tile8,Values.lifted(kind)),Values.start(~F32,kind),
    truth => bounded_read(~Logical,~original,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,image,group,lane,row,x,image_inside,group_inside,small,inside,truth),
    Values.start_lane(lane,kind))

# The taps of a kernel row from column on agree at a lane.
def columns_agree(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: WeightLanes(~Parameters,~weight,~Related),~bias_lanes: BiasLanes(~Parameters,~bias,~Related),count: Nat,+column: U32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+image: Nat,+group: Nat,+lane: Nat,+top: U32,+left: U32,+row: U32,
  +kernel_row: U32,+image_inside: {Nat.is_lt(image,images) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool},
  +row_index: Nat,+row_at: {U32.from_nat(row_index) == kernel_row : U32},+row_inside: {Nat.is_lt(row_index,U32.to_nat(kernel)) == True{} : Bool},+column_index: Nat,
  +column_at: {U32.from_nat(column_index) == column : U32},+column_end: {Nat.add(column_index,count) == U32.to_nat(kernel) : Nat}) ->
  Values.ColumnsAgree(~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Input<Logical>,~input_read(~Logical,~original),~Rows.Weights,~Rows.weight_read,~Parameters,~weight,count,column,lane,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},Input{logical,lanes,height,width},Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},
    image,group,Nat.add(lane,Nat.mul(group,8n)),top,left,row,kernel_row):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +below = {%Equal.sym(Nat,Nat.add(column_index,1n+rest),1n+Nat.add(column_index,rest),Addition.successor_right(column_index,rest)) : {Nat.is_lt(column_index,_) == True{} : Bool}
        Order.lt_succ(column_index,Nat.add(column_index,rest),Order.le_add(column_index,rest)) : {Nat.is_lt(column_index,Nat.add(column_index,1n+rest)) == True{} : Bool}}
      +column_inside = {%column_end : {Nat.is_lt(column_index,_) == True{} : Bool} below : {Nat.is_lt(column_index,U32.to_nat(kernel)) == True{} : Bool}}
      +next_end = Equal.trans(Nat,Nat.add(1n+column_index,rest),Nat.add(column_index,1n+rest),U32.to_nat(kernel),
        Equal.sym(Nat,Nat.add(column_index,1n+rest),1n+Nat.add(column_index,rest),Addition.successor_right(column_index,rest)),column_end)
      +next_at = Equal.trans(U32,U32.from_nat(1n+column_index),(U32.from_nat(column_index) + 1 : U32),(column + 1 : U32),
        Equal.sym(U32,(U32.from_nat(column_index) + 1 : U32),U32.from_nat(1n+column_index),Bounds.increment_encoded(column_index)),
        Equal.cong(U32,U32,word => (word + 1 : U32),U32.from_nat(column_index),column,column_at))
      +weighed = weight_lanes(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes),related,group,lane,small,inside,group_inside,row_index,column_index,row_inside,column_inside)
      +at_row = {%row_at : {TileLaws.select_lane(lane,Rows.weight_read(Rows.Weights{weights,kernel,Tiled.groups(lanes)},group,_,U32.from_nat(column_index))) == weight(parameters,Nat.add(lane,Nat.mul(group,8n)),_,U32.from_nat(column_index)) : F32}
        weighed : {TileLaws.select_lane(lane,Rows.weight_read(Rows.Weights{weights,kernel,Tiled.groups(lanes)},group,kernel_row,U32.from_nat(column_index))) == weight(parameters,Nat.add(lane,Nat.mul(group,8n)),kernel_row,U32.from_nat(column_index)) : F32}}
      +at_column = {%column_at : {TileLaws.select_lane(lane,Rows.weight_read(Rows.Weights{weights,kernel,Tiled.groups(lanes)},group,kernel_row,_)) == weight(parameters,Nat.add(lane,Nat.mul(group,8n)),kernel_row,_) : F32}
        at_row : {TileLaws.select_lane(lane,Rows.weight_read(Rows.Weights{weights,kernel,Tiled.groups(lanes)},group,kernel_row,column)) == weight(parameters,Nat.add(lane,Nat.mul(group,8n)),kernel_row,column) : F32}}
      KernelGemm.Both{KernelGemm.Both{sample_lane(~Logical,~original,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,kernel,stride,padding,kind,image,group,lane,top,left,row,column,image_inside,group_inside,small,
            inside),
          at_column},
        columns_agree(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,rest,(column + 1 : U32),logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,image,group,lane,top,left,row,kernel_row,image_inside,group_inside,small,inside,row_index,row_at,row_inside,1n+column_index,next_at,next_end)}

# The kernel rows of a window from kernel_row on agree at a lane.
def rows_agree(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: WeightLanes(~Parameters,~weight,~Related),~bias_lanes: BiasLanes(~Parameters,~bias,~Related),count: Nat,+kernel_row: U32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+image: Nat,+group: Nat,+lane: Nat,+top: U32,+left: U32,
  +image_inside: {Nat.is_lt(image,images) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
  +small: {Nat.is_le(lane,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(lanes)) == True{} : Bool},
  +row_index: Nat,+row_at: {U32.from_nat(row_index) == kernel_row : U32},+row_end: {Nat.add(row_index,count) == U32.to_nat(kernel) : Nat}) ->
  Values.RowsAgree(~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Input<Logical>,~input_read(~Logical,~original),~Rows.Weights,~Rows.weight_read,~Parameters,~weight,count,kernel_row,lane,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},Input{logical,lanes,height,width},Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},
    image,group,Nat.add(lane,Nat.mul(group,8n)),top,left):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +below = {%Equal.sym(Nat,Nat.add(row_index,1n+rest),1n+Nat.add(row_index,rest),Addition.successor_right(row_index,rest)) : {Nat.is_lt(row_index,_) == True{} : Bool}
        Order.lt_succ(row_index,Nat.add(row_index,rest),Order.le_add(row_index,rest)) : {Nat.is_lt(row_index,Nat.add(row_index,1n+rest)) == True{} : Bool}}
      +row_inside = {%row_end : {Nat.is_lt(row_index,_) == True{} : Bool} below : {Nat.is_lt(row_index,U32.to_nat(kernel)) == True{} : Bool}}
      +next_end = Equal.trans(Nat,Nat.add(1n+row_index,rest),Nat.add(row_index,1n+rest),U32.to_nat(kernel),
        Equal.sym(Nat,Nat.add(row_index,1n+rest),1n+Nat.add(row_index,rest),Addition.successor_right(row_index,rest)),row_end)
      +next_at = Equal.trans(U32,U32.from_nat(1n+row_index),(U32.from_nat(row_index) + 1 : U32),(kernel_row + 1 : U32),
        Equal.sym(U32,(U32.from_nat(row_index) + 1 : U32),U32.from_nat(1n+row_index),Bounds.increment_encoded(row_index)),
        Equal.cong(U32,U32,word => (word + 1 : U32),U32.from_nat(row_index),kernel_row,row_at))
      KernelGemm.Both{columns_agree(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,U32.to_nat(kernel),0,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,image,group,lane,top,left,
          Values.row_of(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Values.Cell{image,group,top,left},kernel_row),kernel_row,image_inside,group_inside,small,inside,row_index,row_at,row_inside,0n,{==},{==}),
        rows_agree(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,rest,(kernel_row + 1 : U32),logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,image,group,lane,top,left,image_inside,group_inside,small,inside,1n+row_index,next_at,next_end)}

# ---- One output cell ----

# The logical index of an output lane lies below the output values.
def spot_below(+out_outer: U32,+out_after: U32,+lanes: U32,+out_height: U32,+out_width: U32,+images: Nat,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},+outer: Nat,+channel: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(images,Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool}) ->
  {Nat.is_lt(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel),Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))) == True{} : Bool}:
  %out_after_exact : {Nat.is_lt(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel),Nat.mul(Nat.mul(images,U32.to_nat(lanes)),_)) == True{} : Bool}
  Representation.logical_below(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel,images,
    %Equal.sym(Nat,Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)),out_after_exact) : {Nat.is_lt(outer,Nat.mul(images,_)) == True{} : Bool} outer_inside,
    channel_inside)

# The window of the logical index of an output lane is the window of its
# output position and its channel.
def output_decoded(+out_outer: U32,+out_after: U32,+lanes: U32,+stride: U32,+out_height: U32,+out_width: U32,+images: Nat,+limits: Machine.Limits,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+outer: Nat,+channel: Nat,
  +outer_inside: {Nat.is_lt(outer,Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,U32.to_nat(lanes)) == True{} : Bool},
  +height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(out_width)) == True{} : Bool}) ->
  {output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),U32.to_nat(U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel)))) ==
    Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),
      (U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)} : Values.Cell}:
  +plane = Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))
  +spot = Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel)
  +line = Nat.add(Nat.mul(Nat.div(outer,plane),U32.to_nat(lanes)),channel)
  +position = Nat.mod(outer,plane)
  +split = Nat.add(Nat.mul(line,plane),position)
  +grouped = positions_grouped(images,U32.to_nat(out_height),U32.to_nat(out_width),outer,outer_inside)
  +plane_positive = DivisionOrder.right_factor_positive(images,plane,Order.le_lt_trans(0n,outer,Nat.mul(images,plane),Order.zero_le(outer),grouped))
  +position_small = DivisionOrder.remainder_small(outer,plane,plane_positive)
  +spot_exact = Bounds.roundtrip(spot,Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(spot,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(spot,Nat.mul(Nat.mul(images,U32.to_nat(lanes)),plane),U32.to_nat(Machine.capacity(limits)),spot_below(out_outer,out_after,lanes,out_height,out_width,images,out_after_exact,outer,channel,
      grouped,channel_inside),out_room)))
  +split_exact = Equal.cong(Nat,Nat,after => Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,after),U32.to_nat(lanes)),channel),after),Nat.mod(outer,after)),
    Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}),plane,out_after_exact)
  Equal.trans(Values.Cell,output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),U32.to_nat(U32.from_nat(spot))),output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),spot),Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),value),U32.to_nat(U32.from_nat(spot)),spot,spot_exact),
    Equal.trans(Values.Cell,output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),spot),output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),split),Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),value),spot,split,split_exact),
    Equal.trans(Values.Cell,output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),split),Values.Cell{Nat.div(line,U32.to_nat(lanes)),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{Nat.div(value,U32.to_nat(lanes)),Nat.mod(value,U32.to_nat(lanes)),(U32.from_nat(Nat.div(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32)},Nat.div(split,plane),line,parted_quotient(line,position,plane,position_small)),
    Equal.trans(Values.Cell,Values.Cell{Nat.div(line,U32.to_nat(lanes)),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(Nat.mod(split,plane),U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(line,U32.to_nat(lanes)),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{Nat.div(line,U32.to_nat(lanes)),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(value,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(value,U32.to_nat(out_width))) * stride : U32)},Nat.mod(split,plane),position,parted_remainder(line,position,plane,position_small)),
    Equal.trans(Values.Cell,Values.Cell{Nat.div(line,U32.to_nat(lanes)),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(outer,plane),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{value,Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Nat.div(line,U32.to_nat(lanes)),Nat.div(outer,plane),parted_quotient(Nat.div(outer,plane),channel,U32.to_nat(lanes),channel_inside)),
    Equal.trans(Values.Cell,Values.Cell{Nat.div(outer,plane),Nat.mod(line,U32.to_nat(lanes)),(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(outer,plane),channel,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{Nat.div(outer,plane),value,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Nat.mod(line,U32.to_nat(lanes)),channel,parted_remainder(Nat.div(outer,plane),channel,U32.to_nat(lanes),channel_inside)),
    Equal.trans(Values.Cell,Values.Cell{Nat.div(outer,plane),channel,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{value,channel,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Nat.div(outer,plane),Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),DivisionOrder.product_quotient(outer,U32.to_nat(out_height),U32.to_nat(out_width),height_positive,width_positive)),
    Equal.trans(Values.Cell,Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.div(position,U32.to_nat(out_width))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)},
    Equal.cong(Nat,Values.Cell,value => Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(value) * stride : U32),(U32.from_nat(Nat.mod(position,U32.to_nat(out_width))) * stride : U32)},Nat.div(position,U32.to_nat(out_width)),Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),DivisionOrder.remainder_quotient(outer,U32.to_nat(out_height),U32.to_nat(out_width),height_positive,width_positive)),
    Equal.cong(Nat,Values.Cell,value => Values.Cell{Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height)),channel,(U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32),(U32.from_nat(value) * stride : U32)},Nat.mod(position,U32.to_nat(out_width)),Nat.mod(outer,U32.to_nat(out_width)),DivisionOrder.remainder_remainder(outer,U32.to_nat(out_height),U32.to_nat(out_width),height_positive,width_positive))))))))))

# A lane of an output cell below the channels holds the value of its window.
def lane_value(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: WeightLanes(~Parameters,~weight,~Related),~bias_lanes: BiasLanes(~Parameters,~bias,~Related),~operation: F32 -> F32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+out_outer: U32,+out_after: U32,+out_height: U32,+out_width: U32,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)))) == True{} : Bool},+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},
  +lane_inside: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n)),U32.to_nat(lanes)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))) == spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation,Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n))))) : F32}:
  +group = Nat.mod(index,U32.to_nat(Tiled.groups(lanes)))
  +outer = Nat.div(index,U32.to_nat(Tiled.groups(lanes)))
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +positions = Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width))
  +all_positive = Order.le_lt_trans(0n,index,Nat.mul(positions,U32.to_nat(Tiled.groups(lanes))),Order.zero_le(index),inside)
  +groups_positive = DivisionOrder.right_factor_positive(positions,U32.to_nat(Tiled.groups(lanes)),all_positive)
  +positions_positive = RowMajor.left_factor_positive(positions,U32.to_nat(Tiled.groups(lanes)),all_positive)
  +width_positive = DivisionOrder.right_factor_positive(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width),positions_positive)
  +rows_positive = RowMajor.left_factor_positive(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width),positions_positive)
  +height_positive = DivisionOrder.right_factor_positive(images,U32.to_nat(out_height),rows_positive)
  +outer_inside = DivisionOrder.quotient_below(index,positions,U32.to_nat(Tiled.groups(lanes)),groups_positive,inside)
  +row_inside = DivisionOrder.quotient_below(outer,Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width),width_positive,outer_inside)
  +image_inside = DivisionOrder.quotient_below(Nat.div(outer,U32.to_nat(out_width)),images,U32.to_nat(out_height),height_positive,row_inside)
  +group_inside = DivisionOrder.remainder_small(index,U32.to_nat(Tiled.groups(lanes)),groups_positive)
  +image = Nat.div(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))
  +top = (U32.from_nat(Nat.mod(Nat.div(outer,U32.to_nat(out_width)),U32.to_nat(out_height))) * stride : U32)
  +left = (U32.from_nat(Nat.mod(outer,U32.to_nat(out_width))) * stride : U32)
  Equal.trans(F32,TileLaws.select_lane(lane,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))),Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Input<Logical>,~input_read(~Logical,~original),~Parameters,~weight,~bias,Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Values.Cell{image,channel,top,left}),spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation,Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n))))),
    Values.at_lane(~operation,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Input<Logical>,~input_read(~Logical,~original),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,~Parameters,~weight,~bias,lane,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},Input{logical,lanes,height,width},Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,kind,
      Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},image,group,channel,top,left,
      rows_agree(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,U32.to_nat(kernel),0,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,image,group,lane,top,left,image_inside,group_inside,small,lane_inside,0n,{==},{==}),
      bias_lanes(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes),related,group,lane,small,lane_inside,group_inside)),
    Equal.sym(F32,spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation,Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n))))),Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Input<Logical>,~input_read(~Logical,~original),~Parameters,~weight,~bias,Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Values.Cell{image,channel,top,left}),
      Equal.cong(Values.Cell,F32,cell => Values.at(~F32,~operation,~Values.lane_update,~Values.lane_add,~Input<Logical>,~input_read(~Logical,~original),~Parameters,~weight,~bias,Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},cell),
        output_of(U32.to_nat(lanes),stride,U32.to_nat(out_height),U32.to_nat(out_width),U32.to_nat(U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},outer,channel)))),Values.Cell{image,channel,top,left},
        output_decoded(out_outer,out_after,lanes,stride,out_height,out_width,images,limits,out_after_exact,out_room,outer,channel,outer_inside,lane_inside,height_positive,
          width_positive))))

# A lane of an output cell holds what padding keeps of the lane of the window.
def lane_valid(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: WeightLanes(~Parameters,~weight,~Related),~bias_lanes: BiasLanes(~Parameters,~bias,~Related),~operation: F32 -> F32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+out_outer: U32,+out_after: U32,+out_height: U32,+out_width: U32,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)))) == True{} : Bool},+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {PaddedCells.kept(U32.to_nat(lanes),Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),lane,TileLaws.select_lane(lane,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index)))) ==
    Representation.lane(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),lane) : F32}:
  PaddedCells.kept_agree(U32.to_nat(lanes),Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),lane,TileLaws.select_lane(lane,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))),spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation,Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},U32.from_nat(Representation.logical_index(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n))))),
    Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),8n)),U32.to_nat(lanes)),{==},
    lane_inside => lane_value(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,lane,small,lane_inside))

# The output cell of a window operation at an index below the output cells
# holds the lifted values of the window.
def cell_agrees(~Logical: Data,~original: Logical -> U32 -> F32,~Parameters: Data,~weight: Parameters -> Nat -> U32 -> U32 -> F32,~bias: Parameters -> Nat -> F32,
  ~Related: Rows.Weights -> Parameters -> Nat -> Data,~weight_lanes: WeightLanes(~Parameters,~weight,~Related),~bias_lanes: BiasLanes(~Parameters,~bias,~Related),~operation: F32 -> F32 -> F32,+logical: Logical,+in_outer: U32,+in_after: U32,+lanes: U32,+height: U32,+width: U32,+limits: Machine.Limits,+images: Nat,
  +in_after_exact: {Representation.after_count(Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
  +in_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+weights: Tree.Tree<Tile.Tile8>,+parameters: Parameters,+kind: Values.Sampling<F32>,+kernel: U32,+stride: U32,+padding: U32,
  +related: Related(Rows.Weights{weights,kernel,Tiled.groups(lanes)},parameters,U32.to_nat(lanes)),+out_outer: U32,+out_after: U32,+out_height: U32,+out_width: U32,
  +out_after_exact: {Representation.after_count(Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}) == Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)) : Nat},
  +out_room: {Nat.is_le(Nat.mul(Nat.mul(images,U32.to_nat(lanes)),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+kind_w: Window.Sampling,
  +kind_exact: {Rows.of_sampling(kind_w) == Values.lifted(kind) : Values.Sampling<Tile.Tile8>},
  +channels_room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cell: U32,
  +inside: {Nat.is_lt(U32.to_nat(cell),Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)))) == True{} : Bool}) ->
  {Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation),
      Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},cell) ==
    Representation.lifted(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Representation.Lifted{Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},cell) : Tile.Tile8}:
  +index = U32.to_nat(cell)
  +all_positive = Order.le_lt_trans(0n,index,Nat.mul(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes))),Order.zero_le(index),inside)
  +groups_positive = DivisionOrder.right_factor_positive(Nat.mul(Nat.mul(images,U32.to_nat(out_height)),U32.to_nat(out_width)),U32.to_nat(Tiled.groups(lanes)),all_positive)
  +target = Representation.cell(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,U32.to_nat(Tiled.groups(lanes))),Nat.mod(index,U32.to_nat(Tiled.groups(lanes))))
  Equal.trans(Tile.Tile8,
    Task.problem_value(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~TiledMap.pairs(~operation),
      Task.Problem{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},weights,kind_w,kernel,stride,padding,height,width,Tiled.groups(lanes),lanes,out_height,out_width},cell),
    Padding.cell(TiledMap.Padding{lanes,Tiled.groups(lanes),0,Grid.Innermost{}},index,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))),
    Representation.lifted(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Representation.Lifted{Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},cell),
    Equal.cong(Values.Sampling<Tile.Tile8>,Tile.Tile8,sampling => Padding.cell(TiledMap.Padding{lanes,Tiled.groups(lanes),0,Grid.Innermost{}},index,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},sampling,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))),
      Rows.of_sampling(kind_w),Values.lifted(kind),kind_exact),
    Equal.trans(Tile.Tile8,Padding.cell(TiledMap.Padding{lanes,Tiled.groups(lanes),0,Grid.Innermost{}},index,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index))),target,
      Representation.lifted(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Representation.Lifted{Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},cell),
      PaddedCells.padded_group(limits,lanes,0,Grid.Innermost{},Nat.mod(index,U32.to_nat(Tiled.groups(lanes))),index,Values.at(~Tile.Tile8,~TiledMap.pairs(~operation),~Tile.update_lanes,~Tile.plus,~Rows.Input<Representation.Lifted<Logical>>,~Rows.input_read(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Rows.Weights,~Rows.weight_read,~Rows.bias_read,Rows.Input{Representation.Lifted{logical,Cells.Cells{in_outer,in_after,lanes,Tiled.groups(lanes),Grid.Innermost{}}},height,width,Tiled.groups(lanes)},
      Rows.Weights{weights,kernel,Tiled.groups(lanes)},Values.lifted(kind),Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},Piece.output_cell(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width,index)),target,TiledValues.grouped(limits,lanes,channels_room),
        DivisionOrder.remainder_small(index,U32.to_nat(Tiled.groups(lanes)),groups_positive),{==},
        PaddedCells.Valid{lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,0n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,1n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,2n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,3n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,4n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,5n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,6n,{==}),
          lane_valid(~Logical,~original,~Parameters,~weight,~bias,~Related,~weight_lanes,~bias_lanes,~operation,logical,in_outer,in_after,lanes,height,width,limits,images,in_after_exact,in_room,weights,parameters,kind,kernel,stride,padding,related,out_outer,out_after,out_height,out_width,out_after_exact,out_room,index,inside,7n,{==})}),
      Equal.cong(Nat,Tile.Tile8,divisor => Representation.cell(~Spec<Logical,Parameters>,~spec_value(~Logical,~original,~Parameters,~weight,~bias,~operation),Spec{Input{logical,lanes,height,width},parameters,kind,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},out_height,out_width},Cells.Cells{out_outer,out_after,lanes,Tiled.groups(lanes),Grid.Innermost{}},Nat.div(index,divisor),Nat.mod(index,divisor)),U32.to_nat(Tiled.groups(lanes)),
        Representation.least_one(U32.to_nat(Tiled.groups(lanes))),Equal.sym(Nat,Representation.least_one(U32.to_nat(Tiled.groups(lanes))),U32.to_nat(Tiled.groups(lanes)),Representation.least_positive(U32.to_nat(Tiled.groups(lanes)),groups_positive)))))
