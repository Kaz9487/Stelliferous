# Phase geometry and pure samples for the shared WindowDot specification bridge.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./kernel_weight_gradient_packing.bend as Tiles
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_order.bend as Order
import ../math_sequence.bend as Sequence
import ./indexed_sequence_laws.bend as SequenceLaws
import ./kernel_accumulator.bend as Accumulator
import ./kernel_coordinate_refinement.bend as Coordinates
import ../kernel_rows.bend as Rows

type Sample is Data:
  Sample{base: U32,row: U32,column: U32,rows: U32,columns: U32}

type WeightSample is Data:
  WeightSample{channels: U32,channel: U32,base: U32,spatial: U32}

def sample_value(input: Buffer.Model<F32>,context: Sample,lane: Nat) -> F32:
  Sample{base,+row,+column,+rows,+columns} = context
  +column = (column + U32.from_nat(lane) : U32)
  Bool.pick(F32,U32.is_lt(row,rows) && U32.is_lt(column,columns),Tree.read(F32,Buffer.tree(~F32,input),(base + row * columns + column : U32)),0.0)

def sample_tile(input: Buffer.Model<F32>,context: Sample) -> Tile.Tile8:
  Tiles.tile_value(~Sample,~sample_value,input,context)

def sample_tile_lane(+input: Buffer.Model<F32>,+context: Sample,+lane: Nat,inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,sample_tile(input,context)) == sample_value(input,context,lane) : F32}:
  Tiles.tile_value_at(~Sample,~sample_value,input,context,lane,inside)

def tile_value_index(~Context: Data,~sample: Buffer.Model<F32> -> Context -> Nat -> F32,+input: Buffer.Model<F32>,+context: Context,+index: Sequence.Index(8n)) ->
  {Tile.element_at(index,Tiles.tile_value(~Context,~sample,input,context)) == sample(input,context,Sequence.index_value(8n,index)) : F32}:
  match index:
    case Sequence.First{}: {==}
    case Sequence.Next{+previous}:
      +position = Sequence.index_value(7n,previous)
      %Order.commutative(position,1n) :
        {Tile.element_at(Sequence.Next{previous},Tiles.tile_value(~Context,~sample,input,context)) == sample(input,context,_) : F32}
      SequenceLaws.get_tabulated(~F32,~TileLaws.SampleContext<Buffer.Model<F32>,Context>,~TileLaws.sample_context(~Buffer.Model<F32>,~Context,~sample),7n,previous,1n,TileLaws.SampleContext{input,context})

def weight_value(input: Buffer.Model<F32>,context: WeightSample,lane: Nat) -> F32:
  WeightSample{+channels,+channel,+base,+spatial} = context
  +channel = (channel + U32.from_nat(lane) : U32)
  Bool.pick(F32,U32.is_lt(channel,channels),Tree.read(F32,Buffer.tree(~F32,input),(base + channel * spatial : U32)),0.0)

def weight_tile(input: Buffer.Model<F32>,context: WeightSample) -> Tile.Tile8:
  Tiles.tile_value(~WeightSample,~weight_value,input,context)

def weight_tile_half(+upper: Bool,+row: Accumulator.Row,+input: Buffer.Model<F32>,+context: WeightSample) ->
  {Accumulator.row(~F32,row,Rows.half(upper,weight_tile(input,context))) == weight_value(input,context,Tiles.patch_key(upper,row)) : F32}:
  Equal.trans(F32,Accumulator.row(~F32,row,Rows.half(upper,weight_tile(input,context))),Tile.element_at(Coordinates.weight_lane(upper,row),weight_tile(input,context)),weight_value(input,context,Tiles.patch_key(upper,row)),
    Coordinates.weight_sample(upper,row,weight_tile(input,context)),
    Equal.trans(F32,Tile.element_at(Coordinates.weight_lane(upper,row),weight_tile(input,context)),weight_value(input,context,Sequence.index_value(8n,Coordinates.weight_lane(upper,row))),weight_value(input,context,Tiles.patch_key(upper,row)),
      tile_value_index(~WeightSample,~weight_value,input,context,Coordinates.weight_lane(upper,row)),
      Equal.cong(Nat,F32,weight_value(input,context),Sequence.index_value(8n,Coordinates.weight_lane(upper,row)),Tiles.patch_key(upper,row),Coordinates.weight_coordinate(upper,row))))

def phase_columns(context: Kernel.Phase) -> U32:
  Kernel.Phase{geometry,y,Kernel.Axis{first,count,tap,taps,source},inner} = context
  taps

def gradient_rows(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  gh

def gradient_columns(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  gw

def phase_rows(context: Kernel.Phase) -> U32:
  Kernel.Phase{geometry,Kernel.Axis{first,count,tap,taps,source},x,inner} = context
  taps

def phase_area(+context: Kernel.Phase) -> Nat:
  Nat.mul(U32.to_nat(phase_rows(context)),U32.to_nat(phase_columns(context)))

def gradient_step(+context: Kernel.Phase) -> U32:
  (gradient_rows(context) * gradient_columns(context) : U32)

def phase_outputs(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  o

def phase_y(context: Kernel.Phase) -> Kernel.Axis:
  Kernel.Phase{geometry,y,x,inner} = context
  y

def phase_x(context: Kernel.Phase) -> Kernel.Axis:
  Kernel.Phase{geometry,y,x,inner} = context
  x

def axis_source(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  source

def phase_index(+context: Kernel.Phase,+output: Nat,+row: Nat,+column: Nat) -> Nat:
  Nat.add(Nat.mul(output,phase_area(context)),Nat.add(Nat.mul(row,U32.to_nat(phase_columns(context))),column))

def input_base(+context: Kernel.Phase,+image: U32) -> U32:
  (image * phase_outputs(context) * gradient_rows(context) * gradient_columns(context) : U32)

# The same base and plane step from the geometry alone: every phase of a
# geometry reads the same gradient planes.
def image_base(+geometry: Kernel.Geometry,+image: U32) -> U32:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  (image * o * gh * gw : U32)
def plane_size(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = geometry
  (gh * gw : U32)
def input_base_matches(+geometry: Kernel.Geometry,+y: U32,+x: U32,+image: U32) -> {input_base(Kernel.phase(geometry,y,x),image) == image_base(geometry,image) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def gradient_step_matches(+geometry: Kernel.Geometry,+y: U32,+x: U32) -> {gradient_step(Kernel.phase(geometry,y,x)) == plane_size(geometry) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def input_row(+context: Kernel.Phase,+row: U32) -> U32:
  (axis_source(phase_y(context)) + row : U32)

def input_column(+context: Kernel.Phase,+column: U32) -> U32:
  (axis_source(phase_x(context)) + column : U32)

def input_tile(+context: Kernel.Phase,+image: U32,+row: U32,+column: U32,+gradient: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat) -> Tile.Tile8:
  sample_tile(gradient,Sample{(input_base(context,image) + U32.from_nat(Nat.mul(output,U32.to_nat(gradient_step(context)))) : U32),
    (input_row(context,row) + U32.from_nat(tap_row) : U32),(input_column(context,column) + U32.from_nat(tap_column) : U32),gradient_rows(context),gradient_columns(context)})
