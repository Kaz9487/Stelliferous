# Phase geometry and pure samples for the shared WindowDot specification bridge.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./kernel_tile_refinement.bend as TileLaws
import ../math_sequence.bend as Sequence
import ./indexed_sequence_laws.bend as SequenceLaws
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./natural_addition.bend as Addition
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./encoded_coordinates.bend as Encoded
import ./u32_division.bend as NativeDivision
import ./row_major_index.bend as RowMajor

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

def phase_outputs(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  o

# The cells of a gradient pixel: one a group of outputs.
def gradient_groups(+context: Kernel.Phase) -> U32:
  Tiled.groups(phase_outputs(context))

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

def output_groups(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,gw} = geometry
  Tiled.groups(o)

def input_row(+context: Kernel.Phase,+row: U32) -> U32:
  (axis_source(phase_y(context)) + row : U32)

def input_column(+context: Kernel.Phase,+column: U32) -> U32:
  (axis_source(phase_x(context)) + column : U32)

# Lane lane of the pixels of one channel along a row, groups cells apart.
type CellsSample is Data:
  CellsSample{base: U32,row: U32,column: U32,rows: U32,columns: U32,channel: U32,groups: U32}

def cells_value(input: Buffer.Model<Tile.Tile8>,context: CellsSample,lane: Nat) -> F32:
  CellsSample{base,+row,+column,+rows,+columns,channel,groups} = context
  +column = (column + U32.from_nat(lane) : U32)
  Bool.pick(F32,U32.is_lt(row,rows) && U32.is_lt(column,columns),
    Cells.lane_of(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),(base + (row * columns + column) * groups : U32)),(channel % 8 : U32)),0.0)

def cells_tile(+input: Buffer.Model<Tile.Tile8>,+context: CellsSample) -> Tile.Tile8:
  Tile.from_sequence(Sequence.tabulate(~F32,~TileLaws.SampleContext<Buffer.Model<Tile.Tile8>,CellsSample>,~TileLaws.sample_context(~Buffer.Model<Tile.Tile8>,~CellsSample,~cells_value),8n,0n,TileLaws.SampleContext{input,context}))

def cells_tile_index(+input: Buffer.Model<Tile.Tile8>,+context: CellsSample,+index: Sequence.Index(8n)) ->
  {Tile.element_at(index,cells_tile(input,context)) == cells_value(input,context,Sequence.index_value(8n,index)) : F32}:
  %Addition.zero_right(Sequence.index_value(8n,index)) : {Tile.element_at(index,cells_tile(input,context)) == cells_value(input,context,_) : F32}
  SequenceLaws.get_tabulated(~F32,~TileLaws.SampleContext<Buffer.Model<Tile.Tile8>,CellsSample>,~TileLaws.sample_context(~Buffer.Model<Tile.Tile8>,~CellsSample,~cells_value),8n,index,0n,TileLaws.SampleContext{input,context})

# The first cell of an image of the gradient.
def cells_base(+context: Kernel.Phase,+image: U32) -> U32:
  (image * (gradient_rows(context) * gradient_columns(context)) * Tiled.groups(phase_outputs(context)) : U32)

def cells_image_base(+geometry: Kernel.Geometry,+image: U32) -> U32:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  (image * (gh * gw) * Tiled.groups(o) : U32)

def cells_base_matches(+geometry: Kernel.Geometry,+y: U32,+x: U32,+image: U32) -> {cells_base(Kernel.phase(geometry,y,x),image) == cells_image_base(geometry,image) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def cells_input_tile(+context: Kernel.Phase,+image: U32,+row: U32,+column: U32,+gradient: Buffer.Model<Tile.Tile8>,+output: Nat,+tap_row: Nat,+tap_column: Nat) -> Tile.Tile8:
  cells_tile(gradient,CellsSample{(cells_base(context,image) + U32.from_nat(output) / 8 : U32),
    (input_row(context,row) + U32.from_nat(tap_row) : U32),(input_column(context,column) + U32.from_nat(tap_column) : U32),gradient_rows(context),gradient_columns(context),U32.from_nat(output),
    gradient_groups(context)})
