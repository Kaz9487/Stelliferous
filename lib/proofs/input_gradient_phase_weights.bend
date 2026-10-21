# Pure phase weight samples for the shared mapped-weight specification bridge.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./input_gradient_phase_packing.bend as Packing
import ./storage_buffer_model.bend as Buffer

def stride(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  s

def channels(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  c

def kernel(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  k

def column_value(input: Buffer.Model<F32>,channel: U32,base: U32,row: U32,column: U32,+context: Kernel.Phase) -> Tile.Tile8:
  Packing.weight_tile(input,Packing.WeightSample{channels(context),channel,(base + row * kernel(context) + column : U32),(kernel(context) * kernel(context) : U32)})

def output_step(+context: Kernel.Phase) -> U32:
  (channels(context) * kernel(context) * kernel(context) : U32)

def axis_tap(axis: Kernel.Axis) -> U32:
  Kernel.Axis{first,count,tap,taps,source} = axis
  tap

def output_tile(+input: Buffer.Model<F32>,+channel: U32,+base: U32,+row: U32,+column: U32,+context: Kernel.Phase,+index: Nat,+tap_row: Nat,+tap_column: Nat) -> Tile.Tile8:
  column_value(input,channel,(base + U32.from_nat(Nat.mul(index,U32.to_nat(output_step(context)))) : U32),
    (row - U32.from_nat(Nat.mul(tap_row,U32.to_nat(stride(context)))) : U32),(column - U32.from_nat(Nat.mul(tap_column,U32.to_nat(stride(context)))) : U32),context)
