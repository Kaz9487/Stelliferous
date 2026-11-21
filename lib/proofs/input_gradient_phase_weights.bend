# Phase weight cells [C/8][O][k][k] and their checked view addresses.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ../tensor_tiled.bend as Tiled
import ./tensor_view_lanes.bend as ViewLanes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded

def cell_address(context: Kernel.Phase,+block: Nat,+output: Nat,+row: Nat,+column: Nat) -> U32:
  Kernel.Phase{Kernel.Geometry{c,+o,h,w,+k,+s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = context
  (U32.from_nat(output) * (k * k) + U32.from_nat(block) * (o * k * k) +
    (yt - U32.from_nat(Nat.mul(row,U32.to_nat(s)))) * k + (xt - U32.from_nat(Nat.mul(column,U32.to_nat(s)))) : U32)

def cell_tile(+input: Buffer.Model<Tile.Tile8>,context: Kernel.Phase,block: Nat,output: Nat,row: Nat,column: Nat) -> Tile.Tile8:
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,input),cell_address(context,block,output,row,column))

def backward_row(+prefix: U32,+row: U32,+column: U32,+tap: U32,+step: U32,+width: U32) ->
  {(tap * (0 - step * width : U32) + (prefix + row * width + column : U32) : U32) ==
    (prefix + (row - tap * step) * width + column : U32) : U32}:
  +advance = (tap * step * width : U32)
  +start = (prefix + row * width + column : U32)
  +middle = (start - advance : U32)
  Equal.trans(U32,(tap * (0 - step * width : U32) + start : U32),middle,(prefix + (row - tap * step) * width + column : U32),
    %Equal.sym(U32,(tap * step * width : U32),(tap * (step * width : U32) : U32),Arithmetic.multiply_associative(tap,step,width)) : {(tap * (0 - step * width : U32) + start : U32) == (start - _ : U32) : U32}
    Arithmetic.negative_step(tap,(step * width : U32),start),
    %Arithmetic.add_after_subtract((prefix + row * width : U32),advance,column) : {_ == (prefix + (row - tap * step) * width + column : U32) : U32}
    %Equal.sym(U32,(prefix + row * width - advance : U32),(prefix + (row * width - advance : U32) : U32),Arithmetic.subtract_from_sum(prefix,(row * width : U32),advance)) : {(_ + column : U32) == (prefix + (row - tap * step) * width + column : U32) : U32}
    Equal.cong(U32,U32,value => (prefix + value + column : U32),(row * width - advance : U32),((row - tap * step : U32) * width : U32),
      Equal.sym(U32,((row - tap * step : U32) * width : U32),(row * width - advance : U32),Arithmetic.multiply_subtractive(row,(tap * step : U32),width))))

def cell_view_address(+context: Kernel.Phase,+block: Nat,+output: Nat,+row: Nat,+column: Nat) ->
  {ViewLanes.view_terms(Kernel.weight_view(context),[block,output,row,column]) == cell_address(context,block,output,row,column) : U32}:
  Kernel.Phase{Kernel.Geometry{c,+o,h,w,+k,+s,p,gh,gw},Kernel.Axis{yf,yc,+yt,yr,ys},Kernel.Axis{xf,xc,+xt,xr,xs},inner} = context
  +outer = (U32.from_nat(output) * (k * k) : U32)
  +inner = (U32.from_nat(block) * (o * k * k) : U32)
  +prefix = (outer + inner : U32)
  +tr = U32.from_nat(row)
  +tc = U32.from_nat(column)
  +initial = (outer + (inner + (yt * k + xt)) : U32)
  +joined = (prefix + yt * k + xt : U32)
  +join = Equal.trans(U32,initial,(prefix + (yt * k + xt) : U32),joined,
    Equal.sym(U32,(prefix + (yt * k + xt) : U32),initial,Arithmetic.add_associative(outer,inner,(yt * k + xt : U32))),
    Equal.sym(U32,joined,(prefix + (yt * k + xt) : U32),Arithmetic.add_associative(prefix,(yt * k : U32),xt)))
  +shifted = (prefix + (yt - tr * s) * k + xt : U32)
  +row_same = Equal.trans(U32,(tr * (0 - s * k : U32) + initial : U32),(tr * (0 - s * k : U32) + joined : U32),shifted,
    Equal.cong(U32,U32,value => (tr * (0 - s * k : U32) + value : U32),initial,joined,join),backward_row(prefix,yt,xt,tr,s,k))
  %Equal.sym(U32,(tr * (0 - s * k : U32) + initial : U32),shifted,row_same) :
    {(tc * (0 - s : U32) + _ : U32) == cell_address(context,block,output,row,column) : U32}
  %Encoded.scale_encoded(row,s) : {(tc * (0 - s : U32) + shifted : U32) == (prefix + (yt - _) * k + (xt - U32.from_nat(Nat.mul(column,U32.to_nat(s)))) : U32) : U32}
  %Encoded.scale_encoded(column,s) : {(tc * (0 - s : U32) + shifted : U32) == (prefix + (yt - tr * s) * k + (xt - _) : U32) : U32}
  Equal.trans(U32,(tc * (0 - s : U32) + shifted : U32),(shifted - tc * s : U32),(prefix + (yt - tr * s) * k + (xt - tc * s) : U32),Arithmetic.negative_step(tc,s,shifted),
    Arithmetic.subtract_from_sum((prefix + (yt - tr * s) * k : U32),xt,(tc * s : U32)))
