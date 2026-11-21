# Native stencil counters denote the canonical packed entry and its eight pixels.
import Base
import ../kernel_spatial_packing.bend as Packing
import ./binary_radix.bend as Radix
import ./encoded_coordinates.bend as Encoded
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./word_multiplication.bend as WordMultiplication

def stencil_rows(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  rows

def stencil_columns(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  columns

def stencil_inner(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  inner

def stencil_width(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner,groups} = window
  output_width

type Geometry is Data:
  Geometry{window: Packing.Stencil,channels: Nat,word_capacity: U32,
    word_ok: {U32.is_le(word_capacity,2147483648) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(stencil_rows(window))) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,U32.to_nat(stencil_columns(window))) == True{} : Bool},
    inner_positive: {Nat.is_lt(0n,U32.to_nat(stencil_inner(window))) == True{} : Bool},
    width_positive: {Nat.is_lt(0n,U32.to_nat(stencil_width(window))) == True{} : Bool},
    inner_exact: {U32.to_nat(stencil_inner(window)) == Nat.mul(channels,Nat.mul(U32.to_nat(stencil_rows(window)),U32.to_nat(stencil_columns(window)))) : Nat},
    channels_exact: {U32.to_nat((stencil_inner(window) / (stencil_rows(window) * stencil_columns(window)) : U32)) == channels : Nat},
    rows_room: {Nat.is_le(Nat.double(U32.to_nat(stencil_rows(window))),U32.to_nat(word_capacity)) == True{} : Bool},
    columns_room: {Nat.is_le(Nat.double(U32.to_nat(stencil_columns(window))),U32.to_nat(word_capacity)) == True{} : Bool},
    area_room: {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(stencil_rows(window)),U32.to_nat(stencil_columns(window)))),U32.to_nat(word_capacity)) == True{} : Bool},
    inner_room: {Nat.is_le(Nat.double(U32.to_nat(stencil_inner(window))),U32.to_nat(word_capacity)) == True{} : Bool},
    width_room: {Nat.is_le(Nat.double(U32.to_nat(stencil_width(window))),U32.to_nat(word_capacity)) == True{} : Bool}}

def window(context: Geometry) -> Packing.Stencil:
  Geometry{window,channels,capacity,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  window

def row_word(context: Geometry) -> U32:
  stencil_rows(window(context))

def column_word(context: Geometry) -> U32:
  stencil_columns(window(context))

def rows(context: Geometry) -> Nat:
  U32.to_nat(row_word(context))

def columns(context: Geometry) -> Nat:
  U32.to_nat(column_word(context))
