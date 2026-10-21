# Pixel reads of an entry whose tile crosses an output row.
import Base

def masked(value: F32, inside: Bool) -> F32:
  match inside:
    case True{}:
      value
    case False{}:
      0.0

def mask_pixel(inside: Bool, pair: Array<F32> & F32) -> Array<F32> & F32:
  (input_storage, value) = pair
  (input_storage, masked(value, inside))

def pixel(input_storage: Array<F32>, +output_position: U32, +reduction_index: U32, +input_height: U32, +input_width: U32, +kernel_size: U32, +stride: U32, +padding: U32, +output_width: U32) -> Array<F32> & F32:
  +input_row = ((output_position / output_width) * stride + (reduction_index / kernel_size) % kernel_size - padding : U32)
  +input_column = ((output_position % output_width) * stride + reduction_index % kernel_size - padding : U32)
  at = ((reduction_index / (kernel_size * kernel_size)) * input_height * input_width + input_row * input_width + input_column : U32)
  mask_pixel(U32.is_lt(input_row, input_height) && U32.is_lt(input_column, input_width), Array.get(F32, input_storage, at))

type PixelAddress is Data:
  PixelAddress{index: U32,inside: Bool}

def read_pixel_address(input: Array<F32>,address: PixelAddress) -> Array<F32> & F32:
  PixelAddress{index,inside} = address
  mask_pixel(inside,Array.get(F32,input,index))

# Reduction coordinates are invariant across the whole block, at any width.
type WindowPixelBlock is Data:
  WindowPixelBlock{position: U32,kernel_row: U32,kernel_column: U32,channel_base: U32,height: U32,width: U32,stride: U32,row_origin: U32,column_origin: U32,output_width: U32}

def window_pixel_block(position: U32,+reduction: U32,+base: U32,+height: U32,+width: U32,+tap_rows: U32,+tap_columns: U32,stride: U32,row_origin: U32,column_origin: U32,output_width: U32) -> WindowPixelBlock:
  WindowPixelBlock{position,(reduction / tap_columns % tap_rows : U32),(reduction % tap_columns : U32),
    (base + reduction / (tap_rows * tap_columns) * height * width : U32),height,width,stride,row_origin,column_origin,output_width}

def window_pixel_address(context: WindowPixelBlock,index: Nat) -> PixelAddress:
  WindowPixelBlock{position,kernel_row,kernel_column,channel_base,+height,+width,+stride,+row_origin,+column_origin,+output_width} = context
  +position = (position + U32.from_nat(index) : U32)
  +row = (position / output_width * stride + kernel_row + row_origin : U32)
  +column = (position % output_width * stride + kernel_column + column_origin : U32)
  PixelAddress{(channel_base + row * width + column : U32),U32.is_lt(row,height) && U32.is_lt(column,width)}

def read_window_pixel(input: Array<F32>,context: WindowPixelBlock,index: Nat) -> Array<F32> & F32:
  read_pixel_address(input,window_pixel_address(context,index))

def window_pixel(input: Array<F32>,position: U32,reduction: U32,base: U32,height: U32,width: U32,tap_rows: U32,tap_columns: U32,stride: U32,row_origin: U32,column_origin: U32,output_width: U32) -> Array<F32> & F32:
  read_window_pixel(input,window_pixel_block(position,reduction,base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,output_width),0n)
