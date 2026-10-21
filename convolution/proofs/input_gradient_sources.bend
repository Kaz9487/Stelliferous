# The specification observes image slices of the actual retained buffer contents.
import Base
import ../../lib/kernel_spatial_packing.bend as Spatial
import ../../lib/proofs/kernel_weights_readback.bend as MappedSamples
import ../../lib/proofs/kernel_packing_window_coordinates.bend as WindowCoordinates
import ../../lib/proofs/storage_shape.bend as Shape
import ../specification.bend as ConvSpec
import ../input_gradient_specification.bend as Spec
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_adjoint_index.bend as WeightIndex
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/padded_interval.bend as Padded
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_phase_weights.bend as Weights
import ../../lib/proofs/input_gradient_phase_window_weights.bend as WindowWeights
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Patches
import ../../lib/kernel_rows.bend as Rows
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/input_gradient_phase_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/input_gradient_phase_taps.bend as Taps
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./input_gradient_order.bend as StridedOrder
import ./band_window_contents.bend as Contents


type Reader is Data:
  Reader{gradient: Buffer.Content<F32>,weights: Buffer.Content<F32>}

def models_reader(gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>) -> Reader:
  Reader{Buffer.content(~F32,gradient),Buffer.content(~F32,weights)}

def phase_geometry(context: Kernel.Phase) -> Kernel.Geometry:
  Kernel.Phase{geometry,y,x,inner} = context
  geometry

def shape(geometry: Kernel.Geometry) -> ConvSpec.Shape:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  ConvSpec.Shape{U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)}

def bare(geometry: Kernel.Geometry) -> Spec.Problem:
  Spec.Problem{shape(geometry),[],[]}

def gradient_size(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw)))

def weight_size(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))

def source_value(content: Buffer.Content<F32>,index: U32) -> F32:
  Buffer.value(~F32,index,content)

def problem(+geometry: Kernel.Geometry,reader: Reader,+image: Nat) -> Spec.Problem:
  Reader{gradient,weights} = reader
  Spec.Problem{shape(geometry),Contents.image_values(~Buffer.Content<F32>,~source_value,gradient_size(geometry),Nat.mul(image,gradient_size(geometry)),gradient),
    Contents.image_values(~Buffer.Content<F32>,~source_value,weight_size(geometry),0n,weights)}

def weight_content(reader: Reader) -> Buffer.Content<F32>:
  Reader{gradient,weights} = reader
  weights

def gradient_content(reader: Reader) -> Buffer.Content<F32>:
  Reader{gradient,weights} = reader
  gradient

def gradient_height(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(gh)

def gradient_width(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(gw)

def gradient_rows(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  gh

def gradient_columns(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  gw

def channels(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  c

def kernel(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  k

def weight(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+channel: Nat,+row: Nat,+column: Nat,
  output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},channel_inside: {Nat.is_lt(channel,Spec.channels(bare(geometry))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,Spec.kernel_size(bare(geometry))) == True{} : Bool},column_inside: {Nat.is_lt(column,Spec.kernel_size(bare(geometry))) == True{} : Bool}) ->
  {Spec.weight(problem(geometry,reader,image),output,channel,row,column) == Buffer.value(~F32,U32.from_nat(WeightIndex.index(channels(geometry),kernel(geometry),output,channel,row,column)),weight_content(reader)) : F32}:
  Kernel.Geometry{+c,+o,h,w,+k,s,p,gh,gw} = geometry
  Reader{gradient,+weights} = reader
  Contents.image_at(~Buffer.Content<F32>,~source_value,weight_size(geometry),0n,weights,WeightIndex.index(c,k,output,channel,row,column),
    %WeightIndex.elements_four(o,c,k) : {Nat.is_lt(WeightIndex.index(c,k,output,channel,row,column),_) == True{} : Bool}
    WeightIndex.inside(o,c,k,output,channel,row,column,output_inside,channel_inside,row_inside,column_inside))

def gradient(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+row: Nat,+column: Nat,
  output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},row_inside: {Nat.is_lt(row,gradient_height(geometry)) == True{} : Bool},
  column_inside: {Nat.is_lt(column,gradient_width(geometry)) == True{} : Bool}) ->
  {ConvSpec.at(RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),row,column),Spec.output_gradient(problem(geometry,reader,image))) ==
    Buffer.value(~F32,U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),row,column))),gradient_content(reader)) : F32}:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  Reader{+gradient,weights} = reader
  Contents.image_at(~Buffer.Content<F32>,~source_value,gradient_size(geometry),Nat.mul(image,gradient_size(geometry)),gradient,RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),row,column),
    RowMajor.in_bounds(output,U32.to_nat(o),U32.to_nat(gh),U32.to_nat(gw),row,column,output_inside,row_inside,column_inside))

def gradient_chosen(enabled: Bool,+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+row: U32,+column: U32,+logical_row: Nat,+logical_column: Nat,
  +output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},
  +row_exact: Boolean.When(enabled,{U32.to_nat(row) == logical_row : Nat}),+column_exact: Boolean.When(enabled,{U32.to_nat(column) == logical_column : Nat}),
  +valid: {U32.is_lt(row,gradient_rows(geometry)) && U32.is_lt(column,gradient_columns(geometry)) == enabled : Bool}) ->
  {Bool.pick(F32,enabled,Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + row * gradient_columns(geometry) + column : U32),gradient_content(reader)),0.0) ==
    ConvSpec.sample_if(enabled,RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),logical_row,logical_column),Spec.output_gradient(problem(geometry,reader,image))) : F32}:
  match enabled:
    case False{}: {==}
    case True{}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      Reader{+input,weights} = reader
      +base = Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(U32.to_nat(gh),U32.to_nat(gw))))
      +address = (U32.from_nat(base) + row * gw + column : U32)
      +global = Nat.add(Nat.mul(image,gradient_size(geometry)),RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),logical_row,logical_column))
      +row_inside = {
        %row_exact : {Nat.is_lt(_,U32.to_nat(gh)) == True{} : Bool}
        %Comparison.lt(row,gh) : {_ == True{} : Bool} Boolean.left(U32.is_lt(row,gh),U32.is_lt(column,gw),valid) : {Nat.is_lt(logical_row,U32.to_nat(gh)) == True{} : Bool}}
      +column_inside = {
        %column_exact : {Nat.is_lt(_,U32.to_nat(gw)) == True{} : Bool}
        %Comparison.lt(column,gw) : {_ == True{} : Bool} Boolean.right(U32.is_lt(row,gh),U32.is_lt(column,gw),valid) : {Nat.is_lt(logical_column,U32.to_nat(gw)) == True{} : Bool}}
      +encoded = {
        %Addition.associative(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(U32.to_nat(gh),U32.to_nat(gw))),Nat.add(Nat.mul(logical_row,U32.to_nat(gw)),logical_column)) : {address == U32.from_nat(_) : U32}
        %row_exact : {address == U32.from_nat(Nat.add(base,Nat.add(Nat.mul(_,U32.to_nat(gw)),logical_column))) : U32}
        %column_exact : {address == U32.from_nat(Nat.add(base,Nat.add(Nat.mul(U32.to_nat(row),U32.to_nat(gw)),_))) : U32}
        Encoded.row_offset_encoded(base,row,gw,column) : {address == U32.from_nat(global) : U32}}
      %Equal.sym(U32,address,U32.from_nat(global),encoded) :
        {Buffer.value(~F32,_,input) == ConvSpec.at(RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),logical_row,logical_column),Spec.output_gradient(problem(geometry,reader,image))) : F32}
      Equal.sym(F32,ConvSpec.at(RowMajor.index(output,U32.to_nat(gh),U32.to_nat(gw),logical_row,logical_column),Spec.output_gradient(problem(geometry,reader,image))),Buffer.value(~F32,U32.from_nat(global),input),gradient(geometry,reader,image,output,logical_row,logical_column,output_inside,row_inside,column_inside))

def wrapped_chosen(enabled: Bool,+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+y: U32,+x: U32,+tap_y: U32,+tap_x: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(gradient_height(geometry),U32.to_nat(tap_y)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(gradient_width(geometry),U32.to_nat(tap_x)),U32.to_nat(capacity)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},
  +valid: {U32.is_lt((y - tap_y : U32),gradient_rows(geometry)) && U32.is_lt((x - tap_x : U32),gradient_columns(geometry)) == enabled : Bool}) ->
  {Bool.pick(F32,enabled,Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (y - tap_y : U32) * gradient_columns(geometry) + (x - tap_x : U32) : U32),gradient_content(reader)),0.0) ==
    ConvSpec.sample_if(enabled,RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),Spec.output_gradient(problem(geometry,reader,image))) : F32}:
  match enabled:
    case False{}: {==}
    case True{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      gradient_chosen(True{},geometry,reader,image,output,(y - tap_y : U32),(x - tap_x : U32),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x)),output_inside,
        Padded.offset(y,tap_y,gradient_rows(geometry),capacity,capacity_ok,rows_room,Boolean.left(U32.is_lt((y - tap_y : U32),gradient_rows(geometry)),U32.is_lt((x - tap_x : U32),gradient_columns(geometry)),valid)),
        Padded.offset(x,tap_x,gradient_columns(geometry),capacity,capacity_ok,columns_room,Boolean.right(U32.is_lt((y - tap_y : U32),gradient_rows(geometry)),U32.is_lt((x - tap_x : U32),gradient_columns(geometry)),valid)),valid)

def wrapped_gradient(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+y: U32,+x: U32,+tap_y: U32,+tap_x: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(gradient_height(geometry),U32.to_nat(tap_y)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(gradient_width(geometry),U32.to_nat(tap_x)),U32.to_nat(capacity)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool}) ->
  {Bool.pick(F32,U32.is_lt((y - tap_y : U32),gradient_rows(geometry)) && U32.is_lt((x - tap_x : U32),gradient_columns(geometry)),
    Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (y - tap_y : U32) * gradient_columns(geometry) + (x - tap_x : U32) : U32),gradient_content(reader)),0.0) ==
    ConvSpec.sample_if((Nat.is_le(U32.to_nat(tap_y),U32.to_nat(y)) && Nat.is_lt(U32.to_nat(y),Nat.add(U32.to_nat(tap_y),gradient_height(geometry)))) &&
      (Nat.is_le(U32.to_nat(tap_x),U32.to_nat(x)) && Nat.is_lt(U32.to_nat(x),Nat.add(U32.to_nat(tap_x),gradient_width(geometry)))),
      RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),Spec.output_gradient(problem(geometry,reader,image))) : F32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +gy = U32.is_lt((y - tap_y : U32),gradient_rows(geometry))
  +gx = U32.is_lt((x - tap_x : U32),gradient_columns(geometry))
  %Padded.range(y,tap_y,gradient_rows(geometry),capacity,capacity_ok,rows_room) :
    {Bool.pick(F32,gy && gx,Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (y - tap_y : U32) * gradient_columns(geometry) + (x - tap_x : U32) : U32),gradient_content(reader)),0.0) ==
      ConvSpec.sample_if(_ && (Nat.is_le(U32.to_nat(tap_x),U32.to_nat(x)) && Nat.is_lt(U32.to_nat(x),Nat.add(U32.to_nat(tap_x),gradient_width(geometry)))),RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),Spec.output_gradient(problem(geometry,reader,image))) : F32}
  %Padded.range(x,tap_x,gradient_columns(geometry),capacity,capacity_ok,columns_room) :
    {Bool.pick(F32,gy && gx,Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (y - tap_y : U32) * gradient_columns(geometry) + (x - tap_x : U32) : U32),gradient_content(reader)),0.0) ==
      ConvSpec.sample_if(gy && _,RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),Spec.output_gradient(problem(geometry,reader,image))) : F32}
  wrapped_chosen(gy && gx,geometry,reader,image,output,y,x,tap_y,tap_x,capacity,capacity_ok,rows_room,columns_room,output_inside,{==})

def radix_sample(+rows: Nat,+columns: Nat,+stride: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat,+ry: Nat,+rx: Nat,+output: Nat,+gradient: List<&2,F32>,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {ConvSpec.sample_if(Nat.is_le(Nat.add(Nat.mul(row,stride),ry),Nat.add(Nat.mul(y,stride),ry)) &&
    Nat.is_lt(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.add(Nat.mul(row,stride),ry),Nat.mul(stride,rows))) &&
    Nat.is_le(Nat.add(Nat.mul(column,stride),rx),Nat.add(Nat.mul(x,stride),rx)) &&
    Nat.is_lt(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.add(Nat.mul(column,stride),rx),Nat.mul(stride,columns))),
    RowMajor.index(output,rows,columns,Nat.div(Nat.sub(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.mul(row,stride),ry)),stride),Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride)),gradient) ==
    ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}:
  +y_guard = Nat.is_le(Nat.add(Nat.mul(row,stride),ry),Nat.add(Nat.mul(y,stride),ry)) && Nat.is_lt(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.add(Nat.mul(row,stride),ry),Nat.mul(stride,rows)))
  +x_low = Nat.is_le(Nat.add(Nat.mul(column,stride),rx),Nat.add(Nat.mul(x,stride),rx))
  +x_high = Nat.is_lt(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.add(Nat.mul(column,stride),rx),Nat.mul(stride,columns)))
  %Equal.sym(Bool,(y_guard && x_low) && x_high,y_guard && (x_low && x_high),Boolean.and_associative(y_guard,x_low,x_high)) :
    {ConvSpec.sample_if(_,RowMajor.index(output,rows,columns,Nat.div(Nat.sub(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.mul(row,stride),ry)),stride),Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride)),gradient) ==
      ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}
  %Equal.sym(Bool,y_guard,Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows)),Padded.congruent_interval(y,row,stride,ry,rows,positive)) :
    {ConvSpec.sample_if(_ && (x_low && x_high),RowMajor.index(output,rows,columns,Nat.div(Nat.sub(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.mul(row,stride),ry)),stride),Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride)),gradient) ==
      ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}
  %Equal.sym(Bool,x_low && x_high,Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns)),Padded.congruent_interval(x,column,stride,rx,columns,positive)) :
    {ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && _,RowMajor.index(output,rows,columns,Nat.div(Nat.sub(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.mul(row,stride),ry)),stride),Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride)),gradient) ==
      ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}
  %Equal.sym(Nat,Nat.div(Nat.sub(Nat.add(Nat.mul(y,stride),ry),Nat.add(Nat.mul(row,stride),ry)),stride),Nat.sub(y,row),Padded.congruent_quotient(y,row,stride,ry,positive)) :
    {ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,_,Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride)),gradient) ==
      ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}
  %Equal.sym(Nat,Nat.div(Nat.sub(Nat.add(Nat.mul(x,stride),rx),Nat.add(Nat.mul(column,stride),rx)),stride),Nat.sub(x,column),Padded.congruent_quotient(x,column,stride,rx,positive)) :
    {ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),_),gradient) ==
      ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient) : F32}
  {==}

def congruent_sample(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat,
  +yq: U32,+xq: U32,+rq: U32,+cq: U32,+ry: Nat,+rx: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(gradient_height(geometry),U32.to_nat(rq)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(gradient_width(geometry),U32.to_nat(cq)),U32.to_nat(capacity)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Spec.stride(bare(geometry))) == True{} : Bool},
  +rows_exact: {gradient_height(geometry) == Spec.out_height(bare(geometry)) : Nat},
  +columns_exact: {gradient_width(geometry) == Spec.out_width(bare(geometry)) : Nat},
  +y_exact: {Nat.add(y,Spec.padding(bare(geometry))) == Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(bare(geometry))),ry) : Nat},
  +x_exact: {Nat.add(x,Spec.padding(bare(geometry))) == Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(bare(geometry))),rx) : Nat},
  +row_exact: {row == Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(bare(geometry))),ry) : Nat},
  +column_exact: {column == Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(bare(geometry))),rx) : Nat}) ->
  {Bool.pick(F32,U32.is_lt((yq - rq : U32),gradient_rows(geometry)) && U32.is_lt((xq - cq : U32),gradient_columns(geometry)),
    Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (yq - rq : U32) * gradient_columns(geometry) + (xq - cq : U32) : U32),gradient_content(reader)),0.0) ==
    Spec.sample(problem(geometry,reader,image),output,y,x,row,column) : F32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Reader{gradient,weights} = reader
  +products = problem(geometry,reader,image)
  +sample = ConvSpec.sample_if((Nat.is_le(U32.to_nat(rq),U32.to_nat(yq)) && Nat.is_lt(U32.to_nat(yq),Nat.add(U32.to_nat(rq),gradient_height(geometry)))) &&
    (Nat.is_le(U32.to_nat(cq),U32.to_nat(xq)) && Nat.is_lt(U32.to_nat(xq),Nat.add(U32.to_nat(cq),gradient_width(geometry)))),
    RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.sub(U32.to_nat(yq),U32.to_nat(rq)),Nat.sub(U32.to_nat(xq),U32.to_nat(cq))),Spec.output_gradient(products))
  Equal.trans(F32,
    Bool.pick(F32,U32.is_lt((yq - rq : U32),gradient_rows(geometry)) && U32.is_lt((xq - cq : U32),gradient_columns(geometry)),
      Buffer.value(~F32,(U32.from_nat(Nat.add(Nat.mul(image,gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (yq - rq : U32) * gradient_columns(geometry) + (xq - cq : U32) : U32),gradient_content(reader)),0.0),
    sample,Spec.sample(products,output,y,x,row,column),
    wrapped_gradient(geometry,reader,image,output,yq,xq,rq,cq,capacity,capacity_ok,rows_room,columns_room,output_inside),
    %rows_exact :
      {sample == ConvSpec.sample_if(Nat.is_le(row,Nat.add(y,Spec.padding(products))) && Nat.is_lt(Nat.add(y,Spec.padding(products)),Nat.add(row,Nat.mul(Spec.stride(products),_))) && Nat.is_le(column,Nat.add(x,Spec.padding(products))) && Nat.is_lt(Nat.add(x,Spec.padding(products)),Nat.add(column,Nat.mul(Spec.stride(products),Spec.out_width(products)))),
        RowMajor.index(output,_,Spec.out_width(products),Nat.div(Nat.sub(Nat.add(y,Spec.padding(products)),row),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(x,Spec.padding(products)),column),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    %columns_exact :
      {sample == ConvSpec.sample_if(Nat.is_le(row,Nat.add(y,Spec.padding(products))) && Nat.is_lt(Nat.add(y,Spec.padding(products)),Nat.add(row,Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(column,Nat.add(x,Spec.padding(products))) && Nat.is_lt(Nat.add(x,Spec.padding(products)),Nat.add(column,Nat.mul(Spec.stride(products),_))),
        RowMajor.index(output,gradient_height(geometry),_,Nat.div(Nat.sub(Nat.add(y,Spec.padding(products)),row),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(x,Spec.padding(products)),column),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    %Equal.sym(Nat,Nat.add(y,Spec.padding(bare(geometry))),Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(bare(geometry))),ry),y_exact) :
      {sample == ConvSpec.sample_if(Nat.is_le(row,_) && Nat.is_lt(_,Nat.add(row,Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(column,Nat.add(x,Spec.padding(products))) && Nat.is_lt(Nat.add(x,Spec.padding(products)),Nat.add(column,Nat.mul(Spec.stride(products),gradient_width(geometry)))),
        RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.div(Nat.sub(_,row),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(x,Spec.padding(products)),column),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    %Equal.sym(Nat,Nat.add(x,Spec.padding(bare(geometry))),Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(bare(geometry))),rx),x_exact) :
      {sample == ConvSpec.sample_if(Nat.is_le(row,Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(row,Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(column,_) && Nat.is_lt(_,Nat.add(column,Nat.mul(Spec.stride(products),gradient_width(geometry)))),
        RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),row),Spec.stride(products)),Nat.div(Nat.sub(_,column),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    %Equal.sym(Nat,row,Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(bare(geometry))),ry),row_exact) :
      {sample == ConvSpec.sample_if(Nat.is_le(_,Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(_,Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(column,Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),Nat.add(column,Nat.mul(Spec.stride(products),gradient_width(geometry)))),
        RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),_),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),column),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    %Equal.sym(Nat,column,Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(bare(geometry))),rx),column_exact) :
      {sample == ConvSpec.sample_if(Nat.is_le(Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry),Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry),Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(_,Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),Nat.add(_,Nat.mul(Spec.stride(products),gradient_width(geometry)))),
        RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry)),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),_),Spec.stride(products))),Spec.output_gradient(products)) : F32}
    Equal.sym(F32,
      ConvSpec.sample_if(Nat.is_le(Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry),Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry),Nat.mul(Spec.stride(products),gradient_height(geometry)))) && Nat.is_le(Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(products)),rx),Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx)) && Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),Nat.add(Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(products)),rx),Nat.mul(Spec.stride(products),gradient_width(geometry)))),
        RowMajor.index(output,gradient_height(geometry),gradient_width(geometry),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(products)),ry),Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(products)),ry)),Spec.stride(products)),Nat.div(Nat.sub(Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(products)),rx),Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(products)),rx)),Spec.stride(products))),Spec.output_gradient(products)),
      sample,radix_sample(gradient_height(geometry),gradient_width(geometry),Spec.stride(products),U32.to_nat(yq),U32.to_nat(xq),U32.to_nat(rq),U32.to_nat(cq),ry,rx,output,Spec.output_gradient(products),positive)))

def weight_index_flat(+channels: U32,+kernel: U32,+output: Nat,+channel: Nat,+row: Nat,+column: Nat) ->
  {WeightIndex.index(channels,kernel,output,channel,row,column) ==
    Nat.add(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)))),Nat.add(Nat.mul(row,U32.to_nat(kernel)),column)),Nat.mul(channel,Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel)))) : Nat}:
  +c = U32.to_nat(channels)
  +k = U32.to_nat(kernel)
  +square = Nat.mul(k,k)
  +base = Nat.mul(output,Nat.mul(c,square))
  +weight = Nat.mul(channel,square)
  +tap = Nat.add(Nat.mul(row,k),column)
  Equal.trans(Nat,WeightIndex.index(channels,kernel,output,channel,row,column),RowMajor.index(Nat.add(Nat.mul(output,c),channel),k,k,row,column),Nat.add(Nat.add(base,tap),weight),
    RowMajor.mixed(Nat.add(Nat.mul(output,c),channel),k,k,row,column),
    %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(output,c),channel),square),Nat.add(Nat.mul(Nat.mul(output,c),square),weight),RowMajor.distrib(Nat.mul(output,c),channel,square)) :
      {Nat.add(_,tap) == Nat.add(Nat.add(base,tap),weight) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.mul(output,c),square),base,RowMajor.assoc(output,c,square)) :
      {Nat.add(Nat.add(_,weight),tap) == Nat.add(Nat.add(base,tap),weight) : Nat}
    Equal.trans(Nat,Nat.add(Nat.add(base,weight),tap),Nat.add(base,Nat.add(weight,tap)),Nat.add(Nat.add(base,tap),weight),Addition.associative(base,weight,tap),
      Equal.trans(Nat,Nat.add(base,Nat.add(weight,tap)),Nat.add(base,Nat.add(tap,weight)),Nat.add(Nat.add(base,tap),weight),Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(weight,tap),Nat.add(tap,weight),Order.commutative(weight,tap)),Equal.sym(Nat,Nat.add(Nat.add(base,tap),weight),Nat.add(base,Nat.add(tap,weight)),Addition.associative(base,tap,weight)))))

def weight_sample_native(+geometry: Kernel.Geometry,+reader: Reader,+image: Nat,+output: Nat,+channel: U32,+lane: Nat,+row: Nat,+column: Nat,+base: U32,+spatial: U32,
  +output_inside: {Nat.is_lt(output,Spec.outputs(bare(geometry))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel),lane),Spec.channels(bare(geometry))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,Spec.kernel_size(bare(geometry))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Spec.kernel_size(bare(geometry))) == True{} : Bool},
  +base_exact: {base == U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(channels(geometry)),Nat.mul(U32.to_nat(kernel(geometry)),U32.to_nat(kernel(geometry))))),Nat.add(Nat.mul(row,U32.to_nat(kernel(geometry))),column))) : U32},
  +spatial_exact: {U32.to_nat(spatial) == Nat.mul(U32.to_nat(kernel(geometry)),U32.to_nat(kernel(geometry))) : Nat}) ->
  {Bool.pick(F32,U32.is_lt((channel + U32.from_nat(lane) : U32),channels(geometry)),Buffer.value(~F32,(base + (channel + U32.from_nat(lane) : U32) * spatial : U32),weight_content(reader)),0.0) ==
    Spec.weight(problem(geometry,reader,image),output,Nat.add(U32.to_nat(channel),lane),row,column) : F32}:
  Kernel.Geometry{+c,o,h,w,+k,s,p,gh,gw} = geometry
  +logical = Nat.add(U32.to_nat(channel),lane)
  +sum_exact = {
    %Arithmetic.native_roundtrip(channel) : {( _ + U32.from_nat(lane) : U32) == U32.from_nat(logical) : U32}
    WordArithmetic.native_add(U32.to_nat(channel),lane) : {(channel + U32.from_nat(lane) : U32) == U32.from_nat(logical) : U32}}
  +valid = {
    %Equal.sym(U32,(channel + U32.from_nat(lane) : U32),U32.from_nat(logical),sum_exact) : {U32.is_lt(_,c) == True{} : Bool}
    ListRead.inside(logical,c,channel_inside) : {U32.is_lt((channel + U32.from_nat(lane) : U32),c) == True{} : Bool}}
  +address = {
    %Equal.sym(U32,base,U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))),base_exact) :
      {(_ + (channel + U32.from_nat(lane) : U32) * spatial : U32) == U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)) : U32}
    %Equal.sym(U32,(channel + U32.from_nat(lane) : U32),U32.from_nat(logical),sum_exact) :
      {(U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))) + _ * spatial : U32) == U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)) : U32}
    %Arithmetic.native_roundtrip(spatial) :
      {(U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))) + U32.from_nat(logical) * _ : U32) == U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)) : U32}
    %Equal.sym(Nat,U32.to_nat(spatial),Nat.mul(U32.to_nat(k),U32.to_nat(k)),spatial_exact) :
      {(U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))) + U32.from_nat(logical) * U32.from_nat(_) : U32) == U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)) : U32}
    %Equal.sym(Nat,WeightIndex.index(c,k,output,logical,row,column),Nat.add(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column)),Nat.mul(logical,Nat.mul(U32.to_nat(k),U32.to_nat(k)))),weight_index_flat(c,k,output,logical,row,column)) :
      {(U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))) + U32.from_nat(logical) * U32.from_nat(Nat.mul(U32.to_nat(k),U32.to_nat(k))) : U32) == U32.from_nat(_) : U32}
    %Equal.sym(U32,(U32.from_nat(logical) * U32.from_nat(Nat.mul(U32.to_nat(k),U32.to_nat(k))) : U32),U32.from_nat(Nat.mul(logical,Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Multiplication.native_encoded_multiply(logical,Nat.mul(U32.to_nat(k),U32.to_nat(k)))) :
      {(U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column))) + _ : U32) == U32.from_nat(Nat.add(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column)),Nat.mul(logical,Nat.mul(U32.to_nat(k),U32.to_nat(k))))) : U32}
    WordArithmetic.native_add(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(row,U32.to_nat(k)),column)),Nat.mul(logical,Nat.mul(U32.to_nat(k),U32.to_nat(k)))) :
    {(base + (channel + U32.from_nat(lane) : U32) * spatial : U32) == U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)) : U32}}
  %Equal.sym(Bool,U32.is_lt((channel + U32.from_nat(lane) : U32),c),True{},valid) :
    {Bool.pick(F32,_,Buffer.value(~F32,(base + (channel + U32.from_nat(lane) : U32) * spatial : U32),weight_content(reader)),0.0) == Spec.weight(problem(geometry,reader,image),output,logical,row,column) : F32}
  %Equal.sym(U32,(base + (channel + U32.from_nat(lane) : U32) * spatial : U32),U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)),address) :
    {Buffer.value(~F32,_,weight_content(reader)) == Spec.weight(problem(geometry,reader,image),output,logical,row,column) : F32}
  Equal.sym(F32,Spec.weight(problem(geometry,reader,image),output,logical,row,column),Buffer.value(~F32,U32.from_nat(WeightIndex.index(c,k,output,logical,row,column)),weight_content(reader)),weight(geometry,reader,image,output,logical,row,column,output_inside,channel_inside,row_inside,column_inside))

def tile_weight(~upper: Bool,+row: Accumulator.Row,+context: Kernel.Phase,+image: Nat,+channel: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  +step_exact: {U32.to_nat(Weights.output_step(context)) == Nat.mul(U32.to_nat(Weights.channels(context)),Nat.mul(U32.to_nat(Weights.kernel(context)),U32.to_nat(Weights.kernel(context)))) : Nat},
  +square_exact: {U32.to_nat((Weights.kernel(context) * Weights.kernel(context) : U32)) == Nat.mul(U32.to_nat(Weights.kernel(context)),U32.to_nat(Weights.kernel(context))) : Nat},
  +output_inside: {Nat.is_lt(output,U32.to_nat(Packing.phase_outputs(context))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),U32.to_nat(Weights.channels(context))) == True{} : Bool},
  +row_inside: {Nat.is_lt(U32.to_nat((Weights.axis_tap(Packing.phase_y(context)) - U32.from_nat(Nat.mul(tap_row,U32.to_nat(Weights.stride(context)))) : U32)),U32.to_nat(Weights.kernel(context))) == True{} : Bool},
  +column_inside: {Nat.is_lt(U32.to_nat((Weights.axis_tap(Packing.phase_x(context)) - U32.from_nat(Nat.mul(tap_column,U32.to_nat(Weights.stride(context)))) : U32)),U32.to_nat(Weights.kernel(context))) == True{} : Bool}) ->
  {Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,Weights.axis_tap(Packing.phase_y(context)),Weights.axis_tap(Packing.phase_x(context)),context,output,tap_row,tap_column))) ==
    Spec.weight(problem(phase_geometry(context),models_reader(gradient,weights),image),output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),
      U32.to_nat((Weights.axis_tap(Packing.phase_y(context)) - U32.from_nat(Nat.mul(tap_row,U32.to_nat(Weights.stride(context)))) : U32)),
      U32.to_nat((Weights.axis_tap(Packing.phase_x(context)) - U32.from_nat(Nat.mul(tap_column,U32.to_nat(Weights.stride(context)))) : U32))) : F32}:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,+k,+s,p,gh,gw},Kernel.Axis{yf,yc,+yt,yr,ys},Kernel.Axis{xf,xc,+xt,xr,xs},inner} = context
  Buffer.Model{gradient_tree,gradient_length,gradient_certificate} = gradient
  Buffer.Model{weight_tree,weight_length,weight_certificate} = weights
  +tap_y = (yt - U32.from_nat(Nat.mul(tap_row,U32.to_nat(s))) : U32)
  +tap_x = (xt - U32.from_nat(Nat.mul(tap_column,U32.to_nat(s))) : U32)
  +base = ((0 + U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(context)))) : U32) + tap_y * k + tap_x : U32)
  +geometry = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +reader = models_reader(gradient,weights)
  +sample = {Packing.WeightSample{c,channel,base,(k * k : U32)} : Packing.WeightSample}
  +base_exact = {
    %step_exact : {base == U32.from_nat(Nat.add(Nat.mul(output,_),Nat.add(Nat.mul(U32.to_nat(tap_y),U32.to_nat(k)),U32.to_nat(tap_x)))) : U32}
    %Equal.sym(U32,(0 + U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(context)))) : U32),U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(context)))),Arithmetic.add_zero_left(U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(context)))))) :
      {(_ + tap_y * k + tap_x : U32) == U32.from_nat(Nat.add(Nat.mul(output,U32.to_nat(Weights.output_step(context))),Nat.add(Nat.mul(U32.to_nat(tap_y),U32.to_nat(k)),U32.to_nat(tap_x)))) : U32}
    Encoded.row_offset_encoded(Nat.mul(output,U32.to_nat(Weights.output_step(context))),tap_y,k,tap_x) :
    {base == U32.from_nat(Nat.add(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.add(Nat.mul(U32.to_nat(tap_y),U32.to_nat(k)),U32.to_nat(tap_x)))) : U32}}
  Equal.trans(F32,Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,yt,xt,context,output,tap_row,tap_column))),Packing.weight_value(weights,sample,Patches.patch_key(upper,row)),
    Spec.weight(problem(geometry,reader,image),output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),U32.to_nat(tap_y),U32.to_nat(tap_x)),
    Packing.weight_tile_half(upper,row,weights,sample),weight_sample_native(geometry,reader,image,output,channel,Patches.patch_key(upper,row),U32.to_nat(tap_y),U32.to_nat(tap_x),base,(k * k : U32),output_inside,channel_inside,row_inside,column_inside,base_exact,square_exact))

def tile_gradient(+context: Kernel.Phase,+image: U32,+local_row: U32,+local_column: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat,+lane: Nat,
  +y: Nat,+x: Nat,+row: Nat,+column: Nat,+yq: U32,+xq: U32,+rq: U32,+cq: U32,+ry: Nat,+rx: Nat,+capacity: U32,
  +within: {Nat.is_lt(lane,8n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(gradient_height(phase_geometry(context)),U32.to_nat(rq)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(gradient_width(phase_geometry(context)),U32.to_nat(cq)),U32.to_nat(capacity)) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,U32.to_nat(Packing.phase_outputs(context))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Spec.stride(bare(phase_geometry(context)))) == True{} : Bool},
  +rows_exact: {gradient_height(phase_geometry(context)) == Spec.out_height(bare(phase_geometry(context))) : Nat},
  +columns_exact: {gradient_width(phase_geometry(context)) == Spec.out_width(bare(phase_geometry(context))) : Nat},
  +y_exact: {Nat.add(y,Spec.padding(bare(phase_geometry(context)))) == Nat.add(Nat.mul(U32.to_nat(yq),Spec.stride(bare(phase_geometry(context)))),ry) : Nat},
  +x_exact: {Nat.add(x,Spec.padding(bare(phase_geometry(context)))) == Nat.add(Nat.mul(U32.to_nat(xq),Spec.stride(bare(phase_geometry(context)))),rx) : Nat},
  +row_exact: {row == Nat.add(Nat.mul(U32.to_nat(rq),Spec.stride(bare(phase_geometry(context)))),ry) : Nat},
  +column_exact: {column == Nat.add(Nat.mul(U32.to_nat(cq),Spec.stride(bare(phase_geometry(context)))),rx) : Nat},
  +base_exact: {(Packing.input_base(context,image) + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(context)))) : U32) ==
    U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),gradient_size(phase_geometry(context))),Nat.mul(output,Nat.mul(gradient_height(phase_geometry(context)),gradient_width(phase_geometry(context)))))) : U32},
  +row_counter: {(Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32) == (yq - rq : U32) : U32},
  +column_counter: {((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32) == (xq - cq : U32) : U32}) ->
  {Tile.element_at(TileLaws.lane_of(lane),Packing.input_tile(context,image,local_row,local_column,gradient,output,tap_row,tap_column)) ==
    Spec.sample(problem(phase_geometry(context),models_reader(gradient,weights),U32.to_nat(image)),output,y,x,row,column) : F32}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw},ay,ax,inner} = context
  Buffer.Model{tree,length,certificate} = gradient
  +geometry = phase_geometry(context)
  +reader = models_reader(gradient,weights)
  +sample = {Packing.Sample{(Packing.input_base(context,image) + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(context)))) : U32),
    (Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32),(Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32),gh,gw} : Packing.Sample}
  Equal.trans(F32,Tile.element_at(TileLaws.lane_of(lane),Packing.input_tile(context,image,local_row,local_column,gradient,output,tap_row,tap_column)),Packing.sample_value(gradient,sample,lane),
    Spec.sample(problem(geometry,reader,U32.to_nat(image)),output,y,x,row,column),Equal.trans(F32,Tile.element_at(TileLaws.lane_of(lane),Packing.sample_tile(gradient,sample)),TileLaws.select_lane(lane,Packing.sample_tile(gradient,sample)),Packing.sample_value(gradient,sample,lane),Equal.sym(F32,TileLaws.select_lane(lane,Packing.sample_tile(gradient,sample)),Tile.element_at(TileLaws.lane_of(lane),Packing.sample_tile(gradient,sample)),TileLaws.selected_lane(lane,Packing.sample_tile(gradient,sample))),Packing.sample_tile_lane(gradient,sample,lane,within)),
    %Equal.sym(U32,(Packing.input_base(context,image) + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(context)))) : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))),base_exact) :
      {Bool.pick(F32,U32.is_lt((Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32),gh) && U32.is_lt(((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32),gw),Tree.read(F32,tree,(_ + (Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32) * gw + ((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32) : U32)),0.0) == Spec.sample(problem(geometry,reader,U32.to_nat(image)),output,y,x,row,column) : F32}
    %Equal.sym(U32,(Packing.input_row(context,local_row) + U32.from_nat(tap_row) : U32),(yq - rq : U32),row_counter) :
      {Bool.pick(F32,U32.is_lt(_,gh) && U32.is_lt(((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32),gw),Tree.read(F32,tree,(U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + _ * gw + ((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32) : U32)),0.0) == Spec.sample(problem(geometry,reader,U32.to_nat(image)),output,y,x,row,column) : F32}
    %Equal.sym(U32,((Packing.input_column(context,local_column) + U32.from_nat(tap_column) : U32) + U32.from_nat(lane) : U32),(xq - cq : U32),column_counter) :
      {Bool.pick(F32,U32.is_lt((yq - rq : U32),gh) && U32.is_lt(_,gw),Tree.read(F32,tree,(U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),gradient_size(geometry)),Nat.mul(output,Nat.mul(gradient_height(geometry),gradient_width(geometry))))) + (yq - rq : U32) * gw + _ : U32)),0.0) == Spec.sample(problem(geometry,reader,U32.to_nat(image)),output,y,x,row,column) : F32}
    congruent_sample(geometry,reader,U32.to_nat(image),output,y,x,row,column,yq,xq,rq,cq,ry,rx,capacity,capacity_ok,rows_room,columns_room,output_inside,positive,rows_exact,columns_exact,y_exact,x_exact,row_exact,column_exact))

def padded_pixel_residue(+first: Nat,+stride: Nat,+padding: Nat,+position: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.add(first,Nat.mul(position,stride)),padding),stride) == Nat.mod(Nat.add(first,padding),stride) : Nat}:
  +step = Nat.mul(position,stride)
  +reordered = Equal.trans(Nat,Nat.add(Nat.add(first,step),padding),Nat.add(Nat.add(step,first),padding),Nat.add(step,Nat.add(first,padding)),
    Equal.cong(Nat,Nat,pixel => Nat.add(pixel,padding),Nat.add(first,step),Nat.add(step,first),Order.commutative(first,step)),Addition.associative(step,first,padding))
  Equal.trans(Nat,Nat.mod(Nat.add(Nat.add(first,step),padding),stride),Nat.mod(Nat.add(step,Nat.add(first,padding)),stride),Nat.mod(Nat.add(first,padding),stride),
    Equal.cong(Nat,Nat,pixel => Nat.mod(pixel,stride),Nat.add(Nat.add(first,step),padding),Nat.add(step,Nat.add(first,padding)),reordered),DivisionOrder.remainder_shift(Nat.add(first,padding),stride,position,positive))

def axis_spec_residue(+first: U32,+stride: U32,+padding: U32,+position: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(Geometry.residue(first,stride,padding)) == Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)) : Nat}:
  Equal.trans(Nat,U32.to_nat(Geometry.residue(first,stride,padding)),Nat.mod(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(stride)),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),
    Geometry.residue_exact(first,stride,padding,capacity,capacity_ok,padded_room,stride_room,positive),Equal.sym(Nat,Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),Nat.mod(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(stride)),padded_pixel_residue(U32.to_nat(first),U32.to_nat(stride),U32.to_nat(padding),position,positive)))

def axis_spec_count(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))) ==
    Taps.count(U32.to_nat(kernel),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),U32.to_nat(stride)) : Nat}:
  Equal.trans(Nat,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))),Geometry.tap_count(kernel,Geometry.residue(first,stride,padding),stride),
    Taps.count(U32.to_nat(kernel),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),U32.to_nat(stride)),
    Geometry.taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive),
    Equal.cong(Nat,Nat,residue => Taps.count(U32.to_nat(kernel),residue,U32.to_nat(stride)),U32.to_nat(Geometry.residue(first,stride,padding)),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),axis_spec_residue(first,stride,padding,position,capacity,capacity_ok,padded_room,stride_room,positive)))

def axis_spec_tap(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: Nat,+offset: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +inside: {Nat.is_lt(offset,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.to_nat(PhaseBounds.tap(first,size,kernel,stride,padding,offset)) ==
    Nat.add(Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),
      Nat.mul(Nat.sub(Nat.sub(Taps.count(U32.to_nat(kernel),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),U32.to_nat(stride)),1n),offset),U32.to_nat(stride))) : Nat}:
  +residue = U32.to_nat(Geometry.residue(first,stride,padding))
  +count = U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)))
  +active = PhaseBounds.active(first,size,kernel,stride,padding,offset,inside)
  +count_exact = Geometry.active_taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)
  +tap_exact = Geometry.axis_offset_exact(first,size,kernel,stride,padding,offset,capacity,capacity_ok,kernel_room,stride_room,positive,active,inside)
  %axis_spec_residue(first,stride,padding,position,capacity,capacity_ok,padded_room,stride_room,positive) :
    {U32.to_nat(PhaseBounds.tap(first,size,kernel,stride,padding,offset)) == Nat.add(_,Nat.mul(Nat.sub(Nat.sub(Taps.count(U32.to_nat(kernel),Nat.mod(Nat.add(Nat.add(U32.to_nat(first),Nat.mul(position,U32.to_nat(stride))),U32.to_nat(padding)),U32.to_nat(stride)),U32.to_nat(stride)),1n),offset),U32.to_nat(stride))) : Nat}
  %axis_spec_count(first,size,kernel,stride,padding,position,capacity,capacity_ok,padded_room,kernel_room,stride_room,positive) :
    {U32.to_nat(PhaseBounds.tap(first,size,kernel,stride,padding,offset)) == Nat.add(residue,Nat.mul(Nat.sub(Nat.sub(_,1n),offset),U32.to_nat(stride))) : Nat}
  %Equal.sym(Nat,count,1n+Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride)),count_exact) :
    {U32.to_nat(PhaseBounds.tap(first,size,kernel,stride,padding,offset)) == Nat.add(residue,Nat.mul(Nat.sub(Nat.sub(_,1n),offset),U32.to_nat(stride))) : Nat}
  %Equal.sym(Nat,Nat.sub(Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride)),0n),Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride)),Order.subtract_zero(Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride)))) :
    {U32.to_nat(PhaseBounds.tap(first,size,kernel,stride,padding,offset)) == Nat.add(residue,Nat.mul(Nat.sub(_,offset),U32.to_nat(stride))) : Nat}
  tap_exact

def tile_product_given(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+call: Domain.Call,
  +first_y: U32,+first_x: U32,+image: U32,+channel: U32,+local_row: U32,+local_column: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  +first_y_inside: {Nat.is_lt(U32.to_nat(first_y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +first_x_inside: {Nat.is_lt(U32.to_nat(first_x),Domain.width(Domain.geometry(call))) == True{} : Bool},
  +position_y_inside: {Nat.is_lt(U32.to_nat(local_row),U32.to_nat(Geometry.axis_count(Packing.phase_y(Kernel.phase(Domain.geometry(call),first_y,first_x))))) == True{} : Bool},
  +position_x_inside: {Nat.is_lt(Nat.add(U32.to_nat(local_column),lane),U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),first_y,first_x))))) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(tap_row,U32.to_nat(Packing.phase_rows(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(tap_column,U32.to_nat(Packing.phase_columns(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),Domain.channels(Domain.geometry(call))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +yfacts: PhaseBounds.AxisFacts<first_y,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),local_row,tap_row,Domain.gradient_rows(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>,
  +xfacts: PhaseBounds.AxisFacts<first_x,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32),tap_column,Domain.gradient_columns(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>) ->
  {ConvSpec.Product{Tile.element_at(TileLaws.lane_of(lane),Packing.input_tile(Kernel.phase(Domain.geometry(call),first_y,first_x),image,local_row,local_column,gradient,output,tap_row,tap_column)),
    Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,Weights.axis_tap(Packing.phase_y(Kernel.phase(Domain.geometry(call),first_y,first_x))),Weights.axis_tap(Packing.phase_x(Kernel.phase(Domain.geometry(call),first_y,first_x))),Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)))} ==
    StridedOrder.flat_product(StridedOrder.Flat{problem(Domain.geometry(call),models_reader(gradient,weights),U32.to_nat(image)),Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),PhaseBounds.pixel(first_y,Domain.stride_word(Domain.geometry(call)),local_row),PhaseBounds.pixel(first_x,Domain.stride_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32))},Packing.phase_index(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)) : ConvSpec.Product}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Buffer.Model{gradient_tree,gradient_length,gradient_certificate} = gradient
  Buffer.Model{weight_tree,weight_length,weight_certificate} = weights
  PhaseBounds.AxisFacts{yrin,ypin,+yrad,+ytin,+ytrad,+ycounter,+ysroom} = yfacts
  PhaseBounds.AxisFacts{xrin,xpin,+xrad,+xtin,+xtrad,+xcounter,+xsroom} = xfacts
  +kernel_coordinate_room = PhaseBounds.kernel_coordinate_room(call)
  +stride_double_room = PhaseBounds.stride_double_room(call)
  +context = Kernel.phase(geometry,first_y,first_x)
  +capacity = Machine.coordinate(limits)
  +capacity_ok = Limits.coordinate_fits(limits)
  +positive = Domain.stride_positive(call)
  +x_position = (local_column + U32.from_nat(lane) : U32)
  +y = PhaseBounds.pixel(first_y,s,local_row)
  +x = PhaseBounds.pixel(first_x,s,x_position)
  +ytap = PhaseBounds.tap(first_y,h,k,s,p,tap_row)
  +xtap = PhaseBounds.tap(first_x,w,k,s,p,tap_column)
  +problem = problem(geometry,models_reader(gradient,weights),U32.to_nat(image))
  +flat = {StridedOrder.Flat{problem,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),y,x} : StridedOrder.Flat}
  +first_y_padded = Order.transitive(Nat.add(U32.to_nat(first_y),U32.to_nat(p)),Nat.add(U32.to_nat(h),U32.to_nat(p)),U32.to_nat(capacity),Order.add_le(U32.to_nat(first_y),U32.to_nat(h),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_y),U32.to_nat(h),first_y_inside)),PhaseBounds.padded_height_room(call))
  +first_x_padded = Order.transitive(Nat.add(U32.to_nat(first_x),U32.to_nat(p)),Nat.add(U32.to_nat(w),U32.to_nat(p)),U32.to_nat(capacity),Order.add_le(U32.to_nat(first_x),U32.to_nat(w),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_x),U32.to_nat(w),first_x_inside)),PhaseBounds.padded_width_room(call))
  +rows = axis_spec_count(first_y,h,k,s,p,U32.to_nat(local_row),capacity,capacity_ok,first_y_padded,kernel_coordinate_room,stride_double_room,positive)
  +columns = axis_spec_count(first_x,w,k,s,p,U32.to_nat(x_position),capacity,capacity_ok,first_x_padded,kernel_coordinate_room,stride_double_room,positive)
  +y_spec = axis_spec_tap(first_y,h,k,s,p,U32.to_nat(local_row),tap_row,capacity,capacity_ok,first_y_padded,kernel_coordinate_room,stride_double_room,positive,row_inside)
  +x_spec = axis_spec_tap(first_x,w,k,s,p,U32.to_nat(x_position),tap_column,capacity,capacity_ok,first_x_padded,kernel_coordinate_room,stride_double_room,positive,column_inside)
  +input_value = tile_gradient(context,image,local_row,local_column,gradient,weights,output,tap_row,tap_column,lane,y,x,U32.to_nat(ytap),U32.to_nat(xtap),PhaseBounds.pixel_quotient(first_y,s,p,local_row),PhaseBounds.pixel_quotient(first_x,s,p,x_position),PhaseBounds.tap_quotient(first_y,h,k,s,p,tap_row),PhaseBounds.tap_quotient(first_x,w,k,s,p,tap_column),U32.to_nat(Geometry.residue(first_y,s,p)),U32.to_nat(Geometry.residue(first_x,s,p)),capacity,
    lane_inside,capacity_ok,ysroom,xsroom,output_inside,positive,Domain.rows_exact(call),Domain.columns_exact(call),yrad,xrad,ytrad,xtrad,PhaseBounds.gradient_base_exact(call,first_y,first_x,image,output),ycounter,PhaseBounds.column_lane_counter(geometry,first_y,first_x,local_column,lane,tap_column,xcounter))
  +weight_value = tile_weight(~upper,row,context,U32.to_nat(image),channel,gradient,weights,output,tap_row,tap_column,PhaseBounds.weight_step_exact(call,first_y,first_x),PhaseBounds.weight_square_exact(call,first_y,first_x),output_inside,channel_inside,ytin,xtin)
  +native_input = Tile.element_at(TileLaws.lane_of(lane),Packing.input_tile(context,image,local_row,local_column,gradient,output,tap_row,tap_column))
  +native_weight = Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,Weights.axis_tap(Packing.phase_y(context)),Weights.axis_tap(Packing.phase_x(context)),context,output,tap_row,tap_column)))
  +input = Spec.sample(problem,output,y,x,U32.to_nat(ytap),U32.to_nat(xtap))
  +weight = Spec.weight(problem,output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),U32.to_nat(ytap),U32.to_nat(xtap))
  +spec_product = {ConvSpec.Product{Spec.sample(problem,output,y,x,StridedOrder.flat_row_tap(flat,tap_row),StridedOrder.flat_column_tap(flat,tap_column)),Spec.weight(problem,output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),StridedOrder.flat_row_tap(flat,tap_row),StridedOrder.flat_column_tap(flat,tap_column))} : ConvSpec.Product}
  +index_exact = Equal.trans(Nat,Packing.phase_index(context,output,tap_row,tap_column),RowMajor.index(output,StridedOrder.flat_rows(flat),U32.to_nat(Packing.phase_columns(context)),tap_row,tap_column),StridedOrder.flat_index(flat,output,tap_row,tap_column),
    Equal.cong(Nat,Nat,rows => RowMajor.index(output,rows,U32.to_nat(Packing.phase_columns(context)),tap_row,tap_column),U32.to_nat(Packing.phase_rows(context)),StridedOrder.flat_rows(flat),rows),
    Equal.cong(Nat,Nat,columns => RowMajor.index(output,StridedOrder.flat_rows(flat),columns,tap_row,tap_column),U32.to_nat(Packing.phase_columns(context)),StridedOrder.flat_columns(flat),columns))
  Equal.trans(ConvSpec.Product,ConvSpec.Product{native_input,native_weight},ConvSpec.Product{input,weight},StridedOrder.flat_product(flat,Packing.phase_index(context,output,tap_row,tap_column)),
    Equal.trans(ConvSpec.Product,ConvSpec.Product{native_input,native_weight},ConvSpec.Product{input,native_weight},ConvSpec.Product{input,weight},Equal.cong(F32,ConvSpec.Product,input => ConvSpec.Product{input,native_weight},native_input,input,input_value),Equal.cong(F32,ConvSpec.Product,weight => ConvSpec.Product{input,weight},native_weight,weight,weight_value)),
    Equal.trans(ConvSpec.Product,ConvSpec.Product{input,weight},spec_product,StridedOrder.flat_product(flat,Packing.phase_index(context,output,tap_row,tap_column)),
      Equal.trans(ConvSpec.Product,ConvSpec.Product{input,weight},ConvSpec.Product{Spec.sample(problem,output,y,x,StridedOrder.flat_row_tap(flat,tap_row),U32.to_nat(xtap)),Spec.weight(problem,output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),StridedOrder.flat_row_tap(flat,tap_row),U32.to_nat(xtap))},spec_product,
        Equal.cong(Nat,ConvSpec.Product,tap => ConvSpec.Product{Spec.sample(problem,output,y,x,tap,U32.to_nat(xtap)),Spec.weight(problem,output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),tap,U32.to_nat(xtap))},U32.to_nat(ytap),StridedOrder.flat_row_tap(flat,tap_row),y_spec),
        Equal.cong(Nat,ConvSpec.Product,tap => ConvSpec.Product{Spec.sample(problem,output,y,x,StridedOrder.flat_row_tap(flat,tap_row),tap),Spec.weight(problem,output,Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),StridedOrder.flat_row_tap(flat,tap_row),tap)},U32.to_nat(xtap),StridedOrder.flat_column_tap(flat,tap_column),x_spec)),
      %Equal.sym(Nat,Packing.phase_index(context,output,tap_row,tap_column),StridedOrder.flat_index(flat,output,tap_row,tap_column),index_exact) :
        {spec_product == StridedOrder.flat_product(flat,_) : ConvSpec.Product}
      Equal.sym(ConvSpec.Product,StridedOrder.flat_product(flat,StridedOrder.flat_index(flat,output,tap_row,tap_column)),spec_product,
        StridedOrder.flat_product_at(flat,output,tap_row,tap_column,%rows : {Nat.is_lt(tap_row,_) == True{} : Bool} row_inside,%columns : {Nat.is_lt(tap_column,_) == True{} : Bool} column_inside))))

def tile_product_spec(~upper: Bool,+row: Accumulator.Row,+lane: Nat,+call: Domain.Call,
  +first_y: U32,+first_x: U32,+image: U32,+channel: U32,+local_row: U32,+local_column: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  +first_y_inside: {Nat.is_lt(U32.to_nat(first_y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +first_x_inside: {Nat.is_lt(U32.to_nat(first_x),Domain.width(Domain.geometry(call))) == True{} : Bool},
  +position_y_inside: {Nat.is_lt(U32.to_nat(local_row),U32.to_nat(Geometry.axis_count(Packing.phase_y(Kernel.phase(Domain.geometry(call),first_y,first_x))))) == True{} : Bool},
  +position_x_inside: {Nat.is_lt(Nat.add(U32.to_nat(local_column),lane),U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),first_y,first_x))))) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Domain.geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(tap_row,U32.to_nat(Packing.phase_rows(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(tap_column,U32.to_nat(Packing.phase_columns(Kernel.phase(Domain.geometry(call),first_y,first_x)))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),Domain.channels(Domain.geometry(call))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {ConvSpec.Product{Tile.element_at(TileLaws.lane_of(lane),Packing.input_tile(Kernel.phase(Domain.geometry(call),first_y,first_x),image,local_row,local_column,gradient,output,tap_row,tap_column)),
    Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,Weights.axis_tap(Packing.phase_y(Kernel.phase(Domain.geometry(call),first_y,first_x))),Weights.axis_tap(Packing.phase_x(Kernel.phase(Domain.geometry(call),first_y,first_x))),Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)))} ==
    StridedOrder.flat_product(StridedOrder.Flat{problem(Domain.geometry(call),models_reader(gradient,weights),U32.to_nat(image)),Nat.add(U32.to_nat(channel),Patches.patch_key(upper,row)),PhaseBounds.pixel(first_y,Domain.stride_word(Domain.geometry(call)),local_row),PhaseBounds.pixel(first_x,Domain.stride_word(Domain.geometry(call)),(local_column + U32.from_nat(lane) : U32))},Packing.phase_index(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)) : ConvSpec.Product}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  tile_product_given(~upper,row,lane,call,first_y,first_x,image,channel,local_row,local_column,gradient,weights,output,tap_row,tap_column,first_y_inside,first_x_inside,position_y_inside,position_x_inside,output_inside,row_inside,column_inside,channel_inside,lane_inside,
    PhaseBounds.row_facts(call,first_y,local_row,tap_row,first_y_inside,position_y_inside,row_inside),PhaseBounds.column_lane_facts(call,first_x,local_column,lane,tap_column,first_x_inside,position_x_inside,column_inside))

def phase_flat(+geometry: Kernel.Geometry,+first_y: U32,+first_x: U32,+position_y: U32,+position_x: U32,+image: U32,+channel: Nat,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>) -> StridedOrder.Flat:
  StridedOrder.Flat{problem(geometry,models_reader(gradient,weights),U32.to_nat(image)),channel,PhaseBounds.pixel(first_y,Domain.stride_word(geometry),position_y),PhaseBounds.pixel(first_x,Domain.stride_word(geometry),position_x)}

def phase_spec_count(+call: Domain.Call,
  +first_y: U32,+first_x: U32,+position_y: U32,+position_x: U32,+image: U32,+channel: Nat,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,
  +first_y_inside: {Nat.is_lt(U32.to_nat(first_y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +first_x_inside: {Nat.is_lt(U32.to_nat(first_x),Domain.width(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat(Kernel.phase_inner(Kernel.phase(Domain.geometry(call),first_y,first_x))) == StridedOrder.flat_count(phase_flat(Domain.geometry(call),first_y,first_x,position_y,position_x,image,channel,gradient,weights)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Buffer.Model{gradient_tree,gradient_length,gradient_certificate} = gradient
  Buffer.Model{weight_tree,weight_length,weight_certificate} = weights
  +kernel_coordinate_room = PhaseBounds.kernel_coordinate_room(call)
  +stride_double_room = PhaseBounds.stride_double_room(call)
  +context = Kernel.phase(geometry,first_y,first_x)
  +flat = phase_flat(geometry,first_y,first_x,position_y,position_x,image,channel,gradient,weights)
  +capacity = Machine.coordinate(limits)
  +capacity_ok = Limits.coordinate_fits(limits)
  +positive = Domain.stride_positive(call)
  +first_y_padded = Order.transitive(Nat.add(U32.to_nat(first_y),U32.to_nat(p)),Nat.add(U32.to_nat(h),U32.to_nat(p)),U32.to_nat(capacity),Order.add_le(U32.to_nat(first_y),U32.to_nat(h),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_y),U32.to_nat(h),first_y_inside)),PhaseBounds.padded_height_room(call))
  +first_x_padded = Order.transitive(Nat.add(U32.to_nat(first_x),U32.to_nat(p)),Nat.add(U32.to_nat(w),U32.to_nat(p)),U32.to_nat(capacity),Order.add_le(U32.to_nat(first_x),U32.to_nat(w),U32.to_nat(p),Bounds.strict_le(U32.to_nat(first_x),U32.to_nat(w),first_x_inside)),PhaseBounds.padded_width_room(call))
  +rows = axis_spec_count(first_y,h,k,s,p,U32.to_nat(position_y),capacity,capacity_ok,first_y_padded,kernel_coordinate_room,stride_double_room,positive)
  +columns = axis_spec_count(first_x,w,k,s,p,U32.to_nat(position_x),capacity,capacity_ok,first_x_padded,kernel_coordinate_room,stride_double_room,positive)
  Equal.trans(Nat,U32.to_nat(Kernel.phase_inner(context)),Nat.mul(U32.to_nat(o),Packing.phase_area(context)),StridedOrder.flat_count(flat),PhaseBounds.phase_inner_exact(call,first_y,first_x),
    Equal.trans(Nat,Nat.mul(U32.to_nat(o),Packing.phase_area(context)),Nat.mul(U32.to_nat(o),Nat.mul(StridedOrder.flat_rows(flat),U32.to_nat(Packing.phase_columns(context)))),StridedOrder.flat_count(flat),
      Equal.cong(Nat,Nat,rows => Nat.mul(U32.to_nat(o),Nat.mul(rows,U32.to_nat(Packing.phase_columns(context)))),U32.to_nat(Packing.phase_rows(context)),StridedOrder.flat_rows(flat),rows),
      Equal.cong(Nat,Nat,columns => Nat.mul(U32.to_nat(o),Nat.mul(StridedOrder.flat_rows(flat),columns)),U32.to_nat(Packing.phase_columns(context)),StridedOrder.flat_columns(flat),columns)))


def WindowDecoded(+window: Spatial.Stencil,+output: Nat,+row: Nat,+column: Nat) -> Data:
  +index = U32.from_nat(RowMajor.index(output,U32.to_nat(WindowCoordinates.stencil_rows(window)),U32.to_nat(WindowCoordinates.stencil_columns(window)),row,column))
  Shape.Both<{(index / (WindowCoordinates.stencil_rows(window) * WindowCoordinates.stencil_columns(window)) : U32) == U32.from_nat(output) : U32},
    Shape.Both<{(index / WindowCoordinates.stencil_columns(window) % WindowCoordinates.stencil_rows(window) : U32) == U32.from_nat(row) : U32},
      {(index % WindowCoordinates.stencil_columns(window) : U32) == U32.from_nat(column) : U32}>>

def phase_weight_offset(+phase: Kernel.Phase,+output: Nat,+tap_row: Nat,+tap_column: Nat,decoded: WindowDecoded(PhaseBounds.phase_stencil(phase),output,tap_row,tap_column)) ->
  {Kernel.weight_offset(phase,U32.from_nat(Packing.phase_index(phase,output,tap_row,tap_column))) ==
    (U32.from_nat(output) * channels(phase_geometry(phase)) * kernel(phase_geometry(phase)) * kernel(phase_geometry(phase)) +
      (Weights.axis_tap(Packing.phase_y(phase)) - U32.from_nat(tap_row) * Domain.stride_word(phase_geometry(phase)) : U32) * kernel(phase_geometry(phase)) +
      Weights.axis_tap(Packing.phase_x(phase)) - U32.from_nat(tap_column) * Domain.stride_word(phase_geometry(phase)) : U32) : U32}:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,+k,+s,p,gh,gw},Kernel.Axis{yf,yc,+yt,+yr,ys},Kernel.Axis{xf,xc,+xt,+xr,xs},inner} = phase
  Shape.Both{channel,Shape.Both{row,column}} = decoded
  +index = U32.from_nat(Packing.phase_index(phase,output,tap_row,tap_column))
  %channel : {Kernel.weight_offset(phase,index) == (_ * c * k * k + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) : U32}
  %row : {Kernel.weight_offset(phase,index) == (index / (yr * xr) * c * k * k + (yt - _ * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) : U32}
  %column : {Kernel.weight_offset(phase,index) == (index / (yr * xr) * c * k * k + (yt - index / xr % yr * s : U32) * k + xt - _ * s : U32) : U32}
  Equal.cong(U32,U32,value => (value + (yt - index / xr % yr * s : U32) * k + xt - index % xr * s : U32),
    (index / (yr * xr) * (c * k * k : U32) : U32),(index / (yr * xr) * c * k * k : U32),WindowWeights.output_step_product((index / (yr * xr) : U32),c,k))

def mapped_scalar_address(+input: Buffer.Model<F32>,+channels: U32,+spatial: U32,+channel: U32,+base: U32) ->
  {MappedSamples.value(Buffer.tree(~F32,input),channels,spatial,channel,base) == Bool.pick(F32,U32.is_lt(channel,channels),Tree.read(F32,Buffer.tree(~F32,input),(base + channel * spatial : U32)),0.0) : F32}:
  Equal.cong(U32,F32,address => Bool.pick(F32,U32.is_lt(channel,channels),Tree.read(F32,Buffer.tree(~F32,input),address),0.0),(channel * spatial + base : U32),(base + channel * spatial : U32),Arithmetic.add_commutative((channel * spatial : U32),base))

def mapped_half_value(upper: Bool,row: Accumulator.Row,+input: Buffer.Model<F32>,+channels: U32,+spatial: U32,+channel: U32,+base: U32) ->
  {Accumulator.row(~F32,row,Rows.half(upper,MappedSamples.sample_linear(Buffer.tree(~F32,input),channels,spatial,channel,base))) == Packing.weight_value(input,Packing.WeightSample{channels,channel,base,spatial},Patches.patch_key(upper,row)) : F32}:
  match upper row:
    case False{} Accumulator.R0{}:
      %Equal.sym(U32,(channel + 0 : U32),channel,Arithmetic.add_zero(channel)) : {MappedSamples.value(Buffer.tree(~F32,input),channels,spatial,channel,base) == Bool.pick(F32,U32.is_lt(_,channels),Tree.read(F32,Buffer.tree(~F32,input),(base + _ * spatial : U32)),0.0) : F32}
      mapped_scalar_address(input,channels,spatial,channel,base)
    case False{} Accumulator.R1{}: mapped_scalar_address(input,channels,spatial,(channel + 1 : U32),base)
    case False{} Accumulator.R2{}: mapped_scalar_address(input,channels,spatial,(channel + 2 : U32),base)
    case False{} Accumulator.R3{}: mapped_scalar_address(input,channels,spatial,(channel + 3 : U32),base)
    case True{} Accumulator.R0{}: mapped_scalar_address(input,channels,spatial,(channel + 4 : U32),base)
    case True{} Accumulator.R1{}: mapped_scalar_address(input,channels,spatial,(channel + 5 : U32),base)
    case True{} Accumulator.R2{}: mapped_scalar_address(input,channels,spatial,(channel + 6 : U32),base)
    case True{} Accumulator.R3{}: mapped_scalar_address(input,channels,spatial,(channel + 7 : U32),base)

def weight_output_base(+geometry: Kernel.Geometry,+output: Nat) -> {(U32.from_nat(output) * channels(geometry) * kernel(geometry) * kernel(geometry) : U32) == U32.from_nat(Nat.mul(output,Nat.mul(U32.to_nat(channels(geometry)),Nat.mul(U32.to_nat(kernel(geometry)),U32.to_nat(kernel(geometry)))))) : U32}:
  Kernel.Geometry{+c,o,h,w,+k,s,p,gh,gw} = geometry
  +middle = Nat.mul(output,U32.to_nat(c))
  +last = Nat.mul(middle,U32.to_nat(k))
  Equal.trans(U32,(U32.from_nat(output) * c * k * k : U32),(U32.from_nat(last) * k : U32),U32.from_nat(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))),
    Equal.cong(U32,U32,prefix => (prefix * k : U32),(U32.from_nat(output) * c * k : U32),U32.from_nat(last),Equal.trans(U32,(U32.from_nat(output) * c * k : U32),(U32.from_nat(middle) * k : U32),U32.from_nat(last),Equal.cong(U32,U32,prefix => (prefix * k : U32),(U32.from_nat(output) * c : U32),U32.from_nat(middle),Encoded.scale_encoded(output,c)),Encoded.scale_encoded(middle,k))),
    Equal.trans(U32,(U32.from_nat(last) * k : U32),U32.from_nat(Nat.mul(last,U32.to_nat(k))),U32.from_nat(Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))),Encoded.scale_encoded(last,k),
      Equal.cong(Nat,U32,U32.from_nat,Nat.mul(last,U32.to_nat(k)),Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Equal.trans(Nat,Nat.mul(last,U32.to_nat(k)),Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),
        Equal.trans(Nat,Nat.mul(last,U32.to_nat(k)),Nat.mul(middle,Nat.mul(U32.to_nat(k),U32.to_nat(k))),Nat.mul(output,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),RowMajor.assoc(middle,U32.to_nat(k),U32.to_nat(k)),RowMajor.assoc(output,U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),{==}))))

def phase_old_weight_base(+phase: Kernel.Phase,+output: Nat,+tap_row: Nat,+tap_column: Nat) -> U32:
  ((0 + U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))) : U32) +
    (Weights.axis_tap(Packing.phase_y(phase)) - U32.from_nat(Nat.mul(tap_row,U32.to_nat(Domain.stride_word(phase_geometry(phase))))) : U32) * kernel(phase_geometry(phase)) +
    (Weights.axis_tap(Packing.phase_x(phase)) - U32.from_nat(Nat.mul(tap_column,U32.to_nat(Domain.stride_word(phase_geometry(phase))))) : U32) : U32)

def phase_weight_base(+call: Domain.Call,+first_y: U32,+first_x: U32,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  +decoded: WindowDecoded(PhaseBounds.phase_stencil(Kernel.phase(Domain.geometry(call),first_y,first_x)),output,tap_row,tap_column)) ->
  {Kernel.weight_offset(Kernel.phase(Domain.geometry(call),first_y,first_x),U32.from_nat(Packing.phase_index(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column))) == phase_old_weight_base(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +phase = Kernel.phase(geometry,first_y,first_x)
  +yt = Weights.axis_tap(Packing.phase_y(phase))
  +xt = Weights.axis_tap(Packing.phase_x(phase))
  +prefix = (U32.from_nat(output) * c * k * k : U32)
  +index = U32.from_nat(Packing.phase_index(phase,output,tap_row,tap_column))
  Equal.trans(U32,Kernel.weight_offset(phase,index),(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32),phase_old_weight_base(phase,output,tap_row,tap_column),phase_weight_offset(phase,output,tap_row,tap_column,decoded),
    %Equal.sym(U32,(0 + U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))) : U32),U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))),Arithmetic.add_zero_left(U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))))) :
      {(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) == (_ + (yt - U32.from_nat(Nat.mul(tap_row,U32.to_nat(s))) : U32) * k + (xt - U32.from_nat(Nat.mul(tap_column,U32.to_nat(s))) : U32) : U32) : U32}
    %Encoded.scale_encoded(tap_row,s) : {(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) == (U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))) + (yt - _ : U32) * k + (xt - U32.from_nat(Nat.mul(tap_column,U32.to_nat(s))) : U32) : U32) : U32}
    %Encoded.scale_encoded(tap_column,s) : {(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) == (U32.from_nat(Nat.mul(output,U32.to_nat(Weights.output_step(phase)))) + (yt - U32.from_nat(tap_row) * s : U32) * k + (xt - _ : U32) : U32) : U32}
    %Equal.sym(Nat,U32.to_nat(Weights.output_step(phase)),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),PhaseBounds.weight_step_exact(call,first_y,first_x)) :
      {(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) == (U32.from_nat(Nat.mul(output,_)) + (yt - U32.from_nat(tap_row) * s : U32) * k + (xt - U32.from_nat(tap_column) * s : U32) : U32) : U32}
    %weight_output_base(geometry,output) : {(prefix + (yt - U32.from_nat(tap_row) * s : U32) * k + xt - U32.from_nat(tap_column) * s : U32) == (_ + (yt - U32.from_nat(tap_row) * s : U32) * k + (xt - U32.from_nat(tap_column) * s : U32) : U32) : U32}
    Arithmetic.subtract_from_sum((prefix + (yt - U32.from_nat(tap_row) * s : U32) * k : U32),xt,(U32.from_nat(tap_column) * s : U32)))

def phase_mapped_weight(+upper: Bool,+row: Accumulator.Row,+call: Domain.Call,+first_y: U32,+first_x: U32,+channel: U32,+weights: Buffer.Model<F32>,+output: Nat,+tap_row: Nat,+tap_column: Nat,
  decoded: WindowDecoded(PhaseBounds.phase_stencil(Kernel.phase(Domain.geometry(call),first_y,first_x)),output,tap_row,tap_column)) ->
  {Accumulator.row(~F32,row,Rows.half(upper,MappedSamples.sample_linear(Buffer.tree(~F32,weights),channels(Domain.geometry(call)),(kernel(Domain.geometry(call)) * kernel(Domain.geometry(call)) : U32),channel,Kernel.weight_offset(Kernel.phase(Domain.geometry(call),first_y,first_x),U32.from_nat(Packing.phase_index(Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column)))))) ==
    Accumulator.row(~F32,row,Rows.half(upper,Weights.output_tile(weights,channel,0,Weights.axis_tap(Packing.phase_y(Kernel.phase(Domain.geometry(call),first_y,first_x))),Weights.axis_tap(Packing.phase_x(Kernel.phase(Domain.geometry(call),first_y,first_x))),Kernel.phase(Domain.geometry(call),first_y,first_x),output,tap_row,tap_column))) : F32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +phase = Kernel.phase(geometry,first_y,first_x)
  +base = phase_old_weight_base(phase,output,tap_row,tap_column)
  +sample = {Packing.WeightSample{c,channel,base,(k * k : U32)} : Packing.WeightSample}
  %Equal.sym(U32,Kernel.weight_offset(phase,U32.from_nat(Packing.phase_index(phase,output,tap_row,tap_column))),base,phase_weight_base(call,first_y,first_x,output,tap_row,tap_column,decoded)) :
    {Accumulator.row(~F32,row,Rows.half(upper,MappedSamples.sample_linear(Buffer.tree(~F32,weights),c,(k * k : U32),channel,_))) == Accumulator.row(~F32,row,Rows.half(upper,Packing.weight_tile(weights,sample))) : F32}
  Equal.trans(F32,Accumulator.row(~F32,row,Rows.half(upper,MappedSamples.sample_linear(Buffer.tree(~F32,weights),c,(k * k : U32),channel,base))),Packing.weight_value(weights,sample,Patches.patch_key(upper,row)),Accumulator.row(~F32,row,Rows.half(upper,Packing.weight_tile(weights,sample))),mapped_half_value(upper,row,weights,c,(k * k : U32),channel,base),Equal.sym(F32,Accumulator.row(~F32,row,Rows.half(upper,Packing.weight_tile(weights,sample))),Packing.weight_value(weights,sample,Patches.patch_key(upper,row)),Packing.weight_tile_half(upper,row,weights,sample)))
