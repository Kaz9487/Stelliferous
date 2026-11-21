# Logical image extents and strided coordinates shared by cell source proofs.
import Base
import ../specification.bend as ConvSpec
import ../input_gradient_specification.bend as Spec
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/padded_interval.bend as Padded
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order

import ../../lib/proofs/input_gradient_phase_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_phase_bounds.bend as PhaseBounds
import ../../lib/proofs/input_gradient_phase_taps.bend as Taps
import ../../lib/proofs/nat_division_order.bend as DivisionOrder

def shape(geometry: Kernel.Geometry) -> ConvSpec.Shape:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  ConvSpec.Shape{U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)}

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

def interval_sample(+rows: Nat,+columns: Nat,+output: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat,gradient: List<&2,F32>) -> F32:
  ConvSpec.sample_if((Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,rows))) && (Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,columns))),
    RowMajor.index(output,rows,columns,Nat.sub(y,row),Nat.sub(x,column)),gradient)

def native_sample(enabled: Bool,+gradient: List<&2,F32>,+output: Nat,+rows: U32,+columns: U32,+y: U32,+x: U32,+tap_y: U32,+tap_x: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),U32.to_nat(tap_y)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(U32.to_nat(columns),U32.to_nat(tap_x)),U32.to_nat(capacity)) == True{} : Bool},
  +valid: {U32.is_lt((y - tap_y : U32),rows) && U32.is_lt((x - tap_x : U32),columns) == enabled : Bool}) ->
  {ConvSpec.sample_if(enabled,RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),U32.to_nat((x - tap_x : U32))),gradient) ==
    ConvSpec.sample_if(enabled,RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),gradient) : F32}:
  match enabled:
    case False{}: {==}
    case True{}:
      +ry = Padded.offset(y,tap_y,rows,capacity,capacity_ok,rows_room,Boolean.left(U32.is_lt((y - tap_y : U32),rows),U32.is_lt((x - tap_x : U32),columns),valid))
      +rx = Padded.offset(x,tap_x,columns,capacity,capacity_ok,columns_room,Boolean.right(U32.is_lt((y - tap_y : U32),rows),U32.is_lt((x - tap_x : U32),columns),valid))
      %ry : {ConvSpec.at(RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),U32.to_nat((x - tap_x : U32))),gradient) ==
        ConvSpec.at(RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),_,Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),gradient) : F32}
      %rx : {ConvSpec.at(RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),U32.to_nat((x - tap_x : U32))),gradient) ==
        ConvSpec.at(RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),_),gradient) : F32}
      {==}

def native_wrapped(+gradient: List<&2,F32>,+output: Nat,+rows: U32,+columns: U32,+y: U32,+x: U32,+tap_y: U32,+tap_x: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),U32.to_nat(tap_y)),U32.to_nat(capacity)) == True{} : Bool},
  +columns_room: {Nat.is_le(Nat.add(U32.to_nat(columns),U32.to_nat(tap_x)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {ConvSpec.sample_if(U32.is_lt((y - tap_y : U32),rows) && U32.is_lt((x - tap_x : U32),columns),
      RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),U32.to_nat((x - tap_x : U32))),gradient) ==
    ConvSpec.sample_if((Nat.is_le(U32.to_nat(tap_y),U32.to_nat(y)) && Nat.is_lt(U32.to_nat(y),Nat.add(U32.to_nat(tap_y),U32.to_nat(rows)))) &&
      (Nat.is_le(U32.to_nat(tap_x),U32.to_nat(x)) && Nat.is_lt(U32.to_nat(x),Nat.add(U32.to_nat(tap_x),U32.to_nat(columns)))),
      RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x))),gradient) : F32}:
  +gy = U32.is_lt((y - tap_y : U32),rows)
  +gx = U32.is_lt((x - tap_x : U32),columns)
  +initial = ConvSpec.sample_if(gy && gx,RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat((y - tap_y : U32)),U32.to_nat((x - tap_x : U32))),gradient)
  +index = RowMajor.index(output,U32.to_nat(rows),U32.to_nat(columns),Nat.sub(U32.to_nat(y),U32.to_nat(tap_y)),Nat.sub(U32.to_nat(x),U32.to_nat(tap_x)))
  %Padded.range(y,tap_y,rows,capacity,capacity_ok,rows_room) :
    {initial == ConvSpec.sample_if(_ && (Nat.is_le(U32.to_nat(tap_x),U32.to_nat(x)) && Nat.is_lt(U32.to_nat(x),Nat.add(U32.to_nat(tap_x),U32.to_nat(columns)))),index,gradient) : F32}
  %Padded.range(x,tap_x,columns,capacity,capacity_ok,columns_room) : {initial == ConvSpec.sample_if(gy && _,index,gradient) : F32}
  native_sample(gy && gx,gradient,output,rows,columns,y,x,tap_y,tap_x,capacity,capacity_ok,rows_room,columns_room,{==})

def padded_sample(+problem: Spec.Problem,+output: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat) -> F32:
  +s = Spec.stride(problem)
  +rows = Spec.out_height(problem)
  +columns = Spec.out_width(problem)
  ConvSpec.sample_if(Nat.is_le(row,y) && Nat.is_lt(y,Nat.add(row,Nat.mul(s,rows))) && Nat.is_le(column,x) && Nat.is_lt(x,Nat.add(column,Nat.mul(s,columns))),
    RowMajor.index(output,rows,columns,Nat.div(Nat.sub(y,row),s),Nat.div(Nat.sub(x,column),s)),Spec.output_gradient(problem))

def radix_problem(+problem: Spec.Problem,+output: Nat,+y: Nat,+x: Nat,+row: Nat,+column: Nat,+yq: Nat,+xq: Nat,+rq: Nat,+cq: Nat,+ry: Nat,+rx: Nat,
  +positive: {Nat.is_lt(0n,Spec.stride(problem)) == True{} : Bool},
  +y_exact: {Nat.add(y,Spec.padding(problem)) == Nat.add(Nat.mul(yq,Spec.stride(problem)),ry) : Nat},
  +x_exact: {Nat.add(x,Spec.padding(problem)) == Nat.add(Nat.mul(xq,Spec.stride(problem)),rx) : Nat},
  +row_exact: {row == Nat.add(Nat.mul(rq,Spec.stride(problem)),ry) : Nat},
  +column_exact: {column == Nat.add(Nat.mul(cq,Spec.stride(problem)),rx) : Nat}) ->
  {Spec.sample(problem,output,y,x,row,column) ==
    ConvSpec.sample_if((Nat.is_le(rq,yq) && Nat.is_lt(yq,Nat.add(rq,Spec.out_height(problem)))) &&
      (Nat.is_le(cq,xq) && Nat.is_lt(xq,Nat.add(cq,Spec.out_width(problem)))),
      RowMajor.index(output,Spec.out_height(problem),Spec.out_width(problem),Nat.sub(yq,rq),Nat.sub(xq,cq)),Spec.output_gradient(problem)) : F32}:
  +s = Spec.stride(problem)
  +wanted = ConvSpec.sample_if((Nat.is_le(rq,yq) && Nat.is_lt(yq,Nat.add(rq,Spec.out_height(problem)))) &&
    (Nat.is_le(cq,xq) && Nat.is_lt(xq,Nat.add(cq,Spec.out_width(problem)))),
    RowMajor.index(output,Spec.out_height(problem),Spec.out_width(problem),Nat.sub(yq,rq),Nat.sub(xq,cq)),Spec.output_gradient(problem))
  %Equal.sym(Nat,Nat.add(y,Spec.padding(problem)),Nat.add(Nat.mul(yq,s),ry),y_exact) :
    {padded_sample(problem,output,_,Nat.add(x,Spec.padding(problem)),row,column) == wanted : F32}
  %Equal.sym(Nat,Nat.add(x,Spec.padding(problem)),Nat.add(Nat.mul(xq,s),rx),x_exact) :
    {padded_sample(problem,output,Nat.add(Nat.mul(yq,s),ry),_,row,column) == wanted : F32}
  %Equal.sym(Nat,row,Nat.add(Nat.mul(rq,s),ry),row_exact) :
    {padded_sample(problem,output,Nat.add(Nat.mul(yq,s),ry),Nat.add(Nat.mul(xq,s),rx),_,column) == wanted : F32}
  %Equal.sym(Nat,column,Nat.add(Nat.mul(cq,s),rx),column_exact) :
    {padded_sample(problem,output,Nat.add(Nat.mul(yq,s),ry),Nat.add(Nat.mul(xq,s),rx),Nat.add(Nat.mul(rq,s),ry),_) == wanted : F32}
  radix_sample(Spec.out_height(problem),Spec.out_width(problem),s,yq,xq,rq,cq,ry,rx,output,Spec.output_gradient(problem),positive)

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
