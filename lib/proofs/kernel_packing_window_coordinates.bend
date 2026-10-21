# Native stencil counters denote the canonical packed entry and its eight pixels.
import Base
import ../kernel_spatial_packing.bend as Packing
import ./binary_radix.bend as Radix
import ./encoded_coordinates.bend as Encoded
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as WordDivision
import ./u32_comparison.bend as Comparison
import ./word_arithmetic.bend as Words
import ../kernel_packing_samples.bend as NativeSamples
import ../kernel_tile.bend as Tile
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree

def stencil_rows(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  rows

def stencil_columns(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  columns

def stencil_inner(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  inner

def stencil_width(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  output_width

def stencil_base(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  base

def stencil_height(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  height

def stencil_source_width(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  width

def stencil_stride(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  stride

def stencil_row_origin(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  row_origin

def stencil_column_origin(window: Packing.Stencil) -> U32:
  Packing.Stencil{base,height,width,rows,columns,stride,row_origin,column_origin,output_width,inner} = window
  column_origin

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

def channels(context: Geometry) -> Nat:
  Geometry{window,channels,capacity,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  channels

def capacity(context: Geometry) -> U32:
  Geometry{window,channels,capacity,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  capacity

def row_word(context: Geometry) -> U32:
  stencil_rows(window(context))

def column_word(context: Geometry) -> U32:
  stencil_columns(window(context))

def inner_word(context: Geometry) -> U32:
  stencil_inner(window(context))

def width_word(context: Geometry) -> U32:
  stencil_width(window(context))

def rows(context: Geometry) -> Nat:
  U32.to_nat(row_word(context))

def columns(context: Geometry) -> Nat:
  U32.to_nat(column_word(context))

def inner(context: Geometry) -> Nat:
  U32.to_nat(inner_word(context))

def width(context: Geometry) -> Nat:
  U32.to_nat(width_word(context))

type Entry<-context: Geometry,-index: U32,-group: Nat,-reduction: Nat> is Data:
  Entry{group: {(index / inner_word(context) : U32) == U32.from_nat(group) : U32},reduction: {(index % inner_word(context) : U32) == U32.from_nat(reduction) : U32}}

def entry(+context: Geometry,+index: U32,+group: Nat,+reduction: Nat,
  +located: {U32.to_nat(index) == Nat.add(Nat.mul(group,inner(context)),reduction) : Nat},
  +inside: {Nat.is_lt(reduction,inner(context)) == True{} : Bool}) -> Entry<context,index,group,reduction>:
  Geometry{window,channels,+cap,+word_ok,rp,cp,+ip,wp,ie,ce,rr,cr,ar,+ir,wr} = context
  Entry{
    Encoded.encoding((index / inner_word(context) : U32),group,Radix.quotient_word(index,inner_word(context),group,reduction,inner(context),cap,word_ok,located,{==},ip,inside,ir)),
    Encoded.encoding((index % inner_word(context) : U32),reduction,Radix.remainder_word(index,inner_word(context),group,reduction,inner(context),cap,word_ok,located,{==},ip,inside,ir))}

def reduction(+context: Geometry,c: Nat,r: Nat,t: Nat) -> Nat:
  RowMajor.index(c,rows(context),columns(context),r,t)

type Decoded<-context: Geometry,-c: Nat,-r: Nat,-t: Nat> is Data:
  Decoded{channel: {(U32.from_nat(reduction(context,c,r,t)) / (row_word(context) * column_word(context)) : U32) == U32.from_nat(c) : U32},
    row: {(U32.from_nat(reduction(context,c,r,t)) / column_word(context) % row_word(context) : U32) == U32.from_nat(r) : U32},
    column: {(U32.from_nat(reduction(context,c,r,t)) % column_word(context) : U32) == U32.from_nat(t) : U32}}

def decode(+context: Geometry,+c: Nat,+r: Nat,+t: Nat,
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool}) -> Decoded<context,c,r,t>:
  Geometry{window,channels,+cap,+word_ok,+rp,+cp,ip,wp,+ie,ce,+rr,+cr,+ar,+ir,wr} = context
  +nr = rows(context)
  +nc = columns(context)
  +area = Nat.mul(nr,nc)
  +q = U32.from_nat(reduction(context,c,r,t))
  +q_exact = Bounds.roundtrip(reduction(context,c,r,t),cap,word_ok,Bounds.strict_le(reduction(context,c,r,t),U32.to_nat(cap),
      Bounds.nat_trans(reduction(context,c,r,t),inner(context),U32.to_nat(cap),
        %Equal.sym(Nat,inner(context),Nat.mul(channels,Nat.mul(nr,nc)),ie) : {Nat.is_lt(reduction(context,c,r,t),_) == True{} : Bool} RowMajor.in_bounds(c,channels,nr,nc,r,t,c_inside,r_inside,t_inside),
        Order.transitive(inner(context),Nat.double(inner(context)),U32.to_nat(cap),Order.le_double(inner(context)),ir))))
  +area_exact = Arithmetic.multiply(row_word(context),column_word(context),cap,word_ok,Order.transitive(area,Nat.double(area),U32.to_nat(cap),Order.le_double(area),ar))
  +area_positive = Order.product_positive(nr,nc,rp,cp)
  +pixel_inside = RowMajor.radix(r,nr,t,nc,r_inside,t_inside)
  +head = Nat.add(Nat.mul(c,nr),r)
  +mixed = Equal.trans(Nat,U32.to_nat(q),reduction(context,c,r,t),Nat.add(Nat.mul(head,nc),t),q_exact,Equal.sym(Nat,Nat.add(Nat.mul(head,nc),t),reduction(context,c,r,t),RowMajor.mixed(c,nr,nc,r,t)))
  +quotient_exact = Radix.quotient_word(q,column_word(context),head,t,nc,cap,word_ok,mixed,{==},cp,t_inside,cr)
  Decoded{
    Encoded.encoding((q / (row_word(context) * column_word(context)) : U32),c,Radix.quotient_word(q,(row_word(context) * column_word(context) : U32),c,Nat.add(Nat.mul(r,nc),t),area,cap,word_ok,q_exact,area_exact,area_positive,pixel_inside,ar)),
    Encoded.encoding((q / column_word(context) % row_word(context) : U32),r,Radix.remainder_word((q / column_word(context) : U32),row_word(context),c,r,nr,cap,word_ok,quotient_exact,{==},rp,r_inside,rr)),
    Encoded.encoding((q % column_word(context) : U32),t,Radix.remainder_word(q,column_word(context),head,t,nc,cap,word_ok,mixed,{==},cp,t_inside,cr))}

def column_guard(+context: Geometry,+position: U32,+limit: U32,+limit_nat: Nat,
  +limit_exact: {U32.to_nat(limit) == limit_nat : Nat},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  selected: {U32.is_le((position % width_word(context) + 8 : U32),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),limit_nat) == True{} : Bool}:
  Geometry{window,channels,+cap,+word_ok,rp,cp,ip,+wp,ie,ce,rr,cr,ar,ir,+wr} = context
  +w = width(context)
  +p = U32.to_nat(position)
  +remainder = (position % width_word(context) : U32)
  +remainder_exact = WordDivision.modulo(position,width_word(context),cap,word_ok,wr,wp)
  +remainder_le = {
    %Equal.sym(Nat,p,Nat.add(Nat.mul(Nat.div(p,w),w),Nat.mod(p,w)),DivisionOrder.quotient_exact(p,w,wp)) : {Nat.is_le(Nat.mod(p,w),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(Nat.div(p,w),w),Nat.mod(p,w)),Nat.add(Nat.mod(p,w),Nat.mul(Nat.div(p,w),w)),Order.commutative(Nat.mul(Nat.div(p,w),w),Nat.mod(p,w))) : {Nat.is_le(Nat.mod(p,w),_) == True{} : Bool}
    Order.le_add(Nat.mod(p,w),Nat.mul(Nat.div(p,w),w)) : {Nat.is_le(Nat.mod(p,w),p) == True{} : Bool}}
  +sum_exact = Arithmetic.add(remainder,8,cap,word_ok,
    %Equal.sym(Nat,U32.to_nat(remainder),Nat.mod(p,w),remainder_exact) : {Nat.is_le(Nat.add(_,8n),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mod(p,w),8n),Nat.add(p,8n),U32.to_nat(cap),Order.add_le(Nat.mod(p,w),p,8n,remainder_le),position_room))
  %limit_exact : {Nat.is_le(Nat.add(Nat.mod(p,w),8n),_) == True{} : Bool}
  %remainder_exact : {Nat.is_le(Nat.add(_,8n),U32.to_nat(limit)) == True{} : Bool}
  %sum_exact : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  %Comparison.le((remainder + 8 : U32),limit) : {_ == True{} : Bool}
  selected

def within_columns(+context: Geometry,+position: U32,+position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  selected: {U32.is_le((position % width_word(context) + 8 : U32),width_word(context)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),width(context)) == True{} : Bool}:
  column_guard(context,position,width_word(context),width(context),{==},position_room,selected)

def wrap_columns(+context: Geometry,+position: U32,+position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  selected: {U32.is_le((position % width_word(context) + 8 : U32),(width_word(context) + width_word(context) : U32)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool}:
  Geometry{window,channels,+cap,+word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  column_guard(context,position,(width_word(context) + width_word(context) : U32),Nat.add(width(context),width(context)),
    Arithmetic.add(width_word(context),width_word(context),cap,word_ok,
      %Equal.sym(Nat,Nat.add(width(context),width(context)),Nat.double(width(context)),Addition.double(width(context))) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} wr),position_room,selected)

type LaneNumbers<-context: Geometry,-position: U32,-lane: Nat> is Data:
  LaneNumbers{row: {U32.to_nat((position / width_word(context) : U32)) == Nat.div(U32.to_nat(position),width(context)) : Nat},
    column: {U32.to_nat((position % width_word(context) : U32)) == Nat.mod(U32.to_nat(position),width(context)) : Nat},
    shifted: {U32.to_nat((position + U32.from_nat(lane) : U32)) == Nat.add(U32.to_nat(position),lane) : Nat},
    sum: {U32.to_nat((position % width_word(context) + U32.from_nat(lane) : U32)) == Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane) : Nat},
    located: {U32.to_nat((position + U32.from_nat(lane) : U32)) == Nat.add(Nat.mul(Nat.div(U32.to_nat(position),width(context)),width(context)),Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane)) : Nat}}

def lane_numbers(+context: Geometry,+position: U32,+lane: Nat,
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) -> LaneNumbers<context,position,lane>:
  Geometry{window,channels,+cap,+word_ok,rp,cp,ip,+wp,ie,ce,rr,cr,ar,ir,+wr} = context
  +p = U32.to_nat(position)
  +w = width(context)
  +r = Nat.div(p,w)
  +c = Nat.mod(p,w)
  +column = WordDivision.modulo(position,width_word(context),cap,word_ok,wr,wp)
  +row = WordDivision.divide(position,width_word(context),cap,word_ok,wr,wp)
  +lane_le = Bounds.strict_le(lane,8n,lane_inside)
  +eight_room = {%Order.commutative(8n,p) : {Nat.is_le(8n,_) == True{} : Bool} Order.le_add(8n,p) : {Nat.is_le(8n,Nat.add(p,8n)) == True{} : Bool}}
  +lane_exact = Bounds.roundtrip(lane,cap,word_ok,Order.transitive(lane,8n,U32.to_nat(cap),lane_le,Order.transitive(8n,Nat.add(p,8n),U32.to_nat(cap),eight_room,position_room)))
  +shifted = Arithmetic.add(position,U32.from_nat(lane),cap,word_ok,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_le(Nat.add(p,_),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(p,lane),Nat.add(p,8n),U32.to_nat(cap),Order.offset_le(p,lane,8n,lane_le),position_room))
  +shifted_exact = {%lane_exact : {U32.to_nat((position + U32.from_nat(lane) : U32)) == Nat.add(p,_) : Nat} shifted : {U32.to_nat((position + U32.from_nat(lane) : U32)) == Nat.add(p,lane) : Nat}}
  +c_le = {
    %Equal.sym(Nat,p,Nat.add(Nat.mul(r,w),c),DivisionOrder.quotient_exact(p,w,wp)) : {Nat.is_le(c,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(r,w),c),Nat.add(c,Nat.mul(r,w)),Order.commutative(Nat.mul(r,w),c)) : {Nat.is_le(c,_) == True{} : Bool}
    Order.le_add(c,Nat.mul(r,w)) : {Nat.is_le(c,p) == True{} : Bool}}
  +sum = Arithmetic.add((position % width_word(context) : U32),U32.from_nat(lane),cap,word_ok,
    %Equal.sym(Nat,U32.to_nat((position % width_word(context) : U32)),c,column) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(lane))),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_le(Nat.add(c,_),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(c,lane),Nat.add(p,lane),U32.to_nat(cap),Order.add_le(c,p,lane,c_le),Order.transitive(Nat.add(p,lane),Nat.add(p,8n),U32.to_nat(cap),Order.offset_le(p,lane,8n,lane_le),position_room)))
  +sum_exact = {%column : {U32.to_nat((position % width_word(context) + U32.from_nat(lane) : U32)) == Nat.add(_,lane) : Nat}
    %lane_exact : {U32.to_nat((position % width_word(context) + U32.from_nat(lane) : U32)) == Nat.add(U32.to_nat((position % width_word(context) : U32)),_) : Nat}
    sum : {U32.to_nat((position % width_word(context) + U32.from_nat(lane) : U32)) == Nat.add(c,lane) : Nat}}
  LaneNumbers{row,column,shifted_exact,sum_exact,
    Equal.trans(Nat,U32.to_nat((position + U32.from_nat(lane) : U32)),Nat.add(p,lane),Nat.add(Nat.mul(r,w),Nat.add(c,lane)),shifted_exact,
      %Equal.sym(Nat,p,Nat.add(Nat.mul(r,w),c),DivisionOrder.quotient_exact(p,w,wp)) : {Nat.add(_,lane) == Nat.add(Nat.mul(r,w),Nat.add(c,lane)) : Nat}
      Addition.associative(Nat.mul(r,w),c,lane))}

type SameLane<-context: Geometry,-position: U32,-lane: Nat> is Data:
  SameLane{row: {((position + U32.from_nat(lane) : U32) / width_word(context) : U32) == (position / width_word(context) : U32) : U32},
    column: {((position + U32.from_nat(lane) : U32) % width_word(context) : U32) == (position % width_word(context) + U32.from_nat(lane) : U32) : U32}}

def same_lane_given(+context: Geometry,+position: U32,+lane: Nat,
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+within: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == True{} : Bool},
  numbers: LaneNumbers<context,position,lane>) -> SameLane<context,position,lane>:
  Geometry{window,channels,+cap,+word_ok,rp,cp,ip,+wp,ie,ce,rr,cr,ar,ir,+wr} = context
  LaneNumbers{+row,+column,shifted,+sum,+located} = numbers
  +p = (position + U32.from_nat(lane) : U32)
  +r = Nat.div(U32.to_nat(position),width(context))
  +c = Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane)
  SameLane{
    Equal.trans(U32,(p / width_word(context) : U32),U32.from_nat(r),(position / width_word(context) : U32),
      Encoded.encoding((p / width_word(context) : U32),r,Radix.quotient_word(p,width_word(context),r,c,width(context),cap,word_ok,located,{==},wp,within,wr)),
      Equal.sym(U32,(position / width_word(context) : U32),U32.from_nat(r),Encoded.encoding((position / width_word(context) : U32),r,row))),
    Equal.trans(U32,(p % width_word(context) : U32),U32.from_nat(c),(position % width_word(context) + U32.from_nat(lane) : U32),
      Encoded.encoding((p % width_word(context) : U32),c,Radix.remainder_word(p,width_word(context),r,c,width(context),cap,word_ok,located,{==},wp,within,wr)),
      Equal.sym(U32,(position % width_word(context) + U32.from_nat(lane) : U32),U32.from_nat(c),Encoded.encoding((position % width_word(context) + U32.from_nat(lane) : U32),c,sum)))}

def same_lane(+context: Geometry,+position: U32,+lane: Nat,
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+within: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == True{} : Bool}) -> SameLane<context,position,lane>:
  same_lane_given(context,position,lane,position_room,lane_inside,within,lane_numbers(context,position,lane,position_room,lane_inside))

def lane_column_offset(+column: U32,+lane: Nat,+stride: U32,+tap: U32,+origin: U32) ->
  {((column * stride + tap + origin : U32) + U32.from_nat(lane) * stride : U32) == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}:
  +c = U32.to_nat(column)
  +s = U32.to_nat(stride)
  +a = (column * stride : U32)
  +b = (U32.from_nat(lane) * stride : U32)
  +rest = (tap + origin : U32)
  +product = {
    %Arithmetic.native_roundtrip(column) : {(_ * stride + b : U32) == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(c) * stride : U32),U32.from_nat(Nat.mul(c,s)),Encoded.scale_encoded(c,stride)) : {(_ + b : U32) == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %Equal.sym(U32,b,U32.from_nat(Nat.mul(lane,s)),Encoded.scale_encoded(lane,stride)) : {(U32.from_nat(Nat.mul(c,s)) + _ : U32) == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(Nat.mul(c,s)) + U32.from_nat(Nat.mul(lane,s)) : U32),U32.from_nat(Nat.add(Nat.mul(c,s),Nat.mul(lane,s))),Words.native_add(Nat.mul(c,s),Nat.mul(lane,s))) : {_ == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %RowMajor.distrib(c,lane,s) : {U32.from_nat(_) == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %Arithmetic.native_roundtrip(column) : {U32.from_nat(Nat.mul(Nat.add(c,lane),s)) == ((_ + U32.from_nat(lane) : U32) * stride : U32) : U32}
    %Equal.sym(U32,(U32.from_nat(c) + U32.from_nat(lane) : U32),U32.from_nat(Nat.add(c,lane)),Words.native_add(c,lane)) : {U32.from_nat(Nat.mul(Nat.add(c,lane),s)) == (_ * stride : U32) : U32}
    Equal.sym(U32,(U32.from_nat(Nat.add(c,lane)) * stride : U32),U32.from_nat(Nat.mul(Nat.add(c,lane),s)),Encoded.scale_encoded(Nat.add(c,lane),stride)) : {(a + b : U32) == ((column + U32.from_nat(lane) : U32) * stride : U32) : U32}}
  %Equal.sym(U32,(a + tap + origin : U32),(a + rest : U32),Arithmetic.add_associative(a,tap,origin)) : {(_ + b : U32) == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}
  %Equal.sym(U32,((a + rest : U32) + b : U32),(a + (rest + b : U32) : U32),Arithmetic.add_associative(a,rest,b)) : {_ == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}
  %Equal.sym(U32,(rest + b : U32),(b + rest : U32),Arithmetic.add_commutative(rest,b)) : {(a + _ : U32) == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}
  %Arithmetic.add_associative(a,b,rest) : {_ == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}
  %Arithmetic.add_associative((a + b : U32),tap,origin) : {_ == ((column + U32.from_nat(lane) : U32) * stride + tap + origin : U32) : U32}
  Equal.cong(U32,U32,x => (x + tap + origin : U32),(a + b : U32),((column + U32.from_nat(lane) : U32) * stride : U32),product)

def decoded_context(+context: Geometry,+position: U32,+c: Nat,+r: Nat,+t: Nat,facts: Decoded<context,c,r,t>) ->
  {Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))) ==
    NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),
      (stencil_base(window(context)) + U32.from_nat(c) * stencil_height(window(context)) * stencil_source_width(window(context)) : U32),
      stencil_height(window(context)),stencil_source_width(window(context)),stencil_stride(window(context)),stencil_row_origin(window(context)),stencil_column_origin(window(context)),width_word(context)} : NativeSamples.WindowPixelBlock}:
  Geometry{stencil,channels,cap,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  Packing.Stencil{+base,+h,+w,+tr,+tc,+s,+oy,+ox,+ow,inner} = stencil
  Decoded{channel,row,column} = facts
  +q = U32.from_nat(reduction(context,c,r,t))
  %Equal.sym(U32,(q / tc % tr : U32),U32.from_nat(r),row) : {NativeSamples.WindowPixelBlock{position,_,(q % tc : U32),(base + q / (tr * tc) * h * w : U32),h,w,s,oy,ox,ow} == NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),(base + U32.from_nat(c) * h * w : U32),h,w,s,oy,ox,ow} : NativeSamples.WindowPixelBlock}
  %Equal.sym(U32,(q % tc : U32),U32.from_nat(t),column) : {NativeSamples.WindowPixelBlock{position,U32.from_nat(r),_,(base + q / (tr * tc) * h * w : U32),h,w,s,oy,ox,ow} == NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),(base + U32.from_nat(c) * h * w : U32),h,w,s,oy,ox,ow} : NativeSamples.WindowPixelBlock}
  %Equal.sym(U32,(q / (tr * tc) : U32),U32.from_nat(c),channel) : {NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),(base + _ * h * w : U32),h,w,s,oy,ox,ow} == NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),(base + U32.from_nat(c) * h * w : U32),h,w,s,oy,ox,ow} : NativeSamples.WindowPixelBlock}
  {==}

def block_context(+context: Geometry,+position: U32,+c: Nat,+r: Nat,+t: Nat) -> Packing.PixelBlock:
  +stencil = window(context)
  +s = stencil_stride(stencil)
  Packing.PixelBlock{(position / width_word(context) * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % width_word(context) * s + U32.from_nat(t) + stencil_column_origin(stencil) : U32),
    (stencil_base(stencil) + U32.from_nat(c) * stencil_height(stencil) * stencil_source_width(stencil) : U32),stencil_height(stencil),stencil_source_width(stencil),s}

def row_vector(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat) -> Tile.Tile8:
  +stencil = window(context)
  +s = stencil_stride(stencil)
  +ow = width_word(context)
  Model.row_lanes(input,(position / ow * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow * s + U32.from_nat(t) + stencil_column_origin(stencil) : U32),
    (stencil_base(stencil) + U32.from_nat(c) * stencil_height(stencil) * stencil_source_width(stencil) : U32),
    stencil_height(stencil),stencil_source_width(stencil),s,
    ((position / ow + 1) * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow : U32),ow,U32.from_nat(t),(0 - stencil_column_origin(stencil) : U32))

def row_lane_given(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  decoded: Decoded<context,c,r,t>,location: SameLane<context,position,lane>) ->
  {Samples.block_sample(input,block_context(context,position,c,r,t),lane) ==
    Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  Geometry{stencil,channels,cap,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  Packing.Stencil{+base,+h,+w,tr,tc,+s,+oy,+ox,+ow,inner} = stencil
  SameLane{row,column} = location
  +iy = (position / ow * s + U32.from_nat(r) + oy : U32)
  +ix = (position % ow * s + U32.from_nat(t) + ox : U32)
  +cb = (base + U32.from_nat(c) * h * w : U32)
  +canonical = Samples.window_context(stencil,position,U32.from_nat(reduction(context,c,r,t)))
  +decoded_context_value = {NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),cb,h,w,s,oy,ox,ow} : NativeSamples.WindowPixelBlock}
  %Equal.sym(NativeSamples.WindowPixelBlock,canonical,decoded_context_value,decoded_context(context,position,c,r,t,decoded)) :
    {Samples.block_sample(input,block_context(context,position,c,r,t),lane) == Samples.window_sample(input,_,lane) : F32}
  %Equal.sym(U32,((position + U32.from_nat(lane) : U32) / ow : U32),(position / ow : U32),row) :
    {Samples.value(input,iy,(ix + U32.from_nat(lane) * s : U32),cb,h,w) == Samples.value(input,(_ * s + U32.from_nat(r) + oy : U32),((position + U32.from_nat(lane) : U32) % ow * s + U32.from_nat(t) + ox : U32),cb,h,w) : F32}
  %Equal.sym(U32,((position + U32.from_nat(lane) : U32) % ow : U32),(position % ow + U32.from_nat(lane) : U32),column) :
    {Samples.value(input,iy,(ix + U32.from_nat(lane) * s : U32),cb,h,w) == Samples.value(input,iy,(_ * s + U32.from_nat(t) + ox : U32),cb,h,w) : F32}
  Equal.cong(U32,F32,x => Samples.value(input,iy,x,cb,h,w),(ix + U32.from_nat(lane) * s : U32),((position % ow + U32.from_nat(lane) : U32) * s + U32.from_nat(t) + ox : U32),
    lane_column_offset((position % ow : U32),lane,s,U32.from_nat(t),ox))

def row_lane(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),width(context)) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Samples.block_sample(input,block_context(context,position,c,r,t),lane) == Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  row_lane_given(context,input,position,c,r,t,lane,decode(context,c,r,t,c_inside,r_inside,t_inside),
    same_lane(context,position,lane,position_room,lane_inside,
      Bounds.nat_trans(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),width(context),
        Order.offset_lt(Nat.mod(U32.to_nat(position),width(context)),lane,8n,lane_inside),within)))

def row_entry(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),width(context)) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool}) ->
  {row_vector(context,input,position,c,r,t) == Samples.window_vector(input,window(context),position,U32.from_nat(reduction(context,c,r,t))) : Tile.Tile8}:
  TileLaws.tiles_equal(row_vector(context,input,position,c,r,t),Samples.window_vector(input,window(context),position,U32.from_nat(reduction(context,c,r,t))),
    row_lane(context,input,position,c,r,t,0n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,1n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,2n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,3n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,4n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,5n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,6n,within,position_room,c_inside,r_inside,t_inside,{==}),
    row_lane(context,input,position,c,r,t,7n,within,position_room,c_inside,r_inside,t_inside,{==}))

type NextLane<-context: Geometry,-position: U32,-lane: Nat> is Data:
  NextLane{row: {((position + U32.from_nat(lane) : U32) / width_word(context) : U32) == (position / width_word(context) + 1 : U32) : U32},
    column: {((position + U32.from_nat(lane) : U32) % width_word(context) : U32) == (position % width_word(context) + U32.from_nat(lane) - width_word(context) : U32) : U32}}

def next_lane_given(+context: Geometry,+position: U32,+lane: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +different: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == False{} : Bool},
  numbers: LaneNumbers<context,position,lane>) -> NextLane<context,position,lane>:
  Geometry{window,channels,+cap,+word_ok,rp,cp,ip,+wp,ie,ce,rr,cr,ar,ir,+wr} = context
  LaneNumbers{+row,column,shifted,+sum,+located} = numbers
  +w = width(context)
  +r = Nat.div(U32.to_nat(position),w)
  +c = Nat.mod(U32.to_nat(position),w)
  +value = Nat.add(c,lane)
  +m = Nat.sub(value,w)
  +lower = Order.not_less(value,w,different)
  +value_inside = Bounds.nat_trans(value,Nat.add(c,8n),Nat.add(w,w),Order.offset_lt(c,lane,8n,lane_inside),within)
  +m_inside = Order.sub_lt(value,w,w,lower,value_inside)
  +split = Order.restore_shift(w,value,lower)
  +next_located = Equal.trans(Nat,U32.to_nat((position + U32.from_nat(lane) : U32)),Nat.add(Nat.mul(r,w),value),Nat.add(Nat.mul(1n+r,w),m),located,
    %split : {Nat.add(Nat.mul(r,w),_) == Nat.add(Nat.mul(1n+r,w),m) : Nat}
    Equal.trans(Nat,Nat.add(Nat.mul(r,w),Nat.add(w,m)),Nat.add(Nat.add(Nat.mul(r,w),w),m),Nat.add(Nat.add(w,Nat.mul(r,w)),m),
      Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(r,w),w),m),Nat.add(Nat.mul(r,w),Nat.add(w,m)),Addition.associative(Nat.mul(r,w),w,m)),
      Equal.cong(Nat,Nat,x => Nat.add(x,m),Nat.add(Nat.mul(r,w),w),Nat.add(w,Nat.mul(r,w)),Order.commutative(Nat.mul(r,w),w))))
  +sum_word = (position % width_word(context) + U32.from_nat(lane) : U32)
  +m_room = Order.transitive(m,w,U32.to_nat(cap),Bounds.strict_le(m,w,m_inside),Order.transitive(w,Nat.double(w),U32.to_nat(cap),Order.le_double(w),wr))
  +subtracted = Arithmetic.subtract(sum_word,width_word(context),cap,word_ok,
    %Equal.sym(Nat,U32.to_nat(sum_word),value,sum) : {Nat.is_le(w,_) == True{} : Bool} lower,
    %Equal.sym(Nat,U32.to_nat(sum_word),value,sum) : {Nat.is_le(Nat.sub(_,w),U32.to_nat(cap)) == True{} : Bool} m_room)
  +subtracted_exact = {%sum : {U32.to_nat((sum_word - width_word(context) : U32)) == Nat.sub(_,w) : Nat} subtracted : {U32.to_nat((sum_word - width_word(context) : U32)) == m : Nat}}
  +next_row_encoded = {
    %Equal.sym(U32,(position / width_word(context) : U32),U32.from_nat(r),Encoded.encoding((position / width_word(context) : U32),r,row)) : {(_ + 1 : U32) == U32.from_nat(1n+r) : U32}
    Bounds.increment_encoded(r) : {(position / width_word(context) + 1 : U32) == U32.from_nat(1n+r) : U32}}
  +p = (position + U32.from_nat(lane) : U32)
  NextLane{
    Equal.trans(U32,(p / width_word(context) : U32),U32.from_nat(1n+r),(position / width_word(context) + 1 : U32),
      Encoded.encoding((p / width_word(context) : U32),1n+r,Radix.quotient_word(p,width_word(context),1n+r,m,w,cap,word_ok,next_located,{==},wp,m_inside,wr)),
      Equal.sym(U32,(position / width_word(context) + 1 : U32),U32.from_nat(1n+r),next_row_encoded)),
    Equal.trans(U32,(p % width_word(context) : U32),U32.from_nat(m),(sum_word - width_word(context) : U32),
      Encoded.encoding((p % width_word(context) : U32),m,Radix.remainder_word(p,width_word(context),1n+r,m,w,cap,word_ok,next_located,{==},wp,m_inside,wr)),
      Equal.sym(U32,(sum_word - width_word(context) : U32),U32.from_nat(m),Encoded.encoding((sum_word - width_word(context) : U32),m,subtracted_exact)))}

def next_lane(+context: Geometry,+position: U32,+lane: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +different: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == False{} : Bool}) -> NextLane<context,position,lane>:
  next_lane_given(context,position,lane,within,position_room,lane_inside,different,lane_numbers(context,position,lane,position_room,lane_inside))

def lane_guard_given(+context: Geometry,+position: U32,+lane: Nat,+same: Bool,
  decided: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == same : Bool},
  numbers: LaneNumbers<context,position,lane>) ->
  {U32.is_lt((position % width_word(context) + U32.from_nat(lane) : U32),width_word(context)) == same : Bool}:
  LaneNumbers{row,column,shifted,sum,located} = numbers
  +native = (position % width_word(context) + U32.from_nat(lane) : U32)
  %Equal.sym(Bool,U32.is_lt(native,width_word(context)),Nat.is_lt(U32.to_nat(native),width(context)),Comparison.lt(native,width_word(context))) : {_ == same : Bool}
  %Equal.sym(Nat,U32.to_nat(native),Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),sum) : {Nat.is_lt(_,width(context)) == same : Bool}
  decided

def lane_guard(+context: Geometry,+position: U32,+lane: Nat,+same: Bool,
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  decided: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == same : Bool}) ->
  {U32.is_lt((position % width_word(context) + U32.from_nat(lane) : U32),width_word(context)) == same : Bool}:
  lane_guard_given(context,position,lane,same,decided,lane_numbers(context,position,lane,position_room,lane_inside))

def wrap_context(+context: Geometry,+position: U32,+c: Nat,+r: Nat,+t: Nat) -> Packing.WrapBlock:
  +stencil = window(context)
  +s = stencil_stride(stencil)
  +ow = width_word(context)
  Packing.WrapBlock{(position / ow * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow * s + U32.from_nat(t) + stencil_column_origin(stencil) : U32),
    (stencil_base(stencil) + U32.from_nat(c) * stencil_height(stencil) * stencil_source_width(stencil) : U32),
    stencil_height(stencil),stencil_source_width(stencil),s,
    ((position / ow + 1) * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow : U32),ow,U32.from_nat(t),(0 - stencil_column_origin(stencil) : U32)}

def wrap_vector(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat) -> Tile.Tile8:
  +stencil = window(context)
  +s = stencil_stride(stencil)
  +ow = width_word(context)
  Model.wrap_lanes(input,(position / ow * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow * s + U32.from_nat(t) + stencil_column_origin(stencil) : U32),
    (stencil_base(stencil) + U32.from_nat(c) * stencil_height(stencil) * stencil_source_width(stencil) : U32),
    stencil_height(stencil),stencil_source_width(stencil),s,
    ((position / ow + 1) * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow : U32),ow,U32.from_nat(t),(0 - stencil_column_origin(stencil) : U32))

def wrap_same_sample(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  decoded: Decoded<context,c,r,t>,location: SameLane<context,position,lane>,
  guard: {U32.is_lt((position % width_word(context) + U32.from_nat(lane) : U32),width_word(context)) == True{} : Bool}) ->
  {Samples.wrap_sample(input,wrap_context(context,position,c,r,t),lane) == Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  Geometry{stencil,channels,cap,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  Packing.Stencil{base,h,w,tr,tc,s,oy,ox,+ow,inner} = stencil
  %Equal.sym(Bool,U32.is_lt((position % ow + U32.from_nat(lane) : U32),ow),True{},guard) :
    {Samples.wrap_value(_,input,(position / ow * s + U32.from_nat(r) + oy : U32),(position % ow * s + U32.from_nat(t) + ox : U32),(base + U32.from_nat(c) * h * w : U32),h,w,s,
      ((position / ow + 1) * s + U32.from_nat(r) + oy : U32),(position % ow : U32),ow,U32.from_nat(t),(0 - ox : U32),U32.from_nat(lane)) == Samples.window_sample(input,Samples.window_context(stencil,position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}
  row_lane_given(context,input,position,c,r,t,lane,decoded,location)

def wrap_next_sample(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  decoded: Decoded<context,c,r,t>,location: NextLane<context,position,lane>,
  guard: {U32.is_lt((position % width_word(context) + U32.from_nat(lane) : U32),width_word(context)) == False{} : Bool}) ->
  {Samples.wrap_sample(input,wrap_context(context,position,c,r,t),lane) == Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  Geometry{stencil,channels,cap,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = context
  Packing.Stencil{+base,+h,+w,tr,tc,+s,+oy,+ox,+ow,inner} = stencil
  NextLane{row,column} = location
  +iy = ((position / ow + 1) * s + U32.from_nat(r) + oy : U32)
  +x = ((position % ow + U32.from_nat(lane) - ow : U32) * s + U32.from_nat(t) : U32)
  +cb = (base + U32.from_nat(c) * h * w : U32)
  +canonical = Samples.window_context(stencil,position,U32.from_nat(reduction(context,c,r,t)))
  +decoded_value = {NativeSamples.WindowPixelBlock{position,U32.from_nat(r),U32.from_nat(t),cb,h,w,s,oy,ox,ow} : NativeSamples.WindowPixelBlock}
  %Equal.sym(Bool,U32.is_lt((position % ow + U32.from_nat(lane) : U32),ow),False{},guard) :
    {Samples.wrap_value(_,input,(position / ow * s + U32.from_nat(r) + oy : U32),(position % ow * s + U32.from_nat(t) + ox : U32),cb,h,w,s,iy,(position % ow : U32),ow,U32.from_nat(t),(0 - ox : U32),U32.from_nat(lane)) == Samples.window_sample(input,canonical,lane) : F32}
  %Equal.sym(NativeSamples.WindowPixelBlock,canonical,decoded_value,decoded_context(context,position,c,r,t,decoded)) :
    {Samples.value(input,iy,(x - (0 - ox : U32) : U32),cb,h,w) == Samples.window_sample(input,_,lane) : F32}
  %Equal.sym(U32,((position + U32.from_nat(lane) : U32) / ow : U32),(position / ow + 1 : U32),row) :
    {Samples.value(input,iy,(x - (0 - ox : U32) : U32),cb,h,w) == Samples.value(input,(_ * s + U32.from_nat(r) + oy : U32),((position + U32.from_nat(lane) : U32) % ow * s + U32.from_nat(t) + ox : U32),cb,h,w) : F32}
  %Equal.sym(U32,((position + U32.from_nat(lane) : U32) % ow : U32),(position % ow + U32.from_nat(lane) - ow : U32),column) :
    {Samples.value(input,iy,(x - (0 - ox : U32) : U32),cb,h,w) == Samples.value(input,iy,(_ * s + U32.from_nat(t) + ox : U32),cb,h,w) : F32}
  Equal.cong(U32,F32,ix => Samples.value(input,iy,ix,cb,h,w),(x - (0 - ox : U32) : U32),(x + ox : U32),
    Equal.trans(U32,(x - (0 - ox : U32) : U32),(x + (0 - (0 - ox : U32) : U32) : U32),(x + ox : U32),
      Equal.sym(U32,(x + (0 - (0 - ox : U32) : U32) : U32),(x - (0 - ox : U32) : U32),Arithmetic.add_zero_subtract(x,(0 - ox : U32))),
      Equal.cong(U32,U32,o => (x + o : U32),(0 - (0 - ox : U32) : U32),ox,Arithmetic.negate_twice(ox))))

def wrap_lane_selected(same: Bool,+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+decided: {Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)) == same : Bool}) ->
  {Samples.wrap_sample(input,wrap_context(context,position,c,r,t),lane) == Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  match same:
    case True{}:
      wrap_same_sample(context,input,position,c,r,t,lane,decode(context,c,r,t,c_inside,r_inside,t_inside),same_lane(context,position,lane,position_room,lane_inside,decided),lane_guard(context,position,lane,True{},position_room,lane_inside,decided))
    case False{}:
      wrap_next_sample(context,input,position,c,r,t,lane,decode(context,c,r,t,c_inside,r_inside,t_inside),next_lane(context,position,lane,within,position_room,lane_inside,decided),lane_guard(context,position,lane,False{},position_room,lane_inside,decided))

def wrap_lane(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,+lane: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Samples.wrap_sample(input,wrap_context(context,position,c,r,t),lane) == Samples.window_sample(input,Samples.window_context(window(context),position,U32.from_nat(reduction(context,c,r,t))),lane) : F32}:
  wrap_lane_selected(Nat.is_lt(Nat.add(Nat.mod(U32.to_nat(position),width(context)),lane),width(context)),context,input,position,c,r,t,lane,within,position_room,c_inside,r_inside,t_inside,lane_inside,{==})

def first_value(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat) -> F32:
  +stencil = window(context)
  +s = stencil_stride(stencil)
  +ow = width_word(context)
  Samples.value(input,(position / ow * s + U32.from_nat(r) + stencil_row_origin(stencil) : U32),
    (position % ow * s + U32.from_nat(t) + stencil_column_origin(stencil) : U32),
    (stencil_base(stencil) + U32.from_nat(c) * stencil_height(stencil) * stencil_source_width(stencil) : U32),stencil_height(stencil),stencil_source_width(stencil))

def first_lane(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool}) ->
  {first_value(context,input,position,c,r,t) == Samples.window_value(input,window(context),position,U32.from_nat(reduction(context,c,r,t))) : F32}:
  Geometry{stencil,channels,cap,word_ok,rp,cp,ip,+wp,ie,ce,rr,cr,ar,ir,wr} = context
  Packing.Stencil{base,h,w,tr,tc,+s,oy,+ox,+ow,inner} = stencil
  +ix = (position % ow * s + U32.from_nat(t) + ox : U32)
  %Samples.zero_lane_column(ix,s) :
    {Samples.value(input,(position / ow * s + U32.from_nat(r) + oy : U32),_,(base + U32.from_nat(c) * h * w : U32),h,w) == Samples.window_value(input,stencil,position,U32.from_nat(reduction(context,c,r,t))) : F32}
  row_lane_given(context,input,position,c,r,t,0n,decode(context,c,r,t,c_inside,r_inside,t_inside),same_lane(context,position,0n,position_room,{==},
    %Equal.sym(Nat,Nat.add(Nat.mod(U32.to_nat(position),width(context)),0n),Nat.mod(U32.to_nat(position),width(context)),Addition.zero_right(Nat.mod(U32.to_nat(position),width(context)))) : {Nat.is_lt(_,width(context)) == True{} : Bool}
    DivisionOrder.remainder_small(U32.to_nat(position),width(context),wp)))

def wrap_entry(+context: Geometry,+input: Tree.Tree<F32>,+position: U32,+c: Nat,+r: Nat,+t: Nat,
  +within: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(position),width(context)),8n),Nat.add(width(context),width(context))) == True{} : Bool},
  +position_room: {Nat.is_le(Nat.add(U32.to_nat(position),8n),U32.to_nat(capacity(context))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,channels(context)) == True{} : Bool},+r_inside: {Nat.is_lt(r,rows(context)) == True{} : Bool},+t_inside: {Nat.is_lt(t,columns(context)) == True{} : Bool}) ->
  {wrap_vector(context,input,position,c,r,t) == Samples.window_vector(input,window(context),position,U32.from_nat(reduction(context,c,r,t))) : Tile.Tile8}:
  TileLaws.tiles_equal(wrap_vector(context,input,position,c,r,t),Samples.window_vector(input,window(context),position,U32.from_nat(reduction(context,c,r,t))),
    first_lane(context,input,position,c,r,t,position_room,c_inside,r_inside,t_inside),
    wrap_lane(context,input,position,c,r,t,1n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,2n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,3n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,4n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,5n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,6n,within,position_room,c_inside,r_inside,t_inside,{==}),
    wrap_lane(context,input,position,c,r,t,7n,within,position_room,c_inside,r_inside,t_inside,{==}))
