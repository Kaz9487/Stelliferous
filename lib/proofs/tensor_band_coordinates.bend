# Coordinate semantics for a whole-channel row band, independent of an
# operator: a local row and its image row translate both ways, and a global
# index splits into its image, channel, row and column parts.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./storage_shape.bend as Shape

def row_inside(+first: Nat,+rows: Nat,+height: Nat,+row: Nat,inside: {Nat.is_lt(row,rows) == True{} : Bool},
  fits: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first,row),height) == True{} : Bool}:
  Bounds.nat_trans(Nat.add(first,row),Nat.add(first,rows),height,Order.offset_lt(first,row,rows,inside),fits)

# A located row in an image-local band has the same image and image-relative
# row as the global NCHW point. Both coordinates follow from radix uniqueness.
def row_coordinates(+first: Nat,+height: Nat,+rows: Nat,+local: Nat,+image: Nat,+row: Nat,
  +height_positive: {Nat.is_lt(0n,height) == True{} : Bool},+band_span: {Nat.is_le(Nat.add(Nat.mod(first,height),rows),height) == True{} : Bool},
  +local_inside: {Nat.is_lt(local,rows) == True{} : Bool},+row_bound: {Nat.is_lt(row,height) == True{} : Bool},
  +located: {Nat.add(Nat.mul(image,height),row) == Nat.add(first,local) : Nat}) ->
  Shape.Both<{Nat.div(first,height) == image : Nat},{Nat.add(Nat.mod(first,height),local) == row : Nat}>:
  +quotient = Nat.div(first,height)
  +remainder = Nat.mod(first,height)
  +row_exact = Equal.trans(Nat,Nat.add(first,local),Nat.add(Nat.add(Nat.mul(quotient,height),remainder),local),Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),
    Equal.cong(Nat,Nat,value => Nat.add(value,local),first,Nat.add(Nat.mul(quotient,height),remainder),DivisionOrder.quotient_exact(first,height,height_positive)),
    Addition.associative(Nat.mul(quotient,height),remainder,local))
  +same = Equal.trans(Nat,Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),Nat.add(first,local),Nat.add(Nat.mul(image,height),row),
    Equal.sym(Nat,Nat.add(first,local),Nat.add(Nat.mul(quotient,height),Nat.add(remainder,local)),row_exact),
    Equal.sym(Nat,Nat.add(Nat.mul(image,height),row),Nat.add(first,local),located))
  +inside = row_inside(remainder,rows,height,local,local_inside,band_span)
  Shape.Both{Radix.row_unique(quotient,image,height,Nat.add(remainder,local),row,height_positive,inside,row_bound,same),
    Radix.col_unique(quotient,image,height,Nat.add(remainder,local),row,height_positive,inside,row_bound,same)}

def global_parts(image: Nat,first: Nat,channel: Nat,remainder: Nat,channels: Nat,height: Nat,width: Nat) -> Nat:
  RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,first,remainder)

def row_join(+channel: Nat,+height: Nat,+width: Nat,+first: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(channel,height,width,first,Nat.add(Nat.mul(row,width),column)) == RowMajor.index(channel,height,width,Nat.add(first,row),column) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(first,row),width),Nat.add(Nat.mul(first,width),Nat.mul(row,width)),RowMajor.distrib(first,row,width)) :
    {RowMajor.index(channel,height,width,first,Nat.add(Nat.mul(row,width),column)) == Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(_,column)) : Nat}
  Equal.cong(Nat,Nat,n => Nat.add(Nat.mul(channel,Nat.mul(height,width)),n),
    Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)),Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,width),Nat.mul(row,width)),column),Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)),Addition.associative(Nat.mul(first,width),Nat.mul(row,width),column)))

def native_parts(+image: U32,+first: U32,+channel: U32,+remainder: U32,+channels: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +room: {Nat.is_le(global_parts(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channel),U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) ==
    global_parts(U32.to_nat(image),U32.to_nat(first),U32.to_nat(channel),U32.to_nat(remainder),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)) : Nat}:
  +ic = Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(channels)),U32.to_nat(channel))
  +row = Nat.add(Nat.mul(ic,U32.to_nat(height)),U32.to_nat(first))
  +mixed = RowMajor.mixed(ic,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),U32.to_nat(remainder))
  +outer_room = {
    %Equal.sym(Nat,Nat.add(Nat.mul(row,U32.to_nat(width)),U32.to_nat(remainder)),RowMajor.index(ic,U32.to_nat(height),U32.to_nat(width),U32.to_nat(first),U32.to_nat(remainder)),mixed) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(Nat.add(Nat.mul(row,U32.to_nat(width)),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}}
  +row_room = Encoded.left_bound(row,U32.to_nat(width),U32.to_nat(remainder),U32.to_nat(capacity),width_positive,outer_room)
  +ic_exact = Arithmetic.multiply_add_exact(image,channels,channel,capacity,capacity_ok,Encoded.left_bound(ic,U32.to_nat(height),U32.to_nat(first),U32.to_nat(capacity),height_positive,row_room))
  +row_exact = {
    %ic_exact : {U32.to_nat(((image * channels + channel) * height + first : U32)) == Nat.add(Nat.mul(_,U32.to_nat(height)),U32.to_nat(first)) : Nat}
    Arithmetic.multiply_add_exact((image * channels + channel : U32),height,first,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((image * channels + channel : U32)),ic,ic_exact) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(height)),U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}
      row_room) : {U32.to_nat(((image * channels + channel) * height + first : U32)) == row : Nat}}
  %mixed : {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) == _ : Nat}
  %row_exact : {U32.to_nat((((image * channels + channel) * height + first) * width + remainder : U32)) == Nat.add(Nat.mul(_,U32.to_nat(width)),U32.to_nat(remainder)) : Nat}
  Arithmetic.multiply_add_exact(((image * channels + channel) * height + first : U32),width,remainder,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(((image * channels + channel) * height + first : U32)),row,row_exact) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(width)),U32.to_nat(remainder)),U32.to_nat(capacity)) == True{} : Bool}
    outer_room)
