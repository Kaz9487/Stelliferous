# Output address digits identify the channel, row and column uniquely.
import Base
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_output_layout.bend as Layout
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./binary_radix.bend as Radix

def canonical_address(+query: Chunks.Query,+point: Layout.Point<query>) ->
  {Layout.address(query,point) == Nat.add(U32.to_nat(Chunks.target(query)),Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point))) : Nat}:
  +base = U32.to_nat(Chunks.target(query))
  +channel = Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query)))
  %Equal.sym(Nat,Nat.add(base,0n),base,Addition.zero_right(base)) : {Nat.add(Nat.add(_,channel),Layout.pixel(query,point)) == Nat.add(base,Nat.add(channel,Layout.pixel(query,point))) : Nat}
  Addition.associative(base,channel,Layout.pixel(query,point))
def relative(+query: Chunks.Query,+point: Layout.Point<query>) -> Nat:
  Nat.sub(Layout.address(query,point),U32.to_nat(Chunks.target(query)))
def relative_exact(+query: Chunks.Query,+point: Layout.Point<query>) ->
  {relative(query,point) == Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point)) : Nat}:
  +base = U32.to_nat(Chunks.target(query))
  +offset = Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point))
  %Equal.sym(Nat,Layout.address(query,point),Nat.add(base,offset),canonical_address(query,point)) : {Nat.sub(_,base) == offset : Nat}
  Order.subtract_shift(base,offset)
def pixel_inside_plane(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>) ->
  {Nat.is_lt(Layout.pixel(query,point),U32.to_nat(Chunks.plane(query))) == True{} : Bool}:
  Layout.Layout{capacity,capacity_ok,plane_span,room} = layout
  Bounds.nat_trans(Layout.pixel(query,point),Layout.pixels(query),U32.to_nat(Chunks.plane(query)),Layout.pixel_inside(query,point),plane_span)
def plane_positive(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>) -> {Nat.is_lt(0n,U32.to_nat(Chunks.plane(query))) == True{} : Bool}:
  Order.positive_bound(Layout.pixel(query,point),U32.to_nat(Chunks.plane(query)),pixel_inside_plane(query,layout,point))
def channel_decoded(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>) -> {Nat.div(relative(query,point),U32.to_nat(Chunks.plane(query))) == Layout.channel(query,point) : Nat}:
  %Equal.sym(Nat,relative(query,point),Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point)),relative_exact(query,point)) : {Nat.div(_,U32.to_nat(Chunks.plane(query))) == Layout.channel(query,point) : Nat}
  Radix.quotient(Layout.channel(query,point),Layout.pixel(query,point),U32.to_nat(Chunks.plane(query)),plane_positive(query,layout,point),pixel_inside_plane(query,layout,point))
def pixel_decoded(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>) -> {Nat.mod(relative(query,point),U32.to_nat(Chunks.plane(query))) == Layout.pixel(query,point) : Nat}:
  %Equal.sym(Nat,relative(query,point),Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point)),relative_exact(query,point)) : {Nat.mod(_,U32.to_nat(Chunks.plane(query))) == Layout.pixel(query,point) : Nat}
  Radix.remainder(Layout.channel(query,point),Layout.pixel(query,point),U32.to_nat(Chunks.plane(query)),plane_positive(query,layout,point),pixel_inside_plane(query,layout,point))
def width_positive(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.is_lt(0n,Layout.width(query)) == True{} : Bool}:
  Layout.Point{channel,row,+column,ci,rl,ru,+xi} = point
  Order.positive_bound(column,Layout.width(query),xi)
def local_row_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.div(Layout.pixel(query,point),Layout.width(query)) == Layout.local_row(query,point) : Nat}:
  Layout.Point{channel,row,+column,ci,rl,ru,+xi} = point
  Radix.quotient(Layout.local_row(query,point),column,Layout.width(query),width_positive(query,point),xi)
def column_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.mod(Layout.pixel(query,point),Layout.width(query)) == Layout.column(query,point) : Nat}:
  Layout.Point{channel,row,+column,ci,rl,ru,+xi} = point
  Radix.remainder(Layout.local_row(query,point),column,Layout.width(query),width_positive(query,point),xi)
def address_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.address(query,left) == Layout.address(query,right) : Nat}:
  Equal.trans(Nat,Layout.address(query,left),U32.to_nat(Layout.native_address(query,left)),Layout.address(query,right),Equal.sym(Nat,U32.to_nat(Layout.native_address(query,left)),Layout.address(query,left),Layout.native_address_exact(query,layout,left)),
    Equal.trans(Nat,U32.to_nat(Layout.native_address(query,left)),U32.to_nat(Layout.native_address(query,right)),Layout.address(query,right),Equal.cong(U32,Nat,U32.to_nat,Layout.native_address(query,left),Layout.native_address(query,right),same),Layout.native_address_exact(query,layout,right)))
def channel_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.channel(query,left) == Layout.channel(query,right) : Nat}:
  Equal.trans(Nat,Layout.channel(query,left),Nat.div(relative(query,left),U32.to_nat(Chunks.plane(query))),Layout.channel(query,right),Equal.sym(Nat,Nat.div(relative(query,left),U32.to_nat(Chunks.plane(query))),Layout.channel(query,left),channel_decoded(query,layout,left)),
    Equal.trans(Nat,Nat.div(relative(query,left),U32.to_nat(Chunks.plane(query))),Nat.div(relative(query,right),U32.to_nat(Chunks.plane(query))),Layout.channel(query,right),Equal.cong(Nat,Nat,address => Nat.div(Nat.sub(address,U32.to_nat(Chunks.target(query))),U32.to_nat(Chunks.plane(query))),Layout.address(query,left),Layout.address(query,right),address_equal(query,layout,left,right,same)),channel_decoded(query,layout,right)))
def pixel_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.pixel(query,left) == Layout.pixel(query,right) : Nat}:
  Equal.trans(Nat,Layout.pixel(query,left),Nat.mod(relative(query,left),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,right),Equal.sym(Nat,Nat.mod(relative(query,left),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,left),pixel_decoded(query,layout,left)),
    Equal.trans(Nat,Nat.mod(relative(query,left),U32.to_nat(Chunks.plane(query))),Nat.mod(relative(query,right),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,right),Equal.cong(Nat,Nat,address => Nat.mod(Nat.sub(address,U32.to_nat(Chunks.target(query))),U32.to_nat(Chunks.plane(query))),Layout.address(query,left),Layout.address(query,right),address_equal(query,layout,left,right,same)),pixel_decoded(query,layout,right)))
def column_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.column(query,left) == Layout.column(query,right) : Nat}:
  Equal.trans(Nat,Layout.column(query,left),Nat.mod(Layout.pixel(query,left),Layout.width(query)),Layout.column(query,right),Equal.sym(Nat,Nat.mod(Layout.pixel(query,left),Layout.width(query)),Layout.column(query,left),column_decoded(query,left)),
    Equal.trans(Nat,Nat.mod(Layout.pixel(query,left),Layout.width(query)),Nat.mod(Layout.pixel(query,right),Layout.width(query)),Layout.column(query,right),Equal.cong(Nat,Nat,pixel => Nat.mod(pixel,Layout.width(query)),Layout.pixel(query,left),Layout.pixel(query,right),pixel_equal(query,layout,left,right,same)),column_decoded(query,right)))
def local_row_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.local_row(query,left) == Layout.local_row(query,right) : Nat}:
  Equal.trans(Nat,Layout.local_row(query,left),Nat.div(Layout.pixel(query,left),Layout.width(query)),Layout.local_row(query,right),Equal.sym(Nat,Nat.div(Layout.pixel(query,left),Layout.width(query)),Layout.local_row(query,left),local_row_decoded(query,left)),
    Equal.trans(Nat,Nat.div(Layout.pixel(query,left),Layout.width(query)),Nat.div(Layout.pixel(query,right),Layout.width(query)),Layout.local_row(query,right),Equal.cong(Nat,Nat,pixel => Nat.div(pixel,Layout.width(query)),Layout.pixel(query,left),Layout.pixel(query,right),pixel_equal(query,layout,left,right,same)),local_row_decoded(query,right)))
def row_restored(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,point)) == Layout.row(query,point) : Nat}:
  Layout.Point{channel,+row,column,ci,+rl,ru,xi} = point
  Order.restore_shift(U32.to_nat(Chunks.first(query)),row,rl)
def row_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.row(query,left) == Layout.row(query,right) : Nat}:
  Equal.trans(Nat,Layout.row(query,left),Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Layout.row(query,right),Equal.sym(Nat,Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Layout.row(query,left),row_restored(query,left)),
    Equal.trans(Nat,Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,right)),Layout.row(query,right),Equal.cong(Nat,Nat,row => Nat.add(U32.to_nat(Chunks.first(query)),row),Layout.local_row(query,left),Layout.local_row(query,right),local_row_equal(query,layout,left,right,same)),row_restored(query,right)))
