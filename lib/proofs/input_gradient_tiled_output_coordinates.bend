# Output address digits identify the pixel, tile group, row and column
# uniquely.
import Base
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_output_layout.bend as Layout
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./binary_radix.bend as Radix

def relative(+query: Chunks.Query,+point: Layout.Point<query>) -> Nat:
  Nat.sub(Layout.address(query,point),U32.to_nat(Chunks.target(query)))
def relative_exact(+query: Chunks.Query,+point: Layout.Point<query>) -> {relative(query,point) == Layout.offset(query,point) : Nat}:
  Order.subtract_shift(U32.to_nat(Chunks.target(query)),Layout.offset(query,point))
def groups_positive(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.is_lt(0n,Layout.groups(query)) == True{} : Bool}:
  Layout.Point{+group,row,column,+gi,rl,ru,xi} = point
  Order.positive_bound(group,Layout.groups(query),gi)
def pixel_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.div(relative(query,point),Layout.groups(query)) == Layout.pixel(query,point) : Nat}:
  Layout.Point{+group,row,column,+gi,rl,ru,xi} = point
  %Equal.sym(Nat,relative(query,point),Layout.offset(query,point),relative_exact(query,point)) : {Nat.div(_,Layout.groups(query)) == Layout.pixel(query,point) : Nat}
  Radix.quotient(Layout.pixel(query,point),group,Layout.groups(query),groups_positive(query,point),gi)
def group_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.mod(relative(query,point),Layout.groups(query)) == Layout.group(query,point) : Nat}:
  Layout.Point{+group,row,column,+gi,rl,ru,xi} = point
  %Equal.sym(Nat,relative(query,point),Layout.offset(query,point),relative_exact(query,point)) : {Nat.mod(_,Layout.groups(query)) == Layout.group(query,point) : Nat}
  Radix.remainder(Layout.pixel(query,point),group,Layout.groups(query),groups_positive(query,point),gi)
def width_positive(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.is_lt(0n,Layout.width(query)) == True{} : Bool}:
  Layout.Point{group,row,+column,ci,rl,ru,+xi} = point
  Order.positive_bound(column,Layout.width(query),xi)
def local_row_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.div(Layout.pixel(query,point),Layout.width(query)) == Layout.local_row(query,point) : Nat}:
  Layout.Point{group,row,+column,ci,rl,ru,+xi} = point
  Radix.quotient(Layout.local_row(query,point),column,Layout.width(query),width_positive(query,point),xi)
def column_decoded(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.mod(Layout.pixel(query,point),Layout.width(query)) == Layout.column(query,point) : Nat}:
  Layout.Point{group,row,+column,ci,rl,ru,+xi} = point
  Radix.remainder(Layout.local_row(query,point),column,Layout.width(query),width_positive(query,point),xi)
def address_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.address(query,left) == Layout.address(query,right) : Nat}:
  Equal.trans(Nat,Layout.address(query,left),U32.to_nat(Layout.native_address(query,left)),Layout.address(query,right),Equal.sym(Nat,U32.to_nat(Layout.native_address(query,left)),Layout.address(query,left),Layout.native_address_exact(query,layout,left)),
    Equal.trans(Nat,U32.to_nat(Layout.native_address(query,left)),U32.to_nat(Layout.native_address(query,right)),Layout.address(query,right),Equal.cong(U32,Nat,U32.to_nat,Layout.native_address(query,left),Layout.native_address(query,right),same),Layout.native_address_exact(query,layout,right)))
def group_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.group(query,left) == Layout.group(query,right) : Nat}:
  Equal.trans(Nat,Layout.group(query,left),Nat.mod(relative(query,left),Layout.groups(query)),Layout.group(query,right),Equal.sym(Nat,Nat.mod(relative(query,left),Layout.groups(query)),Layout.group(query,left),group_decoded(query,left)),
    Equal.trans(Nat,Nat.mod(relative(query,left),Layout.groups(query)),Nat.mod(relative(query,right),Layout.groups(query)),Layout.group(query,right),Equal.cong(Nat,Nat,address => Nat.mod(Nat.sub(address,U32.to_nat(Chunks.target(query))),Layout.groups(query)),Layout.address(query,left),Layout.address(query,right),address_equal(query,layout,left,right,same)),group_decoded(query,right)))
def pixel_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.pixel(query,left) == Layout.pixel(query,right) : Nat}:
  Equal.trans(Nat,Layout.pixel(query,left),Nat.div(relative(query,left),Layout.groups(query)),Layout.pixel(query,right),Equal.sym(Nat,Nat.div(relative(query,left),Layout.groups(query)),Layout.pixel(query,left),pixel_decoded(query,left)),
    Equal.trans(Nat,Nat.div(relative(query,left),Layout.groups(query)),Nat.div(relative(query,right),Layout.groups(query)),Layout.pixel(query,right),Equal.cong(Nat,Nat,address => Nat.div(Nat.sub(address,U32.to_nat(Chunks.target(query))),Layout.groups(query)),Layout.address(query,left),Layout.address(query,right),address_equal(query,layout,left,right,same)),pixel_decoded(query,right)))
def column_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.column(query,left) == Layout.column(query,right) : Nat}:
  Equal.trans(Nat,Layout.column(query,left),Nat.mod(Layout.pixel(query,left),Layout.width(query)),Layout.column(query,right),Equal.sym(Nat,Nat.mod(Layout.pixel(query,left),Layout.width(query)),Layout.column(query,left),column_decoded(query,left)),
    Equal.trans(Nat,Nat.mod(Layout.pixel(query,left),Layout.width(query)),Nat.mod(Layout.pixel(query,right),Layout.width(query)),Layout.column(query,right),Equal.cong(Nat,Nat,pixel => Nat.mod(pixel,Layout.width(query)),Layout.pixel(query,left),Layout.pixel(query,right),pixel_equal(query,layout,left,right,same)),column_decoded(query,right)))
def local_row_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.local_row(query,left) == Layout.local_row(query,right) : Nat}:
  Equal.trans(Nat,Layout.local_row(query,left),Nat.div(Layout.pixel(query,left),Layout.width(query)),Layout.local_row(query,right),Equal.sym(Nat,Nat.div(Layout.pixel(query,left),Layout.width(query)),Layout.local_row(query,left),local_row_decoded(query,left)),
    Equal.trans(Nat,Nat.div(Layout.pixel(query,left),Layout.width(query)),Nat.div(Layout.pixel(query,right),Layout.width(query)),Layout.local_row(query,right),Equal.cong(Nat,Nat,pixel => Nat.div(pixel,Layout.width(query)),Layout.pixel(query,left),Layout.pixel(query,right),pixel_equal(query,layout,left,right,same)),local_row_decoded(query,right)))
def row_restored(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,point)) == Layout.row(query,point) : Nat}:
  Layout.Point{group,+row,column,ci,+rl,ru,xi} = point
  Order.restore_shift(U32.to_nat(Chunks.first(query)),row,rl)
def row_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+left: Layout.Point<query>,+right: Layout.Point<query>,+same: {Layout.native_address(query,left) == Layout.native_address(query,right) : U32}) -> {Layout.row(query,left) == Layout.row(query,right) : Nat}:
  Equal.trans(Nat,Layout.row(query,left),Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Layout.row(query,right),Equal.sym(Nat,Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Layout.row(query,left),row_restored(query,left)),
    Equal.trans(Nat,Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,left)),Nat.add(U32.to_nat(Chunks.first(query)),Layout.local_row(query,right)),Layout.row(query,right),Equal.cong(Nat,Nat,row => Nat.add(U32.to_nat(Chunks.first(query)),row),Layout.local_row(query,left),Layout.local_row(query,right),local_row_equal(query,layout,left,right,same)),row_restored(query,right)))
