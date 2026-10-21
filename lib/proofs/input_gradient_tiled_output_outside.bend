# Addresses outside the leaf's output rows are disjoint from every written point.
import Base
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_output_layout.bend as Layout
import ./input_gradient_tiled_output_coordinates.bend as Coordinates
import ./input_gradient_phase_domain.bend as Domain
import ./strided_range_bounds.bend as Strides
import ./nat_order.bend as Order
import ./word_encoding.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic

def end(+query: Chunks.Query) -> Nat:
  Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(Domain.channels(Chunks.geometry(query)),U32.to_nat(Chunks.plane(query))))
type Outside<-query: Chunks.Query,-address: U32> is Data:
  Before{before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Chunks.target(query))) == True{} : Bool}}
  After{after: {Nat.is_le(end(query),U32.to_nat(address)) == True{} : Bool}}
  Gap{gap: {Nat.is_le(Layout.pixels(query),Nat.mod(Nat.sub(U32.to_nat(address),U32.to_nat(Chunks.target(query))),U32.to_nat(Chunks.plane(query)))) == True{} : Bool}}
def region_outside(+query: Chunks.Query,+address: U32,outside: Outside<query,address>) -> Outside<Layout.region(query),address>:
  match outside:
    case Before{before}: Before{before}
    case After{after}: After{after}
    case Gap{gap}: Gap{gap}
def point_after_begin(+query: Chunks.Query,+point: Layout.Point<query>) -> {Nat.is_le(U32.to_nat(Chunks.target(query)),Layout.address(query,point)) == True{} : Bool}:
  +base = U32.to_nat(Chunks.target(query))
  +offset = Nat.add(Nat.mul(Layout.channel(query,point),U32.to_nat(Chunks.plane(query))),Layout.pixel(query,point))
  %Equal.sym(Nat,Layout.address(query,point),Nat.add(base,offset),Coordinates.canonical_address(query,point)) : {Nat.is_le(base,_) == True{} : Bool}
  Order.le_add(base,offset)
def point_before_end(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>) -> {Nat.is_lt(Layout.address(query,point),end(query)) == True{} : Bool}:
  Layout.Layout{capacity,capacity_ok,+plane_span,room} = layout
  Layout.Point{+channel,row,column,+ci,rl,ru,xi} = point
  Strides.query_inside(U32.to_nat(Chunks.target(query)),0n,U32.to_nat(Chunks.plane(query)),Layout.pixels(query),Domain.channels(Chunks.geometry(query)),channel,Layout.pixel(query,point),end(query),ci,Layout.pixel_inside(query,point),plane_span,Order.reflexive(end(query)))
def collision_equal(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>,+address: U32,outside: Outside<query,address>,+equal: {U32.to_nat(address) == Layout.address(query,point) : Nat}) -> Empty:
  match outside:
    case Before{before}:
      +position = Layout.address(query,point)
      Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(position,U32.to_nat(Chunks.target(query))),True{},Equal.sym(Bool,Nat.is_lt(position,U32.to_nat(Chunks.target(query))),False{},Order.not_below(U32.to_nat(Chunks.target(query)),position,point_after_begin(query,point))),
        %equal : {Nat.is_lt(_,U32.to_nat(Chunks.target(query))) == True{} : Bool} before))
    case After{after}:
      +position = Layout.address(query,point)
      Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(position,end(query)),True{},Equal.sym(Bool,Nat.is_lt(position,end(query)),False{},Order.not_below(end(query),position,%equal : {Nat.is_le(end(query),_) == True{} : Bool} after)),point_before_end(query,layout,point)))
    case Gap{gap}:
      +below = Layout.pixel_inside(query,point)
      +ordered = {
        %Coordinates.pixel_decoded(query,layout,point) : {Nat.is_le(Layout.pixels(query),_) == True{} : Bool}
        %equal : {Nat.is_le(Layout.pixels(query),Nat.mod(Nat.sub(_,U32.to_nat(Chunks.target(query))),U32.to_nat(Chunks.plane(query)))) == True{} : Bool}
        gap : {Nat.is_le(Layout.pixels(query),Layout.pixel(query,point)) == True{} : Bool}}
      Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(Layout.pixel(query,point),Layout.pixels(query)),True{},Equal.sym(Bool,Nat.is_lt(Layout.pixel(query,point),Layout.pixels(query)),False{},Order.not_below(Layout.pixels(query),Layout.pixel(query,point),ordered)),below))
def collision(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>,+address: U32,outside: Outside<query,address>,same: {Layout.native_address(query,point) == address : U32}) -> Empty:
  +position = Layout.address(query,point)
  +equal = Equal.trans(Nat,U32.to_nat(address),U32.to_nat(Layout.native_address(query,point)),position,Equal.cong(U32,Nat,U32.to_nat,address,Layout.native_address(query,point),Equal.sym(U32,Layout.native_address(query,point),address,same)),Layout.native_address_exact(query,layout,point))
  collision_equal(query,layout,point,address,outside,equal)
def compared(result: Bool,+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>,+address: U32,+outside: Outside<query,address>,same: {U32.is_eq(Layout.native_address(query,point),address) == result : Bool}) -> {result == False{} : Bool}:
  match result:
    case False{}: {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool},collision(query,layout,point,address,outside,Arithmetic.equal_of_is_eq(Layout.native_address(query,point),address,same)))
def disjoint(+query: Chunks.Query,+layout: Layout.Layout<query>,+point: Layout.Point<query>,+address: U32,+outside: Outside<query,address>) -> {U32.is_eq(Layout.native_address(query,point),address) == False{} : Bool}:
  compared(U32.is_eq(Layout.native_address(query,point),address),query,layout,point,address,outside,{==})
