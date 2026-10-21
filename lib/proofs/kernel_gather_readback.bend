# Readback follows the actual channel copies and their clipped intervals.
import Base
import ../kernel_gather.bend as Gather
import ./kernel_gather_blocks.bend as Blocks
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./contiguous_copy.bend as Contiguous
import ./contiguous_copy_readback.bend as Copy
import ./kernel_window_copy.bend as Planes
import ./tensor_band_plan.bend as Plan
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_interval.bend as Interval

def plane_room(+channel: U32,+remaining: Nat,+offset: U32,+width: U32,+spatial: U32,+capacity: U32,
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool},
  slots: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),1n+remaining),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),U32.to_nat(width))),Addition.associative(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset),U32.to_nat(width))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  Order.transitive(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),U32.to_nat(width))),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(spatial)),U32.to_nat(capacity),
    Order.offset_le(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial),clipped),Plan.image_room(channel,spatial,remaining,capacity,slots))

def destination_exact(+channel: U32,+remaining: Nat,+offset: U32,+width: U32,+spatial: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool},
  slots: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),1n+remaining),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((channel * spatial + offset : U32)) == Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)) : Nat}:
  Arithmetic.multiply_add_exact(channel,spatial,offset,capacity,capacity_ok,
    Order.transitive(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),U32.to_nat(capacity),Order.le_add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),plane_room(channel,remaining,offset,width,spatial,capacity,clipped,slots)))

def stores_inside(remaining: Nat,+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +channels_room: {Nat.is_le(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(capacity)) == True{} : Bool},
  +slots_room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool}) ->
  Readback.Inside(~F32,capacity,Blocks.writes(remaining,channel,width,offset,spatial,span,input)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +next = (channel + 1 : U32)
      +next_exact = Planes.channel_successor(channel,rest,capacity,capacity_ok,channels_room)
      +start_exact = destination_exact(channel,rest,offset,width,spatial,capacity,capacity_ok,clipped,slots_room)
      Readback.inside_append(~F32,capacity,Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + offset : U32)),Blocks.writes(rest,next,width,offset,spatial,span,input),
        Copy.writes_inside(~F32,U32.to_nat(width),0n,input,(channel * span : U32),(channel * spatial + offset : U32),capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat((channel * spatial + offset : U32)),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {Nat.is_le(Nat.add(Nat.add(_,0n),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),0n),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),Addition.zero_right(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)))) : {Nat.is_le(Nat.add(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
          plane_room(channel,rest,offset,width,spatial,capacity,clipped,slots_room)),
        stores_inside(rest,next,width,offset,spatial,span,input,capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool}
          %Addition.successor_right(U32.to_nat(channel),rest) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} channels_room,
          %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}
          %Addition.successor_right(U32.to_nat(channel),rest) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots_room,clipped))

def later_selected(remaining: Nat,+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+query: U32,+previous: F32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +channels_room: {Nat.is_le(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(capacity)) == True{} : Bool},
  +slots_room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(query),Nat.mul(U32.to_nat(channel),U32.to_nat(spatial))) == True{} : Bool}) ->
  {Readback.selected(~F32,Blocks.writes(remaining,channel,width,offset,spatial,span,input),query,previous) == previous : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +next = (channel + 1 : U32)
      +next_exact = Planes.channel_successor(channel,rest,capacity,capacity_ok,channels_room)
      +start = (channel * spatial + offset : U32)
      +start_exact = destination_exact(channel,rest,offset,width,spatial,capacity,capacity_ok,clipped,slots_room)
      +here = Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),start)
      +later = Blocks.writes(rest,next,width,offset,spatial,span,input)
      %Equal.sym(F32,Readback.selected(~F32,List.append(&2,Writes.Assignment<F32>,here,later),query,previous),Readback.selected(~F32,later,query,Readback.selected(~F32,here,query,previous)),Readback.selected_append(~F32,here,later,query,previous)) : {_ == previous : F32}
      %Equal.sym(F32,Readback.selected(~F32,here,query,previous),previous,
        Copy.selected_outside(~F32,U32.to_nat(width),0n,query,input,(channel * span : U32),start,capacity,capacity_ok,
          %Equal.sym(Nat,U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {Nat.is_le(Nat.add(Nat.add(_,0n),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
          %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),0n),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),Addition.zero_right(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)))) : {Nat.is_le(Nat.add(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
          plane_room(channel,rest,offset,width,spatial,capacity,clipped,slots_room),
          Interval.Before{%Equal.sym(Nat,U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {Nat.is_lt(U32.to_nat(query),Nat.add(_,0n)) == True{} : Bool}
            %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),0n),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),Addition.zero_right(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)))) : {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool}
            Bounds.nat_trans(U32.to_nat(query),Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),below,Order.le_add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)))},previous)) : {Readback.selected(~F32,later,query,_) == previous : F32}
      later_selected(rest,next,width,offset,spatial,span,input,query,previous,capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),rest) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} channels_room,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),rest) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots_room,clipped,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_lt(U32.to_nat(query),Nat.mul(_,U32.to_nat(spatial))) == True{} : Bool}
        Bounds.nat_trans(U32.to_nat(query),Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.mul(1n+U32.to_nat(channel),U32.to_nat(spatial)),below,RowMajor.mul_le(U32.to_nat(channel),1n+U32.to_nat(channel),U32.to_nat(spatial),Order.successor(U32.to_nat(channel),U32.to_nat(channel),Order.reflexive(U32.to_nat(channel))))))

def selected_target(leading: Nat,+trailing: Nat,+channel: U32,+target: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+query: U32,+column: Nat,+previous: F32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +channels_room: {Nat.is_le(Nat.add(U32.to_nat(channel),Nat.add(leading,1n+trailing)),U32.to_nat(capacity)) == True{} : Bool},
  +slots_room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),Nat.add(leading,1n+trailing)),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool},
  +target_exact: {U32.to_nat(target) == Nat.add(U32.to_nat(channel),leading) : Nat},
  +query_exact: {U32.to_nat(query) == Nat.add(Nat.add(Nat.mul(U32.to_nat(target),U32.to_nat(spatial)),U32.to_nat(offset)),column) : Nat},
  +column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {Readback.selected(~F32,Blocks.writes(Nat.add(leading,1n+trailing),channel,width,offset,spatial,span,input),query,previous) == Tree.read(F32,input,(target * span + U32.from_nat(column) : U32)) : F32}:
  match leading:
    case 0n:
      +same = Arithmetic.to_nat_injective(target,channel,Equal.trans(Nat,U32.to_nat(target),Nat.add(U32.to_nat(channel),0n),U32.to_nat(channel),target_exact,Addition.zero_right(U32.to_nat(channel))))
      +start = (channel * spatial + offset : U32)
      +start_exact = destination_exact(channel,trailing,offset,width,spatial,capacity,capacity_ok,clipped,slots_room)
      +copy_room = {
        %Equal.sym(Nat,U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {Nat.is_le(Nat.add(Nat.add(_,0n),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),0n),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),Addition.zero_right(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)))) : {Nat.is_le(Nat.add(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
        plane_room(channel,trailing,offset,width,spatial,capacity,clipped,slots_room) : {Nat.is_le(Nat.add(Nat.add(U32.to_nat(start),0n),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}}
      +query_room = Bounds.strict_le(Nat.add(U32.to_nat(start),column),U32.to_nat(capacity),Bounds.nat_trans(Nat.add(U32.to_nat(start),column),Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(start),column,U32.to_nat(width),column_inside),
        %Addition.zero_right(U32.to_nat(start)) : {Nat.is_le(Nat.add(_,U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool} copy_room))
      +query_equal = Arithmetic.to_nat_injective(query,(start + U32.from_nat(column) : U32),Equal.trans(Nat,U32.to_nat(query),Nat.add(U32.to_nat(start),column),U32.to_nat((start + U32.from_nat(column) : U32)),
        %Equal.sym(Nat,U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {U32.to_nat(query) == Nat.add(_,column) : Nat}
        %same : {U32.to_nat(query) == Nat.add(Nat.add(Nat.mul(U32.to_nat(_),U32.to_nat(spatial)),U32.to_nat(offset)),column) : Nat} query_exact,
        Equal.sym(Nat,U32.to_nat((start + U32.from_nat(column) : U32)),Nat.add(U32.to_nat(start),column),Copy.offset_exact(start,column,capacity,capacity_ok,query_room))))
      +next = (channel + 1 : U32)
      +next_exact = Planes.channel_successor(channel,trailing,capacity,capacity_ok,channels_room)
      +here = Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),start)
      +later = Blocks.writes(trailing,next,width,offset,spatial,span,input)
      %Equal.sym(U32,target,channel,same) : {Readback.selected(~F32,List.append(&2,Writes.Assignment<F32>,here,later),query,previous) == Tree.read(F32,input,(_ * span + U32.from_nat(column) : U32)) : F32}
      %Equal.sym(F32,Readback.selected(~F32,List.append(&2,Writes.Assignment<F32>,here,later),query,previous),Readback.selected(~F32,later,query,Readback.selected(~F32,here,query,previous)),Readback.selected_append(~F32,here,later,query,previous)) : {_ == Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)) : F32}
      %Equal.sym(F32,Readback.selected(~F32,here,query,previous),Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)),
        %Equal.sym(U32,query,(start + U32.from_nat(column) : U32),query_equal) : {Readback.selected(~F32,here,_,previous) == Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)) : F32}
        Copy.selected_inside(~F32,U32.to_nat(width),0n,column,input,(channel * span : U32),start,capacity,capacity_ok,copy_room,column_inside,previous)) : {Readback.selected(~F32,later,query,_) == Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)) : F32}
      later_selected(trailing,next,width,offset,spatial,span,input,query,Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)),capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.add(_,trailing),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),trailing) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} channels_room,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,trailing),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),trailing) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots_room,clipped,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_lt(U32.to_nat(query),Nat.mul(_,U32.to_nat(spatial))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(query),Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column),%same : {U32.to_nat(query) == Nat.add(Nat.add(Nat.mul(U32.to_nat(_),U32.to_nat(spatial)),U32.to_nat(offset)),column) : Nat} query_exact) : {Nat.is_lt(_,Nat.mul(1n+U32.to_nat(channel),U32.to_nat(spatial))) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),column)),Addition.associative(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset),column)) : {Nat.is_lt(_,Nat.mul(1n+U32.to_nat(channel),U32.to_nat(spatial))) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(U32.to_nat(spatial),Nat.mul(U32.to_nat(channel),U32.to_nat(spatial))),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(spatial)),Order.commutative(U32.to_nat(spatial),Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)))) : {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),column)),_) == True{} : Bool}
        Order.offset_lt(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),Nat.add(U32.to_nat(offset),column),U32.to_nat(spatial),Bounds.nat_trans(Nat.add(U32.to_nat(offset),column),Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial),Order.offset_lt(U32.to_nat(offset),column,U32.to_nat(width),column_inside),clipped)))
    case 1n+ +rest:
      +next = (channel + 1 : U32)
      +next_exact = Planes.channel_successor(channel,Nat.add(rest,1n+trailing),capacity,capacity_ok,channels_room)
      +here = Contiguous.writes(~F32,U32.to_nat(width),input,0,(channel * span : U32),(channel * spatial + offset : U32))
      +later = Blocks.writes(Nat.add(rest,1n+trailing),next,width,offset,spatial,span,input)
      %Equal.sym(F32,Readback.selected(~F32,List.append(&2,Writes.Assignment<F32>,here,later),query,previous),Readback.selected(~F32,later,query,Readback.selected(~F32,here,query,previous)),Readback.selected_append(~F32,here,later,query,previous)) : {_ == Tree.read(F32,input,(target * span + U32.from_nat(column) : U32)) : F32}
      selected_target(rest,trailing,next,target,width,offset,spatial,span,input,query,column,Readback.selected(~F32,here,query,previous),capacity,capacity_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.add(_,Nat.add(rest,1n+trailing)),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),Nat.add(rest,1n+trailing)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} channels_room,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,Nat.add(rest,1n+trailing)),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(U32.to_nat(channel),Nat.add(rest,1n+trailing)) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots_room,clipped,
        %Equal.sym(Nat,U32.to_nat(next),1n+U32.to_nat(channel),next_exact) : {U32.to_nat(target) == Nat.add(_,rest) : Nat}
        %Addition.successor_right(U32.to_nat(channel),rest) : {U32.to_nat(target) == _ : Nat} target_exact,query_exact,column_inside)

def target_room(+rows: U32,+channel: U32,+offset: U32,+width: U32,+spatial: U32,+capacity: U32,
  +slots: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(width)),U32.to_nat(spatial)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(channel),U32.to_nat(rows)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}:
  +tail = Nat.sub(U32.to_nat(rows),1n+U32.to_nat(channel))
  plane_room(channel,tail,offset,width,spatial,capacity,clipped,
    %Equal.sym(Nat,Nat.add(U32.to_nat(channel),1n+tail),U32.to_nat(rows),Order.split_at_element(U32.to_nat(rows),U32.to_nat(channel),inside)) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots)

def target(+rows: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<F32>,+output: Tree.Tree<F32>,+depth: Nat,+channel: U32,+column: Nat,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(Tree.capacity(F32,output),2147483648) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +slots_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(spatial)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +clipped: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(Gather.gather_width(offset,spatial,span))),U32.to_nat(spatial)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(U32.to_nat(channel),U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Gather.gather_width(offset,spatial,span))) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,Blocks.block_writes(rows,offset,spatial,span,input),output),(channel * spatial + offset + U32.from_nat(column) : U32)) == Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)) : F32}:
  +capacity = Tree.capacity(F32,output)
  +width = Gather.gather_width(offset,spatial,span)
  +query = (channel * spatial + offset + U32.from_nat(column) : U32)
  +start = (channel * spatial + offset : U32)
  +tail = Nat.sub(U32.to_nat(rows),1n+U32.to_nat(channel))
  +split = Order.split_at_element(U32.to_nat(rows),U32.to_nat(channel),channel_inside)
  +row_slots = {%Equal.sym(Nat,Nat.add(U32.to_nat(channel),1n+tail),U32.to_nat(rows),split) : {Nat.is_le(Nat.mul(_,U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool} slots_room : {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),1n+tail),U32.to_nat(spatial)),U32.to_nat(capacity)) == True{} : Bool}}
  +start_exact = destination_exact(channel,tail,offset,width,spatial,capacity,capacity_ok,clipped,row_slots)
  +room = target_room(rows,channel,offset,width,spatial,capacity,slots_room,clipped,channel_inside)
  +query_inside = Order.le_lt_trans(0n,column,U32.to_nat(width),Order.zero_le(column),column_inside)
  +query_bound = Bounds.nat_trans(Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column),Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),U32.to_nat(width)),U32.to_nat(capacity),Order.offset_lt(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column,U32.to_nat(width),column_inside),room)
  +exact = {
    %start_exact : {U32.to_nat(query) == Nat.add(_,column) : Nat}
    Copy.offset_exact(start,column,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(start),Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),start_exact) : {Nat.is_le(Nat.add(_,column),U32.to_nat(capacity)) == True{} : Bool} Bounds.strict_le(Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column),U32.to_nat(capacity),query_bound)) : {U32.to_nat(query) == Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column) : Nat}}
  Equal.trans(F32,Tree.read(F32,Writes.pure(F32,Blocks.block_writes(rows,offset,spatial,span,input),output),query),Readback.selected(~F32,Blocks.block_writes(rows,offset,spatial,span,input),query,Tree.read(F32,output,query)),Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)),
    Readback.program(~F32,Blocks.block_writes(rows,offset,spatial,span,input),depth,output,query,balanced,depth_ok,
      %Equal.sym(Bool,U32.is_lt(query,capacity),Nat.is_lt(U32.to_nat(query),U32.to_nat(capacity)),Comparison.lt(query,capacity)) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(query),Nat.add(Nat.add(Nat.mul(U32.to_nat(channel),U32.to_nat(spatial)),U32.to_nat(offset)),column),exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool} query_bound,
      stores_inside(U32.to_nat(rows),0,width,offset,spatial,span,input,capacity,capacity_ok,rows_room,slots_room,clipped)),
    %split : {Readback.selected(~F32,Blocks.writes(_,0,width,offset,spatial,span,input),query,Tree.read(F32,output,query)) == Tree.read(F32,input,(channel * span + U32.from_nat(column) : U32)) : F32}
    selected_target(U32.to_nat(channel),tail,0,channel,width,offset,spatial,span,input,query,column,Tree.read(F32,output,query),capacity,capacity_ok,
      %Equal.sym(Nat,Nat.add(U32.to_nat(channel),1n+tail),U32.to_nat(rows),split) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} rows_room,row_slots,clipped,{==},exact,column_inside))

