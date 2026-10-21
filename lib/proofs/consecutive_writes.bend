# Writes at consecutive addresses: element k of a part goes to base plus k.
# Contiguous views, indexed stores and copies write this list; its readback is
# proved here once.
import Base
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./list_storage_readback.bend as ListReadback
import ./traversal_write_bounds.bend as WriteBounds
import ./traversal_map_readback.bend as MapReadback

# The writes of count elements from position on.
def writes(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+base: Nat,+position: Nat,+part: Part) ->
  List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest: Writes.Put{U32.from_nat(Nat.add(base,position)),value(part,position)} <> writes(~Element,~Part,~value,rest,base,1n+position,part)

# The last write at a query below every remaining address leaves the
# previous element.
def before(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+base: Nat,+position: Nat,+part: Part,
  +previous: Element,+query: Nat,+length: U32,
  +bound: {Nat.is_le(Nat.add(base,Nat.add(position,count)),U32.to_nat(length)) == True{} : Bool},
  +below: {Nat.is_lt(query,Nat.add(base,position)) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,base,position,part),U32.from_nat(query),previous) == previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +first_bound = Order.room_index(Nat.add(base,position),rest,U32.to_nat(length),
        %Equal.sym(Nat,Nat.add(Nat.add(base,position),1n+rest),Nat.add(base,Nat.add(position,1n+rest)),Addition.associative(base,position,1n+rest)) :
          {Nat.is_le(_,U32.to_nat(length)) == True{} : Bool}
        bound)
      +missed = ListReadback.distinct(Nat.add(base,position),query,length,first_bound,
        Bounds.nat_trans(query,Nat.add(base,position),U32.to_nat(length),below,Bounds.strict_le(Nat.add(base,position),U32.to_nat(length),first_bound)),below)
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(Nat.add(base,position)),U32.from_nat(query)),False{},missed) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,rest,base,1n+position,part),U32.from_nat(query),
          Bool.pick(Element,_,value(part,position),previous)) == previous : Element}
      before(~Element,~Part,~value,rest,base,1n+position,part,previous,query,length,
        %Addition.successor_right(position,rest) : {Nat.is_le(Nat.add(base,_),U32.to_nat(length)) == True{} : Bool} bound,
        Bounds.nat_trans(query,Nat.add(base,position),Nat.add(base,1n+position),below,Order.offset_le(base,position,1n+position,Order.successor(position,position,Order.reflexive(position)))))

# The last write at a query from every remaining address on leaves the
# previous element.
def after(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+base: Nat,+position: Nat,+part: Part,
  +previous: Element,+query: Nat,+length: U32,
  +beyond: {Nat.is_le(Nat.add(base,Nat.add(position,count)),query) == True{} : Bool},
  +query_inside: {Nat.is_lt(query,U32.to_nat(length)) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,base,position,part),U32.from_nat(query),previous) == previous : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +missed = WriteBounds.encoded_distinct(Nat.add(base,position),query,length,
        Bounds.nat_trans(Nat.add(base,position),Nat.add(base,Nat.add(position,1n+rest)),query,Order.offset_lt(base,position,Nat.add(position,1n+rest),Order.below_sum(position,rest)),beyond),
        Bounds.strict_le(query,U32.to_nat(length),query_inside))
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(Nat.add(base,position)),U32.from_nat(query)),False{},missed) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,rest,base,1n+position,part),U32.from_nat(query),
          Bool.pick(Element,_,value(part,position),previous)) == previous : Element}
      after(~Element,~Part,~value,rest,base,1n+position,part,previous,query,length,
        %Addition.successor_right(position,rest) : {Nat.is_le(Nat.add(base,_),query) == True{} : Bool} beyond,query_inside)

# The last write at the address of a remaining element stores that element:
# later elements lie at later addresses.
def hit(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+base: Nat,+position: Nat,+part: Part,
  +previous: Element,distance: Nat,+length: U32,
  +bound: {Nat.is_le(Nat.add(base,Nat.add(position,count)),U32.to_nat(length)) == True{} : Bool},
  +selected: {Nat.is_lt(distance,count) == True{} : Bool}) ->
  {Readback.selected(~Element,writes(~Element,~Part,~value,count,base,position,part),U32.from_nat(Nat.add(base,Nat.add(position,distance))),previous) ==
    value(part,Nat.add(position,distance)) : Element}:
  match count distance:
    case 0n distance:
      Empty.absurd({Readback.selected(~Element,writes(~Element,~Part,~value,0n,base,position,part),U32.from_nat(Nat.add(base,Nat.add(position,distance))),previous) ==
        value(part,Nat.add(position,distance)) : Element},Order.below_zero(distance,selected))
    case 1n+ +rest 0n:
      +start = Nat.add(base,position)
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,1n+rest,base,position,part),U32.from_nat(Nat.add(base,_)),previous) == value(part,_) : Element}
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(start),U32.from_nat(start)),True{},Arithmetic.equal_reflexive(U32.from_nat(start))) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,rest,base,1n+position,part),U32.from_nat(start),
          Bool.pick(Element,_,value(part,position),previous)) == value(part,position) : Element}
      before(~Element,~Part,~value,rest,base,1n+position,part,value(part,position),start,length,
        %Addition.successor_right(position,rest) : {Nat.is_le(Nat.add(base,_),U32.to_nat(length)) == True{} : Bool} bound,
        Order.offset_lt(base,position,1n+position,Order.lt_succ(position,position,Order.reflexive(position))))
    case 1n+ +rest 1n+ +more:
      +query = Nat.add(base,Nat.add(position,1n+more))
      +missed = WriteBounds.encoded_distinct(Nat.add(base,position),query,length,Order.offset_lt(base,position,Nat.add(position,1n+more),Order.below_sum(position,more)),
        Order.transitive(query,Nat.add(base,Nat.add(position,1n+rest)),U32.to_nat(length),
          Order.offset_le(base,Nat.add(position,1n+more),Nat.add(position,1n+rest),Order.offset_le(position,1n+more,1n+rest,Bounds.strict_le(1n+more,1n+rest,selected))),bound))
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(Nat.add(base,position)),U32.from_nat(query)),False{},missed) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,rest,base,1n+position,part),U32.from_nat(query),
          Bool.pick(Element,_,value(part,position),previous)) == value(part,Nat.add(position,1n+more)) : Element}
      %Equal.sym(Nat,Nat.add(position,1n+more),1n+Nat.add(position,more),Addition.successor_right(position,more)) :
        {Readback.selected(~Element,writes(~Element,~Part,~value,rest,base,1n+position,part),U32.from_nat(Nat.add(base,_)),previous) == value(part,_) : Element}
      hit(~Element,~Part,~value,rest,base,1n+position,part,previous,more,length,
        %Addition.successor_right(position,rest) : {Nat.is_le(Nat.add(base,_),U32.to_nat(length)) == True{} : Bool} bound,selected)

# Every write lies below a capacity that holds the whole run.
def inside(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,count: Nat,+base: Nat,+position: Nat,+part: Part,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(base,Nat.add(position,count)),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,writes(~Element,~Part,~value,count,base,position,part)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{
        MapReadback.address_inside(Nat.add(base,position),capacity,capacity_ok,Order.room_index(Nat.add(base,position),rest,U32.to_nat(capacity),
          %Equal.sym(Nat,Nat.add(Nat.add(base,position),1n+rest),Nat.add(base,Nat.add(position,1n+rest)),Addition.associative(base,position,1n+rest)) :
            {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
          room)),
        inside(~Element,~Part,~value,rest,base,1n+position,part,capacity,capacity_ok,
          %Addition.successor_right(position,rest) : {Nat.is_le(Nat.add(base,_),U32.to_nat(capacity)) == True{} : Bool} room)}

# After the run, the address base plus index holds element index.
def read(~Element: Data,~Part: Data,~value: @+part: Part -> @+position: Nat -> Element,+count: Nat,+base: Nat,+part: Part,+output: Tree.Tree<Element>,+depth: Nat,+index: Nat,
  +balanced: Shape.Balanced(Element,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(base,count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes(~Element,~Part,~value,count,base,0n,part),output),U32.from_nat(Nat.add(base,index))) == value(part,index) : Element}:
  +capacity = Tree.capacity(Element,output)
  +query = U32.from_nat(Nat.add(base,index))
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,writes(~Element,~Part,~value,count,base,0n,part),output),query),
    Readback.selected(~Element,writes(~Element,~Part,~value,count,base,0n,part),query,Tree.read(Element,output,query)),value(part,index),
    Readback.program(~Element,writes(~Element,~Part,~value,count,base,0n,part),depth,output,query,balanced,depth_ok,
      MapReadback.address_inside(Nat.add(base,index),capacity,capacity_ok,Bounds.nat_trans(Nat.add(base,index),Nat.add(base,count),U32.to_nat(capacity),Order.offset_lt(base,index,count,selected),room)),
      inside(~Element,~Part,~value,count,base,0n,part,capacity,capacity_ok,room)),
    hit(~Element,~Part,~value,count,base,0n,part,Tree.read(Element,output,query),index,capacity,room,selected))
