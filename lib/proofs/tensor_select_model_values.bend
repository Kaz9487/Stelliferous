# Values of the ternary model use the shared indexed map readback contracts.
import Base
import ../math_compare.bend as Compare
import ./tensor_select_lanes_refinement.bend as Lanes
import ./tensor_lanes_refinement.bend as Binary
import ./math_select_values.bend as SelectMath
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./traversal_map_readback.bend as Map
import ./traversal_batch_refinement.bend as Batch
import ./traversal_lanes_refinement.bend as Rounds
import ./word_power_of_two.bend as Power
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_interval.bend as Interval
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

def active(+count: Nat,+start: Nat,+query: Nat) -> Bool:
  Nat.is_le(start,query) && Nat.is_lt(query,Nat.add(start,count))

def chosen(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat) -> F32:
  Bool.pick(F32,active(count,start,query),Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no)),Buffer.value(~F32,U32.from_nat(query),yes))

def capacity_bound(+content: Buffer.Content<F32>,ready: Buffer.ContentReady<F32,content>) -> {U32.is_le(Tree.capacity(F32,Buffer.content_tree(~F32,content)),2147483648) == True{} : Bool}:
  Buffer.Content{+tree,length} = content
  Buffer.ContentReady{+depth,+balanced,depth_ok,room} = ready
  %Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,depth),Shape.capacity(F32,depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok))

def mapped_length(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>) ->
  {Buffer.content_length(~F32,Lanes.mapped(~operation,count,start,left,right)) == Buffer.content_length(~F32,left) : Nat}:
  Buffer.Content{tree,length} = left
  Buffer.Content{source,source_length} = right
  {==}

def mapped_ready(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>,ready: Buffer.ContentReady<F32,left>) -> Buffer.ContentReady<F32,Lanes.mapped(~operation,count,start,left,right)>:
  Buffer.Content{+tree,+length} = left
  Buffer.Content{+source,source_length} = right
  Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
  Buffer.ContentReady{depth,Map.balanced_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),source,tree,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(F32,Map.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),source,tree)),Tree.capacity(F32,tree),Map.capacity_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),source,tree)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

def not_lower(+start: Nat,+query: Nat,h: {Nat.is_le(start,query) == False{} : Bool}) -> {Nat.is_lt(query,start) == True{} : Bool}:
  Order.ge_false(query,start,
    %Equal.sym(Bool,Nat.is_ge(query,start),Nat.is_le(start,query),Order.ge_le(query,start)) : {_ == False{} : Bool} h)

# The readback of a binary pass at a native query, selected by its interval.
def mapped_inside(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>,+query: Nat,
  +ready: Buffer.ContentReady<F32,left>,+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,left)) == True{} : Bool},
  +lower: {Nat.is_le(start,query) == True{} : Bool},+upper: {Nat.is_lt(query,Nat.add(start,count)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.mapped(~operation,count,start,left,right)) == operation(Buffer.value(~F32,U32.from_nat(query),left),Buffer.value(~F32,U32.from_nat(query),right)) : F32}:
  Buffer.Content{+tree,+length} = left
  Buffer.Content{+source,source_length} = right
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +capacity = Tree.capacity(F32,tree)
  +index = Nat.sub(query,start)
  +index_inside = Order.cancel_offset_lt(start,index,count,
    %Equal.sym(Nat,Nat.add(start,index),query,Order.restore_shift(start,query,lower)) : {Nat.is_lt(_,Nat.add(start,count)) == True{} : Bool} upper)
  %Order.restore_shift(start,query,lower) :
    {Tree.read(F32,Map.run_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,U32.from_nat(start),source,tree),U32.from_nat(_)) == operation(Tree.read(F32,tree,U32.from_nat(_)),Tree.read(F32,source,U32.from_nat(_))) : F32}
  Map.inside_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,start,index,source,tree,depth,capacity,{==},balanced,Order.successor(depth,30n,depth_ok),capacity_bound(left,ready),Order.transitive(Nat.add(start,count),length,U32.to_nat(capacity),span,room),index_inside)

def mapped_outside(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>,+query: Nat,
  +ready: Buffer.ContentReady<F32,left>,+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,left)) == True{} : Bool},
  +inside: {Nat.is_lt(query,Buffer.content_length(~F32,left)) == True{} : Bool},outside: Interval.Outside<start,count,query>) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.mapped(~operation,count,start,left,right)) == Buffer.value(~F32,U32.from_nat(query),left) : F32}:
  Buffer.Content{+tree,+length} = left
  Buffer.Content{+source,source_length} = right
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +capacity = Tree.capacity(F32,tree)
  +query_inside = Bounds.nat_trans(query,length,U32.to_nat(capacity),inside,room)
  Map.outside_indexed(~F32,~Tree.Tree<F32>,~Binary.pair_value(~operation),count,start,U32.from_nat(query),source,tree,depth,capacity,{==},balanced,Order.successor(depth,30n,depth_ok),capacity_bound(left,ready),Order.transitive(Nat.add(start,count),length,U32.to_nat(capacity),span,room),Map.address_inside(query,capacity,capacity_bound(left,ready),query_inside),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(query)),query,Bounds.roundtrip(query,capacity,capacity_bound(left,ready),Bounds.strict_le(query,U32.to_nat(capacity),query_inside))) : Interval.Outside<start,count,_> outside)

def mapped_point_when(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>,+query: Nat,
  +ready: Buffer.ContentReady<F32,left>,+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,left)) == True{} : Bool},
  +inside: {Nat.is_lt(query,Buffer.content_length(~F32,left)) == True{} : Bool},+lower: Bool,+upper: Bool,
  lower_equal: {Nat.is_le(start,query) == lower : Bool},upper_equal: {Nat.is_lt(query,Nat.add(start,count)) == upper : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.mapped(~operation,count,start,left,right)) == Bool.pick(F32,lower && upper,operation(Buffer.value(~F32,U32.from_nat(query),left),Buffer.value(~F32,U32.from_nat(query),right)),Buffer.value(~F32,U32.from_nat(query),left)) : F32}:
  match lower upper:
    case True{} True{}: mapped_inside(~operation,count,start,left,right,query,ready,span,lower_equal,upper_equal)
    case True{} False{}: mapped_outside(~operation,count,start,left,right,query,ready,span,inside,Interval.After{Order.not_less(query,Nat.add(start,count),upper_equal)})
    case False{} upper: mapped_outside(~operation,count,start,left,right,query,ready,span,inside,Interval.Before{not_lower(start,query,lower_equal)})

def mapped_point(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+left: Buffer.Content<F32>,+right: Buffer.Content<F32>,+query: Nat,
  +ready: Buffer.ContentReady<F32,left>,+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,left)) == True{} : Bool},
  +inside: {Nat.is_lt(query,Buffer.content_length(~F32,left)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.mapped(~operation,count,start,left,right)) == Bool.pick(F32,active(count,start,query),operation(Buffer.value(~F32,U32.from_nat(query),left),Buffer.value(~F32,U32.from_nat(query),right)),Buffer.value(~F32,U32.from_nat(query),left)) : F32}:
  mapped_point_when(~operation,count,start,left,right,query,ready,span,inside,Nat.is_le(start,query),Nat.is_lt(query,Nat.add(start,count)),{==},{==})

def tail_ready(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,ready: Buffer.ContentReady<F32,yes>) -> Buffer.ContentReady<F32,Lanes.tail_model(count,start,mask,yes,no)>:
  mapped_ready(~Compare.union,count,start,Lanes.mapped(~Compare.selected,count,start,yes,mask),Lanes.mapped(~Compare.rejected,count,start,no,mask),mapped_ready(~Compare.selected,count,start,yes,mask,ready))

def tail_length(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) ->
  {Buffer.content_length(~F32,Lanes.tail_model(count,start,mask,yes,no)) == Buffer.content_length(~F32,yes) : Nat}:
  Equal.trans(Nat,Buffer.content_length(~F32,Lanes.tail_model(count,start,mask,yes,no)),Buffer.content_length(~F32,Lanes.mapped(~Compare.selected,count,start,yes,mask)),Buffer.content_length(~F32,yes),mapped_length(~Compare.union,count,start,Lanes.mapped(~Compare.selected,count,start,yes,mask),Lanes.mapped(~Compare.rejected,count,start,no,mask)),mapped_length(~Compare.selected,count,start,yes,mask))

def joined_point(+enabled: Bool,+mask: F32,+yes: F32,+no: F32,+selected: F32,+rejected: F32,
  selected_equal: {selected == Bool.pick(F32,enabled,Compare.selected(yes,mask),yes) : F32},rejected_equal: {rejected == Bool.pick(F32,enabled,Compare.rejected(no,mask),no) : F32}) ->
  {Bool.pick(F32,enabled,Compare.union(selected,rejected),selected) == Bool.pick(F32,enabled,Compare.select(mask,yes,no),yes) : F32}:
  match enabled:
    case False{}: selected_equal
    case True{}:
      %Equal.sym(F32,selected,Compare.selected(yes,mask),selected_equal) : {Compare.union(_,rejected) == Compare.select(mask,yes,no) : F32}
      %Equal.sym(F32,rejected,Compare.rejected(no,mask),rejected_equal) : {Compare.union(Compare.selected(yes,mask),_) == Compare.select(mask,yes,no) : F32}
      SelectMath.tail_join(mask,yes,no)

def tail_point(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,
  +same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.tail_model(count,start,mask,yes,no)) == chosen(count,start,mask,yes,no,query) : F32}:
  +selected = Lanes.mapped(~Compare.selected,count,start,yes,mask)
  +rejected = Lanes.mapped(~Compare.rejected,count,start,no,mask)
  +selected_ready = mapped_ready(~Compare.selected,count,start,yes,mask,yes_ready)
  +selected_span = {
    %Equal.sym(Nat,Buffer.content_length(~F32,selected),Buffer.content_length(~F32,yes),mapped_length(~Compare.selected,count,start,yes,mask)) : {Nat.is_le(Nat.add(start,count),_) == True{} : Bool} span : {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,selected)) == True{} : Bool}}
  +selected_inside = {
    %Equal.sym(Nat,Buffer.content_length(~F32,selected),Buffer.content_length(~F32,yes),mapped_length(~Compare.selected,count,start,yes,mask)) : {Nat.is_lt(query,_) == True{} : Bool} inside : {Nat.is_lt(query,Buffer.content_length(~F32,selected)) == True{} : Bool}}
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(query),Lanes.tail_model(count,start,mask,yes,no)),Bool.pick(F32,active(count,start,query),Compare.union(Buffer.value(~F32,U32.from_nat(query),selected),Buffer.value(~F32,U32.from_nat(query),rejected)),Buffer.value(~F32,U32.from_nat(query),selected)),chosen(count,start,mask,yes,no,query),
    mapped_point(~Compare.union,count,start,selected,rejected,query,selected_ready,selected_span,selected_inside),
    joined_point(active(count,start,query),Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no),Buffer.value(~F32,U32.from_nat(query),selected),Buffer.value(~F32,U32.from_nat(query),rejected),
      mapped_point(~Compare.selected,count,start,yes,mask,query,yes_ready,span,inside),
      mapped_point(~Compare.rejected,count,start,no,mask,query,no_ready,
        %Equal.sym(Nat,Buffer.content_length(~F32,no),Buffer.content_length(~F32,yes),same_length) : {Nat.is_le(Nat.add(start,count),_) == True{} : Bool} span,
        %Equal.sym(Nat,Buffer.content_length(~F32,no),Buffer.content_length(~F32,yes),same_length) : {Nat.is_lt(query,_) == True{} : Bool} inside)))

type Context is Data:
  Context{mask: Tree.Tree<F32>,no: Tree.Tree<F32>}

def value(context: Context,position: U32,yes: F32) -> F32:
  Context{mask,no} = context
  +position = position
  Compare.select(Tree.read(F32,mask,position),yes,Tree.read(F32,no,position))

def batch(+mask: Tree.Tree<F32>,+yes: Tree.Tree<F32>,+no: Tree.Tree<F32>,+position: U32) ->
  {Binary.written(yes,position,Lanes.cell(mask,yes,no,position)) == Batch.batch(~F32,~Context,~value,8n,position,Context{mask,no},yes,yes) : Tree.Tree<F32>}:
  {==}

def round_tree(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) -> Tree.Tree<F32>:
  Binary.written(Buffer.content_tree(~F32,yes),U32.from_nat(start),Lanes.cell(Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,yes),Buffer.content_tree(~F32,no),U32.from_nat(start)))

def round_content(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) -> Buffer.Content<F32>:
  Buffer.Content{round_tree(start,mask,yes,no),Buffer.content_length(~F32,yes)}

def round_map(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+ready: Buffer.ContentReady<F32,yes>,+span: {Nat.is_le(Nat.add(start,8n),Buffer.content_length(~F32,yes)) == True{} : Bool}) ->
  {round_tree(start,mask,yes,no) == Map.run_indexed(~F32,~Context,~value,8n,U32.from_nat(start),Context{Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,no)},Buffer.content_tree(~F32,yes)) : Tree.Tree<F32>}:
  Buffer.Content{+yt,+yl} = yes
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +mt = Buffer.content_tree(~F32,mask)
  +nt = Buffer.content_tree(~F32,no)
  Equal.trans(Tree.Tree<F32>,round_tree(start,mask,yes,no),Batch.batch(~F32,~Context,~value,8n,U32.from_nat(start),Context{mt,nt},yt,yt),Map.run_indexed(~F32,~Context,~value,8n,U32.from_nat(start),Context{mt,nt},yt),batch(mt,yt,nt,U32.from_nat(start)),Batch.mapped(~F32,~Context,~value,8n,start,Context{mt,nt},yt,depth,Tree.capacity(F32,yt),{==},balanced,Order.successor(depth,30n,depth_ok),capacity_bound(yes,ready),Order.transitive(Nat.add(start,8n),yl,U32.to_nat(Tree.capacity(F32,yt)),span,room)))

def round_ready(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+ready: Buffer.ContentReady<F32,yes>,+span: {Nat.is_le(Nat.add(start,8n),Buffer.content_length(~F32,yes)) == True{} : Bool}) -> Buffer.ContentReady<F32,round_content(start,mask,yes,no)>:
  Buffer.Content{+yt,+yl} = yes
  Buffer.ContentReady{+depth,balanced,depth_ok,room} = ready
  +context = {Context{Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,no)} : Context}
  +mapped = Map.run_indexed(~F32,~Context,~value,8n,U32.from_nat(start),context,yt)
  %Equal.sym(Tree.Tree<F32>,round_tree(start,mask,yes,no),mapped,round_map(start,mask,yes,no,ready,span)) : Buffer.ContentReady<F32,Buffer.Content{_,yl}>
  Buffer.ContentReady{depth,Map.balanced_indexed(~F32,~Context,~value,8n,U32.from_nat(start),context,yt,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(F32,mapped),Tree.capacity(F32,yt),Map.capacity_indexed(~F32,~Context,~value,8n,U32.from_nat(start),context,yt)) : {Nat.is_le(yl,U32.to_nat(_)) == True{} : Bool} room}

type Meta<-output: Buffer.Content<F32>,-input_length: Nat> is Data:
  Meta{length: {Buffer.content_length(~F32,output) == input_length : Nat},ready: Buffer.ContentReady<F32,output>}

def meta_property(count: Nat) -> Type:
  @+start: Nat -> @+mask: Buffer.Content<F32> -> @+yes: Buffer.Content<F32> -> @+no: Buffer.Content<F32> ->
  @+ready: Buffer.ContentReady<F32,yes> -> @+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,yes)) == True{} : Bool} -> Meta<Lanes.model(count,start,mask,yes,no),Buffer.content_length(~F32,yes)>

def few_meta(+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> meta_property(count):
  +small = small
  start => mask => yes => no => ready => span =>
    %Equal.sym(Buffer.Content<F32>,Lanes.model(count,start,mask,yes,no),Lanes.tail_model(count,start,mask,yes,no),Lanes.few_model(count,start,mask,yes,no,small)) : Meta<_,Buffer.content_length(~F32,yes)>
    Meta{tail_length(count,start,mask,yes,no),tail_ready(count,start,mask,yes,no,ready)}

def more_meta(+rest: Nat,previous: meta_property(rest)) -> meta_property(8n+rest):
  start => mask => yes => no => ready => span =>
    previous(8n+start,mask,round_content(start,mask,yes,no),no,round_ready(start,mask,yes,no,ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span)),Rounds.rest_room(start,rest,Buffer.content_length(~F32,yes),span))

def meta(count: Nat) -> meta_property(count):
  Rounds.by_rounds(~meta_property,~few_meta,~more_meta,count)

type Position<-start: Nat,-count: Nat,-query: Nat> is Data:
  Inside{index: Nat,equation: {Nat.add(start,index) == query : Nat},inside: {Nat.is_lt(index,count) == True{} : Bool}}
  Earlier{before: {Nat.is_lt(query,start) == True{} : Bool}}
  Later{after: {Nat.is_le(Nat.add(start,count),query) == True{} : Bool}}

def expected(+start: Nat,+count: Nat,+query: Nat,position: Position<start,count,query>,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>) -> F32:
  match position:
    case Inside{index,equation,inside}: Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no))
    case Earlier{before}: Buffer.value(~F32,U32.from_nat(query),yes)
    case Later{after}: Buffer.value(~F32,U32.from_nat(query),yes)

# The interval predicates are used only to state the reusable scalar readback.
def less_not_ge_comparison(+comparison: Cmp,less: {Cmp.is_lt(comparison) == True{} : Bool}) -> {Cmp.is_ge(comparison) == False{} : Bool}:
  match comparison:
    case LT{}: {==}
    case EQ{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(less))
    case GT{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(less))

# The interval predicates are used only to state the reusable scalar readback.
def earlier_active(+count: Nat,+start: Nat,+query: Nat,before: {Nat.is_lt(query,start) == True{} : Bool}) -> {active(count,start,query) == False{} : Bool}:
  %Equal.sym(Bool,Nat.is_le(start,query),Nat.is_ge(query,start),Equal.sym(Bool,Nat.is_ge(query,start),Nat.is_le(start,query),Order.ge_le(query,start))) : {_ && Nat.is_lt(query,Nat.add(start,count)) == False{} : Bool}
  %Equal.sym(Bool,Nat.is_ge(query,start),False{},less_not_ge_comparison(Nat.cmp(query,start),before)) : {_ && Nat.is_lt(query,Nat.add(start,count)) == False{} : Bool}
  {==}

def later_active(+count: Nat,+start: Nat,+query: Nat,after: {Nat.is_le(Nat.add(start,count),query) == True{} : Bool}) -> {active(count,start,query) == False{} : Bool}:
  %Equal.sym(Bool,Nat.is_lt(query,Nat.add(start,count)),False{},Order.not_below(Nat.add(start,count),query,after)) : {Nat.is_le(start,query) && _ == False{} : Bool}
  Boolean.and_false_right(Nat.is_le(start,query))

def inside_active(+count: Nat,+start: Nat,+index: Nat,+inside: {Nat.is_lt(index,count) == True{} : Bool}) -> {active(count,start,Nat.add(start,index)) == True{} : Bool}:
  %Equal.sym(Bool,Nat.is_le(start,Nat.add(start,index)),True{},Order.le_add(start,index)) : {_ && Nat.is_lt(Nat.add(start,index),Nat.add(start,count)) == True{} : Bool}
  Order.offset_lt(start,index,count,inside)

def expected_chosen(+count: Nat,+start: Nat,+query: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,position: Position<start,count,query>) ->
  {chosen(count,start,mask,yes,no,query) == expected(start,count,query,position,mask,yes,no) : F32}:
  match position:
    case Inside{+index,equation,+inside}:
      %equation : {chosen(count,start,mask,yes,no,_) == Compare.select(Buffer.value(~F32,U32.from_nat(_),mask),Buffer.value(~F32,U32.from_nat(_),yes),Buffer.value(~F32,U32.from_nat(_),no)) : F32}
      %Equal.sym(Bool,active(count,start,Nat.add(start,index)),True{},inside_active(count,start,index,inside)) : {Bool.pick(F32,_,Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes)) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)) : F32}
      {==}
    case Earlier{before}:
      %Equal.sym(Bool,active(count,start,query),False{},earlier_active(count,start,query,before)) : {Bool.pick(F32,_,Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no)),Buffer.value(~F32,U32.from_nat(query),yes)) == Buffer.value(~F32,U32.from_nat(query),yes) : F32}
      {==}
    case Later{after}:
      %Equal.sym(Bool,active(count,start,query),False{},later_active(count,start,query,after)) : {Bool.pick(F32,_,Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no)),Buffer.value(~F32,U32.from_nat(query),yes)) == Buffer.value(~F32,U32.from_nat(query),yes) : F32}
      {==}

def tail_position(+count: Nat,+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},+position: Position<start,count,query>) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.tail_model(count,start,mask,yes,no)) == expected(start,count,query,position,mask,yes,no) : F32}:
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(query),Lanes.tail_model(count,start,mask,yes,no)),chosen(count,start,mask,yes,no,query),expected(start,count,query,position,mask,yes,no),tail_point(count,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside),expected_chosen(count,start,query,mask,yes,no,position))

def round_inside(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+index: Nat,
  +ready: Buffer.ContentReady<F32,yes>,+span: {Nat.is_le(Nat.add(start,8n),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),round_content(start,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)) : F32}:
  Buffer.Content{+mt,+ml} = mask
  Buffer.Content{+yt,+yl} = yes
  Buffer.Content{+nt,+nl} = no
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +context = {Context{Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,no)} : Context}
  %Equal.sym(Tree.Tree<F32>,round_tree(start,mask,yes,no),Map.run_indexed(~F32,~Context,~value,8n,U32.from_nat(start),context,yt),round_map(start,mask,yes,no,ready,span)) :
    {Tree.read(F32,_,U32.from_nat(Nat.add(start,index))) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Tree.read(F32,yt,U32.from_nat(Nat.add(start,index))),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)) : F32}
  Map.inside_indexed(~F32,~Context,~value,8n,start,index,context,yt,depth,Tree.capacity(F32,yt),{==},balanced,Order.successor(depth,30n,depth_ok),capacity_bound(yes,ready),Order.transitive(Nat.add(start,8n),yl,U32.to_nat(Tree.capacity(F32,yt)),span,room),inside)

def round_outside(+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +ready: Buffer.ContentReady<F32,yes>,+span: {Nat.is_le(Nat.add(start,8n),Buffer.content_length(~F32,yes)) == True{} : Bool},
  +inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},outside: Interval.Outside<start,8n,query>) ->
  {Buffer.value(~F32,U32.from_nat(query),round_content(start,mask,yes,no)) == Buffer.value(~F32,U32.from_nat(query),yes) : F32}:
  Buffer.Content{+yt,+yl} = yes
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +context = {Context{Buffer.content_tree(~F32,mask),Buffer.content_tree(~F32,no)} : Context}
  +capacity = Tree.capacity(F32,yt)
  +query_inside = Bounds.nat_trans(query,yl,U32.to_nat(capacity),inside,room)
  %Equal.sym(Tree.Tree<F32>,round_tree(start,mask,yes,no),Map.run_indexed(~F32,~Context,~value,8n,U32.from_nat(start),context,yt),round_map(start,mask,yes,no,ready,span)) :
    {Tree.read(F32,_,U32.from_nat(query)) == Tree.read(F32,yt,U32.from_nat(query)) : F32}
  Map.outside_indexed(~F32,~Context,~value,8n,start,U32.from_nat(query),context,yt,depth,capacity,{==},balanced,Order.successor(depth,30n,depth_ok),capacity_bound(yes,ready),Order.transitive(Nat.add(start,8n),yl,U32.to_nat(capacity),span,room),Map.address_inside(query,capacity,capacity_bound(yes,ready),query_inside),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(query)),query,Bounds.roundtrip(query,capacity,capacity_bound(yes,ready),Bounds.strict_le(query,U32.to_nat(capacity),query_inside))) : Interval.Outside<start,8n,_> outside)

def point_property(count: Nat) -> Type:
  @+start: Nat -> @+mask: Buffer.Content<F32> -> @+yes: Buffer.Content<F32> -> @+no: Buffer.Content<F32> -> @+query: Nat ->
  @+yes_ready: Buffer.ContentReady<F32,yes> -> @+no_ready: Buffer.ContentReady<F32,no> -> @+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat} ->
  @+span: {Nat.is_le(Nat.add(start,count),Buffer.content_length(~F32,yes)) == True{} : Bool} -> @+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool} -> @+position: Position<start,count,query> ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.model(count,start,mask,yes,no)) == expected(start,count,query,position,mask,yes,no) : F32}

def few_point(+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> point_property(count):
  +small = small
  start => mask => yes => no => query => yes_ready => no_ready => same_length => span => inside => position =>
    %Equal.sym(Buffer.Content<F32>,Lanes.model(count,start,mask,yes,no),Lanes.tail_model(count,start,mask,yes,no),Lanes.few_model(count,start,mask,yes,no,small)) :
      {Buffer.value(~F32,U32.from_nat(query),_) == expected(start,count,query,position,mask,yes,no) : F32}
    tail_position(count,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,position)

def next_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},+position: Position<8n+start,rest,query>) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)) == expected(8n+start,rest,query,position,mask,round_content(start,mask,yes,no),no) : F32}:
  previous(8n+start,mask,round_content(start,mask,yes,no),no,query,round_ready(start,mask,yes,no,yes_ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span)),no_ready,same_length,Rounds.rest_room(start,rest,Buffer.content_length(~F32,yes),span),inside,position)

def before_advanced(+start: Nat,+query: Nat,before: {Nat.is_lt(query,start) == True{} : Bool}) -> {Nat.is_lt(query,8n+start) == True{} : Bool}:
  %Order.commutative(start,8n) : {Nat.is_lt(query,_) == True{} : Bool}
  Bounds.nat_trans(query,start,Nat.add(start,8n),before,Order.le_add(start,8n))

def before_round(+start: Nat,+index: Nat,inside: {Nat.is_lt(index,8n) == True{} : Bool}) -> {Nat.is_lt(Nat.add(start,index),8n+start) == True{} : Bool}:
  %Order.commutative(start,8n) : {Nat.is_lt(Nat.add(start,index),_) == True{} : Bool}
  Order.offset_lt(start,index,8n,inside)

def after_round_query(+start: Nat,+index: Nat) -> {Nat.is_le(Nat.add(start,8n),Nat.add(start,8n+index)) == True{} : Bool}:
  %Addition.associative(start,8n,index) : {Nat.is_le(Nat.add(start,8n),_) == True{} : Bool}
  Order.le_add(Nat.add(start,8n),index)

def shifted_query(+start: Nat,+index: Nat) -> {Nat.add(8n+start,index) == Nat.add(start,8n+index) : Nat}:
  %Order.commutative(start,8n) : {Nat.add(_,index) == Nat.add(start,8n+index) : Nat}
  Addition.associative(start,8n,index)

def first_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+index: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(start,index),Buffer.content_length(~F32,yes)) == True{} : Bool},+small: {Nat.is_lt(index,8n) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),Lanes.model(8n+rest,start,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)) : F32}:
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),Lanes.model(8n+rest,start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),round_content(start,mask,yes,no)),Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)),
    next_point(rest,previous,start,mask,yes,no,Nat.add(start,index),yes_ready,no_ready,same_length,span,inside,Earlier{before_round(start,index,small)}),
    round_inside(start,mask,yes,no,index,yes_ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span),small))

def rest_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+index: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(start,8n+index),Buffer.content_length(~F32,yes)) == True{} : Bool},+selected: {Nat.is_lt(index,rest) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(Nat.add(start,8n+index)),Lanes.model(8n+rest,start,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,8n+index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,8n+index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,8n+index)),no)) : F32}:
  +query = Nat.add(start,8n+index)
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)),Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),round_content(start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),no)),Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),Buffer.value(~F32,U32.from_nat(query),yes),Buffer.value(~F32,U32.from_nat(query),no)),
    next_point(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,Inside{index,shifted_query(start,index),selected}),
    Equal.cong(F32,F32,yes => Compare.select(Buffer.value(~F32,U32.from_nat(query),mask),yes,Buffer.value(~F32,U32.from_nat(query),no)),Buffer.value(~F32,U32.from_nat(query),round_content(start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),yes),round_outside(start,mask,yes,no,query,yes_ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span),inside,Interval.After{after_round_query(start,index)})))

def selected_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+index: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(start,index),Buffer.content_length(~F32,yes)) == True{} : Bool},selected: {Nat.is_lt(index,8n+rest) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),Lanes.model(8n+rest,start,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),mask),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),yes),Buffer.value(~F32,U32.from_nat(Nat.add(start,index)),no)) : F32}:
  match index:
    case 0n: first_point(rest,previous,start,mask,yes,no,0n,yes_ready,no_ready,same_length,span,inside,{==})
    case 1n: first_point(rest,previous,start,mask,yes,no,1n,yes_ready,no_ready,same_length,span,inside,{==})
    case 2n: first_point(rest,previous,start,mask,yes,no,2n,yes_ready,no_ready,same_length,span,inside,{==})
    case 3n: first_point(rest,previous,start,mask,yes,no,3n,yes_ready,no_ready,same_length,span,inside,{==})
    case 4n: first_point(rest,previous,start,mask,yes,no,4n,yes_ready,no_ready,same_length,span,inside,{==})
    case 5n: first_point(rest,previous,start,mask,yes,no,5n,yes_ready,no_ready,same_length,span,inside,{==})
    case 6n: first_point(rest,previous,start,mask,yes,no,6n,yes_ready,no_ready,same_length,span,inside,{==})
    case 7n: first_point(rest,previous,start,mask,yes,no,7n,yes_ready,no_ready,same_length,span,inside,{==})
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +index: rest_point(rest,previous,start,mask,yes,no,index,yes_ready,no_ready,same_length,span,inside,selected)

def earlier_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},+before: {Nat.is_lt(query,start) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)) == Buffer.value(~F32,U32.from_nat(query),yes) : F32}:
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),round_content(start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),yes),
    next_point(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,Earlier{before_advanced(start,query,before)}),
    round_outside(start,mask,yes,no,query,yes_ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span),inside,Interval.Before{before}))

def later_point(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},+after: {Nat.is_le(Nat.add(start,8n+rest),query) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)) == Buffer.value(~F32,U32.from_nat(query),yes) : F32}:
  +rest_after = {
    %Equal.sym(Nat,Nat.add(8n+start,rest),Nat.add(start,8n+rest),shifted_query(start,rest)) : {Nat.is_le(_,query) == True{} : Bool} after : {Nat.is_le(Nat.add(8n+start,rest),query) == True{} : Bool}}
  Equal.trans(F32,Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),round_content(start,mask,yes,no)),Buffer.value(~F32,U32.from_nat(query),yes),
    next_point(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,Later{rest_after}),
    round_outside(start,mask,yes,no,query,yes_ready,Rounds.round_room(start,rest,Buffer.content_length(~F32,yes),span),inside,Interval.After{Order.transitive(Nat.add(start,8n),Nat.add(start,8n+rest),query,Order.offset_le(start,8n,8n+rest,Order.le_add(8n,rest)),after)}))

def more_point_canonical(+rest: Nat,previous: point_property(rest),+start: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+query: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(Nat.add(start,8n+rest),Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(query,Buffer.content_length(~F32,yes)) == True{} : Bool},position: Position<start,8n+rest,query>) ->
  {Buffer.value(~F32,U32.from_nat(query),Lanes.model(8n+rest,start,mask,yes,no)) == expected(start,8n+rest,query,position,mask,yes,no) : F32}:
  match position:
    case Inside{+index,+equation,selected}:
      %equation : {Buffer.value(~F32,U32.from_nat(_),Lanes.model(8n+rest,start,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(_),mask),Buffer.value(~F32,U32.from_nat(_),yes),Buffer.value(~F32,U32.from_nat(_),no)) : F32}
      selected_point(rest,previous,start,mask,yes,no,index,yes_ready,no_ready,same_length,span,
        %Equal.sym(Nat,Nat.add(start,index),query,equation) : {Nat.is_lt(_,Buffer.content_length(~F32,yes)) == True{} : Bool} inside,selected)
    case Earlier{before}: earlier_point(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,before)
    case Later{after}: later_point(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,after)

def more_point(+rest: Nat,previous: point_property(rest)) -> point_property(8n+rest):
  start => mask => yes => no => query => yes_ready => no_ready => same_length => span => inside => position =>
    more_point_canonical(rest,previous,start,mask,yes,no,query,yes_ready,no_ready,same_length,span,inside,position)

def point(count: Nat) -> point_property(count):
  Rounds.by_rounds(~point_property,~few_point,~more_point,count)

def selected(+count: Nat,+mask: Buffer.Content<F32>,+yes: Buffer.Content<F32>,+no: Buffer.Content<F32>,+index: Nat,
  +yes_ready: Buffer.ContentReady<F32,yes>,+no_ready: Buffer.ContentReady<F32,no>,+same_length: {Buffer.content_length(~F32,no) == Buffer.content_length(~F32,yes) : Nat},
  +span: {Nat.is_le(count,Buffer.content_length(~F32,yes)) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Buffer.value(~F32,U32.from_nat(index),Lanes.model(count,0n,mask,yes,no)) == Compare.select(Buffer.value(~F32,U32.from_nat(index),mask),Buffer.value(~F32,U32.from_nat(index),yes),Buffer.value(~F32,U32.from_nat(index),no)) : F32}:
  point(count)(0n,mask,yes,no,index,yes_ready,no_ready,same_length,span,Bounds.nat_trans(index,count,Buffer.content_length(~F32,yes),inside,span),Inside{index,{==},inside})
