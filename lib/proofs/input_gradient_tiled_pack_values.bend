# Readback survives the complete concatenation of actual packed phases.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_weights.bend as Packing
import ./input_gradient_tiled_packing_bounds.bend as Bounds
import ./input_gradient_phase_domain.bend as Domain
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./indexed_output_readback.bend as Readback
import ./traversal_array_refinement.bend as Arrays
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./natural_range_sums.bend as Sums
import ./u32_division.bend as NativeDivision
import ./contiguous_copy_readback.bend as Addresses
import ./write_program.bend as Writes
import ./write_program_readback.bend as WriteReadback
import ./storage_allocation_refinement.bend as Allocation
import ../tensor.bend as Tensors
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ./tensor_allocation_depth.bend as Depth
import ./machine_limit_bounds.bend as Limits
import ./word_power_of_two.bend as Power
import ../tensor_view.bend as Views
import ./tensor_copy_values.bend as Values
import ./tensor_shape_refinement.bend as ShapeLaws
import ./consecutive_writes.bend as Consecutive

# A packed weight tile asked for: slot of phase, in packed weights of the
# given depth and capacity.
type Query is Data:
  Query{bounds: Bounds.Query,source: Buffer.Model<Tile.Tile8>,phase: Nat,slot: Nat,depth: Nat,capacity: U32}

def bounds(query: Query) -> Bounds.Query:
  Query{bounds,source,phase,slot,depth,capacity} = query
  bounds
def source(query: Query) -> Buffer.Model<Tile.Tile8>:
  Query{bounds,source,phase,slot,depth,capacity} = query
  source
def phase(query: Query) -> Nat:
  Query{bounds,source,phase,slot,depth,capacity} = query
  phase
def slot(query: Query) -> Nat:
  Query{bounds,source,phase,slot,depth,capacity} = query
  slot
def depth(query: Query) -> Nat:
  Query{bounds,source,phase,slot,depth,capacity} = query
  depth
def capacity(query: Query) -> U32:
  Query{bounds,source,phase,slot,depth,capacity} = query
  capacity

def count(query: Query) -> Nat:
  Bounds.count(bounds(query))
def geometry(query: Query) -> Kernel.Geometry:
  Bounds.geometry(bounds(query))
def selected(query: Query) -> Kernel.Phase:
  +query = query
  Kernel.phase_of(geometry(query),U32.from_nat(phase(query)))
def address(query: Query) -> U32:
  +query = query
  (Bounds.prefix_word(bounds(query),phase(query)) + U32.from_nat(slot(query)) : U32)
def weight_tree(state: Packing.Packed) -> Tree.Tree<Tile.Tile8>:
  Buffer.tree(~Tile.Tile8,Packing.weights(state))

def read(query: Query,state: Packing.Packed) -> Tile.Tile8:
  +query = query
  +state = state
  Tree.read(Tile.Tile8,weight_tree(state),address(query))
def wanted(query: Query) -> Tile.Tile8:
  +query = query
  Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,source(query)),Values.address(Kernel.weight_view(selected(query)),slot(query)))
def update(index: U32,query: Query,state: Packing.Packed) -> Packing.Packed:
  Packing.pack_phase(index,geometry(query),state)

type Facts<-query: Query> is Data:
  Facts{phase_inside: {Nat.is_lt(phase(query),count(query)) == True{} : Bool},
    slot_inside: {Nat.is_lt(slot(query),U32.to_nat(Packing.phase_count(selected(query)))) == True{} : Bool},
    capacity_ok: {U32.is_le(capacity(query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,count(query),0n,bounds(query)),U32.to_nat(capacity(query))) == True{} : Bool}}

type Frame<-query: Query,-index: Nat,-state: Packing.Packed> is Data:
  Frame{source_equal: {Packing.source(state) == source(query) : Buffer.Model<Tile.Tile8>},
    weight_base: {Packing.weight_base(state) == Bounds.prefix_word(bounds(query),index) : U32},
    capacity_equal: {Tree.capacity(Tile.Tile8,weight_tree(state)) == capacity(query) : U32},
    balanced: Shape.Balanced(Tile.Tile8,depth(query),weight_tree(state)),depth_ok: {Nat.is_le(depth(query),31n) == True{} : Bool},facts: Facts<query>}

def frame(query: Query,index: Nat,state: Packing.Packed) -> Data:
  Frame<query,index,state>

# A phase copy keeps the capacity and the balance of the packed weights.
type Shaped<-state: Packing.Packed,-depth: Nat,-capacity: U32> is Data:
  Shaped{capacity_equal: {Tree.capacity(Tile.Tile8,weight_tree(state)) == capacity : U32},balance: Shape.Balanced(Tile.Tile8,depth,weight_tree(state))}

def shape_selected(empty: Bool,+selected: Kernel.Phase,+state: Packing.Packed,+depth: Nat,+capacity: U32,
  evidence: Shaped<state,depth,capacity>) -> Shaped<Packing.pack_selected(empty,selected,state),depth,capacity>:
  match empty:
    case True{}: evidence
    case False{}:
      Packing.Packed{+source,+weights,+base} = state
      Shaped{+capacity_equal,+balance} = evidence
      +writes = Packing.copy_writes(selected,source,base)
      +tree = Buffer.tree(~Tile.Tile8,weights)
      +copied = Packing.copied_tree(selected,source,weights,base)
      Shaped{
        Equal.trans(U32,Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.copied(selected,source,weights,base))),Tree.capacity(Tile.Tile8,Writes.pure(Tile.Tile8,writes,tree)),capacity,
          Equal.cong(Tree.Tree<Tile.Tile8>,U32,Tree.capacity(Tile.Tile8),Buffer.tree(~Tile.Tile8,Packing.copied(selected,source,weights,base)),Writes.pure(Tile.Tile8,writes,tree),copied),
          Equal.trans(U32,Tree.capacity(Tile.Tile8,Writes.pure(Tile.Tile8,writes,tree)),Tree.capacity(Tile.Tile8,tree),capacity,
            Writes.writes_preserve_capacity(Tile.Tile8,writes,tree,depth,balance),capacity_equal)),
        %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Packing.copied(selected,source,weights,base)),Writes.pure(Tile.Tile8,writes,tree),copied) :
          Shape.Balanced(Tile.Tile8,depth,_)
          Writes.writes_preserve_balance(Tile.Tile8,writes,tree,depth,balance)}

def shape_phase(+index: U32,+geometry: Kernel.Geometry,+state: Packing.Packed,+depth: Nat,+capacity: U32,
  evidence: Shaped<state,depth,capacity>) -> Shaped<Packing.pack_phase(index,geometry,state),depth,capacity>:
  +selected = Kernel.phase_of(geometry,index)
  shape_selected(U32.is_eq(Kernel.phase_inner(selected),0),selected,state,depth,capacity,evidence)

def weight_next(+query: Query,+index: Nat,+state: Packing.Packed,
  before: {Packing.weight_base(state) == Bounds.prefix_word(bounds(query),index) : U32}) ->
  {Packing.weight_base(update(U32.from_nat(index),query,state)) == Bounds.prefix_word(bounds(query),1n+index) : U32}:
  +b = bounds(query)
  +g = geometry(query)
  +previous = Packing.weight_base(state)
  +base = Bounds.prefix_word(b,index)
  +step = Kernel.weight_total(U32.from_nat(index),g,base)
  Equal.trans(U32,Packing.weight_base(update(U32.from_nat(index),query,state)),Kernel.weight_total(U32.from_nat(index),g,previous),Bounds.prefix_word(b,1n+index),
    Packing.weight_step(U32.from_nat(index),g,state),
    Equal.trans(U32,Kernel.weight_total(U32.from_nat(index),g,previous),step,Bounds.prefix_word(b,1n+index),
      Equal.cong(U32,U32,base => Kernel.weight_total(U32.from_nat(index),g,base),previous,base,before),
      Equal.trans(U32,step,Bounds.update(U32.from_nat(index),b,base),Bounds.prefix_word(b,1n+index),
        Equal.sym(U32,Bounds.update(U32.from_nat(index),b,base),step,Bounds.weights_step(U32.from_nat(index),b,base)),
        Equal.sym(U32,Bounds.prefix_word(b,1n+index),Bounds.update(U32.from_nat(index),b,base),Bounds.prefix_step(b,index)))))

def framed_next(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,
  shaped: Shaped<update(U32.from_nat(index),query,state),depth(query),capacity(query)>) ->
  Frame<query,1n+index,update(U32.from_nat(index),query,state)>:
  Frame{source_equal,weight_base,capacity_equal,balanced,+depth_ok,+facts} = evidence
  Shaped{new_capacity,new_balance} = shaped
  Frame{Equal.trans(Buffer.Model<Tile.Tile8>,Packing.source(update(U32.from_nat(index),query,state)),Packing.source(state),source(query),Packing.phase_keeps_source(U32.from_nat(index),geometry(query),state),source_equal),
    weight_next(query,index,state,weight_base),new_capacity,new_balance,depth_ok,facts}

def frame_step(+query: Query,+index: Nat,+state: Packing.Packed,inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  evidence: Frame<query,index,state>) -> Frame<query,1n+index,update(U32.from_nat(index),query,state)>:
  Frame{+source_equal,+weight_base,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
  +evidence = {Frame{source_equal,weight_base,capacity_equal,balanced,depth_ok,facts} : Frame<query,index,state>}
  framed_next(query,index,state,evidence,shape_phase(U32.from_nat(index),geometry(query),state,depth(query),capacity(query),Shaped{capacity_equal,balanced}))

# The phases [index, index + 1) end where the next phase starts.
def native_range_room(+query: Bounds.Query,+index: Nat,+capacity: U32,+inside: {Nat.is_lt(index,Bounds.count(query)) == True{} : Bool},
  room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Bounds.prefix_word(query,index)),U32.to_nat(Bounds.native(query,U32.from_nat(index)))),U32.to_nat(capacity)) == True{} : Bool}:
  +prefix = Sums.sum(~Bounds.Query,~Bounds.pure,index,0n,query)
  +next = Sums.sum(~Bounds.Query,~Bounds.pure,1n+index,0n,query)
  +whole = Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query)
  %Equal.sym(Nat,U32.to_nat(Bounds.prefix_word(query,index)),prefix,Bounds.prefix_exact(query,index,Words.strict_le(index,Bounds.count(query),inside))) :
    {Nat.is_le(Nat.add(_,U32.to_nat(Bounds.native(query,U32.from_nat(index)))),U32.to_nat(capacity)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Bounds.native(query,U32.from_nat(index))),Bounds.pure(query,index),Bounds.native_exact(query,index,inside)) :
    {Nat.is_le(Nat.add(prefix,_),U32.to_nat(capacity)) == True{} : Bool}
  %Bounds.pure_prefix_step(query,index) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
  Order.transitive(next,whole,U32.to_nat(capacity),Sums.prefix_bounded(~Bounds.Query,~Bounds.pure,1n+index,Bounds.count(query),0n,query,Order.strict_follow(index,Bounds.count(query),inside)),room)

def copy_native(+query: Query,+index: Nat) ->
  {Bounds.native(bounds(query),U32.from_nat(index)) == Packing.phase_count(Kernel.phase_of(geometry(query),U32.from_nat(index))) : U32}:
  Query{b,source,phase,slot,depth,capacity} = query
  Bounds.Query{call} = b
  Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} = call
  {==}

def copy_room(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Packing.weight_base(state)),U32.to_nat(Packing.phase_count(Kernel.phase_of(geometry(query),U32.from_nat(index))))),U32.to_nat(capacity(query))) == True{} : Bool}:
  Frame{source_equal,weight_base,capacity_equal,balanced,depth_ok,facts} = evidence
  Facts{phase_inside,slot_inside,capacity_ok,room} = facts
  %Equal.sym(U32,Packing.weight_base(state),Bounds.prefix_word(bounds(query),index),weight_base) :
    {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(Packing.phase_count(Kernel.phase_of(geometry(query),U32.from_nat(index))))),U32.to_nat(capacity(query))) == True{} : Bool}
  %copy_native(query,index) :
    {Nat.is_le(Nat.add(U32.to_nat(Bounds.prefix_word(bounds(query),index)),U32.to_nat(_)),U32.to_nat(capacity(query))) == True{} : Bool}
  native_range_room(bounds(query),index,capacity(query),inside,room)

# The copy of a phase stores its weights from the weight base on.

def phase_line(+selected: Kernel.Phase,+source: Buffer.Model<Tile.Tile8>,+base: U32,+capacity: U32,
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(Packing.phase_count(selected))),U32.to_nat(capacity)) == True{} : Bool},
  exact: {ShapeLaws.elements(Views.axis_extents_of(Kernel.weight_view(selected))) == U32.to_nat(Packing.phase_count(selected)) : Nat}) ->
  {Packing.copy_writes(selected,source,base) == Consecutive.writes(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),
    U32.to_nat(Packing.phase_count(selected)),U32.to_nat(base),0n,Values.Sampled{Kernel.weight_view(selected),Buffer.tree(~Tile.Tile8,source)}) : List<&2,Writes.Assignment<Tile.Tile8>>}:
  +count = U32.to_nat(Packing.phase_count(selected))
  Values.copy_line_at(~Tile.Tile8,Views.is_contiguous(Kernel.weight_view(selected)),count,Kernel.weight_view(selected),base,Buffer.tree(~Tile.Tile8,source),capacity,{==},exact,capacity_ok,
    Order.transitive(count,Nat.add(U32.to_nat(base),count),U32.to_nat(capacity),
      %Order.commutative(count,U32.to_nat(base)) : {Nat.is_le(count,_) == True{} : Bool} Order.le_add(count,U32.to_nat(base)),room))
def copied_value(+selected: Kernel.Phase,+state: Packing.Packed,+slot: Nat,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,weight_tree(state)) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,weight_tree(state)),
  +depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Packing.weight_base(state)),U32.to_nat(Packing.phase_count(selected))),U32.to_nat(capacity)) == True{} : Bool},
  +slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(selected))) == True{} : Bool},
  +query_inside: {U32.is_lt((Packing.weight_base(state) + U32.from_nat(slot) : U32),capacity) == True{} : Bool},
  exact: {ShapeLaws.elements(Views.axis_extents_of(Kernel.weight_view(selected))) == U32.to_nat(Packing.phase_count(selected)) : Nat}) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected,state)),(Packing.weight_base(state) + U32.from_nat(slot) : U32)) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.source(state)),Values.address(Kernel.weight_view(selected),slot)) : Tile.Tile8}:
  Packing.Packed{+source,+weights,+base} = state
  +input = Buffer.tree(~Tile.Tile8,source)
  +count = U32.to_nat(Packing.phase_count(selected))
  +part = {Values.Sampled{Kernel.weight_view(selected),input} : Values.Sampled<Tile.Tile8>}
  +writes = Consecutive.writes(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part)
  +tree = Buffer.tree(~Tile.Tile8,weights)
  +query = (base + U32.from_nat(slot) : U32)
  +inside = {
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),capacity,capacity_equal) : WriteReadback.Inside(~Tile.Tile8,_,writes)
    Consecutive.inside(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part,capacity,capacity_ok,room) : WriteReadback.Inside(~Tile.Tile8,Tree.capacity(Tile.Tile8,tree),writes)}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Packing.copied(selected,source,weights,base)),Writes.pure(Tile.Tile8,Packing.copy_writes(selected,source,base),tree),
    Packing.copied_tree(selected,source,weights,base)) :
    {Tree.read(Tile.Tile8,_,query) == Values.sampled(~Tile.Tile8,part,slot) : Tile.Tile8}
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile8>>,Packing.copy_writes(selected,source,base),writes,phase_line(selected,source,base,capacity,capacity_ok,room,exact)) :
    {Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,_,tree),query) == Values.sampled(~Tile.Tile8,part,slot) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,writes,tree),query),WriteReadback.selected(~Tile.Tile8,writes,query,Tree.read(Tile.Tile8,tree,query)),
    Values.sampled(~Tile.Tile8,part,slot),
    WriteReadback.program_full(~Tile.Tile8,writes,depth,tree,query,balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),capacity,capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,inside),
    %Equal.sym(U32,query,U32.from_nat(Nat.add(U32.to_nat(base),slot)),Values.offset_address(base,slot)) :
      {WriteReadback.selected(~Tile.Tile8,writes,_,Tree.read(Tile.Tile8,tree,query)) == Values.sampled(~Tile.Tile8,part,slot) : Tile.Tile8}
    Consecutive.hit(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part,Tree.read(Tile.Tile8,tree,query),slot,capacity,room,slot_inside))

# A later phase copies above every address of an earlier one.
def copied_before(+selected: Kernel.Phase,+state: Packing.Packed,+address: U32,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,weight_tree(state)) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,weight_tree(state)),
  +depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Packing.weight_base(state)),U32.to_nat(Packing.phase_count(selected))),U32.to_nat(capacity)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Packing.weight_base(state))) == True{} : Bool},+inside: {U32.is_lt(address,capacity) == True{} : Bool},
  exact: {ShapeLaws.elements(Views.axis_extents_of(Kernel.weight_view(selected))) == U32.to_nat(Packing.phase_count(selected)) : Nat}) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected,state)),address) == Tree.read(Tile.Tile8,weight_tree(state),address) : Tile.Tile8}:
  Packing.Packed{+source,+weights,+base} = state
  +input = Buffer.tree(~Tile.Tile8,source)
  +count = U32.to_nat(Packing.phase_count(selected))
  +part = {Values.Sampled{Kernel.weight_view(selected),input} : Values.Sampled<Tile.Tile8>}
  +writes = Consecutive.writes(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part)
  +tree = Buffer.tree(~Tile.Tile8,weights)
  +writes_inside = {
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),capacity,capacity_equal) : WriteReadback.Inside(~Tile.Tile8,_,writes)
    Consecutive.inside(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part,capacity,capacity_ok,room) : WriteReadback.Inside(~Tile.Tile8,Tree.capacity(Tile.Tile8,tree),writes)}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Packing.copied(selected,source,weights,base)),Writes.pure(Tile.Tile8,Packing.copy_writes(selected,source,base),tree),
    Packing.copied_tree(selected,source,weights,base)) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,tree,address) : Tile.Tile8}
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile8>>,Packing.copy_writes(selected,source,base),writes,phase_line(selected,source,base,capacity,capacity_ok,room,exact)) :
    {Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,_,tree),address) == Tree.read(Tile.Tile8,tree,address) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Writes.pure(Tile.Tile8,writes,tree),address),WriteReadback.selected(~Tile.Tile8,writes,address,Tree.read(Tile.Tile8,tree,address)),
    Tree.read(Tile.Tile8,tree,address),
    WriteReadback.program_full(~Tile.Tile8,writes,depth,tree,address,balanced,depth_ok,
      %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),capacity,capacity_equal) : {U32.is_lt(address,_) == True{} : Bool} inside,writes_inside),
    %Arithmetic.native_roundtrip(address) : {WriteReadback.selected(~Tile.Tile8,writes,_,Tree.read(Tile.Tile8,tree,address)) == Tree.read(Tile.Tile8,tree,address) : Tile.Tile8}
    Consecutive.before(~Tile.Tile8,~Values.Sampled<Tile.Tile8>,~Values.sampled(~Tile.Tile8),count,U32.to_nat(base),0n,part,Tree.read(Tile.Tile8,tree,address),U32.to_nat(address),capacity,room,
      %Equal.sym(Nat,Nat.add(U32.to_nat(base),0n),U32.to_nat(base),Addition.zero_right(U32.to_nat(base))) : {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool} before))

def range_end_exact(+query: Bounds.Query,+index: Nat,+inside: {Nat.is_lt(index,Bounds.count(query)) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(Bounds.prefix_word(query,index)),U32.to_nat(Bounds.native(query,U32.from_nat(index)))) == Sums.sum(~Bounds.Query,~Bounds.pure,1n+index,0n,query) : Nat}:
  +base = U32.to_nat(Bounds.prefix_word(query,index))
  +span = U32.to_nat(Bounds.native(query,U32.from_nat(index)))
  +prefix = Sums.sum(~Bounds.Query,~Bounds.pure,index,0n,query)
  +sample = Bounds.pure(query,index)
  Equal.trans(Nat,Nat.add(base,span),Nat.add(prefix,span),Sums.sum(~Bounds.Query,~Bounds.pure,1n+index,0n,query),
    Equal.cong(Nat,Nat,value => Nat.add(value,span),base,prefix,Bounds.prefix_exact(query,index,Words.strict_le(index,Bounds.count(query),inside))),
    Equal.trans(Nat,Nat.add(prefix,span),Nat.add(prefix,sample),Sums.sum(~Bounds.Query,~Bounds.pure,1n+index,0n,query),
      Equal.cong(Nat,Nat,value => Nat.add(prefix,value),span,sample,Bounds.native_exact(query,index,inside)),
      Equal.sym(Nat,Sums.sum(~Bounds.Query,~Bounds.pure,1n+index,0n,query),Nat.add(prefix,sample),Bounds.pure_prefix_step(query,index))))

def selected_address_inside(+query: Bounds.Query,+selected: Nat,+slot: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +inside: {Nat.is_lt(selected,Bounds.count(query)) == True{} : Bool},+slot_inside: {Nat.is_lt(slot,U32.to_nat(Bounds.native(query,U32.from_nat(selected)))) == True{} : Bool},
  +room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt((Bounds.prefix_word(query,selected) + U32.from_nat(slot) : U32),capacity) == True{} : Bool}:
  +base = Bounds.prefix_word(query,selected)
  +span = U32.to_nat(Bounds.native(query,U32.from_nat(selected)))
  Addresses.offset_inside(base,slot,capacity,capacity_ok,Words.nat_trans(Nat.add(U32.to_nat(base),slot),Nat.add(U32.to_nat(base),span),U32.to_nat(capacity),
    Order.offset_lt(U32.to_nat(base),slot,span,slot_inside),native_range_room(query,selected,capacity,inside,room)))

def selected_address_before(+query: Bounds.Query,+selected: Nat,+slot: Nat,+index: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +selected_inside: {Nat.is_lt(selected,Bounds.count(query)) == True{} : Bool},+index_inside: {Nat.is_lt(index,Bounds.count(query)) == True{} : Bool},
  +slot_inside: {Nat.is_lt(slot,U32.to_nat(Bounds.native(query,U32.from_nat(selected)))) == True{} : Bool},
  +after: {Nat.is_lt(selected,index) == True{} : Bool},+room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((Bounds.prefix_word(query,selected) + U32.from_nat(slot) : U32)),U32.to_nat(Bounds.prefix_word(query,index))) == True{} : Bool}:
  +base = Bounds.prefix_word(query,selected)
  +span = U32.to_nat(Bounds.native(query,U32.from_nat(selected)))
  +next = Sums.sum(~Bounds.Query,~Bounds.pure,1n+selected,0n,query)
  +last = Sums.sum(~Bounds.Query,~Bounds.pure,index,0n,query)
  +position_inside = Words.nat_trans(Nat.add(U32.to_nat(base),slot),Nat.add(U32.to_nat(base),span),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(base),slot,span,slot_inside),native_range_room(query,selected,capacity,selected_inside,room))
  %Equal.sym(Nat,U32.to_nat((base + U32.from_nat(slot) : U32)),Nat.add(U32.to_nat(base),slot),Addresses.offset_exact(base,slot,capacity,capacity_ok,Words.strict_le(Nat.add(U32.to_nat(base),slot),U32.to_nat(capacity),position_inside))) :
    {Nat.is_lt(_,U32.to_nat(Bounds.prefix_word(query,index))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Bounds.prefix_word(query,index)),last,Bounds.prefix_exact(query,index,Words.strict_le(index,Bounds.count(query),index_inside))) :
    {Nat.is_lt(Nat.add(U32.to_nat(base),slot),_) == True{} : Bool}
  Words.nat_trans(Nat.add(U32.to_nat(base),slot),Nat.add(U32.to_nat(base),span),last,Order.offset_lt(U32.to_nat(base),slot,span,slot_inside),
    %Equal.sym(Nat,Nat.add(U32.to_nat(base),span),next,range_end_exact(query,selected,selected_inside)) : {Nat.is_le(_,last) == True{} : Bool}
      Sums.prefix_bounded(~Bounds.Query,~Bounds.pure,1n+selected,index,0n,query,Order.strict_follow(selected,index,after)))

def slot_native(+query: Query,+facts: Facts<query>) -> {Nat.is_lt(slot(query),U32.to_nat(Bounds.native(bounds(query),U32.from_nat(phase(query))))) == True{} : Bool}:
  Facts{phase_inside,slot_inside,capacity_ok,room} = facts
  %Equal.sym(U32,Bounds.native(bounds(query),U32.from_nat(phase(query))),Packing.phase_count(selected(query)),copy_native(query,phase(query))) : {Nat.is_lt(slot(query),U32.to_nat(_)) == True{} : Bool}
  slot_inside

def address_inside(+query: Query,+facts: Facts<query>) -> {U32.is_lt(address(query),capacity(query)) == True{} : Bool}:
  Facts{+phase_inside,slot_inside,+capacity_ok,+room} = facts
  selected_address_inside(bounds(query),phase(query),slot(query),capacity(query),capacity_ok,phase_inside,slot_native(query,facts),room)

def address_before(+query: Query,+facts: Facts<query>,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+after: {Nat.is_lt(phase(query),index) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(address(query)),U32.to_nat(Bounds.prefix_word(bounds(query),index))) == True{} : Bool}:
  Facts{+phase_inside,slot_inside,+capacity_ok,+room} = facts
  selected_address_before(bounds(query),phase(query),slot(query),index,capacity(query),capacity_ok,phase_inside,inside,slot_native(query,facts),after,room)

def selected_active(+query: Query,+facts: Facts<query>) -> {U32.is_eq(Kernel.phase_inner(selected(query)),0) == False{} : Bool}:
  Facts{phase_inside,slot_inside,capacity_ok,room} = facts
  Packing.count_active(U32.is_eq(Kernel.phase_inner(selected(query)),0),selected(query),slot(query),{==},slot_inside)

def produced_at_target(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  {read(query,update(U32.from_nat(phase(query)),query,state)) == wanted(query) : Tile.Tile8}:
  Frame{+source_equal,+weight_base,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
  Facts{+phase_inside,+slot_inside,+capacity_ok,room} = facts
  %Equal.sym(Bool,U32.is_eq(Kernel.phase_inner(selected(query)),0),False{},selected_active(query,facts)) :
    {read(query,Packing.pack_selected(_,selected(query),state)) == wanted(query) : Tile.Tile8}
  %weight_base :
    {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected(query),state)),(_ + U32.from_nat(slot(query)) : U32)) == wanted(query) : Tile.Tile8}
  %source_equal :
    {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected(query),state)),(Packing.weight_base(state) + U32.from_nat(slot(query)) : U32)) ==
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,_),Values.address(Kernel.weight_view(selected(query)),slot(query))) : Tile.Tile8}
  copied_value(selected(query),state,slot(query),depth(query),capacity(query),capacity_equal,balanced,depth_ok,capacity_ok,
    copy_room(query,phase(query),state,evidence,phase_inside),slot_inside,
    %Equal.sym(U32,Packing.weight_base(state),Bounds.prefix_word(bounds(query),phase(query)),weight_base) :
      {U32.is_lt((_ + U32.from_nat(slot(query)) : U32),capacity(query)) == True{} : Bool} address_inside(query,facts),
    %copy_native(query,phase(query)) : {ShapeLaws.elements(Views.axis_extents_of(Kernel.weight_view(selected(query)))) == U32.to_nat(_) : Nat}
    Bounds.view_exact(bounds(query),U32.from_nat(phase(query))))

def produced(+query: Query,+index: Nat,+state: Packing.Packed,evidence: Frame<query,index,state>,same: {index == phase(query) : Nat}) ->
  {read(query,update(U32.from_nat(index),query,state)) == wanted(query) : Tile.Tile8}:
  +evidence = evidence
  +same = same
  %Equal.sym(Nat,index,phase(query),same) : {read(query,update(U32.from_nat(_),query,state)) == wanted(query) : Tile.Tile8}
  produced_at_target(query,state,%same : Frame<query,_,state> evidence)

def later_selected(empty: Bool,+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  +inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {read(query,Packing.pack_selected(empty,Kernel.phase_of(geometry(query),U32.from_nat(index)),state)) == read(query,state) : Tile.Tile8}:
  match empty:
    case True{}: {==}
    case False{}:
      Frame{source_equal,+weight_base,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
      Facts{phase_inside,slot_inside,+capacity_ok,room} = facts
      +before = {
        %Equal.sym(U32,Packing.weight_base(state),Bounds.prefix_word(bounds(query),index),weight_base) :
          {Nat.is_lt(U32.to_nat(address(query)),U32.to_nat(_)) == True{} : Bool}
        address_before(query,facts,index,inside,after) : {Nat.is_lt(U32.to_nat(address(query)),U32.to_nat(Packing.weight_base(state))) == True{} : Bool}}
      copied_before(Kernel.phase_of(geometry(query),U32.from_nat(index)),state,address(query),depth(query),capacity(query),capacity_equal,balanced,depth_ok,capacity_ok,
        copy_room(query,index,state,evidence,inside),before,address_inside(query,facts),
        %copy_native(query,index) : {ShapeLaws.elements(Views.axis_extents_of(Kernel.weight_view(Kernel.phase_of(geometry(query),U32.from_nat(index))))) == U32.to_nat(_) : Nat}
        Bounds.view_exact(bounds(query),U32.from_nat(index)))

def later(+query: Query,+index: Nat,+state: Packing.Packed,evidence: Frame<query,index,state>,after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) -> {read(query,update(U32.from_nat(index),query,state)) == read(query,state) : Tile.Tile8}:
  Frame{+source_equal,+weight_base,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
  +evidence = {Frame{source_equal,weight_base,capacity_equal,balanced,depth_ok,facts} : Frame<query,index,state>}
  +after = after
  +inside = inside
  later_selected(U32.is_eq(Kernel.phase_inner(Kernel.phase_of(geometry(query),U32.from_nat(index))),0),query,index,state,evidence,after,inside)

def readback(+query: Query,+state: Packing.Packed,+evidence: Frame<query,0n,state>) ->
  {read(query,Traversal.indexed_fold(~Packing.Packed,~Query,~update,count(query),0,query,state)) == wanted(query) : Tile.Tile8}:
  Frame{source_equal,weight_base,capacity_equal,balanced,depth_ok,+facts} = evidence
  Facts{+phase_inside,slot_inside,capacity_ok,room} = facts
  Readback.indexed_complete(~Packing.Packed,~Query,~Tile.Tile8,~count,~phase,~wanted,~frame,~read,~update,~frame_step,~produced,~later,query,state,evidence,phase_inside)

def step_same(+index: U32,+query: Query,+state: Packing.Packed) ->
  {update(index,query,state) == Packing.pack_phase(index,geometry(query),state) : Packing.Packed}:
  {==}

def complete(query: Query,state: Packing.Packed) -> Packing.Packed:
  +query = query
  Traversal.indexed_fold(~Packing.Packed,~Kernel.Geometry,~Packing.pack_phase,count(query),0,geometry(query),state)

def complete_readback(+query: Query,+state: Packing.Packed,+evidence: Frame<query,0n,state>) ->
  {read(query,complete(query,state)) == wanted(query) : Tile.Tile8}:
  %Arrays.indexed_fold_change_context(~Packing.Packed,~Query,~Kernel.Geometry,~geometry,~update,~Packing.pack_phase,~step_same,count(query),0,query,state) :
    {read(query,_) == wanted(query) : Tile.Tile8}
  readback(query,state,evidence)

def complete_matches(+query: Query,+state: Packing.Packed) ->
  {Traversal.indexed_fold(~Kernel.Packed,~Kernel.Geometry,~Kernel.pack_phase,count(query),0,geometry(query),Packing.represent(state)) == Packing.represent(complete(query,state)) : Kernel.Packed}:
  Packing.packing_matches(count(query),0,geometry(query),state)

# The packed weights are allocated for the tiles of every phase.
def total(query: Bounds.Query) -> U32:
  +query = query
  Bounds.prefix_word(query,Bounds.count(query))
def allocated_depth(query: Bounds.Query) -> Nat:
  Tensors.depth_for((total(query) + 1 : U32))
def allocated_capacity(query: Bounds.Query) -> U32:
  U32.shln(1,allocated_depth(query))

type Allocated<-query: Bounds.Query> is Data:
  Allocated{depth_ok: {Nat.is_le(allocated_depth(query),31n) == True{} : Bool},capacity_ok: {U32.is_le(allocated_capacity(query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query),U32.to_nat(allocated_capacity(query))) == True{} : Bool}}

def total_below(+query: Bounds.Query,+upper: U32,capacity_below: {U32.is_lt(Machine.capacity(Bounds.limits(query)),upper) == True{} : Bool}) ->
  {U32.is_lt(total(query),upper) == True{} : Bool}:
  +capacity = Machine.capacity(Bounds.limits(query))
  +count = Bounds.count(query)
  +whole = Sums.sum(~Bounds.Query,~Bounds.pure,count,0n,query)
  NativeDivision.nat_lt_native(total(query),upper,
    %Equal.sym(Nat,U32.to_nat(total(query)),whole,Bounds.prefix_exact(query,count,Order.reflexive(count))) :
      {Nat.is_lt(_,U32.to_nat(upper)) == True{} : Bool}
      Order.le_lt_trans(whole,U32.to_nat(capacity),U32.to_nat(upper),Bounds.total_room(query,count,Order.reflexive(count)),Words.as_nat_lt(capacity,upper,capacity_below)))

def total_below_half(+query: Bounds.Query) -> {U32.is_lt(total(query),2147483648) == True{} : Bool}:
  total_below(query,2147483648,Limits.capacity_below_half(Bounds.limits(query)))

def allocation_finish(+query: Bounds.Query,evidence: Depth.Guarded<total(query)>) -> Allocated<query>:
  Depth.Guarded{+depth_ok,room} = evidence
  Allocated{depth_ok,Power.native_power_bound(allocated_depth(query),depth_ok),
    %Bounds.prefix_exact(query,Bounds.count(query),Order.reflexive(Bounds.count(query))) :
      {Nat.is_le(_,U32.to_nat(allocated_capacity(query))) == True{} : Bool}
      Words.strict_le(U32.to_nat(total(query)),U32.to_nat(allocated_capacity(query)),Words.as_nat_lt(total(query),allocated_capacity(query),room))}

def allocation(+query: Bounds.Query) -> Allocated<query>:
  allocation_finish(query,Depth.guarded(total(query),total_below_half(query)))

def allocated_query(bounds: Bounds.Query,source: Buffer.Model<Tile.Tile8>,phase: Nat,slot: Nat) -> Query:
  +bounds = bounds
  Query{bounds,source,phase,slot,allocated_depth(bounds),allocated_capacity(bounds)}

def fresh_buffer(bounds: Bounds.Query) -> Storage.Buffer<Tile.Tile8>:
  +bounds = bounds
  Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(total(bounds) + 1 : U32))

def initial(source: Buffer.Model<Tile.Tile8>,bounds: Bounds.Query) -> Packing.Packed:
  Packing.Packed{source,Buffer.remodel(~Tile.Tile8,fresh_buffer(bounds)),0}

def fresh_tree(+bounds: Bounds.Query) ->
  {Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,fresh_buffer(bounds))) == Tree.fresh(Tile.Tile8,allocated_depth(bounds),Tile.filled(0.0)) : Tree.Tree<Tile.Tile8>}:
  +model = Buffer.remodel(~Tile.Tile8,fresh_buffer(bounds))
  +count = (total(bounds) + 1 : U32)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,model),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),Tree.fresh(Tile.Tile8,allocated_depth(bounds),Tile.filled(0.0)),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),Buffer.tree(~Tile.Tile8,model),Buffer.content_tree_of(~Tile.Tile8,model)),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,content => Buffer.content_tree(~Tile.Tile8,content),Buffer.content(~Tile.Tile8,model),
      Buffer.Content{Tree.fresh(Tile.Tile8,allocated_depth(bounds),Tile.filled(0.0)),U32.to_nat(count)},
      Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,model),Buffer.contents(~Tile.Tile8,fresh_buffer(bounds)),
        Buffer.Content{Tree.fresh(Tile.Tile8,allocated_depth(bounds),Tile.filled(0.0)),U32.to_nat(count)},
        Buffer.remodel_content(~Tile.Tile8,fresh_buffer(bounds)),Allocation.contents(~Tile.Tile8,allocated_depth(bounds),Tile.filled(0.0),U32.to_nat(count)))))

def allocated_frame_finish(+bounds: Bounds.Query,+source: Buffer.Model<Tile.Tile8>,+phase: Nat,+slot: Nat,
  ready: Allocated<bounds>,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  Frame<allocated_query(bounds,source,phase,slot),0n,initial(source,bounds)>:
  Allocated{+depth_ok,+capacity_ok,room} = ready
  +depth = allocated_depth(bounds)
  +fresh = Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))
  +same = Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,fresh_buffer(bounds))),fresh,fresh_tree(bounds))
  Frame{{==},{==},
    %same : {Tree.capacity(Tile.Tile8,_) == allocated_capacity(bounds) : U32} Shape.capacity(Tile.Tile8,depth,fresh,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0))),
    %same : Shape.Balanced(Tile.Tile8,depth,_) Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),depth_ok,
    Facts{phase_inside,slot_inside,capacity_ok,room}}

def allocated_frame(+bounds: Bounds.Query,+source: Buffer.Model<Tile.Tile8>,+phase: Nat,+slot: Nat,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  Frame<allocated_query(bounds,source,phase,slot),0n,initial(source,bounds)>:
  allocated_frame_finish(bounds,source,phase,slot,allocation(bounds),phase_inside,slot_inside)

def allocated_readback(+bounds: Bounds.Query,+source: Buffer.Model<Tile.Tile8>,+phase: Nat,+slot: Nat,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  {read(allocated_query(bounds,source,phase,slot),complete(allocated_query(bounds,source,phase,slot),initial(source,bounds))) ==
    wanted(allocated_query(bounds,source,phase,slot)) : Tile.Tile8}:
  +query = allocated_query(bounds,source,phase,slot)
  complete_readback(query,initial(source,bounds),allocated_frame(bounds,source,phase,slot,phase_inside,slot_inside))

def weight_total_program(+bounds: Bounds.Query) ->
  {total(bounds) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,Kernel.window_jobs(Bounds.geometry(bounds)),0,Bounds.geometry(bounds),0) : U32}:
  %Equal.sym(Nat,Kernel.window_jobs(Bounds.geometry(bounds)),Bounds.count(bounds),Bounds.window_jobs_exact(bounds)) :
    {total(bounds) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,_,0,Bounds.geometry(bounds),0) : U32}
  Bounds.prefix_program(bounds,Bounds.count(bounds))

# The packing the kernel runs, from the source weights and a fresh buffer.
def native_initial(bounds: Bounds.Query,source: Buffer.Model<Tile.Tile8>) -> Kernel.Packed:
  +geometry = Bounds.geometry(bounds)
  +weight_tiles = Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,Kernel.window_jobs(geometry),0,geometry,0)
  Kernel.Packed{Buffer.pack(~Tile.Tile8,source),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(weight_tiles + 1 : U32)),0}

def initial_matches(+bounds: Bounds.Query,+source: Buffer.Model<Tile.Tile8>) ->
  {native_initial(bounds,source) == Packing.represent(initial(source,bounds)) : Kernel.Packed}:
  %weight_total_program(bounds) :
    {Kernel.Packed{Buffer.pack(~Tile.Tile8,source),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(_ + 1 : U32)),0} == Packing.represent(initial(source,bounds)) : Kernel.Packed}
  %Buffer.remodeled(~Tile.Tile8,fresh_buffer(bounds)) :
    {Kernel.Packed{Buffer.pack(~Tile.Tile8,source),_,0} == Packing.represent(initial(source,bounds)) : Kernel.Packed}
  {==}

def native_complete(bounds: Bounds.Query,source: Buffer.Model<Tile.Tile8>) -> Kernel.Packed:
  +bounds = bounds
  Traversal.indexed_fold(~Kernel.Packed,~Kernel.Geometry,~Kernel.pack_phase,Kernel.window_jobs(Bounds.geometry(bounds)),0,Bounds.geometry(bounds),native_initial(bounds,source))

def native_complete_matches(+bounds: Bounds.Query,+source: Buffer.Model<Tile.Tile8>,+phase: Nat,+slot: Nat) ->
  {native_complete(bounds,source) == Packing.represent(complete(allocated_query(bounds,source,phase,slot),initial(source,bounds))) : Kernel.Packed}:
  +query = allocated_query(bounds,source,phase,slot)
  %Equal.sym(Nat,Kernel.window_jobs(Bounds.geometry(bounds)),Bounds.count(bounds),Bounds.window_jobs_exact(bounds)) :
    {Traversal.indexed_fold(~Kernel.Packed,~Kernel.Geometry,~Kernel.pack_phase,_,0,geometry(query),native_initial(bounds,source)) == Packing.represent(complete(query,initial(source,bounds))) : Kernel.Packed}
  %Equal.sym(Kernel.Packed,native_initial(bounds,source),Packing.represent(initial(source,bounds)),initial_matches(bounds,source)) :
    {Traversal.indexed_fold(~Kernel.Packed,~Kernel.Geometry,~Kernel.pack_phase,Bounds.count(bounds),0,geometry(query),_) == Packing.represent(complete(query,initial(source,bounds))) : Kernel.Packed}
  complete_matches(query,initial(source,bounds))
