# Readback survives the complete concatenation of actual packed phases.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./input_gradient_tiled_weights.bend as Packing
import ./input_gradient_tiled_offsets.bend as Offsets
import ./input_gradient_tiled_packing_bounds.bend as Bounds
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_refinement.bend as Execution
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./indexed_output_readback.bend as Readback
import ./traversal_array_refinement.bend as Arrays
import ./traversal_indexed_values.bend as Indexed
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Words
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_range_sums.bend as Sums
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as NativeDivision
import ./contiguous_copy_readback.bend as Addresses
import ../tensor.bend as Tensors
import ../machine_limits.bend as Machine
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as NativeAllocation
import ./tensor_allocation_depth.bend as Depth
import ./machine_limit_bounds.bend as Limits
import ./word_power_of_two.bend as Power
import ./storage_refinement.bend as NativeStorage
import ./storage_allocation_refinement.bend as Allocate

type Query is Data:
  Query{bounds: Bounds.Query,layout: Kernel.Tiling,source: Buffer.Model<F32>,phase: Nat,slot: Nat,tap: Nat,weight_depth: Nat,weight_capacity: U32,offset_depth: Nat,offset_capacity: U32}

def bounds(query: Query) -> Bounds.Query:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  bounds
def layout(query: Query) -> Kernel.Tiling:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  layout
def source(query: Query) -> Buffer.Model<F32>:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  source
def phase(query: Query) -> Nat:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  phase
def slot(query: Query) -> Nat:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  slot
def tap(query: Query) -> Nat:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  tap
def weight_depth(query: Query) -> Nat:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  weight_depth
def weight_capacity(query: Query) -> U32:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  weight_capacity
def offset_depth(query: Query) -> Nat:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  offset_depth
def offset_capacity(query: Query) -> U32:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  offset_capacity

def reflag(flag: Bool,query: Bounds.Query) -> Bounds.Query:
  Bounds.Query{weights,call} = query
  Bounds.Query{flag,call}
def weight_bounds(query: Query) -> Bounds.Query:
  reflag(True{},bounds(query))
def offset_bounds(query: Query) -> Bounds.Query:
  reflag(False{},bounds(query))
def count(query: Query) -> Nat:
  Bounds.count(bounds(query))
def selected(query: Query) -> Kernel.Phase:
  +query = query
  Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(phase(query)))
def context(query: Query) -> Kernel.PackContext:
  +query = query
  Kernel.PackContext{Bounds.geometry(bounds(query)),layout(query)}
def weight_address(query: Query) -> U32:
  +query = query
  (Bounds.prefix_word(weight_bounds(query),phase(query)) + U32.from_nat(slot(query)) : U32)
def offset_address(query: Query) -> U32:
  +query = query
  (Bounds.prefix_word(offset_bounds(query),phase(query)) + U32.from_nat(tap(query)) : U32)
def weight_tree(state: Packing.Packed) -> Tree.Tree<Tile.Tile8>:
  Packing.Packed{source,weights,offsets,weight_base,offset_base} = state
  weights
def offset_tree(state: Packing.Packed) -> Tree.Tree<U32>:
  Packing.Packed{source,weights,offsets,weight_base,offset_base} = state
  offsets

type Value is Data:
  Value{weight: Tile.Tile8,offset: U32}
def read(query: Query,state: Packing.Packed) -> Value:
  +query = query
  +state = state
  Value{Tree.read(Tile.Tile8,weight_tree(state),weight_address(query)),Tree.read(U32,offset_tree(state),offset_address(query))}
def wanted(query: Query) -> Value:
  +query = query
  Value{Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected(query),source(query))),U32.from_nat(slot(query))),
    Offsets.value(Packing.phase_table(selected(query),layout(query),0),U32.from_nat(tap(query)))}
def update(index: U32,query: Query,state: Packing.Packed) -> Packing.Packed:
  Packing.pack_phase(index,context(query),state)

type Facts<-query: Query> is Data:
  Facts{phase_inside: {Nat.is_lt(phase(query),count(query)) == True{} : Bool},
    slot_inside: {Nat.is_lt(slot(query),U32.to_nat(Packing.phase_count(selected(query)))) == True{} : Bool},
    tap_inside: {Nat.is_lt(tap(query),U32.to_nat(Kernel.phase_inner(selected(query)))) == True{} : Bool},
    weight_capacity_ok: {U32.is_le(weight_capacity(query),2147483648) == True{} : Bool},
    offset_capacity_ok: {U32.is_le(offset_capacity(query),2147483648) == True{} : Bool},
    weight_room: {Nat.is_le(Sums.sum(Bounds.Query,Bounds.pure,count(query),0n,weight_bounds(query)),U32.to_nat(weight_capacity(query))) == True{} : Bool},
    offset_room: {Nat.is_le(Sums.sum(Bounds.Query,Bounds.pure,count(query),0n,offset_bounds(query)),U32.to_nat(offset_capacity(query))) == True{} : Bool}}

type Frame<-query: Query,-index: Nat,-state: Packing.Packed> is Data:
  Frame{source_equal: {Packing.source(state) == source(query) : Buffer.Model<F32>},
    weight_base: {Packing.weight_base(state) == Bounds.prefix_word(weight_bounds(query),index) : U32},
    offset_base: {Packing.offset_base(state) == Bounds.prefix_word(offset_bounds(query),index) : U32},
    weight_capacity_equal: {Tree.capacity(Tile.Tile8,weight_tree(state)) == weight_capacity(query) : U32},
    weight_balanced: Shape.Balanced(Tile.Tile8,weight_depth(query),weight_tree(state)),weight_depth_ok: {Nat.is_le(weight_depth(query),31n) == True{} : Bool},
    offset_capacity_equal: {Tree.capacity(U32,offset_tree(state)) == offset_capacity(query) : U32},
    offset_balanced: Shape.Balanced(U32,offset_depth(query),offset_tree(state)),offset_depth_ok: {Nat.is_le(offset_depth(query),31n) == True{} : Bool},facts: Facts<query>}

def frame(query: Query,index: Nat,state: Packing.Packed) -> Data:
  Frame<query,index,state>

type Shaped<-state: Packing.Packed,-wd: Nat,-wc: U32,-od: Nat,-oc: U32> is Data:
  Shaped{weight_capacity: {Tree.capacity(Tile.Tile8,weight_tree(state)) == wc : U32},weight_balance: Shape.Balanced(Tile.Tile8,wd,weight_tree(state)),
    offset_capacity: {Tree.capacity(U32,offset_tree(state)) == oc : U32},offset_balance: Shape.Balanced(U32,od,offset_tree(state))}

def shape_selected(empty: Bool,+selected: Kernel.Phase,+layout: Kernel.Tiling,+state: Packing.Packed,+wd: Nat,+wc: U32,+od: Nat,+oc: U32,
  evidence: Shaped<state,wd,wc,od,oc>) -> Shaped<Packing.pack_selected(empty,selected,layout,state),wd,wc,od,oc>:
  match empty:
    case True{}: evidence
    case False{}:
      Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,+inner} = selected
      Packing.Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
      Shaped{weight_capacity,weight_balance,offset_capacity,offset_balance} = evidence
      +count = Packing.phase_count(selected)
      +copy = {Execution.TileCopy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected,source)),weights,0,weight_base} : Execution.TileCopy}
      +table = Packing.phase_table(selected,layout,offset_base)
      Shaped{Equal.trans(U32,Tree.capacity(Tile.Tile8,Execution.tile_target(Loop.run(~Execution.TileCopy,~Execution.tile_step,U32.to_nat(count),copy))),Tree.capacity(Tile.Tile8,weights),wc,
          Packing.copy_capacity(U32.to_nat(count),copy),weight_capacity),Packing.copy_balanced(U32.to_nat(count),wd,copy,weight_balance),
        Equal.trans(U32,Tree.capacity(U32,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Execution.offset,U32.to_nat(inner),0,table,offsets)),Tree.capacity(U32,offsets),oc,
          Offsets.capacity_kept(U32.to_nat(inner),table,offsets),offset_capacity),Offsets.balanced(U32.to_nat(inner),table,offsets,od,offset_balance)}

def shape_phase(+index: U32,+context: Kernel.PackContext,+state: Packing.Packed,+wd: Nat,+wc: U32,+od: Nat,+oc: U32,
  evidence: Shaped<state,wd,wc,od,oc>) -> Shaped<Packing.pack_phase(index,context,state),wd,wc,od,oc>:
  Kernel.PackContext{+geometry,+layout} = context
  +selected = Kernel.phase_of(geometry,index)
  shape_selected(U32.is_eq(Kernel.phase_inner(selected),0),selected,layout,state,wd,wc,od,oc,evidence)

def weight_next(+query: Query,+index: Nat,+state: Packing.Packed,
  before: {Packing.weight_base(state) == Bounds.prefix_word(weight_bounds(query),index) : U32}) ->
  {Packing.weight_base(update(U32.from_nat(index),query,state)) == Bounds.prefix_word(weight_bounds(query),1n+index) : U32}:
  Query{b,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = b
  Domain.Call{g,batch,limits,facts} = call
  +previous = Packing.weight_base(state)
  +base = Bounds.prefix_word(weight_bounds(query),index)
  +step = Kernel.weight_total(U32.from_nat(index),g,base)
  Equal.trans(U32,Packing.weight_base(update(U32.from_nat(index),query,state)),Kernel.weight_total(U32.from_nat(index),g,previous),Bounds.prefix_word(weight_bounds(query),1n+index),
    Packing.weight_step(U32.from_nat(index),context(query),state),
    Equal.trans(U32,Kernel.weight_total(U32.from_nat(index),g,previous),step,Bounds.prefix_word(weight_bounds(query),1n+index),
      Equal.cong(U32,U32,base => Kernel.weight_total(U32.from_nat(index),g,base),previous,base,before),
      Equal.trans(U32,step,Bounds.update(U32.from_nat(index),weight_bounds(query),base),Bounds.prefix_word(weight_bounds(query),1n+index),
        Equal.sym(U32,Bounds.update(U32.from_nat(index),weight_bounds(query),base),step,Bounds.weights_step(U32.from_nat(index),weight_bounds(query),base,{==})),
        Equal.sym(U32,Bounds.prefix_word(weight_bounds(query),1n+index),Bounds.update(U32.from_nat(index),weight_bounds(query),base),Bounds.prefix_step(weight_bounds(query),index)))))

def offset_next(+query: Query,+index: Nat,+state: Packing.Packed,
  before: {Packing.offset_base(state) == Bounds.prefix_word(offset_bounds(query),index) : U32}) ->
  {Packing.offset_base(update(U32.from_nat(index),query,state)) == Bounds.prefix_word(offset_bounds(query),1n+index) : U32}:
  Query{b,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = b
  Domain.Call{g,batch,limits,facts} = call
  +previous = Packing.offset_base(state)
  +base = Bounds.prefix_word(offset_bounds(query),index)
  +step = Kernel.inner_total(U32.from_nat(index),g,base)
  Equal.trans(U32,Packing.offset_base(update(U32.from_nat(index),query,state)),Kernel.inner_total(U32.from_nat(index),g,previous),Bounds.prefix_word(offset_bounds(query),1n+index),
    Packing.offset_step(U32.from_nat(index),context(query),state),
    Equal.trans(U32,Kernel.inner_total(U32.from_nat(index),g,previous),step,Bounds.prefix_word(offset_bounds(query),1n+index),
      Equal.cong(U32,U32,base => Kernel.inner_total(U32.from_nat(index),g,base),previous,base,before),
      Equal.trans(U32,step,Bounds.update(U32.from_nat(index),offset_bounds(query),base),Bounds.prefix_word(offset_bounds(query),1n+index),
        Equal.sym(U32,Bounds.update(U32.from_nat(index),offset_bounds(query),base),step,Bounds.entries_step(U32.from_nat(index),offset_bounds(query),base,{==})),
        Equal.sym(U32,Bounds.prefix_word(offset_bounds(query),1n+index),Bounds.update(U32.from_nat(index),offset_bounds(query),base),Bounds.prefix_step(offset_bounds(query),index)))))

def framed_next(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,
  shaped: Shaped<update(U32.from_nat(index),query,state),weight_depth(query),weight_capacity(query),offset_depth(query),offset_capacity(query)>) ->
  Frame<query,1n+index,update(U32.from_nat(index),query,state)>:
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,+weight_depth_ok,offset_capacity_equal,offset_balanced,+offset_depth_ok,+facts} = evidence
  Shaped{new_weight_capacity,new_weight_balance,new_offset_capacity,new_offset_balance} = shaped
  Frame{Equal.trans(Buffer.Model<F32>,Packing.source(update(U32.from_nat(index),query,state)),Packing.source(state),source(query),Packing.phase_keeps_source(U32.from_nat(index),context(query),state),source_equal),
    weight_next(query,index,state,weight_base),offset_next(query,index,state,offset_base),new_weight_capacity,new_weight_balance,weight_depth_ok,new_offset_capacity,new_offset_balance,offset_depth_ok,facts}

def frame_step(+query: Query,+index: Nat,+state: Packing.Packed,inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  evidence: Frame<query,index,state>) -> Frame<query,1n+index,update(U32.from_nat(index),query,state)>:
  Frame{+source_equal,+weight_base,+offset_base,+weight_capacity_equal,+weight_balanced,+weight_depth_ok,+offset_capacity_equal,+offset_balanced,+offset_depth_ok,+facts} = evidence
  +evidence = {Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} : Frame<query,index,state>}
  framed_next(query,index,state,evidence,shape_phase(U32.from_nat(index),context(query),state,weight_depth(query),weight_capacity(query),offset_depth(query),offset_capacity(query),
    Shaped{weight_capacity_equal,weight_balanced,offset_capacity_equal,offset_balanced}))

def count_reflagged(+flag: Bool,+query: Bounds.Query) -> {Bounds.count(reflag(flag,query)) == Bounds.count(query) : Nat}:
  Bounds.Query{weights,call} = query
  Domain.Call{g,batch,limits,facts} = call
  {==}

def geometry_reflagged(+flag: Bool,+query: Bounds.Query) -> {Bounds.geometry(reflag(flag,query)) == Bounds.geometry(query) : Kernel.Geometry}:
  Bounds.Query{weights,call} = query
  Domain.Call{g,batch,limits,facts} = call
  {==}

def flag_reflagged(+flag: Bool,+query: Bounds.Query) -> {Bounds.weights(reflag(flag,query)) == flag : Bool}:
  Bounds.Query{weights,call} = query
  Domain.Call{g,batch,limits,facts} = call
  {==}

def flagged_active(+flag: Bool,+query: Query,+index: Nat,active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == False{} : Bool}) ->
  {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(reflag(flag,bounds(query))),U32.from_nat(index))),0) == False{} : Bool}:
  %Equal.sym(Kernel.Geometry,Bounds.geometry(reflag(flag,bounds(query))),Bounds.geometry(bounds(query)),geometry_reflagged(flag,bounds(query))) :
    {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(_,U32.from_nat(index))),0) == False{} : Bool}
  active

def flagged_inside(+flag: Bool,+query: Query,+index: Nat,inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_lt(index,Bounds.count(reflag(flag,bounds(query)))) == True{} : Bool}:
  %Equal.sym(Nat,Bounds.count(reflag(flag,bounds(query))),count(query),count_reflagged(flag,bounds(query))) : {Nat.is_lt(index,_) == True{} : Bool}
  inside

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

def phase_copy(query: Query,index: Nat,state: Packing.Packed) -> Packing.Copy:
  +query = query
  +index = index
  +state = state
  Packing.Copy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),Packing.source(state))),0,Packing.weight_base(state)}

def copy_query(query: Query,index: Nat,state: Packing.Packed) -> Packing.CopyQuery:
  +query = query
  +index = index
  +state = state
  Packing.CopyQuery{phase_copy(query,index,state),U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)))),slot(query),weight_depth(query),weight_capacity(query)}

def table_query(query: Query,index: Nat,state: Packing.Packed) -> Offsets.Query:
  +query = query
  +index = index
  +state = state
  +phase = Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))
  Offsets.Query{Packing.phase_table(phase,layout(query),Packing.offset_base(state)),U32.to_nat(Kernel.phase_inner(phase)),tap(query),offset_depth(query),offset_capacity(query)}

def copy_native(+query: Query,+index: Nat) ->
  {Bounds.native(weight_bounds(query),U32.from_nat(index)) == Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))) : U32}:
  Query{b,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = b
  Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} = call
  {==}

def table_native(+query: Query,+index: Nat) ->
  {Bounds.native(offset_bounds(query),U32.from_nat(index)) == Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))) : U32}:
  Query{b,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = b
  Domain.Call{g,batch,limits,facts} = call
  {==}

def copy_room(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Packing.weight_base(state)),U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))))),U32.to_nat(weight_capacity(query))) == True{} : Bool}:
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} = evidence
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  %Equal.sym(U32,Packing.weight_base(state),Bounds.prefix_word(weight_bounds(query),index),weight_base) :
    {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))))),U32.to_nat(weight_capacity(query))) == True{} : Bool}
  %copy_native(query,index) :
    {Nat.is_le(Nat.add(U32.to_nat(Bounds.prefix_word(weight_bounds(query),index)),U32.to_nat(_)),U32.to_nat(weight_capacity(query))) == True{} : Bool}
  native_range_room(weight_bounds(query),index,weight_capacity(query),flagged_inside(True{},query,index,inside),
    %Equal.sym(Nat,Bounds.count(weight_bounds(query)),count(query),count_reflagged(True{},bounds(query))) :
      {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,_,0n,weight_bounds(query)),U32.to_nat(weight_capacity(query))) == True{} : Bool} weight_room)

def table_room(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Packing.offset_base(state)),U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))))),U32.to_nat(offset_capacity(query))) == True{} : Bool}:
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} = evidence
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  %Equal.sym(U32,Packing.offset_base(state),Bounds.prefix_word(offset_bounds(query),index),offset_base) :
    {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))))),U32.to_nat(offset_capacity(query))) == True{} : Bool}
  %table_native(query,index) :
    {Nat.is_le(Nat.add(U32.to_nat(Bounds.prefix_word(offset_bounds(query),index)),U32.to_nat(_)),U32.to_nat(offset_capacity(query))) == True{} : Bool}
  native_range_room(offset_bounds(query),index,offset_capacity(query),flagged_inside(False{},query,index,inside),
    %Equal.sym(Nat,Bounds.count(offset_bounds(query)),count(query),count_reflagged(False{},bounds(query))) :
      {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,_,0n,offset_bounds(query)),U32.to_nat(offset_capacity(query))) == True{} : Bool} offset_room)

def producer_copy_frame(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  Indexed.AffineFrame<Tile.Tile8,Packing.Copy,Packing.copy_affine(copy_query(query,phase(query),state)),weight_tree(state)>:
  Frame{source_equal,weight_base,offset_base,+weight_capacity_equal,+weight_balanced,+weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  Facts{+phase_inside,+slot_inside,tap_inside,+weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  Indexed.AffineFrame{weight_capacity_equal,weight_balanced,weight_depth_ok,weight_capacity_ok,copy_room(query,phase(query),state,evidence,phase_inside),slot_inside}

def producer_table_frame(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  Offsets.Frame<table_query(query,phase(query),state),offset_tree(state)>:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Packing.Packed{source_state,weights_tree,offsets_tree,weight_base,offset_base} = state
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,+offset_capacity_equal,+offset_balanced,+offset_depth_ok,+facts} = evidence
  Facts{+phase_inside,slot_inside,+tap_inside,weight_capacity_ok,+offset_capacity_ok,weight_room,offset_room} = facts
  Offsets.Frame{offset_capacity_equal,offset_balanced,offset_depth_ok,offset_capacity_ok,table_room(query,phase,state,evidence,phase_inside),tap_inside}

def copied_value(+phase: Kernel.Phase,+layout: Kernel.Tiling,+state: Packing.Packed,+slot: Nat,+depth: Nat,+capacity: U32,
  evidence: Indexed.AffineFrame<Tile.Tile8,Packing.Copy,
    Packing.copy_affine(Packing.CopyQuery{Packing.Copy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,Packing.source(state))),0,Packing.weight_base(state)},U32.to_nat(Packing.phase_count(phase)),slot,depth,capacity}),weight_tree(state)>) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},phase,layout,state)),(Packing.weight_base(state) + U32.from_nat(slot) : U32)) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,Packing.source(state))),U32.from_nat(slot)) : Tile.Tile8}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = phase
  Packing.Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
  +copy = {Packing.Copy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,source)),0,weight_base} : Packing.Copy}
  +query = {Packing.CopyQuery{copy,U32.to_nat(Packing.phase_count(phase)),slot,depth,capacity} : Packing.CopyQuery}
  %Packing.copy_initial(copy,weights) :
    {Tree.read(Tile.Tile8,Execution.tile_target(Loop.run(~Execution.TileCopy,~Execution.tile_step,U32.to_nat(Packing.phase_count(phase)),_)),(weight_base + U32.from_nat(slot) : U32)) ==
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,source)),U32.from_nat(slot)) : Tile.Tile8}
  %Arithmetic.add_zero_left(U32.from_nat(slot)) :
    {Tree.read(Tile.Tile8,Execution.tile_target(Loop.run(~Execution.TileCopy,~Execution.tile_step,U32.to_nat(Packing.phase_count(phase)),Packing.copy_represent(copy,0,weights))),
      (weight_base + U32.from_nat(slot) : U32)) == Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,source)),_) : Tile.Tile8}
  Packing.copy_readback(query,weights,evidence)

def producer_weight(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected(query),layout(query),state)),weight_address(query)) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected(query),source(query))),U32.from_nat(slot(query))) : Tile.Tile8}:
  Frame{+source_equal,+weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} = evidence
  %weight_base :
    {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected(query),layout(query),state)),(_ + U32.from_nat(slot(query)) : U32)) ==
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected(query),source(query))),U32.from_nat(slot(query))) : Tile.Tile8}
  %source_equal :
    {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},selected(query),layout(query),state)),(Packing.weight_base(state) + U32.from_nat(slot(query)) : U32)) ==
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected(query),_)),U32.from_nat(slot(query))) : Tile.Tile8}
  copied_value(selected(query),layout(query),state,slot(query),weight_depth(query),weight_capacity(query),producer_copy_frame(query,state,evidence))

def value_rebased(+phase: Kernel.Phase,+layout: Kernel.Tiling,+first: U32,+second: U32,+index: U32) ->
  {Offsets.value(Packing.phase_table(phase,layout,first),index) == Offsets.value(Packing.phase_table(phase,layout,second),index) : U32}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  {==}

def copied_offset_value(+phase: Kernel.Phase,+layout: Kernel.Tiling,+state: Packing.Packed,+tap: Nat,+depth: Nat,+capacity: U32,
  +evidence: Offsets.Frame<Offsets.Query{Packing.phase_table(phase,layout,Packing.offset_base(state)),U32.to_nat(Kernel.phase_inner(phase)),tap,depth,capacity},offset_tree(state)>) ->
  {Tree.read(U32,offset_tree(Packing.pack_selected(False{},phase,layout,state)),(Packing.offset_base(state) + U32.from_nat(tap) : U32)) ==
    Offsets.value(Packing.phase_table(phase,layout,0),U32.from_nat(tap)) : U32}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Packing.Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
  +query = {Offsets.Query{Packing.phase_table(phase,layout,offset_base),U32.to_nat(inner),tap,depth,capacity} : Offsets.Query}
  %value_rebased(phase,layout,offset_base,0,U32.from_nat(tap)) :
    {Tree.read(U32,offset_tree(Packing.pack_selected(False{},phase,layout,state)),(offset_base + U32.from_nat(tap) : U32)) == _ : U32}
  Offsets.readback(query,offsets,evidence)

def producer_offset(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  {Tree.read(U32,offset_tree(Packing.pack_selected(False{},selected(query),layout(query),state)),offset_address(query)) ==
    Offsets.value(Packing.phase_table(selected(query),layout(query),0),U32.from_nat(tap(query))) : U32}:
  Frame{source_equal,weight_base,+offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} = evidence
  %offset_base :
    {Tree.read(U32,offset_tree(Packing.pack_selected(False{},selected(query),layout(query),state)),(_ + U32.from_nat(tap(query)) : U32)) ==
      Offsets.value(Packing.phase_table(selected(query),layout(query),0),U32.from_nat(tap(query))) : U32}
  copied_offset_value(selected(query),layout(query),state,tap(query),offset_depth(query),offset_capacity(query),producer_table_frame(query,state,evidence))

def selected_active(+query: Query,+facts: Facts<query>) -> {U32.is_eq(Kernel.phase_inner(selected(query)),0) == False{} : Bool}:
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  %Equal.sym(Bool,U32.is_eq(Kernel.phase_inner(selected(query)),0),Nat.is_eq(U32.to_nat(Kernel.phase_inner(selected(query))),0n),Comparison.eq(Kernel.phase_inner(selected(query)),0)) : {_ == False{} : Bool}
  NativeDivision.nat_nonzero(U32.to_nat(Kernel.phase_inner(selected(query))),Order.positive_bound(tap(query),U32.to_nat(Kernel.phase_inner(selected(query))),tap_inside))

def produced_at_target(+query: Query,+state: Packing.Packed,+evidence: Frame<query,phase(query),state>) ->
  {read(query,update(U32.from_nat(phase(query)),query,state)) == wanted(query) : Value}:
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  +next = Packing.pack_selected(False{},selected(query),layout(query),state)
  +weight = Tree.read(Tile.Tile8,weight_tree(next),weight_address(query))
  +offset = Tree.read(U32,offset_tree(next),offset_address(query))
  +want_weight = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Packing.phase_weights(selected(query),source(query))),U32.from_nat(slot(query)))
  +want_offset = Offsets.value(Packing.phase_table(selected(query),layout(query),0),U32.from_nat(tap(query)))
  %Equal.sym(Bool,U32.is_eq(Kernel.phase_inner(selected(query)),0),False{},selected_active(query,facts)) :
    {read(query,Packing.pack_selected(_,selected(query),layout(query),state)) == wanted(query) : Value}
  Equal.trans(Value,Value{weight,offset},Value{want_weight,offset},Value{want_weight,want_offset},
    Equal.cong(Tile.Tile8,Value,value => Value{value,offset},weight,want_weight,producer_weight(query,state,evidence)),
    Equal.cong(U32,Value,value => Value{want_weight,value},offset,want_offset,producer_offset(query,state,evidence)))

def produced(+query: Query,+index: Nat,+state: Packing.Packed,evidence: Frame<query,index,state>,same: {index == phase(query) : Nat}) ->
  {read(query,update(U32.from_nat(index),query,state)) == wanted(query) : Value}:
  +evidence = evidence
  +same = same
  %Equal.sym(Nat,index,phase(query),same) : {read(query,update(U32.from_nat(_),query,state)) == wanted(query) : Value}
  produced_at_target(query,state,%same : Frame<query,_,state> evidence)

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

type Selection<-query: Bounds.Query,-phase: Nat,-slot: Nat,-capacity: U32> is Data:
  Selection{phase_inside: {Nat.is_lt(phase,Bounds.count(query)) == True{} : Bool},
    slot_inside: {Nat.is_lt(slot,U32.to_nat(Bounds.native(query,U32.from_nat(phase)))) == True{} : Bool},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,Bounds.count(query),0n,query),U32.to_nat(capacity)) == True{} : Bool}}

def weight_selection(+query: Query,+facts: Facts<query>) -> Selection<weight_bounds(query),phase(query),slot(query),weight_capacity(query)>:
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  Selection{flagged_inside(True{},query,phase(query),phase_inside),
    %Equal.sym(U32,Bounds.native(weight_bounds(query),U32.from_nat(phase(query))),Packing.phase_count(selected(query)),copy_native(query,phase(query))) : {Nat.is_lt(slot(query),U32.to_nat(_)) == True{} : Bool} slot_inside,
    weight_capacity_ok,%Equal.sym(Nat,Bounds.count(weight_bounds(query)),count(query),count_reflagged(True{},bounds(query))) :
      {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,_,0n,weight_bounds(query)),U32.to_nat(weight_capacity(query))) == True{} : Bool} weight_room}

def offset_selection(+query: Query,+facts: Facts<query>) -> Selection<offset_bounds(query),phase(query),tap(query),offset_capacity(query)>:
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  Selection{flagged_inside(False{},query,phase(query),phase_inside),
    %Equal.sym(U32,Bounds.native(offset_bounds(query),U32.from_nat(phase(query))),Kernel.phase_inner(selected(query)),table_native(query,phase(query))) : {Nat.is_lt(tap(query),U32.to_nat(_)) == True{} : Bool} tap_inside,
    offset_capacity_ok,%Equal.sym(Nat,Bounds.count(offset_bounds(query)),count(query),count_reflagged(False{},bounds(query))) :
      {Nat.is_le(Sums.sum(~Bounds.Query,~Bounds.pure,_,0n,offset_bounds(query)),U32.to_nat(offset_capacity(query))) == True{} : Bool} offset_room}

def selection_inside(+query: Bounds.Query,+phase: Nat,+slot: Nat,+capacity: U32,+evidence: Selection<query,phase,slot,capacity>) ->
  {U32.is_lt((Bounds.prefix_word(query,phase) + U32.from_nat(slot) : U32),capacity) == True{} : Bool}:
  Selection{phase_inside,slot_inside,capacity_ok,room} = evidence
  selected_address_inside(query,phase,slot,capacity,capacity_ok,phase_inside,slot_inside,room)

def selection_before(+query: Bounds.Query,+phase: Nat,+slot: Nat,+index: Nat,+capacity: U32,+evidence: Selection<query,phase,slot,capacity>,
  +inside: {Nat.is_lt(index,Bounds.count(query)) == True{} : Bool},after: {Nat.is_lt(phase,index) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((Bounds.prefix_word(query,phase) + U32.from_nat(slot) : U32)),U32.to_nat(Bounds.prefix_word(query,index))) == True{} : Bool}:
  Selection{phase_inside,slot_inside,capacity_ok,room} = evidence
  selected_address_before(query,phase,slot,index,capacity,capacity_ok,phase_inside,inside,slot_inside,after,room)

def later_copy_query(query: Query,index: Nat,state: Packing.Packed) -> Packing.CopyQuery:
  +query = query
  +index = index
  +state = state
  Packing.CopyQuery{phase_copy(query,index,state),U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)))),0n,weight_depth(query),weight_capacity(query)}

def later_table_query(query: Query,index: Nat,state: Packing.Packed) -> Offsets.Query:
  +query = query
  +index = index
  +state = state
  +phase = Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))
  Offsets.Query{Packing.phase_table(phase,layout(query),Packing.offset_base(state)),U32.to_nat(Kernel.phase_inner(phase)),0n,offset_depth(query),offset_capacity(query)}

def later_copy_frame(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == False{} : Bool}) ->
  Indexed.AffineFrame<Tile.Tile8,Packing.Copy,Packing.copy_affine(later_copy_query(query,index,state)),weight_tree(state)>:
  Frame{source_equal,weight_base,offset_base,+weight_capacity_equal,+weight_balanced,+weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  Facts{phase_inside,slot_inside,tap_inside,+weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  Indexed.AffineFrame{weight_capacity_equal,weight_balanced,weight_depth_ok,weight_capacity_ok,copy_room(query,index,state,evidence,inside),
    %copy_native(query,index) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
    Bounds.weight_span_positive(weight_bounds(query),index,flag_reflagged(True{},bounds(query)),flagged_active(True{},query,index,active))}

def later_table_frame(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == False{} : Bool}) ->
  Offsets.Frame<later_table_query(query,index,state),offset_tree(state)>:
  Query{bounds,layout,source,phase,slot,tap,weight_depth,weight_capacity,offset_depth,offset_capacity} = query
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Packing.Packed{source_state,weights_tree,offsets_tree,weight_base,offset_base} = state
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,+offset_capacity_equal,+offset_balanced,+offset_depth_ok,+facts} = evidence
  Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,+offset_capacity_ok,weight_room,offset_room} = facts
  Offsets.Frame{offset_capacity_equal,offset_balanced,offset_depth_ok,offset_capacity_ok,table_room(query,index,state,evidence,inside),
    Order.nonzero_positive(U32.to_nat(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index)))),
      %Comparison.eq(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index))),0) : {_ == False{} : Bool} active)}

def copied_before(+phase: Kernel.Phase,+layout: Kernel.Tiling,+state: Packing.Packed,+address: U32,+depth: Nat,+capacity: U32,
  +evidence: Indexed.AffineFrame<Tile.Tile8,Packing.Copy,
    Packing.copy_affine(Packing.CopyQuery{Packing.Copy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,Packing.source(state))),0,Packing.weight_base(state)},U32.to_nat(Packing.phase_count(phase)),0n,depth,capacity}),weight_tree(state)>,
  +before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Packing.weight_base(state))) == True{} : Bool},
  +inside: {U32.is_lt(address,capacity) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},phase,layout,state)),address) == Tree.read(Tile.Tile8,weight_tree(state),address) : Tile.Tile8}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = phase
  Packing.Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
  +copy = {Packing.Copy{Buffer.tree(~Tile.Tile8,Packing.phase_weights(phase,source)),0,weight_base} : Packing.Copy}
  +query = {Packing.CopyQuery{copy,U32.to_nat(Packing.phase_count(phase)),0n,depth,capacity} : Packing.CopyQuery}
  %Packing.copy_initial(copy,weights) :
    {Tree.read(Tile.Tile8,Execution.tile_target(Loop.run(~Execution.TileCopy,~Execution.tile_step,U32.to_nat(Packing.phase_count(phase)),_)),address) == Tree.read(Tile.Tile8,weights,address) : Tile.Tile8}
  Packing.copy_read_before(query,address,weights,evidence,before,inside)

def copied_offsets_before(+phase: Kernel.Phase,+layout: Kernel.Tiling,+state: Packing.Packed,+address: U32,+depth: Nat,+capacity: U32,
  +evidence: Offsets.Frame<Offsets.Query{Packing.phase_table(phase,layout,Packing.offset_base(state)),U32.to_nat(Kernel.phase_inner(phase)),0n,depth,capacity},offset_tree(state)>,
  +before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Packing.offset_base(state))) == True{} : Bool},+inside: {U32.is_lt(address,capacity) == True{} : Bool}) ->
  {Tree.read(U32,offset_tree(Packing.pack_selected(False{},phase,layout,state)),address) == Tree.read(U32,offset_tree(state),address) : U32}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Packing.Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
  +query = {Offsets.Query{Packing.phase_table(phase,layout,offset_base),U32.to_nat(inner),0n,depth,capacity} : Offsets.Query}
  Offsets.read_before(query,address,offsets,evidence,before,inside)

def later_weight(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  +inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == False{} : Bool}) ->
  {Tree.read(Tile.Tile8,weight_tree(Packing.pack_selected(False{},Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state)),weight_address(query)) ==
    Tree.read(Tile.Tile8,weight_tree(state),weight_address(query)) : Tile.Tile8}:
  Frame{source_equal,+weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  +selection = weight_selection(query,facts)
  +before = {
    %Equal.sym(U32,Packing.weight_base(state),Bounds.prefix_word(weight_bounds(query),index),weight_base) :
      {Nat.is_lt(U32.to_nat(weight_address(query)),U32.to_nat(_)) == True{} : Bool}
    selection_before(weight_bounds(query),phase(query),slot(query),index,weight_capacity(query),selection,flagged_inside(True{},query,index,inside),after) :
      {Nat.is_lt(U32.to_nat(weight_address(query)),U32.to_nat(Packing.weight_base(state))) == True{} : Bool}}
  copied_before(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state,weight_address(query),weight_depth(query),weight_capacity(query),
    later_copy_frame(query,index,state,evidence,inside,active),before,selection_inside(weight_bounds(query),phase(query),slot(query),weight_capacity(query),selection))

def later_offset(+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  +inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == False{} : Bool}) ->
  {Tree.read(U32,offset_tree(Packing.pack_selected(False{},Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state)),offset_address(query)) ==
    Tree.read(U32,offset_tree(state),offset_address(query)) : U32}:
  Frame{source_equal,weight_base,+offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  +selection = offset_selection(query,facts)
  +before = {
    %Equal.sym(U32,Packing.offset_base(state),Bounds.prefix_word(offset_bounds(query),index),offset_base) :
      {Nat.is_lt(U32.to_nat(offset_address(query)),U32.to_nat(_)) == True{} : Bool}
    selection_before(offset_bounds(query),phase(query),tap(query),index,offset_capacity(query),selection,flagged_inside(False{},query,index,inside),after) :
      {Nat.is_lt(U32.to_nat(offset_address(query)),U32.to_nat(Packing.offset_base(state))) == True{} : Bool}}
  copied_offsets_before(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state,offset_address(query),offset_depth(query),offset_capacity(query),
    later_table_frame(query,index,state,evidence,inside,active),before,selection_inside(offset_bounds(query),phase(query),tap(query),offset_capacity(query),selection))

def later_selected(empty: Bool,+query: Query,+index: Nat,+state: Packing.Packed,+evidence: Frame<query,index,state>,+after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  +inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  compared: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0) == empty : Bool}) ->
  {read(query,Packing.pack_selected(empty,Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state)) == read(query,state) : Value}:
  match empty:
    case True{}: {==}
    case False{}:
      +active = compared
      +next = Packing.pack_selected(False{},Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index)),layout(query),state)
      +weight = Tree.read(Tile.Tile8,weight_tree(next),weight_address(query))
      +offset = Tree.read(U32,offset_tree(next),offset_address(query))
      +old_weight = Tree.read(Tile.Tile8,weight_tree(state),weight_address(query))
      +old_offset = Tree.read(U32,offset_tree(state),offset_address(query))
      Equal.trans(Value,Value{weight,offset},Value{old_weight,offset},Value{old_weight,old_offset},
        Equal.cong(Tile.Tile8,Value,value => Value{value,offset},weight,old_weight,later_weight(query,index,state,evidence,after,inside,active)),
        Equal.cong(U32,Value,value => Value{old_weight,value},offset,old_offset,later_offset(query,index,state,evidence,after,inside,active)))

def later(+query: Query,+index: Nat,+state: Packing.Packed,evidence: Frame<query,index,state>,after: {Nat.is_lt(phase(query),index) == True{} : Bool},
  inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) -> {read(query,update(U32.from_nat(index),query,state)) == read(query,state) : Value}:
  Frame{+source_equal,+weight_base,+offset_base,+weight_capacity_equal,+weight_balanced,+weight_depth_ok,+offset_capacity_equal,+offset_balanced,+offset_depth_ok,+facts} = evidence
  +evidence = {Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,facts} : Frame<query,index,state>}
  +after = after
  +inside = inside
  later_selected(U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds(query)),U32.from_nat(index))),0),query,index,state,evidence,after,inside,{==})

def readback(+query: Query,+state: Packing.Packed,+evidence: Frame<query,0n,state>) ->
  {read(query,Traversal.indexed_fold(~Packing.Packed,~Query,~update,count(query),0,query,state)) == wanted(query) : Value}:
  Frame{source_equal,weight_base,offset_base,weight_capacity_equal,weight_balanced,weight_depth_ok,offset_capacity_equal,offset_balanced,offset_depth_ok,+facts} = evidence
  Facts{+phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room} = facts
  Readback.indexed_complete(~Packing.Packed,~Query,~Value,~count,~phase,~wanted,~frame,~read,~update,~frame_step,~produced,~later,query,state,evidence,phase_inside)

def step_same(+index: U32,+query: Query,+state: Packing.Packed) ->
  {update(index,query,state) == Packing.pack_phase(index,context(query),state) : Packing.Packed}:
  {==}

def folded(+query: Query,+state: Packing.Packed) ->
  {Traversal.indexed_fold(~Packing.Packed,~Query,~update,count(query),0,query,state) ==
    Traversal.indexed_fold(~Packing.Packed,~Kernel.PackContext,~Packing.pack_phase,count(query),0,context(query),state) : Packing.Packed}:
  Arrays.indexed_fold_change_context(~Packing.Packed,~Query,~Kernel.PackContext,~context,~update,~Packing.pack_phase,~step_same,count(query),0,query,state)

def complete(query: Query,state: Packing.Packed) -> Packing.Packed:
  +query = query
  Traversal.indexed_fold(~Packing.Packed,~Kernel.PackContext,~Packing.pack_phase,count(query),0,context(query),state)

def complete_readback(+query: Query,+state: Packing.Packed,+evidence: Frame<query,0n,state>) ->
  {read(query,complete(query,state)) == wanted(query) : Value}:
  %folded(query,state) : {read(query,_) == wanted(query) : Value}
  readback(query,state,evidence)

def complete_matches(+query: Query,+state: Packing.Packed) ->
  {Traversal.indexed_fold(~Kernel.Packed,~Kernel.PackContext,~Kernel.pack_phase,count(query),0,context(query),Packing.represent(state)) == Packing.represent(complete(query,state)) : Kernel.Packed}:
  Packing.packing_matches(count(query),0,context(query),state)

def total(query: Bounds.Query) -> U32:
  +query = query
  Bounds.prefix_word(query,Bounds.count(query))
def allocated_depth(query: Bounds.Query) -> Nat:
  Tensors.depth_for((total(query) + 1 : U32))
def allocated_capacity(query: Bounds.Query) -> U32:
  U32.shln(1,allocated_depth(query))

type Allocation<-query: Bounds.Query> is Data:
  Allocation{depth_ok: {Nat.is_le(allocated_depth(query),31n) == True{} : Bool},capacity_ok: {U32.is_le(allocated_capacity(query),2147483648) == True{} : Bool},
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

def allocation_finish(+query: Bounds.Query,evidence: Depth.Guarded<total(query)>) -> Allocation<query>:
  Depth.Guarded{+depth_ok,room} = evidence
  Allocation{depth_ok,Power.native_power_bound(allocated_depth(query),depth_ok),
    %Bounds.prefix_exact(query,Bounds.count(query),Order.reflexive(Bounds.count(query))) :
      {Nat.is_le(_,U32.to_nat(allocated_capacity(query))) == True{} : Bool}
      Words.strict_le(U32.to_nat(total(query)),U32.to_nat(allocated_capacity(query)),Words.as_nat_lt(total(query),allocated_capacity(query),room))}

def allocation(+query: Bounds.Query) -> Allocation<query>:
  allocation_finish(query,Depth.guarded(total(query),total_below_half(query)))

def allocated_query(bounds: Bounds.Query,layout: Kernel.Tiling,source: Buffer.Model<F32>,phase: Nat,slot: Nat,tap: Nat) -> Query:
  +bounds = bounds
  +weights = reflag(True{},bounds)
  +offsets = reflag(False{},bounds)
  Query{bounds,layout,source,phase,slot,tap,allocated_depth(weights),allocated_capacity(weights),allocated_depth(offsets),allocated_capacity(offsets)}

def initial(query: Query) -> Packing.Packed:
  +query = query
  Packing.Packed{source(query),Tree.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0)),Tree.fresh(U32,offset_depth(query),0),0,0}

def allocated_frame_finish(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat,
  weights_ready: Allocation<reflag(True{},bounds)>,offsets_ready: Allocation<reflag(False{},bounds)>,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  Frame<allocated_query(bounds,layout,source,phase,slot,tap),0n,initial(allocated_query(bounds,layout,source,phase,slot,tap))>:
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  Allocation{weight_depth_ok,+weight_capacity_ok,weight_room} = weights_ready
  Allocation{offset_depth_ok,+offset_capacity_ok,offset_room} = offsets_ready
  +query = allocated_query(bounds,layout,source,phase,slot,tap)
  Frame{{==},{==},{==},Shape.capacity(Tile.Tile8,weight_depth(query),Tree.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0)),Shape.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0))),
    Shape.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0)),weight_depth_ok,
    Shape.capacity(U32,offset_depth(query),Tree.fresh(U32,offset_depth(query),0),Shape.fresh(U32,offset_depth(query),0)),Shape.fresh(U32,offset_depth(query),0),offset_depth_ok,
    Facts{phase_inside,slot_inside,tap_inside,weight_capacity_ok,offset_capacity_ok,weight_room,offset_room}}

def allocated_frame(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  Frame<allocated_query(bounds,layout,source,phase,slot,tap),0n,initial(allocated_query(bounds,layout,source,phase,slot,tap))>:
  allocated_frame_finish(bounds,layout,source,phase,slot,tap,allocation(reflag(True{},bounds)),allocation(reflag(False{},bounds)),phase_inside,slot_inside,tap_inside)

def allocated_readback(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  {read(allocated_query(bounds,layout,source,phase,slot,tap),complete(allocated_query(bounds,layout,source,phase,slot,tap),initial(allocated_query(bounds,layout,source,phase,slot,tap)))) ==
    wanted(allocated_query(bounds,layout,source,phase,slot,tap)) : Value}:
  +query = allocated_query(bounds,layout,source,phase,slot,tap)
  complete_readback(query,initial(query),allocated_frame(bounds,layout,source,phase,slot,tap,phase_inside,slot_inside,tap_inside))

def weight_total_program(+bounds: Bounds.Query) ->
  {total(reflag(True{},bounds)) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,Kernel.window_jobs(Bounds.geometry(bounds)),0,Bounds.geometry(bounds),0) : U32}:
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  %Equal.sym(Nat,Kernel.window_jobs(g),Bounds.count(bounds),Bounds.window_jobs_exact(bounds)) :
    {total(reflag(True{},bounds)) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,_,0,g,0) : U32}
  Bounds.prefix_program(reflag(True{},bounds),Bounds.count(bounds))

def offset_total_program(+bounds: Bounds.Query) ->
  {total(reflag(False{},bounds)) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,Kernel.window_jobs(Bounds.geometry(bounds)),0,Bounds.geometry(bounds),0) : U32}:
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  %Equal.sym(Nat,Kernel.window_jobs(g),Bounds.count(bounds),Bounds.window_jobs_exact(bounds)) :
    {total(reflag(False{},bounds)) == Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,_,0,g,0) : U32}
  Bounds.prefix_program(reflag(False{},bounds),Bounds.count(bounds))

def native_initial(bounds: Bounds.Query,layout: Kernel.Tiling,source: Buffer.Model<F32>) -> Kernel.Packed:
  +bounds = bounds
  +geometry = Bounds.geometry(bounds)
  +phases = Kernel.window_jobs(geometry)
  +weight_tiles = Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,phases,0,geometry,0)
  +entries = Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,phases,0,geometry,0)
  Kernel.Packed{Buffer.pack(~F32,source),NativeAllocation.array(~Tile.Tile8,Tensors.depth_for((weight_tiles + 1 : U32)),Tile.filled(0.0)),
    NativeAllocation.array(~U32,Tensors.depth_for((entries + 1 : U32)),0),0,0}

def initial_matches(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat) ->
  {native_initial(bounds,layout,source) == Packing.represent(initial(allocated_query(bounds,layout,source,phase,slot,tap))) : Kernel.Packed}:
  Bounds.Query{weights,call} = bounds
  Domain.Call{g,batch,limits,facts} = call
  +query = allocated_query(bounds,layout,source,phase,slot,tap)
  +weight_tiles = total(reflag(True{},bounds))
  +entries = total(reflag(False{},bounds))
  %weight_total_program(bounds) :
    {Kernel.Packed{Buffer.pack(~F32,source),NativeAllocation.array(~Tile.Tile8,Tensors.depth_for((_ + 1 : U32)),Tile.filled(0.0)),
      NativeAllocation.array(~U32,Tensors.depth_for((Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,Kernel.window_jobs(g),0,g,0) + 1 : U32)),0),0,0} == Packing.represent(initial(query)) : Kernel.Packed}
  %offset_total_program(bounds) :
    {Kernel.Packed{Buffer.pack(~F32,source),NativeAllocation.array(~Tile.Tile8,weight_depth(query),Tile.filled(0.0)),
      NativeAllocation.array(~U32,Tensors.depth_for((_ + 1 : U32)),0),0,0} == Packing.represent(initial(query)) : Kernel.Packed}
  %Equal.sym(Array<Tile.Tile8>,NativeAllocation.array(~Tile.Tile8,weight_depth(query),Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0))),Allocate.fresh(~Tile.Tile8,weight_depth(query),Tile.filled(0.0))) :
    {Kernel.Packed{Buffer.pack(~F32,source),_,NativeAllocation.array(~U32,offset_depth(query),0),0,0} == Packing.represent(initial(query)) : Kernel.Packed}
  %Equal.sym(Array<U32>,NativeAllocation.array(~U32,offset_depth(query),0),Tree.pack(U32,Tree.fresh(U32,offset_depth(query),0)),Allocate.fresh(~U32,offset_depth(query),0)) :
    {Kernel.Packed{Buffer.pack(~F32,source),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,weight_depth(query),Tile.filled(0.0))),_,0,0} == Packing.represent(initial(query)) : Kernel.Packed}
  {==}

def native_complete(bounds: Bounds.Query,layout: Kernel.Tiling,source: Buffer.Model<F32>) -> Kernel.Packed:
  +bounds = bounds
  +layout = layout
  Traversal.indexed_fold(~Kernel.Packed,~Kernel.PackContext,~Kernel.pack_phase,Kernel.window_jobs(Bounds.geometry(bounds)),0,Kernel.PackContext{Bounds.geometry(bounds),layout},native_initial(bounds,layout,source))

def native_complete_matches(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat) ->
  {native_complete(bounds,layout,source) == Packing.represent(complete(allocated_query(bounds,layout,source,phase,slot,tap),initial(allocated_query(bounds,layout,source,phase,slot,tap)))) : Kernel.Packed}:
  +query = allocated_query(bounds,layout,source,phase,slot,tap)
  %Equal.sym(Nat,Kernel.window_jobs(Bounds.geometry(bounds)),Bounds.count(bounds),Bounds.window_jobs_exact(bounds)) :
    {Traversal.indexed_fold(~Kernel.Packed,~Kernel.PackContext,~Kernel.pack_phase,_,0,context(query),native_initial(bounds,layout,source)) == Packing.represent(complete(query,initial(query))) : Kernel.Packed}
  %Equal.sym(Kernel.Packed,native_initial(bounds,layout,source),Packing.represent(initial(query)),initial_matches(bounds,layout,source,phase,slot,tap)) :
    {Traversal.indexed_fold(~Kernel.Packed,~Kernel.PackContext,~Kernel.pack_phase,Bounds.count(bounds),0,context(query),_) == Packing.represent(complete(query,initial(query))) : Kernel.Packed}
  complete_matches(query,initial(query))

def native_read_pair(source: Storage.Buffer<F32>,weight_base: U32,offset_base: U32,weight: Array<Tile.Tile8> & Tile.Tile8,offset: Array<U32> & U32) -> Kernel.Packed & Value:
  (weights,weight_value) = weight
  (offsets,offset_value) = offset
  (Kernel.Packed{source,weights,offsets,weight_base,offset_base},Value{weight_value,offset_value})

def native_read(state: Kernel.Packed,weight_address: U32,offset_address: U32) -> Kernel.Packed & Value:
  Kernel.Packed{source,weights,offsets,weight_base,offset_base} = state
  native_read_pair(source,weight_base,offset_base,Array.get(Tile.Tile8,weights,weight_address),Array.get(U32,offsets,offset_address))

def native_read_matches(+query: Query,+state: Packing.Packed) ->
  {native_read(Packing.represent(state),weight_address(query),offset_address(query)) == (Packing.represent(state),read(query,state)) : Kernel.Packed & Value}:
  Packing.Packed{+source,+weights,+offsets,weight_base,offset_base} = state
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,weights),weight_address(query)),(Tree.pack(Tile.Tile8,weights),Tree.read(Tile.Tile8,weights,weight_address(query))),NativeStorage.read(Tile.Tile8,weights,weight_address(query))) :
    {native_read_pair(Buffer.pack(~F32,source),weight_base,offset_base,_,Array.get(U32,Tree.pack(U32,offsets),offset_address(query))) == (Packing.represent(state),read(query,state)) : Kernel.Packed & Value}
  %Equal.sym(Array<U32> & U32,Array.get(U32,Tree.pack(U32,offsets),offset_address(query)),(Tree.pack(U32,offsets),Tree.read(U32,offsets,offset_address(query))),NativeStorage.read(U32,offsets,offset_address(query))) :
    {native_read_pair(Buffer.pack(~F32,source),weight_base,offset_base,(Tree.pack(Tile.Tile8,weights),Tree.read(Tile.Tile8,weights,weight_address(query))),_) == (Packing.represent(state),read(query,state)) : Kernel.Packed & Value}
  {==}

def allocated_native_readback(+bounds: Bounds.Query,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+phase: Nat,+slot: Nat,+tap: Nat,
  phase_inside: {Nat.is_lt(phase,Bounds.count(bounds)) == True{} : Bool},
  slot_inside: {Nat.is_lt(slot,U32.to_nat(Packing.phase_count(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Kernel.phase_of(Bounds.geometry(bounds),U32.from_nat(phase))))) == True{} : Bool}) ->
  {native_read(native_complete(bounds,layout,source),weight_address(allocated_query(bounds,layout,source,phase,slot,tap)),offset_address(allocated_query(bounds,layout,source,phase,slot,tap))) ==
    (Packing.represent(complete(allocated_query(bounds,layout,source,phase,slot,tap),initial(allocated_query(bounds,layout,source,phase,slot,tap)))),wanted(allocated_query(bounds,layout,source,phase,slot,tap))) : Kernel.Packed & Value}:
  +query = allocated_query(bounds,layout,source,phase,slot,tap)
  +final = complete(query,initial(query))
  %Equal.sym(Kernel.Packed,native_complete(bounds,layout,source),Packing.represent(final),native_complete_matches(bounds,layout,source,phase,slot,tap)) :
    {native_read(_,weight_address(query),offset_address(query)) == (Packing.represent(final),wanted(query)) : Kernel.Packed & Value}
  Equal.trans(Kernel.Packed & Value,native_read(Packing.represent(final),weight_address(query),offset_address(query)),(Packing.represent(final),read(query,final)),(Packing.represent(final),wanted(query)),
    native_read_matches(query,final),Equal.cong(Value,Kernel.Packed & Value,value => (Packing.represent(final),value),read(query,final),wanted(query),
      allocated_readback(bounds,layout,source,phase,slot,tap,phase_inside,slot_inside,tap_inside)))
