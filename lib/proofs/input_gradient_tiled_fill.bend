# Every selected slot in the actual group traversal retains its spatial tile.
import Base
import ../tensor.bend as Tensors
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_refinement.bend as Execution
import ./input_gradient_tiled_tiles.bend as Tiles
import ./input_gradient_tiled_bounds.bend as Groups
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./indexed_output_readback.bend as OutputReadback
import ./traversal_array_refinement.bend as Arrays
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as Division
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison

type Query is Data:
  Query{context: Kernel.Filling,input: Tree.Tree<F32>,rows: Nat,position: Nat,channel: Nat,shift: Nat,depth: Nat,capacity: U32}

def filling(query: Query) -> Kernel.Filling:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  context
def original(query: Query) -> Tree.Tree<F32>:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  input
def rows(query: Query) -> Nat:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  rows
def position(query: Query) -> Nat:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  position
def channel(query: Query) -> Nat:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  channel
def shift(query: Query) -> Nat:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  shift
def depth(query: Query) -> Nat:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  depth
def capacity(query: Query) -> U32:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  capacity
def span(context: Kernel.Filling) -> U32:
  Kernel.Filling{layout,low,base,plane,first,rows,group,span} = context
  span
def groups(context: Kernel.Filling) -> U32:
  Kernel.Filling{Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows},low,base,plane,first,rows,group,span} = context
  chunk_groups
def shifts(context: Kernel.Filling) -> U32:
  Kernel.Filling{Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows},low,base,plane,first,rows,group,span} = context
  shifts
def tiling_outputs(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  outputs
def outputs(context: Kernel.Filling) -> U32:
  Kernel.Filling{layout,low,base,plane,first,rows,group,span} = context
  tiling_outputs(layout)
def block(+context: Kernel.Filling) -> U32:
  (outputs(context) * shifts(context) + 1 : U32)
def block_size(+context: Kernel.Filling) -> Nat:
  Nat.add(Nat.mul(U32.to_nat(outputs(context)),U32.to_nat(shifts(context))),1n)
def count(query: Query) -> Nat:
  +query = query
  Nat.mul(rows(query),U32.to_nat(span(filling(query))))
def tile_count(+query: Query) -> Nat:
  Nat.mul(Nat.mul(rows(query),U32.to_nat(groups(filling(query)))),block_size(filling(query)))
def at_group(query: Query,index: Nat) -> Tiles.ChannelQuery:
  Query{+context,input,rows,position,channel,shift,depth,capacity} = query
  Tiles.ChannelQuery{Tiles.group_channels(U32.from_nat(index),context,input),U32.to_nat(outputs(context)),channel,shift,depth,capacity}
def selected_channels(query: Query) -> Tiles.ChannelQuery:
  +query = query
  at_group(query,position(query))
def address(query: Query) -> U32:
  Tiles.channel_address(selected_channels(query))
def wanted(query: Query) -> Tile.Tile8:
  Tiles.channel_wanted(selected_channels(query))
def tree(state: Execution.Owners) -> Tree.Tree<Tile.Tile8>:
  Execution.Owners{input,tiles} = state
  tiles
def read(query: Query,state: Execution.Owners) -> Tile.Tile8:
  Tree.read(Tile.Tile8,tree(state),address(query))
def update(index: U32,query: Query,state: Execution.Owners) -> Execution.Owners:
  Execution.fill_group(index,filling(query),state)

type Facts<-query: Query> is Data:
  Facts{capacity_ok: {U32.is_le(capacity(query),2147483648) == True{} : Bool},
    span_positive: {Nat.is_lt(0n,U32.to_nat(span(filling(query)))) == True{} : Bool},
    span_ordered: {Nat.is_le(U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query)))) == True{} : Bool},
    double_span: {Nat.is_le(Nat.double(U32.to_nat(span(filling(query)))),U32.to_nat(capacity(query))) == True{} : Bool},
    room: {Nat.is_le(tile_count(query),U32.to_nat(capacity(query))) == True{} : Bool},
    position_inside: {Nat.is_lt(position(query),count(query)) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel(query),U32.to_nat(outputs(filling(query)))) == True{} : Bool},
    shift_inside: {Nat.is_lt(shift(query),U32.to_nat(shifts(filling(query)))) == True{} : Bool}}

type Frame<-query: Query,-state: Execution.Owners> is Data:
  Frame{source: {Execution.source(state) == original(query) : Tree.Tree<F32>},
    capacity_equal: {Tree.capacity(Tile.Tile8,tree(state)) == capacity(query) : U32},
    balanced: Shape.Balanced(Tile.Tile8,depth(query),tree(state)),depth_ok: {Nat.is_le(depth(query),31n) == True{} : Bool},facts: Facts<query>}

def frame(query: Query,state: Execution.Owners) -> Data:
  Frame<query,state>

def frame_step(+query: Query,+index: U32,+state: Execution.Owners,evidence: Frame<query,state>) -> Frame<query,update(index,query,state)>:
  Query{+context,+input,rows,position,channel,shift,+depth,+capacity} = query
  Execution.Owners{+source,+tiles} = state
  Frame{source_equal,capacity_equal,balanced,depth_ok,facts} = evidence
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Tiles.Channels,~Tiles.channel_update,Tiles.group_outputs(context),0,Tiles.group_channels(index,context,source),tiles)
  %Equal.sym(Execution.Owners,Execution.fill_group(index,context,state),Execution.Owners{source,written},Tiles.group_written(index,context,source,tiles)) : Frame<query,_>
  Frame{source_equal,Equal.trans(U32,Tree.capacity(Tile.Tile8,written),Tree.capacity(Tile.Tile8,tiles),capacity,
    Tiles.channels_capacity(Tiles.group_outputs(context),Tiles.group_channels(index,context,source),tiles),capacity_equal),
    Tiles.channels_balanced(Tiles.group_outputs(context),Tiles.group_channels(index,context,source),tiles,depth,balanced),depth_ok,facts}

def group_count(+query: Query) -> Nat:
  Nat.mul(rows(query),U32.to_nat(groups(filling(query))))

def block_positive(+context: Kernel.Filling) -> {Nat.is_lt(0n,block_size(context)) == True{} : Bool}:
  +product = Nat.mul(U32.to_nat(outputs(context)),U32.to_nat(shifts(context)))
  Order.le_lt_trans(0n,product,block_size(context),Order.zero_le(product),Order.below_sum(product,0n))

def groups_room(+query: Query,+facts: Facts<query>) -> {Nat.is_le(group_count(query),U32.to_nat(capacity(query))) == True{} : Bool}:
  Facts{capacity_ok,span_positive,span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  Order.transitive(group_count(query),tile_count(query),U32.to_nat(capacity(query)),RowMajor.product_covers(group_count(query),block_size(filling(query)),block_positive(filling(query))),room)

def block_room(+query: Query,+facts: Facts<query>) -> {Nat.is_le(block_size(filling(query)),U32.to_nat(capacity(query))) == True{} : Bool}:
  Facts{capacity_ok,span_positive,span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  +count_positive = Order.positive_bound(position(query),count(query),position_inside)
  +groups_positive = Bounds.nat_trans(0n,count(query),group_count(query),count_positive,
    Division.multiply_left_le(rows(query),U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query))),span_ordered))
  Order.transitive(block_size(filling(query)),tile_count(query),U32.to_nat(capacity(query)),Division.le_multiple(group_count(query),block_size(filling(query)),groups_positive),room)

def block_exact(+query: Query,+facts: Facts<query>) -> {U32.to_nat(block(filling(query))) == block_size(filling(query)) : Nat}:
  Facts{+capacity_ok,span_positive,span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  Arithmetic.multiply_add_exact(outputs(filling(query)),shifts(filling(query)),1,capacity(query),capacity_ok,block_room(query,facts))

def group_base_exact(+query: Query,+index: Nat,+facts: Facts<query>,inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {U32.to_nat(Tiles.channel_destination(Tiles.group_channels(U32.from_nat(index),filling(query),original(query)))) ==
    Nat.mul(Groups.group_position(index,U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query)))),block_size(filling(query))) : Nat}:
  Query{context,input,+rows,position,channel,shift,depth,+capacity} = query
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,chunk_rows},low,base,plane,first,source_rows,group,+span} = context
  Facts{+capacity_ok,+span_positive,+span_ordered,+double_span,+room,position_inside,channel_inside,shift_inside} = facts
  +size_exact = block_exact(query,facts)
  +size_positive = {
    %Equal.sym(Nat,U32.to_nat(block(context)),block_size(context),size_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    block_positive(context) : {Nat.is_lt(0n,U32.to_nat(block(context))) == True{} : Bool}}
  %size_exact : {U32.to_nat(((U32.from_nat(index) / span * chunk_groups + U32.from_nat(index) % span) * block(context) : U32)) ==
    Nat.mul(Groups.group_position(index,U32.to_nat(span),U32.to_nat(chunk_groups)),_) : Nat}
  Groups.group_address_exact(index,span,chunk_groups,block(context),rows,capacity,capacity_ok,span_positive,span_ordered,size_positive,double_span,
    %Equal.sym(Nat,U32.to_nat(block(context)),block_size(context),size_exact) : {Nat.is_le(Nat.mul(Nat.mul(rows,U32.to_nat(chunk_groups)),_),U32.to_nat(capacity)) == True{} : Bool} room,inside)

def group_channel_room(+query: Query,+index: Nat,+facts: Facts<query>,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Tiles.channel_destination(Tiles.channel_context(at_group(query,index)))),Nat.mul(U32.to_nat(outputs(filling(query))),U32.to_nat(shifts(filling(query))))),U32.to_nat(capacity(query))) == True{} : Bool}:
  Query{+context,input,+rows,position,channel,shift,depth,+capacity} = query
  Facts{capacity_ok,+span_positive,+span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  +group = Groups.group_position(index,U32.to_nat(span(context)),U32.to_nat(groups(context)))
  +product = Nat.mul(U32.to_nat(outputs(context)),U32.to_nat(shifts(context)))
  %Equal.sym(Nat,U32.to_nat(Tiles.channel_destination(Tiles.channel_context(at_group(query,index)))),Nat.mul(group,block_size(context)),group_base_exact(query,index,facts,inside)) :
    {Nat.is_le(Nat.add(_,product),U32.to_nat(capacity)) == True{} : Bool}
  Bounds.strict_le(Nat.add(Nat.mul(group,block_size(context)),product),U32.to_nat(capacity),
    Bounds.nat_trans(Nat.add(Nat.mul(group,block_size(context)),product),tile_count(query),U32.to_nat(capacity),
      RowMajor.radix(group,group_count(query),product,block_size(context),Groups.group_inside(index,rows,U32.to_nat(span(context)),U32.to_nat(groups(context)),span_positive,span_ordered,inside),
        Order.below_sum(product,0n)),room))

def channels_frame_at(+query: Query,+index: Nat,+state: Execution.Owners,+evidence: Frame<query,state>,inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  Tiles.ChannelFrame<at_group(query,index),tree(state)>:
  Query{context,input,rows,position,channel,shift,depth,capacity} = query
  Kernel.Filling{Kernel.Tiling{layout_groups,+shift_count,+output_count,column_low,kernel,height,width,chunk_groups,chunk_rows},low,source_base,source_plane,source_first,source_rows,source_group,group_span} = context
  Frame{source,capacity_equal,balanced,depth_ok,+facts} = evidence
  Facts{capacity_ok,span_positive,span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  Tiles.ChannelFrame{capacity_equal,balanced,depth_ok,capacity_ok,group_channel_room(query,index,facts,inside),channel_inside,shift_inside}

def at_target(+query: Query,+state: Execution.Owners,evidence: Frame<query,state>) ->
  {read(query,update(U32.from_nat(position(query)),query,state)) == wanted(query) : Tile.Tile8}:
  Query{+context,+input,rows,+position,channel,shift,depth,capacity} = query
  Kernel.Filling{Kernel.Tiling{layout_groups,+shift_count,+output_count,column_low,kernel,height,width,chunk_groups,chunk_rows},low,source_base,source_plane,source_first,source_rows,source_group,group_span} = context
  Execution.Owners{+source,+tiles} = state
  Frame{+source_equal,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
  Facts{capacity_ok,span_positive,span_ordered,double_span,room,+position_inside,channel_inside,shift_inside} = facts
  +evidence = {Frame{source_equal,capacity_equal,balanced,depth_ok,facts} : Frame<query,state>}
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Tiles.Channels,~Tiles.channel_update,Tiles.group_outputs(context),0,Tiles.group_channels(U32.from_nat(position),context,source),tiles)
  %Equal.sym(Execution.Owners,update(U32.from_nat(position),query,state),Execution.Owners{source,written},Tiles.group_written(U32.from_nat(position),context,source,tiles)) :
    {read(query,_) == wanted(query) : Tile.Tile8}
  %Equal.sym(Tree.Tree<F32>,source,input,source_equal) :
    {Tree.read(Tile.Tile8,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Tiles.Channels,~Tiles.channel_update,Tiles.group_outputs(context),0,Tiles.group_channels(U32.from_nat(position),context,_),tiles),address(query)) == wanted(query) : Tile.Tile8}
  Tiles.channels_readback(selected_channels(query),tiles,channels_frame_at(query,position,state,evidence,position_inside))

def address_exact(+query: Query,+state: Execution.Owners,+evidence: Frame<query,state>) ->
  {U32.to_nat(address(query)) == Nat.add(Nat.mul(Groups.group_position(position(query),U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query)))),block_size(filling(query))),
    Nat.add(Nat.mul(channel(query),U32.to_nat(shifts(filling(query)))),shift(query))) : Nat}:
  Query{+context,input,rows,+position,+channel,+shift,depth,capacity} = query
  Kernel.Filling{Kernel.Tiling{layout_groups,+shift_count,+output_count,column_low,kernel,height,width,chunk_groups,chunk_rows},low,source_base,source_plane,source_first,source_rows,source_group,group_span} = context
  Frame{source,capacity_equal,balanced,depth_ok,+facts} = evidence
  Facts{+capacity_ok,span_positive,span_ordered,double_span,room,+position_inside,+channel_inside,+shift_inside} = facts
  +base = Tiles.channel_destination(Tiles.channel_context(selected_channels(query)))
  +content = Nat.add(Nat.mul(channel,U32.to_nat(shifts(context))),shift)
  Equal.trans(Nat,U32.to_nat(address(query)),Nat.add(U32.to_nat(base),content),
    Nat.add(Nat.mul(Groups.group_position(position,U32.to_nat(span(context)),U32.to_nat(groups(context))),block_size(context)),content),
    Tiles.channel_address_exact(selected_channels(query),capacity_ok,group_channel_room(query,position,facts,position_inside),channel_inside,shift_inside),
    Equal.cong(Nat,Nat,base => Nat.add(base,content),U32.to_nat(base),Nat.mul(Groups.group_position(position,U32.to_nat(span(context)),U32.to_nat(groups(context))),block_size(context)),group_base_exact(query,position,facts,position_inside)))

def later(+query: Query,+index: Nat,+state: Execution.Owners,evidence: Frame<query,state>,
  after: {Nat.is_lt(position(query),index) == True{} : Bool},inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {read(query,update(U32.from_nat(index),query,state)) == read(query,state) : Tile.Tile8}:
  Query{+context,+input,+rows,+position,+channel,+shift,depth,+capacity} = query
  Kernel.Filling{Kernel.Tiling{layout_groups,+shift_count,+output_count,column_low,kernel,height,width,chunk_groups,chunk_rows},low,source_base,source_plane,source_first,source_rows,source_group,group_span} = context
  Execution.Owners{+source,+tiles} = state
  Frame{+source_equal,+capacity_equal,+balanced,+depth_ok,+facts} = evidence
  Facts{+capacity_ok,+span_positive,+span_ordered,double_span,+room,+position_inside,+channel_inside,+shift_inside} = facts
  +evidence = {Frame{source_equal,capacity_equal,balanced,depth_ok,facts} : Frame<query,state>}
  +inside = inside
  +group = Groups.group_position(position,U32.to_nat(span(context)),U32.to_nat(groups(context)))
  +later_group = Groups.group_position(index,U32.to_nat(span(context)),U32.to_nat(groups(context)))
  +product = Nat.mul(U32.to_nat(outputs(context)),U32.to_nat(shifts(context)))
  +content = Nat.add(Nat.mul(channel,U32.to_nat(shifts(context))),shift)
  +content_inside = RowMajor.radix(channel,U32.to_nat(outputs(context)),shift,U32.to_nat(shifts(context)),channel_inside,shift_inside)
  +content_block = Bounds.nat_trans(content,product,block_size(context),content_inside,Order.le_add(product,1n))
  +observed_inside = {
    %Equal.sym(Bool,U32.is_lt(address(query),capacity),Nat.is_lt(U32.to_nat(address(query)),U32.to_nat(capacity)),Comparison.lt(address(query),capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(address(query)),Nat.add(Nat.mul(group,block_size(context)),content),address_exact(query,state,evidence)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.nat_trans(Nat.add(Nat.mul(group,block_size(context)),content),tile_count(query),U32.to_nat(capacity),
      RowMajor.radix(group,group_count(query),content,block_size(context),Groups.group_inside(position,rows,U32.to_nat(span(context)),U32.to_nat(groups(context)),span_positive,span_ordered,position_inside),content_block),room) :
      {U32.is_lt(address(query),capacity) == True{} : Bool}}
  +before = {
    %Equal.sym(Nat,U32.to_nat(address(query)),Nat.add(Nat.mul(group,block_size(context)),content),address_exact(query,state,evidence)) :
      {Nat.is_lt(_,U32.to_nat(Tiles.channel_destination(Tiles.channel_context(at_group(query,index))))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Tiles.channel_destination(Tiles.channel_context(at_group(query,index)))),Nat.mul(later_group,block_size(context)),group_base_exact(query,index,facts,inside)) :
      {Nat.is_lt(Nat.add(Nat.mul(group,block_size(context)),content),_) == True{} : Bool}
    RowMajor.radix(group,later_group,content,block_size(context),Groups.group_increasing(position,index,U32.to_nat(span(context)),U32.to_nat(groups(context)),span_positive,span_ordered,after),content_block) :
      {Nat.is_lt(U32.to_nat(address(query)),U32.to_nat(Tiles.channel_destination(Tiles.channel_context(at_group(query,index))))) == True{} : Bool}}
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Tiles.Channels,~Tiles.channel_update,Tiles.group_outputs(context),0,Tiles.group_channels(U32.from_nat(index),context,source),tiles)
  %Equal.sym(Execution.Owners,update(U32.from_nat(index),query,state),Execution.Owners{source,written},Tiles.group_written(U32.from_nat(index),context,source,tiles)) :
    {read(query,_) == read(query,state) : Tile.Tile8}
  %Equal.sym(Tree.Tree<F32>,source,input,source_equal) :
    {Tree.read(Tile.Tile8,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Tiles.Channels,~Tiles.channel_update,Tiles.group_outputs(context),0,Tiles.group_channels(U32.from_nat(index),context,_),tiles),address(query)) == read(query,state) : Tile.Tile8}
  Tiles.channels_read_before(at_group(query,index),address(query),tiles,channels_frame_at(query,index,state,evidence,inside),observed_inside,before)

def same_step(+index: U32,+query: Query,+state: Execution.Owners) ->
  {update(index,query,state) == Execution.fill_group(index,filling(query),state) : Execution.Owners}:
  {==}

def readback(+query: Query,+state: Execution.Owners,+evidence: Frame<query,state>) ->
  {read(query,Traversal.indexed_fold(~Execution.Owners,~Kernel.Filling,~Execution.fill_group,count(query),0,filling(query),state)) == wanted(query) : Tile.Tile8}:
  Frame{source,capacity_equal,balanced,depth_ok,+facts} = evidence
  Facts{capacity_ok,span_positive,span_ordered,double_span,room,+position_inside,channel_inside,shift_inside} = facts
  %Arrays.indexed_fold_change_context(~Execution.Owners,~Query,~Kernel.Filling,~filling,~update,~Execution.fill_group,~same_step,count(query),0,query,state) :
    {read(query,_) == wanted(query) : Tile.Tile8}
  OutputReadback.complete_readback(~Execution.Owners,~Query,~Tile.Tile8,~count,~position,~wanted,~frame,~read,~update,~frame_step,~at_target,~later,query,state,evidence,position_inside)

# Allocation facts for an actual guarded tile quantity.
def allocated_query(context: Kernel.Filling,input: Tree.Tree<F32>,rows: Nat,position: Nat,channel: Nat,shift: Nat,count: U32) -> Query:
  +count = count
  +depth = Tensors.depth_for((count + 1 : U32))
  Query{context,input,rows,position,channel,shift,depth,U32.shln(1,depth)}
def initial(+query: Query) -> Execution.Owners:
  Execution.Owners{original(query),Tree.fresh(Tile.Tile8,depth(query),Tile.filled(0.0))}
def span_positive_from_position(+query: Query,+inside: {Nat.is_lt(position(query),count(query)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(span(filling(query)))) == True{} : Bool}:
  RowMajor.left_factor_positive(U32.to_nat(span(filling(query))),rows(query),
    %RowMajor.multiply_commutative(rows(query),U32.to_nat(span(filling(query)))) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_bound(position(query),count(query),inside))
def rows_positive_from_position(+query: Query,+inside: {Nat.is_lt(position(query),count(query)) == True{} : Bool}) ->
  {Nat.is_lt(0n,rows(query)) == True{} : Bool}:
  RowMajor.left_factor_positive(rows(query),U32.to_nat(span(filling(query))),Order.positive_bound(position(query),count(query),inside))
def two_inside_block(+query: Query,
  +channel_inside: {Nat.is_lt(channel(query),U32.to_nat(outputs(filling(query)))) == True{} : Bool},
  +shift_inside: {Nat.is_lt(shift(query),U32.to_nat(shifts(filling(query)))) == True{} : Bool}) ->
  {Nat.is_le(2n,block_size(filling(query))) == True{} : Bool}:
  +o = U32.to_nat(outputs(filling(query)))
  +s = U32.to_nat(shifts(filling(query)))
  +positive = Order.product_positive(o,s,Order.positive_bound(channel(query),o,channel_inside),Order.positive_bound(shift(query),s,shift_inside))
  %Order.commutative(1n,Nat.mul(o,s)) : {Nat.is_le(2n,_) == True{} : Bool}
  Order.offset_le(1n,1n,Nat.mul(o,s),Order.positive_one(Nat.mul(o,s),positive))
def double_span_room(+query: Query,
  +span_ordered: {Nat.is_le(U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query)))) == True{} : Bool},
  +room: {Nat.is_le(tile_count(query),U32.to_nat(capacity(query))) == True{} : Bool},
  +position_inside: {Nat.is_lt(position(query),count(query)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel(query),U32.to_nat(outputs(filling(query)))) == True{} : Bool},
  +shift_inside: {Nat.is_lt(shift(query),U32.to_nat(shifts(filling(query)))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(span(filling(query)))),U32.to_nat(capacity(query))) == True{} : Bool}:
  +span = U32.to_nat(span(filling(query)))
  +groups = U32.to_nat(groups(filling(query)))
  +block = block_size(filling(query))
  +row_count = rows(query)
  +row_positive = rows_positive_from_position(query,position_inside)
  %Equal.trans(Nat,Nat.mul(2n,span),Nat.add(span,span),Nat.double(span),Order.twice(span),Addition.double(span)) : {Nat.is_le(_,U32.to_nat(capacity(query))) == True{} : Bool}
  Order.transitive(Nat.mul(2n,span),Nat.mul(groups,block),U32.to_nat(capacity(query)),
    Order.transitive(Nat.mul(2n,span),Nat.mul(block,span),Nat.mul(groups,block),RowMajor.mul_le(2n,block,span,two_inside_block(query,channel_inside,shift_inside)),
      %Equal.sym(Nat,Nat.mul(block,span),Nat.mul(span,block),RowMajor.multiply_commutative(block,span)) : {Nat.is_le(_,Nat.mul(groups,block)) == True{} : Bool} RowMajor.mul_le(span,groups,block,span_ordered)),
    Order.transitive(Nat.mul(groups,block),tile_count(query),U32.to_nat(capacity(query)),
      %Equal.sym(Nat,Nat.mul(Nat.mul(row_count,groups),block),Nat.mul(row_count,Nat.mul(groups,block)),RowMajor.assoc(row_count,groups,block)) : {Nat.is_le(Nat.mul(groups,block),_) == True{} : Bool}
      %RowMajor.multiply_commutative(Nat.mul(groups,block),row_count) : {Nat.is_le(Nat.mul(groups,block),_) == True{} : Bool} RowMajor.product_covers(Nat.mul(groups,block),row_count,row_positive),room))

def initial_guarded_frame(+query: Query,+word_count: U32,
  +count_exact: {U32.to_nat(word_count) == tile_count(query) : Nat},
  +depth_exact: {depth(query) == Tensors.depth_for((word_count + 1 : U32)) : Nat},
  +capacity_exact: {capacity(query) == U32.shln(1,depth(query)) : U32},
  +guard: Depth.Guarded<word_count>,
  +span_ordered: {Nat.is_le(U32.to_nat(span(filling(query))),U32.to_nat(groups(filling(query)))) == True{} : Bool},
  +position_inside: {Nat.is_lt(position(query),count(query)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel(query),U32.to_nat(outputs(filling(query)))) == True{} : Bool},
  +shift_inside: {Nat.is_lt(shift(query),U32.to_nat(shifts(filling(query)))) == True{} : Bool}) -> Frame<query,initial(query)>:
  Depth.Guarded{+bounded,+within} = guard
  +depth_ok = {%Equal.sym(Nat,depth(query),Tensors.depth_for((word_count + 1 : U32)),depth_exact) : {Nat.is_le(_,31n) == True{} : Bool} bounded : {Nat.is_le(depth(query),31n) == True{} : Bool}}
  +cap_ok = {%Equal.sym(U32,capacity(query),U32.shln(1,depth(query)),capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool} Power.native_power_bound(depth(query),depth_ok) : {U32.is_le(capacity(query),2147483648) == True{} : Bool}}
  +native_room = {%Equal.sym(U32,capacity(query),U32.shln(1,depth(query)),capacity_exact) : {U32.is_lt(word_count,_) == True{} : Bool}
    %Equal.sym(Nat,depth(query),Tensors.depth_for((word_count + 1 : U32)),depth_exact) : {U32.is_lt(word_count,U32.shln(1,_)) == True{} : Bool} within : {U32.is_lt(word_count,capacity(query)) == True{} : Bool}}
  +room = {%count_exact : {Nat.is_le(_,U32.to_nat(capacity(query))) == True{} : Bool}
    Bounds.strict_le(U32.to_nat(word_count),U32.to_nat(capacity(query)),Bounds.as_nat_lt(word_count,capacity(query),native_room)) : {Nat.is_le(tile_count(query),U32.to_nat(capacity(query))) == True{} : Bool}}
  +facts = {Facts{cap_ok,span_positive_from_position(query,position_inside),span_ordered,double_span_room(query,span_ordered,room,position_inside,channel_inside,shift_inside),room,position_inside,channel_inside,shift_inside} : Facts<query>}
  +balanced = Shape.fresh(Tile.Tile8,depth(query),Tile.filled(0.0))
  Frame{{==},Equal.trans(U32,Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,depth(query),Tile.filled(0.0))),U32.shln(1,depth(query)),capacity(query),Shape.capacity(Tile.Tile8,depth(query),Tree.fresh(Tile.Tile8,depth(query),Tile.filled(0.0)),balanced),Equal.sym(U32,capacity(query),U32.shln(1,depth(query)),capacity_exact)),balanced,depth_ok,facts}

def allocated_frame(+context: Kernel.Filling,+input: Tree.Tree<F32>,+rows: Nat,+position: Nat,+channel: Nat,+shift: Nat,+word_count: U32,
  +count_exact: {U32.to_nat(word_count) == tile_count(allocated_query(context,input,rows,position,channel,shift,word_count)) : Nat},
  +guard: Depth.Guarded<word_count>,
  +span_ordered: {Nat.is_le(U32.to_nat(span(context)),U32.to_nat(groups(context))) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,Nat.mul(rows,U32.to_nat(span(context)))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(outputs(context))) == True{} : Bool},
  +shift_inside: {Nat.is_lt(shift,U32.to_nat(shifts(context))) == True{} : Bool}) ->
  Frame<allocated_query(context,input,rows,position,channel,shift,word_count),initial(allocated_query(context,input,rows,position,channel,shift,word_count))>:
  +query = allocated_query(context,input,rows,position,channel,shift,word_count)
  initial_guarded_frame(query,word_count,count_exact,{==},{==},guard,span_ordered,position_inside,channel_inside,shift_inside)

def count_room(+query: Query,+facts: Facts<query>) -> {Nat.is_le(count(query),U32.to_nat(capacity(query))) == True{} : Bool}:
  Facts{capacity_ok,span_positive,+span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  +span = U32.to_nat(span(filling(query)))
  +groups = U32.to_nat(groups(filling(query)))
  +rows = rows(query)
  Order.transitive(count(query),group_count(query),U32.to_nat(capacity(query)),
    %Equal.sym(Nat,Nat.mul(rows,span),Nat.mul(span,rows),RowMajor.multiply_commutative(rows,span)) : {Nat.is_le(_,group_count(query)) == True{} : Bool}
    %RowMajor.multiply_commutative(groups,rows) : {Nat.is_le(Nat.mul(span,rows),_) == True{} : Bool} RowMajor.mul_le(span,groups,rows,span_ordered),groups_room(query,facts))
def count_exact(+query: Query,+row_count: U32,+rows_exact: {U32.to_nat(row_count) == rows(query) : Nat},+facts: Facts<query>) ->
  {U32.to_nat((row_count * span(filling(query)) : U32)) == count(query) : Nat}:
  Facts{+capacity_ok,span_positive,span_ordered,double_span,room,position_inside,channel_inside,shift_inside} = facts
  %rows_exact : {U32.to_nat((row_count * span(filling(query)) : U32)) == Nat.mul(_,U32.to_nat(span(filling(query)))) : Nat}
  Arithmetic.multiply(row_count,span(filling(query)),capacity(query),capacity_ok,
    %Equal.sym(Nat,U32.to_nat(row_count),rows(query),rows_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(span(filling(query)))),U32.to_nat(capacity(query))) == True{} : Bool} count_room(query,facts))
def frame_facts(-query: Query,-state: Execution.Owners,evidence: Frame<query,state>) -> Facts<query>:
  Frame{source,capacity_equal,balanced,depth_ok,facts} = evidence
  facts
