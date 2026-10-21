# Readback of the actual contiguous shift fill, with the original source owner.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../kernel_spatial_packing.bend as Spatial
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./input_gradient_tiled_refinement.bend as Execution
import ./storage_tree_model.bend as Tree
import ./kernel_spatial_packing_samples.bend as Samples
import ./traversal_array_refinement.bend as Arrays
import ./traversal_indexed_values.bend as Indexed
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./contiguous_copy_readback.bend as Offset
import ./storage_shape.bend as Shape
import ./indexed_output_readback.bend as OutputReadback
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./loop_projection.bend as Projection

type Shift is Data:
  Shift{input: Tree.Tree<F32>,destination: U32,row: U32,column: U32,first: U32,base: U32,plane: U32,height: U32,width: U32,shifts: Nat}

def sample(context: Shift,position: U32) -> Tile.Tile8:
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  Samples.pure_vector(input,Spatial.PixelBlock{row,(column + position : U32),base,height,width,1})

def update(position: U32,context: Shift,tiles: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Shift{input,+destination,row,column,first,base,plane,height,width,shifts} = context
  +position = position
  Tree.write(Tile.Tile8,tiles,(destination + position : U32),sample(context,position))

def represent(context: Shift,position: U32,tiles: Tree.Tree<Tile.Tile8>) -> Execution.Fill:
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  +position = position
  Execution.Fill{input,tiles,(destination + position : U32),row,(column + position : U32),first,base,plane,height,width,shifts}

def step_matches(+context: Shift,+position: U32,+tiles: Tree.Tree<Tile.Tile8>) ->
  {Execution.fill_shift(represent(context,position,tiles)) == represent(context,(position + 1 : U32),update(position,context,tiles)) : Execution.Fill}:
  Shift{input,+destination,row,+column,first,base,plane,height,width,shifts} = context
  %Equal.sym(U32,((destination + position : U32) + 1 : U32),(destination + (position + 1 : U32) : U32),Arithmetic.add_associative(destination,position,1)) :
    {Execution.Fill{input,update(position,context,tiles),_,row,((column + position : U32) + 1 : U32),first,base,plane,height,width,shifts} ==
      represent(context,(position + 1 : U32),update(position,context,tiles)) : Execution.Fill}
  %Equal.sym(U32,((column + position : U32) + 1 : U32),(column + (position + 1 : U32) : U32),Arithmetic.add_associative(column,position,1)) :
    {Execution.Fill{input,update(position,context,tiles),(destination + (position + 1 : U32) : U32),row,_,first,base,plane,height,width,shifts} ==
      represent(context,(position + 1 : U32),update(position,context,tiles)) : Execution.Fill}
  {==}

def tiles(state: Execution.Fill) -> Tree.Tree<Tile.Tile8>:
  Execution.Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  tiles

def tiles_represent(+context: Shift,+position: U32,+output: Tree.Tree<Tile.Tile8>) ->
  {tiles(represent(context,position,output)) == output : Tree.Tree<Tile.Tile8>}:
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  {==}

def filled(+count: Nat,+context: Shift,+output: Tree.Tree<Tile.Tile8>) ->
  {tiles(Loop.run(~Execution.Fill,~Execution.fill_shift,count,represent(context,0,output))) ==
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output) : Tree.Tree<Tile.Tile8>}:
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output)
  %Equal.sym(Execution.Fill,Loop.run(~Execution.Fill,~Execution.fill_shift,count,represent(context,0,output)),
    represent(context,(0 + U32.from_nat(count) : U32),written),
    Arrays.indexed_loop(~Execution.Fill,~Tree.Tree<Tile.Tile8>,~Shift,~Execution.fill_shift,~update,~represent,~step_matches,count,0,context,output)) :
    {tiles(_) == written : Tree.Tree<Tile.Tile8>}
  tiles_represent(context,(0 + U32.from_nat(count) : U32),written)

def destination(context: Shift) -> U32:
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  destination

type Query is Data:
  Query{context: Shift,count: Nat,index: Nat,depth: Nat,capacity: U32}

def shift(query: Query) -> Shift:
  Query{context,count,index,depth,capacity} = query
  context

def affine(query: Query) -> Indexed.AffineQuery<Shift>:
  Query{+context,count,index,depth,capacity} = query
  Indexed.AffineQuery{context,destination(context),count,index,depth,capacity}

def queried(position: U32,query: Query,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  update(position,shift(query),output)

def same_step(+position: U32,+query: Query,+output: Tree.Tree<Tile.Tile8>) ->
  {queried(position,query,output) == update(position,shift(query),output) : Tree.Tree<Tile.Tile8>}:
  {==}

def affine_step_matches(+position: U32,+query: Query,+output: Tree.Tree<Tile.Tile8>) ->
  {queried(position,query,output) == Indexed.affine_update(~Tile.Tile8,~Shift,~sample,position,affine(query),output) : Tree.Tree<Tile.Tile8>}:
  Query{context,count,index,depth,capacity} = query
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  {==}

def readback(+query: Query,+output: Tree.Tree<Tile.Tile8>,evidence: Indexed.AffineFrame<Tile.Tile8,Shift,affine(query),output>) ->
  {Tree.read(Tile.Tile8,tiles(Loop.run(~Execution.Fill,~Execution.fill_shift,Indexed.affine_count(~Shift,affine(query)),represent(shift(query),0,output))),
    Indexed.affine_address(~Shift,affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Shift,~sample,affine(query)) : Tile.Tile8}:
  Query{context,count,index,depth,capacity} = query
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles(Loop.run(~Execution.Fill,~Execution.fill_shift,count,represent(context,0,output))),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output),filled(count,context,output)) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Shift,affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Shift,~sample,affine(query)) : Tile.Tile8}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~Query,~Shift,~shift,~queried,~update,~same_step,count,0,query,output) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Shift,affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Shift,~sample,affine(query)) : Tile.Tile8}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Query,~queried,count,0,query,output),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Indexed.AffineQuery<Shift>,~Indexed.affine_update(~Tile.Tile8,~Shift,~sample),count,0,affine(query),output),
    Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~Query,~Indexed.AffineQuery<Shift>,~affine,~queried,~Indexed.affine_update(~Tile.Tile8,~Shift,~sample),~affine_step_matches,count,0,query,output)) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Shift,affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Shift,~sample,affine(query)) : Tile.Tile8}
  Indexed.affine_readback(~Tile.Tile8,~Shift,~sample,affine(query),output,evidence)

type Channels is Data:
  Channels{input: Tree.Tree<F32>,destination: U32,row: U32,column: U32,base: U32,plane: U32,height: U32,width: U32,shifts: U32}

def channel_shift(context: Channels,channel: U32) -> Shift:
  Channels{input,destination,row,+column,base,+plane,height,width,+shifts} = context
  +channel = channel
  Shift{input,(destination + channel * shifts : U32),row,column,column,(base + channel * plane : U32),plane,height,width,U32.to_nat(shifts)}

def channel_represent(context: Channels,channel: U32,output: Tree.Tree<Tile.Tile8>) -> Execution.Fill:
  Channels{input,destination,row,+column,base,+plane,height,width,+shifts} = context
  +channel = channel
  Execution.Fill{input,output,(destination + channel * shifts : U32),row,column,column,(base + channel * plane : U32),plane,height,width,U32.to_nat(shifts)}

def channel_update(channel: U32,context: Channels,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Channels{input,destination,row,column,base,plane,height,width,+shifts} = context
  Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,U32.to_nat(shifts),0,channel_shift(context,channel),output)

def channel_initial(+context: Channels,+channel: U32,+output: Tree.Tree<Tile.Tile8>) ->
  {represent(channel_shift(context,channel),0,output) == channel_represent(context,channel,output) : Execution.Fill}:
  Channels{input,+destination,row,+column,base,plane,height,width,+shifts} = context
  %Equal.sym(U32,((destination + channel * shifts : U32) + 0 : U32),(destination + channel * shifts : U32),Arithmetic.add_zero((destination + channel * shifts : U32))) :
    {Execution.Fill{input,output,_,row,(column + 0 : U32),column,(base + channel * plane : U32),plane,height,width,U32.to_nat(shifts)} == channel_represent(context,channel,output) : Execution.Fill}
  %Equal.sym(U32,(column + 0 : U32),column,Arithmetic.add_zero(column)) :
    {Execution.Fill{input,output,(destination + channel * shifts : U32),row,_,column,(base + channel * plane : U32),plane,height,width,U32.to_nat(shifts)} == channel_represent(context,channel,output) : Execution.Fill}
  {==}

def channel_step_matches(+context: Channels,+channel: U32,+output: Tree.Tree<Tile.Tile8>) ->
  {Execution.fill_channel(channel_represent(context,channel,output)) == channel_represent(context,(channel + 1 : U32),channel_update(channel,context,output)) : Execution.Fill}:
  Channels{+input,+destination,+row,+column,+base,+plane,+height,+width,+shifts} = context
  +shift = channel_shift(context,channel)
  +written = channel_update(channel,context,output)
  %channel_initial(context,channel,output) :
    {Execution.next_channel(Loop.run(~Execution.Fill,~Execution.fill_shift,U32.to_nat(shifts),_)) == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Equal.sym(Execution.Fill,Loop.run(~Execution.Fill,~Execution.fill_shift,U32.to_nat(shifts),represent(shift,0,output)),
    represent(shift,(0 + U32.from_nat(U32.to_nat(shifts)) : U32),written),
    Arrays.indexed_loop(~Execution.Fill,~Tree.Tree<Tile.Tile8>,~Shift,~Execution.fill_shift,~update,~represent,~step_matches,U32.to_nat(shifts),0,shift,output)) :
    {Execution.next_channel(_) == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(shifts)),shifts,Arithmetic.native_roundtrip(shifts)) :
    {Execution.next_channel(represent(shift,(0 + _ : U32),written)) == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Equal.sym(U32,(0 + shifts : U32),shifts,Arithmetic.add_zero_left(shifts)) :
    {Execution.next_channel(represent(shift,_,written)) == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Equal.sym(U32,((destination + channel * shifts : U32) + shifts : U32),(destination + (channel * shifts + shifts : U32) : U32),Arithmetic.add_associative(destination,(channel * shifts : U32),shifts)) :
    {Execution.Fill{input,written,_,row,column,column,((base + channel * plane : U32) + plane : U32),plane,height,width,U32.to_nat(shifts)} == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Arithmetic.multiply_successor(channel,shifts) :
    {Execution.Fill{input,written,(destination + _ : U32),row,column,column,((base + channel * plane : U32) + plane : U32),plane,height,width,U32.to_nat(shifts)} == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Equal.sym(U32,((base + channel * plane : U32) + plane : U32),(base + (channel * plane + plane : U32) : U32),Arithmetic.add_associative(base,(channel * plane : U32),plane)) :
    {Execution.Fill{input,written,(destination + (channel + 1) * shifts : U32),row,column,column,_,plane,height,width,U32.to_nat(shifts)} == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  %Arithmetic.multiply_successor(channel,plane) :
    {Execution.Fill{input,written,(destination + (channel + 1) * shifts : U32),row,column,column,(base + _ : U32),plane,height,width,U32.to_nat(shifts)} == channel_represent(context,(channel + 1 : U32),written) : Execution.Fill}
  {==}

def read_before(+query: Query,+address: U32,+output: Tree.Tree<Tile.Tile8>,
  frame: Indexed.AffineFrame<Tile.Tile8,Shift,affine(query),output>,
  before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Indexed.affine_base(~Shift,affine(query)))) == True{} : Bool},
  inside: {U32.is_lt(address,Indexed.affine_capacity(~Shift,affine(query))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,tiles(Loop.run(~Execution.Fill,~Execution.fill_shift,Indexed.affine_count(~Shift,affine(query)),represent(shift(query),0,output))),address) ==
    Tree.read(Tile.Tile8,output,address) : Tile.Tile8}:
  Query{context,count,index,depth,capacity} = query
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles(Loop.run(~Execution.Fill,~Execution.fill_shift,count,represent(context,0,output))),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output),filled(count,context,output)) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~Query,~Shift,~shift,~queried,~update,~same_step,count,0,query,output) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Query,~queried,count,0,query,output),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Indexed.AffineQuery<Shift>,~Indexed.affine_update(~Tile.Tile8,~Shift,~sample),count,0,affine(query),output),
    Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~Query,~Indexed.AffineQuery<Shift>,~affine,~queried,~Indexed.affine_update(~Tile.Tile8,~Shift,~sample),~affine_step_matches,count,0,query,output)) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  Indexed.affine_read_before(~Tile.Tile8,~Shift,~sample,affine(query),address,output,frame,before,inside)

# Channel addresses use the same bounded row-major arithmetic as the callers.
def channel_base_exact(+destination: U32,+shifts: U32,+channel: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(shifts)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((destination + U32.from_nat(channel) * shifts : U32)) == Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))) : Nat}:
  +product = Nat.mul(channel,U32.to_nat(shifts))
  +product_room = Order.transitive(product,Nat.add(U32.to_nat(destination),product),U32.to_nat(capacity),
    %Equal.sym(Nat,Nat.add(U32.to_nat(destination),product),Nat.add(product,U32.to_nat(destination)),Order.commutative(U32.to_nat(destination),product)) :
      {Nat.is_le(product,_) == True{} : Bool} Order.le_add(product,U32.to_nat(destination)),room)
  +channel_room = Order.transitive(channel,product,U32.to_nat(capacity),RowMajor.product_covers(channel,U32.to_nat(shifts),positive),product_room)
  +channel_exact = Bounds.roundtrip(channel,capacity,capacity_ok,channel_room)
  +product_exact = {
    %channel_exact : {U32.to_nat((U32.from_nat(channel) * shifts : U32)) == Nat.mul(_,U32.to_nat(shifts)) : Nat}
    Arithmetic.multiply(U32.from_nat(channel),shifts,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(channel)),channel,channel_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(shifts)),U32.to_nat(capacity)) == True{} : Bool} product_room) :
      {U32.to_nat((U32.from_nat(channel) * shifts : U32)) == product : Nat}}
  %product_exact : {U32.to_nat((destination + U32.from_nat(channel) * shifts : U32)) == Nat.add(U32.to_nat(destination),_) : Nat}
  Arithmetic.add(destination,(U32.from_nat(channel) * shifts : U32),capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(channel) * shifts : U32)),product,product_exact) :
      {Nat.is_le(Nat.add(U32.to_nat(destination),_),U32.to_nat(capacity)) == True{} : Bool} room)

def channel_span_room(+destination: Nat,+shifts: Nat,+channel: Nat,+count: Nat,
  inside: {Nat.is_lt(channel,count) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(destination,Nat.mul(channel,shifts)),shifts),Nat.add(destination,Nat.mul(count,shifts))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(destination,Nat.mul(channel,shifts)),shifts),Nat.add(destination,Nat.add(Nat.mul(channel,shifts),shifts)),
    Addition.associative(destination,Nat.mul(channel,shifts),shifts)) : {Nat.is_le(_,Nat.add(destination,Nat.mul(count,shifts))) == True{} : Bool}
  %Order.commutative(shifts,Nat.mul(channel,shifts)) : {Nat.is_le(Nat.add(destination,_),Nat.add(destination,Nat.mul(count,shifts))) == True{} : Bool}
  Order.offset_le(destination,Nat.mul(1n+channel,shifts),Nat.mul(count,shifts),RowMajor.mul_le(1n+channel,count,shifts,Order.strict_follow(channel,count,inside)))

type ChannelQuery is Data:
  ChannelQuery{context: Channels,count: Nat,channel: Nat,shift: Nat,depth: Nat,capacity: U32}

def channel_context(query: ChannelQuery) -> Channels:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  context
def channel_count(query: ChannelQuery) -> Nat:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  count
def channel_index(query: ChannelQuery) -> Nat:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  channel
def selected_shift(query: ChannelQuery) -> Nat:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  shift
def channel_depth(query: ChannelQuery) -> Nat:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  depth
def channel_capacity(query: ChannelQuery) -> U32:
  ChannelQuery{context,count,channel,shift,depth,capacity} = query
  capacity
def channel_destination(context: Channels) -> U32:
  Channels{input,destination,row,column,base,plane,height,width,shifts} = context
  destination
def channel_shifts(context: Channels) -> U32:
  Channels{input,destination,row,column,base,plane,height,width,shifts} = context
  shifts
def channel_address(+query: ChannelQuery) -> U32:
  (destination(channel_shift(channel_context(query),U32.from_nat(channel_index(query)))) + U32.from_nat(selected_shift(query)) : U32)
def channel_wanted(query: ChannelQuery) -> Tile.Tile8:
  +query = query
  sample(channel_shift(channel_context(query),U32.from_nat(channel_index(query))),U32.from_nat(selected_shift(query)))
def channel_read(query: ChannelQuery,output: Tree.Tree<Tile.Tile8>) -> Tile.Tile8:
  Tree.read(Tile.Tile8,output,channel_address(query))
def channel_queried(position: U32,query: ChannelQuery,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  channel_update(position,channel_context(query),output)

type ChannelFrame<-query: ChannelQuery,-output: Tree.Tree<Tile.Tile8>> is Data:
  ChannelFrame{capacity_equal: {Tree.capacity(Tile.Tile8,output) == channel_capacity(query) : U32},
    balanced: Shape.Balanced(Tile.Tile8,channel_depth(query),output),depth_ok: {Nat.is_le(channel_depth(query),31n) == True{} : Bool},
    capacity_ok: {U32.is_le(channel_capacity(query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.mul(channel_count(query),U32.to_nat(channel_shifts(channel_context(query))))),U32.to_nat(channel_capacity(query))) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel_index(query),channel_count(query)) == True{} : Bool},
    shift_inside: {Nat.is_lt(selected_shift(query),U32.to_nat(channel_shifts(channel_context(query)))) == True{} : Bool}}

def channel_range_room(+query: ChannelQuery,+channel: Nat,
  +capacity_ok: {U32.is_le(channel_capacity(query),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.mul(channel_count(query),U32.to_nat(channel_shifts(channel_context(query))))),U32.to_nat(channel_capacity(query))) == True{} : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(channel_shifts(channel_context(query)))) == True{} : Bool},
  +inside: {Nat.is_lt(channel,channel_count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(destination(channel_shift(channel_context(query),U32.from_nat(channel)))),U32.to_nat(channel_shifts(channel_context(query)))),U32.to_nat(channel_capacity(query))) == True{} : Bool}:
  ChannelQuery{context,+count,selected,shift,depth,+capacity} = query
  Channels{input,+destination,row,column,base,plane,height,width,+shifts} = context
  +base_room = Order.transitive(Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),Nat.add(U32.to_nat(destination),Nat.mul(count,U32.to_nat(shifts))),U32.to_nat(capacity),
    Order.offset_le(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts)),Nat.mul(count,U32.to_nat(shifts)),RowMajor.mul_le(channel,count,U32.to_nat(shifts),Bounds.strict_le(channel,count,inside))),room)
  %Equal.sym(Nat,U32.to_nat((destination + U32.from_nat(channel) * shifts : U32)),Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),
    channel_base_exact(destination,shifts,channel,capacity,capacity_ok,positive,base_room)) : {Nat.is_le(Nat.add(_,U32.to_nat(shifts)),U32.to_nat(capacity)) == True{} : Bool}
  Order.transitive(Nat.add(Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),U32.to_nat(shifts)),Nat.add(U32.to_nat(destination),Nat.mul(count,U32.to_nat(shifts))),U32.to_nat(capacity),
    channel_span_room(U32.to_nat(destination),U32.to_nat(shifts),channel,count,inside),room)

def placed_sample(unit: Unit,context: Shift,position: U32) -> Tile.Tile8:
  sample(context,position)
def place(context: Shift,position: U32) -> U32:
  (destination(context) + position : U32)
def placed_context(query: Indexed.Query<Unit,Shift>) -> Shift:
  Indexed.query_context(~Unit,~Shift,query)
def placed_same(+position: U32,+query: Indexed.Query<Unit,Shift>,+output: Tree.Tree<Tile.Tile8>) ->
  {Indexed.placed_update(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place,position,query,output) == update(position,placed_context(query),output) : Tree.Tree<Tile.Tile8>}:
  Indexed.Query{count,index,depth,capacity,Unit{},context} = query
  Shift{input,destination,row,column,first,base,plane,height,width,shifts} = context
  {==}

def placed_run(+count: Nat,+context: Shift,+output: Tree.Tree<Tile.Tile8>) ->
  {Indexed.run(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place,count,0,context,Unit{},output) ==
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output) : Tree.Tree<Tile.Tile8>}:
  +query = {Indexed.Query{0n,0n,0n,0,Unit{},context} : Indexed.Query<Unit,Shift>}
  Equal.trans(Tree.Tree<Tile.Tile8>,Indexed.run(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place,count,0,context,Unit{},output),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Indexed.Query<Unit,Shift>,~Indexed.placed_update(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place),count,0,query,output),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output),
    Indexed.placed_folded(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place,query,count,0,output),
    Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~Indexed.Query<Unit,Shift>,~Shift,~placed_context,
      ~Indexed.placed_update(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place),~update,~placed_same,count,0,query,output))

def shift_capacity(+count: Nat,+context: Shift,+output: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output)) == Tree.capacity(Tile.Tile8,output) : U32}:
  %placed_run(count,context,output) : {Tree.capacity(Tile.Tile8,_) == Tree.capacity(Tile.Tile8,output) : U32}
  Indexed.capacity(~Unit,~Tile.Tile8,~Shift,~placed_sample,~place,count,0,context,Unit{},output)

def fill_balance(depth: Nat,state: Execution.Fill) -> Data:
  Shape.Balanced(Tile.Tile8,depth,tiles(state))

def fill_keeps_balance(+depth: Nat,+state: Execution.Fill,evidence: Shape.Balanced(Tile.Tile8,depth,tiles(state))) ->
  Shape.Balanced(Tile.Tile8,depth,tiles(Execution.fill_shift(state))):
  Execution.Fill{+input,+output,+index,+row,+column,first,+base,plane,+height,+width,shifts} = state
  Shape.write(Tile.Tile8,depth,output,index,Samples.pure_vector(input,Spatial.PixelBlock{row,column,base,height,width,1}),evidence)

def shift_balanced(+count: Nat,+context: Shift,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,
  balanced: Shape.Balanced(Tile.Tile8,depth,output)) ->
  Shape.Balanced(Tile.Tile8,depth,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Shift,~update,count,0,context,output)):
  %filled(count,context,output) : Shape.Balanced(Tile.Tile8,depth,_)
  Projection.preserves_invariant(~Execution.Fill,~Nat,~fill_balance,~Execution.fill_shift,~fill_keeps_balance,count,depth,represent(context,0,output),
    %Equal.sym(Tree.Tree<Tile.Tile8>,tiles(represent(context,0,output)),output,tiles_represent(context,0,output)) : Shape.Balanced(Tile.Tile8,depth,_) balanced)

def channel_frame(query: ChannelQuery,output: Tree.Tree<Tile.Tile8>) -> Data:
  ChannelFrame<query,output>

def channel_frame_step(+query: ChannelQuery,+position: U32,+output: Tree.Tree<Tile.Tile8>,evidence: ChannelFrame<query,output>) ->
  ChannelFrame<query,channel_queried(position,query,output)>:
  ChannelQuery{context,count,channel,shift,+depth,+capacity} = query
  Channels{input,destination,row,column,base,plane,height,width,+shifts} = context
  ChannelFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,channel_inside,shift_inside} = evidence
  ChannelFrame{Equal.trans(U32,Tree.capacity(Tile.Tile8,channel_queried(position,query,output)),Tree.capacity(Tile.Tile8,output),capacity,
      shift_capacity(U32.to_nat(shifts),channel_shift(context,position),output),capacity_equal),
    shift_balanced(U32.to_nat(shifts),channel_shift(context,position),output,depth,balanced),depth_ok,capacity_ok,room,channel_inside,shift_inside}

def channel_at_target(+query: ChannelQuery,+output: Tree.Tree<Tile.Tile8>,evidence: ChannelFrame<query,output>) ->
  {channel_read(query,channel_queried(U32.from_nat(channel_index(query)),query,output)) == channel_wanted(query) : Tile.Tile8}:
  ChannelQuery{context,count,+channel,+shift,+depth,+capacity} = query
  Channels{input,destination,row,column,base,plane,height,width,+shifts} = context
  ChannelFrame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,+channel_inside,+shift_inside} = evidence
  +selected = {Query{channel_shift(context,U32.from_nat(channel)),U32.to_nat(shifts),shift,depth,capacity} : Query}
  +selected_context = channel_shift(context,U32.from_nat(channel))
  %filled(U32.to_nat(shifts),selected_context,output) :
    {Tree.read(Tile.Tile8,_,channel_address(query)) == channel_wanted(query) : Tile.Tile8}
  readback(selected,output,Indexed.AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,
    channel_range_room(query,channel,capacity_ok,room,Order.positive_bound(shift,U32.to_nat(shifts),shift_inside),channel_inside),shift_inside})

def channel_base_room(+query: ChannelQuery,+channel: Nat,
  room: {Nat.is_le(Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.mul(channel_count(query),U32.to_nat(channel_shifts(channel_context(query))))),U32.to_nat(channel_capacity(query))) == True{} : Bool},
  inside: {Nat.is_le(channel,channel_count(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.mul(channel,U32.to_nat(channel_shifts(channel_context(query))))),U32.to_nat(channel_capacity(query))) == True{} : Bool}:
  ChannelQuery{context,+count,selected,shift,depth,+capacity} = query
  Channels{input,+destination,row,column,base,plane,height,width,+shifts} = context
  Order.transitive(Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),Nat.add(U32.to_nat(destination),Nat.mul(count,U32.to_nat(shifts))),U32.to_nat(capacity),
    Order.offset_le(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts)),Nat.mul(count,U32.to_nat(shifts)),RowMajor.mul_le(channel,count,U32.to_nat(shifts),inside)),room)

def channel_address_exact(+query: ChannelQuery,
  +capacity_ok: {U32.is_le(channel_capacity(query),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.mul(channel_count(query),U32.to_nat(channel_shifts(channel_context(query))))),U32.to_nat(channel_capacity(query))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel_index(query),channel_count(query)) == True{} : Bool},
  +shift_inside: {Nat.is_lt(selected_shift(query),U32.to_nat(channel_shifts(channel_context(query)))) == True{} : Bool}) ->
  {U32.to_nat(channel_address(query)) == Nat.add(U32.to_nat(channel_destination(channel_context(query))),Nat.add(Nat.mul(channel_index(query),U32.to_nat(channel_shifts(channel_context(query)))),selected_shift(query))) : Nat}:
  ChannelQuery{context,count,+channel,+shift,depth,+capacity} = query
  Channels{input,+destination,row,column,base,plane,height,width,+shifts} = context
  +positive = Order.positive_bound(shift,U32.to_nat(shifts),shift_inside)
  +channel_base = (destination + U32.from_nat(channel) * shifts : U32)
  +base_exact = channel_base_exact(destination,shifts,channel,capacity,capacity_ok,positive,channel_base_room(query,channel,room,Bounds.strict_le(channel,count,channel_inside)))
  +range_room = channel_range_room(query,channel,capacity_ok,room,positive,channel_inside)
  Equal.trans(Nat,U32.to_nat(channel_address(query)),Nat.add(U32.to_nat(channel_base),shift),Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)),
    Offset.offset_exact(channel_base,shift,capacity,capacity_ok,
      Bounds.strict_le(Nat.add(U32.to_nat(channel_base),shift),U32.to_nat(capacity),Bounds.nat_trans(Nat.add(U32.to_nat(channel_base),shift),Nat.add(U32.to_nat(channel_base),U32.to_nat(shifts)),U32.to_nat(capacity),
        Order.offset_lt(U32.to_nat(channel_base),shift,U32.to_nat(shifts),shift_inside),range_room))),
    %Equal.sym(Nat,U32.to_nat(channel_base),Nat.add(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts))),base_exact) :
      {Nat.add(_,shift) == Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)) : Nat}
      Addition.associative(U32.to_nat(destination),Nat.mul(channel,U32.to_nat(shifts)),shift))

def channel_later(+query: ChannelQuery,+position: Nat,+output: Tree.Tree<Tile.Tile8>,evidence: ChannelFrame<query,output>,
  after: {Nat.is_lt(channel_index(query),position) == True{} : Bool},inside: {Nat.is_lt(position,channel_count(query)) == True{} : Bool}) ->
  {channel_read(query,channel_queried(U32.from_nat(position),query,output)) == channel_read(query,output) : Tile.Tile8}:
  ChannelQuery{context,+count,+channel,+shift,+depth,+capacity} = query
  Channels{input,+destination,row,column,base,plane,height,width,+shifts} = context
  ChannelFrame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,+channel_inside,+shift_inside} = evidence
  +inside = inside
  +positive = Order.positive_bound(shift,U32.to_nat(shifts),shift_inside)
  +selected = {Query{channel_shift(context,U32.from_nat(position)),U32.to_nat(shifts),0n,depth,capacity} : Query}
  +selected_context = channel_shift(context,U32.from_nat(position))
  +range_room = channel_range_room(query,position,capacity_ok,room,positive,inside)
  +observed_room = RowMajor.radix(channel,count,shift,U32.to_nat(shifts),channel_inside,shift_inside)
  +address_inside = {
    %Equal.sym(Bool,U32.is_lt(channel_address(query),capacity),Nat.is_lt(U32.to_nat(channel_address(query)),U32.to_nat(capacity)),Comparison.lt(channel_address(query),capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(channel_address(query)),Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)),channel_address_exact(query,capacity_ok,room,channel_inside,shift_inside)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.nat_trans(Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)),Nat.add(U32.to_nat(destination),Nat.mul(count,U32.to_nat(shifts))),U32.to_nat(capacity),
      Order.offset_lt(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift),Nat.mul(count,U32.to_nat(shifts)),observed_room),room) :
      {U32.is_lt(channel_address(query),capacity) == True{} : Bool}}
  +before = {
    %Equal.sym(Nat,U32.to_nat(channel_address(query)),Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)),channel_address_exact(query,capacity_ok,room,channel_inside,shift_inside)) :
      {Nat.is_lt(_,U32.to_nat(Indexed.affine_base(~Shift,affine(selected)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Indexed.affine_base(~Shift,affine(selected))),Nat.add(U32.to_nat(destination),Nat.mul(position,U32.to_nat(shifts))),
      channel_base_exact(destination,shifts,position,capacity,capacity_ok,positive,channel_base_room(query,position,room,Bounds.strict_le(position,count,inside)))) :
      {Nat.is_lt(Nat.add(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift)),_) == True{} : Bool}
    Order.offset_lt(U32.to_nat(destination),Nat.add(Nat.mul(channel,U32.to_nat(shifts)),shift),Nat.mul(position,U32.to_nat(shifts)),
      RowMajor.radix(channel,position,shift,U32.to_nat(shifts),after,shift_inside)) :
      {Nat.is_lt(U32.to_nat(channel_address(query)),U32.to_nat(Indexed.affine_base(~Shift,affine(selected)))) == True{} : Bool}}
  %filled(U32.to_nat(shifts),selected_context,output) :
    {Tree.read(Tile.Tile8,_,channel_address(query)) == channel_read(query,output) : Tile.Tile8}
  read_before(selected,channel_address(query),output,Indexed.AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,range_room,positive},before,address_inside)

def channel_same_step(+position: U32,+query: ChannelQuery,+output: Tree.Tree<Tile.Tile8>) ->
  {channel_queried(position,query,output) == channel_update(position,channel_context(query),output) : Tree.Tree<Tile.Tile8>}:
  {==}

def channels_readback(+query: ChannelQuery,+output: Tree.Tree<Tile.Tile8>,+evidence: ChannelFrame<query,output>) ->
  {channel_read(query,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,channel_count(query),0,channel_context(query),output)) == channel_wanted(query) : Tile.Tile8}:
  ChannelFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,+channel_inside,shift_inside} = evidence
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~ChannelQuery,~Channels,~channel_context,~channel_queried,~channel_update,~channel_same_step,channel_count(query),0,query,output) :
    {channel_read(query,_) == channel_wanted(query) : Tile.Tile8}
  OutputReadback.complete_readback(~Tree.Tree<Tile.Tile8>,~ChannelQuery,~Tile.Tile8,~channel_count,~channel_index,~channel_wanted,~channel_frame,~channel_read,~channel_queried,
    ~channel_frame_step,~channel_at_target,~channel_later,query,output,evidence,channel_inside)

def initial_channel(context: Channels,output: Tree.Tree<Tile.Tile8>) -> Execution.Fill:
  Channels{input,destination,row,+column,base,plane,height,width,shifts} = context
  Execution.Fill{input,output,destination,row,column,column,base,plane,height,width,U32.to_nat(shifts)}

def channel_zero(+context: Channels,+output: Tree.Tree<Tile.Tile8>) ->
  {channel_represent(context,0,output) == initial_channel(context,output) : Execution.Fill}:
  Channels{input,+destination,row,+column,+base,+plane,height,width,+shifts} = context
  %Equal.sym(U32,(0 * shifts : U32),0,Arithmetic.multiply_zero_left(shifts)) :
    {Execution.Fill{input,output,(destination + _ : U32),row,column,column,(base + 0 * plane : U32),plane,height,width,U32.to_nat(shifts)} == initial_channel(context,output) : Execution.Fill}
  %Equal.sym(U32,(0 * plane : U32),0,Arithmetic.multiply_zero_left(plane)) :
    {Execution.Fill{input,output,(destination + 0 : U32),row,column,column,(base + _ : U32),plane,height,width,U32.to_nat(shifts)} == initial_channel(context,output) : Execution.Fill}
  %Equal.sym(U32,(destination + 0 : U32),destination,Arithmetic.add_zero(destination)) :
    {Execution.Fill{input,output,_,row,column,column,(base + 0 : U32),plane,height,width,U32.to_nat(shifts)} == initial_channel(context,output) : Execution.Fill}
  %Equal.sym(U32,(base + 0 : U32),base,Arithmetic.add_zero(base)) :
    {Execution.Fill{input,output,destination,row,column,column,_,plane,height,width,U32.to_nat(shifts)} == initial_channel(context,output) : Execution.Fill}
  {==}

def group_channels(position: U32,context: Kernel.Filling,input: Tree.Tree<F32>) -> Channels:
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,base,plane,+first,+source_rows,+group,+span} = context
  +position = position
  Channels{input,((position / span * chunk_groups + position % span) * (outputs * shifts + 1) : U32),
    (position / span + low - kernel - first : U32),((position % span + group) * 8 + column_low - kernel : U32),base,plane,source_rows,width,shifts}

def group_outputs(context: Kernel.Filling) -> Nat:
  Kernel.Filling{Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows},low,base,plane,first,source_rows,group,span} = context
  U32.to_nat(outputs)

def group_written(+position: U32,+context: Kernel.Filling,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Execution.fill_group(position,context,Execution.Owners{input,output}) ==
    Execution.Owners{input,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,group_outputs(context),0,group_channels(position,context,input),output)} : Execution.Owners}:
  Kernel.Filling{Kernel.Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,+base,+plane,+first,+source_rows,+group,+span} = context
  +channels = group_channels(position,context,input)
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,U32.to_nat(outputs),0,channels,output)
  %channel_zero(channels,output) :
    {Execution.owners(Loop.run(~Execution.Fill,~Execution.fill_channel,U32.to_nat(outputs),_)) == Execution.Owners{input,written} : Execution.Owners}
  %Equal.sym(Execution.Fill,Loop.run(~Execution.Fill,~Execution.fill_channel,U32.to_nat(outputs),channel_represent(channels,0,output)),
    channel_represent(channels,(0 + U32.from_nat(U32.to_nat(outputs)) : U32),written),
    Arrays.indexed_loop(~Execution.Fill,~Tree.Tree<Tile.Tile8>,~Channels,~Execution.fill_channel,~channel_update,~channel_represent,~channel_step_matches,U32.to_nat(outputs),0,channels,output)) :
    {Execution.owners(_) == Execution.Owners{input,written} : Execution.Owners}
  {==}

# All channels of a later group preserve any earlier valid tile address.
type BeforeQuery is Data:
  BeforeQuery{query: ChannelQuery,address: U32,original: Tree.Tree<Tile.Tile8>}

def before_query(context: BeforeQuery) -> ChannelQuery:
  BeforeQuery{query,address,original} = context
  query
def before_count(context: BeforeQuery) -> Nat:
  channel_count(before_query(context))
def before_address(context: BeforeQuery) -> U32:
  BeforeQuery{query,address,original} = context
  address
def before_wanted(context: BeforeQuery) -> Tile.Tile8:
  BeforeQuery{query,address,original} = context
  Tree.read(Tile.Tile8,original,address)
def before_read(context: BeforeQuery,output: Tree.Tree<Tile.Tile8>) -> Tile.Tile8:
  Tree.read(Tile.Tile8,output,before_address(context))
def before_update(position: U32,context: BeforeQuery,output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  channel_queried(position,before_query(context),output)

type BeforeFrame<-context: BeforeQuery,-output: Tree.Tree<Tile.Tile8>> is Data:
  BeforeFrame{channels: ChannelFrame<before_query(context),output>,
    inside: {U32.is_lt(before_address(context),channel_capacity(before_query(context))) == True{} : Bool},
    before: {Nat.is_lt(U32.to_nat(before_address(context)),U32.to_nat(channel_destination(channel_context(before_query(context))))) == True{} : Bool}}

def before_frame(context: BeforeQuery,output: Tree.Tree<Tile.Tile8>) -> Data:
  BeforeFrame<context,output>

def before_frame_step(+context: BeforeQuery,+position: U32,+output: Tree.Tree<Tile.Tile8>,evidence: BeforeFrame<context,output>) ->
  BeforeFrame<context,before_update(position,context,output)>:
  BeforeFrame{channels,inside,before} = evidence
  BeforeFrame{channel_frame_step(before_query(context),position,output,channels),inside,before}

def before_unchanged(+context: BeforeQuery,+position: Nat,+output: Tree.Tree<Tile.Tile8>,evidence: BeforeFrame<context,output>,
  selected: {Nat.is_lt(position,before_count(context)) == True{} : Bool}) ->
  {before_read(context,before_update(U32.from_nat(position),context,output)) == before_read(context,output) : Tile.Tile8}:
  BeforeQuery{query,+address,original} = context
  ChannelQuery{channels,count,channel,+shift,+depth,+capacity} = query
  Channels{input,+destination,row,column,base,plane,height,width,+shifts} = channels
  BeforeFrame{channels_evidence,observed_inside,before} = evidence
  ChannelFrame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,channel_inside,+shift_inside} = channels_evidence
  +selected = selected
  +positive = Order.positive_bound(shift,U32.to_nat(shifts),shift_inside)
  +shift_query = {Query{channel_shift(channels,U32.from_nat(position)),U32.to_nat(shifts),0n,depth,capacity} : Query}
  +shift_context = channel_shift(channels,U32.from_nat(position))
  +range_room = channel_range_room(query,position,capacity_ok,room,positive,selected)
  +earlier = {
    %Equal.sym(Nat,U32.to_nat(Indexed.affine_base(~Shift,affine(shift_query))),Nat.add(U32.to_nat(destination),Nat.mul(position,U32.to_nat(shifts))),
      channel_base_exact(destination,shifts,position,capacity,capacity_ok,positive,channel_base_room(query,position,room,Bounds.strict_le(position,count,selected)))) :
      {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool}
    Bounds.nat_trans(U32.to_nat(address),U32.to_nat(destination),Nat.add(U32.to_nat(destination),Nat.mul(position,U32.to_nat(shifts))),before,Order.le_add(U32.to_nat(destination),Nat.mul(position,U32.to_nat(shifts)))) :
      {Nat.is_lt(U32.to_nat(address),U32.to_nat(Indexed.affine_base(~Shift,affine(shift_query)))) == True{} : Bool}}
  %filled(U32.to_nat(shifts),shift_context,output) : {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  read_before(shift_query,address,output,Indexed.AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,range_room,positive},earlier,observed_inside)

def before_same(+position: U32,+context: BeforeQuery,+output: Tree.Tree<Tile.Tile8>) ->
  {before_update(position,context,output) == channel_queried(position,before_query(context),output) : Tree.Tree<Tile.Tile8>}:
  {==}

def channels_read_before(+query: ChannelQuery,+address: U32,+output: Tree.Tree<Tile.Tile8>,channels: ChannelFrame<query,output>,
  inside: {U32.is_lt(address,channel_capacity(query)) == True{} : Bool},
  before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(channel_destination(channel_context(query)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,channel_count(query),0,channel_context(query),output),address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}:
  +context = {BeforeQuery{query,address,output} : BeforeQuery}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~ChannelQuery,~Channels,~channel_context,~channel_queried,~channel_update,~channel_same_step,channel_count(query),0,query,output) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~BeforeQuery,~ChannelQuery,~before_query,~before_update,~channel_queried,~before_same,channel_count(query),0,context,output) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,output,address) : Tile.Tile8}
  OutputReadback.preserves_readback(~Tree.Tree<Tile.Tile8>,~BeforeQuery,~Tile.Tile8,~before_count,~before_read,~before_wanted,~before_frame,~before_update,~before_frame_step,~before_unchanged,
    channel_count(query),0n,context,output,Order.reflexive(channel_count(query)),BeforeFrame{channels,inside,before},{==})

def kept_capacity(position: U32,context: Channels,capacity: U32) -> U32:
  capacity

def kept_capacity_fixed(+position: U32,+context: Channels,+capacity: U32) -> {kept_capacity(position,context,capacity) == capacity : U32}:
  {==}

def channel_capacity_step(+position: U32,+context: Channels,+output: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,channel_update(position,context,output)) == kept_capacity(position,context,Tree.capacity(Tile.Tile8,output)) : U32}:
  Channels{input,destination,row,column,base,plane,height,width,+shifts} = context
  shift_capacity(U32.to_nat(shifts),channel_shift(context,position),output)

def tree_capacity(output: Tree.Tree<Tile.Tile8>) -> U32:
  Tree.capacity(Tile.Tile8,output)

def channels_capacity(+count: Nat,+context: Channels,+output: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,count,0,context,output)) == Tree.capacity(Tile.Tile8,output) : U32}:
  Equal.trans(U32,tree_capacity(Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,count,0,context,output)),
    Traversal.indexed_fold(~U32,~Channels,~kept_capacity,count,0,context,tree_capacity(output)),tree_capacity(output),
    Arrays.indexed_fold_observes(~Tree.Tree<Tile.Tile8>,~U32,~Channels,~channel_update,~tree_capacity,~kept_capacity,~channel_capacity_step,count,0,context,output),
    Arrays.indexed_fold_fixed(~U32,~Channels,~kept_capacity,~kept_capacity_fixed,count,0,context,tree_capacity(output)))

def channel_balance(query: ChannelQuery,index: Nat,output: Tree.Tree<Tile.Tile8>) -> Data:
  Shape.Balanced(Tile.Tile8,channel_depth(query),output)

def channel_balance_step(+query: ChannelQuery,+index: Nat,+output: Tree.Tree<Tile.Tile8>,
  inside: {Nat.is_lt(index,channel_count(query)) == True{} : Bool},balanced: Shape.Balanced(Tile.Tile8,channel_depth(query),output)) ->
  Shape.Balanced(Tile.Tile8,channel_depth(query),channel_queried(U32.from_nat(index),query,output)):
  ChannelQuery{context,count,channel,shift,+depth,capacity} = query
  Channels{input,destination,row,column,base,plane,height,width,+shifts} = context
  shift_balanced(U32.to_nat(shifts),channel_shift(context,U32.from_nat(index)),output,depth,balanced)

def channels_balanced(+count: Nat,+context: Channels,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,balanced: Shape.Balanced(Tile.Tile8,depth,output)) ->
  Shape.Balanced(Tile.Tile8,depth,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Channels,~channel_update,count,0,context,output)):
  +query = {ChannelQuery{context,count,0n,0n,depth,0} : ChannelQuery}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~ChannelQuery,~Channels,~channel_context,~channel_queried,~channel_update,~channel_same_step,count,0,query,output) :
    Shape.Balanced(Tile.Tile8,depth,_)
  Arrays.indexed_fold_preserves_invariant(~Tree.Tree<Tile.Tile8>,~ChannelQuery,~channel_count,~channel_balance,~channel_queried,~channel_balance_step,count,0n,query,output,Order.reflexive(count),balanced)
