# Valid four-channel jobs select the allocated mapped eight-channel weights.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_weights.bend as NativeWeights
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_window_weights.bend as Window
import ./input_gradient_tiled_weights.bend as Weights
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_phase_index.bend as Index
import ./input_gradient_tiled_pack_values.bend as PackedValues
import ./input_gradient_tiled_offsets.bend as Offsets
import ./kernel_weights_refinement.bend as Mapped
import ./kernel_weights_readback.bend as Samples
import ./tensor_allocation_depth.bend as Depth
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor

def block_inside(+phase: Jobs.Phase,+unit: Nat,+inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(unit,2n),Ceiling.ceiling(Domain.channels(Jobs.geometry(phase)),8n)) == True{} : Bool}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  +channels = Domain.channels(g)
  +blocks = Ceiling.ceiling(channels,8n)
  +units = Ceiling.ceiling(channels,4n)
  +covers = Ceiling.covers(channels,8n,{==})
  +unit_cover = {%Equal.sym(Nat,Nat.mul(Nat.mul(blocks,2n),4n),Nat.mul(blocks,8n),RowMajor.assoc(blocks,2n,4n)) : {Nat.is_le(channels,_) == True{} : Bool} covers : {Nat.is_le(channels,Nat.mul(Nat.mul(blocks,2n),4n)) == True{} : Bool}}
  Division.quotient_below(unit,blocks,2n,{==},Bounds.nat_trans(unit,units,Nat.mul(blocks,2n),
    %Jobs.units_exact(call,e) : {Nat.is_lt(unit,_) == True{} : Bool} inside,Ceiling.minimal(channels,4n,Nat.mul(blocks,2n),{==},unit_cover)))
def block_count(+phase: Jobs.Phase) -> Nat:
  Mapped.mapped_blocks(~Kernel.WeightWindow,Weights.phase_layout(Jobs.context(phase)))
def block_count_exact(+phase: Jobs.Phase) -> {block_count(phase) == Ceiling.ceiling(Domain.channels(Jobs.geometry(phase)),8n) : Nat}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Window.block_count_exact(call,U32.from_nat(y),U32.from_nat(x))
def quantity(+phase: Jobs.Phase) -> U32:
  (((Domain.channel_word(Jobs.geometry(phase)) + 7) / 8) * Kernel.phase_inner(Jobs.context(phase)) : U32)
def query_from(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool},
  allocation: Depth.Sized<quantity(phase)>) -> Mapped.Query<Kernel.WeightWindow>:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Depth.Sized{+depth_ok,+room} = allocation
  +layout = Weights.phase_layout(Jobs.context(phase))
  +count_exact = Window.packed_exact(call,U32.from_nat(y),U32.from_nat(x))
  Mapped.Query{layout,Buffer.tree(~F32,original),block,tap,depth_ok,
    %count_exact : {Nat.is_le(_,U32.to_nat(U32.shln(1,Tensors.depth_for(quantity(phase))))) == True{} : Bool} room,block_inside,tap_inside}
def query(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) -> Mapped.Query<Kernel.WeightWindow>:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  query_from(phase,original,block,tap,block_inside,tap_inside,Window.allocation(call,U32.from_nat(y),U32.from_nat(x)))
def query_layout_from(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool},
  allocation: Depth.Sized<quantity(phase)>) ->
  {Mapped.query_layout(~Kernel.WeightWindow,query_from(phase,original,block,tap,block_inside,tap_inside,allocation)) == Weights.phase_layout(Jobs.context(phase)) : NativeWeights.MappedLayout<Kernel.WeightWindow>}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Depth.Sized{depth_ok,room} = allocation
  {==}
def query_input_from(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool},
  allocation: Depth.Sized<quantity(phase)>) ->
  {Mapped.query_input(~Kernel.WeightWindow,query_from(phase,original,block,tap,block_inside,tap_inside,allocation)) == Buffer.tree(~F32,original) : Tree.Tree<F32>}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Depth.Sized{depth_ok,room} = allocation
  {==}
def readback(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Weights.phase_weights(Jobs.context(phase),original)),Mapped.query_address(~Kernel.WeightWindow,query(phase,original,block,tap,block_inside,tap_inside))) ==
    Mapped.query_wanted(~Kernel.WeightWindow,~Kernel.weight_window_offset,query(phase,original,block,tap,block_inside,tap_inside)) : Tile.Tile8}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +allocated = Window.allocation(call,U32.from_nat(y),U32.from_nat(x))
  Weights.phase_readback(Jobs.context(phase),original,query(phase,original,block,tap,block_inside,tap_inside),
    query_layout_from(phase,original,block,tap,block_inside,tap_inside,allocated),query_input_from(phase,original,block,tap,block_inside,tap_inside,allocated))

def tap_index_inside(+phase: Jobs.Phase,+output: Nat,+row: Nat,+column: Nat,
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(Packing.phase_index(Jobs.context(phase),output,row,column),U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +rows = U32.to_nat(Packing.phase_rows(Jobs.context(phase)))
  +columns = U32.to_nat(Packing.phase_columns(Jobs.context(phase)))
  %Equal.sym(Nat,U32.to_nat(Kernel.phase_inner(Jobs.context(phase))),Nat.mul(Domain.outputs(g),Nat.mul(rows,columns)),PhaseBounds.phase_inner_exact(call,U32.from_nat(y),U32.from_nat(x))) :
    {Nat.is_lt(Packing.phase_index(Jobs.context(phase),output,row,column),_) == True{} : Bool}
  RowMajor.in_bounds(output,Domain.outputs(g),rows,columns,row,column,output_inside,row_inside,column_inside)
def unit_query(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+unit: Nat,+output: Nat,+row: Nat,+column: Nat,
  +unit_inside: {Nat.is_lt(unit,U32.to_nat(Jobs.units(Jobs.geometry(phase)))) == True{} : Bool},
  +output_inside: {Nat.is_lt(output,Domain.outputs(Jobs.geometry(phase))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Packing.phase_rows(Jobs.context(phase)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Packing.phase_columns(Jobs.context(phase)))) == True{} : Bool}) -> Mapped.Query<Kernel.WeightWindow>:
  query(phase,original,Nat.div(unit,2n),Packing.phase_index(Jobs.context(phase),output,row,column),
    %Equal.sym(Nat,block_count(phase),Ceiling.ceiling(Domain.channels(Jobs.geometry(phase)),8n),block_count_exact(phase)) : {Nat.is_lt(Nat.div(unit,2n),_) == True{} : Bool} block_inside(phase,unit,unit_inside),
    tap_index_inside(phase,output,row,column,output_inside,row_inside,column_inside))

def selected_sample(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat) -> Tile.Tile8:
  Samples.sample_linear(Buffer.tree(~F32,original),Domain.channel_word(Jobs.geometry(phase)),Weights.spatial(Jobs.context(phase)),(U32.from_nat(block) * 8 : U32),
    Kernel.weight_window_offset(Kernel.weight_window(Jobs.context(phase)),U32.from_nat(tap)))
def address_from(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool},
  allocation: Depth.Sized<quantity(phase)>) ->
  {Mapped.query_address(~Kernel.WeightWindow,query_from(phase,original,block,tap,block_inside,tap_inside,allocation)) == U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap)) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Depth.Sized{depth_ok,room} = allocation
  {==}
def wanted_from(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool},
  allocation: Depth.Sized<quantity(phase)>) ->
  {Mapped.query_wanted(~Kernel.WeightWindow,~Kernel.weight_window_offset,query_from(phase,original,block,tap,block_inside,tap_inside,allocation)) == selected_sample(phase,original,block,tap) : Tile.Tile8}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Depth.Sized{depth_ok,room} = allocation
  {==}

def slot_inside(+phase: Jobs.Phase,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),U32.to_nat(Weights.phase_count(Jobs.context(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.sym(Nat,U32.to_nat(Weights.phase_count(Jobs.context(phase))),Nat.mul(block_count(phase),U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),Window.packed_exact(call,U32.from_nat(y),U32.from_nat(x))) :
    {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),_) == True{} : Bool}
  RowMajor.radix(block,block_count(phase),tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase))),block_inside,tap_inside)

def packed_query(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat) -> PackedValues.Query:
  PackedValues.allocated_query(Index.packing_bounds(phase),Kernel.tiling(Jobs.geometry(phase)),original,Index.ordinal(phase),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),tap)
def packed_readback(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {PackedValues.read(packed_query(phase,original,block,tap),PackedValues.complete(packed_query(phase,original,block,tap),PackedValues.initial(packed_query(phase,original,block,tap)))) == PackedValues.wanted(packed_query(phase,original,block,tap)) : PackedValues.Value}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  PackedValues.allocated_readback(Index.packing_bounds(phase),Kernel.tiling(g),original,Index.ordinal(phase),Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),tap,Index.packing_inside(phase),
    %Equal.sym(Kernel.Phase,Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)) : {Nat.is_lt(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap),U32.to_nat(Weights.phase_count(_))) == True{} : Bool} slot_inside(phase,block,tap,block_inside,tap_inside),
    %Equal.sym(Kernel.Phase,Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)) : {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(_))) == True{} : Bool} tap_inside)

def weight(value: PackedValues.Value) -> Tile.Tile8:
  PackedValues.Value{weight,offset} = value
  weight
def offset(value: PackedValues.Value) -> U32:
  PackedValues.Value{weight,offset} = value
  offset
def packed_weight_readback(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {weight(PackedValues.read(packed_query(phase,original,block,tap),PackedValues.complete(packed_query(phase,original,block,tap),PackedValues.initial(packed_query(phase,original,block,tap))))) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Weights.phase_weights(Jobs.context(phase),original)),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap))) : Tile.Tile8}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +q = packed_query(phase,original,block,tap)
  +state = PackedValues.complete(q,PackedValues.initial(q))
  +position = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap))
  Equal.trans(Tile.Tile8,weight(PackedValues.read(q,state)),weight(PackedValues.wanted(q)),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Weights.phase_weights(Jobs.context(phase),original)),position),
    Equal.cong(PackedValues.Value,Tile.Tile8,weight,PackedValues.read(q,state),PackedValues.wanted(q),packed_readback(phase,original,block,tap,block_inside,tap_inside)),
    Equal.cong(Kernel.Phase,Tile.Tile8,selected => Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Weights.phase_weights(selected,original)),position),Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)))
def packed_offset_readback(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {offset(PackedValues.read(packed_query(phase,original,block,tap),PackedValues.complete(packed_query(phase,original,block,tap),PackedValues.initial(packed_query(phase,original,block,tap))))) ==
    Offsets.value(Weights.phase_table(Jobs.context(phase),Kernel.tiling(Jobs.geometry(phase)),0),U32.from_nat(tap)) : U32}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +q = packed_query(phase,original,block,tap)
  +state = PackedValues.complete(q,PackedValues.initial(q))
  Equal.trans(U32,offset(PackedValues.read(q,state)),offset(PackedValues.wanted(q)),Offsets.value(Weights.phase_table(Jobs.context(phase),Kernel.tiling(g),0),U32.from_nat(tap)),
    Equal.cong(PackedValues.Value,U32,offset,PackedValues.read(q,state),PackedValues.wanted(q),packed_readback(phase,original,block,tap,block_inside,tap_inside)),
    Equal.cong(Kernel.Phase,U32,selected => Offsets.value(Weights.phase_table(selected,Kernel.tiling(g),0),U32.from_nat(tap)),Kernel.phase_of(g,U32.from_nat(Index.ordinal(phase))),Jobs.context(phase),Index.phase_matches(phase)))

def packed_weight_sample(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat,
  +block_inside: {Nat.is_lt(block,block_count(phase)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))) == True{} : Bool}) ->
  {weight(PackedValues.read(packed_query(phase,original,block,tap),PackedValues.complete(packed_query(phase,original,block,tap),PackedValues.initial(packed_query(phase,original,block,tap))))) == selected_sample(phase,original,block,tap) : Tile.Tile8}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +pq = packed_query(phase,original,block,tap)
  +packed = PackedValues.complete(pq,PackedValues.initial(pq))
  +mq = query(phase,original,block,tap,block_inside,tap_inside)
  +allocated = Window.allocation(call,U32.from_nat(y),U32.from_nat(x))
  +address = U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),tap))
  +tree = Buffer.tree(~Tile.Tile8,Weights.phase_weights(Jobs.context(phase),original))
  Equal.trans(Tile.Tile8,weight(PackedValues.read(pq,packed)),Tree.read(Tile.Tile8,tree,address),selected_sample(phase,original,block,tap),packed_weight_readback(phase,original,block,tap,block_inside,tap_inside),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,address),Tree.read(Tile.Tile8,tree,Mapped.query_address(~Kernel.WeightWindow,mq)),selected_sample(phase,original,block,tap),Equal.cong(U32,Tile.Tile8,Tree.read(Tile.Tile8,tree),address,Mapped.query_address(~Kernel.WeightWindow,mq),Equal.sym(U32,Mapped.query_address(~Kernel.WeightWindow,mq),address,address_from(phase,original,block,tap,block_inside,tap_inside,allocated))),
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,Mapped.query_address(~Kernel.WeightWindow,mq)),Mapped.query_wanted(~Kernel.WeightWindow,~Kernel.weight_window_offset,mq),selected_sample(phase,original,block,tap),readback(phase,original,block,tap,block_inside,tap_inside),wanted_from(phase,original,block,tap,block_inside,tap_inside,allocated))))

def selected_sample_phase(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat) ->
  {selected_sample(phase,original,block,tap) == Samples.sample_linear(Buffer.tree(~F32,original),Domain.channel_word(Jobs.geometry(phase)),Weights.spatial(Jobs.context(phase)),(U32.from_nat(block) * 8 : U32),
    Kernel.weight_offset(Jobs.context(phase),U32.from_nat(tap))) : Tile.Tile8}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Window.slim_sample(Buffer.tree(~F32,original),c,(k * k : U32),(U32.from_nat(block) * 8 : U32),Jobs.context(phase),U32.from_nat(tap))

def full_state(+phase: Jobs.Phase,+original: Buffer.Model<F32>) -> Weights.Packed:
  +q = packed_query(phase,original,0n,0n)
  PackedValues.complete(q,PackedValues.initial(q))
def full_state_matches(+phase: Jobs.Phase,+original: Buffer.Model<F32>,+block: Nat,+tap: Nat) ->
  {PackedValues.complete(packed_query(phase,original,block,tap),PackedValues.initial(packed_query(phase,original,block,tap))) == full_state(phase,original) : Weights.Packed}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
