# Every leaf of the band partition holds the specified input gradient at every
# point of its band: the leaf theorem at a band of an image, on the fresh local
# output of the band, with its premises taken from the image the band belongs
# to.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_extents.bend as Extents
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_producer.bend as Producer
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_coverage.bend as Coverage
import ../../lib/proofs/input_gradient_tiled_sweep_selected.bend as Selected
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_partition.bend as Partition
import ../../lib/proofs/input_gradient_tiled_retained.bend as Retained
import ../../lib/proofs/input_gradient_tiled_phase_index.bend as Index
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weights.bend as WeightPacking
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_readback.bend as StorageReadback
import ../../lib/proofs/storage_allocation_refinement.bend as Allocation
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/input_gradient_tiled_leaf_length.bend as LeafLength
import ../../lib/proofs/strided_range_bounds.bend as Strides
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/tensor_band.bend as Bands
import ../../lib/proofs/nat_interval.bend as Interval
import ../../lib/proofs/input_gradient_tiled_output_outside.bend as Outside
import ./input_gradient_tiled_leaf.bend as Leaf
import ./input_gradient_sources.bend as Sources
import ../input_gradient_specification.bend as Specification

# One image of a call: the gradient and weight buffers, the image index and
# the machine fit the leaf theorem asks for.
type Sample is Data:
  Sample{call: Domain.Call,envelope: Geometry.WordEnvelope,
    fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
    gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,index: U32,
    inside: {Nat.is_lt(U32.to_nat(index),Domain.batch(call)) == True{} : Bool}}
def image_call(image: Sample) -> Domain.Call:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  call
def image_gradient(image: Sample) -> Buffer.Model<F32>:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  gradient
def image_weights(image: Sample) -> Buffer.Model<F32>:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  weights
def image_index(image: Sample) -> U32:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  index
def setting(image: Sample) -> Retained.Setting:
  Sample{+call,envelope,fits,gradient,weights,+index,inside} = image
  Retained.Setting{call,envelope,gradient,Packing.image_base(Domain.geometry(call),index),Packing.plane_size(Domain.geometry(call))}
def task(+image: Sample) -> Kernel.Task:
  Kernel.Task{Domain.geometry(image_call(image)),Kernel.tiling(Domain.geometry(image_call(image)))}
def spec(+image: Sample) -> Specification.Problem:
  Sources.problem(Domain.geometry(image_call(image)),Sources.models_reader(image_gradient(image),image_weights(image)),U32.to_nat(image_index(image)))

# The packed weights and the table of a call: the kernel packs them once, before
# any image.
def packing(+image: Sample) -> Kernel.Packed:
  Packed.native_complete(PackingBounds.Query{False{},image_call(image)},Kernel.tiling(Domain.geometry(image_call(image))),image_weights(image))
def native_weights(state: Kernel.Packed) -> Array<Tile.Tile8>:
  Kernel.Packed{source,weights,offsets,weight_base,offset_base} = state
  weights
def native_offsets(state: Kernel.Packed) -> Array<U32>:
  Kernel.Packed{source,weights,offsets,weight_base,offset_base} = state
  offsets
def represent_weights(+state: WeightPacking.Packed) -> {native_weights(WeightPacking.represent(state)) == Tree.pack(Tile.Tile8,Packed.weight_tree(state)) : Array<Tile.Tile8>}:
  WeightPacking.Packed{source,weights,offsets,weight_base,offset_base} = state
  {==}
def represent_offsets(+state: WeightPacking.Packed) -> {native_offsets(WeightPacking.represent(state)) == Tree.pack(U32,Packed.offset_tree(state)) : Array<U32>}:
  WeightPacking.Packed{source,weights,offsets,weight_base,offset_base} = state
  {==}

# A leaf reads the packed arrays, whichever trees they came from.
def packed_leaf(+context: Kernel.LeafContext,-source: Kernel.Source,-weights: Array<Tile.Tile8>,-offsets: Array<U32>,
  -packed_weights: Array<Tile.Tile8>,-packed_offsets: Array<U32>,-output: Storage.Buffer<F32>,
  same_weights: {weights == packed_weights : Array<Tile.Tile8>},same_offsets: {offsets == packed_offsets : Array<U32>}) ->
  {Kernel.leaf_band(context,source,weights,offsets,output) == Kernel.leaf_band(context,source,packed_weights,packed_offsets,output) : Kernel.Part}:
  %Equal.sym(Array<Tile.Tile8>,weights,packed_weights,same_weights) :
    {Kernel.leaf_band(context,source,_,offsets,output) == Kernel.leaf_band(context,source,packed_weights,packed_offsets,output) : Kernel.Part}
  %Equal.sym(Array<U32>,offsets,packed_offsets,same_offsets) :
    {Kernel.leaf_band(context,source,packed_weights,_,output) == Kernel.leaf_band(context,source,packed_weights,packed_offsets,output) : Kernel.Part}
  {==}
def packed_run(+query: Chunks.Query,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+packed_weights: Tree.Tree<Tile.Tile8>,+packed_offsets: Tree.Tree<U32>,
  +output: Buffer.Model<F32>,same_weights: {Tree.pack(Tile.Tile8,weights) == Tree.pack(Tile.Tile8,packed_weights) : Array<Tile.Tile8>},
  same_offsets: {Tree.pack(U32,offsets) == Tree.pack(U32,packed_offsets) : Array<U32>}) ->
  {Selected.native_run(query,Selected.Packed{weights,offsets},output) == Selected.native_run(query,Selected.Packed{packed_weights,packed_offsets},output) : Kernel.Part}:
  packed_leaf(Selected.native_context(query),Execution.represent_source(Chunks.source(query)),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),
    Tree.pack(Tile.Tile8,packed_weights),Tree.pack(U32,packed_offsets),Buffer.pack(~F32,output),same_weights,same_offsets)

# Words below a bound stay below a larger bound.
def word_le_le(+a: U32,+b: U32,+c: U32,ab: {U32.is_le(a,b) == True{} : Bool},bc: {U32.is_le(b,c) == True{} : Bool}) -> {U32.is_le(a,c) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(a,c),Nat.is_le(U32.to_nat(a),U32.to_nat(c)),Comparison.le(a,c)) : {_ == True{} : Bool}
  Order.transitive(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c),Bounds.as_nat_le(a,b,ab),Bounds.as_nat_le(b,c,bc))
def sized_depth(+count: U32,sized: Depth.Sized<count>) -> {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool}:
  Depth.Sized{depth_ok,room} = sized
  depth_ok

# The output layout of a leaf, and its capacity by name.
def leaf_layout(+query: Chunks.Query,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +plane_span: {Nat.is_le(Output.pixels(query),U32.to_nat(Chunks.plane(query))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(Domain.channels(Chunks.geometry(query)),U32.to_nat(Chunks.plane(query)))),U32.to_nat(capacity)) == True{} : Bool}) -> Output.Layout<query>:
  Output.Layout{capacity,capacity_ok,plane_span,room}
def layout_capacity(+query: Chunks.Query,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +plane_span: {Nat.is_le(Output.pixels(query),U32.to_nat(Chunks.plane(query))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(Domain.channels(Chunks.geometry(query)),U32.to_nat(Chunks.plane(query)))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {capacity == Output.capacity(query,leaf_layout(query,capacity,capacity_ok,plane_span,room)) : U32}:
  {==}

# The geometry of an image, of the output as bands over the batch, and the
# words that place the image in the output.
def geometry(+image: Sample) -> Kernel.Geometry:
  Domain.geometry(image_call(image))
def band_geometry(image: Sample) -> Bands.Geometry:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Bands.Geometry{U32.from_nat(batch),c,h,w}
def start(image: Sample) -> U32:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  (index * h : U32)
def target(image: Sample) -> U32:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  (index * c * h * w : U32)
def plane(image: Sample) -> U32:
  Sample{call,envelope,fits,gradient,weights,index,inside} = image
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  (h * w : U32)

# The address of channel, row and column of a band of rows from first in an
# output whose planes start at target and are plane apart.
def address(+image: Sample,+first: U32,+target: U32,+plane: U32,+channel: Nat,+row: Nat,+column: Nat) -> U32:
  U32.from_nat(Nat.add(Strides.start(U32.to_nat(target),0n,U32.to_nat(plane),channel),Nat.add(Nat.mul(Nat.sub(row,U32.to_nat(first)),Domain.width(geometry(image))),column)))

# The local output of a leaf of rows rows: [channels][rows][width] values.
def local_count(+image: Sample,+rows: U32) -> U32:
  Partition.local_count(geometry(image),rows)
def local(+image: Sample,+rows: U32) -> Buffer.Model<F32>:
  Partition.local_model(local_count(image,rows))
def local_plane(+image: Sample,+rows: U32) -> U32:
  (rows * Domain.width_word(geometry(image)) : U32)
def local_address(+image: Sample,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat) -> U32:
  address(image,first,0,local_plane(image,rows),channel,row,column)
def local_part(+image: Sample,+first: U32,+rows: U32,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>) -> Execution.Part:
  Execution.leaf_band(Kernel.LeafContext{task(image),first,rows,0,local_plane(image,rows)},source,weights,offsets,local(image,rows))

# What the fresh local output of a band satisfies.
type Local<-image: Sample,-rows: U32> is Data:
  Local{count_exact: {U32.to_nat(local_count(image,rows)) == Nat.mul(Nat.mul(Domain.channels(geometry(image)),U32.to_nat(rows)),Domain.width(geometry(image))) : Nat},
    plane_exact: {U32.to_nat(local_plane(image,rows)) == Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))) : Nat},
    tree_equal: {Buffer.tree(~F32,local(image,rows)) == Tree.fresh(F32,Tensors.depth_for(local_count(image,rows)),(0.0 + 0.0 : F32)) : Tree.Tree<F32>},
    length_exact: {Buffer.length(~F32,local(image,rows)) == U32.to_nat(local_count(image,rows)) : Nat},
    depth_ok: {Nat.is_le(Buffer.depth(~F32,local(image,rows)),30n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))),2147483648) == True{} : Bool},
    count_room: {Nat.is_le(U32.to_nat(local_count(image,rows)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))))) == True{} : Bool},
    plane_span: {Nat.is_le(Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))),U32.to_nat(local_plane(image,rows))) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Domain.channels(geometry(image)),U32.to_nat(local_plane(image,rows))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))))) == True{} : Bool}}
def local_count_exact(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {U32.to_nat(local_count(image,rows)) == Nat.mul(Nat.mul(Domain.channels(geometry(image)),U32.to_nat(rows)),Domain.width(geometry(image))) : Nat}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  count_exact
def local_plane_exact(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {U32.to_nat(local_plane(image,rows)) == Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))) : Nat}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  plane_exact
def local_tree_equal(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {Buffer.tree(~F32,local(image,rows)) == Tree.fresh(F32,Tensors.depth_for(local_count(image,rows)),(0.0 + 0.0 : F32)) : Tree.Tree<F32>}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  tree_equal
def local_length_exact(-image: Sample,-rows: U32,facts: Local<image,rows>) -> {Buffer.length(~F32,local(image,rows)) == U32.to_nat(local_count(image,rows)) : Nat}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  length_exact
def local_depth_ok(-image: Sample,-rows: U32,facts: Local<image,rows>) -> {Nat.is_le(Buffer.depth(~F32,local(image,rows)),30n) == True{} : Bool}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  depth_ok
def local_capacity_ok(-image: Sample,-rows: U32,facts: Local<image,rows>) -> {U32.is_le(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))),2147483648) == True{} : Bool}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  capacity_ok
def local_count_room(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {Nat.is_le(U32.to_nat(local_count(image,rows)),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))))) == True{} : Bool}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  count_room
def local_plane_span(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))),U32.to_nat(local_plane(image,rows))) == True{} : Bool}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  plane_span
def local_room(-image: Sample,-rows: U32,facts: Local<image,rows>) ->
  {Nat.is_le(Nat.mul(Domain.channels(geometry(image)),U32.to_nat(local_plane(image,rows))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,local(image,rows))))) == True{} : Bool}:
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room} = facts
  room

def local_facts(+image: Sample,+first: U32,+rows: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(geometry(image))) == True{} : Bool}) -> Local<image,rows>:
  Sample{+call,+envelope,+fits,+gradient,+weights,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +height_n = U32.to_nat(h)
  +width_n = U32.to_nat(w)
  +channels_n = U32.to_nat(c)
  +contained = Order.transitive(rows_n,Nat.add(first_n,rows_n),height_n,
    %Order.commutative(rows_n,first_n) : {Nat.is_le(rows_n,_) == True{} : Bool} Order.le_add(rows_n,first_n),room)
  +plane_word = (rows * w : U32)
  +count = (c * rows * w : U32)
  +count_n = Nat.mul(Nat.mul(channels_n,rows_n),width_n)
  +model = Partition.local_model(count)
  +capacity_word = Machine.capacity(limits)
  +capacity_n = U32.to_nat(capacity_word)
  +positive_width = Domain.width_positive(whole)
  +rows_area = RowMajor.mul_le(rows_n,height_n,width_n,contained)
  +volume_bound = {%Equal.sym(Nat,Nat.mul(Nat.mul(channels_n,rows_n),width_n),Nat.mul(channels_n,Nat.mul(rows_n,width_n)),RowMajor.assoc(channels_n,rows_n,width_n)) :
      {Nat.is_le(_,capacity_n) == True{} : Bool}
    Order.transitive(Nat.mul(channels_n,Nat.mul(rows_n,width_n)),Nat.mul(channels_n,Nat.mul(height_n,width_n)),capacity_n,
      DivisionOrder.multiply_left_le(channels_n,Nat.mul(rows_n,width_n),Nat.mul(height_n,width_n),rows_area),Domain.image_room(whole)) :
    {Nat.is_le(count_n,capacity_n) == True{} : Bool}}
  +count_exact = Encoded.multiply_three(c,rows,w,capacity_word,Limits.capacity_fits(limits),positive_width,volume_bound)
  +count_fits = {%Equal.sym(Bool,U32.is_le(count,capacity_word),Nat.is_le(U32.to_nat(count),capacity_n),Comparison.le(count,capacity_word)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),count_n,count_exact) : {Nat.is_le(_,capacity_n) == True{} : Bool}
    volume_bound : {U32.is_le(count,capacity_word) == True{} : Bool}}
  +sized = Depth.bounded(count,word_le_le(count,capacity_word,1073741824,count_fits,Limits.capacity_allocatable(limits)))
  +depth = Tensors.depth_for(count)
  +zero = (0.0 + 0.0 : F32)
  +fresh = Tree.fresh(F32,depth,zero)
  +content_equal = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.contents(~F32,Tensors.buffer(~F32,zero,count)),Buffer.Content{fresh,U32.to_nat(count)},
    Buffer.remodel_content(~F32,Tensors.buffer(~F32,zero,count)),Allocation.contents(~F32,depth,zero,U32.to_nat(count)))
  +tree_equal = Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,model),Buffer.content_tree(~F32,Buffer.content(~F32,model)),fresh,
    Equal.sym(Tree.Tree<F32>,Buffer.content_tree(~F32,Buffer.content(~F32,model)),Buffer.tree(~F32,model),Buffer.content_tree_of(~F32,model)),
    Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,content => Buffer.content_tree(~F32,content),Buffer.content(~F32,model),Buffer.Content{fresh,U32.to_nat(count)},content_equal))
  +length_exact = Equal.trans(Nat,Buffer.length(~F32,model),Buffer.content_length(~F32,Buffer.content(~F32,model)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),Buffer.content_length_of(~F32,model)),
    Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,model),Buffer.Content{fresh,U32.to_nat(count)},content_equal))
  +balanced = Buffer.balanced(~F32,model)
  +fresh_balanced = {%Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),fresh,tree_equal) : Shape.Balanced(F32,depth,_)
    Shape.fresh(F32,depth,zero) : Shape.Balanced(F32,depth,Buffer.tree(~F32,model))}
  +depth_equal = Shape.unique_depth(F32,Buffer.tree(~F32,model),Buffer.depth(~F32,model),depth,balanced,fresh_balanced)
  +depth_ok = {%Equal.sym(Nat,Buffer.depth(~F32,model),depth,depth_equal) : {Nat.is_le(_,30n) == True{} : Bool}
    sized_depth(count,sized) : {Nat.is_le(Buffer.depth(~F32,model),30n) == True{} : Bool}}
  +tree_capacity = Tree.capacity(F32,Buffer.tree(~F32,model))
  +capacity_ok = Buffer.capacity_bound(~F32,model,depth_ok)
  +plane_exact = FullRange.multiply(rows,w,capacity_word,Order.transitive(Nat.mul(rows_n,width_n),Nat.mul(height_n,width_n),capacity_n,rows_area,Domain.plane_room(whole)))
  +plane_span = {%Equal.sym(Nat,U32.to_nat(plane_word),Nat.mul(rows_n,width_n),plane_exact) : {Nat.is_le(Nat.mul(rows_n,width_n),_) == True{} : Bool}
    Order.reflexive(Nat.mul(rows_n,width_n)) : {Nat.is_le(Nat.mul(rows_n,width_n),U32.to_nat(plane_word)) == True{} : Bool}}
  +capacity_equal = Shape.capacity(F32,Buffer.depth(~F32,model),Buffer.tree(~F32,model),balanced)
  +count_room = {%Equal.sym(U32,tree_capacity,U32.shln(1,Buffer.depth(~F32,model)),capacity_equal) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,Buffer.depth(~F32,model),depth,depth_equal) : {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,_))) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(tree_capacity)) == True{} : Bool}}
  +room_local = {%Equal.sym(Nat,U32.to_nat(plane_word),Nat.mul(rows_n,width_n),plane_exact) : {Nat.is_le(Nat.mul(channels_n,_),U32.to_nat(tree_capacity)) == True{} : Bool}
    %RowMajor.assoc(channels_n,rows_n,width_n) : {Nat.is_le(_,U32.to_nat(tree_capacity)) == True{} : Bool}
    %count_exact : {Nat.is_le(_,U32.to_nat(tree_capacity)) == True{} : Bool}
    count_room : {Nat.is_le(Nat.mul(channels_n,U32.to_nat(plane_word)),U32.to_nat(tree_capacity)) == True{} : Bool}}
  Local{count_exact,plane_exact,tree_equal,length_exact,depth_ok,capacity_ok,count_room,plane_span,room_local}

# What a leaf run on the given trees stores is what the native run on the
# packed arrays stores.
def part_read(+query: Chunks.Query,+given: Selected.Packed,+packed: Selected.Packed,+output: Buffer.Model<F32>,
  +same: {Selected.native_run(query,given,output) == Selected.native_run(query,packed,output) : Kernel.Part},+cell: U32) ->
  {Tree.read(F32,Buffer.tree(~F32,Selected.part_output(Selected.run(query,given,output))),cell) == Selected.native_value(Selected.native_run(query,packed,output),cell) : F32}:
  +part = Selected.run(query,given,output)
  Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),cell),Selected.native_value(Selected.native_run(query,given,output),cell),
    Selected.native_value(Selected.native_run(query,packed,output),cell),
    Equal.sym(F32,Selected.native_value(Selected.native_run(query,given,output),cell),Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),cell),
      Equal.trans(F32,Selected.native_value(Selected.native_run(query,given,output),cell),Selected.native_value(Execution.represent_part(part),cell),
        Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),cell),
        Equal.cong(Kernel.Part,F32,native => Selected.native_value(native,cell),Selected.native_run(query,given,output),Execution.represent_part(part),Selected.run_matches(query,given,output)),
        Selected.represented_value(part,cell))),
    Equal.cong(Kernel.Part,F32,native => Selected.native_value(native,cell),Selected.native_run(query,given,output),Selected.native_run(query,packed,output),same))

# An address outside the planes of the leaf, as the leaf theorem names it.
def excluded_query(+context: Leaf.Context,+index: U32,
  outside: Interval.Outside<U32.to_nat(Chunks.target(Leaf.query(context))),Nat.mul(Domain.channels(Chunks.geometry(Leaf.query(context))),U32.to_nat(Chunks.plane(Leaf.query(context)))),U32.to_nat(index)>) ->
  Outside.Outside<Leaf.query(context),index>:
  match outside:
    case Interval.Before{bound}: Outside.Before{bound}
    case Interval.After{bound}: Outside.After{bound}

# The leaf of a band holds the specified gradient at every point of its band,
# in any output whose planes start at target and are plane apart, when the
# point of the output is still zero.
def band_value(+image: Sample,+first: U32,+rows: U32,+source: Execution.Source,+held: Retained.Held<setting(image),first,rows,source>,
  +weights_tree: Tree.Tree<Tile.Tile8>,+offsets_tree: Tree.Tree<U32>,
  +same_weights: {Tree.pack(Tile.Tile8,weights_tree) == native_weights(packing(image)) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets_tree) == native_offsets(packing(image)) : Array<U32>},
  +target: U32,+plane: U32,+output: Buffer.Model<F32>,
  +depth_ok: {Nat.is_le(Buffer.depth(~F32,output),30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(F32,Buffer.tree(~F32,output)),2147483648) == True{} : Bool},
  +plane_span: {Nat.is_le(Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))),U32.to_nat(plane)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(target),Nat.mul(Domain.channels(geometry(image)),U32.to_nat(plane))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,Domain.channels(geometry(image))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),row) == True{} : Bool},
  +upper: {Nat.is_lt(row,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(geometry(image))) == True{} : Bool},
  +initial_zero: {Tree.read(F32,Buffer.tree(~F32,output),address(image,first,target,plane,channel,row,column)) == (0.0 + 0.0 : F32) : F32}) ->
  {Tree.read(F32,LeafWrites.part_tree(Execution.leaf_band(Kernel.LeafContext{task(image),first,rows,target,plane},source,weights_tree,offsets_tree,output)),
      address(image,first,target,plane,channel,row,column)) ==
    Specification.value(spec(image),channel,row,column) : F32}:
  Sample{+call,+envelope,+fits,+gradient,+weights,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Retained.Held{+contained_rows,witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +sample = {Sample{whole,envelope,fits,gradient,weights,index,inside} : Sample}
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +height_n = U32.to_nat(h)
  +contained = Order.transitive(rows_n,Nat.add(first_n,rows_n),height_n,
    %Order.commutative(rows_n,first_n) : {Nat.is_le(rows_n,_) == True{} : Bool} Order.le_add(rows_n,first_n),contained_rows)
  +grid = {GeometryChunks.fitted_grid(whole,envelope,rows,contained,fits) : GeometryChunks.Grid}
  +query = {Chunks.Query{grid,first,target,plane,source,0n,0n,0,0} : Chunks.Query}
  +point = {Domain.Point{U32.to_nat(index),channel,row,column} : Domain.Point}
  +row_inside = Bounds.nat_trans(row,Nat.add(first_n,rows_n),height_n,upper,contained_rows)
  +problem = {Producer.Problem{query,point,Domain.Inside{inside,channel_inside,row_inside,column_inside},lower,upper,contained_rows} : Producer.Problem}
  +tree_capacity = Tree.capacity(F32,Buffer.tree(~F32,output))
  +layout = leaf_layout(Producer.query(problem),tree_capacity,capacity_ok,plane_span,room)
  +context = {Leaf.Context{problem,index,gradient,weights,output,witness,fits,layout,
    layout_capacity(Producer.query(problem),tree_capacity,capacity_ok,plane_span,room),depth_ok,initial_zero} : Leaf.Context}
  +state = Weights.full_state(Index.zero(grid),weights)
  +bounds = {PackingBounds.Query{False{},whole} : PackingBounds.Query}
  +tiling = Kernel.tiling(shape)
  +matches = {Packed.native_complete_matches(bounds,tiling,weights,Index.ordinal(Index.zero(grid)),
      Nat.add(Nat.mul(0n,U32.to_nat(Kernel.phase_inner(Jobs.context(Index.zero(grid))))),0n),0n) : {Packed.native_complete(bounds,tiling,weights) == WeightPacking.represent(state) : Kernel.Packed}}
  +weights_match = Equal.trans(Array<Tile.Tile8>,Tree.pack(Tile.Tile8,weights_tree),native_weights(Packed.native_complete(bounds,tiling,weights)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),same_weights,
    Equal.trans(Array<Tile.Tile8>,native_weights(Packed.native_complete(bounds,tiling,weights)),native_weights(WeightPacking.represent(state)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),
      Equal.cong(Kernel.Packed,Array<Tile.Tile8>,packed => native_weights(packed),Packed.native_complete(bounds,tiling,weights),WeightPacking.represent(state),matches),represent_weights(state)))
  +offsets_match = Equal.trans(Array<U32>,Tree.pack(U32,offsets_tree),native_offsets(Packed.native_complete(bounds,tiling,weights)),Tree.pack(U32,Packed.offset_tree(state)),same_offsets,
    Equal.trans(Array<U32>,native_offsets(Packed.native_complete(bounds,tiling,weights)),native_offsets(WeightPacking.represent(state)),Tree.pack(U32,Packed.offset_tree(state)),
      Equal.cong(Kernel.Packed,Array<U32>,packed => native_offsets(packed),Packed.native_complete(bounds,tiling,weights),WeightPacking.represent(state),matches),represent_offsets(state)))
  +given = {Selected.Packed{weights_tree,offsets_tree} : Selected.Packed}
  +packed = {Selected.Packed{Packed.weight_tree(state),Packed.offset_tree(state)} : Selected.Packed}
  +cell = Coverage.address(problem)
  +part = Selected.run(query,given,output)
  %Equal.sym(Tree.Tree<F32>,LeafWrites.part_tree(Execution.leaf_band(Kernel.LeafContext{task(sample),first,rows,target,plane},source,weights_tree,offsets_tree,output)),
      Buffer.tree(~F32,Selected.part_output(part)),Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,Selected.part_output(part)),LeafWrites.part_tree(part),Leaf.run_tree(part))) :
    {Tree.read(F32,_,address(sample,first,target,plane,channel,row,column)) == Specification.value(spec(sample),channel,row,column) : F32}
  Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),cell),Selected.native_value(Selected.native_run(query,packed,output),cell),Leaf.wanted(context),
    part_read(query,given,packed,output,packed_run(query,weights_tree,offsets_tree,Packed.weight_tree(state),Packed.offset_tree(state),output,weights_match,offsets_match),cell),
    Leaf.value(context))

# The leaf of a band leaves every address outside its planes as it was.
def band_outside(+image: Sample,+first: U32,+rows: U32,+source: Execution.Source,+held: Retained.Held<setting(image),first,rows,source>,
  +weights_tree: Tree.Tree<Tile.Tile8>,+offsets_tree: Tree.Tree<U32>,
  +same_weights: {Tree.pack(Tile.Tile8,weights_tree) == native_weights(packing(image)) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets_tree) == native_offsets(packing(image)) : Array<U32>},
  +target: U32,+plane: U32,+output: Buffer.Model<F32>,
  +depth_ok: {Nat.is_le(Buffer.depth(~F32,output),30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(F32,Buffer.tree(~F32,output)),2147483648) == True{} : Bool},
  +plane_span: {Nat.is_le(Nat.mul(U32.to_nat(rows),Domain.width(geometry(image))),U32.to_nat(plane)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(target),Nat.mul(Domain.channels(geometry(image)),U32.to_nat(plane))),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool},
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,Domain.channels(geometry(image))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),row) == True{} : Bool},
  +upper: {Nat.is_lt(row,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(geometry(image))) == True{} : Bool},
  +initial_zero: {Tree.read(F32,Buffer.tree(~F32,output),address(image,first,target,plane,channel,row,column)) == (0.0 + 0.0 : F32) : F32},
  +probe: U32,+excluded: Interval.Outside<U32.to_nat(target),Nat.mul(Domain.channels(geometry(image)),U32.to_nat(plane)),U32.to_nat(probe)>,
  +probe_inside: {U32.is_lt(probe,Tree.capacity(F32,Buffer.tree(~F32,output))) == True{} : Bool}) ->
  {Tree.read(F32,LeafWrites.part_tree(Execution.leaf_band(Kernel.LeafContext{task(image),first,rows,target,plane},source,weights_tree,offsets_tree,output)),probe) ==
    Tree.read(F32,Buffer.tree(~F32,output),probe) : F32}:
  Sample{+call,+envelope,+fits,+gradient,+weights,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Retained.Held{+contained_rows,witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +sample = {Sample{whole,envelope,fits,gradient,weights,index,inside} : Sample}
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +height_n = U32.to_nat(h)
  +contained = Order.transitive(rows_n,Nat.add(first_n,rows_n),height_n,
    %Order.commutative(rows_n,first_n) : {Nat.is_le(rows_n,_) == True{} : Bool} Order.le_add(rows_n,first_n),contained_rows)
  +grid = {GeometryChunks.fitted_grid(whole,envelope,rows,contained,fits) : GeometryChunks.Grid}
  +query = {Chunks.Query{grid,first,target,plane,source,0n,0n,0,0} : Chunks.Query}
  +point = {Domain.Point{U32.to_nat(index),channel,row,column} : Domain.Point}
  +row_inside = Bounds.nat_trans(row,Nat.add(first_n,rows_n),height_n,upper,contained_rows)
  +problem = {Producer.Problem{query,point,Domain.Inside{inside,channel_inside,row_inside,column_inside},lower,upper,contained_rows} : Producer.Problem}
  +tree_capacity = Tree.capacity(F32,Buffer.tree(~F32,output))
  +layout = leaf_layout(Producer.query(problem),tree_capacity,capacity_ok,plane_span,room)
  +context = {Leaf.Context{problem,index,gradient,weights,output,witness,fits,layout,
    layout_capacity(Producer.query(problem),tree_capacity,capacity_ok,plane_span,room),depth_ok,initial_zero} : Leaf.Context}
  +state = Weights.full_state(Index.zero(grid),weights)
  +bounds = {PackingBounds.Query{False{},whole} : PackingBounds.Query}
  +tiling = Kernel.tiling(shape)
  +matches = {Packed.native_complete_matches(bounds,tiling,weights,Index.ordinal(Index.zero(grid)),
      Nat.add(Nat.mul(0n,U32.to_nat(Kernel.phase_inner(Jobs.context(Index.zero(grid))))),0n),0n) : {Packed.native_complete(bounds,tiling,weights) == WeightPacking.represent(state) : Kernel.Packed}}
  +weights_match = Equal.trans(Array<Tile.Tile8>,Tree.pack(Tile.Tile8,weights_tree),native_weights(Packed.native_complete(bounds,tiling,weights)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),same_weights,
    Equal.trans(Array<Tile.Tile8>,native_weights(Packed.native_complete(bounds,tiling,weights)),native_weights(WeightPacking.represent(state)),Tree.pack(Tile.Tile8,Packed.weight_tree(state)),
      Equal.cong(Kernel.Packed,Array<Tile.Tile8>,packed => native_weights(packed),Packed.native_complete(bounds,tiling,weights),WeightPacking.represent(state),matches),represent_weights(state)))
  +offsets_match = Equal.trans(Array<U32>,Tree.pack(U32,offsets_tree),native_offsets(Packed.native_complete(bounds,tiling,weights)),Tree.pack(U32,Packed.offset_tree(state)),same_offsets,
    Equal.trans(Array<U32>,native_offsets(Packed.native_complete(bounds,tiling,weights)),native_offsets(WeightPacking.represent(state)),Tree.pack(U32,Packed.offset_tree(state)),
      Equal.cong(Kernel.Packed,Array<U32>,packed => native_offsets(packed),Packed.native_complete(bounds,tiling,weights),WeightPacking.represent(state),matches),represent_offsets(state)))
  +given = {Selected.Packed{weights_tree,offsets_tree} : Selected.Packed}
  +packed = {Selected.Packed{Packed.weight_tree(state),Packed.offset_tree(state)} : Selected.Packed}
  +part = Selected.run(query,given,output)
  %Equal.sym(Tree.Tree<F32>,LeafWrites.part_tree(Execution.leaf_band(Kernel.LeafContext{task(sample),first,rows,target,plane},source,weights_tree,offsets_tree,output)),
      Buffer.tree(~F32,Selected.part_output(part)),Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,Selected.part_output(part)),LeafWrites.part_tree(part),Leaf.run_tree(part))) :
    {Tree.read(F32,_,probe) == Tree.read(F32,Buffer.tree(~F32,output),probe) : F32}
  Equal.trans(F32,Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),probe),Selected.native_value(Selected.native_run(query,packed,output),probe),Tree.read(F32,Buffer.tree(~F32,output),probe),
    part_read(query,given,packed,output,packed_run(query,weights_tree,offsets_tree,Packed.weight_tree(state),Packed.offset_tree(state),output,weights_match,offsets_match),probe),
    Leaf.outside(context,probe,excluded_query(context,probe,excluded),
      %Leaf.capacity(context) : {U32.is_lt(probe,_) == True{} : Bool}
      probe_inside))

# The leaf of a band on its fresh local output.
def leaf_value(+image: Sample,+first: U32,+rows: U32,+source: Execution.Source,+held: Retained.Held<setting(image),first,rows,source>,
  +weights_tree: Tree.Tree<Tile.Tile8>,+offsets_tree: Tree.Tree<U32>,
  +same_weights: {Tree.pack(Tile.Tile8,weights_tree) == native_weights(packing(image)) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets_tree) == native_offsets(packing(image)) : Array<U32>},
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,Domain.channels(geometry(image))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),row) == True{} : Bool},
  +upper: {Nat.is_lt(row,Nat.add(U32.to_nat(first),U32.to_nat(rows))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(geometry(image))) == True{} : Bool}) ->
  {Tree.read(F32,LeafWrites.part_tree(local_part(image,first,rows,source,weights_tree,offsets_tree)),local_address(image,first,rows,channel,row,column)) ==
    Specification.value(spec(image),channel,row,column) : F32}:
  Sample{+call,+envelope,+fits,+gradient,+weights,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Retained.Held{+room,+witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +sample = {Sample{whole,envelope,fits,gradient,weights,index,inside} : Sample}
  +facts = local_facts(sample,first,rows,room)
  +depth = Tensors.depth_for(local_count(sample,rows))
  +cell = local_address(sample,first,rows,channel,row,column)
  +initial_zero = {%Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,local(sample,rows)),Tree.fresh(F32,depth,(0.0 + 0.0 : F32)),local_tree_equal(sample,rows,facts)) :
      {Tree.read(F32,_,cell) == (0.0 + 0.0 : F32) : F32}
    StorageReadback.read_fresh(~F32,depth,(0.0 + 0.0 : F32),cell) : {Tree.read(F32,Buffer.tree(~F32,local(sample,rows)),cell) == (0.0 + 0.0 : F32) : F32}}
  band_value(sample,first,rows,source,Retained.Held{room,witness},weights_tree,offsets_tree,same_weights,same_offsets,0,local_plane(sample,rows),local(sample,rows),
    local_depth_ok(sample,rows,facts),local_capacity_ok(sample,rows,facts),local_plane_span(sample,rows,facts),local_room(sample,rows,facts),
    channel,row,column,channel_inside,lower,upper,column_inside,initial_zero)

# What the band of a leaf holds: its rows lie in the image, its local output
# has the cells of its band and is ready, and every cell holds the specified
# gradient of its row of the image.
def Cell(+image: Sample,+part: Execution.Part,+first: U32,+rows: U32,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  {Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),U32.from_nat(BandValues.position(band_geometry(image),rows,channel,row,column))) ==
    Specification.value(spec(image),channel,Nat.add(U32.to_nat(first),row),column) : F32}
type Band<-image: Sample,-part: Execution.Part,-first: U32,-rows: U32> is Data:
  Band{room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Domain.height(geometry(image))) == True{} : Bool},
    length: {Buffer.length(~F32,Selected.part_output(part)) == BandValues.cell_count(band_geometry(image),rows) : Nat},
    sized: Buffer.Sized<F32,Selected.part_output(part)>,
    cells: Members.All(Nat,{channel => Members.All(Nat,{row => Members.All(Nat,{column => Cell(image,part,first,rows,channel,row,column) : Nat -> Data},
      Ranges.range(Domain.width(geometry(image)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(rows),0n)) : Nat -> Data},Ranges.range(Domain.channels(geometry(image)),0n))}
def band(+image: Sample,+part: Execution.Part,+first: U32,+rows: U32) -> Data:
  Band<image,part,first,rows>

# A leaf, with what its cells need.
type Scope is Data:
  Scope{image: Sample,first: U32,rows: U32,source: Execution.Source,held: Retained.Held<setting(image),first,rows,source>,
    weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,
    same_weights: {Tree.pack(Tile.Tile8,weights) == native_weights(packing(image)) : Array<Tile.Tile8>},
    same_offsets: {Tree.pack(U32,offsets) == native_offsets(packing(image)) : Array<U32>}}
def scope_cell(+scope: Scope,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  Scope{image,first,rows,source,held,weights,offsets,same_weights,same_offsets} = scope
  Cell(image,local_part(image,first,rows,source,weights,offsets),first,rows,channel,row,column)
def scope_channels(+scope: Scope) -> Nat:
  Scope{image,first,rows,source,held,weights,offsets,same_weights,same_offsets} = scope
  Domain.channels(geometry(image))
def scope_rows(+scope: Scope) -> Nat:
  Scope{image,first,rows,source,held,weights,offsets,same_weights,same_offsets} = scope
  U32.to_nat(rows)
def scope_width(+scope: Scope) -> Nat:
  Scope{image,first,rows,source,held,weights,offsets,same_weights,same_offsets} = scope
  Domain.width(geometry(image))
def local_cell(+scope: Scope,+channel: Nat,+row: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,scope_channels(scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,scope_rows(scope)) == True{} : Bool},+column_inside: {Nat.is_lt(column,scope_width(scope)) == True{} : Bool}) ->
  scope_cell(scope,channel,row,column):
  Scope{+image,+first,+rows,+source,+held,+weights,+offsets,+same_weights,+same_offsets} = scope
  Sample{+call,+envelope,+fits,+gradient,+weight_model,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Retained.Held{+room,+witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +sample = {Sample{whole,envelope,fits,gradient,weight_model,index,inside} : Sample}
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +width_n = U32.to_nat(w)
  +local_facts = local_facts(sample,first,rows,room)
  +part = local_part(sample,first,rows,source,weights,offsets)
  %local_plane_exact(sample,rows,local_facts) :
    {Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),U32.from_nat(Nat.add(Nat.mul(channel,_),Nat.add(Nat.mul(row,width_n),column)))) ==
      Specification.value(spec(sample),channel,Nat.add(first_n,row),column) : F32}
  %Order.subtract_shift(first_n,row) :
    {Tree.read(F32,Buffer.tree(~F32,Selected.part_output(part)),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(local_plane(sample,rows))),Nat.add(Nat.mul(_,width_n),column)))) ==
      Specification.value(spec(sample),channel,Nat.add(first_n,row),column) : F32}
  %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,Selected.part_output(part)),LeafWrites.part_tree(part),Leaf.run_tree(part)) :
    {Tree.read(F32,_,U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(local_plane(sample,rows))),Nat.add(Nat.mul(Nat.sub(Nat.add(first_n,row),first_n),width_n),column)))) ==
      Specification.value(spec(sample),channel,Nat.add(first_n,row),column) : F32}
  leaf_value(sample,first,rows,source,Retained.Held{room,witness},weights,offsets,same_weights,same_offsets,channel,Nat.add(first_n,row),column,
    channel_inside,Order.le_add(first_n,row),Order.offset_lt(first_n,row,rows_n,row_inside),column_inside)

def band_of(+image: Sample,+first: U32,+rows: U32,+source: Execution.Source,+held: Retained.Held<setting(image),first,rows,source>,
  +weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == native_weights(packing(image)) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == native_offsets(packing(image)) : Array<U32>}) ->
  Band<image,local_part(image,first,rows,source,weights,offsets),first,rows>:
  Sample{+call,+envelope,+fits,+gradient,+weight_model,+index,+inside} = image
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Retained.Held{+room,+witness} = held
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +sample = {Sample{whole,envelope,fits,gradient,weight_model,index,inside} : Sample}
  +channels_n = U32.to_nat(c)
  +rows_n = U32.to_nat(rows)
  +width_n = U32.to_nat(w)
  +local_facts = local_facts(sample,first,rows,room)
  +fresh = local(sample,rows)
  +context = {Kernel.LeafContext{task(sample),first,rows,0,local_plane(sample,rows)} : Kernel.LeafContext}
  +part = local_part(sample,first,rows,source,weights,offsets)
  +output = Selected.part_output(part)
  +writes = LeafWrites.leaf_writes(context,source,weights,offsets)
  +written = Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,output),LeafWrites.part_tree(part),Writes.pure(F32,writes,Buffer.tree(~F32,fresh)),
    Leaf.run_tree(part),LeafWrites.leaf_tree_writes(context,source,weights,offsets,fresh))
  +length = Equal.trans(Nat,Buffer.length(~F32,output),Buffer.length(~F32,fresh),Nat.mul(channels_n,Nat.mul(rows_n,width_n)),
    LeafLength.leaf_band_length(context,source,weights,offsets,fresh),
    Equal.trans(Nat,Buffer.length(~F32,fresh),U32.to_nat(local_count(sample,rows)),Nat.mul(channels_n,Nat.mul(rows_n,width_n)),local_length_exact(sample,rows,local_facts),
      Equal.trans(Nat,U32.to_nat(local_count(sample,rows)),Nat.mul(Nat.mul(channels_n,rows_n),width_n),Nat.mul(channels_n,Nat.mul(rows_n,width_n)),
        local_count_exact(sample,rows,local_facts),RowMajor.assoc(channels_n,rows_n,width_n))))
  +balanced_written = {%Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,output),Writes.pure(F32,writes,Buffer.tree(~F32,fresh)),written) : Shape.Balanced(F32,Buffer.depth(~F32,fresh),_)
    Writes.writes_preserve_balance(F32,writes,Buffer.tree(~F32,fresh),Buffer.depth(~F32,fresh),Buffer.balanced(~F32,fresh)) : Shape.Balanced(F32,Buffer.depth(~F32,fresh),Buffer.tree(~F32,output))}
  +depth_equal = Shape.unique_depth(F32,Buffer.tree(~F32,output),Buffer.depth(~F32,output),Buffer.depth(~F32,fresh),Buffer.balanced(~F32,output),balanced_written)
  +depth_ok = {%Equal.sym(Nat,Buffer.depth(~F32,output),Buffer.depth(~F32,fresh),depth_equal) : {Nat.is_le(_,30n) == True{} : Bool}
    local_depth_ok(sample,rows,local_facts) : {Nat.is_le(Buffer.depth(~F32,output),30n) == True{} : Bool}}
  +capacity_equal = Equal.trans(U32,Tree.capacity(F32,Buffer.tree(~F32,output)),Tree.capacity(F32,Writes.pure(F32,writes,Buffer.tree(~F32,fresh))),Tree.capacity(F32,Buffer.tree(~F32,fresh)),
    Equal.cong(Tree.Tree<F32>,U32,Tree.capacity(F32),Buffer.tree(~F32,output),Writes.pure(F32,writes,Buffer.tree(~F32,fresh)),written),
    Writes.writes_preserve_capacity(F32,writes,Buffer.tree(~F32,fresh),Buffer.depth(~F32,fresh),Buffer.balanced(~F32,fresh)))
  +room_output = {%Equal.sym(U32,Tree.capacity(F32,Buffer.tree(~F32,output)),Tree.capacity(F32,Buffer.tree(~F32,fresh)),capacity_equal) : {Nat.is_le(Buffer.length(~F32,output),U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,Buffer.length(~F32,output),Buffer.length(~F32,fresh),LeafLength.leaf_band_length(context,source,weights,offsets,fresh)) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,fresh)))) == True{} : Bool}
    %Equal.sym(Nat,Buffer.length(~F32,fresh),U32.to_nat(local_count(sample,rows)),local_length_exact(sample,rows,local_facts)) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,fresh)))) == True{} : Bool}
    local_count_room(sample,rows,local_facts) : {Nat.is_le(Buffer.length(~F32,output),U32.to_nat(Tree.capacity(F32,Buffer.tree(~F32,output)))) == True{} : Bool}}
  Band{room,length,Buffer.Sized{depth_ok,room_output},
    BandValues.table(~Scope,~scope_cell,~scope_channels,~scope_rows,~scope_width,~local_cell,Scope{sample,first,rows,source,Retained.Held{room,witness},weights,offsets,same_weights,same_offsets})}
