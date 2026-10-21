# The images of the input-gradient kernel: the value every cell of the output
# holds, the bands one image makes, and the bands of the batch, with the value
# of every cell they hold.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_band.bend as Bands
import ../../lib/traversal_array.bend as Traversal
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/input_gradient_tiled_sweep_selected.bend as Selected
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_partition.bend as Parts
import ../../lib/proofs/input_gradient_tiled_placement.bend as Placement
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_tiled_leaf_length.bend as LeafLength
import ../../lib/proofs/input_gradient_tiled_retained.bend as Retained
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_walk.bend as BandWalk
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/nat_interval.bend as Interval
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/traversal_array_refinement.bend as Arrays
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_band_each.bend as Each
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_coordinates.bend as BandCoordinates
import ../../lib/proofs/strided_range_bounds.bend as Strides
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/row_major_coordinates.bend as Grid
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ./input_gradient_tiled_leaves.bend as Leaves
import ./input_gradient_sources.bend as Sources
import ../input_gradient_specification.bend as Specification

# What the value of a cell is read from: the call and the two operands.
type Setup is Data:
  Setup{call: Domain.Call,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>}
def setup_call(setup: Setup) -> Domain.Call:
  Setup{call,gradient,weights} = setup
  call
def setup_gradient(setup: Setup) -> Buffer.Model<F32>:
  Setup{call,gradient,weights} = setup
  gradient
def setup_weights(setup: Setup) -> Buffer.Model<F32>:
  Setup{call,gradient,weights} = setup
  weights
def setup_geometry(+setup: Setup) -> Kernel.Geometry:
  Domain.geometry(setup_call(setup))
def setup_images(+setup: Setup) -> Nat:
  Domain.batch(setup_call(setup))
def size(+setup: Setup) -> Nat:
  Domain.image_size(setup_geometry(setup))

# The specification of one image, and the value of the cell at an index of
# the output: the specification at the point the index decodes to.
def spec(+setup: Setup,+image: Nat) -> Specification.Problem:
  Sources.problem(setup_geometry(setup),Sources.models_reader(setup_gradient(setup),setup_weights(setup)),image)
def value(setup: Setup,index: U32) -> F32:
  Setup{+call,+gradient,+weights} = setup
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  +point = Domain.point(geometry,U32.to_nat(index))
  Specification.value(spec(Setup{Domain.Call{geometry,batch,limits,facts},gradient,weights},Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point))

# The band geometry of the output and the index of a cell of it.
def band_geometry(+geometry: Kernel.Geometry,+images: Nat) -> Bands.Geometry:
  Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
  Bands.Geometry{U32.from_nat(images),c,h,w}
def cell_point(+geometry: Kernel.Geometry,+images: Nat,+image: Nat,+channel: Nat,+row: Nat,+column: Nat) -> Nat:
  BandValues.point(band_geometry(geometry,images),image,channel,row,column)

# The index of a cell is the image offset plus the index inside the image.
def image_split(+image: Nat,+channels: Nat,+height: Nat,+width: Nat,+channel: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column) ==
    Nat.add(Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),RowMajor.index(channel,height,width,row,column)) : Nat}:
  +plane = Nat.mul(height,width)
  +within = Nat.add(Nat.mul(row,width),column)
  %RowMajor.assoc(image,channels,plane) :
    {RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column) == Nat.add(_,RowMajor.index(channel,height,width,row,column)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),Nat.add(Nat.mul(Nat.mul(image,channels),plane),Nat.mul(channel,plane)),
    RowMajor.distrib(Nat.mul(image,channels),channel,plane)) :
    {Nat.add(_,within) == Nat.add(Nat.mul(Nat.mul(image,channels),plane),RowMajor.index(channel,height,width,row,column)) : Nat}
  Addition.associative(Nat.mul(Nat.mul(image,channels),plane),Nat.mul(channel,plane),within)

# The index of a cell is below the batch size, and the batch fits the output.
def cell_bound(+setup: Setup,+capacity: U32,
  +room: {Nat.is_le(Nat.mul(setup_images(setup),size(setup)),U32.to_nat(capacity)) == True{} : Bool},
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +image_inside: {Nat.is_lt(image,setup_images(setup)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Domain.channels(setup_geometry(setup))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,Domain.height(setup_geometry(setup))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(setup_geometry(setup))) == True{} : Bool}) ->
  {Nat.is_lt(cell_point(setup_geometry(setup),setup_images(setup),image,channel,row,column),U32.to_nat(capacity)) == True{} : Bool}:
  Setup{+call,+gradient,+weights} = setup
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +cn = U32.to_nat(c)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +plane = Nat.mul(hn,wn)
  +position = RowMajor.index(Nat.add(Nat.mul(image,cn),channel),hn,wn,row,column)
  Bounds.nat_trans(position,Nat.mul(Nat.mul(batch,cn),plane),U32.to_nat(capacity),
    RowMajor.in_bounds(Nat.add(Nat.mul(image,cn),channel),Nat.mul(batch,cn),hn,wn,row,column,RowMajor.radix(image,batch,channel,cn,image_inside,channel_inside),row_inside,column_inside),
    %Equal.sym(Nat,Nat.mul(Nat.mul(batch,cn),plane),Nat.mul(batch,Nat.mul(cn,plane)),RowMajor.assoc(batch,cn,plane)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room)

# The value of the cell of a point is the specification at the point.
def decoded(+setup: Setup,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(setup_images(setup),size(setup)),U32.to_nat(capacity)) == True{} : Bool},
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +image_inside: {Nat.is_lt(image,setup_images(setup)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Domain.channels(setup_geometry(setup))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,Domain.height(setup_geometry(setup))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(setup_geometry(setup))) == True{} : Bool}) ->
  {value(setup,U32.from_nat(cell_point(setup_geometry(setup),setup_images(setup),image,channel,row,column))) == Specification.value(spec(setup,image),channel,row,column) : F32}:
  Setup{+call,+gradient,+weights} = setup
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +reader = {Setup{whole,gradient,weights} : Setup}
  +bound = cell_bound(reader,capacity,room,image,channel,row,column,image_inside,channel_inside,row_inside,column_inside)
  +cn = U32.to_nat(c)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +plane = Nat.mul(hn,wn)
  +extent = Nat.mul(cn,plane)
  +within = Nat.add(Nat.mul(row,wn),column)
  +local = RowMajor.index(channel,hn,wn,row,column)
  +position = RowMajor.index(Nat.add(Nat.mul(image,cn),channel),hn,wn,row,column)
  +width_positive = Order.le_lt_trans(0n,column,wn,Order.zero_le(column),column_inside)
  +height_positive = Order.le_lt_trans(0n,row,hn,Order.zero_le(row),row_inside)
  +plane_positive = Order.product_positive(hn,wn,height_positive,width_positive)
  +extent_positive = Order.product_positive(cn,plane,Order.le_lt_trans(0n,channel,cn,Order.zero_le(channel),channel_inside),plane_positive)
  +within_inside = RowMajor.radix(row,hn,column,wn,row_inside,column_inside)
  +local_inside = RowMajor.in_bounds(channel,cn,hn,wn,row,column,channel_inside,row_inside,column_inside)
  +parts = Division.unique(position,extent,image,local,image_split(image,cn,hn,wn,channel,row,column),extent_positive,local_inside)
  +decoded_point = {
    %Equal.sym(Nat,Nat.div(position,extent),image,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(position,extent),(image,local),parts)) :
      {Domain.from_coordinates(_,Grid.values(hn,wn,Nat.mod(position,extent))) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(position,extent),local,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(position,extent),(image,local),parts)) :
      {Domain.from_coordinates(image,Grid.values(hn,wn,_)) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.div(local,plane),channel,Radix.quotient(channel,within,plane,plane_positive,within_inside)) :
      {Domain.from_coordinates(image,Grid.Values{_,Nat.div(Nat.mod(local,plane),wn),Nat.mod(Nat.mod(local,plane),wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(local,plane),within,Radix.remainder(channel,within,plane,plane_positive,within_inside)) :
      {Domain.from_coordinates(image,Grid.Values{channel,Nat.div(_,wn),Nat.mod(_,wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.div(within,wn),row,Radix.quotient(row,column,wn,width_positive,column_inside)) :
      {Domain.from_coordinates(image,Grid.Values{channel,_,Nat.mod(within,wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(within,wn),column,Radix.remainder(row,column,wn,width_positive,column_inside)) :
      {Domain.from_coordinates(image,Grid.Values{channel,row,_}) == Domain.Point{image,channel,row,column} : Domain.Point}
    {==} : {Domain.point(shape,position) == Domain.Point{image,channel,row,column} : Domain.Point}}
  Equal.cong(Domain.Point,F32,point => Specification.value(spec(reader,Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point)),
    Domain.point(shape,U32.to_nat(U32.from_nat(position))),Domain.Point{image,channel,row,column},
    Equal.trans(Domain.Point,Domain.point(shape,U32.to_nat(U32.from_nat(position))),Domain.point(shape,position),Domain.Point{image,channel,row,column},
      Equal.cong(Nat,Domain.Point,index => Domain.point(shape,index),U32.to_nat(U32.from_nat(position)),position,
        FullRange.roundtrip(position,capacity,Bounds.strict_le(position,U32.to_nat(capacity),bound))),
      decoded_point))

# What a band of an image is checked under: its image, the split depth and
# the output capacity.
type Scope is Data:
  Scope{sample: Leaves.Sample,depth: Nat,capacity: U32,allocatable: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(Domain.batch(Leaves.image_call(sample)),Domain.image_size(Leaves.geometry(sample))),U32.to_nat(capacity)) == True{} : Bool},
    limits: Machine.Limits,
    machine_room: {Nat.is_le(Nat.mul(Domain.batch(Leaves.image_call(sample)),Domain.image_size(Leaves.geometry(sample))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
def scope_sample(+scope: Scope) -> Leaves.Sample:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  sample
def scope_start(scope: Scope) -> U32:
  Leaves.start(scope_sample(scope))
def scope_geometry(+scope: Scope) -> Bands.Geometry:
  Leaves.band_geometry(scope_sample(scope))
def scope_setup(scope: Scope) -> Setup:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Leaves.Sample{call,envelope,fits,gradient,weights,index,inside} = sample
  Setup{call,gradient,weights}
def capacity_ok(+capacity: U32,+allocatable: {U32.is_le(capacity,2147483648) == True{} : Bool}) -> {U32.is_le(capacity,2147483648) == True{} : Bool}:
  allocatable

# Word arithmetic of a band of an image: its first row over the batch, and
# the rows of the batch and of an image inside the output.
def shifted(+index: U32,+height: U32,+first: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(index),U32.to_nat(height)),U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((index * height + first : U32)) == Nat.add(Nat.mul(U32.to_nat(index),U32.to_nat(height)),U32.to_nat(first)) : Nat}:
  +product = Nat.mul(U32.to_nat(index),U32.to_nat(height))
  +offset = U32.to_nat(first)
  +start_exact = Arithmetic.multiply(index,height,capacity,capacity_ok,Order.transitive(product,Nat.add(product,offset),U32.to_nat(capacity),Order.le_add(product,offset),bound))
  Equal.trans(Nat,U32.to_nat((index * height + first : U32)),Nat.add(U32.to_nat((index * height : U32)),offset),Nat.add(product,offset),
    Arithmetic.add((index * height : U32),first,capacity,capacity_ok,%Equal.sym(Nat,U32.to_nat((index * height : U32)),product,start_exact) : {Nat.is_le(Nat.add(_,offset),U32.to_nat(capacity)) == True{} : Bool} bound),
    Equal.cong(Nat,Nat,start => Nat.add(start,offset),U32.to_nat((index * height : U32)),product,start_exact))
def batch_rows_room(+batch: Nat,+c: U32,+h: U32,+w: U32,+capacity: U32,
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},+width_positive: {Nat.is_lt(0n,U32.to_nat(w)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(batch,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(batch,U32.to_nat(h)),U32.to_nat(capacity)) == True{} : Bool}:
  +cn = U32.to_nat(c)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +plane = Nat.mul(hn,wn)
  +frame = Nat.mul(cn,plane)
  +height_frame = Order.transitive(hn,plane,frame,RowMajor.product_covers(hn,wn,width_positive),DivisionOrder.le_multiple(cn,plane,channels_positive))
  Order.transitive(Nat.mul(batch,hn),Nat.mul(batch,frame),U32.to_nat(capacity),DivisionOrder.multiply_left_le(batch,hn,frame,height_frame),room)
def image_rows_room(+batch: Nat,+h: U32,+index: U32,+capacity: U32,+image_inside: {Nat.is_lt(U32.to_nat(index),batch) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(batch,U32.to_nat(h)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+U32.to_nat(index),U32.to_nat(h)),U32.to_nat(capacity)) == True{} : Bool}:
  Order.transitive(Nat.mul(1n+U32.to_nat(index),U32.to_nat(h)),Nat.mul(batch,U32.to_nat(h)),U32.to_nat(capacity),
    RowMajor.mul_le(1n+U32.to_nat(index),batch,U32.to_nat(h),Order.strict_follow(U32.to_nat(index),batch,image_inside)),rows_room)
def band_fits(+image: Nat,+height: Nat,+first: Nat,+rows: Nat,+room: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(Nat.mul(image,height),first),rows),Nat.mul(1n+image,height)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(height,Nat.mul(image,height)),Nat.add(Nat.mul(image,height),height),Order.commutative(height,Nat.mul(image,height))) :
    {Nat.is_le(Nat.add(Nat.add(Nat.mul(image,height),first),rows),_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,height),first),rows),Nat.add(Nat.mul(image,height),Nat.add(first,rows)),Addition.associative(Nat.mul(image,height),first,rows)) :
    {Nat.is_le(_,Nat.add(Nat.mul(image,height),height)) == True{} : Bool}
  Order.offset_le(Nat.mul(image,height),Nat.add(first,rows),height,room)
def band_first_room(+image: Nat,+height: Nat,+first: Nat,+rows: Nat,+capacity: Nat,+room: {Nat.is_le(Nat.add(first,rows),height) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(1n+image,height),capacity) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.mul(image,height),first),capacity) == True{} : Bool}:
  Order.transitive(Nat.add(Nat.mul(image,height),first),Nat.add(Nat.add(Nat.mul(image,height),first),rows),capacity,
    Order.le_add(Nat.add(Nat.mul(image,height),first),rows),Order.transitive(Nat.add(Nat.add(Nat.mul(image,height),first),rows),Nat.mul(1n+image,height),capacity,band_fits(image,height,first,rows,room),rows_room))
def remainder_le(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {Nat.is_le(Nat.mod(value,divisor),value) == True{} : Bool}:
  +quotient = Nat.mul(Nat.div(value,divisor),divisor)
  %Equal.sym(Nat,value,Nat.add(quotient,Nat.mod(value,divisor)),DivisionOrder.quotient_exact(value,divisor,positive)) : {Nat.is_le(Nat.mod(value,divisor),_) == True{} : Bool}
  %Order.commutative(Nat.mod(value,divisor),quotient) : {Nat.is_le(Nat.mod(value,divisor),_) == True{} : Bool}
  Order.le_add(Nat.mod(value,divisor),quotient)

# The pixel of a band row over the batch is the point of the image and the
# row inside it.
def band_pixel(+batch: Nat,+c: U32,+h: U32,+w: U32,+index: U32,+first: U32,+row: Nat,+channel: Nat,+column: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(h)) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(U32.to_nat(first),row),U32.to_nat(h)) == True{} : Bool},
  +bound: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(index),U32.to_nat(h)),U32.to_nat(first)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {BandValues.pixel(Bands.Geometry{U32.from_nat(batch),c,h,w},channel,Nat.add(U32.to_nat((index * h + first : U32)),row),column) ==
    BandValues.point(Bands.Geometry{U32.from_nat(batch),c,h,w},U32.to_nat(index),channel,Nat.add(U32.to_nat(first),row),column) : Nat}:
  +i = U32.to_nat(index)
  +fn = U32.to_nat(first)
  +hn = U32.to_nat(h)
  +geometry = {Bands.Geometry{U32.from_nat(batch),c,h,w} : Bands.Geometry}
  %Equal.sym(Nat,U32.to_nat((index * h + first : U32)),Nat.add(Nat.mul(i,hn),fn),shifted(index,h,first,capacity,capacity_ok,bound)) :
    {BandValues.pixel(geometry,channel,Nat.add(_,row),column) == BandValues.point(geometry,i,channel,Nat.add(fn,row),column) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(i,hn),fn),row),Nat.add(Nat.mul(i,hn),Nat.add(fn,row)),Addition.associative(Nat.mul(i,hn),fn,row)) :
    {BandValues.pixel(geometry,channel,_,column) == BandValues.point(geometry,i,channel,Nat.add(fn,row),column) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(i,hn),Nat.add(fn,row)),hn),i,Radix.quotient(i,Nat.add(fn,row),hn,height_positive,inside)) :
    {BandValues.point(geometry,_,channel,Nat.mod(Nat.add(Nat.mul(i,hn),Nat.add(fn,row)),hn),column) == BandValues.point(geometry,i,channel,Nat.add(fn,row),column) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(i,hn),Nat.add(fn,row)),hn),Nat.add(fn,row),Radix.remainder(i,Nat.add(fn,row),hn,height_positive,inside)) :
    {BandValues.point(geometry,i,channel,_,column) == BandValues.point(geometry,i,channel,Nat.add(fn,row),column) : Nat}
  {==}

# The facts about every band of an image, read from the leaf facts of its
# pieces: the metadata, the readiness, the values and the rows of every band.
def fields_after(+scope: Scope) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Leaves.Sample{call,envelope,fits,gradient,weights,index,inside} = sample
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  BandShape.band_fields(~F32,Bands.Geometry{U32.from_nat(batch),c,h,w})
def ready_after(+scope: Scope) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  Model.ready_band(~F32)
def values_after(+scope: Scope) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Leaves.Sample{call,envelope,fits,gradient,weights,index,inside} = sample
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  BandValues.band_values(~F32,~Setup,~value,Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts},gradient,weights},Bands.Geometry{U32.from_nat(batch),c,h,w})
def band_rows(+geometry: Bands.Geometry,+image: Nat,+first: U32,+rows: U32,+band: Buffer.Content<F32>) -> Data:
  Shape.Both<{Nat.is_le(Nat.mul(image,BandValues.height(geometry)),U32.to_nat(first)) == True{} : Bool},
    {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(1n+image,BandValues.height(geometry))) == True{} : Bool}>
def rows_after(+scope: Scope) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Leaves.Sample{call,envelope,fits,gradient,weights,index,inside} = sample
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  band_rows(Bands.Geometry{U32.from_nat(batch),c,h,w},U32.to_nat(index))

def fields_rule(+scope: Scope,+part: Execution.Part,+first: U32,+rows: U32,
  fact: Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample,scope,part,first,rows)) ->
  fields_after(scope,(scope_start(scope) + first : U32),rows,Buffer.content(~F32,Selected.part_output(part))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Execution.Part{+gradient_tree,+output} = part
  Buffer.Model{+tree,+length,+certificate} = output
  Leaves.Band{+band_room,+length_exact,sized,cells} = fact
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts} : Domain.Call}
  +i = U32.to_nat(index)
  +fn = U32.to_nat(first)
  +rn = U32.to_nat(rows)
  +cn = U32.to_nat(c)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +extent = Nat.mul(cn,Nat.mul(hn,wn))
  +capacity_fits = capacity_ok(capacity,allocatable)
  +height_positive = Domain.height_positive(whole)
  +rows_room = batch_rows_room(batch,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room)
  +image_room = image_rows_room(batch,h,index,capacity,inside,rows_room)
  +first_word = (index * h + first : U32)
  +first_exact = shifted(index,h,first,capacity,capacity_fits,band_first_room(i,hn,fn,rn,U32.to_nat(capacity),band_room,image_room))
  +height_bound = Limits.double_bound(limits,hn,
    Order.transitive(hn,Nat.mul(1n+i,hn),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(1n+i,hn,{==}),
      image_rows_room(batch,h,index,Machine.capacity(limits),inside,batch_rows_room(batch,c,h,w,Machine.capacity(limits),Domain.channels_positive(whole),Domain.width_positive(whole),machine_room))))
  +span = {
    %Equal.sym(Nat,U32.to_nat(first_word),Nat.add(Nat.mul(i,hn),fn),first_exact) : {Nat.is_le(Nat.add(Nat.mod(_,hn),rn),hn) == True{} : Bool}
    %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(i,hn),fn),hn),Nat.mod(fn,hn),DivisionOrder.remainder_shift(fn,hn,i,height_positive)) : {Nat.is_le(Nat.add(_,rn),hn) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mod(fn,hn),rn),Nat.add(fn,rn),hn,Order.add_le(Nat.mod(fn,hn),fn,rn,remainder_le(fn,hn,height_positive)),band_room)
    : {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(first_word),hn),rn),hn) == True{} : Bool}}
  +batch_exact = FullRange.roundtrip(batch,capacity,Order.transitive(batch,Nat.mul(batch,extent),U32.to_nat(capacity),RowMajor.product_covers(batch,extent,Domain.image_positive(whole)),room))
  +rows_fit = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(batch)),batch,batch_exact) : {Nat.is_le(Nat.add(U32.to_nat(first_word),rn),Nat.mul(_,hn)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(first_word),Nat.add(Nat.mul(i,hn),fn),first_exact) : {Nat.is_le(Nat.add(_,rn),Nat.mul(batch,hn)) == True{} : Bool}
    Order.transitive(Nat.add(Nat.add(Nat.mul(i,hn),fn),rn),Nat.mul(1n+i,hn),Nat.mul(batch,hn),band_fits(i,hn,fn,rn,band_room),RowMajor.mul_le(1n+i,batch,hn,Order.strict_follow(i,batch,inside)))
    : {Nat.is_le(Nat.add(U32.to_nat(first_word),rn),Nat.mul(U32.to_nat(U32.from_nat(batch)),hn)) == True{} : Bool}}
  BandShape.Fields{Inspection.local_guard(h,first_word,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,height_bound,span),rows_fit,length_exact}

def ready_rule(+scope: Scope,+part: Execution.Part,+first: U32,+rows: U32,
  fact: Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample,scope,part,first,rows)) ->
  ready_after(scope,(scope_start(scope) + first : U32),rows,Buffer.content(~F32,Selected.part_output(part))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Band{band_room,length_exact,sized,cells} = fact
  Buffer.content_ready(~F32,Selected.part_output(part),sized)

def rows_rule(+scope: Scope,+part: Execution.Part,+first: U32,+rows: U32,
  fact: Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample,scope,part,first,rows)) ->
  rows_after(scope,(scope_start(scope) + first : U32),rows,Buffer.content(~F32,Selected.part_output(part))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Leaves.Band{+band_room,length_exact,sized,cells} = fact
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts} : Domain.Call}
  +i = U32.to_nat(index)
  +fn = U32.to_nat(first)
  +rn = U32.to_nat(rows)
  +hn = U32.to_nat(h)
  +image_room = image_rows_room(batch,h,index,capacity,inside,batch_rows_room(batch,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room))
  +first_exact = shifted(index,h,first,capacity,capacity_ok(capacity,allocatable),band_first_room(i,hn,fn,rn,U32.to_nat(capacity),band_room,image_room))
  %Equal.sym(Nat,U32.to_nat((index * h + first : U32)),Nat.add(Nat.mul(i,hn),fn),first_exact) :
    Shape.Both<{Nat.is_le(Nat.mul(i,hn),_) == True{} : Bool},{Nat.is_le(Nat.add(_,rn),Nat.mul(1n+i,hn)) == True{} : Bool}>
  Shape.Both{Order.le_add(Nat.mul(i,hn),fn),band_fits(i,hn,fn,rn,band_room)}

# The cells of one piece, read from its leaf facts.
type CellTable is Data:
  CellTable{image: Leaves.Sample,part: Execution.Part,first: U32,rows: U32}
def cell_of(+table: CellTable,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellTable{image,part,first,rows} = table
  Leaves.Cell(image,part,first,rows,channel,row,column)
def table_channels(+table: CellTable) -> Nat:
  CellTable{image,part,first,rows} = table
  Domain.channels(Leaves.geometry(image))
def table_rows(+table: CellTable) -> Nat:
  CellTable{image,part,first,rows} = table
  U32.to_nat(rows)
def table_width(+table: CellTable) -> Nat:
  CellTable{image,part,first,rows} = table
  Domain.width(Leaves.geometry(image))
type CellScope is Data:
  CellScope{scope: Scope,part: Execution.Part,first: U32,rows: U32,band: Leaves.Band<scope_sample(scope),part,first,rows>}
def cell_at(+cell: CellScope,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{scope,part,first,rows,band} = cell
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Leaves.Sample{call,envelope,fits,gradient,weights,index,inside} = sample
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Execution.Part{gradient_tree,output} = part
  Buffer.Model{tree,length,certificate} = output
  BandValues.At(~F32,~Setup,~value,Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts},gradient,weights},Bands.Geometry{U32.from_nat(batch),c,h,w},
    (index * h + first : U32),rows,tree,length,channel,row,column)
def cell_channels(+cell: CellScope) -> Nat:
  CellScope{scope,part,first,rows,band} = cell
  Domain.channels(Leaves.geometry(scope_sample(scope)))
def cell_rows(+cell: CellScope) -> Nat:
  CellScope{scope,part,first,rows,band} = cell
  U32.to_nat(rows)
def cell_width(+cell: CellScope) -> Nat:
  CellScope{scope,part,first,rows,band} = cell
  Domain.width(Leaves.geometry(scope_sample(scope)))
def cell_rule(+cell: CellScope,+channel: Nat,+row: Nat,+column: Nat,+channel_inside: {Nat.is_lt(channel,cell_channels(cell)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,cell_rows(cell)) == True{} : Bool},+column_inside: {Nat.is_lt(column,cell_width(cell)) == True{} : Bool}) ->
  cell_at(cell,channel,row,column):
  CellScope{+scope,+part,+first,+rows,band} = cell
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Execution.Part{+gradient_tree,+output} = part
  Buffer.Model{+tree,+length,+certificate} = output
  Leaves.Band{+band_room,+length_exact,sized,cells} = band
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits_call,facts} : Domain.Call}
  +image = {Leaves.Sample{whole,envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +reader = {Setup{whole,gradient,weights_model} : Setup}
  +geom = {Bands.Geometry{U32.from_nat(batch),c,h,w} : Bands.Geometry}
  +piece = {Execution.Part{gradient_tree,Buffer.Model{tree,length,certificate}} : Execution.Part}
  +i = U32.to_nat(index)
  +fn = U32.to_nat(first)
  +rn = U32.to_nat(rows)
  +hn = U32.to_nat(h)
  +first_word = (index * h + first : U32)
  +capacity_fits = capacity_ok(capacity,allocatable)
  +image_room = image_rows_room(batch,h,index,capacity,inside,batch_rows_room(batch,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room))
  +row_room = Bounds.nat_trans(Nat.add(fn,row),Nat.add(fn,rn),hn,Order.offset_lt(fn,row,rn,row_inside),band_room)
  +found = BandValues.table_cell(~CellTable,~cell_of,~table_channels,~table_rows,~table_width,CellTable{image,piece,first,rows},channel,row,column,cells,channel_inside,row_inside,column_inside)
  Shape.Both{BandValues.position_within(geom,rows,length,channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    Equal.trans(F32,Tree.read(F32,tree,U32.from_nat(BandValues.position(geom,rows,channel,row,column))),Specification.value(spec(reader,i),channel,Nat.add(fn,row),column),
      value(reader,U32.from_nat(BandValues.pixel(geom,channel,Nat.add(U32.to_nat(first_word),row),column))),
      found,
      Equal.sym(F32,value(reader,U32.from_nat(BandValues.pixel(geom,channel,Nat.add(U32.to_nat(first_word),row),column))),Specification.value(spec(reader,i),channel,Nat.add(fn,row),column),
        Equal.trans(F32,value(reader,U32.from_nat(BandValues.pixel(geom,channel,Nat.add(U32.to_nat(first_word),row),column))),
          value(reader,U32.from_nat(BandValues.point(geom,i,channel,Nat.add(fn,row),column))),Specification.value(spec(reader,i),channel,Nat.add(fn,row),column),
          Equal.cong(Nat,F32,point => value(reader,U32.from_nat(point)),BandValues.pixel(geom,channel,Nat.add(U32.to_nat(first_word),row),column),BandValues.point(geom,i,channel,Nat.add(fn,row),column),
            band_pixel(batch,c,h,w,index,first,row,channel,column,capacity,capacity_fits,Domain.height_positive(whole),row_room,
              band_first_room(i,hn,fn,rn,U32.to_nat(capacity),band_room,image_room))),
          decoded(reader,capacity,capacity_fits,room,i,channel,Nat.add(fn,row),column,inside,channel_inside,row_room,column_inside))))}
def values_rule(+scope: Scope,+part: Execution.Part,+first: U32,+rows: U32,
  fact: Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample,scope,part,first,rows)) ->
  values_after(scope,(scope_start(scope) + first : U32),rows,Buffer.content(~F32,Selected.part_output(part))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Execution.Part{+gradient_tree,+output} = part
  Buffer.Model{+tree,+length,+certificate} = output
  +image = {Leaves.Sample{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +piece = {Execution.Part{gradient_tree,Buffer.Model{tree,length,certificate}} : Execution.Part}
  BandValues.table(~CellScope,~cell_at,~cell_channels,~cell_rows,~cell_width,~cell_rule,CellScope{Scope{image,depth,capacity,allocatable,room,limits,machine_room},piece,first,rows,fact})

# What every split keeps beside the source: the packed trees and the task.
def carried(+scope: Leaves.Sample,+context: Kernel.Task,+state: Parts.Owners) -> Data:
  Shape.Both<Shape.Both<{Tree.pack(Tile.Tile8,Parts.owner_weights(state)) == Leaves.native_weights(Leaves.packing(scope)) : Array<Tile.Tile8>},
      {Tree.pack(U32,Parts.owner_offsets(state)) == Leaves.native_offsets(Leaves.packing(scope)) : Array<U32>}>,
    {context == Retained.task(Leaves.setting(scope),Kernel.tiling(Leaves.geometry(scope))) : Kernel.Task}>
def carried_kept(+scope: Leaves.Sample,+context: Kernel.Task,+left: Bands.Plan,+right: Bands.Plan,+state: Parts.Owners,+fact: carried(scope,context,state)) ->
  Shape.Both<carried(scope,context,PartitionLaws.first(~Parts.Owners,Parts.split(context,left,right,state))),carried(scope,context,PartitionLaws.second(~Parts.Owners,Parts.split(context,left,right,state)))>:
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = scope
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Parts.Owners{+source,+weights,+offsets} = state
  Shape.Both{+same,task_same} = fact
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +image = {Leaves.Sample{Domain.Call{shape,batch,limits,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +task = {Kernel.Task{shape,Kernel.tiling(shape)} : Kernel.Task}
  %Equal.sym(Kernel.Task,context,task,task_same) :
    Shape.Both<carried(image,_,PartitionLaws.first(~Parts.Owners,Parts.split(_,left,right,Parts.Owners{source,weights,offsets}))),
      carried(image,_,PartitionLaws.second(~Parts.Owners,Parts.split(_,left,right,Parts.Owners{source,weights,offsets})))>
  Shape.Both{Shape.Both{same,{==}},Shape.Both{same,{==}}}
def leaf_rows(+context: Kernel.Task,+first: U32,+rows: U32,+state: Parts.Owners) ->
  {Model.shaped_rows(~Execution.Part,Model.shaped_parts(~Execution.Part,Parts.model_leaf(context,first,rows,state))) == Bands.Rows{first,rows} : Bands.Plan}:
  Kernel.Task{+geometry,+layout} = context
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Parts.Owners{+source,+weights,+offsets} = state
  {==}

# The leaf facts of one piece of the walk, from what the walk keeps at it.
def derive(+scope: Leaves.Sample,+context: Kernel.Task,+first: U32,+rows: U32,+state: Parts.Owners,
  fact: BandWalk.StateBoth(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~Retained.scoped_held(~Leaves.Sample,~Leaves.setting),~BandWalk.Kept(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~carried),scope,context,first,rows,state)) ->
  Model.ShapedAll(~Execution.Part,~Leaves.Sample,~Leaves.band,scope,Model.shaped_parts(~Execution.Part,Parts.model_leaf(context,first,rows,state))):
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = scope
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Parts.Owners{+source,+weights,+offsets} = state
  Shape.Both{held,carried_facts} = fact
  Shape.Both{same,task_same} = carried_facts
  Shape.Both{same_weights,same_offsets} = same
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +image = {Leaves.Sample{Domain.Call{shape,batch,limits,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +task = {Kernel.Task{shape,Kernel.tiling(shape)} : Kernel.Task}
  %Equal.sym(Kernel.Task,context,task,task_same) :
    Model.ShapedAll(~Execution.Part,~Leaves.Sample,~Leaves.band,image,Model.shaped_parts(~Execution.Part,Parts.model_leaf(_,first,rows,Parts.Owners{source,weights,offsets})))
  Leaves.band_of(image,first,rows,source,held,weights,offsets,same_weights,same_offsets)

# The model of an image: its pieces over the row plan of the depth, placed
# at its rows of the output.
def image_plan(+scope: Scope) -> Bands.Plan:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  +height = Domain.height_word(Leaves.geometry(sample))
  Bands.row_plan(depth,U32.is_lt(1,height),0,height)
def tree_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>) -> Partition.Tree<&2,Execution.Part,Unit>:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  Parts.model(image_plan(scope),Retained.task(Leaves.setting(sample),Kernel.tiling(Leaves.geometry(sample))),Parts.Owners{source,weights,offsets})
def image_coverage(+scope: Scope) -> BandPlan.Coverage(image_plan(scope),0n,Domain.height(Leaves.geometry(scope_sample(scope)))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts} : Domain.Call}
  +i = U32.to_nat(index)
  +hn = U32.to_nat(h)
  +image_room = image_rows_room(batch,h,index,capacity,inside,batch_rows_room(batch,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room))
  BandPlan.refinement_row_plan(depth,U32.is_lt(1,h),0,h,capacity,capacity_ok(capacity,allocatable),
    Order.transitive(hn,Nat.mul(1n+i,hn),U32.to_nat(capacity),Order.le_add(hn,Nat.mul(i,hn)),image_room))
def model_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>) -> Model.Model<F32>:
  Model.shaped_model(~Execution.Part,~F32,~Selected.part_output,Model.shaped_shift(~Execution.Part,scope_start(scope),Model.shaped_parts(~Execution.Part,tree_of(scope,source,weights,offsets))))
def piece_facts(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +held: Retained.Held<Leaves.setting(scope_sample(scope)),0,Domain.height_word(Leaves.geometry(scope_sample(scope))),source>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(Leaves.packing(scope_sample(scope))) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == Leaves.native_offsets(Leaves.packing(scope_sample(scope))) : Array<U32>}) ->
  Model.ShapedAll(~Execution.Part,~Scope,~Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample),scope,Model.shaped_parts(~Execution.Part,tree_of(scope,source,weights,offsets))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +image = {Leaves.Sample{Domain.Call{shape,batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +frame = {Scope{image,depth,capacity,allocatable,room,limits,machine_room} : Scope}
  +plan = image_plan(frame)
  +task = {Kernel.Task{shape,Kernel.tiling(shape)} : Kernel.Task}
  +owners = {Parts.Owners{source,weights,offsets} : Parts.Owners}
  Model.shaped_all_view(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample,Scope{image,depth,capacity,allocatable,room,limits,machine_room},Model.shaped_parts(~Execution.Part,Parts.model(plan,task,owners)),
    BandWalk.model_all(~Parts.Owners,~Execution.Part,~Kernel.Task,~Leaves.Sample,~Parts.split,~Parts.model_leaf,
      ~BandWalk.StateBoth(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~Retained.scoped_held(~Leaves.Sample,~Leaves.setting),~BandWalk.Kept(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~carried)),
      ~Leaves.band,~derive,image,plan,task,owners,
      BandWalk.state_both(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~Parts.split,~Retained.scoped_held(~Leaves.Sample,~Leaves.setting),~BandWalk.Kept(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~carried),
        image,plan,task,owners,
        Retained.held_leaves(~Leaves.Sample,~Leaves.setting,plan,image,Kernel.tiling(shape),0,h,source,weights,offsets,image_coverage(frame),held),
        BandWalk.split_kept(~Parts.Owners,~Kernel.Task,~Leaves.Sample,~Parts.split,~carried,~carried_kept,image,plan,task,owners,Shape.Both{Shape.Both{same_weights,same_offsets},{==}}))))

# The four facts about every band of the model of an image.
def metadata_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +held: Retained.Held<Leaves.setting(scope_sample(scope)),0,Domain.height_word(Leaves.geometry(scope_sample(scope))),source>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(Leaves.packing(scope_sample(scope))) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == Leaves.native_offsets(Leaves.packing(scope_sample(scope))) : Array<U32>}) ->
  BandShape.Contents(~F32,scope_geometry(scope),Model.content(~F32,model_of(scope,source,weights,offsets))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +image = {Leaves.Sample{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +frame = {Scope{image,depth,capacity,allocatable,room,limits,machine_room} : Scope}
  Model.shaped_content_all(~Execution.Part,~F32,~Selected.part_output,~Scope,~scope_start,~Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample),~fields_after,~fields_rule,
    frame,Model.shaped_parts(~Execution.Part,tree_of(frame,source,weights,offsets)),piece_facts(frame,source,weights,offsets,held,same_weights,same_offsets))
def ready_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +held: Retained.Held<Leaves.setting(scope_sample(scope)),0,Domain.height_word(Leaves.geometry(scope_sample(scope))),source>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(Leaves.packing(scope_sample(scope))) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == Leaves.native_offsets(Leaves.packing(scope_sample(scope))) : Array<U32>}) ->
  Model.ContentReady(~F32,Model.content(~F32,model_of(scope,source,weights,offsets))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +image = {Leaves.Sample{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +frame = {Scope{image,depth,capacity,allocatable,room,limits,machine_room} : Scope}
  Model.shaped_content_all(~Execution.Part,~F32,~Selected.part_output,~Scope,~scope_start,~Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample),~ready_after,~ready_rule,
    frame,Model.shaped_parts(~Execution.Part,tree_of(frame,source,weights,offsets)),piece_facts(frame,source,weights,offsets,held,same_weights,same_offsets))
def values_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +held: Retained.Held<Leaves.setting(scope_sample(scope)),0,Domain.height_word(Leaves.geometry(scope_sample(scope))),source>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(Leaves.packing(scope_sample(scope))) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == Leaves.native_offsets(Leaves.packing(scope_sample(scope))) : Array<U32>}) ->
  BandValues.Values(~F32,~Setup,~value,scope_setup(scope),scope_geometry(scope),Model.content(~F32,model_of(scope,source,weights,offsets))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +image = {Leaves.Sample{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +frame = {Scope{image,depth,capacity,allocatable,room,limits,machine_room} : Scope}
  Model.shaped_content_all(~Execution.Part,~F32,~Selected.part_output,~Scope,~scope_start,~Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample),~values_after,~values_rule,
    frame,Model.shaped_parts(~Execution.Part,tree_of(frame,source,weights,offsets)),piece_facts(frame,source,weights,offsets,held,same_weights,same_offsets))
def rows_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,
  +held: Retained.Held<Leaves.setting(scope_sample(scope)),0,Domain.height_word(Leaves.geometry(scope_sample(scope))),source>,
  +same_weights: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(Leaves.packing(scope_sample(scope))) : Array<Tile.Tile8>},
  +same_offsets: {Tree.pack(U32,offsets) == Leaves.native_offsets(Leaves.packing(scope_sample(scope))) : Array<U32>}) ->
  Model.ContentAll(~F32,~band_rows(scope_geometry(scope),U32.to_nat(Leaves.image_index(scope_sample(scope)))),Model.content(~F32,model_of(scope,source,weights,offsets))):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +image = {Leaves.Sample{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits_call,facts},envelope,fits,gradient,weights_model,index,inside} : Leaves.Sample}
  +frame = {Scope{image,depth,capacity,allocatable,room,limits,machine_room} : Scope}
  Model.shaped_content_all(~Execution.Part,~F32,~Selected.part_output,~Scope,~scope_start,~Model.viewed(~Execution.Part,~Leaves.Sample,~Leaves.band,~Scope,~scope_sample),~rows_after,~rows_rule,
    frame,Model.shaped_parts(~Execution.Part,tree_of(frame,source,weights,offsets)),piece_facts(frame,source,weights,offsets,held,same_weights,same_offsets))

# The accepted band model of an image: its layout covers the rows of
# the image, so every band is inside its image and the inspection accepts it.
def scope_limits(scope: Scope) -> Machine.Limits:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  limits
def scope_capacity(scope: Scope) -> U32:
  Scope{sample,depth,capacity,allocatable,room,limits,machine_room} = scope
  capacity
def scope_image(+scope: Scope) -> Nat:
  U32.to_nat(Leaves.image_index(scope_sample(scope)))
def scope_height(+scope: Scope) -> Nat:
  Domain.height(Leaves.geometry(scope_sample(scope)))
def layout_of(+scope: Scope,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>) ->
  BandPlan.Coverage(Model.layout(~F32,model_of(scope,source,weights,offsets)),Nat.mul(scope_image(scope),scope_height(scope)),scope_height(scope)):
  Scope{+sample,+depth,+capacity,+allocatable,+room,+limits,+machine_room} = scope
  Leaves.Sample{+call,+envelope,+fits,+gradient,+weights_model,+index,+inside} = sample
  Domain.Call{+geometry,+batch,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits_call,facts} : Domain.Call}
  +frame = {Scope{Leaves.Sample{whole,envelope,fits,gradient,weights_model,index,inside},depth,capacity,allocatable,room,limits,machine_room} : Scope}
  +plan = image_plan(frame)
  +task = {Kernel.Task{shape,Kernel.tiling(shape)} : Kernel.Task}
  +owners = {Parts.Owners{source,weights,offsets} : Parts.Owners}
  +parts = Model.shaped_parts(~Execution.Part,Parts.model(plan,task,owners))
  +start = (index * h : U32)
  +i = U32.to_nat(index)
  +hn = U32.to_nat(h)
  +capacity_fits = capacity_ok(capacity,allocatable)
  +image_room = image_rows_room(batch,h,index,capacity,inside,batch_rows_room(batch,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room))
  +start_exact = Arithmetic.multiply(index,h,capacity,capacity_fits,
    Order.transitive(Nat.mul(i,hn),Nat.mul(1n+i,hn),U32.to_nat(capacity),%Order.commutative(Nat.mul(i,hn),hn) : {Nat.is_le(Nat.mul(i,hn),_) == True{} : Bool} Order.le_add(Nat.mul(i,hn),hn),image_room))
  +covered = {
    %Equal.sym(Bands.Plan,Model.shaped_rows(~Execution.Part,parts),plan,BandWalk.model_rows(~Parts.Owners,~Execution.Part,~Kernel.Task,~Parts.split,~Parts.model_leaf,~leaf_rows,plan,task,owners)) :
      BandPlan.Coverage(_,0n,hn)
    image_coverage(frame) : BandPlan.Coverage(Model.shaped_rows(~Execution.Part,parts),0n,hn)}
  +moved = Model.shaped_shift_coverage(~Execution.Part,start,parts,0n,hn,capacity,capacity_fits,
    %Equal.sym(Nat,U32.to_nat(start),Nat.mul(i,hn),start_exact) : {Nat.is_le(Nat.add(_,hn),U32.to_nat(capacity)) == True{} : Bool}
    %Order.commutative(hn,Nat.mul(i,hn)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    image_room,
    covered)
  %Equal.sym(Bands.Plan,Model.layout(~F32,Model.shaped_model(~Execution.Part,~F32,~Selected.part_output,Model.shaped_shift(~Execution.Part,start,parts))),
    Model.shaped_rows(~Execution.Part,Model.shaped_shift(~Execution.Part,start,parts)),Model.shaped_model_layout(~Execution.Part,~F32,~Selected.part_output,Model.shaped_shift(~Execution.Part,start,parts))) :
    BandPlan.Coverage(_,Nat.mul(i,hn),hn)
  %Equal.trans(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),Nat.mul(i,hn),Addition.zero_right(U32.to_nat(start)),start_exact) :
    BandPlan.Coverage(Model.shaped_rows(~Execution.Part,Model.shaped_shift(~Execution.Part,start,parts)),_,hn)
  moved
# ---- One image ----

# The model of the state the images thread: the gradient and the packed trees.
# The runtime state is its pack, and what is seen of a runtime state is the
# model it reads back as.
type State is Data:
  State{gradient: Tree.Tree<F32>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>}
def represent(state: State) -> Kernel.ImageState:
  State{gradient,weights,offsets} = state
  Kernel.ImageState{Tree.pack(F32,gradient),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets)}
def held_parts(-gradient: Array<F32>,-weights: Array<Tile.Tile8>,-offsets: Array<U32>,
  gradient_observation: StorageLaws.Observation<F32,gradient>,weights_observation: StorageLaws.Observation<Tile.Tile8,weights>,
  offsets_observation: StorageLaws.Observation<U32,offsets>) -> Each.Held<Kernel.ImageState,State,represent,Kernel.ImageState{gradient,weights,offsets}>:
  StorageLaws.Observation{+gradient_tree,+gradient_packed,gradient_reflected} = gradient_observation
  StorageLaws.Observation{+weights_tree,+weights_packed,weights_reflected} = weights_observation
  StorageLaws.Observation{+offsets_tree,+offsets_packed,offsets_reflected} = offsets_observation
  Each.Held{State{gradient_tree,weights_tree,offsets_tree},
    %gradient_packed : {Kernel.ImageState{Tree.pack(F32,gradient_tree),Tree.pack(Tile.Tile8,weights_tree),Tree.pack(U32,offsets_tree)} ==
      Kernel.ImageState{_,weights,offsets} : Kernel.ImageState}
    %weights_packed : {Kernel.ImageState{Tree.pack(F32,gradient_tree),Tree.pack(Tile.Tile8,weights_tree),Tree.pack(U32,offsets_tree)} ==
      Kernel.ImageState{Tree.pack(F32,gradient_tree),_,offsets} : Kernel.ImageState}
    %offsets_packed : {Kernel.ImageState{Tree.pack(F32,gradient_tree),Tree.pack(Tile.Tile8,weights_tree),Tree.pack(U32,offsets_tree)} ==
      Kernel.ImageState{Tree.pack(F32,gradient_tree),Tree.pack(Tile.Tile8,weights_tree),_} : Kernel.ImageState}
    {==}}
def held(state: Kernel.ImageState) -> Each.Held<Kernel.ImageState,State,represent,state>:
  Kernel.ImageState{gradient,weights,offsets} = state
  held_parts(gradient,weights,offsets,StorageLaws.observe(F32,gradient),StorageLaws.observe(Tile.Tile8,weights),StorageLaws.observe(U32,offsets))
def see(state: Kernel.ImageState) -> State:
  Kernel.ImageState{gradient,weights,offsets} = state
  State{Tree.reflect(F32,gradient),Tree.reflect(Tile.Tile8,weights),Tree.reflect(U32,offsets)}
def seen_packed(+state: State) -> {see(represent(state)) == state : State}:
  State{+gradient,+weights,+offsets} = state
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,gradient)),gradient,StorageLaws.reflect_packed(F32,gradient)) :
    {State{_,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),Tree.reflect(U32,Tree.pack(U32,offsets))} == State{gradient,weights,offsets} : State}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),weights,StorageLaws.reflect_packed(Tile.Tile8,weights)) :
    {State{gradient,_,Tree.reflect(U32,Tree.pack(U32,offsets))} == State{gradient,weights,offsets} : State}
  %Equal.sym(Tree.Tree<U32>,Tree.reflect(U32,Tree.pack(U32,offsets)),offsets,StorageLaws.reflect_packed(U32,offsets)) :
    {State{gradient,weights,_} == State{gradient,weights,offsets} : State}
  {==}

# The bands one image makes from what is seen of the state: its pieces over
# the row plan of the depth, at its rows of the batch.
def image_owners(+geometry: Kernel.Geometry,+first: U32,+state: State) -> Parts.Owners:
  Kernel.Geometry{c,+o,+h,w,k,s,p,+gh,+gw} = geometry
  State{gradient,weights,offsets} = state
  Parts.Owners{Execution.Source{gradient,(first / h * o * gh * gw : U32),(gh * gw : U32),0,gh},weights,offsets}
def image_pieces(+context: Kernel.Images,+first: U32,+state: State) -> Partition.Tree<&2,Execution.Part,Unit>:
  Kernel.Images{+geometry,+layout,+depth} = context
  +height = Domain.height_word(geometry)
  Parts.model(Bands.row_plan(depth,U32.is_lt(1,height),0,height),Kernel.Task{geometry,layout},image_owners(geometry,first,state))
def image_part(context: Kernel.Images,state: State,+first: U32,+rows: U32) -> Model.Content<F32>:
  Placement.bands(first,image_pieces(context,first,state))

# One image on any state gives back the state it found, and its bands are the
# bands of its pieces.
def image_seen(+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,made: Model.Content<F32> & Tree.Tree<F32>) -> State & Model.Content<F32>:
  (bands,gradient) = made
  (State{gradient,weights,offsets},bands)
def image_seen_equation(+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,made: Bands.Bands(~F32) & Array<F32>) ->
  {Each.seen(~F32,~Kernel.ImageState,~State,~see,Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),made)) ==
    image_seen(weights,offsets,Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~Placement.see,made)) : State & Model.Content<F32>}:
  (bands,gradient) = made
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),weights,StorageLaws.reflect_packed(Tile.Tile8,weights)) :
    {(State{Tree.reflect(F32,gradient),_,Tree.reflect(U32,Tree.pack(U32,offsets))},Model.contents(~F32,bands)) ==
      (State{Tree.reflect(F32,gradient),weights,offsets},Model.contents(~F32,bands)) : State & Model.Content<F32>}
  %Equal.sym(Tree.Tree<U32>,Tree.reflect(U32,Tree.pack(U32,offsets)),offsets,StorageLaws.reflect_packed(U32,offsets)) :
    {(State{Tree.reflect(F32,gradient),weights,_},Model.contents(~F32,bands)) == (State{Tree.reflect(F32,gradient),weights,offsets},Model.contents(~F32,bands)) : State & Model.Content<F32>}
  {==}
def image_rule(+context: Kernel.Images,+first: U32,+rows: U32,+kept: State) ->
  {Each.seen(~F32,~Kernel.ImageState,~State,~see,Kernel.image(context,first,rows,represent(kept))) ==
    (see(represent(kept)),image_part(context,see(represent(kept)),first,rows)) : State & Model.Content<F32>}:
  Kernel.Images{+geometry,+layout,+depth} = context
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  State{+gradient,+weights,+offsets} = kept
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +images = {Kernel.Images{shape,layout,depth} : Kernel.Images}
  +state = {State{gradient,weights,offsets} : State}
  +plan = Bands.row_plan(depth,U32.is_lt(1,h),0,h)
  +task = {Kernel.Task{shape,layout} : Kernel.Task}
  +owners = image_owners(shape,first,state)
  +pieces = image_pieces(images,first,state)
  +runtime = Equal.trans(Kernel.ImageState & Bands.Bands(~F32),
    Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights))),
    Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),(Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights))),
    Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0)))),
    Equal.cong(Array<Tile.Tile8> & Array<Tile.Tile8>,Kernel.ImageState & Bands.Bands(~F32),
      pair => Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),pair),
      Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)),(Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights)),StorageLaws.clone(Tile.Tile8,weights)),
    Equal.trans(Kernel.ImageState & Bands.Bands(~F32),
      Kernel.image_offsets(first,images,Tree.pack(F32,gradient),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights),Array.clone(U32,Tree.pack(U32,offsets))),
      Kernel.image_offsets(first,images,Tree.pack(F32,gradient),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights),(Tree.pack(U32,offsets),Tree.pack(U32,offsets))),
      Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0)))),
      Equal.cong(Array<U32> & Array<U32>,Kernel.ImageState & Bands.Bands(~F32),
        pair => Kernel.image_offsets(first,images,Tree.pack(F32,gradient),Tree.pack(Tile.Tile8,weights),Tree.pack(Tile.Tile8,weights),pair),
        Array.clone(U32,Tree.pack(U32,offsets)),(Tree.pack(U32,offsets),Tree.pack(U32,offsets)),StorageLaws.clone(U32,offsets)),
      Equal.trans(Kernel.ImageState & Bands.Bands(~F32),
        Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,
          Bands.each_part(~Kernel.Owners,~Kernel.Part,~Unit,~Kernel.Task,~Unit,~Bands.always,~Kernel.divide,~Kernel.leaf,~Kernel.whole,plan,True{},Unit{},task,Parts.represent(owners)),
          Array.new(F32,0n,0.0))),
        Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,
          Parts.represented(pieces),Array.new(F32,0n,0.0))),
        Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0)))),
        Equal.cong(Partition.Tree<&1,Kernel.Part,Unit>,Kernel.ImageState & Bands.Bands(~F32),
          tree => Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,tree,Array.new(F32,0n,0.0))),
          Bands.each_part(~Kernel.Owners,~Kernel.Part,~Unit,~Kernel.Task,~Unit,~Bands.always,~Kernel.divide,~Kernel.leaf,~Kernel.whole,plan,True{},Unit{},task,Parts.represent(owners)),
          Parts.represented(pieces),Parts.walk(plan,True{},task,owners)),
        Equal.cong(Array<F32>,Kernel.ImageState & Bands.Bands(~F32),
          storage => Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),storage)),
          Array.new(F32,0n,0.0),Tree.pack(F32,Tree.fresh(F32,0n,0.0)),StorageLaws.fresh(F32,0n,0.0)))))
  Equal.trans(State & Model.Content<F32>,
    Each.seen(~F32,~Kernel.ImageState,~State,~see,Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)))),
    image_seen(weights,offsets,Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~Placement.see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0))))),
    (see(represent(state)),image_part(images,see(represent(state)),first,rows)),
    Equal.trans(State & Model.Content<F32>,
      Each.seen(~F32,~Kernel.ImageState,~State,~see,Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights)))),
      Each.seen(~F32,~Kernel.ImageState,~State,~see,Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0))))),
      image_seen(weights,offsets,Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~Placement.see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0))))),
      Equal.cong(Kernel.ImageState & Bands.Bands(~F32),State & Model.Content<F32>,made => Each.seen(~F32,~Kernel.ImageState,~State,~see,made),
        Kernel.image_weights(first,images,Tree.pack(F32,gradient),Tree.pack(U32,offsets),Array.clone(Tile.Tile8,Tree.pack(Tile.Tile8,weights))),
        Kernel.image_bands(Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0)))),runtime),
      image_seen_equation(weights,offsets,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0))))),
    Equal.trans(State & Model.Content<F32>,image_seen(weights,offsets,Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~Placement.see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0))))),
      (State{Parts.last(pieces),weights,offsets},Placement.bands(first,pieces)),
      (see(represent(state)),image_part(images,see(represent(state)),first,rows)),
      Equal.cong(Model.Content<F32> & Tree.Tree<F32>,State & Model.Content<F32>,made => image_seen(weights,offsets,made),
        Each.walk_seen(~F32,~Array<F32>,~Tree.Tree<F32>,~Placement.see,Bands.walk(~Kernel.Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~Kernel.placed,first,Parts.represented(pieces),Placement.hold(Tree.fresh(F32,0n,0.0)))),(Placement.bands(first,pieces),Parts.last(pieces)),
        Placement.placed(pieces,first,Tree.fresh(F32,0n,0.0))),
      Equal.trans(State & Model.Content<F32>,(State{Parts.last(pieces),weights,offsets},Placement.bands(first,pieces)),
        (state,image_part(images,state,first,rows)),(see(represent(state)),image_part(images,see(represent(state)),first,rows)),
        Equal.cong(Tree.Tree<F32>,State & Model.Content<F32>,last => (State{last,weights,offsets},Placement.bands(first,pieces)),
          Parts.last(pieces),gradient,Parts.model_last(plan,task,owners)),
        Equal.cong(State,State & Model.Content<F32>,model => (model,image_part(images,model,first,rows)),
          state,see(represent(state)),Equal.sym(State,see(represent(state)),state,seen_packed(state))))))

# Every image of a plan of images, in order, on the state.
def images_seen(+context: Kernel.Images,+layout: Bands.Plan,+kept: State) ->
  {Each.seen(~F32,~Kernel.ImageState,~State,~see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,layout,context,represent(kept))) ==
    (see(represent(kept)),Model.planned(~F32,~image_part(context,see(represent(kept))),layout)) : State & Model.Content<F32>}:
  Each.built_parts(~F32,~Kernel.ImageState,~Kernel.Images,~State,~represent,~held,~State,~see,~Kernel.image,~image_part,~image_rule,layout,context,kept)

# ---- The batch ----

# The images of the batch with everything their facts name: the reader, the
# machine fit of the leaf theorem, the split depth, the
# capacity of the output and the packed trees the images read.
def packing(+setup: Setup) -> Kernel.Packed:
  Packed.native_complete(PackingBounds.Query{False{},setup_call(setup)},Kernel.tiling(setup_geometry(setup)),setup_weights(setup))
type Batch is Data:
  Batch{setup: Setup,envelope: Geometry.WordEnvelope,
    fits: {Kernel.fits(setup_geometry(setup),Machine.capacity(Domain.limits(setup_call(setup)))) == True{} : Bool},
    depth: Nat,capacity: U32,allocatable: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(setup_images(setup),size(setup)),U32.to_nat(capacity)) == True{} : Bool},
    limits: Machine.Limits,machine_room: {Nat.is_le(Nat.mul(setup_images(setup),size(setup)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,
    weights_same: {Tree.pack(Tile.Tile8,weights) == Leaves.native_weights(packing(setup)) : Array<Tile.Tile8>},
    offsets_same: {Tree.pack(U32,offsets) == Leaves.native_offsets(packing(setup)) : Array<U32>}}
def batch_setup(batch: Batch) -> Setup:
  Batch{setup,envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same} = batch
  setup
def batch_images(+batch: Batch) -> Nat:
  setup_images(batch_setup(batch))
def batch_height(+batch: Batch) -> U32:
  Domain.height_word(setup_geometry(batch_setup(batch)))
def batch_geometry(+batch: Batch) -> Bands.Geometry:
  band_geometry(setup_geometry(batch_setup(batch)),batch_images(batch))
def batch_context(+batch: Batch) -> Kernel.Images:
  Batch{setup,envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same} = batch
  +geometry = setup_geometry(setup)
  Kernel.Images{geometry,Kernel.tiling(geometry),depth}
def batch_state(+batch: Batch) -> State:
  Batch{setup,envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same} = batch
  State{Buffer.tree(~F32,setup_gradient(setup)),weights,offsets}
def batch_part(+batch: Batch,+first: U32,+rows: U32) -> Model.Content<F32>:
  image_part(batch_context(batch),batch_state(batch),first,rows)
def batch_plan(+batch: Batch) -> Bands.Plan:
  Bands.image_plan(batch_images(batch),0,0n,batch_height(batch))
# The parts of the batch are the parts of its images.
def batch_parts(+batch: Batch,layout: Bands.Plan) ->
  {Model.planned(~F32,~image_part(batch_context(batch),batch_state(batch)),layout) == Model.planned(~F32,~batch_part(batch),layout) : Model.Content<F32>}:
  match layout:
    case Bands.Rows{first,rows}: {==}
    case Bands.Split{+left,+right}:
      %Equal.sym(Model.Content<F32>,Model.planned(~F32,~image_part(batch_context(batch),batch_state(batch)),left),Model.planned(~F32,~batch_part(batch),left),batch_parts(batch,left)) :
        {Model.Joined{_,Model.planned(~F32,~image_part(batch_context(batch),batch_state(batch)),right)} ==
          Model.Joined{Model.planned(~F32,~batch_part(batch),left),Model.planned(~F32,~batch_part(batch),right)} : Model.Content<F32>}
      %Equal.sym(Model.Content<F32>,Model.planned(~F32,~image_part(batch_context(batch),batch_state(batch)),right),Model.planned(~F32,~batch_part(batch),right),batch_parts(batch,right)) :
        {Model.Joined{Model.planned(~F32,~batch_part(batch),left),_} ==
          Model.Joined{Model.planned(~F32,~batch_part(batch),left),Model.planned(~F32,~batch_part(batch),right)} : Model.Content<F32>}
      {==}

# The runtime images of the batch make its bands.
def built(+batch: Batch) ->
  {Each.seen(~F32,~Kernel.ImageState,~State,~see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,batch_plan(batch),batch_context(batch),represent(batch_state(batch)))) ==
    (batch_state(batch),Model.planned(~F32,~batch_part(batch),batch_plan(batch))) : State & Model.Content<F32>}:
  +state = batch_state(batch)
  %batch_parts(batch,batch_plan(batch)) : {Each.seen(~F32,~Kernel.ImageState,~State,~see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,batch_plan(batch),batch_context(batch),represent(state))) ==
    (state,_) : State & Model.Content<F32>}
  %seen_packed(state) : {Each.seen(~F32,~Kernel.ImageState,~State,~see,Bands.build(~Storage.Buffer<F32>,~Kernel.ImageState,~Kernel.Images,~Kernel.image,batch_plan(batch),batch_context(batch),represent(state))) ==
    (_,Model.planned(~F32,~image_part(batch_context(batch),_),batch_plan(batch))) : State & Model.Content<F32>}
  images_seen(batch_context(batch),batch_plan(batch),state)

# A leaf of the plan of images: the rows of one image of the batch.
type ImageRows is Data:
  ImageRows{images: Nat,height: U32}
def rows_images(rows: ImageRows) -> Nat:
  ImageRows{images,height} = rows
  images
def rows_height(rows: ImageRows) -> U32:
  ImageRows{images,height} = rows
  height
def batch_rows(+batch: Batch) -> ImageRows:
  ImageRows{batch_images(batch),batch_height(batch)}
type ImageLeaf<-rows: ImageRows,-first: U32,-count: U32> is Data:
  ImageLeaf{index: U32,inside: {Nat.is_lt(U32.to_nat(index),rows_images(rows)) == True{} : Bool},
    first_exact: {(index * rows_height(rows) : U32) == first : U32},count_exact: {rows_height(rows) == count : U32}}
def Leaf(rows: ImageRows,+first: U32,+count: U32) -> Data:
  ImageLeaf<rows,first,count>

def room_inside(+rest: Nat,+done: Nat,+images: Nat,+room: {Nat.is_le(Nat.add(done,1n+rest),images) == True{} : Bool}) -> {Nat.is_lt(done,images) == True{} : Bool}:
  +step = {
    %Equal.sym(Nat,Nat.add(done,1n+rest),1n+Nat.add(done,rest),Addition.successor_right(done,rest)) : {Nat.is_le(1n+done,_) == True{} : Bool}
    Order.le_add(done,rest) : {Nat.is_le(1n+done,Nat.add(done,1n+rest)) == True{} : Bool}}
  Bounds.predecessor_strict(done,images,Order.transitive(1n+done,Nat.add(done,1n+rest),images,step,room))
def image_leaves(rest: Nat,+image: U32,+done: Nat,+rows: ImageRows,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +roundtrip: {U32.to_nat(image) == done : Nat},+room: {Nat.is_le(Nat.add(done,1n+rest),rows_images(rows)) == True{} : Bool},
  +images_fit: {Nat.is_le(rows_images(rows),U32.to_nat(capacity)) == True{} : Bool}) ->
  Model.PlanAll(~ImageRows,~Leaf,rows,Bands.image_plan(1n+rest,image,0n,rows_height(rows))):
  match rest:
    case 0n:
      ImageLeaf{image,%Equal.sym(Nat,U32.to_nat(image),done,roundtrip) : {Nat.is_lt(_,rows_images(rows)) == True{} : Bool} room_inside(0n,done,rows_images(rows),room),{==},{==}}
    case 1n+ +more:
      +inside_done = room_inside(1n+more,done,rows_images(rows),room)
      +inside = {
        %Equal.sym(Nat,U32.to_nat(image),done,roundtrip) : {Nat.is_lt(_,rows_images(rows)) == True{} : Bool}
        inside_done : {Nat.is_lt(U32.to_nat(image),rows_images(rows)) == True{} : Bool}}
      +successor = Equal.trans(Nat,Nat.add(done,1n),1n+Nat.add(done,0n),1n+done,Addition.successor_right(done,0n),
        Equal.cong(Nat,Nat,value => 1n+value,Nat.add(done,0n),done,Addition.zero_right(done)))
      +bound = {
        %Equal.sym(Nat,U32.to_nat(image),done,roundtrip) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(done,1n),1n+done,successor) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Order.transitive(1n+done,rows_images(rows),U32.to_nat(capacity),Order.strict_follow(done,rows_images(rows),inside_done),images_fit) :
          {Nat.is_le(Nat.add(U32.to_nat(image),U32.to_nat(1)),U32.to_nat(capacity)) == True{} : Bool}}
      +next_exact = Equal.trans(Nat,U32.to_nat((image + 1 : U32)),Nat.add(U32.to_nat(image),1n),Nat.add(done,1n),
        Arithmetic.add(image,1,capacity,capacity_ok,bound),Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(image),done,roundtrip))
      +next_room = {
        %Equal.sym(Nat,Nat.add(Nat.add(done,1n),1n+more),Nat.add(done,Nat.add(1n,1n+more)),Addition.associative(done,1n,1n+more)) : {Nat.is_le(_,rows_images(rows)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(Nat.add(done,1n),1n+more),rows_images(rows)) == True{} : Bool}}
      Shape.Both{ImageLeaf{image,inside,{==},{==}},
        image_leaves(more,(image + 1 : U32),Nat.add(done,1n),rows,capacity,capacity_ok,next_exact,next_room,images_fit)}

# One leaf of the plan: the image of a word of the batch, read as the model
# of its pieces.
def leaf_sample(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) -> Leaves.Sample:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Leaves.Sample{call,envelope,fits,gradient,weights_model,index,inside}
def leaf_scope(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) -> Scope:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Scope{leaf_sample(Batch{Setup{call,gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same},index,inside),
    depth,capacity,allocatable,room,limits,machine_room}
def leaf_source(+batch: Batch,+index: U32) -> Execution.Source:
  +geometry = setup_geometry(batch_setup(batch))
  Execution.Source{Buffer.tree(~F32,setup_gradient(batch_setup(batch))),Packing.image_base(geometry,index),Packing.plane_size(geometry),0,Domain.gradient_rows(geometry)}
def batch_weights(batch: Batch) -> Tree.Tree<Tile.Tile8>:
  Batch{setup,envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same} = batch
  weights
def batch_offsets(batch: Batch) -> Tree.Tree<U32>:
  Batch{setup,envelope,fits,depth,capacity,allocatable,room,limits,machine_room,weights,offsets,weights_same,offsets_same} = batch
  offsets
def leaf_model(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) -> Model.Model<F32>:
  model_of(leaf_scope(batch,index,inside),leaf_source(batch,index),batch_weights(batch),batch_offsets(batch))

# The image whose rows start at a multiple of the height reads the gradient
# of that image.
def start_index(+index: U32,+height: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +start_room: {Nat.is_le(Nat.mul(U32.to_nat(index),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +height_room: {Nat.is_le(U32.to_nat(height),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {((index * height : U32) / height : U32) == index : U32}:
  +i = U32.to_nat(index)
  +hn = U32.to_nat(height)
  +product = Arithmetic.multiply(index,height,Machine.capacity(limits),Limits.capacity_fits(limits),start_room)
  Arithmetic.to_nat_injective(((index * height : U32) / height : U32),index,
    Equal.trans(Nat,U32.to_nat(((index * height : U32) / height : U32)),Nat.div(U32.to_nat((index * height : U32)),hn),i,
      U32Division.divide((index * height : U32),height,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,hn,height_room),positive),
      %Equal.sym(Nat,U32.to_nat((index * height : U32)),Nat.mul(i,hn),product) : {Nat.div(_,hn) == i : Nat}
      %Addition.zero_right(Nat.mul(i,hn)) : {Nat.div(_,hn) == i : Nat}
      Radix.quotient(i,0n,hn,positive,positive)))
def leaf_content(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) ->
  {batch_part(batch,(index * batch_height(batch) : U32),batch_height(batch)) == Model.content(~F32,leaf_model(batch,index,inside)) : Model.Content<F32>}:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+count,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,count,limits_call,facts} : Domain.Call}
  +plan = Bands.row_plan(depth,U32.is_lt(1,h),0,h)
  +task = {Kernel.Task{shape,Kernel.tiling(shape)} : Kernel.Task}
  +i = U32.to_nat(index)
  +hn = U32.to_nat(h)
  +rows_room = batch_rows_room(count,c,h,w,Machine.capacity(limits),Domain.channels_positive(whole),Domain.width_positive(whole),machine_room)
  +image_room = image_rows_room(count,h,index,Machine.capacity(limits),inside,rows_room)
  +start_room = Order.transitive(Nat.mul(i,hn),Nat.mul(1n+i,hn),U32.to_nat(Machine.capacity(limits)),
    %Order.commutative(Nat.mul(i,hn),hn) : {Nat.is_le(Nat.mul(i,hn),_) == True{} : Bool} Order.le_add(Nat.mul(i,hn),hn),image_room)
  +height_room = Order.transitive(hn,Nat.mul(1n+i,hn),U32.to_nat(Machine.capacity(limits)),Order.le_add(hn,Nat.mul(i,hn)),image_room)
  +quotient = start_index(index,h,limits,Domain.height_positive(whole),start_room,height_room)
  Equal.cong(U32,Model.Content<F32>,base => Placement.bands((index * h : U32),Parts.model(plan,task,
      Parts.Owners{Execution.Source{Buffer.tree(~F32,gradient),base,(gh * gw : U32),0,gh},weights,offsets})),
    ((index * h : U32) / h * o * gh * gw : U32),(index * o * gh * gw : U32),
    Equal.cong(U32,U32,word => (word * o * gh * gw : U32),((index * h : U32) / h : U32),index,quotient))

# The facts of a leaf of the plan, from the facts of its image.
def leaf_held(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) ->
  Retained.Held<Leaves.setting(scope_sample(leaf_scope(batch,index,inside))),0,Domain.height_word(Leaves.geometry(scope_sample(leaf_scope(batch,index,inside)))),leaf_source(batch,index)>:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+count,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +sample = leaf_sample(Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},count,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same},index,inside)
  Retained.full_held(Leaves.setting(sample),0,h,Order.reflexive(U32.to_nat(h)))
def leaf_weights(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) ->
  {Tree.pack(Tile.Tile8,batch_weights(batch)) == Leaves.native_weights(Leaves.packing(scope_sample(leaf_scope(batch,index,inside)))) : Array<Tile.Tile8>}:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  weights_same
def leaf_offsets(+batch: Batch,+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),batch_images(batch)) == True{} : Bool}) ->
  {Tree.pack(U32,batch_offsets(batch)) == Leaves.native_offsets(Leaves.packing(scope_sample(leaf_scope(batch,index,inside)))) : Array<U32>}:
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  offsets_same

def fields_at(+batch: Batch) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  BandShape.band_fields(~F32,batch_geometry(batch))
def ready_at(+batch: Batch) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  Model.ready_band(~F32)
def values_at(+batch: Batch) -> @+first: U32 -> @+rows: U32 -> @+band: Buffer.Content<F32> -> Data:
  BandValues.band_values(~F32,~Setup,~value,batch_setup(batch),batch_geometry(batch))

def leaf_fields(+batch: Batch,+first: U32,+count: U32,leaf: Leaf(batch_rows(batch),first,count)) -> Model.ContentAll(~F32,~fields_at(batch),batch_part(batch,first,count)):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  ImageLeaf{+index,+inside,first_exact,count_exact} = leaf
  +frame = {Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same} : Batch}
  %first_exact : Model.ContentAll(~F32,~fields_at(frame),batch_part(frame,_,count))
  %count_exact : Model.ContentAll(~F32,~fields_at(frame),batch_part(frame,(index * h : U32),_))
  %Equal.sym(Model.Content<F32>,batch_part(frame,(index * h : U32),h),Model.content(~F32,leaf_model(frame,index,inside)),leaf_content(frame,index,inside)) :
    Model.ContentAll(~F32,~fields_at(frame),_)
  metadata_of(leaf_scope(frame,index,inside),leaf_source(frame,index),weights,offsets,leaf_held(frame,index,inside),leaf_weights(frame,index,inside),leaf_offsets(frame,index,inside))
def leaf_ready(+batch: Batch,+first: U32,+count: U32,leaf: Leaf(batch_rows(batch),first,count)) -> Model.ContentAll(~F32,~ready_at(batch),batch_part(batch,first,count)):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  ImageLeaf{+index,+inside,first_exact,count_exact} = leaf
  +frame = {Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same} : Batch}
  %first_exact : Model.ContentAll(~F32,~ready_at(frame),batch_part(frame,_,count))
  %count_exact : Model.ContentAll(~F32,~ready_at(frame),batch_part(frame,(index * h : U32),_))
  %Equal.sym(Model.Content<F32>,batch_part(frame,(index * h : U32),h),Model.content(~F32,leaf_model(frame,index,inside)),leaf_content(frame,index,inside)) :
    Model.ContentAll(~F32,~ready_at(frame),_)
  ready_of(leaf_scope(frame,index,inside),leaf_source(frame,index),weights,offsets,leaf_held(frame,index,inside),leaf_weights(frame,index,inside),leaf_offsets(frame,index,inside))
def leaf_values(+batch: Batch,+first: U32,+count: U32,leaf: Leaf(batch_rows(batch),first,count)) -> Model.ContentAll(~F32,~values_at(batch),batch_part(batch,first,count)):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  ImageLeaf{+index,+inside,first_exact,count_exact} = leaf
  +frame = {Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same} : Batch}
  %first_exact : Model.ContentAll(~F32,~values_at(frame),batch_part(frame,_,count))
  %count_exact : Model.ContentAll(~F32,~values_at(frame),batch_part(frame,(index * h : U32),_))
  %Equal.sym(Model.Content<F32>,batch_part(frame,(index * h : U32),h),Model.content(~F32,leaf_model(frame,index,inside)),leaf_content(frame,index,inside)) :
    Model.ContentAll(~F32,~values_at(frame),_)
  values_of(leaf_scope(frame,index,inside),leaf_source(frame,index),weights,offsets,leaf_held(frame,index,inside),leaf_weights(frame,index,inside),leaf_offsets(frame,index,inside))
def leaf_coverage(+batch: Batch,+first: U32,+count: U32,leaf: Leaf(batch_rows(batch),first,count)) ->
  BandPlan.Coverage(Model.content_layout(~F32,batch_part(batch,first,count)),U32.to_nat(first),U32.to_nat(count)):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+count_images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  ImageLeaf{+index,+inside,first_exact,count_exact} = leaf
  +frame = {Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},count_images,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same} : Batch}
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},count_images,limits_call,facts} : Domain.Call}
  +model = leaf_model(frame,index,inside)
  +i = U32.to_nat(index)
  +hn = U32.to_nat(h)
  +image_room = image_rows_room(count_images,h,index,capacity,inside,batch_rows_room(count_images,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room))
  +start_exact = Arithmetic.multiply(index,h,capacity,capacity_ok(capacity,allocatable),
    Order.transitive(Nat.mul(i,hn),Nat.mul(1n+i,hn),U32.to_nat(capacity),%Order.commutative(Nat.mul(i,hn),hn) : {Nat.is_le(Nat.mul(i,hn),_) == True{} : Bool} Order.le_add(Nat.mul(i,hn),hn),image_room))
  %first_exact : BandPlan.Coverage(Model.content_layout(~F32,batch_part(frame,_,count)),U32.to_nat(_),U32.to_nat(count))
  %count_exact : BandPlan.Coverage(Model.content_layout(~F32,batch_part(frame,(index * h : U32),_)),U32.to_nat((index * h : U32)),U32.to_nat(_))
  %Equal.sym(Model.Content<F32>,batch_part(frame,(index * h : U32),h),Model.content(~F32,model),leaf_content(frame,index,inside)) :
    BandPlan.Coverage(Model.content_layout(~F32,_),U32.to_nat((index * h : U32)),hn)
  %Equal.sym(Nat,U32.to_nat((index * h : U32)),Nat.mul(i,hn),start_exact) : BandPlan.Coverage(Model.content_layout(~F32,Model.content(~F32,model)),_,hn)
  %Model.layout_matches_content(~F32,model) : BandPlan.Coverage(_,Nat.mul(i,hn),hn)
  layout_of(leaf_scope(frame,index,inside),leaf_source(frame,index),weights,offsets)

# The bands of the batch: well formed, and every band holds the value of its
# cells.
def batch_leaves(+batch: Batch,+rest: Nat,+count: {batch_images(batch) == 1n+rest : Nat}) -> Model.PlanAll(~ImageRows,~Leaf,batch_rows(batch),batch_plan(batch)):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts} : Domain.Call}
  +extent = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
  +images_fit = Order.transitive(images,Nat.mul(images,extent),U32.to_nat(capacity),RowMajor.product_covers(images,extent,Domain.image_positive(whole)),room)
  +rows = {ImageRows{images,h} : ImageRows}
  %Equal.sym(Nat,images,1n+rest,count) : Model.PlanAll(~ImageRows,~Leaf,rows,Bands.image_plan(_,0,0n,h))
  image_leaves(rest,0,0n,rows,capacity,capacity_ok(capacity,allocatable),{==},
    %Equal.sym(Nat,images,1n+rest,count) : {Nat.is_le(1n+rest,_) == True{} : Bool} Order.reflexive(1n+rest),images_fit)
def batch_coverage(+batch: Batch,evidence: Model.PlanAll(~ImageRows,~Leaf,batch_rows(batch),batch_plan(batch))) ->
  BandPlan.Coverage(Model.content_layout(~F32,Model.planned(~F32,~batch_part(batch),batch_plan(batch))),0n,Inspection.total_rows(batch_geometry(batch))):
  Batch{+setup,+envelope,+fits,+depth,+capacity,+allocatable,+room,+limits,+machine_room,+weights,+offsets,+weights_same,+offsets_same} = batch
  Setup{+call,+gradient,+weights_model} = setup
  Domain.Call{+geometry,+images,+limits_call,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +frame = {Batch{Setup{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts},gradient,weights_model},envelope,fits,depth,capacity,allocatable,room,
    limits,machine_room,weights,offsets,weights_same,offsets_same} : Batch}
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},images,limits_call,facts} : Domain.Call}
  +extent = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
  +images_fit = Order.transitive(images,Nat.mul(images,extent),U32.to_nat(capacity),RowMajor.product_covers(images,extent,Domain.image_positive(whole)),room)
  +rows_fit = batch_rows_room(images,c,h,w,capacity,Domain.channels_positive(whole),Domain.width_positive(whole),room)
  +images_exact = FullRange.roundtrip(images,capacity,images_fit)
  +plan_coverage = {
    %BandPlan.first_zero(images,h) : BandPlan.Coverage(Bands.image_plan(images,0,0n,h),_,Nat.mul(images,U32.to_nat(h)))
    BandPlan.refinement_image_plan(images,0,0n,h,capacity,capacity_ok(capacity,allocatable),images_fit,rows_fit) :
      BandPlan.Coverage(Bands.image_plan(images,0,0n,h),0n,Nat.mul(images,U32.to_nat(h)))}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(images)),images,images_exact) : BandPlan.Coverage(Model.content_layout(~F32,Model.planned(~F32,~batch_part(frame),batch_plan(frame))),0n,Nat.mul(_,U32.to_nat(h)))
  Model.planned_coverage(~F32,~Batch,~batch_part,~ImageRows,~Leaf,~batch_rows,~leaf_coverage,frame,batch_plan(frame),0n,Nat.mul(images,U32.to_nat(h)),evidence,plan_coverage)
def batch_facts(+batch: Batch,+rest: Nat,+count: {batch_images(batch) == 1n+rest : Nat}) ->
  Shape.Both<BandValues.Values(~F32,~Setup,~value,batch_setup(batch),batch_geometry(batch),Model.planned(~F32,~batch_part(batch),batch_plan(batch))),BandShape.WellFormed<F32,batch_geometry(batch),Model.planned(~F32,~batch_part(batch),batch_plan(batch))>>:
  +leaves = batch_leaves(batch,rest,count)
  Shape.Both{Model.planned_all(~F32,~Batch,~batch_part,~ImageRows,~Leaf,~batch_rows,~values_at,~leaf_values,batch,batch_plan(batch),leaves),
    BandShape.WellFormed{Model.planned_all(~F32,~Batch,~batch_part,~ImageRows,~Leaf,~batch_rows,~fields_at,~leaf_fields,batch,batch_plan(batch),leaves),
      Model.planned_all(~F32,~Batch,~batch_part,~ImageRows,~Leaf,~batch_rows,~ready_at,~leaf_ready,batch,batch_plan(batch),leaves),
      batch_coverage(batch,leaves)}}
