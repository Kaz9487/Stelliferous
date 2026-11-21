# The cells [first, first + count) of the input gradient a task computes:
# the rows that hold them, computed whole, the cells cut from them and +0
# in the lanes past the channels hold the lifted input gradient there.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/tensor_move.bend as Move
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_tiled_map.bend as TiledMap
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/tensor_grid.bend as Grid
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
import ../../lib/tensor_view.bend as Views
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_band_source.bend as BandSource
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_readback.bend as StorageRead
import ../../lib/proofs/storage_allocation_refinement.bend as AllocationLaws
import ../../lib/proofs/storage_copy_refinement.bend as CopyLaws
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy_readback.bend as CopyRead
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/tensor_whole_values.bend as WholeValues
import ../../lib/proofs/tensor_pieces_content.bend as Content
import ../../lib/proofs/tensor_tiled_padding.bend as Padding
import ../../lib/proofs/tensor_tiled_padded_cells.bend as PaddedCells
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/row_major_coordinates.bend as Coordinates
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/u32_division.bend as NativeDivision
import ../../lib/proofs/u32_extrema.bend as Extrema
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_batch_rows.bend as BatchRows
import ./input_gradient_cell_sources.bend as Sources
import ../input_gradient_specification.bend as Specification

# ---- The input gradient of the call ----

type Setup is Data:
  Setup{call: Domain.Call,gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>}
def setup_call(setup: Setup) -> Domain.Call:
  Setup{call,gradient,weights} = setup
  call
def setup_gradient(setup: Setup) -> Buffer.Model<Tile.Tile8>:
  Setup{call,gradient,weights} = setup
  gradient
def setup_weights(setup: Setup) -> Buffer.Model<Tile.Tile8>:
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
  Sources.problem(setup_geometry(setup),Sources.models_reader(setup_geometry(setup),setup_gradient(setup),setup_weights(setup)),image)
def input_value(setup: Setup,index: U32) -> F32:
  Setup{+call,+gradient,+weights} = setup
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  +point = Domain.point(geometry,U32.to_nat(index))
  Specification.value(spec(Setup{Domain.Call{geometry,batch,limits,facts},gradient,weights},Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point))

# The band geometry of the output and the index of a cell of it.

# The index of a point of the batch: image, channel, row and column.
def cell_point(+geometry: Kernel.Geometry,+images: Nat,+image: Nat,+channel: Nat,+row: Nat,+column: Nat) -> Nat:
  RowMajor.index(Nat.add(Nat.mul(image,Domain.channels(geometry)),channel),Domain.height(geometry),Domain.width(geometry),row,column)

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
  {input_value(setup,U32.from_nat(cell_point(setup_geometry(setup),setup_images(setup),image,channel,row,column))) == Specification.value(spec(setup,image),channel,row,column) : F32}:
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
      {Domain.from_coordinates(_,Coordinates.values(hn,wn,Nat.mod(position,extent))) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(position,extent),local,Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(position,extent),(image,local),parts)) :
      {Domain.from_coordinates(image,Coordinates.values(hn,wn,_)) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.div(local,plane),channel,Radix.quotient(channel,within,plane,plane_positive,within_inside)) :
      {Domain.from_coordinates(image,Coordinates.Values{_,Nat.div(Nat.mod(local,plane),wn),Nat.mod(Nat.mod(local,plane),wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(local,plane),within,Radix.remainder(channel,within,plane,plane_positive,within_inside)) :
      {Domain.from_coordinates(image,Coordinates.Values{channel,Nat.div(_,wn),Nat.mod(_,wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.div(within,wn),row,Radix.quotient(row,column,wn,width_positive,column_inside)) :
      {Domain.from_coordinates(image,Coordinates.Values{channel,_,Nat.mod(within,wn)}) == Domain.Point{image,channel,row,column} : Domain.Point}
    %Equal.sym(Nat,Nat.mod(within,wn),column,Radix.remainder(row,column,wn,width_positive,column_inside)) :
      {Domain.from_coordinates(image,Coordinates.Values{channel,row,_}) == Domain.Point{image,channel,row,column} : Domain.Point}
    {==} : {Domain.point(shape,position) == Domain.Point{image,channel,row,column} : Domain.Point}}
  Equal.cong(Domain.Point,F32,point => Specification.value(spec(reader,Domain.image(point)),Domain.channel(point),Domain.row(point),Domain.column(point)),
    Domain.point(shape,U32.to_nat(U32.from_nat(position))),Domain.Point{image,channel,row,column},
    Equal.trans(Domain.Point,Domain.point(shape,U32.to_nat(U32.from_nat(position))),Domain.point(shape,position),Domain.Point{image,channel,row,column},
      Equal.cong(Nat,Domain.Point,index => Domain.point(shape,index),U32.to_nat(U32.from_nat(position)),position,
        FullRange.roundtrip(position,capacity,Bounds.strict_le(position,U32.to_nat(capacity),bound))),
      decoded_point))

# ---- The cells ----

# The cells of the output, and the lanes they lift.

# ---- The rows of a task and the lifted input gradient ----

# What the input gradient of a task's batch is read from, and the cells of
# the input gradient of the call.
def batch_setup(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>) -> Setup:
  Setup{BatchRows.batch_call(~Holder,~value,batch),BatchRows.batch_model(~Holder,~value,batch),BatchRows.batch_weights(~Holder,~value,batch)}
def cells_of(+geometry: Kernel.Geometry,+outer: U32,+after: U32) -> Cells.Cells:
  Cells.Cells{outer,after,Domain.channel_word(geometry),Tiled.groups(Domain.channel_word(geometry)),Grid.Innermost{}}

# The input gradient of a channel at pixel p of the rows of a task is the
# lifted value at the logical index of its channel at pixel start * width + p
# of the batch.
def pixel_agrees(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)) ==
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),Domain.image_size(BatchRows.batch_geometry(~Holder,~value,batch))),
    U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
  +pixel: Nat,+channel: Nat,
  +channel_inside: {Nat.is_lt(channel,Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch))) == True{} : Bool},
  +pixel_inside: {Nat.is_lt(pixel,Nat.mul(Nat.sub(U32.to_nat(BatchRows.batch_end(~Holder,~value,batch)),U32.to_nat(BatchRows.batch_start(~Holder,~value,batch))),
    Domain.width(BatchRows.batch_geometry(~Holder,~value,batch)))) == True{} : Bool}) ->
  {BatchRows.pixel_value(~Holder,~value,batch,pixel,channel) ==
    input_value(batch_setup(~Holder,~value,batch),U32.from_nat(Representation.logical_index(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after),
      Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))),pixel),channel))) : F32}:
  BatchRows.Batch{+holder,+call,+envelope,+fits,+weights,+low,+high,+capacity,+capacity_ok,+bounded,+layout,+start,+end,+length,+ordered,+rows_inside,+rows_room,+cells_room} = batch
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +setup = {Setup{whole,BandSource.gradient_model(~Holder,~value,holder,Rows.cells_word(whole)),weights} : Setup}
  +cells = cells_of(shape,outer,after)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +cn = U32.to_nat(c)
  +plane = Nat.mul(hn,wn)
  +first = U32.to_nat(start)
  +last = U32.to_nat(end)
  +height_positive = Domain.height_positive(whole)
  +width_positive = Domain.width_positive(whole)
  +plane_positive = Order.product_positive(hn,wn,height_positive,width_positive)
  +q = Nat.div(pixel,wn)
  +x = Nat.mod(pixel,wn)
  +row = Nat.add(first,q)
  +n = Nat.div(row,hn)
  +r = Nat.mod(row,hn)
  +within = Nat.add(Nat.mul(r,wn),x)
  +global = Nat.add(Nat.mul(first,wn),pixel)
  +x_less = DivisionOrder.remainder_small(pixel,wn,width_positive)
  +r_less = DivisionOrder.remainder_small(row,hn,height_positive)
  +within_less = RowMajor.radix(r,hn,x,wn,r_less,x_less)
  +q_less = DivisionOrder.quotient_below(pixel,Nat.sub(last,first),wn,width_positive,pixel_inside)
  +row_less = {%Order.restore_shift(first,last,ordered) : {Nat.is_lt(row,_) == True{} : Bool}
    Order.offset_lt(first,q,Nat.sub(last,first),q_less) : {Nat.is_lt(row,last) == True{} : Bool}}
  +n_less = DivisionOrder.quotient_below(row,images,hn,height_positive,Bounds.nat_trans(row,last,Nat.mul(images,hn),row_less,rows_inside))
  # The pixel of the batch is image n, row r and column x.
  +rows_exact = {%Equal.sym(Nat,Nat.mul(Nat.add(first,q),wn),Nat.add(Nat.mul(first,wn),Nat.mul(q,wn)),RowMajor.distrib(first,q,wn)) : {global == Nat.add(_,x) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,wn),Nat.mul(q,wn)),x),Nat.add(Nat.mul(first,wn),Nat.add(Nat.mul(q,wn),x)),Addition.associative(Nat.mul(first,wn),Nat.mul(q,wn),x)) :
      {global == _ : Nat}
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,wn),value),pixel,Nat.add(Nat.mul(q,wn),x),DivisionOrder.quotient_exact(pixel,wn,width_positive)) :
    {global == Nat.add(Nat.mul(row,wn),x) : Nat}}
  +image_exact = {%Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(n,hn),r),wn),Nat.add(Nat.mul(Nat.mul(n,hn),wn),Nat.mul(r,wn)),RowMajor.distrib(Nat.mul(n,hn),r,wn)) :
      {Nat.add(_,x) == Nat.add(Nat.mul(n,plane),within) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.mul(n,hn),wn),Nat.mul(n,plane),RowMajor.assoc(n,hn,wn)) :
      {Nat.add(Nat.add(_,Nat.mul(r,wn)),x) == Nat.add(Nat.mul(n,plane),within) : Nat}
    Addition.associative(Nat.mul(n,plane),Nat.mul(r,wn),x) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(n,hn),r),wn),x) == Nat.add(Nat.mul(n,plane),within) : Nat}}
  +global_exact = Equal.trans(Nat,global,Nat.add(Nat.mul(row,wn),x),Nat.add(Nat.mul(n,plane),within),rows_exact,
    Equal.trans(Nat,Nat.add(Nat.mul(row,wn),x),Nat.add(Nat.mul(Nat.add(Nat.mul(n,hn),r),wn),x),Nat.add(Nat.mul(n,plane),within),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,wn),x),row,Nat.add(Nat.mul(n,hn),r),DivisionOrder.quotient_exact(row,hn,height_positive)),
      image_exact))
  +global_image = {%Equal.sym(Nat,global,Nat.add(Nat.mul(n,plane),within),global_exact) : {Nat.div(_,plane) == n : Nat}
    Radix.quotient(n,within,plane,plane_positive,within_less) : {Nat.div(global,plane) == n : Nat}}
  +global_within = {%Equal.sym(Nat,global,Nat.add(Nat.mul(n,plane),within),global_exact) : {Nat.mod(_,plane) == within : Nat}
    Radix.remainder(n,within,plane,plane_positive,within_less) : {Nat.mod(global,plane) == within : Nat}}
  # The logical index of the channel there is the index of that point.
  +point = cell_point(shape,images,n,channel,r,x)
  +unfolded = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(global,plane),cn),channel),plane),Nat.mod(global,plane))
  +index_exact = Equal.trans(Nat,Representation.logical_index(cells,global,channel),unfolded,point,
    Equal.cong(Nat,Nat,divisor => Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(global,divisor),cn),channel),divisor),Nat.mod(global,divisor)),
      Representation.after_count(cells),plane,after_exact),
    {%Equal.sym(Nat,Nat.div(global,plane),n,global_image) : {Nat.add(Nat.mul(Nat.add(Nat.mul(_,cn),channel),plane),Nat.mod(global,plane)) == point : Nat}
      %Equal.sym(Nat,Nat.mod(global,plane),within,global_within) : {Nat.add(Nat.mul(Nat.add(Nat.mul(n,cn),channel),plane),_) == point : Nat}
      {==} : {unfolded == point : Nat}})
  +decoded_value = decoded(setup,Machine.capacity(limits),Limits.capacity_fits(limits),values_room,n,channel,r,x,n_less,channel_inside,r_less,x_less)
  Equal.sym(F32,input_value(setup,U32.from_nat(Representation.logical_index(cells,global,channel))),Specification.value(spec(setup,n),channel,r,x),
    Equal.trans(F32,input_value(setup,U32.from_nat(Representation.logical_index(cells,global,channel))),input_value(setup,U32.from_nat(point)),
      Specification.value(spec(setup,n),channel,r,x),
      Equal.cong(Nat,F32,index => input_value(setup,U32.from_nat(index)),Representation.logical_index(cells,global,channel),point,index_exact),
      decoded_value))

# The groups of eight channels are positive.
def groups_positive(+call: Domain.Call) -> {Nat.is_lt(0n,Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n)) == True{} : Bool}:
  +channels = Domain.channels(Domain.geometry(call))
  Equal.trans(Bool,Nat.is_lt(0n,Ceiling.ceiling(channels,8n)),Nat.is_lt(0n,channels),True{},Ceiling.upper_comparison(channels,8n,0n,{==}),Domain.channels_positive(call))

# A cell of the rows of a task names a pixel inside them.
def pixel_below(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+cell: Nat,
  +inside: {Nat.is_lt(cell,BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(cell,BatchRows.batch_groups(~Holder,~value,batch)),Nat.mul(Nat.sub(U32.to_nat(BatchRows.batch_end(~Holder,~value,batch)),
    U32.to_nat(BatchRows.batch_start(~Holder,~value,batch))),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch)))) == True{} : Bool}:
  +rows = Nat.sub(U32.to_nat(BatchRows.batch_end(~Holder,~value,batch)),U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)))
  +width = Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))
  +groups = BatchRows.batch_groups(~Holder,~value,batch)
  DivisionOrder.quotient_below(cell,Nat.mul(rows,width),groups,groups_positive(BatchRows.batch_call(~Holder,~value,batch)),
    {%Equal.sym(Nat,Nat.mul(Nat.mul(rows,width),groups),Nat.mul(rows,Nat.mul(width,groups)),RowMajor.assoc(rows,width,groups)) : {Nat.is_lt(cell,_) == True{} : Bool}
      inside : {Nat.is_lt(cell,Nat.mul(Nat.mul(rows,width),groups)) == True{} : Bool}})

# A lane of a written cell agrees with the lifted input gradient at its
# pixel of the batch.
def lane_valid(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)) ==
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),Domain.image_size(BatchRows.batch_geometry(~Holder,~value,batch))),
    U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
  +cell: Nat,+lane: Nat,+tile: Tile.Tile8,
  +inside: {Nat.is_lt(cell,BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool},
  +written: {PaddedCells.kept(Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch)),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),lane,
      TileLaws.select_lane(lane,tile)) ==
    PaddedCells.kept(Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch)),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),lane,
      BatchRows.pixel_value(~Holder,~value,batch,Nat.div(cell,BatchRows.batch_groups(~Holder,~value,batch)),
        Nat.add(lane,Nat.mul(Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),8n)))) : F32}) ->
  {PaddedCells.kept(Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch)),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),lane,
      TileLaws.select_lane(lane,tile)) ==
    PaddedCells.kept(Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch)),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),lane,
      input_value(batch_setup(~Holder,~value,batch),U32.from_nat(Representation.logical_index(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after),
        Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))),
          Nat.div(cell,BatchRows.batch_groups(~Holder,~value,batch))),
        Nat.add(lane,Nat.mul(Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),8n)))))) : F32}:
  +channels = Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch))
  +groups = BatchRows.batch_groups(~Holder,~value,batch)
  +group = Nat.mod(cell,groups)
  +pixel = Nat.div(cell,groups)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +wanted = input_value(batch_setup(~Holder,~value,batch),U32.from_nat(Representation.logical_index(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after),
    Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))),pixel),channel)))
  Equal.trans(F32,PaddedCells.kept(channels,group,lane,TileLaws.select_lane(lane,tile)),
    PaddedCells.kept(channels,group,lane,BatchRows.pixel_value(~Holder,~value,batch,pixel,channel)),PaddedCells.kept(channels,group,lane,wanted),written,
    PaddedCells.kept_agree(channels,group,lane,BatchRows.pixel_value(~Holder,~value,batch,pixel,channel),wanted,Nat.is_lt(channel,channels),{==},
      below => pixel_agrees(~Holder,~value,batch,outer,after,after_exact,values_room,pixel,channel,below,pixel_below(~Holder,~value,batch,cell,inside))))

# A written cell of the rows of a task agrees, lane by lane, with the lifted
# input gradient at its pixel of the batch.
def cell_valid(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)) ==
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),Domain.image_size(BatchRows.batch_geometry(~Holder,~value,batch))),
    U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
  +tree: Tree.Tree<Tile.Tile8>,+cell: Nat,+inside: {Nat.is_lt(cell,BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool},
  +written: BatchRows.WrittenCell(~Holder,~value,batch,tree,cell)) ->
  PaddedCells.Valid<Domain.channels(BatchRows.batch_geometry(~Holder,~value,batch)),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)),
    Tree.read(Tile.Tile8,tree,U32.from_nat(cell)),
    Representation.cell(~Setup,~input_value,batch_setup(~Holder,~value,batch),cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after),
      Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))),
        Nat.div(cell,BatchRows.batch_groups(~Holder,~value,batch))),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch)))>:
  BatchRows.Batch{+holder,+call,+envelope,+fits,+weights,+low,+high,+capacity,+capacity_ok,+bounded,+layout,+start,+end,+length,+ordered,+rows_inside,+rows_room,+cells_room} = batch
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  PaddedCells.Valid{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = written
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +rebuilt = {BatchRows.Batch{holder,whole,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} :
    BatchRows.Batch<Holder,value>}
  +tile = Tree.read(Tile.Tile8,tree,U32.from_nat(cell))
  PaddedCells.Valid{
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,0n,tile,inside,w0),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,1n,tile,inside,w1),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,2n,tile,inside,w2),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,3n,tile,inside,w3),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,4n,tile,inside,w4),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,5n,tile,inside,w5),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,6n,tile,inside,w6),
    lane_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,cell,7n,tile,inside,w7)}

# ---- The padded cells of a piece ----

# The padding of the cells of a piece: +0 in the lanes past the channels.
def piece_padding(+geometry: Kernel.Geometry) -> TiledMap.Padding:
  TiledMap.Padding{Domain.channel_word(geometry),Tiled.groups(Domain.channel_word(geometry)),0,Grid.Innermost{}}

# Padding keeps the length of a piece.
def length_of_ready(+padding: TiledMap.Padding,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +both: KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})) == length : Nat}>) ->
  {Buffer.content_length(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})) == length : Nat}:
  KernelGemm.Both{+ready,+same} = both
  same

# The cell of the batch at a cell of the rows of a task is the lifted cell
# of its pixel and group.
def lifted_exact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+outer: U32,+after: U32,
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch)))) : Nat},
  +cell: Nat,+inside: {Nat.is_lt(cell,BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),BatchRows.batch_size(~Holder,~value,batch)),cell),
    U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool}) ->
  {Representation.lifted(~Setup,~input_value,Representation.Lifted{batch_setup(~Holder,~value,batch),cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)},
      U32.from_nat(Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),BatchRows.batch_size(~Holder,~value,batch)),cell))) ==
    Representation.cell(~Setup,~input_value,batch_setup(~Holder,~value,batch),cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after),
      Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))),
        Nat.div(cell,BatchRows.batch_groups(~Holder,~value,batch))),Nat.mod(cell,BatchRows.batch_groups(~Holder,~value,batch))) : Tile.Tile8}:
  BatchRows.Batch{+holder,+call,+envelope,+fits,+weights,+low,+high,+capacity,+capacity_ok,+bounded,+layout,+start,+end,+length,+ordered,+rows_inside,+rows_room,+cells_room} = batch
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +rebuilt = {BatchRows.Batch{holder,whole,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} :
    BatchRows.Batch<Holder,value>}
  +setup = batch_setup(~Holder,~value,rebuilt)
  +cells = cells_of(shape,outer,after)
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +cn = U32.to_nat(c)
  +groups = Ceiling.ceiling(cn,8n)
  +counted = U32.to_nat(Grid.groups(c))
  +first = U32.to_nat(start)
  +last = U32.to_nat(end)
  +size = Nat.mul(wn,groups)
  +pixel = Nat.div(cell,groups)
  +group = Nat.mod(cell,groups)
  +global = Nat.add(Nat.mul(first,wn),pixel)
  +number = Nat.add(Nat.mul(first,size),cell)
  +positive = groups_positive(whole)
  +channels_room = Limits.dimension_capacity(limits,cn,Domain.channels_room(whole))
  +groups_exact = BatchRows.groups_number(limits,c,TiledValues.grouped(limits,c,channels_room))
  +number_split = {%Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(first,wn),pixel),groups),Nat.add(Nat.mul(Nat.mul(first,wn),groups),Nat.mul(pixel,groups)),
        RowMajor.distrib(Nat.mul(first,wn),pixel,groups)) : {number == Nat.add(_,group) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.mul(first,wn),groups),Nat.mul(first,size),RowMajor.assoc(first,wn,groups)) :
      {number == Nat.add(Nat.add(_,Nat.mul(pixel,groups)),group) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,size),Nat.mul(pixel,groups)),group),Nat.add(Nat.mul(first,size),Nat.add(Nat.mul(pixel,groups),group)),
        Addition.associative(Nat.mul(first,size),Nat.mul(pixel,groups),group)) : {number == _ : Nat}
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(first,size),value),cell,Nat.add(Nat.mul(pixel,groups),group),DivisionOrder.quotient_exact(cell,groups,positive)) :
    {number == Nat.add(Nat.mul(global,groups),group) : Nat}}
  +exact = Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),number,Nat.add(Nat.mul(global,counted),group),
    FullRange.roundtrip(number,Machine.capacity(limits),room),
    Equal.trans(Nat,number,Nat.add(Nat.mul(global,groups),group),Nat.add(Nat.mul(global,counted),group),number_split,
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(global,value),group),groups,counted,Equal.sym(Nat,counted,groups,groups_exact))))
  +below = pixel_below(~Holder,~value,rebuilt,cell,inside)
  +global_below = {%Order.restore_shift(first,last,ordered) : {Nat.is_lt(global,Nat.mul(_,wn)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.add(first,Nat.sub(last,first)),wn),Nat.add(Nat.mul(first,wn),Nat.mul(Nat.sub(last,first),wn)),RowMajor.distrib(first,Nat.sub(last,first),wn)) :
      {Nat.is_lt(global,_) == True{} : Bool}
    Order.offset_lt(Nat.mul(first,wn),pixel,Nat.mul(Nat.sub(last,first),wn),below) : {Nat.is_lt(global,Nat.mul(last,wn)) == True{} : Bool}}
  +outer_inside = {%Equal.sym(Nat,U32.to_nat(outer),Nat.mul(images,Nat.mul(hn,wn)),outer_exact) : {Nat.is_lt(global,_) == True{} : Bool}
    Bounds.nat_trans(global,Nat.mul(last,wn),Nat.mul(images,Nat.mul(hn,wn)),global_below,
      {%RowMajor.assoc(images,hn,wn) : {Nat.is_le(Nat.mul(last,wn),_) == True{} : Bool}
        RowMajor.mul_le(last,Nat.mul(images,hn),wn,rows_inside) : {Nat.is_le(Nat.mul(last,wn),Nat.mul(images,Nat.mul(hn,wn))) == True{} : Bool}}) :
    {Nat.is_lt(global,U32.to_nat(outer)) == True{} : Bool}}
  +group_inside = {%Equal.sym(Nat,counted,groups,groups_exact) : {Nat.is_lt(group,_) == True{} : Bool}
    DivisionOrder.remainder_small(cell,groups,positive) : {Nat.is_lt(group,counted) == True{} : Bool}}
  Representation.decoded(~Setup,~input_value,setup,cells,global,group,U32.from_nat(number),exact,outer_inside,group_inside)

# A cell of a piece holding a written cell of the rows of a task holds, after
# padding, the lifted input gradient at its cell of the batch.
def padded_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)) ==
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),
    Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),Domain.image_size(BatchRows.batch_geometry(~Holder,~value,batch))),
    U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
  +tree: Tree.Tree<Tile.Tile8>,
  +written: Members.All(Nat,{cell => BatchRows.WrittenCell(~Holder,~value,batch,tree,cell) : Nat -> Data},Ranges.range(BatchRows.batch_cells(~Holder,~value,batch),0n)),
  +first: U32,+count: U32,+offset: Nat,
  +first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),BatchRows.batch_size(~Holder,~value,batch)),offset) : Nat},
  +fits: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool},
  +span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
  +kept: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{kept,length}>,+length_exact: {length == U32.to_nat(count) : Nat},
  +index: Nat,+below: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},
  +read_exact: {Tree.read(Tile.Tile8,kept,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index))) : Tile.Tile8}) ->
  Content.At(~Tile.Tile8,~Representation.Lifted<Setup>,~Representation.lifted(~Setup,~input_value),
    Representation.Lifted{batch_setup(~Holder,~value,batch),cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)},first,
    Buffer.content_tree(~Tile.Tile8,Padding.step(piece_padding(BatchRows.batch_geometry(~Holder,~value,batch)),first,count,Buffer.Content{kept,length})),
    Buffer.content_length(~Tile.Tile8,Padding.step(piece_padding(BatchRows.batch_geometry(~Holder,~value,batch)),first,count,Buffer.Content{kept,length})),index):
  BatchRows.Batch{+holder,+call,+envelope,+fits_word,+weights,+low,+high,+capacity,+capacity_ok,+bounded,+layout,+start,+end,+rows_length,+ordered,+rows_inside,+rows_room,+cells_room} = batch
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +rebuilt = {BatchRows.Batch{holder,whole,envelope,fits_word,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,rows_length,ordered,rows_inside,rows_room,cells_room} :
    BatchRows.Batch<Holder,value>}
  +setup = batch_setup(~Holder,~value,rebuilt)
  +cells = cells_of(shape,outer,after)
  +padding = piece_padding(shape)
  +wn = U32.to_nat(w)
  +cn = U32.to_nat(c)
  +groups = Ceiling.ceiling(cn,8n)
  +counted = U32.to_nat(Grid.groups(c))
  +first_row = U32.to_nat(start)
  +size = Nat.mul(wn,groups)
  +total = BatchRows.batch_cells(~Holder,~value,rebuilt)
  +cell = Nat.add(offset,index)
  +number = Nat.add(U32.to_nat(first),index)
  +cap = Machine.capacity(limits)
  +half = Machine.half_word(limits)
  +positive = groups_positive(whole)
  +channels_room = Limits.dimension_capacity(limits,cn,Domain.channels_room(whole))
  +facts = TiledValues.grouped(limits,c,channels_room)
  +groups_exact = BatchRows.groups_number(limits,c,facts)
  +cell_inside = Bounds.nat_trans(cell,Nat.add(offset,U32.to_nat(count)),total,Order.offset_lt(offset,index,U32.to_nat(count),below),fits)
  +written_cell = Members.all_member(Nat,{at => BatchRows.WrittenCell(~Holder,~value,rebuilt,tree,at) : Nat -> Data},cell,Ranges.range(total,0n),written,
    Members.range_member(total,0n,cell,cell_inside))
  +target = Representation.cell(~Setup,~input_value,setup,cells,Nat.add(Nat.mul(first_row,wn),Nat.div(cell,groups)),Nat.mod(cell,groups))
  +tile = Tree.read(Tile.Tile8,kept,U32.from_nat(index))
  +valid = {%Equal.sym(Tile.Tile8,tile,Tree.read(Tile.Tile8,tree,U32.from_nat(cell)),read_exact) : PaddedCells.Valid<cn,Nat.mod(cell,groups),_,target>
    cell_valid(~Holder,~value,rebuilt,outer,after,after_exact,values_room,tree,cell,cell_inside,written_cell) : PaddedCells.Valid<cn,Nat.mod(cell,groups),tile,target>}
  +number_exact = Equal.trans(Nat,number,Nat.add(Nat.add(Nat.mul(first_row,size),offset),index),Nat.add(Nat.mul(first_row,size),cell),
    Equal.cong(Nat,Nat,value => Nat.add(value,index),U32.to_nat(first),Nat.add(Nat.mul(first_row,size),offset),first_exact),
    Addition.associative(Nat.mul(first_row,size),offset,index))
  +group_exact = Equal.trans(Nat,Nat.mod(number,groups),Nat.mod(Nat.add(Nat.mul(first_row,size),cell),groups),Nat.mod(cell,groups),
    Equal.cong(Nat,Nat,value => Nat.mod(value,groups),number,Nat.add(Nat.mul(first_row,size),cell),number_exact),
    {%RowMajor.assoc(first_row,wn,groups) : {Nat.mod(Nat.add(_,cell),groups) == Nat.mod(cell,groups) : Nat}
      DivisionOrder.remainder_shift(cell,groups,Nat.mul(first_row,wn),positive) : {Nat.mod(Nat.add(Nat.mul(first_row,size),cell),groups) == Nat.mod(cell,groups) : Nat}})
  +last_exact = Equal.cong(Nat,Bool,value => Nat.is_eq(value,Nat.sub(counted,1n)),Nat.mod(number,counted),Nat.mod(cell,groups),
    Equal.trans(Nat,Nat.mod(number,counted),Nat.mod(number,groups),Nat.mod(cell,groups),
      Equal.cong(Nat,Nat,divisor => Nat.mod(number,divisor),counted,groups,groups_exact),group_exact))
  +group_inside = {%Equal.sym(Nat,counted,groups,groups_exact) : {Nat.is_lt(Nat.mod(cell,groups),_) == True{} : Bool}
    DivisionOrder.remainder_small(cell,groups,positive) : {Nat.is_lt(Nat.mod(cell,groups),counted) == True{} : Bool}}
  +placed = PaddedCells.padded_group(limits,c,0,Grid.Innermost{},Nat.mod(cell,groups),number,tile,target,facts,group_inside,last_exact,valid)
  +half_room = Bounds.as_nat_le(cap,half,Limits.capacity_within_half(limits))
  +span_half = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),U32.to_nat(half),span_room,half_room)
  +groups_room = Order.transitive(counted,cn,U32.to_nat(cap),TiledValues.fewer_of(limits,c,facts),channels_room)
  +unaligned = {KernelGemm.Both{{%Equal.sym(Nat,counted,groups,groups_exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,counted) == True{} : Bool}},
      KernelGemm.Both{Limits.double_bound(limits,counted,groups_room),
        KernelGemm.Both{{%Equal.sym(Nat,Nat.mul(counted,0n),0n,RowMajor.multiply_zero_right(counted)) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
            Order.zero_le(U32.to_nat(half)) : {Nat.is_le(Nat.mul(counted,0n),U32.to_nat(half)) == True{} : Bool}},span_half}}} :
    Padding.Unaligned(half,counted,0n,U32.to_nat(first),U32.to_nat(count))}
  +fitted = PaddedCells.fits_when(U32.is_zero((c % 8 : U32)),half,counted,0n,U32.to_nat(first),U32.to_nat(count),unaligned)
  +bounded = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
    Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(half),Order.right_le(U32.to_nat(first),U32.to_nat(count)),span_half) :
    {Nat.is_le(length,U32.to_nat(half)) == True{} : Bool}}
  +counted_room = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
    Order.reflexive(U32.to_nat(count)) : {Nat.is_le(U32.to_nat(count),length) == True{} : Bool}}
  +read_padded = Padding.read(half,Limits.half_word_fits(limits),padding,first,count,kept,length,ready,fitted,bounded,counted_room,index,below)
  +length_kept = length_of_ready(padding,first,count,kept,length,Padding.kept_ready(padding,first,count,kept,length,ready))
  +number_room = {%number_exact : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(number,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),
      Order.offset_le(U32.to_nat(first),index,U32.to_nat(count),Bounds.strict_le(index,U32.to_nat(count),below)),span_room) :
    {Nat.is_le(Nat.add(Nat.mul(first_row,size),cell),U32.to_nat(cap)) == True{} : Bool}}
  +lifted_number = Equal.trans(Tile.Tile8,Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(number)),
    Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(Nat.add(Nat.mul(first_row,size),cell))),target,
    Equal.cong(Nat,Tile.Tile8,at => Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(at)),number,
      Nat.add(Nat.mul(first_row,size),cell),number_exact),
    lifted_exact(~Holder,~value,rebuilt,outer,after,outer_exact,cell,cell_inside,number_room))
  +output = Padding.step(padding,first,count,Buffer.Content{kept,length})
  KernelGemm.Both{
    {%Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,output),length,length_kept) : {Nat.is_lt(index,_) == True{} : Bool}
      %Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_lt(index,_) == True{} : Bool}
      below : {Nat.is_lt(index,Buffer.content_length(~Tile.Tile8,output)) == True{} : Bool}},
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,output),U32.from_nat(index)),Padding.cell(padding,number,tile),
      Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(number)),read_padded,
      Equal.trans(Tile.Tile8,Padding.cell(padding,number,tile),target,Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(number)),
        placed,Equal.sym(Tile.Tile8,Representation.lifted(~Setup,~input_value,Representation.Lifted{setup,cells},U32.from_nat(number)),target,lifted_number)))}

# ---- The cells a piece keeps of the rows ----

# The cells a piece keeps of the rows of its task: the rows themselves when
# the piece holds whole rows, otherwise count cells copied from an offset.
type Kept<-tree: Tree.Tree<Tile.Tile8>,-offset: Nat,-count: U32,-kept: Tree.Tree<Tile.Tile8>> is Data:
  WholeKept{same: {kept == tree : Tree.Tree<Tile.Tile8>},zero: {offset == 0n : Nat}}
  CutKept{word: U32,word_exact: {U32.to_nat(word) == offset : Nat},
    depth_ok: {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tiled.zero_tile())),2147483648) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tiled.zero_tile())))) == True{} : Bool},
    same: {kept == Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat(count),tree,0,word,0),Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tiled.zero_tile())) :
      Tree.Tree<Tile.Tile8>}}

# A kept cell is the cell of the rows at the offset.
def kept_read(+tree: Tree.Tree<Tile.Tile8>,+offset: Nat,+count: U32,+kept: Tree.Tree<Tile.Tile8>,+fact: Kept<tree,offset,count,kept>,
  +index: Nat,+below: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,kept,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index))) : Tile.Tile8}:
  match fact:
    case WholeKept{+same,+zero}:
      %Equal.sym(Tree.Tree<Tile.Tile8>,kept,tree,same) :
        {Tree.read(Tile.Tile8,_,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index))) : Tile.Tile8}
      %Equal.sym(Nat,offset,0n,zero) : {Tree.read(Tile.Tile8,tree,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(_,index))) : Tile.Tile8}
      {==}
    case CutKept{+word,+word_exact,+depth_ok,+capacity_ok,+room,+same}:
      +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tiled.zero_tile())
      +copied = Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat(count),tree,0,word,0),fresh)
      +address = Equal.trans(U32,(word + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(word),U32.to_nat(U32.from_nat(index)))),
        U32.from_nat(Nat.add(offset,index)),
        Arithmetic.native_add(word,U32.from_nat(index)),
        Equal.cong(Nat & Nat,U32,pair => U32.from_nat(Nat.add(Pair.fst(Nat,Nat,pair),Pair.snd(Nat,Nat,pair))),(U32.to_nat(word),U32.to_nat(U32.from_nat(index))),(offset,index),
          Equal.trans(Nat & Nat,(U32.to_nat(word),U32.to_nat(U32.from_nat(index))),(offset,U32.to_nat(U32.from_nat(index))),(offset,index),
            Equal.cong(Nat,Nat & Nat,first => (first,U32.to_nat(U32.from_nat(index))),U32.to_nat(word),offset,word_exact),
            Equal.cong(Nat,Nat & Nat,second => (offset,second),U32.to_nat(U32.from_nat(index)),index,
              FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),below))))))
      %Equal.sym(Tree.Tree<Tile.Tile8>,kept,copied,same) :
        {Tree.read(Tile.Tile8,_,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index))) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,copied,U32.from_nat(index)),Tree.read(Tile.Tile8,tree,(word + U32.from_nat(index) : U32)),
        Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(offset,index))),
        CopyRead.fresh_read(~Tile.Tile8,U32.to_nat(count),index,tree,word,Tensors.depth_for(count),Tiled.zero_tile(),depth_ok,capacity_ok,room,below),
        Equal.cong(U32,Tile.Tile8,at => Tree.read(Tile.Tile8,tree,at),(word + U32.from_nat(index) : U32),U32.from_nat(Nat.add(offset,index)),address))

# ---- Every cell of a piece ----

# A piece of the rows of a task and what each of its cells reads.
type PieceScope<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  PieceScope{batch: BatchRows.Batch<Holder,value>,outer: U32,after: U32,
    after_exact: {Representation.after_count(cells_of(BatchRows.batch_geometry(~Holder,~value,batch),outer,after)) ==
      Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch))) : Nat},
    outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),
      Nat.mul(Domain.height(BatchRows.batch_geometry(~Holder,~value,batch)),Domain.width(BatchRows.batch_geometry(~Holder,~value,batch)))) : Nat},
    values_room: {Nat.is_le(Nat.mul(Domain.batch(BatchRows.batch_call(~Holder,~value,batch)),Domain.image_size(BatchRows.batch_geometry(~Holder,~value,batch))),
      U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
    tree: Tree.Tree<Tile.Tile8>,
    written: Members.All(Nat,{cell => BatchRows.WrittenCell(~Holder,~value,batch,tree,cell) : Nat -> Data},Ranges.range(BatchRows.batch_cells(~Holder,~value,batch),0n)),
    first: U32,count: U32,offset: Nat,
    first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(U32.to_nat(BatchRows.batch_start(~Holder,~value,batch)),BatchRows.batch_size(~Holder,~value,batch)),offset) : Nat},
    fits: {Nat.is_le(Nat.add(offset,U32.to_nat(count)),BatchRows.batch_cells(~Holder,~value,batch)) == True{} : Bool},
    span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(Domain.limits(BatchRows.batch_call(~Holder,~value,batch))))) == True{} : Bool},
    kept: Tree.Tree<Tile.Tile8>,length: Nat,ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{kept,length}>,length_exact: {length == U32.to_nat(count) : Nat},
    kept_fact: Kept<tree,offset,count,kept>}
def scope_batch(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> BatchRows.Batch<Holder,value>:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  batch
def scope_outer(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> U32:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  outer
def scope_after(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> U32:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  after
def scope_first(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> U32:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  first
def scope_count(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> U32:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  count
def scope_kept(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> Tree.Tree<Tile.Tile8>:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  kept
def scope_length(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> Nat:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  length
def scope_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: PieceScope<Holder,value>) -> Nat:
  PieceScope{batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,kept_fact} = scope
  U32.to_nat(count)

# The padded cells of a piece and what a cell of them holds.
def scope_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: PieceScope<Holder,value>) -> Buffer.Content<Tile.Tile8>:
  Padding.step(piece_padding(BatchRows.batch_geometry(~Holder,~value,scope_batch(~Holder,~value,scope))),scope_first(~Holder,~value,scope),scope_count(~Holder,~value,scope),
    Buffer.Content{scope_kept(~Holder,~value,scope),scope_length(~Holder,~value,scope)})
def PieceAt(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: PieceScope<Holder,value>,+index: Nat) -> Data:
  +batch = scope_batch(~Holder,~value,scope)
  Content.At(~Tile.Tile8,~Representation.Lifted<Setup>,~Representation.lifted(~Setup,~input_value),
    Representation.Lifted{batch_setup(~Holder,~value,batch),cells_of(BatchRows.batch_geometry(~Holder,~value,batch),scope_outer(~Holder,~value,scope),scope_after(~Holder,~value,scope))},
    scope_first(~Holder,~value,scope),Buffer.content_tree(~Tile.Tile8,scope_output(~Holder,~value,scope)),Buffer.content_length(~Tile.Tile8,scope_output(~Holder,~value,scope)),index)

def piece_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: PieceScope<Holder,value>,+index: Nat,
  +below: {Nat.is_lt(index,scope_cells(~Holder,~value,scope)) == True{} : Bool}) -> PieceAt(~Holder,~value,scope,index):
  PieceScope{+batch,+outer,+after,+after_exact,+outer_exact,+values_room,+tree,+written,+first,+count,+offset,+first_exact,+fits,+span_room,+kept,+length,+ready,
    +length_exact,+kept_fact} = scope
  padded_cell(~Holder,~value,batch,outer,after,after_exact,outer_exact,values_room,tree,written,first,count,offset,first_exact,fits,span_room,kept,length,ready,length_exact,
    index,below,kept_read(tree,offset,count,kept,kept_fact,index,below))

# Every cell of a piece holds the lifted input gradient at its cell.
def piece_leaf(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: PieceScope<Holder,value>) ->
  Members.All(Nat,{index => PieceAt(~Holder,~value,scope,index) : Nat -> Data},Ranges.range(scope_cells(~Holder,~value,scope),0n)):
  Members.range_all(~PieceScope<Holder,value>,~PieceAt(~Holder,~value),~scope_cells(~Holder,~value),~piece_at(~Holder,~value),
    scope_cells(~Holder,~value,scope),0n,scope,Order.reflexive(scope_cells(~Holder,~value,scope)))

# ---- The runtime of a piece ----

# Cells copied from an offset of a buffer into a target hold the copy of
# them over the target.
def cut_contents(+offset: U32,+count: U32,+source: Buffer.Model<Tile.Tile8>,+target: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,target)))) ==
    Buffer.Content{Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat(count),Buffer.tree(~Tile.Tile8,source),0,offset,0),Buffer.tree(~Tile.Tile8,target)),
      Buffer.length(~Tile.Tile8,target)} : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = source
  Buffer.Model{+output,+output_length,+output_certificate} = target
  Equal.trans(Buffer.Content<Tile.Tile8>,
    Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,Storage.Buffer{Tree.pack(Tile.Tile8,tree),length,certificate},Storage.Buffer{Tree.pack(Tile.Tile8,output),output_length,output_certificate}))),
    Held.owners_output(~Tile.Tile8,CopyLaws.observe(~Tile.Tile8,
      Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,Storage.Buffer{Tree.pack(Tile.Tile8,tree),length,certificate},Storage.Buffer{Tree.pack(Tile.Tile8,output),output_length,output_certificate}))),
    Buffer.Content{Writes.pure(Tile.Tile8,Contiguous.writes(~Tile.Tile8,U32.to_nat(count),tree,0,offset,0),output),output_length},
    Held.output_contents(~Tile.Tile8,
      Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,Storage.Buffer{Tree.pack(Tile.Tile8,tree),length,certificate},Storage.Buffer{Tree.pack(Tile.Tile8,output),output_length,output_certificate})),
    Equal.cong(CopyLaws.Owners<Tile.Tile8>,Buffer.Content<Tile.Tile8>,owners => Held.owners_output(~Tile.Tile8,owners),
      CopyLaws.observe(~Tile.Tile8,
        Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,Storage.Buffer{Tree.pack(Tile.Tile8,tree),length,certificate},Storage.Buffer{Tree.pack(Tile.Tile8,output),output_length,output_certificate})),
      CopyLaws.model(~Tile.Tile8,count,offset,0,CopyLaws.Owners{tree,length,output,output_length}),
      CopyLaws.packed(~Tile.Tile8,CopyLaws.Call{count,offset,0,length,output_length},[Views.Axis{count,1}],tree,output,certificate,output_certificate)))

# No rows lie between a row and itself.
def same_start(+value: Nat) -> {Nat.sub(value,value) == 0n : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +previous: same_start(previous)

# Every cell of a fresh tree from a base on holds its fill.
def fresh_zeros(remaining: Nat,+position: Nat,+depth: Nat,+base: Nat) ->
  Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled((0.0 + 0.0 : F32))),U32.from_nat(Nat.add(base,cell))) ==
    Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{StorageRead.read_fresh(~Tile.Tile8,depth,Tile.filled((0.0 + 0.0 : F32)),U32.from_nat(Nat.add(base,position))),fresh_zeros(rest,1n+position,depth,base)}

# A task before its first step: no rows written, a fresh output of +0.
def started(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: BatchRows.Batch<Holder,value>,+holding: Held.Holding<Tile.Tile8>,
  +tiled: Held.Tiled(~Tile.Tile8,holding,BatchRows.batch_low(~Holder,~value,batch),BatchRows.batch_high(~Holder,~value,batch)),
  +values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,BatchRows.batch_holder(~Holder,~value,batch),holding),
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == BatchRows.batch_layout(~Holder,~value,batch) : Held.Spans},
  +tiles: Buffer.Model<Tile.Tile8>,
  +tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(BatchRows.batch_call(~Holder,~value,batch),BatchRows.batch_weights(~Holder,~value,batch))) :
    Tree.Tree<Tile.Tile8>},
  +rows: U32,+rows_exact: {U32.to_nat(rows) == BatchRows.batch_cells(~Holder,~value,batch) : Nat},
  +length_exact: {U32.to_nat(rows) == BatchRows.batch_length(~Holder,~value,batch) : Nat},
  +depth_ok: {Nat.is_le(Tensors.depth_for(rows),30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,Tensors.depth_for(rows),Tile.filled((0.0 + 0.0 : F32)))))) == True{} : Bool}) ->
  BatchRows.Reached<Holder,value,batch,BatchRows.batch_start(~Holder,~value,batch),0,
    Kernel.Rows{Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Tile.filled(0.0)),Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,Tile.filled((0.0 + 0.0 : F32)),rows)}>:
  +fill = Tile.filled((0.0 + 0.0 : F32))
  +output = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,rows))
  +fresh = Tree.fresh(Tile.Tile8,Tensors.depth_for(rows),fill)
  +tree_exact = AllocationLaws.buffer_tree(~Tile.Tile8,fill,rows)
  +start = U32.to_nat(BatchRows.batch_start(~Holder,~value,batch))
  +last = U32.to_nat(BatchRows.batch_end(~Holder,~value,batch))
  +size = BatchRows.batch_size(~Holder,~value,batch)
  +depth_exact = Shape.unique_depth(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),Buffer.depth(~Tile.Tile8,output),Tensors.depth_for(rows),Buffer.balanced(~Tile.Tile8,output),
    {%Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),fresh,tree_exact) : Shape.Balanced(Tile.Tile8,Tensors.depth_for(rows),_)
      Shape.fresh(Tile.Tile8,Tensors.depth_for(rows),fill) : Shape.Balanced(Tile.Tile8,Tensors.depth_for(rows),Buffer.tree(~Tile.Tile8,output))})
  +output_depth = {%Equal.sym(Nat,Buffer.depth(~Tile.Tile8,output),Tensors.depth_for(rows),depth_exact) : {Nat.is_le(_,30n) == True{} : Bool}
    depth_ok : {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool}}
  +output_room = {%rows_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),fresh,tree_exact) : {Nat.is_le(U32.to_nat(rows),U32.to_nat(Tree.capacity(Tile.Tile8,_))) == True{} : Bool}
    room : {Nat.is_le(BatchRows.batch_cells(~Holder,~value,batch),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool}}
  +output_length = Equal.trans(Nat,Buffer.length(~Tile.Tile8,output),U32.to_nat(rows),BatchRows.batch_length(~Holder,~value,batch),
    AllocationLaws.buffer_length(~Tile.Tile8,fill,rows),length_exact)
  +target_exact = {%Equal.sym(Nat,Nat.sub(start,start),0n,same_start(start)) : {U32.to_nat(0) == Nat.mul(_,size) : Nat}
    {==} : {U32.to_nat(0) == Nat.mul(Nat.sub(start,start),size) : Nat}}
  +written = {%Equal.sym(Nat,Nat.sub(start,start),0n,same_start(start)) :
      Members.All(Nat,{cell => BatchRows.WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},Ranges.range(Nat.mul(_,size),0n))
    Unit{} :
    Members.All(Nat,{cell => BatchRows.WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(start,start),size),0n))}
  +zero = {%Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output),fresh,tree_exact) :
      Members.All(Nat,{cell => {Tree.read(Tile.Tile8,_,U32.from_nat(Nat.add(U32.to_nat(0),cell))) == fill : Tile.Tile8} : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(last,start),size),0n))
    fresh_zeros(Nat.mul(Nat.sub(last,start),size),0n,Tensors.depth_for(rows),U32.to_nat(0)) :
    Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(U32.to_nat(0),cell))) == fill : Tile.Tile8} : Nat -> Data},
      Ranges.range(Nat.mul(Nat.sub(last,start),size),0n))}
  +progress = {BatchRows.Progress{holding,tiled,values,ordered,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0)),Held.fresh_spare_room(~Tile.Tile8,Tile.filled(0.0)),tiles,tiles_exact,output,output_depth,output_room,output_length,
      Order.reflexive(start),BatchRows.batch_ordered(~Holder,~value,batch),target_exact,written,zero} :
    BatchRows.Progress<Holder,value,batch,BatchRows.batch_start(~Holder,~value,batch),0>}
  BatchRows.Reached{progress,
    Equal.trans(Kernel.Rows,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Tile.filled(0.0)),Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,fill,rows)},
      Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,fill,rows)},
      Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)},
      Equal.cong(Move.Spare<Tile.Tile8>,Kernel.Rows,spare => Kernel.Rows{Held.pack(~Tile.Tile8,holding),spare,Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,fill,rows)},
        Move.spare(~Tile.Tile8,Tile.filled(0.0)),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Held.spare_packed(~Tile.Tile8,Tile.filled(0.0))),
      Equal.cong(Storage.Buffer<Tile.Tile8>,Kernel.Rows,buffer => Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,Held.fresh_spare(~Tile.Tile8,Tile.filled(0.0))),Buffer.pack(~Tile.Tile8,tiles),buffer},
        Tensors.buffer(~Tile.Tile8,fill,rows),Buffer.pack(~Tile.Tile8,output),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,fill,rows),Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,rows)))))}

# ---- The rows of a piece ----

# The runtime rounds the rows of a piece up by adding a row less one cell.
def rounded_up(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.div(Nat.sub(Nat.add(value,divisor),1n),divisor) == Ceiling.ceiling(value,divisor) : Nat}:
  match value:
    case 0n: Ceiling.shifted_formula(0n,divisor,positive)
    case 1n+ +previous: Ceiling.added_formula(1n+previous,divisor,{==},positive)

# The cells of a row fit the capacity.
def row_room(+call: Domain.Call) -> {Nat.is_le(BatchRows.row_size(call),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  +channels = Domain.channels(geometry)
  +width = Domain.width(geometry)
  +groups = Ceiling.ceiling(channels,8n)
  +room = U32.to_nat(Machine.capacity(Domain.limits(call)))
  +plane = Nat.mul(Domain.height(geometry),width)
  +fewer = RowMajor.mul_le(groups,channels,width,Ceiling.bounded(channels,8n,{==}))
  +taller = RowMajor.mul_le(width,plane,channels,DivisionOrder.le_multiple(Domain.height(geometry),width,Domain.height_positive(call)))
  %RowMajor.multiply_commutative(groups,width) : {Nat.is_le(_,room) == True{} : Bool}
  Order.transitive(Nat.mul(groups,width),Nat.mul(channels,width),room,fewer,
    {%RowMajor.multiply_commutative(width,channels) : {Nat.is_le(_,room) == True{} : Bool}
      Order.transitive(Nat.mul(width,channels),Nat.mul(plane,channels),room,taller,
        {%RowMajor.multiply_commutative(channels,plane) : {Nat.is_le(_,room) == True{} : Bool}
          Domain.image_room(call) : {Nat.is_le(Nat.mul(plane,channels),room) == True{} : Bool}}) :
      {Nat.is_le(Nat.mul(channels,width),room) == True{} : Bool}})

# The rows of the batch that hold a piece, the cell of them the piece starts
# at and the words the runtime computes them with.
def piece_start(+call: Domain.Call,+first: U32) -> U32:
  Kernel.first_row(Domain.geometry(call),first)
def piece_end(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  Kernel.end_row(Domain.geometry(call),first,count)
def piece_offset(+call: Domain.Call,+first: U32) -> U32:
  (first - piece_start(call,first) * Kernel.row_cells(Domain.geometry(call)) : U32)
def piece_rows_word(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  ((piece_end(call,first,count) - piece_start(call,first)) * Kernel.row_cells(Domain.geometry(call)) : U32)
def piece_span(+call: Domain.Call,+first: U32,+count: U32) -> Nat:
  Nat.mul(Nat.sub(U32.to_nat(piece_end(call,first,count)),U32.to_nat(piece_start(call,first))),BatchRows.row_size(call))

# What the words of the rows of a piece are.
type PieceRows<-call: Domain.Call,-first: U32,-count: U32> is Data:
  PieceRows{size_exact: {U32.to_nat(Kernel.row_cells(Domain.geometry(call))) == BatchRows.row_size(call) : Nat},
    size_positive: {Nat.is_lt(0n,BatchRows.row_size(call)) == True{} : Bool},
    end_exact: {U32.to_nat(piece_end(call,first,count)) == Ceiling.ceiling(Nat.add(U32.to_nat(first),U32.to_nat(count)),BatchRows.row_size(call)) : Nat},
    ordered: {Nat.is_le(U32.to_nat(piece_start(call,first)),U32.to_nat(piece_end(call,first,count))) == True{} : Bool},
    rows_inside: {Nat.is_le(U32.to_nat(piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
    rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
    cells_room: {Nat.is_le(piece_span(call,first,count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
    rows_exact: {U32.to_nat(piece_rows_word(call,first,count)) == piece_span(call,first,count) : Nat},
    offset_exact: {U32.to_nat(piece_offset(call,first)) == Nat.mod(U32.to_nat(first),BatchRows.row_size(call)) : Nat},
    first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(U32.to_nat(piece_start(call,first)),BatchRows.row_size(call)),U32.to_nat(piece_offset(call,first))) : Nat},
    fits: {Nat.is_le(Nat.add(U32.to_nat(piece_offset(call,first)),U32.to_nat(count)),piece_span(call,first,count)) == True{} : Bool},
    count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
    span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
    first_rest: {U32.to_nat((first % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) == Nat.mod(U32.to_nat(first),BatchRows.row_size(call)) : Nat},
    count_rest: {U32.to_nat((count % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) == Nat.mod(U32.to_nat(count),BatchRows.row_size(call)) : Nat},
    start_exact: {U32.to_nat(piece_start(call,first)) == Nat.div(U32.to_nat(first),BatchRows.row_size(call)) : Nat}}
def rows_ordered(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(U32.to_nat(piece_start(call,first)),U32.to_nat(piece_end(call,first,count))) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  ordered
def rows_inside_of(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(U32.to_nat(piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  rows_inside
def rows_room_of(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  rows_room
def rows_cells_room(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(piece_span(call,first,count),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  cells_room
def rows_exact_of(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {U32.to_nat(piece_rows_word(call,first,count)) == piece_span(call,first,count) : Nat}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  rows_exact
def rows_first_exact(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {U32.to_nat(first) == Nat.add(Nat.mul(U32.to_nat(piece_start(call,first)),BatchRows.row_size(call)),U32.to_nat(piece_offset(call,first))) : Nat}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  first_exact
def rows_fits(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(piece_offset(call,first)),U32.to_nat(count)),piece_span(call,first,count)) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  fits
def rows_span_room(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  span_room

def rows_start_exact(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {U32.to_nat(piece_start(call,first)) == Nat.div(U32.to_nat(first),BatchRows.row_size(call)) : Nat}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  start_exact
def rows_end_exact(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {U32.to_nat(piece_end(call,first,count)) == Ceiling.ceiling(Nat.add(U32.to_nat(first),U32.to_nat(count)),BatchRows.row_size(call)) : Nat}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  end_exact
def rows_size_positive(-call: Domain.Call,-first: U32,-count: U32,rows: PieceRows<call,first,count>) ->
  {Nat.is_lt(0n,BatchRows.row_size(call)) == True{} : Bool}:
  PieceRows{size_exact,size_positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,span_room,first_rest,count_rest,start_exact} = rows
  size_positive

# Cells that are not empty have rows.
def rows_nonempty(+call: Domain.Call,+first: U32,+count: U32,+rows: PieceRows<call,first,count>,
  +less: {Nat.is_lt(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(piece_start(call,first)),U32.to_nat(piece_end(call,first,count))) == True{} : Bool}:
  +size = BatchRows.row_size(call)
  +positive = rows_size_positive(call,first,count,rows)
  +f = U32.to_nat(first)
  +stop = Nat.add(f,U32.to_nat(count))
  +image = Nat.div(f,size)
  %Equal.sym(Nat,U32.to_nat(piece_start(call,first)),image,rows_start_exact(call,first,count,rows)) :
    {Nat.is_lt(_,U32.to_nat(piece_end(call,first,count))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(piece_end(call,first,count)),Ceiling.ceiling(stop,size),rows_end_exact(call,first,count,rows)) :
    {Nat.is_lt(image,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(image,Ceiling.ceiling(stop,size)),Nat.is_lt(Nat.mul(image,size),stop),Ceiling.upper_comparison(stop,size,image,positive)) :
    {_ == True{} : Bool}
  Order.le_lt_trans(Nat.mul(image,size),f,stop,DivisionOrder.product_below(f,size,positive),less)
# The rows of the cells [first, first + count) of a batch of images, whose
# cells fit the capacity.
def piece_rows(+call: Domain.Call,+first: U32,+count: U32,
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),
    U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) -> PieceRows<call,first,count>:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +cap = Machine.capacity(limits)
  +room = U32.to_nat(cap)
  +half = Machine.half_word(limits)
  +half_fits = Limits.half_word_fits(limits)
  +size = BatchRows.row_size(whole)
  +positive = Order.product_positive(U32.to_nat(w),Ceiling.ceiling(U32.to_nat(c),8n),Domain.width_positive(whole),groups_positive(whole))
  +one = Order.positive_one(size,positive)
  +size_room = row_room(whole)
  +first_n = U32.to_nat(first)
  +count_n = U32.to_nat(count)
  +value = Nat.add(first_n,count_n)
  +rows = Nat.mul(images,U32.to_nat(h))
  +total = Nat.mul(rows,size)
  +value_room = Order.transitive(value,total,room,inside,total_room)
  +first_room = Order.transitive(first_n,value,room,Order.le_add(first_n,count_n),value_room)
  +count_room = Order.transitive(count_n,value,room,Order.right_le(first_n,count_n),value_room)
  +channels_room = Limits.dimension_capacity(limits,U32.to_nat(c),Domain.channels_room(whole))
  +groups = TiledValues.grouped(limits,c,channels_room)
  +groups_exact = BatchRows.groups_number(limits,c,groups)
  +size_word = Kernel.row_cells(shape)
  +size_exact = Equal.trans(Nat,U32.to_nat((w * Grid.groups(c) : U32)),Nat.mul(U32.to_nat(w),U32.to_nat(Grid.groups(c))),size,
    FullRange.multiply(w,Grid.groups(c),cap,
      {%Equal.sym(Nat,U32.to_nat(Grid.groups(c)),Ceiling.ceiling(U32.to_nat(c),8n),groups_exact) : {Nat.is_le(Nat.mul(U32.to_nat(w),_),room) == True{} : Bool}
        size_room : {Nat.is_le(Nat.mul(U32.to_nat(w),U32.to_nat(Grid.groups(c))),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.mul(U32.to_nat(w),number),U32.to_nat(Grid.groups(c)),Ceiling.ceiling(U32.to_nat(c),8n),groups_exact))
  +bounded_word = U32.max(size_word,1)
  +bounded_exact = Equal.trans(Nat,U32.to_nat(bounded_word),Nat.max(U32.to_nat(size_word),1n),size,Extrema.maximum(size_word,1),
    Equal.trans(Nat,Nat.max(U32.to_nat(size_word),1n),Nat.max(size,1n),size,
      Equal.cong(Nat,Nat,number => Nat.max(number,1n),U32.to_nat(size_word),size,size_exact),
      Equal.trans(Nat,Nat.max(size,1n),Bool.pick(Nat,Nat.is_lt(size,1n),1n,size),size,NatExtrema.maximum_choice(size,1n),
        Equal.cong(Bool,Nat,less => Bool.pick(Nat,less,1n,size),Nat.is_lt(size,1n),False{},Order.not_below(1n,size,one)))))
  +bounded_room = {%Equal.sym(Nat,U32.to_nat(bounded_word),size,bounded_exact) : {Nat.is_le(_,room) == True{} : Bool}
    size_room : {Nat.is_le(U32.to_nat(bounded_word),room) == True{} : Bool}}
  +bounded_positive = {%Equal.sym(Nat,U32.to_nat(bounded_word),size,bounded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    positive : {Nat.is_lt(0n,U32.to_nat(bounded_word)) == True{} : Bool}}
  +doubled = Limits.double_bound(limits,U32.to_nat(bounded_word),bounded_room)
  +start_word = (first / bounded_word : U32)
  +start = Nat.div(first_n,size)
  +start_exact = Equal.trans(Nat,U32.to_nat(start_word),Nat.div(first_n,U32.to_nat(bounded_word)),start,
    NativeDivision.divide(first,bounded_word,half,half_fits,doubled,bounded_positive),
    Equal.cong(Nat,Nat,divisor => Nat.div(first_n,divisor),U32.to_nat(bounded_word),size,bounded_exact))
  +first_rest = Equal.trans(Nat,U32.to_nat((first % bounded_word : U32)),Nat.mod(first_n,U32.to_nat(bounded_word)),Nat.mod(first_n,size),
    NativeDivision.modulo(first,bounded_word,half,half_fits,doubled,bounded_positive),
    Equal.cong(Nat,Nat,divisor => Nat.mod(first_n,divisor),U32.to_nat(bounded_word),size,bounded_exact))
  +count_rest = Equal.trans(Nat,U32.to_nat((count % bounded_word : U32)),Nat.mod(count_n,U32.to_nat(bounded_word)),Nat.mod(count_n,size),
    NativeDivision.modulo(count,bounded_word,half,half_fits,doubled,bounded_positive),
    Equal.cong(Nat,Nat,divisor => Nat.mod(count_n,divisor),U32.to_nat(bounded_word),size,bounded_exact))
  +sum_word = (first + count : U32)
  +sum_exact = FullRange.add(first,count,cap,value_room)
  +half_exact = Equal.trans(Nat,U32.to_nat(half),Nat.double(room),Nat.add(room,room),Limits.doubled_capacity(limits),
    Equal.sym(Nat,Nat.add(room,room),Nat.double(room),Addition.double(room)))
  +grown = Nat.add(value,size)
  +grown_room = {%Equal.sym(Nat,U32.to_nat(half),Nat.add(room,room),half_exact) : {Nat.is_le(grown,_) == True{} : Bool}
    Encoded.sum_bounds(value,size,room,room,value_room,size_room) : {Nat.is_le(grown,U32.to_nat(half)) == True{} : Bool}}
  +grown_word = (sum_word + bounded_word : U32)
  +grown_exact = Equal.trans(Nat,U32.to_nat(grown_word),Nat.add(U32.to_nat(sum_word),U32.to_nat(bounded_word)),grown,
    FullRange.add(sum_word,bounded_word,half,
      {%Equal.sym(Nat,U32.to_nat(sum_word),value,sum_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(bounded_word)),U32.to_nat(half)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(bounded_word),size,bounded_exact) : {Nat.is_le(Nat.add(value,_),U32.to_nat(half)) == True{} : Bool}
        grown_room : {Nat.is_le(Nat.add(U32.to_nat(sum_word),U32.to_nat(bounded_word)),U32.to_nat(half)) == True{} : Bool}}),
    {%Equal.sym(Nat,U32.to_nat(sum_word),value,sum_exact) : {Nat.add(_,U32.to_nat(bounded_word)) == grown : Nat}
      %Equal.sym(Nat,U32.to_nat(bounded_word),size,bounded_exact) : {Nat.add(value,_) == grown : Nat}
      {==} : {Nat.add(U32.to_nat(sum_word),U32.to_nat(bounded_word)) == grown : Nat}})
  +less = Nat.sub(grown,1n)
  +less_word = (grown_word - 1 : U32)
  +less_exact = Equal.trans(Nat,U32.to_nat(less_word),Nat.sub(U32.to_nat(grown_word),1n),less,
    FullRange.subtract(grown_word,1,half,
      {%Equal.sym(Nat,U32.to_nat(grown_word),grown,grown_exact) : {Nat.is_le(1n,_) == True{} : Bool}
        Order.transitive(1n,size,grown,one,Order.right_le(value,size)) : {Nat.is_le(1n,U32.to_nat(grown_word)) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(grown_word),grown,grown_exact) : {Nat.is_le(Nat.sub(_,1n),U32.to_nat(half)) == True{} : Bool}
        Order.transitive(less,grown,U32.to_nat(half),Order.sub_le(grown,1n),grown_room) :
        {Nat.is_le(Nat.sub(U32.to_nat(grown_word),1n),U32.to_nat(half)) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.sub(number,1n),U32.to_nat(grown_word),grown,grown_exact))
  +last = Ceiling.ceiling(value,size)
  +end_word = (less_word / bounded_word : U32)
  +end_exact = Equal.trans(Nat,U32.to_nat(end_word),Nat.div(U32.to_nat(less_word),U32.to_nat(bounded_word)),last,
    NativeDivision.divide(less_word,bounded_word,half,half_fits,doubled,bounded_positive),
    {%Equal.sym(Nat,U32.to_nat(less_word),less,less_exact) : {Nat.div(_,U32.to_nat(bounded_word)) == last : Nat}
      %Equal.sym(Nat,U32.to_nat(bounded_word),size,bounded_exact) : {Nat.div(less,_) == last : Nat}
      rounded_up(value,size,positive) : {Nat.div(U32.to_nat(less_word),U32.to_nat(bounded_word)) == last : Nat}})
  +below = DivisionOrder.product_below(first_n,size,positive)
  +covers = Ceiling.covers(value,size,positive)
  +product = Nat.mul(start,size)
  +start_last = Ceiling.cancel_product_le(start,last,size,positive,
    Order.transitive(product,first_n,Nat.mul(last,size),below,Order.transitive(first_n,value,Nat.mul(last,size),Order.le_add(first_n,count_n),covers)))
  +ordered = {%Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {Nat.is_le(_,U32.to_nat(end_word)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(end_word),last,end_exact) : {Nat.is_le(start,_) == True{} : Bool}
    start_last : {Nat.is_le(U32.to_nat(start_word),U32.to_nat(end_word)) == True{} : Bool}}
  +last_inside = Ceiling.minimal(value,size,rows,positive,inside)
  +rows_inside = {%Equal.sym(Nat,U32.to_nat(end_word),last,end_exact) : {Nat.is_le(_,rows) == True{} : Bool}
    last_inside : {Nat.is_le(U32.to_nat(end_word),rows) == True{} : Bool}}
  +rows_room = Order.transitive(rows,total,room,RowMajor.product_covers(rows,size,positive),total_room)
  +spread = Nat.sub(last,start)
  +spread_word = Nat.sub(U32.to_nat(end_word),U32.to_nat(start_word))
  +spread_exact = {%Equal.sym(Nat,U32.to_nat(end_word),last,end_exact) : {Nat.sub(_,U32.to_nat(start_word)) == spread : Nat}
    %Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {Nat.sub(last,_) == spread : Nat}
    {==} : {Nat.sub(U32.to_nat(end_word),U32.to_nat(start_word)) == spread : Nat}}
  +span = Nat.mul(spread,size)
  +span_exact = Equal.cong(Nat,Nat,number => Nat.mul(number,size),spread_word,spread,spread_exact)
  +last_room = Order.transitive(Nat.mul(last,size),total,room,RowMajor.mul_le(last,rows,size,last_inside),total_room)
  +span_room = Order.transitive(span,Nat.mul(last,size),room,RowMajor.mul_le(spread,last,size,Order.sub_le(last,start)),last_room)
  +cells_room = {%Equal.sym(Nat,Nat.mul(spread_word,size),span,span_exact) : {Nat.is_le(_,room) == True{} : Bool}
    span_room : {Nat.is_le(Nat.mul(spread_word,size),room) == True{} : Bool}}
  +difference_word = (end_word - start_word : U32)
  +spread_room = Order.transitive(spread,last,room,Order.sub_le(last,start),Order.transitive(last,rows,room,last_inside,rows_room))
  +difference_exact = Equal.trans(Nat,U32.to_nat(difference_word),spread_word,spread,
    FullRange.subtract(end_word,start_word,cap,ordered,
      {%Equal.sym(Nat,spread_word,spread,spread_exact) : {Nat.is_le(_,room) == True{} : Bool} spread_room : {Nat.is_le(spread_word,room) == True{} : Bool}}),
    spread_exact)
  +rows_word = (difference_word * size_word : U32)
  +rows_span = Equal.trans(Nat,U32.to_nat(rows_word),Nat.mul(U32.to_nat(difference_word),U32.to_nat(size_word)),span,
    FullRange.multiply(difference_word,size_word,cap,
      {%Equal.sym(Nat,U32.to_nat(difference_word),spread,difference_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(size_word)),room) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_le(Nat.mul(spread,_),room) == True{} : Bool}
        span_room : {Nat.is_le(Nat.mul(U32.to_nat(difference_word),U32.to_nat(size_word)),room) == True{} : Bool}}),
    {%Equal.sym(Nat,U32.to_nat(difference_word),spread,difference_exact) : {Nat.mul(_,U32.to_nat(size_word)) == span : Nat}
      %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.mul(spread,_) == span : Nat}
      {==} : {Nat.mul(U32.to_nat(difference_word),U32.to_nat(size_word)) == span : Nat}})
  +rows_exact = Equal.trans(Nat,U32.to_nat(rows_word),span,Nat.mul(spread_word,size),rows_span,Equal.sym(Nat,Nat.mul(spread_word,size),span,span_exact))
  +product_word = (start_word * size_word : U32)
  +product_room = Order.transitive(product,first_n,room,below,first_room)
  +product_exact = Equal.trans(Nat,U32.to_nat(product_word),Nat.mul(U32.to_nat(start_word),U32.to_nat(size_word)),product,
    FullRange.multiply(start_word,size_word,cap,
      {%Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(size_word)),room) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.is_le(Nat.mul(start,_),room) == True{} : Bool}
        product_room : {Nat.is_le(Nat.mul(U32.to_nat(start_word),U32.to_nat(size_word)),room) == True{} : Bool}}),
    {%Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {Nat.mul(_,U32.to_nat(size_word)) == product : Nat}
      %Equal.sym(Nat,U32.to_nat(size_word),size,size_exact) : {Nat.mul(start,_) == product : Nat}
      {==} : {Nat.mul(U32.to_nat(start_word),U32.to_nat(size_word)) == product : Nat}})
  +rest = Nat.mod(first_n,size)
  +split = DivisionOrder.quotient_exact(first_n,size,positive)
  +offset_word = (first - product_word : U32)
  +offset_exact = Equal.trans(Nat,U32.to_nat(offset_word),Nat.sub(first_n,U32.to_nat(product_word)),rest,
    FullRange.subtract(first,product_word,cap,
      {%Equal.sym(Nat,U32.to_nat(product_word),product,product_exact) : {Nat.is_le(_,first_n) == True{} : Bool}
        below : {Nat.is_le(U32.to_nat(product_word),first_n) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(product_word),product,product_exact) : {Nat.is_le(Nat.sub(first_n,_),room) == True{} : Bool}
        Order.transitive(Nat.sub(first_n,product),first_n,room,Order.sub_le(first_n,product),first_room) :
        {Nat.is_le(Nat.sub(first_n,U32.to_nat(product_word)),room) == True{} : Bool}}),
    Equal.trans(Nat,Nat.sub(first_n,U32.to_nat(product_word)),Nat.sub(first_n,product),rest,
      Equal.cong(Nat,Nat,number => Nat.sub(first_n,number),U32.to_nat(product_word),product,product_exact),
      Equal.trans(Nat,Nat.sub(first_n,product),Nat.sub(Nat.add(product,rest),product),rest,
        Equal.cong(Nat,Nat,number => Nat.sub(number,product),first_n,Nat.add(product,rest),split),
        Order.subtract_shift(product,rest))))
  +first_exact = {%Equal.sym(Nat,U32.to_nat(start_word),start,start_exact) : {first_n == Nat.add(Nat.mul(_,size),U32.to_nat(offset_word)) : Nat}
    %Equal.sym(Nat,U32.to_nat(offset_word),rest,offset_exact) : {first_n == Nat.add(product,_) : Nat}
    split : {first_n == Nat.add(Nat.mul(U32.to_nat(start_word),size),U32.to_nat(offset_word)) : Nat}}
  +value_split = Equal.trans(Nat,value,Nat.add(Nat.add(product,rest),count_n),Nat.add(product,Nat.add(rest,count_n)),
    Equal.cong(Nat,Nat,number => Nat.add(number,count_n),first_n,Nat.add(product,rest),split),
    Addition.associative(product,rest,count_n))
  +last_split = Equal.trans(Nat,Nat.mul(last,size),Nat.mul(Nat.add(start,spread),size),Nat.add(product,span),
    Equal.cong(Nat,Nat,number => Nat.mul(number,size),last,Nat.add(start,spread),Equal.sym(Nat,Nat.add(start,spread),last,Order.restore_shift(start,last,start_last))),
    RowMajor.distrib(start,spread,size))
  +shifted = {%value_split : {Nat.is_le(_,Nat.add(product,span)) == True{} : Bool}
    %last_split : {Nat.is_le(value,_) == True{} : Bool}
    covers : {Nat.is_le(Nat.add(product,Nat.add(rest,count_n)),Nat.add(product,span)) == True{} : Bool}}
  +rest_fits = Order.cancel_offset_le(product,Nat.add(rest,count_n),span,shifted)
  +fits = {%Equal.sym(Nat,U32.to_nat(offset_word),rest,offset_exact) : {Nat.is_le(Nat.add(_,count_n),Nat.mul(spread_word,size)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(spread_word,size),span,span_exact) : {Nat.is_le(Nat.add(rest,count_n),_) == True{} : Bool}
    rest_fits : {Nat.is_le(Nat.add(U32.to_nat(offset_word),count_n),Nat.mul(spread_word,size)) == True{} : Bool}}
  PieceRows{size_exact,positive,end_exact,ordered,rows_inside,rows_room,cells_room,rows_exact,offset_exact,first_exact,fits,count_room,value_room,first_rest,count_rest,start_exact}

# ---- The cells a piece keeps of its rows ----

# The content of a buffer is its tree and length.
def model_content(+model: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,model) == Buffer.Content{Buffer.tree(~Tile.Tile8,model),Buffer.length(~Tile.Tile8,model)} : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = model
  {==}

# A piece of whole rows keeps every cell of its rows.
def whole_span(+call: Domain.Call,+first: U32,+count: U32,
  +positive: {Nat.is_lt(0n,BatchRows.row_size(call)) == True{} : Bool},
  +end_exact: {U32.to_nat(piece_end(call,first,count)) == Ceiling.ceiling(Nat.add(U32.to_nat(first),U32.to_nat(count)),BatchRows.row_size(call)) : Nat},
  +ordered: {Nat.is_le(U32.to_nat(piece_start(call,first)),U32.to_nat(piece_end(call,first,count))) == True{} : Bool},
  +offset_exact: {U32.to_nat(piece_offset(call,first)) == Nat.mod(U32.to_nat(first),BatchRows.row_size(call)) : Nat},
  +first_exact: {U32.to_nat(first) == Nat.add(Nat.mul(U32.to_nat(piece_start(call,first)),BatchRows.row_size(call)),U32.to_nat(piece_offset(call,first))) : Nat},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(piece_offset(call,first)),U32.to_nat(count)),piece_span(call,first,count)) == True{} : Bool},
  +first_zero: {Nat.mod(U32.to_nat(first),BatchRows.row_size(call)) == 0n : Nat},
  +count_zero: {Nat.mod(U32.to_nat(count),BatchRows.row_size(call)) == 0n : Nat}) ->
  {piece_span(call,first,count) == U32.to_nat(count) : Nat}:
  +size = BatchRows.row_size(call)
  +first_n = U32.to_nat(first)
  +count_n = U32.to_nat(count)
  +offset = U32.to_nat(piece_offset(call,first))
  +start = U32.to_nat(piece_start(call,first))
  +last = U32.to_nat(piece_end(call,first,count))
  +span = piece_span(call,first,count)
  +blocks = Nat.div(count_n,size)
  +whole_blocks = Nat.mul(blocks,size)
  +product = Nat.mul(start,size)
  +count_split = Equal.trans(Nat,count_n,Nat.add(whole_blocks,Nat.mod(count_n,size)),whole_blocks,
    DivisionOrder.quotient_exact(count_n,size,positive),
    Equal.trans(Nat,Nat.add(whole_blocks,Nat.mod(count_n,size)),Nat.add(whole_blocks,0n),whole_blocks,
      Equal.cong(Nat,Nat,number => Nat.add(whole_blocks,number),Nat.mod(count_n,size),0n,count_zero),
      Addition.zero_right(whole_blocks)))
  +offset_zero = Equal.trans(Nat,offset,Nat.mod(first_n,size),0n,offset_exact,first_zero)
  +first_split = Equal.trans(Nat,first_n,Nat.add(product,offset),product,first_exact,
    Equal.trans(Nat,Nat.add(product,offset),Nat.add(product,0n),product,
      Equal.cong(Nat,Nat,number => Nat.add(product,number),offset,0n,offset_zero),
      Addition.zero_right(product)))
  +joined = Nat.add(start,blocks)
  +value_split = Equal.trans(Nat,Nat.add(first_n,count_n),Nat.add(product,count_n),Nat.mul(joined,size),
    Equal.cong(Nat,Nat,number => Nat.add(number,count_n),first_n,product,first_split),
    Equal.trans(Nat,Nat.add(product,count_n),Nat.add(product,whole_blocks),Nat.mul(joined,size),
      Equal.cong(Nat,Nat,number => Nat.add(product,number),count_n,whole_blocks,count_split),
      Equal.sym(Nat,Nat.mul(joined,size),Nat.add(product,whole_blocks),RowMajor.distrib(start,blocks,size))))
  +value_le = {%Equal.sym(Nat,Nat.add(first_n,count_n),Nat.mul(joined,size),value_split) : {Nat.is_le(_,Nat.mul(joined,size)) == True{} : Bool}
    Order.reflexive(Nat.mul(joined,size)) : {Nat.is_le(Nat.add(first_n,count_n),Nat.mul(joined,size)) == True{} : Bool}}
  +last_bound = {%Equal.sym(Nat,last,Ceiling.ceiling(Nat.add(first_n,count_n),size),end_exact) : {Nat.is_le(_,joined) == True{} : Bool}
    Ceiling.minimal(Nat.add(first_n,count_n),size,joined,positive,value_le) : {Nat.is_le(last,joined) == True{} : Bool}}
  +spread_bound = Order.cancel_offset_le(start,Nat.sub(last,start),blocks,
    {%Equal.sym(Nat,Nat.add(start,Nat.sub(last,start)),last,Order.restore_shift(start,last,ordered)) : {Nat.is_le(_,joined) == True{} : Bool}
      last_bound : {Nat.is_le(Nat.add(start,Nat.sub(last,start)),joined) == True{} : Bool}})
  +span_le = {%Equal.sym(Nat,count_n,whole_blocks,count_split) : {Nat.is_le(span,_) == True{} : Bool}
    RowMajor.mul_le(Nat.sub(last,start),blocks,size,spread_bound) : {Nat.is_le(span,count_n) == True{} : Bool}}
  +count_le = Order.transitive(count_n,Nat.add(offset,count_n),span,Order.right_le(offset,count_n),fits)
  Order.antisymmetric(span,count_n,span_le,count_le)

# A word below a bound of the capacity is below the bound.
def word_below(+count: U32,+capacity: U32,+upper: U32,+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},
  +bounded: {U32.is_le(capacity,upper) == True{} : Bool}) -> {U32.is_le(count,upper) == True{} : Bool}:
  NativeDivision.nat_le_native(count,upper,Order.transitive(U32.to_nat(count),U32.to_nat(capacity),U32.to_nat(upper),room,Bounds.as_nat_le(capacity,upper,bounded)))

# The cells a piece keeps and the buffer that holds them.
type KeptPiece<-tree: Tree.Tree<Tile.Tile8>,-offset: Nat,-count: U32,-result: Storage.Buffer<Tile.Tile8> & Kernel.Operands(),-operands: Kernel.Operands()> is Data:
  KeptPiece{model: Buffer.Model<Tile.Tile8>,kept: Tree.Tree<Tile.Tile8>,length: Nat,
    exact: {result == (Buffer.pack(~Tile.Tile8,model),operands) : Storage.Buffer<Tile.Tile8> & Kernel.Operands()},
    contents: {Buffer.content(~Tile.Tile8,model) == Buffer.Content{kept,length} : Buffer.Content<Tile.Tile8>},
    ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{kept,length}>,length_exact: {length == U32.to_nat(count) : Nat},
    fact: Kept<tree,offset,count,kept>}

# The cells a piece keeps of the rows of its task: the rows when the piece
# holds whole rows, otherwise the cells cut from them.
def kept_case(whole: Bool,+call: Domain.Call,+first: U32,+count: U32,+rows: PieceRows<call,first,count>,
  +classified: {U32.is_zero((first % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) &&
    U32.is_zero((count % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) == whole : Bool},
  +holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +depth_ok: {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool},
  +output_room: {Nat.is_le(Buffer.length(~Tile.Tile8,output),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool},
  +length_exact: {Buffer.length(~Tile.Tile8,output) == piece_span(call,first,count) : Nat}) ->
  KeptPiece<Buffer.tree(~Tile.Tile8,output),U32.to_nat(piece_offset(call,first)),count,
    Kernel.piece_kept(whole,piece_offset(call,first),count,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),
    (Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))>:
  match whole:
    case True{}:
      PieceRows{+size_exact,+positive,+end_exact,+ordered,+rows_inside,+rows_room,+cells_room,+rows_exact,+offset_exact,+first_exact,+fits,+count_room,+span_room,
        +first_rest,+count_rest,+start_exact} = rows
      +size = BatchRows.row_size(call)
      +bounded_word = U32.max(Kernel.row_cells(Domain.geometry(call)),1)
      +first_word = (first % bounded_word : U32)
      +count_word = (count % bounded_word : U32)
      +first_zero = Equal.trans(Nat,Nat.mod(U32.to_nat(first),size),U32.to_nat(first_word),0n,
        Equal.sym(Nat,U32.to_nat(first_word),Nat.mod(U32.to_nat(first),size),first_rest),
        Equal.cong(U32,Nat,word => U32.to_nat(word),first_word,0,
          Arithmetic.word_zero(first_word,Boolean.and_true_left(U32.is_zero(first_word),U32.is_zero(count_word),classified))))
      +count_zero = Equal.trans(Nat,Nat.mod(U32.to_nat(count),size),U32.to_nat(count_word),0n,
        Equal.sym(Nat,U32.to_nat(count_word),Nat.mod(U32.to_nat(count),size),count_rest),
        Equal.cong(U32,Nat,word => U32.to_nat(word),count_word,0,
          Arithmetic.word_zero(count_word,Boolean.and_true_right(U32.is_zero(first_word),U32.is_zero(count_word),classified))))
      +offset_zero = Equal.trans(Nat,U32.to_nat(piece_offset(call,first)),Nat.mod(U32.to_nat(first),size),0n,offset_exact,first_zero)
      +length_count = Equal.trans(Nat,Buffer.length(~Tile.Tile8,output),piece_span(call,first,count),U32.to_nat(count),length_exact,
        whole_span(call,first,count,positive,end_exact,ordered,offset_exact,first_exact,fits,first_zero,count_zero))
      +ready = {Buffer.ContentReady{Buffer.depth(~Tile.Tile8,output),Buffer.balanced(~Tile.Tile8,output),depth_ok,output_room} :
        Buffer.ContentReady<Tile.Tile8,Buffer.Content{Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output)}>}
      +fact = {WholeKept{{==},offset_zero} :
        Kept<Buffer.tree(~Tile.Tile8,output),U32.to_nat(piece_offset(call,first)),count,Buffer.tree(~Tile.Tile8,output)>}
      KeptPiece{output,Buffer.tree(~Tile.Tile8,output),Buffer.length(~Tile.Tile8,output),{==},model_content(output),ready,length_count,fact}
    case False{}:
      PieceRows{+size_exact,+positive,+end_exact,+ordered,+rows_inside,+rows_room,+cells_room,+rows_exact,+offset_exact,+first_exact,+fits,+count_room,+span_room,
        +first_rest,+count_rest,+start_exact} = rows
      +offset_word = piece_offset(call,first)
      +count_n = U32.to_nat(count)
      +limits = Domain.limits(call)
      +depth = Tensors.depth_for(count)
      +zero = Tiled.zero_tile()
      +fresh = Tree.fresh(Tile.Tile8,depth,zero)
      +target = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,zero,count))
      +allocatable = word_below(count,Machine.capacity(limits),1073741824,count_room,Limits.capacity_allocatable(limits))
      +sized = Depth.bounded(count,allocatable)
      +cut_depth = Depth.allocation_depth(count,sized)
      +cut_room = {%Equal.sym(U32,Tree.capacity(Tile.Tile8,fresh),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh,Shape.fresh(Tile.Tile8,depth,zero))) :
          {Nat.is_le(count_n,U32.to_nat(_)) == True{} : Bool}
        Depth.allocation_room(count,sized) : {Nat.is_le(count_n,U32.to_nat(Tree.capacity(Tile.Tile8,fresh))) == True{} : Bool}}
      +cut_capacity = Buffer.capacity_bound(~Tile.Tile8,Buffer.fresh(~Tile.Tile8,depth,zero,count_n),cut_depth)
      +writes = Contiguous.writes(~Tile.Tile8,count_n,Buffer.tree(~Tile.Tile8,output),0,offset_word,0)
      +kept = Writes.pure(Tile.Tile8,writes,fresh)
      +copied_contents = Equal.trans(Buffer.Content<Tile.Tile8>,
        Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count)))),
        Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Buffer.pack(~Tile.Tile8,target)))),
        Buffer.Content{kept,count_n},
        Equal.cong(Storage.Buffer<Tile.Tile8>,Buffer.Content<Tile.Tile8>,
          buffer => Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
            Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),buffer))),
          Tensors.buffer(~Tile.Tile8,zero,count),Buffer.pack(~Tile.Tile8,target),
          Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,target),Tensors.buffer(~Tile.Tile8,zero,count),
            Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,zero,count)))),
        Equal.trans(Buffer.Content<Tile.Tile8>,
          Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
            Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Buffer.pack(~Tile.Tile8,target)))),
          Buffer.Content{Writes.pure(Tile.Tile8,writes,Buffer.tree(~Tile.Tile8,target)),Buffer.length(~Tile.Tile8,target)},
          Buffer.Content{kept,count_n},
          cut_contents(offset_word,count,output,target),
          Equal.trans(Buffer.Content<Tile.Tile8>,
            Buffer.Content{Writes.pure(Tile.Tile8,writes,Buffer.tree(~Tile.Tile8,target)),Buffer.length(~Tile.Tile8,target)},
            Buffer.Content{kept,Buffer.length(~Tile.Tile8,target)},
            Buffer.Content{kept,count_n},
            Equal.cong(Tree.Tree<Tile.Tile8>,Buffer.Content<Tile.Tile8>,cells => Buffer.Content{Writes.pure(Tile.Tile8,writes,cells),Buffer.length(~Tile.Tile8,target)},
              Buffer.tree(~Tile.Tile8,target),fresh,AllocationLaws.buffer_tree(~Tile.Tile8,zero,count)),
            Equal.cong(Nat,Buffer.Content<Tile.Tile8>,size => Buffer.Content{kept,size},
              Buffer.length(~Tile.Tile8,target),count_n,AllocationLaws.buffer_length(~Tile.Tile8,zero,count)))))
      +model = Buffer.remodel(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
        Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count))))
      +contents = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,model),
        Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count)))),
        Buffer.Content{kept,count_n},
        Buffer.remodel_content(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count)))),
        copied_contents)
      +exact = Equal.cong(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8> & Kernel.Operands(),
        buffer => (buffer,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
        Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count))),
        Buffer.pack(~Tile.Tile8,model),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,model),
          Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
            Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count))),
          Buffer.remodeled(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
            Copy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset_word},0,Buffer.pack(~Tile.Tile8,output),Tensors.buffer(~Tile.Tile8,zero,count))))))
      +ready = Writes.content_ready(~Tile.Tile8,writes,fresh,count_n,Buffer.ContentReady{depth,Shape.fresh(Tile.Tile8,depth,zero),cut_depth,cut_room})
      +fact = {CutKept{offset_word,{==},cut_depth,cut_capacity,cut_room,{==}} : Kept<Buffer.tree(~Tile.Tile8,output),U32.to_nat(offset_word),count,kept>}
      KeptPiece{model,kept,count_n,exact,contents,ready,{==},fact}

# ---- The piece a task computes ----

# The runtime pads the cells a piece keeps.
def padded_exact(+geometry: Kernel.Geometry,+first: U32,+count: U32,-buffer: Storage.Buffer<Tile.Tile8>,-operands: Kernel.Operands()) ->
  {Kernel.piece_padded(geometry,first,count,(buffer,operands)) == (TiledMap.padded(piece_padding(geometry),first,count,buffer),operands) :
    Storage.Buffer<Tile.Tile8> & Kernel.Operands()}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  {==}

# The rows of the batch a piece reads, with the holding and weights its task
# lends them from.
def piece_batch(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+first: U32,+count: U32,+rows: PieceRows<call,first,count>) ->
  BatchRows.Batch<Holder,value>:
  BatchRows.Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,piece_start(call,first),piece_end(call,first,count),
    piece_span(call,first,count),rows_ordered(call,first,count,rows),rows_inside_of(call,first,count,rows),rows_room_of(call,first,count,rows),
    rows_cells_room(call,first,count,rows)}

# The piece of cells [first, first + count) a task computes: its output, what
# the task holds after it, the runtime result and the values of the piece.
type PieceComputed<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-call: Domain.Call,-weights: Buffer.Model<Tile.Tile8>,
  -low: Nat,-high: Nat,-layout: Held.Spans,-outer: U32,-after: U32,-first: U32,-count: U32,-result: Storage.Buffer<Tile.Tile8> & Kernel.Operands()> is Data:
  PieceComputed{output: Buffer.Model<Tile.Tile8>,holding: Held.Holding<Tile.Tile8>,
    tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
    ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
    tiles: Buffer.Model<Tile.Tile8>,
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(call,weights)) : Tree.Tree<Tile.Tile8>},
    exact: {result == (Buffer.pack(~Tile.Tile8,output),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))) : Storage.Buffer<Tile.Tile8> & Kernel.Operands()},
    leaf: Content.Leaf(~Tile.Tile8,~Representation.Lifted<Setup>,~Representation.lifted(~Setup,~input_value),
      Representation.Lifted{Setup{call,BandSource.gradient_model(~Holder,~value,holder,Rows.cells_word(call)),weights},cells_of(Domain.geometry(call),outer,after)},
      first,count,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output))),
    sized: {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,output)) == U32.to_nat(count) : Nat},
    ready: Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,output)>}

# The piece from the cells it keeps of the rows of its task.
def piece_from_kept(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+first: U32,+count: U32,+rows: PieceRows<call,first,count>,
  +outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
  +spare: Held.SpareModel<Tile.Tile8>,
  +tiles: Buffer.Model<Tile.Tile8>,+tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(call,weights)) : Tree.Tree<Tile.Tile8>},
  +output: Buffer.Model<Tile.Tile8>,
  +written: Members.All(Nat,{cell => BatchRows.WrittenCell(~Holder,~value,
      piece_batch(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows),Buffer.tree(~Tile.Tile8,output),cell) :
    Nat -> Data},Ranges.range(piece_span(call,first,count),0n)),
  -whole: Bool,
  +kept_piece: KeptPiece<Buffer.tree(~Tile.Tile8,output),U32.to_nat(piece_offset(call,first)),count,
    Kernel.piece_kept(whole,piece_offset(call,first),count,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),
    (Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))>) ->
  PieceComputed<Holder,value,holder,call,weights,low,high,layout,outer,after,first,count,
    Kernel.piece_padded(Domain.geometry(call),first,count,
      Kernel.piece_kept(whole,piece_offset(call,first),count,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}))>:
  KeptPiece{+model,+kept,+length,+exact,+contents,+ready,+length_exact,+fact} = kept_piece
  +batch = piece_batch(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows)
  +geometry = Domain.geometry(call)
  +padding = piece_padding(geometry)
  +scope = {PieceScope{batch,outer,after,after_exact,outer_exact,values_room,Buffer.tree(~Tile.Tile8,output),written,first,count,U32.to_nat(piece_offset(call,first)),
      rows_first_exact(call,first,count,rows),rows_fits(call,first,count,rows),rows_span_room(call,first,count,rows),kept,length,ready,length_exact,fact} :
    PieceScope<Holder,value>}
  +content = {Buffer.Content{kept,length} : Buffer.Content<Tile.Tile8>}
  +stepped = Padding.step(padding,first,count,content)
  +out = Buffer.remodel(~Tile.Tile8,TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model)))
  +out_content = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,out),
    Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model))),stepped,
    Buffer.remodel_content(~Tile.Tile8,TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model))),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model))),
      Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))),stepped,
      Padding.rule(padding,first,count,Buffer.pack(~Tile.Tile8,model)),
      Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,cells => Padding.step(padding,first,count,cells),
        Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),content,
        Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.content(~Tile.Tile8,model),content,
          Buffer.packed_contents(~Tile.Tile8,model),contents))))
  +lifted = {Representation.Lifted{Setup{call,BandSource.gradient_model(~Holder,~value,holder,Rows.cells_word(call)),weights},cells_of(geometry,outer,after)} :
    Representation.Lifted<Setup>}
  +leaf = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,out),stepped,out_content) :
      Content.Leaf(~Tile.Tile8,~Representation.Lifted<Setup>,~Representation.lifted(~Setup,~input_value),lifted,first,count,
        Buffer.content_tree(~Tile.Tile8,_),Buffer.content_length(~Tile.Tile8,_))
    piece_leaf(~Holder,~value,scope) :
    Content.Leaf(~Tile.Tile8,~Representation.Lifted<Setup>,~Representation.lifted(~Setup,~input_value),lifted,first,count,
      Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,out)),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,out)))}
  +padded = Padding.kept_ready(padding,first,count,kept,length,ready)
  +out_ready = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,out),stepped,out_content) : Buffer.ContentReady<Tile.Tile8,_>
    Shape.first(Buffer.ContentReady<Tile.Tile8,stepped>,{Buffer.content_length(~Tile.Tile8,stepped) == length : Nat},padded) :
    Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,out)>}
  +sized = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,out),stepped,out_content) : {Buffer.content_length(~Tile.Tile8,_) == U32.to_nat(count) : Nat}
    Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,stepped),length,U32.to_nat(count),
      Shape.second(Buffer.ContentReady<Tile.Tile8,stepped>,{Buffer.content_length(~Tile.Tile8,stepped) == length : Nat},padded),length_exact) :
    {Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,out)) == U32.to_nat(count) : Nat}}
  +out_exact = Equal.trans(Storage.Buffer<Tile.Tile8> & Kernel.Operands(),
    Kernel.piece_padded(geometry,first,count,
      Kernel.piece_kept(whole,piece_offset(call,first),count,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)})),
    Kernel.piece_padded(geometry,first,count,(Buffer.pack(~Tile.Tile8,model),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles)))),
    (Buffer.pack(~Tile.Tile8,out),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
    Equal.cong(Storage.Buffer<Tile.Tile8> & Kernel.Operands(),Storage.Buffer<Tile.Tile8> & Kernel.Operands(),pair => Kernel.piece_padded(geometry,first,count,pair),
      Kernel.piece_kept(whole,piece_offset(call,first),count,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),
      (Buffer.pack(~Tile.Tile8,model),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),exact),
    Equal.trans(Storage.Buffer<Tile.Tile8> & Kernel.Operands(),
      Kernel.piece_padded(geometry,first,count,(Buffer.pack(~Tile.Tile8,model),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles)))),
      (TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model)),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
      (Buffer.pack(~Tile.Tile8,out),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
      padded_exact(geometry,first,count,Buffer.pack(~Tile.Tile8,model),(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
      Equal.cong(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8> & Kernel.Operands(),buffer => (buffer,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles))),
        TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model)),Buffer.pack(~Tile.Tile8,out),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,out),TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model)),
          Buffer.remodeled(~Tile.Tile8,TiledMap.padded(padding,first,count,Buffer.pack(~Tile.Tile8,model)))))))
  PieceComputed{out,holding,tiled,values,ordered,tiles,tiles_exact,out_exact,leaf,sized,out_ready}

# The piece from the rows its task computes.
def piece_finished(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+first: U32,+count: U32,+rows: PieceRows<call,first,count>,
  +outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  -state: Kernel.Rows,
  +finished: BatchRows.Finished<Holder,value,piece_batch(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows),state>) ->
  PieceComputed<Holder,value,holder,call,weights,low,high,layout,outer,after,first,count,
    Kernel.piece_padded(Domain.geometry(call),first,count,
      Kernel.piece_kept(U32.is_zero((first % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) &&
        U32.is_zero((count % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)),piece_offset(call,first),count,state))>:
  BatchRows.Finished{+holding,+tiled,+values,+ordered,+spare,+spare_room,+tiles,+tiles_exact,+output,+depth_ok,+room,+length_exact,+written,+packed} = finished
  +whole = U32.is_zero((first % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32)) &&
    U32.is_zero((count % U32.max(Kernel.row_cells(Domain.geometry(call)),1) : U32))
  +capacity_word = Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output))
  +output_room = {%Equal.sym(Nat,Buffer.length(~Tile.Tile8,output),piece_span(call,first,count),length_exact) :
      {Nat.is_le(_,U32.to_nat(capacity_word)) == True{} : Bool}
    room : {Nat.is_le(Buffer.length(~Tile.Tile8,output),U32.to_nat(capacity_word)) == True{} : Bool}}
  %Equal.sym(Kernel.Rows,state,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)},packed) :
    PieceComputed<Holder,value,holder,call,weights,low,high,layout,outer,after,first,count,
      Kernel.piece_padded(Domain.geometry(call),first,count,Kernel.piece_kept(whole,piece_offset(call,first),count,_))>
  piece_from_kept(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows,outer,after,after_exact,outer_exact,values_room,
    holding,tiled,values,ordered,spare,tiles,tiles_exact,output,written,whole,
    kept_case(whole,call,first,count,rows,{==},holding,spare,tiles,output,depth_ok,output_room,length_exact))

# The piece of cells [first, first + count) a task computes from what it
# holds, given that the bands of its rows lie in the holding.
def piece_task(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,+first: U32,+count: U32,+rows: PieceRows<call,first,count>,
  +outer: U32,+after: U32,
  +after_exact: {Representation.after_count(cells_of(Domain.geometry(call),outer,after)) == Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call))) : Nat},
  +outer_exact: {U32.to_nat(outer) == Nat.mul(Domain.batch(call),Nat.mul(Domain.height(Domain.geometry(call)),Domain.width(Domain.geometry(call)))) : Nat},
  +values_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.image_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +holding: Held.Holding<Tile.Tile8>,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
  +ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
  +tiles: Buffer.Model<Tile.Tile8>,+tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(call,weights)) : Tree.Tree<Tile.Tile8>},
  +facts: BatchRows.CallFacts(~Holder,~value,Kernel.images_between(Domain.geometry(call),piece_start(call,first),piece_end(call,first,count)),
    piece_batch(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows),piece_start(call,first))) ->
  PieceComputed<Holder,value,holder,call,weights,low,high,layout,outer,after,first,count,
    Kernel.piece_cells(Rows.task_of(Domain.geometry(call)),first,count,(Held.pack(~Tile.Tile8,holding),Buffer.pack(~Tile.Tile8,tiles)))>:
  +batch = piece_batch(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows)
  +geometry = Domain.geometry(call)
  +start = piece_start(call,first)
  +rows_word = piece_rows_word(call,first,count)
  +rows_exact = rows_exact_of(call,first,count,rows)
  +cap = Machine.capacity(Domain.limits(call))
  +fill = Tile.filled((0.0 + 0.0 : F32))
  +rows_room = {%Equal.sym(Nat,U32.to_nat(rows_word),piece_span(call,first,count),rows_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    rows_cells_room(call,first,count,rows) : {Nat.is_le(U32.to_nat(rows_word),U32.to_nat(cap)) == True{} : Bool}}
  +sized = Depth.bounded(rows_word,word_below(rows_word,cap,1073741824,rows_room,Limits.capacity_allocatable(Domain.limits(call))))
  +depth = Tensors.depth_for(rows_word)
  +fresh = Tree.fresh(Tile.Tile8,depth,fill)
  +fresh_room = {%Equal.sym(U32,Tree.capacity(Tile.Tile8,fresh),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,fresh,Shape.fresh(Tile.Tile8,depth,fill))) :
      {Nat.is_le(U32.to_nat(rows_word),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(rows_word,sized) : {Nat.is_le(U32.to_nat(rows_word),U32.to_nat(Tree.capacity(Tile.Tile8,fresh))) == True{} : Bool}}
  +reached = started(~Holder,~value,batch,holding,tiled,values,ordered,tiles,tiles_exact,rows_word,rows_exact,rows_exact,Depth.allocation_depth(rows_word,sized),fresh_room)
  piece_finished(~Holder,~value,holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,first,count,rows,outer,after,after_exact,outer_exact,values_room,
    Kernel.batch_rows(Kernel.images_between(geometry,start,piece_end(call,first,count)),Rows.task_of(geometry),start,piece_end(call,first,count),0,
      Kernel.Rows{Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Tile.filled(0.0)),Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,fill,rows_word)}),
    BatchRows.batch_loop(~Holder,~value,Kernel.images_between(geometry,start,piece_end(call,first,count)),batch,start,0,
      Kernel.Rows{Held.pack(~Tile.Tile8,holding),Move.spare(~Tile.Tile8,Tile.filled(0.0)),Buffer.pack(~Tile.Tile8,tiles),Tensors.buffer(~Tile.Tile8,fill,rows_word)},facts,reached))
