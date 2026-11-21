# The rows of the batch an input-gradient task computes, image by image: after
# the calls on rows [start, end), cell c of the output holds the input
# gradient of pixel c / G and group c % G of the rows from start on, in its
# lanes below the channels.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/tensor_move.bend as Move
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_band_source.bend as BandSource
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/tensor_tiled_padded_cells.bend as PaddedCells
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/u32_extrema.bend as Extrema
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/u32_division.bend as NativeDivision
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/tensor_grid.bend as Grid
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_cell_sources.bend as Sources
import ../input_gradient_specification.bend as Specification

# ---- The batch a task computes rows of ----

# The rows [start, end) of a batch of a call, the gradient its holdings hold
# cells [low, high) of and the weights. The rows lie in the batch and their
# cells fit the machine.
type Batch<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Batch{holder: Holder,call: Domain.Call,envelope: Geometry.WordEnvelope,
    fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
    weights: Buffer.Model<Tile.Tile8>,low: Nat,high: Nat,capacity: U32,
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    layout: Held.Spans,start: U32,end: U32,length: Nat,
    ordered: {Nat.is_le(U32.to_nat(start),U32.to_nat(end)) == True{} : Bool},
    rows_inside: {Nat.is_le(U32.to_nat(end),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
    rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.mul(Nat.sub(U32.to_nat(end),U32.to_nat(start)),
      Nat.mul(Domain.width(Domain.geometry(call)),Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}}
def batch_holder(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Holder:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  holder
def batch_call(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Domain.Call:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  call
def batch_envelope(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Geometry.WordEnvelope:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  envelope
def batch_fits(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) ->
  {Kernel.fits(Domain.geometry(batch_call(~Holder,~value,batch)),Machine.capacity(Domain.limits(batch_call(~Holder,~value,batch)))) == True{} : Bool}:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  fits
def batch_weights(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  weights
def batch_low(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Nat:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  low
def batch_high(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Nat:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  high
def batch_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> U32:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  capacity
def batch_capacity_ok(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) ->
  {U32.is_le(batch_capacity(~Holder,~value,batch),2147483648) == True{} : Bool}:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  capacity_ok
def batch_bounded(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) ->
  {Nat.is_le(batch_high(~Holder,~value,batch),U32.to_nat(batch_capacity(~Holder,~value,batch))) == True{} : Bool}:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  bounded
def batch_layout(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Held.Spans:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  layout
def batch_start(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> U32:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  start
def batch_end(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> U32:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  end
def batch_length(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) -> Nat:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  length
def batch_ordered(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) ->
  {Nat.is_le(U32.to_nat(batch_start(~Holder,~value,batch)),U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool}:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  ordered
def batch_cells_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,batch: Batch<Holder,value>) ->
  {Nat.is_le(Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(batch_start(~Holder,~value,batch))),
    Nat.mul(Domain.width(Domain.geometry(batch_call(~Holder,~value,batch))),Ceiling.ceiling(Domain.channels(Domain.geometry(batch_call(~Holder,~value,batch))),8n))),
    U32.to_nat(Machine.capacity(Domain.limits(batch_call(~Holder,~value,batch))))) == True{} : Bool}:
  Batch{holder,call,envelope,fits,weights,low,high,capacity,capacity_ok,bounded,layout,start,end,length,ordered,rows_inside,rows_room,cells_room} = batch
  cells_room

# The cells of an output row of a call: the groups of eight channels of each
# column.
def row_size(+call: Domain.Call) -> Nat:
  Nat.mul(Domain.width(Domain.geometry(call)),Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n))
def batch_geometry(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Kernel.Geometry:
  Domain.geometry(batch_call(~Holder,~value,batch))
def batch_task(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Kernel.Task:
  Rows.task_of(batch_geometry(~Holder,~value,batch))
def batch_size(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Nat:
  row_size(batch_call(~Holder,~value,batch))
def batch_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Nat:
  Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch))
def batch_groups(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Nat:
  Ceiling.ceiling(Domain.channels(batch_geometry(~Holder,~value,batch)),8n)
def batch_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  BandSource.gradient_model(~Holder,~value,batch_holder(~Holder,~value,batch),Rows.cells_word(batch_call(~Holder,~value,batch)))

# The input gradient of a channel at pixel p of the rows from start on: row
# start + p / width of the batch, column p % width.
def pixel_value(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+pixel: Nat,+channel: Nat) -> F32:
  +geometry = batch_geometry(~Holder,~value,batch)
  +width = Domain.width(geometry)
  +height = Domain.height(geometry)
  +row = Nat.add(U32.to_nat(batch_start(~Holder,~value,batch)),Nat.div(pixel,width))
  Specification.value(Sources.problem(geometry,Sources.models_reader(geometry,batch_model(~Holder,~value,batch),batch_weights(~Holder,~value,batch)),Nat.div(row,height)),
    channel,Nat.mod(row,height),Nat.mod(pixel,width))
# The tile of a group at a pixel, +0 past the channels, and an output cell
# that holds the tile of its pixel and group in its lanes below the channels.
def written_tile(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+pixel: Nat,+group: Nat) -> Tile.Tile8:
  +channels = Domain.channels(batch_geometry(~Holder,~value,batch))
  PaddedCells.kept_tile(~(lane => pixel_value(~Holder,~value,batch,pixel,lane)),channels,group)
def WrittenCell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+tree: Tree.Tree<Tile.Tile8>,+cell: Nat) -> Data:
  +groups = batch_groups(~Holder,~value,batch)
  PaddedCells.Valid<Domain.channels(batch_geometry(~Holder,~value,batch)),Nat.mod(cell,groups),Tree.read(Tile.Tile8,tree,U32.from_nat(cell)),
    written_tile(~Holder,~value,batch,Nat.div(cell,groups),Nat.mod(cell,groups))>

# ---- The words of the loop ----

# One step of the loop from row row: the image of the row, its row in the
# image, the rows up to the end of the image or of the batch, and the next
# row and output cell.
def image_word(+geometry: Kernel.Geometry,+row: U32) -> U32:
  (row / Kernel.image_height(geometry) : U32)
def first_word(+geometry: Kernel.Geometry,+row: U32) -> U32:
  (row % Kernel.image_height(geometry) : U32)
def count_word(+geometry: Kernel.Geometry,+row: U32,+end: U32) -> U32:
  (U32.min(end,((image_word(geometry,row) + 1) * Kernel.image_height(geometry) : U32)) - row : U32)
def next_word(+geometry: Kernel.Geometry,+row: U32,+end: U32) -> U32:
  (row + count_word(geometry,row,end) : U32)
def target_word(+geometry: Kernel.Geometry,+row: U32,+end: U32,+target: U32) -> U32:
  (target + count_word(geometry,row,end) * Kernel.row_cells(geometry) : U32)
def shifted_sum(+start: Nat,+row: Nat,+count: Nat,+after: {Nat.is_le(start,row) == True{} : Bool}) ->
  {Nat.sub(Nat.add(row,count),start) == Nat.add(Nat.sub(row,start),count) : Nat}:
  +offset = Nat.sub(row,start)
  %Order.restore_shift(start,row,after) : {Nat.sub(Nat.add(_,count),start) == Nat.add(offset,count) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(start,offset),count),Nat.add(start,Nat.add(offset,count)),Addition.associative(start,offset,count)) :
    {Nat.sub(_,start) == Nat.add(offset,count) : Nat}
  Order.subtract_shift(start,Nat.add(offset,count))
def rest_split(+row: Nat,+count: Nat,+end: Nat,+fits: {Nat.is_le(Nat.add(row,count),end) == True{} : Bool}) ->
  {Nat.sub(end,row) == Nat.add(count,Nat.sub(end,Nat.add(row,count))) : Nat}:
  +rest = Nat.sub(end,Nat.add(row,count))
  %Order.restore_shift(Nat.add(row,count),end,fits) : {Nat.sub(_,row) == Nat.add(count,rest) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(row,count),rest),Nat.add(row,Nat.add(count,rest)),Addition.associative(row,count,rest)) :
    {Nat.sub(_,row) == Nat.add(count,rest) : Nat}
  Order.subtract_shift(row,Nat.add(count,rest))
# The rows from start to the end of the batch are those before the row, the
# rows of a step and those after it.
def whole_split(+start: Nat,+row: Nat,+count: Nat,+end: Nat,+after: {Nat.is_le(start,row) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(row,count),end) == True{} : Bool}) ->
  {Nat.sub(end,start) == Nat.add(Nat.sub(row,start),Nat.add(count,Nat.sub(end,Nat.add(row,count)))) : Nat}:
  +rest = Nat.sub(end,Nat.add(row,count))
  %Order.restore_shift(Nat.add(row,count),end,fits) : {Nat.sub(_,start) == Nat.add(Nat.sub(row,start),Nat.add(count,rest)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(row,count),rest),Nat.add(row,Nat.add(count,rest)),Addition.associative(row,count,rest)) :
    {Nat.sub(_,start) == Nat.add(Nat.sub(row,start),Nat.add(count,rest)) : Nat}
  shifted_sum(start,row,Nat.add(count,rest),after)

# A cell of pixel (offset + row) * width + column and group g.
def cell_split(+offset: Nat,+row: Nat,+width: Nat,+column: Nat,+groups: Nat,+group: Nat) ->
  {Nat.add(Nat.mul(offset,Nat.mul(width,groups)),Nat.add(Nat.mul(Nat.add(Nat.mul(row,width),column),groups),group)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(offset,row),width),column),groups),group) : Nat}:
  +inner = Nat.add(Nat.mul(row,width),column)
  +left = Nat.mul(Nat.mul(offset,width),groups)
  %RowMajor.assoc(offset,width,groups) :
    {Nat.add(_,Nat.add(Nat.mul(inner,groups),group)) == Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(offset,row),width),column),groups),group) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(offset,row),width),Nat.add(Nat.mul(offset,width),Nat.mul(row,width)),RowMajor.distrib(offset,row,width)) :
    {Nat.add(left,Nat.add(Nat.mul(inner,groups),group)) == Nat.add(Nat.mul(Nat.add(_,column),groups),group) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(offset,width),Nat.mul(row,width)),column),Nat.add(Nat.mul(offset,width),inner),Addition.associative(Nat.mul(offset,width),Nat.mul(row,width),column)) :
    {Nat.add(left,Nat.add(Nat.mul(inner,groups),group)) == Nat.add(Nat.mul(_,groups),group) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(offset,width),inner),groups),Nat.add(left,Nat.mul(inner,groups)),RowMajor.distrib(Nat.mul(offset,width),inner,groups)) :
    {Nat.add(left,Nat.add(Nat.mul(inner,groups),group)) == Nat.add(_,group) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(left,Nat.mul(inner,groups)),group),Nat.add(left,Nat.add(Nat.mul(inner,groups),group)),
    Addition.associative(left,Nat.mul(inner,groups),group))

# What a step from row row before end computes: the image and the row in
# it, the rows it takes, which stay in the image and before end, and the
# next row.
type RowStep<-call: Domain.Call,-row: U32,-end: U32> is Data:
  RowStep{image_exact: {U32.to_nat(image_word(Domain.geometry(call),row)) == Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))) : Nat},
    first_exact: {U32.to_nat(first_word(Domain.geometry(call),row)) == Nat.mod(U32.to_nat(row),Domain.height(Domain.geometry(call))) : Nat},
    count_exact: {U32.to_nat(count_word(Domain.geometry(call),row,end)) ==
      Nat.sub(Nat.min(U32.to_nat(end),Nat.mul(Nat.add(Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))),1n),Domain.height(Domain.geometry(call)))),U32.to_nat(row)) : Nat},
    next_exact: {U32.to_nat(next_word(Domain.geometry(call),row,end)) == Nat.add(U32.to_nat(row),U32.to_nat(count_word(Domain.geometry(call),row,end))) : Nat},
    next_le: {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(count_word(Domain.geometry(call),row,end))),U32.to_nat(end)) == True{} : Bool},
    image_inside: {Nat.is_lt(Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))),Domain.batch(call)) == True{} : Bool},
    contained: {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(row),Domain.height(Domain.geometry(call))),U32.to_nat(count_word(Domain.geometry(call),row,end))),
      Domain.height(Domain.geometry(call))) == True{} : Bool}}
def row_image(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {U32.to_nat(image_word(Domain.geometry(call),row)) == Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))) : Nat}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  image_exact
def row_first(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {U32.to_nat(first_word(Domain.geometry(call),row)) == Nat.mod(U32.to_nat(row),Domain.height(Domain.geometry(call))) : Nat}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  first_exact
def row_count(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {U32.to_nat(count_word(Domain.geometry(call),row,end)) ==
    Nat.sub(Nat.min(U32.to_nat(end),Nat.mul(Nat.add(Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))),1n),Domain.height(Domain.geometry(call)))),U32.to_nat(row)) : Nat}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  count_exact
def row_next(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {U32.to_nat(next_word(Domain.geometry(call),row,end)) == Nat.add(U32.to_nat(row),U32.to_nat(count_word(Domain.geometry(call),row,end))) : Nat}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  next_exact
def row_next_le(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(count_word(Domain.geometry(call),row,end))),U32.to_nat(end)) == True{} : Bool}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  next_le
def row_image_inside(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {Nat.is_lt(Nat.div(U32.to_nat(row),Domain.height(Domain.geometry(call))),Domain.batch(call)) == True{} : Bool}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  image_inside
def row_contained(-call: Domain.Call,-row: U32,-end: U32,step: RowStep<call,row,end>) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(row),Domain.height(Domain.geometry(call))),U32.to_nat(count_word(Domain.geometry(call),row,end))),
    Domain.height(Domain.geometry(call))) == True{} : Bool}:
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained} = step
  contained

# A step from row row before end, whose rows lie in the batch of images
# and fit the capacity.
def row_step(+call: Domain.Call,+row: U32,+end: U32,
  +before: {Nat.is_lt(U32.to_nat(row),U32.to_nat(end)) == True{} : Bool},
  +rows_inside: {Nat.is_le(U32.to_nat(end),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  RowStep<call,row,end>:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +cap = Machine.capacity(limits)
  +room = U32.to_nat(cap)
  +r = U32.to_nat(row)
  +e = U32.to_nat(end)
  +height = U32.to_nat(h)
  +positive = Domain.height_positive(whole)
  +one = Order.positive_one(height,positive)
  +word = U32.max(h,1)
  +word_exact = Equal.trans(Nat,U32.to_nat(word),Nat.max(height,1n),height,Extrema.maximum(h,1),
    Equal.trans(Nat,Nat.max(height,1n),Bool.pick(Nat,Nat.is_lt(height,1n),1n,height),height,NatExtrema.maximum_choice(height,1n),
      Equal.cong(Bool,Nat,less => Bool.pick(Nat,less,1n,height),Nat.is_lt(height,1n),False{},Order.not_below(1n,height,one))))
  +height_room = Limits.dimension_capacity(limits,height,Domain.height_room(whole))
  +word_room = {%Equal.sym(Nat,U32.to_nat(word),height,word_exact) : {Nat.is_le(_,room) == True{} : Bool} height_room : {Nat.is_le(U32.to_nat(word),room) == True{} : Bool}}
  +word_positive = {%Equal.sym(Nat,U32.to_nat(word),height,word_exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}}
  +half = Machine.half_word(limits)
  +doubled = Limits.double_bound(limits,U32.to_nat(word),word_room)
  +image_exact = Equal.trans(Nat,U32.to_nat((row / word : U32)),Nat.div(r,U32.to_nat(word)),Nat.div(r,height),
    NativeDivision.divide(row,word,half,Limits.half_word_fits(limits),doubled,word_positive),
    Equal.cong(Nat,Nat,divisor => Nat.div(r,divisor),U32.to_nat(word),height,word_exact))
  +first_exact = Equal.trans(Nat,U32.to_nat((row % word : U32)),Nat.mod(r,U32.to_nat(word)),Nat.mod(r,height),
    NativeDivision.modulo(row,word,half,Limits.half_word_fits(limits),doubled,word_positive),
    Equal.cong(Nat,Nat,divisor => Nat.mod(r,divisor),U32.to_nat(word),height,word_exact))
  +image = Nat.div(r,height)
  +rest = Nat.mod(r,height)
  +below_rows = Bounds.nat_trans(r,e,Nat.mul(images,height),before,rows_inside)
  +image_inside = DivisionOrder.quotient_below(r,images,height,positive,below_rows)
  +split = DivisionOrder.quotient_exact(r,height,positive)
  +rest_small = DivisionOrder.remainder_small(r,height,positive)
  +up = Nat.add(image,1n)
  +up_le = {%Equal.sym(Nat,Nat.add(image,1n),Nat.add(1n,image),Order.commutative(image,1n)) : {Nat.is_le(_,images) == True{} : Bool}
    Order.strict_follow(image,images,image_inside) : {Nat.is_le(up,images) == True{} : Bool}}
  +images_le = {%RowMajor.multiply_commutative(height,images) : {Nat.is_le(images,_) == True{} : Bool}
    DivisionOrder.le_multiple(height,images,positive) : {Nat.is_le(images,Nat.mul(images,height)) == True{} : Bool}}
  +image_word = (row / word : U32)
  +one_bound = {%Equal.sym(Nat,U32.to_nat(image_word),image,image_exact) : {Nat.is_le(Nat.add(_,1n),room) == True{} : Bool}
    Order.transitive(up,images,room,up_le,Order.transitive(images,Nat.mul(images,height),room,images_le,rows_room)) :
    {Nat.is_le(Nat.add(U32.to_nat(image_word),1n),room) == True{} : Bool}}
  +one_word = (image_word + 1 : U32)
  +one_exact = Equal.trans(Nat,U32.to_nat(one_word),Nat.add(U32.to_nat(image_word),1n),up,
    Arithmetic.add(image_word,1,cap,Limits.capacity_fits(limits),one_bound),
    Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(image_word),image,image_exact))
  +limit = Nat.mul(up,height)
  +limit_room = Order.transitive(limit,Nat.mul(images,height),room,RowMajor.mul_le(up,images,height,up_le),rows_room)
  +limit_word = (one_word * word : U32)
  +limit_exact = Equal.trans(Nat,U32.to_nat(limit_word),Nat.mul(U32.to_nat(one_word),U32.to_nat(word)),limit,
    FullRange.multiply(one_word,word,cap,
      {%Equal.sym(Nat,U32.to_nat(one_word),up,one_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(word)),room) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(word),height,word_exact) : {Nat.is_le(Nat.mul(up,_),room) == True{} : Bool}
        limit_room : {Nat.is_le(Nat.mul(U32.to_nat(one_word),U32.to_nat(word)),room) == True{} : Bool}}),
    {%Equal.sym(Nat,U32.to_nat(one_word),up,one_exact) : {Nat.mul(_,U32.to_nat(word)) == limit : Nat}
      %Equal.sym(Nat,U32.to_nat(word),height,word_exact) : {Nat.mul(up,_) == limit : Nat}
      {==} : {Nat.mul(U32.to_nat(one_word),U32.to_nat(word)) == limit : Nat}})
  +upper_split = Equal.trans(Nat,limit,Nat.add(Nat.mul(image,height),Nat.mul(1n,height)),Nat.add(Nat.mul(image,height),height),
    RowMajor.distrib(image,1n,height),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,height),value),Nat.mul(1n,height),height,
      Equal.trans(Nat,Nat.mul(1n,height),Nat.mul(height,1n),height,RowMajor.multiply_commutative(1n,height),Order.times_one(height))))
  +row_below = {%Equal.sym(Nat,limit,Nat.add(Nat.mul(image,height),height),upper_split) : {Nat.is_lt(r,_) == True{} : Bool}
    %Equal.sym(Nat,r,Nat.add(Nat.mul(image,height),rest),split) : {Nat.is_lt(_,Nat.add(Nat.mul(image,height),height)) == True{} : Bool}
    Order.offset_lt(Nat.mul(image,height),rest,height,rest_small) : {Nat.is_lt(r,limit) == True{} : Bool}}
  +minimum = Nat.min(e,limit)
  +row_le = NatExtrema.minimum_lower(e,limit,r,Bounds.strict_le(r,e,before),Bounds.strict_le(r,limit,row_below))
  +minimum_le = NatExtrema.minimum_left(e,limit)
  +minimum_room = Order.transitive(minimum,e,room,minimum_le,Order.transitive(e,Nat.mul(images,height),room,rows_inside,rows_room))
  +minimum_word = U32.min(end,limit_word)
  +minimum_exact = Equal.trans(Nat,U32.to_nat(minimum_word),Nat.min(e,U32.to_nat(limit_word)),minimum,Extrema.minimum(end,limit_word),
    Equal.cong(Nat,Nat,value => Nat.min(e,value),U32.to_nat(limit_word),limit,limit_exact))
  +count_word = (minimum_word - row : U32)
  +count = Nat.sub(minimum,r)
  +count_exact = Equal.trans(Nat,U32.to_nat(count_word),Nat.sub(U32.to_nat(minimum_word),r),count,
    FullRange.subtract(minimum_word,row,cap,
      {%Equal.sym(Nat,U32.to_nat(minimum_word),minimum,minimum_exact) : {Nat.is_le(r,_) == True{} : Bool} row_le : {Nat.is_le(r,U32.to_nat(minimum_word)) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(minimum_word),minimum,minimum_exact) : {Nat.is_le(Nat.sub(_,r),room) == True{} : Bool}
        Order.transitive(count,minimum,room,Order.sub_le(minimum,r),minimum_room) : {Nat.is_le(Nat.sub(U32.to_nat(minimum_word),r),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.sub(value,r),U32.to_nat(minimum_word),minimum,minimum_exact))
  +sum = Order.restore_shift(r,minimum,row_le)
  +sum_exact = {%Equal.sym(Nat,U32.to_nat(count_word),count,count_exact) : {Nat.add(r,_) == minimum : Nat} sum : {Nat.add(r,U32.to_nat(count_word)) == minimum : Nat}}
  +next_exact = Arithmetic.add(row,count_word,cap,Limits.capacity_fits(limits),
    {%Equal.sym(Nat,Nat.add(r,U32.to_nat(count_word)),minimum,sum_exact) : {Nat.is_le(_,room) == True{} : Bool} minimum_room :
      {Nat.is_le(Nat.add(r,U32.to_nat(count_word)),room) == True{} : Bool}})
  +next_le = {%Equal.sym(Nat,Nat.add(r,U32.to_nat(count_word)),minimum,sum_exact) : {Nat.is_le(_,e) == True{} : Bool} minimum_le :
    {Nat.is_le(Nat.add(r,U32.to_nat(count_word)),e) == True{} : Bool}}
  +offset = Nat.mul(image,height)
  +shifted = Equal.trans(Nat,Nat.add(offset,Nat.add(rest,U32.to_nat(count_word))),Nat.add(Nat.add(offset,rest),U32.to_nat(count_word)),minimum,
    Equal.sym(Nat,Nat.add(Nat.add(offset,rest),U32.to_nat(count_word)),Nat.add(offset,Nat.add(rest,U32.to_nat(count_word))),Addition.associative(offset,rest,U32.to_nat(count_word))),
    {%split : {Nat.add(_,U32.to_nat(count_word)) == minimum : Nat} sum_exact : {Nat.add(Nat.add(offset,rest),U32.to_nat(count_word)) == minimum : Nat}})
  +contained = Order.cancel_offset_le(offset,Nat.add(rest,U32.to_nat(count_word)),height,
    {%Equal.sym(Nat,Nat.add(offset,Nat.add(rest,U32.to_nat(count_word))),minimum,shifted) : {Nat.is_le(_,Nat.add(offset,height)) == True{} : Bool}
      %upper_split : {Nat.is_le(minimum,_) == True{} : Bool}
      NatExtrema.minimum_right(e,limit) : {Nat.is_le(Nat.add(offset,Nat.add(rest,U32.to_nat(count_word))),Nat.add(offset,height)) == True{} : Bool}})
  RowStep{image_exact,first_exact,count_exact,next_exact,next_le,image_inside,contained}

# What a step of the loop from row row before the end of the batch computes:
# the step of its rows and the cells of a row.
type Step<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32> is Data:
  Step{rows: RowStep<batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)>,
    size_exact: {U32.to_nat(Kernel.row_cells(batch_geometry(~Holder,~value,batch))) == batch_size(~Holder,~value,batch) : Nat}}
def step_image(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {U32.to_nat(image_word(batch_geometry(~Holder,~value,batch),row)) == Nat.div(U32.to_nat(row),Domain.height(batch_geometry(~Holder,~value,batch))) : Nat}:
  Step{rows,size_exact} = step
  row_image(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_first(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {U32.to_nat(first_word(batch_geometry(~Holder,~value,batch),row)) == Nat.mod(U32.to_nat(row),Domain.height(batch_geometry(~Holder,~value,batch))) : Nat}:
  Step{rows,size_exact} = step
  row_first(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_next(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {U32.to_nat(next_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))) ==
    Nat.add(U32.to_nat(row),U32.to_nat(count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))) : Nat}:
  Step{rows,size_exact} = step
  row_next(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_next_le(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))),
    U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool}:
  Step{rows,size_exact} = step
  row_next_le(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_image_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {Nat.is_lt(Nat.div(U32.to_nat(row),Domain.height(batch_geometry(~Holder,~value,batch))),Domain.batch(batch_call(~Holder,~value,batch))) == True{} : Bool}:
  Step{rows,size_exact} = step
  row_image_inside(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_contained(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(row),Domain.height(batch_geometry(~Holder,~value,batch))),
    U32.to_nat(count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))),Domain.height(batch_geometry(~Holder,~value,batch))) == True{} : Bool}:
  Step{rows,size_exact} = step
  row_contained(batch_call(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),rows)
def step_size(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,step: Step<Holder,value,batch,row>) ->
  {U32.to_nat(Kernel.row_cells(batch_geometry(~Holder,~value,batch))) == batch_size(~Holder,~value,batch) : Nat}:
  Step{rows,size_exact} = step
  size_exact

# The groups of eight channels of a call as a number.
def groups_number(+limits: Machine.Limits,+channels: U32,facts: TiledValues.Groups<limits,channels>) ->
  {U32.to_nat(Grid.groups(channels)) == Ceiling.ceiling(U32.to_nat(channels),8n) : Nat}:
  TiledValues.Groups{exact,fewer,lanes} = facts
  exact

def step_facts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,
  +after: {Nat.is_le(U32.to_nat(batch_start(~Holder,~value,batch)),U32.to_nat(row)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(row),U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool}) -> Step<Holder,value,batch,row>:
  Batch{+holder,+call,+envelope,+fits,+weights,+low,+high,+capacity,+capacity_ok,+bounded,+layout,+start,+end,+length,+ordered,+rows_inside,+rows_room,+cells_room} = batch
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +cap = Machine.capacity(limits)
  +room = U32.to_nat(cap)
  +r = U32.to_nat(row)
  +e = U32.to_nat(end)
  +first = U32.to_nat(start)
  +channels_room = Limits.dimension_capacity(limits,U32.to_nat(c),Domain.channels_room(whole))
  +groups = TiledValues.grouped(limits,c,channels_room)
  +groups_exact = groups_number(limits,c,groups)
  +size = Nat.mul(U32.to_nat(w),Ceiling.ceiling(U32.to_nat(c),8n))
  +spread = Nat.sub(e,first)
  +spread_positive = Order.positive_difference(e,first,Order.le_lt_trans(first,r,e,after,before))
  +size_room = Order.transitive(size,Nat.mul(spread,size),room,DivisionOrder.le_multiple(spread,size,spread_positive),cells_room)
  +size_exact = Equal.trans(Nat,U32.to_nat((w * Grid.groups(c) : U32)),Nat.mul(U32.to_nat(w),U32.to_nat(Grid.groups(c))),size,
    FullRange.multiply(w,Grid.groups(c),cap,
      {%Equal.sym(Nat,U32.to_nat(Grid.groups(c)),Ceiling.ceiling(U32.to_nat(c),8n),groups_exact) : {Nat.is_le(Nat.mul(U32.to_nat(w),_),room) == True{} : Bool}
        size_room : {Nat.is_le(Nat.mul(U32.to_nat(w),U32.to_nat(Grid.groups(c))),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(w),value),U32.to_nat(Grid.groups(c)),Ceiling.ceiling(U32.to_nat(c),8n),groups_exact))
  Step{row_step(whole,row,end,before,rows_inside,rows_room),size_exact}

# ---- The task between its steps ----

# The task at row row with output cell target: the holding it lends bands
# from, the packed weights and the output, whose cells before target hold
# the gradient of the rows from start and whose cells from target to the end
# of the rows of the batch hold +0.
type Progress<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32> is Data:
  Progress{holding: Held.Holding<Tile.Tile8>,
    tiled: Held.Tiled(~Tile.Tile8,holding,batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch)),
    values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,batch_holder(~Holder,~value,batch),holding),
    ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == batch_layout(~Holder,~value,batch) : Held.Spans},
    spare: Held.SpareModel<Tile.Tile8>,spare_room: Held.SpareRoom(~Tile.Tile8,spare),
    tiles: Buffer.Model<Tile.Tile8>,
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(batch_call(~Holder,~value,batch),batch_weights(~Holder,~value,batch))) : Tree.Tree<Tile.Tile8>},
    output: Buffer.Model<Tile.Tile8>,depth_ok: {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool},
    room: {Nat.is_le(batch_cells(~Holder,~value,batch),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool},
    length_exact: {Buffer.length(~Tile.Tile8,output) == batch_length(~Holder,~value,batch) : Nat},
    after: {Nat.is_le(U32.to_nat(batch_start(~Holder,~value,batch)),U32.to_nat(row)) == True{} : Bool},
    before: {Nat.is_le(U32.to_nat(row),U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool},
    target_exact: {U32.to_nat(target) == Nat.mul(Nat.sub(U32.to_nat(row),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch)) : Nat},
    written: Members.All(Nat,{cell => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},
      Ranges.range(Nat.mul(Nat.sub(U32.to_nat(row),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch)),0n)),
    zero: Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(U32.to_nat(target),cell))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},
      Ranges.range(Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(row)),batch_size(~Holder,~value,batch)),0n))}
def progress_holding(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Held.Holding<Tile.Tile8>:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  holding
def progress_tiled(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Held.Tiled(~Tile.Tile8,progress_holding(~Holder,~value,batch,row,target,progress),batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch)):
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  tiled
def progress_values(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Held.HoldingValues(~Tile.Tile8,~Holder,~value,batch_holder(~Holder,~value,batch),progress_holding(~Holder,~value,batch,row,target,progress)):
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  values
def progress_ordered(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,progress_holding(~Holder,~value,batch,row,target,progress))) == batch_layout(~Holder,~value,batch) : Held.Spans}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  ordered
def progress_spare(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Held.SpareModel<Tile.Tile8>:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  spare
def progress_spare_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Held.SpareRoom(~Tile.Tile8,progress_spare(~Holder,~value,batch,row,target,progress)):
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  spare_room
def progress_tiles(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Buffer.Model<Tile.Tile8>:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  tiles
def progress_tiles_exact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Buffer.tree(~Tile.Tile8,progress_tiles(~Holder,~value,batch,row,target,progress)) ==
    Rows.native_tree(Rows.packing(batch_call(~Holder,~value,batch),batch_weights(~Holder,~value,batch))) : Tree.Tree<Tile.Tile8>}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  tiles_exact
def progress_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Buffer.Model<Tile.Tile8>:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  output
def progress_depth_ok(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Nat.is_le(Buffer.depth(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress)),30n) == True{} : Bool}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  depth_ok
def progress_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Nat.is_le(batch_cells(~Holder,~value,batch),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress))))) == True{} : Bool}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  room
def progress_length(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Buffer.length(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress)) == batch_length(~Holder,~value,batch) : Nat}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  length_exact
def progress_after(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {Nat.is_le(U32.to_nat(batch_start(~Holder,~value,batch)),U32.to_nat(row)) == True{} : Bool}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  after
def progress_target(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  {U32.to_nat(target) == Nat.mul(Nat.sub(U32.to_nat(row),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch)) : Nat}:
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  target_exact
def progress_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Members.All(Nat,{cell => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress)),cell) : Nat -> Data},
    Ranges.range(Nat.mul(Nat.sub(U32.to_nat(row),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch)),0n)):
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  written
def progress_zero(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,progress: Progress<Holder,value,batch,row,target>) ->
  Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress)),U32.from_nat(Nat.add(U32.to_nat(target),cell))) ==
      Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},
    Ranges.range(Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(row)),batch_size(~Holder,~value,batch)),0n)):
  Progress{holding,tiled,values,ordered,spare,spare_room,tiles,tiles_exact,output,depth_ok,room,length_exact,after,before,target_exact,written,zero} = progress
  zero

# The task and the rows the runtime holds for it.
type Reached<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,-target: U32,-state: Kernel.Rows> is Data:
  Reached{progress: Progress<Holder,value,batch,row,target>,
    packed: {state == Kernel.Rows{Held.pack(~Tile.Tile8,progress_holding(~Holder,~value,batch,row,target,progress)),Held.pack_spare(~Tile.Tile8,progress_spare(~Holder,~value,batch,row,target,progress)),
      Buffer.pack(~Tile.Tile8,progress_tiles(~Holder,~value,batch,row,target,progress)),Buffer.pack(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress))} : Kernel.Rows}}

# What a step needs from the task: its row comes before the end, and its
# band, if it holds cells, lies in the holding.
type CallFact<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32> is Data:
  CallFact{before: {Nat.is_lt(U32.to_nat(row),U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool},
    placed: BandSource.Placed<batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch),
      Rows.band_first(batch_geometry(~Holder,~value,batch),image_word(batch_geometry(~Holder,~value,batch),row),first_word(batch_geometry(~Holder,~value,batch),row),
        count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))),
      Rows.band_cells(batch_geometry(~Holder,~value,batch),first_word(batch_geometry(~Holder,~value,batch),row),count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))>}
def fact_before(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,fact: CallFact<Holder,value,batch,row>) ->
  {Nat.is_lt(U32.to_nat(row),U32.to_nat(batch_end(~Holder,~value,batch))) == True{} : Bool}:
  CallFact{before,placed} = fact
  before
def fact_placed(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-row: U32,fact: CallFact<Holder,value,batch,row>) ->
  BandSource.Placed<batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch),
    Rows.band_first(batch_geometry(~Holder,~value,batch),image_word(batch_geometry(~Holder,~value,batch),row),first_word(batch_geometry(~Holder,~value,batch),row),
      count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))),
    Rows.band_cells(batch_geometry(~Holder,~value,batch),first_word(batch_geometry(~Holder,~value,batch),row),count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))>:
  CallFact{before,placed} = fact
  placed
# The facts of the remaining steps, the last of which ends at the end.
def CallFacts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,remaining: Nat,+batch: Batch<Holder,value>,+row: U32) -> Data:
  match remaining:
    case 0n: {U32.to_nat(row) == U32.to_nat(batch_end(~Holder,~value,batch)) : Nat}
    case 1n+rest:
      KernelGemm.Both<CallFact<Holder,value,batch,row>,
        CallFacts(~Holder,~value,rest,batch,next_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))>

# ---- One step ----

def step_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) -> Step<Holder,value,batch,row>:
  step_facts(~Holder,~value,batch,row,progress_after(~Holder,~value,batch,row,target,progress),fact_before(~Holder,~value,batch,row,fact))

# The rows of a step are the cells count * size after cell target.
def region_exact(+call: Domain.Call,+count: U32) -> {Rows.region_cells(call,count) == Nat.mul(U32.to_nat(count),row_size(call)) : Nat}:
  RowMajor.assoc(U32.to_nat(count),Domain.width(Domain.geometry(call)),Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n))

def call_image_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) ->
  {Nat.is_lt(U32.to_nat(image_word(batch_geometry(~Holder,~value,batch),row)),Domain.batch(batch_call(~Holder,~value,batch))) == True{} : Bool}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  %Equal.sym(Nat,U32.to_nat(image_word(batch_geometry(~Holder,~value,batch),row)),Nat.div(U32.to_nat(row),Domain.height(batch_geometry(~Holder,~value,batch))),
      step_image(~Holder,~value,batch,row,step)) :
    {Nat.is_lt(_,Domain.batch(batch_call(~Holder,~value,batch))) == True{} : Bool}
  step_image_inside(~Holder,~value,batch,row,step)
def call_contained(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first_word(batch_geometry(~Holder,~value,batch),row)),U32.to_nat(count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))),
    Domain.height(batch_geometry(~Holder,~value,batch))) == True{} : Bool}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +geometry = batch_geometry(~Holder,~value,batch)
  +count = U32.to_nat(count_word(geometry,row,batch_end(~Holder,~value,batch)))
  %Equal.sym(Nat,U32.to_nat(first_word(geometry,row)),Nat.mod(U32.to_nat(row),Domain.height(geometry)),step_first(~Holder,~value,batch,row,step)) :
    {Nat.is_le(Nat.add(_,count),Domain.height(geometry)) == True{} : Bool}
  step_contained(~Holder,~value,batch,row,step)

# The rows of a step fit the output and the rows left.
def call_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) ->
  {Nat.is_le(Nat.add(U32.to_nat(target),Rows.region_cells(batch_call(~Holder,~value,batch),count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))),
    U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress))))) == True{} : Bool}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +counted = count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +size = row_size(call)
  +offset = Nat.sub(r,start)
  +rest = Nat.sub(e,Nat.add(r,count))
  +capacity = U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress))))
  +fits = step_next_le(~Holder,~value,batch,row,step)
  %Equal.sym(Nat,Rows.region_cells(call,counted),Nat.mul(count,size),region_exact(call,counted)) : {Nat.is_le(Nat.add(U32.to_nat(target),_),capacity) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(target),Nat.mul(offset,size),progress_target(~Holder,~value,batch,row,target,progress)) : {Nat.is_le(Nat.add(_,Nat.mul(count,size)),capacity) == True{} : Bool}
  %RowMajor.distrib(offset,count,size) : {Nat.is_le(_,capacity) == True{} : Bool}
  Order.transitive(Nat.mul(Nat.add(offset,count),size),Nat.mul(Nat.sub(e,start),size),capacity,
    {%Equal.sym(Nat,Nat.sub(e,start),Nat.add(offset,Nat.add(count,rest)),whole_split(start,r,count,e,progress_after(~Holder,~value,batch,row,target,progress),fits)) :
        {Nat.is_le(Nat.mul(Nat.add(offset,count),size),Nat.mul(_,size)) == True{} : Bool}
      RowMajor.mul_le(Nat.add(offset,count),Nat.add(offset,Nat.add(count,rest)),size,Order.offset_le(offset,count,Nat.add(count,rest),Order.le_add(count,rest))) :
      {Nat.is_le(Nat.mul(Nat.add(offset,count),size),Nat.mul(Nat.sub(e,start),size)) == True{} : Bool}},
    progress_room(~Holder,~value,batch,row,target,progress))
def call_zeros_cover(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) ->
  {Nat.is_le(Rows.region_cells(batch_call(~Holder,~value,batch),count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))),
    Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(row)),row_size(batch_call(~Holder,~value,batch)))) == True{} : Bool}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +counted = count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +size = row_size(call)
  +rest = Nat.sub(e,Nat.add(r,count))
  %Equal.sym(Nat,Rows.region_cells(call,counted),Nat.mul(count,size),region_exact(call,counted)) : {Nat.is_le(_,Nat.mul(Nat.sub(e,r),size)) == True{} : Bool}
  %Equal.sym(Nat,Nat.sub(e,r),Nat.add(count,rest),rest_split(r,count,e,step_next_le(~Holder,~value,batch,row,step))) : {Nat.is_le(Nat.mul(count,size),Nat.mul(_,size)) == True{} : Bool}
  RowMajor.mul_le(count,Nat.add(count,rest),size,Order.le_add(count,rest))

# The call of a step.
def step_record(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) -> Rows.Call<Holder,value>:
  +call = batch_call(~Holder,~value,batch)
  +geometry = Domain.geometry(call)
  +end = batch_end(~Holder,~value,batch)
  Rows.Call{Rows.Lender{batch_holder(~Holder,~value,batch),progress_holding(~Holder,~value,batch,row,target,progress),batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch),
      batch_capacity(~Holder,~value,batch),batch_capacity_ok(~Holder,~value,batch),batch_bounded(~Holder,~value,batch),
      progress_tiled(~Holder,~value,batch,row,target,progress),progress_values(~Holder,~value,batch,row,target,progress),batch_layout(~Holder,~value,batch),
      progress_ordered(~Holder,~value,batch,row,target,progress),progress_spare(~Holder,~value,batch,row,target,progress),progress_spare_room(~Holder,~value,batch,row,target,progress)},
    call,batch_envelope(~Holder,~value,batch),batch_fits(~Holder,~value,batch),batch_weights(~Holder,~value,batch),
    progress_tiles(~Holder,~value,batch,row,target,progress),progress_tiles_exact(~Holder,~value,batch,row,target,progress),
    image_word(geometry,row),call_image_inside(~Holder,~value,batch,row,target,progress,fact),first_word(geometry,row),count_word(geometry,row,end),
    call_contained(~Holder,~value,batch,row,target,progress,fact),target,progress_output(~Holder,~value,batch,row,target,progress),
    progress_depth_ok(~Holder,~value,batch,row,target,progress),call_room(~Holder,~value,batch,row,target,progress,fact),
    Nat.mul(Nat.sub(U32.to_nat(end),U32.to_nat(row)),row_size(call)),call_zeros_cover(~Holder,~value,batch,row,target,progress,fact),
    progress_zero(~Holder,~value,batch,row,target,progress),fact_placed(~Holder,~value,batch,row,fact)}

# The output cell after a step.
def target_exact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) ->
  {U32.to_nat(target_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),target)) ==
    Nat.add(U32.to_nat(target),Nat.mul(U32.to_nat(count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))),batch_size(~Holder,~value,batch))) : Nat}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +limits = Domain.limits(call)
  +geometry = Domain.geometry(call)
  +counted = count_word(geometry,row,batch_end(~Holder,~value,batch))
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +size = row_size(call)
  +offset = Nat.sub(r,start)
  +rest = Nat.sub(e,Nat.add(r,count))
  +room = U32.to_nat(Machine.capacity(limits))
  +whole = whole_split(start,r,count,e,progress_after(~Holder,~value,batch,row,target,progress),step_next_le(~Holder,~value,batch,row,step))
  +cells_room = {%whole : {Nat.is_le(Nat.mul(_,size),room) == True{} : Bool}
    batch_cells_room(~Holder,~value,batch) : {Nat.is_le(Nat.mul(Nat.add(offset,Nat.add(count,rest)),size),room) == True{} : Bool}}
  +sum_room = Order.transitive(Nat.mul(Nat.add(offset,count),size),Nat.mul(Nat.add(offset,Nat.add(count,rest)),size),room,
    RowMajor.mul_le(Nat.add(offset,count),Nat.add(offset,Nat.add(count,rest)),size,Order.offset_le(offset,count,Nat.add(count,rest),Order.le_add(count,rest))),cells_room)
  +count_room = Order.transitive(Nat.mul(count,size),Nat.mul(Nat.add(offset,count),size),room,RowMajor.mul_le(count,Nat.add(offset,count),size,Order.right_le(offset,count)),sum_room)
  +cells = Kernel.row_cells(geometry)
  +size_exact = step_size(~Holder,~value,batch,row,step)
  +product = (counted * cells : U32)
  +product_exact = Equal.trans(Nat,U32.to_nat(product),Nat.mul(count,U32.to_nat(cells)),Nat.mul(count,size),
    FullRange.multiply(counted,cells,Machine.capacity(limits),
      {%Equal.sym(Nat,U32.to_nat(cells),size,size_exact) : {Nat.is_le(Nat.mul(count,_),room) == True{} : Bool} count_room : {Nat.is_le(Nat.mul(count,U32.to_nat(cells)),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.mul(count,value),U32.to_nat(cells),size,size_exact))
  Equal.trans(Nat,U32.to_nat((target + product : U32)),Nat.add(U32.to_nat(target),U32.to_nat(product)),Nat.add(U32.to_nat(target),Nat.mul(count,size)),
    Arithmetic.add(target,product,Machine.capacity(limits),Limits.capacity_fits(limits),
      {%Equal.sym(Nat,U32.to_nat(product),Nat.mul(count,size),product_exact) : {Nat.is_le(Nat.add(U32.to_nat(target),_),room) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(target),Nat.mul(offset,size),progress_target(~Holder,~value,batch,row,target,progress)) : {Nat.is_le(Nat.add(_,Nat.mul(count,size)),room) == True{} : Bool}
        %RowMajor.distrib(offset,count,size) : {Nat.is_le(_,room) == True{} : Bool}
        sum_room : {Nat.is_le(Nat.add(U32.to_nat(target),U32.to_nat(product)),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(target),value),U32.to_nat(product),Nat.mul(count,size),product_exact))

# ---- The cells after a step ----

# The cells of the rows from start to a row, the next row and output cell of
# a step, and the cells written before and left +0 after the next row.
def written_rows(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32) -> Nat:
  Nat.mul(Nat.sub(U32.to_nat(row),U32.to_nat(batch_start(~Holder,~value,batch))),batch_size(~Holder,~value,batch))
def next_row(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32) -> U32:
  next_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))
def next_target(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32) -> U32:
  target_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),target)
def written_next(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32) -> Nat:
  written_rows(~Holder,~value,batch,next_row(~Holder,~value,batch,row))
def zero_next(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32) -> Nat:
  Nat.mul(Nat.sub(U32.to_nat(batch_end(~Holder,~value,batch)),U32.to_nat(next_row(~Holder,~value,batch,row))),batch_size(~Holder,~value,batch))
# The output after the call of a step.
def next_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>) -> Buffer.Model<Tile.Tile8>:
  Execution.part_output(Rows.record_part(~Holder,~value,step_record(~Holder,~value,batch,row,target,progress,fact)))

# A cell before the rows of a step keeps the gradient it held.
def old_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,+cell: Nat,
  +before: {Nat.is_lt(cell,written_rows(~Holder,~value,batch,row)) == True{} : Bool}) ->
  WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),cell):
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +record = step_record(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +counted = count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))
  +output = progress_output(~Holder,~value,batch,row,target,progress)
  +groups = batch_groups(~Holder,~value,batch)
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +size = batch_size(~Holder,~value,batch)
  +offset = Nat.sub(r,start)
  +rest = Nat.sub(e,Nat.add(r,count))
  +base = Nat.mul(offset,size)
  +capacity = U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))
  +before_target = {%Equal.sym(Nat,U32.to_nat(target),base,progress_target(~Holder,~value,batch,row,target,progress)) : {Nat.is_lt(cell,_) == True{} : Bool}
    before : {Nat.is_lt(cell,U32.to_nat(target)) == True{} : Bool}}
  +base_le = {%Equal.sym(Nat,Nat.sub(e,start),Nat.add(offset,Nat.add(count,rest)),
        whole_split(start,r,count,e,progress_after(~Holder,~value,batch,row,target,progress),step_next_le(~Holder,~value,batch,row,step))) :
      {Nat.is_le(base,Nat.mul(_,size)) == True{} : Bool}
    RowMajor.mul_le(offset,Nat.add(offset,Nat.add(count,rest)),size,Order.le_add(offset,Nat.add(count,rest))) :
    {Nat.is_le(base,Nat.mul(Nat.sub(e,start),size)) == True{} : Bool}}
  +inside = Bounds.nat_trans(cell,base,capacity,before,
    Order.transitive(base,batch_cells(~Holder,~value,batch),capacity,base_le,progress_room(~Holder,~value,batch,row,target,progress)))
  +outside = {Rows.Before{before_target} : Rows.Outside<U32.to_nat(target),Rows.region_cells(call,counted),cell>}
  +kept = Rows.record_kept(~Holder,~value,record,cell,outside,inside)
  +old = Members.all_member(Nat,{index => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),index) : Nat -> Data},cell,Ranges.range(base,0n),
    progress_written(~Holder,~value,batch,row,target,progress),Members.range_member(base,0n,cell,before))
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),U32.from_nat(cell)),
      Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(cell)),kept) :
    PaddedCells.Valid<Domain.channels(batch_geometry(~Holder,~value,batch)),Nat.mod(cell,groups),_,
      written_tile(~Holder,~value,batch,Nat.div(cell,groups),Nat.mod(cell,groups))>
  old

# A lane below the channels of a cell of the rows of a step holds the input
# gradient of its pixel and channel.
def new_lane_value(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,+cell: Nat,+lane: Nat,
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(cell,batch_groups(~Holder,~value,batch)),8n)),Domain.channels(batch_geometry(~Holder,~value,batch))) == True{} : Bool},
  +lower: {Nat.is_le(written_rows(~Holder,~value,batch,row),cell) == True{} : Bool},
  +upper: {Nat.is_lt(cell,written_next(~Holder,~value,batch,row)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),U32.from_nat(cell))) ==
    pixel_value(~Holder,~value,batch,Nat.div(cell,batch_groups(~Holder,~value,batch)),Nat.add(lane,Nat.mul(Nat.mod(cell,batch_groups(~Holder,~value,batch)),8n))) : F32}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +record = step_record(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +geometry = batch_geometry(~Holder,~value,batch)
  +width = Domain.width(geometry)
  +height = Domain.height(geometry)
  +channels = Domain.channels(geometry)
  +groups = batch_groups(~Holder,~value,batch)
  +size = batch_size(~Holder,~value,batch)
  +tree = Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact))
  +reader = Sources.models_reader(geometry,batch_model(~Holder,~value,batch),batch_weights(~Holder,~value,batch))
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +counted = count_word(geometry,row,batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +offset = Nat.sub(r,start)
  +base = Nat.mul(offset,size)
  +j = Nat.sub(cell,base)
  +g = Nat.mod(j,groups)
  +p = Nat.div(j,groups)
  +jr = Nat.div(p,width)
  +x = Nat.mod(p,width)
  +q = Nat.add(offset,jr)
  +pixel = Nat.add(Nat.mul(q,width),x)
  +channel = Nat.add(lane,Nat.mul(g,8n))
  +first = U32.to_nat(first_word(geometry,row))
  +image = U32.to_nat(image_word(geometry,row))
  +after = progress_after(~Holder,~value,batch,row,target,progress)
  +width_positive = Domain.width_positive(call)
  +height_positive = Domain.height_positive(call)
  +groups_positive = Equal.trans(Bool,Nat.is_lt(0n,groups),Nat.is_lt(0n,channels),True{},Ceiling.upper_comparison(channels,8n,0n,{==}),Domain.channels_positive(call))
  # The cell is cell j of the rows of the step: pixel p of them and group g.
  +upper_split = {%RowMajor.distrib(offset,count,size) : {Nat.is_lt(cell,_) == True{} : Bool}
    %shifted_sum(start,r,count,after) : {Nat.is_lt(cell,Nat.mul(_,size)) == True{} : Bool}
    %step_next(~Holder,~value,batch,row,step) : {Nat.is_lt(cell,Nat.mul(Nat.sub(_,start),size)) == True{} : Bool}
    upper : {Nat.is_lt(cell,Nat.add(base,Nat.mul(count,size))) == True{} : Bool}}
  +j_exact = Order.restore_shift(base,cell,lower)
  +j_less = Order.cancel_offset_lt(base,j,Nat.mul(count,size),
    {%Equal.sym(Nat,Nat.add(base,j),cell,j_exact) : {Nat.is_lt(_,Nat.add(base,Nat.mul(count,size))) == True{} : Bool}
      upper_split : {Nat.is_lt(Nat.add(base,j),Nat.add(base,Nat.mul(count,size))) == True{} : Bool}})
  +j_grouped = {%Equal.sym(Nat,Nat.mul(Nat.mul(count,width),groups),Nat.mul(count,size),RowMajor.assoc(count,width,groups)) : {Nat.is_lt(j,_) == True{} : Bool}
    j_less : {Nat.is_lt(j,Nat.mul(Nat.mul(count,width),groups)) == True{} : Bool}}
  +p_less = DivisionOrder.quotient_below(j,Nat.mul(count,width),groups,groups_positive,j_grouped)
  +jr_less = DivisionOrder.quotient_below(p,count,width,width_positive,p_less)
  +x_less = DivisionOrder.remainder_small(p,width,width_positive)
  +g_less = DivisionOrder.remainder_small(j,groups,groups_positive)
  +j_split = DivisionOrder.quotient_exact(j,groups,groups_positive)
  +p_split = DivisionOrder.quotient_exact(p,width,width_positive)
  +cell_whole = {%cell_split(offset,jr,width,x,groups,g) : {cell == _ : Nat}
    %p_split : {cell == Nat.add(base,Nat.add(Nat.mul(_,groups),g)) : Nat}
    %j_split : {cell == Nat.add(base,_) : Nat}
    Equal.sym(Nat,Nat.add(base,j),cell,j_exact) : {cell == Nat.add(Nat.mul(pixel,groups),g) : Nat}}
  +cell_pair = Division.unique(cell,groups,pixel,g,cell_whole,groups_positive,g_less)
  +cell_pixel = {Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(cell,groups),(pixel,g),cell_pair) : {Nat.div(cell,groups) == pixel : Nat}}
  +cell_group = {Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(cell,groups),(pixel,g),cell_pair) : {Nat.mod(cell,groups) == g : Nat}}
  +pixel_pair = Division.unique(pixel,width,q,x,{==},width_positive,x_less)
  +pixel_row = {Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(pixel,width),(q,x),pixel_pair) : {Nat.div(pixel,width) == q : Nat}}
  +pixel_column = {Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(pixel,width),(q,x),pixel_pair) : {Nat.mod(pixel,width) == x : Nat}}
  +channel_pair = Division.unique(channel,8n,g,lane,Order.commutative(lane,Nat.mul(g,8n)),{==},lane_small)
  +channel_group = {Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(channel,8n),(g,lane),channel_pair) : {Nat.div(channel,8n) == g : Nat}}
  +channel_lane = {Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(channel,8n),(g,lane),channel_pair) : {Nat.mod(channel,8n) == lane : Nat}}
  +channel_ok = {%cell_group : {Nat.is_lt(Nat.add(lane,Nat.mul(_,8n)),channels) == True{} : Bool} channel_inside : {Nat.is_lt(channel,channels) == True{} : Bool}}
  # The leaf writes the lane at the cell.
  +address = Nat.add(U32.to_nat(target),Rows.output_offset(geometry,Nat.sub(Nat.add(first,jr),first),x,channel))
  +address_exact = {%Equal.sym(Nat,cell,Nat.add(Nat.mul(pixel,groups),g),cell_whole) : {address == _ : Nat}
    %cell_split(offset,jr,width,x,groups,g) : {address == _ : Nat}
    %Equal.sym(Nat,U32.to_nat(target),base,progress_target(~Holder,~value,batch,row,target,progress)) :
      {Nat.add(_,Rows.output_offset(geometry,Nat.sub(Nat.add(first,jr),first),x,channel)) == Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.mul(jr,width),x),groups),g)) : Nat}
    %Equal.sym(Nat,Nat.sub(Nat.add(first,jr),first),jr,Order.subtract_shift(first,jr)) :
      {Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.mul(_,width),x),groups),Nat.div(channel,8n))) == Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.mul(jr,width),x),groups),g)) : Nat}
    %Equal.sym(Nat,Nat.div(channel,8n),g,channel_group) :
      {Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.mul(jr,width),x),groups),_)) == Nat.add(base,Nat.add(Nat.mul(Nat.add(Nat.mul(jr,width),x),groups),g)) : Nat}
    {==} : {address == cell : Nat}}
  +spec_image = Specification.value(Sources.problem(geometry,reader,image),channel,Nat.add(first,jr),x)
  +lane_cell = {%address_exact : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(_))) == spec_image : F32}
    %channel_lane : {TileLaws.select_lane(_,Tree.read(Tile.Tile8,tree,U32.from_nat(address))) == spec_image : F32}
    Rows.record_lane(~Holder,~value,record,channel,Nat.add(first,jr),x,channel_ok,Order.le_add(first,jr),Order.offset_lt(first,jr,count,jr_less),x_less) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) == spec_image : F32}}
  # The pixel lies at row r + jr of the batch: row first + jr of the image.
  +sum = Nat.add(r,jr)
  +start_sum = Equal.trans(Nat,Nat.add(start,q),Nat.add(Nat.add(start,offset),jr),sum,
    Equal.sym(Nat,Nat.add(Nat.add(start,offset),jr),Nat.add(start,q),Addition.associative(start,offset,jr)),
    Equal.cong(Nat,Nat,value => Nat.add(value,jr),Nat.add(start,offset),r,Order.restore_shift(start,r,after)))
  +sum_split = {%Addition.associative(Nat.mul(Nat.div(r,height),height),Nat.mod(r,height),jr) : {sum == _ : Nat}
    Equal.cong(Nat,Nat,value => Nat.add(value,jr),r,Nat.add(Nat.mul(Nat.div(r,height),height),Nat.mod(r,height)),DivisionOrder.quotient_exact(r,height,height_positive)) :
    {sum == Nat.add(Nat.mul(Nat.div(r,height),height),Nat.add(Nat.mod(r,height),jr)) : Nat}}
  +sum_small = Bounds.nat_trans(Nat.add(Nat.mod(r,height),jr),Nat.add(Nat.mod(r,height),count),height,
    Order.offset_lt(Nat.mod(r,height),jr,count,jr_less),step_contained(~Holder,~value,batch,row,step))
  +row_pair = Division.unique(sum,height,Nat.div(r,height),Nat.add(Nat.mod(r,height),jr),sum_split,height_positive,sum_small)
  +pv = pixel_value(~Holder,~value,batch,pixel,channel)
  +s1 = Specification.value(Sources.problem(geometry,reader,Nat.div(Nat.add(start,q),height)),channel,Nat.mod(Nat.add(start,q),height),Nat.mod(pixel,width))
  +s2 = Specification.value(Sources.problem(geometry,reader,Nat.div(Nat.add(start,q),height)),channel,Nat.mod(Nat.add(start,q),height),x)
  +s3 = Specification.value(Sources.problem(geometry,reader,Nat.div(sum,height)),channel,Nat.mod(sum,height),x)
  +s4 = Specification.value(Sources.problem(geometry,reader,Nat.div(r,height)),channel,Nat.add(Nat.mod(r,height),jr),x)
  +s5 = Specification.value(Sources.problem(geometry,reader,image),channel,Nat.add(Nat.mod(r,height),jr),x)
  +spec_exact = Equal.trans(F32,pv,s1,spec_image,
    Equal.cong(Nat,F32,value => Specification.value(Sources.problem(geometry,reader,Nat.div(Nat.add(start,value),height)),channel,Nat.mod(Nat.add(start,value),height),Nat.mod(pixel,width)),
      Nat.div(pixel,width),q,pixel_row),
    Equal.trans(F32,s1,s2,spec_image,
      Equal.cong(Nat,F32,value => Specification.value(Sources.problem(geometry,reader,Nat.div(Nat.add(start,q),height)),channel,Nat.mod(Nat.add(start,q),height),value),
        Nat.mod(pixel,width),x,pixel_column),
      Equal.trans(F32,s2,s3,spec_image,
        Equal.cong(Nat,F32,value => Specification.value(Sources.problem(geometry,reader,Nat.div(value,height)),channel,Nat.mod(value,height),x),Nat.add(start,q),sum,start_sum),
        Equal.trans(F32,s3,s4,spec_image,
          Equal.cong(Nat & Nat,F32,pair => Specification.value(Sources.problem(geometry,reader,Pair.fst(Nat,Nat,pair)),channel,Pair.snd(Nat,Nat,pair),x),
            Nat.divmod(sum,height),(Nat.div(r,height),Nat.add(Nat.mod(r,height),jr)),row_pair),
          Equal.trans(F32,s4,s5,spec_image,
            Equal.cong(Nat,F32,value => Specification.value(Sources.problem(geometry,reader,value),channel,Nat.add(Nat.mod(r,height),jr),x),Nat.div(r,height),image,
              Equal.sym(Nat,image,Nat.div(r,height),step_image(~Holder,~value,batch,row,step))),
            Equal.cong(Nat,F32,value => Specification.value(Sources.problem(geometry,reader,image),channel,Nat.add(value,jr),x),Nat.mod(r,height),first,
              Equal.sym(Nat,first,Nat.mod(r,height),step_first(~Holder,~value,batch,row,step))))))))
  %Equal.sym(Nat,Nat.mod(cell,groups),g,cell_group) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) == pixel_value(~Holder,~value,batch,Nat.div(cell,groups),Nat.add(lane,Nat.mul(_,8n))) : F32}
  %Equal.sym(Nat,Nat.div(cell,groups),pixel,cell_pixel) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))) == pixel_value(~Holder,~value,batch,_,channel) : F32}
  Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(cell))),spec_image,pv,lane_cell,Equal.sym(F32,pv,spec_image,spec_exact))

# A cell of the rows of a step holds the gradient of its pixel and group.
def new_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,+cell: Nat,
  +lower: {Nat.is_le(written_rows(~Holder,~value,batch,row),cell) == True{} : Bool},
  +upper: {Nat.is_lt(cell,written_next(~Holder,~value,batch,row)) == True{} : Bool}) ->
  WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),cell):
  +channels = Domain.channels(batch_geometry(~Holder,~value,batch))
  +groups = batch_groups(~Holder,~value,batch)
  +group = Nat.mod(cell,groups)
  +pixel = Nat.div(cell,groups)
  +tile = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),U32.from_nat(cell))
  PaddedCells.Valid{
    PaddedCells.kept_agree(channels,group,0n,Tile.lane0(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(0n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,0n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,1n,Tile.lane1(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(1n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,1n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,2n,Tile.lane2(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(2n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,2n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,3n,Tile.lane3(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(3n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,3n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,4n,Tile.lane4(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(4n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,4n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,5n,Tile.lane5(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(5n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,5n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,6n,Tile.lane6(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(6n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,6n,{==},inside,lower,upper)),
    PaddedCells.kept_agree(channels,group,7n,Tile.lane7(tile),pixel_value(~Holder,~value,batch,pixel,Nat.add(7n,Nat.mul(group,8n))),
      Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),channels),{==},inside => new_lane_value(~Holder,~value,batch,row,target,progress,fact,cell,7n,{==},inside,lower,upper))}

def written_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,+cell: Nat,below: Bool,
  +below_exact: {Nat.is_lt(cell,written_rows(~Holder,~value,batch,row)) == below : Bool},
  +inside: {Nat.is_lt(cell,written_next(~Holder,~value,batch,row)) == True{} : Bool}) ->
  WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),cell):
  match below:
    case True{}: old_cell(~Holder,~value,batch,row,target,progress,fact,cell,below_exact)
    case False{}: new_cell(~Holder,~value,batch,row,target,progress,fact,cell,Order.not_less(cell,written_rows(~Holder,~value,batch,row),below_exact),inside)

# A cell after the rows of a step keeps its +0.
def zero_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,+cell: Nat,
  +inside: {Nat.is_lt(cell,zero_next(~Holder,~value,batch,row)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next_output(~Holder,~value,batch,row,target,progress,fact)),
    U32.from_nat(Nat.add(U32.to_nat(next_target(~Holder,~value,batch,row,target)),cell))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8}:
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +record = step_record(~Holder,~value,batch,row,target,progress,fact)
  +call = batch_call(~Holder,~value,batch)
  +counted = count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch))
  +output = progress_output(~Holder,~value,batch,row,target,progress)
  +next = next_output(~Holder,~value,batch,row,target,progress,fact)
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +t = U32.to_nat(target)
  +count = U32.to_nat(counted)
  +size = batch_size(~Holder,~value,batch)
  +offset = Nat.sub(r,start)
  +rest = Nat.sub(e,Nat.add(r,count))
  +moved = Nat.mul(count,size)
  +k = Nat.add(moved,cell)
  +capacity = U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))
  +fits = step_next_le(~Holder,~value,batch,row,step)
  +rest_less = {%step_next(~Holder,~value,batch,row,step) : {Nat.is_lt(cell,Nat.mul(Nat.sub(e,_),size)) == True{} : Bool}
    inside : {Nat.is_lt(cell,Nat.mul(rest,size)) == True{} : Bool}}
  +k_less = {%Equal.sym(Nat,Nat.sub(e,r),Nat.add(count,rest),rest_split(r,count,e,fits)) : {Nat.is_lt(k,Nat.mul(_,size)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.add(count,rest),size),Nat.add(moved,Nat.mul(rest,size)),RowMajor.distrib(count,rest,size)) : {Nat.is_lt(k,_) == True{} : Bool}
    Order.offset_lt(moved,cell,Nat.mul(rest,size),rest_less) : {Nat.is_lt(k,Nat.mul(Nat.sub(e,r),size)) == True{} : Bool}}
  +old = Members.all_member(Nat,
    {index => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(t,index))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},
    k,Ranges.range(Nat.mul(Nat.sub(e,r),size),0n),progress_zero(~Holder,~value,batch,row,target,progress),Members.range_member(Nat.mul(Nat.sub(e,r),size),0n,k,k_less))
  +beyond = {%Equal.sym(Nat,Rows.region_cells(call,counted),moved,region_exact(call,counted)) : {Nat.is_le(Nat.add(t,_),Nat.add(t,k)) == True{} : Bool}
    Order.offset_le(t,moved,k,Order.le_add(moved,cell)) : {Nat.is_le(Nat.add(t,Rows.region_cells(call,counted)),Nat.add(t,k)) == True{} : Bool}}
  +span = {%Equal.sym(Nat,t,Nat.mul(offset,size),progress_target(~Holder,~value,batch,row,target,progress)) :
      {Nat.is_le(Nat.add(_,Nat.mul(Nat.sub(e,r),size)),capacity) == True{} : Bool}
    %RowMajor.distrib(offset,Nat.sub(e,r),size) : {Nat.is_le(_,capacity) == True{} : Bool}
    %Equal.sym(Nat,Nat.sub(e,r),Nat.add(count,rest),rest_split(r,count,e,fits)) : {Nat.is_le(Nat.mul(Nat.add(offset,_),size),capacity) == True{} : Bool}
    %whole_split(start,r,count,e,progress_after(~Holder,~value,batch,row,target,progress),fits) : {Nat.is_le(Nat.mul(_,size),capacity) == True{} : Bool}
    progress_room(~Holder,~value,batch,row,target,progress) : {Nat.is_le(Nat.add(t,Nat.mul(Nat.sub(e,r),size)),capacity) == True{} : Bool}}
  +k_inside = Bounds.nat_trans(Nat.add(t,k),Nat.add(t,Nat.mul(Nat.sub(e,r),size)),capacity,Order.offset_lt(t,k,Nat.mul(Nat.sub(e,r),size),k_less),span)
  +outside = {Rows.Beyond{beyond} : Rows.Outside<t,Rows.region_cells(call,counted),Nat.add(t,k)>}
  +kept = Rows.record_kept(~Holder,~value,record,Nat.add(t,k),outside,k_inside)
  +index_exact = Equal.trans(Nat,Nat.add(t,k),Nat.add(Nat.add(t,moved),cell),Nat.add(U32.to_nat(next_target(~Holder,~value,batch,row,target)),cell),
    Equal.sym(Nat,Nat.add(Nat.add(t,moved),cell),Nat.add(t,k),Addition.associative(t,moved,cell)),
    Equal.cong(Nat,Nat,value => Nat.add(value,cell),Nat.add(t,moved),U32.to_nat(next_target(~Holder,~value,batch,row,target)),
      Equal.sym(Nat,U32.to_nat(next_target(~Holder,~value,batch,row,target)),Nat.add(t,moved),target_exact(~Holder,~value,batch,row,target,progress,fact))))
  %index_exact : {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next),U32.from_nat(_)) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,next),U32.from_nat(Nat.add(t,k))),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(t,k))),
    Tile.filled((0.0 + 0.0 : F32)),kept,old)

# ---- Every cell after a step ----

# A step of the task, as the scope of its cells.
type Stepping<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Stepping{batch: Batch<Holder,value>,row: U32,target: U32,progress: Progress<Holder,value,batch,row,target>,fact: CallFact<Holder,value,batch,row>}
def stepping_batch(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) -> Batch<Holder,value>:
  Stepping{batch,row,target,progress,fact} = stepping
  batch
def stepping_row(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) -> U32:
  Stepping{batch,row,target,progress,fact} = stepping
  row
def stepping_target(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) -> U32:
  Stepping{batch,row,target,progress,fact} = stepping
  target
def stepping_progress(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) ->
  Progress<Holder,value,stepping_batch(~Holder,~value,stepping),stepping_row(~Holder,~value,stepping),stepping_target(~Holder,~value,stepping)>:
  Stepping{batch,row,target,progress,fact} = stepping
  progress
def stepping_fact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) ->
  CallFact<Holder,value,stepping_batch(~Holder,~value,stepping),stepping_row(~Holder,~value,stepping)>:
  Stepping{batch,row,target,progress,fact} = stepping
  fact
def stepping_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+stepping: Stepping<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  next_output(~Holder,~value,stepping_batch(~Holder,~value,stepping),stepping_row(~Holder,~value,stepping),stepping_target(~Holder,~value,stepping),
    stepping_progress(~Holder,~value,stepping),stepping_fact(~Holder,~value,stepping))
def stepping_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) -> Nat:
  Stepping{batch,row,target,progress,fact} = stepping
  written_next(~Holder,~value,batch,row)
def stepping_zeros(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,stepping: Stepping<Holder,value>) -> Nat:
  Stepping{batch,row,target,progress,fact} = stepping
  zero_next(~Holder,~value,batch,row)

def NextWritten(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+stepping: Stepping<Holder,value>,+cell: Nat) -> Data:
  WrittenCell(~Holder,~value,stepping_batch(~Holder,~value,stepping),Buffer.tree(~Tile.Tile8,stepping_output(~Holder,~value,stepping)),cell)
def NextZero(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+stepping: Stepping<Holder,value>,+cell: Nat) -> Data:
  +target = next_target(~Holder,~value,stepping_batch(~Holder,~value,stepping),stepping_row(~Holder,~value,stepping),stepping_target(~Holder,~value,stepping))
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,stepping_output(~Holder,~value,stepping)),U32.from_nat(Nat.add(U32.to_nat(target),cell))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8}

def next_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+stepping: Stepping<Holder,value>,+cell: Nat,
  +inside: {Nat.is_lt(cell,stepping_written(~Holder,~value,stepping)) == True{} : Bool}) -> NextWritten(~Holder,~value,stepping,cell):
  Stepping{+batch,+row,+target,+progress,+fact} = stepping
  written_cell(~Holder,~value,batch,row,target,progress,fact,cell,Nat.is_lt(cell,written_rows(~Holder,~value,batch,row)),{==},inside)
def next_zero(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+stepping: Stepping<Holder,value>,+cell: Nat,
  +inside: {Nat.is_lt(cell,stepping_zeros(~Holder,~value,stepping)) == True{} : Bool}) -> NextZero(~Holder,~value,stepping,cell):
  Stepping{+batch,+row,+target,+progress,+fact} = stepping
  zero_cell(~Holder,~value,batch,row,target,progress,fact,cell,inside)

# ---- The loop ----

# The task after the step from a row: the holding gets its band back, the
# leaf keeps the packed weights, and the output holds the gradient of the
# rows up to the next row and +0 after it.
def step_done(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,-state: Kernel.Rows,
  +progress: Progress<Holder,value,batch,row,target>,+fact: CallFact<Holder,value,batch,row>,
  +packed: {state == Kernel.Rows{Held.pack(~Tile.Tile8,progress_holding(~Holder,~value,batch,row,target,progress)),Held.pack_spare(~Tile.Tile8,progress_spare(~Holder,~value,batch,row,target,progress)),
    Buffer.pack(~Tile.Tile8,progress_tiles(~Holder,~value,batch,row,target,progress)),Buffer.pack(~Tile.Tile8,progress_output(~Holder,~value,batch,row,target,progress))} : Kernel.Rows},
  +covered: Held.Covered<Tile.Tile8,Holder,value,batch_holder(~Holder,~value,batch),
    Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(progress_holding(~Holder,~value,batch,row,target,progress),progress_spare(~Holder,~value,batch,row,target,progress),batch_geometry(~Holder,~value,batch),
        image_word(batch_geometry(~Holder,~value,batch),row),first_word(batch_geometry(~Holder,~value,batch),row),
        count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)))),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_source(Rows.record_part(~Holder,~value,step_record(~Holder,~value,batch,row,target,progress,fact))))),
    batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch),batch_layout(~Holder,~value,batch)>) ->
  Reached<Holder,value,batch,next_row(~Holder,~value,batch,row),next_target(~Holder,~value,batch,row,target),
    Kernel.image_rows(batch_task(~Holder,~value,batch),image_word(batch_geometry(~Holder,~value,batch),row),first_word(batch_geometry(~Holder,~value,batch),row),
      count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)),target,state)>:
  Held.Covered{+model,+returned,+tiled,+values,+ordered} = covered
  +step = step_of(~Holder,~value,batch,row,target,progress,fact)
  +record = step_record(~Holder,~value,batch,row,target,progress,fact)
  +geometry = batch_geometry(~Holder,~value,batch)
  +image = image_word(geometry,row)
  +first = first_word(geometry,row)
  +counted = count_word(geometry,row,batch_end(~Holder,~value,batch))
  +holding = progress_holding(~Holder,~value,batch,row,target,progress)
  +spare = progress_spare(~Holder,~value,batch,row,target,progress)
  +tiles = progress_tiles(~Holder,~value,batch,row,target,progress)
  +output = progress_output(~Holder,~value,batch,row,target,progress)
  +part = Rows.record_part(~Holder,~value,record)
  +weights = LeafWrites.part_weights(part)
  +next = Execution.part_output(part)
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +r = U32.to_nat(row)
  +e = U32.to_nat(batch_end(~Holder,~value,batch))
  +count = U32.to_nat(counted)
  +size = batch_size(~Holder,~value,batch)
  +offset = Nat.sub(r,start)
  +after = progress_after(~Holder,~value,batch,row,target,progress)
  +moved = U32.to_nat(next_row(~Holder,~value,batch,row))
  +tiles_exact = Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,tiles),
    Rows.native_tree(Rows.packing(batch_call(~Holder,~value,batch),batch_weights(~Holder,~value,batch))),
    Rows.call_weights(holding,spare,geometry,image,first,counted,target,tiles,output),progress_tiles_exact(~Holder,~value,batch,row,target,progress))
  +depth_ok = {%Equal.sym(Nat,Buffer.depth(~Tile.Tile8,next),Buffer.depth(~Tile.Tile8,output),Rows.record_depth(~Holder,~value,record)) : {Nat.is_le(_,30n) == True{} : Bool}
    progress_depth_ok(~Holder,~value,batch,row,target,progress) : {Nat.is_le(Buffer.depth(~Tile.Tile8,next),30n) == True{} : Bool}}
  +room = {%Equal.sym(U32,Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,next)),Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)),Rows.record_kept_capacity(~Holder,~value,record)) :
      {Nat.is_le(batch_cells(~Holder,~value,batch),U32.to_nat(_)) == True{} : Bool}
    progress_room(~Holder,~value,batch,row,target,progress) : {Nat.is_le(batch_cells(~Holder,~value,batch),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,next)))) == True{} : Bool}}
  +length_exact = Equal.trans(Nat,Buffer.length(~Tile.Tile8,next),Buffer.length(~Tile.Tile8,output),batch_length(~Holder,~value,batch),
    Rows.record_length(~Holder,~value,record),progress_length(~Holder,~value,batch,row,target,progress))
  +next_after = {%Equal.sym(Nat,moved,Nat.add(r,count),step_next(~Holder,~value,batch,row,step)) : {Nat.is_le(start,_) == True{} : Bool}
    Order.transitive(start,r,Nat.add(r,count),after,Order.le_add(r,count)) : {Nat.is_le(start,moved) == True{} : Bool}}
  +next_before = {%Equal.sym(Nat,moved,Nat.add(r,count),step_next(~Holder,~value,batch,row,step)) : {Nat.is_le(_,e) == True{} : Bool}
    step_next_le(~Holder,~value,batch,row,step) : {Nat.is_le(moved,e) == True{} : Bool}}
  +landed = U32.to_nat(next_target(~Holder,~value,batch,row,target))
  +landed_exact = {%Equal.sym(Nat,moved,Nat.add(r,count),step_next(~Holder,~value,batch,row,step)) : {landed == Nat.mul(Nat.sub(_,start),size) : Nat}
    %Equal.sym(Nat,Nat.sub(Nat.add(r,count),start),Nat.add(offset,count),shifted_sum(start,r,count,after)) : {landed == Nat.mul(_,size) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.add(offset,count),size),Nat.add(Nat.mul(offset,size),Nat.mul(count,size)),RowMajor.distrib(offset,count,size)) : {landed == _ : Nat}
    %progress_target(~Holder,~value,batch,row,target,progress) : {landed == Nat.add(_,Nat.mul(count,size)) : Nat}
    target_exact(~Holder,~value,batch,row,target,progress,fact) : {landed == Nat.mul(Nat.sub(moved,start),size) : Nat}}
  +spare_next = Held.spare_after(~Tile.Tile8,holding,Rows.band_first(geometry,image,first,counted),Rows.band_cells(geometry,first,counted),spare,LeafWrites.part_source(part))
  +spare_room_next = Rows.call_spare_room(holding,spare,progress_spare_room(~Holder,~value,batch,row,target,progress),geometry,image,first,counted,target,tiles,output,
    Rows.band_allocatable(batch_call(~Holder,~value,batch),batch_envelope(~Holder,~value,batch),first,counted,Rows.record_contained(~Holder,~value,record),Rows.record_positive(~Holder,~value,record)))
  +stepping = {Stepping{batch,row,target,progress,fact} : Stepping<Holder,value>}
  +written = Members.range_all(~Stepping<Holder,value>,~NextWritten(~Holder,~value),~stepping_written(~Holder,~value),~next_written(~Holder,~value),
    written_next(~Holder,~value,batch,row),0n,stepping,Order.reflexive(written_next(~Holder,~value,batch,row)))
  +zero = Members.range_all(~Stepping<Holder,value>,~NextZero(~Holder,~value),~stepping_zeros(~Holder,~value),~next_zero(~Holder,~value),
    zero_next(~Holder,~value,batch,row),0n,stepping,Order.reflexive(zero_next(~Holder,~value,batch,row)))
  +advanced = {Progress{model,tiled,values,ordered,spare_next,spare_room_next,weights,tiles_exact,next,depth_ok,room,length_exact,next_after,next_before,landed_exact,written,zero} :
    Progress<Holder,value,batch,next_row(~Holder,~value,batch,row),next_target(~Holder,~value,batch,row,target)>}
  +rows_exact = {%Equal.sym(Kernel.Rows,state,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)},packed) :
      {Kernel.image_rows(batch_task(~Holder,~value,batch),image,first,counted,target,_) == Kernel.Rows{Held.pack(~Tile.Tile8,model),Held.pack_spare(~Tile.Tile8,spare_next),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)} : Kernel.Rows}
    Equal.trans(Kernel.Rows,Kernel.image_rows(batch_task(~Holder,~value,batch),image,first,counted,target,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),Kernel.Rows{Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},Kernel.Rows{Held.pack(~Tile.Tile8,model),Held.pack_spare(~Tile.Tile8,spare_next),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},
      Rows.call_rows(holding,spare,geometry,image,first,counted,target,tiles,output),
      Equal.trans(Kernel.Rows,Kernel.Rows{Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},Kernel.Rows{Held.pack(~Tile.Tile8,model),Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},Kernel.Rows{Held.pack(~Tile.Tile8,model),Held.pack_spare(~Tile.Tile8,spare_next),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},
        Equal.cong(Move.Holding<Tile.Tile8>,Kernel.Rows,gradient => Kernel.Rows{gradient,Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Held.pack(~Tile.Tile8,model),returned),
        Equal.cong(Move.Spare<Tile.Tile8>,Kernel.Rows,kept => Kernel.Rows{Held.pack(~Tile.Tile8,model),kept,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)},Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,Rows.call_band(holding,spare,geometry,image,first,counted)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Held.pack_spare(~Tile.Tile8,spare_next),Rows.call_spare(holding,spare,geometry,image,first,counted,target,tiles,output)))) :
    {Kernel.image_rows(batch_task(~Holder,~value,batch),image,first,counted,target,state) == Kernel.Rows{Held.pack(~Tile.Tile8,model),Held.pack_spare(~Tile.Tile8,spare_next),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,next)} : Kernel.Rows}}
  Reached{advanced,rows_exact}

def batch_step(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,-state: Kernel.Rows,
  +done: Reached<Holder,value,batch,row,target,state>,+fact: CallFact<Holder,value,batch,row>) ->
  Reached<Holder,value,batch,next_row(~Holder,~value,batch,row),next_target(~Holder,~value,batch,row,target),
    Kernel.image_rows(batch_task(~Holder,~value,batch),image_word(batch_geometry(~Holder,~value,batch),row),first_word(batch_geometry(~Holder,~value,batch),row),
      count_word(batch_geometry(~Holder,~value,batch),row,batch_end(~Holder,~value,batch)),target,state)>:
  Reached{+progress,+packed} = done
  +geometry = batch_geometry(~Holder,~value,batch)
  +covered = Rows.call_covered(~Holder,~value,batch_holder(~Holder,~value,batch),progress_holding(~Holder,~value,batch,row,target,progress),progress_spare(~Holder,~value,batch,row,target,progress),geometry,
    image_word(geometry,row),first_word(geometry,row),count_word(geometry,row,batch_end(~Holder,~value,batch)),target,
    progress_tiles(~Holder,~value,batch,row,target,progress),progress_output(~Holder,~value,batch,row,target,progress),
    batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch),progress_tiled(~Holder,~value,batch,row,target,progress),
    progress_values(~Holder,~value,batch,row,target,progress),batch_layout(~Holder,~value,batch),progress_ordered(~Holder,~value,batch,row,target,progress))
  step_done(~Holder,~value,batch,row,target,state,progress,fact,packed,covered)

# The task after the rows of the batch: the holding, the packed weights and
# the output, whose cells hold the gradient of the rows of the batch.
type Finished<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-batch: Batch<Holder,value>,-state: Kernel.Rows> is Data:
  Finished{holding: Held.Holding<Tile.Tile8>,
    tiled: Held.Tiled(~Tile.Tile8,holding,batch_low(~Holder,~value,batch),batch_high(~Holder,~value,batch)),
    values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,batch_holder(~Holder,~value,batch),holding),
    ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == batch_layout(~Holder,~value,batch) : Held.Spans},
    spare: Held.SpareModel<Tile.Tile8>,spare_room: Held.SpareRoom(~Tile.Tile8,spare),
    tiles: Buffer.Model<Tile.Tile8>,
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == Rows.native_tree(Rows.packing(batch_call(~Holder,~value,batch),batch_weights(~Holder,~value,batch))) : Tree.Tree<Tile.Tile8>},
    output: Buffer.Model<Tile.Tile8>,depth_ok: {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool},
    room: {Nat.is_le(batch_cells(~Holder,~value,batch),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool},
    length_exact: {Buffer.length(~Tile.Tile8,output) == batch_length(~Holder,~value,batch) : Nat},
    written: Members.All(Nat,{cell => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},
      Ranges.range(batch_cells(~Holder,~value,batch),0n)),
    packed: {state == Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)} : Kernel.Rows}}

# The task at the end of the batch has finished it.
def finish(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+batch: Batch<Holder,value>,+row: U32,+target: U32,-state: Kernel.Rows,
  +ended: {U32.to_nat(row) == U32.to_nat(batch_end(~Holder,~value,batch)) : Nat},+done: Reached<Holder,value,batch,row,target,state>) ->
  Finished<Holder,value,batch,state>:
  Reached{+progress,+packed} = done
  +output = progress_output(~Holder,~value,batch,row,target,progress)
  +start = U32.to_nat(batch_start(~Holder,~value,batch))
  +size = batch_size(~Holder,~value,batch)
  +all = {%ended : Members.All(Nat,{cell => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(_,start),size),0n))
    progress_written(~Holder,~value,batch,row,target,progress) : Members.All(Nat,{cell => WrittenCell(~Holder,~value,batch,Buffer.tree(~Tile.Tile8,output),cell) : Nat -> Data},Ranges.range(batch_cells(~Holder,~value,batch),0n))}
  Finished{progress_holding(~Holder,~value,batch,row,target,progress),progress_tiled(~Holder,~value,batch,row,target,progress),progress_values(~Holder,~value,batch,row,target,progress),progress_ordered(~Holder,~value,batch,row,target,progress),progress_spare(~Holder,~value,batch,row,target,progress),progress_spare_room(~Holder,~value,batch,row,target,progress),
    progress_tiles(~Holder,~value,batch,row,target,progress),progress_tiles_exact(~Holder,~value,batch,row,target,progress),output,progress_depth_ok(~Holder,~value,batch,row,target,progress),progress_room(~Holder,~value,batch,row,target,progress),progress_length(~Holder,~value,batch,row,target,progress),all,packed}

# The rows a task computes from a row on, step by step.
def batch_loop(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,fuel: Nat,+batch: Batch<Holder,value>,+row: U32,+target: U32,-state: Kernel.Rows,
  facts: CallFacts(~Holder,~value,fuel,batch,row),done: Reached<Holder,value,batch,row,target,state>) ->
  Finished<Holder,value,batch,Kernel.batch_rows(fuel,batch_task(~Holder,~value,batch),row,batch_end(~Holder,~value,batch),target,state)>:
  match fuel:
    case 0n: finish(~Holder,~value,batch,row,target,state,facts,done)
    case 1n+ +rest:
      KernelGemm.Both{fact,later} = facts
      +geometry = batch_geometry(~Holder,~value,batch)
      batch_loop(~Holder,~value,rest,batch,next_row(~Holder,~value,batch,row),next_target(~Holder,~value,batch,row,target),
        Kernel.image_rows(batch_task(~Holder,~value,batch),image_word(geometry,row),first_word(geometry,row),count_word(geometry,row,batch_end(~Holder,~value,batch)),target,state),
        later,batch_step(~Holder,~value,batch,row,target,state,done,fact))
