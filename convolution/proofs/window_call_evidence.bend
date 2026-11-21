# The window problem of one call of the windowed convolution: the rows of
# its band, its packing start, and the facts the blocked loop of the call
# needs.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./window_problem.bend as WindowProblem
import ./window_samples.bend as Samples
import ./window_positions.bend as Positions
import ./window_start.bend as Start
import ./window_contents.bend as Contents
import ./window_band_cover.bend as BandCover
import ../matrix_specification.bend as MatrixSpec
import ./output_values.bend as Values
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/u32_extrema.bend as Extrema
import ./packing_tile_bounds.bend as PackingBounds
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ./blocked_convolution_readback.bend as Blocked
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/kernel_partition_bounds.bend as Partition

# The problem a call on positions [offset, offset + span) of an image solves.
def window(+problem: ConvSpec.Problem,+offset: U32,+span: U32) -> ConvSpec.Problem:
  Samples.window(problem,U32.to_nat(BandCover.first_word(problem,offset)),U32.to_nat(BandCover.rows_word(problem,offset,span)))

def pack_start(+problem: ConvSpec.Problem,+offset: U32) -> U32:
  KernelWindow.start(offset,BandCover.stride_word(problem),BandCover.width_word(problem),BandCover.first_word(problem,offset))

# A covering band gives a valid and supported window problem.
def cover_valid(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +cover: Positions.Cover<problem,first,rows,offset,span>) -> {ConvSpec.valid(Samples.window(problem,first,rows)) == True{} : Bool}:
  Positions.Cover{block,first_exact,fits,+rows_positive,+kernel_fits,top,bottom} = cover
  WindowProblem.valid(problem,first,rows,valid,rows_positive,kernel_fits)

def cover_supported(+limits: Machine.Limits,+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +cover: Positions.Cover<problem,first,rows,offset,span>) -> {NativeConv.supports_packed_with_limits(limits,Samples.window(problem,first,rows)) == True{} : Bool}:
  Positions.Cover{block,first_exact,+fits,rows_positive,kernel_fits,top,bottom} = cover
  WindowProblem.supported(limits,problem,first,rows,valid,supported,fits)

# The packing start of a call moves its offset back by the output rows above
# the band.
def cover_start(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+first: U32,+rows: Nat,+span: Nat,
  +cover: Positions.Cover<problem,U32.to_nat(first),rows,U32.to_nat(offset),span>,
  +offset_room: {Nat.is_le(U32.to_nat(offset),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.start(offset,U32.from_nat(Guards.stride(problem)),U32.from_nat(ConvSpec.out_width(problem)),first)) ==
    Nat.sub(U32.to_nat(offset),Nat.mul(Positions.block(problem,U32.to_nat(first),rows,U32.to_nat(offset),span,cover),ConvSpec.out_width(problem))) : Nat}:
  Positions.Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  +start = U32.to_nat(offset)
  +width = ConvSpec.out_width(problem)
  +facts = Guards.valid(problem,valid)
  +reach = Order.transitive(Nat.mul(block,width),Nat.mul(Nat.div(start,width),width),start,
    RowMajor.mul_le(block,Nat.div(start,width),width,
      Positions.block_le_start(problem,U32.to_nat(first),block,start,span,Contents.valid_condition(problem,valid,5n),first_exact,top)),
    DivisionOrder.product_below(start,width,Coordinates.width_positive(problem)))
  Start.start_exact(limits,problem,facts,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),offset,first,block,first_exact,reach,offset_room)

# The positions of a call, moved to the window, fit its output grid: the last
# position reads an output row of the window.
def window_span(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,
  +cover: Positions.Cover<problem,first,rows,offset,span>,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(offset,span),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.sub(offset,Nat.mul(Positions.block(problem,first,rows,offset,span,cover),ConvSpec.out_width(problem))),span),
    Nat.mul(ConvSpec.axis_size(rows,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem)),ConvSpec.out_width(problem))) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+image_height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,image_height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +width = ConvSpec.out_width(problem)
  +height = ConvSpec.out_height(problem)
  +skip = Positions.block(problem,first,rows,offset,span,cover)
  +stride_ok = Contents.valid_condition(problem,valid,5n)
  +width_ok = Coordinates.width_positive(problem)
  +skipped = Nat.mul(skip,width)
  +reach = Order.transitive(skipped,Nat.mul(Nat.div(offset,width),width),offset,
    RowMajor.mul_le(skip,Nat.div(offset,width),width,Positions.block_le_row(problem,first,rows,offset,span,cover,offset,stride_ok,width_ok,Order.reflexive(offset))),
    DivisionOrder.product_below(offset,width,width_ok))
  +end = Nat.add(offset,span)
  +last = Nat.sub(end,1n)
  +end_positive = Bounds.nat_trans(0n,span,end,positive,Order.right_le(offset,span))
  +last_inside = Order.last_below(end,end_positive)
  +last_after = Order.before_last(offset,end,BandCover.offset_after(problem,offset,span,positive,grid))
  +last_row = Nat.div(last,width)
  +last_valid = DivisionOrder.quotient_below(last,height,width,width_ok,Bounds.nat_trans(last,end,Nat.mul(height,width),last_inside,grid))
  +last_ordered = Positions.block_le_row(problem,first,rows,offset,span,cover,last,stride_ok,width_ok,last_after)
  +last_virtual = Positions.virtual_row_inside(problem,first,rows,offset,span,cover,last,stride_ok,width_ok,last_after,last_inside,last_valid)
  +window_height = ConvSpec.axis_size(rows,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem))
  %Positions.sub_after(offset,skipped,span,reach) : {Nat.is_le(_,Nat.mul(window_height,width)) == True{} : Bool}
  Order.transitive(Nat.sub(end,skipped),Nat.mul(1n+Nat.sub(last_row,skip),width),Nat.mul(window_height,width),
    %Positions.sub_succ(last_row,skip,last_ordered) : {Nat.is_le(Nat.sub(end,skipped),Nat.mul(_,width)) == True{} : Bool}
    %RowMajor.subtract_products(1n+last_row,skip,width) : {Nat.is_le(Nat.sub(end,skipped),_) == True{} : Bool}
    DivisionOrder.subtract_le(end,Nat.mul(1n+last_row,width),skipped,
      %Order.positive_has_last(end,end_positive) : {Nat.is_le(_,Nat.mul(1n+last_row,width)) == True{} : Bool}
      Order.strict_follow(last,Nat.mul(1n+last_row,width),DivisionOrder.next_above(last,width,width_ok))),
    RowMajor.mul_le(1n+Nat.sub(last_row,skip),window_height,width,Order.strict_follow(Nat.sub(last_row,skip),window_height,last_virtual)))

def span(+target: U32,+problem: ConvSpec.Problem) -> U32:
  (8 * U32.max(1,(target / Coordinates.native_inner(problem) : U32)) : U32)

type Region<-problem: ConvSpec.Problem,-origin: U32,-positions: U32> is Data:
  Region{within: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingBounds.grid(problem)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(positions)) == True{} : Bool}}

def positions_bound(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,region: Region<problem,origin,positions>) ->
  {Nat.is_le(U32.to_nat(positions),PackingBounds.grid(problem)) == True{} : Bool}:
  Region{within,positive} = region
  Order.transitive(U32.to_nat(positions),Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingBounds.grid(problem),(%Order.commutative(U32.to_nat(positions),U32.to_nat(origin)) : {Nat.is_le(U32.to_nat(positions),_) == True{} : Bool} Order.le_add(U32.to_nat(positions),U32.to_nat(origin))),within)

def inner_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(U32.to_nat(Coordinates.native_inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Coordinates.inner_bound(limits,problem,valid,Padded.scalar(limits,problem,fast))

def inner_positive(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_lt(0n,U32.to_nat(Coordinates.native_inner(problem))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))) : {Nat.is_lt(0n,_) == True{} : Bool}
  Coordinates.inner_positive(problem,valid)

# The outputs a call writes and the positions after them stay below the
# loop bound.
def output_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+positions: U32,
  +ends: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),Nat.add(U32.to_nat(offset),U32.to_nat(positions))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat((offset + positions : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((offset + positions : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(positions)),Arithmetic.add(offset,positions,Machine.capacity(limits),Limits.capacity_fits(limits),ends)) :
    {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  outputs

def offset_room(+limits: Machine.Limits,+offset: U32,+positions: U32,+allocation_cap: U32,+loop_cap: U32,
  +capacity_room: {U32.is_le(Machine.capacity(limits),allocation_cap) == True{} : Bool},+doubled: {U32.to_nat(loop_cap) == Nat.double(U32.to_nat(allocation_cap)) : Nat},
  +ends: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +padded_end: {Nat.is_le(U32.to_nat(KernelWindow.padded(positions)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(KernelWindow.padded(positions))),U32.to_nat(loop_cap)) == True{} : Bool}:
  +allocation_room = Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)
  +front = Order.offset_le(U32.to_nat(offset),U32.to_nat(KernelWindow.padded(positions)),Nat.add(U32.to_nat(positions),7n),padded_end)
  +sum_room = Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),ends,allocation_room)
  +seven = Order.transitive(7n,U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)),allocation_room)
  +back = Encoded.add_in_double(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),allocation_cap,loop_cap,doubled,sum_room,seven)
  +tail = {%Addition.associative(U32.to_nat(offset),U32.to_nat(positions),7n) : {Nat.is_le(_,U32.to_nat(loop_cap)) == True{} : Bool} back : {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(positions),7n)),U32.to_nat(loop_cap)) == True{} : Bool}}
  Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(KernelWindow.padded(positions))),Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(positions),7n)),U32.to_nat(loop_cap),front,tail)

# The blocked loop of a call: positions [origin, origin + positions) of a
# problem written from output position offset on.
type Loop<-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-target: U32,-loop_cap: U32> is Data:
  Loop{geometry: Blocked.Geometry<limits,problem,origin,offset,positions,KernelWindow.padded(positions),span(target,problem),loop_cap>,
    schedule: BlockBounds.Schedule<U32.to_nat(((KernelWindow.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),0n,U32.to_nat(KernelWindow.padded(positions)),U32.to_nat(span(target,problem))>}

def loop(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+target: U32,+allocation_cap: U32,+loop_cap: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +ends: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),Nat.add(U32.to_nat(offset),U32.to_nat(positions))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +configuration: BlockBounds.Configuration<limits,target,allocation_cap,loop_cap>) ->
  Loop<limits,problem,origin,offset,positions,target,loop_cap>:
  Region{+within,+positions_positive} = region
  BlockBounds.Configuration{+target_positive,+target_room,+target_span_room,+allocation_fits,+allocation_bound,+capacity_room,+loop_fits,+doubled} = configuration
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +inner = Coordinates.native_inner(problem)
  +inner_bound = inner_room(limits,problem,facts,fast)
  +inner_positive = inner_positive(limits,problem,facts,fast)
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +groups = BlockBounds.groups(target,inner)
  +span = span(target,problem)
  +span_exact = BlockBounds.span_exact(target,target_positive,limits,inner,allocation_cap,allocation_fits,target_span_room,inner_positive,inner_bound)
  +span_room = {
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(groups,8n),span_exact) : {Nat.is_le(_,U32.to_nat(allocation_cap)) == True{} : Bool}
    Order.transitive(Nat.mul(groups,8n),Nat.mul(U32.to_nat(target),8n),U32.to_nat(allocation_cap),BlockBounds.span_room(target,target_positive,limits,inner,inner_bound,inner_positive),target_span_room) : {Nat.is_le(U32.to_nat(span),U32.to_nat(allocation_cap)) == True{} : Bool}}
  +span_positive = {
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(groups,8n),span_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(groups,8n,BlockBounds.groups_positive(target,inner),{==}) : {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}}
  +region = {Region{within,positions_positive} : Region<problem,origin,positions>}
  +grid_room = Padded.spatial_bound_facts(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +positions_grid = positions_bound(problem,origin,positions,region)
  +positions_room = Order.transitive(U32.to_nat(positions),PackingBounds.grid(problem),U32.to_nat(Machine.capacity(limits)),positions_grid,grid_room)
  +padded = KernelWindow.padded(positions)
  +padded_exact = Encoded.padded_exact(limits,positions,positions_room)
  +padded_bound = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Encoded.padded_le(U32.to_nat(positions),PackingBounds.grid(problem),positions_grid) : {Nat.is_le(U32.to_nat(padded),Support.padded(problem)) == True{} : Bool}}
  +padded_end = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}
    DivisionOrder.product_below(Nat.add(U32.to_nat(positions),7n),8n,{==}) : {Nat.is_le(U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}}
  +origin_end = {Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(positions),7n)),Nat.add(PackingBounds.grid(problem),7n),
    Order.offset_le(U32.to_nat(origin),U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n),padded_end),
    %Addition.associative(U32.to_nat(origin),U32.to_nat(positions),7n) : {Nat.is_le(_,Nat.add(PackingBounds.grid(problem),7n)) == True{} : Bool}
    Order.add_le(Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingBounds.grid(problem),7n,within)) : {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingBounds.grid(problem),7n)) == True{} : Bool}}
  +pack_room = Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingBounds.grid(problem),7n),U32.to_nat(loop_cap),origin_end,
    Encoded.add_in_double(PackingBounds.grid(problem),allocation_cap,loop_cap,doubled,
      Order.transitive(PackingBounds.grid(problem),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),grid_room,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)),
      Order.transitive(7n,U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)),Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room))))
  +padded_room = Order.transitive(U32.to_nat(padded),Support.padded(problem),U32.to_nat(allocation_cap),padded_bound,
    Order.transitive(Support.padded(problem),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),Padded.padded_bound(limits,problem,facts,fast),Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)))
  +sum_room = BlockBounds.sum_room(U32.to_nat(padded),U32.to_nat(span),allocation_cap,loop_cap,padded_room,span_room,doubled)
  +divisor_room = {
    %Equal.sym(Nat,U32.to_nat(loop_cap),Nat.double(U32.to_nat(allocation_cap)),doubled) : {Nat.is_le(Nat.double(U32.to_nat(span)),_) == True{} : Bool}
    Order.double_le(U32.to_nat(span),U32.to_nat(allocation_cap),span_room) : {Nat.is_le(Nat.double(U32.to_nat(span)),U32.to_nat(loop_cap)) == True{} : Bool}}
  +padded_positive = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,U32.to_nat(positions),BlockBounds.ceiling_eight(U32.to_nat(positions)),positions_positive,Encoded.within_ceiling(U32.to_nat(positions))) : {Nat.is_lt(0n,U32.to_nat(padded)) == True{} : Bool}}
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  +remaining_exact = BlockBounds.remaining_exact(padded,span,loop_cap,loop_fits,padded_positive,span_positive,sum_room,divisor_room)
  +end_exact = Arithmetic.add(offset,positions,Machine.capacity(limits),Limits.capacity_fits(limits),ends)
  +outputs_room = output_room(limits,problem,offset,positions,ends,outputs)
  +offset_room = offset_room(limits,offset,positions,allocation_cap,loop_cap,capacity_room,doubled,ends,padded_end)
  Loop{
    Blocked.Geometry{pack_room,valid,supported,loop_fits,padded_bound,padded_end,
      outputs_room,end_exact,offset_room,
      sum_room,span_positive,Partition.BlockAligned{Nat.div(Nat.add(U32.to_nat(positions),7n),8n),padded_exact},Partition.BlockAligned{groups,span_exact},{==}},
    %Equal.sym(Nat,remaining,Ceiling.ceiling(U32.to_nat(padded),U32.to_nat(span)),remaining_exact) : BlockBounds.Schedule<_,0n,U32.to_nat(padded),U32.to_nat(span)>
    BlockBounds.initial_schedule(U32.to_nat(padded),U32.to_nat(span),padded_positive,span_positive)}

# The block of the loop is the block of the pipeline.
def loop_geometry(-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-target: U32,-loop_cap: U32,loop: Loop<limits,problem,origin,offset,positions,target,loop_cap>) ->
  Blocked.Geometry<limits,problem,origin,offset,positions,KernelWindow.padded(positions),span(target,problem),loop_cap>:
  Loop{geometry,schedule} = loop
  geometry

def loop_schedule(-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-target: U32,-loop_cap: U32,loop: Loop<limits,problem,origin,offset,positions,target,loop_cap>) ->
  BlockBounds.Schedule<U32.to_nat(((KernelWindow.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),0n,U32.to_nat(KernelWindow.padded(positions)),U32.to_nat(span(target,problem))>:
  Loop{geometry,schedule} = loop
  schedule

def point(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(origin),Coordinates.position(problem,row,column)) == True{} : Bool},
  +inside: {Nat.is_lt(Blocked.local_position(problem,origin,row,column),U32.to_nat(positions)) == True{} : Bool}) ->
  Blocked.Point<problem,origin,positions,channel,row,column,KernelWindow.padded(positions)>:
  +room = Order.transitive(U32.to_nat(positions),PackingBounds.grid(problem),U32.to_nat(Machine.capacity(limits)),positions_bound(problem,origin,positions,region),
    Padded.spatial_bound_facts(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))
  Blocked.Point{Order.restore_shift(U32.to_nat(origin),Coordinates.position(problem,row,column),lower),channel_inside,row_inside,column_inside,inside,
    %Equal.sym(Nat,U32.to_nat(KernelWindow.padded(positions)),BlockBounds.ceiling_eight(U32.to_nat(positions)),Encoded.padded_exact(limits,positions,room)) : {Nat.is_lt(Blocked.local_position(problem,origin,row,column),_) == True{} : Bool}
    Bounds.nat_trans(Blocked.local_position(problem,origin,row,column),U32.to_nat(positions),BlockBounds.ceiling_eight(U32.to_nat(positions)),inside,Encoded.within_ceiling(U32.to_nat(positions)))}

# The positions of a call are a region of its window problem.
def covered_region(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,offset)),U32.to_nat(BandCover.rows_word(problem,offset,span)),U32.to_nat(offset),U32.to_nat(span)>) ->
  Region<window(problem,offset,span),pack_start(problem,offset),span>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+image_height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,image_height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +first = BandCover.first_word(problem,offset)
  +rows = U32.to_nat(BandCover.rows_word(problem,offset,span))
  +start = U32.to_nat(offset)
  +count = U32.to_nat(span)
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +offset_room = Order.transitive(start,Nat.add(start,count),U32.to_nat(Machine.capacity(limits)),Order.le_add(start,count),
    Order.transitive(Nat.add(start,count),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),U32.to_nat(Machine.capacity(limits)),grid,
      Padded.spatial_bound_facts(limits,problem,facts,support)))
  Region{%Equal.sym(Nat,U32.to_nat(pack_start(problem,offset)),Nat.sub(start,Nat.mul(Positions.block(problem,U32.to_nat(first),rows,start,count,cover),ConvSpec.out_width(problem))),
      cover_start(limits,problem,valid,supported,offset,first,rows,count,cover,offset_room)) :
      {Nat.is_le(Nat.add(_,count),PackingBounds.grid(window(problem,offset,span))) == True{} : Bool}
    window_span(problem,valid,U32.to_nat(first),rows,start,count,cover,positive,grid),positive}

def tile_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+count: U32,+plane: U32,
  +ordered: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},+plane_room: {Nat.is_le(U32.to_nat(plane),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target: U32,+allocation_cap: U32,+loop_cap: U32,+configuration: BlockBounds.Configuration<limits,target,allocation_cap,loop_cap>,
  +workspace: U32,+room: {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(target,Coordinates.native_inner(problem),plane)),U32.to_nat(workspace)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(PackingBounds.groups(U32.min(span(target,problem),KernelWindow.padded(count))),Coordinates.inner(problem)),U32.to_nat(workspace)) == True{} : Bool}:
  BlockBounds.Configuration{+target_positive,+target_room,+target_span_room,+allocation_fits,+allocation_bound,+capacity_room,+loop_fits,+doubled} = configuration
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +inner = Coordinates.native_inner(problem)
  +inner_bound = inner_room(limits,problem,facts,fast)
  +inner_positive = inner_positive(limits,problem,facts,fast)
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +span = span(target,problem)
  +span_exact = BlockBounds.span_exact(target,target_positive,limits,inner,allocation_cap,allocation_fits,target_span_room,inner_positive,inner_bound)
  +prefix_room = BlockBounds.tile_prefix_room(target,limits,inner,inner_bound,inner_positive)
  +max_room = BlockBounds.tile_count_room(target,inner,allocation_cap,target_room,
    Order.transitive(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),inner_bound,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)))
  +tiles_exact = BlockBounds.workspace_count_exact(target,inner,plane,allocation_cap,allocation_fits,span_exact,prefix_room,max_room)
  +count_room = Order.transitive(U32.to_nat(count),U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),ordered,plane_room)
  +padded_ordered = {%Equal.sym(Nat,U32.to_nat(KernelWindow.padded(count)),Encoded.ceiling_eight(U32.to_nat(count)),Encoded.padded_exact(limits,count,count_room)) : {Nat.is_le(_,U32.to_nat(KernelWindow.padded(plane))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(KernelWindow.padded(plane)),Encoded.ceiling_eight(U32.to_nat(plane)),Encoded.padded_exact(limits,plane,plane_room)) : {Nat.is_le(Encoded.ceiling_eight(U32.to_nat(count)),_) == True{} : Bool}
    Encoded.padded_le(U32.to_nat(count),U32.to_nat(plane),ordered) : {Nat.is_le(U32.to_nat(KernelWindow.padded(count)),U32.to_nat(KernelWindow.padded(plane))) == True{} : Bool}}
  +prefix_ordered = {%Equal.sym(Nat,U32.to_nat(U32.min(span,KernelWindow.padded(count))),Nat.min(U32.to_nat(span),U32.to_nat(KernelWindow.padded(count))),Extrema.minimum(span,KernelWindow.padded(count))) :
      {Nat.is_le(_,U32.to_nat(U32.min(span,KernelWindow.padded(plane)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.min(span,KernelWindow.padded(plane))),Nat.min(U32.to_nat(span),U32.to_nat(KernelWindow.padded(plane))),Extrema.minimum(span,KernelWindow.padded(plane))) :
      {Nat.is_le(Nat.min(U32.to_nat(span),U32.to_nat(KernelWindow.padded(count))),_) == True{} : Bool}
    NatExtrema.minimum_monotone(U32.to_nat(KernelWindow.padded(count)),U32.to_nat(KernelWindow.padded(plane)),U32.to_nat(span),padded_ordered) :
    {Nat.is_le(U32.to_nat(U32.min(span,KernelWindow.padded(count))),U32.to_nat(U32.min(span,KernelWindow.padded(plane)))) == True{} : Bool}}
  +groups_ordered = DivisionOrder.quotient_le(U32.to_nat(U32.min(span,KernelWindow.padded(count))),U32.to_nat(U32.min(span,KernelWindow.padded(plane))),8n,{==},prefix_ordered)
  %inner_exact :
    {Nat.is_le(Nat.mul(PackingBounds.groups(U32.min(span,KernelWindow.padded(count))),_),U32.to_nat(workspace)) == True{} : Bool}
  Order.transitive(Nat.mul(PackingBounds.groups(U32.min(span,KernelWindow.padded(count))),U32.to_nat(inner)),BlockBounds.workspace_product(target,inner,plane),U32.to_nat(workspace),
    RowMajor.mul_le(PackingBounds.groups(U32.min(span,KernelWindow.padded(count))),PackingBounds.groups(U32.min(span,KernelWindow.padded(plane))),U32.to_nat(inner),groups_ordered),
    %tiles_exact :
      {Nat.is_le(_,U32.to_nat(workspace)) == True{} : Bool}
    room)

# Position offset + at of an image is the point of the window problem of a
# call on positions [offset, offset + span) that the call writes at local
# position at, and the window has the matrix value of the image there.
type Located<-problem: ConvSpec.Problem,-offset: U32,-span: U32,-channel: Nat,-at: Nat> is Data:
  Located{row: Nat,column: Nat,
    point: Blocked.Point<window(problem,offset,span),pack_start(problem,offset),span,channel,row,column,KernelWindow.padded(span)>,
    local_exact: {Blocked.local_position(window(problem,offset,span),pack_start(problem,offset),row,column) == at : Nat},
    value_exact: {Values.matrix_value(window(problem,offset,span),channel,row,column) ==
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(offset),at),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(offset),at),ConvSpec.out_width(problem))) : F32}}

def cover_first(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {first == Nat.mul(Positions.block(problem,first,rows,offset,span,cover),Guards.stride(problem)) : Nat}:
  Positions.Cover{block,+first_exact,fits,rows_positive,kernel_fits,top,bottom} = cover
  first_exact

def cover_fits(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}:
  Positions.Cover{block,first_exact,+fits,rows_positive,kernel_fits,top,bottom} = cover
  fits

def cover_positive(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_lt(0n,rows) == True{} : Bool}:
  Positions.Cover{block,first_exact,fits,+rows_positive,kernel_fits,top,bottom} = cover
  rows_positive

def covered_located(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+offset: U32,+span: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,offset)),U32.to_nat(BandCover.rows_word(problem,offset,span)),U32.to_nat(offset),U32.to_nat(span)>,+channel: Nat,+at: Nat,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+at_inside: {Nat.is_lt(at,U32.to_nat(span)) == True{} : Bool}) ->
  Located<problem,offset,span,channel,at>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+image_width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,image_width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +first = U32.to_nat(BandCover.first_word(problem,offset))
  +rows = U32.to_nat(BandCover.rows_word(problem,offset,span))
  +start = U32.to_nat(offset)
  +count = U32.to_nat(span)
  +width = ConvSpec.out_width(problem)
  +skip = Positions.block(problem,first,rows,start,count,cover)
  +stride_ok = Contents.valid_condition(problem,valid,5n)
  +width_ok = Coordinates.width_positive(problem)
  +skipped = Nat.mul(skip,width)
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +offset_room = Order.transitive(start,Nat.add(start,count),U32.to_nat(Machine.capacity(limits)),Order.le_add(start,count),
    Order.transitive(Nat.add(start,count),Nat.mul(ConvSpec.out_height(problem),width),U32.to_nat(Machine.capacity(limits)),grid,
      Padded.spatial_bound_facts(limits,problem,facts,support)))
  +reach = Order.transitive(skipped,Nat.mul(Nat.div(start,width),width),start,
    RowMajor.mul_le(skip,Nat.div(start,width),width,Positions.block_le_row(problem,first,rows,start,count,cover,start,stride_ok,width_ok,Order.reflexive(start))),
    DivisionOrder.product_below(start,width,width_ok))
  +start_exact = cover_start(limits,problem,valid,supported,offset,BandCover.first_word(problem,offset),rows,count,cover,offset_room)
  +position = Nat.add(start,at)
  +after = Order.le_add(start,at)
  +inside = {%Equal.sym(Bool,Nat.is_lt(position,Nat.add(start,count)),Nat.is_lt(at,count),Order.lt_offset(start,at,count)) : {_ == True{} : Bool}
    at_inside : {Nat.is_lt(position,Nat.add(start,count)) == True{} : Bool}}
  +row = Nat.div(position,width)
  +column = Nat.mod(position,width)
  +row_valid = DivisionOrder.quotient_below(position,ConvSpec.out_height(problem),width,width_ok,
    Bounds.nat_trans(position,Nat.add(start,count),Nat.mul(ConvSpec.out_height(problem),width),inside,grid))
  +ordered = Positions.block_le_row(problem,first,rows,start,count,cover,position,stride_ok,width_ok,after)
  +virtual = Nat.sub(row,skip)
  +window = window(problem,offset,span)
  +origin = pack_start(problem,offset)
  +moved = {%Equal.sym(Nat,U32.to_nat(origin),Nat.sub(start,skipped),start_exact) :
      {Nat.add(_,at) == Nat.add(Nat.mul(virtual,width),column) : Nat}
    Positions.window_position(start,at,width,skip,width_ok,reach,ordered) :
    {Nat.add(U32.to_nat(origin),at) == Coordinates.position(window,virtual,column) : Nat}}
  +local_exact = {%moved : {Nat.sub(_,U32.to_nat(origin)) == at : Nat}
    Order.subtract_shift(U32.to_nat(origin),at) : {Blocked.local_position(window,origin,virtual,column) == at : Nat}}
  +point = point(limits,window,origin,span,channel,virtual,column,
    cover_valid(problem,first,rows,start,count,valid,cover),cover_supported(limits,problem,first,rows,start,count,valid,supported,cover),
    covered_region(limits,problem,valid,supported,offset,span,positive,grid,cover),channel_inside,
    Positions.virtual_row_inside(problem,first,rows,start,count,cover,position,stride_ok,width_ok,after,inside,row_valid),
    DivisionOrder.remainder_small(position,width,width_ok),
    %moved : {Nat.is_le(U32.to_nat(origin),_) == True{} : Bool} Order.le_add(U32.to_nat(origin),at),
    %Equal.sym(Nat,Blocked.local_position(window,origin,virtual,column),at,local_exact) : {Nat.is_lt(_,count) == True{} : Bool} at_inside)
  +indices = MatrixSpec.kchannels(ci,0n,kernel)
  +transfer = Samples.matrix_dot(problem,first,rows,skip,channel,virtual,column,cover_first(problem,first,rows,start,count,cover),
    cover_fits(problem,first,rows,start,count,cover),
    Positions.virtual_top(problem,first,rows,start,count,cover,position,stride_ok,width_ok,after,inside),
    Positions.virtual_bottom(problem,first,rows,start,count,cover,position,stride_ok,width_ok,after,inside))
  +restored = Equal.cong(Nat,F32,value => Values.matrix_dot(indices,problem,channel,value,column,0.0),Nat.add(skip,virtual),row,
    Order.restore_shift(skip,row,ordered))
  +same = Equal.cong(F32,F32,dot => (dot + ConvSpec.bias_at(problem,channel) : F32),Values.matrix_dot(indices,window,channel,virtual,column,0.0),
    Values.matrix_dot(indices,problem,channel,row,column,0.0),
    Equal.trans(F32,Values.matrix_dot(indices,window,channel,virtual,column,0.0),Values.matrix_dot(indices,problem,channel,Nat.add(skip,virtual),column,0.0),
      Values.matrix_dot(indices,problem,channel,row,column,0.0),transfer,restored))
  Located{virtual,column,point,local_exact,same}
