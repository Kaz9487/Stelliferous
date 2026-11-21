# The planes of a call: the words of the planes the transpose writes for
# count positions from start are exact, so the planes are a block of those
# positions.
import Base
import ../kernel_spatial_packing.bend as Packing
import ./kernel_spatial_packing_planes.bend as Planes
import ./kernel_spatial_packing_passes.bend as Passes
import ./kernel_spatial_packing_pixels.bend as Pixels
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as WordDivision
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./encoded_coordinates.bend as Encoded

# The last position of a call.
def call_last(+start: U32,+count: U32) -> Nat:
  Nat.sub(Nat.add(U32.to_nat(start),U32.to_nat(count)),1n)

# The output rows from the first position of a call to its last, stride
# input rows each.
def call_moved(+window: Packing.Stencil,+start: U32,+count: U32) -> Nat:
  Nat.mul(Nat.sub(Nat.div(call_last(start,count),Pixels.output_width_of(window)),Nat.div(U32.to_nat(start),Pixels.output_width_of(window))),Pixels.stride_of(window))

# The input rows of the planes of a call: the moved rows and the taps.
def call_rows(+window: Packing.Stencil,+start: U32,+count: U32) -> Nat:
  Nat.add(call_moved(window,start,count),Passes.tap_rows_of(window))

# The columns of a phase of a row: the output width and the taps past it.
def call_reach(+window: Packing.Stencil) -> Nat:
  Nat.add(Pixels.output_width_of(window),Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),Pixels.stride_of(window)))

# The tiles of a row of a plane.
def call_tiles(+window: Packing.Stencil) -> Nat:
  Nat.div(Nat.add(call_reach(window),7n),8n)

# What the planes of count positions from start need of their window, their
# positions and the depth of the pairs that hold them.
type CallRoom<-window: Packing.Stencil,-start: U32,-count: U32,-depth: Nat,-cap: U32> is Data:
  CallRoom{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
    width_bound: {Nat.is_le(Nat.double(Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
    stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
    stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
    taps_positive: {Nat.is_lt(0n,Passes.tap_rows_of(window)) == True{} : Bool},columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
    lanes_room: {Nat.is_le(Nat.add(Pixels.output_width_of(window),8n),U32.to_nat(cap)) == True{} : Bool},columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},
    tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool},
    channels_positive: {Nat.is_lt(0n,Planes.channel_count(window)) == True{} : Bool},
    divisor_bound: {Nat.is_le(Nat.double(Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
    planes_bound: {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
    count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
    positions_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
    area_room: {Nat.is_le(Nat.mul(call_rows(window,start,count),Nat.mul(call_tiles(window),8n)),U32.to_nat(cap)) == True{} : Bool},
    limit: {Nat.is_le(depth,31n) == True{} : Bool},capped: {Nat.is_le(U32.to_nat(U32.shln(1,depth)),U32.to_nat(cap)) == True{} : Bool},
    fits: {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Nat.mul(call_rows(window,start,count),call_tiles(window))),Planes.channel_count(window)),
      U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}

# The tiles of a row cover its reach.
def reach_covered(+window: Packing.Stencil) -> {Nat.is_le(call_reach(window),Nat.mul(call_tiles(window),8n)) == True{} : Bool}:
  Encoded.within_ceiling(call_reach(window))

def width_positive(+window: Packing.Stencil,+positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.mul(call_tiles(window),8n)) == True{} : Bool}:
  +reach = call_reach(window)
  Bounds.nat_trans(0n,reach,Nat.mul(call_tiles(window),8n),
    Bounds.nat_trans(0n,Pixels.output_width_of(window),reach,positive,
      Order.le_add(Pixels.output_width_of(window),Nat.div(Nat.sub(Passes.tap_columns_of(window),1n),Pixels.stride_of(window)))),
    reach_covered(window))

# The reach of a row and its rounding fit the room of its tiles.
def reach_room(+window: Packing.Stencil,+cap: U32,+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(call_reach(window),7n),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(Nat.add(call_reach(window),7n),Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap),
    Order.add_le(call_reach(window),Nat.mul(call_tiles(window),8n),7n,reach_covered(window)),tiles_room)

# The first output row of the planes.
def first_exact(+window: Packing.Stencil,+start: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
  +width_bound: {Nat.is_le(Nat.double(Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Pixels.first_row_of(Packing.planes_of(window,start,count)) == Nat.div(U32.to_nat(start),Pixels.output_width_of(window)) : Nat}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,stride,row_origin,column_origin,+output_width,inner,groups} = window
  WordDivision.quotient_exact(cap,cap_ok,start,output_width,U32.to_nat(start),U32.to_nat(output_width),{==},{==},width_bound,positive)

# The row width of the planes: the tiles of a row, eight positions each.
def width_exact(+window: Packing.Stencil,+start: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
  +columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Pixels.row_width_of(Packing.planes_of(window,start,count)) == Nat.mul(call_tiles(window),8n) : Nat}:
  Packing.Stencil{base,height,width,tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +tw = U32.to_nat(tap_columns)
  +quotient = Nat.div(Nat.sub(tw,1n),s)
  +reach = Nat.add(ow,quotient)
  +total = Nat.add(reach,7n)
  +tiles = Nat.div(total,8n)
  +total_room = reach_room(window,cap,tiles_room)
  +reach_bound = Order.transitive(reach,total,U32.to_nat(cap),Order.le_add(reach,7n),total_room)
  +less = Arithmetic.difference_exact(cap,cap_ok,tap_columns,1,tw,1n,{==},{==},Order.positive_one(tw,columns_positive),columns_room)
  +divided = WordDivision.quotient_exact(cap,cap_ok,(tap_columns - 1 : U32),stride,Nat.sub(tw,1n),s,less,{==},stride_bound,stride_positive)
  +reached = Arithmetic.sum_exact(cap,cap_ok,output_width,((tap_columns - 1) / stride : U32),ow,quotient,{==},divided,reach_bound)
  +rounded = Arithmetic.sum_exact(cap,cap_ok,(output_width + (tap_columns - 1) / stride : U32),7,reach,7n,reached,{==},total_room)
  +grouped = WordDivision.quotient_exact(cap,cap_ok,(output_width + (tap_columns - 1) / stride + 7 : U32),8,total,8n,rounded,{==},sixteen,{==})
  Arithmetic.product_exact(cap,cap_ok,((output_width + (tap_columns - 1) / stride + 7) / 8 : U32),8,tiles,8n,grouped,{==},
    Order.transitive(Nat.mul(tiles,8n),total,U32.to_nat(cap),DivisionOrder.product_below(total,8n,{==}),total_room))

# The tiles of a row of the planes.
def tiles_exact(+window: Packing.Stencil,+start: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},+stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
  +columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool}) ->
  {Planes.tiles_of(Packing.planes_of(window,start,count)) == call_tiles(window) : Nat}:
  Packing.Stencil{base,height,width,tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +tiles = call_tiles(window)
  +row_width = ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8 : U32)
  Equal.trans(Nat,U32.to_nat((row_width / 8 : U32)),Nat.div(Nat.mul(tiles,8n),8n),tiles,
    WordDivision.quotient_exact(cap,cap_ok,row_width,8,Nat.mul(tiles,8n),8n,
      width_exact(window,start,count,cap,cap_ok,sixteen,stride_positive,stride_bound,columns_positive,columns_room,tiles_room),{==},sixteen,{==}),
    Division.exact_quotient(tiles,8n,{==}))

# The elements of a plane: its rows times the tiles of a row.
def elements_exact(+window: Packing.Stencil,+start: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
  +width_bound: {Nat.is_le(Nat.double(Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
  +columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +area_room: {Nat.is_le(Nat.mul(call_rows(window,start,count),Nat.mul(call_tiles(window),8n)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Planes.elements_of(Packing.planes_of(window,start,count)) == Nat.mul(call_rows(window,start,count),call_tiles(window)) : Nat}:
  Packing.Stencil{base,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +st = U32.to_nat(start)
  +n = U32.to_nat(count)
  +sum = Nat.add(st,n)
  +last = Nat.sub(sum,1n)
  +first = Nat.div(st,ow)
  +upper = Nat.div(last,ow)
  +difference = Nat.sub(upper,first)
  +moved = Nat.mul(difference,s)
  +rows = call_rows(window,start,count)
  +tiles = call_tiles(window)
  +width = Nat.mul(tiles,8n)
  +area = Nat.mul(rows,width)
  +rows_room = Order.transitive(rows,area,U32.to_nat(cap),RowMajor.product_covers(rows,width,width_positive(window,positive)),area_room)
  +moved_room = Order.transitive(moved,rows,U32.to_nat(cap),Order.le_add(moved,U32.to_nat(tap_rows)),rows_room)
  +inside = {
    %Addition.zero_right(st) : {Nat.is_lt(_,sum) == True{} : Bool}
    Order.offset_lt(st,0n,n,count_positive) : {Nat.is_lt(st,sum) == True{} : Bool}}
  +sum_positive = Order.le_lt_trans(0n,st,sum,Order.zero_le(st),inside)
  +last_room = Order.transitive(last,sum,U32.to_nat(cap),Order.sub_le(sum,1n),positions_room)
  +upper_room = Order.transitive(upper,last,U32.to_nat(cap),DivisionOrder.quotient_at_most(last,ow,positive),last_room)
  +ordered = DivisionOrder.quotient_le(st,last,ow,positive,Order.before_last(st,sum,inside))
  +summed = Arithmetic.sum_exact(cap,cap_ok,start,count,st,n,{==},{==},positions_room)
  +lasted = Arithmetic.difference_exact(cap,cap_ok,(start + count : U32),1,sum,1n,summed,{==},Order.positive_one(sum,sum_positive),positions_room)
  +uppered = WordDivision.quotient_exact(cap,cap_ok,(start + count - 1 : U32),output_width,last,ow,lasted,{==},width_bound,positive)
  +firsted = WordDivision.quotient_exact(cap,cap_ok,start,output_width,st,ow,{==},{==},width_bound,positive)
  +differenced = Arithmetic.difference_exact(cap,cap_ok,((start + count - 1) / output_width : U32),(start / output_width : U32),upper,first,uppered,firsted,ordered,
    upper_room)
  +multiplied = Arithmetic.product_exact(cap,cap_ok,((start + count - 1) / output_width - start / output_width : U32),stride,difference,s,differenced,{==},moved_room)
  +rows_word = (((start + count - 1) / output_width - start / output_width) * stride + tap_rows : U32)
  +rowed = Arithmetic.sum_exact(cap,cap_ok,(((start + count - 1) / output_width - start / output_width) * stride : U32),tap_rows,moved,U32.to_nat(tap_rows),multiplied,{==},
    rows_room)
  +width_word = ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8 : U32)
  +areaed = Arithmetic.product_exact(cap,cap_ok,rows_word,width_word,rows,width,rowed,
    width_exact(window,start,count,cap,cap_ok,sixteen,stride_positive,stride_bound,columns_positive,columns_room,tiles_room),area_room)
  Equal.trans(Nat,U32.to_nat((rows_word * width_word / 8 : U32)),Nat.div(area,8n),Nat.mul(rows,tiles),
    WordDivision.quotient_exact(cap,cap_ok,(rows_word * width_word : U32),8,area,8n,areaed,{==},sixteen,{==}),
    %RowMajor.assoc(rows,tiles,8n) : {Nat.div(_,8n) == Nat.mul(rows,tiles) : Nat}
    Division.exact_quotient(Nat.mul(rows,tiles),8n,{==}))

# The planes the transpose writes: the stride's phases times the channels.
def planes_exact(+window: Packing.Stencil,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +planes_bound: {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Planes.plane_count(window) == Nat.mul(Planes.phases_of(window),Planes.channel_count(window)) : Nat}:
  Packing.Stencil{base,height,width,tap_rows,tap_columns,+stride,row_origin,column_origin,output_width,inner,groups} = window
  Arithmetic.product_exact(cap,cap_ok,stride,Packing.channels_of(window),U32.to_nat(stride),Planes.channel_count(window),{==},{==},planes_bound)

# A call on some positions has the planes of those positions.
def call_planes_of(+window: Packing.Stencil,+start: U32,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {Packing.call_planes(window,start,count) == Packing.planes_of(window,start,count) : Packing.Planes}:
  %Equal.sym(Bool,U32.is_zero(count),False{},Arithmetic.word_positive_nonzero(count,positive)) :
    {Packing.call_planes_when(_,window,start,count) == Packing.planes_of(window,start,count) : Packing.Planes}
  {==}

# The pairs of the planes of count positions from start: the planes times the
# elements of a plane.
def pairs_exact(+window: Packing.Stencil,+start: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
  +width_bound: {Nat.is_le(Nat.double(Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},
  +columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +positions_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +area_room: {Nat.is_le(Nat.mul(call_rows(window,start,count),Nat.mul(call_tiles(window),8n)),U32.to_nat(cap)) == True{} : Bool},
  +planes_bound: {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +pairs_room: {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(call_rows(window,start,count),call_tiles(window))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Packing.plane_pairs(window,Packing.planes_of(window,start,count))) ==
    Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(call_rows(window,start,count),call_tiles(window))) : Nat}:
  Packing.Stencil{base,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +rows_word = (((start + count - 1) / output_width - start / output_width) * stride + tap_rows : U32)
  +width_word = ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8 : U32)
  +elemented = elements_exact(window,start,count,cap,cap_ok,sixteen,positive,width_bound,stride_positive,stride_bound,columns_positive,columns_room,tiles_room,count_positive,
    positions_room,area_room)
  +planed = Arithmetic.product_exact(cap,cap_ok,stride,Packing.channels_of(window),U32.to_nat(stride),Planes.channel_count(window),{==},{==},planes_bound)
  Arithmetic.product_exact(cap,cap_ok,(stride * Packing.channels_of(window) : U32),(rows_word * width_word / 8 : U32),
    Nat.mul(U32.to_nat(stride),Planes.channel_count(window)),Nat.mul(call_rows(window,start,count),call_tiles(window)),planed,elemented,pairs_room)

# The planes of count positions from start are a block of units eight
# positions.
def call_block(+window: Packing.Stencil,+start: U32,+count: U32,+units: Nat,+depth: Nat,+cap: U32,+room: CallRoom<window,start,count,depth,cap>,
  +units_exact: {U32.to_nat(count) == Nat.mul(units,8n) : Nat}) -> Pixels.Block<window,Packing.planes_of(window,start,count),start,units,depth,cap>:
  CallRoom{+cap_ok,+sixteen,+positive,+width_bound,+stride_positive,+stride_bound,+taps_positive,+columns_positive,+lanes_room,+columns_room,+tiles_room,+channels_positive,+divisor_bound,
    +planes_bound,+count_positive,+positions_room,+area_room,+limit,+capped,+fits} = room
  +planes = Packing.planes_of(window,start,count)
  +ow = Pixels.output_width_of(window)
  +s = Pixels.stride_of(window)
  +th = Passes.tap_rows_of(window)
  +tw = Passes.tap_columns_of(window)
  +st = U32.to_nat(start)
  +rows = call_rows(window,start,count)
  +tiles = call_tiles(window)
  +reach = call_reach(window)
  +firsted = first_exact(window,start,count,cap,cap_ok,positive,width_bound)
  +widthed = width_exact(window,start,count,cap,cap_ok,sixteen,stride_positive,stride_bound,columns_positive,columns_room,tiles_room)
  +tiled = tiles_exact(window,start,count,cap,cap_ok,sixteen,stride_positive,stride_bound,columns_positive,columns_room,tiles_room)
  +elemented = elements_exact(window,start,count,cap,cap_ok,sixteen,positive,width_bound,stride_positive,stride_bound,columns_positive,columns_room,tiles_room,count_positive,
    positions_room,area_room)
  +tiles_positive = RowMajor.cancel_multiply_lt(0n,tiles,8n,width_positive(window,positive))
  +total_room = reach_room(window,cap,tiles_room)
  +tiles_room = Order.transitive(tiles,Nat.mul(tiles,8n),U32.to_nat(cap),RowMajor.product_covers(tiles,8n,{==}),
    Order.transitive(Nat.mul(tiles,8n),Nat.add(reach,7n),U32.to_nat(cap),DivisionOrder.product_below(Nat.add(reach,7n),8n,{==}),total_room))
  +rows_positive = Bounds.nat_trans(0n,th,rows,taps_positive,Order.right_le(call_moved(window,start,count),th))
  +room_fits = {
    %Equal.sym(Nat,Planes.elements_of(planes),Nat.mul(rows,tiles),elemented) :
      {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),_),Planes.channel_count(window)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
    fits : {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.elements_of(planes)),Planes.channel_count(window)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}}
  +elements_positive = {
    %Equal.sym(Nat,Planes.elements_of(planes),Nat.mul(rows,tiles),elemented) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(rows,tiles,rows_positive,tiles_positive) : {Nat.is_lt(0n,Planes.elements_of(planes)) == True{} : Bool}}
  +tiles_bound = {
    %Equal.sym(Nat,Planes.tiles_of(planes),tiles,tiled) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    tiles_room : {Nat.is_le(Planes.tiles_of(planes),U32.to_nat(cap)) == True{} : Bool}}
  +tiles_present = {
    %Equal.sym(Nat,Planes.tiles_of(planes),tiles,tiled) : {Nat.is_lt(0n,_) == True{} : Bool}
    tiles_positive : {Nat.is_lt(0n,Planes.tiles_of(planes)) == True{} : Bool}}
  +geometry = {Planes.Geometry{Planes.Room{cap_ok,limit,room_fits,capped,elements_positive,channels_positive},tiles_bound,tiles_present,divisor_bound,planes_bound,
    planes_exact(window,cap,cap_ok,planes_bound)} : Planes.Geometry<window,planes,depth,cap>}
  +width = Nat.mul(tiles,8n)
  +ordered = {
    %Equal.sym(Nat,Pixels.row_width_of(planes),width,widthed) : {Nat.is_le(ow,Nat.mul(s,_)) == True{} : Bool}
    %RowMajor.multiply_commutative(width,s) : {Nat.is_le(ow,_) == True{} : Bool}
    Order.transitive(ow,width,Nat.mul(width,s),
      Order.transitive(ow,reach,width,Order.le_add(ow,Nat.div(Nat.sub(tw,1n),s)),reach_covered(window)),RowMajor.product_covers(width,s,stride_positive)) :
      {Nat.is_le(ow,Nat.mul(s,Pixels.row_width_of(planes))) == True{} : Bool}}
  +row_exact = {
    %Equal.sym(Nat,Pixels.row_width_of(planes),width,widthed) : {_ == Nat.mul(Planes.tiles_of(planes),8n) : Nat}
    %Equal.sym(Nat,Planes.tiles_of(planes),tiles,tiled) : {width == Nat.mul(_,8n) : Nat}
    {==} : {Pixels.row_width_of(planes) == Nat.mul(Planes.tiles_of(planes),8n) : Nat}}
  +width_fits = {
    %Equal.sym(Nat,Planes.tiles_of(planes),tiles,tiled) : {Nat.is_le(reach,Nat.mul(_,8n)) == True{} : Bool}
    reach_covered(window) : {Nat.is_le(reach,Nat.mul(Planes.tiles_of(planes),8n)) == True{} : Bool}}
  +values_room = {
    %Equal.sym(Nat,Planes.elements_of(planes),Nat.mul(rows,tiles),elemented) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(rows,tiles),8n),Nat.mul(rows,width),RowMajor.assoc(rows,tiles,8n)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    area_room : {Nat.is_le(Nat.mul(Planes.elements_of(planes),8n),U32.to_nat(cap)) == True{} : Bool}}
  +below_first = {
    %Equal.sym(Nat,Pixels.first_row_of(planes),Nat.div(st,ow),firsted) : {Nat.is_le(_,Nat.div(st,ow)) == True{} : Bool}
    Order.reflexive(Nat.div(st,ow)) : {Nat.is_le(Pixels.first_row_of(planes),Nat.div(st,ow)) == True{} : Bool}}
  +covered = {
    %units_exact :
      {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.sub(Nat.div(Nat.sub(Nat.add(st,_),1n),ow),Pixels.first_row_of(planes)),s),th),Planes.tiles_of(planes)),
        Planes.elements_of(planes)) == True{} : Bool}
    %Equal.sym(Nat,Pixels.first_row_of(planes),Nat.div(st,ow),firsted) :
      {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.sub(Nat.div(call_last(start,count),ow),_),s),th),Planes.tiles_of(planes)),Planes.elements_of(planes)) == True{} : Bool}
    %Equal.sym(Nat,Planes.tiles_of(planes),tiles,tiled) : {Nat.is_le(Nat.mul(rows,_),Planes.elements_of(planes)) == True{} : Bool}
    %Equal.sym(Nat,Planes.elements_of(planes),Nat.mul(rows,tiles),elemented) : {Nat.is_le(Nat.mul(rows,tiles),_) == True{} : Bool}
    Order.reflexive(Nat.mul(rows,tiles)) :
      {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.sub(Nat.div(Pixels.block_last(start,units),ow),Pixels.first_row_of(planes)),s),th),Planes.tiles_of(planes)),
        Planes.elements_of(planes)) == True{} : Bool}}
  +ending = {
    %units_exact : {Nat.is_le(Nat.add(st,_),U32.to_nat(cap)) == True{} : Bool}
    positions_room : {Nat.is_le(Pixels.block_end(start,units),U32.to_nat(cap)) == True{} : Bool}}
  Pixels.Block{geometry,sixteen,positive,width_bound,lanes_room,stride_positive,stride_bound,ordered,row_exact,width_fits,columns_room,values_room,below_first,
    covered,ending}

# ---- The room of the planes of any call ----

# The rows Packing.plane_bound allots to count positions.
def bound_rows(+window: Packing.Stencil,+count: U32) -> Nat:
  Nat.add(Nat.mul(Nat.div(Nat.sub(Nat.add(U32.to_nat(count),Pixels.output_width_of(window)),1n),Pixels.output_width_of(window)),Pixels.stride_of(window)),
    Passes.tap_rows_of(window))

# The words of Packing.plane_bound are exact: the planes of the phases and
# channels, the rows times the tiles of a row each.
def bound_exact(+window: Packing.Stencil,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
  +width_bound: {Nat.is_le(Nat.double(Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +stride_positive: {Nat.is_lt(0n,Pixels.stride_of(window)) == True{} : Bool},
  +stride_bound: {Nat.is_le(Nat.double(Pixels.stride_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Passes.tap_columns_of(window)) == True{} : Bool},+columns_room: {Nat.is_le(Passes.tap_columns_of(window),U32.to_nat(cap)) == True{} : Bool},+tiles_room: {Nat.is_le(Nat.add(Nat.mul(call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(count),Pixels.output_width_of(window)),U32.to_nat(cap)) == True{} : Bool},
  +area: {Nat.is_le(Nat.mul(bound_rows(window,count),Nat.mul(call_tiles(window),8n)),U32.to_nat(cap)) == True{} : Bool},
  +planes_bound: {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
  +pairs: {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(bound_rows(window,count),call_tiles(window))),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Packing.plane_bound(window,count)) == Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(bound_rows(window,count),call_tiles(window))) :
    Nat}:
  Packing.Stencil{base,height,width,+tap_rows,+tap_columns,+stride,row_origin,column_origin,+output_width,inner,groups} = window
  +ow = U32.to_nat(output_width)
  +s = U32.to_nat(stride)
  +n = U32.to_nat(count)
  +sum = Nat.add(n,ow)
  +last = Nat.sub(sum,1n)
  +quotient = Nat.div(last,ow)
  +moved = Nat.mul(quotient,s)
  +rows = bound_rows(window,count)
  +tiles = call_tiles(window)
  +width = Nat.mul(tiles,8n)
  +area_value = Nat.mul(rows,width)
  +phases = Nat.mul(s,Planes.channel_count(window))
  +rows_room = Order.transitive(rows,area_value,U32.to_nat(cap),RowMajor.product_covers(rows,width,width_positive(window,positive)),area)
  +moved_room = Order.transitive(moved,rows,U32.to_nat(cap),Order.le_add(moved,U32.to_nat(tap_rows)),rows_room)
  +sum_positive = Bounds.nat_trans(0n,ow,sum,positive,Order.right_le(n,ow))
  +last_room = Order.transitive(last,sum,U32.to_nat(cap),Order.sub_le(sum,1n),count_room)
  +summed = Arithmetic.sum_exact(cap,cap_ok,count,output_width,n,ow,{==},{==},count_room)
  +lasted = Arithmetic.difference_exact(cap,cap_ok,(count + output_width : U32),1,sum,1n,summed,{==},Order.positive_one(sum,sum_positive),count_room)
  +divided = WordDivision.quotient_exact(cap,cap_ok,(count + output_width - 1 : U32),output_width,last,ow,lasted,{==},width_bound,positive)
  +multiplied = Arithmetic.product_exact(cap,cap_ok,((count + output_width - 1) / output_width : U32),stride,quotient,s,divided,{==},moved_room)
  +rows_word = ((count + output_width - 1) / output_width * stride + tap_rows : U32)
  +rowed = Arithmetic.sum_exact(cap,cap_ok,((count + output_width - 1) / output_width * stride : U32),tap_rows,moved,U32.to_nat(tap_rows),multiplied,{==},rows_room)
  +width_word = ((output_width + (tap_columns - 1) / stride + 7) / 8 * 8 : U32)
  +areaed = Arithmetic.product_exact(cap,cap_ok,rows_word,width_word,rows,width,rowed,
    width_exact(window,0,0,cap,cap_ok,sixteen,stride_positive,stride_bound,columns_positive,columns_room,tiles_room),area)
  +elements = Equal.trans(Nat,U32.to_nat((rows_word * width_word / 8 : U32)),Nat.div(area_value,8n),Nat.mul(rows,tiles),
    WordDivision.quotient_exact(cap,cap_ok,(rows_word * width_word : U32),8,area_value,8n,areaed,{==},sixteen,{==}),
    %RowMajor.assoc(rows,tiles,8n) : {Nat.div(_,8n) == Nat.mul(rows,tiles) : Nat}
    Division.exact_quotient(Nat.mul(rows,tiles),8n,{==}))
  +planed = Arithmetic.product_exact(cap,cap_ok,stride,Packing.channels_of(window),s,Planes.channel_count(window),{==},{==},planes_bound)
  Arithmetic.product_exact(cap,cap_ok,(stride * Packing.channels_of(window) : U32),(rows_word * width_word / 8 : U32),phases,Nat.mul(rows,tiles),planed,elements,pairs)

# The rows of the planes of count positions from start are at most those
# Packing.plane_bound allots to bound positions, bound at least count.
def rows_within(+window: Packing.Stencil,+start: U32,+count: U32,+bound: U32,+positive: {Nat.is_lt(0n,Pixels.output_width_of(window)) == True{} : Bool},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+ordered: {Nat.is_le(U32.to_nat(count),U32.to_nat(bound)) == True{} : Bool}) ->
  {Nat.is_le(call_rows(window,start,count),bound_rows(window,bound)) == True{} : Bool}:
  +w = Pixels.output_width_of(window)
  +s = Pixels.stride_of(window)
  +th = Passes.tap_rows_of(window)
  +st = U32.to_nat(start)
  +n = U32.to_nat(count)
  +b = U32.to_nat(bound)
  +q = Nat.div(st,w)
  +r = Nat.mod(st,w)
  +a = Nat.mul(q,w)
  +reach = Nat.add(r,n)
  +value = Nat.sub(reach,1n)
  +upper = Nat.sub(Nat.add(b,w),1n)
  +reach_positive = Bounds.nat_trans(0n,n,reach,count_positive,Order.right_le(r,n))
  +shifted = Equal.trans(Nat,Nat.div(call_last(start,count),w),Nat.div(Nat.sub(Nat.add(Nat.add(a,r),n),1n),w),Nat.add(q,Nat.div(value,w)),
    Equal.cong(Nat,Nat,x => Nat.div(Nat.sub(Nat.add(x,n),1n),w),st,Nat.add(a,r),DivisionOrder.quotient_exact(st,w,positive)),
    Equal.trans(Nat,Nat.div(Nat.sub(Nat.add(Nat.add(a,r),n),1n),w),Nat.div(Nat.sub(Nat.add(a,reach),1n),w),Nat.add(q,Nat.div(value,w)),
      Equal.cong(Nat,Nat,x => Nat.div(Nat.sub(x,1n),w),Nat.add(Nat.add(a,r),n),Nat.add(a,reach),Addition.associative(a,r,n)),
      Equal.trans(Nat,Nat.div(Nat.sub(Nat.add(a,reach),1n),w),Nat.div(Nat.add(a,value),w),Nat.add(q,Nat.div(value,w)),
        Equal.cong(Nat,Nat,x => Nat.div(x,w),Nat.sub(Nat.add(a,reach),1n),Nat.add(a,value),Order.subtract_from_sum(a,reach,1n,Order.positive_one(reach,reach_positive))),
        DivisionOrder.quotient_shift(value,w,q,positive))))
  +rows_exact = Equal.trans(Nat,Nat.sub(Nat.div(call_last(start,count),w),q),Nat.sub(Nat.add(q,Nat.div(value,w)),q),Nat.div(value,w),
    Equal.cong(Nat,Nat,x => Nat.sub(x,q),Nat.div(call_last(start,count),w),Nat.add(q,Nat.div(value,w)),shifted),
    Order.subtract_shift(q,Nat.div(value,w)))
  +remainder_le = Bounds.strict_le(r,w,DivisionOrder.remainder_small(st,w,positive))
  +reach_le = {
    %Order.commutative(w,b) : {Nat.is_le(reach,_) == True{} : Bool}
    Order.transitive(reach,Nat.add(w,n),Nat.add(w,b),Order.add_le(r,w,n,remainder_le),Order.offset_le(w,n,b,ordered)) : {Nat.is_le(reach,Nat.add(b,w)) == True{} : Bool}}
  +value_le = DivisionOrder.subtract_le(reach,Nat.add(b,w),1n,reach_le)
  %Equal.sym(Nat,Nat.sub(Nat.div(call_last(start,count),w),q),Nat.div(value,w),rows_exact) :
    {Nat.is_le(Nat.add(Nat.mul(_,s),th),bound_rows(window,bound)) == True{} : Bool}
  Order.add_le(Nat.mul(Nat.div(value,w),s),Nat.mul(Nat.div(upper,w),s),th,RowMajor.mul_le(Nat.div(value,w),Nat.div(upper,w),s,
    DivisionOrder.quotient_le(value,upper,w,positive,value_le)))
