# Output addresses for a plane of any base and length.
import Base
import ./encoded_coordinates.bend as Encoded
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../kernel_input_gradient.bend as Kernel
import ./kernel_accumulator.bend as Accumulator
import ./input_gradient_tiled_job_store.bend as Store
import ./input_gradient_tiled_store_geometry.bend as Channels
import ./input_gradient_tiled_store_columns.bend as Columns
import ./kernel_strided_store_writes.bend as StoreModel
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_jobs.bend as NativeJobs
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./storage_shape.bend as Shape
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_phase_domain.bend as Domain
import ./strided_range_bounds.bend as Strides
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition

def rows(+query: Chunks.Query) -> Nat:
  U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query)))
def width(+query: Chunks.Query) -> Nat:
  Domain.width(Chunks.geometry(query))
def pixels(+query: Chunks.Query) -> Nat:
  Nat.mul(rows(query),width(query))

type Layout<-query: Chunks.Query> is Data:
  Layout{capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    plane_span: {Nat.is_le(pixels(query),U32.to_nat(Chunks.plane(query))) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(Domain.channels(Chunks.geometry(query)),U32.to_nat(Chunks.plane(query)))),U32.to_nat(capacity)) == True{} : Bool}}
def capacity(+query: Chunks.Query,layout: Layout<query>) -> U32:
  Layout{capacity,capacity_ok,plane_span,room} = layout
  capacity
def capacity_ok(+query: Chunks.Query,layout: Layout<query>) -> {U32.is_le(capacity(query,layout),2147483648) == True{} : Bool}:
  Layout{capacity,capacity_ok,plane_span,room} = layout
  capacity_ok
def plane_span(+query: Chunks.Query,layout: Layout<query>) -> {Nat.is_le(pixels(query),U32.to_nat(Chunks.plane(query))) == True{} : Bool}:
  Layout{capacity,capacity_ok,plane_span,room} = layout
  plane_span
def room(+query: Chunks.Query,layout: Layout<query>) ->
  {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(Domain.channels(Chunks.geometry(query)),U32.to_nat(Chunks.plane(query)))),U32.to_nat(capacity(query,layout))) == True{} : Bool}:
  Layout{capacity,capacity_ok,plane_span,room} = layout
  room

type Point<-query: Chunks.Query> is Data:
  Point{channel: Nat,row: Nat,column: Nat,
    channel_inside: {Nat.is_lt(channel,Domain.channels(Chunks.geometry(query))) == True{} : Bool},
    row_lower: {Nat.is_le(U32.to_nat(Chunks.first(query)),row) == True{} : Bool},
    row_upper: {Nat.is_lt(row,Nat.add(U32.to_nat(Chunks.first(query)),rows(query))) == True{} : Bool},
    column_inside: {Nat.is_lt(column,width(query)) == True{} : Bool}}
def channel(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{channel,row,column,ci,rl,ru,xi} = point
  channel
def row(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{channel,row,column,ci,rl,ru,xi} = point
  row
def column(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{channel,row,column,ci,rl,ru,xi} = point
  column
def local_row(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.sub(row(query,point),U32.to_nat(Chunks.first(query)))
def pixel(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.add(Nat.mul(local_row(query,point),width(query)),column(query,point))
def address(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.add(Strides.start(U32.to_nat(Chunks.target(query)),0n,U32.to_nat(Chunks.plane(query)),channel(query,point)),pixel(query,point))
def native_address(+query: Chunks.Query,+point: Point<query>) -> U32:
  U32.from_nat(address(query,point))
def pixel_inside(+query: Chunks.Query,+point: Point<query>) -> {Nat.is_lt(pixel(query,point),pixels(query)) == True{} : Bool}:
  Point{channel,+row,+column,ci,+rl,+ru,+xi} = point
  RowMajor.radix(Nat.sub(row,U32.to_nat(Chunks.first(query))),rows(query),column,width(query),Order.sub_lt(row,U32.to_nat(Chunks.first(query)),rows(query),rl,ru),xi)
def address_inside(+query: Chunks.Query,+layout: Layout<query>,+point: Point<query>) -> {Nat.is_lt(address(query,point),U32.to_nat(capacity(query,layout))) == True{} : Bool}:
  Layout{+capacity,capacity_ok,+plane_span,+room} = layout
  Point{+channel,row,column,+ci,rl,ru,xi} = point
  Strides.query_inside(U32.to_nat(Chunks.target(query)),0n,U32.to_nat(Chunks.plane(query)),pixels(query),Domain.channels(Chunks.geometry(query)),channel,pixel(query,point),U32.to_nat(capacity),ci,pixel_inside(query,point),plane_span,room)
def native_address_exact(+query: Chunks.Query,+layout: Layout<query>,+point: Point<query>) -> {U32.to_nat(native_address(query,point)) == address(query,point) : Nat}:
  Layout{+capacity,+capacity_ok,plane_span,room} = layout
  Bounds.roundtrip(address(query,point),capacity,capacity_ok,Bounds.strict_le(address(query,point),U32.to_nat(capacity),address_inside(query,layout,point)))

# Every enabled cell of the real store supplies these layout coordinates.
def job_point(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row_key: Accumulator.Row,+lane: Nat,
  +row_inside: {Nat.is_lt(Store.row_number(row_key),U32.to_nat(StoreModel.store_rows(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,U32.to_nat(StoreModel.store_count(NativeJobs.job_store(Runtime.job_index(seed),Runtime.plan(seed))))) == True{} : Bool}) -> Point<Chunks.query(Chunks.selection(seed))>:
  +query = Chunks.query(Chunks.selection(seed))
  +first = Chunks.first(query)
  +count = GeometryChunks.grid_count(Chunks.grid(query))
  +channel_job = Channels.channel_job(seed,valid)
  +channel = Nat.add(Channels.first(channel_job),Store.row_number(row_key))
  +physical_row = Jobs.physical_row(Chunks.work(Chunks.selection(seed)),Chunks.line(seed))
  +physical_column = Columns.physical_column(seed,lane)
  +row_span = Chunks.row_in_leaf(seed,valid)
  Point{channel,physical_row,physical_column,
    Channels.row_inside(channel_job,Store.row_number(row_key),%Channels.rows_matches(seed,valid) : {Nat.is_lt(Store.row_number(row_key),U32.to_nat(_)) == True{} : Bool} row_inside),
    Shape.first({Nat.is_le(U32.to_nat(first),physical_row) == True{} : Bool},{Nat.is_lt(physical_row,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span),
    Shape.second({Nat.is_le(U32.to_nat(first),physical_row) == True{} : Bool},{Nat.is_lt(physical_row,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span),
    Columns.physical_column_inside(seed,valid,lane,lane_inside)}

# Encode the output coordinate once, using the existing modular address bridge.
def coordinate_address(+query: Chunks.Query,channel_word: U32,row_word: U32,column_word: U32) -> U32:
  (Chunks.target(query) + channel_word * Chunks.plane(query) + row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32)
def coordinate_address_matches(+query: Chunks.Query,+point: Point<query>,+channel_word: U32,+row_word: U32,+column_word: U32,
  +channel_exact: {U32.to_nat(channel_word) == channel(query,point) : Nat},
  +row_exact: {U32.to_nat(row_word) == local_row(query,point) : Nat},
  +column_exact: {U32.to_nat(column_word) == column(query,point) : Nat}) ->
  {coordinate_address(query,channel_word,row_word,column_word) == native_address(query,point) : U32}:
  Chunks.Query{+grid,leaf_first,+base,+plane_word,source,chunk_row,chunk_group,wb,ob} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,s,p,gh,gw} = g
  +channel_product = Nat.mul(channel(query,point),U32.to_nat(plane_word))
  +base_nat = Nat.add(U32.to_nat(base),channel_product)
  +base_word = (base + channel_word * plane_word : U32)
  +base_exact = {Equal.trans(U32,base_word,(channel_word * plane_word + base : U32),U32.from_nat(base_nat),Arithmetic.add_commutative(base,(channel_word * plane_word : U32)),
    Equal.trans(U32,(channel_word * plane_word + base : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(channel_word),U32.to_nat(plane_word)),U32.to_nat(base))),U32.from_nat(base_nat),Encoded.native_madd(channel_word,plane_word,base),
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.mul(U32.to_nat(channel_word),U32.to_nat(plane_word)),U32.to_nat(base)),base_nat,
        Equal.trans(Nat,Nat.add(Nat.mul(U32.to_nat(channel_word),U32.to_nat(plane_word)),U32.to_nat(base)),Nat.add(channel_product,U32.to_nat(base)),base_nat,
          Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(plane_word)),U32.to_nat(base)),U32.to_nat(channel_word),channel(query,point),channel_exact),Order.commutative(channel_product,U32.to_nat(base)))))) : {base_word == U32.from_nat(base_nat) : U32}}
  %Equal.sym(U32,base_word,U32.from_nat(base_nat),base_exact) : {(_ + row_word * w + column_word : U32) == native_address(query,point) : U32}
  %Equal.cong(Nat,Nat,value => Nat.add(value,pixel(query,point)),base_nat,Strides.start(U32.to_nat(base),0n,U32.to_nat(plane_word),channel(query,point)),Equal.cong(Nat,Nat,value => Nat.add(value,channel_product),U32.to_nat(base),Nat.add(U32.to_nat(base),0n),Equal.sym(Nat,Nat.add(U32.to_nat(base),0n),U32.to_nat(base),Addition.zero_right(U32.to_nat(base))))) : {(U32.from_nat(base_nat) + row_word * w + column_word : U32) == U32.from_nat(_) : U32}
  %row_exact : {(U32.from_nat(base_nat) + row_word * w + column_word : U32) == U32.from_nat(Nat.add(base_nat,Nat.add(Nat.mul(_,width(query)),column(query,point)))) : U32}
  %column_exact : {(U32.from_nat(base_nat) + row_word * w + column_word : U32) == U32.from_nat(Nat.add(base_nat,Nat.add(Nat.mul(U32.to_nat(row_word),width(query)),_))) : U32}
  Encoded.row_offset_encoded(base_nat,row_word,w,column_word)
def address_congruence(+query: Chunks.Query,+left: Point<query>,+right: Point<query>,
  +channel_equal: {channel(query,left) == channel(query,right) : Nat},+row_equal: {row(query,left) == row(query,right) : Nat},+column_equal: {column(query,left) == column(query,right) : Nat}) -> {address(query,left) == address(query,right) : Nat}:
  +base = U32.to_nat(Chunks.target(query))
  +stride = U32.to_nat(Chunks.plane(query))
  +width = width(query)
  +first = U32.to_nat(Chunks.first(query))
  +channel_changed = Nat.add(Strides.start(base,0n,stride,channel(query,right)),pixel(query,left))
  +row_changed = Nat.add(Strides.start(base,0n,stride,channel(query,right)),Nat.add(Nat.mul(Nat.sub(row(query,right),first),width),column(query,left)))
  Equal.trans(Nat,address(query,left),channel_changed,address(query,right),Equal.cong(Nat,Nat,value => Nat.add(Strides.start(base,0n,stride,value),pixel(query,left)),channel(query,left),channel(query,right),channel_equal),
    Equal.trans(Nat,channel_changed,row_changed,address(query,right),Equal.cong(Nat,Nat,value => Nat.add(Strides.start(base,0n,stride,channel(query,right)),Nat.add(Nat.mul(Nat.sub(value,first),width),column(query,left))),row(query,left),row(query,right),row_equal),
      Equal.cong(Nat,Nat,value => Nat.add(Strides.start(base,0n,stride,channel(query,right)),Nat.add(Nat.mul(Nat.sub(row(query,right),first),width),value)),column(query,left),column(query,right),column_equal)))

# Chunk positions and phase packing cursors do not change the output region.
def scheduled(+query: Chunks.Query,row: Nat,group: Nat,weight_base: U32,offset_base: U32) -> Chunks.Query:
  Chunks.Query{Chunks.grid(query),Chunks.first(query),Chunks.target(query),Chunks.plane(query),Chunks.source(query),row,group,weight_base,offset_base}
# The output region of a query: its points and their addresses do not depend
# on the chunk or the packing bases.
def region(+query: Chunks.Query) -> Chunks.Query:
  scheduled(query,0n,0n,0,0)
def region_point(+query: Chunks.Query,point: Point<query>) -> Point<region(query)>:
  Point{channel,row,column,ci,rl,ru,xi} = point
  Point{channel,row,column,ci,rl,ru,xi}
def region_layout(+query: Chunks.Query,+layout: Layout<query>) -> Layout<region(query)>:
  Layout{capacity(query,layout),capacity_ok(query,layout),plane_span(query,layout),room(query,layout)}
