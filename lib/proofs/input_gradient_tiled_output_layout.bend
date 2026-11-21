# Output cell addresses: the tile groups of a pixel sit side by side, so tile
# group g of pixel p is at target + p * G + g for the G tile groups of the
# channels.
import Base
import ./encoded_coordinates.bend as Encoded
import ./word_arithmetic.bend as Words
import ../kernel_input_gradient.bend as Kernel
import ./input_gradient_tiled_store_geometry.bend as Channels
import ./input_gradient_tiled_store_columns.bend as Columns
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_product_reads.bend as Runtime
import ./kernel_gemm_jobs.bend as GemmJobs
import ./storage_shape.bend as Shape
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_separation.bend as Separation
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_ceiling.bend as Ceiling

def rows(+query: Chunks.Query) -> Nat:
  U32.to_nat(GeometryChunks.grid_count(Chunks.grid(query)))
def width(+query: Chunks.Query) -> Nat:
  Domain.width(Chunks.geometry(query))
def pixels(+query: Chunks.Query) -> Nat:
  Nat.mul(rows(query),width(query))
def groups(+query: Chunks.Query) -> Nat:
  Ceiling.ceiling(Domain.channels(Chunks.geometry(query)),8n)
def cells(+query: Chunks.Query) -> U32:
  Separation.block_count(Chunks.geometry(query))
def cells_exact(+query: Chunks.Query) -> {U32.to_nat(cells(query)) == groups(query) : Nat}:
  Chunks.Query{+grid,first,target,source,row,group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  +channels = Domain.channels(Domain.geometry(call))
  Equal.trans(Nat,U32.to_nat(Separation.block_count(Domain.geometry(call))),Nat.div(Nat.add(channels,7n),8n),Ceiling.ceiling(channels,8n),
    Separation.block_count_exact(call),Ceiling.shifted_formula(channels,8n,{==}))

type Layout<-query: Chunks.Query> is Data:
  Layout{capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(pixels(query),groups(query))),U32.to_nat(capacity)) == True{} : Bool}}
def capacity(+query: Chunks.Query,layout: Layout<query>) -> U32:
  Layout{capacity,capacity_ok,room} = layout
  capacity
def capacity_ok(+query: Chunks.Query,layout: Layout<query>) -> {U32.is_le(capacity(query,layout),2147483648) == True{} : Bool}:
  Layout{capacity,capacity_ok,room} = layout
  capacity_ok
def room(+query: Chunks.Query,layout: Layout<query>) ->
  {Nat.is_le(Nat.add(U32.to_nat(Chunks.target(query)),Nat.mul(pixels(query),groups(query))),U32.to_nat(capacity(query,layout))) == True{} : Bool}:
  Layout{capacity,capacity_ok,room} = layout
  room

type Point<-query: Chunks.Query> is Data:
  Point{group: Nat,row: Nat,column: Nat,
    group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
    row_lower: {Nat.is_le(U32.to_nat(Chunks.first(query)),row) == True{} : Bool},
    row_upper: {Nat.is_lt(row,Nat.add(U32.to_nat(Chunks.first(query)),rows(query))) == True{} : Bool},
    column_inside: {Nat.is_lt(column,width(query)) == True{} : Bool}}
def group(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{group,row,column,gi,rl,ru,xi} = point
  group
def row(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{group,row,column,gi,rl,ru,xi} = point
  row
def column(+query: Chunks.Query,point: Point<query>) -> Nat:
  Point{group,row,column,gi,rl,ru,xi} = point
  column
def local_row(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.sub(row(query,point),U32.to_nat(Chunks.first(query)))
def pixel(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.add(Nat.mul(local_row(query,point),width(query)),column(query,point))
def offset(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.add(Nat.mul(pixel(query,point),groups(query)),group(query,point))
def address(+query: Chunks.Query,+point: Point<query>) -> Nat:
  Nat.add(U32.to_nat(Chunks.target(query)),offset(query,point))
def native_address(+query: Chunks.Query,+point: Point<query>) -> U32:
  U32.from_nat(address(query,point))
def pixel_inside(+query: Chunks.Query,+point: Point<query>) -> {Nat.is_lt(pixel(query,point),pixels(query)) == True{} : Bool}:
  Point{group,+row,+column,gi,+rl,+ru,+xi} = point
  RowMajor.radix(Nat.sub(row,U32.to_nat(Chunks.first(query))),rows(query),column,width(query),Order.sub_lt(row,U32.to_nat(Chunks.first(query)),rows(query),rl,ru),xi)
def offset_inside(+query: Chunks.Query,+point: Point<query>) -> {Nat.is_lt(offset(query,point),Nat.mul(pixels(query),groups(query))) == True{} : Bool}:
  Point{+group,row,column,+gi,rl,ru,xi} = point
  RowMajor.radix(pixel(query,point),pixels(query),group,groups(query),pixel_inside(query,point),gi)
def address_inside(+query: Chunks.Query,+layout: Layout<query>,+point: Point<query>) -> {Nat.is_lt(address(query,point),U32.to_nat(capacity(query,layout))) == True{} : Bool}:
  Layout{+capacity,capacity_ok,+room} = layout
  +target = U32.to_nat(Chunks.target(query))
  +size = Nat.mul(pixels(query),groups(query))
  Bounds.nat_trans(address(query,point),Nat.add(target,size),U32.to_nat(capacity),Order.offset_lt(target,offset(query,point),size,offset_inside(query,point)),room)
def native_address_exact(+query: Chunks.Query,+layout: Layout<query>,+point: Point<query>) -> {U32.to_nat(native_address(query,point)) == address(query,point) : Nat}:
  Layout{+capacity,+capacity_ok,room} = layout
  Bounds.roundtrip(address(query,point),capacity,capacity_ok,Bounds.strict_le(address(query,point),U32.to_nat(capacity),address_inside(query,layout,point)))

# Row r of a phase job is a tile point of the leaf output: the job's unit,
# its line and the position of the row.
def job_point(+seed: Chunks.Reduction,+valid: Chunks.Valid<seed>,+row: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Kernel.Cursor(),Runtime.job(seed))) == True{} : Bool}) -> Point<Chunks.query(Chunks.selection(seed))>:
  +query = Chunks.query(Chunks.selection(seed))
  +first = Chunks.first(query)
  +count = GeometryChunks.grid_count(Chunks.grid(query))
  +channel_job = Channels.channel_job(seed,valid)
  +physical_row = Jobs.physical_row(Chunks.work(Chunks.selection(seed)),Chunks.line(seed))
  +physical_column = Columns.physical_column(seed,row)
  +row_span = Chunks.row_in_leaf(seed,valid)
  Point{Channels.unit(channel_job),physical_row,physical_column,Channels.unit_below(channel_job),
    Shape.first({Nat.is_le(U32.to_nat(first),physical_row) == True{} : Bool},{Nat.is_lt(physical_row,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span),
    Shape.second({Nat.is_le(U32.to_nat(first),physical_row) == True{} : Bool},{Nat.is_lt(physical_row,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool},row_span),
    Columns.physical_column_inside(seed,valid,row,row_inside)}

# The address of the words of a row, a column and a tile group.
def coordinate_address(+query: Chunks.Query,group_word: U32,row_word: U32,column_word: U32) -> U32:
  (Chunks.target(query) + ((row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32) * cells(query) + group_word : U32) : U32)
def pixel_word_exact(+query: Chunks.Query,+point: Point<query>,+row_word: U32,+column_word: U32,
  +row_exact: {U32.to_nat(row_word) == local_row(query,point) : Nat},
  +column_exact: {U32.to_nat(column_word) == column(query,point) : Nat}) ->
  {(row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32) == U32.from_nat(pixel(query,point)) : U32}:
  Chunks.Query{+grid,first,target,source,chunk_row,chunk_group,wb} = query
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,s,p,gh,gw} = g
  %column_exact : {(row_word * w + column_word : U32) == U32.from_nat(Nat.add(Nat.mul(local_row(query,point),U32.to_nat(w)),_)) : U32}
  %row_exact : {(row_word * w + column_word : U32) == U32.from_nat(Nat.add(Nat.mul(_,U32.to_nat(w)),U32.to_nat(column_word))) : U32}
  Encoded.native_madd(row_word,w,column_word)
def offset_word_exact(+query: Chunks.Query,+point: Point<query>,+group_word: U32,+row_word: U32,+column_word: U32,
  +group_exact: {U32.to_nat(group_word) == group(query,point) : Nat},
  +row_exact: {U32.to_nat(row_word) == local_row(query,point) : Nat},
  +column_exact: {U32.to_nat(column_word) == column(query,point) : Nat}) ->
  {((row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32) * cells(query) + group_word : U32) == U32.from_nat(offset(query,point)) : U32}:
  +pixel_word = (row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32)
  %Equal.sym(U32,pixel_word,U32.from_nat(pixel(query,point)),pixel_word_exact(query,point,row_word,column_word,row_exact,column_exact)) :
    {(_ * cells(query) + group_word : U32) == U32.from_nat(offset(query,point)) : U32}
  %Equal.sym(U32,cells(query),U32.from_nat(groups(query)),Encoded.encoding(cells(query),groups(query),cells_exact(query))) :
    {(U32.from_nat(pixel(query,point)) * _ + group_word : U32) == U32.from_nat(offset(query,point)) : U32}
  %Equal.sym(U32,group_word,U32.from_nat(group(query,point)),Encoded.encoding(group_word,group(query,point),group_exact)) :
    {(U32.from_nat(pixel(query,point)) * U32.from_nat(groups(query)) + _ : U32) == U32.from_nat(offset(query,point)) : U32}
  Encoded.encoded_madd(pixel(query,point),groups(query),group(query,point))
def coordinate_address_matches(+query: Chunks.Query,+point: Point<query>,+group_word: U32,+row_word: U32,+column_word: U32,
  +group_exact: {U32.to_nat(group_word) == group(query,point) : Nat},
  +row_exact: {U32.to_nat(row_word) == local_row(query,point) : Nat},
  +column_exact: {U32.to_nat(column_word) == column(query,point) : Nat}) ->
  {coordinate_address(query,group_word,row_word,column_word) == native_address(query,point) : U32}:
  +target = Chunks.target(query)
  +inner = ((row_word * Domain.width_word(Chunks.geometry(query)) + column_word : U32) * cells(query) + group_word : U32)
  %Equal.sym(U32,target,U32.from_nat(U32.to_nat(target)),Encoded.encoding(target,U32.to_nat(target),{==})) : {(_ + inner : U32) == native_address(query,point) : U32}
  %Equal.sym(U32,inner,U32.from_nat(offset(query,point)),offset_word_exact(query,point,group_word,row_word,column_word,group_exact,row_exact,column_exact)) :
    {(U32.from_nat(U32.to_nat(target)) + _ : U32) == native_address(query,point) : U32}
  Words.native_add(U32.to_nat(target),offset(query,point))
def address_congruence(+query: Chunks.Query,+left: Point<query>,+right: Point<query>,
  +group_equal: {group(query,left) == group(query,right) : Nat},+row_equal: {row(query,left) == row(query,right) : Nat},+column_equal: {column(query,left) == column(query,right) : Nat}) -> {address(query,left) == address(query,right) : Nat}:
  +target = U32.to_nat(Chunks.target(query))
  +count = groups(query)
  +width = width(query)
  +first = U32.to_nat(Chunks.first(query))
  %Equal.sym(Nat,group(query,left),group(query,right),group_equal) :
    {Nat.add(target,Nat.add(Nat.mul(pixel(query,left),count),_)) == address(query,right) : Nat}
  %Equal.sym(Nat,row(query,left),row(query,right),row_equal) :
    {Nat.add(target,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.sub(_,first),width),column(query,left)),count),group(query,right))) == address(query,right) : Nat}
  %Equal.sym(Nat,column(query,left),column(query,right),column_equal) :
    {Nat.add(target,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.sub(row(query,right),first),width),_),count),group(query,right))) == address(query,right) : Nat}
  {==}

# Chunk positions and the packed weight base do not change the output region.
def scheduled(+query: Chunks.Query,row: Nat,group: Nat,weight_base: U32) -> Chunks.Query:
  Chunks.Query{Chunks.grid(query),Chunks.first(query),Chunks.target(query),Chunks.source(query),row,group,weight_base}
# The output region of a query: its points and their addresses do not depend
# on the chunk or the packing bases.
def region(+query: Chunks.Query) -> Chunks.Query:
  scheduled(query,0n,0n,0)
def region_point(+query: Chunks.Query,point: Point<query>) -> Point<region(query)>:
  Point{group,row,column,gi,rl,ru,xi} = point
  Point{group,row,column,gi,rl,ru,xi}
def region_layout(+query: Chunks.Query,+layout: Layout<query>) -> Layout<region(query)>:
  Layout{capacity(query,layout),capacity_ok(query,layout),room(query,layout)}
